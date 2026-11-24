"""Machine Learning Alpha Strategy.

Implements an offline-trained LightGBM model executed via ONNX Runtime for
sub-millisecond directional forecasting with volatility drag filtering,
confidence thresholds, and dual-track bear execution.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np

from trading_system.config import settings
from trading_system.strategies.base import (
    BaseStrategy,
    StrategySignal,
    read_config_value,
    safe_float as _safe_float,
)
from trading_system.strategies.registry import StrategyRegistry
from trading_system.ml.onnx_runner import OnnxModelRunner

if TYPE_CHECKING:
    from trading_system.config_bus import ConfigSnapshot

logger = logging.getLogger(__name__)

# Broad index benchmarks eligible for inverse 3x Short ETP execution
DEFAULT_BROAD_INDICES: Set[str] = {
    "QQQ",
    "SPY",
    "SMH",
    "^NDX",
    "^GSPC",
    "^STOXX50E",
    "3USL",
    "QQQ3",
}


@StrategyRegistry.register("ml_alpha")
class MLAlphaStrategy(BaseStrategy):
    """Machine learning alpha strategy using ONNX model runner."""

    def __init__(
        self,
        name: str = "ml_alpha",
        model_path: Optional[str] = None,
        min_confidence: Optional[float] = None,
        min_edge_margin: Optional[float] = None,
        max_vol_drag_cutoff: Optional[float] = None,
        cooldown_bars: Optional[int] = None,
        broad_indices: Optional[Set[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.model_path = model_path or getattr(settings, "ML_MODEL_PATH", "models/ml_alpha_v1.onnx")
        self.min_confidence = (
            float(min_confidence)
            if min_confidence is not None
            else float(getattr(settings, "ML_MIN_CONFIDENCE", 0.58))
        )
        self.min_edge_margin = (
            float(min_edge_margin)
            if min_edge_margin is not None
            else float(getattr(settings, "ML_MIN_EDGE_MARGIN", 0.15))
        )
        self.max_vol_drag_cutoff = (
            float(max_vol_drag_cutoff)
            if max_vol_drag_cutoff is not None
            else float(getattr(settings, "ML_MAX_VOL_DRAG_CUTOFF", 0.05))
        )
        self.cooldown_bars = (
            int(cooldown_bars)
            if cooldown_bars is not None
            else int(getattr(settings, "ML_COOLDOWN_BARS", 3))
        )
        self.broad_indices = set(broad_indices) if broad_indices is not None else set(DEFAULT_BROAD_INDICES)

        self._runners: Dict[str, OnnxModelRunner] = {}
        self._last_signal_bar: Dict[str, int] = {}

    def get_runner(self, model_path: str) -> OnnxModelRunner:
        """Get or initialize cached OnnxModelRunner for given model path."""
        if model_path not in self._runners:
            self._runners[model_path] = OnnxModelRunner(model_path=model_path)
        return self._runners[model_path]

    def _resolve_params(
        self, config_snapshot: Optional[ConfigSnapshot] = None
    ) -> Dict[str, Any]:
        """Resolve strategy parameters dynamically from config snapshot or settings."""
        if config_snapshot is not None:
            raw_indices = read_config_value(config_snapshot, "BROAD_INDICES", None)
            broad_set = set(raw_indices) if raw_indices is not None else self.broad_indices
            return {
                "config_epoch_id": read_config_value(config_snapshot, "config_epoch_id", None),
                "model_path": str(
                    read_config_value(
                        config_snapshot,
                        "ML_MODEL_PATH",
                        read_config_value(config_snapshot, "model_path", self.model_path),
                    )
                ),
                "min_confidence": float(
                    read_config_value(
                        config_snapshot,
                        "ML_MIN_CONFIDENCE",
                        read_config_value(config_snapshot, "min_confidence", self.min_confidence),
                    )
                ),
                "min_edge_margin": float(
                    read_config_value(
                        config_snapshot,
                        "ML_MIN_EDGE_MARGIN",
                        read_config_value(config_snapshot, "min_edge_margin", self.min_edge_margin),
                    )
                ),
                "max_vol_drag_cutoff": float(
                    read_config_value(
                        config_snapshot,
                        "ML_MAX_VOL_DRAG_CUTOFF",
                        read_config_value(config_snapshot, "max_vol_drag_cutoff", self.max_vol_drag_cutoff),
                    )
                ),
                "cooldown_bars": int(
                    read_config_value(
                        config_snapshot,
                        "ML_COOLDOWN_BARS",
                        read_config_value(config_snapshot, "cooldown_bars", self.cooldown_bars),
                    )
                ),
                "broad_indices": broad_set,
            }

        return {
            "config_epoch_id": None,
            "model_path": getattr(settings, "ML_MODEL_PATH", self.model_path),
            "min_confidence": float(getattr(settings, "ML_MIN_CONFIDENCE", self.min_confidence)),
            "min_edge_margin": float(getattr(settings, "ML_MIN_EDGE_MARGIN", self.min_edge_margin)),
            "max_vol_drag_cutoff": float(getattr(settings, "ML_MAX_VOL_DRAG_CUTOFF", self.max_vol_drag_cutoff)),
            "cooldown_bars": int(getattr(settings, "ML_COOLDOWN_BARS", self.cooldown_bars)),
            "broad_indices": self.broad_indices,
        }

    def is_broad_index(
        self, ticker: str, broad_indices: Optional[Set[str]] = None
    ) -> bool:
        """Return True if ticker is a broad benchmark index eligible for short ETPs."""
        index_set = broad_indices if broad_indices is not None else self.broad_indices
        sym_clean = ticker.upper().replace("^", "")
        return (
            ticker.upper() in index_set
            or sym_clean in index_set
            or any(idx in sym_clean for idx in ("QQQ", "SPY", "SMH", "STOXX"))
        )

    def _extract_features(
        self, metrics: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], float]:
        """Extract or compute 14-feature vector from candidate metrics or raw bars."""
        # 1. Precomputed features array directly in metrics
        if isinstance(metrics, dict) and "features" in metrics:
            raw_feat = metrics["features"]
            feat_arr = np.asarray(raw_feat, dtype=np.float32)
            if feat_arr.ndim == 1:
                feat_arr = feat_arr.reshape(1, -1)
            vol_drag = (
                float(metrics.get("vol_drag_proxy", feat_arr[0, 0]))
                if feat_arr.size > 0
                else float(metrics.get("vol_drag_proxy", 0.0))
            )
            return feat_arr, vol_drag

        # 2. Extract from bars / candles / history list
        bars = None
        if isinstance(metrics, dict):
            bars = metrics.get("bars") or metrics.get("candles") or metrics.get("history")
        elif isinstance(metrics, (list, tuple)):
            bars = metrics

        if bars is not None:
            try:
                from trading_system.ml.features import extract_features

                feat_arr = extract_features(bars)
                vol_drag = float(feat_arr[0, 0]) if feat_arr.size > 0 else 0.0
                return feat_arr, vol_drag
            except Exception as err:
                logger.warning("Feature extraction from bars failed: %s", err)
                return None, 0.0

        # 3. Single bar dict with OHLCV
        if isinstance(metrics, dict) and "close" in metrics:
            try:
                from trading_system.ml.features import extract_features

                feat_arr = extract_features([metrics])
                vol_drag = (
                    float(feat_arr[0, 0])
                    if feat_arr.size > 0
                    else float(metrics.get("vol_drag_proxy", 0.0))
                )
                return feat_arr, vol_drag
            except Exception as err:
                logger.warning("Feature extraction from single bar failed: %s", err)

        return None, float(metrics.get("vol_drag_proxy", 0.0)) if isinstance(metrics, dict) else 0.0

    def evaluate_candidate(
        self,
        ticker: str,
        metrics: Dict[str, Any],
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> StrategySignal:
        """Evaluate asset using ONNX model inference with risk guards."""
        params = self._resolve_params(config_snapshot)
        epoch_id: Optional[str] = params["config_epoch_id"]
        model_path: str = params["model_path"]
        min_confidence: float = params["min_confidence"]
        min_edge_margin: float = params["min_edge_margin"]
        max_vol_drag_cutoff: float = params["max_vol_drag_cutoff"]
        cooldown_bars: int = params["cooldown_bars"]
        broad_indices: Set[str] = params["broad_indices"]

        runner = self.get_runner(model_path)
        gate_rejections: List[Dict[str, Any]] = []
        bull_evidence: List[str] = []
        bear_evidence: List[str] = []

        feat_arr, vol_drag_proxy = self._extract_features(metrics)

        # Baseline metrics dictionary to update
        res_metrics: Dict[str, Any] = dict(metrics) if isinstance(metrics, dict) else {}
        res_metrics["vol_drag_proxy"] = float(vol_drag_proxy)
        res_metrics["model_version"] = str(runner.model_version)

        # Volatility drag guard
        if vol_drag_proxy > max_vol_drag_cutoff:
            logger.warning(
                "MLAlpha vol drag warning on %s: vol_drag_proxy %.4f > cutoff %.4f",
                ticker,
                vol_drag_proxy,
                max_vol_drag_cutoff,
            )
            res_metrics["p_cash"] = 1.0
            res_metrics["p_long"] = 0.0
            res_metrics["p_short"] = 0.0
            res_metrics["inference_ms"] = 0.0
            gate_rejections.append({
                "gate_id": "vol_drag_cutoff",
                "param_keys": ["ML_MAX_VOL_DRAG_CUTOFF"],
                "observed": vol_drag_proxy,
                "threshold": max_vol_drag_cutoff,
                "label": "Volatility Drag Cutoff",
                "reason": f"vol_drag_proxy {vol_drag_proxy:.4f} > cutoff {max_vol_drag_cutoff:.4f}",
            })
            return StrategySignal(
                symbol=ticker,
                direction="NEUTRAL",
                conviction=0.0,
                setup_quality="unfavorable",
                strategy_name=self.name,
                rationale=(
                    f"MLAlpha Cash Defense on {ticker}: Volatility drag proxy ({vol_drag_proxy:.4f}) "
                    f"exceeds safety cutoff ({max_vol_drag_cutoff:.4f})."
                ),
                metrics=res_metrics,
                bull_evidence=bull_evidence,
                bear_evidence=bear_evidence,
                catalyst_risk=float(metrics.get("catalyst_risk", 0.0)) if isinstance(metrics, dict) else 0.0,
                timeframe=str(metrics.get("timeframe", "1d")) if isinstance(metrics, dict) else "1d",
                gate_rejections=gate_rejections,
                config_epoch_id=epoch_id,
            )

        # Model inference
        probs = runner.predict_proba(feat_arr if feat_arr is not None else np.zeros((1, 14), dtype=np.float32))
        p_cash = float(probs[0, 0])
        p_long = float(probs[0, 1])
        p_short = float(probs[0, 2])
        inference_ms = float(runner.last_inference_ms)

        res_metrics["p_cash"] = p_cash
        res_metrics["p_long"] = p_long
        res_metrics["p_short"] = p_short
        res_metrics["inference_ms"] = inference_ms

        p_winner = max(p_long, p_short)
        edge_margin = p_winner - p_cash

        # Decision threshold evaluation
        if p_winner >= min_confidence and edge_margin >= min_edge_margin:
            # Check cooldown if bar_index is available
            bar_idx = metrics.get("bar_index") if isinstance(metrics, dict) else None
            if bar_idx is not None and cooldown_bars > 0:
                last_bar = self._last_signal_bar.get(ticker)
                if last_bar is not None and (bar_idx - last_bar) < cooldown_bars:
                    gate_rejections.append({
                        "gate_id": "cooldown_bars_gate",
                        "param_keys": ["ML_COOLDOWN_BARS"],
                        "observed": bar_idx - last_bar,
                        "threshold": cooldown_bars,
                        "label": "ML Alpha Cooldown",
                        "reason": f"Bars since last entry ({bar_idx - last_bar}) < cooldown ({cooldown_bars})",
                    })
                    return StrategySignal(
                        symbol=ticker,
                        direction="NEUTRAL",
                        conviction=0.0,
                        setup_quality="unfavorable",
                        strategy_name=self.name,
                        rationale=f"MLAlpha Cooldown active on {ticker} ({bar_idx - last_bar} < {cooldown_bars} bars).",
                        metrics=res_metrics,
                        bull_evidence=bull_evidence,
                        bear_evidence=bear_evidence,
                        catalyst_risk=float(metrics.get("catalyst_risk", 0.0)) if isinstance(metrics, dict) else 0.0,
                        timeframe=str(metrics.get("timeframe", "1d")) if isinstance(metrics, dict) else "1d",
                        gate_rejections=gate_rejections,
                        config_epoch_id=epoch_id,
                    )

            if p_long >= p_short:
                # LONG Signal
                setup_quality = "exceptional" if p_long >= 0.70 else "favorable"
                bull_evidence.append(
                    f"MLAlpha P(LONG)={p_long:.3f} >= {min_confidence:.2f}, edge over CASH={edge_margin:+.3f} >= {min_edge_margin:.2f}."
                )
                if bar_idx is not None:
                    self._last_signal_bar[ticker] = bar_idx

                return StrategySignal(
                    symbol=ticker,
                    direction="LONG",
                    conviction=round(p_long, 2),
                    setup_quality=setup_quality,
                    strategy_name=self.name,
                    rationale=(
                        f"MLAlpha LONG: High confidence bull forecast on {ticker} "
                        f"(P(LONG)={p_long:.3f}, edge={edge_margin:+.3f})."
                    ),
                    metrics=res_metrics,
                    bull_evidence=bull_evidence,
                    bear_evidence=bear_evidence,
                    catalyst_risk=float(metrics.get("catalyst_risk", 0.0)) if isinstance(metrics, dict) else 0.0,
                    timeframe=str(metrics.get("timeframe", "1d")) if isinstance(metrics, dict) else "1d",
                    gate_rejections=gate_rejections,
                    config_epoch_id=epoch_id,
                )
            else:
                # SHORT Signal - Dual-Track Bear Shield
                if self.is_broad_index(ticker, broad_indices=broad_indices):
                    setup_quality = "exceptional" if p_short >= 0.70 else "favorable"
                    bear_evidence.append(
                        f"MLAlpha P(SHORT)={p_short:.3f} >= {min_confidence:.2f}, edge over CASH={edge_margin:+.3f} >= {min_edge_margin:.2f}."
                    )
                    bear_evidence.append(
                        f"Dual-track rule: Benchmark index {ticker} approved for inverse ETP execution."
                    )
                    if bar_idx is not None:
                        self._last_signal_bar[ticker] = bar_idx

                    return StrategySignal(
                        symbol=ticker,
                        direction="SHORT",
                        conviction=round(p_short, 2),
                        setup_quality=setup_quality,
                        strategy_name=self.name,
                        rationale=(
                            f"MLAlpha SHORT: Bearish breakdown forecast on benchmark {ticker} "
                            f"(P(SHORT)={p_short:.3f}, edge={edge_margin:+.3f})."
                        ),
                        metrics=res_metrics,
                        bull_evidence=bull_evidence,
                        bear_evidence=bear_evidence,
                        catalyst_risk=float(metrics.get("catalyst_risk", 0.0)) if isinstance(metrics, dict) else 0.0,
                        timeframe=str(metrics.get("timeframe", "1d")) if isinstance(metrics, dict) else "1d",
                        gate_rejections=gate_rejections,
                        config_epoch_id=epoch_id,
                    )
                else:
                    # Single-stock short suppression -> route to NEUTRAL (CASH_DEFENSE)
                    bear_evidence.append(
                        f"Dual-track cash defense: Short ETP skipped for single stock {ticker} to protect against short squeeze risk."
                    )
                    gate_rejections.append({
                        "gate_id": "dual_track_cash_defense",
                        "param_keys": ["BROAD_INDICES"],
                        "observed": ticker,
                        "threshold": "broad_index",
                        "label": "Dual-Track Cash Defense",
                        "reason": f"Single stock {ticker} short suppressed: cash defense enforced",
                    })
                    return StrategySignal(
                        symbol=ticker,
                        direction="NEUTRAL",
                        conviction=0.0,
                        setup_quality="unfavorable",
                        strategy_name=self.name,
                        rationale=(
                            f"Cash Defense on {ticker}: MLAlpha bearish signal suppressed on single equity "
                            f"to prevent short squeeze risk (P(SHORT)={p_short:.3f})."
                        ),
                        metrics=res_metrics,
                        bull_evidence=bull_evidence,
                        bear_evidence=bear_evidence,
                        catalyst_risk=float(metrics.get("catalyst_risk", 0.0)) if isinstance(metrics, dict) else 0.0,
                        timeframe=str(metrics.get("timeframe", "1d")) if isinstance(metrics, dict) else "1d",
                        gate_rejections=gate_rejections,
                        config_epoch_id=epoch_id,
                    )

        # Default Neutral / Cash Defense
        return StrategySignal(
            symbol=ticker,
            direction="NEUTRAL",
            conviction=0.0,
            setup_quality="unfavorable",
            strategy_name=self.name,
            rationale=(
                f"MLAlpha Neutral (CASH_DEFENSE): P(LONG)={p_long:.3f}, P(SHORT)={p_short:.3f}, "
                f"P(CASH)={p_cash:.3f} below confidence/edge threshold."
            ),
            metrics=res_metrics,
            bull_evidence=bull_evidence,
            bear_evidence=bear_evidence,
            catalyst_risk=float(metrics.get("catalyst_risk", 0.0)) if isinstance(metrics, dict) else 0.0,
            timeframe=str(metrics.get("timeframe", "1d")) if isinstance(metrics, dict) else "1d",
            gate_rejections=gate_rejections,
            config_epoch_id=epoch_id,
        )

    def filter_universe(
        self,
        universe_metrics: Dict[str, Dict[str, Any]],
        top_n: int = 5,
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> List[StrategySignal]:
        """Screen and rank universe metrics, returning top-ranked actionable signals."""
        signals: List[StrategySignal] = []

        for ticker, metrics in universe_metrics.items():
            try:
                sig = self.evaluate_candidate(ticker, metrics, config_snapshot=config_snapshot)
                if sig.is_actionable:
                    signals.append(sig)
            except Exception as exc:
                logger.debug("MLAlpha evaluation failed for %s: %s", ticker, exc)

        def _sort_key(s: StrategySignal) -> Tuple[float, float]:
            edge = float(s.conviction) - float(s.metrics.get("p_cash", 0.0))
            return (s.conviction, edge)

        signals.sort(key=_sort_key, reverse=True)
        return signals[:top_n]
