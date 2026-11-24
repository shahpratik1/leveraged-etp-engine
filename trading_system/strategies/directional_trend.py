"""Directional Multi-Timeframe Trend Strategy.

Implements Directional Movement Index (+DI vs -DI) polarity gating combined with
a multi-timeframe Daily 20/50 EMA slope hierarchy and dual-track short execution.
Eliminates counter-trend knife-catching and trades inverse short ETPs exclusively
on broad market indices during verified bear impulses.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from trading_system.config import settings
from trading_system.config_schema import evaluate_screener_metric_gate
from trading_system.strategies.base import BaseStrategy, StrategySignal, read_config_value, safe_float as _safe_float
from trading_system.strategies.registry import StrategyRegistry

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


@StrategyRegistry.register("directional_trend")
class DirectionalTrendStrategy(BaseStrategy):
    """Directional trend strategy with DMI polarity gating and dual-track shorting."""

    def __init__(
        self,
        name: str = "directional_trend",
        min_adx_trend: Optional[float] = None,
        max_catalyst_risk: Optional[float] = None,
        broad_indices: Optional[Set[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self._explicit_min_adx_trend = min_adx_trend is not None
        self._explicit_max_catalyst_risk = max_catalyst_risk is not None
        self._explicit_broad_indices = broad_indices is not None
        self.min_adx_trend = (
            float(min_adx_trend)
            if min_adx_trend is not None
            else float(getattr(settings, "MIN_ADX_TREND", 18.0))
        )
        self.max_catalyst_risk = (
            float(max_catalyst_risk)
            if max_catalyst_risk is not None
            else float(getattr(settings, "SCREENER_MAX_CATALYST_RISK", 0.25))
        )
        self.broad_indices = set(broad_indices) if broad_indices is not None else set(DEFAULT_BROAD_INDICES)

    def _resolve_params(
        self, config_snapshot: Optional[ConfigSnapshot] = None
    ) -> Dict[str, Any]:
        """Resolve strategy thresholds from a ConfigSnapshot or runtime settings."""
        if config_snapshot is not None:
            raw_indices = read_config_value(config_snapshot, "BROAD_INDICES", None)
            broad_set = set(raw_indices) if raw_indices is not None else self.broad_indices
            return {
                "config_epoch_id": read_config_value(config_snapshot, "config_epoch_id", None),
                "min_adx_trend": float(read_config_value(config_snapshot, "MIN_ADX_TREND", 18.0)),
                "max_catalyst_risk": float(read_config_value(config_snapshot, "SCREENER_MAX_CATALYST_RISK", 0.25)),
                "exceptional_adx": float(read_config_value(config_snapshot, "STRATEGY_EXCEPTIONAL_ADX", 25.0)),
                "min_ema20_slope_pct": float(read_config_value(config_snapshot, "MIN_EMA20_SLOPE_PCT", 0.0)),
                "earnings_blackout_days": float(read_config_value(config_snapshot, "ENTRY_EARNINGS_BLACKOUT_DAYS", 2.0)),
                "base_conviction": float(read_config_value(config_snapshot, "STRATEGY_BASE_CONVICTION", 0.70)),
                "short_base_conviction": float(read_config_value(config_snapshot, "STRATEGY_SHORT_BASE_CONVICTION", 0.75)),
                "macro_tailwind_bonus": float(read_config_value(config_snapshot, "STRATEGY_MACRO_TAILWIND_BONUS", 0.10)),
                "max_conviction": float(read_config_value(config_snapshot, "DIRECTIONAL_MAX_CONVICTION", 0.95)),
                "broad_indices": broad_set,
            }

        return {
            "config_epoch_id": None,
            "min_adx_trend": (
                self.min_adx_trend
                if self._explicit_min_adx_trend
                else float(getattr(settings, "MIN_ADX_TREND", self.min_adx_trend))
            ),
            "max_catalyst_risk": (
                self.max_catalyst_risk
                if self._explicit_max_catalyst_risk
                else float(getattr(settings, "SCREENER_MAX_CATALYST_RISK", self.max_catalyst_risk))
            ),
            "exceptional_adx": float(getattr(settings, "STRATEGY_EXCEPTIONAL_ADX", 25.0)),
            "min_ema20_slope_pct": float(getattr(settings, "MIN_EMA20_SLOPE_PCT", 0.0)),
            "earnings_blackout_days": float(getattr(settings, "ENTRY_EARNINGS_BLACKOUT_DAYS", 2.0)),
            "base_conviction": float(getattr(settings, "STRATEGY_BASE_CONVICTION", 0.70)),
            "short_base_conviction": float(getattr(settings, "STRATEGY_SHORT_BASE_CONVICTION", 0.75)),
            "macro_tailwind_bonus": float(getattr(settings, "STRATEGY_MACRO_TAILWIND_BONUS", 0.10)),
            "max_conviction": float(getattr(settings, "DIRECTIONAL_MAX_CONVICTION", 0.95)),
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

    @staticmethod
    def _evaluate_dynamic_gates(
        metrics: Dict[str, Any], config_snapshot: Optional[ConfigSnapshot]
    ) -> List[Dict[str, Any]]:
        """Evaluate wired dynamic screener_metric_gate rules against candidate metrics."""
        rejections: List[Dict[str, Any]] = []
        if config_snapshot is None:
            return rejections

        specs = read_config_value(config_snapshot, "dynamic_parameter_specs", ()) or ()
        dyn_values = read_config_value(config_snapshot, "dynamic_parameters", {}) or {}
        for spec in specs:
            gate = getattr(spec, "screener_metric_gate", None)
            if gate is None:
                continue
            active_val = (
                dyn_values.get(spec.key, getattr(spec, "default", None))
                if hasattr(dyn_values, "get")
                else getattr(spec, "default", None)
            )
            passed, observed, threshold, reason = evaluate_screener_metric_gate(
                gate, metrics, active_value=active_val
            )
            if not passed:
                rejections.append(
                    {
                        "gate_id": f"dynamic_gate:{spec.key}",
                        "param_keys": [spec.key],
                        "observed": observed,
                        "threshold": threshold,
                        "label": getattr(spec, "label", spec.key),
                        "reason": reason,
                    }
                )
        return rejections

    def evaluate_candidate(
        self,
        ticker: str,
        metrics: Dict[str, Any],
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> StrategySignal:
        """Evaluate an asset using multi-timeframe DMI polarity and EMA slope."""
        params = self._resolve_params(config_snapshot)
        epoch_id: Optional[str] = params["config_epoch_id"]
        min_adx: float = params["min_adx_trend"]
        exceptional_adx: float = params["exceptional_adx"]
        min_slope_pct: float = params["min_ema20_slope_pct"]
        blackout_days: float = params["earnings_blackout_days"]
        base_conv: float = params["base_conviction"]
        short_base_conv: float = params["short_base_conviction"]
        tailwind_bonus: float = params["macro_tailwind_bonus"]
        max_conv: float = params["max_conviction"]
        broad_indices: Set[str] = params["broad_indices"]

        close = _safe_float(metrics.get("close"), 0.0) or 0.0
        sma200 = _safe_float(metrics.get("sma200"), 0.0) or 0.0
        ema20_raw = metrics.get("ema20_daily") if metrics.get("ema20_daily") is not None else metrics.get("ema20")
        ema50_raw = metrics.get("ema50_daily") if metrics.get("ema50_daily") is not None else metrics.get("ema50")
        ema20 = _safe_float(ema20_raw, 0.0) or 0.0
        ema50 = _safe_float(ema50_raw, 0.0) or 0.0
        adx = _safe_float(metrics.get("adx"), 0.0) or 0.0
        plus_di = _safe_float(metrics.get("plus_di"), 25.0 if close > ema20 else 15.0) or 0.0
        minus_di = _safe_float(metrics.get("minus_di"), 15.0 if close > ema20 else 25.0) or 0.0

        slope_raw = _safe_float(metrics.get("ema20_slope_5d"), None)
        if slope_raw is not None:
            slope_20 = slope_raw
        else:
            slope_20 = 0.01 if (close > ema20 or (ema50 > 0 and ema20 > ema50)) else -0.01

        macd = _safe_float(metrics.get("macd"), None)
        macd_sig = _safe_float(metrics.get("macd_signal"), None)
        has_macd = macd is not None and macd_sig is not None
        macd_conflicts_bull = has_macd and macd < macd_sig  # type: ignore[operator]
        macd_conflicts_bear = has_macd and macd > macd_sig  # type: ignore[operator]

        # Catalyst risk assessment (e.g. impending earnings)
        earnings_val = _safe_float(metrics.get("earnings_in_days"), None)
        is_earnings_blackout = (
            earnings_val is not None and 0.0 <= earnings_val <= blackout_days
        )
        catalyst_risk = 0.90 if is_earnings_blackout else 0.05

        bull_evidence: List[str] = []
        bear_evidence: List[str] = []
        gate_rejections: List[Dict[str, Any]] = self._evaluate_dynamic_gates(metrics, config_snapshot)

        is_macro_bull = close > sma200 if sma200 > 0 else True
        is_bull_slope = slope_20 > min_slope_pct
        is_intermediate_bull = (
            (close > ema50 and ema20 >= ema50 and is_bull_slope)
            if ema50 > 0.0
            else (close > ema20 and is_bull_slope)
        )
        is_bear_slope = (
            slope_20 < -min_slope_pct
            if min_slope_pct != 0.0
            else (close < ema20 or slope_20 < 0.0)
        )
        is_dmi_bull = plus_di > minus_di and adx >= min_adx
        is_dmi_bear = minus_di > plus_di and adx >= min_adx

        if gate_rejections:
            rejected_labels = ", ".join(r["label"] for r in gate_rejections)
            return StrategySignal(
                symbol=ticker,
                direction="NEUTRAL",
                conviction=0.0,
                setup_quality="unfavorable",
                strategy_name=self.name,
                rationale=f"Rejected by dynamic screener metric gate(s): {rejected_labels}.",
                metrics=metrics,
                bull_evidence=bull_evidence,
                bear_evidence=bear_evidence,
                catalyst_risk=catalyst_risk,
                timeframe="1d",
                gate_rejections=gate_rejections,
                config_epoch_id=epoch_id,
            )

        # ── 1. Bullish Impulse Setup ─────────────────────────────
        if is_intermediate_bull and is_dmi_bull and not is_dmi_bear and not macd_conflicts_bull:
            conviction = round(
                min(max_conv, base_conv + (adx / 100.0) + (tailwind_bonus if is_macro_bull else 0.0)),
                2,
            )
            setup_quality = "exceptional" if adx >= exceptional_adx else "favorable"
            bull_evidence.append(
                f"Directional Polarity Bullish: +DI ({plus_di:.1f}) > -DI ({minus_di:.1f}) with ADX={adx:.1f} >= {min_adx:.1f}."
            )
            bull_evidence.append(
                f"Multi-Timeframe Trend Aligned: Close ({close:.2f}) > 50 EMA ({ema50:.2f}) with 20 EMA upward slope ({slope_20:+.2f}%)."
            )
            if is_macro_bull:
                bull_evidence.append(f"Macro tailwind: Price ({close:.2f}) > 200 SMA ({sma200:.2f}).")

            return StrategySignal(
                symbol=ticker,
                direction="LONG",
                conviction=conviction,
                setup_quality=setup_quality,
                strategy_name=self.name,
                rationale=(
                    f"Directional Trend: Confirmed bullish momentum on {ticker} (+DI={plus_di:.1f} > -DI={minus_di:.1f}, "
                    f"ADX={adx:.1f}, 20 EMA slope={slope_20:+.2f}%)."
                ),
                metrics=metrics,
                bull_evidence=bull_evidence,
                bear_evidence=bear_evidence,
                catalyst_risk=catalyst_risk,
                timeframe="1d",
                gate_rejections=[],
                config_epoch_id=epoch_id,
            )

        # ── 2. Bearish Regime & Dual-Track Short Execution ─────────
        rsi3_val = _safe_float(metrics.get("rsi3"), None)
        rsi14_val = _safe_float(metrics.get("rsi14"), None)
        is_oversold = (rsi3_val is not None and rsi3_val < 30.0) or (rsi14_val is not None and rsi14_val < 35.0)
        is_intermediate_bear = (
            (close < ema50 and ema20 <= ema50 and is_bear_slope)
            if ema50 > 0.0
            else (close < ema20 and is_bear_slope)
        )

        if is_dmi_bear and is_intermediate_bear and not is_oversold and not macd_conflicts_bear:
            bear_evidence.append(
                f"Directional Polarity Bearish: -DI ({minus_di:.1f}) > +DI ({plus_di:.1f}) with ADX={adx:.1f} >= {min_adx:.1f}."
            )
            bear_evidence.append(
                f"Trend Breakdown Aligned: Close ({close:.2f}) < 50 EMA ({ema50:.2f}) with downward 20 EMA slope ({slope_20:+.2f}%)."
            )

            # Dual-track check: Index underlying vs Single Equity
            if self.is_broad_index(ticker, broad_indices=broad_indices):
                conviction = round(min(max_conv, short_base_conv + (adx / 100.0)), 2)
                setup_quality = "exceptional" if adx >= exceptional_adx else "favorable"
                bear_evidence.append(
                    f"Dual-track rule: Benchmark index {ticker} approved for inverse 3x Short ETP execution."
                )

                return StrategySignal(
                    symbol=ticker,
                    direction="SHORT",
                    conviction=conviction,
                    setup_quality=setup_quality,
                    strategy_name=self.name,
                    rationale=(
                        f"Directional Trend SHORT: Bearish breakdown on benchmark index {ticker} "
                        f"(-DI={minus_di:.1f} > +DI={plus_di:.1f}, ADX={adx:.1f}, 20 EMA slope={slope_20:+.2f}%)."
                    ),
                    metrics=metrics,
                    bull_evidence=bull_evidence,
                    bear_evidence=bear_evidence,
                    catalyst_risk=catalyst_risk,
                    timeframe="1d",
                    gate_rejections=[],
                    config_epoch_id=epoch_id,
                )

            # Single-stock cash defense
            bear_evidence.append(
                f"Dual-track cash defense: Short ETP skipped for single stock {ticker} to protect against short squeeze risk."
            )
            gate_rejections.append(
                {
                    "gate_id": "dual_track_cash_defense",
                    "param_keys": [],
                    "observed": ticker,
                    "threshold": "broad_index",
                    "label": "Dual-Track Cash Defense",
                }
            )
            return StrategySignal(
                symbol=ticker,
                direction="NEUTRAL",
                conviction=0.0,
                setup_quality="unfavorable",
                strategy_name=self.name,
                rationale=(
                    f"Cash Defense on {ticker}: Bear regime active (-DI={minus_di:.1f} > +DI={plus_di:.1f}, ADX={adx:.1f}). "
                    f"Long dip-buying locked out; equity shorting suppressed."
                ),
                metrics=metrics,
                bull_evidence=bull_evidence,
                bear_evidence=bear_evidence,
                catalyst_risk=catalyst_risk,
                timeframe="1d",
                gate_rejections=gate_rejections,
                config_epoch_id=epoch_id,
            )

        # ── 3. Neutral / Counter-Trend Filter ─────────────────────
        neutral_reason = "No clean directional trend impulse"
        if adx < min_adx:
            gate_rejections.append(
                {
                    "gate_id": "adx_trend_gate",
                    "param_keys": ["MIN_ADX_TREND"],
                    "observed": round(adx, 4),
                    "threshold": min_adx,
                    "label": "Minimum ADX Trend",
                }
            )
            neutral_reason = f"ADX ({adx:.1f}) below MIN_ADX_TREND ({min_adx:.1f})."
        elif is_dmi_bear:
            neutral_reason = f"Bear regime active (-DI={minus_di:.1f} > +DI={plus_di:.1f}). Long dip-buying disarmed."
            if macd_conflicts_bear:
                gate_rejections.append(
                    {
                        "gate_id": "macd_bear_conflict",
                        "param_keys": [],
                        "observed": round(float(macd) - float(macd_sig), 4),
                        "threshold": 0.0,
                        "label": "MACD Histogram Conflict",
                    }
                )
            elif not is_bear_slope:
                gate_rejections.append(
                    {
                        "gate_id": "ema_slope_gate",
                        "param_keys": ["MIN_EMA20_SLOPE_PCT"],
                        "observed": round(slope_20, 4),
                        "threshold": -min_slope_pct,
                        "label": "Min 20-EMA 5d Slope",
                    }
                )
        else:
            if not is_bull_slope:
                neutral_reason = (
                    f"Intermediate 20 EMA slope ({slope_20:+.2f}%) is not above {min_slope_pct:+.2f}%. "
                    "Dip-buying locked out to prevent knife-catching."
                )
                gate_rejections.append(
                    {
                        "gate_id": "ema_slope_gate",
                        "param_keys": ["MIN_EMA20_SLOPE_PCT"],
                        "observed": round(slope_20, 4),
                        "threshold": min_slope_pct,
                        "label": "Min 20-EMA 5d Slope",
                    }
                )
            elif (ema50 > 0.0 and (close <= ema50 or ema20 < ema50)) or (ema50 <= 0.0 and close <= ema20):
                neutral_reason = f"Price ({close:.2f}) below 50 EMA ({ema50:.2f}). Waiting for trend reclaim."
                gate_rejections.append(
                    {
                        "gate_id": "ema_trend_hierarchy",
                        "param_keys": [],
                        "observed": round(close, 4),
                        "threshold": round(ema50 if ema50 > 0 else ema20, 4),
                        "label": "EMA Trend Alignment",
                    }
                )
            elif plus_di > minus_di and macd_conflicts_bull:
                gate_rejections.append(
                    {
                        "gate_id": "macd_bull_conflict",
                        "param_keys": [],
                        "observed": round(float(macd) - float(macd_sig), 4),
                        "threshold": 0.0,
                        "label": "MACD Histogram Conflict",
                    }
                )
                neutral_reason = "MACD conflicts with bullish DMI impulse."
            else:
                gate_rejections.append(
                    {
                        "gate_id": "dmi_polarity_gate",
                        "param_keys": [],
                        "observed": round(plus_di - minus_di, 4),
                        "threshold": 0.0,
                        "label": "DMI Polarity Direction",
                    }
                )

        return StrategySignal(
            symbol=ticker,
            direction="NEUTRAL",
            conviction=0.0,
            setup_quality="unfavorable",
            strategy_name=self.name,
            rationale=neutral_reason,
            metrics=metrics,
            bull_evidence=bull_evidence,
            bear_evidence=bear_evidence,
            catalyst_risk=catalyst_risk,
            timeframe="1d",
            gate_rejections=gate_rejections,
            config_epoch_id=epoch_id,
        )

    def filter_universe(
        self,
        universe_metrics: Dict[str, Dict[str, Any]],
        top_n: int = 5,
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> List[StrategySignal]:
        """Filter universe and return top actionable directional trend signals."""
        params = self._resolve_params(config_snapshot)
        max_catalyst_risk: float = params["max_catalyst_risk"]
        signals: List[StrategySignal] = []

        for ticker, metrics in universe_metrics.items():
            try:
                sig = self.evaluate_candidate(ticker, metrics, config_snapshot=config_snapshot)
                if sig.is_actionable and sig.catalyst_risk <= max_catalyst_risk:
                    signals.append(sig)
            except Exception as exc:
                logger.debug("Directional trend evaluation failed for %s: %s", ticker, exc)

        # Sort by conviction descending, then ADX descending
        def _sort_key(s: StrategySignal) -> Tuple[float, float]:
            adx_val = _safe_float(s.metrics.get("adx"), 0.0) or 0.0
            return (s.conviction, adx_val)

        signals.sort(key=_sort_key, reverse=True)
        return signals[:top_n]
