"""VWAP Session Trend-Following Strategy.

Implements Volume-Weighted Average Price (VWAP) session trend-following with
dual-track bear execution. Long entries require price breakout and hold above
institutional VWAP with trend alignment. Bearish breakdowns below VWAP route
broad market index benchmarks to 3x Inverse Short ETPs and enforce 100% Cash Defense
on individual equities to eliminate short squeeze risk.
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


@StrategyRegistry.register("vwap")
class VWAPStrategy(BaseStrategy):
    """Volume-Weighted Average Price (VWAP) session trend strategy with dual-track shorting."""

    def __init__(
        self,
        name: str = "vwap",
        min_vwap_dist_pct: Optional[float] = None,
        max_vwap_dist_pct: Optional[float] = None,
        max_catalyst_risk: Optional[float] = None,
        broad_indices: Optional[Set[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self._explicit_min_dist = min_vwap_dist_pct is not None
        self._explicit_max_dist = max_vwap_dist_pct is not None
        self._explicit_max_catalyst = max_catalyst_risk is not None
        self._explicit_broad = broad_indices is not None

        self.min_vwap_dist_pct = (
            float(min_vwap_dist_pct)
            if min_vwap_dist_pct is not None
            else float(getattr(settings, "MIN_VWAP_DIST_PCT", 0.1))
        )
        self.max_vwap_dist_pct = (
            float(max_vwap_dist_pct)
            if max_vwap_dist_pct is not None
            else float(getattr(settings, "MAX_VWAP_DIST_PCT", 3.5))
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
                "min_vwap_dist_pct": float(read_config_value(config_snapshot, "MIN_VWAP_DIST_PCT", 0.1)),
                "max_vwap_dist_pct": float(read_config_value(config_snapshot, "MAX_VWAP_DIST_PCT", 3.5)),
                "max_catalyst_risk": float(read_config_value(config_snapshot, "SCREENER_MAX_CATALYST_RISK", 0.25)),
                "earnings_blackout_days": float(read_config_value(config_snapshot, "ENTRY_EARNINGS_BLACKOUT_DAYS", 2.0)),
                "base_conviction": float(read_config_value(config_snapshot, "STRATEGY_BASE_CONVICTION", 0.70)),
                "short_base_conviction": float(read_config_value(config_snapshot, "STRATEGY_SHORT_BASE_CONVICTION", 0.75)),
                "max_conviction": float(read_config_value(config_snapshot, "DIRECTIONAL_MAX_CONVICTION", 0.95)),
                "broad_indices": broad_set,
            }

        return {
            "config_epoch_id": None,
            "min_adx_trend": float(getattr(settings, "MIN_ADX_TREND", 18.0)),
            "min_vwap_dist_pct": (
                self.min_vwap_dist_pct
                if self._explicit_min_dist
                else float(getattr(settings, "MIN_VWAP_DIST_PCT", self.min_vwap_dist_pct))
            ),
            "max_vwap_dist_pct": (
                self.max_vwap_dist_pct
                if self._explicit_max_dist
                else float(getattr(settings, "MAX_VWAP_DIST_PCT", self.max_vwap_dist_pct))
            ),
            "max_catalyst_risk": (
                self.max_catalyst_risk
                if self._explicit_max_catalyst
                else float(getattr(settings, "SCREENER_MAX_CATALYST_RISK", self.max_catalyst_risk))
            ),
            "earnings_blackout_days": float(getattr(settings, "ENTRY_EARNINGS_BLACKOUT_DAYS", 2.0)),
            "base_conviction": float(getattr(settings, "STRATEGY_BASE_CONVICTION", 0.70)),
            "short_base_conviction": float(getattr(settings, "STRATEGY_SHORT_BASE_CONVICTION", 0.75)),
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
        """Evaluate an asset using session VWAP and dual-track short execution."""
        params = self._resolve_params(config_snapshot)
        epoch_id: Optional[str] = params["config_epoch_id"]
        min_dist: float = params["min_vwap_dist_pct"]
        max_dist: float = params["max_vwap_dist_pct"]
        blackout_days: float = params["earnings_blackout_days"]
        base_conv: float = params["base_conviction"]
        short_base_conv: float = params["short_base_conviction"]
        max_conv: float = params["max_conviction"]
        broad_indices: Set[str] = params["broad_indices"]
        timeframe: str = str(metrics.get("timeframe", "1d")) if isinstance(metrics, dict) else "1d"

        close = _safe_float(metrics.get("close"), 0.0) or 0.0
        vwap = _safe_float(metrics.get("vwap"), close) or close
        vwap_dist_pct = _safe_float(metrics.get("vwap_dist_pct"), 0.0) or 0.0
        if vwap > 0 and vwap_dist_pct == 0.0 and close > 0:
            vwap_dist_pct = ((close - vwap) / vwap) * 100.0

        sma200 = _safe_float(metrics.get("sma200"), 0.0) or 0.0
        ema20_raw = metrics.get("ema20_daily") if metrics.get("ema20_daily") is not None else metrics.get("ema20")
        ema50_raw = metrics.get("ema50_daily") if metrics.get("ema50_daily") is not None else metrics.get("ema50")
        ema20 = _safe_float(ema20_raw, 0.0) or 0.0
        ema50 = _safe_float(ema50_raw, 0.0) or 0.0

        plus_di = _safe_float(metrics.get("plus_di"), 25.0 if close > vwap else 15.0) or 0.0
        minus_di = _safe_float(metrics.get("minus_di"), 15.0 if close > vwap else 25.0) or 0.0
        adx = _safe_float(metrics.get("adx"), 20.0) or 0.0

        # Catalyst risk assessment (e.g. impending earnings)
        earnings_val = _safe_float(metrics.get("earnings_in_days"), None)
        is_earnings_blackout = (
            earnings_val is not None and 0.0 <= earnings_val <= blackout_days
        )
        catalyst_risk = 1.0 if is_earnings_blackout else 0.0

        gate_rejections: List[Dict[str, Any]] = []

        if is_earnings_blackout:
            gate_rejections.append({
                "gate_id": "catalyst_gate:EARNINGS_BLACKOUT",
                "param_keys": ["ENTRY_EARNINGS_BLACKOUT_DAYS"],
                "observed": earnings_val,
                "threshold": blackout_days,
                "label": "Earnings Blackout",
                "reason": f"Earnings in {earnings_val:.1f}d within {blackout_days:.0f}d window",
            })

        min_adx: float = params.get("min_adx_trend", 18.0)
        if adx < min_adx:
            gate_rejections.append({
                "gate_id": "trend_filter:MIN_ADX_TREND",
                "param_keys": ["MIN_ADX_TREND"],
                "observed": adx,
                "threshold": min_adx,
                "label": "Minimum ADX Trend Strength",
                "reason": f"ADX {adx:.1f} below minimum {min_adx:.1f}",
            })

        # Dynamic screener metric gates
        gate_rejections.extend(self._evaluate_dynamic_gates(metrics, config_snapshot))

        bull_evidence: List[str] = []
        bear_evidence: List[str] = []

        # ── 1. Long Path: Price above VWAP ───────────────────────────
        if close > vwap and vwap_dist_pct >= min_dist:
            bull_evidence.append(f"Price ({close:.2f}) holding above VWAP ({vwap:.2f}) by +{vwap_dist_pct:.2f}%")

            # Check for extreme distance exhaustion
            if vwap_dist_pct > max_dist:
                gate_rejections.append({
                    "gate_id": "trend_filter:VWAP_OVEREXTENDED",
                    "param_keys": ["MAX_VWAP_DIST_PCT"],
                    "observed": vwap_dist_pct,
                    "threshold": max_dist,
                    "label": "VWAP Overextended",
                    "reason": f"VWAP distance +{vwap_dist_pct:.2f}% exceeds maximum +{max_dist:.1f}% threshold",
                })

            # Macro Trend Hierarchy check: Price > SMA200 or EMA20 > EMA50
            is_uptrend = (sma200 > 0 and close > sma200) or (ema50 > 0 and ema20 > ema50) or (ema20 > 0 and close > ema20)
            if is_uptrend:
                bull_evidence.append("Macro trend hierarchy confirmed (above 200 SMA or 20/50 EMA aligned)")
            else:
                gate_rejections.append({
                    "gate_id": "trend_filter:COUNTER_TREND_FILTER",
                    "param_keys": ["REQUIRE_TREND_ALIGNMENT"],
                    "observed": close,
                    "threshold": sma200,
                    "label": "Macro Trend Filter",
                    "reason": f"Price ({close:.2f}) below 200 SMA ({sma200:.2f}) during VWAP breakout",
                })

            if plus_di > minus_di:
                bull_evidence.append(f"+DI ({plus_di:.1f}) dominant over -DI ({minus_di:.1f})")

            if gate_rejections:
                return StrategySignal(
                    symbol=ticker,
                    direction="NEUTRAL",
                    conviction=0.0,
                    setup_quality="unfavorable",
                    strategy_name=self.name,
                    rationale=f"VWAP Long filtered out by {len(gate_rejections)} gate(s).",
                    metrics=metrics,
                    bull_evidence=bull_evidence,
                    bear_evidence=bear_evidence,
                    catalyst_risk=catalyst_risk,
                    timeframe=timeframe,
                    gate_rejections=gate_rejections,
                    config_epoch_id=epoch_id,
                )

            # Sizing conviction
            conv = base_conv
            if plus_di > minus_di:
                conv += 0.05
            if adx >= 25.0:
                conv += 0.05
            if 0.5 <= vwap_dist_pct <= 2.0:
                conv += 0.05  # Sweet spot for institutional trend continuation
            conv = min(max_conv, conv)

            quality = "exceptional" if conv >= 0.85 else ("favorable" if conv >= 0.75 else "mediocre")
            rationale = (
                f"VWAP Session Long: Price {close:.2f} holding +{vwap_dist_pct:.2f}% above session VWAP {vwap:.2f} "
                f"with confirmed trend hierarchy and positive volume."
            )

            return StrategySignal(
                symbol=ticker,
                direction="LONG",
                conviction=round(conv, 2),
                setup_quality=quality,
                strategy_name=self.name,
                rationale=rationale,
                metrics=metrics,
                bull_evidence=bull_evidence,
                bear_evidence=bear_evidence,
                catalyst_risk=catalyst_risk,
                timeframe=timeframe,
                gate_rejections=None,
                config_epoch_id=epoch_id,
            )

        # ── 2. Short / Breakdown Path: Price below VWAP ───────────────
        elif close < vwap and vwap_dist_pct <= -min_dist:
            bear_evidence.append(f"Price ({close:.2f}) broke below session VWAP ({vwap:.2f}) by {vwap_dist_pct:.2f}%")

            if minus_di > plus_di:
                bear_evidence.append(f"-DI ({minus_di:.1f}) dominant over +DI ({plus_di:.1f})")

            # Dual-Track Bear Shield: Only broad market index benchmarks trade short ETPs
            if self.is_broad_index(ticker, broad_indices):
                if gate_rejections:
                    return StrategySignal(
                        symbol=ticker,
                        direction="NEUTRAL",
                        conviction=0.0,
                        setup_quality="unfavorable",
                        strategy_name=self.name,
                        rationale=f"VWAP Short filtered out by {len(gate_rejections)} gate(s).",
                        metrics=metrics,
                        bull_evidence=bull_evidence,
                        bear_evidence=bear_evidence,
                        catalyst_risk=catalyst_risk,
                        timeframe=timeframe,
                        gate_rejections=gate_rejections,
                        config_epoch_id=epoch_id,
                    )

                conv = short_base_conv
                if minus_di > plus_di:
                    conv += 0.05
                if adx >= 25.0:
                    conv += 0.05
                conv = min(max_conv, conv)

                quality = "exceptional" if conv >= 0.85 else ("favorable" if conv >= 0.75 else "mediocre")
                rationale = (
                    f"VWAP Benchmark Short: Broad index {ticker} below session VWAP ({vwap_dist_pct:.2f}%). "
                    f"Routing to 3x Inverse Short ETP."
                )

                return StrategySignal(
                    symbol=ticker,
                    direction="SHORT",
                    conviction=round(conv, 2),
                    setup_quality=quality,
                    strategy_name=self.name,
                    rationale=rationale,
                    metrics=metrics,
                    bull_evidence=bull_evidence,
                    bear_evidence=bear_evidence,
                    catalyst_risk=catalyst_risk,
                    timeframe=timeframe,
                    gate_rejections=None,
                    config_epoch_id=epoch_id,
                )
            else:
                # Single Equity Cash Defense
                rationale = (
                    f"VWAP Cash Defense: Equity {ticker} is trading below session VWAP ({vwap_dist_pct:.2f}%). "
                    f"100% Cash Defense enforced to eliminate short squeeze risk."
                )
                return StrategySignal(
                    symbol=ticker,
                    direction="NEUTRAL",
                    conviction=0.0,
                    setup_quality="unfavorable",
                    strategy_name=self.name,
                    rationale=rationale,
                    metrics=metrics,
                    bull_evidence=bull_evidence,
                    bear_evidence=bear_evidence,
                    catalyst_risk=catalyst_risk,
                    timeframe=timeframe,
                    gate_rejections=[{
                        "gate_id": "dual_track_gate:SINGLE_EQUITY_CASH_DEFENSE",
                        "param_keys": ["BROAD_INDICES"],
                        "observed": ticker,
                        "threshold": "broad_indices_only",
                        "label": "Dual-Track Cash Defense",
                        "reason": f"Single equity {ticker} below VWAP: 100% Cash Defense enforced",
                    }],
                    config_epoch_id=epoch_id,
                )

        # ── 3. Neutral / Dead-Zone Buffer Path ─────────────────────────
        gate_rejections.append({
            "gate_id": "trend_filter:VWAP_DEAD_ZONE",
            "param_keys": ["MIN_VWAP_DIST_PCT"],
            "observed": vwap_dist_pct,
            "threshold": min_dist,
            "label": "VWAP Dead-Zone",
            "reason": f"Price within dead-zone buffer of VWAP ({vwap_dist_pct:+.2f}% vs ±{min_dist:.2f}%)",
        })

        return StrategySignal(
            symbol=ticker,
            direction="NEUTRAL",
            conviction=0.0,
            setup_quality="unfavorable",
            strategy_name=self.name,
            rationale=f"Price near VWAP ({vwap_dist_pct:+.2f}%): inside dead-zone buffer.",
            metrics=metrics,
            bull_evidence=bull_evidence,
            bear_evidence=bear_evidence,
            catalyst_risk=catalyst_risk,
            timeframe=timeframe,
            gate_rejections=gate_rejections,
            config_epoch_id=epoch_id,
        )

    def filter_universe(
        self,
        universe_metrics: Dict[str, Dict[str, Any]],
        top_n: int = 5,
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> List[StrategySignal]:
        """Filter universe and return top actionable VWAP trend signals."""
        params = self._resolve_params(config_snapshot)
        max_catalyst_risk: float = params["max_catalyst_risk"]
        signals: List[StrategySignal] = []

        for ticker, metrics in universe_metrics.items():
            try:
                sig = self.evaluate_candidate(ticker, metrics, config_snapshot=config_snapshot)
                if sig.is_actionable and sig.catalyst_risk <= max_catalyst_risk:
                    signals.append(sig)
            except Exception as exc:
                logger.debug("VWAP evaluation failed for %s: %s", ticker, exc)

        # Sort by conviction descending, then absolute distance from VWAP
        def _sort_key(s: StrategySignal) -> Tuple[float, float]:
            dist = abs(_safe_float(s.metrics.get("vwap_dist_pct"), 0.0) or 0.0)
            return (s.conviction, dist)

        signals.sort(key=_sort_key, reverse=True)
        return signals[:top_n]
