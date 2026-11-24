"""Candidate screening and quantitative ranking layer using technical metrics and NLI routing.

This module provides rapid directional bias classification, setup quality
scoring, and catalyst-risk estimation for a universe of assets.  It uses
Laya's ``Router`` (ModernBERT-backed) to evaluate natural-language state
documents derived from technical metrics, returning structured screening
results suitable for downstream portfolio construction.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

from trading_system.config import settings
from trading_system.config_schema import (
    ALLOWLISTED_SCREENER_METRICS,
    evaluate_screener_metric_gate,
)
from trading_system.strategies.base import BaseStrategy, StrategySignal, read_config_value, safe_float as _safe_float

if TYPE_CHECKING:
    from trading_system.config_bus import ConfigSnapshot

try:
    from laya import Router
    HAS_LAYA = True
except ImportError:
    Router = None
    HAS_LAYA = False

logger = logging.getLogger(__name__)

# Mapping from canonical internal gate_id -> tunable config schema parameter keys and display label.
GATE_ID_METADATA: Dict[str, Tuple[List[str], str]] = {
    "adx_trend_gate": (["MIN_ADX_TREND"], "Minimum ADX Trend"),
    "fallback_adx_gate": (["FALLBACK_SCREENER_MIN_ADX"], "Fallback Screener Min ADX"),
    "setup_score_gate": (["SCREENER_MIN_SETUP_SCORE"], "Min Screener Setup Score"),
    "catalyst_risk_gate": (["SCREENER_MAX_CATALYST_RISK", "ENTRY_EARNINGS_BLACKOUT_DAYS"], "Max Catalyst Risk"),
    "earnings_blackout_gate": (["ENTRY_EARNINGS_BLACKOUT_DAYS", "SCREENER_MAX_CATALYST_RISK"], "Entry Earnings Blackout"),
    "ema_slope_gate": (["MIN_EMA20_SLOPE_PCT"], "Min 20-EMA 5d Slope"),
    "high_vol_regime_gate": (
        ["HIGH_VOL_MAX_REALIZED_VOL", "HIGH_VOL_MAX_ATR_SPIKE_RATIO"],
        "High Volatility Regime Gate",
    ),
    "dual_track_cash_defense": ([], "Dual-Track Cash Defense"),
    "ema_trend_hierarchy": ([], "EMA Trend Alignment"),
    "macd_bull_conflict": ([], "MACD Histogram Conflict"),
    "macd_bear_conflict": ([], "MACD Histogram Conflict"),
    "dmi_polarity_gate": ([], "DMI Polarity Direction"),
    "spread_gate": (["ENFORCE_SPREAD_GATE", "MAX_BID_ASK_SPREAD_PCT", "MAX_SPREAD_TO_STOP_RATIO"], "Bid-Ask Spread Gate"),
    "cooldown_gate": (["TRADE_LOSS_COOLDOWN_SECONDS"], "Loss Cooldown Gate"),
    "circuit_breaker_gate": (["ENFORCE_DAILY_CIRCUIT_BREAKER", "DAILY_MAX_DRAWDOWN_PCT"], "Daily Circuit Breaker"),
    "correlation_gate": (["ENFORCE_CORRELATION_LIMITS", "MAX_SECTOR_POSITIONS", "MAX_PORTFOLIO_BETA_HEAT"], "Correlation & Beta Heat"),
    "stale_epoch_superseded": ([], "Stale Epoch Superseded"),
}


def _call_strategy_evaluate(
    strategy: BaseStrategy,
    ticker: str,
    metrics: Dict[str, Any],
    config_snapshot: Optional[ConfigSnapshot] = None,
) -> StrategySignal:
    """Invoke strategy.evaluate_candidate, gracefully handling legacy 2-arg mocks."""
    if config_snapshot is None:
        return strategy.evaluate_candidate(ticker, metrics)
    try:
        return strategy.evaluate_candidate(ticker, metrics, config_snapshot=config_snapshot)
    except TypeError as exc:
        if "config_snapshot" not in str(exc):
            raise
        return strategy.evaluate_candidate(ticker, metrics)


def _compact_metric_row(raw_metrics: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    """Extract allowlisted finite numeric indicators sorted deterministically by key."""
    if not raw_metrics:
        return {}
    row: Dict[str, float] = {}
    for key in sorted(ALLOWLISTED_SCREENER_METRICS):
        if key not in raw_metrics:
            continue
        parsed = _safe_float(raw_metrics.get(key), None)
        if parsed is not None:
            row[key] = round(parsed, 6)
    return row


def extract_compact_indicators(
    universe_metrics: Optional[Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Dict[str, float]]:
    """Build compact per-ticker indicator vectors restricted to ALLOWLISTED_SCREENER_METRICS."""
    if not universe_metrics:
        return {}
    return {
        str(ticker): _compact_metric_row(metrics)
        for ticker, metrics in universe_metrics.items()
    }


def _percentile(sorted_vals: List[float], q: float) -> float:
    """Compute linear-interpolated quantile q in [0, 1] on a non-empty sorted list."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def compute_metric_percentiles(
    compact_indicators: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Compute cross-sectional distribution stats (min, p25, median, p75, max, count)."""
    buckets: Dict[str, List[float]] = {}
    for row in compact_indicators.values():
        if not row:
            continue
        for key, raw_val in row.items():
            if key not in ALLOWLISTED_SCREENER_METRICS:
                continue
            val = _safe_float(raw_val, None)
            if val is not None:
                buckets.setdefault(key, []).append(val)

    summary: Dict[str, Dict[str, float]] = {}
    for key in sorted(buckets):
        vals = sorted(buckets[key])
        summary[key] = {
            "count": float(len(vals)),
            "min": round(vals[0], 6),
            "p25": round(_percentile(vals, 0.25), 6),
            "median": round(_percentile(vals, 0.50), 6),
            "p75": round(_percentile(vals, 0.75), 6),
            "max": round(vals[-1], 6),
        }
    return summary


def _diversify_by_category(
    results: List[ScreenerResult],
    top_n: int,
    max_per_category: int,
) -> List[ScreenerResult]:
    """Select top_n results while prioritizing balanced category coverage."""
    if top_n <= 0:
        return []
    try:
        from trading_system.data.leveraged_universe import LeveragedUniverse

        univ = LeveragedUniverse()
        category_counts: Dict[str, int] = {}
        diversified: List[ScreenerResult] = []
        overflow: List[ScreenerResult] = []
        cap = max(1, int(max_per_category))

        for res in results:
            cat = univ.get_category_for_symbol(res.ticker)
            count = category_counts.get(cat, 0)
            if count < cap:
                diversified.append(res)
                category_counts[cat] = count + 1
            else:
                overflow.append(res)
        return (diversified + overflow)[:top_n]
    except Exception as exc:
        logger.debug("Sector-diversified selection fallback: %s", exc)
        return results[:top_n]


def _append_unique_rejection(
    rejections: List[Dict[str, Any]], entry: Dict[str, Any]
) -> None:
    """Append a structured rejection dict if gate_id is not already present."""
    gate_id = entry.get("gate_id")
    if any(r.get("gate_id") == gate_id for r in rejections):
        return
    rejections.append(entry)


def _evaluate_dynamic_snapshot_gates(
    metrics: Mapping[str, Any],
    config_snapshot: Optional[ConfigSnapshot],
) -> List[Dict[str, Any]]:
    """Evaluate dynamic screener_metric_gate rules on a ConfigSnapshot safely."""
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


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ScreenerResult:
    """Result produced by :class:`LayaScreener` or :class:`TechnicalFallbackScreener` for a single asset."""

    ticker: str
    bias: str = "bullish"
    bias_confidence: float = 0.85
    setup_score: str = "favorable"
    setup_confidence: float = 0.85
    catalyst_risk: float = 0.0
    overall_confidence: float = 0.85
    strategy_type: str = "vwap"
    metrics: Dict[str, Any] = field(default_factory=dict)
    state_document: str = ""
    market_regime: str = "TRENDING_BREAKOUT"
    # Optional compatibility attributes
    close: float = 0.0
    trend: str = "BULLISH"
    adx: float = 25.0
    rsi: float = 50.0
    strategy_signal: Optional[Any] = None
    gate_rejections: Optional[List[Dict[str, Any]]] = None
    config_epoch_id: Optional[str] = None


@dataclass
class CycleFunnelReport:
    """Structured end-to-end screening, strategy, debate, gate, and execution funnel telemetry."""

    timestamp: str = ""
    config_epoch_id: Optional[str] = None
    stage_counts: Dict[str, int] = field(
        default_factory=lambda: {
            "universe": 0,
            "screened": 0,
            "strategy": 0,
            "debate": 0,
            "gates": 0,
            "executed": 0,
        }
    )
    gate_breakdown: List[Dict[str, Any]] = field(default_factory=list)
    _compact_indicators: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metric_percentiles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    passed_tickers: List[str] = field(default_factory=list)
    executed_tickers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize funnel telemetry into a JSON-compatible dictionary."""
        return {
            "timestamp": self.timestamp,
            "config_epoch_id": self.config_epoch_id,
            "stage_counts": dict(self.stage_counts),
            "gate_breakdown": [dict(item) for item in self.gate_breakdown],
            "_compact_indicators": {
                ticker: dict(vals) for ticker, vals in self._compact_indicators.items()
            },
            "metric_percentiles": {
                metric: dict(stats) for metric, stats in self.metric_percentiles.items()
            },
            "passed_tickers": list(self.passed_tickers),
            "executed_tickers": list(self.executed_tickers),
        }


# ---------------------------------------------------------------------------
# Screener
# ---------------------------------------------------------------------------


class LayaScreener:
    """Quantitative candidate screener powered by technical indicators and NLI routing.

    Parameters:
        min_setup_score: Minimum acceptable setup quality label.  Assets
            scoring below this are filtered out.  Default ``"favorable"``.
        max_catalyst_risk: Maximum acceptable catalyst-risk probability.
            Assets above this threshold are filtered out.  Default ``0.25``.
        top_n: Number of top-ranked candidates to return from
            :meth:`scan_universe`.  Default ``5``.
    """

    # Canonical ordering for the ``score``-type setup labels.
    _SCORE_RANK: Dict[str, int] = {
        "unfavorable": 0,
        "mediocre": 1,
        "favorable": 2,
        "exceptional": 3,
    }

    def __init__(
        self,
        min_setup_score: Optional[str] = None,
        max_catalyst_risk: Optional[float] = None,
        top_n: Optional[int] = None,
        strategy_name: Optional[str] = None,
    ) -> None:
        self.router: Any = Router(preload=True) if (HAS_LAYA and Router is not None) else None
        self._explicit_min_setup_score = min_setup_score is not None
        self._explicit_max_catalyst_risk = max_catalyst_risk is not None
        self._explicit_top_n = top_n is not None
        self._explicit_strategy_name = strategy_name is not None

        self.min_setup_score: str = (
            str(min_setup_score).lower()
            if min_setup_score is not None
            else str(getattr(settings, "SCREENER_MIN_SETUP_SCORE", "favorable")).lower()
        )
        self.max_catalyst_risk: float = (
            float(max_catalyst_risk)
            if max_catalyst_risk is not None
            else float(getattr(settings, "SCREENER_MAX_CATALYST_RISK", 0.25))
        )
        self.top_n: int = (
            int(top_n)
            if top_n is not None
            else int(getattr(settings, "SCREENER_TOP_N", 5))
        )
        self._score_rank: Dict[str, int] = dict(self._SCORE_RANK)
        self.last_evaluated_results: List[ScreenerResult] = []
        self.last_passed_results: List[ScreenerResult] = []

        # Resolve active pluggable trading strategy
        from trading_system.strategies.registry import StrategyRegistry

        active_name = strategy_name or getattr(settings, "ACTIVE_STRATEGY", "vwap")
        self._init_strategy_name = active_name
        try:
            self.strategy = StrategyRegistry.get(active_name)
        except Exception as exc:
            logger.warning(
                "Failed to initialize strategy '%s' (%s), falling back to vwap",
                active_name,
                exc,
            )
            self.strategy = StrategyRegistry.get("vwap")

        # ----- Laya question bank -----
        self.questions: Dict[str, Dict[str, Any]] = {
            "bias": {
                "type": "choice",
                "instructions": "What is the technical market structure and momentum bias?",
                "criteria": {
                    "bullish": "Price above Daily 200 SMA, 20 EMA > 50 EMA, RSI > 50, breaking out or pulling back to support",
                    "bearish": "Price below 200 SMA, breakdown below support, RSI < 45, heavy sell volume",
                    "neutral": "Choppy sideways consolidation, range-bound, conflicting moving averages",
                },
            },
            "setup_score": {
                "type": "score",
                "instructions": "Rate the breakout or trend-following quality and reward-to-risk asymmetry.",
                "criteria": [
                    "unfavorable",
                    "mediocre",
                    "favorable",
                    "exceptional",
                ],
            },
            "catalyst_risk": {
                "type": "noul",
                "instructions": (
                    "Is there an imminent earnings release, FDA decision, "
                    "or macro binary event within 48 hours?"
                ),
            },
        }

    def _resolve_params(
        self, config_snapshot: Optional[ConfigSnapshot] = None
    ) -> Dict[str, Any]:
        """Resolve screening cutoffs from a ConfigSnapshot or legacy constructor/settings."""
        if config_snapshot is not None:
            return {
                "config_epoch_id": read_config_value(config_snapshot, "config_epoch_id", None),
                "min_setup_score": str(
                    read_config_value(config_snapshot, "SCREENER_MIN_SETUP_SCORE", self.min_setup_score)
                ).lower(),
                "max_catalyst_risk": float(
                    read_config_value(config_snapshot, "SCREENER_MAX_CATALYST_RISK", self.max_catalyst_risk)
                ),
                "top_n": int(read_config_value(config_snapshot, "SCREENER_TOP_N", self.top_n)),
                "max_per_category": int(
                    read_config_value(config_snapshot, "SCREENER_MAX_PER_CATEGORY", 2)
                ),
                "min_adx_trend": float(read_config_value(config_snapshot, "MIN_ADX_TREND", 18.0)),
                "fallback_min_adx": float(
                    read_config_value(config_snapshot, "FALLBACK_SCREENER_MIN_ADX", 20.0)
                ),
                "active_strategy": str(
                    read_config_value(config_snapshot, "ACTIVE_STRATEGY", self._init_strategy_name)
                ),
            }

        return {
            "config_epoch_id": None,
            "min_setup_score": (
                self.min_setup_score
                if self._explicit_min_setup_score
                else str(getattr(settings, "SCREENER_MIN_SETUP_SCORE", self.min_setup_score)).lower()
            ),
            "max_catalyst_risk": (
                self.max_catalyst_risk
                if self._explicit_max_catalyst_risk
                else float(getattr(settings, "SCREENER_MAX_CATALYST_RISK", self.max_catalyst_risk))
            ),
            "top_n": (
                self.top_n
                if self._explicit_top_n
                else int(getattr(settings, "SCREENER_TOP_N", self.top_n))
            ),
            "max_per_category": int(getattr(settings, "SCREENER_MAX_PER_CATEGORY", 2)),
            "min_adx_trend": float(getattr(settings, "MIN_ADX_TREND", 18.0)),
            "fallback_min_adx": float(getattr(settings, "FALLBACK_SCREENER_MIN_ADX", 20.0)),
            "active_strategy": self._init_strategy_name,
        }

    def _active_strategy(self, params: Dict[str, Any]) -> BaseStrategy:
        """Resolve strategy object for current epoch without clobbering injected mocks."""
        desired = params["active_strategy"]
        if (
            not self._explicit_strategy_name
            and desired != getattr(self.strategy, "name", desired)
            and desired != self._init_strategy_name
        ):
            try:
                from trading_system.strategies.registry import StrategyRegistry

                return StrategyRegistry.get(desired)
            except Exception:
                pass
        return self.strategy

    # ------------------------------------------------------------------
    # Market Regime Classifier
    # ------------------------------------------------------------------

    @staticmethod
    def classify_regime(
        metrics: Dict[str, Any],
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> str:
        """Classify market regime into CHOP_RANGE, TRENDING_BREAKOUT, or HIGH_VOL_ERRATIC.

        Logic:
        - HIGH_VOL_ERRATIC: Realized Vol > max_realized_vol or ATR spike ratio >= max_atr_spike
        - TRENDING_BREAKOUT: ADX >= breakout_adx_min and EMA20 > EMA50
        - CHOP_RANGE: Low ADX or range-bound moving averages
        """
        if config_snapshot is not None:
            max_vol = float(read_config_value(config_snapshot, "HIGH_VOL_MAX_REALIZED_VOL", 0.65))
            max_atr_spike = float(read_config_value(config_snapshot, "HIGH_VOL_MAX_ATR_SPIKE_RATIO", 1.50))
            breakout_adx_min = float(read_config_value(config_snapshot, "REGIME_BREAKOUT_ADX_MIN", 25.0))
        else:
            max_vol = float(getattr(settings, "HIGH_VOL_MAX_REALIZED_VOL", 0.65))
            max_atr_spike = float(getattr(settings, "HIGH_VOL_MAX_ATR_SPIKE_RATIO", 1.50))
            breakout_adx_min = float(getattr(settings, "REGIME_BREAKOUT_ADX_MIN", 25.0))

        vol = _safe_float(metrics.get("realized_vol"), 0.0) or 0.0
        atr_spike_raw = (
            metrics.get("atr_spike_ratio")
            if metrics.get("atr_spike_ratio") is not None
            else metrics.get("atr_ratio")
        )
        atr_spike = _safe_float(atr_spike_raw, 1.0)
        if atr_spike is None:
            atr_spike = 1.0
        if vol > max_vol or atr_spike >= max_atr_spike:
            return "HIGH_VOL_ERRATIC"

        adx = _safe_float(metrics.get("adx"), 25.0)
        if adx is None:
            adx = 25.0
        ema20 = _safe_float(metrics.get("ema20"), 0.0) or 0.0
        ema50 = _safe_float(metrics.get("ema50"), 0.0) or 0.0
        if adx >= breakout_adx_min and (ema20 > ema50):
            return "TRENDING_BREAKOUT"

        return "CHOP_RANGE"

    get_market_regime = classify_regime

    # ------------------------------------------------------------------
    # State-document formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_asset_state(ticker: str, metrics: Dict[str, Any]) -> str:
        """Build a natural-language state document for Laya.

        The document summarises price structure, trend, momentum, volatility,
        and upcoming catalysts in a deterministic, human-readable format.
        """
        close: float = _safe_float(metrics.get("close"), 0.0) or 0.0
        sma200: float = _safe_float(metrics.get("sma200"), 0.0) or 0.0
        sma200_dist: float = _safe_float(metrics.get("sma200_dist"), 0.0) or 0.0
        rsi_base = _safe_float(metrics.get("rsi"), 50.0) or 50.0
        rsi3: float = _safe_float(metrics.get("rsi3"), rsi_base) or rsi_base
        rsi7: float = _safe_float(metrics.get("rsi7"), rsi_base) or rsi_base
        rsi14: float = _safe_float(metrics.get("rsi14"), rsi_base) or rsi_base
        sma5: float = _safe_float(metrics.get("sma5"), close) or close
        ema20: float = _safe_float(metrics.get("ema20"), 0.0) or 0.0
        ema50: float = _safe_float(metrics.get("ema50"), 0.0) or 0.0
        adx: float = _safe_float(metrics.get("adx"), 0.0) or 0.0
        atr: float = _safe_float(metrics.get("atr"), 0.0) or 0.0

        earnings_raw = metrics.get("earnings_in_days")
        earnings_num = _safe_float(earnings_raw, None)
        earnings_info: str = (
            f"{earnings_num:.1f}d"
            if earnings_num is not None
            else (str(earnings_raw) if earnings_raw is not None else "None within 7d")
        )

        state: str = (
            f"Asset: {ticker}\n"
            f"Daily Close: {close:.2f} | 200 SMA: {sma200:.2f} "
            f"(Distance: {sma200_dist:+.1f}%)\n"
            f"1H Trend: 20 EMA={ema20:.2f}, "
            f"50 EMA={ema50:.2f} | "
            f"Trend: {metrics.get('trend', 'unknown')}\n"
            f"Momentum: RSI(3)={rsi3:.1f}, RSI(7)={rsi7:.1f}, "
            f"RSI(14)={rsi14:.1f}, ADX(14)={adx:.1f}\n"
            f"Mean Reversion Benchmarks: 5-Day SMA={sma5:.2f}, "
            f"ATR(14)={atr:.2f}\n"
            f"Upcoming Earnings: {earnings_info}"
        )
        return state

    # ------------------------------------------------------------------
    # Single-asset evaluation
    # ------------------------------------------------------------------

    def evaluate_single(
        self,
        ticker: str,
        metrics: Dict[str, Any],
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> ScreenerResult:
        """Screen a single asset and return a :class:`ScreenerResult` using the active strategy.

        On any error the method returns a *neutral* result with zero
        confidence so that downstream filters safely exclude it.
        """
        params = self._resolve_params(config_snapshot)
        epoch_id: Optional[str] = params["config_epoch_id"]
        strategy = self._active_strategy(params)
        state: str = self.format_asset_state(ticker, metrics)

        close = _safe_float(metrics.get("close"), 0.0) or 0.0
        adx = _safe_float(metrics.get("adx"), 20.0)
        if adx is None:
            adx = 20.0
        rsi_base = _safe_float(metrics.get("rsi"), 50.0) or 50.0
        rsi14 = _safe_float(metrics.get("rsi14"), rsi_base)
        if rsi14 is None:
            rsi14 = rsi_base

        # Delegate evaluation to the active pluggable strategy
        try:
            signal = _call_strategy_evaluate(
                strategy, ticker, metrics, config_snapshot=config_snapshot
            )
        except Exception:
            logger.exception("Strategy evaluation failed for %s", ticker)
            return ScreenerResult(
                ticker=ticker,
                bias="neutral",
                bias_confidence=0.0,
                setup_score="unfavorable",
                setup_confidence=0.0,
                catalyst_risk=1.0,
                overall_confidence=0.0,
                metrics=metrics,
                state_document=state,
                gate_rejections=[],
                config_epoch_id=epoch_id,
            )

        bias = (
            "bullish"
            if signal.direction == "LONG"
            else ("bearish" if signal.direction == "SHORT" else "neutral")
        )

        # Refine catalyst risk with Laya NLI router if available
        setup_score = signal.setup_quality
        bias_confidence = signal.conviction
        overall_confidence = signal.conviction
        catalyst_risk = signal.catalyst_risk
        gate_rejections: List[Dict[str, Any]] = (
            [dict(item) for item in signal.gate_rejections]
            if signal.gate_rejections
            else []
        )
        # Ensure dynamic gates from snapshot are also captured if a non-Directional strategy ran
        if config_snapshot is not None and not any(
            str(r.get("gate_id", "")).startswith("dynamic_gate:")
            for r in gate_rejections
        ):
            for dyn_rej in _evaluate_dynamic_snapshot_gates(metrics, config_snapshot):
                _append_unique_rejection(gate_rejections, dyn_rej)

        if self.router is not None and signal.is_actionable:
            try:
                raw: Dict[str, Any] = self.router.predict(state, self.questions)
                answers: Dict[str, Any] = raw.get("answers", {})
                cat_answer: Dict[str, Any] = answers.get("catalyst_risk", {})
                nli_cat_risk = _safe_float(cat_answer.get("noul"), catalyst_risk)
                if nli_cat_risk is not None:
                    catalyst_risk = max(catalyst_risk, nli_cat_risk)
            except Exception:
                logger.debug("Laya router prediction notice for %s", ticker)

        return ScreenerResult(
            ticker=ticker,
            bias=bias,
            bias_confidence=bias_confidence,
            setup_score=setup_score,
            setup_confidence=overall_confidence,
            catalyst_risk=catalyst_risk,
            overall_confidence=overall_confidence,
            strategy_type=signal.strategy_name,
            metrics=metrics,
            state_document=state,
            market_regime=self.classify_regime(metrics, config_snapshot=config_snapshot),
            close=close,
            trend=metrics.get("trend", "BULLISH"),
            adx=adx,
            rsi=rsi14,
            strategy_signal=signal,
            gate_rejections=gate_rejections,
            config_epoch_id=epoch_id or signal.config_epoch_id,
        )

    # ------------------------------------------------------------------
    # Universe scan
    # ------------------------------------------------------------------

    def scan_universe(
        self,
        universe_metrics: Dict[str, Dict[str, Any]],
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> List[ScreenerResult]:
        """Screen an entire universe of assets in parallel.

        Assets are evaluated concurrently (up to 4 threads).  Results are
        filtered by :meth:`passes_filter`, sorted by setup quality and
        confidence, and trimmed to resolved ``top_n``.
        """
        params = self._resolve_params(config_snapshot)
        top_n: int = params["top_n"]
        max_per_category: int = params["max_per_category"]
        n_assets: int = len(universe_metrics)
        logger.info("Screening %d assets...", n_assets)

        results: List[ScreenerResult] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_ticker: Dict[
                concurrent.futures.Future[ScreenerResult], str
            ] = {
                executor.submit(
                    self.evaluate_single, ticker, metrics, config_snapshot
                ): ticker
                for ticker, metrics in universe_metrics.items()
            }

            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result: ScreenerResult = future.result()
                except Exception:
                    logger.exception(
                        "Unexpected error evaluating %s", ticker
                    )
                    continue
                results.append(result)

        self.last_evaluated_results = list(results)

        # Filter ----------------------------------------------------------
        passed: List[ScreenerResult] = [
            r for r in results if self.passes_filter(r, config_snapshot=config_snapshot)
        ]
        self.last_passed_results = list(passed)

        for r in passed:
            logger.info(
                "Passed: %s (score=%s, risk=%.3f)",
                r.ticker,
                r.setup_score,
                r.catalyst_risk,
            )

        # Sort: best setup first, then highest confidence ----------------
        passed.sort(
            key=lambda r: (
                self._score_rank.get(r.setup_score, 0),
                r.overall_confidence,
            ),
            reverse=True,
        )

        # Diversified selection across market categories/sectors
        laya_results = _diversify_by_category(
            passed, top_n=top_n, max_per_category=max_per_category
        )

        # If Laya returned 0 candidates (degenerate: all neutral, model not loaded, etc.)
        # fall back to the pure-technical screener so the engine always has signals to act on.
        if not laya_results:
            logger.warning(
                "LayaScreener returned 0 candidates — falling back to TechnicalFallbackScreener "
                "for pure-indicator signal generation."
            )
            fallback = TechnicalFallbackScreener(
                top_n=top_n,
                min_adx=params["fallback_min_adx"],
                strategy=self._active_strategy(params),
            )
            laya_results = fallback.scan_universe(
                universe_metrics, config_snapshot=config_snapshot
            )

        return laya_results

    # ------------------------------------------------------------------
    # Filter predicate
    # ------------------------------------------------------------------

    def passes_filter(
        self,
        result: ScreenerResult,
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> bool:
        """Check whether a screening result meets minimum quality criteria."""
        params = self._resolve_params(config_snapshot)
        min_adx: float = params["min_adx_trend"]
        min_setup_score: str = params["min_setup_score"]
        max_catalyst_risk: float = params["max_catalyst_risk"]

        rejections: List[Dict[str, Any]] = (
            list(result.gate_rejections) if result.gate_rejections is not None else []
        )
        passed_all = True

        # Check dynamic rules when snapshot provided (handles external ScreenerResult inputs)
        if config_snapshot is not None and not any(
            str(r.get("gate_id", "")).startswith("dynamic_gate:")
            for r in rejections
        ):
            dyn_rejections = _evaluate_dynamic_snapshot_gates(result.metrics, config_snapshot)
            for dyn in dyn_rejections:
                _append_unique_rejection(rejections, dyn)
        if any(str(r.get("gate_id", "")).startswith("dynamic_gate:") for r in rejections):
            passed_all = False

        if result.bias not in ("bullish", "bearish"):
            passed_all = False

        # Volatility decay protection for 3x leverage
        adx: float = _safe_float(result.metrics.get("adx"), 0.0) or 0.0
        if adx < min_adx:
            _append_unique_rejection(
                rejections,
                {
                    "gate_id": "adx_trend_gate",
                    "param_keys": ["MIN_ADX_TREND"],
                    "observed": round(adx, 4),
                    "threshold": min_adx,
                    "label": "Minimum ADX Trend",
                },
            )
            passed_all = False

        result_rank: int = self._score_rank.get(result.setup_score, 0)
        min_rank: int = self._score_rank.get(min_setup_score, 0)
        if result_rank < min_rank:
            _append_unique_rejection(
                rejections,
                {
                    "gate_id": "setup_score_gate",
                    "param_keys": ["SCREENER_MIN_SETUP_SCORE"],
                    "observed": result.setup_score,
                    "threshold": min_setup_score,
                    "label": "Min Screener Setup Score",
                },
            )
            passed_all = False

        if result.catalyst_risk >= max_catalyst_risk:
            _append_unique_rejection(
                rejections,
                {
                    "gate_id": "catalyst_risk_gate",
                    "param_keys": ["SCREENER_MAX_CATALYST_RISK", "ENTRY_EARNINGS_BLACKOUT_DAYS"],
                    "observed": round(float(result.catalyst_risk), 4),
                    "threshold": max_catalyst_risk,
                    "label": "Max Catalyst Risk",
                },
            )
            passed_all = False

        result.gate_rejections = rejections
        return passed_all


# ---------------------------------------------------------------------------
# Technical Fallback Screener
# ---------------------------------------------------------------------------


class TechnicalFallbackScreener:
    """Pure-indicator screener used when Laya NLI is unavailable or returns 0 candidates.

    Produces directional signals from pre-computed technical metrics without
    requiring a natural language inference engine.
    """

    def __init__(
        self,
        top_n: Optional[int] = None,
        min_adx: Optional[float] = None,
        strategy: Optional[BaseStrategy] = None,
    ) -> None:
        self._explicit_top_n = top_n is not None
        self._explicit_min_adx = min_adx is not None
        self.top_n = (
            int(top_n)
            if top_n is not None
            else int(getattr(settings, "SCREENER_TOP_N", 10))
        )
        self.min_adx = (
            float(min_adx)
            if min_adx is not None
            else float(getattr(settings, "FALLBACK_SCREENER_MIN_ADX", 20.0))
        )
        if strategy is not None:
            self.strategy = strategy
        else:
            from trading_system.strategies.registry import StrategyRegistry
            active_name = getattr(settings, "ACTIVE_STRATEGY", "vwap")
            try:
                self.strategy = StrategyRegistry.get(active_name)
            except Exception:
                self.strategy = StrategyRegistry.get("vwap")

    def _resolve_params(
        self, config_snapshot: Optional[ConfigSnapshot] = None
    ) -> Dict[str, Any]:
        """Resolve fallback screener thresholds from a ConfigSnapshot or legacy defaults."""
        if config_snapshot is not None:
            return {
                "config_epoch_id": read_config_value(config_snapshot, "config_epoch_id", None),
                "top_n": int(read_config_value(config_snapshot, "SCREENER_TOP_N", self.top_n)),
                "min_adx": float(read_config_value(config_snapshot, "FALLBACK_SCREENER_MIN_ADX", self.min_adx)),
                "max_per_category": int(read_config_value(config_snapshot, "SCREENER_MAX_PER_CATEGORY", 2)),
                "min_setup_score": str(
                    read_config_value(config_snapshot, "SCREENER_MIN_SETUP_SCORE", "favorable")
                ).lower(),
                "max_catalyst_risk": float(
                    read_config_value(config_snapshot, "SCREENER_MAX_CATALYST_RISK", 0.25)
                ),
            }
        return {
            "config_epoch_id": None,
            "top_n": self.top_n if self._explicit_top_n else int(getattr(settings, "SCREENER_TOP_N", self.top_n)),
            "min_adx": (
                self.min_adx
                if self._explicit_min_adx
                else float(getattr(settings, "FALLBACK_SCREENER_MIN_ADX", self.min_adx))
            ),
            "max_per_category": int(getattr(settings, "SCREENER_MAX_PER_CATEGORY", 2)),
            "min_setup_score": None,
            "max_catalyst_risk": None,
        }

    def evaluate_candidate(
        self,
        ticker: str,
        m: Dict[str, Any],
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> Optional[ScreenerResult]:
        """Evaluate a single ticker candidate using active directional strategy."""
        close_val = _safe_float(m.get("close"), None) if m else None
        if not m or close_val is None or close_val <= 0.0:
            return None

        params = self._resolve_params(config_snapshot)
        epoch_id: Optional[str] = params["config_epoch_id"]
        regime = LayaScreener.classify_regime(m, config_snapshot=config_snapshot)
        rsi_fallback = _safe_float(m.get("rsi"), 50.0) or 50.0
        rsi: float = _safe_float(m.get("rsi14"), rsi_fallback) or rsi_fallback
        adx: float = _safe_float(m.get("adx"), 0.0) or 0.0
        close: float = close_val

        if regime == "HIGH_VOL_ERRATIC":
            return ScreenerResult(
                ticker=ticker,
                bias="neutral",
                bias_confidence=0.0,
                setup_score="unfavorable",
                setup_confidence=0.0,
                catalyst_risk=1.0,
                overall_confidence=0.0,
                strategy_type="capital_preservation",
                metrics=m,
                market_regime="HIGH_VOL_ERRATIC",
                state_document=f"[HighVol] {ticker} Vol/ATR spike exceeds risk tolerance",
                gate_rejections=[
                    {
                        "gate_id": "high_vol_regime_gate",
                        "param_keys": ["HIGH_VOL_MAX_REALIZED_VOL", "HIGH_VOL_MAX_ATR_SPIKE_RATIO"],
                        "observed": _safe_float(m.get("realized_vol"), 0.0),
                        "threshold": (
                            getattr(config_snapshot, "HIGH_VOL_MAX_REALIZED_VOL", 0.65)
                            if config_snapshot is not None
                            else getattr(settings, "HIGH_VOL_MAX_REALIZED_VOL", 0.65)
                        ),
                        "label": "High Volatility Regime Gate",
                    }
                ],
                config_epoch_id=epoch_id,
            )

        if adx < params["min_adx"]:
            return None

        try:
            sig = _call_strategy_evaluate(
                self.strategy, ticker, m, config_snapshot=config_snapshot
            )
            if not sig.is_actionable or sig.direction == "NEUTRAL":
                return None

            gate_rejections = (
                [dict(item) for item in sig.gate_rejections]
                if sig.gate_rejections
                else []
            )
            if config_snapshot is not None:
                if not any(
                    str(r.get("gate_id", "")).startswith("dynamic_gate:")
                    for r in gate_rejections
                ):
                    gate_rejections.extend(
                        _evaluate_dynamic_snapshot_gates(m, config_snapshot)
                    )
                if gate_rejections:
                    return None
                score_rank = LayaScreener._SCORE_RANK
                if score_rank.get(sig.setup_quality, 0) < score_rank.get(params["min_setup_score"], 0):
                    return None
                if sig.catalyst_risk >= params["max_catalyst_risk"]:
                    return None

            bias = "bullish" if sig.direction == "LONG" else "bearish"
            return ScreenerResult(
                ticker=ticker,
                bias=bias,
                bias_confidence=sig.conviction,
                setup_score=sig.setup_quality,
                setup_confidence=sig.conviction,
                catalyst_risk=sig.catalyst_risk,
                overall_confidence=sig.conviction,
                strategy_type=sig.strategy_name,
                metrics=m,
                market_regime=regime,
                state_document=f"[Strategy:{sig.strategy_name}] {ticker} | Dir={sig.direction} Conv={sig.conviction:.2f}",
                close=close,
                trend=m.get("trend", "BULLISH" if bias == "bullish" else "BEARISH"),
                adx=adx,
                rsi=rsi,
                strategy_signal=sig,
                gate_rejections=gate_rejections,
                config_epoch_id=epoch_id or sig.config_epoch_id,
            )
        except Exception:
            logger.exception("Strategy evaluation failed in TechnicalFallbackScreener for %s", ticker)
            return None

    def scan_universe(
        self,
        universe_metrics: Dict[str, Dict[str, Any]],
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> List[ScreenerResult]:
        """Evaluate all assets in *universe_metrics* using pure technical indicators."""
        params = self._resolve_params(config_snapshot)
        top_n: int = params["top_n"]
        max_per_category: int = params["max_per_category"]
        results: List[ScreenerResult] = []

        for ticker, m in universe_metrics.items():
            candidate = self.evaluate_candidate(ticker, m, config_snapshot=config_snapshot)
            if candidate is not None and candidate.market_regime != "HIGH_VOL_ERRATIC" and candidate.bias != "neutral":
                results.append(candidate)

        # Sort: best setup first, then highest confidence
        results.sort(
            key=lambda r: (
                LayaScreener._SCORE_RANK.get(r.setup_score, 0),
                r.overall_confidence,
            ),
            reverse=True,
        )

        final_candidates = _diversify_by_category(
            results, top_n=top_n, max_per_category=max_per_category
        )

        logger.info(
            "TechnicalFallbackScreener: %d/%d assets passed (top_n=%d, returned=%d)",
            len(results), len(universe_metrics), top_n, len(final_candidates),
        )
        return final_candidates


# ---------------------------------------------------------------------------
# Funnel Aggregation & Offline Config Diff Simulation (R13 / A-ASM2)
# ---------------------------------------------------------------------------


def _enrich_rejection_entry(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a rejection entry with mapped param_keys and human label."""
    gate_id = str(raw.get("gate_id") or raw.get("gate") or "unknown_gate")
    default_keys, default_label = GATE_ID_METADATA.get(gate_id, ([], gate_id))
    param_keys = raw.get("param_keys")
    resolved_keys = (
        [str(k) for k in param_keys]
        if isinstance(param_keys, (list, tuple))
        else list(default_keys)
    )
    label = str(raw.get("label") or default_label)
    if gate_id.startswith("dynamic_gate:") and not resolved_keys:
        resolved_keys = [gate_id.split(":", 1)[1]]
    return {
        "gate_id": gate_id,
        "param_keys": resolved_keys,
        "label": label,
        "ticker": raw.get("ticker"),
    }


def _aggregate_gate_breakdown(
    rejection_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate per-gate rejection counts, mapped parameter keys, and sample tickers."""
    grouped: Dict[str, Dict[str, Any]] = {}
    seen_per_ticker: set[Tuple[str, str]] = set()

    for raw in rejection_records:
        item = _enrich_rejection_entry(raw)
        gate_id = item["gate_id"]
        ticker = item.get("ticker")
        if ticker is not None:
            dedup_key = (str(ticker), gate_id)
            if dedup_key in seen_per_ticker:
                continue
            seen_per_ticker.add(dedup_key)

        bucket = grouped.get(gate_id)
        if bucket is None:
            bucket = {
                "gate_id": gate_id,
                "label": item["label"],
                "param_keys": list(item["param_keys"]),
                "count": 0,
                "tickers": [],
            }
            grouped[gate_id] = bucket
        else:
            for key in item["param_keys"]:
                if key not in bucket["param_keys"]:
                    bucket["param_keys"].append(key)

        bucket["count"] += 1
        if ticker is not None and str(ticker) not in bucket["tickers"]:
            bucket["tickers"].append(str(ticker))

    breakdown = list(grouped.values())
    breakdown.sort(key=lambda entry: (-int(entry["count"]), str(entry["gate_id"])))
    return breakdown


def build_cycle_funnel_report(
    *,
    universe_metrics: Optional[Mapping[str, Mapping[str, Any]]] = None,
    screener_results: Optional[Sequence[ScreenerResult]] = None,
    passed_screener: Optional[Sequence[ScreenerResult]] = None,
    strategy_signals: Optional[Sequence[StrategySignal]] = None,
    debate_outcomes: Optional[Sequence[Any]] = None,
    gate_decisions: Optional[Sequence[Any]] = None,
    executed_orders: Optional[Sequence[Any]] = None,
    gate_rejections: Optional[Sequence[Mapping[str, Any]]] = None,
    stage_counts: Optional[Mapping[str, int]] = None,
    config_snapshot: Optional[ConfigSnapshot] = None,
    config_epoch_id: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> CycleFunnelReport:
    """Aggregate per-stage cycle counts, gate rejection attribution, and compact metrics."""
    epoch_id = config_epoch_id or (
        read_config_value(config_snapshot, "config_epoch_id", None)
        if config_snapshot is not None
        else None
    )
    ts = timestamp or datetime.now(timezone.utc).isoformat()

    compact_indicators: Dict[str, Dict[str, float]] = {}
    if universe_metrics is not None:
        compact_indicators = extract_compact_indicators(universe_metrics)
    else:
        for seq in (screener_results or (), strategy_signals or ()):
            for item in seq:
                if isinstance(item, Mapping):
                    sym = item.get("ticker") or item.get("symbol")
                    m = item.get("metrics")
                else:
                    sym = getattr(item, "ticker", None) or getattr(item, "symbol", None)
                    m = getattr(item, "metrics", None)
                if sym and sym not in compact_indicators and isinstance(m, Mapping):
                    compact_indicators[str(sym)] = _compact_metric_row(m)

    percentiles = compute_metric_percentiles(compact_indicators)

    # Collect rejections from all pipeline stages
    all_rejections: List[Dict[str, Any]] = []
    if screener_results:
        for res in screener_results:
            if res.gate_rejections:
                for rej in res.gate_rejections:
                    all_rejections.append({**rej, "ticker": rej.get("ticker", res.ticker)})
            sig = getattr(res, "strategy_signal", None)
            if sig is not None and getattr(sig, "gate_rejections", None):
                for rej in sig.gate_rejections:
                    all_rejections.append({**rej, "ticker": rej.get("ticker", res.ticker)})
    if strategy_signals:
        for sig in strategy_signals:
            if getattr(sig, "gate_rejections", None):
                for rej in sig.gate_rejections:
                    all_rejections.append({**rej, "ticker": rej.get("ticker", sig.symbol)})
    if gate_decisions:
        for card in gate_decisions:
            rejs = getattr(card, "gate_rejections", None)
            if rejs:
                sym = getattr(card, "ticker", None)
                for rej in rejs:
                    all_rejections.append({**rej, "ticker": rej.get("ticker", sym)})
    if gate_rejections:
        for rej in gate_rejections:
            all_rejections.append(dict(rej))

    breakdown = _aggregate_gate_breakdown(all_rejections)

    # Compute stage counts cleanly
    universe_count = (
        len(universe_metrics)
        if universe_metrics is not None
        else len(screener_results or strategy_signals or compact_indicators)
    )
    if passed_screener is not None:
        screened_count = len(passed_screener)
        passed_tickers = [r.ticker for r in passed_screener]
    elif screener_results is not None:
        passed_list = [
            r
            for r in screener_results
            if r.bias in ("bullish", "bearish") and r.setup_score != "unfavorable"
        ]
        screened_count = len(passed_list)
        passed_tickers = [r.ticker for r in passed_list]
    else:
        screened_count = universe_count
        passed_tickers = []

    if strategy_signals is not None:
        actionable_sigs = [s for s in strategy_signals if getattr(s, "is_actionable", False)]
        strategy_count = len(actionable_sigs)
        if not passed_tickers:
            passed_tickers = [s.symbol for s in actionable_sigs]
    elif screener_results is not None:
        sig_objs = [getattr(r, "strategy_signal", None) for r in screener_results]
        if any(s is not None for s in sig_objs):
            strategy_count = sum(1 for s in sig_objs if s is not None and getattr(s, "is_actionable", False))
        else:
            strategy_count = screened_count
    else:
        strategy_count = screened_count

    if debate_outcomes is not None:
        debate_count = sum(
            1
            for d in debate_outcomes
            if getattr(d, "consensus_decision", None) in ("BUY_ENTRY", "SELL_ENTRY", "SHORT_ENTRY")
            or bool(getattr(d, "approved", False))
        )
    else:
        debate_count = strategy_count

    if gate_decisions is not None:
        gates_count = sum(1 for c in gate_decisions if not getattr(c, "gate_rejections", None))
    else:
        gates_count = debate_count

    executed_tickers: List[str] = []
    if executed_orders is not None:
        executed_count = len(executed_orders)
        for item in executed_orders:
            sym = item if isinstance(item, str) else getattr(item, "ticker", None)
            if isinstance(item, Mapping):
                sym = item.get("ticker", sym)
            if sym:
                executed_tickers.append(str(sym))
    else:
        executed_count = 0

    resolved_counts = {
        "universe": int(universe_count),
        "screened": int(screened_count),
        "strategy": int(strategy_count),
        "debate": int(debate_count),
        "gates": int(gates_count),
        "executed": int(executed_count),
    }
    if stage_counts:
        for k, v in stage_counts.items():
            resolved_counts[str(k)] = int(v)

    return CycleFunnelReport(
        timestamp=ts,
        config_epoch_id=epoch_id,
        stage_counts=resolved_counts,
        gate_breakdown=breakdown,
        _compact_indicators=compact_indicators,
        metric_percentiles=percentiles,
        passed_tickers=passed_tickers,
        executed_tickers=executed_tickers,
    )


def simulate_screener_gate_impact(
    compact_universe_indicators: Mapping[str, Mapping[str, Any]],
    snapshot: ConfigSnapshot,
) -> Dict[str, Any]:
    """Offline deterministic simulation of screener and strategy impact for diff previews (A-ASM2)."""
    from trading_system.strategies.registry import StrategyRegistry

    sanitized_universe = extract_compact_indicators(compact_universe_indicators)
    active_strat = str(read_config_value(snapshot, "ACTIVE_STRATEGY", "vwap"))
    try:
        strategy = StrategyRegistry.get(active_strat)
    except Exception:
        strategy = StrategyRegistry.get("vwap")
    epoch_id = read_config_value(snapshot, "config_epoch_id", None)

    min_setup_score = str(read_config_value(snapshot, "SCREENER_MIN_SETUP_SCORE", "favorable")).lower()
    max_catalyst_risk = float(read_config_value(snapshot, "SCREENER_MAX_CATALYST_RISK", 0.25))
    min_adx = float(read_config_value(snapshot, "MIN_ADX_TREND", 18.0))
    top_n = int(read_config_value(snapshot, "SCREENER_TOP_N", 5))
    max_per_category = int(read_config_value(snapshot, "SCREENER_MAX_PER_CATEGORY", 2))
    score_rank = LayaScreener._SCORE_RANK

    all_rejections: List[Dict[str, Any]] = []
    passed_results: List[ScreenerResult] = []
    actionable_count = 0
    regime_counts: Dict[str, int] = {
        "TRENDING_BREAKOUT": 0,
        "CHOP_RANGE": 0,
        "HIGH_VOL_ERRATIC": 0,
    }

    for ticker, metrics in sanitized_universe.items():
        regime = LayaScreener.classify_regime(metrics, config_snapshot=snapshot)
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        signal = strategy.evaluate_candidate(ticker, metrics, config_snapshot=snapshot)
        if signal.is_actionable:
            actionable_count += 1

        rejections = [dict(r) for r in (signal.gate_rejections or [])]
        adx_val = _safe_float(metrics.get("adx"), 0.0) or 0.0
        bias = (
            "bullish"
            if signal.direction == "LONG"
            else ("bearish" if signal.direction == "SHORT" else "neutral")
        )

        if regime == "HIGH_VOL_ERRATIC":
            _append_unique_rejection(
                rejections,
                {
                    "gate_id": "high_vol_regime_gate",
                    "param_keys": ["HIGH_VOL_MAX_REALIZED_VOL", "HIGH_VOL_MAX_ATR_SPIKE_RATIO"],
                    "label": "High Volatility Regime Gate",
                },
            )
        if signal.is_actionable:
            if adx_val < min_adx:
                _append_unique_rejection(
                    rejections,
                    {
                        "gate_id": "adx_trend_gate",
                        "param_keys": ["MIN_ADX_TREND"],
                        "label": "Minimum ADX Trend",
                    },
                )
            if score_rank.get(signal.setup_quality, 0) < score_rank.get(min_setup_score, 0):
                _append_unique_rejection(
                    rejections,
                    {
                        "gate_id": "setup_score_gate",
                        "param_keys": ["SCREENER_MIN_SETUP_SCORE"],
                        "label": "Min Screener Setup Score",
                    },
                )
            if signal.catalyst_risk >= max_catalyst_risk:
                _append_unique_rejection(
                    rejections,
                    {
                        "gate_id": "catalyst_risk_gate",
                        "param_keys": ["SCREENER_MAX_CATALYST_RISK", "ENTRY_EARNINGS_BLACKOUT_DAYS"],
                        "label": "Max Catalyst Risk",
                    },
                )

        if signal.is_actionable and bias in ("bullish", "bearish") and not rejections:
            passed_results.append(
                ScreenerResult(
                    ticker=ticker,
                    bias=bias,
                    bias_confidence=signal.conviction,
                    setup_score=signal.setup_quality,
                    setup_confidence=signal.conviction,
                    catalyst_risk=signal.catalyst_risk,
                    overall_confidence=signal.conviction,
                    strategy_type=signal.strategy_name,
                    metrics=dict(metrics),
                    market_regime=regime,
                    close=_safe_float(metrics.get("close"), 0.0) or 0.0,
                    adx=adx_val,
                    rsi=_safe_float(metrics.get("rsi14"), _safe_float(metrics.get("rsi"), 50.0) or 50.0) or 50.0,
                    strategy_signal=signal,
                    gate_rejections=[],
                    config_epoch_id=epoch_id,
                )
            )
        else:
            for rej in rejections:
                all_rejections.append({**rej, "ticker": ticker})

    passed_results.sort(
        key=lambda r: (score_rank.get(r.setup_score, 0), r.overall_confidence),
        reverse=True,
    )
    selected = _diversify_by_category(
        passed_results, top_n=top_n, max_per_category=max_per_category
    )

    return {
        "config_epoch_id": epoch_id,
        "universe_count": len(sanitized_universe),
        "strategy_actionable_count": actionable_count,
        "passed_count": len(passed_results),
        "selected_count": len(selected),
        "passed_tickers": [r.ticker for r in passed_results],
        "selected_tickers": [r.ticker for r in selected],
        "regime_counts": regime_counts,
        "gate_breakdown": _aggregate_gate_breakdown(all_rejections),
        "metric_percentiles": compute_metric_percentiles(sanitized_universe),
    }
