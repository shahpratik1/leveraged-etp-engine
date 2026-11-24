"""Strategy Improvement Agent — Autonomous Pipeline Health & Alpha Calibration.

Continuously monitors screening yield, indicator integrity, and market regime conditions.
Detects signal starvation, validates mathematical feature completeness, and recommends
adaptive threshold adjustments based on real-time realized volatility.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Union

from trading_system.config import atomic_write_text, settings
from trading_system.strategies.base import read_config_value, safe_float as _safe_float

if TYPE_CHECKING:
    from trading_system.config_bus import ConfigSnapshot

logger = logging.getLogger(__name__)


class _SnapshotProxy:
    """Lightweight read-only view overriding select settings attributes for simulations."""

    def __init__(self, base: Any, **overrides: Any) -> None:
        self._base = base
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        if isinstance(self._base, Mapping):
            if name in self._base:
                return self._base[name]
            raise AttributeError(name)
        return getattr(self._base, name)


@dataclass
class ImprovementAudit:
    """Diagnostic health and optimization report produced by the Strategy Improvement Agent."""

    timestamp: str
    consecutive_zero_cycles: int
    health_status: str  # "HEALTHY", "DEGRADED", "STARVED", "CRITICAL"
    missing_indicators: List[str] = field(default_factory=list)
    null_metrics_count: int = 0
    total_assets_audited: int = 0
    mean_adx: float = 0.0
    mean_realized_vol: float = 0.0
    regime: str = "NORMAL"  # "LOW_VOL_CHOP", "NORMAL", "HIGH_VOL_TREND", "CRISIS"
    recommendations: List[str] = field(default_factory=list)
    action_taken: Optional[str] = None
    active_strategy: str = "vwap"
    config_epoch_id: Optional[str] = None
    parameter_recommendations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize audit report to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImprovementAudit":
        """Reconstruct ImprovementAudit safely from persisted JSONL rows."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


class StrategyImprovementAgent:
    """Autonomous agent that guards pipeline health and dynamically tunes strategy parameters.

    Key Responsibilities:
    1. Zero-Yield Watchdog: Tracks zero-signal cycles and warns before operator notices.
    2. Feature Completeness: Ensures indicators required by screeners exist and are valid.
    3. Regime-Aware Calibration: Audits active directional_trend parameters via Wilder ADX(14)
       and emits deep-link tuning recommendations backed by backtest simulation.
    """

    REQUIRED_INDICATORS = (
        "close",
        "sma200",
        "ema20",
        "ema50",
        "rsi3",
        "rsi7",
        "rsi14",
        "adx",
        "atr",
        "macd",
        "macd_signal",
    )

    def __init__(
        self,
        zero_yield_threshold: int = 3,
        audit_history_path: Optional[Union[str, Path]] = None,
        max_persisted_audits: int = 50,
    ) -> None:
        self.zero_yield_threshold = zero_yield_threshold
        self._max_persisted_audits = max(1, int(max_persisted_audits))
        self.consecutive_zero_cycles: int = 0
        self.last_audit: Optional[ImprovementAudit] = None
        self.audit_history: List[ImprovementAudit] = []

        self._explicit_history_path = audit_history_path is not None
        if audit_history_path is not None:
            self.audit_history_path = Path(audit_history_path)
        elif "unittest" in sys.modules:
            self.audit_history_path = (
                Path(tempfile.gettempdir())
                / f"audit_history_{os.getpid()}_{id(self)}.jsonl"
            )
        else:
            self.audit_history_path = Path("data") / "audit_history.jsonl"

        if self._explicit_history_path:
            self._load_audit_history()

    def _load_audit_history(self) -> None:
        """Load bounded audit history from JSONL file if present, ignoring corrupt lines."""
        if not self.audit_history_path.exists():
            return
        loaded: List[ImprovementAudit] = []
        try:
            with open(self.audit_history_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            loaded.append(ImprovementAudit.from_dict(parsed))
                    except Exception:
                        logger.warning(
                            "Skipping malformed audit_history line in %s",
                            self.audit_history_path,
                        )
        except Exception as exc:
            logger.debug("Could not read audit history from %s: %s", self.audit_history_path, exc)
            return

        if loaded:
            self.audit_history = loaded[-self._max_persisted_audits :]
            self.last_audit = self.audit_history[-1]
            self.consecutive_zero_cycles = int(self.last_audit.consecutive_zero_cycles or 0)

    def _persist_audit_history(self) -> None:
        """Atomically persist bounded audit history to JSONL via atomic_write_text."""
        bounded = self.audit_history[-self._max_persisted_audits :]
        lines = "".join(json.dumps(item.to_dict(), sort_keys=True) + "\n" for item in bounded)
        try:
            atomic_write_text(self.audit_history_path, lines)
        except Exception as exc:
            logger.warning("Failed to persist audit history atomically: %s", exc)

    def record_cycle_yield(
        self,
        candidate_count: int,
        universe_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
        config_snapshot: Optional[ConfigSnapshot] = None,
        gate_breakdown: Optional[List[Dict[str, Any]]] = None,
    ) -> ImprovementAudit:
        """Record candidate yield of a universe scan cycle and run directional diagnostics."""
        now_str = datetime.now(timezone.utc).isoformat()
        source: Any = config_snapshot if config_snapshot is not None else settings
        epoch_id = read_config_value(config_snapshot, "config_epoch_id", None) if config_snapshot is not None else None
        active_strategy = str(read_config_value(source, "ACTIVE_STRATEGY", "vwap"))
        current_min_adx = float(read_config_value(source, "MIN_ADX_TREND", 18.0))
        current_max_cat = float(read_config_value(source, "SCREENER_MAX_CATALYST_RISK", 0.25))

        if candidate_count > 0:
            self.consecutive_zero_cycles = 0
        else:
            self.consecutive_zero_cycles += 1

        missing_fields: set[str] = set()
        null_count: int = 0
        total_assets = len(universe_metrics or {})
        adx_vals: List[float] = []
        vol_vals: List[float] = []

        if universe_metrics:
            for ticker, m in universe_metrics.items():
                if not isinstance(m, dict):
                    continue
                for req in self.REQUIRED_INDICATORS:
                    val = m.get(req)
                    if val is None:
                        missing_fields.add(req)
                        null_count += 1
                adx = _safe_float(m.get("adx"), 0.0) or 0.0
                vol = _safe_float(m.get("realized_vol"), 0.0) or 0.0
                if adx > 0:
                    adx_vals.append(adx)
                if vol > 0:
                    vol_vals.append(vol)

        mean_adx = sum(adx_vals) / len(adx_vals) if adx_vals else 0.0
        mean_vol = sum(vol_vals) / len(vol_vals) if vol_vals else 0.20

        # Market regime classification
        if mean_vol > 0.35:
            regime = "CRISIS"
        elif mean_vol < 0.14 and mean_adx < 20.0:
            regime = "LOW_VOL_CHOP"
        elif mean_adx >= 28.0:
            regime = "HIGH_VOL_TREND"
        else:
            regime = "NORMAL"

        recommendations: List[str] = []
        parameter_recommendations: List[Dict[str, Any]] = []
        action_taken: Optional[str] = None

        if missing_fields:
            health = "CRITICAL"
            recommendations.append(
                f"Missing indicators detected in metrics feed: {sorted(list(missing_fields))}. "
                "Ensure compute_indicators() derives these values."
            )
            logger.critical(
                "StrategyImprovementAgent: Missing required indicators: %s",
                sorted(list(missing_fields)),
            )
        elif self.consecutive_zero_cycles >= self.zero_yield_threshold:
            health = "STARVED"
            recommendations.append(
                f"Zero signals generated across {self.consecutive_zero_cycles} consecutive {active_strategy} scan cycles. "
                f"Market regime is {regime} (mean ADX={mean_adx:.1f}, mean vol={mean_vol:.1%})."
            )

            suggested_adx = round(max(12.0, min(current_min_adx - 2.0, 16.0 if current_min_adx > 16.0 else current_min_adx - 1.0)), 1)
            bt_result = self.verify_parameter_adaptation(
                candidate_adx_threshold=suggested_adx,
                candidate_max_catalyst_risk=current_max_cat,
                config_snapshot=config_snapshot,
            )
            verified = bool(bt_result.get("verified"))
            recommendations.append(
                f"Calibrate {active_strategy} ADX floor via MIN_ADX_TREND={suggested_adx:.1f} "
                f"(currently {current_min_adx:.1f}, deep-link #/config?key=MIN_ADX_TREND); "
                f"verify SCREENER_MAX_CATALYST_RISK={current_max_cat:.2f} (#/config?key=SCREENER_MAX_CATALYST_RISK). "
                f"Backtest check: {bt_result.get('candidate_win_rate', 0.0):.1f}% win-rate (PF: {bt_result.get('profit_factor', 0.0):.2f})."
            )
            parameter_recommendations.append(
                {
                    "param_key": "MIN_ADX_TREND",
                    "label": "Minimum ADX Trend",
                    "current_value": current_min_adx,
                    "suggested_value": suggested_adx,
                    "gate_id": "adx_trend_gate",
                    "deep_link": "#/config?key=MIN_ADX_TREND",
                    "verified": verified,
                    "rationale": f"Lower MIN_ADX_TREND toward {suggested_adx:.1f} under {regime} (mean ADX={mean_adx:.1f}).",
                }
            )
            if gate_breakdown:
                for entry in gate_breakdown[:3]:
                    keys = entry.get("param_keys") or []
                    for key in keys:
                        if key == "MIN_ADX_TREND":
                            continue
                        cur_val = read_config_value(source, key, None)
                        if cur_val is None and config_snapshot is not None:
                            dyn = read_config_value(config_snapshot, "dynamic_parameters", {}) or {}
                            if isinstance(dyn, Mapping):
                                cur_val = dyn.get(key)
                        parameter_recommendations.append(
                            {
                                "param_key": str(key),
                                "label": str(entry.get("label") or key),
                                "current_value": cur_val,
                                "suggested_value": cur_val,
                                "gate_id": str(entry.get("gate_id") or ""),
                                "deep_link": f"#/config?key={key}",
                                "verified": False,
                                "rationale": (
                                    f"Gate '{entry.get('label', key)}' blocked {entry.get('count', 0)} candidate(s)."
                                ),
                            }
                        )
            logger.warning(
                "StrategyImprovementAgent: Pipeline starved (%d zero cycles). Recommendations: %s",
                self.consecutive_zero_cycles, recommendations,
            )
        elif self.consecutive_zero_cycles > 0:
            health = "DEGRADED"
        else:
            health = "HEALTHY"

        audit = ImprovementAudit(
            timestamp=now_str,
            consecutive_zero_cycles=self.consecutive_zero_cycles,
            health_status=health,
            missing_indicators=sorted(list(missing_fields)),
            null_metrics_count=null_count,
            total_assets_audited=total_assets,
            mean_adx=round(mean_adx, 2),
            mean_realized_vol=round(mean_vol, 4),
            regime=regime,
            recommendations=recommendations,
            action_taken=action_taken,
            active_strategy=active_strategy,
            config_epoch_id=epoch_id,
            parameter_recommendations=parameter_recommendations,
        )

        self.last_audit = audit
        self.audit_history.append(audit)
        if len(self.audit_history) > self._max_persisted_audits:
            self.audit_history = self.audit_history[-self._max_persisted_audits :]

        self._persist_audit_history()
        return audit

    def verify_parameter_adaptation(
        self,
        symbol: str = "QQQ",
        candidate_candles: Optional[List[Dict[str, Any]]] = None,
        candidate_rsi3_threshold: float = 25.0,
        candidate_adx_threshold: Optional[float] = None,
        candidate_max_catalyst_risk: Optional[float] = None,
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> Dict[str, Any]:
        """Run Wilder ADX(14) + directional_trend backtest & candidate check for adaptation verification."""
        _ = candidate_rsi3_threshold  # Retained harmlessly for backward signature compatibility
        try:
            from trading_system.backtest.engine import BacktestEngine
            from trading_system.data.candle_db import CandleDatabase
            from trading_system.strategies.registry import StrategyRegistry

            source: Any = config_snapshot if config_snapshot is not None else settings
            active_strat = str(read_config_value(source, "ACTIVE_STRATEGY", "vwap"))
            try:
                strategy = StrategyRegistry.get(active_strat)
            except Exception:
                strategy = StrategyRegistry.get("vwap")
            min_adx = float(
                candidate_adx_threshold
                if candidate_adx_threshold is not None
                else read_config_value(source, "MIN_ADX_TREND", 18.0)
            )
            max_cat_risk = float(
                candidate_max_catalyst_risk
                if candidate_max_catalyst_risk is not None
                else read_config_value(source, "SCREENER_MAX_CATALYST_RISK", 0.25)
            )

            candles = candidate_candles
            if not candles:
                db = CandleDatabase()
                candles = db.get_candles(symbol, timeframe="1d", limit=100)

            if not candles or len(candles) < 30:
                return {
                    "verified": False,
                    "symbol": symbol,
                    "strategy": active_strat,
                    "reason": "Insufficient historical candles for backtesting",
                    "candidate_win_rate": 0.0,
                    "profit_factor": 0.0,
                }

            highs = [float(c["high"]) for c in candles]
            lows = [float(c["low"]) for c in candles]
            closes = [float(c["close"]) for c in candles]
            volumes = [float(c.get("volume", 1.0) or 1.0) for c in candles]
            sma200 = BacktestEngine._calculate_sma(closes, 200)
            ema20 = BacktestEngine._calculate_ema(closes, 20)
            ema50 = BacktestEngine._calculate_ema(closes, 50)
            adx14, pdi14, mdi14 = BacktestEngine._calculate_wilder_adx(highs, lows, closes, 14)
            vwap20 = BacktestEngine._calculate_rolling_vwap(highs, lows, closes, volumes, 20)
            atr14 = BacktestEngine._calculate_atr(highs, lows, closes, 14)
            vol20 = BacktestEngine._calculate_realized_vol(closes, 20)

            eval_snapshot = _SnapshotProxy(
                source,
                MIN_ADX_TREND=min_adx,
                SCREENER_MAX_CATALYST_RISK=max_cat_risk,
            )
            if active_strat == "vwap":
                from trading_system.strategies.vwap import VWAPStrategy
                strategy = VWAPStrategy(max_catalyst_risk=max_cat_risk)
            else:
                try:
                    from trading_system.strategies.directional_trend import DirectionalTrendStrategy
                    strategy = DirectionalTrendStrategy(
                        min_adx_trend=min_adx, max_catalyst_risk=max_cat_risk
                    )
                except Exception:
                    strategy = StrategyRegistry.get(active_strat)

            actionable_bars = 0
            latest_dir = "NEUTRAL"
            for idx in range(20, len(candles)):
                curr_c = closes[idx]
                curr_vwap = vwap20[idx] if (vwap20 and vwap20[idx] is not None) else curr_c
                vwap_dist = ((curr_c - curr_vwap) / curr_vwap) * 100.0 if curr_vwap > 0 else 0.0
                e20 = ema20[idx] if ema20[idx] is not None else curr_c
                e50 = ema50[idx] if ema50[idx] is not None else curr_c
                slope_5d = (
                    ((ema20[idx] - ema20[idx - 5]) / ema20[idx - 5]) * 100.0
                    if idx >= 5 and ema20[idx] is not None and ema20[idx - 5] is not None and ema20[idx - 5] > 0
                    else None
                )
                bar_metrics = {
                    "close": curr_c,
                    "vwap": curr_vwap,
                    "vwap_dist_pct": vwap_dist,
                    "sma200": sma200[idx] if sma200[idx] is not None else 0.0,
                    "ema20": e20,
                    "ema50": e50,
                    "ema20_daily": e20,
                    "ema50_daily": e50,
                    "ema20_slope_5d": slope_5d,
                    "adx": adx14[idx] if adx14[idx] is not None else 0.0,
                    "plus_di": pdi14[idx] if pdi14[idx] is not None else 0.0,
                    "minus_di": mdi14[idx] if mdi14[idx] is not None else 0.0,
                    "atr": atr14[idx] or (curr_c * 0.02),
                    "realized_vol": vol20[idx] or 0.25,
                }
                sig = strategy.evaluate_candidate(symbol, bar_metrics, config_snapshot=eval_snapshot)
                if sig.is_actionable and sig.catalyst_risk <= max_cat_risk:
                    actionable_bars += 1
                if idx == len(candles) - 1:
                    latest_dir = sig.direction

            engine = BacktestEngine(leverage=3)
            result = engine.run(
                symbol,
                candles,
                strategy_filter=active_strat,
                config_snapshot=eval_snapshot,
            )

            win_rate = result.win_rate_pct
            pf = result.profit_factor
            verified = win_rate >= 70.0 and (pf >= 1.2 or result.total_trades == 0)
            latest_adx = adx14[-1] if adx14 and adx14[-1] is not None else 0.0

            return {
                "verified": verified,
                "symbol": symbol,
                "strategy": active_strat,
                "adx_threshold": min_adx,
                "max_catalyst_risk": max_cat_risk,
                "latest_adx": round(float(latest_adx), 2),
                "actionable_bars": actionable_bars,
                "latest_signal_direction": latest_dir,
                "total_trades": result.total_trades,
                "candidate_win_rate": round(win_rate, 2),
                "profit_factor": round(pf, 2),
                "total_return_pct": round(result.total_return_pct, 2),
                "max_drawdown_pct": round(result.max_drawdown_pct, 2),
            }
        except Exception as exc:
            logger.debug("Backtest verification failed: %s", exc)
            return {
                "verified": False,
                "symbol": symbol,
                "strategy": active_strat,
                "reason": str(exc),
                "candidate_win_rate": 0.0,
                "profit_factor": 0.0,
            }
