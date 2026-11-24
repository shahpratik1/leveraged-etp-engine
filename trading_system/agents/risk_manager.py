"""Risk Manager Agent for Institutional-Grade Execution.

Enforces strict position sizing, ATR-based Stop Loss, multi-target Take Profits
(TP1, TP2, TP3), and dynamic Trailing Stop Loss (TSL) for both Long and Short positions.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

from trading_system.strategies.base import read_config_value as _read_cfg, safe_float as _safe_float

if TYPE_CHECKING:
    from trading_system.config_bus import ConfigSnapshot

logger = logging.getLogger(__name__)

TradeDecision = Literal["BUY_ENTRY", "SELL_ENTRY", "DO_NOT_TRADE", "EXIT"]


def _append_card_rejection(
    card: "DecisionCard",
    gate_id: str,
    param_keys: List[str],
    observed: Any,
    threshold: Any,
    label: str,
) -> None:
    """Record structured gate rejection metadata on DecisionCard."""
    if card.gate_rejections is None:
        card.gate_rejections = []
    if any(r.get("gate_id") == gate_id for r in card.gate_rejections):
        return
    card.gate_rejections.append(
        {
            "gate_id": gate_id,
            "param_keys": list(param_keys),
            "observed": observed,
            "threshold": threshold,
            "label": label,
            "ticker": card.ticker,
        }
    )


@dataclass
class DecisionCard:
    """Dataclass representing a comprehensive trading decision."""
    ticker: str                         # Underlying or reference ticker (e.g., 'QQQ')
    trading212_ticker: str              # Execution ticker (e.g. 'QQQ3l_EQ' or 'QQQSl_EQ')
    direction: str                      # 'LONG' or 'SHORT'
    decision: str                       # 'BUY_ENTRY', 'SELL_ENTRY', 'DO_NOT_TRADE', 'EXIT'
    entry_price: float
    stop_loss: float
    trailing_stop_loss: float           # Current dynamic trailing stop
    stop_distance_pct: float            # Percentage distance from entry to SL
    take_profit_1: float                # TP1 at +1.0R
    take_profit_2: float                # TP2 at +2.0R
    take_profit_3: float                # TP3 at +3.0R
    position_size_shares: float         # Number of shares to execute
    position_notional: float            # Dollar value of position
    risk_amount: float                  # Dollar amount at risk
    risk_pct_of_equity: float           # Actual risk as % of equity
    conviction_score: float
    rationale: str                      # Brief text explaining the decision
    strategy_type: str = "vwap"
    leverage: int = 3                   # 3 or 5 (ETP leverage multiplier)
    earnings_in_days: Optional[float] = None
    realized_vol: Optional[float] = None
    is_synthetic: bool = False          # Flag indicating whether underlying data was synthetically generated
    timestamp: str = ""                 # ISO format timestamp
    config_epoch_id: Optional[str] = None
    gate_rejections: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def position_size(self) -> float:
        """Alias for position_size_shares."""
        return self.position_size_shares

    @property
    def tp1(self) -> float:
        """Alias for take_profit_1."""
        return self.take_profit_1

    @property
    def tp2(self) -> float:
        """Alias for take_profit_2."""
        return self.take_profit_2

    @property
    def tp3(self) -> float:
        """Alias for take_profit_3."""
        return self.take_profit_3


class RiskManager:
    """Risk Manager agent that enforces position sizing and calculates entry/exit levels."""

    def __init__(
        self,
        max_risk_per_trade: Optional[float] = None,
        max_position_pct: Optional[float] = None,
        atr_sl_multiplier: Optional[float] = None,
        min_sl_pct: Optional[float] = None,              # 0.5% minimum stop distance
        max_sl_pct: Optional[float] = None,              # 3.0% maximum stop distance cap
        tp1_r: Optional[float] = None,
        tp2_r: Optional[float] = None,
        tp3_r: Optional[float] = None,
        tsl_atr_multiplier: Optional[float] = None,
        tsl_hysteresis_atr_mult: Optional[float] = None,
        tp1_trim_pct: Optional[float] = None,
        tp2_trim_pct: Optional[float] = None,
        tp3_trim_pct: Optional[float] = None,
        target_annual_volatility: Optional[float] = None,
        enforce_spread_gate: Optional[bool] = None,
        max_bid_ask_spread_pct: Optional[float] = None,
        enforce_correlation_limits: Optional[bool] = None,
        max_sector_positions: Optional[int] = None,
        max_portfolio_beta_heat: Optional[float] = None,
        enforce_daily_circuit_breaker: Optional[bool] = None,
        daily_max_drawdown_pct: Optional[float] = None,
    ):
        from trading_system.config import settings

        self._explicit_args: set[str] = set()
        raw_inputs = {
            "max_risk_per_trade": max_risk_per_trade,
            "max_position_pct": max_position_pct,
            "atr_sl_multiplier": atr_sl_multiplier,
            "min_sl_pct": min_sl_pct,
            "max_sl_pct": max_sl_pct,
            "tp1_r": tp1_r,
            "tp2_r": tp2_r,
            "tp3_r": tp3_r,
            "tsl_atr_multiplier": tsl_atr_multiplier,
            "tsl_hysteresis_atr_mult": tsl_hysteresis_atr_mult,
            "tp1_trim_pct": tp1_trim_pct,
            "tp2_trim_pct": tp2_trim_pct,
            "tp3_trim_pct": tp3_trim_pct,
            "target_annual_volatility": target_annual_volatility,
            "enforce_spread_gate": enforce_spread_gate,
            "max_bid_ask_spread_pct": max_bid_ask_spread_pct,
            "enforce_correlation_limits": enforce_correlation_limits,
            "max_sector_positions": max_sector_positions,
            "max_portfolio_beta_heat": max_portfolio_beta_heat,
            "enforce_daily_circuit_breaker": enforce_daily_circuit_breaker,
            "daily_max_drawdown_pct": daily_max_drawdown_pct,
        }
        for name, val in raw_inputs.items():
            if val is not None:
                self._explicit_args.add(name)

        self.max_risk_per_trade = getattr(settings, "MAX_RISK_PER_TRADE", 0.015) if max_risk_per_trade is None else max_risk_per_trade
        self.max_position_pct = getattr(settings, "MAX_POSITION_PCT", 0.20) if max_position_pct is None else max_position_pct
        self.atr_sl_multiplier = getattr(settings, "ATR_SL_MULTIPLIER", 1.5) if atr_sl_multiplier is None else atr_sl_multiplier
        self.min_sl_pct = getattr(settings, "MIN_SL_PCT", 0.005) if min_sl_pct is None else min_sl_pct
        self.max_sl_pct = getattr(settings, "MAX_SL_PCT", 0.03) if max_sl_pct is None else max_sl_pct
        self.tp1_r = getattr(settings, "TP1_R_MULTIPLE", 1.5) if tp1_r is None else tp1_r
        self.tp2_r = getattr(settings, "TP2_R_MULTIPLE", 2.5) if tp2_r is None else tp2_r
        self.tp3_r = getattr(settings, "TP3_R_MULTIPLE", 3.5) if tp3_r is None else tp3_r
        self.tsl_atr_multiplier = getattr(settings, "TSL_ATR_MULTIPLIER", 1.8) if tsl_atr_multiplier is None else tsl_atr_multiplier
        self.tsl_hysteresis_atr_mult = getattr(settings, "TSL_HYSTERESIS_ATR_MULT", 0.5) if tsl_hysteresis_atr_mult is None else tsl_hysteresis_atr_mult
        self.tp1_trim_pct = getattr(settings, "TP1_TRIM_PCT", 0.50) if tp1_trim_pct is None else tp1_trim_pct
        self.tp2_trim_pct = getattr(settings, "TP2_TRIM_PCT", 0.25) if tp2_trim_pct is None else tp2_trim_pct
        self.tp3_trim_pct = getattr(settings, "TP3_TRIM_PCT", 0.25) if tp3_trim_pct is None else tp3_trim_pct
        self.target_annual_volatility = getattr(settings, "TARGET_ANNUAL_VOLATILITY", 0.20) if target_annual_volatility is None else target_annual_volatility

        # Advanced Risk & Safety Gates
        self.enforce_spread_gate = getattr(settings, "ENFORCE_SPREAD_GATE", True) if enforce_spread_gate is None else enforce_spread_gate
        self.max_bid_ask_spread_pct = getattr(settings, "MAX_BID_ASK_SPREAD_PCT", 0.0150) if max_bid_ask_spread_pct is None else max_bid_ask_spread_pct
        self.enforce_correlation_limits = getattr(settings, "ENFORCE_CORRELATION_LIMITS", True) if enforce_correlation_limits is None else enforce_correlation_limits
        self.max_sector_positions = getattr(settings, "MAX_SECTOR_POSITIONS", 4) if max_sector_positions is None else max_sector_positions
        self.max_portfolio_beta_heat = getattr(settings, "MAX_PORTFOLIO_BETA_HEAT", 6.0) if max_portfolio_beta_heat is None else max_portfolio_beta_heat
        self.enforce_daily_circuit_breaker = getattr(settings, "ENFORCE_DAILY_CIRCUIT_BREAKER", True) if enforce_daily_circuit_breaker is None else enforce_daily_circuit_breaker
        self.daily_max_drawdown_pct = getattr(settings, "DAILY_MAX_DRAWDOWN_PCT", 0.02) if daily_max_drawdown_pct is None else daily_max_drawdown_pct
        self.trade_loss_cooldown_seconds = getattr(settings, "TRADE_LOSS_COOLDOWN_SECONDS", 3600.0)
        self.max_spread_to_stop_ratio = getattr(settings, "MAX_SPREAD_TO_STOP_RATIO", 0.50)

        self._init_snapshot_values: Dict[str, Any] = {
            attr: getattr(self, attr) for attr in raw_inputs
        }
        self._init_snapshot_values["trade_loss_cooldown_seconds"] = self.trade_loss_cooldown_seconds
        self._init_snapshot_values["max_spread_to_stop_ratio"] = self.max_spread_to_stop_ratio

        self.loss_cooldowns: Dict[str, float] = {}
        self.is_circuit_breaker_tripped: bool = False
        self.session_start_equity: Optional[float] = None
        self.current_drawdown_pct: float = 0.0

        logger.debug("Initialized RiskManager with max_risk_per_trade=%s", max_risk_per_trade)

    def _resolve_field(
        self,
        attr: str,
        setting_key: str,
        default: Any,
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> Any:
        """Resolve a risk setting from snapshot first, preserving explicit legacy overrides."""
        if config_snapshot is not None:
            return _read_cfg(config_snapshot, setting_key, default)
        current_val = getattr(self, attr, default)
        if attr in self._explicit_args or current_val != self._init_snapshot_values.get(attr, current_val):
            return current_val
        from trading_system.config import settings

        return getattr(settings, setting_key, current_val)

    def _resolve_params(
        self, config_snapshot: Optional[ConfigSnapshot] = None
    ) -> Dict[str, Any]:
        """Resolve all risk sizing, exit, and safety thresholds for the active epoch."""
        from trading_system.config import settings

        source = config_snapshot if config_snapshot is not None else settings
        return {
            "config_epoch_id": _read_cfg(config_snapshot, "config_epoch_id", None) if config_snapshot is not None else None,
            "max_risk_per_trade": float(self._resolve_field("max_risk_per_trade", "MAX_RISK_PER_TRADE", 0.015, config_snapshot)),
            "max_position_pct": float(self._resolve_field("max_position_pct", "MAX_POSITION_PCT", 0.20, config_snapshot)),
            "atr_sl_multiplier": float(self._resolve_field("atr_sl_multiplier", "ATR_SL_MULTIPLIER", 1.5, config_snapshot)),
            "min_sl_pct": float(self._resolve_field("min_sl_pct", "MIN_SL_PCT", 0.005, config_snapshot)),
            "max_sl_pct": float(self._resolve_field("max_sl_pct", "MAX_SL_PCT", 0.03, config_snapshot)),
            "etp_min_sl_pct": float(_read_cfg(source, "ETP_MIN_SL_PCT", 0.025)),
            "etp_max_sl_pct": float(_read_cfg(source, "ETP_MAX_SL_PCT", 0.080)),
            "tp1_r": float(self._resolve_field("tp1_r", "TP1_R_MULTIPLE", 1.5, config_snapshot)),
            "tp2_r": float(self._resolve_field("tp2_r", "TP2_R_MULTIPLE", 2.5, config_snapshot)),
            "tp3_r": float(self._resolve_field("tp3_r", "TP3_R_MULTIPLE", 3.5, config_snapshot)),
            "tsl_atr_multiplier": float(self._resolve_field("tsl_atr_multiplier", "TSL_ATR_MULTIPLIER", 1.2, config_snapshot)),
            "tsl_hysteresis_atr_mult": float(self._resolve_field("tsl_hysteresis_atr_mult", "TSL_HYSTERESIS_ATR_MULT", 0.5, config_snapshot)),
            "target_annual_volatility": float(self._resolve_field("target_annual_volatility", "TARGET_ANNUAL_VOLATILITY", 0.20, config_snapshot)),
            "enforce_spread_gate": bool(self._resolve_field("enforce_spread_gate", "ENFORCE_SPREAD_GATE", True, config_snapshot)),
            "max_bid_ask_spread_pct": float(self._resolve_field("max_bid_ask_spread_pct", "MAX_BID_ASK_SPREAD_PCT", 0.0150, config_snapshot)),
            "max_spread_to_stop_ratio": float(self._resolve_field("max_spread_to_stop_ratio", "MAX_SPREAD_TO_STOP_RATIO", 0.50, config_snapshot)),
            "enforce_correlation_limits": bool(self._resolve_field("enforce_correlation_limits", "ENFORCE_CORRELATION_LIMITS", True, config_snapshot)),
            "max_sector_positions": int(self._resolve_field("max_sector_positions", "MAX_SECTOR_POSITIONS", 4, config_snapshot)),
            "max_portfolio_beta_heat": float(self._resolve_field("max_portfolio_beta_heat", "MAX_PORTFOLIO_BETA_HEAT", 6.0, config_snapshot)),
            "enforce_daily_circuit_breaker": bool(self._resolve_field("enforce_daily_circuit_breaker", "ENFORCE_DAILY_CIRCUIT_BREAKER", True, config_snapshot)),
            "daily_max_drawdown_pct": float(self._resolve_field("daily_max_drawdown_pct", "DAILY_MAX_DRAWDOWN_PCT", 0.02, config_snapshot)),
            "trade_loss_cooldown_seconds": float(self._resolve_field("trade_loss_cooldown_seconds", "TRADE_LOSS_COOLDOWN_SECONDS", 3600.0, config_snapshot)),
            "enable_5x_leverage": bool(_read_cfg(source, "ENABLE_5X_LEVERAGE", False)),
            "prefer_5x_leverage": bool(_read_cfg(source, "PREFER_5X_LEVERAGE", False)),
            "use_leveraged_3x": bool(_read_cfg(source, "USE_LEVERAGED_3X", True)),
            "max_leverage": int(_read_cfg(source, "MAX_LEVERAGE", 3)),
            "pure_tsl_mode": bool(_read_cfg(source, "PURE_TSL_MODE", True)),
        }

    def get_etp_sl_bounds(
        self, config_snapshot: Optional[ConfigSnapshot] = None
    ) -> Tuple[float, float]:
        """Return resolved (ETP_MIN_SL_PCT, ETP_MAX_SL_PCT) stop clamp bounds."""
        params = self._resolve_params(config_snapshot)
        return params["etp_min_sl_pct"], params["etp_max_sl_pct"]

    def check_circuit_breaker(
        self,
        current_equity: float,
        session_start_equity: Optional[float] = None,
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> Tuple[bool, str]:
        """Check if portfolio daily drawdown exceeds circuit breaker threshold.

        Returns:
            Tuple[bool, str]: (is_tripped, explanation)
        """
        params = self._resolve_params(config_snapshot)
        if not params["enforce_daily_circuit_breaker"]:
            return False, "Circuit breaker disabled"

        if session_start_equity is not None and session_start_equity > 0:
            self.session_start_equity = session_start_equity
        elif self.session_start_equity is None or self.session_start_equity <= 0:
            self.session_start_equity = current_equity
            return False, "Session baseline established"

        if self.session_start_equity <= 0:
            return False, "Invalid session baseline"

        drawdown = (current_equity - self.session_start_equity) / self.session_start_equity
        self.current_drawdown_pct = drawdown
        max_dd = params["daily_max_drawdown_pct"]

        if drawdown <= -max_dd:
            self.is_circuit_breaker_tripped = True
            msg = (
                f"🚨 CIRCUIT BREAKER TRIPPED: Daily drawdown {drawdown*100:.2f}% "
                f"breached -{max_dd*100:.1f}% limit! "
                f"Equity: ${current_equity:.2f} vs Start: ${self.session_start_equity:.2f}."
            )
            logger.critical(msg)
            return True, msg

        return False, f"Drawdown {drawdown*100:.2f}% within safe threshold (-{max_dd*100:.1f}%)"

    def reset_circuit_breaker(self, new_session_equity: Optional[float] = None) -> None:
        """Reset the circuit breaker for a new session day."""
        self.is_circuit_breaker_tripped = False
        self.current_drawdown_pct = 0.0
        if new_session_equity and new_session_equity > 0:
            self.session_start_equity = new_session_equity

    def register_loss_exit(
        self,
        ticker: str,
        underlying: Optional[str] = None,
        cooldown_seconds: Optional[float] = None,
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> None:
        """Place a ticker and its underlying on anti-churn cooldown after a stop loss exit."""
        params = self._resolve_params(config_snapshot)
        duration = cooldown_seconds if cooldown_seconds is not None else params["trade_loss_cooldown_seconds"]
        expiry = time.time() + duration
        t_clean = (ticker or "").upper().strip()
        u_clean = (underlying or "").upper().strip()
        if t_clean:
            self.loss_cooldowns[t_clean] = expiry
        if u_clean:
            self.loss_cooldowns[u_clean] = expiry
        logger.info(
            "Placed %s (underlying %s) on post-loss anti-churn cooldown for %.0fs (expires in %.0f min)",
            t_clean, u_clean or "N/A", duration, duration / 60.0
        )

    def is_in_loss_cooldown(self, ticker: str, underlying: Optional[str] = None) -> Tuple[bool, float]:
        """Check if ticker or underlying is currently under anti-churn loss cooldown."""
        now = time.time()
        for sym in ((ticker or "").upper().strip(), (underlying or "").upper().strip()):
            if sym and sym in self.loss_cooldowns:
                exp = self.loss_cooldowns[sym]
                if now < exp:
                    return True, exp - now
                else:
                    self.loss_cooldowns.pop(sym, None)
        return False, 0.0

    @staticmethod
    def is_true_short_contract(ticker: str = "", direction: str = "") -> bool:
        """Return True only if position is a margin/CFD short where falling price is profit.

        For all Trading 212 Invest assets (including 3x and 5x inverse ETPs),
        shares are held long, so rising ETP price is profit and Stop Loss must be BELOW live price.
        """
        dir_u = str(direction or "").upper()
        if dir_u in ("LONG", "BUY"):
            return False
        t_u = str(ticker or "").upper()
        if (
            t_u.endswith("_EQ")
            or t_u.endswith(".L")
            or "_ETP" in t_u
            or dir_u.endswith("_ETP")
            or t_u.startswith("3")
            or t_u.startswith("5")
        ):
            return False
        return dir_u in ("SHORT", "SELL")

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        direction: str = "LONG",
        ticker: str = "",
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> float:
        """Calculate stop loss based on ATR, bounded between min_sl_pct and max_sl_pct.

        For LONG:  SL = Entry - clamp(multiplier * ATR, Entry * min_sl_pct, Entry * max_sl_pct)
        For SHORT (if CFD): SL = Entry + clamp(multiplier * ATR, Entry * min_sl_pct, Entry * max_sl_pct)
        Note: When buying an Inverse 3x/5x ETP, treat the ETP as a Long position with respect to its own price.
        """
        params = self._resolve_params(config_snapshot)
        raw_sl_distance = max(
            params["atr_sl_multiplier"] * atr,
            entry_price * params["min_sl_pct"],
        )
        max_distance = entry_price * params["max_sl_pct"]
        sl_distance = min(raw_sl_distance, max_distance)
        is_cfd_short = self.is_true_short_contract(ticker=ticker, direction=direction)
        if is_cfd_short:
            stop_loss = entry_price + sl_distance
        else:
            stop_loss = max(0.01, entry_price - sl_distance)
        return round(stop_loss, 2)

    def calculate_take_profits(
        self,
        entry_price: float,
        stop_loss: float,
        direction: str = "LONG",
        ticker: str = "",
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> Tuple[float, float, float]:
        """Calculate multiple take profit levels based on Risk (R).

        R = abs(entry_price - stop_loss)
        For LONG:  TP = Entry + (multiplier * R)
        For SHORT (if CFD): TP = Entry - (multiplier * R)
        """
        params = self._resolve_params(config_snapshot)
        r_distance = abs(entry_price - stop_loss)
        is_cfd_short = self.is_true_short_contract(ticker=ticker, direction=direction)
        if is_cfd_short:
            tp1 = max(0.01, entry_price - (params["tp1_r"] * r_distance))
            tp2 = max(0.01, entry_price - (params["tp2_r"] * r_distance))
            tp3 = max(0.01, entry_price - (params["tp3_r"] * r_distance))
        else:
            tp1 = entry_price + (params["tp1_r"] * r_distance)
            tp2 = entry_price + (params["tp2_r"] * r_distance)
            tp3 = entry_price + (params["tp3_r"] * r_distance)
        return round(tp1, 2), round(tp2, 2), round(tp3, 2)

    def calculate_escape_threshold(
        self,
        tp1: float,
        atr: float,
        direction: str = "LONG",
        multiplier: Optional[float] = None,
        ticker: str = "",
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> float:
        """Calculate the escape threshold for Schmitt Hysteresis past TP1.

        Requires price to advance beyond TP1 by (multiplier * ATR) before
        elevating the Stop Loss to TP1, eliminating zero-distance stop liquidations.
        """
        params = self._resolve_params(config_snapshot)
        mult = multiplier if multiplier is not None else params["tsl_hysteresis_atr_mult"]
        buffer = mult * atr if atr > 0 else tp1 * 0.015
        is_cfd_short = self.is_true_short_contract(ticker=ticker, direction=direction)
        if is_cfd_short:
            escape_price = max(0.01, tp1 - buffer)
        else:
            escape_price = tp1 + buffer
        return round(escape_price, 2)

    @classmethod
    def get_decay_atr_multiplier(
        cls,
        session_count: int = 1,
        is_runner: bool = False,
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> float:
        """Calculate the active ATR trailing stop multiplier with session-day decay."""
        if config_snapshot is not None:
            source: Any = config_snapshot
        else:
            from trading_system.config import settings
            source = settings

        runner_mult = float(_read_cfg(source, "TSL_DECAY_RUNNER_MULT", 2.00))
        if is_runner:
            return round(runner_mult, 2)

        day1_mult = float(_read_cfg(source, "TSL_DECAY_DAY1_MULT", 2.20))
        step_mult = float(_read_cfg(source, "TSL_DECAY_STEP_MULT", 0.35))
        floor_mult = float(_read_cfg(source, "TSL_DECAY_FLOOR_MULT", 1.15))
        decayed = day1_mult - (step_mult * max(0, session_count - 1))
        return round(max(floor_mult, decayed), 2)

    def calculate_trailing_stop(
        self,
        current_price: float,
        peak_price: float,
        atr: float,
        current_tsl: float,
        direction: str = "LONG",
        ticker: str = "",
        session_count: int = 1,
        is_runner: bool = False,
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> float:
        """Calculate dynamic trailing stop level with session-day decay envelope.

        The trailing stop ratchets ONLY in the favorable direction (never widens).
        """
        effective_mult = self.get_decay_atr_multiplier(
            session_count=session_count,
            is_runner=is_runner,
            config_snapshot=config_snapshot,
        )
        min_floor_pct = (
            float(_read_cfg(config_snapshot, "ETP_MIN_SL_PCT", 0.025))
            if config_snapshot is not None
            else 0.02
        )
        distance = max(effective_mult * atr, peak_price * min_floor_pct)
        is_cfd_short = self.is_true_short_contract(ticker=ticker, direction=direction)
        if is_cfd_short:
            new_tsl = peak_price + distance
            # For short, lower is better; ratchet down
            return round(min(current_tsl, new_tsl) if current_tsl > 0 else new_tsl, 2)
        else:
            new_tsl = peak_price - distance
            # For long, higher is better; ratchet up
            return round(max(current_tsl, new_tsl), 2)

    @classmethod
    def calculate_progressive_trailing_stop(
        cls,
        entry_price: float,
        current_price: float,
        current_sl: float,
        direction: str = "LONG",
        ticker: str = "",
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> float:
        """Lock in gains progressively:
        - At +1.5% unrealized profit: move stop loss to Entry Price (Breakeven).
        - At +3.0% unrealized profit: move stop loss to Entry Price + 1.5% (lock profit cushion).
        - Stop never loosens.
        """
        _ = config_snapshot
        is_short = cls.is_true_short_contract(ticker=ticker, direction=direction)
        if is_short:
            gain_pct = ((entry_price - current_price) / entry_price * 100.0) if entry_price > 0 else 0.0
            if gain_pct >= 3.0:
                candidate = entry_price * (1.0 - 0.015)
            elif gain_pct >= 1.5:
                candidate = entry_price
            else:
                candidate = current_sl
            return round(min(current_sl, candidate) if current_sl > 0 else candidate, 4)
        else:
            gain_pct = ((current_price - entry_price) / entry_price * 100.0) if entry_price > 0 else 0.0
            if gain_pct >= 3.0:
                candidate = entry_price * 1.015
            elif gain_pct >= 1.5:
                candidate = entry_price
            else:
                candidate = current_sl
            return round(max(current_sl, candidate), 4)

    def calculate_volatility_targeted_size(
        self,
        account_equity: float,
        annualized_vol: float,
        target_vol: Optional[float] = None,
        leverage: int = 3,
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> float:
        """Calculate dynamic volatility-targeted position notional."""
        params = self._resolve_params(config_snapshot)
        tgt_vol = target_vol if target_vol is not None else params["target_annual_volatility"]
        max_position_notional = account_equity * params["max_position_pct"]
        if annualized_vol <= 0 or account_equity <= 0:
            return max_position_notional

        effective_vol = max(0.10, leverage * annualized_vol)
        vol_budget = max_position_notional * (tgt_vol / effective_vol)
        return min(max_position_notional, round(vol_budget, 2))

    def calculate_position_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss: float,
        realized_vol: Optional[float] = None,
        leverage: int = 3,
        ticker: Optional[str] = None,
        currency: Optional[str] = None,
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> Tuple[float, float, float]:
        """Calculate number of shares, notional position, and actual risk amount."""
        if entry_price <= 0 or account_equity <= 0:
            return 0.0, 0.0, 0.0

        params = self._resolve_params(config_snapshot)

        from trading_system.data.leveraged_universe import LeveragedUniverse, GBX_TICKERS
        price_in_eur = LeveragedUniverse.normalize_quote_to_eur(ticker or "", entry_price, currency)
        if price_in_eur <= 0:
            price_in_eur = entry_price

        stop_distance_pct = abs(entry_price - stop_loss) / entry_price if entry_price > 0 else 0.0
        risk_amount = account_equity * params["max_risk_per_trade"]

        if stop_distance_pct <= 0:
            position_notional_from_risk = 0.0
        else:
            position_notional_from_risk = risk_amount / stop_distance_pct

        max_position_notional = account_equity * params["max_position_pct"]
        position_notional = min(position_notional_from_risk, max_position_notional)

        # Apply volatility-targeted sizing if realized_vol is available
        if realized_vol is not None and realized_vol > 0.05:
            vol_notional = self.calculate_volatility_targeted_size(
                account_equity=account_equity,
                annualized_vol=realized_vol,
                leverage=leverage,
                config_snapshot=config_snapshot,
            )
            position_notional = min(position_notional, vol_notional)

        raw_shares = position_notional / price_in_eur if price_in_eur > 0 else 0.0
        max_shares = max_position_notional / price_in_eur if price_in_eur > 0 else 0.0
        if raw_shares > max_shares:
            raw_shares = max_shares

        is_gbx_or_lse = (
            (currency and currency.upper() in ("GBX", "GBP"))
            or (ticker and (ticker in GBX_TICKERS or ticker.endswith("l_EQ") or ticker.endswith(".L")))
            or (entry_price > 500.0)
            or (abs(entry_price - 93.97) < 1.0)
        )
        if is_gbx_or_lse and raw_shares >= 1.0:
            shares = float(math.floor(raw_shares))
            if shares <= 0:
                shares = 1.0
        elif raw_shares >= 1.0:
            shares = round(raw_shares, 2)
            if shares * price_in_eur > max_position_notional + 1e-4:
                shares = math.floor(raw_shares * 100.0) / 100.0
            if shares * price_in_eur > max_position_notional + 1e-4:
                shares = float(math.floor(raw_shares))
        else:
            shares = max(1.0, round(raw_shares, 2)) if position_notional > 0 else 0.0

        actual_notional = round(shares * price_in_eur, 2)
        if actual_notional > max_position_notional + 0.01 and shares > 1.0:
            shares = float(math.floor(max_position_notional / price_in_eur))
            actual_notional = round(shares * price_in_eur, 2)

        eur_scale = price_in_eur / entry_price if entry_price > 0 else 1.0
        actual_risk = round(shares * abs(entry_price - stop_loss) * eur_scale, 2)

        return shares, actual_notional, actual_risk

    def generate_decision_card(
        self,
        ticker: str,
        trading212_ticker: str,
        direction: str,
        metrics: Dict[str, Any],
        debate_outcome: Optional[Any] = None,
        account_equity: float = 0.0,
        strategy_type: str = "vwap",
        leverage: Optional[int] = None,
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> DecisionCard:
        """Generate a complete DecisionCard from technical metrics and quantitative signal sizing."""
        params = self._resolve_params(config_snapshot)
        epoch_id: Optional[str] = params["config_epoch_id"]

        atr = _safe_float(metrics.get("atr"), 0.0) or 0.0

        # Multi-Tier Allocation: Support both 3x and 5x dynamically based on conviction and volatility
        from trading_system.data.leveraged_universe import LeveragedUniverse

        universe = LeveragedUniverse()
        has_5x = universe.has_5x_leverage(ticker, direction)
        enable_5x = params["enable_5x_leverage"]
        prefer_5x = params["prefer_5x_leverage"]
        use_3x = params["use_leveraged_3x"]
        max_lev = params["max_leverage"]

        if debate_outcome is not None:
            conviction_score = _safe_float(getattr(debate_outcome, "conviction_score", 0.0), 0.0) or 0.0
        else:
            conviction_score = _safe_float(metrics.get("confidence") or metrics.get("overall_confidence"), 0.85) or 0.85
        rvol_raw = metrics.get("realized_vol") if metrics.get("realized_vol") is not None else metrics.get("historical_vol")
        realized_vol = _safe_float(rvol_raw, 0.25) or 0.25

        if has_5x and enable_5x and max_lev >= 5:
            if leverage == 5:
                resolved_leverage = 5
            elif leverage == 3:
                resolved_leverage = 3
            elif prefer_5x or (conviction_score >= 0.85 and realized_vol <= 0.35):
                resolved_leverage = 5
            else:
                resolved_leverage = 3
        else:
            resolved_leverage = 3 if (use_3x and max_lev >= 3) else 1

        sanitized_t212_ticker: Optional[str] = None
        if enable_5x and resolved_leverage >= 5:
            pair = universe.get_pair_for_underlying(ticker)
            if pair and pair.has_5x:
                dir_u = direction.upper()
                if dir_u in ("LONG", "BUY", "BULLISH") and pair.long_5x_ticker:
                    sanitized_t212_ticker = pair.long_5x_ticker
                elif dir_u in ("SHORT", "SELL", "BEARISH") and pair.short_5x_ticker and pair.short_5x_ticker != "None":
                    sanitized_t212_ticker = pair.short_5x_ticker
        if not sanitized_t212_ticker:
            sanitized_t212_ticker = (
                universe.get_execution_ticker(ticker, direction, leverage=resolved_leverage)
                or trading212_ticker
            )

        # Query live quote and currency of execution ticker before sizing
        from trading_system.data.trading212_client import MarketDataProvider
        etp_quote = MarketDataProvider.get_quote(sanitized_t212_ticker)
        etp_price = _safe_float(etp_quote.get("price") if etp_quote else None, 0.0) or 0.0
        etp_currency = (etp_quote.get("currency") if etp_quote else "") or universe.get_ticker_currency(sanitized_t212_ticker)

        underlying_price = _safe_float(metrics.get("close"), 0.0) or 0.0
        is_derivative_etp = sanitized_t212_ticker.upper() != ticker.upper()
        if etp_price > 0:
            actual_entry_price = etp_price
            if underlying_price > 0 and atr > 0:
                atr_ratio = atr / underlying_price
                etp_atr = actual_entry_price * atr_ratio
            else:
                etp_atr = atr if atr > 0 else (actual_entry_price * 0.02)
        elif is_derivative_etp:
            # Scale guard: Derivative/inverse ETP quote is missing; never fall back to underlying price
            # as it creates catastrophic scale mismatches (e.g. ETH-USD $2310 vs ETH3EUR €5.70).
            logger.warning(
                "Scale guard veto: Quote unavailable for derivative ETP %s (underlying %s). Rejecting underlying price fallback.",
                sanitized_t212_ticker, ticker
            )
            actual_entry_price = 0.0
            etp_atr = 0.0
        else:
            actual_entry_price = underlying_price
            etp_atr = atr

        if actual_entry_price <= 0.0:
            return DecisionCard(
                ticker=ticker,
                trading212_ticker=sanitized_t212_ticker,
                direction=direction.upper(),
                decision="DO_NOT_TRADE",
                entry_price=0.0,
                stop_loss=0.0,
                trailing_stop_loss=0.0,
                stop_distance_pct=0.0,
                take_profit_1=0.0,
                take_profit_2=0.0,
                take_profit_3=0.0,
                position_size_shares=0.0,
                position_notional=0.0,
                risk_amount=0.0,
                risk_pct_of_equity=0.0,
                conviction_score=0.0,
                rationale="Scale guard veto: Invalid or unresolvable ETP execution price.",
                strategy_type=strategy_type,
                leverage=resolved_leverage,
                config_epoch_id=epoch_id,
                gate_rejections=[
                    {
                        "gate_id": "scale_guard_veto",
                        "param_keys": [],
                        "observed": actual_entry_price,
                        "threshold": "> 0",
                        "label": "Execution Quote Scale Guard",
                        "ticker": ticker,
                    }
                ],
            )

        stop_loss = self.calculate_stop_loss(
            actual_entry_price,
            etp_atr,
            direction=direction,
            ticker=sanitized_t212_ticker,
            config_snapshot=config_snapshot,
        )
        stop_distance_pct = (
            abs(actual_entry_price - stop_loss) / actual_entry_price if actual_entry_price > 0 else 0.0
        )
        tp1, tp2, tp3 = self.calculate_take_profits(
            actual_entry_price,
            stop_loss,
            direction=direction,
            ticker=sanitized_t212_ticker,
            config_snapshot=config_snapshot,
        )

        if debate_outcome is None:
            consensus = "BUY_ENTRY" if direction.upper() in ("LONG", "BUY", "BULLISH") else "SELL_ENTRY"
            conviction_score = _safe_float(metrics.get("confidence") or metrics.get("overall_confidence"), 0.85) or 0.85
            synthesis = f"Quantitative {strategy_type.upper()} signal verified for {ticker}."
            winning_side = "bull" if direction.upper() in ("LONG", "BUY", "BULLISH") else "bear"
        else:
            consensus = getattr(debate_outcome, "consensus_decision", "DO_NOT_TRADE")
            conviction_score = _safe_float(getattr(debate_outcome, "conviction_score", 0.0), 0.0) or 0.0
            synthesis = getattr(debate_outcome, "synthesis", "No synthesis provided.")
            winning_side = str(getattr(debate_outcome, "winning_side", "") or "").lower()

        dir_upper = direction.upper()
        is_long = dir_upper in ("LONG", "BUY", "BULLISH")
        is_short = dir_upper in ("SHORT", "SELL", "BEARISH")

        # Directional alignment gate: If debate specifically concluded the opposite direction won, veto entry!
        if isinstance(winning_side, str) and winning_side:
            direction_mismatch = (is_short and winning_side == "bull") or (is_long and winning_side == "bear")
        else:
            direction_mismatch = False

        card_rejections: List[Dict[str, Any]] = []
        if consensus in ("DO_NOT_TRADE", "EXIT") or direction_mismatch:
            if direction_mismatch:
                logger.warning(
                    "Directional conflict veto for %s: candidate requested %s but debate winning side was %s (consensus=%s). Setting DO_NOT_TRADE.",
                    ticker, direction, winning_side, consensus
                )
                card_rejections.append(
                    {
                        "gate_id": "directional_conflict_veto",
                        "param_keys": [],
                        "observed": f"{direction}/{winning_side}",
                        "threshold": direction,
                        "label": "Debate Directional Alignment",
                        "ticker": ticker,
                    }
                )
            else:
                card_rejections.append(
                    {
                        "gate_id": "debate_consensus_veto",
                        "param_keys": [],
                        "observed": consensus,
                        "threshold": "BUY_ENTRY/SELL_ENTRY",
                        "label": "Debate Consensus Gate",
                        "ticker": ticker,
                    }
                )
            shares, position_notional, actual_risk, risk_pct_of_equity = 0.0, 0.0, 0.0, 0.0
            decision_action = "DO_NOT_TRADE"
        else:
            shares, position_notional, actual_risk = self.calculate_position_size(
                account_equity=account_equity,
                entry_price=actual_entry_price,
                stop_loss=stop_loss,
                realized_vol=realized_vol,
                leverage=resolved_leverage,
                ticker=sanitized_t212_ticker,
                currency=etp_currency,
                config_snapshot=config_snapshot,
            )
            risk_pct_of_equity = (
                actual_risk / account_equity if account_equity > 0 else 0.0
            )
            decision_action = "BUY_ENTRY" if is_long else "SELL_ENTRY"

        return DecisionCard(
            ticker=ticker,
            trading212_ticker=sanitized_t212_ticker,
            direction=direction.upper(),
            decision=decision_action,
            entry_price=actual_entry_price,
            stop_loss=stop_loss,
            trailing_stop_loss=stop_loss,
            stop_distance_pct=stop_distance_pct,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            position_size_shares=shares,
            position_notional=position_notional,
            risk_amount=actual_risk,
            risk_pct_of_equity=risk_pct_of_equity,
            conviction_score=conviction_score,
            rationale=synthesis,
            strategy_type=strategy_type,
            leverage=resolved_leverage,
            earnings_in_days=_safe_float(metrics.get("earnings_in_days"), None),
            realized_vol=realized_vol,
            is_synthetic=MarketDataProvider.is_ticker_synthetic(ticker),
            timestamp=datetime.now(timezone.utc).isoformat(),
            config_epoch_id=epoch_id,
            gate_rejections=card_rejections,
        )

    calculate_decision_card = generate_decision_card

    def validate_trade(
        self,
        card: DecisionCard,
        account_cash: float,
        existing_positions: List[Any],
        spread_pct: Optional[float] = None,
        account_equity: Optional[float] = None,
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> Tuple[bool, str]:
        """Validate whether a trade should be executed."""
        params = self._resolve_params(config_snapshot)
        if config_snapshot is not None and card.config_epoch_id is None:
            card.config_epoch_id = params["config_epoch_id"]

        if card.decision not in ("BUY_ENTRY", "SELL_ENTRY"):
            _append_card_rejection(
                card,
                "decision_not_entry",
                [],
                card.decision,
                "BUY_ENTRY/SELL_ENTRY",
                "Entry Decision Required",
            )
            return False, f"Decision is {card.decision}, not an entry decision"

        # 1. Anti-Churn Loss Cooldown Gate
        in_cooldown, remaining_sec = self.is_in_loss_cooldown(card.trading212_ticker, card.ticker)
        if in_cooldown:
            _append_card_rejection(
                card,
                "cooldown_gate",
                ["TRADE_LOSS_COOLDOWN_SECONDS"],
                round(remaining_sec, 1),
                0.0,
                "Anti-Churn Loss Cooldown",
            )
            return (
                False,
                f"Anti-churn cooldown active for {card.trading212_ticker} ({card.ticker}) after recent stop-loss exit ({remaining_sec:.0f}s remaining)",
            )

        # 2. Daily Drawdown Circuit Breaker Gate
        if params["enforce_daily_circuit_breaker"] and self.is_circuit_breaker_tripped:
            _append_card_rejection(
                card,
                "circuit_breaker_gate",
                ["ENFORCE_DAILY_CIRCUIT_BREAKER", "DAILY_MAX_DRAWDOWN_PCT"],
                round(self.current_drawdown_pct, 4),
                -params["daily_max_drawdown_pct"],
                "Daily Drawdown Circuit Breaker",
            )
            return (
                False,
                f"Daily circuit breaker tripped (Drawdown {self.current_drawdown_pct*100:.2f}% breached -{params['daily_max_drawdown_pct']*100:.1f}%)",
            )

        # 3. Bid-Ask Spread Gate & Spread-to-Stop Ratio Gate
        if params["enforce_spread_gate"] and spread_pct is not None:
            max_spread = params["max_bid_ask_spread_pct"]
            max_ratio = params["max_spread_to_stop_ratio"]
            if spread_pct > max_spread:
                _append_card_rejection(
                    card,
                    "spread_gate",
                    ["ENFORCE_SPREAD_GATE", "MAX_BID_ASK_SPREAD_PCT"],
                    round(spread_pct, 6),
                    max_spread,
                    "Max Bid-Ask Spread Gate",
                )
                return (
                    False,
                    f"Bid-ask spread too wide ({spread_pct*100:.2f}% > {max_spread*100:.2f}%)",
                )
            if card.stop_distance_pct > 0 and spread_pct >= (card.stop_distance_pct * max_ratio):
                _append_card_rejection(
                    card,
                    "spread_gate",
                    ["ENFORCE_SPREAD_GATE", "MAX_SPREAD_TO_STOP_RATIO"],
                    round(spread_pct, 6),
                    round(card.stop_distance_pct * max_ratio, 6),
                    "Max Spread-to-Stop Ratio Gate",
                )
                return (
                    False,
                    f"Spread ({spread_pct*100:.2f}%) exceeds {max_ratio*100:.0f}% of planned stop loss ({card.stop_distance_pct*100:.2f}%), trade would be vulnerable to immediate spread-stop whipsaw",
                )

        max_allowed_leverage = params["max_leverage"]
        if card.leverage > max_allowed_leverage or (card.leverage >= 5 and not params["enable_5x_leverage"]):
            cap = min(max_allowed_leverage, 3 if not params["enable_5x_leverage"] else max_allowed_leverage)
            _append_card_rejection(
                card,
                "leverage_cap_gate",
                ["MAX_LEVERAGE", "ENABLE_5X_LEVERAGE"],
                card.leverage,
                cap,
                "Max Leverage Gate",
            )
            return False, f"Leverage {card.leverage}x exceeds maximum allowed cap of {cap}x"

        if card.position_notional > account_cash:
            _append_card_rejection(
                card,
                "insufficient_capital_gate",
                ["MAX_POSITION_PCT"],
                round(card.position_notional, 2),
                round(account_cash, 2),
                "Free Cash Capital Check",
            )
            return False, f"Insufficient free capital (${account_cash:.2f} < ${card.position_notional:.2f})"

        max_risk_per_trade = params["max_risk_per_trade"]
        if card.risk_pct_of_equity > max_risk_per_trade * 3.0:
            _append_card_rejection(
                card,
                "risk_ceiling_gate",
                ["MAX_RISK_PER_TRADE"],
                round(card.risk_pct_of_equity, 6),
                round(max_risk_per_trade * 3.0, 6),
                "Max Risk Tolerance Gate",
            )
            return False, "Risk exceeds maximum allowed by more than tolerance"

        if card.stop_distance_pct < params["min_sl_pct"]:
            _append_card_rejection(
                card,
                "min_sl_gate",
                ["MIN_SL_PCT"],
                round(card.stop_distance_pct, 6),
                params["min_sl_pct"],
                "Min Stop Loss Distance",
            )
            return False, "Stop distance percentage is below minimum allowed"

        # Check existing positions to avoid duplicate exposure
        for pos in existing_positions:
            pos_ticker = getattr(pos, "ticker", None)
            if pos_ticker == card.trading212_ticker or pos_ticker == card.ticker:
                _append_card_rejection(
                    card,
                    "duplicate_position_gate",
                    [],
                    card.trading212_ticker,
                    "unique",
                    "Duplicate Position Gate",
                )
                return False, f"Already holding a position in {card.trading212_ticker}"

        # 3. Sector Concentration & Aggregate Beta Heat Limits
        if params["enforce_correlation_limits"]:
            from trading_system.data.leveraged_universe import LeveragedUniverse
            univ = LeveragedUniverse()
            candidate_sector = univ.get_category_for_symbol(card.ticker)

            sector_count = 0
            total_active_beta_notional = 0.0

            if existing_positions:
                for pos in existing_positions:
                    pos_ticker = getattr(pos, "ticker", "") or ""
                    pos_underlying = getattr(pos, "underlying", "") or pos_ticker
                    pos_sector = univ.get_category_for_symbol(pos_underlying or pos_ticker)
                    if pos_sector == candidate_sector:
                        sector_count += 1

                    pos_qty = float(getattr(pos, "remaining_quantity", getattr(pos, "quantity", 0.0)) or 0.0)
                    pos_price = float(getattr(pos, "entry_price", getattr(pos, "current_price", getattr(pos, "average_price", 0.0))) or 0.0)
                    pos_price_eur = univ.normalize_quote_to_eur(pos_ticker, pos_price, getattr(pos, "currency", None))
                    pos_notional = pos_qty * (pos_price_eur if pos_price_eur > 0 else pos_price)
                    pos_lev = 3 if any(k in pos_ticker for k in ("3l_EQ", "3Sl_EQ", "3L", "3S", "3US", "QQQ3", "3NVD", "3LTS", "AAP3", "MSF3")) else 1
                    total_active_beta_notional += (pos_notional * pos_lev)

                max_sector = params["max_sector_positions"]
                if sector_count >= max_sector:
                    _append_card_rejection(
                        card,
                        "correlation_gate",
                        ["ENFORCE_CORRELATION_LIMITS", "MAX_SECTOR_POSITIONS"],
                        sector_count,
                        max_sector,
                        "Sector Concentration Gate",
                    )
                    return False, f"Sector concentration limit reached for {candidate_sector} ({sector_count} >= {max_sector})"

            effective_equity = account_equity if (account_equity and account_equity > 0) else (account_cash + total_active_beta_notional / 3.0)
            if effective_equity > 0:
                candidate_beta_notional = card.position_notional * card.leverage
                new_total_beta_heat = (total_active_beta_notional + candidate_beta_notional) / effective_equity
                max_heat = params["max_portfolio_beta_heat"]
                if new_total_beta_heat > max_heat:
                    _append_card_rejection(
                        card,
                        "correlation_gate",
                        ["ENFORCE_CORRELATION_LIMITS", "MAX_PORTFOLIO_BETA_HEAT"],
                        round(new_total_beta_heat, 4),
                        max_heat,
                        "Portfolio Beta Heat Gate",
                    )
                    return (
                        False,
                        f"Aggregate portfolio beta heat ceiling exceeded ({new_total_beta_heat:.2f}x > {max_heat:.2f}x)",
                    )

        return True, "All checks passed"

    def format_decision_card(
        self,
        card: DecisionCard,
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> str:
        """Format the DecisionCard into a readable string block."""
        params = self._resolve_params(config_snapshot)
        pure_tsl = params["pure_tsl_mode"]
        exit_line = (
            "Exit Protocol:   100% Pure Trailing Stop Loss (+1.5% BE, +3.0% Cushion, Dynamic ATR Trail)"
            if pure_tsl
            else f"Take Profits:    TP1=${card.take_profit_1:.2f} | TP2=${card.take_profit_2:.2f} | TP3=${card.take_profit_3:.2f}"
        )
        lines = [
            f"=== DECISION CARD: {card.ticker} -> {card.trading212_ticker} ===",
            f"Timestamp:       {card.timestamp}",
            f"Direction:       {card.direction}",
            f"Decision:        {card.decision}",
            f"Conviction:      {card.conviction_score}/10",
            f"Entry Price:     ${card.entry_price:.2f}",
            f"Stop Loss:       ${card.stop_loss:.2f} ({card.stop_distance_pct*100:.2f}%)",
            f"Trailing Stop:   ${card.trailing_stop_loss:.2f}",
            exit_line,
            f"Position Size:   {card.position_size_shares:.3f} shares",
            f"Notional Value:  ${card.position_notional:.2f}",
            f"Risk Amount:     ${card.risk_amount:.2f} ({card.risk_pct_of_equity*100:.2f}% of equity)",
            f"Rationale:       {card.rationale}",
            "==================================================",
        ]
        return "\n".join(lines)
