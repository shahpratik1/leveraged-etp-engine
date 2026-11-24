"""Schema registry, built-in presets, allowlisted metrics, and dynamic parameter validation.

Provides the authoritative metadata contract for all tunable engine parameters,
explicitly excluding credentials, broker mode flags, API URLs, ports, and file paths.
"""

from __future__ import annotations

import copy
import dataclasses
import keyword
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple, Union

from trading_system.config import Settings, collect_settings_invariant_errors


RELOAD_MODES: Tuple[str, ...] = ("hot_reload", "restart_required")
REVALIDATION_SCOPES: Tuple[str, ...] = ("abort_stale_scan", "resize_entry")
ALLOWED_GATE_OPERATORS: Tuple[str, ...] = ("<", "<=", ">", ">=", "==", "!=")
ALLOWED_ON_MISSING_MODES: Tuple[str, ...] = ("reject", "skip_gate")
MAX_DYNAMIC_PARAMETERS: int = 64
MAX_DYNAMIC_STRING_LENGTH: int = 256
DYNAMIC_PARAM_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{1,63}$")
_TIME_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")

_FORBIDDEN_EVAL_KEYS: FrozenSet[str] = frozenset({
    "eval",
    "exec",
    "expr",
    "expression",
    "code",
    "lambda",
    "formula",
    "script",
})

# Environment-only secrets, URLs, broker mode, ports, and file paths.
# Strictly excluded from CONFIG_SCHEMA, ConfigSnapshot, and HTTP overrides.
ENV_ONLY_KEYS: FrozenSet[str] = frozenset({
    "TRADING212_API_KEY",
    "TRADING212_API_SECRET",
    "TRADING212_BASE_URL",
    "TRADING212_LIVE_URL",
    "USE_LIVE",
    "OPENAI_API_KEY",
    "FMP_API_KEY",
    "TIINGO_API_KEY",
    "POLYGON_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "FINNHUB_API_KEY",
    "OLOSTEP_API_KEY",
    "NIXTLA_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "DISCORD_WEBHOOK_URL",
    "DASHBOARD_AUTH_TOKEN",
    "PANIC_FLATTEN_TOKEN",
    "NOTIFICATIONS_LOG_PATH",
    "ACTIVE_TRADES_PATH",
    "ACTIVE_TRADES_FILE",
    "CANDLE_DB_PATH",
    "TRADES_LOG_PATH",
    "ENABLE_EMBEDDED_DASHBOARD",
    "DASHBOARD_PORT",
})

_ENV_ONLY_LOWER: FrozenSet[str] = frozenset(k.lower() for k in ENV_ONLY_KEYS)
_SECRET_KEY_SUFFIXES: Tuple[str, ...] = (
    "_API_KEY",
    "_API_SECRET",
    "_TOKEN",
    "_WEBHOOK_URL",
    "_PATH",
    "_URL",
    "_PORT",
)

# Snapshot metadata and reserved attributes that custom parameters cannot collide with.
SNAPSHOT_METADATA_KEYS: FrozenSet[str] = frozenset({
    "config_epoch_id",
    "generation",
    "committed_at",
    "committed_by",
    "commit_reason",
    "pending_restart_keys",
    "config_quarantine_warnings",
    "dynamic_parameters",
    "dynamic_parameter_specs",
    "api_base_url",
    "values",
    "get",
    "to_dict",
    "as_dict",
    "from_dict",
    "validate",
})
_SNAPSHOT_METADATA_LOWER: FrozenSet[str] = frozenset(k.lower() for k in SNAPSHOT_METADATA_KEYS)

# Scalar numeric indicator keys produced by MarketDataProvider.compute_indicators()
# and consumed by DirectionalTrendStrategy / LayaScreener.
ALLOWLISTED_SCREENER_METRICS: FrozenSet[str] = frozenset({
    "close",
    "sma200",
    "sma200_dist",
    "sma5",
    "ema20",
    "ema50",
    "ema20_daily",
    "ema50_daily",
    "ema20_slope_5d",
    "rsi",
    "rsi3",
    "rsi7",
    "rsi14",
    "adx",
    "plus_di",
    "minus_di",
    "atr",
    "atr_20d_sma",
    "atr_spike_ratio",
    "atr_ratio",
    "donchian_high_20",
    "donchian_low_20",
    "donchian_mid_20",
    "bb_upper",
    "bb_lower",
    "bb_mid",
    "bb_bandwidth",
    "bb_pct_b",
    "kc_upper",
    "kc_lower",
    "kc_mid",
    "macd",
    "macd_signal",
    "macd_hist",
    "realized_vol",
    "high_20d",
    "earnings_in_days",
    "vwap",
    "vwap_dist_pct",
    "vwap_upper_1std",
    "vwap_lower_1std",
})

# Strategy / screener groups that can be deterministically simulated in Area 1 preview.
SIMULATABLE_GROUPS: FrozenSet[str] = frozenset({"strategy", "screener"})


def is_finite_number(value: Any) -> bool:
    """Return True iff *value* is a finite int/float (and not a bool)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def is_env_only_key(key: str) -> bool:
    """Return True if *key* is env-only, a credential, URL, port, or file path."""
    if not isinstance(key, str):
        return False
    upper = key.strip().upper()
    if upper == "ML_MODEL_PATH":
        return False
    if key.strip().lower() in _ENV_ONLY_LOWER or upper in ENV_ONLY_KEYS:
        return True
    return any(upper.endswith(suffix) for suffix in _SECRET_KEY_SUFFIXES)


def _check_string_length(value: str, field_name: str, max_len: int = MAX_DYNAMIC_STRING_LENGTH) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")
    if len(value) > max_len:
        raise ValueError(f"{field_name} exceeds maximum length of {max_len} characters (got {len(value)})")


@dataclass(frozen=True)
class ConfigParamSchema:
    """Schema descriptor for a single tunable Settings parameter."""

    key: str
    label: str
    type: str
    group: str
    unit: str
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    allowed_values: Optional[Tuple[Any, ...]] = None
    reload_mode: str = "hot_reload"
    revalidation_scope: str = "resize_entry"
    impact_hint: str = ""
    gate_tags: Tuple[str, ...] = ()
    nullable: bool = False

    def __post_init__(self) -> None:
        if self.reload_mode not in RELOAD_MODES:
            raise ValueError(f"{self.key}: invalid reload_mode {self.reload_mode!r}")
        if self.revalidation_scope not in REVALIDATION_SCOPES:
            raise ValueError(f"{self.key}: invalid revalidation_scope {self.revalidation_scope!r}")
        if is_env_only_key(self.key):
            raise ValueError(f"{self.key} is env-only and cannot appear in CONFIG_SCHEMA")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize schema metadata into a JSON-compatible dictionary."""
        return {
            "key": self.key,
            "label": self.label,
            "type": self.type,
            "group": self.group,
            "unit": self.unit,
            "default": list(self.default) if isinstance(self.default, tuple) else self.default,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "allowed_values": list(self.allowed_values) if self.allowed_values is not None else None,
            "reload_mode": self.reload_mode,
            "revalidation_scope": self.revalidation_scope,
            "impact_hint": self.impact_hint,
            "gate_tags": list(self.gate_tags),
            "nullable": self.nullable,
            "is_dynamic": False,
        }


_S = ConfigParamSchema

CONFIG_SCHEMA: Tuple[ConfigParamSchema, ...] = (
    # ── Strategy & Trend Gates ───────────────────────────────────────────────
    _S(
        key="ACTIVE_STRATEGY",
        label="Active Strategy",
        type="str",
        group="strategy",
        unit="",
        default="vwap",
        allowed_values=("vwap", "ml_alpha"),
        reload_mode="restart_required",
        revalidation_scope="abort_stale_scan",
        impact_hint="Selects the registered BaseStrategy implementation used across screening and backtests.",
        gate_tags=("strategy_evaluation",),
    ),
    _S(
        key="MIN_VWAP_DIST_PCT",
        label="Min VWAP Distance",
        type="float",
        group="strategy",
        unit="%",
        default=0.1,
        min=0.0,
        max=5.0,
        step=0.05,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Minimum distance above VWAP (for Longs) or below VWAP (for Shorts) to avoid dead-zone noise.",
        gate_tags=("vwap_gate", "screener_filter"),
    ),
    _S(
        key="MAX_VWAP_DIST_PCT",
        label="Max VWAP Distance",
        type="float",
        group="strategy",
        unit="%",
        default=3.5,
        min=1.0,
        max=10.0,
        step=0.1,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Maximum distance above VWAP before entry is rejected for exhaustion/overextension risk.",
        gate_tags=("vwap_gate", "screener_filter"),
    ),
    _S(
        key="MIN_ADX_TREND",
        label="Minimum ADX Trend",
        type="float",
        group="strategy",
        unit="ADX",
        default=18.0,
        min=5.0,
        max=60.0,
        step=0.5,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Minimum ADX(14) required for directional trend confirmation and screener passage.",
        gate_tags=("adx_trend_gate", "dmi_trend_gate", "screener_filter"),
    ),
    _S(
        key="STRATEGY_EXCEPTIONAL_ADX",
        label="Exceptional Setup ADX",
        type="float",
        group="strategy",
        unit="ADX",
        default=25.0,
        min=10.0,
        max=70.0,
        step=0.5,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="ADX threshold promoting setup_quality from 'favorable' to 'exceptional'.",
        gate_tags=("setup_quality_tier", "setup_score_gate"),
    ),
    _S(
        key="MIN_EMA20_SLOPE_PCT",
        label="Min 20-EMA 5d Slope",
        type="float",
        group="strategy",
        unit="%",
        default=0.0,
        min=-5.0,
        max=5.0,
        step=0.05,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Minimum 5-day 20-EMA slope threshold (> for LONG, < -threshold for SHORT).",
        gate_tags=("ema_slope_gate",),
    ),
    _S(
        key="ENTRY_EARNINGS_BLACKOUT_DAYS",
        label="Entry Earnings Blackout",
        type="float",
        group="strategy",
        unit="days",
        default=2.0,
        min=0.0,
        max=14.0,
        step=0.5,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Days ahead of earnings that raise catalyst_risk and block new directional entries.",
        gate_tags=("earnings_blackout_gate", "catalyst_risk_gate"),
    ),
    _S(
        key="STRATEGY_BASE_CONVICTION",
        label="Long Base Conviction",
        type="float",
        group="strategy",
        unit="score",
        default=0.70,
        min=0.30,
        max=0.90,
        step=0.01,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Base directional conviction score before ADX and macro 200-SMA bonuses.",
        gate_tags=("conviction_scoring",),
    ),
    _S(
        key="STRATEGY_SHORT_BASE_CONVICTION",
        label="Short Base Conviction",
        type="float",
        group="strategy",
        unit="score",
        default=0.75,
        min=0.30,
        max=0.90,
        step=0.01,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Base directional conviction score for dual-track index short signals.",
        gate_tags=("conviction_scoring",),
    ),
    _S(
        key="STRATEGY_MACRO_TAILWIND_BONUS",
        label="200-SMA Tailwind Bonus",
        type="float",
        group="strategy",
        unit="score",
        default=0.10,
        min=0.0,
        max=0.30,
        step=0.01,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Conviction bonus added when price trades above Daily 200 SMA.",
        gate_tags=("conviction_scoring",),
    ),
    _S(
        key="DIRECTIONAL_MAX_CONVICTION",
        label="Max Strategy Conviction",
        type="float",
        group="strategy",
        unit="score",
        default=0.95,
        min=0.50,
        max=1.00,
        step=0.01,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Upper cap on strategy conviction score emitted in StrategySignal.",
        gate_tags=("conviction_scoring",),
    ),
    _S(
        key="MAX_SWING_HOLD_DAYS",
        label="Max Swing Hold Days",
        type="int",
        group="strategy",
        unit="days",
        default=10,
        min=1,
        max=60,
        step=1,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Maximum holding duration in trading days for directional swing positions.",
        gate_tags=("max_hold_exit",),
    ),
    _S(
        key="MR_MAX_HOLD_DAYS",
        label="Max Hold Days (Compat)",
        type="int",
        group="strategy",
        unit="days",
        default=10,
        min=1,
        max=60,
        step=1,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Backward-compatible hold duration limit passed to RiskManager and BacktestEngine.",
        gate_tags=("max_hold_exit",),
    ),

    # ── Screener & Regime Filters ────────────────────────────────────────────
    _S(
        key="SCREENER_MIN_SETUP_SCORE",
        label="Min Setup Quality",
        type="str",
        group="screener",
        unit="",
        default="favorable",
        allowed_values=("unfavorable", "mediocre", "favorable", "exceptional"),
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Minimum setup quality rank required by LayaScreener.passes_filter().",
        gate_tags=("setup_score_gate", "screener_filter"),
    ),
    _S(
        key="SCREENER_MAX_CATALYST_RISK",
        label="Max Catalyst Risk",
        type="float",
        group="screener",
        unit="prob",
        default=0.25,
        min=0.0,
        max=1.0,
        step=0.05,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Maximum acceptable binary catalyst risk probability before screening rejection.",
        gate_tags=("catalyst_risk_gate", "screener_filter"),
    ),
    _S(
        key="SCREENER_TOP_N",
        label="Screener Top-N Cap",
        type="int",
        group="screener",
        unit="assets",
        default=15,
        min=1,
        max=100,
        step=1,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Maximum number of top-ranked candidates promoted to System 2 debate.",
        gate_tags=("top_n_cap",),
    ),
    _S(
        key="SCREENER_MAX_PER_CATEGORY",
        label="Max Candidates Per Sector",
        type="int",
        group="screener",
        unit="assets",
        default=2,
        min=1,
        max=20,
        step=1,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="First-pass diversification cap per asset category before overflow fill.",
        gate_tags=("category_diversification_cap",),
    ),
    _S(
        key="REGIME_BREAKOUT_ADX_MIN",
        label="Breakout Regime Min ADX",
        type="float",
        group="screener",
        unit="ADX",
        default=25.0,
        min=10.0,
        max=60.0,
        step=0.5,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="ADX threshold classifying market regime as TRENDING_BREAKOUT.",
        gate_tags=("regime_classifier",),
    ),
    _S(
        key="HIGH_VOL_MAX_REALIZED_VOL",
        label="Erratic Realized Vol Cap",
        type="float",
        group="screener",
        unit="fraction",
        default=0.65,
        min=0.15,
        max=2.00,
        step=0.05,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Annualized volatility above which assets classify as HIGH_VOL_ERRATIC.",
        gate_tags=("high_vol_erratic_gate", "regime_classifier"),
    ),
    _S(
        key="HIGH_VOL_MAX_ATR_SPIKE_RATIO",
        label="Erratic ATR Spike Ratio",
        type="float",
        group="screener",
        unit="ratio",
        default=1.50,
        min=1.0,
        max=5.0,
        step=0.1,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="ATR/ATR_20d_SMA spike ratio triggering HIGH_VOL_ERRATIC capital preservation.",
        gate_tags=("high_vol_erratic_gate", "regime_classifier"),
    ),
    _S(
        key="FALLBACK_SCREENER_MIN_ADX",
        label="Fallback Screener Min ADX",
        type="float",
        group="screener",
        unit="ADX",
        default=20.0,
        min=5.0,
        max=60.0,
        step=0.5,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Minimum ADX gate enforced by TechnicalFallbackScreener when primary NLI yields 0.",
        gate_tags=("fallback_screener_adx_gate",),
    ),

    # ── Risk & Position Sizing ───────────────────────────────────────────────
    _S(
        key="DEFAULT_ACCOUNT_EQUITY",
        label="Fallback Account Equity",
        type="float",
        group="risk",
        unit="EUR",
        default=10000.0,
        min=100.0,
        max=10000000.0,
        step=100.0,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Fallback equity used only in dry-run or when broker cash lookup fails.",
        gate_tags=("equity_fallback",),
    ),
    _S(
        key="MAX_RISK_PER_TRADE",
        label="Max Risk Per Trade",
        type="float",
        group="risk",
        unit="fraction",
        default=0.015,
        min=0.001,
        max=1.0,
        step=0.001,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Maximum fraction of equity risked to initial stop loss per new entry.",
        gate_tags=("risk_sizing", "risk_tolerance_gate"),
    ),
    _S(
        key="MAX_POSITION_PCT",
        label="Max Position Notional Pct",
        type="float",
        group="risk",
        unit="fraction",
        default=0.20,
        min=0.01,
        max=1.0,
        step=0.01,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Hard cap on position notional as a fraction of total equity (must >= MAX_RISK_PER_TRADE).",
        gate_tags=("position_size_cap", "risk_sizing"),
    ),
    _S(
        key="TARGET_ANNUAL_VOLATILITY",
        label="Target Annual Volatility",
        type="float",
        group="risk",
        unit="fraction",
        default=0.20,
        min=0.05,
        max=1.0,
        step=0.01,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Target annualized portfolio volatility scaling leveraged notional size.",
        gate_tags=("volatility_sizing",),
    ),
    _S(
        key="ATR_SL_MULTIPLIER",
        label="Initial Stop ATR Multiplier",
        type="float",
        group="risk",
        unit="ATR",
        default=2.2,
        min=0.5,
        max=6.0,
        step=0.1,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="ATR multiplier determining initial stop-loss distance for new entries.",
        gate_tags=("stop_loss_sizing",),
    ),
    _S(
        key="MIN_SL_PCT",
        label="Underlying Min Stop Pct",
        type="float",
        group="risk",
        unit="fraction",
        default=0.005,
        min=0.001,
        max=0.20,
        step=0.001,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Minimum initial stop-loss distance floor for RiskManager.",
        gate_tags=("min_sl_gate", "stop_loss_sizing"),
    ),
    _S(
        key="MAX_SL_PCT",
        label="Underlying Max Stop Pct",
        type="float",
        group="risk",
        unit="fraction",
        default=0.03,
        min=0.005,
        max=0.50,
        step=0.005,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Maximum initial stop-loss distance cap for RiskManager (must > MIN_SL_PCT).",
        gate_tags=("max_sl_cap", "stop_loss_sizing"),
    ),
    _S(
        key="ETP_MIN_SL_PCT",
        label="ETP Min Stop Clamp",
        type="float",
        group="risk",
        unit="fraction",
        default=0.025,
        min=0.005,
        max=0.25,
        step=0.005,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Minimum stop-loss distance clamp applied to leveraged ETP exit levels.",
        gate_tags=("etp_sl_clamp",),
    ),
    _S(
        key="ETP_MAX_SL_PCT",
        label="ETP Max Stop Clamp",
        type="float",
        group="risk",
        unit="fraction",
        default=0.080,
        min=0.01,
        max=0.50,
        step=0.005,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Maximum stop-loss distance clamp applied to leveraged ETP exit levels.",
        gate_tags=("etp_sl_clamp",),
    ),
    _S(
        key="ENFORCE_SPREAD_GATE",
        label="Enforce Bid-Ask Spread Gate",
        type="bool",
        group="risk",
        unit="",
        default=True,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Blocks new orders when live bid-ask spread exceeds safety caps.",
        gate_tags=("spread_gate",),
    ),
    _S(
        key="MAX_BID_ASK_SPREAD_PCT",
        label="Max Bid-Ask Spread Pct",
        type="float",
        group="risk",
        unit="fraction",
        default=0.015,
        min=0.001,
        max=0.10,
        step=0.001,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Maximum allowed relative bid-ask spread for entry execution.",
        gate_tags=("spread_gate",),
    ),
    _S(
        key="MAX_SPREAD_TO_STOP_RATIO",
        label="Max Spread-to-Stop Ratio",
        type="float",
        group="risk",
        unit="ratio",
        default=0.50,
        min=0.05,
        max=0.95,
        step=0.05,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Blocks trades if bid-ask spread consumes more than this fraction of planned stop distance.",
        gate_tags=("spread_to_stop_gate",),
    ),
    _S(
        key="TRADE_LOSS_COOLDOWN_SECONDS",
        label="Post-Loss Cooldown",
        type="float",
        group="risk",
        unit="seconds",
        default=3600.0,
        min=0.0,
        max=86400.0,
        step=60.0,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Anti-churn re-entry lockout duration after a stop-loss exit.",
        gate_tags=("loss_cooldown_gate",),
    ),
    _S(
        key="MIN_HOLD_SECONDS",
        label="Minimum Hold Time",
        type="float",
        group="risk",
        unit="seconds",
        default=120.0,
        min=0.0,
        max=3600.0,
        step=10.0,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Minimum hold window before algorithmic soft exits trigger.",
        gate_tags=("min_hold_gate",),
    ),
    _S(
        key="ENFORCE_CORRELATION_LIMITS",
        label="Enforce Correlation Limits",
        type="bool",
        group="risk",
        unit="",
        default=True,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Enforces per-sector position concentration and aggregate portfolio beta heat limits.",
        gate_tags=("sector_concentration_gate", "portfolio_beta_heat_gate"),
    ),
    _S(
        key="MAX_SECTOR_POSITIONS",
        label="Max Sector Positions",
        type="int",
        group="risk",
        unit="positions",
        default=4,
        min=1,
        max=20,
        step=1,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Maximum concurrent open positions allowed in a single sector cluster.",
        gate_tags=("sector_concentration_gate",),
    ),
    _S(
        key="MAX_PORTFOLIO_BETA_HEAT",
        label="Max Portfolio Beta Heat",
        type="float",
        group="risk",
        unit="x",
        default=6.0,
        min=0.5,
        max=20.0,
        step=0.5,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Maximum aggregate leverage-weighted notional exposure relative to equity.",
        gate_tags=("portfolio_beta_heat_gate",),
    ),
    _S(
        key="ENFORCE_DAILY_CIRCUIT_BREAKER",
        label="Enforce Daily Circuit Breaker",
        type="bool",
        group="risk",
        unit="",
        default=True,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Halts new entries if intraday portfolio drawdown breaches DAILY_MAX_DRAWDOWN_PCT.",
        gate_tags=("circuit_breaker_gate",),
    ),
    _S(
        key="DAILY_MAX_DRAWDOWN_PCT",
        label="Daily Max Drawdown Pct",
        type="float",
        group="risk",
        unit="fraction",
        default=0.02,
        min=0.002,
        max=0.25,
        step=0.002,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Maximum intraday equity drawdown allowed before tripping the circuit breaker.",
        gate_tags=("circuit_breaker_gate",),
    ),

    # ── Exit, Stops & TSL Decay ──────────────────────────────────────────────
    _S(
        key="PURE_TSL_MODE",
        label="Pure TSL Mode (No Partial TP)",
        type="bool",
        group="exit_tsl",
        unit="",
        default=True,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Protects 100% of position size under broker GTC trailing stop (evaluated per position epoch).",
        gate_tags=("exit_protocol",),
    ),
    _S(
        key="TSL_ATR_MULTIPLIER",
        label="Base TSL ATR Multiplier",
        type="float",
        group="exit_tsl",
        unit="ATR",
        default=1.8,
        min=0.3,
        max=5.0,
        step=0.1,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Base trailing stop distance multiplier when session-day decay is inactive.",
        gate_tags=("tsl_trail",),
    ),
    _S(
        key="TSL_HYSTERESIS_ATR_MULT",
        label="Schmitt Hysteresis ATR Mult",
        type="float",
        group="exit_tsl",
        unit="ATR",
        default=0.5,
        min=0.05,
        max=3.0,
        step=0.05,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Escape dead-band beyond TP1 required before ratcheting stop up to TP1.",
        gate_tags=("schmitt_hysteresis",),
    ),
    _S(
        key="TSL_DECAY_DAY1_MULT",
        label="TSL Decay Day-1 ATR Mult",
        type="float",
        group="exit_tsl",
        unit="ATR",
        default=2.20,
        min=0.5,
        max=5.0,
        step=0.05,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Session-1 ATR trailing stop multiplier prior to multi-day decay compression.",
        gate_tags=("tsl_decay_envelope",),
    ),
    _S(
        key="TSL_DECAY_STEP_MULT",
        label="TSL Decay Step Per Session",
        type="float",
        group="exit_tsl",
        unit="ATR/day",
        default=0.35,
        min=0.0,
        max=2.0,
        step=0.05,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Daily ATR multiplier compression per elapsed exchange trading session.",
        gate_tags=("tsl_decay_envelope",),
    ),
    _S(
        key="TSL_DECAY_FLOOR_MULT",
        label="TSL Decay Floor ATR Mult",
        type="float",
        group="exit_tsl",
        unit="ATR",
        default=1.15,
        min=0.3,
        max=4.0,
        step=0.05,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Minimum ATR trailing stop multiplier floor for compressed non-runner holds.",
        gate_tags=("tsl_decay_envelope",),
    ),
    _S(
        key="TSL_DECAY_RUNNER_MULT",
        label="TSL Runner Frozen ATR Mult",
        type="float",
        group="exit_tsl",
        unit="ATR",
        default=2.00,
        min=0.5,
        max=5.0,
        step=0.05,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Frozen ATR trailing multiplier for tenured runners immune to decay compression.",
        gate_tags=("tsl_decay_envelope", "runner_immunity"),
    ),
    _S(
        key="TP1_R_MULTIPLE",
        label="Take Profit 1 (R-Multiple)",
        type="float",
        group="exit_tsl",
        unit="R",
        default=1.5,
        min=0.5,
        max=10.0,
        step=0.1,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Risk-multiple target for TP1 (must satisfy TP1 < TP2 < TP3).",
        gate_tags=("take_profit_targets",),
    ),
    _S(
        key="TP2_R_MULTIPLE",
        label="Take Profit 2 (R-Multiple)",
        type="float",
        group="exit_tsl",
        unit="R",
        default=2.5,
        min=0.6,
        max=15.0,
        step=0.1,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Risk-multiple target for TP2 (must satisfy TP1 < TP2 < TP3).",
        gate_tags=("take_profit_targets",),
    ),
    _S(
        key="TP3_R_MULTIPLE",
        label="Take Profit 3 (R-Multiple)",
        type="float",
        group="exit_tsl",
        unit="R",
        default=3.5,
        min=0.7,
        max=20.0,
        step=0.1,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Risk-multiple target for TP3 (must satisfy TP1 < TP2 < TP3).",
        gate_tags=("take_profit_targets",),
    ),
    _S(
        key="TP1_TRIM_PCT",
        label="TP1 Trim Fraction",
        type="float",
        group="exit_tsl",
        unit="fraction",
        default=0.50,
        min=0.0,
        max=1.0,
        step=0.05,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Legacy TP1 partial trim weight (TP1+TP2+TP3 must sum to 1.0).",
        gate_tags=("take_profit_trims",),
    ),
    _S(
        key="TP2_TRIM_PCT",
        label="TP2 Trim Fraction",
        type="float",
        group="exit_tsl",
        unit="fraction",
        default=0.25,
        min=0.0,
        max=1.0,
        step=0.05,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Legacy TP2 partial trim weight (TP1+TP2+TP3 must sum to 1.0).",
        gate_tags=("take_profit_trims",),
    ),
    _S(
        key="TP3_TRIM_PCT",
        label="TP3 Trim Fraction",
        type="float",
        group="exit_tsl",
        unit="fraction",
        default=0.25,
        min=0.0,
        max=1.0,
        step=0.05,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Legacy TP3 runner trim weight (TP1+TP2+TP3 must sum to 1.0).",
        gate_tags=("take_profit_trims",),
    ),

    # ── Session & Rate Governance ────────────────────────────────────────────
    _S(
        key="SESSION_CLOSE_CUTOFF_UTC",
        label="Session Close Cutoff (UTC)",
        type="str",
        group="session",
        unit="HH:MM",
        default="16:20",
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="UTC time when Asymmetric Session Protocol evaluates intraday/overnight liquidation rules.",
        gate_tags=("session_close_protocol",),
    ),
    _S(
        key="ENFORCE_INVERSE_INTRADAY_ONLY",
        label="Inverse ETP Intraday Only",
        type="bool",
        group="session",
        unit="",
        default=True,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Forces inverse short ETP positions to flatten before session close cutoff.",
        gate_tags=("inverse_intraday_protocol",),
    ),
    _S(
        key="FLATTEN_ALL_LEVERAGED_AT_SESSION_CLOSE",
        label="Flatten All Leveraged at Close",
        type="bool",
        group="session",
        unit="",
        default=False,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Forces 100% intraday liquidation of all 3x/5x leveraged ETPs at cutoff.",
        gate_tags=("session_close_protocol",),
    ),
    _S(
        key="ENFORCE_LONG_BREAKEVEN_OVERNIGHT",
        label="Enforce Long Overnight Cushion",
        type="bool",
        group="session",
        unit="",
        default=True,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Liquidates long leveraged ETPs at cutoff unless profit cushion >= MIN_OVERNIGHT_PROFIT_CUSHION_PCT.",
        gate_tags=("overnight_cushion_gate",),
    ),
    _S(
        key="MIN_OVERNIGHT_PROFIT_CUSHION_PCT",
        label="Min Overnight Profit Cushion",
        type="float",
        group="session",
        unit="fraction",
        default=0.030,
        min=0.0,
        max=0.25,
        step=0.005,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Minimum unrealized return required to hold a long leveraged ETP overnight.",
        gate_tags=("overnight_cushion_gate",),
    ),
    _S(
        key="EARNINGS_BLACKOUT_HOURS",
        label="Overnight Earnings Blackout",
        type="int",
        group="session",
        unit="hours",
        default=24,
        min=0,
        max=336,
        step=1,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Hours before earnings requiring overnight position evacuation.",
        gate_tags=("earnings_blackout_gate",),
    ),
    _S(
        key="RATE_LIMIT_INTER_CALL_DELAY",
        label="Broker Inter-Call Delay",
        type="float",
        group="session",
        unit="seconds",
        default=1.25,
        min=0.2,
        max=10.0,
        step=0.05,
        reload_mode="hot_reload",
        revalidation_scope="resize_entry",
        impact_hint="Minimum delay between sequential Trading 212 API calls during order reconciliation.",
        gate_tags=("rate_limit_pacing",),
    ),

    # ── Leverage & Universe Mode ─────────────────────────────────────────────
    _S(
        key="USE_LEVERAGED_3X",
        label="Use 3x Leveraged ETPs",
        type="bool",
        group="leverage",
        unit="",
        default=True,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Routes underlying signals to 3x/5x European leveraged ETP instruments.",
        gate_tags=("leverage_resolution",),
    ),
    _S(
        key="ENABLE_CRYPTO_3X",
        label="Enable Crypto 3x Universe",
        type="bool",
        group="leverage",
        unit="",
        default=True,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Includes crypto ETP underlyings in universe scans.",
        gate_tags=("universe_exposure",),
    ),
    _S(
        key="ENABLE_COMMODITIES_3X",
        label="Enable Commodities 3x Universe",
        type="bool",
        group="leverage",
        unit="",
        default=True,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Includes commodities ETP underlyings in universe scans.",
        gate_tags=("universe_exposure",),
    ),
    _S(
        key="ENABLE_5X_LEVERAGE",
        label="Enable 5x Index ETPs",
        type="bool",
        group="leverage",
        unit="",
        default=False,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Allows 5x leverage on eligible benchmark ETPs (requires MAX_LEVERAGE=5).",
        gate_tags=("leverage_resolution", "leverage_cap_gate"),
    ),
    _S(
        key="PREFER_5X_LEVERAGE",
        label="Prefer 5x When Eligible",
        type="bool",
        group="leverage",
        unit="",
        default=False,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Prefers 5x over 3x when 5x leverage is enabled (requires ENABLE_5X_LEVERAGE=True).",
        gate_tags=("leverage_resolution",),
    ),
    _S(
        key="MAX_LEVERAGE",
        label="Max Leverage Cap",
        type="int",
        group="leverage",
        unit="x",
        default=3,
        min=1,
        max=5,
        step=1,
        allowed_values=(1, 3, 5),
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Hard leverage ceiling enforced by RiskManager.validate_trade() (must align with ENABLE_5X_LEVERAGE).",
        gate_tags=("leverage_cap_gate",),
    ),

    # ── External Data & AI Providers ─────────────────────────────────────────
    _S(
        key="ENABLE_SPECULATIVE_ROUTING",
        label="Enable Speculative Routing",
        type="bool",
        group="providers",
        unit="",
        default=False,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Legacy fast-reflex toggle (kept disabled in System 2 mode).",
        gate_tags=("debate_routing",),
    ),
    _S(
        key="SPECULATIVE_CONFIDENCE_THRESHOLD",
        label="Speculative Threshold",
        type="float",
        group="providers",
        unit="score",
        default=0.70,
        min=0.50,
        max=0.99,
        step=0.01,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Confidence cutoff associated with speculative debate routing.",
        gate_tags=("debate_routing",),
    ),

    # ── Runtime & Restart-Required Settings ──────────────────────────────────
    _S(
        key="LLM_DEEP_MODEL",
        label="Deep Reasoning LLM Model",
        type="str",
        group="runtime",
        unit="",
        default="gpt-4o",
        reload_mode="restart_required",
        revalidation_scope="abort_stale_scan",
        impact_hint="OpenAI model bound at boot to Technical, Fundamental, and Debate agents.",
        gate_tags=("llm_deliberation",),
    ),
    _S(
        key="LLM_QUICK_MODEL",
        label="Quick Analysis LLM Model",
        type="str",
        group="runtime",
        unit="",
        default="gpt-4o-mini",
        reload_mode="restart_required",
        revalidation_scope="abort_stale_scan",
        impact_hint="OpenAI quick model bound at boot to DebateEngine.",
        gate_tags=("llm_deliberation",),
    ),
    _S(
        key="PARALLEL_INGESTION_WORKERS",
        label="Ingestion Worker Threads",
        type="int",
        group="runtime",
        unit="workers",
        default=24,
        min=1,
        max=64,
        step=1,
        reload_mode="restart_required",
        revalidation_scope="abort_stale_scan",
        impact_hint="Thread pool concurrency for OHLCV & indicator ingestion.",
        gate_tags=("ingestion_pool",),
    ),
    _S(
        key="PARALLEL_DELIBERATION_WORKERS",
        label="Deliberation Worker Threads",
        type="int",
        group="runtime",
        unit="workers",
        default=8,
        min=1,
        max=32,
        step=1,
        reload_mode="restart_required",
        revalidation_scope="abort_stale_scan",
        impact_hint="Thread pool concurrency for System 2 candidate deliberation.",
        gate_tags=("deliberation_pool",),
    ),
    _S(
        key="ENABLE_CANDLE_CACHE",
        label="Enable Candle Cache",
        type="bool",
        group="runtime",
        unit="",
        default=True,
        reload_mode="restart_required",
        revalidation_scope="abort_stale_scan",
        impact_hint="Enables in-memory/SQLite candle caching across scan cycles.",
        gate_tags=("candle_cache",),
    ),
    _S(
        key="CANDLE_CACHE_TTL_SECONDS",
        label="Candle Cache TTL",
        type="int",
        group="runtime",
        unit="seconds",
        default=300,
        min=10,
        max=86400,
        step=30,
        reload_mode="restart_required",
        revalidation_scope="abort_stale_scan",
        impact_hint="TTL before cached OHLCV candles refresh from providers.",
        gate_tags=("candle_cache",),
    ),
    _S(
        key="RECONCILIATION_INTERVAL_SECONDS",
        label="Broker Reconciliation Interval",
        type="int",
        group="runtime",
        unit="seconds",
        default=60,
        min=10,
        max=3600,
        step=10,
        reload_mode="restart_required",
        revalidation_scope="resize_entry",
        impact_hint="Interval between full broker portfolio and GTC stop-order reconciliations.",
        gate_tags=("broker_reconciliation",),
    ),
    _S(
        key="UNIVERSE_SIZE",
        label="Fallback Equity Universe Size",
        type="int",
        group="runtime",
        unit="assets",
        default=100,
        min=1,
        max=500,
        step=5,
        reload_mode="restart_required",
        revalidation_scope="abort_stale_scan",
        impact_hint="Maximum number of equities scanned when 3x ETP mode or watchlist override is off.",
        gate_tags=("universe_size",),
    ),
    _S(
        key="WATCHLIST",
        label="Custom Watchlist Override",
        type="string_list",
        group="runtime",
        unit="symbols",
        default=None,
        nullable=True,
        reload_mode="restart_required",
        revalidation_scope="abort_stale_scan",
        impact_hint="Explicit list of underlying symbols overriding the default universe (restart required).",
        gate_tags=("universe_watchlist",),
    ),
    _S(
        key="LOG_LEVEL",
        label="Application Log Level",
        type="str",
        group="runtime",
        unit="",
        default="INFO",
        allowed_values=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        reload_mode="restart_required",
        revalidation_scope="resize_entry",
        impact_hint="Root logging verbosity configured at process startup.",
        gate_tags=("logging",),
    ),
    _S(
        key="ENABLE_DESKTOP_NOTIFICATIONS",
        label="Enable Desktop Notifications",
        type="bool",
        group="runtime",
        unit="",
        default=True,
        reload_mode="restart_required",
        revalidation_scope="resize_entry",
        impact_hint="Enables OS desktop alerts on the Notifier initialized at boot.",
        gate_tags=("notifications",),
    ),
)

SCHEMA_BY_KEY: Dict[str, ConfigParamSchema] = {entry.key: entry for entry in CONFIG_SCHEMA}
TUNABLE_KEYS: Tuple[str, ...] = tuple(entry.key for entry in CONFIG_SCHEMA)
SETTINGS_FIELD_NAMES: FrozenSet[str] = frozenset(f.name for f in dataclasses.fields(Settings))
SETTINGS_FIELD_NAMES_LOWER: FrozenSet[str] = frozenset(name.lower() for name in SETTINGS_FIELD_NAMES)

# ── Dynamic Parameter Schemas for ML Alpha Strategy ──────────────────────────
ML_ALPHA_DYNAMIC_SCHEMAS: Tuple[ConfigParamSchema, ...] = (
    _S(
        key="ML_MIN_CONFIDENCE",
        label="ML Min Confidence",
        type="float",
        group="strategy",
        unit="prob",
        default=0.58,
        min=0.0,
        max=1.0,
        step=0.01,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Minimum winning probability required for ML Alpha signal emission.",
        gate_tags=("ml_alpha", "confidence_gate"),
    ),
    _S(
        key="ML_MIN_EDGE_MARGIN",
        label="ML Min Edge Margin",
        type="float",
        group="strategy",
        unit="margin",
        default=0.15,
        min=0.0,
        max=1.0,
        step=0.01,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Minimum edge margin over cash defense (p_winner - p_cash) for ML Alpha signals.",
        gate_tags=("ml_alpha", "edge_margin_gate"),
    ),
    _S(
        key="ML_MAX_VOL_DRAG_CUTOFF",
        label="ML Max Vol Drag Cutoff",
        type="float",
        group="strategy",
        unit="drag",
        default=0.05,
        min=0.0,
        max=1.0,
        step=0.005,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Maximum volatility drag proxy cutoff for ML Alpha strategy.",
        gate_tags=("ml_alpha", "vol_drag_gate"),
    ),
    _S(
        key="ML_COOLDOWN_BARS",
        label="ML Cooldown Bars",
        type="int",
        group="strategy",
        unit="bars",
        default=3,
        min=0,
        max=50,
        step=1,
        reload_mode="hot_reload",
        revalidation_scope="abort_stale_scan",
        impact_hint="Cooldown bars between signals for ML Alpha strategy.",
        gate_tags=("ml_alpha", "cooldown_gate"),
    ),
    _S(
        key="ML_MODEL_PATH",
        label="ML Model Path",
        type="str",
        group="strategy",
        unit="",
        default="models/ml_alpha_v1.onnx",
        reload_mode="restart_required",
        revalidation_scope="abort_stale_scan",
        impact_hint="Path to ONNX model artifact for ML Alpha strategy.",
        gate_tags=("ml_alpha", "model_artifact"),
    ),
)

ML_ALPHA_DYNAMIC_SCHEMAS_BY_KEY: Dict[str, ConfigParamSchema] = {
    entry.key: entry for entry in ML_ALPHA_DYNAMIC_SCHEMAS
}
DYNAMIC_PARAMETER_SCHEMAS: Dict[str, ConfigParamSchema] = ML_ALPHA_DYNAMIC_SCHEMAS_BY_KEY
ML_ALPHA_PARAM_SCHEMAS: Tuple[ConfigParamSchema, ...] = ML_ALPHA_DYNAMIC_SCHEMAS
ML_ALPHA_DEFAULT_DYNAMIC_PARAMETERS: Dict[str, Any] = {
    entry.key: entry.default for entry in ML_ALPHA_DYNAMIC_SCHEMAS
}


@dataclass(frozen=True)
class ScreenerMetricGate:
    """Declarative screener/strategy gate rule bound to a custom parameter."""

    metric: str
    op: str
    threshold: float
    on_missing: str = "reject"

    def __post_init__(self) -> None:
        if not isinstance(self.metric, str) or self.metric not in ALLOWLISTED_SCREENER_METRICS:
            raise ValueError(
                f"screener_metric_gate.metric must be in ALLOWLISTED_SCREENER_METRICS, got {self.metric!r}"
            )
        if not isinstance(self.op, str) or self.op not in ALLOWED_GATE_OPERATORS:
            raise ValueError(
                f"screener_metric_gate.op must be one of {ALLOWED_GATE_OPERATORS}, got {self.op!r}"
            )
        if not is_finite_number(self.threshold):
            raise ValueError(
                f"screener_metric_gate.threshold must be a finite int/float, got {self.threshold!r}"
            )
        object.__setattr__(self, "threshold", float(self.threshold))
        if not isinstance(self.on_missing, str) or self.on_missing not in ALLOWED_ON_MISSING_MODES:
            raise ValueError(
                f"screener_metric_gate.on_missing must be one of {ALLOWED_ON_MISSING_MODES}, got {self.on_missing!r}"
            )

    @classmethod
    def from_dict(cls, data: Any) -> "ScreenerMetricGate":
        """Validate and construct a :class:`ScreenerMetricGate` from a dictionary."""
        if not isinstance(data, Mapping):
            raise ValueError("screener_metric_gate must be a JSON object")
        allowed_keys = {"metric", "op", "threshold", "on_missing"}
        unknown = set(data.keys()) - allowed_keys
        if unknown:
            raise ValueError(f"Unknown screener_metric_gate field(s): {sorted(unknown)}")
        if "metric" not in data or "op" not in data or "threshold" not in data:
            raise ValueError("screener_metric_gate requires 'metric', 'op', and 'threshold'")
        return cls(
            metric=data["metric"],
            op=data["op"],
            threshold=data["threshold"],
            on_missing=data.get("on_missing", "reject"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize gate rule into a JSON-compatible dictionary."""
        return {
            "metric": self.metric,
            "op": self.op,
            "threshold": self.threshold,
            "on_missing": self.on_missing,
        }


def _to_sequence_tuple(value: Any, field_name: str) -> Tuple[Any, ...]:
    """Convert a non-string iterable to a tuple, raising ValueError otherwise."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{field_name} must be a sequence")
    return tuple(value)


_DYNAMIC_SPEC_ALLOWED_KEYS: FrozenSet[str] = frozenset({
    "key",
    "label",
    "type",
    "default",
    "value",
    "group",
    "unit",
    "min",
    "max",
    "step",
    "allowed_values",
    "reload_mode",
    "revalidation_scope",
    "impact_hint",
    "gate_tags",
    "screener_metric_gate",
    "is_dynamic",
    "nullable",
})


@dataclass(frozen=True)
class DynamicParameterSpec:
    """Validated user-defined parameter wired to a declarative screener_metric_gate."""

    key: str
    label: str
    type: str
    default: Any
    screener_metric_gate: ScreenerMetricGate
    group: str = "screener"
    unit: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    allowed_values: Optional[Tuple[Any, ...]] = None
    reload_mode: str = "hot_reload"
    revalidation_scope: str = "abort_stale_scan"
    impact_hint: str = ""
    gate_tags: Tuple[str, ...] = ("dynamic_metric_gate",)

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not DYNAMIC_PARAM_KEY_RE.match(self.key):
            raise ValueError(
                f"Dynamic parameter key {self.key!r} must match {DYNAMIC_PARAM_KEY_RE.pattern}"
            )
        key_lower = self.key.lower()
        if keyword.iskeyword(self.key) or self.key.startswith("__"):
            raise ValueError(f"Dynamic parameter key {self.key!r} is a reserved identifier")
        if (
            self.key in SETTINGS_FIELD_NAMES
            or key_lower in SETTINGS_FIELD_NAMES_LOWER
            or is_env_only_key(self.key)
            or self.key in SCHEMA_BY_KEY
        ):
            raise ValueError(f"Dynamic parameter key {self.key!r} collides with Settings or env-only key")
        if self.key in SNAPSHOT_METADATA_KEYS or key_lower in _SNAPSHOT_METADATA_LOWER:
            raise ValueError(f"Dynamic parameter key {self.key!r} collides with ConfigSnapshot metadata")

        _check_string_length(self.label, "label")
        if not self.label.strip():
            raise ValueError("Dynamic parameter label must be non-empty")
        _check_string_length(self.group, "group")
        if not self.group.strip():
            raise ValueError("Dynamic parameter group must be non-empty")
        _check_string_length(self.unit, "unit")
        _check_string_length(self.impact_hint, "impact_hint")

        if self.type not in ("float", "int", "bool", "str"):
            raise ValueError(f"Unsupported dynamic parameter type {self.type!r}")
        if self.reload_mode not in RELOAD_MODES:
            raise ValueError(f"Invalid reload_mode {self.reload_mode!r}")
        if self.revalidation_scope not in REVALIDATION_SCOPES:
            raise ValueError(f"Invalid revalidation_scope {self.revalidation_scope!r}")

        if not isinstance(self.screener_metric_gate, ScreenerMetricGate):
            raise ValueError("DynamicParameterSpec requires a validated ScreenerMetricGate")

        min_val = self.min
        max_val = self.max
        step_val = self.step
        if min_val is not None:
            if not is_finite_number(min_val):
                raise ValueError(f"min must be a finite number, got {min_val!r}")
            min_val = float(min_val)
            object.__setattr__(self, "min", min_val)
        if max_val is not None:
            if not is_finite_number(max_val):
                raise ValueError(f"max must be a finite number, got {max_val!r}")
            max_val = float(max_val)
            object.__setattr__(self, "max", max_val)
        if min_val is not None and max_val is not None and min_val > max_val:
            raise ValueError(f"min ({min_val}) cannot exceed max ({max_val})")
        if step_val is not None:
            if not is_finite_number(step_val) or float(step_val) <= 0:
                raise ValueError(f"step must be a positive finite number, got {step_val!r}")
            object.__setattr__(self, "step", float(step_val))

        if self.allowed_values is not None:
            raw_allowed = _to_sequence_tuple(self.allowed_values, "allowed_values")
            coerced_allowed: List[Any] = []
            for item in raw_allowed:
                coerced_item, err = validate_value_against_spec(self, item, ignore_allowed=True)
                if err is not None:
                    raise ValueError(f"Invalid allowed_values entry {item!r}: {err['message']}")
                coerced_allowed.append(coerced_item)
            if not coerced_allowed:
                raise ValueError("allowed_values cannot be empty if provided")
            object.__setattr__(self, "allowed_values", tuple(coerced_allowed))

        raw_tags = _to_sequence_tuple(self.gate_tags, "gate_tags")
        tags_list: List[str] = []
        for tag in raw_tags:
            _check_string_length(tag, "gate_tags[]")
            tags_list.append(tag)
        object.__setattr__(self, "gate_tags", tuple(tags_list))

        coerced_default, err = validate_value_against_spec(self, self.default)
        if err is not None:
            raise ValueError(f"Invalid default for {self.key}: {err['message']}")
        object.__setattr__(self, "default", coerced_default)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DynamicParameterSpec":
        """Validate and construct a :class:`DynamicParameterSpec` from a raw mapping."""
        if not isinstance(data, Mapping):
            raise ValueError("Dynamic parameter specification must be a JSON object")

        for k in data.keys():
            if str(k).lower() in _FORBIDDEN_EVAL_KEYS:
                raise ValueError(f"Forbidden dynamic parameter field {k!r}")
        unknown = set(data.keys()) - _DYNAMIC_SPEC_ALLOWED_KEYS
        if unknown:
            raise ValueError(f"Unsupported dynamic parameter field(s): {sorted(unknown)}")
        if "is_dynamic" in data and data["is_dynamic"] is not True:
            raise ValueError("Dynamic parameter field 'is_dynamic' must be True when present")
        if "nullable" in data and data["nullable"] is not False:
            raise ValueError("Dynamic parameter field 'nullable' must be False when present")

        if "key" not in data:
            raise ValueError("Dynamic parameter specification requires 'key'")
        if "type" not in data:
            raise ValueError("Dynamic parameter specification requires 'type'")
        if "screener_metric_gate" not in data or data["screener_metric_gate"] is None:
            raise ValueError("Dynamic parameter specification requires a declarative 'screener_metric_gate'")

        gate = (
            data["screener_metric_gate"]
            if isinstance(data["screener_metric_gate"], ScreenerMetricGate)
            else ScreenerMetricGate.from_dict(data["screener_metric_gate"])
        )

        param_type = data["type"]
        if "default" in data:
            default_val = data["default"]
        elif "value" in data:
            default_val = data["value"]
        elif param_type == "float":
            default_val = float(gate.threshold)
        elif param_type == "int":
            default_val = int(round(gate.threshold))
        else:
            raise ValueError(f"Dynamic parameter of type {param_type!r} requires 'default'")

        label = data["label"] if "label" in data else str(data["key"])
        raw_allowed = data.get("allowed_values")
        allowed_values_tuple: Optional[Tuple[Any, ...]] = (
            _to_sequence_tuple(raw_allowed, "allowed_values")
            if raw_allowed is not None
            else None
        )
        gate_tags_raw = data.get("gate_tags", ("dynamic_metric_gate",))
        gate_tags_tuple: Tuple[str, ...] = (
            _to_sequence_tuple(gate_tags_raw, "gate_tags")
            if gate_tags_raw is not None
            else ()
        )

        return cls(
            key=data["key"],
            label=label,
            type=param_type,
            default=default_val,
            screener_metric_gate=gate,
            group=data.get("group", "screener"),
            unit=data.get("unit", ""),
            min=data.get("min"),
            max=data.get("max"),
            step=data.get("step"),
            allowed_values=allowed_values_tuple,
            reload_mode=data.get("reload_mode", "hot_reload"),
            revalidation_scope=data.get("revalidation_scope", "abort_stale_scan"),
            impact_hint=data.get("impact_hint", ""),
            gate_tags=gate_tags_tuple,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize custom parameter spec into a JSON-compatible dictionary."""
        return {
            "key": self.key,
            "label": self.label,
            "type": self.type,
            "default": self.default,
            "group": self.group,
            "unit": self.unit,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "allowed_values": list(self.allowed_values) if self.allowed_values is not None else None,
            "reload_mode": self.reload_mode,
            "revalidation_scope": self.revalidation_scope,
            "impact_hint": self.impact_hint,
            "gate_tags": list(self.gate_tags),
            "screener_metric_gate": self.screener_metric_gate.to_dict(),
            "nullable": False,
            "is_dynamic": True,
        }


def validate_value_against_spec(
    spec: Union[ConfigParamSchema, DynamicParameterSpec],
    value: Any,
    ignore_allowed: bool = False,
) -> Tuple[Any, Optional[Dict[str, str]]]:
    """Validate and coerce *value* against a schema or dynamic parameter spec.

    Returns:
        ``(coerced_value, None)`` on success, or ``(None, error_dict)`` on failure.
    """
    key = spec.key
    nullable = getattr(spec, "nullable", False)

    def _err(code: str, msg: str) -> Tuple[Any, Dict[str, str]]:
        return None, {"key": key, "code": code, "message": msg}

    if value is None:
        if nullable:
            return None, None
        return _err("invalid_type", f"{key} cannot be None")

    ptype = spec.type
    if ptype == "bool":
        if not isinstance(value, bool):
            return _err("invalid_type", f"{key} must be a bool, got {type(value).__name__}")
        coerced: Any = value
    elif ptype == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            return _err("invalid_type", f"{key} must be an int, got {type(value).__name__}")
        coerced = int(value)
    elif ptype == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return _err("invalid_type", f"{key} must be a finite float/int, got {type(value).__name__}")
        if not math.isfinite(float(value)):
            return _err("non_finite_float", f"{key} must be a finite float, got {value!r}")
        coerced = float(value)
    elif ptype == "str":
        if not isinstance(value, str):
            return _err("invalid_type", f"{key} must be a string, got {type(value).__name__}")
        if len(value) > MAX_DYNAMIC_STRING_LENGTH:
            return _err("string_too_long", f"{key} exceeds {MAX_DYNAMIC_STRING_LENGTH} characters")
        if key != "impact_hint" and not value.strip():
            return _err("invalid_value", f"{key} must be a non-empty string")
        if key == "SESSION_CLOSE_CUTOFF_UTC" and not _TIME_HHMM_RE.match(value.strip()):
            return _err("invalid_format", f"{key} must match HH:MM (UTC 24h), got {value!r}")
        coerced = value.strip()
        if key == "LOG_LEVEL":
            coerced = coerced.upper()
        elif key == "SCREENER_MIN_SETUP_SCORE":
            coerced = coerced.lower()
    elif ptype == "string_list":
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            return _err("invalid_type", f"{key} must be a list/tuple of strings or None")
        cleaned: List[str] = []
        for item in value:
            if not isinstance(item, str):
                return _err("invalid_type", f"{key} elements must be strings, got {type(item).__name__}")
            if len(item) > MAX_DYNAMIC_STRING_LENGTH:
                return _err("string_too_long", f"{key} element exceeds {MAX_DYNAMIC_STRING_LENGTH} characters")
            s = item.strip()
            if s:
                cleaned.append(s)
        coerced = tuple(cleaned)
    else:
        return _err("invalid_type", f"Unsupported schema type {ptype!r} for {key}")

    if ptype in ("int", "float"):
        if spec.min is not None and coerced < spec.min:
            return _err("out_of_bounds", f"{key} ({coerced}) is below minimum {spec.min}")
        if spec.max is not None and coerced > spec.max:
            return _err("out_of_bounds", f"{key} ({coerced}) exceeds maximum {spec.max}")

    if not ignore_allowed and spec.allowed_values is not None:
        if coerced not in spec.allowed_values:
            return _err(
                "invalid_value",
                f"{key} value {coerced!r} must be one of {list(spec.allowed_values)}",
            )

    return coerced, None


def evaluate_screener_metric_gate(
    gate: Union[ScreenerMetricGate, Mapping[str, Any]],
    metrics: Mapping[str, Any],
    active_value: Any = None,
) -> Tuple[bool, Optional[float], float, str]:
    """Deterministically evaluate a declarative ``screener_metric_gate`` rule.

    Args:
        gate: Validated :class:`ScreenerMetricGate` (or dict convertible via ``from_dict``).
        metrics: Candidate technical indicator dictionary.
        active_value: Current custom parameter value from ``ConfigSnapshot``. If ``bool``
            and ``False``, the gate is treated as disabled (passes). If numeric, overrides
            the static gate threshold.

    Returns:
        ``(passed, observed_float_or_none, effective_threshold, reason)``.
    """
    g = gate if isinstance(gate, ScreenerMetricGate) else ScreenerMetricGate.from_dict(gate)
    if isinstance(active_value, bool) and not active_value:
        return True, None, g.threshold, "gate_disabled"

    threshold = float(active_value) if is_finite_number(active_value) else g.threshold
    raw_val = metrics.get(g.metric)
    if not is_finite_number(raw_val):
        if g.on_missing == "skip_gate":
            return True, None, threshold, "missing_metric_skipped"
        return False, None, threshold, f"missing_metric:{g.metric}"

    observed = float(raw_val)
    op = g.op
    if op == "<":
        passed = observed < threshold
    elif op == "<=":
        passed = observed <= threshold
    elif op == ">":
        passed = observed > threshold
    elif op == ">=":
        passed = observed >= threshold
    elif op == "==":
        passed = observed == threshold
    elif op == "!=":
        passed = observed != threshold
    else:
        raise ValueError(f"Unsupported operator {op!r}")

    reason = "passed" if passed else f"failed:{g.metric}({observed}){op}{threshold}"
    return passed, observed, threshold, reason


BUILTIN_PROFILES: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "id": "balanced",
        "label": "Balanced (Default)",
        "description": "Default institutional 3x directional trend configuration with standard ATR noise buffers.",
        "read_only": True,
        "values": {
            "MIN_ADX_TREND": 18.0,
            "STRATEGY_EXCEPTIONAL_ADX": 25.0,
            "MIN_EMA20_SLOPE_PCT": 0.0,
            "SCREENER_MIN_SETUP_SCORE": "favorable",
            "SCREENER_MAX_CATALYST_RISK": 0.25,
            "SCREENER_TOP_N": 15,
            "REGIME_BREAKOUT_ADX_MIN": 25.0,
            "HIGH_VOL_MAX_REALIZED_VOL": 0.65,
            "MAX_RISK_PER_TRADE": 0.015,
            "MAX_POSITION_PCT": 0.20,
            "ATR_SL_MULTIPLIER": 2.2,
            "ETP_MIN_SL_PCT": 0.025,
            "ETP_MAX_SL_PCT": 0.080,
            "TSL_DECAY_DAY1_MULT": 2.20,
            "TSL_DECAY_STEP_MULT": 0.35,
            "TSL_DECAY_FLOOR_MULT": 1.15,
            "TSL_DECAY_RUNNER_MULT": 2.00,
            "MAX_BID_ASK_SPREAD_PCT": 0.015,
            "MAX_SECTOR_POSITIONS": 4,
            "MAX_PORTFOLIO_BETA_HEAT": 6.0,
            "DAILY_MAX_DRAWDOWN_PCT": 0.02,
        },
    },
    "conservative": {
        "id": "conservative",
        "label": "Conservative Capital Preservation",
        "description": "Tighter risk-per-trade, lower beta heat, stricter spread and volatility caps, and faster TSL decay.",
        "read_only": True,
        "values": {
            "MIN_ADX_TREND": 22.0,
            "STRATEGY_EXCEPTIONAL_ADX": 27.0,
            "MIN_EMA20_SLOPE_PCT": 0.05,
            "SCREENER_MIN_SETUP_SCORE": "favorable",
            "SCREENER_MAX_CATALYST_RISK": 0.15,
            "SCREENER_TOP_N": 8,
            "REGIME_BREAKOUT_ADX_MIN": 27.0,
            "HIGH_VOL_MAX_REALIZED_VOL": 0.50,
            "MAX_RISK_PER_TRADE": 0.0075,
            "MAX_POSITION_PCT": 0.12,
            "ATR_SL_MULTIPLIER": 2.0,
            "ETP_MIN_SL_PCT": 0.025,
            "ETP_MAX_SL_PCT": 0.060,
            "TSL_DECAY_DAY1_MULT": 1.85,
            "TSL_DECAY_STEP_MULT": 0.35,
            "TSL_DECAY_FLOOR_MULT": 1.00,
            "TSL_DECAY_RUNNER_MULT": 1.75,
            "MAX_BID_ASK_SPREAD_PCT": 0.010,
            "MAX_SECTOR_POSITIONS": 2,
            "MAX_PORTFOLIO_BETA_HEAT": 3.5,
            "DAILY_MAX_DRAWDOWN_PCT": 0.015,
        },
    },
    "momentum_strict": {
        "id": "momentum_strict",
        "label": "Momentum Strict",
        "description": "High-ADX breakout selectivity requiring exceptional setups and strong 20-EMA slope.",
        "read_only": True,
        "values": {
            "MIN_ADX_TREND": 25.0,
            "STRATEGY_EXCEPTIONAL_ADX": 28.0,
            "MIN_EMA20_SLOPE_PCT": 0.25,
            "SCREENER_MIN_SETUP_SCORE": "exceptional",
            "SCREENER_MAX_CATALYST_RISK": 0.20,
            "SCREENER_TOP_N": 10,
            "REGIME_BREAKOUT_ADX_MIN": 28.0,
            "HIGH_VOL_MAX_REALIZED_VOL": 0.60,
            "MAX_RISK_PER_TRADE": 0.012,
            "MAX_POSITION_PCT": 0.18,
            "ATR_SL_MULTIPLIER": 2.3,
            "ETP_MIN_SL_PCT": 0.025,
            "ETP_MAX_SL_PCT": 0.080,
            "TSL_DECAY_DAY1_MULT": 2.20,
            "TSL_DECAY_STEP_MULT": 0.35,
            "TSL_DECAY_FLOOR_MULT": 1.25,
            "TSL_DECAY_RUNNER_MULT": 2.20,
            "MAX_BID_ASK_SPREAD_PCT": 0.012,
            "MAX_SECTOR_POSITIONS": 3,
            "MAX_PORTFOLIO_BETA_HEAT": 5.0,
            "DAILY_MAX_DRAWDOWN_PCT": 0.02,
        },
    },
}


def validate_builtin_profiles() -> None:
    """Validate that every built-in profile conforms strictly to schema bounds and Settings invariants."""
    base_defaults = {k: schema.default for k, schema in SCHEMA_BY_KEY.items()}
    for profile_id, profile in BUILTIN_PROFILES.items():
        if profile.get("id") != profile_id or not profile.get("read_only"):
            raise ValueError(f"Built-in profile {profile_id!r} must define matching id and read_only=True")
        values = profile.get("values")
        if not isinstance(values, Mapping) or not values:
            raise ValueError(f"Built-in profile {profile_id!r} must have non-empty values")
        merged = dict(base_defaults)
        for k, raw_v in values.items():
            if k not in SCHEMA_BY_KEY:
                raise ValueError(f"Built-in profile {profile_id!r} references unknown/env-only key {k!r}")
            coerced, err = validate_value_against_spec(SCHEMA_BY_KEY[k], raw_v)
            if err is not None:
                raise ValueError(f"Built-in profile {profile_id!r} invalid {k}: {err['message']}")
            merged[k] = coerced
        inv_errors = collect_settings_invariant_errors(merged)
        if inv_errors:
            raise ValueError(f"Built-in profile {profile_id!r} violates invariant: {inv_errors[0]['message']}")


validate_builtin_profiles()


def get_builtin_profiles() -> Dict[str, Dict[str, Any]]:
    """Return a deep copy of built-in read-only configuration profiles."""
    return copy.deepcopy(BUILTIN_PROFILES)


def export_config_schema(dynamic_specs: Iterable[DynamicParameterSpec] = ()) -> Dict[str, Any]:
    """Export schema entries, groups, profiles, and allowlisted metrics for API consumers."""
    params = [entry.to_dict() for entry in CONFIG_SCHEMA]
    for spec in dynamic_specs:
        params.append(spec.to_dict())
    return {
        "parameters": params,
        "dynamic_parameter_schemas": [s.to_dict() for s in ML_ALPHA_DYNAMIC_SCHEMAS],
        "ml_alpha_parameter_schemas": [s.to_dict() for s in ML_ALPHA_DYNAMIC_SCHEMAS],
        "env_only_keys": sorted(ENV_ONLY_KEYS),
        "allowlisted_screener_metrics": sorted(ALLOWLISTED_SCREENER_METRICS),
        "allowed_gate_operators": list(ALLOWED_GATE_OPERATORS),
        "allowed_on_missing_modes": list(ALLOWED_ON_MISSING_MODES),
        "max_dynamic_parameters": MAX_DYNAMIC_PARAMETERS,
        "max_dynamic_string_length": MAX_DYNAMIC_STRING_LENGTH,
        "builtin_profiles": get_builtin_profiles(),
    }
