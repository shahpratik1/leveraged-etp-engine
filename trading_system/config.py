"""Central configuration module for an institutional-grade hybrid quantitative trading engine.

Provides strongly typed configuration settings, environment variable loading,
risk and universe management parameters, and API base URL resolution.
"""

from __future__ import annotations

import logging
import math
import os
import re
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union


logger = logging.getLogger(__name__)

_TIME_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
_VALID_SETUP_SCORES = ("unfavorable", "mediocre", "favorable", "exceptional")
_VALID_LEVERAGE_CAPS = (1, 3, 5)


def parse_env_file(path: Union[str, Path]) -> Dict[str, str]:
    """Parse a dotenv-formatted file into a ``{key: value}`` dictionary."""
    parsed: Dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return parsed
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k:
                        parsed[k] = v
    except Exception:
        pass
    return parsed


def _parse_bool_env(raw: str) -> bool:
    """Parse common boolean environment variable representations."""
    return raw.strip().lower() in ("1", "true", "yes", "t")


def _load_env_file() -> None:
    """Load environment variables from .env file if present."""
    env_paths = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
    ]
    for p in env_paths:
        if os.path.exists(p):
            for k, v in parse_env_file(p).items():
                if k not in os.environ:
                    os.environ[k] = v
            break


_load_env_file()


def collect_settings_invariant_errors(source: Any) -> List[Dict[str, str]]:
    """Validate numeric finiteness, bounds, and cross-field Settings invariants.

    Args:
        source: Either a ``Settings`` instance or a mapping of configuration keys.

    Returns:
        List of ``{"key": str, "code": str, "message": str}`` error dictionaries.
    """
    is_mapping = isinstance(source, Mapping)

    def _val(key: str, default: Any = None) -> Any:
        if is_mapping:
            return source.get(key, default)
        return getattr(source, key, default)

    errors: List[Dict[str, str]] = []

    def _err(key: str, code: str, message: str) -> None:
        errors.append({"key": key, "code": code, "message": message})

    float_keys = (
        "DEFAULT_ACCOUNT_EQUITY",
        "SPECULATIVE_CONFIDENCE_THRESHOLD",
        "MAX_RISK_PER_TRADE",
        "MAX_POSITION_PCT",
        "ATR_SL_MULTIPLIER",
        "MIN_SL_PCT",
        "MAX_SL_PCT",
        "ETP_MIN_SL_PCT",
        "ETP_MAX_SL_PCT",
        "TP1_R_MULTIPLE",
        "TP2_R_MULTIPLE",
        "TP3_R_MULTIPLE",
        "TARGET_ANNUAL_VOLATILITY",
        "SCREENER_MAX_CATALYST_RISK",
        "MIN_ADX_TREND",
        "STRATEGY_EXCEPTIONAL_ADX",
        "MIN_EMA20_SLOPE_PCT",
        "ENTRY_EARNINGS_BLACKOUT_DAYS",
        "STRATEGY_BASE_CONVICTION",
        "STRATEGY_SHORT_BASE_CONVICTION",
        "STRATEGY_MACRO_TAILWIND_BONUS",
        "DIRECTIONAL_MAX_CONVICTION",
        "REGIME_BREAKOUT_ADX_MIN",
        "HIGH_VOL_MAX_REALIZED_VOL",
        "HIGH_VOL_MAX_ATR_SPIKE_RATIO",
        "FALLBACK_SCREENER_MIN_ADX",
        "TSL_ATR_MULTIPLIER",
        "TSL_HYSTERESIS_ATR_MULT",
        "TSL_DECAY_DAY1_MULT",
        "TSL_DECAY_STEP_MULT",
        "TSL_DECAY_FLOOR_MULT",
        "TSL_DECAY_RUNNER_MULT",
        "TP1_TRIM_PCT",
        "TP2_TRIM_PCT",
        "TP3_TRIM_PCT",
        "MIN_OVERNIGHT_PROFIT_CUSHION_PCT",
        "RATE_LIMIT_INTER_CALL_DELAY",
        "MAX_BID_ASK_SPREAD_PCT",
        "MAX_SPREAD_TO_STOP_RATIO",
        "TRADE_LOSS_COOLDOWN_SECONDS",
        "MIN_HOLD_SECONDS",
        "MAX_PORTFOLIO_BETA_HEAT",
        "DAILY_MAX_DRAWDOWN_PCT",
    )
    non_finite_keys: set[str] = set()
    for fk in float_keys:
        v = _val(fk)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            non_finite_keys.add(fk)
            _err(fk, "non_finite_float", f"{fk} must be a finite number, got {v!r}")

    max_risk = _val("MAX_RISK_PER_TRADE", 0.015)
    max_pos = _val("MAX_POSITION_PCT", 0.20)
    if "MAX_RISK_PER_TRADE" not in non_finite_keys and not (0.0 < max_risk <= 1.0):
        _err("MAX_RISK_PER_TRADE", "out_of_bounds", f"MAX_RISK_PER_TRADE must be in (0.0, 1.0], got {max_risk}")
    if "MAX_POSITION_PCT" not in non_finite_keys and not (0.0 < max_pos <= 1.0):
        _err("MAX_POSITION_PCT", "out_of_bounds", f"MAX_POSITION_PCT must be in (0.0, 1.0], got {max_pos}")
    if (
        "MAX_RISK_PER_TRADE" not in non_finite_keys
        and "MAX_POSITION_PCT" not in non_finite_keys
        and max_risk > max_pos
    ):
        _err(
            "MAX_RISK_PER_TRADE",
            "cross_field_error",
            f"MAX_RISK_PER_TRADE ({max_risk}) cannot exceed MAX_POSITION_PCT ({max_pos})",
        )

    atr_mult = _val("ATR_SL_MULTIPLIER", 2.2)
    if "ATR_SL_MULTIPLIER" not in non_finite_keys and atr_mult <= 0:
        _err("ATR_SL_MULTIPLIER", "out_of_bounds", f"ATR_SL_MULTIPLIER must be positive, got {atr_mult}")

    min_sl = _val("MIN_SL_PCT", 0.005)
    max_sl = _val("MAX_SL_PCT", 0.03)
    if "MIN_SL_PCT" not in non_finite_keys and min_sl <= 0:
        _err("MIN_SL_PCT", "out_of_bounds", f"MIN_SL_PCT must be positive, got {min_sl}")
    if "MIN_SL_PCT" not in non_finite_keys and "MAX_SL_PCT" not in non_finite_keys and max_sl <= min_sl:
        _err("MAX_SL_PCT", "cross_field_error", f"MAX_SL_PCT ({max_sl}) must be greater than MIN_SL_PCT ({min_sl})")

    etp_min_sl = _val("ETP_MIN_SL_PCT", 0.025)
    etp_max_sl = _val("ETP_MAX_SL_PCT", 0.080)
    if "ETP_MIN_SL_PCT" not in non_finite_keys and etp_min_sl <= 0:
        _err("ETP_MIN_SL_PCT", "out_of_bounds", f"ETP_MIN_SL_PCT must be positive, got {etp_min_sl}")
    if "ETP_MAX_SL_PCT" not in non_finite_keys and etp_max_sl > 1.0:
        _err("ETP_MAX_SL_PCT", "out_of_bounds", f"ETP_MAX_SL_PCT must be <= 1.0, got {etp_max_sl}")
    if (
        "ETP_MIN_SL_PCT" not in non_finite_keys
        and "ETP_MAX_SL_PCT" not in non_finite_keys
        and etp_max_sl <= etp_min_sl
    ):
        _err(
            "ETP_MAX_SL_PCT",
            "cross_field_error",
            f"ETP_MAX_SL_PCT ({etp_max_sl}) must be greater than ETP_MIN_SL_PCT ({etp_min_sl})",
        )

    tp1 = _val("TP1_R_MULTIPLE", 1.5)
    tp2 = _val("TP2_R_MULTIPLE", 2.5)
    tp3 = _val("TP3_R_MULTIPLE", 3.5)
    if (
        "TP1_R_MULTIPLE" not in non_finite_keys
        and "TP2_R_MULTIPLE" not in non_finite_keys
        and "TP3_R_MULTIPLE" not in non_finite_keys
    ):
        if tp1 <= 0:
            _err("TP1_R_MULTIPLE", "out_of_bounds", f"TP1_R_MULTIPLE must be positive, got {tp1}")
        if not (tp1 < tp2 < tp3):
            _err(
                "TP2_R_MULTIPLE",
                "cross_field_error",
                f"Take-profit multiples must satisfy TP1 < TP2 < TP3, got TP1={tp1}, TP2={tp2}, TP3={tp3}",
            )

    tsl_atr = _val("TSL_ATR_MULTIPLIER", 1.2)
    if "TSL_ATR_MULTIPLIER" not in non_finite_keys and tsl_atr <= 0:
        _err("TSL_ATR_MULTIPLIER", "out_of_bounds", f"TSL_ATR_MULTIPLIER must be positive, got {tsl_atr}")

    hyst = _val("TSL_HYSTERESIS_ATR_MULT", 0.5)
    if "TSL_HYSTERESIS_ATR_MULT" not in non_finite_keys and hyst <= 0:
        _err("TSL_HYSTERESIS_ATR_MULT", "out_of_bounds", f"TSL_HYSTERESIS_ATR_MULT must be positive, got {hyst}")

    decay_day1 = _val("TSL_DECAY_DAY1_MULT", 2.20)
    decay_step = _val("TSL_DECAY_STEP_MULT", 0.35)
    decay_floor = _val("TSL_DECAY_FLOOR_MULT", 1.15)
    decay_runner = _val("TSL_DECAY_RUNNER_MULT", 2.00)
    decay_keys = {"TSL_DECAY_DAY1_MULT", "TSL_DECAY_STEP_MULT", "TSL_DECAY_FLOOR_MULT", "TSL_DECAY_RUNNER_MULT"}
    if not (decay_keys & non_finite_keys):
        if decay_floor <= 0:
            _err("TSL_DECAY_FLOOR_MULT", "out_of_bounds", f"TSL_DECAY_FLOOR_MULT must be positive, got {decay_floor}")
        if decay_step < 0:
            _err("TSL_DECAY_STEP_MULT", "out_of_bounds", f"TSL_DECAY_STEP_MULT must be >= 0, got {decay_step}")
        if decay_day1 < decay_floor:
            _err(
                "TSL_DECAY_DAY1_MULT",
                "cross_field_error",
                f"TSL_DECAY_DAY1_MULT ({decay_day1}) cannot be below TSL_DECAY_FLOOR_MULT ({decay_floor})",
            )
        if decay_runner < decay_floor:
            _err(
                "TSL_DECAY_RUNNER_MULT",
                "cross_field_error",
                f"TSL_DECAY_RUNNER_MULT ({decay_runner}) cannot be below TSL_DECAY_FLOOR_MULT ({decay_floor})",
            )

    t1 = _val("TP1_TRIM_PCT", 0.50)
    t2 = _val("TP2_TRIM_PCT", 0.25)
    t3 = _val("TP3_TRIM_PCT", 0.25)
    if not ({"TP1_TRIM_PCT", "TP2_TRIM_PCT", "TP3_TRIM_PCT"} & non_finite_keys):
        for tk, tv in (("TP1_TRIM_PCT", t1), ("TP2_TRIM_PCT", t2), ("TP3_TRIM_PCT", t3)):
            if not (0.0 <= tv <= 1.0):
                _err(tk, "out_of_bounds", f"{tk} must be in [0.0, 1.0], got {tv}")
        total_trim = round(t1 + t2 + t3, 4)
        if total_trim != 1.0:
            _err(
                "TP1_TRIM_PCT",
                "cross_field_error",
                f"Trim percentages must sum to 1.0, got {total_trim} (TP1={t1}, TP2={t2}, TP3={t3})",
            )

    universe_size = _val("UNIVERSE_SIZE", 100)
    if not isinstance(universe_size, int) or isinstance(universe_size, bool) or universe_size <= 0:
        _err("UNIVERSE_SIZE", "out_of_bounds", f"UNIVERSE_SIZE must be positive, got {universe_size}")

    max_cat = _val("SCREENER_MAX_CATALYST_RISK", 0.25)
    if "SCREENER_MAX_CATALYST_RISK" not in non_finite_keys and not (0.0 <= max_cat <= 1.0):
        _err(
            "SCREENER_MAX_CATALYST_RISK",
            "out_of_bounds",
            f"SCREENER_MAX_CATALYST_RISK must be in [0.0, 1.0], got {max_cat}",
        )

    top_n = _val("SCREENER_TOP_N", 15)
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
        _err("SCREENER_TOP_N", "out_of_bounds", f"SCREENER_TOP_N must be positive, got {top_n}")

    min_setup = _val("SCREENER_MIN_SETUP_SCORE", "favorable")
    if min_setup not in _VALID_SETUP_SCORES:
        _err(
            "SCREENER_MIN_SETUP_SCORE",
            "invalid_value",
            f"SCREENER_MIN_SETUP_SCORE must be one of {_VALID_SETUP_SCORES}, got {min_setup!r}",
        )

    cutoff = _val("SESSION_CLOSE_CUTOFF_UTC", "16:20")
    if not isinstance(cutoff, str) or not _TIME_HHMM_RE.match(cutoff.strip()):
        _err(
            "SESSION_CLOSE_CUTOFF_UTC",
            "invalid_format",
            f"SESSION_CLOSE_CUTOFF_UTC must be HH:MM (UTC 24h), got {cutoff!r}",
        )

    enable_5x = bool(_val("ENABLE_5X_LEVERAGE", False))
    prefer_5x = bool(_val("PREFER_5X_LEVERAGE", False))
    max_lev = _val("MAX_LEVERAGE", 3)
    if isinstance(max_lev, bool) or not isinstance(max_lev, int) or max_lev not in _VALID_LEVERAGE_CAPS:
        _err("MAX_LEVERAGE", "out_of_bounds", f"MAX_LEVERAGE must be one of {_VALID_LEVERAGE_CAPS}, got {max_lev!r}")
    else:
        if enable_5x and max_lev < 5:
            _err(
                "MAX_LEVERAGE",
                "leverage_conflict",
                f"ENABLE_5X_LEVERAGE=True conflicts with MAX_LEVERAGE={max_lev} (must be 5)",
            )
        if not enable_5x and max_lev > 3:
            _err(
                "MAX_LEVERAGE",
                "leverage_conflict",
                f"ENABLE_5X_LEVERAGE=False conflicts with MAX_LEVERAGE={max_lev} (must be <= 3)",
            )
    if prefer_5x and not enable_5x:
        _err(
            "PREFER_5X_LEVERAGE",
            "leverage_conflict",
            "PREFER_5X_LEVERAGE=True requires ENABLE_5X_LEVERAGE=True",
        )

    return errors


@dataclass
class Settings:
    """Central configuration settings for the hybrid quantitative trading engine.

    Attributes:
        TRADING212_API_KEY: Trading 212 API key (loaded from TRADING212_API_KEY env var).
        TRADING212_API_SECRET: Trading 212 API secret (loaded from TRADING212_API_SECRET env var).
        TRADING212_BASE_URL: Trading 212 Demo API base URL.
        TRADING212_LIVE_URL: Trading 212 Live API base URL.
        USE_LIVE: Whether to trade in live environment (False = demo for safety).
        OPENAI_API_KEY: OpenAI API key (loaded from OPENAI_API_KEY env var).
        POLYGON_API_KEY: Polygon.io API key (loaded from POLYGON_API_KEY env var).
        FINNHUB_API_KEY: Finnhub API key (loaded from FINNHUB_API_KEY env var).
        LLM_DEEP_MODEL: LLM model name for deep reasoning debate.
        LLM_QUICK_MODEL: LLM model name for rapid analysis.
        MAX_RISK_PER_TRADE: Maximum fractional equity risk per trade (e.g. 0.005 = 0.5%).
        MAX_POSITION_PCT: Maximum fractional equity notional per position (e.g. 0.05 = 5%).
        ATR_SL_MULTIPLIER: Multiplier for ATR when computing stop-loss distance.
        MIN_SL_PCT: Minimum stop-loss percentage distance from entry price.
        TP1_R_MULTIPLE: Risk-multiple target for Take Profit 1.
        TP2_R_MULTIPLE: Risk-multiple target for Take Profit 2.
        TP3_R_MULTIPLE: Risk-multiple target for Take Profit 3.
        SCREENER_MIN_SETUP_SCORE: Minimum acceptable setup quality for candidate screening.
        SCREENER_MAX_CATALYST_RISK: Maximum acceptable catalyst-risk probability.
        SCREENER_TOP_N: Number of top-ranked candidates to select from screener.
        LOG_LEVEL: Logging level for application (INFO, DEBUG, etc.).
        TRADES_LOG_PATH: Filepath for persisting trade records in JSONL format.
        UNIVERSE_SIZE: Maximum number of candidate assets in trading universe.
        WATCHLIST: Optional list of tickers to override universe scanning.
    """

    # Broker settings
    TRADING212_API_KEY: str = ""
    TRADING212_API_SECRET: str = ""
    TRADING212_BASE_URL: str = "https://demo.trading212.com/api/v0"
    TRADING212_LIVE_URL: str = "https://live.trading212.com/api/v0"
    USE_LIVE: bool = False
    DEFAULT_ACCOUNT_EQUITY: float = 10000.0

    # AI / LLM API settings
    OPENAI_API_KEY: str = ""
    LLM_DEEP_MODEL: str = "gpt-4o"
    LLM_QUICK_MODEL: str = "gpt-4o-mini"

    # Parallel Concurrency & Deliberative Multi-Agent Architecture
    PARALLEL_INGESTION_WORKERS: int = 24
    PARALLEL_DELIBERATION_WORKERS: int = 8
    ENABLE_SPECULATIVE_ROUTING: bool = False  # Pure System 2: Fast reflex screening bypass completely disabled
    SPECULATIVE_CONFIDENCE_THRESHOLD: float = 0.70

    # Market Data API keys
    FMP_API_KEY: str = ""
    TIINGO_API_KEY: str = ""
    POLYGON_API_KEY: str = ""
    ALPHA_VANTAGE_API_KEY: str = ""
    FINNHUB_API_KEY: str = ""
    OLOSTEP_API_KEY: str = ""
    NIXTLA_API_KEY: str = ""

    # Risk parameters
    MAX_RISK_PER_TRADE: float = 0.015  # 1.5% of equity per trade (increased from 0.5% for fuller sizing)
    MAX_POSITION_PCT: float = 0.20  # 20% of equity max per position (increased from 5% for meaningful portfolio size)
    ATR_SL_MULTIPLIER: float = 2.2
    MIN_SL_PCT: float = 0.005  # 0.5% minimum stop distance
    MAX_SL_PCT: float = 0.03   # 3.0% maximum stop distance (strict risk cap)
    ETP_MIN_SL_PCT: float = 0.025  # 2.5% minimum ETP stop-clamp distance
    ETP_MAX_SL_PCT: float = 0.080  # 8.0% maximum ETP stop-clamp distance
    TP1_R_MULTIPLE: float = 1.5
    TP2_R_MULTIPLE: float = 2.5
    TP3_R_MULTIPLE: float = 3.5
    TARGET_ANNUAL_VOLATILITY: float = 0.20  # 20% annual target volatility for volatility-targeted sizing

    # Screener parameters
    SCREENER_MIN_SETUP_SCORE: str = "favorable"
    SCREENER_MAX_CATALYST_RISK: float = 0.25
    SCREENER_TOP_N: int = 15
    SCREENER_MAX_PER_CATEGORY: int = 2
    MIN_ADX_TREND: float = 18.0  # Trend strength gate for 3x leverage (relaxed from 20.0 to catch trends earlier)
    REGIME_BREAKOUT_ADX_MIN: float = 25.0
    HIGH_VOL_MAX_REALIZED_VOL: float = 0.65
    HIGH_VOL_MAX_ATR_SPIKE_RATIO: float = 1.50
    FALLBACK_SCREENER_MIN_ADX: float = 20.0

    # Quantitative Strategy (VWAP Session Trend with Multi-Timeframe Gating & Dual-Track Short Execution)
    ACTIVE_STRATEGY: str = "vwap"
    STRATEGY_EXCEPTIONAL_ADX: float = 25.0
    MIN_EMA20_SLOPE_PCT: float = 0.0
    ENTRY_EARNINGS_BLACKOUT_DAYS: float = 2.0
    STRATEGY_BASE_CONVICTION: float = 0.70
    STRATEGY_SHORT_BASE_CONVICTION: float = 0.75
    STRATEGY_MACRO_TAILWIND_BONUS: float = 0.10
    DIRECTIONAL_MAX_CONVICTION: float = 0.95
    MIN_VWAP_DIST_PCT: float = 0.1
    MAX_VWAP_DIST_PCT: float = 3.5
    MAX_SWING_HOLD_DAYS: int = 10
    MR_MAX_HOLD_DAYS: int = 10  # Backward compatibility alias

    # Trailing Stop Loss & Schmitt Hysteresis Ratcheting
    PURE_TSL_MODE: bool = True            # Pure 100% Trailing Stop Loss mode (no partial TP trims, lets winners run fully)
    TSL_ATR_MULTIPLIER: float = 1.8       # Widened to 1.8 ATR to eliminate premature runner choke
    TSL_HYSTERESIS_ATR_MULT: float = 0.5  # Escape buffer past TP1 before elevating stop to TP1
    TSL_DECAY_DAY1_MULT: float = 2.20     # Day 1 session ATR trail multiplier
    TSL_DECAY_STEP_MULT: float = 0.35     # Per-session ATR trail compression step
    TSL_DECAY_FLOOR_MULT: float = 1.15    # Minimum compressed ATR trail floor
    TSL_DECAY_RUNNER_MULT: float = 2.00   # Frozen ATR trail multiplier for tenured runners
    TP1_TRIM_PCT: float = 0.50            # Upfront cash profit trim (50%) - legacy
    TP2_TRIM_PCT: float = 0.25            # Second milestone trim (25%) - legacy
    TP3_TRIM_PCT: float = 0.25            # Remaining runner tranche (25%) - legacy

    # Asymmetric Session Protocol & API Rate Quota Management
    SESSION_CLOSE_CUTOFF_UTC: str = "16:20"  # 10m before 16:30 LSE close
    ENFORCE_INVERSE_INTRADAY_ONLY: bool = True
    FLATTEN_ALL_LEVERAGED_AT_SESSION_CLOSE: bool = False  # Set True to force 100% intraday liquidation of all 3x/5x ETPs at 16:20 UTC
    ENFORCE_LONG_BREAKEVEN_OVERNIGHT: bool = True
    MIN_OVERNIGHT_PROFIT_CUSHION_PCT: float = 0.030      # 3.0% minimum profit cushion if long swing overnight is allowed (resists 45 bps friction)
    EARNINGS_BLACKOUT_HOURS: int = 24
    RATE_LIMIT_INTER_CALL_DELAY: float = 1.25  # seconds between Trading 212 API calls (strictly obeys 1 req/sec quota)

    # Advanced Risk & Safety Gates
    ENFORCE_SPREAD_GATE: bool = True
    MAX_BID_ASK_SPREAD_PCT: float = 0.0150    # 1.50% max relative spread (adapted for European 3x leveraged ETPs)
    MAX_SPREAD_TO_STOP_RATIO: float = 0.50    # Spread must not exceed 50% of the stop loss distance
    TRADE_LOSS_COOLDOWN_SECONDS: float = 3600.0  # 1 hour cooldown after a stop-out to prevent churn/re-entry whipsaw
    MIN_HOLD_SECONDS: float = 120.0           # 2 minutes minimum hold time to absorb entry bid-ask bounce
    ENFORCE_CORRELATION_LIMITS: bool = True
    MAX_SECTOR_POSITIONS: int = 4             # Max concurrent positions per sector cluster (increased from 2)
    MAX_PORTFOLIO_BETA_HEAT: float = 6.0      # Max aggregate portfolio beta exposure (increased from 3.0 to allow 4-6 concurrent 3x positions)
    ENFORCE_DAILY_CIRCUIT_BREAKER: bool = True
    DAILY_MAX_DRAWDOWN_PCT: float = 0.02      # 2.0% max daily portfolio drawdown

    # Speed & Cache Optimizations
    ENABLE_CANDLE_CACHE: bool = True
    CANDLE_CACHE_TTL_SECONDS: int = 300       # 5 minutes cache TTL

    # Broker Reconciliation & Safety Kill Switch
    RECONCILIATION_INTERVAL_SECONDS: int = 60
    PANIC_FLATTEN_TOKEN: str = ""
    DASHBOARD_AUTH_TOKEN: str = ""

    # Notifications
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    DISCORD_WEBHOOK_URL: str = ""
    ENABLE_DESKTOP_NOTIFICATIONS: bool = True
    NOTIFICATIONS_LOG_PATH: str = "notifications.jsonl"
    ACTIVE_TRADES_PATH: str = "active_trades.json"
    CANDLE_DB_PATH: str = "data/candles.db"

    # Embedded Dashboard Server
    ENABLE_EMBEDDED_DASHBOARD: bool = True
    DASHBOARD_PORT: int = 8888

    # Universe & Mode
    USE_LEVERAGED_3X: bool = True
    ENABLE_CRYPTO_3X: bool = True
    ENABLE_COMMODITIES_3X: bool = True
    ENABLE_5X_LEVERAGE: bool = False  # 5x replaced with 3x per user requirements
    PREFER_5X_LEVERAGE: bool = False
    MAX_LEVERAGE: int = 3  # Maximum leverage capped at 3x
    LOG_LEVEL: str = "INFO"
    TRADES_LOG_PATH: str = "trades.jsonl"
    UNIVERSE_SIZE: int = 100
    WATCHLIST: Optional[List[str]] = field(default=None)

    def __post_init__(self) -> None:
        """Load configuration from environment variables if not explicitly provided."""
        secret_keys = (
            "TRADING212_API_KEY",
            "TRADING212_API_SECRET",
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
            "PANIC_FLATTEN_TOKEN",
            "DASHBOARD_AUTH_TOKEN",
        )
        for key in secret_keys:
            if not getattr(self, key):
                setattr(self, key, os.environ.get(key, ""))

        env_use_live = os.environ.get("USE_LIVE")
        if env_use_live is not None and not self.USE_LIVE:
            self.USE_LIVE = _parse_bool_env(env_use_live)

        str_fields = (
            ("TRADING212_BASE_URL", False),
            ("TRADING212_LIVE_URL", False),
            ("LLM_DEEP_MODEL", False),
            ("LLM_QUICK_MODEL", False),
            ("LOG_LEVEL", True),
            ("TRADES_LOG_PATH", False),
            ("NOTIFICATIONS_LOG_PATH", False),
            ("ACTIVE_TRADES_PATH", False),
            ("CANDLE_DB_PATH", False),
            ("SCREENER_MIN_SETUP_SCORE", False),
            ("ACTIVE_STRATEGY", False),
            ("SESSION_CLOSE_CUTOFF_UTC", False),
        )
        for key, upper in str_fields:
            raw = os.environ.get(key)
            if raw and raw.strip():
                val = raw.strip().upper() if upper else raw.strip()
                if key == "SCREENER_MIN_SETUP_SCORE":
                    val = val.lower()
                setattr(self, key, val)

        float_fields = (
            "DEFAULT_ACCOUNT_EQUITY",
            "SPECULATIVE_CONFIDENCE_THRESHOLD",
            "MAX_RISK_PER_TRADE",
            "MAX_POSITION_PCT",
            "ATR_SL_MULTIPLIER",
            "MIN_SL_PCT",
            "MAX_SL_PCT",
            "ETP_MIN_SL_PCT",
            "ETP_MAX_SL_PCT",
            "TP1_R_MULTIPLE",
            "TP2_R_MULTIPLE",
            "TP3_R_MULTIPLE",
            "TARGET_ANNUAL_VOLATILITY",
            "SCREENER_MAX_CATALYST_RISK",
            "MIN_ADX_TREND",
            "STRATEGY_EXCEPTIONAL_ADX",
            "MIN_EMA20_SLOPE_PCT",
            "ENTRY_EARNINGS_BLACKOUT_DAYS",
            "STRATEGY_BASE_CONVICTION",
            "STRATEGY_SHORT_BASE_CONVICTION",
            "STRATEGY_MACRO_TAILWIND_BONUS",
            "DIRECTIONAL_MAX_CONVICTION",
            "MIN_VWAP_DIST_PCT",
            "MAX_VWAP_DIST_PCT",
            "REGIME_BREAKOUT_ADX_MIN",
            "HIGH_VOL_MAX_REALIZED_VOL",
            "HIGH_VOL_MAX_ATR_SPIKE_RATIO",
            "FALLBACK_SCREENER_MIN_ADX",
            "TSL_ATR_MULTIPLIER",
            "TSL_HYSTERESIS_ATR_MULT",
            "TSL_DECAY_DAY1_MULT",
            "TSL_DECAY_STEP_MULT",
            "TSL_DECAY_FLOOR_MULT",
            "TSL_DECAY_RUNNER_MULT",
            "TP1_TRIM_PCT",
            "TP2_TRIM_PCT",
            "TP3_TRIM_PCT",
            "MIN_OVERNIGHT_PROFIT_CUSHION_PCT",
            "RATE_LIMIT_INTER_CALL_DELAY",
            "MAX_BID_ASK_SPREAD_PCT",
            "MAX_SPREAD_TO_STOP_RATIO",
            "TRADE_LOSS_COOLDOWN_SECONDS",
            "MIN_HOLD_SECONDS",
            "MAX_PORTFOLIO_BETA_HEAT",
            "DAILY_MAX_DRAWDOWN_PCT",
        )
        for key in float_fields:
            raw = os.environ.get(key)
            if raw and raw.strip():
                try:
                    parsed = float(raw.strip())
                    if math.isfinite(parsed):
                        setattr(self, key, parsed)
                except ValueError:
                    pass

        is_testing = "unittest" in sys.modules or bool(os.environ.get("PYTEST_CURRENT_TEST"))
        if not is_testing and not os.environ.get("DEFAULT_ACCOUNT_EQUITY"):
            cached_cash_p = Path("data/cached_cash.json")
            if cached_cash_p.is_file():
                try:
                    p_c = json.loads(cached_cash_p.read_text(encoding="utf-8"))
                    c_tot = float(p_c.get("total") or 0.0)
                    if c_tot > 0 and math.isfinite(c_tot):
                        self.DEFAULT_ACCOUNT_EQUITY = c_tot
                except Exception:
                    pass

        int_fields = (
            "PARALLEL_INGESTION_WORKERS",
            "PARALLEL_DELIBERATION_WORKERS",
            "NIXTLA_FORECAST_HORIZON",
            "SCREENER_TOP_N",
            "SCREENER_MAX_PER_CATEGORY",
            "MAX_SWING_HOLD_DAYS",
            "MR_MAX_HOLD_DAYS",
            "EARNINGS_BLACKOUT_HOURS",
            "MAX_SECTOR_POSITIONS",
            "CANDLE_CACHE_TTL_SECONDS",
            "RECONCILIATION_INTERVAL_SECONDS",
            "DASHBOARD_PORT",
            "UNIVERSE_SIZE",
        )
        for key in int_fields:
            raw = os.environ.get(key)
            if raw and raw.strip():
                try:
                    setattr(self, key, int(raw.strip()))
                except ValueError:
                    pass

        bool_fields = (
            "ENABLE_SPECULATIVE_ROUTING",
            "PURE_TSL_MODE",
            "ENFORCE_INVERSE_INTRADAY_ONLY",
            "ENFORCE_LONG_BREAKEVEN_OVERNIGHT",
            "ENFORCE_SPREAD_GATE",
            "ENFORCE_CORRELATION_LIMITS",
            "ENFORCE_DAILY_CIRCUIT_BREAKER",
            "ENABLE_CANDLE_CACHE",
            "ENABLE_DESKTOP_NOTIFICATIONS",
            "ENABLE_EMBEDDED_DASHBOARD",
            "USE_LEVERAGED_3X",
            "ENABLE_CRYPTO_3X",
            "ENABLE_COMMODITIES_3X",
            "PREFER_5X_LEVERAGE",
        )
        for key in bool_fields:
            raw = os.environ.get(key)
            if raw is not None and raw.strip():
                setattr(self, key, _parse_bool_env(raw))

        env_5x = os.environ.get("ENABLE_5X_LEVERAGE")
        if env_5x is not None and env_5x.strip():
            self.ENABLE_5X_LEVERAGE = _parse_bool_env(env_5x)

        env_max_lev = os.environ.get("MAX_LEVERAGE")
        if env_max_lev is not None and env_max_lev.strip():
            try:
                self.MAX_LEVERAGE = int(env_max_lev.strip())
            except ValueError:
                pass
        elif (env_5x is not None or self.ENABLE_5X_LEVERAGE) and self.MAX_LEVERAGE == 3:
            self.MAX_LEVERAGE = 5 if self.ENABLE_5X_LEVERAGE else 3

        if self.WATCHLIST is None:
            env_watchlist = os.environ.get("WATCHLIST")
            if env_watchlist:
                self.WATCHLIST = [s.strip() for s in env_watchlist.split(",") if s.strip()]

        is_testing = "unittest" in sys.modules or any("test" in arg.lower() for arg in sys.argv)
        env_flatten_all = os.environ.get("FLATTEN_ALL_LEVERAGED_AT_SESSION_CLOSE")
        if env_flatten_all is not None and env_flatten_all.strip() and not is_testing:
            self.FLATTEN_ALL_LEVERAGED_AT_SESSION_CLOSE = _parse_bool_env(env_flatten_all)

    @property
    def api_base_url(self) -> str:
        """Return the active Trading 212 API base URL based on the USE_LIVE flag.

        Returns:
            str: TRADING212_LIVE_URL if USE_LIVE is True, otherwise TRADING212_BASE_URL.
        """
        return self.TRADING212_LIVE_URL if self.USE_LIVE else self.TRADING212_BASE_URL

    def validate(
        self,
        required_keys: Optional[List[str]] = None,
        require_openai: bool = False,
    ) -> None:
        """Validate that required configuration keys and parameters are properly set.

        Args:
            required_keys: Optional list of attribute names that must be set and non-empty.
                Defaults to ['TRADING212_API_KEY']. To also require OpenAI or other keys,
                pass them in required_keys or set require_openai=True.
            require_openai: If True, adds 'OPENAI_API_KEY' to required keys check.

        Raises:
            ValueError: If any required key is missing/empty, or if parameter values are invalid.
        """
        keys_to_check: List[str] = list(required_keys) if required_keys is not None else ["TRADING212_API_KEY"]
        if require_openai and "OPENAI_API_KEY" not in keys_to_check:
            keys_to_check.append("OPENAI_API_KEY")

        missing: List[str] = [
            k for k in keys_to_check
            if not getattr(self, k, None) or not str(getattr(self, k)).strip()
        ]
        if missing:
            raise ValueError(
                f"Missing required configuration key(s): {', '.join(missing)}. "
                f"Please ensure they are set in environment variables or Settings."
            )

        invariant_errors = collect_settings_invariant_errors(self)
        if invariant_errors:
            raise ValueError(invariant_errors[0]["message"])


def atomic_write_text(target_path: Union[str, Path], content: str) -> None:
    """Write text atomically using a thread-unique temp file, flush, fsync, and os.replace."""
    resolved = os.path.abspath(os.fspath(target_path))
    parent = os.path.dirname(resolved) or "."
    os.makedirs(parent, exist_ok=True)
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=f".{os.path.basename(resolved)}.",
            suffix=f".{os.getpid()}.{threading.get_ident()}.tmp",
            delete=False,
        ) as handle:
            tmp_path = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, resolved)
        tmp_path = None
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# Module-level singleton instance
settings = Settings()
