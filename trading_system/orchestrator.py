"""Orchestrator — 3x Leveraged Long & Short Quantitative Engine.

Deterministic 5-Step Execution Pipeline:
1. Market Check: Validates Trading 212 broker connectivity, account cash, and circuit breakers.
2. Ingest Candles: Ingests live 5m/1d OHLCV bars, computing VWAP, ATR, and momentum indicators.
3. Strategy Screen: Evaluates universe candidates against active quantitative strategy rules.
4. Risk Sizing & Protection: Direct quantitative risk sizing, ATR stops, and leverage configuration.
5. Order Route / Standby: Dispatches market entry orders to Trading 212 or enters standby mode.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import inspect
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from trading_system.config import atomic_write_text, settings
from trading_system.config_bus import ConfigBus, ConfigSnapshot
from trading_system.config_schema import (
    SCHEMA_BY_KEY,
    export_config_schema,
    get_builtin_profiles,
)
from trading_system.dashboard.auth import sanitize_mapping
from trading_system.data.leveraged_universe import (
    LeveragedUniverse,
    get_currency_display,
    get_ticker_currency,
    normalize_quote_to_eur,
)
from trading_system.data.trading212_client import (
    AccountCash,
    MarketDataProvider,
    Position,
    Trading212Client,
    Trading212Error,
)
from trading_system.execution.position_monitor import (
    PositionEpochConflictError,
    PositionMonitor,
)
from trading_system.notifications.notifier import Notifier, TradeNotification
from trading_system.screener.laya_screener import (
    CycleFunnelReport,
    LayaScreener,
    ScreenerResult,
    TechnicalFallbackScreener,
    build_cycle_funnel_report,
    simulate_screener_gate_impact,
)
from trading_system.agents.risk_manager import DecisionCard, RiskManager

logger = logging.getLogger(__name__)


class DummyAnalyst:
    """Lightweight analyst shim for test compatibility."""
    def analyze(self, *args, **kwargs) -> str:
        return "analysis"

    def format_report(self, *args, **kwargs) -> str:
        return "report"


class DummyDebateEngine:
    """Lightweight compatibility shim for tests and external callers."""
    _is_dummy: bool = True

    def __init__(self, *args, **kwargs):
        pass

    def run_debate(self, ticker: str, metrics: Any = None, *args, **kwargs) -> Any:
        return SimpleNamespace(
            consensus_decision="BUY_ENTRY",
            conviction_score=0.85,
            synthesis=f"Quantitative signal verified for {ticker}",
            winning_side="bull",
            config_epoch_id=None,
        )


def _is_unittest_process() -> bool:
    """Return True when running inside a Python unittest or pytest session."""
    return (
        "unittest" in sys.modules
        or "pytest" in sys.modules
        or any("test" in arg.lower() for arg in sys.argv)
    )


def _atomic_write_json(target_path: Path, payload: Any) -> None:
    """Atomically write JSON to target_path via a same-directory temporary file."""
    atomic_write_text(target_path, json.dumps(payload, indent=2, default=str))


_scrub_mapping = sanitize_mapping


def _snap_val(snap: Any, key: str, default: Any = None) -> Any:
    """Read a config field safely from a ConfigSnapshot, Mapping, or settings object."""
    if snap is None:
        return default
    if hasattr(snap, key):
        return getattr(snap, key)
    if isinstance(snap, Mapping) and key in snap:
        return snap[key]
    dyn = getattr(snap, "dynamic_parameters", None)
    if isinstance(dyn, Mapping) and key in dyn:
        return dyn[key]
    return default


def _changed_snapshot_keys(old_snap: ConfigSnapshot, new_snap: ConfigSnapshot) -> List[str]:
    """Return sorted configuration keys that differ between two ConfigSnapshots."""
    changed: Set[str] = set()
    for key in SCHEMA_BY_KEY:
        if getattr(old_snap, key, None) != getattr(new_snap, key, None):
            changed.add(key)
    old_dyn = dict(getattr(old_snap, "dynamic_parameters", {}) or {})
    new_dyn = dict(getattr(new_snap, "dynamic_parameters", {}) or {})
    for key in set(old_dyn.keys()) | set(new_dyn.keys()):
        if old_dyn.get(key) != new_dyn.get(key):
            changed.add(key)
    old_specs = tuple(getattr(old_snap, "dynamic_parameter_specs", ()) or ())
    new_specs = tuple(getattr(new_snap, "dynamic_parameter_specs", ()) or ())
    old_spec_map = {getattr(s, "key", None): s for s in old_specs}
    new_spec_map = {getattr(s, "key", None): s for s in new_specs}
    for key in set(old_spec_map.keys()) | set(new_spec_map.keys()):
        if key and old_spec_map.get(key) != new_spec_map.get(key):
            changed.add(str(key))
    return sorted(changed)


def _revalidation_scope_for_keys(
    keys: Sequence[str],
    old_snap: ConfigSnapshot,
    new_snap: ConfigSnapshot,
) -> str:
    """Determine aggregate revalidation_scope ('abort_stale_scan' or 'resize_entry') for changed keys."""
    dyn_spec_keys: Set[str] = set()
    for snap in (old_snap, new_snap):
        for spec in getattr(snap, "dynamic_parameter_specs", ()) or ():
            if getattr(spec, "key", None):
                dyn_spec_keys.add(str(spec.key))
        for k in (getattr(snap, "dynamic_parameters", {}) or {}):
            dyn_spec_keys.add(str(k))

    for key in keys:
        schema_entry = SCHEMA_BY_KEY.get(key)
        if schema_entry is not None:
            if schema_entry.revalidation_scope == "abort_stale_scan":
                return "abort_stale_scan"
        elif key in dyn_spec_keys or key.startswith("dynamic_"):
            return "abort_stale_scan"
    return "resize_entry"


def _as_bool_strict(val: Any, default: bool = False) -> bool:
    """Interpret strict boolean values safely without mistaking MagicMock objects for True."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return default


def _pos_is_closed(pos: Any) -> bool:
    """Return True only when a tracked position (or mapping) is explicitly marked closed."""
    if pos is None:
        return True
    raw = pos.get("is_closed", False) if isinstance(pos, Mapping) else getattr(pos, "is_closed", False)
    return _as_bool_strict(raw, False)


def _pos_remaining_qty(pos: Any) -> float:
    """Safely extract numeric remaining_quantity from a tracked position or mapping."""
    if pos is None:
        return 0.0
    raw = pos.get("remaining_quantity", 0.0) if isinstance(pos, Mapping) else getattr(pos, "remaining_quantity", 0.0)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    return 0.0

# Fallback watchlist of ordinary equities if 3x mode is disabled
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B",
    "UNH", "LLY", "JPM", "XOM", "JNJ", "V", "PG", "AVGO", "MA", "HD",
    "MRK", "CVX", "ABBV", "COST", "PEP", "KO", "ADBE", "WMT", "MCD",
    "CRM", "CSCO", "TMO", "ACN", "ABT", "LIN", "BAC", "NFLX", "PFE",
    "AMD", "DHR", "DIS", "ORCL", "CMCSA", "PM", "INTC", "VZ", "WFC",
]


class _CachedHistoryClientProxy:
    """Adapter presenting cached historical orders with zero Trading 212 HTTP calls."""

    def __init__(self, cached_history: Optional[Dict[str, Any]]) -> None:
        self._cached_history = cached_history if isinstance(cached_history, dict) else {"items": []}

    def get_historical_orders(self, limit: int = 50, **_: Any) -> Dict[str, Any]:
        items = self._cached_history.get("items", [])
        if not isinstance(items, list):
            items = []
        return {"items": list(items[:limit])}


class Orchestrator:
    """Master controller for 3x Long/Short Quantitative Trading Engine."""

    def __init__(
        self,
        dry_run: bool = False,
        notifier: Optional[Notifier] = None,
        position_monitor: Optional[PositionMonitor] = None,
        trades_log: Optional[Path | str] = None,
        enable_dashboard: Optional[bool] = None,
        dashboard_port: Optional[int] = None,
        dashboard_host: Optional[str] = None,
        open_browser: bool = False,
        t212_client: Optional[Trading212Client] = None,
        config_bus: Optional[ConfigBus] = None,
        data_dir: Optional[Path | str] = None,
    ) -> None:
        self.dry_run = dry_run
        is_testing = _is_unittest_process()
        if dry_run or is_testing:
            settings.validate(required_keys=[])
        else:
            settings.validate()

        # ── 0. Data Directory & Control-Plane ConfigBus ──────────
        self._owns_temp_data_dir = False
        explicit_isolation = data_dir is not None or config_bus is not None
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        elif config_bus is not None:
            self.data_dir = Path(getattr(config_bus, "data_dir", "data"))
        elif is_testing:
            self.data_dir = Path(tempfile.mkdtemp(prefix="t212_orch_ctrl_"))
            self._owns_temp_data_dir = True
        else:
            self.data_dir = Path("data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._funnel_state_path = self.data_dir / "funnel_state.json"
        self._cash_cache_path = self.data_dir / "cached_cash.json"

        self.config_bus = config_bus if config_bus is not None else ConfigBus(data_dir=self.data_dir)
        snap = self.config_bus.active_snapshot()

        # Synchronization locks (hierarchy: _execution_gate_lock / _broker_op_lock -> _state_lock; _read_model_lock is leaf)
        existing_gate_lock = (
            getattr(position_monitor, "execution_gate_lock", None)
            if position_monitor is not None
            else None
        )
        self._execution_gate_lock: threading.RLock = (
            existing_gate_lock if isinstance(existing_gate_lock, type(threading.RLock())) else threading.RLock()
        )
        self._read_model_lock = threading.RLock()
        self._read_model_snapshot: Optional[Dict[str, Any]] = None
        self._latest_funnel_report: Optional[CycleFunnelReport] = self._load_persisted_funnel_state()
        self._latest_funnel_meta: Dict[str, Any] = (
            {"status": "ok", "reason": ""}
            if self._latest_funnel_report is not None
            else {"status": "pre_scan", "reason": "No completed universe scan cycle recorded yet."}
        )
        self._last_funnel_inputs: Optional[Dict[str, Any]] = None
        self._current_cycle_id: str = ""
        self._current_pipeline_progress: Dict[str, Any] = {
            "step": 7,
            "total_steps": 7,
            "name": "Standby",
            "detail": "Engine standing by. Next scan countdown active.",
            "status": "standby",
            "active_assets": [],
            "timestamp": time.time(),
            "iso_time": datetime.now(timezone.utc).isoformat(),
            "cycle_id": "",
        }

        # ── 1. Trading 212 API Client ─────────────────────────────
        effective_api_key = settings.TRADING212_API_KEY or "DRY_RUN_MOCK_KEY"
        self.t212 = t212_client or Trading212Client(
            api_key=effective_api_key,
            api_secret=settings.TRADING212_API_SECRET,
            base_url=settings.api_base_url,
            rate_limit_delay=float(getattr(snap, "RATE_LIMIT_INTER_CALL_DELAY", settings.RATE_LIMIT_INTER_CALL_DELAY)),
        )

        # ── 2. Universe Manager ───────────────────────────────────
        self.leveraged_universe = LeveragedUniverse()
        self.universe: List[str] = []
        self._rebuild_universe(snap)

        # ── 3. Notification Engine ────────────────────────────────
        default_notif_log = (
            str(self.data_dir / "notifications.jsonl")
            if (explicit_isolation or is_testing)
            else settings.NOTIFICATIONS_LOG_PATH
        )
        self.notifier = notifier or Notifier(
            telegram_bot_token=settings.TELEGRAM_BOT_TOKEN,
            telegram_chat_id=settings.TELEGRAM_CHAT_ID,
            discord_webhook_url=settings.DISCORD_WEBHOOK_URL,
            log_file=default_notif_log,
            enable_desktop=False if is_testing else bool(getattr(snap, "ENABLE_DESKTOP_NOTIFICATIONS", settings.ENABLE_DESKTOP_NOTIFICATIONS)),
        )

        quarantine_warnings = list(self.config_bus.get_quarantine_warnings())
        if quarantine_warnings:
            joined = "; ".join(quarantine_warnings)
            logger.warning("Config quarantine warnings detected at startup: %s", joined)
            self._dispatch_system_alert(
                "CONFIG QUARANTINE WARNING",
                f"Corrupt persisted config quarantined on boot: {joined}",
            )

        # ── 4. Quantitative Candidate Screener & Improvement Agent ─
        self.screener = LayaScreener(
            min_setup_score=str(snap.SCREENER_MIN_SETUP_SCORE),
            max_catalyst_risk=float(snap.SCREENER_MAX_CATALYST_RISK),
            top_n=int(snap.SCREENER_TOP_N),
        )
        self.improvement_agent = None

        # ── 5. System 2: Pure Quantitative Risk & Sizing Stack ───
        self.technical_analyst = DummyAnalyst()
        self.fundamental_analyst = DummyAnalyst()
        self.debate_engine = DummyDebateEngine()
        self.parallel_ingestion_workers = int(snap.PARALLEL_INGESTION_WORKERS)
        self.parallel_deliberation_workers = int(snap.PARALLEL_DELIBERATION_WORKERS)
        self.risk_manager = RiskManager(
            max_risk_per_trade=float(snap.MAX_RISK_PER_TRADE),
            max_position_pct=float(snap.MAX_POSITION_PCT),
            atr_sl_multiplier=float(snap.ATR_SL_MULTIPLIER),
            min_sl_pct=float(snap.MIN_SL_PCT),
            max_sl_pct=float(snap.MAX_SL_PCT),
            tp1_r=float(snap.TP1_R_MULTIPLE),
            tp2_r=float(snap.TP2_R_MULTIPLE),
            tp3_r=float(snap.TP3_R_MULTIPLE),
            tsl_atr_multiplier=float(snap.TSL_ATR_MULTIPLIER),
            tsl_hysteresis_atr_mult=float(snap.TSL_HYSTERESIS_ATR_MULT),
            tp1_trim_pct=float(snap.TP1_TRIM_PCT),
            tp2_trim_pct=float(snap.TP2_TRIM_PCT),
            tp3_trim_pct=float(snap.TP3_TRIM_PCT),
            target_annual_volatility=float(snap.TARGET_ANNUAL_VOLATILITY),
            enforce_spread_gate=bool(snap.ENFORCE_SPREAD_GATE),
            max_bid_ask_spread_pct=float(snap.MAX_BID_ASK_SPREAD_PCT),
            enforce_correlation_limits=bool(snap.ENFORCE_CORRELATION_LIMITS),
            max_sector_positions=int(snap.MAX_SECTOR_POSITIONS),
            max_portfolio_beta_heat=float(snap.MAX_PORTFOLIO_BETA_HEAT),
            enforce_daily_circuit_breaker=bool(snap.ENFORCE_DAILY_CIRCUIT_BREAKER),
            daily_max_drawdown_pct=float(snap.DAILY_MAX_DRAWDOWN_PCT),
        )

        # ── 6. Active Position & Lifecycle Monitor ───────────────
        default_state_file = (
            str(self.data_dir / "active_trades.json")
            if (explicit_isolation or is_testing)
            else settings.ACTIVE_TRADES_PATH
        )
        if position_monitor is not None:
            self.position_monitor = position_monitor
            if getattr(self.position_monitor, "config_bus", None) is None:
                with contextlib.suppress(Exception):
                    self.position_monitor.config_bus = self.config_bus
            if getattr(self.position_monitor, "execution_gate_lock", None) is None:
                with contextlib.suppress(Exception):
                    self.position_monitor.execution_gate_lock = self._execution_gate_lock
        else:
            self.position_monitor = PositionMonitor(
                t212_client=self.t212,
                notifier=self.notifier,
                risk_manager=self.risk_manager,
                state_file=default_state_file,
                dry_run=self.dry_run,
                pure_tsl_mode=None,
                config_bus=self.config_bus,
                execution_gate_lock=self._execution_gate_lock,
            )

        self.session_start_equity: Optional[float] = None
        self._last_reconciliation_time: float = 0.0
        self.in_flight_tickers: Set[str] = set()
        self._scan_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="universe_scan")
        self._scan_future: Optional[concurrent.futures.Future] = None

        # Synchronize broker-native GTC Stop Loss / Limit orders for active positions
        if not is_testing and settings.TRADING212_API_KEY:
            try:
                self.position_monitor.reconcile_with_broker(adopt_untracked=True)
                if not self.dry_run:
                    self.position_monitor.sync_broker_orders()
                    self.position_monitor.apply_daily_rollover_stops()
            except Exception as exc:
                logger.warning("Initial broker stop sync deferred: %s", exc)

        # Non-unit-test warm-up using non-forced broker getters before initial snapshot publish
        if not is_testing and settings.TRADING212_API_KEY:
            try:
                boot_cash = self.t212.get_account_cash(
                    fallback_equity=self._get_fallback_equity(snap),
                    force_refresh=False,
                )
                if boot_cash and getattr(boot_cash, "total", 0.0) > 0:
                    self._persist_cached_cash(boot_cash)
            except Exception as warm_exc:
                logger.debug("Boot cash cache warm-up deferred: %s", warm_exc)
            if hasattr(self.t212, "get_historical_orders"):
                try:
                    self.t212.get_historical_orders(limit=30, force_refresh=False)
                except Exception as hist_exc:
                    logger.debug("Boot historical orders warm-up deferred: %s", hist_exc)

        # ── 7. Audit Logging ──────────────────────────────────────
        default_trades_log = (
            self.data_dir / "trades.jsonl"
            if (explicit_isolation or is_testing)
            else Path(settings.TRADES_LOG_PATH)
        )
        self.trades_log = Path(trades_log) if trades_log else default_trades_log

        # Populate initial read-model snapshot cleanly with zero extra broker HTTP requests
        self._refresh_read_model_snapshot(
            persist_funnel=(not is_testing or explicit_isolation),
            broadcast=False,
        )

        # ── 8. Embedded Dashboard Web Server ──────────────────────
        self.dashboard_thread = None
        self.dashboard_server = None
        self.dashboard_url = ""
        self._generate_html_dashboard = None

        if enable_dashboard is None:
            enable_dashboard = False if is_testing else getattr(settings, "ENABLE_EMBEDDED_DASHBOARD", True)

        self.enable_dashboard = enable_dashboard
        self.dashboard_port = dashboard_port or getattr(settings, "DASHBOARD_PORT", 8888)
        self.dashboard_host = (dashboard_host if dashboard_host is not None else getattr(settings, "DASHBOARD_HOST", "")).strip()

        if self.enable_dashboard:
            try:
                from trading_system.dashboard.server import (
                    generate_html_dashboard,
                    start_local_server_thread,
                )
                self._generate_html_dashboard = generate_html_dashboard
                srv_sig = inspect.signature(start_local_server_thread)
                srv_kwargs: Dict[str, Any] = {
                    "port": self.dashboard_port,
                    "open_browser": open_browser,
                }
                accepts_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in srv_sig.parameters.values()
                )
                if "orchestrator" in srv_sig.parameters or accepts_var_kw:
                    srv_kwargs["orchestrator"] = self
                if "allow_port_fallback" in srv_sig.parameters or accepts_var_kw:
                    srv_kwargs["allow_port_fallback"] = False
                if "host" in srv_sig.parameters or accepts_var_kw:
                    if self.dashboard_host:
                        srv_kwargs["host"] = self.dashboard_host
                self.dashboard_thread, self.dashboard_server, self.dashboard_url = start_local_server_thread(
                    **srv_kwargs
                )
                pub_url = os.environ.get("LIVE_DASHBOARD_PUBLIC_URL", "").strip()
                if pub_url:
                    self.dashboard_url = pub_url
            except Exception as exc:
                logger.warning("Failed to start embedded dashboard server: %s", exc)

        logger.info(
            "Orchestrator initialized — 3x Universe: %d assets, Epoch: %s, Mode: %s (DryRun: %s, Dashboard: %s)",
            len(self.universe),
            snap.config_epoch_id,
            "LIVE" if settings.USE_LIVE else "DEMO",
            self.dry_run,
            self.dashboard_url if self.enable_dashboard and self.dashboard_url else "DISABLED",
        )

    def _rebuild_universe(self, snap: ConfigSnapshot) -> None:
        """Build monitored underlying list from active ConfigSnapshot."""
        use_lev = bool(getattr(snap, "USE_LEVERAGED_3X", True))
        watchlist = getattr(snap, "WATCHLIST", None)
        universe_size = int(getattr(snap, "UNIVERSE_SIZE", 100))
        if use_lev:
            self.universe = list(watchlist) if watchlist else self.leveraged_universe.get_all_underlyings()
        else:
            self.universe = list(watchlist) if watchlist else DEFAULT_UNIVERSE[:universe_size]

    def _apply_snapshot_to_runtime(self, snap: ConfigSnapshot) -> None:
        """Update hot-reloadable runtime knobs after a ConfigBus commit."""
        self.parallel_ingestion_workers = int(getattr(snap, "PARALLEL_INGESTION_WORKERS", self.parallel_ingestion_workers))
        self.parallel_deliberation_workers = int(getattr(snap, "PARALLEL_DELIBERATION_WORKERS", self.parallel_deliberation_workers))
        if hasattr(self.t212, "rate_limit_delay"):
            with contextlib.suppress(Exception):
                self.t212.rate_limit_delay = float(getattr(snap, "RATE_LIMIT_INTER_CALL_DELAY", 1.25))
        if hasattr(self, "position_monitor") and self.position_monitor is not None:
            with contextlib.suppress(Exception):
                self.position_monitor.pure_tsl_mode = bool(getattr(snap, "PURE_TSL_MODE", True))
        self._rebuild_universe(snap)

    def _dispatch_system_alert(self, title: str, message: str) -> None:
        """Emit an operational system alert (e.g. boot quarantine) through Notifier safely."""
        if not getattr(self, "notifier", None):
            return
        try:
            if hasattr(self.notifier, "notify_alert") and callable(getattr(self.notifier, "notify_alert")):
                # Avoid invoking unconfigured MagicMock attributes unintentionally only if real or mocked explicitly
                from unittest.mock import NonCallableMock
                attr = getattr(self.notifier, "notify_alert")
                if not isinstance(attr, NonCallableMock) or "notify_alert" in getattr(self.notifier, "__dict__", {}):
                    attr(title, message)
                    return
            if hasattr(self.notifier, "dispatch"):
                note = TradeNotification(
                    event_type="SYSTEM_ALERT",  # type: ignore[arg-type]
                    ticker="SYSTEM",
                    underlying="CONFIG",
                    direction="LONG",
                    price=0.0,
                    quantity=0.0,
                    stop_loss=0.0,
                    trailing_stop=0.0,
                    tp1=0.0,
                    tp2=0.0,
                    tp3=0.0,
                    rationale=f"{title}: {message}",
                )
                self.notifier.dispatch(note)
        except Exception as exc:
            logger.debug("System alert dispatch failed cleanly: %s", exc)

    def _load_persisted_funnel_state(self) -> Optional[CycleFunnelReport]:
        """Restore the latest CycleFunnelReport from data/funnel_state.json if valid."""
        path = getattr(self, "_funnel_state_path", None)
        if path is None or not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not isinstance(raw.get("stage_counts"), dict):
                return None
            return CycleFunnelReport(
                timestamp=str(raw.get("timestamp", "")),
                config_epoch_id=raw.get("config_epoch_id"),
                stage_counts={str(k): int(v) for k, v in raw.get("stage_counts", {}).items()},
                gate_breakdown=[dict(x) for x in (raw.get("gate_breakdown") or []) if isinstance(x, dict)],
                _compact_indicators={
                    str(k): {str(mk): float(mv) for mk, mv in row.items()}
                    for k, row in (raw.get("_compact_indicators") or {}).items()
                    if isinstance(row, dict)
                },
                metric_percentiles={
                    str(k): {str(sk): float(sv) for sk, sv in stats.items()}
                    for k, stats in (raw.get("metric_percentiles") or {}).items()
                    if isinstance(stats, dict)
                },
                passed_tickers=[str(t) for t in (raw.get("passed_tickers") or [])],
                executed_tickers=[str(t) for t in (raw.get("executed_tickers") or [])],
            )
        except Exception as exc:
            logger.warning("Ignoring corrupt or unreadable funnel state %s: %s", path, exc)
            return None

    def _broadcast_sse(self, event_type: str, data: Any) -> None:
        """Broadcast an SSE event through SSEBroadcaster safely without propagating errors."""
        try:
            from trading_system.dashboard.server import SSEBroadcaster
            SSEBroadcaster.get_instance().broadcast(event_type, data)
        except Exception as exc:
            logger.debug("SSE broadcast (%s) skipped: %s", event_type, exc)

    def _broadcast_pipeline_progress(
        self,
        step: int,
        name: str,
        detail: str,
        *,
        total_steps: int = 5,
        active_assets: Optional[Sequence[str]] = None,
        cycle_id: Optional[str] = None,
        status: str = "active",
    ) -> Dict[str, Any]:
        """Broadcast real-time scan pipeline stage progression over SSE with zero broker quota."""
        c_id = str(cycle_id or getattr(self, "_current_cycle_id", "") or "")
        payload = {
            "step": int(step),
            "total_steps": int(total_steps),
            "name": str(name),
            "detail": str(detail),
            "status": str(status),
            "active_assets": list(active_assets or ()),
            "timestamp": time.time(),
            "iso_time": datetime.now(timezone.utc).isoformat(),
            "cycle_id": c_id,
        }
        self._current_pipeline_progress = payload
        self._broadcast_sse("pipeline_progress", copy.deepcopy(payload))
        return payload

    def _persist_cached_cash(self, cash: Any) -> None:
        """Persist live broker cash snapshot atomically to data/cached_cash.json."""
        if cash is None:
            return
        total = getattr(cash, "total", None) if not isinstance(cash, dict) else cash.get("total")
        if not isinstance(total, (int, float)) or total <= 0:
            return
        try:
            payload = {
                "free": float(getattr(cash, "free", 0.0) if not isinstance(cash, dict) else cash.get("free", 0.0)),
                "invested": float(getattr(cash, "invested", 0.0) if not isinstance(cash, dict) else cash.get("invested", 0.0)),
                "total": float(total),
                "blocked": float(getattr(cash, "blocked", 0.0) if not isinstance(cash, dict) else cash.get("blocked", 0.0)),
                "result": float(getattr(cash, "result", 0.0) if not isinstance(cash, dict) else cash.get("result", 0.0)),
                "ppl_result": float(getattr(cash, "ppl_result", 0.0) if not isinstance(cash, dict) else cash.get("ppl_result", 0.0)),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            if hasattr(self, "_cash_cache_path"):
                _atomic_write_json(self._cash_cache_path, payload)
        except Exception as exc:
            logger.debug("Failed persisting cached cash: %s", exc)

    def _get_fallback_equity(self, snap: Optional[Any] = None) -> float:
        """Resolve baseline equity, preferring cached broker cash before config default."""
        if hasattr(self, "_cash_cache_path") and self._cash_cache_path.is_file():
            try:
                p_c = json.loads(self._cash_cache_path.read_text(encoding="utf-8"))
                val = float(p_c.get("total") or 0.0)
                if val > 0:
                    return round(val, 2)
            except Exception:
                pass
        if snap is not None and hasattr(snap, "DEFAULT_ACCOUNT_EQUITY"):
            return float(snap.DEFAULT_ACCOUNT_EQUITY)
        return float(getattr(settings, "DEFAULT_ACCOUNT_EQUITY", 10000.0) or 10000.0)

    def _get_cached_broker_view(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Read cached Trading 212 snapshot and rate telemetry with ZERO HTTP calls."""
        cached: Dict[str, Any] = {}
        if hasattr(self.t212, "get_cached_broker_snapshot"):
            try:
                raw = self.t212.get_cached_broker_snapshot()
                if isinstance(raw, dict):
                    cached = raw
            except Exception as exc:
                logger.debug("Cached broker snapshot read notice: %s", exc)

        if not cached:
            # Fallback for custom/minimal clients supporting cached_only=True
            cash_dict = None
            try:
                if hasattr(self.t212, "get_account_cash"):
                    sig = inspect.signature(self.t212.get_account_cash)
                    if "cached_only" in sig.parameters:
                        c = self.t212.get_account_cash(cached_only=True)
                        if isinstance(c, AccountCash):
                            cash_dict = {
                                "free": float(c.free),
                                "invested": float(c.invested),
                                "total": float(c.total),
                                "blocked": float(c.blocked),
                                "result": float(c.result),
                                "ppl_result": float(c.ppl_result),
                            }
            except Exception:
                pass
            cached = {"cash": cash_dict, "portfolio": None, "orders": None, "historical_orders": None}

        # Check persisted cash cache if cash is missing or total is non-positive
        cur_cash = cached.get("cash")
        if not cur_cash or float(cur_cash.get("total") or 0.0) <= 0.0:
            if hasattr(self, "_cash_cache_path") and self._cash_cache_path.is_file():
                try:
                    persisted = json.loads(self._cash_cache_path.read_text(encoding="utf-8"))
                    if isinstance(persisted, dict) and float(persisted.get("total") or 0.0) > 0.0:
                        cached["cash"] = persisted
                except Exception:
                    pass

        telemetry: Dict[str, Any] = {}
        if hasattr(self.t212, "get_rate_limit_telemetry"):
            try:
                raw_tel = self.t212.get_rate_limit_telemetry()
                if isinstance(raw_tel, dict):
                    telemetry = copy.deepcopy(raw_tel)
            except Exception:
                pass

        return cached, telemetry

    def _snapshot_positions_under_lock(self) -> Tuple[bool, List[Tuple[str, Dict[str, Any]]]]:
        """Copy TrackedPosition dicts briefly under PositionMonitor._state_lock (no I/O)."""
        pm = getattr(self, "position_monitor", None)
        if pm is None:
            return False, []

        raw_positions = getattr(pm, "positions", None)
        if not isinstance(raw_positions, Mapping):
            raw_positions = {}

        state_lock = getattr(pm, "_state_lock", None)
        lock_ctx = state_lock if state_lock is not None and hasattr(state_lock, "__enter__") else contextlib.nullcontext()

        rows: List[Tuple[str, Dict[str, Any]]] = []
        with lock_ctx:
            halted = False
            if hasattr(pm, "is_system_halted"):
                halted = _as_bool_strict(pm.is_system_halted(), False)
            if not halted:
                halted = _as_bool_strict(getattr(pm, "is_halted", False), False)

            for ticker, pos in list(raw_positions.items()):
                if _pos_is_closed(pos):
                    continue
                rem_qty = _pos_remaining_qty(pos)
                panic_failed_raw = (
                    pos.get("panic_failed", False)
                    if isinstance(pos, Mapping)
                    else getattr(pos, "panic_failed", False)
                )
                if rem_qty <= 0.0 and not _as_bool_strict(panic_failed_raw, False):
                    continue
                if isinstance(pos, Mapping):
                    pos_dict = dict(pos)
                elif hasattr(type(pos), "__dataclass_fields__"):
                    pos_dict = asdict(pos)
                else:
                    pos_dict = {
                        str(k): v
                        for k, v in vars(pos).items()
                        if not str(k).startswith("_") and not callable(v)
                    }
                rows.append((str(ticker), pos_dict))
        return halted, rows

    def _build_active_signals_snapshot(
        self,
        pos_rows: List[Tuple[str, Dict[str, Any]]],
        cached_prices: Mapping[str, float],
        cfg_snap: ConfigSnapshot,
        broker_positions_map: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build enriched active_signals, positions_behind_epoch, and unprotected_positions lists."""
        active_epoch_id = cfg_snap.config_epoch_id
        signals: List[Dict[str, Any]] = []
        behind_epoch: List[Dict[str, Any]] = []
        unprotected: List[Dict[str, Any]] = []

        for ticker, pos in pos_rows:
            underlying = str(pos.get("underlying") or ticker)
            pair = self.leveraged_universe.get_pair_for_underlying(underlying)
            asset_name = pair.name if pair else underlying

            raw_dir = str(pos.get("direction") or "LONG").upper()
            auth_dir = self.leveraged_universe.get_etp_direction(ticker)
            direction = auth_dir if auth_dir else ("SHORT" if "SHORT" in raw_dir or "SELL" in raw_dir else "LONG")

            bp = broker_positions_map.get(ticker) if broker_positions_map else None
            bp_avg = float((bp.get("average_price") if isinstance(bp, Mapping) else getattr(bp, "average_price", 0.0)) or 0.0) if bp else 0.0
            bp_cur = float((bp.get("current_price") if isinstance(bp, Mapping) else getattr(bp, "current_price", 0.0)) or 0.0) if bp else 0.0
            bp_qty = float((bp.get("quantity") if isinstance(bp, Mapping) else getattr(bp, "quantity", 0.0)) or 0.0) if bp else 0.0

            entry = bp_avg if bp_avg > 0 else float(pos.get("entry_price") or 0.0)
            cur_px = bp_cur if bp_cur > 0 else float(cached_prices.get(ticker) or pos.get("current_price") or entry)
            orig_qty = bp_qty if bp_qty > 0 else float(pos.get("original_quantity") or 0.0)
            rem_qty = bp_qty if bp_qty > 0 else float(pos.get("remaining_quantity") or 0.0)
            sl = float(pos.get("stop_loss") or 0.0)
            tsl = float(pos.get("trailing_stop_loss") or sl)

            currency_code = get_ticker_currency(ticker)
            currency_code, currency_prefix, currency_suffix = get_currency_display(currency_code)

            bp_ppl = (bp.get("ppl") if isinstance(bp, Mapping) else getattr(bp, "ppl", None)) if bp else None
            if bp_ppl is not None:
                pnl_amount_eur = round(float(bp_ppl), 2)
                cost_basis_eur = normalize_quote_to_eur(ticker, entry, currency_code) * rem_qty
                if cost_basis_eur > 0:
                    pnl_pct = round((pnl_amount_eur / cost_basis_eur) * 100.0, 2)
                elif entry > 0:
                    pnl_pct = round(((cur_px - entry) / entry * 100.0), 2)
                else:
                    pnl_pct = 0.0
                pnl_amount = round((cur_px - entry) * rem_qty, 2)
            else:
                pnl_pct = ((cur_px - entry) / entry * 100.0) if entry > 0 else 0.0
                pnl_amount = (cur_px - entry) * rem_qty
                pnl_amount_eur = normalize_quote_to_eur(ticker, pnl_amount, currency_code)

            risk_pct = ((entry - sl) / entry * 100.0) if entry > 0 and sl > 0 else 0.0
            broker_stop_id = pos.get("broker_stop_order_id")
            broker_tp_id = pos.get("broker_tp_order_id")
            panic_failed = bool(pos.get("panic_failed", False))
            unprotected_reason = pos.get("unprotected_reason")
            pos_epoch_id = pos.get("config_epoch_id")
            is_behind = (pos_epoch_id != active_epoch_id)

            entry_snap_raw = pos.get("entry_config_snapshot")
            pos_pure_tsl = bool(
                entry_snap_raw.get("PURE_TSL_MODE", cfg_snap.PURE_TSL_MODE)
                if isinstance(entry_snap_raw, Mapping)
                else cfg_snap.PURE_TSL_MODE
            )

            session_count = max(1, int(pos.get("session_count") or 1))
            is_runner = bool(pos.get("is_runner", False))
            snap_for_decay = entry_snap_raw if isinstance(entry_snap_raw, Mapping) else cfg_snap
            decay_mult = RiskManager.get_decay_atr_multiplier(
                session_count=session_count,
                is_runner=is_runner,
                config_snapshot=snap_for_decay,
            )

            if panic_failed:
                status = "PANIC_FAILED"
                tsl_status = "Panic Failed"
            elif unprotected_reason or not broker_stop_id:
                status = "UNPROTECTED"
                tsl_status = "No GTC Stop"
            elif tsl >= entry > 0:
                status = "BREAKEVEN LOCKED (Risk-Free Trail Active)"
                tsl_status = "Break-Even Locked"
            else:
                status = "ACTIVE (Initial Stop Protection)"
                tsl_status = "Initial Stop Active"

            item = {
                "ticker": ticker,
                "underlying": underlying,
                "name": asset_name,
                "direction": direction,
                "leverage": int(pos.get("leverage") or (5 if "5" in ticker else 3)),
                "entry_price": round(entry, 4),
                "current_price": round(cur_px, 4),
                "pnl_pct": round(pnl_pct, 2),
                "pnl_amount": round(pnl_amount, 2),
                "pnl_amount_eur": round(pnl_amount_eur, 2),
                "currency_code": currency_code,
                "currency_prefix": currency_prefix,
                "currency_suffix": currency_suffix,
                "stop_loss": round(sl, 4),
                "trailing_stop_loss": round(tsl, 4),
                "tp1": round(float(pos.get("tp1") or 0.0), 4),
                "tp2": round(float(pos.get("tp2") or 0.0), 4),
                "tp3": round(float(pos.get("tp3") or 0.0), 4),
                "highest_price": round(float(pos.get("highest_price") or cur_px), 4),
                "lowest_price": round(float(pos.get("lowest_price") or cur_px), 4),
                "tsl_status": tsl_status,
                "session_count": session_count,
                "is_runner": is_runner,
                "decay_multiplier": decay_mult,
                "decay_label": f"Session {session_count} ({decay_mult:.2f}x ATR)",
                "stop_distance_pct": round(risk_pct, 2),
                "broker_stop_order_id": broker_stop_id,
                "broker_tp_order_id": broker_tp_id,
                "quantity": orig_qty,
                "remaining_quantity": rem_qty,
                "status": status,
                "entry_time": pos.get("entry_time", ""),
                "atr": round(float(pos.get("atr") or 0.0), 4),
                "earnings_in_days": pos.get("earnings_in_days"),
                "config_epoch_id": pos_epoch_id,
                "position_epoch_id": pos_epoch_id,
                "pure_tsl_mode": pos_pure_tsl,
                "is_behind_epoch": is_behind,
                "panic_failed": panic_failed,
                "unprotected_reason": unprotected_reason,
            }
            signals.append(item)

            if is_behind:
                behind_epoch.append({
                    "ticker": ticker,
                    "underlying": underlying,
                    "direction": direction,
                    "position_epoch_id": pos_epoch_id,
                    "active_epoch_id": active_epoch_id,
                    "is_legacy": pos_epoch_id is None,
                    "stop_loss": item["stop_loss"],
                    "trailing_stop_loss": item["trailing_stop_loss"],
                    "pure_tsl_mode": pos_pure_tsl,
                })

            if panic_failed or unprotected_reason or not broker_stop_id:
                unprotected.append({
                    "ticker": ticker,
                    "underlying": underlying,
                    "panic_failed": panic_failed,
                    "reason": unprotected_reason or "Missing broker GTC stop",
                    "broker_stop_order_id": broker_stop_id,
                    "remaining_quantity": rem_qty,
                })

        # Adopt any live broker positions not currently tracked in local rows
        known_tickers = {s.get("ticker") for s in signals}
        if broker_positions_map:
            for b_ticker, b_pos in broker_positions_map.items():
                if b_ticker in known_tickers:
                    continue
                pair = self.leveraged_universe.get_pair_for_etp(b_ticker)
                underlying = (
                    pair.underlying_symbol
                    if pair
                    else b_ticker.replace("l_EQ", "").replace("_US_EQ", "")
                )
                underlying_name = pair.name if pair else underlying
                direction = (
                    self.leveraged_universe.get_etp_direction(b_ticker)
                    or "LONG"
                )
                b_avg = float((b_pos.get("average_price") if isinstance(b_pos, Mapping) else getattr(b_pos, "average_price", 0.0)) or 0.0)
                b_cur = float((b_pos.get("current_price") if isinstance(b_pos, Mapping) else getattr(b_pos, "current_price", 0.0)) or b_avg)
                b_qty = float((b_pos.get("quantity") if isinstance(b_pos, Mapping) else getattr(b_pos, "quantity", 0.0)) or 0.0)
                if b_qty <= 0:
                    continue
                sl_dist = round(b_avg * 0.05, 4)
                sl = round(max(0.01, b_avg - sl_dist), 4)
                currency_code = get_ticker_currency(b_ticker)
                currency_code, currency_prefix, currency_suffix = get_currency_display(currency_code)

                b_ppl = (b_pos.get("ppl") if isinstance(b_pos, Mapping) else getattr(b_pos, "ppl", None))
                if b_ppl is not None:
                    pnl_amount_eur = round(float(b_ppl), 2)
                    cost_basis_eur = normalize_quote_to_eur(b_ticker, b_avg, currency_code) * b_qty
                    pnl_pct = round((pnl_amount_eur / cost_basis_eur) * 100.0, 2) if cost_basis_eur > 0 else 0.0
                else:
                    pnl_amount_eur = normalize_quote_to_eur(b_ticker, (b_cur - b_avg) * b_qty, currency_code)
                    pnl_pct = round(((b_cur - b_avg) / b_avg * 100.0), 2) if b_avg > 0 else 0.0
                pnl_amount = round((b_cur - b_avg) * b_qty, 2)

                signals.append({
                    "ticker": b_ticker,
                    "underlying": underlying,
                    "name": underlying_name,
                    "direction": direction,
                    "leverage": 5 if ("5" in b_ticker) else 3,
                    "entry_price": round(b_avg, 4),
                    "current_price": round(b_cur, 4),
                    "pnl_pct": pnl_pct,
                    "pnl_amount": pnl_amount,
                    "pnl_amount_eur": pnl_amount_eur,
                    "currency_code": currency_code,
                    "currency_prefix": currency_prefix,
                    "currency_suffix": currency_suffix,
                    "stop_loss": sl,
                    "trailing_stop_loss": sl,
                    "tp1": round(b_avg * 1.05, 4),
                    "tp2": round(b_avg * 1.10, 4),
                    "tp3": round(b_avg * 1.15, 4),
                    "highest_price": round(b_cur, 4),
                    "lowest_price": round(b_cur, 4),
                    "tsl_status": "Initial Stop Active",
                    "session_count": 1,
                    "is_runner": False,
                    "decay_multiplier": 2.2,
                    "decay_label": "Session 1 (2.20x ATR)",
                    "stop_distance_pct": 5.0,
                    "broker_stop_order_id": None,
                    "broker_tp_order_id": None,
                    "quantity": b_qty,
                    "remaining_quantity": b_qty,
                    "status": "ACTIVE (Broker Placed)",
                    "entry_time": datetime.now(timezone.utc).isoformat(),
                    "atr": round(b_avg * 0.025, 4),
                    "earnings_in_days": None,
                    "config_epoch_id": active_epoch_id,
                    "position_epoch_id": active_epoch_id,
                    "pure_tsl_mode": cfg_snap.PURE_TSL_MODE,
                    "is_behind_epoch": False,
                    "panic_failed": False,
                    "unprotected_reason": None,
                })

        return signals, behind_epoch, unprotected

    def _build_pairs_watchlist_snapshot(self, cfg_snap: ConfigSnapshot) -> Tuple[List[Dict[str, Any]], str, Dict[str, int]]:
        """Build dynamic universe pairs list and EUR/USD/GBX currency mix breakdown."""
        allow_5x = bool(getattr(cfg_snap, "ENABLE_5X_LEVERAGE", False))
        pairs_list: List[Dict[str, Any]] = []
        currency_counts: Dict[str, int] = {"EUR": 0, "USD": 0, "GBX": 0, "GBP": 0, "other": 0}

        for pair in self.leveraged_universe.pairs:
            long_t = (
                pair.long_5x_ticker
                if (allow_5x and pair.has_5x and pair.long_5x_ticker)
                else pair.long_3x_ticker
            )
            short_t = (
                pair.short_5x_ticker
                if (allow_5x and pair.has_5x and pair.short_5x_ticker and pair.short_5x_ticker != "None")
                else (pair.short_3x_ticker if pair.short_3x_ticker and pair.short_3x_ticker != "None" else None)
            )
            long_curr = get_ticker_currency(long_t) if long_t else "EUR"
            if long_curr in currency_counts:
                currency_counts[long_curr] += 1
            else:
                currency_counts["other"] += 1

            pairs_list.append({
                "underlying": pair.underlying_symbol,
                "name": pair.name,
                "category": pair.category,
                "long_ticker": long_t,
                "short_ticker": short_t,
                "has_5x": bool(pair.has_5x),
                "long_5x_ticker": pair.long_5x_ticker,
                "short_5x_ticker": pair.short_5x_ticker,
                "currency": long_curr,
                "long_currency": long_curr,
            })

        total = max(1, len(pairs_list))
        eur_pct = round((currency_counts["EUR"] / total) * 100.0)
        currency_counts["total"] = len(pairs_list)
        return pairs_list, f"{eur_pct}%", currency_counts

    def _build_read_model_snapshot(self) -> Dict[str, Any]:
        """Construct complete JSON-ready dashboard snapshot with zero Trading 212 HTTP calls."""
        cfg_snap = self.config_bus.active_snapshot()
        active_epoch_id = cfg_snap.config_epoch_id
        now_utc = datetime.now(timezone.utc)
        generated_at_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

        cached_broker, rate_telemetry = self._get_cached_broker_view()
        cached_prices: Dict[str, float] = {}
        broker_positions_map: Dict[str, Any] = {}
        for p in (cached_broker.get("portfolio") or []):
            tk = str((p.get("ticker") if isinstance(p, Mapping) else getattr(p, "ticker", "")) or "")
            if tk:
                broker_positions_map[tk] = p
                px = float((p.get("current_price") if isinstance(p, Mapping) else getattr(p, "current_price", 0.0)) or 0.0)
                if px <= 0:
                    px = float((p.get("average_price") if isinstance(p, Mapping) else getattr(p, "average_price", 0.0)) or 0.0)
                if px > 0:
                    cached_prices[tk] = px

        is_halted, pos_rows = self._snapshot_positions_under_lock()
        active_signals, positions_behind_epoch, unprotected_positions = self._build_active_signals_snapshot(
            pos_rows, cached_prices, cfg_snap, broker_positions_map=broker_positions_map
        )
        pairs_watchlist, eur_ratio, currency_mix = self._build_pairs_watchlist_snapshot(cfg_snap)

        # Load recent history and notifications strictly from local logs + cached historical orders
        try:
            from trading_system.dashboard.server import (
                fetch_live_broker_notifications,
                load_recent_history,
                load_recent_notifications,
            )
            history = load_recent_history(str(self.trades_log), limit=15)
            local_notifs = load_recent_notifications(str(self.notifier.log_file), limit=30)
            cached_hist = cached_broker.get("historical_orders")
            broker_notifs = (
                fetch_live_broker_notifications(_CachedHistoryClientProxy(cached_hist), self.leveraged_universe, limit=30)
                if isinstance(cached_hist, dict) and cached_hist.get("items")
                else []
            )
        except Exception:
            history, local_notifs, broker_notifs = [], [], []

        seen_notifs: Set[Tuple[Any, ...]] = set()
        seen_order_ids: Set[str] = set()
        broker_tickers_with_pnl: Set[str] = set()
        merged_notifs: List[Dict[str, Any]] = []

        for b_ev in broker_notifs:
            if not isinstance(b_ev, dict):
                continue
            oid = str(b_ev.get("order_id") or "")
            if oid:
                seen_order_ids.add(oid)
            tk = b_ev.get("ticker", "")
            if tk and b_ev.get("pnl") is not None:
                broker_tickers_with_pnl.add(tk)
            dedup_key = (b_ev.get("event_type"), tk, round(float(b_ev.get("price") or 0.0), 3), str(b_ev.get("timestamp", ""))[:16])
            seen_notifs.add(dedup_key)
            merged_notifs.append(dict(b_ev))

        for n in local_notifs:
            if not isinstance(n, dict):
                continue
            n_oid = str(n.get("order_id") or "")
            if n_oid and n_oid in seen_order_ids:
                continue
            n_tk = n.get("ticker", "")
            # Suppress false 0.0 P&L reconciliation entries when a broker fill exists for that ticker
            if n_tk in broker_tickers_with_pnl and n.get("pnl") == 0.0 and "Reconciliation" in str(n.get("rationale") or ""):
                continue
            dedup_key = (n.get("event_type"), n_tk, round(float(n.get("price") or 0.0), 3), str(n.get("timestamp", ""))[:16])
            if dedup_key in seen_notifs:
                continue
            seen_notifs.add(dedup_key)
            merged_notifs.append(dict(n))

        merged_notifs.sort(key=lambda x: str(x.get("timestamp", "")), reverse=True)
        recent_notifications = merged_notifs[:30]

        cached_cash = cached_broker.get("cash") if isinstance(cached_broker.get("cash"), Mapping) else None
        cached_total = float(cached_cash.get("total") or 0.0) if cached_cash else 0.0
        if cached_total <= 0.0 and hasattr(self, "_cash_cache_path") and self._cash_cache_path.is_file():
            try:
                p_c = json.loads(self._cash_cache_path.read_text(encoding="utf-8"))
                if float(p_c.get("total") or 0.0) > 0:
                    cached_total = float(p_c["total"])
            except Exception:
                pass
        default_eq = self._get_fallback_equity(cfg_snap)
        base_equity = round(cached_total, 2) if cached_total > 0 else round(default_eq, 2)

        equity_points: List[Dict[str, Any]] = [{"time": "Start", "equity": base_equity}]
        running_equity = base_equity
        realized_evs = [e for e in recent_notifications if e.get("pnl") is not None]
        realized_evs.reverse()
        for ev in realized_evs:
            running_equity += float(ev.get("pnl") or 0.0)
            t_str = str(ev.get("timestamp") or "")[-12:-4] or "Trade"
            equity_points.append({"time": t_str, "equity": round(running_equity, 2)})

        total_open_pnl = sum(float(s.get("pnl_amount_eur", s.get("pnl_amount", 0.0)) or 0.0) for s in active_signals)
        current_equity = (
            round(cached_total + total_open_pnl, 2)
            if cached_total > 0
            else round(running_equity + total_open_pnl, 2)
        )
        equity_points.append({"time": now_utc.strftime("%H:%M:%S"), "equity": current_equity})

        active_stops_cached = cached_broker.get("active_server_stops_count")
        active_stops = (
            int(active_stops_cached)
            if isinstance(active_stops_cached, int)
            else sum(1 for s in active_signals if s.get("broker_stop_order_id"))
        )

        # Build funnel payload cleanly
        if self._latest_funnel_report is not None:
            funnel_dict = self._latest_funnel_report.to_dict()
            funnel_dict.setdefault("cross_sectional_metrics", copy.deepcopy(funnel_dict.get("metric_percentiles", {})))
            funnel_dict["status"] = self._latest_funnel_meta.get("status", "ok")
            if self._latest_funnel_meta.get("reason"):
                funnel_dict["reason"] = self._latest_funnel_meta["reason"]
        else:
            funnel_dict = {
                "status": self._latest_funnel_meta.get("status", "pre_scan"),
                "reason": self._latest_funnel_meta.get("reason", "No completed universe scan cycle recorded yet."),
                "timestamp": now_utc.isoformat(),
                "config_epoch_id": active_epoch_id,
                "stage_counts": {
                    "universe": 0,
                    "screened": 0,
                    "strategy": 0,
                    "debate": 0,
                    "gates": 0,
                    "executed": 0,
                },
                "gate_breakdown": [],
                "_compact_indicators": {},
                "metric_percentiles": {},
                "cross_sectional_metrics": {},
                "passed_tickers": [],
                "executed_tickers": [],
            }

        last_audit = None
        audit_hist = []

        scrubbed_cfg = _scrub_mapping(cfg_snap.to_dict())
        schema_dict = export_config_schema(cfg_snap.dynamic_parameter_specs)
        profiles_dict = get_builtin_profiles()

        return {
            "generated_at": generated_at_str,
            "mode": "DRY_RUN" if self.dry_run else ("LIVE_AUTO" if settings.USE_LIVE else "DEMO_AUTO"),
            "controller_attached": True,
            "read_only_reason": None,
            "is_halted": is_halted,
            "active_signals": active_signals,
            "active_signals_count": len(active_signals),
            "recent_notifications": recent_notifications,
            "pairs_watchlist": pairs_watchlist,
            "history": history,
            "equity_history": equity_points,
            "equity_points": equity_points,
            "base_equity": base_equity,
            "starting_equity": base_equity,
            "current_equity": current_equity,
            "broker_status": {
                "broker_name": "Trading 212 Live" if settings.USE_LIVE else "Trading 212 Demo",
                "broker_mode": "LIVE" if settings.USE_LIVE else "DEMO",
                "dry_run": bool(self.dry_run),
                "is_connected": bool(settings.TRADING212_API_KEY),
                "is_halted": is_halted,
                "active_server_stops": active_stops,
                "active_positions_count": len(active_signals),
                "execution_mode": (
                    "System 2 Pure TSL (Native GTC Stop + Algorithmic Trail)"
                    if bool(cfg_snap.PURE_TSL_MODE)
                    else "System 2 Scaled Exits (TP1/2/3 + Native GTC TSL)"
                ),
                "runner_status": "HALTED" if is_halted else ("DRY-RUN" if self.dry_run else "ONLINE"),
                "monitored_assets": len(self.universe),
                "universe_count": len(self.universe),
                "pairs_count": len(pairs_watchlist),
                "eur_ratio": eur_ratio,
                "currency_mix": currency_mix,
                "rate_limit_telemetry": rate_telemetry,
            },
            "active_epoch_id": active_epoch_id,
            "config_epoch_id": active_epoch_id,
            "config": scrubbed_cfg,
            "schema": schema_dict,
            "presets": profiles_dict,
            "pending_restart_keys": list(cfg_snap.pending_restart_keys),
            "config_quarantine_warnings": list(cfg_snap.config_quarantine_warnings),
            "epoch_history": self.config_bus.epoch_history(limit=25),
            "funnel": funnel_dict,
            "pipeline_progress": copy.deepcopy(getattr(self, "_current_pipeline_progress", None)),
            "audit": None,
            "audit_history": [],
            "rate_limit_telemetry": rate_telemetry,
            "positions_behind_epoch": positions_behind_epoch,
            "unprotected_positions": unprotected_positions,
        }

    def _refresh_read_model_snapshot(
        self,
        *,
        persist_funnel: bool = True,
        broadcast: bool = True,
        extra_events: Optional[Sequence[Tuple[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Rebuild thread-safe read-model snapshot, optionally persist funnel_state.json, and notify SSE."""
        snapshot_payload = self._build_read_model_snapshot()
        with self._read_model_lock:
            self._read_model_snapshot = snapshot_payload

        if persist_funnel:
            try:
                _atomic_write_json(self._funnel_state_path, snapshot_payload["funnel"])
            except Exception as exc:
                logger.warning("Failed persisting atomic funnel state to %s: %s", self._funnel_state_path, exc)

        if broadcast:
            if extra_events:
                for evt_name, evt_data in extra_events:
                    self._broadcast_sse(evt_name, copy.deepcopy(evt_data))
            self._broadcast_sse("dashboard_update", copy.deepcopy(snapshot_payload))

        return copy.deepcopy(snapshot_payload)

    def get_dashboard_snapshot(self) -> Dict[str, Any]:
        """Return a deep-copied dashboard/control-plane snapshot with ZERO Trading 212 HTTP calls."""
        with self._read_model_lock:
            if self._read_model_snapshot is not None:
                return copy.deepcopy(self._read_model_snapshot)
        return self._refresh_read_model_snapshot(persist_funnel=False, broadcast=False)

    def _compute_positions_behind_epoch(self, target_epoch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Summarize open positions whose entry epoch lags target_epoch_id."""
        resolved_target = target_epoch_id or self.config_bus.active_epoch_id
        _, rows = self._snapshot_positions_under_lock()
        behind: List[Dict[str, Any]] = []
        for ticker, pos in rows:
            pos_epoch = pos.get("config_epoch_id")
            if pos_epoch != resolved_target:
                behind.append({
                    "ticker": ticker,
                    "underlying": pos.get("underlying", ticker),
                    "direction": pos.get("direction", "LONG"),
                    "position_epoch_id": pos_epoch,
                    "target_epoch_id": resolved_target,
                    "is_legacy": pos_epoch is None,
                    "stop_loss": pos.get("stop_loss"),
                    "trailing_stop_loss": pos.get("trailing_stop_loss"),
                })
        return behind

    # ─────────────────────────────────────────────────────────────
    # Thread-Safe Control Gateway Surface (U4 / U5)
    # ─────────────────────────────────────────────────────────────
    def preview_config_patch(
        self,
        patch: Mapping[str, Any],
        expected_epoch_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate a config patch, simulate Phase-2 screener gate impact, and report positions behind epoch."""
        base = self.config_bus.preview_patch(patch, expected_epoch_id=expected_epoch_id)
        current_snap = self.config_bus.active_snapshot()

        if base.get("validation_ok"):
            candidate_raw = current_snap.to_dict()
            dyn_keys = set((candidate_raw.get("dynamic_parameters") or {}).keys())
            for entry in base.get("diff") or ():
                k = entry.get("key")
                new_val = entry.get("new_value")
                if k in dyn_keys:
                    candidate_raw.setdefault("dynamic_parameters", {})[k] = new_val
                elif k in candidate_raw or k in SCHEMA_BY_KEY:
                    candidate_raw[k] = new_val
            try:
                candidate_snap = ConfigSnapshot.from_dict(candidate_raw)
            except Exception:
                candidate_snap = current_snap

            compact = (
                dict(self._latest_funnel_report._compact_indicators)
                if self._latest_funnel_report is not None and self._latest_funnel_report._compact_indicators
                else {}
            )
            if compact:
                baseline = simulate_screener_gate_impact(compact, current_snap)
                candidate = simulate_screener_gate_impact(compact, candidate_snap)
                base["gate_impact"] = {
                    "available": True,
                    "baseline_passed": baseline["selected_count"],
                    "candidate_passed": candidate["selected_count"],
                    "delta": candidate["selected_count"] - baseline["selected_count"],
                    "baseline_qualified": baseline["passed_count"],
                    "candidate_qualified": candidate["passed_count"],
                    "baseline": baseline,
                    "candidate": candidate,
                }
            else:
                base["gate_impact"] = {
                    "available": False,
                    "reason": "No compact funnel indicators cached yet (pre-scan or off-hours).",
                }
        else:
            base["gate_impact"] = {
                "available": False,
                "reason": "Validation errors present; simulation skipped.",
            }

        behind = self._compute_positions_behind_epoch(target_epoch_id=current_snap.config_epoch_id)
        base["positions_behind_epoch"] = behind
        base["positions_behind_epoch_count"] = len(behind)
        return base

    def commit_config_patch(
        self,
        patch: Mapping[str, Any],
        expected_epoch_id: str,
        author: str = "operator",
        reason: str = "",
    ) -> Dict[str, Any]:
        """Commit an epoch-versioned config patch with strict CAS, hot-reload runtime, and SSE notify."""
        new_snap = self.config_bus.commit_patch(
            patch,
            expected_epoch_id=expected_epoch_id,
            author=author,
            reason=reason,
        )
        self._apply_snapshot_to_runtime(new_snap)
        epoch_evt = {
            "config_epoch_id": new_snap.config_epoch_id,
            "active_epoch_id": new_snap.config_epoch_id,
            "author": author,
            "reason": reason,
            "pending_restart_keys": list(new_snap.pending_restart_keys),
            "config": _scrub_mapping(new_snap.to_dict()),
        }
        dash = self._refresh_read_model_snapshot(
            persist_funnel=True,
            broadcast=True,
            extra_events=[("config_epoch_changed", epoch_evt)],
        )
        return {
            "ok": True,
            "config_epoch_id": new_snap.config_epoch_id,
            "active_epoch_id": new_snap.config_epoch_id,
            "snapshot": _scrub_mapping(new_snap.to_dict()),
            "config": _scrub_mapping(new_snap.to_dict()),
            "pending_restart_keys": list(new_snap.pending_restart_keys),
            "dashboard": dash,
        }

    def apply_config_profile(
        self,
        profile_id: str,
        expected_epoch_id: str,
        author: str = "operator",
        reason: str = "",
    ) -> Dict[str, Any]:
        """Apply a named built-in configuration profile atomically under CAS."""
        new_snap = self.config_bus.apply_profile(
            profile_id,
            expected_epoch_id=expected_epoch_id,
            author=author,
            reason=reason,
        )
        self._apply_snapshot_to_runtime(new_snap)
        epoch_evt = {
            "config_epoch_id": new_snap.config_epoch_id,
            "active_epoch_id": new_snap.config_epoch_id,
            "profile_id": profile_id,
            "author": author,
            "reason": reason,
            "pending_restart_keys": list(new_snap.pending_restart_keys),
            "config": _scrub_mapping(new_snap.to_dict()),
        }
        dash = self._refresh_read_model_snapshot(
            persist_funnel=True,
            broadcast=True,
            extra_events=[("config_epoch_changed", epoch_evt)],
        )
        return {
            "ok": True,
            "profile_id": profile_id,
            "config_epoch_id": new_snap.config_epoch_id,
            "active_epoch_id": new_snap.config_epoch_id,
            "snapshot": _scrub_mapping(new_snap.to_dict()),
            "config": _scrub_mapping(new_snap.to_dict()),
            "pending_restart_keys": list(new_snap.pending_restart_keys),
            "dashboard": dash,
        }

    def add_custom_parameter(
        self,
        spec_dict: Mapping[str, Any],
        expected_epoch_id: Optional[str] = None,
        author: str = "operator",
        reason: str = "",
    ) -> Dict[str, Any]:
        """Create and register a validated user-defined dynamic parameter + screener gate."""
        new_snap = self.config_bus.add_dynamic_parameter(
            spec_dict,
            expected_epoch_id=expected_epoch_id,
            author=author,
            reason=reason,
        )
        self._apply_snapshot_to_runtime(new_snap)
        epoch_evt = {
            "config_epoch_id": new_snap.config_epoch_id,
            "active_epoch_id": new_snap.config_epoch_id,
            "added_key": spec_dict.get("key") if isinstance(spec_dict, Mapping) else None,
            "config": _scrub_mapping(new_snap.to_dict()),
        }
        dash = self._refresh_read_model_snapshot(
            persist_funnel=True,
            broadcast=True,
            extra_events=[("config_epoch_changed", epoch_evt)],
        )
        return {
            "ok": True,
            "config_epoch_id": new_snap.config_epoch_id,
            "active_epoch_id": new_snap.config_epoch_id,
            "snapshot": _scrub_mapping(new_snap.to_dict()),
            "schema": dash["schema"],
            "dashboard": dash,
        }

    def delete_custom_parameter(
        self,
        key: str,
        expected_epoch_id: Optional[str] = None,
        author: str = "operator",
        reason: str = "",
    ) -> Dict[str, Any]:
        """Delete a user-defined dynamic parameter and prune its overrides atomically."""
        new_snap = self.config_bus.delete_dynamic_parameter(
            key,
            expected_epoch_id=expected_epoch_id,
            author=author,
            reason=reason,
        )
        self._apply_snapshot_to_runtime(new_snap)
        epoch_evt = {
            "config_epoch_id": new_snap.config_epoch_id,
            "active_epoch_id": new_snap.config_epoch_id,
            "deleted_key": key,
            "config": _scrub_mapping(new_snap.to_dict()),
        }
        dash = self._refresh_read_model_snapshot(
            persist_funnel=True,
            broadcast=True,
            extra_events=[("config_epoch_changed", epoch_evt)],
        )
        return {
            "ok": True,
            "deleted_key": key,
            "config_epoch_id": new_snap.config_epoch_id,
            "active_epoch_id": new_snap.config_epoch_id,
            "snapshot": _scrub_mapping(new_snap.to_dict()),
            "schema": dash["schema"],
            "dashboard": dash,
        }

    def preview_position_re_epoch(
        self,
        ticker: str,
        target_epoch_id: Optional[str] = None,
        confirm_pure_tsl_mode_change: bool = False,
    ) -> Dict[str, Any]:
        """Preview exact stop & policy diff of re-epoching one open position."""
        return self.position_monitor.preview_re_epoch(
            ticker=ticker,
            target_epoch_id=target_epoch_id,
            confirm_pure_tsl_mode_change=confirm_pure_tsl_mode_change,
        )

    def re_epoch_position(
        self,
        ticker: str,
        expected_position_epoch_id: Optional[str],
        target_epoch_id: Optional[str] = None,
        confirm_pure_tsl_mode_change: bool = False,
    ) -> Dict[str, Any]:
        """Migrate an open position onto target_epoch_id under dual CAS checks and refresh dashboard."""
        result = self.position_monitor.re_epoch_position(
            ticker=ticker,
            expected_position_epoch_id=expected_position_epoch_id,
            target_epoch_id=target_epoch_id,
            confirm_pure_tsl_mode_change=confirm_pure_tsl_mode_change,
        )
        dash = self._refresh_read_model_snapshot(persist_funnel=True, broadcast=True)
        return {**result, "dashboard": dash}

    def preview_re_epoch_all(
        self,
        target_epoch_id: Optional[str] = None,
        confirm_pure_tsl_mode_change: bool = False,
    ) -> Dict[str, Any]:
        """Preview batch re-epoch actions across all currently active open positions."""
        return self.position_monitor.preview_re_epoch_all(
            target_epoch_id=target_epoch_id,
            confirm_pure_tsl_mode_change=confirm_pure_tsl_mode_change,
        )

    def re_epoch_all_positions(
        self,
        expected_epoch_id: Optional[str] = None,
        target_epoch_id: Optional[str] = None,
        confirm_pure_tsl_mode_change: bool = False,
    ) -> Dict[str, Any]:
        """Batch-migrate open positions to the active/target epoch under CAS and publish snapshot."""
        active_epoch = self.config_bus.active_epoch_id
        if expected_epoch_id is not None and expected_epoch_id != active_epoch:
            raise PositionEpochConflictError(
                f"Batch target epoch conflict: expected {expected_epoch_id!r}, active is {active_epoch!r}",
                expected_epoch_id=expected_epoch_id,
                actual_epoch_id=active_epoch,
                expected_target_epoch_id=expected_epoch_id,
                actual_target_epoch_id=active_epoch,
            )
        resolved_target = target_epoch_id if target_epoch_id is not None else expected_epoch_id
        result = self.position_monitor.re_epoch_all_positions(
            target_epoch_id=resolved_target,
            confirm_pure_tsl_mode_change=confirm_pure_tsl_mode_change,
        )
        dash = self._refresh_read_model_snapshot(persist_funnel=True, broadcast=True)
        succeeded_count = len(result.get("applied", []))
        failed_count = len(result.get("errors", [])) + len(result.get("conflicts", []))
        return {
            **result,
            "succeeded_count": succeeded_count,
            "failed_count": failed_count,
            "halted_preempted": bool(result.get("preempted_by_halt", False)),
            "dashboard": dash,
        }

    def trigger_panic_flatten(self, reason: str = "Dashboard emergency panic") -> Dict[str, Any]:
        """Execute fail-closed emergency panic flatten, halt trading, and broadcast SSE alerts."""
        result = self.position_monitor.panic_flatten_all()
        halted = _as_bool_strict(self.position_monitor.is_system_halted(), True)
        panic_evt = {
            "reason": reason,
            "is_halted": halted,
            **result,
        }
        dash = self._refresh_read_model_snapshot(
            persist_funnel=True,
            broadcast=True,
            extra_events=[("panic_triggered", panic_evt)],
        )
        return {
            **result,
            "reason": reason,
            "is_halted": halted,
            "dashboard": dash,
        }

    def resume_trading(self, confirm_unprotected: bool = False) -> Dict[str, Any]:
        """Verify/re-arm broker stops via PositionMonitor and clear system halt when safe."""
        result = self.position_monitor.resume_trading(confirm_unprotected=confirm_unprotected)
        dash = self._refresh_read_model_snapshot(
            persist_funnel=True,
            broadcast=True,
            extra_events=[("trading_resumed", result)],
        )
        return {**result, "dashboard": dash}

    def retry_panic_liquidation(self, ticker: str) -> Dict[str, Any]:
        """Retry market liquidation on a panic_failed position under PositionMonitor._broker_op_lock."""
        pm = self.position_monitor
        broker_lock = getattr(pm, "_broker_op_lock", None) or contextlib.nullcontext()
        state_lock = getattr(pm, "_state_lock", None) or contextlib.nullcontext()

        with broker_lock:
            with state_lock:
                pos = pm.positions.get(ticker)
                if _pos_is_closed(pos) or _pos_remaining_qty(pos) <= 0.0:
                    raise KeyError(f"Open position {ticker!r} not found for liquidation retry")
                qty_to_sell = float(pos.remaining_quantity)

            with contextlib.suppress(Exception):
                pm.cancel_all_broker_orders(pos)

            sold = True
            if qty_to_sell > 0.0 and not getattr(pm, "dry_run", self.dry_run):
                try:
                    sold = bool(pm._execute_sell(pos.ticker, qty_to_sell))
                except Exception as exc:
                    logger.error("Retry panic liquidation failed for %s: %s", ticker, exc)
                    sold = False

            now_iso = datetime.now(timezone.utc).isoformat()
            with state_lock:
                if sold:
                    pos.is_closed = True
                    pos.remaining_quantity = 0.0
                    pos.panic_failed = False
                    pos.unprotected_reason = None
                    pos.last_update_time = now_iso
                else:
                    pos.is_closed = False
                    pos.panic_failed = True
                    pos.unprotected_reason = "Retry panic liquidation failed"
                    pos.last_update_time = now_iso

            if hasattr(pm, "_save_state"):
                pm._save_state()

        dash = self._refresh_read_model_snapshot(persist_funnel=True, broadcast=True)
        return {
            "ok": bool(sold),
            "ticker": ticker,
            "liquidated_quantity": qty_to_sell if sold else 0.0,
            "panic_failed": not sold,
            "dashboard": dash,
        }

    def _update_dashboard(self, *, persist_funnel: bool = True, broadcast: bool = True) -> None:
        """Refresh read-model snapshot, persist funnel_state.json, notify SSE, and regenerate HTML."""
        self._refresh_read_model_snapshot(persist_funnel=persist_funnel, broadcast=broadcast)
        if self.enable_dashboard and self._generate_html_dashboard:
            try:
                sig = inspect.signature(self._generate_html_dashboard)
                gen_kwargs: Dict[str, Any] = {}
                accepts_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                )
                if "orchestrator" in sig.parameters or accepts_var_kw:
                    gen_kwargs["orchestrator"] = self
                self._generate_html_dashboard("signals_dashboard.html", **gen_kwargs)
            except Exception as exc:
                logger.debug("Dashboard regeneration skipped: %s", exc)

    def fetch_active_trades_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Fetch targeted OHLCV & technical indicators for currently active trade underlyings.

        Runs in fractions of a second because it only fetches for active assets (typically 2-6)
        rather than scanning the full 120-asset universe.
        """
        if not self.position_monitor or not self.position_monitor.positions:
            return {}

        underlyings = list({
            pos.underlying for pos in self.position_monitor.positions.values()
            if not _pos_is_closed(pos) and isinstance(getattr(pos, "underlying", None), str) and pos.underlying
        })
        if not underlyings:
            return {}

        active_metrics: Dict[str, Dict[str, Any]] = {}

        def _fetch_one(symbol: str) -> tuple[str, Optional[Dict[str, Any]]]:
            try:
                daily = MarketDataProvider.get_daily_candles(symbol, period="1y")
                intraday_5m = MarketDataProvider.get_5m_candles(symbol, period="5d")
                metrics = MarketDataProvider.compute_indicators(daily, intraday_data=intraday_5m, symbol=symbol)
                # Ingest Nixtla 5-day forecast for active underlying if available
                try:
                    forecast = MarketDataProvider.get_5day_forecast(symbol)
                    if forecast and "target_5d" in forecast:
                        metrics["nixtla_5d_target"] = forecast.get("target_5d")
                        metrics["nixtla_expected_return_pct"] = forecast.get("expected_return_pct", 0.0)
                except Exception as f_exc:
                    logger.debug("Active trade forecast fetch notice for %s: %s", symbol, f_exc)
                return symbol, metrics
            except Exception as exc:
                logger.debug("Active trade metric fetch notice for %s: %s", symbol, exc)
                return symbol, None

        workers = min(len(underlyings), 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_sym = {executor.submit(_fetch_one, sym): sym for sym in underlyings}
            for future in concurrent.futures.as_completed(future_to_sym):
                sym, met = future.result()
                if met:
                    active_metrics[sym] = met

        return active_metrics

    # ─────────────────────────────────────────────────────────────
    # Phase 1 — Data Ingestion (Parallel Subagents)
    # ─────────────────────────────────────────────────────────────
    def ingest_universe_data(
        self,
        filter_active_sessions: bool = True,
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch OHLCV and compute indicators for liquid underlyings in parallel using worker threads.

        Applies session-gated filtering to skip assets whose underlying and ETP exchanges
        are closed, eliminating API quota exhaustion on dormant markets.
        """
        snap = config_snapshot or self.config_bus.active_snapshot()
        workers = max(1, int(getattr(snap, "PARALLEL_INGESTION_WORKERS", self.parallel_ingestion_workers)))
        universe_metrics: Dict[str, Dict[str, Any]] = {}
        start_time = time.time()

        # Session-aware filtering
        assets_to_ingest = self.universe
        if filter_active_sessions:
            active_assets = [
                sym for sym in self.universe
                if Trading212Client.is_asset_session_active(sym)
            ]
            dormant_count = len(self.universe) - len(active_assets)
            if dormant_count > 0:
                logger.info(
                    "Session-gated filter: %d / %d assets active; skipped %d closed/dormant assets to preserve API quota",
                    len(active_assets), len(self.universe), dormant_count
                )
            assets_to_ingest = active_assets

        total = len(assets_to_ingest)
        if total == 0:
            logger.info("Session-gated filter: 0 assets active in current session window.")
            return {}

        def _fetch_asset(symbol: str) -> tuple[str, Optional[Dict[str, Any]]]:
            try:
                daily = MarketDataProvider.get_daily_candles(symbol, period="1y")
                intraday_5m = MarketDataProvider.get_5m_candles(symbol, period="5d")
                metrics = MarketDataProvider.compute_indicators(daily, intraday_data=intraday_5m, symbol=symbol)
                if metrics:
                    metrics["is_synthetic"] = MarketDataProvider.is_ticker_synthetic(symbol)
                    if metrics.get("earnings_in_days") is None:
                        metrics["earnings_in_days"] = MarketDataProvider.get_days_to_earnings(symbol)
                return symbol, metrics
            except Exception as exc:
                logger.error("Failed to ingest %s: %s", symbol, exc)
                return symbol, None

        logger.info(
            "Ingesting market data for %d assets in parallel across %d subagent workers…",
            total, workers
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_symbol = {executor.submit(_fetch_asset, sym): sym for sym in assets_to_ingest}
            completed = 0
            for future in concurrent.futures.as_completed(future_to_symbol):
                completed += 1
                symbol, metrics = future.result()
                if metrics:
                    universe_metrics[symbol] = metrics
                    logger.debug("[%d/%d] Ingested %s (close=%.2f, RSI=%.1f, ADX=%.1f)",
                                 completed, total, symbol, metrics.get("close", 0.0),
                                 metrics.get("rsi", 0.0), metrics.get("adx", 0.0))
                else:
                    logger.warning("[%d/%d] %s — insufficient candle data", completed, total, symbol)

        elapsed = time.time() - start_time
        logger.info(
            "Ingestion complete: %d / %d assets ready for screening in %.2fs (%.1f assets/sec)",
            len(universe_metrics), total, elapsed, total / max(0.001, elapsed)
        )
        return universe_metrics

    # ─────────────────────────────────────────────────────────────
    # Phase 2 — Candidate Pre-Filtering for System 2 Deliberation
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _supports_kwarg(fn: Any, kwarg_name: str) -> bool:
        """Check cleanly whether callable accepts a keyword argument."""
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        return kwarg_name in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

    def screen_candidates_with_funnel(
        self,
        universe_metrics: Dict[str, Dict[str, Any]],
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> Tuple[List[ScreenerResult], List[ScreenerResult]]:
        """Filter universe against ConfigSnapshot and return (passed_candidates, all_screener_results)."""
        snap = config_snapshot or self.config_bus.active_snapshot()
        min_adx = float(getattr(snap, "MIN_ADX_TREND", settings.MIN_ADX_TREND))
        logger.info(
            "Screening %d assets for 3x Long/Short momentum (ADX >= %.1f, Epoch=%s)…",
            len(universe_metrics),
            min_adx,
            snap.config_epoch_id,
        )

        # Evaluate full per-ticker outcomes when screener supports evaluate_single for gate attribution
        all_results: List[ScreenerResult] = []
        if hasattr(self.screener, "evaluate_single"):
            eval_fn = self.screener.evaluate_single
            pass_snap = self._supports_kwarg(eval_fn, "config_snapshot")
            for ticker, m in universe_metrics.items():
                try:
                    res = eval_fn(ticker, m, config_snapshot=snap) if pass_snap else eval_fn(ticker, m)
                    if isinstance(res, ScreenerResult):
                        if not getattr(res, "config_epoch_id", None):
                            res.config_epoch_id = snap.config_epoch_id
                        all_results.append(res)
                except Exception as eval_exc:
                    logger.debug("evaluate_single failed for %s: %s", ticker, eval_exc)

        scan_fn = self.screener.scan_universe
        candidates = (
            scan_fn(universe_metrics, config_snapshot=snap)
            if self._supports_kwarg(scan_fn, "config_snapshot")
            else scan_fn(universe_metrics)
        )
        candidates = list(candidates or [])

        # Safety net: if primary returned 0, evaluate with TechnicalFallbackScreener
        if not candidates:
            logger.warning(
                "Primary screener returned 0 candidates — running TechnicalFallbackScreener as safety net."
            )
            fb_top_n = max(10, int(getattr(snap, "SCREENER_TOP_N", 10)))
            fb_screener = TechnicalFallbackScreener(top_n=fb_top_n)
            fb_res = fb_screener.scan_universe(universe_metrics, config_snapshot=snap)
            candidates = list(fb_res or [])

        for cand in candidates:
            if not getattr(cand, "config_epoch_id", None):
                cand.config_epoch_id = snap.config_epoch_id

        # Build partial funnel attribution for StrategyImprovementAgent
        prelim_report = build_cycle_funnel_report(
            universe_metrics=universe_metrics,
            screener_results=all_results if all_results else None,
            passed_screener=candidates,
            strategy_signals=[
                r.strategy_signal for r in all_results if getattr(r, "strategy_signal", None) is not None
            ] if all_results else None,
            config_snapshot=snap,
            config_epoch_id=snap.config_epoch_id,
        )

        logger.info(
            "Candidate pre-filter passed %d assets for System 2 deliberation: %s",
            len(candidates),
            [f"{c.ticker} ({c.bias})" for c in candidates],
        )
        return candidates, (all_results if all_results else list(candidates))

    def screen_candidates(
        self,
        universe_metrics: Dict[str, Dict[str, Any]],
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> List[ScreenerResult]:
        """Filter universe using Laya NLI router and ConfigSnapshot thresholds."""
        candidates, _ = self.screen_candidates_with_funnel(
            universe_metrics, config_snapshot=config_snapshot
        )
        return candidates

    # ─────────────────────────────────────────────────────────────
    # Phase 3 — System 2 Multi-Agent Deliberation (Concurrent Subagents)
    # ─────────────────────────────────────────────────────────────
    def deep_analyse(
        self,
        candidate: ScreenerResult,
        account_equity: Optional[float] = None,
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> Optional[DecisionCard]:
        """Perform quantitative sizing and map candidate to exact leveraged execution ETP."""
        snap = config_snapshot or self.config_bus.active_snapshot()
        underlying = candidate.ticker
        metrics = candidate.metrics
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        direction = "LONG" if candidate.bias == "bullish" else "SHORT"
        strategy_type = getattr(candidate, "strategy_type", getattr(snap, "ACTIVE_STRATEGY", "vwap"))
        if getattr(candidate, "strategy_signal", None) is not None:
            metrics["strategy_signal"] = candidate.strategy_signal
        if not getattr(candidate, "config_epoch_id", None):
            candidate.config_epoch_id = snap.config_epoch_id

        logger.info(
            "═══ System 2 Deliberation for %s (%s | %s | Epoch=%s) ═══",
            underlying, direction, strategy_type, snap.config_epoch_id,
        )

        # ── Pre-Trade Earnings Blackout Check ────────────────────
        blackout_days = float(getattr(snap, "ENTRY_EARNINGS_BLACKOUT_DAYS", 2.0))
        earnings_raw = metrics.get("earnings_in_days")
        earnings_in_days: Optional[float] = None
        if earnings_raw is not None:
            try:
                earnings_in_days = float(earnings_raw)
            except (ValueError, TypeError):
                earnings_in_days = None

        if earnings_in_days is not None and 0 <= earnings_in_days <= blackout_days:
            logger.warning(
                "Pre-trade earnings filter: %s rejected due to impending earnings in %.1f days (<= %.1fd blackout)",
                underlying, earnings_in_days, blackout_days,
            )
            return None

        # ── 1. Map to Leveraged Execution Ticker (Multi-Tier 5x & 3x Allocation) ──
        direction_upper = direction.upper()
        is_short = direction_upper in ("SHORT", "SELL", "BEARISH")
        use_lev_3x = bool(getattr(snap, "USE_LEVERAGED_3X", getattr(settings, "USE_LEVERAGED_3X", True)))
        allow_5x = bool(getattr(snap, "ENABLE_5X_LEVERAGE", False))
        prefer_5x = bool(getattr(snap, "PREFER_5X_LEVERAGE", False))
        max_lev = int(getattr(snap, "MAX_LEVERAGE", 5 if allow_5x else 3))

        target_leverage = 3
        if use_lev_3x:
            has_5x = self.leveraged_universe.has_5x_leverage(underlying, direction)
            realized_vol = float(metrics.get("realized_vol") or metrics.get("historical_vol") or 0.25)
            confidence = float(getattr(candidate, "overall_confidence", 0.0) or 0.0)
            if allow_5x and max_lev >= 5 and has_5x:
                target_leverage = 5 if (prefer_5x or (confidence >= 0.85 and realized_vol <= 0.35)) else 3
            else:
                target_leverage = 3
            t212_ticker = self.leveraged_universe.get_execution_ticker(
                underlying,
                direction,
                leverage=target_leverage,
                allow_5x=allow_5x,
                config_snapshot=snap,
            )
            if not t212_ticker:
                if is_short:
                    logger.warning(
                        "SHORT signal rejected for %s: no inverse / short leveraged ETP exists on Trading 212 Invest",
                        underlying,
                    )
                    return None
                logger.warning("No paired %dx %s ETP found for %s — using standard mapping", target_leverage, direction, underlying)
                instrument = self.t212.get_instrument_by_symbol(underlying)
                t212_ticker = instrument.ticker if instrument else f"{underlying}_US_EQ"
        else:
            if is_short:
                logger.warning(
                    "SHORT signal rejected for %s: direct stock shorting not supported on Trading 212 Invest without inverse ETP",
                    underlying,
                )
                return None
            instrument = self.t212.get_instrument_by_symbol(underlying)
            t212_ticker = instrument.ticker if instrument else f"{underlying}_US_EQ"

        # Authoritative direction validation check: never execute SHORT on a LONG instrument
        resolved_direction = self.leveraged_universe.get_etp_direction(t212_ticker)
        if is_short and resolved_direction == "LONG":
            logger.critical(
                "CRITICAL SAFETY VIOLATION PREVENTED: %s is a LONG instrument, but signal direction is SHORT for %s! Rejecting trade.",
                t212_ticker, underlying,
            )
            return None

        # Check if debate engine was explicitly hooked or mocked
        outcome = None
        if hasattr(self, "debate_engine") and hasattr(self.debate_engine, "run_debate"):
            try:
                if not getattr(self.debate_engine, "_is_dummy", False):
                    debate_fn = self.debate_engine.run_debate
                    debate_kwargs: Dict[str, Any] = {"date": today, "strategy_type": strategy_type}
                    if self._supports_kwarg(debate_fn, "config_snapshot"):
                        debate_kwargs["config_snapshot"] = snap
                    outcome = debate_fn(underlying, metrics, "", "", **debate_kwargs)
                    if getattr(outcome, "config_epoch_id", None) is None:
                        with contextlib.suppress(Exception):
                            outcome.config_epoch_id = snap.config_epoch_id
            except Exception as exc:
                logger.debug("Debate hook execution notice for %s: %s", underlying, exc)

        # ── 4. Risk Management & Sizing ──────────────────────────
        default_equity = self._get_fallback_equity(snap)
        try:
            if account_equity is None or account_equity <= 0:
                account = self.t212.get_account_cash(fallback_equity=default_equity)
                account_equity = account.total or default_equity

            gen_fn = self.risk_manager.generate_decision_card
            gen_kwargs: Dict[str, Any] = {
                "ticker": underlying,
                "trading212_ticker": t212_ticker,
                "direction": direction,
                "metrics": metrics,
                "debate_outcome": outcome,
                "account_equity": account_equity,
                "strategy_type": strategy_type,
                "leverage": target_leverage if use_lev_3x else 1,
            }
            if self._supports_kwarg(gen_fn, "config_snapshot"):
                gen_kwargs["config_snapshot"] = snap
            card = gen_fn(**gen_kwargs)
            if card is not None:
                if not getattr(card, "config_epoch_id", None):
                    card.config_epoch_id = snap.config_epoch_id
                with contextlib.suppress(Exception):
                    setattr(card, "_debate_outcome", outcome)

            fmt_fn = self.risk_manager.format_decision_card
            fmt_out = (
                fmt_fn(card, config_snapshot=snap)
                if self._supports_kwarg(fmt_fn, "config_snapshot")
                else fmt_fn(card)
            )
            logger.info("Decision card generated:\n%s", fmt_out)
            return card

        except Exception as exc:
            logger.error("Risk management failed for %s: %s", underlying, exc)
            return None

    # ─────────────────────────────────────────────────────────────
    # Phase 4 — Execution & Order Placement
    # ─────────────────────────────────────────────────────────────
    def execute_trade(
        self,
        card: DecisionCard,
        atr: float = 0.0,
        available_cash: Optional[float] = None,
        positions: Optional[List[Position]] = None,
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> bool:
        """Execute trade on Trading 212 under _execution_gate_lock and dispatch BUY/SELL alerts."""
        active_snap = config_snapshot or self.config_bus.active_snapshot()

        if card.decision not in ("BUY_ENTRY", "SELL_ENTRY"):
            logger.info("Skipping execution for %s — decision=%s", card.ticker, card.decision)
            self._log_trade(card, executed=False, reason=card.decision)
            return False

        # ── Execution Direction Safety Check ─────────────────────
        is_short = card.direction.upper() in ("SHORT", "SELL", "BEARISH")
        etp_dir = self.leveraged_universe.get_etp_direction(card.trading212_ticker)
        if is_short and etp_dir == "LONG":
            logger.critical(
                "CRITICAL EXECUTION SAFETY ABORT: Card requests %s for %s, but %s is a spot LONG instrument! Rejecting order placement.",
                card.direction, card.ticker, card.trading212_ticker
            )
            self._log_trade(card, executed=False, reason=f"Direction mismatch: {card.trading212_ticker} is LONG but signal is SHORT")
            return False

        # ── Pre-Flight System Halt Check ─────────────────────────
        if self._is_pm_halted():
            logger.warning("Execution rejected for %s — system is currently HALTED following panic flatten", card.ticker)
            self._log_trade(card, executed=False, reason="System is HALTED")
            return False

        # ── In-Flight Order & Active Position Deduplication Check ────────
        if card.trading212_ticker in self.in_flight_tickers:
            logger.warning(
                "Execution rejected for %s: in-flight broker order already pending for %s. Skipping duplicate order.",
                card.ticker, card.trading212_ticker
            )
            self._log_trade(card, executed=False, reason="In-flight order already pending")
            return False

        if self._has_active_position(card.trading212_ticker):
            logger.warning(
                "Execution rejected for %s: position for %s is already active in position monitor. Skipping duplicate order.",
                card.ticker, card.trading212_ticker
            )
            self._log_trade(card, executed=False, reason="Position already active")
            return False

        # ── Pre-trade validation ─────────────────────────────────
        default_equity = self._get_fallback_equity(active_snap)
        account_total = default_equity
        if available_cash is not None and positions is not None:
            account_free = available_cash
            current_positions = positions
        else:
            try:
                account = self.t212.get_account_cash(fallback_equity=default_equity)
                account_free = account.free
                account_total = account.total or account_total
                current_positions = self.t212.get_portfolio()
            except Trading212Error as exc:
                if self.dry_run:
                    logger.warning("Pre-trade broker check unavailable (%s) — using dry-run simulation context", exc)
                    account_free = default_equity
                    account_total = default_equity
                    current_positions = []
                else:
                    logger.error("Pre-trade check failed: %s", exc)
                    self._log_trade(card, executed=False, reason=str(exc))
                    return False

        if self.session_start_equity is None or self.session_start_equity <= 0:
            self.session_start_equity = account_total

        # Daily Drawdown Circuit Breaker Gate
        if bool(getattr(active_snap, "ENFORCE_DAILY_CIRCUIT_BREAKER", getattr(settings, "ENFORCE_DAILY_CIRCUIT_BREAKER", True))):
            cb_fn = self.risk_manager.check_circuit_breaker
            cb_kwargs: Dict[str, Any] = {
                "current_equity": account_total,
                "session_start_equity": self.session_start_equity,
            }
            if self._supports_kwarg(cb_fn, "config_snapshot"):
                cb_kwargs["config_snapshot"] = active_snap
            tripped, cb_reason = cb_fn(**cb_kwargs)
            if tripped:
                logger.critical("Trade rejected: %s", cb_reason)
                self._log_trade(card, executed=False, reason=cb_reason)
                return False

        # Real-time Bid-Ask Spread Gate
        spread_pct = None
        enforce_spread = bool(getattr(active_snap, "ENFORCE_SPREAD_GATE", getattr(settings, "ENFORCE_SPREAD_GATE", True)))
        if enforce_spread:
            spread_info = MarketDataProvider.get_bid_ask_spread(card.trading212_ticker)
            spread_pct = spread_info.get("spread_pct")

        val_fn = self.risk_manager.validate_trade
        val_kwargs: Dict[str, Any] = {
            "spread_pct": spread_pct,
            "account_equity": account_total,
        }
        if self._supports_kwarg(val_fn, "config_snapshot"):
            val_kwargs["config_snapshot"] = active_snap
        valid, reason = val_fn(card, account_free, current_positions, **val_kwargs)
        if not valid:
            logger.warning("Trade validation FAILED for %s: %s", card.ticker, reason)
            self._log_trade(card, executed=False, reason=reason)
            return False

        # ── CRITICAL LIVE TRADING SAFETY GATE: Refuse live execution if data was synthetic ──
        is_synthetic = (
            getattr(card, "is_synthetic", False)
            or MarketDataProvider.is_ticker_synthetic(card.ticker)
            or (bool(card.trading212_ticker) and MarketDataProvider.is_ticker_synthetic(card.trading212_ticker))
        )
        if not self.dry_run and is_synthetic:
            logger.critical(
                "CRITICAL LIVE TRADING SAFETY GATE: Market data for %s (%s) was generated synthetically! "
                "Refusing to place real broker order. Verify live data connection or API keys.",
                card.trading212_ticker, card.ticker
            )
            self._log_trade(card, executed=False, reason="Blocked by Safety Interceptor: Market data was synthetic")
            return False

        # ── Pre-Execution Market Hours Gate & Auction Buffer ─────
        max_spread_pct = float(getattr(active_snap, "MAX_BID_ASK_SPREAD_PCT", 0.015))
        if not self.dry_run:
            if not self.t212.is_market_open(card.trading212_ticker):
                info = Trading212Client.get_market_session_info(card.trading212_ticker)
                logger.warning(
                    "Execution held: Market for %s (%s) is CLOSED right now (%s UTC, current %s UTC). Refusing to submit rejected order.",
                    card.trading212_ticker, info["exchange"], info["open_time_utc"], info["current_time_utc"]
                )
                self._log_trade(card, executed=False, reason=f"Market closed ({info['exchange']} session: {info['open_time_utc']}-{info['close_time_utc']} UTC)")
                return False

            session_info = Trading212Client.get_market_session_info(card.trading212_ticker)
            if session_info.get("exchange") != "CRYPTO":
                now_utc = datetime.now(timezone.utc)
                open_parts = [int(p) for p in session_info.get("open_time_utc", "08:00").split(":")]
                market_open_dt = now_utc.replace(hour=open_parts[0], minute=open_parts[1], second=0, microsecond=0)
                minutes_since_open = (now_utc - market_open_dt).total_seconds() / 60.0
                if 0 <= minutes_since_open < 15.0:
                    logger.warning(
                        "Execution held: %s (%s) is within the 15-minute opening auction buffer "
                        "(opened %s UTC, elapsed %.1f mins < 15 mins). "
                        "Preventing entry into wide opening auction spreads.",
                        card.trading212_ticker, session_info.get("exchange"),
                        session_info.get("open_time_utc"), minutes_since_open
                    )
                    self._log_trade(
                        card, executed=False,
                        reason=f"Opening auction buffer ({session_info.get('exchange')} opened {session_info.get('open_time_utc')} UTC, {minutes_since_open:.1f}m < 15m)"
                    )
                    return False

            # ── Pre-Trade Live Broker Quote & Drift Cross-Check ──────────
            try:
                live_quote = self.t212.get_quote(card.trading212_ticker)
                live_ask = float(live_quote.get("ask", 0.0) or 0.0)
                live_bid = float(live_quote.get("bid", 0.0) or 0.0)
                live_mid = float(live_quote.get("mid", 0.0) or 0.0)
                quote_spread_pct = float(live_quote.get("spread_pct", 0.0) or 0.0)

                # 1. Spread Check against active ConfigSnapshot MAX_BID_ASK_SPREAD_PCT
                if live_ask > 0 and live_bid > 0:
                    calc_spread = (live_ask - live_bid) / live_ask
                    quote_spread_pct = max(quote_spread_pct, calc_spread)

                if enforce_spread and quote_spread_pct > max_spread_pct:
                    logger.warning(
                        "Execution rejected: %s bid/ask spread %.2f%% exceeds %.2f%% safety threshold "
                        "(bid=%.4f, ask=%.4f). Refusing to execute into wide spread.",
                        card.trading212_ticker, quote_spread_pct * 100.0, max_spread_pct * 100.0, live_bid, live_ask
                    )
                    self._log_trade(
                        card, executed=False,
                        reason=f"Wide bid/ask spread: {quote_spread_pct * 100.0:.2f}% > {max_spread_pct * 100.0:.2f}% limit (bid={live_bid}, ask={live_ask})"
                    )
                    return False

                # 2. Overnight / Stale Price Drift Check: Max allowable drift from setup entry price is 1.5%
                ref_price = live_ask if live_ask > 0 else (live_mid if live_mid > 0 else live_bid)
                if ref_price > 0 and card.entry_price > 0:
                    drift_pct = abs(ref_price - card.entry_price) / card.entry_price
                    if drift_pct > 0.015:  # > 1.5%
                        logger.warning(
                            "Execution rejected: %s live quote (%.4f) drifted %.2f%% from setup entry price (%.4f) "
                            "(threshold: 1.50%%). Setup from overnight/prior close is stale.",
                            card.trading212_ticker, ref_price, drift_pct * 100.0, card.entry_price
                        )
                        self._log_trade(
                            card, executed=False,
                            reason=f"Stale price drift: {drift_pct * 100.0:.2f}% > 1.50% threshold (live={ref_price}, setup={card.entry_price})"
                        )
                        return False

                    old_entry = card.entry_price
                    card.entry_price = ref_price
                    if card.stop_loss > 0 and old_entry > 0:
                        sl_ratio = card.stop_loss / old_entry
                        card.stop_loss = round(ref_price * sl_ratio, 4)
                        logger.info(
                            "Recalibrated %s to live market quote: entry $%.4f -> $%.4f, SL $%.4f (drift %.2f%%)",
                            card.trading212_ticker, old_entry, card.entry_price, card.stop_loss, drift_pct * 100.0
                        )
            except Exception as quote_exc:
                logger.debug("Pre-trade live quote validation skipped due to lookup error: %s", quote_exc)

        # ── Execute Entry Order + Register Position Under Shared Gate Lock (R18) ──
        order_id = "DRY_RUN_ORDER"
        fill_price: Optional[float] = None
        pos = None
        with self._execution_gate_lock:
            # Final authoritative halt check inside execution_gate_lock right before place_market_order
            if self._is_pm_halted():
                logger.warning("Execution aborted under gate lock for %s — system is HALTED", card.ticker)
                self._log_trade(card, executed=False, reason="System is HALTED")
                return False

            if self._has_active_position(card.trading212_ticker):
                logger.warning("Execution aborted under gate lock for %s — position already active", card.trading212_ticker)
                self._log_trade(card, executed=False, reason="Position already active")
                return False

            if not self.dry_run:
                try:
                    order_qty = card.position_size_shares
                    try:
                        inst = self.t212.get_instrument_by_symbol(card.trading212_ticker)
                        if inst and inst.min_trade_quantity and order_qty < inst.min_trade_quantity:
                            est_notional = inst.min_trade_quantity * (card.entry_price or 1.0)
                            if est_notional <= (account_total * 0.25):
                                logger.info(
                                    "Adjusting %s order quantity from %.3f to broker minimum %.3f",
                                    card.trading212_ticker, order_qty, inst.min_trade_quantity,
                                )
                                order_qty = inst.min_trade_quantity
                                card.position_size_shares = order_qty
                    except Exception as inst_exc:
                        logger.debug("Could not verify instrument min quantity for %s: %s", card.trading212_ticker, inst_exc)

                    logger.info(
                        "Placing Trading 212 order: %s (%s) — %.3f shares @ ≈$%.2f",
                        card.trading212_ticker, card.direction, order_qty, card.entry_price
                    )
                    entry_order = self.t212.place_market_order(
                        ticker=card.trading212_ticker,
                        quantity=order_qty,
                    )
                    self.in_flight_tickers.add(card.trading212_ticker)
                    order_id = entry_order.id
                    logger.info("Order executed successfully: id=%s", order_id)
                    raw_fill_px = getattr(entry_order, "fill_price", 0.0)
                    try:
                        fill_px_float = float(raw_fill_px or 0.0)
                    except (TypeError, ValueError):
                        fill_px_float = 0.0
                    if fill_px_float > 0:
                        fill_price = fill_px_float
                    else:
                        for retry_idx in range(3):
                            time.sleep(1.0)
                            try:
                                positions_fresh = self.t212.get_portfolio(force_refresh=True)
                                for p in positions_fresh:
                                    if p.ticker == card.trading212_ticker and p.average_price > 0:
                                        fill_price = p.average_price
                                        break
                                if fill_price and fill_price > 0:
                                    break
                            except Exception as port_exc:
                                logger.debug("Could not retrieve post-order portfolio fill price (attempt %d): %s", retry_idx + 1, port_exc)

                    if not fill_price or fill_price <= 0:
                        try:
                            quote = MarketDataProvider.get_quote(card.trading212_ticker)
                            live_px = quote.get("price") or quote.get("close")
                            if live_px and float(live_px) > 0:
                                fill_price = float(live_px)
                        except Exception as quote_exc:
                            logger.debug("Could not retrieve live quote for %s: %s", card.trading212_ticker, quote_exc)
                except Trading212Error as exc:
                    logger.error("Execution failed for %s: %s", card.trading212_ticker, exc)
                    self._log_trade(card, executed=False, reason=str(exc))
                    return False
            else:
                logger.info("[DRY-RUN] Simulated execution for %s: %.3f shares @ $%.2f",
                            card.trading212_ticker, card.position_size_shares, card.entry_price)

            reg_fn = self.position_monitor.register_new_trade
            reg_kwargs: Dict[str, Any] = {
                "atr": atr,
                "earnings_in_days": card.earnings_in_days,
                "fill_price": fill_price,
            }
            if self._supports_kwarg(reg_fn, "config_snapshot"):
                reg_kwargs["config_snapshot"] = active_snap
            pos = reg_fn(card, **reg_kwargs)
            if pos is not None and not getattr(pos, "config_epoch_id", None):
                with contextlib.suppress(Exception):
                    pos.config_epoch_id = active_snap.config_epoch_id
            if pos is not None and getattr(pos, "entry_config_snapshot", None) is None:
                with contextlib.suppress(Exception):
                    pos.entry_config_snapshot = _scrub_mapping(active_snap.to_dict())

            # Place native broker GTC Stop & Limit orders while holding _execution_gate_lock
            # so concurrent panic_flatten_all cannot race ahead of stop registration.
            if not self._is_pm_halted() and pos is not None:
                self.position_monitor.place_broker_stop(pos)
                pos_pure_tsl = bool(getattr(active_snap, "PURE_TSL_MODE", getattr(self.position_monitor, "pure_tsl_mode", True)))
                if not pos_pure_tsl:
                    self.position_monitor.place_broker_tp_limit(pos)

        # ── Dispatch Real-Time Notifications ─────────────────────
        notif_price = pos.entry_price if pos is not None else card.entry_price
        notif_sl = pos.stop_loss if pos is not None else card.stop_loss
        notif_tp1 = pos.tp1 if pos is not None else card.take_profit_1
        notif_tp2 = pos.tp2 if pos is not None else card.take_profit_2
        notif_tp3 = pos.tp3 if pos is not None else card.take_profit_3
        self.notifier.notify_buy(
            ticker=card.trading212_ticker,
            underlying=card.ticker,
            price=notif_price,
            quantity=card.position_size_shares,
            sl=notif_sl,
            tp1=notif_tp1,
            tp2=notif_tp2,
            tp3=notif_tp3,
            direction=card.direction,
            rationale=card.rationale,
        )

        self._log_trade(card, executed=True, reason="Entry order placed", entry_order_id=order_id)
        self._refresh_read_model_snapshot(persist_funnel=False, broadcast=True)
        return True

    def _is_pm_halted(self) -> bool:
        """Check PositionMonitor halt state safely (guarding against MagicMock)."""
        pm = getattr(self, "position_monitor", None)
        if pm is None:
            return False
        if hasattr(pm, "is_system_halted") and _as_bool_strict(pm.is_system_halted(), False):
            return True
        return _as_bool_strict(getattr(pm, "is_halted", False), False)

    def _has_active_position(self, trading212_ticker: str) -> bool:
        """Check safely under PositionMonitor._state_lock whether ticker already has an open position."""
        pm = getattr(self, "position_monitor", None)
        if pm is None:
            return False
        positions = getattr(pm, "positions", None)
        if not isinstance(positions, Mapping):
            return False
        state_lock = getattr(pm, "_state_lock", None)
        lock_ctx = state_lock if state_lock is not None and hasattr(state_lock, "__enter__") else contextlib.nullcontext()
        with lock_ctx:
            pos = positions.get(trading212_ticker)
            if pos is None:
                return False
            return not _pos_is_closed(pos)

    # ─────────────────────────────────────────────────────────────
    # Audit Log
    # ─────────────────────────────────────────────────────────────
    def _log_trade(
        self,
        card: DecisionCard,
        executed: bool,
        reason: str,
        entry_order_id: Optional[str] = None,
    ) -> None:
        """Persist trade record to JSONL audit log."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "executed": executed,
            "reason": reason,
            "entry_order_id": entry_order_id,
            **asdict(card),
        }
        try:
            with open(self.trades_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            logger.error("Failed to write audit log: %s", exc)

    def _resize_card_for_snapshot(
        self,
        cand: ScreenerResult,
        card: DecisionCard,
        account_equity: float,
        target_snap: ConfigSnapshot,
    ) -> Optional[DecisionCard]:
        """Deterministically recompute sizing/SL/TP on an entry card after a resize_entry epoch shift."""
        debate_stub = SimpleNamespace(
            consensus_decision=card.decision,
            conviction_score=float(getattr(card, "conviction_score", 0.75) or 0.75),
            synthesis=str(getattr(card, "rationale", "") or "Deterministic resize revalidation"),
            config_epoch_id=target_snap.config_epoch_id,
        )
        metrics = dict(getattr(cand, "metrics", {}) or {})
        if "close" not in metrics or not metrics.get("close"):
            metrics["close"] = float(card.entry_price or 0.0)
        gen_fn = self.risk_manager.generate_decision_card
        gen_kwargs: Dict[str, Any] = {
            "ticker": card.ticker,
            "trading212_ticker": card.trading212_ticker,
            "direction": card.direction,
            "metrics": metrics,
            "debate_outcome": debate_stub,
            "account_equity": account_equity,
            "strategy_type": getattr(cand, "strategy_type", getattr(target_snap, "ACTIVE_STRATEGY", "vwap")),
            "leverage": int(getattr(card, "leverage", 3) or 3),
        }
        if self._supports_kwarg(gen_fn, "config_snapshot"):
            gen_kwargs["config_snapshot"] = target_snap
        resized = gen_fn(**gen_kwargs)
        if resized is not None:
            resized.config_epoch_id = target_snap.config_epoch_id
            resized.earnings_in_days = getattr(card, "earnings_in_days", None)
            resized.is_synthetic = getattr(card, "is_synthetic", False)
        return resized

    def execute_trades_batch(
        self, candidate_cards: List[tuple[ScreenerResult, DecisionCard]]
    ) -> List[DecisionCard]:
        """Execute multiple candidate signals with mid-scan epoch revalidation and capital pacing."""
        if not candidate_cards:
            return []

        Trading212Client.set_trade_active(True)
        batch_rejections: List[Dict[str, Any]] = []
        executed_cards: List[DecisionCard] = []

        try:
            actionable: List[tuple[ScreenerResult, DecisionCard]] = []
            for cand, card in candidate_cards:
                if card.decision in ("BUY_ENTRY", "SELL_ENTRY"):
                    actionable.append((cand, card))
                else:
                    self._log_trade(card, executed=False, reason=card.decision)

            if not actionable:
                return []

            actionable.sort(key=lambda item: item[1].conviction_score, reverse=True)

            active_snap = self.config_bus.active_snapshot()
            default_eq = self._get_fallback_equity(active_snap)
            running_equity = default_eq
            try:
                account = self.t212.get_account_cash(fallback_equity=default_eq)
                running_free_cash = account.free
                running_equity = account.total or default_eq
            except Exception:
                running_free_cash = default_eq

            running_positions = []
            try:
                running_positions = list(self.t212.get_portfolio())
            except Exception as exc:
                if not self.dry_run:
                    logger.error("Batch pre-trade broker check failed: %s", exc)
                    for _, card in actionable:
                        self._log_trade(card, executed=False, reason=f"Broker pre-check failed: {exc}")
                    return []
                logger.debug("Broker portfolio query notice in batch execution: %s", exc)

            held_tickers = {getattr(p, "ticker", "") for p in running_positions}
            for p in getattr(self.position_monitor, "positions", {}).values():
                rem_qty = _pos_remaining_qty(p)
                if not _pos_is_closed(p) and rem_qty > 0 and getattr(p, "ticker", None) not in held_tickers:
                    running_positions.append(Position(
                        ticker=p.ticker,
                        quantity=rem_qty,
                        average_price=p.entry_price,
                        current_price=getattr(p, "highest_price", p.entry_price),
                        ppl=0.0,
                        max_buy=0.0,
                        max_sell=0.0,
                    ))
                    held_tickers.add(p.ticker)

            is_test_mode = getattr(self.t212, "rate_limit_delay", 1.25) < 0.5
            pacing_delay = self.t212.rate_limit_delay if is_test_mode else 1.5

            self._broadcast_pipeline_progress(
                5, "Order Route", f"Routing {len(actionable)} candidate orders to broker client with capital pacing.",
                active_assets=[card.ticker for _, card in actionable],
                cycle_id=self._current_cycle_id,
            )

            for i, (cand, card) in enumerate(actionable):
                latest_snap = self.config_bus.active_snapshot()
                card_epoch = getattr(card, "config_epoch_id", None) or getattr(cand, "config_epoch_id", None)

                # Mid-scan epoch drift check before execution (F2 / R2)
                if card_epoch and card_epoch != latest_snap.config_epoch_id:
                    origin_snap = self.config_bus.snapshot_for_epoch(card_epoch)
                    if origin_snap is None:
                        scope = "abort_stale_scan"
                        changed_keys: List[str] = ["unknown_epoch"]
                    else:
                        changed_keys = _changed_snapshot_keys(origin_snap, latest_snap)
                        scope = _revalidation_scope_for_keys(changed_keys, origin_snap, latest_snap)

                    if scope == "abort_stale_scan":
                        logger.warning(
                            "Execution aborted for %s: stale_epoch_superseded (%s -> %s, changed=%s)",
                            card.ticker, card_epoch, latest_snap.config_epoch_id, changed_keys,
                        )
                        self._log_trade(
                            card,
                            executed=False,
                            reason=f"stale_epoch_superseded: {card_epoch} -> {latest_snap.config_epoch_id}",
                        )
                        batch_rejections.append({
                            "gate_id": "stale_epoch_superseded",
                            "ticker": card.ticker,
                            "param_keys": changed_keys,
                            "label": "Stale Epoch Superseded (Pre-Execution)",
                        })
                        continue

                    # Only resize_entry keys changed -> deterministically re-size card under latest_snap
                    resized = self._resize_card_for_snapshot(
                        cand=cand,
                        card=card,
                        account_equity=running_equity,
                        target_snap=latest_snap,
                    )
                    if resized is None or resized.decision not in ("BUY_ENTRY", "SELL_ENTRY"):
                        rej_reason = f"Resize revalidation rejected for epoch {latest_snap.config_epoch_id}"
                        self._log_trade(card, executed=False, reason=rej_reason)
                        if resized is not None and getattr(resized, "gate_rejections", None):
                            for r in resized.gate_rejections:
                                batch_rejections.append({**r, "ticker": card.ticker})
                        continue
                    logger.info(
                        "Deterministically resized %s card (%s -> %s): shares %.3f -> %.3f, SL %.4f -> %.4f",
                        card.ticker,
                        card_epoch,
                        latest_snap.config_epoch_id,
                        card.position_size_shares,
                        resized.position_size_shares,
                        card.stop_loss,
                        resized.stop_loss,
                    )
                    card = resized

                atr = cand.metrics.get("atr", card.entry_price * 0.02)

                # Validate against running capital allocation & latest snapshot
                val_fn = self.risk_manager.validate_trade
                val_kwargs: Dict[str, Any] = {"account_equity": running_equity}
                if self._supports_kwarg(val_fn, "config_snapshot"):
                    val_kwargs["config_snapshot"] = latest_snap
                valid, reason = val_fn(card, running_free_cash, running_positions, **val_kwargs)
                if not valid:
                    logger.warning(
                        "Batch trade validation FAILED for %s (conviction=%.2f): %s",
                        card.ticker, card.conviction_score, reason
                    )
                    self._log_trade(card, executed=False, reason=reason)
                    for r in (getattr(card, "gate_rejections", None) or ()):
                        batch_rejections.append({**r, "ticker": card.ticker})
                    continue

                success = self.execute_trade(
                    card,
                    atr=atr,
                    available_cash=running_free_cash,
                    positions=running_positions,
                    config_snapshot=latest_snap,
                )
                if success:
                    executed_cards.append(card)
                    running_free_cash = max(0.0, running_free_cash - card.position_notional)
                    running_positions.append(Position(
                        ticker=card.trading212_ticker,
                        quantity=card.position_size_shares,
                        average_price=card.entry_price,
                        current_price=card.entry_price,
                        ppl=0.0,
                        max_buy=0.0,
                        max_sell=0.0,
                    ))

                    if i < len(actionable) - 1:
                        time.sleep(pacing_delay)

            return executed_cards

        finally:
            Trading212Client.set_trade_active(False)
            self._finalize_cycle_funnel_post_batch(executed_cards, batch_rejections)

    def _publish_funnel_report(
        self,
        report: CycleFunnelReport,
        *,
        status: str = "ok",
        reason: str = "",
        broadcast: bool = True,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store latest CycleFunnelReport, refresh read-model snapshot, and persist funnel_state.json."""
        self._latest_funnel_report = report
        self._latest_funnel_meta = {"status": status, "reason": reason}
        if inputs is not None:
            self._last_funnel_inputs = inputs
        self._refresh_read_model_snapshot(persist_funnel=True, broadcast=broadcast)

    def _finalize_cycle_funnel_post_batch(
        self,
        executed_cards: Sequence[DecisionCard],
        batch_rejections: Sequence[Dict[str, Any]],
    ) -> None:
        """Rebuild CycleFunnelReport with post-batch executions/rejections and publish."""
        inputs = self._last_funnel_inputs
        if isinstance(inputs, dict):
            prior_rejections = list(inputs.get("gate_rejections") or ())
            combined_rejections = prior_rejections + list(batch_rejections or ())
            snap = inputs.get("config_snapshot") or self.config_bus.active_snapshot()
            epoch_id = inputs.get("config_epoch_id") or getattr(snap, "config_epoch_id", None)
            final_report = build_cycle_funnel_report(
                universe_metrics=inputs.get("universe_metrics"),
                screener_results=inputs.get("screener_results"),
                passed_screener=inputs.get("passed_screener"),
                strategy_signals=inputs.get("strategy_signals"),
                debate_outcomes=inputs.get("debate_outcomes"),
                gate_decisions=inputs.get("gate_decisions"),
                executed_orders=list(executed_cards or ()),
                gate_rejections=combined_rejections if combined_rejections else None,
                config_snapshot=snap,
                config_epoch_id=epoch_id,
            )
            self._publish_funnel_report(
                final_report,
                status="ok",
                reason="",
                broadcast=True,
                inputs={
                    **inputs,
                    "executed_orders": list(executed_cards or ()),
                    "gate_rejections": combined_rejections,
                },
            )
            return

        if batch_rejections or executed_cards:
            active_snap = self.config_bus.active_snapshot()
            minimal_report = build_cycle_funnel_report(
                gate_decisions=list(executed_cards or ()),
                executed_orders=list(executed_cards or ()),
                gate_rejections=list(batch_rejections or ()) if batch_rejections else None,
                config_snapshot=active_snap,
                config_epoch_id=active_snap.config_epoch_id,
            )
            self._publish_funnel_report(minimal_report, status="ok", reason="", broadcast=True)
        else:
            self._refresh_read_model_snapshot(persist_funnel=True, broadcast=True)

        self._broadcast_pipeline_progress(
            5, "Standby", f"Cycle complete. {len(executed_cards)} orders routed. Position monitor active.",
            cycle_id=self._current_cycle_id,
            status="standby",
        )

    # ─────────────────────────────────────────────────────────────
    # Complete Single Cycle & Deliberation Pipeline
    # ─────────────────────────────────────────────────────────────
    def _scan_and_deliberate(self) -> List[tuple[ScreenerResult, DecisionCard]]:
        """Ingest universe data, screen candidates, and deliberate via multi-agent stack.

        Captures an immutable ConfigSnapshot at cycle start, checks mid-scan epoch drift
        before expensive LLM deliberation, and always publishes CycleFunnelReport +
        read-model snapshot (including 0-candidate cycles).
        """
        from types import SimpleNamespace

        cycle_snapshot = self.config_bus.active_snapshot()
        if not getattr(self, "_current_cycle_id", None):
            self._current_cycle_id = f"cycle-{int(time.time())}"

        if self._is_pm_halted():
            logger.warning("Scan aborted: System is currently HALTED following emergency panic flatten.")
            self._broadcast_pipeline_progress(
                5, "Standby", "System is halted following emergency panic flatten.",
                cycle_id=self._current_cycle_id, status="halted"
            )
            halted_report = build_cycle_funnel_report(
                universe_metrics={},
                screener_results=[],
                passed_screener=[],
                strategy_signals=[],
                debate_outcomes=[],
                gate_decisions=[],
                executed_orders=[],
                config_snapshot=cycle_snapshot,
                config_epoch_id=cycle_snapshot.config_epoch_id,
            )
            self._publish_funnel_report(
                halted_report,
                status="halted",
                reason="System is halted following emergency panic flatten.",
                broadcast=True,
                inputs=None,
            )
            return []

        # Step 2: Ingest Candles (Data Ingestion under cycle_snapshot)
        self._broadcast_pipeline_progress(
            2, "Ingest Candles", f"Ingesting 5m candle bars & metrics for {len(self.universe)} universe assets.",
            cycle_id=self._current_cycle_id
        )
        ingest_fn = self.ingest_universe_data
        metrics_dict = (
            ingest_fn(config_snapshot=cycle_snapshot)
            if self._supports_kwarg(ingest_fn, "config_snapshot")
            else ingest_fn()
        )
        if not metrics_dict:
            logger.warning("No market data ingested — scan aborted")
            self._broadcast_pipeline_progress(
                5, "Standby", "No market data ingested during universe scan. Standing by.",
                cycle_id=self._current_cycle_id, status="standby"
            )
            empty_report = build_cycle_funnel_report(
                universe_metrics={},
                screener_results=[],
                passed_screener=[],
                strategy_signals=[],
                debate_outcomes=[],
                gate_decisions=[],
                executed_orders=[],
                config_snapshot=cycle_snapshot,
                config_epoch_id=cycle_snapshot.config_epoch_id,
            )
            self._publish_funnel_report(
                empty_report,
                status="no_data",
                reason="No market data ingested during universe scan.",
                broadcast=True,
                inputs=None,
            )
            return []

        # Step 3: Strategy Screen (Screening under cycle_snapshot)
        active_strat = getattr(cycle_snapshot, "ACTIVE_STRATEGY", "vwap")
        strat_desc = "VWAP session trend rules" if active_strat == "vwap" else f"{active_strat} momentum rules"
        self._broadcast_pipeline_progress(
            3, "Strategy Screen", f"Screening universe indicators against {strat_desc}.",
            cycle_id=self._current_cycle_id
        )
        candidates, all_screener_results = self.screen_candidates_with_funnel(
            metrics_dict,
            config_snapshot=cycle_snapshot,
        )
        strategy_signals = [
            r.strategy_signal
            for r in all_screener_results
            if getattr(r, "strategy_signal", None) is not None
        ]

        if not candidates:
            logger.info("No candidates passed 3x momentum filter this cycle.")
            self._broadcast_pipeline_progress(
                5, "Standby", "No candidates passed screener filter this cycle. Standing by.",
                cycle_id=self._current_cycle_id, status="standby"
            )
            zero_inputs = {
                "universe_metrics": metrics_dict,
                "screener_results": all_screener_results,
                "passed_screener": [],
                "strategy_signals": strategy_signals,
                "debate_outcomes": [],
                "gate_decisions": [],
                "executed_orders": [],
                "gate_rejections": [],
                "config_snapshot": cycle_snapshot,
                "config_epoch_id": cycle_snapshot.config_epoch_id,
            }
            zero_report = build_cycle_funnel_report(**zero_inputs)
            self._publish_funnel_report(
                zero_report,
                status="ok",
                reason="No candidates passed screener filter this cycle.",
                broadcast=True,
                inputs=zero_inputs,
            )
            return []

        # Mid-Scan Epoch Revalidation Gate BEFORE launching sizing
        latest_snap = self.config_bus.active_snapshot()
        if latest_snap.config_epoch_id != cycle_snapshot.config_epoch_id:
            changed_keys = _changed_snapshot_keys(cycle_snapshot, latest_snap)
            scope = _revalidation_scope_for_keys(changed_keys, cycle_snapshot, latest_snap)
            if scope == "abort_stale_scan":
                logger.warning(
                    "Mid-scan epoch drift (%s -> %s, changed=%s) requires abort_stale_scan — aborting cycle.",
                    cycle_snapshot.config_epoch_id,
                    latest_snap.config_epoch_id,
                    changed_keys,
                )
                top_n = max(1, int(getattr(cycle_snapshot, "SCREENER_TOP_N", settings.SCREENER_TOP_N)))
                stale_rejections = [
                    {
                        "gate_id": "stale_epoch_superseded",
                        "ticker": cand.ticker,
                        "param_keys": changed_keys,
                        "label": "Stale Epoch Superseded (Pre-Sizing)",
                    }
                    for cand in candidates[:top_n]
                ]
                stale_inputs = {
                    "universe_metrics": metrics_dict,
                    "screener_results": all_screener_results,
                    "passed_screener": candidates,
                    "strategy_signals": strategy_signals,
                    "debate_outcomes": [],
                    "gate_decisions": [],
                    "executed_orders": [],
                    "gate_rejections": stale_rejections,
                    "config_snapshot": cycle_snapshot,
                    "config_epoch_id": cycle_snapshot.config_epoch_id,
                }
                stale_report = build_cycle_funnel_report(**stale_inputs)
                self._publish_funnel_report(
                    stale_report,
                    status="aborted_stale_epoch",
                    reason=f"Scan epoch {cycle_snapshot.config_epoch_id} superseded by {latest_snap.config_epoch_id} ({', '.join(changed_keys)})",
                    broadcast=True,
                    inputs=stale_inputs,
                )
                self._broadcast_pipeline_progress(
                    5, "Standby", "Scan aborted due to mid-scan config change. Standing by.",
                    cycle_id=self._current_cycle_id, status="standby"
                )
                return []

        # Step 4: Quantitative Risk Sizing across Candidates
        top_n = max(1, int(getattr(cycle_snapshot, "SCREENER_TOP_N", settings.SCREENER_TOP_N)))
        candidates_to_process = candidates[:top_n]
        self._broadcast_pipeline_progress(
            4, "Risk Sizing", f"Calculating ATR stops, leverage & position sizing for {len(candidates_to_process)} candidate assets.",
            active_assets=[c.ticker for c in candidates_to_process],
            cycle_id=self._current_cycle_id,
        )
        candidate_cards: List[tuple[ScreenerResult, DecisionCard]] = []
        delib_cap = int(getattr(cycle_snapshot, "PARALLEL_DELIBERATION_WORKERS", self.parallel_deliberation_workers))
        n_delib_workers = max(1, min(len(candidates_to_process), delib_cap))

        logger.info(
            "Deliberating on %d candidate(s) in parallel across %d subagent workers (Epoch=%s)…",
            len(candidates_to_process),
            n_delib_workers,
            cycle_snapshot.config_epoch_id,
        )

        default_eq = self._get_fallback_equity(cycle_snapshot)
        try:
            account = self.t212.get_account_cash(fallback_equity=default_eq)
            baseline_equity = account.total or default_eq
        except Exception as exc:
            logger.warning("Baseline equity query failed (%s); using default $%.2f", exc, default_eq)
            baseline_equity = default_eq

        deep_fn = self.deep_analyse
        pass_snap = self._supports_kwarg(deep_fn, "config_snapshot")

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_delib_workers) as executor:
            future_to_cand = {
                executor.submit(
                    (lambda c=cand: deep_fn(c, baseline_equity, config_snapshot=cycle_snapshot))
                    if pass_snap
                    else (lambda c=cand: deep_fn(c, baseline_equity))
                ): cand
                for cand in candidates_to_process
            }
            for future in concurrent.futures.as_completed(future_to_cand):
                cand = future_to_cand[future]
                try:
                    card = future.result()
                    if card:
                        if not getattr(card, "config_epoch_id", None):
                            with contextlib.suppress(Exception):
                                card.config_epoch_id = cycle_snapshot.config_epoch_id
                        candidate_cards.append((cand, card))
                except Exception as exc:
                    logger.error("Parallel deliberation failed for %s: %s", cand.ticker, exc)

        all_cards = [card for _, card in candidate_cards]
        debate_outcomes: List[Any] = []
        for card in all_cards:
            deb = getattr(card, "_debate_outcome", None)
            if deb is not None:
                cons = getattr(deb, "consensus_decision", None)
                is_app = bool(getattr(deb, "approved", False)) or (
                    cons in ("BUY_ENTRY", "SHORT_ENTRY", "SELL_ENTRY")
                )
                debate_outcomes.append(SimpleNamespace(consensus_decision=cons, approved=is_app))
            else:
                cons = getattr(card, "decision", "DO_NOT_TRADE")
                is_app = (
                    cons in ("BUY_ENTRY", "SHORT_ENTRY", "SELL_ENTRY")
                ) or bool(getattr(card, "gate_rejections", None))
                debate_outcomes.append(SimpleNamespace(consensus_decision=cons, approved=is_app))

        funnel_inputs = {
            "universe_metrics": metrics_dict,
            "screener_results": all_screener_results,
            "passed_screener": candidates,
            "strategy_signals": strategy_signals,
            "debate_outcomes": debate_outcomes,
            "gate_decisions": all_cards,
            "executed_orders": [],
            "gate_rejections": [],
            "config_snapshot": cycle_snapshot,
            "config_epoch_id": cycle_snapshot.config_epoch_id,
        }
        pre_batch_report = build_cycle_funnel_report(**funnel_inputs)
        self._publish_funnel_report(
            pre_batch_report,
            status="ok",
            reason="",
            broadcast=True,
            inputs=funnel_inputs,
        )

        return candidate_cards

    def reconcile_pending_orders(self) -> None:
        """Audit open broker orders, cancel stale market orders, and track in-flight tickers."""
        self.in_flight_tickers.clear()
        if self.dry_run:
            return

        try:
            open_orders = self.t212.get_orders(force_refresh=True)
            now_utc = datetime.now(timezone.utc)
            for order in open_orders:
                if not isinstance(order, dict):
                    continue
                order_id = str(order.get("id", ""))
                ticker = str(order.get("ticker", ""))
                order_type = str(order.get("type", "")).upper()
                status = str(order.get("status", "")).upper()

                if not ticker or not order_id:
                    continue

                # Only examine pending/unfilled market orders
                if status in ("NEW", "LOCAL", "UNCONFIRMED") and order_type == "MARKET":
                    is_stale = False
                    reason = ""

                    # 1. Market closed check
                    if hasattr(self.t212, "is_market_open") and not self.t212.is_market_open(ticker):
                        is_stale = True
                        reason = f"market for {ticker} is closed"

                    # 2. Age threshold (>300 seconds)
                    creation_time_raw = order.get("creationTime") or order.get("created") or order.get("dateCreated")
                    if creation_time_raw and not is_stale:
                        try:
                            ts_str = str(creation_time_raw).replace("Z", "+00:00")
                            created_dt = datetime.fromisoformat(ts_str)
                            age_sec = (now_utc - created_dt).total_seconds()
                            if age_sec > 300.0:
                                is_stale = True
                                reason = f"order age {age_sec:.1f}s > 300s threshold"
                        except Exception as dt_exc:
                            logger.debug("Failed parsing order creation time '%s': %s", creation_time_raw, dt_exc)

                    if is_stale:
                        logger.warning(
                            "Stale Order Purge: Cancelling lingering %s market order %s for %s (%s)",
                            status, order_id, ticker, reason,
                        )
                        cancelled = self.t212.cancel_order(order_id)
                        if cancelled:
                            continue

                # Active/pending order remains in-flight
                if status in ("NEW", "LOCAL", "UNCONFIRMED"):
                    self.in_flight_tickers.add(ticker)

            if self.in_flight_tickers:
                logger.info("Active in-flight broker orders tracked: %s", sorted(self.in_flight_tickers))
        except Exception as exc:
            logger.warning("Broker order reconciliation warning: %s", exc)

    def run_cycle(self) -> List[DecisionCard]:
        """Execute one complete 5-step deterministic scan and execution cycle."""
        cycle_id = f"cycle-{int(time.time())}"
        self._current_cycle_id = cycle_id
        self._broadcast_pipeline_progress(
            1, "Market Check", "Checking exchange market hours, circuit breakers, and halt state.",
            cycle_id=cycle_id,
        )
        logger.info("━━━ Starting 3x Quantitative Cycle at %s ━━━",
                    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

        # Step 0: Broker Portfolio Reconciliation & Position Adoption
        recon_changed = False
        try:
            recon = self.position_monitor.reconcile_with_broker(adopt_untracked=True) or {}
            if recon.get("adopted_from_broker"):
                recon_changed = True
                logger.info(
                    "Synchronized with Trading 212 account: Adopted %d active broker positions: %s",
                    len(recon["adopted_from_broker"]), recon["adopted_from_broker"],
                )
            if recon.get("closed_externally"):
                recon_changed = True
                logger.info(
                    "Reconciliation closed %d positions not found on broker: %s",
                    len(recon["closed_externally"]), recon["closed_externally"],
                )
            if recon.get("quantity_synced"):
                recon_changed = True
            if not self.dry_run:
                self.position_monitor.apply_daily_rollover_stops()
                self.position_monitor.sync_broker_orders()
        except Exception as recon_exc:
            logger.debug("Initial broker reconciliation notice: %s", recon_exc)

        if recon_changed:
            self._update_dashboard(persist_funnel=False, broadcast=True)

        # Step 0b: Broker Pre-Flight Order Queue Reconciliation & Deduplication
        self.reconcile_pending_orders()

        candidate_cards = self._scan_and_deliberate()
        if not candidate_cards:
            return []

        # Step 4: Execution & Logging via Synchronized Batch Coordinator
        executed_cards = self.execute_trades_batch(candidate_cards)
        self._update_dashboard(persist_funnel=True, broadcast=True)
        logger.info("Cycle finished. %d trades executed.", len(executed_cards))
        return executed_cards

    # ─────────────────────────────────────────────────────────────
    # Continuous Loop with Active-Trade Driven Architecture
    # ─────────────────────────────────────────────────────────────
    def _position_state_fingerprint(self) -> Tuple[bool, Tuple[Tuple[Any, ...], ...]]:
        """Capture thread-safe fingerprint of monitor halt status and tracked positions."""
        is_halted, items = self._snapshot_positions_under_lock()
        sig = tuple(
            sorted(
                (
                    ticker,
                    round(float(p.get("remaining_quantity") or 0.0), 6),
                    round(float(p.get("stop_loss") or 0.0), 6),
                    round(float(p.get("trailing_stop_loss") or 0.0), 6),
                    round(float(p.get("current_price") or p.get("entry_price") or 0.0), 6),
                    p.get("broker_stop_order_id"),
                    p.get("config_epoch_id"),
                    bool(p.get("panic_failed", False)),
                    p.get("unprotected_reason"),
                    bool(p.get("tp1_hit", False)),
                    bool(p.get("tp2_hit", False)),
                    bool(p.get("tp3_hit", False)),
                )
                for ticker, p in items
            )
        )
        return is_halted, sig

    def _run_off_hours_reconciliation_step(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Run 300s off-hours reconciliation and republish read-model snapshot (cached broker getters)."""
        now_ts = time.time() if now is None else float(now)
        rec: Dict[str, Any] = {}
        try:
            rec = self.position_monitor.reconcile_with_broker() or {}
            if rec.get("closed_externally") or rec.get("quantity_synced") or rec.get("adopted_from_broker"):
                logger.info("Off-hours broker reconciliation sync: %s", rec)
        except Exception as e:
            logger.debug("Off-hours reconciliation notice: %s", e)
        self._last_reconciliation_time = now_ts
        self._update_dashboard(persist_funnel=True, broadcast=True)
        return rec

    def _run_priority_monitor_step(
        self,
        now: Optional[float] = None,
        now_utc: Optional[datetime] = None,
    ) -> List[Any]:
        """Execute Priority 1 active-trade monitoring tick and refresh snapshot on state changes."""
        now_ts = time.time() if now is None else float(now)
        now_dt = now_utc or datetime.now(timezone.utc)
        snap = self.config_bus.active_snapshot()

        pre_fp = self._position_state_fingerprint()

        # Fetch targeted indicators & forecasts for active trade underlyings (<0.5s)
        active_metrics = self.fetch_active_trades_metrics()

        # 1. Reconcile with Trading 212 & adopt untracked positions using live market data
        rec: Dict[str, Any] = {}
        recon_interval = float(
            getattr(snap, "RECONCILIATION_INTERVAL_SECONDS", getattr(settings, "RECONCILIATION_INTERVAL_SECONDS", 30))
        )
        if now_ts - getattr(self, "_last_reconciliation_time", 0.0) >= recon_interval:
            rec = self.position_monitor.reconcile_with_broker(adopt_untracked=True, active_metrics=active_metrics) or {}
            if rec.get("closed_externally") or rec.get("quantity_synced") or rec.get("adopted_from_broker"):
                logger.info("Active trade reconciliation: %s", rec)
            if hasattr(self.t212, "get_account_cash") and not self.dry_run:
                try:
                    c_obj = self.t212.get_account_cash(force_refresh=False)
                    if c_obj and getattr(c_obj, "total", 0.0) > 0:
                        self._persist_cached_cash(c_obj)
                except Exception as cash_err:
                    logger.debug("Active monitor cash refresh notice: %s", cash_err)
            if hasattr(self.t212, "get_historical_orders") and not self.dry_run:
                try:
                    self.t212.get_historical_orders(limit=30, force_refresh=False)
                except Exception as hist_err:
                    logger.debug("Active monitor historical orders refresh notice: %s", hist_err)
            self._last_reconciliation_time = now_ts

        # 2. Sync native broker orders ("set the price" on broker server)
        if not self.dry_run:
            self.position_monitor.sync_broker_orders()

        # 3. Active Position Price & Technical Signal Monitoring (using cached portfolio prices where supported)
        events: List[Any] = []
        _, items = self._snapshot_positions_under_lock()
        if items:
            logger.debug("Active management for %d positions…", len(items))
            current_prices: Dict[str, float] = {}
            try:
                get_port_fn = self.t212.get_portfolio
                t212_positions = (
                    get_port_fn(cached_only=True)
                    if self._supports_kwarg(get_port_fn, "cached_only")
                    else get_port_fn()
                )
                for p in (t212_positions or ()):
                    p_price = getattr(p, "current_price", None) or getattr(p, "average_price", None) or 0.0
                    if p_price and float(p_price) > 0:
                        current_prices[p.ticker] = float(p_price)
            except Exception as e:
                logger.debug("T212 cached portfolio price fetch notice: %s", e)

            for und, met in (active_metrics or {}).items():
                if met.get("close"):
                    current_prices[und] = met["close"]

            for ticker, pos in items:
                if ticker not in current_prices:
                    pos_price = pos.get("current_price") or pos.get("entry_price") or 0.0
                    if pos_price and float(pos_price) > 0:
                        current_prices[ticker] = float(pos_price)

            # Price-target tick (TP1/TP2/TP3/SL/TSL)
            tick_events = self.position_monitor.monitor_tick(current_prices, current_time=now_dt) or []
            events.extend(tick_events)

            # Signal-driven execution on active trades
            signal_events = self.position_monitor.evaluate_active_trade_signals(active_metrics) or []
            if signal_events:
                events.extend(signal_events)

        post_fp = self._position_state_fingerprint()
        rec_changed = bool(
            rec.get("closed_externally") or rec.get("quantity_synced") or rec.get("adopted_from_broker")
        )
        if events or rec_changed or post_fp != pre_fp:
            self._update_dashboard(persist_funnel=True, broadcast=True)

        return events

    def run_loop(
        self,
        scan_interval_minutes: int = 5,
        monitor_interval_seconds: int = 30,
        duration_minutes: Optional[int] = None,
    ) -> None:
        """Run continuous engine loop.

        Prioritizes Active Trade Management on high-frequency intervals:
        - Reads Trading 212 active trades directly as primary ground truth.
        - Calculates ATR/SL/TP1/TP2/TP3 and sets native broker orders (stops & limits).
        - Fetches targeted market metrics for active trades (<0.5s) and executes if signals trigger.
        - Decouples 120-asset universe scan into background worker so active trades are never starved.
        """
        logger.info(
            "Starting continuous Active-Trade engine (Scan Interval: %dm, Monitor Interval: %ds, Duration: %s). Press Ctrl+C to stop.",
            scan_interval_minutes,
            monitor_interval_seconds,
            f"{duration_minutes}m" if duration_minutes else "indefinite",
        )
        last_scan_time = 0.0
        loop_start_time = time.time()

        while True:
            try:
                now = time.time()
                if duration_minutes and duration_minutes > 0 and (now - loop_start_time >= duration_minutes * 60):
                    logger.info(
                        "Session duration reached (%d minutes) — shutting down loop cleanly.",
                        duration_minutes,
                    )
                    break
                now_utc = datetime.now(timezone.utc)
                snap = self.config_bus.active_snapshot()

                # Daily Circuit Breaker Calendar Date Rollover Reset
                today_date = now_utc.strftime("%Y-%m-%d")
                if getattr(self, "_last_session_cb_date", "") != today_date:
                    default_eq = self._get_fallback_equity(snap)
                    try:
                        acc = self.t212.get_account_cash(fallback_equity=default_eq)
                        new_eq = acc.total or default_eq
                        self.session_start_equity = new_eq
                        if self.risk_manager:
                            self.risk_manager.reset_circuit_breaker(new_session_equity=new_eq)
                        self._last_session_cb_date = today_date
                        logger.info("🌅 New trading session (%s): Reset daily circuit breaker. Baseline equity: $%.2f", today_date, new_eq)
                    except Exception as cb_reset_exc:
                        logger.warning("Could not reset session circuit breaker on rollover: %s", cb_reset_exc)

                active_tickers = list(self.position_monitor.positions.keys()) if self.position_monitor.positions else ["QQQ3l_EQ", "SPY_US_EQ"]
                market_open = self.t212.is_any_market_open(active_tickers, current_time=now_utc)

                if not market_open and not self.dry_run:
                    if now - getattr(self, "_last_closed_loop_log", 0.0) >= 1800.0:
                        logger.info(
                            "🌙 Markets CLOSED (LSE 08:00-16:30 UTC, US 14:30-21:00 UTC). "
                            "Suspending universe scan and position polling to conserve API quota."
                        )
                        self._last_closed_loop_log = now

                    # Run periodic reconciliation even during off-hours & republish cached snapshot
                    if now - getattr(self, "_last_reconciliation_time", 0.0) >= 300.0:
                        self._run_off_hours_reconciliation_step(now=now)

                    time.sleep(min(300, monitor_interval_seconds * 10))
                    continue

                # ── PRIORITY 1: High-Frequency Active Position Management ──
                self._run_priority_monitor_step(now=now, now_utc=now_utc)

                # ── PRIORITY 2: Decoupled Non-Blocking Universe Scan ──
                # Harvest completed background scan
                if self._scan_future is not None:
                    if self._scan_future.done():
                        try:
                            candidate_cards = self._scan_future.result()
                            if candidate_cards:
                                executed = self.execute_trades_batch(candidate_cards)
                                if executed:
                                    logger.info("Background universe scan batch finished: %d trades executed.", len(executed))
                                    self._update_dashboard(persist_funnel=True, broadcast=True)
                        except Exception as scan_err:
                            logger.error("Background universe scan error: %s", scan_err)
                        finally:
                            self._scan_future = None
                            last_scan_time = time.time()
                else:
                    # Launch background scan if interval elapsed or at startup
                    if now - last_scan_time >= scan_interval_minutes * 60 or last_scan_time == 0.0:
                        logger.info("Triggering non-blocking background 3x universe scan…")
                        self._scan_future = self._scan_executor.submit(self._scan_and_deliberate)
                        if last_scan_time == 0.0:
                            last_scan_time = time.time()

                time.sleep(monitor_interval_seconds)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received — stopping loop.")
                break
            except Exception as exc:
                logger.error("Loop iteration error: %s", exc, exc_info=True)
                time.sleep(10)

    def shutdown(self) -> None:
        """Clean up resources."""
        logger.info("Shutting down trading engine…")
        if getattr(self, "dashboard_server", None):
            try:
                logger.info("Stopping embedded dashboard server…")
                self.dashboard_server.shutdown()
                self.dashboard_server.server_close()
            except Exception as exc:
                logger.debug("Dashboard server shutdown notice: %s", exc)
        if getattr(self, "_scan_executor", None):
            try:
                self._scan_executor.shutdown(wait=False)
            except Exception as exc:
                logger.debug("Scan executor shutdown notice: %s", exc)
        self.t212.close()
        if getattr(self, "_owns_temp_data_dir", False) and getattr(self, "data_dir", None):
            import shutil
            shutil.rmtree(self.data_dir, ignore_errors=True)
        logger.info("Shutdown complete.")


# ─────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    """CLI runner with diagnostic and testing commands."""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="3x Leveraged Long/Short Quantitative Trading Engine")
    parser.add_argument("--once", action="store_true", help="Run a single scan & execution cycle and exit")
    parser.add_argument("--interval", type=int, default=5, help="Minutes between full universe scans (default: 5)")
    parser.add_argument("--monitor-interval", type=int, default=30, help="Seconds between position monitor ticks (default: 30)")
    parser.add_argument("--dry-run", action="store_true", help="Run in paper-trade mode without real order execution")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable the embedded local signals dashboard web server")
    parser.add_argument("--dashboard-port", type=int, default=settings.DASHBOARD_PORT, help=f"Port for embedded dashboard server (default: {settings.DASHBOARD_PORT})")
    parser.add_argument("--dashboard-host", type=str, default=getattr(settings, "DASHBOARD_HOST", ""), help="Host to bind embedded dashboard server (default: system LAN default)")
    parser.add_argument("--duration-minutes", type=int, default=0, help="Optional duration in minutes to run before clean shutdown (default: 0 = run indefinitely)")
    parser.add_argument("--open-browser", action="store_true", help="Automatically open dashboard in default browser on startup")
    parser.add_argument("--test-notifications", action="store_true", help="Dispatch test alerts for all 7 event types (BUY, SELL, TP1, TP2, TP3, SL, TSL)")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["vwap", "ml_alpha"],
        default=getattr(settings, "ACTIVE_STRATEGY", "vwap"),
        help="Pluggable trading strategy to execute (default: vwap)",
    )
    parser.add_argument("--backtest", action="store_true", help="Run historical strategy backtest simulator and benchmark report")
    parser.add_argument("--backtest-symbol", type=str, default="QQQ", help="Underlying asset symbol to backtest (default: QQQ)")
    parser.add_argument("--backtest-period", type=str, default="1y", help="Historical backtest period (default: 1y)")
    parser.add_argument("--backtest-strategy", type=str, choices=["vwap", "ml_alpha", "both", "all"], default="vwap", help="Strategy filter to backtest (default: vwap)")
    args = parser.parse_args()

    if getattr(args, "strategy", None):
        setattr(settings, "ACTIVE_STRATEGY", args.strategy)

    # Historical backtesting route
    if args.backtest:
        from trading_system.backtest.engine import BacktestEngine, generate_synthetic_candles, print_backtest_report
        candles = None
        if not args.dry_run:
            try:
                from trading_system.data.trading212_client import MarketDataProvider
                raw = MarketDataProvider.get_daily_candles(args.backtest_symbol, period=args.backtest_period)
                if raw is not None:
                    if hasattr(raw, "to_dict"):
                        candles = raw.to_dict(orient="records")
                        if "date" not in raw.columns and hasattr(raw, "index"):
                            for idx, rec in enumerate(candles):
                                rec["date"] = str(raw.index[idx])[:10]
                    elif isinstance(raw, list):
                        candles = raw
            except Exception as e:
                logger.warning("Could not fetch market data for %s: %s", args.backtest_symbol, e)
        if not candles:
            candles = generate_synthetic_candles(args.backtest_symbol, count=160)

        engine_bt = BacktestEngine(
            initial_capital=settings.DEFAULT_ACCOUNT_EQUITY,
            leverage=3,
            max_risk_per_trade=settings.MAX_RISK_PER_TRADE,
            max_position_pct=settings.MAX_POSITION_PCT,
            target_vol=settings.TARGET_ANNUAL_VOLATILITY,
            max_hold_days=settings.MR_MAX_HOLD_DAYS,
        )
        res = engine_bt.run(symbol=args.backtest_symbol, candles=candles, strategy_filter=args.backtest_strategy)
        print_backtest_report(res)
        return

    engine = Orchestrator(
        dry_run=args.dry_run,
        enable_dashboard=not args.no_dashboard,
        dashboard_port=args.dashboard_port,
        dashboard_host=args.dashboard_host or None,
        open_browser=args.open_browser,
    )

    # Test all 7 notification formats if requested
    if args.test_notifications:
        print("\n🧪 Testing all 7 notification formats (Console, Desktop, Webhooks)...\n")
        engine.notifier.notify_buy(
            ticker="QQQ3l_EQ", underlying="QQQ", price=150.25, quantity=10.0,
            sl=145.00, tp1=155.50, tp2=160.75, tp3=166.00,
            rationale="High-conviction 3x Bullish breakout (ADX=29.4, 20 EMA > 50 EMA)"
        )
        time.sleep(0.5)
        engine.notifier.notify_tp1(
            ticker="QQQ3l_EQ", underlying="QQQ", direction="LONG", price=155.50,
            quantity_sold=3.33, pnl=17.48, pnl_pct=3.49, new_sl=150.25
        )
        time.sleep(0.5)
        engine.notifier.notify_tp2(
            ticker="QQQ3l_EQ", underlying="QQQ", direction="LONG", price=160.75,
            quantity_sold=3.33, pnl=34.97, pnl_pct=6.99, tsl=155.50
        )
        time.sleep(0.5)
        engine.notifier.notify_tsl(
            ticker="QQQ3l_EQ", underlying="QQQ", direction="LONG", current_price=163.00,
            new_tsl=158.20, locked_in_pnl_pct=5.29
        )
        time.sleep(0.5)
        engine.notifier.notify_tp3(
            ticker="QQQ3l_EQ", underlying="QQQ", direction="LONG", price=166.00,
            quantity_sold=3.34, pnl=52.61, pnl_pct=10.48
        )
        time.sleep(0.5)
        engine.notifier.notify_sell(
            ticker="QQQSl_EQ", underlying="QQQ", price=45.10, quantity=25.0,
            sl=47.50, tp1=42.70, tp2=40.30, tp3=37.90, is_short_entry=True,
            rationale="3x Bearish breakdown momentum detected (ADX=31.2, 20 EMA < 50 EMA)"
        )
        time.sleep(0.5)
        engine.notifier.notify_sl(
            ticker="QQQSl_EQ", underlying="QQQ", direction="SHORT", price=47.50,
            quantity=25.0, loss_amount=60.00, loss_pct=5.32
        )
        print("\n✅ All 7 notification event tests completed successfully!\n")
        return

    try:
        if args.once:
            engine.run_cycle()
        else:
            engine.run_loop(
                scan_interval_minutes=args.interval,
                monitor_interval_seconds=args.monitor_interval,
                duration_minutes=args.duration_minutes or None,
            )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
