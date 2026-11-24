"""Thread-safe, epoch-versioned configuration bus and immutable ConfigSnapshot."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, MISSING
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

from trading_system.config import (
    Settings,
    atomic_write_text,
    collect_settings_invariant_errors,
    parse_env_file,
)
from trading_system.config_schema import (
    BUILTIN_PROFILES,
    MAX_DYNAMIC_PARAMETERS,
    SCHEMA_BY_KEY,
    SIMULATABLE_GROUPS,
    SNAPSHOT_METADATA_KEYS,
    ConfigParamSchema,
    DynamicParameterSpec,
    export_config_schema,
    is_env_only_key,
    is_finite_number,
    validate_value_against_spec,
)

DEFAULT_DATA_DIR = Path("data")
DEFAULT_MAX_EPOCH_HISTORY = 500

_TRUTHY_ENV = frozenset({"1", "true", "yes", "on"})
_FALSY_ENV = frozenset({"0", "false", "no", "off"})


class FrozenMapping(Mapping[str, Any]):
    """Immutable mapping safe for deep-copying, hashing, and dataclasses.asdict()."""

    __slots__ = ("_data",)

    def __init__(self, data: Optional[Mapping[str, Any]] = None) -> None:
        frozen: Dict[str, Any] = {}
        if data:
            for key, value in data.items():
                frozen[str(key)] = _freeze_value(value)
        object.__setattr__(self, "_data", frozen)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("FrozenMapping is immutable")

    def __delattr__(self, name: str) -> None:
        raise TypeError("FrozenMapping is immutable")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FrozenMapping):
            return self._data == other._data
        if isinstance(other, Mapping):
            return self._data == dict(other)
        return NotImplemented

    def __repr__(self) -> str:
        return f"FrozenMapping({self._data!r})"

    def __deepcopy__(self, memo: Dict[int, Any]) -> "FrozenMapping":
        return FrozenMapping({k: copy.deepcopy(v, memo) for k, v in self._data.items()})

    def __reduce__(self) -> Tuple[Any, Tuple[Dict[str, Any]]]:
        return (FrozenMapping, (dict(self._data),))

    def to_dict(self) -> Dict[str, Any]:
        """Return a mutable plain-dict copy for serialization."""
        return {k: _unfreeze_value(v) for k, v in self._data.items()}


def _freeze_value(value: Any) -> Any:
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _unfreeze_value(value: Any) -> Any:
    if isinstance(value, FrozenMapping):
        return value.to_dict()
    if isinstance(value, (tuple, list)):
        return [_unfreeze_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_unfreeze_value(item) for item in sorted(value, key=str)]
    return value


class ConfigConflictError(Exception):
    """Raised when optimistic CAS commit detects a stale or mismatched epoch."""

    def __init__(
        self,
        expected_epoch_id: Optional[str],
        actual_epoch_id: str,
        latest_snapshot: Optional["ConfigSnapshot"] = None,
    ) -> None:
        self.expected_epoch_id = expected_epoch_id
        self.actual_epoch_id = actual_epoch_id
        self.latest_snapshot = latest_snapshot
        super().__init__(
            f"Config epoch conflict: expected {expected_epoch_id!r}, active is {actual_epoch_id!r}"
        )


class ConfigValidationError(ValueError):
    """Raised when a patch or dynamic parameter fails schema or invariant checks."""

    def __init__(self, errors: Iterable[Mapping[str, str]]) -> None:
        self.errors: Tuple[Dict[str, str], ...] = tuple(dict(err) for err in errors)
        summary = "; ".join(
            f"{err.get('key', '?')}: {err.get('message', 'invalid')}"
            for err in self.errors
        )
        super().__init__(summary or "Configuration validation failed")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_json_constant(token: str) -> Any:
    raise ValueError(f"Non-finite JSON numeric constant {token!r} is not allowed")


def _json_loads_strict(raw: str) -> Any:
    return json.loads(raw, parse_constant=_reject_json_constant)


def _json_dumps_strict(obj: Any, *, indent: Optional[int] = 2) -> str:
    return json.dumps(obj, indent=indent, sort_keys=True, allow_nan=False)


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically using thread-unique temp file, flush, fsync, and os.replace."""
    atomic_write_text(path, content)


def _quarantine_file(path: Path, reason: str) -> Optional[str]:
    """Safely isolate a corrupt/invalid JSON store file without crashing boot."""
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tid = threading.get_ident()
    pid = os.getpid()
    for attempt in range(100):
        suffix = f".corrupt.{stamp}.{pid}.{tid}" if attempt == 0 else f".corrupt.{stamp}.{pid}.{tid}.{attempt}"
        candidate = path.with_name(f"{path.name}{suffix}")
        if not candidate.exists():
            os.replace(str(path), str(candidate))
            return f"Quarantined '{path.name}' to '{candidate.name}': {reason}"
    fallback = path.with_name(f"{path.name}.corrupt.{stamp}.{pid}.{tid}.latest")
    os.replace(str(path), str(fallback))
    return f"Quarantined '{path.name}' to '{fallback.name}': {reason}"


def _is_testing_process() -> bool:
    return "unittest" in sys.modules or any("test" in arg.lower() for arg in sys.argv)


def _pure_settings_defaults() -> Dict[str, Any]:
    """Extract pure dataclass defaults from Settings without reading env vars."""
    defaults: Dict[str, Any] = {}
    for f in fields(Settings):
        if f.name not in SCHEMA_BY_KEY:
            continue
        if f.default is not MISSING:
            val = f.default
        elif f.default_factory is not MISSING:
            val = f.default_factory()
        else:
            val = None
        defaults[f.name] = _to_snapshot_value(f.name, val)
    return defaults


def _snapshot_values_from_settings(settings_obj: Settings) -> Dict[str, Any]:
    return {
        key: _to_snapshot_value(key, getattr(settings_obj, key))
        for key in SCHEMA_BY_KEY
    }


def _to_snapshot_value(key: str, value: Any) -> Any:
    if key == "WATCHLIST":
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return tuple(str(item).strip().upper() for item in value if str(item).strip())
    if isinstance(value, list):
        return tuple(value)
    return value


def _to_serializable_value(key: str, value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, FrozenMapping):
        return value.to_dict()
    if isinstance(value, DynamicParameterSpec):
        return value.to_dict()
    return value


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return list(a) == list(b)
    return a == b


def _parse_env_string(spec: ConfigParamSchema, raw: str) -> Tuple[bool, Any]:
    """Parse an environment variable string matching Settings.__post_init__ conventions."""
    stripped = raw.strip()
    if spec.key == "WATCHLIST":
        if not stripped:
            return True, None
        parsed = tuple(t.strip().upper() for t in stripped.split(",") if t.strip())
        return True, (parsed or None)
    if spec.type == "bool":
        lower = stripped.lower()
        if lower in _TRUTHY_ENV:
            return True, True
        if lower in _FALSY_ENV:
            return True, False
        return False, None
    if not stripped:
        return False, None
    if spec.type == "int":
        try:
            return True, int(stripped)
        except ValueError:
            return False, None
    if spec.type == "float":
        try:
            parsed_float = float(stripped)
        except ValueError:
            return False, None
        if not is_finite_number(parsed_float):
            return False, None
        return True, parsed_float
    if spec.type == "str":
        if spec.key == "SCREENER_MIN_SETUP_SCORE":
            return True, stripped.lower()
        if spec.key == "LOG_LEVEL":
            return True, stripped.upper()
        return True, raw
    return False, None


@dataclass(frozen=True)
class ConfigSnapshot:
    """Immutable point-in-time configuration snapshot returned by ConfigBus."""

    # --- strategy (11) ---
    ACTIVE_STRATEGY: str = "vwap"
    MIN_ADX_TREND: float = 18.0
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
    MR_MAX_HOLD_DAYS: int = 10

    # --- screener (8) ---
    SCREENER_MIN_SETUP_SCORE: str = "favorable"
    SCREENER_MAX_CATALYST_RISK: float = 0.25
    SCREENER_TOP_N: int = 15
    SCREENER_MAX_PER_CATEGORY: int = 2
    REGIME_BREAKOUT_ADX_MIN: float = 25.0
    HIGH_VOL_MAX_REALIZED_VOL: float = 0.65
    HIGH_VOL_MAX_ATR_SPIKE_RATIO: float = 1.50
    FALLBACK_SCREENER_MIN_ADX: float = 20.0

    # --- risk (19) ---
    DEFAULT_ACCOUNT_EQUITY: float = 10000.0
    MAX_RISK_PER_TRADE: float = 0.015
    MAX_POSITION_PCT: float = 0.20
    TARGET_ANNUAL_VOLATILITY: float = 0.20
    ATR_SL_MULTIPLIER: float = 2.2
    MIN_SL_PCT: float = 0.005
    MAX_SL_PCT: float = 0.03
    ETP_MIN_SL_PCT: float = 0.025
    ETP_MAX_SL_PCT: float = 0.080
    ENFORCE_SPREAD_GATE: bool = True
    MAX_BID_ASK_SPREAD_PCT: float = 0.015
    MAX_SPREAD_TO_STOP_RATIO: float = 0.50
    TRADE_LOSS_COOLDOWN_SECONDS: float = 3600.0
    MIN_HOLD_SECONDS: float = 120.0
    ENFORCE_CORRELATION_LIMITS: bool = True
    MAX_SECTOR_POSITIONS: int = 4
    MAX_PORTFOLIO_BETA_HEAT: float = 6.0
    ENFORCE_DAILY_CIRCUIT_BREAKER: bool = True
    DAILY_MAX_DRAWDOWN_PCT: float = 0.02

    # --- exit_tsl (13) ---
    PURE_TSL_MODE: bool = True
    TSL_ATR_MULTIPLIER: float = 1.8
    TSL_HYSTERESIS_ATR_MULT: float = 0.5
    TSL_DECAY_DAY1_MULT: float = 2.20
    TSL_DECAY_STEP_MULT: float = 0.35
    TSL_DECAY_FLOOR_MULT: float = 1.15
    TSL_DECAY_RUNNER_MULT: float = 2.00
    TP1_R_MULTIPLE: float = 1.5
    TP2_R_MULTIPLE: float = 2.5
    TP3_R_MULTIPLE: float = 3.5
    TP1_TRIM_PCT: float = 0.50
    TP2_TRIM_PCT: float = 0.25
    TP3_TRIM_PCT: float = 0.25

    # --- session (7) ---
    SESSION_CLOSE_CUTOFF_UTC: str = "16:20"
    ENFORCE_INVERSE_INTRADAY_ONLY: bool = True
    FLATTEN_ALL_LEVERAGED_AT_SESSION_CLOSE: bool = False
    ENFORCE_LONG_BREAKEVEN_OVERNIGHT: bool = True
    MIN_OVERNIGHT_PROFIT_CUSHION_PCT: float = 0.030
    EARNINGS_BLACKOUT_HOURS: int = 24
    RATE_LIMIT_INTER_CALL_DELAY: float = 1.25

    # --- leverage (6) ---
    USE_LEVERAGED_3X: bool = True
    ENABLE_CRYPTO_3X: bool = True
    ENABLE_COMMODITIES_3X: bool = True
    ENABLE_5X_LEVERAGE: bool = False
    PREFER_5X_LEVERAGE: bool = False
    MAX_LEVERAGE: int = 3

    # --- providers (2) ---
    ENABLE_SPECULATIVE_ROUTING: bool = False
    SPECULATIVE_CONFIDENCE_THRESHOLD: float = 0.70

    # --- runtime (11) ---
    LLM_DEEP_MODEL: str = "gpt-4o"
    LLM_QUICK_MODEL: str = "gpt-4o-mini"
    PARALLEL_INGESTION_WORKERS: int = 24
    PARALLEL_DELIBERATION_WORKERS: int = 8
    ENABLE_CANDLE_CACHE: bool = True
    CANDLE_CACHE_TTL_SECONDS: int = 300
    RECONCILIATION_INTERVAL_SECONDS: int = 60
    UNIVERSE_SIZE: int = 100
    WATCHLIST: Optional[Tuple[str, ...]] = None
    LOG_LEVEL: str = "INFO"
    ENABLE_DESKTOP_NOTIFICATIONS: bool = True

    # --- epoch & metadata fields ---
    config_epoch_id: str = "epoch-0"
    generation: int = 0
    committed_at: str = ""
    committed_by: str = "system"
    commit_reason: str = ""
    pending_restart_keys: Tuple[str, ...] = ()
    config_quarantine_warnings: Tuple[str, ...] = ()
    dynamic_parameters: FrozenMapping = field(default_factory=FrozenMapping)
    dynamic_parameter_specs: Tuple[DynamicParameterSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.WATCHLIST is not None and not isinstance(self.WATCHLIST, tuple):
            object.__setattr__(self, "WATCHLIST", _to_snapshot_value("WATCHLIST", self.WATCHLIST))
        if not isinstance(self.dynamic_parameters, FrozenMapping):
            object.__setattr__(
                self,
                "dynamic_parameters",
                FrozenMapping(self.dynamic_parameters if isinstance(self.dynamic_parameters, Mapping) else {}),
            )
        if not isinstance(self.dynamic_parameter_specs, tuple):
            object.__setattr__(self, "dynamic_parameter_specs", tuple(self.dynamic_parameter_specs))
        if not isinstance(self.pending_restart_keys, tuple):
            object.__setattr__(self, "pending_restart_keys", tuple(self.pending_restart_keys))
        if not isinstance(self.config_quarantine_warnings, tuple):
            object.__setattr__(
                self,
                "config_quarantine_warnings",
                tuple(str(w) for w in self.config_quarantine_warnings),
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Read a Settings key, snapshot metadata field, or custom dynamic parameter."""
        if key in SCHEMA_BY_KEY or key in SNAPSHOT_METADATA_KEYS:
            return getattr(self, key, default)
        if key in self.dynamic_parameters:
            return self.dynamic_parameters[key]
        return default

    def to_dict(self, include_specs: bool = True) -> Dict[str, Any]:
        """Serialize snapshot to a plain JSON-safe dictionary."""
        payload: Dict[str, Any] = {
            key: _to_serializable_value(key, getattr(self, key))
            for key in SCHEMA_BY_KEY
        }
        payload.update(
            {
                "config_epoch_id": self.config_epoch_id,
                "generation": self.generation,
                "committed_at": self.committed_at,
                "committed_by": self.committed_by,
                "commit_reason": self.commit_reason,
                "pending_restart_keys": list(self.pending_restart_keys),
                "config_quarantine_warnings": list(self.config_quarantine_warnings),
                "dynamic_parameters": self.dynamic_parameters.to_dict(),
            }
        )
        if include_specs:
            payload["dynamic_parameter_specs"] = [
                spec.to_dict() for spec in self.dynamic_parameter_specs
            ]
        return payload

    def as_dict(self, include_specs: bool = True) -> Dict[str, Any]:
        """Alias for to_dict()."""
        return self.to_dict(include_specs=include_specs)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        base_defaults: Optional[Mapping[str, Any]] = None,
    ) -> "ConfigSnapshot":
        """Reconstruct an immutable ConfigSnapshot from a serialized dict."""
        defaults = dict(base_defaults) if base_defaults is not None else _pure_settings_defaults()
        kwargs: Dict[str, Any] = {}
        for key in SCHEMA_BY_KEY:
            raw = data.get(key, defaults.get(key))
            kwargs[key] = _to_snapshot_value(key, raw)

        specs_raw = data.get("dynamic_parameter_specs", ())
        specs: List[DynamicParameterSpec] = []
        if isinstance(specs_raw, (list, tuple)):
            for item in specs_raw:
                if isinstance(item, DynamicParameterSpec):
                    specs.append(item)
                elif isinstance(item, Mapping):
                    specs.append(DynamicParameterSpec.from_dict(item))

        dyn_raw = data.get("dynamic_parameters", {})
        dyn_dict: Dict[str, Any] = dict(dyn_raw) if isinstance(dyn_raw, Mapping) else {}
        for spec in specs:
            dyn_dict.setdefault(spec.key, spec.default)

        kwargs["config_epoch_id"] = str(data.get("config_epoch_id", "epoch-0"))
        kwargs["generation"] = int(data.get("generation", 0))
        kwargs["committed_at"] = str(data.get("committed_at", ""))
        kwargs["committed_by"] = str(data.get("committed_by", "system"))
        kwargs["commit_reason"] = str(data.get("commit_reason", ""))
        kwargs["pending_restart_keys"] = tuple(str(k) for k in data.get("pending_restart_keys", ()))
        kwargs["config_quarantine_warnings"] = tuple(
            str(w) for w in data.get("config_quarantine_warnings", ())
        )
        kwargs["dynamic_parameters"] = FrozenMapping(dyn_dict)
        kwargs["dynamic_parameter_specs"] = tuple(specs)
        return cls(**kwargs)


class ConfigBus:
    """Thread-safe configuration bus with optimistic CAS, schema validation, and atomic stores."""

    def __init__(
        self,
        *,
        data_dir: Optional[Union[str, Path]] = None,
        overrides_path: Optional[Union[str, Path]] = None,
        dynamic_params_path: Optional[Union[str, Path]] = None,
        epochs_path: Optional[Union[str, Path]] = None,
        env_file: Optional[Union[str, Path]] = None,
        base_settings: Optional[Settings] = None,
        environ: Optional[Mapping[str, str]] = None,
        max_epoch_history: int = DEFAULT_MAX_EPOCH_HISTORY,
        ignore_process_env: bool = False,
    ) -> None:
        if max_epoch_history < 1:
            raise ValueError("max_epoch_history must be >= 1")

        self._lock = threading.RLock()
        self.max_epoch_history = int(max_epoch_history)

        resolved_data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
        self.data_dir = resolved_data_dir
        self.overrides_path = (
            Path(overrides_path)
            if overrides_path is not None
            else resolved_data_dir / "config_overrides.json"
        )
        self.dynamic_params_path = (
            Path(dynamic_params_path)
            if dynamic_params_path is not None
            else resolved_data_dir / "dynamic_parameters.json"
        )
        self.epochs_path = (
            Path(epochs_path)
            if epochs_path is not None
            else resolved_data_dir / "config_epochs.jsonl"
        )

        # Only auto-load workspace .env on default production boot; isolated tests specify data_dir/env_file.
        if env_file is not None:
            resolved_env_file: Optional[Path] = Path(env_file)
        elif data_dir is None and overrides_path is None:
            default_dotenv = Path(".env")
            resolved_env_file = default_dotenv if default_dotenv.exists() else None
        else:
            resolved_env_file = None
        self.env_file = resolved_env_file

        if base_settings is not None:
            self._base_defaults = _snapshot_values_from_settings(base_settings)
        else:
            self._base_defaults = _pure_settings_defaults()

        if ignore_process_env:
            env_mapping: Mapping[str, str] = {}
        elif environ is not None:
            env_mapping = environ
        else:
            env_mapping = os.environ

        self._env_baseline = self._compute_env_baseline(resolved_env_file, env_mapping)
        self._epoch_cache: Dict[str, ConfigSnapshot] = {}
        self._quarantine_warnings: Tuple[str, ...] = ()
        self._dynamic_specs: Dict[str, DynamicParameterSpec] = {}
        self._overrides: Dict[str, Any] = {}
        self._process_boot_values: Dict[str, Any] = dict(self._env_baseline)
        self._generation: int = 0
        self._epoch_line_count: int = 0

        self._boot()

    def _compute_env_baseline(
        self,
        env_file: Optional[Path],
        env_mapping: Mapping[str, str],
    ) -> Dict[str, Any]:
        """Compute Settings defaults < .env < process environ for tunable keys."""
        resolved = dict(self._base_defaults)
        testing = _is_testing_process()

        if env_file is not None:
            dotenv_values = parse_env_file(env_file)
            self._apply_env_layer(resolved, dotenv_values, is_process_env=False, testing=testing)

        self._apply_env_layer(resolved, env_mapping, is_process_env=True, testing=testing)
        return resolved

    @staticmethod
    def _apply_env_layer(
        target: Dict[str, Any],
        source: Mapping[str, str],
        *,
        is_process_env: bool,
        testing: bool,
    ) -> None:
        applied_keys = set()
        for key, spec in SCHEMA_BY_KEY.items():
            if is_process_env and testing and key == "FLATTEN_ALL_LEVERAGED_AT_SESSION_CLOSE":
                continue
            if key not in source:
                continue
            ok, parsed = _parse_env_string(spec, source[key])
            if not ok:
                continue
            coerced, err = validate_value_against_spec(spec, parsed)
            if err is None:
                target[key] = _to_snapshot_value(key, coerced)
                applied_keys.add(key)

        # Mirror Settings.__post_init__ 5x <-> MAX_LEVERAGE parity when env leaves MAX_LEVERAGE unset.
        if "ENABLE_5X_LEVERAGE" in applied_keys and "MAX_LEVERAGE" not in applied_keys:
            if target["ENABLE_5X_LEVERAGE"] and target["MAX_LEVERAGE"] < 5:
                target["MAX_LEVERAGE"] = 5
            elif not target["ENABLE_5X_LEVERAGE"] and target["MAX_LEVERAGE"] > 3:
                target["MAX_LEVERAGE"] = 3
        if (
            "ENABLE_5X_LEVERAGE" in applied_keys
            and not target["ENABLE_5X_LEVERAGE"]
            and "PREFER_5X_LEVERAGE" not in applied_keys
        ):
            target["PREFER_5X_LEVERAGE"] = False

    def _boot(self) -> None:
        warnings: List[str] = []

        # Load dynamic_parameters.json BEFORE config_overrides.json so valid custom overrides resolve cleanly.
        specs, dyn_quarantined, dyn_warning = self._load_dynamic_parameters_on_boot()
        if dyn_warning:
            warnings.append(dyn_warning)
        self._dynamic_specs = specs

        overrides, ovr_quarantined, ovr_warning = self._load_overrides_on_boot(specs)
        if ovr_warning:
            warnings.append(ovr_warning)
        self._overrides = overrides

        max_gen, last_epoch_meta, epoch_count, epoch_warnings = self._load_epochs_on_boot()
        warnings.extend(epoch_warnings)
        self._epoch_line_count = epoch_count
        self._quarantine_warnings = tuple(warnings)

        # Effective boot values establish the process baseline for restart_required drift detection.
        effective_settings = dict(self._env_baseline)
        effective_dyn = {spec.key: spec.default for spec in specs.values()}
        for key, val in overrides.items():
            if key in SCHEMA_BY_KEY:
                effective_settings[key] = _to_snapshot_value(key, val)
            elif key in specs:
                effective_dyn[key] = val

        self._process_boot_values = dict(effective_settings)
        self._generation = max_gen

        recovery_epoch_created = False
        if (dyn_quarantined or ovr_quarantined) and max_gen > 0:
            epoch_id = f"epoch-{max_gen}-quarantine-recovery"
            committed_at = _utc_now_iso()
            committed_by = "system"
            commit_reason = "boot quarantine recovery"
            recovery_epoch_created = (
                last_epoch_meta is None
                or str(last_epoch_meta.get("config_epoch_id", "")) != epoch_id
            )
        elif max_gen > 0 and last_epoch_meta is not None:
            epoch_id = f"epoch-{max_gen}"
            committed_at = str(last_epoch_meta.get("committed_at", ""))
            committed_by = str(last_epoch_meta.get("committed_by", "system"))
            commit_reason = str(last_epoch_meta.get("commit_reason", ""))
        else:
            epoch_id = "epoch-0"
            committed_at = ""
            committed_by = "system"
            commit_reason = ""

        self._snapshot = self._build_snapshot(
            settings_values=effective_settings,
            dynamic_values=effective_dyn,
            specs=specs,
            epoch_id=epoch_id,
            generation=self._generation,
            committed_at=committed_at,
            committed_by=committed_by,
            commit_reason=commit_reason,
        )
        self._epoch_cache[self._snapshot.config_epoch_id] = self._snapshot
        if recovery_epoch_created:
            self._persist_epoch_locked(
                self._snapshot,
                [
                    {
                        "key": "__quarantine_recovery__",
                        "action": "quarantine_recovery",
                        "warnings": list(warnings),
                    }
                ],
            )

    def _load_dynamic_parameters_on_boot(
        self,
    ) -> Tuple[Dict[str, DynamicParameterSpec], bool, Optional[str]]:
        if not self.dynamic_params_path.exists():
            return {}, False, None
        try:
            raw_text = self.dynamic_params_path.read_text(encoding="utf-8")
            parsed = _json_loads_strict(raw_text)
            if not isinstance(parsed, list):
                raise ValueError("dynamic_parameters.json must contain a JSON array")
            if len(parsed) > MAX_DYNAMIC_PARAMETERS:
                raise ValueError(
                    f"dynamic_parameters.json exceeds max count of {MAX_DYNAMIC_PARAMETERS}"
                )
            specs: Dict[str, DynamicParameterSpec] = {}
            for item in parsed:
                if not isinstance(item, Mapping):
                    raise ValueError("Each dynamic parameter entry must be a JSON object")
                spec = DynamicParameterSpec.from_dict(item)
                if spec.key in specs:
                    raise ValueError(f"Duplicate dynamic parameter key '{spec.key}'")
                specs[spec.key] = spec
            return specs, False, None
        except Exception as exc:
            warning = _quarantine_file(self.dynamic_params_path, str(exc))
            return {}, True, warning

    def _load_overrides_on_boot(
        self,
        specs: Mapping[str, DynamicParameterSpec],
    ) -> Tuple[Dict[str, Any], bool, Optional[str]]:
        if not self.overrides_path.exists():
            return {}, False, None
        try:
            raw_text = self.overrides_path.read_text(encoding="utf-8")
            parsed = _json_loads_strict(raw_text)
            if not isinstance(parsed, Mapping):
                raise ValueError("config_overrides.json must contain a JSON object")

            validated: Dict[str, Any] = {}
            candidate_settings = dict(self._env_baseline)
            for key, raw_val in parsed.items():
                if not isinstance(key, str):
                    raise ValueError("Override key must be a string")
                if is_env_only_key(key):
                    raise ValueError(f"Env-only key '{key}' is forbidden in config_overrides.json")
                if key in SNAPSHOT_METADATA_KEYS:
                    raise ValueError(f"Read-only metadata key '{key}' is forbidden in config_overrides.json")
                spec: Optional[Union[ConfigParamSchema, DynamicParameterSpec]] = (
                    SCHEMA_BY_KEY.get(key) or specs.get(key)
                )
                if spec is None:
                    raise ValueError(f"Unknown parameter '{key}' in config_overrides.json")
                coerced, err = validate_value_against_spec(spec, raw_val)
                if err is not None:
                    raise ValueError(f"Invalid override for '{key}': {err['message']}")
                snap_val = _to_snapshot_value(key, coerced)
                validated[key] = snap_val
                if key in SCHEMA_BY_KEY:
                    candidate_settings[key] = snap_val

            cross_errors = collect_settings_invariant_errors(candidate_settings)
            if cross_errors:
                raise ValueError(
                    "; ".join(f"{e['key']}: {e['message']}" for e in cross_errors)
                )
            return validated, False, None
        except Exception as exc:
            warning = _quarantine_file(self.overrides_path, str(exc))
            return {}, True, warning

    def _load_epochs_on_boot(
        self,
    ) -> Tuple[int, Optional[Dict[str, Any]], int, List[str]]:
        if not self.epochs_path.exists():
            return 0, None, 0, []

        warnings: List[str] = []
        valid_records: List[Dict[str, Any]] = []
        try:
            raw_lines = self.epochs_path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            warning = _quarantine_file(self.epochs_path, str(exc))
            if warning:
                warnings.append(warning)
            return 0, None, 0, warnings

        corrupt_lines = 0
        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = _json_loads_strict(stripped)
                if not isinstance(rec, dict) or "config_epoch_id" not in rec or "snapshot" not in rec:
                    raise ValueError("Missing epoch record fields")
                valid_records.append(rec)
            except Exception:
                corrupt_lines += 1

        if corrupt_lines:
            warnings.append(
                f"Skipped {corrupt_lines} corrupt line(s) in '{self.epochs_path.name}'"
            )

        max_gen = 0
        last_rec: Optional[Dict[str, Any]] = None
        for rec in valid_records[-self.max_epoch_history :]:
            snap_data = rec.get("snapshot")
            if not isinstance(snap_data, Mapping):
                continue
            try:
                snap = ConfigSnapshot.from_dict(snap_data, base_defaults=self._base_defaults)
                self._epoch_cache[snap.config_epoch_id] = snap
                if snap.generation >= max_gen:
                    max_gen = snap.generation
                    last_rec = rec
            except Exception as exc:
                warnings.append(
                    f"Failed to hydrate historical epoch '{rec.get('config_epoch_id')}': {exc}"
                )

        return max_gen, last_rec, len(raw_lines), warnings

    def _compute_pending_restart_keys(self, settings_values: Mapping[str, Any]) -> Tuple[str, ...]:
        pending = [
            key
            for key, spec in SCHEMA_BY_KEY.items()
            if spec.reload_mode == "restart_required"
            and not _values_equal(settings_values[key], self._process_boot_values[key])
        ]
        pending.sort()
        return tuple(pending)

    def _build_snapshot(
        self,
        *,
        settings_values: Mapping[str, Any],
        dynamic_values: Mapping[str, Any],
        specs: Mapping[str, DynamicParameterSpec],
        epoch_id: str,
        generation: int,
        committed_at: str,
        committed_by: str,
        commit_reason: str,
    ) -> ConfigSnapshot:
        kwargs: Dict[str, Any] = {
            key: _to_snapshot_value(key, settings_values[key])
            for key in SCHEMA_BY_KEY
        }
        ordered_specs = tuple(specs[k] for k in sorted(specs))
        pending_restart = self._compute_pending_restart_keys(kwargs)
        kwargs.update(
            {
                "config_epoch_id": epoch_id,
                "generation": generation,
                "committed_at": committed_at,
                "committed_by": committed_by,
                "commit_reason": commit_reason,
                "pending_restart_keys": pending_restart,
                "config_quarantine_warnings": self._quarantine_warnings,
                "dynamic_parameters": FrozenMapping(dynamic_values),
                "dynamic_parameter_specs": ordered_specs,
            }
        )
        return ConfigSnapshot(**kwargs)

    @property
    def active_epoch_id(self) -> str:
        """Current committed epoch identifier."""
        with self._lock:
            return self._snapshot.config_epoch_id

    def active_snapshot(self) -> ConfigSnapshot:
        """Return an immutable snapshot of active configuration under lock."""
        with self._lock:
            return copy.copy(self._snapshot)

    def get_quarantine_warnings(self) -> Tuple[str, ...]:
        """Return any quarantine warnings captured during boot."""
        with self._lock:
            return self._snapshot.config_quarantine_warnings

    def list_dynamic_parameters(self) -> Tuple[DynamicParameterSpec, ...]:
        """Return active custom dynamic parameter specifications."""
        with self._lock:
            return self._snapshot.dynamic_parameter_specs

    def get_schema(self) -> Dict[str, Any]:
        """Export metadata schema including active custom dynamic parameters and profiles."""
        with self._lock:
            return export_config_schema(self._snapshot.dynamic_parameter_specs)

    def _evaluate_patch_locked(
        self,
        patch: Any,
        expected_epoch_id: Optional[str],
    ) -> Dict[str, Any]:
        current = self._snapshot
        active_epoch = current.config_epoch_id
        conflict = expected_epoch_id is not None and expected_epoch_id != active_epoch

        validation_errors: List[Dict[str, str]] = []
        if not isinstance(patch, Mapping):
            validation_errors.append(
                {
                    "key": "__patch__",
                    "code": "invalid_type",
                    "message": "Patch must be a mapping/dict",
                }
            )
            return {
                "valid": False,
                "validation_ok": False,
                "conflict": conflict,
                "expected_epoch_id": expected_epoch_id,
                "active_epoch_id": active_epoch,
                "diff": [],
                "validation_errors": validation_errors,
                "classification": {
                    "reload_mode": "hot_reload",
                    "revalidation_scope": "resize_entry",
                    "revalidation_scopes": [],
                    "hot_reload_keys": [],
                    "restart_required_keys": [],
                },
                "non_simulatable_keys": [],
                "non_simulatable_flags": [],
                "pending_restart_keys_after": list(current.pending_restart_keys),
                "_candidate_settings": None,
                "_candidate_dynamic": None,
                "_validated_patch": {},
            }

        candidate_settings = {key: getattr(current, key) for key in SCHEMA_BY_KEY}
        candidate_dynamic = current.dynamic_parameters.to_dict()
        validated_patch: Dict[str, Any] = {}
        diff: List[Dict[str, Any]] = []

        for key, raw_value in patch.items():
            if not isinstance(key, str):
                validation_errors.append(
                    {
                        "key": str(key),
                        "code": "invalid_key",
                        "message": "Configuration key must be a string",
                    }
                )
                continue
            if is_env_only_key(key):
                validation_errors.append(
                    {
                        "key": key,
                        "code": "env_only_key",
                        "message": f"Key '{key}' is env-only and cannot be modified at runtime",
                    }
                )
                continue
            if key in SNAPSHOT_METADATA_KEYS:
                validation_errors.append(
                    {
                        "key": key,
                        "code": "readonly_metadata",
                        "message": f"Key '{key}' is read-only snapshot metadata",
                    }
                )
                continue

            spec: Optional[Union[ConfigParamSchema, DynamicParameterSpec]] = (
                SCHEMA_BY_KEY.get(key) or self._dynamic_specs.get(key)
            )
            if spec is None:
                validation_errors.append(
                    {
                        "key": key,
                        "code": "unknown_key",
                        "message": f"Unknown configuration parameter '{key}'",
                    }
                )
                continue

            coerced, err = validate_value_against_spec(spec, raw_value)
            if err is not None:
                validation_errors.append(err)
                continue

            snap_val = _to_snapshot_value(key, coerced)
            validated_patch[key] = snap_val
            old_val = current.get(key)
            if key in SCHEMA_BY_KEY:
                candidate_settings[key] = snap_val
            else:
                candidate_dynamic[key] = snap_val

            if not _values_equal(old_val, snap_val):
                diff.append(
                    {
                        "key": key,
                        "label": spec.label,
                        "group": spec.group,
                        "old_value": _to_serializable_value(key, old_val),
                        "new_value": _to_serializable_value(key, snap_val),
                        "unit": spec.unit,
                        "reload_mode": spec.reload_mode,
                        "revalidation_scope": spec.revalidation_scope,
                        "impact_hint": spec.impact_hint,
                        "gate_tags": list(spec.gate_tags),
                        "simulatable": spec.group in SIMULATABLE_GROUPS,
                        "is_dynamic": isinstance(spec, DynamicParameterSpec),
                    }
                )

        cross_errors = collect_settings_invariant_errors(candidate_settings)
        seen_err = {(e["key"], e["code"], e["message"]) for e in validation_errors}
        for err in cross_errors:
            sig = (err["key"], err["code"], err["message"])
            if sig not in seen_err:
                validation_errors.append(err)
                seen_err.add(sig)

        hot_reload_keys = [d["key"] for d in diff if d["reload_mode"] == "hot_reload"]
        restart_required_keys = [d["key"] for d in diff if d["reload_mode"] == "restart_required"]
        revalidation_scopes = sorted({d["revalidation_scope"] for d in diff})
        primary_scope = (
            "abort_stale_scan"
            if "abort_stale_scan" in revalidation_scopes
            else (revalidation_scopes[0] if revalidation_scopes else "resize_entry")
        )
        non_simulatable_flags = [
            {
                "key": d["key"],
                "group": d["group"],
                "reload_mode": d["reload_mode"],
                "reason": (
                    f"Group '{d['group']}' affects downstream execution/risk/runtime state "
                    "not replayed by candidate-gate simulation"
                ),
            }
            for d in diff
            if not d["simulatable"]
        ]
        non_simulatable_keys = [flag["key"] for flag in non_simulatable_flags]

        return {
            "valid": len(validation_errors) == 0 and not conflict,
            "validation_ok": len(validation_errors) == 0,
            "conflict": conflict,
            "expected_epoch_id": expected_epoch_id,
            "active_epoch_id": active_epoch,
            "diff": diff,
            "validation_errors": validation_errors,
            "classification": {
                "reload_mode": "restart_required" if restart_required_keys else "hot_reload",
                "revalidation_scope": primary_scope,
                "revalidation_scopes": revalidation_scopes,
                "hot_reload_keys": hot_reload_keys,
                "restart_required_keys": restart_required_keys,
            },
            "non_simulatable_keys": non_simulatable_keys,
            "non_simulatable_flags": non_simulatable_flags,
            "pending_restart_keys_after": list(
                self._compute_pending_restart_keys(candidate_settings)
            ),
            "_candidate_settings": candidate_settings,
            "_candidate_dynamic": candidate_dynamic,
            "_validated_patch": validated_patch,
        }

    def preview_patch(
        self,
        patch: Mapping[str, Any],
        expected_epoch_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Dry-run validation, diff classification, simulation caveats, and CAS conflict check."""
        with self._lock:
            evaluated = self._evaluate_patch_locked(patch, expected_epoch_id)
        return {k: v for k, v in evaluated.items() if not k.startswith("_")}

    def _compute_next_overrides(
        self,
        validated_updates: Mapping[str, Any],
        specs: Mapping[str, DynamicParameterSpec],
        *,
        prune_keys: Iterable[str] = (),
    ) -> Dict[str, Any]:
        new_overrides = dict(self._overrides)
        for dead_key in prune_keys:
            new_overrides.pop(dead_key, None)

        for key, new_val in validated_updates.items():
            baseline_val = (
                self._env_baseline[key]
                if key in SCHEMA_BY_KEY
                else specs[key].default
            )
            if _values_equal(new_val, baseline_val):
                new_overrides.pop(key, None)
            else:
                new_overrides[key] = _to_serializable_value(key, new_val)

        return {k: _to_serializable_value(k, v) for k, v in new_overrides.items()}

    def _persist_epoch_locked(self, snapshot: ConfigSnapshot, diff: List[Dict[str, Any]]) -> None:
        record = {
            "config_epoch_id": snapshot.config_epoch_id,
            "generation": snapshot.generation,
            "committed_at": snapshot.committed_at,
            "committed_by": snapshot.committed_by,
            "commit_reason": snapshot.commit_reason,
            "diff": diff,
            "snapshot": snapshot.to_dict(include_specs=True),
        }
        line = _json_dumps_strict(record, indent=None) + "\n"
        self.epochs_path.parent.mkdir(parents=True, exist_ok=True)

        if self._epoch_line_count + 1 > self.max_epoch_history and self.epochs_path.exists():
            existing_lines = [
                ln
                for ln in self.epochs_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            keep = existing_lines[-(self.max_epoch_history - 1) :] if self.max_epoch_history > 1 else []
            keep.append(line.rstrip("\n"))
            _atomic_write_text(self.epochs_path, "\n".join(keep) + "\n")
            self._epoch_line_count = len(keep)
            return

        with open(self.epochs_path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        self._epoch_line_count += 1

    def _commit_state_locked(
        self,
        *,
        new_snapshot: ConfigSnapshot,
        new_overrides: Dict[str, Any],
        new_specs: Optional[Dict[str, DynamicParameterSpec]],
        diff: List[Dict[str, Any]],
    ) -> ConfigSnapshot:
        """Persist updated files atomically with rollback before publishing in-memory state."""
        prev_overrides_raw = (
            self.overrides_path.read_text(encoding="utf-8")
            if self.overrides_path.exists()
            else None
        )
        prev_dyn_raw = (
            self.dynamic_params_path.read_text(encoding="utf-8")
            if (new_specs is not None and self.dynamic_params_path.exists())
            else None
        )
        wrote_dyn = False
        wrote_overrides = False

        try:
            if new_specs is not None:
                dyn_payload = [new_specs[k].to_dict() for k in sorted(new_specs)]
                _atomic_write_text(self.dynamic_params_path, _json_dumps_strict(dyn_payload))
                wrote_dyn = True

            _atomic_write_text(self.overrides_path, _json_dumps_strict(new_overrides))
            wrote_overrides = True

            self._persist_epoch_locked(new_snapshot, diff)
        except Exception:
            if wrote_overrides:
                if prev_overrides_raw is not None:
                    _atomic_write_text(self.overrides_path, prev_overrides_raw)
                elif self.overrides_path.exists():
                    self.overrides_path.unlink()
            if wrote_dyn:
                if prev_dyn_raw is not None:
                    _atomic_write_text(self.dynamic_params_path, prev_dyn_raw)
                elif self.dynamic_params_path.exists():
                    self.dynamic_params_path.unlink()
            raise

        self._overrides = dict(new_overrides)
        if new_specs is not None:
            self._dynamic_specs = dict(new_specs)
        self._generation = new_snapshot.generation
        self._snapshot = new_snapshot
        self._epoch_cache[new_snapshot.config_epoch_id] = new_snapshot
        while len(self._epoch_cache) > self.max_epoch_history:
            oldest_key = next(iter(self._epoch_cache))
            if oldest_key != new_snapshot.config_epoch_id:
                self._epoch_cache.pop(oldest_key, None)
            else:
                break
        return copy.copy(new_snapshot)

    def commit_patch(
        self,
        patch: Mapping[str, Any],
        expected_epoch_id: str,
        author: str = "operator",
        reason: str = "",
    ) -> ConfigSnapshot:
        """Validate and atomically commit a configuration patch with optimistic CAS."""
        with self._lock:
            active_epoch = self._snapshot.config_epoch_id
            if not expected_epoch_id or expected_epoch_id != active_epoch:
                raise ConfigConflictError(
                    expected_epoch_id=expected_epoch_id,
                    actual_epoch_id=active_epoch,
                    latest_snapshot=copy.copy(self._snapshot),
                )

            evaluated = self._evaluate_patch_locked(patch, expected_epoch_id)
            if evaluated["validation_errors"]:
                raise ConfigValidationError(evaluated["validation_errors"])

            next_gen = self._generation + 1
            next_epoch_id = f"epoch-{next_gen}"
            new_overrides = self._compute_next_overrides(
                evaluated["_validated_patch"],
                self._dynamic_specs,
            )
            new_snapshot = self._build_snapshot(
                settings_values=evaluated["_candidate_settings"],
                dynamic_values=evaluated["_candidate_dynamic"],
                specs=self._dynamic_specs,
                epoch_id=next_epoch_id,
                generation=next_gen,
                committed_at=_utc_now_iso(),
                committed_by=str(author or "operator"),
                commit_reason=str(reason or ""),
            )
            return self._commit_state_locked(
                new_snapshot=new_snapshot,
                new_overrides=new_overrides,
                new_specs=None,
                diff=evaluated["diff"],
            )

    def apply_profile(
        self,
        profile_id: str,
        expected_epoch_id: str,
        author: str = "operator",
        reason: str = "",
    ) -> ConfigSnapshot:
        """Apply a built-in read-only preset profile via the standard CAS commit path."""
        profile = BUILTIN_PROFILES.get(profile_id)
        if profile is None:
            raise ConfigValidationError(
                [
                    {
                        "key": "profile_id",
                        "code": "unknown_profile",
                        "message": f"Unknown built-in profile '{profile_id}'",
                    }
                ]
            )
        effective_reason = reason or f"Apply built-in profile '{profile_id}'"
        return self.commit_patch(
            profile["values"],
            expected_epoch_id=expected_epoch_id,
            author=author,
            reason=effective_reason,
        )

    def add_dynamic_parameter(
        self,
        spec_dict: Mapping[str, Any],
        expected_epoch_id: Optional[str] = None,
        author: str = "operator",
        reason: str = "",
    ) -> ConfigSnapshot:
        """Validate, persist, and activate a new custom dynamic parameter gated by an allowlisted metric."""
        with self._lock:
            active_epoch = self._snapshot.config_epoch_id
            if expected_epoch_id is not None and expected_epoch_id != active_epoch:
                raise ConfigConflictError(
                    expected_epoch_id=expected_epoch_id,
                    actual_epoch_id=active_epoch,
                    latest_snapshot=copy.copy(self._snapshot),
                )

            try:
                spec = DynamicParameterSpec.from_dict(spec_dict)
            except (ValueError, TypeError) as exc:
                raise ConfigValidationError(
                    [
                        {
                            "key": str(spec_dict.get("key", "__dynamic__"))
                            if isinstance(spec_dict, Mapping)
                            else "__dynamic__",
                            "code": "invalid_dynamic_parameter",
                            "message": str(exc),
                        }
                    ]
                ) from exc

            if spec.key in self._dynamic_specs:
                raise ConfigValidationError(
                    [
                        {
                            "key": spec.key,
                            "code": "duplicate_key",
                            "message": f"Dynamic parameter '{spec.key}' already exists",
                        }
                    ]
                )
            if len(self._dynamic_specs) >= MAX_DYNAMIC_PARAMETERS:
                raise ConfigValidationError(
                    [
                        {
                            "key": spec.key,
                            "code": "max_dynamic_parameters_exceeded",
                            "message": f"Maximum of {MAX_DYNAMIC_PARAMETERS} custom parameters reached",
                        }
                    ]
                )

            new_specs = dict(self._dynamic_specs)
            new_specs[spec.key] = spec

            current_settings = {k: getattr(self._snapshot, k) for k in SCHEMA_BY_KEY}
            new_dynamic = self._snapshot.dynamic_parameters.to_dict()
            new_dynamic[spec.key] = spec.default

            next_gen = self._generation + 1
            next_epoch_id = f"epoch-{next_gen}"
            new_snapshot = self._build_snapshot(
                settings_values=current_settings,
                dynamic_values=new_dynamic,
                specs=new_specs,
                epoch_id=next_epoch_id,
                generation=next_gen,
                committed_at=_utc_now_iso(),
                committed_by=str(author or "operator"),
                commit_reason=str(reason or f"Add dynamic parameter '{spec.key}'"),
            )
            diff = [
                {
                    "key": spec.key,
                    "label": spec.label,
                    "group": spec.group,
                    "old_value": None,
                    "new_value": spec.default,
                    "unit": spec.unit,
                    "reload_mode": spec.reload_mode,
                    "revalidation_scope": spec.revalidation_scope,
                    "impact_hint": spec.impact_hint,
                    "gate_tags": list(spec.gate_tags),
                    "simulatable": spec.group in SIMULATABLE_GROUPS,
                    "is_dynamic": True,
                    "action": "add_dynamic_parameter",
                }
            ]
            return self._commit_state_locked(
                new_snapshot=new_snapshot,
                new_overrides=dict(self._overrides),
                new_specs=new_specs,
                diff=diff,
            )

    def delete_dynamic_parameter(
        self,
        key: str,
        expected_epoch_id: Optional[str] = None,
        author: str = "operator",
        reason: str = "",
    ) -> ConfigSnapshot:
        """Delete a custom dynamic parameter, prune any orphaned override, and advance epoch."""
        with self._lock:
            active_epoch = self._snapshot.config_epoch_id
            if expected_epoch_id is not None and expected_epoch_id != active_epoch:
                raise ConfigConflictError(
                    expected_epoch_id=expected_epoch_id,
                    actual_epoch_id=active_epoch,
                    latest_snapshot=copy.copy(self._snapshot),
                )

            spec = self._dynamic_specs.get(key)
            if spec is None:
                raise ConfigValidationError(
                    [
                        {
                            "key": str(key),
                            "code": "unknown_dynamic_parameter",
                            "message": f"Dynamic parameter '{key}' does not exist",
                        }
                    ]
                )

            new_specs = dict(self._dynamic_specs)
            new_specs.pop(key, None)

            new_overrides = self._compute_next_overrides(
                {},
                new_specs,
                prune_keys=[key],
            )
            current_settings = {k: getattr(self._snapshot, k) for k in SCHEMA_BY_KEY}
            old_value = self._snapshot.dynamic_parameters.get(key, spec.default)
            new_dynamic = self._snapshot.dynamic_parameters.to_dict()
            new_dynamic.pop(key, None)

            next_gen = self._generation + 1
            next_epoch_id = f"epoch-{next_gen}"
            new_snapshot = self._build_snapshot(
                settings_values=current_settings,
                dynamic_values=new_dynamic,
                specs=new_specs,
                epoch_id=next_epoch_id,
                generation=next_gen,
                committed_at=_utc_now_iso(),
                committed_by=str(author or "operator"),
                commit_reason=str(reason or f"Delete dynamic parameter '{key}'"),
            )
            diff = [
                {
                    "key": key,
                    "label": spec.label,
                    "group": spec.group,
                    "old_value": old_value,
                    "new_value": None,
                    "unit": spec.unit,
                    "reload_mode": spec.reload_mode,
                    "revalidation_scope": spec.revalidation_scope,
                    "impact_hint": spec.impact_hint,
                    "gate_tags": list(spec.gate_tags),
                    "simulatable": spec.group in SIMULATABLE_GROUPS,
                    "is_dynamic": True,
                    "action": "delete_dynamic_parameter",
                }
            ]
            return self._commit_state_locked(
                new_snapshot=new_snapshot,
                new_overrides=new_overrides,
                new_specs=new_specs,
                diff=diff,
            )

    def snapshot_for_epoch(self, epoch_id: str) -> Optional[ConfigSnapshot]:
        """Resolve an epoch snapshot from in-memory cache or bounded config_epochs.jsonl."""
        with self._lock:
            cached = self._epoch_cache.get(epoch_id)
            if cached is not None:
                return copy.copy(cached)

            if not self.epochs_path.exists():
                return None

            try:
                lines = self.epochs_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                return None

            for raw in reversed(lines):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    rec = _json_loads_strict(stripped)
                except Exception:
                    continue
                if isinstance(rec, Mapping) and rec.get("config_epoch_id") == epoch_id:
                    snap_data = rec.get("snapshot")
                    if not isinstance(snap_data, Mapping):
                        return None
                    snap = ConfigSnapshot.from_dict(snap_data, base_defaults=self._base_defaults)
                    self._epoch_cache[epoch_id] = snap
                    return copy.copy(snap)
            return None

    def epoch_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent epoch audit entries (without full duplicate snapshots unless requested)."""
        if limit < 1:
            return []
        with self._lock:
            if not self.epochs_path.exists():
                return []
            try:
                lines = self.epochs_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                return []

            results: List[Dict[str, Any]] = []
            for raw in reversed(lines):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    rec = _json_loads_strict(stripped)
                except Exception:
                    continue
                if isinstance(rec, Mapping):
                    results.append(
                        {
                            "config_epoch_id": rec.get("config_epoch_id"),
                            "generation": rec.get("generation"),
                            "committed_at": rec.get("committed_at"),
                            "committed_by": rec.get("committed_by"),
                            "commit_reason": rec.get("commit_reason"),
                            "diff": rec.get("diff", []),
                        }
                    )
                    if len(results) >= limit:
                        break
            return results
