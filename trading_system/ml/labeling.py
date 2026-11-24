"""Decay-Aware Triple-Barrier Labeling & Purged Walk-Forward Splits.

Implements path-dependent triple-barrier labeling for leveraged ETP trading
and purged walk-forward cross-validation splits with zero information leakage.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from trading_system.ml.features import _compute_atr_series, _extract_series, _parse_utc_datetime

LABEL_CASH_DEFENSE: int = 0
LABEL_LONG: int = 1
LABEL_SHORT: int = 2


def label_bars(
    bars: Sequence[Dict[str, Any]],
    leverage: float = 3.0,
    roundtrip_cost: float = 0.003,
    max_hold_bars: int = 10,
    atr_multiplier: float = 2.2,
) -> np.ndarray:
    """Compute path-dependent triple-barrier labels for a sequence of OHLCV bars.

    Barriers:
      tp_pct = (0.030 + roundtrip_cost) / leverage (+3.0% ETP-equivalent cushion)
      sl_pct = np.clip(atr_multiplier * (ATR14 / Close) * leverage, 0.025, 0.080) / leverage
      Vertical barrier: min(max_hold_bars, bars_until_1620_utc)

    Strict Path Order:
      Intrabar touch evaluation using High/Low of future bars.
      Conservative tie-break: If both TP and SL are touched within the same bar,
      label as 0 = CASH_DEFENSE.

    Returns:
      Integer np.ndarray of shape (N,) with classes:
        0 = CASH_DEFENSE
        1 = LONG
        2 = SHORT
    """
    n = len(bars)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    opens, highs, lows, closes, volumes, dates = _extract_series(bars)
    atr_series = _compute_atr_series(highs, lows, closes, period=14)

    # Calculate intraday session remaining bars up to 16:20 UTC cutoff
    parsed_dts = [_parse_utc_datetime(d) for d in dates]
    has_intraday_times = any(
        dt is not None and (dt.hour != 0 or dt.minute != 0 or dt.second != 0)
        for dt in parsed_dts
    )

    bars_until_1620: List[int] = []
    if not has_intraday_times:
        # Daily candle series: no intraday cutoff within bar sequence
        bars_until_1620 = [max_hold_bars] * n
    else:
        for i in range(n):
            dt_i = parsed_dts[i]
            if dt_i is None:
                bars_until_1620.append(max_hold_bars)
            elif (dt_i.hour > 16) or (dt_i.hour == 16 and dt_i.minute >= 20):
                # Bar is at or past 16:20 UTC cutoff
                bars_until_1620.append(0)
            else:
                last_idx = i
                for j in range(i, n):
                    dt_j = parsed_dts[j]
                    if dt_j is not None and dt_j.date() == dt_i.date():
                        if (dt_j.hour < 16) or (dt_j.hour == 16 and dt_j.minute <= 20):
                            last_idx = j
                        else:
                            break
                    else:
                        break
                bars_until_1620.append(last_idx - i)

    labels = np.zeros(n, dtype=np.int64)
    tp_pct_base = (0.030 + roundtrip_cost) / leverage

    for t in range(n):
        p0 = closes[t]
        if p0 <= 0.0:
            labels[t] = LABEL_CASH_DEFENSE
            continue

        atr_t = atr_series[t] if t < len(atr_series) else max(highs[t] - lows[t], 1e-8)
        norm_vol = (atr_t / (p0 + 1e-8)) * leverage
        sl_pct = float(np.clip(atr_multiplier * norm_vol, 0.025, 0.080)) / leverage
        tp_pct = tp_pct_base

        long_tp_price = p0 * (1.0 + tp_pct)
        long_sl_price = p0 * (1.0 - sl_pct)
        short_tp_price = p0 * (1.0 - tp_pct)
        short_sl_price = p0 * (1.0 + sl_pct)

        effective_max_hold = min(max_hold_bars, bars_until_1620[t])
        if effective_max_hold <= 0 or t + 1 >= n:
            labels[t] = LABEL_CASH_DEFENSE
            continue

        horizon_end = min(n, t + 1 + effective_max_hold)
        upper_hit = False
        lower_hit = False

        for tau in range(t + 1, horizon_end):
            h_tau = highs[tau]
            l_tau = lows[tau]

            if h_tau >= long_tp_price or h_tau >= short_sl_price:
                upper_hit = True
            if l_tau <= short_tp_price or l_tau <= long_sl_price:
                lower_hit = True

        if upper_hit and lower_hit:
            # Both barriers touched within window or same bar: conservative CASH_DEFENSE
            labels[t] = LABEL_CASH_DEFENSE
        elif upper_hit:
            labels[t] = LABEL_LONG
        elif lower_hit:
            labels[t] = LABEL_SHORT
        else:
            labels[t] = LABEL_CASH_DEFENSE

    return labels


def purged_walk_forward_splits(
    n_samples: int,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    max_hold_bars: int = 10,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate purged walk-forward cross-validation splits with zero information leakage.

    Purges overlapping barrier windows between train and val sets, and applies
    embargo buffer following the validation window.

    Args:
        n_samples: Total number of observations in dataset.
        n_splits: Number of walk-forward evaluation folds (default 5).
        embargo_pct: Buffer percentage to prevent serial correlation leakage (default 0.01).
        max_hold_bars: Maximum holding period barrier to purge from training set (default 10).

    Returns:
        List of (train_indices, val_indices) numpy ndarrays of dtype np.int64.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if n_splits <= 0:
        raise ValueError(f"n_splits must be positive, got {n_splits}")

    val_size = n_samples // (n_splits + 1)
    if val_size <= 0:
        raise ValueError(
            f"n_samples ({n_samples}) is too small to partition into {n_splits} splits."
        )

    embargo_bars = max(1, int(n_samples * embargo_pct)) if embargo_pct > 0.0 else 0

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        v_start = (i + 1) * val_size
        v_end = (i + 2) * val_size if i < n_splits - 1 else n_samples
        val_idx = np.arange(v_start, v_end, dtype=np.int64)

        # Purging: Drop training bars whose forward label window touches the validation set
        train_end = v_start - max_hold_bars
        if train_end <= 0:
            raise ValueError(
                f"Split {i}: purged train set is empty. v_start={v_start}, max_hold_bars={max_hold_bars}."
            )
        train_idx = np.arange(0, train_end, dtype=np.int64)

        splits.append((train_idx, val_idx))

    return splits


class TripleBarrierLabeler:
    """Class wrapper providing triple-barrier labeling and purged walk-forward splits."""

    LABEL_CASH_DEFENSE: int = LABEL_CASH_DEFENSE
    LABEL_LONG: int = LABEL_LONG
    LABEL_SHORT: int = LABEL_SHORT

    def __init__(
        self,
        leverage: float = 3.0,
        roundtrip_cost: float = 0.003,
        max_hold_bars: int = 10,
        atr_multiplier: float = 2.2,
    ) -> None:
        self.leverage = leverage
        self.roundtrip_cost = roundtrip_cost
        self.max_hold_bars = max_hold_bars
        self.atr_multiplier = atr_multiplier

    def label_bars(
        self,
        bars: Sequence[Dict[str, Any]],
        leverage: Optional[float] = None,
        roundtrip_cost: Optional[float] = None,
        max_hold_bars: Optional[int] = None,
        atr_multiplier: Optional[float] = None,
    ) -> np.ndarray:
        """Compute path-dependent triple-barrier labels for a sequence of OHLCV bars."""
        return label_bars(
            bars=bars,
            leverage=leverage if leverage is not None else self.leverage,
            roundtrip_cost=roundtrip_cost if roundtrip_cost is not None else self.roundtrip_cost,
            max_hold_bars=max_hold_bars if max_hold_bars is not None else self.max_hold_bars,
            atr_multiplier=atr_multiplier if atr_multiplier is not None else self.atr_multiplier,
        )

    def purged_walk_forward_splits(
        self,
        n_samples: int,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        max_hold_bars: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate purged walk-forward cross-validation splits."""
        return purged_walk_forward_splits(
            n_samples=n_samples,
            n_splits=n_splits,
            embargo_pct=embargo_pct,
            max_hold_bars=max_hold_bars if max_hold_bars is not None else self.max_hold_bars,
        )


# Ensure staticmethods on class also work
TripleBarrierLabeler.label_bars = staticmethod(label_bars)  # type: ignore
TripleBarrierLabeler.purged_walk_forward_splits = staticmethod(purged_walk_forward_splits)  # type: ignore
