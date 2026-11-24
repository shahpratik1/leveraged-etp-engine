"""Vectorized Feature Engine with Zero Lookahead Bias for MLAlphaStrategy.

Computes 14 distinct quantitative features over chronological OHLCV bars
strictly utilizing historical data (t <= T). Designed for sub-millisecond execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


FEATURE_NAMES: Tuple[str, ...] = (
    "vol_drag_proxy",
    "kaufman_er_10",
    "kaufman_er_20",
    "variance_ratio_5",
    "atr_pct_rank_60",
    "ema_spread_atr",
    "ema20_slope_z",
    "ret_10",
    "donchian_pos_20",
    "signed_dmi_strength",
    "vol_zscore_20",
    "range_expansion",
    "gap_ratio",
    "lse_time_remaining",
)


def compute_feature_schema_hash() -> str:
    """Compute SHA-256 fingerprint of ordered FEATURE_NAMES schema."""
    schema_str = ",".join(FEATURE_NAMES)
    return hashlib.sha256(schema_str.encode("utf-8")).hexdigest()


def _parse_utc_datetime(val: Any) -> Optional[datetime]:
    """Parse various datetime representations into UTC datetime."""
    if val is None:
        return None
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val.astimezone(timezone.utc)
    if isinstance(val, (int, float)):
        ts = float(val)
        if ts > 1e11:  # Milliseconds
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(val, str):
        val = val.strip()
        # Fast ISO format parsing
        try:
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(val, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
    return None


def _extract_series(
    bars: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Any]]:
    """Extract numeric arrays and date values from bar dicts or objects."""
    n = len(bars)
    opens = np.empty(n, dtype=np.float64)
    highs = np.empty(n, dtype=np.float64)
    lows = np.empty(n, dtype=np.float64)
    closes = np.empty(n, dtype=np.float64)
    volumes = np.empty(n, dtype=np.float64)
    dates: List[Any] = []

    for i, b in enumerate(bars):
        if isinstance(b, dict):
            o = b.get("open", b.get("Open", 0.0))
            h = b.get("high", b.get("High", 0.0))
            l = b.get("low", b.get("Low", 0.0))
            c = b.get("close", b.get("Close", 0.0))
            v = b.get("volume", b.get("Volume", b.get("vol", 0.0)))
            d = b.get("date", b.get("Date", b.get("timestamp", b.get("time", None))))
        else:
            o = getattr(b, "open", 0.0)
            h = getattr(b, "high", 0.0)
            l = getattr(b, "low", 0.0)
            c = getattr(b, "close", 0.0)
            v = getattr(b, "volume", 0.0)
            d = getattr(b, "date", getattr(b, "timestamp", getattr(b, "time", None)))

        opens[i] = float(o if o is not None else 0.0)
        highs[i] = float(h if h is not None else 0.0)
        lows[i] = float(l if l is not None else 0.0)
        closes[i] = float(c if c is not None else 0.0)
        volumes[i] = float(v if v is not None and not math.isnan(float(v)) else 0.0)
        dates.append(d)

    return opens, highs, lows, closes, volumes, dates


def _compute_atr_series(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Vectorized Wilder True Range and ATR series."""
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([max(highs[0] - lows[0], 1e-8)], dtype=np.float64)

    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    prev_c = closes[:-1]
    curr_h = highs[1:]
    curr_l = lows[1:]
    hl = curr_h - curr_l
    hc = np.abs(curr_h - prev_c)
    lc = np.abs(curr_l - prev_c)
    tr[1:] = np.maximum(hl, np.maximum(hc, lc))

    atr = np.empty(n, dtype=np.float64)
    if n <= period:
        atr = np.cumsum(tr) / np.arange(1, n + 1)
        return atr

    atr[:period - 1] = np.cumsum(tr[:period - 1]) / np.arange(1, period)
    curr_atr = float(np.mean(tr[:period]))
    atr[period - 1] = curr_atr
    alpha = 1.0 / period
    for i in range(period, n):
        curr_atr = curr_atr + alpha * (tr[i] - curr_atr)
        atr[i] = curr_atr

    return atr


def _compute_ema_series(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average series with smoothing factor 2 / (period + 1)."""
    n = len(values)
    if n == 0:
        return np.array([], dtype=np.float64)
    alpha = 2.0 / (period + 1.0)
    ema = np.empty(n, dtype=np.float64)
    curr = values[0]
    ema[0] = curr
    for i in range(1, n):
        curr = alpha * values[i] + (1.0 - alpha) * curr
        ema[i] = curr
    return ema


def _compute_wilder_dmi_adx(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> Tuple[float, float, float]:
    """Calculate Wilder +DI14, -DI14, and ADX14 for the latest bar."""
    n = len(closes)
    if n < 2:
        return 0.0, 0.0, 0.0

    prev_h = highs[:-1]
    prev_l = lows[:-1]
    prev_c = closes[:-1]
    curr_h = highs[1:]
    curr_l = lows[1:]

    up = curr_h - prev_h
    down = prev_l - curr_l

    pdm = np.where((up > down) & (up > 0.0), up, 0.0)
    mdm = np.where((down > up) & (down > 0.0), down, 0.0)

    hl = curr_h - curr_l
    hc = np.abs(curr_h - prev_c)
    lc = np.abs(curr_l - prev_c)
    tr = np.maximum(hl, np.maximum(hc, lc))

    m = len(tr)
    alpha = 1.0 / period

    if m < period:
        s_tr = float(np.sum(tr))
        s_pdm = float(np.sum(pdm))
        s_mdm = float(np.sum(mdm))
        pdi = 100.0 * s_pdm / (s_tr + 1e-8)
        mdi = 100.0 * s_mdm / (s_tr + 1e-8)
        dx = 100.0 * abs(pdi - mdi) / (pdi + mdi + 1e-8)
        return pdi, mdi, dx

    # Initial Wilder sum over first period bars
    s_tr = float(np.sum(tr[:period]))
    s_pdm = float(np.sum(pdm[:period]))
    s_mdm = float(np.sum(mdm[:period]))

    pdi = 100.0 * s_pdm / (s_tr + 1e-8)
    mdi = 100.0 * s_mdm / (s_tr + 1e-8)
    dx_warmup: List[float] = [100.0 * abs(pdi - mdi) / (pdi + mdi + 1e-8)]

    for i in range(period, m):
        s_tr = s_tr - (s_tr / period) + tr[i]
        s_pdm = s_pdm - (s_pdm / period) + pdm[i]
        s_mdm = s_mdm - (s_mdm / period) + mdm[i]
        pdi = 100.0 * s_pdm / (s_tr + 1e-8)
        mdi = 100.0 * s_mdm / (s_tr + 1e-8)
        dx = 100.0 * abs(pdi - mdi) / (pdi + mdi + 1e-8)
        dx_warmup.append(dx)

    if len(dx_warmup) < period:
        adx = float(np.mean(dx_warmup))
    else:
        adx = float(np.mean(dx_warmup[:period]))
        for dx_val in dx_warmup[period:]:
            adx = adx + alpha * (dx_val - adx)

    return pdi, mdi, adx


def extract_features(
    bars: Sequence[Dict[str, Any]],
    leverage: float = 3.0,
    current_utc_time: Optional[datetime] = None,
) -> np.ndarray:
    """Extract 14 vectorized quantitative features strictly on historical bars [:t+1].

    Args:
        bars: Chronological sequence of OHLCV bar dicts (open, high, low, close, volume, date).
        leverage: Underlying strategy leverage factor (default 3.0).
        current_utc_time: Optional explicit current UTC time for intraday session calculation.

    Returns:
        np.ndarray of shape (1, 14) and dtype np.float32 with zero lookahead bias.
    """
    n = len(bars)
    if n == 0:
        return np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32)

    opens, highs, lows, closes, volumes, dates = _extract_series(bars)
    c_t = closes[-1]
    h_t = highs[-1]
    l_t = lows[-1]
    v_t = volumes[-1]

    # Pre-compute ATR series
    atr_series = _compute_atr_series(highs, lows, closes, period=14)
    atr14 = atr_series[-1] if len(atr_series) > 0 else max(h_t - l_t, 1e-8)

    # 1. vol_drag_proxy: (leverage ** 2) * realized_var_20
    if n >= 21:
        sub_c = closes[-21:]
        log_rets = np.log(np.maximum(sub_c[1:], 1e-8) / np.maximum(sub_c[:-1], 1e-8))
        realized_var_20 = float(np.var(log_rets, ddof=1)) if len(log_rets) > 1 else 0.0
    elif n >= 2:
        log_rets = np.log(np.maximum(closes[1:], 1e-8) / np.maximum(closes[:-1], 1e-8))
        realized_var_20 = float(np.var(log_rets, ddof=1)) if len(log_rets) > 1 else 0.0
    else:
        realized_var_20 = 0.0
    vol_drag_proxy = float((leverage ** 2) * realized_var_20)
    vol_drag_proxy = float(np.nan_to_num(vol_drag_proxy, nan=0.0, posinf=0.0, neginf=0.0))

    # 2. kaufman_er_10: |C_t - C_{t-10}| / (sum_{k=0..9} |C_{t-k} - C_{t-k-1}| + 1e-8)
    if n >= 11:
        sub_c10 = closes[-11:]
        net_change_10 = abs(sub_c10[-1] - sub_c10[0])
        volatility_10 = float(np.sum(np.abs(np.diff(sub_c10))))
        kaufman_er_10 = float(net_change_10 / (volatility_10 + 1e-8))
    elif n >= 2:
        net_change_10 = abs(closes[-1] - closes[0])
        volatility_10 = float(np.sum(np.abs(np.diff(closes))))
        kaufman_er_10 = float(net_change_10 / (volatility_10 + 1e-8))
    else:
        kaufman_er_10 = 0.0
    kaufman_er_10 = float(np.clip(kaufman_er_10, 0.0, 1.0))

    # 3. kaufman_er_20: |C_t - C_{t-20}| / (sum_{k=0..19} |C_{t-k} - C_{t-k-1}| + 1e-8)
    if n >= 21:
        sub_c20 = closes[-21:]
        net_change_20 = abs(sub_c20[-1] - sub_c20[0])
        volatility_20 = float(np.sum(np.abs(np.diff(sub_c20))))
        kaufman_er_20 = float(net_change_20 / (volatility_20 + 1e-8))
    elif n >= 2:
        net_change_20 = abs(closes[-1] - closes[0])
        volatility_20 = float(np.sum(np.abs(np.diff(closes))))
        kaufman_er_20 = float(net_change_20 / (volatility_20 + 1e-8))
    else:
        kaufman_er_20 = 0.0
    kaufman_er_20 = float(np.clip(kaufman_er_20, 0.0, 1.0))

    # 4. variance_ratio_5: Var(5-bar returns) / (5.0 * Var(1-bar returns) + 1e-8)
    if n < 7:
        variance_ratio_5 = 1.0
    else:
        sub_window = min(n, 35)
        sub_closes = closes[-sub_window:]
        r1 = np.diff(sub_closes) / (sub_closes[:-1] + 1e-8)
        r5 = (sub_closes[5:] - sub_closes[:-5]) / (sub_closes[:-5] + 1e-8)
        var1 = float(np.var(r1, ddof=1)) if len(r1) > 1 else 0.0
        var5 = float(np.var(r5, ddof=1)) if len(r5) > 1 else 0.0
        vr = var5 / (5.0 * var1 + 1e-8)
        variance_ratio_5 = float(np.nan_to_num(vr, nan=1.0, posinf=1.0, neginf=1.0))
        variance_ratio_5 = float(np.clip(variance_ratio_5, 0.0, 20.0))

    # 5. atr_pct_rank_60: Percentile rank in [0.0, 1.0] of current (ATR14 / Close) within rolling 60-bar window
    norm_atr = atr_series / np.maximum(closes, 1e-8)
    w_atr = norm_atr[-min(len(norm_atr), 60):]
    curr_norm_atr = norm_atr[-1]
    if len(w_atr) > 1:
        rank_60 = float(np.mean(w_atr <= curr_norm_atr))
    else:
        rank_60 = 0.5
    atr_pct_rank_60 = float(np.clip(rank_60, 0.0, 1.0))

    # 6. ema_spread_atr: (EMA8 - EMA21) / (ATR14 + 1e-8)
    ema8 = _compute_ema_series(closes, 8)
    ema21 = _compute_ema_series(closes, 21)
    ema_spread = (ema8[-1] - ema21[-1]) / (atr14 + 1e-8)
    ema_spread_atr = float(np.clip(ema_spread, -10.0, 10.0))

    # 7. ema20_slope_z: 5-bar slope of EMA20 ((EMA20_t - EMA20_{t-5}) / (EMA20_{t-5} + 1e-8)) normalized by rolling 20-bar std of 5-bar slopes
    ema20 = _compute_ema_series(closes, 20)
    if n >= 6:
        slopes_5 = (ema20[5:] - ema20[:-5]) / (ema20[:-5] + 1e-8)
        curr_slope = slopes_5[-1]
        w_slopes = slopes_5[-min(len(slopes_5), 20):]
        s_std = float(np.std(w_slopes, ddof=1)) if len(w_slopes) > 1 else 0.0
        if s_std > 1e-8:
            ema20_slope_z = float(curr_slope / (s_std + 1e-8))
        else:
            ema20_slope_z = 0.0
    else:
        ema20_slope_z = 0.0
    ema20_slope_z = float(np.clip(ema20_slope_z, -10.0, 10.0))

    # 8. ret_10: ln(C_t / (C_{t-10} + 1e-8))
    if n >= 11:
        ret_10 = float(np.log(max(c_t, 1e-8) / max(closes[-11], 1e-8)))
    elif n >= 2:
        ret_10 = float(np.log(max(c_t, 1e-8) / max(closes[0], 1e-8)))
    else:
        ret_10 = 0.0
    ret_10 = float(np.clip(ret_10, -5.0, 5.0))

    # 9. donchian_pos_20: (Close - Low20) / (High20 - Low20 + 1e-8), clamped to [0.0, 1.0]
    w_highs = highs[-min(n, 20):]
    w_lows = lows[-min(n, 20):]
    high20 = float(np.max(w_highs))
    low20 = float(np.min(w_lows))
    if high20 > low20:
        donchian_pos = (c_t - low20) / (high20 - low20 + 1e-8)
    else:
        donchian_pos = 0.5
    donchian_pos_20 = float(np.clip(donchian_pos, 0.0, 1.0))

    # 10. signed_dmi_strength: ((+DI14 - -DI14) / (+DI14 + -DI14 + 1e-8)) * (ADX14 / 100.0)
    pdi14, mdi14, adx14 = _compute_wilder_dmi_adx(highs, lows, closes, period=14)
    di_diff = pdi14 - mdi14
    di_sum = pdi14 + mdi14
    signed_dmi = (di_diff / (di_sum + 1e-8)) * (adx14 / 100.0)
    signed_dmi_strength = float(np.clip(signed_dmi, -1.0, 1.0))

    # 11. vol_zscore_20: (Volume - mean(Vol20)) / (std(Vol20) + 1e-8) with fallback 0.0
    w_vols = volumes[-min(n, 20):]
    if len(w_vols) >= 2 and not np.all(w_vols == 0.0):
        v_mean = float(np.mean(w_vols))
        v_std = float(np.std(w_vols, ddof=1))
        if v_std > 1e-8:
            vol_z = (v_t - v_mean) / (v_std + 1e-8)
        else:
            vol_z = 0.0
    else:
        vol_z = 0.0
    vol_zscore_20 = float(np.clip(vol_z, -10.0, 10.0))

    # 12. range_expansion: (High_t - Low_t) / (ATR14 + 1e-8)
    range_expansion = float((h_t - l_t) / (atr14 + 1e-8))
    range_expansion = float(np.clip(range_expansion, 0.0, 20.0))

    # 13. gap_ratio: (Open_t - Close_{t-1}) / (ATR14 + 1e-8)
    if n >= 2:
        gap = opens[-1] - closes[-2]
        gap_ratio = float(gap / (atr14 + 1e-8))
    else:
        gap_ratio = 0.0
    gap_ratio = float(np.clip(gap_ratio, -10.0, 10.0))

    # 14. lse_time_remaining: Clamped [0.0, 1.0] fraction of session left before 16:20 UTC
    target_dt: Optional[datetime] = current_utc_time
    if target_dt is None and dates[-1] is not None:
        target_dt = _parse_utc_datetime(dates[-1])

    if target_dt is not None:
        # Check if intraday time component exists
        if target_dt.hour == 0 and target_dt.minute == 0 and target_dt.second == 0:
            # Pure daily candle without intraday time
            lse_time_remaining = 0.0
        else:
            minutes = target_dt.hour * 60.0 + target_dt.minute + target_dt.second / 60.0
            open_m = 480.0   # 08:00 UTC
            close_m = 980.0  # 16:20 UTC
            if minutes <= open_m:
                rem = 1.0
            elif minutes >= close_m:
                rem = 0.0
            else:
                rem = (close_m - minutes) / (close_m - open_m)
            lse_time_remaining = float(np.clip(rem, 0.0, 1.0))
    else:
        # Default for daily candles when no time is given
        lse_time_remaining = 0.0

    features = np.array(
        [[
            vol_drag_proxy,
            kaufman_er_10,
            kaufman_er_20,
            variance_ratio_5,
            atr_pct_rank_60,
            ema_spread_atr,
            ema20_slope_z,
            ret_10,
            donchian_pos_20,
            signed_dmi_strength,
            vol_zscore_20,
            range_expansion,
            gap_ratio,
            lse_time_remaining,
        ]],
        dtype=np.float32,
    )

    return features


def extract_features_matrix(
    bars: Sequence[Dict[str, Any]],
    leverage: float = 3.0,
) -> np.ndarray:
    """Extract 14 features across all historical bars [:i+1] for offline model training.

    Args:
        bars: Chronological sequence of OHLCV bar dicts.
        leverage: Underlying strategy leverage factor.

    Returns:
        np.ndarray of shape (N, 14) and dtype np.float32.
    """
    n = len(bars)
    if n == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    matrix = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    # Fast sliding extraction strictly [:i+1]
    for i in range(n):
        # Slicing up to 100 bars for warmth is sufficient and zero-lookahead
        start_idx = max(0, i - 100)
        matrix[i] = extract_features(bars[start_idx : i + 1], leverage=leverage)[0]

    return matrix
