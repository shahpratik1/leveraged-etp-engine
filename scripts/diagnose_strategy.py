#!/usr/bin/env python3
"""Standalone, Read-Only Strategy Diagnostics & Performance Audit Tool.

Analyzes execution logs (live/paper) and historical backtest outputs for leveraged-etp-engine:
1. Headline Edge (Gross vs. True Net after LSE ETP friction).
2. Exit-Reason Attribution (INITIAL_GTC_STOP, BREAKEVEN_RATCHET, PROFIT_CUSHION, CHANDELIER_TRAIL, EVACUATION_1620_UTC, OTHER).
3. MFE / MAE & Early-Exit Choke Audit (Intrabar excursion & post-exit 24-48h moves via CandleDatabase).
4. Segment Breakdown (LONG vs SHORT & 3x Index, 5x Index, 3x Single Stock, Commodity/Bond).
5. Automated Bottleneck Verdict (Data-backed diagnostic flags).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
import sqlite3
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Models & Constants
# ─────────────────────────────────────────────────────────────────────────────

VALID_EXIT_BUCKETS = (
    "INITIAL_GTC_STOP",
    "BREAKEVEN_RATCHET",
    "PROFIT_CUSHION",
    "CHANDELIER_TRAIL",
    "EVACUATION_1620_UTC",
    "OTHER",
)

VALID_ASSET_TIERS = (
    "3x Index",
    "5x Index",
    "3x Single Stock",
    "Commodity/Bond",
)


@dataclass
class TradeRecord:
    """Normalized trade execution record for diagnostics analysis."""
    trade_id: Union[int, str]
    symbol: str
    underlying: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float
    entry_date: str
    exit_date: str
    shares: float = 1.0
    notional: float = 0.0
    initial_sl: float = 0.0
    gross_pnl: float = 0.0
    gross_return_pct: float = 0.0
    net_pnl: float = 0.0
    net_return_pct: float = 0.0
    exit_reason: str = "OTHER"
    raw_exit_reason: str = ""
    leverage: int = 3
    asset_tier: str = "3x Index"
    # MFE / MAE metrics
    mfe_pct: Optional[float] = None
    mae_pct: Optional[float] = None
    post_exit_run_3pct: Optional[bool] = None
    hold_days: int = 0
    r_multiple_gross: float = 0.0
    r_multiple_net: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Classification & Mapping Helpers
# ─────────────────────────────────────────────────────────────────────────────

def map_exit_reason(raw: Optional[str]) -> str:
    """Auto-map various log/backtest exit causes to canonical diagnostic buckets.
    
    Canonical buckets:
    - INITIAL_GTC_STOP
    - BREAKEVEN_RATCHET
    - PROFIT_CUSHION
    - CHANDELIER_TRAIL
    - EVACUATION_1620_UTC
    - OTHER
    """
    if not raw:
        return "OTHER"
    r = str(raw).strip().upper()

    # Initial stop-loss triggers
    if r in {"SL", "INITIAL", "INITIAL_SL", "INITIAL_GTC_STOP", "STOP_LOSS", "GTC_STOP", "STOP"}:
        return "INITIAL_GTC_STOP"

    # Breakeven ratchets
    if r in {"TSL_BREAKEVEN", "TP1_SL_BE", "BREAKEVEN", "BREAKEVEN_RATCHET", "BE_RATCHET", "BE", "TSL_BE"}:
        return "BREAKEVEN_RATCHET"

    # Profit cushion ratchets
    if r in {"TSL_CUSHION", "PROFIT_CUSHION", "CUSHION", "LOCK_CUSHION"}:
        return "PROFIT_CUSHION"

    # Chandelier dynamic trailing stop exits
    if r in {"TP2_TSL", "TSL", "CHANDELIER", "CHANDELIER_TRAIL", "DYNAMIC_ATR", "TSL_RATCHET", "TRAILING_STOP"}:
        return "CHANDELIER_TRAIL"

    # Session evacuation exits (16:20 UTC cutoff, zero-alpha evacuation, inverse flatten)
    if any(k in r for k in ("EVACUATION", "ASYMMETRIC", "1620", "SESSION_CLOSE", "INVERSE_INTRADAY", "FLATTEN", "ZERO_ALPHA")):
        return "EVACUATION_1620_UTC"

    return "OTHER"


def classify_asset_tier(
    symbol: str,
    underlying: Optional[str] = None,
    leverage: Optional[int] = None,
    universe: Optional[Any] = None,
) -> str:
    """Classify instrument into 3x Index, 5x Index, 3x Single Stock, or Commodity/Bond."""
    sym = (symbol or "").strip()
    und = (underlying or sym).strip().upper()
    lev = int(leverage or 3)

    # 1. Query LeveragedUniverse if available
    pair = None
    if universe is not None:
        try:
            pair = universe.get_pair_for_underlying(und)
            if not pair:
                pair = getattr(universe, "_etp_map", {}).get(sym.upper())
        except Exception:
            pair = None

    cat = getattr(pair, "category", "") if pair else ""
    actual_und = getattr(pair, "underlying_symbol", und) if pair else und

    # Check 5x indicators from symbol name or universe
    is_5x = (
        lev == 5
        or "5" in sym
        or (pair and getattr(pair, "has_5x", False) and lev == 5)
    )

    # Known commodity & fixed income underlyings & categories
    commodity_bond_underlyings = {
        "GLD", "SLV", "USO", "WEAT", "COPPER", "TLT", "BUND", "3GIL", "GDX", "SILV",
        "WTI", "BNO", "UNG", "DBA", "CORN", "SOYB", "PALL", "PPLT", "HYG", "LQD",
    }
    commodity_bond_categories = {
        "Commodity", "Fixed Income", "Commodity Equity", "Currency",
    }
    if cat in commodity_bond_categories or actual_und in commodity_bond_underlyings or any(
        sym.startswith(p) for p in ("3WHL", "WTI3", "3TYL", "TLT5", "3GLE", "GLD5", "3SIL", "3OIL")
    ):
        return "Commodity/Bond"

    # Known broad / tech indices
    index_underlyings = {
        "QQQ", "SPY", "IWM", "DIA", "SOXX", "SMH", "XLF", "^FTSE", "^GDAXI", "^STOXX50E",
        "^FCHI", "FXI", "EEM", "MAGS", "NDX", "DAX", "VIX", "EWG", "EWU", "EWJ",
    }
    index_categories = {
        "Index", "Global Index", "Sector ETF", "Tech Basket",
    }

    if cat in index_categories or actual_und in index_underlyings or any(
        k in sym.upper() for k in ("QQQ", "SPY", "5QQE", "SP5Y", "3USL", "3DES", "3EUL", "5SX5")
    ):
        if is_5x:
            return "5x Index"
        return "3x Index"

    # Single stock equities
    return "3x Single Stock"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Trade Ingestion & Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_trade(
    data: Dict[str, Any],
    cost_bps: float = 45.0,
    universe: Optional[Any] = None,
) -> TradeRecord:
    """Normalize a raw dictionary or SimulatedTrade into a TradeRecord with friction."""
    symbol = str(data.get("symbol") or data.get("ticker") or "UNKNOWN").strip()
    underlying = str(data.get("underlying") or data.get("ticker") or symbol).strip().upper()
    direction = str(data.get("direction") or "LONG").strip().upper()
    if direction not in ("LONG", "SHORT"):
        direction = "LONG"

    entry_price = float(data.get("entry_price") or 0.0)
    exit_price = float(data.get("exit_price") or data.get("price") or entry_price)
    entry_date = str(data.get("entry_date") or data.get("entry_time") or data.get("timestamp") or "")[:10]
    exit_date = str(data.get("exit_date") or data.get("exit_time") or data.get("timestamp") or entry_date)[:10]

    shares = float(data.get("shares") or data.get("quantity") or data.get("position_size_shares") or 1.0)
    notional = float(data.get("notional") or data.get("position_notional") or 0.0)
    if notional <= 0.0 and entry_price > 0.0 and shares > 0.0:
        notional = entry_price * shares

    initial_sl = float(data.get("initial_sl") or data.get("stop_loss") or 0.0)
    leverage = int(data.get("leverage") or 3)
    raw_exit_reason = str(data.get("exit_reason") or data.get("event_type") or data.get("reason") or "OTHER")
    exit_reason = map_exit_reason(raw_exit_reason)

    # Gross PnL and return %
    gross_pnl = 0.0
    gross_return_pct = 0.0

    if data.get("realized_pnl") is not None:
        gross_pnl = float(data["realized_pnl"])
    elif data.get("pnl") is not None:
        gross_pnl = float(data["pnl"])

    if data.get("return_pct") is not None:
        gross_return_pct = float(data["return_pct"])
    elif data.get("pnl_pct") is not None:
        gross_return_pct = float(data["pnl_pct"])

    # If missing, calculate from entry and exit prices
    if gross_return_pct == 0.0 and entry_price > 0.0 and exit_price > 0.0 and entry_price != exit_price:
        if direction == "LONG":
            gross_return_pct = ((exit_price - entry_price) / entry_price) * 100.0
        else:
            gross_return_pct = ((entry_price - exit_price) / entry_price) * 100.0

    if gross_pnl == 0.0 and notional > 0.0 and gross_return_pct != 0.0:
        gross_pnl = notional * (gross_return_pct / 100.0)
    elif notional <= 0.0 and gross_pnl != 0.0 and gross_return_pct != 0.0:
        notional = abs(gross_pnl / (gross_return_pct / 100.0))

    if notional <= 0.0:
        notional = 1000.0  # Fallback standard notional unit

    # ── Realistic LSE ETP Friction Math (spread + FX) ──
    # cost_bps is round-trip friction in basis points (default 45.0 bps = 0.45%)
    friction_rate = cost_bps / 10000.0
    friction_pct = cost_bps / 100.0  # e.g., 0.45%
    friction_cost = notional * friction_rate

    net_pnl = gross_pnl - friction_cost
    net_return_pct = gross_return_pct - friction_pct

    # R-multiple computation
    risk_unit = 0.0
    if initial_sl > 0.0 and entry_price > 0.0:
        stop_dist = abs(entry_price - initial_sl)
        risk_unit = stop_dist * shares if shares > 0.0 else notional * (stop_dist / entry_price)

    if risk_unit <= 0.0:
        # Fallback standard risk unit: 2.5% stop distance (ETP_MIN_SL_PCT)
        risk_unit = notional * 0.025

    r_multiple_gross = gross_pnl / risk_unit if risk_unit > 0.0 else 0.0
    r_multiple_net = net_pnl / risk_unit if risk_unit > 0.0 else 0.0

    asset_tier = classify_asset_tier(symbol, underlying=underlying, leverage=leverage, universe=universe)

    hold_days = int(data.get("hold_days") or 0)
    mfe_pct = float(data["mfe_pct"]) if data.get("mfe_pct") is not None else None
    mae_pct = float(data["mae_pct"]) if data.get("mae_pct") is not None else None
    post_exit_run_3pct = bool(data["post_exit_run_3pct"]) if data.get("post_exit_run_3pct") is not None else None

    return TradeRecord(
        trade_id=data.get("trade_id") or data.get("entry_order_id") or f"{symbol}_{entry_date}",
        symbol=symbol,
        underlying=underlying,
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        entry_date=entry_date,
        exit_date=exit_date,
        shares=shares,
        notional=notional,
        initial_sl=initial_sl,
        gross_pnl=gross_pnl,
        gross_return_pct=gross_return_pct,
        net_pnl=net_pnl,
        net_return_pct=net_return_pct,
        exit_reason=exit_reason,
        raw_exit_reason=raw_exit_reason,
        leverage=leverage,
        asset_tier=asset_tier,
        mfe_pct=mfe_pct,
        mae_pct=mae_pct,
        post_exit_run_3pct=post_exit_run_3pct,
        hold_days=hold_days,
        r_multiple_gross=r_multiple_gross,
        r_multiple_net=r_multiple_net,
    )


# Alias for explicit friction application API
apply_friction_and_normalize = normalize_trade


def load_trades_from_file(
    path: str,
    cost_bps: float = 45.0,
    universe: Optional[Any] = None,
) -> List[TradeRecord]:
    """Load trades from JSONL log file (handling closed trade records, backtest dumps, and paired audit logs)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trades file not found: {path}")

    raw_lines: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw_lines.append(json.loads(line))
            except Exception:
                continue

    if not raw_lines:
        return []

    # Check if lines are directly closed trades or SimulatedTrade dicts
    closed_trades: List[Dict[str, Any]] = []
    entry_records: List[Dict[str, Any]] = []

    for item in raw_lines:
        has_pnl = item.get("realized_pnl") is not None or item.get("pnl") is not None
        has_ret = item.get("return_pct") is not None or item.get("pnl_pct") is not None
        has_exit = item.get("exit_price") is not None or item.get("exit_reason") is not None

        if has_pnl or (has_ret and has_exit):
            closed_trades.append(item)
        elif item.get("executed") or item.get("decision") in ("BUY_ENTRY", "SELL_ENTRY"):
            entry_records.append(item)

    # If we already have closed trade records, normalize and return them directly
    if closed_trades:
        return [normalize_trade(t, cost_bps=cost_bps, universe=universe) for t in closed_trades]

    # If file contains only executed entry cards (like raw trades.jsonl), attempt reconciliation
    # with notifications.jsonl if present in the same directory or current workspace
    if entry_records:
        candidates_dir = os.path.dirname(os.path.abspath(path))
        notifs_path = os.path.join(candidates_dir, "notifications.jsonl")
        if not os.path.exists(notifs_path):
            notifs_path = "notifications.jsonl"

        if os.path.exists(notifs_path):
            exit_notifs = []
            try:
                with open(notifs_path, "r", encoding="utf-8") as nf:
                    for line in nf:
                        line = line.strip()
                        if line:
                            try:
                                n_item = json.loads(line)
                                if n_item.get("event_type") in (
                                    "SL", "TSL", "TP1", "TP2", "TP3",
                                    "ASYMMETRIC_SESSION_EXIT", "INVERSE_INTRADAY_FLATTEN", "SELL",
                                ) and (n_item.get("pnl") is not None or n_item.get("pnl_pct") is not None):
                                    exit_notifs.append(n_item)
                            except Exception:
                                continue
            except Exception:
                pass

            # Match executed entries with exit notifications chronologically
            matched_trades = []
            used_exits = set()
            for entry in entry_records:
                e_tkr = entry.get("trading212_ticker") or entry.get("ticker")
                e_und = entry.get("ticker")
                for i, ex in enumerate(exit_notifs):
                    if i in used_exits:
                        continue
                    x_tkr = ex.get("ticker")
                    x_und = ex.get("underlying")
                    if (e_tkr and x_tkr == e_tkr) or (e_und and x_und == e_und):
                        used_exits.add(i)
                        merged = {
                            **entry,
                            "exit_price": ex.get("price"),
                            "realized_pnl": ex.get("pnl"),
                            "return_pct": ex.get("pnl_pct"),
                            "exit_reason": ex.get("event_type"),
                            "exit_date": str(ex.get("timestamp") or "")[:10],
                        }
                        matched_trades.append(merged)
                        break

            if matched_trades:
                return [normalize_trade(t, cost_bps=cost_bps, universe=universe) for t in matched_trades]

    return []


def run_in_memory_backtest(
    symbol: str,
    period: str = "1y",
    cost_bps: float = 45.0,
    universe: Optional[Any] = None,
) -> List[TradeRecord]:
    """Execute backtest in-memory and return normalized TradeRecord list."""
    try:
        from trading_system.backtest.engine import BacktestEngine, generate_synthetic_candles
    except ImportError as e:
        raise RuntimeError(f"Backtest engine could not be imported: {e}")

    clean_sym = symbol.strip().upper()
    candles = None

    # Try CandleDatabase first (offline fast-path with explicit dates)
    try:
        from trading_system.data.candle_db import CandleDatabase
        cdb = CandleDatabase.get_instance()
        cached = cdb.get_candles(clean_sym, "1d", limit=250)
        if cached and len(cached) >= 30:
            candles = cached
    except Exception:
        pass

    # Try MarketDataProvider if CandleDatabase was empty
    if not candles:
        try:
            from trading_system.data.trading212_client import MarketDataProvider
            raw = MarketDataProvider.get_daily_candles(clean_sym, period=period)
            if raw is not None:
                if hasattr(raw, "to_dict"):
                    raw_dict = raw.to_dict(orient="records")
                    if "date" not in raw.columns and hasattr(raw, "index"):
                        for idx, rec in enumerate(raw_dict):
                            rec["date"] = str(raw.index[idx])[:10]
                    candles = raw_dict
                elif isinstance(raw, list):
                    candles = raw
                    # Ensure each candle has a date field
                    if candles and "date" not in candles[0]:
                        base_date = datetime.now(timezone.utc)
                        for idx, c in enumerate(candles):
                            dt = base_date.replace(year=base_date.year - 1)
                            c["date"] = f"2025-01-{(idx % 28) + 1:02d}"
        except Exception:
            pass

    # Fallback to deterministic synthetic candles if market data is unavailable
    if not candles:
        candles = generate_synthetic_candles(clean_sym, count=250)

    engine = BacktestEngine(
        initial_capital=10000.0,
        leverage=3,
        max_risk_per_trade=0.015,
        max_position_pct=0.20,
        target_vol=0.20,
        max_hold_days=4,
    )
    result = engine.run(symbol=clean_sym, candles=candles, strategy_filter="both")
    simulated_trades = [asdict(t) if hasattr(t, "to_dict") else asdict(t) for t in result.trades]

    return [normalize_trade(t, cost_bps=cost_bps, universe=universe) for t in simulated_trades]


# ─────────────────────────────────────────────────────────────────────────────
# 4. SQLite CandleDatabase Enrichment (MFE/MAE & Post-Exit Lookups)
# ─────────────────────────────────────────────────────────────────────────────

def enrich_trades_with_db(
    trades: List[TradeRecord],
    db_path: str,
) -> List[TradeRecord]:
    """Enrich trades with intrabar MFE/MAE and post-exit 24-48h moves using read-only SQLite."""
    if not os.path.exists(db_path):
        return trades

    conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        for t in trades:
            entry_p = t.entry_price
            if entry_p <= 0.0:
                continue

            # 1. Intrabar MFE / MAE
            if (t.mfe_pct is None or t.mae_pct is None) and t.entry_date and t.exit_date:
                # Try symbol first, then underlying
                cur.execute(
                    "SELECT high, low FROM candles WHERE symbol = ? AND date >= ? AND date <= ? ORDER BY date ASC",
                    (t.symbol, t.entry_date, t.exit_date),
                )
                rows = cur.fetchall()
                if not rows and t.underlying and t.underlying != t.symbol:
                    cur.execute(
                        "SELECT high, low FROM candles WHERE symbol = ? AND date >= ? AND date <= ? ORDER BY date ASC",
                        (t.underlying, t.entry_date, t.exit_date),
                    )
                    rows = cur.fetchall()

                if rows:
                    highs = [float(r[0]) for r in rows]
                    lows = [float(r[1]) for r in rows]
                    if t.direction == "LONG":
                        max_h = max(highs)
                        min_l = min(lows)
                        t.mfe_pct = max(0.0, ((max_h - entry_p) / entry_p) * 100.0)
                        t.mae_pct = min(0.0, ((min_l - entry_p) / entry_p) * 100.0)
                    else:
                        min_l = min(lows)
                        max_h = max(highs)
                        t.mfe_pct = max(0.0, ((entry_p - min_l) / entry_p) * 100.0)
                        t.mae_pct = min(0.0, ((entry_p - max_h) / entry_p) * 100.0)

            # 2. Post-Exit Continuation (+3%+ within next 24-48 hours / 2 daily bars)
            if t.post_exit_run_3pct is None and t.exit_date and t.exit_price > 0.0:
                if t.exit_reason in ("BREAKEVEN_RATCHET", "CHANDELIER_TRAIL"):
                    cur.execute(
                        "SELECT high, low FROM candles WHERE symbol = ? AND date > ? ORDER BY date ASC LIMIT 2",
                        (t.symbol, t.exit_date),
                    )
                    next_bars = cur.fetchall()
                    if not next_bars and t.underlying and t.underlying != t.symbol:
                        cur.execute(
                            "SELECT high, low FROM candles WHERE symbol = ? AND date > ? ORDER BY date ASC LIMIT 2",
                            (t.underlying, t.exit_date),
                        )
                        next_bars = cur.fetchall()

                    if next_bars:
                        if t.direction == "LONG":
                            t.post_exit_run_3pct = any(float(r[0]) >= t.exit_price * 1.03 for r in next_bars)
                        else:
                            t.post_exit_run_3pct = any(float(r[1]) <= t.exit_price * 0.97 for r in next_bars)
    finally:
        conn.close()

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# 5. Core Metrics & Section Calculations
# ─────────────────────────────────────────────────────────────────────────────

def calculate_headline_edge(trades: List[TradeRecord], initial_capital: float = 10000.0) -> Dict[str, Any]:
    """Compute Headline Edge (Gross vs. True Net) performance metrics."""
    n = len(trades)
    if n == 0:
        return {
            "trade_count": 0,
            "gross_win_rate_pct": 0.0,
            "net_win_rate_pct": 0.0,
            "gross_avg_win_pct": 0.0,
            "net_avg_win_pct": 0.0,
            "gross_avg_loss_pct": 0.0,
            "net_avg_loss_pct": 0.0,
            "gross_payoff_ratio": 0.0,
            "net_payoff_ratio": 0.0,
            "gross_expectancy_pct": 0.0,
            "net_expectancy_pct": 0.0,
            "gross_expectancy_r": 0.0,
            "net_expectancy_r": 0.0,
            "gross_profit_factor": 0.0,
            "net_profit_factor": 0.0,
            "gross_max_drawdown_pct": 0.0,
            "net_max_drawdown_pct": 0.0,
        }

    gross_wins = [t for t in trades if t.gross_pnl > 0.0 or (t.gross_pnl == 0.0 and t.gross_return_pct > 0.0)]
    gross_losses = [t for t in trades if t.gross_pnl < 0.0 or (t.gross_pnl == 0.0 and t.gross_return_pct < 0.0)]
    gross_zeroes = [t for t in trades if t.gross_pnl == 0.0 and t.gross_return_pct == 0.0]

    net_wins = [t for t in trades if t.net_pnl > 0.0 or (t.net_pnl == 0.0 and t.net_return_pct > 0.0)]
    net_losses = [t for t in trades if t.net_pnl < 0.0 or (t.net_pnl == 0.0 and t.net_return_pct < 0.0)]
    net_zeroes = [t for t in trades if t.net_pnl == 0.0 and t.net_return_pct == 0.0]

    gross_win_rate = (len(gross_wins) / n) * 100.0
    net_win_rate = (len(net_wins) / n) * 100.0

    gross_avg_win = statistics.mean([t.gross_return_pct for t in gross_wins]) if gross_wins else 0.0
    net_avg_win = statistics.mean([t.net_return_pct for t in net_wins]) if net_wins else 0.0

    gross_avg_loss = statistics.mean([t.gross_return_pct for t in gross_losses]) if gross_losses else 0.0
    net_avg_loss = statistics.mean([t.net_return_pct for t in net_losses]) if net_losses else 0.0

    gross_payoff = (gross_avg_win / abs(gross_avg_loss)) if abs(gross_avg_loss) > 1e-9 else (float("inf") if gross_avg_win > 0 else 0.0)
    net_payoff = (net_avg_win / abs(net_avg_loss)) if abs(net_avg_loss) > 1e-9 else (float("inf") if net_avg_win > 0 else 0.0)

    gross_expectancy_pct = statistics.mean([t.gross_return_pct for t in trades])
    net_expectancy_pct = statistics.mean([t.net_return_pct for t in trades])

    gross_expectancy_r = statistics.mean([t.r_multiple_gross for t in trades])
    net_expectancy_r = statistics.mean([t.r_multiple_net for t in trades])

    gross_profit = sum(t.gross_pnl for t in gross_wins)
    gross_loss = abs(sum(t.gross_pnl for t in gross_losses))
    gross_pf = (gross_profit / gross_loss) if gross_loss > 1e-9 else (float("inf") if gross_profit > 0 else 0.0)

    net_profit = sum(t.net_pnl for t in net_wins)
    net_loss = abs(sum(t.net_pnl for t in net_losses))
    net_pf = (net_profit / net_loss) if net_loss > 1e-9 else (float("inf") if net_profit > 0 else 0.0)

    # Cumulative peak-to-trough Drawdown calculation
    def calc_mdd(pnls: List[float]) -> float:
        eq = initial_capital
        peak = eq
        mdd = 0.0
        for p in pnls:
            eq += p
            if eq > peak:
                peak = eq
            dd = ((peak - eq) / peak) * 100.0 if peak > 0 else 0.0
            if dd > mdd:
                mdd = dd
        return mdd

    gross_mdd = calc_mdd([t.gross_pnl for t in trades])
    net_mdd = calc_mdd([t.net_pnl for t in trades])

    return {
        "trade_count": n,
        "gross_win_rate_pct": gross_win_rate,
        "net_win_rate_pct": net_win_rate,
        "gross_avg_win_pct": gross_avg_win,
        "net_avg_win_pct": net_avg_win,
        "gross_avg_loss_pct": gross_avg_loss,
        "net_avg_loss_pct": net_avg_loss,
        "gross_payoff_ratio": gross_payoff,
        "net_payoff_ratio": net_payoff,
        "gross_expectancy_pct": gross_expectancy_pct,
        "net_expectancy_pct": net_expectancy_pct,
        "gross_expectancy_r": gross_expectancy_r,
        "net_expectancy_r": net_expectancy_r,
        "gross_profit_factor": gross_pf,
        "net_profit_factor": net_pf,
        "gross_max_drawdown_pct": gross_mdd,
        "net_max_drawdown_pct": net_mdd,
    }


def calculate_exit_attribution(trades: List[TradeRecord]) -> List[Dict[str, Any]]:
    """Group trades by exit cause and compute attribution statistics."""
    n_total = len(trades)
    total_net_pnl = sum(t.net_pnl for t in trades)

    buckets: Dict[str, List[TradeRecord]] = {b: [] for b in VALID_EXIT_BUCKETS}
    for t in trades:
        b = t.exit_reason if t.exit_reason in buckets else "OTHER"
        buckets[b].append(t)

    result = []
    for b in VALID_EXIT_BUCKETS:
        b_trades = buckets[b]
        b_count = len(b_trades)
        pct_share = (b_count / n_total * 100.0) if n_total > 0 else 0.0

        b_wins = [t for t in b_trades if t.net_pnl > 0.0 or (t.net_pnl == 0.0 and t.net_return_pct > 0.0)]
        net_win_rate = (len(b_wins) / b_count * 100.0) if b_count > 0 else 0.0

        b_pnl_sum = sum(t.net_pnl for t in b_trades)
        b_avg_pnl = (b_pnl_sum / b_count) if b_count > 0 else 0.0
        pnl_share = (b_pnl_sum / total_net_pnl * 100.0) if abs(total_net_pnl) > 1e-9 else 0.0

        result.append({
            "exit_reason": b,
            "trade_count": b_count,
            "pct_share": pct_share,
            "net_win_rate_pct": net_win_rate,
            "avg_pnl": b_avg_pnl,
            "total_pnl": b_pnl_sum,
            "pnl_share_pct": pnl_share,
        })

    return result


def calculate_mfe_mae_audit(trades: List[TradeRecord]) -> Dict[str, Any]:
    """Compute MFE / MAE metrics and early-exit choke continuation statistics."""
    mfe_list = [t.mfe_pct for t in trades if t.mfe_pct is not None]
    mae_list = [t.mae_pct for t in trades if t.mae_pct is not None]

    mean_mfe = statistics.mean(mfe_list) if mfe_list else 0.0
    median_mfe = statistics.median(mfe_list) if mfe_list else 0.0
    mean_mae = statistics.mean(mae_list) if mae_list else 0.0
    median_mae = statistics.median(mae_list) if mae_list else 0.0

    edge_ratio_mean = (mean_mfe / abs(mean_mae)) if abs(mean_mae) > 1e-9 else (float("inf") if mean_mfe > 0 else 0.0)
    edge_ratio_median = (median_mfe / abs(median_mae)) if abs(median_mae) > 1e-9 else (float("inf") if median_mfe > 0 else 0.0)

    # Capture Efficiency = (Realized Net Return % / MFE %)
    efficiencies = []
    for t in trades:
        if t.mfe_pct is not None and t.mfe_pct > 0.05:
            eff = (t.net_return_pct / t.mfe_pct) * 100.0
            efficiencies.append(eff)

    mean_capture_eff = statistics.mean(efficiencies) if efficiencies else 0.0
    median_capture_eff = statistics.median(efficiencies) if efficiencies else 0.0

    # Post-Exit Choke Audit: % of BREAKEVEN_RATCHET and CHANDELIER_TRAIL exits that ran +3%+ within next 24-48h
    be_trades = [t for t in trades if t.exit_reason == "BREAKEVEN_RATCHET"]
    be_evaluated = [t for t in be_trades if t.post_exit_run_3pct is not None]
    be_choke_count = sum(1 for t in be_evaluated if t.post_exit_run_3pct)
    be_choke_pct = (be_choke_count / len(be_evaluated) * 100.0) if be_evaluated else None

    tsl_trades = [t for t in trades if t.exit_reason == "CHANDELIER_TRAIL"]
    tsl_evaluated = [t for t in tsl_trades if t.post_exit_run_3pct is not None]
    tsl_choke_count = sum(1 for t in tsl_evaluated if t.post_exit_run_3pct)
    tsl_choke_pct = (tsl_choke_count / len(tsl_evaluated) * 100.0) if tsl_evaluated else None

    return {
        "mfe_count": len(mfe_list),
        "mean_mfe_pct": mean_mfe,
        "median_mfe_pct": median_mfe,
        "mean_mae_pct": mean_mae,
        "median_mae_pct": median_mae,
        "edge_ratio_mean": edge_ratio_mean,
        "edge_ratio_median": edge_ratio_median,
        "mean_capture_efficiency_pct": mean_capture_eff,
        "median_capture_efficiency_pct": median_capture_eff,
        "be_evaluated_count": len(be_evaluated),
        "be_choke_count": be_choke_count,
        "be_choke_pct": be_choke_pct,
        "tsl_evaluated_count": len(tsl_evaluated),
        "tsl_choke_count": tsl_choke_count,
        "tsl_choke_pct": tsl_choke_pct,
    }


def calculate_segment_breakdown(trades: List[TradeRecord]) -> Dict[str, List[Dict[str, Any]]]:
    """Split Win Rate, Expectancy, and Profit Factor by Direction and Asset Tier."""
    def calc_group_stats(sub_trades: List[TradeRecord], label: str) -> Dict[str, Any]:
        count = len(sub_trades)
        if count == 0:
            return {
                "label": label,
                "trade_count": 0,
                "net_win_rate_pct": 0.0,
                "net_expectancy_pct": 0.0,
                "net_expectancy_r": 0.0,
                "net_profit_factor": 0.0,
                "total_net_pnl": 0.0,
            }

        wins = [t for t in sub_trades if t.net_pnl > 0.0 or (t.net_pnl == 0.0 and t.net_return_pct > 0.0)]
        losses = [t for t in sub_trades if t.net_pnl < 0.0 or (t.net_pnl == 0.0 and t.net_return_pct < 0.0)]

        win_rate = (len(wins) / count) * 100.0
        exp_pct = statistics.mean([t.net_return_pct for t in sub_trades])
        exp_r = statistics.mean([t.r_multiple_net for t in sub_trades])

        tot_profit = sum(t.net_pnl for t in wins)
        tot_loss = abs(sum(t.net_pnl for t in losses))
        pf = (tot_profit / tot_loss) if tot_loss > 1e-9 else (float("inf") if tot_profit > 0 else 0.0)
        tot_pnl = sum(t.net_pnl for t in sub_trades)

        return {
            "label": label,
            "trade_count": count,
            "net_win_rate_pct": win_rate,
            "net_expectancy_pct": exp_pct,
            "net_expectancy_r": exp_r,
            "net_profit_factor": pf,
            "total_net_pnl": tot_pnl,
        }

    # Direction breakdown
    direction_stats = [
        calc_group_stats([t for t in trades if t.direction == "LONG"], "LONG"),
        calc_group_stats([t for t in trades if t.direction == "SHORT"], "SHORT"),
    ]

    # Asset Tier breakdown
    tier_stats = [
        calc_group_stats([t for t in trades if t.asset_tier == tier], tier)
        for tier in VALID_ASSET_TIERS
    ]

    return {
        "direction": direction_stats,
        "asset_tier": tier_stats,
    }


def generate_bottleneck_verdict(
    headline: Dict[str, Any],
    exit_attr: List[Dict[str, Any]],
    mfe_mae: Dict[str, Any],
    segments: Dict[str, List[Dict[str, Any]]],
    cost_bps: float,
) -> List[str]:
    """Generate 2-4 data-backed bottleneck diagnostic flags."""
    flags = []

    # Flag 1: Friction Fragility
    gross_exp = headline.get("gross_expectancy_pct", 0.0)
    net_exp = headline.get("net_expectancy_pct", 0.0)
    if gross_exp > 0.0 and net_exp <= 0.0:
        flags.append(
            f"[CRITICAL] Friction Fragility: Gross expectancy of +{gross_exp:.2f}% flips negative to {net_exp:+.2f}% "
            f"under {cost_bps:.1f} bps friction, indicating system profits are entirely consumed by round-trip slippage/FX."
        )
    elif net_exp > 0.0 and gross_exp > 0.0 and (gross_exp - net_exp) / gross_exp > 0.35:
        friction_drag = ((gross_exp - net_exp) / gross_exp) * 100.0
        flags.append(
            f"[WARNING] High Friction Drag: Round-trip friction ({cost_bps:.1f} bps) consumes {friction_drag:.1f}% of gross edge "
            f"(Gross Expectancy: +{gross_exp:.2f}% vs True Net: +{net_exp:.2f}%)."
        )
    elif headline["trade_count"] > 0:
        flags.append(
            f"[PASS] Friction Resistance: True Net edge (+{net_exp:.2f}%) remains positive and resilient under {cost_bps:.1f} bps friction."
        )

    # Flag 2: Early-Exit Choke Rate (Breakeven Ratchet + 16:20 UTC Evacuation)
    total_trades = headline.get("trade_count", 0)
    be_count = next((x["trade_count"] for x in exit_attr if x["exit_reason"] == "BREAKEVEN_RATCHET"), 0)
    evac_count = next((x["trade_count"] for x in exit_attr if x["exit_reason"] == "EVACUATION_1620_UTC"), 0)
    choke_count = be_count + evac_count
    choke_pct = (choke_count / total_trades * 100.0) if total_trades > 0 else 0.0

    if choke_pct > 35.0:
        flags.append(
            f"[WARNING] Early-Exit Choke: BREAKEVEN_RATCHET + EVACUATION_1620_UTC account for {choke_pct:.1f}% of all exits "
            f"({choke_count}/{total_trades}), suggesting defensive trailing stops and session cutoffs are suffocating runners prematurely."
        )
    elif total_trades > 0:
        flags.append(
            f"[PASS] Balanced Exit Dynamics: Protective ratchets and evacuations represent {choke_pct:.1f}% of exits (within healthy <=35.0% envelope)."
        )

    # Flag 3: Post-Exit Continuation Choke Confirmation
    be_choke_pct = mfe_mae.get("be_choke_pct")
    tsl_choke_pct = mfe_mae.get("tsl_choke_pct")
    if be_choke_pct is not None and be_choke_pct >= 30.0:
        flags.append(
            f"[WARNING] Post-Exit Money Left on Table: {be_choke_pct:.1f}% of BREAKEVEN_RATCHET exits continued an additional +3%+ "
            f"within 24-48h, indicating breakeven triggers are placed inside normal intraday market noise."
        )
    elif tsl_choke_pct is not None and tsl_choke_pct >= 35.0:
        flags.append(
            f"[WARNING] Tight Trailing Choke: {tsl_choke_pct:.1f}% of CHANDELIER_TRAIL exits continued an additional +3%+ within 24-48h."
        )

    # Flag 4: Risk-Reward Payoff Asymmetry
    payoff = headline.get("net_payoff_ratio", 0.0)
    avg_win = headline.get("net_avg_win_pct", 0.0)
    avg_loss = headline.get("net_avg_loss_pct", 0.0)
    if payoff > 0.0 and payoff < 1.0 and headline["trade_count"] > 0:
        flags.append(
            f"[CAUTION] Adverse Payoff Ratio: Payoff ratio is {payoff:.2f} (< 1.00), with average loss ({avg_loss:.2f}%) "
            f"outweighing average win (+{avg_win:.2f}%). Strategy is vulnerable to mean-reversion win-rate decay."
        )

    # Flag 5: Directional Divergence
    dir_stats = segments.get("direction", [])
    long_stat = next((s for s in dir_stats if s["label"] == "LONG"), None)
    short_stat = next((s for s in dir_stats if s["label"] == "SHORT"), None)
    if long_stat and short_stat and long_stat["trade_count"] > 0 and short_stat["trade_count"] > 0:
        l_exp = long_stat["net_expectancy_pct"]
        s_exp = short_stat["net_expectancy_pct"]
        if l_exp > 0.0 and s_exp < -0.5:
            flags.append(
                f"[CAUTION] Directional Asymmetry: LONG trades delivered {l_exp:+.2f}% net expectancy while SHORT trades lost {s_exp:+.2f}%. "
                f"Consider tightening short volatility gates or reducing inverse position limits."
            )

    return flags[:4]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Terminal Table Formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_terminal_tables(
    source_desc: str,
    cost_bps: float,
    headline: Dict[str, Any],
    exit_attr: List[Dict[str, Any]],
    mfe_mae: Dict[str, Any],
    segments: Dict[str, List[Dict[str, Any]]],
    verdict_flags: List[str],
) -> str:
    """Format diagnostic report into clean, aligned terminal tables."""
    lines = []
    w = 88
    border = "=" * w
    sub_border = "-" * w

    lines.append(border)
    lines.append(f"{'STRATEGY DIAGNOSTICS & EDGE AUDIT':^{w}}")
    lines.append(border)
    lines.append(f"Source: {source_desc:<40} Round-Trip Friction: {cost_bps:.1f} bps ({cost_bps/100:.2f}%)")
    lines.append(sub_border)

    # 1. Headline Edge
    lines.append("\n[1] HEADLINE EDGE (GROSS vs. TRUE NET)")
    lines.append("+-----------------------------------+--------------------+--------------------+")
    lines.append("| Metric                            | Gross Performance  | True Net (45.0bps) |")
    lines.append("+-----------------------------------+--------------------+--------------------+")

    def pf_str(val: float) -> str:
        return f"{val:.2f}" if math.isfinite(val) else "N/A"

    def sign_str(val: float, is_r: bool = False) -> str:
        s = f"{val:+.2f}"
        return f"{s}R" if is_r else f"{s}%"

    h = headline
    rows_h = [
        ("Trade Count", f"{h['trade_count']}", f"{h['trade_count']}"),
        ("Win Rate", f"{h['gross_win_rate_pct']:.2f}%", f"{h['net_win_rate_pct']:.2f}%"),
        ("Avg Win %", f"{h['gross_avg_win_pct']:+.2f}%", f"{h['net_avg_win_pct']:+.2f}%"),
        ("Avg Loss %", f"{h['gross_avg_loss_pct']:+.2f}%", f"{h['net_avg_loss_pct']:+.2f}%"),
        ("Payoff Ratio (Win/Loss)", pf_str(h['gross_payoff_ratio']), pf_str(h['net_payoff_ratio'])),
        ("Expectancy (%)", sign_str(h['gross_expectancy_pct']), sign_str(h['net_expectancy_pct'])),
        ("Expectancy ($R$)", sign_str(h['gross_expectancy_r'], is_r=True), sign_str(h['net_expectancy_r'], is_r=True)),
        ("Profit Factor", pf_str(h['gross_profit_factor']), pf_str(h['net_profit_factor'])),
        ("Max Drawdown", f"{h['gross_max_drawdown_pct']:.2f}%", f"{h['net_max_drawdown_pct']:.2f}%"),
    ]
    for lbl, g_val, n_val in rows_h:
        lines.append(f"| {lbl:<33} | {g_val:>18} | {n_val:>18} |")
    lines.append("+-----------------------------------+--------------------+--------------------+")

    # 2. Exit-Reason Attribution
    lines.append("\n[2] EXIT-REASON ATTRIBUTION")
    lines.append("+----------------------+-------+---------+--------------+--------------+---------------+--------------+")
    lines.append("| Exit Reason          | Count | % Share | Net Win Rate | Avg PnL ($)  | Total PnL ($) | PnL Share %  |")
    lines.append("+----------------------+-------+---------+--------------+--------------+---------------+--------------+")
    for row in exit_attr:
        lines.append(
            f"| {row['exit_reason']:<20} | {row['trade_count']:>5} | {row['pct_share']:>6.1f}% | "
            f"{row['net_win_rate_pct']:>11.1f}% | {row['avg_pnl']:>+11.2f} | "
            f"{row['total_pnl']:>+12.2f} | {row['pnl_share_pct']:>11.1f}% |"
        )
    lines.append("+----------------------+-------+---------+--------------+--------------+---------------+--------------+")

    # 3. MFE / MAE & Early-Exit Choke Audit
    lines.append("\n[3] MFE / MAE & EARLY-EXIT CHOKE AUDIT")
    lines.append("+-----------------------------------+--------------------+--------------------+")
    lines.append("| Metric                            | Mean Value         | Median Value       |")
    lines.append("+-----------------------------------+--------------------+--------------------+")
    lines.append(f"| Maximum Favorable Excursion (MFE) | {mfe_mae['mean_mfe_pct']:>+18.2f}% | {mfe_mae['median_mfe_pct']:>+18.2f}% |")
    lines.append(f"| Maximum Adverse Excursion (MAE)   | {mfe_mae['mean_mae_pct']:>+18.2f}% | {mfe_mae['median_mae_pct']:>+18.2f}% |")
    lines.append(f"| Edge Ratio (MFE / abs(MAE))       | {pf_str(mfe_mae['edge_ratio_mean']):>19} | {pf_str(mfe_mae['edge_ratio_median']):>19} |")
    lines.append(f"| Capture Efficiency (PnL / MFE)    | {mfe_mae['mean_capture_efficiency_pct']:>18.1f}% | {mfe_mae['median_capture_efficiency_pct']:>18.1f}% |")
    lines.append("+-----------------------------------+--------------------+--------------------+")

    be_pct_str = f"{mfe_mae['be_choke_pct']:.1f}% ({mfe_mae['be_choke_count']}/{mfe_mae['be_evaluated_count']})" if mfe_mae['be_choke_pct'] is not None else "N/A (No candle DB)"
    tsl_pct_str = f"{mfe_mae['tsl_choke_pct']:.1f}% ({mfe_mae['tsl_choke_count']}/{mfe_mae['tsl_evaluated_count']})" if mfe_mae['tsl_choke_pct'] is not None else "N/A (No candle DB)"
    lines.append(f"  * Post-Exit Continuation (+3%+ within 24-48h):")
    lines.append(f"    - BREAKEVEN_RATCHET exits that ran +3%+:  {be_pct_str}")
    lines.append(f"    - CHANDELIER_TRAIL exits that ran +3%+:   {tsl_pct_str}")

    # 4. Segment Breakdown
    lines.append("\n[4] SEGMENT BREAKDOWN (DIRECTION & ASSET TIER)")
    lines.append("+-------------------+-------+--------------+------------------+------------------+---------------+---------------+")
    lines.append("| Segment           | Count | Net Win Rate | Net Exp (%)      | Net Exp ($R$)    | Profit Factor | Total PnL ($) |")
    lines.append("+-------------------+-------+--------------+------------------+------------------+---------------+---------------+")

    lines.append("| -- Direction --   |       |              |                  |                  |               |               |")
    for row in segments["direction"]:
        lines.append(
            f"| {row['label']:<17} | {row['trade_count']:>5} | {row['net_win_rate_pct']:>11.1f}% | "
            f"{sign_str(row['net_expectancy_pct']):>16} | {sign_str(row['net_expectancy_r'], is_r=True):>16} | "
            f"{pf_str(row['net_profit_factor']):>13} | {row['total_net_pnl']:>+13.2f} |"
        )

    lines.append("| -- Asset Tier --  |       |              |                  |                  |               |               |")
    for row in segments["asset_tier"]:
        lines.append(
            f"| {row['label']:<17} | {row['trade_count']:>5} | {row['net_win_rate_pct']:>11.1f}% | "
            f"{sign_str(row['net_expectancy_pct']):>16} | {sign_str(row['net_expectancy_r'], is_r=True):>16} | "
            f"{pf_str(row['net_profit_factor']):>13} | {row['total_net_pnl']:>+13.2f} |"
        )
    lines.append("+-------------------+-------+--------------+------------------+------------------+---------------+---------------+")

    # 5. Automated Bottleneck Verdict
    lines.append("\n[5] AUTOMATED BOTTLENECK VERDICT")
    for f in verdict_flags:
        lines.append(f"  • {f}")

    lines.append(border)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone, Read-Only Strategy Diagnostics & Performance Audit Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trades",
        type=str,
        default="trades.jsonl",
        help="Path to live/paper trade execution log (JSONL)",
    )
    parser.add_argument(
        "--backtest-symbol",
        type=str,
        default=None,
        help="Run existing backtest in-memory for SYM (e.g., QQQ) and pipe trades directly into analyzer",
    )
    parser.add_argument(
        "--backtest-period",
        type=str,
        default="1y",
        help="Historical period for in-memory backtest (e.g. 1y, 6m)",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=45.0,
        help="Round-trip LSE ETP friction (spread + FX) in basis points",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/candles.db",
        help="Path to CandleDatabase SQLite file for intrabar MFE/MAE and post-exit move enrichment",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Export structured diagnostics report to JSON file",
    )
    return parser


def run_diagnostics(
    trades: List[TradeRecord],
    cost_bps: float = 45.0,
    db_path: Optional[str] = None,
    source_desc: str = "Execution Log",
) -> Dict[str, Any]:
    """Run full diagnostics pipeline on a list of TradeRecords."""
    if db_path and os.path.exists(db_path):
        enrich_trades_with_db(trades, db_path)

    headline = calculate_headline_edge(trades)
    exit_attr = calculate_exit_attribution(trades)
    mfe_mae = calculate_mfe_mae_audit(trades)
    segments = calculate_segment_breakdown(trades)
    verdict = generate_bottleneck_verdict(headline, exit_attr, mfe_mae, segments, cost_bps=cost_bps)

    return {
        "source": source_desc,
        "cost_bps": cost_bps,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "headline_edge": headline,
        "exit_reason_attribution": exit_attr,
        "mfe_mae_audit": mfe_mae,
        "segment_breakdown": segments,
        "bottleneck_verdict": verdict,
        "trades": [t.to_dict() for t in trades],
    }


def main(args: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    opts = parser.parse_args(args)

    universe = None
    try:
        from trading_system.data.leveraged_universe import LeveragedUniverse
        universe = LeveragedUniverse()
    except Exception:
        pass

    trades: List[TradeRecord] = []
    source_desc = ""

    if opts.backtest_symbol:
        source_desc = f"In-Memory Backtest ({opts.backtest_symbol.upper()}, {opts.backtest_period})"
        try:
            trades = run_in_memory_backtest(
                symbol=opts.backtest_symbol,
                period=opts.backtest_period,
                cost_bps=opts.cost_bps,
                universe=universe,
            )
        except Exception as exc:
            print(f"Error running in-memory backtest: {exc}", file=sys.stderr)
            return 1
    else:
        source_desc = f"File: {opts.trades}"
        try:
            trades = load_trades_from_file(opts.trades, cost_bps=opts.cost_bps, universe=universe)
        except FileNotFoundError:
            print(f"Warning: Trades log not found at '{opts.trades}'. Operating with 0 trades.", file=sys.stderr)
            trades = []
        except Exception as exc:
            print(f"Error loading trades from '{opts.trades}': {exc}", file=sys.stderr)
            return 1

    db_path = opts.db if opts.db and os.path.exists(opts.db) else None
    report_data = run_diagnostics(
        trades=trades,
        cost_bps=opts.cost_bps,
        db_path=db_path,
        source_desc=source_desc,
    )

    output_tables = format_terminal_tables(
        source_desc=source_desc,
        cost_bps=opts.cost_bps,
        headline=report_data["headline_edge"],
        exit_attr=report_data["exit_reason_attribution"],
        mfe_mae=report_data["mfe_mae_audit"],
        segments=report_data["segment_breakdown"],
        verdict_flags=report_data["bottleneck_verdict"],
    )
    print(output_tables)

    if opts.json:
        try:
            with open(opts.json, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
            print(f"\n[OK] Structured report successfully exported to '{opts.json}'.")
        except Exception as exc:
            print(f"Error exporting JSON report to '{opts.json}': {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
