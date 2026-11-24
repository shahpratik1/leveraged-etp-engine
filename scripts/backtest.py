#!/usr/bin/env python3
"""Fast CLI Backtest and Validation Runner for Strategy Benchmarking.

Evaluates registered trading strategies on historical or synthetic OHLCV data.
Reports standardized quantitative performance metrics:
- Sharpe Ratio (Annualized)
- Win Rate (%)
- Max Drawdown (%)
- Total Return (%)
- Profit Factor & Trade Count

Usage:
    ./venv/bin/python3 scripts/backtest.py --strategy vwap_trend --symbol QQQ
    ./venv/bin/python3 scripts/backtest.py --strategy ml_strategy --symbol SPY --bars 1000
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# Add workspace root to sys.path to ensure src imports succeed
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.core.interfaces import BaseStrategy, OHLCVBar, Signal, SignalAction
from src.strategies.registry import StrategyRegistry

logger = logging.getLogger("backtest")


# ─────────────────────────────────────────────────────────────────────────────
# Domain Models & Metrics Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """Record of an executed and closed trade."""
    symbol: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float
    bars_held: int


@dataclass
class BacktestReport:
    """Comprehensive performance statistics returned by the backtest engine."""
    strategy_name: str
    symbol: str
    total_bars: int
    initial_capital: float
    final_equity: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: float
    equity_curve: List[float] = field(default_factory=list, repr=False)
    trades: List[TradeRecord] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to serializable dictionary."""
        d = asdict(self)
        d["equity_curve"] = [round(v, 2) for v in self.equity_curve]
        d["trades"] = [
            {
                "symbol": t.symbol,
                "side": t.side,
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "entry_price": round(t.entry_price, 4),
                "exit_price": round(t.exit_price, 4),
                "quantity": round(t.quantity, 4),
                "pnl": round(t.pnl, 2),
                "return_pct": round(t.return_pct, 2),
                "bars_held": t.bars_held,
            }
            for t in self.trades
        ]
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic & Historical Data Ingestion
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(
    symbol: str = "QQQ",
    bars_count: int = 500,
    base_price: float = 450.0,
    annual_vol: float = 0.22,
    seed: int = 42,
    timeframe_minutes: int = 1,
) -> List[OHLCVBar]:
    """Generate realistic OHLCV bars using geometric Brownian motion with intraday dynamics."""
    rng = np.random.default_rng(seed)
    bars: List[OHLCVBar] = []

    dt_per_bar = timeframe_minutes / (252.0 * 390.0)
    drift = 0.08 * dt_per_bar
    vol_step = annual_vol * math.sqrt(dt_per_bar)

    current_price = base_price
    start_time = datetime(2026, 1, 5, 9, 30, tzinfo=timezone.utc)

    # Intraday volume curve: higher at open/close, lower midday
    vol_mean = 50000.0

    cumulative_pv = 0.0
    cumulative_v = 0.0

    for i in range(bars_count):
        bar_time = start_time + timedelta(minutes=i * timeframe_minutes)

        # Log return step
        ret = drift + rng.normal(0.0, vol_step)
        next_price = max(0.5, current_price * math.exp(ret))

        # Intraday spread
        intraday_range = rng.exponential(scale=vol_step * current_price * 1.5)
        high = max(current_price, next_price) + (intraday_range * rng.uniform(0.2, 0.8))
        low = min(current_price, next_price) - (intraday_range * rng.uniform(0.2, 0.8))
        low = max(0.1, low)
        open_price = current_price
        close_price = next_price

        # Intraday U-shaped volume curve
        minute_of_day = (i % 390)
        u_factor = 1.0 + 1.5 * ((minute_of_day - 195) / 195.0) ** 2
        volume = max(100.0, rng.normal(vol_mean * u_factor, vol_mean * 0.25))

        # Reset session VWAP every 390 bars (standard trading day)
        if minute_of_day == 0:
            cumulative_pv = 0.0
            cumulative_v = 0.0

        bar_typical = (high + low + close_price) / 3.0
        cumulative_pv += bar_typical * volume
        cumulative_v += volume
        session_vwap = cumulative_pv / cumulative_v

        bar = OHLCVBar(
            symbol=symbol,
            open=round(open_price, 4),
            high=round(high, 4),
            low=round(low, 4),
            close=round(close_price, 4),
            volume=round(volume, 1),
            timestamp=bar_time,
            timeframe=f"{timeframe_minutes}m",
            vwap=round(session_vwap, 4),
        )
        bars.append(bar)
        current_price = next_price

    return bars


def load_csv_data(filepath: str, symbol: str) -> List[OHLCVBar]:
    """Load historical OHLCV data from CSV file."""
    import csv

    bars: List[OHLCVBar] = []
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row.get("timestamp") or row.get("Date") or row.get("time") or ""
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception:
                ts = datetime.now(timezone.utc)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            bar = OHLCVBar(
                symbol=symbol,
                open=float(row.get("open") or row.get("Open") or 0.0),
                high=float(row.get("high") or row.get("High") or 0.0),
                low=float(row.get("low") or row.get("Low") or 0.0),
                close=float(row.get("close") or row.get("Close") or 0.0),
                volume=float(row.get("volume") or row.get("Volume") or 0.0),
                timestamp=ts,
                vwap=float(row.get("vwap")) if row.get("vwap") else None,
            )
            bars.append(bar)
    return bars


# ─────────────────────────────────────────────────────────────────────────────
# Backtest Engine
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    strategy: Union[str, BaseStrategy],
    symbol: str = "QQQ",
    data: Optional[Sequence[OHLCVBar]] = None,
    bars_count: int = 500,
    initial_capital: float = 100000.0,
    slippage_bps: float = 5.0,
    commission_per_trade: float = 0.0,
    strategy_params: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> BacktestReport:
    """Execute backtest simulation on bars and return quantitative performance report.

    Args:
        strategy: Strategy name or instantiated BaseStrategy.
        symbol: Market ticker identifier.
        data: Optional pre-loaded sequence of OHLCVBar objects.
        bars_count: Number of synthetic bars if data is not provided.
        initial_capital: Starting cash balance in USD.
        slippage_bps: Execution slippage penalty in basis points (5 bps = 0.05%).
        commission_per_trade: Fixed fee per entry/exit in USD.
        strategy_params: Optional parameter dictionary for strategy instantiation.
        seed: Random seed for synthetic data generation.

    Returns:
        BacktestReport with Sharpe, Win Rate, Max Drawdown, and Total Return.
    """
    # 1. Resolve strategy instance
    strat_instance: BaseStrategy
    if isinstance(strategy, BaseStrategy):
        strat_instance = strategy
    else:
        params = dict(strategy_params or {})
        strat_instance = StrategyRegistry.create(strategy, **params)

    strat_instance.reset()

    # 2. Ingest or generate market bars
    if data is None or len(data) == 0:
        bars = generate_synthetic_data(symbol=symbol, bars_count=bars_count, seed=seed)
    else:
        bars = list(data)

    slippage_factor = slippage_bps / 10000.0

    # 3. Simulation state tracking
    cash = float(initial_capital)
    position = 0.0  # units (positive = long, negative = short)
    entry_price = 0.0
    entry_time: Optional[datetime] = None
    entry_bar_idx = 0

    equity_curve: List[float] = []
    trades: List[TradeRecord] = []

    # Iterate sequentially through bars
    for i, bar in enumerate(bars):
        current_price = bar.close

        # Strategy evaluation
        signal = strat_instance.on_data(bar)

        # Execution logic based on signal action
        if signal.action == SignalAction.BUY:
            if position < 0:
                # Cover existing short position
                exit_price = current_price * (1.0 + slippage_factor)
                trade_qty = abs(position)
                pnl = (entry_price - exit_price) * trade_qty - (2.0 * commission_per_trade)
                ret_pct = ((entry_price - exit_price) / entry_price) * 100.0
                cash += (2.0 * entry_price - exit_price) * trade_qty - commission_per_trade
                trades.append(
                    TradeRecord(
                        symbol=symbol,
                        side="SHORT",
                        entry_time=entry_time or bar.timestamp,
                        exit_time=bar.timestamp,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=trade_qty,
                        pnl=pnl,
                        return_pct=ret_pct,
                        bars_held=i - entry_bar_idx,
                    )
                )
                position = 0.0

            if position == 0:
                # Enter Long
                exec_price = current_price * (1.0 + slippage_factor)
                invest_cash = max(0.0, cash * 0.95 - commission_per_trade)
                units = invest_cash / exec_price if exec_price > 0 else 0.0
                if units > 0.001:
                    position = units
                    entry_price = exec_price
                    entry_time = bar.timestamp
                    entry_bar_idx = i
                    cash -= (units * exec_price) + commission_per_trade

        elif signal.action == SignalAction.SELL:
            if position > 0:
                # Close existing long position
                exit_price = current_price * (1.0 - slippage_factor)
                trade_qty = position
                pnl = (exit_price - entry_price) * trade_qty - (2.0 * commission_per_trade)
                ret_pct = ((exit_price - entry_price) / entry_price) * 100.0
                cash += (trade_qty * exit_price) - commission_per_trade
                trades.append(
                    TradeRecord(
                        symbol=symbol,
                        side="LONG",
                        entry_time=entry_time or bar.timestamp,
                        exit_time=bar.timestamp,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=trade_qty,
                        pnl=pnl,
                        return_pct=ret_pct,
                        bars_held=i - entry_bar_idx,
                    )
                )
                position = 0.0

            can_short = (
                strat_instance.name != "vwap_trend"
                or (hasattr(strat_instance, "is_broad_index") and getattr(strat_instance, "is_broad_index")(symbol))
            )
            if position == 0 and can_short:
                # Enter Short if authorized
                exec_price = current_price * (1.0 - slippage_factor)
                invest_cash = max(0.0, cash * 0.95 - commission_per_trade)
                units = invest_cash / exec_price if exec_price > 0 else 0.0
                if units > 0.001:
                    position = -units
                    entry_price = exec_price
                    entry_time = bar.timestamp
                    entry_bar_idx = i
                    cash -= (units * exec_price) + commission_per_trade

        # Mark-to-market equity calculation at bar close
        if position > 0:
            current_equity = cash + (position * current_price)
        elif position < 0:
            current_equity = cash + (2.0 * entry_price - current_price) * abs(position)
        else:
            current_equity = cash

        equity_curve.append(current_equity)

    # Close any open position at the final bar close
    if position != 0.0 and len(bars) > 0:
        final_bar = bars[-1]
        if position > 0:
            exit_price = final_bar.close * (1.0 - slippage_factor)
            trade_qty = position
            pnl = (exit_price - entry_price) * trade_qty - (2.0 * commission_per_trade)
            ret_pct = ((exit_price - entry_price) / entry_price) * 100.0
            cash += trade_qty * exit_price - commission_per_trade
            trades.append(
                TradeRecord(
                    symbol=symbol,
                    side="LONG",
                    entry_time=entry_time or final_bar.timestamp,
                    exit_time=final_bar.timestamp,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=trade_qty,
                    pnl=pnl,
                    return_pct=ret_pct,
                    bars_held=len(bars) - 1 - entry_bar_idx,
                )
            )
        else:
            exit_price = final_bar.close * (1.0 + slippage_factor)
            trade_qty = abs(position)
            pnl = (entry_price - exit_price) * trade_qty - (2.0 * commission_per_trade)
            ret_pct = ((entry_price - exit_price) / entry_price) * 100.0
            cash += (2.0 * entry_price - exit_price) * trade_qty - commission_per_trade
            trades.append(
                TradeRecord(
                    symbol=symbol,
                    side="SHORT",
                    entry_time=entry_time or final_bar.timestamp,
                    exit_time=final_bar.timestamp,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=trade_qty,
                    pnl=pnl,
                    return_pct=ret_pct,
                    bars_held=len(bars) - 1 - entry_bar_idx,
                )
            )
        position = 0.0
        equity_curve[-1] = cash

    final_equity = cash

    # 4. Metric Calculations
    # Total Return (%)
    total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100.0

    # Max Drawdown (%)
    peak = initial_capital
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = ((peak - eq) / peak) * 100.0 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # Win Rate (%) and Profit Factor
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.pnl > 0.0)
    losing_trades = sum(1 for t in trades if t.pnl < 0.0)
    win_rate_pct = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    gross_profit = sum(t.pnl for t in trades if t.pnl > 0.0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0.0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)

    # Annualized Sharpe Ratio of bar returns
    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    if len(returns) > 1 and np.std(returns) > 1e-9:
        # Assuming 1m bars in standard US session: 252 days * 390 min = 98,280 bars/year
        annualization_factor = math.sqrt(252.0 * 390.0)
        sharpe_ratio = float((np.mean(returns) / np.std(returns)) * annualization_factor)
    else:
        sharpe_ratio = 0.0

    return BacktestReport(
        strategy_name=strat_instance.name,
        symbol=symbol,
        total_bars=len(bars),
        initial_capital=initial_capital,
        final_equity=round(final_equity, 2),
        total_return_pct=round(total_return_pct, 2),
        sharpe_ratio=round(sharpe_ratio, 2),
        max_drawdown_pct=round(max_dd, 2),
        win_rate_pct=round(win_rate_pct, 2),
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        profit_factor=round(profit_factor, 2),
        equity_curve=equity_curve,
        trades=trades,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI Formatter & Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def format_terminal_report(report: BacktestReport) -> str:
    """Format the backtest report into a clean, human-readable terminal dashboard."""
    return f"""
================================================================================
                    STRATEGY BACKTEST BENCHMARK REPORT
================================================================================
Strategy Identifier:    {report.strategy_name}
Market Symbol:          {report.symbol}
Dataset Size:           {report.total_bars:,} bars
Initial Capital:        ${report.initial_capital:,.2f}
Final Portfolio Equity: ${report.final_equity:,.2f}
--------------------------------------------------------------------------------
Key Performance Metrics:
  * Total Return:       {report.total_return_pct:+.2f}%
  * Sharpe Ratio:       {report.sharpe_ratio:.2f}
  * Max Drawdown:       {report.max_drawdown_pct:.2f}%
  * Win Rate:           {report.win_rate_pct:.2f}% ({report.winning_trades} wins / {report.total_trades} trades)
  * Profit Factor:      {report.profit_factor:.2f}
  * Losses:             {report.losing_trades}
================================================================================
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fast backtest benchmarking on any registered trading strategy."
    )
    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        required=True,
        help="Registered strategy name (e.g. 'vwap_trend', 'ml_strategy')",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="QQQ",
        help="Market symbol ticker (default: QQQ)",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=None,
        help="Optional path to CSV file containing historical OHLCV data",
    )
    parser.add_argument(
        "--bars",
        "-b",
        type=int,
        default=500,
        help="Number of synthetic bars to simulate if no CSV is passed (default: 500)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial capital in USD (default: 100,000.0)",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="Execution slippage in basis points (default: 5.0 bps)",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.0,
        help="Commission per trade in USD (default: 0.0)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="JSON string of parameter overrides for the strategy",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON format results instead of text table",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic synthetic data generation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Parse optional parameters
    params: Dict[str, Any] = {}
    if args.params:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError as err:
            logger.error("Failed to parse --params JSON: %s", err)
            print(f"Error: Invalid JSON passed to --params: {err}", file=sys.stderr)
            return 1

    # Load data
    data_bars: Optional[List[OHLCVBar]] = None
    if args.data and os.path.exists(args.data):
        try:
            data_bars = load_csv_data(args.data, args.symbol)
        except Exception as err:
            print(f"Error loading CSV data from {args.data}: {err}", file=sys.stderr)
            return 1

    try:
        report = run_backtest(
            strategy=args.strategy,
            symbol=args.symbol,
            data=data_bars,
            bars_count=args.bars,
            initial_capital=args.capital,
            slippage_bps=args.slippage_bps,
            commission_per_trade=args.commission,
            strategy_params=params,
            seed=args.seed,
        )
    except Exception as err:
        print(f"Backtest execution failed: {err}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_terminal_report(report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
