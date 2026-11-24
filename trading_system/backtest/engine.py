"""Historical Backtesting Engine and Benchmark Simulator.

Replays historical daily OHLCV market data through Connors 3-RSI Mean Reversion
and ADX Momentum Breakout strategies, faithfully modeling 3x leveraged ETP execution,
volatility-targeted position sizing, and 3-tier profit-taking lifecycle exits.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

try:
    from trading_system.config import settings
except ImportError:
    settings = None

if TYPE_CHECKING:
    from trading_system.config_bus import ConfigSnapshot

logger = logging.getLogger(__name__)


@dataclass
class SimulatedTrade:
    """Represents a closed or open simulated trade during backtesting."""
    trade_id: int
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    strategy: str   # 'directional_trend'
    entry_date: str
    entry_price: float
    exit_date: Optional[str] = None
    exit_price: Optional[str] = None
    shares: float = 0.0
    notional: float = 0.0
    initial_sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    realized_pnl: float = 0.0
    return_pct: float = 0.0
    hold_days: int = 0
    exit_reason: str = ""  # 'SL', 'TP1_SL_BE', 'TP2_TSL', 'TP3', 'MAX_HOLD', 'END_OF_DATA'
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False


@dataclass
class BacktestResult:
    """Comprehensive performance statistics from a backtest run."""
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    ending_capital: float
    total_return_pct: float
    benchmark_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_trade_pnl: float
    avg_hold_days: float
    trades: List[SimulatedTrade] = field(default_factory=list)
    time_in_cash_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["trades"] = [asdict(t) for t in self.trades]
        return d

    def meets_benchmark_targets(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify quantitative statistical edge against benchmark requirements:
        - Sharpe ratio >= 1.5
        - Profit factor >= 1.75
        - Win rate >= 60.0%
        """
        sharpe = float(self.sharpe_ratio or 0.0)
        pf = float(self.profit_factor or 0.0)
        win_rate = float(self.win_rate_pct or 0.0)

        sharpe_ok = sharpe >= 1.5
        pf_ok = pf >= 1.75
        win_rate_ok = win_rate >= 60.0

        passed = bool(sharpe_ok and pf_ok and win_rate_ok)
        details = {
            "sharpe_ratio": sharpe,
            "sharpe_ok": sharpe_ok,
            "profit_factor": pf,
            "profit_factor_ok": pf_ok,
            "win_rate_pct": win_rate,
            "win_rate_ok": win_rate_ok,
            "passed": passed,
        }
        return passed, details


class BacktestEngine:
    """Event-driven simulator for 3x leveraged trading strategies."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: int = 3,
        max_risk_per_trade: float = 0.015,
        max_position_pct: float = 0.20,
        target_vol: float = 0.20,
        max_hold_days: int = 4,
        tp1_r: float = 1.5,
        tp2_r: float = 2.5,
        tp3_r: float = 3.5,
        multi_regime: bool = True,
        pure_tsl_mode: Optional[bool] = None,
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_pct = max_position_pct
        self.target_vol = target_vol
        self.max_hold_days = max_hold_days
        self.tp1_r = tp1_r
        self.tp2_r = tp2_r
        self.tp3_r = tp3_r
        self.multi_regime = multi_regime
        if pure_tsl_mode is None:
            self.pure_tsl_mode = getattr(settings, "PURE_TSL_MODE", True) if settings else True
        else:
            self.pure_tsl_mode = pure_tsl_mode

    def run_universe(self, symbols: List[str], limit_candles: int = 250) -> Dict[str, Any]:
        """Batch evaluate performance across multiple symbols in universe."""
        results = {}
        total_trades = 0
        winning_trades = 0
        for sym in symbols:
            try:
                res = self.run(sym)
                results[sym] = res
                total_trades += res.total_trades
                winning_trades += res.winning_trades
            except Exception:
                continue
        overall_win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0
        return {
            "symbols_evaluated": len(results),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate_pct": overall_win_rate,
            "results": results,
            "initial_capital": self.initial_capital,
        }

    @staticmethod
    def _calculate_sma(values: List[float], period: int) -> List[Optional[float]]:
        res: List[Optional[float]] = []
        for i in range(len(values)):
            if i + 1 < period:
                res.append(sum(values[: i + 1]) / (i + 1))
            else:
                res.append(sum(values[i + 1 - period : i + 1]) / period)
        return res

    @staticmethod
    def _calculate_rsi(closes: List[float], period: int) -> List[Optional[float]]:
        if len(closes) < period + 1:
            return [None] * len(closes)

        gains = []
        losses = []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i - 1]
            gains.append(max(0.0, diff))
            losses.append(max(0.0, -diff))

        rsi: List[Optional[float]] = [None] * (period)
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        if avg_loss == 0:
            rsi.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi.append(100.0 - (100.0 / (1.0 + rs)))

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                rsi.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi.append(100.0 - (100.0 / (1.0 + rs)))

        return rsi

    @staticmethod
    def _calculate_atr(
        highs: List[float], lows: List[float], closes: List[float], period: int = 14
    ) -> List[Optional[float]]:
        if len(closes) < 2:
            return [None] * len(closes)

        tr = [highs[0] - lows[0]]
        for i in range(1, len(closes)):
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            ))

        atr: List[Optional[float]] = [None] * (period - 1)
        if len(tr) < period:
            return [None] * len(closes)

        curr_atr = sum(tr[:period]) / period
        atr.append(curr_atr)

        for i in range(period, len(tr)):
            curr_atr = (curr_atr * (period - 1) + tr[i]) / period
            atr.append(curr_atr)

        return atr

    @staticmethod
    def _calculate_realized_vol(closes: List[float], window: int = 20) -> List[Optional[float]]:
        vol: List[Optional[float]] = [None] * len(closes)
        for i in range(window, len(closes)):
            sub = closes[i - window : i + 1]
            rets = [(sub[j] - sub[j - 1]) / sub[j - 1] for j in range(1, len(sub))]
            mean_r = sum(rets) / len(rets)
            var_r = sum((r - mean_r) ** 2 for r in rets) / max(1, len(rets) - 1)
            ann_vol = math.sqrt(var_r) * math.sqrt(252)
            vol[i] = ann_vol
        return vol

    @staticmethod
    def _calculate_rolling_nixtla_forecast(closes: List[float], horizon: int = 5) -> List[Optional[float]]:
        """Compute rolling 5-day forecasted expected return percentage (Nixtla keyless model).

        For each bar i, forecasts 5 days forward strictly using closes[:i+1] (zero lookahead).
        Returns list of forecasted expected return percentages.
        """
        forecast_returns: List[Optional[float]] = [None] * len(closes)
        for i in range(10, len(closes)):
            sub_closes = closes[max(0, i - 40) : i + 1]
            last_p = sub_closes[-1]
            level = sub_closes[0]
            trend = (sub_closes[min(4, len(sub_closes) - 1)] - sub_closes[0]) / max(1, min(4, len(sub_closes) - 1))
            alpha = 0.35
            beta = 0.15
            for price in sub_closes:
                prev_level = level
                level = alpha * price + (1 - alpha) * (level + trend)
                trend = beta * (level - prev_level) + (1 - beta) * trend

            damping = 0.85
            curr = last_p
            for h in range(horizon):
                curr = curr + (trend * (damping ** (h + 1)))
            exp_ret = ((curr - last_p) / last_p) * 100.0 if last_p > 0 else 0.0
            forecast_returns[i] = round(exp_ret, 2)
        return forecast_returns

    @staticmethod
    def _calculate_sentiment_proxy(closes: List[float], period: int = 14) -> List[Optional[float]]:
        """Compute normalized sentiment/momentum proxy (0 to 100) reflecting keyless web score.

        Combines rolling short-to-medium price positioning and rate-of-change.
        """
        sentiment: List[Optional[float]] = [None] * len(closes)
        for i in range(period, len(closes)):
            sub = closes[i - period : i + 1]
            min_p = min(sub)
            max_p = max(sub)
            if max_p > min_p:
                stoch = (closes[i] - min_p) / (max_p - min_p) * 100.0
                roc = ((closes[i] - sub[0]) / sub[0]) * 100.0
                # Weighted score centered at 50
                score = round(max(0.0, min(100.0, (stoch * 0.7) + (roc * 2.0) + 15.0)), 1)
                sentiment[i] = score
            else:
                sentiment[i] = 50.0
        return sentiment

    @staticmethod
    def _calculate_ema(values: List[float], period: int) -> List[Optional[float]]:
        """Compute exponential moving average series aligned with *values*."""
        if not values:
            return []
        alpha = 2.0 / (float(period) + 1.0)
        res: List[Optional[float]] = []
        ema = float(values[0])
        res.append(ema)
        for v in values[1:]:
            ema = alpha * float(v) + (1.0 - alpha) * ema
            res.append(ema)
        return res

    @staticmethod
    def _calculate_wilder_adx(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14,
    ) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
        """Compute Wilder-smoothed ADX(14), +DI, and -DI series with zero lookahead."""
        n = len(closes)
        if n < period + 1 or len(highs) != n or len(lows) != n:
            return [None] * n, [None] * n, [None] * n

        alpha = 1.0 / float(period)
        adx_series: List[Optional[float]] = [None] * n
        pdi_series: List[Optional[float]] = [None] * n
        mdi_series: List[Optional[float]] = [None] * n

        tr_warmup: List[float] = []
        pdm_warmup: List[float] = []
        mdm_warmup: List[float] = []
        smooth_tr = 0.0
        smooth_pdm = 0.0
        smooth_mdm = 0.0
        dx_warmup: List[float] = []
        smooth_dx: Optional[float] = None

        for i in range(1, n):
            hi, lo = float(highs[i]), float(lows[i])
            prev_hi, prev_lo, prev_c = float(highs[i - 1]), float(lows[i - 1]), float(closes[i - 1])

            tr = max(hi - lo, abs(hi - prev_c), abs(lo - prev_c))
            up_move = hi - prev_hi
            down_move = prev_lo - lo
            pdm = up_move if (up_move > down_move and up_move > 0.0) else 0.0
            mdm = down_move if (down_move > up_move and down_move > 0.0) else 0.0

            if i <= period:
                tr_warmup.append(tr)
                pdm_warmup.append(pdm)
                mdm_warmup.append(mdm)
                if i < period:
                    continue
                smooth_tr = sum(tr_warmup) / float(period)
                smooth_pdm = sum(pdm_warmup) / float(period)
                smooth_mdm = sum(mdm_warmup) / float(period)
            else:
                smooth_tr = alpha * tr + (1.0 - alpha) * smooth_tr
                smooth_pdm = alpha * pdm + (1.0 - alpha) * smooth_pdm
                smooth_mdm = alpha * mdm + (1.0 - alpha) * smooth_mdm

            plus_di = 100.0 * (smooth_pdm / (smooth_tr + 1e-9))
            minus_di = 100.0 * (smooth_mdm / (smooth_tr + 1e-9))
            dx = 100.0 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)

            if smooth_dx is None:
                dx_warmup.append(dx)
                if len(dx_warmup) >= period:
                    smooth_dx = sum(dx_warmup) / float(period)
                    adx_val = smooth_dx
                else:
                    adx_val = sum(dx_warmup) / float(len(dx_warmup))
            else:
                smooth_dx = alpha * dx + (1.0 - alpha) * smooth_dx
                adx_val = smooth_dx

            pdi_series[i] = round(plus_di, 4)
            mdi_series[i] = round(minus_di, 4)
            adx_series[i] = round(adx_val, 4)

        return adx_series, pdi_series, mdi_series

    @staticmethod
    def _calculate_rolling_vwap(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
        window: int = 20,
    ) -> List[Optional[float]]:
        """Calculate rolling volume-weighted average price (VWAP) over *window* bars."""
        vwap: List[Optional[float]] = []
        for i in range(len(closes)):
            start_idx = max(0, i - window + 1)
            h_win = highs[start_idx : i + 1]
            l_win = lows[start_idx : i + 1]
            c_win = closes[start_idx : i + 1]
            v_win = volumes[start_idx : i + 1]
            sum_v = sum(v_win)
            if sum_v > 0:
                typ_px = [(h_win[j] + l_win[j] + c_win[j]) / 3.0 for j in range(len(c_win))]
                val = sum(typ_px[j] * v_win[j] for j in range(len(c_win))) / sum_v
                vwap.append(round(val, 4))
            else:
                vwap.append(round(sum(c_win) / len(c_win), 4))
        return vwap

    def run(
        self,
        symbol: str,
        candles: List[Dict[str, Any]],
        strategy_filter: str = "both",
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> BacktestResult:
        """Run the simulation over a chronological list of daily OHLCV candles.

        Each candle must have:
        'date': str (YYYY-MM-DD), 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float
        """
        if len(candles) < 30:
            raise ValueError(f"Insufficient historical candles ({len(candles)}) to run backtest.")

        dates = [c["date"] for c in candles]
        opens = [float(c["open"]) for c in candles]
        highs = [float(c["high"]) for c in candles]
        lows = [float(c["low"]) for c in candles]
        closes = [float(c["close"]) for c in candles]
        volumes = [float(c.get("volume", 1.0) or 1.0) for c in candles]

        # Indicator Series
        sma200 = self._calculate_sma(closes, 200)
        ema20 = self._calculate_ema(closes, 20)
        ema50 = self._calculate_ema(closes, 50)
        adx14, plus_di14, minus_di14 = self._calculate_wilder_adx(highs, lows, closes, 14)
        vwap20 = self._calculate_rolling_vwap(highs, lows, closes, volumes, 20)
        rsi3 = self._calculate_rsi(closes, 3)
        rsi7 = self._calculate_rsi(closes, 7)
        rsi14 = self._calculate_rsi(closes, 14)
        atr14 = self._calculate_atr(highs, lows, closes, 14)
        vol20 = self._calculate_realized_vol(closes, 20)
        forecast_5d = self._calculate_rolling_nixtla_forecast(closes, 5)
        sentiment_score = self._calculate_sentiment_proxy(closes, 14)

        from trading_system.strategies.base import read_config_value
        from trading_system.strategies.directional_trend import DirectionalTrendStrategy
        from trading_system.strategies.vwap import VWAPStrategy

        directional_strategy = DirectionalTrendStrategy()
        vwap_strategy = VWAPStrategy()
        ml_alpha_strategy = None
        if strategy_filter in ("ml_alpha", "all"):
            try:
                from trading_system.strategies.ml_alpha import MLAlphaStrategy
                ml_alpha_strategy = MLAlphaStrategy()
            except Exception:
                ml_alpha_strategy = None
        if config_snapshot is not None:
            max_catalyst_risk = float(
                read_config_value(config_snapshot, "SCREENER_MAX_CATALYST_RISK", 0.25)
            )
        elif settings is not None:
            max_catalyst_risk = float(
                getattr(settings, "SCREENER_MAX_CATALYST_RISK", 0.25)
            )
        else:
            max_catalyst_risk = 0.25

        capital = self.initial_capital
        equity_curve: List[float] = [capital]
        trades: List[SimulatedTrade] = []
        open_trades: List[Dict[str, Any]] = []
        trade_counter = 0
        total_bars_simulated = 0
        cash_bars = 0

        # Daily simulation loop
        for i in range(25, len(candles)):
            total_bars_simulated += 1
            curr_date = dates[i]
            o = opens[i]
            h = highs[i]
            l = lows[i]
            c = closes[i]

            # ── 1. Process Open Trades (Exits) ──
            still_open: List[Dict[str, Any]] = []
            for pos in open_trades:
                pos["hold_days"] += 1
                entry_p = pos["entry_price"]
                direction = pos["direction"]
                stop_loss = pos["current_sl"]
                rem_shares = pos["remaining_shares"]
                is_long = direction == "LONG"

                if self.pure_tsl_mode:
                    pos["highest_price"] = max(pos.get("highest_price", h), h)
                    pos["lowest_price"] = min(pos.get("lowest_price", l), l)

                    # Check SL hit
                    sl_hit = (l <= stop_loss) if is_long else (h >= stop_loss)
                    if sl_hit:
                        exit_p = stop_loss
                        pnl_per_share = (exit_p - entry_p) if is_long else (entry_p - exit_p)
                        trade_pnl = rem_shares * pnl_per_share * self.leverage
                        pos["realized_pnl"] += trade_pnl
                        capital += (rem_shares * entry_p) + trade_pnl

                        reason = pos.get("tsl_status", "SL")
                        if reason == "INITIAL":
                            reason = "SL"

                        t = SimulatedTrade(
                            trade_id=pos["id"],
                            symbol=symbol,
                            direction=direction,
                            strategy=pos["strategy"],
                            entry_date=pos["entry_date"],
                            entry_price=entry_p,
                            exit_date=curr_date,
                            exit_price=exit_p,
                            shares=pos["total_shares"],
                            notional=pos["notional"],
                            initial_sl=pos["initial_sl"],
                            tp1=pos["tp1"],
                            tp2=pos["tp2"],
                            tp3=pos["tp3"],
                            realized_pnl=round(pos["realized_pnl"], 2),
                            return_pct=round((pos["realized_pnl"] / pos["notional"]) * 100, 2) if pos["notional"] > 0 else 0.0,
                            hold_days=pos["hold_days"],
                            exit_reason=reason,
                            tp1_hit=False,
                            tp2_hit=False,
                            tp3_hit=False,
                        )
                        trades.append(t)
                        continue

                    # Progressive TSL ratchet
                    curr_atr = atr14[i] if atr14 and i < len(atr14) and atr14[i] is not None and atr14[i] > 0 else (c * 0.02)
                    if is_long:
                        peak_gain_pct = (pos["highest_price"] - entry_p) / entry_p * 100.0
                        if peak_gain_pct >= 3.0:
                            cushion = entry_p * 1.015
                            if cushion > pos["current_sl"]:
                                pos["current_sl"] = cushion
                                pos["tsl_status"] = "TSL_CUSHION"
                        elif peak_gain_pct >= 1.5:
                            if entry_p > pos["current_sl"]:
                                pos["current_sl"] = entry_p
                                pos["tsl_status"] = "TSL_BREAKEVEN"

                        if peak_gain_pct >= 1.5 or pos["current_sl"] >= entry_p:
                            dyn_tsl = pos["highest_price"] - (1.8 * curr_atr)
                            if dyn_tsl > pos["current_sl"]:
                                pos["current_sl"] = dyn_tsl
                                pos["tsl_status"] = "TSL_DYNAMIC"
                    else:
                        trough_gain_pct = (entry_p - pos["lowest_price"]) / entry_p * 100.0
                        if trough_gain_pct >= 3.0:
                            cushion = entry_p * 0.985
                            if cushion < pos["current_sl"]:
                                pos["current_sl"] = cushion
                                pos["tsl_status"] = "TSL_CUSHION"
                        elif trough_gain_pct >= 1.5:
                            if entry_p < pos["current_sl"]:
                                pos["current_sl"] = entry_p
                                pos["tsl_status"] = "TSL_BREAKEVEN"

                        if trough_gain_pct >= 1.5 or pos["current_sl"] <= entry_p:
                            dyn_tsl = pos["lowest_price"] + (1.8 * curr_atr)
                            if dyn_tsl < pos["current_sl"]:
                                pos["current_sl"] = dyn_tsl
                                pos["tsl_status"] = "TSL_DYNAMIC"
                else:
                    # ── Legacy Milestone Profit Taking (TP1/TP2/TP3) ──
                    # Check SL hit
                    sl_hit = (l <= stop_loss) if is_long else (h >= stop_loss)
                    if sl_hit:
                        exit_p = stop_loss
                        pnl_per_share = (exit_p - entry_p) if is_long else (entry_p - exit_p)
                        # Scale PnL by 3x leverage
                        trade_pnl = rem_shares * pnl_per_share * self.leverage
                        pos["realized_pnl"] += trade_pnl
                        capital += (rem_shares * entry_p) + trade_pnl

                        reason = "SL" if not pos["tp1_hit"] else "TP1_SL_BE"
                        t = SimulatedTrade(
                            trade_id=pos["id"],
                            symbol=symbol,
                            direction=direction,
                            strategy=pos["strategy"],
                            entry_date=pos["entry_date"],
                            entry_price=entry_p,
                            exit_date=curr_date,
                            exit_price=exit_p,
                            shares=pos["total_shares"],
                            notional=pos["notional"],
                            initial_sl=pos["initial_sl"],
                            tp1=pos["tp1"],
                            tp2=pos["tp2"],
                            tp3=pos["tp3"],
                            realized_pnl=round(pos["realized_pnl"], 2),
                            return_pct=round((pos["realized_pnl"] / pos["notional"]) * 100, 2) if pos["notional"] > 0 else 0.0,
                            hold_days=pos["hold_days"],
                            exit_reason=reason,
                            tp1_hit=pos["tp1_hit"],
                            tp2_hit=pos["tp2_hit"],
                            tp3_hit=pos["tp3_hit"],
                        )
                        trades.append(t)
                        continue

                    # Check TP1 hit (+1.0R): Banks 50%, moves SL to Break-Even
                    if not pos["tp1_hit"]:
                        tp1_hit = (h >= pos["tp1"]) if is_long else (l <= pos["tp1"])
                        if tp1_hit:
                            pos["tp1_hit"] = True
                            trim_shares = pos["total_shares"] * 0.50
                            pos["remaining_shares"] -= trim_shares
                            exit_p = pos["tp1"]
                            pnl = trim_shares * ((exit_p - entry_p) if is_long else (entry_p - exit_p)) * self.leverage
                            pos["realized_pnl"] += pnl
                            capital += (trim_shares * entry_p) + pnl
                            # Elevate stop to break-even
                            pos["current_sl"] = entry_p

                    # Check TP2 hit (+2.0R): Banks 25%, ratchets stop to TP1
                    if pos["tp1_hit"] and not pos["tp2_hit"]:
                        tp2_hit = (h >= pos["tp2"]) if is_long else (l <= pos["tp2"])
                        if tp2_hit:
                            pos["tp2_hit"] = True
                            trim_shares = pos["total_shares"] * 0.25
                            pos["remaining_shares"] -= trim_shares
                            exit_p = pos["tp2"]
                            pnl = trim_shares * ((exit_p - entry_p) if is_long else (entry_p - exit_p)) * self.leverage
                            pos["realized_pnl"] += pnl
                            capital += (trim_shares * entry_p) + pnl
                            # Ratchet stop to TP1
                            pos["current_sl"] = pos["tp1"]

                    # Check TP3 hit (+3.0R): Banks remaining 25%
                    if pos["tp2_hit"] and not pos["tp3_hit"]:
                        tp3_hit = (h >= pos["tp3"]) if is_long else (l <= pos["tp3"])
                        if tp3_hit:
                            pos["tp3_hit"] = True
                            rem = pos["remaining_shares"]
                            pos["remaining_shares"] = 0.0
                            exit_p = pos["tp3"]
                            pnl = rem * ((exit_p - entry_p) if is_long else (entry_p - exit_p)) * self.leverage
                            pos["realized_pnl"] += pnl
                            capital += (rem * entry_p) + pnl

                            t = SimulatedTrade(
                                trade_id=pos["id"],
                                symbol=symbol,
                                direction=direction,
                                strategy=pos["strategy"],
                                entry_date=pos["entry_date"],
                                entry_price=entry_p,
                                exit_date=curr_date,
                                exit_price=exit_p,
                                shares=pos["total_shares"],
                                notional=pos["notional"],
                                initial_sl=pos["initial_sl"],
                                tp1=pos["tp1"],
                                tp2=pos["tp2"],
                                tp3=pos["tp3"],
                                realized_pnl=round(pos["realized_pnl"], 2),
                                return_pct=round((pos["realized_pnl"] / pos["notional"]) * 100, 2) if pos["notional"] > 0 else 0.0,
                                hold_days=pos["hold_days"],
                                exit_reason="TP3",
                                tp1_hit=True,
                                tp2_hit=True,
                                tp3_hit=True,
                            )
                            trades.append(t)
                            continue

                    # Check Nixtla forecast reversal trim for multi_factor trades
                    if pos["strategy"] == "multi_factor" and not pos.get("forecast_trimmed", False):
                        curr_f = forecast_5d[i]
                        if curr_f is not None:
                            unrealized_pnl_pct = (
                                ((c - entry_p) / entry_p * 100.0)
                                if is_long
                                else ((entry_p - c) / entry_p * 100.0)
                            )
                            if ((is_long and curr_f <= -3.0) or (not is_long and curr_f >= 3.0)) and unrealized_pnl_pct >= 1.5:
                                pos["forecast_trimmed"] = True
                                trim_shares = pos["remaining_shares"] * 0.50
                                pos["remaining_shares"] -= trim_shares
                                exit_p = c
                                pnl = trim_shares * ((exit_p - entry_p) if is_long else (entry_p - exit_p)) * self.leverage
                                pos["realized_pnl"] += pnl
                                capital += (trim_shares * entry_p) + pnl
                                # Elevate stop loss to break-even to protect remaining shares
                                if is_long and pos["current_sl"] < entry_p:
                                    pos["current_sl"] = entry_p
                                elif not is_long and pos["current_sl"] > entry_p:
                                    pos["current_sl"] = entry_p

                # Check Max Hold Days cap (Connors <= 4 days to kill beta decay)
                if pos["hold_days"] >= self.max_hold_days and pos["remaining_shares"] > 0:
                    exit_p = c
                    rem = pos["remaining_shares"]
                    pos["remaining_shares"] = 0.0
                    pnl = rem * ((exit_p - entry_p) if is_long else (entry_p - exit_p)) * self.leverage
                    pos["realized_pnl"] += pnl
                    capital += (rem * entry_p) + pnl

                    t = SimulatedTrade(
                        trade_id=pos["id"],
                        symbol=symbol,
                        direction=direction,
                        strategy=pos["strategy"],
                        entry_date=pos["entry_date"],
                        entry_price=entry_p,
                        exit_date=curr_date,
                        exit_price=exit_p,
                        shares=pos["total_shares"],
                        notional=pos["notional"],
                        initial_sl=pos["initial_sl"],
                        tp1=pos["tp1"],
                        tp2=pos["tp2"],
                        tp3=pos["tp3"],
                        realized_pnl=round(pos["realized_pnl"], 2),
                        return_pct=round((pos["realized_pnl"] / pos["notional"]) * 100, 2) if pos["notional"] > 0 else 0.0,
                        hold_days=pos["hold_days"],
                        exit_reason="MAX_HOLD",
                        tp1_hit=pos["tp1_hit"],
                        tp2_hit=pos["tp2_hit"],
                        tp3_hit=pos["tp3_hit"],
                    )
                    trades.append(t)
                    continue

                still_open.append(pos)

            open_trades = still_open

            # ── 2. Signal Generation (Entries) ──
            # Only open if no existing open position for this asset
            if not open_trades and i < len(candles) - 1:
                curr_c = closes[i]
                curr_atr = atr14[i] or (curr_c * 0.02)
                curr_r3 = rsi3[i]
                curr_r7 = rsi7[i]
                curr_r14 = rsi14[i]
                curr_sma = sma200[i]
                curr_vol = vol20[i] or 0.25
                curr_forecast = forecast_5d[i]
                curr_sentiment = sentiment_score[i]

                signal_dir: Optional[str] = None
                strategy_name: str = ""

                # Multi-Factor Strategy: Connors 3-RSI + 20d Trend + Nixtla Forecast + Sentiment
                if strategy_filter in ("both", "multi_factor"):
                    if curr_r3 is not None and curr_r7 is not None and curr_sma is not None:
                        # Long Dip confirmed by positive Nixtla forecast and non-capitulating sentiment
                        if (
                            curr_c >= curr_sma
                            and curr_r3 <= 25.0
                            and curr_r7 <= 45.0
                            and (curr_forecast is not None and curr_forecast > 0.0)
                            and (curr_sentiment is None or curr_sentiment >= 35.0)
                        ):
                            signal_dir = "LONG"
                            strategy_name = "multi_factor"
                        # Short Rip confirmed by negative Nixtla forecast and bearish/neutral sentiment
                        elif (
                            curr_c < curr_sma
                            and (ema50[i] is None or curr_c < ema50[i])
                            and curr_r3 >= 75.0
                            and curr_r7 >= 55.0
                            and (curr_forecast is not None and curr_forecast < 0.0)
                            and (curr_sentiment is None or curr_sentiment <= 65.0)
                        ):
                            signal_dir = "SHORT"
                            strategy_name = "multi_factor"

                    if not signal_dir:
                        # Multi-Factor Momentum Breakout confirmed by positive forecast
                        high_20d = max(highs[max(0, i - 20) : i])
                        low_20d = min(lows[max(0, i - 20) : i])
                        if (
                            curr_c > high_20d
                            and curr_sma
                            and curr_c > curr_sma
                            and (curr_forecast is not None and curr_forecast > 0.5)
                            and (curr_sentiment is None or curr_sentiment >= 45.0)
                        ):
                            signal_dir = "LONG"
                            strategy_name = "multi_factor"
                        elif (
                            curr_c < low_20d
                            and curr_sma
                            and curr_c < curr_sma
                            and (curr_forecast is not None and curr_forecast < -0.5)
                            and (curr_sentiment is None or curr_sentiment <= 55.0)
                            and (curr_r3 is None or curr_r3 >= 30.0)
                            and (curr_r7 is None or curr_r7 >= 35.0)
                        ):
                            signal_dir = "SHORT"
                            strategy_name = "multi_factor"

                # Directional Trend Strategy
                if not signal_dir and strategy_filter in ("directional_trend", "all"):
                    e20 = ema20[i] if i < len(ema20) and ema20[i] is not None else curr_c
                    e50 = ema50[i] if i < len(ema50) and ema50[i] is not None else curr_c
                    slope_5d: Optional[float] = None
                    if i >= 5 and ema20[i] is not None and ema20[i - 5] is not None and ema20[i - 5] > 0:
                        slope_5d = ((ema20[i] - ema20[i - 5]) / ema20[i - 5]) * 100.0
                    bar_metrics: Dict[str, Any] = {
                        "close": curr_c,
                        "sma200": curr_sma if curr_sma is not None else 0.0,
                        "ema20": e20,
                        "ema50": e50,
                        "ema20_slope_5d": slope_5d,
                        "adx": adx14[i] if i < len(adx14) and adx14[i] is not None else 0.0,
                        "plus_di": plus_di14[i] if i < len(plus_di14) and plus_di14[i] is not None else 0.0,
                        "minus_di": minus_di14[i] if i < len(minus_di14) and minus_di14[i] is not None else 0.0,
                        "atr": curr_atr,
                        "realized_vol": curr_vol,
                        "rsi3": curr_r3,
                        "rsi7": curr_r7,
                        "rsi14": curr_r14,
                    }
                    if candles[i].get("earnings_in_days") is not None:
                        bar_metrics["earnings_in_days"] = candles[i]["earnings_in_days"]
                    dir_sig = directional_strategy.evaluate_candidate(
                        symbol, bar_metrics, config_snapshot=config_snapshot
                    )
                    if dir_sig.is_actionable and dir_sig.catalyst_risk <= max_catalyst_risk:
                        signal_dir = dir_sig.direction
                        strategy_name = "directional_trend"

                # VWAP Session Trend Strategy
                if not signal_dir and strategy_filter in ("vwap", "all"):
                    curr_vwap = vwap20[i] if (vwap20 and vwap20[i] is not None) else curr_c
                    vwap_dist = ((curr_c - curr_vwap) / curr_vwap) * 100.0 if curr_vwap > 0 else 0.0
                    e20 = ema20[i] if i < len(ema20) and ema20[i] is not None else curr_c
                    e50 = ema50[i] if i < len(ema50) and ema50[i] is not None else curr_c
                    vwap_bar_metrics: Dict[str, Any] = {
                        "close": curr_c,
                        "vwap": curr_vwap,
                        "vwap_dist_pct": vwap_dist,
                        "sma200": curr_sma if curr_sma is not None else 0.0,
                        "ema20": e20,
                        "ema50": e50,
                        "ema20_daily": e20,
                        "ema50_daily": e50,
                        "adx": adx14[i] if i < len(adx14) and adx14[i] is not None else 0.0,
                        "plus_di": plus_di14[i] if i < len(plus_di14) and plus_di14[i] is not None else 0.0,
                        "minus_di": minus_di14[i] if i < len(minus_di14) and minus_di14[i] is not None else 0.0,
                        "atr": curr_atr,
                        "realized_vol": curr_vol,
                        "rsi3": curr_r3,
                        "rsi7": curr_r7,
                        "rsi14": curr_r14,
                    }
                    if candles[i].get("earnings_in_days") is not None:
                        vwap_bar_metrics["earnings_in_days"] = candles[i]["earnings_in_days"]
                    vwap_sig = vwap_strategy.evaluate_candidate(
                        symbol, vwap_bar_metrics, config_snapshot=config_snapshot
                    )
                    if vwap_sig.is_actionable and vwap_sig.catalyst_risk <= max_catalyst_risk:
                        signal_dir = vwap_sig.direction
                        strategy_name = "vwap"

                # ML Alpha Strategy (Multi-Factor ONNX with Dual-Track Bear Shield)
                if not signal_dir and strategy_filter in ("ml_alpha", "all"):
                    if ml_alpha_strategy is None:
                        try:
                            from trading_system.strategies.ml_alpha import MLAlphaStrategy
                            ml_alpha_strategy = MLAlphaStrategy()
                        except Exception:
                            ml_alpha_strategy = None
                    if ml_alpha_strategy is not None:
                        e20 = ema20[i] if i < len(ema20) and ema20[i] is not None else curr_c
                        e50 = ema50[i] if i < len(ema50) and ema50[i] is not None else curr_c
                        ml_bar_metrics: Dict[str, Any] = {
                            "close": curr_c,
                            "open": o,
                            "high": h,
                            "low": l,
                            "volume": volumes[i],
                            "bars": candles[: i + 1],
                            "candles": candles[: i + 1],
                            "date": curr_date,
                            "sma200": curr_sma if curr_sma is not None else 0.0,
                            "ema20": e20,
                            "ema50": e50,
                            "adx": adx14[i] if i < len(adx14) and adx14[i] is not None else 0.0,
                            "plus_di": plus_di14[i] if i < len(plus_di14) and plus_di14[i] is not None else 0.0,
                            "minus_di": minus_di14[i] if i < len(minus_di14) and minus_di14[i] is not None else 0.0,
                            "atr": curr_atr,
                            "realized_vol": curr_vol,
                        }
                        if candles[i].get("earnings_in_days") is not None:
                            ml_bar_metrics["earnings_in_days"] = candles[i]["earnings_in_days"]
                        try:
                            ml_sig = ml_alpha_strategy.evaluate_candidate(
                                symbol, ml_bar_metrics, config_snapshot=config_snapshot
                            )
                            if ml_sig and ml_sig.is_actionable and ml_sig.catalyst_risk <= max_catalyst_risk:
                                signal_dir = ml_sig.direction
                                strategy_name = "ml_alpha"
                        except Exception as e:
                            logger.debug("MLAlphaStrategy evaluation exception: %s", e)

                # 3-RSI Mean Reversion (standalone or fallback if not multi_factor only)
                if not signal_dir and strategy_filter in ("both", "3rsi"):
                    if curr_r3 is not None and curr_r7 is not None and curr_sma is not None:
                        # Long Dip: In uptrend (close >= sma) and extreme oversold
                        if curr_c >= curr_sma and curr_r3 <= 22.0 and curr_r7 <= 42.0:
                            signal_dir = "LONG"
                            strategy_name = "3rsi_mean_reversion"
                        # Short Rip: In downtrend (close < sma and close < ema50) and extreme overbought
                        elif (
                            curr_c < curr_sma
                            and (ema50[i] is None or curr_c < ema50[i])
                            and curr_r3 >= 78.0
                            and curr_r7 >= 58.0
                        ):
                            signal_dir = "SHORT"
                            strategy_name = "3rsi_mean_reversion"

                # Momentum breakout (fallback if no mean reversion signal)
                if not signal_dir and strategy_filter in ("both", "momentum"):
                    # 20-day high breakout
                    high_20d = max(highs[max(0, i - 20) : i])
                    low_20d = min(lows[max(0, i - 20) : i])
                    if curr_c > high_20d and curr_sma and curr_c > curr_sma:
                        signal_dir = "LONG"
                        strategy_name = "momentum"
                    elif (
                        curr_c < low_20d
                        and curr_sma
                        and curr_c < curr_sma
                        and (curr_r3 is None or curr_r3 >= 30.0)
                        and (curr_r7 is None or curr_r7 >= 35.0)
                    ):
                        signal_dir = "SHORT"
                        strategy_name = "momentum"

                if signal_dir:
                    trade_counter += 1
                    # Enter at next candle open (or today close)
                    entry_p = curr_c
                    sl_dist = max(2.2 * curr_atr, entry_p * 0.005)
                    initial_sl = (entry_p - sl_dist) if signal_dir == "LONG" else (entry_p + sl_dist)
                    stop_dist_pct = abs(entry_p - initial_sl) / entry_p
                    risk_amount = capital * self.max_risk_per_trade

                    notional_from_risk = risk_amount / stop_dist_pct if stop_dist_pct > 0 else 0.0
                    max_notional = capital * self.max_position_pct
                    target_notional = min(notional_from_risk, max_notional)

                    # Dynamic Volatility Sizing:
                    if curr_vol > 0.05:
                        effective_vol = max(0.10, self.leverage * curr_vol)
                        vol_budget = max_notional * (self.target_vol / effective_vol)
                        target_notional = min(target_notional, vol_budget)

                    shares = max(1.0, round(target_notional / entry_p, 2))
                    actual_notional = round(shares * entry_p, 2)

                    # R targets
                    r_dist = abs(entry_p - initial_sl)
                    if signal_dir == "LONG":
                        tp1 = entry_p + (1.0 * r_dist)
                        tp2 = entry_p + (2.0 * r_dist)
                        tp3 = entry_p + (3.0 * r_dist)
                    else:
                        tp1 = max(0.01, entry_p - (1.0 * r_dist))
                        tp2 = max(0.01, entry_p - (2.0 * r_dist))
                        tp3 = max(0.01, entry_p - (3.0 * r_dist))

                    capital -= actual_notional
                    open_trades.append({
                        "id": trade_counter,
                        "entry_date": curr_date,
                        "entry_price": entry_p,
                        "direction": signal_dir,
                        "strategy": strategy_name,
                        "total_shares": shares,
                        "remaining_shares": shares,
                        "notional": actual_notional,
                        "initial_sl": initial_sl,
                        "current_sl": initial_sl,
                        "highest_price": entry_p,
                        "lowest_price": entry_p,
                        "tsl_status": "INITIAL",
                        "tp1": tp1,
                        "tp2": tp2,
                        "tp3": tp3,
                        "tp1_hit": False,
                        "tp2_hit": False,
                        "tp3_hit": False,
                        "forecast_trimmed": False,
                        "realized_pnl": 0.0,
                        "hold_days": 0,
                    })

            if not open_trades:
                cash_bars += 1

            # Track end-of-day equity
            open_pnl = 0.0
            for pos in open_trades:
                is_long = pos["direction"] == "LONG"
                cur_pnl = pos["remaining_shares"] * ((c - pos["entry_price"]) if is_long else (pos["entry_price"] - c)) * self.leverage
                open_pnl += pos["realized_pnl"] + cur_pnl + (pos["remaining_shares"] * pos["entry_price"])
            equity_curve.append(capital + open_pnl)

        # Close any lingering positions at end of dataset
        if open_trades:
            last_date = dates[-1]
            last_close = closes[-1]
            for pos in open_trades:
                rem = pos["remaining_shares"]
                if rem > 0:
                    is_long = pos["direction"] == "LONG"
                    pnl = rem * ((last_close - pos["entry_price"]) if is_long else (pos["entry_price"] - last_close)) * self.leverage
                    pos["realized_pnl"] += pnl
                    capital += (rem * pos["entry_price"]) + pnl
                    trades.append(SimulatedTrade(
                        trade_id=pos["id"],
                        symbol=symbol,
                        direction=pos["direction"],
                        strategy=pos["strategy"],
                        entry_date=pos["entry_date"],
                        entry_price=pos["entry_price"],
                        exit_date=last_date,
                        exit_price=last_close,
                        shares=pos["total_shares"],
                        notional=pos["notional"],
                        initial_sl=pos["initial_sl"],
                        tp1=pos["tp1"],
                        tp2=pos["tp2"],
                        tp3=pos["tp3"],
                        realized_pnl=round(pos["realized_pnl"], 2),
                        return_pct=round((pos["realized_pnl"] / pos["notional"]) * 100, 2) if pos["notional"] > 0 else 0.0,
                        hold_days=pos["hold_days"],
                        exit_reason="END_OF_DATA",
                        tp1_hit=pos["tp1_hit"],
                        tp2_hit=pos["tp2_hit"],
                        tp3_hit=pos["tp3_hit"],
                    ))

        # ── 3. Performance Metric Computations ──
        ending_capital = capital
        total_ret_pct = ((ending_capital - self.initial_capital) / self.initial_capital) * 100.0
        # Benchmark (1x Buy & Hold)
        bench_ret_pct = ((closes[-1] - closes[25]) / closes[25]) * 100.0

        win_trades = [t for t in trades if t.realized_pnl > 0]
        loss_trades = [t for t in trades if t.realized_pnl <= 0]
        win_count = len(win_trades)
        loss_count = len(loss_trades)
        total_count = len(trades)
        win_rate = (win_count / total_count * 100.0) if total_count > 0 else 0.0

        gross_profit = sum(t.realized_pnl for t in win_trades)
        gross_loss = abs(sum(t.realized_pnl for t in loss_trades))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 1.0)

        # Max Drawdown
        peak = equity_curve[0]
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        max_dd_pct = max_dd * 100.0

        # Sharpe Ratio (daily equity changes)
        daily_returns = [
            (equity_curve[j] - equity_curve[j - 1]) / equity_curve[j - 1]
            for j in range(1, len(equity_curve))
            if equity_curve[j - 1] > 0
        ]
        if len(daily_returns) > 5:
            mean_ret = sum(daily_returns) / len(daily_returns)
            var_ret = sum((r - mean_ret) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
            std_ret = math.sqrt(var_ret) if var_ret > 0 else 0.0
            sharpe = (mean_ret / std_ret * math.sqrt(252)) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        avg_pnl = sum(t.realized_pnl for t in trades) / total_count if total_count > 0 else 0.0
        avg_hold = sum(t.hold_days for t in trades) / total_count if total_count > 0 else 0.0
        time_in_cash_pct = (cash_bars / total_bars_simulated * 100.0) if total_bars_simulated > 0 else 100.0

        return BacktestResult(
            symbol=symbol,
            start_date=dates[25],
            end_date=dates[-1],
            initial_capital=self.initial_capital,
            ending_capital=round(ending_capital, 2),
            total_return_pct=round(total_ret_pct, 2),
            benchmark_return_pct=round(bench_ret_pct, 2),
            total_trades=total_count,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate_pct=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            sharpe_ratio=round(sharpe, 2),
            avg_trade_pnl=round(avg_pnl, 2),
            avg_hold_days=round(avg_hold, 1),
            trades=trades,
            time_in_cash_pct=round(time_in_cash_pct, 1),
        )

    def run_comparison(
        self,
        symbol: str,
        candles: List[Dict[str, Any]],
        strategy_a: str = "baseline",
        strategy_b: str = "multi_factor",
        config_snapshot: Optional[ConfigSnapshot] = None,
    ) -> Dict[str, BacktestResult]:
        """Run two strategies and return comparative results."""
        filter_a = "both" if strategy_a == "baseline" else strategy_a
        filter_b = "both" if strategy_b == "baseline" else strategy_b
        res_a = self.run(symbol, candles, strategy_filter=filter_a, config_snapshot=config_snapshot)
        res_b = self.run(symbol, candles, strategy_filter=filter_b, config_snapshot=config_snapshot)
        return {strategy_a: res_a, strategy_b: res_b}


def generate_synthetic_candles(symbol: str, count: int = 150) -> List[Dict[str, Any]]:
    """Generate realistic daily OHLCV candles for backtest simulation verification."""
    import random
    base_price = 150.0 if "QQQ" in symbol else (100.0 if "NVDA" in symbol else 80.0)
    current_price = base_price
    candles: List[Dict[str, Any]] = []

    # Deterministic seed for reproducible tests
    rng = random.Random(42)

    for i in range(count):
        # Drift + volatility with occasional mean-reverting pullbacks
        drift = 0.0008
        shock = rng.gauss(0, 0.02)
        if i % 15 == 0:  # Occasional oversold dip
            shock = -0.04
        daily_ret = drift + shock
        open_p = current_price
        close_p = open_p * (1.0 + daily_ret)
        high_p = max(open_p, close_p) * (1.0 + abs(rng.gauss(0, 0.008)))
        low_p = min(open_p, close_p) * (1.0 - abs(rng.gauss(0, 0.008)))
        vol = rng.randint(2000000, 10000000)

        dt = f"2025-01-{(i % 28) + 1:02d}"
        candles.append({
            "date": dt,
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low_p, 2),
            "close": round(close_p, 2),
            "volume": vol,
        })
        current_price = close_p

    return candles


def print_backtest_report(res: BacktestResult) -> None:
    """Print an institutional-grade ASCII performance report."""
    sep = "═" * 68
    sub_sep = "─" * 68
    print(sep)
    print(f"  📊 HISTORICAL BACKTEST BENCHMARK REPORT: {res.symbol}")
    print(sep)
    print(f"  Period:            {res.start_date} to {res.end_date}")
    print(f"  Initial Capital:   ${res.initial_capital:,.2f}")
    print(f"  Ending Capital:    ${res.ending_capital:,.2f}")
    print(f"  Strategy Return:   {res.total_return_pct:+.2f}%")
    print(f"  Benchmark (1x B&H): {res.benchmark_return_pct:+.2f}%")
    print(sub_sep)
    print(f"  Total Trades:      {res.total_trades}")
    print(f"  Win Rate:          {res.win_rate_pct:.1f}% ({res.winning_trades} wins, {res.losing_trades} losses)")
    print(f"  Profit Factor:     {res.profit_factor:.2f}")
    print(f"  Max Drawdown:      {res.max_drawdown_pct:.2f}%")
    print(f"  Sharpe Ratio:      {res.sharpe_ratio:.2f}")
    print(f"  Avg Trade PnL:     ${res.avg_trade_pnl:+.2f}")
    print(f"  Avg Holding Period: {res.avg_hold_days:.1f} days")
    print(f"  Time in Cash:      {res.time_in_cash_pct:.1f}%")
    print(sub_sep)
    if res.trades:
        print("  Sample Trades:")
        for t in res.trades[:5]:
            outcome = "WIN" if t.realized_pnl > 0 else "LOSS"
            print(f"   [{outcome}] #{t.trade_id} {t.direction} {t.entry_date}->{t.exit_date} "
                  f"Entry: ${t.entry_price:.2f} Exit: ${float(t.exit_price or 0):.2f} "
                  f"PnL: ${t.realized_pnl:+.2f} ({t.return_pct:+.1f}%) via {t.exit_reason}")
    print(sep)


def print_comparison_report(comp: Dict[str, BacktestResult]) -> None:
    """Print side-by-side comparative table between two strategies."""
    keys = list(comp.keys())
    if len(keys) < 2:
        return
    k_a, k_b = keys[0], keys[1]
    b = comp[k_a]
    m = comp[k_b]
    if not b or not m:
        return

    name_map = {
        "baseline": "Baseline (3-RSI/Mom)",
        "multi_factor": "Multi-Factor (Nixtla/Sent)",
        "vwap": "VWAP Session Trend",
        "directional_trend": "Directional Trend",
        "ml_alpha": "ML Alpha Strategy",
        "3rsi": "3-RSI Mean Reversion",
        "momentum": "Momentum Breakout",
    }
    label_a = name_map.get(k_a, k_a)
    label_b = name_map.get(k_b, k_b)

    sep = "═" * 78
    sub_sep = "─" * 78
    print(sep)
    print(f"  📊 STRATEGY COMPARISON REPORT: {label_a.upper()} vs {label_b.upper()} ({b.symbol})")
    print(sep)
    print(f"  {'Metric':<26} | {label_a:<22} | {label_b:<22}")
    print(sub_sep)
    print(f"  {'Total Return':<26} | {b.total_return_pct:>+21.2f}% | {m.total_return_pct:>+21.2f}%")
    print(f"  {'Benchmark (1x B&H)':<26} | {b.benchmark_return_pct:>+21.2f}% | {m.benchmark_return_pct:>+21.2f}%")
    print(f"  {'Total Trades':<26} | {b.total_trades:>22} | {m.total_trades:>22}")
    print(f"  {'Win Rate':<26} | {f'{b.win_rate_pct:.1f}%':>22} | {f'{m.win_rate_pct:.1f}%':>22}")
    print(f"  {'Profit Factor':<26} | {b.profit_factor:>22.2f} | {m.profit_factor:>22.2f}")
    print(f"  {'Max Drawdown':<26} | {f'{b.max_drawdown_pct:.2f}%':>22} | {f'{m.max_drawdown_pct:.2f}%':>22}")
    print(f"  {'Sharpe Ratio':<26} | {b.sharpe_ratio:>22.2f} | {m.sharpe_ratio:>22.2f}")
    print(f"  {'Avg Trade PnL':<26} | {f'${b.avg_trade_pnl:+.2f}':>22} | {f'${m.avg_trade_pnl:+.2f}':>22}")
    print(f"  {'Avg Holding Period':<26} | {f'{b.avg_hold_days:.1f} days':>22} | {f'{m.avg_hold_days:.1f} days':>22}")
    print(f"  {'Time in Cash':<26} | {f'{b.time_in_cash_pct:.1f}%':>22} | {f'{m.time_in_cash_pct:.1f}%':>22}")
    print(sep)


def main() -> None:
    """Command-line entry point for backtesting."""
    parser = argparse.ArgumentParser(description="3x Leveraged Strategy Backtest & Benchmark Engine")
    parser.add_argument("--symbol", type=str, default="QQQ", help="Underlying asset symbol (e.g. QQQ, NVDA, SPY)")
    parser.add_argument("--period", type=str, default="1y", help="Historical data window (e.g. 6m, 1y, 2y)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Starting account equity")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["3rsi", "momentum", "multi_factor", "directional_trend", "vwap", "ml_alpha", "both", "all"],
        default="both",
        help="Strategy to test",
    )
    parser.add_argument("--leverage", type=int, default=3, help="Leverage multiplier (default 3)")
    parser.add_argument("--compare", action="store_true", help="Compare baseline vs multi-factor strategy side-by-side")
    parser.add_argument(
        "--compare-strategies",
        nargs=2,
        metavar=("STRAT_A", "STRAT_B"),
        default=None,
        help="Compare any two strategies side-by-side (e.g. --compare-strategies vwap directional_trend)",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON results")
    parser.add_argument("--dry-run", action="store_true", help="Run quick verification with synthetic candles")
    parser.add_argument("--pure-tsl", dest="pure_tsl", action="store_true", default=None, help="Enforce 100% Pure Trailing Stop Loss engine")
    parser.add_argument("--legacy-tp", dest="pure_tsl", action="store_false", help="Use legacy TP1/TP2/TP3 milestone trims")

    args = parser.parse_args()

    candles = None
    if not args.dry_run:
        try:
            from trading_system.data.trading212_client import MarketDataProvider
            raw = MarketDataProvider.get_daily_candles(args.symbol, period=args.period)
            if raw is not None:
                if hasattr(raw, "to_dict"):
                    candles = raw.to_dict(orient="records")
                    if "date" not in raw.columns and hasattr(raw, "index"):
                        for idx, rec in enumerate(candles):
                            rec["date"] = str(raw.index[idx])[:10]
                elif isinstance(raw, list):
                    candles = raw
        except Exception as e:
            logger.warning("Could not fetch market data for %s (%s). Falling back.", args.symbol, e)

    if not candles:
        candles = generate_synthetic_candles(args.symbol, count=160)

    engine = BacktestEngine(
        initial_capital=args.capital,
        leverage=args.leverage,
        max_risk_per_trade=0.005,
        max_position_pct=0.05,
        target_vol=0.20,
        max_hold_days=4,
        pure_tsl_mode=args.pure_tsl,
    )

    if args.compare_strategies:
        strat_a, strat_b = args.compare_strategies
        comp = engine.run_comparison(symbol=args.symbol, candles=candles, strategy_a=strat_a, strategy_b=strat_b)
        if args.json:
            print(json.dumps({k: v.to_dict() for k, v in comp.items()}, indent=2))
        else:
            print_comparison_report(comp)
    elif args.compare:
        comp = engine.run_comparison(symbol=args.symbol, candles=candles)
        if args.json:
            print(json.dumps({k: v.to_dict() for k, v in comp.items()}, indent=2))
        else:
            print_comparison_report(comp)
    else:
        result = engine.run(symbol=args.symbol, candles=candles, strategy_filter=args.strategy)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print_backtest_report(result)


if __name__ == "__main__":
    main()
