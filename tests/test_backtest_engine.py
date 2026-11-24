"""Tests for Historical Backtesting Engine and Strategy Benchmark Simulator."""

import unittest
from trading_system.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    SimulatedTrade,
    generate_synthetic_candles,
)


class TestBacktestEngine(unittest.TestCase):
    def setUp(self):
        self.engine = BacktestEngine(
            initial_capital=10000.0,
            leverage=3,
            max_risk_per_trade=0.005,
            max_position_pct=0.05,
            target_vol=0.20,
            max_hold_days=4,
        )

    def test_indicator_calculations(self):
        """Verify SMA, RSI, ATR, and Realized Vol math functions."""
        closes = [100.0 + i for i in range(30)]
        sma10 = self.engine._calculate_sma(closes, 10)
        self.assertEqual(len(sma10), 30)
        self.assertAlmostEqual(sma10[8], 104.0)  # expanding mean of 100..108
        self.assertAlmostEqual(sma10[9], 104.5)  # rolling mean of 100..109

        # RSI should be 100 for strictly increasing closes
        rsi3 = self.engine._calculate_rsi(closes, 3)
        self.assertEqual(len(rsi3), 30)
        self.assertAlmostEqual(rsi3[-1], 100.0)

        # Realized vol on constant positive returns
        vol = self.engine._calculate_realized_vol(closes, 20)
        self.assertEqual(len(vol), 30)
        self.assertIsNotNone(vol[-1])
        self.assertGreater(vol[-1], 0.0)

    def test_simulation_with_synthetic_candles(self):
        """Run full backtest over synthetic candles and verify metrics."""
        candles = generate_synthetic_candles("QQQ", count=120)
        res = self.engine.run(symbol="QQQ", candles=candles, strategy_filter="both")

        self.assertIsInstance(res, BacktestResult)
        self.assertEqual(res.symbol, "QQQ")
        self.assertEqual(res.initial_capital, 10000.0)
        self.assertGreater(res.total_trades, 0)
        self.assertGreaterEqual(res.win_rate_pct, 0.0)
        self.assertLessEqual(res.win_rate_pct, 100.0)
        self.assertGreaterEqual(res.profit_factor, 0.0)
        self.assertGreaterEqual(res.max_drawdown_pct, 0.0)
        self.assertLessEqual(res.max_drawdown_pct, 100.0)

        # Ensure trade serialization works
        res_dict = res.to_dict()
        self.assertIn("trades", res_dict)
        self.assertEqual(len(res_dict["trades"]), len(res.trades))

    def test_max_hold_days_enforced(self):
        """Verify simulated trades do not exceed the max_hold_days limit."""
        candles = generate_synthetic_candles("NVDA", count=100)
        res = self.engine.run(symbol="NVDA", candles=candles)
        for t in res.trades:
            self.assertLessEqual(t.hold_days, self.engine.max_hold_days)

    def test_insufficient_candles_raises_error(self):
        """Verify ValueError is raised if candle series is too short."""
        short_candles = [
            {"date": "2025-01-01", "open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000}
            for _ in range(10)
        ]
        with self.assertRaises(ValueError):
            self.engine.run(symbol="QQQ", candles=short_candles)

    def test_tp1_be_and_tsl_progression(self):
        """Construct synthetic scenario where price hits TP1, ratchets to BE, then hits TP2 and TP3."""
        # Baseline warm-up candles (30 rising candles from 60 to 110)
        candles = []
        for i in range(30):
            p = 60.0 + i * (50.0 / 30.0)
            candles.append({
                "date": f"2025-01-{(i%28)+1:02d}",
                "open": round(p - 0.5, 2),
                "high": round(p + 1.0, 2),
                "low": round(p - 1.0, 2),
                "close": round(p, 2),
                "volume": 5000000,
            })

        # Candle 30 & 31: oversold dip to 92 (close 92 > sma ~85, RSI3 < 22)
        candles.append({
            "date": "2025-02-01",
            "open": 108.0,
            "high": 108.0,
            "low": 97.0,
            "close": 98.0,
            "volume": 10000000,
        })
        candles.append({
            "date": "2025-02-02",
            "open": 97.0,
            "high": 97.0,
            "low": 91.0,
            "close": 92.0,
            "volume": 10000000,
        })

        # Candle 32: massive surge to 135 (hits TP1, TP2, and TP3)
        candles.append({
            "date": "2025-02-03",
            "open": 93.0,
            "high": 135.0,
            "low": 92.0,
            "close": 130.0,
            "volume": 15000000,
        })

        legacy_engine = BacktestEngine(
            initial_capital=10000.0,
            leverage=3,
            max_risk_per_trade=0.005,
            max_position_pct=0.05,
            target_vol=0.20,
            max_hold_days=4,
            pure_tsl_mode=False,
        )
        res = legacy_engine.run(symbol="TEST", candles=candles, strategy_filter="3rsi")
        self.assertGreaterEqual(len(res.trades), 1)
        first_trade = res.trades[0]
        self.assertTrue(first_trade.tp1_hit)
        self.assertEqual(first_trade.exit_reason, "TP3")
        self.assertGreater(first_trade.realized_pnl, 0.0)

    def test_pure_tsl_progression(self):
        """Construct scenario where price rises into profit zone, ratchets TSL, and exits 100% on reversal."""
        candles = []
        for i in range(30):
            p = 60.0 + i * (50.0 / 30.0)
            candles.append({
                "date": f"2025-01-{(i%28)+1:02d}",
                "open": round(p - 0.5, 2),
                "high": round(p + 1.0, 2),
                "low": round(p - 1.0, 2),
                "close": round(p, 2),
                "volume": 5000000,
            })
        candles.append({"date": "2025-02-01", "open": 108.0, "high": 108.0, "low": 97.0, "close": 98.0, "volume": 10000000})
        candles.append({"date": "2025-02-02", "open": 97.0, "high": 97.0, "low": 91.0, "close": 92.0, "volume": 10000000})
        candles.append({"date": "2025-02-03", "open": 93.0, "high": 110.0, "low": 92.0, "close": 108.0, "volume": 15000000})
        candles.append({"date": "2025-02-04", "open": 106.0, "high": 107.0, "low": 95.0, "close": 96.0, "volume": 15000000})

        res = self.engine.run(symbol="TEST", candles=candles, strategy_filter="3rsi")
        self.assertGreaterEqual(len(res.trades), 1)
        first_trade = res.trades[0]
        self.assertIn("TSL", first_trade.exit_reason)
        self.assertGreater(first_trade.realized_pnl, 0.0)
        self.assertFalse(first_trade.tp1_hit)

    def test_rolling_nixtla_forecast_and_sentiment_proxy(self):
        """Verify rolling 5-day forecast and sentiment proxy mathematical integrity."""
        closes = [100.0 + (i * 0.5) for i in range(50)]
        forecasts = self.engine._calculate_rolling_nixtla_forecast(closes, horizon=5)
        self.assertEqual(len(forecasts), 50)
        # Rising price series should generate positive 5-day forecasted return
        self.assertIsNotNone(forecasts[-1])
        self.assertGreater(forecasts[-1], 0.0)

        sentiments = self.engine._calculate_sentiment_proxy(closes, period=14)
        self.assertEqual(len(sentiments), 50)
        self.assertIsNotNone(sentiments[-1])
        self.assertGreaterEqual(sentiments[-1], 0.0)
        self.assertLessEqual(sentiments[-1], 100.0)

    def test_multi_factor_strategy_simulation(self):
        """Run backtest with multi_factor strategy filter and verify trade tagging."""
        candles = generate_synthetic_candles("NVDA", count=160)
        res = self.engine.run(symbol="NVDA", candles=candles, strategy_filter="multi_factor")
        self.assertIsInstance(res, BacktestResult)
        for t in res.trades:
            self.assertEqual(t.strategy, "multi_factor")

    def test_run_comparison(self):
        """Verify run_comparison returns both baseline and multi_factor results."""
        candles = generate_synthetic_candles("QQQ", count=150)
        comp = self.engine.run_comparison(symbol="QQQ", candles=candles)
        self.assertIn("baseline", comp)
        self.assertIn("multi_factor", comp)
        self.assertIsInstance(comp["baseline"], BacktestResult)
        self.assertIsInstance(comp["multi_factor"], BacktestResult)
        self.assertEqual(comp["baseline"].symbol, "QQQ")
        self.assertEqual(comp["multi_factor"].symbol, "QQQ")

    def test_vwap_strategy_simulation(self):
        """Run backtest with vwap strategy filter and verify trade tagging and execution."""
        candles = generate_synthetic_candles("QQQ", count=160)
        res = self.engine.run(symbol="QQQ", candles=candles, strategy_filter="vwap")
        self.assertIsInstance(res, BacktestResult)
        self.assertGreater(res.total_trades, 0)
        for t in res.trades:
            self.assertEqual(t.strategy, "vwap")

    def test_run_comparison_vwap_directional(self):
        """Verify side-by-side comparison between vwap and directional_trend."""
        candles = generate_synthetic_candles("QQQ", count=160)
        comp = self.engine.run_comparison(
            symbol="QQQ", candles=candles, strategy_a="vwap", strategy_b="directional_trend"
        )
        self.assertIn("vwap", comp)
        self.assertIn("directional_trend", comp)
        self.assertIsInstance(comp["vwap"], BacktestResult)
        self.assertIsInstance(comp["directional_trend"], BacktestResult)
        self.assertGreater(comp["vwap"].total_trades, 0)
        self.assertGreater(comp["directional_trend"].total_trades, 0)

    def test_ml_alpha_strategy_simulation_and_time_in_cash(self):
        """Verify running backtest with ml_alpha filter and time-in-cash percentage."""
        candles = generate_synthetic_candles("QQQ", count=160)
        res = self.engine.run(symbol="QQQ", candles=candles, strategy_filter="ml_alpha")
        self.assertIsInstance(res, BacktestResult)
        self.assertGreaterEqual(res.time_in_cash_pct, 0.0)
        self.assertLessEqual(res.time_in_cash_pct, 100.0)
        res_dict = res.to_dict()
        self.assertIn("time_in_cash_pct", res_dict)

    def test_run_comparison_directional_and_ml_alpha(self):
        """Verify comparison between directional_trend and ml_alpha."""
        candles = generate_synthetic_candles("QQQ", count=160)
        comp = self.engine.run_comparison(
            symbol="QQQ", candles=candles, strategy_a="directional_trend", strategy_b="ml_alpha"
        )
        self.assertIn("directional_trend", comp)
        self.assertIn("ml_alpha", comp)
        self.assertIsInstance(comp["directional_trend"], BacktestResult)
        self.assertIsInstance(comp["ml_alpha"], BacktestResult)
        self.assertGreaterEqual(comp["directional_trend"].time_in_cash_pct, 0.0)
        self.assertGreaterEqual(comp["ml_alpha"].time_in_cash_pct, 0.0)


if __name__ == "__main__":
    unittest.main()
