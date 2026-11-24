"""Unit tests for TechnicalFallbackScreener and Stale Stop-Loss Healing."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from trading_system.screener.laya_screener import ScreenerResult, TechnicalFallbackScreener
from trading_system.execution.position_monitor import PositionMonitor, TrackedPosition


class TestTechnicalFallbackScreener(unittest.TestCase):
    """Test suite for pure-indicator TechnicalFallbackScreener."""

    def setUp(self) -> None:
        self.screener = TechnicalFallbackScreener(top_n=5, min_adx=20.0)

    def test_bullish_signal_generation(self) -> None:
        """Bullish signal is triggered when price holds above session VWAP with confirmed trend."""
        metrics = {
            "close": 150.0,
            "vwap": 148.0,
            "vwap_dist_pct": 1.35,
            "ema20": 145.0,
            "rsi": 52.0,
            "adx": 28.5,
            "macd": 1.25,
            "macd_signal": 0.85,
            "atr": 3.2,
        }
        results = self.screener.scan_universe({"NVDA": metrics})
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r.ticker, "NVDA")
        self.assertEqual(r.bias, "bullish")
        self.assertIn(r.setup_score, ("favorable", "exceptional"))
        self.assertGreater(r.overall_confidence, 0.7)

    def test_bearish_signal_generation(self) -> None:
        """Bearish signal is triggered for benchmark indices when price breaks below session VWAP."""
        metrics = {
            "close": 90.0,
            "vwap": 91.5,
            "vwap_dist_pct": -1.64,
            "ema20": 98.0,
            "rsi": 42.0,
            "adx": 34.0,
            "macd": -1.5,
            "macd_signal": -0.8,
            "atr": 2.5,
        }
        results = self.screener.scan_universe({"QQQ": metrics})
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r.ticker, "QQQ")
        self.assertEqual(r.bias, "bearish")
        self.assertEqual(r.setup_score, "exceptional")
        self.assertGreaterEqual(r.overall_confidence, 0.85)

    def test_single_equity_bearish_routes_to_cash_defense(self) -> None:
        """Single equity bear impulse routes to Cash Defense (neutral), emitting no trade signal."""
        metrics = {
            "close": 90.0,
            "ema20": 98.0,
            "rsi": 42.0,
            "adx": 34.0,
            "macd": -1.5,
            "macd_signal": -0.8,
            "atr": 2.5,
        }
        results = self.screener.scan_universe({"TSLA": metrics})
        self.assertEqual(len(results), 0)

    def test_low_adx_choppy_filtering(self) -> None:
        """Assets in low ADX consolidation (< 20) are excluded to prevent 3x volatility decay."""
        metrics = {
            "close": 100.0,
            "ema20": 99.0,
            "rsi": 50.0,
            "adx": 15.0,  # Below min_adx
            "macd": 0.5,
            "macd_signal": 0.2,
        }
        results = self.screener.scan_universe({"SPY": metrics})
        self.assertEqual(len(results), 0)

    def test_conflicting_indicators_ignored(self) -> None:
        """Conflicting signals (e.g. close > EMA20 but MACD < signal) produce no signal."""
        metrics = {
            "close": 110.0,
            "ema20": 105.0,
            "rsi": 55.0,
            "adx": 26.0,
            "macd": 0.1,
            "macd_signal": 0.4,  # Bearish MACD despite bullish price
        }
        results = self.screener.scan_universe({"AAPL": metrics})
        self.assertEqual(len(results), 0)


class TestStaleStopLossHealing(unittest.TestCase):
    """Test suite for stale stop healing and entry price reconciliation in PositionMonitor."""

    def setUp(self) -> None:
        self.mock_t212 = MagicMock()
        self.mock_notifier = MagicMock()
        self.mock_risk_mgr = MagicMock()
        self.monitor = PositionMonitor(
            t212_client=self.mock_t212,
            notifier=self.mock_notifier,
            risk_manager=self.mock_risk_mgr,
            state_file="/tmp/test_active_trades.json",
            dry_run=False,
        )

    def test_heal_stale_stop_recomputes_sl(self) -> None:
        """When _heal_stale_stop is called, SL is recomputed from broker live price."""
        pos = TrackedPosition(
            ticker="3BPEl_EQ",
            underlying="BP",
            direction="LONG",
            entry_price=6.80,
            original_quantity=100.0,
            remaining_quantity=100.0,
            stop_loss=1.50,  # Stale, far too low
            trailing_stop_loss=1.50,
            highest_price=6.80,
            lowest_price=6.80,
            tp1=7.50,
            tp2=8.00,
            tp3=8.50,
            atr=0.15,
        )

        mock_broker_pos = MagicMock()
        mock_broker_pos.ticker = "3BPEl_EQ"
        mock_broker_pos.current_price = 6.90
        mock_broker_pos.average_price = 6.85
        self.mock_t212.get_portfolio.return_value = [mock_broker_pos]

        self.monitor._heal_stale_stop(pos)

        # SL should now be around 6.90 * (1 - 0.02 * 2.5) = 6.90 * 0.95 = 6.555
        self.assertAlmostEqual(pos.stop_loss, 6.555, delta=0.05)
        self.assertAlmostEqual(pos.trailing_stop_loss, 6.555, delta=0.05)
        self.assertEqual(getattr(pos, "_last_stop_fail_time", -1), 0.0)

    def test_reconcile_fixes_entry_price_currency_scale(self) -> None:
        """Reconciliation corrects stored entry_price if it differs >50x from broker avg price."""
        pos = TrackedPosition(
            ticker="3XOMl_EQ",
            underlying="XOM",
            direction="LONG",
            entry_price=178.0,  # Stored in GBX scale or wrong currency
            original_quantity=10.0,
            remaining_quantity=10.0,
            stop_loss=3000.0,
            trailing_stop_loss=3000.0,
            highest_price=3200.0,
            lowest_price=3100.0,
            tp1=3400.0,
            tp2=3600.0,
            tp3=3800.0,
            atr=50.0,
        )
        self.monitor.positions["3XOMl_EQ"] = pos

        mock_broker_pos = MagicMock()
        mock_broker_pos.ticker = "3XOMl_EQ"
        mock_broker_pos.quantity = 10.0
        mock_broker_pos.current_price = 3200.0
        mock_broker_pos.average_price = 3190.0  # Real broker avg price in EUR
        self.mock_t212.get_portfolio.return_value = [mock_broker_pos]
        self.mock_t212.get_orders.return_value = []

        summary = self.monitor.reconcile_with_broker()
        self.assertEqual(pos.entry_price, 3190.0)


if __name__ == "__main__":
    unittest.main()
