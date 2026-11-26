"""Unit tests for 5-minute intraday candle ingestion, session VWAP, and ML Alpha bars."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from trading_system.data.candle_db import CandleDatabase
from trading_system.data.multi_provider import MultiProviderRouter
from trading_system.data.trading212_client import MarketDataProvider
from trading_system.orchestrator import Orchestrator
from trading_system.screener.laya_screener import LayaScreener
from trading_system.strategies.registry import StrategyRegistry


def _make_daily_bars(n: int = 60, start_price: float = 100.0, step: float = 0.2) -> list[dict]:
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    bars = []
    px = start_price
    for i in range(n):
        px = px + step
        bars.append(
            {
                "date": (start + timedelta(days=i)).strftime("%Y-%m-%d"),
                "open": round(px - 0.3, 2),
                "high": round(px + 0.8, 2),
                "low": round(px - 0.8, 2),
                "close": round(px, 2),
                "volume": 1_500_000.0,
            }
        )
    return bars


class Test5mStrategyPipeline(unittest.TestCase):
    """Verify 5m intraday candle retrieval, indicator anchoring, and strategy firing."""

    def setUp(self):
        MarketDataProvider.clear_candle_cache()
        MarketDataProvider.SYNTHETIC_TICKERS.clear()
        MarketDataProvider.LAST_FETCH_WAS_SYNTHETIC = False

    def tearDown(self):
        MarketDataProvider.clear_candle_cache()
        MarketDataProvider.SYNTHETIC_TICKERS.clear()
        MarketDataProvider.LAST_FETCH_WAS_SYNTHETIC = False
        CandleDatabase.reset_instance()

    def test_multi_provider_5m_cascade_failover(self):
        """get_5m_candles_cascade tries FMP 5min and fails over to Polygon 5/minute on 429."""
        router = MultiProviderRouter()
        mock_settings = type(
            "S",
            (),
            {"FMP_API_KEY": "fmp_k", "POLYGON_API_KEY": "poly_k"},
        )()

        router.record_response("fmp", 429, error_msg="Rate limited")
        expected = [
            {
                "date": "2026-11-26 14:35:00",
                "open": 500.0,
                "high": 501.0,
                "low": 499.5,
                "close": 500.8,
                "volume": 25000.0,
            }
        ]
        with patch.object(router, "fetch_polygon_5m", return_value=expected) as mock_poly:
            with patch.object(router, "fetch_fmp_5m") as mock_fmp:
                res = router.get_5m_candles_cascade("QQQ", mock_settings)
                self.assertEqual(res, expected)
                mock_fmp.assert_not_called()
                mock_poly.assert_called_once_with("QQQ", "poly_k")

    def test_get_5m_candles_caches_and_preserves_dates(self):
        """get_5m_candles stores 5m bars with dates in CandleDatabase and fast-paths fresh bars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "candles.db")
            CandleDatabase.reset_instance()
            cdb = CandleDatabase.get_instance(db_path=db_path)

            now_utc = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            now_utc = now_utc.replace(minute=(now_utc.minute // 5) * 5)
            recent_bars = [
                {
                    "date": (now_utc - timedelta(minutes=5 * (34 - i))).strftime("%Y-%m-%d %H:%M:%S"),
                    "open": 100.0 + i * 0.1,
                    "high": 100.5 + i * 0.1,
                    "low": 99.8 + i * 0.1,
                    "close": 100.2 + i * 0.1,
                    "volume": 10000.0,
                }
                for i in range(35)
            ]

            with patch.object(MarketDataProvider, "_fetch_yahoo_chart_data", return_value=recent_bars) as mock_yahoo:
                first = MarketDataProvider.get_5m_candles("TEST5M", force_refresh=True)
                self.assertGreaterEqual(len(first), 35)
                mock_yahoo.assert_called_once()
                if hasattr(first, "columns"):
                    self.assertIn("date", first.columns)
                else:
                    self.assertIn("date", first[0])

            MarketDataProvider.clear_candle_cache()
            # Latest bar is < 4 minutes old, so CandleDatabase fast-path serves with 0 HTTP calls
            with patch.object(MarketDataProvider, "_fetch_yahoo_chart_data") as mock_yahoo2:
                second = MarketDataProvider.get_5m_candles("TEST5M")
                mock_yahoo2.assert_not_called()
                self.assertGreaterEqual(len(second), 35)

            self.assertEqual(cdb.get_candle_count("TEST5M", "5m"), 35)

    def test_compute_indicators_uses_5m_close_session_vwap_and_bars(self):
        """compute_indicators uses latest 5m close, today's session VWAP, 5m ADX, and serializes bars."""
        daily = _make_daily_bars(n=60, start_price=90.0, step=0.1)  # daily close ends around 96.0
        prev_day = "2026-11-25"
        curr_day = "2026-11-26"
        intraday_5m = []
        # Previous day bars much lower (around 80.0) that must NOT pull down today's session VWAP
        for i in range(20):
            hh = 10 + (i * 5) // 60
            mm = (i * 5) % 60
            intraday_5m.append(
                {
                    "date": f"{prev_day} {hh:02d}:{mm:02d}:00",
                    "open": 80.0,
                    "high": 80.5,
                    "low": 79.5,
                    "close": 80.0,
                    "volume": 500_000.0,
                }
            )
        # Today's session 5m bars trending steadily around 100 -> 102.5
        for i in range(30):
            hh = 8 + (i * 5) // 60
            mm = (i * 5) % 60
            px = 100.0 + i * 0.08
            intraday_5m.append(
                {
                    "date": f"{curr_day} {hh:02d}:{mm:02d}:00",
                    "open": round(px - 0.05, 2),
                    "high": round(px + 0.10, 2),
                    "low": round(px - 0.10, 2),
                    "close": round(px, 2),
                    "volume": 20_000.0,
                }
            )

        metrics = MarketDataProvider.compute_indicators(daily, intraday_data=intraday_5m, symbol="TEST")
        self.assertEqual(metrics["timeframe"], "5m")
        self.assertAlmostEqual(metrics["close"], intraday_5m[-1]["close"], places=2)
        # Session VWAP must stay within today's 100..102.5 range (not dragged down to 80s by prior day)
        self.assertGreater(metrics["vwap"], 99.5)
        self.assertLess(metrics["vwap_dist_pct"], 3.5)
        self.assertGreater(metrics["vwap_dist_pct"], 0.1)
        self.assertGreater(metrics["adx"], 18.0)
        self.assertIn("bars", metrics)
        self.assertEqual(len(metrics["bars"]), 50)
        self.assertEqual(metrics["bars"][-1]["date"], intraday_5m[-1]["date"])

    def test_ml_alpha_and_vwap_produce_differentiated_actionable_signals_with_5m(self):
        """Both vwap and ml_alpha evaluate 5m computed metrics cleanly without degenerate collapse."""
        symbols = ["SPY", "NVDA", "QQQ", "SOXX", "AAPL", "GOOGL", "TLT"]
        universe_metrics = {}
        for sym in symbols:
            d = MarketDataProvider._generate_synthetic_candles(sym, interval="1d")
            m5 = MarketDataProvider._generate_synthetic_candles(sym, interval="5m")
            universe_metrics[sym] = MarketDataProvider.compute_indicators(d, intraday_data=m5, symbol=sym)

        vwap_strat = StrategyRegistry.get("vwap")
        vwap_signals = [vwap_strat.evaluate_candidate(s, m) for s, m in universe_metrics.items()]
        vwap_actionable = [sig for sig in vwap_signals if sig.is_actionable]
        self.assertGreater(len(vwap_actionable), 0)
        self.assertTrue(all(sig.timeframe == "5m" for sig in vwap_actionable))

        ml_strat = StrategyRegistry.get("ml_alpha")
        ml_signals = [ml_strat.evaluate_candidate(s, m) for s, m in universe_metrics.items()]
        unique_p_long = {round(float(sig.metrics.get("p_long", 0.0)), 4) for sig in ml_signals}
        self.assertGreater(len(unique_p_long), 1, "MLAlpha probabilities collapsed to identical fallback values")
        self.assertTrue(any(sig.is_actionable for sig in ml_signals))

        screener_vwap = LayaScreener(strategy_name="vwap")
        self.assertGreater(len(screener_vwap.scan_universe(universe_metrics)), 0)

    def test_orchestrator_ingests_5m_candles_and_broadcasts_5m_step(self):
        """Orchestrator uses get_5m_candles for universe + active trades and broadcasts 5m label."""
        orch = Orchestrator(dry_run=True, enable_dashboard=False)
        daily = _make_daily_bars(n=60)
        m5 = MarketDataProvider._generate_synthetic_candles("SPY", interval="5m")

        events = []
        with patch.object(orch, "_broadcast_sse", side_effect=lambda ev, data: events.append((ev, data))):
            with patch("trading_system.orchestrator.Trading212Client.is_asset_session_active", side_effect=lambda s: s == "SPY"):
                with patch("trading_system.orchestrator.MarketDataProvider.get_daily_candles", return_value=daily):
                    with patch("trading_system.orchestrator.MarketDataProvider.get_5m_candles", return_value=m5) as mock_5m:
                        with patch("trading_system.orchestrator.MarketDataProvider.get_days_to_earnings", return_value=None):
                            orch._scan_and_deliberate()
                            mock_5m.assert_called_with("SPY", period="5d")

        step2 = [d for ev, d in events if ev == "pipeline_progress" and d.get("step") == 2]
        self.assertTrue(step2)
        self.assertIn("5m", step2[0]["detail"])


if __name__ == "__main__":
    unittest.main()
