"""Unit tests for standalone strategy diagnostics tool (scripts/diagnose_strategy.py)."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
import unittest

from scripts.diagnose_strategy import (
    TradeRecord,
    apply_friction_and_normalize,
    calculate_exit_attribution,
    calculate_headline_edge,
    calculate_mfe_mae_audit,
    calculate_segment_breakdown,
    classify_asset_tier,
    enrich_trades_with_db,
    generate_bottleneck_verdict,
    load_trades_from_file,
    map_exit_reason,
    normalize_trade,
    run_diagnostics,
)


class TestDiagnoseStrategy(unittest.TestCase):
    """Comprehensive test suite for strategy diagnostics, friction math, and edge audits."""

    def test_gross_vs_net_friction_math(self):
        """Verify round-trip LSE ETP friction (default 45 bps) math and win-to-loss flips."""
        # A trade with +0.30% gross gain on $10,000 notional:
        # Gross PnL = +$30.00
        # Friction at 45.0 bps = $10,000 * 0.0045 = $45.00
        # Net PnL = +$30.00 - $45.00 = -$15.00
        # Net Return % = +0.30% - 0.45% = -0.15% (flips to Net Loss)
        raw_trade = {
            "symbol": "QQQ",
            "direction": "LONG",
            "entry_price": 100.0,
            "exit_price": 100.30,
            "shares": 100.0,
            "notional": 10000.0,
            "realized_pnl": 30.0,
            "return_pct": 0.30,
            "initial_sl": 97.5,
            "exit_reason": "TP1_SL_BE",
        }

        trade_45bps = normalize_trade(raw_trade, cost_bps=45.0)
        self.assertEqual(trade_45bps.gross_pnl, 30.0)
        self.assertAlmostEqual(trade_45bps.gross_return_pct, 0.30, places=4)
        self.assertAlmostEqual(trade_45bps.net_pnl, -15.0, places=4)
        self.assertAlmostEqual(trade_45bps.net_return_pct, -0.15, places=4)
        self.assertGreater(trade_45bps.gross_pnl, 0.0)
        self.assertLess(trade_45bps.net_pnl, 0.0)

        # Zero friction test
        trade_0bps = normalize_trade(raw_trade, cost_bps=0.0)
        self.assertEqual(trade_0bps.net_pnl, 30.0)
        self.assertAlmostEqual(trade_0bps.net_return_pct, 0.30, places=4)

        # 100 bps friction test (1.0% friction)
        trade_100bps = normalize_trade(raw_trade, cost_bps=100.0)
        self.assertAlmostEqual(trade_100bps.net_pnl, -70.0, places=4)
        self.assertAlmostEqual(trade_100bps.net_return_pct, -0.70, places=4)

    def test_expectancy_and_headline_metrics_calculation(self):
        """Verify Expectancy (%, $R$), Win Rate, Payoff Ratio, Profit Factor, and Drawdown."""
        # Create 3 synthetic trades:
        # Trade 1: +5.0% gross (+4.55% net), notional 1000, net pnl +45.50, risk 25.0 -> R = +1.82
        # Trade 2: -2.0% gross (-2.45% net), notional 1000, net pnl -24.50, risk 25.0 -> R = -0.98
        # Trade 3: +1.0% gross (+0.55% net), notional 1000, net pnl +5.50, risk 25.0 -> R = +0.22
        t1 = normalize_trade({
            "symbol": "NVDA", "direction": "LONG", "entry_price": 100.0, "exit_price": 105.0,
            "notional": 1000.0, "shares": 10.0, "initial_sl": 97.5, "return_pct": 5.0, "realized_pnl": 50.0,
        }, cost_bps=45.0)

        t2 = normalize_trade({
            "symbol": "TSLA", "direction": "LONG", "entry_price": 100.0, "exit_price": 98.0,
            "notional": 1000.0, "shares": 10.0, "initial_sl": 97.5, "return_pct": -2.0, "realized_pnl": -20.0,
        }, cost_bps=45.0)

        t3 = normalize_trade({
            "symbol": "AAPL", "direction": "LONG", "entry_price": 100.0, "exit_price": 101.0,
            "notional": 1000.0, "shares": 10.0, "initial_sl": 97.5, "return_pct": 1.0, "realized_pnl": 10.0,
        }, cost_bps=45.0)

        trades = [t1, t2, t3]
        headline = calculate_headline_edge(trades, initial_capital=10000.0)

        self.assertEqual(headline["trade_count"], 3)
        self.assertAlmostEqual(headline["gross_win_rate_pct"], (2 / 3) * 100.0, places=2)
        self.assertAlmostEqual(headline["net_win_rate_pct"], (2 / 3) * 100.0, places=2)

        # Expected Net Expectancy % = (4.55 - 2.45 + 0.55) / 3 = 2.65 / 3 = 0.8833%
        expected_net_exp_pct = (4.55 - 2.45 + 0.55) / 3.0
        self.assertAlmostEqual(headline["net_expectancy_pct"], expected_net_exp_pct, places=2)

        # Expectancy in R: (45.5 - 24.5 + 5.5) / (25 * 3) = 26.5 / 75 = 0.3533 R
        self.assertAlmostEqual(headline["net_expectancy_r"], 26.5 / 75.0, places=2)

        # Profit Factor: Net Wins / Net Losses = (45.50 + 5.50) / 24.50 = 51.0 / 24.50 = 2.0816
        self.assertAlmostEqual(headline["net_profit_factor"], 51.0 / 24.50, places=2)

        # Drawdown is non-negative
        self.assertGreaterEqual(headline["net_max_drawdown_pct"], 0.0)

    def test_exit_reason_bucketing(self):
        """Verify exact auto-mapping of raw reasons to the 6 canonical buckets."""
        mappings = [
            ("SL", "INITIAL_GTC_STOP"),
            ("INITIAL_SL", "INITIAL_GTC_STOP"),
            ("stop_loss", "INITIAL_GTC_STOP"),
            ("TSL_BREAKEVEN", "BREAKEVEN_RATCHET"),
            ("TP1_SL_BE", "BREAKEVEN_RATCHET"),
            ("breakeven", "BREAKEVEN_RATCHET"),
            ("TSL_CUSHION", "PROFIT_CUSHION"),
            ("profit_cushion", "PROFIT_CUSHION"),
            ("TP2_TSL", "CHANDELIER_TRAIL"),
            ("TSL", "CHANDELIER_TRAIL"),
            ("dynamic_atr", "CHANDELIER_TRAIL"),
            ("ASYMMETRIC_SESSION_EXIT", "EVACUATION_1620_UTC"),
            ("ZERO_ALPHA_EVACUATION", "EVACUATION_1620_UTC"),
            ("inverse_intraday_flatten", "EVACUATION_1620_UTC"),
            ("1620_UTC_EVACUATION", "EVACUATION_1620_UTC"),
            ("TP3", "OTHER"),
            ("MAX_HOLD", "OTHER"),
            ("END_OF_DATA", "OTHER"),
            ("CUSTOM_EXIT", "OTHER"),
            ("", "OTHER"),
            (None, "OTHER"),
        ]
        for raw, expected in mappings:
            self.assertEqual(map_exit_reason(raw), expected, f"Failed mapping for {raw}")

        # Test exit attribution table aggregation
        t_be = normalize_trade({"symbol": "QQQ", "exit_reason": "TSL_BREAKEVEN", "realized_pnl": 0.0, "return_pct": 0.0})
        t_sl = normalize_trade({"symbol": "SPY", "exit_reason": "INITIAL_SL", "realized_pnl": -50.0, "return_pct": -5.0})
        t_tsl = normalize_trade({"symbol": "NVDA", "exit_reason": "TP2_TSL", "realized_pnl": 100.0, "return_pct": 10.0})

        attr = calculate_exit_attribution([t_be, t_sl, t_tsl])
        attr_map = {row["exit_reason"]: row for row in attr}

        self.assertEqual(attr_map["BREAKEVEN_RATCHET"]["trade_count"], 1)
        self.assertEqual(attr_map["INITIAL_GTC_STOP"]["trade_count"], 1)
        self.assertEqual(attr_map["CHANDELIER_TRAIL"]["trade_count"], 1)
        self.assertEqual(attr_map["PROFIT_CUSHION"]["trade_count"], 0)
        self.assertEqual(attr_map["EVACUATION_1620_UTC"]["trade_count"], 0)

    def test_mfe_mae_and_post_exit_continuation_sqlite(self):
        """Verify intrabar MFE/MAE calculation and post-exit 24-48h continuation against SQLite candles."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name

        try:
            # Create schema and synthetic candle data
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE candles (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (symbol, timeframe, date)
                );
            """)

            # Symbol TEST:
            # 2025-01-01: Entry bar at 100
            # 2025-01-02: High 110, Low 95
            # 2025-01-03: Exit bar (Exit at 100, Breakeven)
            # 2025-01-04: Post-exit bar 1: High 105 (+5% move relative to exit 100)
            # 2025-01-05: Post-exit bar 2: High 106
            rows = [
                ("TEST", "1d", "2025-01-01", 100.0, 102.0, 99.0, 101.0, 1000.0),
                ("TEST", "1d", "2025-01-02", 101.0, 110.0, 95.0, 104.0, 1000.0),
                ("TEST", "1d", "2025-01-03", 104.0, 104.0, 99.0, 100.0, 1000.0),
                ("TEST", "1d", "2025-01-04", 100.0, 105.0, 99.0, 104.0, 1000.0),
                ("TEST", "1d", "2025-01-05", 104.0, 106.0, 103.0, 105.0, 1000.0),
            ]
            conn.executemany("INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
            conn.commit()
            conn.close()

            # Test LONG trade:
            # Entry 100, Exit 100, Intrabar High = 110 (+10%), Low = 95 (-5%)
            long_trade = normalize_trade({
                "symbol": "TEST",
                "direction": "LONG",
                "entry_price": 100.0,
                "exit_price": 100.0,
                "entry_date": "2025-01-01",
                "exit_date": "2025-01-03",
                "exit_reason": "BREAKEVEN_RATCHET",
            })

            enriched = enrich_trades_with_db([long_trade], db_path)
            self.assertEqual(len(enriched), 1)
            t = enriched[0]

            self.assertIsNotNone(t.mfe_pct)
            self.assertAlmostEqual(t.mfe_pct, 10.0, places=2)  # (110 - 100)/100 = +10%
            self.assertIsNotNone(t.mae_pct)
            self.assertAlmostEqual(t.mae_pct, -5.0, places=2)  # (95 - 100)/100 = -5%

            # Post-exit continuation: Next day reached 105, which is +5% >= +3% above exit 100
            self.assertTrue(t.post_exit_run_3pct)

            # Test audit aggregation
            audit = calculate_mfe_mae_audit([t])
            self.assertAlmostEqual(audit["mean_mfe_pct"], 10.0, places=2)
            self.assertAlmostEqual(audit["mean_mae_pct"], -5.0, places=2)
            self.assertAlmostEqual(audit["edge_ratio_mean"], 2.0, places=2)  # 10 / 5 = 2.0
            self.assertEqual(audit["be_choke_count"], 1)
            self.assertEqual(audit["be_choke_pct"], 100.0)

        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_asset_tier_classification(self):
        """Verify accurate tier segmentation: 3x Index, 5x Index, 3x Single Stock, Commodity/Bond."""
        self.assertEqual(classify_asset_tier("QQQ", leverage=3), "3x Index")
        self.assertEqual(classify_asset_tier("QQQ", leverage=5), "5x Index")
        self.assertEqual(classify_asset_tier("5QQEl_EQ", leverage=5), "5x Index")
        self.assertEqual(classify_asset_tier("SPY", leverage=3), "3x Index")
        self.assertEqual(classify_asset_tier("NVDA", leverage=3), "3x Single Stock")
        self.assertEqual(classify_asset_tier("AAPL", leverage=3), "3x Single Stock")
        self.assertEqual(classify_asset_tier("ARM", leverage=3), "3x Single Stock")
        self.assertEqual(classify_asset_tier("GLD", leverage=3), "Commodity/Bond")
        self.assertEqual(classify_asset_tier("USO", leverage=3), "Commodity/Bond")
        self.assertEqual(classify_asset_tier("TLT", leverage=3), "Commodity/Bond")
        self.assertEqual(classify_asset_tier("3WHLl_EQ", leverage=3), "Commodity/Bond")

    def test_graceful_handling_empty_and_missing_fields(self):
        """Verify zero division crashes and robust fallbacks on empty or malformed data."""
        # 1. Empty list
        headline = calculate_headline_edge([])
        self.assertEqual(headline["trade_count"], 0)
        self.assertEqual(headline["net_win_rate_pct"], 0.0)
        self.assertEqual(headline["net_expectancy_pct"], 0.0)

        attr = calculate_exit_attribution([])
        self.assertEqual(len(attr), 6)
        self.assertTrue(all(r["trade_count"] == 0 for r in attr))

        mfe_mae = calculate_mfe_mae_audit([])
        self.assertEqual(mfe_mae["mfe_count"], 0)
        self.assertIsNone(mfe_mae["be_choke_pct"])

        segments = calculate_segment_breakdown([])
        self.assertEqual(len(segments["direction"]), 2)
        self.assertEqual(len(segments["asset_tier"]), 4)

        verdict = generate_bottleneck_verdict(headline, attr, mfe_mae, segments, cost_bps=45.0)
        self.assertIsInstance(verdict, list)

        # 2. Trade with completely missing optional fields
        sparse_trade = normalize_trade({
            "ticker": "XYZ",
        })
        self.assertEqual(sparse_trade.symbol, "XYZ")
        self.assertEqual(sparse_trade.shares, 1.0)
        self.assertEqual(sparse_trade.notional, 1000.0)  # Standard fallback
        self.assertEqual(sparse_trade.gross_pnl, 0.0)
        self.assertEqual(sparse_trade.net_pnl, -4.5)  # 1000 * 0.0045 friction
        self.assertEqual(sparse_trade.exit_reason, "OTHER")

        # 3. File loading on empty or non-existent file
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            empty_path = tf.name
        try:
            trades = load_trades_from_file(empty_path)
            self.assertEqual(trades, [])
        finally:
            if os.path.exists(empty_path):
                os.remove(empty_path)

        with self.assertRaises(FileNotFoundError):
            load_trades_from_file("non_existent_file_path.jsonl")

    def test_bottleneck_verdict_triggers(self):
        """Verify automated bottleneck flags trigger on friction fragility and high choke rates."""
        # Setup: Strategy has positive gross expectancy (+0.20%) but negative net expectancy (-0.25%)
        # and high choke rate (50% Breakeven ratchets)
        t_gross_win = normalize_trade({
            "symbol": "QQQ", "direction": "LONG", "notional": 10000.0,
            "realized_pnl": 50.0, "return_pct": 0.50, "exit_reason": "TSL_BREAKEVEN",
        }, cost_bps=45.0)

        t_loss = normalize_trade({
            "symbol": "SPY", "direction": "SHORT", "notional": 10000.0,
            "realized_pnl": -10.0, "return_pct": -0.10, "exit_reason": "INITIAL_SL",
        }, cost_bps=45.0)

        trades = [t_gross_win, t_loss]
        diag = run_diagnostics(trades, cost_bps=45.0)
        verdict = diag["bottleneck_verdict"]

        # Expect friction fragility flag
        fragility_flags = [f for f in verdict if "Friction Fragility" in f or "Friction Drag" in f]
        self.assertGreater(len(fragility_flags), 0)

        # Expect early-exit choke flag (TSL_BREAKEVEN is 1 out of 2 = 50% > 35%)
        choke_flags = [f for f in verdict if "Early-Exit Choke" in f]
        self.assertGreater(len(choke_flags), 0)


if __name__ == "__main__":
    unittest.main()
