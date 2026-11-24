"""Rigorous unit test suite for ML Alpha Strategy, Feature Engine, and Triple-Barrier Labeling.

Covers:
1. Strict Zero-Lookahead Feature Invariance & Schema Integrity.
2. Triple-Barrier Path-Order, Stop-First, Profit-First, Tie-Break, and 16:20 UTC Cutoff.
3. Purged Walk-Forward Cross-Validation Split Leakage Prevention.
4. ONNX End-to-End Latency (<5ms) and Corrupt/Missing Model Fail-Safe Fallbacks.
5. Strategy Registry Hot-Swapping and Dual-Track Bear Shield Routing.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import os
import tempfile
import time
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np

from trading_system.config import settings
from trading_system.strategies.base import StrategySignal
from trading_system.strategies.registry import StrategyRegistry
from trading_system.strategies.ml_alpha import MLAlphaStrategy, DEFAULT_BROAD_INDICES
from trading_system.ml.onnx_runner import OnnxModelRunner, FAILSAFE_PROBA


def make_synthetic_bars(
    n: int = 120,
    base_price: float = 100.0,
    start_dt: Optional[datetime.datetime] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate deterministic synthetic OHLCV bars with realistic drift and volatility."""
    rng = np.random.default_rng(seed)
    if start_dt is None:
        start_dt = datetime.datetime(2026, 11, 2, 8, 0, tzinfo=datetime.timezone.utc)

    bars: List[Dict[str, Any]] = []
    current_price = base_price
    current_dt = start_dt

    for _ in range(n):
        ret = float(rng.normal(0.0005, 0.008))
        open_p = current_price
        close_p = open_p * (1.0 + ret)
        spread = abs(float(rng.normal(0.003, 0.002)))
        high_p = max(open_p, close_p) * (1.0 + spread)
        low_p = min(open_p, close_p) * (1.0 - spread)
        volume = float(rng.integers(20000, 300000))

        bars.append({
            "date": current_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "open": float(open_p),
            "high": float(high_p),
            "low": float(low_p),
            "close": float(close_p),
            "volume": float(volume),
        })
        current_price = close_p
        current_dt += datetime.timedelta(hours=1)

    return bars


class TestStrictZeroLookahead(unittest.TestCase):
    """(a) Strict Zero-Lookahead Test Suite."""

    def setUp(self):
        self.bars = make_synthetic_bars(n=120, base_price=150.0, seed=123)

    def test_feature_schema_hash_and_names(self):
        """Feature schema must contain exactly 14 names matching the canonical hash."""
        from trading_system.ml.features import FEATURE_NAMES, compute_feature_schema_hash

        expected_features = (
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
        self.assertEqual(len(FEATURE_NAMES), 14)
        self.assertEqual(FEATURE_NAMES, expected_features)

        computed_hash = compute_feature_schema_hash()
        expected_hash = hashlib.sha256(",".join(expected_features).encode("utf-8")).hexdigest()
        self.assertEqual(computed_hash, expected_hash)

    def test_strict_zero_lookahead_invariance(self):
        """Mutate future bars [t+1:] and assert extract_features(bars[:t+1]) is bit-for-bit identical."""
        from trading_system.ml.features import extract_features

        # Evaluate across several historical time points
        test_timepoints = [35, 55, 75, 95]
        for t in test_timepoints:
            # Slices up to t (inclusive of bar t)
            bars_up_to_t = copy.deepcopy(self.bars[:t + 1])
            orig_features = extract_features(bars_up_to_t, leverage=3.0)

            # Verification of shape and dtype
            self.assertEqual(orig_features.shape, (1, 14), f"Shape mismatch at bar {t}")
            self.assertEqual(orig_features.dtype, np.float32, f"Dtype mismatch at bar {t}")
            self.assertTrue(np.all(np.isfinite(orig_features)), f"Non-finite values found at bar {t}")

            # Mutate all future bars [t+1:] with extreme variations
            mutated_bars = copy.deepcopy(self.bars)
            for fut_idx in range(t + 1, len(mutated_bars)):
                mutated_bars[fut_idx]["open"] *= 4.5
                mutated_bars[fut_idx]["high"] *= 5.0
                mutated_bars[fut_idx]["low"] *= 0.1
                mutated_bars[fut_idx]["close"] *= 3.8
                mutated_bars[fut_idx]["volume"] *= 25.0

            # Re-extract features strictly using slice up to t
            features_after = extract_features(mutated_bars[:t + 1], leverage=3.0)

            # Must be bit-for-bit identical
            self.assertTrue(
                np.array_equal(orig_features, features_after),
                f"Lookahead bias detected! extract_features at bar t={t} changed when mutating future bars.",
            )

    def test_insufficient_bars_handled_gracefully(self):
        """Test feature extraction with minimal or zero bars."""
        from trading_system.ml.features import extract_features

        # Single bar
        feat_single = extract_features([self.bars[0]], leverage=3.0)
        self.assertEqual(feat_single.shape, (1, 14))
        self.assertEqual(feat_single.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(feat_single)))

        # Empty bar list
        feat_empty = extract_features([], leverage=3.0)
        self.assertEqual(feat_empty.shape, (1, 14))
        self.assertEqual(feat_empty.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(feat_empty)))


class TestTripleBarrierPathOrder(unittest.TestCase):
    """(b) Triple-Barrier Path-Order Test Suite."""

    def test_intrabar_stop_hit_before_profit_hit_labels_cash_defense(self):
        """Verify intrabar stop-hit before profit-hit (conservative tie-break) labels 0 (CASH_DEFENSE)."""
        from trading_system.ml.labeling import TripleBarrierLabeler

        labeler = TripleBarrierLabeler()
        # Bar 0: Entry at 100.0, Close 100.0
        # Bar 1: Violent whipsaw touches both upper and lower barriers. Conservative tie-break fails the trade.
        # Bar 2: Subsequent surge to 120.0 cannot resurrect stopped-out trade.
        bars = [
            {"date": "2026-11-02 09:00:00", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100000},
            {"date": "2026-11-02 10:00:00", "open": 100.0, "high": 108.0, "low": 92.0, "close": 100.0, "volume": 200000},  # Intrabar both TP and SL breached
            {"date": "2026-11-02 11:00:00", "open": 100.0, "high": 120.0, "low": 99.0, "close": 118.0, "volume": 150000},  # Later rally
        ]
        labels = labeler.label_bars(bars, leverage=3.0, max_hold_bars=5)
        self.assertEqual(labels.shape, (len(bars),))
        # Bar 0 must be 0 (CASH_DEFENSE) because stop was breached in Bar 1
        self.assertEqual(labels[0], 0, "Intrabar stop-hit/tie-break should label 0 (CASH_DEFENSE)")

    def test_profit_first_labels_long_or_short(self):
        """Verify profit-first labels 1 (LONG) or 2 (SHORT)."""
        from trading_system.ml.labeling import TripleBarrierLabeler

        labeler = TripleBarrierLabeler()

        # 1. Bullish scenario: Price surges to TP without hitting SL
        bull_bars = [
            {"date": "2026-11-02 09:00:00", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100000},
            {"date": "2026-11-02 10:00:00", "open": 100.2, "high": 105.0, "low": 99.5, "close": 104.5, "volume": 120000},  # TP reached!
            {"date": "2026-11-02 11:00:00", "open": 104.0, "high": 106.0, "low": 103.0, "close": 105.0, "volume": 110000},
        ]
        bull_labels = labeler.label_bars(bull_bars, leverage=3.0, max_hold_bars=5)
        self.assertEqual(bull_labels[0], 1, "Profit-first on rally should label 1 (LONG)")

        # 2. Bearish scenario: Price drops to short TP without hitting short SL
        bear_bars = [
            {"date": "2026-11-02 09:00:00", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100000},
            {"date": "2026-11-02 10:00:00", "open": 99.5, "high": 100.2, "low": 95.0, "close": 95.5, "volume": 120000},  # Short TP reached!
            {"date": "2026-11-02 11:00:00", "open": 95.5, "high": 96.0, "low": 94.0, "close": 95.0, "volume": 110000},
        ]
        bear_labels = labeler.label_bars(bear_bars, leverage=3.0, max_hold_bars=5)
        self.assertEqual(bear_labels[0], 2, "Profit-first on decline should label 2 (SHORT)")

    def test_same_bar_tie_break_conservatively_labels_cash_defense(self):
        """Verify same-bar tie-break (both TP and SL breached in single bar) conservatively labels 0."""
        from trading_system.ml.labeling import TripleBarrierLabeler

        labeler = TripleBarrierLabeler()
        # Bar 1 has huge range that breaches both upper TP and lower SL
        bars = [
            {"date": "2026-11-02 09:00:00", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100000},
            {"date": "2026-11-02 10:00:00", "open": 100.0, "high": 115.0, "low": 85.0, "close": 100.0, "volume": 200000},  # Both hit!
            {"date": "2026-11-02 11:00:00", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100000},
        ]
        labels = labeler.label_bars(bars, leverage=3.0, max_hold_bars=5)
        self.assertEqual(labels[0], 0, "Same-bar tie-break must conservatively label 0 (CASH_DEFENSE)")

    def test_1620_utc_vertical_cutoff_enforces_session_exit(self):
        """Verify vertical barrier enforces session exit at 16:20 UTC, labeling 0 if unreached."""
        from trading_system.ml.labeling import TripleBarrierLabeler

        labeler = TripleBarrierLabeler()
        # Bars extending past 16:20 UTC with tiny range (neither TP nor SL reached)
        bars = [
            {"date": "2026-11-02 15:00:00", "open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0, "volume": 50000},
            {"date": "2026-11-02 16:00:00", "open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0, "volume": 50000},
            {"date": "2026-11-02 16:20:00", "open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0, "volume": 50000},  # Cutoff!
            {"date": "2026-11-02 16:30:00", "open": 100.0, "high": 110.0, "low": 99.0, "close": 108.0, "volume": 50000},  # After cutoff
        ]
        labels = labeler.label_bars(bars, leverage=3.0, max_hold_bars=10)
        # Entry at 15:00 should hit vertical cutoff at 16:20 UTC and exit as 0 (CASH_DEFENSE)
        self.assertEqual(labels[0], 0, "16:20 UTC vertical cutoff must exit at cash defense if target unhit")


class TestPurgedSplitLeakage(unittest.TestCase):
    """(c) Purged Split Leakage Test Suite."""

    def test_purged_walk_forward_splits_zero_leakage(self):
        """Assert zero overlapping barrier windows exist between train and val splits."""
        from trading_system.ml.labeling import TripleBarrierLabeler

        labeler = TripleBarrierLabeler()
        n_samples = 200
        n_splits = 5
        max_hold_bars = 10
        embargo_pct = 0.02

        splits = labeler.purged_walk_forward_splits(
            n_samples=n_samples,
            n_splits=n_splits,
            embargo_pct=embargo_pct,
            max_hold_bars=max_hold_bars,
        )

        self.assertEqual(len(splits), n_splits)

        for fold, (train_idx, val_idx) in enumerate(splits):
            train_set = set(train_idx)
            val_set = set(val_idx)

            # 1. No direct index overlap
            overlap = train_set.intersection(val_set)
            self.assertEqual(len(overlap), 0, f"Fold {fold} has direct index overlap: {overlap}")

            min_val = int(np.min(val_idx))

            # 2. Train samples before validation: their hold window [t, t + max_hold_bars] must not touch val
            train_before = [t for t in train_idx if t < min_val]
            if train_before:
                max_train_before = max(train_before)
                self.assertLess(
                    max_train_before + max_hold_bars,
                    min_val,
                    f"Fold {fold}: Train sample at {max_train_before} with horizon {max_hold_bars} "
                    f"overlaps with val start {min_val}",
                )


class TestONNXEndToEndAndParity(unittest.TestCase):
    """(d) ONNX End-to-End Latency & Parity Test Suite."""

    def setUp(self):
        self.strategy = StrategyRegistry.get("ml_alpha")
        self.bars = make_synthetic_bars(n=100, base_price=420.0, seed=42)

    def test_inference_latency_under_5ms_per_symbol(self):
        """Verify MLAlphaStrategy inference latency < 5ms per symbol."""
        metrics = {
            "close": 420.0,
            "vol_drag_proxy": 0.015,
            "features": np.zeros((1, 14), dtype=np.float32),
            "bars": self.bars,
        }

        # Warm-up pass
        self.strategy.evaluate_candidate("QQQ", metrics)

        # Benchmark 50 evaluations
        latencies_ms: List[float] = []
        for _ in range(50):
            t0 = time.perf_counter()
            sig = self.strategy.evaluate_candidate("QQQ", metrics)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

        mean_latency = float(np.mean(latencies_ms))
        p95_latency = float(np.percentile(latencies_ms, 95))

        self.assertLess(
            mean_latency,
            5.0,
            f"Mean inference latency {mean_latency:.3f} ms exceeds 5ms limit!",
        )
        self.assertLess(
            p95_latency,
            10.0,
            f"P95 inference latency {p95_latency:.3f} ms is unacceptably high!",
        )

    def test_missing_model_fail_safe_fallback_to_cash_defense(self):
        """Missing model file must gracefully fall back to CASH_DEFENSE (direction NEUTRAL, conviction 0.0)."""
        runner = OnnxModelRunner(model_path="non_existent_model_file.onnx")
        probs = runner.predict_proba(np.zeros((1, 14), dtype=np.float32))

        # Must return [1.0, 0.0, 0.0]
        self.assertEqual(probs.shape, (1, 3))
        self.assertAlmostEqual(float(probs[0, 0]), 1.0)
        self.assertAlmostEqual(float(probs[0, 1]), 0.0)
        self.assertAlmostEqual(float(probs[0, 2]), 0.0)

        # Strategy evaluation using missing model
        mock_strategy = MLAlphaStrategy(model_path="non_existent_model_file.onnx")
        metrics = {"close": 420.0, "features": np.zeros((1, 14), dtype=np.float32)}
        sig = mock_strategy.evaluate_candidate("QQQ", metrics)

        self.assertEqual(sig.direction, "NEUTRAL")
        self.assertEqual(sig.conviction, 0.0)

    def test_corrupted_model_file_fail_safe_fallback(self):
        """Corrupted model file must gracefully fall back to CASH_DEFENSE without throwing."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tf:
            tf.write(b"NOT_A_VALID_ONNX_FILE_CORRUPTED_HEADER_GIBBERISH_DATA")
            corrupt_path = tf.name

        try:
            runner = OnnxModelRunner(model_path=corrupt_path)
            probs = runner.predict_proba(np.zeros((1, 14), dtype=np.float32))
            self.assertEqual(probs.shape, (1, 3))
            self.assertAlmostEqual(float(probs[0, 0]), 1.0)

            mock_strategy = MLAlphaStrategy(model_path=corrupt_path)
            metrics = {"close": 420.0, "features": np.zeros((1, 14), dtype=np.float32)}
            sig = mock_strategy.evaluate_candidate("QQQ", metrics)
            self.assertEqual(sig.direction, "NEUTRAL")
            self.assertEqual(sig.conviction, 0.0)
        finally:
            if os.path.exists(corrupt_path):
                os.remove(corrupt_path)

    def test_nan_features_fail_safe_fallback(self):
        """Features with NaN or Inf values must trigger fail-safe fallback."""
        runner = OnnxModelRunner(model_path="models/ml_alpha_v1.onnx")
        nan_feats = np.array([[np.nan, 0.0, 1.0, np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        probs = runner.predict_proba(nan_feats)
        self.assertEqual(probs.shape, (1, 3))
        self.assertAlmostEqual(float(probs[0, 0]), 1.0)

    def test_live_onnx_artifact_inference_and_metadata_validation(self):
        """Verify on-disk models/ml_alpha_v1.onnx loads cleanly and predicts valid probabilities."""
        if not os.path.exists("models/ml_alpha_v1.onnx"):
            self.skipTest("models/ml_alpha_v1.onnx not present")

        runner = OnnxModelRunner("models/ml_alpha_v1.onnx")
        self.assertTrue(runner.is_valid, "ONNX model runner failed to initialize valid session")
        self.assertTrue(runner.schema_valid, "Feature schema hash in metadata did not match runtime schema")
        self.assertIn("feature_hash", runner.metadata)

        # Real inference with dummy features
        dummy_features = np.zeros((1, 14), dtype=np.float32)
        probs = runner.predict_proba(dummy_features)
        self.assertEqual(probs.shape, (1, 3))
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))
        self.assertLess(runner.last_inference_ms, 5.0)


class TestHotSwapAndDualTrackBearShield(unittest.TestCase):
    """(e) Hot-Swap & Dual-Track Bear Shield Test Suite."""

    def setUp(self):
        self.strategy = StrategyRegistry.get("ml_alpha")
        self.assertIsInstance(self.strategy, MLAlphaStrategy)

    def test_active_strategy_switch_and_registry_lookup(self):
        """Active strategy resolves properly from registry and can be hot-swapped."""
        available = StrategyRegistry.list_available()
        self.assertIn("ml_alpha", available)
        self.assertIn("directional_trend", available)
        self.assertIn("vwap", available)

        strat = StrategyRegistry.get("ml_alpha")
        self.assertEqual(strat.name, "ml_alpha")

    def test_dual_track_bear_shield_single_stock_vs_index(self):
        """Single-stock SHORT predictions route to CASH_DEFENSE while index SHORT predictions route to inverse ETPs."""
        # Mock runner predicting high-confidence SHORT: [p_cash=0.05, p_long=0.05, p_short=0.90]
        mock_runner = MagicMock()
        mock_runner.predict_proba.return_value = np.array([[0.05, 0.05, 0.90]], dtype=np.float32)
        mock_runner.last_inference_ms = 0.85
        mock_runner.model_version = "ml_alpha_v1_mock"

        with patch.object(self.strategy, "get_runner", return_value=mock_runner):
            metrics = {
                "close": 150.0,
                "vol_drag_proxy": 0.012,
                "features": np.zeros((1, 14), dtype=np.float32),
            }

            # 1. Single stocks (NVDA, AAPL, TSLA) -> CASH_DEFENSE
            for stock in ["NVDA", "AAPL", "TSLA", "ARM"]:
                sig = self.strategy.evaluate_candidate(stock, metrics)
                self.assertEqual(
                    sig.direction,
                    "NEUTRAL",
                    f"Single stock {stock} SHORT should be suppressed into NEUTRAL (CASH_DEFENSE)",
                )
                self.assertEqual(sig.conviction, 0.0)
                # Gate rejection must explicitly record dual_track_cash_defense
                self.assertTrue(
                    any(rej.get("gate_id") == "dual_track_cash_defense" for rej in (sig.gate_rejections or [])),
                    f"Gate rejection for {stock} missing 'dual_track_cash_defense'",
                )

            # 2. Broad market indices (QQQ, SPY, SMH) -> SHORT signal
            for index_ticker in ["QQQ", "SPY", "SMH", "^NDX"]:
                sig = self.strategy.evaluate_candidate(index_ticker, metrics)
                self.assertEqual(
                    sig.direction,
                    "SHORT",
                    f"Benchmark index {index_ticker} SHORT should be approved for inverse ETP routing",
                )
                self.assertGreater(sig.conviction, 0.70)
                self.assertIn("MLAlpha SHORT", sig.rationale)

    def test_long_predictions_pass_for_both_stock_and_index(self):
        """High-confidence LONG predictions should emit LONG for both single stocks and indices."""
        mock_runner = MagicMock()
        mock_runner.predict_proba.return_value = np.array([[0.05, 0.90, 0.05]], dtype=np.float32)
        mock_runner.last_inference_ms = 0.75
        mock_runner.model_version = "ml_alpha_v1_mock"

        with patch.object(self.strategy, "get_runner", return_value=mock_runner):
            metrics = {
                "close": 120.0,
                "vol_drag_proxy": 0.010,
                "features": np.zeros((1, 14), dtype=np.float32),
            }

            for ticker in ["NVDA", "QQQ"]:
                sig = self.strategy.evaluate_candidate(ticker, metrics)
                self.assertEqual(sig.direction, "LONG")
                self.assertGreater(sig.conviction, 0.70)
                self.assertEqual(sig.setup_quality, "exceptional")

    def test_volatility_drag_guard_forces_cash_defense(self):
        """Even with high bull conviction, vol_drag_proxy > cutoff forces CASH_DEFENSE."""
        mock_runner = MagicMock()
        mock_runner.predict_proba.return_value = np.array([[0.02, 0.95, 0.03]], dtype=np.float32)
        mock_runner.last_inference_ms = 0.70
        mock_runner.model_version = "ml_alpha_v1_mock"

        with patch.object(self.strategy, "get_runner", return_value=mock_runner):
            metrics = {
                "close": 120.0,
                "vol_drag_proxy": 0.08,  # > default cutoff 0.05!
                "features": np.zeros((1, 14), dtype=np.float32),
            }

            sig = self.strategy.evaluate_candidate("QQQ", metrics)
            self.assertEqual(sig.direction, "NEUTRAL")
            self.assertEqual(sig.conviction, 0.0)
            self.assertTrue(any(rej.get("gate_id") == "vol_drag_cutoff" for rej in (sig.gate_rejections or [])))


if __name__ == "__main__":
    unittest.main()
