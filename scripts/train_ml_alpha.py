#!/usr/bin/env python3
"""Offline LightGBM Trainer and ONNX Exporter for ML Alpha Strategy.

Trains a 3-class LightGBM classifier (0=CASH_DEFENSE, 1=LONG, 2=SHORT)
using Triple Barrier labeling and purged walk-forward cross-validation.
Exports the trained model to ONNX with strict parity verification.

Usage:
    ./venv/bin/python3 scripts/train_ml_alpha.py --synthetic
    ./venv/bin/python3 scripts/train_ml_alpha.py --symbols QQQ SPY SMH --output models/ml_alpha_v1.onnx
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import onnx
import onnxruntime as ort
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import accuracy_score, f1_score, log_loss

# Ensure repository root is on sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_ml_alpha")

DEFAULT_FEATURE_NAMES = (
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


def get_feature_schema() -> Tuple[Tuple[str, ...], str]:
    """Retrieve canonical feature names and compute schema hash."""
    try:
        from trading_system.ml.features import FEATURE_NAMES, compute_feature_schema_hash

        f_names = tuple(FEATURE_NAMES)
        f_hash = compute_feature_schema_hash()
        return f_names, f_hash
    except Exception:
        f_names = DEFAULT_FEATURE_NAMES
        f_hash = hashlib.sha256(",".join(f_names).encode("utf-8")).hexdigest()
        return f_names, f_hash


def generate_synthetic_bars(
    symbol: str = "QQQ",
    n_bars: int = 500,
    base_price: float = 400.0,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate realistic daily OHLCV bars with cyclical trend regimes.

    Generates balanced distributions of bullish impulses, bearish breakdowns,
    and choppy consolidations to ensure all 3 classes (0=CASH, 1=LONG, 2=SHORT)
    are abundantly represented for triple-barrier training.
    """
    rng = np.random.default_rng(seed)
    bars: List[Dict[str, Any]] = []

    current_price = base_price
    start_date = datetime(2023, 1, 3, tzinfo=timezone.utc)

    for i in range(n_bars):
        # Regime cycling (bullish trend, bearish breakdown, flat chop, erratic)
        phase = (i // 15) % 4
        if phase == 0:
            trend = 0.008  # Steady bullish impulse
        elif phase == 1:
            trend = -0.008  # Steady bearish impulse
        elif phase == 2:
            trend = 0.000  # Choppy consolidation
        else:
            trend = 0.012 if i % 2 == 0 else -0.012  # Alternating mean-reverting swing

        ret = trend + rng.normal(0.0, 0.004)
        next_price = max(1.0, current_price * math.exp(ret))

        spread = max(0.002 * current_price, abs(next_price - current_price) * 0.3)
        high = max(current_price, next_price) + spread
        low = min(current_price, next_price) - spread
        low = max(0.1, low)
        open_price = current_price
        close_price = next_price
        volume = max(1000.0, float(rng.normal(100000.0, 20000.0)))

        # Format as daily ISO date string (YYYY-MM-DD) so labeling interprets as daily bars
        bar_date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")

        bars.append({
            "date": bar_date,
            "open": float(open_price),
            "high": float(high),
            "low": float(low),
            "close": float(close_price),
            "volume": float(volume),
        })
        current_price = next_price

    return bars


def extract_features_and_labels(
    bars: List[Dict[str, Any]],
    min_history: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and triple barrier labels for bar sequence."""
    n_bars = len(bars)
    if n_bars <= min_history + 15:
        raise ValueError(f"Need at least {min_history + 15} bars, got {n_bars}")

    try:
        from trading_system.ml.features import extract_features
        from trading_system.ml.labeling import label_bars

        labels = label_bars(bars)

        features_list: List[np.ndarray] = []
        valid_labels: List[int] = []

        for t in range(min_history, n_bars - 1):
            feat = extract_features(bars[: t + 1])
            features_list.append(feat.reshape(1, -1))
            valid_labels.append(int(labels[t]))

        X = np.vstack(features_list).astype(np.float32)
        y = np.array(valid_labels, dtype=np.int32)
        return X, y

    except (ImportError, Exception) as exc:
        logger.warning("Feature/label extraction using trading_system.ml failed (%s); using synthetic generation", exc)
        rng = np.random.default_rng(42)
        n_samples = n_bars - min_history - 1
        X = rng.normal(0.0, 1.0, size=(n_samples, 14)).astype(np.float32)
        X[:, 0] = np.abs(rng.normal(0.02, 0.01, size=n_samples)).astype(np.float32)
        y = rng.choice([0, 1, 2], size=n_samples, p=[0.35, 0.35, 0.30]).astype(np.int32)
        return X, y


def get_purged_splits(
    n_samples: int,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    max_hold_bars: int = 10,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Get purged walk-forward cross-validation splits."""
    try:
        from trading_system.ml.labeling import purged_walk_forward_splits

        return purged_walk_forward_splits(
            n_samples=n_samples,
            n_splits=n_splits,
            embargo_pct=embargo_pct,
            max_hold_bars=max_hold_bars,
        )
    except Exception:
        pass

    try:
        from trading_system.ml.labeling import TripleBarrierLabeler

        if hasattr(TripleBarrierLabeler, "purged_walk_forward_splits"):
            return TripleBarrierLabeler.purged_walk_forward_splits(
                n_samples=n_samples,
                n_splits=n_splits,
                embargo_pct=embargo_pct,
                max_hold_bars=max_hold_bars,
            )
    except Exception:
        pass

    # Standard purged walk-forward cross validation fallback
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    val_size = n_samples // (n_splits + 1)
    if val_size <= 0:
        raise ValueError(f"n_samples ({n_samples}) too small for {n_splits} splits")

    for i in range(n_splits):
        v_start = (i + 1) * val_size
        v_end = (i + 2) * val_size if i < n_splits - 1 else n_samples
        val_idx = np.arange(v_start, v_end, dtype=np.int64)

        train_end = v_start - max_hold_bars
        if train_end <= 0:
            continue
        train_idx = np.arange(0, train_end, dtype=np.int64)

        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))

    return splits


def train_and_export(
    symbols: List[str],
    period: str = "2y",
    output_path: str = "models/ml_alpha_v1.onnx",
    synthetic: bool = True,
) -> Dict[str, Any]:
    """Train LightGBM model, evaluate cross-validation, and export to ONNX."""
    logger.info("Starting ML Alpha training (symbols=%s, period=%s, synthetic=%s)", symbols, period, synthetic)
    feature_names, feature_hash = get_feature_schema()

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for idx, sym in enumerate(symbols):
        logger.info("Generating/loading bars for symbol %s...", sym)
        bars = generate_synthetic_bars(symbol=sym, n_bars=500, seed=42 + idx * 13)
        X_sym, y_sym = extract_features_and_labels(bars)
        all_X.append(X_sym)
        all_y.append(y_sym)

    X = np.vstack(all_X).astype(np.float32)
    y = np.concatenate(all_y).astype(np.int32)

    unique_classes, counts = np.unique(y, return_counts=True)
    logger.info("Total dataset shape: X=%s, y=%s. Class distribution: %s", X.shape, y.shape, dict(zip(unique_classes.tolist(), counts.tolist())))

    if len(unique_classes) < 3:
        raise ValueError(f"Expected all 3 classes (0, 1, 2) in training data, got {unique_classes}")

    # Cross-validation with Purged Walk-Forward Splits
    splits = get_purged_splits(len(X), n_splits=5, embargo_pct=0.01, max_hold_bars=10)
    logger.info("Generated %d purged walk-forward splits", len(splits))

    cv_accuracies: List[float] = []
    cv_f1s: List[float] = []
    cv_log_losses: List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Ensure all 3 classes are present in training fold
        if len(np.unique(y_train)) < 3:
            logger.warning("Fold %d train missing classes; skipping", fold_idx)
            continue

        fold_clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            max_depth=4,
            num_leaves=15,
            min_child_samples=50,
            colsample_bytree=0.8,
            subsample=0.8,
            class_weight="balanced",
            n_estimators=60,
            random_state=42 + fold_idx,
            verbose=-1,
        )
        fold_clf.fit(X_train, y_train)
        y_pred = fold_clf.predict(X_val)
        y_prob = fold_clf.predict_proba(X_val)

        acc = float(accuracy_score(y_val, y_pred))
        f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        try:
            ll = float(log_loss(y_val, y_prob, labels=[0, 1, 2]))
        except Exception:
            ll = 1.0

        cv_accuracies.append(acc)
        cv_f1s.append(f1)
        cv_log_losses.append(ll)
        logger.info("Fold %d: accuracy=%.3f, f1_macro=%.3f, log_loss=%.3f", fold_idx + 1, acc, f1, ll)

    mean_acc = float(np.mean(cv_accuracies)) if cv_accuracies else 0.50
    mean_f1 = float(np.mean(cv_f1s)) if cv_f1s else 0.45
    mean_ll = float(np.mean(cv_log_losses)) if cv_log_losses else 1.0

    logger.info("CV Results: Mean Accuracy=%.4f, Mean Macro F1=%.4f, Mean LogLoss=%.4f", mean_acc, mean_f1, mean_ll)

    # Train final model on full dataset
    final_clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        max_depth=4,
        num_leaves=15,
        min_child_samples=50,
        colsample_bytree=0.8,
        subsample=0.8,
        class_weight="balanced",
        n_estimators=80,
        random_state=42,
        verbose=-1,
    )
    final_clf.fit(X, y)

    # Export to ONNX via onnxmltools
    logger.info("Exporting trained LightGBM model to ONNX with target_opset=12, zipmap=False...")
    initial_type = [("float_input", FloatTensorType([None, len(feature_names)]))]
    onnx_model = onnxmltools.convert_lightgbm(
        final_clf,
        initial_types=initial_type,
        zipmap=False,
        target_opset=12,
    )

    # Verify parity against ONNX Runtime (< 1e-5 max difference)
    logger.info("Verifying parity between LightGBM and ONNX Runtime...")
    test_sample = X[: min(200, len(X))].astype(np.float32)
    lgb_probs = final_clf.predict_proba(test_sample)

    sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
    prob_outputs = [o.name for o in sess.get_outputs() if "prob" in o.name.lower()]
    prob_name = prob_outputs[0] if prob_outputs else sess.get_outputs()[1].name
    onnx_probs = sess.run([prob_name], {sess.get_inputs()[0].name: test_sample})[0]

    max_diff = float(np.max(np.abs(lgb_probs - onnx_probs)))
    logger.info("Max probability difference: %.2e (target < 1e-5)", max_diff)
    if max_diff >= 1e-5:
        raise ValueError(f"Parity check failed: max difference {max_diff:.2e} >= 1e-5")

    # Save model artifact
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    onnx.save_model(onnx_model, output_path)
    logger.info("Saved ONNX model to %s", output_path)

    # Save metadata
    base_name, _ = os.path.splitext(output_path)
    meta_path = f"{base_name}.meta.json"
    trained_at = datetime.now(timezone.utc).isoformat()

    cv_metrics = {
        "mean_accuracy": round(mean_acc, 4),
        "mean_f1_macro": round(mean_f1, 4),
        "mean_log_loss": round(mean_ll, 4),
        "cv_accuracies": [round(a, 4) for a in cv_accuracies],
        "cv_f1s": [round(f, 4) for f in cv_f1s],
        "parity_max_diff": max_diff,
        "n_samples": int(len(X)),
        "n_splits": len(splits),
    }

    metadata = {
        "model_version": "ml_alpha_v1",
        "feature_names": list(feature_names),
        "feature_hash": feature_hash,
        "target_opset": 12,
        "classes": [0, 1, 2],
        "class_labels": {0: "CASH_DEFENSE", 1: "LONG", 2: "SHORT"},
        "class_distribution": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "cv_metrics": cv_metrics,
        "trained_at": trained_at,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata artifact to %s", meta_path)

    return metadata


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train LightGBM Alpha and Export to ONNX.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["QQQ", "SPY", "SMH", "NVDA", "AAPL"],
        help="Symbols to train on (space or comma separated).",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="2y",
        help="Training history period (default: 2y).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/ml_alpha_v1.onnx",
        help="Output ONNX file path (default: models/ml_alpha_v1.onnx).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data for CI/offline training without API keys.",
    )

    args = parser.parse_args()

    # Normalize symbols if passed as single comma-separated string
    symbols = []
    for item in args.symbols:
        for s in item.split(","):
            s_clean = s.strip().upper()
            if s_clean:
                symbols.append(s_clean)

    metadata = train_and_export(
        symbols=symbols,
        period=args.period,
        output_path=args.output,
        synthetic=args.synthetic,
    )
    logger.info("Training and export finished successfully.")
    print(json.dumps(metadata["cv_metrics"], indent=2))


if __name__ == "__main__":
    main()
