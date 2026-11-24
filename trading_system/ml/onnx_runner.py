"""ONNX Model Runner for ML Alpha Strategy.

Provides robust, low-latency ONNX Runtime inference with latency guards,
feature schema validation, and fail-safe cash-defense fallbacks.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Canonical fail-safe probability distribution: 100% Cash Defense [p_cash, p_long, p_short]
FAILSAFE_PROBA = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)


class OnnxModelRunner:
    """Thread-safe ONNX model execution runner with runtime safeguards."""

    def __init__(
        self,
        model_path: str,
        meta_path: Optional[str] = None,
        max_inference_ms: float = 25.0,
    ) -> None:
        self.model_path = model_path
        self.max_inference_ms = float(max_inference_ms)
        self.meta_path = meta_path
        self.session = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        self.model_version: str = "unknown"
        self.last_inference_ms: float = 0.0
        self.is_valid: bool = False
        self.schema_valid: bool = True

        self._load_model_and_meta()

    def _load_model_and_meta(self) -> None:
        """Safely load ONNX model and validate metadata schema."""
        # 1. Resolve metadata file path
        if not self.meta_path:
            base, _ = os.path.splitext(self.model_path)
            candidate_meta = f"{base}.meta.json"
            if os.path.exists(candidate_meta):
                self.meta_path = candidate_meta

        # 2. Load metadata if available
        if self.meta_path and os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                self.model_version = self.metadata.get("model_version", "ml_alpha_v1")
                expected_hash = self.metadata.get("feature_hash")
                if expected_hash:
                    try:
                        from trading_system.ml.features import compute_feature_schema_hash

                        current_hash = compute_feature_schema_hash()
                        if current_hash != expected_hash:
                            logger.error(
                                "Feature schema hash mismatch: metadata has %s, current schema has %s",
                                expected_hash,
                                current_hash,
                            )
                            self.schema_valid = False
                    except ImportError:
                        logger.warning("trading_system.ml.features not yet available to validate schema hash.")
                    except Exception as err:
                        logger.warning("Could not compute feature schema hash: %s", err)
            except Exception as e:
                logger.warning("Failed to read metadata at %s: %s", self.meta_path, e)

        # 3. Check model file existence
        if not os.path.exists(self.model_path):
            logger.warning(
                "ONNX model file not found at %s. Fail-safe CASH_DEFENSE fallback active.",
                self.model_path,
            )
            self.session = None
            self.is_valid = False
            return

        # 4. Load ONNX session
        try:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name

            # Multiclass LightGBM with zipmap=False typically has outputs: label, probabilities
            outputs = self.session.get_outputs()
            if len(outputs) > 1:
                prob_out = [o for o in outputs if "prob" in o.name.lower()]
                self.output_name = prob_out[0].name if prob_out else outputs[1].name
            else:
                self.output_name = outputs[0].name

            self.is_valid = True
            logger.info("Successfully loaded ONNX model from %s", self.model_path)
        except Exception as e:
            logger.error("Failed to load ONNX model from %s: %s", self.model_path, e)
            self.session = None
            self.is_valid = False

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Execute model inference with latency guard and fail-safe fallback.

        Args:
            features: 1D or 2D feature array. Expected shape (1, N) or (N,).

        Returns:
            np.ndarray: float32 array [p_cash, p_long, p_short] of shape (1, 3).
        """
        # Fail-safe check
        if self.session is None or not self.is_valid or not self.schema_valid:
            return FAILSAFE_PROBA.copy()

        if features is None:
            logger.warning("Features array is None. Returning CASH_DEFENSE.")
            return FAILSAFE_PROBA.copy()

        try:
            feat_arr = np.asarray(features, dtype=np.float32)
            if feat_arr.ndim == 1:
                feat_arr = feat_arr.reshape(1, -1)
            elif feat_arr.ndim != 2:
                logger.warning("Invalid features dimensions %s. Returning CASH_DEFENSE.", feat_arr.shape)
                return FAILSAFE_PROBA.copy()

            # Check for NaN / Inf
            if np.isnan(feat_arr).any() or np.isinf(feat_arr).any():
                logger.warning("Features contain NaN or Inf values. Returning CASH_DEFENSE.")
                return FAILSAFE_PROBA.copy()

            # Execute with latency monitoring
            t0 = time.perf_counter()
            outputs = self.session.run([self.output_name], {self.input_name: feat_arr})
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.last_inference_ms = elapsed_ms

            # Latency guard
            if elapsed_ms > self.max_inference_ms:
                logger.warning(
                    "ONNX inference latency %.2f ms exceeded threshold %.2f ms. Returning CASH_DEFENSE.",
                    elapsed_ms,
                    self.max_inference_ms,
                )
                return FAILSAFE_PROBA.copy()

            raw_probs = outputs[0]
            if raw_probs.ndim == 1:
                raw_probs = raw_probs.reshape(1, -1)

            if raw_probs.shape != (1, 3):
                logger.warning("Unexpected output shape %s from ONNX model. Returning CASH_DEFENSE.", raw_probs.shape)
                return FAILSAFE_PROBA.copy()

            return raw_probs.astype(np.float32)

        except Exception as e:
            logger.error("Exception during ONNX inference: %s. Returning CASH_DEFENSE.", e)
            return FAILSAFE_PROBA.copy()
