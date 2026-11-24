"""Machine Learning Alpha Engine, Feature Extractor & Labeling Package."""

from trading_system.ml.features import (
    FEATURE_NAMES,
    compute_feature_schema_hash,
    extract_features,
    extract_features_matrix,
)
from trading_system.ml.labeling import (
    LABEL_CASH_DEFENSE,
    LABEL_LONG,
    LABEL_SHORT,
    TripleBarrierLabeler,
    label_bars,
    purged_walk_forward_splits,
)

__all__ = [
    "FEATURE_NAMES",
    "compute_feature_schema_hash",
    "extract_features",
    "extract_features_matrix",
    "LABEL_CASH_DEFENSE",
    "LABEL_LONG",
    "LABEL_SHORT",
    "TripleBarrierLabeler",
    "label_bars",
    "purged_walk_forward_splits",
]
