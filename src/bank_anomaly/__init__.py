"""Public exports for the Bank Transaction Anomaly Pipeline package."""

from __future__ import annotations

from bank_anomaly.anomaly import (
    BankSegmentationAnomalyDetector,
    ClusteringAnomalyConfig,
    build_feature_preprocessor,
)
from bank_anomaly.api import create_app
from bank_anomaly.config import PipelineSettings, configure_logging
from bank_anomaly.data import (
    MODEL_CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    MODEL_NUMERICAL_FEATURES,
    RAW_SCHEMA_COLUMNS,
    TransactionRecord,
    clean_transactions,
    compute_simultaneous_iqr_mask,
    fetch_remote_transactions,
    filter_iqr_outliers,
    generate_synthetic_transactions,
    load_transactions_with_fallback,
)
from bank_anomaly.pipeline import (
    ClassificationConfig,
    FraudClassificationPipeline,
    HybridAnomalyPipeline,
    run_offline_demo,
)

__version__ = "1.0.0"

__all__ = [
    "BankSegmentationAnomalyDetector",
    "ClassificationConfig",
    "ClusteringAnomalyConfig",
    "FraudClassificationPipeline",
    "HybridAnomalyPipeline",
    "MODEL_CATEGORICAL_FEATURES",
    "MODEL_FEATURES",
    "MODEL_NUMERICAL_FEATURES",
    "PipelineSettings",
    "RAW_SCHEMA_COLUMNS",
    "TransactionRecord",
    "__version__",
    "build_feature_preprocessor",
    "clean_transactions",
    "compute_simultaneous_iqr_mask",
    "configure_logging",
    "create_app",
    "fetch_remote_transactions",
    "filter_iqr_outliers",
    "generate_synthetic_transactions",
    "load_transactions_with_fallback",
    "run_offline_demo",
]
