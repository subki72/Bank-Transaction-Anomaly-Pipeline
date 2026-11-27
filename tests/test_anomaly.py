"""Unit tests for unsupervised clustering segmentation and anomaly scoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bank_anomaly.anomaly import BankSegmentationAnomalyDetector, ClusteringAnomalyConfig
from bank_anomaly.data import generate_synthetic_transactions


def test_detector_fit_and_score_identifies_extreme_anomalies() -> None:
    """Detector assigns higher anomaly scores and anomaly flags to extreme behavior."""
    train_df = generate_synthetic_transactions(n_rows=200, seed=42, include_anomalies=True)
    detector = BankSegmentationAnomalyDetector(
        ClusteringAnomalyConfig(n_clusters=2, random_state=42, anomaly_quantile=0.92, zscore_threshold=2.5)
    )
    detector.fit(train_df)

    probe = pd.DataFrame(
        [
            {
                "TransactionAmount": 120.0,
                "TransactionType": "Debit",
                "Location": "Houston",
                "Channel": "Online",
                "CustomerAge": 32.0,
                "CustomerOccupation": "Engineer",
                "TransactionDuration": 100.0,
                "LoginAttempts": 1.0,
                "AccountBalance": 4000.0,
            },
            {
                "TransactionAmount": 9500.0,
                "TransactionType": "Debit",
                "Location": "UnseenCity",
                "Channel": "UnseenChannel",
                "CustomerAge": 22.0,
                "CustomerOccupation": "Student",
                "TransactionDuration": 12.0,
                "LoginAttempts": 8.0,
                "AccountBalance": 150.0,
            },
        ]
    )

    scored = detector.score(probe)
    assert len(scored) == 2
    assert 0.0 <= float(scored.loc[0, "anomaly_score"]) <= 1.0
    assert float(scored.loc[1, "anomaly_score"]) > float(scored.loc[0, "anomaly_score"])
    assert bool(scored.loc[1, "is_anomaly"]) is True
    assert scored.loc[1, "risk_level"] in {"medium", "high"}
    assert "silhouette_score" in detector.clustering_metrics_


def test_detector_rejects_unfitted_score_and_legacy_h5_loading(tmp_path: Path) -> None:
    """Detector enforces fit-before-score and blocks untrusted .h5 pickle paths."""
    detector = BankSegmentationAnomalyDetector()
    df = generate_synthetic_transactions(n_rows=30, seed=5)

    with pytest.raises(RuntimeError, match="must be fitted"):
        detector.score(df)

    detector.fit(df)
    saved_path = detector.save(tmp_path / "detector.joblib")
    loaded = BankSegmentationAnomalyDetector.load(saved_path)
    pd.testing.assert_frame_equal(detector.score(df.head(5)), loaded.score(df.head(5)))

    legacy_h5 = tmp_path / "legacy_model.h5"
    legacy_h5.write_bytes(b"untrusted")
    with pytest.raises(ValueError, match="Refusing non-joblib"):
        BankSegmentationAnomalyDetector.load(legacy_h5)


def test_detector_config_validates_bounds() -> None:
    """ClusteringAnomalyConfig rejects invalid cluster count and quantile values."""
    with pytest.raises(ValueError, match="n_clusters"):
        ClusteringAnomalyConfig(n_clusters=1)
    with pytest.raises(ValueError, match="anomaly_quantile"):
        ClusteringAnomalyConfig(anomaly_quantile=1.0)
    with pytest.raises(ValueError, match="zscore_threshold"):
        ClusteringAnomalyConfig(zscore_threshold=0.0)
