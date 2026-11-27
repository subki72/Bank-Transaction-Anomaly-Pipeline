"""Unit tests for leak-free supervised classification and end-to-end hybrid pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from bank_anomaly.anomaly import ClusteringAnomalyConfig
from bank_anomaly.data import generate_synthetic_transactions
from bank_anomaly.pipeline import (
    ClassificationConfig,
    FraudClassificationPipeline,
    HybridAnomalyPipeline,
    run_offline_demo,
)


@pytest.mark.parametrize("model_name", ["random_forest", "decision_tree", "xgboost"])
def test_classification_pipeline_trains_and_evaluates_offline(model_name: str) -> None:
    """All supported supervised classifiers train leak-free and emit bounded metrics."""
    df = generate_synthetic_transactions(n_rows=180, seed=42, include_anomalies=True)
    df["Target"] = ((df["TransactionAmount"] > 900.0) | (df["LoginAttempts"] >= 3.0)).astype(int)

    clf = FraudClassificationPipeline(ClassificationConfig(model_name=model_name, random_state=42, test_size=0.25))
    metrics = clf.fit_and_evaluate(df, target_col="Target")

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.5 <= metrics["roc_auc"] <= 1.0
    labels, probs = clf.predict(df.head(10))
    assert len(labels) == 10
    assert len(probs) == 10
    assert set(labels).issubset({0, 1})
    assert all(0.0 <= float(p) <= 1.0 for p in probs)


def test_classification_pipeline_handles_unseen_categories_and_persistence(tmp_path: Path) -> None:
    """Saved .joblib Pipeline predicts cleanly even when inference contains unseen categories."""
    df = generate_synthetic_transactions(n_rows=120, seed=17, include_anomalies=True)
    df["Target"] = (df["LoginAttempts"] >= 2.0).astype(int)

    clf = FraudClassificationPipeline(ClassificationConfig(drop_dominant_features=True))
    clf.fit_and_evaluate(df)

    artifact = clf.save(tmp_path / "classifier.joblib")
    loaded = FraudClassificationPipeline.load(artifact)

    unseen = df.head(3).copy()
    unseen["Location"] = "BrandNewCity"
    unseen["CustomerOccupation"] = "Astronaut"
    labels, probs = loaded.predict(unseen)
    assert len(labels) == 3
    assert len(probs) == 3

    with pytest.raises(ValueError, match="Refusing non-joblib"):
        FraudClassificationPipeline.load(tmp_path / "bad.h5")


def test_hybrid_pipeline_and_offline_demo_end_to_end() -> None:
    """HybridAnomalyPipeline coordinates segmentation, scoring, and classification offline."""
    demo_output = run_offline_demo(n_rows=160, seed=42, model_name="random_forest")

    assert demo_output["status"] == "ok"
    assert demo_output["mode"] == "offline_synthetic"
    summary = demo_output["summary"]
    assert summary["n_rows"] == 160
    assert summary["anomaly_count"] >= 4
    assert "classification_metrics" in summary
    assert len(demo_output["sample_predictions"]) == 5

    pipeline = HybridAnomalyPipeline(
        anomaly_config=ClusteringAnomalyConfig(n_clusters=2),
        classification_config=ClassificationConfig(model_name="random_forest"),
    )
    with pytest.raises(RuntimeError, match="must be fitted"):
        pipeline.analyze(generate_synthetic_transactions(n_rows=10, seed=1))
