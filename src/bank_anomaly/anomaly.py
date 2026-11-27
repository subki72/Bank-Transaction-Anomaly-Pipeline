"""Unsupervised clustering segmentation and centroid/z-score anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bank_anomaly.data import (
    MODEL_CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    MODEL_NUMERICAL_FEATURES,
    clean_transactions,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClusteringAnomalyConfig:
    """Configuration parameters for unsupervised segmentation and anomaly scoring."""

    n_clusters: int = 2
    random_state: int = 42
    anomaly_quantile: float = 0.92
    zscore_threshold: float = 2.5

    def __post_init__(self) -> None:
        if self.n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {self.n_clusters}.")
        if not 0.5 <= self.anomaly_quantile < 1.0:
            raise ValueError(f"anomaly_quantile must be in [0.5, 1.0), got {self.anomaly_quantile}.")
        if self.zscore_threshold <= 0.0:
            raise ValueError(f"zscore_threshold must be > 0, got {self.zscore_threshold}.")


def build_feature_preprocessor(
    numerical_features: tuple[str, ...] = MODEL_NUMERICAL_FEATURES,
    categorical_features: tuple[str, ...] = MODEL_CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Build a schema-safe, leak-free ColumnTransformer for transaction features.

    Args:
        numerical_features: Ordered numeric feature column names.
        categorical_features: Ordered categorical feature column names.

    Returns:
        Unfitted `ColumnTransformer` scaling numeric fields and one-hot encoding categories
        with `handle_unknown="ignore"` for safe production inference.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(numerical_features)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_features),
            ),
        ],
        remainder="drop",
    )


def _assign_risk_level(score: float, is_anomaly: bool) -> str:
    """Map continuous anomaly score and flag into discrete operational risk tier."""
    if score >= 0.75 or (is_anomaly and score >= 0.60):
        return "high"
    if is_anomaly or score >= 0.45:
        return "medium"
    return "low"


class BankSegmentationAnomalyDetector:
    """K-Means segmentation combined with centroid-distance and Z-score anomaly detection."""

    def __init__(self, config: ClusteringAnomalyConfig | None = None) -> None:
        self.config = config or ClusteringAnomalyConfig()
        self.preprocessor: ColumnTransformer = build_feature_preprocessor()
        self.kmeans: KMeans = KMeans(
            n_clusters=self.config.n_clusters,
            random_state=self.config.random_state,
            n_init=10,
        )
        self.pca: PCA | None = None
        self.num_means_: np.ndarray | None = None
        self.num_stds_: np.ndarray | None = None
        self.distance_threshold_: float = 0.0
        self.clustering_metrics_: dict[str, float] = {}
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> BankSegmentationAnomalyDetector:
        """Fit preprocessing, K-Means segmentation, PCA projection, and anomaly thresholds.

        Args:
            df: Training transaction DataFrame containing required modeling features.

        Returns:
            Fitted `BankSegmentationAnomalyDetector` instance.

        Raises:
            ValueError: If cleaned rows are fewer than `n_clusters`.
        """
        cleaned = clean_transactions(df, drop_duplicates=True)
        if len(cleaned) < self.config.n_clusters:
            raise ValueError(f"Need at least {self.config.n_clusters} rows to fit K-Means, got {len(cleaned)}.")

        feature_frame = cleaned[list(MODEL_FEATURES)]
        encoded = self.preprocessor.fit_transform(feature_frame)

        labels = self.kmeans.fit_predict(encoded)
        n_components = min(2, encoded.shape[1], len(cleaned))
        self.pca = PCA(n_components=n_components, random_state=self.config.random_state)
        self.pca.fit(encoded)

        num_matrix = cleaned[list(MODEL_NUMERICAL_FEATURES)].to_numpy(dtype=float)
        self.num_means_ = np.mean(num_matrix, axis=0)
        raw_stds = np.std(num_matrix, axis=0)
        self.num_stds_ = np.where(raw_stds < 1e-9, 1.0, raw_stds)

        distances = self.kmeans.transform(encoded).min(axis=1)
        self.distance_threshold_ = float(np.quantile(distances, self.config.anomaly_quantile))
        if self.distance_threshold_ <= 1e-9:
            self.distance_threshold_ = 1.0

        self.clustering_metrics_ = self._compute_cluster_metrics(encoded, labels)
        self._is_fitted = True

        logger.info(
            "Fitted BankSegmentationAnomalyDetector on %d rows (threshold=%.4f, silhouette=%.4f).",
            len(cleaned),
            self.distance_threshold_,
            self.clustering_metrics_.get("silhouette_score", 0.0),
        )
        return self

    def _compute_cluster_metrics(self, encoded: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """Compute internal clustering metrics safely when at least 2 clusters exist."""
        unique_labels = len(set(labels))
        if unique_labels < 2 or len(encoded) <= unique_labels:
            return {"silhouette_score": 0.0, "davies_bouldin_score": 0.0, "calinski_harabasz_score": 0.0}

        return {
            "silhouette_score": round(float(silhouette_score(encoded, labels)), 4),
            "davies_bouldin_score": round(float(davies_bouldin_score(encoded, labels)), 4),
            "calinski_harabasz_score": round(float(calinski_harabasz_score(encoded, labels)), 2),
        }

    def _compute_anomaly_metrics(
        self, encoded: np.ndarray, num_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute cluster IDs, centroid distances, max Z-scores, bounded scores, and flags."""
        assert self.num_means_ is not None and self.num_stds_ is not None
        cluster_ids = self.kmeans.predict(encoded)
        centroid_distances = self.kmeans.transform(encoded).min(axis=1)
        z_scores = np.abs((num_matrix - self.num_means_) / self.num_stds_)
        max_zscores = np.max(z_scores, axis=1)

        # Bounded logistic blend of centroid distance ratio and maximum feature Z-score
        dist_ratio = centroid_distances / self.distance_threshold_
        z_ratio = max_zscores / self.config.zscore_threshold
        combined_intensity = 0.55 * dist_ratio + 0.45 * z_ratio
        anomaly_scores = 1.0 / (1.0 + np.exp(-2.2 * (combined_intensity - 1.0)))
        is_anomaly = (centroid_distances > self.distance_threshold_) | (max_zscores >= self.config.zscore_threshold)
        return cluster_ids, centroid_distances, max_zscores, anomaly_scores, is_anomaly

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign cluster segments and compute centroid/Z-score anomaly scores.

        Args:
            df: Inference DataFrame preserving exact row count and order.

        Returns:
            DataFrame containing `cluster_id`, `centroid_distance`, `max_zscore`,
            `anomaly_score` in `[0, 1]`, `is_anomaly`, and `risk_level`.
        """
        if not self._is_fitted or self.num_means_ is None or self.num_stds_ is None:
            raise RuntimeError("Detector must be fitted before calling score().")

        cleaned = clean_transactions(df, drop_duplicates=False)
        if len(cleaned) != len(df):
            raise ValueError("Inference batch contains missing or non-numeric feature values.")

        encoded = self.preprocessor.transform(cleaned[list(MODEL_FEATURES)])
        num_matrix = cleaned[list(MODEL_NUMERICAL_FEATURES)].to_numpy(dtype=float)
        cluster_ids, centroid_distances, max_zscores, anomaly_scores, is_anomaly = self._compute_anomaly_metrics(
            encoded, num_matrix
        )
        return pd.DataFrame(
            {
                "cluster_id": cluster_ids.astype(int),
                "centroid_distance": np.round(centroid_distances, 4),
                "max_zscore": np.round(max_zscores, 4),
                "anomaly_score": np.round(anomaly_scores, 4),
                "is_anomaly": is_anomaly.astype(bool),
                "risk_level": [
                    _assign_risk_level(float(sc), bool(flag))
                    for sc, flag in zip(anomaly_scores, is_anomaly, strict=True)
                ],
            },
            index=df.index,
        )

    def save(self, path: Path) -> Path:
        """Persist fitted detector safely using `.joblib` format.

        Args:
            path: Target artifact file path ending in `.joblib`.

        Returns:
            Resolved saved artifact path.

        Raises:
            RuntimeError: If model is unfitted.
            ValueError: If extension is not `.joblib`.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted detector.")
        if path.suffix.lower() != ".joblib":
            raise ValueError(f"Artifacts must use '.joblib' extension, got {path.suffix!r}.")

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved BankSegmentationAnomalyDetector to %s.", path)
        return path

    @classmethod
    def load(cls, path: Path) -> BankSegmentationAnomalyDetector:
        """Load a verified `.joblib` detector artifact.

        Args:
            path: Path to `.joblib` artifact.

        Returns:
            Loaded `BankSegmentationAnomalyDetector` instance.

        Raises:
            ValueError: If extension is not `.joblib` or object type mismatches.
            FileNotFoundError: If artifact file does not exist.
        """
        if path.suffix.lower() != ".joblib":
            raise ValueError(
                f"Refusing non-joblib artifact {path.name!r}; legacy .h5 pickles are unsupported at runtime."
            )
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found at {path}.")

        loaded: Any = joblib.load(path)
        if not isinstance(loaded, cls):
            raise ValueError(f"Expected {cls.__name__} in {path}, got {type(loaded).__name__}.")
        logger.info("Loaded BankSegmentationAnomalyDetector from %s.", path)
        return loaded
