"""Leak-free supervised fraud classification and end-to-end hybrid anomaly pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from imblearn.over_sampling import SMOTE
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from bank_anomaly.anomaly import (
    BankSegmentationAnomalyDetector,
    ClusteringAnomalyConfig,
    build_feature_preprocessor,
)
from bank_anomaly.data import (
    MODEL_CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    MODEL_NUMERICAL_FEATURES,
    clean_transactions,
    generate_synthetic_transactions,
)

logger = logging.getLogger(__name__)

SUPPORTED_MODELS: frozenset[str] = frozenset({"random_forest", "decision_tree", "xgboost"})


@dataclass(frozen=True)
class ClassificationConfig:
    """Configuration for leak-free supervised classification training and evaluation."""

    model_name: str = "random_forest"
    random_state: int = 42
    test_size: float = 0.2
    use_smote: bool = True
    drop_dominant_features: bool = False

    def __post_init__(self) -> None:
        if self.model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_name {self.model_name!r}; choose from {sorted(SUPPORTED_MODELS)}."
            )
        if not 0.05 <= self.test_size <= 0.5:
            raise ValueError(f"test_size must be in [0.05, 0.5], got {self.test_size}.")


def _build_estimator(model_name: str, random_state: int) -> BaseEstimator:
    """Instantiate a supported scikit-learn or XGBoost classifier."""
    if model_name == "decision_tree":
        return DecisionTreeClassifier(max_depth=10, random_state=random_state)
    if model_name == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
        )
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=4,
        random_state=random_state,
        n_jobs=1,
    )


class FraudClassificationPipeline:
    """Leak-free supervised classifier wrapped in a schema-safe sklearn Pipeline."""

    def __init__(self, config: ClassificationConfig | None = None) -> None:
        self.config = config or ClassificationConfig()
        self.feature_cols_: tuple[str, ...] = self._resolve_feature_cols()
        self.pipeline_: Pipeline | None = None
        self.metrics_: dict[str, float] = {}
        self._is_fitted: bool = False

    def _resolve_feature_cols(self) -> tuple[str, ...]:
        """Select modeling columns based on dominant-feature ablation flag."""
        if not self.config.drop_dominant_features:
            return MODEL_FEATURES
        excluded = {"TransactionAmount", "AccountBalance"}
        return tuple(col for col in MODEL_FEATURES if col not in excluded)

    def _build_preprocessor(self) -> Any:
        """Construct ColumnTransformer scoped to active numerical and categorical columns."""
        num_cols = tuple(c for c in MODEL_NUMERICAL_FEATURES if c in self.feature_cols_)
        cat_cols = tuple(c for c in MODEL_CATEGORICAL_FEATURES if c in self.feature_cols_)
        return build_feature_preprocessor(numerical_features=num_cols, categorical_features=cat_cols)

    def fit_and_evaluate(self, df: pd.DataFrame, target_col: str = "Target") -> dict[str, float]:
        """Split data first, fit preprocessing and optional SMOTE on train only, and evaluate.

        Args:
            df: Input DataFrame containing modeling features and binary `target_col`.
            target_col: Name of binary target column (`0` = normal, `1` = suspicious).

        Returns:
            Dictionary of held-out evaluation metrics (`accuracy`, `precision`, `recall`,
            `f1_score`, `roc_auc`, `pr_auc`).

        Raises:
            ValueError: If target column is missing or contains fewer than 2 classes.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col!r} not found in DataFrame.")

        cleaned = clean_transactions(df, drop_duplicates=False)
        y = pd.to_numeric(cleaned[target_col], errors="coerce").dropna().astype(int)
        if len(y) != len(cleaned):
            raise ValueError("Target column contains invalid or null labels.")
        if y.nunique() < 2:
            raise ValueError("Target column must contain at least 2 classes for classification.")

        x_raw = cleaned[list(self.feature_cols_)].copy()
        x_train, x_test, y_train, y_test = train_test_split(
            x_raw,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        preprocessor = self._build_preprocessor()
        x_train_enc = preprocessor.fit_transform(x_train)
        x_test_enc = preprocessor.transform(x_test)

        x_fit, y_fit = self._resample_training_split(x_train_enc, y_train.to_numpy())
        classifier = _build_estimator(self.config.model_name, self.config.random_state)
        classifier.fit(x_fit, y_fit)

        self.pipeline_ = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
        self._is_fitted = True

        y_pred = classifier.predict(x_test_enc)
        y_prob = classifier.predict_proba(x_test_enc)[:, 1]
        self.metrics_ = self._evaluate_predictions(y_test.to_numpy(), y_pred, y_prob)

        logger.info(
            "Trained %s classifier (ROC-AUC=%.4f, F1=%.4f, train=%d, test=%d).",
            self.config.model_name,
            self.metrics_["roc_auc"],
            self.metrics_["f1_score"],
            len(x_fit),
            len(x_test),
        )
        return dict(self.metrics_)

    def _resample_training_split(
        self, x_train_enc: np.ndarray, y_train: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE strictly on encoded training data when minority count allows."""
        if not self.config.use_smote:
            return x_train_enc, y_train

        _, counts = np.unique(y_train, return_counts=True)
        min_count = int(counts.min())
        max_count = int(counts.max())
        if min_count < 2 or min_count == max_count:
            return x_train_enc, y_train

        k_neighbors = min(5, min_count - 1)
        smote = SMOTE(random_state=self.config.random_state, k_neighbors=k_neighbors)
        return smote.fit_resample(x_train_enc, y_train)

    def _evaluate_predictions(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> dict[str, float]:
        """Compute standard binary classification and ranking metrics on held-out test set."""
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
            "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        }

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict binary labels and suspicious probabilities for raw transaction rows.

        Args:
            df: Inference DataFrame containing required transaction features.

        Returns:
            Tuple of `(predicted_labels, suspicious_probabilities)`.

        Raises:
            RuntimeError: If classifier pipeline has not been trained.
        """
        if not self._is_fitted or self.pipeline_ is None:
            raise RuntimeError("Classification pipeline must be fitted before predict().")

        cleaned = clean_transactions(df, drop_duplicates=False)
        x_raw = cleaned[list(self.feature_cols_)]
        labels = self.pipeline_.predict(x_raw).astype(int)
        probabilities = np.round(self.pipeline_.predict_proba(x_raw)[:, 1].astype(float), 4)
        return labels, probabilities

    def save(self, path: Path) -> Path:
        """Persist trained classification pipeline as `.joblib`."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted classification pipeline.")
        if path.suffix.lower() != ".joblib":
            raise ValueError(f"Artifacts must use '.joblib' extension, got {path.suffix!r}.")

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved FraudClassificationPipeline to %s.", path)
        return path

    @classmethod
    def load(cls, path: Path) -> FraudClassificationPipeline:
        """Load a verified `.joblib` classification pipeline artifact."""
        if path.suffix.lower() != ".joblib":
            raise ValueError(
                f"Refusing non-joblib artifact {path.name!r}; legacy .h5 pickles are unsupported at runtime."
            )
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found at {path}.")

        loaded: Any = joblib.load(path)
        if not isinstance(loaded, cls):
            raise ValueError(f"Expected {cls.__name__} in {path}, got {type(loaded).__name__}.")
        return loaded


class HybridAnomalyPipeline:
    """Two-stage pipeline combining unsupervised segmentation with supervised classification."""

    def __init__(
        self,
        anomaly_config: ClusteringAnomalyConfig | None = None,
        classification_config: ClassificationConfig | None = None,
    ) -> None:
        self.detector = BankSegmentationAnomalyDetector(config=anomaly_config)
        self.classifier = FraudClassificationPipeline(config=classification_config)
        self.training_summary_: dict[str, Any] = {}
        self._is_ready: bool = False

    def fit(self, df: pd.DataFrame) -> dict[str, Any]:
        """Fit unsupervised anomaly detector then train supervised classifier on derived flags.

        Args:
            df: Training transaction DataFrame.

        Returns:
            Structured summary dictionary containing clustering and classification metrics.
        """
        cleaned = clean_transactions(df, drop_duplicates=True)
        self.detector.fit(cleaned)
        scored = self.detector.score(cleaned)

        labeled = cleaned.copy()
        labeled["Target"] = scored["is_anomaly"].astype(int)

        # Guarantee at least two target examples if dataset has ultra-uniform distances
        if labeled["Target"].sum() < 4 and len(labeled) >= 20:
            top_k = max(4, int(len(labeled) * 0.1))
            top_idx = scored["anomaly_score"].nlargest(top_k).index
            labeled.loc[top_idx, "Target"] = 1

        cls_metrics = self.classifier.fit_and_evaluate(labeled, target_col="Target")
        anomaly_count = int(labeled["Target"].sum())
        self.training_summary_ = {
            "n_rows": len(cleaned),
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_count / len(cleaned), 4),
            "clustering_metrics": dict(self.detector.clustering_metrics_),
            "classification_metrics": cls_metrics,
        }
        self._is_ready = True
        return dict(self.training_summary_)

    def analyze(self, df: pd.DataFrame, include_classification: bool = True) -> pd.DataFrame:
        """Score incoming transactions using both segmentation anomaly detection and classifier.

        Args:
            df: Incoming transaction DataFrame.
            include_classification: Whether to include supervised suspicious probability estimates.

        Returns:
            Enriched analysis DataFrame aligned to input row order.

        Raises:
            RuntimeError: If pipeline has not been fitted.
        """
        if not self._is_ready:
            raise RuntimeError("HybridAnomalyPipeline must be fitted before analyze().")

        results = self.detector.score(df)
        if include_classification:
            labels, probs = self.classifier.predict(df)
            results["predicted_label"] = labels
            results["suspicious_probability"] = probs
        else:
            results["predicted_label"] = None
            results["suspicious_probability"] = None

        return results


def run_offline_demo(
    n_rows: int = 400,
    seed: int = 42,
    model_name: str = "random_forest",
) -> dict[str, Any]:
    """Execute the complete hybrid pipeline offline on deterministic synthetic transactions.

    Args:
        n_rows: Number of synthetic transactions to generate.
        seed: Random seed for reproducible execution.
        model_name: Classifier architecture (`random_forest`, `decision_tree`, or `xgboost`).

    Returns:
        Structured JSON-serializable summary of metrics and sample predictions.
    """
    raw_df = generate_synthetic_transactions(n_rows=n_rows, seed=seed, include_anomalies=True)
    pipeline = HybridAnomalyPipeline(
        anomaly_config=ClusteringAnomalyConfig(random_state=seed),
        classification_config=ClassificationConfig(model_name=model_name, random_state=seed),
    )
    summary = pipeline.fit(raw_df)
    sample_batch = raw_df.head(5)
    scored_sample = pipeline.analyze(sample_batch, include_classification=True)

    return {
        "status": "ok",
        "mode": "offline_synthetic",
        "model_name": model_name,
        "summary": summary,
        "sample_predictions": scored_sample.to_dict(orient="records"),
    }
