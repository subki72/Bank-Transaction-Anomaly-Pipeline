# ADR 001: Dual-Mode Modular ML Pipeline, FastAPI Service, and Notebook Remediation

- **Status**: Accepted
- **Date**: 2026-11-27
- **Decision Drivers**: Production reliability, train/test leakage elimination, deterministic offline verification, and artifact deserialization safety.

## Context

The initial repository contained two exploratory Jupyter notebooks (`Clustering.ipynb` and `Classification.ipynb`) with several production-readiness gaps:
1. Global `warnings.filterwarnings('ignore')` hid library deprecations and numerical warnings.
2. Live Google Sheets CSV ingestion had no explicit HTTP timeout or offline fallback.
3. Iterative IQR filtering mutated `df` inside a loop across `numerical_cols`, causing order-dependent quartile drift.
4. `Classification.ipynb` used global `pd.get_dummies(df, drop_first=True)` and injected 5% random label noise across all samples prior to `train_test_split`, risking data leakage and fragile schema alignment at inference time.
5. Scikit-learn estimators were saved as `.h5` pickle files, creating format confusion with Keras HDF5 files and deserialization risk if loaded blindly at runtime.

## Decision

1. **Dual-Mode Non-Breaking Architecture**:
   - Patch the real defects directly in `Clustering.ipynb` and `Classification.ipynb` while preserving exploratory markdown and visual storytelling.
   - Extract reusable, type-safe components into `src/bank_anomaly/` (`config.py`, `data.py`, `anomaly.py`, `pipeline.py`, `api.py`) fronted by `main.py` CLI (`--health`, `--demo`, `--serve`) and FastAPI (`/health`, `/model-info`, `/analyze`).
2. **Leak-Free Pipelines & Simultaneous IQR**:
   - Compute IQR bounds simultaneously on the un-mutated DataFrame (`compute_simultaneous_iqr_mask`).
   - Encapsulate scaling (`StandardScaler`) and encoding (`OneHotEncoder(handle_unknown="ignore")`) inside `ColumnTransformer` + sklearn `Pipeline`, fitted strictly on training splits.
3. **Offline Determinism & Artifact Safety**:
   - Provide `generate_synthetic_transactions` matching all 16 raw schema columns for offline testing, CI execution, and notebook fallback.
   - Refuse `.h5` pickle loading at runtime; enforce `.joblib` validation (`BankSegmentationAnomalyDetector.load`, `FraudClassificationPipeline.load`) and deterministic in-memory baseline initialization during API startup.

## Consequences

- **Positive**: 100% offline reproducible tests (`pytest` coverage >= 75%), zero train/test leakage, schema-safe handling of unseen categories in API requests, and hardened non-root Docker runtime.
- **Trade-off**: Initial FastAPI startup fits a small deterministic 400-row baseline (~0.5s) unless a pre-trained verified `.joblib` pipeline is configured, avoiding untrusted legacy `.h5` deserialization.
