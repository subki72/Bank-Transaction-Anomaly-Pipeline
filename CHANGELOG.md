# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-27

### Added
- Modular `src/bank_anomaly/` Python package (`config.py`, `data.py`, `anomaly.py`, `pipeline.py`, `api.py`).
- Unified `main.py` CLI supporting `--health`, `--demo`, and `--serve` modes.
- Deterministic 16-column synthetic bank transaction generator (`generate_synthetic_transactions`) and Pydantic `TransactionRecord` validation schema.
- Leak-free `ColumnTransformer` preprocessing (`StandardScaler` + `OneHotEncoder(handle_unknown="ignore")`), K-Means segmentation, centroid/Z-score anomaly scorer, and supervised Random Forest / Decision Tree / XGBoost pipeline.
- FastAPI inference service (`/health`, `/model-info`, `/analyze`) with sliding-window rate limiting, bounded TTL caching, restrictive CORS, and masked global exception handling.
- Offline unit and FastAPI integration test suite under `tests/` (`>= 75%` coverage gate).
- Production packaging (`Dockerfile` multi-stage non-root `python:3.11-slim`, `docker-compose.yml`, `Makefile`, `pyproject.toml`, pinned `requirements.txt`).
- GitHub Actions CI/CD workflows (`ci.yml`, `release.yml`), `.pre-commit-config.yaml`, Dependabot (`dependabot.yml`), `LICENSE`, `SECURITY.md`, `CONTRIBUTING.md`, `docs/runbook.md`, and `docs/adr/001-dual-mode-pipeline.md`.

### Fixed
- **Clustering.ipynb**: Replaced blind global `warnings.filterwarnings('ignore')` with structured logging and scoped warning filters.
- **Clustering.ipynb**: Added explicit `timeout=10` HTTP fetching with offline synthetic fallback for Google Sheets data ingestion.
- **Clustering.ipynb**: Replaced order-dependent iterative IQR filtering with simultaneous IQR mask calculation against the un-mutated DataFrame.
- **Classification.ipynb**: Replaced pre-split global `pd.get_dummies` and global-seed label noise injection with split-first `ColumnTransformer` + train-only resampling/noise and `.joblib` `Pipeline` persistence; fixed Seaborn v0.14 `palette` without `hue` deprecation warning.
