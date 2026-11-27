# Operational Runbook

## 1. Service Overview

- **Service Name**: Bank Transaction Anomaly Pipeline (`bank_anomaly.api:app`)
- **Port**: `8000`
- **Health Endpoint**: `GET /health`
- **Inference Endpoint**: `POST /analyze`
- **Model Metadata Endpoint**: `GET /model-info`

## 2. Startup & Health Verification

### Local CLI
```bash
python main.py --health
python main.py --demo --rows 400 --seed 42
python main.py --serve --host 0.0.0.0 --port 8000
```

### Containerized Runtime
```bash
docker compose up -d --build
curl -f http://127.0.0.1:8000/health
```

Expected healthy payload:
```json
{
  "status": "healthy",
  "service": "Bank Transaction Anomaly Pipeline",
  "version": "1.0.0",
  "environment": "production",
  "model_ready": true,
  "n_training_rows": 400
}
```

## 3. Configuration Reference

All runtime tuning parameters are loaded via `.env` / environment variables (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | Deployment environment label |
| `LOG_LEVEL` | `INFO` | Python logging level (`DEBUG..CRITICAL`) |
| `RANDOM_SEED` | `42` | Seed for deterministic reference baseline |
| `N_CLUSTERS` | `2` | Number of K-Means customer segments |
| `ANOMALY_QUANTILE` | `0.92` | Quantile cutoff for centroid distance anomaly threshold |
| `ZSCORE_THRESHOLD` | `2.5` | Feature Z-score trigger for anomaly flagging |
| `MAX_BATCH_SIZE` | `5000` | Maximum transactions accepted per `/analyze` call |
| `RATE_LIMIT_PER_MINUTE` | `120` | Sliding-window request limit per client IP |
| `CACHE_TTL_SECONDS` | `300` | Bounded LRU cache TTL for repeated batch payloads |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:8000` | Allowed CORS origins |

## 4. Incident Troubleshooting

- **`422 Unprocessable Entity` on `/analyze`**: Verify required fields (`TransactionAmount >= 0`, `CustomerAge` in `18..120`, `TransactionDuration > 0`, `LoginAttempts >= 1`, `AccountBalance >= 0`, non-empty `TransactionType`, `Location`, `Channel`, `CustomerOccupation`).
- **`429 Too Many Requests`**: Client exceeded `RATE_LIMIT_PER_MINUTE`. Back off for 60 seconds or tune `RATE_LIMIT_PER_MINUTE` for trusted internal callers.
- **`500 Internal Server Error`**: The global exception handler suppresses raw tracebacks and returns a `request_id`. Search structured server logs by `request_id` to inspect the root cause.
- **Offline Network Degradation (Notebooks)**: `Clustering.ipynb` automatically falls back to deterministic synthetic transactions (`n_rows=2537`, `seed=42`) if Google Sheets CSV fetch times out after 10 seconds.

## 5. Artifact Security Policy

Legacy `.h5` files under `models/` are archival outputs from initial notebook experiments and are **never deserialized** by the FastAPI service or CLI runtime. Production artifacts must use verified `.joblib` sklearn pipelines.
