"""FastAPI service exposing segmentation, anomaly scoring, and fraud classification endpoints."""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import hashlib
import json
import logging
import time
import uuid

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel, Field

from bank_anomaly.anomaly import ClusteringAnomalyConfig
from bank_anomaly.config import PipelineSettings, configure_logging
from bank_anomaly.data import (
    TransactionRecord,
    generate_synthetic_transactions,
    records_to_dataframe,
)
from bank_anomaly.pipeline import ClassificationConfig, HybridAnomalyPipeline

logger = logging.getLogger(__name__)

MAX_CACHE_ENTRIES = 128
DEFAULT_REFERENCE_ROWS = 400


class AnalyzeRequest(BaseModel):
    """Request payload for batch transaction anomaly scoring and classification."""

    transactions: list[TransactionRecord] = Field(..., min_length=1, max_length=5000)
    include_classification: bool = Field(
        default=True,
        description="Include supervised Random Forest probability scores alongside anomaly scores.",
    )


class TransactionAnalysisResult(BaseModel):
    """Per-transaction risk assessment output."""

    transaction_id: str
    cluster_id: int
    centroid_distance: float
    max_zscore: float
    anomaly_score: float
    is_anomaly: bool
    risk_level: str
    predicted_label: int | None = None
    suspicious_probability: float | None = None


class BatchSummary(BaseModel):
    """Aggregate summary metrics for an analyzed transaction batch."""

    total_transactions: int
    anomaly_count: int
    anomaly_rate: float
    cluster_counts: dict[str, int]
    processing_ms: float


class AnalyzeResponse(BaseModel):
    """Response payload returned by `POST /analyze`."""

    summary: BatchSummary
    results: list[TransactionAnalysisResult]


class HealthResponse(BaseModel):
    """Liveness and readiness payload returned by `GET /health`."""

    status: str
    service: str
    version: str
    environment: str
    model_ready: bool
    n_training_rows: int


class SlidingWindowRateLimiter:
    """In-memory per-client sliding-window rate limiter."""

    def __init__(self, max_requests: int = 120, window_seconds: float = 60.0) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be >= 1.")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, deque[float]] = {}

    def check_allowed(self, client_key: str) -> bool:
        """Record a request timestamp and return True if within rate budget."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        bucket = self._buckets.setdefault(client_key, deque())
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= self.max_requests:
            return False
        bucket.append(now)
        return True


class BoundedPredictionCache:
    """TTL-aware LRU cache for deterministic batch analysis responses."""

    def __init__(self, max_entries: int = MAX_CACHE_ENTRIES, ttl_seconds: int = 300) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, tuple[float, AnalyzeResponse]] = OrderedDict()

    def get(self, key: str) -> AnalyzeResponse | None:
        """Fetch cached response if present and unexpired."""
        item = self._store.get(key)
        if item is None:
            return None
        created_at, response = item
        if time.monotonic() - created_at > self.ttl_seconds:
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return response

    def set(self, key: str, value: AnalyzeResponse) -> None:
        """Insert or refresh cached response bounded by `max_entries`."""
        self._store[key] = (time.monotonic(), value)
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)


def _build_ready_pipeline(settings: PipelineSettings) -> HybridAnomalyPipeline:
    """Initialize and fit an offline hybrid pipeline safely without loading legacy .h5 pickles."""
    reference_df = generate_synthetic_transactions(
        n_rows=DEFAULT_REFERENCE_ROWS,
        seed=settings.random_seed,
        include_anomalies=True,
    )
    pipeline = HybridAnomalyPipeline(
        anomaly_config=ClusteringAnomalyConfig(
            n_clusters=settings.n_clusters,
            random_state=settings.random_seed,
            anomaly_quantile=settings.anomaly_quantile,
            zscore_threshold=settings.zscore_threshold,
        ),
        classification_config=ClassificationConfig(
            model_name="random_forest",
            random_state=settings.random_seed,
        ),
    )
    pipeline.fit(reference_df)
    return pipeline


def _get_or_init_pipeline(request: Request) -> HybridAnomalyPipeline:
    """Retrieve fitted pipeline from app state, lazily initializing if lifespan did not run."""
    pipeline: HybridAnomalyPipeline | None = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        settings: PipelineSettings = request.app.state.settings
        pipeline = _build_ready_pipeline(settings)
        request.app.state.pipeline = pipeline
    return pipeline


def _enforce_rate_limit(request: Request) -> None:
    """FastAPI dependency enforcing sliding-window per-client rate limiting."""
    limiter: SlidingWindowRateLimiter = request.app.state.rate_limiter
    client_ip = request.client.host if request.client else "unknown"
    if not limiter.check_allowed(client_ip):
        logger.warning("Rate limit exceeded for client=%s.", client_ip)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Retry later.",
        )


def _hash_request(payload: AnalyzeRequest) -> str:
    """Build deterministic SHA-256 cache key for an `AnalyzeRequest` payload."""
    serialized = json.dumps(payload.model_dump(by_alias=True), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _format_results(
    records: list[TransactionRecord],
    scored_df: pd.DataFrame,
    elapsed_ms: float,
) -> AnalyzeResponse:
    """Assemble typed `AnalyzeResponse` from scored pipeline DataFrame."""
    results: list[TransactionAnalysisResult] = []
    cluster_counts: dict[str, int] = {}
    anomaly_count = 0

    for record, (_, row) in zip(records, scored_df.iterrows(), strict=True):
        cluster_key = str(int(row["cluster_id"]))
        cluster_counts[cluster_key] = cluster_counts.get(cluster_key, 0) + 1
        is_anom = bool(row["is_anomaly"])
        anomaly_count += int(is_anom)

        pred_label = row["predicted_label"]
        susp_prob = row["suspicious_probability"]
        results.append(
            TransactionAnalysisResult(
                transaction_id=record.TransactionID,
                cluster_id=int(row["cluster_id"]),
                centroid_distance=float(row["centroid_distance"]),
                max_zscore=float(row["max_zscore"]),
                anomaly_score=float(row["anomaly_score"]),
                is_anomaly=is_anom,
                risk_level=str(row["risk_level"]),
                predicted_label=int(pred_label) if pd.notna(pred_label) else None,
                suspicious_probability=float(susp_prob) if pd.notna(susp_prob) else None,
            )
        )

    total = len(results)
    summary = BatchSummary(
        total_transactions=total,
        anomaly_count=anomaly_count,
        anomaly_rate=round(anomaly_count / total, 4) if total else 0.0,
        cluster_counts=cluster_counts,
        processing_ms=round(elapsed_ms, 2),
    )
    return AnalyzeResponse(summary=summary, results=results)


async def _handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """Global exception boundary preventing internal tracebacks from leaking to HTTP callers."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    logger.exception("Unhandled error on %s [request_id=%s]: %s", request.url.path, request_id, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "request_id": request_id},
    )


def health_check(
    request: Request,
    pipeline: HybridAnomalyPipeline = Depends(_get_or_init_pipeline),
) -> HealthResponse:
    """Liveness and readiness probe verifying model initialization."""
    cfg: PipelineSettings = request.app.state.settings
    summary = pipeline.training_summary_
    return HealthResponse(
        status="healthy",
        service=cfg.app_name,
        version=cfg.app_version,
        environment=cfg.environment,
        model_ready=bool(summary),
        n_training_rows=int(summary.get("n_rows", 0)),
    )


def model_info(
    pipeline: HybridAnomalyPipeline = Depends(_get_or_init_pipeline),
) -> dict[str, object]:
    """Return reference model metrics and schema metadata."""
    return {
        "model_name": pipeline.classifier.config.model_name,
        "n_clusters": pipeline.detector.config.n_clusters,
        "distance_threshold": round(pipeline.detector.distance_threshold_, 4),
        "training_summary": pipeline.training_summary_,
    }


def analyze_transactions(
    payload: AnalyzeRequest,
    request: Request,
    pipeline: HybridAnomalyPipeline = Depends(_get_or_init_pipeline),
) -> AnalyzeResponse:
    """Score a batch of bank transactions for cluster anomalies and suspicious risk."""
    cfg: PipelineSettings = request.app.state.settings
    if len(payload.transactions) > cfg.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size exceeds maximum limit of {cfg.max_batch_size}.",
        )

    cache: BoundedPredictionCache = request.app.state.cache
    cache_key = _hash_request(payload)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    start = time.perf_counter()
    raw_df = records_to_dataframe(payload.transactions)
    scored_df = pipeline.analyze(raw_df, include_classification=payload.include_classification)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    response = _format_results(payload.transactions, scored_df, elapsed_ms)
    cache.set(cache_key, response)
    logger.info(
        "Analyzed %d transactions (anomalies=%d, duration=%.2fms).",
        response.summary.total_transactions,
        response.summary.anomaly_count,
        response.summary.processing_ms,
    )
    return response


def _build_router() -> APIRouter:
    """Construct modular API router for health, metadata, and transaction inference."""
    router = APIRouter()
    router.add_api_route("/health", health_check, methods=["GET"], response_model=HealthResponse, tags=["Operations"])
    router.add_api_route("/model-info", model_info, methods=["GET"], tags=["Operations"])
    router.add_api_route(
        "/analyze",
        analyze_transactions,
        methods=["POST"],
        response_model=AnalyzeResponse,
        dependencies=[Depends(_enforce_rate_limit)],
        tags=["Inference"],
    )
    return router


def create_app(settings: PipelineSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance.

    Args:
        settings: Optional injected `PipelineSettings`; defaults to `PipelineSettings.from_env()`.

    Returns:
        Configured `FastAPI` application.
    """
    cfg = settings or PipelineSettings.from_env()
    configure_logging(cfg.log_level)

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
        logger.info("Initializing %s v%s (%s).", cfg.app_name, cfg.app_version, cfg.environment)
        fastapi_app.state.pipeline = _build_ready_pipeline(cfg)
        yield
        logger.info("Shutting down %s cleanly.", cfg.app_name)

    fastapi_app = FastAPI(
        title=cfg.app_name,
        version=cfg.app_version,
        description="Hybrid unsupervised clustering and supervised fraud risk scoring API.",
        lifespan=lifespan,
    )
    fastapi_app.state.settings = cfg
    fastapi_app.state.pipeline = None
    fastapi_app.state.rate_limiter = SlidingWindowRateLimiter(max_requests=cfg.rate_limit_per_minute)
    fastapi_app.state.cache = BoundedPredictionCache(ttl_seconds=cfg.cache_ttl_seconds)
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    )
    fastapi_app.add_exception_handler(Exception, _handle_unhandled_exception)
    fastapi_app.include_router(_build_router())
    return fastapi_app


app = create_app()
