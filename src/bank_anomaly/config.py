"""Externalized configuration and structured logging for Bank Transaction Anomaly Pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_ALLOWED_ORIGINS: tuple[str, ...] = (
    "http://localhost:3000",
    "http://127.0.0.1:8000",
)
VALID_LOG_LEVELS: frozenset[str] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def _parse_int_env(name: str, default: int, minimum: int = 1) -> int:
    """Parse an integer environment variable with lower-bound validation."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}.") from exc
    if value < minimum:
        raise ValueError(f"Environment variable {name} must be >= {minimum}, got {value}.")
    return value


def _parse_float_env(name: str, default: float, minimum: float = 0.0) -> float:
    """Parse a float environment variable with lower-bound validation."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got {raw!r}.") from exc
    if value <= minimum:
        raise ValueError(f"Environment variable {name} must be > {minimum}, got {value}.")
    return value


def _parse_origins_env(name: str, default: tuple[str, ...]) -> list[str]:
    """Parse comma-separated CORS allowed origins from environment."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return list(default)
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or list(default)


@dataclass(frozen=True)
class PipelineSettings:
    """Immutable runtime settings resolved from environment variables."""

    app_name: str = "Bank Transaction Anomaly Pipeline"
    app_version: str = "1.0.0"
    environment: str = "development"
    log_level: str = "INFO"
    random_seed: int = 42
    n_clusters: int = 2
    anomaly_quantile: float = 0.95
    zscore_threshold: float = 2.5
    request_timeout_seconds: float = 10.0
    max_batch_size: int = 5000
    rate_limit_per_minute: int = 120
    cache_ttl_seconds: int = 300
    model_dir: Path = Path("models")
    allowed_origins: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_ORIGINS))

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> PipelineSettings:
        """Load configuration from optional dotenv file and OS environment variables.

        Args:
            env_file: Optional path to a `.env` file.

        Returns:
            Validated `PipelineSettings` instance.

        Raises:
            ValueError: If numeric bounds or log level settings are invalid.
        """
        load_dotenv(dotenv_path=env_file, override=False)
        level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
        if level not in VALID_LOG_LEVELS:
            raise ValueError(f"Unsupported LOG_LEVEL {level!r}; choose from {sorted(VALID_LOG_LEVELS)}.")

        quantile = _parse_float_env("ANOMALY_QUANTILE", 0.95, minimum=0.0)
        if quantile >= 1.0:
            raise ValueError(f"ANOMALY_QUANTILE must be < 1.0, got {quantile}.")

        return cls(
            app_name=os.getenv("APP_NAME", "Bank Transaction Anomaly Pipeline").strip(),
            app_version=os.getenv("APP_VERSION", "1.0.0").strip(),
            environment=os.getenv("APP_ENV", "development").strip(),
            log_level=level,
            random_seed=_parse_int_env("RANDOM_SEED", 42, minimum=0),
            n_clusters=_parse_int_env("N_CLUSTERS", 2, minimum=2),
            anomaly_quantile=quantile,
            zscore_threshold=_parse_float_env("ZSCORE_THRESHOLD", 2.5, minimum=0.0),
            request_timeout_seconds=_parse_float_env("REQUEST_TIMEOUT_SECONDS", 10.0, minimum=0.0),
            max_batch_size=_parse_int_env("MAX_BATCH_SIZE", 5000, minimum=1),
            rate_limit_per_minute=_parse_int_env("RATE_LIMIT_PER_MINUTE", 120, minimum=1),
            cache_ttl_seconds=_parse_int_env("CACHE_TTL_SECONDS", 300, minimum=1),
            model_dir=Path(os.getenv("MODEL_DIR", "models")),
            allowed_origins=_parse_origins_env("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS),
        )


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured root logging for CLI, pipeline, and API execution.

    Args:
        level: Logging severity level name.

    Returns:
        Configured package logger.
    """
    normalized = level.strip().upper()
    resolved_level = getattr(logging, normalized, logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    pkg_logger = logging.getLogger("bank_anomaly")
    pkg_logger.setLevel(resolved_level)
    return pkg_logger
