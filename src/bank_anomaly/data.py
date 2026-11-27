"""Dataset schema validation, deterministic offline generation, and leak-free preprocessing."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
import io
import logging

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
import requests

logger = logging.getLogger(__name__)

RAW_SCHEMA_COLUMNS: tuple[str, ...] = (
    "TransactionID",
    "AccountID",
    "TransactionAmount",
    "PreviousTransactionDate",
    "TransactionType",
    "Location",
    "DeviceID",
    "IP Address",
    "MerchantID",
    "Channel",
    "CustomerAge",
    "CustomerOccupation",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
    "TransactionDate",
)

MODEL_NUMERICAL_FEATURES: tuple[str, ...] = (
    "TransactionAmount",
    "CustomerAge",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
)

MODEL_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "TransactionType",
    "Location",
    "Channel",
    "CustomerOccupation",
)

MODEL_FEATURES: tuple[str, ...] = MODEL_NUMERICAL_FEATURES + MODEL_CATEGORICAL_FEATURES

TRANSACTION_TYPES: tuple[str, ...] = ("Debit", "Credit")
CHANNELS: tuple[str, ...] = ("Branch", "ATM", "Online")
OCCUPATIONS: tuple[str, ...] = ("Student", "Doctor", "Engineer", "Retired")
LOCATIONS: tuple[str, ...] = (
    "San Diego",
    "Houston",
    "Mesa",
    "Raleigh",
    "Atlanta",
    "Fort Worth",
    "Austin",
    "Seattle",
    "Denver",
    "Chicago",
)

MIN_SYNTHETIC_ROWS = 10
DEFAULT_ANOMALY_RATE = 0.08
DEFAULT_IQR_MULTIPLIER = 1.5


class TransactionRecord(BaseModel):
    """Validated schema for a single bank transaction record."""

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    TransactionID: str = Field(default="TX000001", min_length=1, max_length=64)
    AccountID: str = Field(default="AC00001", min_length=1, max_length=64)
    TransactionAmount: float = Field(..., ge=0.0, le=10_000_000.0)
    PreviousTransactionDate: str = Field(default="2024-10-01 10:00:00", min_length=1)
    TransactionType: str = Field(..., min_length=1, max_length=32)
    Location: str = Field(..., min_length=1, max_length=64)
    DeviceID: str = Field(default="D000001", min_length=1, max_length=64)
    IP_Address: str = Field(default="192.0.2.1", alias="IP Address", min_length=1, max_length=64)
    MerchantID: str = Field(default="M001", min_length=1, max_length=64)
    Channel: str = Field(..., min_length=1, max_length=32)
    CustomerAge: float = Field(..., ge=18.0, le=120.0)
    CustomerOccupation: str = Field(..., min_length=1, max_length=64)
    TransactionDuration: float = Field(..., gt=0.0, le=86400.0)
    LoginAttempts: float = Field(..., ge=1.0, le=100.0)
    AccountBalance: float = Field(..., ge=0.0, le=100_000_000.0)
    TransactionDate: str = Field(default="2024-11-04 08:00:00", min_length=1)

    @field_validator("TransactionType", "Location", "Channel", "CustomerOccupation")
    @classmethod
    def validate_non_empty_category(cls, value: str) -> str:
        """Ensure categorical strings are non-empty after stripping whitespace."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Categorical field must not be blank.")
        return cleaned

    def to_raw_dict(self) -> dict[str, object]:
        """Convert record into dictionary keyed by exact raw 16-column dataset headers."""
        return self.model_dump(by_alias=True)


def _sample_regimes(
    rng: np.random.Generator,
    n_rows: int,
    include_anomalies: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate correlated numerical and categorical behavioral arrays for synthetic data."""
    n_anom = max(1, int(round(n_rows * DEFAULT_ANOMALY_RATE))) if include_anomalies else 0
    n_norm = n_rows - n_anom
    half_norm = n_norm // 2

    # Two distinct customer behavioral clusters for realistic segmentation
    amounts_norm = np.concatenate([
        rng.gamma(shape=1.6, scale=75.0, size=half_norm),
        rng.gamma(shape=2.6, scale=140.0, size=n_norm - half_norm),
    ])
    balances_norm = np.concatenate([
        rng.normal(loc=2200.0, scale=700.0, size=half_norm).clip(120.0, 6500.0),
        rng.normal(loc=8600.0, scale=2100.0, size=n_norm - half_norm).clip(3500.0, 15000.0),
    ])
    ages_norm = np.concatenate([
        rng.integers(18, 38, size=half_norm).astype(float),
        rng.integers(40, 80, size=n_norm - half_norm).astype(float),
    ])
    durations_norm = rng.normal(loc=115.0, scale=42.0, size=n_norm).clip(12.0, 260.0)
    attempts_norm = rng.choice([1.0, 2.0], size=n_norm, p=[0.93, 0.07])
    occ_norm = np.concatenate([
        rng.choice(["Student", "Engineer"], size=half_norm, p=[0.7, 0.3]),
        rng.choice(["Doctor", "Retired", "Engineer"], size=n_norm - half_norm, p=[0.4, 0.35, 0.25]),
    ])
    channel_norm = np.concatenate([
        rng.choice(["Online", "ATM", "Branch"], size=half_norm, p=[0.6, 0.25, 0.15]),
        rng.choice(["Branch", "ATM", "Online"], size=n_norm - half_norm, p=[0.55, 0.3, 0.15]),
    ])

    if n_anom == 0:
        return amounts_norm, balances_norm, ages_norm, durations_norm, attempts_norm, occ_norm, channel_norm

    amounts_anom = rng.uniform(1150.0, 1950.0, size=n_anom)
    balances_anom = rng.uniform(100.0, 900.0, size=n_anom)
    ages_anom = rng.integers(19, 75, size=n_anom).astype(float)
    durations_anom = rng.choice([10.0, 285.0], size=n_anom)
    attempts_anom = rng.integers(3, 6, size=n_anom).astype(float)
    occ_anom = rng.choice(OCCUPATIONS, size=n_anom)
    channel_anom = rng.choice(["Online", "ATM"], size=n_anom, p=[0.7, 0.3])

    return (
        np.concatenate([amounts_norm, amounts_anom]),
        np.concatenate([balances_norm, balances_anom]),
        np.concatenate([ages_norm, ages_anom]),
        np.concatenate([durations_norm, durations_anom]),
        np.concatenate([attempts_norm, attempts_anom]),
        np.concatenate([occ_norm, occ_anom]),
        np.concatenate([channel_norm, channel_anom]),
    )


def generate_synthetic_transactions(
    n_rows: int = 500,
    seed: int = 42,
    include_anomalies: bool = True,
) -> pd.DataFrame:
    """Generate deterministic offline synthetic bank transactions matching all 16 raw columns.

    Args:
        n_rows: Number of rows to generate (must be >= 10).
        seed: Random seed for reproducible generation.
        include_anomalies: Whether to inject high-amount/high-login-attempt anomaly cases.

    Returns:
        DataFrame with exact 16 columns matching the Bank Transactions dataset schema.

    Raises:
        ValueError: If `n_rows` is smaller than 10.
    """
    if n_rows < MIN_SYNTHETIC_ROWS:
        raise ValueError(f"n_rows must be >= {MIN_SYNTHETIC_ROWS}, got {n_rows}.")

    rng = np.random.default_rng(seed)
    amounts, balances, ages, durations, attempts, occupations, channels = _sample_regimes(
        rng, n_rows, include_anomalies
    )
    perm = rng.permutation(n_rows)

    base_prev = datetime(2023, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    base_curr = datetime(2024, 11, 4, 8, 0, 0, tzinfo=timezone.utc)
    prev_offsets = rng.integers(0, 365 * 86400, size=n_rows)
    curr_offsets = rng.integers(0, 3600, size=n_rows)

    octets_c = rng.integers(1, 255, size=n_rows)
    octets_d = rng.integers(1, 255, size=n_rows)

    df = pd.DataFrame({
        "TransactionID": [f"TX{idx:06d}" for idx in range(1, n_rows + 1)],
        "AccountID": [f"AC{int(acc):05d}" for acc in rng.integers(1, 496, size=n_rows)],
        "TransactionAmount": np.round(amounts[perm], 2),
        "PreviousTransactionDate": [
            (base_prev + timedelta(seconds=int(sec))).strftime("%Y-%m-%d %H:%M:%S")
            for sec in prev_offsets
        ],
        "TransactionType": rng.choice(TRANSACTION_TYPES, size=n_rows, p=[0.77, 0.23]),
        "Location": rng.choice(LOCATIONS, size=n_rows),
        "DeviceID": [f"D{int(dev):06d}" for dev in rng.integers(1, 500, size=n_rows)],
        "IP Address": [f"192.0.{int(c)}.{int(d)}" for c, d in zip(octets_c, octets_d, strict=True)],
        "MerchantID": [f"M{int(m):03d}" for m in rng.integers(1, 101, size=n_rows)],
        "Channel": channels[perm],
        "CustomerAge": ages[perm],
        "CustomerOccupation": occupations[perm],
        "TransactionDuration": np.round(durations[perm], 1),
        "LoginAttempts": attempts[perm],
        "AccountBalance": np.round(balances[perm], 2),
        "TransactionDate": [
            (base_curr + timedelta(seconds=int(sec))).strftime("%Y-%m-%d %H:%M:%S")
            for sec in curr_offsets
        ],
    })
    logger.info("Generated %d deterministic synthetic bank transactions (seed=%d).", len(df), seed)
    return df


def fetch_remote_transactions(url: str, timeout_seconds: float = 10.0) -> pd.DataFrame:
    """Fetch remote transaction CSV over HTTP with explicit timeout validation.

    Args:
        url: Remote CSV endpoint URL.
        timeout_seconds: HTTP request timeout in seconds (> 0).

    Returns:
        Loaded raw DataFrame.

    Raises:
        ValueError: If URL is blank, timeout is non-positive, or payload is empty.
        requests.RequestException: If network fetch fails.
    """
    if not url or not url.strip():
        raise ValueError("Remote dataset URL must not be empty.")
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}.")

    response = requests.get(url.strip(), timeout=timeout_seconds)
    response.raise_for_status()
    if not response.text.strip():
        raise ValueError("Remote CSV response body is empty.")

    df = pd.read_csv(io.StringIO(response.text))
    logger.info("Fetched %d remote transaction rows.", len(df))
    return df


def load_transactions_with_fallback(
    url: str | None = None,
    n_rows: int = 500,
    seed: int = 42,
    timeout_seconds: float = 10.0,
) -> pd.DataFrame:
    """Load transactions from URL if provided, falling back to offline synthetic data on error.

    Args:
        url: Optional remote CSV URL. When None, returns synthetic data immediately offline.
        n_rows: Fallback synthetic row count.
        seed: Fallback synthetic seed.
        timeout_seconds: Network timeout if `url` is specified.

    Returns:
        Validated transaction DataFrame.
    """
    if url is None:
        return generate_synthetic_transactions(n_rows=n_rows, seed=seed)

    try:
        return fetch_remote_transactions(url=url, timeout_seconds=timeout_seconds)
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Remote dataset fetch failed (%s); falling back to synthetic data.", exc)
        return generate_synthetic_transactions(n_rows=n_rows, seed=seed)


def records_to_dataframe(records: Sequence[TransactionRecord]) -> pd.DataFrame:
    """Convert validated Pydantic `TransactionRecord` sequence to raw DataFrame.

    Args:
        records: Non-empty sequence of `TransactionRecord` objects.

    Returns:
        DataFrame containing the 16 raw schema columns.

    Raises:
        ValueError: If `records` is empty.
    """
    if not records:
        raise ValueError("At least one TransactionRecord is required.")
    return pd.DataFrame([record.to_raw_dict() for record in records])


def clean_transactions(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Validate required model columns and clean missing/duplicate transaction rows.

    Args:
        df: Input DataFrame containing transaction features.
        drop_duplicates: Whether to deduplicate identical feature rows.

    Returns:
        Cleaned copy of DataFrame containing required modeling columns.

    Raises:
        ValueError: If DataFrame is empty or missing required modeling columns.
    """
    if df.empty:
        raise ValueError("Input transaction DataFrame must not be empty.")

    missing = [col for col in MODEL_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required transaction columns: {missing}")

    cleaned = df.dropna(subset=list(MODEL_FEATURES)).copy()
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates().copy()

    if cleaned.empty:
        raise ValueError("No valid transaction rows remain after cleaning missing values.")

    for col in MODEL_NUMERICAL_FEATURES:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned = cleaned.dropna(subset=list(MODEL_NUMERICAL_FEATURES)).copy()
    if cleaned.empty:
        raise ValueError("All numeric transaction columns became NaN after coercion.")

    for col in MODEL_CATEGORICAL_FEATURES:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    return cleaned


def compute_simultaneous_iqr_mask(
    df: pd.DataFrame,
    numerical_cols: Sequence[str] = MODEL_NUMERICAL_FEATURES,
    multiplier: float = DEFAULT_IQR_MULTIPLIER,
) -> pd.Series:
    """Compute a simultaneous IQR inlier mask without mutating DataFrame between columns.

    Evaluating quantiles sequentially after filtering each column shifts subsequent quartiles.
    This helper computes all column bounds against the initial un-mutated DataFrame first.

    Args:
        df: Input DataFrame.
        numerical_cols: Columns to evaluate IQR fences on.
        multiplier: IQR fence multiplier (> 0, standard Tukey fence = 1.5).

    Returns:
        Boolean Series indexed like `df` where True indicates an inlier row.

    Raises:
        ValueError: If `df` is empty, `multiplier <= 0`, or columns are missing.
    """
    if df.empty:
        raise ValueError("Cannot compute IQR mask on an empty DataFrame.")
    if multiplier <= 0:
        raise ValueError(f"IQR multiplier must be > 0, got {multiplier}.")

    missing = [col for col in numerical_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for IQR mask: {missing}")

    inlier_mask = pd.Series(True, index=df.index)
    for col in numerical_cols:
        q1 = float(df[col].quantile(0.25))
        q3 = float(df[col].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        inlier_mask &= (df[col] >= lower) & (df[col] <= upper)

    return inlier_mask


def filter_iqr_outliers(
    df: pd.DataFrame,
    numerical_cols: Sequence[str] = MODEL_NUMERICAL_FEATURES,
    multiplier: float = DEFAULT_IQR_MULTIPLIER,
) -> pd.DataFrame:
    """Filter IQR outliers simultaneously across numerical columns.

    Args:
        df: Input DataFrame.
        numerical_cols: Numerical columns to inspect.
        multiplier: Tukey IQR multiplier.

    Returns:
        Filtered DataFrame copy containing inliers only.
    """
    mask = compute_simultaneous_iqr_mask(df=df, numerical_cols=numerical_cols, multiplier=multiplier)
    filtered = df.loc[mask].copy()
    logger.info("Simultaneous IQR filter kept %d/%d rows.", len(filtered), len(df))
    return filtered
