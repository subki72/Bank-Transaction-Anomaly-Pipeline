"""Unit tests for dataset schema validation, synthetic generation, and IQR filtering."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
from pydantic import ValidationError
import pytest
import requests

from bank_anomaly.data import (
    MODEL_NUMERICAL_FEATURES,
    RAW_SCHEMA_COLUMNS,
    TransactionRecord,
    clean_transactions,
    compute_simultaneous_iqr_mask,
    fetch_remote_transactions,
    filter_iqr_outliers,
    generate_synthetic_transactions,
    load_transactions_with_fallback,
)


def test_generate_synthetic_transactions_matches_16_column_schema() -> None:
    """Synthetic generator emits exact 16 raw columns deterministically."""
    df_a = generate_synthetic_transactions(n_rows=50, seed=7)
    df_b = generate_synthetic_transactions(n_rows=50, seed=7)

    assert tuple(df_a.columns) == RAW_SCHEMA_COLUMNS
    assert len(df_a) == 50
    assert df_a.isna().sum().sum() == 0
    pd.testing.assert_frame_equal(df_a, df_b)


def test_generate_synthetic_transactions_rejects_too_few_rows() -> None:
    """Guard clause rejects synthetic row counts below minimum threshold."""
    with pytest.raises(ValueError, match="n_rows must be >= 10"):
        generate_synthetic_transactions(n_rows=5)


def test_transaction_record_validation_and_alias_mapping() -> None:
    """Pydantic TransactionRecord enforces positive bounds and maps IP Address alias."""
    valid = TransactionRecord(
        TransactionAmount=125.50,
        TransactionType="Debit",
        Location="Houston",
        Channel="Online",
        CustomerAge=34.0,
        CustomerOccupation="Engineer",
        TransactionDuration=85.0,
        LoginAttempts=1.0,
        AccountBalance=4500.0,
        **{"IP Address": "198.51.100.12"},
    )
    dumped = valid.to_raw_dict()
    assert dumped["IP Address"] == "198.51.100.12"

    with pytest.raises(ValidationError):
        TransactionRecord(
            TransactionAmount=-5.0,
            TransactionType="Debit",
            Location="Houston",
            Channel="Online",
            CustomerAge=16.0,
            CustomerOccupation="Student",
            TransactionDuration=50.0,
            LoginAttempts=1.0,
            AccountBalance=100.0,
        )


def test_simultaneous_iqr_mask_prevents_sequential_quantile_drift() -> None:
    """Simultaneous IQR bounds remain invariant to column ordering unlike sequential mutation."""
    df = pd.DataFrame(
        {
            "TransactionAmount": [18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 500.0],
            "AccountBalance": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0],
            "CustomerAge": [25.0, 27.0, 30.0, 31.0, 32.0, 35.0, 36.0, 37.0],
            "TransactionDuration": [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0],
            "LoginAttempts": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    mask_forward = compute_simultaneous_iqr_mask(df, list(MODEL_NUMERICAL_FEATURES))
    mask_reversed = compute_simultaneous_iqr_mask(df, list(reversed(MODEL_NUMERICAL_FEATURES)))

    pd.testing.assert_series_equal(mask_forward, mask_reversed)
    filtered = filter_iqr_outliers(df, list(MODEL_NUMERICAL_FEATURES))
    assert 500.0 not in filtered["TransactionAmount"].tolist()
    assert len(filtered) == 7


def test_clean_transactions_removes_missing_and_validates_schema() -> None:
    """clean_transactions drops incomplete rows and raises on missing required columns."""
    df = generate_synthetic_transactions(n_rows=20, seed=9)
    df.loc[0, "TransactionAmount"] = None
    cleaned = clean_transactions(df)
    assert len(cleaned) == 19

    with pytest.raises(ValueError, match="Missing required transaction columns"):
        clean_transactions(df.drop(columns=["Channel"]))


def test_load_transactions_with_fallback_handles_offline_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Network errors trigger deterministic offline fallback without crashing."""

    def _raise_timeout(*args: object, **kwargs: object) -> object:
        raise requests.Timeout("Simulated timeout")

    monkeypatch.setattr(requests, "get", _raise_timeout)
    df = load_transactions_with_fallback(
        url="https://example.invalid/dataset.csv",
        n_rows=30,
        seed=11,
        timeout_seconds=1.0,
    )
    assert len(df) == 30
    assert tuple(df.columns) == RAW_SCHEMA_COLUMNS


def test_fetch_remote_transactions_parses_valid_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch_remote_transactions enforces timeout and parses CSV payload."""
    sample_csv = generate_synthetic_transactions(n_rows=12, seed=3).to_csv(index=False)
    mock_resp = MagicMock()
    mock_resp.text = sample_csv
    mock_resp.raise_for_status.return_value = None

    called_timeouts: list[object] = []

    def _mock_get(url: str, timeout: float) -> MagicMock:
        called_timeouts.append(timeout)
        return mock_resp

    monkeypatch.setattr(requests, "get", _mock_get)
    df = fetch_remote_transactions("https://example.invalid/tx.csv", timeout_seconds=5.0)
    assert len(df) == 12
    assert called_timeouts == [5.0]
