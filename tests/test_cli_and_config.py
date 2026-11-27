"""Tests for externalized PipelineSettings configuration and CLI main.py modes."""

from __future__ import annotations

import json

import pytest

from bank_anomaly.config import PipelineSettings
import main as cli_main


def test_pipeline_settings_from_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """PipelineSettings reads valid overrides and rejects invalid bounds."""
    monkeypatch.setenv("APP_ENV", "staging")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("N_CLUSTERS", "3")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://app.example.com,https://admin.example.com")

    settings = PipelineSettings.from_env()
    assert settings.environment == "staging"
    assert settings.log_level == "DEBUG"
    assert settings.n_clusters == 3
    assert settings.allowed_origins == ["https://app.example.com", "https://admin.example.com"]

    monkeypatch.setenv("LOG_LEVEL", "INVALID_LEVEL")
    with pytest.raises(ValueError, match="Unsupported LOG_LEVEL"):
        PipelineSettings.from_env()


def test_cli_health_and_demo_modes(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI --health and --demo exit 0 and emit valid JSON payloads offline."""
    health_exit = cli_main.main(["--health"])
    assert health_exit == 0
    health_out = json.loads(capsys.readouterr().out)
    assert health_out["status"] == "healthy"
    assert health_out["model_ready"] is True

    demo_exit = cli_main.main(["--demo", "--rows", "80", "--seed", "7", "--model", "decision_tree"])
    assert demo_exit == 0
    demo_out = json.loads(capsys.readouterr().out)
    assert demo_out["status"] == "ok"
    assert demo_out["summary"]["n_rows"] == 80
    assert demo_out["model_name"] == "decision_tree"
