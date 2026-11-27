"""Unified CLI entrypoint for health verification, offline demo pipeline, and FastAPI serving."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Sequence

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from bank_anomaly.config import PipelineSettings, configure_logging  # noqa: E402
from bank_anomaly.pipeline import HybridAnomalyPipeline, run_offline_demo  # noqa: E402
from bank_anomaly.data import generate_synthetic_transactions  # noqa: E402

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Bank Transaction Anomaly Pipeline CLI (--health | --demo | --serve)"
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--health",
        action="store_true",
        help="Verify configuration and offline pipeline readiness.",
    )
    mode_group.add_argument(
        "--demo",
        action="store_true",
        help="Run the end-to-end offline synthetic anomaly & classification pipeline.",
    )
    mode_group.add_argument(
        "--serve",
        action="store_true",
        help="Start the FastAPI HTTP service with uvicorn.",
    )
    parser.add_argument("--rows", type=int, default=400, help="Synthetic row count for --demo.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --demo.")
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "decision_tree", "xgboost"],
        help="Supervised classifier architecture for --demo.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host for --serve.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port for --serve.")
    return parser


def run_health_check(settings: PipelineSettings) -> dict[str, object]:
    """Run deterministic offline initialization check for CLI health verification."""
    sample_df = generate_synthetic_transactions(n_rows=60, seed=settings.random_seed)
    pipeline = HybridAnomalyPipeline()
    summary = pipeline.fit(sample_df)
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "model_ready": True,
        "sample_rows_verified": summary["n_rows"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Execute CLI mode selected by user arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = PipelineSettings.from_env()
    configure_logging(settings.log_level)

    if args.health:
        payload = run_health_check(settings)
        logger.info("Health check passed (version=%s).", settings.app_version)
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 0

    if args.demo:
        logger.info(
            "Starting offline demo pipeline (rows=%d, seed=%d, model=%s).",
            args.rows,
            args.seed,
            args.model,
        )
        result = run_offline_demo(n_rows=args.rows, seed=args.seed, model_name=args.model)
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
        return 0

    if args.serve:
        import uvicorn

        logger.info("Starting FastAPI server on %s:%d.", args.host, args.port)
        uvicorn.run("bank_anomaly.api:app", host=args.host, port=args.port, reload=False)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
