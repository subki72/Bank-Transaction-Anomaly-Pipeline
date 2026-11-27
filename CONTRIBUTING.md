# Contributing Guide

Thank you for contributing to **Bank Transaction Anomaly Pipeline**.

## Local Development Setup

```bash
git clone https://github.com/subki72/Bank-Transaction-Anomaly-Pipeline.git
cd Bank-Transaction-Anomaly-Pipeline
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Quality Gates

Before opening a Pull Request, ensure every check passes locally:

```bash
python -m ruff check src tests main.py
python -m ruff format --check src tests main.py
python -m flake8 --max-line-length=120 src tests main.py
python -m pytest tests/ -v --cov=src --cov-fail-under=75
python main.py --health
python main.py --demo --rows 120
```

## Coding & Testing Standards

- Follow **Python 3.11+** type hints, Google-style docstrings, and `logging.getLogger(__name__)` (avoid bare `print()` inside `src/`).
- Never introduce network dependencies in unit/integration tests; mock HTTP calls or use `generate_synthetic_transactions`.
- Never load legacy `.h5` pickle files in `src/` or `api.py`.
- Use conventional commit messages (for example: `feat(api): ...`, `fix(ml): ...`, `test(data): ...`, `docs(runbook): ...`).
