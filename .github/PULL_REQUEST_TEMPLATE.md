## Summary
<!-- Describe the change, bug fix, or architectural improvement clearly. -->

## Type of Change
- [ ] Bug fix (non-breaking)
- [ ] Feature (non-breaking)
- [ ] Security / Dependency update
- [ ] Documentation / Governance

## Verification Checklist
- [ ] `python -m ruff check src tests main.py` passes with 0 errors
- [ ] `python -m ruff format --check src tests main.py` passes
- [ ] `python -m flake8 --max-line-length=120 src tests main.py` passes
- [ ] `python -m pytest tests/ -v --cov=src --cov-fail-under=75` passes offline
- [ ] `python main.py --health` and `python main.py --demo` verified locally
- [ ] No secrets, `.env` values, or untrusted `.h5` pickle loaders introduced
