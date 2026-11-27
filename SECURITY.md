# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in this repository:

1. **Do NOT open a public GitHub issue** for undisclosed vulnerabilities.
2. Submit a private report via GitHub Security Advisories (`Security -> Advisories -> Report a vulnerability`).
3. Include affected file paths, line numbers, reproduction steps, and estimated severity.
4. Initial acknowledgement will be provided within **48 hours**.

## Security Controls

- **Secret Hygiene**: `.env` and `.env.*` files are excluded via `.gitignore`; zero active credentials are stored in source control.
- **Artifact Deserialization Boundary**: Runtime services never deserialize legacy `.h5` pickle artifacts; models are trained deterministically or loaded strictly from verified `.joblib` pipelines.
- **Input Validation & Defense-in-Depth**: FastAPI enforces Pydantic `TransactionRecord` schemas, sliding-window rate limiting (`429`), restrictive CORS origins, explicit HTTP client timeouts, and masked `500` error boundaries.
- **Automated Vulnerability Management**: Weekly Dependabot updates (`semver-major` ignored), Semgrep SAST, and Trivy container image scanning in CI.
- **Container Hardening**: Multi-stage `python:3.11-slim` Docker image running as non-root `appuser`.
