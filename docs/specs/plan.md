# Implementation Plan: Sentiment Analysis Service

**Branch**: `001-sentiment-analysis-service` | **Date**: 2026-07-29 | **Spec**: `specs/001-sentiment-analysis-service/spec.md`

**Input**: Feature specification from `/specs/001-sentiment-analysis-service/spec.md`

## Summary

A production-ready sentiment analysis service that exposes a real Hugging Face DistilBERT model (pinned revision) via both FastAPI HTTP and CLI interfaces, evaluated end-to-end on a public 3,000-sentence dataset (92.80% accuracy). Implements dual backends (local on-device and Hugging Face Inference API), in-process LRU caching with no TTL (pure-function property), sliding-window rate limiting per client IP, and comprehensive error handling with uniform JSON responses. Dockerized with model weights baked in, verified by CI smoke test.

## Technical Context

**Language/Version**: Python 3.11

**Primary Dependencies**: FastAPI, transformers, torch, huggingface_hub, stdlib unittest (no pytest)

**Storage**: Local filesystem (dataset download to `data/` directory), no persistent state. In-process cache (in-memory LRU).

**Testing**: stdlib `unittest.discover`, fully offline (all backend calls mocked), 34 tests, <1 second runtime

**Target Platform**: Linux/macOS server (CLI) and containerized HTTP service (Docker)

**Project Type**: CLI + HTTP API microservice

**Performance Goals**: Single `/analyze` <2s (first, cold model); cached <100ms. Batch `/analyze/batch` (64 texts) <10s. Rate limit: 30 requests per 60 seconds per IP.

**Constraints**: Max text length 2000 chars, max batch size 64, no GPU (CPU-only with torch), model weights ~268 MB (downloaded once, cached locally).

**Scale/Scope**: 3 CLI subcommands, 5 HTTP endpoints, 2 interchangeable backends, 34 unit/integration tests, 1 Dockerfile, 1 CI workflow.

## Constitution Check

1. **Principle I (Real Model, Real Data)**: PASS — DistilBERT model pinned at revision `714eb0fa89d2f80546fda750413ed43d93601a13` is a real model. UCI "Sentiment Labelled Sentences" dataset is public (Kotzias 2015, CC BY 4.0), not synthetic. Accuracy measured once against full 3,000 sentences (92.80%), reported as-is without rounding or cherry-picking per source.

2. **Principle II (Secrets Only Via Environment)**: PASS — `HF_TOKEN` is read only via `os.getenv()` / `os.environ` in `sentiment.py`, never hardcoded, never logged, never included in error messages beyond a presence boolean. `.gitignore` excludes `.env*` files. Missing token causes ConfigError at import, exit code 3.

3. **Principle III (Dual Interface, Shared Core)**: PASS — All logic lives in `sentiment.py` (model backend, cache, validation, rate limiter). `app.py` (FastAPI) and `cli.py` (argparse) are thin wrappers that delegate to `sentiment.py` functions; no code duplication of validation or backend dispatch.

4. **Principle IV (Fail Fast, Fail Honest)**: PASS — ConfigError raised at import time for bad `MODEL_BACKEND` or missing `HF_TOKEN`. Every error (input validation, backend failure, rate limit) returns a uniform JSON shape `{"error": "<code>", "message": "<actionable text>"}` with no secrets. 16 distinct error codes mapped to HTTP status codes (422, 429, 500, 502, 503, 504).

5. **Principle V (Offline-Safe by Default)**: PASS — `MODEL_BACKEND=local` (default) requires no credentials. `/health` endpoint never loads model or touches network (uses `request.client.host` for rate-limit key, pure Python, no external calls). Model pipeline lazy-loads on first `/analyze` request, not at import.

6. **Principle VI (English-Only Deliverable)**: PASS — All code, docstrings, log messages, error messages, README, and spec are in English. No Spanish or mixed-language content in the repository.

7. **Principle VII (Measured, Not Guessed)**: PASS — Model revision SHA copied from Hugging Face Files-and-versions tab. Accuracy (92.80% overall, per-source breakdown) comes from one documented, reproducible run: `python cli.py evaluate` on 2026-07-29 at 16:23 sentences/second over 3,000 sentences in 184.89 seconds. Dependency versions pinned in `requirements.txt` (torch, transformers, etc.).

## Project Structure

### Documentation (this feature)

```text
specs/001-sentiment-analysis-service/
├── spec.md              # Feature specification (user stories, requirements, success criteria)
├── plan.md              # This file (technical context, constitution, structure)
└── tasks.md             # Task list (retroactive documentation of what was built)
```

### Source Code (repository root — flat, single-module layout)

```text
.
├── sentiment.py                 # Service layer: backends, cache, validation, rate limiter
├── app.py                       # FastAPI HTTP API: routes, error handlers, middleware
├── cli.py                       # CLI: fetch-data, analyze, evaluate subcommands
├── requirements.txt             # Pinned dependencies (transformers, torch, etc.)
├── Dockerfile                   # Build with baked-in model weights, python:3.11-slim
├── .github/workflows/ci.yml     # CI: test suite + Docker build & smoke test
├── tests/
│   ├── test_sentiment.py        # Unit tests for sentiment.py (label norm, validation, cache, rate limit)
│   └── test_api.py              # Integration tests for app.py via FastAPI TestClient
└── data/                        # Dataset directory (gitignored, created by fetch-data)
```

**Structure Decision**: Single-project flat layout (no `src/`, no `models/`, no `services/`) is intentional per constitution. Rationale: only 3 user-facing modules (sentiment.py, app.py, cli.py) with <1000 LOC each. Adding abstractions (backend ABC registry, config.py, schemas.py) would add boilerplate without benefit for this scope. The distinction between modules is already clear: sentiment.py is the service layer, app.py is HTTP, cli.py is CLI.

## Complexity Tracking

No violations. Constitution's "Additional Constraints" explicitly forbid premature abstraction:

- No backend ABC + registry; a dict of two functions in `_BACKENDS` suffices.
- No `config.py`; six `os.getenv()` calls at the top of `sentiment.py` are clear and live where they're used.
- No `schemas.py`; two Pydantic models in `app.py` (AnalyzeRequest, BatchRequest, AnalyzeResponse) are co-located with their routes.
- No `dataset.py`; `load_dataset()` and `find_labelled_files()` live in `cli.py` where they're called.
- No sklearn/pandas; accuracy, precision, recall, F1, and confusion matrix are <50 lines of arithmetic on counters.

This keeps the codebase readable, testable, and maintainable. Upgrade path (per constitution): if the service scales to multiple processes or high throughput, introduce Redis for shared cache/limiter keyed the same way (no local state restructuring needed).
