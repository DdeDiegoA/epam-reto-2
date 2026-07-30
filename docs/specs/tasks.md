# Tasks: Sentiment Analysis Service

**Input**: Design documents from `/specs/001-sentiment-analysis-service/`

**Prerequisites**: plan.md (implemented), spec.md (implemented)

**Status**: All tasks complete. This file is written retroactively after implementation for traceability — it documents what was built, not a TDD roadmap.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- File paths are real and verified

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create project structure (sentiment.py, app.py, cli.py, tests/, Dockerfile, .github/workflows/)
- [X] T002 Initialize Python 3.11 project with requirements.txt (FastAPI, transformers, torch, huggingface_hub)
- [X] T003 [P] Configure .gitignore to exclude .env*, __pycache__, .venv, data/, and model weights
- [X] T004 [P] Set up GitHub Actions CI workflow in .github/workflows/ci.yml to run tests + Docker smoke test

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core service layer that all user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 [P] Implement sentiment.py service layer: environment variable configuration (MODEL_ID, MODEL_REVISION, BACKEND, CACHE_SIZE, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW, HF_TIMEOUT)
- [X] T006 [P] Implement exception hierarchy in sentiment.py: ConfigError, InputError, BackendError with proper error codes and HTTP status mapping
- [X] T007 Implement local backend (_run_local) in sentiment.py using transformers.pipeline, lazy-load on first use, thread-safe initialization
- [X] T008 Implement Hugging Face backend (_run_hf) in sentiment.py using huggingface_hub.InferenceClient, with timeout and error handling
- [X] T009 Implement label normalization (_normalize_label) mapping POSITIVE/NEGATIVE/LABEL_0/LABEL_1 to lowercase positive/negative
- [X] T010 [P] Implement input validation functions in sentiment.py: validate_text (1–2000 chars), validate_batch (1–64 items, each 1–2000 chars)
- [X] T011 Implement LRU cache in sentiment.py (functools.lru_cache on _cached_classify, key: backend/model_id/revision/text, no TTL)
- [X] T012 Implement RateLimiter class in sentiment.py (sliding-window log per client IP, O(n) deque cleanup, thread-safe)
- [X] T013 Implement core analyze() function in sentiment.py dispatching to _cached_classify, handling both backends
- [X] T014 Implement backend_info() in sentiment.py returning dict with model, revision, backend, HF_TOKEN presence (boolean only, no token value)
- [X] T015 Add clear_cache() function in sentiment.py for testing

**Checkpoint**: Foundation ready — user story implementation can now begin

---

## Phase 3: User Story 1 — Single-Text Sentiment Analysis (Priority: P1)

**Goal**: Enable classification of a single text via API or CLI with confidence score

**Independent Test**: (a) `POST /analyze` with `{"text": "I love this."}` returns 200 with label, score, model, revision, backend, cached flag. (b) `python cli.py analyze "I love this."` outputs summary. (c) Empty/too-long text returns 422 with error code `validation_error`.

### Tests for User Story 1 ⚠️

- [X] T016 [P] [US1] Contract test: label normalization (POSITIVE→positive, NEGATIVE→negative, LABEL_0→negative, LABEL_1→positive) in tests/test_sentiment.py
- [X] T017 [P] [US1] Contract test: input validation (empty text, too-long text, valid text) in tests/test_sentiment.py
- [X] T018 [P] [US1] Contract test: cache hits and misses tracked correctly in tests/test_sentiment.py
- [X] T019 [P] [US1] Integration test: POST /analyze happy path in tests/test_api.py
- [X] T020 [P] [US1] Integration test: POST /analyze with invalid input (empty, too-long) returns 422 in tests/test_api.py
- [X] T021 [US1] Unit test: _classify backend dispatch in tests/test_sentiment.py

### Implementation for User Story 1

- [X] T022 [P] [US1] Implement AnalyzeRequest and AnalyzeResponse Pydantic models in app.py
- [X] T023 [US1] Implement POST /analyze endpoint in app.py (calls sentiment.analyze, returns AnalyzeResponse, handles InputError/BackendError)
- [X] T024 [US1] Implement POST /analyze error handler in app.py (InputError→422, BackendError→mapped status, others→500)
- [X] T025 [US1] Implement analyze subcommand in cli.py (argparse, calls sentiment.analyze, outputs text/JSON format)
- [X] T026 [US1] Implement error handling in cli.py analyze (exit codes: 0=success, 1=error, 3=config, 4=input, 5=backend)

**Checkpoint**: User Story 1 fully functional and testable independently

---

## Phase 4: User Story 2 — Batch Analysis and Accuracy Evaluation (Priority: P2)

**Goal**: Classify multiple texts in one request; measure model accuracy on labeled dataset

**Independent Test**: (a) `POST /analyze/batch` with `{"texts": ["Good.", "Bad."]}` returns 200 with count and results array. (b) `python cli.py fetch-data` downloads dataset. (c) `python cli.py evaluate` reports overall and per-source metrics (accuracy 92.80%, per-source breakdown). (d) Oversized batch (>64) and empty batch return 422.

### Tests for User Story 2 ⚠️

- [X] T027 [P] [US2] Contract test: batch validation (0 items, 64 items, 65 items) in tests/test_sentiment.py
- [X] T028 [P] [US2] Contract test: batch deduplication (duplicate texts use cache) in tests/test_sentiment.py
- [X] T029 [US2] Integration test: POST /analyze/batch happy path in tests/test_api.py
- [X] T030 [P] [US2] Integration test: POST /analyze/batch with invalid batch (0, >64 items) returns 422 in tests/test_api.py
- [X] T031 [P] [US2] Unit test: dataset loading (find_labelled_files, load_dataset with filters) in tests/test_sentiment.py

### Implementation for User Story 2

- [X] T032 [P] [US2] Implement BatchRequest Pydantic model in app.py (texts: list[str] with 1–64 constraints)
- [X] T033 [US2] Implement batch_analyze() function in sentiment.py (calls _cached_classify for each text, deduplicates, collects results)
- [X] T034 [US2] Implement POST /analyze/batch endpoint in app.py (calls sentiment.batch_analyze, returns count + results array)
- [X] T035 [US2] Implement fetch-data subcommand in cli.py (downloads UCI dataset zip, extracts, saves to data_dir, --force flag)
- [X] T036 [US2] Implement evaluate subcommand in cli.py (loads dataset, runs batch_analyze, computes accuracy/precision/recall/F1/confusion, reports per-source and overall)
- [X] T037 [US2] Implement _metrics() function in cli.py (computes TP/FP/FN/TN, accuracy, precision, recall, F1)
- [X] T038 [US2] Implement _confusion_matrix() function in cli.py (prints readable 2x2 matrix: TP, FP, FN, TN)

**Checkpoint**: User Stories 1 and 2 both work independently

---

## Phase 5: User Story 3 — Pluggable Backends (Priority: P3)

**Goal**: Switch inference backend via MODEL_BACKEND env var; same model/revision on both backends

**Independent Test**: (a) `MODEL_BACKEND=local` starts without credentials, `/health` <50ms, model loads on first `/analyze` request. (b) `MODEL_BACKEND=hf HF_TOKEN=$TOKEN` routes requests to Hugging Face, same results as local. (c) `MODEL_BACKEND=hf` without HF_TOKEN fails at import with ConfigError, exit code 3. (d) `/health` never loads model.

### Tests for User Story 3 ⚠️

- [X] T039 [P] [US3] Contract test: backend dispatch (local vs. hf) in tests/test_sentiment.py
- [X] T040 [US3] Contract test: config error (missing HF_TOKEN when MODEL_BACKEND=hf) in tests/test_sentiment.py
- [X] T041 [P] [US3] Integration test: GET /health happy path, no model load in tests/test_api.py
- [X] T042 [P] [US3] Integration test: backend selection in /health response in tests/test_api.py

### Implementation for User Story 3

- [X] T043 [US3] Implement _BACKENDS dict in sentiment.py ({"local": _run_local, "hf": _run_hf}) for dispatch
- [X] T043b [US3] Add MODEL_BACKEND env var validation in sentiment.py (ConfigError if not "local" or "hf")
- [X] T044 [US3] Add HF_TOKEN presence check in sentiment.py (ConfigError at import if MODEL_BACKEND=hf and HF_TOKEN missing)
- [X] T045 [US3] Implement GET /health endpoint in app.py (returns status, backend, model, revision, hf_token_present: bool, cache stats; never loads model)
- [X] T046 [US3] Implement GET / root endpoint in app.py (service banner with description and link to /docs)
- [X] T047 [US3] Implement rate-limit middleware in app.py (per-client-IP, sliding-window, only on /analyze and /analyze/batch)

**Checkpoint**: All user stories independently functional

---

## Phase 6: Bonus A — Docker

**Goal**: Containerize service with model weights baked in at build time

**Independent Test**: `docker build -t sentiment-api . && docker run -p 8000:8000 sentiment-api && curl -s http://localhost:8000/health | grep -q '"backend"' && curl -s -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d '{"text":"Great"}' | grep -q '"positive"'` returns 0.

- [X] T048 [US3] Create Dockerfile: base image python:3.11-slim, install deps with CPU torch index, bake in model via `python -c "from transformers import pipeline; ..."`
- [X] T049 [US3] Configure Dockerfile: runs app.py at port 8000, non-root user `app`, Python healthcheck using urllib (no curl/wget in slim)
- [X] T050 [US3] Verify Docker build in CI (.github/workflows/ci.yml): build image, smoke test GET /health, POST /analyze response contains "positive"

**Checkpoint**: Docker containerization complete

---

## Phase 7: Bonus B — Caching & Rate Limiting

**Goal**: Implement in-process LRU cache and sliding-window rate limiting

**Independent Test**: (a) First `/analyze` for text X takes ~2s (cold model). Second call takes <100ms (cache hit). Third call from same client over rate limit returns 429 with `Retry-After`. (b) `/health` and `/docs` bypass rate limit.

- [X] T051 [P] [US2] [Bonus] Unit test: cache hits/misses/size tracking in tests/test_sentiment.py
- [X] T052 [P] [US2] [Bonus] Unit test: rate limiter sliding-window per IP in tests/test_sentiment.py
- [X] T053 [US2] [Bonus] Integration test: rate-limit rejection on POST /analyze returns 429 with Retry-After in tests/test_api.py
- [X] T054 [US2] [Bonus] Integration test: /health and /docs bypass rate limit in tests/test_api.py

**Checkpoint**: Bonuses implemented and tested

---

## Phase 8: Error Handling & API Documentation

**Purpose**: Uniform error responses, interactive documentation, exit codes

- [X] T055 [P] Implement error handler middleware in app.py for uniform JSON error shape: `{"error": "<code>", "message": "<text>"}`
- [X] T056 [P] Implement HTTP status code mapping in app.py for all error codes (422, 429, 500, 502, 503, 504)
- [X] T057 Implement FastAPI Swagger UI integration in app.py (GET /docs, GET /redoc auto-generated)
- [X] T058 [P] Implement CLI exit code mapping in cli.py (0=success, 1=unexpected error, 2=argparse, 3=config, 4=input, 5=backend)
- [X] T059 Document all error codes in README.md with HTTP status, when they occur, and no-secrets policy

---

## Phase 9: Documentation & README

**Purpose**: Complete API/CLI reference, quick start, design notes

- [X] T060 Write README.md: quick start, dataset link, model & backends section, API reference table, CLI reference, results/accuracy, Docker bonus, caching/rate limiting bonus
- [X] T061 Write .specify/memory/constitution.md: 7 core principles, additional constraints, governance
- [X] T062 Write error code table in README.md with HTTP status, `error` code, `message`, and when each occurs
- [X] T063 Write environment variable table in README.md (MODEL_BACKEND, HF_TOKEN, HF_MODEL_ID, MODEL_REVISION, CACHE_SIZE, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW, HF_TIMEOUT)
- [X] T064 Document measured accuracy in README.md: 92.80% overall (2026-07-29), per-source breakdown (amazon 92.50%, imdb 93.50%, yelp 92.40%)

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Final integration, testing, deployment verification

- [X] T065 [P] Run full test suite: `python -m unittest discover -s tests -t .` (34 tests, all pass, <1s)
- [X] T066 Verify all tests run offline (no real model download, no network calls, all backends mocked)
- [X] T067 Verify CLI evaluate command end-to-end: fetch-data, evaluate, check reported accuracy is 92.80%
- [X] T068 [P] Verify Docker build succeeds and smoke test passes (health + analyze endpoint)
- [X] T069 [P] Verify GitHub Actions CI passes (test suite + Docker build & smoke test) on push to main
- [X] T070 Test configuration error handling: bad MODEL_BACKEND, missing HF_TOKEN (both exit at import, code 3)
- [X] T071 Test offline-safe default: MODEL_BACKEND=local, no HF_TOKEN, /health succeeds, model lazy-loads on first /analyze
- [X] T072 Verify rate limiter correctly rejects >30 requests per 60 seconds, exempts /health and /docs
- [X] T073 Verify cache deduplication: same text analyzed twice hits cache, cached=true on second result
- [X] T074 [P] Security check: HF_TOKEN never logged, never in error messages, never in /health response (only boolean presence flag)

---

## Notes

This tasks.md is written retroactively after implementation to document what was built, for traceability and reproducibility. All tasks are marked complete [X]. 

The implementation followed the constitution principles exactly: real model + real data (92.80% measured accuracy), secrets via env vars only (HF_TOKEN never hardcoded or logged), dual interface (API + CLI sharing sentiment.py core), fail fast (ConfigError at import), offline-safe default (local backend, no credentials), English-only deliverable, and measured not guessed (pinned revisions, reproducible evaluate run).

All 34 unit and integration tests pass fully offline in <1 second. The CI pipeline (GitHub Actions) verifies tests + Docker build + smoke test on every push. The service is production-ready for single-process deployment; multi-process scaling would add Redis for shared cache/rate-limiter (same key schema, no restructuring).

The flat project structure (no src/, no premature abstraction) keeps the codebase readable and maintainable. Complexity is justified by real requirements (two backends, caching, rate limiting, CLI + API) and tracked against the constitution.
