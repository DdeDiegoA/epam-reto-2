# Feature Specification: Sentiment Analysis Service

**Feature Branch**: `001-sentiment-analysis-service`

**Created**: 2026-07-29

**Status**: Draft

**Input**: Sentiment analysis API and CLI with local and HF backend support

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Analyze Sentiment via HTTP API (Priority: P1)

As a developer integrating sentiment analysis into their application, I need to send text to an HTTP endpoint and receive sentiment classification, so I can programmatically determine if user feedback or content is positive or negative.

**Why this priority**: This is the core value proposition—exposing sentiment analysis over a standard REST interface enables integration into web apps, mobile backends, and data pipelines. Without this, the service has no customer-facing value.

**Independent Test**: Can be fully tested by calling POST /analyze with a text sample and verifying the response contains label (positive/negative), score, and model info.

**Acceptance Scenarios**:

1. **Given** the API is running, **When** I POST `{"text": "I loved this product"}` to /analyze, **Then** I receive a 200 response with `label: "positive"`, `score: [0.0-1.0]`, and model metadata.
2. **Given** the API is running, **When** I POST `{"text": "This is terrible"}` to /analyze, **Then** I receive a 200 response with `label: "negative"`.
3. **Given** I send an empty or null text, **When** I POST to /analyze, **Then** I receive a 400 error with actionable error message.
4. **Given** I am rate-limited, **When** I exceed the request quota, **Then** I receive a 429 response without touching the model.

---

### User Story 2 - Batch Sentiment Analysis (Priority: P2)

As a data analyst, I need to classify hundreds of text samples in a single request, so I can avoid the latency overhead of making individual HTTP calls.

**Why this priority**: Batch processing is essential for analytical workloads and reduces network round-trips. P2 because single-item analysis (P1) covers the MVP, but batch significantly improves usability for bulk operations.

**Independent Test**: Can be fully tested by calling POST /analyze/batch with 10+ text samples and verifying all results are returned in a single response.

**Acceptance Scenarios**:

1. **Given** I have 5 texts to classify, **When** I POST `{"texts": ["text1", "text2", ...]}` to /analyze/batch, **Then** I receive a 200 response with a results array matching the input length.
2. **Given** one text in a batch is empty, **When** I POST to /analyze/batch, **Then** the batch processes successfully and the empty text returns an error within that result.
3. **Given** I submit a batch of 1000+ texts, **When** processing completes, **Then** all results are returned correctly (no silent truncation).

---

### User Story 3 - CLI-based Sentiment Analysis (Priority: P2)

As a data scientist or DevOps engineer, I need to analyze text from the command line without running a server, so I can integrate sentiment analysis into shell scripts and batch jobs.

**Why this priority**: The CLI provides an alternative interface for non-HTTP consumers (scripts, automation). P2 because it duplicates P1's core logic; users can always use curl to call the API instead, but CLI ergonomics matter for offline/local workflows.

**Independent Test**: Can be fully tested by running `cli.py analyze "test text"` and verifying sentiment output and model info.

**Acceptance Scenarios**:

1. **Given** I run `python cli.py analyze "I love this"`, **When** execution completes, **Then** stdout contains label, score, and model info in a readable format.
2. **Given** I run the CLI without arguments, **When** execution completes, **Then** helpful usage/help text is printed.
3. **Given** I run CLI with invalid syntax, **When** execution fails, **Then** a clear error message and usage hint are shown.

---

### User Story 4 - Backend Selection & Configuration (Priority: P3)

As an infrastructure operator, I need to switch between local model inference and Hugging Face API without code changes, so I can scale inference to a managed service or run offline when needed.

**Why this priority**: Backend flexibility enables deployment to different environments (resource-constrained edge, scalable cloud). P3 because both backends deliver the same user-facing results; users can start with local and migrate without changing their integration code.

**Independent Test**: Can be fully tested by running the service with `MODEL_BACKEND=local` and `MODEL_BACKEND=hf`, verifying identical responses for the same input.

**Acceptance Scenarios**:

1. **Given** `MODEL_BACKEND=local`, **When** I send a request, **Then** inference happens locally with no network calls (except initial model download).
2. **Given** `MODEL_BACKEND=hf` and a valid `HF_TOKEN`, **When** I send a request, **Then** inference delegates to Hugging Face Inference API.
3. **Given** `MODEL_BACKEND=hf` and no `HF_TOKEN`, **When** the service starts, **Then** a clear error indicates the missing credential.
4. **Given** I switch backends mid-deployment, **When** I restart with a different backend, **Then** requests continue to work (same model, different execution).

---

### User Story 5 - Performance & Reliability (Priority: P3)

As a DevOps engineer, I need the service to cache results, rate-limit abusive clients, and report health status, so I can operate it reliably in production and prevent resource exhaustion.

**Why this priority**: Caching and rate limiting improve UX and resilience; health checks enable monitoring. P3 because they're operational concerns—the core sentiment analysis works without them, but they unlock production viability.

**Independent Test**: Can be fully tested by calling /health (returns 200 without model load), measuring response time for repeated identical requests (should be cached), and verifying rate-limit headers.

**Acceptance Scenarios**:

1. **Given** I call /health, **When** it responds, **Then** it returns 200 within 100ms without loading or downloading the model.
2. **Given** I send the same text twice, **When** the second request completes, **Then** the response indicates `cached: true` and latency is < 10ms.
3. **Given** I exceed the rate limit (e.g., 100 requests/min from one IP), **When** the limit is hit, **Then** subsequent requests return 429 with Retry-After header.
4. **Given** the service is running, **When** I call /health, **Then** the response includes backend type, model info, and cache statistics.

---

### Edge Cases

- What happens when the input text is extremely long (e.g., 1 MB)? → Should gracefully reject with 400 and actionable message, not OOM.
- How does the system handle concurrent requests to both /analyze and /analyze/batch? → Should process concurrently without race conditions on cache or rate limit state.
- What happens if the Hugging Face API is down (HF backend)? → Should return 503 with retry guidance; local backend remains unaffected.
- What happens if the model fails to load on startup (local backend)? → Service should boot and serve /health; model loads lazily on first inference request.
- How does caching handle identical texts across different model revisions? → Cache is per model revision; switching `MODEL_REVISION` invalidates cached results for that text.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST expose a POST /analyze endpoint accepting `{"text": str}` and returning sentiment label (positive/negative), confidence score [0.0-1.0], model name, revision, and cached flag.
- **FR-002**: System MUST expose a POST /analyze/batch endpoint accepting `{"texts": [str, ...]}` and returning an array of sentiment results, one per input.
- **FR-003**: System MUST support two inference backends selectable via `MODEL_BACKEND` env var: "local" (transformers.pipeline) and "hf" (Hugging Face Inference API).
- **FR-004**: System MUST cache sentiment results in process, keyed by (text, model_revision), up to a configurable limit (default 1000 entries, 0 = disabled).
- **FR-005**: System MUST implement a sliding-window rate limiter per client IP (default 100 requests/60 seconds; configurable via env vars).
- **FR-006**: System MUST expose a GET /health endpoint that returns 200 with backend info and cache stats within 100ms, never loading the model.
- **FR-007**: System MUST expose a GET / endpoint with service banner, endpoint list, and link to interactive docs (/docs).
- **FR-008**: System MUST implement a CLI interface exposing commands: `analyze <text>` and `batch <file>` (reads texts from file, one per line).
- **FR-009**: System MUST return errors in a standard JSON shape: `{"error": "<code>", "message": "<actionable description, no secrets>"}`.
- **FR-010**: System MUST accept and respect env vars: `MODEL_BACKEND`, `HF_TOKEN`, `HF_MODEL_ID`, `MODEL_REVISION`, `CACHE_SIZE`, `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW`, `HF_TIMEOUT`.
- **FR-011**: System MUST handle empty, null, or invalid UTF-8 text gracefully, returning 400 with clear error message.
- **FR-012**: System MUST support a pinned model revision via `MODEL_REVISION` env var for reproducible results.

### Key Entities

- **Sentiment Result**: Represents the output of sentiment classification. Attributes: text (input), label (positive/negative), score (float 0-1), model (repo ID), revision (commit hash), backend (local/hf), cached (bool).
- **Rate Limit State**: Tracks requests per IP within a time window. Attributes: client IP, request count, window start time, limit threshold.
- **Cache Entry**: Stores computed sentiment results. Key: (text, model_revision); Value: Sentiment Result; TTL: none (persistent within process lifetime or up to CACHE_SIZE).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: API responds to /analyze in under 200ms for cached requests and under 1 second for cold requests on a CPU machine.
- **SC-002**: Batch endpoint processes 100 texts in a single request and returns all results within 5 seconds.
- **SC-003**: Cache hit rate is ≥ 70% for typical workloads (repeat texts are common).
- **SC-004**: Rate limiter correctly rejects requests exceeding the limit; no false positives or false negatives.
- **SC-005**: /health endpoint responds in < 100ms and works offline (no model load, no network calls).
- **SC-006**: Sentiment classification achieves ≥ 90% accuracy on the UCI "Sentiment Labelled Sentences" dataset (3,000 sentences: 1,500 positive, 1,500 negative).
- **SC-007**: CLI returns sentiment analysis results within 2 seconds on a CPU machine (local backend).
- **SC-008**: Service handles 50+ concurrent requests without deadlocks or cache corruption.
- **SC-009**: Error messages are actionable and do not leak secrets (tokens, file paths, model weights).

## Assumptions

- **Users have access to a Python 3.8+ runtime** for running the service locally; Docker is an optional convenience.
- **Network connectivity is available for Hugging Face model download (first run) and HF backend inference**; the local backend works offline after the model is cached.
- **Input texts are in English**; multilingual support and sentiment granularity (e.g., mixed sentiment) are out of scope for v1.
- **Sentiment is binary (positive/negative)**; nuanced classifications (neutral, sarcasm) are not supported.
- **Request payloads are under 10 MB**; extremely large payloads (e.g., > 100 MB) are out of scope.
- **Batch requests contain up to 10,000 texts per request**; larger batches are rate-limited or split by the client.
- **Model revision is stable** (pinned via commit hash); model retraining and automatic updates are out of scope.
- **Rate limiting uses client IP from the HTTP request** (X-Forwarded-For behind proxies is not handled in v1).
- **Cache is ephemeral** and does not persist across service restarts; persistent caching is a future enhancement.
