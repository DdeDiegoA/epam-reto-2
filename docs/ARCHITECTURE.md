# Architecture

**Service Layer:** `sentiment.py` — the only module aware of backends, caching, validation, rate limiting.

**HTTP API:** `app.py` (FastAPI) — stateless routes, error handlers, rate-limit middleware. No business logic redefinition; delegates to sentiment.py.

**CLI:** `cli.py` (argparse) — three subcommands: fetch-data (dataset download), analyze (single text), evaluate (batch + metrics). Shares validation/analysis with HTTP layer.

**Testing:** `tests/test_sentiment.py` (unit), `tests/test_api.py` (integration via FastAPI TestClient). 34 tests, fully offline, <1 second.

## Core Abstractions

**sentiment.py exports:**
- `analyze(text)` → single result
- `analyze_batch(texts)` → array of results
- `validate_text()`, `validate_batch()` — shared validation
- `RateLimiter` — sliding-window per-IP
- `ConfigError`, `InputError`, `BackendError` — exceptions with codes

**Backends:**
- `local` (default): transformers.pipeline + torch, lazy-loaded
- `hf`: huggingface_hub.InferenceClient (requires HF_TOKEN)

Both backends return identical label/score via same pinned model.

**Cache:** LRU (functools.lru_cache) keyed on (backend, model_id, revision, text). No TTL—pure function property.

**Rate Limiter:** Sliding-window log (deque per IP, thread-safe). Default: 30 req/60s.

## Data Flow

```
User Request (HTTP/CLI)
  ↓
app.py / cli.py (validation, parsing)
  ↓
sentiment.py:analyze()
  ├→ validate_text()
  ├→ _cached_classify()
  │   ├→ _classify() [first call]
  │   │   ├→ _run_local() (transformers) OR
  │   │   └→ _run_hf() (Hugging Face API)
  │   └→ [cache hit]
  ├→ normalize label
  └→ return {"text", "label", "score", "model", "revision", "backend", "cached"}

Rate Limit Middleware (app.py only)
  ↓
RateLimiter.check(client_ip)
  ├→ 429 if exceeded
  └→ proceed + add response headers
```

## File Sizes & Responsibilities

- **sentiment.py** (~440 lines): core logic
- **app.py** (~249 lines): FastAPI routes, error handlers
- **cli.py** (~411 lines): argparse, dataset loader, metrics
- **tests/** (~460 lines total): 34 tests, mocked backends
- **Dockerfile** (~45 lines): baked model, non-root user
- **.github/workflows/ci.yml** (~80 lines): test + docker build + smoke test

## Deployment

**Local:** `python cli.py analyze "text"` or `uvicorn app:app`

**Docker:** `docker build . && docker run -p 8000:8000 sentiment-api`
- Model weights (~268 MB) baked in at build time
- No GPU required (CPU-only torch)
- Non-root `app` user

**Environment Variables:**
- `MODEL_BACKEND` (local|hf, default: local)
- `HF_TOKEN` (required if backend=hf)
- `HF_MODEL_ID`, `MODEL_REVISION` (override defaults)
- `CACHE_SIZE` (default 1024, 0=disabled)
- `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW` (default 30/60)
