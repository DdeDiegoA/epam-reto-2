# Design Decisions

## Real Model, Real Data

**Decision:** Use a real Hugging Face model (distilbert-sst2) pinned at revision SHA, evaluated against a public labeled dataset (UCI Sentiment Labelled Sentences).

**Why:** Judges need measurable, reproducible accuracy. Mocked inference or synthetic numbers are not credible. The pinned revision ensures deterministic behavior and allows cross-backend comparison (local vs. HF API return identical scores).

**Trade-off:** Model weights (~268 MB) must be downloaded or baked into Docker. Cold-start latency on first request with local backend. Mitigated by caching and model baking in Dockerfile.

## Dual Backends, Environment-Variable Switch

**Decision:** Support two backends (local transformers, Hugging Face Inference API) selectable by `MODEL_BACKEND` env var, both using the same pinned model.

**Why:** 
- Local backend (default) lets judges run without credentials, offline.
- HF backend demonstrates secure credential handling (env var only, never hardcoded/logged).
- Same model on both paths means no re-implementation risk; differences are network calls only.

**Trade-off:** HF_TOKEN is required if backend=hf, causing hard failure at import (fail-fast). Users must set it or stick with local. Documented in README.

## Secrets via Environment Only

**Decision:** `HF_TOKEN` read exclusively from `os.getenv()`, never hardcoded, never logged, never included in error messages beyond a presence boolean.

**Why:** EPAM requirement. Best practice for credential handling. `.gitignore` excludes `.env*` from commit 1, guaranteeing no token leakage.

**Trade-off:** Token is not validated at startup (would require a test API call). Bad tokens fail on first /analyze request, not earlier. Acceptable—most users will test immediately and catch it.

## Shared Service Layer, Thin Interfaces

**Decision:** All logic lives in `sentiment.py`. `app.py` (FastAPI) and `cli.py` (CLI) are thin wrappers that delegate validation, backend dispatch, caching, and error semantics to `sentiment.py`.

**Why:** No code duplication. Same behavior across HTTP/CLI. Single source of truth for error codes, label normalization, cache hits. Easier to test (test sentiment.py, not the HTTP layer).

**Trade-off:** Coupling between interfaces and sentiment.py. Acceptable for a small service (3 modules, ~1100 LOC). No benefit to an ABC or registry for 2 backends.

## LRU Cache, No TTL

**Decision:** Use `functools.lru_cache` with no TTL. Cache is keyed on (backend, model_id, revision, text).

**Why:** Pinned model revision + fixed text = pure function. Output cannot go stale; same input always yields the same label/score. TTL is unnecessary complexity and adds latency on re-fetches.

**Trade-off:** Stale data is not possible (design property). Memory grows unbounded in long-running processes with diverse inputs. Mitigated by `CACHE_SIZE` env var (default 1024 entries, ~0.5 MB in practice).

**When to upgrade:** If the service runs for weeks with high throughput, switch to Redis for shared, memory-bounded cache (no local-state restructuring needed; same LRU keying scheme).

## Score Bounds Validation Before Caching

**Decision:** `_cached_classify()` validates that the backend's score is in the range [0.0, 1.0] before returning, raising `BackendError` code `unexpected_upstream_response` if out-of-range.

**Why:** Prevents a malformed upstream response (e.g., a broken model or corrupted API response) from poisoning the cache permanently. With no TTL, a bad value would be cached forever. Validation upstream of the cache ensures the cache only holds valid data.

**Trade-off:** Adds a numeric comparison per non-cached request. Negligible overhead (~nanoseconds). The cost is more than paid back by avoiding cache corruption.

## Sliding-Window Rate Limiter

**Decision:** Sliding-window log (deque per IP, thread-safe) instead of fixed-window counters.

**Why:** Fixed-window allows a client to fire 2× the limit across a boundary (59 requests at second 59, then 60 requests at second 0 = 119 in 2 seconds). Sliding window prevents this.

**Algorithm:** Per-IP deque of request timestamps. On each request, drop timestamps older than the window, then check if the remaining count exceeds the limit. ~25 lines, no external dependencies.

**Trade-off:** O(n) per request where n = limit (30 by default; ~30 timestamp lookups). Fixed-window is O(1) but less fair. For this scale, acceptable.

**When to upgrade:** If throughput demands >1000 req/s, switch to Redis (per-process locks insufficient). Same API (RateLimiter.check); no code changes needed.

## Uniform Error Response Shape

**Decision:** All errors return `{"error": "<code>", "message": "<actionable text>"}` with no secrets.

**Why:** Predictable client handling. Machines parse the code; humans read the message. Avoids accidentally leaking tokens, file paths, or API URLs.

**Examples:**
- `validation_error` — input bounds violated
- `rate_limited` — quota exceeded
- `configuration_error` — bad env var
- `backend_unavailable` — model download failed offline
- `backend_timeout` — HF API slow

**Trade-off:** Less detail in error messages (no exception stack). Acceptable—judges/users can check logs for more info.

## Flat Project Layout

**Decision:** No `src/`, no `models/`, `services/`, `schemas/` subdirectories. Three modules: `sentiment.py`, `app.py`, `cli.py`.

**Why:** Only 3 user-facing modules, <1100 LOC total. Adding abstractions (backend ABC, config.py, schemas.py) adds boilerplate without benefit. Constitution forbids premature abstraction.

**Trade-off:** Can grow confusing if 10+ modules. For this scope, simplicity wins.

## Model Weights Baked into Docker

**Decision:** Dockerfile runs `python -c "from transformers import pipeline; pipeline(model=..., revision=...)"` at build time, populating `HF_HOME=/opt/hf` before copying source.

**Why:** Judges get a container that runs instantly without downloading ~260 MB on first start. Offline-friendly.

**Trade-off:** Docker image is ~1.5 GB (python:3.11-slim + torch + transformers + model weights). Builds take ~5–10 min. Mitigated by GitHub Actions layer caching (re-builds <30s).

## Measured Accuracy, Not Guessed

**Decision:** Run `python cli.py evaluate` once on all 3000 sentences with the real deployed code. Report the exact numbers (92.80% overall) without rounding or cherry-picking per source.

**Why:** Credibility. The model's 3 per-source breakdowns (amazon 92.50%, imdb 93.50%, yelp 92.40%) show domain shift honestly. Hiding that or rounding up to "93%" is misleading.

**Trade-off:** Exposes any implementation bugs or model weaknesses. Acceptable—we found none.

## English-Only Deliverable

**Decision:** All code, docstrings, error messages, README, specs, commits in English.

**Why:** International judging. Repo is the sole artifact judges see. Code is English by convention; no reason to break it.

**Trade-off:** Internal working notes/chat with user in Spanish; no code output does.

## stdlib unittest, No Pytest

**Decision:** Use `unittest.discover` for tests. No pytest, no fixtures, no plugins.

**Why:** 34 tests, fully offline. Stdlib is sufficient. pytest would add unjustified dependency just to run 5 test files.

**Trade-off:** More verbose assertions, no parametrized tests. Acceptable for this scope.
