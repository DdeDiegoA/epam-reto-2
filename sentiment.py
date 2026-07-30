"""Sentiment-analysis service layer.

This is the only module in the project aware that a model exists. Both the
FastAPI app (``app.py``) and the CLI (``cli.py``) call into this module and
never touch a model, tokenizer, or HTTP client directly.

Two backends are supported, selected by the ``MODEL_BACKEND`` env var:

- ``local`` (default): runs the model on this machine via a
  ``transformers.pipeline``. The pipeline is built lazily on first use so the
  process can boot and serve health checks even fully offline.
- ``hf``: delegates inference to the Hugging Face Inference API via
  ``huggingface_hub.InferenceClient``. Requires ``HF_TOKEN`` to be set.

Environment variables:

- ``HF_MODEL_ID``: model repo id to load/query.
- ``MODEL_REVISION``: pinned model revision (commit hash), for reproducible
  results.
- ``MODEL_BACKEND``: ``"local"`` or ``"hf"``.
- ``CACHE_SIZE``: max entries in the in-process result cache; ``0`` disables
  caching.
- ``RATE_LIMIT_REQUESTS`` / ``RATE_LIMIT_WINDOW``: sliding-window rate limit
  parameters, exposed for callers (e.g. app.py) to build a ``RateLimiter``.
- ``HF_TIMEOUT``: per-request timeout (seconds) for the ``hf`` backend.
- ``HF_TOKEN``: Hugging Face API token, required only when
  ``MODEL_BACKEND=hf``. Never logged or returned.
"""

from __future__ import annotations

import functools
import logging
import os
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised for invalid server configuration, detected at import time."""


class InputError(Exception):
    """Raised when caller-supplied text/batch input fails validation."""


class BackendError(Exception):
    """Raised when the model backend (local or hosted) cannot fulfil a request."""

    def __init__(
        self,
        code: str,
        message: str,
        http_status: int = 502,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.retry_after = retry_after


# --------------------------------------------------------------------------
# Configuration (read once at import; fail fast on misconfiguration)
# --------------------------------------------------------------------------

MODEL_ID: str = os.getenv("HF_MODEL_ID", "distilbert/distilbert-base-uncased-finetuned-sst-2-english")
MODEL_REVISION: str = os.getenv("MODEL_REVISION", "714eb0fa89d2f80546fda750413ed43d93601a13")
BACKEND: str = os.getenv("MODEL_BACKEND", "local")
CACHE_SIZE: int = int(os.getenv("CACHE_SIZE", "1024"))
RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
HF_TIMEOUT: float = float(os.getenv("HF_TIMEOUT", "20.0"))
MAX_TEXT_LENGTH = 2000
MAX_BATCH_SIZE = 64

if BACKEND not in ("local", "hf"):
    raise ConfigError(f"Invalid MODEL_BACKEND={BACKEND!r}; must be 'local' or 'hf'")

if BACKEND == "hf" and not os.getenv("HF_TOKEN"):
    raise ConfigError(
        "MODEL_BACKEND=hf requires the HF_TOKEN environment variable to be set"
    )

logger.info("Sentiment backend selected: %s (model=%s)", BACKEND, MODEL_ID)


# --------------------------------------------------------------------------
# Label normalization
# --------------------------------------------------------------------------

_LABELS = {"POSITIVE": "positive", "NEGATIVE": "negative", "LABEL_0": "negative", "LABEL_1": "positive"}


def _normalize_label(raw_label: str) -> str:
    normalized = _LABELS.get(raw_label.upper())
    if normalized is None:
        raise BackendError(
            "unexpected_upstream_response",
            f"Unrecognized label {raw_label!r} returned by the model backend",
            502,
        )
    return normalized


# --------------------------------------------------------------------------
# Local backend
# --------------------------------------------------------------------------

_local_pipeline: Any = None
_local_pipeline_lock = threading.Lock()


def _run_local(texts: list[str]) -> list[tuple[str, float]]:
    global _local_pipeline
    if _local_pipeline is None:
        with _local_pipeline_lock:
            if _local_pipeline is None:  # re-check: another thread may have built it
                try:
                    import transformers
                except ImportError as exc:
                    raise BackendError(
                        "backend_unavailable",
                        "The 'transformers' package is not installed; "
                        "run pip install -r requirements.txt",
                        500,
                    ) from exc
                try:
                    _local_pipeline = transformers.pipeline(
                        "text-classification",
                        model=MODEL_ID,
                        revision=MODEL_REVISION,
                        device=-1,
                        truncation=True,
                        max_length=512,
                    )
                except Exception as exc:
                    raise BackendError(
                        "model_unavailable",
                        f"Could not load model {MODEL_ID}@{MODEL_REVISION} locally "
                        "(offline, or the download failed); set MODEL_BACKEND=hf to "
                        "use the hosted Hugging Face Inference API instead",
                        503,
                    ) from exc

    results = _local_pipeline(texts)
    return [(r["label"], float(r["score"])) for r in results]


# --------------------------------------------------------------------------
# Hugging Face hosted backend
# --------------------------------------------------------------------------


def _entry_label_score(entry: Any) -> tuple[str, float]:
    """Read (label, score) off a result entry that may be a dict or an object."""
    if isinstance(entry, dict):
        return entry["label"], float(entry["score"])
    return entry.label, float(entry.score)


def _map_hf_error(exc: Exception, http_error_cls: type[Exception]) -> tuple[str, int, int | None]:
    """Map an upstream huggingface_hub exception to (code, http_status, retry_after)."""
    status = None
    if isinstance(exc, http_error_cls):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    message = str(exc).lower()

    if status in (401, 403) or "unauthorized" in message or "forbidden" in message:
        return "backend_auth_error", 500, None
    if status == 429 or "rate limit" in message or "too many requests" in message:
        return "backend_rate_limited", 503, 5
    if status == 503 or "loading" in message:
        return "model_loading", 503, 10
    if status == 404 or "not found" in message:
        return "model_not_found", 500, None
    if isinstance(exc, TimeoutError) or "timeout" in message or "timed out" in message:
        return "backend_timeout", 504, None
    if isinstance(exc, ConnectionError) or "connection" in message or "dns" in message:
        return "backend_unreachable", 502, None
    return "backend_error", 502, None


def _hf_error_message(code: str, exc: Exception) -> str:
    messages = {
        "backend_auth_error": "Hugging Face rejected the request credentials; check HF_TOKEN.",
        "backend_rate_limited": "Hugging Face Inference API rate limit exceeded; retry shortly.",
        "model_loading": "The model is still loading on Hugging Face infrastructure; retry shortly.",
        "model_not_found": f"Model {MODEL_ID}@{MODEL_REVISION} was not found on Hugging Face.",
        "backend_timeout": "The request to the Hugging Face Inference API timed out.",
        "backend_unreachable": "Could not reach the Hugging Face Inference API (network/DNS issue).",
        "backend_error": "The Hugging Face Inference API returned an unexpected error.",
    }
    return messages.get(code, str(exc))


def _hf_classify_one(client: Any, text: str, http_error_cls: type[Exception]) -> tuple[str, float]:
    attempts_left = 2  # one retry, only for a cold-start "model_loading" error
    while True:
        try:
            result = client.text_classification(text)
        except Exception as exc:
            code, http_status, retry_after = _map_hf_error(exc, http_error_cls)
            attempts_left -= 1
            if code == "model_loading" and attempts_left > 0:
                time.sleep(3)
                continue
            raise BackendError(code, _hf_error_message(code, exc), http_status, retry_after) from exc
        top = max(result, key=lambda e: _entry_label_score(e)[1])
        return _entry_label_score(top)


def _run_hf(texts: list[str]) -> list[tuple[str, float]]:
    try:
        from huggingface_hub import InferenceClient
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError as exc:
        raise BackendError(
            "backend_unavailable",
            "The 'huggingface_hub' package is not installed; "
            "run pip install -r requirements.txt",
            500,
        ) from exc

    client = InferenceClient(
        model=MODEL_ID,
        token=os.environ["HF_TOKEN"],
        provider="hf-inference",
        timeout=HF_TIMEOUT,
    )
    return [_hf_classify_one(client, text, HfHubHTTPError) for text in texts]


_BACKENDS: dict[str, Callable[[list[str]], list[tuple[str, float]]]] = {
    "local": _run_local,
    "hf": _run_hf,
}


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

# ponytail: process-local lru_cache, not shared across workers. If this ever
# runs behind multiple worker processes, swap for Redis keyed the same way.
_CACHE_ENABLED = CACHE_SIZE > 0


@functools.lru_cache(maxsize=CACHE_SIZE or 1)
def _cached_classify(backend: str, model_id: str, revision: str, text: str) -> tuple[str, float]:
    # No TTL: a pinned model revision scoring a fixed input string is a pure
    # function of its arguments, so a cached result can never go stale.
    raw_label, score = _BACKENDS[backend]([text])[0]
    if not 0.0 <= score <= 1.0:
        # Checked before returning so a bad value never enters the lru_cache
        # (it would otherwise be cached forever -- no TTL, see above).
        raise BackendError(
            "unexpected_upstream_response",
            f"Backend returned out-of-range score {score!r} for label {raw_label!r}",
            502,
        )
    return _normalize_label(raw_label), score


def _classify(text: str) -> tuple[tuple[str, float], bool]:
    """Return ((label, score), was_cache_hit) for one text, honoring CACHE_SIZE."""
    if not _CACHE_ENABLED:
        raw_label, score = _BACKENDS[BACKEND]([text])[0]
        return (_normalize_label(raw_label), score), False

    hits_before = _cached_classify.cache_info().hits
    result = _cached_classify(BACKEND, MODEL_ID, MODEL_REVISION, text)
    cached = _cached_classify.cache_info().hits > hits_before
    return result, cached


def cache_stats() -> dict:
    if not _CACHE_ENABLED:
        return {"enabled": False, "hits": 0, "misses": 0, "size": 0, "max_size": 0}
    info = _cached_classify.cache_info()
    return {
        "enabled": True,
        "hits": info.hits,
        "misses": info.misses,
        "size": info.currsize,
        "max_size": info.maxsize,
    }


def clear_cache() -> None:
    if _CACHE_ENABLED:
        _cached_classify.cache_clear()


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def validate_text(text: str) -> str:
    if not isinstance(text, str):
        raise InputError(f"Expected a string, got {type(text).__name__}")
    stripped = text.strip()
    if not stripped:
        raise InputError("Text must not be empty")
    if len(stripped) > MAX_TEXT_LENGTH:
        raise InputError(f"Text length {len(stripped)} exceeds the {MAX_TEXT_LENGTH}-character limit")
    return stripped


def validate_batch(texts: list[str]) -> list[str]:
    if not isinstance(texts, list):
        raise InputError(f"Expected a list of strings, got {type(texts).__name__}")
    if not texts:
        raise InputError("Batch must not be empty")
    if len(texts) > MAX_BATCH_SIZE:
        raise InputError(f"Batch size {len(texts)} exceeds the {MAX_BATCH_SIZE}-item limit")
    return [validate_text(t) for t in texts]


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def analyze(text: str) -> dict:
    text = validate_text(text)
    (label, score), cached = _classify(text)
    return {
        "text": text,
        "label": label,
        "score": round(score, 4),
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "backend": BACKEND,
        "cached": cached,
        # Per-call metadata: generated here, never cached, so a cache hit
        # still gets its own fresh request_id/timestamp.
        "request_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def analyze_batch(texts: list[str]) -> dict:
    texts = validate_batch(texts)
    results = [analyze(text) for text in texts]
    cache_hits = sum(1 for r in results if r["cached"])
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    return {
        "results": results,
        "cache_hits": cache_hits,
        "cache_misses": len(results) - cache_hits,
        "avg_score": avg_score,
    }


def backend_info() -> dict:
    return {
        "backend": BACKEND,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "hf_token_present": bool(os.getenv("HF_TOKEN")),
    }


class RateLimiter:
    """Sliding-window rate limiter, keyed per caller (e.g. per client IP)."""

    # ponytail: one global lock for all keys, not per-key locks. Simpler and
    # plenty fast for a single-process API; shard the lock if contention
    # across many distinct keys ever shows up in profiling.
    def __init__(self, limit: int, window: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.limit = limit
        self.window = window
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int, float]:
        now = self._clock()
        with self._lock:
            timestamps = self._hits.get(key, deque())
            cutoff = now - self.window
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            if not timestamps:
                self._hits.pop(key, None)  # drop empty entries so the dict can't grow forever

            allowed = len(timestamps) < self.limit
            if allowed:
                timestamps.append(now)
                self._hits[key] = timestamps

            if not timestamps:
                return allowed, max(self.limit, 0), 0.0

            remaining = max(self.limit - len(timestamps), 0)
            reset_in = max(timestamps[0] + self.window - now, 0.0)
            return allowed, remaining, reset_in


# --------------------------------------------------------------------------
# Self-check (pure-logic paths only; no model/network calls)
# --------------------------------------------------------------------------


def _demo() -> None:
    assert validate_text("  hello  ") == "hello"
    for bad in ("", "   ", 123, "x" * (MAX_TEXT_LENGTH + 1)):
        try:
            validate_text(bad)  # type: ignore[arg-type]
            raise AssertionError(f"expected InputError for {bad!r}")
        except InputError:
            pass

    assert validate_batch(["a", " b "]) == ["a", "b"]
    for bad_batch in ([], "not a list", ["x"] * (MAX_BATCH_SIZE + 1)):
        try:
            validate_batch(bad_batch)  # type: ignore[arg-type]
            raise AssertionError(f"expected InputError for {bad_batch!r}")
        except InputError:
            pass

    assert _normalize_label("POSITIVE") == "positive"
    assert _normalize_label("label_1") == "positive"
    assert _normalize_label("label_0") == "negative"
    try:
        _normalize_label("SURPRISE")
        raise AssertionError("expected BackendError for an unmapped label")
    except BackendError as exc:
        assert exc.code == "unexpected_upstream_response"

    clock = [0.0]
    limiter = RateLimiter(limit=2, window=10, clock=lambda: clock[0])
    allowed, remaining, _ = limiter.check("ip")
    assert allowed and remaining == 1
    allowed, remaining, _ = limiter.check("ip")
    assert allowed and remaining == 0
    allowed, remaining, reset_in = limiter.check("ip")
    assert not allowed and remaining == 0 and reset_in > 0
    clock[0] = 11.0  # past the window: oldest hits expire
    allowed, remaining, _ = limiter.check("ip")
    assert allowed and remaining == 1

    print("sentiment.py self-check: OK")


if __name__ == "__main__":
    _demo()
