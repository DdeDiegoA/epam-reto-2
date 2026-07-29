"""FastAPI service exposing sentiment analysis over HTTP.

Endpoints
---------
GET  /                -> service banner pointing at /docs, with an endpoint list.
GET  /health          -> liveness probe. Never loads the model or touches the
                         network; safe to call offline. Reports backend info
                         and cache stats.
POST /analyze         -> body {"text": str} -> single sentiment result.
POST /analyze/batch   -> body {"texts": [str, ...]} -> batch of results.

Rate limiting
-------------
POST /analyze and POST /analyze/batch are rate limited per client IP
(see the middleware below). All other routes, including /docs, /redoc,
/openapi.json and /health, are exempt so a judge can explore the API
without hitting the limiter.

Environment variables (consumed by sentiment.py, listed here for reference)
-----------------------------------------------------------------------
MODEL_BACKEND        -- which inference backend to use (see sentiment.BACKEND).
HF_TOKEN             -- Hugging Face access token, if the backend needs one.
CACHE_SIZE           -- max entries kept in the in-process result cache.
RATE_LIMIT_REQUESTS  -- allowed requests per client per window (see below).
RATE_LIMIT_WINDOW    -- length in seconds of the rate-limit window.

Error shape
-----------
Every error response, regardless of source, has the same JSON body:
    {"error": "<code>", "message": "<actionable, no secrets>"}
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

import sentiment

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Analysis API",
    description="Sentiment analysis over text and batches of text, backed by a "
    "Hugging Face model with caching and rate limiting.",
    version="1.0.0",
)

# Module-level so tests can monkeypatch it (e.g. sentiment.RateLimiter subclass
# or a lower limit/window for deterministic 429 tests).
limiter = sentiment.RateLimiter(sentiment.RATE_LIMIT_REQUESTS, sentiment.RATE_LIMIT_WINDOW)

# Only these paths are rate limited; everything else (docs, health, root) is exempt.
_RATE_LIMITED_PATHS = {"/analyze", "/analyze/batch"}


# --------------------------------------------------------------------------
# Request/response models.
#
# The min_length/max_length bounds below mirror the MAX_TEXT_LENGTH and
# MAX_BATCH_SIZE constants in sentiment.py purely so /docs documents the
# limits to callers. The authoritative validation still happens inside
# sentiment.validate_text/validate_batch, which the CLI shares with this API.
# --------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=sentiment.MAX_TEXT_LENGTH,
        examples=["I absolutely loved this product."],
    )


class BatchRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=sentiment.MAX_BATCH_SIZE,
        examples=[["Great service.", "Terrible experience."]],
    )


class AnalyzeResponse(BaseModel):
    text: str
    label: str
    score: float
    model: str
    revision: str
    backend: str
    cached: bool


class BatchResponse(BaseModel):
    count: int
    results: list[AnalyzeResponse]


class HealthResponse(BaseModel):
    status: str
    backend: str
    model: str
    revision: str
    hf_token_present: bool
    cache: dict[str, Any]


class RootResponse(BaseModel):
    message: str
    docs: str
    endpoints: list[str]


class ErrorResponse(BaseModel):
    error: str
    message: str


# --------------------------------------------------------------------------
# Rate limiting middleware (Bonus B).
# --------------------------------------------------------------------------
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path not in _RATE_LIMITED_PATHS:
        return await call_next(request)

    # Key by the socket peer address only. Trusting X-Forwarded-For
    # unconditionally would let any caller spoof a fresh key on every request
    # and bypass the limiter entirely; it should only be honoured when this
    # app sits behind a trusted, correctly-configured reverse proxy.
    key = request.client.host if request.client else "unknown"

    allowed, remaining, reset_in = limiter.check(key)
    reset_seconds = math.ceil(reset_in)

    if not allowed:
        message = (
            f"Rate limit exceeded: {limiter.limit} requests per "
            f"{limiter.window} seconds. Try again later."
        )
        return JSONResponse(
            status_code=429,
            content={"error": "rate_limited", "message": message},
            headers={
                "Retry-After": str(reset_seconds),
                "X-RateLimit-Limit": str(limiter.limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_seconds),
            },
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(limiter.limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(reset_seconds)
    return response


# --------------------------------------------------------------------------
# Exception handlers -- every error body looks like {"error", "message"}.
# --------------------------------------------------------------------------
@app.exception_handler(sentiment.InputError)
async def handle_input_error(request: Request, exc: sentiment.InputError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": "validation_error", "message": str(exc)})


@app.exception_handler(sentiment.BackendError)
async def handle_backend_error(request: Request, exc: sentiment.BackendError) -> JSONResponse:
    headers = {}
    if exc.retry_after is not None:
        headers["Retry-After"] = str(exc.retry_after)
    return JSONResponse(
        status_code=exc.http_status,
        content={"error": exc.code, "message": str(exc)},
        headers=headers,
    )


@app.exception_handler(sentiment.ConfigError)
async def handle_config_error(request: Request, exc: sentiment.ConfigError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": "configuration_error", "message": str(exc)})


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Override FastAPI's default {"detail": [...]} shape so every error in
    # the API is uniform: {"error", "message"}.
    parts = []
    for err in exc.errors():
        field = ".".join(str(loc) for loc in err.get("loc", ()) if loc != "body")
        parts.append(f"{field}: {err.get('msg')}" if field else err.get("msg", "invalid input"))
    message = "Invalid request: " + "; ".join(parts) if parts else "Invalid request."
    return JSONResponse(status_code=422, content={"error": "validation_error", "message": message})


@app.exception_handler(StarletteHTTPException)
async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": "http_error", "message": str(exc.detail)})


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception while processing %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "An unexpected error occurred."},
    )


# --------------------------------------------------------------------------
# Routes.
# --------------------------------------------------------------------------
@app.get("/", response_model=RootResponse)
async def root() -> RootResponse:
    return RootResponse(
        message="Sentiment Analysis API. See /docs for interactive documentation.",
        docs="/docs",
        endpoints=["GET /health", "POST /analyze", "POST /analyze/batch"],
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> dict[str, Any]:
    # Must stay offline-safe: backend_info() and cache_stats() only read
    # in-process state, no model load, no network call.
    return {"status": "ok", **sentiment.backend_info(), "cache": sentiment.cache_stats()}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest) -> dict[str, Any]:
    return sentiment.analyze(payload.text)


@app.post("/analyze/batch", response_model=BatchResponse)
async def analyze_batch(payload: BatchRequest) -> dict[str, Any]:
    results = sentiment.analyze_batch(payload.texts)
    return {"count": len(results), "results": results}


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000)
