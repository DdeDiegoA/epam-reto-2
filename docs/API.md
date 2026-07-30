# HTTP API Reference

**Base URL:** `http://localhost:8000`

## Endpoints

### GET /

Service banner. Safe to call offline.

**Response:** 200 OK
```json
{
  "message": "Sentiment Analysis API. See /docs for interactive documentation.",
  "docs": "/docs",
  "endpoints": ["GET /health", "POST /analyze", "POST /analyze/batch"]
}
```

### GET /health

Liveness probe. Never loads the model or calls the network.

**Response:** 200 OK
```json
{
  "status": "ok",
  "backend": "local",
  "model": "distilbert-base-uncased-finetuned-sst-2-english",
  "revision": "714eb0fa89d2f80546fda750413ed43d93601a13",
  "hf_token_present": false,
  "cache": {
    "hits": 0,
    "misses": 0,
    "currsize": 0
  }
}
```

### POST /analyze

Classify a single text.

**Request:** 200–2000 characters
```json
{
  "text": "I absolutely loved this product!"
}
```

**Response:** 200 OK
```json
{
  "text": "I absolutely loved this product!",
  "label": "positive",
  "score": 0.9998,
  "model": "distilbert-base-uncased-finetuned-sst-2-english",
  "revision": "714eb0fa89d2f80546fda750413ed43d93601a13",
  "backend": "local",
  "cached": false,
  "request_id": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
  "timestamp": "2026-07-29T18:30:45.123456+00:00"
}
```

**Errors:**
- `422 validation_error` — empty, missing, or >2000 char text
- `429 rate_limited` — exceeded 30 req/60s per IP
- `500 configuration_error` — bad MODEL_BACKEND or missing HF_TOKEN
- `502 backend_error` — model/API failure
- `503 model_unavailable` — model download offline
- `504 backend_timeout` — HF API timeout

**Error Response:**
```json
{
  "error": "validation_error",
  "message": "Text must be 1–2000 characters."
}
```

### POST /analyze/batch

Classify 1–64 texts in one request.

**Request:**
```json
{
  "texts": [
    "Great service!",
    "Terrible experience.",
    "Just okay."
  ]
}
```

**Response:** 200 OK
```json
{
  "count": 3,
  "results": [
    {
      "text": "Great service!",
      "label": "positive",
      "score": 0.9997,
      "model": "distilbert-base-uncased-finetuned-sst-2-english",
      "revision": "714eb0fa89d2f80546fda750413ed43d93601a13",
      "backend": "local",
      "cached": false,
      "request_id": "b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7",
      "timestamp": "2026-07-29T18:30:46.234567+00:00"
    }
  ],
  "cache_hits": 1,
  "cache_misses": 2,
  "avg_score": 0.8765
}
```

**Errors:** Same as `/analyze`, plus `422` if batch is empty or >64 items.

### GET /docs

Interactive Swagger UI. Exempt from rate limiting.

## Rate Limiting

Applied only to `/analyze` and `/analyze/batch` per client IP (sliding-window, 30 req/60s default).

**Response Headers (success):**
```
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 27
X-RateLimit-Reset: 45
```

**429 Response:**
```json
{
  "error": "rate_limited",
  "message": "Rate limit exceeded: 30 requests per 60 seconds. Try again later."
}
```

**Headers:**
```
Retry-After: 45
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 45
```

## Environment Variables

- `MODEL_BACKEND` — `local` (default) or `hf`
- `HF_TOKEN` — Hugging Face token (required if backend=hf)
- `HF_MODEL_ID` — override model
- `MODEL_REVISION` — override revision
- `CACHE_SIZE` — LRU max entries (default 1024)
- `RATE_LIMIT_REQUESTS` — default 30
- `RATE_LIMIT_WINDOW` — default 60
