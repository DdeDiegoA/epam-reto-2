# 🎭 Sentiment Analysis API — EPAM "Python Run, Debug the Future" Challenge 2

A small, well-documented sentiment analysis service. It classifies English text as **positive** or **negative** using a real, pinned Hugging Face model, exposes it as both a **FastAPI HTTP API** and a **CLI**, and is evaluated end-to-end against a public, labeled dataset (3,000 sentences, 92.80% accuracy — see [Results](#-results--accuracy)). It ships with an in-process cache, a sliding-window rate limiter, a Dockerfile, and a CI pipeline that runs the test suite and a Docker smoke test on every push.

**Author:** Diego Arenas (diegoarenas111@gmail.com)

---

## 🚀 Quick Start

Two paths to run the service. Choose one:

- **Docker (recommended)** — zero Python setup, model weights baked into the image at build time, runs fully offline. Best for grading — the judge can `docker run` the image with no credentials needed. See [setup instructions](#-docker-setup-macos) if you're on macOS without Docker Desktop.
- **venv + pip** — classical local setup. Good for development, running tests, or inspecting the code.

> ⚠️ **Before you start:** the first local run downloads the model weights (~268 MB) from Hugging Face, and installing `torch` itself is a large download (several hundred MB, more on some platforms). Make sure you have a decent connection and a few minutes free. After that first run, everything is cached and works offline. Docker users get this for free at build time.

### Option A: Docker (recommended)

```bash
# 1. Clone the repository
git clone <this-repo-url>
cd "resolucion de reto_Claude"

# 2. Build the image (downloads model weights once at build time)
docker build -t sentiment-api .

# 3. Run it
docker run -p 8000:8000 sentiment-api

# 4. Open the interactive docs
open http://127.0.0.1:8000/docs

# With the hf backend and a token:
export HF_TOKEN=hf_xxx...
docker run -p 8000:8000 -e MODEL_BACKEND=hf -e HF_TOKEN=$HF_TOKEN sentiment-api
```

The Docker image bakes model weights at **build time**, so the container needs no network at runtime with the default `local` backend. It runs as a non-root `app` user. Healthcheck is pure Python (`urllib.request` against `/health`) since the slim image has no `curl`/`wget`.

### Option B: venv + pip

```bash
# 1. Clone the repository
git clone <this-repo-url>
cd "resolucion de reto_Claude"

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
# On Linux/amd64, add the PyTorch CPU index to skip ~800 MB of unused CUDA wheels
# (use --extra-index-url, not --index-url, so arm64 can still fall back to PyPI's aarch64 wheel):
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# 4. Run the API (default backend: local, no credentials needed)
python app.py
# or: uvicorn app:app --host 127.0.0.1 --port 8000

# 5. Open the interactive docs
open http://127.0.0.1:8000/docs
```

Try it with `curl`:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely loved this product."}'
```

Real response shape (score/cached will vary by run):

```json
{
  "text": "I absolutely loved this product.",
  "label": "positive",
  "score": 0.9998,
  "model": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  "revision": "714eb0fa89d2f80546fda750413ed43d93601a13",
  "backend": "local",
  "cached": false
}
```

Or skip the API entirely and use the CLI:

```bash
python cli.py analyze "This product is great!"
```

---

## 📊 Dataset

**UCI "Sentiment Labelled Sentences"**

| | |
|---|---|
| Page | https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences |
| Direct download | https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip |
| Size | 3,000 labeled sentences: 1,000 each from IMDB, Amazon, Yelp; 1,500 positive / 1,500 negative |
| Format | `sentence<TAB>label`, one per line; label `1` = positive, `0` = negative |
| License | CC BY 4.0 |
| Citation | Kotzias, D. (2015). *Sentiment Labelled Sentences* [Dataset]. UCI Machine Learning Repository. |

The dataset is **not** committed to the repo. Download it with:

```bash
python cli.py fetch-data
# or: python cli.py fetch-data --data-dir data --force
```

---

## 🧠 Model & Backends

**Model:** [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english), pinned at revision `714eb0fa89d2f80546fda750413ed43d93601a13`. A binary SST-2 classifier that outputs `POSITIVE`/`NEGATIVE`, normalized here to lowercase `positive`/`negative`. Inputs are truncated to 512 tokens.

Two interchangeable backends, selected by `MODEL_BACKEND`:

- **`local` (default)** — runs the model on your own machine via `transformers`. Needs **no credentials**. Guarantees any judge can run this project with zero setup beyond installing dependencies. Downloads ~268 MB of weights on first use, then runs fully offline.
- **`hf`** — calls the Hugging Face Inference API (`huggingface_hub.InferenceClient`, provider `hf-inference`). Requires `HF_TOKEN`. This path exists specifically to demonstrate the challenge's mandatory environment-variable secret handling. It is *not* the default because Hugging Face's free tier provides only about $0.10/month of inference credits, and HF-side model availability is outside this project's control — `local` is the reliable path for grading.

The app **fails fast** (exits at import, before serving anything) if `MODEL_BACKEND=hf` and `HF_TOKEN` is missing.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_BACKEND` | `local` | `local` (on-device `transformers`) or `hf` (Hugging Face Inference API) |
| `HF_TOKEN` | *(none)* | Hugging Face access token. Required only when `MODEL_BACKEND=hf`. Never hardcoded, never logged, never returned by any endpoint |
| `HF_MODEL_ID` | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | Model repo id to load/query |
| `MODEL_REVISION` | `714eb0fa89d2f80546fda750413ed43d93601a13` | Pinned model revision (commit hash), for reproducible results |
| `CACHE_SIZE` | `1024` | Max entries in the in-process result cache; `0` disables it |
| `RATE_LIMIT_REQUESTS` | `30` | Requests allowed per client per window |
| `RATE_LIMIT_WINDOW` | `60` | Rate-limit window length, in seconds |
| `HF_TIMEOUT` | `20.0` | Per-request timeout (seconds) for the `hf` backend |

Running with the `hf` backend:

```bash
export HF_TOKEN=hf_xxx...          # never written to code or committed to the repo
export MODEL_BACKEND=hf
python app.py
```

`HF_TOKEN` is read only from the environment (`os.getenv`/`os.environ`) — it is never hardcoded anywhere in the source. `.env` and `.env.*` are gitignored, so a local secrets file can never be committed by accident.

---

## 🌐 API Reference

| Method | Path | Rate limited? | Description |
|---|---|---|---|
| GET | `/` | no | Service banner + endpoint list |
| GET | `/health` | no | Liveness probe; never loads the model or touches the network |
| GET | `/docs` | no | Interactive Swagger UI |
| POST | `/analyze` | yes | Single-text sentiment |
| POST | `/analyze/batch` | yes | Batch sentiment |

**`GET /health`**

```json
{
  "status": "ok",
  "backend": "local",
  "model": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  "revision": "714eb0fa89d2f80546fda750413ed43d93601a13",
  "hf_token_present": false,
  "cache": {"enabled": true, "hits": 0, "misses": 0, "size": 0, "max_size": 1024}
}
```

**`POST /analyze`** — request `{"text": "I absolutely loved this product."}` → response as shown in [Quick Start](#-quick-start).

**`POST /analyze/batch`** — request:

```json
{"texts": ["Great service.", "Terrible experience."]}
```

response:

```json
{"count": 2, "results": [ { "...": "one AnalyzeResponse per text" } ]}
```

### Error handling

Every error, from every source, has the same JSON shape:

```json
{"error": "<code>", "message": "<actionable message, no secrets>"}
```

| Status | `error` code | When |
|---|---|---|
| 422 | `validation_error` | Empty/too-long text, empty/oversized batch, malformed request body |
| 429 | `rate_limited` | This service's own rate limiter rejected the request (`Retry-After` header included) |
| 500 | `configuration_error` | Bad server configuration (should only occur at import/boot) |
| 500 | `backend_unavailable` | Required package (`transformers`/`huggingface_hub`) not installed |
| 500 | `backend_auth_error` | Hugging Face rejected the credentials (bad `HF_TOKEN`) |
| 500 | `model_not_found` | Model/revision not found on Hugging Face |
| 500 | `internal_error` | Unhandled exception |
| 502 | `backend_unreachable` / `backend_error` | Network/DNS issue or unexpected upstream error |
| 502 | `unexpected_upstream_response` | Model returned a label this service doesn't recognize |
| 503 | `model_unavailable` | Local model failed to load (offline, no cached weights) |
| 503 | `model_loading` | Hosted model is cold-starting on Hugging Face infrastructure |
| 503 | `backend_rate_limited` | Hugging Face's **own** API rate limit was hit — deliberately surfaced as 503, not 429, because 429 is reserved for *this service's* rate limiter, so a client can always tell whose quota ran out |
| 504 | `backend_timeout` | Request to the Hugging Face Inference API timed out |

---

## 🖥️ CLI Reference

```
python cli.py fetch-data [--data-dir data] [--force]
python cli.py analyze "<text>" [--json]
python cli.py evaluate [--data-dir data] [--limit N] [--source {imdb,yelp,amazon,all}] [--json]
```

- **`fetch-data`** — downloads and extracts the UCI dataset. Skips re-downloading if files already exist, unless `--force` is passed.
- **`analyze`** — runs one string of text through the model. `--json` prints the raw result object instead of a human-readable summary.
- **`evaluate`** — runs the labeled dataset through the model and reports accuracy/precision/recall/F1 overall and per source, plus a confusion matrix. `--limit N` takes the first N rows *per source* (0 = all). `--source` restricts to one source.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Unexpected error |
| 2 | argparse usage error (argparse's own default) |
| 3 | Configuration error (e.g. bad `MODEL_BACKEND` setup) |
| 4 | Input/data error (e.g. missing dataset — run `fetch-data` first) |
| 5 | Backend error (e.g. dataset download failure) |

---

## 📈 Results / Accuracy

Measured with one real, full run over all 3,000 sentences, `MODEL_BACKEND=local`, on an Apple M4 (10 cores), macOS 26.5.1, Python 3.11.15, on 2026-07-29.

**Command:** `python cli.py evaluate`

```
Overall:  N=3000  Accuracy 92.80%  Precision 93.26%  Recall 92.27%  F1 92.76%

Per source:
  amazon  n=1000  accuracy 92.50%  precision 93.46%  recall 91.40%  f1 92.42%
  imdb    n=1000  accuracy 93.50%  precision 94.66%  recall 92.20%  f1 93.41%
  yelp    n=1000  accuracy 92.40%  precision 91.73%  recall 93.20%  f1 92.46%

Confusion matrix (positive class = 'positive'):  TP 1384   FP 100   FN 116   TN 1400

Elapsed 184.89s, 16.23 sentences/second
```

**Domain-shift observation:** IMDB scores highest (93.50% accuracy) because the underlying SST-2 fine-tuning corpus is itself movie-review text — that's an in-domain match. Amazon and Yelp reviews are a slight domain shift from that training distribution, which shows up as a ~1-point accuracy dip on both. The spread is real and reported as-is, not smoothed into one headline number.

---

## 🐳 Docker Setup (macOS)

If you're on macOS and don't have Docker Desktop, use **Colima** — a lightweight, open-source container runtime. It provides the Docker daemon without the resource-heavy Docker Desktop GUI.

### One-time setup

```bash
# Install Colima + Docker CLI + Docker Compose (macOS)
brew install colima docker docker-compose

# Start the Colima daemon
colima start

# Verify everything works
docker run hello-world
```

After this, `docker build` and `docker run` work just like on any Linux machine. No Docker Desktop required.

### Why Colima?

| | Docker Desktop | Colima |
|---|---|---|
| Resource usage | ~2-4 GB RAM, heavy background daemon | Minimal; VM starts on demand |
| GUI | Yes | No (CLI-only, like a real server) |
| License | Proprietary, requires Docker subscription for many orgs | Open source (MIT) |
| Compatibility | Full Docker API | Full Docker API — 1:1 compatible |
| macOS footprint | Heavy launch daemon + menu bar app | Single `colima` binary, no menu bar |

### Troubleshooting

**`docker: command not found`** — `brew install docker` (the CLI is a separate package from Colima).

**`docker compose` not recognized** — two possible causes:
1. Old binary with broken compose plugin: `brew reinstall docker`
2. Daemon not running: `colima start`

**`colima start` fails with `dependency check failed for docker`** — this is the exact error: Colima checks that the `docker` CLI binary exists before starting, because the daemon is useless without a client. Install it:
```bash
brew install docker
colima start
```

### Build and run the sentimental analysis image

```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api

# With the hf backend and a token:
docker run -p 8000:8000 -e MODEL_BACKEND=hf -e HF_TOKEN=$HF_TOKEN sentiment-api
```

- **Base image:** `python:3.11-slim`.
- Model weights are baked in at **build time** (`RUN python -c "from transformers import pipeline; ..."`), so the container needs no network access at runtime with the default `local` backend.
- Runs as a non-root `app` user.
- Healthcheck is pure Python (`urllib.request` against `/health`) since the slim image has no `curl`/`wget`.
- Dependencies install with `--extra-index-url https://download.pytorch.org/whl/cpu` (never `--index-url`), so amd64 builds skip ~800 MB of CUDA wheels while arm64 still falls back to PyPI's CPU-only aarch64 wheel.

The Docker image was built and smoke-tested locally on macOS (Apple Silicon) with Colima, and verified by the GitHub Actions CI pipeline (`.github/workflows/ci.yml`), which builds the image and runs a smoke test (`docker run` → poll `/health` → `POST /analyze` and check for `"positive"` in the response) on every push to `main` and every pull request.

---

## ⚡ Bonus B: Caching & Rate Limiting

**Caching:** an in-process LRU cache (`functools.lru_cache`) keyed on `(backend, model_id, revision, text)`. `CACHE_SIZE=0` disables it entirely. There is deliberately **no TTL**: a pinned model revision scoring a fixed input string is a pure function of its arguments — the result cannot go stale, so expiring it would only add complexity for no benefit. Every `/analyze` and CLI `analyze` response carries a `cached` boolean so callers can see whether they hit it.

**Rate limiting:** a sliding-window log (not a fixed-window counter — a fixed window would let a client burst up to 2x the limit across a window boundary), keyed per client IP (`request.client.host`). Applied **only** to `POST /analyze` and `POST /analyze/batch`, so `/health`, `/docs`, `/redoc` and `/openapi.json` stay freely explorable by a judge without burning quota. `X-Forwarded-For` is deliberately **not** trusted — honoring it unconditionally would let any caller spoof a fresh key on every request and bypass the limiter entirely. Every rate-limited response carries `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`; a rejection returns `429` with a `Retry-After` header.

---

## ✅ Testing

```bash
python -m unittest discover -s tests -t .
```

34 tests, stdlib `unittest`, fully offline (no network calls, no model download — every backend call is swapped out via `mock.patch.dict(sentiment._BACKENDS, ...)`). Runs in about 0.1–0.3 seconds. Covers: label normalization, input validation, caching (hits/misses/batch dedup), result shape, the sliding-window rate limiter, secret-safety of `/health` and `backend_info()`, dataset parsing (`cli.load_dataset`), and the FastAPI endpoints end-to-end (happy paths, validation errors, backend errors, rate limiting, docs/health exemption).

---

## 📁 Project Structure

```
.
├── app.py                     # FastAPI app: routes, error handlers, rate-limit middleware
├── sentiment.py                # Model backends (local/hf), cache, validation, RateLimiter — the only module that knows a model exists
├── cli.py                      # CLI: fetch-data / analyze / evaluate subcommands
├── requirements.txt             # Pinned dependencies
├── Dockerfile                   # Build with baked-in model weights, non-root user, Python healthcheck
├── .github/workflows/ci.yml     # Runs the test suite + a Docker build & smoke test on every push
├── tests/
│   ├── test_sentiment.py        # Unit tests for sentiment.py
│   └── test_api.py              # Integration tests for app.py (via FastAPI TestClient)
└── data/                        # Downloaded dataset (gitignored; created by `fetch-data`)
```

---

## 🧩 Design Notes / Trade-offs

- **No TTL on the cache** — the cache key already includes the model revision, so a cached entry can only ever be replaced by re-pinning the model, not by time passing.
- **Sliding window, not fixed window** — a fixed-window counter resets fully at each boundary, letting a client fire the full limit twice back-to-back around the edge; the sliding log closes that gap for one extra `deque` per key.
- **Same model on both backends** — `local` and `hf` call the identical pinned model/revision, so switching backends changes *where* inference runs, never *what* it computes; results are comparable across backends.
- **stdlib `unittest`, no pytest** — no extra test-runner dependency to install or document; `unittest discover` is enough for 34 tests.
- **No pandas/sklearn** — accuracy/precision/recall/F1/confusion-matrix are a handful of counters and arithmetic (see `cli._metrics`); a full data-science stack would be dead weight for four formulas.
- **Known limitation:** the cache and rate limiter are per-process, in-memory state. Run a single worker process; if this needs to scale to multiple workers/replicas, the upgrade path is a shared Redis instance keyed the same way (backend/model/revision/text for the cache, client IP for the limiter).

---

## 📋 Requirements Met

| Requirement | Where |
|---|---|
| Connect real code to a real AI model, produce a useful result | `sentiment.py` → DistilBERT SST-2 sentiment classifier, exposed via `app.py`/`cli.py` |
| Public dataset, documented with source link | [Dataset](#-dataset) — UCI Sentiment Labelled Sentences, linked and cited |
| Model/service documented, name and version | [Model & Backends](#-model--backends) — exact model id + pinned revision hash |
| API key from environment variables, never hardcoded | `sentiment.py`: `os.getenv("HF_TOKEN")` / `os.environ["HF_TOKEN"]` only; `.env` gitignored |
| Exposed as a Flask/FastAPI endpoint OR CLI, documented | Both: FastAPI in `app.py` ([API Reference](#-api-reference)), CLI in `cli.py` ([CLI Reference](#-cli-reference)) |
| Works correctly, handles errors, well documented | Uniform error JSON + HTTP status table above; 34 tests in `tests/` |
| Bonus A: Dockerize | [Docker Setup](#-docker-setup-macos) — `Dockerfile`, Colima-compatible, verified by CI |
| Bonus B: Caching & rate limiting | [Bonus B](#-bonus-b-caching--rate-limiting) |
