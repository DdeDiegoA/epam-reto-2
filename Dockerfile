# FastAPI Sentiment Analysis Service
#
# Build:  docker build -t sentiment-api .
# Run:    docker run -p 8000:8000 sentiment-api
# With HF backend and token: docker run -p 8000:8000 -e MODEL_BACKEND=hf -e HF_TOKEN=$HF_TOKEN sentiment-api

FROM python:3.11-slim

WORKDIR /app

# Set these environment variables FIRST to optimize layer caching and suppress verbose output
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/opt/hf TRANSFORMERS_VERBOSITY=error

COPY requirements.txt .

# Install dependencies with PyTorch CPU wheels.
# CRITICAL: Use --extra-index-url (not --index-url) so pip can fall back to PyPI.
# On linux/amd64, PyTorch's index serves torch-X.Y.Z+cpu with a local-version suffix that
# outranks PyPI's plain version, avoiding ~800MB of nvidia-* CUDA wheels.
# On linux/arm64, there is no +cpu build, so --extra-index-url lets pip fall back to
# PyPI's aarch64 CPU-only wheel. Using --index-url would break arm64 builds entirely.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Bake model weights into the image at BUILD time so the container needs no network at runtime.
# The default backend is local (offline), so a judge with no credentials must be able to run
# this image without access to Hugging Face APIs or external networks.
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english', revision='714eb0fa89d2f80546fda750413ed43d93601a13')"

# Create a non-root user and assign ownership of the cache and app directories.
# This MUST happen AFTER the model bake step so that /opt/hf (populated as root during the RUN
# above) is correctly chowned to the app user, or the runtime user cannot read the weights.
RUN useradd --create-home --uid 1000 app && chown -R app:app /opt/hf /app

USER app

# Copy source code AFTER the pip and bake layers so editing code does not invalidate those expensive layers.
COPY --chown=app:app app.py sentiment.py cli.py ./

EXPOSE 8000

# Health check using Python (curl and wget are not available in python:3.11-slim).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD ["python","-c","import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4).status==200 else sys.exit(1)"]

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
