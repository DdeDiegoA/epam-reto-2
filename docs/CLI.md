# CLI Reference

**Command:** `python cli.py <subcommand> [options]`

## fetch-data

Download and extract the UCI "Sentiment Labelled Sentences" dataset.

```bash
python cli.py fetch-data [--data-dir DIR] [--force]
```

**Options:**
- `--data-dir DIR` — extract to DIR (default: `data/`)
- `--force` — re-download even if files exist

**Output:**
```
Dataset extracted into data/. Found 3 labelled file(s):
  data/imdb_labelled.txt
  data/amazon_cells_labelled.txt
  data/yelp_labelled.txt
```

**Exit Codes:**
- `0` — success
- `4` — InputError (e.g., directory creation failed)
- `5` — BackendError (download failed)

**URL:** https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip (82 KB)

## analyze

Classify the sentiment of a single text.

```bash
python cli.py analyze "Text to analyze" [--json]
```

**Arguments:**
- `text` — 1–2000 characters (required)

**Options:**
- `--json` — output as JSON instead of human-readable

**Output (default):**
```
Text:     The product is great!
Label:    positive
Score:    95.67%
Model:    distilbert-base-uncased-finetuned-sst-2-english (revision 714eb0fa...)
Backend:  local
Cached:   false
```

**Output (--json):**
```json
{
  "text": "The product is great!",
  "label": "positive",
  "score": 0.9567,
  "model": "distilbert-base-uncased-finetuned-sst-2-english",
  "revision": "714eb0fa89d2f80546fda750413ed43d93601a13",
  "backend": "local",
  "cached": false
}
```

**Exit Codes:**
- `0` — success
- `2` — argparse error
- `3` — ConfigError (bad env vars)
- `4` — InputError (text too long/empty)
- `5` — BackendError (model/API failure)

## evaluate

Run the labeled dataset through the model and report metrics.

```bash
python cli.py evaluate [--data-dir DIR] [--source SOURCE] [--limit N] [--json]
```

**Options:**
- `--data-dir DIR` — dataset location (default: `data/`)
- `--source {imdb,yelp,amazon,all}` — filter by source (default: all)
- `--limit N` — take first N sentences per source (0=all, default: 0)
- `--json` — output as JSON instead of human-readable

**Output (default):**
```
=== Overall ===
N:         3000
Accuracy:  92.80%
Precision: 93.26%
Recall:    92.27%
F1:        92.76%

=== Per-source breakdown ===
source          n   accuracy  precision     recall         f1
amazon       1000     92.50%     93.46%     91.40%     92.42%
imdb         1000     93.50%     94.66%     92.20%     93.41%
yelp         1000     92.40%     91.73%     93.20%     92.46%

=== Confusion matrix (positive class = 'positive') ===
TP: 1384   FP: 100
FN: 116   TN: 1400

Elapsed:          184.89s
Sentences/second: 16.23
Model:            distilbert-base-uncased-finetuned-sst-2-english (revision 714eb0fa...)
Backend:          local
```

**Output (--json):**
```json
{
  "overall": {
    "accuracy": 0.928,
    "precision": 0.9326,
    "recall": 0.9227,
    "f1": 0.9276
  },
  "confusion_matrix": {
    "tp": 1384,
    "fp": 100,
    "tn": 1400,
    "fn": 116
  },
  "per_source": {
    "amazon": {
      "accuracy": 0.925,
      "precision": 0.9346,
      "recall": 0.914,
      "f1": 0.9242,
      "tp": 428,
      "fp": 28,
      "tn": 459,
      "fn": 85,
      "count": 1000
    },
    ...
  },
  "n": 3000,
  "elapsed_seconds": 184.89,
  "sentences_per_second": 16.23,
  "model": "distilbert-base-uncased-finetuned-sst-2-english",
  "revision": "714eb0fa89d2f80546fda750413ed43d93601a13",
  "backend": "local"
}
```

**Exit Codes:**
- `0` — success
- `4` — InputError (dataset missing, run fetch-data first)
- `5` — BackendError

## Environment Variables

- `MODEL_BACKEND` — `local` (default) or `hf`
- `HF_TOKEN` — Hugging Face token (required if backend=hf)
- `HF_MODEL_ID` — override model ID
- `MODEL_REVISION` — override model revision
- `CACHE_SIZE` — LRU cache size (default 1024)
- `RATE_LIMIT_REQUESTS` — default 30 (CLI doesn't rate-limit; API does)
- `RATE_LIMIT_WINDOW` — default 60 (seconds)

## Examples

```bash
# Fetch dataset
python cli.py fetch-data

# Analyze one text
python cli.py analyze "This product is amazing!"
python cli.py analyze "This product is terrible." --json

# Evaluate on subset
python cli.py evaluate --limit 100 --source amazon
python cli.py evaluate --source yelp --json > results.json

# Use Hugging Face backend
MODEL_BACKEND=hf HF_TOKEN=hf_xyz123... python cli.py analyze "test"
```
