"""Command-line interface for the sentiment analysis challenge.

Provides three subcommands:

  fetch-data  Download and extract the UCI "Sentiment Labelled Sentences"
              dataset into a local directory.
  analyze     Run a single piece of text through the sentiment model.
  evaluate    Run the labeled dataset through the model and report accuracy,
              precision, recall, F1 and a confusion matrix.

Usage examples:

    python cli.py fetch-data
    python cli.py fetch-data --data-dir data --force

    python cli.py analyze "This product is great!"
    python cli.py analyze "Terrible service." --json

    python cli.py evaluate
    python cli.py evaluate --limit 100 --source yelp
    python cli.py evaluate --data-dir data --json

Exit codes:
    0  success
    1  unexpected error
    2  argparse usage error (argparse's own default)
    3  configuration error (ConfigError, e.g. bad model backend setup)
    4  input/data error (InputError, e.g. missing dataset -- run fetch-data)
    5  backend error (BackendError, e.g. network/download failure)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# sentiment.py can itself raise ConfigError at import time (bad MODEL_BACKEND,
# or MODEL_BACKEND=hf without HF_TOKEN). Handle that without a raw traceback.
try:
    import sentiment
except Exception as exc:  # noqa: BLE001 - deliberately broad, see module docstring
    print(f"error: failed to load sentiment module: {exc}", file=sys.stderr)
    sys.exit(3)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def find_labelled_files(data_dir: Path) -> list[Path]:
    """Return the dataset's ``*_labelled.txt`` files, ignoring macOS junk.

    The UCI archive was zipped on macOS, so it carries a ``__MACOSX/`` directory
    holding AppleDouble resource forks named ``._<original>``. Those match the
    glob but contain binary metadata, not sentences. Filtering them here keeps
    the file count reported to the user honest and avoids parsing garbage.
    """
    return sorted(
        path
        for path in data_dir.rglob("*_labelled.txt")
        if not path.name.startswith("._") and "__MACOSX" not in path.parts
    )


def load_dataset(data_dir: Path, source: str = "all", limit: int = 0) -> list[tuple[str, int, str]]:
    """Load labeled sentences from the extracted UCI dataset.

    Locates ``*_labelled.txt`` files recursively under ``data_dir``. Each line
    is ``sentence<TAB>label``; some sentences contain literal tab characters,
    so lines are split from the right (rsplit) on the last tab only.

    Args:
        data_dir: Directory containing the extracted dataset (searched recursively).
        source: One of "imdb", "yelp", "amazon", or "all".
        limit: 0 means all rows; a positive N takes the first N rows *per source*.

    Returns:
        A list of (sentence, label, source_name) tuples.

    Raises:
        sentiment.InputError: if the directory or dataset files are missing.
    """
    if not data_dir.exists():
        raise sentiment.InputError(
            f"Dataset directory not found: {data_dir}. Run 'python cli.py fetch-data' first."
        )

    files = find_labelled_files(data_dir)
    if not files:
        raise sentiment.InputError(
            f"No '*_labelled.txt' files found under {data_dir}. "
            "Run 'python cli.py fetch-data' first."
        )

    wanted = None if source == "all" else source
    rows: list[tuple[str, int, str]] = []
    skipped = 0

    for path in files:
        name = path.stem.replace("_labelled", "")
        if name == "amazon_cells":
            name = "amazon"
        if wanted is not None and name != wanted:
            continue

        per_source_count = 0
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.split("\n"):
            stripped = line.rstrip("\n")
            if not stripped.strip():
                continue
            parts = stripped.rsplit("\t", 1)
            if len(parts) != 2:
                skipped += 1
                continue
            sentence, label_str = parts
            sentence = sentence.strip()
            label_str = label_str.strip()
            if label_str not in ("0", "1") or not sentence:
                skipped += 1
                continue
            if limit and per_source_count >= limit:
                continue
            rows.append((sentence, int(label_str), name))
            per_source_count += 1

    print(f"Skipped {skipped} malformed/blank row(s) while loading dataset.", file=sys.stderr)
    return rows


# ---------------------------------------------------------------------------
# fetch-data
# ---------------------------------------------------------------------------

def cmd_fetch_data(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    existing = find_labelled_files(data_dir)
    if len(existing) >= 3 and not args.force:
        print(f"Dataset already present ({len(existing)} files found) under {data_dir}. Use --force to re-download.")
        return 0

    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        raise sentiment.BackendError(
            "dataset_download_failed",
            "Failed to download the dataset "
            f"({exc}). Please download it manually from {DATASET_URL} "
            f"and extract it into '{data_dir}' yourself.",
        ) from exc

    try:
        with zipfile.ZipFile(BytesIO(payload)) as zf:
            zf.extractall(data_dir)
    except zipfile.BadZipFile as exc:
        raise sentiment.BackendError(
            "dataset_archive_invalid",
            f"Downloaded file is not a valid zip archive ({exc}). "
            f"Please download it manually from {DATASET_URL} and extract it into '{data_dir}' yourself.",
        ) from exc

    found = find_labelled_files(data_dir)
    print(f"Dataset extracted into {data_dir}. Found {len(found)} labelled file(s):")
    for f in found:
        print(f"  {f}")
    return 0


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace) -> int:
    result = sentiment.analyze(args.text)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Text:     {result['text']}")
        print(f"Label:    {result['label']}")
        print(f"Score:    {result['score'] * 100:.2f}%")
        print(f"Model:    {result['model']} (revision {result['revision']})")
        print(f"Backend:  {result['backend']}")
        print(f"Cached:   {result['cached']}")
    return 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def _chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def cmd_evaluate(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    rows = load_dataset(data_dir, source=args.source, limit=args.limit)

    if not rows:
        raise sentiment.InputError(
            f"No usable rows loaded from {data_dir} for source={args.source!r}."
        )

    texts = [r[0] for r in rows]
    true_labels = [r[1] for r in rows]
    sources = [r[2] for r in rows]

    predictions: list[dict] = []
    start = time.monotonic()
    processed = 0
    next_report = 200
    for chunk in _chunked(texts, sentiment.MAX_BATCH_SIZE):
        predictions.extend(sentiment.analyze_batch(chunk)["results"])
        processed += len(chunk)
        if processed >= next_report:
            print(f"...processed {processed}/{len(texts)}", file=sys.stderr)
            next_report += 200
    elapsed = time.monotonic() - start
    print(f"...processed {len(texts)}/{len(texts)}", file=sys.stderr)

    # Overall confusion matrix: "positive" is the positive class.
    tp = fp = tn = fn = 0
    per_source: dict[str, dict[str, int]] = {}

    for pred, true_label, src in zip(predictions, true_labels, sources):
        predicted_positive = pred["label"] == "positive"
        actual_positive = true_label == 1

        bucket = per_source.setdefault(src, {"tp": 0, "fp": 0, "tn": 0, "fn": 0})

        if predicted_positive and actual_positive:
            tp += 1
            bucket["tp"] += 1
        elif predicted_positive and not actual_positive:
            fp += 1
            bucket["fp"] += 1
        elif not predicted_positive and not actual_positive:
            tn += 1
            bucket["tn"] += 1
        else:
            fn += 1
            bucket["fn"] += 1

    overall = _metrics(tp, fp, tn, fn)
    n = len(rows)
    sentences_per_sec = n / elapsed if elapsed > 0 else 0.0
    info = sentiment.backend_info()

    if args.json:
        result = {
            "overall": overall,
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "per_source": {
                src: {**_metrics(b["tp"], b["fp"], b["tn"], b["fn"]), **b, "count": sum(b.values())}
                for src, b in per_source.items()
            },
            "n": n,
            "elapsed_seconds": elapsed,
            "sentences_per_second": sentences_per_sec,
            "model": info.get("model", sentiment.MODEL_ID),
            "revision": info.get("revision", sentiment.MODEL_REVISION),
            "backend": info.get("backend", sentiment.BACKEND),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print()
    print("=== Overall ===")
    print(f"N:         {n}")
    print(f"Accuracy:  {overall['accuracy'] * 100:.2f}%")
    print(f"Precision: {overall['precision'] * 100:.2f}%")
    print(f"Recall:    {overall['recall'] * 100:.2f}%")
    print(f"F1:        {overall['f1'] * 100:.2f}%")

    print()
    print("=== Per-source breakdown ===")
    header = f"{'source':<10} {'n':>6} {'accuracy':>10} {'precision':>10} {'recall':>10} {'f1':>10}"
    print(header)
    for src in sorted(per_source):
        b = per_source[src]
        m = _metrics(b["tp"], b["fp"], b["tn"], b["fn"])
        count = sum(b.values())
        print(
            f"{src:<10} {count:>6} {m['accuracy'] * 100:>9.2f}% {m['precision'] * 100:>9.2f}% "
            f"{m['recall'] * 100:>9.2f}% {m['f1'] * 100:>9.2f}%"
        )

    print()
    print("=== Confusion matrix (positive class = 'positive') ===")
    print(f"TP: {tp}   FP: {fp}")
    print(f"FN: {fn}   TN: {tn}")

    print()
    print(f"Elapsed:          {elapsed:.2f}s")
    print(f"Sentences/second: {sentences_per_sec:.2f}")
    print(f"Model:            {info.get('model', sentiment.MODEL_ID)} "
          f"(revision {info.get('revision', sentiment.MODEL_REVISION)})")
    print(f"Backend:          {info.get('backend', sentiment.BACKEND)}")
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

EPILOG = """\
Exit codes:
  0  success
  1  unexpected error
  2  argparse usage error
  3  configuration error (ConfigError)
  4  input/data error (InputError, e.g. missing dataset)
  5  backend error (BackendError, e.g. network/download failure)
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Sentiment analysis CLI: fetch data, analyze text, and evaluate the model.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser(
        "fetch-data",
        help="Download and extract the UCI sentiment labelled sentences dataset.",
    )
    fetch_parser.add_argument(
        "--data-dir", default="data", help="Directory to extract the dataset into (default: data)."
    )
    fetch_parser.add_argument(
        "--force", action="store_true", help="Re-download even if dataset files already exist."
    )
    fetch_parser.set_defaults(func=cmd_fetch_data)

    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze the sentiment of a single piece of text."
    )
    analyze_parser.add_argument("text", help="Text to analyze.")
    analyze_parser.add_argument(
        "--json", action="store_true", help="Print the raw result as JSON instead of human-readable text."
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run the labeled dataset through the model and report accuracy/precision/recall/F1.",
    )
    evaluate_parser.add_argument(
        "--data-dir", default="data", help="Directory containing the extracted dataset (default: data)."
    )
    evaluate_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 (default) means use ALL rows. A positive N takes the first N rows "
        "PER SOURCE (e.g. --limit 100 with --source all takes up to 100 imdb + "
        "100 yelp + 100 amazon rows).",
    )
    evaluate_parser.add_argument(
        "--source",
        choices=["imdb", "yelp", "amazon", "all"],
        default="all",
        help="Restrict evaluation to one dataset source (default: all).",
    )
    evaluate_parser.add_argument(
        "--json", action="store_true", help="Print results as JSON instead of a human-readable report."
    )
    evaluate_parser.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.func(args)
    except sentiment.ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except sentiment.InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    except sentiment.BackendError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 5
    except Exception as exc:  # noqa: BLE001 - top-level guard, see module docstring
        logger.debug("Unexpected error", exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
