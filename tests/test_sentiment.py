"""Unit tests for sentiment.py.

Nothing here touches the network or loads a real model. Every backend call is
intercepted by swapping the entry in ``sentiment._BACKENDS`` for the active
backend name, rather than patching ``sentiment._run_local``/``_run_hf`` by
name.

Why: ``sentiment._BACKENDS`` is built at import time as
``{"local": _run_local, "hf": _run_hf}`` -- a dict of direct function-object
references. ``unittest.mock.patch("sentiment._run_local", ...)`` only rebinds
the module-level *name* ``sentiment._run_local``; it does not touch the
reference already stored in ``_BACKENDS``. Since ``_classify``/
``_cached_classify`` always dispatch via ``_BACKENDS[backend](...)``, patching
the name has no effect on what actually runs -- verified empirically: doing so
still loaded the real transformers pipeline. Patching
``sentiment._BACKENDS[<active backend>]`` (e.g. via ``mock.patch.dict``) is
the target that is actually consulted at call time, so that is what every
test below uses.
"""

from __future__ import annotations

import os

os.environ.setdefault("MODEL_BACKEND", "local")

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sentiment
from cli import load_dataset


def _backend_returning(label: str, score: float = 0.9):
    """Build a fake backend callable that returns (label, score) for every text."""

    def _fn(texts: list[str]) -> list[tuple[str, float]]:
        return [(label, score) for _ in texts]

    return _fn


def _counting_backend(label: str = "POSITIVE", score: float = 0.9):
    """Build a fake backend callable that also records every call it receives."""
    calls: list[list[str]] = []

    def _fn(texts: list[str]) -> list[tuple[str, float]]:
        calls.append(list(texts))
        return [(label, score) for _ in texts]

    return _fn, calls


class LabelNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        sentiment.clear_cache()

    def tearDown(self) -> None:
        sentiment.clear_cache()

    def _analyze_with_raw_label(self, raw_label: str, text: str) -> dict:
        with mock.patch.dict(sentiment._BACKENDS, {sentiment.BACKEND: _backend_returning(raw_label)}):
            return sentiment.analyze(text)

    def test_positive_maps_to_positive(self) -> None:
        result = self._analyze_with_raw_label("POSITIVE", "text a")
        self.assertEqual(result["label"], "positive")

    def test_negative_maps_to_negative(self) -> None:
        result = self._analyze_with_raw_label("NEGATIVE", "text b")
        self.assertEqual(result["label"], "negative")

    def test_label_0_maps_to_negative(self) -> None:
        result = self._analyze_with_raw_label("LABEL_0", "text c")
        self.assertEqual(result["label"], "negative")

    def test_label_1_maps_to_positive(self) -> None:
        result = self._analyze_with_raw_label("LABEL_1", "text d")
        self.assertEqual(result["label"], "positive")

    def test_unknown_label_raises_backend_error_with_code(self) -> None:
        with self.assertRaises(sentiment.BackendError) as ctx:
            self._analyze_with_raw_label("3 stars", "text e")
        self.assertEqual(ctx.exception.code, "unexpected_upstream_response")


class ValidationTests(unittest.TestCase):
    def test_empty_string_raises(self) -> None:
        with self.assertRaises(sentiment.InputError):
            sentiment.validate_text("")

    def test_whitespace_only_raises(self) -> None:
        with self.assertRaises(sentiment.InputError):
            sentiment.validate_text("   ")

    def test_over_length_raises(self) -> None:
        with self.assertRaises(sentiment.InputError):
            sentiment.validate_text("x" * (sentiment.MAX_TEXT_LENGTH + 1))

    def test_non_string_raises(self) -> None:
        with self.assertRaises(sentiment.InputError):
            sentiment.validate_text(123)  # type: ignore[arg-type]

    def test_exact_max_length_is_accepted(self) -> None:
        text = "x" * sentiment.MAX_TEXT_LENGTH
        self.assertEqual(sentiment.validate_text(text), text)

    def test_batch_empty_raises(self) -> None:
        with self.assertRaises(sentiment.InputError):
            sentiment.validate_batch([])

    def test_batch_over_size_raises(self) -> None:
        with self.assertRaises(sentiment.InputError):
            sentiment.validate_batch(["a"] * (sentiment.MAX_BATCH_SIZE + 1))

    def test_batch_non_list_raises(self) -> None:
        with self.assertRaises(sentiment.InputError):
            sentiment.validate_batch("not a list")  # type: ignore[arg-type]

    def test_batch_exact_max_size_is_accepted(self) -> None:
        texts = ["a"] * sentiment.MAX_BATCH_SIZE
        self.assertEqual(sentiment.validate_batch(texts), texts)


class CacheTests(unittest.TestCase):
    def setUp(self) -> None:
        sentiment.clear_cache()

    def tearDown(self) -> None:
        sentiment.clear_cache()

    def test_same_text_hits_cache_on_second_call(self) -> None:
        backend, calls = _counting_backend()
        with mock.patch.dict(sentiment._BACKENDS, {sentiment.BACKEND: backend}):
            first = sentiment.analyze("repeat me")
            second = sentiment.analyze("repeat me")

        self.assertEqual(len(calls), 1)
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])

    def test_different_text_calls_backend_again(self) -> None:
        backend, calls = _counting_backend()
        with mock.patch.dict(sentiment._BACKENDS, {sentiment.BACKEND: backend}):
            sentiment.analyze("text one")
            sentiment.analyze("text two")

        self.assertEqual(len(calls), 2)

    def test_cache_stats_hits_and_misses(self) -> None:
        backend, _ = _counting_backend()
        with mock.patch.dict(sentiment._BACKENDS, {sentiment.BACKEND: backend}):
            sentiment.analyze("stat check")
            after_miss = sentiment.cache_stats()
            sentiment.analyze("stat check")
            after_hit = sentiment.cache_stats()

        self.assertEqual(after_miss["misses"], 1)
        self.assertEqual(after_miss["hits"], 0)
        self.assertEqual(after_hit["misses"], 1)
        self.assertEqual(after_hit["hits"], 1)

    def test_analyze_batch_reuses_cache_for_repeated_items(self) -> None:
        backend, calls = _counting_backend()
        with mock.patch.dict(sentiment._BACKENDS, {sentiment.BACKEND: backend}):
            results = sentiment.analyze_batch(["dup", "dup", "unique"])

        # "dup" classified once, "unique" once => 2 backend calls total.
        self.assertEqual(len(calls), 2)
        self.assertFalse(results[0]["cached"])
        self.assertTrue(results[1]["cached"])
        self.assertFalse(results[2]["cached"])


class ResultShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        sentiment.clear_cache()

    def tearDown(self) -> None:
        sentiment.clear_cache()

    def test_result_has_exact_keys_and_expected_echoed_values(self) -> None:
        backend = _backend_returning("POSITIVE", 0.123456789)
        with mock.patch.dict(sentiment._BACKENDS, {sentiment.BACKEND: backend}):
            result = sentiment.analyze("shape check")

        self.assertEqual(
            set(result.keys()),
            {"text", "label", "score", "model", "revision", "backend", "cached"},
        )
        self.assertIsInstance(result["score"], float)
        self.assertEqual(result["score"], round(result["score"], 4))
        self.assertEqual(result["model"], sentiment.MODEL_ID)
        self.assertEqual(result["revision"], sentiment.MODEL_REVISION)
        self.assertEqual(result["backend"], sentiment.BACKEND)


class RateLimiterTests(unittest.TestCase):
    def test_allow_allow_deny_then_recover_after_window(self) -> None:
        clock = [0.0]
        limiter = sentiment.RateLimiter(limit=2, window=60, clock=lambda: clock[0])

        allowed, remaining, _ = limiter.check("ip-1")
        self.assertTrue(allowed)
        self.assertEqual(remaining, 1)

        allowed, remaining, _ = limiter.check("ip-1")
        self.assertTrue(allowed)
        self.assertEqual(remaining, 0)

        allowed, remaining, reset_in = limiter.check("ip-1")
        self.assertFalse(allowed)
        self.assertEqual(remaining, 0)
        self.assertGreater(reset_in, 0)

        clock[0] = 60.001  # past the window: earliest hit has expired
        allowed, remaining, _ = limiter.check("ip-1")
        self.assertTrue(allowed)
        self.assertEqual(remaining, 1)

    def test_different_keys_have_independent_budgets(self) -> None:
        limiter = sentiment.RateLimiter(limit=1, window=60, clock=lambda: 0.0)

        allowed_a1, _, _ = limiter.check("key-a")
        allowed_a2, _, _ = limiter.check("key-a")
        allowed_b1, _, _ = limiter.check("key-b")

        self.assertTrue(allowed_a1)
        self.assertFalse(allowed_a2)
        self.assertTrue(allowed_b1)

    def test_reset_in_is_zero_for_a_fresh_key_with_zero_limit(self) -> None:
        # With limit=0 nothing is ever allowed, and a key that has never
        # recorded a hit has no pending window to wait out, so reset_in is
        # 0.0 rather than the window length.
        limiter = sentiment.RateLimiter(limit=0, window=60, clock=lambda: 0.0)
        allowed, remaining, reset_in = limiter.check("brand-new-key")
        self.assertFalse(allowed)
        self.assertEqual(remaining, 0)
        self.assertEqual(reset_in, 0.0)


class BackendInfoSecretSafetyTests(unittest.TestCase):
    def test_hf_token_present_is_bool_and_value_never_leaks(self) -> None:
        synthetic_token = "hf_synthetic_do_not_use"
        with mock.patch.dict(os.environ, {"HF_TOKEN": synthetic_token}):
            info = sentiment.backend_info()

        self.assertIsInstance(info["hf_token_present"], bool)
        self.assertNotIn(synthetic_token, repr(info))


class LoadDatasetTests(unittest.TestCase):
    def test_parses_valid_rows_and_skips_malformed_ones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            content = (
                "A normal sentence.\t1\n"
                "Sentence with a\tliteral tab inside it.\t0\n"
                "\n"
                "A row with a junk label.\tmaybe\n"
            )
            (data_dir / "imdb_labelled.txt").write_text(content, encoding="utf-8")

            rows = load_dataset(data_dir)

        # Only the 2 well-formed rows survive; blank line and junk-label row skipped.
        self.assertEqual(len(rows), 2)
        self.assertIn(("A normal sentence.", 1, "imdb"), rows)
        # Regression test for rsplit vs split: the literal tab inside the
        # sentence must survive intact, only the last tab is the separator.
        self.assertIn(("Sentence with a\tliteral tab inside it.", 0, "imdb"), rows)


if __name__ == "__main__":
    unittest.main()
