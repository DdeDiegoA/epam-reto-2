"""Integration tests for the FastAPI app in app.py.

Nothing here touches the network or loads a real model: every backend call
is intercepted by swapping the entry in ``sentiment._BACKENDS`` for the
active backend name (see the module docstring in test_sentiment.py for why
patching ``sentiment._run_local``/``_run_hf`` by name would not work here).
"""

from __future__ import annotations

import os

os.environ.setdefault("MODEL_BACKEND", "local")

import unittest
from contextlib import contextmanager
from unittest import mock
from unittest.mock import Mock

from fastapi.testclient import TestClient

import app as app_module
import sentiment

client = TestClient(app_module.app)


@contextmanager
def patched_backend(fn):
    with mock.patch.dict(sentiment._BACKENDS, {sentiment.BACKEND: fn}):
        yield


def _fixed_result_backend(label: str = "POSITIVE", score: float = 0.95):
    def _fn(texts: list[str]) -> list[tuple[str, float]]:
        return [(label, score) for _ in texts]

    return _fn


class AnalyzeEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        sentiment.clear_cache()

    def tearDown(self) -> None:
        sentiment.clear_cache()

    def test_analyze_happy_path(self) -> None:
        with patched_backend(_fixed_result_backend("POSITIVE", 0.95)):
            response = client.post("/analyze", json={"text": "I love this."})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            set(body.keys()),
            {"text", "label", "score", "model", "revision", "backend", "cached", "request_id", "timestamp"},
        )
        self.assertEqual(body["label"], "positive")
        self.assertIsInstance(body["request_id"], str)
        self.assertTrue(len(body["request_id"]) > 0)
        self.assertIsInstance(body["timestamp"], str)
        self.assertTrue(len(body["timestamp"]) > 0)

    def test_analyze_empty_text_returns_422_with_uniform_error_body(self) -> None:
        response = client.post("/analyze", json={"text": ""})

        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertIn("error", body)
        self.assertIn("message", body)
        self.assertEqual(body["error"], "validation_error")
        self.assertNotIn("detail", body)

    def test_analyze_too_long_text_returns_422_with_uniform_error_body(self) -> None:
        response = client.post("/analyze", json={"text": "x" * 2001})

        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertIn("error", body)
        self.assertIn("message", body)
        self.assertEqual(body["error"], "validation_error")
        self.assertNotIn("detail", body)

    def test_analyze_batch_happy_path(self) -> None:
        with patched_backend(_fixed_result_backend("NEGATIVE", 0.6)):
            response = client.post(
                "/analyze/batch", json={"texts": ["bad service", "also bad"]}
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["count"], 2)
        self.assertEqual(len(body["results"]), 2)
        self.assertIn("cache_hits", body)
        self.assertIn("cache_misses", body)
        self.assertIn("avg_score", body)
        self.assertIsInstance(body["cache_hits"], int)
        self.assertIsInstance(body["cache_misses"], int)
        self.assertIsInstance(body["avg_score"], float)

    def test_backend_error_maps_to_status_and_retry_after_header(self) -> None:
        def failing_backend(texts: list[str]) -> list[tuple[str, float]]:
            raise sentiment.BackendError("model_loading", "still loading", 503, retry_after=10)

        with patched_backend(failing_backend):
            response = client.post("/analyze", json={"text": "anything"})

        self.assertEqual(response.status_code, 503)
        body = response.json()
        self.assertEqual(body["error"], "model_loading")
        self.assertEqual(response.headers["Retry-After"], "10")


class RateLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        sentiment.clear_cache()
        self._original_limiter = app_module.limiter

    def tearDown(self) -> None:
        app_module.limiter = self._original_limiter
        sentiment.clear_cache()

    def test_third_request_is_rate_limited(self) -> None:
        app_module.limiter = sentiment.RateLimiter(2, 60)

        with patched_backend(_fixed_result_backend()):
            r1 = client.post("/analyze", json={"text": "one"})
            r2 = client.post("/analyze", json={"text": "two"})
            r3 = client.post("/analyze", json={"text": "three"})

        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r3.status_code, 429)

        body = r3.json()
        self.assertEqual(body["error"], "rate_limited")
        self.assertIn("Retry-After", r3.headers)
        self.assertEqual(r3.headers["X-RateLimit-Remaining"], "0")

        for r in (r1, r2):
            self.assertIn("X-RateLimit-Limit", r.headers)
            self.assertIn("X-RateLimit-Remaining", r.headers)

    def test_health_and_docs_are_not_rate_limited(self) -> None:
        app_module.limiter = sentiment.RateLimiter(1, 60)

        for _ in range(5):
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)

        response = client.get("/docs")
        self.assertEqual(response.status_code, 200)


class HealthEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        sentiment.clear_cache()

    def tearDown(self) -> None:
        sentiment.clear_cache()

    def test_health_never_calls_the_backend(self) -> None:
        mock_backend = Mock()
        with patched_backend(mock_backend):
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        mock_backend.assert_not_called()

        body = response.json()
        self.assertIn("hf_token_present", body)
        self.assertIsInstance(body["hf_token_present"], bool)

    def test_health_never_leaks_a_real_token(self) -> None:
        synthetic_token = "hf_synthetic_do_not_use"
        with mock.patch.dict(os.environ, {"HF_TOKEN": synthetic_token}):
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        for key, value in response.json().items():
            self.assertNotIn(synthetic_token, str(key))
            self.assertNotIn(synthetic_token, str(value))


class RootEndpointTests(unittest.TestCase):
    def test_root_mentions_docs(self) -> None:
        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("/docs", response.text)


if __name__ == "__main__":
    unittest.main()
