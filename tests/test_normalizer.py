"""Tests for llmtrace.transform.normalizer."""

from __future__ import annotations

from typing import Any

from llmtrace.capture.extractors import ExtractedData
from llmtrace.config import LLMTraceConfig
from llmtrace.models import ErrorTrace, TokenUsage
from llmtrace.transform.normalizer import (
    SENSITIVE_REQUEST_KEYS,
    _classify_error,
    _compute_throughput,
    _deep_redact_keys,
    normalize,
)

# ── _deep_redact_keys ────────────────────────────────────────────────────


class TestDeepRedactKeys:
    def test_redacts_api_key_at_top_level(self) -> None:
        data = {"api_key": "sk-123", "model": "gpt-4"}
        result = _deep_redact_keys(data, SENSITIVE_REQUEST_KEYS)
        assert result["api_key"] == "[REDACTED]"
        assert result["model"] == "gpt-4"

    def test_redacts_authorization_nested_two_levels(self) -> None:
        data = {"headers": {"inner": {"authorization": "Bearer tok"}}}
        result = _deep_redact_keys(data, SENSITIVE_REQUEST_KEYS)
        assert result["headers"]["inner"]["authorization"] == "[REDACTED]"

    def test_case_insensitive(self) -> None:
        data = {"API_KEY": "sk-123", "Authorization": "Bearer tok"}
        result = _deep_redact_keys(data, SENSITIVE_REQUEST_KEYS)
        assert result["API_KEY"] == "[REDACTED]"
        assert result["Authorization"] == "[REDACTED]"

    def test_handles_lists_containing_dicts(self) -> None:
        data = {"items": [{"token": "secret", "ok": 1}, {"safe": "val"}]}
        result = _deep_redact_keys(data, SENSITIVE_REQUEST_KEYS)
        assert result["items"][0]["token"] == "[REDACTED]"
        assert result["items"][0]["ok"] == 1
        assert result["items"][1]["safe"] == "val"

    def test_does_not_mutate_input(self) -> None:
        data = {"api_key": "sk-123", "nested": {"secret": "s"}}
        _deep_redact_keys(data, SENSITIVE_REQUEST_KEYS)
        assert data["api_key"] == "sk-123"
        assert data["nested"]["secret"] == "s"


# ── _compute_throughput ──────────────────────────────────────────────────


class TestComputeThroughput:
    def test_correct_tokens_per_second(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        result = _compute_throughput(usage, 1000.0)  # 1 second
        assert result["tokens_per_second"] == 150.0
        assert result["input_tokens_per_second"] == 100.0
        assert result["output_tokens_per_second"] == 50.0

    def test_zero_latency_returns_empty(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert _compute_throughput(usage, 0.0) == {}

    def test_none_token_usage_returns_empty(self) -> None:
        assert _compute_throughput(None, 1000.0) == {}


# ── _classify_error ──────────────────────────────────────────────────────


class TestClassifyError:
    def test_rate_limit_error_type(self) -> None:
        error = ErrorTrace(error_type="RateLimitError", message="slow down")
        assert _classify_error(error) == "rate_limit"

    def test_context_length_message(self) -> None:
        error = ErrorTrace(error_type="BadRequestError", message="context length exceeded")
        assert _classify_error(error) == "context_length"

    def test_unknown_error(self) -> None:
        error = ErrorTrace(error_type="WeirdError", message="something completely novel")
        assert _classify_error(error) == "unknown"


# ── normalize ────────────────────────────────────────────────────────────


def _make_config(**kwargs: Any) -> LLMTraceConfig:
    return LLMTraceConfig(**kwargs)


def _make_extracted(**kwargs: Any) -> ExtractedData:
    defaults: dict[str, Any] = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "request_payload": {"model": "claude-sonnet-4-20250514", "max_tokens": 100},
        "response_payload": {"content": "hello"},
    }
    defaults.update(kwargs)
    return ExtractedData(**defaults)


class TestNormalize:
    def test_happy_path_with_token_usage(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        extracted = _make_extracted(token_usage=usage)
        config = _make_config()

        event = normalize(extracted, latency_ms=500.0, config=config)

        assert event.provider == "anthropic"
        assert event.token_usage == usage
        assert event.cost is not None
        assert "tokens_per_second" in event.metadata

    def test_without_token_usage(self) -> None:
        extracted = _make_extracted()
        config = _make_config()

        event = normalize(extracted, latency_ms=500.0, config=config)

        assert event.cost is None
        assert "tokens_per_second" not in event.metadata

    def test_capture_request_false(self) -> None:
        extracted = _make_extracted()
        config = _make_config(capture_request=False)

        event = normalize(extracted, latency_ms=100.0, config=config)

        assert event.request == {}

    def test_capture_response_false(self) -> None:
        extracted = _make_extracted()
        config = _make_config(capture_response=False)

        event = normalize(extracted, latency_ms=100.0, config=config)

        assert event.response == {}

    def test_sensitive_keys_redacted_even_when_capture_request_true(self) -> None:
        extracted = _make_extracted(request_payload={"api_key": "sk-123", "model": "claude"})
        config = _make_config(capture_request=True)

        event = normalize(extracted, latency_ms=100.0, config=config)

        assert event.request["api_key"] == "[REDACTED]"
        assert event.request["model"] == "claude"

    def test_tag_and_metadata_merging(self) -> None:
        extracted = _make_extracted()
        config = _make_config(
            default_tags={"env": "prod"},
            default_metadata={"service": "api"},
        )

        event = normalize(
            extracted,
            latency_ms=100.0,
            config=config,
            extra_tags={"env": "staging", "team": "ml"},
            extra_metadata={"version": "2"},
        )

        assert event.tags["env"] == "staging"
        assert event.tags["team"] == "ml"
        assert event.metadata["service"] == "api"
        assert event.metadata["version"] == "2"

    def test_error_classified_in_metadata(self) -> None:
        error = ErrorTrace(error_type="RateLimitError", message="slow down")
        extracted = _make_extracted(error=error)
        config = _make_config()

        event = normalize(extracted, latency_ms=100.0, config=config)

        assert event.error is not None
        assert event.metadata["error_category"] == "rate_limit"

    def test_byte_sizes_in_metadata(self) -> None:
        extracted = _make_extracted()
        config = _make_config()

        event = normalize(extracted, latency_ms=100.0, config=config)

        assert "request_byte_size" in event.metadata
        assert "response_byte_size" in event.metadata
        assert isinstance(event.metadata["request_byte_size"], int)
        assert isinstance(event.metadata["response_byte_size"], int)

    def test_redact_sensitive_keys_false_skips_redaction(self) -> None:
        extracted = _make_extracted(request_payload={"api_key": "sk-123", "model": "claude"})
        config = _make_config()

        event = normalize(extracted, latency_ms=100.0, config=config, redact_sensitive_keys=False)

        assert event.request["api_key"] == "sk-123"


# ── Deep redaction edge cases ──────────────────────────────────────────


class TestDeepRedactEdgeCases:
    def test_deeply_nested_5_levels(self) -> None:
        data = {"a": {"b": {"c": {"d": {"e": {"api_key": "secret"}}}}}}
        result = _deep_redact_keys(data, SENSITIVE_REQUEST_KEYS)
        assert result["a"]["b"]["c"]["d"]["e"]["api_key"] == "[REDACTED]"

    def test_lists_containing_dicts_with_sensitive_keys(self) -> None:
        data = {
            "items": [
                {"token": "secret1", "safe": "ok"},
                {"password": "secret2", "safe": "fine"},
                "plain-string",
                42,
            ]
        }
        result = _deep_redact_keys(data, SENSITIVE_REQUEST_KEYS)
        assert result["items"][0]["token"] == "[REDACTED]"
        assert result["items"][0]["safe"] == "ok"
        assert result["items"][1]["password"] == "[REDACTED]"
        assert result["items"][1]["safe"] == "fine"
        assert result["items"][2] == "plain-string"
        assert result["items"][3] == 42

    def test_mixed_types_in_dict(self) -> None:
        data = {
            "str_val": "hello",
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "none_val": None,
            "list_val": [1, 2, 3],
            "secret": "should-be-redacted",
        }
        result = _deep_redact_keys(data, SENSITIVE_REQUEST_KEYS)
        assert result["str_val"] == "hello"
        assert result["int_val"] == 42
        assert result["float_val"] == 3.14
        assert result["bool_val"] is True
        assert result["none_val"] is None
        assert result["list_val"] == [1, 2, 3]
        assert result["secret"] == "[REDACTED]"

    def test_empty_dict(self) -> None:
        result = _deep_redact_keys({}, SENSITIVE_REQUEST_KEYS)
        assert result == {}

    def test_all_keys_sensitive(self) -> None:
        data = {"api_key": "a", "password": "b", "secret": "c"}
        result = _deep_redact_keys(data, SENSITIVE_REQUEST_KEYS)
        assert all(v == "[REDACTED]" for v in result.values())
