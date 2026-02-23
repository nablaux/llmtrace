"""Tests for llmtrace capture extractors."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from llmtrace.capture.extractors import (
    ExtractedData,
    ExtractorRegistry,
    extract_anthropic,
    extract_generic,
    extract_openai,
)

# ── Mock response objects ────────────────────────────────────────────────


class MockAnthropicUsage:
    def __init__(self, input_tokens: int = 100, output_tokens: int = 50) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockAnthropicContentBlock:
    def __init__(self, block_type: str, **kwargs: Any) -> None:
        self.type = block_type
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockAnthropicMessage:
    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        usage: MockAnthropicUsage | None = None,
        content: list[MockAnthropicContentBlock] | None = None,
    ) -> None:
        self.model = model
        self.usage = usage or MockAnthropicUsage()
        self.content = content or []

    def model_dump(self) -> dict[str, Any]:
        return {"model": self.model, "content": []}


class MockOpenAIUsage:
    def __init__(
        self,
        prompt_tokens: int = 80,
        completion_tokens: int = 40,
        total_tokens: int = 120,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class MockOpenAIFunction:
    def __init__(self, name: str = "search", arguments: str = "{}") -> None:
        self.name = name
        self.arguments = arguments


class MockOpenAIToolCall:
    def __init__(self, function: MockOpenAIFunction | None = None) -> None:
        self.function = function or MockOpenAIFunction()


class MockOpenAIMessage:
    def __init__(self, tool_calls: list[MockOpenAIToolCall] | None = None) -> None:
        self.tool_calls = tool_calls


class MockOpenAIChoice:
    def __init__(self, message: MockOpenAIMessage | None = None) -> None:
        self.message = message or MockOpenAIMessage()


class MockOpenAICompletion:
    def __init__(
        self,
        model: str = "gpt-4",
        usage: MockOpenAIUsage | None = None,
        choices: list[MockOpenAIChoice] | None = None,
    ) -> None:
        self.model = model
        self.usage = usage or MockOpenAIUsage()
        self.choices = choices if choices is not None else [MockOpenAIChoice()]

    def model_dump(self) -> dict[str, Any]:
        return {"model": self.model, "choices": []}


# ── Anthropic extractor tests ───────────────────────────────────────────


class TestExtractAnthropic:
    """Tests for extract_anthropic."""

    def test_basic_message(self) -> None:
        response = MockAnthropicMessage()
        result = extract_anthropic({"model": "claude-3-sonnet"}, response)

        assert isinstance(result, ExtractedData)
        assert result.provider == "anthropic"
        assert result.model == "claude-3-sonnet-20240229"
        assert result.token_usage is not None
        assert result.token_usage.prompt_tokens == 100
        assert result.token_usage.completion_tokens == 50
        assert result.tool_calls == []

    def test_message_with_tool_use(self) -> None:
        content = [
            MockAnthropicContentBlock("text", text="Let me search for that."),
            MockAnthropicContentBlock(
                "tool_use",
                name="web_search",
                input={"query": "python async"},
            ),
            MockAnthropicContentBlock(
                "tool_use",
                name="calculator",
                input={"expression": "2+2"},
            ),
        ]
        response = MockAnthropicMessage(content=content)
        result = extract_anthropic({}, response)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].tool_name == "web_search"
        assert result.tool_calls[0].arguments == {"query": "python async"}
        assert result.tool_calls[1].tool_name == "calculator"
        assert result.tool_calls[1].arguments == {"expression": "2+2"}

    def test_handles_missing_usage(self) -> None:
        response = MagicMock(spec=[])
        response.model = "claude-3"
        response.content = []
        result = extract_anthropic({}, response)

        assert result.model == "claude-3"
        assert result.token_usage is None

    def test_handles_missing_model(self) -> None:
        response = MagicMock(spec=[])
        result = extract_anthropic({}, response)

        assert result.model == "unknown"

    def test_response_payload_uses_model_dump(self) -> None:
        response = MockAnthropicMessage()
        result = extract_anthropic({}, response)

        assert "model" in result.response_payload


# ── OpenAI extractor tests ──────────────────────────────────────────────


class TestExtractOpenAI:
    """Tests for extract_openai."""

    def test_basic_completion(self) -> None:
        response = MockOpenAICompletion()
        result = extract_openai({"model": "gpt-4"}, response)

        assert isinstance(result, ExtractedData)
        assert result.provider == "openai"
        assert result.model == "gpt-4"
        assert result.token_usage is not None
        assert result.token_usage.prompt_tokens == 80
        assert result.token_usage.completion_tokens == 40
        assert result.token_usage.total_tokens == 120
        assert result.tool_calls == []

    def test_completion_with_tool_calls(self) -> None:
        tool_calls = [
            MockOpenAIToolCall(MockOpenAIFunction("get_weather", json.dumps({"city": "London"}))),
            MockOpenAIToolCall(MockOpenAIFunction("get_time", json.dumps({"timezone": "UTC"}))),
        ]
        message = MockOpenAIMessage(tool_calls=tool_calls)
        choice = MockOpenAIChoice(message=message)
        response = MockOpenAICompletion(choices=[choice])
        result = extract_openai({}, response)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].tool_name == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "London"}
        assert result.tool_calls[1].tool_name == "get_time"
        assert result.tool_calls[1].arguments == {"timezone": "UTC"}

    def test_handles_missing_usage(self) -> None:
        response = MagicMock(spec=[])
        response.model = "gpt-4"
        response.choices = []
        result = extract_openai({}, response)

        assert result.model == "gpt-4"
        assert result.token_usage is None

    def test_handles_missing_model(self) -> None:
        response = MagicMock(spec=[])
        response.choices = []
        result = extract_openai({}, response)

        assert result.model == "unknown"

    def test_handles_invalid_tool_call_json(self) -> None:
        tool_calls = [
            MockOpenAIToolCall(MockOpenAIFunction("broken", "not-valid-json")),
        ]
        message = MockOpenAIMessage(tool_calls=tool_calls)
        choice = MockOpenAIChoice(message=message)
        response = MockOpenAICompletion(choices=[choice])
        result = extract_openai({}, response)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "broken"
        assert result.tool_calls[0].arguments == {}

    def test_response_payload_uses_model_dump(self) -> None:
        response = MockOpenAICompletion()
        result = extract_openai({}, response)

        assert "model" in result.response_payload


# ── Generic extractor tests ─────────────────────────────────────────────


class TestExtractGeneric:
    """Tests for extract_generic."""

    def test_arbitrary_object(self) -> None:
        response = MagicMock()
        response.model = "custom-model"
        response.model_dump.side_effect = AttributeError
        del response.model_dump
        result = extract_generic("custom", {}, response)

        assert result.provider == "custom"
        assert result.model == "custom-model"
        assert result.tool_calls == []

    def test_with_usage_input_output_fields(self) -> None:
        usage = MagicMock()
        usage.input_tokens = 50
        usage.output_tokens = 25
        usage.prompt_tokens = None
        usage.completion_tokens = None
        response = MagicMock()
        response.model = "custom-model"
        response.usage = usage
        response.model_dump.side_effect = Exception("nope")
        result = extract_generic("custom", {}, response)

        assert result.token_usage is not None
        assert result.token_usage.prompt_tokens == 50
        assert result.token_usage.completion_tokens == 25

    def test_no_model_attribute(self) -> None:
        response = MagicMock(spec=[])
        result = extract_generic("custom", {"key": "value"}, response)

        assert result.model == "unknown"
        assert result.request_payload == {"key": "value"}

    def test_response_fallback_to_str(self) -> None:
        response = object()
        result = extract_generic("custom", {}, response)

        assert "raw" in result.response_payload


# ── ExtractorRegistry tests ─────────────────────────────────────────────


class TestExtractorRegistry:
    """Tests for ExtractorRegistry."""

    def test_get_anthropic(self) -> None:
        registry = ExtractorRegistry()
        extractor = registry.get("anthropic")
        response = MockAnthropicMessage()
        result = extractor({}, response)

        assert result.provider == "anthropic"

    def test_get_openai(self) -> None:
        registry = ExtractorRegistry()
        extractor = registry.get("openai")
        response = MockOpenAICompletion()
        result = extractor({}, response)

        assert result.provider == "openai"

    def test_unknown_provider_falls_back_to_generic(self) -> None:
        registry = ExtractorRegistry()
        extractor = registry.get("cohere")
        response = MagicMock()
        response.model = "command-r"
        response.model_dump.side_effect = Exception
        result = extractor({}, response)

        assert result.provider == "cohere"
        assert result.model == "command-r"

    def test_register_custom_extractor(self) -> None:
        registry = ExtractorRegistry()

        def custom_extractor(kwargs: dict[str, Any], resp: Any) -> ExtractedData:
            return ExtractedData(
                provider="custom",
                model="custom-v1",
                request_payload=kwargs,
                response_payload={"custom": True},
            )

        registry.register("custom", custom_extractor)
        extractor = registry.get("custom")
        result = extractor({}, None)

        assert result.provider == "custom"
        assert result.model == "custom-v1"
        assert result.response_payload == {"custom": True}


# ── Request sanitization tests ──────────────────────────────────────────


class TestRequestSanitization:
    """Tests that sensitive fields are redacted from request payloads."""

    def test_api_key_redacted_anthropic(self) -> None:
        response = MockAnthropicMessage()
        result = extract_anthropic({"model": "claude-3", "api_key": "sk-secret-123"}, response)

        assert result.request_payload["api_key"] == "[REDACTED]"
        assert result.request_payload["model"] == "claude-3"

    def test_api_key_redacted_openai(self) -> None:
        response = MockOpenAICompletion()
        result = extract_openai({"model": "gpt-4", "api_key": "sk-secret-456"}, response)

        assert result.request_payload["api_key"] == "[REDACTED]"
        assert result.request_payload["model"] == "gpt-4"

    def test_authorization_redacted(self) -> None:
        response = MockAnthropicMessage()
        result = extract_anthropic(
            {"model": "claude-3", "authorization": "Bearer sk-123"}, response
        )

        assert result.request_payload["authorization"] == "[REDACTED]"

    def test_nested_sensitive_keys_redacted(self) -> None:
        response = MockAnthropicMessage()
        result = extract_anthropic(
            {"model": "claude-3", "headers": {"x-api-key": "sk-123", "accept": "json"}},
            response,
        )

        assert result.request_payload["headers"]["x-api-key"] == "[REDACTED]"
        assert result.request_payload["headers"]["accept"] == "json"

    def test_other_fields_preserved(self) -> None:
        response = MockAnthropicMessage()
        kwargs = {
            "model": "claude-3",
            "max_tokens": 1024,
            "temperature": 0.7,
            "api_key": "sk-secret",
        }
        result = extract_anthropic(kwargs, response)

        assert result.request_payload["max_tokens"] == 1024
        assert result.request_payload["temperature"] == 0.7
        assert result.request_payload["api_key"] == "[REDACTED]"

    def test_original_kwargs_not_mutated(self) -> None:
        response = MockAnthropicMessage()
        kwargs = {"model": "claude-3", "api_key": "sk-secret"}
        extract_anthropic(kwargs, response)

        assert kwargs["api_key"] == "sk-secret"

    def test_generic_extractor_sanitizes_request(self) -> None:
        response = MagicMock(spec=[])
        result = extract_generic("custom", {"api_key": "sk-123", "model": "test"}, response)

        assert result.request_payload["api_key"] == "[REDACTED]"
        assert result.request_payload["model"] == "test"
