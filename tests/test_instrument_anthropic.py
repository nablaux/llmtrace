"""Tests for AnthropicInstrumentor using mock SDK classes."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from llmtrace.config import configure, reset
from llmtrace.models import TraceEvent
from llmtrace.sinks.callback import CallbackSink

# ── Mock Anthropic SDK structure ─────────────────────────────────────────


class MockUsage:
    def __init__(self, input_tokens: int = 100, output_tokens: int = 50) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockResponse:
    def __init__(self, model: str = "claude-3-sonnet-20240229") -> None:
        self.model = model
        self.usage = MockUsage()
        self.content = []

    def model_dump(self) -> dict[str, Any]:
        return {"model": self.model, "content": []}


class MockMessages:
    def create(self, **kwargs: Any) -> MockResponse:
        return MockResponse()


class MockAsyncMessages:
    async def create(self, **kwargs: Any) -> MockResponse:
        return MockResponse()


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_config() -> Any:
    """Reset global config before and after each test."""
    reset()
    yield
    reset()


@pytest.fixture()
def mock_anthropic_module() -> Any:
    """Install a fake anthropic package into sys.modules and clean up after."""
    # Save originals on the mock classes so we can restore after test
    original_sync_create = MockMessages.create
    original_async_create = MockAsyncMessages.create

    # Save any real anthropic modules
    saved_modules: dict[str, types.ModuleType] = {}
    for key in list(sys.modules):
        if key == "anthropic" or key.startswith("anthropic."):
            saved_modules[key] = sys.modules.pop(key)

    # Also remove cached instrumentor module so it re-imports fresh
    sys.modules.pop("llmtrace.instruments.anthropic", None)

    # Build fake module tree
    mod = types.ModuleType("anthropic")
    resources = types.ModuleType("anthropic.resources")
    resources.Messages = MockMessages  # type: ignore[attr-defined]
    resources.AsyncMessages = MockAsyncMessages  # type: ignore[attr-defined]
    mod.resources = resources  # type: ignore[attr-defined]

    sys.modules["anthropic"] = mod
    sys.modules["anthropic.resources"] = resources

    yield mod

    # Restore class methods (in case test instrumented without uninstrumenting)
    MockMessages.create = original_sync_create  # type: ignore[assignment]
    MockAsyncMessages.create = original_async_create  # type: ignore[assignment]

    # Remove fake modules
    sys.modules.pop("anthropic", None)
    sys.modules.pop("anthropic.resources", None)
    sys.modules.pop("llmtrace.instruments.anthropic", None)

    # Restore real modules
    sys.modules.update(saved_modules)


@pytest.fixture()
def captured_events() -> list[TraceEvent]:
    """Return a list that collects emitted TraceEvents via CallbackSink."""
    events: list[TraceEvent] = []

    def _collect(event: TraceEvent) -> None:
        events.append(event)

    configure(sink=CallbackSink(_collect))
    return events


# ── Tests ────────────────────────────────────────────────────────────────


class TestAnthropicInstrumentor:
    """Tests for AnthropicInstrumentor."""

    def test_import_error_when_sdk_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        # Save and remove all anthropic-related modules
        saved_modules: dict[str, types.ModuleType] = {}
        for key in list(sys.modules):
            if key == "anthropic" or key.startswith("anthropic."):
                saved_modules[key] = sys.modules.pop(key)
        sys.modules.pop("llmtrace.instruments.anthropic", None)

        # Intercept the import to block anthropic
        real_import = builtins.__import__

        def _blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "anthropic" or name.startswith("anthropic."):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocking_import)

        try:
            from llmtrace.instruments.anthropic import AnthropicInstrumentor

            instr = AnthropicInstrumentor()
            with pytest.raises(ImportError, match="pip install llmtrace"):
                instr.instrument()
        finally:
            sys.modules.pop("llmtrace.instruments.anthropic", None)
            sys.modules.update(saved_modules)

    def test_provider_name(self, mock_anthropic_module: Any) -> None:
        from llmtrace.instruments.anthropic import AnthropicInstrumentor

        instr = AnthropicInstrumentor()
        assert instr.provider_name == "anthropic"

    def test_instrument_wraps_create(
        self, mock_anthropic_module: Any, captured_events: list[TraceEvent]
    ) -> None:
        from llmtrace.instruments.anthropic import AnthropicInstrumentor

        original_create = MockMessages.create
        instr = AnthropicInstrumentor()
        instr.instrument()

        assert MockMessages.create is not original_create

        instr.uninstrument()

    def test_uninstrument_restores_original(
        self, mock_anthropic_module: Any, captured_events: list[TraceEvent]
    ) -> None:
        from llmtrace.instruments.anthropic import AnthropicInstrumentor

        original_create = MockMessages.create
        instr = AnthropicInstrumentor()
        instr.instrument()
        instr.uninstrument()

        assert MockMessages.create is original_create

    def test_sync_create_emits_event(
        self, mock_anthropic_module: Any, captured_events: list[TraceEvent]
    ) -> None:
        from llmtrace.instruments.anthropic import AnthropicInstrumentor

        instr = AnthropicInstrumentor()
        instr.instrument()

        messages = MockMessages()
        result = messages.create(model="claude-3-sonnet", max_tokens=100)

        assert isinstance(result, MockResponse)
        assert len(captured_events) == 1
        assert captured_events[0].provider == "anthropic"

        instr.uninstrument()

    @pytest.mark.asyncio()
    async def test_async_create_emits_event(
        self, mock_anthropic_module: Any, captured_events: list[TraceEvent]
    ) -> None:
        from llmtrace.instruments.anthropic import AnthropicInstrumentor

        instr = AnthropicInstrumentor()
        instr.instrument()

        async_messages = MockAsyncMessages()
        result = await async_messages.create(model="claude-3-haiku", max_tokens=50)

        assert isinstance(result, MockResponse)
        assert len(captured_events) == 1
        assert captured_events[0].provider == "anthropic"

        instr.uninstrument()

    def test_create_returns_response(
        self, mock_anthropic_module: Any, captured_events: list[TraceEvent]
    ) -> None:
        from llmtrace.instruments.anthropic import AnthropicInstrumentor

        instr = AnthropicInstrumentor()
        instr.instrument()

        messages = MockMessages()
        result = messages.create(model="claude-3-sonnet", max_tokens=200)

        assert isinstance(result, MockResponse)
        assert result.model == "claude-3-sonnet-20240229"

        instr.uninstrument()

    def test_tracing_failure_does_not_break_llm_call(self, mock_anthropic_module: Any) -> None:
        from llmtrace.instruments.anthropic import AnthropicInstrumentor

        # Configure with a sink whose callback raises
        def _exploding_callback(event: TraceEvent) -> None:
            msg = "sink exploded"
            raise RuntimeError(msg)

        configure(sink=CallbackSink(_exploding_callback))

        instr = AnthropicInstrumentor()
        instr.instrument()

        messages = MockMessages()
        result = messages.create(model="claude-3-sonnet", max_tokens=100)

        # The LLM call should still succeed
        assert isinstance(result, MockResponse)

        instr.uninstrument()

    def test_double_instrument_is_idempotent(
        self, mock_anthropic_module: Any, captured_events: list[TraceEvent]
    ) -> None:
        from llmtrace.instruments.anthropic import AnthropicInstrumentor

        instr = AnthropicInstrumentor()
        instr.instrument()
        instr.instrument()  # second call should be no-op

        messages = MockMessages()
        messages.create(model="claude-3-sonnet", max_tokens=100)

        # Should only get one event, not doubled
        assert len(captured_events) == 1

        instr.uninstrument()
