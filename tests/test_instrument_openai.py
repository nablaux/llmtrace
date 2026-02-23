"""Tests for OpenAIInstrumentor using mock SDK classes."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from llmtrace.config import configure, reset
from llmtrace.models import TraceEvent
from llmtrace.sinks.callback import CallbackSink

# ── Mock OpenAI SDK structure ────────────────────────────────────────────


class MockUsage:
    def __init__(
        self,
        prompt_tokens: int = 50,
        completion_tokens: int = 100,
        total_tokens: int = 150,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class MockMessage:
    def __init__(self) -> None:
        self.role = "assistant"
        self.content = "Hello!"
        self.tool_calls = None


class MockChoice:
    def __init__(self) -> None:
        self.index = 0
        self.message = MockMessage()
        self.finish_reason = "stop"


class MockResponse:
    def __init__(self, model: str = "gpt-4") -> None:
        self.model = model
        self.usage = MockUsage()
        self.choices = [MockChoice()]

    def model_dump(self) -> dict[str, Any]:
        return {"model": self.model, "choices": []}


class MockCompletions:
    def create(self, **kwargs: Any) -> MockResponse:
        return MockResponse()


class MockAsyncCompletions:
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
def mock_openai_module() -> Any:
    """Install a fake openai package into sys.modules and clean up after."""
    original_sync_create = MockCompletions.create
    original_async_create = MockAsyncCompletions.create

    saved_modules: dict[str, types.ModuleType] = {}
    for key in list(sys.modules):
        if key == "openai" or key.startswith("openai."):
            saved_modules[key] = sys.modules.pop(key)

    sys.modules.pop("llmtrace.instruments.openai", None)

    # Build fake module tree: openai.resources.chat.completions
    mod = types.ModuleType("openai")
    resources = types.ModuleType("openai.resources")
    chat = types.ModuleType("openai.resources.chat")
    completions = types.ModuleType("openai.resources.chat.completions")
    completions.Completions = MockCompletions  # type: ignore[attr-defined]
    completions.AsyncCompletions = MockAsyncCompletions  # type: ignore[attr-defined]
    chat.completions = completions  # type: ignore[attr-defined]
    resources.chat = chat  # type: ignore[attr-defined]
    mod.resources = resources  # type: ignore[attr-defined]

    sys.modules["openai"] = mod
    sys.modules["openai.resources"] = resources
    sys.modules["openai.resources.chat"] = chat
    sys.modules["openai.resources.chat.completions"] = completions

    yield mod

    MockCompletions.create = original_sync_create  # type: ignore[assignment]
    MockAsyncCompletions.create = original_async_create  # type: ignore[assignment]

    for key in [
        "openai",
        "openai.resources",
        "openai.resources.chat",
        "openai.resources.chat.completions",
    ]:
        sys.modules.pop(key, None)
    sys.modules.pop("llmtrace.instruments.openai", None)

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


class TestOpenAIInstrumentor:
    """Tests for OpenAIInstrumentor."""

    def test_import_error_when_sdk_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        saved_modules: dict[str, types.ModuleType] = {}
        for key in list(sys.modules):
            if key == "openai" or key.startswith("openai."):
                saved_modules[key] = sys.modules.pop(key)
        sys.modules.pop("llmtrace.instruments.openai", None)

        real_import = builtins.__import__

        def _blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "openai" or name.startswith("openai."):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocking_import)

        try:
            from llmtrace.instruments.openai import OpenAIInstrumentor

            instr = OpenAIInstrumentor()
            with pytest.raises(ImportError, match="pip install llmtrace"):
                instr.instrument()
        finally:
            sys.modules.pop("llmtrace.instruments.openai", None)
            sys.modules.update(saved_modules)

    def test_provider_name(self, mock_openai_module: Any) -> None:
        from llmtrace.instruments.openai import OpenAIInstrumentor

        instr = OpenAIInstrumentor()
        assert instr.provider_name == "openai"

    def test_instrument_wraps_create(
        self, mock_openai_module: Any, captured_events: list[TraceEvent]
    ) -> None:
        from llmtrace.instruments.openai import OpenAIInstrumentor

        original_create = MockCompletions.create
        instr = OpenAIInstrumentor()
        instr.instrument()

        assert MockCompletions.create is not original_create

        instr.uninstrument()

    def test_uninstrument_restores_original(
        self, mock_openai_module: Any, captured_events: list[TraceEvent]
    ) -> None:
        from llmtrace.instruments.openai import OpenAIInstrumentor

        original_create = MockCompletions.create
        instr = OpenAIInstrumentor()
        instr.instrument()
        instr.uninstrument()

        assert MockCompletions.create is original_create

    def test_sync_create_emits_event(
        self, mock_openai_module: Any, captured_events: list[TraceEvent]
    ) -> None:
        from llmtrace.instruments.openai import OpenAIInstrumentor

        instr = OpenAIInstrumentor()
        instr.instrument()

        completions = MockCompletions()
        result = completions.create(model="gpt-4", messages=[{"role": "user", "content": "hi"}])

        assert isinstance(result, MockResponse)
        assert len(captured_events) == 1
        assert captured_events[0].provider == "openai"

        instr.uninstrument()

    @pytest.mark.asyncio()
    async def test_async_create_emits_event(
        self, mock_openai_module: Any, captured_events: list[TraceEvent]
    ) -> None:
        from llmtrace.instruments.openai import OpenAIInstrumentor

        instr = OpenAIInstrumentor()
        instr.instrument()

        async_completions = MockAsyncCompletions()
        result = await async_completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "hi"}]
        )

        assert isinstance(result, MockResponse)
        assert len(captured_events) == 1
        assert captured_events[0].provider == "openai"

        instr.uninstrument()

    def test_create_returns_response(
        self, mock_openai_module: Any, captured_events: list[TraceEvent]
    ) -> None:
        from llmtrace.instruments.openai import OpenAIInstrumentor

        instr = OpenAIInstrumentor()
        instr.instrument()

        completions = MockCompletions()
        result = completions.create(model="gpt-4", messages=[])

        assert isinstance(result, MockResponse)
        assert result.model == "gpt-4"

        instr.uninstrument()

    def test_tracing_failure_does_not_break_llm_call(self, mock_openai_module: Any) -> None:
        from llmtrace.instruments.openai import OpenAIInstrumentor

        def _exploding_callback(event: TraceEvent) -> None:
            msg = "sink exploded"
            raise RuntimeError(msg)

        configure(sink=CallbackSink(_exploding_callback))

        instr = OpenAIInstrumentor()
        instr.instrument()

        completions = MockCompletions()
        result = completions.create(model="gpt-4", messages=[])

        assert isinstance(result, MockResponse)

        instr.uninstrument()

    def test_double_instrument_is_idempotent(
        self, mock_openai_module: Any, captured_events: list[TraceEvent]
    ) -> None:
        from llmtrace.instruments.openai import OpenAIInstrumentor

        instr = OpenAIInstrumentor()
        instr.instrument()
        instr.instrument()  # second call should be no-op

        completions = MockCompletions()
        completions.create(model="gpt-4", messages=[])

        assert len(captured_events) == 1

        instr.uninstrument()
