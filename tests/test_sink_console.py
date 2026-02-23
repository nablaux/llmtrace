"""Tests for ConsoleSink."""

import io
from datetime import UTC, datetime
from decimal import Decimal

from llmtrace.models import Cost, ErrorTrace, TokenUsage, TraceEvent
from llmtrace.sinks.console import CYAN, DIM, GREEN, RED, RESET, ConsoleSink


def _make_event(**kwargs: object) -> TraceEvent:
    """Create a TraceEvent with sensible defaults for testing."""
    defaults: dict[str, object] = {
        "provider": "openai",
        "model": "gpt-4",
        "latency_ms": 123.0,
        "timestamp": datetime(2024, 6, 15, 14, 30, 45, tzinfo=UTC),
    }
    defaults.update(kwargs)
    return TraceEvent(**defaults)  # type: ignore[arg-type]


class TestConsoleSinkBasicFormatting:
    """Tests for basic event formatting."""

    async def test_basic_event_output(self) -> None:
        output = io.StringIO()
        sink = ConsoleSink(colorize=False, output=output)
        event = _make_event()

        await sink.write(event)

        line = output.getvalue()
        assert "14:30:45" in line
        assert "openai/gpt-4" in line
        assert "123ms" in line

    async def test_with_token_usage_and_cost(self) -> None:
        output = io.StringIO()
        sink = ConsoleSink(colorize=False, output=output)
        event = _make_event(
            token_usage=TokenUsage(prompt_tokens=500, completion_tokens=200),
            cost=Cost(input_cost=Decimal("0.0025"), output_cost=Decimal("0.0020")),
        )

        await sink.write(event)

        line = output.getvalue()
        assert "500\u2192200 tokens" in line
        assert "$0.0045" in line
        assert "\u2713" in line

    async def test_without_token_usage(self) -> None:
        output = io.StringIO()
        sink = ConsoleSink(colorize=False, output=output)
        event = _make_event(token_usage=None)

        await sink.write(event)

        line = output.getvalue()
        assert "\u2014" in line

    async def test_without_cost(self) -> None:
        output = io.StringIO()
        sink = ConsoleSink(colorize=False, output=output)
        event = _make_event(cost=None)

        await sink.write(event)

        parts = output.getvalue().split("|")
        # cost is the 4th segment (index 3)
        cost_part = parts[3].strip()
        assert cost_part == "\u2014"


class TestConsoleSinkErrorEvent:
    """Tests for error event formatting."""

    async def test_error_event_shows_cross_and_type(self) -> None:
        output = io.StringIO()
        sink = ConsoleSink(colorize=False, output=output)
        event = _make_event(
            error=ErrorTrace(error_type="RateLimitError", message="Too many requests"),
        )

        await sink.write(event)

        line = output.getvalue()
        assert "\u2717" in line
        assert "RateLimitError" in line


class TestConsoleSinkVerboseMode:
    """Tests for verbose output."""

    async def test_verbose_shows_request_preview(self) -> None:
        output = io.StringIO()
        sink = ConsoleSink(colorize=False, output=output, verbose=True)
        event = _make_event(
            request={"messages": [{"role": "user", "content": "Hello world"}]},
            response={"choices": [{"message": {"content": "Hi"}}]},
        )

        await sink.write(event)

        lines = output.getvalue().splitlines()
        assert len(lines) == 2
        assert "req=" in lines[1]
        assert "res=" in lines[1]
        assert "Hello world" in lines[1]


class TestConsoleSinkColorize:
    """Tests for colorization behavior."""

    async def test_colorize_false_no_ansi_codes(self) -> None:
        output = io.StringIO()
        sink = ConsoleSink(colorize=False, output=output)
        event = _make_event()

        await sink.write(event)

        line = output.getvalue()
        assert "\033[" not in line

    async def test_colorize_auto_disabled_for_non_tty(self) -> None:
        output = io.StringIO()
        # StringIO has no isatty or returns False — colorize should auto-disable
        sink = ConsoleSink(colorize=True, output=output)
        event = _make_event()

        await sink.write(event)

        line = output.getvalue()
        assert "\033[" not in line

    async def test_colorize_with_tty_output(self) -> None:
        output = io.StringIO()
        # Simulate TTY
        output.isatty = lambda: True  # type: ignore[assignment]
        sink = ConsoleSink(colorize=True, output=output)
        event = _make_event()

        await sink.write(event)

        line = output.getvalue()
        assert CYAN in line
        assert GREEN in line
        assert RESET in line

    async def test_colorize_error_uses_red(self) -> None:
        output = io.StringIO()
        output.isatty = lambda: True  # type: ignore[assignment]
        sink = ConsoleSink(colorize=True, output=output)
        event = _make_event(
            error=ErrorTrace(error_type="APIError", message="Server error"),
        )

        await sink.write(event)

        line = output.getvalue()
        assert RED in line

    async def test_colorize_verbose_uses_dim(self) -> None:
        output = io.StringIO()
        output.isatty = lambda: True  # type: ignore[assignment]
        sink = ConsoleSink(colorize=True, output=output, verbose=True)
        event = _make_event()

        await sink.write(event)

        text = output.getvalue()
        assert DIM in text


class TestConsoleSinkFlushAndClose:
    """Tests for flush and close behavior."""

    async def test_flush_calls_stream_flush(self) -> None:
        output = io.StringIO()
        flushed = False
        original_flush = output.flush

        def tracking_flush() -> None:
            nonlocal flushed
            flushed = True
            original_flush()

        output.flush = tracking_flush  # type: ignore[assignment]
        sink = ConsoleSink(output=output)

        await sink.flush()

        assert flushed

    async def test_close_is_noop(self) -> None:
        output = io.StringIO()
        sink = ConsoleSink(output=output)
        # Should not raise
        await sink.close()

    async def test_async_context_manager(self) -> None:
        output = io.StringIO()
        async with ConsoleSink(colorize=False, output=output) as sink:
            event = _make_event()
            await sink.write(event)

        assert "openai/gpt-4" in output.getvalue()
