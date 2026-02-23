"""Console sink that prints formatted trace event summaries to stderr."""

import sys
from typing import TextIO

from llmtrace.models import TraceEvent
from llmtrace.sinks.base import BaseSink

RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
DIM = "\033[2m"
YELLOW = "\033[33m"


class ConsoleSink(BaseSink):
    """Sink that prints one-line trace summaries to a text stream.

    Formats each event as a human-readable line with timestamp, provider/model,
    latency, token counts, cost, and success/error status.
    """

    def __init__(
        self,
        *,
        colorize: bool = True,
        output: TextIO | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the console sink.

        Args:
            colorize: Use ANSI color codes. Auto-disabled if output is not a TTY.
            output: Output stream. Defaults to sys.stderr.
            verbose: Show truncated request/response preview on a second line.
        """
        self._output = output or sys.stderr
        self._colorize = colorize and hasattr(self._output, "isatty") and self._output.isatty()
        self._verbose = verbose

    async def write(self, event: TraceEvent) -> None:
        """Format and print a one-line trace summary.

        Format: [HH:MM:SS] provider/model | 123ms | 500→200 tokens | $0.0045 | ✓
        """
        timestamp = event.timestamp.strftime("%H:%M:%S")
        latency = f"{event.latency_ms:.0f}ms"
        provider_model = f"{event.provider}/{event.model}"

        if event.token_usage is not None:
            tokens = f"{event.token_usage.prompt_tokens}\u2192{event.token_usage.completion_tokens} tokens"
        else:
            tokens = "\u2014"

        cost = f"${event.cost.total_cost:.4f}" if event.cost is not None else "\u2014"

        status = f"\u2717 {event.error.error_type}" if event.error is not None else "\u2713"

        if self._colorize:
            provider_model = f"{CYAN}{provider_model}{RESET}"
            if event.error is not None:
                status = f"{RED}{status}{RESET}"
            else:
                status = f"{GREEN}{status}{RESET}"

        line = f"[{timestamp}] {provider_model} | {latency} | {tokens} | {cost} | {status}"
        self._output.write(line + "\n")

        if self._verbose:
            request_preview = str(event.request)[:100]
            response_preview = str(event.response)[:100]
            detail = f"  req={request_preview} res={response_preview}"
            if self._colorize:
                detail = f"{DIM}{detail}{RESET}"
            self._output.write(detail + "\n")

    async def flush(self) -> None:
        """Flush the underlying output stream."""
        self._output.flush()

    async def close(self) -> None:
        """No-op — we don't own the output stream."""
