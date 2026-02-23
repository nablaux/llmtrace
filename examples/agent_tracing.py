"""Agent tracing: automatic parent-child hierarchy via @trace and @trace_tool.

Supports both Anthropic and OpenAI — set whichever API key you have.
"""

import asyncio

from _common import create_client, print_raw_traces, print_trace_summary

import llmtrace
from llmtrace.models import TraceEvent
from llmtrace.sinks import CallbackSink, ConsoleSink, MultiSink

# ── Credentials (set one, or use environment variables) ──────────────────
ANTHROPIC_API_KEY = ""  # e.g. "sk-ant-..."
OPENAI_API_KEY = ""  # e.g. "sk-..."

# ── Collect events for a summary at the end ──────────────────────────────
collected: list[TraceEvent] = []

llmtrace.configure(
    sink=MultiSink([ConsoleSink(verbose=True), CallbackSink(lambda e: collected.append(e))]),
)


async def main() -> None:
    client, run_agent, provider = create_client(ANTHROPIC_API_KEY, OPENAI_API_KEY)
    llmtrace.instrument(provider)

    answer = await run_agent(client, "What's the weather and time in Paris and Tokyo?")
    print(f"\nAgent answer: {answer}")
    print_raw_traces(collected)
    print_trace_summary(collected)


if __name__ == "__main__":
    asyncio.run(main())
