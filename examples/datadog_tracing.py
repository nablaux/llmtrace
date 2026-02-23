"""Datadog tracing: send traces to Datadog via OTLP intake with tool calls.

Supports both Anthropic and OpenAI — set whichever API key you have.
"""

import asyncio

from _common import create_client, print_raw_traces, print_trace_summary

import llmtrace
from llmtrace.models import TraceEvent
from llmtrace.sinks import CallbackSink, ConsoleSink, MultiSink
from llmtrace.sinks.datadog import DatadogSink

# ── Credentials (set one, or use environment variables) ──────────────────
ANTHROPIC_API_KEY = ""  # e.g. "sk-ant-..."
OPENAI_API_KEY = ""  # e.g. "sk-..."
DD_API_KEY = ""  # Datadog API key
DD_SITE = "us1"  # Datadog site: us1, us3, us5, eu1, ap1, gov
# DD_AGENT_ENDPOINT = "http://localhost:4318"  # Use this for Datadog Agent setups

# ── Collect events for a summary at the end ──────────────────────────────
collected: list[TraceEvent] = []

llmtrace.configure(
    sink=MultiSink(
        [
            ConsoleSink(verbose=True),
            DatadogSink(
                api_key=DD_API_KEY,
                site=DD_SITE,
                # endpoint=DD_AGENT_ENDPOINT,  # uncomment for Agent-based setups
                service_name="llmtrace-datadog-example",
                capture_content=True,
            ),
            CallbackSink(lambda e: collected.append(e)),
        ]
    ),
)


async def main() -> None:
    client, run_agent, provider = create_client(ANTHROPIC_API_KEY, OPENAI_API_KEY)
    llmtrace.instrument(provider)

    answer = await run_agent(client, "What's the weather and time in New York?")
    print(f"\nAgent answer: {answer}")
    print_raw_traces(collected)
    print_trace_summary(collected)


if __name__ == "__main__":
    asyncio.run(main())
