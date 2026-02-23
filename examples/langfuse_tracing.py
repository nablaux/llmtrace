"""Langfuse tracing: send traces to Langfuse with tool calls.

Supports both Anthropic and OpenAI — set whichever API key you have.
"""

import asyncio

from _common import create_client, print_raw_traces, print_trace_summary

import llmtrace
from llmtrace.models import TraceEvent
from llmtrace.sinks import CallbackSink, ConsoleSink, MultiSink
from llmtrace.sinks.langfuse import LangfuseSink

# ── Credentials (set one, or use environment variables) ──────────────────
ANTHROPIC_API_KEY = ""  # e.g. "sk-ant-..."
OPENAI_API_KEY = ""  # e.g. "sk-..."
LANGFUSE_PUBLIC_KEY = ""  # e.g. "pk-lf-..."
LANGFUSE_SECRET_KEY = ""  # e.g. "sk-lf-..."
LANGFUSE_HOST = "https://cloud.langfuse.com"  # EU cloud (default)
# LANGFUSE_HOST = "https://us.cloud.langfuse.com"  # US cloud

# ── Collect events for a summary at the end ──────────────────────────────
collected: list[TraceEvent] = []

llmtrace.configure(
    sink=MultiSink(
        [
            ConsoleSink(verbose=True),
            LangfuseSink(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST,
                service_name="llmtrace-langfuse-example",
                capture_content=True,
            ),
            CallbackSink(lambda e: collected.append(e)),
        ]
    ),
)


async def main() -> None:
    client, run_agent, provider = create_client(ANTHROPIC_API_KEY, OPENAI_API_KEY)
    llmtrace.instrument(provider)

    answer = await run_agent(client, "What's the weather and time in London?")
    print(f"\nAgent answer: {answer}")
    print_raw_traces(collected)
    print_trace_summary(collected)


if __name__ == "__main__":
    asyncio.run(main())
