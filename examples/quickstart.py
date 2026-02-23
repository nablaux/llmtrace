"""Quickstart: trace a real LLM call in 3 lines of config.

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

# 1. Configure — console for live output + callback to collect events
llmtrace.configure(
    sink=MultiSink([ConsoleSink(), CallbackSink(lambda e: collected.append(e))]),
)


async def main() -> None:
    # 2. Detect provider and create client
    client, _, provider = create_client(ANTHROPIC_API_KEY, OPENAI_API_KEY)

    # 3. Instrument the detected provider
    llmtrace.instrument(provider)

    prompt = "What is the capital of France? One sentence."

    if provider == "anthropic":
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
    else:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content

    print(f"\nResponse: {text}")
    print_raw_traces(collected)
    print_trace_summary(collected)


if __name__ == "__main__":
    asyncio.run(main())
