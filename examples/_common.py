"""Shared utilities for llmtrace examples.

Provides tool implementations, agent loops for both Anthropic and OpenAI,
provider detection, and display helpers.
"""

import json
import os
from decimal import Decimal
from typing import Any

import llmtrace
from llmtrace.models import TraceEvent

# ── Tool schemas ─────────────────────────────────────────────────────────

ANTHROPIC_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
    {
        "name": "get_local_time",
        "description": "Get the current local time for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
]

OPENAI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_local_time",
            "description": "Get the current local time for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    },
]

# ── Tool implementations ─────────────────────────────────────────────────


@llmtrace.trace_tool(tags={"category": "weather"})
def get_weather(city: str) -> str:
    """Simulate fetching weather data."""
    return json.dumps({"city": city, "temp_c": 22, "condition": "sunny"})


@llmtrace.trace_tool(tags={"category": "time"})
def get_local_time(city: str) -> str:
    """Simulate fetching local time."""
    return json.dumps({"city": city, "time": "14:30", "timezone": "CET"})


TOOL_DISPATCH: dict[str, Any] = {
    "get_weather": get_weather,
    "get_local_time": get_local_time,
}

# ── Agent loops ──────────────────────────────────────────────────────────


@llmtrace.trace(provider="agent")
async def run_agent_anthropic(client: Any, query: str) -> str:
    """Run an Anthropic agent turn: LLM -> tools -> LLM synthesis."""
    messages: list[dict] = [{"role": "user", "content": query}]

    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=ANTHROPIC_TOOLS,
        messages=messages,
    )

    if response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if block.type == "tool_use":
                result = TOOL_DISPATCH[block.name](**block.input)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )

        messages.append({"role": "user", "content": tool_results})

        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=ANTHROPIC_TOOLS,
            messages=messages,
        )

    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return str(response.content)


@llmtrace.trace(provider="agent")
async def run_agent_openai(client: Any, query: str) -> str:
    """Run an OpenAI agent turn: LLM -> tools -> LLM synthesis."""
    messages: list[dict] = [{"role": "user", "content": query}]

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        tools=OPENAI_TOOLS,
        messages=messages,
    )

    message = response.choices[0].message
    if message.tool_calls:
        messages.append(message)
        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments)
            result = TOOL_DISPATCH[tc.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1024,
            tools=OPENAI_TOOLS,
            messages=messages,
        )
        message = response.choices[0].message

    return message.content or str(message)


# ── Provider detection ───────────────────────────────────────────────────


def create_client(anthropic_key: str = "", openai_key: str = "") -> tuple[Any, Any, str]:
    """Create a provider client based on which API key is available.

    Checks explicit keys first, then falls back to environment variables.

    Returns:
        (client, run_agent_fn, provider_name) tuple.
    """
    if anthropic_key:
        import anthropic

        return anthropic.AsyncAnthropic(api_key=anthropic_key), run_agent_anthropic, "anthropic"

    if openai_key:
        import openai

        return openai.AsyncOpenAI(api_key=openai_key), run_agent_openai, "openai"

    # Fall back to environment variables
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic

        return anthropic.AsyncAnthropic(), run_agent_anthropic, "anthropic"

    if os.environ.get("OPENAI_API_KEY"):
        import openai

        return openai.AsyncOpenAI(), run_agent_openai, "openai"

    raise ValueError(
        "No API key provided. Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
        "environment variable, or pass anthropic_key/openai_key directly."
    )


# ── Display helpers ──────────────────────────────────────────────────────


def print_raw_traces(events: list[TraceEvent]) -> None:
    """Pretty-print raw trace events as JSON."""
    print("\n── Raw Trace Events ──────────────────────────────────────────")
    for ev in events:
        print(json.dumps(ev.to_dict(), indent=2))
    print("──────────────────────────────────────────────────────────────")


def print_trace_summary(events: list[TraceEvent]) -> None:
    """Print a formatted trace summary table."""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  Trace Summary                                             ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    total_latency = 0.0
    total_tokens = 0
    total_cost = Decimal("0")

    for i, ev in enumerate(events, 1):
        print(f"║  {i}. {ev.provider}/{ev.model}")
        print(f"║     Latency: {ev.latency_ms:,.0f}ms")
        if ev.token_usage:
            toks = (
                f"{ev.token_usage.prompt_tokens:,}→"
                f"{ev.token_usage.completion_tokens:,} "
                f"({ev.token_usage.total_tokens:,} total)"
            )
            print(f"║     Tokens:  {toks}")
            total_tokens += ev.token_usage.total_tokens
        if ev.cost:
            print(f"║     Cost:    ${ev.cost.total_cost:.6f}")
            total_cost += ev.cost.total_cost
        if ev.tool_calls:
            names = ", ".join(tc.tool_name for tc in ev.tool_calls)
            print(f"║     Tools:   {names}")
        if ev.tags:
            print(f"║     Tags:    {ev.tags}")
        if ev.error:
            print(f"║     Error:   {ev.error.error_type}: {ev.error.message}")
        total_latency += ev.latency_ms
        print("║")

    print("╠══════════════════════════════════════════════════════════════╣")
    cost_str = f"${total_cost:.6f}" if total_cost else "—"
    tok_str = f"{total_tokens:,}" if total_tokens else "—"
    print(f"║  Totals: {total_latency:,.0f}ms | {tok_str} tokens | {cost_str}")
    print("╚══════════════════════════════════════════════════════════════╝")
