"""Benchmark trace emission overhead for sinks, enrichment, and normalizer."""

import asyncio
import tempfile
import time
from decimal import Decimal
from pathlib import Path

from llmtrace.capture.extractors import ExtractedData
from llmtrace.config import LLMTraceConfig
from llmtrace.models import TokenUsage, ToolCallTrace, TraceEvent
from llmtrace.pricing import PricingRegistry
from llmtrace.sinks.callback import CallbackSink
from llmtrace.sinks.console import ConsoleSink
from llmtrace.sinks.jsonfile import JsonFileSink
from llmtrace.transform.enrichment import (
    CostAlertEnricher,
    EnrichmentPipeline,
    LatencyClassifierEnricher,
    RedactPIIEnricher,
)
from llmtrace.transform.normalizer import normalize

WARMUP = 100
ITERATIONS = 1000


def create_realistic_event() -> TraceEvent:
    """Build a realistic TraceEvent with full token usage, cost, tool calls, tags, and metadata."""
    token_usage = TokenUsage(
        prompt_tokens=500,
        completion_tokens=200,
        cache_read_tokens=50,
    )
    registry = PricingRegistry()
    cost = registry.compute_cost("anthropic", "claude-sonnet-4-20250514", token_usage)

    return TraceEvent(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        request={"messages": [{"role": "user", "content": "Hello, world!"}]},
        response={"content": [{"type": "text", "text": "Hi there!"}]},
        token_usage=token_usage,
        cost=cost,
        latency_ms=1500.0,
        tool_calls=[
            ToolCallTrace(
                tool_name="get_weather",
                arguments={"location": "San Francisco", "units": "celsius"},
                result={"temperature": 18, "condition": "foggy"},
            ),
            ToolCallTrace(
                tool_name="search_docs",
                arguments={"query": "deployment guide", "limit": 5},
                result={"count": 3, "results": ["doc1", "doc2", "doc3"]},
            ),
        ],
        tags={
            "env": "production",
            "team": "ml-platform",
            "service": "chat-api",
            "version": "2.1.0",
            "region": "us-west-2",
        },
        metadata={
            "session_id": "sess-abc123",
            "user_tier": "enterprise",
            "experiment": "latency-opt-v3",
        },
    )


async def bench_sink(name: str, sink, event: TraceEvent) -> float:
    """Benchmark a single sink: warmup then measure ITERATIONS writes. Returns per-event µs."""
    for _ in range(WARMUP):
        await sink.write(event)
    await sink.flush()

    start = time.perf_counter()
    for _ in range(ITERATIONS):
        await sink.write(event)
    await sink.flush()
    elapsed = time.perf_counter() - start

    return (elapsed / ITERATIONS) * 1_000_000  # microseconds


async def bench_sinks(event: TraceEvent) -> dict[str, float]:
    """Benchmark all three non-network sinks."""
    results: dict[str, float] = {}

    # ConsoleSink -> /dev/null
    with Path("/dev/null").open("w") as devnull:
        console = ConsoleSink(colorize=False, output=devnull)
        results["ConsoleSink"] = await bench_sink("ConsoleSink", console, event)

    # JsonFileSink -> temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "trace.jsonl")
        json_sink = JsonFileSink(path, buffer_size=1)
        results["JsonFileSink"] = await bench_sink("JsonFileSink", json_sink, event)
        await json_sink.close()

    # CallbackSink -> no-op
    callback_sink = CallbackSink(lambda _event: None)
    results["CallbackSink"] = await bench_sink("CallbackSink", callback_sink, event)

    return results


def bench_enrichment(event: TraceEvent) -> float:
    """Benchmark enrichment pipeline with PII in payload. Returns per-event µs."""
    pii_event = event.model_copy(
        update={
            "request": {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Contact john.doe@example.com or call +14155551234. "
                            "Card: 4111 1111 1111 1111"
                        ),
                    }
                ]
            },
            "response": {
                "content": [
                    {
                        "type": "text",
                        "text": "Sure, I found the email john.doe@example.com in our records.",
                    }
                ]
            },
        }
    )

    pipeline = EnrichmentPipeline(
        [
            RedactPIIEnricher(locales=("global", "intl")),
            CostAlertEnricher(threshold_usd=Decimal("1.00")),
            LatencyClassifierEnricher(),
        ]
    )

    for _ in range(WARMUP):
        pipeline.apply(pii_event)

    start = time.perf_counter()
    for _ in range(ITERATIONS):
        pipeline.apply(pii_event)
    elapsed = time.perf_counter() - start

    return (elapsed / ITERATIONS) * 1_000_000


def bench_normalizer() -> float:
    """Benchmark normalizer with full token usage. Returns per-event µs."""
    token_usage = TokenUsage(
        prompt_tokens=500,
        completion_tokens=200,
        cache_read_tokens=50,
    )
    extracted = ExtractedData(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        request_payload={"messages": [{"role": "user", "content": "Hello"}]},
        response_payload={"content": [{"type": "text", "text": "Hi"}]},
        token_usage=token_usage,
    )
    config = LLMTraceConfig()

    for _ in range(WARMUP):
        normalize(extracted, latency_ms=1500.0, config=config)

    start = time.perf_counter()
    for _ in range(ITERATIONS):
        normalize(extracted, latency_ms=1500.0, config=config)
    elapsed = time.perf_counter() - start

    return (elapsed / ITERATIONS) * 1_000_000


def main() -> None:
    event = create_realistic_event()

    # Sink benchmarks
    sink_results = asyncio.run(bench_sinks(event))

    # Enrichment benchmark
    enrichment_us = bench_enrichment(event)

    # Normalizer benchmark
    normalizer_us = bench_normalizer()

    # Assertions
    SINK_LIMIT = 1000  # µs
    ENRICHMENT_LIMIT = 2000  # µs
    NORMALIZER_LIMIT = 500  # µs

    rows: list[tuple[str, float, float]] = [
        ("ConsoleSink", sink_results["ConsoleSink"], SINK_LIMIT),
        ("JsonFileSink", sink_results["JsonFileSink"], SINK_LIMIT),
        ("CallbackSink", sink_results["CallbackSink"], SINK_LIMIT),
        ("Enrichment (3)", enrichment_us, ENRICHMENT_LIMIT),
        ("Normalizer", normalizer_us, NORMALIZER_LIMIT),
    ]

    # Print summary table
    print()
    print(f"{'Component':<20} | {'Per-event (µs)':>14} | Status")
    print(f"{'-' * 20}-+-{'-' * 14}-+-{'-' * 10}")
    for name, us, limit in rows:
        passed = us < limit
        status = "\u2713 PASS" if passed else "\u2717 FAIL"
        print(f"{name:<20} | {us:>14.1f} | {status}")
    print()

    # Assert all pass
    for name, us, limit in rows:
        assert us < limit, f"{name}: {us:.1f}µs exceeds {limit}µs limit"

    print("All benchmarks passed.")


if __name__ == "__main__":
    main()
