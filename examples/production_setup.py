"""Production setup: multi-sink, PII redaction, cost alerts, enrichment.

Demonstrates ConsoleSink, JsonFileSink, and CallbackSink together via
MultiSink, with a real LLM call so you can see actual trace output.

Supports both Anthropic and OpenAI — set whichever API key you have.
"""

import asyncio
import json
import logging
from decimal import Decimal
from pathlib import Path

from _common import create_client, print_raw_traces, print_trace_summary

import llmtrace
from llmtrace.models import TraceEvent
from llmtrace.sinks import CallbackSink, ConsoleSink, JsonFileSink, MultiSink
from llmtrace.transform.enrichment import (
    CostAlertEnricher,
    LatencyClassifierEnricher,
    RedactionStrategy,
    RedactPIIEnricher,
)

# ── Credentials (set one, or use environment variables) ──────────────────
ANTHROPIC_API_KEY = ""  # e.g. "sk-ant-..."
OPENAI_API_KEY = ""  # e.g. "sk-..."

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Callback sink: custom handler for every trace event ──────────────────
collected: list[TraceEvent] = []


async def on_trace_event(event: TraceEvent) -> None:
    """Example async callback — collect events for later analysis."""
    collected.append(event)
    logger.info(
        "CallbackSink received: provider=%s model=%s latency=%.0fms",
        event.provider,
        event.model,
        event.latency_ms,
    )


# ── Sinks: fan-out to Console + JSON file + Callback ─────────────────────
TRACE_DIR = Path("traces")

sink = MultiSink(
    [
        ConsoleSink(colorize=True),
        JsonFileSink(TRACE_DIR / "llm.jsonl", buffer_size=1),
        CallbackSink(on_trace_event),
    ]
)

# ── Enrichers: transform events before they reach sinks ──────────────────
enrichers = [
    RedactPIIEnricher(
        locales=("global", "intl", "eu"),
        strategy=RedactionStrategy.HASH,
    ),
    CostAlertEnricher(threshold_usd=Decimal("0.50")),
    LatencyClassifierEnricher(),
]

# ── Configure ────────────────────────────────────────────────────────────
llmtrace.configure(
    sink=sink,
    enrichers=enrichers,
    default_tags={"service": "example-app", "env": "development"},
)


@llmtrace.trace(provider="app")
async def main() -> None:
    client, _, provider = create_client(ANTHROPIC_API_KEY, OPENAI_API_KEY)
    llmtrace.instrument(provider)

    prompt = (
        "Summarize this customer request: "
        "'Hi, I'm John. My email is john@example.com and my "
        "phone is +1-555-867-5309. Please update my billing address.'"
    )

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

    await sink.flush()

    print(f"\nLLM response: {text}")
    print_raw_traces(collected)
    print_trace_summary(collected)

    # ── Show JSONL file output ───────────────────────────────────────────
    jsonl_path = TRACE_DIR / "llm.jsonl"
    if jsonl_path.exists():
        print(f"\nJsonFileSink output → {jsonl_path}")
        for line in jsonl_path.read_text().strip().splitlines():
            data = json.loads(line)
            print(f"  trace_id={data['trace_id'][:8]}... provider={data['provider']} model={data['model']}")
            req = json.dumps(data.get("request", {}))
            if "SHA:" in req or "REDACTED" in req:
                print(f"  PII redacted: ...{req[:120]}...")

    await sink.close()


if __name__ == "__main__":
    asyncio.run(main())
