"""End-to-end test: real Anthropic call → OTLP collector + Datadog Agent.

Sends traces to:
  1. Console (live output)
  2. Local OTLP collector → Jaeger (http://localhost:4318)
  3. Local Datadog Agent → Datadog (http://localhost:4319)

The Datadog Agent receives OTLP traces and forwards them to Datadog.
This avoids the direct OTLP intake endpoint which requires preview access.

Usage:
    # Start infrastructure (collector + Jaeger only)
    cd examples && docker compose up -d

    # Or include the Datadog Agent (requires DD_API_KEY)
    DD_API_KEY=your-key docker compose --profile datadog up -d

    # Run the test
    export ANTHROPIC_API_KEY="sk-ant-..."
    uv run python real_test_otlp_datadog.py

    # View traces
    # Jaeger:  http://localhost:16686
    # Datadog: https://app.datadoghq.eu → APM → Traces
"""

import asyncio
import os
import sys

from _common import create_client, print_raw_traces, print_trace_summary

import llmtrace
from llmtrace.models import TraceEvent
from llmtrace.sinks import CallbackSink, ConsoleSink, MultiSink
from llmtrace.sinks.datadog import DatadogSink
from llmtrace.sinks.otlp import OTLPSink

# ── Config from env ──────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get(
    "ANTHROPIC_API_KEY",
    "",
)
DD_API_KEY = os.environ.get("DD_API_KEY", "")
DD_SERVICE = os.environ.get("DD_SERVICE", "llmtrace-test")
DD_ENV = os.environ.get("DD_ENV", "development")

# Datadog Agent OTLP endpoint (port 4319 on host, mapped to 4318 in container)
DD_AGENT_ENDPOINT = os.environ.get("DD_AGENT_ENDPOINT", "http://localhost:4319")

# OTel collector endpoint (Jaeger)
OTLP_ENDPOINT = os.environ.get("OTLP_ENDPOINT", "http://localhost:4318")

# ── Validate ─────────────────────────────────────────────────────────────
if not ANTHROPIC_API_KEY:
    print("ERROR: ANTHROPIC_API_KEY is required")
    sys.exit(1)

if not DD_API_KEY:
    print("INFO: DD_API_KEY not set — Datadog Agent sink will be skipped")
    print("      Start with: DD_API_KEY=your-key docker compose --profile datadog up -d")

# ── Build sinks ──────────────────────────────────────────────────────────
collected: list[TraceEvent] = []

resource_attrs = {
    "deployment.environment": DD_ENV,
    "service.version": llmtrace.__version__,
}

sinks = [
    ConsoleSink(verbose=True),
    OTLPSink(
        endpoint=OTLP_ENDPOINT,
        service_name=DD_SERVICE,
        resource_attributes=resource_attrs,
        capture_content=True,
    ),
    CallbackSink(lambda e: collected.append(e)),
]

if DD_API_KEY:
    # Send to the local Datadog Agent (not the direct intake endpoint).
    # The Agent receives OTLP, enriches with DD metadata, and forwards to Datadog.
    sinks.append(
        DatadogSink(
            api_key=DD_API_KEY,
            endpoint=DD_AGENT_ENDPOINT,
            service_name=DD_SERVICE,
            resource_attributes=resource_attrs,
            capture_content=True,
        )
    )

llmtrace.configure(sink=MultiSink(sinks))


async def main() -> None:
    client, run_agent, provider = create_client(anthropic_key=ANTHROPIC_API_KEY)
    llmtrace.instrument(provider)

    print(f"\nProvider:  {provider}")
    print(f"OTLP:     {OTLP_ENDPOINT}")
    dd_status = f"enabled (agent={DD_AGENT_ENDPOINT})" if DD_API_KEY else "disabled"
    print(f"Datadog:  {dd_status}")
    print(f"Service:  {DD_SERVICE}")
    print(f"Env:      {DD_ENV}")
    print()

    answer = await run_agent(client, "What's the weather and time in New York?")
    print(f"\nAgent answer: {answer}")

    # Flush and close to ensure all spans are exported before exit
    sink = llmtrace.get_config().sink
    await sink.flush()
    await sink.close()

    print_raw_traces(collected)
    print_trace_summary(collected)


if __name__ == "__main__":
    asyncio.run(main())
