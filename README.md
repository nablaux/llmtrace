# llmtrace

Lightweight structured tracing for LLM applications. Zero required dependencies beyond Pydantic. No backend needed.

## Why llmtrace?

- **3 lines to start** — Configure, instrument, done. Traces flow immediately.
- **No backend required** — Traces go to console, file, webhook, OTLP, Langfuse, Datadog, or anywhere.
- **Typed everything** — Pydantic v2 models, full `mypy --strict`, IDE autocomplete on every field.
- **Zero-dep core** — Only Pydantic required at runtime. Provider SDKs and sink dependencies are optional extras.

## Installation

Requires **Python 3.11+**.

```bash
pip install llmtrace                    # core only (pydantic)
pip install llmtrace[anthropic]         # + Anthropic SDK
pip install llmtrace[openai]            # + OpenAI SDK
pip install llmtrace[webhook]           # + WebhookSink (httpx)
pip install llmtrace[otlp]             # + OTLP export (OpenTelemetry)
pip install llmtrace[otlp-grpc]        # + gRPC OTLP export
pip install llmtrace[langfuse]         # + Langfuse (uses OTLP)
pip install llmtrace[datadog]          # + Datadog (uses OTLP)
pip install llmtrace[presidio]         # + NLP-based PII detection
pip install llmtrace[all]              # everything
```

Combine extras as needed:

```bash
pip install llmtrace[anthropic,openai,webhook]
pip install llmtrace[anthropic,langfuse]
pip install llmtrace[openai,datadog,presidio]
```

## Quick Start

```python
import llmtrace

llmtrace.configure(sink="console")
llmtrace.instrument("anthropic")
```

That's it. Every Anthropic API call is now traced:

```python
import anthropic

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=200,
    messages=[{"role": "user", "content": "Explain monads in one sentence."}],
)
```

Console output:

```
[14:32:01] anthropic/claude-sonnet-4-20250514 | 1,243ms | 500→200 tokens | $0.0045 | ✓
```

## Features

| Feature | Description |
|---|---|
| Auto-instrumentation | Wrap provider SDKs with one call — no code changes to your LLM calls |
| Tool tracing | `@trace_tool` decorator captures tool execution as part of the trace tree |
| Cost tracking | Auto-computed per-call cost from built-in pricing registry |
| Typed trace events | Every field is a Pydantic v2 model with full validation |
| Pluggable sinks | Console, JSONL file, webhook, OTLP, Langfuse, Datadog, callback, or compose with MultiSink |
| Key redaction | API keys, auth headers, and secrets stripped from traces by default |
| PII redaction | Locale-aware pattern detection + optional Presidio NLP engine |
| OpenTelemetry-compatible IDs | `trace_id`, `span_id`, `parent_id` follow OTel conventions for correct span trees |
| Span-based tracing | Group related LLM calls with `span()` context manager |
| Error taxonomy | Normalizes provider errors into categories: rate_limit, timeout, auth, etc. |
| Throughput metrics | Auto-computed tokens/sec in trace metadata |

## The `@trace` Decorator

Trace any function that makes LLM calls. Works on both sync and async:

```python
import llmtrace

@llmtrace.trace(provider="anthropic", tags={"team": "search"})
async def summarize(text: str) -> str:
    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": f"Summarize: {text}"}],
    )
    return response.content[0].text
```

Nested `@trace` calls automatically form a parent-child hierarchy:

```python
@llmtrace.trace()
async def inner() -> str:
    return await client.messages.create(...)

@llmtrace.trace()
async def outer() -> str:
    await inner()  # inner's parent_id → outer's span_id
    return await client.messages.create(...)
```

## The `@trace_tool` Decorator

Trace tool/function executions that are called by your agent. Captures arguments, return values, latency, and errors:

```python
import json
import llmtrace

@llmtrace.trace_tool(tags={"category": "weather"})
def get_weather(city: str) -> str:
    """Fetch current weather for a city."""
    return json.dumps({"city": city, "temp_c": 22, "condition": "sunny"})

@llmtrace.trace_tool(name="search_docs", tags={"category": "retrieval"})
async def search(query: str, limit: int = 10) -> list[str]:
    """Search the document index."""
    return await my_index.query(query, top_k=limit)
```

When called inside a `@trace`-decorated agent function, tool events automatically get the correct `trace_id` and `parent_id`:

```python
@llmtrace.trace(provider="agent")
async def run_agent(query: str) -> str:
    response = await client.messages.create(...)  # LLM call, parent_id → agent's span_id
    result = get_weather("Paris")                 # tool call, parent_id → agent's span_id
    ...
```

You can also wrap a dict of tool functions in bulk:

```python
tools = {"get_weather": get_weather_fn, "search": search_fn}
wrapped = llmtrace.instrument_tools(tools, tags={"source": "agent"})
```

## Trace ID Conventions

llmtrace follows OpenTelemetry conventions for trace correlation:

| Field | Description |
|---|---|
| `trace_id` | Shared by all events in the same trace. Created by the root span. |
| `span_id` | Unique per event. Every `TraceEvent` gets its own `span_id`. |
| `parent_id` | References the parent's `span_id`. `None` for root spans. |

Example trace tree from an agent with tool use:

```
@trace("agent")              trace_id=T, span_id=A, parent_id=null
  ├── LLM call               trace_id=T, span_id=B, parent_id=A
  ├── tool: get_weather       trace_id=T, span_id=C, parent_id=A
  └── LLM call               trace_id=T, span_id=D, parent_id=A
```

This means traces export correctly to OTLP, Langfuse, and Datadog without broken span trees.

## Sinks

### ConsoleSink

Pretty-prints one-line summaries to stderr with ANSI colors.

```python
from llmtrace.sinks import ConsoleSink

sink = ConsoleSink(colorize=True, verbose=False)
```

### JsonFileSink

Writes JSONL with optional size-based rotation.

```python
from llmtrace.sinks import JsonFileSink

sink = JsonFileSink("traces.jsonl", rotate_mb=50, rotate_count=5)
```

### WebhookSink

Batched HTTP POST with retry and exponential backoff.

```python
from llmtrace.sinks import WebhookSink

sink = WebhookSink(
    "https://my-endpoint.example.com/traces",
    headers={"Authorization": "Bearer tok_xxx"},
    batch_size=50,
)
```

### MultiSink

Fan-out to multiple sinks simultaneously.

```python
from llmtrace.sinks import ConsoleSink, JsonFileSink, MultiSink

sink = MultiSink([ConsoleSink(), JsonFileSink("traces.jsonl")])
```

### CallbackSink

Call your own function for each event.

```python
from llmtrace.sinks import CallbackSink

sink = CallbackSink(lambda event: my_database.insert(event.to_dict()))
```

### OTLPSink

Export traces as OpenTelemetry spans to any OTLP-compliant backend.

```python
from llmtrace.sinks import OTLPSink

sink = OTLPSink(
    endpoint="http://localhost:4318",
    service_name="my-app",
    capture_content=False,  # opt-in for request/response bodies
)
```

### LangfuseSink

Pre-configured OTLP export to [Langfuse](https://langfuse.com).

```python
from llmtrace.sinks import LangfuseSink

sink = LangfuseSink(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://cloud.langfuse.com",  # EU cloud (default)
    # host="https://us.cloud.langfuse.com",  # US cloud
)
```

Or via environment variables: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.

### DatadogSink

Pre-configured OTLP export to [Datadog](https://www.datadoghq.com).

```python
from llmtrace.sinks import DatadogSink

# Using a site shortcode
sink = DatadogSink(api_key="dd-...", site="us1")

# Using a custom endpoint (e.g., Datadog Agent or a different region URL)
sink = DatadogSink(
    api_key="dd-...",
    endpoint="http://localhost:4318",  # Datadog Agent
)
```

Supported site shortcodes: `us1`, `us3`, `us5`, `eu1`, `ap1`, `gov`. These resolve to the corresponding `otlp-http-intake.*.datadoghq.com` URLs. Pass `endpoint` directly to use any custom URL.

Or via environment variables: `DD_API_KEY`, `DD_SITE`.

## Provider Support

| Provider | Install extra | Auto-traced methods |
|---|---|---|
| Anthropic | `[anthropic]` | `messages.create` (sync + async) |
| OpenAI | `[openai]` | `chat.completions.create` (sync + async) |
| Google | `[google]` | Coming soon |
| LiteLLM | `[litellm]` | Coming soon |

## Sensitive Key Redaction

By default, llmtrace strips sensitive keys (`api_key`, `authorization`, `token`, `password`, `secret`, `credential`, and variants) from request payloads in traces.

**Default behavior (redaction on):**

```python
llmtrace.configure(sink="console")
# Request payload in traces:
# {"model": "claude-sonnet-4-20250514", "api_key": "[REDACTED]", "messages": [...]}
```

**Disable redaction** if you need full payloads (e.g., local debugging):

```python
llmtrace.configure(sink="console", redact_sensitive_keys=False)
# Request payload in traces:
# {"model": "claude-sonnet-4-20250514", "api_key": "sk-ant-...", "messages": [...]}
```

You can also control whether request and response payloads are captured at all:

```python
llmtrace.configure(
    sink="console",
    capture_request=False,   # traces will have request={}
    capture_response=False,  # traces will have response={}
)
```

## Advanced Usage

### Span-Based Tracing

Group related LLM calls into a span tree:

```python
import llmtrace

async def agent_loop(query: str) -> str:
    async with llmtrace.span("agent_turn", tags={"query_type": "search"}) as turn:
        turn.annotate(user_query=query)

        plan = await plan_step(query)

        async with turn.child("tool_execution") as tool_span:
            tool_span.annotate(tool="web_search")
            results = await search(plan)

        async with turn.child("synthesis") as synth_span:
            answer = await synthesize(query, results)
            synth_span.annotate(answer_length=len(answer))

        return answer
```

### PII Redaction

Three levels of PII protection, from zero-config to NLP-powered:

**1. Automatic key redaction (on by default)** — see [Sensitive Key Redaction](#sensitive-key-redaction) above.

**2. Pattern-based redaction with locale support**

```python
from llmtrace.transform.enrichment import RedactPIIEnricher

enricher = RedactPIIEnricher(locales=("global", "intl", "eu"))
llmtrace.configure(enrichers=[enricher])
```

**3. NLP-based redaction (names, addresses, medical terms)**

```python
enricher = RedactPIIEnricher(use_presidio=True, presidio_language="en")
llmtrace.configure(enrichers=[enricher])
```

**Redaction strategies:**

| Strategy | Example input | Example output |
|---|---|---|
| `REPLACE` (default) | `john@example.com` | `[EMAIL_REDACTED]` |
| `MASK` | `john@example.com` | `j***@*******.c*m` |
| `HASH` | `john@example.com` | `[SHA:a1b2c3d4]` |

```python
from llmtrace.transform.enrichment import RedactPIIEnricher, RedactionStrategy

enricher = RedactPIIEnricher(strategy=RedactionStrategy.HASH)
```

### Cost Tracking

Costs are auto-computed from a built-in pricing registry. Register custom models or override prices:

```python
from decimal import Decimal
from llmtrace.pricing import ModelPricing, PricingRegistry

registry = PricingRegistry()

# Register a custom or new model
registry.register("anthropic", "claude-4-opus", ModelPricing(
    input_per_million=Decimal("15.00"),
    output_per_million=Decimal("75.00"),
))

llmtrace.configure(sink="console", pricing_registry=registry)
```

See the [Known Limitations](#known-limitations) section for details on the built-in pricing data.

### Cost and Latency Enrichers

Flag expensive calls:

```python
from decimal import Decimal
from llmtrace.transform.enrichment import CostAlertEnricher

enricher = CostAlertEnricher(threshold_usd=Decimal("0.50"))
llmtrace.configure(enrichers=[enricher])
# Adds tags={"cost_alert": "high"} when a single call exceeds $0.50
```

Classify latency:

```python
from llmtrace.transform.enrichment import LatencyClassifierEnricher

enricher = LatencyClassifierEnricher(fast_ms=500, normal_ms=2000, slow_ms=5000)
llmtrace.configure(enrichers=[enricher])
# Adds tags={"latency_class": "fast"|"normal"|"slow"|"critical"}
```

### Custom Enrichers

An enricher is any callable that takes a `TraceEvent` and returns a `TraceEvent`:

```python
from llmtrace.models import TraceEvent

class StripLongResponses:
    def __call__(self, event: TraceEvent) -> TraceEvent:
        if len(str(event.response)) > 10_000:
            return event.model_copy(update={"response": {"truncated": True}})
        return event
```

### Environment Variable Configuration

| Variable | Type | Default | Description |
|---|---|---|---|
| `LLMTRACE_SINK` | string | `console` | Sink: `"console"`, `"jsonfile:<path>"`, `"webhook:<url>"`, `"otlp"`, `"otlp:<endpoint>"`, `"langfuse"`, `"datadog"` |
| `LLMTRACE_TAGS` | string | (empty) | Default tags: `"k1=v1,k2=v2"` |
| `LLMTRACE_SAMPLE_RATE` | float | `1.0` | Trace sampling rate (0.0 to 1.0) |
| `LLMTRACE_CAPTURE_REQUEST` | bool | `true` | Include full request payload in traces |
| `LLMTRACE_CAPTURE_RESPONSE` | bool | `true` | Include full response payload in traces |
| `LLMTRACE_REDACT_KEYS` | bool | `true` | Redact sensitive keys (api_key, auth headers) from payloads |
| `LANGFUSE_PUBLIC_KEY` | string | — | Langfuse public key (for `sink="langfuse"`) |
| `LANGFUSE_SECRET_KEY` | string | — | Langfuse secret key (for `sink="langfuse"`) |
| `LANGFUSE_HOST` | string | `https://cloud.langfuse.com` | Langfuse host URL |
| `DD_API_KEY` | string | — | Datadog API key (for `sink="datadog"`) |
| `DD_SITE` | string | `us1` | Datadog site identifier |

## Trace Event Schema

Every trace is a `TraceEvent` Pydantic model:

| Field | Type | Description |
|---|---|---|
| `trace_id` | `UUID` | Shared trace identifier — same for all events in one trace |
| `parent_id` | `UUID \| None` | Parent span's `span_id`. `None` for root spans |
| `span_id` | `UUID` | Unique identifier for this event |
| `timestamp` | `datetime` | UTC timestamp (auto-generated) |
| `provider` | `str` | Provider name: `"anthropic"`, `"openai"`, `"tool"`, etc. |
| `model` | `str` | Model identifier (or tool name for tool events) |
| `request` | `dict` | Request payload (empty if `capture_request=False`) |
| `response` | `dict` | Response payload (empty if `capture_response=False`) |
| `token_usage` | `TokenUsage \| None` | Prompt, completion, total, and cache token counts |
| `cost` | `Cost \| None` | Input, output, and total cost as `Decimal` |
| `latency_ms` | `float` | Wall-clock latency in milliseconds |
| `tool_calls` | `list[ToolCallTrace]` | Tool calls extracted from the LLM response |
| `error` | `ErrorTrace \| None` | Error type, message, retryability, stack trace |
| `tags` | `dict[str, str]` | String key-value tags for filtering |
| `metadata` | `dict[str, Any]` | Arbitrary metadata from enrichers or user code |

Serialize with `event.to_json()` or `event.to_dict()`.

## Examples

The [`examples/`](examples/) directory contains runnable scripts demonstrating common setups:

| Example | Description |
|---|---|
| [`quickstart.py`](examples/quickstart.py) | Minimal setup — configure, instrument, call |
| [`agent_tracing.py`](examples/agent_tracing.py) | Agent loop with `@trace`, `@trace_tool`, and auto parent-child hierarchy |
| [`production_setup.py`](examples/production_setup.py) | MultiSink, PII redaction, cost alerts, latency classification |
| [`otlp_tracing.py`](examples/otlp_tracing.py) | OTLP export to a local collector |
| [`langfuse_tracing.py`](examples/langfuse_tracing.py) | Langfuse integration |
| [`datadog_tracing.py`](examples/datadog_tracing.py) | Datadog integration |

Each example supports both Anthropic and OpenAI — set whichever API key you have available.

## Known Limitations

**Cost tracking uses a hard-coded pricing registry.** The built-in registry covers popular models (Claude Sonnet/Opus/Haiku, GPT-4o/4o-mini, o1, o3-mini, Gemini 2.0 Flash/Pro) but prices may become stale as providers update them. For new or custom models, register pricing manually via `PricingRegistry.register()`. If a model is not in the registry, `cost` will be `None` rather than wrong.

**Error taxonomy is regex-based with a fixed set of categories.** Errors are classified into `rate_limit`, `timeout`, `auth`, `context_length`, `content_filter`, `server`, `invalid_request`, or `unknown` using pattern matching against the error message and type. Provider-specific error codes or nuanced categories are not yet supported — unrecognized errors fall through to `unknown`.

**Google and LiteLLM providers are not yet instrumented.** The extractor and instrumentor skeletons exist, but auto-instrumentation is not wired up. Contributions welcome.

**No streaming support.** Tracing currently captures complete request/response cycles. Streaming responses (SSE, async iterators) are not traced. The wrapper sees the initial call but not streamed chunks.

## Contributing

```bash
git clone https://github.com/yourname/llmtrace && cd llmtrace
uv sync --all-extras
uv run mypy src/llmtrace/ --strict
uv run ruff check src/ tests/
uv run pytest tests/ -v --cov=src/llmtrace --cov-report=term-missing
```

## License

Apache 2.0
