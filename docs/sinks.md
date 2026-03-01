# Sinks

Sinks are async destinations for trace events. Configure one or compose many with `MultiSink`.

## ConsoleSink

Prints one-line summaries to stderr.

```python
from llmtrace.sinks import ConsoleSink

sink = ConsoleSink(colorize=True, verbose=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `colorize` | `bool` | `True` | ANSI colors. Auto-disabled if output is not a TTY |
| `output` | `TextIO` | `sys.stderr` | Output stream |
| `verbose` | `bool` | `False` | Show truncated request/response preview on second line |

Output format:

```
[14:32:01] anthropic/claude-sonnet-4-20250514 | 823ms | 12→45 tokens | $0.000711 | ✓
```

Errors show `✗` with the error type. Verbose mode adds a second line with the first 100 chars of request and response.

## JsonFileSink

Writes events as JSON Lines with optional rotation.

```python
from llmtrace.sinks import JsonFileSink

sink = JsonFileSink(
    "/var/log/llmtrace/events.jsonl",
    rotate_mb=50,
    rotate_count=5,
    buffer_size=10,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| Path` | required | File path for JSONL output |
| `rotate_mb` | `float \| None` | `None` | Rotate when file exceeds this size (MB). `None` disables |
| `rotate_count` | `int` | `5` | Max rotated files to keep |
| `buffer_size` | `int` | `10` | Events buffered before flushing to disk |

Each line is a complete JSON object (`event.model_dump_json()`). Rotation renames `path` → `path.1` → `path.2` etc.

## WebhookSink

POSTs JSON batches to an HTTP endpoint with retry.

```python
from llmtrace.sinks import WebhookSink

sink = WebhookSink(
    "https://ingest.example.com/traces",
    headers={"Authorization": "Bearer tok_xxx"},
    batch_size=50,
    flush_interval_s=10.0,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | required | Endpoint URL |
| `headers` | `dict[str, str] \| None` | `None` | Custom HTTP headers |
| `batch_size` | `int` | `50` | Flush when buffer reaches this count |
| `flush_interval_s` | `float` | `10.0` | Periodic flush interval (seconds) |
| `max_retries` | `int` | `3` | Retries on transient failure |
| `timeout_s` | `float` | `30.0` | HTTP request timeout (seconds) |

**Retry behavior:** Retries on 5xx responses and transport errors with exponential backoff (2s, 4s, 8s). Non-5xx errors are logged and the batch is dropped. Uses `httpx.AsyncClient`.

**Flush triggers:** buffer reaches `batch_size`, or `flush_interval_s` elapses — whichever comes first.

## MultiSink

Fan-out to multiple sinks. Errors in one sink never affect others.

```python
from llmtrace.sinks import ConsoleSink, JsonFileSink, MultiSink

sink = MultiSink([
    ConsoleSink(),
    JsonFileSink("/tmp/traces.jsonl"),
])
```

All operations (`write`, `flush`, `close`) dispatch concurrently via `asyncio.gather`. Exceptions are logged but never propagated.

## CallbackSink

Route events to a custom function. Supports sync and async callbacks.

```python
from llmtrace.sinks import CallbackSink

# Sync callback
def on_event(event):
    print(f"Got trace: {event.provider}/{event.model}")

sink = CallbackSink(on_event)

# Async callback
async def on_event_async(event):
    await db.insert(event.to_dict())

sink = CallbackSink(on_event_async)
```

The callback type (sync/async) is detected at init via `asyncio.iscoroutinefunction`. Exceptions are logged, never propagated.

## OTLPSink

Exports trace events as OpenTelemetry spans via OTLP. Maps `TraceEvent` fields to `gen_ai.*` semantic convention attributes.

`pip install llmtrace[otlp]` (HTTP) or `pip install llmtrace[otlp-grpc]` (+ gRPC support)

```python
from llmtrace.sinks import OTLPSink

sink = OTLPSink(
    endpoint="http://localhost:4318",
    protocol="http/protobuf",
    service_name="my-app",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `endpoint` | `str` | `"http://localhost:4318"` | OTLP collector/backend URL |
| `headers` | `dict[str, str] \| None` | `None` | Custom HTTP headers (e.g., for auth) |
| `protocol` | `Literal["http/protobuf", "grpc"]` | `"http/protobuf"` | Export protocol |
| `service_name` | `str` | `"llmtrace"` | OTel resource `service.name` |
| `resource_attributes` | `dict[str, str] \| None` | `None` | Additional OTel resource attributes |
| `capture_content` | `bool` | `False` | Include request/response bodies in spans (PII risk) |

**Attribute mapping:**

| TraceEvent field | OTel span attribute |
|---|---|
| `provider` | `gen_ai.provider.name` |
| `model` | `gen_ai.request.model` |
| `token_usage.prompt_tokens` | `gen_ai.usage.input_tokens` |
| `token_usage.completion_tokens` | `gen_ai.usage.output_tokens` |
| `cost.total_cost` | `llmtrace.cost.total` |
| `tags` | `llmtrace.tag.{key}` |
| `metadata` | `llmtrace.meta.{key}` |
| `error` | Span status `ERROR` + `error.type` attribute |

Span name: `"chat {model}"`. Span kind: `CLIENT`. Tool calls become child spans with `gen_ai.operation.name = "execute_tool"`.

## LangfuseSink

Pre-configured OTLPSink for [Langfuse](https://langfuse.com). Handles Basic Auth and endpoint construction.

`pip install llmtrace[langfuse]`

```python
from llmtrace.sinks import LangfuseSink

sink = LangfuseSink(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://cloud.langfuse.com",  # EU (default)
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `public_key` | `str` | required | Langfuse public API key |
| `secret_key` | `str` | required | Langfuse secret API key |
| `host` | `str` | `"https://cloud.langfuse.com"` | Langfuse host. Use `"https://us.cloud.langfuse.com"` for US cloud |
| `**kwargs` | | | Passed to `OTLPSink` (e.g., `service_name`, `capture_content`) |

Protocol is forced to `http/protobuf` (Langfuse does not support gRPC).

## DatadogSink

Pre-configured OTLPSink for [Datadog](https://www.datadoghq.com).

`pip install llmtrace[datadog]`

> **Important:** Datadog's direct OTLP traces intake (`otlp-http-intake.*.datadoghq.com`) is in [preview](https://docs.datadoghq.com/opentelemetry/setup/otlp_ingest/) and requires requesting access from your Datadog account team. The recommended approach is to run a [Datadog Agent](https://docs.datadoghq.com/opentelemetry/setup/otlp_ingest_in_the_agent/) that receives OTLP locally and forwards traces to Datadog.

```python
from llmtrace.sinks import DatadogSink

# Recommended: via Datadog Agent running locally
sink = DatadogSink(
    api_key="dd-...",
    endpoint="http://localhost:4318",  # Agent OTLP endpoint
    service_name="my-app",
    resource_attributes={"deployment.environment": "production"},
)

# Direct intake (requires preview access)
sink = DatadogSink(api_key="dd-...", site="eu1")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | required | Datadog API key (set via `DD-API-KEY` header) |
| `site` | `str` | `"us1"` | Datadog site: `us1`, `us3`, `us5`, `eu1`, `ap1`, `gov` |
| `endpoint` | `str \| None` | `None` | Override endpoint. Use this for Datadog Agent setups (e.g., `http://localhost:4318`) |
| `**kwargs` | | | Passed to `OTLPSink` (e.g., `service_name`, `resource_attributes`, `capture_content`) |

**Datadog Agent setup:**

The Datadog Agent receives OTLP traces and forwards them to Datadog. This is the standard pattern in production — the agent runs as a sidecar (Kubernetes, ECS) or on each host (VMs) and handles buffering, retries, and host metadata enrichment.

To enable OTLP ingestion on the agent, set:

```
DD_OTLP_CONFIG_RECEIVER_PROTOCOLS_HTTP_ENDPOINT=0.0.0.0:4318
DD_APM_ENABLED=true
```

See [`examples/docker-compose.yaml`](../examples/docker-compose.yaml) for a ready-to-use local setup with the Datadog Agent, an OTLP collector, and Jaeger UI.

**Supported sites (direct intake):**

| Site | Endpoint |
|---|---|
| `us1` | `https://otlp-http-intake.datadoghq.com` |
| `us3` | `https://otlp-http-intake.us3.datadoghq.com` |
| `us5` | `https://otlp-http-intake.us5.datadoghq.com` |
| `eu1` | `https://otlp-http-intake.datadoghq.eu` |
| `ap1` | `https://otlp-http-intake.ap1.datadoghq.com` |
| `gov` | `https://otlp-http-intake.ddog-gov.com` |

## Writing a Custom Sink

Subclass `BaseSink` and implement `write`, `flush`, `close`:

```python
from llmtrace.sinks import BaseSink
from llmtrace.models import TraceEvent

class PostgresSink(BaseSink):
    def __init__(self, dsn: str):
        self._dsn = dsn

    async def write(self, event: TraceEvent) -> None:
        await self._pool.execute("INSERT INTO traces ...", event.to_dict())

    async def flush(self) -> None:
        pass  # no buffering

    async def close(self) -> None:
        await self._pool.close()
```

`BaseSink` provides:
- Async context manager support (`async with sink:`)
- `_log_error(error, context)` helper for consistent error logging

## Configuring Sinks via String

`configure(sink=...)` accepts string shortcuts:

| String | Resolves to |
|---|---|
| `"console"` | `ConsoleSink()` |
| `"jsonfile:/path/to/file.jsonl"` | `JsonFileSink("/path/to/file.jsonl")` |
| `"webhook:https://example.com/traces"` | `WebhookSink("https://example.com/traces")` |
| `"otlp"` | `OTLPSink()` with default endpoint |
| `"otlp:http://collector:4318"` | `OTLPSink(endpoint="http://collector:4318")` |
| `"langfuse"` | `LangfuseSink` from `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` env vars |
| `"datadog"` | `DatadogSink` from `DD_API_KEY`, `DD_SITE` env vars |
