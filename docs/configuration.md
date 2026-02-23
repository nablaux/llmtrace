# Configuration

## `configure(**kwargs)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sink` | `Sink \| str \| None` | `None` | Event destination. String shortcuts: `"console"`, `"jsonfile:<path>"`, `"webhook:<url>"`, `"otlp"`, `"otlp:<endpoint>"`, `"langfuse"`, `"datadog"` |
| `default_tags` | `dict[str, str]` | `{}` | Tags merged into every event |
| `default_metadata` | `dict[str, Any]` | `{}` | Metadata merged into every event |
| `pricing_registry` | `PricingRegistry` | built-in registry | Custom pricing data |
| `enrichers` | `list[Enricher]` | `[]` | Pipeline of event enrichers |
| `capture_request` | `bool` | `True` | Include request payload in events |
| `capture_response` | `bool` | `True` | Include response payload in events |
| `sample_rate` | `float` | `1.0` | Fraction of calls to trace (0.0–1.0) |
| `redact_sensitive_keys` | `bool` | `True` | Redact keys like `api_key`, `authorization` from requests |

Calling `configure()` with no arguments uses defaults (auto-configures console sink on first `get_config()`).

## Environment Variables

| Variable | Maps to | Example |
|---|---|---|
| `LLMTRACE_SINK` | `sink` | `console`, `jsonfile:/tmp/traces.jsonl`, `webhook:https://...`, `otlp`, `langfuse`, `datadog` |
| `LLMTRACE_TAGS` | `default_tags` | `env=prod,team=ml` |
| `LLMTRACE_SAMPLE_RATE` | `sample_rate` | `0.1` |
| `LLMTRACE_CAPTURE_REQUEST` | `capture_request` | `true` / `false` |
| `LLMTRACE_CAPTURE_RESPONSE` | `capture_response` | `true` / `false` |
| `LLMTRACE_REDACT_KEYS` | `redact_sensitive_keys` | `true` / `false` |
| `LANGFUSE_PUBLIC_KEY` | (used by `sink="langfuse"`) | `pk-lf-...` |
| `LANGFUSE_SECRET_KEY` | (used by `sink="langfuse"`) | `sk-lf-...` |
| `LANGFUSE_HOST` | (used by `sink="langfuse"`) | `https://cloud.langfuse.com` |
| `DD_API_KEY` | (used by `sink="datadog"`) | `dd-...` |
| `DD_SITE` | (used by `sink="datadog"`) | `us1`, `eu1`, etc. |

**Precedence:** explicit `configure()` args > environment variables > defaults.

## Sensitive Key Redaction

When `redact_sensitive_keys=True` (default), request payloads are deep-walked and values for these keys are replaced with `"[REDACTED]"` (case-insensitive):

`api_key`, `apikey`, `api-key`, `authorization`, `x-api-key`, `secret`, `token`, `password`, `credential`

Disable with `configure(redact_sensitive_keys=False)` or `LLMTRACE_REDACT_KEYS=false`.

## Configuration Patterns

### Minimal

```python
import llmtrace

llmtrace.configure()  # console sink, all defaults
llmtrace.instrument("anthropic")
```

### Production

```python
from llmtrace.sinks import JsonFileSink, WebhookSink, MultiSink
from llmtrace.transform.enrichment import (
    RedactPIIEnricher,
    CostAlertEnricher,
    LatencyClassifierEnricher,
    AddEnvironmentEnricher,
)

llmtrace.configure(
    sink=MultiSink([
        JsonFileSink("/var/log/llmtrace/events.jsonl", rotate_mb=50),
        WebhookSink("https://ingest.example.com/traces", headers={"Authorization": "Bearer ..."}),
    ]),
    default_tags={"service": "chatbot", "env": "prod"},
    enrichers=[
        RedactPIIEnricher(locales=("global", "en", "eu")),
        CostAlertEnricher(threshold_usd=Decimal("0.50")),
        LatencyClassifierEnricher(fast_ms=300, slow_ms=3000),
        AddEnvironmentEnricher(),
    ],
    sample_rate=0.5,
    capture_response=False,
)

# Or send traces to Langfuse:
# llmtrace.configure(sink="langfuse")  # uses LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY env vars

# Or to Datadog:
# llmtrace.configure(sink="datadog")  # uses DD_API_KEY, DD_SITE env vars
```

### Environment-Based (Zero Code Config)

```bash
export LLMTRACE_SINK="jsonfile:/var/log/traces.jsonl"
export LLMTRACE_TAGS="env=prod,service=api"
export LLMTRACE_SAMPLE_RATE="0.1"
export LLMTRACE_CAPTURE_RESPONSE="false"
export LLMTRACE_REDACT_KEYS="true"
```

```python
import llmtrace

llmtrace.configure()  # picks up all env vars
llmtrace.instrument("anthropic")
```

## Other Functions

- **`get_config() -> LLMTraceConfig`** — returns current config; auto-configures with console sink if unconfigured.
- **`reset() -> None`** — resets to unconfigured state. Intended for tests.
