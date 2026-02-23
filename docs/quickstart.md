# Quickstart

## Prerequisites

- Python 3.11+
- pip or uv

## Install

```bash
pip install llmtrace[anthropic]
# or: pip install llmtrace[openai]
```

## Configure & Trace

```python
import llmtrace
from anthropic import Anthropic

# 1. Configure — console sink is the default
llmtrace.configure()
llmtrace.instrument("anthropic")

# 2. Use the SDK as normal — calls are traced automatically
client = Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=128,
    messages=[{"role": "user", "content": "Say hello"}],
)
```

Output (stderr):

```
[14:32:01] anthropic/claude-sonnet-4-20250514 | 823ms | 12→45 tokens | $0.000711 | ✓
```

## What's in a Trace?

Each `TraceEvent` captures:

- **trace_id** — unique UUID per call
- **provider / model** — e.g. `"anthropic"`, `"claude-sonnet-4-20250514"`
- **request / response** — full payloads (redactable)
- **token_usage** — prompt, completion, cache tokens
- **cost** — input + output cost in USD (auto-computed)
- **latency_ms** — wall-clock time
- **tool_calls** — tool name, arguments, result
- **error** — type, message, retryability, stack trace
- **tags / metadata** — user-defined key-value pairs

## Decorator-Based Tracing

For custom functions that call LLM APIs:

```python
@llmtrace.trace(provider="anthropic", tags={"team": "ml"})
async def summarize(text: str):
    return await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{"role": "user", "content": f"Summarize: {text}"}],
    )
```

## Next Steps

- [Configuration](configuration.md) — sinks, sampling, environment variables
- [Sinks](sinks.md) — console, JSON file, webhook, custom sinks
- [Enrichment](enrichment.md) — PII redaction, cost alerts, custom enrichers
- [Providers](providers.md) — Anthropic, OpenAI, custom instrumentors
- [API Reference](api_reference.md) — full module-by-module reference
