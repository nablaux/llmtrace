# CLAUDE.md — llmtrace Project Context

## What This Project Is

llmtrace is a **zero-dependency, structured tracing library for LLM applications**. It wraps LLM client calls and emits typed trace events (latency, token usage, cost, tool calls, errors) to configurable sinks. Positioning: "structlog for LLM calls."

## Architecture

Three-stage pipeline: **Capture → Transform → Emit**

- **Capture**: `@trace` decorator, `@trace_tool` decorator, `span()` context manager, or explicit `emit()`. Intercepts LLM calls and tool executions, times them, extracts request/response data.
- **Transform**: Normalizes raw data into `TraceEvent` (Pydantic v2 model). Computes cost from token usage + pricing registry. Applies user enrichers.
- **Emit**: Dispatches `TraceEvent` to one or more sinks (console, JSONL file, webhook, OTLP, callback, multi).

## Key Invariants

1. **Zero required runtime dependencies beyond Pydantic.** Provider SDKs and sink dependencies are optional. All provider imports are lazy (`try/except ImportError` inside function bodies, NEVER at module level).
2. **No monkey-patching at import time.** Instrumentation wraps methods only when `instrument()` is explicitly called.
3. **Async-first, sync-compatible.** Internal pipeline is async. Sync wrappers provided. Sinks never block the caller.
4. **All data is typed.** Every model is Pydantic v2. Every protocol uses `typing.Protocol`.
5. **Thread-safe.** Global config uses `threading.Lock`. Context propagation uses `contextvars`.

## Code Conventions

- **Python >=3.11**, using modern syntax (PEP 604 unions `X | Y`, PEP 585 generics `list[X]`)
- **src layout**: all source in `src/llmtrace/`
- **Pydantic v2** with `model_config = ConfigDict(...)`. Use `field_validator`, not `@validator`.
- **Protocols over ABCs** for public interfaces. ABCs only for internal base classes with shared implementation.
- **Decimal for money**, never float.
- **Type hints on everything.** mypy strict mode. No `# type: ignore` without explanation.
- **No `print()`.** Use `from llmtrace._logging import get_logger; logger = get_logger()` everywhere.
- **Docstrings**: Google style on all public classes and methods. One-line summary, then details.
- **Naming**: snake_case for functions/methods/variables, PascalCase for classes, UPPER_CASE for module constants.
- **Imports**: absolute imports only. Group: stdlib → third-party → local. isort handles ordering.
- **Error handling**: Never silently swallow exceptions in sinks — log warnings. Never let trace emission crash the user's application.

## Project Structure

```
src/llmtrace/
├── __init__.py          # Public API: configure, instrument, trace, trace_tool, span, emit, LangfuseSink, DatadogSink
├── _version.py          # __version__ = "0.1.0"
├── _logging.py          # get_logger(), NullHandler setup
├── models.py            # TraceEvent, TokenUsage, Cost, ToolCallTrace, ErrorTrace, SpanContext
├── config.py            # LLMTraceConfig, configure(), get_config(), reset()
├── pricing.py           # PricingRegistry, ModelPricing
├── protocols.py         # Sink, SinkSync, Instrumentor, Enricher, Metric protocols
├── capture/
│   ├── decorator.py     # @trace
│   ├── tool_decorator.py # @trace_tool, instrument_tools()
│   ├── context.py       # span() context manager
│   └── extractors.py    # Provider-specific data extraction
├── transform/
│   ├── normalizer.py    # ExtractedData → TraceEvent
│   └── enrichment.py    # EnrichmentPipeline, built-in enrichers
├── sinks/
│   ├── base.py          # BaseSink ABC
│   ├── console.py       # ConsoleSink
│   ├── jsonfile.py      # JsonFileSink
│   ├── webhook.py       # WebhookSink
│   ├── otlp.py          # OTLPSink
│   ├── langfuse.py      # LangfuseSink
│   ├── datadog.py       # DatadogSink
│   ├── callback.py      # CallbackSink
│   └── multi.py         # MultiSink
└── instruments/
    ├── _base.py         # BaseInstrumentor ABC
    ├── anthropic.py     # AnthropicInstrumentor
    ├── openai.py        # OpenAIInstrumentor
    ├── google.py        # GoogleInstrumentor
    └── litellm.py       # LiteLLMInstrumentor
```

## Testing

- Framework: **pytest** + **pytest-asyncio** (auto mode)
- HTTP mocking: **respx** (for httpx-based WebhookSink)
- Coverage target: **90%+ total**, **95%+ on models and config**
- Tests live in `tests/` mirroring source structure: `test_models.py`, `test_sink_console.py`, etc.
- Mock provider SDKs — never require actual SDK installation for tests
- Each test file must be runnable independently

## Commands

```bash
# Install all deps
uv sync --all-extras

# Run all checks
uv run mypy src/llmtrace/ --strict
uv run ruff check src/ tests/
uv run pytest tests/ -v --cov=src/llmtrace --cov-report=term-missing

# Run single test file
uv run pytest tests/test_models.py -v

# Format
uv run ruff format src/ tests/

# Build
uv build
```

## Dependencies

### Required (runtime)
- pydantic>=2.0

### Optional (by group)
- `anthropic`: anthropic>=0.40
- `openai`: openai>=1.50
- `google`: google-genai>=1.0
- `otlp`: opentelemetry-api, opentelemetry-sdk
- `otlp-grpc`: opentelemetry-exporter-otlp-proto-grpc
- `langfuse`: llmtrace[otlp]
- `datadog`: llmtrace[otlp]
- `webhook`: httpx>=0.27
- `all`: everything above

### Dev
- pytest, pytest-asyncio, pytest-cov, mypy, ruff, respx

## Critical Rules

1. **NEVER import provider SDKs at module level.** Always inside function bodies with `try/except ImportError`.
2. **NEVER let tracing crash the user's app.** All sink operations must be wrapped in try/except with logging.
3. **NEVER store API keys in traces.** Redact Authorization headers and api_key fields by default.
4. **ALWAYS compute cost as Decimal, not float.** Money math must be exact.
5. **ALWAYS use Pydantic v2 syntax.** No v1 compat (`@validator`, `Config` inner class, etc.).
6. **ALWAYS preserve the wrapped function's signature and metadata** (`functools.wraps`).
