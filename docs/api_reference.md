# API Reference

## `llmtrace` (top-level)

| Function | Description |
|---|---|
| `configure(**kwargs) -> None` | Configure global settings. See [Configuration](configuration.md) |
| `instrument(*providers: str) -> None` | Activate tracing for providers. Idempotent |
| `uninstrument(*providers: str) -> None` | Deactivate tracing. No args = all |
| `emit(event: TraceEvent) -> None` | Manually emit event to configured sink |
| `@trace(*, provider=None, tags=None, metadata=None)` | Decorator for sync/async functions. Auto-detects provider |
| `span(name, *, tags=None) -> SpanHandle` | Async context manager for grouping events |
| `span_sync(name, *, tags=None) -> SpanHandleSync` | Sync variant of `span` |

---

## `llmtrace.models`

### `TraceEvent` (mutable)

| Field | Type | Default |
|---|---|---|
| `trace_id` | `UUID` | `uuid4()` |
| `parent_id` / `span_id` | `UUID \| None` | `None` |
| `timestamp` | `datetime` | `now(UTC)` |
| `provider` / `model` | `str` | required |
| `request` / `response` | `dict[str, Any]` | `{}` |
| `token_usage` | `TokenUsage \| None` | `None` |
| `cost` | `Cost \| None` | `None` |
| `latency_ms` | `float` | required (≥0) |
| `tool_calls` | `list[ToolCallTrace]` | `[]` |
| `error` | `ErrorTrace \| None` | `None` |
| `tags` | `dict[str, str]` | `{}` |
| `metadata` | `dict[str, Any]` | `{}` |

Methods: `to_json() -> str`, `to_dict() -> dict[str, Any]`

### `TokenUsage` (frozen)

Auto-computes `total_tokens = prompt_tokens + completion_tokens` if not set.

Fields: `prompt_tokens: int`, `completion_tokens: int`, `total_tokens: int`, `cache_read_tokens: int | None`, `cache_write_tokens: int | None`

### `Cost` (frozen)

Auto-computes `total_cost = input_cost + output_cost`. Uses `Decimal`.

Fields: `input_cost: Decimal`, `output_cost: Decimal`, `total_cost: Decimal`, `currency: str = "USD"`

### `ToolCallTrace` (frozen)

Fields: `tool_name: str`, `arguments: dict`, `result: Any | None`, `latency_ms: float | None`, `success: bool = True`, `error_message: str | None`

### `ErrorTrace` (frozen)

Fields: `error_type: str`, `message: str`, `provider_error_code: str | None`, `is_retryable: bool = False`, `stack_trace: str | None`

### `SpanContext` (mutable)

Fields: `span_id: UUID`, `parent_span_id: UUID | None`, `name: str`, `started_at: datetime`, `ended_at: datetime | None`, `events: list[TraceEvent]`, `children: list[SpanContext]`, `tags: dict`, `annotations: dict`

Methods: `duration_ms() -> float | None`, `total_cost() -> Decimal`, `total_tokens() -> int`

---

## `llmtrace.config`

`LLMTraceConfig(BaseModel)` — see [Configuration](configuration.md) for all fields.

`configure(**kwargs)`, `get_config() -> LLMTraceConfig`, `reset() -> None`

---

## `llmtrace.pricing`

### `ModelPricing` (frozen)

Fields: `input_per_million: Decimal`, `output_per_million: Decimal`, `cache_read_per_million: Decimal | None`, `cache_write_per_million: Decimal | None`

### `PricingRegistry`

```python
registry = PricingRegistry()  # loads defaults for Anthropic, OpenAI, Google
registry.register(provider, model, ModelPricing(...))
registry.get(provider, model) -> ModelPricing | None  # exact + prefix match
registry.compute_cost(provider, model, usage) -> Cost | None
registry.list_models(provider=None) -> list[tuple[str, str]]
```

---

## `llmtrace.protocols`

```python
class Sink(Protocol):
    async def write(self, event: TraceEvent) -> None: ...
    async def flush(self) -> None: ...
    async def close(self) -> None: ...

class SinkSync(Protocol):          # sync variant
class Instrumentor(Protocol):       # provider_name, instrument(), uninstrument()
class Enricher(Protocol):           # __call__(event: TraceEvent) -> TraceEvent
```

Type aliases: `TagDict = dict[str, str]`, `MetadataDict = dict[str, Any]`

---

## `llmtrace.sinks`

See [Sinks](sinks.md). Classes: `BaseSink`, `ConsoleSink`, `JsonFileSink`, `WebhookSink`, `MultiSink`, `CallbackSink`, `OTLPSink`, `LangfuseSink`, `DatadogSink`.

---

## `llmtrace.capture`

### `SpanHandle` / `SpanHandleSync`

| Member | Description |
|---|---|
| `context -> SpanContext` | Underlying span context |
| `annotate(**kwargs)` | Add annotations |
| `add_event(event)` | Attach a `TraceEvent` |
| `child(name, *, tags=None)` | Nested child span (context manager) |

### `ExtractorRegistry`

```python
registry = ExtractorRegistry()  # built-in: anthropic, openai
registry.register("cohere", extractor_fn)
registry.get("anthropic")  # -> Callable[..., ExtractedData]
```

---

## `llmtrace.transform`

### `normalize`

`normalize(extracted, latency_ms, config, *, parent_id=None, span_id=None, extra_tags=None, extra_metadata=None, redact_sensitive_keys=True) -> TraceEvent` — applies key redaction, cost computation, throughput metrics, error classification.

### `EnrichmentPipeline`

```python
pipeline = EnrichmentPipeline([enricher_a, enricher_b])
event = pipeline.apply(event)
```

### Redaction

```python
PatternRedactor(*, locales=("global",), custom_patterns=None)
PresidioRedactor(*, language="en", score_threshold=0.5, entities=None)
RedactionEngine(redactors, strategy=RedactionStrategy.REPLACE)
RedactionEngine.redact(text) -> str
RedactionEngine.redact_dict(data) -> dict[str, Any]
RedactionMatch(start: int, end: int, entity_type: str)  # NamedTuple
RedactionStrategy: REPLACE | MASK | HASH
```

### Built-in Enrichers

```python
RedactPIIEnricher(*, locales=("global","intl"), strategy=REPLACE, use_presidio=False, presidio_language="en", custom_patterns=None)
CostAlertEnricher(threshold_usd=Decimal("1.00"))
LatencyClassifierEnricher(*, fast_ms=500, normal_ms=2000, slow_ms=5000)
AddEnvironmentEnricher()
```

---

## `llmtrace.instruments`

`BaseInstrumentor(ABC)`: abstract `provider_name -> str`, `_get_targets() -> list[tuple[object, str]]`; concrete `instrument()`, `uninstrument()`.

Concrete: `AnthropicInstrumentor`, `OpenAIInstrumentor`.

`INSTRUMENTOR_REGISTRY: dict[str, tuple[str, str]]` — provider → (module, class). `get_instrumentor(provider) -> BaseInstrumentor` — factory.
