# Providers

## How Auto-Instrumentation Works

`llmtrace.instrument("provider")` monkey-patches the provider SDK's API methods at call time. Original methods are stored and restored on `uninstrument()`.

```python
import llmtrace

llmtrace.instrument("anthropic", "openai")
# ... use SDKs normally — all calls are traced ...
llmtrace.uninstrument()  # restore originals for all providers
llmtrace.uninstrument("openai")  # or restore a specific one
```

Both sync and async methods are wrapped. The wrapper:
1. Checks `sample_rate` — skips tracing if sampled out
2. Times the call
3. Captures errors as `ErrorTrace` (with retryability detection)
4. Extracts provider-specific data (model, tokens, tool calls)
5. Computes cost from the pricing registry
6. Applies enrichers
7. Writes the event to the configured sink

## Provider Reference

### Anthropic

- **Patched methods:** `Messages.create`, `AsyncMessages.create`
- **Extracted fields:** `model`, `input_tokens`, `output_tokens`, tool calls from `tool_use` content blocks
- **Streaming:** Not yet supported
- **Install:** `pip install llmtrace[anthropic]`

### OpenAI

- **Patched methods:** `Completions.create`, `AsyncCompletions.create`
- **Extracted fields:** `model`, `prompt_tokens`, `completion_tokens`, `total_tokens`, tool calls from `choices[0].message.tool_calls`
- **Streaming:** Not yet supported
- **Install:** `pip install llmtrace[openai]`

### Google (Gemini) / LiteLLM

Coming soon.

## Writing a Custom Instrumentor

Subclass `BaseInstrumentor`:

```python
from llmtrace.instruments._base import BaseInstrumentor

class CohereInstrumentor(BaseInstrumentor):
    @property
    def provider_name(self) -> str:
        return "cohere"

    def _get_targets(self) -> list[tuple[object, str]]:
        import cohere
        return [(cohere.Client, "chat")]
```

`BaseInstrumentor` handles monkey-patching, timing, error capture, enrichment, and sink dispatch.

Register it so `instrument("cohere")` works:

```python
from llmtrace.instruments import INSTRUMENTOR_REGISTRY
INSTRUMENTOR_REGISTRY["cohere"] = ("mypackage.instruments", "CohereInstrumentor")
```

## Registering a Custom Extractor

Extractors convert provider-specific responses into `ExtractedData`:

```python
from llmtrace.capture.extractors import ExtractorRegistry, ExtractedData
from llmtrace.models import TokenUsage

registry = ExtractorRegistry()

def extract_cohere(request_kwargs, response):
    return ExtractedData(
        provider="cohere",
        model=response.model or request_kwargs.get("model", "unknown"),
        request_payload=request_kwargs,
        response_payload={"text": response.text},
        token_usage=TokenUsage(
            prompt_tokens=response.meta.tokens.input_tokens,
            completion_tokens=response.meta.tokens.output_tokens,
        ),
    )

registry.register("cohere", extract_cohere)
```

Unknown providers fall back to a generic extractor that attempts common field names.
