"""Global configuration for llmtrace."""

from __future__ import annotations

import threading
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from llmtrace._logging import get_logger
from llmtrace.pricing import PricingRegistry

logger = get_logger()

_config: LLMTraceConfig | None = None
_lock = threading.Lock()


class LLMTraceConfig(BaseModel):
    """Global configuration for the llmtrace library.

    Controls sink routing, default tags/metadata, pricing, enrichment,
    capture behavior, and sampling.
    """

    sink: Any = None
    default_tags: dict[str, str] = {}
    default_metadata: dict[str, Any] = {}
    pricing_registry: PricingRegistry = PricingRegistry()
    enrichers: list[Any] = []
    capture_request: bool = True
    capture_response: bool = True
    sample_rate: float = 1.0
    redact_sensitive_keys: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("sample_rate")
    @classmethod
    def _validate_sample_rate(cls, v: float) -> float:
        """Ensure sample_rate is between 0.0 and 1.0 inclusive."""
        if v < 0.0 or v > 1.0:
            msg = "sample_rate must be between 0.0 and 1.0"
            raise ValueError(msg)
        return v


def _resolve_sink(sink_str: str) -> Any:
    """Resolve a sink string identifier to a sink instance.

    Supported formats:
        - "console" -> ConsoleSink
        - "jsonfile:<path>" -> JsonFileSink(path)
        - "webhook:<url>" -> WebhookSink(url)
        - "otlp" -> OTLPSink (default endpoint)
        - "otlp:<endpoint>" -> OTLPSink(endpoint=...)
        - "langfuse" -> LangfuseSink (reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST)
        - "datadog" -> DatadogSink (reads DD_API_KEY, DD_SITE)
    """
    if sink_str == "console":
        from llmtrace.sinks.console import ConsoleSink

        return ConsoleSink()

    if sink_str.startswith("jsonfile:"):
        path = sink_str[len("jsonfile:") :]
        from llmtrace.sinks.jsonfile import JsonFileSink

        return JsonFileSink(path)

    if sink_str.startswith("webhook:"):
        url = sink_str[len("webhook:") :]
        from llmtrace.sinks.webhook import WebhookSink

        return WebhookSink(url)

    if sink_str == "otlp":
        from llmtrace.sinks.otlp import OTLPSink

        return OTLPSink()

    if sink_str.startswith("otlp:"):
        endpoint = sink_str[len("otlp:"):]
        from llmtrace.sinks.otlp import OTLPSink

        return OTLPSink(endpoint=endpoint)

    if sink_str == "langfuse":
        import os

        from llmtrace.sinks.langfuse import LangfuseSink

        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        if not public_key or not secret_key:
            msg = "Langfuse sink requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables"
            raise ValueError(msg)
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        return LangfuseSink(public_key=public_key, secret_key=secret_key, host=host)

    if sink_str == "datadog":
        import os

        from llmtrace.sinks.datadog import DatadogSink

        api_key = os.environ.get("DD_API_KEY")
        if not api_key:
            msg = "Datadog sink requires DD_API_KEY environment variable"
            raise ValueError(msg)
        site = os.environ.get("DD_SITE", "us1")
        return DatadogSink(api_key=api_key, site=site)

    msg = f"Unknown sink format: {sink_str!r}"
    raise ValueError(msg)


def _parse_tags(raw: str) -> dict[str, str]:
    """Parse a comma-separated key=value string into a dict."""
    tags: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" in pair:
            key, value = pair.split("=", 1)
            tags[key.strip()] = value.strip()
    return tags


def configure(**kwargs: Any) -> None:
    """Configure the global llmtrace settings.

    Accepts keyword arguments matching LLMTraceConfig fields. String sink
    values are resolved to sink instances. Environment variables provide
    fallbacks for arguments not explicitly provided.
    """
    global _config
    with _lock:
        import os

        # Resolve string sink
        if isinstance(kwargs.get("sink"), str):
            kwargs["sink"] = _resolve_sink(kwargs["sink"])

        # Environment variable fallbacks
        if "sink" not in kwargs:
            env_sink = os.environ.get("LLMTRACE_SINK")
            if env_sink is not None:
                kwargs["sink"] = _resolve_sink(env_sink)

        if "default_tags" not in kwargs:
            env_tags = os.environ.get("LLMTRACE_TAGS")
            if env_tags is not None:
                kwargs["default_tags"] = _parse_tags(env_tags)

        if "sample_rate" not in kwargs:
            env_rate = os.environ.get("LLMTRACE_SAMPLE_RATE")
            if env_rate is not None:
                kwargs["sample_rate"] = float(env_rate)

        if "capture_request" not in kwargs:
            env_req = os.environ.get("LLMTRACE_CAPTURE_REQUEST")
            if env_req is not None:
                kwargs["capture_request"] = env_req.lower() == "true"

        if "capture_response" not in kwargs:
            env_resp = os.environ.get("LLMTRACE_CAPTURE_RESPONSE")
            if env_resp is not None:
                kwargs["capture_response"] = env_resp.lower() == "true"

        if "redact_sensitive_keys" not in kwargs:
            env_redact = os.environ.get("LLMTRACE_REDACT_KEYS")
            if env_redact is not None:
                kwargs["redact_sensitive_keys"] = env_redact.lower() == "true"

        _config = LLMTraceConfig(**kwargs)


def get_config() -> LLMTraceConfig:
    """Return the current global configuration.

    Auto-configures with defaults (console sink) if not yet configured.
    """
    global _config
    if _config is None:
        configure(sink="console")
    assert _config is not None
    return _config


def reset() -> None:
    """Reset the global configuration to unconfigured state.

    Intended for use in tests.
    """
    global _config
    with _lock:
        _config = None
