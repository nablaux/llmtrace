"""Configurable sinks for llmtrace event output."""

from llmtrace.sinks.base import BaseSink
from llmtrace.sinks.callback import CallbackSink
from llmtrace.sinks.console import ConsoleSink
from llmtrace.sinks.jsonfile import JsonFileSink
from llmtrace.sinks.multi import MultiSink
from llmtrace.sinks.webhook import WebhookSink

__all__ = [
    "BaseSink",
    "CallbackSink",
    "ConsoleSink",
    "JsonFileSink",
    "MultiSink",
    "WebhookSink",
]

# Optional sinks — available only when their dependencies are installed
try:
    from llmtrace.sinks.otlp import OTLPSink  # noqa: F401

    __all__.append("OTLPSink")
except ImportError:
    pass

try:
    from llmtrace.sinks.langfuse import LangfuseSink  # noqa: F401

    __all__.append("LangfuseSink")
except ImportError:
    pass

try:
    from llmtrace.sinks.datadog import DatadogSink  # noqa: F401

    __all__.append("DatadogSink")
except ImportError:
    pass
