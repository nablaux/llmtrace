"""Instrumentor registry and factory."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmtrace.instruments._base import BaseInstrumentor

INSTRUMENTOR_REGISTRY: dict[str, tuple[str, str]] = {
    "anthropic": ("llmtrace.instruments.anthropic", "AnthropicInstrumentor"),
    "openai": ("llmtrace.instruments.openai", "OpenAIInstrumentor"),
}


def get_instrumentor(provider: str) -> BaseInstrumentor:
    """Create an instrumentor instance for the given provider."""
    entry = INSTRUMENTOR_REGISTRY.get(provider)
    if entry is None:
        msg = f"Unknown provider: {provider}. Available: {list(INSTRUMENTOR_REGISTRY.keys())}"
        raise ValueError(msg)
    module_path, class_name = entry
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()  # type: ignore[no-any-return]  # dynamic import returns Any
