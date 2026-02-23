"""Instrumentor for the Anthropic Python SDK."""

from __future__ import annotations

from typing import Any

from llmtrace.instruments._base import BaseInstrumentor


class AnthropicInstrumentor(BaseInstrumentor):
    """Instruments anthropic.resources.Messages.create and AsyncMessages.create."""

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _get_targets(self) -> list[tuple[Any, str]]:
        try:
            import anthropic.resources
        except ImportError as err:
            msg = (
                "AnthropicInstrumentor requires the anthropic package. "
                "Install with: pip install llmtrace[anthropic]"
            )
            raise ImportError(msg) from err

        targets: list[tuple[Any, str]] = []

        if hasattr(anthropic.resources, "Messages"):
            targets.append((anthropic.resources.Messages, "create"))

        if hasattr(anthropic.resources, "AsyncMessages"):
            targets.append((anthropic.resources.AsyncMessages, "create"))

        return targets
