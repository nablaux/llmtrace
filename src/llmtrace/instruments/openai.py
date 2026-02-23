"""Instrumentor for the OpenAI Python SDK."""

from __future__ import annotations

from typing import Any

from llmtrace.instruments._base import BaseInstrumentor


class OpenAIInstrumentor(BaseInstrumentor):
    """Instruments openai.resources.chat.completions.Completions.create and AsyncCompletions.create."""

    @property
    def provider_name(self) -> str:
        return "openai"

    def _get_targets(self) -> list[tuple[Any, str]]:
        try:
            import openai.resources.chat.completions
        except ImportError as err:
            msg = (
                "OpenAIInstrumentor requires the openai package. "
                "Install with: pip install llmtrace[openai]"
            )
            raise ImportError(msg) from err

        targets: list[tuple[Any, str]] = []

        if hasattr(openai.resources.chat.completions, "Completions"):
            targets.append((openai.resources.chat.completions.Completions, "create"))

        if hasattr(openai.resources.chat.completions, "AsyncCompletions"):
            targets.append((openai.resources.chat.completions.AsyncCompletions, "create"))

        return targets
