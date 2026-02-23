"""Enrichment pipeline, redaction framework, and built-in enrichers."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import socket
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from llmtrace.models import TraceEvent

from llmtrace._logging import get_logger

logger = get_logger()


# ═══════════════════════════════════════════════════════════════════════════
# Part A: Enrichment Pipeline
# ═══════════════════════════════════════════════════════════════════════════


class EnrichmentPipeline:
    """Applies a sequence of enrichers to a TraceEvent.

    Each enricher receives the output of the previous one. If any enricher
    raises an exception, it is logged and skipped — the pipeline never crashes.
    """

    def __init__(self, enrichers: Sequence[Callable[[TraceEvent], TraceEvent]]) -> None:
        self._enrichers = list(enrichers)

    def apply(self, event: TraceEvent) -> TraceEvent:
        """Run each enricher in order, skipping any that raise."""
        current = event
        for enricher in self._enrichers:
            try:
                current = enricher(current)
            except Exception:
                name = getattr(enricher, "__name__", enricher.__class__.__name__)
                logger.warning("Enricher %s raised an exception, skipping", name, exc_info=True)
        return current

    def __len__(self) -> int:
        return len(self._enrichers)

    def __repr__(self) -> str:
        names = [getattr(e, "__name__", e.__class__.__name__) for e in self._enrichers]
        return f"EnrichmentPipeline({names!r})"


# ═══════════════════════════════════════════════════════════════════════════
# Part B: Redaction Framework
# ═══════════════════════════════════════════════════════════════════════════


class RedactionMatch(NamedTuple):
    """A span within text identified as containing sensitive data."""

    start: int
    end: int
    entity_type: str


class RedactionStrategy(StrEnum):
    """Strategy for replacing detected PII spans."""

    REPLACE = "replace"
    MASK = "mask"
    HASH = "hash"


def _luhn_check(digits: str) -> bool:
    """Validate a credit card number using the Luhn algorithm."""
    total = 0
    reverse = digits[::-1]
    for i, ch in enumerate(reverse):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


_VERSION_PREFIX_RE = re.compile(
    r"(?:^|[\s(])(?:v|ver|version|release|build)[\s.:=\-]*$",
    re.IGNORECASE,
)


def _is_version_context(text: str, match_start: int) -> bool:
    """Check whether an IPv4 match is preceded by version-like context."""
    prefix = text[:match_start]
    return _VERSION_PREFIX_RE.search(prefix) is not None


def _merge_overlapping(matches: list[RedactionMatch]) -> list[RedactionMatch]:
    """Merge overlapping RedactionMatch spans, keeping the widest."""
    if not matches:
        return []
    sorted_matches = sorted(matches, key=lambda m: (m.start, -m.end))
    merged: list[RedactionMatch] = [sorted_matches[0]]
    for current in sorted_matches[1:]:
        prev = merged[-1]
        if current.start <= prev.end:
            # Overlapping — take widest span, keep the entity_type of whichever is wider
            if current.end > prev.end:
                merged[-1] = RedactionMatch(prev.start, current.end, prev.entity_type)
        else:
            merged.append(current)
    return merged


class PatternRedactor:
    """Detects PII using regex patterns. Supports multiple locales."""

    _PATTERNS: ClassVar[dict[str, list[tuple[str, re.Pattern[str]]]]] = {
        "global": [
            ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
            ("CREDIT_CARD", re.compile(r"\b(\d[ \-]?){12,18}\d\b")),
            (
                "IPV4",
                re.compile(
                    r"(?<!\d\.)(?<!\d)"
                    r"\b(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}"
                    r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\b"
                    r"(?!\.\d)"
                ),
            ),
            (
                "IPV6",
                re.compile(
                    r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
                    r"|"
                    r"\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b"
                    r"|"
                    r"\b::(?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4}\b"
                ),
            ),
            ("URL_WITH_AUTH", re.compile(r"https?://[^\s:]+:[^\s@]+@[^\s]+")),
            ("AWS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
            (
                "GENERIC_SECRET",
                re.compile(
                    r"(?:key|token|secret|password|apikey|api_key)[\s]*[=:]\s*['\"]?"
                    r"([A-Za-z0-9+/=_\-]{32,})['\"]?",
                    re.IGNORECASE,
                ),
            ),
        ],
        "en": [
            ("US_PHONE", re.compile(r"(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b")),
            ("US_SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
            ("US_PASSPORT", re.compile(r"\b[A-Z]\d{8}\b")),
        ],
        "eu": [
            (
                "EU_PHONE",
                re.compile(r"\+(?:33|44|49|34|39|31|32|46|47|48|41)\s?\d[\d\s\-]{7,14}\b"),
            ),
            (
                "IBAN",
                re.compile(
                    r"\b[A-Z]{2}\d{2}\s?[A-Z0-9]{4}[\s]?[A-Z0-9]{4}[\s]?[A-Z0-9]{4}(?:[\s]?[A-Z0-9]{4}){0,6}(?:[\s]?[A-Z0-9]{1,4})?\b"
                ),
            ),
            (
                "EU_VAT",
                re.compile(
                    r"\b(?:AT|BE|BG|CY|CZ|DE|DK|EE|EL|ES|FI|FR|HR|HU|IE|IT|LT|LU|LV|MT|NL|PL|PT|RO|SE|SI|SK)[A-Z0-9]{8,12}\b"
                ),
            ),
        ],
        "intl": [
            ("INTL_PHONE", re.compile(r"\+[1-9]\d{6,14}\b")),
            ("PASSPORT_GENERIC", re.compile(r"\b[A-Z0-9]{6,9}\b")),
        ],
    }

    def __init__(
        self,
        *,
        locales: Sequence[str] = ("global",),
        custom_patterns: dict[str, str] | None = None,
    ) -> None:
        self._patterns: list[tuple[str, re.Pattern[str]]] = []
        for locale in locales:
            locale_patterns = self._PATTERNS.get(locale, [])
            self._patterns.extend(locale_patterns)
        if custom_patterns:
            for entity_type, regex_str in custom_patterns.items():
                self._patterns.append((entity_type, re.compile(regex_str)))

    def detect(self, text: str) -> list[RedactionMatch]:
        """Run all loaded patterns against the text and return sorted, de-duplicated matches."""
        raw_matches: list[RedactionMatch] = []
        for entity_type, pattern in self._patterns:
            for m in pattern.finditer(text):
                # Apply Luhn check for credit cards to reduce false positives
                if entity_type == "CREDIT_CARD":
                    digits = re.sub(r"[^\d]", "", m.group())
                    if not _luhn_check(digits):
                        continue
                # Reject IPv4 matches preceded by version-like context
                if entity_type == "IPV4" and _is_version_context(text, m.start()):
                    continue
                raw_matches.append(RedactionMatch(m.start(), m.end(), entity_type))
        return _merge_overlapping(raw_matches)

    def __call__(self, text: str) -> list[RedactionMatch]:
        """Alias for detect()."""
        return self.detect(text)


class PresidioRedactor:
    """Detects PII using Microsoft Presidio NLP engine. Supports 50+ languages."""

    def __init__(
        self,
        *,
        language: str = "en",
        score_threshold: float = 0.5,
        entities: list[str] | None = None,
    ) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
        except ImportError:
            raise ImportError(
                "PresidioRedactor requires presidio-analyzer. "
                "Install with: pip install presidio-analyzer"
            ) from None

        self._language = language
        self._score_threshold = score_threshold
        self._entities = entities or [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "LOCATION",
            "MEDICAL_LICENSE",
            "NRP",
            "US_SSN",
        ]
        self._analyzer: Any = AnalyzerEngine()

    def detect(self, text: str) -> list[RedactionMatch]:
        """Run Presidio analyzer and convert results to RedactionMatch list."""
        results: Any = self._analyzer.analyze(
            text=text,
            language=self._language,
            entities=self._entities,
            score_threshold=self._score_threshold,
        )
        matches: list[RedactionMatch] = []
        for r in results:
            matches.append(RedactionMatch(r.start, r.end, r.entity_type))
        return _merge_overlapping(matches)

    def __call__(self, text: str) -> list[RedactionMatch]:
        """Alias for detect()."""
        return self.detect(text)


def _mask_text(text: str) -> str:
    """Mask a string, keeping structure chars and first/last characters."""
    if len(text) <= 2:
        return "*" * len(text)
    structure_chars = {"@", "-", ".", "+", " "}
    result = list(text)
    result[0] = text[0]
    result[-1] = text[-1]
    for i in range(1, len(text) - 1):
        if text[i] not in structure_chars:
            result[i] = "*"
    return "".join(result)


class RedactionEngine:
    """Composes multiple Redactors and applies a redaction strategy to text."""

    def __init__(
        self,
        redactors: Sequence[PatternRedactor | PresidioRedactor | Any],
        strategy: RedactionStrategy = RedactionStrategy.REPLACE,
    ) -> None:
        self._redactors = list(redactors)
        self._strategy = strategy

    def redact(self, text: str) -> str:
        """Apply all redactors and replace detected spans using the configured strategy."""
        all_matches: list[RedactionMatch] = []
        for redactor in self._redactors:
            all_matches.extend(redactor(text))
        merged = _merge_overlapping(all_matches)

        # Process from end to start to preserve indices
        result = text
        for match in reversed(merged):
            matched_text = result[match.start : match.end]
            if self._strategy == RedactionStrategy.REPLACE:
                replacement = f"[{match.entity_type}_REDACTED]"
            elif self._strategy == RedactionStrategy.MASK:
                replacement = _mask_text(matched_text)
            elif self._strategy == RedactionStrategy.HASH:
                h = hashlib.sha256(matched_text.encode()).hexdigest()[:8]
                replacement = f"[SHA:{h}]"
            else:  # pragma: no cover
                replacement = f"[{match.entity_type}_REDACTED]"
            result = result[: match.start] + replacement + result[match.end :]
        return result

    def redact_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Deep-walk a dict/list structure, applying redact() to every string value.

        Returns a new dict — never mutates input.
        """
        result = self._walk(data)
        assert isinstance(result, dict)  # always true for dict input
        return result

    def _walk(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._walk(item) for item in obj]
        if isinstance(obj, str):
            return self.redact(obj)
        return obj


# ═══════════════════════════════════════════════════════════════════════════
# Part C: Built-in Enrichers
# ═══════════════════════════════════════════════════════════════════════════


class RedactPIIEnricher:
    """Enricher that redacts PII from request and response payloads."""

    def __init__(
        self,
        *,
        locales: Sequence[str] = ("global", "intl"),
        strategy: RedactionStrategy = RedactionStrategy.REPLACE,
        use_presidio: bool = False,
        presidio_language: str = "en",
        custom_patterns: dict[str, str] | None = None,
    ) -> None:
        redactors: list[PatternRedactor | PresidioRedactor] = [
            PatternRedactor(locales=locales, custom_patterns=custom_patterns)
        ]
        if use_presidio:
            redactors.append(PresidioRedactor(language=presidio_language))
        self._engine = RedactionEngine(redactors, strategy=strategy)
        self._locales = list(locales)
        self._strategy = strategy

    def __call__(self, event: TraceEvent) -> TraceEvent:
        """Redact PII from event request and response payloads."""
        redacted_request = self._engine.redact_dict(event.request)
        redacted_response = self._engine.redact_dict(event.response)
        event = event.model_copy(
            update={
                "request": redacted_request,
                "response": redacted_response,
                "metadata": {
                    **event.metadata,
                    "pii_redaction_applied": True,
                    "pii_locales": self._locales,
                    "pii_strategy": self._strategy.value,
                },
            }
        )
        return event


class AddEnvironmentEnricher:
    """Enricher that adds host environment information to event metadata."""

    def __call__(self, event: TraceEvent) -> TraceEvent:
        """Add hostname, pid, python_version, and llmtrace_version to metadata."""
        import llmtrace

        event = event.model_copy(
            update={
                "metadata": {
                    **event.metadata,
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "python_version": platform.python_version(),
                    "llmtrace_version": llmtrace.__version__,
                },
            }
        )
        return event


class CostAlertEnricher:
    """Adds a cost_alert tag when a single trace exceeds a cost threshold."""

    def __init__(self, threshold_usd: Decimal = Decimal("1.00")) -> None:
        self._threshold_usd = threshold_usd

    def __call__(self, event: TraceEvent) -> TraceEvent:
        """Add cost_alert tag if event cost exceeds threshold."""
        if event.cost is not None and event.cost.total_cost > self._threshold_usd:
            event = event.model_copy(
                update={
                    "tags": {**event.tags, "cost_alert": "high"},
                    "metadata": {
                        **event.metadata,
                        "cost_alert_threshold_usd": str(self._threshold_usd),
                    },
                }
            )
        return event


class LatencyClassifierEnricher:
    """Adds a latency_class tag based on latency thresholds."""

    def __init__(
        self,
        *,
        fast_ms: float = 500,
        normal_ms: float = 2000,
        slow_ms: float = 5000,
    ) -> None:
        self._fast_ms = fast_ms
        self._normal_ms = normal_ms
        self._slow_ms = slow_ms

    def __call__(self, event: TraceEvent) -> TraceEvent:
        """Classify latency and add latency_class tag."""
        latency = event.latency_ms
        if latency < self._fast_ms:
            classification = "fast"
        elif latency < self._normal_ms:
            classification = "normal"
        elif latency < self._slow_ms:
            classification = "slow"
        else:
            classification = "critical"

        event = event.model_copy(
            update={
                "tags": {**event.tags, "latency_class": classification},
                "metadata": {
                    **event.metadata,
                    "latency_thresholds": {
                        "fast_ms": self._fast_ms,
                        "normal_ms": self._normal_ms,
                        "slow_ms": self._slow_ms,
                    },
                },
            }
        )
        return event
