"""Tests for llmtrace.transform.enrichment."""

from __future__ import annotations

import copy
import logging
from decimal import Decimal
from typing import Any

import pytest

from llmtrace.models import Cost, TraceEvent
from llmtrace.transform.enrichment import (
    AddEnvironmentEnricher,
    CostAlertEnricher,
    EnrichmentPipeline,
    LatencyClassifierEnricher,
    PatternRedactor,
    RedactionEngine,
    RedactionStrategy,
    RedactPIIEnricher,
)

# ── Helpers ─────────────────────────────────────────────────────────────


def _make_event(**kwargs: Any) -> TraceEvent:
    defaults: dict[str, Any] = {
        "provider": "openai",
        "model": "gpt-4",
        "latency_ms": 500.0,
        "request": {"prompt": "hello"},
        "response": {"text": "world"},
    }
    defaults.update(kwargs)
    return TraceEvent(**defaults)


# ═══════════════════════════════════════════════════════════════════════════
# EnrichmentPipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestEnrichmentPipeline:
    def test_applies_enrichers_in_order(self) -> None:
        def add_first(event: TraceEvent) -> TraceEvent:
            return event.model_copy(update={"metadata": {**event.metadata, "first": True}})

        def add_second(event: TraceEvent) -> TraceEvent:
            return event.model_copy(update={"metadata": {**event.metadata, "second": True}})

        pipeline = EnrichmentPipeline([add_first, add_second])
        result = pipeline.apply(_make_event())

        assert result.metadata["first"] is True
        assert result.metadata["second"] is True

    def test_handles_enricher_exception(self, caplog: pytest.LogCaptureFixture) -> None:
        def bad_enricher(event: TraceEvent) -> TraceEvent:
            raise ValueError("boom")

        def good_enricher(event: TraceEvent) -> TraceEvent:
            return event.model_copy(update={"metadata": {**event.metadata, "good": True}})

        pipeline = EnrichmentPipeline([bad_enricher, good_enricher])
        with caplog.at_level(logging.WARNING):
            result = pipeline.apply(_make_event())

        assert result.metadata["good"] is True
        assert "bad_enricher" in caplog.text

    def test_empty_enricher_list_returns_event_unchanged(self) -> None:
        event = _make_event()
        pipeline = EnrichmentPipeline([])
        result = pipeline.apply(event)
        assert result == event

    def test_len_returns_correct_count(self) -> None:
        pipeline = EnrichmentPipeline([lambda e: e, lambda e: e, lambda e: e])
        assert len(pipeline) == 3


# ═══════════════════════════════════════════════════════════════════════════
# PatternRedactor — "global" locale
# ═══════════════════════════════════════════════════════════════════════════


class TestPatternRedactorGlobal:
    def test_detects_standard_email(self) -> None:
        redactor = PatternRedactor(locales=("global",))
        matches = redactor.detect("Contact john@example.com for info")
        assert len(matches) >= 1
        email_match = next(m for m in matches if m.entity_type == "EMAIL")
        assert (
            "Contact john@example.com for info"[email_match.start : email_match.end]
            == "john@example.com"
        )

    def test_detects_credit_card_with_spaces(self) -> None:
        redactor = PatternRedactor(locales=("global",))
        matches = redactor.detect("Card: 4111 1111 1111 1111")
        assert any(m.entity_type == "CREDIT_CARD" for m in matches)

    def test_detects_credit_card_with_dashes(self) -> None:
        redactor = PatternRedactor(locales=("global",))
        matches = redactor.detect("Card: 4111-1111-1111-1111")
        assert any(m.entity_type == "CREDIT_CARD" for m in matches)

    def test_no_false_positive_on_version_number(self) -> None:
        redactor = PatternRedactor(locales=("global",))
        # "version X.X.X.X" should not match as IPv4
        for text in [
            "version 1.2.3.4 release",
            "v1.2.3.4",
            "release 10.20.30.40",
            "build 2.0.0.1",
        ]:
            matches = redactor.detect(text)
            ipv4_matches = [m for m in matches if m.entity_type == "IPV4"]
            assert len(ipv4_matches) == 0, f"False positive on: {text}"

    def test_detects_real_ipv4(self) -> None:
        redactor = PatternRedactor(locales=("global",))
        # Real IPs without version context should match
        for text, expected_ip in [
            ("DNS server 8.8.8.8 is Google", "8.8.8.8"),
            ("host: 192.168.1.1", "192.168.1.1"),
            ("connect to 10.0.0.1 now", "10.0.0.1"),
        ]:
            matches = redactor.detect(text)
            ipv4_matches = [m for m in matches if m.entity_type == "IPV4"]
            assert len(ipv4_matches) == 1, f"Missed real IP in: {text}"
            assert text[ipv4_matches[0].start : ipv4_matches[0].end] == expected_ip

    def test_detects_aws_key(self) -> None:
        redactor = PatternRedactor(locales=("global",))
        matches = redactor.detect("key is AKIAIOSFODNN7EXAMPLE")
        assert any(m.entity_type == "AWS_KEY" for m in matches)

    def test_no_match_on_non_pii(self) -> None:
        redactor = PatternRedactor(locales=("global",))
        matches = redactor.detect("just a normal sentence with no PII")
        assert len(matches) == 0

    def test_detects_multiple_pii_in_same_string(self) -> None:
        redactor = PatternRedactor(locales=("global",))
        text = "Email: john@example.com, key: AKIAIOSFODNN7EXAMPLE"
        matches = redactor.detect(text)
        entity_types = {m.entity_type for m in matches}
        assert "EMAIL" in entity_types
        assert "AWS_KEY" in entity_types


# ═══════════════════════════════════════════════════════════════════════════
# PatternRedactor — "intl" locale
# ═══════════════════════════════════════════════════════════════════════════


class TestPatternRedactorIntl:
    def test_detects_international_phone(self) -> None:
        redactor = PatternRedactor(locales=("intl",))
        matches = redactor.detect("Call +33612345678")
        assert any(m.entity_type == "INTL_PHONE" for m in matches)

    def test_detects_e164_format(self) -> None:
        redactor = PatternRedactor(locales=("intl",))
        matches = redactor.detect("Phone: +447911123456")
        assert any(m.entity_type == "INTL_PHONE" for m in matches)

    def test_detects_iban(self) -> None:
        redactor = PatternRedactor(locales=("eu",))
        matches = redactor.detect("IBAN: DE89370400440532013000")
        assert any(m.entity_type == "IBAN" for m in matches)


# ═══════════════════════════════════════════════════════════════════════════
# PatternRedactor — "en" locale
# ═══════════════════════════════════════════════════════════════════════════


class TestPatternRedactorEN:
    def test_detects_us_phone(self) -> None:
        redactor = PatternRedactor(locales=("en",))
        matches = redactor.detect("Call (555) 123-4567")
        assert any(m.entity_type == "US_PHONE" for m in matches)

    def test_detects_us_ssn(self) -> None:
        redactor = PatternRedactor(locales=("en",))
        matches = redactor.detect("SSN: 123-45-6789")
        assert any(m.entity_type == "US_SSN" for m in matches)


# ═══════════════════════════════════════════════════════════════════════════
# PatternRedactor — custom patterns
# ═══════════════════════════════════════════════════════════════════════════


class TestPatternRedactorCustom:
    def test_custom_pattern_detected(self) -> None:
        redactor = PatternRedactor(
            locales=("global",),
            custom_patterns={"EMPLOYEE_ID": r"\bEMP-\d{6}\b"},
        )
        matches = redactor.detect("Employee EMP-123456 logged in")
        assert any(m.entity_type == "EMPLOYEE_ID" for m in matches)


# ═══════════════════════════════════════════════════════════════════════════
# RedactionEngine
# ═══════════════════════════════════════════════════════════════════════════


class TestRedactionEngine:
    def test_replace_strategy(self) -> None:
        redactor = PatternRedactor(locales=("global",))
        engine = RedactionEngine([redactor], strategy=RedactionStrategy.REPLACE)
        result = engine.redact("Email: john@example.com")
        assert "[EMAIL_REDACTED]" in result
        assert "john@example.com" not in result

    def test_mask_strategy(self) -> None:
        redactor = PatternRedactor(locales=("global",))
        engine = RedactionEngine([redactor], strategy=RedactionStrategy.MASK)
        result = engine.redact("Email: john@example.com")
        # Structure chars preserved, middle chars masked
        assert "@" in result
        assert "john@example.com" not in result

    def test_hash_strategy(self) -> None:
        redactor = PatternRedactor(locales=("global",))
        engine = RedactionEngine([redactor], strategy=RedactionStrategy.HASH)
        result = engine.redact("Email: john@example.com")
        assert "[SHA:" in result
        assert "john@example.com" not in result

    def test_redact_dict_handles_nested(self) -> None:
        redactor = PatternRedactor(locales=("global",))
        engine = RedactionEngine([redactor], strategy=RedactionStrategy.REPLACE)
        data = {
            "user": {"email": "john@example.com"},
            "items": ["Contact admin@test.org", 42],
            "count": 10,
        }
        result = engine.redact_dict(data)
        assert "[EMAIL_REDACTED]" in result["user"]["email"]
        assert "[EMAIL_REDACTED]" in result["items"][0]
        assert result["items"][1] == 42
        assert result["count"] == 10

    def test_redact_dict_does_not_mutate_input(self) -> None:
        redactor = PatternRedactor(locales=("global",))
        engine = RedactionEngine([redactor], strategy=RedactionStrategy.REPLACE)
        data = {"email": "john@example.com", "nested": {"key": "admin@test.org"}}
        original = copy.deepcopy(data)
        engine.redact_dict(data)
        assert data == original

    def test_overlapping_matches_merged(self) -> None:
        """When two patterns overlap, the widest span should win."""
        redactor = PatternRedactor(
            locales=("global",),
            custom_patterns={"BROAD": r"john@example\.com"},
        )
        engine = RedactionEngine([redactor], strategy=RedactionStrategy.REPLACE)
        result = engine.redact("Email: john@example.com end")
        # Should only have one redaction, not nested/duplicated
        assert result.count("REDACTED]") == 1


# ═══════════════════════════════════════════════════════════════════════════
# RedactPIIEnricher
# ═══════════════════════════════════════════════════════════════════════════


class TestRedactPIIEnricher:
    def test_redacts_email_from_request(self) -> None:
        enricher = RedactPIIEnricher(locales=("global",))
        event = _make_event(request={"prompt": "Email me at john@example.com"})
        result = enricher(event)
        assert "john@example.com" not in str(result.request)
        assert "[EMAIL_REDACTED]" in result.request["prompt"]

    def test_redacts_phone_from_response(self) -> None:
        enricher = RedactPIIEnricher(locales=("global", "intl"))
        event = _make_event(response={"text": "Call +33612345678"})
        result = enricher(event)
        assert "+33612345678" not in str(result.response)

    def test_adds_pii_metadata(self) -> None:
        enricher = RedactPIIEnricher(locales=("global",))
        event = _make_event()
        result = enricher(event)
        assert result.metadata["pii_redaction_applied"] is True
        assert result.metadata["pii_locales"] == ["global"]
        assert result.metadata["pii_strategy"] == "replace"

    def test_multiple_locales_catch_global_and_intl(self) -> None:
        enricher = RedactPIIEnricher(locales=("global", "intl"))
        event = _make_event(request={"prompt": "john@example.com and +447911123456"})
        result = enricher(event)
        assert "john@example.com" not in str(result.request)
        assert "+447911123456" not in str(result.request)


# ═══════════════════════════════════════════════════════════════════════════
# PresidioRedactor
# ═══════════════════════════════════════════════════════════════════════════


class TestPresidioRedactor:
    def _try_create_redactor(self) -> Any:
        """Attempt to create a PresidioRedactor, skipping if spacy model is unavailable."""
        pytest.importorskip("presidio_analyzer")
        from llmtrace.transform.enrichment import PresidioRedactor

        try:
            return PresidioRedactor(language="en")
        except (OSError, SystemExit):
            pytest.skip("spacy language model not available")

    def test_detects_person_name(self) -> None:
        redactor = self._try_create_redactor()
        matches = redactor.detect("My name is John Smith and I live in New York")
        entity_types = {m.entity_type for m in matches}
        assert "PERSON" in entity_types

    def test_detects_email_via_nlp(self) -> None:
        redactor = self._try_create_redactor()
        matches = redactor.detect("Email me at john@example.com")
        entity_types = {m.entity_type for m in matches}
        assert "EMAIL_ADDRESS" in entity_types


# ═══════════════════════════════════════════════════════════════════════════
# AddEnvironmentEnricher
# ═══════════════════════════════════════════════════════════════════════════


class TestAddEnvironmentEnricher:
    def test_adds_env_metadata(self) -> None:
        enricher = AddEnvironmentEnricher()
        event = _make_event()
        result = enricher(event)
        assert "hostname" in result.metadata
        assert "pid" in result.metadata
        assert "python_version" in result.metadata


# ═══════════════════════════════════════════════════════════════════════════
# CostAlertEnricher
# ═══════════════════════════════════════════════════════════════════════════


class TestCostAlertEnricher:
    def test_cost_above_threshold_gets_alert(self) -> None:
        enricher = CostAlertEnricher(threshold_usd=Decimal("1.00"))
        cost = Cost(input_cost=Decimal("0.80"), output_cost=Decimal("0.50"))
        event = _make_event(cost=cost)
        result = enricher(event)
        assert result.tags["cost_alert"] == "high"
        assert "cost_alert_threshold_usd" in result.metadata

    def test_cost_below_threshold_unchanged(self) -> None:
        enricher = CostAlertEnricher(threshold_usd=Decimal("1.00"))
        cost = Cost(input_cost=Decimal("0.10"), output_cost=Decimal("0.10"))
        event = _make_event(cost=cost)
        result = enricher(event)
        assert "cost_alert" not in result.tags

    def test_no_cost_unchanged(self) -> None:
        enricher = CostAlertEnricher(threshold_usd=Decimal("1.00"))
        event = _make_event()
        result = enricher(event)
        assert "cost_alert" not in result.tags


# ═══════════════════════════════════════════════════════════════════════════
# LatencyClassifierEnricher
# ═══════════════════════════════════════════════════════════════════════════


class TestLatencyClassifierEnricher:
    def test_fast(self) -> None:
        enricher = LatencyClassifierEnricher()
        result = enricher(_make_event(latency_ms=200.0))
        assert result.tags["latency_class"] == "fast"

    def test_normal(self) -> None:
        enricher = LatencyClassifierEnricher()
        result = enricher(_make_event(latency_ms=1500.0))
        assert result.tags["latency_class"] == "normal"

    def test_slow(self) -> None:
        enricher = LatencyClassifierEnricher()
        result = enricher(_make_event(latency_ms=3000.0))
        assert result.tags["latency_class"] == "slow"

    def test_critical(self) -> None:
        enricher = LatencyClassifierEnricher()
        result = enricher(_make_event(latency_ms=8000.0))
        assert result.tags["latency_class"] == "critical"

    def test_exact_boundary_fast_to_normal(self) -> None:
        enricher = LatencyClassifierEnricher()
        result = enricher(_make_event(latency_ms=500.0))
        assert result.tags["latency_class"] == "normal"

    def test_exact_boundary_normal_to_slow(self) -> None:
        enricher = LatencyClassifierEnricher()
        result = enricher(_make_event(latency_ms=2000.0))
        assert result.tags["latency_class"] == "slow"

    def test_exact_boundary_slow_to_critical(self) -> None:
        enricher = LatencyClassifierEnricher()
        result = enricher(_make_event(latency_ms=5000.0))
        assert result.tags["latency_class"] == "critical"


# ═══════════════════════════════════════════════════════════════════════════
# EnrichmentPipeline — edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEnrichmentPipelineEdge:
    def test_repr_shows_enricher_names(self) -> None:
        def my_enricher(event: TraceEvent) -> TraceEvent:
            return event

        pipeline = EnrichmentPipeline([my_enricher])
        r = repr(pipeline)
        assert "EnrichmentPipeline" in r
        assert "my_enricher" in r

    def test_repr_empty(self) -> None:
        pipeline = EnrichmentPipeline([])
        assert "EnrichmentPipeline" in repr(pipeline)

    def test_class_enricher_uses_class_name(self) -> None:
        class MyEnricher:
            def __call__(self, event: TraceEvent) -> TraceEvent:
                return event

        pipeline = EnrichmentPipeline([MyEnricher()])
        assert "MyEnricher" in repr(pipeline)


# ═══════════════════════════════════════════════════════════════════════════
# Luhn validation edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestLuhnValidation:
    def test_valid_visa_detected(self) -> None:
        redactor = PatternRedactor(locales=("global",))
        # 4111111111111111 is a well-known valid test card
        matches = redactor.detect("Card: 4111111111111111")
        assert any(m.entity_type == "CREDIT_CARD" for m in matches)

    def test_invalid_luhn_rejected(self) -> None:
        redactor = PatternRedactor(locales=("global",))
        # 4111111111111112 fails Luhn check
        matches = redactor.detect("Card: 4111111111111112")
        assert not any(m.entity_type == "CREDIT_CARD" for m in matches)

    def test_valid_mastercard_detected(self) -> None:
        redactor = PatternRedactor(locales=("global",))
        # 5500000000000004 is a valid test Mastercard
        matches = redactor.detect("MC: 5500000000000004")
        assert any(m.entity_type == "CREDIT_CARD" for m in matches)


# ═══════════════════════════════════════════════════════════════════════════
# Overlapping span merge edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestMergeOverlapping:
    def test_empty_input(self) -> None:
        from llmtrace.transform.enrichment import _merge_overlapping

        assert _merge_overlapping([]) == []

    def test_non_overlapping_preserved(self) -> None:
        from llmtrace.transform.enrichment import RedactionMatch, _merge_overlapping

        matches = [
            RedactionMatch(0, 5, "A"),
            RedactionMatch(10, 15, "B"),
        ]
        result = _merge_overlapping(matches)
        assert len(result) == 2

    def test_fully_contained_merge(self) -> None:
        from llmtrace.transform.enrichment import RedactionMatch, _merge_overlapping

        matches = [
            RedactionMatch(0, 20, "OUTER"),
            RedactionMatch(5, 10, "INNER"),
        ]
        result = _merge_overlapping(matches)
        assert len(result) == 1
        assert result[0].start == 0
        assert result[0].end == 20

    def test_partial_overlap_extends(self) -> None:
        from llmtrace.transform.enrichment import RedactionMatch, _merge_overlapping

        matches = [
            RedactionMatch(0, 10, "A"),
            RedactionMatch(8, 20, "B"),
        ]
        result = _merge_overlapping(matches)
        assert len(result) == 1
        assert result[0].start == 0
        assert result[0].end == 20

    def test_adjacent_not_merged(self) -> None:
        from llmtrace.transform.enrichment import RedactionMatch, _merge_overlapping

        matches = [
            RedactionMatch(0, 5, "A"),
            RedactionMatch(6, 10, "B"),
        ]
        result = _merge_overlapping(matches)
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════════════
# _mask_text edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestMaskText:
    def test_single_char(self) -> None:
        from llmtrace.transform.enrichment import _mask_text

        assert _mask_text("a") == "*"

    def test_two_chars(self) -> None:
        from llmtrace.transform.enrichment import _mask_text

        assert _mask_text("ab") == "**"

    def test_preserves_first_and_last(self) -> None:
        from llmtrace.transform.enrichment import _mask_text

        result = _mask_text("abcde")
        assert result[0] == "a"
        assert result[-1] == "e"
        assert result[1:-1] == "***"


# ═══════════════════════════════════════════════════════════════════════════
# PatternRedactor — unknown locale
# ═══════════════════════════════════════════════════════════════════════════


class TestPatternRedactorUnknownLocale:
    def test_unknown_locale_produces_no_patterns(self) -> None:
        redactor = PatternRedactor(locales=("nonexistent",))
        matches = redactor.detect("john@example.com")
        assert len(matches) == 0

    def test_callable_alias(self) -> None:
        redactor = PatternRedactor(locales=("global",))
        matches = redactor("john@example.com")
        assert len(matches) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# PresidioRedactor import error
# ═══════════════════════════════════════════════════════════════════════════


class TestPresidioImportError:
    def test_import_error_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        real_import = builtins.__import__

        def blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "presidio_analyzer" or name.startswith("presidio_analyzer."):
                raise ImportError("No module named 'presidio_analyzer'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocking_import)

        with pytest.raises(ImportError, match="presidio-analyzer"):
            from llmtrace.transform.enrichment import PresidioRedactor

            PresidioRedactor(language="en")
