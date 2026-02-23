"""Datadog sink — pre-configured OTLPSink for Datadog backends."""

from __future__ import annotations

from typing import Any

from llmtrace.sinks.otlp import OTLPSink

DATADOG_SITES: dict[str, str] = {
    "us1": "https://otlp-http-intake.datadoghq.com",
    "us3": "https://otlp-http-intake.us3.datadoghq.com",
    "us5": "https://otlp-http-intake.us5.datadoghq.com",
    "eu1": "https://otlp-http-intake.datadoghq.eu",
    "ap1": "https://otlp-http-intake.ap1.datadoghq.com",
    "gov": "https://otlp-http-intake.ddog-gov.com",
}


class DatadogSink(OTLPSink):
    """OTLPSink pre-configured for Datadog OTLP intake.

    Supports both direct OTLP ingestion (with DD-API-KEY header) and
    Datadog Agent-based setups (custom endpoint, no auth needed).
    """

    def __init__(
        self,
        api_key: str,
        site: str = "us1",
        endpoint: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Datadog sink.

        Args:
            api_key: Datadog API key.
            site: Datadog site identifier (us1, us3, us5, eu1, ap1, gov).
            endpoint: Override endpoint URL. Use for Datadog Agent setups
                      (e.g. "http://localhost:4318").
            **kwargs: Additional arguments passed to OTLPSink.
        """
        resolved = endpoint or DATADOG_SITES[site]

        kwargs.pop("protocol", None)
        kwargs.pop("headers", None)

        super().__init__(
            endpoint=resolved,
            headers={"DD-API-KEY": api_key},
            protocol="http/protobuf",
            **kwargs,
        )
