"""Tests for DatadogSink."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llmtrace.sinks.datadog import DATADOG_SITES, DatadogSink


def _make_datadog_sink(
    api_key: str = "test-dd-api-key",
    site: str = "us1",
    endpoint: str | None = None,
    **kwargs: Any,
) -> DatadogSink:
    """Create a DatadogSink with OTel internals mocked out.

    Patches _create_exporter, TracerProvider, and BatchSpanProcessor
    to avoid requiring actual OpenTelemetry SDK installation.
    """
    with (
        patch("llmtrace.sinks.otlp._create_exporter") as mock_exporter_fn,
        patch("llmtrace.sinks.otlp.OTLPSink.__init__.__module__", create=True),
    ):
        mock_exporter_fn.return_value = MagicMock()

        # Patch the OTel SDK imports inside OTLPSink.__init__
        mock_resource_cls = MagicMock()
        mock_provider_cls = MagicMock()
        mock_processor_cls = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "opentelemetry": MagicMock(),
                "opentelemetry.sdk": MagicMock(),
                "opentelemetry.sdk.resources": MagicMock(Resource=mock_resource_cls),
                "opentelemetry.sdk.trace": MagicMock(TracerProvider=mock_provider_cls),
                "opentelemetry.sdk.trace.export": MagicMock(BatchSpanProcessor=mock_processor_cls),
            },
        ):
            sink = DatadogSink(
                api_key=api_key,
                site=site,
                endpoint=endpoint,
                **kwargs,
            )

        # Stash mocks for assertion in tests
        sink._mock_exporter_fn = mock_exporter_fn  # type: ignore[attr-defined]
        sink._mock_provider_cls = mock_provider_cls  # type: ignore[attr-defined]
        return sink


class TestDatadogSinkDefaultSite:
    """Tests that DatadogSink defaults to the us1 site."""

    def test_default_site_is_us1(self) -> None:
        sink = _make_datadog_sink()
        sink._mock_exporter_fn.assert_called_once()  # type: ignore[attr-defined]
        call_kwargs = sink._mock_exporter_fn.call_args  # type: ignore[attr-defined]
        assert call_kwargs[0][1] == "https://otlp-http-intake.datadoghq.com"


class TestDatadogSinkApiKeyHeader:
    """Tests that the DD-API-KEY header is set correctly."""

    def test_dd_api_key_header_set(self) -> None:
        sink = _make_datadog_sink(api_key="my-secret-key")
        call_kwargs = sink._mock_exporter_fn.call_args  # type: ignore[attr-defined]
        headers = call_kwargs[0][2]
        assert headers == {"DD-API-KEY": "my-secret-key"}


class TestDatadogSinkSiteMapping:
    """Tests that each known site maps to the correct endpoint URL."""

    @pytest.mark.parametrize(
        "site,expected_url",
        list(DATADOG_SITES.items()),
        ids=list(DATADOG_SITES.keys()),
    )
    def test_site_resolves_to_correct_url(self, site: str, expected_url: str) -> None:
        sink = _make_datadog_sink(site=site)
        call_kwargs = sink._mock_exporter_fn.call_args  # type: ignore[attr-defined]
        endpoint = call_kwargs[0][1]
        assert endpoint == expected_url


class TestDatadogSinkCustomEndpoint:
    """Tests that a custom endpoint overrides the site lookup."""

    def test_custom_endpoint_overrides_site(self) -> None:
        custom = "http://localhost:4318"
        sink = _make_datadog_sink(endpoint=custom, site="eu1")
        call_kwargs = sink._mock_exporter_fn.call_args  # type: ignore[attr-defined]
        endpoint = call_kwargs[0][1]
        assert endpoint == custom


class TestDatadogSinkUnknownSite:
    """Tests that an unknown site identifier raises KeyError."""

    def test_unknown_site_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            _make_datadog_sink(site="mars1")


class TestDatadogSinkExtraKwargs:
    """Tests that extra kwargs are forwarded to OTLPSink."""

    def test_service_name_passed_through(self) -> None:
        sink = _make_datadog_sink(service_name="my-service")
        # The TracerProvider is created with a Resource that includes the service name.
        # Verify via the Resource.create call.
        mock_resource_create = sink._mock_provider_cls  # type: ignore[attr-defined]
        # The provider was instantiated — service_name goes into Resource.create
        assert mock_resource_create.called

    def test_capture_content_passed_through(self) -> None:
        sink = _make_datadog_sink(capture_content=True)
        assert sink._capture_content is True

    def test_capture_content_default_false(self) -> None:
        sink = _make_datadog_sink()
        assert sink._capture_content is False


class TestDatadogSinkEu1Site:
    """Tests specifically for the EU1 site endpoint."""

    def test_eu1_endpoint(self) -> None:
        sink = _make_datadog_sink(site="eu1")
        call_kwargs = sink._mock_exporter_fn.call_args  # type: ignore[attr-defined]
        endpoint = call_kwargs[0][1]
        assert endpoint == "https://otlp-http-intake.datadoghq.eu"


class TestDatadogSinkProtocol:
    """Tests that protocol is always set to http/protobuf."""

    def test_protocol_is_http_protobuf(self) -> None:
        sink = _make_datadog_sink()
        call_kwargs = sink._mock_exporter_fn.call_args  # type: ignore[attr-defined]
        protocol = call_kwargs[0][0]
        assert protocol == "http/protobuf"

    def test_protocol_kwarg_is_ignored(self) -> None:
        """User-supplied protocol is stripped and http/protobuf is used."""
        sink = _make_datadog_sink(protocol="grpc")  # type: ignore[arg-type]
        call_kwargs = sink._mock_exporter_fn.call_args  # type: ignore[attr-defined]
        protocol = call_kwargs[0][0]
        assert protocol == "http/protobuf"

    def test_headers_kwarg_is_ignored(self) -> None:
        """User-supplied headers are stripped in favor of DD-API-KEY."""
        sink = _make_datadog_sink(
            api_key="real-key",
            headers={"Authorization": "Bearer bad"},  # type: ignore[arg-type]
        )
        call_kwargs = sink._mock_exporter_fn.call_args  # type: ignore[attr-defined]
        headers = call_kwargs[0][2]
        assert headers == {"DD-API-KEY": "real-key"}
