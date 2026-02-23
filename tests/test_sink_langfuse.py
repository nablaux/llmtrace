"""Tests for LangfuseSink."""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import MagicMock, patch


def _build_langfuse_sink(
    public_key: str = "pk-lf-test",
    secret_key: str = "sk-lf-test",
    host: str = "https://cloud.langfuse.com",
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Create a LangfuseSink with OTel internals mocked out.

    Returns the sink instance and a dict of captured arguments passed
    to _create_exporter (protocol, endpoint, headers).
    """
    captured: dict[str, Any] = {}

    mock_exporter = MagicMock()
    mock_processor = MagicMock()
    mock_provider = MagicMock()
    mock_tracer = MagicMock()
    mock_provider.get_tracer.return_value = mock_tracer

    def fake_create_exporter(
        protocol: str, endpoint: str, headers: dict[str, str] | None
    ) -> MagicMock:
        captured["protocol"] = protocol
        captured["endpoint"] = endpoint
        captured["headers"] = headers
        return mock_exporter

    mock_resource_cls = MagicMock()
    mock_provider_cls = MagicMock(return_value=mock_provider)
    mock_processor_cls = MagicMock(return_value=mock_processor)

    with (
        patch("llmtrace.sinks.otlp._create_exporter", side_effect=fake_create_exporter),
        patch("llmtrace.sinks.otlp.Resource", mock_resource_cls, create=True),
        patch("llmtrace.sinks.otlp.TracerProvider", mock_provider_cls, create=True),
        patch("llmtrace.sinks.otlp.BatchSpanProcessor", mock_processor_cls, create=True),
    ):
        from llmtrace.sinks.langfuse import LangfuseSink

        sink = LangfuseSink(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            **kwargs,
        )

    return sink, captured


class TestLangfuseSinkDefaultEndpoint:
    """Tests that the default endpoint points to Langfuse EU cloud."""

    def test_default_endpoint(self) -> None:
        """Default host produces the correct OTLP endpoint."""
        _, captured = _build_langfuse_sink()
        assert captured["endpoint"] == "https://cloud.langfuse.com/api/public/otel"


class TestLangfuseSinkBasicAuth:
    """Tests that Basic Auth header is correctly encoded."""

    def test_basic_auth_encoding(self) -> None:
        """Authorization header contains base64(public_key:secret_key)."""
        _, captured = _build_langfuse_sink(
            public_key="pk-lf-abc123",
            secret_key="sk-lf-xyz789",
        )
        expected = base64.b64encode(b"pk-lf-abc123:sk-lf-xyz789").decode()
        assert captured["headers"] == {"Authorization": f"Basic {expected}"}

    def test_basic_auth_round_trip(self) -> None:
        """The encoded credentials decode back to the original key pair."""
        _, captured = _build_langfuse_sink(
            public_key="mykey",
            secret_key="mysecret",
        )
        auth_value = captured["headers"]["Authorization"]
        _, encoded = auth_value.split(" ", 1)
        decoded = base64.b64decode(encoded).decode()
        assert decoded == "mykey:mysecret"


class TestLangfuseSinkCustomHost:
    """Tests that custom hosts are correctly used."""

    def test_us_cloud_host(self) -> None:
        """US cloud host produces the correct endpoint."""
        _, captured = _build_langfuse_sink(
            host="https://us.cloud.langfuse.com",
        )
        assert captured["endpoint"] == "https://us.cloud.langfuse.com/api/public/otel"

    def test_self_hosted(self) -> None:
        """Self-hosted URL produces the correct endpoint."""
        _, captured = _build_langfuse_sink(
            host="https://langfuse.internal.company.com",
        )
        assert captured["endpoint"] == "https://langfuse.internal.company.com/api/public/otel"

    def test_trailing_slash_stripped(self) -> None:
        """Trailing slash on host does not produce double slashes."""
        _, captured = _build_langfuse_sink(
            host="https://cloud.langfuse.com/",
        )
        assert captured["endpoint"] == "https://cloud.langfuse.com/api/public/otel"


class TestLangfuseSinkProtocolForced:
    """Tests that protocol is always forced to http/protobuf."""

    def test_protocol_is_http_protobuf(self) -> None:
        """Protocol is always http/protobuf regardless of kwargs."""
        _, captured = _build_langfuse_sink()
        assert captured["protocol"] == "http/protobuf"

    def test_grpc_override_ignored(self) -> None:
        """Passing protocol='grpc' in kwargs is silently ignored."""
        _, captured = _build_langfuse_sink(protocol="grpc")
        assert captured["protocol"] == "http/protobuf"


class TestLangfuseSinkKwargsPassthrough:
    """Tests that extra kwargs are passed through to OTLPSink."""

    def test_service_name_passthrough(self) -> None:
        """service_name kwarg reaches OTLPSink."""
        sink, _ = _build_langfuse_sink(service_name="my-app")
        # OTLPSink stores no public service_name attr, but if we got here
        # without error, the kwarg was accepted by OTLPSink.__init__.
        assert sink is not None

    def test_capture_content_passthrough(self) -> None:
        """capture_content kwarg is stored by OTLPSink."""
        sink, _ = _build_langfuse_sink(capture_content=True)
        assert sink._capture_content is True

    def test_capture_content_default_false(self) -> None:
        """capture_content defaults to False when not passed."""
        sink, _ = _build_langfuse_sink()
        assert sink._capture_content is False

    def test_endpoint_override_ignored(self) -> None:
        """Passing endpoint in kwargs is silently ignored."""
        _, captured = _build_langfuse_sink(endpoint="http://evil.example.com")
        assert captured["endpoint"] == "https://cloud.langfuse.com/api/public/otel"

    def test_headers_override_ignored(self) -> None:
        """Passing headers in kwargs is silently ignored."""
        _, captured = _build_langfuse_sink(
            headers={"X-Evil": "hacker"},
        )
        assert "X-Evil" not in captured["headers"]
        assert "Authorization" in captured["headers"]
