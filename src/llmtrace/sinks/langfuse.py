"""Langfuse sink — pre-configured OTLPSink for Langfuse backends."""

from __future__ import annotations

import base64
from typing import Any

from llmtrace.sinks.otlp import OTLPSink


class LangfuseSink(OTLPSink):
    """OTLPSink pre-configured for Langfuse.

    Handles Basic Auth encoding and endpoint construction. Langfuse only
    supports OTLP/HTTP, so the protocol is forced to "http/protobuf".
    """

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str = "https://cloud.langfuse.com",
        **kwargs: Any,
    ) -> None:
        """Initialize the Langfuse sink.

        Args:
            public_key: Langfuse public API key.
            secret_key: Langfuse secret API key.
            host: Langfuse host URL. Defaults to EU cloud.
                  Use "https://us.cloud.langfuse.com" for US cloud.
            **kwargs: Additional arguments passed to OTLPSink.
        """
        credentials = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()

        kwargs.pop("protocol", None)
        kwargs.pop("endpoint", None)
        kwargs.pop("headers", None)

        super().__init__(
            endpoint=f"{host.rstrip('/')}/api/public/otel",
            headers={"Authorization": f"Basic {credentials}"},
            protocol="http/protobuf",
            **kwargs,
        )
