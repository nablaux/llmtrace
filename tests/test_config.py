"""Tests for llmtrace configuration module."""

import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from llmtrace.config import LLMTraceConfig, configure, get_config, reset


@pytest.fixture()
def _mock_console_sink(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a fake llmtrace.sinks.console module so sink='console' resolves."""
    fake_module = MagicMock()
    monkeypatch.setitem(sys.modules, "llmtrace.sinks.console", fake_module)
    return fake_module.ConsoleSink


class TestConfigure:
    """Tests for configure() function."""

    def setup_method(self) -> None:
        reset()

    def teardown_method(self) -> None:
        reset()

    def test_configure_with_explicit_sink_object(self) -> None:
        """configure() with an explicit sink object stores it directly."""
        mock_sink = MagicMock()
        configure(sink=mock_sink)
        cfg = get_config()
        assert cfg.sink is mock_sink

    @pytest.mark.usefixtures("_mock_console_sink")
    def test_configure_with_console_string(self) -> None:
        """configure() with sink='console' resolves to a ConsoleSink instance."""
        configure(sink="console")
        cfg = get_config()
        # The sink should be the return value of ConsoleSink()
        assert cfg.sink is not None

    def test_configure_with_default_tags(self) -> None:
        """configure() with default_tags stores them correctly."""
        mock_sink = MagicMock()
        configure(sink=mock_sink, default_tags={"env": "test", "service": "api"})
        cfg = get_config()
        assert cfg.default_tags == {"env": "test", "service": "api"}

    def test_configure_reads_env_tags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """configure() reads LLMTRACE_TAGS env var as fallback."""
        monkeypatch.setenv("LLMTRACE_TAGS", "env=prod,service=web")
        mock_sink = MagicMock()
        configure(sink=mock_sink)
        cfg = get_config()
        assert cfg.default_tags == {"env": "prod", "service": "web"}

    def test_configure_explicit_tags_override_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit default_tags take precedence over LLMTRACE_TAGS env var."""
        monkeypatch.setenv("LLMTRACE_TAGS", "env=prod")
        mock_sink = MagicMock()
        configure(sink=mock_sink, default_tags={"env": "staging"})
        cfg = get_config()
        assert cfg.default_tags == {"env": "staging"}

    def test_configure_reads_env_sample_rate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """configure() reads LLMTRACE_SAMPLE_RATE env var as fallback."""
        monkeypatch.setenv("LLMTRACE_SAMPLE_RATE", "0.5")
        mock_sink = MagicMock()
        configure(sink=mock_sink)
        cfg = get_config()
        assert cfg.sample_rate == 0.5

    def test_configure_reads_env_capture_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """configure() reads LLMTRACE_CAPTURE_REQUEST env var."""
        monkeypatch.setenv("LLMTRACE_CAPTURE_REQUEST", "false")
        mock_sink = MagicMock()
        configure(sink=mock_sink)
        cfg = get_config()
        assert cfg.capture_request is False

    def test_configure_reads_env_capture_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """configure() reads LLMTRACE_CAPTURE_RESPONSE env var."""
        monkeypatch.setenv("LLMTRACE_CAPTURE_RESPONSE", "false")
        mock_sink = MagicMock()
        configure(sink=mock_sink)
        cfg = get_config()
        assert cfg.capture_response is False

    def test_configure_reads_env_redact_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """configure() reads LLMTRACE_REDACT_KEYS env var as fallback."""
        monkeypatch.setenv("LLMTRACE_REDACT_KEYS", "false")
        mock_sink = MagicMock()
        configure(sink=mock_sink)
        cfg = get_config()
        assert cfg.redact_sensitive_keys is False


class TestGetConfig:
    """Tests for get_config() function."""

    def setup_method(self) -> None:
        reset()

    def teardown_method(self) -> None:
        reset()

    @pytest.mark.usefixtures("_mock_console_sink")
    def test_get_config_auto_configures(self) -> None:
        """get_config() auto-configures with defaults if not yet configured."""
        cfg = get_config()
        assert isinstance(cfg, LLMTraceConfig)
        assert cfg.sample_rate == 1.0
        assert cfg.capture_request is True
        assert cfg.capture_response is True


class TestReset:
    """Tests for reset() function."""

    @pytest.mark.usefixtures("_mock_console_sink")
    def test_reset_clears_config(self) -> None:
        """reset() sets _config back to None so get_config re-initializes."""
        mock_sink = MagicMock()
        configure(sink=mock_sink, default_tags={"k": "v"})
        cfg1 = get_config()
        assert cfg1.default_tags == {"k": "v"}

        reset()

        cfg2 = get_config()
        assert cfg2.default_tags == {}


class TestSampleRateValidation:
    """Tests for sample_rate field validation."""

    def test_sample_rate_zero_is_valid(self) -> None:
        cfg = LLMTraceConfig(sample_rate=0.0)
        assert cfg.sample_rate == 0.0

    def test_sample_rate_one_is_valid(self) -> None:
        cfg = LLMTraceConfig(sample_rate=1.0)
        assert cfg.sample_rate == 1.0

    def test_sample_rate_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMTraceConfig(sample_rate=-0.1)

    def test_sample_rate_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMTraceConfig(sample_rate=1.1)


class TestThreadSafety:
    """Tests for thread safety of configure()."""

    def setup_method(self) -> None:
        reset()

    def teardown_method(self) -> None:
        reset()

    def test_concurrent_configure_does_not_crash(self) -> None:
        """configure() from 10 threads simultaneously doesn't crash."""
        errors: list[Any] = []

        def _configure(i: int) -> None:
            try:
                mock_sink = MagicMock()
                configure(sink=mock_sink, default_tags={"thread": str(i)})
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_configure, i) for i in range(10)]
            for f in futures:
                f.result()

        assert errors == []
        cfg = get_config()
        assert isinstance(cfg, LLMTraceConfig)


def _mock_otel_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject fake OTel SDK modules so OTLPSink/LangfuseSink/DatadogSink can be created."""
    fake_resources = MagicMock()
    fake_trace = MagicMock()
    fake_export = MagicMock()
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.resources", fake_resources)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace", fake_trace)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace.export", fake_export)


class TestResolveSinkOTLP:
    """Tests for OTLP, Langfuse, and Datadog sink string shortcuts."""

    def setup_method(self) -> None:
        reset()

    def teardown_method(self) -> None:
        reset()

    @patch("llmtrace.sinks.otlp._create_exporter", return_value=MagicMock())
    def test_configure_otlp_default(
        self, _mock_exporter: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """configure(sink='otlp') creates an OTLPSink with default endpoint."""
        _mock_otel_sdk(monkeypatch)
        configure(sink="otlp")
        cfg = get_config()
        # Import here to check isinstance after mocking
        from llmtrace.sinks.otlp import OTLPSink

        assert isinstance(cfg.sink, OTLPSink)

    @patch("llmtrace.sinks.otlp._create_exporter", return_value=MagicMock())
    def test_configure_otlp_with_endpoint(
        self, _mock_exporter: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """configure(sink='otlp:<endpoint>') creates OTLPSink with that endpoint."""
        _mock_otel_sdk(monkeypatch)
        configure(sink="otlp:http://my-collector:4318")
        cfg = get_config()
        from llmtrace.sinks.otlp import OTLPSink

        assert isinstance(cfg.sink, OTLPSink)

    @patch("llmtrace.sinks.otlp._create_exporter", return_value=MagicMock())
    def test_configure_langfuse_with_env_vars(
        self, _mock_exporter: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """configure(sink='langfuse') with env vars creates a LangfuseSink."""
        _mock_otel_sdk(monkeypatch)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        configure(sink="langfuse")
        cfg = get_config()
        from llmtrace.sinks.langfuse import LangfuseSink

        assert isinstance(cfg.sink, LangfuseSink)

    @patch("llmtrace.sinks.otlp._create_exporter", return_value=MagicMock())
    def test_configure_datadog_with_env_vars(
        self, _mock_exporter: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """configure(sink='datadog') with env vars creates a DatadogSink."""
        _mock_otel_sdk(monkeypatch)
        monkeypatch.setenv("DD_API_KEY", "dd-test-key")
        configure(sink="datadog")
        cfg = get_config()
        from llmtrace.sinks.datadog import DatadogSink

        assert isinstance(cfg.sink, DatadogSink)

    def test_configure_langfuse_missing_env_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """configure(sink='langfuse') without env vars raises ValueError."""
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY"):
            configure(sink="langfuse")

    def test_configure_datadog_missing_env_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """configure(sink='datadog') without DD_API_KEY raises ValueError."""
        monkeypatch.delenv("DD_API_KEY", raising=False)
        with pytest.raises(ValueError, match="DD_API_KEY"):
            configure(sink="datadog")
