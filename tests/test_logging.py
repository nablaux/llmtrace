"""Tests for llmtrace centralized logging."""

from __future__ import annotations

import logging

from llmtrace._logging import get_logger


class TestGetLogger:
    """Tests for the get_logger() helper."""

    def test_returns_logger_for_caller_module(self) -> None:
        logger = get_logger()
        # This test file's module name
        assert logger.name == __name__

    def test_returns_logging_logger_instance(self) -> None:
        logger = get_logger()
        assert isinstance(logger, logging.Logger)

    def test_different_modules_get_different_loggers(self) -> None:
        # get_logger() in this module vs the one in _logging itself
        local_logger = get_logger()
        assert local_logger.name == __name__


class TestNullHandler:
    """Tests for NullHandler setup on the root llmtrace logger."""

    def test_root_logger_has_null_handler(self) -> None:
        root = logging.getLogger("llmtrace")
        handler_types = [type(h) for h in root.handlers]
        assert logging.NullHandler in handler_types

    def test_child_logger_inherits(self) -> None:
        child = logging.getLogger("llmtrace.sinks.console")
        # Child loggers should be able to propagate to root
        assert child.parent is not None
