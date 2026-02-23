"""Centralized logging setup for llmtrace."""

from __future__ import annotations

import inspect
import logging


def _setup_null_handler() -> None:
    """Attach a NullHandler to the root 'llmtrace' logger.

    Prevents 'No handlers could be found' warnings when the library
    is used without user-configured logging.
    """
    root = logging.getLogger("llmtrace")
    if not root.handlers:
        root.addHandler(logging.NullHandler())


_setup_null_handler()


def get_logger() -> logging.Logger:
    """Return a logger named after the caller's module.

    Uses ``inspect.stack()`` to resolve the caller's module name,
    so every call site gets a correctly-scoped logger without having
    to pass ``__name__`` explicitly.
    """
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    name = module.__name__ if module is not None else __name__
    return logging.getLogger(name)
