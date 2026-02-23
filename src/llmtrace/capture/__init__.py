"""Capture module for llmtrace — decorator and span context managers."""

from llmtrace.capture.context import SpanHandle, SpanHandleSync, span, span_sync
from llmtrace.capture.decorator import trace
from llmtrace.capture.tool_decorator import instrument_tools, trace_tool

__all__ = [
    "SpanHandle",
    "SpanHandleSync",
    "instrument_tools",
    "span",
    "span_sync",
    "trace",
    "trace_tool",
]
