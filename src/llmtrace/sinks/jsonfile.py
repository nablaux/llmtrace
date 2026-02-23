"""JSON Lines file sink with optional rotation and buffered writes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from llmtrace._logging import get_logger
from llmtrace.sinks.base import BaseSink

if TYPE_CHECKING:
    from llmtrace.models import TraceEvent

logger = get_logger()


class JsonFileSink(BaseSink):
    """Sink that writes trace events as JSON Lines to a file.

    Supports buffered writes and optional size-based file rotation.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        rotate_mb: float | None = None,
        rotate_count: int = 5,
        buffer_size: int = 10,
    ) -> None:
        """Initialize the JSON file sink.

        Args:
            path: File path for JSONL output.
            rotate_mb: Rotate file when it exceeds this size in megabytes.
                None disables rotation.
            rotate_count: Maximum number of rotated files to keep.
            buffer_size: Number of events to buffer before flushing to disk.
        """
        self._path = Path(path)
        self._rotate_mb = rotate_mb
        self._rotate_count = rotate_count
        self._buffer_size = buffer_size
        self._buffer: list[str] = []
        self._lock = asyncio.Lock()
        self._closed = False

    async def write(self, event: TraceEvent) -> None:
        """Serialize event to a compact JSON line and buffer it.

        Flushes automatically when the buffer reaches buffer_size.
        """
        if self._closed:
            return
        line = event.model_dump_json()
        self._buffer.append(line)
        if len(self._buffer) >= self._buffer_size:
            await self._flush_buffer()

    async def flush(self) -> None:
        """Flush all buffered events to disk."""
        await self._flush_buffer()

    async def close(self) -> None:
        """Flush remaining events and mark the sink as closed."""
        await self.flush()
        self._closed = True

    async def _flush_buffer(self) -> None:
        """Write buffered lines to disk under a lock.

        Checks file size and rotates if needed before writing.
        """
        async with self._lock:
            if not self._buffer:
                return
            try:
                if self._rotate_mb is not None and self._path.exists():
                    size = self._path.stat().st_size
                    if size >= self._rotate_mb * 1024 * 1024:
                        self._rotate()
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a") as f:
                    for line in self._buffer:
                        f.write(line + "\n")
                self._buffer.clear()
            except OSError as exc:
                self._log_error(exc, "flush buffer to file")

    def _rotate(self) -> None:
        """Rotate log files: path.N -> path.N+1, delete oldest beyond rotate_count."""
        for i in range(self._rotate_count - 1, 0, -1):
            src = self._path.with_suffix(f"{self._path.suffix}.{i}")
            dst = self._path.with_suffix(f"{self._path.suffix}.{i + 1}")
            if src.exists():
                if i + 1 > self._rotate_count:
                    src.unlink()
                else:
                    src.rename(dst)
        # Delete the file that exceeds rotate_count
        overflow = self._path.with_suffix(f"{self._path.suffix}.{self._rotate_count + 1}")
        if overflow.exists():
            overflow.unlink()
        # Move current file to .1
        if self._path.exists():
            self._path.rename(self._path.with_suffix(f"{self._path.suffix}.1"))
