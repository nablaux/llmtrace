"""Tests for JsonFileSink."""

import json
from datetime import UTC, datetime
from pathlib import Path

from llmtrace.models import TraceEvent
from llmtrace.sinks.jsonfile import JsonFileSink


def _make_event(**kwargs: object) -> TraceEvent:
    """Create a TraceEvent with sensible defaults for testing."""
    defaults: dict[str, object] = {
        "provider": "openai",
        "model": "gpt-4",
        "latency_ms": 123.0,
        "timestamp": datetime(2024, 6, 15, 14, 30, 45, tzinfo=UTC),
    }
    defaults.update(kwargs)
    return TraceEvent(**defaults)  # type: ignore[arg-type]


def _read_lines(path: Path) -> list[str]:
    """Read non-empty lines from a file."""
    if not path.exists():
        return []
    return [line for line in path.read_text().splitlines() if line.strip()]


class TestJsonFileSinkBasicWrite:
    """Tests for basic write and flush."""

    async def test_write_and_flush_produces_valid_json_line(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        sink = JsonFileSink(path, buffer_size=10)

        await sink.write(_make_event())
        await sink.flush()

        lines = _read_lines(path)
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-4"

    async def test_each_line_is_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        sink = JsonFileSink(path, buffer_size=100)

        for i in range(5):
            await sink.write(_make_event(latency_ms=float(i)))
        await sink.flush()

        lines = _read_lines(path)
        assert len(lines) == 5
        for line in lines:
            data = json.loads(line)
            assert isinstance(data, dict)
            assert "trace_id" in data


class TestJsonFileSinkBuffering:
    """Tests for buffered write behavior."""

    async def test_buffer_not_flushed_until_full(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        sink = JsonFileSink(path, buffer_size=5)

        for _ in range(4):
            await sink.write(_make_event())

        # File should not exist yet (nothing flushed)
        assert not path.exists() or len(_read_lines(path)) == 0

    async def test_buffer_auto_flushes_at_buffer_size(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        sink = JsonFileSink(path, buffer_size=5)

        for _ in range(5):
            await sink.write(_make_event())

        lines = _read_lines(path)
        assert len(lines) == 5

    async def test_explicit_flush_writes_partial_buffer(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        sink = JsonFileSink(path, buffer_size=100)

        for _ in range(3):
            await sink.write(_make_event())
        await sink.flush()

        lines = _read_lines(path)
        assert len(lines) == 3


class TestJsonFileSinkAppend:
    """Tests for file append behavior."""

    async def test_multiple_flushes_append(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        sink = JsonFileSink(path, buffer_size=100)

        await sink.write(_make_event(provider="openai"))
        await sink.flush()
        await sink.write(_make_event(provider="anthropic"))
        await sink.flush()

        lines = _read_lines(path)
        assert len(lines) == 2
        assert json.loads(lines[0])["provider"] == "openai"
        assert json.loads(lines[1])["provider"] == "anthropic"


class TestJsonFileSinkRotation:
    """Tests for file rotation."""

    async def test_rotation_creates_rotated_file(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        # Tiny threshold so any write triggers rotation
        sink = JsonFileSink(path, rotate_mb=0.001, buffer_size=1)

        # Write enough events to exceed 0.001 MB (~1 KB)
        event = _make_event()
        for _ in range(20):
            await sink.write(event)

        rotated = path.with_suffix(".jsonl.1")
        assert rotated.exists()

    async def test_rotation_count_limit(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        sink = JsonFileSink(path, rotate_mb=0.001, buffer_size=1, rotate_count=3)

        event = _make_event()
        # Write many events to trigger multiple rotations
        for _ in range(200):
            await sink.write(event)

        # .1, .2, .3 should exist, .4 should not
        for i in range(1, 4):
            assert path.with_suffix(f".jsonl.{i}").exists(), f".jsonl.{i} should exist"
        assert not path.with_suffix(".jsonl.4").exists(), ".jsonl.4 should not exist"


class TestJsonFileSinkRotationOverflow:
    """Tests for rotation overflow deletion edge cases."""

    async def test_rotation_deletes_overflow_file(self, tmp_path: Path) -> None:
        """When a file beyond rotate_count exists, it gets deleted."""
        path = tmp_path / "trace.jsonl"
        # Create an artificial overflow file (.jsonl.4 with rotate_count=3)
        overflow = path.with_suffix(".jsonl.4")
        overflow.write_text("old data\n")
        assert overflow.exists()

        sink = JsonFileSink(path, rotate_mb=0.0001, buffer_size=1, rotate_count=3)

        # Write enough to trigger rotation
        for _ in range(30):
            await sink.write(_make_event())

        # Overflow file should be cleaned up
        assert not overflow.exists()
        await sink.close()

    async def test_rotation_unlinks_src_when_exceeds_count(self, tmp_path: Path) -> None:
        """When i+1 > rotate_count, src is unlinked instead of renamed."""
        path = tmp_path / "trace.jsonl"
        # Pre-create files: .jsonl, .jsonl.1, .jsonl.2, .jsonl.3
        path.write_text("current\n")
        for i in range(1, 4):
            path.with_suffix(f".jsonl.{i}").write_text(f"rotated {i}\n")

        sink = JsonFileSink(path, rotate_mb=0.0001, buffer_size=1, rotate_count=2)

        # Write to trigger rotation (file exists and exceeds threshold)
        for _ in range(5):
            await sink.write(_make_event())

        # .jsonl.3 and beyond should not exist with rotate_count=2
        assert not path.with_suffix(".jsonl.3").exists()
        await sink.close()


class TestJsonFileSinkClose:
    """Tests for close behavior."""

    async def test_close_flushes_remaining(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        sink = JsonFileSink(path, buffer_size=100)

        await sink.write(_make_event())
        await sink.close()

        lines = _read_lines(path)
        assert len(lines) == 1

    async def test_write_after_close_is_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        sink = JsonFileSink(path, buffer_size=100)

        await sink.write(_make_event())
        await sink.close()
        await sink.write(_make_event(provider="late"))
        await sink.flush()

        lines = _read_lines(path)
        assert len(lines) == 1


class TestJsonFileSinkErrorHandling:
    """Tests for IOError handling."""

    async def test_ioerror_does_not_crash(self, tmp_path: Path) -> None:
        # Point at a directory that doesn't allow writing
        bad_dir = tmp_path / "readonly"
        bad_dir.mkdir()
        bad_dir.chmod(0o444)
        path = bad_dir / "sub" / "trace.jsonl"

        sink = JsonFileSink(path, buffer_size=1)
        # Should not raise
        await sink.write(_make_event())

        # Restore permissions for cleanup
        bad_dir.chmod(0o755)


class TestJsonFileSinkContextManager:
    """Tests for async context manager."""

    async def test_async_context_manager(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        async with JsonFileSink(path, buffer_size=100) as sink:
            await sink.write(_make_event())

        lines = _read_lines(path)
        assert len(lines) == 1
