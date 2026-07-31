"""Tests for shared.safe_io.load_json_or_warn."""

from __future__ import annotations

import json
import logging
import os
import stat
import threading
from pathlib import Path

import pytest

SENTINEL = object()  # unique default marker for assertions


@pytest.fixture(autouse=True)
def clear_warned_paths():
    """Clear the per-process dedup set before and after each test.

    Mirrors fused-memory/tests/test_sqlite_task_backend.py lines 44-48 to
    ensure warn-once assertions are deterministic across test isolation.
    """
    import shared.safe_io as _safe_io

    _safe_io._warned_corrupt_paths.clear()
    yield
    _safe_io._warned_corrupt_paths.clear()


class TestBenignPaths:
    """(·, True) paths: valid JSON and missing file are both benign."""

    def test_valid_json_returns_parsed_and_true(self, tmp_path, caplog):
        """Valid JSON file returns (parsed_obj, True) with no WARNING."""
        from shared.safe_io import load_json_or_warn

        data = {'key': 'value', 'num': 42}
        p = tmp_path / 'state.json'
        p.write_text(json.dumps(data))

        with caplog.at_level(logging.WARNING):
            result, ok = load_json_or_warn(p, default=SENTINEL)

        assert ok is True
        assert result == data
        assert result is not SENTINEL
        assert len(caplog.records) == 0, f'Expected no WARNING, got: {caplog.records}'

    def test_missing_file_returns_default_and_true_silently(self, tmp_path, caplog):
        """Nonexistent path returns (default, True) with no log records at all."""
        from shared.safe_io import load_json_or_warn

        p = tmp_path / 'nonexistent.json'
        assert not p.exists()

        with caplog.at_level(logging.WARNING):
            result, ok = load_json_or_warn(p, default=SENTINEL)

        assert ok is True
        assert result is SENTINEL
        assert len(caplog.records) == 0, f'Expected silent return, got: {caplog.records}'


class TestCorruptWarnMode:
    """Default on_corrupt='warn': return (default, False) + one WARNING."""

    def test_corrupt_file_returns_default_false_and_warns(self, tmp_path, caplog):
        """Corrupt JSON returns (SENTINEL, False) with exactly one WARNING naming the path."""
        from shared.safe_io import load_json_or_warn

        p = tmp_path / 'state.json'
        p.write_bytes(b'{not json')

        with caplog.at_level(logging.WARNING):
            result, ok = load_json_or_warn(p, default=SENTINEL)

        assert ok is False
        assert result is SENTINEL
        assert len(caplog.records) == 1, f'Expected exactly one WARNING, got: {caplog.records}'
        assert str(p) in caplog.records[0].message


class TestOnCorruptDispatch:
    """on_corrupt dispatch: fail_closed raises, unknown mode raises ValueError."""

    def test_fail_closed_raises_and_emits_no_warning(self, tmp_path, caplog):
        """on_corrupt='fail_closed' raises ValueError (JSONDecodeError subclass) with NO WARNING."""
        from shared.safe_io import load_json_or_warn

        p = tmp_path / 'state.json'
        p.write_bytes(b'{not json')

        with caplog.at_level(logging.WARNING), pytest.raises(ValueError):
            load_json_or_warn(p, default=SENTINEL, on_corrupt='fail_closed')

        assert len(caplog.records) == 0, (
            f'fail_closed must emit NO WARNING (exception is the loud channel), '
            f'got: {caplog.records}'
        )

    def test_unknown_on_corrupt_raises_value_error(self, tmp_path):
        """Unknown on_corrupt value raises ValueError eagerly, before any I/O."""
        from shared.safe_io import load_json_or_warn

        # File is valid JSON — the guard fires before reading so file state
        # is irrelevant.
        p = tmp_path / 'state.json'
        p.write_text(json.dumps({'ok': True}))

        with pytest.raises(ValueError, match='unknown on_corrupt'):
            load_json_or_warn(p, default=SENTINEL, on_corrupt='bogus')


class TestQuarantine:
    """on_corrupt='quarantine': rename file to <name>.corrupt, return (default, False)."""

    def test_quarantine_renames_file_and_returns_default_false(self, tmp_path, caplog):
        """Corrupt file is renamed to x.json.corrupt; returns (SENTINEL, False) with one WARNING."""
        from shared.safe_io import load_json_or_warn

        corrupt_bytes = b'{not json'
        p = tmp_path / 'x.json'
        p.write_bytes(corrupt_bytes)

        with caplog.at_level(logging.WARNING):
            result, ok = load_json_or_warn(p, default=SENTINEL, on_corrupt='quarantine')

        assert ok is False
        assert result is SENTINEL

        # Original file must be gone.
        assert not p.exists(), f'Original file should have been quarantined but still exists: {p}'

        # Quarantine sibling must exist with the original bytes.
        quarantine = tmp_path / 'x.json.corrupt'
        assert quarantine.exists(), f'Quarantine file missing: {quarantine}'
        assert quarantine.read_bytes() == corrupt_bytes

        # Exactly one WARNING must name the path.
        assert len(caplog.records) == 1, f'Expected exactly one WARNING, got: {caplog.records}'
        assert str(p) in caplog.records[0].message


class TestDedup:
    """Warn-once-per-path dedup: repeated calls on the same path emit only one WARNING."""

    def test_same_corrupt_path_warns_only_once(self, tmp_path, caplog):
        """Two calls on the same corrupt path return (default, False) twice but emit ONE WARNING."""
        from shared.safe_io import load_json_or_warn

        p = tmp_path / 'state.json'
        p.write_bytes(b'{not json')

        with caplog.at_level(logging.WARNING):
            r1, ok1 = load_json_or_warn(p, default=SENTINEL)
            r2, ok2 = load_json_or_warn(p, default=SENTINEL)

        assert ok1 is False
        assert ok2 is False
        assert r1 is SENTINEL
        assert r2 is SENTINEL
        assert len(caplog.records) == 1, (
            f'Same path should emit exactly ONE WARNING across two calls, got: {len(caplog.records)}'
        )

    def test_two_different_corrupt_paths_each_warn_once(self, tmp_path, caplog):
        """One call each on two different corrupt paths emits TWO WARNINGs (dedup is per-path)."""
        from shared.safe_io import load_json_or_warn

        p1 = tmp_path / 'a.json'
        p2 = tmp_path / 'b.json'
        p1.write_bytes(b'{bad')
        p2.write_bytes(b'{also bad')

        with caplog.at_level(logging.WARNING):
            load_json_or_warn(p1, default=SENTINEL)
            load_json_or_warn(p2, default=SENTINEL)

        assert len(caplog.records) == 2, (
            f'Two distinct paths should each emit one WARNING, got: {len(caplog.records)}'
        )


class TestNonUtf8Corruption:
    """Non-UTF-8 / binary garbage bytes are routed to the corrupt branch."""

    def test_non_utf8_bytes_returns_default_false_and_warns(self, tmp_path, caplog):
        """Binary garbage (non-UTF-8) returns (SENTINEL, False) with exactly one WARNING.

        The four bytes written are 0xFF, 0xFE, 0x00, followed by ASCII 'bad' — a
        UTF-16-LE BOM then junk — which is not valid UTF-8.

        RED: the current read phase calls p.read_text(encoding='utf-8'), which raises
        UnicodeDecodeError (a ValueError subclass, NOT an OSError) on these bytes.
        Because UnicodeDecodeError is only caught in the parse phase and the read
        phase only catches FileNotFoundError, the exception propagates uncaught.
        """
        from shared.safe_io import load_json_or_warn

        p = tmp_path / 'state.json'
        p.write_bytes(b'\xff\xfe\x00bad')  # UTF-16-LE BOM + junk — invalid UTF-8

        with caplog.at_level(logging.WARNING):
            result, ok = load_json_or_warn(p, default=SENTINEL)

        assert ok is False
        assert result is SENTINEL
        assert len(caplog.records) == 1, (
            f'Expected exactly one WARNING for non-UTF-8 bytes, got: {caplog.records}'
        )
        assert str(p) in caplog.records[0].message


class TestOSErrorPropagation:
    """Non-FileNotFoundError OS errors propagate uncaught (stay loud)."""

    def test_directory_path_raises_is_a_directory_error(self, tmp_path):
        """Passing a directory path raises IsADirectoryError — not laundered into a default.

        This locks in the three-way split: absent=benign, corrupt=warn, other-OS-error=propagate.
        """
        from shared.safe_io import load_json_or_warn

        # tmp_path itself is a directory; read_text on a directory raises IsADirectoryError.
        with pytest.raises(IsADirectoryError):
            load_json_or_warn(tmp_path, default=SENTINEL)


class TestEdgeCases:
    """Edge JSON values: empty file and whitespace are treated as corrupt (not absent)."""

    def test_empty_file_is_treated_as_corrupt(self, tmp_path, caplog):
        """Empty file returns (default, False) with a WARNING.

        json.loads('') raises ValueError, so an empty file lands in the corrupt
        branch — not the benign-absent branch.
        """
        from shared.safe_io import load_json_or_warn

        p = tmp_path / 'empty.json'
        p.write_text('')

        with caplog.at_level(logging.WARNING):
            result, ok = load_json_or_warn(p, default=SENTINEL)

        assert ok is False
        assert result is SENTINEL
        assert len(caplog.records) == 1, f'Expected one WARNING for empty file, got: {caplog.records}'
        assert str(p) in caplog.records[0].message


class TestAtomicWriteText:
    """Core contract of ``atomic_write_text``: content lands, nothing else does."""

    def test_writes_text_to_destination(self, tmp_path):
        """The destination holds exactly *text* after the call."""
        from shared.safe_io import atomic_write_text

        p = tmp_path / 'state.json'
        atomic_write_text(p, '{"a": 1}')

        assert p.read_text(encoding='utf-8') == '{"a": 1}'

    def test_overwrites_existing_content(self, tmp_path):
        """A pre-existing destination is replaced wholesale, not appended to."""
        from shared.safe_io import atomic_write_text

        p = tmp_path / 'state.json'
        p.write_text('OLD-CONTENT-THAT-IS-LONGER', encoding='utf-8')

        atomic_write_text(p, 'new')

        assert p.read_text(encoding='utf-8') == 'new'

    def test_happy_path_leaves_no_temp_file(self, tmp_path):
        """After a successful write the parent dir holds ONLY the destination."""
        from shared.safe_io import atomic_write_text

        p = tmp_path / 'state.json'
        atomic_write_text(p, 'payload')

        assert sorted(q.name for q in tmp_path.iterdir()) == ['state.json']

    def test_default_encoding_is_utf8(self, tmp_path):
        """Non-ASCII payload round-trips through the utf-8 default."""
        from shared.safe_io import atomic_write_text

        payload = 'héllo — wörld ✓ 日本語'
        p = tmp_path / 'state.json'
        atomic_write_text(p, payload)

        assert p.read_text(encoding='utf-8') == payload
        assert p.read_bytes() == payload.encode('utf-8')

    def test_accepts_str_path(self, tmp_path):
        """*path* accepts a plain ``str`` as well as ``os.PathLike``."""
        from shared.safe_io import atomic_write_text

        p = tmp_path / 'state.json'
        atomic_write_text(str(p), 'via-str')

        assert p.read_text(encoding='utf-8') == 'via-str'

    def test_exported_in_dunder_all(self):
        """``atomic_write_text`` is part of the module's public surface."""
        import shared.safe_io as _safe_io

        assert 'atomic_write_text' in _safe_io.__all__


class TestAtomicWriteFailurePaths:
    """On failure the destination is untouched and no temp residue survives."""

    def test_write_failure_propagates_unchanged(self, tmp_path, monkeypatch):
        """An exception raised mid-write reaches the caller — never swallowed."""
        import shared.safe_io as _safe_io

        p = tmp_path / 'state.json'
        p.write_text('ORIGINAL', encoding='utf-8')

        boom = RuntimeError('disk on fire')

        real_fdopen = os.fdopen

        def exploding_fdopen(fd, *a, **kw):
            f = real_fdopen(fd, *a, **kw)
            f.write = lambda _text: (_ for _ in ()).throw(boom)  # type: ignore[method-assign]
            return f

        monkeypatch.setattr(_safe_io.os, 'fdopen', exploding_fdopen)

        with pytest.raises(RuntimeError) as excinfo:
            _safe_io.atomic_write_text(p, 'NEW')

        assert excinfo.value is boom

    def test_write_failure_leaves_destination_intact(self, tmp_path, monkeypatch):
        """The pre-existing destination is byte-for-byte unchanged after a failure."""
        import shared.safe_io as _safe_io

        p = tmp_path / 'state.json'
        p.write_bytes(b'ORIGINAL-BYTES')

        real_fdopen = os.fdopen

        def exploding_fdopen(fd, *a, **kw):
            f = real_fdopen(fd, *a, **kw)
            f.write = lambda _text: (_ for _ in ()).throw(RuntimeError('boom'))  # type: ignore[method-assign]
            return f

        monkeypatch.setattr(_safe_io.os, 'fdopen', exploding_fdopen)

        with pytest.raises(RuntimeError):
            _safe_io.atomic_write_text(p, 'NEW')

        assert p.read_bytes() == b'ORIGINAL-BYTES'

    def test_write_failure_leaves_no_temp_file(self, tmp_path, monkeypatch):
        """A failed write cleans up its temp — the parent dir holds only the dest."""
        import shared.safe_io as _safe_io

        p = tmp_path / 'state.json'
        p.write_text('ORIGINAL', encoding='utf-8')

        real_fdopen = os.fdopen

        def exploding_fdopen(fd, *a, **kw):
            f = real_fdopen(fd, *a, **kw)
            f.write = lambda _text: (_ for _ in ()).throw(RuntimeError('boom'))  # type: ignore[method-assign]
            return f

        monkeypatch.setattr(_safe_io.os, 'fdopen', exploding_fdopen)

        with pytest.raises(RuntimeError):
            _safe_io.atomic_write_text(p, 'NEW')

        assert sorted(q.name for q in tmp_path.iterdir()) == ['state.json']

    def test_replace_failure_cleans_up_temp(self, tmp_path, monkeypatch):
        """An os.replace failure also unlinks the temp and re-raises."""
        import shared.safe_io as _safe_io

        p = tmp_path / 'state.json'
        p.write_text('ORIGINAL', encoding='utf-8')

        def exploding_replace(_src, _dst):
            raise OSError('rename failed')

        monkeypatch.setattr(_safe_io.os, 'replace', exploding_replace)

        with pytest.raises(OSError, match='rename failed'):
            _safe_io.atomic_write_text(p, 'NEW')

        assert p.read_text(encoding='utf-8') == 'ORIGINAL'
        assert sorted(q.name for q in tmp_path.iterdir()) == ['state.json']

    def test_base_exception_mid_write_also_cleans_up(self, tmp_path, monkeypatch):
        """A KeyboardInterrupt (BaseException) mid-write still removes the temp."""
        import shared.safe_io as _safe_io

        p = tmp_path / 'state.json'
        p.write_text('ORIGINAL', encoding='utf-8')

        real_fdopen = os.fdopen

        def exploding_fdopen(fd, *a, **kw):
            f = real_fdopen(fd, *a, **kw)
            f.write = lambda _text: (_ for _ in ()).throw(KeyboardInterrupt())  # type: ignore[method-assign]
            return f

        monkeypatch.setattr(_safe_io.os, 'fdopen', exploding_fdopen)

        with pytest.raises(KeyboardInterrupt):
            _safe_io.atomic_write_text(p, 'NEW')

        assert p.read_text(encoding='utf-8') == 'ORIGINAL'
        assert sorted(q.name for q in tmp_path.iterdir()) == ['state.json']


class TestAtomicWriteMode:
    """Permission contract — guards against the silent 0664 -> 0600 narrowing.

    ``tempfile.mkstemp`` creates 0600; ``open()``/``Path.write_text`` create
    0o666 & ~umask.  Consolidating every site onto one writer must NOT change
    the mode any site produces today, so ``mode`` is explicit per call.
    Expected values are computed from a live ``write_text`` reference file so
    the assertions stay umask-independent.
    """

    @staticmethod
    def _mode_of(p):
        return p.stat().st_mode & 0o777

    def test_default_mode_matches_write_text(self, tmp_path):
        """mode=None yields exactly what Path.write_text yields in the same dir."""
        from shared.safe_io import atomic_write_text

        reference = tmp_path / 'reference.json'
        reference.write_text('ref', encoding='utf-8')

        p = tmp_path / 'state.json'
        atomic_write_text(p, 'payload')

        assert self._mode_of(p) == self._mode_of(reference)

    def test_explicit_0600_is_exact(self, tmp_path):
        """mode=0o600 yields exactly 0o600 (the mkstemp-created sites' mode)."""
        from shared.safe_io import atomic_write_text

        p = tmp_path / 'state.json'
        atomic_write_text(p, 'payload', mode=0o600)

        assert self._mode_of(p) == 0o600

    def test_explicit_0644_survives_umask(self, tmp_path):
        """mode=0o644 is exact even when the umask would mask bits off."""
        from shared.safe_io import atomic_write_text

        old = os.umask(0o077)  # would strip group/other from an unguarded create
        try:
            p = tmp_path / 'state.json'
            atomic_write_text(p, 'payload', mode=0o644)
        finally:
            os.umask(old)

        assert self._mode_of(p) == 0o644

    def test_mode_applies_to_destination_after_replace(self, tmp_path):
        """The mode lands on the DESTINATION, not merely on the temp."""
        from shared.safe_io import atomic_write_text

        p = tmp_path / 'state.json'
        p.write_text('old', encoding='utf-8')
        os.chmod(p, 0o600)

        atomic_write_text(p, 'new', mode=0o644)

        assert p.read_text(encoding='utf-8') == 'new'
        assert self._mode_of(p) == 0o644


class TestAtomicWriteMkdir:
    """``mkdir`` opt-in: only the sites that create their parents today do so."""

    def test_mkdir_true_creates_missing_parents(self, tmp_path):
        """mkdir=True creates the full parent chain before writing."""
        from shared.safe_io import atomic_write_text

        p = tmp_path / 'a' / 'b' / 'state.json'
        assert not p.parent.exists()

        atomic_write_text(p, 'payload', mkdir=True)

        assert p.read_text(encoding='utf-8') == 'payload'

    def test_mkdir_true_is_idempotent(self, tmp_path):
        """mkdir=True against an existing parent is a no-op, not an error."""
        from shared.safe_io import atomic_write_text

        p = tmp_path / 'state.json'
        atomic_write_text(p, 'one', mkdir=True)
        atomic_write_text(p, 'two', mkdir=True)

        assert p.read_text(encoding='utf-8') == 'two'

    def test_mkdir_false_raises_on_missing_parent(self, tmp_path):
        """mkdir=False (default) surfaces the missing dir instead of creating it."""
        from shared.safe_io import atomic_write_text

        p = tmp_path / 'nope' / 'state.json'

        with pytest.raises(FileNotFoundError):
            atomic_write_text(p, 'payload')

        assert not p.parent.exists()


class TestAtomicWriteDurability:
    """``fsync`` opt-in — the durability contract ``landed_outbox`` depends on.

    Ordering is the substance here: fsyncing the parent directory BEFORE the
    rename provides no durability for that rename, so the test pins the temp-fd
    fsync as pre-replace and the dir-fd fsync as post-replace, not merely a
    call count.
    """

    @staticmethod
    def _record_calls(monkeypatch):
        """Patch os.fsync/os.replace on the module and return the event log."""
        import shared.safe_io as _safe_io

        events: list[str] = []
        real_fsync = os.fsync
        real_replace = os.replace

        def traced_fsync(fd):
            # A directory fd reports S_ISDIR; a regular temp file does not.
            kind = 'dir' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file'
            events.append(f'fsync:{kind}')
            return real_fsync(fd)

        def traced_replace(src, dst):
            events.append('replace')
            return real_replace(src, dst)

        monkeypatch.setattr(_safe_io.os, 'fsync', traced_fsync)
        monkeypatch.setattr(_safe_io.os, 'replace', traced_replace)
        return events

    def test_default_does_not_fsync(self, tmp_path, monkeypatch):
        """fsync=False (default) calls os.fsync zero times — nine sites rely on this."""
        from shared.safe_io import atomic_write_text

        events = self._record_calls(monkeypatch)
        atomic_write_text(tmp_path / 'state.json', 'payload')

        assert [e for e in events if e.startswith('fsync')] == []
        assert events == ['replace']

    def test_fsync_true_syncs_file_before_and_dir_after_replace(self, tmp_path, monkeypatch):
        """fsync=True: temp fd before the rename, parent dir fd after it."""
        from shared.safe_io import atomic_write_text

        events = self._record_calls(monkeypatch)
        p = tmp_path / 'state.json'
        atomic_write_text(p, 'payload', fsync=True)

        assert events == ['fsync:file', 'replace', 'fsync:dir']
        assert len([e for e in events if e.startswith('fsync')]) == 2
        assert p.read_text(encoding='utf-8') == 'payload'

    def test_dir_fd_closed_even_when_dir_fsync_raises(self, tmp_path, monkeypatch):
        """A failing directory fsync must not leak the fd it opened."""
        import shared.safe_io as _safe_io

        opened: list[int] = []
        closed: list[int] = []
        real_open = os.open
        real_close = os.close
        real_fsync = os.fsync

        def traced_open(path, flags, *a, **kw):
            fd = real_open(path, flags, *a, **kw)
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                opened.append(fd)
            return fd

        def traced_close(fd):
            closed.append(fd)
            return real_close(fd)

        def traced_fsync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError('dir fsync failed')
            return real_fsync(fd)

        monkeypatch.setattr(_safe_io.os, 'open', traced_open)
        monkeypatch.setattr(_safe_io.os, 'close', traced_close)
        monkeypatch.setattr(_safe_io.os, 'fsync', traced_fsync)

        with pytest.raises(OSError, match='dir fsync failed'):
            _safe_io.atomic_write_text(tmp_path / 'state.json', 'payload', fsync=True)

        assert opened, 'expected a directory fd to have been opened for the dir fsync'
        assert set(opened) <= set(closed), f'leaked dir fds: {set(opened) - set(closed)}'


class TestAtomicWriteRaceSafety:
    """The regression this consolidation exists to pin.

    Five of the migrated sites derived their temp name deterministically from
    the destination (``<dest>.json.tmp``), so two concurrent writers shared one
    temp and could interleave into a torn ``os.replace``.  These tests pin the
    properties that make that impossible.
    """

    def test_destination_never_holds_a_partial_write(self, tmp_path, monkeypatch):
        """At the instant before the rename the destination still holds the OLD content.

        Deterministic — no threads.  The payload is >1MiB so that a naive
        direct write would necessarily be chunked and observably partial.
        """
        import shared.safe_io as _safe_io

        old = 'OLD' * 8
        new = 'N' * (1024 * 1024 + 7)

        p = tmp_path / 'state.json'
        p.write_text(old, encoding='utf-8')

        observed: list[str] = []
        real_replace = os.replace

        def peeking_replace(src, dst):
            # Read the DESTINATION just before it is atomically swapped.
            observed.append(Path(dst).read_text(encoding='utf-8'))
            return real_replace(src, dst)

        monkeypatch.setattr(_safe_io.os, 'replace', peeking_replace)
        _safe_io.atomic_write_text(p, new)

        assert observed == [old], 'destination was mutated before the rename'
        assert p.read_text(encoding='utf-8') == new

    def test_repeated_writes_leave_exactly_one_file(self, tmp_path):
        """50 sequential writes to one destination leave no temp residue."""
        from shared.safe_io import atomic_write_text

        p = tmp_path / 'state.json'
        for i in range(50):
            atomic_write_text(p, f'payload-{i}')

        assert sorted(q.name for q in tmp_path.iterdir()) == ['state.json']
        assert p.read_text(encoding='utf-8') == 'payload-49'

    def test_temp_name_is_unique_per_writer(self, tmp_path, monkeypatch):
        """Concurrent writers to one destination never share a temp path.

        This is precisely what a fixed ``<dest>.json.tmp`` name cannot satisfy.
        """
        import shared.safe_io as _safe_io

        sources: list[str] = []
        lock = threading.Lock()
        real_replace = os.replace

        def recording_replace(src, dst):
            with lock:
                sources.append(str(src))
            return real_replace(src, dst)

        monkeypatch.setattr(_safe_io.os, 'replace', recording_replace)

        p = tmp_path / 'state.json'
        barrier = threading.Barrier(8)

        def writer(n):
            barrier.wait()
            _safe_io.atomic_write_text(p, f'payload-{n}')

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(sources) == 8
        assert len(set(sources)) == 8, f'temp names collided across writers: {sources}'

    def test_concurrent_writers_never_expose_torn_json(self, tmp_path):
        """A reader racing N writers only ever sees one writer's complete payload.

        Asserts completeness of whatever was observed — never a minimum number
        of observed reads — so scheduling can never make this flake.
        """
        from shared.safe_io import atomic_write_text

        p = tmp_path / 'state.json'
        payloads = {n: json.dumps({'writer': n, 'filler': 'x' * 50_000}) for n in range(6)}
        p.write_text(payloads[0], encoding='utf-8')

        stop = threading.Event()
        bad: list[str] = []
        seen: list[int] = []

        def reader():
            while not stop.is_set():
                try:
                    raw = p.read_text(encoding='utf-8')
                except FileNotFoundError:
                    bad.append('destination vanished mid-write')
                    continue
                try:
                    seen.append(json.loads(raw)['writer'])
                except ValueError:
                    bad.append(f'torn read: {raw[:60]!r}... len={len(raw)}')

        def writer(n):
            for _ in range(20):
                atomic_write_text(p, payloads[n])

        r = threading.Thread(target=reader, daemon=True)
        r.start()
        threads = [threading.Thread(target=writer, args=(n,)) for n in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stop.set()
        r.join(timeout=10)

        assert bad == [], f'reader observed incomplete state: {bad[:3]}'
        assert all(w in payloads for w in seen), 'reader saw a payload no writer wrote'
        assert p.read_text(encoding='utf-8') in payloads.values()
        assert sorted(q.name for q in tmp_path.iterdir()) == ['state.json']


# ---------------------------------------------------------------------------
# Anti-regrowth guard (task 3223)
# ---------------------------------------------------------------------------
#
# Task 3223 consolidated ten hand-rolled tmp+rename writers into
# ``atomic_write_text`` above.  Nothing stops the eleventh from being written
# by hand next month — the pattern is short enough to look harmless, which is
# exactly how the first ten accumulated (two of them carried a docstring
# saying they were copied because "there is no atomic writer in shared/ to
# reuse").  These tests are the fence: a NEW rename-into-place anywhere in the
# three source trees fails loudly and points the author at this module.

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_TREES = ('shared/src', 'orchestrator/src', 'escalation/src')

# Every (module, function) in the three trees that renames a path into place,
# with the reason it is not calling atomic_write_text.  Adding an entry is a
# deliberate act that needs a reason; growing this set silently is the failure
# mode the guard exists to prevent.
_ALLOWED_RENAMERS = {
    ('shared/src/shared/safe_io.py', 'atomic_write_text'):
        'THE consolidated implementation — the one blessed home for this pattern.',
    ('shared/src/shared/safe_io.py', 'load_json_or_warn'):
        'Quarantine rename of a corrupt file (<name>.corrupt); not a write path.',
    ('orchestrator/src/orchestrator/session_hooks.py', '_run_install'):
        'Out of task 3223 enumerated scope; left alone deliberately.',
    ('orchestrator/src/orchestrator/verify_cancel.py', 'write_pgid_file'):
        'Out of task 3223 enumerated scope; left alone deliberately.',
    ('escalation/src/escalation/queue.py', 'EscalationQueue._atomic_write_path'):
        'Out of task 3223 enumerated scope. This is the helper 3223 was MODELLED '
        'on (its durable= flag became fsync=); a follow-up may migrate it.',
    ('escalation/src/escalation/queue.py', 'EscalationQueue._archive_resolved'):
        'Moves a resolved file into the dated archive dir; not a write path.',
    ('escalation/src/escalation/sweep.py', '_atomic_move'):
        'Out of task 3223 enumerated scope; moves an existing file, does not write.',
    ('orchestrator/src/orchestrator/digest.py', 'write_digest_entry'):
        'Out of task 3223 enumerated scope. Notable: b3_gate._save_state was '
        'originally documented as "modelled on digest.py" — this is where one '
        'of the ten copies came from, so it is a prime candidate for the '
        'follow-up migration.',
    ('orchestrator/src/orchestrator/evals/rereview.py', 'atomic_write_json'):
        'Out of task 3223 enumerated scope; left alone deliberately.',
    (
        'orchestrator/src/orchestrator/service_restart.py',
        'StaleServiceRestartCoordinator._persist_last_fire_wall',
    ):
        'Out of task 3223 enumerated scope; left alone deliberately.',
    ('orchestrator/src/orchestrator/agents/invoke.py', '_invoke_pi'):
        'Swaps .mcp.json with a backup and back again; renames existing files '
        'rather than writing new content, so atomic_write_text does not apply.',
}


def _find_renamers(source: str) -> list[str]:
    """Return the qualified names of functions in *source* that rename a path.

    Covers all four spellings the repo actually uses, because a scan for any
    one of them leaves a hole big enough to hide a copy in:

    * ``os.replace(tmp, dest)`` — eight of the ten sites 3223 consolidated.
    * ``os.rename(tmp, dest)`` — ``escalation.queue._atomic_write_path``.
    * ``tmp.replace(dest)`` — ``evals.rereview``, ``service_restart``.
    * ``tmp.rename(dest)`` — ``digest.write_digest_entry``, which is the
      writer ``b3_gate._save_state`` was documented as modelled on.

    The two ``Path``-method forms are matched by arity: ``Path.replace`` and
    ``Path.rename`` take exactly one positional argument, while the far more
    common ``str.replace(old, new)`` takes two, so requiring exactly one
    positional arg and no keywords separates them without a type-inference
    pass.

    AST-based rather than a text grep on purpose: six of the migrated modules
    still *mention* ``os.replace`` in a docstring describing what
    ``atomic_write_text`` does for them, and a text scan would flag all six as
    false positives.

    Limitation, stated rather than papered over: this finds the rename, which
    is the half of the pattern that cannot be omitted.  A copy that factored
    its rename out into a helper would be attributed to that helper instead.
    """
    import ast

    found: list[str] = []

    def is_rename(sub) -> bool:
        if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)):
            return False
        if sub.func.attr not in ('replace', 'rename'):
            return False
        receiver = sub.func.value
        if isinstance(receiver, ast.Name) and receiver.id == 'os':
            return True
        # Path.replace(target) / Path.rename(target): exactly one positional
        # argument.  str.replace(old, new) takes two and is excluded here.
        return len(sub.args) == 1 and not sub.keywords

    def visit(node, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f'{prefix}{child.name}.')
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                qualname = f'{prefix}{child.name}'
                if any(is_rename(sub) for sub in ast.walk(child)):
                    found.append(qualname)
                visit(child, f'{qualname}.')
            else:
                visit(child, prefix)

    visit(ast.parse(source), '')
    return found


def _iter_source_files():
    """Yield (repo-relative posix path, source text) for the three src trees."""
    for tree in _SRC_TREES:
        root = _REPO_ROOT / tree
        assert root.is_dir(), (
            f'{tree} not found under {_REPO_ROOT} — this guard walks fixed tree '
            f'names, so a moved/renamed package must update _SRC_TREES rather '
            f'than let the guard silently scan nothing.'
        )
        for py in sorted(root.rglob('*.py')):
            yield py.relative_to(_REPO_ROOT).as_posix(), py.read_text(encoding='utf-8')


class TestNoRegrownAtomicWriters:
    """The consolidated pattern cannot silently re-duplicate."""

    def test_detector_fires_on_a_regrown_copy(self):
        """The detector has teeth: it flags a re-inlined tmp+rename block.

        Without this, a detector that silently matched nothing would make
        every other test in this class pass vacuously.  The sample below is
        the exact shape task 3223 removed from five call sites.
        """
        regrown = (
            'import json, os\n'
            'def _save_raw(self, state):\n'
            '    tmp = self._path.with_suffix(".json.tmp")\n'
            '    tmp.write_text(json.dumps(state), encoding="utf-8")\n'
            '    os.replace(str(tmp), str(self._path))\n'
        )
        assert _find_renamers(regrown) == ['_save_raw']

        os_rename_variant = (
            'import os\n'
            'class S:\n'
            '    def _write(self, path, text):\n'
            '        os.rename(tmp, path)\n'
        )
        assert _find_renamers(os_rename_variant) == ['S._write']

        # Path-method spellings: the hole that let digest.write_digest_entry,
        # rereview.atomic_write_json and service_restart sit unseen by an
        # os.replace-only scan.
        path_method_variant = (
            'def _write(path, text):\n'
            '    tmp = path.with_suffix(".tmp")\n'
            '    tmp.write_text(text)\n'
            '    tmp.rename(path)\n'
        )
        assert _find_renamers(path_method_variant) == ['_write']

        path_replace_variant = (
            'def _write(path, text):\n'
            '    tmp.replace(path)\n'
        )
        assert _find_renamers(path_replace_variant) == ['_write']

    def test_detector_ignores_str_replace(self):
        """``str.replace(old, new)`` is not a rename.

        The Path-method branch keys on arity, so this is the false positive
        that would fire on half the repo if the arity check regressed.
        """
        two_arg = (
            'def normalise(s):\n'
            '    return s.replace("-", "_").replace(" ", "")\n'
        )
        assert _find_renamers(two_arg) == []

    def test_detector_ignores_docstring_mentions(self):
        """A docstring describing the pattern is not an implementation of it.

        Six migrated modules still reference ``os.replace`` in prose; flagging
        those would make the guard unusable and train people to disable it.
        """
        prose_only = (
            'def _save_raw(self, state):\n'
            '    """Delegates to safe_io.atomic_write_text (tmp + os.replace)."""\n'
            '    safe_io.atomic_write_text(self._path, state)\n'
        )
        assert _find_renamers(prose_only) == []

    def test_no_unapproved_renamers_in_source_trees(self):
        """Every rename-into-place in the three trees is a known, reasoned survivor."""
        actual = {
            (relpath, qualname)
            for relpath, source in _iter_source_files()
            for qualname in _find_renamers(source)
        }

        unapproved = actual - set(_ALLOWED_RENAMERS)
        assert not unapproved, (
            'New hand-rolled rename-into-place found:\n  '
            + '\n  '.join(f'{f}::{q}' for f, q in sorted(unapproved))
            + '\nUse shared.safe_io.atomic_write_text instead. If this site '
            'genuinely cannot (it moves an existing file rather than writing '
            'one, say), add it to _ALLOWED_RENAMERS with the reason.'
        )

        stale = set(_ALLOWED_RENAMERS) - actual
        assert not stale, (
            'Stale _ALLOWED_RENAMERS entries (site is gone or no longer '
            f'renames): {sorted(stale)}. Remove them so the allowlist keeps '
            'describing reality.'
        )

    def test_atomic_write_text_helpers_only_delegate(self):
        """The four surviving ``_atomic_write_text`` names must stay one-liners.

        Task 3223 kept these module-level names (test_prompt_artifact.py
        monkeypatches one at five sites) but emptied their bodies down to a
        delegation.  Re-inlining a real implementation under the old name is
        the most likely way the duplication comes back, because the name would
        still look consolidated from every call site.
        """
        import ast

        offenders = []
        for relpath, source in _iter_source_files():
            if relpath == 'shared/src/shared/safe_io.py':
                continue  # the blessed implementation lives here
            for node in ast.walk(ast.parse(source)):
                if (
                    isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                    and node.name in ('_atomic_write_text', 'atomic_write_text')
                ):
                    body = ast.get_source_segment(source, node) or ''
                    if 'safe_io.atomic_write_text(' not in body:
                        offenders.append(f'{relpath}::{node.name} does not delegate')

        assert not offenders, (
            'These helpers stopped delegating to shared.safe_io.atomic_write_text '
            'and re-inlined their own implementation:\n  ' + '\n  '.join(offenders)
        )
