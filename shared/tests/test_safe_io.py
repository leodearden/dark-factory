"""Tests for shared.safe_io.load_json_or_warn."""

from __future__ import annotations

import json
import logging
import os

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
