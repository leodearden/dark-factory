"""Tests for shared.transcript_archive — best-effort per-task transcript archival."""

from __future__ import annotations

import errno
import logging
import os
from pathlib import Path

from shared import transcript_archive as transcript_archive_module
from shared.transcript_archive import archive_task_transcripts

# A representative encoded-project directory name (Claude Code encodes the
# absolute project path into this leaf; the exact encoding is irrelevant here).
ENC = '-home-leo-src-dark-factory'


def _write(path: Path, data: bytes) -> Path:
    """Create *path* (and parents) and write *data*; return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class TestCoreArchiveAndCredentialSafety:
    """E1 (core archive) + E4 (credential-safety)."""

    def test_archives_main_and_subagent_but_never_credentials(self, tmp_path):
        config_dir = tmp_path / 'claude-config-42'
        root = tmp_path / 'archive'
        sid = 'sess-aaaa'
        task_id = '42'

        main_bytes = b'{"type":"main","line":1}\n{"type":"main","line":2}\n'
        sub_bytes = b'{"type":"subagent","line":1}\n'
        creds_bytes = b'{"claudeAiOauth":{"accessToken":"SECRET"}}'
        notes_bytes = b'not a transcript'

        _write(config_dir / 'projects' / ENC / f'{sid}.jsonl', main_bytes)
        _write(
            config_dir / 'projects' / ENC / sid / 'subagents' / 'agent-1.jsonl',
            sub_bytes,
        )
        # .credentials.json at the config-dir ROOT (outside projects/).
        _write(config_dir / '.credentials.json', creds_bytes)
        # A non-transcript file under projects/ (must not be archived).
        _write(config_dir / 'projects' / ENC / 'notes.txt', notes_bytes)

        count = archive_task_transcripts(config_dir, task_id, sid, archive_root=root)

        assert count == 2

        main_dest = root / task_id / ENC / f'{sid}.jsonl'
        sub_dest = root / task_id / ENC / sid / 'subagents' / 'agent-1.jsonl'
        assert main_dest.exists()
        assert sub_dest.exists()
        assert main_dest.read_bytes() == main_bytes
        assert sub_dest.read_bytes() == sub_bytes

        # Credential-safety: nothing matching the credential or the non-jsonl
        # file appears anywhere under the archive root.
        archived = [p.name for p in root.rglob('*') if p.is_file()]
        assert not any('credentials' in name for name in archived)
        assert not any(name.startswith('notes') for name in archived)
        assert not any(name.endswith('.txt') for name in archived)


class TestArchiveAll:
    """session_id=None archives every projects/**/*.jsonl."""

    def test_archives_every_session_when_session_id_none(self, tmp_path):
        config_dir = tmp_path / 'claude-config-42'
        root = tmp_path / 'archive'
        task_id = '42'

        enc_a = '-home-leo-projA'
        enc_b = '-home-leo-projB'
        sid_a = 'sess-a'
        sid_b = 'sess-b'

        a_bytes = b'{"s":"a"}\n'
        b_bytes = b'{"s":"b"}\n'
        sub_bytes = b'{"s":"a-sub"}\n'

        _write(config_dir / 'projects' / enc_a / f'{sid_a}.jsonl', a_bytes)
        _write(config_dir / 'projects' / enc_b / f'{sid_b}.jsonl', b_bytes)
        _write(
            config_dir / 'projects' / enc_a / sid_a / 'subagents' / 'agent-1.jsonl',
            sub_bytes,
        )
        # A non-transcript file must still be ignored by the recursive glob.
        _write(config_dir / 'projects' / enc_a / 'notes.txt', b'ignore me')

        count = archive_task_transcripts(config_dir, task_id, None, archive_root=root)

        assert count == 3

        a_dest = root / task_id / enc_a / f'{sid_a}.jsonl'
        b_dest = root / task_id / enc_b / f'{sid_b}.jsonl'
        sub_dest = root / task_id / enc_a / sid_a / 'subagents' / 'agent-1.jsonl'
        assert a_dest.read_bytes() == a_bytes
        assert b_dest.read_bytes() == b_bytes
        assert sub_dest.read_bytes() == sub_bytes

        archived = [p.name for p in root.rglob('*') if p.is_file()]
        assert not any(name.endswith('.txt') for name in archived)


class TestIdempotencyAndResume:
    """E3 (idempotency skip on unchanged source) + E6 (resume last-write-wins)."""

    def test_skip_unchanged_then_rearchive_grown(self, tmp_path):
        config_dir = tmp_path / 'claude-config-42'
        root = tmp_path / 'archive'
        task_id = '42'
        sid = 'sess-x'

        src = config_dir / 'projects' / ENC / f'{sid}.jsonl'
        orig = b'{"line":1}\n'
        _write(src, orig)

        # First call writes the archive.
        assert archive_task_transcripts(config_dir, task_id, sid, archive_root=root) == 1
        dest = root / task_id / ENC / f'{sid}.jsonl'
        assert dest.read_bytes() == orig

        # Robustly prove "not rewritten": replace the archive bytes with a
        # sentinel and restore its mtime to the source's, so the skip predicate
        # (int(dest.mtime) == int(src.mtime)) still holds. A genuine skip leaves
        # the sentinel intact; an unconditional rewrite replaces it with the
        # source bytes.
        src_stat = src.stat()
        sentinel = b'SENTINEL-not-rewritten'
        dest.write_bytes(sentinel)
        os.utime(dest, (src_stat.st_atime, src_stat.st_mtime))
        before_mtime_ns = dest.stat().st_mtime_ns

        # Second call, source unchanged → skipped (returns 0, archive untouched).
        assert archive_task_transcripts(config_dir, task_id, sid, archive_root=root) == 0
        assert dest.read_bytes() == sentinel
        assert dest.stat().st_mtime_ns == before_mtime_ns

        # Resume: grow the source AND advance its mtime → re-archived.
        grown = orig + b'{"line":2}\n{"line":3}\n'
        src.write_bytes(grown)
        os.utime(src, (src_stat.st_atime + 10, src_stat.st_mtime + 10))

        assert archive_task_transcripts(config_dir, task_id, sid, archive_root=root) == 1
        assert dest.read_bytes() == grown


class TestBestEffortLoud:
    """E7: a per-file failure is caught, counted, and logged; never raises."""

    def test_partial_failure_counted_logged_not_raised(self, tmp_path, caplog):
        config_dir = tmp_path / 'claude-config-42'
        root = tmp_path / 'archive'
        task_id = '42'

        good = config_dir / 'projects' / ENC / 'sess-good.jsonl'
        bad = config_dir / 'projects' / ENC / 'sess-bad.jsonl'
        good_bytes = b'{"ok":1}\n'
        _write(good, good_bytes)
        _write(bad, b'{"bad":1}\n')

        # Induce a per-file write failure: pre-create a DIRECTORY at the bad
        # file's dest path so the writer's copy raises IsADirectoryError.
        bad_dest = root / task_id / ENC / 'sess-bad.jsonl'
        bad_dest.mkdir(parents=True)
        # Ensure the idempotency check does NOT skip it (dir mtime != src mtime).
        os.utime(bad_dest, (0, 0))

        transcript_archive_module._reset_archival_failures()

        with caplog.at_level(logging.WARNING, logger='shared.transcript_archive'):
            count = archive_task_transcripts(config_dir, task_id, None, archive_root=root)

        # Partial success — the good file archived, the bad one did not.
        assert count == 1
        good_dest = root / task_id / ENC / 'sess-good.jsonl'
        assert good_dest.read_bytes() == good_bytes

        # Failure counted (loud-over-silent).
        assert transcript_archive_module._archival_failures() == 1

        # Exactly one structured WARNING carrying path, task_id, errno.
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        rec = warnings[0]
        assert rec.task_id == task_id
        assert str(bad) in rec.path
        assert rec.errno == errno.EISDIR

    @staticmethod
    def _copyfile_dies_part_way(monkeypatch, prefix: bytes):
        """Make ``shutil.copyfile`` write *prefix* to its dest, then ENOSPC.

        The real shape of an interrupted copy: bytes land on disk and THEN the
        write fails. A mock that raised before writing anything would pass
        against an in-place copy too, and prove nothing about where the partial
        bytes went.
        """
        def fake_copyfile(src, dst, **kwargs):
            Path(dst).write_bytes(prefix)
            raise OSError(errno.ENOSPC, 'No space left on device')

        monkeypatch.setattr(
            transcript_archive_module.shutil, 'copyfile', fake_copyfile
        )

    def test_an_interrupted_copy_publishes_no_partial_transcript(
        self, tmp_path, monkeypatch, caplog
    ):
        # A truncated PLAIN .jsonl at the canonical archive path is SILENT
        # damage: every reader's *.jsonl glob admits it as an ordinary
        # transcript and iter_json_lines skips the partial trailing line, so
        # the session under-reports with nothing counting it. The staged write
        # is what keeps the failure loud (counted + logged) instead.
        config_dir = tmp_path / 'claude-config-42'
        root = tmp_path / 'archive'
        task_id = '42'
        sid = 'sess-killed'

        src = config_dir / 'projects' / ENC / f'{sid}.jsonl'
        _write(src, b'{"line":1}\n{"line":2}\n{"line":3}\n')
        self._copyfile_dies_part_way(monkeypatch, b'{"line":1}\n{"li')

        transcript_archive_module._reset_archival_failures()

        with caplog.at_level(logging.WARNING, logger='shared.transcript_archive'):
            count = archive_task_transcripts(
                config_dir, task_id, sid, archive_root=root
            )

        assert count == 0
        dest = root / task_id / ENC / f'{sid}.jsonl'
        assert not dest.exists()
        # Nothing at all is left behind — not the partial transcript, and not
        # the staging file it was written to.
        assert [p for p in root.rglob('*') if p.is_file()] == []

        # Loud, not silent: counted and logged with the real errno.
        assert transcript_archive_module._archival_failures() == 1
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert warnings[0].errno == errno.ENOSPC

    def test_an_interrupted_copy_does_not_truncate_the_previous_archive(
        self, tmp_path, monkeypatch
    ):
        # The resumed-session shape: an archived copy already exists and the
        # grown source is being re-archived when the copy dies. An in-place
        # copy would truncate the only durable copy of the earlier records;
        # staging leaves it byte-for-byte intact, mtime included.
        config_dir = tmp_path / 'claude-config-42'
        root = tmp_path / 'archive'
        task_id = '42'
        sid = 'sess-grown'

        src = config_dir / 'projects' / ENC / f'{sid}.jsonl'
        orig = b'{"line":1}\n'
        _write(src, orig)
        assert archive_task_transcripts(
            config_dir, task_id, sid, archive_root=root
        ) == 1
        dest = root / task_id / ENC / f'{sid}.jsonl'
        before_mtime_ns = dest.stat().st_mtime_ns

        # Grow the source (mtime advances, so the idempotency skip does not
        # apply) and kill the copy part-way through.
        src_stat = src.stat()
        src.write_bytes(orig + b'{"line":2}\n')
        os.utime(src, (src_stat.st_atime + 10, src_stat.st_mtime + 10))
        self._copyfile_dies_part_way(monkeypatch, b'{"line":1}\n{"li')

        transcript_archive_module._reset_archival_failures()
        count = archive_task_transcripts(
            config_dir, task_id, sid, archive_root=root
        )

        assert count == 0
        assert dest.read_bytes() == orig
        assert dest.stat().st_mtime_ns == before_mtime_ns
        assert [p.name for p in root.rglob('*') if p.is_file()] == [dest.name]
        assert transcript_archive_module._archival_failures() == 1
