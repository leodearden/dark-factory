"""Tests for shared.transcript_archive — best-effort per-task transcript archival."""

from __future__ import annotations

import gzip
import os
from pathlib import Path

import pytest

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


def _gunzip(path: Path) -> bytes:
    with gzip.open(path, 'rb') as fh:
        return fh.read()


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

        main_gz = root / task_id / ENC / f'{sid}.jsonl.gz'
        sub_gz = root / task_id / ENC / sid / 'subagents' / 'agent-1.jsonl.gz'
        assert main_gz.exists()
        assert sub_gz.exists()
        assert _gunzip(main_gz) == main_bytes
        assert _gunzip(sub_gz) == sub_bytes

        # Credential-safety: nothing matching the credential or the non-jsonl
        # file appears anywhere under the archive root.
        archived = [p.name for p in root.rglob('*') if p.is_file()]
        assert not any('credentials' in name for name in archived)
        assert not any(name.startswith('notes') for name in archived)
        assert not any(name.endswith('.txt') for name in archived)
        assert not any(name.endswith('.txt.gz') for name in archived)


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

        a_gz = root / task_id / enc_a / f'{sid_a}.jsonl.gz'
        b_gz = root / task_id / enc_b / f'{sid_b}.jsonl.gz'
        sub_gz = root / task_id / enc_a / sid_a / 'subagents' / 'agent-1.jsonl.gz'
        assert _gunzip(a_gz) == a_bytes
        assert _gunzip(b_gz) == b_bytes
        assert _gunzip(sub_gz) == sub_bytes

        archived = [p.name for p in root.rglob('*') if p.is_file()]
        assert not any(name.endswith('.txt') or name.endswith('.txt.gz') for name in archived)


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
        dest = root / task_id / ENC / f'{sid}.jsonl.gz'
        assert _gunzip(dest) == orig

        # Robustly prove "not rewritten": replace the archive bytes with a
        # sentinel and restore its mtime to the source's, so the skip predicate
        # (int(dest.mtime) == int(src.mtime)) still holds. A genuine skip leaves
        # the sentinel intact; an unconditional rewrite replaces it with gzip.
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
        assert _gunzip(dest) == grown
