"""Tests for shared.transcript_archive — best-effort per-task transcript archival."""

from __future__ import annotations

import errno
import gzip
import logging
import os
from pathlib import Path

from shared import transcript_archive as transcript_archive_module
from shared.transcript_archive import archive_task_transcripts, durable_archive_path

# A representative encoded-project directory name (Claude Code encodes the
# absolute project path into this leaf; the exact encoding is irrelevant here).
ENC = '-home-leo-src-dark-factory'

# A SECOND encoded-project dir: the same session archived from a different
# worktree lane (the observed re-dispatched-across-lanes case). ENC is a strict
# prefix of ENC_B, so ENC sorts FIRST lexically — the multi-match rows below
# rely on that known order to make "first match" and "newest mtime" differ.
ENC_B = '-home-leo-src-dark-factory--worktrees-3727'


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
        # file's dest .gz path so gzip.open(dest, 'wb') raises IsADirectoryError.
        bad_dest = root / task_id / ENC / 'sess-bad.jsonl.gz'
        bad_dest.mkdir(parents=True)
        # Ensure the idempotency check does NOT skip it (dir mtime != src mtime).
        os.utime(bad_dest, (0, 0))

        transcript_archive_module._reset_archival_failures()

        with caplog.at_level(logging.WARNING, logger='shared.transcript_archive'):
            count = archive_task_transcripts(config_dir, task_id, None, archive_root=root)

        # Partial success — the good file archived, the bad one did not.
        assert count == 1
        good_dest = root / task_id / ENC / 'sess-good.jsonl.gz'
        assert _gunzip(good_dest) == good_bytes

        # Failure counted (loud-over-silent).
        assert transcript_archive_module._archival_failures() == 1

        # Exactly one structured WARNING carrying path, task_id, errno.
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        rec = warnings[0]
        assert rec.task_id == task_id
        assert str(bad) in rec.path
        assert rec.errno == errno.EISDIR


class TestDurableArchivePathLookup:
    """B1/B2/B4 — the read side: locate one session's archived transcript.

    See plans/session-resume-eligibility-seam-prd.md §8 for the contract this
    class pins. The archive layout asserted here is read off the producer
    (:func:`_archive_one`), not off prose, so locator and writer cannot drift.
    """

    def test_b1_same_lane_returns_the_archived_transcript(self, tmp_path):
        """B1: an archive written under the caller's own lane is found."""
        root = tmp_path / 'archive'
        sid = 'sess-b1'
        task_id = '42'

        archived = _write(root / task_id / ENC / f'{sid}.jsonl.gz', b'gz-bytes')

        found = durable_archive_path(root, task_id, sid)

        assert found == archived
        # D6: the locator returns a Path, never a bool — "does it exist" is
        # spelled `is not None`, and 3578's restore reuses this same path
        # rather than growing a second glob that must agree forever (INV-5).
        assert isinstance(found, Path)
        assert not isinstance(found, bool)
        assert found.exists()

    def test_b2_cross_lane_archive_is_found(self, tmp_path):
        """B2 (I-B): the encoded-cwd component is globbed, never assumed.

        The MEASURED reify case: a session archived from lane A is looked up
        after re-dispatch into lane B, so the caller knows nothing about which
        encoded-cwd dir produced it.
        """
        root = tmp_path / 'archive'
        sid = 'sess-b2'
        task_id = '42'

        # Written ONLY under ENC_B; the caller passes no lane information.
        archived = _write(root / task_id / ENC_B / f'{sid}.jsonl.gz', b'gz-bytes')

        found = durable_archive_path(root, task_id, sid)

        assert found == archived
        assert found.exists()

    def test_b4_absent_session_returns_none(self, tmp_path):
        """B4: a populated root that holds a DIFFERENT session yields None."""
        root = tmp_path / 'archive'
        task_id = '42'

        _write(root / task_id / ENC / 'sess-other.jsonl.gz', b'gz-bytes')

        assert durable_archive_path(root, task_id, 'sess-missing') is None

    def test_b4_task_id_component_is_not_globbed(self, tmp_path):
        """B4: the task_id is an exact path component, not a wildcard.

        A session archived under task 42 must NOT be reported as recoverable
        when task 99 asks — the archive is keyed on (task_id, session_id).
        """
        root = tmp_path / 'archive'
        sid = 'sess-b4'

        _write(root / '42' / ENC / f'{sid}.jsonl.gz', b'gz-bytes')

        assert durable_archive_path(root, '42', sid) is not None
        assert durable_archive_path(root, '99', sid) is None

    def test_b3_uncompressed_archive_is_found(self, tmp_path):
        """B3 (I-C): the post-task-3618 plain-``.jsonl`` shape is located too."""
        root = tmp_path / 'archive'
        sid = 'sess-b3'
        task_id = '42'

        archived = _write(root / task_id / ENC / f'{sid}.jsonl', b'plain-bytes')

        assert durable_archive_path(root, task_id, sid) == archived

    def test_b3_both_formats_coexist_under_one_root(self, tmp_path):
        """B3 (I-C): no flag day — the two shapes may sit side by side.

        Task 3618 drops the gzip; during and after that cutover an archive
        root holds both shapes for different sessions, and each lookup must
        return its OWN file. This is exactly why this task carries no
        dependency on 3618.
        """
        root = tmp_path / 'archive'
        task_id = '42'

        gz = _write(root / task_id / ENC / 'sess-old.jsonl.gz', b'gz-bytes')
        plain = _write(root / task_id / ENC / 'sess-new.jsonl', b'plain-bytes')

        assert durable_archive_path(root, task_id, 'sess-old') == gz
        assert durable_archive_path(root, task_id, 'sess-new') == plain

    def test_b3_subagent_directory_is_never_returned(self, tmp_path):
        """The ``<session_id>/`` subagent DIRECTORY is not "the transcript".

        :func:`_archive_one` mirrors subagent transcripts under a directory
        NAMED for the session (``<enc>/<sid>/subagents/agent-*.jsonl.gz``).
        That dir carries no ``.jsonl`` suffix so today's pattern misses it by
        luck; the ``is_file()`` filter makes returning it structurally
        impossible.
        """
        root = tmp_path / 'archive'
        sid = 'sess-b3-decoy'
        task_id = '42'

        main = _write(root / task_id / ENC / f'{sid}.jsonl.gz', b'gz-bytes')
        _write(
            root / task_id / ENC / sid / 'subagents' / 'agent-1.jsonl.gz',
            b'sub-bytes',
        )

        found = durable_archive_path(root, task_id, sid)

        assert found == main
        assert found.is_file()

    def test_b6_newest_mtime_wins_across_lanes(self, tmp_path):
        """B6 (I-F): two lanes archived the same session — newest wins.

        The observed re-dispatched-across-lanes case (reify tasks 5848/5766).
        The OLDER copy is written first and lives under the lexically-earlier
        lane dir, so a naive first-match glob returns exactly the wrong one.
        Newest rather than oldest because a resumed session's transcript only
        ever grows, so the newest archive is the most complete one.
        """
        root = tmp_path / 'archive'
        sid = 'sess-b6'
        task_id = '42'

        older = _write(root / task_id / ENC / f'{sid}.jsonl.gz', b'older')
        newer = _write(root / task_id / ENC_B / f'{sid}.jsonl.gz', b'newer')
        os.utime(older, (1_000_000, 1_000_000))
        os.utime(newer, (2_000_000, 2_000_000))

        assert durable_archive_path(root, task_id, sid) == newer

    def test_b6_repeat_calls_are_stable(self, tmp_path):
        """B6 (I-F): the same question gets the same answer every time."""
        root = tmp_path / 'archive'
        sid = 'sess-b6-stable'
        task_id = '42'

        older = _write(root / task_id / ENC / f'{sid}.jsonl.gz', b'older')
        newer = _write(root / task_id / ENC_B / f'{sid}.jsonl.gz', b'newer')
        os.utime(older, (1_000_000, 1_000_000))
        os.utime(newer, (2_000_000, 2_000_000))

        answers = {durable_archive_path(root, task_id, sid) for _ in range(5)}

        assert answers == {newer}

    def test_b6_equal_mtimes_still_resolve_deterministically(self, tmp_path):
        """B6 (I-F): an exact mtime tie still yields one stable answer.

        Exact ties are plausible rather than theoretical: _archive_one mirrors
        the SOURCE mtime onto the archived copy via os.utime, so one session
        archived from two lanes can tie to the nanosecond. Under a tie, mtime
        alone is not a total order and max() falls back to filesystem-dependent
        glob iteration order — non-reproducible across calls and machines, and
        any resume built on it non-reproducible with it.
        """
        root = tmp_path / 'archive'
        sid = 'sess-b6-tie'
        task_id = '42'

        a = _write(root / task_id / ENC / f'{sid}.jsonl.gz', b'lane-a')
        b = _write(root / task_id / ENC_B / f'{sid}.jsonl.gz', b'lane-b')
        os.utime(a, (1_500_000, 1_500_000))
        os.utime(b, (1_500_000, 1_500_000))

        answers = {durable_archive_path(root, task_id, sid) for _ in range(5)}

        assert len(answers) == 1
        assert answers.pop() in {a, b}
