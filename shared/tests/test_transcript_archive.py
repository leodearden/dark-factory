"""Tests for shared.transcript_archive — best-effort per-task transcript archival."""

from __future__ import annotations

import errno
import logging
import os
from pathlib import Path

import pytest

from shared import transcript_archive as transcript_archive_module
from shared.transcript_archive import archive_task_transcripts, durable_archive_path

# A representative encoded-project directory name (Claude Code encodes the
# absolute project path into this leaf; the exact encoding is irrelevant here).
ENC = '-home-leo-src-dark-factory'

# A SECOND encoded-project dir: the same session archived from a different
# worktree lane (the observed re-dispatched-across-lanes case). ENC is a strict
# prefix of ENC_B, so ENC sorts FIRST lexically — the multi-match rows below
# rely on that known order to make "first match" and "newest mtime" differ.
#
# CAREFUL, and MEASURED rather than reasoned: that dirname order INVERTS once
# the FULL path is compared, which is what I-F's `str(p)` tiebreak actually
# keys on. After the shared prefix, the ENC path continues with the separator
# '/' (0x2F) while the ENC_B path continues with '-' (0x2D), so
# str(<...>/ENC/<sid>) > str(<...>/ENC_B/<sid>) and the tiebreak names the ENC
# copy. The tie rows below assert that specific winner, so the direction is
# stated here once instead of being re-derived (wrongly) at each call site.
ENC_B = '-home-leo-src-dark-factory--worktrees-3727'


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

        # D6: the locator returns the PATH, not a bool — "does it exist" is
        # spelled `is not None`, and 3578's restore reuses this very path
        # rather than growing a second glob that must agree forever (INV-5).
        # The equality below is what pins that: a bool return would fail it.
        # (An `assert not isinstance(found, bool)` was deliberately dropped
        # here — bool and Path are unrelated types, so it is vacuously true
        # once the equality holds and pins nothing.)
        assert found == archived
        assert isinstance(found, Path)
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

        assert found is not None
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
        Be precise about what this pins and what it does NOT: that dir carries
        no ``.jsonl`` suffix, so it is the PATTERN — ``<sid>.jsonl*`` — that
        excludes it, and this test would still pass with the ``is_file()``
        filter deleted. What it guards is the pattern staying narrow: widening
        it to ``<sid>*`` would start matching the subagent dir and this test
        reddens. The ``is_file()`` filter is pinned separately, by
        :meth:`test_b3_a_matching_directory_is_never_returned` below.
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

        assert found is not None
        assert found == main
        assert found.is_file()

    def test_b3_a_matching_directory_is_never_returned(self, tmp_path):
        """The ``is_file()`` filter itself: a DIRECTORY that matches the glob.

        The decoy is a directory named exactly ``<sid>.jsonl.gz`` under a
        second lane, so the pattern matches it and only ``is_file()`` can
        reject it. It is also given the NEWER mtime, so with the filter removed
        the I-F newest-wins selection would actively prefer it and hand a
        caller a directory as "the transcript" — which is what makes this a
        discriminating test rather than a restatement of the pattern.
        """
        root = tmp_path / 'archive'
        sid = 'sess-b3-dirdecoy'
        task_id = '42'

        main = _write(root / task_id / ENC / f'{sid}.jsonl.gz', b'gz-bytes')
        decoy_dir = root / task_id / ENC_B / f'{sid}.jsonl.gz'
        decoy_dir.mkdir(parents=True)
        os.utime(main, (1_000_000, 1_000_000))
        os.utime(decoy_dir, (2_000_000, 2_000_000))

        found = durable_archive_path(root, task_id, sid)

        assert found == main
        assert found is not None and found.is_file()

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

        Scope, stated honestly: this row pins the winner (``a``, the ENC copy
        — see the ENC/ENC_B note at module scope for why the FULL-path
        comparison inverts the bare-dirname order) and that repeats agree. It
        does NOT on its own discriminate the tiebreak: within one process glob
        order is stable, so deleting ``str(p)`` leaves this row green. The
        cross-process direction the tiebreak actually exists for is pinned by
        :meth:`test_b6_tiebreak_beats_glob_order_in_both_directions` below,
        which forces both orders — verified by mutation, not assumed.
        """
        root = tmp_path / 'archive'
        sid = 'sess-b6-tie'
        task_id = '42'

        a = _write(root / task_id / ENC / f'{sid}.jsonl.gz', b'lane-a')
        b = _write(root / task_id / ENC_B / f'{sid}.jsonl.gz', b'lane-b')
        os.utime(a, (1_500_000, 1_500_000))
        os.utime(b, (1_500_000, 1_500_000))

        answers = {durable_archive_path(root, task_id, sid) for _ in range(5)}

        assert answers == {a}
        assert b not in answers

    @pytest.mark.parametrize('glob_order_reversed', [False, True])
    def test_b6_tiebreak_beats_glob_order_in_both_directions(
        self, tmp_path, monkeypatch, glob_order_reversed
    ):
        """B6 (I-F): under a tie the answer does not depend on glob order.

        The reproducibility claim I-F actually makes is ACROSS processes and
        machines, where ``Path.glob``'s iteration order is filesystem- and
        readdir-dependent. A same-process repeat cannot observe that, so the
        order is forced here — once ascending, once descending — and the
        tiebreak must name the same file (``a``, the lexicographically greater
        FULL path) both times.

        This is the discriminating direction: with the ``str(p)`` tiebreak
        removed, ``max()`` degrades to first-encountered-maximum, so the
        ASCENDING case yields ``b`` and the descending case ``a`` — the two
        parameterisations disagree, which is precisely the cross-machine
        non-reproducibility the tiebreak exists to prevent.
        """
        root = tmp_path / 'archive'
        sid = 'sess-b6-order'
        task_id = '42'

        a = _write(root / task_id / ENC / f'{sid}.jsonl.gz', b'lane-a')
        b = _write(root / task_id / ENC_B / f'{sid}.jsonl.gz', b'lane-b')
        os.utime(a, (1_500_000, 1_500_000))
        os.utime(b, (1_500_000, 1_500_000))

        real_glob = Path.glob

        def _ordered(self, pattern):
            return iter(
                sorted(real_glob(self, pattern), key=str, reverse=glob_order_reversed)
            )

        monkeypatch.setattr(Path, 'glob', _ordered)

        assert durable_archive_path(root, task_id, sid) == a

    def test_b5_missing_root_returns_none(self, tmp_path):
        """B5 (I-A): an archive root that does not exist at all."""
        assert durable_archive_path(tmp_path / 'nope', '42', 'sess-b5') is None

    def test_b5_root_is_a_file_returns_none(self, tmp_path):
        """B5 (I-A): an archive root that is a FILE, not a directory."""
        root = _write(tmp_path / 'archive', b'not a directory')

        assert durable_archive_path(root, '42', 'sess-b5') is None

    def test_b5_absent_task_subtree_returns_none(self, tmp_path):
        """B5 (I-A): the root exists but holds no subtree for this task."""
        root = tmp_path / 'archive'
        _write(root / '99' / ENC / 'sess-other.jsonl.gz', b'gz-bytes')

        assert durable_archive_path(root, '42', 'sess-b5') is None

    def test_b5_raising_glob_returns_none(self, tmp_path, monkeypatch):
        """B5 (I-A): a glob that raises yields None, never a propagated error.

        The equivalent of the blanket ``except Exception`` that
        :func:`shared.cli_invoke._resolve_transcript_path` carries — the
        property callers on the dispatch path lean on to stay total.
        """
        root = tmp_path / 'archive'
        _write(root / '42' / ENC / 'sess-b5.jsonl.gz', b'gz-bytes')

        def _boom(self, pattern):
            raise OSError(errno.EIO, 'boom')

        monkeypatch.setattr(Path, 'glob', _boom)

        assert durable_archive_path(root, '42', 'sess-b5') is None

    def test_b5_raising_stat_returns_none(self, tmp_path, monkeypatch):
        """B5 (I-A): a concurrent unlink between glob and stat yields None.

        Models task 2731's GC sweep unlinking an archive in the window between
        the glob listing it and the newest-mtime key stat-ing it.
        """
        root = tmp_path / 'archive'
        sid = 'sess-b5-stat'
        _write(root / '42' / ENC / f'{sid}.jsonl.gz', b'a')
        _write(root / '42' / ENC_B / f'{sid}.jsonl.gz', b'b')

        real_stat = Path.stat
        # Let the is_file() filter's stats through (both matches are real
        # files), then unlink underneath the newest-mtime selection key — the
        # window the GC actually races. Failing earlier would be caught by
        # is_file()'s own OSError swallow and would not exercise max().
        survived = {'calls': 0}

        def _boom(self, *args, **kwargs):
            if self.name.startswith(sid):
                survived['calls'] += 1
                if survived['calls'] > 2:
                    raise FileNotFoundError(errno.ENOENT, 'gone')
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, 'stat', _boom)

        assert durable_archive_path(root, '42', sid) is None
        # The selection key really did get reached (2 is_file() stats + more).
        assert survived['calls'] > 2

    def test_b5_non_str_ids_do_not_raise(self, tmp_path):
        """B5 (I-A): an integer task_id is coerced, not a TypeError.

        Task ids are numeric strings, so a caller can easily hold the int.

        The ``type: ignore``s below are the POINT of this test, not noise: the
        declared signature stays strict (``archive_root: Path``, ``task_id:
        str``) so real callers get a type error at the seam, and the coercion
        inside the locator is defence-in-depth against a caller bug reaching it
        anyway. Pinning that requires violating the annotation on purpose —
        widening the signature instead would advertise the sloppy shapes as
        supported and lose the static signal that keeps I-E's sole locator crisp.
        """
        root = tmp_path / 'archive'
        sid = 'sess-b5-int'
        archived = _write(root / '42' / ENC / f'{sid}.jsonl.gz', b'gz-bytes')

        assert durable_archive_path(root, 42, sid) == archived  # type: ignore[arg-type]
        assert durable_archive_path(root, 99, sid) is None  # type: ignore[arg-type]
        assert durable_archive_path(str(root), '42', sid) == archived  # type: ignore[arg-type]

    def test_i_d_lookup_leaves_the_archive_byte_identical(self, tmp_path):
        """I-D: the locator is strictly read-only — glob and stat only.

        Restoration (decompressing, moving a transcript back into a config
        dir) is task 3578's, under task 3619's archive-before-delete guard.
        Nothing here may create, move, delete or decompress a file.
        """
        root = tmp_path / 'archive'
        sid = 'sess-ro'
        task_id = '42'
        _write(root / task_id / ENC / f'{sid}.jsonl.gz', b'gz-bytes')
        _write(root / task_id / ENC_B / 'sess-other.jsonl', b'other-bytes')

        def _snapshot():
            return sorted(
                (
                    str(p.relative_to(root)),
                    p.is_file(),
                    p.read_bytes() if p.is_file() else None,
                    p.stat().st_mtime_ns,
                )
                for p in root.rglob('*')
            )

        before = _snapshot()

        assert durable_archive_path(root, task_id, sid) is not None

        assert _snapshot() == before
