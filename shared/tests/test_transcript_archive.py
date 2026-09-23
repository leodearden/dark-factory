"""Tests for shared.transcript_archive — best-effort per-task transcript archival."""

from __future__ import annotations

import errno
import logging
import os
from pathlib import Path

import pytest

from shared import transcript_archive as transcript_archive_module
from shared.transcript_archive import (
    archive_before_delete,
    archive_task_transcripts,
    durable_archive_path,
    restore_archived_transcript,
)

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


class TestArchiveBeforeDelete:
    """Task 3619 leaf 2 — archival as a PRECONDITION of config-dir deletion.

    :func:`archive_task_transcripts` COPIES and leaves the source alone; the
    caller then deletes the config dir as a *separate, later* step. That gap is
    the measured bug: every teardown site (``_cleanup_config_dir``,
    ``_recycle_config_dir``, ``cleanup_worktree``) can lose the copy step —
    to a SIGTERM cancellation landing on the archival ``await``, or to a code
    path that simply never had one — and still run the ``rmtree``, destroying
    the only copy of the transcript.

    :func:`archive_before_delete` closes it by making the two ONE operation:
    a transcript is moved out first, and only what is provably durable is
    deleted. These tests pin the happy path (this class); the credential-purge
    hard constraint and the EXDEV fallback are pinned separately below.
    """

    def test_moves_every_transcript_then_removes_the_whole_config_dir(self, tmp_path):
        """(a)+(b)+(d): archive at the canonical path, sources gone, dir gone."""
        config_dir = tmp_path / 'claude-config-3619'
        root = tmp_path / 'archive'
        task_id = '3619'
        sid = 'sess-move'

        main_bytes = b'{"type":"main","line":1}\n{"type":"main","line":2}\n'
        sub_bytes = b'{"type":"subagent","line":1}\n'

        src_main = _write(config_dir / 'projects' / ENC / f'{sid}.jsonl', main_bytes)
        src_sub = _write(
            config_dir / 'projects' / ENC / sid / 'subagents' / 'agent-1.jsonl',
            sub_bytes,
        )

        outcome = archive_before_delete(config_dir, task_id, archive_root=root)

        # (a) Byte-identical at the SAME layout _archive_one writes, so the
        # already-current skip is shared across all three producer sites.
        main_dest = root / task_id / ENC / f'{sid}.jsonl'
        sub_dest = root / task_id / ENC / sid / 'subagents' / 'agent-1.jsonl'
        assert main_dest.read_bytes() == main_bytes
        assert sub_dest.read_bytes() == sub_bytes

        # (b) The sources are gone and the ENTIRE config dir is gone — parity
        # with today's TaskConfigDir.cleanup() rmtree, which this replaces.
        assert not src_main.exists()
        assert not src_sub.exists()
        assert not config_dir.exists()

        # (d) Structured outcome: two moved, nothing skipped, nothing held.
        assert outcome.archived == 2
        assert outcome.already_current == 0
        assert outcome.held == ()
        assert outcome.config_dir_removed is True

    def test_a_rename_preserves_the_source_mtime_so_the_next_call_skips(
        self, tmp_path
    ):
        """The move must leave the archive stamped from the SOURCE, not `now`.

        ``_archive_one``'s copy path needs an explicit ``os.utime`` mirror for
        this; a rename preserves the inode, so it comes free. It is asserted
        anyway because it is load-bearing in two directions: a ``now``-stamped
        archive reads to ``gc_agent_transcripts`` as a reset retention age, and
        it would defeat the already-current skip the sweeper and the producer
        both depend on.
        """
        config_dir = tmp_path / 'claude-config-3619'
        root = tmp_path / 'archive'
        task_id = '3619'
        sid = 'sess-mtime'

        src = _write(config_dir / 'projects' / ENC / f'{sid}.jsonl', b'{"l":1}\n')
        os.utime(src, (1_600_000_000, 1_600_000_000))
        src_mtime = src.stat().st_mtime

        archive_before_delete(config_dir, task_id, archive_root=root)

        dest = root / task_id / ENC / f'{sid}.jsonl'
        assert int(dest.stat().st_mtime) == int(src_mtime)

    def test_an_already_current_archive_is_corroborated_not_rewritten(self, tmp_path):
        """(c): dest already current → source deleted, archive left untouched.

        Corroborate-then-delete, not re-archive-then-delete. The producer hook
        already archived this session on its way out (workflow.py's
        ``_invoke`` finally), so by teardown the common case is that the
        durable copy is ALREADY there. Rewriting it would be wasted I/O; more
        importantly, proving the archive is current is exactly the precondition
        that licenses the delete.
        """
        config_dir = tmp_path / 'claude-config-3619'
        root = tmp_path / 'archive'
        task_id = '3619'
        sid = 'sess-current'

        src = _write(config_dir / 'projects' / ENC / f'{sid}.jsonl', b'{"src":1}\n')
        # Pre-write the archive with a DISTINCTIVE payload and mirror the
        # source mtime onto it, so the already-current predicate holds. A
        # genuine skip leaves the marker intact; an unconditional re-archive
        # replaces it with the source bytes.
        marker = b'MARKER-already-current-not-rewritten'
        dest = _write(root / task_id / ENC / f'{sid}.jsonl', marker)
        st = src.stat()
        os.utime(dest, (st.st_atime, st.st_mtime))

        outcome = archive_before_delete(config_dir, task_id, archive_root=root)

        assert dest.read_bytes() == marker
        # ...yet the source is still deleted, and the dir still torn down.
        assert not src.exists()
        assert not config_dir.exists()
        assert outcome.archived == 0
        assert outcome.already_current == 1
        assert outcome.held == ()
        assert outcome.config_dir_removed is True

    def test_a_mixed_dir_reports_both_moved_and_already_current(self, tmp_path):
        """One transcript already archived, one not — both resolved, both gone."""
        config_dir = tmp_path / 'claude-config-3619'
        root = tmp_path / 'archive'
        task_id = '3619'

        stale = _write(config_dir / 'projects' / ENC / 'sess-new.jsonl', b'{"n":1}\n')
        done = _write(config_dir / 'projects' / ENC / 'sess-old.jsonl', b'{"o":1}\n')
        done_dest = _write(root / task_id / ENC / 'sess-old.jsonl', b'{"o":1}\n')
        st = done.stat()
        os.utime(done_dest, (st.st_atime, st.st_mtime))

        outcome = archive_before_delete(config_dir, task_id, archive_root=root)

        assert outcome.archived == 1
        assert outcome.already_current == 1
        assert outcome.held == ()
        assert (root / task_id / ENC / 'sess-new.jsonl').read_bytes() == b'{"n":1}\n'
        assert not stale.exists()
        assert not config_dir.exists()

    def test_a_missing_config_dir_is_an_idempotent_no_op(self, tmp_path):
        """(e): never raises on an absent dir — teardown sites call it blind.

        ``cleanup_worktree`` fires on lanes that never got a config dir, and
        ``_cleanup_config_dir`` can run twice; both must be cheap no-ops rather
        than a raise inside a teardown path.
        """
        root = tmp_path / 'archive'
        outcome = archive_before_delete(
            tmp_path / 'no-such-config-dir', '3619', archive_root=root
        )
        assert outcome.archived == 0
        assert outcome.already_current == 0
        assert outcome.held == ()
        # Nothing to remove, so nothing was removed — and no archive tree was
        # conjured for a task that has no transcripts.
        assert outcome.config_dir_removed is False
        assert not root.exists()

    def test_a_config_dir_with_no_transcripts_is_still_torn_down(self, tmp_path):
        """No ``projects/`` at all → nothing held, so the dir still goes."""
        config_dir = tmp_path / 'claude-config-3619'
        _write(config_dir / '.credentials.json', b'{"claudeAiOauth":{}}')

        outcome = archive_before_delete(
            config_dir, '3619', archive_root=tmp_path / 'archive'
        )

        assert outcome.archived == 0
        assert outcome.held == ()
        assert outcome.config_dir_removed is True
        assert not config_dir.exists()

    def test_running_it_twice_is_idempotent(self, tmp_path):
        """A second call over the already-removed dir is a silent no-op."""
        config_dir = tmp_path / 'claude-config-3619'
        root = tmp_path / 'archive'
        _write(config_dir / 'projects' / ENC / 'sess-twice.jsonl', b'{"t":1}\n')

        first = archive_before_delete(config_dir, '3619', archive_root=root)
        second = archive_before_delete(config_dir, '3619', archive_root=root)

        assert first.archived == 1
        assert second.archived == 0
        assert second.already_current == 0
        assert second.held == ()
        assert (root / '3619' / ENC / 'sess-twice.jsonl').exists()


class TestArchiveBeforeDeleteHoldsOnlyTheTranscript:
    """D1 / INV-7: a failing archive must not become a hold on credentials.

    The obvious way to honour "never delete an un-archived transcript" is to
    abort the whole teardown on a failure — and that trades one bounded loss
    for a worse unbounded one: a permanently-failing archive (a full or
    read-only archive root) would leave every task's ``.credentials.json``,
    its ``~/.claude`` settings symlinks, its ``sessions/`` and its
    ``telemetry/`` on disk forever, defeating the per-task credential
    isolation the config dir exists to provide.

    So the hold is SCOPED: exactly the un-archivable ``.jsonl`` stays, and
    every other member of the directory is deleted unconditionally. The held
    file is owned by the next process start's sweeper, so the hold is bounded
    by a restart rather than unbounded in time.
    """

    @staticmethod
    def _build_realistic_config_dir(tmp_path):
        """A config dir shaped like TaskConfigDir builds one, plus transcripts."""
        config_dir = tmp_path / 'claude-config-3619'
        config_dir.mkdir(parents=True)

        creds = config_dir / '.credentials.json'
        creds.write_bytes(b'{"claudeAiOauth":{"accessToken":"SECRET"}}')
        creds.chmod(0o600)

        # Mirrors TaskConfigDir._setup_symlinks: a SYMLINK into ~/.claude.
        # The purge must unlink the link and never follow it — deleting the
        # user's real settings.json would be a far worse bug than the one
        # this task fixes.
        settings_target = tmp_path / 'home-claude' / 'settings.json'
        _write(settings_target, b'{"real":"settings"}')
        (config_dir / 'settings.json').symlink_to(settings_target)

        _write(config_dir / 'sessions' / 'x.json', b'{"session":1}')
        _write(config_dir / 'telemetry' / 'y.log', b'telemetry line\n')

        good = _write(config_dir / 'projects' / ENC / 'sess-good.jsonl', b'{"g":1}\n')
        bad = _write(config_dir / 'projects' / ENC / 'sess-bad.jsonl', b'{"b":1}\n')
        return config_dir, settings_target, good, bad

    @staticmethod
    def _deny(monkeypatch, predicate):
        """Make both archive routes raise EACCES for sources matching *predicate*."""
        real_rename = os.rename
        real_copyfile = transcript_archive_module.shutil.copyfile

        def fake_rename(src, dst, **kwargs):
            if predicate(str(src)):
                raise PermissionError(errno.EACCES, 'Permission denied')
            return real_rename(src, dst, **kwargs)

        def fake_copyfile(src, dst, **kwargs):
            if predicate(str(src)):
                raise PermissionError(errno.EACCES, 'Permission denied')
            return real_copyfile(src, dst, **kwargs)

        monkeypatch.setattr(transcript_archive_module.os, 'rename', fake_rename)
        monkeypatch.setattr(
            transcript_archive_module.shutil, 'copyfile', fake_copyfile
        )

    def test_a_total_archive_failure_still_purges_every_credential(
        self, tmp_path, monkeypatch, caplog
    ):
        root = tmp_path / 'archive'
        task_id = '3619'
        config_dir, settings_target, good, bad = self._build_realistic_config_dir(
            tmp_path
        )
        self._deny(monkeypatch, lambda src: True)

        transcript_archive_module._reset_archival_failures()

        with caplog.at_level(logging.WARNING, logger='shared.transcript_archive'):
            outcome = archive_before_delete(config_dir, task_id, archive_root=root)

        # (a) Every non-transcript member is gone, unconditionally.
        assert not (config_dir / '.credentials.json').exists()
        assert not (config_dir / 'settings.json').is_symlink()
        assert not (config_dir / 'sessions').exists()
        assert not (config_dir / 'telemetry').exists()
        # ...and the symlink was UNLINKED, never followed.
        assert settings_target.read_bytes() == b'{"real":"settings"}'

        # (b) Only the un-archivable transcripts are held.
        assert good.exists()
        assert bad.exists()
        assert set(outcome.held) == {good, bad}
        assert outcome.archived == 0
        assert outcome.already_current == 0
        assert outcome.config_dir_removed is False
        assert config_dir.exists()

        # (c) The EXISTING counter advanced once per failed file, and each
        # carries the structured shape an operator greps for.
        assert transcript_archive_module._archival_failures() == 2
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 2
        assert {r.errno for r in warnings} == {errno.EACCES}
        assert {r.task_id for r in warnings} == {task_id}
        assert {r.path for r in warnings} == {str(good), str(bad)}
        # (d) Nothing was raised — reaching here at all is the assertion.

    def test_one_failing_transcript_does_not_hold_its_sibling(
        self, tmp_path, monkeypatch
    ):
        """Partial failure: the archivable one still leaves; the purge still runs."""
        root = tmp_path / 'archive'
        task_id = '3619'
        config_dir, settings_target, good, bad = self._build_realistic_config_dir(
            tmp_path
        )
        self._deny(monkeypatch, lambda src: 'sess-bad' in src)

        transcript_archive_module._reset_archival_failures()
        outcome = archive_before_delete(config_dir, task_id, archive_root=root)

        assert (root / task_id / ENC / 'sess-good.jsonl').read_bytes() == b'{"g":1}\n'
        assert not good.exists()
        assert bad.exists()
        assert outcome.archived == 1
        assert outcome.held == (bad,)
        assert outcome.config_dir_removed is False
        assert transcript_archive_module._archival_failures() == 1

        # The credential purge is NOT contingent on a clean archive run.
        assert not (config_dir / '.credentials.json').exists()
        assert not (config_dir / 'settings.json').is_symlink()
        assert not (config_dir / 'sessions').exists()
        assert not (config_dir / 'telemetry').exists()
        assert settings_target.exists()

    def test_the_held_transcript_keeps_its_content_and_its_path(
        self, tmp_path, monkeypatch
    ):
        """A hold means UNTOUCHED — the sweeper must find it where it was.

        The startup sweeper globs ``<worktree>/.task/claude-config-*/projects/
        **/*.jsonl``, so a held file relocated or emptied by the purge would be
        unrecoverable. Its ancestor directories under ``projects/`` therefore
        survive too.
        """
        root = tmp_path / 'archive'
        config_dir, _settings_target, _good, bad = self._build_realistic_config_dir(
            tmp_path
        )
        self._deny(monkeypatch, lambda src: True)

        transcript_archive_module._reset_archival_failures()
        archive_before_delete(config_dir, '3619', archive_root=root)

        assert bad.read_bytes() == b'{"b":1}\n'
        assert bad.parent.is_dir()
        assert (config_dir / 'projects').is_dir()


class TestArchiveBeforeDeleteCrossDevice:
    """PRD §9 Q4: the rename fast path is a deployment property, not a promise.

    ``os.rename`` only works within one filesystem. On this host the config dir
    (under ``<project_root>/.worktrees``) and the archive root (under
    ``<project_root>/data/orchestrator``) are measurably on the SAME device, so
    the fast path is the live route and ``EXDEV`` is unreachable here. But a
    Linux ``st_dev`` is an ephemeral mount handle, not a stable identifier, and
    an operator is free to mount either path elsewhere — so the cross-device
    case is HANDLED rather than assumed. These tests are the only thing that
    keeps that branch honest, since production never exercises it.

    Deliberately no assertion on any device-id LITERAL: the invariant is that
    the two paths agree with EACH OTHER, and a recorded number would go stale
    across the next remount and make a healthy host look broken.
    """

    @staticmethod
    def _rename_is_cross_device(monkeypatch):
        monkeypatch.setattr(
            transcript_archive_module.os,
            'rename',
            lambda *a, **kw: (_ for _ in ()).throw(
                OSError(errno.EXDEV, 'Invalid cross-device link')
            ),
        )

    def test_exdev_falls_back_to_copy_then_unlinks_the_source(
        self, tmp_path, monkeypatch, caplog
    ):
        config_dir = tmp_path / 'claude-config-3619'
        root = tmp_path / 'archive'
        task_id = '3619'
        sid = 'sess-xdev'

        payload = b'{"line":1}\n{"line":2}\n'
        src = _write(config_dir / 'projects' / ENC / f'{sid}.jsonl', payload)
        os.utime(src, (1_600_000_000, 1_600_000_000))
        src_mtime = src.stat().st_mtime

        self._rename_is_cross_device(monkeypatch)
        transcript_archive_module._reset_archival_failures()

        with caplog.at_level(logging.WARNING, logger='shared.transcript_archive'):
            outcome = archive_before_delete(config_dir, task_id, archive_root=root)

        dest = root / task_id / ENC / f'{sid}.jsonl'
        # (a) The transcript is durable, with the SOURCE mtime mirrored onto
        # it. The copy path needs an explicit os.utime for that where the
        # rename got it free — and without it the already-current skip would
        # never fire, so every later pass would re-archive forever and
        # gc_agent_transcripts would read a permanently reset retention age.
        assert dest.read_bytes() == payload
        assert int(dest.stat().st_mtime) == int(src_mtime)

        # (b) A cross-device host still gets deletion-AFTER-archival, not an
        # unbounded hold: the whole point is that the copy licenses the delete.
        assert not src.exists()

        # (c) Nothing held, dir gone — outwardly indistinguishable from the
        # rename path, which is what makes the fallback a real fallback.
        assert outcome.archived == 1
        assert outcome.held == ()
        assert outcome.config_dir_removed is True
        assert not config_dir.exists()

        # (d) EXDEV is a handled ROUTE, not a failure. A counter that climbed
        # here would page an operator on every archive on a two-mount host.
        assert transcript_archive_module._archival_failures() == 0
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

        # (e) No staging debris left in the archive tree.
        assert [p.name for p in root.rglob('*') if p.is_file()] == [dest.name]
        assert not any(
            p.name.endswith('.archive-tmp') for p in root.rglob('*')
        )

    def test_exdev_then_a_failing_copy_holds_the_transcript(
        self, tmp_path, monkeypatch, caplog
    ):
        """Both routes fail → hold, count once, publish nothing partial."""
        config_dir = tmp_path / 'claude-config-3619'
        root = tmp_path / 'archive'
        task_id = '3619'
        sid = 'sess-xdev-dead'

        src = _write(
            config_dir / 'projects' / ENC / f'{sid}.jsonl',
            b'{"line":1}\n{"line":2}\n{"line":3}\n',
        )
        self._rename_is_cross_device(monkeypatch)
        # The realistic shape: bytes land, THEN the write fails. A mock that
        # raised before writing anything would pass against an in-place copy
        # too and prove nothing about where the partial bytes went.
        TestBestEffortLoud._copyfile_dies_part_way(monkeypatch, b'{"line":1}\n{"li')

        transcript_archive_module._reset_archival_failures()

        with caplog.at_level(logging.WARNING, logger='shared.transcript_archive'):
            outcome = archive_before_delete(config_dir, task_id, archive_root=root)

        assert outcome.archived == 0
        assert outcome.held == (src,)
        assert outcome.config_dir_removed is False
        assert src.read_bytes() == b'{"line":1}\n{"line":2}\n{"line":3}\n'

        # Nothing partial published at the canonical path, and no staging
        # debris — the fallback inherits _archive_one's staged-write property.
        dest = root / task_id / ENC / f'{sid}.jsonl'
        assert not dest.exists()
        assert [p for p in root.rglob('*') if p.is_file()] == []

        # Counted ONCE: the EXDEV retry must not double-count one file.
        assert transcript_archive_module._archival_failures() == 1
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert warnings[0].errno == errno.ENOSPC
        assert warnings[0].path == str(src)


class TestArchivalFailureHook:
    """INV-4 substrate: ONE machine-readable seam onto the failure counter.

    The α task left ``_ARCHIVAL_FAILURES`` deliberately owned-but-not-consumed.
    A consumer needs to be TOLD, not to poll a module global, so this adds a
    single notification seam beside the counter — one hook, fired by the one
    ``_record_failure`` both producers already funnel through, rather than a
    second counter or a per-producer callback that could drift out of step with
    the first. The policy that consumes it (a rate threshold, an escalation)
    lives in the orchestrator, where the live config is; this module is on the
    PURE_STDLIB_LEAVES contract and stays a mechanism.
    """

    @staticmethod
    def _teardown_hook():
        transcript_archive_module.set_archival_failure_hook(None)

    def test_the_hook_fires_once_per_failed_file_from_both_producers(
        self, tmp_path
    ):
        """(a) One seam, both producers — no second notification path."""
        root = tmp_path / 'archive'
        task_id = '3619'
        seen: list[dict] = []

        # Producer 1: the COPY path (archive_task_transcripts).
        copy_dir = tmp_path / 'claude-config-copy'
        bad_copy = _write(copy_dir / 'projects' / ENC / 'sess-copy.jsonl', b'{}\n')
        blocker = root / task_id / ENC / 'sess-copy.jsonl'
        blocker.mkdir(parents=True)
        os.utime(blocker, (0, 0))

        # Producer 2: the MOVE path (archive_before_delete).
        move_dir = tmp_path / 'claude-config-move'
        bad_move = _write(move_dir / 'projects' / ENC / 'sess-move.jsonl', b'{}\n')
        blocker2 = root / task_id / ENC / 'sess-move.jsonl'
        blocker2.mkdir(parents=True)
        os.utime(blocker2, (0, 0))

        transcript_archive_module._reset_archival_failures()
        transcript_archive_module.set_archival_failure_hook(seen.append)
        try:
            archive_task_transcripts(copy_dir, task_id, None, archive_root=root)
            archive_before_delete(move_dir, task_id, archive_root=root)
        finally:
            self._teardown_hook()

        assert len(seen) == 2
        assert [p['path'] for p in seen] == [str(bad_copy), str(bad_move)]
        assert {p['task_id'] for p in seen} == {task_id}
        assert {p['errno'] for p in seen} == {errno.EISDIR}
        # The counter is unchanged in meaning: the hook is a notification, not
        # a replacement for the substrate a digest consumer samples.
        assert transcript_archive_module._archival_failures() == 2

    def test_the_hook_can_be_uninstalled_and_does_not_leak_between_tests(
        self, tmp_path
    ):
        """(b) None uninstalls, and _reset_archival_failures also clears it.

        The reset accessor is what every test in this module already calls for
        isolation; if it cleared the counter but left a previous test's hook
        installed, one test's callback would be fed another's failures.
        """
        root = tmp_path / 'archive'
        seen: list[dict] = []

        def one_failure(config_dir_name):
            config_dir = tmp_path / config_dir_name
            src = _write(config_dir / 'projects' / ENC / 'sess.jsonl', b'{}\n')
            blocker = root / '3619' / ENC / 'sess.jsonl'
            if not blocker.exists():
                blocker.mkdir(parents=True)
                os.utime(blocker, (0, 0))
            archive_task_transcripts(config_dir, '3619', None, archive_root=root)
            return src

        transcript_archive_module._reset_archival_failures()
        transcript_archive_module.set_archival_failure_hook(seen.append)
        try:
            one_failure('cfg-a')
            assert len(seen) == 1

            transcript_archive_module.set_archival_failure_hook(None)
            one_failure('cfg-b')
            assert len(seen) == 1

            transcript_archive_module.set_archival_failure_hook(seen.append)
            transcript_archive_module._reset_archival_failures()
            one_failure('cfg-c')
            assert len(seen) == 1
        finally:
            self._teardown_hook()

    def test_a_raising_hook_cannot_become_an_archival_outage(
        self, tmp_path, caplog
    ):
        """(c) A broken consumer must not take archival down with it.

        The hook is called from inside the failure path of a function whose
        whole contract is totality — it runs in ``finally`` blocks and teardown
        paths. Letting a consumer's exception escape would turn "the digest
        consumer has a bug" into "teardown raises", which is a strictly worse
        failure than the one being reported.
        """
        root = tmp_path / 'archive'
        config_dir = tmp_path / 'claude-config-3619'
        good = _write(config_dir / 'projects' / ENC / 'sess-good.jsonl', b'{"g":1}\n')
        _write(config_dir / 'projects' / ENC / 'sess-bad.jsonl', b'{"b":1}\n')
        blocker = root / '3619' / ENC / 'sess-bad.jsonl'
        blocker.mkdir(parents=True)
        os.utime(blocker, (0, 0))

        def exploding_hook(_payload):
            raise RuntimeError('consumer is broken')

        transcript_archive_module._reset_archival_failures()
        transcript_archive_module.set_archival_failure_hook(exploding_hook)
        try:
            with caplog.at_level(logging.WARNING, logger='shared.transcript_archive'):
                count = archive_task_transcripts(
                    config_dir, '3619', None, archive_root=root
                )
        finally:
            self._teardown_hook()

        # The surrounding archive still completed...
        assert count == 1
        assert (root / '3619' / ENC / 'sess-good.jsonl').read_bytes() == good.read_bytes()
        # ...and still counted the ORIGINAL failure, which the hook must not
        # be able to suppress by dying.
        assert transcript_archive_module._archival_failures() == 1

        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('consumer is broken' in m for m in messages)
        assert any('sess-bad.jsonl' in m for m in messages)

    def test_with_no_hook_installed_behaviour_is_unchanged(self, tmp_path, caplog):
        """(d) The default path is byte-identical to before the seam existed."""
        root = tmp_path / 'archive'
        config_dir = tmp_path / 'claude-config-3619'
        _write(config_dir / 'projects' / ENC / 'sess-bad.jsonl', b'{"b":1}\n')
        blocker = root / '3619' / ENC / 'sess-bad.jsonl'
        blocker.mkdir(parents=True)
        os.utime(blocker, (0, 0))

        transcript_archive_module._reset_archival_failures()
        with caplog.at_level(logging.WARNING, logger='shared.transcript_archive'):
            archive_task_transcripts(config_dir, '3619', None, archive_root=root)

        assert transcript_archive_module._archival_failures() == 1
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert warnings[0].errno == errno.EISDIR


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


class TestRestoreArchivedTranscript:
    """The write-back sibling of :func:`durable_archive_path` (task 3578).

    ``restore_archived_transcript`` rehydrates one session's archived main
    transcript back into a live ``CLAUDE_CONFIG_DIR`` so ``--resume`` works
    after the original config dir was destroyed — the pooled-warm-lane case
    where the orchestrator recovers a session id but the lane it re-dispatches
    into has never seen that session's JSONL.

    Every archive in this class is written by the REAL producer
    (:func:`archive_task_transcripts`) rather than hand-placed, so the layout
    under test can never drift from the layout actually shipped.
    """

    @staticmethod
    def _seed_archive(tmp_path, sid: str, task_id: str, payload: bytes):
        """Archive *payload* as *sid*'s transcript via the real producer.

        Returns ``(archive_root, archived_path)``. The source config dir is a
        throwaway: it stands in for the lane that has since been destroyed,
        which is precisely the situation the restore exists to repair.
        """
        source_config = tmp_path / 'lane-a' / 'claude-config'
        root = tmp_path / 'archive'
        _write(source_config / 'projects' / ENC / f'{sid}.jsonl', payload)

        assert archive_task_transcripts(
            source_config, task_id, sid, archive_root=root
        ) == 1
        archived = root / task_id / ENC / f'{sid}.jsonl'
        assert archived.is_file()
        return root, archived

    def test_restores_the_transcript_into_the_config_dir(self, tmp_path):
        """Happy path: an archived session is rehydrated into a fresh lane.

        Pins the four properties the dispatch path leans on: the destination
        path is RETURNED (not a bool), the archive's own ``<enc>/<name>``
        relative path is mirrored VERBATIM with no cwd re-encoding, the bytes
        round-trip, and — the property the CLI actually keys on —
        :func:`shared.cli_invoke.transcript_exists` now answers True.
        """
        from shared.cli_invoke import transcript_exists

        sid = 'sess-restore-happy'
        task_id = '42'
        payload = b'{"type":"user"}\n{"type":"assistant"}\n'
        root, archived = self._seed_archive(tmp_path, sid, task_id, payload)

        # A DIFFERENT lane's config dir, freshly created and empty — the
        # pooled-warm-lane shape. It knows nothing about ENC.
        config_dir = tmp_path / 'lane-b' / 'claude-config'
        config_dir.mkdir(parents=True)

        restored = restore_archived_transcript(root, task_id, sid, config_dir)

        # (a) the destination path is returned.
        assert restored is not None
        assert isinstance(restored, Path)
        # (b) the archive's relative path is mirrored VERBATIM: the encoded-cwd
        # dir name is carried across untouched, NOT re-derived from lane-b's
        # cwd. Measured on Claude Code CLI 2.1.236: the CLI scans every
        # ``projects/*/`` subdir by session id and ignores both the directory
        # name and the ``cwd`` recorded inside the records, so mirroring is
        # correct AND lane-portable.
        assert restored == config_dir / 'projects' / ENC / f'{sid}.jsonl'
        # (c) bytes round-trip exactly.
        assert restored.read_bytes() == payload
        # (d) the predicate the CLI keys on.
        assert transcript_exists(config_dir, sid) is True

    def test_the_restored_copy_carries_the_archived_mtime(self, tmp_path):
        """The archive's mtime is mirrored onto the restored copy.

        Same reason :func:`_archive_one` mirrors it on the way out: a
        restored transcript stamped ``now`` would read to the next
        ``archive_task_transcripts`` pass as newer than its own archive and be
        pointlessly re-archived over it. int-truncated, dodging FS
        mtime-granularity mismatch exactly as the producer's own skip test does.
        """
        sid = 'sess-restore-mtime'
        task_id = '42'
        root, archived = self._seed_archive(tmp_path, sid, task_id, b'payload\n')
        # Back-date the archive so "mirrored" is distinguishable from "now".
        os.utime(archived, (1_700_000_000, 1_700_000_000))

        config_dir = tmp_path / 'lane-b' / 'claude-config'
        config_dir.mkdir(parents=True)

        restored = restore_archived_transcript(root, task_id, sid, config_dir)

        assert restored is not None
        assert int(restored.stat().st_mtime) == int(archived.stat().st_mtime)
        # And therefore the next archival pass treats it as already-current.
        assert archive_task_transcripts(
            config_dir, task_id, sid, archive_root=root
        ) == 0

    def test_a_gzipped_pre_3618_archive_is_restored_decompressed(self, tmp_path):
        """I-C: the ``.jsonl.gz`` corpus restores as plain, usable JSONL.

        :func:`durable_archive_path` is deliberately format-agnostic because
        archives written BEFORE task 3618's flag day are still ``.jsonl.gz`` on
        disk, so a restore that only handled plain files would silently no-op
        on the older corpus — the exact population most likely to need a
        rehydrate, since it is the oldest.

        The destination DROPS the ``.gz``: the CLI parses plain JSONL and would
        reject a gzip blob named ``.jsonl`` exactly as it rejects a zero-byte
        one (measured — a zero-byte file and a preamble-only file both yield
        ``No conversation found with session ID`` on CLI 2.1.236), and
        ``transcript_exists`` globs ``*.jsonl``, so a ``.gz``-suffixed
        destination would not even be seen.
        """
        import gzip

        from shared.cli_invoke import transcript_exists

        sid = 'sess-restore-gz'
        task_id = '42'
        root = tmp_path / 'archive'
        payload = b'{"type":"user"}\n{"type":"assistant"}\n'
        # Hand-placed rather than produced: today's producer emits only plain
        # .jsonl, so the pre-3618 shape can only be reconstructed directly.
        _write(root / task_id / ENC / f'{sid}.jsonl.gz', gzip.compress(payload))

        config_dir = tmp_path / 'lane-b' / 'claude-config'
        config_dir.mkdir(parents=True)

        restored = restore_archived_transcript(root, task_id, sid, config_dir)

        assert restored is not None
        assert restored == config_dir / 'projects' / ENC / f'{sid}.jsonl'
        assert restored.read_bytes() == payload
        assert transcript_exists(config_dir, sid) is True

    def test_a_miss_returns_none_and_leaves_no_trace(self, tmp_path):
        """(a) No archive for this (task_id, session_id) -> None, nothing made.

        A MISS is the common case (the PRD §2 reference measurement puts ~36%
        of sessions there), so it must be free: no ``projects/`` skeleton
        created, no log noise. Creating the directory eagerly would leave every
        missed lookup with a half-built config dir a later reader could mistake
        for a partially-restored one.
        """
        root = tmp_path / 'archive'
        task_id = '42'
        _write(root / task_id / ENC / 'sess-other.jsonl', b'other\n')

        config_dir = tmp_path / 'lane-b' / 'claude-config'
        config_dir.mkdir(parents=True)

        assert restore_archived_transcript(root, task_id, 'sess-missing', config_dir) is None
        assert not (config_dir / 'projects').exists()

    def test_a_live_transcript_is_never_clobbered(self, tmp_path):
        """(b) A transcript already live in the config dir wins, always.

        A resumed session's transcript only ever GROWS — the same premise
        durable_archive_path's I-F newest-mtime rule rests on — so a live copy
        is by construction at least as complete as any archive. Overwriting it
        would destroy context on the exact path meant to preserve it.

        Seeded here under a DIFFERENT encoded-cwd dir than the archive's, which
        is the realistic shape (the live copy was written by this lane, the
        archive by another) and also proves the guard keys on the session id
        via transcript_exists rather than on the destination path.
        """
        sid = 'sess-live-wins'
        task_id = '42'
        root = tmp_path / 'archive'
        _write(root / task_id / ENC / f'{sid}.jsonl', b'ARCHIVED-older\n')

        config_dir = tmp_path / 'lane-b' / 'claude-config'
        live = _write(
            config_dir / 'projects' / ENC_B / f'{sid}.jsonl', b'LIVE-newer-and-longer\n'
        )

        restored = restore_archived_transcript(root, task_id, sid, config_dir)

        assert restored == live
        assert live.read_bytes() == b'LIVE-newer-and-longer\n'
        # And the archive's own encoded-cwd dir was never even created.
        assert not (config_dir / 'projects' / ENC).exists()

    def test_a_genuine_fault_returns_none_loudly_and_never_raises(
        self, tmp_path, monkeypatch, caplog
    ):
        """(c) Totality: a real I/O fault yields None + a structured WARNING.

        Mirrors durable_archive_path's I-A: this runs on the production
        dispatch path, so nothing may escape onto it. But silence would be the
        silent degradation design-invariants INV-2/INV-4 forbid, so a genuine
        fault (as distinct from a plain miss, which stays quiet) is logged at
        WARNING with the same structured ``extra`` shape _record_failure and
        durable_archive_path already emit — path/task_id/errno — so all three
        archive-side failure signals stay greppable the same way.
        """
        sid = 'sess-fault'
        task_id = '42'
        root = tmp_path / 'archive'
        _write(root / task_id / ENC / f'{sid}.jsonl', b'payload\n')

        config_dir = tmp_path / 'lane-b' / 'claude-config'
        config_dir.mkdir(parents=True)

        def _boom(src, dst, **kwargs):
            raise OSError(errno.ENOSPC, 'No space left on device')

        monkeypatch.setattr(transcript_archive_module.shutil, 'copyfile', _boom)

        with caplog.at_level(logging.WARNING, logger='shared.transcript_archive'):
            assert restore_archived_transcript(root, task_id, sid, config_dir) is None

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        rec = warnings[0]
        assert rec.task_id == task_id
        assert rec.errno == errno.ENOSPC
        assert sid in rec.getMessage()

    def test_a_plain_miss_logs_nothing(self, tmp_path, caplog):
        """(a, cont.) The ~36% miss population must never become log noise."""
        root = tmp_path / 'archive'
        config_dir = tmp_path / 'lane-b' / 'claude-config'
        config_dir.mkdir(parents=True)

        with caplog.at_level(logging.DEBUG, logger='shared.transcript_archive'):
            assert restore_archived_transcript(root, '42', 'sess-nope', config_dir) is None

        assert caplog.records == []

    def test_an_interrupted_restore_publishes_nothing_at_all(
        self, tmp_path, monkeypatch, caplog
    ):
        """A torn restore must be indistinguishable from NO restore.

        Not defensive decoration. The gate measurement proved the CLI PARSES
        the transcript rather than stat-ing it — on CLI 2.1.236 a zero-byte
        file and a preamble-only file BOTH yield ``No conversation found with
        session ID`` — so a truncated restore would arm ``--resume`` against a
        file the CLI then rejects, converting a cheap fresh dispatch into a
        wasted invocation plus a spurious cap-net candidate. Exactly the
        argument :func:`_archive_one` already records for the write side.

        The fake writes bytes and THEN raises, which is the real shape of an
        interrupted copy: a mock that raised before writing anything would pass
        against an in-place write too, and prove nothing about where the
        partial bytes went.
        """
        sid = 'sess-torn'
        task_id = '42'
        root = tmp_path / 'archive'
        _write(root / task_id / ENC / f'{sid}.jsonl', b'{"a":1}\n{"b":2}\n')

        config_dir = tmp_path / 'lane-b' / 'claude-config'
        config_dir.mkdir(parents=True)

        def _dies_part_way(src, dst, **kwargs):
            Path(dst).write_bytes(b'{"a":1}\n{"b')
            raise OSError(errno.ENOSPC, 'No space left on device')

        monkeypatch.setattr(
            transcript_archive_module.shutil, 'copyfile', _dies_part_way
        )

        with caplog.at_level(logging.WARNING, logger='shared.transcript_archive'):
            assert restore_archived_transcript(root, task_id, sid, config_dir) is None

        # No transcript published at the canonical name...
        assert not (config_dir / 'projects' / ENC / f'{sid}.jsonl').exists()
        # ...and no staging residue either. Debris that survived would be inert
        # (it matches no reader's *.jsonl glob) but is still a leak.
        assert list((config_dir / 'projects' / ENC).glob('*')) == []
        # The predicate the dispatch path corroborates on still says "absent",
        # so the arm-site veto fires and the invocation starts fresh — cheap —
        # instead of resuming into a file the CLI will reject.
        from shared.cli_invoke import transcript_exists

        assert transcript_exists(config_dir, sid) is False

    def test_an_interrupted_gz_restore_publishes_nothing_either(
        self, tmp_path, monkeypatch
    ):
        """The staging discipline spans the decompressing branch too.

        A gunzip stream is the likelier place to tear (a corrupt member raises
        part-way through, after bytes have already landed), so pinning only the
        plain-copy branch would leave the riskier path unguarded.
        """
        import gzip

        from shared.cli_invoke import transcript_exists

        sid = 'sess-torn-gz'
        task_id = '42'
        root = tmp_path / 'archive'
        _write(
            root / task_id / ENC / f'{sid}.jsonl.gz',
            gzip.compress(b'{"a":1}\n{"b":2}\n'),
        )

        config_dir = tmp_path / 'lane-b' / 'claude-config'
        config_dir.mkdir(parents=True)

        def _dies_part_way(src_fh, dest_fh, *args, **kwargs):
            dest_fh.write(b'{"a":1}\n{"b')
            raise OSError(errno.ENOSPC, 'No space left on device')

        monkeypatch.setattr(
            transcript_archive_module.shutil, 'copyfileobj', _dies_part_way
        )

        assert restore_archived_transcript(root, task_id, sid, config_dir) is None
        assert list((config_dir / 'projects' / ENC).glob('*')) == []
        assert transcript_exists(config_dir, sid) is False

    def test_strict_propagates_a_genuine_fault_after_logging_it(
        self, tmp_path, monkeypatch, caplog
    ):
        """(a) ``strict=True``: a real fault RAISES — and still logs first.

        The miss/fault distinction has to survive the helper boundary. Total
        by default, this function answers a plain miss and a broken restore
        with the same ``None``, so its one production caller — which brackets
        the call in its own blanket ``except`` — can only ever bucket an
        internal fault as ``restore='miss'``: an operator is then sent to chase
        archive COVERAGE while the restore path itself is the broken thing.
        That mis-bucketing covers nearly every real fault, since the only
        exception the caller can catch unaided comes from ``resolve_archive_root``.

        Logged AND raised, not one or the other: the WARNING is the greppable
        helper-layer record carrying ``errno``/``path``, and the raise is what
        makes the fault classifiable one frame up. Asserting the record after
        the raise escapes pins the order — emit, then propagate.
        """
        sid = 'sess-strict-fault'
        task_id = '42'
        root = tmp_path / 'archive'
        _write(root / task_id / ENC / f'{sid}.jsonl', b'payload\n')

        config_dir = tmp_path / 'lane-b' / 'claude-config'
        config_dir.mkdir(parents=True)

        def _boom(src, dst, **kwargs):
            raise OSError(errno.ENOSPC, 'No space left on device')

        monkeypatch.setattr(transcript_archive_module.shutil, 'copyfile', _boom)

        with (
            caplog.at_level(logging.WARNING, logger='shared.transcript_archive'),
            pytest.raises(OSError) as excinfo,
        ):
            restore_archived_transcript(root, task_id, sid, config_dir, strict=True)

        assert excinfo.value.errno == errno.ENOSPC
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        rec = warnings[0]
        # The helper-layer half of the fault pair, deliberately named apart
        # from the dispatch layer's session_resume_restore_fault so one fault
        # cannot be double-counted under a single greppable key.
        assert rec.event == 'transcript_restore_fault'
        assert rec.task_id == task_id
        assert rec.errno == errno.ENOSPC
        assert sid in rec.getMessage()
        # Nothing published, exactly as in the non-strict tear: strict changes
        # who learns about the fault, never what lands on disk.
        assert list((config_dir / 'projects' / ENC).glob('*')) == []

    def test_strict_leaves_a_plain_miss_a_quiet_none(self, tmp_path, caplog):
        """(b) ``strict=True`` + a MISS still returns None, silently.

        Load-bearing, not symmetry. A miss is not a fault: it is the ~36%
        common case. If ``strict`` ever regressed into raising on misses, the
        caller's ``except`` would stamp that whole population ``fault`` and
        INVERT the diagnosis this seam exists to restore — the archive-coverage
        signal would read as a broken restore path. By construction the miss
        returns from a branch ABOVE the blanket handler, so it can never reach
        the ``raise``; this pins that structure rather than trusting it.
        """
        root = tmp_path / 'archive'
        config_dir = tmp_path / 'lane-b' / 'claude-config'
        config_dir.mkdir(parents=True)

        with caplog.at_level(logging.DEBUG, logger='shared.transcript_archive'):
            assert restore_archived_transcript(
                root, '42', 'sess-nope', config_dir, strict=True
            ) is None

        assert caplog.records == []

    def test_the_default_call_stays_total_and_gains_the_event_key(
        self, tmp_path, monkeypatch, caplog
    ):
        """(c) No ``strict`` kwarg: the published total contract is preserved.

        ``strict`` is an additive seam for the ONE caller that can hold the
        totality at the composite level, not a migration: every other caller
        keeps a helper that cannot fail (the I-A sibling contract
        ``test_a_genuine_fault_returns_none_loudly_and_never_raises`` pins).
        The WARNING gains ``event=`` so OPERATIONS.md's grep-by-event-name
        instruction resolves against a record that today has no such key.
        """
        sid = 'sess-default-fault'
        task_id = '42'
        root = tmp_path / 'archive'
        _write(root / task_id / ENC / f'{sid}.jsonl', b'payload\n')

        config_dir = tmp_path / 'lane-b' / 'claude-config'
        config_dir.mkdir(parents=True)

        def _boom(src, dst, **kwargs):
            raise OSError(errno.ENOSPC, 'No space left on device')

        monkeypatch.setattr(transcript_archive_module.shutil, 'copyfile', _boom)

        with caplog.at_level(logging.WARNING, logger='shared.transcript_archive'):
            assert restore_archived_transcript(root, task_id, sid, config_dir) is None

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert warnings[0].event == 'transcript_restore_fault'
        assert warnings[0].errno == errno.ENOSPC
