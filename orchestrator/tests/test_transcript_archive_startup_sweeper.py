"""Boot-time sweeper for the SIGKILL tail (task 3619, leaf 2 / INV-7).

``archive_before_delete`` makes archival a PRECONDITION of config-dir deletion
on every path the orchestrator itself walks — but nothing runs at all when the
process is SIGKILLed.  That tail (and the deliberate HOLD
``archive_before_delete`` leaves behind when a transcript cannot be made
durable) is owned by ``Harness._sweep_orphaned_transcripts``: a boot-time pass
over the worktrees that survived the crash, bounded by the next process start.

The sweeper COPIES (``archive_task_transcripts``) and never deletes — moving a
transcript out from under a recovered session would turn it into a
``no_transcript`` resume fallback, which is precisely the failure this task
exists to remove.

Fixtures are kept module-local (no conftest.py edit) — a conftest.py edit trips
verify.py's ``has_conftest`` and forces the merge-time verify to run the full
owning-package suite instead of a scoped subset.  The ``harness`` fixture below
mirrors ``test_crash_recovery.py``'s.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from shared.cli_invoke import transcript_exists
from shared.config_dir import CONFIG_DIR_PREFIX

from orchestrator.config import TranscriptArchiveConfig
from orchestrator.harness import Harness
from orchestrator.lane_lifecycle import LaneLifecycle

# A representative encoded-project directory name (Claude Code encodes the
# absolute project path into this leaf; the exact encoding is irrelevant here).
ENC = '-home-leo-src-dark-factory'


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config):
    """A Harness with mocked internals, mirroring test_crash_recovery.py's fixture.

    ``transcript_archive`` is assigned a REAL ``TranscriptArchiveConfig``: the
    shared ``mock_orch_config`` leaves it a bare MagicMock, whose ``root``
    would compose into a nonsense archive path that silently matches nothing —
    every assertion below would then pass vacuously.
    """
    mock_orch_config.transcript_archive = TranscriptArchiveConfig()
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[])
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.get_task = AsyncMock(return_value={})
    h.scheduler.get_status = AsyncMock(return_value=None)
    h.scheduler._dispatched = set()
    h.scheduler.is_deterministic = MagicMock(return_value=False)

    h.git_ops.worktree_base = (tmp_path / '.worktrees').resolve()
    h.git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
    h.git_ops.mark_pool_storage_present()
    h.git_ops.cleanup_worktree = AsyncMock()
    h.git_ops.quarantine_worktree = AsyncMock(return_value=None)
    h.git_ops._lane_lifecycle = LaneLifecycle(
        h.git_ops.worktree_base, quarantine_worktree=h.git_ops.quarantine_worktree,
    )
    h.git_ops._is_registered_worktree = AsyncMock(return_value=True)
    h.event_store = MagicMock()
    return h


def _archive_root(harness: Harness) -> Path:
    """The archive root the sweeper must compose — project_root / ta.root."""
    return Path(harness.config.project_root) / harness.config.transcript_archive.root


def _seed_transcript(
    harness: Harness,
    worktree_name: str,
    config_dir_name: str,
    session_id: str,
    payload: bytes = b'{"type":"summary"}\n',
) -> Path:
    """Create ``<wt>/.task/<config_dir_name>/projects/<ENC>/<sid>.jsonl``.

    Returns the config-dir path (what ``transcript_exists`` globs under).
    """
    cfg = harness.git_ops.worktree_base / worktree_name / '.task' / config_dir_name
    src = cfg / 'projects' / ENC / f'{session_id}.jsonl'
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(payload)
    return cfg


class TestSweepOrphanedTranscripts:
    """The sweeper itself: covers every surviving lane, copies, never deletes."""

    def test_archives_every_worktree_keyed_by_stripped_config_dir_name(
        self, harness: Harness,
    ):
        """(a) Every ``<wt>/.task/claude-config-*/projects/**/*.jsonl`` archives
        under the config-dir name with ``CONFIG_DIR_PREFIX`` stripped — the SAME
        key ``_cleanup_config_dir`` and ``cleanup_worktree`` use, so a swept
        archive and a teardown archive collide idempotently instead of forking
        two archives of one session (INV-5).
        """
        main = b'{"type":"main","line":1}\n'
        sub = b'{"type":"subagent"}\n'
        cfg_a = _seed_transcript(harness, '3619', 'claude-config-3619', 'sess-a', main)
        # A subagent transcript under the same config dir — the ** glob must
        # reach it, not just the top-level projects/<ENC>/*.jsonl row.
        subsrc = cfg_a / 'projects' / ENC / 'sess-a' / 'subagents' / 'agent-1.jsonl'
        subsrc.parent.mkdir(parents=True, exist_ok=True)
        subsrc.write_bytes(sub)
        # A second lane whose config dir carries a NON-numeric suffix: the key
        # is the stripped dir name, never a bare task id parse.
        _seed_transcript(
            harness, '3464-unblock', 'claude-config-3464-unblock', 'sess-b',
        )

        n = harness._sweep_orphaned_transcripts()

        root = _archive_root(harness)
        assert (root / '3619' / ENC / 'sess-a.jsonl').read_bytes() == main
        assert (
            root / '3619' / ENC / 'sess-a' / 'subagents' / 'agent-1.jsonl'
        ).read_bytes() == sub
        assert (root / '3464-unblock' / ENC / 'sess-b.jsonl').exists()
        assert n == 3
        # The prefix is DERIVED, never restated: the key must not carry it.
        assert not (root / f'{CONFIG_DIR_PREFIX}3619').exists()

    def test_leaves_the_sources_in_place(self, harness: Harness):
        """(b) COPY, never move.  A recovered session must still be able to read
        its own live transcript — moving it would make every resume degrade to
        a ``no_transcript`` fallback, inverting the point of this task.
        """
        cfg = _seed_transcript(harness, '3619', 'claude-config-3619', 'sess-a')

        harness._sweep_orphaned_transcripts()

        assert transcript_exists(cfg, 'sess-a') is True
        assert (cfg / 'projects' / ENC / 'sess-a.jsonl').exists()

    def test_second_sweep_archives_nothing_new(self, harness: Harness):
        """(c) Idempotent — the already-current mtime check makes a re-sweep free."""
        _seed_transcript(harness, '3619', 'claude-config-3619', 'sess-a')

        assert harness._sweep_orphaned_transcripts() == 1
        assert harness._sweep_orphaned_transcripts() == 0

    def test_one_bad_entry_cannot_abort_the_sweep(
        self, harness: Harness, monkeypatch, caplog,
    ):
        """(d) Total — a per-entry failure is logged and skipped, siblings still
        archive, and nothing propagates to the boot path that calls this.
        """
        import orchestrator.harness as harness_mod

        _seed_transcript(harness, 'aaa-bad', 'claude-config-bad', 'sess-bad')
        _seed_transcript(harness, 'zzz-good', 'claude-config-good', 'sess-good')

        real = harness_mod.archive_task_transcripts

        def _explode(config_dir, task_id, session_id=None, **kw):
            if task_id == 'bad':
                raise PermissionError(13, 'Permission denied')
            return real(config_dir, task_id, session_id, **kw)

        monkeypatch.setattr(harness_mod, 'archive_task_transcripts', _explode)

        with caplog.at_level(logging.WARNING):
            n = harness._sweep_orphaned_transcripts()

        assert n == 1
        assert (_archive_root(harness) / 'good' / ENC / 'sess-good.jsonl').exists()
        assert any('bad' in r.getMessage() for r in caplog.records)

    @pytest.mark.skipif(os.geteuid() == 0, reason='root ignores mode bits')
    def test_unreadable_worktree_entry_is_skipped(self, harness: Harness):
        """(d) The on-disk shape of the same property: a permission-denied
        entry is skipped rather than raising out of boot-time recovery.
        """
        blocked = harness.git_ops.worktree_base / 'aaa-blocked' / '.task'
        blocked.mkdir(parents=True, exist_ok=True)
        _seed_transcript(harness, 'zzz-good', 'claude-config-good', 'sess-good')
        blocked.chmod(0o000)
        try:
            n = harness._sweep_orphaned_transcripts()
        finally:
            blocked.chmod(0o755)

        assert n == 1
        assert (_archive_root(harness) / 'good' / ENC / 'sess-good.jsonl').exists()

    def test_deadline_truncation_is_loud(self, harness: Harness, caplog):
        """(e) Deadline-bounded, and it SAYS SO when it truncates.

        Mirrors ``sweep_stale_pid_dirs``: a bounded sweep that returns quietly
        reads to an operator as "swept everything".  A truncated pass must be
        legible as INCOMPLETE, with the examined/archived counts, or the
        remaining held transcripts are silently invisible until the next boot.
        """
        _seed_transcript(harness, '3619', 'claude-config-3619', 'sess-a')

        with caplog.at_level(logging.WARNING):
            n = harness._sweep_orphaned_transcripts(deadline_secs=0.0)

        assert n == 0
        assert not (_archive_root(harness) / '3619' / ENC / 'sess-a.jsonl').exists()
        msgs = ' '.join(r.getMessage().lower() for r in caplog.records)
        assert 'incomplete' in msgs
        assert 'deadline' in msgs

    def test_noop_when_archival_disabled(self, harness: Harness):
        """(f) The kill switch gates the sweeper too — it is archival, not teardown."""
        harness.config.transcript_archive.enabled = False
        _seed_transcript(harness, '3619', 'claude-config-3619', 'sess-a')

        assert harness._sweep_orphaned_transcripts() == 0
        assert not _archive_root(harness).exists()


@pytest.mark.asyncio
class TestSweeperWiredIntoCrashRecovery:
    """Where the sweeper hangs off the boot path, and where it must NOT."""

    async def test_called_once_before_the_first_cleanup(self, harness: Harness):
        """Exactly one call, and it lands BEFORE the entry loop's first
        ``cleanup_worktree`` — a worktree removed first has already taken its
        transcripts with it, so ordering is the whole property.
        """
        order: list[str] = []
        harness._sweep_orphaned_transcripts = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda *a, **k: order.append('sweep') or 0,
        )

        async def _cleanup(*a, **k):
            order.append('cleanup')

        harness.git_ops.cleanup_worktree = AsyncMock(side_effect=_cleanup)
        # A plan-less worktree — the branch that routes to cleanup_worktree.
        (harness.git_ops.worktree_base / '51').mkdir(parents=True, exist_ok=True)

        await harness._recover_crashed_tasks()

        assert order.count('sweep') == 1
        assert 'cleanup' in order
        assert order.index('sweep') < order.index('cleanup')

    async def test_not_called_when_pool_storage_absent(self, harness: Harness):
        """The pool-storage guard defers the ENTIRE recovery pass — including
        the sweep.  Globbing an unmounted worktree_base finds nothing, and a
        "swept 0" tally would read as "no orphans" rather than "not mounted".
        """
        harness._sweep_orphaned_transcripts = MagicMock(  # type: ignore[method-assign]
            return_value=0,
        )
        harness.git_ops.pool_in_use = MagicMock(return_value=True)
        harness.git_ops.pool_storage_present = MagicMock(return_value=False)
        harness._file_pool_storage_absent_escalation = MagicMock()  # type: ignore[method-assign]

        await harness._recover_crashed_tasks()

        harness._sweep_orphaned_transcripts.assert_not_called()
        harness._file_pool_storage_absent_escalation.assert_called_once()
