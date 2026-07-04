"""Tests for the shared guard pipeline ``classify_and_merge`` (MQ-refactor
kappa / task 1995).

Extracts the duplicated pre-merge guard+merge+drop-guard pipeline shared by
``MergeWorker._do_merge``, ``SpeculativeMergeWorker._merger_loop`` (inline
body), and ``SpeculativeMergeWorker._remerge`` into one module-level async
function, ``classify_and_merge(worker, req, base_sha, *, speculative,
started_monotonic) -> MergedOk | Decided``, and routes all three consumers
through it.

Steps covered (TDD order):
  prereq  —    re-anchor line numbers + this fixture scaffold (no prod change)
  step-0  RED   — MergedOk / Decided value-type tests
  step-2  GREEN — add MergedOk / Decided dataclasses + merge_queue re-export
  step-3  RED   — classify_and_merge guard matrix on SpeculativeMergeWorker
  step-4  GREEN — implement classify_and_merge
  step-5  GREEN — adopt classify_and_merge in _merger_loop
  step-6  RED   — classify_and_merge guard matrix on MergeWorker (serial)
  step-7  GREEN — capability-gate SpeculativeMergeWorker-only behaviour
  step-8  GREEN — adopt classify_and_merge in _do_merge
  step-9  RED   — parameterized path-equivalence (merger loop vs _remerge)
  step-10 GREEN — wire _remerge through classify_and_merge
  step-11 GREEN — finalize: docs + full-suite + frozen-surface check

This module reuses the standard per-file fixture quartet (git_repo/git_ops/
git_config/config) and helper functions (_make_request, _make_branch_with_file,
_make_event_store, _count_events) copied from test_merge_queue.py — there is
no shared conftest git_ops fixture; per-file duplication is the established
convention (see test_merge_queue_resource_audit.py's module docstring).
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest
from _orch_helpers import make_placeholder_future

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps, MergeResult, _run
from orchestrator.merge_queue import (
    DROPPED_PLAN_TARGETS_REASON_PREFIX,
    Decided,
    DropGuardResult,
    MergedOk,
    MergeOutcome,
    MergeRequest,
    MergeWorker,
    SpeculativeMergeWorker,
)

# ---------------------------------------------------------------------------
# Fixtures (copied from test_merge_queue.py:62-101 — per-file duplication
# convention, no shared conftest git_ops fixture)
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


async def _setup_repo(repo: Path):
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        # Tests use a tmp repo with no real remote; disabling the push avoids
        # per-test subprocess noise.
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


# ---------------------------------------------------------------------------
# Helpers (copied from test_merge_queue.py: _make_request:104,
# _make_branch_with_file:2313, _make_event_store:11719, _count_events:11698)
# ---------------------------------------------------------------------------


def _make_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
    pre_rebased: bool = False,
    lane: Literal['normal', 'high'] = 'normal',
    merge_first_enqueued_at: float | None = None,
    request_id: str | None = None,
) -> MergeRequest:
    try:
        future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
    except RuntimeError:
        future = make_placeholder_future()
    kwargs: dict = dict(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=pre_rebased,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
        lane=lane,
        merge_first_enqueued_at=merge_first_enqueued_at,
    )
    if request_id is not None:
        kwargs['request_id'] = request_id
    return MergeRequest(**kwargs)


async def _make_branch_with_file(
    git_ops: GitOps,
    branch_name: str,
    filename: str,
    content: str,
) -> Path:
    """Create a worktree branch with one committed file and return its path."""
    worktree = (await git_ops.create_worktree(branch_name)).path
    (worktree / filename).write_text(content)
    await git_ops.commit(worktree, f'Add {filename}')
    return worktree


def _make_event_store(tmp_path: Path) -> EventStore:
    """Create an EventStore backed by a tmp sqlite db.

    Module-level (rather than test_merge_queue.py's per-class-method copy)
    since this file has a single fixture scope — see module docstring.
    """
    db = tmp_path / 'guard_pipeline_events.db'
    return EventStore(db_path=db, run_id='guard-pipeline-test')


def _count_events(db_path, event_type: str) -> int:
    """Query the EventStore SQLite DB for a specific event type count."""
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            'SELECT COUNT(*) FROM events WHERE event_type = ?', (event_type,),
        ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


def _merge_attempt_subtypes(db_path) -> list[str]:
    """Return the ordered list of merge_attempt outcome subtypes.

    Used by the guard-matrix tests (step-3/step-6) and the path-equivalence
    test (step-9) to assert the SAME event stream is emitted regardless of
    which consumer (merger loop / _remerge / _do_merge) drove
    classify_and_merge.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT json_extract(data, '$.outcome') FROM events "
            "WHERE event_type = 'merge_attempt' ORDER BY id",
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# step-0 (RED): MergedOk / Decided value types.
#
# MergedOk/Decided don't exist yet — imported LOCALLY inside each test
# (mirrors test_merge_queue_resource_audit.py / test_merge_request_ledger.py
# convention) so the not-yet-implemented symbols don't break collection of
# the rest of this file across the incremental RED/GREEN steps that follow.
# ---------------------------------------------------------------------------


class TestMergedOkDecidedTypes:
    """Value-type tests for MergedOk / Decided (step-0 / step-2, task 1995)."""

    def test_merged_ok_exposes_merge_result_wt_and_branch_tip(self):
        from orchestrator.merge_types import MergedOk

        mr = MergeResult(success=True, conflicts=False, details='ok')
        ok = MergedOk(merge_result=mr, merge_wt=Path('/x'), branch_tip='abc123')

        assert ok.merge_result is mr
        assert ok.merge_wt == Path('/x')
        assert ok.branch_tip == 'abc123'

    def test_decided_exposes_outcome_and_defaults_merge_result_to_none(self):
        from orchestrator.merge_types import Decided

        d = Decided(outcome=MergeOutcome('conflict', conflict_details='x'))

        assert d.outcome.status == 'conflict'
        assert d.outcome.conflict_details == 'x'
        assert d.merge_result is None

    def test_decided_carries_merge_result_when_explicitly_set(self):
        from orchestrator.merge_types import Decided

        mr = MergeResult(success=False, conflicts=True, details='c')
        d = Decided(outcome=MergeOutcome('blocked'), merge_result=mr)

        assert d.merge_result is mr

    def test_merged_ok_and_decided_reexported_from_merge_queue(self):
        """`from orchestrator.merge_queue import MergedOk, Decided` must work
        and resolve to the SAME objects defined in merge_types.py (shim, not
        a parallel redefinition) — mirrors the existing MergeOutcome/
        SpeculativeItem re-export pattern."""
        from orchestrator.merge_queue import Decided as Decided_via_queue
        from orchestrator.merge_queue import MergedOk as MergedOk_via_queue
        from orchestrator.merge_types import Decided as Decided_via_types
        from orchestrator.merge_types import MergedOk as MergedOk_via_types

        assert MergedOk_via_queue is MergedOk_via_types
        assert Decided_via_queue is Decided_via_types


# ---------------------------------------------------------------------------
# step-3 (RED): classify_and_merge guard matrix on SpeculativeMergeWorker.
#
# classify_and_merge doesn't exist yet — imported LOCALLY inside each test
# (same convention as TestMergedOkDecidedTypes above) so the missing symbol
# doesn't break collection of the rest of this file.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestClassifyAndMergeSpeculativeWorker:
    """Direct-drive guard-matrix coverage for classify_and_merge on the
    speculative (capability-gated diagnostic/drift) worker (step-3, task 1995).
    """

    async def test_missing_branch_returns_decided_unknown_branch(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path, tmp_path: Path,
    ):
        from orchestrator.merge_queue import classify_and_merge

        es = _make_event_store(tmp_path)
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, event_store=es)
        # git_repo (== git_ops.project_root) as the worktree: HEAD is main's own
        # tip, trivially an ancestor of main, so the worktree-HEAD fallback in
        # _classify_branch_presence does NOT kick in and the branch falls
        # through to unknown_branch (no live ref, no merge marker).
        req = _make_request('ghost-1', 'ghost-branch-nope', git_repo, config)
        main_sha = await git_ops.get_main_sha()

        result = await classify_and_merge(
            worker, req, main_sha, speculative=False, started_monotonic=time.monotonic(),
        )

        assert isinstance(result, Decided)
        assert result.outcome.status == 'unknown_branch'
        assert _merge_attempt_subtypes(es.db_path) == ['unknown_branch']

    async def test_already_merged_returns_decided_already_merged(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        from orchestrator.merge_queue import classify_and_merge

        worktree = await _make_branch_with_file(git_ops, 'already-merged-1', 'am.py', 'x = 1\n')
        merge_result = await git_ops.merge_to_main(worktree, 'already-merged-1')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        await git_ops.advance_main(merge_result.merge_commit)
        if merge_result.merge_worktree:
            await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        # `worktree` (the source, not merge_result.merge_worktree) is left
        # alive with a live branch ref whose HEAD is now an ancestor of main
        # — this exercises classify_and_merge's OWN already-merged check
        # (distinct from _classify_branch_presence's marker-based check,
        # which requires the ref to be ABSENT).

        es = _make_event_store(tmp_path)
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, event_store=es)
        req = _make_request('already-merged-1', 'already-merged-1', worktree, config)
        main_sha = await git_ops.get_main_sha()

        result = await classify_and_merge(
            worker, req, main_sha, speculative=False, started_monotonic=time.monotonic(),
        )

        assert isinstance(result, Decided)
        assert result.outcome.status == 'already_merged'
        assert _merge_attempt_subtypes(es.db_path) == ['already_merged']

    async def test_real_conflict_returns_decided_conflict_and_records_drift_sample(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        from orchestrator.merge_queue import classify_and_merge

        worktree = (await git_ops.create_worktree('conflict-cm-1')).path
        (git_ops.project_root / 'README.md').write_text('# Main version\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Main change'], cwd=git_ops.project_root)
        (worktree / 'README.md').write_text('# Task version\n')
        await git_ops.commit(worktree, 'Task change')

        es = _make_event_store(tmp_path)
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, event_store=es)
        req = _make_request('conflict-cm-1', 'conflict-cm-1', worktree, config)
        main_sha = await git_ops.get_main_sha()

        # Mirrors _merger_loop's pre-dequeue bookkeeping (stays inline at call
        # sites per design — NOT part of classify_and_merge), so
        # _note_conflict_detected has a real stashed base to pop.
        worker._note_merge_started(req.request_id)
        drift_count_before = worker._merge_metrics.drift_summary()['count']

        result = await classify_and_merge(
            worker, req, main_sha, speculative=False, started_monotonic=time.monotonic(),
        )

        assert isinstance(result, Decided)
        assert result.outcome.status == 'conflict'
        assert result.outcome.conflict_details  # non-empty
        assert _merge_attempt_subtypes(es.db_path) == ['conflict']
        drift_count_after = worker._merge_metrics.drift_summary()['count']
        assert drift_count_after == drift_count_before + 1, (
            'expected _note_conflict_detected to record one drift sample'
        )

    async def test_non_conflict_merge_failure_returns_decided_blocked_with_diagnostic(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from orchestrator.merge_queue import classify_and_merge

        worktree = await _make_branch_with_file(
            git_ops, 'nonconflict-fail-1', 'f.py', 'x = 1\n',
        )
        main_sha = await git_ops.get_main_sha()

        async def _fake_merge_to_main(wt, br, base_sha=None):
            return MergeResult(
                success=False,
                conflicts=False,
                details='fatal: task/nonconflict-fail-1 - not something we can merge',
                pre_merge_sha=main_sha,
            )

        monkeypatch.setattr(git_ops, 'merge_to_main', _fake_merge_to_main)

        es = _make_event_store(tmp_path)
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, event_store=es)
        req = _make_request('nonconflict-fail-1', 'nonconflict-fail-1', worktree, config)

        result = await classify_and_merge(
            worker, req, main_sha, speculative=False, started_monotonic=time.monotonic(),
        )

        assert isinstance(result, Decided)
        assert result.outcome.status == 'blocked'
        assert result.outcome.failure_diagnostic is not None
        assert set(result.outcome.failure_diagnostic.keys()) == {
            'base_sha', 'base_label', 'branch_ref_in_worktree', 'git_stderr',
        }
        assert result.merge_result is not None
        assert result.merge_result.success is False
        # Infrastructure-failure blocked outcomes are intentionally NOT
        # emitted as merge_attempt (see _emit_merge_attempt's docstring).
        assert _count_events(es.db_path, 'merge_attempt') == 0

    async def test_drop_guard_returns_decided_blocked_dropped_plan_targets(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from orchestrator.merge_queue import classify_and_merge

        worktree = (await git_ops.create_worktree('drop-guard-cm-1')).path
        (worktree / 'retained.py').write_text('retained = 1\n')
        await git_ops.commit(worktree, 'Add retained')
        rc, pre_drop_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
        assert rc == 0
        pre_drop_sha = pre_drop_sha.strip()

        (worktree / 'dropped.py').write_text('dropped = 1\n')
        await git_ops.commit(worktree, 'Add dropped')

        artifacts = TaskArtifacts(worktree)
        artifacts.init('drop-guard-cm-1', 'Drop guard cm', 'desc')
        artifacts.write_plan({
            'files': ['retained.py', 'dropped.py'],
            'modules': [],
            'steps': [],
        })

        async def _fake_merge_to_main(*_args, **_kwargs) -> MergeResult:
            return MergeResult(
                success=True,
                merge_commit=pre_drop_sha,
                pre_merge_sha=pre_drop_sha,
                merge_worktree=None,
            )

        monkeypatch.setattr(git_ops, 'merge_to_main', _fake_merge_to_main)

        es = _make_event_store(tmp_path)
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, event_store=es)
        req = _make_request('drop-guard-cm-1', 'drop-guard-cm-1', worktree, config)
        main_sha = await git_ops.get_main_sha()

        result = await classify_and_merge(
            worker, req, main_sha, speculative=False, started_monotonic=time.monotonic(),
        )

        assert isinstance(result, Decided)
        assert result.outcome.status == 'blocked'
        assert result.outcome.reason.startswith(DROPPED_PLAN_TARGETS_REASON_PREFIX)
        assert _merge_attempt_subtypes(es.db_path) == ['dropped_plan_targets']

    async def test_clean_success_returns_merged_ok(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        from orchestrator.merge_queue import classify_and_merge

        worktree = await _make_branch_with_file(git_ops, 'success-cm-1', 'ok.py', 'x = 1\n')
        rc, expected_tip, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
        assert rc == 0
        expected_tip = expected_tip.strip()

        es = _make_event_store(tmp_path)
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, event_store=es)
        req = _make_request('success-cm-1', 'success-cm-1', worktree, config)
        main_sha = await git_ops.get_main_sha()

        result = await classify_and_merge(
            worker, req, main_sha, speculative=False, started_monotonic=time.monotonic(),
        )

        assert isinstance(result, MergedOk)
        assert result.merge_result.success
        assert result.merge_wt is not None
        assert result.branch_tip == expected_tip

        if result.merge_wt:
            await git_ops.cleanup_merge_worktree(result.merge_wt)

    async def test_speculative_merge_event_emitted_when_speculative_true(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        from orchestrator.merge_queue import classify_and_merge

        worktree = await _make_branch_with_file(git_ops, 'spec-evt-true', 'f.py', 'x = 1\n')
        es = _make_event_store(tmp_path)
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, event_store=es)
        req = _make_request('spec-evt-true', 'spec-evt-true', worktree, config)
        main_sha = await git_ops.get_main_sha()

        result = await classify_and_merge(
            worker, req, main_sha, speculative=True, started_monotonic=time.monotonic(),
        )

        assert isinstance(result, MergedOk)
        assert _count_events(es.db_path, 'speculative_merge') == 1

        if result.merge_wt:
            await git_ops.cleanup_merge_worktree(result.merge_wt)

    async def test_speculative_merge_event_not_emitted_when_speculative_false(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        from orchestrator.merge_queue import classify_and_merge

        worktree = await _make_branch_with_file(git_ops, 'spec-evt-false', 'f.py', 'x = 1\n')
        es = _make_event_store(tmp_path)
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, event_store=es)
        req = _make_request('spec-evt-false', 'spec-evt-false', worktree, config)
        main_sha = await git_ops.get_main_sha()

        result = await classify_and_merge(
            worker, req, main_sha, speculative=False, started_monotonic=time.monotonic(),
        )

        assert isinstance(result, MergedOk)
        assert _count_events(es.db_path, 'speculative_merge') == 0

        if result.merge_wt:
            await git_ops.cleanup_merge_worktree(result.merge_wt)


# ---------------------------------------------------------------------------
# step-6: classify_and_merge on the SERIAL worker (MergeWorker) preserves
# _do_merge's exact quirks (no drift bookkeeping, no rich diagnostic).
#
# classify_and_merge already capability-gates the SpeculativeMergeWorker-only
# calls (isinstance check added in step-4, ahead of the plan's step-7), so
# this suite is GREEN on first run rather than RED — see the commit message
# for the full accounting. Written as originally scoped for the guard-matrix
# coverage regardless.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestClassifyAndMergeSerialWorker:
    """Direct-drive guard-matrix coverage for classify_and_merge on the plain
    (non-capability) serial worker (step-6, task 1995).
    """

    async def test_real_conflict_returns_decided_conflict_no_drift_side_effect(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        from orchestrator.merge_queue import classify_and_merge

        worktree = (await git_ops.create_worktree('conflict-serial-1')).path
        (git_ops.project_root / 'README.md').write_text('# Main version\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Main change'], cwd=git_ops.project_root)
        (worktree / 'README.md').write_text('# Task version\n')
        await git_ops.commit(worktree, 'Task change')

        es = _make_event_store(tmp_path)
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        # MergeWorker has neither _merge_metrics nor _note_conflict_detected —
        # classify_and_merge must not raise AttributeError reaching for them.
        worker = MergeWorker(git_ops, queue, event_store=es)
        req = _make_request('conflict-serial-1', 'conflict-serial-1', worktree, config)
        main_sha = await git_ops.get_main_sha()

        result = await classify_and_merge(
            worker, req, main_sha, speculative=False, started_monotonic=time.monotonic(),
        )

        assert isinstance(result, Decided)
        assert result.outcome.status == 'conflict'
        assert result.outcome.conflict_details
        assert _merge_attempt_subtypes(es.db_path) == ['conflict']

    async def test_non_conflict_merge_failure_returns_plain_reason_no_diagnostic(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from orchestrator.merge_queue import classify_and_merge

        worktree = await _make_branch_with_file(
            git_ops, 'nonconflict-serial-1', 'f.py', 'x = 1\n',
        )
        main_sha = await git_ops.get_main_sha()

        async def _fake_merge_to_main(wt, br, base_sha=None):
            return MergeResult(success=False, conflicts=False, details='boom')

        monkeypatch.setattr(git_ops, 'merge_to_main', _fake_merge_to_main)

        es = _make_event_store(tmp_path)
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue, event_store=es)
        req = _make_request('nonconflict-serial-1', 'nonconflict-serial-1', worktree, config)

        result = await classify_and_merge(
            worker, req, main_sha, speculative=False, started_monotonic=time.monotonic(),
        )

        assert isinstance(result, Decided)
        assert result.outcome.status == 'blocked'
        assert result.outcome.reason == 'boom'
        assert result.outcome.failure_diagnostic is None
        assert _count_events(es.db_path, 'merge_attempt') == 0

    async def test_already_merged_honors_snapshot_tip(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """Mirrors test_merge_queue.py's TestClassifyBranchPresence-adjacent
        test_already_merged_honors_snapshot_tip (task 1917 regression):
        branch tip SNAP is merged onto main, worktree HEAD is then rewritten
        to a DIVERGENT commit D (non-ancestor of main); req.snapshot_tip=SNAP
        must still resolve to already_merged."""
        from orchestrator.merge_queue import classify_and_merge

        wt = (await git_ops.create_worktree('snap-cm-1')).path
        (wt / 'snap_cm.py').write_text('snap = 1\n')
        await git_ops.commit(wt, 'Add snap_cm.py')

        rc, snap_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        assert rc == 0
        snap = snap_out.strip()

        merge_result = await git_ops.merge_to_main(wt, 'snap-cm-1')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        await git_ops.advance_main(
            merge_result.merge_commit, merge_result.merge_worktree,
            branch='snap-cm-1', max_attempts=1,
        )
        if merge_result.merge_worktree:
            await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

        # Rewrite worktree HEAD to a DIVERGENT commit D (non-ancestor of main).
        (wt / 'snap_cm.py').write_text('DIVERGENT = 999\n')
        rc, _, _ = await _run(['git', 'add', 'snap_cm.py'], cwd=wt)
        assert rc == 0
        rc, _, _ = await _run(['git', 'commit', '-m', 'Divergent'], cwd=wt)
        assert rc == 0

        es = _make_event_store(tmp_path)
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue, event_store=es)
        req = _make_request('snap-cm-1', 'snap-cm-1', wt, config)
        req.snapshot_tip = snap  # the already-merged tip — check must use this
        main_sha = await git_ops.get_main_sha()

        result = await classify_and_merge(
            worker, req, main_sha, speculative=False, started_monotonic=time.monotonic(),
        )

        assert isinstance(result, Decided)
        assert result.outcome.status == 'already_merged', (
            f'expected already_merged (snapshot_tip=SNAP is ancestor of main); '
            f'got {result.outcome.status!r}'
        )

    async def test_drop_guard_uses_passed_base_sha(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """The base_sha parameter — not some internally-recomputed value —
        must reach _check_plan_targets_in_tree as its main_sha argument."""
        from orchestrator.merge_queue import classify_and_merge

        worktree = await _make_branch_with_file(
            git_ops, 'dropguard-base-1', 'f.py', 'x = 1\n',
        )
        custom_base_sha = 'deadbeef' * 5  # 40 hex chars; deliberately not real main

        captured_base_shas: list[str] = []

        async def _spy_check_plan_targets(
            merge_commit_sha, task_worktree, git_ops_arg, main_sha, *, task_id=None,
        ):
            captured_base_shas.append(main_sha)
            return DropGuardResult(dropped=[])

        es = _make_event_store(tmp_path)
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue, event_store=es)
        req = _make_request('dropguard-base-1', 'dropguard-base-1', worktree, config)

        with patch(
            'orchestrator.merge_queue._check_plan_targets_in_tree',
            _spy_check_plan_targets,
        ):
            result = await classify_and_merge(
                worker, req, custom_base_sha, speculative=False,
                started_monotonic=time.monotonic(),
            )

        assert isinstance(result, MergedOk)
        assert captured_base_shas == [custom_base_sha]

        if result.merge_wt:
            await git_ops.cleanup_merge_worktree(result.merge_wt)
