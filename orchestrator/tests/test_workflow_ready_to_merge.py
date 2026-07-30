"""Tests for ``TaskWorkflow._handle_ready_to_merge_report``.

The handler processes the architect's ``.task/ready_to_merge.json`` artifact —
the *merge-landing desync* exit (PRD ``plans/architect-already-complete-exits
.md`` §β).  The architect reports that this task's work is complete on the
BRANCH and only the physical merge to main is missing; today the advised exit
(``report_unactionable_task``) opens an L1 that costs a human ~100k tokens AND
vetoes the verified-green auto-merge reaper.

The handler NEVER trusts the report.  It re-validates the desync predicate
first-hand — clean fast-forward of main, verify PASSED on this exact tip,
review PASS on this exact tree — and only then enqueues a
``MergeRequest(source='architect-desync')``.  It never lands main itself: the
merge worker's scoped re-verify remains the sole gate.  Predicate misses route
to ``_mark_blocked`` WITHOUT a human escalation (an architect mistake, not an
unworkable spec), carrying the failed predicate + its measured values.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

_TASK_ID = '50'
_TIP = 'aaaa111122223333444455556666777788889999'
_MAIN = 'bbbb1111222233334444555566667777888899aa'
_TREE = 'cccc1111222233334444555566667777888899bb'


@dataclass
class _Fixture:
    wf: TaskWorkflow
    artifacts: TaskArtifacts
    merge_queue: asyncio.Queue
    event_store: EventStore
    scheduler: MagicMock
    git_ops: MagicMock
    mark_blocked: AsyncMock
    outcome: WorkflowOutcome | None = None

    def marker_stamps(self) -> list[dict]:
        """Every ``architect_merge_request`` payload written via update_task."""
        return [
            call.args[1]['architect_merge_request']
            for call in self.scheduler.update_task.await_args_list
            if len(call.args) > 1
            and isinstance(call.args[1], dict)
            and 'architect_merge_request' in call.args[1]
        ]


def _make(
    *,
    tmp_path: Path,
    task_id: str = _TASK_ID,
    tip: str | None = _TIP,
    main_sha: str = _MAIN,
    main_is_ancestor_of_tip: bool = True,
    tip_is_ancestor_of_main: bool = False,
    verified_tip: str | None = _TIP,
    tree_hash: str | None = _TREE,
    review_verdict: str | None = 'PASS',
    metadata: dict | None = None,
    with_merge_queue: bool = True,
) -> _Fixture:
    """Build a TaskWorkflow with fakes wired for the desync predicate.

    Defaults describe the C-1 happy path: the branch is a clean fast-forward of
    main, verify passed on ``tip``, and review returned PASS on ``tree_hash``.
    Each keyword breaks exactly one predicate for the C-2 reject cases.
    """
    worktree = tmp_path / 'wt'
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {
        'id': task_id, 'title': 'T', 'description': 'd',
        'metadata': metadata if metadata is not None else {},
    }
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.claimant_heartbeat_interval_secs = 60.0
    config.project_root = tmp_path / 'proj'
    config.git.branch_prefix = 'task/'
    config.module_configs_or_empty = {}

    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.get_status = AsyncMock(return_value=None)
    scheduler.mark_done = AsyncMock()

    async def _is_ancestor(ancestor: str, descendant: str) -> bool:
        if ancestor == main_sha and descendant == tip:
            return main_is_ancestor_of_tip
        if ancestor == tip and descendant == main_sha:
            return tip_is_ancestor_of_main
        return False

    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value=main_sha)
    git_ops.resolve_branch_sha = AsyncMock(return_value=tip)
    git_ops.is_ancestor = AsyncMock(side_effect=_is_ancestor)
    git_ops.get_head_tree_hash = AsyncMock(return_value=tree_hash)

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )

    worktree.mkdir(parents=True, exist_ok=True)
    artifacts = TaskArtifacts(worktree)
    artifacts.init(task_id, 'T', 'd', base_commit='oldbase')
    if review_verdict is not None and tree_hash is not None:
        artifacts.record_review_verdict(tree_hash, review_verdict, False)
    wf.artifacts = artifacts
    wf.worktree = worktree

    event_store = EventStore(tmp_path / 'runs.db', 'run-1')
    if verified_tip is not None:
        event_store.emit(
            EventType.workflow_verify,
            task_id=task_id,
            data={'passed': True, 'tip_sha': verified_tip},
        )
    wf.event_store = event_store

    merge_queue: asyncio.Queue = asyncio.Queue()
    wf.merge_queue = merge_queue if with_merge_queue else None

    # _mark_blocked is a heavy terminal path (status writes + escalation
    # filing); stub it so the tests observe the CALL, not its side effects.
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    return _Fixture(
        wf=wf, artifacts=artifacts, merge_queue=merge_queue,
        event_store=event_store, scheduler=scheduler, git_ops=git_ops,
        mark_blocked=mark_blocked,
    )


# ---------------------------------------------------------------------------
# C-1 — the happy path: clean FF + verify PASSED + review PASS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReadyToMergeHappyPath:
    """A genuine merge-landing desync enqueues a deterministic merge."""

    async def _run(self, tmp_path: Path, **kw) -> _Fixture:
        f = _make(tmp_path=tmp_path, **kw)
        f.artifacts.write_ready_to_merge(
            commit=_TIP, evidence='clean FF of main; verify PASSED; review PASS',
        )
        f.outcome = await f.wf._handle_ready_to_merge_report()
        return f

    async def test_enqueues_exactly_one_merge_request(self, tmp_path: Path):
        f = await self._run(tmp_path)

        assert f.merge_queue.qsize() == 1
        req = f.merge_queue.get_nowait()
        assert req.task_id == _TASK_ID
        assert req.branch.bare_id == _TASK_ID
        assert req.snapshot_tip == _TIP
        assert req.pre_rebased is False

    async def test_tags_the_merge_queued_event_source(self, tmp_path: Path):
        """``source='architect-desync'`` keeps this submit distinguishable from
        the stranded reaper's in the event stream."""
        f = await self._run(tmp_path)

        rows = f.event_store.fetch_events_by_type_all_runs(
            EventType.merge_queued, task_id=_TASK_ID,
        )
        assert len(rows) == 1
        assert (rows[0].get('data') or {}).get('source') == 'architect-desync'

    async def test_stamps_the_idempotency_marker(self, tmp_path: Path):
        f = await self._run(tmp_path)

        stamps = f.marker_stamps()
        assert len(stamps) == 1
        assert stamps[0]['tip_sha'] == _TIP
        assert stamps[0]['request_id'].startswith('mr-')

    async def test_files_no_human_escalation(self, tmp_path: Path):
        """The whole point of the exit: no L1, so no human pays ~100k tokens
        and the verified-green auto-merge reaper is not vetoed."""
        f = await self._run(tmp_path)

        f.mark_blocked.assert_not_awaited()

    async def test_clears_the_artifact(self, tmp_path: Path):
        f = await self._run(tmp_path)

        assert f.artifacts.read_ready_to_merge() is None

    async def test_leaves_task_blocked_not_done_inline(self, tmp_path: Path):
        """The handler never marks done itself — the merge queue's re-verify is
        the sole gate, and the done flip comes from the merge callback."""
        f = await self._run(tmp_path)

        assert f.outcome == WorkflowOutcome.BLOCKED
        f.scheduler.mark_done.assert_not_awaited()

    async def test_suggestions_only_verdict_also_passes(self, tmp_path: Path):
        """``record_review_verdict`` only ever caches non-blocking verdicts, so
        suggestions_only is as much proof of a review PASS as PASS itself."""
        f = await self._run(tmp_path, review_verdict='suggestions_only')

        assert f.merge_queue.qsize() == 1
        f.mark_blocked.assert_not_awaited()


# ---------------------------------------------------------------------------
# C-2 — reject: any single broken predicate blocks WITHOUT a human escalation
# ---------------------------------------------------------------------------

# (kwargs that break exactly one predicate, expected `predicate` event name).
_BROKEN_PREDICATES = [
    pytest.param(
        {'tip_is_ancestor_of_main': True}, 'clean_ff', id='already-on-main',
    ),
    pytest.param(
        {'main_is_ancestor_of_tip': False}, 'clean_ff', id='diverged-from-main',
    ),
    pytest.param(
        {'verified_tip': 'd' * 40}, 'verify_passed', id='verify-on-other-tip',
    ),
    pytest.param(
        {'verified_tip': None}, 'verify_passed', id='verify-never-passed',
    ),
    pytest.param(
        {'review_verdict': None}, 'review_pass', id='no-cached-review-verdict',
    ),
    pytest.param(
        {'tree_hash': None}, 'review_pass', id='tree-hash-unresolvable',
    ),
    pytest.param(
        {'tip': 'e' * 40, 'verified_tip': 'e' * 40}, 'cited_commit',
        id='cited-commit-is-not-the-tip',
    ),
    pytest.param({'tip': None}, 'branch_resolved', id='branch-does-not-resolve'),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(('broken', 'predicate'), _BROKEN_PREDICATES)
class TestReadyToMergeRejects:
    """A false desync claim is an ARCHITECT MISTAKE, not an unworkable spec —
    it blocks without escalating, and never enqueues a merge."""

    async def _run(self, tmp_path: Path, broken: dict) -> _Fixture:
        f = _make(tmp_path=tmp_path, **broken)
        f.artifacts.write_ready_to_merge(commit=_TIP, evidence='ev')
        f.outcome = await f.wf._handle_ready_to_merge_report()
        return f

    async def test_blocks_without_escalating_to_human(
        self, tmp_path: Path, broken: dict, predicate: str,
    ):
        f = await self._run(tmp_path, broken)

        assert f.outcome == WorkflowOutcome.BLOCKED
        f.mark_blocked.assert_awaited_once()
        assert f.mark_blocked.await_args.kwargs.get('escalate_to_human') is not True

    async def test_enqueues_no_merge_request(
        self, tmp_path: Path, broken: dict, predicate: str,
    ):
        f = await self._run(tmp_path, broken)

        assert f.merge_queue.qsize() == 0

    async def test_stamps_no_marker(
        self, tmp_path: Path, broken: dict, predicate: str,
    ):
        f = await self._run(tmp_path, broken)

        assert f.marker_stamps() == []

    async def test_emits_structured_reject_event(
        self, tmp_path: Path, broken: dict, predicate: str,
    ):
        """INV-2 structured-facts-at-failure: the FAILED PREDICATE and the
        values it was judged against land on an event, not a log line."""
        f = await self._run(tmp_path, broken)

        rows = f.event_store.fetch_events_by_type_all_runs(
            EventType.architect_desync_merge, task_id=_TASK_ID,
        )
        assert len(rows) == 1
        data = rows[0].get('data') or {}
        assert data['decision'] == 'rejected'
        assert data['predicate'] == predicate
        assert isinstance(data.get('measured'), dict) and data['measured']

    async def test_block_reason_names_the_failed_predicate(
        self, tmp_path: Path, broken: dict, predicate: str,
    ):
        """The blocked record is self-describing without cross-referencing the
        event stream — reason names the predicate, detail carries the values."""
        f = await self._run(tmp_path, broken)

        reason = f.mark_blocked.await_args.args[0]
        detail = f.mark_blocked.await_args.kwargs.get('detail') or ''
        assert predicate in reason
        assert detail


@pytest.mark.asyncio
class TestReadyToMergeMalformedArtifact:
    """Missing/malformed report and missing infrastructure also block."""

    async def test_missing_commit_blocks(self, tmp_path: Path):
        f = _make(tmp_path=tmp_path)
        f.artifacts.write_ready_to_merge(commit='', evidence='ev')

        outcome = await f.wf._handle_ready_to_merge_report()

        assert outcome == WorkflowOutcome.BLOCKED
        f.mark_blocked.assert_awaited_once()
        assert f.mark_blocked.await_args.kwargs.get('escalate_to_human') is not True
        assert f.merge_queue.qsize() == 0
        assert f.artifacts.read_ready_to_merge() is None  # still cleared

    async def test_unavailable_merge_queue_blocks(self, tmp_path: Path):
        f = _make(tmp_path=tmp_path, with_merge_queue=False)
        f.artifacts.write_ready_to_merge(commit=_TIP, evidence='ev')

        outcome = await f.wf._handle_ready_to_merge_report()

        assert outcome == WorkflowOutcome.BLOCKED
        f.mark_blocked.assert_awaited_once()
        assert f.mark_blocked.await_args.kwargs.get('escalate_to_human') is not True
