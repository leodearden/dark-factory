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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
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
    escalations: Any = None  # the _FakeEscalationQueue, when one is wired

    def marker_stamps(self) -> list[dict]:
        """Every ``architect_merge_request`` payload written via update_task."""
        return [
            call.args[1]['architect_merge_request']
            for call in self.scheduler.update_task.await_args_list
            if len(call.args) > 1
            and isinstance(call.args[1], dict)
            and 'architect_merge_request' in call.args[1]
        ]

    def status_forcing_updates(self) -> list[dict]:
        """Any ``update_task`` payload attempting to write the task status.

        The blocked-row persist must go through ``set_task_status`` alone; a
        terminal-exit rejection must be ACCEPTED, never re-forced through the
        metadata back door.
        """
        return [
            call.args[1]
            for call in self.scheduler.update_task.await_args_list
            if len(call.args) > 1
            and isinstance(call.args[1], dict)
            and 'status' in call.args[1]
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
    set_task_status_side_effect: object = None,
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
    scheduler.set_task_status = AsyncMock(side_effect=set_task_status_side_effect)
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

    async def test_persists_the_blocked_task_row(self, tmp_path: Path):
        """The blocked row must reach the SCHEDULER, not just the in-memory
        phase machine.

        β files no escalation by design, so an in-progress row left behind
        matches the recovery table's REVERT_TO_PENDING shape — (in-progress,
        no live claimant, branch off main, no open escalation)
        (task_ground_truth.py row (c)) — and the sweep would set the task back
        to 'pending' (harness.py:5564) and replan it while its MergeRequest is
        still sitting in the queue.  Persisting 'blocked' is what keeps the
        enqueued merge from being raced by a duplicate replan.
        """
        f = await self._run(tmp_path)

        f.scheduler.set_task_status.assert_awaited_once_with(_TASK_ID, 'blocked')
        f.mark_blocked.assert_not_awaited()  # still no human escalation
        assert f.merge_queue.qsize() == 1

    async def test_terminal_exit_rejection_is_accepted_not_forced(
        self, tmp_path: Path,
    ):
        """The benign already-terminal race: the merge callback or the
        found_on_main backstop beat us to mark_done.

        That row is CORRECT — the work landed.  Dragging it back to 'blocked'
        would undo a legitimate done, so the rejection must be swallowed, not
        retried and not re-forced through ``update_task``.
        """
        from orchestrator.scheduler import TerminalExitRejection

        f = await self._run(
            tmp_path,
            set_task_status_side_effect=TerminalExitRejection(
                task_id=_TASK_ID, old_status='done',
                target_status='blocked', raw='terminal_exit_rejected',
            ),
        )

        assert f.outcome == WorkflowOutcome.BLOCKED
        assert f.merge_queue.qsize() == 1  # the merge still went out
        f.scheduler.set_task_status.assert_awaited_once()  # no retry
        assert f.status_forcing_updates() == []  # and no forced re-write

    async def test_status_write_failure_never_fails_the_enqueue(
        self, tmp_path: Path,
    ):
        """A transient scheduler failure must never turn a successfully
        enqueued merge into a raised exception — the request is already in the
        queue, and the found_on_main reconciler remains the durable backstop.
        """
        f = await self._run(
            tmp_path, set_task_status_side_effect=RuntimeError('boom'),
        )

        assert f.outcome == WorkflowOutcome.BLOCKED
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


# ---------------------------------------------------------------------------
# C-4 — idempotency: a re-invoked architect must not double-enqueue
# ---------------------------------------------------------------------------


def _marker(*, tip: str = _TIP, age_s: float = 0.0) -> dict:
    """An ``architect_merge_request`` metadata marker *age_s* seconds old."""
    return {
        'tip_sha': tip,
        'request_id': 'mr-prev1234',
        'submitted_at': (
            datetime.now(UTC) - timedelta(seconds=age_s)
        ).isoformat(),
    }


@pytest.mark.asyncio
class TestReadyToMergeIdempotency:
    """INV-4 storm-escape.  The marker key is deliberately distinct from the
    stranded reaper's ``stranded_merge_request`` so the two submit paths never
    clobber each other."""

    async def _run(self, tmp_path: Path, marker: object) -> _Fixture:
        f = _make(
            tmp_path=tmp_path,
            metadata={'architect_merge_request': marker},
        )
        f.artifacts.write_ready_to_merge(commit=_TIP, evidence='ev')
        f.outcome = await f.wf._handle_ready_to_merge_report()
        return f

    async def test_fresh_marker_same_tip_skips_resubmit(self, tmp_path: Path):
        f = await self._run(tmp_path, _marker(tip=_TIP))

        assert f.merge_queue.qsize() == 0  # NO second MergeRequest
        assert f.marker_stamps() == []  # and no re-stamp
        assert f.outcome == WorkflowOutcome.BLOCKED
        f.mark_blocked.assert_not_awaited()  # still no human escalation

    async def test_duplicate_skip_persists_the_blocked_task_row(
        self, tmp_path: Path,
    ):
        """The skip exit is just as terminal as the enqueue exit, so it owes
        the same durable 'blocked' row.

        Otherwise the in-progress row left behind matches the recovery table's
        REVERT_TO_PENDING shape (no escalation is open by design) and the task
        gets replanned out from under the merge that is still in flight — the
        exact storm the marker exists to prevent.
        """
        f = await self._run(tmp_path, _marker(tip=_TIP))

        f.scheduler.set_task_status.assert_awaited_once_with(_TASK_ID, 'blocked')
        assert f.merge_queue.qsize() == 0  # still no second MergeRequest

    async def test_duplicate_skip_is_recorded_structurally(self, tmp_path: Path):
        """The skip is a decision, not a silence — it lands on the event."""
        f = await self._run(tmp_path, _marker(tip=_TIP))

        rows = f.event_store.fetch_events_by_type_all_runs(
            EventType.architect_desync_merge, task_id=_TASK_ID,
        )
        assert len(rows) == 1
        assert (rows[0].get('data') or {})['decision'] == 'duplicate'

    async def test_stale_marker_resubmits(self, tmp_path: Path):
        """Fail-safe: a stale marker must self-heal into a fresh submit, never
        a wedged skip."""
        f = await self._run(tmp_path, _marker(tip=_TIP, age_s=7200.0))

        assert f.merge_queue.qsize() == 1
        assert len(f.marker_stamps()) == 1

    async def test_mismatched_tip_marker_resubmits(self, tmp_path: Path):
        f = await self._run(tmp_path, _marker(tip='f' * 40))

        assert f.merge_queue.qsize() == 1
        assert f.merge_queue.get_nowait().snapshot_tip == _TIP

    async def test_malformed_marker_resubmits(self, tmp_path: Path):
        f = await self._run(tmp_path, 'not-a-dict')

        assert f.merge_queue.qsize() == 1


# ---------------------------------------------------------------------------
# C-3 — the merge queue's re-verify is the SOLE gate on main
# ---------------------------------------------------------------------------

_LANDED = '1234abcd5678ef901234abcd5678ef901234abcd'


async def _fire_merge_done(
    tmp_path: Path, outcome: Any, *, escalation_queue: object = None,
) -> _Fixture:
    """Drive ``_on_architect_merge_done`` with a terminal *outcome*."""
    f = _make(tmp_path=tmp_path)
    if escalation_queue is not None:
        f.wf.escalation_queue = escalation_queue  # type: ignore[assignment]
    await f.wf._on_architect_merge_done(outcome, tip=_TIP, evidence='ev')
    return f


@pytest.mark.asyncio
class TestOnArchitectMergeDone:
    """``_on_architect_merge_done`` — the FAST done flip.

    The handler itself never advances main; it only enqueues.  Whether main
    moves is decided entirely by the merge worker, and this callback merely
    reports that decision back onto the task row.  The existing found_on_main
    reconciler is the restart-safe DURABLE backstop for a lost callback.
    """

    _fire = staticmethod(_fire_merge_done)

    async def test_success_marks_done_via_found_on_main(self, tmp_path: Path):
        from orchestrator.merge_types import MergeOutcome

        f = await self._fire(
            tmp_path, MergeOutcome(status='done', merge_sha=_LANDED),
        )

        f.scheduler.mark_done.assert_awaited_once()
        call = f.scheduler.mark_done.await_args
        assert call.args[0] == _TASK_ID
        assert call.kwargs['kind'] == 'found_on_main'
        assert call.kwargs['sha'] == _LANDED
        assert 'architect' in call.kwargs['note']

    @pytest.mark.parametrize('status', ['conflict', 'error', 'unmerged_state'])
    async def test_durable_failure_leaves_task_blocked(
        self, tmp_path: Path, status: str,
    ):
        from orchestrator.merge_types import MergeOutcome

        f = await self._fire(tmp_path, MergeOutcome(status=cast(Any, status)))

        f.scheduler.mark_done.assert_not_awaited()
        rows = f.event_store.fetch_events_by_type_all_runs(
            EventType.architect_desync_merge, task_id=_TASK_ID,
        )
        assert len(rows) == 1
        data = rows[0].get('data') or {}
        assert data['decision'] == 'merge_failed'
        assert data['status'] == status

    @pytest.mark.parametrize('status', ['already_merged', 'superseded'])
    async def test_transient_success_status_is_a_no_op_for_failure(
        self, tmp_path: Path, status: str,
    ):
        """A superseded/already-merged outcome is not a failure — no L2, no
        failure event; the branch and lane are preserved by omission."""
        from orchestrator.merge_types import MergeOutcome

        f = await self._fire(tmp_path, MergeOutcome(status=cast(Any, status)))

        rows = f.event_store.fetch_events_by_type_all_runs(
            EventType.architect_desync_merge, task_id=_TASK_ID,
        )
        assert [
            r for r in rows
            if (r.get('data') or {}).get('decision') == 'merge_failed'
        ] == []

    async def test_unclassified_status_is_treated_as_a_durable_failure(
        self, tmp_path: Path,
    ):
        """Loud over silent: a status in neither classification set must not
        silently no-op into a stranded task with no operator signal."""
        from orchestrator.merge_types import MergeOutcome

        f = await self._fire(
            tmp_path, MergeOutcome(status='brand_new_status'),  # type: ignore[arg-type]
        )

        f.scheduler.mark_done.assert_not_awaited()
        rows = f.event_store.fetch_events_by_type_all_runs(
            EventType.architect_desync_merge, task_id=_TASK_ID,
        )
        assert (rows[0].get('data') or {})['decision'] == 'merge_failed'

    async def test_success_without_a_merge_sha_does_not_mark_done(
        self, tmp_path: Path,
    ):
        """found_on_main provenance needs a landed SHA — a 'done' with none is
        an unusable signal, so degrade to the reconciler backstop rather than
        writing bogus provenance."""
        from orchestrator.merge_types import MergeOutcome

        f = await self._fire(tmp_path, MergeOutcome(status='done'))

        f.scheduler.mark_done.assert_not_awaited()


class _FakeEscalationQueue:
    """Minimal stand-in for ``EscalationQueue`` (submit / get_by_task / make_id)."""

    def __init__(self) -> None:
        self.submitted: list[Any] = []
        self._n = 0

    def make_id(self, task_id: str) -> str:
        self._n += 1
        return f'esc-{task_id}-{self._n}'

    def submit(self, escalation: Any) -> str:
        self.submitted.append(escalation)
        return escalation.id

    def get_by_task(
        self, task_id: str, status: str | None = None,
        level: int | None = None, agent_role: str | None = None,
    ) -> list[Any]:
        return [
            e for e in self.submitted
            if e.task_id == task_id
            and (level is None or e.level == level)
            and (agent_role is None or e.agent_role == agent_role)
        ]


@pytest.mark.asyncio
class TestArchitectMergeDurableFailureEscalates:
    """A durable merge failure must reach a HUMAN, not just a log line.

    The enqueue exit deliberately leaves the task ``blocked`` with no open
    escalation — LEAVE-shaped for the recovery sweep — and the verified-green
    stranded reaper cannot re-drive it either (``detect_verified_green``
    requires an assigned lane AND an all-steps-done plan.json; this exit fires
    precisely when the architect wrote no plan).  So on a durable failure the
    born-at-L2 is the ONLY operator signal; without it the exit that exists to
    un-strand a branch would itself strand the task forever.
    """

    @pytest.mark.parametrize('status', ['conflict', 'error', 'unmerged_state'])
    async def test_durable_failure_files_a_born_at_l2(
        self, tmp_path: Path, status: str,
    ):
        from orchestrator.merge_types import MergeOutcome

        q = _FakeEscalationQueue()
        f = await self._fire(
            tmp_path, MergeOutcome(status=cast(Any, status), reason='boom'),
            escalation_queue=q,
        )

        assert len(q.submitted) == 1, 'durable failure filed no escalation'
        esc = q.submitted[0]
        assert esc.task_id == _TASK_ID
        assert esc.category == 'stranded_merge_failed'
        assert esc.severity == 'critical'
        assert esc.level == 2  # born-at-L2: bypasses the auto-watcher
        assert esc.agent_role.startswith('orchestrator-')
        assert status in esc.summary
        assert 'boom' in esc.detail and _TIP in esc.detail
        f.scheduler.mark_done.assert_not_awaited()

    async def test_unclassified_status_also_files_the_l2(self, tmp_path: Path):
        """'Loud' for an unclassified outcome means an escalation too — the
        shared DURABLE_MERGE_FAILURE_STATUSES docstring promises exactly this
        of BOTH submit paths."""
        from orchestrator.merge_types import MergeOutcome

        q = _FakeEscalationQueue()
        await self._fire(
            tmp_path, MergeOutcome(status=cast(Any, 'brand_new_status')),
            escalation_queue=q,
        )

        assert len(q.submitted) == 1
        assert q.submitted[0].category == 'stranded_merge_failed'

    @pytest.mark.parametrize('status', ['done', 'already_merged', 'superseded'])
    async def test_success_or_transient_files_nothing(
        self, tmp_path: Path, status: str,
    ):
        from orchestrator.merge_types import MergeOutcome

        q = _FakeEscalationQueue()
        await self._fire(
            tmp_path, MergeOutcome(status=cast(Any, status), merge_sha=_LANDED),
            escalation_queue=q,
        )

        assert q.submitted == []

    async def test_second_durable_failure_is_deduped(self, tmp_path: Path):
        from orchestrator.merge_types import MergeOutcome

        q = _FakeEscalationQueue()
        f = _make(tmp_path=tmp_path)
        f.wf.escalation_queue = q  # type: ignore[assignment]
        for _ in range(2):
            await f.wf._on_architect_merge_done(
                MergeOutcome(status='error'), tip=_TIP, evidence='ev',
            )

        assert len(q.submitted) == 1

    async def test_missing_queue_does_not_raise(self, tmp_path: Path):
        """Fail-safe: a done-callback must never propagate a bookkeeping error.
        The structured failure event is still emitted."""
        from orchestrator.merge_types import MergeOutcome

        f = await self._fire(tmp_path, MergeOutcome(status='error'))

        rows = f.event_store.fetch_events_by_type_all_runs(
            EventType.architect_desync_merge, task_id=_TASK_ID,
        )
        assert (rows[0].get('data') or {})['decision'] == 'merge_failed'

    async def test_submit_failure_is_swallowed(self, tmp_path: Path):
        from orchestrator.merge_types import MergeOutcome

        q = _FakeEscalationQueue()
        q.submit = MagicMock(side_effect=RuntimeError('disk full'))  # type: ignore[method-assign]
        await self._fire(
            tmp_path, MergeOutcome(status='error'), escalation_queue=q,
        )  # must not raise

    _fire = staticmethod(_fire_merge_done)


@pytest.mark.asyncio
class TestCancelledMergeFutureIsNotAFailure:
    """A CANCELLED merge future is abandonment/supersession, not a failure.

    Both callers that cancel one of these futures are benign — ``merge_cancel``
    (a deliberate operator action) and the registry's stale-reap
    ``release(detach_waiters=True)`` (which immediately re-acquires the slot for
    a fresh request on the same branch).  The queue's own classification agrees:
    ``enqueue_merge_request._on_finalized`` maps a cancelled future to state
    ``'abandoned'``, distinct from ``'error'``.  Routing cancellation into the
    durable branch would file a critical human-routed L2 for both.
    """

    async def _cancel(self, tmp_path: Path) -> _Fixture:
        """Drive the REAL registered callback with a cancelled future."""
        q = _FakeEscalationQueue()
        f = _make(tmp_path=tmp_path)
        f.wf.escalation_queue = q  # type: ignore[assignment]
        f.artifacts.write_ready_to_merge(commit=_TIP, evidence='ev')
        await f.wf._handle_ready_to_merge_report()
        req = f.merge_queue.get_nowait()

        req.result.cancel()
        await asyncio.sleep(0)
        if f.wf._background_tasks:
            await asyncio.gather(*list(f.wf._background_tasks))
        f.escalations = q
        return f

    async def test_cancellation_files_no_escalation(self, tmp_path: Path):
        f = await self._cancel(tmp_path)

        assert f.escalations.submitted == [], (
            'a cancelled (abandoned/superseded) merge paged a human'
        )

    async def test_cancellation_does_not_mark_done(self, tmp_path: Path):
        """Never a silent success either — the merge did not land."""
        f = await self._cancel(tmp_path)

        f.scheduler.mark_done.assert_not_awaited()

    async def test_cancellation_is_recorded_structurally(self, tmp_path: Path):
        """Not silent: the abandonment lands on the event as its own decision,
        distinct from 'merge_failed' (INV-2 structured-facts-at-failure)."""
        f = await self._cancel(tmp_path)

        rows = f.event_store.fetch_events_by_type_all_runs(
            EventType.architect_desync_merge, task_id=_TASK_ID,
        )
        decisions = [(r.get('data') or {}).get('decision') for r in rows]
        assert 'merge_abandoned' in decisions
        assert 'merge_failed' not in decisions
        abandoned = next(
            (r.get('data') or {}) for r in rows
            if (r.get('data') or {}).get('decision') == 'merge_abandoned'
        )
        assert abandoned['tip'] == _TIP

    async def test_raising_future_is_still_a_durable_failure(
        self, tmp_path: Path,
    ):
        """Only CANCELLATION is reclassified — an exception still pages."""
        q = _FakeEscalationQueue()
        f = _make(tmp_path=tmp_path)
        f.wf.escalation_queue = q  # type: ignore[assignment]
        f.artifacts.write_ready_to_merge(commit=_TIP, evidence='ev')
        await f.wf._handle_ready_to_merge_report()
        req = f.merge_queue.get_nowait()

        req.result.set_exception(RuntimeError('merge worker died'))
        await asyncio.sleep(0)
        await asyncio.gather(*list(f.wf._background_tasks))

        assert len(q.submitted) == 1
        assert q.submitted[0].category == 'stranded_merge_failed'
        assert q.submitted[0].level == 2


@pytest.mark.asyncio
class TestHandlerNeverAdvancesMain:
    """C-3 boundary: the handler enqueues and nothing more."""

    async def test_handler_calls_no_land_main_primitive(self, tmp_path: Path):
        f = _make(tmp_path=tmp_path)
        f.artifacts.write_ready_to_merge(commit=_TIP, evidence='ev')

        await f.wf._handle_ready_to_merge_report()

        for primitive in ('advance_main', 'merge_to_main', 'push_main'):
            assert not getattr(f.git_ops, primitive).called, primitive
        # A queued request whose future is still pending is the whole point:
        # main moves only if the merge worker's own re-verify says so.
        assert f.merge_queue.qsize() == 1
        assert not f.merge_queue.get_nowait().result.done()

    async def test_registered_callback_routes_to_on_architect_merge_done(
        self, tmp_path: Path,
    ):
        """The submitted request's future is wired to the terminal callback —
        without this the FAST done flip would never fire."""
        from orchestrator.merge_types import MergeOutcome

        f = _make(tmp_path=tmp_path)
        f.artifacts.write_ready_to_merge(commit=_TIP, evidence='ev')
        await f.wf._handle_ready_to_merge_report()
        req = f.merge_queue.get_nowait()

        req.result.set_result(MergeOutcome(status='done', merge_sha=_LANDED))
        # The sync callback schedules the async body; drain it.
        await asyncio.sleep(0)
        await asyncio.gather(*list(f.wf._background_tasks))

        f.scheduler.mark_done.assert_awaited_once()
        assert f.scheduler.mark_done.await_args.kwargs['sha'] == _LANDED


# ---------------------------------------------------------------------------
# Dispatch wiring — both _plan sites route the artifact to the handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReadyToMergeDispatch:
    """The exit chain at both sites is unactionable_task → false_premise →
    already_done → ready_to_merge → blocking_dependency.

    ready_to_merge sits AFTER already_done (a clean DONE is more decisive than
    a merge submit) and BEFORE blocking_dependency (a deterministic merge
    submit beats a re-looping dependency report).
    """

    async def _drive(
        self, tmp_path: Path, *, site: str, with_already_done: bool,
    ) -> tuple[AsyncMock, AsyncMock, WorkflowOutcome]:
        """Drive the REAL dispatch site with exit artifacts a stub agent wrote.

        Both handlers are stubbed, so what is observed is purely which branch
        of the live exit chain fired — not either handler's behaviour (covered
        exhaustively above).
        """
        from shared.cli_invoke import AgentResult

        f = _make(tmp_path=tmp_path)
        worktree = f.wf.worktree
        assert worktree is not None

        already_done = AsyncMock(return_value=WorkflowOutcome.DONE)
        ready_to_merge = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        f.wf._handle_already_done_report = already_done  # type: ignore[method-assign]
        f.wf._handle_ready_to_merge_report = ready_to_merge  # type: ignore[method-assign]

        def _invoke_writes_the_exits(*args, **kwargs):
            if with_already_done:
                f.artifacts.write_already_done(commit=_MAIN, evidence='on main')
            f.artifacts.write_ready_to_merge(commit=_TIP, evidence='ev')
            return AgentResult(success=True, output='', cost_usd=0.5, turns=3)

        f.wf._invoke = AsyncMock(side_effect=_invoke_writes_the_exits)  # type: ignore[method-assign]

        if site == 'architect':
            f.wf.briefing.build_architect_prompt = AsyncMock(return_value='P')
            # No plan.json / plan.lock / _old_plan_base ⇒ revalidation skipped.
            (worktree / 'plan.json').unlink(missing_ok=True)
            (worktree / 'plan.lock').unlink(missing_ok=True)
            if hasattr(f.wf, '_old_plan_base'):
                del f.wf._old_plan_base
            outcome = await f.wf._plan()
        else:
            f.wf.briefing.build_simple_task_prompt = AsyncMock(return_value='P')
            outcome = await f.wf._run_simple_task()
        return already_done, ready_to_merge, outcome

    @pytest.mark.parametrize('site', ['architect', 'simple'])
    async def test_already_done_wins_when_both_artifacts_are_written(
        self, tmp_path: Path, site: str,
    ):
        """ready_to_merge sits AFTER already_done, so a co-present clean DONE
        wins — landing on main is more decisive than submitting a merge for a
        branch, and honouring ready_to_merge first would queue a merge for work
        that is already there."""
        already_done, ready_to_merge, outcome = await self._drive(
            tmp_path, site=site, with_already_done=True,
        )

        already_done.assert_awaited_once()
        ready_to_merge.assert_not_called()
        assert outcome == WorkflowOutcome.DONE

    @pytest.mark.parametrize('site', ['architect', 'simple'])
    async def test_ready_to_merge_alone_routes_to_its_handler(
        self, tmp_path: Path, site: str,
    ):
        """The other branch of the same ordering: with no already_done to lose
        to, ready_to_merge is honoured at BOTH sites — the SIMPLE_TASK escape
        hatch reuses the same artifacts, so its chain must stay in lockstep."""
        already_done, ready_to_merge, outcome = await self._drive(
            tmp_path, site=site, with_already_done=False,
        )

        ready_to_merge.assert_awaited_once()
        already_done.assert_not_called()
        assert outcome == WorkflowOutcome.BLOCKED

    async def test_run_does_not_enter_execute_or_double_enqueue(
        self, tmp_path: Path,
    ):
        """``run()`` honours the handler's terminal BLOCKED — no EXECUTE, and
        no SECOND merge request on top of the one the handler enqueued."""
        f = _make(tmp_path=tmp_path)
        f.wf._plan = AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[method-assign]
        f.wf._execute_iterations = AsyncMock(  # type: ignore[method-assign]
            side_effect=AssertionError(
                'EXECUTE must not run when _plan returns BLOCKED',
            ),
        )
        f.wf._recover_if_already_merged = AsyncMock(return_value=None)  # type: ignore[method-assign]
        f.wf._check_branch_on_main = AsyncMock(return_value=None)  # type: ignore[method-assign]

        outcome = (await f.wf.run()).outcome

        assert outcome == WorkflowOutcome.BLOCKED
        f.wf._execute_iterations.assert_not_called()
        assert f.merge_queue.empty()
