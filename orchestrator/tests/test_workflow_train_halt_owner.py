"""Regression tests for task 1599: train GroupMergeRequest WIP halt has no owner.

Tests that:
  1. _escalate_train_halt(result, train_id) builds a per-status L1 escalation,
     registers the halt owner via _submit_halt_owning_escalation, marks the tip
     blocked (skip_escalation=True), and returns WorkflowOutcome.BLOCKED.
  2. The consumer _maybe_enqueue_group_merge gates on the live orphan-halt probe
     (merge_worker is not None and merge_worker.is_wip_halted and
     merge_worker.halt_owner_esc_id is None) and routes halt outcomes through
     _escalate_train_halt instead of the plain _mark_blocked(escalate_to_human=True)
     fall-through.

See also: test_workflow_halt_owner.py (single-task path cancel-cleanup tests).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _workflow_helpers import FakeBriefing, FakeMcp, FakeScheduler
from escalation.queue import EscalationQueue

from orchestrator.merge_queue import MergeOutcome
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


# ---------------------------------------------------------------------------
# _FakeMergeWorker — mirrors test_workflow_halt_owner._FakeMergeWorker
# ---------------------------------------------------------------------------

class _FakeMergeWorker:
    """Minimal halt-owner state machine — same contract as MergeWorker.

    Mirrors test_workflow_halt_owner._FakeMergeWorker exactly so the two test
    files remain independently runnable without shared infrastructure.
    """

    def __init__(self) -> None:
        self._halted = False
        self._owner: str | None = None
        self.last_unhalt_reason: str | None = None

    @property
    def is_wip_halted(self) -> bool:
        return self._halted

    @property
    def halt_owner_esc_id(self) -> str | None:
        return self._owner

    def halt_for_wip(self, reason: str) -> None:
        self._halted = True
        self._owner = None

    def set_halt_owner(self, esc_id: str) -> None:
        assert self._owner is None, (
            f'halt owner already set to {self._owner!r}, '
            f'refusing to overwrite with {esc_id!r}'
        )
        self._owner = esc_id

    def is_halt_owner(self, esc_id: str) -> bool:
        return self._owner is not None and self._owner == esc_id

    def unhalt_wip(self, reason: str | None = None) -> None:
        self.last_unhalt_reason = reason
        self._halted = False
        self._owner = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_worker() -> _FakeMergeWorker:
    return _FakeMergeWorker()


@pytest.fixture
def workflow(
    tmp_path: Path,
    fake_worker: _FakeMergeWorker,
    mock_orch_config: MagicMock,
) -> TaskWorkflow:
    """Minimal TaskWorkflow wired for train halt-owner tests.

    Uses a real EscalationQueue (rooted at tmp_path) so submit/make_id/
    get_by_task work end-to-end.  FakeScheduler captures set_task_status
    calls so we can assert the tip was marked 'blocked'.
    """
    queue = EscalationQueue(tmp_path / 'escalations')
    assignment = TaskAssignment(
        task_id='1599-test',
        task={
            'id': '1599-test',
            'title': 'Train tip for halt-owner test',
            'description': 'Train wip_overlap halt owner registration',
            'status': 'merge-deferred',
            'metadata': {},
            'dependencies': [],
        },
        modules=[],
    )
    git_ops = MagicMock()
    escalation_event = asyncio.Event()
    return TaskWorkflow(
        assignment=assignment,
        config=mock_orch_config,
        git_ops=git_ops,
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        briefing=FakeBriefing(),    # type: ignore[arg-type]
        mcp=FakeMcp(),              # type: ignore[arg-type]
        escalation_queue=queue,
        escalation_event=escalation_event,
        merge_worker=fake_worker,
    )


@pytest.fixture
def workflow_no_esc(
    tmp_path: Path,
    fake_worker: _FakeMergeWorker,
    mock_orch_config: MagicMock,
) -> TaskWorkflow:
    """Same as workflow but with escalation_queue=None (config-absent deployment)."""
    assignment = TaskAssignment(
        task_id='1599-noesc',
        task={
            'id': '1599-noesc',
            'title': 'Train tip — no escalation queue',
            'description': 'Degrade to plain blocked when esc_queue=None',
            'status': 'merge-deferred',
            'metadata': {},
            'dependencies': [],
        },
        modules=[],
    )
    git_ops = MagicMock()
    return TaskWorkflow(
        assignment=assignment,
        config=mock_orch_config,
        git_ops=git_ops,
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        briefing=FakeBriefing(),    # type: ignore[arg-type]
        mcp=FakeMcp(),              # type: ignore[arg-type]
        escalation_queue=None,
        merge_worker=fake_worker,
    )


# ---------------------------------------------------------------------------
# Section A: _escalate_train_halt — wip_halted (overlap files)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_train_halt_wip_halted_registers_owner(
    workflow: TaskWorkflow,
    fake_worker: _FakeMergeWorker,
) -> None:
    """_escalate_train_halt with wip_halted registers halt owner and returns BLOCKED.

    Precondition: queue is halted (halt_for_wip) with no owner.
    After calling _escalate_train_halt:
      (a) exactly one level-1 Escalation was submitted with category='wip_conflict'
      (b) fake_worker.halt_owner_esc_id == that esc.id
      (c) fake_worker.is_halt_owner(esc.id) is True — harness._on_escalation_resolved
          will call unhalt_wip when this escalation is resolved
      (d) the return value is WorkflowOutcome.BLOCKED
      (e) the tip status was set to 'blocked' by _mark_blocked(skip_escalation=True)
      (f) detail text cites the overlap_files

    Fails today: _escalate_train_halt does not exist (AttributeError).
    """
    fake_worker.halt_for_wip('wip_overlap')
    assert fake_worker.is_wip_halted
    assert fake_worker.halt_owner_esc_id is None

    result_outcome = MergeOutcome(status='wip_halted', overlap_files=['a.py', 'b/c.py'])

    outcome = await workflow._escalate_train_halt(result_outcome, 'T-train-1')  # type: ignore[attr-defined]

    # (a) Exactly one L1 submitted with wip_conflict category.
    assert workflow.escalation_queue is not None
    submitted = workflow.escalation_queue.get_by_task('1599-test')
    assert len(submitted) == 1, (
        f'Expected exactly 1 escalation submitted, got {len(submitted)}: {submitted!r}'
    )
    esc = submitted[0]
    assert esc.level == 1, f'Expected level=1, got {esc.level}'
    assert esc.category == 'wip_conflict', f'Expected wip_conflict, got {esc.category!r}'

    # (b) Halt owner registered.
    assert fake_worker.halt_owner_esc_id == esc.id, (
        f'Expected halt_owner_esc_id={esc.id!r}, got {fake_worker.halt_owner_esc_id!r}'
    )

    # (c) is_halt_owner returns True — the harness unhalt path fires on resolution.
    assert fake_worker.is_halt_owner(esc.id), (
        'is_halt_owner must return True so harness._on_escalation_resolved unhalts'
    )

    # (d) Returns BLOCKED.
    assert outcome == WorkflowOutcome.BLOCKED, (
        f'Expected WorkflowOutcome.BLOCKED, got {outcome!r}'
    )

    # (e) Tip status set to blocked.
    assert isinstance(workflow.scheduler, FakeScheduler)
    history = workflow.scheduler.statuses.get('1599-test', [])
    assert 'blocked' in history, (
        f'Expected "blocked" in scheduler status history, got {history!r}'
    )

    # (f) Detail mentions the overlap files.
    assert 'a.py' in (esc.detail or ''), (
        f'Expected overlap file "a.py" in esc.detail: {esc.detail!r}'
    )


# ---------------------------------------------------------------------------
# Section B: _escalate_train_halt — parametrize remaining three outcomes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize('outcome_kwargs,expected_category,detail_hint', [
    # done_wip_recovery: merge landed on main but stash pop conflicted → wip_conflict
    (
        {'status': 'done_wip_recovery', 'recovery_branch': 'wip/recovery-1', 'merge_sha': 'dead1234'},
        'wip_conflict',
        'wip/recovery-1',
    ),
    # wip_recovery_no_advance: CAS-failure path, no advance, stash pop conflict → wip_conflict
    (
        {'status': 'wip_recovery_no_advance', 'recovery_branch': 'wip/no-advance'},
        'wip_conflict',
        'wip/no-advance',
    ),
    # unmerged_state: project_root had pre-existing UU/AA/DD markers → unmerged_state
    (
        {'status': 'unmerged_state'},
        'unmerged_state',
        'UU',  # any word that signals UU/AA/DD guidance
    ),
], ids=['done_wip_recovery', 'wip_recovery_no_advance', 'unmerged_state'])
async def test_train_halt_other_outcomes_register_owner(
    outcome_kwargs: dict,
    expected_category: str,
    detail_hint: str,
    tmp_path: Path,
    mock_orch_config: MagicMock,
) -> None:
    """All four halt-inducing statuses register owner + return BLOCKED.

    Each creates a fresh workflow + worker so parametrize runs don't share state.

    Assertions per outcome:
      (a) halt owner registered (halt_owner_esc_id != None)
      (b) is_halt_owner(esc.id) is True
      (c) return is WorkflowOutcome.BLOCKED
      (d) escalation category matches expected_category
      (e) detail cites the outcome-specific context (recovery_branch / UU guidance)
    """
    worker = _FakeMergeWorker()
    worker.halt_for_wip('test-halt')

    queue = EscalationQueue(tmp_path / 'esc')
    assignment = TaskAssignment(
        task_id='1599-param',
        task={
            'id': '1599-param',
            'title': 'Parametrized halt test',
            'description': 'Covers all four _map_advance_failure halt statuses',
            'status': 'merge-deferred',
            'metadata': {},
            'dependencies': [],
        },
        modules=[],
    )
    wf = TaskWorkflow(
        assignment=assignment,
        config=mock_orch_config,
        git_ops=MagicMock(),
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        briefing=FakeBriefing(),    # type: ignore[arg-type]
        mcp=FakeMcp(),              # type: ignore[arg-type]
        escalation_queue=queue,
        merge_worker=worker,
    )

    result_outcome = MergeOutcome(**outcome_kwargs)
    returned = await wf._escalate_train_halt(result_outcome, 'T-param')  # type: ignore[attr-defined]

    # (a) Halt owner registered.
    assert worker.halt_owner_esc_id is not None, (
        f'halt_owner_esc_id must be set after _escalate_train_halt '
        f'(outcome={outcome_kwargs["status"]!r})'
    )

    submitted = queue.get_by_task('1599-param')
    assert len(submitted) == 1, (
        f'Expected 1 escalation for status={outcome_kwargs["status"]!r}, got {len(submitted)}'
    )
    esc = submitted[0]

    # (b) is_halt_owner returns True.
    assert worker.is_halt_owner(esc.id), (
        f'is_halt_owner must be True for status={outcome_kwargs["status"]!r}'
    )

    # (c) Returns BLOCKED.
    assert returned == WorkflowOutcome.BLOCKED, (
        f'Expected BLOCKED for status={outcome_kwargs["status"]!r}, got {returned!r}'
    )

    # (d) Correct escalation category.
    assert esc.category == expected_category, (
        f'Expected category={expected_category!r} for status={outcome_kwargs["status"]!r}, '
        f'got {esc.category!r}'
    )

    # (e) Detail contains context hint (recovery_branch or UU guidance).
    assert detail_hint in (esc.detail or ''), (
        f'Expected {detail_hint!r} in detail for status={outcome_kwargs["status"]!r}: '
        f'{esc.detail!r}'
    )


# ---------------------------------------------------------------------------
# Section C: escalation_queue=None degrades gracefully
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_train_halt_no_escalation_queue_degrades_to_blocked(
    workflow_no_esc: TaskWorkflow,
    fake_worker: _FakeMergeWorker,
) -> None:
    """When escalation_queue=None, _escalate_train_halt degrades to plain BLOCKED.

    No exception is raised. halt_owner_esc_id stays None (nothing to own).
    Returns WorkflowOutcome.BLOCKED.

    This is the config-absent deployment path (dead in normal production).
    """
    fake_worker.halt_for_wip('wip_overlap')
    assert fake_worker.is_wip_halted
    assert fake_worker.halt_owner_esc_id is None
    assert workflow_no_esc.escalation_queue is None

    result_outcome = MergeOutcome(status='wip_halted', overlap_files=['x.py'])
    outcome = await workflow_no_esc._escalate_train_halt(result_outcome, 'T-noesc')  # type: ignore[attr-defined]

    # No owner — nothing to register without an escalation package.
    assert fake_worker.halt_owner_esc_id is None, (
        'halt_owner_esc_id must stay None when escalation_queue=None'
    )

    # Returns BLOCKED (not an exception).
    assert outcome == WorkflowOutcome.BLOCKED, (
        f'Expected WorkflowOutcome.BLOCKED even without escalation queue, got {outcome!r}'
    )

    # Tip status still set to 'blocked'.
    assert isinstance(workflow_no_esc.scheduler, FakeScheduler)
    history = workflow_no_esc.scheduler.statuses.get('1599-noesc', [])
    assert 'blocked' in history, (
        f'Expected "blocked" in scheduler history even without esc_queue: {history!r}'
    )


# ---------------------------------------------------------------------------
# Section D: consumer-routing tests for _maybe_enqueue_group_merge
#
# (Written in step-3; extended here for step-3's tests.)
# ---------------------------------------------------------------------------

# ── Helper: fixture that wires a halted merge_worker ────────────────────────

def _make_consumer_fixture(
    *,
    task_id: str = '103',
    metadata: dict | None = None,
    tasks_by_train_return: list[dict] | None = None,
    worker_halted: bool = False,
    worker_owner: str | None = None,
    tmp_path: Path,
    mock_orch_config: MagicMock,
):
    """Build a TaskWorkflow suitable for testing _maybe_enqueue_group_merge.

    Extends the _make() pattern from test_workflow_train_completion.py with:
      - A wired _FakeMergeWorker (optionally pre-halted)
      - A real EscalationQueue (so _escalate_train_halt end-to-end registers owner)
    """
    from _orch_helpers import pydantic_spec
    from orchestrator.config import OrchestratorConfig

    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {
        'id': task_id, 'title': 'T', 'description': 'd',
        'metadata': metadata or {},
    }
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = Path('/tmp/non-existent-for-test')
    config.max_consecutive_infra_resumes = 3
    config.max_consecutive_merge_thrash = 3

    scheduler = MagicMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.set_task_status = AsyncMock()
    scheduler.get_status = AsyncMock(return_value='merge-deferred')
    scheduler.get_task = AsyncMock(return_value={'id': task_id, 'metadata': metadata or {}})
    scheduler.mark_done = AsyncMock()
    scheduler.clear_requeue_count = MagicMock()
    scheduler.get_tasks = AsyncMock(return_value=[])
    scheduler.get_statuses = AsyncMock(
        return_value=(
            {'101': 'merge-deferred', '102': 'merge-deferred', '103': 'merge-deferred'},
            None,
        )
    )

    if tasks_by_train_return is not None:
        scheduler.tasks_by_train = AsyncMock(return_value=tasks_by_train_return)
    else:
        scheduler.tasks_by_train = AsyncMock(return_value=[])

    git_ops = MagicMock()
    git_ops.config.branch_prefix = 'task/'
    git_ops.config.main_branch = 'main'

    queue = EscalationQueue(tmp_path / f'esc-{task_id}')
    merge_queue: asyncio.Queue = asyncio.Queue()

    worker = _FakeMergeWorker()
    if worker_halted:
        worker.halt_for_wip('test-halt')
        # If an owner was pre-set (non-None), bypass the assertion
        if worker_owner is not None:
            worker._owner = worker_owner  # type: ignore[attr-defined]

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        escalation_queue=queue,
        merge_queue=merge_queue,
        merge_worker=worker,
    )

    wf.artifacts = MagicMock()
    wf.artifacts.read_iteration_log = MagicMock(return_value=([], []))
    wf.worktree = Path(f'/tmp/wt-{task_id}')
    wf.event_store = None

    return wf, scheduler, queue, merge_queue, worker


@pytest.mark.asyncio
async def test_consumer_routes_halt_outcome_to_escalate_and_own(
    tmp_path: Path,
    mock_orch_config: MagicMock,
) -> None:
    """Consumer routes halt outcome through _escalate_train_halt when probe fires.

    Precondition:
      - merge_worker is wired and pre-halted (is_wip_halted=True, owner=None)
      - _await_cancellable returns MergeOutcome('done_wip_recovery', ...)

    The orphan-halt probe fires (merge_worker is not None AND is_wip_halted AND
    halt_owner_esc_id is None) → consumer calls _escalate_train_halt → owner
    registered (halt_owner_esc_id is not None), result is BLOCKED.

    Fails today: the consumer has no orphan-halt branch so the owner stays None.
    """
    members = [
        {'id': '101', 'status': 'merge-deferred',
         'metadata': {'train': {'id': 'T1', 'order': 0}}},
        {'id': '102', 'status': 'merge-deferred',
         'metadata': {'train': {'id': 'T1', 'order': 1}}},
        {'id': '103', 'status': 'merge-deferred',
         'metadata': {'train': {'id': 'T1', 'order': 2}}},
    ]
    wf, scheduler, queue, merge_queue, worker = _make_consumer_fixture(
        task_id='103',
        metadata={'train': {'id': 'T1', 'order': 2}},
        tasks_by_train_return=members,
        worker_halted=True,
        tmp_path=tmp_path,
        mock_orch_config=mock_orch_config,
    )

    halt_outcome = MergeOutcome(
        status='done_wip_recovery',
        recovery_branch='wip/recovery-t1',
        merge_sha='abcd1234',
    )
    wf._await_cancellable = AsyncMock(return_value=halt_outcome)  # type: ignore[method-assign]

    result = await wf._maybe_enqueue_group_merge()

    # Probe fired → owner registered.
    assert worker.halt_owner_esc_id is not None, (
        'halt_owner_esc_id must be set after consumer routes through _escalate_train_halt '
        '— the orphan-halt probe must detect is_wip_halted=True, owner=None and call '
        '_escalate_train_halt instead of the plain _mark_blocked fall-through'
    )

    # Result is BLOCKED.
    assert result == WorkflowOutcome.BLOCKED, (
        f'Expected WorkflowOutcome.BLOCKED from consumer, got {result!r}'
    )


@pytest.mark.asyncio
async def test_consumer_non_halt_blocked_preserves_existing_path(
    tmp_path: Path,
    mock_orch_config: MagicMock,
) -> None:
    """Non-halt 'blocked' outcome skips the new branch — old path preserved.

    Precondition: worker NOT halted (is_wip_halted=False).
    Outcome: MergeOutcome('blocked', reason='rebase conflict').

    Orphan-halt probe is False (not halted) → consumer falls through to plain
    _mark_blocked(escalate_to_human=True).  halt_owner_esc_id stays None.
    """
    members = [
        {'id': '101', 'status': 'merge-deferred',
         'metadata': {'train': {'id': 'T2', 'order': 0}}},
        {'id': '102', 'status': 'merge-deferred',
         'metadata': {'train': {'id': 'T2', 'order': 1}}},
        {'id': '103', 'status': 'merge-deferred',
         'metadata': {'train': {'id': 'T2', 'order': 2}}},
    ]
    wf, scheduler, queue, merge_queue, worker = _make_consumer_fixture(
        task_id='103',
        metadata={'train': {'id': 'T2', 'order': 2}},
        tasks_by_train_return=members,
        worker_halted=False,  # NOT halted
        tmp_path=tmp_path,
        mock_orch_config=mock_orch_config,
    )

    # Mock _mark_blocked so we can assert it's called with escalate_to_human=True.
    mock_mb = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mock_mb  # type: ignore[method-assign]

    non_halt_outcome = MergeOutcome(
        status='blocked',
        reason='Train merge rejected: tip branch rebase conflict on main',
    )
    wf._await_cancellable = AsyncMock(return_value=non_halt_outcome)  # type: ignore[method-assign]

    result = await wf._maybe_enqueue_group_merge()

    # Owner stays None — not a halt outcome.
    assert worker.halt_owner_esc_id is None, (
        'halt_owner_esc_id must stay None for a non-halt blocked outcome'
    )

    # Plain _mark_blocked(escalate_to_human=True) was called.
    mock_mb.assert_awaited_once()
    call_args = mock_mb.call_args
    all_args_str = str(call_args)
    assert 'escalate_to_human=True' in all_args_str or (
        call_args.kwargs.get('escalate_to_human') is True
    ), f'Expected escalate_to_human=True in _mark_blocked call: {call_args!r}'

    assert result == WorkflowOutcome.BLOCKED


@pytest.mark.asyncio
async def test_consumer_no_merge_worker_preserves_existing_path(
    tmp_path: Path,
    mock_orch_config: MagicMock,
) -> None:
    """merge_worker=None skips the new branch — regression guard for existing test.

    Mirrors test_workflow_train_completion.test_blocked_outcome_escalates_to_human
    which wires merge_queue but NOT merge_worker.  merge_worker is None → probe
    is False → plain _mark_blocked(escalate_to_human=True) fall-through.
    """
    from _orch_helpers import pydantic_spec
    from orchestrator.config import OrchestratorConfig

    members = [
        {'id': '101', 'status': 'merge-deferred',
         'metadata': {'train': {'id': 'T3', 'order': 0}}},
        {'id': '102', 'status': 'merge-deferred',
         'metadata': {'train': {'id': 'T3', 'order': 1}}},
        {'id': '103', 'status': 'merge-deferred',
         'metadata': {'train': {'id': 'T3', 'order': 2}}},
    ]

    assignment = MagicMock()
    assignment.task_id = '103'
    assignment.task = {
        'id': '103', 'title': 'T', 'description': 'd',
        'metadata': {'train': {'id': 'T3', 'order': 2}},
    }
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = Path('/tmp/non-existent')
    config.max_consecutive_infra_resumes = 3
    config.max_consecutive_merge_thrash = 3

    scheduler = MagicMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.set_task_status = AsyncMock()
    scheduler.get_status = AsyncMock(return_value='merge-deferred')
    scheduler.get_task = AsyncMock(return_value={'id': '103', 'metadata': {'train': {'id': 'T3', 'order': 2}}})
    scheduler.mark_done = AsyncMock()
    scheduler.clear_requeue_count = MagicMock()
    scheduler.get_tasks = AsyncMock(return_value=[])
    scheduler.get_statuses = AsyncMock(
        return_value=({'101': 'merge-deferred', '102': 'merge-deferred', '103': 'merge-deferred'}, None)
    )
    scheduler.tasks_by_train = AsyncMock(return_value=members)

    esc_queue = MagicMock()
    esc_queue.has_open_l1 = MagicMock(return_value=False)
    esc_queue.make_id = MagicMock(return_value='esc-103-1')
    esc_queue.submit = MagicMock()
    esc_queue.get_by_task = MagicMock(return_value=[])

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=MagicMock(),
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        escalation_queue=esc_queue,  # type: ignore[arg-type]
        merge_queue=asyncio.Queue(),
        # merge_worker intentionally omitted → None
    )
    wf.artifacts = MagicMock()
    wf.artifacts.read_iteration_log = MagicMock(return_value=([], []))
    wf.worktree = Path('/tmp/wt-103-noworker')
    wf.event_store = None

    mock_mb = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mock_mb  # type: ignore[method-assign]

    failed_outcome = MergeOutcome(
        'blocked', reason='Train merge rejected: tip branch rebase conflict on main',
    )
    wf._await_cancellable = AsyncMock(return_value=failed_outcome)  # type: ignore[method-assign]

    result = await wf._maybe_enqueue_group_merge()

    # merge_worker is None → probe is False → plain _mark_blocked.
    mock_mb.assert_awaited_once()
    call_args = mock_mb.call_args
    all_args_str = str(call_args)
    assert 'escalate_to_human=True' in all_args_str or (
        call_args.kwargs.get('escalate_to_human') is True
    ), f'Expected escalate_to_human=True: {call_args!r}'

    assert result == WorkflowOutcome.BLOCKED
