"""Tests for TaskWorkflow._build_train_state helper and its wiring into
_ensure_l1_escalation_for_blocked (PRD § 9.8 park-prefix derail context).

_build_train_state is an async helper that returns:
  - None for non-train tasks or malformed metadata
  - {'id', 'order', 'parked_members', 'failing_member'} for valid train members

parked_members = sibling task_ids (excluding self) at status 'merge-deferred',
discovered via the metadata.train.members cache (fast path) or a get_tasks()
scan filtered by train.id (fallback when members is absent).

get_statuses returns a (dict, error) TUPLE — mocks must match this shape.
get_tasks returns a list[dict] — mocks return a plain list.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec
from escalation.models import Escalation

from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

# ---------------------------------------------------------------------------
# Shared fixture helper
# ---------------------------------------------------------------------------

@dataclass
class _Fixture:
    wf: TaskWorkflow
    scheduler: MagicMock
    mark_blocked: AsyncMock
    queue: MagicMock


def _make(
    *,
    task_id: str = '103',
    metadata: dict | None = None,
    get_statuses_return: tuple[dict[str, str], Exception | None] | None = None,
    get_tasks_return: list[dict] | None = None,
) -> _Fixture:
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
    scheduler.get_status = AsyncMock(return_value='blocked')
    scheduler.get_task = AsyncMock(return_value={'id': task_id, 'metadata': metadata or {}})

    if get_statuses_return is not None:
        scheduler.get_statuses = AsyncMock(return_value=get_statuses_return)
    else:
        scheduler.get_statuses = AsyncMock(return_value=({}, None))

    if get_tasks_return is not None:
        scheduler.get_tasks = AsyncMock(return_value=get_tasks_return)
    else:
        scheduler.get_tasks = AsyncMock(return_value=[])

    git_ops = MagicMock()
    git_ops.config.branch_prefix = 'task/'
    git_ops.config.main_branch = 'main'

    queue = MagicMock()
    queue.has_open_l1 = MagicMock(return_value=False)
    queue.make_id = MagicMock(return_value=f'esc-{task_id}-1')
    queue.submit = MagicMock()
    queue.get_by_task = MagicMock(return_value=[])

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        escalation_queue=queue,  # type: ignore[arg-type]
    )

    wf.artifacts = MagicMock()
    wf.artifacts.read_iteration_log = MagicMock(return_value=([], []))

    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    return _Fixture(wf=wf, scheduler=scheduler, mark_blocked=mark_blocked, queue=queue)


# ---------------------------------------------------------------------------
# _build_train_state tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_returns_none_for_non_train_task():
    """No metadata.train → _build_train_state returns None."""
    f = _make(metadata={})
    result = await f.wf._build_train_state()
    assert result is None


@pytest.mark.asyncio
async def test_returns_none_for_malformed_train_not_dict():
    """metadata.train present but not a dict → return None."""
    f = _make(metadata={'train': 'not-a-dict'})
    result = await f.wf._build_train_state()
    assert result is None


@pytest.mark.asyncio
async def test_returns_none_for_malformed_train_missing_id():
    """metadata.train is a dict but missing 'id' → return None."""
    f = _make(metadata={'train': {'order': 1}})  # no 'id'
    result = await f.wf._build_train_state()
    assert result is None


@pytest.mark.asyncio
async def test_returns_none_for_malformed_train_missing_order():
    """metadata.train is a dict but missing 'order' → return None."""
    f = _make(metadata={'train': {'id': 'T1'}})  # no 'order'
    result = await f.wf._build_train_state()
    assert result is None


@pytest.mark.asyncio
async def test_uses_members_cache_when_present():
    """Fast path: members list in metadata → get_statuses called, get_tasks NOT called.

    metadata.train = {'id': 'T1', 'order': 2, 'members': ['101', '102', '103']}
    self.task_id = '103'
    statuses: '101' → 'merge-deferred', '102' → 'merge-deferred', '103' → 'blocked'

    Expected result:
      {'id': 'T1', 'order': 2, 'parked_members': ['101', '102'], 'failing_member': '103'}

    NOTE: get_statuses returns a (dict, error) TUPLE.
    """
    f = _make(
        task_id='103',
        metadata={'train': {'id': 'T1', 'order': 2, 'members': ['101', '102', '103']}},
        get_statuses_return=(
            {'101': 'merge-deferred', '102': 'merge-deferred', '103': 'blocked'},
            None,
        ),
    )

    result = await f.wf._build_train_state()

    assert result == {
        'id': 'T1',
        'order': 2,
        'parked_members': ['101', '102'],
        'failing_member': '103',
    }
    # Fast path: get_tasks must NOT have been called
    f.scheduler.get_tasks.assert_not_awaited()
    # get_statuses must have been called (with the members list)
    f.scheduler.get_statuses.assert_awaited_once()


@pytest.mark.asyncio
async def test_falls_back_to_get_tasks_scan_when_members_absent():
    """Fallback path: no members in metadata → scan get_tasks() for same train.id.

    metadata.train = {'id': 'T1', 'order': 2}  (no members)

    get_tasks() returns tasks including siblings with matching train.id.  Each
    task dict carries a 'status' field; the fallback path reads status directly
    from those dicts — it must NOT issue a second get_statuses() round-trip.

    Expected result: parked_members contains only the merge-deferred siblings
    excluding self ('103').
    """
    sibling_tasks = [
        {'id': '101', 'status': 'merge-deferred', 'metadata': {'train': {'id': 'T1', 'order': 0}}},
        {'id': '102', 'status': 'done', 'metadata': {'train': {'id': 'T1', 'order': 1}}},
        {'id': '103', 'status': 'blocked', 'metadata': {'train': {'id': 'T1', 'order': 2}}},  # self
        {'id': '200', 'status': 'in-progress', 'metadata': {'train': {'id': 'T2', 'order': 0}}},  # different train
        {'id': '201', 'status': 'in-progress', 'metadata': {}},  # no train at all
    ]
    f = _make(
        task_id='103',
        metadata={'train': {'id': 'T1', 'order': 2}},  # no 'members'
        get_tasks_return=sibling_tasks,
        # No get_statuses_return — fallback reads status directly from task dicts
    )

    result = await f.wf._build_train_state()

    assert result is not None
    assert result['id'] == 'T1'
    assert result['order'] == 2
    assert result['failing_member'] == '103'
    # Only '101' is merge-deferred (not self, not '102' which is 'done')
    assert result['parked_members'] == ['101']
    # Fallback path: get_tasks must have been called
    f.scheduler.get_tasks.assert_awaited_once()
    # Fallback path: status comes from task dicts — get_statuses must NOT be called
    f.scheduler.get_statuses.assert_not_awaited()


@pytest.mark.asyncio
async def test_returns_train_state_for_order_zero():
    """order=0 is a valid train order; the `is None` check must not treat it as absent.

    Regression guard: a truthiness check (``if not train_order``) would silently
    return None for order=0, dropping train_state for the first member of a train.
    This test pins the ``is None`` semantics so a future refactor cannot regress it.
    """
    f = _make(
        task_id='103',
        metadata={'train': {'id': 'T1', 'order': 0, 'members': ['101', '102', '103']}},
        get_statuses_return=(
            {'101': 'merge-deferred', '102': 'merge-deferred', '103': 'blocked'},
            None,
        ),
    )

    result = await f.wf._build_train_state()

    assert result is not None, '_build_train_state must not return None for order=0'
    assert result['order'] == 0


# ---------------------------------------------------------------------------
# _ensure_l1_escalation_for_blocked wiring tests (step-9)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_l1_escalation_carries_train_state_for_train_member():
    """For a train member, the submitted Escalation must carry train_state.

    Drives _ensure_l1_escalation_for_blocked and inspects the submitted
    Escalation object's train_state field.
    """
    f = _make(
        task_id='103',
        metadata={'train': {'id': 'T1', 'order': 2, 'members': ['101', '102', '103']}},
        get_statuses_return=(
            {'101': 'merge-deferred', '102': 'merge-deferred', '103': 'blocked'},
            None,
        ),
    )

    await f.wf._ensure_l1_escalation_for_blocked('blocked reason', 'detail text')

    f.queue.submit.assert_called_once()
    submitted: Escalation = f.queue.submit.call_args.args[0]
    assert submitted.train_state == {
        'id': 'T1',
        'order': 2,
        'parked_members': ['101', '102'],
        'failing_member': '103',
    }


@pytest.mark.asyncio
async def test_l1_escalation_train_state_none_for_non_train():
    """For a non-train task, the submitted Escalation must have train_state=None."""
    f = _make(
        task_id='50',
        metadata={},  # no train
    )

    await f.wf._ensure_l1_escalation_for_blocked('blocked reason', 'detail text')

    f.queue.submit.assert_called_once()
    submitted: Escalation = f.queue.submit.call_args.args[0]
    assert submitted.train_state is None
