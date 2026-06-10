"""Tests for task 1707 δ workflow wiring — train attribution in _maybe_enqueue_group_merge.

Step-7  (RED): detection wiring — TRAIN_VERIFY_FAILED_REASON_PREFIX routes to
               _attribute_train_failure; other blocked reasons still route to
               _mark_blocked.
Step-9  (RED): all-pass → escalate the TRAIN (interaction); land nothing.
Step-11 (RED): some-fail → land passers, block offender.
Step-13 (RED): edge cases — tip-as-offender, un-stack conflict, advance failure.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.merge_queue import MergeOutcome, SoloVerifyResult, TRAIN_VERIFY_FAILED_REASON_PREFIX
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


# ---------------------------------------------------------------------------
# Shared fixture helper (mirrors test_workflow_train_completion._make)
# ---------------------------------------------------------------------------


@dataclass
class _Fixture:
    wf: TaskWorkflow
    scheduler: MagicMock
    git_ops: MagicMock
    mark_blocked: AsyncMock
    esc_queue: MagicMock
    merge_queue: asyncio.Queue


def _make(
    *,
    task_id: str = '103',
    metadata: dict | None = None,
    tasks_by_train_return: list[dict] | None = None,
    get_statuses_return: tuple[dict[str, str], Exception | None] | None = None,
) -> _Fixture:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {
        'id': task_id, 'title': 'T', 'description': 'd',
        'metadata': metadata or {},
        'status': 'merge-deferred',
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

    if tasks_by_train_return is not None:
        scheduler.tasks_by_train = AsyncMock(return_value=tasks_by_train_return)
    else:
        scheduler.tasks_by_train = AsyncMock(return_value=[])

    if get_statuses_return is not None:
        scheduler.get_statuses = AsyncMock(return_value=get_statuses_return)
    else:
        scheduler.get_statuses = AsyncMock(
            return_value=(
                {'101': 'merge-deferred', '102': 'merge-deferred', '103': 'merge-deferred'},
                None,
            )
        )

    git_ops = MagicMock()
    git_ops.config.branch_prefix = 'task/'
    git_ops.config.main_branch = 'main'
    git_ops.advance_main = AsyncMock(return_value='advanced')
    git_ops.cleanup_merge_worktree = AsyncMock()

    esc_queue = MagicMock()
    esc_queue.has_open_l1 = MagicMock(return_value=False)
    esc_queue.make_id = MagicMock(return_value=f'esc-{task_id}-1')
    esc_queue.submit = MagicMock()
    esc_queue.get_by_task = MagicMock(return_value=[])

    merge_queue: asyncio.Queue = asyncio.Queue()

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        escalation_queue=esc_queue,  # type: ignore[arg-type]
        merge_queue=merge_queue,
    )

    wf.artifacts = MagicMock()
    wf.artifacts.read_iteration_log = MagicMock(return_value=([], []))
    wf.worktree = Path(f'/tmp/wt-{task_id}')
    wf.event_store = None

    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    return _Fixture(
        wf=wf,
        scheduler=scheduler,
        git_ops=git_ops,
        mark_blocked=mark_blocked,
        esc_queue=esc_queue,
        merge_queue=merge_queue,
    )


def _train_members(
    train_id: str = 'T-attr', tip_id: str = '103',
) -> list[dict]:
    """Return a 3-member ordered member list suitable for tasks_by_train."""
    return [
        {'id': '101', 'status': 'merge-deferred',
         'metadata': {'train': {'id': train_id, 'order': 0, 'members': ['101', '102', tip_id]}}},
        {'id': '102', 'status': 'merge-deferred',
         'metadata': {'train': {'id': train_id, 'order': 1, 'members': ['101', '102', tip_id]}}},
        {'id': tip_id, 'status': 'merge-deferred',
         'metadata': {'train': {'id': train_id, 'order': 2, 'members': ['101', '102', tip_id]}}},
    ]


# ---------------------------------------------------------------------------
# Step-7: Detection wiring — TRAIN_VERIFY_FAILED_REASON_PREFIX branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDetectionWiring:
    """TRAIN_VERIFY_FAILED_REASON_PREFIX routes to _attribute_train_failure."""

    async def test_tagged_outcome_calls_attribute_train_failure(self) -> None:
        """Tagged blocked outcome → _attribute_train_failure is called; _mark_blocked is NOT."""
        from orchestrator.merge_queue import TRAIN_VERIFY_FAILED_REASON_PREFIX

        members = _train_members(train_id='T-attr', tip_id='103')
        f = _make(
            task_id='103',
            metadata={'train': {'id': 'T-attr', 'order': 2, 'members': ['101', '102', '103']}},
            tasks_by_train_return=members,
        )

        tagged_outcome = MergeOutcome(
            'blocked',
            reason=f'{TRAIN_VERIFY_FAILED_REASON_PREFIX}: 3 tests failed',
            failure_category='cargo_test',
        )
        f.wf._await_cancellable = AsyncMock(return_value=tagged_outcome)  # type: ignore[method-assign]

        attr_mock = AsyncMock(return_value=WorkflowOutcome.DONE)
        f.wf._attribute_train_failure = attr_mock  # type: ignore[method-assign]

        result = await f.wf._maybe_enqueue_group_merge()

        # _attribute_train_failure called once with the result, train_id, and members
        attr_mock.assert_awaited_once()
        call_args = attr_mock.call_args
        assert call_args[0][0] is tagged_outcome, (
            f'first arg should be the result, got {call_args[0][0]!r}'
        )
        assert call_args[0][1] == 'T-attr', (
            f'second arg should be train_id="T-attr", got {call_args[0][1]!r}'
        )
        # third arg: member list (ordered root→tip)
        passed_members = call_args[0][2]
        assert len(passed_members) == 3

        # return value from _attribute_train_failure is propagated
        assert result == WorkflowOutcome.DONE

        # _mark_blocked must NOT be called
        f.mark_blocked.assert_not_awaited()

    async def test_untagged_blocked_routes_to_mark_blocked(self) -> None:
        """Non-tagged blocked outcome still routes to _mark_blocked (unaffected path)."""
        from orchestrator.merge_queue import TRAIN_VERIFY_FAILED_REASON_PREFIX

        members = _train_members(train_id='T-attr2', tip_id='103')
        f = _make(
            task_id='103',
            metadata={'train': {'id': 'T-attr2', 'order': 2, 'members': ['101', '102', '103']}},
            tasks_by_train_return=members,
        )

        other_outcome = MergeOutcome(
            'blocked',
            reason='Train merge advance failed: cas_failed',
        )
        f.wf._await_cancellable = AsyncMock(return_value=other_outcome)  # type: ignore[method-assign]

        attr_mock = AsyncMock(return_value=WorkflowOutcome.DONE)
        f.wf._attribute_train_failure = attr_mock  # type: ignore[method-assign]

        # Add a fake merge_worker with is_wip_halted=False so the orphan-halt
        # probe doesn't fire (we want to confirm _mark_blocked is the fallback)
        wip_worker = MagicMock()
        wip_worker.is_wip_halted = False
        wip_worker.halt_owner_esc_id = None
        f.wf.merge_worker = wip_worker  # type: ignore[attr-defined]

        await f.wf._maybe_enqueue_group_merge()

        # _attribute_train_failure must NOT be called
        attr_mock.assert_not_awaited()
        # _mark_blocked IS called
        f.mark_blocked.assert_awaited_once()

    async def test_train_incomplete_path_unaffected(self) -> None:
        """train_incomplete blocked outcome still returns None (park) — unaffected."""
        from orchestrator.merge_queue import (
            TRAIN_INCOMPLETE_REASON_PREFIX,
            TRAIN_VERIFY_FAILED_REASON_PREFIX,
        )

        members = _train_members(train_id='T-incomplete', tip_id='103')
        f = _make(
            task_id='103',
            metadata={'train': {'id': 'T-incomplete', 'order': 2, 'members': ['101', '102', '103']}},
            tasks_by_train_return=members,
        )

        incomplete_outcome = MergeOutcome(
            'blocked',
            reason=f'{TRAIN_INCOMPLETE_REASON_PREFIX}: member 101 is in-progress',
        )
        f.wf._await_cancellable = AsyncMock(return_value=incomplete_outcome)  # type: ignore[method-assign]

        attr_mock = AsyncMock(return_value=WorkflowOutcome.DONE)
        f.wf._attribute_train_failure = attr_mock  # type: ignore[method-assign]

        result = await f.wf._maybe_enqueue_group_merge()

        assert result is None, f'train_incomplete should return None (park), got {result!r}'
        attr_mock.assert_not_awaited()
        f.mark_blocked.assert_not_awaited()


# ---------------------------------------------------------------------------
# Step-9: All-pass → escalate the TRAIN as an interaction; land nothing.
# ---------------------------------------------------------------------------


def _make_tagged_result(failure_category: str = 'cargo_test') -> MergeOutcome:
    return MergeOutcome(
        'blocked',
        reason=f'{TRAIN_VERIFY_FAILED_REASON_PREFIX}: 3 tests failed',
        failure_category=failure_category,
    )


def _solo_pass(member_id: str) -> SoloVerifyResult:
    return SoloVerifyResult(
        member_id=member_id,
        passed=True,
        merge_sha=f'sha-{member_id}-solo',
        reason='',
    )


@pytest.mark.asyncio
class TestAllPassInteraction:
    """All members pass solo → genuine interaction; escalate the TRAIN, land nothing."""

    async def test_all_pass_calls_reverify_exactly_n_times(self) -> None:
        """_reverify_one_member is called exactly N=3 times (≤N+1 bound)."""
        members = _train_members(train_id='T-all-pass', tip_id='103')
        f = _make(
            task_id='103',
            metadata={'train': {'id': 'T-all-pass', 'order': 2, 'members': ['101', '102', '103']}},
        )
        f.wf.event_store = MagicMock()

        reverify_mock = AsyncMock(side_effect=[
            _solo_pass('101'),
            _solo_pass('102'),
            _solo_pass('103'),
        ])
        f.wf._reverify_one_member = reverify_mock  # type: ignore[method-assign]

        tagged = _make_tagged_result()
        await f.wf._attribute_train_failure(tagged, 'T-all-pass', members)

        assert reverify_mock.await_count == 3, (
            f'Expected _reverify_one_member called 3 times, got {reverify_mock.await_count}'
        )

    async def test_all_pass_does_not_land_any_member(self) -> None:
        """When all pass, advance_main and scheduler.mark_done are never called."""
        members = _train_members(train_id='T-all-pass2', tip_id='103')
        f = _make(
            task_id='103',
            metadata={'train': {'id': 'T-all-pass2', 'order': 2, 'members': ['101', '102', '103']}},
        )
        f.wf.event_store = MagicMock()

        f.wf._reverify_one_member = AsyncMock(side_effect=[  # type: ignore[method-assign]
            _solo_pass('101'), _solo_pass('102'), _solo_pass('103'),
        ])

        tagged = _make_tagged_result()
        await f.wf._attribute_train_failure(tagged, 'T-all-pass2', members)

        f.git_ops.advance_main.assert_not_awaited()
        f.scheduler.mark_done.assert_not_awaited()

    async def test_all_pass_emits_train_derailed_interaction(self) -> None:
        """train_derailed event emitted with data.verdict='interaction'."""
        members = _train_members(train_id='T-interaction', tip_id='103')
        f = _make(
            task_id='103',
            metadata={'train': {'id': 'T-interaction', 'order': 2, 'members': ['101', '102', '103']}},
        )
        event_store = MagicMock()
        f.wf.event_store = event_store

        f.wf._reverify_one_member = AsyncMock(side_effect=[  # type: ignore[method-assign]
            _solo_pass('101'), _solo_pass('102'), _solo_pass('103'),
        ])

        tagged = _make_tagged_result()
        await f.wf._attribute_train_failure(tagged, 'T-interaction', members)

        # Find the train_derailed emit call
        derailed_calls = [
            c for c in event_store.emit.call_args_list
            if c.args and c.args[0] == EventType.train_derailed
        ]
        assert len(derailed_calls) >= 1, (
            f'Expected at least one train_derailed event, got: {event_store.emit.call_args_list}'
        )
        # Check verdict='interaction' in the data payload
        derailed_call = derailed_calls[0]
        data_kwarg = derailed_call.kwargs.get('data') or {}
        assert data_kwarg.get('verdict') == 'interaction', (
            f'Expected verdict="interaction" in data, got: {data_kwarg!r}'
        )

    async def test_all_pass_escalates_train_not_single_member(self) -> None:
        """All-pass: _mark_blocked is called with a reason naming the train (not a single member)."""
        members = _train_members(train_id='T-train-esc', tip_id='103')
        f = _make(
            task_id='103',
            metadata={'train': {'id': 'T-train-esc', 'order': 2, 'members': ['101', '102', '103']}},
        )
        f.wf.event_store = MagicMock()

        f.wf._reverify_one_member = AsyncMock(side_effect=[  # type: ignore[method-assign]
            _solo_pass('101'), _solo_pass('102'), _solo_pass('103'),
        ])

        tagged = _make_tagged_result()
        await f.wf._attribute_train_failure(tagged, 'T-train-esc', members)

        # _mark_blocked should be called (stub or real) with escalate_to_human=True
        f.mark_blocked.assert_awaited_once()
        call_kwargs = f.mark_blocked.call_args.kwargs
        assert call_kwargs.get('escalate_to_human') is True
        # The reason should mention 'interaction' (not a single member id)
        reason_arg = f.mark_blocked.call_args.args[0] if f.mark_blocked.call_args.args else ''
        assert 'interaction' in reason_arg.lower() or 'T-train-esc' in reason_arg, (
            f'Expected reason to mention train or interaction, got: {reason_arg!r}'
        )

    async def test_all_pass_returns_blocked(self) -> None:
        """All-pass: tip workflow returns WorkflowOutcome.BLOCKED."""
        members = _train_members(train_id='T-blocked', tip_id='103')
        f = _make(
            task_id='103',
            metadata={'train': {'id': 'T-blocked', 'order': 2, 'members': ['101', '102', '103']}},
        )
        f.wf.event_store = MagicMock()

        f.wf._reverify_one_member = AsyncMock(side_effect=[  # type: ignore[method-assign]
            _solo_pass('101'), _solo_pass('102'), _solo_pass('103'),
        ])

        tagged = _make_tagged_result()
        result = await f.wf._attribute_train_failure(tagged, 'T-blocked', members)

        assert result == WorkflowOutcome.BLOCKED, (
            f'Expected WorkflowOutcome.BLOCKED, got {result!r}'
        )


# ---------------------------------------------------------------------------
# Step-11: Some-fail → land passers, block offender, bounded.
# ---------------------------------------------------------------------------


def _solo_fail(member_id: str, reason: str = 'test failed') -> SoloVerifyResult:
    return SoloVerifyResult(
        member_id=member_id,
        passed=False,
        merge_sha=None,
        reason=reason,
    )


def _solo_pass_wt(member_id: str) -> SoloVerifyResult:
    """SoloVerifyResult for a passer with solo_wt/solo_branch populated."""
    from pathlib import Path as _Path  # noqa: PLC0415
    r = SoloVerifyResult(
        member_id=member_id,
        passed=True,
        merge_sha=f'sha-{member_id}-solo',
        reason='',
    )
    # Step-12 will add these fields to SoloVerifyResult; pre-populate via
    # monkey-patch so the RED tests are consistent with the expected impl shape.
    object.__setattr__(r, 'solo_wt', _Path(f'/tmp/solo-{member_id}'))
    object.__setattr__(r, 'solo_branch', f'_solo-{member_id}')
    return r


@pytest.mark.asyncio
class TestSomeFailAttribution:
    """Non-tip member fails solo → land passers, block offender; tip passer returns DONE."""

    def _three_member_fixture(self, offender_id: str = '101') -> '_Fixture':
        """3-member train; offender_id is the failer, others pass."""
        members = _train_members(train_id='T-some-fail', tip_id='103')
        f = _make(
            task_id='103',
            metadata={'train': {'id': 'T-some-fail', 'order': 2,
                                'members': ['101', '102', '103']}},
        )
        f.wf.event_store = MagicMock()
        f.wf.git_ops = f.git_ops  # alias so tests can use f.git_ops
        # get_main_sha needed by advance_main expected_main path
        f.git_ops.get_main_sha = AsyncMock(return_value='main-sha-probe')
        return f

    async def test_passers_are_landed_advance_main(self) -> None:
        """advance_main called once per passer with the passer's solo merge_sha."""
        f = self._three_member_fixture()
        members = _train_members(train_id='T-some-fail', tip_id='103')
        # Order 1 ('101') fails; '102' and '103' (tip) pass.
        f.wf._reverify_one_member = AsyncMock(side_effect=[  # type: ignore[method-assign]
            _solo_fail('101'),
            _solo_pass_wt('102'),
            _solo_pass_wt('103'),
        ])

        tagged = _make_tagged_result()
        await f.wf._attribute_train_failure(tagged, 'T-some-fail', members)

        # advance_main should have been called for each passer (2 calls)
        assert f.git_ops.advance_main.await_count == 2, (
            f'Expected advance_main called 2 times, got {f.git_ops.advance_main.await_count}'
        )
        # Verify the solo merge_shas were used
        advanced_shas = {c.args[0] for c in f.git_ops.advance_main.await_args_list}
        assert 'sha-102-solo' in advanced_shas
        assert 'sha-103-solo' in advanced_shas

    async def test_passers_are_marked_done(self) -> None:
        """scheduler.mark_done called for each passer that advanced successfully."""
        f = self._three_member_fixture()
        members = _train_members(train_id='T-some-fail2', tip_id='103')
        f.wf._reverify_one_member = AsyncMock(side_effect=[  # type: ignore[method-assign]
            _solo_fail('101'),
            _solo_pass_wt('102'),
            _solo_pass_wt('103'),
        ])
        # Patch event_store separately (not the mark_blocked one)
        f.wf.event_store = MagicMock()

        tagged = _make_tagged_result()
        await f.wf._attribute_train_failure(tagged, 'T-some-fail2', members)

        # scheduler.mark_done called for '102' and '103'
        called_ids = {c.args[0] for c in f.scheduler.mark_done.await_args_list}
        assert '102' in called_ids, f'Expected mark_done for 102, calls: {f.scheduler.mark_done.await_args_list}'
        assert '103' in called_ids, f'Expected mark_done for 103, calls: {f.scheduler.mark_done.await_args_list}'
        # NOT called for the offender
        assert '101' not in called_ids

    async def test_offender_is_blocked(self) -> None:
        """Offender's status set to 'blocked' via scheduler.set_task_status."""
        f = self._three_member_fixture()
        members = _train_members(train_id='T-offender', tip_id='103')
        f.wf._reverify_one_member = AsyncMock(side_effect=[  # type: ignore[method-assign]
            _solo_fail('101'),
            _solo_pass_wt('102'),
            _solo_pass_wt('103'),
        ])
        f.wf.event_store = MagicMock()

        tagged = _make_tagged_result()
        await f.wf._attribute_train_failure(tagged, 'T-offender', members)

        # scheduler.set_task_status('101', 'blocked') must be called
        set_status_calls = [
            (c.args[0], c.args[1]) if c.args else (c.kwargs.get('task_id'), c.kwargs.get('status'))
            for c in f.scheduler.set_task_status.await_args_list
        ]
        assert ('101', 'blocked') in set_status_calls, (
            f'Expected set_task_status("101", "blocked"), calls: {set_status_calls}'
        )

    async def test_offender_l1_submitted(self) -> None:
        """An L1 is submitted for the offender's task_id."""
        f = self._three_member_fixture()
        members = _train_members(train_id='T-l1-submit', tip_id='103')
        f.wf._reverify_one_member = AsyncMock(side_effect=[  # type: ignore[method-assign]
            _solo_fail('101'),
            _solo_pass_wt('102'),
            _solo_pass_wt('103'),
        ])
        f.wf.event_store = MagicMock()

        tagged = _make_tagged_result()
        await f.wf._attribute_train_failure(tagged, 'T-l1-submit', members)

        # esc_queue.submit should be called at least once for the offender
        assert f.esc_queue.submit.called, 'Expected esc_queue.submit to be called for offender'

    async def test_train_derailed_attributed_event(self) -> None:
        """train_derailed emitted with verdict='attributed', offenders=['101']."""
        f = self._three_member_fixture()
        members = _train_members(train_id='T-attributed', tip_id='103')
        f.wf._reverify_one_member = AsyncMock(side_effect=[  # type: ignore[method-assign]
            _solo_fail('101'),
            _solo_pass_wt('102'),
            _solo_pass_wt('103'),
        ])
        event_store = MagicMock()
        f.wf.event_store = event_store

        tagged = _make_tagged_result()
        await f.wf._attribute_train_failure(tagged, 'T-attributed', members)

        derailed_calls = [
            c for c in event_store.emit.call_args_list
            if c.args and c.args[0] == EventType.train_derailed
        ]
        assert len(derailed_calls) >= 1, (
            f'Expected train_derailed event, got: {event_store.emit.call_args_list}'
        )
        data_kwarg = derailed_calls[0].kwargs.get('data') or {}
        assert data_kwarg.get('verdict') == 'attributed', (
            f'Expected verdict="attributed", got: {data_kwarg!r}'
        )
        offenders = data_kwarg.get('offenders') or []
        assert '101' in offenders, f'Expected offenders to include "101", got: {offenders!r}'

    async def test_tip_passer_returns_done(self) -> None:
        """When tip ('103') passes solo and lands, returns WorkflowOutcome.DONE."""
        f = self._three_member_fixture()
        members = _train_members(train_id='T-tip-done', tip_id='103')
        f.wf._reverify_one_member = AsyncMock(side_effect=[  # type: ignore[method-assign]
            _solo_fail('101'),
            _solo_pass_wt('102'),
            _solo_pass_wt('103'),
        ])
        f.wf.event_store = MagicMock()

        tagged = _make_tagged_result()
        result = await f.wf._attribute_train_failure(tagged, 'T-tip-done', members)

        assert result == WorkflowOutcome.DONE, (
            f'Expected WorkflowOutcome.DONE (tip passed), got {result!r}'
        )

    async def test_exactly_n_solo_verifies(self) -> None:
        """_reverify_one_member called exactly N=3 times (≤N+1 bound)."""
        f = self._three_member_fixture()
        members = _train_members(train_id='T-bound', tip_id='103')
        reverify_mock = AsyncMock(side_effect=[
            _solo_fail('101'),
            _solo_pass_wt('102'),
            _solo_pass_wt('103'),
        ])
        f.wf._reverify_one_member = reverify_mock  # type: ignore[method-assign]
        f.wf.event_store = MagicMock()

        tagged = _make_tagged_result()
        await f.wf._attribute_train_failure(tagged, 'T-bound', members)

        assert reverify_mock.await_count == 3, (
            f'Expected exactly 3 _reverify_one_member calls, got {reverify_mock.await_count}'
        )
