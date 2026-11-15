"""Leaf δ (task 3076): ``LaneState.IN_USE`` is actually written.

Before δ, IN_USE was a declared state with no writer. Now a lane reads IN_USE
while dark-factory has a live child rooted in it and steps back to ASSIGNED
when the last one exits. Which children count, and why it is a span at
dark-factory's own spawn points rather than a /proc scan, is decided in
``plans/warm-lane-infra-repatriation-prd.md`` §11 q2.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _lane_test_helpers import (
    acquire_lane,
    add_warm_lane_scripts,
    build_harness,
    get_head,
    make_orch_config,
    make_plan,
    make_warm_lane_repo,
    pure_fs_git_ops,
    wire_recovery_scheduler,
)
from _orch_helpers import wait_responsive
from _workflow_helpers import fixed_prompt
from escalation.queue import EscalationQueue
from shared.config_dir import TaskConfigDir

# Reused read-only: the merge-phase preamble stubs, and (as
# test_task_verify_lease.py::_build_task_workflow does) the factory that builds
# a TaskWorkflow with every collaborator mocked.
from test_task_verify_lease import _StopMergePhase, _stub_merge_phase_preamble
from test_workflow_verify_retry import _make_workflow

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import SIMPLE_TASK
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.lane_lifecycle import LaneLifecycle, LaneState
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.workflow import TaskWorkflow

LANE_NAME = '_lane-0'


def _record_path(worktree_base: Path, lane_name: str = LANE_NAME) -> Path:
    return worktree_base / '.lane-state' / f'{lane_name}.json'


def _on_disk(worktree_base: Path, lane_name: str = LANE_NAME) -> dict:
    return json.loads(_record_path(worktree_base, lane_name).read_text())


skip_as_root = pytest.mark.skipif(
    os.geteuid() == 0, reason='root ignores permission bits',
)


@contextlib.contextmanager
def _faulted(path: Path, mode: int) -> Iterator[None]:
    """A real filesystem fault: *path* carries *mode* for the block, then its
    original mode again however the block exits."""
    original = path.stat().st_mode & 0o777
    path.chmod(mode)
    try:
        yield
    finally:
        path.chmod(original)


def _warned_about(caplog, lane_name: str) -> bool:
    return any(
        r.levelno == logging.WARNING and lane_name in r.getMessage()
        for r in caplog.records
    )


def _legal_ladder(task_id: str) -> list[tuple[LaneState, dict[str, str]]]:
    return [
        (LaneState.SEED, {'seeded_from_sha': 'abc123'}),
        (LaneState.REGISTERED, {'branch': f'task/{task_id}'}),
        (LaneState.ASSIGNED, {'task_id': task_id, 'title': f'demo-{task_id}'}),
        (LaneState.IN_USE, {}),
        (LaneState.RELEASED, {}),
    ]


def _drive_to(
    lifecycle: LaneLifecycle, lane: Path, target: LaneState, task_id: str = '42',
) -> None:
    """Bring *lane*'s record to *target* through public, legal transitions only."""
    if target is LaneState.QUARANTINED:
        _drive_to(lifecycle, lane, LaneState.ASSIGNED, task_id)
        lifecycle.transition(lane, LaneState.QUARANTINED)
        return
    for state, fields in _legal_ladder(task_id):
        lifecycle.transition(lane, state, **fields)
        if state is target:
            return


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'escalations')


@pytest.fixture
def lifecycle(tmp_path: Path, queue: EscalationQueue) -> LaneLifecycle:
    """Wired to a real queue: a benign race that reached transition()'s
    illegal-edge path would page a human, and must show up here."""
    return LaneLifecycle(worktree_base=tmp_path, escalation_queue=queue)


@pytest.fixture
def lane(tmp_path: Path) -> Path:
    return tmp_path / LANE_NAME


class TestNoteInUseAndIdle:
    def test_assigned_lane_flips_to_in_use_and_back_keeping_the_assignment(
        self, tmp_path: Path, lifecycle: LaneLifecycle, lane: Path,
    ):
        _drive_to(lifecycle, lane, LaneState.ASSIGNED)
        assigned = _on_disk(tmp_path)

        assert lifecycle.note_in_use(lane) is True
        in_use = _on_disk(tmp_path)
        assert in_use['state'] == 'in_use'
        assert (in_use['task_id'], in_use['title'], in_use['branch']) == (
            assigned['task_id'], assigned['title'], assigned['branch'],
        ) == ('42', 'demo-42', 'task/42')

        assert lifecycle.note_idle(lane) is True
        idle = _on_disk(tmp_path)
        assert idle['state'] == 'assigned'
        assert idle['task_id'] == '42'

    @pytest.mark.parametrize(
        'state',
        [LaneState.RELEASED, LaneState.REGISTERED, LaneState.SEED, LaneState.QUARANTINED],
    )
    def test_note_in_use_writes_nothing_on_a_lane_with_no_live_assignment(
        self, tmp_path: Path, lifecycle: LaneLifecycle, queue: EscalationQueue,
        lane: Path, state: LaneState,
    ):
        _drive_to(lifecycle, lane, state)
        before = _record_path(tmp_path).read_bytes()

        assert lifecycle.note_in_use(lane) is False

        assert _record_path(tmp_path).read_bytes() == before
        assert queue.get_pending() == []

    def test_note_in_use_on_an_absent_record_creates_nothing(
        self, tmp_path: Path, lifecycle: LaneLifecycle, queue: EscalationQueue,
        lane: Path,
    ):
        assert lifecycle.note_in_use(lane) is False

        assert not _record_path(tmp_path).exists()
        assert queue.get_pending() == []

    def test_note_in_use_on_a_corrupt_record_writes_and_escalates_nothing(
        self, tmp_path: Path, lifecycle: LaneLifecycle, queue: EscalationQueue,
        lane: Path,
    ):
        record_path = _record_path(tmp_path)
        record_path.parent.mkdir(parents=True)
        record_path.write_text('{not valid json')
        before = record_path.read_bytes()

        assert lifecycle.note_in_use(lane) is False

        assert record_path.read_bytes() == before
        assert queue.get_pending() == []

    @pytest.mark.parametrize(
        'state',
        [
            LaneState.ASSIGNED, LaneState.RELEASED, LaneState.REGISTERED,
            LaneState.SEED, LaneState.QUARANTINED,
        ],
    )
    def test_note_idle_writes_nothing_unless_the_lane_reads_in_use(
        self, tmp_path: Path, lifecycle: LaneLifecycle, queue: EscalationQueue,
        lane: Path, state: LaneState,
    ):
        _drive_to(lifecycle, lane, state)
        before = _record_path(tmp_path).read_bytes()

        assert lifecycle.note_idle(lane) is False

        assert _record_path(tmp_path).read_bytes() == before
        assert queue.get_pending() == []

    def test_note_idle_on_an_absent_record_creates_nothing(
        self, tmp_path: Path, lifecycle: LaneLifecycle, queue: EscalationQueue,
        lane: Path,
    ):
        assert lifecycle.note_idle(lane) is False

        assert not _record_path(tmp_path).exists()
        assert queue.get_pending() == []

    def test_note_in_use_on_an_in_use_lane_is_true_without_a_write(
        self, tmp_path: Path, lifecycle: LaneLifecycle, lane: Path,
    ):
        _drive_to(lifecycle, lane, LaneState.IN_USE)
        before = _record_path(tmp_path).read_bytes()

        assert lifecycle.note_in_use(lane) is True

        assert _record_path(tmp_path).read_bytes() == before


@skip_as_root
class TestNeverRaises:
    """A liveness marker must never abort the child-spawning caller.

    Each fault is a real one: a read-only ``.lane-state`` dir makes the
    atomic write raise ``PermissionError``, and a mode-0 record file makes
    the read raise it.
    """

    def _assert_swallowed(
        self, tmp_path: Path, caplog, call, fault: tuple[Path, int],
        expected_state: str,
    ) -> None:
        with _faulted(*fault), caplog.at_level(
            logging.WARNING, logger='orchestrator.lane_lifecycle',
        ):
            assert call() is False, 'a liveness marker never aborts its caller'
        assert _on_disk(tmp_path)['state'] == expected_state
        assert _warned_about(caplog, LANE_NAME), (
            'the swallowed failure must be logged naming the lane'
        )

    def test_note_in_use_write_failure_returns_false_and_warns(
        self, tmp_path: Path, lifecycle: LaneLifecycle, lane: Path, caplog,
    ):
        _drive_to(lifecycle, lane, LaneState.ASSIGNED)

        self._assert_swallowed(
            tmp_path, caplog, lambda: lifecycle.note_in_use(lane),
            (lifecycle.state_dir, 0o500), 'assigned',
        )

    def test_note_idle_write_failure_returns_false_and_warns(
        self, tmp_path: Path, lifecycle: LaneLifecycle, lane: Path, caplog,
    ):
        _drive_to(lifecycle, lane, LaneState.IN_USE)

        self._assert_swallowed(
            tmp_path, caplog, lambda: lifecycle.note_idle(lane),
            (lifecycle.state_dir, 0o500), 'in_use',
        )

    def test_note_in_use_on_an_unreadable_record_returns_false_and_warns(
        self, tmp_path: Path, lifecycle: LaneLifecycle, lane: Path, caplog,
    ):
        _drive_to(lifecycle, lane, LaneState.ASSIGNED)

        self._assert_swallowed(
            tmp_path, caplog, lambda: lifecycle.note_in_use(lane),
            (_record_path(tmp_path), 0o000), 'assigned',
        )

    def test_note_idle_on_an_unreadable_record_returns_false_and_warns(
        self, tmp_path: Path, lifecycle: LaneLifecycle, lane: Path, caplog,
    ):
        _drive_to(lifecycle, lane, LaneState.IN_USE)

        self._assert_swallowed(
            tmp_path, caplog, lambda: lifecycle.note_idle(lane),
            (_record_path(tmp_path), 0o000), 'in_use',
        )


class TestResetStaleInUse:
    """A crash mid-span leaves IN_USE fossils; the sweep steps them back."""

    STALE = {'_lane-0': '42', '_lane-1': '43'}
    UNTOUCHED = ('_lane-2', '_lane-3', '_lane-4', '_lane-5')

    @pytest.fixture
    def swept(self, tmp_path: Path, lifecycle: LaneLifecycle) -> LaneLifecycle:
        for lane_name, task_id in self.STALE.items():
            _drive_to(lifecycle, tmp_path / lane_name, LaneState.IN_USE, task_id)
        _drive_to(lifecycle, tmp_path / '_lane-2', LaneState.ASSIGNED, '44')
        _drive_to(lifecycle, tmp_path / '_lane-3', LaneState.RELEASED, '45')
        _drive_to(lifecycle, tmp_path / '_lane-4', LaneState.QUARANTINED, '46')
        _record_path(tmp_path, '_lane-5').write_text('{not valid json')
        return lifecycle

    def _snapshot(self, tmp_path: Path, lane_names) -> dict[str, bytes]:
        return {name: _record_path(tmp_path, name).read_bytes() for name in lane_names}

    def test_resets_exactly_the_in_use_lanes_keeping_their_assignments(
        self, tmp_path: Path, swept: LaneLifecycle, queue: EscalationQueue, caplog,
    ):
        before = {name: _on_disk(tmp_path, name) for name in self.STALE}
        untouched = self._snapshot(tmp_path, self.UNTOUCHED)

        with caplog.at_level(logging.INFO, logger='orchestrator.lane_lifecycle'):
            reset = swept.reset_stale_in_use()

        assert reset == ['_lane-0', '_lane-1']
        for name, task_id in self.STALE.items():
            after = _on_disk(tmp_path, name)
            assert after['state'] == 'assigned'
            assert (after['task_id'], after['title'], after['branch']) == (
                before[name]['task_id'], before[name]['title'], before[name]['branch'],
            ) == (task_id, f'demo-{task_id}', f'task/{task_id}')
            assert [
                r for r in caplog.records
                if r.levelno == logging.INFO
                and name in r.getMessage() and task_id in r.getMessage()
            ], f'expected an INFO line naming the reset of {name} (task {task_id})'
        assert self._snapshot(tmp_path, self.UNTOUCHED) == untouched
        assert queue.get_pending() == []

    def test_a_lane_in_the_skip_set_keeps_its_live_in_use_record(
        self, tmp_path: Path, swept: LaneLifecycle,
    ):
        reset = swept.reset_stale_in_use(skip=['_lane-1'])

        assert reset == ['_lane-0']
        assert _on_disk(tmp_path, '_lane-1')['state'] == 'in_use'

    @skip_as_root
    def test_one_unreadable_record_costs_only_that_lane(
        self, tmp_path: Path, swept: LaneLifecycle, caplog,
    ):
        with _faulted(_record_path(tmp_path, '_lane-1'), 0o000), caplog.at_level(
            logging.WARNING, logger='orchestrator.lane_lifecycle',
        ):
            reset = swept.reset_stale_in_use()

        assert reset == ['_lane-0'], (
            'one unreadable record must not abandon the sweep of every other lane'
        )
        assert _warned_about(caplog, '_lane-1')
        assert _on_disk(tmp_path, '_lane-0')['state'] == 'assigned'
        assert _on_disk(tmp_path, '_lane-1')['state'] == 'in_use'

    def test_a_second_sweep_is_a_no_op(self, tmp_path: Path, swept: LaneLifecycle):
        swept.reset_stale_in_use()
        settled = self._snapshot(tmp_path, [*self.STALE, *self.UNTOUCHED])

        assert swept.reset_stale_in_use() == []

        assert self._snapshot(tmp_path, [*self.STALE, *self.UNTOUCHED]) == settled


async def _assigned_lane(git_ops: GitOps, task_id: str = '42') -> Path:
    """Assign the first free pool lane to *task_id* through the pool, the sole
    ASSIGNED writer, which mirrors it onto the durable record."""
    pool = git_ops.warm_lane_pool
    assert pool is not None
    acquired = await pool.acquire_for(
        task_id, title=f'demo-{task_id}', branch=f'task/{task_id}',
    )
    assert acquired is not None
    lane, _reused = acquired
    lane.mkdir(parents=True, exist_ok=True)
    return lane


def _lane_state(git_ops: GitOps, lane: Path) -> str:
    return _on_disk(git_ops.worktree_base, lane.name)['state']


def _assignment(git_ops: GitOps, lane: Path) -> tuple[str, str, str]:
    record = _on_disk(git_ops.worktree_base, lane.name)
    return record['task_id'], record['title'], record['branch']


@pytest.mark.asyncio
class TestLaneInUseSpan:
    async def test_lane_reads_in_use_inside_the_span_and_assigned_after(
        self, tmp_path: Path,
    ):
        git_ops = pure_fs_git_ops(tmp_path, pool_size=1)
        lane = await _assigned_lane(git_ops)
        assert _lane_state(git_ops, lane) == 'assigned'

        async with git_ops.lane_in_use(lane):
            inside = _lane_state(git_ops, lane)
            assignment_inside = _assignment(git_ops, lane)

        assert inside == 'in_use'
        assert _lane_state(git_ops, lane) == 'assigned'
        assert assignment_inside == _assignment(git_ops, lane) == (
            '42', 'demo-42', 'task/42',
        )

    async def test_a_cold_worktree_is_not_tracked(self, tmp_path: Path):
        git_ops = pure_fs_git_ops(tmp_path, pool_size=1)
        cold = git_ops.worktree_base / '3076'
        cold.mkdir(parents=True)
        ran = False

        async with git_ops.lane_in_use(cold):
            ran = True

        assert ran
        assert not (git_ops.worktree_base / '.lane-state').exists()

    async def test_a_gitops_without_a_pool_tracks_nothing(self, tmp_path: Path):
        git_ops = pure_fs_git_ops(tmp_path)
        assert git_ops.warm_lane_pool is None
        lane = git_ops.worktree_base / LANE_NAME
        lane.mkdir(parents=True)
        ran = False

        async with git_ops.lane_in_use(lane):
            ran = True

        assert ran
        assert not (git_ops.worktree_base / '.lane-state').exists()

    async def test_no_lane_tracks_nothing(self, tmp_path: Path):
        git_ops = pure_fs_git_ops(tmp_path, pool_size=1)
        ran = False

        async with git_ops.lane_in_use(None):
            ran = True

        assert ran
        assert not (git_ops.worktree_base / '.lane-state').exists()

    async def test_a_raising_body_propagates_and_the_lane_returns_to_assigned(
        self, tmp_path: Path,
    ):
        git_ops = pure_fs_git_ops(tmp_path, pool_size=1)
        lane = await _assigned_lane(git_ops)

        with pytest.raises(RuntimeError, match='child failed'):
            async with git_ops.lane_in_use(lane):
                raise RuntimeError('child failed')

        assert _lane_state(git_ops, lane) == 'assigned'

    async def test_a_cancelled_body_returns_the_lane_to_assigned(self, tmp_path: Path):
        git_ops = pure_fs_git_ops(tmp_path, pool_size=1)
        lane = await _assigned_lane(git_ops)
        entered = asyncio.Event()

        async def _hold_forever() -> None:
            async with git_ops.lane_in_use(lane):
                entered.set()
                await asyncio.Event().wait()

        holder = asyncio.create_task(_hold_forever())
        await wait_responsive(entered.wait(), label='holder entered lane_in_use')
        assert _lane_state(git_ops, lane) == 'in_use'
        holder.cancel()
        with pytest.raises(asyncio.CancelledError):
            await holder

        assert _lane_state(git_ops, lane) == 'assigned'

    async def test_a_released_lane_is_never_written(self, tmp_path: Path):
        git_ops = pure_fs_git_ops(tmp_path, pool_size=1)
        lane = await _assigned_lane(git_ops)
        pool = git_ops.warm_lane_pool
        assert pool is not None
        await pool.release(lane)
        record_path = _record_path(git_ops.worktree_base, lane.name)
        released = record_path.read_bytes()
        assert json.loads(released)['state'] == 'released'

        async with git_ops.lane_in_use(lane):
            inside = record_path.read_bytes()

        assert inside == released
        assert record_path.read_bytes() == released

    @skip_as_root
    async def test_an_unreadable_record_neither_aborts_the_span_nor_leaks_it(
        self, tmp_path: Path,
    ):
        git_ops = pure_fs_git_ops(tmp_path, pool_size=1)
        lane = await _assigned_lane(git_ops)
        ran = False

        with _faulted(_record_path(git_ops.worktree_base, lane.name), 0o000):
            async with git_ops.lane_in_use(lane):
                ran = True

        assert ran, 'a liveness marker never aborts the child-spawning caller'
        assert _lane_state(git_ops, lane) == 'assigned'
        async with git_ops.lane_in_use(lane):
            inside = _lane_state(git_ops, lane)
        assert (inside, _lane_state(git_ops, lane)) == ('in_use', 'assigned'), (
            'a faulted span must not leak depth, or no later span on the lane '
            'ever steps it back to assigned'
        )
        assert LaneLifecycle(git_ops.worktree_base).note_in_use(lane) is True
        assert git_ops.reset_stale_lane_in_use_records() == [lane.name], (
            'a faulted span must leave no live-span claim, or the sweep spares '
            'an unhealable IN_USE fossil forever'
        )


@pytest.mark.asyncio
class TestLaneInUseNesting:
    """Several live children in one lane: only the last one out steps it back."""

    async def test_nested_spans_hold_in_use_until_the_outer_one_exits(
        self, tmp_path: Path,
    ):
        git_ops = pure_fs_git_ops(tmp_path, pool_size=1)
        lane = await _assigned_lane(git_ops)

        async with git_ops.lane_in_use(lane):
            outer_stamp = _on_disk(git_ops.worktree_base, lane.name)['updated_at']
            async with git_ops.lane_in_use(lane):
                inner = _on_disk(git_ops.worktree_base, lane.name)
            after_inner = _lane_state(git_ops, lane)

        assert (inner['state'], inner['updated_at']) == ('in_use', outer_stamp), (
            'the inner enter must not write again'
        )
        assert after_inner == 'in_use', 'a child is still live after the inner exit'
        assert _lane_state(git_ops, lane) == 'assigned'

    async def test_a_raise_in_the_inner_span_unwinds_both_to_assigned(
        self, tmp_path: Path,
    ):
        git_ops = pure_fs_git_ops(tmp_path, pool_size=1)
        lane = await _assigned_lane(git_ops)

        with pytest.raises(RuntimeError, match='inner child failed'):
            async with git_ops.lane_in_use(lane), git_ops.lane_in_use(lane):
                raise RuntimeError('inner child failed')

        assert _lane_state(git_ops, lane) == 'assigned'

    async def test_spans_on_different_lanes_are_independent(self, tmp_path: Path):
        git_ops = pure_fs_git_ops(tmp_path, pool_size=2)
        lane_a = await _assigned_lane(git_ops, '42')
        lane_b = await _assigned_lane(git_ops, '43')

        async with git_ops.lane_in_use(lane_a):
            only_a = (_lane_state(git_ops, lane_a), _lane_state(git_ops, lane_b))
            async with git_ops.lane_in_use(lane_b):
                both = (_lane_state(git_ops, lane_a), _lane_state(git_ops, lane_b))
            b_exited = (_lane_state(git_ops, lane_a), _lane_state(git_ops, lane_b))

        assert only_a == ('in_use', 'assigned')
        assert both == ('in_use', 'in_use')
        assert b_exited == ('in_use', 'assigned')
        assert (_lane_state(git_ops, lane_a), _lane_state(git_ops, lane_b)) == (
            'assigned', 'assigned',
        )

    async def test_a_fresh_span_after_a_full_unwind_writes_in_use_again(
        self, tmp_path: Path,
    ):
        git_ops = pure_fs_git_ops(tmp_path, pool_size=1)
        lane = await _assigned_lane(git_ops)
        async with git_ops.lane_in_use(lane), git_ops.lane_in_use(lane):
            pass
        assert _lane_state(git_ops, lane) == 'assigned'

        async with git_ops.lane_in_use(lane):
            again = _lane_state(git_ops, lane)

        assert again == 'in_use'
        assert _lane_state(git_ops, lane) == 'assigned'

    async def test_every_enter_re_marks_a_lane_reassigned_under_a_long_span(
        self, tmp_path: Path,
    ):
        git_ops = pure_fs_git_ops(tmp_path, pool_size=1)
        pool = git_ops.warm_lane_pool
        assert pool is not None
        lane = await _assigned_lane(git_ops, '42')

        async with git_ops.lane_in_use(lane):
            await pool.release(lane)
            assert await _assigned_lane(git_ops, '43') == lane
            reassigned = _on_disk(git_ops.worktree_base, lane.name)
            async with git_ops.lane_in_use(lane):
                second = _lane_state(git_ops, lane)
            after_second = _lane_state(git_ops, lane)

        assert (reassigned['state'], reassigned['task_id']) == ('assigned', '43')
        assert second == 'in_use'
        assert after_second == 'in_use', (
            'the outer span is still live: over-reporting is the safe direction'
        )
        assert _lane_state(git_ops, lane) == 'assigned'
        assert _assignment(git_ops, lane)[0] == '43'

    async def test_the_sweep_spares_a_lane_held_by_a_live_span(self, tmp_path: Path):
        git_ops = pure_fs_git_ops(tmp_path, pool_size=2)
        live = await _assigned_lane(git_ops, '42')
        fossil = await _assigned_lane(git_ops, '43')
        assert LaneLifecycle(git_ops.worktree_base).note_in_use(fossil) is True

        async with git_ops.lane_in_use(live):
            reset = git_ops.reset_stale_lane_in_use_records()
            live_inside = _lane_state(git_ops, live)

        assert reset == [fossil.name]
        assert live_inside == 'in_use'
        assert _lane_state(git_ops, fossil) == 'assigned'


# ---------------------------------------------------------------------------
# The five workflow spawn sites (PRD §11 q2). One spy per structurally
# distinct site, so unwrapping any single site fails exactly one test.
# ---------------------------------------------------------------------------


async def _workflow_in_lane(tmp_path: Path) -> tuple[TaskWorkflow, GitOps, Path]:
    """A TaskWorkflow whose worktree IS a warm lane assigned to its task."""
    git_ops = pure_fs_git_ops(tmp_path, pool_size=1)
    config = OrchestratorConfig(project_root=git_ops.project_root, git=git_ops.config)
    lane = await _assigned_lane(git_ops, '42')
    assignment = TaskAssignment(
        task_id='42',
        task={
            'id': '42', 'title': 'X', 'description': '',
            'status': 'pending', 'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )
    workflow, _artifacts = _make_workflow(config, git_ops, assignment, lane)
    return workflow, git_ops, lane


def _passing_result() -> VerifyResult:
    return VerifyResult(
        passed=True, test_output='', lint_output='', type_output='',
        summary='ok', timed_out=False,
    )


@pytest.mark.asyncio
class TestWorkflowSpawnSitesHoldTheSpan:
    @pytest.mark.usefixtures('code_default_config')
    async def test_agent_invocation(self, tmp_path: Path, monkeypatch):
        workflow, git_ops, lane = await _workflow_in_lane(tmp_path)
        workflow._config_dir = TaskConfigDir('42', base_dir=lane / '.task')
        seen: list[str] = []

        def _agent(**_kwargs) -> AgentResult:
            seen.append(_lane_state(git_ops, lane))
            return AgentResult(success=True, output='')

        monkeypatch.setattr(
            'orchestrator.workflow.invoke_with_cap_retry', AsyncMock(side_effect=_agent),
        )
        await TaskWorkflow._invoke(workflow, SIMPLE_TASK, fixed_prompt('p'), lane)

        assert seen == ['in_use'], 'AGENT site: _invoke must spawn inside lane_in_use'
        assert _lane_state(git_ops, lane) == 'assigned'

    async def test_implement_phase_verify(self, tmp_path: Path, monkeypatch):
        workflow, git_ops, lane = await _workflow_in_lane(tmp_path)
        seen: list[str] = []

        def _verify(*_args, **_kwargs) -> VerifyResult:
            seen.append(_lane_state(git_ops, lane))
            return _passing_result()

        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification', AsyncMock(side_effect=_verify),
        )
        await workflow._run_scoped_verification_with_infra_retry(verify_attempt=0)

        assert seen == ['in_use'], (
            'IMPLEMENT-PHASE VERIFY site: the verify must run inside lane_in_use'
        )
        assert _lane_state(git_ops, lane) == 'assigned'

    async def test_merge_phase_reverify(self, tmp_path: Path, monkeypatch):
        workflow, git_ops, lane = await _workflow_in_lane(tmp_path)
        workflow.config.pre_merge_reverify = True
        _stub_merge_phase_preamble(monkeypatch, workflow)
        seen: list[str] = []

        def _reverify_then_bail(*_args, **_kwargs) -> VerifyResult:
            seen.append(_lane_state(git_ops, lane))
            raise _StopMergePhase()

        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(side_effect=_reverify_then_bail),
        )
        with pytest.raises(_StopMergePhase):
            await workflow._run_merge_phase('task/42')

        assert seen == ['in_use'], (
            'MERGE-PHASE RE-VERIFY site: the post-rebase re-verify must run '
            'inside lane_in_use'
        )
        assert _lane_state(git_ops, lane) == 'assigned'

    async def test_venv_sync(self, tmp_path: Path, monkeypatch):
        workflow, git_ops, lane = await _workflow_in_lane(tmp_path)
        (lane / 'lib').mkdir()
        (lane / 'lib' / 'pyproject.toml').write_text(
            '[project]\nname = "lib"\n\n'
            '[tool.uv.sources]\nshared = { path = "../shared", editable = true }\n'
        )
        seen: list[str] = []

        async def _uv_sync(*_args, **_kwargs) -> MagicMock:
            seen.append(_lane_state(git_ops, lane))
            proc = MagicMock(returncode=0)
            proc.communicate = AsyncMock(return_value=(b'', None))
            return proc

        monkeypatch.setattr(
            'orchestrator.workflow.asyncio.create_subprocess_exec', _uv_sync,
        )
        await workflow._sync_worktree_venvs()

        assert seen == ['in_use'], 'VENV SYNC site: uv sync must run inside lane_in_use'
        assert _lane_state(git_ops, lane) == 'assigned'

    async def test_dry_run_investigator(self, tmp_path: Path, monkeypatch):
        workflow, git_ops, lane = await _workflow_in_lane(tmp_path)
        workflow.config.unblock_auto.enabled = True
        seen: list[str] = []

        async def _investigate(**_kwargs) -> None:
            seen.append(_lane_state(git_ops, lane))

        monkeypatch.setattr('orchestrator.workflow.run_dry_run_unblock', _investigate)
        workflow._spawn_dry_run_unblock('blocked: demo', 'detail')
        spawned = [t for t in asyncio.all_tasks() if t.get_name() == 'unblock-auto-42']
        assert len(spawned) == 1
        await spawned[0]

        assert seen == ['in_use'], (
            'DRY-RUN INVESTIGATOR site: the background investigation must run '
            'inside lane_in_use'
        )
        assert _lane_state(git_ops, lane) == 'assigned'


# ---------------------------------------------------------------------------
# Crash recovery: the startup sweep, through the REAL writer and reader.
# ---------------------------------------------------------------------------


@pytest.fixture
def recovery_repo(tmp_path: Path) -> Path:
    return make_warm_lane_repo(tmp_path)


async def _crashed_in_flight_lane(repo: Path) -> tuple[Path, Path]:
    """Acquire a lane for live task 42 with progress on disk, as a run that
    then dies would leave it. Returns (worktree_base, lane)."""
    await add_warm_lane_scripts(repo)
    git_ops = build_harness(make_orch_config(repo)).git_ops
    lane = await acquire_lane(git_ops, '42', await get_head(repo))
    artifacts = TaskArtifacts(
        lane, TaskArtifacts.meta_root_for(git_ops.worktree_base, lane.name),
    )
    artifacts.init('42', 'crashed task', 'd')
    artifacts.write_plan(make_plan(3, 5, '42'))
    return git_ops.worktree_base, lane


async def _restart_and_recover(repo: Path) -> None:
    restarted = build_harness(make_orch_config(repo))
    wire_recovery_scheduler(restarted)
    await restarted._recover_crashed_tasks()


def _reset_lines(caplog, lane: Path) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records
        if r.levelno == logging.INFO and r.name == 'orchestrator.lane_lifecycle'
        and lane.name in r.getMessage() and '42' in r.getMessage()
    ]


@pytest.mark.asyncio
class TestCrashRecoveryResetsInUseFossils:
    async def test_a_record_left_in_use_by_a_crash_is_reset_to_assigned(
        self, recovery_repo: Path, caplog,
    ):
        worktree_base, lane = await _crashed_in_flight_lane(recovery_repo)
        # The shape a SIGKILL mid-span leaves behind.
        assert LaneLifecycle(worktree_base).note_in_use(lane) is True

        with caplog.at_level(logging.INFO, logger='orchestrator.lane_lifecycle'):
            await _restart_and_recover(recovery_repo)

        record = _on_disk(worktree_base, lane.name)
        assert (record['state'], record['task_id']) == ('assigned', '42')
        assert _reset_lines(caplog, lane), 'the startup reset must be logged'

    async def test_an_assigned_record_stays_assigned_and_is_not_reset(
        self, recovery_repo: Path, caplog,
    ):
        worktree_base, lane = await _crashed_in_flight_lane(recovery_repo)

        with caplog.at_level(logging.INFO, logger='orchestrator.lane_lifecycle'):
            await _restart_and_recover(recovery_repo)

        record = _on_disk(worktree_base, lane.name)
        assert (record['state'], record['task_id']) == ('assigned', '42')
        assert _reset_lines(caplog, lane) == []
