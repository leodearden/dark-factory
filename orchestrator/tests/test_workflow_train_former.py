"""Tests for the β train former — selection, stackability, and formation.

Covers:
  step-7:  _line_ranges_stackable (pure predicate)
  step-9:  _select_train_members (pure selection)
  step-11: _train_candidates (async candidate discovery)
  step-13: _maybe_form_train guards (disabled, lone-task, already-member)
  step-15: _maybe_form_train formation, overlap-rejection, and cap
  step-17: _maybe_defer_as_train_member routing helper
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


# ---------------------------------------------------------------------------
# Fixture helpers (mirrors test_workflow_train_completion._make)
# ---------------------------------------------------------------------------

@dataclass
class _Fixture:
    wf: TaskWorkflow
    scheduler: MagicMock
    git_ops: MagicMock


def _make(
    *,
    task_id: str = '200',
    metadata: dict | None = None,
    former_enabled: bool = False,
    max_members: int = 3,
    get_tasks_return: list[dict] | None = None,
) -> _Fixture:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {
        'id': task_id,
        'title': 'Former test task',
        'description': 'd',
        'metadata': metadata or {},
        'status': 'in-progress',
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
    config.merge_train_former_enabled = former_enabled
    config.merge_train_max_members = max_members

    scheduler = MagicMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.set_task_status = AsyncMock()
    scheduler.get_status = AsyncMock(return_value='in-progress')
    scheduler.get_task = AsyncMock(return_value=assignment.task)
    scheduler.mark_done = AsyncMock()
    scheduler.clear_requeue_count = MagicMock()
    scheduler.tasks_by_train = AsyncMock(return_value=[])
    scheduler.get_statuses = AsyncMock(return_value=({}, None))

    if get_tasks_return is not None:
        scheduler.get_tasks = AsyncMock(return_value=get_tasks_return)
    else:
        scheduler.get_tasks = AsyncMock(return_value=[])

    git_ops = MagicMock()
    git_ops.config.branch_prefix = 'task/'
    git_ops.config.main_branch = 'main'
    git_ops.resolve_branch_sha = AsyncMock(return_value='abc123')
    git_ops.get_changed_line_ranges = AsyncMock(return_value={})

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

    return _Fixture(wf=wf, scheduler=scheduler, git_ops=git_ops)


# ---------------------------------------------------------------------------
# step-7: _line_ranges_stackable — pure predicate tests
# ---------------------------------------------------------------------------


class TestLineRangesStackable:
    """Unit tests for the _line_ranges_stackable pure predicate in workflow.py."""

    def _fn(self):
        from orchestrator.workflow import _line_ranges_stackable
        return _line_ranges_stackable

    def test_disjoint_files_are_stackable(self):
        """Two tasks touching completely different files are always stackable."""
        fn = self._fn()
        a = {'src/foo.rs': [(1, 10)]}
        b = {'src/bar.rs': [(1, 10)]}
        assert fn(a, b) is True

    def test_same_file_non_overlapping_ranges_are_stackable(self):
        """Core stackable invariant: same file, different non-overlapping lines."""
        fn = self._fn()
        # Task A modifies lines 10-20 of foo.rs
        a = {'src/foo.rs': [(10, 20)]}
        # Task B modifies lines 30-40 of foo.rs — different lines, stackable
        b = {'src/foo.rs': [(30, 40)]}
        assert fn(a, b) is True

    def test_same_file_overlapping_ranges_not_stackable(self):
        """Same file, ranges overlap → not stackable."""
        fn = self._fn()
        a = {'src/foo.rs': [(10, 25)]}
        b = {'src/foo.rs': [(20, 35)]}
        assert fn(a, b) is False

    def test_same_file_identical_ranges_not_stackable(self):
        """Same file, identical ranges (extreme overlap) → not stackable."""
        fn = self._fn()
        a = {'src/foo.rs': [(10, 20)]}
        b = {'src/foo.rs': [(10, 20)]}
        assert fn(a, b) is False

    def test_one_side_empty_is_stackable(self):
        """A task touching no files is trivially stackable with anything."""
        fn = self._fn()
        a = {'src/foo.rs': [(10, 20)]}
        b: dict = {}
        assert fn(a, b) is True
        assert fn(b, a) is True

    def test_both_empty_is_stackable(self):
        fn = self._fn()
        assert fn({}, {}) is True

    def test_adjacent_ranges_are_stackable(self):
        """Ranges that abut (end of A + 1 == start of B) are non-overlapping."""
        fn = self._fn()
        a = {'src/foo.rs': [(10, 20)]}
        b = {'src/foo.rs': [(21, 30)]}
        assert fn(a, b) is True

    def test_shared_file_with_other_disjoint_file_not_stackable_when_overlap(self):
        """Overlapping shared file makes pair not stackable even if another file is disjoint."""
        fn = self._fn()
        a = {'src/foo.rs': [(10, 20)], 'src/bar.rs': [(1, 5)]}
        b = {'src/foo.rs': [(15, 25)], 'src/baz.rs': [(100, 110)]}
        assert fn(a, b) is False


# ---------------------------------------------------------------------------
# step-9: _select_train_members — pure selection tests
# ---------------------------------------------------------------------------


class TestSelectTrainMembers:
    """Unit tests for _select_train_members pure selection function."""

    def _fn(self):
        from orchestrator.workflow import _select_train_members
        return _select_train_members

    def _ranges(self, file: str, start: int, end: int) -> dict[str, list[tuple[int, int]]]:
        return {file: [(start, end)]}

    def test_anchor_plus_two_stackable_candidates(self):
        """Anchor + 2 mutually-stackable candidates → [anchor, c1, c2]."""
        fn = self._fn()
        # Anchor '200' modifies lines 1-10 of foo.rs
        # c1 '201' modifies lines 20-30 of foo.rs (no overlap with anchor)
        # c2 '202' modifies lines 40-50 of foo.rs (no overlap with either)
        ranges_by_id = {
            '200': {'src/foo.rs': [(1, 10)]},
            '201': {'src/foo.rs': [(20, 30)]},
            '202': {'src/foo.rs': [(40, 50)]},
        }
        result = fn('200', ['201', '202'], ranges_by_id, max_members=3)
        assert result == ['200', '201', '202']

    def test_candidate_overlapping_anchor_excluded(self):
        """A candidate overlapping the anchor is excluded from the train."""
        fn = self._fn()
        ranges_by_id = {
            '200': {'src/foo.rs': [(1, 20)]},
            '201': {'src/foo.rs': [(15, 30)]},  # overlaps anchor
        }
        result = fn('200', ['201'], ranges_by_id, max_members=3)
        # Only anchor, so <2 members → returns []
        assert result == []

    def test_candidate_overlapping_already_selected_member_excluded(self):
        """Mutual stackability: a candidate overlapping an already-selected member is excluded."""
        fn = self._fn()
        # c1 is stackable with anchor (200); c2 overlaps c1 but not anchor
        ranges_by_id = {
            '200': {'src/foo.rs': [(1, 10)]},
            '201': {'src/foo.rs': [(20, 30)]},   # stackable with anchor
            '202': {'src/foo.rs': [(25, 35)]},   # overlaps c1 → excluded
        }
        result = fn('200', ['201', '202'], ranges_by_id, max_members=3)
        assert result == ['200', '201']

    def test_cap_honored(self):
        """3 mutually-stackable candidates but max=2 → exactly 2 members."""
        fn = self._fn()
        ranges_by_id = {
            '200': {'src/foo.rs': [(1, 10)]},
            '201': {'src/foo.rs': [(20, 30)]},
            '202': {'src/foo.rs': [(40, 50)]},
            '203': {'src/foo.rs': [(60, 70)]},
        }
        result = fn('200', ['201', '202', '203'], ranges_by_id, max_members=2)
        assert len(result) == 2
        assert result[0] == '200'  # anchor always first

    def test_fewer_than_2_total_returns_empty(self):
        """Lone anchor (no stackable candidates) → returns [] sentinel for no-train."""
        fn = self._fn()
        ranges_by_id = {
            '200': {'src/foo.rs': [(1, 10)]},
        }
        result = fn('200', [], ranges_by_id, max_members=3)
        assert result == []

    def test_deterministic_order(self):
        """Candidates are processed in sorted id order for determinism."""
        fn = self._fn()
        ranges_by_id = {
            '200': {'src/foo.rs': [(1, 10)]},
            '201': {'src/foo.rs': [(20, 30)]},
            '202': {'src/foo.rs': [(40, 50)]},
        }
        # Shuffle candidate order — result should still be ['200','201','202']
        r1 = fn('200', ['202', '201'], ranges_by_id, max_members=3)
        r2 = fn('200', ['201', '202'], ranges_by_id, max_members=3)
        assert r1 == r2 == ['200', '201', '202']

    def test_anchor_always_order_0(self):
        """Anchor is always the first element (order-0) in the returned list."""
        fn = self._fn()
        ranges_by_id = {
            '200': {'src/foo.rs': [(1, 10)]},
            '201': {'src/foo.rs': [(20, 30)]},
        }
        result = fn('200', ['201'], ranges_by_id, max_members=3)
        assert result[0] == '200'


# ---------------------------------------------------------------------------
# step-11: _train_candidates — async candidate discovery tests
# ---------------------------------------------------------------------------


class TestTrainCandidates:
    """Unit tests for TaskWorkflow._train_candidates()."""

    def _make_task(
        self,
        id: str,
        status: str = 'in-progress',
        metadata: dict | None = None,
    ) -> dict:
        return {'id': id, 'status': status, 'metadata': metadata or {}}

    @pytest.mark.asyncio
    async def test_excludes_self(self):
        """_train_candidates never includes self (same task_id)."""
        fix = _make(
            task_id='200',
            get_tasks_return=[
                self._make_task('200'),  # self — must be excluded
                self._make_task('201'),  # eligible
            ],
        )
        result = await fix.wf._train_candidates()
        ids = [t['id'] for t in result]
        assert '200' not in ids
        assert '201' in ids

    @pytest.mark.asyncio
    async def test_excludes_non_in_progress_statuses(self):
        """Only in-progress tasks are candidates; done/blocked/merge-deferred excluded."""
        fix = _make(
            task_id='200',
            get_tasks_return=[
                self._make_task('201', status='done'),
                self._make_task('202', status='blocked'),
                self._make_task('203', status='merge-deferred'),
                self._make_task('204', status='in-progress'),  # the only eligible one
            ],
        )
        result = await fix.wf._train_candidates()
        ids = [t['id'] for t in result]
        assert ids == ['204']

    @pytest.mark.asyncio
    async def test_excludes_already_train_member(self):
        """Tasks already carrying metadata.train are excluded."""
        fix = _make(
            task_id='200',
            get_tasks_return=[
                self._make_task(
                    '201',
                    metadata={'train': {'id': 'train-x', 'order': 0, 'members': ['201']}},
                ),
                self._make_task('202'),  # no train metadata → eligible
            ],
        )
        result = await fix.wf._train_candidates()
        ids = [t['id'] for t in result]
        assert '201' not in ids
        assert '202' in ids

    @pytest.mark.asyncio
    async def test_excludes_unresolvable_branch(self):
        """Candidates whose branch doesn't resolve (resolve_branch_sha→None) are excluded."""
        fix = _make(
            task_id='200',
            get_tasks_return=[
                self._make_task('201'),  # branch resolves
                self._make_task('210'),  # branch does NOT resolve
            ],
        )
        # Task 210's branch is dead / not yet pushed.
        fix.git_ops.resolve_branch_sha = AsyncMock(
            side_effect=lambda branch: None if branch == '210' else 'abc123'
        )
        result = await fix.wf._train_candidates()
        ids = [t['id'] for t in result]
        assert '201' in ids
        assert '210' not in ids


# ---------------------------------------------------------------------------
# step-13: _maybe_form_train guards — disabled / lone-task / already-member
# ---------------------------------------------------------------------------


class TestMaybeFormTrainGuards:
    """Unit tests for _maybe_form_train() guard conditions (PRD §A.3 / B8)."""

    @pytest.mark.asyncio
    async def test_returns_false_when_former_disabled(self):
        """When merge_train_former_enabled=False, returns False immediately.

        No scheduler.update_task calls, no event emitted, self._train stays None.
        """
        fix = _make(former_enabled=False, get_tasks_return=[])
        fix.wf.event_store = MagicMock()  # shouldn't be called

        result = await fix.wf._maybe_form_train()

        assert result is False
        fix.scheduler.update_task.assert_not_called()
        fix.wf.event_store.emit.assert_not_called()
        assert fix.wf._train is None

    @pytest.mark.asyncio
    async def test_returns_false_when_lone_ready_task(self):
        """Enabled but _train_candidates returns [] → returns False, self merges solo."""
        fix = _make(
            former_enabled=True,
            # No other candidates — get_tasks returns only self
            get_tasks_return=[{'id': '200', 'status': 'in-progress', 'metadata': {}}],
        )
        fix.wf.event_store = MagicMock()

        result = await fix.wf._maybe_form_train()

        assert result is False
        fix.scheduler.update_task.assert_not_called()
        assert fix.wf._train is None

    @pytest.mark.asyncio
    async def test_returns_false_when_already_train_member(self):
        """If self._train is already set (task is a train member), return False immediately.

        Double-forming must be prevented: the former must be a no-op when the
        anchor is already in a train.
        """
        fix = _make(
            former_enabled=True,
            metadata={'train': {'id': 'train-existing', 'order': 0, 'members': ['200']}},
            get_tasks_return=[{'id': '201', 'status': 'in-progress', 'metadata': {}}],
        )
        fix.wf.event_store = MagicMock()

        result = await fix.wf._maybe_form_train()

        assert result is False
        fix.scheduler.update_task.assert_not_called()
        fix.wf.event_store.emit.assert_not_called()


# ---------------------------------------------------------------------------
# step-15: _maybe_form_train formation, overlap-rejection, cap-from-config
# ---------------------------------------------------------------------------


def _ip_task(id: str, metadata: dict | None = None) -> dict:
    """Shorthand: build a minimal in-progress task dict."""
    return {'id': id, 'status': 'in-progress', 'metadata': metadata or {}}


class TestMaybeFormTrainFormation:
    """Formation / overlap / cap tests for _maybe_form_train (PRD §B3, B4)."""

    def _ranges_side_effect(self, table: dict):
        """Return an AsyncMock side-effect that dispatches by ref (= task_id)."""
        async def _fn(ref):
            return table.get(ref, {})
        return _fn

    @pytest.mark.asyncio
    async def test_forms_train_with_non_overlapping_candidate(self):
        """Self + one candidate touching SAME file at NON-overlapping lines → train formed.

        Asserts:
        - returns True
        - scheduler.update_task called once per member with metadata={'train': {...}}
          carrying a shared id, ascending order, members list, and append=True
        - self.task['metadata']['train'] set in-memory (self._train is truthy)
        - event_store.emit called with EventType.train_formed containing train_id + members
        """
        fix = _make(
            task_id='200',
            former_enabled=True,
            max_members=3,
            get_tasks_return=[_ip_task('201')],
        )
        event_store = MagicMock()
        fix.wf.event_store = event_store

        # Self touches lines 1-10; candidate touches lines 20-30 — non-overlapping.
        fix.git_ops.get_changed_line_ranges = AsyncMock(
            side_effect=self._ranges_side_effect({
                '200': {'src/foo.rs': [(1, 10)]},
                '201': {'src/foo.rs': [(20, 30)]},
            })
        )

        result = await fix.wf._maybe_form_train()

        assert result is True

        # update_task called once per member (anchor + candidate = 2 calls).
        assert fix.scheduler.update_task.call_count == 2

        # Inspect calls — each must carry append=True and a valid train shape.
        calls = fix.scheduler.update_task.call_args_list
        by_id = {}
        for c in calls:
            assert c.kwargs.get('append') is True, 'update_task must use append=True'
            m_id = c.args[0]
            payload = c.args[1]
            assert isinstance(payload, dict), 'metadata payload must be a dict'
            assert 'train' in payload
            by_id[m_id] = payload['train']

        assert set(by_id) == {'200', '201'}
        # Both members share the same train id.
        assert by_id['200']['id'] == by_id['201']['id']
        # Anchor is order 0; other member is order 1.
        assert by_id['200']['order'] == 0
        assert by_id['201']['order'] == 1
        # Both see the full members list.
        assert sorted(by_id['200']['members']) == ['200', '201']
        assert sorted(by_id['201']['members']) == ['200', '201']

        # self._train must be truthy (in-memory metadata was updated).
        assert fix.wf._train is not None

        # train_formed event must have been emitted.
        event_store.emit.assert_called_once()
        emit_call = event_store.emit.call_args
        assert emit_call.args[0] is EventType.train_formed
        data = emit_call.kwargs.get('data', {})
        assert 'train_id' in data
        assert sorted(data['members']) == ['200', '201']

    @pytest.mark.asyncio
    async def test_no_train_when_candidate_overlaps_self(self):
        """Candidate overlapping self on a shared file → returns False, no train formed."""
        fix = _make(
            task_id='200',
            former_enabled=True,
            max_members=3,
            get_tasks_return=[_ip_task('201')],
        )
        event_store = MagicMock()
        fix.wf.event_store = event_store

        # Candidate overlaps self on foo.rs → not stackable.
        fix.git_ops.get_changed_line_ranges = AsyncMock(
            side_effect=self._ranges_side_effect({
                '200': {'src/foo.rs': [(1, 20)]},
                '201': {'src/foo.rs': [(15, 30)]},  # overlaps [1,20]
            })
        )

        result = await fix.wf._maybe_form_train()

        assert result is False
        # update_task must NOT have been called with train metadata.
        for c in fix.scheduler.update_task.call_args_list:
            payload = c.args[1] if c.args else {}
            assert 'train' not in payload, 'No train metadata on overlap-rejection'
        event_store.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_cap_limits_train_size(self):
        """3 mutually-stackable candidates with max_members=2 → exactly 2-member train."""
        fix = _make(
            task_id='200',
            former_enabled=True,
            max_members=2,  # cap at 2
            get_tasks_return=[_ip_task('201'), _ip_task('202'), _ip_task('203')],
        )
        fix.wf.event_store = MagicMock()

        fix.git_ops.get_changed_line_ranges = AsyncMock(
            side_effect=self._ranges_side_effect({
                '200': {'src/foo.rs': [(1, 10)]},
                '201': {'src/foo.rs': [(20, 30)]},
                '202': {'src/foo.rs': [(40, 50)]},
                '203': {'src/foo.rs': [(60, 70)]},
            })
        )

        result = await fix.wf._maybe_form_train()

        assert result is True
        # Cap=2 → exactly 2 update_task calls (anchor + 1 candidate).
        assert fix.scheduler.update_task.call_count == 2
        ids_updated = {c.args[0] for c in fix.scheduler.update_task.call_args_list}
        assert '200' in ids_updated  # anchor always included
        assert len(ids_updated) == 2
