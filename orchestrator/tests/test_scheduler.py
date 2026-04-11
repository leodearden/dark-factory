"""Tests for scheduler module lock logic."""


import time
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.scheduler import ModuleLockTable, Scheduler, files_to_modules


@pytest.fixture
def config() -> OrchestratorConfig:
    return OrchestratorConfig(
        max_per_module=1,
        module_overrides={'tests': 2},
    )


@pytest.fixture
def lock_table(config: OrchestratorConfig) -> ModuleLockTable:
    return ModuleLockTable(config)


class TestModuleLockTable:
    def test_acquire_single_module(self, lock_table: ModuleLockTable):
        assert lock_table.try_acquire('task-1', ['backend'])

    def test_acquire_blocks_second_task(self, lock_table: ModuleLockTable):
        assert lock_table.try_acquire('task-1', ['backend'])
        assert not lock_table.try_acquire('task-2', ['backend'])

    def test_release_allows_reacquire(self, lock_table: ModuleLockTable):
        assert lock_table.try_acquire('task-1', ['backend'])
        lock_table.release('task-1')
        assert lock_table.try_acquire('task-2', ['backend'])

    def test_acquire_multiple_modules(self, lock_table: ModuleLockTable):
        assert lock_table.try_acquire('task-1', ['backend', 'server'])
        # Both locked
        assert not lock_table.try_acquire('task-2', ['backend'])
        assert not lock_table.try_acquire('task-3', ['server'])
        # Unrelated module OK
        assert lock_table.try_acquire('task-4', ['frontend'])

    def test_partial_acquire_rolls_back(self, lock_table: ModuleLockTable):
        assert lock_table.try_acquire('task-1', ['backend'])
        # task-2 needs backend + server; backend is locked so should fail
        assert not lock_table.try_acquire('task-2', ['server', 'backend'])
        # server should NOT be locked (atomic failure)
        assert lock_table.try_acquire('task-3', ['server'])

    def test_module_override_allows_concurrency(self, lock_table: ModuleLockTable):
        # 'tests' module allows 2 concurrent
        assert lock_table.try_acquire('task-1', ['tests'])
        assert lock_table.try_acquire('task-2', ['tests'])
        assert not lock_table.try_acquire('task-3', ['tests'])

    def test_try_acquire_additional(self, lock_table: ModuleLockTable):
        assert lock_table.try_acquire('task-1', ['backend'])
        assert lock_table.try_acquire_additional('task-1', ['server'])
        # Both should be locked
        assert not lock_table.try_acquire('task-2', ['server'])

    def test_try_acquire_additional_fails(self, lock_table: ModuleLockTable):
        assert lock_table.try_acquire('task-1', ['backend'])
        assert lock_table.try_acquire('task-2', ['server'])
        # task-1 can't expand to server
        assert not lock_table.try_acquire_additional('task-1', ['server'])
        # task-2 still holds server
        assert not lock_table.try_acquire('task-3', ['server'])

    def test_try_acquire_additional_already_held(self, lock_table: ModuleLockTable):
        assert lock_table.try_acquire('task-1', ['backend', 'server'])
        # Already holds both — should succeed without double-acquiring
        assert lock_table.try_acquire_additional('task-1', ['backend', 'server'])

    def test_release_nonexistent_task(self, lock_table: ModuleLockTable):
        # Should not raise
        lock_table.release('nonexistent')

    def test_is_held_false_for_unknown(self, lock_table: ModuleLockTable):
        assert lock_table.is_held('nonexistent') is False

    def test_is_held_true_after_acquire(self, lock_table: ModuleLockTable):
        lock_table.try_acquire('task-1', ['backend'])
        assert lock_table.is_held('task-1') is True

    def test_is_held_false_after_release(self, lock_table: ModuleLockTable):
        lock_table.try_acquire('task-1', ['backend'])
        lock_table.release('task-1')
        assert lock_table.is_held('task-1') is False


class TestHierarchicalLocking:
    """Test that parent/child modules conflict but siblings don't."""

    def test_parent_blocks_child(self):
        """Lock on autopilot/analyze blocks autopilot/analyze/asr."""
        config = OrchestratorConfig(max_per_module=1, lock_depth=4)
        lt = ModuleLockTable(config)
        assert lt.try_acquire('t1', ['autopilot/analyze'])
        assert not lt.try_acquire('t2', ['autopilot/analyze/asr'])

    def test_child_blocks_parent(self):
        """Lock on autopilot/analyze/asr blocks autopilot/analyze."""
        config = OrchestratorConfig(max_per_module=1, lock_depth=4)
        lt = ModuleLockTable(config)
        assert lt.try_acquire('t1', ['autopilot/analyze/asr'])
        assert not lt.try_acquire('t2', ['autopilot/analyze'])

    def test_siblings_dont_conflict(self):
        """autopilot/analyze/asr and autopilot/analyze/speech are independent."""
        config = OrchestratorConfig(max_per_module=1, lock_depth=4)
        lt = ModuleLockTable(config)
        assert lt.try_acquire('t1', ['autopilot/analyze/asr'])
        assert lt.try_acquire('t2', ['autopilot/analyze/speech'])

    def test_deep_ancestor_blocks_deep_descendant(self):
        """Lock on src blocks src/server/handlers/auth."""
        config = OrchestratorConfig(max_per_module=1, lock_depth=5)
        lt = ModuleLockTable(config)
        assert lt.try_acquire('t1', ['src'])
        assert not lt.try_acquire('t2', ['src/server/handlers/auth'])

    def test_unrelated_paths_dont_conflict(self):
        config = OrchestratorConfig(max_per_module=1, lock_depth=4)
        lt = ModuleLockTable(config)
        assert lt.try_acquire('t1', ['backend/server'])
        assert lt.try_acquire('t2', ['frontend/components'])

    def test_release_parent_unblocks_child(self):
        config = OrchestratorConfig(max_per_module=1, lock_depth=4)
        lt = ModuleLockTable(config)
        assert lt.try_acquire('t1', ['autopilot/analyze'])
        assert not lt.try_acquire('t2', ['autopilot/analyze/asr'])
        lt.release('t1')
        assert lt.try_acquire('t2', ['autopilot/analyze/asr'])

    def test_task_own_modules_dont_self_conflict(self):
        """A task holding A should be able to expand to A/B via additional."""
        config = OrchestratorConfig(max_per_module=1, lock_depth=4)
        lt = ModuleLockTable(config)
        assert lt.try_acquire('t1', ['autopilot/analyze'])
        # Expanding to a child of an already-held parent should work
        assert lt.try_acquire_additional('t1', ['autopilot/analyze/asr'])

    def test_hierarchy_with_limit_gt_1(self):
        """Parent/child conflict still applies when limit > 1."""
        config = OrchestratorConfig(
            max_per_module=2, lock_depth=4,
        )
        lt = ModuleLockTable(config)
        assert lt.try_acquire('t1', ['autopilot/analyze'])
        # Second task on parent: allowed (limit=2)
        assert lt.try_acquire('t2', ['autopilot/analyze'])
        # Third task on child: blocked (2 conflicts from t1 and t2)
        assert not lt.try_acquire('t3', ['autopilot/analyze/asr'])

    def test_exact_prefix_string_not_confused(self):
        """'src/server' must not conflict with 'src/serverless' (not a parent)."""
        config = OrchestratorConfig(max_per_module=1, lock_depth=4)
        lt = ModuleLockTable(config)
        assert lt.try_acquire('t1', ['src/server'])
        assert lt.try_acquire('t2', ['src/serverless'])


class TestFilesToModules:
    def test_basic_derivation(self):
        files = [
            'autopilot/analyze/asr/model.py',
            'autopilot/analyze/asr/tests/test_model.py',
            'autopilot/analyze/speech/recognizer.py',
        ]
        assert files_to_modules(files, depth=3) == [
            'autopilot/analyze/asr',
            'autopilot/analyze/speech',
        ]

    def test_depth_2_collapses(self):
        files = [
            'autopilot/analyze/asr/model.py',
            'autopilot/analyze/speech/recognizer.py',
        ]
        assert files_to_modules(files, depth=2) == ['autopilot/analyze']

    def test_deduplication(self):
        files = [
            'src/server/app.py',
            'src/server/routes.py',
            'src/server/models.py',
        ]
        assert files_to_modules(files, depth=2) == ['src/server']

    def test_empty_list(self):
        assert files_to_modules([], depth=2) == []

    def test_single_component_files(self):
        files = ['setup.py', 'pyproject.toml']
        assert files_to_modules(files, depth=2) == ['pyproject.toml', 'setup.py']

    def test_mixed_depths(self):
        files = [
            'orchestrator/src/orchestrator/scheduler.py',
            'orchestrator/tests/test_scheduler.py',
            'dashboard/src/dashboard/app.py',
        ]
        assert files_to_modules(files, depth=2) == [
            'dashboard/src',
            'orchestrator/src',
            'orchestrator/tests',
        ]


class TestConflictsMethod:
    """Direct unit tests for the _conflicts static method."""

    def test_exact_match(self):
        assert ModuleLockTable._conflicts('a/b', 'a/b')

    def test_parent_child(self):
        assert ModuleLockTable._conflicts('a', 'a/b')

    def test_child_parent(self):
        assert ModuleLockTable._conflicts('a/b', 'a')

    def test_siblings(self):
        assert not ModuleLockTable._conflicts('a/b', 'a/c')

    def test_prefix_string_not_hierarchy(self):
        """'ab' is not a parent of 'abc'."""
        assert not ModuleLockTable._conflicts('ab', 'abc')

    def test_deep_hierarchy(self):
        assert ModuleLockTable._conflicts('a/b/c', 'a/b/c/d/e')

    def test_completely_unrelated(self):
        assert not ModuleLockTable._conflicts('foo', 'bar')


class TestModuleLockWithModuleConfig:
    """Test that ModuleConfig overrides are respected in lock limits."""

    def test_limit_uses_mc_max_per_module(self):
        config = OrchestratorConfig(max_per_module=1, lock_depth=1)
        config._module_configs = {
            'dashboard': ModuleConfig(prefix='dashboard', max_per_module=3),
        }
        lt = ModuleLockTable(config)
        assert lt.try_acquire('t1', ['dashboard'])
        assert lt.try_acquire('t2', ['dashboard'])
        assert lt.try_acquire('t3', ['dashboard'])
        assert not lt.try_acquire('t4', ['dashboard'])

    def test_limit_uses_mc_module_overrides(self):
        config = OrchestratorConfig(max_per_module=1, lock_depth=1)
        config._module_configs = {
            'dashboard': ModuleConfig(
                prefix='dashboard',
                max_per_module=1,
                module_overrides={'dashboard': 2},
            ),
        }
        lt = ModuleLockTable(config)
        assert lt.try_acquire('t1', ['dashboard'])
        assert lt.try_acquire('t2', ['dashboard'])
        assert not lt.try_acquire('t3', ['dashboard'])

    def test_global_override_still_works(self):
        """Global module_overrides takes effect when no ModuleConfig matches."""
        config = OrchestratorConfig(
            max_per_module=1, lock_depth=1,
            module_overrides={'infra': 3},
        )
        lt = ModuleLockTable(config)
        assert lt.try_acquire('t1', ['infra'])
        assert lt.try_acquire('t2', ['infra'])
        assert lt.try_acquire('t3', ['infra'])
        assert not lt.try_acquire('t4', ['infra'])

    def test_mc_module_overrides_beats_global(self):
        """Subproject module_overrides takes precedence over global module_overrides."""
        config = OrchestratorConfig(
            max_per_module=1, lock_depth=1,
            module_overrides={'dashboard': 5},
        )
        config._module_configs = {
            'dashboard': ModuleConfig(
                prefix='dashboard',
                module_overrides={'dashboard': 2},
            ),
        }
        lt = ModuleLockTable(config)
        assert lt.try_acquire('t1', ['dashboard'])
        assert lt.try_acquire('t2', ['dashboard'])
        assert not lt.try_acquire('t3', ['dashboard'])


class TestStatusTransitionGuard:
    """Scheduler.set_task_status must reject transitions from terminal states."""

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @pytest.mark.asyncio
    async def test_rejects_done_to_blocked(self, scheduler: Scheduler, monkeypatch):
        """Once a task is DONE, subsequent set_task_status('blocked') is silently dropped."""
        mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        await scheduler.set_task_status('1', 'in-progress')
        await scheduler.set_task_status('1', 'done')
        await scheduler.set_task_status('1', 'blocked')

        # Only in-progress and done should have triggered MCP calls
        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_rejects_done_to_in_progress(self, scheduler: Scheduler, monkeypatch):
        """done->in-progress is rejected (duplicate workflow start)."""
        mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        await scheduler.set_task_status('1', 'done')
        call_count_after_done = mock.call_count
        await scheduler.set_task_status('1', 'in-progress')

        assert mock.call_count == call_count_after_done  # no new MCP call

    @pytest.mark.asyncio
    async def test_allows_done_to_done_idempotent(self, scheduler: Scheduler, monkeypatch):
        """Idempotent done->done transitions must be allowed."""
        mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        await scheduler.set_task_status('1', 'done')
        await scheduler.set_task_status('1', 'done')

        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_allows_normal_flow(self, scheduler: Scheduler, monkeypatch):
        """in-progress->done is a valid transition and must go through."""
        mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        await scheduler.set_task_status('2', 'in-progress')
        await scheduler.set_task_status('2', 'done')

        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_fail_open_unknown_task(self, scheduler: Scheduler, monkeypatch):
        """First-ever set_task_status for an unknown task always calls mcp_call."""
        mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        await scheduler.set_task_status('unknown-99', 'blocked')

        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_survives_release(self, scheduler: Scheduler, monkeypatch):
        """Status cache persists beyond lock release so stale workflows are still blocked."""
        mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        await scheduler.set_task_status('1', 'done')
        scheduler.release('1')  # clears _dispatched and module lock, NOT status cache
        call_count_before = mock.call_count

        await scheduler.set_task_status('1', 'blocked')

        # Cache still shows 'done', so blocked is rejected
        assert mock.call_count == call_count_before


class TestGetTasksSeedsCache:
    """get_tasks() should populate the status cache so pre-existing terminal tasks are guarded.

    Uses get_cached_status() to verify cache contents.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @pytest.mark.asyncio
    async def test_get_tasks_seeds_status_cache(self, scheduler: Scheduler, monkeypatch):
        """After get_tasks(), set_task_status for a 'done' task is rejected without MCP call."""
        tasks_response = {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': '{"tasks": [{"id": "42", "status": "done", "title": "T"}]}',
                    }
                ]
            }
        }
        mcp_mock = AsyncMock(return_value=tasks_response)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mcp_mock)

        await scheduler.get_tasks()  # seeds cache from tasks response

        call_count_after_seed = mcp_mock.call_count

        # Now try to set blocked on the done task
        await scheduler.set_task_status('42', 'blocked')

        # Cache was seeded: the blocked call must be rejected (no new MCP call)
        assert mcp_mock.call_count == call_count_after_seed


    @pytest.mark.asyncio
    async def test_get_tasks_resyncs_from_store_on_reinstatement(
        self, scheduler: Scheduler, monkeypatch
    ):
        """get_tasks() must update cache when store shows a reinstated (non-terminal) status.

        External processes may reinstate cancelled/done tasks to pending.  The
        store is the source of truth — the cache must reflect the reinstatement
        so that set_task_status() does not silently reject valid transitions.
        """
        # Step 1: Populate cache with terminal status via set_task_status
        set_mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', set_mock)
        await scheduler.set_task_status('42', 'cancelled')
        assert scheduler.get_cached_status('42') == 'cancelled'

        # Step 2: Simulate store reinstatement — task 42 is now 'pending'
        reinstated_response = {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': '{"tasks": [{"id": "42", "status": "pending", "title": "T"}]}',
                    }
                ]
            }
        }
        get_mock = AsyncMock(return_value=reinstated_response)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', get_mock)

        # Step 3: get_tasks() must resync cache to match store
        await scheduler.get_tasks()
        assert scheduler.get_cached_status('42') == 'pending', (
            'get_tasks() must trust store reinstatement of cancelled -> pending'
        )

        # Step 4: set_task_status('in-progress') must now succeed
        call_count_before = get_mock.call_count
        await scheduler.set_task_status('42', 'in-progress')
        assert get_mock.call_count == call_count_before + 1, (
            'set_task_status(in-progress) must succeed after store reinstatement'
        )

    @pytest.mark.asyncio
    async def test_get_tasks_updates_non_terminal_cache(
        self, scheduler: Scheduler, monkeypatch
    ):
        """get_tasks() should update cache entries that are not yet terminal."""
        # Pre-populate cache with a non-terminal status
        set_mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', set_mock)
        await scheduler.set_task_status('42', 'in-progress')
        assert scheduler.get_cached_status('42') == 'in-progress'

        # Simulate taskmaster reporting a different non-terminal status ('blocked')
        update_response = {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': '{"tasks": [{"id": "42", "status": "blocked", "title": "T"}]}',
                    }
                ]
            }
        }
        get_mock = AsyncMock(return_value=update_response)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', get_mock)

        await scheduler.get_tasks()

        # Non-terminal cache entry SHOULD be updated
        assert scheduler.get_cached_status('42') == 'blocked', (
            'get_tasks() should update non-terminal cache entries'
        )


class TestAcquireNextNoDuplicates:
    """acquire_next() must not return the same task twice while its locks are held."""

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @pytest.mark.asyncio
    async def test_acquire_next_skips_already_dispatched(self, scheduler: Scheduler):
        """Second acquire_next() for an already-held task returns None."""
        task = {
            'id': '1',
            'title': 'Task one',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'modules': ['backend']},
        }
        scheduler.get_tasks = AsyncMock(return_value=[task])

        first = await scheduler.acquire_next()
        assert first is not None
        assert first.task_id == '1'

        second = await scheduler.acquire_next()
        assert second is None

    @pytest.mark.asyncio
    async def test_acquire_next_returns_different_tasks_sequentially(
        self, scheduler: Scheduler
    ):
        """With two non-conflicting tasks, returns A then B then None."""
        task_a = {
            'id': '1',
            'title': 'Task A',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'modules': ['backend']},
        }
        task_b = {
            'id': '2',
            'title': 'Task B',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'modules': ['frontend']},
        }
        scheduler.get_tasks = AsyncMock(return_value=[task_a, task_b])

        first = await scheduler.acquire_next()
        assert first is not None
        first_id = first.task_id

        second = await scheduler.acquire_next()
        assert second is not None
        assert second.task_id != first_id

        third = await scheduler.acquire_next()
        assert third is None

    @pytest.mark.asyncio
    async def test_release_clears_dispatched_allowing_redispatch(
        self, scheduler: Scheduler
    ):
        """After release(), the same task can be dispatched again."""
        task = {
            'id': '1',
            'title': 'Task one',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'modules': ['backend']},
        }
        scheduler.get_tasks = AsyncMock(return_value=[task])

        first = await scheduler.acquire_next()
        assert first is not None
        assert first.task_id == '1'

        # Release — should clear _dispatched so task can be re-dispatched
        scheduler.release('1')

        # Same task is still pending (mock unchanged) — should be dispatchable again
        second = await scheduler.acquire_next()
        assert second is not None
        assert second.task_id == '1'

    @pytest.mark.asyncio
    async def test_acquire_next_dispatches_different_tasks_concurrently(
        self, scheduler: Scheduler
    ):
        """Three non-conflicting tasks can each be dispatched in turn; fourth returns None."""
        tasks = [
            {
                'id': '1',
                'title': 'Backend task',
                'status': 'pending',
                'dependencies': [],
                'metadata': {'modules': ['backend']},
            },
            {
                'id': '2',
                'title': 'Frontend task',
                'status': 'pending',
                'dependencies': [],
                'metadata': {'modules': ['frontend']},
            },
            {
                'id': '3',
                'title': 'Infra task',
                'status': 'pending',
                'dependencies': [],
                'metadata': {'modules': ['infra']},
            },
        ]
        scheduler.get_tasks = AsyncMock(return_value=tasks)

        first = await scheduler.acquire_next()
        second = await scheduler.acquire_next()
        third = await scheduler.acquire_next()

        ids = {a.task_id for a in [first, second, third] if a is not None}
        assert ids == {'1', '2', '3'}, f'Expected 3 distinct tasks, got: {ids}'

        fourth = await scheduler.acquire_next()
        assert fourth is None

    @pytest.mark.asyncio
    async def test_acquire_next_lock_conflict_plus_dispatch_guard(
        self, scheduler: Scheduler
    ):
        """Two tasks on the same module: dispatch A, B blocked; release A, B dispatches."""
        task_a = {
            'id': 'A',
            'title': 'Task A',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'modules': ['backend']},
        }
        task_b = {
            'id': 'B',
            'title': 'Task B',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'modules': ['backend']},
        }
        scheduler.get_tasks = AsyncMock(return_value=[task_a, task_b])

        # First dispatch — task A (or B) acquires the module lock
        first = await scheduler.acquire_next()
        assert first is not None
        dispatched_id = first.task_id
        other_id = 'B' if dispatched_id == 'A' else 'A'

        # Both guards (dispatch set + module lock) block re-dispatch of same task
        # AND lock blocks dispatch of the other task on same module
        second = await scheduler.acquire_next()
        assert second is None, 'Both tasks should be blocked: one dispatched, other locked'

        # Release the dispatched task — clears _dispatched AND module lock
        scheduler.release(dispatched_id)

        # Now the module is free: a task can be dispatched again.
        # (dispatched_id's mock status is still pending, so it or other_id may win;
        # what matters is that the lock+dispatch guard no longer blocks everything.)
        third = await scheduler.acquire_next()
        assert third is not None, 'After release(), a task should be dispatchable'
        _ = other_id  # acknowledged; exact winner depends on sort order


class TestDepsSatisfied:
    """Unit tests for Scheduler._deps_satisfied(task, status_map)."""

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    def test_deps_satisfied_returns_false_when_dep_in_progress(
        self, scheduler: Scheduler
    ):
        """_deps_satisfied returns False when a dependency is in-progress."""
        task = {'id': '2', 'dependencies': [{'id': 1}]}
        status_map = {'1': 'in-progress', '2': 'pending'}
        assert scheduler._deps_satisfied(task, status_map) is False

    def test_deps_satisfied_returns_true_when_dep_done(self, scheduler: Scheduler):
        """_deps_satisfied returns True when all dependencies are done."""
        task = {'id': '2', 'dependencies': [{'id': 1}]}
        status_map = {'1': 'done', '2': 'pending'}
        assert scheduler._deps_satisfied(task, status_map) is True

    def test_deps_satisfied_returns_true_when_no_deps(self, scheduler: Scheduler):
        """_deps_satisfied returns True when there are no dependencies."""
        task = {'id': '1', 'dependencies': []}
        status_map = {}
        assert scheduler._deps_satisfied(task, status_map) is True


class TestAcquireNextDependencyGating:
    """acquire_next() must not dispatch tasks whose dependencies are not done."""

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @pytest.mark.asyncio
    async def test_acquire_next_blocks_on_in_progress_dependency(
        self, scheduler: Scheduler
    ):
        """acquire_next returns None when the only candidate's dep is in-progress."""
        task_a = {
            'id': 'A',
            'title': 'Task A',
            'status': 'in-progress',
            'dependencies': [],
            'metadata': {'modules': ['backend']},
        }
        task_b = {
            'id': 'B',
            'title': 'Task B',
            'status': 'pending',
            'dependencies': [{'id': 'A'}],
            'metadata': {'modules': ['frontend']},
        }
        scheduler.get_tasks = AsyncMock(return_value=[task_a, task_b])

        result = await scheduler.acquire_next()
        # A is in-progress (not pending), B is blocked by A — neither can be dispatched
        assert result is None

    @pytest.mark.asyncio
    async def test_acquire_next_blocks_on_pending_dependency(
        self, scheduler: Scheduler
    ):
        """acquire_next returns None for task B when its dep A has been dispatched (not done)."""
        task_a = {
            'id': 'A',
            'title': 'Task A',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'modules': ['backend']},
        }
        task_b = {
            'id': 'B',
            'title': 'Task B',
            'status': 'pending',
            'dependencies': [{'id': 'A'}],
            'metadata': {'modules': ['frontend']},
        }
        scheduler.get_tasks = AsyncMock(return_value=[task_a, task_b])

        # Dispatch A first
        first = await scheduler.acquire_next()
        assert first is not None
        assert first.task_id == 'A'

        # B's dep A is still pending (mock status unchanged) — B must be blocked
        second = await scheduler.acquire_next()
        assert second is None, 'B should be blocked because dep A is not done'

    @pytest.mark.asyncio
    async def test_acquire_next_dispatches_when_all_deps_done(
        self, scheduler: Scheduler
    ):
        """acquire_next returns task B when its dependency A has status 'done'."""
        task_a = {
            'id': 'A',
            'title': 'Task A',
            'status': 'done',
            'dependencies': [],
            'metadata': {'modules': ['backend']},
        }
        task_b = {
            'id': 'B',
            'title': 'Task B',
            'status': 'pending',
            'dependencies': [{'id': 'A'}],
            'metadata': {'modules': ['frontend']},
        }
        scheduler.get_tasks = AsyncMock(return_value=[task_a, task_b])

        result = await scheduler.acquire_next()
        assert result is not None
        assert result.task_id == 'B'

    @pytest.mark.asyncio
    async def test_acquire_next_blocks_on_mixed_dep_statuses(
        self, scheduler: Scheduler
    ):
        """acquire_next blocks task C when one dep is done but another is in-progress."""
        task_a = {
            'id': 'A',
            'title': 'Task A',
            'status': 'done',
            'dependencies': [],
            'metadata': {'modules': ['backend']},
        }
        task_b = {
            'id': 'B',
            'title': 'Task B',
            'status': 'in-progress',
            'dependencies': [],
            'metadata': {'modules': ['backend']},
        }
        task_c = {
            'id': 'C',
            'title': 'Task C',
            'status': 'pending',
            'dependencies': [{'id': 'A'}, {'id': 'B'}],
            'metadata': {'modules': ['frontend']},
        }
        scheduler.get_tasks = AsyncMock(return_value=[task_a, task_b, task_c])

        result = await scheduler.acquire_next()
        # A is done, B is in-progress (not pending), C is blocked by B — nothing to dispatch
        assert result is None

    @pytest.mark.asyncio
    async def test_acquire_next_handles_dict_and_int_dependency_formats(
        self, scheduler: Scheduler
    ):
        """_deps_satisfied correctly resolves dict, int, and str dependency ID formats."""
        # Task with integer dep ID
        task_int = {
            'id': '10',
            'title': 'Task int dep',
            'status': 'pending',
            'dependencies': [1],  # integer format
            'metadata': {'modules': ['backend']},
        }
        # Task with string dep ID
        task_str = {
            'id': '11',
            'title': 'Task str dep',
            'status': 'pending',
            'dependencies': ['1'],  # string format
            'metadata': {'modules': ['frontend']},
        }
        # Task with dict dep ID
        task_dict = {
            'id': '12',
            'title': 'Task dict dep',
            'status': 'pending',
            'dependencies': [{'id': 1}],  # dict format
            'metadata': {'modules': ['ops']},
        }
        dep_done = {
            'id': '1',
            'title': 'Dep task',
            'status': 'done',
            'dependencies': [],
            'metadata': {'modules': []},
        }
        scheduler.get_tasks = AsyncMock(
            return_value=[dep_done, task_int, task_str, task_dict]
        )

        dispatched_ids: set[str] = set()
        for _ in range(3):
            result = await scheduler.acquire_next()
            assert result is not None, 'Expected to dispatch one of the dependent tasks'
            dispatched_ids.add(result.task_id)

        assert dispatched_ids == {'10', '11', '12'}, (
            'All three dependency-format variants should be dispatchable when dep is done'
        )

        # No more tasks
        result = await scheduler.acquire_next()
        assert result is None


class TestDepsSatisfiedLogging:
    """_deps_satisfied emits a debug log identifying the blocking dependency."""

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    def test_deps_satisfied_logs_blocking_reason(
        self, scheduler: Scheduler, caplog: pytest.LogCaptureFixture
    ):
        """_deps_satisfied emits a debug log with dep ID and current status when blocked."""
        import logging

        task = {'id': '99', 'dependencies': [{'id': '42'}]}
        status_map = {'42': 'in-progress'}

        with caplog.at_level(logging.DEBUG, logger='orchestrator.scheduler'):
            result = scheduler._deps_satisfied(task, status_map)

        assert result is False
        assert any(
            '42' in record.message and 'in-progress' in record.message
            for record in caplog.records
        ), f'Expected log about dep 42 being in-progress. Got: {[r.message for r in caplog.records]}'


class TestGetModulesJsonStringMetadata:
    """_get_modules must parse JSON string metadata, not just dict metadata."""

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    def test_get_modules_extracts_modules_from_json_string_metadata(
        self, scheduler: Scheduler
    ):
        """_get_modules returns normalized module list when metadata is a JSON string with 'modules'."""
        task = {
            'id': '5',
            'metadata': '{"modules": ["backend", "server"]}',
        }
        result = scheduler._get_modules(task)
        # Should NOT fall back to task-5 — must extract modules from JSON string
        assert result != ['task-5'], (
            f'Expected modules from JSON string, got fallback: {result}'
        )
        # Should have normalized module entries
        assert len(result) > 0
        assert all(isinstance(m, str) for m in result)
        # Verify neither entry is the fallback
        assert 'task-5' not in result

    def test_get_modules_extracts_files_from_json_string_metadata(
        self, scheduler: Scheduler
    ):
        """_get_modules returns file-derived modules when metadata is a JSON string with 'files'."""
        task = {
            'id': '6',
            'metadata': '{"files": ["src/server/app.py", "src/server/routes.py"]}',
        }
        result = scheduler._get_modules(task)
        # Should NOT fall back to task-6 — must extract from JSON string
        assert result != ['task-6'], (
            f'Expected file-derived modules from JSON string, got fallback: {result}'
        )
        assert len(result) > 0
        assert 'task-6' not in result

    def test_get_modules_handles_malformed_json_string_metadata(
        self, scheduler: Scheduler
    ):
        """_get_modules gracefully degrades to task-<id> fallback on malformed JSON string."""
        task = {
            'id': '7',
            'metadata': 'not valid json',
        }
        # Should not raise — must degrade gracefully to fallback
        result = scheduler._get_modules(task)
        assert result == ['task-7']

    def test_get_modules_logs_warning_on_fallback(
        self, scheduler: Scheduler, caplog: pytest.LogCaptureFixture
    ):
        """_get_modules emits a WARNING when falling back to task-<id> lock."""
        import logging

        task = {'id': '8', 'metadata': {}}
        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduler'):
            result = scheduler._get_modules(task)

        assert result == ['task-8']
        assert any(
            '8' in record.message and 'fallback' in record.message.lower()
            for record in caplog.records
        ), f'Expected fallback warning mentioning task 8. Got: {[r.message for r in caplog.records]}'

    def test_get_modules_fallback_warning_emitted_only_once(
        self, scheduler: Scheduler, caplog: pytest.LogCaptureFixture
    ):
        """_get_modules emits the fallback WARNING at most once per task ID.

        When _get_modules is called multiple times with the same task that has
        no module metadata, the WARNING must appear exactly once — not on every call.
        This prevents log flooding in the scheduler poll loop.
        """
        import logging

        task = {'id': '9', 'metadata': {}}
        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduler'):
            scheduler._get_modules(task)
            scheduler._get_modules(task)

        matching = [
            r for r in caplog.records
            if '9' in r.message and 'fallback' in r.message.lower()
        ]
        assert len(matching) == 1, (
            f'Expected exactly 1 fallback warning for task 9, got {len(matching)}. '
            f'Messages: {[r.message for r in caplog.records]}'
        )


class TestUpdateTaskMetadataSerialization:
    """Regression tests for update_task dict->JSON string coercion."""

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @pytest.mark.asyncio
    async def test_update_task_serializes_dict_to_json_string(
        self, scheduler: Scheduler, monkeypatch
    ):
        """update_task converts dict metadata to a JSON string before the MCP call."""
        import json

        captured_args: list[dict] = []

        async def mock_mcp_call(url, method, payload, **kwargs):
            captured_args.append(payload)
            return {}

        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock_mcp_call)

        await scheduler.update_task('1', {'modules': ['backend']})

        assert len(captured_args) == 1
        arguments = captured_args[0]['arguments']
        metadata = arguments['metadata']
        # Must be a string, not a dict
        assert isinstance(metadata, str), f'Expected str metadata, got {type(metadata)}: {metadata}'
        # Must be valid JSON that round-trips correctly
        parsed = json.loads(metadata)
        assert parsed == {'modules': ['backend']}

    @pytest.mark.asyncio
    async def test_update_task_passes_string_metadata_through(
        self, scheduler: Scheduler, monkeypatch
    ):
        """update_task passes string metadata unchanged — no double-serialization."""
        captured_args: list[dict] = []

        async def mock_mcp_call(url, method, payload, **kwargs):
            captured_args.append(payload)
            return {}

        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock_mcp_call)

        await scheduler.update_task('1', '{"modules": ["backend"]}')

        assert len(captured_args) == 1
        arguments = captured_args[0]['arguments']
        metadata = arguments['metadata']
        # Must be the same string — no double-serialization
        assert metadata == '{"modules": ["backend"]}'

    @pytest.mark.asyncio
    async def test_update_task_serializes_prd_metadata_dict(
        self, scheduler: Scheduler, monkeypatch
    ):
        """update_task converts a PRD dict metadata to a JSON string before the MCP call."""
        import json

        captured_args: list[dict] = []

        async def mock_mcp_call(url, method, payload, **kwargs):
            captured_args.append(payload)
            return {}

        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock_mcp_call)

        await scheduler.update_task('42', {'prd': '/abs/path/to/feature.prd'})

        assert len(captured_args) == 1
        arguments = captured_args[0]['arguments']
        metadata = arguments['metadata']
        # Must be a string, not a dict
        assert isinstance(metadata, str), f'Expected str metadata, got {type(metadata)}: {metadata}'
        # Must be valid JSON that round-trips correctly
        parsed = json.loads(metadata)
        assert parsed == {'prd': '/abs/path/to/feature.prd'}


class TestCancelledTaskResync:
    """Regression: externally reinstated cancelled tasks must not spin-loop.

    Reproduces the bug where 15 cancelled tasks were reinstated to pending in
    the store, but the scheduler's status cache still held 'cancelled'.  The
    terminal guard rejected every cancelled->in-progress transition, and the
    task was immediately reacquired — an infinite spin that starved pending
    tasks (156k+ rejections for a single task).

    Uses get_cached_status() to verify cache contents.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @pytest.mark.asyncio
    async def test_reinstated_cancelled_task_dispatches_successfully(
        self, scheduler: Scheduler, monkeypatch
    ):
        """Full cycle: cancel -> external reinstatement -> acquire -> in-progress succeeds."""
        mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        # 1. Orchestrator cancels the task — cache records 'cancelled'
        await scheduler.set_task_status('64', 'cancelled')
        assert scheduler.get_cached_status('64') == 'cancelled'

        # 2. External process reinstates to pending (store returns pending)
        reinstated_task = {
            'id': '64',
            'title': 'Reinstated task',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'modules': ['backend']},
        }
        reinstated_response = {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': '{"tasks": [' + __import__('json').dumps(reinstated_task) + ']}',
                    }
                ]
            }
        }
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=reinstated_response),
        )

        # 3. acquire_next picks up the task and resyncs cache
        assignment = await scheduler.acquire_next()
        assert assignment is not None, (
            'Reinstated task must be acquirable — was silently rejected before fix'
        )
        assert assignment.task_id == '64'

        # 4. Cache must now show 'pending' (resynced from store)
        #    so that set_task_status('in-progress') is NOT rejected
        assert scheduler.get_cached_status('64') == 'pending'

        # 5. Transition to in-progress must succeed (not be silently dropped)
        ip_mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', ip_mock)
        await scheduler.set_task_status('64', 'in-progress')

        assert ip_mock.call_count == 1, (
            'set_task_status(in-progress) must issue MCP call after reinstatement — '
            'was silently rejected by stale terminal guard before fix'
        )
        assert scheduler.get_cached_status('64') == 'in-progress'

    @pytest.mark.asyncio
    async def test_spin_loop_does_not_occur(
        self, scheduler: Scheduler, monkeypatch
    ):
        """Simulates N acquire cycles — each must make progress, not spin on rejection."""
        mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        # Cancel the task
        await scheduler.set_task_status('57', 'cancelled')

        # Store reinstates to pending
        task = {
            'id': '57',
            'title': 'Spin-loop candidate',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'modules': ['frontend']},
        }
        store_response = {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': '{"tasks": [' + __import__('json').dumps(task) + ']}',
                    }
                ]
            }
        }
        store_mock = AsyncMock(return_value=store_response)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', store_mock)

        # First acquire — should get the task
        a1 = await scheduler.acquire_next()
        assert a1 is not None and a1.task_id == '57'

        # set_task_status should succeed (not be silently rejected)
        ip_mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', ip_mock)
        await scheduler.set_task_status('57', 'in-progress')
        assert ip_mock.call_count == 1, 'in-progress transition must not be rejected'

        # Second acquire — task is already dispatched, should return None
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', store_mock)
        a2 = await scheduler.acquire_next()
        assert a2 is None, 'Task already dispatched — second acquire must return None'


class TestRequeueCooldown:
    """Tests for the requeue cooldown that prevents ghost loops."""

    @pytest.fixture
    def pending_task(self):
        return {
            'id': '99',
            'title': 'Cooldown test task',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'modules': ['backend']},
        }

    @pytest.fixture
    def task_response(self, pending_task):
        import json as _json
        return {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': '{"tasks": [' + _json.dumps(pending_task) + ']}',
                    }
                ]
            }
        }

    @pytest.mark.asyncio
    async def test_requeue_cooldown_blocks_reacquire(self, monkeypatch, task_response):
        """After release(requeued=True), task must not be re-acquired during cooldown."""
        config = OrchestratorConfig(max_per_module=1, requeue_cooldown_secs=30.0)
        scheduler = Scheduler(config)

        mock = AsyncMock(return_value=task_response)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        # Acquire the task
        a1 = await scheduler.acquire_next()
        assert a1 is not None and a1.task_id == '99'

        # Release with requeue flag
        scheduler.release('99', requeued=True)

        # Try to acquire again — should be blocked by cooldown
        a2 = await scheduler.acquire_next()
        assert a2 is None, 'Task must not be re-acquired during requeue cooldown'

    @pytest.mark.asyncio
    async def test_requeue_cooldown_expires(self, monkeypatch, task_response):
        """After cooldown expires, task should be acquirable again."""
        config = OrchestratorConfig(max_per_module=1, requeue_cooldown_secs=30.0)
        scheduler = Scheduler(config)

        mock = AsyncMock(return_value=task_response)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        # Acquire and release with requeue
        a1 = await scheduler.acquire_next()
        assert a1 is not None

        scheduler.release('99', requeued=True)

        # Fast-forward time past cooldown
        import time
        original_monotonic = time.monotonic
        offset = 31.0  # past the 30s cooldown
        monkeypatch.setattr(time, 'monotonic', lambda: original_monotonic() + offset)

        a2 = await scheduler.acquire_next()
        assert a2 is not None and a2.task_id == '99'

    @pytest.mark.asyncio
    async def test_normal_release_no_cooldown(self, monkeypatch, task_response):
        """Normal release (not requeued) should not impose a cooldown."""
        config = OrchestratorConfig(max_per_module=1, requeue_cooldown_secs=30.0)
        scheduler = Scheduler(config)

        mock = AsyncMock(return_value=task_response)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        a1 = await scheduler.acquire_next()
        assert a1 is not None

        # Normal release — no requeue flag
        scheduler.release('99')

        a2 = await scheduler.acquire_next()
        assert a2 is not None and a2.task_id == '99'


class TestFairness:
    """Scheduler anti-starvation (Mode-2 cross-module race) fairness.

    The strict top candidate's consecutive-skip counter is incremented whenever
    a lower-ranked task takes its slot (or when the full loop fails). Once the
    counter reaches ``skip_threshold``, the scheduler installs a reservation
    on each of the top candidate's normalized modules.  Reserved modules
    refuse ``try_acquire`` from everyone except the owner until the owner
    acquires or the lease expires.
    """

    # ---- ModuleLockTable park-level unit tests ----

    def test_install_and_block_non_owner(self):
        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        lt = ModuleLockTable(config)
        lt.install_parks('owner', ['backend'], deadline=time.monotonic() + 60)
        assert not lt.try_acquire('other', ['backend'])
        # Owner can still acquire its own park.
        assert lt.try_acquire('owner', ['backend'])

    def test_park_hierarchical_blocks_child(self):
        """A park on a parent module blocks acquire of any child."""
        config = OrchestratorConfig(max_per_module=1, lock_depth=4)
        lt = ModuleLockTable(config)
        lt.install_parks('A', ['autopilot/analyze'], deadline=time.monotonic() + 60)
        assert not lt.try_acquire('B', ['autopilot/analyze/asr'])

    def test_park_hierarchical_blocks_parent(self):
        """A park on a child blocks acquire of its parent."""
        config = OrchestratorConfig(max_per_module=1, lock_depth=4)
        lt = ModuleLockTable(config)
        lt.install_parks(
            'A', ['autopilot/analyze/asr'], deadline=time.monotonic() + 60
        )
        assert not lt.try_acquire('B', ['autopilot/analyze'])

    def test_park_siblings_dont_conflict(self):
        config = OrchestratorConfig(max_per_module=1, lock_depth=4)
        lt = ModuleLockTable(config)
        lt.install_parks(
            'A', ['autopilot/analyze/asr'], deadline=time.monotonic() + 60
        )
        assert lt.try_acquire('B', ['autopilot/analyze/speech'])

    def test_clear_parks_for_owner(self):
        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        lt = ModuleLockTable(config)
        lt.install_parks('A', ['backend', 'frontend'], deadline=time.monotonic() + 60)
        assert lt.has_parks('A')
        lt.clear_parks_for('A')
        assert not lt.has_parks('A')
        # Unrelated tasks can now acquire.
        assert lt.try_acquire('B', ['backend'])

    def test_prune_expired_returns_owner_and_drops(self):
        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        lt = ModuleLockTable(config)
        lt.install_parks('A', ['backend'], deadline=0.0)  # already expired
        lt.install_parks('B', ['frontend'], deadline=time.monotonic() + 60)
        evicted = lt.prune_expired_parks(time.monotonic())
        assert evicted == ['A']
        # A's park is gone, B's remains.
        assert not lt.has_parks('A')
        assert lt.has_parks('B')

    def test_expired_park_does_not_block(self):
        """An expired park (not yet pruned) must not block acquires."""
        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        lt = ModuleLockTable(config)
        lt.install_parks('A', ['backend'], deadline=0.0)
        # Lease expired; try_acquire from other should succeed without
        # explicit pruning.
        assert lt.try_acquire('B', ['backend'])

    # ---- Scheduler lease computation ----

    def test_compute_lease_midpoint_on_empty_history(self):
        config = OrchestratorConfig(max_per_module=1)
        config.fairness.lease_min_secs = 100.0
        config.fairness.lease_max_secs = 300.0
        s = Scheduler(config)
        assert s._compute_lease() == 200.0

    def test_compute_lease_uses_median_and_multiplier(self):
        config = OrchestratorConfig(max_per_module=1)
        config.fairness.lease_multiplier = 5.0
        config.fairness.lease_min_secs = 0.0
        config.fairness.lease_max_secs = 10_000.0
        s = Scheduler(config)
        s._recent_durations.extend([10.0, 20.0, 30.0])  # median 20.0
        assert s._compute_lease() == 100.0

    def test_compute_lease_clamps_to_min(self):
        config = OrchestratorConfig(max_per_module=1)
        config.fairness.lease_multiplier = 1.0
        config.fairness.lease_min_secs = 100.0
        config.fairness.lease_max_secs = 1000.0
        s = Scheduler(config)
        s._recent_durations.append(5.0)  # 5s * 1.0 = 5s, below floor
        assert s._compute_lease() == 100.0

    def test_compute_lease_clamps_to_max(self):
        config = OrchestratorConfig(max_per_module=1)
        config.fairness.lease_multiplier = 5.0
        config.fairness.lease_min_secs = 60.0
        config.fairness.lease_max_secs = 1000.0
        s = Scheduler(config)
        s._recent_durations.append(500.0)  # 500 * 5 = 2500, above ceiling
        assert s._compute_lease() == 1000.0

    # ---- Mode-2 integration: skip-count promotion ----

    @pytest.fixture
    def fair_config(self) -> OrchestratorConfig:
        """OrchestratorConfig tuned for quick fairness testing."""
        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        config.fairness.skip_threshold = 3
        config.fairness.lease_min_secs = 60.0
        config.fairness.lease_max_secs = 600.0
        return config

    @staticmethod
    def _broad_task():
        return {
            'id': 'A',
            'title': 'Broad task',
            'status': 'pending',
            'priority': 'high',
            'dependencies': [],
            'metadata': {'modules': ['compiler/src', 'eval/src']},
        }

    @staticmethod
    def _narrow_task(tid: str, module: str, priority: str = 'medium'):
        return {
            'id': tid,
            'title': f'Narrow task {tid}',
            'status': 'pending',
            'priority': priority,
            'dependencies': [],
            'metadata': {'modules': [module]},
        }

    @pytest.mark.asyncio
    async def test_skip_count_increments_when_top_passed_over(self, fair_config):
        """A (broad, top) fails, B (narrow, lower) succeeds → A's skip_count = 1."""
        scheduler = Scheduler(fair_config)
        # Seed compiler/src lock so A can't acquire (eval/src free, but broad lock fails).
        scheduler.lock_table.try_acquire('seed', ['compiler/src'])
        scheduler._dispatched.add('seed')  # seed task isn't in the candidate list

        a = self._broad_task()
        b = self._narrow_task('B', 'eval/src', priority='medium')
        scheduler.get_tasks = AsyncMock(return_value=[a, b])

        result = await scheduler.acquire_next()
        # B (lower priority, narrow) won.
        assert result is not None
        assert result.task_id == 'B'
        # A's skip counter was incremented.
        assert scheduler._skip_count.get('A') == 1

    @pytest.mark.asyncio
    async def test_skip_count_resets_on_successful_acquire(self, fair_config):
        """If the top candidate acquires, its skip counter is cleared."""
        scheduler = Scheduler(fair_config)
        scheduler._skip_count['A'] = 2
        a = self._broad_task()
        scheduler.get_tasks = AsyncMock(return_value=[a])

        result = await scheduler.acquire_next()
        assert result is not None and result.task_id == 'A'
        assert 'A' not in scheduler._skip_count

    @pytest.mark.asyncio
    async def test_reservation_installed_after_threshold(self, fair_config):
        """After skip_threshold consecutive skips, A's modules are parked."""
        scheduler = Scheduler(fair_config)
        scheduler.lock_table.try_acquire('seed', ['compiler/src'])
        scheduler._dispatched.add('seed')

        a = self._broad_task()
        b = self._narrow_task('B', 'eval/src', priority='medium')
        scheduler.get_tasks = AsyncMock(return_value=[a, b])

        # Run skip_threshold ticks. Between ticks, free up 'eval/src' (via
        # release of B) so there's a fresh acquire each time for B.
        threshold = fair_config.fairness.skip_threshold
        for _ in range(threshold):
            result = await scheduler.acquire_next()
            assert result is not None and result.task_id == 'B'
            scheduler.release('B')

        assert scheduler._skip_count['A'] == threshold
        assert scheduler.lock_table.has_parks('A')

    @pytest.mark.asyncio
    async def test_reservation_blocks_lower_ranked_tasks(self, fair_config):
        """Once A's park is installed, B can no longer take A's modules."""
        scheduler = Scheduler(fair_config)
        # Manually install a park for A on compiler/src + eval/src.
        scheduler.lock_table.install_parks(
            'A',
            ['compiler/src', 'eval/src'],
            deadline=time.monotonic() + 300,
        )
        # B wants compiler/src only — should be blocked by A's park.
        assert not scheduler.lock_table.try_acquire('B', ['compiler/src'])
        # Unrelated module is fine.
        assert scheduler.lock_table.try_acquire('C', ['other/src'])

    @pytest.mark.asyncio
    async def test_owner_acquires_despite_own_park(self, fair_config):
        """The park owner can still acquire its own reserved modules."""
        scheduler = Scheduler(fair_config)
        scheduler.lock_table.install_parks(
            'A', ['compiler/src', 'eval/src'], deadline=time.monotonic() + 300
        )
        a = self._broad_task()
        scheduler.get_tasks = AsyncMock(return_value=[a])

        result = await scheduler.acquire_next()
        assert result is not None and result.task_id == 'A'
        # Parks were cleared on successful acquire.
        assert not scheduler.lock_table.has_parks('A')

    @pytest.mark.asyncio
    async def test_reservation_expires_and_skip_count_resets(
        self, fair_config, monkeypatch
    ):
        """When a lease expires without acquire, the park drops and the owner's
        skip counter resets so they can re-accumulate instead of loop-parking."""
        scheduler = Scheduler(fair_config)
        # Install a park with a deadline already in the past.
        scheduler.lock_table.install_parks(
            'A', ['compiler/src', 'eval/src'], deadline=0.0
        )
        scheduler._skip_count['A'] = 5

        # Tick the scheduler with no candidates — this triggers prune.
        scheduler.get_tasks = AsyncMock(return_value=[])
        await scheduler.acquire_next()

        assert not scheduler.lock_table.has_parks('A')
        assert 'A' not in scheduler._skip_count

    def test_release_records_duration(self, fair_config, monkeypatch):
        """Scheduler.release() appends (end - start) to the rolling window."""
        scheduler = Scheduler(fair_config)
        # Seed as if acquire_next had recorded a start time.
        scheduler._task_start_times['A'] = 100.0
        monkeypatch.setattr(time, 'monotonic', lambda: 150.0)

        scheduler.release('A')

        assert list(scheduler._recent_durations) == [50.0]

    @pytest.mark.asyncio
    async def test_mode2_broad_task_eventually_wins(self, fair_config):
        """End-to-end Mode-2 regression guard.

        Broad high-priority A is starved by narrow medium-priority B on
        compiler/src.  After skip_threshold ticks, A's reservation parks
        compiler/src; B can no longer grab it; the next tick frees the
        seed lock on compiler/src and A runs.
        """
        scheduler = Scheduler(fair_config)

        # Seed: block compiler/src with a long-running task.
        scheduler.lock_table.try_acquire('seed', ['compiler/src'])
        scheduler._dispatched.add('seed')

        a = self._broad_task()
        b = self._narrow_task('B', 'eval/src', priority='medium')
        scheduler.get_tasks = AsyncMock(return_value=[a, b])

        threshold = fair_config.fairness.skip_threshold
        # skip_threshold ticks: B wins, A's skip counter climbs.
        for _ in range(threshold):
            result = await scheduler.acquire_next()
            assert result is not None and result.task_id == 'B'
            scheduler.release('B')

        # A's reservation is now installed.
        assert scheduler.lock_table.has_parks('A')

        # Release the seed task. Now compiler/src is free, but B is blocked
        # by A's park on it.
        scheduler.release('seed')
        scheduler._dispatched.discard('seed')

        result = await scheduler.acquire_next()
        assert result is not None and result.task_id == 'A'
        # And A's park was cleaned up on successful acquire.
        assert not scheduler.lock_table.has_parks('A')


class TestGetCachedStatus:
    """get_cached_status() exposes the internal status cache via a public method."""

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    def test_get_cached_status_returns_none_for_unknown_task(self, scheduler: Scheduler):
        """get_cached_status returns None when the task has never been seen."""
        assert scheduler.get_cached_status('unknown') is None

    @pytest.mark.asyncio
    async def test_get_cached_status_returns_status_after_set_task_status(
        self, scheduler: Scheduler, monkeypatch
    ):
        """get_cached_status returns the last status written via set_task_status."""
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', AsyncMock(return_value={}))
        await scheduler.set_task_status('42', 'in-progress')
        assert scheduler.get_cached_status('42') == 'in-progress'


class TestSchedulerInternalRouting:
    """Scheduler internal call sites route reads through get_cached_status."""

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @pytest.mark.asyncio
    async def test_set_task_status_reads_via_get_cached_status(
        self, scheduler: Scheduler, monkeypatch
    ):
        """set_task_status() must read the cached value via get_cached_status().

        Stronger invariant: the return value must actually drive the transition
        gate.  Patching get_cached_status to return 'done' causes done->blocked
        to be rejected (is_valid_transition('done', 'blocked') is False), so
        mcp_call is never invoked.  A spy-only assertion would pass even if the
        production code read _status_cache directly alongside a vestigial
        get_cached_status() call; this form does not.
        """
        mcp_mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mcp_mock)

        with patch.object(scheduler, 'get_cached_status', return_value='done'):
            await scheduler.set_task_status('42', 'blocked')
            mcp_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_tasks_reads_via_get_cached_status(
        self, scheduler: Scheduler, monkeypatch
    ):
        """get_tasks() must read the cached value via get_cached_status() for reinstatement detection."""
        import json
        scheduler._status_cache['42'] = 'cancelled'

        tasks_response = {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': json.dumps({'tasks': [{'id': '42', 'status': 'pending', 'title': 'T'}]}),
                    }
                ]
            }
        }
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', AsyncMock(return_value=tasks_response))

        with patch.object(scheduler, 'get_cached_status', wraps=scheduler.get_cached_status) as spy:
            await scheduler.get_tasks()
            spy.assert_any_call('42')

    @pytest.mark.asyncio
    async def test_set_task_status_writes_via_set_cached_status(
        self, scheduler: Scheduler, monkeypatch
    ):
        """set_task_status() must write the new status via _set_cached_status().

        Instance-level rebinding works because Python's attribute lookup finds
        the instance attribute before the class method, so the production call
        self._set_cached_status(...) resolves to the recorder.

        The read of scheduler._set_cached_status raises AttributeError until
        step 3 adds the helper — that is the TDD failing-state signal.
        """
        mcp_mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mcp_mock)

        # Pre-seed the cache so is_valid_transition('pending', 'in-progress')
        # is True regardless of future gate tightening for None->* transitions.
        scheduler._status_cache['42'] = 'pending'

        recorded: list[tuple[str, str]] = []
        original = scheduler._set_cached_status  # AttributeError until step 3

        def recorder(task_id: str, status: str) -> None:
            recorded.append((task_id, status))
            original(task_id, status)

        scheduler._set_cached_status = recorder  # type: ignore[method-assign]

        await scheduler.set_task_status('42', 'in-progress')
        assert ('42', 'in-progress') in recorded

    @pytest.mark.asyncio
    async def test_get_tasks_writes_via_set_cached_status(
        self, scheduler: Scheduler, monkeypatch
    ):
        """get_tasks() must write the cached status via _set_cached_status().

        Fails (AttributeError on recorder install or missed assertion) until
        step 5 routes the get_tasks write through the helper.
        """
        import json

        tasks_response = {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': json.dumps({'tasks': [{'id': '42', 'status': 'pending', 'title': 'T'}]}),
                    }
                ]
            }
        }
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', AsyncMock(return_value=tasks_response))

        recorded: list[tuple[str, str]] = []
        original = scheduler._set_cached_status  # AttributeError until step 3

        def recorder(task_id: str, status: str) -> None:
            recorded.append((task_id, status))
            original(task_id, status)

        scheduler._set_cached_status = recorder  # type: ignore[method-assign]

        await scheduler.get_tasks()
        assert ('42', 'pending') in recorded
