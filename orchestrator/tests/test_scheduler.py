"""Tests for scheduler module lock logic."""


from unittest.mock import AsyncMock

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
