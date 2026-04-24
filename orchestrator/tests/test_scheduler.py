"""Tests for scheduler module lock logic."""


import time
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.config import (
    TIER_BASE,
    TIER_WIDTH,
    ModuleConfig,
    OrchestratorConfig,
)
from orchestrator.evals.runner import _StubMcpSession
from orchestrator.event_store import EventType
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

    def test_release_subset_drops_only_named(self, lock_table: ModuleLockTable):
        assert lock_table.try_acquire('task-1', ['backend', 'server'])
        released = lock_table.release_subset('task-1', ['backend'])
        assert released == ['backend']
        # server still held by task-1, backend now free for task-2
        assert lock_table.try_acquire('task-2', ['backend'])
        assert not lock_table.try_acquire('task-3', ['server'])

    def test_release_subset_clears_entry_when_empty(self, lock_table: ModuleLockTable):
        assert lock_table.try_acquire('task-1', ['backend'])
        released = lock_table.release_subset('task-1', ['backend'])
        assert released == ['backend']
        # Task no longer tracked in _held
        assert lock_table.is_held('task-1') is False

    def test_release_subset_ignores_unheld_modules(self, lock_table: ModuleLockTable):
        assert lock_table.try_acquire('task-1', ['backend'])
        released = lock_table.release_subset('task-1', ['server', 'frontend'])
        assert released == []
        assert lock_table.is_held('task-1') is True

    def test_release_subset_nonexistent_task_is_noop(self, lock_table: ModuleLockTable):
        assert lock_table.release_subset('never-acquired', ['backend']) == []


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


class TestGetTasksExceptionLogging:
    """get_tasks() must emit tracebacks + class names when the MCP call raises.

    Motivated by 2026-04-20 orchestrator hang where `logger.error(f'...: {e}')`
    produced bare '[Errno 2] No such file or directory' lines with no
    traceback and no exception class — leaving future investigators unable
    to locate where in the httpx / mcp_lifecycle stack the OSError originated.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @pytest.mark.asyncio
    async def test_get_tasks_logs_exception_with_traceback(
        self, scheduler: Scheduler, monkeypatch, caplog
    ):
        import logging as _logging

        raiser = AsyncMock(
            side_effect=FileNotFoundError(2, 'No such file or directory')
        )
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', raiser)

        with caplog.at_level(_logging.ERROR, logger='orchestrator.scheduler'):
            result = await scheduler.get_tasks()

        # get_tasks() still swallows and returns an empty list (so the
        # scheduler tick continues), but the log line now carries traceback
        # info AND the exception class name so diagnostics are possible.
        assert result == []
        assert 'Failed to fetch tasks' in caplog.text
        assert 'FileNotFoundError' in caplog.text
        # logger.exception automatically appends 'Traceback (most recent call last):'
        assert 'Traceback' in caplog.text


class TestParseToolTextResultWarning:
    """_parse_tool_text_result must emit a WARNING when JSON parsing fails.

    A malformed text block should still return None (preserving existing
    contract) but must log a WARNING containing a ≤200-char prefix of the
    offending text so operators can identify the source of bad output.
    """

    def test_invalid_json_returns_none_and_logs_warning(self, caplog):
        import logging as _logging

        # Build a long invalid-JSON payload so we can validate truncation.
        bad_text = 'not valid json payload ' * 30  # 23*30 = 690 chars
        result_envelope = {
            'result': {
                'content': [
                    {'type': 'text', 'text': bad_text},
                ]
            }
        }

        with caplog.at_level(_logging.WARNING, logger='orchestrator.scheduler'):
            value = Scheduler._parse_tool_text_result(result_envelope, 'tasks')

        # (1) Return contract is preserved.
        assert value is None

        # (2) A WARNING is emitted.
        warnings = [r for r in caplog.records if r.levelno == _logging.WARNING]
        assert warnings, f'Expected a WARNING log. Records: {[r.message for r in caplog.records]}'

        warning_text = ' '.join(r.getMessage() for r in warnings)

        # (3) The log contains a truncated prefix of the offending text.
        truncated_prefix = bad_text[:200]
        assert truncated_prefix in warning_text, (
            f'Expected log to contain truncated text prefix. Got: {warning_text}'
        )

        # (4) The full original text is NOT present in the log (validates truncation).
        assert bad_text not in warning_text, (
            'Full (690-char) text must NOT appear in log — only the truncated ≤200-char prefix.'
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


class TestGetStatus:
    """``Scheduler.get_status`` returns the fresh store value via MCP."""

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @pytest.mark.asyncio
    async def test_get_status_returns_store_value(
        self, scheduler: Scheduler, monkeypatch
    ):
        """get_status parses the MCP get_task response and returns the status field."""
        import json
        response = {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': json.dumps({'id': '42', 'status': 'done', 'title': 'T'}),
                    }
                ]
            }
        }
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=response),
        )
        assert await scheduler.get_status('42') == 'done'

    @pytest.mark.asyncio
    async def test_get_status_unwraps_data_envelope(
        self, scheduler: Scheduler, monkeypatch
    ):
        """Taskmaster's ``{'data': {...}}`` envelope is unwrapped."""
        import json
        response = {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': json.dumps(
                            {'data': {'id': '42', 'status': 'in-progress'}},
                        ),
                    }
                ]
            }
        }
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=response),
        )
        assert await scheduler.get_status('42') == 'in-progress'

    @pytest.mark.asyncio
    async def test_get_status_returns_none_on_mcp_exception(
        self, scheduler: Scheduler, monkeypatch
    ):
        """MCP failures bubble up as ``None`` — callers treat that as stall-retry."""
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(side_effect=OSError(2, 'No such file')),
        )
        assert await scheduler.get_status('42') is None


class TestSetTaskStatusForwarding:
    """``Scheduler.set_task_status`` is a thin forwarder; server owns the FSM."""

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @pytest.mark.asyncio
    async def test_forwards_done_provenance(
        self, scheduler: Scheduler, monkeypatch
    ):
        """done_provenance kwarg reaches the MCP arguments dict."""
        mcp_mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mcp_mock)

        await scheduler.set_task_status('1', 'done', done_provenance={'commit': 'abc123'})

        mcp_mock.assert_called_once()
        arguments = mcp_mock.call_args[0][2]['arguments']
        assert arguments.get('done_provenance') == {'commit': 'abc123'}

    @pytest.mark.asyncio
    async def test_omits_done_provenance_when_absent(
        self, scheduler: Scheduler, monkeypatch
    ):
        """No done_provenance key when the caller didn't pass one."""
        mcp_mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mcp_mock)

        await scheduler.set_task_status('1', 'in-progress')

        mcp_mock.assert_called_once()
        arguments = mcp_mock.call_args[0][2]['arguments']
        assert 'done_provenance' not in arguments

    @pytest.mark.asyncio
    async def test_forwards_reopen_reason(
        self, scheduler: Scheduler, monkeypatch
    ):
        """reopen_reason kwarg reaches the MCP arguments dict — for un-defer scripts."""
        mcp_mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mcp_mock)

        await scheduler.set_task_status(
            '1', 'pending', reopen_reason='un-defer script',
        )

        mcp_mock.assert_called_once()
        arguments = mcp_mock.call_args[0][2]['arguments']
        assert arguments.get('reopen_reason') == 'un-defer script'

    @pytest.mark.asyncio
    async def test_omits_reopen_reason_when_absent(
        self, scheduler: Scheduler, monkeypatch
    ):
        mcp_mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mcp_mock)

        await scheduler.set_task_status('1', 'in-progress')

        mcp_mock.assert_called_once()
        arguments = mcp_mock.call_args[0][2]['arguments']
        assert 'reopen_reason' not in arguments

    @pytest.mark.asyncio
    async def test_mcp_exception_is_swallowed_and_logged(
        self, scheduler: Scheduler, monkeypatch, caplog
    ):
        """MCP failures are logged but don't raise — scheduler ticks survive."""
        import logging as _logging
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(side_effect=OSError(2, 'No such file')),
        )
        with caplog.at_level(_logging.ERROR, logger='orchestrator.scheduler'):
            await scheduler.set_task_status('1', 'in-progress')
        assert any(
            'Failed to set task 1 status' in rec.message
            for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# Value/h scoring: priority inheritance (P1), age boost (P2), CPM weight (P3),
# per-tier slot caps (Fix 3), per-tier skip thresholds (Fix 2).
# ---------------------------------------------------------------------------


def _pending_task(
    task_id: str,
    *,
    priority: str = 'medium',
    deps: list[str] | None = None,
    modules: list[str] | None = None,
    status: str = 'pending',
) -> dict:
    """Helper: build a task dict with all fields the scheduler reads."""
    return {
        'id': task_id,
        'title': f'Task {task_id}',
        'status': status,
        'priority': priority,
        'dependencies': [{'id': d} for d in (deps or [])],
        'metadata': {'modules': modules or [f'mod{task_id}']},
    }


class _RecordingEventStore:
    """Minimal EventStore stand-in capturing emit() calls in-memory."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def emit(self, event_type, *, task_id=None, phase=None, role=None,
             data=None, cost_usd=None, duration_ms=None):
        self.events.append((
            str(event_type),
            {
                'task_id': task_id,
                'data': dict(data or {}),
            },
        ))


class TestPriorityInheritance:
    """P1: effective_priority walks dependents and inherits the max rank."""

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        return Scheduler(OrchestratorConfig(max_per_module=1))

    def test_effective_priority_inherits_from_dependent(self, scheduler: Scheduler):
        """A medium task with a critical dependent scores as critical."""
        base = _pending_task('10', priority='medium')
        consumer = _pending_task('11', priority='critical', deps=['10'])
        tasks = [base, consumer]
        by_id = {t['id']: t for t in tasks}
        rev = scheduler._build_reverse_index(tasks)
        status_map = {t['id']: t['status'] for t in tasks}
        eff = scheduler._compute_effective_priorities(by_id, rev, status_map)
        assert eff['10'] == 'critical'
        assert eff['11'] == 'critical'

    def test_effective_priority_ignores_done_dependents(self, scheduler: Scheduler):
        """A done descendant must not lift the ancestor's priority."""
        base = _pending_task('10', priority='medium')
        consumer = _pending_task('11', priority='critical', deps=['10'],
                                 status='done')
        tasks = [base, consumer]
        by_id = {t['id']: t for t in tasks}
        rev = scheduler._build_reverse_index(tasks)
        status_map = {t['id']: t['status'] for t in tasks}
        eff = scheduler._compute_effective_priorities(by_id, rev, status_map)
        assert eff['10'] == 'medium'

    def test_effective_priority_cycle_safe(self, scheduler: Scheduler, caplog):
        """A self-cycle must not crash and must log a WARN."""
        import logging

        cyclic = _pending_task('10', priority='high', deps=['10'])
        tasks = [cyclic]
        by_id = {t['id']: t for t in tasks}
        rev = scheduler._build_reverse_index(tasks)
        status_map = {t['id']: t['status'] for t in tasks}
        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduler'):
            eff = scheduler._compute_effective_priorities(by_id, rev, status_map)
        assert eff['10'] == 'high'
        assert any('cycle' in rec.message for rec in caplog.records)

    def test_unknown_priority_treated_as_medium(self, scheduler: Scheduler):
        """A string we don't recognise coerces to the default tier."""
        weird = _pending_task('10', priority='weird')
        tasks = [weird]
        by_id = {t['id']: t for t in tasks}
        rev = scheduler._build_reverse_index(tasks)
        status_map = {t['id']: t['status'] for t in tasks}
        eff = scheduler._compute_effective_priorities(by_id, rev, status_map)
        assert eff['10'] == 'medium'


class TestTransitiveDependents:
    """P3: BFS over the reverse-dependency graph, no double-count."""

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        return Scheduler(OrchestratorConfig(max_per_module=1))

    def test_transitive_linear(self, scheduler: Scheduler):
        """A -> B -> C: A has 2 undone descendants."""
        tasks = [
            _pending_task('A'),
            _pending_task('B', deps=['A']),
            _pending_task('C', deps=['B']),
        ]
        by_id = {t['id']: t for t in tasks}
        rev = scheduler._build_reverse_index(tasks)
        status_map = {t['id']: t['status'] for t in tasks}
        counts = scheduler._compute_transitive_counts(by_id, rev, status_map)
        assert counts['A'] == 2
        assert counts['B'] == 1
        assert counts['C'] == 0

    def test_transitive_diamond_no_double_count(self, scheduler: Scheduler):
        """Diamond A -> B, A -> C, B -> D, C -> D: A has 3 undone descendants."""
        tasks = [
            _pending_task('A'),
            _pending_task('B', deps=['A']),
            _pending_task('C', deps=['A']),
            _pending_task('D', deps=['B', 'C']),
        ]
        by_id = {t['id']: t for t in tasks}
        rev = scheduler._build_reverse_index(tasks)
        status_map = {t['id']: t['status'] for t in tasks}
        counts = scheduler._compute_transitive_counts(by_id, rev, status_map)
        # B, C, D — each counted once.
        assert counts['A'] == 3


class TestScoreFunction:
    """P2/P3: compute_score — tier base dominant, bonuses bounded."""

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        return Scheduler(OrchestratorConfig(max_per_module=1))

    def test_tier_base_dominates(self, scheduler: Scheduler):
        """A fresh medium task with no descendants scores = TIER_BASE[medium]."""
        score = scheduler._compute_score('medium', age=0, transitive_count=0)
        assert score == float(TIER_BASE['medium'])

    def test_age_bonus_bounded_by_tier_width(self, scheduler: Scheduler):
        """age=1e6 + medium tier must never outscore a fresh high tier."""
        aged_medium = scheduler._compute_score('medium', age=1_000_000, transitive_count=0)
        fresh_high = scheduler._compute_score('high', age=0, transitive_count=0)
        assert aged_medium < fresh_high
        # Verify the cap: score - base never exceeds TIER_WIDTH - 1.
        bonus = aged_medium - TIER_BASE['medium']
        assert bonus <= TIER_WIDTH - 1

    def test_cpm_bonus_positive(self, scheduler: Scheduler):
        """A task with many descendants scores higher than one without."""
        alone = scheduler._compute_score('medium', age=0, transitive_count=0)
        unlock_many = scheduler._compute_score('medium', age=0, transitive_count=1000)
        assert unlock_many > alone
        # Still bounded below the next tier.
        assert unlock_many < TIER_BASE['high']

    def test_combined_bonus_bounded(self, scheduler: Scheduler):
        """Age + CPM together never cross a tier boundary."""
        huge = scheduler._compute_score('low', age=10_000, transitive_count=10_000)
        assert huge < TIER_BASE['medium']


class TestAgeAnchor:
    """Age anchor resets on cancellation → pending resurrection."""

    def test_cancelled_resurrection_no_age_jump(self):
        """A previously-cancelled task re-pended scores no higher than brand-new medium."""
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        # Tick 1: task 1 is cancelled, task 100 is pending.
        tasks_tick1 = [
            _pending_task('1', status='cancelled'),
            _pending_task('100', status='pending'),
        ]
        scheduler._update_age_anchors(tasks_tick1, max_id=100)
        age_100_t1 = scheduler._compute_age('100', max_id=100)
        # Tick 2: task 1 is reinstated to pending, task 100 still pending.
        tasks_tick2 = [
            _pending_task('1', status='pending'),
            _pending_task('100', status='pending'),
        ]
        scheduler._update_age_anchors(tasks_tick2, max_id=100)
        age_1 = scheduler._compute_age('1', max_id=100)
        age_100 = scheduler._compute_age('100', max_id=100)
        # Resurrected 1 must not leapfrog brand-new 100.
        assert age_1 <= age_100
        # Brand-new-pending baseline is 0.
        assert age_100_t1 == 0
        assert age_100 == 0
        assert age_1 == 0

    def test_old_pending_accumulates_age(self):
        """Continuously-pending old tasks accumulate age from their creation id."""
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        # First-ever tick sees task 5 as pending — anchor to task_id.
        scheduler._update_age_anchors([_pending_task('5')], max_id=100)
        age = scheduler._compute_age('5', max_id=100)
        assert age == 95


class TestTierSlotCaps:
    """Fix 3: per-tier slot caps reserve headroom for higher-value work."""

    def _config(self, caps: dict[str, float]) -> OrchestratorConfig:
        return OrchestratorConfig(
            max_concurrent_tasks=10,
            max_per_module=1,
            tier_slot_caps=caps,
        )

    def test_allowed_by_cap_under_limit(self):
        s = Scheduler(self._config({'low': 0.5}))
        # 0 dispatched → low allowed (limit = 5).
        assert s._allowed_by_tier_cap('low') is True

    def test_blocked_at_limit(self):
        s = Scheduler(self._config({'low': 0.5}))
        for i in range(5):  # fill 5 low slots
            s._dispatched_priority[f't{i}'] = 'low'
        assert s._allowed_by_tier_cap('low') is False
        # Higher-priority tier still has room (cap 1.0).
        assert s._allowed_by_tier_cap('high') is True

    def test_lower_counts_toward_higher_budget(self):
        """A 'low' slot counts against medium's cap too (at-or-below semantics)."""
        s = Scheduler(self._config({'medium': 0.3}))  # 3 medium-or-below slots
        for i in range(3):
            s._dispatched_priority[f't{i}'] = 'low'
        # Medium is blocked because low slots exhaust the medium-or-below budget.
        assert s._allowed_by_tier_cap('medium') is False
        # Critical/high remain unaffected (cap 1.0).
        assert s._allowed_by_tier_cap('high') is True
        assert s._allowed_by_tier_cap('critical') is True

    @pytest.mark.asyncio
    async def test_cap_emits_idle_event(self):
        """If all candidates are cap-rejected, emit exactly one idle event."""
        # Cap medium-or-below at 0 so even one medium candidate is rejected.
        config = OrchestratorConfig(
            max_concurrent_tasks=4,
            max_per_module=1,
            tier_slot_caps={'medium': 0.0, 'low': 0.0, 'polish': 0.0},
        )
        event_store = _RecordingEventStore()
        scheduler = Scheduler(config, event_store=event_store)  # type: ignore[arg-type]
        tasks = [_pending_task('1', priority='medium')]
        scheduler.get_tasks = AsyncMock(return_value=tasks)
        result = await scheduler.acquire_next()
        assert result is None
        idle_events = [e for e in event_store.events
                       if e[0] == EventType.scheduler_tier_cap_idle.value]
        assert len(idle_events) == 1
        assert idle_events[0][1]['data']['candidates_skipped_by_cap'] == 1

    @pytest.mark.asyncio
    async def test_park_overrides_tier_cap(self):
        """A parked task dispatches even when its tier cap would reject."""
        config = OrchestratorConfig(
            max_concurrent_tasks=4,
            max_per_module=1,
            tier_slot_caps={'low': 0.0, 'polish': 0.0},  # 0 low slots
        )
        scheduler = Scheduler(config)
        low_task = _pending_task('1', priority='low')
        # Install a park for task 1 — this should let it through the cap gate.
        scheduler.lock_table.install_parks(
            '1', ['mod1'], deadline=time.monotonic() + 300,
        )
        scheduler.get_tasks = AsyncMock(return_value=[low_task])
        result = await scheduler.acquire_next()
        assert result is not None and result.task_id == '1'


class TestPerTierSkipThreshold:
    """Fix 2: skip_threshold dict unlocks per-tier parking behaviour."""

    def _config(self, thresholds: dict[str, int]) -> OrchestratorConfig:
        config = OrchestratorConfig(max_per_module=1)
        config.fairness.skip_threshold = thresholds
        return config

    def test_skip_threshold_for_lookup(self):
        config = self._config({'critical': 1, 'high': 2, 'medium': 6,
                               'low': 9999, 'polish': 9999})
        assert config.fairness.skip_threshold_for('critical') == 1
        assert config.fairness.skip_threshold_for('high') == 2
        assert config.fairness.skip_threshold_for('medium') == 6
        assert config.fairness.skip_threshold_for('low') == 9999

    def test_skip_threshold_int_legacy(self):
        """int skip_threshold still works — applies to every tier."""
        config = OrchestratorConfig(max_per_module=1)
        config.fairness.skip_threshold = 4
        assert config.fairness.skip_threshold_for('critical') == 4
        assert config.fairness.skip_threshold_for('polish') == 4

    def test_critical_parks_after_one_skip(self):
        """With threshold=1, a single skip is enough to install a park."""
        config = self._config({'critical': 1, 'high': 2, 'medium': 6,
                               'low': 9999, 'polish': 9999})
        event_store = _RecordingEventStore()
        scheduler = Scheduler(config, event_store=event_store)  # type: ignore[arg-type]
        scheduler._bump_skip_and_maybe_park('A', ['mod'], tier='critical')
        assert scheduler.lock_table.has_parks('A')

    def test_low_never_parks_even_after_many_skips(self):
        """With low=9999, parking is effectively disabled."""
        config = self._config({'critical': 1, 'high': 2, 'medium': 6,
                               'low': 9999, 'polish': 9999})
        scheduler = Scheduler(config)
        for _ in range(50):
            scheduler._bump_skip_and_maybe_park('A', ['mod'], tier='low')
        assert not scheduler.lock_table.has_parks('A')

    def test_task_skipped_rate_limit_for_inf_threshold(self):
        """With threshold>=1000, task_skipped only emits at geometric counts."""
        config = self._config({'critical': 1, 'high': 2, 'medium': 6,
                               'low': 9999, 'polish': 9999})
        event_store = _RecordingEventStore()
        scheduler = Scheduler(config, event_store=event_store)  # type: ignore[arg-type]
        for _ in range(150):
            scheduler._bump_skip_and_maybe_park('A', ['mod'], tier='low')
        skip_events = [e for e in event_store.events
                       if e[0] == EventType.task_skipped.value]
        # Only counts 1, 10, 100 should have emitted.
        counts = [e[1]['data']['skip_count'] for e in skip_events]
        assert counts == [1, 10, 100]

    def test_task_skipped_no_rate_limit_for_finite_threshold(self):
        """With finite threshold, every skip emits an event."""
        config = self._config({'critical': 1, 'high': 2, 'medium': 6,
                               'low': 9999, 'polish': 9999})
        event_store = _RecordingEventStore()
        scheduler = Scheduler(config, event_store=event_store)  # type: ignore[arg-type]
        for _ in range(3):
            scheduler._bump_skip_and_maybe_park('A', ['mod'], tier='medium')
        skip_events = [e for e in event_store.events
                       if e[0] == EventType.task_skipped.value]
        assert len(skip_events) == 3


class TestLeaseMultiplierPerTier:
    """Fix 2: lease_multiplier dict unlocks per-tier lease duration."""

    def test_lease_multiplier_for_lookup(self):
        config = OrchestratorConfig(max_per_module=1)
        config.fairness.lease_multiplier = {
            'critical': 8.0, 'high': 8.0, 'medium': 5.0,
            'low': 2.0, 'polish': 2.0,
        }
        assert config.fairness.lease_multiplier_for('critical') == 8.0
        assert config.fairness.lease_multiplier_for('low') == 2.0

    def test_compute_lease_uses_tier_multiplier(self):
        config = OrchestratorConfig(max_per_module=1)
        config.fairness.lease_multiplier = {
            'critical': 8.0, 'high': 8.0, 'medium': 5.0,
            'low': 2.0, 'polish': 2.0,
        }
        config.fairness.lease_min_secs = 0.0
        config.fairness.lease_max_secs = 10_000.0
        scheduler = Scheduler(config)
        scheduler._recent_durations.extend([10.0])  # median = 10
        assert scheduler._compute_lease(tier='critical') == 80.0
        assert scheduler._compute_lease(tier='low') == 20.0


class TestLegacyOrderingPreserved:
    """With 3-tier data + default tier_slot_caps all 1.0 + no CPM + zero age,
    the new scheduler must match the legacy high>medium>low dispatch order.
    """

    @pytest.mark.asyncio
    async def test_legacy_three_tier_ordering_preserved(self):
        config = OrchestratorConfig(max_per_module=1, max_concurrent_tasks=10)
        # Disable caps and fairness carve-outs for this test.
        config.tier_slot_caps = {}
        scheduler = Scheduler(config)
        tasks = [
            _pending_task('1', priority='low', modules=['modA']),
            _pending_task('2', priority='high', modules=['modB']),
            _pending_task('3', priority='medium', modules=['modC']),
        ]
        scheduler.get_tasks = AsyncMock(return_value=tasks)
        result = await scheduler.acquire_next()
        assert result is not None and result.task_id == '2', 'high wins'

    @pytest.mark.asyncio
    async def test_critical_beats_high(self):
        """New 5-tier: critical outranks high."""
        config = OrchestratorConfig(max_per_module=1, max_concurrent_tasks=10)
        config.tier_slot_caps = {}
        scheduler = Scheduler(config)
        tasks = [
            _pending_task('1', priority='high', modules=['modA']),
            _pending_task('2', priority='critical', modules=['modB']),
        ]
        scheduler.get_tasks = AsyncMock(return_value=tasks)
        result = await scheduler.acquire_next()
        assert result is not None and result.task_id == '2'

    @pytest.mark.asyncio
    async def test_polish_loses_to_low(self):
        """New 5-tier: polish ranks below low."""
        config = OrchestratorConfig(max_per_module=1, max_concurrent_tasks=10)
        config.tier_slot_caps = {}
        scheduler = Scheduler(config)
        tasks = [
            _pending_task('1', priority='polish', modules=['modA']),
            _pending_task('2', priority='low', modules=['modB']),
        ]
        scheduler.get_tasks = AsyncMock(return_value=tasks)
        result = await scheduler.acquire_next()
        assert result is not None and result.task_id == '2'

    @pytest.mark.asyncio
    async def test_inheritance_lifts_dependency(self):
        """A medium task with a critical dependent is dispatched first."""
        config = OrchestratorConfig(max_per_module=1, max_concurrent_tasks=10)
        config.tier_slot_caps = {}
        scheduler = Scheduler(config)
        # Task 1 (medium, available) is needed by task 2 (critical, blocked).
        # Inheritance should lift task 1 above task 3 (high, available).
        tasks = [
            _pending_task('1', priority='medium', modules=['modA']),
            _pending_task('2', priority='critical', deps=['1'], modules=['modB']),
            _pending_task('3', priority='high', modules=['modC']),
        ]
        scheduler.get_tasks = AsyncMock(return_value=tasks)
        result = await scheduler.acquire_next()
        assert result is not None and result.task_id == '1'


class TestDispatchPriorityBookkeeping:
    """_dispatched_priority must be updated on acquire AND release."""

    @pytest.mark.asyncio
    async def test_release_clears_dispatched_priority(self):
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        task = _pending_task('1', priority='high', modules=['modA'])
        scheduler.get_tasks = AsyncMock(return_value=[task])
        result = await scheduler.acquire_next()
        assert result is not None
        assert scheduler._dispatched_priority['1'] == 'high'
        scheduler.release('1')
        assert '1' not in scheduler._dispatched_priority

    @pytest.mark.asyncio
    async def test_dispatched_priority_tracks_effective_not_own(self):
        """dispatched_priority reflects effective (inherited) priority."""
        config = OrchestratorConfig(max_per_module=1, max_concurrent_tasks=10)
        config.tier_slot_caps = {}
        scheduler = Scheduler(config)
        tasks = [
            _pending_task('1', priority='medium', modules=['modA']),
            _pending_task('2', priority='critical', deps=['1'], modules=['modB']),
        ]
        scheduler.get_tasks = AsyncMock(return_value=tasks)
        result = await scheduler.acquire_next()
        assert result is not None and result.task_id == '1'
        # Task 1 was dispatched as critical (inherited from dependent).
        assert scheduler._dispatched_priority['1'] == 'critical'


class TestBlastRadiusRefinement:
    """handle_blast_radius_expansion must treat the plan's file list as a
    replacement, not a union: acquire new modules AND release stale ones so
    other tasks aren't starved behind a lock the refined plan no longer needs.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1, lock_depth=4)
        event_store = _RecordingEventStore()
        sched = Scheduler(config, event_store=event_store)  # type: ignore[arg-type]
        return sched

    @pytest.mark.asyncio
    async def test_narrowing_releases_stale(self, scheduler: Scheduler):
        """Plan scope narrows to a sibling file — the initial lock is freed."""
        lt = scheduler.lock_table
        assert lt.try_acquire('936', ['crates/reify-compiler/src/lib.rs'])
        ok = await scheduler.handle_blast_radius_expansion(
            '936',
            current=['crates/reify-compiler/src/lib.rs'],
            needed=['crates/reify-compiler/src/conformance.rs'],
        )
        assert ok is True
        # lib.rs is free for another task
        assert lt.try_acquire('2035', ['crates/reify-compiler/src/lib.rs'])
        # 936 now holds conformance.rs, not lib.rs
        assert not lt.try_acquire('9999', ['crates/reify-compiler/src/conformance.rs'])
        # Event emitted with plan_refinement reason
        event_store = scheduler.event_store
        assert event_store is not None
        released_events = [
            e for e in event_store.events  # type: ignore[attr-defined]
            if 'lock_released' in e[0]
            and e[1]['data'].get('reason') == 'plan_refinement'
        ]
        assert len(released_events) == 1
        assert released_events[0][1]['task_id'] == '936'
        assert released_events[0][1]['data']['modules'] == [
            'crates/reify-compiler/src/lib.rs',
        ]

    @pytest.mark.asyncio
    async def test_shift_releases_and_acquires(self, scheduler: Scheduler):
        """Plan refines to a mixed set: acquire new, release stale."""
        lt = scheduler.lock_table
        assert lt.try_acquire('936', ['crates/reify-compiler/src/lib.rs'])
        ok = await scheduler.handle_blast_radius_expansion(
            '936',
            current=['crates/reify-compiler/src/lib.rs'],
            needed=[
                'crates/reify-compiler/src/conformance.rs',
                'crates/reify-compiler/tests/trait_conformance_tests.rs',
            ],
        )
        assert ok is True
        held = lt._held['936']
        assert held == {
            'crates/reify-compiler/src/conformance.rs',
            'crates/reify-compiler/tests/trait_conformance_tests.rs',
        }
        assert lt.try_acquire('2035', ['crates/reify-compiler/src/lib.rs'])

    @pytest.mark.asyncio
    async def test_pure_expansion_keeps_current(self, scheduler: Scheduler):
        """Regression: when needed is a superset of current, held grows to
        match needed and no spurious lock_released event fires."""
        lt = scheduler.lock_table
        assert lt.try_acquire('T', ['a/lib.rs'])
        ok = await scheduler.handle_blast_radius_expansion(
            'T',
            current=['a/lib.rs'],
            needed=['a/lib.rs', 'a/other.rs'],
        )
        assert ok is True
        assert lt._held['T'] == {'a/lib.rs', 'a/other.rs'}
        event_store = scheduler.event_store
        assert event_store is not None
        released_events = [
            e for e in event_store.events  # type: ignore[attr-defined]
            if 'lock_released' in e[0]
        ]
        assert released_events == []

    @pytest.mark.asyncio
    async def test_same_set_is_noop(self, scheduler: Scheduler):
        """needed == current: return True without mutating _held or emitting."""
        lt = scheduler.lock_table
        assert lt.try_acquire('T', ['a/lib.rs', 'a/other.rs'])
        ok = await scheduler.handle_blast_radius_expansion(
            'T',
            current=['a/lib.rs', 'a/other.rs'],
            needed=['a/other.rs', 'a/lib.rs'],  # order differs, set equal
        )
        assert ok is True
        assert lt._held['T'] == {'a/lib.rs', 'a/other.rs'}
        event_store = scheduler.event_store
        assert event_store is not None
        assert event_store.events == []  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_acquire_failure_preserves_stale(self, scheduler: Scheduler):
        """If additions conflict with another task, return False without
        touching _held — caller falls through to the full-release requeue path.
        """
        lt = scheduler.lock_table
        assert lt.try_acquire('936', ['crates/reify-compiler/src/lib.rs'])
        # Another task grabs the module 936 would expand into
        assert lt.try_acquire(
            'other', ['crates/reify-compiler/src/conformance.rs']
        )
        # 936 can't expand; since update_task → set_task_status hit MCP,
        # patch scheduler methods to no-ops for this assertion scope.
        scheduler.update_task = AsyncMock(return_value=True)  # type: ignore[method-assign]
        scheduler.set_task_status = AsyncMock(return_value=None)  # type: ignore[method-assign]
        ok = await scheduler.handle_blast_radius_expansion(
            '936',
            current=['crates/reify-compiler/src/lib.rs'],
            needed=['crates/reify-compiler/src/conformance.rs'],
        )
        assert ok is False
        # Full release ran: 936 should no longer hold anything
        assert '936' not in lt._held


class TestSchedulerMcpSessionDI:
    """Tests for the optional mcp_session dependency-injection kwarg on Scheduler.

    Each test injects a _StubMcpSession and monkeypatches orchestrator.scheduler.mcp_call
    to raise AssertionError — proving the HTTP transport is never contacted when a
    session is injected.
    """

    @pytest.mark.asyncio
    async def test_set_task_status_routes_through_stub(self):
        """set_task_status writes to the stub, not to the HTTP mcp_call path."""
        stub = _StubMcpSession()
        cfg = OrchestratorConfig()
        sched = Scheduler(cfg, mcp_session=stub)

        with patch(
            'orchestrator.scheduler.mcp_call',
            new=AsyncMock(side_effect=AssertionError('HTTP path must not be used when mcp_session is injected')),
        ):
            await sched.set_task_status('42', 'in-progress')

        assert stub._statuses['42'] == 'in-progress'

    @pytest.mark.asyncio
    async def test_get_status_round_trips_through_stub(self):
        """get_status reads from the stub after a prior set_task_status."""
        stub = _StubMcpSession()
        cfg = OrchestratorConfig()
        sched = Scheduler(cfg, mcp_session=stub)
        no_http = AsyncMock(side_effect=AssertionError('HTTP path must not be used when mcp_session is injected'))

        with patch('orchestrator.scheduler.mcp_call', new=no_http):
            await sched.set_task_status('77', 'done')
            result = await sched.get_status('77')
            unknown = await sched.get_status('unknown-id')

        assert result == 'done'
        assert unknown is None
        no_http.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_tasks_routes_through_stub(self):
        """get_tasks returns [] from the stub without calling mcp_call."""
        stub = _StubMcpSession()
        cfg = OrchestratorConfig()
        sched = Scheduler(cfg, mcp_session=stub)
        no_http = AsyncMock(side_effect=AssertionError('HTTP path must not be used when mcp_session is injected'))

        with patch('orchestrator.scheduler.mcp_call', new=no_http):
            tasks = await sched.get_tasks()

        assert tasks == []
        no_http.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_task_routes_through_stub(self):
        """update_task returns True (non-error envelope) via the stub without calling mcp_call."""
        stub = _StubMcpSession()
        cfg = OrchestratorConfig()
        sched = Scheduler(cfg, mcp_session=stub)
        no_http = AsyncMock(side_effect=AssertionError('HTTP path must not be used when mcp_session is injected'))

        with patch('orchestrator.scheduler.mcp_call', new=no_http):
            ok = await sched.update_task('77', {'modules': ['foo']})

        assert ok is True
        no_http.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_task_status_accepts_done_provenance(self):
        """set_task_status with done_provenance passes through the stub without error."""
        stub = _StubMcpSession()
        cfg = OrchestratorConfig()
        sched = Scheduler(cfg, mcp_session=stub)

        with patch(
            'orchestrator.scheduler.mcp_call',
            new=AsyncMock(side_effect=AssertionError('HTTP path must not be used when mcp_session is injected')),
        ):
            await sched.set_task_status('42', 'done', done_provenance={'commit': 'abc123'})

        assert stub._statuses['42'] == 'done'

    @pytest.mark.asyncio
    async def test_set_task_status_accepts_reopen_reason(self):
        """set_task_status with reopen_reason passes through the stub without error."""
        stub = _StubMcpSession()
        cfg = OrchestratorConfig()
        sched = Scheduler(cfg, mcp_session=stub)

        with patch(
            'orchestrator.scheduler.mcp_call',
            new=AsyncMock(side_effect=AssertionError('HTTP path must not be used when mcp_session is injected')),
        ):
            await sched.set_task_status('42', 'pending', reopen_reason='un-defer script')

        assert stub._statuses['42'] == 'pending'


class TestGetStatuses:
    """``Scheduler.get_statuses`` returns a ``(statuses, error)`` tuple via MCP."""

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @pytest.mark.asyncio
    async def test_get_statuses_returns_parsed_mapping(
        self, scheduler: Scheduler, monkeypatch
    ):
        """get_statuses parses the MCP response and returns (statuses_dict, None)."""
        import json
        response = {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': json.dumps({'statuses': {'1': 'done', '2': 'pending'}}),
                    }
                ]
            }
        }
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=response),
        )
        assert await scheduler.get_statuses() == ({'1': 'done', '2': 'pending'}, None)

    @pytest.mark.asyncio
    async def test_get_statuses_passes_ids_argument(
        self, scheduler: Scheduler, monkeypatch
    ):
        """When ids=['1','2'] is passed, mcp_call arguments include ids=['1','2']."""
        import json
        mcp_mock = AsyncMock(return_value={
            'result': {
                'content': [
                    {'type': 'text', 'text': json.dumps({'statuses': {'1': 'done'}})}
                ]
            }
        })
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mcp_mock)

        statuses, err = await scheduler.get_statuses(ids=['1', '2'])

        assert err is None
        mcp_mock.assert_called_once()
        arguments = mcp_mock.call_args[0][2]['arguments']
        assert arguments.get('ids') == ['1', '2']
        assert 'project_root' in arguments

    @pytest.mark.asyncio
    async def test_get_statuses_exception_returns_empty_dict(
        self, scheduler: Scheduler, monkeypatch
    ):
        """OSError from mcp_call returns ({}, OSError) tuple."""
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(side_effect=OSError(2, 'No such file')),
        )
        result, err = await scheduler.get_statuses()
        assert result == {}
        assert isinstance(err, OSError)
        assert err.errno == 2

    @pytest.mark.asyncio
    async def test_get_statuses_routes_through_stub(self):
        """When mcp_session is injected, get_statuses uses the stub (not HTTP mcp_call)."""
        stub = _StubMcpSession()
        cfg = OrchestratorConfig()
        sched = Scheduler(cfg, mcp_session=stub)
        no_http = AsyncMock(
            side_effect=AssertionError('HTTP path must not be used when mcp_session is injected')
        )

        with patch('orchestrator.scheduler.mcp_call', new=no_http):
            result, err = await sched.get_statuses()

        assert isinstance(result, dict)
        assert err is None
        no_http.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_statuses_returns_exception_on_failure_and_none_on_success(
        self, scheduler: Scheduler, monkeypatch
    ):
        """Failing call returns ({}, OSError); subsequent success returns (dict, None).

        (a) After a failing call (OSError): returns ({}, OSError) with correct errno.
        (b) After a subsequent successful call: returns (dict, None) — no cross-call state.
        """
        import json

        # (a) Transport failure: error returned in tuple.
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(side_effect=OSError(2, 'No such file')),
        )
        result_fail, err_fail = await scheduler.get_statuses()
        assert result_fail == {}
        assert isinstance(err_fail, OSError)
        assert err_fail.errno == 2

        # (b) Subsequent success: None error, no cross-call state leakage.
        success_response = {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': json.dumps({'statuses': {'1': 'pending'}}),
                    }
                ]
            }
        }
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=success_response),
        )
        result_ok, err_ok = await scheduler.get_statuses()
        assert result_ok == {'1': 'pending'}
        assert err_ok is None

    @pytest.mark.asyncio
    async def test_get_statuses_returns_fresh_exception_per_call(
        self, scheduler: Scheduler, monkeypatch
    ):
        """Two consecutive failing calls each return their own distinct exception.

        Validates that error state lives on the stack (no cross-call leakage via
        a shared attribute): each call's error is independent.
        """
        # First failing call: OSError.
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(side_effect=OSError(2, 'No such file')),
        )
        _result1, err1 = await scheduler.get_statuses()
        assert isinstance(err1, OSError)

        # Second failing call: ValueError — independent from first.
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(side_effect=ValueError('malformed response')),
        )
        result2, err2 = await scheduler.get_statuses()
        assert result2 == {}
        assert isinstance(err2, ValueError)
        assert 'malformed response' in str(err2)
        # err1 must be unchanged (no side-channel mutation)
        assert isinstance(err1, OSError)

    def test_scheduler_has_no_last_get_statuses_error_attribute(self):
        """Regression guard: the _last_get_statuses_error side-channel is gone.

        Neither the private underscore name nor the public property may exist on
        a freshly constructed Scheduler — future callers must use the tuple return.
        """
        sched = Scheduler(OrchestratorConfig())
        assert not hasattr(sched, 'last_get_statuses_error'), (
            'last_get_statuses_error property must be removed'
        )
        assert not hasattr(sched, '_last_get_statuses_error'), (
            '_last_get_statuses_error attribute must be removed'
        )
