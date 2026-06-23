"""Tests for scheduler module lock logic."""


import asyncio
import time
from datetime import UTC
from unittest.mock import AsyncMock, patch

import pytest
from _recording_event_store import _RecordingEventStore

from orchestrator.config import (
    TIER_BASE,
    TIER_WIDTH,
    FairnessConfig,
    ModuleConfig,
    OrchestratorConfig,
    StarvationWatchdogConfig,
)
from orchestrator.evals.runner import _StubMcpSession
from orchestrator.event_store import EventType
from orchestrator.scheduler import (
    ExternalResolverError,
    ModuleLockTable,
    Scheduler,
    files_to_modules,
)
from orchestrator.task_status import ACTIVE_TASK_STATUSES


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

    def test_try_acquire_additional_creates_entry_when_absent(
        self, lock_table: ModuleLockTable
    ):
        # Call try_acquire_additional for a task that has never called try_acquire.
        # Before the fix, line 656 (`self._held[task_id].update(...)`) raises KeyError
        # because task-new is absent from _held.  After the fix (setdefault), it
        # creates the entry and returns True.
        result = lock_table.try_acquire_additional('task-new', ['backend'])
        assert result is True
        assert lock_table.is_held('task-new') is True

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


class TestGetTasksNormalizesMetadata:
    """get_tasks() must coerce ``task['metadata']`` to a dict at the boundary.

    The fused-memory wire format may surface metadata as a JSON string,
    a dict, ``None``, or an unparseable string. Downstream consumers
    (briefing._format_task, _get_modules, workflow no-plan/infra-thrash
    counters) all read dict-keyed sub-fields. Normalizing once here lets
    every consumer assume ``isinstance(task['metadata'], dict)``.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @staticmethod
    def _envelope(tasks: list[dict]) -> dict:
        import json as _json
        return {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': _json.dumps({'tasks': tasks}),
                    }
                ]
            }
        }

    @pytest.mark.parametrize(
        'raw_metadata, expected',
        [
            ('{"foo": 1}', {'foo': 1}),
            ({'foo': 1}, {'foo': 1}),
            (None, {}),
            ('not-json', {}),
            ('"just-a-string"', {}),
            ('[1,2,3]', {}),
        ],
        ids=[
            'json-string',
            'dict-passthrough',
            'absent',
            'invalid-string',
            'parses-to-string',
            'parses-to-list',
        ],
    )
    @pytest.mark.asyncio
    async def test_metadata_is_normalized_to_dict(
        self, scheduler: Scheduler, monkeypatch, raw_metadata, expected
    ):
        task = {'id': '1', 'metadata': raw_metadata}
        # Some envelopes intentionally omit metadata entirely (None case
        # also exercises the absent-key path through .get()).
        if raw_metadata is None:
            task.pop('metadata')

        envelope = self._envelope([task])
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=envelope),
        )

        tasks = await scheduler.get_tasks()

        assert len(tasks) == 1
        assert isinstance(tasks[0]['metadata'], dict)
        assert tasks[0]['metadata'] == expected

    @pytest.mark.asyncio
    async def test_normalize_is_in_place(self, scheduler: Scheduler, monkeypatch):
        """All tasks in a multi-task response are normalized."""
        envelope = self._envelope([
            {'id': '1', 'metadata': '{"a": 1}'},
            {'id': '2', 'metadata': {'b': 2}},
            {'id': '3', 'metadata': 'garbage'},
            {'id': '4'},
        ])
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=envelope),
        )

        tasks = await scheduler.get_tasks()

        assert [t['metadata'] for t in tasks] == [
            {'a': 1},
            {'b': 2},
            {},
            {},
        ]

    def test_normalize_helper_directly(self):
        """Unit-level coverage of the static helper."""
        cases = [
            ({'metadata': '{"foo": 1}'}, {'foo': 1}),
            ({'metadata': {'foo': 1}}, {'foo': 1}),
            ({'metadata': None}, {}),
            ({'metadata': 'not-json'}, {}),
            ({'metadata': '"just-a-string"'}, {}),
            ({'metadata': '[1,2,3]'}, {}),
            ({}, {}),
        ]
        for task, expected in cases:
            Scheduler._normalize_task_metadata(task)
            assert task['metadata'] == expected, f'failed for input: {task}'


# ---------------------------------------------------------------------------
# TestGetTasksAndGetStatusFailsLoud (task 1807 — step-3 RED / step-4 GREEN)
#
# Replaces TestParseToolTextResultWarning: resolver-level loud tests that
# drive get_tasks / get_status directly and assert WARNINGs emitted by the
# shared.mcp_envelope primitive (not from orchestrator.scheduler directly).
# ---------------------------------------------------------------------------

class TestGetTasksAndGetStatusFailsLoud:
    """``get_tasks`` and ``get_status`` must emit loud WARNINGs on malformed envelopes.

    After migrating to ``parse_tool_result``, the WARNING logger is
    ``shared.mcp_envelope``, not ``orchestrator.scheduler``.

    Fails today:
    - ``get_tasks`` non-list 'tasks' branch returns ``[]`` silently (no WARNING).
    - ``get_tasks`` unparseable-JSON WARNING comes from ``orchestrator.scheduler``
      not ``shared.mcp_envelope``.
    - ``get_status`` non-str 'status' branch returns ``None`` silently (no WARNING).
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @staticmethod
    def _envelope(payload: dict) -> dict:
        import json as _json
        return {
            'result': {
                'content': [{'type': 'text', 'text': _json.dumps(payload)}]
            }
        }

    @staticmethod
    def _bad_json_envelope(text: str) -> dict:
        return {
            'result': {
                'content': [{'type': 'text', 'text': text}]
            }
        }

    # --- get_tasks ---

    @pytest.mark.asyncio
    async def test_get_tasks_non_list_emits_warning_from_shared_envelope(
        self, scheduler: Scheduler, monkeypatch, caplog
    ):
        """get_tasks fed a non-list 'tasks' value emits a WARNING from shared.mcp_envelope.

        Fails today: the non-list branch is silent (no WARNING, just returns []).
        """
        import logging

        # 'tasks' is a dict, not a list.
        response = self._envelope({'tasks': {'id': '1'}})
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=response),
        )

        with caplog.at_level(logging.WARNING):
            result = await scheduler.get_tasks()

        assert result == [], f'Expected [] on non-list; got {result!r}'
        mcp_envelope_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and r.name == 'shared.mcp_envelope'
        ]
        assert mcp_envelope_warnings, (
            f'Expected a WARNING from shared.mcp_envelope; '
            f'got records={[(r.name, r.getMessage()) for r in caplog.records]!r}'
        )

    @pytest.mark.asyncio
    async def test_get_tasks_unparseable_json_emits_warning_from_shared_envelope(
        self, scheduler: Scheduler, monkeypatch, caplog
    ):
        """get_tasks fed unparseable JSON emits a WARNING from shared.mcp_envelope.

        Fails today: the WARNING comes from orchestrator.scheduler (via
        _parse_tool_text_result), not from shared.mcp_envelope.
        """
        import logging

        bad_text = 'not valid json payload ' * 30  # 690 chars
        response = self._bad_json_envelope(bad_text)
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=response),
        )

        with caplog.at_level(logging.WARNING):
            result = await scheduler.get_tasks()

        assert result == [], f'Expected [] on unparseable JSON; got {result!r}'
        mcp_envelope_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and r.name == 'shared.mcp_envelope'
        ]
        assert mcp_envelope_warnings, (
            f'Expected a WARNING from shared.mcp_envelope; '
            f'got records={[(r.name, r.getMessage()) for r in caplog.records]!r}'
        )

    # --- get_status ---

    @pytest.mark.asyncio
    async def test_get_status_non_str_emits_warning(
        self, scheduler: Scheduler, monkeypatch, caplog
    ):
        """get_status fed a non-str 'status' value emits a WARNING and returns None.

        Fails today: the non-str branch is silent (no WARNING).
        """
        import logging

        # 'status' is a dict, not a str.
        response = self._envelope({'status': {'value': 'pending'}})
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=response),
        )

        with caplog.at_level(logging.WARNING):
            result = await scheduler.get_status('42')

        assert result is None, f'Expected None on non-str; got {result!r}'
        assert any(
            r.levelno >= logging.WARNING for r in caplog.records
        ), f'Expected a WARNING; got {caplog.records!r}'

    @pytest.mark.asyncio
    async def test_get_status_absent_key_emits_warning(
        self, scheduler: Scheduler, monkeypatch, caplog
    ):
        """get_status fed a response with no 'status' key emits a WARNING and returns None.

        Fails today: the absent-key branch is silent (no WARNING).
        """
        import logging

        # No 'status' key.
        response = self._envelope({'task': {'id': '42'}})
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=response),
        )

        with caplog.at_level(logging.WARNING):
            result = await scheduler.get_status('42')

        assert result is None
        assert any(r.levelno >= logging.WARNING for r in caplog.records)


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
            'metadata': {'files': ['backend']},
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
            'metadata': {'files': ['backend']},
        }
        task_b = {
            'id': '2',
            'title': 'Task B',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['frontend']},
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
            'metadata': {'files': ['backend']},
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
                'metadata': {'files': ['backend']},
            },
            {
                'id': '2',
                'title': 'Frontend task',
                'status': 'pending',
                'dependencies': [],
                'metadata': {'files': ['frontend']},
            },
            {
                'id': '3',
                'title': 'Infra task',
                'status': 'pending',
                'dependencies': [],
                'metadata': {'files': ['infra']},
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
        self
    ):
        """Two tasks on the same module: dispatch A, B blocked; release A, B dispatches.

        Updated for α strip: the fixture previously used 'backend' (an
        extension-less directory entry) which is now stripped by the α filter,
        making both tasks fall through to distinct task-<id> fallbacks that
        don't conflict with each other.  Replaced with co-located real files
        that normalize to the same depth-2 module ('backend/src') so the
        lock conflict is preserved.  lock_depth=2 is set explicitly because
        the default may be overridden by a config.yaml or env var in this env.
        """
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1, lock_depth=2))
        task_a = {
            'id': 'A',
            'title': 'Task A',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['backend/src/app.py']},
        }
        task_b = {
            'id': 'B',
            'title': 'Task B',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['backend/src/routes.py']},
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

    def test_deps_satisfied_returns_true_when_dep_cancelled(
        self, scheduler: Scheduler
    ):
        """_deps_satisfied treats cancelled deps as satisfied alongside done.

        ``cancelled`` represents an obsolete or duplicate task; the dependent
        should re-architect rather than wait indefinitely.
        """
        task = {'id': '2', 'dependencies': [{'id': 1}]}
        status_map = {'1': 'cancelled', '2': 'pending'}
        assert scheduler._deps_satisfied(task, status_map) is True

    def test_deps_satisfied_returns_false_for_blocked_dep(
        self, scheduler: Scheduler
    ):
        """``blocked`` is non-terminal — must still gate dispatch."""
        task = {'id': '2', 'dependencies': [{'id': 1}]}
        status_map = {'1': 'blocked', '2': 'pending'}
        assert scheduler._deps_satisfied(task, status_map) is False

    def test_deps_satisfied_returns_false_for_deferred_dep(
        self, scheduler: Scheduler
    ):
        """``deferred`` is non-terminal — must still gate dispatch."""
        task = {'id': '2', 'dependencies': [{'id': 1}]}
        status_map = {'1': 'deferred', '2': 'pending'}
        assert scheduler._deps_satisfied(task, status_map) is False


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
            'metadata': {'files': ['backend']},
        }
        task_b = {
            'id': 'B',
            'title': 'Task B',
            'status': 'pending',
            'dependencies': [{'id': 'A'}],
            'metadata': {'files': ['frontend']},
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
            'metadata': {'files': ['backend']},
        }
        task_b = {
            'id': 'B',
            'title': 'Task B',
            'status': 'pending',
            'dependencies': [{'id': 'A'}],
            'metadata': {'files': ['frontend']},
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
            'metadata': {'files': ['backend']},
        }
        task_b = {
            'id': 'B',
            'title': 'Task B',
            'status': 'pending',
            'dependencies': [{'id': 'A'}],
            'metadata': {'files': ['frontend']},
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
            'metadata': {'files': ['backend']},
        }
        task_b = {
            'id': 'B',
            'title': 'Task B',
            'status': 'in-progress',
            'dependencies': [],
            'metadata': {'files': ['backend']},
        }
        task_c = {
            'id': 'C',
            'title': 'Task C',
            'status': 'pending',
            'dependencies': [{'id': 'A'}, {'id': 'B'}],
            'metadata': {'files': ['frontend']},
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
            'metadata': {'files': ['backend']},
        }
        # Task with string dep ID
        task_str = {
            'id': '11',
            'title': 'Task str dep',
            'status': 'pending',
            'dependencies': ['1'],  # string format
            'metadata': {'files': ['frontend']},
        }
        # Task with dict dep ID
        task_dict = {
            'id': '12',
            'title': 'Task dict dep',
            'status': 'pending',
            'dependencies': [{'id': 1}],  # dict format
            'metadata': {'files': ['ops']},
        }
        dep_done = {
            'id': '1',
            'title': 'Dep task',
            'status': 'done',
            'dependencies': [],
            'metadata': {'files': []},
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
    """_get_modules consumes the post-normalization invariant.

    Scheduler.get_tasks coerces ``task['metadata']`` to a dict before any
    consumer sees it (see TestGetTasksNormalizesMetadata). _get_modules is
    therefore expected to receive a dict and degrade gracefully to a
    task-<id> fallback if it ever receives a non-dict (malformed test
    fixture, eval-mode bypass, etc.).
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    def test_get_modules_extracts_modules_from_dict_metadata(
        self, scheduler: Scheduler
    ):
        """_get_modules returns normalized module list from real file paths in dict metadata.

        Updated for α strip: the fixture previously used extension-less paths
        ('backend', 'server') which are now classified as directory entries and
        stripped, falling through to the task-<id> fallback.  The test is
        updated to use real file paths so it still exercises the
        files-→-modules derivation path.
        """
        task = {
            'id': '5',
            'metadata': {'files': ['backend/app.py', 'server/main.py']},
        }
        result = scheduler._get_modules(task)
        assert result != ['task-5'], (
            f'Expected modules from dict metadata, got fallback: {result}'
        )
        assert len(result) > 0
        assert all(isinstance(m, str) for m in result)
        assert 'task-5' not in result

    def test_get_modules_ignores_legacy_metadata_modules_key(
        self, scheduler: Scheduler
    ):
        """_get_modules must NOT read metadata.modules — only metadata.files.

        Regression guard for the principled rename: any task that still carries
        the legacy ``metadata.modules`` key (e.g. submitted before the migration
        ran) must fall through to the ``task-<id>`` fallback, not silently
        derive locks from the stale key.
        """
        task = {
            'id': '99',
            'metadata': {'modules': ['legacy-key-should-not-be-read']},
        }
        result = scheduler._get_modules(task)
        assert result == ['task-99']

    def test_get_modules_extracts_files_from_dict_metadata(
        self, scheduler: Scheduler
    ):
        """_get_modules returns file-derived modules from dict metadata."""
        task = {
            'id': '6',
            'metadata': {'files': ['src/server/app.py', 'src/server/routes.py']},
        }
        result = scheduler._get_modules(task)
        assert result != ['task-6'], (
            f'Expected file-derived modules from dict metadata, got fallback: {result}'
        )
        assert len(result) > 0
        assert 'task-6' not in result

    def test_get_modules_falls_back_on_string_metadata(
        self, scheduler: Scheduler
    ):
        """_get_modules degrades to task-<id> when handed non-dict metadata.

        After the boundary normalizer landed in get_tasks, _get_modules
        should never receive a string in production. If it does (e.g. a
        test synthesizes one), the isinstance guard kicks in and we fall
        back rather than crash.
        """
        task = {
            'id': '7',
            'metadata': 'not valid json',
        }
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

        await scheduler.update_task('1', {'files': ['backend']})

        assert len(captured_args) == 1
        arguments = captured_args[0]['arguments']
        metadata = arguments['metadata']
        # Must be a string, not a dict
        assert isinstance(metadata, str), f'Expected str metadata, got {type(metadata)}: {metadata}'
        # Must be valid JSON that round-trips correctly
        parsed = json.loads(metadata)
        assert parsed == {'files': ['backend']}

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

    @pytest.mark.asyncio
    async def test_update_task_append_true_forwards_additive_mode(
        self, scheduler: Scheduler, monkeypatch
    ):
        """update_task(append=True) must forward metadata_mode='additive', not 'append'.

        append=True is the legacy additive-mode shorthand.  The wrapper resolves it
        to metadata_mode='additive' and forwards that on the wire; it must NOT
        forward 'append' (which the #1827 backend still accepts as a shim but which
        the wrapper no longer relies on).
        """
        captured_args: list[dict] = []

        async def mock_mcp_call(url, method, payload, **kwargs):
            captured_args.append(payload)
            return {}

        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock_mcp_call)

        await scheduler.update_task('1', {'branch_base_sha': 'a' * 40}, append=True)

        assert len(captured_args) == 1
        arguments = captured_args[0]['arguments']
        assert arguments.get('metadata_mode') == 'additive', (
            f"Expected metadata_mode='additive' in MCP arguments, got: {arguments}"
        )
        assert 'append' not in arguments, (
            f"'append' key must not be forwarded on the wire; got: {arguments}"
        )

    @pytest.mark.asyncio
    async def test_update_task_default_forwards_merge_mode(
        self, scheduler: Scheduler, monkeypatch
    ):
        """Default update_task call must forward metadata_mode='merge' (the #4271 fix).

        No-append callers (prd-tagger, module-tagger, auto-eval back-link, …) rely
        on merge — shallow last-write-wins that preserves sibling keys — NOT
        full-replacement.  This test locks in the new default-merge contract and
        ensures 'append' is absent from the wire call.
        """
        captured_args: list[dict] = []

        async def mock_mcp_call(url, method, payload, **kwargs):
            captured_args.append(payload)
            return {}

        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock_mcp_call)

        await scheduler.update_task('1', {'files': ['backend']})

        assert len(captured_args) == 1
        arguments = captured_args[0]['arguments']
        assert arguments.get('metadata_mode') == 'merge', (
            f"Default call must forward metadata_mode='merge'; got: {arguments}"
        )
        assert 'append' not in arguments, (
            f"'append' key must not appear in the wire call; got: {arguments}"
        )

    @pytest.mark.asyncio
    async def test_update_task_explicit_replace_mode(
        self, scheduler: Scheduler, monkeypatch
    ):
        """update_task(metadata_mode='replace') forwards metadata_mode='replace'."""
        captured_args: list[dict] = []

        async def mock_mcp_call(url, method, payload, **kwargs):
            captured_args.append(payload)
            return {}

        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock_mcp_call)

        await scheduler.update_task('1', {'files': ['backend']}, metadata_mode='replace')

        assert len(captured_args) == 1
        arguments = captured_args[0]['arguments']
        assert arguments.get('metadata_mode') == 'replace', (
            f"Expected metadata_mode='replace', got: {arguments}"
        )
        assert 'append' not in arguments

    @pytest.mark.asyncio
    async def test_update_task_explicit_additive_mode_passthrough(
        self, scheduler: Scheduler, monkeypatch
    ):
        """update_task(metadata_mode='additive') forwards 'additive' directly."""
        captured_args: list[dict] = []

        async def mock_mcp_call(url, method, payload, **kwargs):
            captured_args.append(payload)
            return {}

        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock_mcp_call)

        await scheduler.update_task('1', {'files': ['backend']}, metadata_mode='additive')

        assert len(captured_args) == 1
        arguments = captured_args[0]['arguments']
        assert arguments.get('metadata_mode') == 'additive', (
            f"Expected metadata_mode='additive', got: {arguments}"
        )
        assert 'append' not in arguments

    @pytest.mark.asyncio
    async def test_update_task_metadata_mode_wins_over_append(
        self, scheduler: Scheduler, monkeypatch
    ):
        """Explicit metadata_mode beats append=True (metadata_mode > append precedence).

        If both append=True and metadata_mode='merge' are supplied, the explicit
        metadata_mode='merge' must win — mirroring the backend _resolve_metadata_mode
        precedence: metadata_mode > append > default.
        """
        captured_args: list[dict] = []

        async def mock_mcp_call(url, method, payload, **kwargs):
            captured_args.append(payload)
            return {}

        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock_mcp_call)

        await scheduler.update_task(
            '1', {'files': ['backend']}, append=True, metadata_mode='merge'
        )

        assert len(captured_args) == 1
        arguments = captured_args[0]['arguments']
        assert arguments.get('metadata_mode') == 'merge', (
            f"Explicit metadata_mode='merge' must win over append=True; got: {arguments}"
        )
        assert 'append' not in arguments


class TestRequeueCooldown:
    """Tests for the requeue cooldown that prevents ghost loops."""

    @pytest.fixture
    def pending_task(self):
        return {
            'id': '99',
            'title': 'Cooldown test task',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['backend']},
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


class TestDispatchCooldownConfig:
    """Validate the dispatch_cooldown_secs config field defaults and constraints."""

    def test_default_dispatch_cooldown_secs(self):
        """OrchestratorConfig() defaults dispatch_cooldown_secs to 1800.0."""
        config = OrchestratorConfig()
        assert config.dispatch_cooldown_secs == 1800.0

    def test_accepts_value_above_floor(self):
        """OrchestratorConfig(dispatch_cooldown_secs=600.0) is valid (above 300.0 floor)."""
        config = OrchestratorConfig(dispatch_cooldown_secs=600.0)
        assert config.dispatch_cooldown_secs == 600.0

    def test_rejects_value_below_floor(self):
        """OrchestratorConfig(dispatch_cooldown_secs=120.0) raises ValidationError."""
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            OrchestratorConfig(dispatch_cooldown_secs=120.0)


class TestDispatchCooldownGate:
    """Tests for the per-task dispatch cooldown gate that prevents immediate
    re-grab after reconciliation reset or steward clear."""

    def _make_task_response(self, task: dict) -> dict:
        import json as _json
        return {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': '{"tasks": [' + _json.dumps(task) + ']}',
                    }
                ]
            }
        }

    def _pending_task_with(self, metadata: dict) -> dict:
        return {
            'id': '99',
            'title': 'Dispatch cooldown test task',
            'status': 'pending',
            'dependencies': [],
            'metadata': metadata,
        }

    @pytest.mark.asyncio
    async def test_recon_reset_gt_1_blocks_immediate_redispatch(self, monkeypatch):
        """recon_reset_count > 1 arms the dispatch cooldown gate.

        First acquire succeeds (gate not yet armed).  After normal release,
        second acquire must be blocked because the task has recon_reset_count=2.
        """
        task = self._pending_task_with({'files': ['backend'], 'recon_reset_count': 2})
        task_response = self._make_task_response(task)

        config = OrchestratorConfig(max_per_module=1, dispatch_cooldown_secs=1800.0)
        scheduler = Scheduler(config)

        mock = AsyncMock(return_value=task_response)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        # First acquire — gate not yet armed
        a1 = await scheduler.acquire_next()
        assert a1 is not None and a1.task_id == '99', 'Initial dispatch must succeed'

        # Normal release (not requeued)
        scheduler.release('99')

        # Second acquire immediately — gate must block (recon_reset_count=2)
        a2 = await scheduler.acquire_next()
        assert a2 is None, (
            'Task with recon_reset_count=2 must not be re-dispatched during cooldown'
        )

    @pytest.mark.asyncio
    async def test_recon_reset_eq_1_does_not_block(self, monkeypatch):
        """recon_reset_count=1 (first reset) must NOT trigger the cooldown gate.

        Only counts > 1 indicate a repeated reset loop.
        """
        task = self._pending_task_with({'files': ['backend'], 'recon_reset_count': 1})
        task_response = self._make_task_response(task)

        config = OrchestratorConfig(max_per_module=1, dispatch_cooldown_secs=1800.0)
        scheduler = Scheduler(config)

        mock = AsyncMock(return_value=task_response)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        a1 = await scheduler.acquire_next()
        assert a1 is not None and a1.task_id == '99'

        scheduler.release('99')

        # Second acquire must succeed — only the first reset, no loop yet
        a2 = await scheduler.acquire_next()
        assert a2 is not None and a2.task_id == '99', (
            'recon_reset_count=1 must not block re-dispatch'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('metadata,signal', [
        (
            {'files': ['backend'], 'steward_clear_at': '2026-04-27T13:04:06Z'},
            'steward_clear_at',
        ),
        (
            {'files': ['backend'], 'recon_stage2_blocked_at': '2026-04-27T13:04:06Z'},
            'recon_stage2_blocked_at',
        ),
        (
            {'files': ['backend'], 'reopen_reason': 'steward stash-pop resolution'},
            'reopen_reason',
        ),
    ])
    async def test_steward_signals_block_immediate_redispatch(
        self, monkeypatch, metadata, signal
    ):
        """Steward signals (steward_clear_at, recon_stage2_blocked_at,
        reopen_reason containing 'steward') arm the dispatch cooldown gate."""
        task = self._pending_task_with(metadata)
        task_response = self._make_task_response(task)

        config = OrchestratorConfig(max_per_module=1, dispatch_cooldown_secs=1800.0)
        scheduler = Scheduler(config)

        mock = AsyncMock(return_value=task_response)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        a1 = await scheduler.acquire_next()
        assert a1 is not None and a1.task_id == '99', 'Initial dispatch must succeed'

        scheduler.release('99')

        a2 = await scheduler.acquire_next()
        assert a2 is None, (
            f'Task with signal {signal!r} must not be re-dispatched during cooldown'
        )

    @pytest.mark.asyncio
    async def test_non_steward_reopen_reason_does_not_block(self, monkeypatch):
        """reopen_reason without 'steward' substring must NOT trigger the gate.

        Only the literal substring 'steward' arms the gate; other reopen reasons
        such as 'un-defer script' are orchestrator-internal and must dispatch normally.
        """
        task = self._pending_task_with(
            {'files': ['backend'], 'reopen_reason': 'un-defer script'}
        )
        task_response = self._make_task_response(task)

        config = OrchestratorConfig(max_per_module=1, dispatch_cooldown_secs=1800.0)
        scheduler = Scheduler(config)

        mock = AsyncMock(return_value=task_response)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        a1 = await scheduler.acquire_next()
        assert a1 is not None and a1.task_id == '99'

        scheduler.release('99')

        # Must succeed — 'un-defer script' does not contain 'steward'
        a2 = await scheduler.acquire_next()
        assert a2 is not None and a2.task_id == '99', (
            "reopen_reason='un-defer script' must not block re-dispatch"
        )

    @pytest.mark.asyncio
    async def test_dispatch_cooldown_expires_after_window(self, monkeypatch):
        """Cooldown gate expires exactly at the configured window edge (strict <).

        After 601s with a 600s window the gate is open; after 599s it is still closed.
        Uses an injected fake clock so only the Scheduler/ModuleLockTable see the
        advanced time — asyncio internals are unaffected.
        """
        t: list[float] = [1000.0]  # mutable clock cell; advance by assigning t[0]

        def fake_clock() -> float:
            return t[0]

        task = self._pending_task_with({'files': ['backend'], 'recon_reset_count': 2})
        task_response = self._make_task_response(task)

        config = OrchestratorConfig(max_per_module=1, dispatch_cooldown_secs=600.0)
        scheduler = Scheduler(config, time_source=fake_clock)

        mock = AsyncMock(return_value=task_response)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        # First dispatch — arms the gate at t=1000.0
        a1 = await scheduler.acquire_next()
        assert a1 is not None and a1.task_id == '99'
        scheduler.release('99')

        # --- just inside window (+599s): gate still active ---
        t[0] = 1000.0 + 599.0
        a_inside = await scheduler.acquire_next()
        assert a_inside is None, 'Gate must still be active at +599s (inside 600s window)'

        # --- just past window (+601s): gate must be open ---
        t[0] = 1000.0 + 601.0
        a_outside = await scheduler.acquire_next()
        assert a_outside is not None and a_outside.task_id == '99', (
            'Gate must be open at +601s (past 600s window)'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('terminal_status', ['done', 'cancelled'])
    async def test_terminal_status_clears_last_dispatch_at(
        self, monkeypatch, terminal_status
    ):
        """When acquire_next observes a task in done/cancelled, _last_dispatch_at
        must be cleared for that task so a future re-dispatch starts clean."""
        import json as _json

        task = {
            'id': '42',
            'title': 'Terminal sweep test',
            'status': terminal_status,
            'dependencies': [],
            'metadata': {},
        }
        task_response = {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': '{"tasks": [' + _json.dumps(task) + ']}',
                    }
                ]
            }
        }

        config = OrchestratorConfig(max_per_module=1)
        scheduler = Scheduler(config)

        # Prime _last_dispatch_at as if task '42' was previously dispatched
        scheduler._last_dispatch_at['42'] = time.monotonic()

        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call', AsyncMock(return_value=task_response)
        )

        # acquire_next returns None (task is terminal, not pending)
        result = await scheduler.acquire_next()
        assert result is None

        # _last_dispatch_at must be cleared after observing the terminal status
        assert '42' not in scheduler._last_dispatch_at, (
            f'_last_dispatch_at must be cleared when task is {terminal_status!r}'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('terminal_status', ['done', 'cancelled'])
    async def test_terminal_status_clears_skip_count_and_module_cache(
        self, monkeypatch, terminal_status
    ):
        """When acquire_next observes a task in done/cancelled, both _skip_count
        and _module_cache must be evicted for that task so a future re-dispatch or
        id-reuse starts from a clean slate."""
        import json as _json

        task = {
            'id': '42',
            'title': 'Terminal sweep test',
            'status': terminal_status,
            'dependencies': [],
            'metadata': {},
        }
        task_response = {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': '{"tasks": [' + _json.dumps(task) + ']}',
                    }
                ]
            }
        }

        config = OrchestratorConfig(max_per_module=1)
        scheduler = Scheduler(config)

        # Prime both dicts as if task '42' has been scheduled before
        scheduler._skip_count['42'] = 5
        scheduler._module_cache['42'] = ['somemod']

        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call', AsyncMock(return_value=task_response)
        )

        # acquire_next returns None (task is terminal, not pending)
        result = await scheduler.acquire_next()
        assert result is None

        # Both caches must be cleared after observing the terminal status
        assert '42' not in scheduler._skip_count, (
            f'_skip_count must be cleared when task is {terminal_status!r}'
        )
        assert '42' not in scheduler._module_cache, (
            f'_module_cache must be cleared when task is {terminal_status!r}'
        )

    @pytest.mark.asyncio
    async def test_in_progress_status_preserves_last_dispatch_at(self, monkeypatch):
        """In-progress status must NOT clear _last_dispatch_at (only terminal clears it)."""
        import json as _json

        task = {
            'id': '42',
            'title': 'In-progress sweep test',
            'status': 'in-progress',
            'dependencies': [],
            'metadata': {},
        }
        task_response = {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': '{"tasks": [' + _json.dumps(task) + ']}',
                    }
                ]
            }
        }

        config = OrchestratorConfig(max_per_module=1)
        scheduler = Scheduler(config)

        original_ts = time.monotonic()
        scheduler._last_dispatch_at['42'] = original_ts

        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call', AsyncMock(return_value=task_response)
        )

        await scheduler.acquire_next()

        # Entry must still be present for in-progress tasks
        assert '42' in scheduler._last_dispatch_at, (
            '_last_dispatch_at must NOT be cleared for in-progress tasks'
        )

    @pytest.mark.asyncio
    async def test_dispatch_cooldown_skip_logged(self, monkeypatch, caplog):
        """When the dispatch cooldown gate suppresses a re-dispatch, an INFO
        log record must be emitted containing: 'cooldown', the task id, the
        signal label, and a remaining-time number."""
        import logging

        task = self._pending_task_with({'files': ['backend'], 'recon_reset_count': 2})
        task_response = self._make_task_response(task)

        config = OrchestratorConfig(max_per_module=1, dispatch_cooldown_secs=1800.0)
        scheduler = Scheduler(config)

        mock = AsyncMock(return_value=task_response)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        # First acquire — gate not yet armed
        a1 = await scheduler.acquire_next()
        assert a1 is not None and a1.task_id == '99'
        scheduler.release('99')

        # Suppressed second acquire — gate active
        with caplog.at_level(logging.INFO, logger='orchestrator.scheduler'):
            a2 = await scheduler.acquire_next()

        assert a2 is None, 'Gate must have suppressed the second dispatch'

        # Verify the log record
        cooldown_records = [
            r for r in caplog.records
            if 'cooldown' in r.getMessage().lower()
            or 'suppressed' in r.getMessage().lower()
        ]
        assert cooldown_records, 'Expected at least one log record mentioning cooldown'

        log_text = cooldown_records[0].getMessage()
        assert '99' in log_text, f'Task id "99" missing from log: {log_text!r}'
        # Check the field-qualified form "signal=recon_reset_count" so a
        # regression that selected a different signal label would fail here.
        assert 'signal=recon_reset_count' in log_text, (
            f'"signal=recon_reset_count" missing from log: {log_text!r}'
        )
        # remaining time should be a number — check any digit appears in the message
        import re
        assert re.search(r'\d+', log_text), (
            f'Remaining time number missing from log: {log_text!r}'
        )

    @pytest.mark.asyncio
    async def test_signal_free_dispatch_does_not_arm_cooldown_gate(self, monkeypatch):
        """A task dispatched with no cooldown signal must NOT arm _last_dispatch_at.

        Only tasks carrying a steward/reconciliation signal (recon_reset_count>1,
        steward_clear_at, recon_stage2_blocked_at, or steward reopen_reason) should
        arm the gate.  Signal-free dispatches must leave _last_dispatch_at empty so
        the dict doesn't accumulate stale entries for tasks removed without ever
        reaching a terminal status.
        """
        task = self._pending_task_with({'files': ['backend']})
        task_response = self._make_task_response(task)

        config = OrchestratorConfig(max_per_module=1, dispatch_cooldown_secs=1800.0)
        scheduler = Scheduler(config)

        mock = AsyncMock(return_value=task_response)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        # Dispatch the task — no steward/recon signal in metadata
        a1 = await scheduler.acquire_next()
        assert a1 is not None and a1.task_id == '99', 'Initial dispatch must succeed'

        # Gate must NOT be armed for signal-free tasks
        assert '99' not in scheduler._last_dispatch_at, (
            '_last_dispatch_at must not be set for signal-free dispatches'
        )

    @pytest.mark.asyncio
    async def test_signal_bearing_dispatch_arms_gate_and_suppresses_redispatch(self, monkeypatch):
        """Signal-bearing dispatch (recon_reset_count≥2) arms _last_dispatch_at via
        _dispatch_cooldown_signal, and the armed gate suppresses re-dispatch within
        the cooldown window.
        """
        task = self._pending_task_with({'files': ['backend'], 'recon_reset_count': 2})
        task_response = self._make_task_response(task)

        config = OrchestratorConfig(max_per_module=1, dispatch_cooldown_secs=1800.0)
        scheduler = Scheduler(config)

        mock = AsyncMock(return_value=task_response)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        # Dispatch the task — recon_reset_count=2 is a gate-arming signal
        a1 = await scheduler.acquire_next()
        assert a1 is not None and a1.task_id == '99', 'Initial dispatch must succeed'

        # Gate MUST be armed for signal-bearing tasks (direct positive assertion)
        assert '99' in scheduler._last_dispatch_at, (
            '_last_dispatch_at must be set for signal-bearing dispatches '
            '(recon_reset_count=2)'
        )

        # End-to-end suppression: follow-up acquire within cooldown window must return None
        scheduler.release('99')
        a2 = await scheduler.acquire_next()
        assert a2 is None, (
            'Follow-up acquire_next within cooldown window must be suppressed '
            'end-to-end for signal-bearing tasks'
        )

    @pytest.mark.asyncio
    async def test_cooldown_log_suppressed_when_deps_unsatisfied(self, monkeypatch, caplog):
        """Cooldown-suppression INFO log must NOT fire for deps-blocked tasks.

        Before the fix, the cooldown gate ran before _deps_satisfied, causing
        a spammy INFO log for every tick a task had a steward signal AND an
        unsatisfied dep.  After the fix, deps are checked first and such tasks
        skip silently.
        """
        import json as _json
        import logging

        # task '50' is in-progress (not pending, not terminal — does not clear
        # _last_dispatch_at).  task '99' is pending but depends on '50'.
        task_blocking = {
            'id': '50',
            'title': 'Blocking dep task',
            'status': 'in-progress',
            'dependencies': [],
            'metadata': {},
        }
        task_waiting = {
            'id': '99',
            'title': 'Waiting task with cooldown signal',
            'status': 'pending',
            'dependencies': [{'id': '50'}],
            'metadata': {'files': ['backend'], 'recon_reset_count': 2},
        }
        task_response = {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': _json.dumps({'tasks': [task_blocking, task_waiting]}),
                    }
                ]
            }
        }

        config = OrchestratorConfig(max_per_module=1, dispatch_cooldown_secs=1800.0)
        scheduler = Scheduler(config)

        # Simulate a prior dispatch — gate would be active for task '99'
        scheduler._last_dispatch_at['99'] = time.monotonic()

        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call', AsyncMock(return_value=task_response)
        )

        with caplog.at_level(logging.INFO, logger='orchestrator.scheduler'):
            result = await scheduler.acquire_next()

        assert result is None, 'No dispatchable tasks — result must be None'

        # On current main the cooldown gate fires before deps check and logs.
        # After the fix deps-blocked tasks are silently skipped — no cooldown log.
        noisy_records = [
            r for r in caplog.records
            if 'cooldown' in r.getMessage().lower()
            or 'suppressed' in r.getMessage().lower()
        ]
        assert not noisy_records, (
            'Cooldown log must not fire for deps-blocked tasks; got: '
            + ', '.join(r.getMessage() for r in noisy_records)
        )

    @pytest.mark.asyncio
    async def test_acquire_next_tolerates_non_string_reopen_reason(self, monkeypatch):
        """A non-string truthy reopen_reason (e.g. a dict) must not raise and must not arm the gate.

        The ``isinstance`` guard in ``_dispatch_cooldown_signal`` rejects non-string
        ``reopen_reason`` values as no-signal rather than str()-coercing them
        (which could produce false positives for dicts like
        ``{'steward_unblock_failure': True}`` whose repr contains 'steward').

        This test explicitly exercises the gate code path: ``_last_dispatch_at``
        is primed manually before the second acquire so that ``_dispatch_cooldown_active``
        is reached.  The gate must return inactive (non-string → no signal), so the
        second dispatch must succeed rather than raise or be suppressed.
        """
        task = self._pending_task_with(
            {'files': ['backend'], 'reopen_reason': {'malformed': 'producer'}}
        )
        task_response = self._make_task_response(task)

        config = OrchestratorConfig(max_per_module=1, dispatch_cooldown_secs=1800.0)
        scheduler = Scheduler(config)

        mock = AsyncMock(return_value=task_response)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        # First acquire — non-string reopen_reason → no signal → gate NOT armed
        a1 = await scheduler.acquire_next()
        assert a1 is not None and a1.task_id == '99', 'Initial dispatch must succeed'
        assert '99' not in scheduler._last_dispatch_at, (
            'Non-string reopen_reason must not arm the cooldown gate on first dispatch'
        )

        scheduler.release('99')

        # Prime _last_dispatch_at manually so the gate code path is reached on
        # the second acquire (simulates a hypothetical prior signal-bearing dispatch).
        scheduler._last_dispatch_at['99'] = time.monotonic()

        # Second acquire — gate path is now reachable.  The isinstance guard
        # sees a dict (not a str) and returns no signal → gate inactive → dispatch
        # must succeed without raising AttributeError.
        a2 = await scheduler.acquire_next()
        assert a2 is not None and a2.task_id == '99', (
            'Non-string reopen_reason must not raise and must not falsely suppress dispatch'
        )

    @pytest.mark.asyncio
    async def test_dict_reopen_reason_with_steward_key_does_not_arm_gate(self, monkeypatch):
        """A dict reopen_reason whose str() repr contains 'steward' must NOT arm the gate.

        Under the old str()-coerce approach, ``{'steward_unblock_failure': True}``
        would stringify to a repr containing ``'steward'`` and falsely arm the
        cooldown gate.  The isinstance guard fixes this by treating all non-string
        values as no-signal, regardless of their repr content.
        """
        task = self._pending_task_with(
            {'files': ['backend'], 'reopen_reason': {'steward_unblock_failure': True}}
        )
        task_response = self._make_task_response(task)

        config = OrchestratorConfig(max_per_module=1, dispatch_cooldown_secs=1800.0)
        scheduler = Scheduler(config)

        # Confirm directly: signal helper must return None for non-string reopen_reason
        assert scheduler._dispatch_cooldown_signal(task) is None, (
            'Dict reopen_reason must not trigger cooldown signal even if its '
            'str() repr contains the "steward" substring'
        )

        mock = AsyncMock(return_value=task_response)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        a = await scheduler.acquire_next()
        assert a is not None and a.task_id == '99', 'Dispatch must succeed'
        assert '99' not in scheduler._last_dispatch_at, (
            'Dict reopen_reason with "steward" key must not arm the cooldown gate'
        )

    @pytest.mark.asyncio
    async def test_signal_evaluated_once_per_dispatch_with_primed_cooldown(
        self, monkeypatch
    ):
        """_dispatch_cooldown_signal is called at most once per task per acquire_next.

        Scenario: ``_last_dispatch_at`` is primed (simulating a prior signal-bearing
        dispatch whose signal has since cleared).  The candidate task carries no
        cooldown markers so ``_dispatch_cooldown_signal`` returns ``None`` and the
        gate is open.  The signal helper must be called exactly once (during the
        filter loop) and its result reused at the arm site rather than re-evaluated.

        Pre-fix this counter is 2:
        - once inside ``_dispatch_cooldown_active`` (filter path, because
          ``_last_dispatch_at`` is set and elapsed < cooldown window), and
        - once at the arm site (``if self._dispatch_cooldown_signal(task) is not None``).

        Post-fix the counter is 1 (filter only; arm site reads the cached value from
        ``candidate_signals``).
        """
        task = self._pending_task_with({'files': ['backend']})
        task_response = self._make_task_response(task)

        config = OrchestratorConfig(max_per_module=1, dispatch_cooldown_secs=1800.0)
        scheduler = Scheduler(config)

        mock = AsyncMock(return_value=task_response)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        # Prime _last_dispatch_at to simulate a prior signal-bearing dispatch whose
        # signal has since cleared (e.g. recon_reset_count was removed from metadata).
        scheduler._last_dispatch_at['99'] = time.monotonic()

        # Wrap the signal helper with a call counter.  Instance-attribute assignment
        # shadows the bound class method for this scheduler instance only; no module-
        # level patching needed since the instance is local to this test.
        counts = [0]
        real_signal = scheduler._dispatch_cooldown_signal  # bound method

        def counting_signal(task):
            counts[0] += 1
            return real_signal(task)

        scheduler._dispatch_cooldown_signal = counting_signal

        # Dispatch must succeed: gate is open (no signal → cooldown inactive).
        result = await scheduler.acquire_next()
        assert result is not None and result.task_id == '99', (
            'Dispatch must succeed when cooldown window is primed but signal has cleared'
        )
        assert counts[0] == 1, (
            f'_dispatch_cooldown_signal must be called exactly once per dispatch '
            f'(got {counts[0]}); pre-fix it is called twice (filter + arm)'
        )


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
        lt.install_parks('owner', ['backend'], priority='medium')
        assert not lt.try_acquire('other', ['backend'])
        # Owner can still acquire its own park.
        assert lt.try_acquire('owner', ['backend'])

    def test_park_hierarchical_blocks_child(self):
        """A park on a parent module blocks acquire of any child."""
        config = OrchestratorConfig(max_per_module=1, lock_depth=4)
        lt = ModuleLockTable(config)
        lt.install_parks('A', ['autopilot/analyze'], priority='medium')
        assert not lt.try_acquire('B', ['autopilot/analyze/asr'])

    def test_park_hierarchical_blocks_parent(self):
        """A park on a child blocks acquire of its parent."""
        config = OrchestratorConfig(max_per_module=1, lock_depth=4)
        lt = ModuleLockTable(config)
        lt.install_parks(
            'A', ['autopilot/analyze/asr'], priority='medium'
        )
        assert not lt.try_acquire('B', ['autopilot/analyze'])

    def test_park_siblings_dont_conflict(self):
        config = OrchestratorConfig(max_per_module=1, lock_depth=4)
        lt = ModuleLockTable(config)
        lt.install_parks(
            'A', ['autopilot/analyze/asr'], priority='medium'
        )
        assert lt.try_acquire('B', ['autopilot/analyze/speech'])

    def test_clear_parks_for_owner(self):
        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        lt = ModuleLockTable(config)
        lt.install_parks('A', ['backend', 'frontend'], priority='medium')
        assert lt.has_parks('A')
        lt.clear_parks_for('A')
        assert not lt.has_parks('A')
        # Unrelated tasks can now acquire.
        assert lt.try_acquire('B', ['backend'])

    # ---- Mode-2 integration: skip-count promotion ----

    @pytest.fixture
    def fair_config(self) -> OrchestratorConfig:
        """OrchestratorConfig tuned for quick fairness testing."""
        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        config.fairness.skip_threshold = 3
        return config

    @staticmethod
    def _broad_task():
        return {
            'id': 'A',
            'title': 'Broad task',
            'status': 'pending',
            'priority': 'high',
            'dependencies': [],
            'metadata': {'files': ['compiler/src', 'eval/src']},
        }

    @staticmethod
    def _narrow_task(tid: str, module: str, priority: str = 'medium'):
        return {
            'id': tid,
            'title': f'Narrow task {tid}',
            'status': 'pending',
            'priority': priority,
            'dependencies': [],
            'metadata': {'files': [module]},
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
            priority='medium',
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
            'A', ['compiler/src', 'eval/src'], priority='medium'
        )
        a = self._broad_task()
        scheduler.get_tasks = AsyncMock(return_value=[a])

        result = await scheduler.acquire_next()
        assert result is not None and result.task_id == 'A'
        # Parks were cleared on successful acquire.
        assert not scheduler.lock_table.has_parks('A')

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

    # ---- skip-threshold / unconditional park tests ----

    @pytest.mark.asyncio
    async def test_parks_install_unconditionally_without_v2_flag(self):
        """Parks install after the per-tier threshold with default config.

        Regression guard: no flag is needed.  Default high-tier threshold=1,
        so one skip is enough.
        """
        # Default config — no flag assignment needed.
        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        scheduler = Scheduler(config)

        # Seed compiler/src so A (high priority) is forced to skip.
        scheduler.lock_table.try_acquire('seed', ['compiler/src'])
        scheduler._dispatched.add('seed')

        a = self._broad_task()  # high priority
        b = self._narrow_task('B', 'eval/src', priority='medium')
        scheduler.get_tasks = AsyncMock(return_value=[a, b])

        # One tick: high threshold=1, so A should park after this single skip.
        result = await scheduler.acquire_next()
        assert result is not None and result.task_id == 'B'

        # Parks must be installed unconditionally — no flag required.
        assert scheduler.lock_table.has_parks('A')

    @pytest.mark.asyncio
    async def test_eager_park_full_module_set(self):
        """Parks install on ALL of A's modules at once (eager, full coverage).

        Even modules not currently held by another task are covered — this
        prevents racing lower-priority tasks from grabbing a free module
        while A waits for a blocked one.
        """
        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        # Per-tier defaults: high -> threshold=1.
        scheduler = Scheduler(config)

        # B holds compiler/src; C holds eval/src; tools/src is FREE.
        scheduler.lock_table.try_acquire('B', ['compiler/src'])
        scheduler.lock_table.try_acquire('C', ['eval/src'])
        scheduler._dispatched.update(['B', 'C'])

        a = {
            'id': 'A',
            'title': 'Broad task',
            'status': 'pending',
            'priority': 'high',
            'dependencies': [],
            'metadata': {'files': ['compiler/src', 'eval/src', 'tools/src']},
        }
        b = self._narrow_task('D', 'tools/src', priority='low')
        scheduler.get_tasks = AsyncMock(return_value=[a, b])

        # One tick — A skips, park fires on all three modules.
        await scheduler.acquire_next()

        # A's park covers all three modules.
        assert scheduler.lock_table.has_parks('A')
        # D cannot acquire tools/src (free slot) because A's park covers it.
        assert not scheduler.lock_table.try_acquire('D', ['tools/src'])

    @pytest.mark.asyncio
    async def test_cross_tier_preemption(self):
        """A high-tier skip-bump SHADOWS (not destroys) an overlapping low-tier park.

        Setup: low-priority L is already parked on m1+m2 with skip_count=3.
        A high-priority H wants m1 and is also forced to skip.  When H's
        _bump_skip_and_maybe_park fires, it SHADOWS L's park on m1 (pushes H
        on top, L stays buried); L's park on m2 remains active.

        Expected events: reservation_installed (H on m1) + reservation_shadowed
        (L shadowed on m1).  ZERO reservation_evicted events — the old destructive
        eviction is no longer emitted on preemption.
        """
        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        event_store = _RecordingEventStore()
        scheduler = Scheduler(config, event_store=event_store)  # type: ignore[arg-type]

        # Pre-install L's low-tier park on m1 + m2 and seed its skip_count.
        scheduler.lock_table.install_parks('L', ['m1', 'm2'], priority='low')
        scheduler._skip_count['L'] = 3

        # Trigger H's skip-bump-and-park directly; high threshold=1 fires
        # parking on the first call.
        scheduler._bump_skip_and_maybe_park('H', ['m1'], tier='high')

        # H is the active top on m1; L is SHADOWED on m1 (buried) but still
        # has its park there (INV-5) AND is still active on m2.
        assert scheduler.lock_table.has_parks('H')
        assert scheduler.lock_table.has_parks('L'), 'L must be retained in the shadow stack'
        # H's park on m1 blocks ALL other tasks (including shadowed L).
        assert not scheduler.lock_table.try_acquire('X', ['m1'])   # stranger blocked
        assert not scheduler.lock_table.try_acquire('L', ['m1'])   # shadowed L blocked (INV-2)
        # L's park on m2 remains active — H cannot acquire m2.
        assert not scheduler.lock_table.try_acquire('Y', ['m2'])   # L's active park
        assert not scheduler.lock_table.try_acquire('H', ['m2'])   # H cannot acquire L's m2

        # L's skip_count was NOT cleared by preemption.
        assert scheduler._skip_count['L'] == 3

        # Exactly one reservation_installed (for H) and ZERO reservation_evicted.
        installed_events = [
            e for e in event_store.events
            if 'reservation_installed' in e[0]
        ]
        evicted_events = [
            e for e in event_store.events
            if 'reservation_evicted' in e[0]
        ]
        shadowed_events = [
            e for e in event_store.events
            if 'reservation_shadowed' in e[0]
        ]
        assert len(installed_events) == 1, f'Expected 1 installed event; got {installed_events}'
        assert installed_events[0][1]['task_id'] == 'H'
        assert len(evicted_events) == 0, (
            f'reservation_evicted must NOT be emitted on shadow preemption; got {evicted_events}'
        )
        assert len(shadowed_events) == 1, f'Expected 1 shadowed event; got {shadowed_events}'
        shadowed_payload = shadowed_events[0][1]
        assert shadowed_payload['task_id'] == 'H'
        assert shadowed_payload['data']['modules'] == ['m1']
        assert shadowed_payload['data']['preempted_by'] == 'H'
        assert shadowed_payload['data']['preempted_by_priority'] == 'high'
        assert shadowed_payload['data']['victim'] == 'L'

    # ---- reservation_restored emission at all three pop sites (step-7) ----

    @pytest.mark.asyncio
    async def test_restore_emitted_on_dispatch(self):
        """DISPATCH pop-site: when the active-top owner dispatches, clear_parks_for
        restores the buried owner and a reservation_restored event is emitted
        alongside reservation_used.
        """
        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        event_store = _RecordingEventStore()
        scheduler = Scheduler(config, event_store=event_store)  # type: ignore[arg-type]

        # L parks m1 at low; H shadows L on m1 at high (H is active top).
        scheduler.lock_table.install_parks('L', ['m1'], priority='low')
        scheduler.lock_table.install_parks('H', ['m1'], priority='high')
        assert scheduler.lock_table.has_parks('H')
        assert scheduler.lock_table.has_parks('L')

        h_task = {
            'id': 'H', 'title': 'h', 'status': 'pending',
            'priority': 'high', 'dependencies': [],
            'metadata': {'files': ['m1']},
        }
        l_task = {
            'id': 'L', 'title': 'l', 'status': 'pending',
            'priority': 'low', 'dependencies': [],
            'metadata': {'files': ['m1']},
        }
        scheduler.get_tasks = AsyncMock(return_value=[h_task, l_task])

        await scheduler.acquire_next()

        # H dispatched, its park cleared; L is now the restored active top.
        assert not scheduler.lock_table.has_parks('H')
        assert scheduler.lock_table.has_parks('L'), 'L must be restored after H dispatches'

        # reservation_used emitted for H (existing behavior).
        used_events = [e for e in event_store.events if 'reservation_used' in e[0]]
        assert len(used_events) == 1, f'Expected 1 reservation_used; got {used_events}'

        # reservation_restored emitted for L alongside reservation_used.
        restored_events = [e for e in event_store.events if 'reservation_restored' in e[0]]
        assert len(restored_events) == 1, (
            f'Expected 1 reservation_restored on dispatch; got {restored_events}'
        )
        r = restored_events[0][1]
        assert r['data']['restored_owner'] == 'L', f'Wrong restored owner: {r}'
        assert 'm1' in r['data']['modules'], f'Expected m1 in restored modules: {r}'

    @pytest.mark.asyncio
    async def test_restore_emitted_on_release(self):
        """RELEASE pop-site: scheduler.release(shadowing_top) clears its parks,
        restoring the buried owner and emitting reservation_restored.
        """
        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        event_store = _RecordingEventStore()
        scheduler = Scheduler(config, event_store=event_store)  # type: ignore[arg-type]

        # H shadows L on m1.
        scheduler.lock_table.install_parks('L', ['m1'], priority='low')
        scheduler.lock_table.install_parks('H', ['m1'], priority='high')
        assert scheduler.lock_table.has_parks('H')
        assert scheduler.lock_table.has_parks('L')

        # H is currently dispatched (holds a lock elsewhere; we simulate by
        # marking it dispatched so release() operates on a real scenario).
        scheduler._dispatched.add('H')
        scheduler._dispatched_priority['H'] = 'high'

        scheduler.release('H')

        # H's parks cleared; L is restored.
        assert not scheduler.lock_table.has_parks('H')
        assert scheduler.lock_table.has_parks('L'), 'L must be restored after H releases'

        # reservation_restored must be emitted.
        restored_events = [e for e in event_store.events if 'reservation_restored' in e[0]]
        assert len(restored_events) == 1, (
            f'Expected 1 reservation_restored on release; got {restored_events}'
        )
        r = restored_events[0][1]
        assert r['data']['restored_owner'] == 'L'
        assert 'm1' in r['data']['modules']

    @pytest.mark.asyncio
    async def test_restore_emitted_on_gc(self):
        """OWNER-GC pop-site: when a shadowing top is terminal/missing, prune_owners
        removes it via _park_gc, restoring the buried owner; both reservation_expired
        (for the GC'd owner) and reservation_restored (for the restored owner) are
        emitted.
        """
        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        event_store = _RecordingEventStore()
        scheduler = Scheduler(config, event_store=event_store)  # type: ignore[arg-type]

        # L parks m1 at medium; H (which will be cancelled) shadows L at high.
        scheduler.lock_table.install_parks('L', ['m1'], priority='medium')
        scheduler.lock_table.install_parks('H', ['m1'], priority='high')
        scheduler._skip_count['H'] = 5
        assert scheduler.lock_table.has_parks('H')
        assert scheduler.lock_table.has_parks('L')

        # Block 'other' so L cannot dispatch this tick (lets us observe restored state).
        scheduler.lock_table.try_acquire('blocker', ['other'])
        scheduler._dispatched.add('blocker')

        # H is cancelled; L is a live pending task.
        h_task = {
            'id': 'H', 'title': 'h', 'status': 'cancelled',
            'priority': 'high', 'dependencies': [],
            'metadata': {'files': ['m1']},
        }
        l_task = {
            'id': 'L', 'title': 'l', 'status': 'pending',
            'priority': 'medium', 'dependencies': [],
            'metadata': {'files': ['other']},
        }
        scheduler.get_tasks = AsyncMock(return_value=[h_task, l_task])

        await scheduler.acquire_next()

        # H GC'd; L is restored as the active reservation on m1.
        assert not scheduler.lock_table.has_parks('H'), 'H (cancelled) must be GC-evicted'
        assert scheduler.lock_table.has_parks('L'), 'L must be restored after H is GC-evicted'

        # reservation_expired emitted for H (existing GC behavior).
        expired_events = [e for e in event_store.events if 'reservation_expired' in e[0]]
        assert len(expired_events) == 1, f'Expected 1 reservation_expired; got {expired_events}'
        assert expired_events[0][1]['task_id'] == 'H'

        # reservation_restored emitted for L alongside reservation_expired.
        restored_events = [e for e in event_store.events if 'reservation_restored' in e[0]]
        assert len(restored_events) == 1, (
            f'Expected 1 reservation_restored on GC; got {restored_events}'
        )
        r = restored_events[0][1]
        assert r['data']['restored_owner'] == 'L'
        assert 'm1' in r['data']['modules']

    # ---- Owner-state park-GC (step-14) ----

    @pytest.mark.asyncio
    async def test_park_gc_on_terminal_owner(self):
        """A park owned by a terminal task is reaped on the next tick."""
        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        event_store = _RecordingEventStore()
        scheduler = Scheduler(config, event_store=event_store)  # type: ignore[arg-type]

        scheduler.lock_table.install_parks('A', ['m1', 'm2'], priority='high')
        scheduler._skip_count['A'] = 5

        # A is cancelled; B is a separate pending task.
        a = {
            'id': 'A', 'title': 'a', 'status': 'cancelled',
            'priority': 'high', 'dependencies': [],
            'metadata': {'files': ['m1']},
        }
        b = {
            'id': 'B', 'title': 'b', 'status': 'pending',
            'priority': 'medium', 'dependencies': [],
            'metadata': {'files': ['other/src']},
        }
        scheduler.get_tasks = AsyncMock(return_value=[a, b])

        await scheduler.acquire_next()

        # A's parks gone, skip_count cleared.
        assert not scheduler.lock_table.has_parks('A')
        assert 'A' not in scheduler._skip_count
        # B (or any unrelated task) can now acquire m1.
        assert scheduler.lock_table.try_acquire('B', ['m1'])
        # reservation_expired event emitted with terminal reason.
        expired = [
            e for e in event_store.events
            if 'reservation_expired' in e[0]
        ]
        assert len(expired) == 1
        assert expired[0][1]['task_id'] == 'A'
        assert 'terminal' in expired[0][1]['data']['reason']

    @pytest.mark.asyncio
    async def test_park_gc_on_missing_owner(self):
        """A park whose owner is no longer in the task list is reaped."""
        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        event_store = _RecordingEventStore()
        scheduler = Scheduler(config, event_store=event_store)  # type: ignore[arg-type]

        scheduler.lock_table.install_parks('X', ['m1'], priority='high')
        scheduler._skip_count['X'] = 2

        # X is NOT in the task list — reconciliation removed it.
        b = {
            'id': 'B', 'title': 'b', 'status': 'pending',
            'priority': 'medium', 'dependencies': [],
            'metadata': {'files': ['other/src']},
        }
        scheduler.get_tasks = AsyncMock(return_value=[b])

        await scheduler.acquire_next()

        assert not scheduler.lock_table.has_parks('X')
        assert 'X' not in scheduler._skip_count
        expired = [
            e for e in event_store.events
            if 'reservation_expired' in e[0]
        ]
        assert len(expired) == 1
        assert expired[0][1]['task_id'] == 'X'
        assert 'missing' in expired[0][1]['data']['reason']

    @pytest.mark.asyncio
    async def test_park_gc_on_deps_unsatisfied(self):
        """A park whose owner has un-satisfied deps is reaped."""
        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        event_store = _RecordingEventStore()
        scheduler = Scheduler(config, event_store=event_store)  # type: ignore[arg-type]

        scheduler.lock_table.install_parks('A', ['m1'], priority='high')

        # A depends on '7' which is in-progress (not satisfied).
        a = {
            'id': 'A', 'title': 'a', 'status': 'pending',
            'priority': 'high',
            'dependencies': [{'id': '7'}],
            'metadata': {'files': ['m1']},
        }
        seven = {
            'id': '7', 'title': 'seven', 'status': 'in-progress',
            'priority': 'high', 'dependencies': [],
            'metadata': {'files': ['other/src']},
        }
        scheduler.get_tasks = AsyncMock(return_value=[a, seven])

        await scheduler.acquire_next()

        assert not scheduler.lock_table.has_parks('A')
        expired = [
            e for e in event_store.events
            if 'reservation_expired' in e[0]
        ]
        assert len(expired) == 1
        assert expired[0][1]['task_id'] == 'A'
        assert 'deps' in expired[0][1]['data']['reason']

    @pytest.mark.asyncio
    async def test_park_gc_skips_eligible_owner(self):
        """Control: an owner whose deps ARE satisfied keeps its park.

        Seed a real holder on m1 so A cannot acquire its own park this tick;
        we only want to verify that the GC sweep doesn't reap an eligible
        owner's park between sweep and dispatch.
        """
        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        event_store = _RecordingEventStore()
        scheduler = Scheduler(config, event_store=event_store)  # type: ignore[arg-type]

        # Block m1 with a seed so A can't acquire its own park.
        scheduler.lock_table.try_acquire('seed', ['m1'])
        scheduler._dispatched.add('seed')

        scheduler.lock_table.install_parks('A', ['m1'], priority='high')

        # A depends on '7' which is done — deps satisfied.
        a = {
            'id': 'A', 'title': 'a', 'status': 'pending',
            'priority': 'high',
            'dependencies': [{'id': '7'}],
            'metadata': {'files': ['m1']},
        }
        seven = {
            'id': '7', 'title': 'seven', 'status': 'done',
            'priority': 'high', 'dependencies': [],
            'metadata': {'files': ['other/src']},
        }
        scheduler.get_tasks = AsyncMock(return_value=[a, seven])

        await scheduler.acquire_next()

        # A's park survives (GC didn't reap it; m1 still blocked so A couldn't
        # acquire-and-clear).
        assert scheduler.lock_table.has_parks('A')
        expired = [
            e for e in event_store.events
            if 'reservation_expired' in e[0]
        ]
        assert len(expired) == 0

    # ---- G2 anti-starvation integration test (step-9, reify 4652→3427) ----

    @pytest.mark.asyncio
    async def test_anti_starvation_shadow_restore_blocks_medium(self):
        """End-to-end regression for reify 4652→3427: shadow+restore blocks medium
        from stealing modules after a critical task completes.

        Scenario
        --------
        H  (high)     parks m1 + m2 (wide footprint via skip-bump parking)
        C  (critical) preempts m1 by pushing H onto shadow stack (C on top)
        M  (medium)   wants m1

        Regression pin (old destructive eviction)
        -----------------------------------------
        Old code: C evicts H from m1.  When C dispatches and clears its park,
        m1 becomes FREE and M steals it before H can reassemble its footprint.
        H starves forever.

        New shadow semantics
        -------------------
        C pushes on top of H on m1.  When C dispatches/clears, H is RESTORED
        as the active top.  M is blocked at EVERY step — during the shadow AND
        after C completes.  Only when H has acquired ALL modules (m1+m2) and
        later releases them can M dispatch.
        """
        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        event_store = _RecordingEventStore()
        scheduler = Scheduler(config, event_store=event_store)  # type: ignore[arg-type]

        # --- Phase 0: Install initial parks ---
        # H accumulated skips and parks m1+m2 at high priority.
        scheduler.lock_table.install_parks('H', ['m1', 'm2'], priority='high')
        # C preempts m1 at critical priority (shadows H on m1; H still active on m2).
        _, shadowed = scheduler.lock_table.install_parks('C', ['m1'], priority='critical')
        assert shadowed == [('H', ['m1'])], f'Expected H shadowed on m1; got {shadowed}'
        assert scheduler.lock_table.has_parks('H'), 'H must remain in shadow stack (INV-5)'
        assert scheduler.lock_table.has_parks('C'), 'C must be active top on m1'

        # Regression pin part 1: M cannot acquire m1 while C is on top.
        assert not scheduler.lock_table.try_acquire('M', ['m1']), (
            'M must be blocked by C (active top of m1) during shadow'
        )

        # --- Phase 1: C dispatches via acquire_next ---
        c_task = {
            'id': 'C', 'title': 'c', 'status': 'pending',
            'priority': 'critical', 'dependencies': [],
            'metadata': {'files': ['m1']},
        }
        h_task = {
            'id': 'H', 'title': 'h', 'status': 'pending',
            'priority': 'high', 'dependencies': [],
            'metadata': {'files': ['m1', 'm2']},
        }
        m_task = {
            'id': 'M', 'title': 'm', 'status': 'pending',
            'priority': 'medium', 'dependencies': [],
            'metadata': {'files': ['m1']},
        }
        scheduler.get_tasks = AsyncMock(return_value=[c_task, h_task, m_task])

        result = await scheduler.acquire_next()
        assert result is not None and result.task_id == 'C', (
            f'C (critical) must dispatch first; got {result}'
        )
        # C holds m1 now; clear_parks_for('C') restored H to top of m1.
        assert scheduler.lock_table.has_parks('H'), (
            'H must be restored to active top of m1 after C dispatches (INV-4)'
        )
        assert not scheduler.lock_table.has_parks('C'), 'C park must be cleared on dispatch'

        # Regression pin part 2: M still blocked after C dispatches — H is restored.
        assert not scheduler.lock_table.try_acquire('M', ['m1']), (
            'M must still be blocked after C dispatches: H is restored as active top'
        )

        # reservation_restored event emitted for H (from dispatch clear_parks_for).
        restored_on_dispatch = [e for e in event_store.events if 'reservation_restored' in e[0]]
        assert len(restored_on_dispatch) == 1, (
            f'Expected 1 reservation_restored on C dispatch; got {restored_on_dispatch}'
        )
        assert restored_on_dispatch[0][1]['data']['restored_owner'] == 'H'

        # --- Phase 2: C releases m1 ---
        scheduler.release('C')
        # m1 is now free (C released it), H's park still active on m1.
        assert not scheduler.lock_table._held.get('C'), 'C must have released all locks'
        assert scheduler.lock_table.has_parks('H'), 'H park on m1 must survive C release'

        # Regression pin part 3: M still cannot steal m1 — H's park guards it.
        assert not scheduler.lock_table.try_acquire('M', ['m1']), (
            'M must NOT steal m1 after C releases — H park still guards it (regression pin)'
        )

        # --- Phase 3: H dispatches (acquires its full footprint m1+m2) ---
        c_task_done = dict(c_task, status='done')
        scheduler.get_tasks = AsyncMock(return_value=[c_task_done, h_task, m_task])

        result2 = await scheduler.acquire_next()
        assert result2 is not None and result2.task_id == 'H', (
            f'H must dispatch next (after C is done); got {result2}'
        )
        # H holds m1 and m2; its parks are cleared.
        assert not scheduler.lock_table.has_parks('H'), 'H parks must be cleared on dispatch'
        # M still cannot acquire m1 — H HOLDS it.
        assert not scheduler.lock_table.try_acquire('M', ['m1']), (
            'M must not acquire m1 while H holds it'
        )

        # --- Phase 4: H completes → M can finally dispatch ---
        scheduler.release('H')

        scheduler.get_tasks = AsyncMock(return_value=[c_task_done, dict(h_task, status='done'), m_task])

        result3 = await scheduler.acquire_next()
        assert result3 is not None and result3.task_id == 'M', (
            f'M must dispatch only after H fully completes; got {result3}'
        )


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
    async def test_persistent_mcp_exception_raises_after_retries(
        self, scheduler: Scheduler, monkeypatch, caplog
    ):
        """Persistent MCP exceptions raise RuntimeError after the retry cap.

        Fix 3: previously the scheduler logged + returned silently on any
        ``dispatch_tool`` exception, which left tasks stranded in-progress
        when the fused-memory backend was reconnecting.  We now retry
        ``_TRANSIENT_RETRIES`` times and raise so callers can decide
        whether to release locks (handle_blast_radius_expansion) or fall
        through to the workflow's exception handler (workflow.run).
        """
        import logging as _logging
        # Tighten retry timing to keep the test fast.
        monkeypatch.setattr('orchestrator.scheduler._TRANSIENT_BACKOFF_BASE', 0.0)
        mock = AsyncMock(side_effect=OSError(2, 'No such file'))
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)
        with caplog.at_level(_logging.ERROR, logger='orchestrator.scheduler'), pytest.raises(RuntimeError, match='3 transient retries'):
            await scheduler.set_task_status('1', 'in-progress')
        # Three attempts before raising.
        assert mock.await_count == 3, (
            f'Expected 3 dispatch attempts, got {mock.await_count}'
        )
        # Each attempt logs an exception traceback at ERROR level.
        assert sum(
            1 for rec in caplog.records if 'set_task_status' in rec.message and rec.exc_info
        ) >= 3

    @pytest.mark.asyncio
    async def test_transient_rejection_retries_until_success(
        self, scheduler: Scheduler, monkeypatch, caplog
    ):
        """A TimeoutError-shaped rejection retries; later success returns clean."""
        import logging as _logging
        monkeypatch.setattr('orchestrator.scheduler._TRANSIENT_BACKOFF_BASE', 0.0)
        # Two transient rejections, then success.
        transient = {
            'result': {'structuredContent': {
                'error': "TimeoutError('ensure_connected timed out')",
                'error_type': 'TimeoutError',
            }},
        }
        success = {
            'result': {'structuredContent': {
                'message': 'ok', 'tasks': [{'success': True}],
            }},
        }
        mock = AsyncMock(side_effect=[transient, transient, success])
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)
        with caplog.at_level(_logging.INFO, logger='orchestrator.scheduler'):
            await scheduler.set_task_status('5', 'in-progress')
        assert mock.await_count == 3
        # Two transient-retry INFO logs (not WARNING — we only WARN for terminal rejections).
        retries = [r for r in caplog.records if 'transient rejection' in r.message]
        assert len(retries) == 2

    @pytest.mark.asyncio
    async def test_transient_rejection_raises_after_exhaust(
        self, scheduler: Scheduler, monkeypatch
    ):
        """Persistent transient rejection raises RuntimeError after the cap."""
        monkeypatch.setattr('orchestrator.scheduler._TRANSIENT_BACKOFF_BASE', 0.0)
        transient = {
            'result': {'structuredContent': {
                'error': "TimeoutError('ensure_connected timed out')",
                'error_type': 'TimeoutError',
            }},
        }
        mock = AsyncMock(return_value=transient)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)
        with pytest.raises(RuntimeError, match='TimeoutError'):
            await scheduler.set_task_status('5', 'in-progress')
        assert mock.await_count == 3

    @pytest.mark.asyncio
    async def test_non_transient_rejection_does_not_retry(
        self, scheduler: Scheduler, monkeypatch
    ):
        """Phantom-done gate (non-transient) raises DoneGateRejection — no retry."""
        from orchestrator.scheduler import DoneGateRejection
        monkeypatch.setattr('orchestrator.scheduler._TRANSIENT_BACKOFF_BASE', 0.0)
        rejection = {
            'result': {'structuredContent': {
                'success': False, 'error': 'done_gate_missing_files',
                'missing_files': ['src/missing.py'],
                'hint': 'metadata.files lists missing paths',
            }},
        }
        mock = AsyncMock(return_value=rejection)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)
        with pytest.raises(DoneGateRejection) as excinfo:
            await scheduler.set_task_status('42', 'done', done_provenance={
                'kind': 'merged', 'commit': 'deadbeef',
            })
        assert mock.await_count == 1, 'non-transient rejection must not retry'
        assert excinfo.value.task_id == '42'
        assert excinfo.value.missing_files == ['src/missing.py']

    @pytest.mark.asyncio
    async def test_structured_rejection_raises_on_provenance_invalid(
        self, scheduler: Scheduler, monkeypatch,
    ):
        """fused-memory's done_provenance_invalid raises ProvenanceValidationRejection.

        Regression for the silent-rejection bug that left tasks stuck
        in-progress after CAS retry: workflow.set_task_status('done', ...) was
        passed a stale merge SHA, fused-memory's done_provenance ancestor
        check rejected it, scheduler dropped the response on the floor.
        Now propagates so the caller can route to L1.
        """
        from orchestrator.scheduler import ProvenanceValidationRejection
        rejection_response = {
            'result': {
                'structuredContent': {
                    'success': False,
                    'error': 'done_provenance_invalid',
                    'hint': 'kind="merged" but commit deadbeef is not on main',
                },
                'isError': False,
            },
        }
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=rejection_response),
        )
        with pytest.raises(ProvenanceValidationRejection) as excinfo:
            await scheduler.set_task_status('42', 'done', done_provenance={
                'kind': 'merged', 'commit': 'deadbeef',
            })
        assert excinfo.value.task_id == '42'
        assert excinfo.value.error_code == 'done_provenance_invalid'

    @pytest.mark.asyncio
    async def test_provenance_required_raises(
        self, scheduler: Scheduler, monkeypatch,
    ):
        """done_provenance_required raises ProvenanceValidationRejection."""
        from orchestrator.scheduler import ProvenanceValidationRejection
        rejection_response = {
            'result': {'structuredContent': {
                'success': False,
                'error': 'done_provenance_required',
                'hint': 'done_provenance is required',
            }},
        }
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=rejection_response),
        )
        with pytest.raises(ProvenanceValidationRejection) as excinfo:
            await scheduler.set_task_status('42', 'done')
        assert excinfo.value.error_code == 'done_provenance_required'

    @pytest.mark.asyncio
    async def test_unknown_non_transient_rejection_raises_base(
        self, scheduler: Scheduler, monkeypatch,
    ):
        """An unrecognised non-transient error_code raises SetTaskStatusRejected."""
        from orchestrator.scheduler import SetTaskStatusRejected
        rejection_response = {
            'result': {'structuredContent': {
                'success': False,
                'error': 'something_unexpected',
                'hint': 'novel server-side rejection',
            }},
        }
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=rejection_response),
        )
        with pytest.raises(SetTaskStatusRejected) as excinfo:
            await scheduler.set_task_status('42', 'in-progress')
        assert excinfo.value.error_code == 'something_unexpected'

    @pytest.mark.asyncio
    async def test_success_response_does_not_log_warning(
        self, scheduler: Scheduler, monkeypatch, caplog,
    ):
        """A normal successful set_task_status response must not trigger the warning."""
        import logging as _logging
        success_response = {
            'result': {
                'structuredContent': {
                    'message': 'Successfully updated 1 task(s) to "done"',
                    'tasks': [{
                        'success': True, 'oldStatus': 'in-progress',
                        'newStatus': 'done', 'taskId': '42',
                    }],
                },
                'isError': False,
            },
        }
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=success_response),
        )
        with caplog.at_level(_logging.WARNING, logger='orchestrator.scheduler'):
            await scheduler.set_task_status('42', 'done', done_provenance={
                'kind': 'merged', 'commit': 'deadbeef',
            })
        assert not any(
            'rejected by fused-memory' in rec.message for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_no_op_response_does_not_log_warning(
        self, scheduler: Scheduler, monkeypatch, caplog,
    ):
        """The same-status no-op (success: True, no_op: True) is not a rejection."""
        import logging as _logging
        noop_response = {
            'result': {
                'structuredContent': {
                    'success': True, 'no_op': True, 'task_id': '42',
                },
                'isError': False,
            },
        }
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=noop_response),
        )
        with caplog.at_level(_logging.WARNING, logger='orchestrator.scheduler'):
            await scheduler.set_task_status('42', 'in-progress')
        assert not any(
            'rejected by fused-memory' in rec.message for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_set_task_status_raises_on_terminal_exit_blocked(
        self, scheduler: Scheduler, monkeypatch
    ):
        """Server-rejected ``done -> blocked`` (no reopen_reason) raises TerminalExitRejection.

        The rejection is a logical contradiction — the row is already terminal
        and the caller asked for a non-terminal target with no reopen_reason.
        Callers (notably ``workflow._mark_blocked``) need to distinguish this
        from a transient backend blip so they can run bypass-detection rather
        than swallow the rejection.
        """
        from orchestrator.scheduler import TerminalExitRejection
        rejection_response = {
            'result': {
                'structuredContent': {
                    'success': False,
                    'error': 'terminal_exit_rejected',
                    'task_id': '42',
                    'from_status': 'done',
                    'to_status': 'blocked',
                    'hint': "Cannot transition from 'done' to 'blocked' …",
                },
                'isError': False,
            },
        }
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=rejection_response),
        )
        with pytest.raises(TerminalExitRejection) as excinfo:
            await scheduler.set_task_status('42', 'blocked')
        exc = excinfo.value
        assert exc.task_id == '42'
        assert exc.old_status == 'done'
        assert exc.target_status == 'blocked'

    @pytest.mark.asyncio
    async def test_set_task_status_silent_on_redundant_terminal_target(
        self, scheduler: Scheduler, monkeypatch, caplog,
    ):
        """``done -> done`` should not raise — same-status writes are idempotent.

        The exception is reserved for logical contradictions (terminal -> non-terminal
        with no reopen_reason); a terminal -> terminal target is just a no-op
        on the server side and we never want callers to see it as an error.
        """
        import logging as _logging
        # Note: this is the unlikely shape where the server elects to return
        # terminal_exit_rejected for a terminal target. Real fused-memory
        # returns no_op for done->done; we simulate the corner case where the
        # server were to ever return the structured error to confirm we don't
        # raise.
        rejection_response = {
            'result': {
                'structuredContent': {
                    'success': False,
                    'error': 'terminal_exit_rejected',
                    'task_id': '42',
                    'from_status': 'done',
                    'to_status': 'done',
                    'hint': '…',
                },
            },
        }
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=rejection_response),
        )
        with caplog.at_level(_logging.WARNING, logger='orchestrator.scheduler'):
            # Must NOT raise — terminal target is not a logical contradiction.
            await scheduler.set_task_status('42', 'done')

    @pytest.mark.asyncio
    async def test_set_task_status_silent_when_reopen_reason_supplied(
        self, scheduler: Scheduler, monkeypatch,
    ):
        """When the caller passed reopen_reason, even a terminal_exit_rejected
        rejection from the server doesn't raise — the caller already
        acknowledged the terminal state.
        """
        rejection_response = {
            'result': {
                'structuredContent': {
                    'success': False,
                    'error': 'terminal_exit_rejected',
                    'task_id': '42',
                    'from_status': 'done',
                    'to_status': 'blocked',
                },
            },
        }
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=rejection_response),
        )
        # Must NOT raise — caller passed reopen_reason; the rejection isn't a
        # logical contradiction, just a server-side validation message.
        await scheduler.set_task_status(
            '42', 'blocked', reopen_reason='manual',
        )


class TestExtractRejection:
    """Direct tests of the response-shape parser used by set_task_status."""

    def test_structured_content_with_error(self):
        from orchestrator.scheduler import extract_rejection
        msg = extract_rejection({
            'result': {'structuredContent': {
                'success': False, 'error': 'done_gate_missing_files',
                'hint': 'metadata.files lists missing paths',
            }},
        })
        assert msg is not None
        assert 'done_gate_missing_files' in msg
        assert 'metadata.files' in msg

    def test_text_block_fallback(self):
        """When structuredContent is absent, parse the JSON text block."""
        import json as _json

        from orchestrator.scheduler import extract_rejection
        payload = {'success': False, 'error': 'terminal_exit_rejected'}
        msg = extract_rejection({
            'result': {
                'content': [{'type': 'text', 'text': _json.dumps(payload)}],
            },
        })
        assert msg is not None
        assert 'terminal_exit_rejected' in msg

    def test_success_returns_none(self):
        from orchestrator.scheduler import extract_rejection
        assert extract_rejection({
            'result': {'structuredContent': {
                'message': 'Successfully updated', 'tasks': [{'success': True}],
            }},
        }) is None

    def test_no_op_returns_none(self):
        from orchestrator.scheduler import extract_rejection
        assert extract_rejection({
            'result': {'structuredContent': {
                'success': True, 'no_op': True, 'task_id': '7',
            }},
        }) is None

    def test_empty_response_returns_none(self):
        from orchestrator.scheduler import extract_rejection
        assert extract_rejection({}) is None
        assert extract_rejection(None) is None


class TestIsTransientRejection:
    """Classifier for retry decisions in ``Scheduler.set_task_status``."""

    def test_timeout_error_is_transient(self):
        from orchestrator.scheduler import is_transient_rejection
        assert is_transient_rejection("TimeoutError('ensure_connected timed out') — TimeoutError")
        assert is_transient_rejection('asyncio.TimeoutError')

    def test_connection_error_is_transient(self):
        from orchestrator.scheduler import is_transient_rejection
        assert is_transient_rejection('ConnectionError: read timeout')
        assert is_transient_rejection('httpx.ConnectError: refused')

    def test_done_gate_is_not_transient(self):
        """Phantom-done-gate rejection is a workflow bug, not a backend blip."""
        from orchestrator.scheduler import is_transient_rejection
        assert not is_transient_rejection(
            'done_gate_missing_files — metadata.files lists missing paths'
        )

    def test_terminal_exit_is_not_transient(self):
        from orchestrator.scheduler import is_transient_rejection
        assert not is_transient_rejection(
            "terminal_exit_rejected — Cannot transition from 'done' to 'pending'"
        )

    def test_none_and_empty_return_false(self):
        from orchestrator.scheduler import is_transient_rejection
        assert not is_transient_rejection(None)
        assert not is_transient_rejection('')


# ---------------------------------------------------------------------------
# Value/h scoring: priority inheritance (P1), age boost (P2), CPM weight (P3),
# per-tier slot caps (Fix 3), per-tier skip thresholds (Fix 2).
# ---------------------------------------------------------------------------


def _pending_task(
    task_id: str,
    *,
    priority: str = 'medium',
    deps: list[str] | None = None,
    files: list[str] | None = None,
    status: str = 'pending',
) -> dict:
    """Helper: build a task dict with all fields the scheduler reads."""
    return {
        'id': task_id,
        'title': f'Task {task_id}',
        'status': status,
        'priority': priority,
        'dependencies': [{'id': d} for d in (deps or [])],
        'metadata': {'files': files or [f'mod{task_id}']},
    }


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


class TestPriorityOverrideBoostOverlay:
    """Boost overlay composes with the existing min-rank race in _compute_effective_priorities."""

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        return Scheduler(OrchestratorConfig(max_per_module=1))

    def test_boost_overlay_lifts_own_priority(self, scheduler: Scheduler):
        """A boost_tier above the task's own tier becomes the effective priority."""
        task_a = _pending_task('A', priority='medium')
        by_id = {'A': task_a}
        rev = scheduler._build_reverse_index([task_a])
        status_map = {'A': 'pending'}

        eff = scheduler._compute_effective_priorities(
            by_id, rev, status_map, override_boosts={'A': 'critical'}
        )
        assert eff['A'] == 'critical'

    def test_boost_overlay_composes_with_inheritance(self, scheduler: Scheduler):
        """Boost + own + inherited priority all race; highest-rank wins.

        base='medium', dependent='critical', boost='high':
        best-rank = min(rank[medium], rank[critical], rank[high]) = rank[critical]
        """
        base = _pending_task('base', priority='medium')
        consumer = _pending_task('consumer', priority='critical', deps=['base'])
        tasks = [base, consumer]
        by_id = {t['id']: t for t in tasks}
        rev = scheduler._build_reverse_index(tasks)
        status_map = {t['id']: t['status'] for t in tasks}

        eff = scheduler._compute_effective_priorities(
            by_id, rev, status_map, override_boosts={'base': 'high'}
        )
        # inherited 'critical' beats own 'medium' and boost 'high'
        assert eff['base'] == 'critical'

    def test_boost_overlay_loses_to_higher_own(self, scheduler: Scheduler):
        """A boost lower than the task's own tier has no effect."""
        task_a = _pending_task('A', priority='critical')
        by_id = {'A': task_a}
        rev = scheduler._build_reverse_index([task_a])
        status_map = {'A': 'pending'}

        eff = scheduler._compute_effective_priorities(
            by_id, rev, status_map, override_boosts={'A': 'high'}
        )
        assert eff['A'] == 'critical'

    def test_compute_effective_priorities_default_overrides_none(
        self, scheduler: Scheduler
    ):
        """Omitting override_boosts (default None) preserves existing behavior."""
        base = _pending_task('10', priority='medium')
        consumer = _pending_task('11', priority='critical', deps=['10'])
        tasks = [base, consumer]
        by_id = {t['id']: t for t in tasks}
        rev = scheduler._build_reverse_index(tasks)
        status_map = {t['id']: t['status'] for t in tasks}

        # No override_boosts kwarg — must not raise
        eff = scheduler._compute_effective_priorities(by_id, rev, status_map)
        assert eff['10'] == 'critical'
        assert eff['11'] == 'critical'


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


class TestLegacyOrderingPreserved:
    """With 3-tier data + default tier_slot_caps all 1.0 + no CPM + zero age,
    the new scheduler must match the legacy high>medium>low dispatch order.
    """

    @pytest.mark.asyncio
    async def test_legacy_three_tier_ordering_preserved(self):
        config = OrchestratorConfig(max_per_module=1, max_concurrent_tasks=10)
        # Disable caps and fairness carve-outs for this test.
        scheduler = Scheduler(config)
        tasks = [
            _pending_task('1', priority='low', files=['modA']),
            _pending_task('2', priority='high', files=['modB']),
            _pending_task('3', priority='medium', files=['modC']),
        ]
        scheduler.get_tasks = AsyncMock(return_value=tasks)
        result = await scheduler.acquire_next()
        assert result is not None and result.task_id == '2', 'high wins'

    @pytest.mark.asyncio
    async def test_critical_beats_high(self):
        """New 5-tier: critical outranks high."""
        config = OrchestratorConfig(max_per_module=1, max_concurrent_tasks=10)
        scheduler = Scheduler(config)
        tasks = [
            _pending_task('1', priority='high', files=['modA']),
            _pending_task('2', priority='critical', files=['modB']),
        ]
        scheduler.get_tasks = AsyncMock(return_value=tasks)
        result = await scheduler.acquire_next()
        assert result is not None and result.task_id == '2'

    @pytest.mark.asyncio
    async def test_polish_loses_to_low(self):
        """New 5-tier: polish ranks below low."""
        config = OrchestratorConfig(max_per_module=1, max_concurrent_tasks=10)
        scheduler = Scheduler(config)
        tasks = [
            _pending_task('1', priority='polish', files=['modA']),
            _pending_task('2', priority='low', files=['modB']),
        ]
        scheduler.get_tasks = AsyncMock(return_value=tasks)
        result = await scheduler.acquire_next()
        assert result is not None and result.task_id == '2'

    @pytest.mark.asyncio
    async def test_inheritance_lifts_dependency(self):
        """A medium task with a critical dependent is dispatched first."""
        config = OrchestratorConfig(max_per_module=1, max_concurrent_tasks=10)
        scheduler = Scheduler(config)
        # Task 1 (medium, available) is needed by task 2 (critical, blocked).
        # Inheritance should lift task 1 above task 3 (high, available).
        tasks = [
            _pending_task('1', priority='medium', files=['modA']),
            _pending_task('2', priority='critical', deps=['1'], files=['modB']),
            _pending_task('3', priority='high', files=['modC']),
        ]
        scheduler.get_tasks = AsyncMock(return_value=tasks)
        result = await scheduler.acquire_next()
        assert result is not None and result.task_id == '1'


class TestDispatchPriorityBookkeeping:
    """_dispatched_priority must be updated on acquire AND release."""

    @pytest.mark.asyncio
    async def test_release_clears_dispatched_priority(self):
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        task = _pending_task('1', priority='high', files=['modA'])
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
        scheduler = Scheduler(config)
        tasks = [
            _pending_task('1', priority='medium', files=['modA']),
            _pending_task('2', priority='critical', deps=['1'], files=['modB']),
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
        # The success branch now persists metadata.files when stale is non-empty.
        # Mock get_task/update_task so the test remains hermetic (no network I/O).
        scheduler.get_task = AsyncMock(  # type: ignore[method-assign]
            return_value={'id': '936', 'metadata': {}}
        )
        scheduler.update_task = AsyncMock(return_value=True)  # type: ignore[method-assign]
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
        # The success branch persists metadata.files when stale is non-empty.
        # Mock get_task/update_task so the test remains hermetic (no network I/O).
        scheduler.get_task = AsyncMock(  # type: ignore[method-assign]
            return_value={'id': '936', 'metadata': {}}
        )
        scheduler.update_task = AsyncMock(return_value=True)  # type: ignore[method-assign]
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

    @pytest.mark.asyncio
    async def test_acquire_failure_preserves_memory_hints(self, scheduler: Scheduler):
        """Sibling keys set by Stage-2 reconciliation survive the blast-radius files write.

        When lock acquisition fails and the scheduler rewrites metadata.files,
        it must read the current backend metadata first so that memory_hints and
        _causation_id attached after the task went in-progress are not clobbered.
        """
        lt = scheduler.lock_table
        assert lt.try_acquire('936', ['crates/reify-compiler/src/lib.rs'])
        assert lt.try_acquire('other', ['crates/reify-compiler/src/conformance.rs'])

        backend_md = {
            'memory_hints': {'entities': ['E1'], 'queries': ['q1']},
            '_causation_id': 'C1',
        }
        scheduler.get_task = AsyncMock(  # type: ignore[method-assign]
            return_value={'id': '936', 'metadata': backend_md}
        )
        update_task = AsyncMock(return_value=True)
        scheduler.update_task = update_task  # type: ignore[method-assign]
        scheduler.set_task_status = AsyncMock(return_value=None)  # type: ignore[method-assign]

        ok = await scheduler.handle_blast_radius_expansion(
            '936',
            current=['crates/reify-compiler/src/lib.rs'],
            needed=['crates/reify-compiler/src/conformance.rs'],
        )

        assert ok is False
        assert update_task.await_args is not None
        persisted = update_task.await_args.args[1]
        assert persisted == {
            'memory_hints': {'entities': ['E1'], 'queries': ['q1']},
            '_causation_id': 'C1',
            'files': ['crates/reify-compiler/src/conformance.rs'],
        }, f'Sibling keys from backend must survive blast-radius files write; got {persisted}'

    @pytest.mark.asyncio
    async def test_narrowing_persists_metadata_files(self, scheduler: Scheduler):
        """Plan narrowing on the SUCCESS branch must persist metadata.files (set-to-plan).

        Sibling keys (memory_hints, _causation_id) must survive the write so
        Stage-2 reconciliation data is not clobbered — mirrors the requeue-branch
        read-modify-write pattern (scheduler.py handle_blast_radius_expansion failure
        branch) on the success path.
        """
        lt = scheduler.lock_table
        assert lt.try_acquire('936', ['crates/reify-compiler/src/lib.rs'])
        backend_md = {
            'memory_hints': {'entities': ['E1']},
            '_causation_id': 'C1',
        }
        scheduler.get_task = AsyncMock(  # type: ignore[method-assign]
            return_value={'id': '936', 'metadata': backend_md}
        )
        update_task = AsyncMock(return_value=True)
        scheduler.update_task = update_task  # type: ignore[method-assign]

        ok = await scheduler.handle_blast_radius_expansion(
            '936',
            current=['crates/reify-compiler/src/lib.rs'],
            needed=['crates/reify-compiler/src/conformance.rs'],
        )

        assert ok is True
        assert update_task.await_args is not None, (
            'update_task must be called to make the narrowed set durable'
        )
        persisted = update_task.await_args.args[1]
        assert persisted == {
            'memory_hints': {'entities': ['E1']},
            '_causation_id': 'C1',
            'files': ['crates/reify-compiler/src/conformance.rs'],
        }, f'Sibling keys must survive; files narrowed to needed; got {persisted}'

    @pytest.mark.asyncio
    async def test_pure_expansion_does_not_persist(self, scheduler: Scheduler):
        """Pure widening (needed ⊃ current) must NOT call update_task.

        Widening self-heals on restart: the scheduler re-reads the smaller
        metadata.files, re-acquires, then re-expands. Gating the persist on
        stale being non-empty avoids unnecessary MCP round-trips on the
        widening path and leaves test_pure_expansion_keeps_current green.
        """
        lt = scheduler.lock_table
        assert lt.try_acquire('T', ['a/lib.rs'])
        update_task = AsyncMock()
        scheduler.update_task = update_task  # type: ignore[method-assign]

        ok = await scheduler.handle_blast_radius_expansion(
            'T',
            current=['a/lib.rs'],
            needed=['a/lib.rs', 'a/other.rs'],
        )

        assert ok is True
        update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_narrowing_emits_set_to_plan_event(self, scheduler: Scheduler):
        """A shift (lib.rs → conformance.rs) must emit exactly one set_to_plan
        event after the persist.

        Payload must carry:
          - files: the needed (narrowed) set
          - released: the stale modules that were released (lib.rs)
          - acquired: the additional modules acquired (non-empty for a shift —
            here, conformance.rs is newly acquired in the same call)
          - persisted: True when update_task succeeds
        """
        lt = scheduler.lock_table
        assert lt.try_acquire('936', ['crates/reify-compiler/src/lib.rs'])
        backend_md = {
            'memory_hints': {'entities': ['E1']},
            '_causation_id': 'C1',
        }
        scheduler.get_task = AsyncMock(  # type: ignore[method-assign]
            return_value={'id': '936', 'metadata': backend_md}
        )
        scheduler.update_task = AsyncMock(return_value=True)  # type: ignore[method-assign]

        ok = await scheduler.handle_blast_radius_expansion(
            '936',
            current=['crates/reify-compiler/src/lib.rs'],
            needed=['crates/reify-compiler/src/conformance.rs'],
        )

        assert ok is True
        event_store = scheduler.event_store
        assert event_store is not None
        set_to_plan_events = [
            e for e in event_store.events  # type: ignore[attr-defined]
            if e[0] == 'set_to_plan'
        ]
        assert len(set_to_plan_events) == 1, (
            f'Expected exactly one set_to_plan event; got {set_to_plan_events}'
        )
        ev = set_to_plan_events[0]
        assert ev[1]['task_id'] == '936'
        # This is a shift (lib.rs → conformance.rs): additional=['conformance.rs'],
        # stale=['lib.rs'].  The acquired field carries what was actually acquired
        # (additional), which for a shift is non-empty.
        assert ev[1]['data'] == {
            'files': ['crates/reify-compiler/src/conformance.rs'],
            'released': ['crates/reify-compiler/src/lib.rs'],
            'acquired': ['crates/reify-compiler/src/conformance.rs'],
            'persisted': True,
        }, f'set_to_plan event payload mismatch; got {ev[1]["data"]}'

    @pytest.mark.asyncio
    async def test_persist_failure_marks_event_not_persisted(self, scheduler: Scheduler):
        """When update_task returns False, handle_blast_radius_expansion must still
        return True (in-memory narrowing applied) and emit set_to_plan with persisted=False.
        """
        lt = scheduler.lock_table
        assert lt.try_acquire('936', ['crates/reify-compiler/src/lib.rs'])
        scheduler.get_task = AsyncMock(  # type: ignore[method-assign]
            return_value={'id': '936', 'metadata': {}}
        )
        scheduler.update_task = AsyncMock(return_value=False)  # type: ignore[method-assign]

        ok = await scheduler.handle_blast_radius_expansion(
            '936',
            current=['crates/reify-compiler/src/lib.rs'],
            needed=['crates/reify-compiler/src/conformance.rs'],
        )

        assert ok is True
        # In-memory narrowing applied: lib.rs released, conformance.rs held
        assert lt.try_acquire('2035', ['crates/reify-compiler/src/lib.rs'])
        assert not lt.try_acquire('9999', ['crates/reify-compiler/src/conformance.rs'])

        event_store = scheduler.event_store
        assert event_store is not None
        set_to_plan_events = [
            e for e in event_store.events  # type: ignore[attr-defined]
            if e[0] == 'set_to_plan'
        ]
        assert len(set_to_plan_events) == 1
        assert set_to_plan_events[0][1]['data']['persisted'] is False, (
            'persisted must be False when update_task returns False'
        )

    @pytest.mark.asyncio
    async def test_pure_expansion_does_not_emit_set_to_plan(self, scheduler: Scheduler):
        """Pure widening must NOT emit a set_to_plan event (no stale release)."""
        lt = scheduler.lock_table
        assert lt.try_acquire('T', ['a/lib.rs'])

        ok = await scheduler.handle_blast_radius_expansion(
            'T',
            current=['a/lib.rs'],
            needed=['a/lib.rs', 'a/other.rs'],
        )

        assert ok is True
        event_store = scheduler.event_store
        assert event_store is not None
        set_to_plan_events = [
            e for e in event_store.events  # type: ignore[attr-defined]
            if e[0] == 'set_to_plan'
        ]
        assert set_to_plan_events == [], (
            f'Widening must not emit set_to_plan; got {set_to_plan_events}'
        )


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


# ---------------------------------------------------------------------------
# TestGetStatusesFailsLoud (task 1807 — step-1 RED / step-2 GREEN)
# ---------------------------------------------------------------------------

class TestGetStatusesFailsLoud:
    """``Scheduler.get_statuses`` must fail LOUD on non-dict/unparseable results.

    Today the non-dict branch falls through to ``return {}, None`` (err is None),
    silently stranding tasks.  After the fix:
    - Non-dict 'statuses' value (e.g. a list) → ``({}, EnvelopeParseError(...))``
    - 'statuses' key absent → ``({}, EnvelopeParseError(...))``
    - A WARNING is logged naming the failure.
    - The existing exception-raised path (``({}, exception)``) is unchanged.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @staticmethod
    def _envelope(payload: dict) -> dict:
        """Return a JSON-RPC envelope with a single text block."""
        import json as _json
        return {
            'result': {
                'content': [{'type': 'text', 'text': _json.dumps(payload)}]
            }
        }

    @pytest.mark.asyncio
    async def test_non_dict_statuses_returns_error(
        self, scheduler: Scheduler, monkeypatch, caplog
    ):
        """Non-dict 'statuses' value (e.g. a list) → ({}, EnvelopeParseError).

        When the 'statuses' payload is a list (not a dict), the error slot must be
        set.  A WARNING must be logged so the failure is visible in journalctl / caplog.

        Fails today: get_statuses falls through to ``return {}, None`` on non-dict.
        """
        import logging

        from shared.mcp_envelope import EnvelopeParseError as _EnvelopeParseError

        # Response whose 'statuses' is a list, not a dict → wrong type.
        response = self._envelope({'statuses': ['not', 'a', 'dict']})
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=response),
        )

        with caplog.at_level(logging.WARNING):
            statuses, err = await scheduler.get_statuses()

        assert statuses == {}, f'Expected empty dict; got {statuses!r}'
        assert err is not None, (
            'Expected EnvelopeParseError in error slot; got None '
            '(non-dict branch fell through to return {}, None)'
        )
        assert isinstance(err, _EnvelopeParseError), (
            f'Expected EnvelopeParseError; got {type(err).__name__}'
        )
        assert any(
            r.levelno >= logging.WARNING for r in caplog.records
        ), f'Expected a WARNING log; got records={caplog.records!r}'

    @pytest.mark.asyncio
    async def test_absent_statuses_key_returns_error(
        self, scheduler: Scheduler, monkeypatch, caplog
    ):
        """'statuses' key absent from response → ({}, EnvelopeParseError).

        When the response JSON has no 'statuses' key, the error slot must be set,
        not silently returned as ``({}, None)``.
        """
        import logging

        from shared.mcp_envelope import EnvelopeParseError as _EnvelopeParseError

        # Response with no 'statuses' key at all.
        response = self._envelope({'data': 'not-a-statuses-dict'})
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=response),
        )

        with caplog.at_level(logging.WARNING):
            statuses, err = await scheduler.get_statuses()

        assert statuses == {}
        assert err is not None, 'Expected error on absent key; got None'
        assert isinstance(err, _EnvelopeParseError)
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    @pytest.mark.asyncio
    async def test_non_dict_error_leaves_no_state(
        self, scheduler: Scheduler, monkeypatch
    ):
        """Non-dict error leaves no persistent state; next call still works correctly."""
        from shared.mcp_envelope import EnvelopeParseError as _EnvelopeParseError

        # First call: non-dict response.
        response_bad = self._envelope({'no_statuses_here': True})
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=response_bad),
        )
        _stat_bad, err_bad = await scheduler.get_statuses()
        assert err_bad is not None
        assert isinstance(err_bad, _EnvelopeParseError)

        # Second call: correct response — no cross-call state leakage.
        response_ok = self._envelope({'statuses': {'1': 'done'}})
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=response_ok),
        )
        stat_ok, err_ok = await scheduler.get_statuses()
        assert err_ok is None
        assert stat_ok == {'1': 'done'}


class TestFairnessConfigSchema:
    """Schema assertions for FairnessConfig and the scheduler's fairness config."""

    def test_scheduler_v2_field_removed(self):
        """scheduler_v2 must no longer exist on FairnessConfig.

        Regression guard mirroring test_lease_fields_removed_from_fairness_config.
        """
        assert not hasattr(FairnessConfig(), 'scheduler_v2'), (
            'scheduler_v2 field must be removed from FairnessConfig'
        )

    def test_skip_threshold_is_per_tier_dict(self):
        """Default skip_threshold is a per-tier dict."""
        cfg = OrchestratorConfig()
        assert cfg.fairness.skip_threshold == {
            'critical': 0,
            'high': 1,
            'medium': 2,
            'low': 4,
            'polish': 9999,
        }

    def test_lease_fields_removed_from_fairness_config(self):
        """lease_min_secs, lease_max_secs, lease_multiplier, lease_multiplier_for,
        and median_window must no longer exist on FairnessConfig."""
        fc = FairnessConfig()
        assert not hasattr(fc, 'lease_min_secs'), 'lease_min_secs must be removed'
        assert not hasattr(fc, 'lease_max_secs'), 'lease_max_secs must be removed'
        assert not hasattr(fc, 'lease_multiplier'), 'lease_multiplier must be removed'
        assert not hasattr(fc, 'lease_multiplier_for'), (
            'lease_multiplier_for must be removed'
        )
        assert not hasattr(fc, 'median_window'), 'median_window must be removed'

    def test_tier_slot_caps_removed_from_orchestrator_config(self):
        """tier_slot_caps, tier_slot_cap_for, and tier_slot_limit must not exist
        on OrchestratorConfig."""
        cfg = OrchestratorConfig()
        assert not hasattr(cfg, 'tier_slot_caps'), 'tier_slot_caps must be removed'
        assert not hasattr(cfg, 'tier_slot_cap_for'), (
            'tier_slot_cap_for must be removed'
        )
        assert not hasattr(cfg, 'tier_slot_limit'), 'tier_slot_limit must be removed'

    def test_reservation_evicted_event_type_exists(self):
        """EventType.reservation_evicted must be defined for cross-tier preemption."""
        assert hasattr(EventType, 'reservation_evicted')
        assert EventType.reservation_evicted == 'reservation_evicted'


class TestReserveNowShortCircuit:
    """reserve_now=True installs parks at the top of acquire_next and auto-clears."""

    @pytest.mark.asyncio
    async def test_reserve_now_installs_parks_and_clears_field(self, tmp_path):
        """reserve_now fires install_parks for the task then clears the flag.

        A's modules are pre-held by a 'seed' task so A cannot be dispatched in
        the normal scored loop, ensuring parks survive the tick (they would
        otherwise be cleared when A dispatches successfully as top candidate).
        B uses different modules and IS dispatched.
        """
        from orchestrator.overrides import OverrideStore

        config = OrchestratorConfig(max_per_module=1)
        store = OverrideStore(tmp_path / 'o.db')
        store.set_override('/proj', 'A', reserve_now=True)

        scheduler = Scheduler(config, override_store=store)
        scheduler._project_root = '/proj'

        # Pre-hold A's modules so A cannot acquire them this tick.
        scheduler.lock_table._held['seed'] = {'compiler/src', 'eval/src'}
        scheduler._dispatched.add('seed')  # seed is treated as already dispatched

        task_a = {
            'id': 'A',
            'title': 'Task A',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['compiler/src', 'eval/src']},
            'priority': 'medium',
        }
        task_b = {
            'id': 'B',
            'title': 'Task B',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['other/module']},
            'priority': 'medium',
        }
        scheduler.get_tasks = AsyncMock(return_value=[task_a, task_b])

        result = await scheduler.acquire_next()

        # (a) A must have parks installed — A was blocked so parks were NOT cleared.
        assert scheduler.lock_table.has_parks('A')

        # (b) The parks must cover A's modules (look inside _parked).
        # _parked is now a dict[str, list[tuple[str,int]]] (LIFO stack);
        # the active-top owner is the LAST entry in each stack.
        parked_owners = {
            m: stack[-1][0] for m, stack in scheduler.lock_table._parked.items()
        }
        assert 'compiler/src' in parked_owners
        assert parked_owners['compiler/src'] == 'A'
        assert 'eval/src' in parked_owners
        assert parked_owners['eval/src'] == 'A'

        # (c) reserve_now flag must be cleared in the store.
        overrides = store.get_overrides('/proj')
        if 'A' in overrides:
            assert overrides['A'].reserve_now is False

        # (d) B was dispatched since A's modules were locked out.
        assert result is not None
        assert result.task_id == 'B'

    @pytest.mark.asyncio
    async def test_install_parks_failure_restores_reserve_now_and_logs(
        self, tmp_path, monkeypatch, caplog
    ):
        """install_parks failure restores reserve_now flag and logs a WARNING.

        On an in-process RuntimeError from install_parks:
        (a) acquire_next() does NOT propagate the exception;
        (b) reserve_now is restored to True in the DB;
        (c) a WARNING with the task id and a traceback is logged;
        (d) no reservation_installed event is emitted.
        """
        import logging as _logging

        from orchestrator.overrides import OverrideStore

        config = OrchestratorConfig(max_per_module=1)
        store = OverrideStore(tmp_path / 'o.db')
        store.set_override('/proj', 'A', reserve_now=True)

        recording_store = _RecordingEventStore()
        scheduler = Scheduler(config, override_store=store, event_store=recording_store)  # type: ignore[arg-type]
        scheduler._project_root = '/proj'

        task_a = {
            'id': 'A',
            'title': 'Task A',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['compiler/src']},
            'priority': 'medium',
        }
        scheduler.get_tasks = AsyncMock(return_value=[task_a])

        def _raise(rid, modules, tier):
            raise RuntimeError('boom')

        monkeypatch.setattr(scheduler.lock_table, 'install_parks', _raise)

        with caplog.at_level(_logging.WARNING, logger='orchestrator.scheduler'):
            # (a) Must NOT propagate the RuntimeError.
            await scheduler.acquire_next()

        # (b) reserve_now flag must be restored in the DB.
        overrides = store.get_overrides('/proj')
        assert 'A' in overrides, 'Override row for A must exist after restore'
        assert overrides['A'].reserve_now is True, (
            'reserve_now must be True after install_parks failure'
        )

        # (c) A WARNING mentioning task id 'A' with a traceback must be logged.
        warnings = [r for r in caplog.records if r.levelno == _logging.WARNING]
        assert warnings, (
            f'Expected a WARNING log. Records: {[r.getMessage() for r in caplog.records]}'
        )
        warning_messages = ' '.join(r.getMessage() for r in warnings)
        assert 'task A' in warning_messages, (
            f'WARNING must mention task id A. Got: {warning_messages!r}'
        )
        assert 'Traceback' in caplog.text, (
            'WARNING must include traceback (exc_info=True). caplog.text:\n' + caplog.text
        )

        # (d) No reservation_installed event must be emitted.
        emitted_types = [ev_type for ev_type, _ in recording_store.events]
        assert EventType.reservation_installed.value not in emitted_types, (
            f'Unexpected reservation_installed event was emitted: {recording_store.events}'
        )

    @pytest.mark.asyncio
    async def test_install_parks_failure_and_restore_failure_no_exception_no_event(
        self, tmp_path, monkeypatch, caplog
    ):
        """When both install_parks and the restore set_override raise, acquire_next()
        still returns cleanly with a WARNING and no reservation_installed event.

        This exercises the inner-except defensive branch added alongside the main
        try/except handler.
        """
        import logging as _logging

        from orchestrator.overrides import OverrideStore

        config = OrchestratorConfig(max_per_module=1)
        store = OverrideStore(tmp_path / 'o.db')
        store.set_override('/proj', 'A', reserve_now=True)

        recording_store = _RecordingEventStore()
        scheduler = Scheduler(config, override_store=store, event_store=recording_store)  # type: ignore[arg-type]
        scheduler._project_root = '/proj'

        task_a = {
            'id': 'A',
            'title': 'Task A',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['compiler/src']},
            'priority': 'medium',
        }
        scheduler.get_tasks = AsyncMock(return_value=[task_a])

        def _raise_install(rid, modules, tier):
            raise RuntimeError('install boom')

        def _raise_set(*args, **kwargs):
            raise RuntimeError('restore boom')

        monkeypatch.setattr(scheduler.lock_table, 'install_parks', _raise_install)
        monkeypatch.setattr(scheduler._override_store, 'set_override', _raise_set)

        with caplog.at_level(_logging.WARNING, logger='orchestrator.scheduler'):
            # Must NOT propagate either RuntimeError.
            await scheduler.acquire_next()

        # Both the outer install_parks-failure warning and the inner restore-failure
        # warning must have been emitted, so that removing the inner-except branch
        # in Scheduler.acquire_next()'s reserve_now install_parks handler would
        # cause this test to fail.
        warnings = [r for r in caplog.records if r.levelno == _logging.WARNING]
        assert len(warnings) >= 2, (
            f'Expected at least 2 WARNING log(s). Records: {[r.getMessage() for r in caplog.records]}'
        )

        warning_messages = [r.getMessage() for r in warnings]
        outer_warn = any('install_parks failed' in m for m in warning_messages)
        assert outer_warn, (
            f'Expected an outer install_parks-failure WARNING. Got: {warning_messages!r}'
        )
        inner_warn = any('failed to restore reserve_now flag' in m for m in warning_messages)
        assert inner_warn, (
            f'Expected an inner restore-failure WARNING. Got: {warning_messages!r}'
        )

        # No reservation_installed event must be emitted.
        emitted_types = [ev_type for ev_type, _ in recording_store.events]
        assert EventType.reservation_installed.value not in emitted_types, (
            f'Unexpected reservation_installed event was emitted: {recording_store.events}'
        )


class TestSchedulerOverrideStoreInjection:
    """Scheduler accepts an optional OverrideStore kwarg; when None, behaves as today."""

    def test_init_accepts_override_store_kwarg_default_none(self):
        """Scheduler(config) sets _override_store=None by default."""
        config = OrchestratorConfig(max_per_module=1)
        scheduler = Scheduler(config)
        assert scheduler._override_store is None

    def test_init_accepts_explicit_override_store(self, tmp_path):
        """Scheduler(config, override_store=...) stores the instance."""
        from orchestrator.overrides import OverrideStore

        config = OrchestratorConfig(max_per_module=1)
        store = OverrideStore(tmp_path / 'overrides.db')
        scheduler = Scheduler(config, override_store=store)
        assert scheduler._override_store is store

    @pytest.mark.asyncio
    async def test_acquire_next_queries_override_store_when_present(self, tmp_path):
        """acquire_next() runs without error when an override_store is wired in."""
        from orchestrator.overrides import OverrideStore

        config = OrchestratorConfig(max_per_module=1)
        store = OverrideStore(tmp_path / 'overrides.db')
        # Set a boost on task A so the override code path is exercised.
        store.set_override('/proj', 'A', boost_tier='critical')

        scheduler = Scheduler(config, override_store=store)
        scheduler._project_root = '/proj'

        task_a = {
            'id': 'A',
            'title': 'Task A',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['mod_a']},
            'priority': 'medium',
        }
        scheduler.get_tasks = AsyncMock(return_value=[task_a])
        result = await scheduler.acquire_next()
        # A should be dispatched (no competing tasks, no lock conflicts).
        assert result is not None
        assert result.task_id == 'A'


class TestSchedulerOverrideRestartSemantics:
    """First-tick snapshot seeding: no spurious events on scheduler restart."""

    @pytest.mark.asyncio
    async def test_first_tick_does_not_emit_spurious_events_for_preexisting_overrides(
        self, tmp_path
    ):
        """On scheduler restart with pre-existing overrides, the first tick must not
        emit priority_override_set / task_pinned events for already-persisted rows.

        Those rows represent state that was already known before the restart — not
        fresh user actions.  Emitting them would spam downstream consumers and
        confuse them into treating a restart as a batch of new user commands.
        """
        from orchestrator.event_store import EventType
        from orchestrator.overrides import OverrideStore

        config = OrchestratorConfig(max_per_module=1)
        event_store = _RecordingEventStore()
        store = OverrideStore(tmp_path / 'o.db')

        # Simulate state that existed BEFORE the scheduler started (e.g. written
        # by a previous process or by the MCP tool while the scheduler was down).
        store.set_override('/proj', 'A', boost_tier='high')
        store.set_override('/proj', 'B', pinned=True)

        scheduler = Scheduler(config, override_store=store, event_store=event_store)  # type: ignore[arg-type]
        scheduler._project_root = '/proj'

        task_a = _pending_task('A', priority='medium', files=['a/src'])
        task_b = _pending_task('B', priority='medium', files=['b/src'])

        # Lock both modules so no task can dispatch (focus on event semantics).
        scheduler.lock_table._held['seed'] = {'a/src', 'b/src'}
        scheduler._dispatched.add('seed')

        scheduler.get_tasks = AsyncMock(return_value=[task_a, task_b])
        await scheduler.acquire_next()  # first tick — snapshot seeded

        override_event_types = {
            EventType.priority_override_set.value,
            EventType.priority_override_cleared.value,
            EventType.task_pinned.value,
            EventType.task_unpinned.value,
            EventType.pin_queue_reordered.value,
        }
        spurious = [e for e in event_store.events if e[0] in override_event_types]
        assert spurious == [], (
            f'First tick must not emit override events for pre-existing rows; '
            f'got: {spurious}'
        )

        # Second tick: change a boost → that change MUST emit an event.
        store.set_override('/proj', 'A', boost_tier='critical')
        await scheduler.acquire_next()

        set_events = [
            e for e in event_store.events
            if e[0] == EventType.priority_override_set.value
        ]
        assert len(set_events) == 1, (
            f'Second tick after a boost change should emit priority_override_set; '
            f'got: {set_events}'
        )
        assert set_events[0][1]['task_id'] == 'A'
        assert set_events[0][1]['data']['boost_tier'] == 'critical'


class TestPinDispatch:
    """Pinned tasks dispatch ahead of all scored candidates, bypassing fairness."""

    @pytest.mark.asyncio
    async def test_pin_dispatches_ahead_of_higher_scored_candidate(self, tmp_path):
        """A pinned low-priority task wins dispatch over an unpin critical-priority task.

        Without pin: B (critical) would score higher and be dispatched first.
        With pin on A: A must dispatch even though B has a far higher score.
        """
        from orchestrator.overrides import OverrideStore

        config = OrchestratorConfig(max_per_module=1)
        store = OverrideStore(tmp_path / 'o.db')
        # Pin A (low priority) — it should dispatch ahead of B (critical).
        store.set_override('/proj', 'A', pinned=True)

        scheduler = Scheduler(config, override_store=store)
        scheduler._project_root = '/proj'

        task_a = {
            'id': 'A',
            'title': 'Task A (pinned, low)',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['x/src']},
            'priority': 'low',
        }
        task_b = {
            'id': 'B',
            'title': 'Task B (critical, not pinned)',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['y/src']},
            'priority': 'critical',
        }
        scheduler.get_tasks = AsyncMock(return_value=[task_a, task_b])

        result = await scheduler.acquire_next()

        # Pinned A must dispatch first, even though B scores higher due to critical priority.
        assert result is not None
        assert result.task_id == 'A'

    @pytest.mark.asyncio
    async def test_multiple_pins_dispatch_in_pin_order_and_lockout_falls_through(
        self, tmp_path
    ):
        """When the first pinned task is locked out, scheduler falls through to next pin.

        A is pinned pin_order=1 but its module (compiler/src) is already held by 'seed'.
        B is pinned pin_order=2 with a free module (eval/src).
        Result: B is dispatched.  A accumulates NO skip bookkeeping.
        """
        from orchestrator.event_store import EventType
        from orchestrator.overrides import OverrideStore

        config = OrchestratorConfig(max_per_module=1)
        event_store = _RecordingEventStore()
        store = OverrideStore(tmp_path / 'o.db')
        # Pin A first (auto pin_order=1), then B (auto pin_order=2).
        store.set_override('/proj', 'A', pinned=True)
        store.set_override('/proj', 'B', pinned=True)

        scheduler = Scheduler(config, override_store=store, event_store=event_store)  # type: ignore[arg-type]
        scheduler._project_root = '/proj'

        # Pre-hold A's module so A cannot acquire it this tick.
        scheduler.lock_table._held['seed'] = {'compiler/src'}
        scheduler._dispatched.add('seed')

        task_a = {
            'id': 'A',
            'title': 'Task A (pinned 1, locked out)',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['compiler/src']},
            'priority': 'medium',
        }
        task_b = {
            'id': 'B',
            'title': 'Task B (pinned 2, free)',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['eval/src']},
            'priority': 'medium',
        }
        scheduler.get_tasks = AsyncMock(return_value=[task_a, task_b])

        result = await scheduler.acquire_next()

        # A was locked out → fall through to B (next pinned candidate).
        assert result is not None
        assert result.task_id == 'B'

        # A must have accumulated NO skip bookkeeping.
        assert scheduler._skip_count.get('A', 0) == 0

        # No task_skipped event must have been emitted for A.
        skipped_for_a = [
            e for e in event_store.events
            if e[0] == EventType.task_skipped.value and e[1].get('task_id') == 'A'
        ]
        assert skipped_for_a == [], f'Unexpected task_skipped events for A: {skipped_for_a}'


class TestOverrideGCIntegration:
    """GC sweep clears overrides for terminal tasks and expired TTLs."""

    @pytest.mark.asyncio
    async def test_park_gc_pass_calls_clear_terminal_for_terminal_owners(
        self, tmp_path
    ):
        """acquire_next() GC sweep removes overrides for done/cancelled tasks.

        A = pending (boost override must survive).
        B = done    (override must be removed by GC).
        C = cancelled (override must be removed by GC).
        """
        from orchestrator.overrides import OverrideStore

        config = OrchestratorConfig(max_per_module=1)
        store = OverrideStore(tmp_path / 'o.db')
        store.set_override('/proj', 'A', boost_tier='high')
        store.set_override('/proj', 'B', boost_tier='critical')
        store.set_override('/proj', 'C', boost_tier='medium')

        scheduler = Scheduler(config, override_store=store)
        scheduler._project_root = '/proj'

        task_a = {
            'id': 'A',
            'title': 'Task A',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['mod_a']},
            'priority': 'medium',
        }
        task_b = {
            'id': 'B',
            'title': 'Task B',
            'status': 'done',
            'dependencies': [],
            'metadata': {'files': ['mod_b']},
            'priority': 'medium',
        }
        task_c = {
            'id': 'C',
            'title': 'Task C',
            'status': 'cancelled',
            'dependencies': [],
            'metadata': {'files': ['mod_c']},
            'priority': 'medium',
        }
        scheduler.get_tasks = AsyncMock(return_value=[task_a, task_b, task_c])

        await scheduler.acquire_next()

        overrides_after = store.get_overrides('/proj')
        # A (pending) still has its override.
        assert 'A' in overrides_after
        # B (done) and C (cancelled) must have been swept.
        assert 'B' not in overrides_after
        assert 'C' not in overrides_after

    @pytest.mark.asyncio
    async def test_ttl_sweep_calls_clear_expired(self, tmp_path):
        """acquire_next() GC sweep removes overrides whose TTL has elapsed.

        A has a past TTL → must be cleared.
        B has a future TTL → must survive.
        """
        from datetime import datetime, timedelta

        from orchestrator.overrides import OverrideStore

        now_dt = datetime.now(UTC)

        config = OrchestratorConfig(max_per_module=1)
        store = OverrideStore(tmp_path / 'o.db')
        # A expired 1 hour ago.
        store.set_override('/proj', 'A', boost_tier='high', ttl_until=now_dt - timedelta(hours=1))
        # B expires 1 hour from now.
        store.set_override('/proj', 'B', boost_tier='critical', ttl_until=now_dt + timedelta(hours=1))

        scheduler = Scheduler(config, override_store=store)
        scheduler._project_root = '/proj'

        task_a = {
            'id': 'A',
            'title': 'Task A (expired TTL)',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['mod_a']},
            'priority': 'medium',
        }
        task_b = {
            'id': 'B',
            'title': 'Task B (future TTL)',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['mod_b']},
            'priority': 'medium',
        }
        scheduler.get_tasks = AsyncMock(return_value=[task_a, task_b])

        await scheduler.acquire_next()

        overrides_after = store.get_overrides('/proj')
        # A's TTL has elapsed — must be swept.
        assert 'A' not in overrides_after
        # B's TTL is in the future — must survive.
        assert 'B' in overrides_after


class TestOverrideEventEmission:
    """Scheduler emits priority_override_* events via per-tick diff-detection."""

    @pytest.mark.asyncio
    async def test_priority_override_set_and_cleared_diff_events(self, tmp_path):
        """Diff-detect fires priority_override_set when boost appears, cleared when it goes.

        Tick 1: no override → no event.
        Tick 2: boost='high' set on A → priority_override_set emitted.
        Tick 3: boost cleared → priority_override_cleared emitted.
        """
        from orchestrator.event_store import EventType
        from orchestrator.overrides import OverrideStore

        config = OrchestratorConfig(max_per_module=1)
        event_store = _RecordingEventStore()
        store = OverrideStore(tmp_path / 'o.db')

        scheduler = Scheduler(config, override_store=store, event_store=event_store)  # type: ignore[arg-type]
        scheduler._project_root = '/proj'

        task_a = {
            'id': 'A',
            'title': 'Task A',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['mod_a']},
            'priority': 'medium',
        }
        scheduler.get_tasks = AsyncMock(return_value=[task_a])

        # Tick 1: no override set → no priority_override_* events.
        await scheduler.acquire_next()
        scheduler.release('A')
        override_events_after_tick1 = [
            e for e in event_store.events
            if e[0].startswith('priority_override_')
        ]
        assert override_events_after_tick1 == [], (
            f'Expected no override events after tick 1, got: {override_events_after_tick1}'
        )

        # Tick 2: set boost='high' on A → priority_override_set must fire.
        store.set_override('/proj', 'A', boost_tier='high')
        await scheduler.acquire_next()
        scheduler.release('A')
        override_events_after_tick2 = [
            e for e in event_store.events
            if e[0] == EventType.priority_override_set.value
        ]
        assert len(override_events_after_tick2) == 1, (
            f'Expected 1 priority_override_set event, got: {override_events_after_tick2}'
        )
        ev2 = override_events_after_tick2[0]
        assert ev2[1]['task_id'] == 'A'
        assert ev2[1]['data'].get('boost_tier') == 'high'

        # Tick 3: clear boost → priority_override_cleared must fire.
        store.clear_override('/proj', 'A', field='boost_tier')
        await scheduler.acquire_next()
        scheduler.release('A')
        override_events_after_tick3 = [
            e for e in event_store.events
            if e[0] == EventType.priority_override_cleared.value
        ]
        assert len(override_events_after_tick3) == 1, (
            f'Expected 1 priority_override_cleared event, got: {override_events_after_tick3}'
        )
        ev3 = override_events_after_tick3[0]
        assert ev3[1]['task_id'] == 'A'

    @pytest.mark.asyncio
    async def test_pin_unpin_and_reorder_events(self, tmp_path):
        """Diff-detect fires task_pinned, task_unpinned, and pin_queue_reordered events.

        Tick 1: pin A → task_pinned for A (pin_order=1).
        Tick 2: pin B → task_pinned for B (pin_order=2).
        Tick 3: reorder [B, A] → pin_queue_reordered with new_order=['B','A'].
        Tick 4: unpin A → task_unpinned for A.
        """
        from orchestrator.event_store import EventType
        from orchestrator.overrides import OverrideStore

        config = OrchestratorConfig(max_per_module=1)
        event_store = _RecordingEventStore()
        store = OverrideStore(tmp_path / 'o.db')

        scheduler = Scheduler(config, override_store=store, event_store=event_store)  # type: ignore[arg-type]
        scheduler._project_root = '/proj'

        # Pre-hold all modules so no task can dispatch (focus is purely on events).
        scheduler.lock_table._held['seed'] = {'a/src', 'b/src'}
        scheduler._dispatched.add('seed')

        task_a = {
            'id': 'A', 'title': 'Task A', 'status': 'pending',
            'dependencies': [], 'metadata': {'files': ['a/src']}, 'priority': 'medium',
        }
        task_b = {
            'id': 'B', 'title': 'Task B', 'status': 'pending',
            'dependencies': [], 'metadata': {'files': ['b/src']}, 'priority': 'medium',
        }

        def events_of_type(et):
            return [e for e in event_store.events if e[0] == et.value]

        # Seed tick: initialise the override snapshot so that the subsequent
        # ticks can diff-detect changes.  On the very first tick the scheduler
        # skips diff-emit (Suggestion 3 — restart semantics) and seeds the
        # snapshot as empty.  All real assertions start from tick 1 onwards.
        scheduler.get_tasks = AsyncMock(return_value=[task_a, task_b])
        await scheduler.acquire_next()
        assert events_of_type(EventType.task_pinned) == [], 'Seed tick must emit no events'

        # Tick 1: pin A (auto pin_order=1)
        store.set_override('/proj', 'A', pinned=True)
        scheduler.get_tasks = AsyncMock(return_value=[task_a])
        await scheduler.acquire_next()

        pinned_after_t1 = events_of_type(EventType.task_pinned)
        assert len(pinned_after_t1) == 1, f'Expected 1 task_pinned after tick 1, got {pinned_after_t1}'
        assert pinned_after_t1[0][1]['task_id'] == 'A'
        assert pinned_after_t1[0][1]['data'].get('pin_order') == 1

        # Tick 2: pin B (auto pin_order=2)
        store.set_override('/proj', 'B', pinned=True)
        scheduler.get_tasks = AsyncMock(return_value=[task_a, task_b])
        await scheduler.acquire_next()

        pinned_after_t2 = events_of_type(EventType.task_pinned)
        assert len(pinned_after_t2) == 2, f'Expected 2 task_pinned events after tick 2, got {pinned_after_t2}'
        b_pinned = [e for e in pinned_after_t2 if e[1]['task_id'] == 'B']
        assert len(b_pinned) == 1
        assert b_pinned[0][1]['data'].get('pin_order') == 2

        # Tick 3: reorder to [B, A]
        store.reorder_pin_queue('/proj', ['B', 'A'])
        await scheduler.acquire_next()

        reorder_events = events_of_type(EventType.pin_queue_reordered)
        assert len(reorder_events) == 1, f'Expected 1 pin_queue_reordered event, got {reorder_events}'
        assert reorder_events[0][1]['data'].get('new_order') == ['B', 'A']

        # Tick 4: unpin A
        store.clear_override('/proj', 'A', field='pinned')
        await scheduler.acquire_next()

        unpinned_events = events_of_type(EventType.task_unpinned)
        assert len(unpinned_events) == 1, f'Expected 1 task_unpinned event, got {unpinned_events}'
        assert unpinned_events[0][1]['task_id'] == 'A'

    @pytest.mark.asyncio
    async def test_adding_pin_does_not_emit_pin_queue_reordered(self, tmp_path):
        """Adding a new pin fires task_pinned but NOT pin_queue_reordered.

        pin_queue_reordered is reserved for pure reorder operations (where
        existing pinned tasks shift position).  Pinning a new task is already
        fully described by task_pinned; emitting pin_queue_reordered on top
        would be redundant noise for downstream consumers.
        """
        from orchestrator.event_store import EventType
        from orchestrator.overrides import OverrideStore

        config = OrchestratorConfig(max_per_module=1)
        event_store = _RecordingEventStore()
        store = OverrideStore(tmp_path / 'o.db')

        scheduler = Scheduler(config, override_store=store, event_store=event_store)  # type: ignore[arg-type]
        scheduler._project_root = '/proj'
        scheduler.lock_table._held['seed'] = {'a/src', 'b/src'}
        scheduler._dispatched.add('seed')

        task_a = _pending_task('A', priority='medium', files=['a/src'])
        task_b = _pending_task('B', priority='medium', files=['b/src'])

        # Seed tick: initialise snapshot (no overrides yet).
        scheduler.get_tasks = AsyncMock(return_value=[task_a, task_b])
        await scheduler.acquire_next()

        # Pin A (first pin) — must emit task_pinned but NOT pin_queue_reordered.
        store.set_override('/proj', 'A', pinned=True)
        await scheduler.acquire_next()

        reorder_after_add = [
            e for e in event_store.events
            if e[0] == EventType.pin_queue_reordered.value
        ]
        assert reorder_after_add == [], (
            'Adding a new pin must not emit pin_queue_reordered; '
            f'got: {reorder_after_add}'
        )
        pinned_events = [
            e for e in event_store.events if e[0] == EventType.task_pinned.value
        ]
        assert len(pinned_events) == 1
        assert pinned_events[0][1]['task_id'] == 'A'

        # Pin B (second pin) — again must emit task_pinned but NOT pin_queue_reordered.
        store.set_override('/proj', 'B', pinned=True)
        scheduler.get_tasks = AsyncMock(return_value=[task_a, task_b])
        await scheduler.acquire_next()

        reorder_after_second_add = [
            e for e in event_store.events
            if e[0] == EventType.pin_queue_reordered.value
        ]
        assert reorder_after_second_add == [], (
            'Adding a second pin must not emit pin_queue_reordered'
        )

    @pytest.mark.asyncio
    async def test_unpinning_does_not_emit_pin_queue_reordered(self, tmp_path):
        """Removing a pin fires task_unpinned but NOT pin_queue_reordered.

        The pin removal changes the effective queue order (remaining tasks may
        implicitly shift), but the change is already described by task_unpinned.
        Consumers that need the updated order can re-query get_pin_queue.
        """
        from orchestrator.event_store import EventType
        from orchestrator.overrides import OverrideStore

        config = OrchestratorConfig(max_per_module=1)
        event_store = _RecordingEventStore()
        store = OverrideStore(tmp_path / 'o.db')

        scheduler = Scheduler(config, override_store=store, event_store=event_store)  # type: ignore[arg-type]
        scheduler._project_root = '/proj'
        scheduler.lock_table._held['seed'] = {'a/src', 'b/src'}
        scheduler._dispatched.add('seed')

        task_a = _pending_task('A', priority='medium', files=['a/src'])
        task_b = _pending_task('B', priority='medium', files=['b/src'])

        # Set up A and B as pinned, then seed the snapshot so both appear as
        # pre-existing (no spurious events on the seed tick).
        store.set_override('/proj', 'A', pinned=True, pin_order=1)
        store.set_override('/proj', 'B', pinned=True, pin_order=2)
        scheduler.get_tasks = AsyncMock(return_value=[task_a, task_b])
        await scheduler.acquire_next()  # seed tick — no events expected

        # Unpin A → must emit task_unpinned but NOT pin_queue_reordered.
        store.clear_override('/proj', 'A', field='pinned')
        await scheduler.acquire_next()

        reorder_after_unpin = [
            e for e in event_store.events
            if e[0] == EventType.pin_queue_reordered.value
        ]
        assert reorder_after_unpin == [], (
            'Unpinning must not emit pin_queue_reordered; '
            f'got: {reorder_after_unpin}'
        )
        unpinned_events = [
            e for e in event_store.events if e[0] == EventType.task_unpinned.value
        ]
        assert len(unpinned_events) == 1
        assert unpinned_events[0][1]['task_id'] == 'A'

    @pytest.mark.asyncio
    async def test_pin_order_recomposition_contract(self, tmp_path):
        """Executable contract: event-only recomposition equals OverrideStore.get_pin_queue.

        Drives a mixed sequence through acquire_next():
          seed → pin A → pin B → reorder [B,A] → unpin A → pin C

        Asserts three invariants of the decided Path-A contract (task 1290):

        (a) NO pin_queue_reordered event is emitted on add ticks (pin A, pin B,
            pin C) or the remove tick (unpin A) — the change is already fully
            described by task_pinned / task_unpinned.
        (b) EXACTLY ONE pin_queue_reordered event is emitted across the whole
            sequence — on the pure-reorder tick (reorder [B,A]) — and its
            data['new_order'] carries the complete post-reorder ordering ['B','A'].
        (c) Recomposing pin order solely from the three event types —
            task_pinned (append), task_unpinned (remove), pin_queue_reordered
            (replace with new_order) — applied in emission order produces the
            same ordering as OverrideStore.get_pin_queue('/proj').  This is the
            executable proof that the documented consumer strategy is correct.

        This test asserts on emitted events and snapshot output only (pure
        runtime behavior).  It never inspects docstrings or comment text.
        Net-new coverage: existing tests assert pin events individually but
        never the recomposition-equals-snapshot invariant.
        """
        from orchestrator.event_store import EventType
        from orchestrator.overrides import OverrideStore

        config = OrchestratorConfig(max_per_module=1)
        event_store = _RecordingEventStore()
        store = OverrideStore(tmp_path / 'o.db')

        scheduler = Scheduler(config, override_store=store, event_store=event_store)  # type: ignore[arg-type]
        scheduler._project_root = '/proj'

        # Pre-hold all module files so nothing dispatches — focus is on events.
        scheduler.lock_table._held['seed'] = {'a/src', 'b/src', 'c/src'}
        scheduler._dispatched.add('seed')

        task_a = _pending_task('A', priority='medium', files=['a/src'])
        task_b = _pending_task('B', priority='medium', files=['b/src'])
        task_c = _pending_task('C', priority='medium', files=['c/src'])

        scheduler.get_tasks = AsyncMock(return_value=[task_a, task_b, task_c])

        def count_events(et: EventType) -> int:
            return sum(1 for e in event_store.events if e[0] == et.value)

        # Seed tick: initialise the override snapshot; no overrides exist yet
        # so no events should fire.  Subsequent ticks diff-detect from this
        # baseline.
        await scheduler.acquire_next()
        assert count_events(EventType.pin_queue_reordered) == 0, \
            'Seed tick must not emit pin_queue_reordered'

        # --- Tick 1: pin A (add) ---
        store.set_override('/proj', 'A', pinned=True)
        await scheduler.acquire_next()
        # (a) No pin_queue_reordered on add
        assert count_events(EventType.pin_queue_reordered) == 0, \
            'Pinning A must not emit pin_queue_reordered'
        assert count_events(EventType.task_pinned) == 1, \
            'Pinning A must emit exactly 1 task_pinned'

        # --- Tick 2: pin B (add) ---
        store.set_override('/proj', 'B', pinned=True)
        await scheduler.acquire_next()
        # (a) Still no pin_queue_reordered after second add
        assert count_events(EventType.pin_queue_reordered) == 0, \
            'Pinning B must not emit pin_queue_reordered'
        assert count_events(EventType.task_pinned) == 2, \
            'Pinning B must bring total task_pinned count to 2'

        # --- Tick 3: reorder [B, A] (pure reorder — both already pinned) ---
        store.reorder_pin_queue('/proj', ['B', 'A'])
        await scheduler.acquire_next()
        # (b) Exactly one pin_queue_reordered with full new_order
        reorder_events = [
            e for e in event_store.events
            if e[0] == EventType.pin_queue_reordered.value
        ]
        assert len(reorder_events) == 1, (
            f'Expected exactly 1 pin_queue_reordered on the reorder tick; '
            f'got {len(reorder_events)}: {reorder_events}'
        )
        assert reorder_events[0][1]['data']['new_order'] == ['B', 'A'], (
            f"new_order must be ['B','A']; "
            f"got {reorder_events[0][1]['data']['new_order']!r}"
        )

        # --- Tick 4: unpin A (remove) ---
        store.clear_override('/proj', 'A', field='pinned')
        await scheduler.acquire_next()
        # (a) No additional pin_queue_reordered on remove
        assert count_events(EventType.pin_queue_reordered) == 1, \
            'Unpinning A must not emit additional pin_queue_reordered'
        assert count_events(EventType.task_unpinned) == 1, \
            'Unpinning A must emit exactly 1 task_unpinned'

        # --- Tick 5: pin C (add) ---
        store.set_override('/proj', 'C', pinned=True)
        await scheduler.acquire_next()
        # (a) No additional pin_queue_reordered on add
        assert count_events(EventType.pin_queue_reordered) == 1, \
            'Pinning C must not emit additional pin_queue_reordered'
        assert count_events(EventType.task_pinned) == 3, \
            'Pinning C must bring total task_pinned count to 3'

        # --- (c) Recompose pin order from events, compare to authoritative snapshot ---
        # Apply events in emission order using the documented consumer strategy:
        #   task_pinned        → append task_id to the ordered list
        #   task_unpinned      → remove task_id from the ordered list
        #   pin_queue_reordered → replace list with data['new_order']
        recomposed: list[str] = []
        for etype, payload in event_store.events:
            if etype == EventType.task_pinned.value:
                recomposed.append(payload['task_id'])
            elif etype == EventType.task_unpinned.value:
                tid = payload['task_id']
                if tid in recomposed:
                    recomposed.remove(tid)
            elif etype == EventType.pin_queue_reordered.value:
                recomposed = list(payload['data']['new_order'])

        authoritative = [tid for tid, _ in store.get_pin_queue('/proj')]

        assert recomposed == authoritative, (
            f'Event-only recomposition {recomposed!r} must equal '
            f'OverrideStore.get_pin_queue {authoritative!r}'
        )
        # Sanity-check the expected final state: B and C pinned (A was removed).
        assert authoritative == ['B', 'C'], (
            f"Expected final authoritative pin order ['B','C']; "
            f'got {authoritative!r}'
        )


class TestRequeueCooldownGc:
    """Unit tests for the _gc_expired_cooldowns helper and the now-pure
    _eligible_for_dispatch predicate, plus an integration test that pins
    acquire_next as the GC owner."""

    def test_gc_expired_cooldowns_removes_only_expired(self):
        """_gc_expired_cooldowns must remove entries whose deadline <= now
        (past *and* exactly-at-now boundary) and leave future entries intact.

        Uses a controllable time_source so the test is deterministic.
        This test will fail until _gc_expired_cooldowns is implemented.
        """
        now = 1_000_000.0
        config = OrchestratorConfig(max_per_module=1, requeue_cooldown_secs=30.0)
        scheduler = Scheduler(config, time_source=lambda: now)

        # Seed three entries:
        # 'past'     — deadline in the past → should be removed
        # 'boundary' — deadline exactly at now → treated as expired (>= semantics)
        # 'future'   — deadline in the future → must survive
        scheduler._requeue_until['past'] = now - 1.0
        scheduler._requeue_until['boundary'] = now
        scheduler._requeue_until['future'] = now + 30.0

        scheduler._gc_expired_cooldowns()

        assert scheduler._requeue_until == {'future': now + 30.0}, (
            f'Expected only the future entry to survive GC; '
            f'got: {scheduler._requeue_until}'
        )

    def test_eligible_for_dispatch_does_not_mutate_requeue_until(self):
        """_eligible_for_dispatch must be a pure predicate: calling it with an
        expired entry in _requeue_until must NOT delete that entry.

        The per-tick GC (_gc_expired_cooldowns) is the only place allowed to
        mutate _requeue_until.  This test will fail on current code because
        _eligible_for_dispatch still contains `del self._requeue_until[tid]`.
        """
        now = 1_000_000.0
        config = OrchestratorConfig(max_per_module=1, requeue_cooldown_secs=30.0)
        scheduler = Scheduler(config, time_source=lambda: now)

        # Seed an expired entry for task '7'.
        scheduler._requeue_until['7'] = now - 5.0

        task = {
            'id': '7',
            'status': 'pending',
            'dependencies': [],
            'metadata': {},
        }
        status_map: dict[str, str] = {}

        result = scheduler._eligible_for_dispatch(task, '7', status_map)

        # Predicate should pass (cooldown has elapsed, no other gates fire).
        assert result == (True, None), (
            f'Expected (True, None) for expired cooldown; got: {result}'
        )
        # The expired entry must still be in _requeue_until — the predicate
        # must NOT have removed it.
        assert '7' in scheduler._requeue_until, (
            '_eligible_for_dispatch must not delete the expired _requeue_until entry; '
            'that is the responsibility of _gc_expired_cooldowns'
        )

    @pytest.mark.asyncio
    async def test_acquire_next_invokes_requeue_cooldown_gc(self, monkeypatch):
        """acquire_next must call _gc_expired_cooldowns once per tick, clearing
        both the dispatched task's expired entry and an orphan entry (a task id
        that doesn't appear in the current task list).

        This pins the invariant that acquire_next is the sole GC owner for
        _requeue_until — expired entries are guaranteed to be removed by the
        end of any tick that calls acquire_next.
        """
        import json as _json

        base_time = 1_000_000.0
        config = OrchestratorConfig(max_per_module=1, requeue_cooldown_secs=30.0)
        scheduler = Scheduler(config, time_source=lambda: base_time)

        pending_task = {
            'id': '99',
            'title': 'GC integration test task',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['backend']},
        }
        task_response = {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': '{"tasks": [' + _json.dumps(pending_task) + ']}',
                    }
                ]
            }
        }

        mock = AsyncMock(return_value=task_response)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        # Seed two expired entries:
        # '99'   — the task that will be dispatched; its cooldown has elapsed
        # 'ghost' — an orphan id with no corresponding task in the MCP response;
        #           exercises that GC is independent of task presence
        scheduler._requeue_until['99'] = base_time - 1.0
        scheduler._requeue_until['ghost'] = base_time - 1.0

        result = await scheduler.acquire_next()

        # (a) The task should be dispatched successfully — eligibility is
        #     preserved after the refactor because GC runs before the loops.
        assert result is not None and result.task_id == '99', (
            f'Expected TaskAssignment for task 99; got: {result}'
        )

        # (b) Both expired entries must be gone — _gc_expired_cooldowns ran
        #     and cleaned up the dispatched task's entry and the orphan.
        assert scheduler._requeue_until == {}, (
            f'Expected _requeue_until to be empty after acquire_next GC sweep; '
            f'got: {scheduler._requeue_until}'
        )


# ---------------------------------------------------------------------------
# Park-and-stop pause mechanism (task 1322)
# ---------------------------------------------------------------------------

class TestSchedulerPause:
    """Unit tests for Scheduler.pause() / resume() / is_paused / pause_reason."""

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    def test_pause_sets_is_paused_and_reason(self, scheduler: Scheduler):
        scheduler.pause('parked-threshold')
        assert scheduler.is_paused is True
        assert scheduler.pause_reason == 'parked-threshold'

    def test_resume_clears_pause(self, scheduler: Scheduler):
        scheduler.pause('parked-threshold')
        scheduler.resume()
        assert scheduler.is_paused is False
        assert scheduler.pause_reason is None

    @pytest.mark.asyncio
    async def test_acquire_next_returns_none_when_paused(self, scheduler: Scheduler):
        """acquire_next() must short-circuit to None when the scheduler is paused."""
        task = {
            'id': '1',
            'title': 'Task one',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['backend']},
        }
        scheduler.get_tasks = AsyncMock(return_value=[task])

        # Sanity: without pause the task is dispatched normally.
        result_before = await scheduler.acquire_next()
        assert result_before is not None, (
            'Expected TaskAssignment before pause; got None'
        )
        scheduler.release('1')

        # Now pause and confirm acquire_next returns None.
        scheduler.pause('test')
        result_paused = await scheduler.acquire_next()
        assert result_paused is None, (
            f'Expected None while paused; got {result_paused}'
        )

        # Resume and confirm the task is dispatchable again.
        scheduler.resume()
        result_resumed = await scheduler.acquire_next()
        assert result_resumed is not None, (
            'Expected TaskAssignment after resume; got None'
        )

    def test_get_state_snapshot_includes_pause_state(self, scheduler: Scheduler):
        """get_state_snapshot() reflects the current pause state.

        - Fresh scheduler: is_paused=False, pause_reason=None.
        - After pause('park-stop: test reason'): is_paused=True, pause_reason=='park-stop: test reason'.
        - After resume(): is_paused=False, pause_reason=None.
        """
        # Fresh scheduler — not paused.
        snap = scheduler.get_state_snapshot()
        assert snap['is_paused'] is False, (
            f"Expected is_paused=False on fresh scheduler; got {snap['is_paused']!r}"
        )
        assert snap['pause_reason'] is None, (
            f"Expected pause_reason=None on fresh scheduler; got {snap['pause_reason']!r}"
        )

        # After pause — snapshot must reflect the paused state.
        reason = 'park-stop: 3 tasks parked in 1h'
        scheduler.pause(reason)
        snap_paused = scheduler.get_state_snapshot()
        assert snap_paused['is_paused'] is True, (
            f"Expected is_paused=True after pause; got {snap_paused['is_paused']!r}"
        )
        assert snap_paused['pause_reason'] == reason, (
            f"Expected pause_reason={reason!r}; got {snap_paused['pause_reason']!r}"
        )

        # After resume — snapshot must return to unpaused state.
        scheduler.resume()
        snap_resumed = scheduler.get_state_snapshot()
        assert snap_resumed['is_paused'] is False, (
            f"Expected is_paused=False after resume; got {snap_resumed['is_paused']!r}"
        )
        assert snap_resumed['pause_reason'] is None, (
            f"Expected pause_reason=None after resume; got {snap_resumed['pause_reason']!r}"
        )


class TestSchedulerBlockedTransitionTracking:
    """Unit tests for the blocked-transition deque used by park-stop trip detection."""

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1, park_stop_parked_window_hours=1.0)
        now = 1_000_000.0
        return Scheduler(config, monotonic_clock_source=lambda: now)

    def test_record_blocked_transition_appends_timestamp(self):
        """_record_blocked_transition() adds entries matching the clock source."""
        timestamps = [1_000_000.0, 1_000_001.0, 1_000_002.0]
        idx = [0]

        def time_source() -> float:
            val = timestamps[idx[0]]
            idx[0] += 1
            return val

        config = OrchestratorConfig(max_per_module=1, park_stop_parked_window_hours=1.0)
        scheduler = Scheduler(config, monotonic_clock_source=time_source)

        # Three distinct task IDs → three entries in the deque.
        scheduler._record_blocked_transition('t1')
        scheduler._record_blocked_transition('t2')
        scheduler._record_blocked_transition('t3')

        assert len(scheduler._blocked_transitions) == 3
        # Each entry is a (task_id, timestamp) tuple.
        assert [ts for _, ts in scheduler._blocked_transitions] == timestamps

    def test_record_evicts_entries_older_than_window(self):
        """Entries older than the rolling window are evicted on each record call."""
        now = 1_000_000.0
        call_times = [now]  # subsequent records return `now`

        def time_source() -> float:
            return call_times[-1]

        config = OrchestratorConfig(max_per_module=1, park_stop_parked_window_hours=1.0)
        scheduler = Scheduler(config, monotonic_clock_source=time_source)

        # Seed three (task_id, timestamp) entries: two outside the 3600s window, one inside.
        from collections import deque
        scheduler._blocked_transitions = deque([
            ('task-a', now - 7200),   # 2h ago — expired
            ('task-b', now - 3600),   # exactly at cutoff boundary — expired (strictly older)
            ('task-c', now - 100),    # 100s ago — within window
        ])
        scheduler._blocked_task_ids_in_window = {'task-a', 'task-b', 'task-c'}

        # Record a new transition at `now`.
        scheduler._record_blocked_transition('task-d')

        timestamps_in_deque = [ts for _, ts in scheduler._blocked_transitions]
        task_ids_in_deque = [tid for tid, _ in scheduler._blocked_transitions]
        assert (now - 7200) not in timestamps_in_deque, 'Entry 2h old must be evicted'
        assert (now - 3600) not in timestamps_in_deque, 'Entry at boundary must be evicted'
        assert (now - 100) in timestamps_in_deque, 'Entry 100s old must survive'
        assert now in timestamps_in_deque, 'New entry must be present'
        assert len(scheduler._blocked_transitions) == 2, (
            f'Expected 2 entries; got {len(scheduler._blocked_transitions)}: '
            f'{list(scheduler._blocked_transitions)}'
        )
        # task-a and task-b must also be removed from the companion set.
        assert 'task-a' not in scheduler._blocked_task_ids_in_window
        assert 'task-b' not in scheduler._blocked_task_ids_in_window
        assert 'task-c' in scheduler._blocked_task_ids_in_window
        assert 'task-d' in scheduler._blocked_task_ids_in_window
        assert 'task-d' in task_ids_in_deque

    def test_same_task_id_not_double_counted(self):
        """Idempotent re-sets of the same task must count as one transition.

        This guards against recovery loops, post-restart replays, or retry
        code paths re-marking the same already-blocked task and artificially
        inflating the trip counter beyond threshold.
        """
        now = 1_000_000.0
        config = OrchestratorConfig(max_per_module=1, park_stop_parked_window_hours=1.0)
        scheduler = Scheduler(config, monotonic_clock_source=lambda: now)

        # Block the same task three times.
        scheduler._record_blocked_transition('task-1')
        scheduler._record_blocked_transition('task-1')
        scheduler._record_blocked_transition('task-1')

        # Only the first one should be counted.
        assert len(scheduler._blocked_transitions) == 1, (
            f'Same task blocked 3× must count as 1; got {len(scheduler._blocked_transitions)}'
        )
        assert scheduler._blocked_task_ids_in_window == {'task-1'}

        # A different task must still be counted separately.
        scheduler._record_blocked_transition('task-2')
        assert len(scheduler._blocked_transitions) == 2
        assert scheduler._blocked_task_ids_in_window == {'task-1', 'task-2'}


class TestSetTaskStatusBlockedRecording:
    """set_task_status('blocked') must record in the deque; other statuses must not."""

    @pytest.mark.asyncio
    async def test_set_task_status_blocked_records_transition(self, monkeypatch):
        """A successful blocked transition adds one entry to _blocked_transitions."""
        now = 1_000_000.0
        config = OrchestratorConfig(max_per_module=1)
        scheduler = Scheduler(config, monotonic_clock_source=lambda: now)

        # Return a clean success response (no rejection structure).
        mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        await scheduler.set_task_status('42', 'blocked')

        assert len(scheduler._blocked_transitions) == 1
        # Each entry is a (task_id, timestamp) tuple.
        assert list(scheduler._blocked_transitions) == [('42', now)]

    @pytest.mark.asyncio
    async def test_set_task_status_non_blocked_does_not_record(self, monkeypatch):
        """A successful non-blocked status transition must not touch _blocked_transitions."""
        config = OrchestratorConfig(max_per_module=1)
        scheduler = Scheduler(config)

        mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        await scheduler.set_task_status('42', 'pending')

        assert len(scheduler._blocked_transitions) == 0

    @pytest.mark.asyncio
    async def test_blocked_transition_not_recorded_on_rejection(self, monkeypatch):
        """A rejected transition must NOT record in _blocked_transitions.

        Uses the terminal_exit_rejected warn-and-return carve-out: when the
        target status is terminal (done/cancelled), fused-memory returns a
        terminal_exit_rejected and the scheduler logs + returns without raising.
        The 'done' target is not 'blocked', so no recording should happen either way.
        Verifies that only confirmed successful writes advance the deque.
        """
        config = OrchestratorConfig(max_per_module=1)
        scheduler = Scheduler(config)

        # Return a terminal_exit_rejected response — 'done' is in _TERMINAL_STATUSES
        # so the warn-and-return carve-out fires (logs warning, returns cleanly).
        rejection_response = {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': '{"error": "terminal_exit_rejected", "from_status": "done", "to_status": "done"}',
                    }
                ]
            }
        }
        mock = AsyncMock(return_value=rejection_response)
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        await scheduler.set_task_status('42', 'done')

        assert len(scheduler._blocked_transitions) == 0, (
            'Rejected (non-successful) transition must not be recorded in _blocked_transitions'
        )


class TestParkStopTrip:
    """Tests for the park-stop trip detection and callback invocation."""

    @pytest.mark.asyncio
    async def test_trip_fires_callback_at_threshold(self, monkeypatch):
        """Callback is invoked exactly once when threshold is reached."""
        import re

        config = OrchestratorConfig(
            max_per_module=1,
            park_stop_parked_threshold=3,
            park_stop_parked_window_hours=1.0,
        )
        scheduler = Scheduler(config)

        callback_args: list[str] = []

        async def recording_callback(reason: str) -> None:
            callback_args.append(reason)

        scheduler._on_park_stop_trip = recording_callback
        mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        await scheduler.set_task_status('1', 'blocked')
        await scheduler.set_task_status('2', 'blocked')
        await scheduler.set_task_status('3', 'blocked')
        # Yield to the event loop so the ensure_future'd callback can execute.
        await asyncio.sleep(0)

        assert len(callback_args) == 1, (
            f'Expected callback called once; got {len(callback_args)}'
        )
        # Reason must reference the trip parameters.
        reason = callback_args[0]
        assert re.search(r'3', reason), f'Reason must mention threshold count: {reason!r}'
        assert re.search(r'1\.0', reason), f'Reason must mention window hours: {reason!r}'

    @pytest.mark.asyncio
    async def test_trip_does_not_re_fire_when_paused(self, monkeypatch):
        """Once paused, additional blocked transitions must not fire the callback again."""
        config = OrchestratorConfig(
            max_per_module=1,
            park_stop_parked_threshold=3,
        )
        scheduler = Scheduler(config)

        callback_count = [0]

        async def counting_callback(reason: str) -> None:
            callback_count[0] += 1

        scheduler._on_park_stop_trip = counting_callback
        mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        # Pause manually before the third transition reaches the threshold.
        await scheduler.set_task_status('1', 'blocked')
        await scheduler.set_task_status('2', 'blocked')
        scheduler.pause('manual')  # already paused; trip check should be suppressed
        await scheduler.set_task_status('3', 'blocked')

        assert callback_count[0] == 0, (
            f'Callback must not fire while already paused; fired {callback_count[0]} time(s)'
        )

    @pytest.mark.asyncio
    async def test_trip_below_threshold_does_not_fire(self, monkeypatch):
        """Below-threshold transitions must never invoke the callback."""
        config = OrchestratorConfig(
            max_per_module=1,
            park_stop_parked_threshold=5,
        )
        scheduler = Scheduler(config)

        callback_count = [0]

        async def counting_callback(reason: str) -> None:
            callback_count[0] += 1

        scheduler._on_park_stop_trip = counting_callback
        mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        for i in range(4):
            await scheduler.set_task_status(str(i), 'blocked')

        assert callback_count[0] == 0, (
            f'Expected 0 callback invocations (only 4 of 5 threshold); got {callback_count[0]}'
        )


    @pytest.mark.asyncio
    async def test_trip_does_not_re_fire_under_concurrent_transitions(self, monkeypatch):
        """Regression test: concurrent blocked transitions must not cause duplicate callbacks.

        Without the synchronous latch fix in _maybe_fire_park_stop_trip, coroutines
        3..N all observe self._paused=False (the async callback hasn't set it yet) and
        each schedule a duplicate ensure_future — so callback_args ends up with length 4+.
        After the fix, the synchronous pause() call inside _maybe_fire_park_stop_trip
        immediately sets _paused=True, so any concurrent coroutine that has already
        reached the guard returns early.
        """
        config = OrchestratorConfig(
            max_per_module=1,
            park_stop_parked_threshold=3,
            park_stop_parked_window_hours=1.0,
        )
        scheduler = Scheduler(config)

        callback_args: list[str] = []

        async def recording_callback(reason: str) -> None:
            # Mirror harness.pause_scheduler shape: call scheduler.pause()
            # so any concurrent guard check sees _paused=True immediately.
            scheduler.pause(reason)
            callback_args.append(reason)

        scheduler._on_park_stop_trip = recording_callback
        mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        # Fire 6 concurrent blocked transitions (threshold=3, so calls 3-6 all
        # observe n >= threshold if _paused is not set synchronously first).
        await asyncio.gather(
            *[scheduler.set_task_status(str(i), 'blocked') for i in range(6)]
        )
        # Yield so any ensure_future'd callbacks have a chance to run.
        await asyncio.sleep(0)

        assert len(callback_args) == 1, (
            f'Callback must fire exactly once (synchronous latch prevents duplicates); '
            f'got {len(callback_args)}: {callback_args!r}'
        )
        assert scheduler.is_paused, 'Scheduler must be paused after trip'
        assert scheduler.pause_reason == callback_args[0], (
            'pause_reason must equal the reason passed to the callback'
        )


    @pytest.mark.asyncio
    async def test_resume_clears_blocked_transitions_deque(self, monkeypatch):
        """Regression: resume() must clear the rolling _blocked_transitions deque.

        Without this, the wall-clock 1h window can still hold ≥ threshold stale
        timestamps after a human resume, so the very next blocked transition
        (e.g. an in-flight workflow finishing shortly after resume) would
        immediately re-trip the park-stop pause and silently undo the operator
        action.  The flow: trip → resume → record one new blocked transition →
        is_paused must still be False.
        """
        config = OrchestratorConfig(
            max_per_module=1,
            park_stop_parked_threshold=3,
            park_stop_parked_window_hours=1.0,
        )
        scheduler = Scheduler(config)

        callback_count = [0]

        async def trip_callback(reason: str) -> None:
            callback_count[0] += 1
            # Mirror harness.pause_scheduler — scheduler.pause is idempotent
            # because the synchronous latch already set _paused=True.
            scheduler.pause(reason)

        scheduler._on_park_stop_trip = trip_callback
        mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        # Trip the park-stop by marking threshold tasks blocked.
        for i in range(3):
            await scheduler.set_task_status(str(i), 'blocked')
        await asyncio.sleep(0)  # let ensure_future callbacks settle
        assert scheduler.is_paused, 'Trip must have fired and paused scheduler'
        assert callback_count[0] == 1

        # Operator resumes.
        scheduler.resume()
        assert scheduler.is_paused is False
        assert scheduler.pause_reason is None

        # One new blocked transition arrives (e.g. an in-flight workflow
        # finishing shortly after resume).  With the deque cleared on resume,
        # the new count (=1) is far below threshold and must NOT re-trip.
        await scheduler.set_task_status('post-resume', 'blocked')
        await asyncio.sleep(0)
        assert scheduler.is_paused is False, (
            'Park-stop must not re-trip on a single post-resume transition '
            '(deque should have been cleared on resume)'
        )
        assert callback_count[0] == 1, (
            f'Callback must not fire a second time; fired {callback_count[0]}'
        )


class TestParkStopDisabled:
    """park_stop_enabled=False suppresses the trip but still records transitions."""

    @pytest.mark.asyncio
    async def test_park_stop_disabled_suppresses_trip(self, monkeypatch):
        """When park_stop_enabled=False the callback must NOT be invoked, but
        _blocked_transitions must still accumulate entries (so a live-enable
        picks up accurate state immediately)."""
        config = OrchestratorConfig(
            max_per_module=1,
            park_stop_enabled=False,
            park_stop_parked_threshold=3,
        )
        scheduler = Scheduler(config)

        callback_count = [0]

        async def recording_callback(reason: str) -> None:
            callback_count[0] += 1

        scheduler._on_park_stop_trip = recording_callback
        mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        await scheduler.set_task_status('1', 'blocked')
        await scheduler.set_task_status('2', 'blocked')
        await scheduler.set_task_status('3', 'blocked')

        # Transitions are recorded even when disabled.
        assert len(scheduler._blocked_transitions) == 3, (
            'Blocked transitions must be recorded even when park_stop_enabled=False'
        )
        # Trip must be suppressed.
        assert callback_count[0] == 0, (
            f'Callback must not fire when park_stop_enabled=False; fired {callback_count[0]} time(s)'
        )


class TestDepsSatisfiedTrainAware:
    """Unit tests for the intra-train merge-deferred allowance in _deps_satisfied.

    PRD § 9.3: a pending train member may dispatch when its immediate predecessor
    has status 'merge-deferred' IFF both share the same metadata.train.id.
    Non-train tasks must see no behavioural change.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    def test_intra_train_merge_deferred_dep_satisfied(
        self, scheduler: Scheduler, caplog: pytest.LogCaptureFixture
    ):
        """PRD Scenario 3: train β dispatches when train α is merge-deferred (same train_id).

        Both tasks carry metadata.train.id == 'T1'.  The dep is in status
        'merge-deferred' (non-terminal) but since it shares the train id,
        _deps_satisfied must return True and emit the intra-train log signal.
        """
        import logging

        alpha_task = {
            'id': '1',
            'status': 'merge-deferred',
            'dependencies': [],
            'metadata': {'train': {'id': 'T1', 'order': 0}},
        }
        beta_task = {
            'id': '2',
            'dependencies': [{'id': 1}],
            'metadata': {'train': {'id': 'T1', 'order': 1}},
        }
        status_map = {'1': 'merge-deferred', '2': 'pending'}
        tasks_by_id = {'1': alpha_task, '2': beta_task}

        with caplog.at_level(logging.DEBUG, logger='orchestrator.scheduler'):
            result = scheduler._deps_satisfied(beta_task, status_map, tasks_by_id)

        assert result is True, (
            'intra-train: β must be satisfied when α is merge-deferred and both share train_id T1'
        )
        assert any(
            'intra-train dep satisfied' in record.message
            and 'dep=1' in record.message
            and 'train_id=T1' in record.message
            for record in caplog.records
        ), (
            f'Expected "intra-train dep satisfied" log with dep=1 and train_id=T1. '
            f'Got: {[r.message for r in caplog.records]}'
        )

    def test_extra_train_dep_no_train_metadata_blocks(
        self, scheduler: Scheduler, caplog: pytest.LogCaptureFixture
    ):
        """PRD Scenario 4: task δ with no train metadata is blocked by merge-deferred dep.

        δ has no metadata.train; α is merge-deferred and has train_id T1.
        The allowance must not fire — δ is extra-train.
        The existing block log naming dep '1' must appear.
        """
        import logging

        alpha_task = {
            'id': '1',
            'status': 'merge-deferred',
            'dependencies': [],
            'metadata': {'train': {'id': 'T1', 'order': 0}},
        }
        delta_task = {
            'id': '5',
            'dependencies': [{'id': 1}],
            # no metadata.train
        }
        status_map = {'1': 'merge-deferred', '5': 'pending'}
        tasks_by_id = {'1': alpha_task, '5': delta_task}

        with caplog.at_level(logging.DEBUG, logger='orchestrator.scheduler'):
            result = scheduler._deps_satisfied(delta_task, status_map, tasks_by_id)

        assert result is False, (
            'extra-train: δ (no train metadata) must be blocked by merge-deferred dep'
        )
        assert any(
            '1' in record.message and 'merge-deferred' in record.message
            for record in caplog.records
        ), (
            f'Expected block log about dep 1 being merge-deferred. '
            f'Got: {[r.message for r in caplog.records]}'
        )

    def test_different_train_id_dep_blocks(self, scheduler: Scheduler):
        """γ (train_id=T2) with dep α (train_id=T1, merge-deferred) is blocked.

        Different train_ids means the predecessor is from a different train — the
        intra-train allowance must not fire.
        """
        alpha_task = {
            'id': '1',
            'status': 'merge-deferred',
            'dependencies': [],
            'metadata': {'train': {'id': 'T1', 'order': 0}},
        }
        gamma_task = {
            'id': '3',
            'dependencies': [{'id': 1}],
            'metadata': {'train': {'id': 'T2', 'order': 0}},
        }
        status_map = {'1': 'merge-deferred', '3': 'pending'}
        tasks_by_id = {'1': alpha_task, '3': gamma_task}

        result = scheduler._deps_satisfied(gamma_task, status_map, tasks_by_id)
        assert result is False, (
            'different-train: γ (T2) must be blocked when dep α (T1) is merge-deferred'
        )

    def test_non_train_regression_done_dep_satisfied(self, scheduler: Scheduler):
        """Non-train task with tasks_by_id=None: done dep → satisfied (regression guard)."""
        task = {'id': '10', 'dependencies': [{'id': 9}]}
        status_map = {'9': 'done', '10': 'pending'}

        result = scheduler._deps_satisfied(task, status_map, tasks_by_id=None)
        assert result is True, (
            'regression: plain task with done dep must remain satisfied when tasks_by_id=None'
        )

    def test_non_train_regression_merge_deferred_dep_blocks(self, scheduler: Scheduler):
        """Non-train task with tasks_by_id=None: merge-deferred dep → blocked (PRD Scenario 11).

        Pins that merge-deferred blocks when tasks_by_id is absent — byte-identical to
        today's behaviour.  tasks_by_id=None is the default when callers omit the arg.
        """
        task = {'id': '10', 'dependencies': [{'id': 9}]}
        status_map = {'9': 'merge-deferred', '10': 'pending'}

        result = scheduler._deps_satisfied(task, status_map, tasks_by_id=None)
        assert result is False, (
            'regression: plain task with merge-deferred dep must be blocked when tasks_by_id=None'
        )

    def test_tasks_by_id_none_disables_train_allowance(self, scheduler: Scheduler):
        """β has train metadata but caller omits tasks_by_id → merge-deferred dep still blocks.

        Defensive backward-compat: when the caller doesn't pass tasks_by_id,
        we cannot verify the dep's train_id so we conservatively block.
        """
        beta_task = {
            'id': '2',
            'dependencies': [{'id': 1}],
            'metadata': {'train': {'id': 'T1', 'order': 1}},
        }
        status_map = {'1': 'merge-deferred', '2': 'pending'}

        # Intentionally omit tasks_by_id (uses default None)
        result = scheduler._deps_satisfied(beta_task, status_map)
        assert result is False, (
            'defensive: β (train member) must be blocked when tasks_by_id is absent '
            '(no way to verify dep train_id)'
        )

    def test_dep_missing_from_tasks_by_id_blocks(self, scheduler: Scheduler):
        """β is a train member but dep is absent from tasks_by_id (stale snapshot) → blocks.

        Conservative behaviour: cannot verify train_id match without dep record.
        """
        beta_task = {
            'id': '2',
            'dependencies': [{'id': 1}],
            'metadata': {'train': {'id': 'T1', 'order': 1}},
        }
        status_map = {'1': 'merge-deferred', '2': 'pending'}
        tasks_by_id = {}  # dep '1' missing from snapshot

        result = scheduler._deps_satisfied(beta_task, status_map, tasks_by_id)
        assert result is False, (
            'stale-snapshot: β must be blocked when dep is absent from tasks_by_id'
        )

    def test_dep_without_train_metadata_blocks_train_member(self, scheduler: Scheduler):
        """dep in tasks_by_id but has no metadata.train → intra-train allowance must not fire.

        β is a train member (train_id='T1').  α (merge-deferred) is present in
        tasks_by_id but carries no metadata.train at all.  dep_train_id resolves
        to None (which != 'T1'), so the allowance must NOT fire and
        _deps_satisfied must return False.

        Regression guard: a mistaken default of 'satisfied when dep_train_id is
        None' would slip past the other missing-dep tests.
        """
        alpha_task = {
            'id': '1',
            'status': 'merge-deferred',
            'dependencies': [],
            'metadata': {},  # no 'train' key at all
        }
        beta_task = {
            'id': '2',
            'dependencies': [{'id': 1}],
            'metadata': {'train': {'id': 'T1', 'order': 1}},
        }
        status_map = {'1': 'merge-deferred', '2': 'pending'}
        tasks_by_id = {'1': alpha_task, '2': beta_task}

        result = scheduler._deps_satisfied(beta_task, status_map, tasks_by_id)
        assert result is False, (
            'dep without train metadata must block train member '
            '(dep_train_id=None cannot match train_id=T1)'
        )


class TestAcquireNextTrainDispatch:
    """End-to-end: acquire_next dispatches intra-train members when predecessor is merge-deferred.

    PRD § 9.3: β (pending, same train as α) must dispatch when α is merge-deferred.
    Non-train task δ must stay blocked by merge-deferred α (extra-train).
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @pytest.mark.asyncio
    async def test_acquire_next_dispatches_train_member_when_predecessor_merge_deferred(
        self, scheduler: Scheduler
    ):
        """acquire_next returns β when α (same train, merge-deferred) unblocks it.

        α is in status 'merge-deferred' (non-terminal but intra-train allowed).
        β is pending with dep α and shares train_id 'T1'.
        acquire_next must return a TaskAssignment with task_id == 'B'.
        """
        alpha = {
            'id': 'A',
            'title': 'Train alpha',
            'status': 'merge-deferred',
            'dependencies': [],
            'metadata': {'files': ['backend'], 'train': {'id': 'T1', 'order': 0}},
        }
        beta = {
            'id': 'B',
            'title': 'Train beta',
            'status': 'pending',
            'dependencies': [{'id': 'A'}],
            'metadata': {'files': ['frontend'], 'train': {'id': 'T1', 'order': 1}},
        }
        scheduler.get_tasks = AsyncMock(return_value=[alpha, beta])

        result = await scheduler.acquire_next()
        assert result is not None, (
            'Expected β to be dispatched (intra-train allowance: α is merge-deferred, same T1)'
        )
        assert result.task_id == 'B', (
            f'Expected task_id == "B", got {result.task_id!r}'
        )

    @pytest.mark.asyncio
    async def test_acquire_next_blocks_non_train_dep_on_merge_deferred_predecessor(
        self, scheduler: Scheduler
    ):
        """acquire_next returns None when δ (no train metadata) depends on merge-deferred α.

        α is merge-deferred with train_id T1.  δ has no train metadata — it is
        extra-train and must not benefit from the intra-train allowance.
        acquire_next must return None (δ is blocked).
        """
        alpha = {
            'id': 'A',
            'title': 'Train alpha',
            'status': 'merge-deferred',
            'dependencies': [],
            'metadata': {'files': ['backend'], 'train': {'id': 'T1', 'order': 0}},
        }
        delta = {
            'id': 'D',
            'title': 'Plain delta',
            'status': 'pending',
            'dependencies': [{'id': 'A'}],
            'metadata': {'files': ['frontend']},  # no train metadata
        }
        scheduler.get_tasks = AsyncMock(return_value=[alpha, delta])

        result = await scheduler.acquire_next()
        assert result is None, (
            'Expected None — δ (no train metadata) must be blocked by merge-deferred α'
        )


class TestTasksByTrain:
    """Unit tests for Scheduler.tasks_by_train(train_id) — δ₂ member-discovery helper.

    tasks_by_train does NOT exist yet → these tests are RED until step-2 adds
    the implementation.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    def _mixed_tasks(self) -> list[dict]:
        """Return a mixed task list: 3 T1 members (out of order), 1 T2, 1 non-train."""
        return [
            {
                'id': '201',
                'title': 'T1 order 2',
                'status': 'merge-deferred',
                'metadata': {'train': {'id': 'T1', 'order': 2}},
            },
            {
                'id': '199',
                'title': 'T1 order 0',
                'status': 'merge-deferred',
                'metadata': {'train': {'id': 'T1', 'order': 0}},
            },
            {
                'id': '200',
                'title': 'T1 order 1',
                'status': 'merge-deferred',
                'metadata': {'train': {'id': 'T1', 'order': 1}},
            },
            {
                'id': '300',
                'title': 'T2 member',
                'status': 'pending',
                'metadata': {'train': {'id': 'T2', 'order': 0}},
            },
            {
                'id': '400',
                'title': 'Non-train task',
                'status': 'in-progress',
                'metadata': {},
            },
        ]

    @pytest.mark.asyncio
    async def test_returns_t1_members_sorted_ascending(self, scheduler: Scheduler):
        """tasks_by_train('T1') returns exactly the 3 T1 members, sorted root→tip."""
        scheduler.get_tasks = AsyncMock(return_value=self._mixed_tasks())

        result = await scheduler.tasks_by_train('T1')

        assert len(result) == 3, f'Expected 3 T1 members, got {len(result)}'
        ids = [str(t.get('id')) for t in result]
        assert ids == ['199', '200', '201'], (
            f'Expected ascending order [199, 200, 201], got {ids}'
        )

    @pytest.mark.asyncio
    async def test_excludes_other_train_and_non_train(self, scheduler: Scheduler):
        """tasks_by_train('T1') excludes T2 members and non-train tasks."""
        scheduler.get_tasks = AsyncMock(return_value=self._mixed_tasks())

        result = await scheduler.tasks_by_train('T1')

        returned_ids = {str(t.get('id')) for t in result}
        assert '300' not in returned_ids, 'T2 member must not appear in T1 results'
        assert '400' not in returned_ids, 'Non-train task must not appear in T1 results'

    @pytest.mark.asyncio
    async def test_unknown_train_returns_empty(self, scheduler: Scheduler):
        """tasks_by_train('UNKNOWN') returns [] when no task has that train_id."""
        scheduler.get_tasks = AsyncMock(return_value=self._mixed_tasks())

        result = await scheduler.tasks_by_train('UNKNOWN')

        assert result == [], f'Expected [], got {result!r}'

    @pytest.mark.asyncio
    async def test_empty_train_id_returns_empty(self, scheduler: Scheduler):
        """tasks_by_train('') returns [] without calling get_tasks (falsy guard)."""
        scheduler.get_tasks = AsyncMock(return_value=self._mixed_tasks())

        result = await scheduler.tasks_by_train('')

        assert result == [], f'Expected [], got {result!r}'

    @pytest.mark.asyncio
    async def test_missing_order_sorts_last_no_crash(self, scheduler: Scheduler):
        """Members with missing/non-int 'order' sort last deterministically — no crash."""
        tasks = [
            {
                'id': '501',
                'title': 'T3 order 0',
                'status': 'merge-deferred',
                'metadata': {'train': {'id': 'T3', 'order': 0}},
            },
            {
                'id': '502',
                'title': 'T3 missing order',
                'status': 'merge-deferred',
                'metadata': {'train': {'id': 'T3'}},          # no 'order' key
            },
            {
                'id': '503',
                'title': 'T3 non-int order',
                'status': 'merge-deferred',
                'metadata': {'train': {'id': 'T3', 'order': 'X'}},  # non-int
            },
        ]
        scheduler.get_tasks = AsyncMock(return_value=tasks)

        result = await scheduler.tasks_by_train('T3')

        assert len(result) == 3, f'Expected 3 members, got {len(result)}'
        # The first member must have id=='501' (order 0 — the only valid int)
        assert str(result[0].get('id')) == '501', (
            f'Expected 501 first (order=0), got {result[0].get("id")!r}'
        )

    @pytest.mark.asyncio
    async def test_tasks_by_train_fetches_active_only(self, scheduler: Scheduler):
        """tasks_by_train must pass statuses=ACTIVE_TASK_STATUSES to get_tasks (γ3b).

        Fails until step-2 patches the call — today tasks_by_train calls
        self.get_tasks() with no statuses kwarg.
        """
        scheduler.get_tasks = AsyncMock(return_value=self._mixed_tasks())

        await scheduler.tasks_by_train('T1')

        scheduler.get_tasks.assert_awaited_once()
        call_kwargs = scheduler.get_tasks.call_args.kwargs
        assert 'statuses' in call_kwargs, (
            f'tasks_by_train must call get_tasks with statuses=ACTIVE_TASK_STATUSES, '
            f'but call_args.kwargs was: {call_kwargs}'
        )
        assert set(call_kwargs['statuses']) == ACTIVE_TASK_STATUSES, (
            f'Expected statuses {ACTIVE_TASK_STATUSES}, got {set(call_kwargs["statuses"])}'
        )

    @pytest.mark.asyncio
    async def test_tasks_by_train_excludes_terminal_members_via_server_filter(
        self, scheduler: Scheduler
    ):
        """Active-only server filter excludes done/cancelled T1 members from results.

        Pins the behavioural consequence of the active-only fetch (not just the
        kwarg shape): when get_tasks respects statuses=ACTIVE_TASK_STATUSES, a
        cancelled train member is absent from the returned list.

        Uses a side_effect mock that honours the statuses kwarg — simulating the
        server-side SQL filter — so the exclusion of terminal members is actually
        exercised rather than assumed.

        Regression target: a train member cancelled mid-flight must NOT appear in
        tasks_by_train() results.  If it did, the all-merge-deferred guard in
        _maybe_enqueue_group_merge (workflow.py:770) would stall waiting for it.
        """
        active_members = [
            {
                'id': '199',
                'title': 'T1 order 0 (active)',
                'status': 'merge-deferred',
                'metadata': {'train': {'id': 'T1', 'order': 0}},
            },
            {
                'id': '200',
                'title': 'T1 order 1 (active)',
                'status': 'merge-deferred',
                'metadata': {'train': {'id': 'T1', 'order': 1}},
            },
        ]
        cancelled_member = {
            'id': '998',
            'title': 'T1 cancelled member',
            'status': 'cancelled',
            'metadata': {'train': {'id': 'T1', 'order': 2}},
        }
        all_tasks = active_members + [cancelled_member]

        async def _get_tasks_with_filter(*, statuses=None, **_kw):
            """Simulate server-side statuses filter."""
            if statuses is not None:
                return [t for t in all_tasks if t['status'] in statuses]
            return all_tasks

        scheduler.get_tasks = AsyncMock(side_effect=_get_tasks_with_filter)

        result = await scheduler.tasks_by_train('T1')

        returned_ids = {str(t['id']) for t in result}
        assert '998' not in returned_ids, (
            'Cancelled T1 member must be excluded by active-only server-side filter'
        )
        assert returned_ids == {'199', '200'}, (
            f'Expected only the two active members {{199, 200}}, got {returned_ids}'
        )


# ---------------------------------------------------------------------------
# TestGetExternalStatuses (task 1580 — step-1 RED / step-2 GREEN)
# ---------------------------------------------------------------------------

class TestGetExternalStatuses:
    """``Scheduler.get_external_statuses`` returns a ``(statuses, error)`` tuple.

    Mirrors the TestGetStatuses shape but for the cross-project resolver:
    - exactly ONE dispatch_tool('get_external_statuses', {'deps': [...]}) call
    - no project_root argument (the tool is cross-project by design)
    - returns ({}, exc) on exception rather than propagating
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @pytest.mark.asyncio
    async def test_get_external_statuses_returns_parsed_mapping(
        self, scheduler: Scheduler, monkeypatch
    ):
        """get_external_statuses parses the MCP response and returns (dict, None)."""
        import json

        # Flat shape: producer returns a bare {dep: status} dict, no 'statuses' wrapper
        # (mirrors fused-memory tools.py get_external_statuses: `return result`).
        response = {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': json.dumps(
                            {
                                'dark_factory:5': 'done',
                                'other_proj:99': 'pending',
                            }
                        ),
                    }
                ]
            }
        }
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=response),
        )
        result, err = await scheduler.get_external_statuses(
            ['dark_factory:5', 'other_proj:99']
        )
        assert err is None
        assert result == {'dark_factory:5': 'done', 'other_proj:99': 'pending'}

    @pytest.mark.asyncio
    async def test_get_external_statuses_passes_deps_argument_no_project_root(
        self, scheduler: Scheduler, monkeypatch
    ):
        """Exactly ONE dispatch_tool call with {'deps': [...]} — no project_root."""
        import json

        # Flat shape (no 'statuses' wrapper — mirrors producer contract).
        mcp_mock = AsyncMock(
            return_value={
                'result': {
                    'content': [
                        {
                            'type': 'text',
                            'text': json.dumps({'dark_factory:5': 'done'}),
                        }
                    ]
                }
            }
        )
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mcp_mock)

        _result, _err = await scheduler.get_external_statuses(['dark_factory:5'])

        mcp_mock.assert_called_once()
        # mcp_call signature: (url, 'tools/call', {'name': ..., 'arguments': ...})
        call_payload = mcp_mock.call_args[0][2]  # third positional arg
        assert call_payload.get('name') == 'get_external_statuses', (
            f"Expected name='get_external_statuses'; got {call_payload.get('name')!r}"
        )
        arguments = call_payload['arguments']
        assert arguments.get('deps') == ['dark_factory:5'], (
            f"Expected deps=['dark_factory:5']; got {arguments!r}"
        )
        assert 'project_root' not in arguments, (
            f'project_root must not be present in arguments; got {arguments!r}'
        )

    @pytest.mark.asyncio
    async def test_get_external_statuses_exception_returns_empty_dict(
        self, scheduler: Scheduler, monkeypatch
    ):
        """OSError from mcp_call returns ({}, OSError) tuple."""
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(side_effect=OSError(2, 'Connection refused')),
        )
        result, err = await scheduler.get_external_statuses(['dark_factory:5'])
        assert result == {}
        assert isinstance(err, OSError)
        assert err.errno == 2

    @pytest.mark.asyncio
    async def test_get_external_statuses_exception_then_success_no_state_leak(
        self, scheduler: Scheduler, monkeypatch
    ):
        """Failing call returns ({}, exc); subsequent success returns (dict, None).

        Error state lives on the stack — no cross-call leakage via shared attribute.
        """
        import json

        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(side_effect=OSError(2, 'Connection refused')),
        )
        result_fail, err_fail = await scheduler.get_external_statuses(['dark_factory:5'])
        assert result_fail == {}
        assert isinstance(err_fail, OSError)

        # Flat shape (no 'statuses' wrapper — mirrors producer contract).
        success_response = {
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': json.dumps({'dark_factory:5': 'done'}),
                    }
                ]
            }
        }
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=success_response),
        )
        result_ok, err_ok = await scheduler.get_external_statuses(['dark_factory:5'])
        assert result_ok == {'dark_factory:5': 'done'}
        assert err_ok is None


# ---------------------------------------------------------------------------
# TestGetExternalStatusesFailsLoud (task 1799 — step-1 RED / step-2 GREEN)
# ---------------------------------------------------------------------------

class TestGetExternalStatusesFailsLoud:
    """``Scheduler.get_external_statuses`` must fail LOUD on non-dict/unparseable results.

    Today the non-dict branch falls through to ``return {}, None`` (err is None),
    silently stranding tasks.  After the fix:
    - Non-dict / missing-'statuses'-key → ``({}, ExternalResolverError(...))``
    - A WARNING is logged naming the failure.
    - The existing exception-raised path (``({}, exception)``) is unchanged.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @staticmethod
    def _envelope(payload: dict) -> dict:
        """Return a JSON-RPC envelope with a single text block."""
        import json as _json
        return {
            'result': {
                'content': [{'type': 'text', 'text': _json.dumps(payload)}]
            }
        }

    @pytest.mark.asyncio
    async def test_non_dict_statuses_returns_error(
        self, scheduler: Scheduler, monkeypatch, caplog
    ):
        """Non-dict inner (list) → ({}, ExternalResolverError) + WARNING from shared.mcp_envelope.

        A non-dict inner (JSON list) trips INNER_NOT_DICT in parse_tool_result, which
        emits a WARNING from shared.mcp_envelope and returns ({}, ExternalResolverError).
        Under the key=None consumer a dict-but-missing-key payload would yield resolver-
        degraded via the missing-dep guard instead; use a genuine non-dict inner here.
        """
        import json as _json
        import logging

        import orchestrator.scheduler as _sched_module

        # Non-dict inner: JSON list → INNER_NOT_DICT → ({}, ExternalResolverError) + WARNING.
        response = {
            'result': {'content': [{'type': 'text', 'text': _json.dumps(['not', 'a', 'dict'])}]}
        }
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=response),
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduler'):
            statuses, err = await scheduler.get_external_statuses(['upstream_proj:1'])

        assert statuses == {}, f'Expected empty dict; got {statuses!r}'
        assert err is not None, (
            'Expected ExternalResolverError in error slot; got None '
            '(non-dict branch fell through to return {}, None)'
        )
        assert isinstance(err, _sched_module.ExternalResolverError), (
            f'Expected ExternalResolverError; got {type(err).__name__}'
        )
        assert any(
            r.levelno >= logging.WARNING for r in caplog.records
        ), f'Expected a WARNING log; got records={caplog.records!r}'

    @pytest.mark.asyncio
    async def test_non_dict_result_no_state_leak(
        self, scheduler: Scheduler, monkeypatch
    ):
        """Non-dict error leaves no persistent state; next call still works correctly."""
        import json as _json

        import orchestrator.scheduler as _sched_module

        # First call: non-dict inner (list) → INNER_NOT_DICT → ExternalResolverError.
        response_bad = {
            'result': {'content': [{'type': 'text', 'text': _json.dumps(['not', 'a', 'dict'])}]}
        }
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=response_bad),
        )
        _stat_bad, err_bad = await scheduler.get_external_statuses(['upstream_proj:1'])
        assert err_bad is not None
        assert isinstance(err_bad, _sched_module.ExternalResolverError)

        # Second call: correct flat-shape response (no 'statuses' wrapper).
        response_ok = self._envelope({'upstream_proj:1': 'done'})
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=response_ok),
        )
        stat_ok, err_ok = await scheduler.get_external_statuses(['upstream_proj:1'])
        assert err_ok is None
        assert stat_ok == {'upstream_proj:1': 'done'}


# ---------------------------------------------------------------------------
# TestGetExternalStatusesFoldRegression (task 1807 — step-5 RED / step-6 GREEN)
# ---------------------------------------------------------------------------

class TestGetExternalStatusesFoldRegression:
    """Fold-regression: get_external_statuses parse must route through shared.mcp_envelope.

    After folding get_external_statuses' parse onto ``parse_tool_result``, a
    non-dict response must:
    - Still return ``({}, ExternalResolverError)`` (existing contract preserved).
    - Emit a WARNING from the ``shared.mcp_envelope`` logger (proving the
      primitive is in the parse path).

    Fails today: the non-dict path warns only from ``orchestrator.scheduler``
    (the hand-rolled ``_parse_tool_text_result`` fallback), so no
    ``shared.mcp_envelope`` record exists.

    Existing ``TestGetExternalStatusesFailsLoud`` / ``TestGetExternalStatusesPartialResult``
    must remain green throughout (they check ``any(level>=WARNING)`` without
    filtering by logger name, so the primitive's WARNING keeps them green after
    the fold drops the redundant scheduler-level WARNING).
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @staticmethod
    def _envelope(payload: dict) -> dict:
        import json as _json
        return {
            'result': {
                'content': [{'type': 'text', 'text': _json.dumps(payload)}]
            }
        }

    @pytest.mark.asyncio
    async def test_non_dict_emits_warning_from_shared_mcp_envelope(
        self, scheduler: Scheduler, monkeypatch, caplog
    ):
        """Non-dict inner returns ({}, ExternalResolverError) + WARNING from shared.mcp_envelope.

        A JSON list inner triggers INNER_NOT_DICT in parse_tool_result, which is the
        shared.mcp_envelope primitive — proven by asserting the WARNING logger name.
        Under key=None a dict-but-missing-key payload would be accepted by the primitive
        (missing-dep guard warns from orchestrator.scheduler, not shared.mcp_envelope).
        Using a genuine non-dict inner keeps the shared.mcp_envelope assertion valid.
        """
        import json as _json
        import logging

        import orchestrator.scheduler as _sched_module

        # Non-dict inner (JSON list) → INNER_NOT_DICT in shared.mcp_envelope.
        response = {
            'result': {'content': [{'type': 'text', 'text': _json.dumps(['not', 'a', 'dict'])}]}
        }
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=response),
        )

        with caplog.at_level(logging.WARNING):
            statuses, err = await scheduler.get_external_statuses(['upstream_proj:1'])

        # Existing contract preserved: ({}, ExternalResolverError).
        assert statuses == {}
        assert err is not None
        assert isinstance(err, _sched_module.ExternalResolverError), (
            f'Expected ExternalResolverError; got {type(err).__name__}'
        )

        # NEW: the WARNING must come from shared.mcp_envelope (fold regression).
        shared_env_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and r.name == 'shared.mcp_envelope'
        ]
        assert shared_env_warnings, (
            f'Expected a WARNING from shared.mcp_envelope logger; '
            f'got records={[(r.name, r.getMessage()) for r in caplog.records]!r}'
        )


# ---------------------------------------------------------------------------
# TestGetExternalStatusesPartialResult (task 1799 — step-3 RED / step-4 GREEN)
# ---------------------------------------------------------------------------

class TestGetExternalStatusesPartialResult:
    """Partial-result guard: missing dep keys in response → resolver-degraded error.

    (a) Response dict present but missing one or more requested dep keys →
        ``(partial_statuses, ExternalResolverError)`` — partial dict PRESERVED
        (not discarded), error slot set, WARNING logged.
    (b) Negative / no-false-positive: all requested keys present (including
        sentinel values like 'unknown_task') → ``(statuses, None)`` — sentinels
        are valid values, not missing keys.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @staticmethod
    def _envelope(payload: dict) -> dict:
        import json as _json
        return {
            'result': {
                'content': [{'type': 'text', 'text': _json.dumps(payload)}]
            }
        }

    @pytest.mark.asyncio
    async def test_partial_result_returns_partial_dict_plus_error(
        self, scheduler: Scheduler, monkeypatch, caplog
    ):
        """Response missing 'upstream_proj:2' → (partial, ExternalResolverError) + WARNING."""
        import logging

        import orchestrator.scheduler as _sched_module

        # Request 2 deps; response only has 1 (flat shape — no 'statuses' wrapper).
        response = self._envelope({'upstream_proj:1': 'done'})
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=response),
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduler'):
            statuses, err = await scheduler.get_external_statuses(
                ['upstream_proj:1', 'upstream_proj:2']
            )

        # Partial dict preserved (not discarded to {}).
        assert statuses == {'upstream_proj:1': 'done'}, (
            f'Expected partial dict to be preserved; got {statuses!r}'
        )
        assert err is not None, (
            'Expected ExternalResolverError for partial result; got None '
            '(partial-result guard not yet implemented)'
        )
        assert isinstance(err, _sched_module.ExternalResolverError), (
            f'Expected ExternalResolverError; got {type(err).__name__}'
        )
        assert any(r.levelno >= logging.WARNING for r in caplog.records), (
            f'Expected a WARNING log for partial result; got {caplog.records!r}'
        )

    @pytest.mark.asyncio
    async def test_all_keys_present_with_sentinel_values_is_success(
        self, scheduler: Scheduler, monkeypatch
    ):
        """All requested keys present (including sentinels) → (statuses, None).

        Sentinels ('unknown_task', 'unknown_project', 'malformed') are PRESENT
        values — only MISSING keys trigger the partial-result guard.
        """
        # Both requested dep keys present (flat shape — no 'statuses' wrapper).
        # One is a real status, one is a sentinel — sentinels are valid values, not missing keys.
        response = self._envelope({
            'upstream_proj:1': 'done',
            'upstream_proj:2': 'unknown_task',  # sentinel, but key IS present
        })
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=response),
        )

        statuses, err = await scheduler.get_external_statuses(
            ['upstream_proj:1', 'upstream_proj:2']
        )

        assert err is None, (
            f'Sentinel value should NOT trigger partial-result guard; got err={err!r}'
        )
        assert statuses == {
            'upstream_proj:1': 'done',
            'upstream_proj:2': 'unknown_task',
        }

    @pytest.mark.asyncio
    async def test_empty_deps_no_false_positive(
        self, scheduler: Scheduler, monkeypatch
    ):
        """Empty deps list → no 'missing' keys → (empty_dict, None) success."""
        # Flat shape: producer returns a bare {} for zero deps (no 'statuses' wrapper).
        response = self._envelope({})
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=response),
        )

        statuses, err = await scheduler.get_external_statuses([])

        assert err is None, (
            f'Empty deps should not trigger partial-result guard; got err={err!r}'
        )
        assert statuses == {}


# ---------------------------------------------------------------------------
# TestExternalDepFlatShapeSeam (task 1854 — step-3 RED / step-4 GREEN)
# ---------------------------------------------------------------------------

class TestExternalDepFlatShapeSeam:
    """Two-way seam test: producer's FLAT {dep:status} shape through the consumer gate.

    Pins the producer<->consumer contract that prior tasks lacked: both sides
    shared the wrong {'statuses':{...}} assumption, so suites stayed green while
    production fail-safe-waited forever.

    The producer (fused-memory tools.py get_external_statuses) returns a BARE
    flat {dep: status} dict — `return result` in fused-memory tools.py get_external_statuses, NO 'statuses' wrapper.
    Driving that REAL shape through scheduler.get_external_statuses AND into
    _deps_satisfied asserts the full dispatch gate (user-observable signal).

    RED today: flat {dep:'done'} -> current consumer keys 'statuses' -> KEY_ABSENT
    -> ExternalResolverError -> external_resolver_failed -> not satisfied.
    GREEN after step-4 consumer flip (key=None whole-inner-dict mode).
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @staticmethod
    def _flat_envelope(statuses: dict) -> dict:
        """Wrap *statuses* in a minimal MCP envelope with NO 'statuses' key.

        Mirrors the producer's canonical flat shape:
            fused-memory/src/fused_memory/server/tools.py `return result`
        The dep strings are the top-level keys; no wrapper dict.
        """
        import json as _json
        return {
            'result': {
                'content': [{'type': 'text', 'text': _json.dumps(statuses)}]
            }
        }

    @pytest.mark.asyncio
    async def test_flat_done_dep_satisfies_gate(
        self, scheduler: Scheduler, monkeypatch
    ):
        """(Positive seam) Flat {dep:'done'} → get_external_statuses returns (dict, None)
        and _deps_satisfied returns True for a task with that external dep.
        """
        dep = 'dark_factory:1846'
        # Producer's REAL flat shape — no 'statuses' wrapper.
        response = self._flat_envelope({dep: 'done'})
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=response),
        )

        statuses, err = await scheduler.get_external_statuses([dep])

        assert err is None, (
            f'Expected err=None for flat {dep!r}=done; got {err!r} — '
            'consumer may still be keying on statuses instead of using key=None'
        )
        assert statuses == {dep: 'done'}, f'Expected flat dict; got {statuses!r}'

        # Drive the gate decision: a task with this external dep should be dispatched.
        task = {
            'id': 'X',
            'dependencies': [],
            'metadata': {'external_deps': [dep]},
        }
        satisfied = scheduler._deps_satisfied(
            task, {},
            external_status_cache=statuses,
            external_resolver_failed=(err is not None),
        )
        assert satisfied is True, (
            f'Expected _deps_satisfied=True for done external dep; got {satisfied!r}'
        )

    @pytest.mark.asyncio
    async def test_flat_pending_dep_does_not_satisfy_gate(
        self, scheduler: Scheduler, monkeypatch
    ):
        """(Negative seam) Flat {dep:'pending'} → err is None, statuses correct,
        but _deps_satisfied returns False (non-done status is not satisfied).
        """
        dep = 'dark_factory:1846'
        response = self._flat_envelope({dep: 'pending'})
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value=response),
        )

        statuses, err = await scheduler.get_external_statuses([dep])

        # The PARSE itself must succeed (err is None) — only the gate status matters.
        assert err is None, (
            f'Expected err=None for flat {dep!r}=pending; got {err!r}'
        )
        assert statuses == {dep: 'pending'}, f'Expected flat dict; got {statuses!r}'

        task = {
            'id': 'Y',
            'dependencies': [],
            'metadata': {'external_deps': [dep]},
        }
        satisfied = scheduler._deps_satisfied(
            task, {},
            external_status_cache=statuses,
            external_resolver_failed=(err is not None),
        )
        assert satisfied is False, (
            f'Expected _deps_satisfied=False for pending external dep; got {satisfied!r}'
        )


# ---------------------------------------------------------------------------
# TestDepsSatisfiedExternalGate (task 1580 — step-3 RED / step-4 GREEN)
# ---------------------------------------------------------------------------

class TestDepsSatisfiedExternalGate:
    """Unit tests for the external-dep boolean gate added to _deps_satisfied.

    The gate is PURE (no side effects, no escalation calls).  It is opt-in:
    passing external_status_cache=None / external_resolver_failed=False (the
    defaults) reproduces byte-identical legacy behaviour — existing tests in
    TestDepsSatisfied remain valid without modification.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    def _task_with_external_dep(self, dep: str = 'dark_factory:5') -> dict:
        """Build a minimal pending task with one external dep."""
        return {
            'id': '10',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'external_deps': [dep]},
        }

    # --- done → satisfied ---------------------------------------------------

    def test_external_dep_done_satisfied(self, scheduler: Scheduler):
        """External dep with status 'done' → _deps_satisfied returns True."""
        task = self._task_with_external_dep()
        status_map: dict[str, str] = {}
        cache = {'dark_factory:5': 'done'}
        assert scheduler._deps_satisfied(task, status_map, external_status_cache=cache) is True

    # --- live / non-done statuses → not satisfied ---------------------------

    @pytest.mark.parametrize('status', ['pending', 'in-progress', 'blocked'])
    def test_external_dep_live_status_not_satisfied(
        self, scheduler: Scheduler, status: str
    ):
        """External dep with a live non-done status → _deps_satisfied returns False."""
        task = self._task_with_external_dep()
        cache = {'dark_factory:5': status}
        assert (
            scheduler._deps_satisfied(task, {}, external_status_cache=cache) is False
        )

    # --- cancelled → not satisfied (strict, PRD decision 1) ----------------

    def test_external_dep_cancelled_not_satisfied(self, scheduler: Scheduler):
        """External dep with status 'cancelled' → False (strict, PRD decision 1)."""
        task = self._task_with_external_dep()
        cache = {'dark_factory:5': 'cancelled'}
        assert (
            scheduler._deps_satisfied(task, {}, external_status_cache=cache) is False
        )

    # --- sentinel statuses → not satisfied ----------------------------------

    @pytest.mark.parametrize(
        'sentinel', ['unknown_project', 'unknown_task', 'malformed']
    )
    def test_external_dep_sentinel_not_satisfied(
        self, scheduler: Scheduler, sentinel: str
    ):
        """External dep resolving to a sentinel → _deps_satisfied returns False."""
        task = self._task_with_external_dep()
        cache = {'dark_factory:5': sentinel}
        assert (
            scheduler._deps_satisfied(task, {}, external_status_cache=cache) is False
        )

    # --- dep missing from cache → not satisfied -----------------------------

    def test_external_dep_missing_from_cache_not_satisfied(
        self, scheduler: Scheduler
    ):
        """A dep absent from the cache → _deps_satisfied returns False (conservative)."""
        task = self._task_with_external_dep()
        # Cache exists but does not include 'dark_factory:5'
        cache: dict[str, str] = {}
        assert (
            scheduler._deps_satisfied(task, {}, external_status_cache=cache) is False
        )

    # --- resolver_failed=True → not satisfied regardless of cache -----------

    def test_external_resolver_failed_not_satisfied(self, scheduler: Scheduler):
        """external_resolver_failed=True → False regardless of cache contents."""
        task = self._task_with_external_dep()
        # Cache says done — but resolver failed so we must not satisfy
        cache = {'dark_factory:5': 'done'}
        assert (
            scheduler._deps_satisfied(
                task, {}, external_status_cache=cache, external_resolver_failed=True
            )
            is False
        )

    def test_external_resolver_failed_no_cache_not_satisfied(
        self, scheduler: Scheduler
    ):
        """external_resolver_failed=True with empty cache → False."""
        task = self._task_with_external_dep()
        assert (
            scheduler._deps_satisfied(
                task, {}, external_status_cache={}, external_resolver_failed=True
            )
            is False
        )

    # --- backward compatibility: no external_deps → legacy behaviour --------

    def test_no_external_deps_legacy_local_dep_done(self, scheduler: Scheduler):
        """Task with no external_deps and legacy params → legacy semantics unchanged."""
        task = {
            'id': '2',
            'dependencies': [{'id': 1}],
            'metadata': {},
        }
        status_map = {'1': 'done'}
        # Omitting external_status_cache / external_resolver_failed — defaults
        assert scheduler._deps_satisfied(task, status_map) is True

    def test_no_external_deps_legacy_local_dep_pending(self, scheduler: Scheduler):
        """Task with no external_deps: local dep pending → still False (legacy)."""
        task = {
            'id': '2',
            'dependencies': [{'id': 1}],
            'metadata': {},
        }
        status_map = {'1': 'pending'}
        assert scheduler._deps_satisfied(task, status_map) is False

    # --- mixed: local-done + external-pending → not satisfied ---------------

    def test_local_done_external_pending_not_satisfied(self, scheduler: Scheduler):
        """Local deps done but external dep 'pending' → overall not satisfied."""
        task = {
            'id': '10',
            'dependencies': [{'id': '9'}],
            'metadata': {'external_deps': ['dark_factory:5']},
        }
        status_map = {'9': 'done'}
        cache = {'dark_factory:5': 'pending'}
        assert (
            scheduler._deps_satisfied(task, status_map, external_status_cache=cache)
            is False
        )

    def test_external_done_local_pending_not_satisfied(self, scheduler: Scheduler):
        """External dep done but local dep pending → overall not satisfied."""
        task = {
            'id': '10',
            'dependencies': [{'id': '9'}],
            'metadata': {'external_deps': ['dark_factory:5']},
        }
        status_map = {'9': 'pending'}
        cache = {'dark_factory:5': 'done'}
        assert (
            scheduler._deps_satisfied(task, status_map, external_status_cache=cache)
            is False
        )


# ---------------------------------------------------------------------------
# TestApplyExternalDepPolicyCancelled (task 1580 — step-5 RED / step-6 GREEN)
# ---------------------------------------------------------------------------

class TestApplyExternalDepPolicyCancelled:
    """_apply_external_dep_policy must fire _on_external_dep_block for cancelled deps.

    PRD invariant 2 / design decision 1: cancelled is STRICT — immediate
    escalation, no grace period, no counter increment.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @pytest.mark.asyncio
    async def test_cancelled_dep_fires_callback_once(self, scheduler: Scheduler):
        """_on_external_dep_block called exactly once when dep is 'cancelled'."""
        callback = AsyncMock()
        scheduler._on_external_dep_block = callback

        task = {
            'id': '10',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'external_deps': ['dark_factory:5']},
        }
        external_cache = {'dark_factory:5': 'cancelled'}

        await scheduler._apply_external_dep_policy(
            [task], external_cache, external_err=None
        )

        callback.assert_called_once()
        call_kwargs = callback.call_args
        # First positional arg or 'task_id' kwarg must be '10'
        args = call_kwargs.args if call_kwargs.args else ()
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        task_id_arg = args[0] if args else kwargs.get('task_id')
        assert str(task_id_arg) == '10', (
            f'Expected task_id="10"; got {task_id_arg!r}'
        )
        # Summary must carry EXTERNAL_DEP_CANCELLED prefix
        summary_arg = kwargs.get('summary', '') or (args[1] if len(args) > 1 else '')
        assert 'EXTERNAL_DEP_CANCELLED' in str(summary_arg), (
            f'Expected EXTERNAL_DEP_CANCELLED in summary; got {summary_arg!r}'
        )

    @pytest.mark.asyncio
    async def test_cancelled_dep_does_not_touch_unresolved_counter(
        self, scheduler: Scheduler
    ):
        """Cancelled dep must NOT increment _external_unresolved_counts."""
        scheduler._on_external_dep_block = AsyncMock()

        task = {
            'id': '10',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'external_deps': ['dark_factory:5']},
        }
        external_cache = {'dark_factory:5': 'cancelled'}

        await scheduler._apply_external_dep_policy(
            [task], external_cache, external_err=None
        )

        assert scheduler._external_unresolved_counts == {}, (
            f'Unresolved counter must remain empty for cancelled; '
            f'got {scheduler._external_unresolved_counts!r}'
        )

    @pytest.mark.asyncio
    async def test_done_dep_no_callback(self, scheduler: Scheduler):
        """Done external dep must NOT invoke callback (satisfied — no action)."""
        callback = AsyncMock()
        scheduler._on_external_dep_block = callback

        task = {
            'id': '10',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'external_deps': ['dark_factory:5']},
        }
        external_cache = {'dark_factory:5': 'done'}

        await scheduler._apply_external_dep_policy(
            [task], external_cache, external_err=None
        )

        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_live_dep_no_callback(self, scheduler: Scheduler):
        """A live (pending/in-progress) dep must wait silently — no callback."""
        callback = AsyncMock()
        scheduler._on_external_dep_block = callback

        task = {
            'id': '10',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'external_deps': ['dark_factory:5']},
        }
        external_cache = {'dark_factory:5': 'in-progress'}

        await scheduler._apply_external_dep_policy(
            [task], external_cache, external_err=None
        )

        callback.assert_not_called()


# ---------------------------------------------------------------------------
# TestApplyExternalDepPolicyUnresolved (task 1580 — step-7 RED / step-8 GREEN)
# ---------------------------------------------------------------------------

class TestApplyExternalDepPolicyUnresolved:
    """Grace-then-escalate counter for unknown/malformed sentinel statuses.

    Uses max_external_dep_unresolved_cycles=2 so tests don't need 3 real ticks.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(
            max_per_module=1,
            max_external_dep_unresolved_cycles=2,
        )
        return Scheduler(config)

    def _pending_task(self, dep: str = 'dark_factory:5') -> dict:
        return {
            'id': '10',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'external_deps': [dep]},
        }

    @pytest.mark.parametrize(
        'sentinel', ['unknown_task', 'unknown_project', 'malformed']
    )
    @pytest.mark.asyncio
    async def test_sentinel_below_threshold_no_callback(
        self, scheduler: Scheduler, sentinel: str
    ):
        """First tick with a sentinel: counter increments, callback NOT called."""
        callback = AsyncMock()
        scheduler._on_external_dep_block = callback
        task = self._pending_task()

        await scheduler._apply_external_dep_policy(
            [task], {'dark_factory:5': sentinel}, external_err=None
        )

        callback.assert_not_called()
        assert scheduler._external_unresolved_counts.get(('10', 'dark_factory:5'), 0) == 1

    @pytest.mark.asyncio
    async def test_sentinel_at_threshold_fires_callback(self, scheduler: Scheduler):
        """At threshold (2 ticks of unknown_task), callback IS called once."""
        callback = AsyncMock()
        scheduler._on_external_dep_block = callback
        task = self._pending_task()
        cache = {'dark_factory:5': 'unknown_task'}

        # Tick 1 — below threshold
        await scheduler._apply_external_dep_policy([task], cache, external_err=None)
        callback.assert_not_called()

        # Tick 2 — reaches threshold
        await scheduler._apply_external_dep_policy([task], cache, external_err=None)
        callback.assert_called_once()

        # Summary must carry EXTERNAL_DEP_UNRESOLVED prefix
        kwargs = callback.call_args.kwargs
        assert 'EXTERNAL_DEP_UNRESOLVED' in kwargs.get('summary', ''), (
            f'Expected EXTERNAL_DEP_UNRESOLVED in summary; got {kwargs!r}'
        )

    @pytest.mark.asyncio
    async def test_counter_resets_when_dep_resolves_to_real_status(
        self, scheduler: Scheduler
    ):
        """When dep later resolves to a non-sentinel, counter is reset to 0."""
        callback = AsyncMock()
        scheduler._on_external_dep_block = callback
        task = self._pending_task()

        # Tick 1 — sentinel; counter goes to 1
        await scheduler._apply_external_dep_policy(
            [task], {'dark_factory:5': 'unknown_task'}, external_err=None
        )
        assert scheduler._external_unresolved_counts.get(('10', 'dark_factory:5')) == 1

        # Tick 2 — dep resolves to 'pending' (real live status)
        await scheduler._apply_external_dep_policy(
            [task], {'dark_factory:5': 'pending'}, external_err=None
        )
        # Counter must be gone (reset)
        assert ('10', 'dark_factory:5') not in scheduler._external_unresolved_counts, (
            f'Counter should be reset; got '
            f'{scheduler._external_unresolved_counts!r}'
        )
        # No callback (pending → wait silently)
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_counter_resets_when_dep_reaches_done(
        self, scheduler: Scheduler
    ):
        """When dep resolves to 'done', counter is reset (not just ignored)."""
        callback = AsyncMock()
        scheduler._on_external_dep_block = callback
        task = self._pending_task()

        # Tick 1 — sentinel
        await scheduler._apply_external_dep_policy(
            [task], {'dark_factory:5': 'unknown_task'}, external_err=None
        )
        assert scheduler._external_unresolved_counts.get(('10', 'dark_factory:5')) == 1

        # Tick 2 — dep is done
        await scheduler._apply_external_dep_policy(
            [task], {'dark_factory:5': 'done'}, external_err=None
        )
        assert ('10', 'dark_factory:5') not in scheduler._external_unresolved_counts
        callback.assert_not_called()


# ---------------------------------------------------------------------------
# TestApplyExternalDepPolicyTransientErr (task 1580 — step-9 RED / step-10 GREEN)
# ---------------------------------------------------------------------------

class TestApplyExternalDepPolicyTransientErr:
    """Invariant 6: transient resolver error → silent wait, no counter, no escalation.

    When external_err is not None (the MCP resolver raised), _apply_external_dep_policy
    must be a no-op: no counter increment, no callback invocation.  The boolean gate
    (_deps_satisfied with external_resolver_failed=True) must also block dispatch.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        return Scheduler(OrchestratorConfig(max_per_module=1))

    def _pending_task(self) -> dict:
        return {
            'id': '20',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'external_deps': ['dark_factory:99']},
        }

    @pytest.mark.asyncio
    async def test_transient_err_no_callback(self, scheduler: Scheduler):
        """external_err set → _on_external_dep_block never called."""
        callback = AsyncMock()
        scheduler._on_external_dep_block = callback
        task = self._pending_task()

        await scheduler._apply_external_dep_policy(
            [task],
            {'dark_factory:99': 'unknown_task'},   # cache has a sentinel but err is set
            external_err=RuntimeError('db unavailable'),
        )

        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_transient_err_no_counter_increment(self, scheduler: Scheduler):
        """external_err set → _external_unresolved_counts stays empty (no increment)."""
        scheduler._on_external_dep_block = AsyncMock()
        task = self._pending_task()

        await scheduler._apply_external_dep_policy(
            [task],
            {'dark_factory:99': 'malformed'},
            external_err=RuntimeError('registry timeout'),
        )

        assert scheduler._external_unresolved_counts == {}, (
            f'Expected empty counter dict; got {scheduler._external_unresolved_counts!r}'
        )

    @pytest.mark.asyncio
    async def test_transient_err_does_not_reset_prior_counter(
        self, scheduler: Scheduler
    ):
        """A transient error must not reset a counter built up in prior ticks."""
        scheduler._on_external_dep_block = AsyncMock()
        task = self._pending_task()

        # Tick 1 — sentinel, no err → counter = 1
        await scheduler._apply_external_dep_policy(
            [task], {'dark_factory:99': 'unknown_task'}, external_err=None
        )
        assert scheduler._external_unresolved_counts.get(('20', 'dark_factory:99')) == 1

        # Tick 2 — transient err → counter must stay at 1 (no reset, no increment)
        await scheduler._apply_external_dep_policy(
            [task],
            {'dark_factory:99': 'unknown_task'},
            external_err=RuntimeError('blip'),
        )
        assert scheduler._external_unresolved_counts.get(('20', 'dark_factory:99')) == 1, (
            f'Counter must be untouched on transient err; '
            f'got {scheduler._external_unresolved_counts!r}'
        )

    def test_deps_satisfied_external_resolver_failed_blocks(
        self, scheduler: Scheduler
    ):
        """external_resolver_failed=True → _deps_satisfied returns False (gate closed)."""
        task = {
            'id': '20',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'external_deps': ['dark_factory:99']},
        }
        # Even with a 'done' entry in the cache, resolver_failed overrides
        result = scheduler._deps_satisfied(
            task,
            {},
            external_status_cache={'dark_factory:99': 'done'},
            external_resolver_failed=True,
        )
        assert result is False, (
            'external_resolver_failed=True must block dispatch regardless of cache'
        )


# ---------------------------------------------------------------------------
# TestExternalDepGateHeld_ResolverDegraded (task 1799 — step-5 RED / step-6 GREEN)
# ---------------------------------------------------------------------------

class TestExternalDepGateHeld_ResolverDegraded:
    """_apply_external_dep_policy emits external_dep_gate_held when resolver is degraded.

    When ``external_err is not None`` (degraded resolver tick):
    - After ``threshold`` consecutive degraded ticks, exactly ONE
      ``EventType.external_dep_gate_held`` event must be recorded with
      ``cause='resolver_degraded'``, ``task_id='T'``, and ``ticks==threshold``.
    - ``_external_unresolved_counts`` must NOT have any entry — degraded ticks
      must NOT bump the sentinel counter (fail-safe invariant from task 1580).
    - No exception must be raised even when ``_on_external_dep_block`` is None
      (no escalation on a degraded tick — only an event).

    This fails today because ``EventType.external_dep_gate_held`` does not exist
    and the degraded branch simply returns without emitting anything.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config, event_store=_RecordingEventStore())  # type: ignore[arg-type]

    def _pending_task_with_ext(self, task_id: str = 'T') -> dict:
        return {
            'id': task_id,
            'status': 'pending',
            'dependencies': [],
            'metadata': {'external_deps': ['upstream_proj:1']},
        }

    @pytest.mark.asyncio
    async def test_degraded_resolver_emits_gate_held_at_threshold(
        self, scheduler: Scheduler
    ):
        """After threshold degraded ticks → one external_dep_gate_held(cause='resolver_degraded')."""
        threshold = scheduler.config.max_external_dep_unresolved_cycles
        task = self._pending_task_with_ext()

        for _ in range(threshold):
            await scheduler._apply_external_dep_policy(
                [task],
                {},  # empty cache — resolver degraded
                ExternalResolverError('simulated degraded resolver'),
            )

        _event_store = scheduler.event_store
        assert _event_store is not None
        gate_held_events = [
            (evt, data)
            for evt, data in _event_store.events  # type: ignore[attr-defined]
            if evt == str(EventType.external_dep_gate_held)
        ]
        assert len(gate_held_events) == 1, (
            f'Expected exactly 1 external_dep_gate_held event after {threshold} degraded ticks; '
            f'got {len(gate_held_events)}: {gate_held_events!r}'
        )
        _evt_type, evt_data = gate_held_events[0]
        assert evt_data['task_id'] == 'T', (
            f'Expected task_id="T"; got {evt_data["task_id"]!r}'
        )
        assert evt_data['data'].get('cause') == 'resolver_degraded', (
            f'Expected cause="resolver_degraded"; got {evt_data["data"]!r}'
        )
        assert evt_data['data'].get('ticks') == threshold, (
            f'Expected ticks={threshold}; got {evt_data["data"].get("ticks")!r}'
        )

    @pytest.mark.asyncio
    async def test_degraded_resolver_does_not_bump_sentinel_counter(
        self, scheduler: Scheduler
    ):
        """Degraded ticks must NOT touch _external_unresolved_counts (fail-safe invariant)."""
        threshold = scheduler.config.max_external_dep_unresolved_cycles
        task = self._pending_task_with_ext()

        for _ in range(threshold):
            await scheduler._apply_external_dep_policy(
                [task],
                {'upstream_proj:1': 'unknown_task'},  # cache has a sentinel
                ExternalResolverError('degraded'),
            )

        assert scheduler._external_unresolved_counts == {}, (
            f'Degraded ticks must NOT bump sentinel counter; '
            f'got {scheduler._external_unresolved_counts!r}'
        )

    @pytest.mark.asyncio
    async def test_degraded_resolver_no_escalation_without_callback(
        self, scheduler: Scheduler
    ):
        """No exception raised when _on_external_dep_block is None (degraded → event only)."""
        threshold = scheduler.config.max_external_dep_unresolved_cycles
        task = self._pending_task_with_ext()
        # _on_external_dep_block is None by default — must not raise.
        for _ in range(threshold):
            await scheduler._apply_external_dep_policy(
                [task],
                {},
                ExternalResolverError('degraded'),
            )
        # No assertion needed — the test passes if no exception was raised.


# ---------------------------------------------------------------------------
# TestExternalDepGateHeld_DepsLive (task 1799 — step-7 RED / step-8 GREEN)
# ---------------------------------------------------------------------------

class TestExternalDepGateHeld_DepsLive:
    """_apply_external_dep_policy emits external_dep_gate_held when deps are live (not done).

    When the resolver is OK (``external_err is None``) but a dep's status is a live
    status (e.g. 'pending', 'in-progress') — NOT a sentinel — the task stays held.

    After ``threshold`` consecutive held ticks:
    - Exactly ONE ``EventType.external_dep_gate_held`` event must be recorded with
      ``cause='deps_live'``, ``task_id='T'``, ``ticks==threshold``.
    - ``_external_unresolved_counts`` must NOT have a ``('T','upstream_proj:1')`` entry
      (live statuses are NOT sentinels; the sentinel counter stays clean).
    - After the hold resolves (dep becomes 'done') the hold streak is reset:
      NO additional gate_held event; ``_external_hold_streak`` has no 'T' entry.

    Fails today because the live-status else-branch does not set any ``held_live``
    flag, so ``_note_external_hold`` is never called for live-status holds.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config, event_store=_RecordingEventStore())  # type: ignore[arg-type]

    def _pending_task_with_ext(self, task_id: str = 'T') -> dict:
        return {
            'id': task_id,
            'status': 'pending',
            'dependencies': [],
            'metadata': {'external_deps': ['upstream_proj:1']},
        }

    @pytest.mark.asyncio
    async def test_live_deps_emits_gate_held_at_threshold(
        self, scheduler: Scheduler
    ):
        """After threshold live-dep ticks → one external_dep_gate_held(cause='deps_live')."""
        threshold = scheduler.config.max_external_dep_unresolved_cycles
        task = self._pending_task_with_ext()

        # Resolver OK; dep is live (pending) — not done, not a sentinel.
        for _ in range(threshold):
            await scheduler._apply_external_dep_policy(
                [task],
                {'upstream_proj:1': 'pending'},
                None,
            )

        _event_store = scheduler.event_store
        assert _event_store is not None
        gate_held_events = [
            (evt, data)
            for evt, data in _event_store.events  # type: ignore[attr-defined]
            if evt == str(EventType.external_dep_gate_held)
        ]
        assert len(gate_held_events) == 1, (
            f'Expected exactly 1 external_dep_gate_held event after {threshold} '
            f'live-dep ticks; got {len(gate_held_events)}: {gate_held_events!r}'
        )
        _evt_type, evt_data = gate_held_events[0]
        assert evt_data['task_id'] == 'T', (
            f'Expected task_id="T"; got {evt_data["task_id"]!r}'
        )
        assert evt_data['data'].get('cause') == 'deps_live', (
            f'Expected cause="deps_live"; got {evt_data["data"]!r}'
        )
        assert evt_data['data'].get('ticks') == threshold, (
            f'Expected ticks={threshold}; got {evt_data["data"].get("ticks")!r}'
        )

    @pytest.mark.asyncio
    async def test_live_deps_does_not_bump_sentinel_counter(
        self, scheduler: Scheduler
    ):
        """Live-dep ticks must NOT touch _external_unresolved_counts (live ≠ sentinel)."""
        threshold = scheduler.config.max_external_dep_unresolved_cycles
        task = self._pending_task_with_ext()

        for _ in range(threshold):
            await scheduler._apply_external_dep_policy(
                [task],
                {'upstream_proj:1': 'in-progress'},
                None,
            )

        assert scheduler._external_unresolved_counts == {}, (
            f'Live-dep ticks must NOT bump sentinel counter; '
            f'got {scheduler._external_unresolved_counts!r}'
        )

    @pytest.mark.asyncio
    async def test_hold_streak_resets_on_resolution(
        self, scheduler: Scheduler
    ):
        """When dep becomes 'done', streak is cleared and no additional event is emitted."""
        threshold = scheduler.config.max_external_dep_unresolved_cycles
        task = self._pending_task_with_ext()

        # Build up a streak of threshold ticks.
        for _ in range(threshold):
            await scheduler._apply_external_dep_policy(
                [task],
                {'upstream_proj:1': 'pending'},
                None,
            )

        # Count events so far.
        _event_store = scheduler.event_store
        assert _event_store is not None
        events_before = [
            e for e in _event_store.events  # type: ignore[attr-defined]
            if e[0] == str(EventType.external_dep_gate_held)
        ]
        assert len(events_before) == 1, 'Setup: expected 1 gate_held event after threshold ticks'

        # Now dep resolves to 'done'.
        await scheduler._apply_external_dep_policy(
            [task],
            {'upstream_proj:1': 'done'},
            None,
        )

        # No additional event.
        events_after = [
            e for e in _event_store.events  # type: ignore[attr-defined]
            if e[0] == str(EventType.external_dep_gate_held)
        ]
        assert len(events_after) == 1, (
            f'After resolution: no additional gate_held event expected; '
            f'got {events_after!r}'
        )

        # Streak cleared.
        assert 'T' not in scheduler._external_hold_streak, (
            f'After dep done: _external_hold_streak must have no "T" entry; '
            f'got {scheduler._external_hold_streak!r}'
        )


# ---------------------------------------------------------------------------
# TestAcquireNextExternalDepGate (task 1580 — step-11 RED / step-12 GREEN)
# ---------------------------------------------------------------------------

class TestAcquireNextExternalDepGate:
    """acquire_next wires the external-dep gate: one batched call, correct dispatch decisions.

    Invariants under test:
    - 5: exactly ONE get_external_statuses call per tick (union of all pending deps);
         ZERO calls when no pending task has external deps.
    - 1: external dep 'done' + local deps done → task IS dispatched.
    - boundary row 2: external dep 'pending' → not dispatched, no escalation.
    - boundary row 3: external dep 'cancelled' → not dispatched, callback fires.
    - boundary row 8: local deps done + external dep 'pending' → not dispatched.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=2)
        return Scheduler(config)

    def _task(self, tid: str, ext_deps: list[str] | None = None) -> dict:
        return {
            'id': tid,
            'title': f'Task {tid}',
            'status': 'pending',
            'dependencies': [],
            'metadata': {
                'files': ['backend'],
                **({'external_deps': ext_deps} if ext_deps else {}),
            },
        }

    @pytest.mark.asyncio
    async def test_no_external_deps_zero_calls(self, scheduler: Scheduler):
        """Invariant 5: zero pending tasks with external_deps → zero get_external_statuses calls."""
        scheduler.get_tasks = AsyncMock(return_value=[self._task('1')])
        scheduler.get_external_statuses = AsyncMock(return_value=({}, None))

        result = await scheduler.acquire_next()
        assert result is not None and result.task_id == '1'

        scheduler.get_external_statuses.assert_not_called()

    @pytest.mark.asyncio
    async def test_one_batched_call_covering_all_pending_deps(
        self, scheduler: Scheduler
    ):
        """Invariant 5: all external deps across pending tasks batched into ONE call."""
        task_a = self._task('1', ext_deps=['proj:10'])
        task_b = self._task('2', ext_deps=['proj:20'])
        scheduler.get_tasks = AsyncMock(return_value=[task_a, task_b])
        scheduler.get_external_statuses = AsyncMock(
            return_value=({'proj:10': 'done', 'proj:20': 'done'}, None)
        )
        scheduler._apply_external_dep_policy = AsyncMock()

        await scheduler.acquire_next()

        # Exactly ONE call with the union of deps (order-independent)
        scheduler.get_external_statuses.assert_called_once()
        call_args = scheduler.get_external_statuses.call_args
        passed_deps = set(call_args.args[0] if call_args.args else call_args.kwargs.get('deps', []))
        assert passed_deps == {'proj:10', 'proj:20'}, (
            f'Expected union of all pending task external_deps; got {passed_deps!r}'
        )

    @pytest.mark.asyncio
    async def test_external_dep_done_task_dispatched(self, scheduler: Scheduler):
        """Invariant 1: external dep 'done' (and no local deps) → task IS dispatched."""
        task = self._task('3', ext_deps=['dark_factory:5'])
        scheduler.get_tasks = AsyncMock(return_value=[task])
        scheduler.get_external_statuses = AsyncMock(
            return_value=({'dark_factory:5': 'done'}, None)
        )

        result = await scheduler.acquire_next()

        assert result is not None and result.task_id == '3', (
            f'External dep done → should dispatch; got {result!r}'
        )

    @pytest.mark.asyncio
    async def test_external_dep_pending_not_dispatched_no_callback(
        self, scheduler: Scheduler
    ):
        """Boundary row 2: external dep 'pending' → not dispatched, no escalation."""
        callback = AsyncMock()
        scheduler._on_external_dep_block = callback

        task = self._task('4', ext_deps=['dark_factory:5'])
        scheduler.get_tasks = AsyncMock(return_value=[task])
        scheduler.get_external_statuses = AsyncMock(
            return_value=({'dark_factory:5': 'pending'}, None)
        )

        result = await scheduler.acquire_next()

        assert result is None, (
            f'External dep pending → must NOT dispatch; got {result!r}'
        )
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_external_dep_cancelled_not_dispatched_callback_fires(
        self, scheduler: Scheduler
    ):
        """Boundary row 3: external dep 'cancelled' → not dispatched, block callback fires."""
        callback = AsyncMock()
        scheduler._on_external_dep_block = callback

        task = self._task('5', ext_deps=['dark_factory:5'])
        scheduler.get_tasks = AsyncMock(return_value=[task])
        scheduler.get_external_statuses = AsyncMock(
            return_value=({'dark_factory:5': 'cancelled'}, None)
        )

        result = await scheduler.acquire_next()

        assert result is None, (
            f'External dep cancelled → must NOT dispatch; got {result!r}'
        )
        callback.assert_called_once()
        kwargs = callback.call_args.kwargs
        assert 'EXTERNAL_DEP_CANCELLED' in kwargs.get('summary', ''), (
            f'Expected EXTERNAL_DEP_CANCELLED in summary; got {kwargs!r}'
        )

    @pytest.mark.asyncio
    async def test_local_deps_done_external_dep_pending_not_dispatched(
        self, scheduler: Scheduler
    ):
        """Boundary row 8: all local deps done but external dep 'pending' → not dispatched."""
        dep_task = {
            'id': '10',
            'status': 'done',
            'dependencies': [],
            'metadata': {},
        }
        task = {
            'id': '11',
            'title': 'Task 11',
            'status': 'pending',
            'dependencies': ['10'],
            'metadata': {
                'files': ['backend'],
                'external_deps': ['dark_factory:99'],
            },
        }
        scheduler.get_tasks = AsyncMock(return_value=[dep_task, task])
        scheduler.get_external_statuses = AsyncMock(
            return_value=({'dark_factory:99': 'in-progress'}, None)
        )

        result = await scheduler.acquire_next()

        assert result is None, (
            f'Local deps done but external dep in-progress → must NOT dispatch; got {result!r}'
        )


# ---------------------------------------------------------------------------
# Pair E — Scheduler._suppress_blocked_write hook (task 1620, step-9)
# RED until step-10 declares the hook attr and inserts the pre-dispatch guard.
# ---------------------------------------------------------------------------

class TestSuppressBlockedWrite:
    """Scheduler._suppress_blocked_write: pre-dispatch suppression guard for 'blocked' writes.

    Pair E (task 1620), step-9.  RED until step-10 adds:
      1. ``self._suppress_blocked_write: Callable[[str], bool] | None = None`` in __init__
         alongside the existing _on_park_stop_trip / _on_external_dep_block declarations.
      2. A guard at the TOP of ``set_task_status`` — BEFORE the
         ``for attempt in range(_TRANSIENT_RETRIES)`` retry loop — that returns
         early when ``status == 'blocked'`` and the predicate flags that task_id.
         (Inserting it at the post-write success branch :1190 would still let the
         write reach fused-memory, defeating C3.2 — see plan design decision 7.)
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @pytest.mark.asyncio
    async def test_baseline_blocked_calls_dispatch_and_records_transition(
        self, scheduler: Scheduler
    ):
        """(a) Without any suppression hook, set_task_status(tid, 'blocked') calls
        dispatch_tool and records the blocked transition in the rolling window.

        Baseline / regression anchor — must stay GREEN before and after step-10.
        """
        scheduler.dispatch_tool = AsyncMock(return_value={})  # empty dict → success
        tid = 'task-baseline'

        await scheduler.set_task_status(tid, 'blocked')

        scheduler.dispatch_tool.assert_called_once()
        # Blocked transition should be recorded (parked_live_count tracks unique tids in window).
        assert scheduler.parked_live_count == 1, (
            f'Expected blocked transition recorded; parked_live_count={scheduler.parked_live_count}'
        )

    @pytest.mark.asyncio
    async def test_suppressed_blocked_skips_dispatch_and_transition(
        self, scheduler: Scheduler
    ):
        """(b) With ``_suppress_blocked_write = lambda t: t == tid`` installed,
        ``set_task_status(tid, 'blocked')`` must NOT call dispatch_tool and must NOT
        record a blocked transition.

        RED until step-10 inserts the pre-dispatch guard.  The RED assertion is that
        dispatch_tool is NOT called — a guard placed at :1190 (post-write success
        branch) would still fail this because the write reaches dispatch_tool first.
        """
        scheduler.dispatch_tool = AsyncMock(return_value={})
        tid = 'task-suppress'
        # Install the suppression predicate: suppress only this specific tid.
        scheduler._suppress_blocked_write = lambda t: t == tid

        await scheduler.set_task_status(tid, 'blocked')

        # RED: guard does not exist yet → dispatch_tool IS called → assertion fails.
        scheduler.dispatch_tool.assert_not_called()
        # No blocked transition recorded either.
        assert scheduler.parked_live_count == 0, (
            f'Suppressed write must not record a transition; got {scheduler.parked_live_count}'
        )

    @pytest.mark.asyncio
    async def test_suppression_applies_only_to_blocked_status(
        self, scheduler: Scheduler
    ):
        """(c) Suppression is status-specific: a 'pending' write for the same tid
        still calls dispatch_tool even when the hook returns True for that tid.
        """
        scheduler.dispatch_tool = AsyncMock(return_value={})
        tid = 'task-pending-not-suppressed'
        # Suppress blocked writes for this tid.
        scheduler._suppress_blocked_write = lambda t: t == tid

        # A 'pending' write must still dispatch.
        await scheduler.set_task_status(tid, 'pending')

        scheduler.dispatch_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_suppression_does_not_affect_other_task_ids(
        self, scheduler: Scheduler
    ):
        """With hook suppressing only 'task-A', a 'blocked' write for 'task-B'
        still dispatches normally and records the transition.
        """
        scheduler.dispatch_tool = AsyncMock(return_value={})
        # Only 'task-A' is suppressed.
        scheduler._suppress_blocked_write = lambda t: t == 'task-A'

        await scheduler.set_task_status('task-B', 'blocked')

        scheduler.dispatch_tool.assert_called_once()
        assert scheduler.parked_live_count == 1


# ---------------------------------------------------------------------------
# step-1 RED: Scheduler.get_tasks forwards statuses kwarg into dispatch_tool
# ---------------------------------------------------------------------------

class TestGetTasksStatusesParam:
    """get_tasks(statuses=[...]) must forward 'statuses' into dispatch_tool args.

    get_tasks() with no arg must omit 'statuses' entirely (full-fetch preserved).
    """

    @staticmethod
    def _envelope(tasks: list) -> dict:
        import json as _json
        return {
            'result': {
                'content': [
                    {'type': 'text', 'text': _json.dumps({'tasks': tasks})}
                ]
            }
        }

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @pytest.mark.asyncio
    async def test_get_tasks_with_statuses_forwards_arg(self, scheduler: Scheduler):
        """Passing statuses=[...] results in 'statuses' present in dispatch_tool args."""
        scheduler.dispatch_tool = AsyncMock(return_value=self._envelope([]))
        await scheduler.get_tasks(statuses=['pending', 'in-progress'])
        call_args = scheduler.dispatch_tool.call_args
        # Second positional arg is the arguments dict
        arguments = call_args[0][1] if call_args[0] else call_args.kwargs.get('arguments', {})
        # Support both positional and keyword call styles
        if not arguments and len(call_args[0]) > 1:
            arguments = call_args[0][1]
        assert 'statuses' in arguments, (
            f"Expected 'statuses' key in dispatch_tool arguments but got: {arguments}"
        )
        assert arguments['statuses'] == ['pending', 'in-progress']

    @pytest.mark.asyncio
    async def test_get_tasks_no_arg_omits_statuses(self, scheduler: Scheduler):
        """Calling get_tasks() without statuses must NOT include 'statuses' in dispatch_tool args."""
        scheduler.dispatch_tool = AsyncMock(return_value=self._envelope([]))
        await scheduler.get_tasks()
        call_args = scheduler.dispatch_tool.call_args
        arguments = call_args[0][1] if (call_args[0] and len(call_args[0]) > 1) else {}
        assert 'statuses' not in arguments, (
            f"'statuses' should be absent from dispatch_tool arguments but got: {arguments}"
        )

    @pytest.mark.asyncio
    async def test_get_tasks_statuses_none_omits_statuses(self, scheduler: Scheduler):
        """get_tasks(statuses=None) must NOT include 'statuses' in dispatch_tool args."""
        scheduler.dispatch_tool = AsyncMock(return_value=self._envelope([]))
        await scheduler.get_tasks(statuses=None)
        call_args = scheduler.dispatch_tool.call_args
        arguments = call_args[0][1] if (call_args[0] and len(call_args[0]) > 1) else {}
        assert 'statuses' not in arguments, (
            f"'statuses' should be absent when None but got: {arguments}"
        )


# ---------------------------------------------------------------------------
# step-3 RED: acquire_next() calls get_tasks with ACTIVE_TASK_STATUSES
# ---------------------------------------------------------------------------

class TestAcquireNextFetchesActiveOnly:
    """acquire_next() must call get_tasks with statuses=ACTIVE_TASK_STATUSES.

    Fails today because acquire_next calls self.get_tasks() with no statuses arg.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @pytest.mark.asyncio
    async def test_acquire_next_passes_active_statuses_to_get_tasks(
        self, scheduler: Scheduler
    ):
        """acquire_next() must issue get_tasks with statuses==ACTIVE_TASK_STATUSES."""
        pending_task = {
            'id': '42',
            'title': 'A pending task',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['backend/module_a']},
        }
        scheduler.get_tasks = AsyncMock(return_value=[pending_task])
        scheduler.get_statuses = AsyncMock(return_value=({}, None))

        await scheduler.acquire_next()

        scheduler.get_tasks.assert_awaited_once()
        call_kwargs = scheduler.get_tasks.call_args.kwargs
        assert 'statuses' in call_kwargs, (
            f"acquire_next must call get_tasks with statuses=ACTIVE_TASK_STATUSES, "
            f"but call_args.kwargs was: {call_kwargs}"
        )
        assert set(call_kwargs['statuses']) == ACTIVE_TASK_STATUSES, (
            f"Expected statuses {ACTIVE_TASK_STATUSES}, got {set(call_kwargs['statuses'])}"
        )


# ---------------------------------------------------------------------------
# step-5 RED: correctness crux — done dep absent from active fetch still satisfies
# ---------------------------------------------------------------------------

class TestAcquireNextDepBackfillFromGetStatuses:
    """A task whose dep is DONE (absent from active get_tasks) still dispatches.

    Active-only fetch drops terminal tasks from the result. acquire_next must
    backfill their status via get_statuses so _deps_satisfied can resolve them.

    Fails today: acquire_next derives status_map only from [B], so dep A
    resolves to 'unknown' → B is blocked → acquire_next returns None, and
    get_statuses is never called.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @pytest.mark.asyncio
    async def test_done_dep_absent_from_active_fetch_still_dispatches(
        self, scheduler: Scheduler
    ):
        """Task B dispatches even when its dep A is absent from the active get_tasks result."""
        # A is done and excluded by the active filter.
        task_b = {
            'id': 'B',
            'title': 'Task B — depends on A',
            'status': 'pending',
            'dependencies': [{'id': 'A'}],
            'metadata': {'files': ['backend/module_b']},
        }
        # get_tasks returns only B (A is terminal → excluded by active filter).
        scheduler.get_tasks = AsyncMock(return_value=[task_b])
        # get_statuses is called to backfill the missing dep A → returns done.
        scheduler.get_statuses = AsyncMock(return_value=({'A': 'done'}, None))

        result = await scheduler.acquire_next()

        # B should have been dispatched.
        assert result is not None, (
            'Expected task B to be dispatched when its done dep A is backfilled '
            'from get_statuses, but acquire_next returned None'
        )
        assert result.task_id == 'B'

        # get_statuses must have been called with dep A to resolve the missing status.
        scheduler.get_statuses.assert_awaited()
        call_kwargs = scheduler.get_statuses.call_args.kwargs
        ids_called = call_kwargs.get('ids', scheduler.get_statuses.call_args[0][0] if scheduler.get_statuses.call_args[0] else [])
        assert 'A' in ids_called, (
            f"Expected get_statuses to be called with 'A' in ids, "
            f"but got ids={ids_called}"
        )


# ---------------------------------------------------------------------------
# TestAcquireNextLocalBackfillFailsSafe (task 1807 — step-7 RED / step-8 GREEN)
# ---------------------------------------------------------------------------

class TestAcquireNextLocalBackfillFailsSafe:
    """acquire_next() must fail-safe-wait + emit WARNING when the dep-backfill get_statuses degrades.

    Drives a full acquire_next() tick via mocked ``dispatch_tool`` (not
    ``get_statuses`` directly) so the real ``get_statuses`` parse is exercised:

    - ``get_tasks`` returns pending B with local dep A.
    - ``get_statuses`` is fed a malformed response (non-dict 'statuses') so the
      real method returns ``({}, EnvelopeParseError)``.
    - The backfill failure must be VISIBLE: ``orchestrator.scheduler`` WARNING
      naming the degraded dep ids, and ``_local_backfill_unresolved_counts``
      bumped for ``('B', 'A')``.
    - B remains held (fail-safe-wait) — NOT dispatched.

    Fails today: ``_backfill_err`` is ignored → no scheduler WARNING, no
    counter (silent strand).
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @staticmethod
    def _envelope(payload: dict) -> dict:
        import json as _json
        return {
            'result': {
                'content': [{'type': 'text', 'text': _json.dumps(payload)}]
            }
        }

    @pytest.mark.asyncio
    async def test_malformed_backfill_emits_warning_and_holds_task(
        self, scheduler: Scheduler, caplog
    ):
        """Malformed get_statuses response → WARNING + counter bump + B not dispatched.

        Fails today: _backfill_err is ignored → no WARNING from orchestrator.scheduler,
        no _local_backfill_unresolved_counts attribute.
        """
        import logging

        task_b = {
            'id': 'B',
            'title': 'Task B — depends on A',
            'status': 'pending',
            'dependencies': [{'id': 'A'}],
            'metadata': {'files': ['backend/module_b']},
        }

        async def _dispatch(tool_name, arguments, **kwargs):
            if tool_name == 'get_tasks':
                # Valid response: only B is active (A is done, absent from active fetch).
                return self._envelope({'tasks': [task_b]})
            if tool_name == 'get_statuses':
                # Malformed: 'statuses' is a list, not a dict → real get_statuses
                # returns ({}, EnvelopeParseError).
                return self._envelope({'statuses': ['not', 'a', 'dict']})
            return {}

        scheduler.dispatch_tool = AsyncMock(side_effect=_dispatch)

        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduler'):
            result = await scheduler.acquire_next()

        # B must NOT be dispatched — fail-safe-wait (dep A status unknown).
        assert result is None, (
            f'Expected None (fail-safe-wait) but got task_id={result.task_id!r}; '
            'backfill failure should hold B, not dispatch it'
        )

        # orchestrator.scheduler must emit a WARNING naming the backfill degradation.
        sched_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and r.name == 'orchestrator.scheduler'
        ]
        assert sched_warnings, (
            'Expected a WARNING from orchestrator.scheduler about backfill degradation; '
            f'got records={[(r.name, r.getMessage()) for r in caplog.records]!r}'
        )

        # _local_backfill_unresolved_counts must be bumped for (B, A).
        assert hasattr(scheduler, '_local_backfill_unresolved_counts'), (
            '_local_backfill_unresolved_counts attribute must exist after backfill failure'
        )
        count = scheduler._local_backfill_unresolved_counts.get(('B', 'A'), 0)
        assert count >= 1, (
            f'Expected _local_backfill_unresolved_counts[(B, A)] >= 1; got {count}'
        )

    @pytest.mark.asyncio
    async def test_counter_resets_on_success_after_failure(
        self, scheduler: Scheduler, caplog
    ):
        """Counter must be RESET (popped) when backfill succeeds after a prior failure.

        Amendment 1 (suggestion 1 from reviewer): _local_backfill_unresolved_counts
        claims consecutive-tick semantics in its comment and warning message, but
        without a reset-on-success the count accumulates across the gap.  The fix
        mirrors _external_unresolved_counts.pop(...) on 'done' in
        _apply_external_dep_policy.

        Step 1: fail → counter bumped for (B, A).
        Step 2: success → counter popped for (B, A).
        """
        import logging

        task_b = {
            'id': 'B',
            'title': 'Task B — depends on A',
            'status': 'pending',
            'dependencies': [{'id': 'A'}],
            'metadata': {'files': ['backend/module_b']},
        }

        call_count = {'n': 0}

        async def _dispatch_fail_then_succeed(tool_name, arguments, **kwargs):
            if tool_name == 'get_tasks':
                return self._envelope({'tasks': [task_b]})
            if tool_name == 'get_statuses':
                call_count['n'] += 1
                if call_count['n'] == 1:
                    # First call: malformed → EnvelopeParseError
                    return self._envelope({'statuses': ['not', 'a', 'dict']})
                else:
                    # Second call: A now resolves as done
                    return self._envelope({'statuses': {'A': 'done'}})
            return {}

        scheduler.dispatch_tool = AsyncMock(side_effect=_dispatch_fail_then_succeed)

        # Tick 1: backfill fails → counter bumped
        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduler'):
            await scheduler.acquire_next()

        assert scheduler._local_backfill_unresolved_counts.get(('B', 'A'), 0) >= 1, (
            'Counter should be > 0 after first (failed) tick'
        )

        # Tick 2: backfill succeeds → counter must be reset (popped)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduler'):
            await scheduler.acquire_next()

        remaining = scheduler._local_backfill_unresolved_counts.get(('B', 'A'), 0)
        assert remaining == 0, (
            f'Expected counter reset to 0 after successful backfill; got {remaining}'
        )

    @pytest.mark.asyncio
    async def test_partial_backfill_warns_for_missing_ids(
        self, scheduler: Scheduler, caplog
    ):
        """Partial backfill (valid dict but missing some requested ids) → WARNING + counter.

        Amendment 2 (suggestion 2 from reviewer): get_statuses returns a non-empty
        dict that omits some of the requested dep ids.  resolver_failed() returns
        False (no error, non-empty), so the old code updated status_map and held
        the still-missing dep silently.  The fix mirrors the missing-key guard in
        get_external_statuses: compute still_missing and treat it as degraded.

        Setup:
        - B depends on both A and C.
        - get_tasks returns [B] (active only; A and C are done, absent from active).
        - get_statuses returns {'C': 'done'} — C resolved but A is missing from
          the response (partial).

        Expected:
        - orchestrator.scheduler WARNING naming A as the missing dep.
        - _local_backfill_unresolved_counts[('B', 'A')] bumped.
        - _local_backfill_unresolved_counts[('B', 'C')] NOT bumped (resolved).
        - B NOT dispatched (A still absent from status_map → deps not satisfied).
        """
        import logging

        task_b = {
            'id': 'B',
            'title': 'Task B — depends on A and C',
            'status': 'pending',
            'dependencies': [{'id': 'A'}, {'id': 'C'}],
            'metadata': {'files': ['backend/module_b']},
        }

        async def _dispatch(tool_name, arguments, **kwargs):
            if tool_name == 'get_tasks':
                return self._envelope({'tasks': [task_b]})
            if tool_name == 'get_statuses':
                # Partial: C resolved, A absent (partial response)
                return self._envelope({'statuses': {'C': 'done'}})
            return {}

        scheduler.dispatch_tool = AsyncMock(side_effect=_dispatch)

        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduler'):
            result = await scheduler.acquire_next()

        # B must NOT be dispatched (A is still unknown)
        assert result is None, (
            f'Expected None (fail-safe-wait for missing A) but got {result!r}'
        )

        # WARNING must be emitted naming the partial result
        partial_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
            and r.name == 'orchestrator.scheduler'
            and 'partial' in r.getMessage().lower()
        ]
        assert partial_warnings, (
            'Expected a WARNING about partial backfill result; '
            f'got records={[(r.name, r.getMessage()) for r in caplog.records]!r}'
        )

        # Counter bumped for still-missing A, not for resolved C
        count_a = scheduler._local_backfill_unresolved_counts.get(('B', 'A'), 0)
        assert count_a >= 1, (
            f'Expected counter bumped for (B, A); got {count_a}'
        )
        count_c = scheduler._local_backfill_unresolved_counts.get(('B', 'C'), 0)
        assert count_c == 0, (
            f'Expected counter 0 for resolved (B, C); got {count_c}'
        )


# ---------------------------------------------------------------------------
# step-7 RED: bookkeeping purged for tasks absent from active fetch (not only terminal)
# ---------------------------------------------------------------------------

class TestAcquireNextBookkeepingPurgesAbsentTasks:
    """Per-tick bookkeeping must be purged for completed tasks absent from active fetch.

    Active-only filtering drops completed tasks from get_tasks. The existing
    terminal-cleanup sweep only purges ids observed TERMINAL in status_map.
    Tasks absent from the active result won't appear in status_map at all,
    so their _skip_count/_last_dispatch_at/_module_cache entries would leak.

    Fails today because the sweep iterates status_map items looking for
    TERMINAL status — but 'X' is absent from the active result → not in
    status_map → never purged.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    @pytest.mark.asyncio
    async def test_bookkeeping_purged_for_absent_task(self, scheduler: Scheduler):
        """Bookkeeping entries for a task absent from the active fetch must be purged.

        Includes _pending_anchor: a task that went pending → terminal without
        ever being dispatched would leak its anchor entry permanently under
        active-only filtering because _update_age_anchors only iterates the
        fetched tasks list (and terminal tasks are absent from it).
        """
        # Seed bookkeeping for task 'X' that is now completed/absent.
        scheduler._skip_count['X'] = 3
        scheduler._last_dispatch_at['X'] = 123.0
        scheduler._module_cache['X'] = ['backend/module_x']
        # Seed _pending_anchor as if X was pending once and assigned an anchor.
        scheduler._pending_anchor['X'] = 5

        # Unrelated pending task (not X); X is terminal → excluded from active fetch.
        other_task = {
            'id': '99',
            'title': 'Other pending task',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['backend/module_other']},
        }
        scheduler.get_tasks = AsyncMock(return_value=[other_task])
        scheduler.get_statuses = AsyncMock(return_value=({}, None))

        await scheduler.acquire_next()

        # X is absent from the active result → bookkeeping must have been purged.
        assert 'X' not in scheduler._skip_count, (
            f"Expected 'X' purged from _skip_count but it's still there: {scheduler._skip_count}"
        )
        assert 'X' not in scheduler._last_dispatch_at, (
            f"Expected 'X' purged from _last_dispatch_at but it's still there: {scheduler._last_dispatch_at}"
        )
        assert 'X' not in scheduler._module_cache, (
            f"Expected 'X' purged from _module_cache but it's still there: {scheduler._module_cache}"
        )
        # _pending_anchor must also be purged so the dict stays bounded.
        assert 'X' not in scheduler._pending_anchor, (
            f"Expected 'X' purged from _pending_anchor but it's still there: {scheduler._pending_anchor}"
        )
        # _was_non_pending must record X so that if it is resurrected to pending
        # it gets a fresh max_id anchor (resurrection semantics) instead of
        # re-using its old stale numeric id.
        assert 'X' in scheduler._was_non_pending, (
            f"Expected 'X' recorded in _was_non_pending for resurrection semantics: {scheduler._was_non_pending}"
        )


class TestActiveTaskStatusesMatchesFusedMemory:
    """Guard: ACTIVE_TASK_STATUSES in task_status.py must stay in sync with
    fused-memory's canonical definition.

    If the server adds a new active status and the orchestrator's local copy
    is not updated, tasks in that status would be silently excluded from the
    active get_tasks fetch and never dispatched.  This test makes divergence
    fail CI rather than strand tasks silently.
    """

    def test_active_task_statuses_matches_fused_memory(self):
        """orchestrator.task_status.ACTIVE_TASK_STATUSES == fused_memory canonical.

        fused_memory is not on the orchestrator test path (cross-package import
        is intentionally avoided per design — mirrored, not imported).  Instead
        we compare against the hardcoded canonical set from
        fused_memory/src/fused_memory/reconciliation/task_filter.py:66.
        Update BOTH files when the server adds a new active status.
        """
        from orchestrator.task_status import ACTIVE_TASK_STATUSES as orch_set

        # Canonical active statuses as defined in
        # fused_memory/reconciliation/task_filter.py (task_filter.ACTIVE_TASK_STATUSES).
        # Keep in sync with that file manually — divergence here is the signal.
        fm_canonical: frozenset[str] = frozenset(
            {
                'pending',
                'in-progress',
                'blocked',
                'deferred',
                'review',
                'merge-deferred',
            }
        )
        assert orch_set == fm_canonical, (
            "ACTIVE_TASK_STATUSES drift detected!\n"
            f"  orchestrator/task_status.py: {sorted(orch_set)}\n"
            f"  fused_memory/reconciliation/task_filter.py (canonical): {sorted(fm_canonical)}\n"
            "Update orchestrator/task_status.py to match the server-side definition."
        )


# ---------------------------------------------------------------------------
# Step-5 RED: Scheduler.carries_substrate_probe
# ---------------------------------------------------------------------------


class TestCarriesSubstrateProbe:
    """``Scheduler.carries_substrate_probe(task)`` single-source-of-truth.

    The staticmethod is a thin delegate to
    ``substrate_gate.extract_probe_set(task) is not None`` so that the harness
    uses a single entry-point to decide whether to run the gate.
    """

    # --- True cases ---

    def test_returns_true_for_probe_task_dict_metadata(self):
        task = {
            'id': '1',
            'status': 'pending',
            'metadata': {
                'substrate_probe': {
                    'probe_set': 'probes/suite.json',
                    'checker': ['python', '-m', 'checker'],
                }
            },
        }
        assert Scheduler.carries_substrate_probe(task) is True

    def test_returns_true_for_probe_task_json_string_metadata(self):
        import json as _json
        meta = _json.dumps({
            'substrate_probe': {
                'probe_set': 'probes/suite.json',
                'checker': ['run_check'],
            }
        })
        task = {'id': '1', 'status': 'pending', 'metadata': meta}
        assert Scheduler.carries_substrate_probe(task) is True

    # --- False cases ---

    def test_returns_false_for_plain_task(self):
        task = {'id': '1', 'status': 'pending', 'metadata': {'files': ['src/foo.py']}}
        assert Scheduler.carries_substrate_probe(task) is False

    def test_returns_false_when_metadata_is_empty_dict(self):
        task = {'id': '1', 'metadata': {}}
        assert Scheduler.carries_substrate_probe(task) is False

    def test_returns_false_when_metadata_absent(self):
        task = {'id': '1', 'status': 'pending'}
        assert Scheduler.carries_substrate_probe(task) is False

    def test_returns_false_when_metadata_is_none(self):
        task = {'id': '1', 'metadata': None}
        assert Scheduler.carries_substrate_probe(task) is False

    def test_returns_false_for_json_string_without_substrate_probe(self):
        import json as _json
        task = {'id': '1', 'metadata': _json.dumps({'other_key': 'value'})}
        assert Scheduler.carries_substrate_probe(task) is False

    def test_returns_false_when_descriptor_has_no_probe_set(self):
        task = {
            'id': '1',
            'metadata': {
                'substrate_probe': {'checker': ['run_check']}  # no probe_set
            },
        }
        assert Scheduler.carries_substrate_probe(task) is False

    def test_returns_false_when_substrate_probe_is_not_dict(self):
        task = {'id': '1', 'metadata': {'substrate_probe': 'bad-value'}}
        assert Scheduler.carries_substrate_probe(task) is False

    # --- Regression guard: acquire_next is untouched ---

    @pytest.mark.asyncio
    async def test_acquire_next_still_dispatches_probe_carrying_task(self):
        """Substrate gate lives in harness._run_slot, not acquire_next.

        A probe-carrying pending task must be returned by acquire_next()
        exactly as a plain task would be — the scoring/locking hot loop
        must be unaffected.
        """
        probe_task = {
            'id': '42',
            'title': 'Probe task',
            'status': 'pending',
            'dependencies': [],
            'metadata': {
                'files': ['src/foo.py'],
                'substrate_probe': {
                    'probe_set': 'probes/suite.json',
                    'checker': ['python', '-m', 'checker'],
                },
            },
        }
        config = OrchestratorConfig(max_per_module=1)
        scheduler = Scheduler(config)
        scheduler.get_tasks = AsyncMock(return_value=[probe_task])

        assignment = await scheduler.acquire_next()
        assert assignment is not None
        assert assignment.task_id == '42'
        # The task dict is unchanged — acquire_next does not remove substrate_probe
        assert assignment.task['metadata']['substrate_probe']['probe_set'] == 'probes/suite.json'


# ---------------------------------------------------------------------------
# TestExternalDepResolverDegradedEscalation (task 1855 — step-1 RED / step-2 GREEN)
# ---------------------------------------------------------------------------

class TestExternalDepResolverDegradedEscalation:
    """Persistent resolver_degraded holds must escalate to a human after threshold ticks.

    When external_err is not None for N consecutive ticks (where N >=
    max_external_dep_unresolved_cycles), the scheduler must invoke
    _on_external_dep_block with:
    - summary containing the 'EXTERNAL_DEP_RESOLVER_DEGRADED' prefix
    - category='infra_issue'

    Below threshold → no callback. Escalation is guarded behind a non-None
    callback (callback=None → visibility-only, no exception, existing tests
    still green).

    This is RED today because the degraded branch only calls _note_external_hold
    and returns — it never touches _on_external_dep_block.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(
            max_per_module=1,
            max_external_dep_unresolved_cycles=2,
        )
        return Scheduler(config)

    def _pending_task(self) -> dict:
        return {
            'id': '10',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'external_deps': ['dark_factory:5']},
        }

    @pytest.mark.asyncio
    async def test_below_threshold_no_escalation(self, scheduler: Scheduler):
        """1 degraded tick (< threshold=2) must NOT invoke _on_external_dep_block."""
        callback = AsyncMock()
        scheduler._on_external_dep_block = callback
        task = self._pending_task()

        await scheduler._apply_external_dep_policy(
            [task],
            {},
            ExternalResolverError('degraded'),
        )

        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_at_threshold_escalates(self, scheduler: Scheduler):
        """2 consecutive degraded ticks (== threshold=2) must invoke callback exactly once.

        The callback must be called with:
        - summary containing 'EXTERNAL_DEP_RESOLVER_DEGRADED'
        - category='infra_issue'
        """
        callback = AsyncMock()
        scheduler._on_external_dep_block = callback
        task = self._pending_task()

        # Tick 1 — below threshold
        await scheduler._apply_external_dep_policy(
            [task],
            {},
            ExternalResolverError('degraded'),
        )
        callback.assert_not_called()

        # Tick 2 — reaches threshold → must escalate
        await scheduler._apply_external_dep_policy(
            [task],
            {},
            ExternalResolverError('degraded'),
        )

        callback.assert_called_once()
        call_kwargs = callback.call_args
        args = call_kwargs.args if call_kwargs.args else ()
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}

        # task_id must be '10'
        task_id_arg = args[0] if args else kwargs.get('task_id')
        assert str(task_id_arg) == '10', (
            f'Expected task_id="10"; got {task_id_arg!r}'
        )

        # summary must carry EXTERNAL_DEP_RESOLVER_DEGRADED prefix
        summary_arg = kwargs.get('summary', '') or (args[1] if len(args) > 1 else '')
        assert 'EXTERNAL_DEP_RESOLVER_DEGRADED' in str(summary_arg), (
            f'Expected EXTERNAL_DEP_RESOLVER_DEGRADED in summary; got {summary_arg!r}'
        )

        # category must be 'infra_issue'
        category_arg = kwargs.get('category')
        assert category_arg == 'infra_issue', (
            f'Expected category="infra_issue"; got {category_arg!r}'
        )

    @pytest.mark.asyncio
    async def test_counter_resets_on_clean_tick(self, scheduler: Scheduler):
        """A non-degraded tick between degraded ticks must reset the consecutive streak.

        Drive: degraded tick 1, clean tick 2 (resolver OK), degraded tick 3.
        Because the clean tick resets the streak, only 1 consecutive degraded tick
        occurs after the reset — threshold=2 is never reached — so the callback
        must NOT be called.
        """
        callback = AsyncMock()
        scheduler._on_external_dep_block = callback
        task = self._pending_task()

        # Tick 1 — degraded (streak=1, below threshold=2)
        await scheduler._apply_external_dep_policy(
            [task],
            {},
            ExternalResolverError('degraded'),
        )
        callback.assert_not_called()

        # Tick 2 — clean (resolver OK, dep live) → must reset the streak
        await scheduler._apply_external_dep_policy(
            [task],
            {'dark_factory:5': 'pending'},
            external_err=None,
        )
        callback.assert_not_called()

        # Tick 3 — degraded again (streak restarts at 1, below threshold=2)
        await scheduler._apply_external_dep_policy(
            [task],
            {},
            ExternalResolverError('degraded'),
        )

        # The clean tick reset the streak, so threshold was never reached
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_callback_none_falls_through_to_visibility_hold(
        self, scheduler: Scheduler
    ):
        """callback=None at threshold: logs warning, no exception, falls through to _note_external_hold.

        When _on_external_dep_block is None (the default) and count reaches
        threshold, the implementation must NOT raise.  It must fall through to
        _note_external_hold so the existing visibility-only (gate_held event)
        path continues to fire — preserving the behaviour of the three existing
        TestExternalDepGateHeld_ResolverDegraded tests.
        """
        # Leave _on_external_dep_block as None (default).
        assert scheduler._on_external_dep_block is None
        task = self._pending_task()

        # Wire an event_store to verify the gate_held event is still emitted.
        scheduler.event_store = _RecordingEventStore()  # type: ignore[assignment]

        # Drive threshold ticks with callback=None — must not raise.
        threshold = scheduler.config.max_external_dep_unresolved_cycles
        for _ in range(threshold):
            await scheduler._apply_external_dep_policy(
                [task],
                {},
                ExternalResolverError('persistent'),
            )

        # The gate_held event must have been emitted (visibility still works).
        gate_held = [
            (evt, data)
            for evt, data in scheduler.event_store.events  # type: ignore[union-attr]
            if evt == str(EventType.external_dep_gate_held)
        ]
        assert len(gate_held) == 1, (
            f'Expected 1 gate_held event from callback=None fall-through; got {gate_held!r}'
        )
        assert gate_held[0][1]['data']['cause'] == 'resolver_degraded'

    @pytest.mark.asyncio
    async def test_counter_pops_on_escalation_so_persistent_outage_refiles(
        self, scheduler: Scheduler
    ):
        """After escalation the counter is popped; next threshold crossing fires again.

        A persistent (never-recovering) resolver outage must re-fire every
        threshold ticks so the human is reminded if the first escalation is
        dismissed without fixing the root cause.
        """
        callback = AsyncMock()
        scheduler._on_external_dep_block = callback
        task = self._pending_task()

        # First crossing (ticks 1-2 with threshold=2) → escalates once.
        for _ in range(2):
            await scheduler._apply_external_dep_policy(
                [task],
                {},
                ExternalResolverError('persistent'),
            )
        callback.assert_called_once()
        callback.reset_mock()

        # Second crossing (ticks 3-4, counter was popped so restarts at 0) → fires again.
        for _ in range(2):
            await scheduler._apply_external_dep_policy(
                [task],
                {},
                ExternalResolverError('persistent'),
            )
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_hold_streak_cleared_on_escalation(self, scheduler: Scheduler):
        """Escalation must pop _external_hold_streak / _external_hold_cause.

        After the threshold is reached and _on_external_dep_block is awaited,
        the hold streak and cause entries for the task must be absent so that no
        spurious external_dep_gate_held event fires for the now-blocked task.
        """
        callback = AsyncMock()
        scheduler._on_external_dep_block = callback
        task = self._pending_task()

        # Pre-populate a hold streak to make the assertion meaningful.
        task_id = str(task['id'])
        scheduler._external_hold_streak[task_id] = 5
        scheduler._external_hold_cause[task_id] = 'resolver_degraded'

        # Drive to threshold → escalation fires.
        threshold = scheduler.config.max_external_dep_unresolved_cycles
        for _ in range(threshold):
            await scheduler._apply_external_dep_policy(
                [task],
                {},
                ExternalResolverError('persistent'),
            )

        callback.assert_called_once()
        assert task_id not in scheduler._external_hold_streak, (
            f'_external_hold_streak must be cleared after escalation; '
            f'got {scheduler._external_hold_streak!r}'
        )
        assert task_id not in scheduler._external_hold_cause, (
            f'_external_hold_cause must be cleared after escalation; '
            f'got {scheduler._external_hold_cause!r}'
        )

    @pytest.mark.asyncio
    async def test_multiple_tasks_escalate_in_one_tick(self, scheduler: Scheduler):
        """A resolver outage affects all pending tasks with external_deps simultaneously.

        When external_err is set for a tick that crosses the threshold for N
        tasks, all N tasks must be escalated in that same tick — one callback
        call per task.
        """
        callback = AsyncMock()
        scheduler._on_external_dep_block = callback

        task_a = {
            'id': 'A',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'external_deps': ['proj:1']},
        }
        task_b = {
            'id': 'B',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'external_deps': ['proj:2']},
        }
        tasks = [task_a, task_b]

        # Drive both tasks to threshold (2 consecutive degraded ticks).
        for _ in range(2):
            await scheduler._apply_external_dep_policy(
                tasks,
                {},
                ExternalResolverError('outage'),
            )

        # Both tasks must have been escalated.
        assert callback.call_count == 2, (
            f'Expected callback called once per task (2 total); got {callback.call_count}'
        )
        escalated_ids = {call.args[0] for call in callback.call_args_list}
        assert escalated_ids == {'A', 'B'}, (
            f'Expected both task ids escalated; got {escalated_ids!r}'
        )


# ---- Park-eviction drain helper (task 1871 step-5 B3+B5) ----

class TestDrainParkEvictionRequests:
    """Unit tests for Scheduler._drain_park_eviction_requests (B3 + B5).

    Tests drive the helper DIRECTLY with constructed status_map/tasks_by_id
    to avoid full-tick dispatch confounds.  Mirrors test_park_gc_on_terminal_owner
    scaffold.
    """

    def _make_scheduler(self, tmp_path, event_store=None):
        from orchestrator.park_eviction_requests import ParkEvictionRequestStore
        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        eviction_store = ParkEvictionRequestStore(tmp_path / 'park_eviction_requests.db')
        if event_store is None:
            event_store = _RecordingEventStore()
        scheduler = Scheduler(
            config,
            event_store=event_store,  # type: ignore[arg-type]
            park_eviction_store=eviction_store,
        )
        return scheduler, eviction_store, event_store

    def test_b3_dead_owner_evicted_with_events_and_row_drained(self, tmp_path):
        """B3: dead owner T (terminal/missing) — force_clear fires, events emitted."""
        scheduler, store, event_store = self._make_scheduler(tmp_path)

        # Stack on m1: [L(buried/low), T(active-top/high)]
        scheduler.lock_table.install_parks('L', ['m1'], priority='low')
        scheduler.lock_table.install_parks('T', ['m1'], priority='high')
        scheduler._skip_count['T'] = 3

        # Enqueue eviction request for T.
        store.enqueue('T', scheduler._project_root)

        # T is terminal (cancelled), L is pending and present.
        status_map = {'T': 'cancelled', 'L': 'pending'}
        tasks_by_id = {
            'L': {
                'id': 'L', 'status': 'pending', 'dependencies': [],
                'priority': 'low', 'metadata': {'files': ['m1']},
            },
        }

        scheduler._drain_park_eviction_requests(status_map, tasks_by_id)

        # T's parks gone.
        assert not scheduler.lock_table.has_parks('T')
        # skip_count cleared.
        assert 'T' not in scheduler._skip_count
        # reservation_force_evicted emitted for T.
        evicted_events = [
            e for e in event_store.events
            if 'reservation_force_evicted' in e[0]
        ]
        assert len(evicted_events) == 1
        assert evicted_events[0][1]['task_id'] == 'T'
        assert evicted_events[0][1]['data']['owner'] == 'T'
        assert 'm1' in evicted_events[0][1]['data']['modules']
        # reservation_restored emitted for newly-exposed L.
        restored_events = [
            e for e in event_store.events
            if 'reservation_restored' in e[0]
        ]
        assert len(restored_events) == 1
        assert restored_events[0][1]['task_id'] == 'L'
        # Row drained — second drain is empty.
        assert store.drain(scheduler._project_root) == []

    def test_b5_no_parks_owner_noop_row_drained(self, tmp_path):
        """B5: owner with no parks — no reservation_force_evicted, no exception, row drained."""
        scheduler, store, event_store = self._make_scheduler(tmp_path)

        # Enqueue for an owner with NO parks.
        store.enqueue('ghost', scheduler._project_root)

        status_map = {}
        tasks_by_id = {}

        scheduler._drain_park_eviction_requests(status_map, tasks_by_id)

        # No eviction event.
        evicted_events = [
            e for e in event_store.events
            if 'reservation_force_evicted' in e[0]
        ]
        assert evicted_events == []
        # Row consumed.
        assert store.drain(scheduler._project_root) == []


# ---- B4 D4 live-owner guard (task 1871 step-7) ----

class TestDrainParkEvictionGuard:
    """B4 — the load-bearing safety test: live owners are REFUSED eviction.

    Tests drive _drain_park_eviction_requests directly.
    B4a: live owner T (pending + in tasks_by_id + deps satisfied) → REFUSAL.
    B4b: pending task with an UNSATISFIED dep → NOT live-dispatchable → ALLOWED.
    """

    def _make_scheduler(self, tmp_path, event_store=None):
        from orchestrator.park_eviction_requests import ParkEvictionRequestStore
        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        eviction_store = ParkEvictionRequestStore(tmp_path / 'park_eviction_requests.db')
        if event_store is None:
            event_store = _RecordingEventStore()
        scheduler = Scheduler(
            config,
            event_store=event_store,  # type: ignore[arg-type]
            park_eviction_store=eviction_store,
        )
        return scheduler, eviction_store, event_store

    def test_b4a_live_owner_eviction_refused(self, tmp_path):
        """B4a: T is pending + in tasks_by_id + no deps → live → force_clear REFUSED."""
        scheduler, store, event_store = self._make_scheduler(tmp_path)

        scheduler.lock_table.install_parks('T', ['m1'], priority='high')
        store.enqueue('T', scheduler._project_root)

        # T is pending and live-dispatchable (no dependencies).
        status_map = {'T': 'pending'}
        tasks_by_id = {
            'T': {
                'id': 'T', 'status': 'pending', 'dependencies': [],
                'priority': 'high', 'metadata': {'files': ['m1']},
            },
        }

        scheduler._drain_park_eviction_requests(status_map, tasks_by_id)

        # T's park must be INTACT.
        assert scheduler.lock_table.has_parks('T'), (
            'force_clear must be REFUSED for a live dispatchable owner'
        )
        # No reservation_force_evicted event.
        evicted_events = [
            e for e in event_store.events
            if 'reservation_force_evicted' in e[0]
        ]
        assert evicted_events == [], (
            f'reservation_force_evicted must NOT be emitted for live owner; got {evicted_events}'
        )
        # Exactly one reservation_force_evict_refused event for T.
        refused_events = [
            e for e in event_store.events
            if 'reservation_force_evict_refused' in e[0]
        ]
        assert len(refused_events) == 1
        assert refused_events[0][1]['task_id'] == 'T'
        assert refused_events[0][1]['data']['reason'] == 'live_owner'
        # Row was still drained (one-shot, refuse does not retry).
        assert store.drain(scheduler._project_root) == []

    def test_b4b_unsatisfied_deps_not_live_eviction_allowed(self, tmp_path):
        """B4b: pending task with an unsatisfied dep → NOT live → force_clear fires."""
        scheduler, store, event_store = self._make_scheduler(tmp_path)

        scheduler.lock_table.install_parks('T', ['m1'], priority='high')
        store.enqueue('T', scheduler._project_root)

        # T depends on dep '99' which is in-progress (not done → unsatisfied).
        status_map = {'T': 'pending', '99': 'in-progress'}
        tasks_by_id = {
            'T': {
                'id': 'T', 'status': 'pending',
                'dependencies': ['99'],
                'priority': 'high', 'metadata': {'files': ['m1']},
            },
        }

        scheduler._drain_park_eviction_requests(status_map, tasks_by_id)

        # T's park must be gone — unsatisfied deps → not live-dispatchable → evict.
        assert not scheduler.lock_table.has_parks('T'), (
            'force_clear must FIRE for an owner with unsatisfied deps'
        )
        evicted_events = [
            e for e in event_store.events
            if 'reservation_force_evicted' in e[0]
        ]
        assert len(evicted_events) == 1
        assert evicted_events[0][1]['task_id'] == 'T'


# ---- acquire_next wiring test (task 1871 step-9) ----

class TestDrainCalledFromAcquireNext:
    """Proves the tick drains the park-eviction table and runs BEFORE park-GC.

    A dead owner T is present in both the park-eviction store AND would be
    reaped by park-GC's reservation_expired sweep.  After acquire_next():
    - the request row is gone
    - T's park is gone
    - a reservation_force_evicted event was emitted (NOT reservation_expired)
      proving the drain ran first.
    """

    @pytest.mark.asyncio
    async def test_acquire_next_drains_eviction_store_before_park_gc(self, tmp_path):
        from orchestrator.park_eviction_requests import ParkEvictionRequestStore

        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        event_store = _RecordingEventStore()
        eviction_store = ParkEvictionRequestStore(tmp_path / 'park_eviction_requests.db')

        scheduler = Scheduler(
            config,
            event_store=event_store,  # type: ignore[arg-type]
            park_eviction_store=eviction_store,
        )

        # Stack on m1: [L(buried/low), T(active-top/high)]
        scheduler.lock_table.install_parks('L', ['m1'], priority='low')
        scheduler.lock_table.install_parks('T', ['m1'], priority='high')

        # Enqueue eviction for T (dead owner).
        eviction_store.enqueue('T', scheduler._project_root)

        # T is terminal; L is pending and live (no deps).
        t_task = {
            'id': 'T', 'title': 'T', 'status': 'cancelled',
            'priority': 'high', 'dependencies': [],
            'metadata': {'files': ['m1']},
        }
        l_task = {
            'id': 'L', 'title': 'L', 'status': 'pending',
            'priority': 'low', 'dependencies': [],
            'metadata': {'files': ['m1']},
        }
        scheduler.get_tasks = AsyncMock(return_value=[t_task, l_task])

        await scheduler.acquire_next()

        # Eviction store row must be drained.
        assert eviction_store.drain(scheduler._project_root) == []
        # T's park is gone.
        assert not scheduler.lock_table.has_parks('T')
        # reservation_force_evicted emitted for T (operator audit wins).
        force_evicted = [
            e for e in event_store.events
            if 'reservation_force_evicted' in e[0]
        ]
        assert len(force_evicted) == 1, (
            f'Expected 1 reservation_force_evicted; got {force_evicted}'
        )
        assert force_evicted[0][1]['task_id'] == 'T'
        # reservation_expired must NOT be emitted for T
        # (park-GC finds nothing left to reap after the drain).
        expired_for_t = [
            e for e in event_store.events
            if 'reservation_expired' in e[0]
            and e[1].get('task_id') == 'T'
        ]
        assert expired_for_t == [], (
            f'reservation_expired must not fire for T after drain took it; got {expired_for_t}'
        )


# ---- Drain fires even when get_tasks returns empty (task 1871 amend-1) ----

class TestDrainFiresWithNoActiveTasks:
    """Prove that _drain_park_eviction_requests runs even when tasks=[] causes
    the early-return path in acquire_next.

    Regression guard for the scenario an operator might hit: all live tasks
    finished but a stranded/buried park remains and the operator enqueues an
    eviction — without this fix the request sits in the table indefinitely.
    """

    @pytest.mark.asyncio
    async def test_drain_processes_request_when_tasks_empty(self, tmp_path):
        """With get_tasks returning [], enqueued evictions are still processed."""
        from orchestrator.park_eviction_requests import ParkEvictionRequestStore

        config = OrchestratorConfig(max_per_module=1, lock_depth=2)
        event_store = _RecordingEventStore()
        eviction_store = ParkEvictionRequestStore(tmp_path / 'park_eviction_requests.db')

        scheduler = Scheduler(
            config,
            event_store=event_store,  # type: ignore[arg-type]
            park_eviction_store=eviction_store,
        )

        # Install a park for a dead owner.
        scheduler.lock_table.install_parks('dead_owner', ['m1'], priority='low')
        eviction_store.enqueue('dead_owner', scheduler._project_root)

        # Simulate a world where all tasks are done/gone.
        scheduler.get_tasks = AsyncMock(return_value=[])

        result = await scheduler.acquire_next()

        # acquire_next still returns None (no tasks to dispatch).
        assert result is None
        # The eviction request was consumed.
        assert eviction_store.drain(scheduler._project_root) == []
        # dead_owner's park was cleared (empty status_map/tasks_by_id →
        # not live-dispatchable → force_clear fired).
        assert not scheduler.lock_table.has_parks('dead_owner')
        # reservation_force_evicted event was emitted.
        force_evicted = [
            e for e in event_store.events
            if 'reservation_force_evicted' in e[0]
        ]
        assert len(force_evicted) == 1
        assert force_evicted[0][1]['task_id'] == 'dead_owner'


# ---- Buried-owner restored-event at drain level (task 1871 amend-4) ----

class TestDrainBuriedOwnerRestoredEvents:
    """Verify that evicting a buried (non-top) owner exposes the new top and
    emits reservation_restored for all newly-exposed shadows at the drain level.

    The force_clear + restored-event path is exercised end-to-end through
    _drain_park_eviction_requests with a multi-layer stack to ensure the
    restored pairs are correctly mapped to events.
    """

    def _make_scheduler(self, tmp_path, event_store=None):
        from orchestrator.park_eviction_requests import ParkEvictionRequestStore
        config = OrchestratorConfig(max_per_module=1, lock_depth=3)
        eviction_store = ParkEvictionRequestStore(tmp_path / 'park_eviction_requests.db')
        if event_store is None:
            event_store = _RecordingEventStore()
        scheduler = Scheduler(
            config,
            event_store=event_store,  # type: ignore[arg-type]
            park_eviction_store=eviction_store,
        )
        return scheduler, eviction_store, event_store

    def test_evicting_non_top_owner_emits_restored_for_exposed_shadow(self, tmp_path):
        """Evict an owner buried beneath a higher-priority top; the new
        exposed shadow emits reservation_restored, and the active top is
        unaffected (its park remains)."""
        scheduler, store, event_store = self._make_scheduler(tmp_path)

        # Stack on m1: [L(bottom/low), B(buried/mid), T(active-top/high)]
        # force_clear('B') removes B; T stays on top; L is now second.
        # restore: L gets a new opportunity as T still blocks — but in LIFO
        # terms, prune_owners only restores when a shadow becomes the new top.
        # Here T > B > L means removing B exposes L only if T's rank was above B.
        # Build a two-module scenario so B is the top on one module.

        # Module m2: stack [L2(low), B(mid)] — B is the active top here.
        scheduler.lock_table.install_parks('L2', ['m2'], priority='low')
        scheduler.lock_table.install_parks('B', ['m2'], priority='mid')

        # Enqueue evict for B (which is active-top on m2).
        store.enqueue('B', scheduler._project_root)

        # B is missing from tasks_by_id → not live-dispatchable → evict.
        status_map = {'L2': 'pending'}
        tasks_by_id = {
            'L2': {
                'id': 'L2', 'status': 'pending', 'dependencies': [],
                'priority': 'low', 'metadata': {'files': ['m2']},
            },
        }

        scheduler._drain_park_eviction_requests(status_map, tasks_by_id)

        # B's park on m2 is gone.
        assert not scheduler.lock_table.has_parks('B')
        # L2's park is still present (it was not evicted).
        assert scheduler.lock_table.has_parks('L2')
        # reservation_force_evicted emitted for B (had parks on m2).
        evicted = [
            e for e in event_store.events
            if 'reservation_force_evicted' in e[0]
        ]
        assert len(evicted) == 1
        assert evicted[0][1]['task_id'] == 'B'
        assert 'm2' in evicted[0][1]['data']['modules']
        # reservation_restored emitted for L2 (newly exposed as top on m2).
        restored = [
            e for e in event_store.events
            if 'reservation_restored' in e[0]
        ]
        assert len(restored) == 1
        assert restored[0][1]['task_id'] == 'L2'
        assert 'm2' in restored[0][1]['data']['modules']


# ---- Starvation watchdog (task 1880) ----


class TestStarvationWatchdog:
    """Tests for Scheduler._apply_starvation_watchdog and dispatch-site resolve.

    The watchdog fires an INFO escalation when a dispatch-eligible task
    keeps being skipped as the TOP-scored candidate past BOTH skip_threshold
    AND idle_secs.  It self-resolves when the task dispatches.
    """

    def _make_scheduler(
        self,
        *,
        skip_threshold: int = 3,
        idle_secs: float = 100.0,
        enabled: bool = True,
    ) -> tuple['Scheduler', list]:
        """Build a Scheduler with watchdog config tuned small and a mutable clock."""
        t: list[float] = [0.0]

        def fake_clock() -> float:
            return t[0]

        config = OrchestratorConfig(
            max_per_module=1,
            starvation_watchdog=StarvationWatchdogConfig(
                enabled=enabled,
                skip_threshold=skip_threshold,
                idle_secs=idle_secs,
            ),
        )
        # Prevent fairness parks from installing before the watchdog threshold
        # is reached (the watchdog skip_threshold=3 is deliberately smaller
        # than the test's default fairness skip_threshold per tier so we can
        # isolate the watchdog logic cleanly).
        config.fairness.skip_threshold = 9999
        scheduler = Scheduler(config, time_source=fake_clock)
        return scheduler, t

    @staticmethod
    def _starved_task(tid: str = 'starved') -> dict:
        return {
            'id': tid,
            'title': f'Starved task {tid}',
            'status': 'pending',
            'priority': 'medium',
            'dependencies': [],
            'metadata': {'files': ['backend']},
        }

    @pytest.mark.asyncio
    async def test_positive_fires_once_after_both_thresholds(self):
        """Watchdog fires exactly once when skip_count >= threshold AND clock >= idle_secs.

        Positive firing path:
        - Drive 3 acquire_next ticks (skip_count bumps to 3 via loop-exhausted path).
        - Advance clock to 150s (>= idle_secs=100).
        - One more tick → _on_starvation_warn called exactly once, with task_id='starved'
          and summary containing the stable 'STARVATION_WATCHDOG' marker.
        - Additional ticks do NOT re-fire (dedup / no re-arm spam).
        - 'starved' is in scheduler._starvation_escalated.
        """
        scheduler, t = self._make_scheduler()
        callback = AsyncMock()
        scheduler._on_starvation_warn = callback

        # Seed a held lock on a dispatched 'seed' task so 'starved' can never
        # acquire its module (loop-exhausted path → skip_count bumped each tick).
        scheduler.lock_table.try_acquire('seed', ['backend'])
        scheduler._dispatched.add('seed')

        task = self._starved_task()
        scheduler.get_tasks = AsyncMock(return_value=[task])

        # Drive 3 ticks to accumulate skip_count >= skip_threshold (3).
        # Clock stays at 0 during these ticks.
        for _ in range(3):
            result = await scheduler.acquire_next()
            assert result is None, 'Expected no dispatch while seed holds the lock'

        assert scheduler._skip_count.get('starved', 0) >= 3, (
            f'Expected skip_count >= 3 after 3 loop-exhausted ticks; '
            f'got {scheduler._skip_count!r}'
        )

        # Advance clock past idle_secs (100s) and tick once more.
        t[0] = 150.0
        await scheduler.acquire_next()

        # Both thresholds are now crossed → callback must fire exactly once.
        callback.assert_called_once()
        call_args = callback.call_args
        args = call_args.args if call_args.args else ()
        kwargs = call_args.kwargs if call_args.kwargs else {}

        # task_id must be 'starved'.
        task_id_arg = args[0] if args else kwargs.get('task_id')
        assert str(task_id_arg) == 'starved', (
            f'Expected task_id="starved"; got {task_id_arg!r}'
        )

        # summary must carry the stable STARVATION_WATCHDOG marker.
        summary_arg = kwargs.get('summary', '') or (args[1] if len(args) > 1 else '')
        assert 'STARVATION_WATCHDOG' in str(summary_arg), (
            f'Expected STARVATION_WATCHDOG in summary; got {summary_arg!r}'
        )

        # Further ticks must NOT re-fire the callback (dedup guard).
        for _ in range(5):
            await scheduler.acquire_next()

        callback.assert_called_once()  # still exactly once

        # 'starved' must be in _starvation_escalated after the first fire.
        assert 'starved' in scheduler._starvation_escalated, (
            '_starvation_escalated must contain task_id after the watchdog fires'
        )

    @pytest.mark.asyncio
    async def test_self_resolve_on_dispatch(self):
        """_on_starvation_resolve is called exactly once when the escalated task dispatches.

        Drive 'starved' to the escalated state (task_id in _starvation_escalated).
        Then release the 'seed' lock so the next acquire_next dispatches 'starved'.
        Assert:
        - acquire_next returns a TaskAssignment for 'starved'
        - _on_starvation_resolve called exactly once with task_id='starved'
        - 'starved' NOT in scheduler._starvation_escalated
        - 'starved' NOT in scheduler._starvation_first_seen
        """
        scheduler, t = self._make_scheduler()
        warn_cb = AsyncMock()
        resolve_cb = AsyncMock()
        scheduler._on_starvation_warn = warn_cb
        scheduler._on_starvation_resolve = resolve_cb

        # Seed a held lock on 'seed' so 'starved' can't acquire initially.
        scheduler.lock_table.try_acquire('seed', ['backend'])
        scheduler._dispatched.add('seed')

        task = self._starved_task()
        scheduler.get_tasks = AsyncMock(return_value=[task])

        # Drive 3 ticks to hit skip_threshold, then advance clock past idle_secs.
        for _ in range(3):
            await scheduler.acquire_next()

        t[0] = 150.0
        await scheduler.acquire_next()

        # Verify warn callback fired (task is now in _starvation_escalated).
        warn_cb.assert_called_once()
        assert 'starved' in scheduler._starvation_escalated, (
            'Precondition: starved must be in _starvation_escalated'
        )
        resolve_cb.assert_not_called()

        # Release 'seed' so 'starved' can acquire 'backend'.
        scheduler.lock_table.release('seed')
        scheduler._dispatched.discard('seed')

        # Next acquire_next must dispatch 'starved'.
        result = await scheduler.acquire_next()
        assert result is not None, 'Expected TaskAssignment after releasing seed lock'
        assert result.task_id == 'starved', (
            f'Expected task_id="starved"; got {result.task_id!r}'
        )

        # Resolve callback must have fired exactly once with 'starved'.
        resolve_cb.assert_called_once()
        call_args = resolve_cb.call_args
        args = call_args.args if call_args.args else ()
        kwargs = call_args.kwargs if call_args.kwargs else {}
        task_id_arg = args[0] if args else kwargs.get('task_id')
        assert str(task_id_arg) == 'starved', (
            f'Expected task_id="starved" in resolve callback; got {task_id_arg!r}'
        )

        # State cleanup: 'starved' must no longer be in the escalated set or seen dict.
        assert 'starved' not in scheduler._starvation_escalated, (
            '_starvation_escalated must be cleared on dispatch'
        )
        assert 'starved' not in scheduler._starvation_first_seen, (
            '_starvation_first_seen must be cleared on dispatch'
        )

    @pytest.mark.asyncio
    async def test_below_skip_threshold_no_callback(self):
        """(a) Below skip_threshold: fewer skips than threshold → callback NOT called.

        skip_threshold=3, only 2 ticks (skip_count=2 < 3).  Clock advanced well
        past idle_secs (100s) to prove the skip-count gate blocks firing.
        """
        scheduler, t = self._make_scheduler()
        callback = AsyncMock()
        scheduler._on_starvation_warn = callback

        scheduler.lock_table.try_acquire('seed', ['backend'])
        scheduler._dispatched.add('seed')
        task = self._starved_task()
        scheduler.get_tasks = AsyncMock(return_value=[task])

        # 2 ticks → skip_count = 2 < threshold=3.
        for _ in range(2):
            await scheduler.acquire_next()

        # Advance clock well past idle_secs.
        t[0] = 500.0
        await scheduler.acquire_next()

        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_idle_gate_below_idle_secs_no_callback(self):
        """(b) Idle gate: skip_count >= threshold but clock < idle_secs → NOT called.

        Proves BOTH conditions are required — the skip-gate alone is not sufficient.
        Clock advances only to 50s < idle_secs=100s.
        """
        scheduler, t = self._make_scheduler()
        callback = AsyncMock()
        scheduler._on_starvation_warn = callback

        scheduler.lock_table.try_acquire('seed', ['backend'])
        scheduler._dispatched.add('seed')
        task = self._starved_task()
        scheduler.get_tasks = AsyncMock(return_value=[task])

        # Drive 3+ ticks to cross skip_threshold.
        for _ in range(4):
            await scheduler.acquire_next()

        assert scheduler._skip_count.get('starved', 0) >= 3, (
            'Precondition: skip_count must be >= 3 for this test to be meaningful'
        )

        # Clock below idle_secs (50s < 100s) — callback must NOT fire.
        t[0] = 50.0
        await scheduler.acquire_next()

        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_disabled_no_callback(self):
        """(c) Disabled: starvation_watchdog.enabled=False → NOT called even when both thresholds crossed.

        Set enabled=False, drive skip_count >= threshold, advance clock past
        idle_secs → callback must never fire.
        """
        scheduler, t = self._make_scheduler(enabled=False)
        callback = AsyncMock()
        scheduler._on_starvation_warn = callback

        scheduler.lock_table.try_acquire('seed', ['backend'])
        scheduler._dispatched.add('seed')
        task = self._starved_task()
        scheduler.get_tasks = AsyncMock(return_value=[task])

        # Drive 5 ticks to clearly cross skip_threshold=3.
        for _ in range(5):
            await scheduler.acquire_next()

        # Advance clock well past idle_secs=100s.
        t[0] = 500.0
        await scheduler.acquire_next()

        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_gc_terminal_task_resolves_escalation(self):
        """(a) GC backstop: escalated task resolved + state cleared when task goes terminal.

        Drive 'starved' to the escalated state (task_id in _starvation_escalated).
        Then make get_tasks report it as 'done' (terminal).  One more
        acquire_next tick must:
        - call _on_starvation_resolve exactly once with task_id='starved',
        - clear 'starved' from _starvation_escalated, AND
        - clear 'starved' from _starvation_first_seen.

        Fails until the dedicated watchdog GC block is added to the stale-id
        sweep inside acquire_next (step-10).
        """
        scheduler, t = self._make_scheduler()
        warn_cb = AsyncMock()
        resolve_cb = AsyncMock()
        scheduler._on_starvation_warn = warn_cb
        scheduler._on_starvation_resolve = resolve_cb

        # Seed a held lock on 'seed' so 'starved' can never acquire.
        scheduler.lock_table.try_acquire('seed', ['backend'])
        scheduler._dispatched.add('seed')

        task = self._starved_task()
        scheduler.get_tasks = AsyncMock(return_value=[task])

        # Drive 3 ticks to cross skip_threshold (3), then advance clock past idle_secs.
        for _ in range(3):
            await scheduler.acquire_next()
        t[0] = 150.0
        await scheduler.acquire_next()

        # Pre-condition: warn fired, task is escalated.
        warn_cb.assert_called_once()
        assert 'starved' in scheduler._starvation_escalated, (
            'Precondition: starved must be in _starvation_escalated before GC tick'
        )
        resolve_cb.assert_not_called()

        # Change get_tasks to return 'starved' as terminal ('done').
        terminal_task = dict(task, status='done')
        scheduler.get_tasks = AsyncMock(return_value=[terminal_task])

        # One tick — the GC backstop must detect 'starved' is terminal,
        # call _on_starvation_resolve, and clear all watchdog state.
        await scheduler.acquire_next()

        resolve_cb.assert_called_once()
        call_args = resolve_cb.call_args
        args = call_args.args if call_args.args else ()
        kwargs = call_args.kwargs if call_args.kwargs else {}
        task_id_arg = args[0] if args else kwargs.get('task_id')
        assert str(task_id_arg) == 'starved', (
            f'Expected task_id="starved" in resolve callback; got {task_id_arg!r}'
        )
        assert 'starved' not in scheduler._starvation_escalated, (
            '_starvation_escalated must be cleared by GC sweep for terminal task'
        )
        assert 'starved' not in scheduler._starvation_first_seen, (
            '_starvation_first_seen must be cleared for terminal task'
        )

    @pytest.mark.asyncio
    async def test_gc_non_eligible_status_resolves_escalation(self):
        """GC backstop resolves escalation when task moves to blocked/deferred.

        A task in 'blocked' or 'deferred' is non-terminal but leaves the
        candidate pool.  The dispatch-site resolve never fires (no dispatch)
        and the _stale_ids sweep skips it (not terminal).  The extended GC
        backstop must detect the non-eligible status and auto-resolve.

        Drive 'starved' to escalated state, then flip its status to 'blocked'.
        One acquire_next tick must:
        - call _on_starvation_resolve exactly once with task_id='starved', AND
        - clear 'starved' from _starvation_escalated.

        Fails until the GC block checks _STARVATION_NON_ELIGIBLE in addition
        to _stale_ids.
        """
        scheduler, t = self._make_scheduler()
        warn_cb = AsyncMock()
        resolve_cb = AsyncMock()
        scheduler._on_starvation_warn = warn_cb
        scheduler._on_starvation_resolve = resolve_cb

        # Seed a held lock on 'seed' so 'starved' can never acquire.
        scheduler.lock_table.try_acquire('seed', ['backend'])
        scheduler._dispatched.add('seed')

        task = self._starved_task()
        scheduler.get_tasks = AsyncMock(return_value=[task])

        # Drive 3 ticks to cross skip_threshold (3), then advance past idle_secs.
        for _ in range(3):
            await scheduler.acquire_next()
        t[0] = 150.0
        await scheduler.acquire_next()

        # Pre-condition: warn fired, task is escalated.
        warn_cb.assert_called_once()
        assert 'starved' in scheduler._starvation_escalated, (
            'Precondition: starved must be in _starvation_escalated before GC tick'
        )
        resolve_cb.assert_not_called()

        # Change get_tasks to return 'starved' as 'blocked' (non-terminal, non-candidate).
        blocked_task = dict(task, status='blocked')
        scheduler.get_tasks = AsyncMock(return_value=[blocked_task])

        # One tick — the extended GC backstop must detect 'starved' is blocked,
        # call _on_starvation_resolve, and clear watchdog state.
        await scheduler.acquire_next()

        resolve_cb.assert_called_once()
        call_args = resolve_cb.call_args
        args = call_args.args if call_args.args else ()
        kwargs = call_args.kwargs if call_args.kwargs else {}
        task_id_arg = args[0] if args else kwargs.get('task_id')
        assert str(task_id_arg) == 'starved', (
            f'Expected task_id="starved" in resolve callback; got {task_id_arg!r}'
        )
        assert 'starved' not in scheduler._starvation_escalated, (
            '_starvation_escalated must be cleared by GC sweep for blocked task'
        )

    @pytest.mark.asyncio
    async def test_continuity_reset_drops_first_seen_when_not_candidate(self):
        """(b) Continuity reset: _starvation_first_seen cleared when task leaves candidates.

        A task eligible in tick 1 (pending, deps satisfied, in candidates) gets
        its _starvation_first_seen stamped.  When it becomes terminal (absent
        from candidates) on tick 2, the entry is dropped so the idle clock
        resets on any future re-appearance as a candidate.
        """
        scheduler, _t = self._make_scheduler()

        # Seed a held lock so 'starved' is a candidate but can't acquire.
        scheduler.lock_table.try_acquire('seed', ['backend'])
        scheduler._dispatched.add('seed')

        task = self._starved_task()
        scheduler.get_tasks = AsyncMock(return_value=[task])

        # Tick 1: 'starved' appears in candidates → _starvation_first_seen stamped.
        await scheduler.acquire_next()
        assert 'starved' in scheduler._starvation_first_seen, (
            'Precondition: _starvation_first_seen must be stamped after tick 1'
        )

        # Now make 'starved' terminal so it is no longer a dispatch-eligible candidate.
        terminal_task = dict(task, status='done')
        scheduler.get_tasks = AsyncMock(return_value=[terminal_task])

        # Tick 2: continuity reset in _apply_starvation_watchdog must drop the entry.
        await scheduler.acquire_next()

        assert 'starved' not in scheduler._starvation_first_seen, (
            '_starvation_first_seen must be cleared by the continuity reset '
            'when the task is no longer a dispatch-eligible candidate'
        )


# ---------------------------------------------------------------------------
# α dispatch-time directory-lock strip tests (PRD α, task 1906)
# ---------------------------------------------------------------------------

class TestGetModulesAlphaDirectoryStrip:
    """_get_modules must strip directory entries (α strip) before lock derivation.

    A directory entry like 'crates/reify-eval/src' (no recognised file extension)
    must NOT produce a subtree-wide prefix lock.  When ALL entries are directories
    the result must fall through to the existing task-<id> synthetic fallback.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        return Scheduler(config)

    def test_directory_only_files_returns_task_fallback(self, scheduler: Scheduler):
        """directory-only metadata.files → task-<id> fallback, not a wide module."""
        task = {
            'id': 'D',
            'metadata': {'files': ['crates/reify-eval/src', 'crates/reify-eval/tests']},
        }
        result = scheduler._get_modules(task)
        assert result == ['task-D'], (
            f'Expected task-D fallback for directory-only files, got: {result}'
        )

    def test_mixed_files_strips_directory_keeps_file_module(self, scheduler: Scheduler):
        """Mixed files: directory stripped, file sibling's module kept."""
        task = {
            'id': 'M',
            'metadata': {'files': ['crates/reify-eval/src', 'crates/reify-eval/src/foo.rs']},
        }
        result = scheduler._get_modules(task)
        # The file 'crates/reify-eval/src/foo.rs' should produce a module;
        # the bare directory 'crates/reify-eval/src' must NOT appear as a module entry.
        assert 'crates/reify-eval/src' not in result, (
            f'Directory entry must not appear as a derived module: {result}'
        )
        assert len(result) > 0, f'Expected at least one module from file sibling, got: {result}'

    def test_no_derived_module_equals_stripped_directory(self, scheduler: Scheduler):
        """No derived module may equal a stripped directory entry."""
        task = {
            'id': 'S',
            'metadata': {'files': ['backend', 'backend/app.py']},
        }
        result = scheduler._get_modules(task)
        # 'backend' is a directory entry and must not appear unchanged as a module.
        assert 'backend' not in result, (
            f'Bare directory "backend" must not survive into derived modules: {result}'
        )


@pytest.mark.asyncio
class TestAcquireNextDirectoryCharterBoundary:
    """G2 boundary / repro (reify-3468): a directory-charter task must NOT block
    an unrelated sibling that edits a file inside the same directory.

    This test reproduces the exact failure: at lock_depth=10, a directory entry
    like 'crates/reify-eval/src' previously survived normalize_lock unchanged and
    became a prefix lock that modules_conflict treated as conflicting with EVERY
    file under that subtree — so the sibling was blocked.

    After the α strip the directory charter yields no subtree lock (or the
    task-<id> synthetic fallback, which conflicts with nothing), so BOTH tasks
    are dispatchable.
    """

    async def test_directory_charter_does_not_block_sibling_file_task(self):
        """Both a directory-charter task and a file sibling are dispatchable."""
        config = OrchestratorConfig(lock_depth=10, max_per_module=1)
        scheduler = Scheduler(config)

        task_dir = {
            'id': 'dir',
            'title': 'Directory charter task',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['crates/reify-eval/src', 'crates/reify-eval/tests']},
        }
        task_sib = {
            'id': 'sib',
            'title': 'Sibling file task',
            'status': 'pending',
            'dependencies': [],
            'metadata': {'files': ['crates/reify-eval/src/unrelated.rs']},
        }
        scheduler.get_tasks = AsyncMock(return_value=[task_dir, task_sib])

        first = await scheduler.acquire_next()
        assert first is not None, 'First acquire_next() must dispatch a task'

        second = await scheduler.acquire_next()
        assert second is not None, (
            'Directory charter must NOT block the sibling: both should be dispatchable. '
            f'First dispatched: {first.task_id}; second was None (blocked).'
        )
