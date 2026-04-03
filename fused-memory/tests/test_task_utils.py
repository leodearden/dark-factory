"""Tests for fused_memory.utils.task_utils pure utility functions."""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.backends.taskmaster_client import TaskmasterBackend
from fused_memory.utils.task_utils import (
    _collect_all_tasks,
    _compact_task,
    _compact_tasks,
    _filter_tasks_by_status,
)


class TestFilterTasksByStatus:
    """Tests for _filter_tasks_by_status."""

    def test_filters_tasks_by_matching_status(self):
        tasks = [
            {'id': '1', 'status': 'pending', 'title': 'A'},
            {'id': '2', 'status': 'done', 'title': 'B'},
            {'id': '3', 'status': 'in-progress', 'title': 'C'},
        ]
        result = _filter_tasks_by_status(tasks, ['pending', 'in-progress'])
        ids = [t['id'] for t in result]
        assert ids == ['1', '3']

    def test_keeps_parent_when_subtask_matches(self):
        tasks = [
            {
                'id': '1',
                'status': 'done',
                'title': 'Parent',
                'subtasks': [
                    {'id': '1.1', 'status': 'pending', 'title': 'Child', 'subtasks': []},
                ],
            },
        ]
        result = _filter_tasks_by_status(tasks, ['pending'])
        assert len(result) == 1
        assert result[0]['id'] == '1'
        assert len(result[0]['subtasks']) == 1

    def test_skips_non_dict_elements(self):
        tasks = [
            {'id': '1', 'status': 'pending', 'title': 'A'},
            'not-a-dict',
            None,
        ]
        result = _filter_tasks_by_status(tasks, ['pending'])
        assert len(result) == 1
        assert result[0]['id'] == '1'

    def test_empty_list_returns_empty(self):
        result = _filter_tasks_by_status([], ['pending'])
        assert result == []

    def test_no_matches_returns_empty(self):
        tasks = [{'id': '1', 'status': 'done', 'title': 'A', 'subtasks': []}]
        result = _filter_tasks_by_status(tasks, ['pending'])
        assert result == []


class TestCollectAllTasks:
    """Tests for _collect_all_tasks."""

    def test_flat_list_returns_same_tasks(self):
        tasks = [
            {'id': '1', 'status': 'pending', 'title': 'A'},
            {'id': '2', 'status': 'done', 'title': 'B'},
        ]
        result = _collect_all_tasks(tasks)
        assert len(result) == 2
        assert result[0]['id'] == '1'
        assert result[1]['id'] == '2'

    def test_nested_subtasks_are_flattened(self):
        tasks = [
            {
                'id': '1',
                'status': 'pending',
                'title': 'Parent',
                'subtasks': [
                    {'id': '1.1', 'status': 'done', 'title': 'Child1', 'subtasks': []},
                    {'id': '1.2', 'status': 'in-progress', 'title': 'Child2', 'subtasks': []},
                ],
            },
        ]
        result = _collect_all_tasks(tasks)
        ids = [t['id'] for t in result]
        assert ids == ['1', '1.1', '1.2']

    def test_deeply_nested_tasks_included(self):
        tasks = [
            {
                'id': '1',
                'status': 'pending',
                'title': 'Root',
                'subtasks': [
                    {
                        'id': '1.1',
                        'status': 'done',
                        'title': 'Child',
                        'subtasks': [
                            {'id': '1.1.1', 'status': 'pending', 'title': 'Grandchild', 'subtasks': []},
                        ],
                    },
                ],
            },
        ]
        result = _collect_all_tasks(tasks)
        ids = [t['id'] for t in result]
        assert ids == ['1', '1.1', '1.1.1']

    def test_non_dict_elements_ignored(self):
        tasks = [
            {'id': '1', 'status': 'pending', 'title': 'A'},
            'not-a-dict',
            42,
        ]
        result = _collect_all_tasks(tasks)
        assert len(result) == 1
        assert result[0]['id'] == '1'

    def test_empty_list_returns_empty(self):
        result = _collect_all_tasks([])
        assert result == []


class TestCompactTask:
    """Tests for _compact_task."""

    def test_strips_description_and_details(self):
        task = {
            'id': '1',
            'status': 'pending',
            'title': 'A task',
            'description': 'Long description here',
            'details': 'Verbose implementation details',
            'dependencies': ['2', '3'],
            'priority': 'high',
        }
        result = _compact_task(task)
        assert 'description' not in result
        assert 'details' not in result

    def test_preserves_id_status_title_dependencies_priority(self):
        task = {
            'id': '1',
            'status': 'pending',
            'title': 'A task',
            'description': 'Should be stripped',
            'dependencies': ['2'],
            'priority': 'medium',
        }
        result = _compact_task(task)
        assert result['id'] == '1'
        assert result['status'] == 'pending'
        assert result['title'] == 'A task'
        assert result['dependencies'] == ['2']
        assert result['priority'] == 'medium'

    def test_recursively_compacts_subtasks(self):
        task = {
            'id': '1',
            'status': 'pending',
            'title': 'Parent',
            'description': 'Parent desc',
            'subtasks': [
                {
                    'id': '1.1',
                    'status': 'done',
                    'title': 'Child',
                    'description': 'Child desc',
                    'details': 'Child details',
                },
            ],
        }
        result = _compact_task(task)
        assert 'description' not in result
        assert len(result['subtasks']) == 1
        subtask = result['subtasks'][0]
        assert 'description' not in subtask
        assert 'details' not in subtask
        assert subtask['id'] == '1.1'

    def test_handles_missing_optional_fields_gracefully(self):
        task = {'id': '1', 'status': 'pending'}
        result = _compact_task(task)
        assert result['id'] == '1'
        assert result['status'] == 'pending'
        # No KeyError

    def test_non_dict_input_returned_unchanged(self):
        assert _compact_task('string') == 'string'
        assert _compact_task(42) == 42
        assert _compact_task(None) is None

    def test_no_subtasks_field_is_fine(self):
        task = {'id': '1', 'status': 'pending', 'title': 'A'}
        result = _compact_task(task)
        assert 'subtasks' not in result


class TestCompactTasks:
    """Tests for _compact_tasks."""

    def test_applies_compact_task_to_list(self):
        tasks = [
            {'id': '1', 'status': 'pending', 'title': 'A', 'description': 'Verbose'},
            {'id': '2', 'status': 'done', 'title': 'B', 'details': 'More verbose'},
        ]
        result = _compact_tasks(tasks)
        assert len(result) == 2
        assert 'description' not in result[0]
        assert 'details' not in result[1]
        assert result[0]['id'] == '1'
        assert result[1]['id'] == '2'

    def test_empty_list_returns_empty(self):
        result = _compact_tasks([])
        assert result == []


class TestTaskmasterBackendGetTasksSignature:
    """Verify TaskmasterBackend.get_tasks() has no dead 'status' parameter.

    Step 17 tests: these fail against the current code which still has the
    dead 'status: list[str] | None = None' parameter.
    """

    def test_get_tasks_signature_has_no_status_parameter(self):
        """get_tasks should only accept (self, project_root, tag) — no status."""
        sig = inspect.signature(TaskmasterBackend.get_tasks)
        param_names = list(sig.parameters.keys())
        # 'self' is included when inspecting an unbound method
        assert 'status' not in param_names, (
            f"TaskmasterBackend.get_tasks() has a dead 'status' parameter: {param_names}. "
            "Filtering is the interceptor's responsibility — remove it from the backend."
        )

    def test_get_tasks_parameters_are_project_root_and_tag(self):
        """get_tasks accepts exactly project_root and tag (besides self)."""
        sig = inspect.signature(TaskmasterBackend.get_tasks)
        param_names = [p for p in sig.parameters if p != 'self']
        assert param_names == ['project_root', 'tag'], (
            f"Expected ['project_root', 'tag'], got {param_names}"
        )

    @pytest.mark.asyncio
    async def test_get_tasks_returns_raw_call_tool_result_unchanged(self):
        """get_tasks must return the call_tool result directly, no post-processing."""
        config = MagicMock()
        backend = TaskmasterBackend(config=config)

        raw_result = {
            'tasks': [
                {'id': '1', 'status': 'done', 'title': 'Done task'},
                {'id': '2', 'status': 'pending', 'title': 'Pending task'},
            ]
        }

        with patch.object(backend, 'call_tool', new=AsyncMock(return_value=raw_result)):
            result = await backend.get_tasks(project_root='/some/project')

        # The result must be the exact dict returned by call_tool —
        # no filtering, no modification.
        assert result == raw_result
        assert len(result['tasks']) == 2  # both tasks preserved
