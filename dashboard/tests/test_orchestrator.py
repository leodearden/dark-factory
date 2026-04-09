"""Tests for dashboard.data.orchestrator — orchestrator discovery and status."""

from __future__ import annotations

import pytest


class TestFindRunningOrchestrators:
    """Tests for find_running_orchestrators — scans ps aux for orchestrator processes."""

    def test_parses_orchestrator_lines(self):
        """Two orchestrator lines with --prd flags produce two dicts with pid, prd, running, started."""
        import subprocess
        from unittest.mock import patch

        from dashboard.data.orchestrator import find_running_orchestrators

        ps_output = (
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\n"
            "leo       1234  0.5  1.2 123456  7890 ?        Sl   Mar18   0:05 python -m orchestrator run --prd /home/leo/prd1.md\n"
            "leo       5678  0.3  0.8 234567  4567 ?        Sl   10:30   0:02 python -m orchestrator run --prd /home/leo/prd2.md\n"
        )
        mock_result = subprocess.CompletedProcess(args=['ps', 'aux'], returncode=0, stdout=ps_output, stderr='')

        with patch('dashboard.data.orchestrator.subprocess.run', return_value=mock_result):
            result = find_running_orchestrators()

        assert len(result) == 2
        assert result[0]['pid'] == 1234
        assert result[0]['prd'] == '/home/leo/prd1.md'
        assert result[0]['config_path'] is None
        assert result[0]['running'] is True
        assert isinstance(result[0]['started'], str)
        assert result[1]['pid'] == 5678
        assert result[1]['prd'] == '/home/leo/prd2.md'

    def test_filters_out_grep_process(self):
        """A 'grep orchestrator' line in ps output is excluded from results."""
        import subprocess
        from unittest.mock import patch

        from dashboard.data.orchestrator import find_running_orchestrators

        ps_output = (
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\n"
            "leo       1234  0.5  1.2 123456  7890 ?        Sl   Mar18   0:05 python -m orchestrator run --prd /home/leo/prd1.md\n"
            "leo       9999  0.0  0.0  12345   678 pts/0    S+   10:31   0:00 grep orchestrator run\n"
        )
        mock_result = subprocess.CompletedProcess(args=['ps', 'aux'], returncode=0, stdout=ps_output, stderr='')

        with patch('dashboard.data.orchestrator.subprocess.run', return_value=mock_result):
            result = find_running_orchestrators()

        assert len(result) == 1
        assert result[0]['pid'] == 1234

    def test_no_orchestrators_running(self):
        """No orchestrator lines in ps output returns empty list."""
        import subprocess
        from unittest.mock import patch

        from dashboard.data.orchestrator import find_running_orchestrators

        ps_output = (
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\n"
            "leo       1111  0.1  0.5  54321  1234 ?        Ss   Mar17   1:23 /usr/bin/bash\n"
        )
        mock_result = subprocess.CompletedProcess(args=['ps', 'aux'], returncode=0, stdout=ps_output, stderr='')

        with patch('dashboard.data.orchestrator.subprocess.run', return_value=mock_result):
            result = find_running_orchestrators()

        assert result == []

    def test_subprocess_failure(self):
        """subprocess.run raising an exception returns empty list."""
        from unittest.mock import patch

        from dashboard.data.orchestrator import find_running_orchestrators

        with patch('dashboard.data.orchestrator.subprocess.run', side_effect=OSError('ps not found')):
            result = find_running_orchestrators()

        assert result == []

    def test_unexpected_exception_propagates(self):
        """subprocess.run raising RuntimeError (non-subprocess error) propagates to caller.

        With the narrowed `except (OSError, subprocess.TimeoutExpired):`, unexpected
        errors like RuntimeError should NOT be swallowed — they indicate bugs and
        should be visible to callers.
        """
        from unittest.mock import patch

        from dashboard.data.orchestrator import find_running_orchestrators

        with pytest.raises(RuntimeError, match='unexpected'), patch(
            'dashboard.data.orchestrator.subprocess.run',
            side_effect=RuntimeError('unexpected'),
        ):
            find_running_orchestrators()

    def test_subprocess_timeout_caught(self):
        """subprocess.run raising TimeoutExpired is caught and returns empty list.

        TimeoutExpired is a legitimate subprocess failure mode and should remain
        handled gracefully.
        """
        import subprocess
        from unittest.mock import patch

        from dashboard.data.orchestrator import find_running_orchestrators

        with patch(
            'dashboard.data.orchestrator.subprocess.run',
            side_effect=subprocess.TimeoutExpired(cmd=['ps', 'aux'], timeout=30),
        ):
            result = find_running_orchestrators()

        assert result == []

    def test_malformed_pid_skips_line(self):
        """A line with a non-integer PID field is silently skipped; valid lines still parsed."""
        import subprocess
        from unittest.mock import patch

        from dashboard.data.orchestrator import find_running_orchestrators

        ps_output = (
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\n"
            "leo       N/A   0.5  1.2 123456  7890 ?        Sl   Mar18   0:05 python -m orchestrator run --prd /home/leo/bad.md\n"
            "leo       4321  0.3  0.8 234567  4567 ?        Sl   10:30   0:02 python -m orchestrator run --prd /home/leo/good.md\n"
        )
        mock_result = subprocess.CompletedProcess(args=['ps', 'aux'], returncode=0, stdout=ps_output, stderr='')

        with patch('dashboard.data.orchestrator.subprocess.run', return_value=mock_result):
            result = find_running_orchestrators()

        assert len(result) == 1
        assert result[0]['pid'] == 4321
        assert result[0]['prd'] == '/home/leo/good.md'

    def test_truncated_ps_line_skips(self):
        """A truncated line that passes filters but has insufficient fields is skipped."""
        import subprocess
        from unittest.mock import patch

        from dashboard.data.orchestrator import find_running_orchestrators

        ps_output = (
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\n"
            "leo 999 orchestrator run --prd\n"
            "leo       8888  0.3  0.8 234567  4567 ?        Sl   10:30   0:02 python -m orchestrator run --prd /home/leo/ok.md\n"
        )
        mock_result = subprocess.CompletedProcess(args=['ps', 'aux'], returncode=0, stdout=ps_output, stderr='')

        with patch('dashboard.data.orchestrator.subprocess.run', return_value=mock_result):
            result = find_running_orchestrators()

        assert len(result) == 1
        assert result[0]['pid'] == 8888
        assert result[0]['prd'] == '/home/leo/ok.md'

    def test_detects_config_flag_orchestrator(self):
        """Orchestrator with --config flag is detected with config_path extracted."""
        import subprocess
        from unittest.mock import patch

        from dashboard.data.orchestrator import find_running_orchestrators

        ps_output = (
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\n"
            "leo       2222  1.6  0.1 2361784 82580 ?       Sl   18:48   0:10 python orchestrator run --config /home/leo/src/reify/orchestrator.yaml\n"
        )
        mock_result = subprocess.CompletedProcess(args=['ps', 'aux'], returncode=0, stdout=ps_output, stderr='')

        with patch('dashboard.data.orchestrator.subprocess.run', return_value=mock_result):
            result = find_running_orchestrators()

        assert len(result) == 1
        assert result[0]['pid'] == 2222
        assert result[0]['prd'] is None
        assert result[0]['config_path'] == '/home/leo/src/reify/orchestrator.yaml'

    def test_detects_bare_orchestrator_run(self):
        """Orchestrator with no flags (bare 'orchestrator run') is detected."""
        import subprocess
        from unittest.mock import patch

        from dashboard.data.orchestrator import find_running_orchestrators

        ps_output = (
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\n"
            "leo       3333  0.4  0.0 847184 52816 ?        Sl   18:48   0:02 /home/leo/src/dark-factory/orchestrator/.venv/bin/python3 /home/leo/src/dark-factory/orchestrator/.venv/bin/orchestrator run\n"
        )
        mock_result = subprocess.CompletedProcess(args=['ps', 'aux'], returncode=0, stdout=ps_output, stderr='')

        with patch('dashboard.data.orchestrator.subprocess.run', return_value=mock_result):
            result = find_running_orchestrators()

        assert len(result) == 1
        assert result[0]['pid'] == 3333
        assert result[0]['prd'] is None
        assert result[0]['config_path'] is None

    def test_filters_non_run_orchestrator_lines(self):
        """Lines containing 'orchestrator' but not 'orchestrator run' are excluded."""
        import subprocess
        from unittest.mock import patch

        from dashboard.data.orchestrator import find_running_orchestrators

        ps_output = (
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\n"
            "leo       4444  0.1  0.5 123456  7890 ?        Sl   Mar18   0:05 python -m dashboard.app --orchestrator-panel\n"
            "leo       5555  0.1  0.5 123456  7890 ?        Sl   Mar18   0:05 uv run --project orchestrator orchestrator status\n"
        )
        mock_result = subprocess.CompletedProcess(args=['ps', 'aux'], returncode=0, stdout=ps_output, stderr='')

        with patch('dashboard.data.orchestrator.subprocess.run', return_value=mock_result):
            result = find_running_orchestrators()

        assert result == []


class TestLoadTaskTree:
    """Tests for load_task_tree — parses tasks.json into a list of task dicts."""

    def test_parses_master_format(self, tmp_path):
        """tasks.json with {'master': {'tasks': [...]}} is parsed correctly."""
        import json

        from dashboard.data.orchestrator import load_task_tree

        tasks = [
            {'id': '1', 'title': 'Setup', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {'assignee': 'leo'}},
            {'id': '2', 'title': 'Build', 'status': 'in-progress', 'priority': 'medium', 'dependencies': ['1'], 'metadata': {}},
            {'id': '3', 'title': 'Test', 'status': 'pending', 'priority': 'low', 'dependencies': ['2'], 'metadata': {}},
        ]
        tasks_json = tmp_path / 'tasks.json'
        tasks_json.write_text(json.dumps({'master': {'tasks': tasks}}))

        result = load_task_tree(tasks_json)

        assert len(result) == 3
        assert result[0]['id'] == 1
        assert result[0]['title'] == 'Setup'
        assert result[0]['status'] == 'done'
        assert result[0]['priority'] == 'high'
        assert result[0]['dependencies'] == []
        assert result[0]['metadata'] == {'assignee': 'leo'}

    def test_parses_flat_format(self, tmp_path):
        """tasks.json with {'tasks': [...]} (no master wrapper) is parsed correctly."""
        import json

        from dashboard.data.orchestrator import load_task_tree

        tasks = [
            {'id': '1', 'title': 'Setup', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {}},
            {'id': '2', 'title': 'Build', 'status': 'pending', 'priority': 'medium', 'dependencies': ['1'], 'metadata': {}},
        ]
        tasks_json = tmp_path / 'tasks.json'
        tasks_json.write_text(json.dumps({'tasks': tasks}))

        result = load_task_tree(tasks_json)

        assert len(result) == 2
        assert result[0]['status'] == 'done'
        assert result[1]['dependencies'] == [1]

    def test_file_not_found(self, tmp_path):
        """Non-existent path returns empty list."""
        from dashboard.data.orchestrator import load_task_tree

        result = load_task_tree(tmp_path / 'nonexistent.json')

        assert result == []

    def test_malformed_json(self, tmp_path):
        """Invalid JSON in file returns empty list."""
        from dashboard.data.orchestrator import load_task_tree

        bad_file = tmp_path / 'tasks.json'
        bad_file.write_text('{not valid json!!!')

        result = load_task_tree(bad_file)

        assert result == []

    def test_extracts_expected_keys(self, tmp_path):
        """Each returned task dict has all expected keys."""
        import json

        from dashboard.data.orchestrator import load_task_tree

        tasks = [
            {'id': '1', 'title': 'Task One', 'status': 'pending', 'priority': 'high', 'dependencies': ['2'], 'metadata': {'tag': 'x'}},
        ]
        tasks_json = tmp_path / 'tasks.json'
        tasks_json.write_text(json.dumps({'tasks': tasks}))

        result = load_task_tree(tasks_json)

        assert len(result) == 1
        expected_keys = {'id', 'title', 'status', 'priority', 'dependencies', 'metadata'}
        assert set(result[0].keys()) == expected_keys

    def test_null_tasks_in_master_format(self, tmp_path):
        """tasks.json with {'master': {'tasks': null}} returns empty list, not TypeError."""
        import json

        from dashboard.data.orchestrator import load_task_tree

        tasks_json = tmp_path / 'tasks.json'
        tasks_json.write_text(json.dumps({'master': {'tasks': None}}))

        result = load_task_tree(tasks_json)

        assert result == []

    def test_null_tasks_in_flat_format(self, tmp_path):
        """tasks.json with {'tasks': null} returns empty list, not TypeError."""
        import json

        from dashboard.data.orchestrator import load_task_tree

        tasks_json = tmp_path / 'tasks.json'
        tasks_json.write_text(json.dumps({'tasks': None}))

        result = load_task_tree(tasks_json)

        assert result == []

    def test_tasks_not_a_list(self, tmp_path):
        """tasks.json with {'tasks': 'oops'} (string instead of list) returns empty list."""
        import json

        from dashboard.data.orchestrator import load_task_tree

        tasks_json = tmp_path / 'tasks.json'
        tasks_json.write_text(json.dumps({'tasks': 'oops'}))

        result = load_task_tree(tasks_json)

        assert result == []


    def test_non_numeric_id_skipped(self, tmp_path):
        """Task with non-numeric ID (e.g. 'task-abc') is skipped, not crashed on."""
        import json

        from dashboard.data.orchestrator import load_task_tree

        tasks_json = tmp_path / 'tasks.json'
        tasks_json.write_text(json.dumps({'tasks': [
            {'id': 'task-abc', 'title': 'Bad', 'status': 'pending'},
            {'id': '5', 'title': 'Good', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {}},
        ]}))

        result = load_task_tree(tasks_json)
        assert len(result) == 1
        assert result[0]['id'] == 5

    def test_string_dependencies_normalised_to_int(self, tmp_path):
        """String dependency values are converted to int for consistent comparisons."""
        import json

        from dashboard.data.orchestrator import load_task_tree

        tasks_json = tmp_path / 'tasks.json'
        tasks_json.write_text(json.dumps({'tasks': [
            {'id': 3, 'title': 'T3', 'status': 'done', 'dependencies': ['1', '2']},
        ]}))

        result = load_task_tree(tasks_json)
        assert result[0]['dependencies'] == [1, 2]

    def test_permission_error_returns_empty_list(self, tmp_path):
        """PermissionError raised by Path.read_text is handled; returns empty list."""
        import json
        from pathlib import Path
        from unittest.mock import patch

        from dashboard.data.orchestrator import load_task_tree

        tasks_json = tmp_path / 'tasks.json'
        tasks_json.write_text(json.dumps({'tasks': [{'id': '1', 'status': 'done'}]}))

        with patch.object(Path, 'read_text', side_effect=PermissionError('Permission denied')):
            result = load_task_tree(tasks_json)

        assert result == []


class TestReadTaskArtifacts:
    """Tests for read_task_artifacts — reads .task/ directory in a worktree."""

    def test_full_artifacts(self, tmp_path):
        """Worktree with complete .task/ returns metadata, phase, progress, iterations, reviews."""
        import json

        from dashboard.data.orchestrator import read_task_artifacts

        task_dir = tmp_path / '.task'
        task_dir.mkdir()
        reviews_dir = task_dir / 'reviews'
        reviews_dir.mkdir()

        # metadata.json
        metadata = {'task_id': '7', 'title': 'Build widget', 'base_commit': 'abc123', 'created_at': '2026-03-18T10:00:00'}
        (task_dir / 'metadata.json').write_text(json.dumps(metadata))

        # plan.json with 3 done, 2 pending out of 5 total + modules
        steps = [
            {'id': f'step-{i}', 'status': 'done'} for i in range(1, 4)
        ] + [
            {'id': f'step-{i}', 'status': 'pending'} for i in range(4, 6)
        ]
        (task_dir / 'plan.json').write_text(json.dumps({'steps': steps, 'modules': ['dashboard']}))

        # iterations.jsonl with 3 lines
        with open(task_dir / 'iterations.jsonl', 'w') as f:
            for i in range(3):
                f.write(json.dumps({'iteration': i + 1}) + '\n')

        # Two reviews
        (reviews_dir / 'reviewer-1.json').write_text(json.dumps({'verdict': 'PASS'}))
        (reviews_dir / 'reviewer-2.json').write_text(json.dumps({'verdict': 'ISSUES_FOUND'}))

        result = read_task_artifacts(tmp_path)

        assert result['metadata'] == metadata
        assert result['phase'] == 'EXECUTE'
        assert result['plan_progress'] == {'done': 3, 'total': 5}
        assert result['iteration_count'] == 3
        assert result['review_summary'] == '1/2 passed'
        assert result['modules'] == ['dashboard']

    def test_empty_worktree(self, tmp_path):
        """Worktree with no .task/ subdir returns defaults."""
        from dashboard.data.orchestrator import read_task_artifacts

        result = read_task_artifacts(tmp_path)

        assert result['metadata'] is None
        assert result['phase'] == 'PLAN'
        assert result['plan_progress'] == {'done': 0, 'total': 0}
        assert result['iteration_count'] == 0
        assert result['review_summary'] == '\u2014'

    def test_plan_all_done(self, tmp_path):
        """All steps done in plan.json results in phase='DONE'."""
        import json

        from dashboard.data.orchestrator import read_task_artifacts

        task_dir = tmp_path / '.task'
        task_dir.mkdir()

        steps = [{'id': f'step-{i}', 'status': 'done'} for i in range(1, 4)]
        (task_dir / 'plan.json').write_text(json.dumps({'steps': steps}))

        result = read_task_artifacts(tmp_path)

        assert result['phase'] == 'DONE'
        assert result['plan_progress'] == {'done': 3, 'total': 3}

    def test_no_plan_file(self, tmp_path):
        """Worktree with .task/ but no plan.json results in phase='PLAN'."""
        import json

        from dashboard.data.orchestrator import read_task_artifacts

        task_dir = tmp_path / '.task'
        task_dir.mkdir()

        metadata = {'task_id': '7', 'title': 'Build widget'}
        (task_dir / 'metadata.json').write_text(json.dumps(metadata))

        result = read_task_artifacts(tmp_path)

        assert result['phase'] == 'PLAN'
        assert result['plan_progress'] == {'done': 0, 'total': 0}
        assert result['metadata'] == metadata

    def test_modules_extracted_from_plan(self, tmp_path):
        """Modules list is extracted from plan.json top-level 'modules' key."""
        import json

        from dashboard.data.orchestrator import read_task_artifacts

        task_dir = tmp_path / '.task'
        task_dir.mkdir()

        plan_data = {
            'modules': ['auth/', 'api/'],
            'steps': [{'id': 'step-1', 'status': 'pending'}],
        }
        (task_dir / 'plan.json').write_text(json.dumps(plan_data))

        result = read_task_artifacts(tmp_path)

        assert result['modules'] == ['auth/', 'api/']

    def test_modules_default_empty_when_missing(self, tmp_path):
        """When plan.json has no 'modules' key, modules defaults to empty list."""
        import json

        from dashboard.data.orchestrator import read_task_artifacts

        task_dir = tmp_path / '.task'
        task_dir.mkdir()

        plan_data = {'steps': [{'id': 'step-1', 'status': 'done'}]}
        (task_dir / 'plan.json').write_text(json.dumps(plan_data))

        result = read_task_artifacts(tmp_path)

        assert result['modules'] == []

    def test_modules_default_empty_no_plan(self, tmp_path):
        """When plan.json doesn't exist, modules defaults to empty list."""
        from dashboard.data.orchestrator import read_task_artifacts

        result = read_task_artifacts(tmp_path)

        assert result['modules'] == []


class TestExtractTaskId:
    """Tests for _extract_task_id — normalises worktree directory names to numeric task IDs."""

    def test_strips_task_prefix(self):
        """'task-7' returns 7."""
        from dashboard.data.orchestrator import _extract_task_id

        assert _extract_task_id('task-7') == 7

    def test_plain_numeric_unchanged(self):
        """Plain numeric string '33' returns 33."""
        from dashboard.data.orchestrator import _extract_task_id

        assert _extract_task_id('33') == 33

    def test_multi_digit_task_prefix(self):
        """'task-123' returns 123."""
        from dashboard.data.orchestrator import _extract_task_id

        assert _extract_task_id('task-123') == 123

    def test_task_prefix_only_returns_none(self):
        """'task-' with empty suffix returns None (invalid)."""
        from dashboard.data.orchestrator import _extract_task_id

        assert _extract_task_id('task-') is None

    def test_empty_string_returns_none(self):
        """Empty string returns None."""
        from dashboard.data.orchestrator import _extract_task_id

        assert _extract_task_id('') is None

    def test_non_task_dir_returns_none(self):
        """'random-dir' returns None (non-task directory excluded)."""
        from dashboard.data.orchestrator import _extract_task_id

        assert _extract_task_id('random-dir') is None

    def test_non_numeric_suffix_returns_none(self):
        """'task-abc' returns None (non-numeric suffix excluded)."""
        from dashboard.data.orchestrator import _extract_task_id

        assert _extract_task_id('task-abc') is None


class TestDiscoverOrchestrators:
    """Tests for discover_orchestrators — combines process, task tree, and artifact data."""

    def test_combines_process_and_worktree_data(self, tmp_path):
        """Running orchestrator is enriched with task tree and worktree artifact data."""
        import json
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        # Create tasks.json with 5 tasks of varying statuses + a worktree task
        tasks_dir = tmp_path / '.taskmaster' / 'tasks'
        tasks_dir.mkdir(parents=True)
        tasks = [
            {'id': '1', 'title': 'Setup', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {}},
            {'id': '2', 'title': 'Build', 'status': 'done', 'priority': 'high', 'dependencies': ['1'], 'metadata': {}},
            {'id': '3', 'title': 'Test', 'status': 'in-progress', 'priority': 'medium', 'dependencies': ['2'], 'metadata': {}},
            {'id': '4', 'title': 'Review', 'status': 'blocked', 'priority': 'medium', 'dependencies': ['3'], 'metadata': {}},
            {'id': '5', 'title': 'Deploy', 'status': 'pending', 'priority': 'low', 'dependencies': ['4'], 'metadata': {}},
            {'id': '7', 'title': 'Widget', 'status': 'in-progress', 'priority': 'high', 'dependencies': [], 'metadata': {}},
        ]
        (tasks_dir / 'tasks.json').write_text(json.dumps({'tasks': tasks}))

        # Create a worktree with .task/ artifacts
        wt_dir = tmp_path / '.worktrees' / '7'
        wt_dir.mkdir(parents=True)
        task_dir = wt_dir / '.task'
        task_dir.mkdir()
        (task_dir / 'metadata.json').write_text(json.dumps({'task_id': '7', 'title': 'Widget'}))
        steps = [{'id': 'step-1', 'status': 'done'}, {'id': 'step-2', 'status': 'pending'}]
        (task_dir / 'plan.json').write_text(json.dumps({'steps': steps}))

        # Mock one running orchestrator — PRD inside tmp_path so project root resolves
        prd_path = str(tmp_path / 'prd.md')
        mock_procs = [{'pid': 1234, 'prd': prd_path, 'config_path': None, 'running': True, 'started': 'Mar18'}]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = discover_orchestrators(config)

        assert len(result) == 1
        entry = result[0]
        assert 1234 in entry['pids']
        assert entry['prd'] == prd_path
        assert entry['running'] is True
        assert len(entry['tasks']) == 6
        assert 7 in entry['worktrees']
        assert entry['worktrees'][7]['phase'] == 'EXECUTE'
        assert entry['summary'] == {'total': 6, 'done': 2, 'in_progress': 2, 'blocked': 1, 'pending': 1}

    def test_no_running_orchestrators(self, tmp_path):
        """Empty process list returns empty result."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=[]):
            result = discover_orchestrators(config)

        assert result == []

    def test_missing_tasks_json(self, tmp_path):
        """Orchestrator returned even when tasks.json is missing (empty task list)."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        # PRD inside tmp_path but no .taskmaster — falls back to config.project_root
        mock_procs = [{'pid': 5678, 'prd': str(tmp_path / 'prd.md'), 'config_path': None, 'running': True, 'started': '10:30'}]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = discover_orchestrators(config)

        assert len(result) == 1
        assert result[0]['tasks'] == []
        assert result[0]['worktrees'] == {}
        assert result[0]['summary'] == {'total': 0, 'done': 0, 'in_progress': 0, 'blocked': 0, 'pending': 0}

    def test_worktree_keyed_by_task_id_not_dirname(self, tmp_path):
        """Worktree dir named 'task-7' is keyed by '7' (not 'task-7') in discover_orchestrators output."""
        import json
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        # Create tasks.json with a task whose id matches the worktree
        tasks_dir = tmp_path / '.taskmaster' / 'tasks'
        tasks_dir.mkdir(parents=True)
        tasks = [{'id': '7', 'title': 'Widget', 'status': 'in-progress', 'priority': 'high', 'dependencies': [], 'metadata': {}}]
        (tasks_dir / 'tasks.json').write_text(json.dumps({'tasks': tasks}))

        # Create worktree directory using 'task-7' naming convention
        wt_dir = tmp_path / '.worktrees' / 'task-7'
        wt_dir.mkdir(parents=True)
        task_dir = wt_dir / '.task'
        task_dir.mkdir()
        steps = [{'id': 'step-1', 'status': 'done'}, {'id': 'step-2', 'status': 'pending'}]
        (task_dir / 'plan.json').write_text(json.dumps({'steps': steps}))

        prd_path = str(tmp_path / 'prd.md')
        mock_procs = [{'pid': 9000, 'prd': prd_path, 'config_path': None, 'running': True, 'started': 'Mar19'}]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = discover_orchestrators(config)

        assert len(result) == 1
        worktrees = result[0]['worktrees']
        # Key must be 7 (int) not 'task-7'
        assert 7 in worktrees
        assert 'task-7' not in worktrees
        assert worktrees[7]['phase'] == 'EXECUTE'

    def test_non_task_worktree_dirs_excluded(self, tmp_path):
        """Non-task directories (e.g. 'tmp-backup') are excluded; plain and 'task-' numeric dirs included."""
        import json
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        # Create tasks.json with tasks matching the worktree IDs
        tasks_dir = tmp_path / '.taskmaster' / 'tasks'
        tasks_dir.mkdir(parents=True)
        tasks = [
            {'id': '3', 'title': 'T3', 'status': 'pending', 'priority': 'medium', 'dependencies': [], 'metadata': {}},
            {'id': '5', 'title': 'T5', 'status': 'pending', 'priority': 'medium', 'dependencies': [], 'metadata': {}},
        ]
        (tasks_dir / 'tasks.json').write_text(json.dumps({'tasks': tasks}))

        worktrees_dir = tmp_path / '.worktrees'
        worktrees_dir.mkdir()

        # Create 'task-3', '5', and 'tmp-backup' under .worktrees/
        for name in ('task-3', '5', 'tmp-backup'):
            d = worktrees_dir / name
            d.mkdir()

        prd_path = str(tmp_path / 'prd.md')
        mock_procs = [{'pid': 1111, 'prd': prd_path, 'config_path': None, 'running': True, 'started': 'Mar19'}]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = discover_orchestrators(config)

        worktrees = result[0]['worktrees']
        assert 3 in worktrees
        assert 5 in worktrees
        assert 'tmp-backup' not in worktrees
        assert 'task-3' not in worktrees

    def test_single_process_produces_pids_list(self, tmp_path):
        """Single running orchestrator produces entry with 'pids' list and no 'pid' key."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        mock_procs = [{'pid': 1234, 'prd': '/home/leo/prd.md', 'config_path': None, 'running': True, 'started': 'Mar18'}]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = discover_orchestrators(config)

        assert len(result) == 1
        entry = result[0]
        assert 'pids' in entry
        assert isinstance(entry['pids'], list)
        assert entry['pids'] == [1234]
        assert 'pid' not in entry

    def test_same_prd_grouped_into_single_entry(self, tmp_path):
        """Two processes with the same PRD path are merged into one entry with both PIDs."""
        import json
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        # Create tasks.json so the shared task tree is populated
        tasks_dir = tmp_path / '.taskmaster' / 'tasks'
        tasks_dir.mkdir(parents=True)
        tasks = [
            {'id': '1', 'title': 'Setup', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {}},
        ]
        (tasks_dir / 'tasks.json').write_text(json.dumps({'tasks': tasks}))

        # Create a worktree so worktrees dict is populated
        wt_dir = tmp_path / '.worktrees' / '1'
        wt_dir.mkdir(parents=True)

        # Two processes with identical PRD path but different PIDs
        prd_path = str(tmp_path / 'prd.md')
        mock_procs = [
            {'pid': 1234, 'prd': prd_path, 'config_path': None, 'running': True, 'started': 'Mar18'},
            {'pid': 5678, 'prd': prd_path, 'config_path': None, 'running': False, 'started': 'Mar18'},
        ]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = discover_orchestrators(config)

        # Should produce exactly one entry (not two)
        assert len(result) == 1
        entry = result[0]
        assert entry['pids'] == [1234, 5678]
        # Shared data should appear once
        assert len(entry['tasks']) == 1
        assert 1 in entry['worktrees']
        assert entry['summary']['total'] == 1

    def test_different_projects_produce_separate_entries(self, tmp_path):
        """Two processes targeting different project roots produce two separate entries."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        # Create two separate projects with .taskmaster/ dirs
        proj_a = tmp_path / 'proj_a'
        (proj_a / '.taskmaster').mkdir(parents=True)
        proj_b = tmp_path / 'proj_b'
        (proj_b / '.taskmaster').mkdir(parents=True)

        mock_procs = [
            {'pid': 1234, 'prd': str(proj_a / 'prd.md'), 'config_path': None, 'running': True, 'started': 'Mar18'},
            {'pid': 5678, 'prd': str(proj_b / 'prd.md'), 'config_path': None, 'running': True, 'started': 'Mar18'},
        ]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = discover_orchestrators(config)

        assert len(result) == 2
        pids_by_root = {entry['project_root']: entry['pids'] for entry in result}
        assert pids_by_root[str(proj_a)] == [1234]
        assert pids_by_root[str(proj_b)] == [5678]

    def test_grouped_running_true_when_any_running(self, tmp_path):
        """Grouped entry has running=True if at least one process is still running."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        # One running, one not — grouped result should be running=True
        mock_procs = [
            {'pid': 1234, 'prd': '/prd.md', 'config_path': None, 'running': True, 'started': 'Mar18'},
            {'pid': 5678, 'prd': '/prd.md', 'config_path': None, 'running': False, 'started': 'Mar17'},
        ]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = discover_orchestrators(config)

        assert len(result) == 1
        assert result[0]['running'] is True

    def test_grouped_running_false_when_all_completed(self, tmp_path):
        """Grouped entry has running=False if all processes in the group have completed."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        # Both not running — grouped result should be running=False
        mock_procs = [
            {'pid': 1234, 'prd': '/prd.md', 'config_path': None, 'running': False, 'started': 'Mar18'},
            {'pid': 5678, 'prd': '/prd.md', 'config_path': None, 'running': False, 'started': 'Mar17'},
        ]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = discover_orchestrators(config)

        assert len(result) == 1
        assert result[0]['running'] is False

    def test_bare_fallback_with_symlink_config_root(self, tmp_path):
        """Bare process (no prd, no config_path) with symlinked config.project_root returns canonical path.

        When a process has neither prd nor config_path, _resolve_root falls back to
        config.project_root (line 308).  DashboardConfig.__post_init__ already resolves
        the symlink, so project_root is canonical before it reaches line 348's
        str(project_root.resolve()).  This test confirms the full bare-fallback pipeline
        produces a canonical project_root in the result.
        """
        import json
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real_dir)

        # Create tasks.json under the real directory
        (real_dir / ".taskmaster" / "tasks").mkdir(parents=True)
        (real_dir / ".taskmaster" / "tasks" / "tasks.json").write_text(
            json.dumps({"tasks": [
                {"id": "1", "title": "T", "status": "done", "priority": "high", "dependencies": [], "metadata": {}},
            ]})
        )

        # Config points at the symlink — __post_init__ resolves it to real_dir
        config = DashboardConfig(project_root=link)

        # Bare process: no prd, no config_path — hits line 308 return config.project_root
        mock_procs = [{"pid": 1234, "prd": None, "config_path": None, "running": True, "started": "Apr09"}]
        with patch("dashboard.data.orchestrator.find_running_orchestrators", return_value=mock_procs):
            result = discover_orchestrators(config)

        assert len(result) == 1
        # project_root must be the resolved canonical path, not the symlink
        assert result[0]["project_root"] == str(real_dir)
        assert result[0]["pids"] == [1234]

    def test_symlink_and_canonical_paths_grouped_into_single_entry(self, tmp_path):
        """Two processes whose PRDs resolve to the same project root are merged into one entry.

        Process A has its PRD under a symlinked directory component; process B has its
        PRD under the real (canonical) path.  Both walk up to the same .taskmaster/
        ancestor.  The grouping dict uses resolved Path objects as keys, so both
        processes map to the same key and appear in a single entry with both PIDs.
        """
        import json
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        real_dir = tmp_path / "real_proj"
        real_dir.mkdir()
        link_dir = tmp_path / "link_proj"
        link_dir.symlink_to(real_dir)

        # Create tasks.json under the real directory
        (real_dir / ".taskmaster" / "tasks").mkdir(parents=True)
        (real_dir / ".taskmaster" / "tasks" / "tasks.json").write_text(
            json.dumps({"tasks": [
                {"id": "2", "title": "Work", "status": "in-progress", "priority": "high", "dependencies": [], "metadata": {}},
            ]})
        )

        (tmp_path / "unrelated").mkdir()
        config = DashboardConfig(project_root=tmp_path / "unrelated")

        # Process A: PRD under the symlink — _resolve_project_root resolves it to real_dir
        prd_via_symlink = str(link_dir / "docs" / "prd.md")
        # Process B: PRD under the canonical path — also resolves to real_dir
        prd_canonical = str(real_dir / "docs" / "prd.md")

        mock_procs = [
            {"pid": 111, "prd": prd_via_symlink, "config_path": None, "running": True, "started": "Apr09"},
            {"pid": 222, "prd": prd_canonical, "config_path": None, "running": True, "started": "Apr09"},
        ]
        with patch("dashboard.data.orchestrator.find_running_orchestrators", return_value=mock_procs):
            result = discover_orchestrators(config)

        # Both processes share the same canonical root → exactly one entry
        assert len(result) == 1
        assert set(result[0]["pids"]) == {111, 222}
        assert result[0]["project_root"] == str(real_dir)


class TestResolveProjectRoot:
    """Tests for _resolve_project_root — finds project root from PRD path."""

    def test_finds_taskmaster_dir(self, tmp_path):
        """Walks up from PRD and finds ancestor with .taskmaster/."""
        from pathlib import Path

        from dashboard.data.orchestrator import _resolve_project_root

        (tmp_path / '.taskmaster').mkdir()
        prd = str(tmp_path / 'docs' / 'prd.md')
        assert _resolve_project_root(prd, Path('/fallback')) == tmp_path

    def test_relative_prd_resolved_against_default(self, tmp_path):
        """Relative PRD is resolved against default_root before walking up."""
        from dashboard.data.orchestrator import _resolve_project_root

        (tmp_path / '.taskmaster').mkdir()
        assert _resolve_project_root('docs/prd.md', tmp_path) == tmp_path

    def test_falls_back_to_default(self, tmp_path):
        """Returns default_root when no .taskmaster/ found in any ancestor."""
        from dashboard.data.orchestrator import _resolve_project_root

        default = tmp_path / 'default'
        default.mkdir()
        assert _resolve_project_root('/nowhere/prd.md', default) == default

    def test_dotdot_in_path(self, tmp_path):
        """Paths with .. are canonicalized before walking."""
        from pathlib import Path

        from dashboard.data.orchestrator import _resolve_project_root

        (tmp_path / '.taskmaster').mkdir()
        prd = str(tmp_path / 'docs' / '..' / 'docs' / 'prd.md')
        assert _resolve_project_root(prd, Path('/fallback')) == tmp_path

    def test_fallback_returns_resolved_default_root(self, tmp_path):
        """Fallback path returns the resolved (canonical) default_root, not a symlink."""
        from dashboard.data.orchestrator import _resolve_project_root

        real = tmp_path / 'real'
        real.mkdir()
        link = tmp_path / 'link'
        link.symlink_to(real)

        # No .taskmaster anywhere in the PRD's ancestor chain → falls back to default_root
        result = _resolve_project_root('/nowhere/prd.md', link)

        # Must return the resolved real path, not the symlink
        assert result == real
        assert result != link

    def test_symlink_component_in_prd_path_returns_canonical_ancestor(self, tmp_path):
        """PRD whose middle directory is a symlink resolves to the canonical ancestor.

        When a component of the PRD path is a symlink, p.resolve() at line 35 rewrites
        the full path to its canonical form before the ancestor walk begins.  Without
        that resolve call, Path.parents would walk symlink parents rather than real
        parents and could miss the .taskmaster/ directory.  This test guards against
        regressions in that resolve call.
        """
        from pathlib import Path

        from dashboard.data.orchestrator import _resolve_project_root

        real_proj = tmp_path / "real_proj"
        real_proj.mkdir()
        (real_proj / ".taskmaster").mkdir()

        # Create a symlink: link -> real_proj
        link = tmp_path / "link"
        link.symlink_to(real_proj)

        # PRD is under the symlink component — middle dir is a symlink
        prd = str(link / "docs" / "prd.md")

        result = _resolve_project_root(prd, Path("/fallback"))

        # Must return the canonical real_proj, not the symlink variant
        assert result == real_proj
        assert result != link


class TestScanWorktrees:
    """Tests for _scan_worktrees — reads worktree artifacts from a directory."""

    def test_scans_task_dirs(self, tmp_path):
        from dashboard.data.orchestrator import _scan_worktrees

        wt_dir = tmp_path / '.worktrees'
        wt_dir.mkdir()
        (wt_dir / '5').mkdir()
        (wt_dir / 'task-7').mkdir()
        (wt_dir / 'tmp-backup').mkdir()

        result = _scan_worktrees(wt_dir)
        assert 5 in result
        assert 7 in result
        assert 'tmp-backup' not in result

    def test_missing_dir_returns_empty(self, tmp_path):
        from dashboard.data.orchestrator import _scan_worktrees

        assert _scan_worktrees(tmp_path / 'nonexistent') == {}


class TestDiscoverOrchestratorsPerProject:
    """Tests for per-project task loading in discover_orchestrators."""

    def test_different_projects_get_own_tasks(self, tmp_path):
        """Two orchestrators in different projects each see their own task tree."""
        import json
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        # Project A
        proj_a = tmp_path / 'proj_a'
        (proj_a / '.taskmaster' / 'tasks').mkdir(parents=True)
        (proj_a / '.taskmaster' / 'tasks' / 'tasks.json').write_text(json.dumps({'tasks': [
            {'id': '1', 'title': 'A1', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {}},
            {'id': '2', 'title': 'A2', 'status': 'pending', 'priority': 'medium', 'dependencies': [], 'metadata': {}},
        ]}))

        # Project B
        proj_b = tmp_path / 'proj_b'
        (proj_b / '.taskmaster' / 'tasks').mkdir(parents=True)
        (proj_b / '.taskmaster' / 'tasks' / 'tasks.json').write_text(json.dumps({'tasks': [
            {'id': '10', 'title': 'B1', 'status': 'pending', 'priority': 'high', 'dependencies': [], 'metadata': {}},
        ]}))

        mock_procs = [
            {'pid': 1000, 'prd': str(proj_a / 'docs' / 'prd.md'), 'config_path': None, 'running': True, 'started': 'Mar18'},
            {'pid': 2000, 'prd': str(proj_b / 'docs' / 'prd.md'), 'config_path': None, 'running': True, 'started': 'Mar18'},
        ]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = discover_orchestrators(config)

        by_root = {e['project_root']: e for e in result}

        a_entry = by_root[str(proj_a)]
        assert len(a_entry['tasks']) == 2
        assert a_entry['summary']['total'] == 2
        assert a_entry['summary']['done'] == 1

        b_entry = by_root[str(proj_b)]
        assert len(b_entry['tasks']) == 1
        assert b_entry['summary']['total'] == 1
        assert b_entry['summary']['pending'] == 1

    def test_same_project_prds_merged_into_single_entry(self, tmp_path):
        """Two PRDs in the same project are merged into one entry (grouped by project root)."""
        import json
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        (tmp_path / '.taskmaster' / 'tasks').mkdir(parents=True)
        (tmp_path / '.taskmaster' / 'tasks' / 'tasks.json').write_text(json.dumps({'tasks': [
            {'id': '1', 'title': 'T1', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {}},
        ]}))

        mock_procs = [
            {'pid': 1000, 'prd': str(tmp_path / 'prd1.md'), 'config_path': None, 'running': True, 'started': 'Mar18'},
            {'pid': 2000, 'prd': str(tmp_path / 'prd2.md'), 'config_path': None, 'running': True, 'started': 'Mar18'},
        ]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = discover_orchestrators(config)

        # Both resolve to the same project root → merged into one entry
        assert len(result) == 1
        assert set(result[0]['pids']) == {1000, 2000}
        assert len(result[0]['tasks']) == 1

    def test_worktrees_loaded_per_project(self, tmp_path):
        """Each orchestrator sees worktrees from its own project."""
        import json
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        # Project A with worktree for task 3
        proj_a = tmp_path / 'proj_a'
        (proj_a / '.taskmaster' / 'tasks').mkdir(parents=True)
        (proj_a / '.taskmaster' / 'tasks' / 'tasks.json').write_text(json.dumps({'tasks': [
            {'id': '3', 'title': 'A-task', 'status': 'in-progress', 'priority': 'high', 'dependencies': [], 'metadata': {}},
        ]}))
        wt_a = proj_a / '.worktrees' / '3' / '.task'
        wt_a.mkdir(parents=True)
        (wt_a / 'plan.json').write_text(json.dumps({'steps': [{'id': 's1', 'status': 'done'}]}))

        # Project B with worktree for task 5
        proj_b = tmp_path / 'proj_b'
        (proj_b / '.taskmaster' / 'tasks').mkdir(parents=True)
        (proj_b / '.taskmaster' / 'tasks' / 'tasks.json').write_text(json.dumps({'tasks': [
            {'id': '5', 'title': 'B-task', 'status': 'in-progress', 'priority': 'high', 'dependencies': [], 'metadata': {}},
        ]}))
        wt_b = proj_b / '.worktrees' / '5' / '.task'
        wt_b.mkdir(parents=True)
        (wt_b / 'plan.json').write_text(json.dumps({'steps': [{'id': 's1', 'status': 'pending'}]}))

        mock_procs = [
            {'pid': 1000, 'prd': str(proj_a / 'prd.md'), 'config_path': None, 'running': True, 'started': 'Mar18'},
            {'pid': 2000, 'prd': str(proj_b / 'prd.md'), 'config_path': None, 'running': True, 'started': 'Mar18'},
        ]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = discover_orchestrators(config)

        by_root = {e['project_root']: e for e in result}
        assert set(by_root[str(proj_a)]['worktrees'].keys()) == {3}
        assert by_root[str(proj_a)]['worktrees'][3]['phase'] == 'DONE'
        assert set(by_root[str(proj_b)]['worktrees'].keys()) == {5}
        assert by_root[str(proj_b)]['worktrees'][5]['phase'] == 'EXECUTE'

    def test_fallback_to_config_project_root(self, tmp_path):
        """When PRD path has no .taskmaster/ ancestor, falls back to config project_root."""
        import json
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        (tmp_path / '.taskmaster' / 'tasks').mkdir(parents=True)
        (tmp_path / '.taskmaster' / 'tasks' / 'tasks.json').write_text(json.dumps({'tasks': [
            {'id': '1', 'title': 'T', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {}},
        ]}))

        mock_procs = [{'pid': 1000, 'prd': '/nonexistent/prd.md', 'config_path': None, 'running': True, 'started': 'Mar18'}]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = discover_orchestrators(config)

        assert len(result) == 1
        assert len(result[0]['tasks']) == 1

    def test_project_root_in_result_is_resolved_when_config_root_is_symlink(self, tmp_path):
        """project_root in result dict is canonicalised even when config.project_root is a symlink."""
        import json
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        real_dir = tmp_path / 'real'
        real_dir.mkdir()
        link = tmp_path / 'link'
        link.symlink_to(real_dir)

        # Create tasks.json under the real directory so discover_orchestrators has data to read
        (real_dir / '.taskmaster' / 'tasks').mkdir(parents=True)
        (real_dir / '.taskmaster' / 'tasks' / 'tasks.json').write_text(json.dumps({'tasks': [
            {'id': '1', 'title': 'T', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {}},
        ]}))

        config = DashboardConfig(project_root=link)

        # PRD under /nonexistent — no .taskmaster ancestor found, falls back to config.project_root (=link)
        mock_procs = [{'pid': 9999, 'prd': '/nonexistent/prd.md', 'config_path': None, 'running': True, 'started': 'Apr07'}]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = discover_orchestrators(config)

        assert len(result) == 1
        # project_root must be the resolved canonical path, not the symlink
        assert result[0]['project_root'] == str(real_dir)
