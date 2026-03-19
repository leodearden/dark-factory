"""Tests for dashboard.data.orchestrator — orchestrator discovery and status."""

from __future__ import annotations


class TestFindRunningOrchestrators:
    """Tests for find_running_orchestrators — scans ps aux for orchestrator processes."""

    def test_parses_orchestrator_lines(self):
        """Two orchestrator lines with --prd flags produce two dicts with pid, prd, running, started."""
        import subprocess
        from unittest.mock import patch

        from dashboard.data.orchestrator import find_running_orchestrators

        ps_output = (
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\n"
            "leo       1234  0.5  1.2 123456  7890 ?        Sl   Mar18   0:05 python -m orchestrator --prd /home/leo/prd1.md\n"
            "leo       5678  0.3  0.8 234567  4567 ?        Sl   10:30   0:02 python -m orchestrator --prd /home/leo/prd2.md\n"
        )
        mock_result = subprocess.CompletedProcess(args=['ps', 'aux'], returncode=0, stdout=ps_output, stderr='')

        with patch('dashboard.data.orchestrator.subprocess.run', return_value=mock_result):
            result = find_running_orchestrators()

        assert len(result) == 2
        assert result[0]['pid'] == 1234
        assert result[0]['prd'] == '/home/leo/prd1.md'
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
            "leo       1234  0.5  1.2 123456  7890 ?        Sl   Mar18   0:05 python -m orchestrator --prd /home/leo/prd1.md\n"
            "leo       9999  0.0  0.0  12345   678 pts/0    S+   10:31   0:00 grep orchestrator\n"
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
        assert result[0]['id'] == '1'
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
        assert result[1]['dependencies'] == ['1']

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

        # plan.json with 3 done, 2 pending out of 5 total
        steps = [
            {'id': f'step-{i}', 'status': 'done'} for i in range(1, 4)
        ] + [
            {'id': f'step-{i}', 'status': 'pending'} for i in range(4, 6)
        ]
        (task_dir / 'plan.json').write_text(json.dumps({'steps': steps}))

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


class TestDiscoverOrchestrators:
    """Tests for discover_orchestrators — combines process, task tree, and artifact data."""

    def test_combines_process_and_worktree_data(self, tmp_path):
        """Running orchestrator is enriched with task tree and worktree artifact data."""
        import json
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        # Create tasks.json with 5 tasks of varying statuses
        tasks_dir = tmp_path / '.taskmaster' / 'tasks'
        tasks_dir.mkdir(parents=True)
        tasks = [
            {'id': '1', 'title': 'Setup', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {}},
            {'id': '2', 'title': 'Build', 'status': 'done', 'priority': 'high', 'dependencies': ['1'], 'metadata': {}},
            {'id': '3', 'title': 'Test', 'status': 'in-progress', 'priority': 'medium', 'dependencies': ['2'], 'metadata': {}},
            {'id': '4', 'title': 'Review', 'status': 'blocked', 'priority': 'medium', 'dependencies': ['3'], 'metadata': {}},
            {'id': '5', 'title': 'Deploy', 'status': 'pending', 'priority': 'low', 'dependencies': ['4'], 'metadata': {}},
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

        # Mock one running orchestrator
        mock_procs = [{'pid': 1234, 'prd': '/home/leo/prd.md', 'running': True, 'started': 'Mar18'}]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = discover_orchestrators(config)

        assert len(result) == 1
        entry = result[0]
        assert entry['pid'] == 1234
        assert entry['prd'] == '/home/leo/prd.md'
        assert entry['running'] is True
        assert len(entry['tasks']) == 5
        assert '7' in entry['worktrees']
        assert entry['worktrees']['7']['phase'] == 'EXECUTE'
        assert entry['summary'] == {'total': 5, 'done': 2, 'in_progress': 1, 'blocked': 1, 'pending': 1}

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

        mock_procs = [{'pid': 5678, 'prd': '/prd.md', 'running': True, 'started': '10:30'}]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = discover_orchestrators(config)

        assert len(result) == 1
        assert result[0]['tasks'] == []
        assert result[0]['worktrees'] == {}
        assert result[0]['summary'] == {'total': 0, 'done': 0, 'in_progress': 0, 'blocked': 0, 'pending': 0}
