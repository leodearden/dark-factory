"""Tests for dashboard.data.orchestrator — orchestrator discovery and status."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _shape_task(task: dict) -> dict | None:
    """Coerce a raw test task dict into the dashboard's per-task wire shape."""
    raw_id = task.get('id')
    if raw_id is None:
        return None
    try:
        tid = int(raw_id)
    except (TypeError, ValueError):
        return None
    raw_deps = task.get('dependencies', []) or []
    deps: list[int] = []
    for d in raw_deps:
        try:
            deps.append(int(d))
        except (TypeError, ValueError):
            continue
    metadata = task.get('metadata', {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        'id': tid,
        'title': task.get('title') or '',
        'status': task.get('status'),
        'priority': task.get('priority'),
        'dependencies': deps,
        'metadata': metadata,
        'updated_at': task.get('updated_at'),
    }


@pytest.fixture()
def fake_fetch_tasks(monkeypatch):
    """Patch fetch_tasks to return shaped tasks for registered project roots."""
    registry: dict[Path, list[dict]] = {}

    async def _fake(client, config, project_root):
        return list(registry.get(Path(project_root).resolve(), []))

    monkeypatch.setattr('dashboard.data.orchestrator.fetch_tasks', _fake)

    def register(root: Path, tasks: list[dict]) -> None:
        shaped = [r for r in (_shape_task(t) for t in tasks) if r is not None]
        registry[Path(root).resolve()] = shaped

    return register


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
        """subprocess.run raising RuntimeError (non-subprocess error) propagates to caller."""
        from unittest.mock import patch

        from dashboard.data.orchestrator import find_running_orchestrators

        with pytest.raises(RuntimeError, match='unexpected'), patch(
            'dashboard.data.orchestrator.subprocess.run',
            side_effect=RuntimeError('unexpected'),
        ):
            find_running_orchestrators()

    def test_subprocess_timeout_caught(self):
        """subprocess.run raising TimeoutExpired is caught and returns empty list."""
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


class TestDiscoverOrchestrators:
    """Tests for discover_orchestrators — combines process, task tree (MCP), and artifact data."""

    async def test_combines_process_and_worktree_data(self, tmp_path, fake_fetch_tasks, dummy_client):
        """Running orchestrator is enriched with task tree and worktree artifact data."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)
        fake_fetch_tasks(tmp_path, [
            {'id': '1', 'title': 'Setup', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {}},
            {'id': '2', 'title': 'Build', 'status': 'done', 'priority': 'high', 'dependencies': ['1'], 'metadata': {}},
            {'id': '3', 'title': 'Test', 'status': 'in-progress', 'priority': 'medium', 'dependencies': ['2'], 'metadata': {}},
            {'id': '4', 'title': 'Review', 'status': 'blocked', 'priority': 'medium', 'dependencies': ['3'], 'metadata': {}},
            {'id': '5', 'title': 'Deploy', 'status': 'pending', 'priority': 'low', 'dependencies': ['4'], 'metadata': {}},
            {'id': '7', 'title': 'Widget', 'status': 'in-progress', 'priority': 'high', 'dependencies': [], 'metadata': {}},
        ])

        wt_dir = tmp_path / '.worktrees' / '7'
        wt_dir.mkdir(parents=True)
        task_dir = wt_dir / '.task'
        task_dir.mkdir()
        (task_dir / 'metadata.json').write_text(json.dumps({'task_id': '7', 'title': 'Widget'}))
        steps = [{'id': 'step-1', 'status': 'done'}, {'id': 'step-2', 'status': 'pending'}]
        (task_dir / 'plan.json').write_text(json.dumps({'steps': steps}))

        prd_path = str(tmp_path / 'prd.md')
        mock_procs = [{'pid': 1234, 'prd': prd_path, 'config_path': None, 'running': True, 'started': 'Mar18'}]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = await discover_orchestrators(client=dummy_client, config=config)

        assert len(result) == 1
        entry = result[0]
        assert 1234 in entry['pids']
        assert entry['prd'] == prd_path
        assert entry['running'] is True
        assert len(entry['tasks']) == 6
        assert entry['summary'] == {'total': 6, 'done': 2, 'in_progress': 2, 'blocked': 1, 'pending': 1}

    async def test_no_running_orchestrators(self, tmp_path, fake_fetch_tasks, dummy_client):
        """Empty process list returns empty result."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=[]):
            result = await discover_orchestrators(client=dummy_client, config=config)

        assert result == []

    async def test_missing_task_tree(self, tmp_path, fake_fetch_tasks, dummy_client):
        """Orchestrator returned even when MCP task list is empty."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)
        # registry is empty for this project — fetch_tasks returns []

        mock_procs = [{'pid': 5678, 'prd': str(tmp_path / 'prd.md'), 'config_path': None, 'running': True, 'started': '10:30'}]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = await discover_orchestrators(client=dummy_client, config=config)

        assert len(result) == 1
        assert result[0]['tasks'] == []
        assert result[0]['summary'] == {'total': 0, 'done': 0, 'in_progress': 0, 'blocked': 0, 'pending': 0}

    async def test_single_process_produces_pids_list(self, tmp_path, fake_fetch_tasks, dummy_client):
        """Single running orchestrator produces entry with 'pids' list and no 'pid' key."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        mock_procs = [{'pid': 1234, 'prd': '/home/leo/prd.md', 'config_path': None, 'running': True, 'started': 'Mar18'}]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = await discover_orchestrators(client=dummy_client, config=config)

        assert len(result) == 1
        entry = result[0]
        assert 'pids' in entry
        assert isinstance(entry['pids'], list)
        assert entry['pids'] == [1234]
        assert 'pid' not in entry

    async def test_same_prd_grouped_into_single_entry(self, tmp_path, fake_fetch_tasks, dummy_client):
        """Two processes with the same PRD path are merged into one entry with both PIDs."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)
        fake_fetch_tasks(tmp_path, [
            {'id': '1', 'title': 'Setup', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {}},
        ])

        wt_dir = tmp_path / '.worktrees' / '1'
        wt_dir.mkdir(parents=True)

        prd_path = str(tmp_path / 'prd.md')
        mock_procs = [
            {'pid': 1234, 'prd': prd_path, 'config_path': None, 'running': True, 'started': 'Mar18'},
            {'pid': 5678, 'prd': prd_path, 'config_path': None, 'running': False, 'started': 'Mar18'},
        ]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = await discover_orchestrators(client=dummy_client, config=config)

        assert len(result) == 1
        entry = result[0]
        assert entry['pids'] == [1234, 5678]
        assert len(entry['tasks']) == 1
        assert entry['summary']['total'] == 1

    async def test_different_projects_produce_separate_entries(self, tmp_path, fake_fetch_tasks, dummy_client):
        """Two processes targeting different project roots produce two separate entries."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        proj_a = tmp_path / 'proj_a'
        (proj_a / '.taskmaster').mkdir(parents=True)
        proj_b = tmp_path / 'proj_b'
        (proj_b / '.taskmaster').mkdir(parents=True)

        mock_procs = [
            {'pid': 1234, 'prd': str(proj_a / 'prd.md'), 'config_path': None, 'running': True, 'started': 'Mar18'},
            {'pid': 5678, 'prd': str(proj_b / 'prd.md'), 'config_path': None, 'running': True, 'started': 'Mar18'},
        ]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = await discover_orchestrators(client=dummy_client, config=config)

        assert len(result) == 2
        pids_by_root = {entry['project_root']: entry['pids'] for entry in result}
        assert pids_by_root[str(proj_a)] == [1234]
        assert pids_by_root[str(proj_b)] == [5678]

    async def test_grouped_running_true_when_any_running(self, tmp_path, fake_fetch_tasks, dummy_client):
        """Grouped entry has running=True if at least one process is still running."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        mock_procs = [
            {'pid': 1234, 'prd': '/prd.md', 'config_path': None, 'running': True, 'started': 'Mar18'},
            {'pid': 5678, 'prd': '/prd.md', 'config_path': None, 'running': False, 'started': 'Mar17'},
        ]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = await discover_orchestrators(client=dummy_client, config=config)

        assert len(result) == 1
        assert result[0]['running'] is True

    async def test_grouped_running_false_when_all_completed(self, tmp_path, fake_fetch_tasks, dummy_client):
        """Grouped entry has running=False if all processes in the group have completed."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        mock_procs = [
            {'pid': 1234, 'prd': '/prd.md', 'config_path': None, 'running': False, 'started': 'Mar18'},
            {'pid': 5678, 'prd': '/prd.md', 'config_path': None, 'running': False, 'started': 'Mar17'},
        ]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = await discover_orchestrators(client=dummy_client, config=config)

        assert len(result) == 1
        assert result[0]['running'] is False

    async def test_bare_fallback_with_symlink_config_root(self, tmp_path, fake_fetch_tasks, dummy_client):
        """Bare process (no prd, no config_path) with symlinked config.project_root returns canonical path."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real_dir)

        fake_fetch_tasks(real_dir, [
            {"id": "1", "title": "T", "status": "done", "priority": "high", "dependencies": [], "metadata": {}},
        ])

        config = DashboardConfig(project_root=link)

        mock_procs = [{"pid": 1234, "prd": None, "config_path": None, "running": True, "started": "Apr09"}]
        with patch("dashboard.data.orchestrator.find_running_orchestrators", return_value=mock_procs):
            result = await discover_orchestrators(client=dummy_client, config=config)

        assert len(result) == 1
        assert result[0]["project_root"] == str(real_dir)
        assert result[0]["pids"] == [1234]

    async def test_multi_bare_processes_grouped_under_project_root(self, tmp_path, fake_fetch_tasks, dummy_client):
        """Multiple bare processes sharing the same config root are merged."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        fake_fetch_tasks(tmp_path, [
            {"id": "1", "title": "Alpha", "status": "done", "priority": "high", "dependencies": [], "metadata": {}},
            {"id": "2", "title": "Beta", "status": "in-progress", "priority": "medium", "dependencies": [], "metadata": {}},
        ])
        config = DashboardConfig(project_root=tmp_path)

        mock_procs = [
            {"pid": 1001, "prd": None, "config_path": None, "running": True, "started": "Apr09"},
            {"pid": 1002, "prd": None, "config_path": None, "running": True, "started": "Apr09"},
            {"pid": 1003, "prd": None, "config_path": None, "running": False, "started": "Apr09"},
        ]
        with patch("dashboard.data.orchestrator.find_running_orchestrators", return_value=mock_procs):
            result = await discover_orchestrators(client=dummy_client, config=config)

        assert len(result) == 1
        assert result[0]["pids"] == [1001, 1002, 1003]
        assert result[0]["project_root"] == str(tmp_path.resolve())
        assert result[0]["prd"] is None
        assert result[0]["label"] == str(tmp_path.resolve())
        assert result[0]["running"] is True
        assert result[0]["summary"] == {"total": 2, "done": 1, "in_progress": 1, "blocked": 0, "pending": 0}

    async def test_symlink_and_canonical_paths_grouped_into_single_entry(self, tmp_path, fake_fetch_tasks, dummy_client):
        """Two processes whose PRDs resolve to the same project root are merged into one entry."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        real_dir = tmp_path / "real_proj"
        real_dir.mkdir()
        (real_dir / '.taskmaster').mkdir()
        link_dir = tmp_path / "link_proj"
        link_dir.symlink_to(real_dir)

        fake_fetch_tasks(real_dir, [
            {"id": "2", "title": "Work", "status": "in-progress", "priority": "high", "dependencies": [], "metadata": {}},
        ])

        (tmp_path / "unrelated").mkdir()
        config = DashboardConfig(project_root=tmp_path / "unrelated")

        prd_via_symlink = str(link_dir / "docs" / "prd.md")
        prd_canonical = str(real_dir / "docs" / "prd.md")

        mock_procs = [
            {"pid": 111, "prd": prd_via_symlink, "config_path": None, "running": True, "started": "Apr09"},
            {"pid": 222, "prd": prd_canonical, "config_path": None, "running": True, "started": "Apr09"},
        ]
        with patch("dashboard.data.orchestrator.find_running_orchestrators", return_value=mock_procs):
            result = await discover_orchestrators(client=dummy_client, config=config)

        assert len(result) == 1
        assert set(result[0]["pids"]) == {111, 222}
        assert result[0]["project_root"] == str(real_dir)


class TestResolveProjectRoot:
    """Tests for _resolve_project_root — finds project root from PRD path."""

    def test_finds_taskmaster_dir(self, tmp_path):
        from dashboard.data.orchestrator import _resolve_project_root

        (tmp_path / '.taskmaster').mkdir()
        prd = str(tmp_path / 'docs' / 'prd.md')
        assert _resolve_project_root(prd, Path('/fallback')) == tmp_path

    def test_relative_prd_resolved_against_default(self, tmp_path):
        from dashboard.data.orchestrator import _resolve_project_root

        (tmp_path / '.taskmaster').mkdir()
        assert _resolve_project_root('docs/prd.md', tmp_path) == tmp_path

    def test_falls_back_to_default(self, tmp_path):
        from dashboard.data.orchestrator import _resolve_project_root

        default = tmp_path / 'default'
        default.mkdir()
        assert _resolve_project_root('/nowhere/prd.md', default) == default

    def test_dotdot_in_path(self, tmp_path):
        from dashboard.data.orchestrator import _resolve_project_root

        (tmp_path / '.taskmaster').mkdir()
        prd = str(tmp_path / 'docs' / '..' / 'docs' / 'prd.md')
        assert _resolve_project_root(prd, Path('/fallback')) == tmp_path

    def test_fallback_returns_resolved_default_root(self, tmp_path):
        from dashboard.data.orchestrator import _resolve_project_root

        real = tmp_path / 'real'
        real.mkdir()
        link = tmp_path / 'link'
        link.symlink_to(real)

        result = _resolve_project_root('/nowhere/prd.md', link)

        assert result == real
        assert result != link

    def test_symlink_component_in_prd_path_returns_canonical_ancestor(self, tmp_path):
        from dashboard.data.orchestrator import _resolve_project_root

        real_proj = tmp_path / "real_proj"
        real_proj.mkdir()
        (real_proj / ".taskmaster").mkdir()

        link = tmp_path / "link"
        link.symlink_to(real_proj)

        prd = str(link / "docs" / "prd.md")

        result = _resolve_project_root(prd, Path("/fallback"))

        assert result == real_proj
        assert result != link


class TestDiscoverOrchestratorsPerProject:
    """Tests for per-project task loading in discover_orchestrators."""

    async def test_different_projects_get_own_tasks(self, tmp_path, fake_fetch_tasks, dummy_client):
        """Two orchestrators in different projects each see their own task tree."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        proj_a = tmp_path / 'proj_a'
        proj_a.mkdir()
        (proj_a / '.taskmaster').mkdir()
        fake_fetch_tasks(proj_a, [
            {'id': '1', 'title': 'A1', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {}},
            {'id': '2', 'title': 'A2', 'status': 'pending', 'priority': 'medium', 'dependencies': [], 'metadata': {}},
        ])

        proj_b = tmp_path / 'proj_b'
        proj_b.mkdir()
        (proj_b / '.taskmaster').mkdir()
        fake_fetch_tasks(proj_b, [
            {'id': '10', 'title': 'B1', 'status': 'pending', 'priority': 'high', 'dependencies': [], 'metadata': {}},
        ])

        mock_procs = [
            {'pid': 1000, 'prd': str(proj_a / 'docs' / 'prd.md'), 'config_path': None, 'running': True, 'started': 'Mar18'},
            {'pid': 2000, 'prd': str(proj_b / 'docs' / 'prd.md'), 'config_path': None, 'running': True, 'started': 'Mar18'},
        ]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = await discover_orchestrators(client=dummy_client, config=config)

        by_root = {e['project_root']: e for e in result}

        a_entry = by_root[str(proj_a)]
        assert len(a_entry['tasks']) == 2
        assert a_entry['summary']['total'] == 2
        assert a_entry['summary']['done'] == 1

        b_entry = by_root[str(proj_b)]
        assert len(b_entry['tasks']) == 1
        assert b_entry['summary']['total'] == 1
        assert b_entry['summary']['pending'] == 1

    async def test_same_project_prds_merged_into_single_entry(self, tmp_path, fake_fetch_tasks, dummy_client):
        """Two PRDs in the same project are merged into one entry (grouped by project root)."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        fake_fetch_tasks(tmp_path, [
            {'id': '1', 'title': 'T1', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {}},
        ])

        mock_procs = [
            {'pid': 1000, 'prd': str(tmp_path / 'prd1.md'), 'config_path': None, 'running': True, 'started': 'Mar18'},
            {'pid': 2000, 'prd': str(tmp_path / 'prd2.md'), 'config_path': None, 'running': True, 'started': 'Mar18'},
        ]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = await discover_orchestrators(client=dummy_client, config=config)

        assert len(result) == 1
        assert set(result[0]['pids']) == {1000, 2000}
        assert len(result[0]['tasks']) == 1

    async def test_fallback_to_config_project_root(self, tmp_path, fake_fetch_tasks, dummy_client):
        """When PRD path has no .taskmaster/ ancestor, falls back to config project_root."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)

        fake_fetch_tasks(tmp_path, [
            {'id': '1', 'title': 'T', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {}},
        ])

        mock_procs = [{'pid': 1000, 'prd': '/nonexistent/prd.md', 'config_path': None, 'running': True, 'started': 'Mar18'}]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = await discover_orchestrators(client=dummy_client, config=config)

        assert len(result) == 1
        assert len(result[0]['tasks']) == 1

    async def test_project_root_in_result_is_resolved_when_config_root_is_symlink(self, tmp_path, fake_fetch_tasks, dummy_client):
        """project_root in result dict is canonicalised even when config.project_root is a symlink."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        real_dir = tmp_path / 'real'
        real_dir.mkdir()
        link = tmp_path / 'link'
        link.symlink_to(real_dir)

        fake_fetch_tasks(real_dir, [
            {'id': '1', 'title': 'T', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {}},
        ])

        config = DashboardConfig(project_root=link)

        mock_procs = [{'pid': 9999, 'prd': '/nonexistent/prd.md', 'config_path': None, 'running': True, 'started': 'Apr07'}]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = await discover_orchestrators(client=dummy_client, config=config)

        assert len(result) == 1
        assert result[0]['project_root'] == str(real_dir)

    async def test_last_update_is_max_updated_at_when_all_tasks_dated(self, tmp_path, fake_fetch_tasks, dummy_client):
        """last_update equals the most-recent updated_at when all tasks carry the field."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)
        fake_fetch_tasks(tmp_path, [
            {'id': '1', 'title': 'A', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {},
             'updated_at': '2026-06-10T10:00:00'},
            {'id': '2', 'title': 'B', 'status': 'in-progress', 'priority': 'high', 'dependencies': [], 'metadata': {},
             'updated_at': '2026-06-12T15:30:00'},
            {'id': '3', 'title': 'C', 'status': 'pending', 'priority': 'medium', 'dependencies': [], 'metadata': {},
             'updated_at': '2026-06-11T08:00:00'},
        ])

        mock_procs = [{'pid': 1234, 'prd': str(tmp_path / 'prd.md'), 'config_path': None, 'running': True, 'started': 'Mar18'}]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = await discover_orchestrators(client=dummy_client, config=config)

        assert len(result) == 1
        assert result[0]['last_update'] == '2026-06-12T15:30:00'

    async def test_last_update_is_none_when_no_task_has_updated_at(self, tmp_path, fake_fetch_tasks, dummy_client):
        """last_update is None when no task in the tree carries an updated_at value."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)
        fake_fetch_tasks(tmp_path, [
            {'id': '1', 'title': 'A', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {}},
            {'id': '2', 'title': 'B', 'status': 'pending', 'priority': 'medium', 'dependencies': [], 'metadata': {}},
        ])

        mock_procs = [{'pid': 5678, 'prd': str(tmp_path / 'prd.md'), 'config_path': None, 'running': True, 'started': '10:30'}]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = await discover_orchestrators(client=dummy_client, config=config)

        assert len(result) == 1
        assert result[0]['last_update'] is None

    async def test_last_update_ignores_tasks_without_updated_at(self, tmp_path, fake_fetch_tasks, dummy_client):
        """last_update equals max over dated tasks; tasks without updated_at are skipped."""
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)
        fake_fetch_tasks(tmp_path, [
            {'id': '1', 'title': 'A', 'status': 'done', 'priority': 'high', 'dependencies': [], 'metadata': {},
             'updated_at': '2026-06-09T09:00:00'},
            {'id': '2', 'title': 'B', 'status': 'pending', 'priority': 'medium', 'dependencies': [], 'metadata': {}},
            {'id': '3', 'title': 'C', 'status': 'in-progress', 'priority': 'high', 'dependencies': [], 'metadata': {},
             'updated_at': '2026-06-14T20:00:00'},
        ])

        mock_procs = [{'pid': 9001, 'prd': str(tmp_path / 'prd.md'), 'config_path': None, 'running': True, 'started': 'Apr01'}]
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = await discover_orchestrators(client=dummy_client, config=config)

        assert len(result) == 1
        assert result[0]['last_update'] == '2026-06-14T20:00:00'


class TestDiscoverOrchestratorsOfflineMarker:
    """Tests for discover_orchestrators propagating the MCP offline marker."""

    async def test_offline_marker_preserved_not_discarded(self, tmp_path, monkeypatch, dummy_client):
        """When fetch_tasks returns the offline marker, the project entry must still appear
        with offline=True and error from the marker, not be silently discarded.

        Fails today because `fetched if isinstance(fetched, list) else []` discards the marker.
        """
        from unittest.mock import patch

        from dashboard.config import DashboardConfig
        from dashboard.data.orchestrator import discover_orchestrators

        config = DashboardConfig(project_root=tmp_path)
        prd_path = str(tmp_path / 'prd.md')
        mock_procs = [{'pid': 9999, 'prd': prd_path, 'config_path': None, 'running': True, 'started': 'Mar18'}]

        async def offline_fetch(client, cfg, project_root):
            return {'offline': True, 'error': 'boom'}

        monkeypatch.setattr('dashboard.data.orchestrator.fetch_tasks', offline_fetch)
        with patch('dashboard.data.orchestrator.find_running_orchestrators', return_value=mock_procs):
            result = await discover_orchestrators(client=dummy_client, config=config)

        assert len(result) == 1, 'offline-marker entry must NOT be dropped from the result'
        entry = result[0]
        assert entry.get('offline') is True, f'expected offline=True, got: {entry}'
        assert entry.get('error') == 'boom', f'expected error=boom, got: {entry}'
        # Summary should be all-zero (no tasks)
        s = entry.get('summary', {})
        assert s.get('total', -1) == 0


class TestReadMaxConcurrentTasks:
    """Tests for ``read_max_concurrent_tasks`` — the parity alarm's denominator.

    ``max_concurrent_tasks`` is a top-level key of the orchestrator config and
    is restart-only (red-tier per CLAUDE.md: absent from ``config.py``'s
    hot-reload allowlist, and the scheduler semaphore is sized once at
    startup).  It is still TIME-VARYING across a burndown window, because such
    a window spans restarts and the cap also differs between projects.  The
    collector therefore reads it once per snapshot so each historical row is
    paired with the cap that was in force at THAT instant; comparing a past
    in-progress census against today's cap would mislabel both directions.

    The contract mirrors this module's ``_read_project_root_from_config``
    sibling: ``yaml.safe_load``, never raise, return ``None`` for anything
    unexpected.  ``None`` means "unknown", which the read side must never
    conflate with "not breaching" — and is reserved STRICTLY for an absent,
    unreadable or malformed config.  A readable config that merely OMITS the
    key is NOT unknown: ``orchestrator.config`` deep-merges ``defaults.yaml``
    under every project config, so that project runs under the orchestrator's
    own default.  See the defaults-layering section below.
    """

    LOGGER = 'dashboard.data.orchestrator'

    @staticmethod
    def _write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    # ---- happy paths -------------------------------------------------

    def test_canonical_config_yields_cap(self, tmp_path):
        """The canonical CLAUDE.md filename is what discovery keys on first."""
        from dashboard.data.orchestrator import read_max_concurrent_tasks

        self._write(tmp_path / 'dark-factory-orchestrator.yaml', 'max_concurrent_tasks: 24\n')

        assert read_max_concurrent_tasks(tmp_path) == 24

    def test_accepts_str_project_root(self, tmp_path):
        """Callers hold roots as Path, but a str must not blow up."""
        from dashboard.data.orchestrator import read_max_concurrent_tasks

        self._write(tmp_path / 'dark-factory-orchestrator.yaml', 'max_concurrent_tasks: 8\n')

        assert read_max_concurrent_tasks(str(tmp_path)) == 8

    def test_ignores_unrelated_top_level_keys(self, tmp_path):
        """A real config carries many keys; only max_concurrent_tasks is read."""
        from dashboard.data.orchestrator import read_max_concurrent_tasks

        self._write(
            tmp_path / 'dark-factory-orchestrator.yaml',
            'project_root: /home/leo/src/dark-factory\n'
            'max_concurrent_tasks: 24\n'
            'escalation:\n'
            '  port: 9101\n',
        )

        assert read_max_concurrent_tasks(tmp_path) == 24

    def test_zero_cap_is_a_real_value_not_unknown(self, tmp_path):
        """A cap of 0 ("dispatch nothing") is a legal config and is preserved.

        Deliberate boundary: 0 is NOT coerced to None.  With tasks still
        in-progress against a 0 cap the parity alarm SHOULD fire, and
        returning None would report that genuine breach as "unknown".
        Only *negative* caps are nonsense and rejected (see below).
        """
        from dashboard.data.orchestrator import read_max_concurrent_tasks

        self._write(tmp_path / 'dark-factory-orchestrator.yaml', 'max_concurrent_tasks: 0\n')

        assert read_max_concurrent_tasks(tmp_path) == 0

    # ---- legacy-spelling precedence ----------------------------------

    @pytest.mark.parametrize(
        'legacy_name',
        ['orchestrator.yaml', 'orchestrator-config.yaml', 'orchestrator/config.yaml'],
    )
    def test_legacy_spellings_honoured_as_fallback(self, tmp_path, legacy_name):
        """Each legacy spelling from config._LEGACY_CONFIG_NAMES still resolves."""
        from dashboard.data.orchestrator import read_max_concurrent_tasks

        self._write(tmp_path / legacy_name, 'max_concurrent_tasks: 12\n')

        assert read_max_concurrent_tasks(tmp_path) == 12

    def test_legacy_precedence_order_is_first_match_wins(self, tmp_path):
        """The documented order is orchestrator.yaml, then -config, then subdir."""
        from dashboard.data.orchestrator import read_max_concurrent_tasks

        self._write(tmp_path / 'orchestrator.yaml', 'max_concurrent_tasks: 11\n')
        self._write(tmp_path / 'orchestrator-config.yaml', 'max_concurrent_tasks: 22\n')
        self._write(tmp_path / 'orchestrator/config.yaml', 'max_concurrent_tasks: 33\n')

        assert read_max_concurrent_tasks(tmp_path) == 11

    def test_canonical_beats_legacy(self, tmp_path):
        from dashboard.data.orchestrator import read_max_concurrent_tasks

        self._write(tmp_path / 'dark-factory-orchestrator.yaml', 'max_concurrent_tasks: 24\n')
        self._write(tmp_path / 'orchestrator.yaml', 'max_concurrent_tasks: 99\n')

        assert read_max_concurrent_tasks(tmp_path) == 24

    def test_present_canonical_is_authoritative_even_without_the_key(self, tmp_path):
        """A canonical config that omits the key must NOT be masked by a legacy file.

        Same rule config._discover_root_escalation_url documents: once the
        canonical file exists on disk it is authoritative, so a stale legacy
        spelling can never silently supply a cap the live config dropped.  The
        omitted key resolves to the orchestrator's own default — never to the
        legacy file's 99.
        """
        from dashboard.data.orchestrator import (
            _ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS,
            read_max_concurrent_tasks,
        )

        self._write(tmp_path / 'dark-factory-orchestrator.yaml', 'project_root: /somewhere\n')
        self._write(tmp_path / 'orchestrator.yaml', 'max_concurrent_tasks: 99\n')

        cap = read_max_concurrent_tasks(tmp_path)

        assert cap == _ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS
        assert cap != 99, 'a stale legacy file must never supply the live config’s cap'

    # ---- defaults.yaml layering --------------------------------------

    def test_omitted_key_yields_the_orchestrator_default_not_unknown(self, tmp_path):
        """The load-bearing case: an omitted key is a REAL cap, not "unknown".

        ``orchestrator.config`` builds its effective config as
        ``_deep_merge(_load_defaults(), project_config)``, and defaults.yaml
        sets ``max_concurrent_tasks``.  A project whose YAML omits the key is
        therefore running under that cap — two of the live configs under
        /home/leo/src do exactly this.  Reporting them as capless would drop
        them out of the parity alarm entirely and reproduce the very E12
        silent miss this feature exists to eliminate.
        """
        from dashboard.data.orchestrator import (
            _ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS,
            read_max_concurrent_tasks,
        )

        self._write(
            tmp_path / 'dark-factory-orchestrator.yaml',
            'project_root: /somewhere\nescalation:\n  port: 9101\n',
        )

        cap = read_max_concurrent_tasks(tmp_path)

        assert cap == _ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS
        assert cap is not None, 'an omitted key is not unknown — the default is in force'

    def test_omitted_key_in_a_legacy_config_also_yields_the_default(self, tmp_path):
        """Defaults layering is a property of the orchestrator, not of a filename."""
        from dashboard.data.orchestrator import (
            _ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS,
            read_max_concurrent_tasks,
        )

        self._write(tmp_path / 'orchestrator.yaml', 'project_root: /somewhere\n')

        assert read_max_concurrent_tasks(tmp_path) == _ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS

    def test_explicit_null_is_malformed_not_absent(self, tmp_path, caplog):
        """``max_concurrent_tasks:`` with no value is a config DEFECT, not an omission.

        ``OrchestratorConfig`` types the field ``int``, so an explicit null
        fails validation and no orchestrator runs from that config at all.
        Silently substituting the default would paper over a file no
        orchestrator can load, so this stays unknown AND is logged.
        """
        import logging

        from dashboard.data.orchestrator import read_max_concurrent_tasks

        self._write(tmp_path / 'dark-factory-orchestrator.yaml', 'max_concurrent_tasks:\n')

        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            assert read_max_concurrent_tasks(tmp_path) is None
        assert [r for r in caplog.records if r.levelno == logging.WARNING], (
            'a config no orchestrator can load must be logged, not silently defaulted'
        )

    def test_restated_default_matches_orchestrator_defaults_yaml(self):
        """FORMAT COUPLING guard: the restated constant must not drift.

        This module deliberately does not import the ``orchestrator`` package
        (see its FORMAT COUPLING note), so the default is restated by hand.
        Whenever the orchestrator source IS present next to the dashboard,
        assert the two agree — a defaults.yaml edit then fails a test instead
        of silently mis-sizing every parity denominator.  Skipped when the
        source is not on disk (installed-package / partial-checkout layouts).
        """
        import yaml

        from dashboard.data.orchestrator import _ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS

        defaults = (
            Path(__file__).resolve().parents[2]
            / 'orchestrator' / 'src' / 'orchestrator' / 'defaults.yaml'
        )
        if not defaults.is_file():
            pytest.skip(f'orchestrator source not present at {defaults}')

        upstream = (yaml.safe_load(defaults.read_text()) or {}).get('max_concurrent_tasks')

        assert upstream == _ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS, (
            f'{defaults} sets max_concurrent_tasks={upstream!r}, but '
            f'dashboard.data.orchestrator restates '
            f'{_ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS!r} — update the constant '
            f'(FORMAT COUPLING item 2)'
        )

    # ---- ${VAR:default} expansion ------------------------------------

    def test_env_var_default_expanded_before_int_coercion(self, tmp_path, monkeypatch):
        from dashboard.data.orchestrator import read_max_concurrent_tasks

        monkeypatch.delenv('DF_TEST_MAX_TASKS', raising=False)
        self._write(
            tmp_path / 'dark-factory-orchestrator.yaml',
            'max_concurrent_tasks: "${DF_TEST_MAX_TASKS:24}"\n',
        )

        assert read_max_concurrent_tasks(tmp_path) == 24

    def test_env_var_value_overrides_default(self, tmp_path, monkeypatch):
        from dashboard.data.orchestrator import read_max_concurrent_tasks

        monkeypatch.setenv('DF_TEST_MAX_TASKS', '6')
        self._write(
            tmp_path / 'dark-factory-orchestrator.yaml',
            'max_concurrent_tasks: "${DF_TEST_MAX_TASKS:24}"\n',
        )

        assert read_max_concurrent_tasks(tmp_path) == 6

    def test_unset_env_var_with_no_default_is_unknown(self, tmp_path, monkeypatch, caplog):
        """``${VAR}`` with VAR unset expands to '' — unknown, not 0."""
        import logging

        from dashboard.data.orchestrator import read_max_concurrent_tasks

        monkeypatch.delenv('DF_TEST_MAX_TASKS', raising=False)
        self._write(
            tmp_path / 'dark-factory-orchestrator.yaml',
            'max_concurrent_tasks: "${DF_TEST_MAX_TASKS}"\n',
        )

        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            assert read_max_concurrent_tasks(tmp_path) is None
        assert [r for r in caplog.records if r.levelno == logging.WARNING], (
            'an unusable cap value must be logged, not dropped silently'
        )

    # ---- unknown / unusable -> None ----------------------------------

    def test_missing_config_is_none(self, tmp_path):
        from dashboard.data.orchestrator import read_max_concurrent_tasks

        assert read_max_concurrent_tasks(tmp_path) is None

    def test_missing_project_root_is_none(self, tmp_path):
        from dashboard.data.orchestrator import read_max_concurrent_tasks

        assert read_max_concurrent_tasks(tmp_path / 'no-such-root') is None

    def test_unparseable_yaml_is_none(self, tmp_path):
        from dashboard.data.orchestrator import read_max_concurrent_tasks

        self._write(tmp_path / 'dark-factory-orchestrator.yaml', 'max_concurrent_tasks: [unclosed\n')

        assert read_max_concurrent_tasks(tmp_path) is None

    def test_non_dict_top_level_is_none(self, tmp_path):
        from dashboard.data.orchestrator import read_max_concurrent_tasks

        self._write(tmp_path / 'dark-factory-orchestrator.yaml', '- a\n- b\n')

        assert read_max_concurrent_tasks(tmp_path) is None

    def test_empty_file_is_none(self, tmp_path):
        from dashboard.data.orchestrator import read_max_concurrent_tasks

        self._write(tmp_path / 'dark-factory-orchestrator.yaml', '')

        assert read_max_concurrent_tasks(tmp_path) is None

    # An ABSENT key is deliberately NOT in this "unknown" section: it resolves
    # to the orchestrator's default, pinned by
    # test_omitted_key_yields_the_orchestrator_default_not_unknown above.  An
    # explicit YAML null IS unknown, pinned with its WARNING by
    # test_explicit_null_is_malformed_not_absent above.

    @pytest.mark.parametrize(
        ('label', 'yaml_value'),
        [
            ('non-numeric string', 'lots'),
            ('float', '2.5'),
            ('list', '[1, 2]'),
            ('mapping', '{a: 1}'),
            ('negative', '-1'),
            ('bool true', 'true'),
            ('bool false', 'false'),
        ],
    )
    def test_unusable_value_is_none_with_warning(self, tmp_path, caplog, label, yaml_value):
        """A malformed cap is 'unknown' + a WARNING — never a silent 0 or True.

        ``bool`` matters specifically: it is an ``int`` subclass, so a bare
        ``isinstance(value, int)`` check would let ``true`` through as a
        cap of 1 and alarm on every snapshot.
        """
        import logging

        from dashboard.data.orchestrator import read_max_concurrent_tasks

        self._write(
            tmp_path / 'dark-factory-orchestrator.yaml',
            f'max_concurrent_tasks: {yaml_value}\n',
        )

        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            result = read_max_concurrent_tasks(tmp_path)

        assert result is None, f'{label} must not be accepted as a cap (got {result!r})'
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, f'{label} must emit a WARNING naming the bad value'

    def test_numeric_string_is_accepted(self, tmp_path):
        """A quoted YAML scalar is how ``${VAR:default}`` configs spell an int."""
        from dashboard.data.orchestrator import read_max_concurrent_tasks

        self._write(tmp_path / 'dark-factory-orchestrator.yaml', "max_concurrent_tasks: '24'\n")

        assert read_max_concurrent_tasks(tmp_path) == 24

    # ---- never raises ------------------------------------------------

    def test_never_raises_for_any_hostile_input(self, tmp_path):
        """Sweep: the reader is called from the collector loop and must not raise.

        A raise here would take down a burndown collection cycle for every
        project, which is strictly worse than an unknown cap.
        """
        from dashboard.data.orchestrator import read_max_concurrent_tasks

        cases = [
            'max_concurrent_tasks: [unclosed\n',
            '\t\tbad indent\n',
            '- not\n- a mapping\n',
            'max_concurrent_tasks: !!python/object:os.system {}\n',
            'max_concurrent_tasks: 999999999999999999999999\n',
            '\x00\x01binary garbage\n',
        ]
        for text in cases:
            root = tmp_path / f'case{cases.index(text)}'
            self._write(root / 'dark-factory-orchestrator.yaml', text)
            result = read_max_concurrent_tasks(root)
            assert result is None or isinstance(result, int)

    def test_config_path_is_a_directory_does_not_raise(self, tmp_path):
        """``dark-factory-orchestrator.yaml`` existing as a DIRECTORY is not a crash."""
        from dashboard.data.orchestrator import read_max_concurrent_tasks

        (tmp_path / 'dark-factory-orchestrator.yaml').mkdir()
        self._write(tmp_path / 'orchestrator.yaml', 'max_concurrent_tasks: 7\n')

        # The canonical name is not a readable FILE, so resolution falls through
        # to the legacy spelling rather than exploding on IsADirectoryError.
        assert read_max_concurrent_tasks(tmp_path) == 7
