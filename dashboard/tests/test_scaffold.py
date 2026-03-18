"""Tests for dashboard scaffold: config, app, and fixtures."""

from pathlib import Path


class TestConfigDefaults:
    def test_config_defaults(self):
        from dashboard.config import DashboardConfig

        cfg = DashboardConfig()
        assert cfg.host == '127.0.0.1'
        assert cfg.port == 8080
        assert cfg.project_root == Path('/home/leo/src/dark-factory')
        assert cfg.fused_memory_url == 'http://localhost:8000'
        assert cfg.fused_memory_project_id == 'dark_factory'

    def test_config_derived_paths(self):
        from dashboard.config import DashboardConfig

        cfg = DashboardConfig()
        root = cfg.project_root
        assert cfg.reconciliation_db == root / 'fused-memory' / 'data' / 'reconciliation' / 'reconciliation.db'
        assert cfg.write_queue_db == root / 'fused-memory' / 'data' / 'queue' / 'write_queue.db'
        assert cfg.tasks_json == root / '.taskmaster' / 'tasks' / 'tasks.json'
        assert cfg.worktrees_dir == root / '.worktrees'
