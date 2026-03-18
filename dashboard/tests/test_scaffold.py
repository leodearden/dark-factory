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


class TestConfigEnvOverrides:
    def test_env_overrides(self, monkeypatch):
        from dashboard.config import DashboardConfig

        monkeypatch.setenv('DASHBOARD_HOST', '0.0.0.0')
        monkeypatch.setenv('DASHBOARD_PORT', '9090')
        monkeypatch.setenv('DASHBOARD_PROJECT_ROOT', '/tmp/test')

        cfg = DashboardConfig.from_env()
        assert cfg.host == '0.0.0.0'
        assert cfg.port == 9090
        assert cfg.project_root == Path('/tmp/test')
        # Non-overridden fields keep defaults
        assert cfg.fused_memory_url == 'http://localhost:8000'
        assert cfg.fused_memory_project_id == 'dark_factory'

    def test_env_derived_paths_update(self, monkeypatch):
        from dashboard.config import DashboardConfig

        monkeypatch.setenv('DASHBOARD_PROJECT_ROOT', '/tmp/test')
        cfg = DashboardConfig.from_env()
        assert cfg.reconciliation_db == Path('/tmp/test/fused-memory/data/reconciliation/reconciliation.db')
        assert cfg.worktrees_dir == Path('/tmp/test/.worktrees')


class TestHealthEndpoint:
    def test_health_endpoint(self):
        from starlette.testclient import TestClient

        from dashboard.app import app

        with TestClient(app) as client:
            resp = client.get('/api/health')
            assert resp.status_code == 200
            assert resp.json() == {'status': 'ok'}


class TestConftestFixtures:
    def test_config_fixture_uses_tmp_path(self, dashboard_config):
        """The dashboard_config fixture should use a temp directory as project_root."""
        assert '/tmp' in str(dashboard_config.project_root) or 'tmp' in str(dashboard_config.project_root)
        assert dashboard_config.project_root.exists()

    def test_config_fixture_derived_paths(self, dashboard_config):
        """Derived paths should be based on the temp project_root."""
        assert dashboard_config.reconciliation_db.is_relative_to(dashboard_config.project_root)
        assert dashboard_config.worktrees_dir.is_relative_to(dashboard_config.project_root)

    def test_client_fixture(self, client):
        """The client fixture should return a working TestClient."""
        resp = client.get('/api/health')
        assert resp.status_code == 200
        assert resp.json() == {'status': 'ok'}
