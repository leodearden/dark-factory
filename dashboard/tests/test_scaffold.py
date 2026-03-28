"""Tests for dashboard scaffold: config, app, and fixtures."""

from pathlib import Path

from dashboard.config import DEFAULT_FUSED_MEMORY_URLS


class TestConfigDefaults:
    def test_default_fused_memory_urls_is_immutable(self):
        assert isinstance(DEFAULT_FUSED_MEMORY_URLS, tuple)

    def test_default_fused_memory_urls_values(self):
        assert DEFAULT_FUSED_MEMORY_URLS == ('http://localhost:8002',)

    def test_config_defaults(self):
        from dashboard.config import DashboardConfig

        cfg = DashboardConfig()
        assert cfg.host == '127.0.0.1'
        assert cfg.port == 8080
        assert cfg.project_root == Path('/home/leo/src/dark-factory')
        assert cfg.fused_memory_urls == list(DEFAULT_FUSED_MEMORY_URLS)

    def test_config_derived_paths(self):
        from dashboard.config import DashboardConfig

        cfg = DashboardConfig()
        root = cfg.project_root
        assert (
            cfg.reconciliation_db
            == root / 'data' / 'reconciliation' / 'reconciliation.db'
        )
        assert cfg.write_queue_db == root / 'data' / 'queue' / 'write_queue.db'
        assert (
            cfg.write_journal_db
            == root / 'data' / 'reconciliation' / 'write_journal.db'
        )
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
        # Non-overridden fields keep defaults (strict equality, not just length)
        assert cfg.fused_memory_urls == list(DEFAULT_FUSED_MEMORY_URLS)

    def test_fused_memory_urls_comma_separated(self, monkeypatch):
        from dashboard.config import DashboardConfig

        monkeypatch.setenv('DASHBOARD_FUSED_MEMORY_URLS', 'http://a:1, http://b:2 ')
        cfg = DashboardConfig.from_env()
        assert cfg.fused_memory_urls == ['http://a:1', 'http://b:2']

    def test_fused_memory_urls_single(self, monkeypatch):
        from dashboard.config import DashboardConfig

        monkeypatch.setenv('DASHBOARD_FUSED_MEMORY_URLS', 'http://localhost:9000')
        cfg = DashboardConfig.from_env()
        assert cfg.fused_memory_urls == ['http://localhost:9000']

    def test_fused_memory_urls_empty_string(self, monkeypatch):
        from dashboard.config import DashboardConfig

        monkeypatch.setenv('DASHBOARD_FUSED_MEMORY_URLS', '')
        cfg = DashboardConfig.from_env()
        assert cfg.fused_memory_urls == []

    def test_fused_memory_urls_extra_commas(self, monkeypatch):
        from dashboard.config import DashboardConfig

        monkeypatch.setenv('DASHBOARD_FUSED_MEMORY_URLS', ',http://a:1,,http://b:2,')
        cfg = DashboardConfig.from_env()
        assert cfg.fused_memory_urls == ['http://a:1', 'http://b:2']

    def test_env_derived_paths_update(self, monkeypatch):
        from dashboard.config import DashboardConfig

        monkeypatch.setenv('DASHBOARD_PROJECT_ROOT', '/tmp/test')
        cfg = DashboardConfig.from_env()
        assert cfg.reconciliation_db == Path(
            '/tmp/test/data/reconciliation/reconciliation.db'
        )
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
        assert '/tmp' in str(dashboard_config.project_root) or 'tmp' in str(
            dashboard_config.project_root
        )
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


DASHBOARD_ROOT = Path(__file__).parent.parent


class TestMakefile:
    """Tests that the Makefile exists and has platform detection + checksum verification."""

    def test_makefile_exists(self):
        assert (DASHBOARD_ROOT / 'Makefile').is_file()

    def test_makefile_has_platform_detection(self):
        content = (DASHBOARD_ROOT / 'Makefile').read_text()
        assert 'uname' in content
        assert 'linux' in content.lower() or 'Linux' in content
        assert 'darwin' in content.lower() or 'Darwin' in content

    def test_makefile_has_checksum_verification(self):
        content = (DASHBOARD_ROOT / 'Makefile').read_text()
        assert 'sha256' in content


class TestStaticFiles:
    def test_static_css_served(self, client):
        """Static CSS file should be served at /static/tailwind.css."""
        resp = client.get('/static/tailwind.css')
        assert resp.status_code == 200
        assert 'text/css' in resp.headers['content-type']
