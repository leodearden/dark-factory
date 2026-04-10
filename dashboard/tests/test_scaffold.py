"""Tests for dashboard scaffold: config, app, and fixtures."""

import dataclasses
import re
from pathlib import Path

import pytest

from dashboard.config import DEFAULT_FUSED_MEMORY_URLS, DashboardConfig


@pytest.fixture
def symlinked_dir(tmp_path):
    """Create a `link -> real` symlink in tmp_path.

    Returns `(link, real.resolve())` so tests can pass `link` to production code
    and assert against the pre-resolved real path, eliminating the 4-line
    setup boilerplate that was previously duplicated across 5 tests.
    """
    real = tmp_path / 'real'
    real.mkdir()
    link = tmp_path / 'link'
    link.symlink_to(real)
    return link, real.resolve()


class TestConfigDefaults:
    def test_default_fused_memory_urls_is_immutable(self):
        assert isinstance(DEFAULT_FUSED_MEMORY_URLS, tuple)

    def test_default_fused_memory_urls_values(self):
        assert DEFAULT_FUSED_MEMORY_URLS == ('http://localhost:8002',)

    def test_config_defaults(self):
        cfg = DashboardConfig()
        assert cfg.host == '127.0.0.1'
        assert cfg.port == 8080
        assert cfg.project_root == Path('/home/leo/src/dark-factory')
        assert cfg.fused_memory_urls == list(DEFAULT_FUSED_MEMORY_URLS)

    def test_known_project_roots_default_empty_list(self):
        cfg = DashboardConfig()
        assert cfg.known_project_roots == []

    def test_known_project_roots_is_list_not_tuple(self):
        cfg = DashboardConfig()
        assert isinstance(cfg.known_project_roots, list)

    def test_config_derived_paths(self):
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
        monkeypatch.setenv('DASHBOARD_FUSED_MEMORY_URLS', 'http://a:1, http://b:2 ')
        cfg = DashboardConfig.from_env()
        assert cfg.fused_memory_urls == ['http://a:1', 'http://b:2']

    def test_fused_memory_urls_single(self, monkeypatch):
        monkeypatch.setenv('DASHBOARD_FUSED_MEMORY_URLS', 'http://localhost:9000')
        cfg = DashboardConfig.from_env()
        assert cfg.fused_memory_urls == ['http://localhost:9000']

    def test_fused_memory_urls_empty_string(self, monkeypatch):
        monkeypatch.setenv('DASHBOARD_FUSED_MEMORY_URLS', '')
        cfg = DashboardConfig.from_env()
        assert cfg.fused_memory_urls == []

    def test_fused_memory_urls_extra_commas(self, monkeypatch):
        monkeypatch.setenv('DASHBOARD_FUSED_MEMORY_URLS', ',http://a:1,,http://b:2,')
        cfg = DashboardConfig.from_env()
        assert cfg.fused_memory_urls == ['http://a:1', 'http://b:2']

    def test_known_project_roots_single(self, monkeypatch):
        monkeypatch.setenv('DASHBOARD_KNOWN_PROJECT_ROOTS', '/home/leo/src/reify')
        cfg = DashboardConfig.from_env()
        assert cfg.known_project_roots == [Path('/home/leo/src/reify')]

    def test_known_project_roots_comma_separated(self, monkeypatch):
        monkeypatch.setenv('DASHBOARD_KNOWN_PROJECT_ROOTS', '/a,/b')
        cfg = DashboardConfig.from_env()
        assert cfg.known_project_roots == [Path('/a'), Path('/b')]

    def test_known_project_roots_extra_commas_and_whitespace(self, monkeypatch):
        monkeypatch.setenv('DASHBOARD_KNOWN_PROJECT_ROOTS', ' ,/a, , /b , ')
        cfg = DashboardConfig.from_env()
        assert cfg.known_project_roots == [Path('/a'), Path('/b')]

    def test_known_project_roots_empty_string(self, monkeypatch):
        monkeypatch.setenv('DASHBOARD_KNOWN_PROJECT_ROOTS', '')
        cfg = DashboardConfig.from_env()
        assert cfg.known_project_roots == []

    def test_from_env_resolves_known_project_roots(self, monkeypatch, symlinked_dir):
        """from_env() must resolve symlinked paths in DASHBOARD_KNOWN_PROJECT_ROOTS."""
        from dashboard.config import DashboardConfig

        link, real_resolved = symlinked_dir
        monkeypatch.setenv('DASHBOARD_KNOWN_PROJECT_ROOTS', str(link))
        cfg = DashboardConfig.from_env()
        # known_project_roots must contain the resolved real path, not the symlink
        assert cfg.known_project_roots == [real_resolved]

    def test_from_env_resolves_project_root_symlink(self, monkeypatch, tmp_path):
        """from_env() must resolve a symlinked path in DASHBOARD_PROJECT_ROOT."""
        real_dir = tmp_path / 'real'
        real_dir.mkdir()
        link = tmp_path / 'link'
        link.symlink_to(real_dir)

        monkeypatch.setenv('DASHBOARD_PROJECT_ROOT', str(link))
        cfg = DashboardConfig.from_env()
        assert cfg.project_root == real_dir.resolve()

    def test_known_project_roots_unset_preserves_default(self, monkeypatch):
        monkeypatch.setenv('DASHBOARD_HOST', '0.0.0.0')
        # DASHBOARD_KNOWN_PROJECT_ROOTS is intentionally NOT set
        cfg = DashboardConfig.from_env()
        assert cfg.known_project_roots == []
        # verify other overrides still apply
        assert cfg.host == '0.0.0.0'

    def test_env_derived_paths_update(self, monkeypatch):
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


class TestGitignore:
    """Tests that .gitignore excludes generated build artifacts."""

    def test_gitignore_excludes_tailwind_css(self):
        """tailwind.css is a generated artifact and must be gitignored."""
        content = (DASHBOARD_ROOT / '.gitignore').read_text()
        assert 'tailwind.css' in content

    def test_gitignore_excludes_bin(self):
        """bin/ (downloaded Tailwind binary) must remain gitignored (regression guard)."""
        content = (DASHBOARD_ROOT / '.gitignore').read_text()
        assert 'bin/' in content


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

    def test_makefile_checksums_are_valid_sha256(self):
        """All CHECKSUM_ variables must be exactly 64 hex characters (SHA-256)."""
        content = (DASHBOARD_ROOT / 'Makefile').read_text()
        checksums = re.findall(r'CHECKSUM_[\w-]+\s*:=\s*([0-9a-f]+)', content)
        assert len(checksums) == 4, f'Expected 4 CHECKSUM_ entries, found {len(checksums)}'
        for checksum in checksums:
            assert len(checksum) == 64, (
                f'Checksum {checksum!r} is {len(checksum)} chars, expected 64'
            )

    def test_makefile_has_delete_on_error(self):
        """Makefile must include .DELETE_ON_ERROR to clean up stale binaries on failure."""
        content = (DASHBOARD_ROOT / 'Makefile').read_text()
        assert '.DELETE_ON_ERROR' in content

    def test_makefile_has_cleanup_guard(self):
        """Download recipe must chain commands with && and have rm -f on failure (|| guard)."""
        content = (DASHBOARD_ROOT / 'Makefile').read_text()
        # The recipe must use && to chain curl/checksum/chmod
        assert '&&' in content
        # The || guard must clean up the binary on any failure
        assert 'rm -f' in content

    def test_makefile_download_is_single_shell_block(self):
        """curl, checksum, and chmod must be in one backslash-continued shell block."""
        content = (DASHBOARD_ROOT / 'Makefile').read_text()
        # In a single-shell block commands are joined with backslash-newline or && on one line
        # Verify curl and chmod appear together with && chaining (not as separate recipe lines)
        # A separate recipe line would have curl on its own line without a trailing backslash
        lines = content.splitlines()
        curl_line_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('curl ') and 'TAILWIND_BIN' in line:
                curl_line_idx = i
                break
        assert curl_line_idx is not None, 'No curl line found in Makefile'
        curl_line = lines[curl_line_idx]
        # The curl line must end with backslash (continuation) or contain &&
        # indicating it's part of a multi-command shell block
        assert curl_line.rstrip().endswith('\\') or '&&' in curl_line, (
            f'curl line is not part of a single shell block: {curl_line!r}'
        )


class TestInputCSS:
    """Tests that input.css contains all required CSS rules including animations."""

    def test_input_css_has_refresh_pulse_keyframes(self):
        """input.css must define the @keyframes section-refresh-pulse animation."""
        content = (DASHBOARD_ROOT / 'src' / 'dashboard' / 'static' / 'input.css').read_text()
        assert '@keyframes section-refresh-pulse' in content

    def test_input_css_has_section_refreshed_class(self):
        """input.css must define the .section-refreshed class."""
        content = (DASHBOARD_ROOT / 'src' / 'dashboard' / 'static' / 'input.css').read_text()
        assert '.section-refreshed' in content


class TestStaticFiles:
    def test_static_css_served(self, client):
        """Static CSS file should be served at /static/tailwind.css."""
        resp = client.get('/static/tailwind.css')
        assert resp.status_code == 200
        assert 'text/css' in resp.headers['content-type']


class TestPostInit:
    """Unit tests for DashboardConfig.__post_init__ path normalization invariant.

    These tests exercise the invariant directly on DashboardConfig construction
    (not through burndown consumer logic), so they remain valid even if consumer
    code is later refactored.
    """

    def test_resolves_project_root_symlink(self, tmp_path):
        """DashboardConfig must resolve a symlinked project_root in __post_init__."""
        real_dir = tmp_path / 'real'
        real_dir.mkdir()
        link = tmp_path / 'link'
        link.symlink_to(real_dir)

        cfg = DashboardConfig(project_root=link)
        assert cfg.project_root == real_dir.resolve()

    def test_resolves_known_project_roots_symlinks(self, tmp_path):
        """DashboardConfig must resolve symlinks in known_project_roots in __post_init__."""
        real1 = tmp_path / 'real1'
        real1.mkdir()
        link1 = tmp_path / 'link1'
        link1.symlink_to(real1)

        real2 = tmp_path / 'real2'
        real2.mkdir()
        link2 = tmp_path / 'link2'
        link2.symlink_to(real2)

        cfg = DashboardConfig(project_root=tmp_path, known_project_roots=[link1, link2])
        assert cfg.known_project_roots == [real1.resolve(), real2.resolve()]

    def test_tasks_json_derived_from_resolved_root(self, tmp_path):
        """tasks_json must be derived from the resolved project_root, not the symlink path."""
        real_dir = tmp_path / 'real'
        real_dir.mkdir()
        link = tmp_path / 'link'
        link.symlink_to(real_dir)

        cfg = DashboardConfig(project_root=link)
        assert cfg.tasks_json == real_dir.resolve() / '.taskmaster' / 'tasks' / 'tasks.json'

    def test_replace_resolves_known_project_roots_symlinks(self, tmp_path):
        """dataclasses.replace() must resolve symlinks in known_project_roots via __post_init__.

        Validates the __post_init__ docstring contract that dataclasses.replace() is a
        covered construction path: 'every construction path — direct kwargs, from_env(),
        dataclass.replace(), test fixtures'.  dataclasses.replace() internally calls
        __init__ which triggers __post_init__, so the invariant must hold.
        """
        base_cfg = DashboardConfig(project_root=tmp_path)

        real_dir = tmp_path / 'real'
        real_dir.mkdir()
        link = tmp_path / 'link'
        link.symlink_to(real_dir)

        new_cfg = dataclasses.replace(base_cfg, known_project_roots=[link])
        assert new_cfg.known_project_roots == [real_dir.resolve()]
