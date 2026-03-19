"""Tests for configuration loading."""

import logging
from pathlib import Path

import yaml

from orchestrator.config import (
    ModuleConfig,
    OrchestratorConfig,
    _discover_module_configs,
    load_config,
)


class TestDefaults:
    """Tests for OrchestratorConfig defaults — isolated from real config files."""

    def test_default_values(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        config = OrchestratorConfig()
        assert config.max_concurrent_tasks == 3
        assert config.max_per_module == 1
        assert config.max_execute_iterations == 10
        assert config.max_verify_attempts == 5
        assert config.max_review_cycles == 2
        assert config.models.architect == 'opus'
        assert config.models.reviewer == 'sonnet'
        assert config.budgets.implementer == 10.0
        assert config.max_turns.implementer == 80
        assert config.test_command == 'pytest'

    def test_git_defaults(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        config = OrchestratorConfig()
        assert config.git.main_branch == 'main'
        assert config.git.branch_prefix == 'task/'

    def test_fused_memory_defaults(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        config = OrchestratorConfig()
        assert config.fused_memory.url == 'http://localhost:8002'
        assert config.fused_memory.project_id == 'dark_factory'
        assert config.fused_memory.config_path == 'fused-memory/config/config.yaml'
        # server_command must contain '--project' followed by 'fused-memory' (no ../)
        cmd = config.fused_memory.server_command
        assert '--project' in cmd
        project_arg_idx = cmd.index('--project')
        assert cmd[project_arg_idx + 1] == 'fused-memory'

    def test_fused_memory_defaults_no_parent_traversal(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        config = OrchestratorConfig()
        assert '../' not in config.fused_memory.config_path
        assert not any('../' in arg for arg in config.fused_memory.server_command)

    def test_project_root_resolved_to_absolute(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        config = OrchestratorConfig(project_root=Path('.'))
        assert config.project_root.is_absolute() is True


class TestYamlLoading:
    def test_load_config_warns_when_no_config_file(self, tmp_path: Path, caplog):
        nonexistent = tmp_path / 'nonexistent.yaml'
        with caplog.at_level(logging.WARNING, logger='orchestrator.config'):
            load_config(nonexistent)
        assert any(
            'No config file' in r.message or 'using defaults' in r.message.lower()
            for r in caplog.records
        )

    def test_load_config_discovers_orchestrator_subdir(self, tmp_path: Path, monkeypatch):
        # Create orchestrator/config.yaml under tmp_path
        orch_dir = tmp_path / 'orchestrator'
        orch_dir.mkdir()
        (orch_dir / 'config.yaml').write_text(yaml.dump({'max_concurrent_tasks': 7}))
        # Change cwd to tmp_path and clear ORCH_CONFIG_PATH
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        config = load_config(None)
        assert config.max_concurrent_tasks == 7

    def test_load_config_warns_when_no_config_found(self, tmp_path: Path, monkeypatch, caplog):
        # Empty dir with no config files and no ORCH_CONFIG_PATH
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        with caplog.at_level(logging.WARNING, logger='orchestrator.config'):
            load_config(None)
        assert any('No config file found' in r.message for r in caplog.records)

    def test_load_from_yaml(self, tmp_path: Path):
        config_data = {
            'max_concurrent_tasks': 5,
            'models': {'architect': 'sonnet'},
            'budgets': {'architect': 3.0},
        }
        config_path = tmp_path / 'config.yaml'
        config_path.write_text(yaml.dump(config_data))

        config = load_config(config_path)
        assert config.max_concurrent_tasks == 5
        assert config.models.architect == 'sonnet'
        assert config.budgets.architect == 3.0
        # Unset values should use defaults
        assert config.models.implementer == 'opus'


class TestModuleConfigDiscovery:
    def test_discover_finds_orchestrator_yaml(self, tmp_path: Path):
        sub = tmp_path / 'dashboard'
        sub.mkdir()
        (sub / 'orchestrator.yaml').write_text(yaml.dump({
            'test_command': 'pytest dashboard/',
            'lint_command': 'ruff check dashboard/',
            'lock_depth': 3,
        }))
        configs = _discover_module_configs(tmp_path)
        assert 'dashboard' in configs
        mc = configs['dashboard']
        assert mc.prefix == 'dashboard'
        assert mc.test_command == 'pytest dashboard/'
        assert mc.lint_command == 'ruff check dashboard/'
        assert mc.lock_depth == 3
        assert mc.type_check_command is None

    def test_discover_ignores_non_overridable_fields(self, tmp_path: Path):
        sub = tmp_path / 'mymod'
        sub.mkdir()
        (sub / 'orchestrator.yaml').write_text(yaml.dump({
            'test_command': 'pytest',
            'models': {'architect': 'sonnet'},
            'budgets': {'architect': 1.0},
            'project_root': '/nope',
        }))
        configs = _discover_module_configs(tmp_path)
        mc = configs['mymod']
        assert mc.test_command == 'pytest'
        assert not hasattr(mc, 'models')
        assert not hasattr(mc, 'budgets')
        assert not hasattr(mc, 'project_root')

    def test_discover_empty_dir(self, tmp_path: Path):
        configs = _discover_module_configs(tmp_path)
        assert configs == {}

    def test_for_module_matches_first_component(self):
        config = OrchestratorConfig()
        config._module_configs = {
            'dashboard': ModuleConfig(prefix='dashboard', test_command='pytest dash/'),
        }
        mc = config.for_module('dashboard/src/app.py')
        assert mc is not None
        assert mc.prefix == 'dashboard'
        assert mc.test_command == 'pytest dash/'

    def test_for_module_returns_none_for_unknown(self):
        config = OrchestratorConfig()
        config._module_configs = {
            'dashboard': ModuleConfig(prefix='dashboard'),
        }
        assert config.for_module('orchestrator/src/config.py') is None

    def test_load_config_populates_module_configs(self, tmp_path: Path):
        # Create a minimal global config
        config_path = tmp_path / 'config.yaml'
        config_path.write_text(yaml.dump({
            'project_root': str(tmp_path),
        }))
        # Create a subproject orchestrator.yaml
        sub = tmp_path / 'backend'
        sub.mkdir()
        (sub / 'orchestrator.yaml').write_text(yaml.dump({
            'test_command': 'cargo test',
            'max_per_module': 2,
        }))
        config = load_config(config_path)
        assert 'backend' in config._module_configs
        assert config._module_configs['backend'].test_command == 'cargo test'
        assert config._module_configs['backend'].max_per_module == 2


class TestPathResolution:
    def test_fused_memory_paths_resolve_under_project_root(self, tmp_path: Path):
        config = OrchestratorConfig(project_root=tmp_path)
        resolved = (config.project_root / config.fused_memory.config_path).resolve()
        assert str(resolved).startswith(str(tmp_path))
        assert '..' not in resolved.parts
