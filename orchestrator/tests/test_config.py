"""Tests for configuration loading."""

from importlib import resources as pkg_resources
from pathlib import Path

import pytest
import yaml

from orchestrator.config import (
    ConfigRequiredError,
    ModuleConfig,
    OrchestratorConfig,
    _deep_merge,
    _discover_module_configs,
    load_config,
)


def _load_package_defaults() -> dict:
    """Read the shipped defaults.yaml so tests stay in sync automatically."""
    defaults_file = pkg_resources.files('orchestrator') / 'defaults.yaml'
    return yaml.safe_load(defaults_file.read_text())


class TestDefaults:
    """Tests for OrchestratorConfig defaults — isolated from real config files."""

    def test_default_values(self, monkeypatch, tmp_path):
        """Package defaults.yaml is loaded via settings_customise_sources."""
        monkeypatch.chdir(tmp_path)
        # Ensure no external config is loaded - must unset env before instantiation
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        defaults = _load_package_defaults()
        # Values from package defaults.yaml (not Pydantic field defaults)
        assert config.max_concurrent_tasks == defaults['max_concurrent_tasks']
        assert config.max_per_module == defaults['max_per_module']
        assert config.max_execute_iterations == defaults['max_execute_iterations']
        assert config.max_verify_attempts == defaults['max_verify_attempts']
        assert config.max_review_cycles == defaults['max_review_cycles']
        assert config.reviewer_stagger_secs == defaults['reviewer_stagger_secs']
        assert config.max_reviewer_retries == defaults['max_reviewer_retries']
        assert config.models.architect == defaults['models']['architect']
        assert config.models.reviewer == defaults['models']['reviewer']
        assert config.budgets.implementer == defaults['budgets']['implementer']
        assert config.max_turns.implementer == defaults['max_turns']['implementer']

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

    def test_steward_timeout_default_is_1800(self, monkeypatch, tmp_path):
        """timeouts.steward default is 1800s and exceeds steward_completion_timeout.

        Documents the decoupling invariant: per-invocation wall-clock must be
        strictly greater than the workflow grace period so a single invocation
        cannot silently blow past the drain window.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        assert config.timeouts.steward == 1800.0
        assert config.timeouts.steward > config.steward_completion_timeout


class TestYamlLoading:
    def test_load_config_raises_when_explicit_path_nonexistent(self, tmp_path: Path):
        """Explicit --config pointing at a missing file raises ConfigRequiredError."""
        nonexistent = tmp_path / 'nonexistent.yaml'
        with pytest.raises(ConfigRequiredError, match='Config file not found'):
            load_config(nonexistent)

    def test_load_config_uses_orch_config_path_env_var(self, tmp_path: Path, monkeypatch):
        """ORCH_CONFIG_PATH alone (no --config flag) loads the right config."""
        cfg = tmp_path / 'config.yaml'
        cfg.write_text(yaml.dump({'max_concurrent_tasks': 13}))
        monkeypatch.setenv('ORCH_CONFIG_PATH', str(cfg))
        config = load_config(None)
        assert config.max_concurrent_tasks == 13

    def test_load_config_explicit_flag_overrides_env_var(
        self, tmp_path: Path, monkeypatch,
    ):
        """When both --config and ORCH_CONFIG_PATH are set, --config wins."""
        env_cfg = tmp_path / 'env.yaml'
        env_cfg.write_text(yaml.dump({'max_concurrent_tasks': 1}))
        flag_cfg = tmp_path / 'flag.yaml'
        flag_cfg.write_text(yaml.dump({'max_concurrent_tasks': 99}))
        monkeypatch.setenv('ORCH_CONFIG_PATH', str(env_cfg))
        config = load_config(flag_cfg)
        assert config.max_concurrent_tasks == 99

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
        # Unset values should use package defaults
        assert config.models.implementer == 'sonnet'


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


class TestLayeredConfig:
    """Tests for deep merge of package defaults + project config."""

    def test_deep_merge_basic(self):
        base = {'a': 1, 'b': {'x': 10, 'y': 20}}
        override = {'b': {'y': 99}, 'c': 3}
        result = _deep_merge(base, override)
        assert result == {'a': 1, 'b': {'x': 10, 'y': 99}, 'c': 3}

    def test_deep_merge_override_replaces_non_dict(self):
        base = {'a': {'nested': 1}}
        override = {'a': 'flat'}
        result = _deep_merge(base, override)
        assert result == {'a': 'flat'}

    def test_deep_merge_does_not_mutate_base(self):
        base = {'a': {'x': 1}}
        override = {'a': {'y': 2}}
        _deep_merge(base, override)
        assert base == {'a': {'x': 1}}

    def test_load_config_raises_when_no_config_and_no_env(self, tmp_path, monkeypatch):
        """Without --config or ORCH_CONFIG_PATH, load_config refuses to start."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        with pytest.raises(ConfigRequiredError, match='--config is required'):
            load_config(None)

    def test_project_config_overrides_defaults(self, tmp_path, monkeypatch):
        """Project config values should override package defaults."""
        project_cfg = tmp_path / 'config.yaml'
        project_cfg.write_text(yaml.dump({
            'models': {'implementer': 'opus'},
            'max_concurrent_tasks': 8,
        }))
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        config = load_config(project_cfg)
        # Overridden
        assert config.models.implementer == 'opus'
        assert config.max_concurrent_tasks == 8
        # Preserved from package defaults
        assert config.models.architect == 'opus'
        assert config.effort.architect == 'max'

    def test_deep_merge_preserves_sibling_keys(self, tmp_path, monkeypatch):
        """Overriding one key in a nested dict should not clobber siblings."""
        project_cfg = tmp_path / 'config.yaml'
        project_cfg.write_text(yaml.dump({
            'budgets': {'architect': 99.0},
        }))
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        config = load_config(project_cfg)
        assert config.budgets.architect == 99.0
        assert config.budgets.implementer == 10.0  # preserved from defaults


class TestPathResolution:
    def test_fused_memory_paths_resolve_under_project_root(self, tmp_path: Path):
        config = OrchestratorConfig(project_root=tmp_path)
        resolved = (config.project_root / config.fused_memory.config_path).resolve()
        assert str(resolved).startswith(str(tmp_path))
        assert '..' not in resolved.parts
