"""Tests for configuration loading."""

from importlib import resources as pkg_resources
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orchestrator.config import (
    ConfigRequiredError,
    ModuleConfig,
    OrchestratorConfig,
    TimeoutsConfig,
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
        """timeouts.steward default is 1800s and satisfies the steward_completion_timeout invariant.

        Documents the decoupling invariant: per-invocation wall-clock must be
        >= the workflow grace period (steward_completion_timeout) so a single
        invocation is never silently cut short inside the drain window.
        Equality is permitted; the validator on OrchestratorConfig enforces >=.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        assert config.timeouts.steward == 1800.0
        assert config.timeouts.steward >= config.steward_completion_timeout


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


class TestStewardTimeoutInvariant:
    """OrchestratorConfig must reject configs where timeouts.steward < steward_completion_timeout."""

    def test_direct_init_violation_raises_validation_error(self, monkeypatch, tmp_path):
        """Directly instantiating OrchestratorConfig with a violating config raises ValidationError."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        with pytest.raises(ValidationError, match='steward'):
            OrchestratorConfig(steward_completion_timeout=900.0, timeouts=TimeoutsConfig(steward=600.0))

    def test_yaml_load_violation_raises_validation_error(self, tmp_path, monkeypatch):
        """Loading a YAML config with timeouts.steward < steward_completion_timeout raises ValidationError."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        bad_yaml = tmp_path / 'bad.yaml'
        bad_yaml.write_text(yaml.dump({
            'timeouts': {'steward': 600.0},
            'steward_completion_timeout': 900.0,
        }))
        with pytest.raises(ValidationError, match='steward'):
            load_config(bad_yaml)

    def test_equal_values_allowed(self, monkeypatch, tmp_path):
        """timeouts.steward == steward_completion_timeout is valid (invariant is >=, not strict >)."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig(steward_completion_timeout=900.0, timeouts=TimeoutsConfig(steward=900.0))
        assert config.timeouts.steward == config.steward_completion_timeout

    def test_greater_value_allowed(self, monkeypatch, tmp_path):
        """timeouts.steward > steward_completion_timeout is valid."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig(steward_completion_timeout=900.0, timeouts=TimeoutsConfig(steward=1800.0))
        assert config.timeouts.steward > config.steward_completion_timeout

    def test_error_message_contains_remediation_hint(self, monkeypatch, tmp_path):
        """ValidationError message must include operator-actionable remediation hint.

        Guards against future refactors silently dropping the guidance text.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        with pytest.raises(ValidationError, match=r'(?i)raise.*timeouts\.steward.*or lower.*steward_completion_timeout'):
            OrchestratorConfig(steward_completion_timeout=900.0, timeouts=TimeoutsConfig(steward=600.0))

    def test_env_var_override_triggers_invariant(self, monkeypatch, tmp_path):
        """ORCH_TIMEOUTS__STEWARD env-var override is caught by the mode='after' validator.

        Regression guard: pins that pydantic-settings env-sourced overrides
        are merged into the model before mode='after' validators run.
        A future pydantic-settings source-ordering regression would cause this test to fail.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_TIMEOUTS__STEWARD', '300')
        monkeypatch.setenv('ORCH_STEWARD_COMPLETION_TIMEOUT', '900')
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        # Both sides are test-pinned: timeouts.steward=300, steward_completion_timeout=900
        # env override: timeouts.steward=300 → 300 < 900 → validator must fire
        with pytest.raises(ValidationError, match='steward'):
            OrchestratorConfig()


class TestValidateAssignment:
    """validate_assignment=True must re-run model validators on top-level field mutations."""

    def test_validate_assignment_rejects_steward_completion_timeout_mutation(
        self, monkeypatch, tmp_path
    ):
        """Setting steward_completion_timeout above timeouts.steward must raise ValidationError.

        With validate_assignment=True, this assignment fires _validate_steward_timeout_invariant
        and raises ValidationError.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        # Construct a valid config: defaults give timeouts.steward=1800, sct=900
        cfg = OrchestratorConfig()
        assert cfg.timeouts.steward == 1800.0
        assert cfg.steward_completion_timeout == 900.0
        # Mutate steward_completion_timeout to 2000.0 — now above timeouts.steward=1800.
        # validate_assignment=True fires _validate_steward_timeout_invariant, raising ValidationError.
        with pytest.raises(ValidationError, match='steward'):
            cfg.steward_completion_timeout = 2000.0

    def test_validate_assignment_rejects_timeouts_replacement(
        self, monkeypatch, tmp_path
    ):
        """Replacing cfg.timeouts with a TimeoutsConfig that violates the invariant must raise.

        With validate_assignment=True, assigning cfg.timeouts fires _validate_steward_timeout_invariant
        and raises ValidationError.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        # steward_completion_timeout=900 is valid against default timeouts.steward=1800
        cfg = OrchestratorConfig(steward_completion_timeout=900.0)
        assert cfg.steward_completion_timeout == 900.0
        # Replace timeouts with steward=300 — now 300 < 900, violating the invariant.
        # validate_assignment=True fires _validate_steward_timeout_invariant, raising ValidationError.
        with pytest.raises(ValidationError, match='steward'):
            cfg.timeouts = TimeoutsConfig(steward=300.0)

    def test_validate_assignment_allows_valid_mutation(self, monkeypatch, tmp_path):
        """A valid mutation of steward_completion_timeout must succeed without errors.

        Regression guard: confirms that validate_assignment does not block
        mutations that satisfy the invariant. Passes before and after step-4.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig()
        # default timeouts.steward=1800, steward_completion_timeout=900
        # Setting sct=500 is valid (500 <= 1800).
        cfg.steward_completion_timeout = 500.0
        assert cfg.steward_completion_timeout == 500.0

    def test_project_root_resolved_on_assignment(self, monkeypatch, tmp_path):
        """Assigning a relative path to project_root after construction must resolve it to absolute.

        With a @field_validator('project_root', mode='after') and validate_assignment=True,
        post-construction assignment fires the field validator, resolving the path.
        This test fails when model_post_init is used (which only fires at construction).
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig()
        # Assign a relative path post-construction; the field validator must resolve it
        cfg.project_root = Path('relative/subdir')
        assert cfg.project_root.is_absolute() is True
