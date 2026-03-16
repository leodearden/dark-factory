"""Tests for configuration loading."""

from pathlib import Path

import yaml

from orchestrator.config import OrchestratorConfig, load_config


class TestDefaults:
    def test_default_values(self):
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

    def test_git_defaults(self):
        config = OrchestratorConfig()
        assert config.git.main_branch == 'main'
        assert config.git.branch_prefix == 'task/'

    def test_fused_memory_defaults(self):
        config = OrchestratorConfig()
        assert config.fused_memory.url == 'http://localhost:8000'
        assert config.fused_memory.project_id == 'dark_factory'


class TestYamlLoading:
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
