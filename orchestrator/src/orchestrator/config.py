"""Configuration schema for the orchestrator."""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source for loading from YAML files."""

    def __init__(self, settings_cls: type[BaseSettings], config_path: Path | None = None):
        super().__init__(settings_cls)
        self.config_path = config_path or Path('config.yaml')

    def _expand_env_vars(self, value: Any) -> Any:
        if isinstance(value, str):
            pattern = r'\$\{([^:}]+)(:([^}]*))?\}'

            def replacer(match):
                var_name = match.group(1)
                default_value = match.group(3) if match.group(3) is not None else ''
                return os.environ.get(var_name, default_value)

            full_match = re.fullmatch(pattern, value)
            if full_match:
                result = replacer(full_match)
                if isinstance(result, str):
                    lower = result.lower().strip()
                    if lower in ('true', '1', 'yes', 'on'):
                        return True
                    elif lower in ('false', '0', 'no', 'off'):
                        return False
                    elif lower == '':
                        return None
                return result
            else:
                return re.sub(pattern, replacer, value)
        elif isinstance(value, dict):
            return {k: self._expand_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._expand_env_vars(item) for item in value]
        return value

    def get_field_value(self, field_name: str, field_info: Any) -> Any:
        return None

    def __call__(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}
        with open(self.config_path) as f:
            raw_config = yaml.safe_load(f) or {}
        return self._expand_env_vars(raw_config)


# --- Sub-models ---


class ModelsConfig(BaseModel):
    """Model selection per agent role."""

    architect: str = Field(default='opus')
    implementer: str = Field(default='opus')
    debugger: str = Field(default='opus')
    reviewer: str = Field(default='sonnet')
    merger: str = Field(default='opus')
    module_tagger: str = Field(default='sonnet')


class BudgetsConfig(BaseModel):
    """Max USD spend per invocation, by role."""

    architect: float = Field(default=5.0)
    implementer: float = Field(default=10.0)
    debugger: float = Field(default=5.0)
    reviewer: float = Field(default=2.0)
    merger: float = Field(default=5.0)
    module_tagger: float = Field(default=2.0)


class TurnsConfig(BaseModel):
    """Max conversation turns per invocation, by role."""

    architect: int = Field(default=50)
    implementer: int = Field(default=80)
    debugger: int = Field(default=50)
    reviewer: int = Field(default=30)
    merger: int = Field(default=50)
    module_tagger: int = Field(default=30)


class FusedMemoryConfig(BaseModel):
    """Fused-memory HTTP server connection."""

    url: str = Field(default='http://localhost:8000')
    project_id: str = Field(default='dark_factory')
    config_path: str = Field(default='../fused-memory/config/config.yaml')
    server_command: list[str] = Field(
        default_factory=lambda: [
            'uv', 'run', '--project', '../fused-memory',
            'python', '-m', 'fused_memory.server.main',
            '--transport', 'http',
        ]
    )


class SandboxConfig(BaseModel):
    """Bubblewrap filesystem sandbox configuration."""

    enabled: bool = Field(default=True)


class EscalationConfig(BaseModel):
    """Escalation MCP server configuration."""

    queue_dir: str = Field(default='data/escalations')
    port: int = Field(default=8100)
    host: str = Field(default='127.0.0.1')


class UsageCapConfig(BaseModel):
    """Usage cap detection and handling."""

    enabled: bool = Field(default=True)
    session_budget_usd: float | None = Field(default=None)
    pause_threshold: float = Field(default=0.96)
    wait_for_reset: bool = Field(default=True)
    probe_interval_secs: int = Field(default=300)
    max_probe_interval_secs: int = Field(default=1800)


class GitConfig(BaseModel):
    """Git operations configuration."""

    main_branch: str = Field(default='main')
    branch_prefix: str = Field(default='task/')
    remote: str = Field(default='origin')
    worktree_dir: str = Field(default='.worktrees')


# --- Top-level ---


class OrchestratorConfig(BaseSettings):
    """Orchestrator configuration with YAML and environment support."""

    # Concurrency
    max_concurrent_tasks: int = Field(default=3)
    max_per_module: int = Field(default=1)
    lock_depth: int = Field(default=2)
    module_overrides: dict[str, int] = Field(default_factory=dict)

    # Iteration limits
    max_execute_iterations: int = Field(default=10)
    max_verify_attempts: int = Field(default=5)
    max_review_cycles: int = Field(default=2)

    # Models, budgets, turns per role
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    budgets: BudgetsConfig = Field(default_factory=BudgetsConfig)
    max_turns: TurnsConfig = Field(default_factory=TurnsConfig)

    # Verification commands
    test_command: str = Field(default='pytest')
    lint_command: str = Field(default='ruff check')
    type_check_command: str = Field(default='pyright')

    # Fused memory
    fused_memory: FusedMemoryConfig = Field(default_factory=FusedMemoryConfig)

    # Sandbox
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)

    # Escalation
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)

    # Git
    git: GitConfig = Field(default_factory=GitConfig)

    # Usage cap handling
    usage_cap: UsageCapConfig = Field(default_factory=UsageCapConfig)

    # Project
    project_root: Path = Field(default=Path('.'))

    model_config = SettingsConfigDict(
        env_prefix='ORCH_',
        env_nested_delimiter='__',
        case_sensitive=False,
        extra='ignore',
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        config_path = Path(os.environ.get('ORCH_CONFIG_PATH', 'config.yaml'))
        yaml_settings = YamlSettingsSource(settings_cls, config_path)
        return (init_settings, env_settings, yaml_settings, dotenv_settings)


def load_config(config_path: Path | None = None) -> OrchestratorConfig:
    """Load configuration from YAML file, env vars, and defaults."""
    if config_path:
        os.environ['ORCH_CONFIG_PATH'] = str(config_path)
    return OrchestratorConfig()
