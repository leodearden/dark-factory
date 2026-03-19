"""Configuration schema for the orchestrator."""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

logger = logging.getLogger(__name__)


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

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

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


class EffortConfig(BaseModel):
    """Reasoning effort level per agent role."""

    architect: str = Field(default='high')
    implementer: str = Field(default='high')
    debugger: str = Field(default='high')
    reviewer: str = Field(default='medium')
    merger: str = Field(default='high')
    module_tagger: str = Field(default='medium')


class BackendsConfig(BaseModel):
    """Backend CLI selection per agent role. Values: 'claude', 'codex', 'gemini'."""

    architect: str = Field(default='claude')
    implementer: str = Field(default='claude')
    debugger: str = Field(default='claude')
    reviewer: str = Field(default='claude')
    merger: str = Field(default='claude')
    module_tagger: str = Field(default='claude')


class FusedMemoryConfig(BaseModel):
    """Fused-memory HTTP server connection."""

    url: str = Field(default='http://localhost:8002')
    project_id: str = Field(default='dark_factory')
    config_path: str = Field(default='fused-memory/config/config.yaml')
    server_command: list[str] = Field(
        default_factory=lambda: [
            'uv', 'run', '--project', 'fused-memory',
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


from shared.config_models import AccountConfig, UsageCapConfig  # noqa: F401, E402


class GitConfig(BaseModel):
    """Git operations configuration."""

    main_branch: str = Field(default='main')
    branch_prefix: str = Field(default='task/')
    remote: str = Field(default='origin')
    worktree_dir: str = Field(default='.worktrees')


# --- Per-module overrides ---

_OVERRIDABLE_FIELDS = frozenset({
    'test_command', 'lint_command', 'type_check_command',
    'lock_depth', 'max_per_module', 'module_overrides',
})


@dataclass
class ModuleConfig:
    """Per-subproject overrides for verification and scheduling."""

    prefix: str
    test_command: str | None = None
    lint_command: str | None = None
    type_check_command: str | None = None
    lock_depth: int | None = None
    max_per_module: int | None = None
    module_overrides: dict[str, int] | None = None


def _discover_module_configs(project_root: Path) -> dict[str, ModuleConfig]:
    """Scan project_root/*/orchestrator.yaml and load overridable fields."""
    configs: dict[str, ModuleConfig] = {}
    for yaml_path in sorted(project_root.glob('*/orchestrator.yaml')):
        prefix = yaml_path.parent.name
        try:
            with open(yaml_path) as f:
                raw = yaml.safe_load(f) or {}
        except Exception:
            logger.warning('Failed to parse %s, skipping', yaml_path)
            continue
        kwargs = {k: raw[k] for k in _OVERRIDABLE_FIELDS if k in raw}
        if kwargs:
            configs[prefix] = ModuleConfig(prefix=prefix, **kwargs)
            logger.info('Loaded module config for %r: %s', prefix, list(kwargs))
    return configs


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
    effort: EffortConfig = Field(default_factory=EffortConfig)
    backends: BackendsConfig = Field(default_factory=BackendsConfig)

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

    # Per-module overrides (populated by load_config via _discover_module_configs)
    _module_configs: dict[str, ModuleConfig] = PrivateAttr(default_factory=dict)

    @model_validator(mode='after')
    def _resolve_project_root(self) -> 'OrchestratorConfig':
        self.project_root = self.project_root.resolve()
        return self

    def for_module(self, module_path: str) -> ModuleConfig | None:
        """Return ModuleConfig matching the first path component, or None."""
        if not self._module_configs:
            return None
        first = module_path.strip('/').split('/')[0]
        return self._module_configs.get(first)

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


def _find_config(explicit_path: Path | None) -> Path | None:
    """Find the config file to load, searching standard locations.

    Search order:
    1. explicit_path (if given and exists)
    2. ORCH_CONFIG_PATH env var (if set and exists)
    3. cwd/config.yaml
    4. cwd/orchestrator/config.yaml
    """
    if explicit_path is not None:
        return explicit_path if explicit_path.exists() else None
    env_path = os.environ.get('ORCH_CONFIG_PATH')
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    cwd_config = Path('config.yaml')
    if cwd_config.exists():
        return cwd_config
    orch_config = Path('orchestrator') / 'config.yaml'
    if orch_config.exists():
        return orch_config
    return None


def load_config(config_path: Path | None = None) -> OrchestratorConfig:
    """Load configuration from YAML file, env vars, and defaults."""
    found = _find_config(config_path)
    if found:
        os.environ['ORCH_CONFIG_PATH'] = str(found)
    elif 'ORCH_CONFIG_PATH' in os.environ:
        # Clear stale env var so YamlSettingsSource returns {}
        del os.environ['ORCH_CONFIG_PATH']
    config = OrchestratorConfig()
    if found is None:
        logger.warning(
            'No config file found (checked config.yaml, orchestrator/config.yaml). '
            'Using defaults. Pass --config or set ORCH_CONFIG_PATH to specify.',
        )
    config._module_configs = _discover_module_configs(config.project_root)
    return config
