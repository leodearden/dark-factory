"""Configuration schemas with pydantic-settings and YAML support."""

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from shared.config_models import UsageCapConfig


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source for loading from YAML files."""

    def __init__(self, settings_cls: type[BaseSettings], config_path: Path | None = None):
        super().__init__(settings_cls)
        self.config_path = config_path or Path('config.yaml')

    def _expand_env_vars(self, value: Any) -> Any:
        """Recursively expand environment variables in configuration values."""
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
                    lower_result = result.lower().strip()
                    if lower_result in ('true', '1', 'yes', 'on'):
                        return True
                    elif lower_result in ('false', '0', 'no', 'off'):
                        return False
                    elif lower_result == '':
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


# --- Server ---

class ServerConfig(BaseModel):
    """Server configuration."""

    transport: Literal['http', 'stdio', 'sse'] = Field(default='http', description='Transport: http, stdio, or sse')
    host: str = Field(default='0.0.0.0', description='Server host')
    port: int = Field(default=8000, description='Server port')
    stateless_http: bool = Field(default=False, description='Stateless HTTP mode (no sessions)')
    json_response: bool = Field(default=False, description='JSON responses instead of SSE')
    keepalive_timeout: int = Field(default=30, description='HTTP keep-alive timeout in seconds')


# --- LLM ---

class OpenAIProviderConfig(BaseModel):
    api_key: str | None = None
    api_url: str = 'https://api.openai.com/v1'
    organization_id: str | None = None


class AnthropicProviderConfig(BaseModel):
    api_key: str | None = None
    api_url: str = 'https://api.anthropic.com'


class LLMProvidersConfig(BaseModel):
    openai: OpenAIProviderConfig | None = None
    anthropic: AnthropicProviderConfig | None = None


class LLMConfig(BaseModel):
    """LLM configuration — shared by both backends."""

    provider: Literal['openai', 'anthropic'] = Field(default='openai')
    model: str = Field(default='gpt-4o-mini')
    temperature: float | None = Field(default=None)
    max_tokens: int = Field(default=4096)
    providers: LLMProvidersConfig = Field(default_factory=LLMProvidersConfig)


# --- Embedder ---

class EmbedderProvidersConfig(BaseModel):
    openai: OpenAIProviderConfig | None = None


class EmbedderConfig(BaseModel):
    """Embedder configuration — shared by both backends."""

    provider: Literal['openai'] = Field(default='openai')
    model: str = Field(default='text-embedding-3-small')
    dimensions: int = Field(default=1536)
    providers: EmbedderProvidersConfig = Field(default_factory=EmbedderProvidersConfig)


# --- Graphiti backend ---

class FalkorDBProviderConfig(BaseModel):
    uri: str = 'redis://localhost:6379'
    password: str | None = None
    database: str = 'default_db'


class GraphitiBackendConfig(BaseModel):
    """Graphiti / FalkorDB backend configuration."""

    provider: Literal['falkordb'] = Field(default='falkordb')
    falkordb: FalkorDBProviderConfig = Field(default_factory=FalkorDBProviderConfig)
    invalidation_guard_enabled: bool = Field(default=True)


# --- Mem0 backend ---

class Mem0BackendConfig(BaseModel):
    """Mem0 / Qdrant backend configuration."""

    qdrant_url: str = Field(default='http://localhost:6333')
    collection_prefix: str = Field(default='fused')


# --- Routing ---

class RoutingConfig(BaseModel):
    """Write/read routing configuration."""

    confidence_threshold: float = Field(default=0.7)
    use_heuristics: bool = Field(default=True)
    llm_fallback: bool = Field(default=True)


# --- Queue ---

class QueueConfig(BaseModel):
    """Durable write queue configuration."""

    semaphore_limit: int = Field(default=3)
    workers_per_group: int = Field(default=1)
    graphiti_max_coroutines: int = Field(default=5)
    max_attempts: int = Field(default=5)
    retry_base_seconds: float = Field(default=5.0)
    retry_max_delay_seconds: float = Field(default=300.0)
    write_timeout_seconds: float = Field(default=120.0)
    backend_read_timeout_seconds: float = Field(default=30.0)
    backend_write_timeout_seconds: float = Field(default=120.0)
    search_timeout_seconds: float = Field(default=15.0)
    data_dir: str = Field(default='./data/queue')


# --- Taskmaster ---

class TaskmasterConfig(BaseModel):
    """Connection to Taskmaster MCP server."""

    transport: str = Field(default='stdio', description='stdio or http')
    command: str = Field(default='node')
    args: list[str] = Field(default_factory=lambda: ['mcp-server/server.js'])
    cwd: str = Field(default='')
    http_url: str = Field(default='')
    project_root: str = Field(default='.')
    tool_mode: str = Field(default='standard')


# --- Reconciliation ---

class ReconciliationConfig(BaseModel):
    """Sleep mode reconciliation settings."""

    enabled: bool = Field(default=True)
    data_dir: str = Field(default='./data/reconciliation')

    # Buffer triggers
    buffer_size_threshold: int = Field(default=10)
    max_staleness_seconds: int = Field(default=1800)

    # Agent settings
    agent_llm_provider: str = Field(default='claude_cli')
    agent_llm_model: str = Field(default='sonnet')
    agent_max_tokens: int = Field(default=8192)
    agent_max_steps: int = Field(default=50)

    # Judge settings
    judge_enabled: bool = Field(default=True)
    judge_llm_provider: str = Field(default='claude_cli')
    judge_llm_model: str = Field(default='sonnet')

    # Explore agent
    explore_codebase_root: str = Field(default='.')

    # Quiescence / burst detection
    conditional_trigger_ratio: float = Field(default=0.33)
    burst_window_seconds: float = Field(default=30.0)
    burst_cooldown_seconds: float = Field(default=150.0)
    stale_lock_seconds: float = Field(default=7200.0)

    # Timeouts
    tool_timeout_seconds: float = Field(default=120.0)
    stage_timeout_seconds: int = Field(default=3600)
    cycle_timeout_seconds: int = Field(default=21600)

    # Safety
    max_mutations_per_stage: int = Field(default=50)
    halt_on_judge_serious: bool = Field(default=True)

    # Escalation
    escalation_port: int = Field(default=8103)
    escalation_host: str = Field(default='127.0.0.1')
    escalation_queue_dir: str = Field(default='./data/reconciliation/escalations')

    # Tiered models
    sonnet_model: str = Field(default='sonnet')
    opus_model: str = Field(default='opus')
    opus_threshold_ratio: float = Field(default=1.5)
    sonnet_episode_limit: int = Field(default=125)
    sonnet_memory_limit: int = Field(default=250)
    opus_episode_limit: int = Field(default=500)
    opus_memory_limit: int = Field(default=1000)

    # Usage cap detection and multi-account failover
    usage_cap: UsageCapConfig = Field(default_factory=UsageCapConfig)


# --- Top-level ---

class FusedMemoryConfig(BaseSettings):
    """Fused Memory configuration with YAML and environment support."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    graphiti: GraphitiBackendConfig = Field(default_factory=GraphitiBackendConfig)
    mem0: Mem0BackendConfig = Field(default_factory=Mem0BackendConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    taskmaster: TaskmasterConfig | None = Field(default=None)
    reconciliation: ReconciliationConfig = Field(default_factory=ReconciliationConfig)

    model_config = SettingsConfigDict(
        env_prefix='',
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
        config_path = Path(os.environ.get('CONFIG_PATH', 'config/config.yaml'))
        yaml_settings = YamlSettingsSource(settings_cls, config_path)
        return (init_settings, env_settings, yaml_settings, dotenv_settings)
