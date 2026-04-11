"""Configuration schemas with pydantic-settings and YAML support."""

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
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

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        # Resolution handled in __call__; satisfy abstract interface with no-op.
        return (None, field_name, False)

    def __call__(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path, encoding='utf-8') as f:
                raw_config = yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError) as e:
            raise RuntimeError(f'Failed to load configuration from {self.config_path}: {e}') from e
        try:
            return self._expand_env_vars(raw_config)
        except Exception as e:
            raise RuntimeError(
                f'Failed to expand environment variables in {self.config_path}: {e}'
            ) from e


# --- Server ---

class ServerConfig(BaseModel):
    """Server configuration."""

    transport: Literal['http', 'stdio', 'sse'] = Field(default='http', description='Transport: http, stdio, or sse')
    host: str = Field(default='0.0.0.0', description='Server host')
    port: int = Field(default=8002, description='Server port (canonical: 8002)')
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
    # database is ignored in shared-server mode — graph name is derived from
    # group_id at request time.  Kept for backward-compat with existing configs.
    database: str | None = None


class GraphitiBackendConfig(BaseModel):
    """Graphiti / FalkorDB backend configuration."""

    provider: Literal['falkordb'] = Field(default='falkordb')
    falkordb: FalkorDBProviderConfig = Field(default_factory=FalkorDBProviderConfig)


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
    tool_mode: str = Field(default='all')


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

    # Token-budget context assembly
    token_budget: int = Field(default=65_000)
    context_search_limit: int = Field(default=5)
    context_fetch_batch_size: int = Field(default=10)

    # Usage cap detection and multi-account failover
    usage_cap: UsageCapConfig = Field(default_factory=UsageCapConfig)


class CuratorConfig(BaseModel):
    """Task curator gate — LLM-judged drop/combine/create decision on task creation.

    Replaces the title-only threshold dedup with a richer corpus + LLM call so
    reviewer-spawned tasks can be combined into existing pending work instead of
    fragmenting the backlog. See docs/reify-task-fragmentation-report-2026-04-11.txt
    for the motivating analysis.
    """

    enabled: bool = Field(default=True)
    model: str = Field(default='sonnet')
    timeout_seconds: float = Field(default=60.0)
    max_budget_usd: float = Field(default=0.30)

    # Corpus caps — see design notes in shared/docs (the four-stream pool).
    pool_module_cap: int = Field(default=15)
    pool_embedding_cap: int = Field(default=10)
    pool_dependency_cap: int = Field(default=3)
    pool_total_cap: int = Field(default=30)

    # Lock-key depth must match the scheduler's lock_depth to make module-pool
    # matching scheduler-consistent. Default 2 matches shared.locking defaults.
    lock_depth: int = Field(default=2)

    # Idempotency cache: skip re-invoking the LLM for the same candidate payload
    # if a decision was already rendered within this window.
    idempotency_ttl_seconds: float = Field(default=600.0)

    # Entry payload limits (applied per pool entry; whole entries trimmed, not
    # truncated — see design notes on preserving concrete code references).
    entry_description_chars: int = Field(default=500)
    entry_details_chars: int = Field(default=1500)


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
    curator: CuratorConfig = Field(default_factory=CuratorConfig)

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
