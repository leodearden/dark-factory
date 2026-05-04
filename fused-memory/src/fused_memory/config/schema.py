"""Configuration schemas with pydantic-settings and YAML support."""

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
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
    stateless_http: bool = Field(default=True, description='Stateless HTTP mode (no sessions)')
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
    """Configuration for the SQLite-backed task store.

    The class name is preserved for back-compat with consumers that still
    import :class:`TaskmasterConfig`. Only ``project_root`` and
    ``tool_mode`` are read by :class:`SqliteTaskBackend`; the other fields
    are kept dormant so existing config files don't fail to load.
    """

    transport: str = Field(default='stdio', description='Unused after SQLite cutover')
    command: str = Field(default='node', description='Unused after SQLite cutover')
    args: list[str] = Field(default_factory=list, description='Unused after SQLite cutover')
    cwd: str = Field(default='')
    http_url: str = Field(default='')
    project_root: str = Field(default='.')
    tool_mode: str = Field(default='all')


# --- Reconciliation ---

class ReconciliationConfig(BaseModel):
    """Sleep mode reconciliation settings."""

    @model_validator(mode='before')
    @classmethod
    def _reject_legacy_bulk_reset_threshold(cls, data: object) -> object:
        """Detect the legacy ``bulk_reset_guard_threshold`` key and raise.

        In task 1016 the single shared threshold was split into two independent
        per-kind fields.  The old key name is unknown to pydantic and would be
        silently dropped by ``extra='ignore'``, leaving the done→pending data-
        loss guard at the default of 10 regardless of any operator tuning.
        Because this is a security-relevant guard, silent default-fallback is
        the worst failure mode — raise instead so the operator knows to update
        their config to ``bulk_reset_guard_done_to_pending_threshold``.
        """
        if isinstance(data, dict) and 'bulk_reset_guard_threshold' in data:
            raise ValueError(
                "ReconciliationConfig: 'bulk_reset_guard_threshold' is no longer "
                "a valid field (renamed in task 1016). Use "
                "'bulk_reset_guard_done_to_pending_threshold' for the done→pending "
                "threshold (default 10) and/or "
                "'bulk_reset_guard_in_progress_to_pending_threshold' for the "
                "in-progress→pending threshold (default 100). The old key would "
                "be silently dropped, leaving the data-loss guard at its default "
                "threshold regardless of your configured value."
            )
        return data

    enabled: bool = Field(default=True)
    data_dir: str = Field(default='./data/reconciliation')

    # Buffer triggers
    buffer_size_threshold: int = Field(default=10)
    max_staleness_seconds: int = Field(default=1800)

    # WP-B: fire-and-forget in-memory event queue sits in front of the
    # SQLite event buffer on the MCP write path. Capacity bounds how many
    # events can sit in memory waiting for the drainer to catch up; overflow
    # is dead-lettered to ``data/reconciliation/event_dead_letter.jsonl``.
    event_queue_capacity: int = Field(default=10_000)
    event_queue_retry_initial_seconds: float = Field(default=0.1)
    event_queue_retry_max_seconds: float = Field(default=30.0)
    event_queue_shutdown_flush_seconds: float = Field(default=10.0)
    # Dead-letter JSONL rotation: rotate when the file reaches max_bytes,
    # keeping at most keep_rotations archived copies (oldest is dropped).
    # Default: 10 MB cap, keep 3 rotations (~40 MB max retention).
    event_dead_letter_max_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    event_dead_letter_keep_rotations: int = Field(default=3, ge=0)

    # WP-C: SQLite drainer watchdog. Logs ERROR with structured diagnostics
    # when the drainer hasn't committed in `stall_threshold` and the queue
    # is non-empty. Re-arms after `rearm_after` so a persistent wedge logs
    # at most once per window.
    event_queue_watchdog_enabled: bool = Field(default=True)
    event_queue_watchdog_check_interval_seconds: float = Field(default=30.0)
    event_queue_watchdog_stall_threshold_seconds: float = Field(default=120.0)
    event_queue_watchdog_rearm_after_seconds: float = Field(default=600.0)

    # WP-D: bounded-backlog escalation / rejection policy. Mutating MCP tools
    # return a structured ReconciliationBacklogExceeded error once the per-
    # project buffered count + queue depth exceeds the hard limit; when an
    # orchestrator is live for the project, an L1 escalation JSON is written
    # instead. Rate-limited per project to avoid spam.
    backlog_hard_limit: int = Field(default=500)
    backlog_escalation_rate_limit_seconds: float = Field(default=900.0)

    # Defence-in-depth bulk-reset circuit-breaker (task 918, refined task 1016).
    # Two autopilot_video incidents (2026-04-21, 2026-04-22) pushed large
    # batches of tasks from done/in-progress back to pending despite the
    # orchestrator's _safe_stash_pop_with_recovery fix being deployed.  This
    # guard limits blast radius when the primary prevention fails.
    #
    # The guard tracks two reversal kinds with independent counters and
    # thresholds sharing a single sliding window and rate limit:
    #   done→pending        (bulk_reset_guard_done_to_pending_threshold, default 10)
    #     — catches the March-2026 advance_main data-loss pattern (task 918).
    #   in-progress→pending (bulk_reset_guard_in_progress_to_pending_threshold, default 100)
    #     — catches pathological runaways while allowing the 27-task startup
    #       stranded-task reconcile seen in the 2026-04-24 reify incident
    #       (esc-bulk-reset-reify-2026-04-24T070944_6456580000).
    #
    # The shared per-project escalation rate limit (bulk_reset_guard_escalation_
    # rate_limit_seconds) prevents the escalations directory from filling up
    # when both kinds trip in quick succession; the escalation filename and JSON
    # body include a kind slug so operators can distinguish at a glance.
    bulk_reset_guard_enabled: bool = Field(default=True)
    bulk_reset_guard_done_to_pending_threshold: int = Field(default=10, ge=1)
    bulk_reset_guard_in_progress_to_pending_threshold: int = Field(default=100, ge=1)
    bulk_reset_guard_window_seconds: float = Field(default=60.0, gt=0)
    bulk_reset_guard_escalation_rate_limit_seconds: float = Field(default=900.0, ge=0)
    bulk_reset_guard_write_failure_backoff_seconds: float = Field(
        default=60.0,
        ge=0,
        description=(
            'Minimum seconds between escalation-write retries after an OSError '
            'from mkdir/write_text. Primarily for operators on flaky mounts. '
            'Zero disables the backoff (retry every trip). Per task 979.'
        ),
    )

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
    tool_timeout_seconds: float = Field(default=120.0, gt=0)
    stage_timeout_seconds: int = Field(default=3600, gt=0)
    cycle_timeout_seconds: int = Field(default=21600, gt=0)
    stale_run_recovery_seconds: int = Field(
        default=600,
        gt=0,
        description='Runs with started_at older than this are recovered on startup if their lock is stale',
    )
    # Per-CLI-invocation wall-clock budgets — semantically distinct from
    # stage_timeout_seconds (the outer stage-level guard).  Task 881 accidentally
    # widened these to 3600s by delegating to stage_timeout_seconds; these fields
    # restore the pre-881 per-call ceilings independently.
    agent_cli_timeout_seconds: int = Field(
        default=180,
        gt=0,
        description=(
            'Wall-clock budget for a single agent_loop._call_claude_cli invocation. '
            'Distinct from stage_timeout_seconds (outer stage guard). '
            'Restores the pre-881 hard-coded 180s ceiling.'
        ),
    )
    judge_cli_timeout_seconds: int = Field(
        default=600,
        gt=0,
        description=(
            'Wall-clock budget for a single judge._call_judge_cli invocation. '
            'Distinct from stage_timeout_seconds (outer stage guard). '
            'Restores the pre-881 hard-coded 600s ceiling.'
        ),
    )
    @model_validator(mode='after')
    def _validate_cli_timeouts_within_stage(self) -> 'ReconciliationConfig':
        """Enforce timeout hierarchy: per-CLI budgets ≤ stage guard ≤ cycle guard."""
        if self.agent_cli_timeout_seconds > self.stage_timeout_seconds:
            raise ValueError(
                f'agent_cli_timeout_seconds ({self.agent_cli_timeout_seconds}) must be '
                f'<= stage_timeout_seconds ({self.stage_timeout_seconds}): '
                'the per-call budget cannot exceed the outer stage guard.'
            )
        if self.judge_cli_timeout_seconds > self.stage_timeout_seconds:
            raise ValueError(
                f'judge_cli_timeout_seconds ({self.judge_cli_timeout_seconds}) must be '
                f'<= stage_timeout_seconds ({self.stage_timeout_seconds}): '
                'the per-call budget cannot exceed the outer stage guard.'
            )
        if self.stage_timeout_seconds > self.cycle_timeout_seconds:
            raise ValueError(
                f'stage_timeout_seconds ({self.stage_timeout_seconds}) must be '
                f'<= cycle_timeout_seconds ({self.cycle_timeout_seconds}): '
                'a stage that outlives its enclosing cycle is nonsensical.'
            )
        return self

    # Safety
    max_mutations_per_stage: int = Field(default=50)
    halt_on_judge_serious: bool = Field(default=True)

    # Done-provenance gate — when True, set_task_status(done) rejects calls
    # lacking done_provenance={commit?, note?}. Default False during phased
    # rollout: warn-only so existing callers (orchestrator, steward, interactive
    # sessions) can be updated to pass provenance before enforcement flips on.
    require_done_provenance: bool = Field(default=False)

    # Judge-halt trend detector. A halt fires when the most recent
    # `halt_trend_consecutive_required` verdicts are all non-ok AND at least
    # `halt_trend_moderate_count` non-ok verdicts sit within the last
    # `halt_trend_window_hours`. Consecutive-most-recent is what prevents a
    # single scattered history of old moderates from keeping a project halted
    # forever; the time window bounds blast radius to the recent past.
    halt_trend_window_hours: float = Field(default=6.0)
    halt_trend_moderate_count: int = Field(default=5)
    halt_trend_consecutive_required: int = Field(default=3)
    # Post-unhalt grace: the first N cycles after an unhalt skip the trend
    # detector, so a manual unhalt has a chance to accumulate fresh verdicts
    # instead of re-halting on the next tick from stale history.
    halt_grace_cycles: int = Field(default=3)
    # Cooldown after a halt fires: even if the trend condition is still true,
    # do not re-halt within this window. Belt-and-braces on top of grace_cycles
    # in case the operator intervenes mid-cycle.
    halt_cooldown_seconds: float = Field(default=1800.0)

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


class TicketJanitorConfig(BaseModel):
    """Background sweep that surfaces failed tickets to the orchestrator.

    Replaces the per-call ``resolve_ticket`` wait the steward / deep_reviewer
    used to chain after every ``submit_task`` (drop plan). On a fixed
    interval the janitor terminalises expired pending tickets, batches
    failed rows by ``(project_id, task_id, escalation_id)``, and submits one
    info-severity ``ticket_failure`` escalation per batch.
    """

    enabled: bool = Field(default=True)
    interval_seconds: float = Field(default=60.0)
    cooldown_seconds: float = Field(default=3600.0)
    batch_limit: int = Field(default=100)


class CuratorConfig(BaseModel):
    """Task curator gate — LLM-judged drop/combine/create decision on task creation.

    Replaces the title-only threshold dedup with a richer corpus + LLM call so
    reviewer-spawned tasks can be combined into existing pending work instead of
    fragmenting the backlog. See docs/reify-task-fragmentation-report-2026-04-11.txt
    for the motivating analysis.
    """

    enabled: bool = Field(default=True)
    model: str = Field(default='sonnet')
    timeout_seconds: float = Field(default=300.0)
    max_budget_usd: float = Field(default=0.30)
    # Per-call turn cap. Must be ≥ 3: schema burns one tool-use turn, an
    # optional reasoning turn may precede it, and the final assistant
    # response is the third. 8 leaves headroom for harder combine-vs-create
    # decisions without tripping ``error_max_turns``.
    max_turns: int = Field(default=8, ge=3)

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

    # Batch-curator knobs — the worker drains up to batch_max tickets per
    # round-trip; timeout, turns, and budget scale linearly with N and are
    # clamped at the caps so no single batch blocks resolve_ticket callers
    # indefinitely.
    # Formula: timeout = min(timeout_seconds + per_item_slack_seconds * (N-1),
    #                        batch_timeout_cap_seconds)
    #          max_turns = min(max_turns + per_item_turns * (N - 1),
    #                          batch_turns_cap)
    #          budget    = min(max_budget_usd + per_item_budget_usd * (N-1),
    #                          batch_budget_cap_usd)
    batch_max: int = Field(default=5)
    per_item_slack_seconds: float = Field(default=30.0)
    per_item_turns: int = Field(default=1)
    per_item_budget_usd: float = Field(default=0.30)
    batch_timeout_cap_seconds: float = Field(default=540.0)
    batch_turns_cap: int = Field(default=10)
    batch_budget_cap_usd: float = Field(default=2.00)
    # Soft cap on batched user-prompt tokens: the worker stops adding tickets
    # to a batch when the next ticket's prepared section would push the
    # estimate over this threshold. A single oversized ticket still fires as
    # a size-1 batch (first ticket is always included). ~50K tokens ≈ 200KB
    # of prompt — aggressive enough to keep batch fan-in high; the scaled
    # budget (per_item_budget_usd) and the bisecting fallback are the safety
    # net for occasional stalls. Tunable.
    batch_token_threshold: int = Field(default=50_000)

    # Background janitor that surfaces failed tickets to the orchestrator.
    janitor: TicketJanitorConfig = Field(default_factory=TicketJanitorConfig)


# --- Top-level ---


def _default_curator_usage_cap() -> UsageCapConfig:
    """Construct the curator's default :class:`UsageCapConfig`.

    Reads ``USAGE_ACCOUNTS_FILE`` if set (aligns with the orchestrator / eval
    runner), otherwise falls back to the canonical shared accounts file. The
    fused-memory server inherits the unset shell env, so it gets the shared
    G→F→E→C→D pool used by the orchestrator — see memory note
    ``project_eval_account_isolation.md``.
    """
    return UsageCapConfig(
        accounts_file=os.environ.get(
            'USAGE_ACCOUNTS_FILE',
            '/home/leo/src/dark-factory/config/usage-accounts.yaml',
        ),
    )


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
    # Curator's usage-gate pool; independent from ``reconciliation.usage_cap``
    # because each UsageGate instance tracks cap state per-process / per-loop.
    usage_cap: UsageCapConfig | None = Field(default_factory=_default_curator_usage_cap)

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
