"""Configuration schemas with pydantic-settings and YAML support."""

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from shared.config_models import UsageCapConfig

# Config path used when the ``CONFIG_PATH`` env var is unset. Single-sourced here
# so both ``settings_customise_sources`` (the actual loader) and the reload_config
# MCP tool's ``config_path`` disposition field agree on the file actually read —
# otherwise the tool could report ``config_path=None`` while the reload silently
# read and applied leaves from this default file.
DEFAULT_CONFIG_PATH = 'config/config.yaml'


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

# Mirrors fused_memory.server.main._FORCE_EXIT_BUDGET and
# _SYSTEMD_TIMEOUT_STOP_SECS (survey finding D1). Duplicated here as literals
# rather than imported: main.py imports FusedMemoryConfig from this module, so
# importing main.py here would be circular. Parity with main.py's constants is
# enforced by TestShutdownBudgetArithmetic::test_schema_mirror_constants_match_main
# in tests/test_server_shutdown.py — update both sides together.
_MIRROR_FORCE_EXIT_BUDGET = 75.0
_MIRROR_SYSTEMD_TIMEOUT_STOP_SECS = 90.0


class ServerConfig(BaseModel):
    """Server configuration."""

    transport: Literal['http', 'stdio', 'sse'] = Field(default='http', description='Transport: http, stdio, or sse')
    host: str = Field(default='0.0.0.0', description='Server host')
    port: int = Field(default=8002, description='Server port (canonical: 8002)')
    stateless_http: bool = Field(default=True, description='Stateless HTTP mode (no sessions)')
    json_response: bool = Field(default=False, description='JSON responses instead of SSE')
    keepalive_timeout: int = Field(default=30, description='HTTP keep-alive timeout in seconds')
    graceful_shutdown_timeout: int = Field(
        default=10,
        description=(
            'Bounds uvicorn.Server timeout_graceful_shutdown (seconds the server '
            'waits for in-flight connections to finish before forcing shutdown). '
            'Without this bound the wait is unbounded and _graceful_shutdown/the '
            'force-exit watchdog become unreachable. default + _FORCE_EXIT_BUDGET '
            '(75s) must stay under systemd TimeoutStopSec (90s) so the in-process '
            'watchdog provably fires before systemd SIGKILLs the cgroup.'
        ),
    )
    thread_warn_threshold: int = Field(
        default=60,
        description=(
            'Threshold above which thread_monitor emits a WARNING (with snapshot breakdown). '
            'Above the observed ~40-thread normal but low enough to flag a real leak quickly.'
        ),
    )
    recon_report_port: int = Field(
        default=8003,
        description='Second uvicorn port for the recon_report MCP namespace (PRD §12 OQ1)',
    )

    @model_validator(mode='after')
    def _validate_port_uniqueness(self) -> 'ServerConfig':
        """Ensure recon_report_port and port are distinct.

        If an operator misconfigures both to the same value the second uvicorn
        server fails to bind, asyncio.gather raises at startup, and the failure
        mode would be non-obvious.  Catching it at config-load time surfaces a
        clear error message immediately.
        """
        if self.recon_report_port == self.port:
            raise ValueError(
                f'recon_report_port ({self.recon_report_port}) must differ from '
                f'port ({self.port}): both uvicorn servers cannot bind to the same port.'
            )
        return self

    @model_validator(mode='after')
    def _validate_graceful_shutdown_timeout_budget(self) -> 'ServerConfig':
        """Fail fast on a graceful_shutdown_timeout that breaks the D1 shutdown
        budget invariant, instead of silently discovering it at systemd SIGKILL
        time.

        The bounded uvicorn graceful-shutdown wait runs BEFORE
        _shutdown_with_watchdog arms the force-exit timer, so worst-case
        shutdown is approximately graceful_shutdown_timeout + _FORCE_EXIT_BUDGET.
        That must stay strictly under systemd's TimeoutStopSec so the
        in-process force-exit watchdog provably fires before systemd SIGKILLs
        the cgroup (preserving exit-code control: exit 0 for operator stop).
        The default (10) satisfies this with 5s of margin; an operator
        override is otherwise unconstrained and could silently violate it.
        """
        if self.graceful_shutdown_timeout <= 0:
            raise ValueError(
                f'graceful_shutdown_timeout ({self.graceful_shutdown_timeout}) must '
                'be a positive number of seconds.'
            )
        budget = self.graceful_shutdown_timeout + _MIRROR_FORCE_EXIT_BUDGET
        if budget >= _MIRROR_SYSTEMD_TIMEOUT_STOP_SECS:
            raise ValueError(
                f'graceful_shutdown_timeout ({self.graceful_shutdown_timeout}) + '
                f'force-exit budget ({_MIRROR_FORCE_EXIT_BUDGET}s) = {budget}s, which '
                f'is not less than systemd TimeoutStopSec '
                f'({_MIRROR_SYSTEMD_TIMEOUT_STOP_SECS}s). systemd would SIGKILL the '
                'cgroup before the in-process force-exit watchdog fires, losing '
                'exit-code control. Lower graceful_shutdown_timeout so the sum stays '
                'under the systemd timeout.'
            )
        return self


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
    # Error-aware retry budget (task 1936): known-transient errors (e.g.
    # graphiti_core's NodeNotFoundError — a graph-visibility race) get this
    # longer attempts ceiling instead of max_attempts. Keep this default list
    # in sync with durable_queue.DEFAULT_TRANSIENT_ERROR_NAMES — drift is
    # caught by test_config_schema.py::
    # TestQueueConfigTransientErrorFields::test_transient_error_names_matches_durable_queue_default.
    transient_max_attempts: int = Field(default=12)
    transient_error_names: list[str] = Field(default_factory=lambda: [
        'NodeNotFoundError',
        'EdgeNotFoundError',
        'EdgesNotFoundError',
        'TimeoutError',
        'ConnectionError',
        'ConnectionResetError',
        'ServerDisconnectedError',
        'OperationalError',
    ])
    write_timeout_seconds: float = Field(default=120.0)
    backend_read_timeout_seconds: float = Field(default=30.0)
    backend_write_timeout_seconds: float = Field(default=120.0)
    search_timeout_seconds: float = Field(default=15.0)
    data_dir: str = Field(default='./data/queue')

    @model_validator(mode='after')
    def _validate_transient_max_attempts(self) -> 'QueueConfig':
        """A known-transient error must never get a shorter retry budget than
        an ordinary error — the inverse of this feature's intent (task 1936).

        The ``DurableWriteQueue`` constructor only falls back to
        ``max(max_attempts, 12)`` when ``transient_max_attempts`` is omitted
        entirely; an explicit-but-too-low override would otherwise pass
        through uncaught. Enforce the invariant here, at config-load time.
        """
        if self.transient_max_attempts < self.max_attempts:
            raise ValueError(
                f'transient_max_attempts ({self.transient_max_attempts}) must be >= '
                f'max_attempts ({self.max_attempts}): a known-transient error must not '
                'receive a shorter retry budget than an ordinary error.'
            )
        return self


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


# --- Task metadata write-boundary validation (task 2162, W3-β) ---

class TaskMetadataConfig(BaseModel):
    """Governs ``SqliteTaskBackend``'s write-boundary validation of ``metadata``.

    Backed by ``shared.task_metadata.parse_metadata`` (the single cross-process
    TaskMetadata schema/parser). This is a top-level config section — NOT
    nested under ``taskmaster`` — because it governs a schema shared with the
    orchestrator, not only the SQLite task store.
    """

    enforce: bool = Field(
        default=False,
        description=(
            'RED-TIER / restart-only: warn-mode when False (default) — '
            'validation violations emit a task_metadata.schema_warning log '
            'line and the write proceeds; True rejects the write with '
            'ValidationError. Not hot-reloadable.'
        ),
    )


# --- Mem0 metadata write-boundary validation (task 3195, leaf β) ---

class MemoryMetadataConfig(BaseModel):
    """Governs ``MemoryService``'s write-boundary validation of Mem0 ``metadata``.

    Backed by ``fused_memory.memory_metadata`` (the single normative home for
    the Mem0 metadata vocabulary — PRD ``docs/prds/memory-metadata-vocabulary.md``
    V1 / INV-5). This is a top-level config section — NOT nested under
    ``reconciliation`` or ``taskmaster`` — because it governs a vocabulary
    shared beyond any one backend or caller, mirroring ``TaskMetadataConfig``
    directly above. PRD D3 names that section as its precedent for
    "census first, tiers later", so this follows it rather than inventing a
    second shape.

    ``extra='forbid'`` so a mistyped leaf fails loud at config load/reload
    rather than silently doing nothing: under ``extra='ignore'`` an operator
    who typed ``enforce_kind_regsitry`` would get a silently-dropped key and
    believe enforcement was on.

    Reload tier: ``reload.py``'s ``RELOADABLE_FIELDS`` is an opt-IN allowlist,
    so every leaf here is restart-required by default and none is listed. That
    is correct for BOTH pairs, not just the enforce flags — the storm-tuning
    leaves are read once when ``MemoryService.__init__`` constructs the
    ``UnknownKeyStormDetector``, so they are captured by value and a hot
    reload genuinely would not take effect. Allowlisting them would advertise
    a reload that silently does nothing.
    """

    model_config = ConfigDict(extra='forbid')

    enforce: bool = Field(
        default=False,
        description=(
            'RED-TIER / restart-only: warn-mode when False (default) — '
            'validation violations emit a memory_metadata.schema_warning log '
            'line and the write proceeds; True rejects the write with '
            'MemoryMetadataValidationError. Not hot-reloadable.'
        ),
    )
    enforce_kind_registry: bool = Field(
        default=False,
        description=(
            'RED-TIER / restart-only: whether an unknown metadata.kind is '
            'FATAL (rejectable) rather than census-only. Carved out from '
            "`enforce` and left OFF even after `enforce` flips, because it is "
            'specifically this check whose safety premise leaf α measured '
            'FALSE: PRD D3 assumed kind writers are in-repo code + prompts, '
            'but 242 of the 329 live kind values are singletons, i.e. the '
            'population is agent-invented free text and open in practice. '
            'Flipping this on day one would turn every newly invented kind '
            'into a hard memory-write failure on the live fleet. See PRD §10 '
            'open question 1. Not hot-reloadable.'
        ),
    )
    unknown_key_storm_threshold: int = Field(
        default=50,
        ge=1,
        description=(
            'Unknown-key census warnings from ONE (project_id, agent_id) '
            'within the window before an escalation is filed. Per-writer, not '
            'global: a storm is a DRIFTING WRITER, and a global counter would '
            'fire on healthy fleet traffic given the 1,627-key baseline while '
            'never identifying the culprit. Calibrated from leaf α\'s census '
            '— the entire sub-1000-count key tail amounts to only a few '
            'thousand occurrences across the whole corpus lifetime, so 50 '
            'from one writer in 5 minutes is orders of magnitude above any '
            'legitimate rate. Restart-only: read once when MemoryService '
            'builds the detector.'
        ),
    )
    unknown_key_storm_window_seconds: int = Field(
        default=300,
        ge=1,
        description=(
            'Rolling window for unknown_key_storm_threshold. Warns older than '
            'this are dropped, so a slow trickle never accumulates into a '
            'false storm. Restart-only: read once when MemoryService builds '
            'the detector.'
        ),
    )


# --- Task status transition authority (task 2175, rho1b) ---

class TaskStatusConfig(BaseModel):
    """Governs the ``TaskInterceptor`` transition-legality gate (Table A).

    Backed by ``shared.task_transitions.is_legal_transition`` (the shared
    (from, to, actor) legality table). This is a top-level config section —
    NOT nested under ``taskmaster`` — because it governs a schema shared with
    the orchestrator, mirroring ``TaskMetadataConfig``.
    """

    enforce_transitions: bool = Field(
        default=False,
        description=(
            "RED-TIER/restart-only log-mode gate: False (default) => an "
            "illegal (from,to,actor) transition emits an 'illegal_transition "
            "would-reject' WARNING at the interceptor and the write "
            "proceeds; True => a typed illegal_transition rejection. "
            "Flipped only after the Gamma soak; not hot-reloadable."
        ),
    )


# --- Reconciliation ---

class ProceduralTopicCluster(BaseModel):
    """One known-contradictory procedural_knowledge topic for the write-time topic guard.

    A cluster is matched deterministically: an incoming ``procedural_knowledge``
    add_memory write whose (lowercased) content contains at least
    ``min_phrase_hits`` DISTINCT ``phrases`` is soft-blocked, routing the writer
    to consolidate/update the existing entries or add context to the human gate
    task named in ``hint`` -- instead of accumulating another paraphrased,
    below-cosine-threshold restatement the semantic near-dup guard cannot catch
    (task 2845). ``extra='forbid'`` so a mistyped cluster key fails loud at config
    load/reload, never silently matches nothing.
    """

    model_config = ConfigDict(extra='forbid')

    topic_id: str = Field(description='Stable identifier for this topic cluster.')
    phrases: list[str] = Field(
        description=(
            'Identifying substring phrases, matched case-insensitively. A write '
            'containing >= min_phrase_hits DISTINCT phrases matches this cluster.'
        ),
    )
    min_phrase_hits: int = Field(
        default=2,
        ge=1,
        description=(
            'Number of DISTINCT phrases that must appear before a write is blocked. '
            'Default 2 so a single incidental keyword never triggers a false block.'
        ),
    )
    hint: str = Field(
        default='',
        description=(
            'Optional remediation hint (e.g. the human gate task to route to). '
            'When empty, the guard substitutes a module-default hint.'
        ),
    )


def _default_topic_guard_clusters() -> list[ProceduralTopicCluster]:
    """Seed the known-contradictory/recurring procedural_knowledge topic clusters.

    The first three are recurring procedural_knowledge topics that grew
    several contradictory or paraphrased entries each because every
    restatement scored BELOW the cosine near-dup threshold. Seeding the two
    eval-worktree clusters closes task 2845 (gate 2841) AND the sibling
    venv-shadowing cluster (gate 2844); the pytest-xdist cluster closes task
    2974 -- it is the topic that originally motivated the guard (9
    duplicates consolidated into canonical memory 8bb3eb15) but was never
    itself seeded. Two more clusters (task 3013) seed architect-related
    families: report_task_already_done / main-reachable-commit, registered
    prospectively ahead of still-open gate 3011; and plan-revalidation after
    requeue/lock, gated to already-adjudicated task 2973 (canonical Mem0
    entries 6a96a020 / 974b0adb).
    Operator-overridable / tunable via config (green-tier hot-reloadable).
    """
    return [
        ProceduralTopicCluster(
            topic_id='eval-worktree-plan-tools-missing',
            phrases=[
                'plan-tools',
                'plan_tools',
                'plan.json',
                'eval-worktree',
                'eval worktree',
                'create_plan',
                'add_plan_step',
            ],
            min_phrase_hits=2,
            hint=(
                'Known-contradictory topic (plan-tools MCP server missing in the '
                'eval worktree) gated to human task 2841. Do NOT add another entry '
                '-- update/consolidate the existing entries, or add context to gate '
                'task 2841.'
            ),
        ),
        ProceduralTopicCluster(
            topic_id='eval-worktree-venv-shadowing',
            # Reviewer (robustness): the earlier seed paired short, generic tokens
            # ('shadow', 'pyright', 'conftest', 'site-packages', bare '.venv') that
            # co-occur in unrelated Python-env notes, so a genuine pyright/conftest
            # or a plain '.venv'/'site-packages' gotcha could reach min_phrase_hits
            # on its own and be mis-routed to gate 2844. Narrowed to eval-worktree
            # anchors plus distinctive multi-word phrases, so a match now requires
            # the eval-worktree context or an unambiguous venv-shadowing phrase
            # (mirrors the more distinctive plan-tools cluster above). 'shadow' as a
            # bare 6-char substring (fires on 'shadowing'/'overshadow'/'shadow copy')
            # is replaced by the longer 'venv shadowing' / '.venv shadow'.
            phrases=[
                'eval-worktree',  # anchor; also substring-matches '.eval-worktrees'
                'eval worktree',  # anchor (space spelling)
                'editable install',
                'venv shadowing',
                '.venv shadow',
            ],
            min_phrase_hits=2,
            hint=(
                'Known-contradictory topic (venv / editable-install shadowing in the '
                'eval worktree) gated to human task 2844. Do NOT add another entry '
                '-- update/consolidate the existing entries, or add context to gate '
                'task 2844.'
            ),
        ),
        ProceduralTopicCluster(
            topic_id='pytest-xdist-serial-override',
            # This is the topic that ORIGINALLY MOTIVATED this guard (task 2845):
            # 9 paraphrased entries consolidated into canonical memory 8bb3eb15,
            # yet the cluster itself was never seeded. Phrases are literal
            # substrings drawn from 8bb3eb15's actual wording (addopts hardcodes
            # `-n auto --dist loadgroup --max-worker-restart=0`; `-n0` is the one
            # reliable serial-override workaround; `-p no:xdist` fails with
            # "unrecognized arguments").
            phrases=[
                'pytest-xdist',
                # Reviewer (robustness): bare '-n0' is a short 3-char substring
                # that could incidentally match an unrelated '-n0.5' / '-n01'
                # token. Verified against canonical memory 8bb3eb15's actual
                # text: neither 'pytest -n0' nor 'run -n0' occurs literally
                # there (only bare '-n0', twice) -- so anchoring would violate
                # this module's literal-substring-from-canonical-text
                # convention AND wouldn't even close the gap, since a prefix
                # anchor like 'pytest -n0' is still a substring of a
                # hypothetical 'pytest -n0.5'. Left bare; min_phrase_hits=2
                # already requires a second, unrelated phrase to co-occur
                # before a write is blocked, which covers the residual risk.
                '-n0',
                '--dist loadgroup',
                'max-worker-restart',
                '-p no:xdist',
            ],
            min_phrase_hits=2,
            hint=(
                'Known-recurring topic (pytest-xdist -n0 serial-override workaround '
                'for the hardcoded -n auto addopts in orchestrator/fused-memory '
                'pyproject.toml). Do NOT add another entry -- update/consolidate '
                'canonical memory 8bb3eb15-1133-4e7b-ac1f-5bac10329b51.'
            ),
        ),
        ProceduralTopicCluster(
            topic_id='architect-report-task-already-done-main-reachability',
            # Reviewer (robustness, task 3013): all four phrases are distinctive
            # literal identifiers/commands -- the report_task_already_done tool
            # name, the _handle_already_done_report handler name, the literal
            # git subcommand 'merge-base --is-ancestor', and the hyphenated
            # 'main-reachable' -- none of which is a short generic token that
            # could substring-match unrelated text, so no further narrowing
            # (unlike venv-shadowing above) is needed. Registered prospectively:
            # gate task 3011's 12-entry cluster is still open awaiting a
            # consolidation ruling, but this guard is forward-looking (it only
            # blocks NEW near-dup writes), so seeding it now stops that cluster
            # from growing further while 3011 is parked.
            phrases=[
                'report_task_already_done',
                'main-reachable',
                'merge-base --is-ancestor',
                '_handle_already_done_report',
            ],
            min_phrase_hits=2,
            hint=(
                'Known-contradictory topic (architect report_task_already_done '
                'requires a main-reachable commit) gated to human task 3011. Do '
                'NOT add another entry -- update/consolidate the existing entries, '
                'or add context to gate task 3011.'
            ),
        ),
        ProceduralTopicCluster(
            topic_id='architect-plan-revalidation-requeue-lock',
            # Reviewer (robustness, task 3013): phrases are distinctive
            # multi-word/literal substrings drawn verbatim from canonical Mem0
            # entries 6a96a020 (subcase plan_json_gitignore_wipe) and 974b0adb
            # (subcase lost_plan_reconstruction). Standalone generic tokens
            # ('lock', 'requeue', 'plan', 'commit', 'main', 'revalidate') are
            # deliberately excluded -- each could substring-match unrelated
            # notes on its own (the venv-shadowing over-match lesson above).
            # Bare 'create_plan'/'plan.json' are also excluded even though
            # they'd be on-topic: those already anchor the
            # eval-worktree-plan-tools-missing cluster, and reusing them here
            # would blur the two clusters and risk mis-routing a write to the
            # wrong gate task.
            phrases=[
                '.task/plan.json',
                'plan-revalidation',
                'requeue rebase',
                'lost-plan reconstruction',
                'committed TDD steps',
            ],
            min_phrase_hits=2,
            hint=(
                'Known-recurring topic (architect plan-revalidation after a '
                'requeue rebase or lock loss) gated to human task 2973 -- see '
                'canonical memory 6a96a020-6193-4a7e-82a6-4f27bcf5378e and '
                '974b0adb-ed54-44bd-aa12-aeaeae8b3ea6. Do NOT add another entry '
                '-- update/consolidate the existing entries, or add context to '
                'gate task 2973.'
            ),
        ),
    ]


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
    # Per-project override map keyed by project_id. Empty map = flat
    # backlog_hard_limit for all projects (default behaviour unchanged).
    # Example: {reify: 1500} raises reify's bound while leaving all others at 500.
    backlog_hard_limit_overrides: dict[str, int] = Field(default_factory=dict)

    @field_validator('backlog_hard_limit_overrides')
    @classmethod
    def _validate_overrides_positive(cls, v: dict[str, int]) -> dict[str, int]:
        """Reject non-positive override values.

        A zero or negative override makes every backlog check exceed the limit,
        producing constant escalations or rejections — the opposite of the
        feature's intent. Fail fast at config load rather than silently spamming
        escalations. Per-project overrides are hand-edited and therefore more
        error-prone than the single global default.
        """
        bad = {k: val for k, val in v.items() if val < 1}
        if bad:
            raise ValueError(
                f'backlog_hard_limit_overrides: all values must be >= 1 (positive); '
                f'got non-positive entries: {bad!r}'
            )
        return v

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

    # Feature flag gating construction of ReconLedgerStore (task 2219, PRD
    # plans/recon-reliability-prd.md §8.1) — the SQLite control-plane ledger
    # for recon markers/suppressions/cycle-summaries. Mirrors the
    # bulk_reset_guard_enabled / event_queue_watchdog_enabled convention:
    # defaults on, and only gates whether server/main.py constructs the
    # store inside the reconciliation-enabled branch.
    recon_ledger_enabled: bool = Field(default=True)

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

    # Defense-in-depth Landlock confinement of recon stage/remediation agents
    # (task 1935).  When enabled, every agent spawned through run_stage_via_cli
    # (Stage 1/2/3 + remediation passes) is confined by a kernel Landlock
    # ruleset (falling back to bwrap) that makes `/` read-only and DENIES writes
    # to every tracked source file in the repository.  The only writable paths
    # are the built-in agent scratch (/tmp, ~/.claude, /dev) plus the gitignored
    # root .task/ directory, preserving the agent's legitimate side-effects
    # (sysprompt/mcp temp files, OAuth state) while blocking hand-patching of
    # production code.
    #
    # sandbox_recon_agents: True  (default-on = safety posture; host kernel ≥5.13
    #   supports Landlock).  Set False only on a sandbox-less host where Landlock
    #   and bwrap are both unavailable — running unconfined is then an explicit
    #   operator opt-out.  Fail-CLOSED when True but no backend is available
    #   (run_stage_via_cli returns an error StageResult rather than running an
    #   unconfined agent).
    #
    #   VENV REQUIREMENT: confinement relies on `orchestrator` being importable at
    #   runtime via the shared uv-workspace venv (both fused-memory and orchestrator
    #   installed in the same venv).  If fused-memory runs in a venv that does NOT
    #   include orchestrator, the ImportError-to-RemediationSandboxUnavailable guard
    #   causes EVERY reconciliation stage to fail (fail-closed).  Operators on
    #   standalone deployments must install orchestrator or set this to False.
    #
    # sandbox_recon_writable_extras: additional paths to add to the writable set
    #   (e.g. a uvx/pip cache dir used by a stdio MCP server).  Empty by default;
    #   use only when an MCP server genuinely needs to write outside /tmp.
    sandbox_recon_agents: bool = Field(default=True)
    sandbox_recon_writable_extras: list[str] = Field(default_factory=list)

    # Quiescence / burst detection
    conditional_trigger_ratio: float = Field(default=0.33)
    burst_window_seconds: float = Field(default=30.0)
    burst_cooldown_seconds: float = Field(default=150.0)
    stale_lock_seconds: float = Field(default=7200.0)

    # Timeouts
    tool_timeout_seconds: float = Field(default=120.0, gt=0)
    stage_timeout_seconds: int = Field(default=3600, gt=0)
    cycle_timeout_seconds: int = Field(default=21600, gt=0)
    backlog_iteration_budget_seconds: int = Field(
        default=1800,
        gt=0,
        description=(
            'Cumulative per-invocation wall-clock budget (s) for '
            'BacklogIterator.run(). Checked between chunks (after at least one '
            'chunk has completed, so a single invocation always makes forward '
            'progress): once elapsed time since the invocation started reaches '
            'this budget, the iterator stops launching new chunks, skips the '
            'final consolidation pass, and leaves the remaining backlog '
            'buffered (not restored) so the next cycle — or an operator '
            'trigger_reconciliation — continues draining it. Must be '
            '<= cycle_timeout_seconds. Default 1800s mirrors '
            'stale_run_recovery_seconds. Task 2040 (bounds the previously '
            'unbounded BacklogIterator chunk loop).'
        ),
    )
    stale_run_recovery_seconds: int = Field(
        default=1800,
        gt=0,
        description=(
            'Age cutoff (s) for journal rows in status="running" before the '
            'reaper considers them orphaned.  The primary protection against '
            'reaping a live cycle is the lock-holder identity check in '
            'ReconciliationHarness._recover_stale_runs; this value is the '
            'palliative outer bound for when that primary guard cannot apply '
            '(brand-new cycle that has not acquired its lock yet, or a future '
            'regression on the primary guard).  Bumped 600 -> 1800 on '
            '2026-05-28 after observing real-world cycle envelopes: a full '
            'cycle measured at 29:58 (run 97b49a64), plus parent + remediation '
            'comfortably exceeding 30 minutes when actionable findings exist. '
            '1800 s gives clean separation from typical cycle length while '
            'still bounding orphan-detection latency well below the lock-'
            'staleness TTL of 7200 s.  See plans/recon-stale-recovery-rca.md. '
            'Green-tier hot-reloadable via the reload_config MCP tool: the '
            'reaper reads this live from the shared config object, so a reload '
            'is observed on the next reaper pass with no restart (task 2718).'
        ),
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
        """Enforce timeout hierarchy: per-CLI budgets ≤ stage guard ≤ cycle guard;
        backlog_iteration_budget_seconds ≤ cycle guard."""
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
        if self.backlog_iteration_budget_seconds > self.cycle_timeout_seconds:
            raise ValueError(
                f'backlog_iteration_budget_seconds ({self.backlog_iteration_budget_seconds}) '
                f'must be <= cycle_timeout_seconds ({self.cycle_timeout_seconds}): '
                'the backlog-iteration budget cannot exceed the outer cycle guard.'
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

    # Reopen-freshness gate (task 2674, PRD task alpha) — governs whether a
    # done-write citing commit-bearing done_provenance (kind in {"merged",
    # "found_on_main"}) on a task whose metadata.reopen_at is set gets
    # rejected when the evidence commit's committer date predates reopen_at
    # (i.e. the cited evidence is stale — from before the reopen). 'enforce'
    # (default, task 2680 / PRD task gamma) returns a typed
    # done_evidence_stale rejection; 'warn' is the explicit opt-out — it
    # logs a 'task_status.done_evidence_stale_warn' census WARNING and lets
    # the write proceed instead. Task alpha shipped 'warn'; task gamma flips
    # the shipped default to 'enforce' now that task beta (the orchestrator
    # consumer — ProvenanceConflictSink + StaleEvidenceRejection) has
    # landed. Mirrors the require_done_provenance / enforce_transitions
    # warn->enforce phased-rollout precedent above.
    reject_stale_done_evidence: Literal['warn', 'enforce'] = Field(
        default='enforce',
        description=(
            "Reopen-freshness gate mode. 'enforce' (default, task gamma): "
            "a stale-evidence done-write on a reopened task is rejected "
            "with a typed done_evidence_stale error. 'warn': the explicit "
            "opt-out — the write instead logs a "
            "'task_status.done_evidence_stale_warn' WARNING and proceeds. "
            "Task alpha shipped warn; task gamma flipped the shipped "
            "default to enforce after task beta (the orchestrator "
            "consumer) landed. NOTE: a present-but-invalid "
            "stale_evidence_override (malformed shape, or supplied by a "
            "recon-stage caller) always rejects with "
            "done_evidence_stale_override_invalid regardless of this "
            'setting — that one rejection is not gated by warn/enforce.'
        ),
    )

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
    # Auto-resume-after-cooldown (task 2920 deliverable c). When True, a project
    # the judge has halted is automatically unhalted once its `cooldown_until`
    # has passed (seeding the normal post-unhalt grace) and reconciliation
    # resumes; the judge re-halts immediately on the next serious verdict or
    # infra-failure threshold if the project is still sick, so a genuinely-broken
    # pipeline re-latches (and re-fires a distinct, now-loud halt escalation)
    # rather than resuming silently. False (default) keeps the legacy semantics:
    # a halt persists until an operator calls unhalt_reconciliation. Motivated by
    # the 2026-07-20 incident where a judge halt of dark_factory sat unhalted for
    # 2 days because cooldown_until expired and nothing acted on the expiry.
    # Restart-tier: read from config each cycle; deliberately NOT in
    # RELOADABLE_FIELDS.
    auto_unhalt_after_cooldown: bool = Field(default=False)
    # Judge transport/infra-failure tolerance (task 2947 ask a). Bounds the
    # number of CONSECUTIVE judge CLI transport/infra failures (api_error,
    # timeout, cap-wait exhaustion, etc. — NOT genuinely-malformed reachable
    # content) a project may accumulate before review_run applies a truthful
    # judge-unreachable halt and routes an infra_issue escalation. Below this
    # threshold each infra failure is bounded backoff: no verdict is stamped
    # and no halt fires (the transient outage is given room to recover). Any
    # successful transport (a parsed verdict) resets the per-project counter.
    # Gated on halt_on_judge_serious, same master switch as the serious-verdict
    # halt. ge=1 → a value of 1 halts on the first infra failure.
    judge_infra_max_consecutive_failures: int = Field(default=3, ge=1)

    # Escalation
    escalation_port: int = Field(default=8103)
    escalation_host: str = Field(default='127.0.0.1')
    escalation_queue_dir: str = Field(default='./data/reconciliation/escalations')

    # Aggregate storm alarm for dead_owner_shielded suppression bursts (task 1755 / PRD β).
    # When >=threshold suppressions occur within a rolling window_seconds, a single loud
    # 'recon_watchdog_kill_storm' escalation fires — the per-event suppression (task 1731)
    # is preserved. Default threshold=6 per 3600s (~= 24× the ~6/day benign-restart
    # baseline) ensures a single benign restart (count=1) never trips the storm.
    dead_owner_suppression_storm_threshold: int = Field(
        default=6,
        ge=1,
        description=(
            'Number of dead_owner_shielded recon_stale_run suppressions within '
            'dead_owner_suppression_storm_window_seconds that triggers a single '
            'loud recon_watchdog_kill_storm escalation. Default 6 per 3600s is '
            '~24× the ~6/day benign-restart baseline. Task 1755 / PRD β.'
        ),
    )
    dead_owner_suppression_storm_window_seconds: float = Field(
        default=3600.0,
        gt=0,
        description=(
            'Rolling time window (seconds) for counting dead_owner_shielded '
            'suppressions. A single storm alarm is emitted per window when the '
            'count crosses dead_owner_suppression_storm_threshold. Also doubles '
            'as the per-window rate limit: the alarm cannot re-fire within this '
            'window. Task 1755 / PRD β.'
        ),
    )

    # Interrupted-run resume knobs (task 2717 / PRD σ, rec 13). Gate the
    # startup adopt-and-resume pass (_resume_interrupted_runs): a fused-memory
    # restart mid-stage marks the run 'interrupted' (keeping its drained events
    # and per-run transcript), and the next startup --resume's the SAME run_id
    # at its stage_cursor. On ANY doubt the pass falls back to today's
    # failed+restore recovery path.
    resume_after_restart: bool = Field(
        default=True,
        description=(
            'Master switch for interrupted-run resume. When True, run_full_cycle '
            "marks a restart-cancelled run 'interrupted' (no restore_drained, "
            'transcript preserved) and the startup pass adopts and --resume\'s it. '
            'Set False to opt a deploy out of resume when it changes recon prompts '
            'or tooling — a resumed session finishes under the OLD system prompt by '
            'construction (--resume skips the system prompt), so a prompt/tool '
            'change must not be resumed into. When False, a cancelled cycle keeps '
            "today's failed + restore_drained + GC behaviour verbatim."
        ),
    )
    resume_freshness_window_seconds: int = Field(
        default=3600,
        gt=0,
        description=(
            'Max age (seconds) of an interrupted run — measured from its '
            'completed_at (the interrupt instant stamped by complete_run) — that '
            'the startup pass will still --resume. Beyond this window the run is '
            "treated as too stale to resume and falls back to today's "
            'failed+restore path. Default 3600s.'
        ),
    )
    resume_max_attempts_per_run: int = Field(
        default=2,
        gt=0,
        description=(
            'Per-run cap on resume attempts (tracked in '
            "stage_reports['_resume'].count). Once a run has been resumed this "
            'many times it falls back to failed+restore instead of resuming again, '
            'bounding a resume→re-interrupt→resume loop. Default 2.'
        ),
    )
    resume_failure_storm_threshold: int = Field(
        default=6,
        gt=0,
        description=(
            'Number of unresumable/failed resume attempts within '
            'resume_failure_storm_window_seconds that triggers a single loud '
            'recon_resume_failure_storm escalation. Mirrors '
            'dead_owner_suppression_storm_threshold. Default 6 per 3600s.'
        ),
    )
    resume_failure_storm_window_seconds: float = Field(
        default=3600.0,
        gt=0,
        description=(
            'Rolling time window (seconds) for counting resume failures. A single '
            'recon_resume_failure_storm alarm fires per window when the count '
            'crosses resume_failure_storm_threshold; also doubles as the '
            'per-window rate limit. Mirrors '
            'dead_owner_suppression_storm_window_seconds. Default 3600.0.'
        ),
    )

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

    # Recon-report in-process state TTL
    recon_report_state_ttl_seconds: int = Field(
        default=300,
        gt=0,
        description='TTL after complete() before reaper sweeps the entry (PRD §9.4)',
    )

    # Recon-report SQLite write-through persistence (task 2716)
    recon_report_persist_enabled: bool = Field(
        default=True,
        description=(
            'When True (and reconciliation.enabled), back ReconReportState with '
            "a durable SQLite store (recon_report_state.db) so a mid-stage server "
            'restart does not lose previously-filed findings. Fresh in-process '
            'runs are byte-identical either way; this only controls whether '
            'writes are also shadowed to disk.'
        ),
    )

    # Write-time near-duplicate guard for procedural_knowledge add_memory writes (task 2467).
    # Moves dedup from reactive Stage-1 consolidation to write time: a high-similarity match
    # against an existing procedural_knowledge Mem0 entry soft-blocks the write instead of
    # letting the duplicate land and consuming another cleanup cycle.
    #
    # Ownership note: the guard itself is enforced in the server layer (see
    # server/near_duplicate_guard.py and the add_memory tool in server/tools.py), not in the
    # reconciliation subsystem -- these two knobs live on ReconciliationConfig anyway because
    # the guard is the write-time counterpart to Stage-1's reactive procedural_knowledge
    # consolidation (same recurring-duplicate problem, same operator-tunable bounded-threshold
    # shape as this model's sibling `*_threshold` fields), not because the write path executes
    # inside reconciliation. Resolved via `memory_service.config.reconciliation.*` by
    # `resolve_near_dup_threshold` / `resolve_near_dup_guard_enabled` in
    # server/near_duplicate_guard.py -- if this guard ever grows independent of reconciliation,
    # move these two fields to a dedicated server-owned config section instead of assuming
    # colocation implies subsystem ownership.
    procedural_knowledge_near_dup_guard_enabled: bool = Field(
        default=True,
        description=(
            'Enable the add_memory write-time near-duplicate guard for '
            'category=procedural_knowledge. Green-tier hot-reloadable via the '
            'reload_config MCP tool (read live per add_memory by '
            'resolve_near_dup_guard_enabled; backed by task 2718).'
        ),
    )
    procedural_knowledge_near_dup_threshold: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description=(
            'Minimum relevance_score (cosine similarity) against an existing '
            'procedural_knowledge Mem0 entry that soft-blocks an add_memory write. '
            'Default 0.92 mirrors Mem0\'s own cited ~0.92 cosine dedup threshold. '
            'Green-tier hot-reloadable via the reload_config MCP tool (read live '
            'per add_memory by resolve_near_dup_threshold; task 2718).'
        ),
    )

    # Write-time TOPIC-keyed cluster guard for procedural_knowledge add_memory writes
    # (task 2845). Complements the cosine near-dup guard above: paraphrased same-topic
    # entries score BELOW any cosine threshold that is also safe for unrelated writes,
    # so a recurring known-contradictory topic (e.g. "plan-tools MCP server missing in
    # the eval worktree") kept accumulating contradictory entries the cosine guard
    # never fired on. This deterministic substring matcher targets exactly those known
    # clusters. Same ownership note as the near-dup fields above: enforced in the
    # server layer (server/near_duplicate_guard.py::find_matching_topic_cluster /
    # resolve_topic_guard_clusters + the add_memory tool), colocated on
    # ReconciliationConfig because it is the write-time counterpart to Stage-1's
    # reactive procedural_knowledge consolidation, not because the write path runs
    # inside reconciliation. Read live per add_memory write off the shared config
    # object, so it satisfies the reload.py live-read reload-safety rule.
    procedural_knowledge_topic_guard_clusters: list[ProceduralTopicCluster] = Field(
        default_factory=_default_topic_guard_clusters,
        description=(
            'Known-contradictory procedural_knowledge topic clusters. An add_memory '
            "write whose content contains >= a cluster's min_phrase_hits distinct "
            'phrases is soft-blocked (error_type='
            'ProceduralKnowledgeKnownTopicClusterWriteRejected) BEFORE the cosine '
            'near-dup search. Seeded with the two known eval-worktree clusters '
            '(gates 2841/2844); an empty list disables the topic guard. Green-tier '
            'hot-reloadable via the reload_config MCP tool (read live per add_memory '
            'by resolve_topic_guard_clusters in server/near_duplicate_guard.py). '
            'Shares the procedural_knowledge_near_dup_guard_enabled kill-switch and '
            'the recon-stage / allow_near_duplicate exemptions with the cosine guard.'
        ),
    )


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


class SummaryRebuildConfig(BaseModel):
    """Scheduled staleness backstop for entity summaries (fix (b), task 1958).

    Follow-up to task 1949's fix (a) — a best-effort post-ingestion refresh
    of the edge endpoints returned by a single write. That leaves residual
    drift for entities graphiti_core resolves against but that are not in
    the returned edge list. Only a periodic sweep of
    ``memory_service.rebuild_entity_summaries`` bounds that drift regardless
    of cause. Disabled by default (``enabled=False``, ``projects=[]``) so
    the sweep costs nothing until an operator opts in.
    """

    enabled: bool = Field(default=False)
    interval_seconds: float = Field(
        default=3600.0,
        gt=0,
        description='Seconds between sweeps of the configured projects.',
    )
    projects: list[str] = Field(
        default_factory=list,
        description='project_ids to sweep; empty = no-op regardless of enabled.',
    )
    force: bool = Field(default=False)


class CuratorConfig(BaseModel):
    """Task curator gate — LLM-judged drop/combine/create decision on task creation.

    Replaces the title-only threshold dedup with a richer corpus + LLM call so
    reviewer-spawned tasks can be combined into existing pending work instead of
    fragmenting the backlog. See docs/reify-task-fragmentation-report-2026-04-11.txt
    for the motivating analysis.
    """

    enabled: bool = Field(default=True)
    model: str = Field(default='sonnet')
    # 180s bounds a silent Anthropic-API hang for the curator's best-effort
    # contract while still giving ~20% headroom over the slowest legitimate
    # single-item decision observed in production (~150s); see esc-task-curator-20.
    timeout_seconds: float = Field(default=180.0)
    # Durable flat $2.00 per-call ceiling (task 1980 / esc-task-curator-194).
    # A prior scale-by-batch-size attempt (reify task 2254) was cancelled and
    # the cap recurred as pools/batches grew; a flat, size-independent ceiling
    # does not recur. Raising this base ALONE is not sufficient — see the
    # single-call comment block below on why single_call_budget_cap_usd must
    # move together with this value.
    max_budget_usd: float = Field(default=2.00)
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
    # NOTE: max_budget_usd and batch_budget_cap_usd are both $2.00 (task 1980
    # / esc-task-curator-194), so the budget formula's additive term
    # (per_item_budget_usd) is currently inert — every N clamps to the same
    # flat $2.00. timeout and max_turns above are unaffected by that raise
    # and still scale normally. per_item_budget_usd is retained (not
    # removed) in case the flat ceiling is ever re-tuned; TestScaleBudget in
    # test_task_curator.py pins the underlying min(base + per_entry*size,
    # cap) arithmetic independent of these currently-equal defaults.
    batch_max: int = Field(default=5)
    per_item_slack_seconds: float = Field(default=30.0)
    per_item_turns: int = Field(default=1)
    per_item_budget_usd: float = Field(default=0.30)
    batch_timeout_cap_seconds: float = Field(default=360.0)
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

    # Single-call budget scaling — mirrors the batch formula but scales by
    # pool size (the prompt-token cost driver for single-candidate calls):
    #   budget = min(max_budget_usd + per_pool_entry_budget_usd * len(pool),
    #                single_call_budget_cap_usd)
    # Durable flat $2.00 ceiling (task 1980 / esc-task-curator-194): a full
    # 30-entry pool (pool_total_cap=30) costs ~$0.30574 (esc-task-curator-191)
    # — comfortably under $2.00 with no per-size tuning required. The prior
    # scaled cap (0.75, derived from 0.30 + 0.0125*30 = $0.675) was raised
    # alongside max_budget_usd to the SAME $2.00 value: single_call_budget_cap_usd
    # must be >= max_budget_usd or it silently clamps the raised base back
    # down, defeating the durable raise (see TestCuratorConfigBudgetRaise in
    # test_config_schema.py). With base == cap == $2.00, this formula now
    # clamps to a flat $2.00 for every pool size, including pathologically
    # large ones — the scaling knob below is retained but inert (its
    # min(base + per_entry*size, cap) arithmetic is pinned independent of
    # these defaults by TestScaleBudget in test_task_curator.py).
    per_pool_entry_budget_usd: float = Field(default=0.0125)
    single_call_budget_cap_usd: float = Field(default=2.00)

    # Background janitor that surfaces failed tickets to the orchestrator.
    janitor: TicketJanitorConfig = Field(default_factory=TicketJanitorConfig)

    # Zero-output-timeout (ZOT) circuit-breaker watchdog.
    # Root cause: transient Anthropic-backend degradation on the curator's
    # sonnet+json-schema call shape (task 1550). Each hang burns the full
    # timeout_seconds (180s); the breaker stops every call burning that cost
    # during a sustained outage while preserving the best-effort
    # degrade-to-create contract.
    # Open after this many CONSECUTIVE ZOT curator LLM failures (reset on
    # any success or on a non-ZOT failure).
    zero_output_breaker_threshold: int = Field(default=2, ge=1)
    # How long the breaker stays open / short-circuits to action='create'
    # before allowing a half-open probe.
    zero_output_breaker_cooldown_seconds: float = Field(default=600.0, gt=0)

    # Cancelled-premise blocklist: path (absolute, or relative to server cwd)
    # of a YAML file listing premises proven wrong by revert. Matching
    # candidates are dropped before any LLM call. None disables the guard.
    # See fused-memory/config/cancelled_premise_blocklist.yaml for the schema.
    cancelled_premise_blocklist_path: str | None = Field(
        default=None,
        description=(
            "Path (absolute or relative to server cwd) of a YAML file listing "
            "premises proven wrong by revert; matching candidates are dropped "
            "before any LLM call. None disables the guard."
        ),
    )

    # Recon code-fix premise-verification registry: path (absolute, or relative
    # to server cwd) of a YAML registry of self-referential recon code-fix
    # premises. Matching candidates are dropped BEFORE any LLM call ONLY WHILE
    # the live source/tests still refute the premise (self-correcting — unlike
    # cancelled_premise_blocklist_path's unconditional forever-drop). None
    # disables the guard. See
    # fused-memory/config/recon_code_fix_premise_registry.yaml for the schema.
    recon_code_fix_premise_registry_path: str | None = Field(
        default=None,
        description=(
            "Path (absolute or relative to server cwd) of a YAML registry of "
            "self-referential recon code-fix premises; matching candidates are "
            "dropped BEFORE any LLM call only while the live source/tests still "
            "refute the premise. None disables the guard."
        ),
    )

    # Recon claim-verification backstop (task 2438): when True, specific
    # code-level tokens a Stage 2 candidate attributes to a completed
    # task/commit/ACTION (a metadata key, a stamp like foo=true, a named
    # identifier) are verified against the live source tree + git history
    # before the candidate is filed. Unlike recon_code_fix_premise_registry_path
    # above, an unverified token is never dropped — it is only logged as an
    # advisory recon_claim_verification.unverified WARNING for human review.
    # False (default) disables the guard, keeping every existing caller
    # byte-identical.
    recon_claim_verification_enabled: bool = Field(
        default=False,
        description=(
            "When True, the recon claim-verification backstop runs on filed "
            "candidates: code-level tokens attributed to a completed "
            "task/commit/ACTION are verified against the live tree + git "
            "history, and unverified tokens are logged as "
            "recon_claim_verification.unverified WARNINGs for human review. "
            "False disables the guard."
        ),
    )

    # Operational-ask registry: path (absolute, or relative to server cwd) of a
    # YAML registry of recurring operational live-data/live-mutation asks —
    # deliverables that are already-built+unit-tested scripts whose remaining
    # work is dry-run -> human-review -> apply against live external stores,
    # modifying ZERO repo files. The TDD architect cannot author a RED->GREEN
    # plan for that shape of work, so matching candidates are routed BEFORE any
    # LLM call to a deterministic PURE-GATE instead of reaching the architect.
    # None disables the gate. See
    # fused-memory/config/operational_ask_registry.yaml for the schema.
    operational_ask_registry_path: str | None = Field(
        default=None,
        description=(
            "Path (absolute or relative to server cwd) of a YAML registry of "
            "recurring operational live-data/live-mutation asks; matching "
            "candidates are routed BEFORE any LLM call to a deterministic "
            "PURE-GATE instead of the TDD architect. None disables the gate."
        ),
    )


class PathScopeAdjudicatorConfig(BaseModel):
    """LLM adjudicator config for Stage-2 path-scope guard filtering.

    Fires only on a Stage-1 heuristic HIT (rare).

    Cost is dominated by FIXED overhead — Sonnet + 3-turn json-schema flow +
    CLI base-context cache-creation (~13k tokens) + the filing project's
    CLAUDE.md/memory auto-loaded from cwd — NOT by the prompt size
    (title + description + matched prefixes).  Measured true cost is ~0.105 USD
    regardless of task size.  max_budget_usd must comfortably exceed that floor
    or every call returns error_max_budget_usd before any verdict (silent no-op).

    Fail-safe asymmetry: only a clean LLM success with verdict=='allow'
    downgrades the heuristic hit; every other outcome preserves the guard.
    """

    enabled: bool = Field(default=True)
    model: str = Field(default='sonnet')
    # 60s bounds a silent Anthropic-API hang; independent of prompt/cost size.
    timeout_seconds: float = Field(default=60.0)
    # True cost floor was measured at ~0.105 USD against Sonnet 4.6 on
    # 2026-06-19 (task 1849); the 0.30 default sized against that floor has
    # since eroded as stack-wide cost-per-call rose (~2-2.5x token-volume
    # growth from filing-project CLAUDE.md bloat + opus/sonnet alias-roll
    # tokenizer inflation) — the sibling CuratorConfig already tripped this
    # same failure mode (task 1980 / esc-task-curator-194). Re-baselined to a
    # flat $2.00 (task 1983) to match CuratorConfig's durable ceiling. Do NOT
    # lower below 0.25 — the adjudicator becomes a silent no-op.
    max_budget_usd: float = Field(default=2.00)
    # Per-call turn cap. Must be ≥ 3: schema burns one tool-use turn,
    # an optional reasoning turn may precede it, and the final assistant
    # response is the third. Matches the json-schema turn floor noted in
    # CuratorConfig.
    max_turns: int = Field(default=3, ge=3)
    # Zero-output-timeout circuit-breaker knobs (mirrors CuratorConfig).
    zero_output_breaker_threshold: int = Field(default=2, ge=1)
    zero_output_breaker_cooldown_seconds: float = Field(default=600.0, gt=0)


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


class WriteTriageConfig(BaseModel):
    """Server-owned band thresholds for add_memory write triage (task 3130).

    This is the dedicated server-owned section that ReconciliationConfig's
    near-dup ownership note anticipated ("if this guard ever grows independent
    of reconciliation, move these two fields to a dedicated server-owned config
    section instead of assuming colocation implies subsystem ownership") —
    write triage IS that growth, so the bands land here rather than accreting
    onto the reconciliation submodel. PRD leaf beta adds `write_triage_enabled`
    to this same section.

    Declared on FusedMemoryConfig as a BARE (non-Optional) submodel so
    config/reload.py's `_iter_leaves` descends into per-leaf paths. An
    `X | None` submodel is compared whole and lands as a single
    restart_required entry (esc-2718-1), which would force a server restart
    for every calibration run.

    Every field defaults to None, and that is load-bearing: None means
    UNCALIBRATED, which the triage router must read as fail-open to `stored`.
    """

    t_high: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            'Similarity at or above which a write is deterministically treated as a '
            'restatement of an existing memory, with no judge call. None means '
            'UNCALIBRATED and triage must fail open to `stored`. This value is an '
            'OUTPUT of scripts/calibrate_write_triage.py — the smallest measured '
            'duplicate score that strictly exceeds every measured negative — and '
            'must never be hand-set to a guessed number. The sibling '
            'reconciliation.procedural_knowledge_near_dup_threshold shows why: its '
            '0.92 was inherited from Mem0\'s cited figure and sits above a genuine '
            'rediscovery pair measured at 0.824, so it could not fire on the case it '
            'exists to catch. Green-tier hot-reloadable via reload_config.'
        ),
    )
    t_low: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            'Similarity below which a write is stored as new without a judge call. '
            'Between t_low and t_high the write is routed to the judge. None means '
            'UNCALIBRATED (fail open to `stored`). Also an OUTPUT of '
            'scripts/calibrate_write_triage.py — the measured lower tail (p05) of the '
            'curator-confirmed duplicate distribution — never hand-set. Green-tier '
            'hot-reloadable via reload_config.'
        ),
    )
    calibration_report_path: str | None = Field(
        default=None,
        description=(
            'Path to the JSON calibration report that produced t_high/t_low — the '
            'traceability link from the thresholds in this config back to the run '
            'and the measured distributions that justify them, including the '
            'deterministic band\'s measured false-positive count. Written by '
            'scripts/calibrate_write_triage.py --write-config. Green-tier '
            'hot-reloadable via reload_config.'
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
    task_metadata: TaskMetadataConfig = Field(default_factory=TaskMetadataConfig)
    memory_metadata: MemoryMetadataConfig = Field(default_factory=MemoryMetadataConfig)
    task_status: TaskStatusConfig = Field(default_factory=TaskStatusConfig)
    reconciliation: ReconciliationConfig = Field(default_factory=ReconciliationConfig)
    # Bare (non-Optional) submodel on purpose — see WriteTriageConfig's docstring:
    # reload.py descends only into required submodels, so nullability here would
    # cost the section its per-leaf hot-reload.
    write_triage: WriteTriageConfig = Field(default_factory=WriteTriageConfig)
    curator: CuratorConfig = Field(default_factory=CuratorConfig)
    summary_rebuild: SummaryRebuildConfig = Field(default_factory=SummaryRebuildConfig)
    path_scope_adjudicator: PathScopeAdjudicatorConfig = Field(
        default_factory=PathScopeAdjudicatorConfig,
    )
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
        config_path = Path(os.environ.get('CONFIG_PATH', DEFAULT_CONFIG_PATH))
        yaml_settings = YamlSettingsSource(settings_cls, config_path)
        return (init_settings, env_settings, yaml_settings, dotenv_settings)
