"""Configuration schema for the orchestrator."""

import importlib.resources
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

logger = logging.getLogger(__name__)


# --- Priority-tier constants (value/h scheduler) ---
#
# Canonical 5-tier priority order.  Lower rank = higher priority.  Unknown
# priority strings coerce to DEFAULT_TIER so legacy tasks and typos never crash
# the scheduler.
PRIORITY_TIERS: tuple[str, ...] = ('critical', 'high', 'medium', 'low', 'polish')
PRIORITY_RANK: dict[str, int] = {tier: i for i, tier in enumerate(PRIORITY_TIERS)}
DEFAULT_TIER: str = 'medium'

# Scoring base per tier, with uniform TIER_WIDTH between adjacent tiers so
# age / CPM bonuses can never bump a task across a tier boundary (Fix 1).
TIER_WIDTH: int = 1000
TIER_BASE: dict[str, int] = {
    'critical': 16000,
    'high': 8000,
    'medium': 4000,
    'low': 2000,
    'polish': 1000,
}


def coerce_tier(value: Any) -> str:
    """Normalize a priority value (possibly None/unknown) to a canonical tier."""
    if isinstance(value, str) and value in PRIORITY_RANK:
        return value
    return DEFAULT_TIER


class ConfigRequiredError(Exception):
    """Raised when no orchestrator config is provided via --config or ORCH_CONFIG_PATH.

    The orchestrator deliberately refuses to auto-detect target projects from cwd,
    because silent defaults previously caused cross-project execution that lost work
    (2026-04-06 incident: /orchestrate run from ~/src/reify silently executed
    dark-factory tasks because cwd-based discovery picked dark-factory's own config).
    """


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge *override* into *base*.  Override values win at leaf level."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_defaults() -> dict[str, Any]:
    """Load the package-bundled defaults.yaml."""
    defaults_path = importlib.resources.files('orchestrator').joinpath('defaults.yaml')
    with importlib.resources.as_file(defaults_path) as p, open(p) as f:
        return yaml.safe_load(f) or {}


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
        # Layer 1: package-bundled defaults
        base = _load_defaults()
        # Layer 2: project config file (overrides defaults via deep merge)
        if self.config_path.exists():
            with open(self.config_path) as f:
                project_config = yaml.safe_load(f) or {}
            base = _deep_merge(base, project_config)
        # Expand env vars on the final merged dict (covers both defaults and overrides)
        return self._expand_env_vars(base)


# --- Sub-models ---


class ModelsConfig(BaseModel):
    """Model selection per agent role."""

    architect: str = Field(default='opus')
    implementer: str = Field(default='opus')
    debugger: str = Field(default='opus')
    reviewer: str = Field(default='sonnet')
    merger: str = Field(default='opus')
    steward: str = Field(default='opus')
    triage: str = Field(default='sonnet')
    module_tagger: str = Field(default='sonnet')
    deep_reviewer: str = Field(default='opus')
    judge: str = Field(default='sonnet')


class BudgetsConfig(BaseModel):
    """Max USD spend per invocation, by role."""

    architect: float = Field(default=8.0)
    implementer: float = Field(default=10.0)
    debugger: float = Field(default=5.0)
    reviewer: float = Field(default=2.0)
    merger: float = Field(default=5.0)
    steward: float = Field(default=5.0)
    triage: float = Field(default=2.0)
    module_tagger: float = Field(default=2.0)
    deep_reviewer: float = Field(default=15.0)
    judge: float = Field(default=0.50)


class TurnsConfig(BaseModel):
    """Max conversation turns per invocation, by role."""

    architect: int = Field(default=75)
    implementer: int = Field(default=80)
    debugger: int = Field(default=50)
    reviewer: int = Field(default=30)
    merger: int = Field(default=50)
    steward: int = Field(default=100)
    triage: int = Field(default=25)
    module_tagger: int = Field(default=30)
    deep_reviewer: int = Field(default=100)
    judge: int = Field(default=15)


class EffortConfig(BaseModel):
    """Reasoning effort level per agent role."""

    architect: str = Field(default='high')
    implementer: str = Field(default='high')
    debugger: str = Field(default='high')
    reviewer: str = Field(default='medium')
    merger: str = Field(default='high')
    steward: str = Field(default='high')
    triage: str = Field(default='medium')
    module_tagger: str = Field(default='medium')
    deep_reviewer: str = Field(default='max')
    judge: str = Field(default='medium')


class TimeoutsConfig(BaseModel):
    """Wall-clock timeout (seconds) per agent role.

    Note: ``steward`` here is the *per-invocation* wall-clock limit for a
    single ``invoke_agent`` call.  It is intentionally decoupled from
    ``OrchestratorConfig.steward_completion_timeout``, which is the workflow
    grace period that controls how long the workflow waits for the steward to
    drain the escalation queue after task completion.  Keep ``steward`` ≥
    ``steward_completion_timeout`` so individual invocations are not silently
    cut short inside the grace window.  This invariant is enforced at
    construction time by a ``model_validator`` on ``OrchestratorConfig``.
    """

    architect: float = Field(default=2400.0)
    implementer: float = Field(default=1200.0)
    debugger: float = Field(default=1200.0)
    reviewer: float = Field(default=600.0)
    merger: float = Field(default=600.0)
    steward: float = Field(default=1800.0)
    triage: float = Field(default=300.0)
    module_tagger: float = Field(default=300.0)
    deep_reviewer: float = Field(default=2400.0)
    judge: float = Field(default=300.0)


class BackendsConfig(BaseModel):
    """Backend CLI selection per agent role. Values: 'claude', 'codex', 'gemini'."""

    architect: str = Field(default='claude')
    implementer: str = Field(default='claude')
    debugger: str = Field(default='claude')
    reviewer: str = Field(default='claude')
    merger: str = Field(default='claude')
    steward: str = Field(default='claude')
    triage: str = Field(default='claude')
    module_tagger: str = Field(default='claude')
    deep_reviewer: str = Field(default='claude')
    judge: str = Field(default='claude')


class ReviewConfig(BaseModel):
    """Periodic deep review checkpoint configuration."""

    enabled: bool = Field(default=True)
    interval: int = Field(default=5, description='Trigger checkpoint every N merges')
    full_review_on_complete: bool = Field(default=True)
    briefing_path: str = Field(default='review/briefing.yaml')
    reports_dir: str = Field(default='data/review-checkpoints')


class FairnessConfig(BaseModel):
    """Scheduler fairness / anti-starvation configuration.

    When a broad-footprint task keeps losing the greedy lock race to narrow
    tasks, the scheduler increments a per-task skip counter.  Once the counter
    reaches ``skip_threshold``, the scheduler installs a short-lived
    reservation ("park") on every module the starved task wants; parked
    modules refuse ``try_acquire`` from anyone else until the owner acquires
    or the lease expires.
    """

    skip_threshold: int | dict[str, int] = Field(
        default=4,
        description=(
            'Consecutive top-candidate skips before installing a reservation.  '
            'Accepts either an int (applies to every tier) or a '
            'dict[tier -> int] for per-tier thresholds.  Thresholds >= 1000 '
            'effectively disable parking for that tier and auto-enable '
            'rate-limited task_skipped emission.'
        ),
    )
    lease_multiplier: float | dict[str, float] = Field(
        default=5.0,
        description=(
            'Lease duration = median(recent task durations) * multiplier.  '
            'Accepts either a float or a dict[tier -> float] for per-tier '
            'multipliers.'
        ),
    )
    lease_min_secs: float = Field(default=60.0)
    lease_max_secs: float = Field(default=3600.0)
    median_window: int = Field(
        default=50,
        description='Rolling window size for task-duration history.',
    )

    @field_validator('skip_threshold', mode='before')
    @classmethod
    def _validate_skip_threshold(cls, v: Any) -> Any:
        if v is None:
            return 4
        if isinstance(v, int):
            return v
        if isinstance(v, dict):
            bad_keys = set(v) - set(PRIORITY_RANK)
            if bad_keys:
                raise ValueError(
                    f'fairness.skip_threshold has unknown priority tier(s): '
                    f'{sorted(bad_keys)}.  Known tiers: {list(PRIORITY_RANK)}.'
                )
            return {k: int(val) for k, val in v.items()}
        raise ValueError(
            f'fairness.skip_threshold must be int or dict[tier -> int]; '
            f'got {type(v).__name__}.'
        )

    @field_validator('lease_multiplier', mode='before')
    @classmethod
    def _validate_lease_multiplier(cls, v: Any) -> Any:
        if v is None:
            return 5.0
        if isinstance(v, int | float):
            return float(v)
        if isinstance(v, dict):
            bad_keys = set(v) - set(PRIORITY_RANK)
            if bad_keys:
                raise ValueError(
                    f'fairness.lease_multiplier has unknown priority tier(s): '
                    f'{sorted(bad_keys)}.  Known tiers: {list(PRIORITY_RANK)}.'
                )
            return {k: float(val) for k, val in v.items()}
        raise ValueError(
            f'fairness.lease_multiplier must be number or dict[tier -> number]; '
            f'got {type(v).__name__}.'
        )

    def skip_threshold_for(self, tier: str) -> int:
        """Return the skip threshold that applies to *tier*."""
        tier = coerce_tier(tier)
        if isinstance(self.skip_threshold, dict):
            return int(
                self.skip_threshold.get(tier, self.skip_threshold.get(DEFAULT_TIER, 4))
            )
        return int(self.skip_threshold)

    def lease_multiplier_for(self, tier: str) -> float:
        """Return the lease multiplier that applies to *tier*."""
        tier = coerce_tier(tier)
        if isinstance(self.lease_multiplier, dict):
            return float(
                self.lease_multiplier.get(
                    tier, self.lease_multiplier.get(DEFAULT_TIER, 5.0)
                )
            )
        return float(self.lease_multiplier)


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
    """Filesystem sandbox configuration.

    ``backend`` selects the enforcement mechanism:
    - ``auto`` (default): prefer landlock if available, else bwrap, else unsandboxed
    - ``landlock``: kernel LSM; works in all namespaces. Requires kernel 5.13+
    - ``bwrap``: bubblewrap + user namespace. Bun v1.3.13 crashes under this
      on kernel 6.17; prefer landlock on affected hosts
    - ``none``: explicit opt-out — run unsandboxed (same effect as ``enabled: false``)
    """

    enabled: bool = Field(default=True)
    backend: Literal['auto', 'bwrap', 'landlock', 'none'] = Field(default='auto')


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
    push_after_advance: bool = Field(
        default=True,
        description=(
            'Push main to <remote> after each successful CAS advance. '
            'Best-effort: failures are logged but do not fail the merge.'
        ),
    )


# --- Per-module overrides ---

_OVERRIDABLE_FIELDS = frozenset({
    'test_command', 'lint_command', 'type_check_command',
    'lock_depth', 'max_per_module', 'module_overrides',
    'verify_command_timeout_secs',
    'verify_cold_command_timeout_secs',
    'concurrent_verify', 'verify_env',
    'scope_cargo',
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
    verify_command_timeout_secs: float | None = None
    # Per-subproject cold-build timeout override (first verify before .task/verify_warmed exists).
    # Falls back to verify_command_timeout_secs when None.
    verify_cold_command_timeout_secs: float | None = None
    concurrent_verify: bool | None = None
    verify_env: dict[str, str] | None = None
    scope_cargo: bool | None = None


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
    reviewer_stagger_secs: float = Field(default=2.0)
    max_reviewer_retries: int = Field(default=4)
    # Max in-workflow amendment rounds after a PASS-with-suggestions review.
    # Each round reinvokes the implementer with in-scope suggestions (scoped
    # by module-lock membership), re-verifies, and re-reviews. Remaining
    # out-of-scope or cap-exhausted suggestions still flow through the
    # existing escalate_suggestions path.
    max_amendment_rounds: int = Field(default=1)

    # Completion judge — opt-in loop-exit hint after each implementer iteration.
    # Default False: production orchestrator runs unaffected. Eval runner
    # enables this per-task (see evals/runner.py build_eval_orch_config).
    judge_after_each_iteration: bool = Field(default=False)

    # Merge conflict reduction
    max_advance_attempts: int = Field(default=3)
    max_pre_merge_retries: int = Field(default=2)
    max_merge_retries: int = Field(default=3)
    inter_iteration_rebase: bool = Field(default=True)
    requeue_cooldown_secs: float = Field(default=30.0)
    requeue_cap: int = Field(
        default=3,
        ge=1,
        description=(
            'Max WorkflowOutcome.REQUEUED iterations per task before '
            'L1-escalating to a human.  Prevents tight requeue loops from '
            'burning budget when the steward repeatedly resolves the same '
            'transient failure.  Counter is per task_id and process-local: '
            'orchestrator restart resets it.  A DONE outcome clears the '
            'counter; triggering cap-exhaustion also clears it so a human '
            'resolution starts from zero.'
        ),
    )

    # Verification timeouts
    verify_command_timeout_secs: float = Field(default=1800.0)
    # Timeout for the *first* verify in a freshly created worktree (cold build
    # cache).  Applies until `.task/verify_warmed` exists.  Falls back to
    # verify_command_timeout_secs when None (preserves existing behaviour).
    # Shipped default comes from defaults.yaml (5400s = 3× warm).
    verify_cold_command_timeout_secs: float | None = Field(default=None)
    verify_timeout_retries: int = Field(default=2)

    # Verification execution mode + env
    # When False, test/lint/type run sequentially within a single verify
    # invocation.  Useful for Rust workspaces where cargo takes an advisory
    # lock on target/ and the concurrent subcommands serialize anyway.
    concurrent_verify: bool = Field(default=True)
    # Extra env vars injected into verify commands (e.g. RUSTC_WRAPPER=sccache).
    # Distinct from env_overrides, which targets agent invocations, not verify.
    verify_env: dict[str, str] = Field(default_factory=dict)
    # When True, task-phase verify for Rust tasks rewrites
    # ``cargo --workspace`` → ``cargo -p <crate>`` for the touched crates.
    # Post-merge verify always runs workspace-wide regardless.
    scope_cargo: bool = Field(default=True)

    # Steward lifecycle
    steward_lifetime_budget: float = Field(default=12.0)
    steward_max_attempts: int = Field(default=1)
    steward_completion_timeout: float = Field(default=900.0)
    steward_max_timeouts_per_escalation: int = Field(default=3, ge=2, le=5)

    # Pre-triage threshold for review suggestions
    suggestion_triage_threshold: int = Field(default=10)

    # Orphan L0 reaper — re-escalates level-0 escalations whose task has no
    # active workflow/steward (e.g. escalations emitted by the deep reviewer
    # against a synthetic ``review-*`` task_id).  Without this, such
    # escalations sit pending until the next orchestrator restart dismisses
    # them unread.  Set ``orphan_l0_reaper_enabled = False`` to disable.
    orphan_l0_reaper_enabled: bool = Field(default=True)
    orphan_l0_timeout_secs: float = Field(default=600.0)
    orphan_l0_check_interval_secs: float = Field(default=60.0)

    # Legacy scalar — ignored if `timeouts` section is present in config.
    # Kept for backwards-compat with config files that haven't migrated.
    invocation_timeout: float = Field(default=1200.0)

    # Models, budgets, turns, timeouts per role
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    budgets: BudgetsConfig = Field(default_factory=BudgetsConfig)
    max_turns: TurnsConfig = Field(default_factory=TurnsConfig)
    effort: EffortConfig = Field(default_factory=EffortConfig)
    timeouts: TimeoutsConfig = Field(default_factory=TimeoutsConfig)
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

    # Review checkpoints
    review: ReviewConfig = Field(default_factory=ReviewConfig)

    # Scheduler fairness / anti-starvation
    fairness: FairnessConfig = Field(default_factory=FairnessConfig)

    # Value/h scheduler scoring (P2/P3 — age boost, CPM weighting).
    age_alpha: float = Field(
        default=10.0,
        description=(
            'Age-boost coefficient in score(): age_bonus = min(alpha * age, '
            'TIER_WIDTH - 1).  "Age" here is proxied by (max_id - task_id) '
            '— number of newer tasks created since this one.  Combined '
            'age+CPM bonus is capped below TIER_WIDTH so bonuses never cross '
            'a tier boundary.'
        ),
    )
    cpm_beta: float = Field(
        default=100.0,
        description=(
            'Transitive-dependent coefficient: cpm_bonus = beta * log1p(n), '
            'where n = count of non-done descendants reachable via the '
            'reverse-dependency graph.  Captures value unlocked by completing '
            'this task (CPM proxy).'
        ),
    )
    # Gentle per-tier slot caps (Fix 3).  Each entry restricts how much of
    # ``max_concurrent_tasks`` can be held by tasks whose effective priority
    # is at *that tier or lower*.  Missing tiers default to 1.0 (no cap).
    # Parks (fairness reservations) override tier caps — once a park is
    # installed, the owner dispatches regardless of cap.
    tier_slot_caps: dict[str, float] = Field(default_factory=dict)

    # Git
    git: GitConfig = Field(default_factory=GitConfig)

    # Usage cap handling
    usage_cap: UsageCapConfig = Field(default_factory=UsageCapConfig)

    # Environment overrides forwarded to agent invocations
    env_overrides: dict[str, str] = Field(default_factory=dict)

    # Project
    project_root: Path = Field(default=Path('.'))

    # Per-module overrides (populated by load_config via _discover_module_configs)
    _module_configs: dict[str, ModuleConfig] = PrivateAttr(default_factory=dict)

    @field_validator('project_root', mode='after')
    @classmethod
    def _resolve_project_root(cls, v: Path) -> Path:
        return v.resolve()

    @field_validator('tier_slot_caps', mode='before')
    @classmethod
    def _validate_tier_slot_caps(cls, v: Any) -> dict[str, float]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError(
                f'tier_slot_caps must be dict[tier -> float]; got {type(v).__name__}.'
            )
        bad_keys = set(v) - set(PRIORITY_RANK)
        if bad_keys:
            raise ValueError(
                f'tier_slot_caps has unknown priority tier(s): {sorted(bad_keys)}.  '
                f'Known tiers: {list(PRIORITY_RANK)}.'
            )
        out: dict[str, float] = {}
        for k, val in v.items():
            fval = float(val)
            if not 0.0 <= fval <= 1.0:
                raise ValueError(
                    f'tier_slot_caps[{k!r}] = {fval} out of range [0.0, 1.0].'
                )
            out[k] = fval
        return out

    def tier_slot_cap_for(self, tier: str) -> float:
        """Return the fractional slot cap (0.0-1.0) for *tier*, 1.0 if unset."""
        return self.tier_slot_caps.get(coerce_tier(tier), 1.0)

    def tier_slot_limit(self, tier: str) -> int:
        """Return the integer slot limit for tasks at *tier or lower*.

        Computed as ``floor(tier_slot_cap_for(tier) * max_concurrent_tasks)``.
        When the cap is 1.0 (the default) this equals ``max_concurrent_tasks``
        and is never binding.
        """
        cap = self.tier_slot_cap_for(tier)
        if cap >= 1.0:
            return self.max_concurrent_tasks
        return int(cap * self.max_concurrent_tasks)

    @model_validator(mode='after')
    def _validate_steward_timeout_invariant(self) -> 'OrchestratorConfig':
        if self.timeouts.steward < self.steward_completion_timeout:
            raise ValueError(
                f'timeouts.steward ({self.timeouts.steward}) must be >= '
                f'steward_completion_timeout ({self.steward_completion_timeout}); '
                'a smaller per-invocation wall-clock would silently cut the steward '
                'short inside the grace window. '
                'Raise timeouts.steward to >= steward_completion_timeout, or lower '
                'steward_completion_timeout in your orchestrator.yaml.'
            )
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
        validate_assignment=True,
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
        config_path = Path(os.environ.get('ORCH_CONFIG_PATH', '') or 'config.yaml')
        yaml_settings = YamlSettingsSource(settings_cls, config_path)
        return (init_settings, env_settings, yaml_settings, dotenv_settings)


def load_config(config_path: Path | None = None) -> OrchestratorConfig:
    """Load configuration from an explicit YAML file.

    Resolution order:
    1. ``config_path`` argument (typically from ``--config`` flag)
    2. ``ORCH_CONFIG_PATH`` environment variable

    If neither is set, raises :class:`ConfigRequiredError`. The orchestrator does
    NOT auto-discover from cwd — see ``ConfigRequiredError`` docstring for the
    rationale.
    """
    if config_path is None:
        env_path = os.environ.get('ORCH_CONFIG_PATH')
        if not env_path:
            raise ConfigRequiredError(
                '--config is required (or set ORCH_CONFIG_PATH).\n\n'
                'The orchestrator does not auto-detect the target project from cwd; '
                'this safeguard exists because silent defaults previously caused '
                'cross-project execution that lost work.\n\n'
                'Examples:\n'
                '  uv run --project orchestrator orchestrator run \\\n'
                '      --config /home/leo/src/reify/orchestrator.yaml\n'
                '  ORCH_CONFIG_PATH=/home/leo/src/reify/orchestrator.yaml \\\n'
                '      uv run --project orchestrator orchestrator run\n\n'
                'See skills/orchestrate/references/project-setup.md for setup '
                'instructions.'
            )
        config_path = Path(env_path)

    if not config_path.exists():
        raise ConfigRequiredError(
            f'Config file not found: {config_path}\n\n'
            f'Pass an explicit --config path or set ORCH_CONFIG_PATH to a valid '
            f'file. See skills/orchestrate/references/project-setup.md for setup '
            f'instructions.'
        )

    os.environ['ORCH_CONFIG_PATH'] = str(config_path)
    config = OrchestratorConfig()
    config._module_configs = _discover_module_configs(config.project_root)
    return config
