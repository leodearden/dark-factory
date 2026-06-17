"""Configuration schema for the orchestrator."""

import importlib.resources
import logging
import os
import re
import stat
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
    startup_grace_secs: float = Field(
        default=120.0,
        description=(
            "Pre-turn-1 STARTUP grace window (seconds). "
            "If no assistant turn appears within this period, _run_subprocess "
            "kills the subprocess fast — this catches genuine from-source-build / "
            "uv / MCP-startup wedges. "
            "Distinct from the per-role post-turn-1 ceiling (e.g. implementer=1200s): "
            "once ≥1 assistant turn is observed, liveness is proven and the full "
            "per-role ceiling applies."
        ),
    )


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


class UnblockAutoConfig(BaseModel):
    """Configuration for the autonomous dry-run unblock hook."""

    enabled: bool = Field(default=True)
    budget_usd: float = Field(default=5.0)
    timeout_seconds: float = Field(default=600.0)
    model: str = Field(default='sonnet')
    max_turns: int = Field(default=50)
    effort: str = Field(default='high')
    backend: str = Field(default='claude')
    # SKILL-FACING: orchestrator code never reads this field.
    # Consumed by the unblock-low-risk / escalation-watcher skills (PRD §4.3).
    attended_b3_enabled: bool = Field(default=False)
    b3_merge_cap_per_24h: int = Field(default=6)
    b3_proposal_keep_last: int = Field(default=5)


class ReviewConfig(BaseModel):
    """Periodic deep review checkpoint configuration."""

    enabled: bool = Field(default=True)
    interval: int = Field(default=5, description='Trigger checkpoint every N merges')
    full_review_on_complete: bool = Field(default=True)
    briefing_path: str = Field(default='review/briefing.yaml')
    reports_dir: str = Field(default='data/review-checkpoints')
    # Run-forever rate-limit on the per-idle-cycle full review.  Both gates
    # must be open (a true ceiling) before a drain-to-idle triggers another
    # full review: at least ``full_review_min_interval_secs`` wall-clock since
    # the last full review AND at least ``full_review_min_tasks`` merges since
    # then.  Only consulted on the run-forever idle path; the exit /
    # --until-idle post-loop review stays unconditional.
    full_review_min_interval_secs: float = Field(default=86400.0)
    full_review_min_tasks: int = Field(default=20)


_DEFAULT_SKIP_THRESHOLD: dict[str, int] = {
    'critical': 0,
    'high': 1,
    'medium': 2,
    'low': 4,
    'polish': 9999,
}


class FairnessConfig(BaseModel):
    """Scheduler fairness / anti-starvation configuration.

    When a broad-footprint task keeps losing the greedy lock race to narrow
    tasks, the scheduler increments a per-task skip counter.  Once the counter
    reaches ``skip_threshold`` for the task's tier, the scheduler installs a
    reservation ("park") on every module the starved task wants.  Parks use
    eager, full-module-set coverage: every module the starved task needs is
    reserved at once, including modules that are currently free, to prevent
    lower-priority tasks from grabbing them while the owner waits.

    Parks are coupled to the owner's live state via owner-state GC: they
    evaporate the moment the owner completes, is cancelled, or has its
    dependencies un-satisfied — no wall-clock lease needed.  Cross-tier
    preemption ensures a parked high-priority task is not blocked indefinitely
    by a flood of lower-priority tasks.
    """

    skip_threshold: int | dict[str, int] = Field(
        default_factory=lambda: dict(_DEFAULT_SKIP_THRESHOLD),
        description=(
            'Consecutive top-candidate skips before installing a reservation.  '
            'Accepts either an int (applies to every tier) or a '
            'dict[tier -> int] for per-tier thresholds.  Thresholds >= 1000 '
            'effectively disable parking for that tier and auto-enable '
            'rate-limited task_skipped emission.'
        ),
    )

    @field_validator('skip_threshold', mode='before')
    @classmethod
    def _validate_skip_threshold(cls, v: Any) -> Any:
        if v is None:
            return dict(_DEFAULT_SKIP_THRESHOLD)
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

    def skip_threshold_for(self, tier: str) -> int:
        """Return the skip threshold that applies to *tier*."""
        tier = coerce_tier(tier)
        if isinstance(self.skip_threshold, dict):
            return int(
                self.skip_threshold.get(
                    tier,
                    self.skip_threshold.get(DEFAULT_TIER, _DEFAULT_SKIP_THRESHOLD.get(tier, 4)),
                )
            )
        return int(self.skip_threshold)


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


class SccacheConfig(BaseModel):
    """Shared sccache backend configuration (κ: the laptop warm multiplier).

    sccache selects its backend entirely via environment variables:
    - SCCACHE_REDIS           — Redis (LAN-local; PRD §10 Open Q1 suggestion)
    - SCCACHE_MEMCACHED       — Memcached
    - SCCACHE_BUCKET + SCCACHE_ENDPOINT — S3-compatible object store
    - SCCACHE_GCS_BUCKET      — GCS bucket

    Expressing the backend as a raw env-var mapping covers all four backends
    with zero per-backend code and lets ops switch by editing config only.

    ``enabled=True`` with an empty ``backend_env`` is rejected by the
    model_validator so a half-configured knob fails fast at load time.
    """

    enabled: bool = Field(default=False)
    backend_env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode='after')
    def _reject_enabled_with_empty_backend(self) -> 'SccacheConfig':
        if self.enabled and not self.backend_env:
            raise ValueError(
                'SccacheConfig.enabled is True but backend_env is empty; '
                'set at least one SCCACHE_* environment variable (e.g. '
                '{SCCACHE_REDIS: redis://host:6379}) or set enabled: false.'
            )
        return self

    def env_overrides(self) -> dict[str, str]:
        """Return a copy of backend_env when enabled, else an empty dict.

        Returns a COPY so callers can mutate the result without affecting
        this model instance.
        """
        return dict(self.backend_env) if self.enabled else {}


class JobserverConfig(BaseModel):
    """Task-pool jobserver wiring for agent inner-loop cargo builds.

    When enabled, injects CARGO_MAKEFLAGS into implementer/debugger/architect
    subprocesses so their cargo build/test/metadata calls participate in the
    task-pool FIFO jobserver instead of running uncoordinated.

    The reify project enables this in /home/leo/src/reify/orchestrator.yaml.
    All other projects leave enabled=False (the default) and receive no change.

    ``enabled=True`` with an empty ``task_fifo`` is rejected by the
    model_validator so a half-configured knob fails fast at load time.
    """

    enabled: bool = Field(
        default=False,
        description='Enable jobserver wiring for agent cargo builds.',
    )
    task_fifo: str = Field(
        default='/tmp/reify-jobserver-task',
        description='Absolute path to the task-pool FIFO created by the jobserver.',
    )
    env_var: str = Field(
        default='CARGO_MAKEFLAGS',
        description='Environment variable name that cargo reads for jobserver auth.',
    )

    @model_validator(mode='after')
    def _reject_enabled_with_empty_task_fifo(self) -> 'JobserverConfig':
        if self.enabled and not self.task_fifo:
            raise ValueError(
                'JobserverConfig.enabled is True but task_fifo is empty; '
                'set task_fifo to the absolute FIFO path or set enabled: false.'
            )
        return self

    def agent_env(self) -> dict[str, str]:
        """Return jobserver env dict when enabled and FIFO exists, else {}.

        Mirrors shell ``[ -p "$FIFO" ]``: a stale regular file or missing path
        returns {} rather than injecting a broken --jobserver-auth value that
        would wedge cargo.

        TOCTOU note: the FIFO is stat-checked here but consumed by cargo later.
        If the FIFO disappears between the check and the subprocess spawn,
        cargo receives a stale ``--jobserver-auth`` value.  Per cargo's
        documented behaviour this causes it to fall back to unbounded
        parallelism (equivalent to ``-j$(nproc)``) rather than hanging, so
        the failure mode is over-subscription — the same state as having no
        jobserver at all — not a deadlock.
        """
        if not self.enabled:
            return {}
        try:
            mode = os.stat(self.task_fifo).st_mode
        except OSError:
            return {}
        if not stat.S_ISFIFO(mode):
            return {}
        return {self.env_var: f'--jobserver-auth=fifo:{self.task_fifo}'}


class CpuPriorityConfig(BaseModel):
    """CPU nice-level de-prioritization for agent inner-loop subprocesses.

    When enabled, injects DF_AGENT_CPU_NICE into architect/implementer/debugger
    subprocesses so that ``cli_invoke._cpu_priority_prefix`` prepends
    ``nice -n <nice>`` to the Claude CLI spawn.  This causes the agent process
    (and all cargo/rustc children it forks) to run at a lower CPU priority,
    yielding to reify's negatively-niced merge/task verifies.

    ``nice`` must be in the range 1..19: positive values de-prioritize without
    requiring CAP_SYS_NICE; 0 is a no-op; negative values need privilege.
    The validator rejects ``enabled=True`` with ``nice`` outside 1..19 to fail
    fast on a misconfiguration.

    Defaults to ``enabled=True, nice=10``:
    - No external setup required (unlike the jobserver FIFO), so default-on is safe.
    - nice +10 vs merge verify nice -5 → 15-step CFS spread ≈ 28x weight to the
      verify; vs task verify nice -15 → ≈ 260x — decisive yield.
    - Default-on means the reify orchestrator restart alone activates the fix;
      no orchestrator.yaml edit required.
    """

    enabled: bool = Field(
        default=True,
        description='Enable CPU nice de-prioritization for agent subprocesses.',
    )
    nice: int = Field(
        default=10,
        description=(
            'nice(1) increment to apply to agent subprocesses (1..19). '
            'Positive values de-prioritize without CAP_SYS_NICE. '
            'Consumed by cli_invoke._cpu_priority_prefix via DF_AGENT_CPU_NICE.'
        ),
    )

    @model_validator(mode='after')
    def _reject_enabled_with_invalid_nice(self) -> 'CpuPriorityConfig':
        if self.enabled and not (1 <= self.nice <= 19):
            raise ValueError(
                f'CpuPriorityConfig.enabled is True but nice={self.nice} is outside '
                'the privilege-free de-prioritizing range 1..19; '
                'set nice to a value in 1..19 or set enabled: false.'
            )
        return self

    def agent_env(self) -> dict[str, str]:
        """Return CPU-priority env dict when enabled, else {}.

        Returns {'DF_AGENT_CPU_NICE': str(self.nice)} when enabled.
        The value is consumed by cli_invoke._cpu_priority_prefix, which pops
        the key (keeping the child env clean) and prepends ['nice', '-n', N]
        to the subprocess argv.

        .. note::
            ``DF_AGENT_CPU_NICE`` is an orchestrator-internal signal and must
            **not** be exported in the parent process environment.  When
            ``enabled=False`` this method returns ``{}`` (no override), so an
            inherited ``DF_AGENT_CPU_NICE`` from the parent shell would pass
            through the ``os.environ`` copy in ``_run_subprocess`` unchanged.
            In practice this is not a concern because the variable is never
            part of a normal login environment — it exists only to carry the
            priority level from ``_build_agent_env`` to ``_cpu_priority_prefix``
            within the same orchestrator invocation.
        """
        if not self.enabled:
            return {}
        return {'DF_AGENT_CPU_NICE': str(self.nice)}


class CpuGovernConfig(BaseModel):
    """CPU cgroup governance for agent inner-loop subprocesses.

    When enabled, resolves ``exec_path`` and ``shim_dir`` against the task
    worktree and injects:

    * ``DF_AGENT_CPU_GOVERN`` — absolute path to reify's ``cpu-governed-exec.sh``,
      consumed by ``cli_invoke._cpu_govern_prefix`` (DF-1), which pops the key
      (keeping the child env clean) and prepends
      ``[<exec>, '--role', 'task', '--']`` to the Claude CLI argv so the agent
      and all cargo/rustc children run inside a ``cpu.weight``-weighted cgroup
      scope.

    * ``PATH`` prepend — ``scripts/agent-bin`` put first so the agent's ad-hoc
      ``cargo …`` (Bash tool) hits the PSI shim instead of the system cargo
      (DF-2).  Unlike ``DF_AGENT_CPU_GOVERN``, PATH is **not** popped — it must
      propagate to the agent and its cargo children.

    **Must default to ``enabled=False``** (fail-open) — unlike default-on
    ``CpuPriorityConfig`` — because governance needs reify-provided paths
    (``cpu-governed-exec.sh``, ``scripts/agent-bin``) that dark-factory cannot
    assume exist at any fixed location.  Reify opts in via its own
    ``orchestrator.yaml`` (sibling task δ, **not** a dependency of ζ), exactly
    like ``JobserverConfig``.

    ``OrchestratorConfig`` uses ``extra='ignore'``, so a ``cpu_governance:``
    block in ``orchestrator.yaml`` would be silently dropped unless this field
    exists.  ``default_factory`` keeps the default instance inert (no
    ``defaults.yaml`` edit required, no reify-file edits — clean
    reciprocal-ownership seam).
    """

    enabled: bool = Field(
        default=False,
        description='Enable CPU cgroup governance for agent subprocesses.',
    )
    exec_path: str = Field(
        default='',
        description=(
            'Worktree-relative (or absolute) path to reify\'s '
            'cpu-governed-exec.sh.  Resolved against the task worktree; '
            'must be executable, else governance is skipped (fail-open).'
        ),
    )
    shim_dir: str = Field(
        default='',
        description=(
            'Worktree-relative (or absolute) path to reify\'s '
            'scripts/agent-bin directory.  Prepended to PATH when resolved.'
        ),
    )

    @staticmethod
    def _resolve(p: str, worktree: 'Path | None') -> 'Path | None':
        """Resolve *p* to an absolute Path, or None if it cannot be resolved.

        * empty string → None
        * absolute → as-is (ignore worktree)
        * relative + worktree None → None (cannot resolve)
        * relative + worktree → worktree / p
        """
        if not p:
            return None
        path = Path(p)
        if path.is_absolute():
            return path
        if worktree is None:
            return None
        return worktree / p

    def resolved_exec_path(self, worktree: 'Path | None') -> 'str | None':
        """Return the resolved, executable exec_path as a string, or None.

        Returns ``None`` when disabled, path is missing, or path is not
        executable — so a bad/missing path always fails open (never breaks a
        spawn).
        """
        if not self.enabled:
            return None
        path = self._resolve(self.exec_path, worktree)
        if path is None:
            return None
        if not os.access(path, os.X_OK):
            return None
        return str(path)

    def resolved_shim_dir(self, worktree: 'Path | None') -> 'str | None':
        """Return the resolved shim_dir as a string if it is a directory, else None."""
        if not self.enabled:
            return None
        path = self._resolve(self.shim_dir, worktree)
        if path is None:
            return None
        if not path.is_dir():
            return None
        return str(path)

    def agent_env(self, worktree: 'Path | None', base_path: str) -> 'dict[str, str]':
        """Return governance env dict when enabled and paths resolve, else {}.

        Mirrors ``JobserverConfig.agent_env``'s FS-checking pattern.

        Returns:
            ``{}`` when disabled or when ``exec_path`` does not resolve to an
            executable.  When resolved:

            * ``DF_AGENT_CPU_GOVERN`` is always included (consumed and popped
              by ``cli_invoke._cpu_govern_prefix``).
            * ``PATH`` is included when ``shim_dir`` resolves to a directory,
              prepending it to *base_path* (pass ``os.environ.get('PATH', '')``
              from the call site so the agent inherits the full system PATH
              with only the shim prepended).
        """
        if not self.enabled:
            return {}
        exec_abs = self.resolved_exec_path(worktree)
        if exec_abs is None:
            return {}
        result: dict[str, str] = {'DF_AGENT_CPU_GOVERN': exec_abs}
        shim_abs = self.resolved_shim_dir(worktree)
        if shim_abs is not None:
            result['PATH'] = (
                f'{shim_abs}{os.pathsep}{base_path}' if base_path else shim_abs
            )
        return result


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
    commit_citation_pattern: str | None = Field(
        default=None,
        description=(
            "Optional override for the reconciler's task-citation pattern "
            "used by git_ops.find_task_citation_commit to gate the "
            "is_ancestor fast-path.  None uses the built-in default "
            "(orchestrator.git_ops.DEFAULT_COMMIT_CITATION_PATTERN), "
            "which matches dark-factory / reify conventional-commit "
            "subjects.  Set to an empty string to disable the citation "
            "check entirely for projects without citation conventions."
        ),
    )
    reap_build_artifact_dirs: list[str] = Field(
        default_factory=lambda: ['target'],
        description=(
            'Regenerable build-output directory names reaped from a done '
            "task's worktree once its merge commit is confirmed on main. "
            'Defaults to [\'target\'] (Rust/Cargo build cache). Override '
            'per project for non-Rust build systems (e.g. [\'build\', '
            '\'dist\']). Uses default_factory to avoid a shared mutable '
            'default across model instances.'
        ),
    )
    main_gate_mark_command: str | None = Field(
        default=None,
        description=(
            'Optional project-configurable shell command run (via ``sh -c`` '
            'in project_root) immediately before the refs/heads/main '
            'update-ref CAS in advance_main.  Intended for reify-style '
            'projects whose ``reference-transaction`` hook (git>=2.28) '
            'requires a one-shot "sanctioned" sentinel to be written before '
            'the orchestrator\'s advance so the hook records the move as '
            'sanctioned rather than UNSANCTIONED.  Set under the ``git:`` '
            'section of orchestrator.yaml.  None (default) => feature off '
            '(no-op) so other projects (autopilot-video, know-live, etc.) '
            'are unaffected.'
        ),
    )
    main_gate_unmark_command: str | None = Field(
        default=None,
        description=(
            'Optional project-configurable shell command run (via ``sh -c`` '
            'in project_root) at the top of the ``if rc != 0:`` block after '
            'a failed update-ref CAS in advance_main.  Clears a sentinel '
            'written by main_gate_mark_command that was not consumed by the '
            'aborted reference-transaction, preventing it from falsely '
            'sanctioning a later non-orchestrator move.  Set under the '
            '``git:`` section of orchestrator.yaml alongside '
            'main_gate_mark_command.  None (default) => feature off (no-op).'
        ),
    )
    persistent_merge_worktree: bool = Field(
        default=False,
        description=(
            'When True, the post-merge verify step reuses a FIXED worktree '
            'at <worktree_dir>/_merge-verify instead of a fresh ephemeral '
            '_merge-<uuid>.  The worktree is reset-in-place to the merge '
            'commit (git reset --hard) and scrubbed of untracked files except '
            'build-artifact dirs (git clean -xfd -e <dir> for each dir in '
            'reap_build_artifact_dirs), retaining target/ warmth across '
            'attempts.  Default False → byte-identical to prior behaviour '
            '(trivially revertible).  Requires _MERGE_AHEAD_BOUND=1 '
            '(serial lane); startup raises PersistentWorktreeConfigError '
            'otherwise.'
        ),
    )
    persistent_merge_worktree_safety_valve_every_n: int = Field(
        default=0,
        ge=0,
        description=(
            'When >0 and persistent_merge_worktree is on, every Nth verifying '
            'serial attempt runs a from-scratch cold verify in a throwaway '
            'ephemeral worktree (target NOT retained); its pass/fail flows '
            'through the existing verify gate (a cold failure on the serial '
            'lane is the alarm).  0 disables the safety valve.'
        ),
    )
    warm_verify_shadow_compare: bool = Field(
        default=False,
        description=(
            'Master enable for PRD §10 invariant 6(b) same-candidate warm-vs-cold '
            'SHADOW compare.  When True (and persistent_merge_worktree is on), '
            'after each warm-verified land the orchestrator periodically spawns '
            'an ASYNC (off the serial lane) cold re-verify against the same '
            'just-landed commit and compares results TEST-LEVEL.  Any per-test '
            'divergence fires a born-at-L2 alarm.  Default False; reify opts in '
            'via orchestrator.yaml.  Cadence is controlled by '
            'warm_verify_shadow_compare_every_n_merges and '
            'warm_verify_shadow_compare_nightly_interval_secs (whichever '
            'fires sooner).  Only meaningful when persistent_merge_worktree is '
            'also on.'
        ),
    )
    warm_verify_shadow_compare_every_n_merges: int = Field(
        default=40,
        ge=0,
        description=(
            'When warm_verify_shadow_compare is on, trigger a shadow compare '
            'after every Nth successful warm-verified land.  0 disables the '
            'merge-count leg; the nightly-timer leg still applies.  Part of '
            'the \"nightly OR every-N-merges, whichever sooner\" = OR cadence.'
        ),
    )
    warm_verify_shadow_compare_nightly_interval_secs: float = Field(
        default=86400.0,
        gt=0,
        description=(
            'When warm_verify_shadow_compare is on, trigger a shadow compare '
            'if at least this many seconds have elapsed since the last run.  '
            'Default 86400 s (nightly).  Part of the '
            '\"nightly OR every-N-merges, whichever sooner\" = OR cadence.  '
            'This is a SHADOW/detective control that runs off the serial merge '
            'lane and never blocks a warm land.'
        ),
    )


class VerifyRunnerConfig(BaseModel):
    """Configuration for a single remote verify runner (Lever C).

    Each entry in OrchestratorConfig.verify_runners describes one remote host
    that participates in the multi-host merge-verify pool.  Follows the
    SccacheConfig/GitConfig BaseModel+Field pattern.

    Note: OrchestratorConfig uses extra='ignore' (config.py:1414); prior to
    adding this model, a verify_runners: block in orchestrator.yaml was
    silently inert.  Adding this field makes the block live.
    """

    name: str = Field(description='Short identifier for this runner (e.g. "laptop").')
    ssh_host: str = Field(description='SSH host string used for git push and ssh invocations.')
    git_remote: str = Field(description='Git remote name pointing to the remote host.')
    config_path: str | None = Field(
        default=None,
        description=(
            'Path to the orchestrator YAML config on the remote host.  '
            'Passed as --config to the remote orchestrator verify-merge CLI.  '
            'None omits --config (remote uses its own ORCH_CONFIG_PATH).'
        ),
    )
    enabled: bool = Field(
        default=True,
        description=(
            'When False, this runner is excluded from the active pool.  '
            'Allows temporary disabling without removing the entry.'
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

_DISCOVERY_EXCLUDED_DIRS = frozenset({
    '.git', '.venv', 'venv', '.worktrees',
    'node_modules', '__pycache__', 'build', 'target', '.gradle',
    # Leftover/backup worktree & build dirs and tooling worktrees.  These hold
    # full project checkouts — each carrying a copy of the root
    # orchestrator.yaml — which would otherwise be mis-registered as phantom
    # module configs (see incident: 224 `.worktrees.old/<id>` modules drove a
    # 226-way merge-verify fan-out into a single worktree).  Belt-and-braces:
    # the `.git`-boundary prune in the walk below catches these (and any other
    # nested checkout) generically; the static names keep discovery cheap and
    # the intent explicit.
    '.worktrees.old', 'target.old', '.claude',
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
    """Walk *project_root* recursively for ``orchestrator.yaml`` files and load overridable fields.

    Contract:
    - Prefix is the POSIX-style relative path from *project_root* (e.g. ``dashboard`` or
      ``foo/bar``).  A root-level ``orchestrator.yaml`` (prefix ``"."``) is skipped because
      it is the top-level orchestrator config, not a module config.
    - Excludes standard build/VCS dirs (``_DISCOVERY_EXCLUDED_DIRS``) by pruning ``dirnames``
      in place during the walk so their subtrees are never visited.  Exclusions are
      **name-based** and applied at *every* level of the tree, so any directory with a
      reserved name — however deeply nested — is silently skipped.  If a legitimate module
      directory shares a name with a reserved dir (e.g. a module literally called ``build``),
      its ``orchestrator.yaml`` will not be discovered; rename the directory or place the
      config at a depth that avoids the collision.  When a pruned directory **directly**
      contains an ``orchestrator.yaml`` (a "shadow" config, i.e. ``<reserved>/orchestrator.yaml``
      with no intervening subdirectory), a runtime warning is emitted so operators can detect
      and resolve the collision rather than having the config silently disappear.
      **Limitation**: an ``orchestrator.yaml`` nested *deeper* inside a pruned directory
      (e.g. ``build/some_sub/orchestrator.yaml``) is **still silently dropped** — the walk
      never descends into the pruned tree, so no stat is performed for deeper levels.  Only
      the immediate child case is diagnosed.
    - Uses ``followlinks=False`` (passed explicitly) for symlink-cycle safety — a symlink that
      points back into an ancestor directory cannot drive infinite recursion because the walk
      will not follow it.
    - Results are inserted into the returned dict in ``(depth, lex)`` order so iteration is
      deterministic regardless of filesystem order.

    Performance note:
    This performs a full recursive walk of *project_root* on every call.  In the normal flow
    it is called once at startup via ``load_config``, which stores the result (even an empty
    ``{}``) in ``OrchestratorConfig._module_configs``.  ``run_full_verification`` reuses that
    cached dict whenever ``_module_configs is not None`` (the ``None`` sentinel means
    "discovery never ran") and *project_root* resolves to the same absolute path as
    ``config.project_root``.  This means the typical case — including a monorepo with no
    subproject yamls, which stores ``{}`` — truly avoids the redundant walk on every
    ``run_full_verification`` call.  On large repositories with many non-excluded
    subdirectories the walk can be noticeable; adding project-specific directories to
    ``_DISCOVERY_EXCLUDED_DIRS`` or reducing the nesting depth of your module layout will
    help when the fallback walk path is exercised.

    Depth and scheduler coherence:
    ``OrchestratorConfig.for_module`` resolves configs via longest-matching prefix walk, but
    the scheduler (``_limit_for``) and workflow (``_resolve_module_configs``) always pass paths
    that have been truncated to ``lock_depth`` components by ``shared.locking.normalize_lock``.
    A module config whose prefix has *more* components than ``lock_depth`` will be honoured by
    ``run_full_verification`` (which iterates ``module_configs.values()`` directly) but will be
    unreachable through the scheduler/workflow path.  ``load_config`` emits a warning when this
    mismatch is detected so misplaced ``orchestrator.yaml`` files are surfaced rather than
    silently half-applied.
    """
    found: list[tuple[str, Path]] = []
    # followlinks=False (the default, stated explicitly so a future refactor cannot flip it
    # silently): prevents infinite recursion from self-referencing symlink cycles.
    for dirpath, dirnames, filenames in os.walk(project_root, followlinks=False):
        # Warn about pruned directories that directly contain an orchestrator.yaml
        # (one cheap stat per about-to-be-excluded sibling; no descent into the tree).
        for d in dirnames:
            if d in _DISCOVERY_EXCLUDED_DIRS:
                shadow = Path(dirpath) / d / 'orchestrator.yaml'
                if shadow.is_file():
                    rel = shadow.parent.relative_to(project_root)
                    logger.warning(
                        'Skipping orchestrator.yaml under pruned directory %s '
                        '(reserved name %r); rename the directory or remove the '
                        'file to suppress this warning.',
                        rel,
                        d,
                    )
        # Prune (a) reserved names and (b) nested checkouts/worktrees.  Any
        # subdirectory that carries its own ``.git`` entry (a worktree's ``.git``
        # *file*, a clone/submodule's ``.git`` *dir*) is a separate working tree,
        # NOT a subproject of THIS project.  Each such checkout contains a copy of
        # the root orchestrator.yaml; descending into it would mis-register a
        # phantom module per nested checkout.  This generic guard is naming-
        # independent — it catches `.worktrees.old/<id>`, `.claude/worktrees/*`,
        # and any future stray checkout regardless of directory name.  The walk
        # root (project_root) is never re-examined here, so the main repo's own
        # ``.git`` cannot prune the root.
        nested_checkouts = {
            d for d in dirnames
            if d not in _DISCOVERY_EXCLUDED_DIRS
            and (Path(dirpath) / d / '.git').exists()
        }
        for d in nested_checkouts:
            logger.debug(
                'Module discovery: pruning nested checkout %s (has its own .git)',
                (Path(dirpath) / d),
            )
        dirnames[:] = [
            d for d in dirnames
            if d not in _DISCOVERY_EXCLUDED_DIRS and d not in nested_checkouts
        ]
        if 'orchestrator.yaml' not in filenames:
            continue
        yaml_path = Path(dirpath) / 'orchestrator.yaml'
        rel = yaml_path.parent.relative_to(project_root)
        prefix = rel.as_posix()
        if prefix == '.':
            # Root-level orchestrator.yaml is the top-level config, not a module config
            continue
        found.append((prefix, yaml_path))
    # Sort by (depth, lex) so dict insertion order is deterministic
    found.sort(key=lambda item: (item[0].count('/'), item[0]))
    configs: dict[str, ModuleConfig] = {}
    for prefix, yaml_path in found:
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

    # Run-forever idle poll cadence (seconds).  When the queue drains and the
    # orchestrator is running forever (no --until-idle), it idles and re-polls
    # the task tree every this many seconds for newly-scheduled work.  Kept
    # separate from the paused-idle constant (_PAUSED_IDLE_POLL_SECS) so the
    # poll cadence can be tuned independently of the pause-recovery cadence.
    idle_poll_secs: float = Field(default=15.0)

    # Iteration limits
    max_execute_iterations: int = Field(default=10)
    max_verify_attempts: int = Field(default=5)
    # Fast-fail cap for ``infra_timeout`` results whose cause_hint is the
    # verifier's own injected ``Command timed out after Ns: …`` wrapper string
    # — there is no actionable signal for the debugger to chase, so retrying
    # up to ``max_verify_attempts`` just burns ~50 min of budget per task.
    # After the streamed-output fix in ``_run_cmd`` a real cause hint should
    # surface on attempt 1 for any genuine in-test hang; default 2 leaves
    # one free retry to absorb stray transient infra blips.
    max_opaque_timeout_attempts: int = Field(default=2, ge=1)
    max_review_cycles: int = Field(default=2)
    reviewer_stagger_secs: float = Field(default=2.0)
    max_reviewer_retries: int = Field(default=4)
    # Max in-workflow amendment rounds after a PASS-with-suggestions review.
    # Each round reinvokes the implementer with in-scope suggestions (scoped
    # by module-lock membership), re-verifies, and re-reviews. Remaining
    # out-of-scope or cap-exhausted suggestions still flow through the
    # existing escalate_suggestions path.
    max_amendment_rounds: int = Field(default=1)
    # Circuit-breaker threshold for consecutive fresh-invocation zero-output
    # timeouts (timed_out=True, turns=0, cost_usd=0.0).  After this many
    # consecutive such timeouts the orchestrator fast-fails to BLOCKED with an
    # infra_issue category instead of burning the full max_execute_iterations
    # budget (~3.3h at 10 iterations × 20 min each).  Default 2 leaves one
    # free retry to absorb a single transient infra blip while still catching
    # the deterministic-wedge pattern observed in reify-4429 (10/10 hung).
    max_consecutive_zero_output_timeouts: int = Field(default=2, ge=1)
    # Whether to recycle (cleanup + recreate) the per-task TaskConfigDir
    # between sub-threshold zero-output retries.  Per-task CLI state is the
    # prime suspect for deterministic wedges; turns==0 means the destroyed
    # session did no work, so discarding it cannot lose progress and aligns
    # with crash-recovery semantics.  Disabled on the final tripping iteration
    # (that dir is preserved for forensics regardless of this flag).
    recycle_config_dir_on_zero_output: bool = Field(default=True)

    # Completion judge — opt-in loop-exit hint after each implementer iteration.
    # Default False: production orchestrator runs unaffected. Eval runner
    # enables this per-task (see evals/runner.py build_eval_orch_config).
    judge_after_each_iteration: bool = Field(default=False)

    # Merge conflict reduction
    max_advance_attempts: int = Field(default=3)
    max_pre_merge_retries: int = Field(default=2)
    max_merge_retries: int = Field(default=3)
    inter_iteration_rebase: bool = Field(default=True)
    # Fix 3 — rebase onto main before each verify (including the first) so
    # transient verify failures fixed by a sibling task on main are picked
    # up without a full re-execute cycle.  Cheap fast-path when main has
    # not advanced; default True closes the demonstrated bug, ops can opt
    # out if it surprises us.
    rebase_before_verify: bool = Field(default=True)
    # Fix 2 — thrash threshold for repeated infra-issue resumes on the
    # same root cause.  Counter increments when an L0 (category=
    # infra_issue) is resolved without iteration-log growth, resets to 1
    # when the log grows (steward/agent ran real work).  At threshold the
    # orchestrator promotes to L1 instead of dispatching the implementer
    # again.  Three matches the empirical reify task-2289 thrash window
    # (15 escalations on the same port-1420 collision before
    # verify-budget exhaustion).
    max_consecutive_infra_resumes: int = Field(default=3, ge=1)
    # Fix 3 — thrash threshold for repeated steward-resolved merge-phase
    # failures with the same outcome signature.  Counter increments when
    # the merge queue returns a blocked outcome whose signature matches
    # the previous attempt; it resets to 1 on a different verdict.  At
    # threshold the orchestrator escalates to L1 instead of resubmitting
    # the same merge.  Default 2 — two identical verdicts is enough to
    # call it a loop in the merge phase (the steward resolution between
    # them is the mediation we already gave it a chance to perform).
    max_consecutive_merge_thrash: int = Field(default=2, ge=1)
    # Cross-project external-dep grace threshold — after N consecutive ticks
    # where an external dep resolves to an unresolvable sentinel
    # (unknown_project / unknown_task / malformed), escalate to L1.  Default 3
    # mirrors max_consecutive_infra_resumes — matches the nearest existing
    # thrash guard per PRD open question 1.  A named field (rather than reusing
    # max_consecutive_infra_resumes directly) lets tests set a low threshold
    # without affecting the infra-resume gate.
    max_external_dep_unresolved_cycles: int = Field(default=3, ge=1)
    # Verify-loop signature-repetition cap — after N consecutive verify
    # failures whose (category, normalised cause_hint) tuple is identical the
    # loop escalates to L1 (WorkflowOutcome.BLOCKED) instead of burning the
    # remaining max_verify_attempts budget on futile debug/retry cycles.
    # Normalisation strips file:line numeric tails, ANSI colour escapes,
    # collapses whitespace, and lowercases so superficial textual variation
    # (line numbers shifting, colour codes) does not defeat the equality check.
    # Default 3 mirrors max_consecutive_infra_resumes — empirically, three
    # identical failures with no variation in error text means the debugger
    # is not making progress and human review is the right next step.
    max_failure_signature_repeat: int = Field(default=3, ge=1)
    # Verify-phase broken-main contagion guard.  Before invoking the debugger
    # on a verify failure, detect whether the SAME (category, normalised
    # cause_hint) signature is present on the current merge-base/main.  If so,
    # classify the failure as inherited (not this task's own), skip
    # self-patching, and escalate ONCE (deduped across N sibling tasks that
    # see the same inherited break) so a single hotfix lands instead of N
    # conflicting duplicate patches.  Default True closes the demonstrated
    # StatusBar.tsx TS2769 contagion incident; ops can opt out if needed.
    escalate_preexisting_main_break: bool = Field(default=True)
    merge_verify_min_free_disk_bytes: int = Field(
        default=10 * 1024**3,
        description=(
            'Pre-verify disk guard threshold. Before post-merge verify, if free '
            'space on the merge-worktree volume is below this, prune stale _merge-* '
            'worktrees; if still below after pruning, skip verify and escalate as '
            'transient infra (disk pressure) instead of running a doomed build.'
        ),
    )
    requeue_cooldown_secs: float = Field(default=30.0)
    # Per-task settle window applied after a dispatch when reconciliation or
    # steward signals are present (recon_reset_count > 1, steward_clear_at,
    # recon_stage2_blocked_at, or reopen_reason containing 'steward').  The
    # gate prevents the orchestrator from immediately re-grabbing a task that
    # was just reset/cleared, giving reconciliation time to settle.  The 5-min
    # floor (ge=300.0) prevents setting a value so small it reproduces the
    # original tight-loop pathology.
    dispatch_cooldown_secs: float = Field(
        default=1800.0,
        ge=300.0,
        description=(
            'Per-task settle window (seconds) applied after dispatch when '
            'reconciliation/steward signals are present.  Default 1800s (30 min); '
            'minimum 300s (5 min floor).'
        ),
    )
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
    transient_requeue_cap: int = Field(
        default=10,
        ge=1,
        description=(
            'Higher ceiling for WorkflowOutcome.REQUEUED iterations caused by '
            'transient external-API failures (HTTP 529/5xx "agent API error" '
            'summaries).  These are exempt from `requeue_cap` and counted '
            'separately in `_transient_requeue_counts`.  Guards against '
            'unbounded retry on a persistent provider outage.  Process-local; '
            'orchestrator restart resets it.  Cleared alongside the genuine '
            'counter on DONE or cap-exhaust.'
        ),
    )
    snapshot_min_write_interval_secs: float = Field(
        default=0.25,
        ge=0.0,
        description=(
            'Minimum wall/monotonic interval (seconds) between scheduler '
            'state-snapshot disk writes.  Ticks within the window are '
            'coalesced: the time-gate check is O(1) (a single monotonic '
            'subtraction) so throttled ticks pay no serialisation or I/O '
            'cost.  Default 250 ms coalesces the ~20 tick/s burst typical '
            'during pin-queue drains (~1 500 tasks, < 1 MB snapshot).  '
            'Set to 0.0 to disable throttling (all ticks write) — useful '
            'for tests that need to isolate the content-dedup path.'
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

    # ── Merge-verify scoping & fan-out bounds (storm guard) ──────────────
    # When True, post-merge verify bypasses per-subproject scoping/fan-out and
    # runs the project-wide command once (the same `force_workspace` path train
    # members use).  Correct for single-workspace projects (e.g. a cargo
    # workspace verified with `--scope all`), where scoping at merge time is a
    # no-op and per-module fan-out across a large module set can launch N full
    # builds into one merge worktree.  Defaults False to preserve the existing
    # scoped/fan-out behaviour for multi-subproject projects.
    # Per-command timeout for cold merge-verify runs.  Applied by
    # _resolve_verify_timeout when is_merge_verify=True and is_cold=True,
    # BEFORE the module.cold → config.cold → warm cascade.  When None, the
    # resolver falls back to verify_cold_command_timeout_secs then warm
    # (byte-identical to behaviour before task 1603).
    #
    # Motivation: task 1602 made the post-merge type-check gate fail-CLOSED
    # on timeout, so a cold build that overruns the warm budget (1800 s) is
    # now a queue-stalling block rather than a fail-open.  This knob lets ops
    # raise the merge path's cold budget without lengthening task-phase cold
    # verifies.  Shipped default (defaults.yaml): 7200 s (4× warm; one step
    # above the 90-min/5400 s general cold), giving headroom for a cold
    # full-workspace compile + frontend install.  Pydantic default is None so
    # configs that do not merge defaults.yaml keep the old fallback behaviour.
    merge_verify_cold_command_timeout_secs: float | None = Field(default=None)
    merge_verify_workspace: bool = Field(default=False)
    # Train-former opt-in knob (β).  OFF by default so β can land before γ/δ
    # complete the full stack; an always-on former would assign metadata.train
    # and route members to merge-deferred with no stacking (γ) in place, which
    # would strand members indefinitely.  Projects opt in once γ,δ,ε land.
    merge_train_former_enabled: bool = Field(default=False)
    # Retroactive coalescing pass (γ/1719).  OFF by default — human-flips after
    # soak (fold-the-decision norm).  Requires merge_train_max_members >= 2.
    # When False (default) the merger loop is byte-identical to pre-γ behaviour.
    merge_train_coalesce_enabled: bool = Field(default=False)
    # Maximum members per train (inclusive of the anchor).  Defaults to 3 per
    # the s(N) go/no-go gate resolved GO at N=3 (reify esc-4455-16;
    # s(3)=0.962 ≫ 1/3, true coupling-failure rate 0/104 at N≥3).  ge=2
    # because a single-member "train" is meaningless.
    merge_train_max_members: int = Field(default=3, ge=2)
    # Upper bound on concurrent per-subproject ``run_verification`` calls inside
    # a single ``run_scoped_verification`` fan-out.  Caps the blast radius if the
    # module set is large (or accidentally polluted) so a fan-out can never spawn
    # an unbounded number of full builds into one worktree at once.
    max_concurrent_module_verifies: int = Field(default=4, ge=1)
    # When True, each verify command is spawned inside a transient systemd
    # ``--scope`` (its own cgroup) so a timeout/cancel can kill the ENTIRE
    # subtree by cgroup, regardless of process-group escapes (e.g. an inner GNU
    # `timeout` that setpgid'd cargo into a separate group, which defeats
    # killpg).  Defaults False (use start_new_session + killpg) so behaviour and
    # the existing test suite are unchanged; opt in per project where
    # `systemd-run --user` is available.
    verify_use_cgroup_scope: bool = Field(default=False)

    # Steward lifecycle
    steward_lifetime_budget: float = Field(default=12.0)
    steward_max_attempts: int = Field(default=1)
    steward_completion_timeout: float = Field(default=900.0)
    steward_max_timeouts_per_escalation: int = Field(default=3, ge=2, le=5)
    steward_max_empty_outputs_per_escalation: int = Field(default=2, ge=2, le=4)

    # Pre-triage threshold for review suggestions
    suggestion_triage_threshold: int = Field(default=10)

    # ── Architect cost optimisations ────────────────────────────────────
    # Lever B — skip the revalidation architect call when the diff main has
    # gained since the prior plan was stamped does not overlap the plan's
    # files. The orchestrator updates _revalidated_at and base_commit
    # directly, mirroring the confirm_plan MCP write semantics.
    revalidation_skip_enabled: bool = Field(default=True)
    max_revalidation_age_hours: float = Field(
        default=24.0,
        description=(
            'Maximum age (hours) of the prior plan provenance for the '
            'overlap=0 short-circuit to apply. Older plans always go through '
            'a full architect revalidation regardless of overlap.'
        ),
    )

    # Lever C — replace architect+implementer with a single SIMPLE_TASK
    # (sonnet) agent when the classifier matches a trivial doc/comment/
    # rename/typo task with at most a couple of files in scope.
    simple_task_enabled: bool = Field(default=True)
    simple_task_budget_usd: float = Field(default=1.50)
    simple_task_max_turns: int = Field(default=30)

    # Auto-eval — when the optimistic path (B-skip or C-simple) blocks at
    # plan/execute/verify/review, automatically rerun the same task from the
    # same branchpoint via the full architect path. The original branch and
    # worktree are renamed with a `-skip-attempt` suffix; the redo is
    # submitted via planning_mode (bypassing curator dedupe) and dispatched
    # in `in-progress` state so the harness picks it up directly.
    auto_eval_enabled: bool = Field(default=True)
    auto_eval_redo_budget_usd: float = Field(
        default=50.0,
        description=(
            'Daily USD budget cap for auto-eval redo invocations. Computed '
            'as the sum of cost_usd from the invocations table for tasks '
            'with metadata.auto_eval_redo=True in the trailing 24h.'
        ),
    )
    auto_eval_phases: set[str] = Field(
        default_factory=lambda: {'plan', 'execute', 'verify', 'review'},
        description=(
            'Block phases that trigger auto-eval. Merge/infra failures are '
            'excluded — they are unlikely to differ on the full path.'
        ),
    )

    # Worktree identity guard (Fix C) — on crash-recovery / worktree reuse,
    # compare the worktree's stored title against the live DB task's title and
    # quarantine on mismatch.  Catches a recycled task id whose orphaned
    # worktree carries unrelated WIP (reify task 3770).  Fails open when either
    # title is missing.
    worktree_identity_guard_enabled: bool = Field(default=True)
    # Orphan worktree reaper (Fix B) — at startup, sweep worktrees whose numeric
    # id no longer maps to a live task: quarantine those with commits/dirty WIP,
    # reap the provably-empty ones, then prune stale git admin entries.
    worktree_orphan_reaper_enabled: bool = Field(default=True)

    # Post-merge staleness hook — restarts fused-memory.service exactly once
    # (debounced) after a merge whose landed diff touches fused-memory/src/.
    # Fires only at the orchestrator's idle quiet-window (no dispatched agents).
    # See orchestrator/src/orchestrator/service_restart.py for policy details.
    fused_memory_restart_on_merge_enabled: bool = Field(default=True)
    fused_memory_restart_debounce_secs: float = Field(default=120.0)
    fused_memory_restart_watch_prefixes: list[str] = Field(
        default_factory=lambda: ['fused-memory/src/']
    )
    fused_memory_restart_script: str = Field(
        default='scripts/restart-fused-memory.sh'
    )

    # Orphan L0 reaper — re-escalates level-0 escalations whose task has no
    # active workflow/steward (e.g. escalations emitted by the deep reviewer
    # against a synthetic ``review-*`` task_id).  Without this, such
    # escalations sit pending until the next orchestrator restart dismisses
    # them unread.  Set ``orphan_l0_reaper_enabled = False`` to disable.
    orphan_l0_reaper_enabled: bool = Field(default=True)
    orphan_l0_timeout_secs: float = Field(default=600.0)
    orphan_l0_check_interval_secs: float = Field(default=60.0)

    # Terminal-status watcher — periodically polls fused-memory for active
    # workflow tasks whose status has gone terminal out-of-band (typical
    # cause: a human marked a task ``done`` and removed its worktree while
    # the orchestrator was still in the merge phase).  When detected the
    # workflow's ``cancel_event`` is set so it exits cleanly without
    # cascading into escalations.  See zombie-escalation fix Step 5.
    terminal_status_watcher_enabled: bool = Field(default=True)
    terminal_status_poll_interval_secs: float = Field(default=30.0)
    # Number of consecutive terminal-status watcher polls a workflow may ignore
    # the soft cancel_event before the watcher escalates to a hard
    # asyncio.Task.cancel().  At the 30 s default poll interval this is ~90 s
    # of grace before the hard-cancel fires.  Must be >= 1 so at least one
    # soft-cancel attempt is always made before escalation.
    terminal_status_hard_cancel_polls: int = Field(
        default=3,
        ge=1,
        description=(
            'Number of consecutive terminal-status watcher polls a workflow may '
            'ignore the soft cancel event before the watcher escalates to a hard '
            'asyncio.Task.cancel(); at the 30s default poll this is ~90s.'
        ),
    )

    # Stranded-in-progress reconcile sweep — periodic re-run of the
    # startup ``_reconcile_stranded_in_progress`` pass during a long run.
    # Catches tasks stranded by transient backend failures (taskmaster
    # restart, fused-memory crash) that the workflow's own retry layer
    # exhausted; they sit in-progress with no live claimant otherwise.
    # Also fires opportunistically when the main loop is stuck-blocked
    # (no acquirable assignment, no active workflows) — see Fix 4.
    stranded_reconcile_enabled: bool = Field(default=True)
    stranded_reconcile_interval_secs: float = Field(default=900.0)

    # Backstop for the stranded-`blocked` gap: a task left `blocked` with no
    # open escalation AND no active workflow is an orphaned recovery (its
    # blocking escalation was resolved directly with no live workflow to
    # re-pend it — the 3576 incident, 2026-05-29).  The stranded-reconcile
    # sweep deliberately refuses `blocked`→`pending` (only flips to `done` on
    # on-main evidence), so such a task falls between every recovery mechanism.
    # When enabled, the sweep re-files a single L1 (it NEVER changes status —
    # re-filing can't yank a deliberate `release_workflow` blocked-park) so a
    # human/auto-watcher re-triages; the event-driven re-pend
    # (``_on_escalation_resolved``) then performs the actual flip once that L1
    # is resolved.  Self-dedupes via the pending-escalation check.
    stranded_blocked_escalate_enabled: bool = Field(default=True)

    # Scheduler park-and-stop (AFK hardening, task 1322).
    # When park_stop_parked_threshold tasks transition to 'blocked' within
    # park_stop_parked_window_hours, the scheduler is paused automatically.
    # Pause persists across orchestrator restart via the scheduler_state table
    # in runs.db.  Resume is human-driven (Harness.resume_scheduler()).
    # Sibling tasks 1323 (cost ceiling) and 1327 (EWA digest) call
    # Harness.pause_scheduler(reason) directly to use the same mechanism.
    park_stop_enabled: bool = Field(
        default=True,
        description=(
            'Enable the park-and-stop trip mechanism. When disabled, blocked '
            'transitions are still recorded in the deque (so a runtime '
            'flip-on takes effect immediately) but the trip callback never fires.'
        ),
    )
    park_stop_parked_threshold: int = Field(
        default=15,
        ge=1,
        description=(
            'Number of tasks transitioned to blocked within '
            'park_stop_parked_window_hours to trip the scheduler park-stop pause.'
        ),
    )
    park_stop_parked_window_hours: float = Field(
        default=1.0,
        gt=0,
        description=(
            'Rolling-window length (hours) for the park-stop parked-count threshold.'
        ),
    )

    # Escalation-watcher subprocess supervisor (AFK hardening, task 1326).
    # Keeps a fresh escalation-watcher-auto agent alive across multi-day AFK
    # windows with rotation, exponential backoff, and a crashloop→pause_scheduler
    # guard.  The supervisor restarts the agent after each clean rotation exit
    # (agent self-exits after ROTATION_ESCALATIONS or ROTATION_HOURS) with no
    # backoff.  Unclean exits (crash, timeout-kill) incur exponential backoff.
    # When ≥watcher_max_crashloop_restarts unclean exits occur within
    # watcher_crashloop_window_secs, pause_scheduler('watcher_crashloop') is
    # called and the supervisor stops.
    #
    # Sibling tasks: 1321 (L1 persistence), 1322 (park-and-stop),
    # 1323 (cost-ceiling), 1325 (unblock-auto), 1327 (EWA digest).
    watcher_supervisor_enabled: bool = Field(default=True)
    watcher_subprocess_restart_backoff_secs: float = Field(default=30.0)
    watcher_rotation_escalations: int = Field(default=50)
    watcher_rotation_hours: float = Field(default=4.0)
    watcher_max_crashloop_restarts: int = Field(default=5)
    watcher_crashloop_window_secs: int = Field(default=600)
    # Cost-runaway guard for degenerate-clean exits (task 1388).
    # A clean rotation shorter than watcher_misconfigured_min_rotation_secs seconds
    # is classified as degenerate (empty queue, SKILL.md drift, misconfigured env).
    # After watcher_max_misconfigured_clean_exits such exits within
    # watcher_crashloop_window_secs, pause_scheduler('watcher_misconfigured') is
    # called and the supervisor stops.  Reuses watcher_crashloop_window_secs as the
    # burst-detection window — semantically identical for both failure modes.
    watcher_misconfigured_min_rotation_secs: float = Field(default=120.0)
    watcher_max_misconfigured_clean_exits: int = Field(default=5)

    # Invocation knobs for each watcher rotation (per UnblockAutoConfig precedent).
    # watcher_rotation_budget_usd is sized for a full 4h rotation at opus rates;
    # using invoke_agent's default $5 would exhaust within minutes and falsely
    # trip the crashloop guard.
    watcher_model: str = Field(default='opus')
    watcher_rotation_budget_usd: float = Field(default=40.0)
    watcher_max_turns: int = Field(default=400)
    watcher_effort: str = Field(default='high')
    watcher_backend: str = Field(default='claude')

    # Daily cost ceilings (AFK hardening, task 1323).
    # Both measure the trailing-24h SUM(cost_usd) from the invocations table.
    # The watcher ceiling is the early-warning trip (runaway dispatch is the
    # most likely cost source); the orch-wide ceiling is the final safety net.
    # On breach, Harness.pause_scheduler() is called with reason
    # 'cost_ceiling_watcher_exceeded' or 'cost_ceiling_orch_exceeded'.
    # Watcher check runs first; when both would trip, the watcher reason wins.
    # Disable either ceiling by setting to a very high value (no enable-flag
    # needed — mirrors auto_eval_redo_budget_usd pattern).
    watcher_daily_cost_ceiling_usd: float = Field(
        default=50.0,
        description=(
            'Daily USD ceiling for escalation-watcher invocations (trailing 24h '
            'sum of cost_usd WHERE role LIKE \'%watcher%\'). On breach, '
            'pause_scheduler is called with reason cost_ceiling_watcher_exceeded. '
            'Task 1323.'
        ),
    )
    orch_daily_cost_ceiling_usd: float = Field(
        default=200.0,
        description=(
            'Daily USD ceiling for ALL orchestrator invocations (trailing 24h '
            'sum of all cost_usd rows). On breach, pause_scheduler is called '
            'with reason cost_ceiling_orch_exceeded. Task 1323.'
        ),
    )

    # Digest + EWA trip (AFK hardening, task 1327).
    # Every digest_every_n_escalations escalation-lifecycle events (both submit
    # AND resolve callbacks each count), _maybe_write_digest() writes an append-only
    # markdown file to digest_dir summarising recent activity.
    # The EWA of escalations/done is updated each digest step; when it exceeds
    # digest_ewa_threshold, Harness.pause_scheduler('ewa_trip_<value>') is called.
    # digest_ewa_alpha: smoothing factor for EWA(t+1) = alpha*(esc/max(done,1)) + (1-alpha)*EWA(t).
    # digest_ewa_threshold default: reify 23-day baseline mean+2σ ≈ 24.6; see task 1327 notes.
    # EWA state is process-local (reset on orchestrator restart — consistent with
    # park-stop and watcher-supervisor counters, documented in design decisions).
    # EWA is also reset to 0.0 on resume_scheduler() when pause was caused by ewa_trip.
    digest_enabled: bool = Field(
        default=True,
        description=(
            'Enable the per-N-escalation digest and EWA trip mechanism. '
            'When False, no digest files are written and EWA is not tracked. Task 1327.'
        ),
    )
    digest_every_n_escalations: int = Field(
        default=10,
        ge=1,
        description=(
            'Number of escalation-lifecycle events (BOTH submit AND resolve callbacks '
            'each increment this counter — a single escalation that is later resolved '
            'contributes 2 events) that must accumulate since the last digest before '
            'the next digest is written. A value of 10 therefore means ~5 distinct '
            'escalations resolved, or ~10 unresolved escalations submitted. Task 1327.'
        ),
    )
    digest_dir: str = Field(
        default='',
        description=(
            'Directory for digest markdown files. Empty string (default) resolves to '
            '<project_root>/data/digests/. Task 1327.'
        ),
    )
    digest_ewa_alpha: float = Field(
        default=0.3,
        gt=0.0,
        le=1.0,
        description=(
            'Smoothing factor for the escalation/done EWA. '
            'EWA(t+1) = alpha*(esc/max(done,1)) + (1-alpha)*EWA(t). Task 1327.'
        ),
    )
    digest_ewa_threshold: float = Field(
        # Rounded from reify 23-day baseline: EWA-smoothed(alpha=0.3) daily
        # escalation/done ratio, mean=21.05 + 2*stddev=3.51 ≈ 24.56.
        # Full derivation in task 1327 completion notes (fused-memory).
        # Re-derive with: walk reify/data/escalations/ (23 days), get daily
        # done counts, compute ratio/day, EWA-smooth with alpha=0.3, mean+2σ.
        default=24.6,
        gt=0.0,
        description=(
            'EWA threshold above which the scheduler is paused via '
            'pause_scheduler(\'ewa_trip_<value>\'). Default derived from '
            'reify 23-day escalation/done ratio history (mean+2σ≈24.6). '
            'EWA starts at 0.0 on process start; reaching 24.6 from a cold '
            'start requires sustained high ratios across multiple digest steps. '
            'Task 1327.'
        ),
    )

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
    # Git
    git: GitConfig = Field(default_factory=GitConfig)

    # κ: shared sccache backend (the laptop warm multiplier)
    # An absent stanza in orchestrator.yaml yields the disabled default;
    # no defaults.yaml edit is required.
    sccache: SccacheConfig = Field(default_factory=SccacheConfig)

    # Jobserver wiring for agent inner-loop cargo builds (implementer/debugger/architect).
    # An absent stanza in orchestrator.yaml yields the disabled default.
    # Reify enables this in /home/leo/src/reify/orchestrator.yaml.
    jobserver: JobserverConfig = Field(default_factory=JobserverConfig)

    # CPU nice de-prioritization for agent inner-loop subprocesses.
    # An absent stanza in orchestrator.yaml yields the enabled-by-default instance;
    # no extra orchestrator.yaml edit is required — the reify restart alone activates it.
    cpu_priority: CpuPriorityConfig = Field(default_factory=CpuPriorityConfig)

    # CPU cgroup governance for agent inner-loop subprocesses (DF_AGENT_CPU_GOVERN +
    # scripts/agent-bin PATH prepend).  An absent stanza yields the disabled default
    # (fail-open — governance needs reify-provided paths that dark-factory cannot
    # assume exist).  Reify opts in via its own orchestrator.yaml (task δ, NOT a
    # dependency of ζ), exactly like jobserver.
    cpu_governance: CpuGovernConfig = Field(default_factory=CpuGovernConfig)

    # Lever C: remote verify runner pool configuration.
    # Adding this field makes a verify_runners: block in orchestrator.yaml live;
    # previously, extra='ignore' (config.py model_config) silently dropped the block.
    # DO NOT flip reify's orchestrator.yaml verify_runners on until the
    # verdict-parity report is green (PRD D6 operator checklist).
    verify_runners: list[VerifyRunnerConfig] = Field(
        default_factory=list,
        description=(
            'List of remote verify runner configs (Lever C).  Each entry describes '
            'one remote host that participates in the multi-host merge-verify pool.  '
            'Default [] means Lever C is off (local-only path, byte-identical to '
            'prior behaviour).  A verify_runners: block in orchestrator.yaml was '
            'silently inert prior to this field being added (extra=\'ignore\').'
        ),
    )
    verify_drift_check_every_n_lands: int = Field(
        default=20,
        ge=1,
        description=(
            'When Lever C is on (verify_runners non-empty), run an async drift check '
            '(local vs remote verdict comparison via DriftDetector) after every Nth '
            'successful land.  Must be >= 1.  Mirrors the '
            'warm_verify_shadow_compare_every_n_merges cadence knob pattern.'
        ),
    )

    # Usage cap handling
    usage_cap: UsageCapConfig = Field(default_factory=UsageCapConfig)

    # Autonomous dry-run unblock hook
    unblock_auto: UnblockAutoConfig = Field(default_factory=UnblockAutoConfig)

    # Environment overrides forwarded to agent invocations
    env_overrides: dict[str, str] = Field(default_factory=dict)

    # Project
    project_root: Path = Field(default=Path('.'))

    # Per-module overrides (populated by load_config via _discover_module_configs).
    # None means "discovery never ran" (default); any dict (including {}) means
    # "discovery ran" — the empty-dict case is treated as a valid discovered-empty
    # result, not as uninitialized.  See run_full_verification for the reuse guard.
    #
    # Consumer normalization contract: any code outside load_config that reads
    # _module_configs for iteration MUST normalize None → {} before calling
    # .values() or iterating, because direct OrchestratorConfig(...) instantiation
    # (e.g. build_eval_orch_config in evals/runner.py) leaves the sentinel at None.
    # Prefer the `module_configs_or_empty` property below over inline `or {}`
    # guards so new consumers cannot silently omit the normalization.
    _module_configs: dict[str, ModuleConfig] | None = PrivateAttr(default=None)

    @field_validator('project_root', mode='after')
    @classmethod
    def _resolve_project_root(cls, v: Path) -> Path:
        return v.resolve()

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

    @property
    def effective_verify_env(self) -> dict[str, str]:
        """Merge sccache.env_overrides() with verify_env; verify_env wins on conflict.

        This is the SINGLE merge rule for the shared sccache backend (κ).
        Both consumers read through here:
        - Local path: load_config folds this back into config.verify_env so that
          run_scoped_verification (verify.py, OUT OF SCOPE) sees the backend.
        - Remote/laptop path: build_merge_verify_spec reads this property directly
          so the spec shipped over the wire carries the shared backend.

        Distinct keys (RUSTC_WRAPPER in verify_env, SCCACHE_REDIS in backend_env)
        simply union.  A shared key uses the verify_env value — an operator who
        hand-sets a single SCCACHE_* var in verify_env beats the structured
        shared-backend default without having to disable the whole knob.
        """
        return {**self.sccache.env_overrides(), **self.verify_env}

    @property
    def enabled_verify_runners(self) -> list[VerifyRunnerConfig]:
        """Return runners with enabled=True (the active Lever C pool).

        Callers use this instead of iterating verify_runners directly so the
        enabled filter cannot be accidentally omitted.
        """
        return [r for r in self.verify_runners if r.enabled]

    @property
    def module_configs_or_empty(self) -> dict[str, ModuleConfig]:
        """Return ``_module_configs``, falling back to ``{}`` when the post-1405
        None sentinel is present.

        Direct ``OrchestratorConfig(...)`` instantiation — e.g.
        ``build_eval_orch_config`` in ``evals/runner.py`` — never calls
        ``load_config``, so ``_module_configs`` stays at its ``None`` default.
        This property encapsulates the normalization so callers cannot forget
        the ``or {}`` guard.
        """
        return self._module_configs or {}

    def for_module(self, module_path: str) -> ModuleConfig | None:
        """Return the ModuleConfig whose registered prefix is the longest (deepest) match
        for *module_path*, or None if no registered prefix matches at all.

        Resolution walks candidate prefixes from the full normalised path inward
        (``foo/bar/baz`` → ``foo/bar`` → ``foo``) and returns the first registered
        match — i.e. the deepest/most-specific config wins.

        This ensures coherence across all three consumers:
        - ``verify.py`` (iterates ``module_configs.values()``)
        - ``scheduler._limit_for`` (passes ``normalize_lock(module, lock_depth)`` —
          exactly a depth-``lock_depth`` path like ``foo/bar``)
        - ``workflow._resolve_module_configs`` (passes normalized module lock paths)

        Precondition / known limitation:
        The scheduler and workflow always pass paths truncated to ``lock_depth``
        components, so a config registered at a prefix *deeper* than ``lock_depth``
        (e.g. ``foo/bar/baz`` with ``lock_depth=2``) is reachable here and by
        ``run_full_verification``, but is **unreachable** through the scheduler /
        workflow path.  ``load_config`` logs a warning when such a mismatch is
        detected.  For full scheduler/workflow integration, keep module configs at a
        prefix depth no greater than ``lock_depth``.

        Backwards-compatible: single-segment prefixes resolve identically to before
        because the deepest candidate that matches is still the first path component.
        """
        if not self._module_configs:
            return None
        parts = module_path.strip('/').split('/')
        for i in range(len(parts), 0, -1):
            candidate = '/'.join(parts[:i])
            result = self._module_configs.get(candidate)
            if result is not None:
                return result
        return None

    @property
    def overrides_db_path(self) -> Path:
        """Path to the scheduler priority-overrides SQLite database.

        Stored separately from runs.db so reify cycles and set_task_status
        writes cannot wipe override rows.
        """
        return self.project_root / 'data' / 'orchestrator' / 'scheduler_overrides.db'

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
    # κ: fold the shared sccache backend into verify_env so the LOCAL merge-verify
    # path (run_scoped_verification in verify.py reads config.verify_env) points at
    # the shared backend without requiring verify.py changes.  The assignment is
    # idempotent w.r.t. effective_verify_env (it re-merges the same keys with
    # verify_env winning), and is a no-op when sccache is disabled.
    config.verify_env = config.effective_verify_env
    config._module_configs = _discover_module_configs(config.project_root)
    # Warn when a discovered config prefix is deeper than lock_depth: its test/lint
    # commands will run in full verification, but scheduler and workflow consumers
    # truncate module paths to lock_depth components via normalize_lock, so the
    # config's scheduling limits (max_per_module, module_overrides) will be silently
    # ignored.  Surface the mismatch so operators can adjust the layout or lock_depth.
    for prefix in config._module_configs:
        prefix_depth = prefix.count('/') + 1  # number of path components
        if prefix_depth > config.lock_depth:
            logger.warning(
                'Module config %r has prefix depth %d but lock_depth=%d; '
                'its scheduling limits are unreachable through the scheduler/workflow path. '
                'Move the orchestrator.yaml up or raise lock_depth.',
                prefix, prefix_depth, config.lock_depth,
            )
    return config
