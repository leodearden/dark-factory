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


class UnblockAutoConfig(BaseModel):
    """Configuration for the autonomous dry-run unblock hook."""

    enabled: bool = Field(default=True)
    budget_usd: float = Field(default=5.0)
    timeout_seconds: float = Field(default=600.0)
    model: str = Field(default='sonnet')
    max_turns: int = Field(default=50)
    effort: str = Field(default='high')
    backend: str = Field(default='claude')


class ReviewConfig(BaseModel):
    """Periodic deep review checkpoint configuration."""

    enabled: bool = Field(default=True)
    interval: int = Field(default=5, description='Trigger checkpoint every N merges')
    full_review_on_complete: bool = Field(default=True)
    briefing_path: str = Field(default='review/briefing.yaml')
    reports_dir: str = Field(default='data/review-checkpoints')


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
    reservation ("park") on every module the starved task wants.  Parks are
    coupled to the owner's live state: they evaporate the moment the owner
    completes, is cancelled, or has its dependencies un-satisfied — no
    wall-clock lease needed.

    Set ``scheduler_v2: true`` to enable the full v2 machinery (eager parks,
    cross-tier preemption, owner-state GC).  When False (default) the skip
    counter still increments and ``task_skipped`` still emits, but
    ``install_parks`` is never called so anti-starvation is inert.
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
    scheduler_v2: bool = Field(
        default=False,
        description=(
            'Enable the v2 anti-starvation machinery: eager parks on the full '
            'module set, cross-tier preemption, and owner-state GC sweep.  '
            'Default False for one burn-in cycle; flip to True after validation.'
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
        dirnames[:] = [d for d in dirnames if d not in _DISCOVERY_EXCLUDED_DIRS]
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
