"""Configuration schema for the orchestrator."""

import fnmatch
import hashlib
import importlib.resources
import logging
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Literal, NamedTuple, Union, get_args, get_origin

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from shared.task_metadata import KNOWN_ROLE_NAMES

from orchestrator.routing import DEFAULT_ALLOWED_MODELS, DEFAULT_LADDER

logger = logging.getLogger(__name__)


# --- Priority-tier constants (value/h scheduler) ---
#
# Canonical 5-tier priority order.  Lower rank = higher priority.  An unset
# (None) priority silently coerces to DEFAULT_TIER — that is a normal,
# expected state (e.g. subtasks in this repo's tasks.json routinely carry no
# priority), not an anomaly.  An unrecognized non-None value (a typo or a
# stale tier string) also coerces to DEFAULT_TIER so it never crashes the
# scheduler, but that path is logged loudly instead, since it means some
# upstream caller passed something the config layer doesn't understand.
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
    """Normalize a priority value (possibly None/unknown) to a canonical tier.

    An unset (None) priority is a NORMAL expected state — not an anomaly —
    so it falls back to DEFAULT_TIER silently; it is a default, not a
    fail-soft fallback, and the repo's no-silent-fail-soft invariant does
    not apply to it. An unrecognized non-None value (a typo or a stale tier
    string) still falls back to DEFAULT_TIER so legacy tasks and typos never
    crash the scheduler, but that fallback IS logged loudly (rather than
    silently): it means some upstream caller passed something the config
    layer doesn't understand, and that should be observable.
    ``stacklevel=2`` attributes the log record to the caller's file/line
    (the actual call site) rather than to this helper.
    """
    if isinstance(value, str) and value in PRIORITY_RANK:
        return value
    if value is None:
        return DEFAULT_TIER
    logger.warning(
        'coerce_tier: unrecognized priority %r, falling back to %r',
        value,
        DEFAULT_TIER,
        stacklevel=2,
    )
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
    module_tagger: str = Field(default='haiku')
    deep_reviewer: str = Field(default='opus')
    judge: str = Field(default='sonnet')
    simple_task: str = Field(default='sonnet')


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
    simple_task: float = Field(default=2.50)


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
    simple_task: int = Field(default=50)


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
    simple_task: str = Field(default='high')


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
    # Dedicated per-role knob (deliberately decoupled from
    # OrchestratorConfig.invocation_timeout — see that field's docstring):
    # today simple_task's timeout comes from the getattr(timeouts_cfg,
    # role_key, self.config.invocation_timeout) fallback, so at stock config
    # (invocation_timeout=7200.0) this literal is byte-equivalent.
    simple_task: float = Field(default=7200.0)
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
    working_idle_secs: float = Field(
        default=1800.0,
        description=(
            "Post-turn-1 no-progress idle bound (seconds). Once the STARTUP "
            "grace window has passed (≥1 assistant turn observed), the "
            "working-regime watchdog no longer kills at a flat per-role "
            "ceiling: it polls the transcript at a coarse cadence and kills "
            "only after no NEW assistant turn has appeared for "
            "max(working_idle_secs, the per-role ceiling) — the per-role "
            "timeout becomes the FLOOR of the idle window (B6 long-tool-call "
            "safety: a single legitimate synchronous tool call must never be "
            "false-killed), not a hard wall on a productive run. Bounded "
            "above by OrchestratorConfig.invocation_timeout, the absolute "
            "cap. Only engages when the transcript is readable; an "
            "unreadable transcript falls back to the old flat "
            "per-role-ceiling kill (B7 conservative degrade)."
        ),
    )


class BackendsConfig(BaseModel):
    """Backend CLI selection per agent role. Values: 'claude', 'codex', 'gemini', 'pi'."""

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
    simple_task: str = Field(default='claude')


class RuleMatch(BaseModel):
    """Closed match-condition vocabulary for a RoutingRule (task epsilon,
    plans/adaptive-model-routing-prd.md).

    Every field is optional; a rule matches iff every condition it sets is
    satisfied (AND) — see ``orchestrator.routing.resolve_route``.
    ``extra='forbid'`` so an unrecognized condition key (e.g. a typo) raises
    a structured ``ValidationError`` naming the key, both at initial config
    load and at hot-reload time (``apply_reload``'s post-apply
    ``model_validate`` re-check) — boundary test 11.

    CAVEAT -- ``task_priority`` matches only ``task_metadata['priority']``,
    NOT the task's top-level ``priority`` field (where priority actually
    lives in this codebase, e.g. ``self.task.get('priority')`` in
    ``workflow.py``). A rule authored against the top-level field will
    silently never match; populate ``metadata['priority']`` explicitly if
    you need this condition to fire. See ``orchestrator.routing.
    _rule_matches``'s docstring for the same note.
    """

    model_config = ConfigDict(extra='forbid')

    role: list[str] | None = None
    task_complexity: str | None = None
    task_priority: str | None = None
    plan_min_steps: int | None = None
    plan_min_modules: int | None = None
    module_prefix: str | None = None
    min_routing_tier: int | None = None
    min_dispatch_count: int | None = None
    simple_saturated: bool | None = None


class RuleSet(BaseModel):
    """Closed override vocabulary a RoutingRule applies upon match (task
    epsilon).

    Every field is optional — a rule may set only a subset of (model,
    effort, budget_usd, max_turns); any field left unset falls through to
    the next-lower resolver layer. ``model`` may be an absolute model string
    or a ladder-relative offset (``'+N'``, resolved against
    ``RoutingConfig.ladder`` and clamped at its top — see
    ``orchestrator.routing.resolve_route``). ``extra='forbid'`` mirrors
    ``RuleMatch``.
    """

    model_config = ConfigDict(extra='forbid')

    model: str | None = None
    effort: str | None = None
    budget_usd: float | None = None
    max_turns: int | None = None


class RoutingRule(BaseModel):
    """One named policy rule: IF ``match`` THEN ``set`` (task epsilon).

    Evaluated in list order by ``orchestrator.routing.resolve_route`` —
    first match wins. ``id`` is carried into the resolved
    ``RoutingDecision.rule_id`` and the ``routing_decision`` telemetry
    event / ``metadata.routing`` mirror.
    """

    id: str
    match: RuleMatch = Field(default_factory=RuleMatch)
    set: RuleSet = Field(default_factory=RuleSet)


class RoutingConfig(BaseModel):
    """Model allowlist + policy-rule table (tasks beta/epsilon,
    plans/adaptive-model-routing-prd.md).

    ``allowed_models`` is the fail-fast admission list enforced by
    ``OrchestratorConfig._validate_models_in_allowlist`` against every
    claude-backend role's configured model string (``models.<role>`` and
    ``unblock_auto.model``). Defaults from ``routing.DEFAULT_ALLOWED_MODELS``
    — the PRD-named "allowlist home" — so this schema and its default stay
    single-sourced.

    ``ladder`` orders models weakest -> strongest for ladder-relative rule
    bumps (``RuleSet.model == '+N'``); defaults from
    ``routing.DEFAULT_LADDER``. ``per_model_daily_ceiling_usd`` is an
    optional per-model trailing-24h USD spend ceiling (task epsilon
    invariant 6) — empty by default, so the ceiling check never trips and
    ``_invoke`` never issues the extra cost_store read at stock config.
    ``rules`` is the ordered policy-rule table (task epsilon); empty by
    default — the shipped default rule lives in defaults.yaml, not here, so
    a later task can retune it without a code change.
    """

    allowed_models: list[str] = Field(default_factory=lambda: list(DEFAULT_ALLOWED_MODELS))
    ladder: list[str] = Field(default_factory=lambda: list(DEFAULT_LADDER))
    per_model_daily_ceiling_usd: dict[str, float] = Field(default_factory=dict)
    rules: list[RoutingRule] = Field(default_factory=list)


class UnblockAutoConfig(BaseModel):
    """Configuration for the autonomous dry-run unblock hook."""

    enabled: bool = Field(default=True)
    budget_usd: float = Field(default=5.0)
    timeout_seconds: float = Field(default=1200.0)
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


class StarvationWatchdogConfig(BaseModel):
    """Scheduler starvation watchdog configuration.

    When a ``pending`` task is dispatch-eligible (all local + external deps
    satisfied, no live claimant) yet keeps being skipped as the TOP-scored
    candidate past both ``skip_threshold`` and ``idle_secs``, the scheduler
    files exactly ONE INFO-level escalation so an AFK operator is notified.
    The escalation auto-resolves when the task finally dispatches.

    The escalation fires when EITHER gate is crossed (OR, not AND):
    - the dual gate — ``skip_threshold`` AND ``idle_secs`` both crossed; or
    - the idle-only backstop — ``idle_only_secs`` crossed regardless of
      skip_count (catches never-top-scored tasks that accrue zero skips and so
      can never cross the dual gate — reify-5166 RCA, task 2755).

    Dual-gate components:
    - ``skip_threshold`` — minimum consecutive top-skips (reuses ``_skip_count``).
    - ``idle_secs`` — minimum wall-clock seconds of continuous dispatch-eligibility
      (anchored on the first tick the task appears as a candidate; resets if it
      leaves the candidate pool for any tick).

    Default 50 skips is well above the fairness per-tier park thresholds (0–9,
    polish 9999) so the watchdog only fires on genuine multi-day starvation (e.g.
    the live reify-3465 case: 475× skipped).  Default 259200 s (72h) provides the
    independent wall-clock backstop required by PROPERTY 3.

    Owner decision (Leo 2026-07-15), backed by archive analysis of all 209
    STARVATION_WATCHDOG escalations filed since 06-17: 94% closed benign/no-action,
    zero produced a direct intervention, and starving duration at fire was median
    0.5h / p90 0.8h / max 5.8h — every one of the 209 would have been silenced by a
    72h threshold, while sibling-project reify tasks legitimately wait ~48h on
    locks.  The watchdog remains the multi-day true-wedge tripwire; it is not being
    removed, only re-tuned so it no longer fires on ordinary lock contention.

    Set ``enabled: false`` in orchestrator.yaml to silence the watchdog entirely.
    """

    enabled: bool = Field(
        default=True,
        description='Set to false to disable starvation escalation filing entirely.',
    )
    skip_threshold: int = Field(
        default=50,
        ge=1,
        description=(
            'Minimum consecutive top-candidate skips before filing an INFO escalation. '
            'Must be >= 1.  Well above the fairness park thresholds (0–9 / polish 9999) '
            'so it only fires on genuine multi-day starvation.  Secondary signal only: '
            'crossing this threshold alone never files an escalation — idle_secs must '
            'also be crossed.'
        ),
    )
    idle_secs: float = Field(
        default=259200.0,
        gt=0,
        description=(
            'Minimum wall-clock seconds of continuous dispatch-eligibility before '
            'filing.  Anchored on the first tick the task enters the candidate pool; '
            'reset if the task leaves the pool for any tick.  Must be > 0.  Default '
            '259200s (72h): archive analysis of 209 prior firings showed max starving '
            'duration was 5.8h and sibling-project reify tasks legitimately wait ~48h '
            'on locks, so 72h isolates genuine multi-day wedges without false-positive '
            'noise.'
        ),
    )
    idle_only_secs: float = Field(
        default=259200.0,
        gt=0,
        description=(
            'Never-top-scored starvation backstop.  A task continuously '
            'dispatch-eligible for this many wall-clock seconds files the '
            'escalation on idle ALONE — the skip-gate is waived (OR, not AND). '
            'This catches the structural blind spot where a low-priority task can '
            'never outscore a medium/high candidate, is never the top-scored '
            'candidate, accrues ZERO skips, and so can never cross the '
            'skip_threshold+idle_secs dual gate (reify-5166 RCA, task 2755).  Must '
            'be > 0 and >= idle_secs (enforced by a model_validator).  Default '
            '259200s (72h) == idle_secs default, so it never fires on ordinary '
            'contention (all 209 prior firings starved < 6h); raise it above '
            'idle_secs to restore a two-tier fast-skip / slow-idle scheme.'
        ),
    )

    @model_validator(mode='after')
    def _reject_idle_only_below_idle(self) -> 'StarvationWatchdogConfig':
        if self.idle_only_secs < self.idle_secs:
            raise ValueError(
                f'StarvationWatchdogConfig.idle_only_secs ({self.idle_only_secs}) '
                f'must be >= idle_secs ({self.idle_secs}); an idle-only backstop '
                'narrower than the dual-gate idle component would silently pre-empt '
                'the skip path for all tasks.  Set idle_only_secs >= idle_secs.'
            )
        return self


class WarmBaseHardDownConfig(BaseModel):
    """Scheduler warm-lane base hard-down watchdog configuration (task 2061).

    The warm-lane pool CoW-seeds every lane's ``target/`` from a single
    HOST-SCOPED rolling base (``warm_lane_base_target_path``).  When that base
    is absent (e.g. mid-rebuild, or the reify reseed ladder is between
    attempts), it is a host-wide infra condition — not a per-task fault — so
    it must produce exactly ONE signal, not one BLOCKED+L1 per dispatched
    task.

    The scheduler probes base health once per ``acquire_next`` tick via an
    injected callback (``_warm_base_health_probe``, installed by the Harness
    against ``GitOps._warm_lane_base_resolvable``).  A definite ``ABSENT``
    reading engages a singleton latch that HALTS dispatch host-wide
    (fail-open — pending tasks stay pending, in-flight warm-lane acquires
    requeue via ``WarmLanePoolHardDown``) and files exactly one INFO notice.
    The latch auto-clears the moment the probe reports ``OK`` again — the
    natural clear happens when ``GitOps.refresh_warm_base`` (git_ops.py:1659)
    successfully rebuilds/advances the base.  ``INDETERMINATE`` readings
    (transient stat/readlink hiccups) never engage, clear, or promote the
    latch — fail-safe hold.

    If the base is still ``ABSENT`` after ``l2_window_secs`` of continuous
    latch time, the reify reseed ladder is presumed stuck and exactly ONE
    born-at-L2 escalation is promoted so a human is notified.

    Set ``enabled: false`` in orchestrator.yaml to silence the watchdog
    entirely (the git_ops pre-acquire gate and ``WarmLanePoolHardDown``
    requeue path remain in effect regardless — this knob only controls the
    scheduler-side halt + escalation latch).
    """

    enabled: bool = Field(
        default=True,
        description='Set to false to disable the warm-base hard-down watchdog entirely.',
    )
    l2_window_secs: float = Field(
        default=300.0,
        gt=0,
        description=(
            'Bounded remediation window, in seconds: how long the warm-lane '
            'base may remain continuously ABSENT before the single host-scoped '
            'L2 escalation is promoted.  Must be > 0.  A healthy reify reseed '
            'ladder clears the latch (via refresh_warm_base) well within this '
            'window; still-absent past it is the definition of a stuck ladder.'
        ),
    )


class PsiAdmissionConfig(BaseModel):
    """L3b dispatch-admission load-cap gate configuration (task 2327, PRD
    docs/prds/dispatch-admission-load-cap.md, task DA2).

    The dispatch-admission gate (DA3) reads these thresholds live each tick
    to decide whether the host is saturated enough to hold new dispatch.
    Memory ranks above io (DA-D1): ``mem_some_avg10`` is deliberately
    TIGHTER than ``io_some_avg10``, and ``mem_full_avg10`` is a separate
    memory `full` hard-trip. ``min_inflight_floor`` (DA-D3) is an
    anti-deadlock floor — the gate must never hold when fewer than this many
    tasks are in flight on this orchestrator, so a value < 1 (which could
    wedge the queue with nothing running) is rejected at load.

    All fields are green-tier hot-tunable via RELOADABLE_FIELDS — an
    operator may adjust thresholds post-observation via
    ``mcp__escalation__reload_config`` without a process restart.

    Set ``enabled: false`` in orchestrator.yaml to disable the gate entirely.
    """

    enabled: bool = Field(
        default=True,
        description='Set to false to disable the dispatch-admission gate entirely.',
    )
    cpu_some_avg10: float = Field(
        default=85.0,
        description='PSI cpu "some" avg10 (%) threshold — primary CPU saturation signal.',
    )
    mem_some_avg10: float = Field(
        default=15.0,
        description=(
            'PSI memory "some" avg10 (%) threshold. Deliberately tighter than '
            'io_some_avg10 — memory ranks above io per DA-D1.'
        ),
    )
    mem_full_avg10: float = Field(
        default=3.0,
        description='PSI memory "full" avg10 (%) threshold — memory hard-trip.',
    )
    io_some_avg10: float = Field(
        default=40.0,
        description='PSI io "some" avg10 (%) threshold — looser than mem_some_avg10.',
    )
    min_inflight_floor: int = Field(
        default=1,
        ge=1,
        description=(
            'DA-D3 anti-deadlock floor: the gate never holds when fewer than '
            'this many tasks are in flight on this orchestrator. Must be >= 1.'
        ),
    )


class DeliveredChecksConfig(BaseModel):
    """Delivered-check dep-gate sweep configuration (capability-delivered-
    checks PRD, plans/capability-delivered-checks-prd.md).

    The scheduler's per-tick sweep (``Scheduler._compute_delivered_check_cache``)
    evaluates every distinct terminal local dep that carries
    ``metadata.delivered_checks`` against the committed ``main`` tree.
    ``max_checks_per_tick`` bounds how many uncached (dep, main_sha) checks
    that sweep evaluates in a single tick — a worst-case fan-out guard so a
    burst of newly-terminal deps can't stall tick latency; checks deferred by
    the budget stay uncached and are retried (fail-safe wait) next tick.

    Task 2580 (delta) owns only ``max_checks_per_tick``. Task 2583 (epsilon)
    extends this sub-model with the grace-streak escalation knobs below:
    ``enabled`` (kill switch), ``grace_cycles`` (consecutive-FAILED-tick
    threshold before a born-at-L2 escalation fires), and
    ``check_timeout_secs`` (per-check wall-clock bound; a timeout maps to
    ``DeliveredCheckResult.ERRORED``, the same fail-safe outcome as a
    runner exception).
    """

    enabled: bool = Field(
        default=True,
        description=(
            'Set to false to disable the delivered-check dep-gate entirely — '
            '_phase_delivered_check_gate short-circuits to a None cache (not '
            'an empty dict), so _deps_satisfied takes its legacy arm-off path '
            'and no sweep, streak, or escalation logic runs at all.'
        ),
    )
    max_checks_per_tick: int = Field(
        default=50,
        ge=1,
        description=(
            'Maximum number of uncached (dep_task_id, main_sha) delivered-checks '
            'evaluated per scheduler tick. Must be >= 1. Checks deferred by this '
            'budget remain uncached and are retried next tick (fail-safe wait).'
        ),
    )
    grace_cycles: int = Field(
        default=3,
        ge=1,
        description=(
            'Consecutive ran-and-FAILED sweep ticks (per dependent, dep pair) '
            'before a born-at-L2 dependency_capability escalation fires and the '
            'dependent is blocked. Must be >= 1. The grace window absorbs the '
            'merge-finalize -> scheduler-tick done->main propagation lag.'
        ),
    )
    check_timeout_secs: float = Field(
        default=120.0,
        gt=0,
        description=(
            'Per-check wall-clock timeout (seconds) for each run_delivered_check '
            'call in the sweep. Must be > 0. A check that exceeds this is treated '
            'as DeliveredCheckResult.ERRORED (fail-safe — no streak bump, dep left '
            'uncached, retried next tick). Primary bound for the timeout-less grep '
            'kind; defense-in-depth for scripts, which also carry their own '
            'descriptor timeout_secs.'
        ),
    )


class SessionResumeConfig(BaseModel):
    """Warm-lane session-resume guard configuration (task γ,
    plans/warm-lane-session-resume-prd.md).

    Guards the β resume injection in ``_run_slot``: a recovered agent
    session is injected as ``--resume`` only when it is fresh, under its
    per-task resume cap, and its transcript is corroborated on disk
    (INV-3). Any ineligible session degrades to today's fresh dispatch
    (I3 — never a stall, never a scheduler-visible error), emitting a
    reason-carrying ``session_resume_fallback``/``session_resume_capped``
    event; a per-boot run of consecutive fallbacks above
    ``fallback_storm_threshold`` files one L1 escalation (INV-4).

    ``enabled=false`` is the kill switch: no ``--resume`` is ever injected
    (B6), and no ``session_resume_*`` event or streak is produced.

    Mirrors DeliveredChecksConfig's shape (a kill switch plus ge-bounded
    int knobs); all four leaves are green-tier hot-reloadable via the
    ``session_resume`` whole-submodel group in RELOADABLE_FIELDS.
    """

    enabled: bool = Field(
        default=True,
        description=(
            'Set to false to disable warm-lane session resume entirely — no '
            '--resume is ever injected (B6), and the _run_slot guard emits no '
            'session_resume_* event and feeds no fallback-storm streak.'
        ),
    )
    freshness_window_secs: int = Field(
        default=86400,
        ge=1,
        description=(
            'A recovered sidecar is eligible only if (now - started_at) is '
            'below this many seconds; a staler sidecar degrades to fresh '
            'dispatch with a session_resume_fallback(reason=stale) event. '
            'Must be >= 1. Default 86400 (1 day) sits at/above the invocation '
            'absolute cap plus slack, so a sidecar is rejected only once it '
            'clearly outlives any legitimate in-flight invocation.'
        ),
    )
    max_resumes_per_task: int = Field(
        default=3,
        ge=1,
        description=(
            'A task whose sidecar resume_count has reached this cap degrades '
            'to fresh dispatch with a session_resume_capped event (by-design '
            'throttling — does NOT feed the fallback-storm streak). Must be '
            '>= 1. Default 3 matches the 8h restart cadence against a '
            'multi-day (~30h) task legitimately resuming ~3x.'
        ),
    )
    fallback_storm_threshold: int = Field(
        default=5,
        ge=1,
        description=(
            'Consecutive per-boot session_resume_fallback degradations '
            '(reset to 0 on any eligible resume) before one L1 escalation is '
            'filed (INV-4 storm escape — suspected systematic clock skew / '
            'wiped transcripts / mass reseed). Must be >= 1. Default 5 is '
            'above both the resume cap and ordinary collision noise, so only '
            'systematic corroboration breakage trips it.'
        ),
    )


class SpeculationProbeConfig(BaseModel):
    """Variable-depth speculative verify placement (task 2359, sibling of
    task 2340's depth telemetry).

    Lets the EXISTING second verify slot occasionally target a DEEPER
    already-built speculative stack (cumulative depth d, d in
    ``probe_depths``) instead of the adjacent depth-1 stack. A passing
    depth-d probe produces a genuine depth>=2 ``merge_verify`` record
    (labelled via task 2340's ``depth`` field), so
    ``scripts/analyze_speculation_depth.py`` can print a multi-point
    P(pass|depth) curve.

    ``probe_fraction=0.0`` (the default) disables the mechanism entirely:
    :func:`orchestrator.merge_queue.select_probe_depth` always returns
    ``None`` at fraction<=0, so dispatch falls through to the unchanged
    ``_verify_frontier_depth()`` path -- byte-identical to pre-task-2359
    behaviour.

    ``suppress_flake_rate`` is a per-verify rolling FAIL-rate threshold (see
    ``SpeculativeMergeWorker._recent_verify_fail_rate``): probing is
    suppressed whenever that rate is at or above this value, so a thrashing
    pipeline is never handed MORE speculative load.

    A depth-d stack only exists to probe when the operator has separately
    raised ``speculation_depth`` (the existing merge-ahead knob) so the
    merger builds >= d items ahead; under the default K=2, the built stack
    stays shallow and every probe safely no-op-falls-back to the adjacent
    depth-1 path (see ``SpeculativeMergeWorker._available_built_depth``).

    All fields are green-tier hot-tunable via RELOADABLE_FIELDS — an
    operator may adjust the probe rate/depths/suppression threshold via
    ``mcp__escalation__reload_config`` without a process restart.
    """

    probe_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            'Fraction of second-slot dispatch rounds that probe a deeper '
            'already-built speculative stack instead of the adjacent '
            'depth-1 stack. 0.0 (default) disables probing entirely -- '
            'byte-identical to pre-task-2359 behaviour.'
        ),
    )
    probe_depths: list[int] = Field(
        default_factory=lambda: [2, 3, 5, 8],
        description=(
            'Candidate cumulative stack depths cycled through on probe '
            'rounds, in order. Must be non-empty; every entry must be a '
            'positive integer.'
        ),
    )
    suppress_flake_rate: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description=(
            'Per-verify rolling FAIL-rate threshold (see '
            'SpeculativeMergeWorker._recent_verify_fail_rate) at or above '
            'which probing is suppressed for the round, so a thrashing '
            'pipeline is never given more speculative load.'
        ),
    )

    @field_validator('probe_depths', mode='after')
    @classmethod
    def _validate_probe_depths(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError('probe_depths must not be empty')
        if any(d <= 0 for d in v):
            raise ValueError(
                f'probe_depths entries must all be positive integers; got {v!r}'
            )
        return v


class MergeDeepConfig(BaseModel):
    """Deep merge-ahead chains (task 3183, plans/deep-merge-ahead-prd.md α).

    Lets a single verify cover a CHAIN of k queued merge items (one scratch
    worktree, sequential in-order merges, one verify on the tip) instead of one
    item at a time, so a passing tip lands the whole clean prefix in one round.

    ``chain_cap`` is the single gate for the whole feature. The dispatch contract
    it feeds (when a chain is built, and how ``target_depth`` is derived) and the
    cap-staging plan live in the PRD, which is the ONE canonical narrative for
    both — deliberately not restated here, because β (task 3184, the chain
    builder) and γ (task 3185, the dispatch gate) are the consumers that
    implement that contract and may change it. Nothing in the orchestrator reads
    the knob yet.

    ``chain_cap=0`` (the shipped default) is the KILL SWITCH: the gate can never
    open, so no chain code runs on any dispatch path and behaviour is
    byte-identical to pre-PRD merging — the ``probe_fraction=0.0`` precedent in
    :class:`SpeculationProbeConfig`.

    Plain BaseModel (no ``frozen``, no ``validate_assignment``) so ``_set_leaf``
    can mutate it in place on hot-reload and held references observe the update
    (invariant I3) — see :class:`RetentionConfig`'s docstring, below, for the
    same requirement. Every leaf is green-tier hot-reloadable via
    RELOADABLE_FIELDS (PRD decision #7), so an operator can enable, retune, or
    KILL the feature (cap -> 0) via ``mcp__escalation__reload_config`` without a
    process restart.
    """

    chain_cap: int = Field(
        default=0,
        ge=0,
        description=(
            'Maximum number of queued items a single deep merge-ahead chain may '
            'contain. 0 (the default) disables the feature entirely -- the kill '
            'switch: no chain is ever built, so merge behaviour is byte-identical '
            'to pre-task-3183 behaviour. Must be >= 0; a negative cap is rejected '
            'at load rather than reaching dispatch. No upper bound is imposed. '
            'See plans/deep-merge-ahead-prd.md for the dispatch contract this '
            'gates and for the cap-staging plan.'
        ),
    )


class RetentionConfig(BaseModel):
    """Retention bounds for the archived-transcript tree (task 2742, PRD α).

    α owns the whole transcript_archive block including these knobs even
    though the GC consumer (δ/task 2731) is what enforces them; the producer
    hook here writes archives but never prunes. Plain BaseModel (no
    frozen/validate_assignment) so _set_leaf can mutate it in place on reload.
    """

    max_age_days: int = Field(
        default=90,
        description=(
            'Maximum age (days) an archived per-task transcript tree is kept '
            'before the GC sweep (δ/task 2731) may prune it.'
        ),
    )
    max_task_dirs: int = Field(
        default=5000,
        description=(
            'Soft cap on the number of per-task archive dirs kept; the GC '
            'sweep (δ/task 2731) prunes oldest-first beyond this.'
        ),
    )


class TranscriptArchiveConfig(BaseModel):
    """Agent-transcript archival (task 2742, plans/agent-transcript-archival-prd.md α).

    The producer hook in TaskWorkflow._invoke's finally gzips each finished
    agent session's transcripts (see shared.transcript_archive) to
    ``<project_root>/<root>`` — a durable location OUTSIDE the per-task
    worktree so the archive survives worktree teardown.

    All fields are green-tier hot-reloadable via RELOADABLE_FIELDS (the
    whole-submodel group for 'transcript_archive'); ``retention`` is compared
    as one atomic BaseModel leaf, so any retention.* edit reloads as a
    whole-retention swap. Plain BaseModels (no frozen/validate_assignment) so
    _set_leaf can mutate them in place on reload.
    """

    enabled: bool = Field(
        default=True,
        description=(
            'Set to false to disable transcript archival entirely — the '
            '_invoke producer hook short-circuits and never calls '
            'archive_task_transcripts.'
        ),
    )
    root: str = Field(
        default='data/orchestrator/agent-transcripts',
        description=(
            'Archive root, resolved against project_root (NOT the worktree): '
            'project_root / root. The durable, git-ignored orchestrator data '
            'dir, reusing the project_root / data / ... idiom of the '
            'verify-logs archive.'
        ),
    )
    retention: RetentionConfig = Field(default_factory=RetentionConfig)


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


# `.task-meta` path-derivation contract (worktree-lane-lifecycle PRD, W11-β):
# the orchestrator's task-scratch base lives at
# <worktree_base>/.task-meta/<worktree_name> — a SIBLING of the worktree
# itself, not nested inside it (unlike the legacy <worktree>/.task).
# worktree_base already derives from GitConfig.worktree_dir below
# (git_ops.py: project_root / config.worktree_dir); this constant supplies
# only the new sibling-dir name, deliberately NOT a pydantic Field (Open Q2:
# no new knob). TaskArtifacts.meta_root_for() is the single place that joins
# these two together — see orchestrator/artifacts.py.
TASK_META_DIRNAME: str = '.task-meta'


class LaneCommand(BaseModel):
    """A single per-project offline-lane generic command entry (task 2789).

    Drives one generic offline-lane sub-run: the runner launches ``command``
    (a shell string, via ``sh -c``) at idle nice/ionice in ``<worktree>/<cwd>``
    with ``DF_VERIFY_ROLE=offline``, off the merge hot path, always from the
    current ``main`` head. A red result routes through the existing
    ``OfflineLaneWorker`` red path (confirm → fingerprint → dedup'd fix task →
    staged L2), filing the fix task at ``fix_task_priority`` — no new mechanism
    (INV-5). This generalizes the previously reify-hard-coded run seams to
    per-project config (PRD plans/integration-test-lane-prd.md, task alpha).
    """

    name: str = Field(
        description=(
            'Short, stable identifier for this sub-run, used in the '
            "``offline-lane: <name> sub-run ...`` log line. Required."
        ),
    )
    command: str = Field(
        description=(
            'Shell command string launched via ``sh -c`` for this sub-run '
            '(e.g. ``pytest -m integration``). Required. NOTE: the default '
            'confirm/dedup path is pytest-oriented — it serializes via '
            '``_serial_pytest_str`` and extracts still-failing pytest '
            'node-ids. A non-pytest command that reproduces red (non-zero '
            'exit, no parseable node-ids) is filed under a stable '
            '``<name>::nonzero-exit`` sentinel rather than being swallowed as '
            'a flake; inject a custom ``command_confirmation_runner`` for '
            'richer per-failure dedup.'
        ),
    )
    cwd: str = Field(
        default='.',
        description=(
            "Working directory for the command, relative to the reset "
            "``_offline-deep`` worktree root. Defaults to '.' (the worktree "
            'root == project_root inside the worktree); a static string, since '
            'a pydantic default cannot reference the runtime project_root.'
        ),
    )
    fix_task_priority: Literal['low', 'medium', 'high'] = Field(
        default='medium',
        description=(
            'Priority for the fix task auto-filed when this sub-run is '
            "confirmed red. Defaults to 'medium' (the generic per-project "
            "default; the legacy reify numeric/infra seams stay 'high')."
        ),
    )
    enabled: bool = Field(
        default=True,
        description=(
            'When False this command is skipped by the offline-lane runner '
            '(a config-only off switch that leaves the entry in place).'
        ),
    )


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
    warm_lane_seed_scrub_globs: list[str] = Field(
        default_factory=list,
        description=(
            'Glob patterns (relative to the seeded lane\'s build-artifact '
            'dir, i.e. reap_build_artifact_dirs[0] / "target" by default), '
            'matched subtrees of which are DELETED after a successful '
            'warm-lane CoW seed (task 2315, BUG 3 fix). Fixes a stale-path '
            'leak: some generated build artifacts (e.g. tauri permission '
            'autogen files) embed an ABSOLUTE OUT_DIR path pointing back at '
            'the shared _merge-verify/target warm base they were originally '
            'generated under; `cp -a --reflink`-seeding that base into a '
            'lane copies those baked-in paths verbatim into every lane. '
            'Deleting the configured subtrees lets them regenerate fresh, '
            'per-lane, on first use, instead of rewriting the baked paths '
            'in place (fragile across TOML/binary formats). Defaults to '
            '[] (opt-in, byte-identical no-op) — reify sets e.g. '
            "['**/permissions/*/autogenerated']. Uses default_factory to "
            'avoid a shared mutable default across model instances.'
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
    main_gate_bypass_command: str | None = Field(
        default=None,
        description=(
            'Optional project-configurable shell command run (via ``sh -c`` '
            'in project_root) immediately BEFORE recover_red_main\'s CAS '
            'update-ref to engage a DURABLE bypass of a project\'s always-on '
            'non-fast-forward main-gate guard.  recover_red_main moves '
            'refs/heads/main BACKWARD to an earlier good SHA, which a '
            'reify-style non-ff guard rejects UNCONDITIONALLY (before the '
            'sanction/sentinel check is even reached), so main_gate_mark_command '
            'alone is insufficient for the recovery move.  When set this bypass '
            'SUPERSEDES the mark: recover_red_main engages the bypass and SKIPS '
            'the mark entirely (running both would leave the one-shot mark '
            'sentinel unconsumed — the hook `continue`s on the bypass before '
            'consuming it — falsely sanctioning the next unrelated ref move).  '
            'The bypass is DURABLE, so recover_red_main clears it on every path '
            'via main_gate_bypass_clear_command.  Set under the ``git:`` '
            'section of orchestrator.yaml.  Generic, not reify-hardcoded '
            '(honors task 1715\'s KEEP IT GENERIC).  None (default) => feature '
            'off (no-op) so other projects are unaffected; advance_main '
            '(forward ff moves) never uses it.'
        ),
    )
    main_gate_bypass_clear_command: str | None = Field(
        default=None,
        description=(
            'Optional project-configurable shell command run (via ``sh -c`` '
            'in project_root) to CLEAR the durable bypass engaged by '
            'main_gate_bypass_command.  recover_red_main runs it in a '
            'try/finally around the CAS update-ref so it fires on EVERY path '
            '(success, CAS failure, AND exception) — unlike '
            'main_gate_unmark_command, which only runs on the failure path — '
            'because a bypassed transaction consumes nothing, so the '
            'durable bypass would otherwise leak into every later ref move and '
            'disable the project\'s non-ff guard.  Set under the ``git:`` '
            'section of orchestrator.yaml alongside main_gate_bypass_command.  '
            'None (default) => feature off (no-op).'
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
    persistent_offline_deep_worktree: bool = Field(
        default=False,
        description=(
            'When True, the offline-deep lane worker (β2) may reuse a '
            'SECOND FIXED worktree at <worktree_dir>/_offline-deep (task '
            '1952, PRD δ / §5 C5) instead of a fresh ephemeral worktree. '
            'Modeled on persistent_merge_worktree: reset-in-place to the '
            'target commit via git reset --hard, scrubbed of untracked '
            'files except build-artifact dirs, retaining its own target/ '
            'warmth across attempts.  That target/ is dedicated and is '
            'never shared with or seeded from the merge-lane target/ at '
            '_merge-verify (C5).  Default False → byte-identical to prior '
            'behaviour (trivially revertible); activation gate owned by '
            'the β2 consumer.'
        ),
    )
    offline_lane_enabled: bool = Field(
        default=False,
        description=(
            'When True (AND persistent_offline_deep_worktree is also True), '
            'Harness launches the singleton offline-deep lane worker (task '
            '1953, β2): a background loop that runs the heavy reify test '
            'suite off the verify hot path, always from the current main '
            'head, on every post-merge advance.  Default False → byte-'
            'identical to prior behaviour (trivially revertible); both '
            'knobs must be on since the worker cannot run without its '
            'dedicated _offline-deep worktree.'
        ),
    )
    offline_lane_infra_enabled: bool = Field(
        default=False,
        description=(
            "When True (and offline_lane_enabled is also True) the "
            "offline-deep worker ALSO invokes reify's `run_all --scope "
            "host-infra` (H9 = reify:4929) as a second sub-run at the same "
            'run-start snapshot head (task 1959, IE1); default False = '
            'numeric-only, byte-identical to prior behavior; gated '
            'separately so infra can be enabled once reify:4929 is '
            'confirmed on main.'
        ),
    )
    offline_lane_test_threads: int = Field(
        default=6,
        ge=1,
        description=(
            'Value passed as --test-threads=N to scripts/run-offline-deep.sh '
            'by the offline-deep lane worker (β2, PRD §11.2).  A small fixed '
            'N in the suggested 4-8 band; not frozen, retunable via '
            'orchestrator.yaml without a code change.'
        ),
    )
    offline_lane_poll_interval_secs: float = Field(
        default=120.0,
        gt=0,
        description=(
            'Poll-backstop cadence (seconds) for the offline-deep lane '
            "worker (β2): when the worker's wake event times out, it "
            'compares a fresh git_ops.get_main_sha() against the head of '
            'its last completed run and treats a mismatch as run-worthy. '
            'Catches a missed on_post_merge trigger (e.g. a crash between '
            'the merge and the notifiee call); correctness lives in the '
            'run-start head snapshot, not the trigger, so a missed trigger '
            'only costs granularity.'
        ),
    )
    offline_lane_red_advances_before_blocker: int = Field(
        default=3,
        ge=1,
        description=(
            'N = number of consecutive confirmed-red advances (same failing-'
            'test-set fingerprint) the offline-deep lane worker (β3, PRD '
            '§11.3 / C4) tolerates before promoting its filed fix task to a '
            'born-at-L2 escalate_blocker. Not frozen — retunable via '
            'orchestrator.yaml without a code change.'
        ),
    )
    offline_lane_commands: list[LaneCommand] = Field(
        default_factory=list,
        description=(
            'Per-project generic offline-lane commands (task 2789). Each '
            'entry drives one additional offline-lane sub-run — launched at '
            'idle nice/ionice via ``sh -c`` in ``<worktree>/<cwd>`` with '
            '``DF_VERIFY_ROLE=offline``, off the merge hot path, always from '
            'head — that reuses the existing red path (confirm → dedup fix '
            'task → staged L2). Generic commands run IN ADDITION to whichever '
            'legacy seams (numeric / infra) are enabled. Defaults to [] '
            '(opt-in, byte-identical no-op). Uses default_factory to avoid a '
            'shared mutable default across model instances.'
        ),
    )
    offline_lane_legacy_numeric_enabled: bool = Field(
        default=True,
        description=(
            'D2 gate: when True the legacy unconditional numeric '
            'run-offline-deep.sh sub-run fires; projects without that script '
            'set False; default True keeps reify byte-identical.'
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
    warm_lane_pool: bool = Field(
        default=False,
        description=(
            'When True, task-dispatch worktree provisioning allocates from a '
            'per-host pool of pre-seeded warm lanes (_lane-0 .. _lane-{N-1}) '
            'instead of creating a cold ephemeral git worktree for each task. '
            'Pool size N = max_concurrent_tasks (passed to GitOps at startup). '
            'Bounded-pool policy (β): acquire_warm_lane returns a discriminated '
            'outcome — EXHAUSTED → WarmLanePoolExhausted (requeue/backpressure, '
            'scheduler caps at pool size and waits, never cold-creates); '
            'FAULT (seed/worktree-add failure or absent seed script) → '
            'RuntimeError → blocked + L1; DISK_PRESSURE (seed exit-75) → '
            'WarmLaneDiskPressure → transient-infra requeue. '
            'The cold-path fall-through is removed while the knob is enabled. '
            'Default False → byte-identical to today (trivially revertible, '
            'mirrors persistent_merge_worktree knob). '
            'OPERATIONAL NOTE — absent seed script (FAULT, rc=127): when '
            'seed-warm-lane.sh is absent from the lane\'s scripts/ dir, every '
            'dispatched task raises RuntimeError → blocked+L1 escalation with '
            'no self-healing fallback.  A single misconfigured host produces '
            'one escalation per task dispatched.  To recover: deploy a valid '
            'seed-warm-lane.sh to the lane\'s scripts/ dir (or disable this '
            'knob temporarily).  Look for "seed script absent (rc=127)" in '
            'acquire_warm_lane WARNING logs to detect this condition early. '
            'OPERATIONAL NOTE — disk-pressure requeues (DISK_PRESSURE, rc=75): '
            'these count against the per-task requeue_cap equally with '
            'backpressure requeues; a persistent disk-full condition will burn '
            'the retry budget with no backoff.  The block_reason '
            '"warm_lane_disk_pressure (transient infra)" is set for future '
            'scheduler special-casing (follow-up: exclude from retry cap). '
            'DISK_PRESSURE is produced in two ways: (1) ε pre-acquire disk-guard '
            '(warm_lane_disk_guard=True) — check γ script → reclaim δ script → '
            'recheck, return DISK_PRESSURE if still pressured (runs before '
            'acquire_for so idle lanes stay FREE; fail-open on absent scripts); '
            '(2) seed exit-75 (EX_TEMPFAIL) after the lane is allocated.  Both '
            'routes thread through the same WarmLaneDiskPressure → REQUEUED path. '
            'Note: the reify-repo PRD inv.6 text is the cross-repo counterpart '
            'of this policy; it is flagged for update separately via escalate_info.'
        ),
    )
    spare_warm_lanes: int = Field(
        default=0,
        ge=0,
        description=(
            'Extra warm task-lanes allocated ABOVE the derived pool size '
            '(max_concurrent_tasks); effective pool N = max_concurrent_tasks + '
            'spare_warm_lanes when warm_lane_pool is on (read once at startup by '
            'Harness, passed to GitOps). Gives acquire headroom so a transient '
            'lane-leak / stuck-ASSIGNED lane does not exhaust the pool and force '
            'the inv.6 cold-fallback under light load. Default 0 → byte-identical '
            'to prior behaviour (max + 0 == max); no effect when warm_lane_pool is '
            'off and does not size the merge-spec pool.'
        ),
    )
    warm_lane_prewarm: bool = Field(
        default=False,
        description=(
            'When True, Harness.run() EAGERLY materializes the full effective_N '
            'warm-lane pool on disk at startup (GitOps.prewarm_pool), instead of '
            'letting each _lane-k be created lazily on its first acquire.  Root '
            'cause it addresses: the pool is effective_N = max_concurrent_tasks + '
            'spare_warm_lanes FREE lanes in memory, but the on-disk git worktree '
            'add for each lane happens only in the acquire create-once branch, '
            'lowest-numbered-first — so on a host peaking below effective_N the '
            'spare lanes are never demanded, never created, and the intended '
            'headroom is a phantom (present in the in-memory state machine, '
            'absent on disk).  prewarm closes that gap: for each lane not already '
            'registered it does git worktree add --detach <lane> <main HEAD> then '
            '_seed_warm_lane --fresh-checkout, leaving a lane byte-identical to a '
            'released idle lane (detached HEAD, no task branch, seeded target/) '
            'that the EXISTING reset-in-place acquire path adopts unchanged.  '
            'Consulted ONCE at startup, AFTER all reconcile sweeps and BEFORE the '
            'first dispatch, so it never races a live acquire/release and never '
            'double-creates a lane the sweeps already restored.  Fail-open: skips '
            'entirely when the CoW seed base is ABSENT (mirrors acquire\'s gate), '
            'tolerates per-lane failures (logged, torn down, counted, loop '
            'continues), never raises, and is idempotent across restarts.  Emits '
            'a shortfall WARNING (the VISIBLE SIGNAL) whenever it cannot reach '
            'effective_N so a disk/floor shortfall can never silently cap the '
            'pool.  Gated on warm_lane_pool (no effect when the pool is off).  '
            'Default False → byte-identical to today (no eager creation) and '
            'trivially revertible, mirroring the warm_lane_pool / '
            'warm_lane_disk_guard / warm_lane_soft_floor knob convention; the '
            'reify host enables it via its own orchestrator.yaml.'
        ),
    )
    warm_lane_base_target_dir: str | None = Field(
        default=None,
        description=(
            'Absolute path of the warm BASE target/ directory to CoW-seed lane '
            'target/ from (passed as first positional arg to seed-warm-lane.sh). '
            'None (default) → derive from persistent_merge_worktree_path / '
            'reap_build_artifact_dirs[0] (i.e. <worktree_base>/_merge-verify/target). '
            'Set only when the seed base lives at a non-default location.'
        ),
    )
    warm_lane_disk_guard: bool = Field(
        default=False,
        description=(
            'When True, GitOps.acquire_warm_lane() runs a pre-acquire disk-pressure '
            'admission check (ε) before allocating a lane: invoke the reify γ script '
            '(scripts/warm-lane-disk-guard.sh check) → on exit 75 (EX_TEMPFAIL) '
            'invoke the reify δ script (scripts/warm-lane-gc.sh reclaim) to free '
            'stale capacity → re-check → if still pressured return '
            'WarmLaneUnavailable.DISK_PRESSURE (workflow requeues as transient infra, '
            'exit-75).  Fail-open on absent scripts (rc 127) — byte-identical to '
            'today until reify γ/δ are deployed, mirroring the _seed_warm_lane / '
            'refresh_warm_base absent-script convention.  Default False → '
            'byte-identical to today and trivially revertible, mirroring the '
            'warm_lane_pool knob convention.'
        ),
    )
    warm_lane_release_thin: bool = Field(
        default=False,
        description=(
            'When True, GitOps.release_warm_lane invokes reify δ '
            'scripts/thin-warm-lane.sh on the released lane after the '
            'ASSIGNED→FREE flip (free-first target reclaim; §9.5 η).  '
            'Free-only — invoked WITHOUT --reseed; the next acquire_warm_lane '
            'always re-seeds target/ from the current base regardless (D10), '
            'so net warmth is unchanged and only the idle-hold of a divergent '
            'target/ between release and a re-acquire that may never come is '
            'eliminated.  Fail-open/no-op when the script is absent.  '
            'Default False → byte-identical and revertible, mirroring the '
            'warm_lane_disk_guard convention; reify enables it in its own '
            'orchestrator.yaml.'
        ),
    )
    warm_lane_min_free_gib: int = Field(
        default=50,
        ge=0,
        description=(
            'Minimum free disk space in GiB required before admitting a warm-lane '
            'acquire.  Passed as --min-free-gib to the reify γ disk-guard script '
            '(warm-lane-disk-guard.sh check).  Effective only when '
            'warm_lane_disk_guard=True.  Matches the reify γ script env default (50 GiB).'
        ),
    )
    warm_lane_min_free_inodes: int = Field(
        default=500_000,
        ge=0,
        description=(
            'Minimum free inodes required before admitting a warm-lane acquire.  '
            'Passed as --min-free-inodes to the reify γ disk-guard script '
            '(warm-lane-disk-guard.sh check).  Effective only when '
            'warm_lane_disk_guard=True.  Matches the reify γ script env default '
            '(500 000 inodes).'
        ),
    )
    warm_lane_soft_floor: bool = Field(
        default=False,
        description=(
            'When True, GitOps._acquire_warm_lane_impl runs a PROACTIVE '
            'soft-floor throttle (θ, task 2443) AFTER the ε hard-floor '
            'disk-guard and BEFORE allocating a lane: invoke the reify ε '
            'script (scripts/warm-lane-disk-guard.sh check --soft) — on '
            'exit 3 (soft pressure, above the hard floor but below the soft '
            'one) a FRESH allocation (no existing lane mapped to the branch) '
            'returns WarmLaneUnavailable.SOFT_PRESSURE so the caller defers '
            '(backpressure requeue) rather than growing resident-divergent '
            'toward the hard floor; a REUSE of an already-mapped branch is '
            'never throttled.  Independent of warm_lane_disk_guard — an '
            'operator may enable either axis alone.  Gap-closing note: if '
            'warm_lane_disk_guard is left disabled while this is enabled, '
            'the soft-floor check also treats an exit 75 (hard pressure — '
            'the same script reports this once free space/inodes drop below '
            'the HARD floor, per its own "75 takes precedence" contract) as '
            'a defer signal, so a soft-only configuration still backpressures '
            'below the hard floor instead of allocating straight into it '
            '(see GitOps._warm_lane_soft_pressure_defer).  Fail-open on absent '
            'script (rc 127) — byte-identical to today until reify ships '
            '`check --soft`.  Default False → byte-identical and trivially '
            'revertible, mirroring the warm_lane_disk_guard knob convention.'
        ),
    )
    warm_lane_soft_free_gib: int = Field(
        default=500,
        ge=0,
        description=(
            'Soft free-GiB floor, ABOVE warm_lane_min_free_gib (the hard '
            'floor).  Passed as --soft-free-gib to the reify ε disk-guard '
            'script (warm-lane-disk-guard.sh check --soft).  Effective only '
            'when warm_lane_soft_floor=True; a model validator rejects a '
            'value <= warm_lane_min_free_gib while the knob is enabled.  '
            'Matches the reify script env default (500 GiB).'
        ),
    )
    warm_lane_soft_free_inodes: int = Field(
        default=5_000_000,
        ge=0,
        description=(
            'Soft free-inodes floor, ABOVE warm_lane_min_free_inodes (the '
            'hard floor).  Passed as --soft-free-inodes to the reify ε '
            'disk-guard script (warm-lane-disk-guard.sh check --soft).  '
            'Effective only when warm_lane_soft_floor=True; a model '
            'validator rejects a value <= warm_lane_min_free_inodes while '
            'the knob is enabled.  Matches the reify script env default '
            '(5 000 000 inodes).'
        ),
    )

    @model_validator(mode='after')
    def _reject_soft_floor_at_or_below_hard_floor(self) -> 'GitConfig':
        if self.warm_lane_soft_floor and (
            self.warm_lane_soft_free_gib <= self.warm_lane_min_free_gib
            or self.warm_lane_soft_free_inodes <= self.warm_lane_min_free_inodes
        ):
            raise ValueError(
                'GitConfig.warm_lane_soft_floor is True but the soft floor '
                'does not exceed the hard floor: warm_lane_soft_free_gib='
                f'{self.warm_lane_soft_free_gib} (hard warm_lane_min_free_gib='
                f'{self.warm_lane_min_free_gib}), warm_lane_soft_free_inodes='
                f'{self.warm_lane_soft_free_inodes} (hard '
                f'warm_lane_min_free_inodes={self.warm_lane_min_free_inodes}); '
                'raise the soft floor above its hard counterpart on both axes, '
                'or set warm_lane_soft_floor: false.'
            )
        return self

    @model_validator(mode='after')
    def _reject_bypass_command_without_clear(self) -> 'GitConfig':
        # A half-configured break-glass knob is a SILENT SAFETY-GUARD LEAK:
        # recover_red_main engages the DURABLE bypass (disabling the project's
        # always-on non-fast-forward main-gate guard) but, with no clear
        # command, can NEVER turn it back off — the guard stays disabled for
        # every subsequent ref move.  Reject at load time (loud, fail-fast)
        # rather than silently degrade at recovery time (honors the project's
        # loud-over-silent-degradation / no-silent-fail-soft norm).
        #
        # Note the deliberate asymmetry with main_gate_mark_command /
        # main_gate_unmark_command, which are NOT paired-validated: the mark
        # sentinel is ONE-SHOT (consumed by the next sanctioned txn), so
        # mark-without-unmark is only a transient leak.  The bypass is DURABLE,
        # so bypass-without-clear is a PERMANENT leak — hence the stricter gate.
        # The reverse (clear set, bypass unset) is harmless — the clear only
        # ever runs when the bypass was engaged — so it is left permissive.
        if self.main_gate_bypass_command and not self.main_gate_bypass_clear_command:
            raise ValueError(
                'GitConfig.main_gate_bypass_command is set but '
                'main_gate_bypass_clear_command is unset; recover_red_main '
                'would engage a DURABLE bypass of the non-fast-forward '
                'main-gate guard and never clear it, leaving that safety guard '
                'DISABLED for all subsequent ref moves.  Set '
                'main_gate_bypass_clear_command to the matching clear command '
                '(the git-config --unset / flag-file rm / env-reset that '
                'reverses main_gate_bypass_command), or unset '
                'main_gate_bypass_command.'
            )
        return self

    merge_spec_warm_lane_pool: bool = Field(
        default=False,
        description=(
            'When True, LOCAL speculative merge-verify slots allocate from a '
            'per-box pool of K CoW-seeded warm lanes (_spec-0 .. _spec-{K-1}), '
            'one per speculation depth, instead of using the single serial '
            '_merge-verify worktree.  K = 1 + len(enabled_verify_runners) '
            '(same expression as speculation_depth).  Each acquire re-seeds '
            'target/ from the CURRENT warm base (inv.8 always-re-seed-at-acquire). '
            'On pool exhaustion or seed failure, falls back to a cold ephemeral '
            'worktree — never blocks the scheduler (inv.6).  The existing '
            'warm/cold SHADOW safety valve (warm_verify_shadow_compare) covers '
            'spec-lane warm verifies automatically (steps 15-18).  '
            'Default False → byte-identical to today (trivially revertible, '
            'mirrors warm_lane_pool convention).  Requires reify §9.5 '
            'seed-warm-lane.sh and a CoW-capable XFS volume.'
        ),
    )
    warm_lane_reclaim_on_exhaustion: bool = Field(
        default=True,
        description=(
            'When True, GitOps.acquire_warm_lane() engages the reclaim-on-exhaustion '
            'SAFETY VALVE (task 1933): instead of returning EXHAUSTED immediately when '
            'all pool lanes are ASSIGNED, attempt to STEAL the oldest non-dispatched '
            'non-terminal lane, commit its uncommitted WIP onto its branch (so 1912 '
            'branch-retention preserves it for future reattach recovery), reset + re-seed '
            'it for the new task, and fall through the existing already-registered '
            'fresh-reset path — zero new git plumbing.  A WARNING log records every steal '
            '(ops signal: safety valve fired = real pool pressure).  NEVER steals a '
            'dispatched (live) lane — the is_dispatched predicate is re-checked '
            'synchronously under the pool lock (TOCTOU guard, task 1933, '
            'git_ops.py _try_reclaim_lane_for / warm_lane_pool.py reclaim_victim).  '
            'Falls back to EXHAUSTED/cold only when no eligible victim exists.  '
            'Requires warm_lane_pool=True.  '
            'Default True (flipped fleet-wide: PRD warm-lane-exhaustion-hardening W4) — '
            'there is no surviving reason to keep this False (Leo investigation '
            '2026-07-23).  The prior deferral rationale referenced the reify '
            '"INTERIM MARGIN" spare_warm_lanes key, which never parsed (a top-level '
            'key with no reader; 2026-07-22 incident).  The 2854 reseed-verify guard '
            '(merge 0c8137d560) protects the steal path\'s shared reseed tail.  '
            'Callbacks are installed in Harness.__init__, so the fleet adopts the new '
            'default on each unit\'s next restart (<=8h redeploy cadence), not '
            'instantaneously.  defaults.yaml deliberately does NOT list this knob — '
            'the Field default here is the single source of truth (do not add it '
            'there; a second source would drift).  Residual accepted risk (decision '
            '6): untracked victim-worktree files are NOT preserved across a steal '
            '(only tracked WIP is committed to the victim branch, per 1912 '
            'retention), and sustained over-demand causes steal-churn, which is '
            'WARNING-logged per steal (ops signal, not silently absorbed).'
        ),
    )
    warm_lane_drift_l2_threshold: int = Field(
        default=3,
        ge=1,
        description=(
            'Number of consecutive durable-record mirror failures the '
            'WarmLanePool tolerates before firing its _on_lane_record_drift '
            'callback (the Harness installs a born-at-L2 lane_record_drift '
            'filer there).  A durable-write failure (OSError — .lane-state '
            'unwritable — or IllegalLaneTransition) increments a drift counter '
            'and logs a WARNING but NEVER fails acquire/release (fail-open, '
            'PRD warm-lane-exhaustion-hardening W2b I3); at this threshold the '
            'callback fires ONCE (deduped to a single pending L2 via '
            'find_pending_l2_by_root_cause), and any subsequent SUCCESSFUL '
            'durable write resets the counter to 0 (re-arm, I4).  GitOps plumbs '
            'this straight into the pool constructor (self.config IS the '
            'GitConfig).  Default 3 per PRD Open Q2; green-tier/hot-reloadable.  '
            'defaults.yaml deliberately does NOT list this knob — the Field '
            'default here is the single source of truth (matching the other '
            'warm_lane_* knobs; a second source would drift).'
        ),
    )
    warm_lane_structural_exhaustion_l2_threshold: int = Field(
        default=5,
        ge=1,
        description=(
            'Number of CONSECUTIVE WarmLanePoolExhausted acquires GitOps '
            'tolerates before firing its _on_structural_exhaustion callback '
            '(the Harness installs a born-at-L2 '
            'warm_lane_pool_structurally_exhausted filer there).  Task 2988 '
            '(PRD ε / W3) closes the second failure pole of the 2026-07-22 '
            'incident (silent infinite requeue): with EXHAUSTED no longer '
            'counting against the per-task requeue cap, a pool that stays '
            'exhausted would otherwise requeue forever with no loud signal.  '
            'The counter is pool-GLOBAL instance state on GitOps, incremented '
            'at the single EXHAUSTED return in _acquire_warm_lane_impl and '
            'reset to 0 on ANY successful fresh lane allocation (acquire_for '
            'reused=False) or reclaim — a reuse/live-requeue does NOT reset '
            'it (not evidence of free capacity).  At this threshold the '
            'callback fires ONCE (deduped to a single pending L2 via '
            'find_pending_l2_by_root_cause), carrying the α pool census.  '
            'Unlike warm_lane_drift_l2_threshold, GitOps reads this straight '
            'off self.config (the GitConfig) — no pool-constructor plumbing, '
            'since the counter lives on GitOps itself.  Default 5 per PRD '
            'Open Q1; green-tier/hot-reloadable.  defaults.yaml deliberately '
            'does NOT list this knob — the Field default here is the single '
            'source of truth (matching the other warm_lane_* knobs; a second '
            'source would drift).'
        ),
    )
    max_interactive_worktrees: int = Field(
        default=2,
        ge=1,
        description=(
            'Cap on concurrently live interactive worktrees (the _iact-* band '
            'created by GitOps.create_interactive_worktree).  Enforced by '
            'REJECT: once the on-disk count of _iact-* worktrees under '
            'worktree_base reaches this cap, create_interactive_worktree raises '
            'InteractiveWorktreeLimitError before any git operation.  Strictly '
            'disjoint from warm_lane_pool / merge_spec_warm_lane_pool sizing — '
            'interactive worktrees never draw from either pool (isolation '
            'invariant I1).'
        ),
    )
    interactive_worktree_ttl: float = Field(
        default=86400.0,
        gt=0,
        description=(
            'Max age in SECONDS (default 86400.0 = 24h) an interactive '
            'worktree may live with no activity before the reaper sweep '
            '(task δ) reclaims it.  Consumed by the δ reaper against the '
            'created_at stamp written to .task/interactive.json; not enforced '
            'by create_interactive_worktree itself.'
        ),
    )
    iact_prefix: str = Field(
        default='_iact-',
        description=(
            'Directory-name prefix for interactive worktrees created by '
            'GitOps.create_interactive_worktree, e.g. worktree_base/'
            '"_iact-<slug>".  MUST NOT collide with the warm-lane pool prefix '
            '(_lane-) or the merge-speculation pool prefix (_spec-) — the '
            '_iact-* band is invariantly disjoint from both (isolation '
            'invariant I1).'
        ),
    )
    load_bearing_oracle_cmd: list[str] | None = Field(
        default=None,
        description=(
            'Per-project load-bearing oracle command for the merge-skew '
            'pipeline-landing tripwire (task 2382, PRD task delta): a '
            'landing whose changed files trip this oracle files exactly one '
            'advisory info escalation naming the in-flight tasks whose own '
            'branch diffs overlap the landing.  Changed files are appended '
            'as trailing argv (list[str], not a shell string, to avoid '
            'shell-quoting/injection); exit 0 = load-bearing, any other '
            'outcome (non-zero, absent, erroring) = not load-bearing.  None '
            '(default) disables the tripwire entirely — logged no-op.'
        ),
    )
    merge_config_only_full_gate_globs: list[str] = Field(
        default_factory=list,
        description=(
            'Per-project list of fnmatch globs for the dark-factory '
            'manifest-drift backstop (task 2838).  When a merge-role '
            'config-only (no .py/.rs) diff TOUCHES a file matching any glob, '
            'the full per-subproject verify gate is forced even if reify\'s '
            'scripts/verify-pipeline-guard.sh consult falls open — closing '
            'the fail-open residual that let a config-only diff CAS-advance a '
            'new manifest-drift RED onto main (incident deb-reify-964887). '
            'Matched (case-sensitive, OS-independent) against the diff\'s '
            'existing added+modified files, a safe over-approximation of '
            '"adds files that shift the manifest".  Empty (default) disables '
            'the backstop entirely — a no-op for dark-factory\'s own merges '
            'and non-reify projects, leaving the config-only fast-path '
            'byte-identical.  Not a merge-lane structural git field: a '
            'behavioural leaf tunable read fresh from the passed config per '
            'verify call (green-tier hot-reloadable).'
        ),
    )
    load_bearing_oracle_timeout_secs: float = Field(
        default=60.0,
        gt=0,
        description=(
            'Wall-clock timeout (seconds) bounding the load_bearing_oracle_cmd '
            'subprocess (task 2382, merge-skew pipeline-landing tripwire). The '
            'oracle runs synchronously in the merge-landed hot path (invariant '
            'I6: the tripwire must never block/delay the advance), so a hung '
            'or slow operator-supplied script is bounded via asyncio.wait_for '
            'rather than running unbounded; exceeding this is treated as a '
            'fail-open not-load-bearing result (logged WARNING, no '
            'escalation). Mirrors delivered_checks.check_timeout_secs.'
        ),
    )


class ChronicFlakeConfig(BaseModel):
    """Chronic pool-infra flake auto-file configuration (task 2358).

    Substrate (reify task 5142, lands separately): reify's
    ``tests/infra/run_all.sh`` persists every serial-retry pass to
    ``data/verify-logs/flaky-ledger.jsonl`` (``{ts, test, role,
    flaky_count_window}``) and emits a line-anchored ``=== CHRONIC-FLAKY
    test=<name> count=<n> window=<m> ===`` marker when a test is flaky
    ``>= threshold`` times in the last ``window`` ledger-recorded runs.

    When enabled, ``TaskWorkflow._maybe_file_chronic_flakes`` reads this
    marker/ledger after a verify completes and auto-files a medium-priority
    De-flake fix task — the gate stays green (retry-once already absorbed
    the failure); the flake debt becomes visible, owned work instead of a
    warning nobody reads. Filing is non-blocking: failures here must never
    fail the verify/merge path.

    Shipped ``enabled: false`` in defaults.yaml — gated until reify:5142
    lands and is confirmed on the target project's main, mirroring the
    ``git.offline_lane_infra_enabled`` precedent for reify-substrate
    features. All fields are green-tier hot-tunable via RELOADABLE_FIELDS.
    """

    enabled: bool = Field(
        default=False,
        description=(
            'Set to true to enable the chronic-flake auto-file detector. '
            'Requires reify task 5142 (the flaky-ledger.jsonl + '
            'CHRONIC-FLAKY marker substrate) to be present on the target '
            'project; a harmless no-op until then.'
        ),
    )
    threshold: int = Field(
        default=3,
        ge=1,
        description=(
            'Minimum number of flaky occurrences for a test within the '
            'last `window` ledger-recorded runs before it is considered '
            'chronic. Must be >= 1. Mirrors reify run_all.sh\'s own '
            'CHRONIC-FLAKY threshold (independent/fallback trigger).'
        ),
    )
    window: int = Field(
        default=20,
        ge=1,
        description=(
            'Number of most-recent ledger-recorded runs considered when '
            'computing chronic-flake occurrences. Must be >= 1. Mirrors '
            'reify run_all.sh\'s own CHRONIC-FLAKY window.'
        ),
    )
    rate_limit_days: int = Field(
        default=7,
        ge=1,
        description=(
            'Minimum days between auto-filed De-flake tasks for the same '
            'test, persisted in the FilingLedger. Must be >= 1.'
        ),
    )
    ledger_relpath: str = Field(
        default='data/verify-logs/flaky-ledger.jsonl',
        description=(
            'Project-root-relative path to reify\'s flaky-ledger.jsonl '
            '(a STABLE project-root path, not the ephemeral verify '
            'worktree).'
        ),
    )


class ZeroProgressRequeueConfig(BaseModel):
    """Zero-progress requeue backstop configuration (task 3068).

    Backstops a blind spot the per-task requeue cap structurally cannot see:
    a task that requeues forever without ever invoking an agent. The full
    causal chain — why ``_disposition_table``'s non-counting warm-lane
    dispositions make this invisible to both ceilings in
    ``Harness._apply_retry_cap`` — is documented once, canonically, in
    ``orchestrator.zero_progress_requeue``'s module docstring. Read that
    before retuning anything here.

    The alarm predicate is deliberately two-dimensional: ``threshold``
    consecutive zero-agent-invocation requeues AND ``min_span_seconds`` of
    wall clock. Requiring both is what separates a stuck task from ordinary
    busy-fleet contention, which is why those dispositions are non-counting
    in the first place.

    Unlike ``chronic_flake`` (shipped ``enabled: false``, gated on an
    un-landed reify substrate) this ships ENABLED: it reads only
    ``TaskReport`` fields that already exist, so shipping it off would leave
    the gap open indefinitely.  All fields are green-tier hot-tunable via
    RELOADABLE_FIELDS, so a noisy detector can be retuned or silenced live.
    """

    enabled: bool = Field(
        default=True,
        description=(
            'Set to false to disable the zero-progress requeue detector. '
            'Shipped enabled — this is the ONLY backstop for requeue loops '
            'that the per-task requeue cap cannot see by design. Disabling '
            'suppresses new alerts only; an already-filed alert still '
            'auto-resolves when its task resumes progress.'
        ),
    )
    threshold: int = Field(
        default=5,
        ge=1,
        description=(
            'Consecutive requeues-with-zero-agent-invocations for a single '
            'task before a blocking L1 is filed (both this AND '
            'min_span_seconds must be satisfied). Must be >= 1. Grounded in '
            'the origin incident (reify esc-5556-1): ~349 requeues across '
            '~24 tasks is ~14.5 consecutive zero-progress requeues per task, '
            'so 5 fires at roughly a third of the observed loop — hours into '
            'a 46h incident rather than at its end.'
        ),
    )
    min_span_seconds: float = Field(
        default=900.0,
        ge=0.0,
        description=(
            'Wall-clock seconds the streak must ALSO span before a blocking '
            'L1 is filed. Set to 0 to alarm on streak count alone. Exists '
            'because the dispositions this watches are non-counting '
            'precisely because they represent NORMAL busy-fleet backpressure '
            '(task 2988 flipped warm_lane_pool_exhausted to non-counting for '
            'exactly that reason): where max_concurrent_tasks exceeds the '
            'warm-lane pool size, a low-priority task can lose the pool race '
            'several dispatches in a row within seconds, and paging a human '
            'at the loudest severity tier for that would be a false '
            'positive. 900s (15 min) of CONTINUOUS zero progress is well '
            'past any contention blip but still hours short of the 46h '
            'origin incident.'
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
    df_checkout_path: str | None = Field(
        default=None,
        description=(
            'Remote-host filesystem path to the Dark-Factory orchestrator '
            '*code* checkout — the tree whose `orchestrator verify-merge` gate '
            'this runner executes over ssh.  Distinct from git_remote (the '
            'PROJECT checkout where the merge sha is pushed/tested).  When set, '
            'enables the INV-2 contract-currency auto-sync at dispatch '
            '(HEAD-compare vs the dispatcher, then git pull --ff-only + uv sync '
            'when stale; plans/merge-verdict-integrity-prd.md §1, §3.1).  '
            'Default None keeps auto-sync OFF (opt-in), byte-identical to the '
            'pre-INV-2 behaviour for every not-yet-migrated runner.'
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
    'concurrent_verify', 'sequential_lint_first', 'verify_env',
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
    sequential_lint_first: bool | None = None
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


_DEFAULT_PRICES: dict[str, dict[str, float]] = {
    # Per-model USD cost per 1M tokens, for backends without native cost
    # reporting (codex, gemini). Migrated from the former invoke.py
    # `_MODEL_COSTS` constant (task 2459). defaults.yaml's `prices:` block is
    # the operator-editable seed source; this constant is the safety-net
    # default for OrchestratorConfig.prices and backs default_price_table(),
    # which orchestrator.agents.invoke's cost estimator falls back to when no
    # config is threaded in. Kept in lockstep with defaults.yaml's `prices:`
    # block by test_config.py's test_default_price_table_matches_defaults_yaml.
    'gpt-5.4': {'input_per_1m': 2.50, 'output_per_1m': 10.00},
    'o4-mini': {'input_per_1m': 1.10, 'output_per_1m': 4.40},
    'gemini-3.1-pro-preview': {'input_per_1m': 1.25, 'output_per_1m': 5.00},
    'gemini-3-flash': {'input_per_1m': 0.075, 'output_per_1m': 0.30},
}


class PriceEntry(BaseModel):
    """Per-model USD price, in dollars per 1M tokens.

    Used by cost estimation for backends without native cost reporting
    (codex, gemini) — see orchestrator.agents.invoke._estimate_cost.
    """

    input_per_1m: float = Field(ge=0, description='USD per 1M input tokens.')
    output_per_1m: float = Field(ge=0, description='USD per 1M output tokens.')


def default_price_table() -> dict[str, dict[str, float]]:
    """Return a fresh copy of the packaged default per-model price seeds."""
    return {model: dict(rates) for model, rates in _DEFAULT_PRICES.items()}


class ConfigUnknownKey(NamedTuple):
    """A project-YAML key that has no matching model field.

    Emitted by the unknown-config-key census (see ``census_unknown_config_keys``
    below OrchestratorConfig).  ``path`` is the dotted location of the key in the
    project YAML (e.g. ``spare_warm_lanes`` or ``git.bogus_nested``).
    ``shadow_hint`` names the real dotted home when the same field name lives
    elsewhere in the model tree (e.g. a top-level ``spare_warm_lanes`` →
    ``git.spare_warm_lanes``), else ``None``.

    Defined here (before OrchestratorConfig) so the ``_unknown_key_census``
    PrivateAttr can reference it eagerly, mirroring the ModuleConfig precedent.
    """

    path: str
    shadow_hint: str | None


class ConfigIgnoredKey(NamedTuple):
    """A project-YAML key with no matching model field that was DELIBERATELY
    excused from the unknown-key census by an escape hatch.

    ``path`` is the dotted location (same shape as ``ConfigUnknownKey.path``).
    ``reason`` is ``'reserved_prefix'`` (the key's name starts with ``x_``/``x-``
    — the forward-looking convention for non-orchestrator knobs, mirroring the
    task-metadata Tier-C ``x_`` namespace) or ``'allowlist'`` (an operator listed
    it under ``config_key_census.ignore``).

    Ignored keys are excluded from ``.unknown`` and therefore from the census
    signature and the born-at-L2, but are still reported informationally by
    ``orchestrator check-config`` (at exit 0) so an over-broad glob stays
    auditable rather than becoming an invisible blind spot.
    """

    path: str
    reason: str


class ConfigKeyCensus(NamedTuple):
    """Both views produced by the ONE census walk (INV-5).

    ``unknown`` drives the loud paths (WARNING, born-at-L2, check-config exit
    code); ``ignored`` is informational only.  Because a single walk classifies
    every key into exactly one of the two, the escalation and the lint can never
    disagree about what is suppressed.
    """

    unknown: list[ConfigUnknownKey]
    ignored: list[ConfigIgnoredKey]


class ConfigKeyCensusConfig(BaseModel):
    """Operator escape hatch for the unknown-config-key census.

    MUST be declared as a real ``OrchestratorConfig`` field (it is, below):
    otherwise the very block that suppresses false positives becomes a new
    false positive, and an operator applying the documented remediation would
    trade one born-at-L2 for another.
    """

    ignore: list[str] = Field(
        default_factory=list,
        description=(
            'Dotted paths of project-YAML keys that are deliberately present for '
            'NON-OrchestratorConfig consumers (e.g. keys read by the project\'s own '
            'scripts) and must therefore not be reported as unknown config keys. '
            'Entries are matched against the dotted key path with '
            'fnmatch.fnmatchcase, so shell-style globs work — NOTE that `*` spans '
            'dots, so `cpu_governance.*` opts out that whole namespace. The '
            'converse fnmatch trap: `<name>.*` does NOT match the bare parent key '
            '`<name>`, so opting out a top-level dict key requires listing it '
            'exactly. Prefer renaming a new non-orchestrator knob under the '
            'reserved `x_`/`x-` prefix (auto-excused at any depth, no config '
            'ceremony) and reserve this list for existing key names that other '
            'tooling already greps for.'
        ),
    )


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
    # Independent bound on progress-timeout+resume churn. A ceiling-kill of a
    # productive run (transcript_turns>0) followed by its resume counts as
    # ONE iteration against max_execute_iterations (see
    # workflow._execute_iterations), so a steadily-advancing task can no
    # longer exhaust its iteration budget purely on kill/resume churn. This
    # field bounds that excluded churn independently: once
    # max_progress_resume_iterations progress-resumes have accumulated, the
    # workflow returns BLOCKED with a distinct reason, separate from the
    # zero-output-hang and generic 'Execution iterations exhausted' paths.
    # Default 20 leaves generous headroom above a normal task's 0-2 resumes.
    #
    # Combined worst case: each progress-resume invocation may itself run up
    # to invocation_timeout (the working-regime absolute cap, default 7200s)
    # before being ceiling-killed, so the churn breaker's worst-case
    # aggregate wall-clock before tripping is
    # max_progress_resume_iterations * invocation_timeout — e.g. the
    # shipped defaults (20 * 7200s) compound to ~40h. That is intentional
    # (the two knobs bound independent things: iteration-count churn vs.
    # per-attempt wall-clock), but the multiplicative interaction is easy to
    # miss when tuning either default in isolation — see invocation_timeout.
    max_progress_resume_iterations: int = Field(default=20, ge=1)
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
    # Gate for the fail-safe prior-round-resolution adjudication layer (task
    # 2523): when True (default), a post-amendment re-review runs one batched
    # LLM adjudication pass that suppresses re-emission of suggestions already
    # SETTLED in a PRIOR amendment round, failing SAFE toward EMIT on any
    # error / timeout / inconclusive result.  Green-tier / opt-out — set False
    # to disable the temporal suppression entirely (the spatial task-2750
    # amendment-delta scope is unaffected).
    suppress_resettled_review_suggestions: bool = Field(default=True)
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
    # Cohort-labelling boundary for post-rebase cost instrumentation.
    # Commits in old_base..new_base below this threshold are labelled
    # 'continuous' (normal orchestrator cadence); at-or-above with
    # is_first_rebase=True → 'post-unblock', otherwise → 'big-jump'.
    # LABELLING ONLY — does NOT trigger any re-seed behaviour ("wear the
    # cost for now").  25 cleanly separates the continuous single-/low-
    # double-digit orchestrator rebases from the 100s-of-commits
    # accumulated drift the /unblock path pays on resume.
    rebase_reseed_distance_threshold: int = Field(default=25, ge=1)
    # Fix 2 — thrash threshold for repeated infra-issue resumes on the
    # same root cause.  Counter increments when an L0 (category=
    # infra_issue) is resolved without iteration-log growth, resets to 1
    # when the log grows (steward/agent ran real work).  At threshold the
    # orchestrator promotes to L1 instead of dispatching the implementer
    # again.  Three matches the empirical reify task-2289 thrash window
    # (15 escalations on the same port-1420 collision before
    # verify-budget exhaustion).
    max_consecutive_infra_resumes: int = Field(default=3, ge=1)
    # Verify-path infra retry knobs — bounded exponential back-off for
    # transient infra OSErrors (ENOSPC etc.) caught during the verify phase.
    # max_attempts: total in-process retry attempts before the task is blocked
    #   with category='infra_issue'.  Default 5 gives ~1 min of back-off at
    #   the default base/ceiling (2+4+8+16+32 = 62s total sleep across all
    #   5 failed attempts) before declaring the infra failure persistent.
    # backoff_secs: base delay for the first retry (seconds).  Subsequent
    #   delays are min(backoff * 2^attempt, max_backoff).
    # max_backoff_secs: ceiling on a single retry delay (seconds).
    verify_infra_retry_max_attempts: int = Field(default=5, ge=1)
    verify_infra_retry_backoff_secs: float = Field(default=2.0, gt=0)
    verify_infra_retry_max_backoff_secs: float = Field(default=60.0, gt=0)
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
    # PRD plans/task-status-authority-prd.md contract C4/D4 (task 2188, omega1):
    # cadence for TaskWorkflow's background claimant-heartbeat loop, which
    # refreshes claimant_run_id/heartbeat_at on the dispatched task so
    # shared.task_claimant.is_stranded() does not treat a still-live workflow
    # as abandoned mid-phase (e.g. during one long execute-agent call).  Must
    # stay comfortably below the (separate, W10-owned) stranded-ttl.
    claimant_heartbeat_interval_secs: float = Field(default=60.0)
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

    # ── Clock-stop verify timeout (task 1916) ────────────────────────────────
    # Generic capability: while streaming verify output, recognise a configurable
    # clock-marker family and EXCLUDE the declared admission-wait span from
    # verify_command_timeout_secs so the wall-clock budget is not consumed during
    # legitimate waits (e.g. reify GPU slot starvation).  A heartbeat-idle backstop
    # ensures a genuinely-wedged wait is still killed.
    #
    # Split default: Pydantic default is False so every directly-constructed
    # OrchestratorConfig (including all existing _run_cmd test doubles) is opt-out
    # by default — byte-identical to pre-1916 behaviour.  defaults.yaml ships True
    # so the deployed orchestrator activates the seam for reify without a separate
    # ops config change (same split used by verify_cold_command_timeout_secs).
    verify_clock_stop_enabled: bool = Field(
        default=False,
        description=(
            'Enable the clock-stop verify timeout seam.  When True, the streamed '
            '_run_cmd path recognises the configured marker family and excludes '
            'declared admission-wait spans from verify_command_timeout_secs.'
        ),
    )
    # Marker strings emitted by the verify subprocess to signal stop/heartbeat/start.
    # Matched by substring containment in complete output lines, tolerant of
    # trailing fields (reason=…, waited=…, pid=…) and leading harness prefixes.
    # Defaults to reify's @@REIFY_CLOCK_*@@ family; configurable so DF can adopt
    # different consumers without code changes.
    verify_clock_stop_marker_stop: str = Field(
        default='@@REIFY_CLOCK_STOP@@',
        description='Marker emitted by the subprocess to start a clock-stop span.',
    )
    verify_clock_stop_marker_heartbeat: str = Field(
        default='@@REIFY_CLOCK_HEARTBEAT@@',
        description='Marker emitted periodically during a clock-stop span to reset the idle backstop.',
    )
    verify_clock_stop_marker_start: str = Field(
        default='@@REIFY_CLOCK_START@@',
        description='Marker emitted by the subprocess to end a clock-stop span and resume the wall-clock.',
    )
    # Heartbeat-idle backstop: if no heartbeat arrives within this many seconds
    # of the last stop/heartbeat while STOPPED, the process is killed
    # (timed_out=True → infra_timeout).  Ensures a genuinely-wedged wait is
    # reaped even though the wall-clock is paused.
    verify_clock_stop_heartbeat_idle_max: float = Field(
        default=180.0,
        gt=0,
        description=(
            'Max seconds between heartbeats (or after STOP) before the idle '
            'backstop kills the verify subprocess (gt 0).'
        ),
    )
    # Optional cumulative-stopped-time cap: when > 0, kill if total time spent
    # in STOPPED state across all stop/start cycles exceeds this many seconds
    # (timed_out=True → infra_timeout).  0 means unlimited (no cap).
    verify_clock_stop_max_total_secs: float = Field(
        default=0.0,
        ge=0,
        description=(
            'Max cumulative seconds allowed in STOPPED state across all '
            'stop/start cycles.  0 = unlimited.'
        ),
    )

    # Verification execution mode + env
    # When False, test/lint/type run sequentially within a single verify
    # invocation.  Useful for Rust workspaces where cargo takes an advisory
    # lock on target/ and the concurrent subcommands serialize anyway.
    concurrent_verify: bool = Field(default=True)
    # Merge-role fail-fast phase order.  When True AND concurrent_verify is
    # False AND role=='merge', the sequential verify branch runs LINT FIRST and
    # short-circuits on a lint failure (test+type recorded as skipped, attempt
    # fails) so a lint-only-red merge doesn't burn the long test phase before
    # the short lint phase.  Default False = opt-in: byte-unchanged for existing
    # configs.  Deliberately NOT in RELOADABLE_FIELDS (restart-tier, mirroring
    # concurrent_verify): flipping verify phase-order mid-process could split
    # behaviour across an in-flight merge.
    sequential_lint_first: bool = Field(default=False)
    # Extra env vars injected into verify commands (e.g. RUSTC_WRAPPER=sccache).
    # Distinct from env_overrides, which targets agent invocations, not verify.
    verify_env: dict[str, str] = Field(default_factory=dict)
    # When True, task-phase verify for Rust tasks rewrites
    # ``cargo --workspace`` → ``cargo -p <crate>`` for the touched crates.
    # Post-merge verify always runs workspace-wide regardless.
    scope_cargo: bool = Field(default=True)
    # Cold-verify shared-venv pre-provision command (task 2997, esc-2913-3).
    # On a COLD verify worktree the shared ``.venv`` is populated only as a SIDE
    # EFFECT of the TEST leg's ``cd <module> && uv run pytest``; the full-repo-
    # scope root LINT (``uv run ruff check …``) and TYPE (``… npx pyright``)
    # commands race that sync and fail spuriously (``Failed to spawn: ruff``;
    # ``Import "pytest" could not be resolved``).  When non-empty,
    # run_verification runs this command ONCE (coalesced per worktree) through
    # ``_run_cmd`` BEFORE the concurrent test/lint/type gather, gated on
    # ``is_cold``, so the venv is populated before the racing commands spawn.
    #
    # Split default: the Pydantic default is '' so every directly-constructed
    # OrchestratorConfig (all ``_run_cmd`` test doubles) AND every unconfigured
    # target (reify=cargo, autopilot-video) is a byte-identical no-op — the
    # gate short-circuits on the empty command.  The deployed value
    # (``uv sync --all-packages`` — see that file for why ``--extra dev`` is
    # WRONG here: the dev deps live in each member's dependency-groups, not an
    # optional ``dev`` extra) lives ONLY in dark-factory-orchestrator.yaml: the
    # orchestrator is project-agnostic, so the uv-workspace assumption must NOT
    # be hardcoded in verify.py.
    # Mirrors the concurrent_verify / verify_cold_command_timeout_secs split-
    # default convention.  Green-tier hot-reloadable (RELOADABLE_FIELDS, beside
    # verify_env): read fresh each verify with no in-flight-split risk.
    # Deliberately NOT in _OVERRIDABLE_FIELDS — it is a whole-worktree concern,
    # not a per-module override.
    verify_cold_preprovision_command: str = Field(default='')

    # Per-model USD/1M-token prices for backends without native cost
    # reporting (codex, gemini). Seeded from defaults.yaml's `prices:` block
    # (task 2459; migrated off invoke.py's former hardcoded _MODEL_COSTS).
    # Green-tier hot-reloadable (see RELOADABLE_FIELDS): a `prices` edit is
    # picked up by reload_config like any other green-tier field.
    #
    # WIRED (task 2462): the shared TaskWorkflow._invoke chokepoint
    # (workflow.py) threads prices=self.config.prices into every
    # invoke_with_cap_retry() call, for every task-workflow role
    # (architect/implementer/debugger/reviewer/merger/...) — it rides the
    # same **invoke_kwargs forwarding path as backend= (task 2457) through
    # to invoke_agent(prices=...), so orchestrator.agents.invoke.
    # _estimate_cost() resolves this LIVE, operator-tunable table for
    # codex/gemini/pi (claude ignores it — reports native cost) instead of
    # always falling back to the packaged default_price_table(). steward.py
    # and cli.py's eval-flow invoke_agent() call sites do NOT yet pass
    # prices=config.prices — still a noted future follow-up.
    prices: dict[str, PriceEntry] = Field(
        default_factory=lambda: {k: PriceEntry(**v) for k, v in _DEFAULT_PRICES.items()},
        description=(
            'Per-model USD/1M token prices for backends without native cost '
            'reporting (codex, gemini). Threaded into every task-workflow '
            'role invocation via the shared TaskWorkflow._invoke chokepoint '
            '(task 2462) — see field comment; steward.py/cli.py eval call '
            'sites are a noted future follow-up.'
        ),
    )

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
    # Staged-rollout gate for the broad merge gate (λ, task 2589).  'scoped'
    # (default) is the legacy per-touched-module verify plan — byte-identical
    # rollback path.  'full' makes merge-role (role='merge') verify run EVERY
    # REGISTERED module's full suite (pytest+ruff+pyright, per configured
    # command), not just the modules touched by the merging diff — closing
    # the source-only-diff hole where an untouched sibling module's tests
    # never ran at merge time.  Restart-only: deliberately NOT in
    # RELOADABLE_FIELDS, because flipping the gate's breadth mid-process
    # could split behaviour across an in-flight merge on the most
    # load-bearing lane.  Flipped 'scoped' → 'full' by the σ capstone and
    # activated by the τ deterministic-deploy fleet restart.
    merge_verify_breadth: Literal['scoped', 'full'] = Field(default='scoped')
    # Fix (b), task 2822 — per-land cross-check of a REMOTE merge-verify green.
    # When True (default), after a remote two-host verify returns a real-suite
    # PASS that would DECIDE a land, the merge worker re-runs the LOCAL
    # trust-anchor on the intact merge worktree as an independent second
    # opinion BEFORE the land: agree -> proceed (verdict_parity_ok); local
    # FAIL vs remote PASS -> fail-closed (withhold the land, quarantine the
    # remote runner, file a blocking verify_cross_check_mismatch escalation);
    # local RunnerUnavailable -> fail-safe (trust the remote green, emit
    # verify_cross_check_inconclusive).  Default True is safety-first per the
    # loud-over-silent norm AND provably INERT on main today: reify's
    # verify_runners is not yet enabled, so _run_post_merge_verify always gets
    # runner=None (local dispatch path) and the ``runner is not None``
    # cross-check branch never executes — zero behaviour change until Lever C
    # is turned on.  UNLIKE its restart-only merge_verify_workspace /
    # merge_verify_breadth neighbours, this knob IS green-tier hot-reloadable
    # (see RELOADABLE_FIELDS, beside verify_env): flipping the cross-check gate
    # mid-process is safe — it only ever ADDS a second-opinion verify, never
    # splits an in-flight merge's breadth.
    verify_cross_check_remote_green: bool = Field(default=True)
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
    # Deliberately NOT in RELOADABLE_FIELDS, even though it is read fresh on
    # every ``run_scoped_verification`` call (same read-fresh mechanism as its
    # merge-role sibling below): this is the general task/background-role cap,
    # and retiering it is outside task 2393's declared scope, which only added
    # the merge-only knob below. Promoting it to green-tier is a candidate
    # follow-up, not bundled here to avoid an unreviewed behaviour change to a
    # pre-existing knob.
    max_concurrent_module_verifies: int = Field(default=4, ge=1)
    # Dedicated merge-role internal-fanout cap (task 2393, T5). Merge-role
    # verifies bypass the T2 counting admission slot (`_admission_slot` is a
    # no-op for role='merge' — the anti-livelock/C-merge-priority guarantee),
    # so the merge fan-out branch of `run_scoped_verification` needs its OWN
    # bound, orthogonal to `verify_admission_task_slots`. Lowering the general
    # `max_concurrent_module_verifies` to tame merge would also over-serialize
    # task-role fan-outs, whose pytests are already bounded by the admission
    # slot. Bounds concurrent ``run_verification`` calls for ANY module,
    # regardless of language/test runner (cargo, pytest, ...) — it is the same
    # kind of fan-out guard as `max_concurrent_module_verifies` above (root
    # cause: a polluted module set once produced 226 concurrent `cargo`
    # pipelines in one merge worktree), just scoped to role='merge'; despite
    # the name, it is not pytest-specific. Default 4 matches
    # `max_concurrent_module_verifies`'s default, so untuned installs are
    # byte-preserved. Note this is a NEW, independent knob: an operator who
    # had previously retuned `max_concurrent_module_verifies` specifically to
    # bound merge fan-out will silently revert to this field's default of 4
    # post-upgrade — it does not inherit the old tuned value — and must retune
    # this knob explicitly.
    merge_verify_max_concurrent_modules: int = Field(default=4, ge=1)
    # When True, each verify command is spawned inside a transient systemd
    # ``--scope`` (its own cgroup) so a timeout/cancel can kill the ENTIRE
    # subtree by cgroup, regardless of process-group escapes (e.g. an inner GNU
    # `timeout` that setpgid'd cargo into a separate group, which defeats
    # killpg).  Defaults False (use start_new_session + killpg) so behaviour and
    # the existing test suite are unchanged; opt in per project where
    # `systemd-run --user` is available.
    verify_use_cgroup_scope: bool = Field(default=False)

    # ── Verify admission control (task 2390 T2; PRD
    # plans/verify-oversubscription-control-prd.md) ────────────────────────
    # Master switch for the per-pytest flock admission gate + role nice
    # prefix wired around the test leg of every verify spawn. Default True
    # per spec; the existing test suite is kept byte-identical via the
    # autouse `_neutralize_verify_admission` conftest fixture rather than by
    # defaulting this off.
    verify_admission_enabled: bool = Field(default=True)
    # Number of concurrent 'task'-role pytest spawns admitted through the
    # flock semaphore. 'merge' never acquires (T1 C-merge-priority no-op).
    verify_admission_task_slots: int = Field(default=1, ge=1)
    # Directory holding the flock slot files. Created lazily by verify.py's
    # admission wiring (T1's acquire_task_slot never creates it itself, so
    # a missing directory always fails open and never gates) — and only for
    # roles that can actually hold a slot ('task'/'background'); 'merge'
    # verifies never touch this directory (C-merge-priority).
    # Sentinel default '' — a default_factory lambda cannot see the sibling
    # project_root field, so the real per-project default (uid + a short
    # hash of the resolved project_root, so co-tenant projects running as
    # the same uid no longer collide on one shared slots dir) is filled in
    # by _default_verify_admission_slots_dir below, post-construction. An
    # explicit non-empty override (config/env/yaml) is preserved verbatim.
    verify_admission_slots_dir: str = Field(default='')
    # Per-role nice/ionice argv override, shlex-split when non-empty. Empty
    # (default) defers to shared.verify_admission.nice_prefix(role) — the
    # canonical tier table — so these only need setting to deviate from it.
    verify_admission_nice_merge: str = Field(default='')
    verify_admission_nice_task: str = Field(default='')
    verify_admission_nice_background: str = Field(default='')
    # pytest-xdist worker count applied to the test leg for roles {task,
    # background} only ('merge' is never `-n`-capped — it bypasses admission
    # slot-counting and is latency-critical). Default 'auto' is the T6
    # benchmark report's recommendation (plans/verify-oversubscription-
    # benchmark-2026-07-14.md): a sustained, heavily-contended host precluded
    # a clean-idle-window measurement supporting a specific cap, so the
    # behavior-preserving value is kept — '' or 'auto' is a no-op (byte-
    # identical to today's `-n auto` pyproject addopts); any other value is
    # rendered as a literal `-n <value>` (see verify_cmd.apply_pytest_numprocesses).
    # Validated below (_validate_verify_admission_pytest_n) against
    # pytest-xdist's own accepted -n values, so a typo fails loud at config
    # load/reload instead of silently reaching pytest-xdist and failing the
    # whole test leg at verify time.
    verify_admission_pytest_n: str = Field(default='auto')

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
    # Deprecated (routing alpha, task 2531): superseded by budgets.simple_task
    # / max_turns.simple_task, the real per-role resolution path _invoke uses.
    # Formally honored, not dead: _honor_deprecated_simple_task_scalars below
    # migrates a non-default value into the matching submodel field (only
    # when that field is still at its own default) with a loud WARNING.
    simple_task_budget_usd: float = Field(
        default=1.50,
        deprecated=True,
        description='Deprecated — set budgets.simple_task instead.',
    )
    simple_task_max_turns: int = Field(
        default=30,
        deprecated=True,
        description='Deprecated — set max_turns.simple_task instead.',
    )

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
    # Prefers the orchestrator's idle quiet-window (no dispatched agents), but
    # additionally force-fires after a bounded owed-age window under chronic
    # saturation — see fused_memory_restart_force_fire_after_secs below.
    # See orchestrator/src/orchestrator/service_restart.py for policy details.
    fused_memory_restart_on_merge_enabled: bool = Field(default=True)
    fused_memory_restart_debounce_secs: float = Field(default=120.0)
    fused_memory_restart_watch_prefixes: list[str] = Field(
        default_factory=lambda: ['fused-memory/src/']
    )
    fused_memory_restart_script: str = Field(
        default='scripts/restart-fused-memory.sh'
    )
    # Force-fire escape for the fused-memory coordinator (task 2817): once a
    # restart is pending, the coordinator normally only fires from the polite
    # idle path (agents_idle + debounce). Under chronic fleet saturation
    # agents_idle is rarely true, so that path can starve indefinitely and the
    # armed restart never fires (fire-once-or-never) — the operator then has to
    # restart fused-memory by hand (born-at-L2 esc-2814-1). Once a pending
    # restart has been owed for this many seconds, maybe_restart bypasses
    # agents_idle and the debounce and force-fires even on the busy-wait branch,
    # while still preferring the polite idle path for the common (healthy) case.
    # fused-memory keeps no min_interval rate cap, so nothing throttles the
    # force-fire. 0 disables force-fire (byte-identical prior behaviour); the
    # 15-min default caps the worst-case stale-bytecode window under saturation
    # (far below the orchestrator's 4500s bound — a `--drain` fused-memory
    # restart is far cheaper than a full orchestrator fleet redeploy).
    # Deliberately NOT in RELOADABLE_FIELDS: red-tier / restart-only, captured
    # once at coordinator construction (_build_service_restart_coordinator),
    # matching its orchestrator_restart_force_fire_after_secs sibling.
    fused_memory_restart_force_fire_after_secs: float = Field(
        default=900.0,
        description=(
            'Max seconds a pending fused-memory restart stays owed before '
            'force-firing (bypassing agents_idle + debounce so it still fires '
            'under chronic fleet saturation, while preferring the polite idle '
            'path in the healthy case); 0 disables; 15-min default.'
        ),
    )

    # Post-merge staleness hook — restarts dark-factory-dashboard.service after a
    # merge whose diff touches dashboard/src/.  The dashboard is a LEAF service
    # (nothing depends on it), so its restart fires regardless of idle state —
    # even while agents are dispatching.
    # See orchestrator/src/orchestrator/service_restart.py for policy details.
    dashboard_restart_on_merge_enabled: bool = Field(default=True)
    dashboard_restart_debounce_secs: float = Field(default=20.0)
    dashboard_restart_watch_prefixes: list[str] = Field(
        default_factory=lambda: ['dashboard/src/']
    )
    dashboard_restart_script: str = Field(
        default='scripts/restart-dashboard.sh'
    )

    # Post-merge staleness hook — restarts orchestrator-dark-factory.service
    # after a merge whose landed diff touches orchestrator/src/. Unlike the
    # fused-memory/dashboard restarts (plain subprocess spawn), this restart
    # MUST cgroup-escape via `systemd-run --user` (see
    # service_restart.schedule_detached_systemd_restart): under
    # orchestrator-dark-factory.service's KillMode=control-group, a
    # `systemctl restart` SIGKILLs the whole cgroup — including a same-cgroup
    # restart child spawned via asyncio.create_subprocess_exec(start_new_session
    # =True) — before it can bring the service back up. systemd-run detaches
    # the transient unit into its own cgroup so it survives the kill.
    #
    # enabled defaults to False: the orchestrator is the single most critical
    # unit, so auto-fire-on-merge must not activate merely because this
    # wiring lands. Enabling is an operator action (flip the flag after a
    # soak period + sign-off) — see U2 design notes; the recommended enable
    # mechanism is a task_kind='deterministic' pure-gate task.
    #
    # The restart is additionally merge-drain-gated (Harness._merge_pipeline_idle,
    # injected as the coordinator's restart_precondition): even at the
    # run-loop's idle quiet-window, the restart is deferred until the merge
    # queue and merge-worker pipeline are both quiescent, so it never fires
    # mid-merge. See orchestrator/src/orchestrator/service_restart.py for
    # coordinator policy details.
    orchestrator_restart_on_merge_enabled: bool = Field(default=False)
    orchestrator_restart_debounce_secs: float = Field(default=300.0)
    orchestrator_restart_watch_prefixes: list[str] = Field(
        default_factory=lambda: ['orchestrator/src/']
    )
    orchestrator_restart_script: str = Field(
        default='scripts/restart-orchestrator.sh'
    )
    # Small deferral so the current run-loop tick can log/settle before the
    # detached transient unit fires; the cgroup escape makes the restart
    # survive regardless of this value. Clamped to a minimum of 5 at the
    # executor call site (Harness._build_orchestrator_restart_coordinator),
    # mirroring DeterministicRunner's analogous clamp, so a misconfigured
    # 0/negative value can't drop the settle window or produce an invalid
    # ``--on-active=`` argument.
    orchestrator_restart_on_active_secs: int = Field(default=10)
    # Rate-cap on the orchestrator's own deploy-on-commit self-redeploy: the
    # minimum wall-clock interval (seconds) enforced between successive fires.
    # Unlike the debounce (which coalesces a *single* merge burst), this cap is
    # restart-safe — persisted to disk — so a stream of orchestrator/src merges
    # over hours can't churn the daemon through repeated redeploys. 0 disables
    # the cap (restoring pure debounce behaviour). Deliberately NOT in
    # RELOADABLE_FIELDS: red-tier / restart-only, matching its sibling
    # orchestrator_restart_* fields.
    orchestrator_restart_min_interval_secs: float = Field(
        default=28800.0,
        description=(
            'Minimum wall-clock seconds between successive orchestrator '
            'self-redeploys; 8h default caps deploy-on-commit churn. '
            '0 disables.'
        ),
    )
    # Force-fire escape for the orchestrator's own coordinator (fleet-redeploy
    # PRD task delta): once a restart is pending, the coordinator normally
    # only fires from the polite path (agents_idle + debounce + the
    # merge-drain preference). Under chronic fleet saturation agents_idle is
    # rarely true, so the polite path can starve indefinitely. Once a pending
    # restart has been owed for this many seconds, maybe_restart bypasses
    # agents_idle, the debounce, and the merge-drain preference — but NEVER
    # the min_interval 8h clock above, which is still enforced unchanged.
    # 0 disables force-fire (byte-identical behaviour); 75-min default.
    # Deliberately NOT in RELOADABLE_FIELDS: red-tier / restart-only,
    # matching its sibling orchestrator_restart_min_interval_secs.
    orchestrator_restart_force_fire_after_secs: float = Field(
        default=4500.0,
        description=(
            'Max seconds a pending orchestrator self-redeploy stays owed '
            'before force-firing (bypassing agents_idle + debounce + the '
            'merge-drain preference, but NOT the min_interval 8h clock); '
            '75-min default.'
        ),
    )
    # Merge-phase grace for the orchestrator's own coordinator (task 2753):
    # max seconds the self-redeploy (polite AND force-fire) is deferred to let
    # a pre-enqueue MERGE-phase workflow (Phase-1 rebase + scoped re-verify)
    # reach the durable merge journal (merge_queued). Post-enqueue merges
    # already survive restart via that journal, so the grace protects ONLY the
    # pre-enqueue window and releases the moment it reaches the queue. On the
    # force-fire path it bounds the hold to force_fire_after_secs + this — an
    # absolute owed-age ceiling that fires within a provable time even under
    # rolling saturation. 0 disables the grace entirely (byte-identical prior
    # behaviour). Deliberately NOT in RELOADABLE_FIELDS: red-tier / restart-only,
    # matching its siblings orchestrator_restart_force_fire_after_secs /
    # orchestrator_restart_min_interval_secs (captured at coordinator
    # construction).
    orchestrator_restart_merge_phase_grace_secs: float = Field(
        default=600.0,
        description=(
            'Max seconds an orchestrator self-redeploy (polite AND force-fire) '
            'is deferred to let a pre-enqueue MERGE-phase workflow reach the '
            'durable merge journal; bounds the force-fire hold to '
            'force_fire_after_secs + this. 0 disables. 10-min default.'
        ),
    )

    # Orphan L0 reaper — re-escalates level-0 escalations whose task has no
    # active workflow/steward (e.g. escalations emitted by the deep reviewer
    # against a synthetic ``review-*`` task_id).  Without this, such
    # escalations sit pending until the next orchestrator restart dismisses
    # them unread.  Set ``orphan_l0_reaper_enabled = False`` to disable.
    orphan_l0_reaper_enabled: bool = Field(default=True)
    orphan_l0_timeout_secs: float = Field(default=600.0)
    orphan_l0_check_interval_secs: float = Field(default=60.0)
    # Task 2931: freshness grace (seconds) for the divergence-class
    # ``routing.latest`` liveness gate in the orphan-L0 reaper. The
    # plan.files/metadata.files divergence false positive recurred after task
    # 2878 because the lock-free ``reviewer_comprehensive`` /
    # ``resettled_adjudicator`` stages hold no module locks and are absent
    # from ``_dispatched``, so ``Scheduler.is_actively_held`` cannot see a task
    # that is genuinely live mid-dispatch. Those stages DO stamp
    # ``metadata.routing.latest.decided_at`` fresh per LLM invocation; if that
    # timestamp is within this grace of the reaper's sweep ``now`` snapshot the
    # task is live mid-dispatch and its divergence L0 is deferred (not
    # promoted). Default 300s covers a single long reviewer/adjudicator
    # invocation while staying well under ``orphan_l0_timeout_secs``=600, so a
    # genuinely stranded task's stale decision still ages out and promotes.
    # Green-tier hot-reloadable (see RELOADABLE_FIELDS).
    orphan_l0_dispatch_freshness_secs: float = Field(default=300.0)
    # Task 2991: freshness grace (seconds) for the merge-phase-liveness gate in
    # the orphan-L0 divergence reaper. The divergence false positive recurred
    # for MERGE-stage tasks (successor to task 2931): the pre-enqueue merge
    # loop (rebase + scoped verify + queue submit) makes NO LLM calls, so it
    # never refreshes ``metadata.routing.latest.decided_at`` — a legitimately
    # live merge-stage task therefore fails the ``_has_fresh_dispatch`` gate
    # and gets false-promoted. A durable ``metadata.merge_phase_liveness.
    # entered_at`` stamp (restart-survivable, written at merge entry) is read
    # by ``_has_fresh_merge_phase``; if within this grace of the sweep ``now``
    # the divergence L0 is deferred (not promoted). Default 600.0 anchored to
    # ``orchestrator_restart_merge_phase_grace_secs`` (the existing legitimate
    # pre-enqueue merge-phase duration bound), not a guessed threshold.
    # Green-tier hot-reloadable (see RELOADABLE_FIELDS).
    orphan_l0_merge_phase_freshness_secs: float = Field(default=600.0)

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

    # Periodic full-suite + lint + typecheck sweep of the current main tip in
    # a throwaway detached worktree, completely off the merge hot-path so
    # per-merge latency is untouched.  When main has advanced since the last
    # sweep, runs run_full_verification (all subprojects in parallel) against a
    # _mainsweep-<hex> worktree pinned at that SHA and escalates (L1,
    # infra_issue) on drift.  Catches test-suite drift that scoped per-merge
    # verify misses (tasks 1829 / esc-1749-16).  SHA dedup avoids redundant
    # full builds when main hasn't advanced within the interval.
    main_tip_sweep_enabled: bool = Field(default=True)
    main_tip_sweep_interval_secs: float = Field(default=1800.0)

    # Periodic deterministic-strand reconciliation sweep (task 2074).
    # Defensive/non-blocking background recovery sweep for deterministic
    # gate/deploy tasks (metadata.task_kind == 'deterministic') that were
    # stranded BLOCKED by a past occurrence — e.g. a cross-unit deploy that
    # severed the orchestrator's own fused-memory connection, leaving
    # before_done_ran_at stamped but no verify/gate/provenance stamp AND an
    # empty escalation queue (task 2059).  Also re-validates already-OPEN
    # deterministic-deploy infra_issue escalations against live systemd unit
    # state, auto-resolving when the stated failure is now contradicted by a
    # healthy unit.  This is the recovery subsystem that task 2066 explicitly
    # scoped OUT (2066 only prevented NEW strands inside DeterministicRunner).
    # Auto-on, zero manual wiring — mirrors main_tip_sweep_enabled's
    # operator kill-switch convention.
    deterministic_recon_sweep_enabled: bool = Field(default=True)
    deterministic_recon_sweep_interval_secs: float = Field(default=900.0)

    # Generalized escalation-revalidation sweep (task 2114): single operator
    # kill-switch gating the harness auto-closing STALE OPEN escalations on
    # POSITIVE, fail-safe evidence — (a) the escalation's subject task has
    # become terminal (done/cancelled), read via scheduler.get_statuses and
    # closed as a new Source C inside _run_deterministic_recon_sweep; and
    # (b) a main-tip-sweep infra_issue escalation's swept SHA is superseded
    # by a later clean full-verify PASS, closed as a self-heal inside
    # _run_main_tip_sweep's passed branch.  Default-on, mirrors
    # main_tip_sweep_enabled's operator kill-switch convention.
    escalation_revalidation_enabled: bool = Field(default=True)

    # Current-tip re-confirmation gate for the main-tip sweep's CRITICAL/L1
    # filer (task 2558).  A single red-main observation is never sufficient to
    # file a destructive-intervention alarm: before submitting, _run_main_tip_sweep
    # requires BOTH task 2370's confirm_main_tip_failure_is_real subset re-run
    # AND that the current main tip is STILL the observed bad SHA (re-resolved
    # via git_ops.get_main_sha() == swept_sha).  This closes the "evidence since
    # mutated" gap (main advancing past the observed SHA during the minutes-long
    # verify), the survey §1.7 "last-green rewind named a commit that also
    # failed" precedent.  Default-on, mirrors main_tip_sweep_enabled's operator
    # kill-switch convention; set to False to restore legacy single-observation
    # filing (tip arm disabled — byte-identical post-2370 behavior).
    main_tip_sweep_rerun_confirm_enabled: bool = Field(default=True)

    # Isolated PRE-FILTER gating the main-tip sweep's full-suite retry (task
    # 3095).  When a first-pass sweep fails, run_main_tip_sweep used to
    # unconditionally re-run the WHOLE suite a second time in the same
    # worktree — minutes of background CPU that itself worsens the contention
    # it is trying to measure.  With this on, the sweep first re-runs just the
    # first-pass failing node-ids, scoped + forced-serial + generous-timeout,
    # in the already-pinned worktree: a deterministic reproduction SKIPS the
    # expensive full retry, while a non-reproduction (or any unconfirmable
    # outcome) still pays for it, so a genuine full-verify PASS remains the
    # precondition for the harness's escalation self-heal.  The pre-filter is
    # a COST gate only and never suppresses — the harness's fresh-worktree
    # confirm_main_tip_failure_is_real gate remains the sole suppression
    # authority.  Default-on, mirrors main_tip_sweep_rerun_confirm_enabled's
    # operator kill-switch convention; set to False to restore byte-identical
    # pre-3095 behavior (unconditional full retry).
    main_tip_sweep_isolated_prefilter_enabled: bool = Field(default=True)

    # Category allowlist narrowing the terminal-subject auto-close (task 2724).
    # The subject-terminal Source-C close (criterion a above) used to fire for
    # ANY category — a status-only heuristic that silently dropped still-required
    # escalations whose real work lives OUTSIDE the task record.  This allowlist
    # restricts the auto-close to categories where a done/cancelled subject truly
    # MOOTS the ask: task_failure and stranded_blocked are reliably about THIS
    # task.  infra_issue is deliberately EXCLUDED (its remediation can outlive the
    # task record; accepted tradeoff — a couple of historically-correct closes
    # return to the human queue).  A non-allowlisted category on a terminal
    # subject stays pending for a human.  Green-tier hot-reloadable (see its leaf
    # name in RELOADABLE_FIELDS) so the allowlist can be widened/narrowed live via
    # mcp__escalation__reload_config without a restart.  frozenset gives
    # order-independent equality (no spurious reload diffs) and a safe immutable
    # default.  Distinct from escalation_revalidation_enabled (the kill switch —
    # left unchanged).
    escalation_revalidation_allowlist: frozenset[str] = Field(
        default=frozenset({'task_failure', 'stranded_blocked'})
    )

    # Warm-lane auto-GC cadence loop (task 1926).
    # Periodic unconditional invocation of scripts/warm-lane-gc.sh reclaim to
    # bound FREE _lane-*/_spec-* target/ re-accretion.  gc.sh only resets FREE
    # lanes (preserves dirty/unlanded), is idempotent/cheap when nothing to
    # reclaim, and _run_warm_lane_gc_reclaim() is fail-soft (rc 127 no-op when
    # the script is absent), so the loop is harmless on hosts without gc.sh and
    # active on hosts that have it.  This is independent of warm_lane_disk_guard
    # (the reactive acquire-time burst gate, defaults False/opt-in) and
    # warm_lane_pool (gc reclaims both _lane-* and _spec-*).
    # Operator kill-switch (mirrors main_tip_sweep_enabled; set to False to
    # disable without a code change).  Default: True (MUST be on automatically
    # — the whole point of this task is to close the ε gap with zero manual
    # wiring, unlike the reactive gate that requires explicit enabling).
    warm_lane_gc_enabled: bool = Field(
        default=True,
        description=(
            'Enable the periodic warm-lane GC reclaim cadence loop '
            '(scripts/warm-lane-gc.sh reclaim).  Defaults to True so the loop '
            'fires automatically with no operator action; set to False as a '
            'kill-switch without requiring a code change.'
        ),
    )
    warm_lane_gc_interval_secs: float = Field(
        default=600.0,
        description=(
            'Interval in seconds between warm-lane GC reclaim passes '
            '(600 s = 10 min, provisional/tunable).  Independent of the '
            'reactive admit-time disk-pressure gate (warm_lane_disk_guard).'
        ),
    )
    lane_stale_report_days: float = Field(
        default=7.0,
        description=(
            'Age threshold in days beyond which a NON-terminal warm-lane '
            'assignment (durable ASSIGNED/IN_USE record for a '
            'pending/in-progress/blocked task) is reported in the digest '
            "stale-lane census (leaf γ).  Such lanes are never auto-reclaimed "
            '(WIP-preserving invariant) — the census only surfaces them for '
            'operator attention.'
        ),
    )

    # No-landings circuit-breaker (θ=1893, Phase-2 backstop, PRD §5.5).
    # When rolling landing-rate == 0 AND warm-lane free-bytes is monotonically
    # falling over a window, halt dispatch and file a non-blocking operator INFO
    # escalation ("stop digging").  Auto-resumes on a clean landing OR disk
    # recovery above margin.  Hysteresis prevents flap.  Window and disk-margin
    # are PROVISIONAL (PRD §11); calibration is a follow-up task.
    # Kill-switch for operators to disable a flapping breaker without a code change
    # (because the breaker HALTS dispatch, operators must be able to turn it off
    # without a code change — mirrors main_tip_sweep_enabled).
    no_landings_breaker_enabled: bool = Field(default=True)
    no_landings_breaker_interval_secs: float = Field(default=60.0)
    no_landings_breaker_window_samples: int = Field(
        default=30,
        ge=1,
        description=(
            'Number of consecutive samples required to trip the no-landings breaker. '
            '30 samples × 60 s/sample = 30 min > worst-case ~25 min serialized reify '
            'verify (test semaphore N=1), so a healthy slow pipeline registers ≥1 '
            'landing in the window; raise this value if verify durations grow.'
        ),
    )
    no_landings_breaker_disk_free_floor_bytes: int = Field(
        default=50 * 1024 * 1024 * 1024,  # 50 GiB
        ge=0,
        description=(
            'Absolute disk-free floor in bytes at which the no-landings breaker '
            'auto-resumes dispatch (after a disk-pressure trip) on the next breaker '
            'pass, regardless of the free-bytes level at which the trip occurred. '
            'Defaults to 50 GiB, aligned with the warm_lane_min_free_gib (50 GiB) '
            'admission threshold — "resume dispatch once disk is back above the '
            'warm-lane admission floor".'
        ),
    )

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

    # Kill-switch for the verified-green merge-queue-direct remediation
    # (stranding-remediation-scheduler-ergonomics-prd.md leaf α).  When enabled
    # (default), a stranded-`blocked` task whose warm lane holds an ASSIGNED,
    # verified-green branch (all steps done, lane tip == last passed
    # workflow_verify tip) is submitted DIRECTLY to the merge queue instead of
    # re-filed/re-pended — the merge queue runs even under a scheduler pause, so
    # this rescues work that a paused scheduler would never re-dispatch.
    # Disabling it falls back byte-identically to today's stranded_blocked
    # re-file/re-pend path (above): the verified-green detector is never
    # consulted and no merge_request is submitted.
    stranded_verified_green_merge_enabled: bool = Field(default=True)

    # Task 2408 — claimant-liveness gate.  Two mechanisms share one liveness
    # signal (shared.task_claimant): mechanism 1 refuses dispatch into a
    # `pending` task that currently has a LIVE claimant (has_live_claimant,
    # consumed by Scheduler._eligible_for_dispatch); mechanism 2 is a tick
    # phase that sweep-redispatches a genuinely-stranded `blocked` task
    # (is_stranded_blocked — no live claimant, deps resolved, not
    # deliberately parked) back to `pending`.  claimant_liveness_ttl_secs is
    # the SHARED heartbeat-staleness threshold consumed by BOTH mechanisms —
    # 300s = 5x the 60s claimant_heartbeat_interval_secs so a briefly-delayed
    # heartbeat (GC pause / backend hiccup) is never misread as stranded.
    # This is the "separate, W10-owned stranded-ttl" the
    # claimant_heartbeat_interval_secs comment above already anticipates.
    claimant_liveness_ttl_secs: float = Field(default=300.0)
    # Kill-switch for mechanism 1 (the acquire_next live-claimant dispatch
    # refusal in _eligible_for_dispatch).
    claimant_dispatch_gate_enabled: bool = Field(default=True)
    # Kill-switch for mechanism 2 (the blocked->pending direct redispatch
    # sweep).  Complements stranded_blocked_escalate_enabled above — that
    # knob gates a re-file-never-flip backstop; this one gates the direct
    # blocked->pending flip safety net.
    stranded_blocked_redispatch_enabled: bool = Field(default=True)

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

    # Empty-queue rotation skip (task 2629): idle re-check cadence when
    # _watcher_has_actionable_l1() finds no actionable L1 work. Polling is a
    # cheap on-disk EscalationQueue scan, so 60s balances responsiveness
    # against cost. Kept distinct from watcher_subprocess_restart_backoff_secs
    # (the clean-restart floor / unclean-backoff base) since the two concepts
    # are independent and would be hard to tune if conflated.
    # Tradeoff: a newly-arrived L1 during an idle period now waits up to this
    # long before the first rotation picks it up (previously a warm rotation
    # blocked on the next L1 and reacted near-immediately). Hot-reloadable
    # (green-tier) — lower it if this worst-case responsiveness is too slow
    # for a given deployment.
    watcher_empty_queue_poll_secs: float = Field(default=60.0)

    # Invocation knobs for each watcher rotation (per UnblockAutoConfig precedent).
    # watcher_model defaults to sonnet (task 2629): the top-level rotation is
    # cheap mechanical triage/orchestration; hard or investigation-class items
    # are delegated to an opus subagent via the Task tool instead of running
    # the whole rotation on opus (see SKILL.md). watcher_rotation_budget_usd
    # is retained at its opus-era sizing — sonnet/high is ~5x cheaper, so the
    # $40 ceiling now doubles as headroom for opus subagents spawned within a
    # rotation; using invoke_agent's default $5 would exhaust within minutes
    # and falsely trip the crashloop guard.
    watcher_model: str = Field(default='sonnet')
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

    # Two LIVE uses: (1) per-role timeout FALLBACK — workflow._invoke reads
    # `getattr(timeouts_cfg, role_key, self.config.invocation_timeout)` keyed
    # on the role's FULL name (role_key = role.name; routing alpha, task
    # 2531 retired the old split('_')[0] derivation), so this only engages
    # for a role whose full name has no matching TimeoutsConfig field —
    # every role _invoke currently routes (architect/implementer/debugger/
    # judge/merger/reviewer_*/simple_task) has one, so today this is a
    # defensive catch-all for a future role added without a submodel field,
    # not a live fallback. module_tagger and deep_reviewer never reach this
    # path at all — both are dispatched out-of-band (harness.py /
    # review_checkpoint.py) via their own full-name config lookups, not
    # through _invoke; (2) the working-regime ABSOLUTE CAP — the outer
    # bound on the post-turn-1 progress extension (see
    # TimeoutsConfig.working_idle_secs), regardless of how much the
    # transcript keeps advancing. Default 7200.0 matches defaults.yaml,
    # which has shipped this value since before this scalar was repurposed
    # as the absolute cap.
    invocation_timeout: float = Field(default=7200.0)

    # Models, budgets, turns, timeouts per role
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    budgets: BudgetsConfig = Field(default_factory=BudgetsConfig)
    max_turns: TurnsConfig = Field(default_factory=TurnsConfig)
    effort: EffortConfig = Field(default_factory=EffortConfig)
    timeouts: TimeoutsConfig = Field(default_factory=TimeoutsConfig)
    backends: BackendsConfig = Field(default_factory=BackendsConfig)

    # Model allowlist + fail-fast validation (task beta).
    routing: RoutingConfig = Field(default_factory=RoutingConfig)

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

    # Scheduler starvation watchdog (task 1880).
    starvation_watchdog: StarvationWatchdogConfig = Field(
        default_factory=StarvationWatchdogConfig,
    )

    # Scheduler warm-lane base hard-down watchdog (task 2061).
    warm_base_hard_down: WarmBaseHardDownConfig = Field(
        default_factory=WarmBaseHardDownConfig,
    )

    # L3b dispatch-admission load-cap gate (task 2327, DA2). An absent stanza
    # in orchestrator.yaml yields the DA-D7 enabled-by-default instance.
    psi_admission: PsiAdmissionConfig = Field(default_factory=PsiAdmissionConfig)

    # Delivered-check dep-gate sweep budget (task 2580, capability-delivered-
    # checks PRD delta). Task 2583 (epsilon) extends this sub-model further.
    delivered_checks: DeliveredChecksConfig = Field(default_factory=DeliveredChecksConfig)

    # Warm-lane session-resume guard (task γ, plans/warm-lane-session-resume-prd.md).
    # An absent stanza yields the enabled-by-default instance (default_factory
    # suffices — no defaults.yaml edit, like delivered_checks). Read by the
    # _run_slot eligibility guard only when a recovered session is present.
    session_resume: SessionResumeConfig = Field(default_factory=SessionResumeConfig)

    # Variable-depth speculative verify placement (task 2359, sibling of task
    # 2340's depth telemetry). An absent stanza in orchestrator.yaml yields
    # the disabled-by-default instance (probe_fraction=0.0, byte-identical).
    speculation_probe: SpeculationProbeConfig = Field(default_factory=SpeculationProbeConfig)

    # Deep merge-ahead chains (task 3183, plans/deep-merge-ahead-prd.md α).
    # An absent stanza in orchestrator.yaml yields the kill-switch instance
    # (chain_cap=0, byte-identical current merge behaviour); the shipped
    # defaults.yaml declares the block explicitly so the knob is discoverable.
    merge_deep: MergeDeepConfig = Field(default_factory=MergeDeepConfig)

    # Agent-transcript archival (task 2742, plans/agent-transcript-archival-prd.md
    # alpha). An absent stanza yields the enabled-by-default instance; the
    # producer hook lives in TaskWorkflow._invoke's finally and resolves its
    # archive root against project_root (not the worktree).
    transcript_archive: TranscriptArchiveConfig = Field(default_factory=TranscriptArchiveConfig)

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

    # Chronic pool-infra flake auto-file detector (task 2358). An absent
    # stanza in orchestrator.yaml yields the disabled-by-default instance
    # (gated until reify:5142's ledger/marker substrate is confirmed).
    chronic_flake: ChronicFlakeConfig = Field(default_factory=ChronicFlakeConfig)

    # Zero-progress requeue backstop (task 3068). An absent stanza in
    # orchestrator.yaml yields the ENABLED-by-default instance — this is the
    # only detector for requeue loops the per-task requeue cap cannot see.
    zero_progress_requeue: ZeroProgressRequeueConfig = Field(
        default_factory=ZeroProgressRequeueConfig
    )

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
    verify_host_unreachable_escalate_after_n: int = Field(
        default=3,
        ge=1,
        description=(
            'When Lever C is on, escalate a verify_host_unreachable alarm after this '
            'many consecutive RunnerUnavailable failures for the same remote host.  '
            'Must be >= 1.  A streak-based trip fires fast when the remote is being '
            'hammered; the time-based threshold (verify_host_unreachable_escalate_after_secs) '
            'fires even under sparse load.  Both feed the same dedup\'d alarm so at '
            'most one L1 is open per host per downtime episode.'
        ),
    )
    verify_host_unreachable_escalate_after_secs: float = Field(
        default=600.0,
        ge=0,
        description=(
            'When Lever C is on, escalate a verify_host_unreachable alarm after the '
            'remote host has been continuously unreachable for at least this many '
            'seconds (evaluated on the reprobe cadence).  0 disables the time-based '
            'trip (streak-only).  Complements verify_host_unreachable_escalate_after_n: '
            'the time-based threshold guarantees the alarm fires within ~T seconds even '
            'when merges are sparse and RU events are infrequent.'
        ),
    )
    verify_host_reprobe_interval_s: float = Field(
        default=120.0,
        ge=1,
        description=(
            'When Lever C is on, the _reprobe_loop checks quarantined remote hosts '
            'for reachability (via RemoteRunner.health()) every this many seconds.  '
            'On recovery the quarantine is cleared and the host re-enters the live '
            'pool without an orchestrator restart.  Must be >= 1.'
        ),
    )

    # Usage cap handling
    usage_cap: UsageCapConfig = Field(default_factory=UsageCapConfig)

    # Autonomous dry-run unblock hook
    unblock_auto: UnblockAutoConfig = Field(default_factory=UnblockAutoConfig)

    # Environment overrides forwarded to agent invocations
    env_overrides: dict[str, str] = Field(default_factory=dict)

    # Per-role OPT-IN endpoint-env map (e.g. ANTHROPIC_BASE_URL /
    # ANTHROPIC_AUTH_TOKEN), keyed on role.name. Empty by default. A role
    # absent from this map receives NO endpoint env from it — forwarding is
    # opt-in per role, not a global broadcast, so pointing (say) the judge at
    # an alternate Claude-compatible endpoint requires explicitly naming
    # 'judge' here. See _build_agent_env (workflow.py).
    role_env_overrides: dict[str, dict[str, str]] = Field(default_factory=dict)

    # Unknown-config-key census escape hatch.  Declared as a REAL model field
    # (not read off the raw tree alone) so the census does not self-flag the very
    # key that configures it — see ConfigKeyCensusConfig.  The allowlist is still
    # READ from the raw YAML tree by census_config_keys, because check-config must
    # keep working when the config has an unrelated value-level validation error
    # that would make a full load raise.
    config_key_census: ConfigKeyCensusConfig = Field(
        default_factory=ConfigKeyCensusConfig
    )

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

    # Unknown-config-key census (populated by load_config via
    # census_unknown_config_keys).  None means "census never ran" (direct
    # OrchestratorConfig() instantiation in evals/tests); any list (including [])
    # means "census ran".  Read through the `unknown_key_census` property below,
    # which normalizes the None sentinel to [] — mirrors _module_configs.
    _unknown_key_census: list[ConfigUnknownKey] | None = PrivateAttr(default=None)

    # Escape-hatched keys from the SAME census walk (reserved x_/x- prefix or an
    # operator config_key_census.ignore entry).  Same None-sentinel contract as
    # _unknown_key_census above; read through `ignored_key_census`.
    _ignored_key_census: list[ConfigIgnoredKey] | None = PrivateAttr(default=None)

    @field_validator('project_root', mode='after')
    @classmethod
    def _resolve_project_root(cls, v: Path) -> Path:
        return v.resolve()

    @field_validator('verify_admission_pytest_n', mode='before')
    @classmethod
    def _validate_verify_admission_pytest_n(cls, v: Any) -> Any:
        """Reject a malformed `-n` value at config load/reload (task 2394 T6).

        Mirrors pytest-xdist's own accepted ``-n``/``--numprocesses`` values:
        ``'auto'``/``'logical'`` (worker-count strategies), ``''`` (this
        knob's own no-op sentinel — see the field comment above), or a
        positive-integer string. Anything else (a typo like ``'1six'``, or a
        non-positive count like ``'0'``) would otherwise reach pytest-xdist
        unvalidated and fail the whole test leg only when a task/background
        verify next runs — this fails loud at construction/reload instead.
        Non-str values are passed through so pydantic's own type-coercion
        error reports the type mismatch rather than this validator's.
        """
        if not isinstance(v, str):
            return v
        if v in {'', 'auto', 'logical'}:
            return v
        if v.isdigit() and int(v) > 0:
            return v
        raise ValueError(
            "verify_admission_pytest_n must be '', 'auto', 'logical', or a "
            f"positive-integer string (pytest-xdist's accepted -n values); got {v!r}"
        )

    @field_validator('role_env_overrides', mode='after')
    @classmethod
    def _warn_unknown_role_env_overrides_keys(
        cls, v: dict[str, dict[str, str]]
    ) -> dict[str, dict[str, str]]:
        """WARN (never reject) on a role_env_overrides key outside KNOWN_ROLE_NAMES.

        Mirrors shared.task_metadata.validate_model_overrides's role-name check
        (task 2460 amendment) but warns instead of raising: unlike
        metadata.model_overrides (an agent-authored task-metadata write, shape-
        rejected at the fused-memory submit/update boundary), role_env_overrides
        is a restart-only operator YAML surface -- a typo here should be loud in
        the logs (the project's loud-over-silent-degradation norm) without
        taking down orchestrator startup over one mis-keyed role. A key outside
        KNOWN_ROLE_NAMES (e.g. 'judg', 'implementor') is silently never read by
        _build_agent_env's ``role_env_overrides.get(role.name, {})`` lookup --
        this makes that silent no-op visible at config load/reload.

        CAVEAT: KNOWN_ROLE_NAMES is a superset that also includes collapsed
        config-only keys ('reviewer', 'triage', 'module_tagger') which pass
        this check but are still inert here -- exactly as documented for
        metadata.model_overrides -- because _build_agent_env keys strictly on
        the full dispatch role.name (e.g. 'reviewer_comprehensive'), never the
        collapsed key. This validator only catches keys that are not even a
        recognized role name at all.
        """
        for role_name in v:
            if role_name not in KNOWN_ROLE_NAMES:
                logger.warning(
                    'role_env_overrides: unrecognized role %r (known roles: %s) -- '
                    'this entry is silently never read by _build_agent_env; check '
                    'for a typo',
                    role_name,
                    sorted(KNOWN_ROLE_NAMES),
                )
        return v

    @model_validator(mode='after')
    def _default_verify_admission_slots_dir(self) -> 'OrchestratorConfig':
        """Fill the per-project verify-admission slots dir when unset.

        ``verify_admission_slots_dir`` defaults to the sentinel ``''`` (see
        the field above) because a ``default_factory`` cannot see sibling
        fields — so the real default is derived here, post-construction,
        from the already-resolved ``project_root`` (the field_validator
        above runs first, before any model-after validator). Guarding on
        ``not self.verify_admission_slots_dir`` makes this idempotent: it
        no-ops both on an explicit override (I3) and when re-validating an
        already-derived value (e.g. inside ``apply_reload``'s
        ``model_validate(model_dump())`` round-trip). Written via
        ``object.__setattr__`` — the same bypass ``_set_leaf`` uses below —
        because ``validate_assignment=True`` would otherwise re-enter
        validation from inside this after-validator.
        """
        if not self.verify_admission_slots_dir:
            digest = hashlib.sha256(str(self.project_root).encode()).hexdigest()[:12]
            object.__setattr__(
                self,
                'verify_admission_slots_dir',
                f'/tmp/df-verify-slots-{os.getuid()}-{digest}',
            )
        return self

    @model_validator(mode='after')
    def _validate_clock_stop_markers(self) -> 'OrchestratorConfig':
        """When clock-stop is enabled, the three marker strings must be non-empty,
        mutually distinct, AND pairwise non-substrings.  No-op when disabled so
        operators can leave markers blank in configs that opt out.

        Distinctness alone is insufficient: ``_match_clock_marker`` checks
        stop→heartbeat→start in priority order by substring containment.  If
        e.g. marker_stop='STOP' and marker_start='STOPSTART', a START line would
        match the 'stop' check first and be silently misclassified.  Only the
        pairwise non-substring invariant guarantees correct classification for
        any configured marker family, not just the default @@REIFY_CLOCK_*@@ one."""
        if not self.verify_clock_stop_enabled:
            return self
        markers = {
            'verify_clock_stop_marker_stop': self.verify_clock_stop_marker_stop,
            'verify_clock_stop_marker_heartbeat': self.verify_clock_stop_marker_heartbeat,
            'verify_clock_stop_marker_start': self.verify_clock_stop_marker_start,
        }
        for name, value in markers.items():
            if not value:
                raise ValueError(
                    f'{name} must be non-empty when verify_clock_stop_enabled is True'
                )
        values = list(markers.values())
        if len(set(values)) != len(values):
            raise ValueError(
                'verify_clock_stop_marker_stop, verify_clock_stop_marker_heartbeat, and '
                'verify_clock_stop_marker_start must be mutually distinct when '
                'verify_clock_stop_enabled is True; got: '
                + repr(values)
            )
        # Reject any marker that is a substring of another.  _match_clock_marker
        # is now line-ANCHORED (line.lstrip().startswith(marker)), so the strict
        # hazard is a marker being a PREFIX of another: 'STOP' is distinct from
        # 'STOPSTART', but the former is a prefix of the latter, so a STOPSTART
        # line would match 'stop' first in the stop→heartbeat→start priority order
        # and be silently misclassified.  The substring test below is a (stronger)
        # superset of the prefix requirement — keeping it rejects prefix collisions
        # a fortiori and stays correct if matching ever loosens again.
        marker_items = list(markers.items())
        for i, (name_a, val_a) in enumerate(marker_items):
            for name_b, val_b in marker_items[i + 1:]:
                if val_a in val_b or val_b in val_a:
                    raise ValueError(
                        f'{name_a!r} ({val_a!r}) must not be a substring of '
                        f'{name_b!r} ({val_b!r}), nor vice versa, when '
                        'verify_clock_stop_enabled is True; _match_clock_marker '
                        'anchors on line-start in stop→heartbeat→start priority '
                        'order and would misclassify a line where one marker '
                        'string is a prefix of another.'
                    )
        return self

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

    @model_validator(mode='after')
    def _honor_deprecated_simple_task_scalars(self) -> 'OrchestratorConfig':
        """Formally honor the deprecated simple_task_budget_usd /
        simple_task_max_turns scalars (routing alpha, task 2531) rather than
        leave them dead: a non-default scalar migrates into the matching
        submodel field (budgets.simple_task / max_turns.simple_task) — but
        ONLY when that submodel field is still at its own default, so an
        explicitly-configured submodel value always wins and the scalar is
        then silently ignored (no migration, no warning). Every migration
        logs a loud deprecation WARNING naming the replacement, honoring the
        project's loud-over-silent-degradation norm — OrchestratorConfig
        uses extra='ignore', so a removed field set in an existing
        orchestrator.yaml would otherwise vanish with no signal at all.

        Idempotent under apply_reload's post-apply
        model_validate(model_dump()) round-trip: once migrated, the
        submodel field is no longer at its default, so a second pass is a
        no-op (the precedence check above skips migration and logs nothing).

        Reads both scalars via ``self.__dict__`` rather than plain attribute
        access — both fields are marked ``Field(deprecated=True)`` below, so
        a normal ``self.simple_task_...`` read would itself fire pydantic's
        deprecated-field access warning on every single config construction,
        not just the ones where an operator actually set a non-default
        value. ``__dict__`` holds the same validated raw value pydantic
        already stored; this only bypasses the deprecated-access warning
        wrapper, not validation.
        """
        budget_default = type(self).model_fields['simple_task_budget_usd'].default
        raw_budget = self.__dict__['simple_task_budget_usd']
        if (
            raw_budget != budget_default
            and self.budgets.simple_task == type(self.budgets).model_fields['simple_task'].default
        ):
            logger.warning(
                'simple_task_budget_usd is deprecated and will be removed; '
                'migrating its value (%s) into budgets.simple_task. Set '
                'budgets.simple_task directly in orchestrator.yaml instead.',
                raw_budget,
            )
            self.budgets.simple_task = raw_budget

        max_turns_default = type(self).model_fields['simple_task_max_turns'].default
        raw_max_turns = self.__dict__['simple_task_max_turns']
        if (
            raw_max_turns != max_turns_default
            and self.max_turns.simple_task
            == type(self.max_turns).model_fields['simple_task'].default
        ):
            logger.warning(
                'simple_task_max_turns is deprecated and will be removed; '
                'migrating its value (%s) into max_turns.simple_task. Set '
                'max_turns.simple_task directly in orchestrator.yaml instead.',
                raw_max_turns,
            )
            self.max_turns.simple_task = raw_max_turns

        return self

    @model_validator(mode='after')
    def _validate_models_in_allowlist(self) -> 'OrchestratorConfig':
        """Fail-fast: every claude-backend role's configured model string must
        be in routing.allowed_models (task beta,
        plans/adaptive-model-routing-prd.md).

        Scoped to roles whose backend is 'claude' — a codex/gemini backend
        model string is owned by the harness-backend PRD (explicitly out of
        scope here), so it is never checked against this claude-centric
        allowlist. Mirrors _validate_steward_timeout_invariant's
        ValueError-naming-the-field idiom so a bad model string fails load
        with a structured, field-named error.
        """
        allowed = self.routing.allowed_models
        for role in type(self.models).model_fields:
            # Defensive default (not `getattr(self.backends, role)` bare):
            # ModelsConfig and BackendsConfig share the same role names today,
            # but a future asymmetric field addition should skip validation
            # for the unmatched role rather than raise a raw AttributeError
            # that bypasses pydantic's ValidationError wrapping.
            if getattr(self.backends, role, 'claude') != 'claude':
                continue
            model = getattr(self.models, role)
            if model not in allowed:
                raise ValueError(
                    f'models.{role} = {model!r} is not in routing.allowed_models '
                    f'{allowed!r}. Add it to routing.allowed_models, or choose an '
                    f'already-allowed model for models.{role}.'
                )
        if self.unblock_auto.backend == 'claude' and self.unblock_auto.model not in allowed:
            raise ValueError(
                f'unblock_auto.model = {self.unblock_auto.model!r} is not in '
                f'routing.allowed_models {allowed!r}. Add it to '
                f'routing.allowed_models, or choose an already-allowed model for '
                f'unblock_auto.model.'
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

    @property
    def unknown_key_census(self) -> list[ConfigUnknownKey]:
        """Return the unknown-config-key census, normalizing the None sentinel to [].

        Populated by ``load_config`` (which stashes
        ``census_unknown_config_keys(config_path)``).  A directly-constructed
        ``OrchestratorConfig()`` never ran the census, so the sentinel stays None
        and this returns [] — mirrors ``module_configs_or_empty``.
        """
        return self._unknown_key_census or []

    @property
    def ignored_key_census(self) -> list[ConfigIgnoredKey]:
        """Return the escape-hatched half of the census, None sentinel → [].

        Informational only: these keys were deliberately excused (reserved
        ``x_``/``x-`` prefix, or an operator ``config_key_census.ignore`` entry)
        and therefore never reach the WARNING, the signature, or the L2.
        """
        return self._ignored_key_census or []

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

    @property
    def park_eviction_requests_db_path(self) -> Path:
        """Path to the park-eviction requests SQLite database.

        Stored in a dedicated file isolated from the overrides schema to
        keep the operator eviction lever narrow-blast-radius (PRD D3).
        """
        return self.project_root / 'data' / 'orchestrator' / 'park_eviction_requests.db'

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


# ---------------------------------------------------------------------------
# Unknown-config-key census (plans/warm-lane-exhaustion-hardening-prd.md leaf ζ)
#
# Pydantic's ``extra='ignore'`` silently DISCARDS unknown keys before validation
# on both OrchestratorConfig and every nested BaseModel, so a misplaced key like
# a top-level ``spare_warm_lanes: 8`` (the field actually lives on GitConfig)
# vanishes with no error.  These pure helpers detect that class of typo by
# walking the RAW project YAML against the model schema — a separate pass from
# pydantic validation, because validation never sees the dropped keys.
#
# ``ConfigUnknownKey`` itself is defined above OrchestratorConfig (so the
# ``_unknown_key_census`` PrivateAttr can reference it eagerly); the walk
# functions live here because they reference OrchestratorConfig's schema.
# ---------------------------------------------------------------------------


def _model_from_annotation(annotation: Any) -> type[BaseModel] | None:
    """Return the single nested ``BaseModel`` subclass an annotation refers to.

    Handles a bare ``SubModel`` and an ``Optional``/``SubModel | None`` wrapper.
    Returns ``None`` for scalars, ``dict[...]`` and ``list[...]`` (whose value
    models carry arbitrary operator DATA keys — the walk deliberately stops
    there) and any other non-single-BaseModel annotation.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = get_origin(annotation)
    # Only unwrap a Union (Optional[X] / X | None); NEVER dict[...]/list[...],
    # whose get_args would expose a DATA value-model (e.g. dict[str, PriceEntry]).
    if origin is Union or origin is UnionType:
        for arg in get_args(annotation):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
    return None


def _build_shadow_index(
    model_cls: type[BaseModel],
    prefix: str = '',
    index: dict[str, list[str]] | None = None,
    _ancestors: frozenset[type[BaseModel]] = frozenset(),
) -> dict[str, list[str]]:
    """Map every field NAME (lowercased) in the model tree → its dotted path(s).

    Recurses into single nested ``BaseModel`` fields only (mirroring the walk),
    so an unknown top-level ``spare_warm_lanes`` can be pointed at its real home
    ``git.spare_warm_lanes``.  ``_ancestors`` guards against pathological
    recursion cycles while still allowing the same submodel at sibling paths.
    """
    if index is None:
        index = {}
    if model_cls in _ancestors:
        return index
    child_ancestors = _ancestors | {model_cls}
    for name, field in model_cls.model_fields.items():
        dotted = f'{prefix}{name}'
        index.setdefault(name.lower(), []).append(dotted)
        sub = _model_from_annotation(field.annotation)
        if sub is not None:
            _build_shadow_index(sub, dotted + '.', index, child_ancestors)
    return index


# Key-name prefixes that excuse a key from the census at ANY depth, with no
# config ceremony.  Mirrors the task-metadata Tier-C ``x_`` namespace documented
# at docs/task-authoring.md — the forward-looking convention for a knob that
# lives in the project YAML but is consumed by the project's OWN tooling rather
# than by OrchestratorConfig.
_CENSUS_RESERVED_PREFIXES = ('x_', 'x-')


def _walk_unknown_keys(
    tree: dict[Any, Any],
    model_cls: type[BaseModel],
    prefix: str,
    shadow_index: dict[str, list[str]],
    ignore_patterns: tuple[str, ...],
    ignored: list[ConfigIgnoredKey],
) -> list[ConfigUnknownKey]:
    """Recursively collect keys in ``tree`` with no matching field on ``model_cls``.

    Matching is case-insensitive on the field NAME (mirrors
    ``model_config.case_sensitive=False``; OrchestratorConfig defines no field
    aliases).  A key with no matching field is CLASSIFIED — reserved-prefix or
    allowlisted keys are appended to *ignored*, everything else is returned as
    unknown — and in all three cases is NOT descended into.  A known key is
    descended into only when its field is a single nested ``BaseModel`` AND its
    value is a dict — scalars, ``list`` values, and ``dict`` DATA fields stop the
    walk so arbitrary operator data keys are never flagged.

    Classification happens at this ONE site (INV-5), so the loud consumers
    (WARNING/L2) and the informational one (check-config) can never disagree.
    """
    fields_lower = {
        name.lower(): (name, field) for name, field in model_cls.model_fields.items()
    }
    unknown: list[ConfigUnknownKey] = []
    for key, value in tree.items():
        key_name = str(key)
        key_lower = key_name.lower()
        dotted = f'{prefix}{key_name}'
        match = fields_lower.get(key_lower)
        if match is None:
            if key_lower.startswith(_CENSUS_RESERVED_PREFIXES):
                ignored.append(ConfigIgnoredKey(dotted, 'reserved_prefix'))
            elif any(fnmatch.fnmatchcase(dotted, pat) for pat in ignore_patterns):
                ignored.append(ConfigIgnoredKey(dotted, 'allowlist'))
            else:
                candidates = [c for c in shadow_index.get(key_lower, []) if c != dotted]
                hint = ' or '.join(candidates) if candidates else None
                unknown.append(ConfigUnknownKey(dotted, hint))
            continue
        _name, field = match
        sub = _model_from_annotation(field.annotation)
        if sub is not None and isinstance(value, dict):
            unknown.extend(
                _walk_unknown_keys(
                    value, sub, dotted + '.', shadow_index, ignore_patterns, ignored
                )
            )
    return unknown


def _census_ignore_patterns(tree: dict[Any, Any]) -> tuple[str, ...]:
    """Read ``config_key_census.ignore`` off the RAW project tree, fail-open.

    Read from the raw tree rather than a validated OrchestratorConfig so the
    census keeps working when the config has an unrelated value-level validation
    error (the same reason check-config calls the census directly).  A malformed
    hatch — non-dict block, non-list ``ignore``, non-str entries — degrades to
    "no allowlist" instead of raising: a broken escape hatch must never take out
    the census that surfaces real phantom keys.
    """
    block = tree.get('config_key_census')
    if not isinstance(block, dict):
        return ()
    raw = block.get('ignore')
    if not isinstance(raw, list):
        return ()
    return tuple(entry for entry in raw if isinstance(entry, str))


def census_config_keys(config_path: Path) -> ConfigKeyCensus:
    """Return BOTH census views for the PROJECT YAML at *config_path*.

    Parses the project file directly (NOT the defaults-merged YamlSettingsSource
    tree — defaults.yaml is version-controlled and trusted; see design decision
    1) and walks it against ``OrchestratorConfig``'s schema in ONE pass,
    classifying every non-model key as either genuinely ``unknown`` or
    deliberately ``ignored`` (reserved ``x_``/``x-`` prefix, or an operator
    ``config_key_census.ignore`` entry).  A ``None``/non-dict document or an
    unreadable/malformed file yields an empty census (fail-open — the census
    cannot detect keys it cannot parse; load_config surfaces parse errors loudly
    on its own path).
    """
    try:
        with open(config_path) as f:
            tree = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return ConfigKeyCensus([], [])
    if not isinstance(tree, dict):
        return ConfigKeyCensus([], [])
    shadow_index = _build_shadow_index(OrchestratorConfig)
    ignored: list[ConfigIgnoredKey] = []
    unknown = _walk_unknown_keys(
        tree, OrchestratorConfig, '', shadow_index,
        _census_ignore_patterns(tree), ignored,
    )
    return ConfigKeyCensus(unknown, ignored)


def census_unknown_config_keys(config_path: Path) -> list[ConfigUnknownKey]:
    """Return only the GENUINELY-unknown half of the census for *config_path*.

    Thin wrapper over ``census_config_keys`` (one walk, two views — INV-5).  Its
    signature and semantics are unchanged from before the escape hatches existed;
    escape-hatched keys are simply never in this list, which is what keeps them
    out of the census signature and therefore out of the born-at-L2.
    """
    return census_config_keys(config_path).unknown


def config_unknown_keys_signature(census: list[ConfigUnknownKey]) -> str:
    """Short, order-independent sha256 hex of the census's unknown-key PATHS.

    Paths only (hints are deterministically derived from paths + model, so they
    are redundant).  Single-sources the escalation dedup discriminator: an
    identical key-set yields a stable signature (same-set dedup, the storm
    escape) while any change to the set re-files.
    """
    joined = ','.join(sorted(uk.path for uk in census))
    return hashlib.sha256(joined.encode('utf-8')).hexdigest()[:16]


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
    # Unknown-config-key census: detect project-YAML keys that pydantic's
    # extra='ignore' silently dropped (the 2026-07-22 spare_warm_lanes incident).
    # Stash it beside _module_configs so consumers (startup L2 filer, reload
    # response, check-config) read one computed result, and warn loudly now
    # (loud-over-silent) so the phantom key is never invisible.
    #
    # SCOPE (intentional): the census walks the TOP-LEVEL project config only.
    # The per-module orchestrator.yaml files discovered just above by
    # _discover_module_configs are deliberately NOT censused, so a typo'd key in
    # a per-package orchestrator.yaml is still silently dropped by pydantic
    # extra='ignore'.  This mirrors design decision 1 (project-config-only walk):
    # the incident key lived in the top-level project YAML.  Extending the census
    # to module configs (walk each discovered ModuleConfig YAML against
    # ModuleConfig's schema) is a deferred follow-up, not an oversight.
    #
    # Both views come from ONE walk (INV-5): escape-hatched keys (reserved
    # x_/x- prefix, or an operator config_key_census.ignore entry) are stashed
    # separately and deliberately kept OUT of the WARNING below — they are
    # informational only, surfaced by `orchestrator check-config`.
    full_census = census_config_keys(config_path)
    census = full_census.unknown
    config._unknown_key_census = census
    config._ignored_key_census = full_census.ignored
    if census:
        logger.warning(
            'Config %s has %d unknown key(s) that pydantic silently dropped '
            '(extra=ignore): %s',
            config_path,
            len(census),
            '; '.join(
                uk.path + (f' (did you mean {uk.shadow_hint}?)' if uk.shadow_hint else '')
                for uk in census
            ),
        )
    return config


# ---------------------------------------------------------------------------
# Config hot-reload (plans/config-hot-reload-prd.md, task alpha): RELOADABLE_FIELDS
# ---------------------------------------------------------------------------


def _submodel_leaf_paths(field_name: str, submodel_cls: type[BaseModel]) -> frozenset[str]:
    """Return {'<field_name>.<leaf>', ...} for every field on *submodel_cls*.

    Generates a whole-submodel RELOADABLE_FIELDS group from the submodel's own
    model_fields, so adding a field to e.g. ModelsConfig automatically becomes
    reloadable without a RELOADABLE_FIELDS edit (PRD Open Q1 resolution).
    """
    return frozenset(f'{field_name}.{leaf}' for leaf in submodel_cls.model_fields)


# Code-owned allowlist of dotted OrchestratorConfig leaf paths that may be
# applied to a live config via apply_reload() without a process restart.
# See plans/config-hot-reload-prd.md §Allowlist (v1). Reload-safety is a code
# property, not operator-tunable, so this constant lives here rather than in
# orchestrator.yaml.
#
# Whole-submodel groups are generated from each submodel class's own
# model_fields (kept in sync automatically as those classes grow). NOTE: the
# "turns" submodel is exposed on OrchestratorConfig under the field name
# `max_turns` (see TurnsConfig, above), so its group is keyed 'max_turns', not
# 'turns' — the PRD's "turns.* (max_turns)" parenthetical is authoritative.
RELOADABLE_FIELDS: frozenset[str] = frozenset().union(
    _submodel_leaf_paths('models', ModelsConfig),
    _submodel_leaf_paths('budgets', BudgetsConfig),
    _submodel_leaf_paths('max_turns', TurnsConfig),
    _submodel_leaf_paths('effort', EffortConfig),
    _submodel_leaf_paths('timeouts', TimeoutsConfig),
    _submodel_leaf_paths('backends', BackendsConfig),
    _submodel_leaf_paths('unblock_auto', UnblockAutoConfig),
    # L3b dispatch-admission gate (task 2327, DA2) — mirrors the
    # fairness/starvation_watchdog submodel-group pattern: every threshold
    # is green-tier hot-reloadable.
    _submodel_leaf_paths('psi_admission', PsiAdmissionConfig),
    # Model allowlist (task beta, plans/adaptive-model-routing-prd.md):
    # green-tier hot-reloadable. apply_reload()'s existing post-apply
    # model_validate re-check already enforces
    # _validate_models_in_allowlist as a hybrid invariant, so a tightened
    # allowlist (or a typo'd models.<role>) that conflicts with the live
    # config's models/unblock_auto leaves fails closed with a
    # {reloaded: False, error: 'hybrid-invariant: ...'} + rollback — no
    # new reload code needed.
    #
    # Audited for task epsilon (resolve_route, same PRD): ladder,
    # per_model_daily_ceiling_usd and rules — the three leaves resolve_route
    # adds to RoutingConfig — are auto-covered by this same whole-submodel
    # group with no RELOADABLE_FIELDS edit. The list-valued `rules` leaf in
    # particular is safe as an _iter_leaves atomic comparison: RoutingRule
    # (and its RuleMatch/RuleSet fields) are plain pydantic BaseModels, whose
    # structural __eq__ makes list[RoutingRule] equality — and therefore
    # diff_config's `live_val != fresh_val` — behave element-wise as
    # expected, with extra='forbid' on RuleMatch/RuleSet still enforced by
    # apply_reload's post-apply model_validate. See
    # test_routing_reload.py (boundary test 11) for the reload-applies +
    # fail-closed-on-unknown-key coverage.
    _submodel_leaf_paths('routing', RoutingConfig),
    # Chronic pool-infra flake auto-file detector (task 2358) — a new
    # dedicated submodel, so its whole-submodel group auto-covers every
    # leaf (idiom shared with psi_admission/routing above).
    _submodel_leaf_paths('chronic_flake', ChronicFlakeConfig),
    # Zero-progress requeue backstop (task 3068) — same whole-submodel-group
    # idiom.  Green-tier deliberately: an operator must be able to retune the
    # threshold or silence a noisy detector WITHOUT a fleet restart, because a
    # detector you can only silence by restarting is one that gets silenced by
    # ignoring it instead.
    _submodel_leaf_paths('zero_progress_requeue', ZeroProgressRequeueConfig),
    # Variable-depth speculative verify placement (task 2359) — a new
    # dedicated submodel, same whole-submodel-group idiom: every probe knob
    # (probe_fraction/probe_depths/suppress_flake_rate) is green-tier
    # hot-reloadable with no separate RELOADABLE_FIELDS edit.
    _submodel_leaf_paths('speculation_probe', SpeculationProbeConfig),
    # Deep merge-ahead chains (task 3183, PRD alpha, decision #7) — a new
    # dedicated submodel, same whole-submodel-group idiom: chain_cap (and any
    # knob beta/gamma add later) is green-tier hot-reloadable with no separate
    # RELOADABLE_FIELDS edit, so the cap can be raised, retuned, or killed
    # (-> 0) without a restart. See MergeDeepConfig for the rest.
    _submodel_leaf_paths('merge_deep', MergeDeepConfig),
    # Agent-transcript archival (task 2742, PRD alpha) — a new dedicated
    # submodel, same whole-submodel-group idiom: enabled/root and the atomic
    # .retention leaf are all green-tier hot-reloadable with no separate
    # RELOADABLE_FIELDS edit. _iter_leaves descends exactly one level, so
    # transcript_archive.retention is one atomic BaseModel leaf (whole-retention
    # swap on any retention.* change), matching the delivered_checks/psi_admission
    # whole-submodel-group precedent.
    _submodel_leaf_paths('transcript_archive', TranscriptArchiveConfig),
    # Warm-lane session-resume guard (task γ) — a new dedicated submodel, same
    # whole-submodel-group idiom as routing/chronic_flake above: the kill switch
    # and all three ge-bounded knobs (freshness_window_secs / max_resumes_per_task
    # / fallback_storm_threshold) are green-tier hot-reloadable with no separate
    # RELOADABLE_FIELDS edit.
    _submodel_leaf_paths('session_resume', SessionResumeConfig),
    # Unknown-config-key census escape hatch (task 2989) — same whole-submodel
    # idiom.  Green-tier ON PURPOSE: the born-at-L2 this census files tells the
    # operator to add a path to config_key_census.ignore and hot-reload, and a
    # restart-only leaf would make that remediation line a lie (the reload would
    # report restart_required instead of applying).  Given the watchdog revive
    # and the 8h fleet-redeploy cadence, clearing a false-positive L2 without a
    # restart is materially better.
    _submodel_leaf_paths('config_key_census', ConfigKeyCensusConfig),
    {
        # Steward grace
        'steward_completion_timeout',
        'steward_lifetime_budget',
        # Scheduler tuning
        'fairness.skip_threshold',
        'starvation_watchdog.enabled',
        'starvation_watchdog.skip_threshold',
        'starvation_watchdog.idle_secs',
        'starvation_watchdog.idle_only_secs',
        'warm_base_hard_down.enabled',
        'warm_base_hard_down.l2_window_secs',
        # Loop-pass thresholds (+ the two crashloop-window params read live
        # per rotation; the misconfigured-guard params are a distinct
        # failure-mode family and stay restart-only)
        'idle_poll_secs',
        'orphan_l0_timeout_secs',
        # Task 2931: sibling of orphan_l0_timeout_secs — freshness grace for
        # the divergence-class routing.latest liveness gate; hot-reloadable so
        # the FP-suppression window can be tuned without a redeploy.
        'orphan_l0_dispatch_freshness_secs',
        # Task 2991: sibling of orphan_l0_dispatch_freshness_secs — freshness
        # grace for the merge-phase-liveness gate in the divergence reaper;
        # hot-reloadable so the merge-phase FP-suppression window can be tuned
        # without a redeploy.
        'orphan_l0_merge_phase_freshness_secs',
        'watcher_rotation_escalations',
        'watcher_rotation_hours',
        'watcher_max_crashloop_restarts',
        'watcher_crashloop_window_secs',
        # Empty-queue rotation-skip poll cadence (task 2629) — read live per
        # loop iteration, same reload tier as the crashloop-window params above.
        'watcher_empty_queue_poll_secs',
        # Review knobs
        'review.enabled',
        'review.interval',
        'review.full_review_on_complete',
        'review.full_review_min_interval_secs',
        'review.full_review_min_tasks',
        # Verify env (fresh config's value already carries the sccache fold)
        'verify_env',
        # Cold-verify shared-venv pre-provision command (task 2997) — green-tier
        # beside verify_env: read fresh each verify (per-verify, no in-flight
        # split), so an operator can tune or disable the pre-provision live
        # without a restart.
        'verify_cold_preprovision_command',
        # Per-land remote-green cross-check gate (task 2822, fix b) — green-tier
        # unlike its restart-only merge_verify_workspace/merge_verify_breadth
        # siblings: it only ever ADDS a second-opinion local verify, so flipping
        # it mid-process cannot split an in-flight merge's breadth.
        'verify_cross_check_remote_green',
        # Per-model USD/1M-token price table (task 2459) — green-tier like
        # verify_env above. Threaded into every task-workflow role
        # invocation via the shared TaskWorkflow._invoke chokepoint (task
        # 2462 — see OrchestratorConfig.prices docstring); a reload here is
        # picked up by the very next invocation.
        'prices',
        # Offline-lane tunables (leaf fields on the existing `git` submodel —
        # leaf-mutation only per I3)
        'git.offline_lane_test_threads',
        'git.offline_lane_poll_interval_secs',
        'git.offline_lane_red_advances_before_blocker',
        # Generic per-project offline-lane commands + legacy-numeric gate (task
        # 2789, D6 green-tier): the worker re-reads config.git each _run_once,
        # so the command list, per-command priorities, and the legacy-numeric
        # toggle hot-reload cleanly. offline_lane_commands is a whole
        # list[LaneCommand] leaf compared by equality (like routing.rules).
        # The offline_lane_enabled START gate stays restart-only (unchanged).
        'git.offline_lane_commands',
        'git.offline_lane_legacy_numeric_enabled',
        # Verify admission control (task 2390 T2; task 2394 T6 adds the
        # seventh, `_pytest_n`) — all seven knobs are green-tier: an operator
        # can retune slot counts / nice tiers / the -n cap / toggle the gate
        # without a process restart.
        'verify_admission_enabled',
        'verify_admission_task_slots',
        'verify_admission_slots_dir',
        'verify_admission_nice_merge',
        'verify_admission_nice_task',
        'verify_admission_nice_background',
        'verify_admission_pytest_n',
        # Merge-role internal-fanout cap (task 2393, T5) — same knob family:
        # read fresh per run_scoped_verification call, so a live reload
        # lowers the merge fan-out without a restart.
        'merge_verify_max_concurrent_modules',
        # Delivered-check dep-gate sweep budget (task 2580, capability-
        # delivered-checks PRD delta) — scheduler tuning, same tier as
        # fairness.skip_threshold / starvation_watchdog.*.
        'delivered_checks.max_checks_per_tick',
        # Delivered-check grace-streak escalation knobs (task 2583, epsilon
        # of the same PRD) — same green tier: an operator may flip the kill
        # switch, retune the grace window, or adjust the per-check timeout
        # without a restart.
        'delivered_checks.enabled',
        'delivered_checks.grace_cycles',
        'delivered_checks.check_timeout_secs',
        # Escalation-revalidation terminal-subject category allowlist (task
        # 2724) — green-tier hot-reloadable via mcp__escalation__reload_config.
        # A frozenset[str] leaf: _iter_leaves yields it as an atomic set-valued
        # leaf compared by equality, so widening/narrowing the allowlist reloads
        # cleanly (order-independent, no spurious diffs) with no new reload code.
        'escalation_revalidation_allowlist',
    },
)


@dataclass
class ConfigDiff:
    """Result of diff_config(): every differing leaf, bucketed by allowlist membership."""

    applied_candidates: dict[str, dict[str, Any]]
    restart_required: dict[str, dict[str, Any]]
    unchanged: int


def _iter_leaves(model: BaseModel):
    """Yield (dotted_path, value) for every leaf field of *model*.

    Descends exactly one level into BaseModel-valued fields (e.g. models,
    timeouts); dict/list/set-valued fields (verify_env, verify_runners, …)
    are yielded whole as atomic leaves compared by equality. PrivateAttrs
    (e.g. _module_configs) are never visited because they are not in
    model_fields.

    Reads values via ``__dict__`` rather than plain ``getattr`` — a field
    marked ``Field(deprecated=True)`` (e.g. simple_task_budget_usd) wraps
    attribute access with a DeprecationWarning, and this generic diff sweep
    reads every leaf on every call regardless of whether it is deprecated,
    so a plain getattr would fire that warning on every diff_config() call.
    ``__dict__`` holds the same validated value pydantic already stored;
    this only bypasses the deprecated-access warning, not validation.
    """
    for name in type(model).model_fields:
        value = model.__dict__[name]
        if isinstance(value, BaseModel):
            for sub in type(value).model_fields:
                yield f'{name}.{sub}', value.__dict__[sub]
        else:
            yield name, value


def diff_config(
    live: 'OrchestratorConfig',
    fresh: 'OrchestratorConfig',
    allowlist: frozenset[str] = RELOADABLE_FIELDS,
) -> ConfigDiff:
    """Structurally diff two fully-constructed OrchestratorConfig instances.

    Every leaf where live != fresh is categorized into applied_candidates
    (path in *allowlist*) or restart_required (otherwise); equal leaves are
    counted in ``unchanged``. Pure and synchronous — no I/O, no mutation of
    either argument.
    """
    fresh_leaves = dict(_iter_leaves(fresh))
    applied_candidates: dict[str, dict[str, Any]] = {}
    restart_required: dict[str, dict[str, Any]] = {}
    unchanged = 0
    for path, live_val in _iter_leaves(live):
        fresh_val = fresh_leaves[path]
        if live_val != fresh_val:
            entry = {'old': live_val, 'new': fresh_val}
            if path in allowlist:
                applied_candidates[path] = entry
            else:
                restart_required[path] = entry
        else:
            unchanged += 1
    return ConfigDiff(
        applied_candidates=applied_candidates,
        restart_required=restart_required,
        unchanged=unchanged,
    )


def _get_leaf(model: 'OrchestratorConfig', path: str) -> Any:
    """Return the value at dotted *path* by walking getattr across its components."""
    obj: Any = model
    for part in path.split('.'):
        obj = getattr(obj, part)
    return obj


def _set_leaf(model: 'OrchestratorConfig', path: str, value: Any) -> None:
    """Write *value* to dotted *path* on *model*, bypassing per-write validation.

    A one-component path (top-level scalar, e.g. 'verify_env') is written via
    object.__setattr__, bypassing OrchestratorConfig's validate_assignment so
    no per-write cross-field validator fires mid-apply — the single
    authoritative check is the post-apply re-validation in apply_reload. A
    two-component path (submodel leaf, e.g. 'models.architect') is written
    via plain setattr on the submodel object — submodels have no
    validate_assignment of their own, and this preserves submodel identity
    (I3) so held references (e.g. GitOps, UsageGate) observe the update in
    place.
    """
    parts = path.split('.')
    if len(parts) == 1:
        object.__setattr__(model, parts[0], value)
    else:
        sub_name, leaf = parts
        setattr(getattr(model, sub_name), leaf, value)


def apply_reload(
    live: 'OrchestratorConfig',
    fresh: 'OrchestratorConfig',
    allowlist: frozenset[str] = RELOADABLE_FIELDS,
) -> dict[str, Any]:
    """Diff *live* against *fresh* and apply every allowlisted differing leaf
    to *live* in place.

    I5 hybrid re-validation: leaf-copies bypass per-write validation (see
    _set_leaf), so after copying, the resulting *live* is re-validated as a
    whole via OrchestratorConfig.model_validate. Two individually-valid
    configs can still combine into an invalid hybrid when an allowlist omits
    one side of a cross-field invariant (e.g.
    _validate_steward_timeout_invariant) — if that happens, every applied
    leaf is synchronously rolled back to its captured old value and the
    reload is reported as failed. Nothing is left mutated on failure.

    Returns {reloaded, applied, restart_required, unchanged, error} — see
    plans/config-hot-reload-prd.md §Contract. config_path is intentionally
    absent; the harness-level reload_config (task beta) supplies it.
    """
    d = diff_config(live, fresh, allowlist)
    applied: dict[str, dict[str, Any]] = {}
    try:
        for path, old_new in d.applied_candidates.items():
            _set_leaf(live, path, old_new['new'])
            applied[path] = old_new
        OrchestratorConfig.model_validate(live.model_dump())
    except (ValidationError, ValueError) as exc:
        # Roll back every leaf applied so far (order-independent: each
        # write restores its own captured old value) so `live` is left
        # exactly as it was before this call, even on a mid-loop raise.
        for path, old_new in applied.items():
            _set_leaf(live, path, old_new['old'])
        return {
            'reloaded': False,
            'applied': {},
            'restart_required': d.restart_required,
            'unchanged': d.unchanged,
            'error': f'hybrid-invariant: {exc}',
        }
    return {
        'reloaded': True,
        'applied': applied,
        'restart_required': d.restart_required,
        'unchanged': d.unchanged,
        'error': None,
    }
