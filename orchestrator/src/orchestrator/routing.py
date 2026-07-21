"""Model routing: allowlist + fail-fast validation + per-account availability
probe + the layered route resolver.

Task beta (Phase-1 substrate of plans/adaptive-model-routing-prd.md). This
module is the PRD-named "allowlist home": ``DEFAULT_ALLOWED_MODELS`` is the
source of truth for ``OrchestratorConfig.routing``'s default (see
``config.py``'s ``RoutingConfig`` submodel and its
``_validate_models_in_allowlist`` cross-field validator).

Task epsilon adds ``resolve_route`` -- the single layered authority for
(model, effort, budget_usd, max_turns) at every LLM invocation, adopted by
``orchestrator.workflow.TaskWorkflow._invoke``. See ``resolve_route``'s own
docstring for the layer precedence.

Kept import-light at module top (stdlib only) so ``config.py`` can
``from orchestrator.routing import DEFAULT_ALLOWED_MODELS`` with no circular
import; heavier imports (e.g. ``shared.cli_invoke.invoke_claude_agent``) are
deferred to inside the functions that need them. ``OrchestratorConfig`` is
imported under ``TYPE_CHECKING`` only (never at runtime) for the same
reason -- ``config.py`` already imports FROM this module at its own top
level, so a runtime `from orchestrator.config import OrchestratorConfig`
here would create a circular import.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping
    from pathlib import Path

    from shared.cli_invoke import AgentResult
    from shared.config_models import AccountConfig

    from orchestrator.config import OrchestratorConfig, RoutingRule

logger = logging.getLogger(__name__)

# Claude-backend model aliases admitted by default (task beta). claude-fable-5
# is deliberately NOT included here — task xi admits it to the runtime
# allowlist once probe_models confirms availability across every pool account
# (see FABLE_CANDIDATE_MODEL below).
DEFAULT_ALLOWED_MODELS: tuple[str, ...] = ('haiku', 'sonnet', 'opus')

# Default model ladder (weakest -> strongest), used by the resolver (task
# epsilon) for ladder-relative rule bumps (RoutingRule.set.model == '+N',
# clamped at the top). Distinct from DEFAULT_ALLOWED_MODELS: the ladder is an
# ORDER consulted only for relative bumps, while the allowlist is an
# unordered admission set consulted at every resolution layer.
DEFAULT_LADDER: tuple[str, ...] = ('haiku', 'sonnet', 'opus')

# Candidate model probed for availability even though it is not yet admitted
# to the runtime allowlist — beta is the G3 gate that produces the
# per-account fable-availability data task xi's admission gate consumes.
FABLE_CANDIDATE_MODEL: str = 'claude-fable-5'

# Default path for the committed probe-models artifact, sibling of
# config/usage-accounts.yaml (the account-pool source of truth).
DEFAULT_PROBE_ARTIFACT_PATH: str = 'config/model-availability.yaml'

# Cheap 1-turn prompt used by probe_models's default invocation -- just
# enough to confirm the model string resolves and the account can complete a
# turn, without incurring meaningful cost.
DEFAULT_PROBE_PROMPT: str = 'Reply with the single word: ok'


def _dedup_preserve_order(items: list[str]) -> list[str]:
    """Deduplicate *items*, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


@dataclass
class ProbeReport:
    """Result of a probe_models run: the target model set actually probed,
    plus per-account x per-model availability status (see
    ``classify_probe_outcome`` for the status vocabulary)."""

    models: list[str]
    accounts: dict[str, dict[str, str]]


def classify_probe_outcome(outcome: object) -> str:
    """Map a ``shared.invocation_outcome.InvocationOutcome`` to a probe
    status string: OK->available, ModelNotFound->unavailable,
    AuthFailed->auth_error, CapHit/NearCap->capped, else->error.

    Pure -- reads only *outcome*, performs no I/O. Note this 'error'
    catch-all is only reachable via a classified Failure *outcome*; a raised
    exception from *invoke_fn* itself is handled separately by
    ``probe_models`` as the distinct ``'invoke_error'`` status.
    """
    from shared.invocation_outcome import OK, AuthFailed, CapHit, ModelNotFound, NearCap

    if isinstance(outcome, OK):
        return 'available'
    if isinstance(outcome, ModelNotFound):
        return 'unavailable'
    if isinstance(outcome, AuthFailed):
        return 'auth_error'
    if isinstance(outcome, (CapHit, NearCap)):
        return 'capped'
    return 'error'


async def probe_models(
    accounts: list[AccountConfig],
    allowed_models: list[str],
    *,
    models: list[str] | None = None,
    invoke_fn: Callable[..., Awaitable[AgentResult]] | None = None,
    token_resolver: Callable[[str], str | None] = os.environ.get,
    prompt: str = DEFAULT_PROBE_PROMPT,
    cwd: Path | None = None,
    max_turns: int = 1,
    budget_usd: float = 0.05,
) -> ProbeReport:
    """Probe every (account, model) pair for availability.

    The target model set defaults to ``dedup(allowed_models +
    [FABLE_CANDIDATE_MODEL])``, order-preserving, so the probe always
    exercises ``claude-fable-5`` even though it is not yet admitted to the
    runtime allowlist (task xi's G3 gate; see this task's plan
    design_decisions) -- pass *models* explicitly to override.

    For each account, the OAuth token is resolved once via
    ``token_resolver(account.oauth_token_env)``. When unresolvable, every
    target model is recorded as ``'no_token'`` for that account and
    *invoke_fn* is never called for it. Otherwise, *invoke_fn* (default
    ``invoke_claude_agent``) is called once per target model with a cheap
    1-turn invocation, and the result is classified via
    ``classify_invocation`` / ``classify_probe_outcome`` into a status
    string. If *invoke_fn* raises (network error, subprocess crash, or any
    other exception not surfaced as an ``AgentResult``), that single
    (account, model) pair is recorded as ``'invoke_error'`` and the probe
    continues -- a single transient failure must not abort the whole run
    and discard every status already collected.

    *invoke_fn* and *token_resolver* are dependency-injected (mirrors
    ``invoke_with_cap_retry``'s ``invoke_fn=`` seam) so callers can drive
    this network-free in tests.

    Dispatch is intentionally sequential (both across accounts and across
    models within an account), not concurrent: this is a manually-run
    diagnostic CLI, not a latency-sensitive path, and sequential dispatch
    keeps each account's probe traffic from hammering that account's own
    rate limit. Revisit with a bounded ``asyncio.gather`` if probe latency
    becomes a real concern.
    """
    from pathlib import Path as _Path

    from shared.cli_invoke import invoke_claude_agent
    from shared.invocation_outcome import classify_invocation

    invoke = invoke_fn or invoke_claude_agent
    target_models = (
        list(models)
        if models is not None
        else _dedup_preserve_order([*allowed_models, FABLE_CANDIDATE_MODEL])
    )
    resolved_cwd = cwd or _Path.cwd()

    report_accounts: dict[str, dict[str, str]] = {}
    for account in accounts:
        token = token_resolver(account.oauth_token_env)
        if not token:
            report_accounts[account.name] = dict.fromkeys(target_models, 'no_token')
            continue

        statuses: dict[str, str] = {}
        for model in target_models:
            try:
                result = await invoke(
                    prompt=prompt,
                    system_prompt='',
                    cwd=resolved_cwd,
                    model=model,
                    max_turns=max_turns,
                    max_budget_usd=budget_usd,
                    oauth_token=token,
                )
            except Exception as exc:
                # A single (account, model) pair crashing (network error,
                # subprocess crash, etc.) must not abort the whole probe and
                # discard every status already collected for prior
                # accounts/models -- record it and keep going.
                logger.warning(
                    'probe_models: invoke_fn raised for account=%s model=%s: %s',
                    account.name, model, exc,
                )
                statuses[model] = 'invoke_error'
                continue
            outcome = classify_invocation(result, strict_confirm=False, backend='claude')
            statuses[model] = classify_probe_outcome(outcome)
        report_accounts[account.name] = statuses

    return ProbeReport(models=target_models, accounts=report_accounts)


def render_probe_artifact(report: ProbeReport, generated_at: str) -> str:
    """Render *report* as a deterministic, committable YAML artifact string.

    Pure -- no clock/env reads; *generated_at* is supplied by the caller so
    the same ``(report, generated_at)`` pair always serializes to the same
    bytes, safe to commit and diff. Field order is fixed (``generated_at``,
    ``models``, ``accounts``) via ``sort_keys=False``.
    """
    import yaml

    payload = {
        'generated_at': generated_at,
        'models': list(report.models),
        'accounts': {
            name: dict(statuses) for name, statuses in report.accounts.items()
        },
    }
    return yaml.safe_dump(payload, sort_keys=False)


# ---------------------------------------------------------------------------
# Route resolution (task epsilon, plans/adaptive-model-routing-prd.md)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanShape:
    """Snapshot of a task's plan shape, consulted by the ``plan_min_steps``/
    ``plan_min_modules``/``module_prefix`` RuleMatch conditions."""

    step_count: int
    module_paths: tuple[str, ...]


@dataclass(frozen=True)
class RoleDefaults:
    """The ``AgentRole`` dataclass's own defaults -- the resolver's layer-4,
    always-available base (invariant 1: Total; ``resolve_route`` never
    raises because this layer is unconditional)."""

    model: str
    effort: str
    budget_usd: float
    max_turns: int


@dataclass(frozen=True)
class RouteInputs:
    """Everything ``resolve_route`` needs to resolve one invocation's route.

    Pure data -- the ``OrchestratorConfig`` is passed to ``resolve_route``
    as a separate argument (not embedded here) so the same inputs can be
    replayed against different config snapshots in tests.

    ``spend_by_model`` is the trailing-24h USD spend per model that carries
    a configured ``routing.per_model_daily_ceiling_usd`` entry -- callers
    (``TaskWorkflow._invoke``) only populate it for ceiling'd models, so it
    is empty ``{}`` at stock config (no ceilings configured -> no cost_store
    read fires).

    ``scope_capacity`` is the resolve-time advisory per-scoped-model headroom
    snapshot (task δ / invariant S7-S8) -- the gate's
    ``scope_capacity_snapshot()`` (task γ), True per scoped model iff >=1
    account has headroom. Threaded from ``TaskWorkflow._invoke``; ``None``
    (no gate wired, or a snapshot read hiccup) or a model ABSENT from the
    mapping skips the capacity fail-safe entirely, so an absent/stale
    snapshot NEVER blocks dispatch (S7). It is advisory only: the gate's own
    invoke-time scope predicate stays authoritative, so a stale snapshot
    degrades to a scope-wait/failover, never a wrong-and-stuck decision (S8).
    """

    role_name: str
    task_id: str
    task_metadata: Mapping[str, Any]
    plan_shape: PlanShape | None
    routing_tier: int
    dispatch_count: int
    role_defaults: RoleDefaults
    spend_by_model: Mapping[str, float] = field(default_factory=dict)
    scope_capacity: Mapping[str, bool] | None = None


@dataclass(frozen=True)
class RoutingDecision:
    """The resolved (model, effort, budget_usd, max_turns) for one
    invocation, plus provenance.

    ``source_layer`` tracks MODEL provenance only (one of
    ``'role_default'``, ``'config'``, ``'policy_rule'``,
    ``'metadata_override'``) -- effort/budget_usd/max_turns are each
    resolved field-wise from the highest layer that specifies them, but a
    single decision carries only one source_layer axis (see
    ``resolve_route``). ``rule_id`` is set whenever a policy rule matched
    (independent of whether that rule went on to set ``model`` -- see
    ``resolve_route``). ``rejected`` accumulates a namespaced reason string
    for every layer that was skipped (fail-safe allowlist/ceiling
    rejections, or a ladder-relative bump that could not be applied).
    """

    model: str
    effort: str
    budget_usd: float
    max_turns: int
    source_layer: str
    rule_id: str | None
    rejected: tuple[str, ...] = ()


def _config_key(role_name: str) -> str:
    """Collapse any ``reviewer*`` variant to the shared ``'reviewer'`` config key.

    Mirrors the pre-epsilon inline collapse in ``workflow.py``'s ``_invoke``
    (``if role.name.startswith('reviewer')``) so every reviewer variant
    resolves the same ``.reviewer`` config fields (byte-equivalence,
    invariant 3).
    """
    return 'reviewer' if role_name.startswith('reviewer') else role_name


def _task_simple_saturated(task_metadata: Mapping[str, Any]) -> bool:
    """Read ``metadata.routing.simple_saturated``, defaulting to False.

    Mirrors ``shared.task_metadata.RoutingState``'s on-disk storage shape
    (``metadata['routing']['simple_saturated']``) without importing that
    module (kept import-light) -- a missing/non-dict ``routing`` key
    degrades to False rather than raising, matching
    ``RoutingState.from_metadata``'s own tolerant-degrade philosophy.
    """
    routing_state = task_metadata.get('routing')
    if not isinstance(routing_state, dict):
        return False
    return bool(routing_state.get('simple_saturated', False))


def _rule_matches(rule: RoutingRule, inputs: RouteInputs) -> bool:
    """Return True iff every condition *rule.match* sets is satisfied (AND).

    Closed vocabulary (PRD task epsilon): ``role`` (membership),
    ``task_complexity``/``task_priority`` (equality vs ``inputs.
    task_metadata``), ``plan_min_steps``/``plan_min_modules``/
    ``module_prefix`` (plan-shape conditions -- a ``None`` plan_shape fails
    ANY of these three), ``min_routing_tier``, ``min_dispatch_count``,
    ``simple_saturated``. When both ``module_prefix`` and
    ``plan_min_modules`` are present, ``plan_min_modules`` counts only the
    prefix-matched modules (not the total) -- this is what reproduces the
    pre-epsilon Rust heuristic (``_select_model_for_role``) exactly.

    CAVEAT: ``task_priority`` matches only ``inputs.task_metadata['priority']``,
    NOT the task's top-level ``priority`` field -- see ``config.RuleMatch``'s
    docstring for the full note.
    """
    match = rule.match

    if match.role is not None and inputs.role_name not in match.role:
        return False
    if (
        match.task_complexity is not None
        and inputs.task_metadata.get('complexity') != match.task_complexity
    ):
        return False
    if (
        match.task_priority is not None
        and inputs.task_metadata.get('priority') != match.task_priority
    ):
        return False
    if match.min_routing_tier is not None and inputs.routing_tier < match.min_routing_tier:
        return False
    if match.min_dispatch_count is not None and inputs.dispatch_count < match.min_dispatch_count:
        return False
    if (
        match.simple_saturated is not None
        and _task_simple_saturated(inputs.task_metadata) != match.simple_saturated
    ):
        return False

    needs_plan = (
        match.plan_min_steps is not None
        or match.plan_min_modules is not None
        or match.module_prefix is not None
    )
    if needs_plan:
        plan_shape = inputs.plan_shape
        if plan_shape is None:
            return False
        if match.plan_min_steps is not None and plan_shape.step_count < match.plan_min_steps:
            return False
        if match.module_prefix is not None:
            prefixed = [m for m in plan_shape.module_paths if m.startswith(match.module_prefix)]
            if not prefixed:
                return False
            if match.plan_min_modules is not None and len(prefixed) < match.plan_min_modules:
                return False
        elif (
            match.plan_min_modules is not None
            and len(plan_shape.module_paths) < match.plan_min_modules
        ):
            return False

    return True


def _model_rejection_reason(
    candidate: str,
    config: OrchestratorConfig,
    spend_by_model: Mapping[str, float],
    scope_capacity: Mapping[str, bool] | None = None,
) -> str | None:
    """Return a fail-safe rejection reason for *candidate*, or None if it
    passes validation (invariants 2, 6, and S7).

    Invariant 2: *candidate* must be a member of ``config.routing.
    allowed_models`` -- ``'model-not-in-allowlist'`` otherwise. Invariant 6:
    when ``config.routing.per_model_daily_ceiling_usd`` configures a ceiling
    for *candidate*, its trailing-24h spend (*spend_by_model*, caller-
    supplied -- see ``RouteInputs.spend_by_model``) must be strictly under
    it -- ``'model-ceiling-exhausted'`` otherwise. A model with no
    configured ceiling never trips this second check.

    Invariant S7 (task δ): when *scope_capacity* (the resolve-time advisory
    snapshot -- see ``RouteInputs.scope_capacity``) reports *candidate*'s
    account scope as having no headroom (``scope_capacity[candidate] is
    False``), the model is rejected -- ``'model-capacity-exhausted'``. This
    check is skipped entirely (no rejection) when *scope_capacity* is
    ``None`` or *candidate* is ABSENT from the mapping, so an absent/stale
    snapshot never blocks dispatch. Checked LAST (after allowlist/ceiling):
    ordering only decides which reason string wins when several checks fail
    at once, and the more fundamental allowlist/ceiling cause is the more
    informative one to surface first. All three checks are gated to
    claude-backend roles by the callers (see ``resolve_route``).
    """
    if candidate not in config.routing.allowed_models:
        return 'model-not-in-allowlist'
    ceiling = config.routing.per_model_daily_ceiling_usd.get(candidate)
    if ceiling is not None and spend_by_model.get(candidate, 0.0) >= ceiling:
        return 'model-ceiling-exhausted'
    if scope_capacity is not None and scope_capacity.get(candidate) is False:
        return 'model-capacity-exhausted'
    return None


def _resolve_ladder_relative(spec: str, incoming_model: str, ladder: list[str]) -> str | None:
    """Resolve a ladder-relative ``RuleSet.model`` *spec* (``'+N'``) against
    *incoming_model*'s position in *ladder*, clamped at the ladder top
    (invariant 5).

    Returns None -- "cannot be bumped" -- when *incoming_model* is not a
    member of *ladder*, or *spec* is not a valid integer offset.
    """
    try:
        offset = int(spec)
    except ValueError:
        return None
    if incoming_model not in ladder:
        return None
    idx = ladder.index(incoming_model)
    return ladder[min(idx + offset, len(ladder) - 1)]


def resolve_route(inputs: RouteInputs, config: OrchestratorConfig) -> RoutingDecision:
    """Resolve the (model, effort, budget_usd, max_turns) for one LLM
    invocation.

    Layered by precedence, highest first:

    1. ``metadata_override`` -- ``inputs.task_metadata['model_overrides']
       [inputs.role_name]``, if present (sets ``model`` only).
    2. ``policy_rule`` -- the first rule in ``config.routing.rules`` (list
       order) whose ``match`` conditions all hold; its ``set`` fields
       override whichever of model/effort/budget_usd/max_turns it
       specifies. ``rule_id`` is recorded as soon as a rule matches,
       independent of whether its ``set.model`` goes on to apply.
    3. ``config`` -- ``config.models``/``budgets``/``max_turns``/``effort``,
       keyed by ``_config_key(inputs.role_name)`` (the reviewer* collapse).
    4. ``role_default`` -- ``inputs.role_defaults`` (invariant 1: Total,
       always available and unconditional -- this is the only layer never
       subject to fail-safe validation -- so this function never raises).

    Each of effort/budget_usd/max_turns is resolved independently
    field-by-field from the highest layer that specifies it;
    ``source_layer`` tracks only ``model``'s provenance. Pure and
    synchronous -- no I/O.

    Fail-safe validation (invariants 2, 6, and S7): whenever layer 3, 2, or
    1 would set ``model``, the candidate (after ladder-relative resolution
    for a policy rule's ``'+N'``) is checked against ``config.routing.
    allowed_models``, ``per_model_daily_ceiling_usd``, and ``inputs.
    scope_capacity`` (the resolve-time advisory account-scope headroom
    snapshot -- see ``RouteInputs.scope_capacity``). On failure that layer's
    model assignment is skipped -- ``model``/``source_layer`` keep whatever
    the next-lower-precedence layer already validated -- and a namespaced
    ``"<layer>:<reason>"`` string is appended to ``rejected``. A dispatch is
    never blocked by a routing mis-config: an absent/stale ``scope_capacity``
    snapshot (``None``, or the model absent from it) simply skips the
    capacity check (S7), and the gate's own invoke-time scope predicate
    stays authoritative so staleness degrades to a scope-wait/failover (S8).

    This validation is scoped to claude-backend roles only (``config.
    backends`` keyed the same way as ``config.models`` -- the reviewer*
    collapse applies identically), mirroring ``config.py``'s
    ``_validate_models_in_allowlist``: a non-claude-backend role's model
    string is the harness-backend PRD's axis and must never be rejected
    against this claude-centric allowlist/ceiling.
    """
    model = inputs.role_defaults.model
    effort = inputs.role_defaults.effort
    budget_usd = inputs.role_defaults.budget_usd
    max_turns = inputs.role_defaults.max_turns
    source_layer = 'role_default'
    rule_id: str | None = None
    rejected: list[str] = []

    key = _config_key(inputs.role_name)

    # Fail-safe validation (invariants 2/6) is scoped to claude-backend
    # roles ONLY -- mirrors config.py's _validate_models_in_allowlist, which
    # never checks a non-claude-backend role's configured model string
    # against the claude-centric routing.allowed_models (same `key`, since
    # ModelsConfig/BackendsConfig share role field names). routing.
    # allowed_models/per_model_daily_ceiling_usd are claude-specific
    # concepts the harness-backend PRD's model space never participates in
    # -- a role running on a non-claude backend must resolve its configured
    # model unconditionally at every layer below, exactly as the
    # pre-epsilon getattr-based resolution did (it never consulted an
    # allowlist at all).
    claude_backend = getattr(config.backends, key, 'claude') == 'claude'

    # Layer 3: static per-role config. Only `model` is fail-safe validated
    # (invariants 2/6) -- effort/budget_usd/max_turns apply unconditionally,
    # they carry no allowlist/ceiling concept.
    if hasattr(config.models, key):
        candidate = getattr(config.models, key)
        reason = (
            _model_rejection_reason(
                candidate, config, inputs.spend_by_model, inputs.scope_capacity,
            )
            if claude_backend else None
        )
        if reason is None:
            model = candidate
            source_layer = 'config'
        else:
            rejected.append(f'config:{reason}')
    if hasattr(config.budgets, key):
        budget_usd = getattr(config.budgets, key)
    if hasattr(config.max_turns, key):
        max_turns = getattr(config.max_turns, key)
    if hasattr(config.effort, key):
        effort = getattr(config.effort, key)

    # Layer 2: first matching policy rule (list order; first match wins).
    for rule in config.routing.rules:
        if not _rule_matches(rule, inputs):
            continue
        rule_id = rule.id
        if rule.set.model is not None:
            if rule.set.model.startswith('+'):
                candidate = _resolve_ladder_relative(rule.set.model, model, config.routing.ladder)
                if candidate is None:
                    rejected.append('policy_rule:model-not-in-ladder')
            else:
                candidate = rule.set.model
            if candidate is not None:
                reason = (
                    _model_rejection_reason(
                        candidate, config, inputs.spend_by_model, inputs.scope_capacity,
                    )
                    if claude_backend else None
                )
                if reason is None:
                    model = candidate
                    source_layer = 'policy_rule'
                else:
                    rejected.append(f'policy_rule:{reason}')
        if rule.set.effort is not None:
            effort = rule.set.effort
        if rule.set.budget_usd is not None:
            budget_usd = rule.set.budget_usd
        if rule.set.max_turns is not None:
            max_turns = rule.set.max_turns
        break

    # Layer 1: per-task metadata override (highest precedence; model only).
    # isinstance-guarded rather than a truthy check: a hand-edited or legacy
    # task can carry a non-dict model_overrides (stray string/list), and
    # this resolver must fail open to "no override" rather than raise
    # AttributeError out of TaskWorkflow._invoke -- mirrors
    # _task_simple_saturated's tolerant-degrade guard above.
    overrides = inputs.task_metadata.get('model_overrides') if inputs.task_metadata else None
    override_model = overrides.get(inputs.role_name) if isinstance(overrides, dict) else None
    if override_model is not None:
        reason = (
            _model_rejection_reason(
                override_model, config, inputs.spend_by_model, inputs.scope_capacity,
            )
            if claude_backend else None
        )
        if reason is None:
            model = override_model
            source_layer = 'metadata_override'
        else:
            rejected.append(f'metadata_override:{reason}')

    return RoutingDecision(
        model=model,
        effort=effort,
        budget_usd=budget_usd,
        max_turns=max_turns,
        source_layer=source_layer,
        rule_id=rule_id,
        rejected=tuple(rejected),
    )
