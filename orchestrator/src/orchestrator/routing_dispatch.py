"""Shared out-of-band routing helper (task η, plans/adaptive-model-routing-prd.md).

Five LLM dispatch sites resolve their route OUTSIDE ``TaskWorkflow._invoke``:
the steward main invoke and its inner triage (``steward.py``), the deep
reviewer (``review_checkpoint.py``), the module tagger (``harness.py``), and
the autonomous dry-run unblock hook (``dry_run_unblock.py``). Task ε adopted
``orchestrator.routing.resolve_route`` inside ``_invoke``; task γ gave
``_invoke`` a ``routing_decision`` event + ``metadata.routing`` mirror
(``workflow.py::_record_routing_decision``). This module gives the five
out-of-band sites the SAME behaviour through one shared async helper
(``resolve_and_record_route``) rather than N copied emit/mirror blocks
(design-invariant: no-lockstep-duplication).

Why a new module (not ``routing.py`` or ``workflow.py``):

* ``routing.py`` is deliberately import-light, pure and synchronous
  (``resolve_route`` never does I/O). This helper is async and emits events /
  writes the scheduler / reads a cost_store — it would pollute that character.
* ``workflow.py`` owns ``_invoke``/``_record_routing_decision`` but is outside
  this task's locked module scope (task ε owns it), and those are
  ``TaskWorkflow`` methods (bound to ``self.task``/``self.scheduler``/an
  ``AgentRole``) — not callable from the out-of-band classes, which have no
  ``AgentRole`` and heterogeneous per-site context.

``resolve_and_record_route`` replicates ``_record_routing_decision``'s three
independent failure boundaries so telemetry never crashes a caller: the
decision-record build is wrapped in try/except (a bad mock/field swallows the
whole record), the event is guarded on ``event_store``, and the scheduler
mirror is guarded on ``scheduler``+``task_id`` in its own try/except. The
resolver-owned axis is (model, effort, budget_usd, max_turns) ONLY — backend
stays each site's own ``config.backends.*`` / ``ua_cfg.backend`` read (PRD
boundary under harness-backend-reconnect-pi).

Backend NOTE: ``_invoke`` keeps its own inline resolve+record copy for now
(``workflow.py`` is locked out of this task); a documented follow-up can
dedupe it onto this helper.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from shared.task_metadata import RoutingDecisionMirror, RoutingState

from orchestrator.event_store import EventType
from orchestrator.routing import RouteInputs, resolve_route

if TYPE_CHECKING:
    from collections.abc import Mapping

    from orchestrator.config import OrchestratorConfig
    from orchestrator.routing import PlanShape, RoleDefaults, RoutingDecision

logger = logging.getLogger(__name__)


def _trailing_24h_window() -> tuple[str, str]:
    """Return ``(start_iso, end_iso)`` for the trailing-24h window ending now.

    Mirrors ``workflow.py::_trailing_24h_window`` (that module is locked out of
    this task, so this helper carries its own copy rather than importing the
    heavy ``workflow`` module just for a two-line datetime helper).
    """
    now = datetime.now(UTC)
    return (now - timedelta(hours=24)).isoformat(), now.isoformat()


async def _ceiling_spend_by_model(
    config: OrchestratorConfig,
    cost_store: Any | None,
) -> dict[str, float]:
    """Trailing-24h USD spend per model carrying a configured
    ``routing.per_model_daily_ceiling_usd`` entry (PRD invariant 6).

    Queried ONLY for ceiling'd models, and only when a cost_store is wired —
    empty ``{}`` at stock config (no ceilings configured), so zero cost_store
    reads fire and byte-equivalence (invariant 3) holds. Best-effort per model:
    a query failure for one model is logged and that model is omitted from the
    result (``resolve_route`` then defaults its spend to 0.0 — fail-open; a
    dispatch is never blocked by a cost-query hiccup). Mirrors
    ``workflow.py::TaskWorkflow._ceiling_spend_by_model``.
    """
    ceilings = config.routing.per_model_daily_ceiling_usd
    if not ceilings or not cost_store:
        return {}
    start_iso, end_iso = _trailing_24h_window()
    spend: dict[str, float] = {}
    for model_name in ceilings:
        try:
            spend[model_name] = await cost_store.model_cost_in_window(
                model_name, start_iso, end_iso,
            )
        except Exception:
            logger.warning(
                'routing_dispatch: failed to fetch trailing-24h spend for model %s',
                model_name, exc_info=True,
            )
    return spend


def _out_of_band_inputs_digest(
    role_name: str,
    task_id: str,
    plan_step_count: int,
    modules: list[str],
    routing_tier: int,
) -> str:
    """Stable sha256[:16] digest of the salient routing-resolution inputs (PRD γ).

    Mirrors ``workflow.py::_routing_inputs_digest`` but keyed on ``role_name``
    (a plain string) rather than an ``AgentRole`` — the out-of-band sites have
    no ``AgentRole``/plan object. Pure and deterministic: telemetry consumers
    can compare/dedupe resolutions without the full payload.
    """
    payload = {
        'role': role_name,
        'task_id': str(task_id),
        'plan_step_count': plan_step_count,
        'module_count': len(modules),
        'modules': sorted(modules),
        'routing_tier': routing_tier,
    }
    basis = json.dumps(payload, sort_keys=True).encode('utf-8')
    return hashlib.sha256(basis).hexdigest()[:16]


async def resolve_and_record_route(
    *,
    role_name: str,
    role_defaults: RoleDefaults,
    config: OrchestratorConfig,
    task_id: str | None = None,
    task_metadata: Mapping[str, Any] | None = None,
    plan_shape: PlanShape | None = None,
    modules: list[str] | None = None,
    event_store: Any | None = None,
    scheduler: Any | None = None,
    in_memory_task: dict | None = None,
    cost_store: Any | None = None,
) -> RoutingDecision:
    """Resolve one out-of-band invocation's route and record the decision.

    Builds a ``RouteInputs`` (with best-effort ceiling spend), calls
    ``resolve_route``, then — best-effort, never raising — emits one
    ``routing_decision`` event and mirrors ``metadata.routing``. Returns the
    resolved ``RoutingDecision`` so the caller feeds
    ``decision.model``/``effort``/``budget_usd``/``max_turns`` into its own
    ``invoke_with_cap_retry`` call (backend stays the caller's own read).

    Parameters
    ----------
    role_name:
        The dispatch role name (``'steward'``/``'triage'``/``'deep_reviewer'``/
        ``'module_tagger'``/``'unblock_auto'``) — resolve_route's layer key and
        the ``model_overrides`` lookup key.
    role_defaults:
        The layer-4, always-available base (``RoleDefaults``) — from the site's
        ``AgentRole`` dataclass defaults, or (module_tagger/unblock_auto) from
        the site's dedicated config block.
    config:
        The ``OrchestratorConfig`` snapshot resolve_route reads.
    task_id / task_metadata:
        Present for the genuinely per-task sites (steward, unblock_auto);
        ``task_metadata`` carries ``model_overrides``/``routing`` for the
        override + tier layers. ``None`` (project-level deep_reviewer, batch
        module_tagger) degrades to empty metadata.
    plan_shape / modules:
        Optional plan context (the out-of-band sites generally have neither);
        fed to policy-rule plan conditions + the inputs digest.
    event_store:
        When set, one ``routing_decision`` event is emitted.
    scheduler / in_memory_task:
        ``metadata.routing`` mirror sinks — persisted via
        ``scheduler.update_task(metadata_mode='merge')`` when a scheduler+task_id
        are given; mutated in-memory on ``in_memory_task`` when given.
    cost_store:
        Trailing-24h ceiling-spend source (only read when ceilings configured).
    """
    md: Mapping[str, Any] = task_metadata or {}
    routing_state = RoutingState.from_metadata(dict(md))
    spend_by_model = await _ceiling_spend_by_model(config, cost_store)
    route_inputs = RouteInputs(
        role_name=role_name,
        task_id=str(task_id) if task_id is not None else '',
        task_metadata=md,
        plan_shape=plan_shape,
        routing_tier=routing_state.routing_tier,
        dispatch_count=int(md.get('dispatch_count') or 0),
        role_defaults=role_defaults,
        spend_by_model=spend_by_model,
    )
    decision = resolve_route(route_inputs, config)

    # Everything below is best-effort telemetry — it must NEVER change or
    # prevent the returned decision (the invocation still runs on it). The
    # record build is wrapped so a strict-field/mock failure swallows the whole
    # record (mirroring workflow._record_routing_decision's boundary 1).
    try:
        digest = _out_of_band_inputs_digest(
            role_name,
            route_inputs.task_id,
            plan_shape.step_count if plan_shape is not None else 0,
            modules or [],
            routing_state.routing_tier,
        )
        mirror = RoutingDecisionMirror(
            role=role_name,
            model=decision.model,
            effort=decision.effort,
            budget_usd=decision.budget_usd,
            max_turns=decision.max_turns,
            source_layer=decision.source_layer,
            rule_id=decision.rule_id,
            rejected=list(decision.rejected),
            routing_tier=routing_state.routing_tier,
            decided_at=datetime.now(UTC).isoformat(),
        )
    except Exception:
        logger.warning(
            'routing_dispatch: failed to build routing decision record (role=%s)',
            role_name, exc_info=True,
        )
        return decision

    if event_store is not None:
        # Derived from `mirror` (not hand-duplicated) so the event payload and
        # the metadata.routing mirror below are two serializations of the one
        # record + the event-only `inputs_digest`.
        event_store.emit(
            EventType.routing_decision,
            task_id=task_id,
            role=role_name,
            data={**mirror.model_dump(), 'inputs_digest': digest},
        )

    new_state_dump = routing_state.with_decision(mirror).model_dump()
    # In-memory mirror is unconditional (regardless of the scheduler write's
    # outcome) so successive calls within one dispatch accumulate history
    # without a round-trip read.
    if in_memory_task is not None:
        in_memory_task.setdefault('metadata', {})['routing'] = new_state_dump
    # Persisted mirror only when this site is genuinely per-task (has BOTH a
    # scheduler and a task_id): the batch module_tagger passes a scheduler but
    # task_id=None, and the steward has a task_id but no scheduler.
    if scheduler is not None and task_id is not None:
        try:
            await scheduler.update_task(
                task_id,
                {'routing': new_state_dump},
                metadata_mode='merge',
            )
        except Exception:
            logger.warning(
                'routing_dispatch: failed to mirror routing decision onto '
                'metadata (role=%s, task=%s)',
                role_name, task_id, exc_info=True,
            )

    return decision
