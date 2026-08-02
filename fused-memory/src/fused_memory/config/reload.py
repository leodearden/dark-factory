"""Green-tier config hot-reload engine for fused-memory.

Ports the orchestrator's proven config-hot-reload engine (``diff_config`` /
``apply_reload`` / ``_iter_leaves`` / ``_set_leaf`` / ``RELOADABLE_FIELDS`` in
``orchestrator/config.py``) retargeted to :class:`FusedMemoryConfig`, per task
2718 / ``plans/fused-memory-restart-survey-2026-07-17.md`` task τ (finding A3).

Reload-safety rule (why ``RELOADABLE_FIELDS`` is code-owned, not
operator-tunable): a knob is only genuinely reload-safe if EVERY consumer reads
it LIVE from the same shared config object the tool mutates in place. A knob
captured by value at construction (e.g. copied into a worker or an event buffer
at startup) would NOT observe an in-place mutation and must stay restart-only.
The allowlist is therefore a code property, seeded ONLY with leaves verified to
be read live from the single shared ``FusedMemoryConfig`` held by
``MemoryService.config`` (and, for ``reconciliation.*`` leaves, the same
instance held as ``ReconciliationHarness.config``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from fused_memory.config.schema import FusedMemoryConfig

# Code-owned allowlist of dotted leaf paths safe to hot-apply in place. Seeded
# minimally; each entry is verified read-live from the shared config object
# (see the reload-safety rule in the module docstring). Near-dup guard knobs
# are added in a later step, driven by their own live-consumer test.
RELOADABLE_FIELDS: frozenset[str] = frozenset({
    # Read live by the reconciliation reaper via
    # ``self.config.stale_run_recovery_seconds`` (ReconciliationHarness holds
    # the same reconciliation submodel instance as memory_service.config),
    # never captured by value — a reload is observed on the next reaper pass.
    'reconciliation.stale_run_recovery_seconds',
    # Read live per add_memory write by resolve_near_dup_guard_enabled
    # (server/near_duplicate_guard.py) via memory_service.config.reconciliation.*
    # — the resolver re-reads the shared config object on every call, so an
    # in-place reload flips the guard without a restart (see
    # TestBehaviorChangesWithoutRestart).
    'reconciliation.procedural_knowledge_near_dup_guard_enabled',
    # Read live per add_memory write by resolve_near_dup_threshold
    # (server/near_duplicate_guard.py), same live-read path as the guard flag
    # above — reload-safe for the same reason.
    'reconciliation.procedural_knowledge_near_dup_threshold',
    # Read live per add_memory write by resolve_topic_guard_clusters
    # (server/near_duplicate_guard.py) off the shared
    # memory_service.config.reconciliation object — the resolver re-reads the
    # list on every call, so an in-place reload adds/tunes topic clusters
    # without a restart. Same live-read reload-safety path as the two near-dup
    # knobs above; _iter_leaves treats the whole list[ProceduralTopicCluster]
    # as a single atomic leaf, so the clusters list reloads atomically (task 2845).
    'reconciliation.procedural_knowledge_topic_guard_clusters',
    # Write-triage band thresholds (task 3130). Written by
    # scripts/calibrate_write_triage.py --write-config, which derives them from
    # measured similarity distributions -- so hot-reload is what lets a
    # re-calibration take effect on a running server instead of waiting for a
    # restart. Read live off the shared memory_service.config.write_triage
    # object per add_memory write, the same live-read path as the near-dup
    # knobs above. None means UNCALIBRATED and triage fails open to `stored`,
    # so a reload that clears a threshold degrades safely.
    'write_triage.t_high',
    'write_triage.t_low',
    # Traceability pointer to the report that produced the two thresholds
    # above; reloaded alongside them so config never names a stale run.
    'write_triage.calibration_report_path',
    # Per-category cutoffs (task 3357), same green tier and same live-read
    # path as the pooled t_high above. _iter_leaves yields the dict WHOLE, so
    # the map reloads as ONE atomic leaf (same treatment as the
    # procedural_knowledge_topic_guard_clusters list): a half-applied set of
    # per-category cutoffs can never gate an audit_duplicate_memories sweep.
    # Replaced wholesale rather than merged, so a category the latest run
    # stopped deriving a cutoff for does not keep a stale one.
    'write_triage.t_high_by_category',
    # In-place update_memory authorization (task 3088). Read live per tool call
    # by resolve_mem0_update_enabled (server/mem0_update_authz.py) off the
    # shared memory_service.config.mem0_update object -- the resolver captures
    # nothing at import or construction, so an in-place reload flips the tool
    # off on the very next call. This is the one leaf that MUST be green-tier:
    # it is what an operator flips to stop an in-flight silent-rewrite
    # incident, and a restart-only kill switch is no kill switch.
    'mem0_update.enabled',
    # Read live per tool call by resolve_mem0_update_allowed_prefixes, same
    # live-read path as the kill switch above. The two lists are registered
    # SEPARATELY because they gate different arms: widening the metadata bar
    # alone is the supported way to admit an interactive curator-gate tagging
    # flow on a running server WITHOUT granting content-amend authority.
    'mem0_update.content_amend_allowed_agent_prefixes',
    'mem0_update.metadata_patch_allowed_agent_prefixes',
    # Storm-alarm knobs, read live off the shared config by
    # MemoryService.update_memory and passed INTO StormCounter.record() on every
    # call. That per-call argument is the non-obvious part: the shared
    # StormCounter (server/storm_counter.py) deliberately does NOT capture
    # threshold/window in __init__, because a value captured at construction
    # cannot observe an in-place mutation (see this module's reload-safety
    # rule) -- which would make these two leaves restart-only while sitting in
    # this allowlist as if they were green-tier.
    'mem0_update.storm_threshold',
    'mem0_update.storm_window_seconds',
})


# Defensive sentinel for a leaf structurally present on only ONE side of a diff.
# With annotation-based ``_iter_leaves`` (which compares nullable submodels
# whole), two same-class ``FusedMemoryConfig`` instances always yield identical
# leaf-path sets, so this is not reached in normal operation. It is retained as
# belt-and-suspenders: were ``_iter_leaves`` ever to regress to value-based
# descent (reintroducing the ``taskmaster``/``usage_cap`` None<->populated
# asymmetry), ``diff_config`` would degrade to bucketing the one-sided leaf as
# ``restart_required`` — rendered as the JSON-serializable string ``'<absent>'``
# so the report stays machine-checkable — rather than raising the KeyError of
# task 2718 review esc-2718-1.
_ABSENT = '<absent>'


@dataclass
class ConfigDiff:
    """Result of :func:`diff_config`: every differing leaf, bucketed by allowlist."""

    applied_candidates: dict[str, dict[str, Any]]
    restart_required: dict[str, dict[str, Any]]
    unchanged: int


def _iter_leaves(model: BaseModel):
    """Yield ``(dotted_path, value)`` for every leaf field of *model*.

    Descends exactly one level into a submodel field ONLY when its DECLARED
    annotation is a bare, non-Optional ``BaseModel`` subclass (e.g.
    ``reconciliation``, ``task_metadata``). The descent decision is driven by the
    annotation, NOT the runtime value — so an Optional/nullable submodel field
    (declared ``X | None``, e.g. ``taskmaster: TaskmasterConfig | None`` and
    ``usage_cap: UsageCapConfig | None``) is ALWAYS compared WHOLE as one atomic
    leaf, whether it is currently None or populated. Scalars, containers, and
    ``Optional[BaseModel]`` fields are all yielded whole.

    Anchoring descent to the annotation keeps ``_iter_leaves(live)`` and
    ``_iter_leaves(fresh)`` structurally identical regardless of an optional
    submodel's nullability state: a None<->populated toggle yields exactly ONE
    whole-object leaf on both sides, never a bare ``name`` on one side and
    descended ``name.<sub>`` sub-leaves on the other. That symmetry is what makes
    such a toggle diff to a single ``restart_required`` entry instead of the
    KeyError / half-missing-sub-path scatter of task 2718 review esc-2718-1.

    Values are read via ``__dict__`` (not ``getattr``) so a
    ``Field(deprecated=True)`` leaf does not fire a DeprecationWarning on every
    diff sweep — ``__dict__`` holds the same validated value pydantic already
    stored, so this only bypasses the deprecated-access warning wrapper, not
    validation.
    """
    fields = type(model).model_fields
    for name in fields:
        value = model.__dict__[name]
        annotation = fields[name].annotation
        # Descend only into a REQUIRED (bare, non-Optional) submodel. An
        # ``X | None`` field has annotation ``Optional[X]`` — a typing Union, not
        # a ``type`` — so it fails this guard and is yielded whole. That is the
        # fix for the nullable-submodel leaf-path-set divergence (esc-2718-1).
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            for sub in type(value).model_fields:
                yield f'{name}.{sub}', value.__dict__[sub]
        else:
            yield name, value


def diff_config(
    live: FusedMemoryConfig,
    fresh: FusedMemoryConfig,
    allowlist: frozenset[str] = RELOADABLE_FIELDS,
) -> ConfigDiff:
    """Structurally diff two fully-constructed FusedMemoryConfig instances.

    Every leaf where ``live != fresh`` is categorized into ``applied_candidates``
    (path in *allowlist*) or ``restart_required`` (otherwise); equal leaves are
    counted in ``unchanged``. Pure and synchronous — no I/O, no mutation of
    either argument.

    Iterates the UNION of both sides' leaf keys as defense-in-depth. In normal
    operation ``_iter_leaves`` (annotation-driven) yields identical leaf-path
    sets for two same-class configs — including for a nullable submodel toggled
    None<->populated, which is compared WHOLE and so lands as a SINGLE
    ``restart_required`` entry (neither the bare name nor its subfields are
    allowlisted). The union + ``_ABSENT`` fallback merely guarantees any residual
    one-sided leaf is still bucketed (never a ``KeyError``) if that symmetry were
    ever broken — see task 2718 review esc-2718-1.
    """
    live_leaves = dict(_iter_leaves(live))
    fresh_leaves = dict(_iter_leaves(fresh))
    applied_candidates: dict[str, dict[str, Any]] = {}
    restart_required: dict[str, dict[str, Any]] = {}
    unchanged = 0
    for path in live_leaves.keys() | fresh_leaves.keys():
        live_val = live_leaves.get(path, _ABSENT)
        fresh_val = fresh_leaves.get(path, _ABSENT)
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


def _set_leaf(model: FusedMemoryConfig, path: str, value: Any) -> None:
    """Write *value* to dotted *path* on *model*, bypassing per-write validation.

    A one-component path (top-level field) is written via ``object.__setattr__``.
    A two-component path (submodel leaf, e.g.
    ``reconciliation.stale_run_recovery_seconds``) is written via plain
    ``setattr`` on the EXISTING submodel object — mutating it in place to
    preserve submodel identity (I3), so held references (e.g.
    ``ReconciliationHarness.config``) observe the update. The single
    authoritative check is the post-apply whole-config re-validation in
    :func:`apply_reload`.
    """
    parts = path.split('.')
    if len(parts) == 1:
        object.__setattr__(model, parts[0], value)
    else:
        sub_name, leaf = parts
        setattr(getattr(model, sub_name), leaf, value)


def apply_reload(
    live: FusedMemoryConfig,
    fresh: FusedMemoryConfig,
    allowlist: frozenset[str] = RELOADABLE_FIELDS,
) -> dict[str, Any]:
    """Diff *live* against *fresh* and apply every allowlisted differing leaf to
    *live* in place.

    Hybrid re-validation: leaf-copies bypass per-write validation (see
    :func:`_set_leaf`), so after copying, the resulting *live* is re-validated as
    a whole via ``FusedMemoryConfig.model_validate(live.model_dump())``. Two
    individually-valid configs can still combine into an invalid hybrid (a field
    bound like ``gt=0``, or a cross-field ``model_validator``) — if that happens,
    every applied leaf is synchronously rolled back to its captured old value and
    the reload is reported failed. Nothing is left mutated on failure.

    Returns ``{reloaded, applied, restart_required, unchanged, error}``.
    ``config_path`` is intentionally omitted here; the MCP tool injects it.
    """
    d = diff_config(live, fresh, allowlist)
    applied: dict[str, dict[str, Any]] = {}
    try:
        for path, old_new in d.applied_candidates.items():
            _set_leaf(live, path, old_new['new'])
            applied[path] = old_new
        FusedMemoryConfig.model_validate(live.model_dump())
    except (ValidationError, ValueError) as exc:
        # Roll back every leaf applied so far (order-independent: each write
        # restores its own captured old value) so `live` is left exactly as it
        # was before this call, even on a mid-loop raise.
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
