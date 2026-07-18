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

from pydantic import BaseModel

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
})


@dataclass
class ConfigDiff:
    """Result of :func:`diff_config`: every differing leaf, bucketed by allowlist."""

    applied_candidates: dict[str, dict[str, Any]]
    restart_required: dict[str, dict[str, Any]]
    unchanged: int


def _iter_leaves(model: BaseModel):
    """Yield ``(dotted_path, value)`` for every leaf field of *model*.

    Descends exactly one level into BaseModel-valued fields (e.g.
    ``reconciliation``, ``task_metadata``); dict/list/set/None-valued fields are
    yielded whole as atomic leaves compared by equality. Values are read via
    ``__dict__`` (not ``getattr``) so a ``Field(deprecated=True)`` leaf does not
    fire a DeprecationWarning on every diff sweep — ``__dict__`` holds the same
    validated value pydantic already stored, so this only bypasses the
    deprecated-access warning wrapper, not validation.
    """
    for name in type(model).model_fields:
        value = model.__dict__[name]
        if isinstance(value, BaseModel):
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
