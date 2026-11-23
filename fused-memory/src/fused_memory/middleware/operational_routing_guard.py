"""Operational routing guard (β).

A deterministic, pure submit-boundary transform that maps the *declared*
``(execution_class, operational_mode)`` to routing metadata, making the
operational→deterministic coercion an **unbypassable boundary invariant**
(PRD ``plans/operational-ask-routing-prd.md``, task β). This deliberately
reverses the "declaration-only, no boundary coercion" scoping of tasks
2085/2563 (owner-greenlit 2026-07-19): the authoritative coercion moves out
of the fail-open curator and onto this boundary inject, which runs on every
submit path (normal, planning_mode, direct interceptor).

## Contract table (first match wins)

| execution_class    | operational_mode   | result                                    |
|--------------------|--------------------|-------------------------------------------|
| ``operational``    | ``gate`` / absent  | pure gate                                 |
| ``decision``       | (any)              | pure gate                                 |
| ``operational``    | ``llm``            | pure gate PLUS the x_ llm-gate marker     |
| ``code_tdd``       | (any)              | untouched (original object returned)      |
| absent / anything else | (any)          | untouched                                 |

"pure gate" = ``task_kind='deterministic'``, ``always_escalates=True``,
``before_done`` dropped — the shape enforced by
``deterministic_task_guard.deterministic_task_error`` (a deterministic task
with no action must always escalate, never be an ill-formed no-op) — plus
``pure_gate_from`` naming the triggering execution_class when the
submission was not already in that shape. The stamp itself is delegated verbatim to
``shared/src/shared/pure_gate_provenance.py::pure_gate_stamped`` (INV-5 — the
pure-gate stamp is reused, never re-implemented).

Only exact ``operational_mode == 'llm'`` takes the llm sub-branch;
``operational`` with ``gate``/absent/``None``/any-other mode takes the plain
pure-gate branch (the conservative default). This transform does NOT reject
invalid ``operational_mode`` values — that rule is owned by the shared
``TaskMetadata`` ``Literal`` + ``parse_metadata`` at the backend write
boundary (task ζ integration matrix), not β.

## Guarantees

- **Idempotent** — re-running on already-coerced metadata yields a
  by-value-identical result, so the tools.py→interceptor double-application
  is a no-op.
- **Override-safe (this transform in isolation)** — a hand-supplied
  ``task_kind='normal'`` / ``before_done`` on an ``operational``|``decision``
  task is overwritten to the pure-gate shape *by this transform*. This
  guarantee is scoped to the transform itself, **not** the full ``submit_task``
  tool boundary. At that boundary the deterministic-task guard
  (``deterministic_task_guard.deterministic_task_error``, invariants 3/3b)
  runs *before* this coercion and hard-rejects an operational submission that
  pre-declares ``before_done`` (not ``None``) or ``always_escalates=True``
  under the default ``task_kind='normal'`` parameter — such a submission is
  rejected, not coerced. The common operational path (neither field
  pre-declared) is coerced as the contract table describes.
- **Pure** — no I/O; the caller's object is never mutated on the coercion
  branch (the reused stamp shallow-copies), and the *original* object is
  returned verbatim on the non-coercion branch (a byte-identical no-op for
  every ``code_tdd``/absent/``None``/``str`` submission).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from shared.pure_gate_provenance import (
    PURE_GATE_FROM_KEY,
    PureGateTrigger,
    is_pure_gate_shape,
    pure_gate_stamped,
)

from fused_memory.middleware.metadata_dict import raw_metadata_dict

__all__ = [
    'OPERATIONAL_LLM_GATE_MARKER_KEY',
    'inject_operational_routing',
    'pure_gate_routing_notice',
]

#: Dedicated, machine-checkable marker key stamped True only for
#: execution_class='operational' + operational_mode='llm'. The ``x_`` prefix
#: is census-clean (shared/task_metadata.py exempts ``x_``-prefixed keys from
#: the unknown_key schema-warning census), so no _BLESSED_METADATA_KEYS edit is
#: needed. Task γ/2803 consumes it (or the preserved operational_mode='llm').
OPERATIONAL_LLM_GATE_MARKER_KEY = 'x_operational_llm_gate'

_TRIGGER_BY_EXECUTION_CLASS: Mapping[str, PureGateTrigger] = MappingProxyType(
    {
        'operational': PureGateTrigger.EXECUTION_CLASS_OPERATIONAL,
        'decision': PureGateTrigger.EXECUTION_CLASS_DECISION,
    }
)


def inject_operational_routing(metadata: str | dict[str, Any] | None) -> Any:
    """Coerce a declared operational/decision submission to a deterministic pure gate.

    See the module docstring for the full contract table and guarantees.

    On the coercion branch (``execution_class`` ∈ {``operational``,
    ``decision``}) returns a fresh coerced dict — the caller's object is not
    mutated. On the non-coercion branch (``code_tdd`` / absent / anything
    else) returns the *original* ``metadata`` object verbatim (``None`` stays
    ``None``, a ``str`` stays a ``str``, a dict is the same object), so the
    transform is a byte-identical no-op for every non-operational submission.

    Args:
        metadata: Incoming task metadata (dict, JSON string, or None).
    """
    parsed = raw_metadata_dict(metadata, source='operational_routing_guard')
    execution_class = parsed.get('execution_class')
    if not isinstance(execution_class, str):
        return metadata
    trigger = _TRIGGER_BY_EXECUTION_CLASS.get(execution_class)
    if trigger is None:
        # code_tdd / absent / anything not operational|decision → untouched.
        return metadata

    coerced = pure_gate_stamped(parsed, trigger)
    if execution_class == 'operational' and parsed.get('operational_mode') == 'llm':
        # Dedicated llm-gate marker (task γ/2803 consumes it). operational_mode
        # itself is ALSO preserved by the pure-gate stamp, so γ may read either.
        coerced[OPERATIONAL_LLM_GATE_MARKER_KEY] = True
    return coerced


def pure_gate_routing_notice(metadata: object) -> dict[str, Any]:
    """The submit-response notice for a submission converted into a pure gate, else ``{}``.

    Derived from the stored stamp, so the notice and the task record cannot
    disagree. A stamp on metadata that is not in the pure-gate shape gets no
    notice: it would claim a human gate that does not exist.
    """
    if not isinstance(metadata, Mapping):
        return {}
    if not is_pure_gate_shape(metadata):
        return {}
    try:
        trigger = PureGateTrigger(metadata.get(PURE_GATE_FROM_KEY))
    except ValueError:
        return {}
    return {
        'routing_notice': {
            'pure_gate_from': trigger.value,
            'detail': (
                f'This submission was converted into a deterministic always-escalates '
                f'pure gate because of {trigger.value}: a task created from it escalates '
                f'to a HUMAN at dispatch, and no agent executes it.'
            ),
            'hint': (
                "If an agent can execute this work (including a docs, comment or rename "
                "edit), file it with task_kind='normal' and metadata.complexity='simple' "
                'instead, without an operational/decision execution_class '
                '(docs/task-authoring.md §4).'
            ),
        },
    }
