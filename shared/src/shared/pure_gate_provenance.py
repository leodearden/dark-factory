"""``metadata.pure_gate_from``: the stamp naming what converted a submission into a deterministic pure gate.

Producer: :func:`pure_gate_stamped`, beside the shape predicate it stamps against.
Contract: docs/task-authoring.md §4. The stamp is written only when a conversion rewrote a submission's
routing. A submission already in the pure-gate shape (:func:`is_pure_gate_shape`) is never stamped,
whatever its execution_class (for example recon's consolidation gate). The stamp is history:
update_task's shallow metadata merge keeps it after a retype, so a reader pairs it with the current
task_kind/always_escalates.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

__all__ = ['PURE_GATE_FROM_KEY', 'PureGateTrigger', 'is_pure_gate_shape', 'pure_gate_stamped']

PURE_GATE_FROM_KEY = 'pure_gate_from'


class PureGateTrigger(StrEnum):
    """What converted a submission into a deterministic pure gate."""

    EXECUTION_CLASS_OPERATIONAL = 'execution_class=operational'
    EXECUTION_CLASS_DECISION = 'execution_class=decision'
    OPERATIONAL_ASK_REGISTRY = 'operational_ask_registry'


def is_pure_gate_shape(metadata: Mapping[str, object]) -> bool:
    """True iff metadata is deterministic, always escalates, and carries no before_done."""
    return (
        metadata.get('task_kind') == 'deterministic'
        and metadata.get('always_escalates') is True
        and not metadata.get('before_done')
    )


def pure_gate_stamped(metadata: Mapping[str, object], trigger: PureGateTrigger) -> dict[str, object]:
    """A new dict: *metadata* rewritten into the pure-gate shape, stamped with *trigger* if that rewrote it.

    Sets ``task_kind='deterministic'`` and ``always_escalates=True`` and drops
    ``before_done``, whatever *metadata* carried, so the result always passes
    :func:`is_pure_gate_shape`. *metadata* itself is never mutated.
    """
    meta = dict(metadata)
    declared = is_pure_gate_shape(metadata)
    meta['task_kind'] = 'deterministic'
    meta['always_escalates'] = True
    meta.pop('before_done', None)
    if not declared:
        meta[PURE_GATE_FROM_KEY] = trigger.value
    return meta
