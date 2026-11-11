"""``metadata.pure_gate_from``: the stamp naming what converted a submission into a deterministic pure gate.

Producer: fused-memory/src/fused_memory/middleware/task_interceptor.py::TaskInterceptor._inject_deterministic_pure_gate.
Contract: docs/task-authoring.md §4. A hand-authored pure gate never carries the stamp, so its presence is
how a reader tells a conversion from an intended gate.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ['PURE_GATE_FROM_KEY', 'PureGateTrigger']

PURE_GATE_FROM_KEY = 'pure_gate_from'


class PureGateTrigger(StrEnum):
    """What converted a submission into a deterministic pure gate."""

    EXECUTION_CLASS_OPERATIONAL = 'execution_class=operational'
    EXECUTION_CLASS_DECISION = 'execution_class=decision'
    OPERATIONAL_ASK_REGISTRY = 'operational_ask_registry'
