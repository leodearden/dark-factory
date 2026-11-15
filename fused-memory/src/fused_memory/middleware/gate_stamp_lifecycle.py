"""Which gate-episode stamps a status write removes (task 3638).

``gate_escalated_at`` and ``gate_escalated_preconditions`` describe a
deterministic task's OPEN gate episode, filed by a pure gate, an act-then-ask
deploy or a predicate; only a pure gate's filing writes the preconditions
stamp. A terminal crossing ends the episode whatever the task's kind.
Entering done or cancelled covers every close, runner or manual; leaving it
neutralises a legacy stale stamp before
``fused_memory/reconciliation/stage1_stall_detector.py::extract_stalled_gate_backlog_task_ids``
can select the reopened row.
"""

from __future__ import annotations

from shared.task_metadata import GATE_EPISODE_STAMP_KEYS
from shared.task_metadata_wire import coerce_task_metadata
from shared.task_statuses import TERMINAL

__all__ = ['terminal_crossing_clear_keys']


def terminal_crossing_clear_keys(
    *, old_status: str, new_status: str, stored_metadata: object,
) -> frozenset[str]:
    """The gate-episode stamps *stored_metadata* carries, when this write crosses TERMINAL."""
    crosses_terminal = old_status != new_status and (
        old_status in TERMINAL or new_status in TERMINAL
    )
    metadata = coerce_task_metadata(stored_metadata)
    if not crosses_terminal or not metadata:
        return frozenset()
    return GATE_EPISODE_STAMP_KEYS.intersection(metadata)
