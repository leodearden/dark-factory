"""A timed gate is a deterministic predicate or pure gate that carries a milestone.

It needs no agent, worktree or module lock, so once its milestone opens the
scheduler dispatches it ahead of the tier-scored queue (task 11330).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from shared.pure_gate_provenance import is_pure_gate_shape

from orchestrator.id_tie_break import id_tie_break

__all__ = ['is_timed_gate', 'timed_gates']


def is_timed_gate(task: Mapping[str, Any]) -> bool:
    """True iff *task* is a deterministic predicate or pure gate with a milestone; a deploy never is."""
    metadata = task.get('metadata')
    if not isinstance(metadata, Mapping):
        return False
    if metadata.get('task_kind') != 'deterministic' or not metadata.get('milestone'):
        return False
    return is_pure_gate_shape(metadata) or _is_predicate(metadata.get('before_done'))


def timed_gates(tasks: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The timed gates among *tasks*, oldest id first."""
    return sorted(
        (task for task in tasks if is_timed_gate(task)),
        key=lambda task: id_tie_break(str(task.get('id', ''))),
    )


def _is_predicate(before_done: object) -> bool:
    """A before_done with no kind is a deploy (shared.task_metadata.BeforeDone's default), so only an explicit predicate counts."""
    return isinstance(before_done, Mapping) and before_done.get('kind') == 'predicate'
