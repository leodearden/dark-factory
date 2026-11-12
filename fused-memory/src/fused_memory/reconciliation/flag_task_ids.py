"""Decompose the task_id value a reconciliation finding carries into its component task ids.

Both flag-side consumers — ``flag_dedup._flag_candidate_task_ids`` and
``preservation_specimen_guard._flag_task_ids`` — decompose through this one
function, so a new separator or value shape reaches both.
"""

from __future__ import annotations

from typing import Any

__all__ = ['task_id_components']


def task_id_components(raw: Any) -> tuple[str, ...]:
    """Return the component task ids of one finding ``task_id`` value.

    - Accepted shapes: a ``str``, or an ``int`` straight off a task dict, which
      is stringified (``3105`` -> ``'3105'``).  Anything else — ``None``, a
      ``bool`` (an ``int`` subclass, never a task id), a ``float``, ``bytes``, a
      list, a dict — yields ``()``, so a malformed value is never stringified
      into a lookup.
    - Split on ``','``, the composite shape (``'3105,4223'``); each component
      stripped, blank components dropped, so ``','`` yields ``()``.
    - Deduped, keeping each component's FIRST position.
    - No usability judgement: ``'0'`` and ``'-5'`` come back as components.
      Whether a component could name a real task is the consumer's call.

    Total over malformed LLM-authored input.  Pure, sync, no I/O.
    """
    if isinstance(raw, bool) or not isinstance(raw, (str, int)):
        return ()
    return tuple(dict.fromkeys(p.strip() for p in str(raw).split(',') if p.strip()))
