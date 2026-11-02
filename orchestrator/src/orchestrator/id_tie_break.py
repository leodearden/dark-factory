"""The id tie-break that dispatch order and park order both end in."""

from __future__ import annotations

import math


def id_tie_break(task_id: str) -> tuple[float, str]:
    """THE id tie-break (PRD I-4): older numeric ids first, then non-numeric ids in string order.

    Dispatch order (``orchestrator/src/orchestrator/scheduler.py::Scheduler._score_candidates``)
    and park order (``orchestrator/src/orchestrator/lock_table.py::_ParkKey``) both
    end in it, so they cannot diverge.
    """
    return (float(int(task_id)) if task_id.isdecimal() else math.inf, task_id)
