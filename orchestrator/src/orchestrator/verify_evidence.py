"""What a verify verdict attests: the identity of the plan it ran and the outcome
of each unit, as typed values and as event fields."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from orchestrator.verify_spec_shared import canonical_json

_PROSE_RUN_KEYS = frozenset({'reason'})


def plan_hash(plan: Mapping[str, Any] | None) -> str | None:
    """sha256 of the plan's rendered runs minus their prose; None when no plan ran.

    Keys are denied rather than allowed, so a new execution-relevant run field
    can only make two hashes more distinct, never collide them.
    """
    if plan is None:
        return None
    runs = [
        {key: value for key, value in run.items() if key not in _PROSE_RUN_KEYS}
        for run in plan['runs']
    ]
    return hashlib.sha256(canonical_json(runs).encode()).hexdigest()
