"""THE definition of a task author declaring the SIMPLE_TASK fast path.

Shared by the dispatch router and the fused-memory submit gate so the two
cannot disagree about which ``metadata.complexity`` values mean "simple".
"""
from __future__ import annotations

__all__ = ['is_simple_complexity']

_SIMPLE = 'simple'


def is_simple_complexity(value: object) -> bool:
    """True iff *value* normalises to ``'simple'`` (case-insensitive, stripped)."""
    return str(value or '').strip().lower() == _SIMPLE
