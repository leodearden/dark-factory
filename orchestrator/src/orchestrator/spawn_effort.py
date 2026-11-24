"""orchestrator.spawn_effort — the reasoning effort a /spawn session runs at.

The closed level set is the one ``claude --effort`` accepts; on the T3 lane a
level travels as the ``effort`` option inside the thread's ``modelSelection``.
"""

from __future__ import annotations

from enum import StrEnum


class EffortLevel(StrEnum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    XHIGH = 'xhigh'
    MAX = 'max'


def effort_option(level: EffortLevel) -> dict[str, str]:
    """*level* as one entry of a T3 ``modelSelection.options`` list."""
    return {'id': 'effort', 'value': level.value}
