"""orchestrator.spawn_effort — the reasoning effort a /spawn session runs at.

The closed level set is the one ``claude --effort`` accepts; on the T3 lane a
level travels as the ``effort`` option inside the thread's ``modelSelection``.
"""

from __future__ import annotations

from collections.abc import Mapping
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


def selection_effort(model_selection: object) -> str | None:
    """The effort a T3 ``modelSelection`` carries in its ``options``, or None when it names none."""
    options = model_selection.get('options') if isinstance(model_selection, Mapping) else None
    if not isinstance(options, list):
        return None
    for option in options:
        if isinstance(option, Mapping) and option.get('id') == 'effort':
            value = option.get('value')
            return value if isinstance(value, str) else None
    return None
