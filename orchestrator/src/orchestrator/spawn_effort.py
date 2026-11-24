"""orchestrator.spawn_effort — the reasoning effort a /spawn session runs at.

The closed level set is the one ``claude --effort`` accepts; on the T3 lane a
level travels as the ``effort`` option inside the thread's ``modelSelection``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Any


class EffortLevel(StrEnum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    XHIGH = 'xhigh'
    MAX = 'max'


_LEVEL_RANK = {level.value: rank for rank, level in enumerate(EffortLevel)}


def in_level_order(levels: Iterable[str]) -> list[str]:
    """*levels* from low to max, any outside the closed set last."""
    return sorted(levels, key=lambda level: (_LEVEL_RANK.get(level, len(_LEVEL_RANK)), level))


def effort_option(level: EffortLevel) -> dict[str, str]:
    """*level* as one entry of a T3 ``modelSelection.options`` list."""
    return {'id': 'effort', 'value': level.value}


def _effort_entry(options: object) -> Mapping[str, Any] | None:
    entries = options if isinstance(options, list) else []
    return next((entry for entry in entries if isinstance(entry, Mapping) and entry.get('id') == 'effort'), None)


def selection_effort(model_selection: object) -> str | None:
    """The effort a T3 ``modelSelection`` carries in its ``options``, or None when it names none."""
    entry = _effort_entry(model_selection.get('options') if isinstance(model_selection, Mapping) else None)
    value = entry.get('value') if entry is not None else None
    return value if isinstance(value, str) else None


def offered_levels(capabilities: object, instance_id: str, model: str) -> frozenset[str] | None:
    """The effort values ``orchestrator_capabilities`` lists for *model* on *instance_id*, or None when it lists none.

    *model* is matched by id, else it is the one model listed there: the
    single match of a search for an alias.
    """
    models = _listed_models(capabilities, instance_id)
    matched = [listed for listed in models if listed.get('id') == model] or (models if len(models) == 1 else [])
    if len(matched) != 1:
        return None
    entry = _effort_entry(matched[0].get('options'))
    values = entry.get('values') if entry is not None else None
    if not isinstance(values, list):
        return None
    return frozenset(value for value in values if isinstance(value, str)) or None


def _listed_models(capabilities: object, instance_id: str) -> list[Mapping[str, Any]]:
    providers = capabilities.get('providers') if isinstance(capabilities, Mapping) else None
    for provider in providers if isinstance(providers, list) else []:
        if isinstance(provider, Mapping) and provider.get('providerInstanceId') == instance_id:
            models = provider.get('models')
            return [listed for listed in models if isinstance(listed, Mapping)] if isinstance(models, list) else []
    return []
