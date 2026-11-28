"""The supersession record of a plan design decision, and the invariants a supersession must satisfy."""
from __future__ import annotations

__all__ = [
    'SUPERSEDED_VERSIONS_KEY',
    'InvalidSupersession',
    'supersede_decision',
]

SUPERSEDED_VERSIONS_KEY = 'superseded_versions'


class InvalidSupersession(ValueError):
    def __init__(self, field: str, value: object, requirement: str) -> None:
        self.field = field
        self.value = value
        super().__init__(f'design decision supersession field {field!r} {requirement}; got {value!r}')


def _non_blank_str(field: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidSupersession(field, value, 'must be a non-blank string')
    return value


def _stored_str(field: str, value: object) -> str:
    if not isinstance(value, str):
        raise InvalidSupersession(field, value, 'must be a stored string')
    return value


def _decision_list(decisions: object, index: object) -> list:
    if not isinstance(decisions, list):
        raise InvalidSupersession(
            'decision_index', index, 'must address a recorded decision, but the plan records no design decision list',
        )
    return decisions


def _in_range_index(index: object, count: int) -> int:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count:
        raise InvalidSupersession(
            'decision_index', index, f'must be an int in range(0, {count}): the plan records {count} design decisions',
        )
    return index


def _stored_entry(field: str, entry: object) -> dict:
    if not isinstance(entry, dict):
        raise InvalidSupersession(field, entry, 'must be a stored decision mapping')
    return entry


def supersede_decision(
    decisions: object, index: object, *, decision: str, rationale: str, reason: str,
) -> list[dict]:
    """*decisions* with entry *index* restated as the new pair, its prior pair and *reason* appended to its history.

    The input list and its entries are never mutated.
    """
    decision = _non_blank_str('decision', decision)
    rationale = _non_blank_str('rationale', rationale)
    reason = _non_blank_str('reason', reason)
    current = _decision_list(decisions, index)
    position = _in_range_index(index, len(current))
    where = f'design_decisions[{position}]'
    entry = _stored_entry(where, current[position])
    old_decision = _stored_str(f'{where}.decision', entry.get('decision'))
    old_rationale = _stored_str(f'{where}.rationale', entry.get('rationale'))
    old_versions = entry.get(SUPERSEDED_VERSIONS_KEY, [])
    if not isinstance(old_versions, list):
        raise InvalidSupersession(f'{where}.{SUPERSEDED_VERSIONS_KEY}', old_versions, 'must be a list')
    if (decision, rationale) == (old_decision, old_rationale):
        raise InvalidSupersession(
            'decision', decision, 'restates the current decision and rationale, so there is nothing to supersede',
        )
    superseded = {'decision': old_decision, 'rationale': old_rationale, 'reason': reason}
    updated = [*current]
    updated[position] = {
        **entry,
        'decision': decision,
        'rationale': rationale,
        SUPERSEDED_VERSIONS_KEY: [*old_versions, superseded],
    }
    return updated
