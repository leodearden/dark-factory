"""The one rule for echoing a caller's value in a JSON refusal payload."""

from __future__ import annotations

import json

__all__ = ['json_safe']


def json_safe(value: object) -> object:
    """`value` when json.dumps accepts it, else its repr, so the payload always serialises."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value
