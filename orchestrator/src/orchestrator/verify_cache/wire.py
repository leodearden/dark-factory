"""Strict decoding of the verify cache's wire values: exact keys, and typed fields whose errors name the field."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from orchestrator.verify_spec_shared import require_exact_keys

_T = TypeVar('_T')


def wire_fields(raw: object, expected: frozenset[str], where: str) -> Mapping[str, object]:
    """*raw* as a JSON object whose keys are exactly *expected*; TypeError naming *where* otherwise."""
    require_exact_keys(raw, expected, where)
    if not isinstance(raw, dict):
        raise TypeError(f'{where} must be a JSON object, got {type(raw).__name__}')
    return raw


def typed_field(fields: Mapping[str, object], name: str, kind: type[_T], where: str) -> _T:
    """*fields*[*name*] when it is a *kind* (a bool is never an int); TypeError naming the field otherwise."""
    value = fields[name]
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise TypeError(f'{where}.{name} must be {kind.__name__}, got {type(value).__name__} {value!r}')
    return value


def optional_field(fields: Mapping[str, object], name: str, kind: type[_T], where: str) -> _T | None:
    return None if fields[name] is None else typed_field(fields, name, kind, where)
