"""scripts/legibility/state_docs.py — read and write a legibility state
document as a typed value.

A read returns one of three distinct answers: no document was ever written
(:class:`Missing`), a document exists but cannot be trusted
(:class:`Malformed`, carrying why), or it holds a valid value (:class:`Ok`).
"Never written" and "unreadable" stay apart because a reader that merges them
cannot tell a fresh project from a broken one (INV-13). A codec turns the
parsed JSON into its value using the field readers below, which is the one
place a document's fields are validated. Only the stdlib and
``shared.safe_io`` are imported: the bare-``python3`` predicate path
imports this module.
"""
from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Generic, Protocol, TypeVar

# Bind `shared` to the SAME checkout as this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'shared' / 'src'))

from shared.safe_io import atomic_write_text, load_json_or_warn

logger = logging.getLogger(__name__)

T = TypeVar('T')
D = TypeVar('D')
E = TypeVar('E', bound=Enum)


@dataclass(frozen=True)
class Missing:
    """No document has been written at ``path``."""

    path: Path


@dataclass(frozen=True)
class Malformed:
    """A document exists at ``path`` but does not read as a valid value."""

    path: Path
    reason: str


@dataclass(frozen=True)
class Ok(Generic[T]):
    """The document's validated value."""

    value: T


StateRead = Missing | Malformed | Ok[T]


class JsonDocument(Protocol):
    def to_json(self) -> dict[str, object]: ...


_ABSENT = object()


def read_state(path: Path, codec: Callable[[object], T]) -> StateRead[T]:
    """Read the document at *path* through *codec*. Never raises.

    *codec* builds the value from the parsed JSON and raises ``ValueError``
    naming the offending field. Every :class:`Malformed` answer is logged
    once as a WARNING naming *path*; :class:`Missing` logs nothing."""
    try:
        raw, parsed = load_json_or_warn(path, default=_ABSENT, on_corrupt='warn')
    except OSError as exc:
        logger.warning('state document at %s is unreadable: %s', path, exc)
        return Malformed(path, f'unreadable: {exc}')
    if not parsed:
        # load_json_or_warn has already logged the parse error.
        return Malformed(path, 'not valid JSON')
    if raw is _ABSENT:
        return Missing(path)
    try:
        return Ok(codec(raw))
    except ValueError as exc:
        logger.warning('state document at %s is malformed: %s', path, exc)
        return Malformed(path, str(exc))


def write_state(path: Path, value: JsonDocument) -> None:
    """Atomically replace *path* with *value*'s JSON, creating its parent
    directories. An ``OSError`` propagates: the caller owns the policy."""
    text = json.dumps(value.to_json(), indent=2, sort_keys=True) + '\n'
    atomic_write_text(path, text, mkdir=True)


# ---------------------------------------------------------------------------
# Field readers: a presence policy (required / defaulted) applied with a
# value parser (as_*). Every rejection is a ValueError naming field and value.
# ---------------------------------------------------------------------------

def json_object(raw: object, what: str) -> Mapping[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f'{what}: expected a JSON object, got {type(raw).__name__}')
    return raw


def required(
    doc: Mapping[str, object], name: str, parse: Callable[[str, object], T]
) -> T:
    raw = doc.get(name)
    if raw is None:
        raise ValueError(f'{name}: required field is missing or null')
    return parse(name, raw)


def defaulted(
    doc: Mapping[str, object],
    name: str,
    parse: Callable[[str, object], T],
    default: D,
) -> T | D:
    """*default* when the field is absent or null, else the parsed value."""
    raw = doc.get(name)
    return default if raw is None else parse(name, raw)


def as_int(name: str, raw: object) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f'{name}: expected an integer, got {raw!r}')
    return raw


def as_count(name: str, raw: object) -> int:
    value = as_int(name, raw)
    if value < 0:
        raise ValueError(f'{name}: expected a non-negative integer, got {raw!r}')
    return value


def as_bool(name: str, raw: object) -> bool:
    if not isinstance(raw, bool):
        raise ValueError(f'{name}: expected a boolean, got {raw!r}')
    return raw


def as_str(name: str, raw: object) -> str:
    if not isinstance(raw, str):
        raise ValueError(f'{name}: expected a string, got {raw!r}')
    return raw


def as_date(name: str, raw: object) -> date:
    if isinstance(raw, str):
        try:
            return date.fromisoformat(raw)
        except ValueError:
            pass
    raise ValueError(f'{name}: expected an ISO date, got {raw!r}')


def as_datetime(name: str, raw: object) -> datetime:
    """An ISO datetime; a naive one is read as UTC."""
    if isinstance(raw, str):
        try:
            stamp = datetime.fromisoformat(raw)
        except ValueError:
            pass
        else:
            return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=UTC)
    raise ValueError(f'{name}: expected an ISO datetime, got {raw!r}')


def as_member(kind: type[E]) -> Callable[[str, object], E]:
    """A parser accepting exactly the values of the enum *kind*."""

    def parse(name: str, raw: object) -> E:
        for member in kind:
            if member.value == raw:
                return member
        allowed = ', '.join(repr(member.value) for member in kind)
        raise ValueError(f'{name}: expected one of {allowed}, got {raw!r}')

    return parse
