"""Session-stable item numbers for a sitting, and the bookkeeping of Leo's answers against them.

A number, once given to an item key, is that item's for the whole sitting. New
keys take the next number, an item that leaves the open set moves to ``done``
under the number it had, and no number is ever issued twice. A sitting seeded
from the nightly ledger keeps the numbers Leo read in ``data/return-brief.md``.

The ledger never reads Leo's prose. The agent tokenises his message into
:class:`Answer` values; this module resolves those structured tokens against
structured state and asks back for any token it cannot resolve, never guessing.

``answer_rounds`` is the resolution_turns instrument, and this is its one
definition: the number of Leo turns (``resolve_answers`` calls) that referenced
an item while it was live in the ledger, counted once per turn however many
tokens named it, ask-backs included. The apply payload stamps it as
``resolution_turns``.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from shared.safe_io import atomic_write_text
from sitting.inventory import key_str, parse_key, parse_stamp
from sitting.preparation import Standing, standing_from_json, standing_to_json

ENTRY_STATES = frozenset({'open', 'standing', 'done'})
ANSWER_SOURCES = frozenset({'terminal', 'docket'})
_ITEM_NUMBER_RE = re.compile(r'[0-9]+')


class LedgerCorrupt(Exception):
    """The ledger at :attr:`path` exists but cannot be read; raised rather than silently renumbering."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f'{path}: sitting ledger is unreadable ({reason}); fix it or start a new sitting')


@dataclass(frozen=True)
class LedgerEntry:
    number: int
    state: str
    first_presented_at: str
    standing: Standing | None = None
    done_at: str | None = None
    answer_rounds: int = 0
    first_answered_at: str | None = None
    last_answered_at: str | None = None

    def __post_init__(self) -> None:
        if not _is_count(self.number) or self.number < 1:
            raise ValueError(f'an item number is a positive integer, not {self.number!r}')
        if not isinstance(self.state, str) or self.state not in ENTRY_STATES:
            raise ValueError(f'entry state {self.state!r} is not one of {sorted(ENTRY_STATES)}')
        if self.standing is not None and not isinstance(self.standing, Standing):
            raise ValueError(f'standing must be a Standing or None, not {self.standing!r}')
        if (self.state == 'standing') != (self.standing is not None):
            raise ValueError('an entry carries a Standing exactly when its state is standing')
        if (self.state == 'done') != (self.done_at is not None):
            raise ValueError('an entry carries done_at exactly when its state is done')
        if not _is_count(self.answer_rounds) or self.answer_rounds < 0:
            raise ValueError(f'answer_rounds is a count, not {self.answer_rounds!r}')
        if (self.first_answered_at is None) != (self.last_answered_at is None):
            raise ValueError('first_answered_at and last_answered_at are set together')
        for name in ('first_presented_at', 'done_at', 'first_answered_at', 'last_answered_at'):
            stamp = getattr(self, name)
            if stamp is not None:
                _instant(name, stamp)


@dataclass(frozen=True)
class Ledger:
    """One sitting's numbering.

    ``next_number`` is the high-water mark: a number stays retired even when
    its entry is not carried into a seeded sitting.
    """

    sitting_id: str
    started_at: str
    entries: Mapping[str, LedgerEntry] = field(default_factory=dict)
    next_number: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.sitting_id, str) or not self.sitting_id.strip():
            raise ValueError('a sitting has an id')
        _instant('started_at', self.started_at)
        for key in self.entries:
            _require_key(key)
        numbers = [entry.number for entry in self.entries.values()]
        if len(set(numbers)) != len(numbers):
            raise ValueError(f'item numbers repeat in sitting {self.sitting_id}: {sorted(numbers)}')
        if not _is_count(self.next_number) or self.next_number <= max(numbers, default=0):
            raise ValueError(f'next_number {self.next_number!r} would reissue a number already given')
        object.__setattr__(self, 'entries', MappingProxyType(dict(self.entries)))

    def in_state(self, state: str) -> tuple[tuple[str, LedgerEntry], ...]:
        """The ``(key, entry)`` pairs in *state*, in number order."""
        matching = [(key, entry) for key, entry in self.entries.items() if entry.state == state]
        return tuple(sorted(matching, key=lambda pair: pair[1].number))


@dataclass(frozen=True)
class Answer:
    """One structured token from Leo's message, e.g. ``3=B``: the agent tokenises; this module resolves."""

    item_ref: str
    option_ref: str
    note: str
    answered_at: str
    source: str

    def __post_init__(self) -> None:
        if not all(isinstance(text, str) for text in (self.item_ref, self.option_ref, self.note)):
            raise ValueError('item_ref, option_ref and note are text')
        _instant('answered_at', self.answered_at)
        if self.source not in ANSWER_SOURCES:
            raise ValueError(f'answer source {self.source!r} is not one of {sorted(ANSWER_SOURCES)}')


@dataclass(frozen=True)
class EchoRow:
    number: int
    key: str
    record_id: str
    option_label: str
    option_text: str
    note: str
    answered_at: str


@dataclass(frozen=True)
class Unresolved:
    answer: Answer
    reason: str


@dataclass(frozen=True)
class ResolvedAnswers:
    rows: tuple[EchoRow, ...]
    unresolved: tuple[Unresolved, ...]


@dataclass(frozen=True)
class SittingSummary:
    started_at: str
    items_presented: int
    items_answered: int
    first_answered_at: str | None
    last_answered_at: str | None
    answer_rounds: int


def new_sitting(started_at: str, seed: Ledger | None = None) -> Ledger:
    """A fresh sitting. A *seed* (the nightly ledger) lends it every entry not yet done, and its high-water mark."""
    sitting_id = f'sitting-{started_at}'
    if seed is None:
        return Ledger(sitting_id, started_at)
    carried = {
        key: replace(entry, answer_rounds=0, first_answered_at=None, last_answered_at=None)
        for key, entry in seed.entries.items()
        if entry.state != 'done'
    }
    return Ledger(sitting_id, started_at, carried, seed.next_number)


def assign(ledger: Ledger, live_keys: Iterable[str], sort_key: Callable[[str], Any], now: str) -> Ledger:
    """Number the new live keys in *sort_key* order (key as the final tiebreak); keys no longer live go done."""
    _instant('now', now)
    live = {_require_key(key) for key in live_keys}
    entries = {key: _carried(entry, key in live, now) for key, entry in ledger.entries.items()}
    next_number = ledger.next_number
    for key in sorted(live - entries.keys(), key=lambda key: (sort_key(key), key)):
        entries[key] = LedgerEntry(number=next_number, state='open', first_presented_at=now)
        next_number += 1
    return replace(ledger, entries=entries, next_number=next_number)


def set_standing(ledger: Ledger, key: str, standing: Standing | None) -> Ledger:
    """Record why a live item is not put to Leo (None clears it); it keeps its number for the footer."""
    entry = _entry(ledger, key)
    if entry.state == 'done':
        raise ValueError(f'item {entry.number} is done; a done item takes no standing')
    updated = replace(entry, state='open' if standing is None else 'standing', standing=standing)
    return replace(ledger, entries={**ledger.entries, key: updated})


def resolve_answers(
    ledger: Ledger, answers: Iterable[Answer], options_by_key: Mapping[str, Mapping[str, str]],
) -> tuple[Ledger, ResolvedAnswers]:
    """Resolve one Leo turn against the ledger; *options_by_key* maps each item key to ``{label: text}``."""
    by_number = {entry.number: key for key, entry in ledger.entries.items()}
    outcomes = [_resolve(answer, ledger, by_number, options_by_key) for answer in answers]
    rows = tuple(outcome for _, outcome in outcomes if isinstance(outcome, EchoRow))
    unresolved = tuple(outcome for _, outcome in outcomes if isinstance(outcome, Unresolved))
    touched = {key for key, _ in outcomes if key is not None}
    entries = {
        key: _answered(entry, [row.answered_at for row in rows if row.key == key]) if key in touched else entry
        for key, entry in ledger.entries.items()
    }
    return replace(ledger, entries=entries), ResolvedAnswers(rows, unresolved)


def resolution_turns(ledger: Ledger, key: str) -> int:
    return _entry(ledger, key).answer_rounds


def sitting_summary(ledger: Ledger) -> SittingSummary:
    """The trial's sitting-time instrument inputs; standing items are not counted as presented."""
    entries = list(ledger.entries.values())
    first = [entry.first_answered_at for entry in entries if entry.first_answered_at is not None]
    last = [entry.last_answered_at for entry in entries if entry.last_answered_at is not None]
    return SittingSummary(
        started_at=ledger.started_at,
        items_presented=sum(entry.state != 'standing' for entry in entries),
        items_answered=len(last),
        first_answered_at=min(first, key=_answered_instant, default=None),
        last_answered_at=max(last, key=_answered_instant, default=None),
        answer_rounds=sum(entry.answer_rounds for entry in entries),
    )


def render_echo_table(resolved: ResolvedAnswers) -> str:
    """Markdown: one row per resolved answer, then an ``ASK BACK:`` block naming each unresolved token and why."""
    if resolved.rows:
        lines = ['| # | record | option | text | note | answered at |', '|---|---|---|---|---|---|']
        lines += [
            '| ' + ' | '.join(_cell(value) for value in (
                str(row.number), row.record_id, row.option_label, row.option_text, row.note, row.answered_at,
            )) + ' |'
            for row in resolved.rows
        ]
    else:
        lines = ['No answer resolved.']
    if resolved.unresolved:
        lines += ['', 'ASK BACK:']
        lines += [
            f'- item {u.answer.item_ref!r}, option {u.answer.option_ref!r}: {u.reason}' for u in resolved.unresolved
        ]
    return '\n'.join(lines) + '\n'


def load(path: Path | str) -> Ledger | None:
    """The ledger at *path*; None when absent, :class:`LedgerCorrupt` when unreadable."""
    path = Path(path)
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise LedgerCorrupt(path, str(exc)) from exc
    try:
        return _decode(json.loads(text))
    except (KeyError, TypeError, ValueError) as exc:
        raise LedgerCorrupt(path, f'{type(exc).__name__}: {exc}') from exc


def save(path: Path | str, ledger: Ledger) -> None:
    atomic_write_text(path, json.dumps(_encode(ledger), indent=2, ensure_ascii=False) + '\n', mkdir=True)


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _instant(name: str, stamp: object) -> datetime:
    parsed = parse_stamp(stamp) if isinstance(stamp, str) else None
    if parsed is None:
        raise ValueError(f'{name} {stamp!r} is not an ISO-8601 timestamp')
    return parsed


def _answered_instant(stamp: str) -> datetime:
    return _instant('answered_at', stamp)


def _require_key(key: object) -> str:
    if not isinstance(key, str) or key_str(parse_key(key)) != key:
        raise ValueError(f'{key!r} is not an open-item key string')
    return key


def _record_id(key: str) -> str:
    """Both key kinds end in the id Leo sees: the escalation id, or the decision id."""
    return parse_key(key)[-1]


def _entry(ledger: Ledger, key: str) -> LedgerEntry:
    entry = ledger.entries.get(key)
    if entry is None:
        raise ValueError(f'{key} is not in sitting {ledger.sitting_id}')
    return entry


def _carried(entry: LedgerEntry, live: bool, now: str) -> LedgerEntry:
    if live:
        return replace(entry, state='open', done_at=None) if entry.state == 'done' else entry
    if entry.state == 'done':
        return entry
    return replace(entry, state='done', done_at=now, standing=None)


def _resolve(
    answer: Answer, ledger: Ledger, by_number: Mapping[int, str], options_by_key: Mapping[str, Mapping[str, str]],
) -> tuple[str | None, EchoRow | Unresolved]:
    """The key the token touched (None when it names no live item), and its outcome."""
    ref = answer.item_ref.strip()
    if not _ITEM_NUMBER_RE.fullmatch(ref):
        return None, Unresolved(answer, f'{answer.item_ref!r} is not an item number')
    number = int(ref)
    key = by_number.get(number)
    if key is None:
        return None, Unresolved(answer, f'no item {number} in this sitting')
    entry = ledger.entries[key]
    if entry.state == 'done':
        return None, Unresolved(answer, f'item {number} is already done')
    if entry.standing is not None:
        return key, Unresolved(
            answer, f'item {number} is standing ({entry.standing.kind}, owner {entry.standing.owner}), '
                    'not a question put to you',
        )
    return key, _match_option(answer, number, key, options_by_key.get(key, {}))


def _match_option(answer: Answer, number: int, key: str, options: Mapping[str, str]) -> EchoRow | Unresolved:
    label = answer.option_ref.strip()
    listed = ', '.join(options)
    if not options:
        return Unresolved(answer, f'item {number} has no options on record')
    if not label:
        return Unresolved(answer, f'item {number}: no option given (options: {listed})')
    if label not in options:
        return Unresolved(answer, f'item {number} has no option {label!r} (options: {listed})')
    return EchoRow(number, key, _record_id(key), label, options[label], answer.note, answer.answered_at)


def _answered(entry: LedgerEntry, stamps: list[str]) -> LedgerEntry:
    known = [stamp for stamp in (entry.first_answered_at, entry.last_answered_at) if stamp is not None]
    ordered = sorted([*known, *stamps], key=_answered_instant)
    return replace(
        entry,
        answer_rounds=entry.answer_rounds + 1,
        first_answered_at=ordered[0] if ordered else None,
        last_answered_at=ordered[-1] if ordered else None,
    )


def _cell(value: str) -> str:
    return ' '.join(value.split()).replace('|', r'\|')


def _encode(ledger: Ledger) -> dict[str, Any]:
    return {
        'sitting_id': ledger.sitting_id,
        'started_at': ledger.started_at,
        'next_number': ledger.next_number,
        'entries': [
            {
                'key': list(parse_key(key)),
                'number': entry.number,
                'state': entry.state,
                'first_presented_at': entry.first_presented_at,
                'standing': None if entry.standing is None else standing_to_json(entry.standing),
                'done_at': entry.done_at,
                'answer_rounds': entry.answer_rounds,
                'first_answered_at': entry.first_answered_at,
                'last_answered_at': entry.last_answered_at,
            }
            for key, entry in sorted(ledger.entries.items(), key=lambda pair: pair[1].number)
        ],
    }


def _decode(data: Any) -> Ledger:
    entries: dict[str, LedgerEntry] = {}
    for raw in data['entries']:
        key = key_str(parse_key(json.dumps(raw['key'])))
        if key in entries:
            raise ValueError(f'{key} appears twice')
        entries[key] = LedgerEntry(
            number=raw['number'],
            state=raw['state'],
            first_presented_at=raw['first_presented_at'],
            standing=None if raw['standing'] is None else standing_from_json(raw['standing']),
            done_at=raw['done_at'],
            answer_rounds=raw['answer_rounds'],
            first_answered_at=raw['first_answered_at'],
            last_answered_at=raw['last_answered_at'],
        )
    return Ledger(data['sitting_id'], data['started_at'], entries, data['next_number'])
