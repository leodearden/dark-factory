"""The run home's one write transaction: program.json's lock held from read to the last append."""
from __future__ import annotations

import fcntl
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import state


@dataclass(frozen=True)
class Snapshot:
    """What the home records: the program, the ledger and the rulings, read together."""

    program: state.Program
    ledger: tuple[Mapping[str, Any], ...]
    rulings: tuple[Mapping[str, Any], ...]


def _uninitialised(home: state.Home) -> state.StateError:
    return state.StateError(f'{home.root} has no program.json; run record.py init first')


def read(home: state.Home) -> Snapshot:
    """The home's state, read without the lock: for a check that writes nothing."""
    try:
        text = home.program_json.read_text(encoding='utf-8')
    except FileNotFoundError:
        raise _uninitialised(home) from None
    try:
        data = json.loads(text)
    except ValueError as bad:
        raise state.StateError(f'{home.program_json} is not JSON: {bad}') from None
    return Snapshot(state.parse_program(data), state.read_ledger(home), state.read_rulings(home))


@contextmanager
def locked(home: state.Home) -> Iterator[Snapshot]:
    """The home's state, read under program.json's exclusive lock, held until the block exits."""
    try:
        handle = home.program_json.open(encoding='utf-8')
    except FileNotFoundError:
        raise _uninitialised(home) from None
    with handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield read(home)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class RulingRefused(state.LedgerRefused):
    """A ruling entry the rulings file would refuse; ``index`` is its place among the entries given."""

    def __init__(self, index: int, refused: state.LedgerRefused) -> None:
        self.index = index
        super().__init__(str(refused))


@dataclass(frozen=True)
class Lines:
    """Stamped lines ready to append: each ledger line numbered, each ruling minted."""

    ledger: tuple[dict[str, Any], ...]
    rulings: tuple[dict[str, Any], ...]


def _stamped(stamps: dict[str, Any], fields: Mapping[str, Any]) -> dict[str, Any]:
    clash = sorted(stamps.keys() & fields.keys())
    if clash:
        raise state.StateError(f'{", ".join(clash)} are stamped by the writer, not given')
    return {**stamps, **fields}


def stamp(
    snapshot: Snapshot, by: str,
    ledger_entries: Sequence[tuple[state.LedgerOp, Mapping[str, Any]]],
    ruling_entries: Sequence[tuple[state.RuledBy, Mapping[str, Any]]],
) -> Lines:
    """Number the ledger lines after the snapshot's and mint the rulings' ids; each line is checked."""
    at = state.now_utc()
    ledger: list[dict[str, Any]] = []
    for op, fields in ledger_entries:
        seq = state.next_seq([*snapshot.ledger, *ledger])
        line = _stamped({'seq': seq, 'at': at, 'by': by, 'op': str(op)}, fields)
        state.check_ledger_line(line)
        ledger.append(line)
    today = datetime.now(UTC).date().isoformat()
    rulings: list[dict[str, Any]] = []
    for ruled_by, fields in ruling_entries:
        ruling_id = state.next_ruling_id([*snapshot.rulings, *rulings], ruled_by)
        line = _stamped({'id': ruling_id, 'ruled_by': str(ruled_by), 'date': today}, fields)
        try:
            state.check_ruling_line(line)
        except state.LedgerRefused as refused:
            raise RulingRefused(len(rulings), refused) from None
        rulings.append(line)
    return Lines(tuple(ledger), tuple(rulings))


def append(home: state.Home, lines: Lines) -> None:
    """The ledger first, then the rulings; a refused ledger append leaves both files untouched.

    Only a ledger writer that skips the lock can make the first append refuse;
    every rulings writer holds it, so the second cannot.
    """
    if lines.ledger:
        state.append_ledger(home, lines.ledger)
    if lines.rulings:
        state.append_rulings(home, lines.rulings)
