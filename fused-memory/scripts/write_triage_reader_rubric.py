"""The one definition of write-triage correctness rubric v2.

A rated item is a (write, record) pair judged on the binaries of
:class:`Binary`. This module defines the verdict-row contract, the per-binary
majority that resolves several raters' rows into one :class:`Judgement`, and
the derivation that turns judgements into the wrong cells a reader meets.

The rater-facing wording is ``fused-memory/calibration/write_triage_rater_brief_v2.md``;
the metrics built on the cells are ``fused-memory/calibration/write_triage_reader_metrics.md``.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from types import MappingProxyType
from typing import Any, TypeAlias

from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    TRIAGE_OUTCOMES,
)


class Answer(StrEnum):
    YES = 'YES'
    NO = 'NO'
    UNCLEAR = 'UNCLEAR'


class Scope(StrEnum):
    CENTRAL = 'CENTRAL'
    SECONDARY = 'SECONDARY'
    UNCLEAR = 'UNCLEAR'


class Binary(StrEnum):
    HOME = 'home'
    LOSS = 'loss'
    CONFLICT = 'conflict'
    SCOPE = 'scope'
    DIGEST = 'digest'


class Basis(StrEnum):
    """Which text of the record the rater read: as it is now, or as it was at attach."""

    CURRENT = 'current'
    ATTACH_TIME = 'attach_time'


class Role(StrEnum):
    """Where a verdict row came from; a ``derived`` row is a label fixed by an earlier rubric."""

    PRIMARY = 'primary'
    SECOND = 'second'
    THIRD = 'third'
    DERIVED = 'derived'


class Resolved(StrEnum):
    """What the rows carrying one binary of one pair say, taken together."""

    YES = 'YES'
    NO = 'NO'
    CENTRAL = 'CENTRAL'
    SECONDARY = 'SECONDARY'
    UNCLEAR = 'UNCLEAR'
    TIED = 'TIED'


_WORDS: Mapping[Binary, type[StrEnum]] = MappingProxyType({
    Binary.HOME: Answer,
    Binary.LOSS: Answer,
    Binary.CONFLICT: Answer,
    Binary.SCOPE: Scope,
    Binary.DIGEST: Answer,
})
"""The answer vocabulary of each binary."""

_FOLLOWS: Mapping[Binary, Binary] = MappingProxyType({
    Binary.SCOPE: Binary.CONFLICT,
    Binary.DIGEST: Binary.LOSS,
})
"""A conditional follow-up, and the binary that must be YES on the same row to ask it."""

_POLES: Mapping[Binary, tuple[Resolved, Resolved]] = MappingProxyType({
    binary: (Resolved.CENTRAL, Resolved.SECONDARY) if words is Scope else (Resolved.YES, Resolved.NO)
    for binary, words in _WORDS.items()
})
"""Each binary's (positive, negative) resolution; the positive word is the one counted."""

M1_BAND = OUTCOME_RESTATED
"""The triage band whose writes M1 counts: band-winner cosine at or above t_high when written."""

_UNCLEAR = 'UNCLEAR'
_REQUIRED_KEYS = ('entry_id', 'target_id', 'basis', 'rater', 'role', 'batch')
_IDENTITY_KEYS = ('entry_id', 'target_id', 'rater')


@dataclass(frozen=True, order=True)
class PairKey:
    """One judged pair on one basis: a write, the record it is read against, which text."""

    entry_id: str
    target_id: str
    basis: Basis


@dataclass(frozen=True)
class Vote:
    """One rater's validated answers on one pair key."""

    rater: str
    role: Role
    answers: Mapping[Binary, StrEnum]


@dataclass(frozen=True)
class Judgement:
    """Every vote on one pair key, and the binaries those votes resolve to."""

    key: PairKey
    votes: tuple[Vote, ...]
    resolved: Mapping[Binary, Resolved]

    def answer(self, binary: Binary) -> Resolved | None:
        """The resolved answer on *binary*, or None when no row carries it."""
        return self.resolved.get(binary)


def resolve_judgements(rows: Iterable[Mapping[str, Any]]) -> Mapping[PairKey, Judgement]:
    """Validate every verdict row and resolve each pair key's binaries by strict majority.

    Votes on one binary come from the rows carrying it. UNCLEAR abstains, an
    all-UNCLEAR binary resolves UNCLEAR and an even split resolves TIED. The
    ``current`` and ``attach_time`` bases are separate pair keys and never mix.
    A malformed row, or a rater voting twice on one pair key, is refused.
    """
    votes: defaultdict[PairKey, dict[str, Vote]] = defaultdict(dict)
    for index, row in enumerate(rows):
        pair = (row.get('entry_id'), row.get('target_id'), row.get('basis'))
        violation = _row_violation(row)
        if violation is not None:
            raise ValueError(f'verdict row {index} on pair {pair}: {violation}')
        key = PairKey(row['entry_id'], row['target_id'], Basis(row['basis']))
        if row['rater'] in votes[key]:
            raise ValueError(
                f'verdict row {index}: rater {row["rater"]!r} already rated pair {pair}'
            )
        votes[key][row['rater']] = Vote(
            rater=row['rater'],
            role=Role(row['role']),
            answers=MappingProxyType({
                binary: words(row[binary]) for binary, words in _WORDS.items() if binary in row
            }),
        )
    return MappingProxyType({
        key: _judgement(key, tuple(votes[key].values())) for key in sorted(votes)
    })


def _row_violation(row: Mapping[str, Any]) -> str | None:
    """The first rubric invariant *row* breaks, naming the offending value; None if none."""
    missing = [key for key in _REQUIRED_KEYS if key not in row]
    if missing:
        return f'lacks {", ".join(missing)}'
    for key in _IDENTITY_KEYS:
        if not isinstance(row[key], str) or not row[key]:
            return f'{key} {row[key]!r} is not a non-empty string'
    for key, words in (('basis', Basis), ('role', Role)):
        if row[key] not in set(words):
            return f'{key} {row[key]!r} is not one of {sorted(words)}'
    carried = [binary for binary in _WORDS if binary in row]
    if not carried:
        return f'carries no binary (one of {", ".join(_WORDS)})'
    for binary in carried:
        if row[binary] not in set(_WORDS[binary]):
            return f'{binary} {row[binary]!r} is not one of {sorted(_WORDS[binary])}'
    unclear = [binary for binary in carried if row[binary] == _UNCLEAR]
    reason = row.get('unclear_reason')
    if unclear and not (isinstance(reason, str) and reason.strip()):
        return f'{unclear[0]} {_UNCLEAR!r} needs a non-empty unclear_reason'
    for follow_up, asked_after in _FOLLOWS.items():
        if follow_up in row and row.get(asked_after) != Answer.YES:
            return (
                f'{follow_up} {row[follow_up]!r} needs {asked_after} {Answer.YES.value!r},'
                f' got {row.get(asked_after)!r}'
            )
    return None


def _judgement(key: PairKey, votes: tuple[Vote, ...]) -> Judgement:
    resolved = {
        binary: _majority([vote.answers[binary] for vote in votes if binary in vote.answers], binary)
        for binary in _WORDS
        if any(binary in vote.answers for vote in votes)
    }
    return Judgement(key=key, votes=votes, resolved=MappingProxyType(resolved))


def _majority(words: list[StrEnum], binary: Binary) -> Resolved:
    decided = [word for word in words if word != _UNCLEAR]
    if not decided:
        return Resolved.UNCLEAR
    positive, negative = _POLES[binary]
    majority = _strict_majority(sum(word == positive for word in decided), len(decided))
    if majority is None:
        return Resolved.TIED
    return positive if majority else negative


def _strict_majority(yes: int, votes: int) -> bool | None:
    """Strict majority of *votes*: True for yes, False for no, None for a tie."""
    if 2 * yes == votes:
        return None
    return 2 * yes > votes


class Cell(StrEnum):
    """A way a write's outcome leaves a reader worse off; one write may fall in several."""

    HIDDEN_DETAIL = 'hidden_detail'
    HIDDEN_CONFLICT = 'hidden_conflict'
    DIGEST_CONFLICT = 'digest_conflict'
    WRONG_HOME = 'wrong_home'
    FALSE_CONFLICT = 'false_conflict'
    DUPLICATE_CLUTTER = 'duplicate_clutter'
    ORPHANED_CONFLICT = 'orphaned_conflict'
    ATTACHED_SUPERSESSION = 'attached_supersession'


CELL_RULES: Mapping[Cell, str] = MappingProxyType({
    Cell.HIDDEN_DETAIL: (
        'A restated write adds something a reader acting on its record would need'
        ' (HOME, LOSS, no CONFLICT), and a sighting is only counted, never shown.'
    ),
    Cell.HIDDEN_CONFLICT: (
        'A restated write disputes its record (CONFLICT), and the dispute sits in a'
        ' sighting a reader of the record never sees.'
    ),
    Cell.DIGEST_CONFLICT: (
        'An amended write disputes its record (CONFLICT), and a reader of the record'
        ' meets the dispute only as an untagged 240-character digest.'
    ),
    Cell.WRONG_HOME: (
        "An attached write is not about its record's claim (HOME=NO); a contest whose"
        ' dispute is true is never a wrong home (ruling 5b).'
    ),
    Cell.FALSE_CONFLICT: (
        'A contested write does not dispute its record (CONFLICT=NO), so a reader is'
        ' told a sound record is in doubt.'
    ),
    Cell.DUPLICATE_CLUTTER: (
        'A standalone write restates an earlier record and adds nothing a reader needs'
        ' (HOME, no LOSS, no CONFLICT).'
    ),
    Cell.ORPHANED_CONFLICT: (
        'A standalone write disputes an earlier record (CONFLICT) that carries no'
        ' superseded-by mark for it, so a reader of that record never learns of it.'
    ),
    Cell.ATTACHED_SUPERSESSION: (
        'A write declaring that it supersedes its record was attached under it; a'
        ' supersession is a new standalone record plus a mark, never an attach (ruling 5c).'
    ),
})
"""The rule of each cell, as a report carries it; :func:`derive` is the rules' one implementation."""


class GapKind(StrEnum):
    UNRATED = 'unrated'
    TIED = 'tied'


@dataclass(frozen=True, order=True)
class Gap:
    """A binary the derivation needed on one record that the verdicts leave unrated or tied."""

    target_id: str
    binary: Binary
    kind: GapKind


@dataclass(frozen=True)
class Decided:
    """A write's cells under triage and under the status quo (every write stored flat).

    ``conflict`` says whether the write disputes its parent or any earlier
    record. ``m1_hidden`` is None outside the M1 band.
    """

    triage_cells: frozenset[Cell]
    status_quo_cells: frozenset[Cell]
    m1_hidden: bool | None
    conflict: bool


@dataclass(frozen=True)
class Unclear:
    """A binary the derivation needed resolved UNCLEAR: every rater abstained on it."""

    binaries: tuple[tuple[str, Binary], ...]


@dataclass(frozen=True)
class Incomplete:
    """The verdicts leave binaries the derivation needed unrated or tied."""

    missing: tuple[Gap, ...]


Derivation: TypeAlias = Decided | Unclear | Incomplete


def derive(
    outcome: str,
    judgements_by_target: Mapping[str, Judgement | None],
    *,
    supersedes_target: bool = False,
    superseded_marks: frozenset[str] = frozenset(),
    in_m1_band: bool = False,
) -> Derivation:
    """Derive one write's cells from the judgements of the records it is read against.

    An attach outcome is read against exactly one target, its parent. A stored
    write is read against each earlier record it might have joined, and each
    one is derived on its own. Binaries are asked lazily, in a fixed order per
    outcome, so a binary is decisive exactly when the derivation asks for it: a
    decisive unrated or tied binary makes the write Incomplete (naming every
    gap the three derivations met), a decisive UNCLEAR one makes it Unclear,
    and a binary never asked never blocks. ``supersedes_target`` and
    ``superseded_marks`` are metadata facts (ruling 5c), never rater answers.
    """
    regimes = _regimes(
        outcome, judgements_by_target, supersedes_target=supersedes_target,
        superseded_marks=superseded_marks, in_m1_band=in_m1_band,
    )
    results: dict[str, list[Any]] = {regime: [] for regime in regimes}
    gaps: set[Gap] = set()
    unclear: dict[tuple[str, Binary], None] = {}
    for regime, thunks in regimes.items():
        for thunk in thunks:
            try:
                results[regime].append(thunk())
            except _Undecided as undecided:
                gaps.add(undecided.gap)
            except _Abstained as abstained:
                unclear[abstained.target_id, abstained.binary] = None
    if gaps:
        return Incomplete(tuple(sorted(gaps)))
    if unclear:
        return Unclear(tuple(unclear))
    triage = _Reading.union(results['triage'])
    return Decided(
        triage_cells=triage.cells | (
            {Cell.ATTACHED_SUPERSESSION} if supersedes_target else frozenset()
        ),
        status_quo_cells=_Reading.union(results['status_quo']).cells,
        m1_hidden=results['m1'][0] if in_m1_band else None,
        conflict=triage.conflict,
    )


class _Undecided(Exception):
    def __init__(self, gap: Gap) -> None:
        super().__init__(gap)
        self.gap = gap


class _Abstained(Exception):
    def __init__(self, target_id: str, binary: Binary) -> None:
        super().__init__(target_id, binary)
        self.target_id = target_id
        self.binary = binary


@dataclass(frozen=True)
class _Reader:
    """Lazy access to one record's judgement; ``marked`` = it carries a superseded-by mark."""

    target_id: str
    judgement: Judgement | None
    marked: bool = False

    def yes(self, binary: Binary) -> bool:
        resolved = None if self.judgement is None else self.judgement.answer(binary)
        if resolved is None:
            raise _Undecided(Gap(self.target_id, binary, GapKind.UNRATED))
        if resolved is Resolved.TIED:
            raise _Undecided(Gap(self.target_id, binary, GapKind.TIED))
        if resolved is Resolved.UNCLEAR:
            raise _Abstained(self.target_id, binary)
        return resolved is Resolved.YES


@dataclass(frozen=True)
class _Reading:
    cells: frozenset[Cell]
    conflict: bool

    @staticmethod
    def union(readings: Sequence[_Reading]) -> _Reading:
        return _Reading(
            cells=frozenset().union(*(reading.cells for reading in readings)),
            conflict=any(reading.conflict for reading in readings),
        )


def _restated_cells(parent: _Reader) -> _Reading:
    if parent.yes(Binary.CONFLICT):
        return _Reading(frozenset({Cell.HIDDEN_CONFLICT}) | _wrong_home(parent), conflict=True)
    if not parent.yes(Binary.HOME):
        return _Reading(frozenset({Cell.WRONG_HOME}), conflict=False)
    hidden = {Cell.HIDDEN_DETAIL} if parent.yes(Binary.LOSS) else set()
    return _Reading(frozenset(hidden), conflict=False)


def _amended_cells(parent: _Reader) -> _Reading:
    conflict = parent.yes(Binary.CONFLICT)
    digest = {Cell.DIGEST_CONFLICT} if conflict else set()
    return _Reading(frozenset(digest) | _wrong_home(parent), conflict=conflict)


def _contested_cells(parent: _Reader) -> _Reading:
    if parent.yes(Binary.CONFLICT):
        return _Reading(frozenset(), conflict=True)
    return _Reading(frozenset({Cell.FALSE_CONFLICT}) | _wrong_home(parent), conflict=False)


def _stored_cells(earlier: _Reader) -> _Reading:
    if earlier.yes(Binary.CONFLICT):
        orphaned = set() if earlier.marked else {Cell.ORPHANED_CONFLICT}
        return _Reading(frozenset(orphaned), conflict=True)
    duplicate = earlier.yes(Binary.HOME) and not earlier.yes(Binary.LOSS)
    return _Reading(frozenset({Cell.DUPLICATE_CLUTTER} if duplicate else set()), conflict=False)


def _wrong_home(parent: _Reader) -> frozenset[Cell]:
    return frozenset() if parent.yes(Binary.HOME) else frozenset({Cell.WRONG_HOME})


def _m1_hidden(outcome: str, readers: Sequence[_Reader]) -> bool:
    """Whether a reader of the parent misses the write's dispute or needed detail (M1)."""
    if outcome not in (OUTCOME_RESTATED, OUTCOME_AMENDED):
        return False
    parent = readers[0]
    if parent.yes(Binary.CONFLICT):
        return True
    if not (parent.yes(Binary.HOME) and parent.yes(Binary.LOSS)):
        return False
    return outcome == OUTCOME_RESTATED or not parent.yes(Binary.DIGEST)


_CELL_RULES: Mapping[str, Callable[[_Reader], _Reading]] = MappingProxyType({
    OUTCOME_RESTATED: _restated_cells,
    OUTCOME_AMENDED: _amended_cells,
    OUTCOME_CONTESTED: _contested_cells,
    OUTCOME_STORED: _stored_cells,
})
"""Each outcome's triage-regime cells, read against one record."""


def _regimes(
    outcome: str,
    judgements_by_target: Mapping[str, Judgement | None],
    *,
    supersedes_target: bool,
    superseded_marks: frozenset[str],
    in_m1_band: bool,
) -> dict[str, list[Callable[[], Any]]]:
    """The deferred derivations of each regime, refusing a write the rubric cannot read."""
    rule = _CELL_RULES.get(outcome)
    if rule is None:
        raise ValueError(f'outcome {outcome!r} is not one of {sorted(TRIAGE_OUTCOMES)}')
    readers = [
        _Reader(target, judgement, marked=target in superseded_marks)
        for target, judgement in sorted(judgements_by_target.items())
    ]
    if outcome == OUTCOME_STORED:
        if supersedes_target:
            raise ValueError('a stored write has no attach target, so supersedes_target must be false')
        flat = readers
    elif len(readers) != 1:
        raise ValueError(f'a {outcome} write is read against exactly one target, got {len(readers)}')
    else:
        flat = [replace(readers[0], marked=False)]
    return {
        'triage': [partial(rule, reader) for reader in readers],
        'status_quo': [partial(_stored_cells, reader) for reader in flat],
        'm1': [partial(_m1_hidden, outcome, readers)] if in_m1_band else [],
    }
