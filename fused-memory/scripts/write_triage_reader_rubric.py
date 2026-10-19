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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any


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
