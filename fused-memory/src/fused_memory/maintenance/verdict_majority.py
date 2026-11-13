"""Resolve per-rater verdict rows into one :class:`Resolution` per pair.

One row per (rater, pair): ``{entry_id, target_id, verdict, rater, batch}``,
plus any extra keys (``reason``). ``verdict`` is one of :data:`VERDICT_WORDS`,
the rater brief's scale, whose home is ``link_heal.py::Verdict``. A pair may
carry several raters' votes. It resolves by majority on each of two binaries,
belongs-vs-misfile first and then CORRECTS-vs-not, with UNCLEAR abstaining. An
even split on either binary resolves TIED.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Any, TypeAlias

from fused_memory.maintenance.link_heal import MISFILE_VERDICTS, Verdict

VERDICT_WORDS: tuple[str, ...] = tuple(verdict.value for verdict in Verdict)
_REQUIRED_VOTE_KEYS = ('entry_id', 'target_id', 'verdict', 'rater', 'batch')

Pair: TypeAlias = tuple[str, str]
"""``(entry_id, target_id)``: a write and the record it was attached to."""


class Resolution(StrEnum):
    """What the raters, taken together, say about one pair.

    AGREEING (SAME/EXTENDS/SUBSUMED) and CORRECTS mean the write belongs under
    the target; MISFILE (RELATED/UNRELATED) means it does not. UNCLEAR means
    every rater abstained. TIED means the raters split evenly on a binary, so
    the corpus still owes this pair another rating.
    """

    AGREEING = 'agreeing'
    CORRECTS = 'corrects'
    MISFILE = 'misfile'
    UNCLEAR = 'unclear'
    TIED = 'tied'

    @property
    def belongs(self) -> bool:
        return self in (Resolution.AGREEING, Resolution.CORRECTS)


def resolve_verdicts(rows: Iterable[Mapping[str, Any]]) -> dict[Pair, Resolution]:
    """Resolve every rated pair in the corpus; refuse a malformed or double vote."""
    votes: defaultdict[Pair, dict[str, Verdict]] = defaultdict(dict)
    for index, row in enumerate(rows):
        pair, rater, word = _vote(index, row)
        if rater in votes[pair]:
            raise ValueError(
                f'verdict row {index}: rater {rater!r} already voted on pair {pair}'
            )
        votes[pair][rater] = word
    return {pair: _majority(by_rater.values()) for pair, by_rater in votes.items()}


def _vote(index: int, row: Mapping[str, Any]) -> tuple[Pair, str, Verdict]:
    pair = (row.get('entry_id'), row.get('target_id'))
    missing = [key for key in _REQUIRED_VOTE_KEYS if key not in row]
    if missing:
        raise ValueError(f'verdict row {index} on pair {pair} lacks {", ".join(missing)}')
    try:
        word = Verdict(row['verdict'])
    except ValueError:
        raise ValueError(
            f'verdict row {index} on pair {pair}: {row["verdict"]!r} is not one of {VERDICT_WORDS}'
        ) from None
    return (row['entry_id'], row['target_id']), row['rater'], word


def _majority(words: Iterable[Verdict]) -> Resolution:
    decided = [word for word in words if word != Verdict.UNCLEAR]
    if not decided:
        return Resolution.UNCLEAR
    misfile = _binary(sum(word in MISFILE_VERDICTS for word in decided), len(decided))
    if misfile is None:
        return Resolution.TIED
    if misfile:
        return Resolution.MISFILE
    corrects = _binary(sum(word == Verdict.CORRECTS for word in decided), len(decided))
    if corrects is None:
        return Resolution.TIED
    return Resolution.CORRECTS if corrects else Resolution.AGREEING


def _binary(yes: int, votes: int) -> bool | None:
    """Strict majority of *votes*: True for yes, False for no, None for a tie."""
    if 2 * yes == votes:
        return None
    return 2 * yes > votes
