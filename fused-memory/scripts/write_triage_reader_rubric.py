"""The one definition of write-triage correctness rubric v2.1.

A rated item is a (write, record) pair judged on the binaries of
:class:`Binary`. This module defines the verdict-row contract, the per-binary
resolution of several raters' rows into one :class:`Judgement`, and the
derivation that turns judgements into the wrong cells a reader meets.

The rater-facing wording is ``fused-memory/calibration/write_triage_rater_brief_v2.md``
(rater brief v2.1);
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

from fused_memory.server.grouped_read import _digest
from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    TRIAGE_OUTCOMES,
)

ABSTAIN = 'UNCLEAR'
"""The word a rater gives to abstain on any binary; it never counts toward a majority."""


def is_abstain(word: object) -> bool:
    """Whether *word* is the abstain word, in any binary's vocabulary or as a raw row value."""
    return word == ABSTAIN


class Answer(StrEnum):
    YES = 'YES'
    NO = 'NO'
    UNCLEAR = ABSTAIN


class Scope(StrEnum):
    CENTRAL = 'CENTRAL'
    SECONDARY = 'SECONDARY'
    UNCLEAR = ABSTAIN


class Binary(StrEnum):
    HOME = 'home'
    LOSS = 'loss'
    CONFLICT = 'conflict'
    SCOPE = 'scope'
    ACTION_CHANGING = 'action_changing'
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
    CROSS_FAMILY = 'cross_family'
    DERIVED = 'derived'


RATER_ROLES = frozenset({Role.PRIMARY, Role.SECOND, Role.THIRD, Role.CROSS_FAMILY})
"""The roles of a row a rater answered; only these rows carry evidence or non_consumed answers."""


class Resolved(StrEnum):
    """What the rows carrying one binary of one pair say, taken together.

    OPEN is a decisive follow-up no row carries, whose preconditions the rater
    votes resolve to its needs.
    """

    YES = 'YES'
    NO = 'NO'
    CENTRAL = 'CENTRAL'
    SECONDARY = 'SECONDARY'
    UNCLEAR = ABSTAIN
    TIED = 'TIED'
    OPEN = 'OPEN'


_WORDS: Mapping[Binary, type[StrEnum]] = MappingProxyType({
    Binary.HOME: Answer,
    Binary.LOSS: Answer,
    Binary.CONFLICT: Answer,
    Binary.SCOPE: Scope,
    Binary.ACTION_CHANGING: Answer,
    Binary.DIGEST: Answer,
})
"""The answer vocabulary of each binary."""

FOLLOW_UPS: Mapping[Binary, Mapping[Binary, Answer]] = MappingProxyType({
    Binary.LOSS: MappingProxyType({Binary.HOME: Answer.YES, Binary.CONFLICT: Answer.NO}),
    Binary.SCOPE: MappingProxyType({Binary.CONFLICT: Answer.YES}),
    Binary.ACTION_CHANGING: MappingProxyType({Binary.CONFLICT: Answer.YES}),
    Binary.DIGEST: MappingProxyType({Binary.LOSS: Answer.YES}),
})
"""Each conditional follow-up, and the answers the same row must give before it is asked."""

DECISIVE_FOLLOW_UPS: frozenset[Binary] = frozenset({Binary.LOSS})
"""The follow-ups :func:`derive` reads that a rater answers; DIGEST is computed from loss_spans.

A rater row whose own answers meet one's ``FOLLOW_UPS`` needs must give it, so a
skipped one is refused, never resolved UNCLEAR. One the rater votes ask that no
rater row reached resolves OPEN.
"""


def _is_conflict_quote(value: object) -> bool:
    return isinstance(value, dict) and all(
        _is_text(value.get(side)) for side in ('parent', 'child')
    )


def _is_loss_spans(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(map(_is_text, value))


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class _Evidence:
    """A row field that must back an answer, the shape it must have, and its check."""

    field: str
    shape: str
    valid: Callable[[object], bool]


_LOSS_SPANS = _Evidence('loss_spans', 'a non-empty list of non-empty strings', _is_loss_spans)

_EVIDENCE: Mapping[tuple[Binary, Answer], _Evidence] = MappingProxyType({
    (Binary.CONFLICT, Answer.YES): _Evidence(
        'conflict_quote', 'an object with non-empty string parent and child', _is_conflict_quote,
    ),
    (Binary.LOSS, Answer.YES): _LOSS_SPANS,
})
"""The evidence a rater row must give for an answer; a derived row gives none."""

_POLES: Mapping[Binary, tuple[Resolved, Resolved]] = MappingProxyType({
    binary: (Resolved.CENTRAL, Resolved.SECONDARY) if words is Scope else (Resolved.YES, Resolved.NO)
    for binary, words in _WORDS.items()
})
"""Each binary's (positive, negative) resolution; the positive word is the one counted."""

M1_BAND = OUTCOME_RESTATED
"""The triage band whose writes M1 counts: band-winner cosine at or above t_high when written."""

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
    """One rater's validated answers on one pair key.

    ``answers`` holds only the voting answers; a binary in ``non_consumed`` was
    answered and recorded, but a derived row fixes it, so it never votes.
    """

    rater: str
    role: Role
    answers: Mapping[Binary, StrEnum]
    non_consumed: frozenset[Binary] = frozenset()


@dataclass(frozen=True)
class Judgement:
    """Every vote on one pair key, and the binaries those votes resolve to.

    ``unreached_follow_ups`` names each follow-up no vote carries whose
    preconditions all resolve to the answers it needs, at least one of them
    fixed by a derived row. A rater row meeting those needs must give a
    decisive follow-up, so no rater reached it, and it resolves UNCLEAR.
    A decisive follow-up the rater votes alone ask, though no rater row met its
    needs, resolves OPEN instead: a further blind rater answering the whole
    item may reach it.
    """

    key: PairKey
    votes: tuple[Vote, ...]
    resolved: Mapping[Binary, Resolved]
    unreached_follow_ups: frozenset[Binary] = frozenset()

    def answer(self, binary: Binary) -> Resolved | None:
        """*binary*'s resolved answer; None only when no row carries it and no rule resolves it."""
        return self.resolved.get(binary)


def resolve_judgements(rows: Iterable[Mapping[str, Any]]) -> Mapping[PairKey, Judgement]:
    """Validate every verdict row and resolve each pair key's binaries.

    Votes on one binary come from the rows carrying it as a voting answer.
    UNCLEAR abstains, an all-UNCLEAR binary resolves UNCLEAR and an even split
    resolves TIED, a follow-up a derived row's precondition left unasked
    resolves UNCLEAR, and a decisive follow-up the rater votes ask but no rater
    row reached resolves OPEN. The ``current`` and ``attach_time`` bases are
    separate pair keys and never mix. A malformed row, a rater voting twice on one pair key,
    and a pair key whose rater answers and derived rows disagree about which
    binaries a derived row fixes are refused.
    """
    votes: defaultdict[PairKey, dict[str, tuple[int, Vote]]] = defaultdict(dict)
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
        votes[key][row['rater']] = (index, _vote(row))
    for key, indexed in votes.items():
        _refuse_mixed_votes(key, list(indexed.values()))
    return MappingProxyType({
        key: _judgement(key, tuple(vote for _, vote in votes[key].values())) for key in sorted(votes)
    })


def _vote(row: Mapping[str, Any]) -> Vote:
    non_consumed = frozenset(Binary(name) for name in row.get('non_consumed', ()))
    return Vote(
        rater=row['rater'],
        role=Role(row['role']),
        answers=MappingProxyType({
            binary: words(row[binary])
            for binary, words in _WORDS.items()
            if binary in row and binary not in non_consumed
        }),
        non_consumed=non_consumed,
    )


def _refuse_mixed_votes(key: PairKey, indexed: list[tuple[int, Vote]]) -> None:
    """Refuse a rater answer that votes beside a derived row, or a non_consumed one beside none."""
    fixed = {binary for _, vote in indexed if vote.role is Role.DERIVED for binary in vote.answers}
    pair = (key.entry_id, key.target_id, key.basis.value)
    for index, vote in indexed:
        for binary in sorted(vote.non_consumed - fixed):
            raise ValueError(
                f'verdict row {index} on pair {pair}: rater {vote.rater!r} lists non_consumed'
                f' {binary.value!r}, which no derived row of the pair fixes'
            )
        if vote.role is Role.DERIVED:
            continue
        for binary in sorted(vote.answers.keys() & fixed):
            raise ValueError(
                f'verdict row {index} on pair {pair}: rater {vote.rater!r} votes {binary.value!r},'
                ' which a derived row of the pair fixes, so the answer must be non_consumed'
            )


def _row_violation(row: Mapping[str, Any]) -> str | None:
    """The first rubric invariant *row* breaks, naming the offending value; None if none."""
    missing = [key for key in _REQUIRED_KEYS if key not in row]
    if missing:
        return f'lacks {", ".join(missing)}'
    for key in _IDENTITY_KEYS:
        if not isinstance(row[key], str) or not row[key]:
            return f'{key} {row[key]!r} is not a non-empty string'
    for key, words in (('basis', Basis), ('role', Role)):
        if not _is_word(row[key], words):
            return f'{key} {row[key]!r} is not one of {sorted(words)}'
    carried = [binary for binary in _WORDS if binary in row]
    if not carried:
        return f'carries no binary (one of {", ".join(_WORDS)})'
    for binary in carried:
        if not _is_word(row[binary], _WORDS[binary]):
            return f'{binary} {row[binary]!r} is not one of {sorted(_WORDS[binary])}'
    unclear = [binary for binary in carried if is_abstain(row[binary])]
    reason = row.get('unclear_reason')
    if unclear and not (isinstance(reason, str) and reason.strip()):
        return f'{unclear[0]} {ABSTAIN!r} needs a non-empty unclear_reason'
    for follow_up, needs in FOLLOW_UPS.items():
        if follow_up not in row:
            continue
        for asked_after, needed in needs.items():
            if row.get(asked_after) != needed:
                return (
                    f'{follow_up} {row[follow_up]!r} needs {asked_after} {needed.value!r},'
                    f' got {row.get(asked_after)!r}'
                )
    return (
        _skipped_follow_up_violation(row) or _evidence_violation(row, carried)
        or _non_consumed_violation(row)
    )


def _is_word(value: object, words: type[StrEnum]) -> bool:
    return isinstance(value, str) and value in set(words)


def _skipped_follow_up_violation(row: Mapping[str, Any]) -> str | None:
    """The first decisive follow-up a rater *row*'s own answers ask that it does not give; None if none."""
    if row['role'] not in RATER_ROLES:
        return None
    for follow_up in sorted(DECISIVE_FOLLOW_UPS):
        needs = FOLLOW_UPS[follow_up]
        if follow_up not in row and all(row.get(binary) == needed for binary, needed in needs.items()):
            asked_by = ' and '.join(f'{binary} {needed.value!r}' for binary, needed in needs.items())
            return f'lacks {follow_up}, which a rater row with {asked_by} must give'
    return None


def _evidence_violation(row: Mapping[str, Any], carried: list[Binary]) -> str | None:
    """The first answer *row* gives without the evidence that must back it; None if none."""
    required = [
        (binary, _EVIDENCE[binary, row[binary]])
        for binary in carried
        if row['role'] in RATER_ROLES and (binary, row[binary]) in _EVIDENCE
    ]
    if Binary.DIGEST in row:
        required.append((Binary.DIGEST, _LOSS_SPANS))
    for binary, evidence in required:
        if not evidence.valid(row.get(evidence.field)):
            return (
                f'{binary} {row[binary]!r} needs {evidence.field}, {evidence.shape},'
                f' got {row.get(evidence.field)!r}'
            )
    return None


def _non_consumed_violation(row: Mapping[str, Any]) -> str | None:
    """Why *row*'s non_consumed list is malformed, naming the offending binary; None if it is not."""
    if 'non_consumed' not in row:
        return None
    listed = row['non_consumed']
    if not isinstance(listed, list):
        return f'non_consumed {listed!r} is not a list of binary names'
    if row['role'] not in RATER_ROLES:
        return f'non_consumed {listed!r} is on a {row["role"]} row; only a rater answer is non_consumed'
    for position, name in enumerate(listed):
        if not isinstance(name, str) or name not in set(_WORDS):
            return f'non_consumed names {name!r}, not one of {", ".join(_WORDS)}'
        if name in listed[:position]:
            return f'non_consumed names {name!r} twice'
        if name not in row:
            return f'non_consumed names {name!r}, which the row does not carry'
    return None


def _judgement(key: PairKey, votes: tuple[Vote, ...]) -> Judgement:
    resolved = {
        binary: _RESOLVERS[binary](
            [vote.answers[binary] for vote in votes if binary in vote.answers], binary,
        )
        for binary in _WORDS
        if any(binary in vote.answers for vote in votes)
    }
    derived = {binary for vote in votes if vote.role is Role.DERIVED for binary in vote.answers}
    met = frozenset(
        follow_up for follow_up, needs in FOLLOW_UPS.items()
        if follow_up not in resolved
        and all(resolved.get(binary) == Resolved(needed) for binary, needed in needs.items())
    )
    unreached = frozenset(
        follow_up for follow_up in met if not derived.isdisjoint(FOLLOW_UPS[follow_up])
    )
    resolved.update(dict.fromkeys(unreached, Resolved.UNCLEAR))
    resolved.update(dict.fromkeys((met - unreached) & DECISIVE_FOLLOW_UPS, Resolved.OPEN))
    return Judgement(
        key=key, votes=votes, resolved=MappingProxyType(resolved), unreached_follow_ups=unreached,
    )


def _majority(words: list[StrEnum], binary: Binary) -> Resolved:
    decided = [word for word in words if not is_abstain(word)]
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


def _all_visible(words: list[StrEnum], binary: Binary) -> Resolved:
    """YES only when every decided vote is YES: a reader of the digest misses any span one rater needed."""
    decided = [word for word in words if not is_abstain(word)]
    if not decided:
        return Resolved.UNCLEAR
    positive, negative = _POLES[binary]
    return positive if all(word == positive for word in decided) else negative


_RESOLVERS: Mapping[Binary, Callable[[list[StrEnum], Binary], Resolved]] = MappingProxyType({
    binary: _all_visible if binary is Binary.DIGEST else _majority for binary in _WORDS
})
"""How each binary's votes resolve: DIGEST by the all-visible rule, every other by strict majority."""


def quote_found(quote: str, text: str) -> bool:
    """Whether *quote* occurs in *text* verbatim up to whitespace; an empty quote is never found."""
    needle = _normalised(quote)
    return bool(needle) and needle in _normalised(text)


def digest_visible(loss_spans: Sequence[str], child_text: str) -> bool:
    """Whether every span lies inside the digest a reader of the parent sees of *child_text*."""
    if not loss_spans:
        raise ValueError('digest_visible needs at least one loss span')
    digest = _digest(child_text)
    return all(quote_found(span, digest) for span in loss_spans)


def _normalised(text: str) -> str:
    return ' '.join(text.split())


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
    """Why a binary blocks: no row carries it; its votes split evenly; or it is a decisive
    follow-up the rater votes ask that no rater row reached."""

    UNRATED = 'unrated'
    TIED = 'tied'
    OPEN = 'open'


@dataclass(frozen=True, order=True)
class Gap:
    """A binary the derivation needed on one record that the verdicts leave unrated, tied or open."""

    target_id: str
    binary: Binary
    kind: GapKind


_BLOCKING: Mapping[Resolved, GapKind] = MappingProxyType({
    Resolved.TIED: GapKind.TIED, Resolved.OPEN: GapKind.OPEN,
})
"""Each resolution that blocks a derivation asking for it, and the gap it is."""


@dataclass(frozen=True)
class Decided:
    """A write's cells under triage and under the status quo (every write stored flat).

    ``conflict`` says whether the write disputes its parent or any earlier
    record. ``m1_hidden`` is None outside the M1 band.
    ``m1_hidden_as_amended`` is the counterfactual M1 of a band attach read as
    if it were amended; None outside the band and for a stored write.
    ``consumed`` names every (target, binary) the derivation read.
    """

    triage_cells: frozenset[Cell]
    status_quo_cells: frozenset[Cell]
    m1_hidden: bool | None
    conflict: bool
    m1_hidden_as_amended: bool | None
    consumed: frozenset[tuple[str, Binary]]


@dataclass(frozen=True)
class Unclear:
    """A binary the derivation needed resolved UNCLEAR: every rater abstained, or none was asked it."""

    binaries: tuple[tuple[str, Binary], ...]


@dataclass(frozen=True)
class Incomplete:
    """The verdicts leave binaries the derivation needed unrated, tied or open."""

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
    decisive unrated, tied or open binary makes the write Incomplete (naming every
    gap the derivations met), a decisive UNCLEAR one makes it Unclear, and a
    binary never asked never blocks. The triage, status-quo, M1 and
    counterfactual-M1 derivations stand or fall together: a gap or an UNCLEAR
    in any one of them decides the whole write, even when the others are
    complete.
    ``supersedes_target`` and ``superseded_marks`` are metadata facts
    (ruling 5c), never rater answers.
    """
    asked: set[tuple[str, Binary]] = set()
    regimes = _regimes(
        outcome, judgements_by_target, supersedes_target=supersedes_target,
        superseded_marks=superseded_marks, in_m1_band=in_m1_band, asked=asked,
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
        m1_hidden_as_amended=results['m1_as_amended'][0] if regimes['m1_as_amended'] else None,
        consumed=frozenset(asked),
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
    """Lazy access to one record's judgement; ``marked`` = it carries a superseded-by mark.

    Every binary asked is recorded in ``asked``, the one derive call's set.
    """

    target_id: str
    judgement: Judgement | None
    asked: set[tuple[str, Binary]]
    marked: bool = False

    def yes(self, binary: Binary) -> bool:
        self.asked.add((self.target_id, binary))
        resolved = None if self.judgement is None else self.judgement.answer(binary)
        if resolved is None:
            raise _Undecided(Gap(self.target_id, binary, GapKind.UNRATED))
        if resolved in _BLOCKING:
            raise _Undecided(Gap(self.target_id, binary, _BLOCKING[resolved]))
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
    asked: set[tuple[str, Binary]],
) -> dict[str, list[Callable[[], Any]]]:
    """The deferred derivations of each regime, refusing a write the rubric cannot read."""
    rule = _CELL_RULES.get(outcome)
    if rule is None:
        raise ValueError(f'outcome {outcome!r} is not one of {sorted(TRIAGE_OUTCOMES)}')
    readers = [
        _Reader(target, judgement, asked, marked=target in superseded_marks)
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
        'm1_as_amended': (
            [partial(_m1_hidden, OUTCOME_AMENDED, readers)]
            if reads_as_amended(outcome, in_m1_band=in_m1_band) else []
        ),
    }


def reads_as_amended(outcome: str, *, in_m1_band: bool) -> bool:
    """Whether :func:`derive` gives a write a counterfactual M1 as amended: a band attach."""
    return in_m1_band and outcome != OUTCOME_STORED
