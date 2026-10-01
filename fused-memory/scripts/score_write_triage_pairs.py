#!/usr/bin/env python3
"""Score write-triage arms by the pair each one attached a write to.

PRD ``plans/write-triage-flip-readiness-prd.md`` §11 C2''. This module is the
one definition of the flip-gate metrics. An *arm* is one judge configuration
run over a frozen population of writes. For every write the arm attached, the
pair (write, the record its judge named) is looked up in a blind verdict
corpus, and the arm is scored on what the raters said about that pair.

It is not ``eval_write_triage_judge.py::score_attachments``. That function
asks whether the attach target is the fixture's canonical record, a
fixture-strict hit rate that feeds the judge-accuracy report. This module asks
whether raters agree the write belongs under the record the judge chose. They
are different quantities, so each has its own definition (INV-9).

Verdict corpus
--------------
One row per (rater, pair): ``{entry_id, target_id, verdict, rater, batch}``,
plus any extra keys (``reason``). ``verdict`` is one of :data:`VERDICT_WORDS`,
the rater brief's vocabulary. A pair may carry several raters' votes. It
resolves to a :class:`Resolution` by majority on each of two binaries,
belongs-vs-misfile first and then CORRECTS-vs-not, with UNCLEAR abstaining.

Cases and metrics
-----------------
:func:`score_pairs` states the per-case row contract and defines every metric
it reports. Only judge-band writes are scored. A write in a deterministic band
was attached to its band winner before any judge ran, so it is the same in
every arm and no rater is asked about it.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeAlias

from fused_memory.server.write_triage import (
    OUTCOME_CONTESTED,
    OUTCOME_JUDGE,
    OUTCOME_STORED,
    TRIAGE_OUTCOMES,
)

VERDICT_WORDS: tuple[str, ...] = (
    'SAME', 'EXTENDS', 'SUBSUMED', 'CORRECTS', 'RELATED', 'UNRELATED', 'UNCLEAR',
)
_CORRECTS = 'CORRECTS'
_MISFILE_WORDS = frozenset({'RELATED', 'UNRELATED'})
_ABSTAIN = 'UNCLEAR'
_REQUIRED_VOTE_KEYS = ('entry_id', 'target_id', 'verdict', 'rater', 'batch')
_REQUIRED_CASE_KEYS = ('arm', 'memory_id', 'band', 'outcome')
_ATTACH_OUTCOMES = TRIAGE_OUTCOMES - {OUTCOME_STORED}
_WILSON_Z = 1.96

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
    votes: defaultdict[Pair, dict[str, str]] = defaultdict(dict)
    for index, row in enumerate(rows):
        pair, rater, word = _vote(index, row)
        if rater in votes[pair]:
            raise ValueError(
                f'verdict row {index}: rater {rater!r} already voted on pair {pair}'
            )
        votes[pair][rater] = word
    return {pair: _majority(by_rater.values()) for pair, by_rater in votes.items()}


def _vote(index: int, row: Mapping[str, Any]) -> tuple[Pair, str, str]:
    pair = (row.get('entry_id'), row.get('target_id'))
    missing = [key for key in _REQUIRED_VOTE_KEYS if key not in row]
    if missing:
        raise ValueError(f'verdict row {index} on pair {pair} lacks {", ".join(missing)}')
    if row['verdict'] not in VERDICT_WORDS:
        raise ValueError(
            f'verdict row {index} on pair {pair}: {row["verdict"]!r} is not one of {VERDICT_WORDS}'
        )
    return (row['entry_id'], row['target_id']), row['rater'], row['verdict']


def _majority(words: Iterable[str]) -> Resolution:
    decided = [word for word in words if word != _ABSTAIN]
    if not decided:
        return Resolution.UNCLEAR
    misfile = _binary(sum(word in _MISFILE_WORDS for word in decided), len(decided))
    if misfile is None:
        return Resolution.TIED
    if misfile:
        return Resolution.MISFILE
    corrects = _binary(sum(word == _CORRECTS for word in decided), len(decided))
    if corrects is None:
        return Resolution.TIED
    return Resolution.CORRECTS if corrects else Resolution.AGREEING


def _binary(yes: int, votes: int) -> bool | None:
    """Strict majority of *votes*: True for yes, False for no, None for a tie."""
    if 2 * yes == votes:
        return None
    return 2 * yes > votes


@dataclass(frozen=True)
class JudgedCase:
    """One arm's answer for one write: a validated row of :func:`score_pairs`' *cases*."""

    arm: str
    memory_id: str
    band: str
    outcome: str
    judged_candidate_id: str | None
    band_winner_id: str | None
    parse_failure: bool
    judge_seconds: float | None
    judge_model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> JudgedCase:
        where = f'case row arm={row.get("arm")!r} memory_id={row.get("memory_id")!r}'
        missing = [key for key in _REQUIRED_CASE_KEYS if key not in row]
        if missing:
            raise ValueError(f'{where} lacks {", ".join(missing)}')
        if row['outcome'] not in TRIAGE_OUTCOMES:
            raise ValueError(
                f'{where}: outcome {row["outcome"]!r} is not one of {sorted(TRIAGE_OUTCOMES)}'
            )
        usage = row.get('usage') or {}
        return cls(
            arm=row['arm'],
            memory_id=row['memory_id'],
            band=row['band'],
            outcome=row['outcome'],
            judged_candidate_id=row.get('judged_candidate_id'),
            band_winner_id=row.get('band_winner_id'),
            parse_failure=bool(row.get('parse_failure')),
            judge_seconds=row.get('judge_seconds'),
            judge_model=row.get('judge_model'),
            prompt_tokens=usage.get('prompt_tokens'),
            completion_tokens=usage.get('completion_tokens'),
        )

    @property
    def in_judge_band(self) -> bool:
        return self.band == OUTCOME_JUDGE

    @property
    def attached(self) -> bool:
        return self.outcome in _ATTACH_OUTCOMES

    @property
    def pair(self) -> Pair | None:
        """The pair the judge attached, or None when it named no record."""
        if self.judged_candidate_id is None:
            return None
        return (self.memory_id, self.judged_candidate_id)


@dataclass(frozen=True)
class ScoredCase:
    """A judge-band case read against the corpus; the one definition of each per-write error.

    ``resolution`` is the corpus's resolution of the attached pair, or None when
    the arm attached nothing. ``has_true_link`` says whether the corpus resolves
    any pair from this write as belongs, whichever arm named it.
    """

    case: JudgedCase
    resolution: Resolution | None
    has_true_link: bool

    @property
    def decided(self) -> bool:
        return self.resolution in (Resolution.AGREEING, Resolution.CORRECTS, Resolution.MISFILE)

    @property
    def misfile(self) -> bool:
        return self.resolution is Resolution.MISFILE

    @property
    def contested(self) -> bool:
        return self.case.outcome == OUTCOME_CONTESTED

    @property
    def contested_decision_error(self) -> bool:
        return self.decided and (self.resolution is Resolution.CORRECTS) != self.contested

    @property
    def missed_contradiction(self) -> bool:
        return self.resolution is Resolution.CORRECTS and not self.contested

    @property
    def false_contested(self) -> bool:
        return self.resolution is Resolution.AGREEING and self.contested

    @property
    def answered_distinct(self) -> bool:
        return self.case.outcome == OUTCOME_STORED and not self.case.parse_failure

    @property
    def true_link_answered_distinct(self) -> bool:
        return self.has_true_link and self.answered_distinct


def score_pairs(
    cases: Iterable[Mapping[str, Any]],
    verdicts: Sequence[Mapping[str, Any]],
    *,
    reference_arm: str,
) -> dict[str, Any]:
    """Score every arm in *cases* against the verdict corpus *verdicts*.

    Each row of *cases* is one arm's answer for one write. Extra keys are ignored.

    - ``arm``: the configuration that produced the row.
    - ``memory_id``: the write.
    - ``band``: ``write_triage.py::OUTCOME_JUDGE`` when the write landed in the
      middle band and the judge answered. Any other band was decided
      deterministically, and its rows are counted but not scored.
    - ``outcome``: one of ``write_triage.py::TRIAGE_OUTCOMES``.
    - ``judged_candidate_id``: the record the judge's verdict names, hoisted to
      the canonical record production attaches to
      (``write_triage.py::_canonical_id_of``, task 6007). None when nothing
      was attached.
    - ``band_winner_id``: the write's nearest neighbour on the frozen slate.
    - ``parse_failure``: the judge's reply could not be parsed, and triage
      failed open to ``stored``.
    - ``judge_seconds``, ``judge_model``, ``usage.prompt_tokens``,
      ``usage.completion_tokens``: the judge call's latency, model and tokens.

    Per arm, ``population`` counts all rows (``n_cases``) and judge-band rows
    (``n_judge_band``). ``quality`` is computed over the judge band only, and
    a rate is None, never 0.0, when its denominator is 0:

    - ``attaches``: writes attached to a record (outcome is not ``stored``).
    - ``unclear_attaches``: attaches whose pair resolved UNCLEAR. They are left
      out of every rate's denominator.
    - ``misfiles``: attaches whose pair resolved MISFILE.
    - ``misfile_rate_of_attaches``, ``misfile_rate_wilson95``: misfiles over
      decided attaches (attaches minus unclear_attaches), with the Wilson 95%
      interval.
    - ``duplicates_placed``: attaches whose pair resolved as belongs
      (AGREEING or CORRECTS).
    - ``contradictions_attached``, ``contradictions_contested``,
      ``contradiction_recall``: attaches resolved CORRECTS, how many of them
      were filed contested, and that ratio.
    - ``agreeing_attaches``, ``agreeing_contested``, ``false_contested_rate``:
      attaches resolved AGREEING, how many of them were filed contested, and
      that ratio.
    - ``true_links``: writes the corpus resolves as belongs under some record,
      whichever arm named the pair.
    - ``true_links_answered_distinct``, ``true_links_answered_distinct_rate``:
      true links this arm answered ``stored`` without a parse failure, and that
      count over true_links.
    - ``contested_decision_errors``: decided attaches where "resolved CORRECTS"
      and "filed contested" disagree.
    - ``parse_failures``: judge replies that could not be parsed.
    - ``unrated_pairs``, ``unrated_pair_ids``: pairs this arm named that the
      corpus does not rate.
    """
    cases_by_arm = _cases_by_arm(cases)
    if reference_arm not in cases_by_arm:
        raise ValueError(
            f'reference arm {reference_arm!r} is not among the arms {sorted(cases_by_arm)}'
        )
    truth = resolve_verdicts(verdicts)
    true_link_writes = frozenset(entry for (entry, _), r in truth.items() if r.belongs)
    return {
        'reference_arm': reference_arm,
        'verdict_corpus': _corpus_summary(verdicts, truth),
        'arms': {
            arm: _arm_report(arm_cases, truth, true_link_writes)
            for arm, arm_cases in cases_by_arm.items()
        },
    }


def _cases_by_arm(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[JudgedCase]]:
    by_arm: defaultdict[str, list[JudgedCase]] = defaultdict(list)
    for row in rows:
        case = JudgedCase.from_row(row)
        by_arm[case.arm].append(case)
    return {arm: by_arm[arm] for arm in sorted(by_arm)}


def _arm_report(
    cases: Sequence[JudgedCase],
    truth: Mapping[Pair, Resolution],
    true_link_writes: frozenset[str],
) -> dict[str, Any]:
    judged = [case for case in cases if case.in_judge_band]
    scored = [_scored(case, truth, true_link_writes) for case in judged]
    return {
        'population': {'n_cases': len(cases), 'n_judge_band': len(judged)},
        'quality': _quality(scored, _unrated(judged, truth)),
    }


def _scored(
    case: JudgedCase, truth: Mapping[Pair, Resolution], true_link_writes: frozenset[str],
) -> ScoredCase:
    pair = case.pair
    return ScoredCase(
        case=case,
        resolution=truth.get(pair) if case.attached and pair is not None else None,
        has_true_link=case.memory_id in true_link_writes,
    )


def _unrated(judged: Iterable[JudgedCase], truth: Mapping[Pair, Resolution]) -> list[Pair]:
    named = {case.pair for case in judged if case.attached}
    return sorted(pair for pair in named if pair is not None and pair not in truth)


def _corpus_summary(
    verdicts: Sequence[Mapping[str, Any]], truth: Mapping[Pair, Resolution],
) -> dict[str, Any]:
    resolved = Counter(truth.values())
    return {
        'pairs': len(truth),
        'ratings': len(verdicts),
        'raters': sorted({row['rater'] for row in verdicts}),
        'resolutions': {member.value: resolved[member] for member in Resolution},
    }


def _quality(scored: Sequence[ScoredCase], unrated: Sequence[Pair]) -> dict[str, Any]:
    attached = [s for s in scored if s.case.attached]
    decided = sum(s.decided for s in attached)
    misfiles = sum(s.misfile for s in attached)
    contradictions = sum(s.resolution is Resolution.CORRECTS for s in attached)
    contradictions_contested = contradictions - sum(s.missed_contradiction for s in scored)
    agreeing = sum(s.resolution is Resolution.AGREEING for s in attached)
    agreeing_contested = sum(s.false_contested for s in scored)
    true_links = sum(s.has_true_link for s in scored)
    answered_distinct = sum(s.true_link_answered_distinct for s in scored)
    return {
        'attaches': len(attached),
        'unclear_attaches': sum(s.resolution is Resolution.UNCLEAR for s in attached),
        'misfiles': misfiles,
        'misfile_rate_of_attaches': _rate(misfiles, decided),
        'misfile_rate_wilson95': _wilson95(misfiles, decided),
        'duplicates_placed': contradictions + agreeing,
        'contradictions_attached': contradictions,
        'contradictions_contested': contradictions_contested,
        'contradiction_recall': _rate(contradictions_contested, contradictions),
        'agreeing_attaches': agreeing,
        'agreeing_contested': agreeing_contested,
        'false_contested_rate': _rate(agreeing_contested, agreeing),
        'true_links': true_links,
        'true_links_answered_distinct': answered_distinct,
        'true_links_answered_distinct_rate': _rate(answered_distinct, true_links),
        'contested_decision_errors': sum(s.contested_decision_error for s in scored),
        'parse_failures': sum(s.case.parse_failure for s in scored),
        'unrated_pairs': len(unrated),
        'unrated_pair_ids': [list(pair) for pair in unrated],
    }


def _rate(hits: int, total: int) -> float | None:
    """*hits*/*total*, or None when nothing was measured; never 0.0 for an empty denominator."""
    return round(hits / total, 4) if total else None


def _wilson95(hits: int, total: int) -> list[float] | None:
    if not total:
        return None
    share = hits / total
    spread = _WILSON_Z * _WILSON_Z / total
    centre = (share + spread / 2) / (1 + spread)
    half = _WILSON_Z * math.sqrt(share * (1 - share) / total + spread / (4 * total)) / (1 + spread)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]
