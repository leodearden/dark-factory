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

Refusal
-------
Every pair that any arm's judge attached a write to must resolve to something
other than TIED. Otherwise :func:`score_pairs` raises
:class:`IncompleteCorpusError`, naming each unrated or tied pair and the arms
that named it, and scores nothing, because a partial corpus must not read as
a result. A write that no arm attached needs no rating (D14). Neither does a
corpus pair that no arm named.

Usage
-----
Run from ``fused-memory/``. This scores the committed seed, the gpt-4o-mini
09-29 reference runs at slate widths 5 and 20::

    uv run python scripts/score_write_triage_pairs.py \
        --cases tests/fixtures/write_triage_pair_cases_seed.jsonl \
        --verdicts tests/fixtures/write_triage_pair_verdicts_seed.jsonl \
        --reference-arm gpt-4o-mini@5 --out /tmp/write-triage-pairs.json

``--cases`` and ``--verdicts`` each take one or more JSONL files. The verdict
corpus that new ratings land in is ``calibration/write_triage_pair_verdicts.jsonl``.
The report is printed to stdout, and also written to ``--out`` when given. An
incomplete corpus exits 1, prints one stderr line per unrated or tied pair,
and writes nothing.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

from fused_memory.server.write_triage import (
    OUTCOME_CONTESTED,
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
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
_BANDS = frozenset({OUTCOME_RESTATED, OUTCOME_JUDGE, OUTCOME_STORED})
"""Every band ``write_triage.py::decide_band`` routes a write to."""
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


@dataclass(frozen=True, order=True)
class JudgedPair:
    """A pair some arm's judge attached a write to, with every arm that named it."""

    entry_id: str
    target_id: str
    arms: tuple[str, ...]


class IncompleteCorpusError(ValueError):
    """The verdict corpus leaves judge-named pairs unrated or tied, so nothing is scored."""

    def __init__(self, unrated: tuple[JudgedPair, ...], tied: tuple[JudgedPair, ...]) -> None:
        self.unrated = unrated
        self.tied = tied
        header = (
            f'verdict corpus is incomplete: {len(unrated)} unrated and {len(tied)} tied'
            ' judge-named pairs'
        )
        lines = [_describe_pair('unrated', pair) for pair in unrated]
        lines += [_describe_pair('tied', pair) for pair in tied]
        super().__init__('\n'.join([header, *lines]))


def _describe_pair(kind: str, pair: JudgedPair) -> str:
    return f'{kind} {pair.entry_id} {pair.target_id} named by {", ".join(pair.arms)}'


def _case_label(arm: object, memory_id: object) -> str:
    return f'case row arm={arm!r} memory_id={memory_id!r}'


@dataclass(frozen=True)
class JudgedCase:
    """One arm's answer for one write: a row of :func:`score_pairs`' *cases*.

    Construction refuses a row the shipped judge could not have produced.
    """

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

    def __post_init__(self) -> None:
        violation = self._contract_violation()
        if violation is not None:
            raise ValueError(f'{_case_label(self.arm, self.memory_id)}: {violation}')

    def _contract_violation(self) -> str | None:
        if self.outcome not in TRIAGE_OUTCOMES:
            return f'outcome {self.outcome!r} is not one of {sorted(TRIAGE_OUTCOMES)}'
        if self.band not in _BANDS:
            return f'band {self.band!r} is not one of {sorted(_BANDS)}'
        if not self.attached and self.judged_candidate_id is not None:
            return f'a {self.outcome!r} write names judged_candidate_id {self.judged_candidate_id!r}'
        if not self.in_judge_band:
            return None
        if self.attached and self.judged_candidate_id is None:
            return (
                f'a judge-band {self.outcome!r} attach names no judged_candidate_id, a verdict'
                ' write_triage_judge.py::parse_judge_verdict refuses'
            )
        if self.band_winner_id is None:
            return 'a judge-band write has no band_winner_id to group it by'
        return None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> JudgedCase:
        missing = [key for key in _REQUIRED_CASE_KEYS if key not in row]
        if missing:
            raise ValueError(
                f'{_case_label(row.get("arm"), row.get("memory_id"))} lacks {", ".join(missing)}'
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
    def routing(self) -> tuple[str, str | None]:
        """What the frozen slate decided before any judge ran: the band and its winner."""
        return (self.band, self.band_winner_id)

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


PAIRED_ERRORS: Mapping[str, Callable[[ScoredCase], bool]] = MappingProxyType({
    'misfile': lambda scored: scored.misfile,
    'contested_decision_error': lambda scored: scored.contested_decision_error,
    'missed_contradiction': lambda scored: scored.missed_contradiction,
    'false_contested': lambda scored: scored.false_contested,
    'true_link_answered_distinct': lambda scored: scored.true_link_answered_distinct,
})
"""The per-write errors each arm is paired against the reference arm on."""


@dataclass(frozen=True)
class ListPrice:
    usd_per_mtok_input: float
    usd_per_mtok_output: float

    def usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.usd_per_mtok_input
            + completion_tokens * self.usd_per_mtok_output
        ) / 1e6


LIST_PRICES_AS_OF = '2026-09-30'
LIST_PRICES_SOURCE = 'https://developers.openai.com/api/docs/pricing'
LIST_PRICES: Mapping[str, ListPrice] = MappingProxyType({
    'gpt-4o-mini': ListPrice(0.15, 0.60),
    'gpt-6-luna': ListPrice(0.10, 0.50),
    'gpt-5.6-luna': ListPrice(0.20, 1.20),
    'gpt-5.6-terra': ListPrice(2.00, 12.00),
    'gpt-6.1-sol': ListPrice(2.00, 10.00),
    'gpt-5.6-sol': ListPrice(4.00, 20.00),
    'gpt-6-astra': ListPrice(10.00, 50.00),
})
"""USD per million tokens at list price, as of :data:`LIST_PRICES_AS_OF`.

Completion tokens include reasoning tokens, which the chat and Responses APIs
both bill as output. No cached-input discount is applied, so a cost is an
upper bound at list price. A model missing here prices as None, never as a
guess, so refresh the table and its date when an arm records a new model.
"""


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
      corpus does not rate. They are always 0 and empty, because an unrated
      pair raises :class:`IncompleteCorpusError` instead.

    ``runtime`` covers the same judge-band rows, one judge call each. A figure
    that some call cannot support is None, never computed over the rest:

    - ``judge_calls``: judge-band rows.
    - ``untimed_calls``, ``p50_judge_seconds``, ``p95_judge_seconds``: calls
      with no ``judge_seconds``, and the nearest-rank latency percentiles.
    - ``judge_models``: the distinct models that answered.
    - ``unpriced_calls``, ``cost_per_write_usd``: calls with no token usage or
      a model missing from :data:`LIST_PRICES`, and the mean list-price cost
      per call.

    ``paired_vs_reference`` is None for the reference arm. Any other arm is
    paired with it on the writes both judge bands hold (``n_common``). For each
    error in :data:`PAIRED_ERRORS` it counts the writes only this arm got wrong
    (``only_arm``) and only the reference got wrong (``only_reference``), with
    the two-sided exact McNemar p. ``parent_sign_test`` runs the same exact
    test over parent groups. A write's parent group is its band winner, which
    is held fixed across arms because the band is decided before the judge
    runs. A group favours whichever arm erred on fewer of its writes.

    Every arm must have been run on the reference arm's frozen population: a
    write both arms hold whose band or band winner differs between them is
    refused, whichever band it is in.

    ``list_prices`` is the table those costs were computed from, with its date
    and source.
    """
    cases_by_arm = _cases_by_arm(cases)
    if reference_arm not in cases_by_arm:
        raise ValueError(
            f'reference arm {reference_arm!r} is not among the arms {sorted(cases_by_arm)}'
        )
    _refuse_unshared_routing(cases_by_arm, reference_arm)
    truth = resolve_verdicts(verdicts)
    unrated, tied = _incomplete(_judged_pairs(cases_by_arm), truth)
    if unrated or tied:
        raise IncompleteCorpusError(unrated, tied)
    true_link_writes = frozenset(entry for (entry, _), r in truth.items() if r.belongs)
    scored = {
        arm: [_scored(case, truth, true_link_writes) for case in arm_cases if case.in_judge_band]
        for arm, arm_cases in cases_by_arm.items()
    }
    return {
        'reference_arm': reference_arm,
        'verdict_corpus': _corpus_summary(verdicts, truth),
        'list_prices': _price_table(),
        'arms': {
            arm: _arm_report(
                cases_by_arm[arm],
                scored[arm],
                None if arm == reference_arm else scored[reference_arm],
            )
            for arm in cases_by_arm
        },
    }


def _cases_by_arm(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[JudgedCase]]:
    by_arm: defaultdict[str, dict[str, JudgedCase]] = defaultdict(dict)
    for row in rows:
        case = JudgedCase.from_row(row)
        if case.memory_id in by_arm[case.arm]:
            raise ValueError(f'{_case_label(case.arm, case.memory_id)} appears twice')
        by_arm[case.arm][case.memory_id] = case
    return {arm: list(by_arm[arm].values()) for arm in sorted(by_arm)}


def _refuse_unshared_routing(
    cases_by_arm: Mapping[str, Sequence[JudgedCase]], reference_arm: str,
) -> None:
    reference_by_write = {case.memory_id: case for case in cases_by_arm[reference_arm]}
    for case in chain.from_iterable(cases_by_arm.values()):
        theirs = reference_by_write.get(case.memory_id)
        if theirs is not None and case.routing != theirs.routing:
            raise ValueError(
                f'write {case.memory_id!r} has band {case.band!r} and band winner'
                f' {case.band_winner_id!r} in arm {case.arm!r} but band {theirs.band!r} and'
                f' band winner {theirs.band_winner_id!r} in reference arm {reference_arm!r};'
                ' the arms were not run on one frozen population'
            )


def _judged_pairs(cases_by_arm: Mapping[str, Sequence[JudgedCase]]) -> dict[Pair, tuple[str, ...]]:
    named_by: defaultdict[Pair, set[str]] = defaultdict(set)
    for arm, cases in cases_by_arm.items():
        for case in cases:
            if case.in_judge_band and case.pair is not None:
                named_by[case.pair].add(arm)
    return {pair: tuple(sorted(arms)) for pair, arms in named_by.items()}


def _incomplete(
    judged_pairs: Mapping[Pair, tuple[str, ...]], truth: Mapping[Pair, Resolution],
) -> tuple[tuple[JudgedPair, ...], tuple[JudgedPair, ...]]:
    named = sorted(JudgedPair(entry, target, arms) for (entry, target), arms in judged_pairs.items())
    unrated = tuple(p for p in named if (p.entry_id, p.target_id) not in truth)
    tied = tuple(p for p in named if truth.get((p.entry_id, p.target_id)) is Resolution.TIED)
    return unrated, tied


def _arm_report(
    cases: Sequence[JudgedCase],
    scored: Sequence[ScoredCase],
    reference: Sequence[ScoredCase] | None,
) -> dict[str, Any]:
    judged = [s.case for s in scored]
    return {
        'population': {'n_cases': len(cases), 'n_judge_band': len(judged)},
        'quality': _quality(scored),
        'runtime': _runtime(judged),
        'paired_vs_reference': None if reference is None else _paired(scored, reference),
    }


def _scored(
    case: JudgedCase, truth: Mapping[Pair, Resolution], true_link_writes: frozenset[str],
) -> ScoredCase:
    pair = case.pair
    return ScoredCase(
        case=case,
        resolution=None if pair is None else truth[pair],
        has_true_link=case.memory_id in true_link_writes,
    )


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


def _quality(scored: Sequence[ScoredCase]) -> dict[str, Any]:
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
        'unrated_pairs': 0,
        'unrated_pair_ids': [],
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


def _runtime(calls: Sequence[JudgedCase]) -> dict[str, Any]:
    seconds = sorted(call.judge_seconds for call in calls if call.judge_seconds is not None)
    costs = [cost for cost in map(_call_cost, calls) if cost is not None]
    timed = bool(calls) and len(seconds) == len(calls)
    priced = bool(calls) and len(costs) == len(calls)
    return {
        'judge_calls': len(calls),
        'untimed_calls': len(calls) - len(seconds),
        'p50_judge_seconds': round(_nearest_rank(seconds, 50), 3) if timed else None,
        'p95_judge_seconds': round(_nearest_rank(seconds, 95), 3) if timed else None,
        'judge_models': sorted({call.judge_model for call in calls if call.judge_model}),
        'unpriced_calls': len(calls) - len(costs),
        'cost_per_write_usd': round(sum(costs) / len(costs), 6) if priced else None,
    }


def _nearest_rank(sorted_values: Sequence[float], percent: int) -> float:
    """The smallest observed value with at least *percent*% of values at or below it."""
    rank = -(-percent * len(sorted_values) // 100)
    return sorted_values[rank - 1]


def _call_cost(call: JudgedCase) -> float | None:
    price = LIST_PRICES.get(call.judge_model) if call.judge_model else None
    if price is None or call.prompt_tokens is None or call.completion_tokens is None:
        return None
    return price.usd(call.prompt_tokens, call.completion_tokens)


def _price_table() -> dict[str, Any]:
    return {
        'as_of': LIST_PRICES_AS_OF,
        'source': LIST_PRICES_SOURCE,
        'usd_per_million_tokens': {
            model: {'input': price.usd_per_mtok_input, 'output': price.usd_per_mtok_output}
            for model, price in LIST_PRICES.items()
        },
    }


def _paired(
    arm: Sequence[ScoredCase], reference: Sequence[ScoredCase],
) -> dict[str, dict[str, Any]]:
    reference_by_write = {s.case.memory_id: s for s in reference}
    common = [
        (mine, reference_by_write[mine.case.memory_id])
        for mine in sorted(arm, key=lambda s: s.case.memory_id)
        if mine.case.memory_id in reference_by_write
    ]
    return {name: _paired_error(common, error) for name, error in PAIRED_ERRORS.items()}


def _paired_error(
    common: Sequence[tuple[ScoredCase, ScoredCase]], error: Callable[[ScoredCase], bool],
) -> dict[str, Any]:
    only_arm = [mine.case.band_winner_id for mine, theirs in common
                if error(mine) and not error(theirs)]
    only_reference = [mine.case.band_winner_id for mine, theirs in common
                      if error(theirs) and not error(mine)]
    net_reference_errors = Counter(only_reference)
    net_reference_errors.subtract(only_arm)
    favouring_arm = sum(net > 0 for net in net_reference_errors.values())
    favouring_reference = sum(net < 0 for net in net_reference_errors.values())
    return {
        'n_common': len(common),
        'only_arm': len(only_arm),
        'only_reference': len(only_reference),
        'mcnemar_p': _mcnemar_exact(len(only_arm), len(only_reference)),
        'parent_sign_test': {
            'groups_favouring_arm': favouring_arm,
            'groups_favouring_reference': favouring_reference,
            'p': _mcnemar_exact(favouring_arm, favouring_reference),
        },
    }


def _mcnemar_exact(only_first: int, only_second: int) -> float:
    """Two-sided exact McNemar (sign) test p on the discordant counts."""
    discordant = only_first + only_second
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(only_first, only_second) + 1))
    return round(min(1.0, 2 * tail / 2 ** discordant), 4)


def _read_jsonl(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as lines:
            for number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{number}: not a JSON object line ({exc.msg})') from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f'{path}:{number}: not a JSON object line (got {type(row).__name__})'
                    )
                rows.append(row)
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score write-triage arms by the pairs their judges attached writes to, read"
            " against a blind verdict corpus (flip-readiness PRD C2''). The corpus new"
            ' ratings land in is fused-memory/calibration/write_triage_pair_verdicts.jsonl;'
            ' pass it explicitly.'
        ),
    )
    parser.add_argument(
        '--cases', type=Path, nargs='+', required=True, metavar='PATH',
        help='JSONL case rows, in one or more files; every row names its arm',
    )
    parser.add_argument(
        '--verdicts', type=Path, nargs='+', required=True, metavar='PATH',
        help='JSONL verdict files, read together as one corpus',
    )
    parser.add_argument(
        '--reference-arm', required=True, metavar='NAME',
        help='the arm every other arm is paired against',
    )
    parser.add_argument('--out', type=Path, metavar='PATH', help='also write the report here')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = score_pairs(
            _read_jsonl(args.cases),
            _read_jsonl(args.verdicts),
            reference_arm=args.reference_arm,
        )
    except IncompleteCorpusError as refusal:
        print(refusal, file=sys.stderr)
        return 1
    report = json.dumps(result, indent=2)
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
