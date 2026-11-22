#!/usr/bin/env python3
"""Judge each call site of task 9945's luna migration eval against its pre-registered bar.

This module is the single code home of those bars: design.md §2 cites the
constants below by name. Every site judges one candidate arm against the
control, on the fixtures both arms ran, by a paired non-inferiority interval.
Only discordant pairs move the difference: b pairs only the candidate got
right, c pairs only the control got right. The extraction site is judged
instead over two population_weighted blocks, as
replay_overreach_extraction.score_replay hands them to its ``decide``.
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_ruling_overreach import WILSON_Z
from replay_overreach_extraction import CONTROL, HOLDING_RETENTION_MIN, RecordKind

# --------------------------------------------------------------------------- #
# The pre-registered bars (design.md §2), every one inclusive
# --------------------------------------------------------------------------- #

EXTRACTION_OVERREACH_MARGIN = 0.05
"""E1: the candidate's population-weighted overreach-per-minted upper 95% bound is at most the
control's point plus this. E2 is HOLDING_RETENTION_MIN, the replay's own holdings/day bar."""

RESOLVE_ACCURACY_MARGIN = 0.05
"""R1: the paired merge-accuracy difference's lower bound is at least minus this."""

RESOLVE_FALSE_MERGE_MARGIN = 0.03
"""R2: the paired no-false-merge difference's lower bound is at least minus this. A false merge
silently destroys a distinct fact, so it has its own, tighter guard."""

SUMMARY_FIDELITY_MARGIN = 0.10
"""S1: the paired faithful-rate difference's lower bound is at least minus this."""

ROUTING_LABEL_MARGIN = 0.05
"""C1 (classifier) and Q1 (router): the paired label-accuracy difference's lower bound is at least
minus this. The routing layer passes only if both pass."""

LUNA_FAILURE_SHARE_MAX = 0.02
"""A site whose candidate arm still has failure lines on more than this share of its fixtures fails,
whatever its statistic."""

SHOWN_DECIMALS = 4
"""Intervals are reported, and their bars decided, at this precision, so every check can be
re-derived from the report's own numbers."""


def _shown(value: float) -> float:
    return round(value, SHOWN_DECIMALS)


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def paired_noninferiority(b: int, c: int, n: int, margin: float) -> dict[str, Any]:
    """The paired difference ``(b - c) / n`` with its Wald 95% interval, and whether its lower bound
    is at least ``-margin``.

    *b* pairs only the candidate got right, *c* pairs only the control got right, of *n* pairs;
    ``se = sqrt((b + c) - (b - c)² / n) / n``. No pair cannot pass.
    """
    if min(b, c, n) < 0 or b + c > n:
        raise ValueError(f'{b} and {c} discordant pairs cannot come from {n} pairs')
    counts = {'n': n, 'candidate_only': b, 'control_only': c, 'margin': margin}
    if not n:
        return counts | {'difference': None, 'ci': None, 'passes': False}
    difference = (b - c) / n
    half_width = WILSON_Z * math.sqrt(max(0.0, (b + c) - (b - c) ** 2 / n)) / n
    ci = [_shown(difference - half_width), _shown(difference + half_width)]
    return counts | {'difference': _shown(difference), 'ci': ci, 'passes': ci[0] >= -margin}


def _paired(outcomes: Iterable[tuple[bool, bool]], margin: float) -> dict[str, Any]:
    """paired_noninferiority over ``(control right, candidate right)`` per pair."""
    outcomes = list(outcomes)
    return paired_noninferiority(
        sum(1 for control, candidate in outcomes if candidate and not control),
        sum(1 for control, candidate in outcomes if control and not candidate),
        len(outcomes), margin,
    )


# --------------------------------------------------------------------------- #
# Extraction: over two population_weighted blocks
# --------------------------------------------------------------------------- #


def extraction_noninferiority(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """E1 and E2 over two arms' ``population_weighted`` blocks: score_replay's ``decide``.

    E1 compares the reported, already rounded, rate and bound. E2's retention is the
    ratio of the unrounded window holdings, rounded only for display, as lever_decision's is.
    """
    control_point = control['overreach_rate_minted']['rate']
    candidate_upper = (candidate['overreach_rate_minted']['ci'] or [None, None])[1]
    retention = _ratio(candidate['window_counts']['holding'], control['window_counts']['holding'])
    checks = {
        'E1_overreach_upper_within_margin': (
            candidate_upper is not None and control_point is not None
            and candidate_upper <= _shown(control_point + EXTRACTION_OVERREACH_MARGIN)
        ),
        'E2_retains_holdings': retention is not None and retention >= HOLDING_RETENTION_MIN,
    }
    return {
        'statistics': {
            'control_overreach_rate': control_point, 'candidate_overreach_upper': candidate_upper,
            'holding_retention': None if retention is None else _shown(retention),
        },
        'bars': {'overreach_margin': EXTRACTION_OVERREACH_MARGIN, 'holding_retention_min': HOLDING_RETENTION_MIN},
        'checks': checks,
        'passes': all(checks.values()),
    }


# --------------------------------------------------------------------------- #
# One site's runs: the control and one candidate arm over the same fixtures
# --------------------------------------------------------------------------- #


class RunFileError(ValueError):
    """Run records a decision must not be built from."""


class VerdictError(ValueError):
    """Gold, ratings or verdicts a decision must not be built from."""

    def __init__(self, message: str, items: Iterable[str] = ()) -> None:
        super().__init__(message)
        self.items: tuple[str, ...] = tuple(sorted(set(items)))


class MissingVerdicts(VerdictError):
    pass


RunRecord = Mapping[str, Any]
"""A luna_eval_sites run line: replayed, or a failure line."""

Pair = tuple[str, RunRecord, RunRecord]
"""``(fixture_id, control record, candidate record)``."""


@dataclass(frozen=True)
class SiteRuns:
    """One site's run records: each arm's run lines by fixture, and the fixtures each arm failed."""

    candidate: str
    """The one arm other than the control."""
    replayed: Mapping[str, Mapping[str, RunRecord]]
    """arm -> fixture_id -> run line."""
    failed: Mapping[str, frozenset[str]]
    """arm -> the fixtures it holds a failure line for."""
    fixtures: frozenset[str]

    @classmethod
    def from_records(cls, records: Iterable[RunRecord]) -> SiteRuns:
        records = list(records)
        if not records:
            raise RunFileError('a decision needs run records, and none were given')
        if len(sites := {record['site'] for record in records}) != 1:
            raise RunFileError(f'a decision judges one site, not {sorted(sites)}')
        arms = {record['arm'] for record in records}
        if CONTROL not in arms or len(arms) != 2:
            raise RunFileError(f'a site is judged as {CONTROL!r} against one candidate arm, not {sorted(arms)}')
        (candidate,) = arms - {CONTROL}
        lines = Counter((record['arm'], record['fixture_id']) for record in records)
        if repeated := sorted(key for key, count in lines.items() if count > 1):
            raise RunFileError(f'(arm, fixture) recorded more than once: {repeated}')
        replayed: dict[str, dict[str, RunRecord]] = {arm: {} for arm in arms}
        failed: dict[str, set[str]] = {arm: set() for arm in arms}
        for record in records:
            if record.get('kind') == RecordKind.replayed:
                replayed[record['arm']][record['fixture_id']] = record
            else:
                failed[record['arm']].add(record['fixture_id'])
        covered = {arm: replayed[arm].keys() | failed[arm] for arm in arms}
        fixtures = frozenset().union(*covered.values())
        for arm in sorted(arms):
            if missing := sorted(fixtures - covered[arm]):
                raise RunFileError(f'arm {arm} holds no line for fixture(s) {missing}')
        return cls(
            candidate=candidate, replayed=replayed,
            failed={arm: frozenset(ids) for arm, ids in failed.items()}, fixtures=fixtures,
        )

    @property
    def arms(self) -> tuple[str, str]:
        return (CONTROL, self.candidate)

    def pairs(self) -> list[Pair]:
        """Every fixture both arms ran, by fixture id."""
        control, candidate = (self.replayed[arm] for arm in self.arms)
        return [(fid, control[fid], candidate[fid]) for fid in sorted(control.keys() & candidate.keys())]

    def failures(self) -> dict[str, Any]:
        candidate_failed = len(self.failed[self.candidate])
        return {
            'candidate_arm': self.candidate, 'fixtures': len(self.fixtures),
            'candidate_failed': candidate_failed, 'control_failed': len(self.failed[CONTROL]),
            'candidate_failure_share': candidate_failed / len(self.fixtures),
        }


def _decision(
    runs: SiteRuns, statistics: dict[str, Any], bars: dict[str, Any], checks: dict[str, bool],
) -> dict[str, Any]:
    """A site's decision: its own checks, and the candidate failure-share gate every site shares."""
    failures = runs.failures()
    checks = checks | {
        'candidate_failures_within_share': failures['candidate_failure_share'] <= LUNA_FAILURE_SHARE_MAX,
    }
    return {
        'statistics': statistics | {'failures': failures},
        'bars': bars | {'candidate_failure_share_max': LUNA_FAILURE_SHARE_MAX},
        'checks': checks,
        'passes': all(checks.values()),
    }


def _refuse_missing(what: str, items: Sequence[str]) -> None:
    if items:
        raise MissingVerdicts(f'{len(items)} {what}: {sorted(items)}', items)


# --------------------------------------------------------------------------- #
# resolve_edge: each arm's merge decision against gold labelled blind
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ResolveGold:
    """One resolve_edge fixture's gold, labelled with no arm output in view."""

    duplicates: frozenset[str]
    """The related edges the new edge duplicates; empty when it duplicates none."""
    contradicted: frozenset[str] = frozenset()
    """The candidates it contradicts: reported, never gated."""
    judgeable: bool = True
    """False for a fixture that cannot be labelled; it leaves n."""


def _merges_into_a_duplicate(record: RunRecord, gold: ResolveGold) -> bool:
    duplicates = record['duplicate_uuids']
    return not duplicates or duplicates[0] in gold.duplicates


def _merge_correct(record: RunRecord, gold: ResolveGold) -> bool:
    """The arm merges exactly when gold holds a duplicate, and into one of gold's duplicates."""
    return bool(record['duplicate_uuids']) == bool(gold.duplicates) and _merges_into_a_duplicate(record, gold)


def _contradiction_agreement(judged: Sequence[tuple[ResolveGold, RunRecord]]) -> float | None:
    agreeing = sum(1 for gold, record in judged if frozenset(record['invalidated_uuids']) == gold.contradicted)
    return None if not judged else _shown(agreeing / len(judged))


def resolve_edge_decision(records: Iterable[RunRecord], gold: Mapping[str, ResolveGold]) -> dict[str, Any]:
    """R1 and R2 over every judgeable fixture both arms ran."""
    runs = SiteRuns.from_records(records)
    pairs = runs.pairs()
    _refuse_missing('judged fixture(s) have no gold', [fid for fid, _, _ in pairs if fid not in gold])
    judged = [(gold[fid], control, candidate) for fid, control, candidate in pairs if gold[fid].judgeable]
    accuracy = _paired(
        ((_merge_correct(control, g), _merge_correct(candidate, g)) for g, control, candidate in judged),
        RESOLVE_ACCURACY_MARGIN,
    )
    no_false_merge = _paired(
        ((_merges_into_a_duplicate(control, g), _merges_into_a_duplicate(candidate, g))
         for g, control, candidate in judged),
        RESOLVE_FALSE_MERGE_MARGIN,
    )
    contradiction_agreement = {
        CONTROL: _contradiction_agreement([(g, control) for g, control, _ in judged]),
        runs.candidate: _contradiction_agreement([(g, candidate) for g, _, candidate in judged]),
    }
    return _decision(
        runs,
        {
            'merge_accuracy': accuracy, 'no_false_merge': no_false_merge, 'unjudgeable': len(pairs) - len(judged),
            'contradiction_agreement': contradiction_agreement,
        },
        {'merge_accuracy_margin': RESOLVE_ACCURACY_MARGIN, 'no_false_merge_margin': RESOLVE_FALSE_MERGE_MARGIN},
        {
            'R1_merge_accuracy_noninferior': accuracy['passes'],
            'R2_no_false_merge_noninferior': no_false_merge['passes'],
        },
    )


# --------------------------------------------------------------------------- #
# summaries: each arm's summary per node, rated blind
# --------------------------------------------------------------------------- #

FAITHFUL = 'faithful'
SUMMARY_RATINGS: tuple[str, ...] = (FAITHFUL, 'unsupported', 'dropped')
"""faithful; unsupported, asserting or contradicting beyond its inputs; dropped, no summary for the node."""

SummaryItem = tuple[str, str]
"""``(fixture_id, node_uuid)``."""


def summary_items(runs: SiteRuns) -> list[SummaryItem]:
    """Every node either arm summarised, over the fixtures both arms ran."""
    return sorted({
        (fid, node) for fid, control, candidate in runs.pairs()
        for record in (control, candidate) for node in record['summaries']
    })


def summary_decision(
    records: Iterable[RunRecord], ratings: Mapping[SummaryItem, Mapping[str, str]],
) -> dict[str, Any]:
    """S1 over every summary item, each rated under both arms."""
    runs = SiteRuns.from_records(records)
    items = summary_items(runs)
    _refuse_missing(
        'summary item(s) lack a rating under both arms',
        [f'{fid}/{node}' for fid, node in items if not all(arm in ratings.get((fid, node), {}) for arm in runs.arms)],
    )
    fidelity = _paired(
        ((ratings[item][CONTROL] == FAITHFUL, ratings[item][runs.candidate] == FAITHFUL) for item in items),
        SUMMARY_FIDELITY_MARGIN,
    )
    return _decision(
        runs, {'faithful_rate': fidelity}, {'faithful_rate_margin': SUMMARY_FIDELITY_MARGIN},
        {'S1_faithful_rate_noninferior': fidelity['passes']},
    )


# --------------------------------------------------------------------------- #
# classifier and router: labels, adjudicated on discordant items only
# --------------------------------------------------------------------------- #


def routing_label_decision(records: Iterable[RunRecord], gold: Mapping[str, str]) -> dict[str, Any]:
    """C1 or Q1 over every item both arms labelled. An agreement is equally right or wrong, so moves
    nothing; on a discordant item *gold* decides, and a gold label neither arm gave makes both wrong."""
    runs = SiteRuns.from_records(records)
    discordant = [
        (fid, control['label'], candidate['label']) for fid, control, candidate in runs.pairs()
        if control['label'] != candidate['label']
    ]
    _refuse_missing('discordant item(s) have no gold label', [fid for fid, _, _ in discordant if fid not in gold])
    accuracy = paired_noninferiority(
        sum(1 for fid, _, candidate in discordant if gold[fid] == candidate),
        sum(1 for fid, control, _ in discordant if gold[fid] == control),
        len(runs.pairs()), ROUTING_LABEL_MARGIN,
    )
    return _decision(
        runs, {'label_accuracy': accuracy, 'discordant': len(discordant)},
        {'label_accuracy_margin': ROUTING_LABEL_MARGIN}, {'label_accuracy_noninferior': accuracy['passes']},
    )


def routing_layer_decision(classifier: Mapping[str, Any], router: Mapping[str, Any]) -> dict[str, Any]:
    """The classifier and the router share the routing.model leaf, so they flip together or not at all."""
    checks = {'C1_classifier_label_accuracy': classifier['passes'], 'Q1_router_label_accuracy': router['passes']}
    return {'checks': checks, 'passes': all(checks.values())}
