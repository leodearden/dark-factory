#!/usr/bin/env python3
"""Score task 9945's luna migration eval: blind worksheets, fail-closed verdict files, each call
site's decision against its pre-registered bar, cost, and report.json.

This module is the single code home of those bars: design.md §2 cites the
constants below by name. Every site judges one candidate arm against the
control, on the fixtures both arms ran, by a paired non-inferiority interval.
Only discordant pairs move the difference: b pairs only the candidate got
right, c pairs only the control got right. The extraction site is judged
instead over two population_weighted blocks, as
replay_overreach_extraction.score_replay hands them to its ``decide``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_ruling_overreach import WILSON_Z
from replay_overreach_extraction import (
    CONTROL,
    HOLDING_RETENTION_MIN,
    RecordKind,
    score_replay,
    token_totals,
)
from score_write_triage_pairs import LIST_PRICES, LIST_PRICES_AS_OF, LIST_PRICES_SOURCE

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


class MalformedVerdicts(VerdictError):
    pass


class UnknownLabel(VerdictError):
    pass


class MissingRationale(VerdictError):
    pass


class DuplicateVerdict(VerdictError):
    pass


class OutOfSampleVerdict(VerdictError):
    pass


class MissingVerdicts(VerdictError):
    pass


RunRecord = Mapping[str, Any]
"""A luna_eval_sites run line: replayed, or a failure line."""

Pair = tuple[str, RunRecord, RunRecord]
"""``(fixture_id, control record, candidate record)``."""


@dataclass(frozen=True)
class SiteRuns:
    """One site's run records: each arm's run lines by fixture, and the fixtures each arm failed."""

    site: str
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
            site=sites.pop(), candidate=candidate, replayed=replayed,
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


def _reject(error: type[VerdictError], items: Sequence[str], problem: str) -> None:
    if items:
        raise error(f'{len(set(items))} {problem}: {sorted(set(items))}', items)


def _refuse_missing(what: str, items: Sequence[str]) -> None:
    _reject(MissingVerdicts, items, what)


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


# --------------------------------------------------------------------------- #
# Blind worksheets: no arm, model or endpoint is ever named
# --------------------------------------------------------------------------- #


def _sha256_order(fixture_id: str, values: Iterable[str]) -> list[str]:
    return sorted(values, key=lambda value: hashlib.sha256(f'{fixture_id}:{value}'.encode()).hexdigest())


def _fact(edge: Mapping[str, Any]) -> dict[str, str]:
    return {'uuid': edge['uuid'], 'fact': edge['fact']}


def resolve_gold_worksheet(fixtures: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per resolve_edge fixture, by fixture id: the new fact and its candidates, read from the
    fixture alone, so gold is labelled before any arm's output exists in view."""
    return [
        {
            'fixture_id': fid, 'fact': fixture['fact'],
            'related': [_fact(edge) for edge in fixture['related']],
            'invalidation_candidates': [_fact(edge) for edge in fixture['invalidation_candidates']],
        }
        for fid, fixture in fixtures.items()
    ]


SUMMARY_LETTERS = ('A', 'B')


def summary_key(runs: SiteRuns) -> list[dict[str, str]]:
    """Each summary item's letters to arms: the arm whose sha256(fixture_id:arm) sorts first is A.

    The rule reads only the fixture id and the arm names, so anyone holding the runs can
    recompute it; blindness rests on the rater not doing so.
    """
    return [
        {'fixture_id': fid, 'node_uuid': node_uuid,
         **dict(zip(SUMMARY_LETTERS, _sha256_order(fid, runs.arms), strict=True))}
        for fid, node_uuid in summary_items(runs)
    ]


def summary_rating_worksheet(
    fixtures: Mapping[str, Mapping[str, Any]], runs: Iterable[RunRecord],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """The rating rows and, separately, their summary_key.

    Each row is one summary item with its inputs and both arms' summaries under the
    key's letters. *fixtures* are run_summaries' restored fixtures by fixture id, each
    with its ``previous`` episode contents. An empty summary shows as None, a node the
    arm dropped. The key is written apart from the rows.
    """
    site_runs = SiteRuns.from_records(runs)
    by_fixture = {fid: dict(zip(site_runs.arms, (control, candidate), strict=True))
                  for fid, control, candidate in site_runs.pairs()}
    key = summary_key(site_runs)
    rows: list[dict[str, Any]] = []
    for entry in key:
        fid, node_uuid, fixture = entry['fixture_id'], entry['node_uuid'], fixtures[entry['fixture_id']]
        node = {node['node_uuid']: node for node in fixture['nodes']}[node_uuid]
        rows.append({
            'fixture_id': fid, 'node_uuid': node_uuid, 'node_name': node['node_name'],
            'previous_summary': node['summary'], 'episode_content': fixture['episode_content'],
            'previous_episodes': list(fixture['previous']),
            'facts': [edge['fact'] for edge in fixture['edges']
                      if node_uuid in (edge['source_node_uuid'], edge['target_node_uuid'])],
            'summaries': {
                letter: by_fixture[fid][entry[letter]]['summaries'].get(node_uuid) or None
                for letter in SUMMARY_LETTERS
            },
        })
    return rows, key


def routing_discordant_worksheet(
    fixtures: Mapping[str, str], runs: Iterable[RunRecord], vocabulary: Sequence[str],
) -> list[dict[str, Any]]:
    """One row per item the two arms labelled differently: its text, from *fixtures* by fixture id,
    both labels unattributed in sha256(fixture_id:label) order, and the site's whole *vocabulary*."""
    site_runs = SiteRuns.from_records(runs)
    return [
        {
            'site': site_runs.site, 'fixture_id': fid, 'text': fixtures[fid],
            'labels': _sha256_order(fid, {control['label'], candidate['label']}), 'vocabulary': list(vocabulary),
        }
        for fid, control, candidate in site_runs.pairs() if control['label'] != candidate['label']
    ]


# --------------------------------------------------------------------------- #
# Verdict files: refused whole, naming each offender
# --------------------------------------------------------------------------- #

FieldCheck = Callable[[Any], bool]


def _is_text(value: Any) -> bool:
    return isinstance(value, str)


def _is_texts(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_flag(value: Any) -> bool:
    return isinstance(value, bool)


def _shaped_entries(obj: Any, field: str, shape: Mapping[str, FieldCheck]) -> list[Mapping[str, Any]]:
    if not isinstance(obj, Mapping) or not isinstance(obj.get(field), list):
        raise MalformedVerdicts(f"this verdict file is {{'{field}': [...]}}")
    entries = obj[field]
    bad = [entry for entry in entries
           if not isinstance(entry, Mapping) or not all(check(entry.get(name)) for name, check in shape.items())]
    if bad:
        raise MalformedVerdicts(f'{len(bad)} entr(ies) lack the fields {sorted(shape)}: {bad!r}')
    return entries


def _checked(
    entries: Sequence[Mapping[str, Any]], identity: Callable[[Mapping[str, Any]], str],
    offered: Mapping[str, Any], known: Callable[[Mapping[str, Any], Any], bool],
) -> list[Mapping[str, Any]]:
    """*entries*, once each is offered, known against its worksheet row, reasoned and unique, and
    every offered item is judged."""
    ids = [identity(entry) for entry in entries]
    _reject(OutOfSampleVerdict, [i for i in ids if i not in offered], "verdict(s) judge an item the worksheet did not offer")
    _reject(UnknownLabel, [i for i, entry in zip(ids, entries, strict=True) if not known(entry, offered[i])],
            'verdict(s) carry a label outside the worksheet vocabulary')
    _reject(MissingRationale, [i for i, entry in zip(ids, entries, strict=True) if not entry['rationale'].strip()],
            'verdict(s) lack a rationale')
    counts = Counter(ids)
    _reject(DuplicateVerdict, [i for i in ids if counts[i] > 1], 'item(s) are judged twice')
    _reject(MissingVerdicts, [i for i in offered if i not in counts], 'offered item(s) have no verdict')
    return list(entries)


GOLD_SHAPE: Mapping[str, FieldCheck] = {
    'fixture_id': _is_text, 'duplicates': _is_texts, 'contradicted': _is_texts, 'judgeable': _is_flag,
    'rationale': _is_text,
}


def _gold_known(entry: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    """Duplicates are among the related edges, contradictions among every candidate, and an
    unjudgeable fixture names neither."""
    related = {edge['uuid'] for edge in row['related']}
    candidates = related | {edge['uuid'] for edge in row['invalidation_candidates']}
    if not entry['judgeable']:
        return not entry['duplicates'] and not entry['contradicted']
    return set(entry['duplicates']) <= related and set(entry['contradicted']) <= candidates


def load_resolve_gold(obj: Any, worksheet: Sequence[Mapping[str, Any]]) -> dict[str, ResolveGold]:
    """``{'gold': [{fixture_id, duplicates, contradicted, judgeable, rationale}]}`` over
    resolve_gold_worksheet's rows, by fixture id."""
    entries = _checked(
        _shaped_entries(obj, 'gold', GOLD_SHAPE), lambda entry: entry['fixture_id'],
        {row['fixture_id']: row for row in worksheet}, _gold_known,
    )
    return {
        entry['fixture_id']: ResolveGold(
            duplicates=frozenset(entry['duplicates']), contradicted=frozenset(entry['contradicted']),
            judgeable=entry['judgeable'],
        )
        for entry in entries
    }


RATING_SHAPE: Mapping[str, FieldCheck] = {
    'fixture_id': _is_text, 'node_uuid': _is_text, 'summary': _is_text, 'rating': _is_text, 'rationale': _is_text,
}


def _item_name(fixture_id: str, node_uuid: str) -> str:
    return f'{fixture_id}/{node_uuid}'


def _rated_name(entry: Mapping[str, Any]) -> str:
    return f"{_item_name(entry['fixture_id'], entry['node_uuid'])}:{entry['summary']}"


def _checked_summary_key(key: Any, runs: SiteRuns) -> dict[SummaryItem, dict[str, str]]:
    """item -> its key entry, once *key* is exactly summary_key over *runs*: a stale or edited key
    would swap the arms when unblinding."""
    if not isinstance(key, list) or not all(isinstance(entry, Mapping) for entry in key):
        raise MalformedVerdicts('the summary key is a list of {fixture_id, node_uuid, A, B}')
    derived = {(entry['fixture_id'], entry['node_uuid']): entry for entry in summary_key(runs)}
    given = Counter((entry.get('fixture_id'), entry.get('node_uuid')) for entry in key)
    by_item = {(entry.get('fixture_id'), entry.get('node_uuid')): dict(entry) for entry in key}
    differing = [
        f'{fid}/{node}' for fid, node in derived.keys() | given.keys()
        if given[(fid, node)] != 1 or by_item.get((fid, node)) != derived.get((fid, node))
    ]
    _reject(MalformedVerdicts, differing, 'summary key entr(ies) differ from the sha256(fixture_id:arm) rule')
    return derived


def load_summary_ratings(obj: Any, records: Iterable[RunRecord], key: Any) -> dict[SummaryItem, dict[str, str]]:
    """``{'ratings': [{fixture_id, node_uuid, summary, rating, rationale}]}``, one per letter of every
    summary item the run *records* offer, unblinded through *key*: item -> arm -> rating.

    *key* is the key file summary_rating_worksheet wrote; it is refused unless it is exactly the
    summary_key the records derive.
    """
    letters = _checked_summary_key(key, SiteRuns.from_records(records))
    offered = {f'{_item_name(*item)}:{letter}': item for item in letters for letter in SUMMARY_LETTERS}
    entries = _checked(
        _shaped_entries(obj, 'ratings', RATING_SHAPE), _rated_name, offered,
        lambda entry, _: entry['rating'] in SUMMARY_RATINGS,
    )
    ratings: dict[SummaryItem, dict[str, str]] = {}
    for entry in entries:
        item = (entry['fixture_id'], entry['node_uuid'])
        ratings.setdefault(item, {})[letters[item][entry['summary']]] = entry['rating']
    return ratings


ROUTING_SHAPE: Mapping[str, FieldCheck] = {
    'site': _is_text, 'fixture_id': _is_text, 'label': _is_text, 'rationale': _is_text,
}


def _routing_name(entry: Mapping[str, Any]) -> str:
    return f"{entry['site']}/{entry['fixture_id']}"


def load_routing_verdicts(obj: Any, worksheet: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    """``{'verdicts': [{site, fixture_id, label, rationale}]}`` over routing_discordant_worksheet's rows,
    each label from its row's vocabulary, which may be neither arm's: site -> fixture id -> label."""
    entries = _checked(
        _shaped_entries(obj, 'verdicts', ROUTING_SHAPE), _routing_name,
        {_routing_name(row): row for row in worksheet}, lambda entry, row: entry['label'] in row['vocabulary'],
    )
    gold: dict[str, dict[str, str]] = {}
    for entry in entries:
        gold.setdefault(entry['site'], {})[entry['fixture_id']] = entry['label']
    return gold


# --------------------------------------------------------------------------- #
# Cost: tokens per call, priced at list for comparison, and the plan-window share
# --------------------------------------------------------------------------- #

INFERRED = 'inferred'
DAYS_PER_WEEK = 7


def _arm_cost(records: Sequence[RunRecord], calls_per_day: float | None) -> dict[str, Any]:
    model = _only_value(records, 'model')
    totals = Counter[str]()
    for record in records:
        for usage in record['tokens'].values():
            totals.update(usage)
    calls = totals['llm_calls']
    price = LIST_PRICES.get(model)
    usd_per_call = None if price is None or not calls else price.usd(
        totals['input_tokens'], totals['output_tokens']) / calls
    return {
        'model': model, 'calls': calls,
        'input_tokens_per_call': round(totals['input_tokens'] / calls, 1) if calls else None,
        'output_tokens_per_call': round(totals['output_tokens'] / calls, 1) if calls else None,
        'usd_per_call': usd_per_call, 'calls_per_day': calls_per_day,
        'daily_usd': None if usd_per_call is None or calls_per_day is None else usd_per_call * calls_per_day,
    }


def _only_value(records: Sequence[RunRecord], field: str) -> Any:
    if len(values := {record[field] for record in records}) != 1:
        raise RunFileError(f'one arm of one site holds records with {field} {sorted(map(str, values))}')
    return values.pop()


def cost_block(
    runs: Iterable[RunRecord], calls_per_day: Mapping[str, float], window_usd_per_week: tuple[float, float],
) -> dict[str, Any]:
    """Each site's arms priced per call at LIST_PRICES, per day at the supplied *calls_per_day* per site,
    and the candidate arms' weekly API-equivalent USD as a share of each end of *window_usd_per_week*.

    The window is an inference, not a measurement, so its share is tagged so.
    """
    by_site: dict[str, dict[str, list[RunRecord]]] = {}
    for record in runs:
        if record.get('kind') == RecordKind.replayed:
            by_site.setdefault(record['site'], {}).setdefault(record['arm'], []).append(record)
    sites = {
        site: {arm: _arm_cost(records, calls_per_day.get(site)) for arm, records in sorted(arms.items())}
        for site, arms in sorted(by_site.items())
    }
    candidate_daily = [block['daily_usd'] for arms in sites.values() for arm, block in arms.items()
                       if arm != CONTROL and block['daily_usd'] is not None]
    weekly = DAYS_PER_WEEK * sum(candidate_daily)
    models = sorted({block['model'] for arms in sites.values() for block in arms.values()})
    return {
        'prices': {
            'as_of': LIST_PRICES_AS_OF, 'source': LIST_PRICES_SOURCE,
            'usd_per_million_tokens': {
                model: None if (price := LIST_PRICES.get(model)) is None
                else {'input': price.usd_per_mtok_input, 'output': price.usd_per_mtok_output}
                for model in models
            },
        },
        'sites': sites,
        'plan_window': {
            'weekly_candidate_usd': weekly, 'window_usd_per_week': list(window_usd_per_week),
            'share': [weekly / end for end in window_usd_per_week], 'tag': INFERRED,
        },
    }


# --------------------------------------------------------------------------- #
# report.json: every number computed here
# --------------------------------------------------------------------------- #


def _arm_statistics(site_runs: SiteRuns, arm: str) -> dict[str, Any]:
    replayed = list(site_runs.replayed[arm].values())
    client = {field: _only_value(replayed, field) if replayed else None
              for field in ('model', 'reasoning_effort', 'endpoint')}
    return client | {
        'replayed': len(replayed), 'failed': len(site_runs.failed[arm]), 'tokens': token_totals(replayed),
    }


def site_block(records: Iterable[RunRecord], decision: Mapping[str, Any]) -> dict[str, Any]:
    """One site's report entry: each arm's client and counts, the paired n, the failures and *decision*."""
    site_runs = SiteRuns.from_records(records)
    return {
        'arms': {arm: _arm_statistics(site_runs, arm) for arm in site_runs.arms},
        'n': len(site_runs.pairs()),
        'failures': site_runs.failures(),
        'decision': dict(decision),
    }


def score_extraction(
    replay_records: Sequence[Mapping[str, Any]], verdict_obj: Any, frame: Mapping[str, Any],
) -> dict[str, Any]:
    """The extraction leg: replay_overreach_extraction's own score over *frame*, each arm judged by E1 and E2."""
    return score_replay(replay_records, verdict_obj, frame, decide=extraction_noninferiority)


def build_report(
    *, frame_text: str, extraction_replay: Sequence[Mapping[str, Any]], extraction_verdicts: Any,
    sites: Mapping[str, Mapping[str, Any]], routing_layer: Mapping[str, Any], cost: Mapping[str, Any],
) -> dict[str, Any]:
    """report.json: the frame.json *frame_text* echoed by its sha256, the extraction leg scored over it,
    each site's site_block, the joint routing verdict, and cost."""
    return {
        'sample': {'frame_sha256': hashlib.sha256(frame_text.encode()).hexdigest()},
        'extraction': score_extraction(extraction_replay, extraction_verdicts, json.loads(frame_text)),
        'sites': dict(sites),
        'routing_layer': dict(routing_layer),
        'cost': dict(cost),
    }
