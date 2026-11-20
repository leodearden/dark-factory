#!/usr/bin/env python3
"""Classify the soak's M3 hits and score each M3 lever arm on the reader rubric (task 7519).

Pure: it reads only committed files. These are the soak items, verdicts and
strata; the text-free replay features (``write_triage_m3_replay_cases.py``);
the hand labels of the M3 hits; and the arm rows ``run_write_triage_m3_levers.py``
wrote. Every figure comes from ``score_write_triage_reader.py``, scored on
item rows whose outcomes are the arm's: rater answers are about a (write,
parent) pair and blind to outcome, so a replayed outcome re-scores against
the committed verdicts.

Two parts:

- ``classify``: why each soak M3 hit was hidden. Each hit is split by its
  soak facts, its rater answers, its replay features and its hand labels.
- ``report``: each complete arm's metrics, its paired M3 delta against the
  baseline arm, the outcome transitions and false contests the delta costs,
  and its calls, latency and list-price spend. It also gives the harness
  checks (the ψ reproduction's agreement with the soak) and the arms left
  partial.

A substituted outcome can need a HOME answer the raters tied, which today's
outcome never asked for. The arm is then scored twice, closing each tie
once each way with an in-memory copy of one tied voter's row
(:data:`BOUND_RATERS`). M1, M3, M5 and M2-contested must agree across the
two closures; M4, M2-amended and harm are reported as ranges. No bound row
is ever written.

Usage
-----
Run from ``fused-memory/``::

    uv run python scripts/score_write_triage_m3_levers.py report \\
        --arms f3p,f3p-repeat,b1-sol-medium --repro-arms psi-repro \\
        --prompt-commit <commit>

The report carries no timestamps, so the same inputs give the same files.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from fused_memory.server.write_triage import OUTCOME_CONTESTED, OUTCOME_JUDGE
from fused_memory.utils.script_loader import load_script

_SCRIPTS = Path(__file__).resolve().parent
CALIBRATION = _SCRIPTS.parent / 'calibration'

if TYPE_CHECKING:
    import score_write_triage_reader as _scorer
    import write_triage_m3_replay_cases as _cases
    import write_triage_reader_rubric as _rubric
    import write_triage_stats as _stats
else:
    _scorer = load_script(_SCRIPTS / 'score_write_triage_reader.py', 'score_write_triage_reader')
    _cases = load_script(_SCRIPTS / 'write_triage_m3_replay_cases.py', 'write_triage_m3_replay_cases')
    _rubric = load_script(_SCRIPTS / 'write_triage_reader_rubric.py', 'write_triage_reader_rubric')
    _stats = load_script(_SCRIPTS / 'write_triage_stats.py', 'write_triage_stats')

RUBRIC = 'v2.1'
"""The rater brief the soak verdicts, and so the 0.35 and 0.10 M3 bounds, were rated under."""

BOUND_RATERS = ('replay-bound-home-yes', 'replay-bound-home-no')
"""The rater ids of the two in-memory HOME-tie closures: a copy of the YES voter, then the NO."""
_BOUND_ANSWERS = (_rubric.Resolved.YES, _rubric.Resolved.NO)

DISPUTE_KINDS: Mapping[str, str] = MappingProxyType({
    'outdated_state': (
        'The entry reports a later state of the code, configuration or rule the candidate'
        ' describes, so the candidate is out of date rather than wrong when written.'
    ),
    'status_overtaken': (
        'The entry shows that work the candidate records as pending, open, undecided or not'
        ' yet done has since been done or decided.'
    ),
    'newer_measurement': (
        'The entry reports a later measurement of a figure the candidate also records (a'
        ' duration, count or ratio) and the two values differ.'
    ),
    'contradiction': (
        'The entry and the candidate say incompatible things about the same thing at the'
        ' same time.'
    ),
})
"""The hand-label vocabulary of what an M3 hit's dispute is; each kind is defined in one sentence.

``status_overtaken`` and ``newer_measurement`` are the two kinds of later
state the soak hits show often enough to need their own name; a hit takes one
of them before ``outdated_state``.
"""

LABEL_FLAGS = ('self_declared_correction', 'dispute_in_child_core')
"""Yes/no hand labels: the entry says it corrects or replaces something; the dispute is its main claim."""

_LABEL_KEYS = frozenset({'entry_id', 'target_id', 'dispute_kind', *LABEL_FLAGS, 'evidence'})
_EXACT_UNDER_CLOSURE = ('m1', 'm3', 'm5', 'm2_contested')
_RANGED_UNDER_CLOSURE = ('m4', 'm2_amended')
_COS_EDGES = ((0.60, '<0.60'), (0.70, '0.60-0.70'), (0.80, '0.70-0.80'), (0.90, '0.80-0.90'),
              (math.inf, '>=0.90'))
_PLACES = 4


def arm_items(
    soak_items: Sequence[Mapping[str, Any]],
    features: Sequence[Any],
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """The scorer item rows an arm's outcomes give; refuse an arm missing any case."""
    soak_ids = {item['entry_id'] for item in soak_items}
    case_ids = {feature.entry_id for feature in features}
    if soak_ids != case_ids:
        raise ValueError(
            f'the features cover {sorted(case_ids ^ soak_ids)} differently from the soak items'
        )
    by_entry = {row['entry_id']: row for row in rows}
    missing = sorted(case_ids - set(by_entry))
    if missing:
        raise ValueError(f'the arm rows miss {len(missing)} cases: {missing}')
    return [_cases.item_row(feature, _scored_outcome(by_entry[feature.entry_id]))
            for feature in features]


def _scored_outcome(row: Mapping[str, Any]) -> str:
    """A dispute-marked write reads as contested: the marker is rendered like a contest."""
    return OUTCOME_CONTESTED if row.get('dispute_marked') else row['routed_outcome']


def _closures(
    items: Sequence[Mapping[str, Any]], verdicts: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[str, str]], tuple[tuple[str, list[Mapping[str, Any]]], ...]] | None:
    """None when *items* score as they are; else the tied pairs and each closure's verdicts."""
    try:
        _scorer.item_readings(items, verdicts)
    except _scorer.IncompleteJudgementError as refusal:
        gaps = refusal.gaps
        if any(gap.binary != _rubric.Binary.HOME or gap.kind != 'tied' for gap in gaps):
            raise
    else:
        return None
    pairs = sorted({(gap.entry_id, gap.target_id) for gap in gaps})
    closures = tuple(
        (rater, [*verdicts, *(_bound_row(verdicts, pair, rater, answer) for pair in pairs)])
        for rater, answer in zip(BOUND_RATERS, _BOUND_ANSWERS, strict=True)
    )
    return pairs, closures


def _bound_row(
    verdicts: Sequence[Mapping[str, Any]], pair: tuple[str, str], rater: str, answer: str,
) -> dict[str, Any]:
    for row in verdicts:
        if (
            (row['entry_id'], row['target_id']) == pair and row['basis'] == _rubric.Basis.CURRENT
            and row['role'] in _rubric.RATER_ROLES and row.get('home') == answer
        ):
            return {**row, 'rater': rater, 'role': str(_rubric.Role.THIRD)}
    raise ValueError(f'pair {pair}: no rater answered HOME {answer}, so its tie cannot be closed')


def score_arm(
    items: Sequence[Mapping[str, Any]],
    verdicts: Sequence[Mapping[str, Any]],
    strata: Mapping[str, int],
) -> dict[str, Any]:
    """The reader metrics of one arm's items, closing any HOME tie the outcomes newly need."""
    closed = _closures(items, verdicts)
    if closed is None:
        return {**_blocks(_score(items, verdicts, strata)), 'closure': None}
    pairs, closures = closed
    blocks = {rater: _blocks(_score(items, rows, strata)) for rater, rows in closures}
    first, second = blocks.values()
    for key in _EXACT_UNDER_CLOSURE:
        if first[key] != second[key]:
            raise ValueError(f'{key} differs between the HOME-tie closures of {pairs}')
    ranged: dict[str, Any] = {
        key: {
            'population_rate': _range(block[key]['population_rate'] for block in blocks.values()),
            'by_closure': {rater: block[key] for rater, block in blocks.items()},
        }
        for key in _RANGED_UNDER_CLOSURE
    }
    ranged['harm'] = {
        'population_delta_per_write': _range(
            block['harm']['population_delta_per_write'] for block in blocks.values()
        ),
        'by_closure': {rater: block['harm'] for rater, block in blocks.items()},
    }
    return {
        **{key: first[key] for key in _EXACT_UNDER_CLOSURE},
        **ranged,
        'closure': {'pairs': [list(pair) for pair in pairs], 'raters': list(BOUND_RATERS)},
    }


def _score(
    items: Sequence[Mapping[str, Any]], verdicts: Sequence[Mapping[str, Any]],
    strata: Mapping[str, int],
) -> dict[str, Any]:
    return _scorer.score_reader(items, verdicts, rubric=RUBRIC, strata=strata)


def _blocks(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = report['metrics']
    return {
        'm1': metrics['m1'], 'm3': metrics['m3'], 'm5': metrics['m5'],
        'm2_contested': metrics['m2']['contested'], 'm2_amended': metrics['m2']['amended'],
        'm4': metrics['m4'], 'harm': report['harm'],
    }


def _range(values: Iterable[float | None]) -> list[Any]:
    """[low, high] of the closures' figures, or the figures as given when one is None."""
    listed = list(values)
    present = [value for value in listed if value is not None]
    return sorted(present) if len(present) == len(listed) else listed


def _weights(items: Sequence[Mapping[str, Any]], strata: Mapping[str, int]) -> dict[str, float]:
    sampled = Counter(item['stratum'] for item in items)
    return {stratum: strata[stratum] / count for stratum, count in sampled.items()}


def _readings(
    items: Sequence[Mapping[str, Any]], verdicts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Each item's reading, under the first HOME-tie closure when one is needed.

    M3 never reads HOME on a CONFLICT write, so either closure gives the same endings.
    """
    closed = _closures(items, verdicts)
    rows = verdicts if closed is None else closed[1][0][1]
    return {reading.entry_id: reading for reading in _scorer.item_readings(items, rows)}


def _is_conflict(reading: Any) -> bool:
    return reading.decided is not None and reading.decided.conflict


def _is_hidden(reading: Any) -> bool:
    return reading.m3_ending in _scorer.HIDDEN_ENDINGS


def _false_conflict(reading: Any) -> int:
    return int(reading.decided is not None and _rubric.Cell.FALSE_CONFLICT in reading.decided.triage_cells)


def paired_delta(
    arm: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    verdicts: Sequence[Mapping[str, Any]],
    strata: Mapping[str, int],
) -> dict[str, Any]:
    """The arm's population M3 minus the baseline's, paired on each write, and what it costs."""
    ours, theirs = _readings(arm, verdicts), _readings(baseline, verdicts)
    if set(ours) != set(theirs):
        raise ValueError('a paired delta needs the same writes on both sides')
    weights = _weights(baseline, strata)
    units = [
        _stats.StratumUnit(
            theirs[entry].stratum,
            float(_is_hidden(ours[entry])) - float(_is_hidden(theirs[entry])),
            float(_is_conflict(ours[entry]) or _is_conflict(theirs[entry])),
        )
        for entry in sorted(theirs)
    ]
    changes = [(theirs[entry], _false_conflict(ours[entry]) - _false_conflict(theirs[entry]))
               for entry in sorted(theirs)]
    transitions: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for entry in sorted(theirs):
        if ours[entry].outcome != theirs[entry].outcome:
            transitions[theirs[entry].stratum][f'{theirs[entry].outcome}->{ours[entry].outcome}'] += 1
    return {
        'm3_delta': _stats.stratified_ratio(units, weights),
        'm3_delta_interval95': _stats.stratified_bootstrap95(
            units, weights, resamples=_scorer.BOOTSTRAP_RESAMPLES, seed=_scorer.BOOTSTRAP_SEED,
        ),
        'bootstrap': {'resamples': _scorer.BOOTSTRAP_RESAMPLES, 'seed': _scorer.BOOTSTRAP_SEED},
        'conflict_set_differs': sum(_is_conflict(ours[e]) != _is_conflict(theirs[e]) for e in theirs),
        'false_conflict_change': {
            'sample': sum(change for _, change in changes),
            'population': round(sum(weights[r.stratum] * change for r, change in changes), _PLACES),
        },
        'transitions': {
            stratum: dict(sorted(counts.items())) for stratum, counts in sorted(transitions.items())
        },
    }


def outcome_agreement(
    rows: Iterable[Mapping[str, Any]], baseline_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """How often two arms' judge verdicts agree, over the writes both judged."""
    theirs = {row['entry_id']: row['judge_outcome'] for row in baseline_rows}
    pairs = [(row['judge_outcome'], theirs.get(row['entry_id'])) for row in rows]
    judged = [(ours, base) for ours, base in pairs if ours is not None and base is not None]
    agree = sum(ours == base for ours, base in judged)
    return {'n': len(judged), 'agree': agree, 'rate': _stats.rate(agree, len(judged))}


def repro_agreement(
    rows: Iterable[Mapping[str, Any]], soak_items: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """The replayed verdict against the soak's own outcome, over the soak's judge band."""
    soak = {item['entry_id']: item for item in soak_items}
    judged = [row for row in rows if soak[row['entry_id']]['band'] == OUTCOME_JUDGE]
    confusion: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in judged:
        confusion[soak[row['entry_id']]['outcome']][row['judge_outcome']] += 1
    agree = sum(row['judge_outcome'] == soak[row['entry_id']]['outcome'] for row in judged)
    return {
        'n': len(judged),
        'agree': agree,
        'rate': _stats.rate(agree, len(judged)),
        'wilson95': _stats.wilson95(agree, len(judged)),
        'fail_open': sum(row.get('error_type') is not None for row in judged),
        'confusion': {
            outcome: dict(sorted(counts.items())) for outcome, counts in sorted(confusion.items())
        },
    }


def cost(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Calls, fail-opens, latency and list-price spend of one arm's rows."""
    called = [row for row in rows if row['calls']]
    seconds = sorted(row['seconds'] for row in called if row['seconds'] is not None)
    usd = sum(row['usd'] or 0.0 for row in rows)
    return {
        'writes': len(rows),
        'calls': sum(row['calls'] for row in rows),
        'calls_per_write': _ratio(sum(row['calls'] for row in rows), len(rows)),
        'fail_open': sum(row.get('error_type') is not None for row in rows),
        'p50_seconds': _stats.nearest_rank(seconds, 50) if seconds else None,
        'p95_seconds': _stats.nearest_rank(seconds, 95) if seconds else None,
        'usd': round(usd, 6),
        'usd_per_1000_writes': _ratio(usd * 1000, len(rows)),
        'usd_per_1000_judged_writes': _ratio(usd * 1000, len(called)),
    }


def _ratio(numerator: float, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _validated_labels(
    labels: Iterable[Mapping[str, Any]], verdicts: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    quotes: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for row in verdicts:
        quote = row.get('conflict_quote')
        if isinstance(quote, Mapping):
            quotes[(row['entry_id'], row['target_id'])].extend(str(text) for text in quote.values())
    validated: dict[str, Mapping[str, Any]] = {}
    for index, label in enumerate(labels):
        problem = _label_problem(label, quotes)
        if problem:
            raise ValueError(f'label row {index} (entry_id {label.get("entry_id")!r}): {problem}')
        validated[label['entry_id']] = label
    return validated


def _label_problem(label: Mapping[str, Any], quotes: Mapping[tuple[str, str], list[str]]) -> str | None:
    if set(label) != _LABEL_KEYS:
        return f'keys {sorted(label)} are not {sorted(_LABEL_KEYS)}'
    if label['dispute_kind'] not in DISPUTE_KINDS:
        return f'dispute_kind {label["dispute_kind"]!r} is not one of {sorted(DISPUTE_KINDS)}'
    if not all(isinstance(label[flag], bool) for flag in LABEL_FLAGS):
        return f'{", ".join(LABEL_FLAGS)} must be booleans'
    evidence = label['evidence']
    pair = (label['entry_id'], label['target_id'])
    if not isinstance(evidence, str) or not any(evidence and evidence in q for q in quotes[pair]):
        return 'evidence must quote a committed conflict_quote of its pair'
    return None


def classify(
    items: Sequence[Mapping[str, Any]],
    verdicts: Sequence[Mapping[str, Any]],
    strata: Mapping[str, int],
    features: Sequence[Any],
    labels: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Split the soak's M3 hits by every dimension, in sample and population counts."""
    readings = _scorer.item_readings(items, verdicts)
    hits = [reading for reading in readings if _is_hidden(reading)]
    by_label = _validated_labels(labels, verdicts)
    hit_ids = {reading.entry_id for reading in hits}
    if hit_ids != set(by_label):
        raise ValueError(
            f'the labels miss M3 hits {sorted(hit_ids - set(by_label))}'
            f' and label non-hits {sorted(set(by_label) - hit_ids)}'
        )
    soak = {item['entry_id']: item for item in items}
    by_feature = {feature.entry_id: feature for feature in features}
    judgements = _rubric.resolve_judgements(verdicts)
    weights = _weights(items, strata)
    facts = [
        _hit_facts(soak[r.entry_id], by_feature[r.entry_id], by_label[r.entry_id], judgements)
        for r in hits
    ]
    dimensions: dict[str, dict[str, dict[str, Any]]] = {}
    for key in facts[0] if facts else ():
        cells: defaultdict[str, list[float]] = defaultdict(list)
        for reading, fact in zip(hits, facts, strict=True):
            cells[fact[key]].append(weights[reading.stratum])
        dimensions[key] = {
            value: {'n': len(found), 'population': round(sum(found), _PLACES)}
            for value, found in sorted(cells.items())
        }
    return {
        'hits': len(hits),
        'population_hits': round(sum(weights[r.stratum] for r in hits), _PLACES),
        'dimensions': dimensions,
        'by_hit': [
            {'entry_id': r.entry_id, 'ending': r.m3_ending, **fact}
            for r, fact in zip(hits, facts, strict=True)
        ],
    }


def _hit_facts(
    item: Mapping[str, Any], feature: Any, label: Mapping[str, Any],
    judgements: Mapping[Any, Any],
) -> dict[str, str]:
    judgement = judgements.get(
        _rubric.PairKey(item['entry_id'], item['target_id'], _rubric.Basis.CURRENT),
    )

    def answer(binary: Any) -> str:
        resolved = None if judgement is None else judgement.answer(binary)
        return 'unknown' if resolved is None else str(resolved).lower()

    return {
        'soak_band': str(item['band']).lower(),
        'stratum': item['stratum'],
        'scope': answer(_rubric.Binary.SCOPE),
        'action_changing': answer(_rubric.Binary.ACTION_CHANGING),
        'home': answer(_rubric.Binary.HOME),
        'via_sibling': str(item['via_sibling']).lower(),
        'cross_category': str(item['cross_category']).lower(),
        'project': item['project'],
        'route': str(feature.route),
        'cos_parent_bucket': _cos_bucket(feature.cos_parent),
        'dispute_kind': label['dispute_kind'],
        **{flag: str(label[flag]).lower() for flag in LABEL_FLAGS},
    }


def _cos_bucket(cosine: float | None) -> str:
    if cosine is None:
        return 'unknown'
    return next(label for edge, label in _COS_EDGES if cosine < edge)


def _arm_provenance(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    provenance: dict[str, Any] = {}
    for key in ('prompt_sha256', 'model', 'effort', 'cases_sha256'):
        values = {row.get(key) for row in rows}
        if len(values) != 1:
            raise ValueError(f'an arm\'s rows carry {len(values)} values of {key}: {sorted(map(str, values))}')
        provenance[key] = values.pop()
    return {**provenance, 'rows': len(rows)}


def arm_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_rows: Sequence[Mapping[str, Any]] | None,
    soak_items: Sequence[Mapping[str, Any]],
    verdicts: Sequence[Mapping[str, Any]],
    strata: Mapping[str, int],
    features: Sequence[Any],
) -> dict[str, Any]:
    """One arm's report entry; *baseline_rows* is None for the baseline arm itself."""
    items = arm_items(soak_items, features, rows)
    paired = agreement = None
    if baseline_rows is not None:
        paired = paired_delta(items, arm_items(soak_items, features, baseline_rows), verdicts, strata)
        agreement = outcome_agreement(rows, baseline_rows)
    return {
        'provenance': _arm_provenance(rows),
        'score': score_arm(items, verdicts, strata),
        'paired_vs_baseline': paired,
        'outcome_agreement_vs_baseline': agreement,
        'cost': cost(rows),
    }


def build_report(
    *,
    soak_items: Sequence[Mapping[str, Any]],
    verdicts: Sequence[Mapping[str, Any]],
    strata: Mapping[str, int],
    features: Sequence[Any],
    labels: Iterable[Mapping[str, Any]],
    arms: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline: str,
    partial: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
    repro: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """The whole lever report: classification, every complete arm against the baseline, harness checks.

    The provenance gains the prompt sha256 the baseline arm ran under, and
    ``arm_order`` keeps *arms*' order, so the markdown re-renders from the JSON.
    """
    if baseline not in arms:
        raise ValueError(f'baseline arm {baseline!r} is not among the complete arms {list(arms)}')
    baseline_prompt = _arm_provenance(arms[baseline])['prompt_sha256']
    scored = {
        name: arm_report(
            rows, baseline_rows=None if name == baseline else arms[baseline],
            soak_items=soak_items, verdicts=verdicts, strata=strata, features=features,
        )
        for name, rows in arms.items()
    }
    return {
        'rubric': RUBRIC,
        'baseline': baseline,
        'arm_order': list(arms),
        'provenance': {**provenance, 'shipped_prompt_sha256': baseline_prompt},
        'strata': dict(sorted(strata.items())),
        'soak': score_arm(soak_items, verdicts, strata),
        'classification': classify(soak_items, verdicts, strata, features, labels),
        'arms': scored,
        'repro': {
            name: {'provenance': _arm_provenance(rows), 'agreement': repro_agreement(rows, soak_items),
                   'cost': cost(rows)}
            for name, rows in (repro or {}).items()
        },
        'partial': {name: dict(figures) for name, figures in partial.items()},
    }


def _shown(value: Any) -> str:
    if value is None:
        return '—'
    if isinstance(value, list):
        return '–'.join(_shown(part) for part in value)
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value)


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines.extend('| ' + ' | '.join(_shown(cell) for cell in row) + ' |' for row in rows)
    return lines


def _rate_of(block: Mapping[str, Any]) -> Any:
    return block['population_rate']


def _arm_line(name: str, arm: Mapping[str, Any]) -> list[Any]:
    score, paired, spend = arm['score'], arm['paired_vs_baseline'] or {}, arm['cost']
    harm = score['harm']
    return [
        name, _rate_of(score['m3']), score['m3']['population_interval95'],
        paired.get('m3_delta'), paired.get('m3_delta_interval95'),
        _rate_of(score['m1']), _rate_of(score['m5']), _rate_of(score['m2_contested']),
        _rate_of(score['m2_amended']),
        (paired.get('false_conflict_change') or {}).get('population'),
        _rate_of(score['m4']), harm['population_delta_per_write'],
        spend['calls_per_write'], spend['p50_seconds'], spend['p95_seconds'], spend['fail_open'],
        spend['usd_per_1000_judged_writes'],
    ]


def render_markdown(report: Mapping[str, Any]) -> str:
    """The report as markdown; deterministic, so a re-render of the same report is identical."""
    soak_m3 = report['soak']['m3']
    lines = [
        '# Write-triage M3 lever replay',
        '',
        f'Rubric {report["rubric"]}; baseline arm `{report["baseline"]}`; provenance '
        + ', '.join(f'{key} `{value}`' for key, value in sorted(report['provenance'].items())) + '.',
        f'Soak M3 (population): {_shown(soak_m3["population_rate"])}'
        f' ({_shown(soak_m3["population_interval95"])}), {soak_m3["hits"]} of {soak_m3["n"]}'
        ' sampled CONFLICT writes hidden.',
        '',
        '## Arms',
        '',
        *_table(
            ['arm', 'M3', 'M3 95%', 'ΔM3 vs base', 'Δ 95%', 'M1', 'M5', 'M2 contested',
             'M2 amended', 'false contests added', 'M4', 'harm Δ', 'calls/write', 'p50 s',
             'p95 s', 'fail-open', 'USD/1k judged'],
            (_arm_line(name, report['arms'][name]) for name in report['arm_order']),
        ),
        '',
        '## Harness checks',
        '',
        *_table(
            ['arm', 'agreement with the soak', '95%', 'n', 'fail-open'],
            ([name, check['agreement']['rate'], check['agreement']['wilson95'],
              check['agreement']['n'], check['agreement']['fail_open']]
             for name, check in report['repro'].items()),
        ),
        '',
        *_table(
            ['arm', 'verdict agreement with the baseline', 'n'],
            ([name, report['arms'][name]['outcome_agreement_vs_baseline']['rate'],
              report['arms'][name]['outcome_agreement_vs_baseline']['n']]
             for name in report['arm_order']
             if report['arms'][name]['outcome_agreement_vs_baseline']),
        ),
        '',
        '## Arms left partial',
        '',
        *_table(
            ['arm', 'rows', 'cases', 'USD so far', 'projected USD'],
            ([name, figures['rows'], figures['cases'], figures['usd'], figures['projected_usd']]
             for name, figures in report['partial'].items()),
        ),
        '',
        '## Classification of the soak M3 hits',
        '',
        f'{report["classification"]["hits"]} hits, population weight'
        f' {_shown(report["classification"]["population_hits"])}.',
    ]
    for dimension, cells in report['classification']['dimensions'].items():
        lines.extend(['', f'### {dimension}', ''])
        lines.extend(_table([dimension, 'n', 'population'],
                            ([value, cell['n'], cell['population']] for value, cell in cells.items())))
    return '\n'.join(lines) + '\n'


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _partial_figures(rows: Sequence[Mapping[str, Any]], cases: int) -> dict[str, Any]:
    usd = round(sum(row['usd'] or 0.0 for row in rows), 6)
    return {'rows': len(rows), 'cases': cases, 'usd': usd,
            'projected_usd': round(usd / len(rows) * cases, 2) if rows else None}


def _inputs(args: argparse.Namespace) -> dict[str, Any]:
    return committed_inputs(
        items=args.items, verdicts=args.verdicts, strata=args.strata, features=args.features,
        labels=args.labels,
    )


def committed_inputs(
    *,
    items: Path = CALIBRATION / 'write_triage_v2_soak_items.jsonl',
    verdicts: Path = CALIBRATION / 'write_triage_v2_soak_verdicts.jsonl',
    strata: Path = CALIBRATION / 'write_triage_v2_soak_strata.json',
    features: Path = CALIBRATION / 'write_triage_m3_replay_cases.jsonl',
    labels: Path = CALIBRATION / 'write_triage_m3_hidden_conflicts.jsonl',
) -> dict[str, Any]:
    """The report's committed inputs, keyed as :func:`build_report` takes them."""
    return {
        'soak_items': _read_jsonl(items),
        'verdicts': _read_jsonl(verdicts),
        'strata': json.loads(strata.read_text()),
        'features': _cases.read_features(features),
        'labels': _read_jsonl(labels),
    }


def assemble_report(
    *,
    arms_dir: Path,
    arms: Sequence[str],
    repro_arms: Sequence[str],
    baseline: str,
    prompt_commit: str,
    items: Path = CALIBRATION / 'write_triage_v2_soak_items.jsonl',
    verdicts: Path = CALIBRATION / 'write_triage_v2_soak_verdicts.jsonl',
    strata: Path = CALIBRATION / 'write_triage_v2_soak_strata.json',
    features: Path = CALIBRATION / 'write_triage_m3_replay_cases.jsonl',
    labels: Path = CALIBRATION / 'write_triage_m3_hidden_conflicts.jsonl',
) -> dict[str, Any]:
    """The lever report from files: an arm missing cases is reported as partial, not scored."""
    inputs = committed_inputs(
        items=items, verdicts=verdicts, strata=strata, features=features, labels=labels,
    )
    case_count = len(inputs['features'])
    complete: dict[str, list[dict[str, Any]]] = {}
    partial: dict[str, dict[str, Any]] = {}
    for name in arms:
        path = arms_dir / f'{name}.jsonl'
        rows = _read_jsonl(path) if path.exists() else []
        if len(rows) == case_count:
            complete[name] = rows
        else:
            partial[name] = _partial_figures(rows, case_count)
    repro = {name: _read_jsonl(arms_dir / f'{name}.jsonl') for name in repro_arms}
    return build_report(
        **inputs, arms=complete, baseline=baseline, partial=partial,
        provenance={'shipped_prompt_commit': prompt_commit}, repro=repro,
    )


def _names(listed: str) -> list[str]:
    return [name.strip() for name in listed.split(',') if name.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('classify', 'report'):
        command = commands.add_parser(name)
        command.add_argument('--items', type=Path, default=CALIBRATION / 'write_triage_v2_soak_items.jsonl')
        command.add_argument('--verdicts', type=Path,
                             default=CALIBRATION / 'write_triage_v2_soak_verdicts.jsonl')
        command.add_argument('--strata', type=Path, default=CALIBRATION / 'write_triage_v2_soak_strata.json')
        command.add_argument('--features', type=Path,
                             default=CALIBRATION / 'write_triage_m3_replay_cases.jsonl')
        command.add_argument('--labels', type=Path,
                             default=CALIBRATION / 'write_triage_m3_hidden_conflicts.jsonl')
    report = commands.choices['report']
    report.add_argument('--arms-dir', type=Path, default=CALIBRATION / 'write_triage_m3_lever_arms')
    report.add_argument('--arms', required=True, help='comma-separated arms to score, in table order')
    report.add_argument('--repro-arms', default='', help='comma-separated soak-reproduction arms')
    report.add_argument('--baseline', default='f3p')
    report.add_argument('--prompt-commit', required=True,
                        help='the commit whose shipped judge prompt the baseline arms ran under')
    report.add_argument('--out-json', type=Path, default=CALIBRATION / 'write_triage_m3_levers_report.json')
    report.add_argument('--out-md', type=Path, default=CALIBRATION / 'write_triage_m3_levers_report.md')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == 'classify':
        inputs = _inputs(args)
        print(json.dumps(classify(inputs['soak_items'], inputs['verdicts'], inputs['strata'],
                                  inputs['features'], inputs['labels']), indent=2, sort_keys=True))
        return 0
    report = assemble_report(
        arms_dir=args.arms_dir, arms=_names(args.arms), repro_arms=_names(args.repro_arms),
        baseline=args.baseline, prompt_commit=args.prompt_commit, items=args.items,
        verdicts=args.verdicts, strata=args.strata, features=args.features, labels=args.labels,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + '\n'
    args.out_json.write_text(text)
    args.out_md.write_text(render_markdown(json.loads(text)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
