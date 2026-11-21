#!/usr/bin/env python3
"""Score recon's repairs of write-triage attaches against rubric v2.2's rated soak corpus.

Each replayed child carries the outcome triage filed it under and one repair
recon chose from a fixed vocabulary (:class:`Decision`). The correct repair is
read off the rubric's own cells by :func:`expected_action`, and every repair
is priced by deriving the repaired item row through the same rubric call that
priced the judge, so recon and the judge are compared on the same rows and
weights. Rationale and findings: ``fused-memory/calibration/recon_triage_replay_report.md``.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
)
from fused_memory.utils.script_loader import load_script

_SCRIPTS = Path(__file__).resolve().parent


if TYPE_CHECKING:
    import score_write_triage_reader as _reader
    import write_triage_reader_rubric as _rubric
    import write_triage_stats as _stats
else:
    _rubric = load_script(_SCRIPTS / 'write_triage_reader_rubric.py', 'write_triage_reader_rubric')
    _reader = load_script(_SCRIPTS / 'score_write_triage_reader.py', 'score_write_triage_reader')
    _stats = load_script(_SCRIPTS / 'write_triage_stats.py', 'write_triage_stats')

Cell = _rubric.Cell

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 7161
NO_DECISION = 'no_decision'
"""The confusion-matrix column of a sampled child recon gave no decision for."""
PROVISIONAL_HARM_WEIGHTS = _reader.PROVISIONAL_HARM_WEIGHTS


class Decision(StrEnum):
    """What recon does with one attached child."""

    KEEP = 'keep'
    DETACH = 'detach'
    RELABEL = 'relabel'
    CONTEST_UPHELD = 'contest_upheld'
    SUPERSEDED_PARENT = 'superseded_parent'
    NOT_A_CONTEST = 'not_a_contest'


RELABEL_TARGETS = frozenset({OUTCOME_AMENDED, OUTCOME_CONTESTED})
"""The outcomes a relabel can give a child."""


@dataclass(frozen=True)
class Action:
    """One repair: a decision, and the outcome a relabel gives (None for every other decision)."""

    decision: Decision
    relabel_to: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, Decision):
            raise ValueError(f'decision {self.decision!r} is not one of {sorted(Decision)}')
        if self.decision is Decision.RELABEL and self.relabel_to not in RELABEL_TARGETS:
            raise ValueError(
                f'decision {self.decision.value!r} needs relabel_to, one of'
                f' {sorted(RELABEL_TARGETS)}; got {self.relabel_to!r}'
            )
        if self.decision is not Decision.RELABEL and self.relabel_to is not None:
            raise ValueError(
                f'decision {self.decision.value!r} takes no relabel_to; got {self.relabel_to!r}'
            )

    @property
    def label(self) -> str:
        """The action as one word: the decision, or ``relabel-><outcome>``."""
        if self.relabel_to is None:
            return self.decision.value
        return f'{self.decision.value}->{self.relabel_to}'


KEEP = Action(Decision.KEEP)
DETACH = Action(Decision.DETACH)
CONTEST_UPHELD = Action(Decision.CONTEST_UPHELD)
SUPERSEDED_PARENT = Action(Decision.SUPERSEDED_PARENT)
NOT_A_CONTEST = Action(Decision.NOT_A_CONTEST)
RELABEL_AMENDED = Action(Decision.RELABEL, OUTCOME_AMENDED)
RELABEL_CONTESTED = Action(Decision.RELABEL, OUTCOME_CONTESTED)

_ALLOWED: Mapping[str, frozenset[Action]] = MappingProxyType({
    OUTCOME_CONTESTED: frozenset({CONTEST_UPHELD, NOT_A_CONTEST, DETACH, SUPERSEDED_PARENT}),
    OUTCOME_AMENDED: frozenset({KEEP, RELABEL_CONTESTED, DETACH, SUPERSEDED_PARENT}),
    OUTCOME_RESTATED: frozenset({KEEP, RELABEL_AMENDED, RELABEL_CONTESTED, DETACH, SUPERSEDED_PARENT}),
})


def allowed_actions(outcome: str) -> frozenset[Action]:
    """Every repair recon may choose for a child triage filed as *outcome*."""
    allowed = _ALLOWED.get(outcome)
    if allowed is None:
        raise ValueError(f'outcome {outcome!r} has no repair vocabulary; attach outcomes: {sorted(_ALLOWED)}')
    return allowed


def action_for(outcome: str, decision: str, relabel_to: str | None = None) -> Action:
    """The action *decision* and *relabel_to* name, refused unless *outcome* allows it."""
    try:
        action = Action(Decision(decision), relabel_to)
    except ValueError as exc:
        raise ValueError(
            f'outcome {outcome!r}: decision {decision!r} with relabel_to {relabel_to!r} is refused: {exc}'
        ) from exc
    allowed = allowed_actions(outcome)
    if action not in allowed:
        raise ValueError(
            f'outcome {outcome!r} does not allow decision {decision!r} with relabel_to'
            f' {relabel_to!r}; it allows {sorted(choice.label for choice in allowed)}'
        )
    return action


_CONFLICT_CELLS = frozenset({Cell.DIGEST_CONFLICT, Cell.HIDDEN_CONFLICT})


def expected_action(outcome: str, decided: _rubric.Decided) -> Action:
    """The repair that clears *decided*'s wrong cells, by one fixed precedence.

    Supersession, then a true dispute (an upheld contest, or a relabel to
    contested), then a wrong home (detach), then a false contest, then hidden
    detail (relabel to amended); otherwise the outcome stands.
    """
    allowed_actions(outcome)
    cells = decided.triage_cells
    if Cell.ATTACHED_SUPERSESSION in cells:
        return SUPERSEDED_PARENT
    if outcome == OUTCOME_CONTESTED:
        if decided.conflict:
            return CONTEST_UPHELD
        return DETACH if Cell.WRONG_HOME in cells else NOT_A_CONTEST
    if cells & _CONFLICT_CELLS:
        return RELABEL_CONTESTED
    if Cell.WRONG_HOME in cells:
        return DETACH
    if outcome == OUTCOME_RESTATED and Cell.HIDDEN_DETAIL in cells:
        return RELABEL_AMENDED
    return KEEP


def repaired_item(row: Mapping[str, Any], action: Action) -> dict[str, Any]:
    """A new item row: the write as it reads after *action*; *row* is left untouched.

    A detach or a supersession stores the write standalone against its old
    parent, a supersession also marking that parent; a relabel or a refuted
    contest re-files the write under its new outcome.
    """
    outcome = row['outcome']
    if action not in allowed_actions(outcome):
        raise ValueError(
            f'item {row.get("entry_id")!r}: outcome {outcome!r} does not allow {action.label!r}'
        )
    repaired = {
        key: list(value) if isinstance(value, list) else value for key, value in row.items()
    }
    if action.decision in (Decision.DETACH, Decision.SUPERSEDED_PARENT):
        parent = row['target_id']
        repaired.update(
            outcome=OUTCOME_STORED, target_id=None, earlier_ids=[parent], supersedes_target=False,
            superseded_marks=[parent] if action.decision is Decision.SUPERSEDED_PARENT else [],
        )
    elif action.decision is Decision.RELABEL:
        repaired['outcome'] = action.relabel_to
    elif action.decision is Decision.NOT_A_CONTEST:
        repaired['outcome'] = OUTCOME_AMENDED
    return repaired


def sample_stratum(outcome: str, project: str) -> str:
    """The replay sample's stratum of a child: its triage outcome and its project."""
    return f'{outcome}|{project}'


def derive_item(
    row: Mapping[str, Any], judgements: Mapping[_rubric.PairKey, _rubric.Judgement],
) -> _rubric.Derivation:
    """*row*'s current-basis derivation, read the way ``score_write_triage_reader`` reads an item."""
    item = _reader.ReaderItem.from_row(0, row)
    return _rubric.derive(
        item.outcome,
        {
            target: judgements.get(_rubric.PairKey(item.entry_id, target, _rubric.Basis.CURRENT))
            for target in item.targets
        },
        supersedes_target=item.supersedes_target,
        superseded_marks=item.superseded_marks,
        in_m1_band=item.band == _rubric.M1_BAND,
    )


def corpus_derivations(
    items: Iterable[Mapping[str, Any]], verdicts: Iterable[Mapping[str, Any]],
) -> dict[str, _rubric.Derivation]:
    """Each corpus item's derivation under *verdicts*, by entry id."""
    judgements = _rubric.resolve_judgements(verdicts)
    return {row['entry_id']: derive_item(row, judgements) for row in items}


@dataclass(frozen=True)
class _Replayed:
    """One sampled child: the truth, the repair it needed, and recon's repair priced."""

    entry_id: str
    outcome: str
    project: str
    stratum: str
    headline: bool
    live_recon_state: str | None
    truth: _rubric.Decided
    expected: Action
    action: Action | None
    repaired: _rubric.Derivation

    @property
    def repaired_decided(self) -> _rubric.Decided | None:
        return self.repaired if isinstance(self.repaired, _rubric.Decided) else None

    @property
    def correct(self) -> bool:
        return self.action == self.expected


_Predicate = Callable[[_Replayed], bool]

_DERIVATION_KINDS: Mapping[type, str] = MappingProxyType({
    _rubric.Incomplete: 'incomplete', _rubric.Unclear: 'unclear',
})


def score_replay(
    sample_rows: Iterable[Mapping[str, Any]],
    corpus_items: Iterable[Mapping[str, Any]],
    verdicts: Iterable[Mapping[str, Any]],
    decisions: Iterable[Mapping[str, Any]],
    *,
    rubric: str,
    strata_populations: Mapping[str, int],
    harm_weights: Mapping[str, float] = PROVISIONAL_HARM_WEIGHTS,
) -> dict[str, Any]:
    """Score recon's *decisions* on the sampled children against the rated corpus.

    *rubric* labels the brief the verdicts were rated under. Headline figures
    read only rows flagged ``headline``; every row carrying a live recon state
    is also tabled in ``touched``. A sampled child with no decision is a miss
    and is never a detach, and its harm is the judge's. The report carries no
    timestamps, so the same inputs always give the same report.
    """
    weights = _validated_weights(harm_weights)
    items = {row['entry_id']: row for row in corpus_items}
    judgements = _rubric.resolve_judgements(verdicts)
    sample = _sample(sample_rows, items)
    chosen = _actions(decisions, sample)
    replayed = [
        _replay(row, items[row['entry_id']], chosen.get(row['entry_id']), judgements)
        for row in sorted(sample.values(), key=lambda row: row['entry_id'])
    ]
    headline = [r for r in replayed if r.headline]
    stratum_weights = _stratum_weights(headline, items.values(), strata_populations)
    projects = sorted({r.project for r in headline})
    return {
        'rubric': rubric,
        'harm_weights': {str(cell): weight for cell, weight in weights.items()},
        'harm_weights_status': (
            _reader.HARM_WEIGHTS_STATUS
            if dict(weights) == dict(PROVISIONAL_HARM_WEIGHTS) else 'custom'
        ),
        'bootstrap': {'resamples': BOOTSTRAP_RESAMPLES, 'seed': BOOTSTRAP_SEED},
        'sample': {
            'n_headline': len(headline),
            'n_report_only': len(replayed) - len(headline),
            'n_touched': sum(r.live_recon_state is not None for r in replayed),
            'by_stratum': dict(sorted(Counter(r.stratum for r in headline).items())),
            'stratum_weights': {
                stratum: round(weight, 4) for stratum, weight in stratum_weights.items()
            },
        },
        'headline': _metric_blocks(headline, stratum_weights),
        'action_accuracy_by_outcome': {
            outcome: _action_accuracy([r for r in headline if r.outcome == outcome], stratum_weights)
            for outcome in sorted({r.outcome for r in headline})
        },
        'by_project': {
            project: _metric_blocks([r for r in headline if r.project == project], stratum_weights)
            for project in projects
        },
        'confusion': _confusion(headline),
        'harm': _harm(headline, weights, stratum_weights, projects),
        'no_decision': _no_decision(headline),
        'touched': [_touched_row(r, weights) for r in replayed if r.live_recon_state is not None],
    }


def _validated_weights(harm_weights: Mapping[str, float]) -> Mapping[str, float]:
    cells = sorted(cell.value for cell in Cell)
    for key, weight in harm_weights.items():
        if key not in cells:
            raise ValueError(f'harm weight {key!r}: {weight!r} names no cell; the cells are {cells}')
        if isinstance(weight, bool) or not isinstance(weight, Real) or not (
            math.isfinite(weight) and weight >= 0
        ):
            raise ValueError(f'harm weight {key!r}: {weight!r} is not a finite non-negative number')
    return MappingProxyType(dict(harm_weights))


_SAMPLE_FACTS = ('target_id', 'outcome', 'project')


def _entry_id(row: Mapping[str, Any], what: str) -> str:
    entry = row.get('entry_id')
    if not isinstance(entry, str) or not entry:
        raise ValueError(f'{what} {dict(row)!r}: entry_id {entry!r} is not a non-empty string')
    return entry


def _sample(
    rows: Iterable[Mapping[str, Any]], items: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Every sample row by entry id, refused unless it agrees with its corpus item."""
    sample: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        entry = _entry_id(row, 'sample row')
        if entry in sample:
            raise ValueError(f'sample row {entry!r}: duplicate entry_id')
        item = items.get(entry)
        if item is None:
            raise ValueError(f'sample row {entry!r}: no corpus item has this entry_id')
        disagree = [key for key in _SAMPLE_FACTS if row.get(key) != item[key]]
        if disagree:
            raise ValueError(
                f'sample row {entry!r}: {", ".join(disagree)} disagree with the corpus item:'
                f' {[(row.get(key), item[key]) for key in disagree]}'
            )
        if row.get('stratum') != sample_stratum(row['outcome'], row['project']):
            raise ValueError(
                f'sample row {entry!r}: stratum {row.get("stratum")!r} is not'
                f' {sample_stratum(row["outcome"], row["project"])!r}'
            )
        if not isinstance(row.get('headline'), bool):
            raise ValueError(f'sample row {entry!r}: headline {row.get("headline")!r} is not a bool')
        sample[entry] = row
    return sample


def _actions(
    decisions: Iterable[Mapping[str, Any]], sample: Mapping[str, Mapping[str, Any]],
) -> dict[str, Action]:
    """Each decided child's action, refused unless it is sampled, unique and allowed."""
    chosen: dict[str, Action] = {}
    for row in decisions:
        entry = _entry_id(row, 'decision')
        sampled = sample.get(entry)
        if sampled is None:
            raise ValueError(f'decision for {entry!r}: the child is not in the sample')
        if entry in chosen:
            raise ValueError(f'decision for {entry!r}: a second decision for the same child')
        if row.get('target_id') != sampled['target_id']:
            raise ValueError(
                f'decision for {entry!r}: target_id {row.get("target_id")!r} is not the'
                f' sampled parent {sampled["target_id"]!r}'
            )
        decision = row.get('decision')
        if not isinstance(decision, str):
            raise ValueError(f'decision for {entry!r}: decision {decision!r} is not a string')
        try:
            chosen[entry] = action_for(sampled['outcome'], decision, row.get('relabel_to'))
        except ValueError as exc:
            raise ValueError(f'decision for {entry!r}: {exc}') from exc
    return chosen


def _replay(
    row: Mapping[str, Any],
    item: Mapping[str, Any],
    action: Action | None,
    judgements: Mapping[_rubric.PairKey, _rubric.Judgement],
) -> _Replayed:
    truth = derive_item(item, judgements)
    if not isinstance(truth, _rubric.Decided):
        raise ValueError(f'sample row {row["entry_id"]!r}: its rated truth is not Decided: {truth}')
    return _Replayed(
        entry_id=row['entry_id'],
        outcome=row['outcome'],
        project=row['project'],
        stratum=row['stratum'],
        headline=row['headline'],
        live_recon_state=row.get('live_recon_state'),
        truth=truth,
        expected=expected_action(row['outcome'], truth),
        action=action,
        repaired=truth if action is None else derive_item(repaired_item(item, action), judgements),
    )


def _stratum_weights(
    headline: Sequence[_Replayed],
    corpus_items: Iterable[Mapping[str, Any]],
    strata_populations: Mapping[str, int],
) -> dict[str, float]:
    """Each sample stratum's weight: its estimated population over the children drawn from it.

    A sample stratum's population is, over each corpus stratum, that stratum's
    population times its corpus items' share falling in the sample stratum.
    """
    corpus = list(corpus_items)
    corpus_sizes = Counter(item['stratum'] for item in corpus)
    shares = Counter(
        (sample_stratum(item['outcome'], item['project']), item['stratum']) for item in corpus
    )
    weights: dict[str, float] = {}
    for stratum, drawn in sorted(Counter(r.stratum for r in headline).items()):
        sources = sorted(source for sampled, source in shares if sampled == stratum)
        missing = [source for source in sources if source not in strata_populations]
        if missing:
            raise ValueError(
                f'sample stratum {stratum!r} has no weight: corpus strata {missing} have no'
                f' population; populations given: {sorted(strata_populations)}'
            )
        population = sum(
            strata_populations[source] * shares[stratum, source] / corpus_sizes[source]
            for source in sources
        )
        weights[stratum] = population / drawn
    return weights


def _rate_block(
    rows: Sequence[_Replayed],
    member: _Predicate,
    hit: _Predicate,
    weights: Mapping[str, float],
) -> dict[str, Any]:
    """*hit* among the *member* rows: the sample rate and the population-weighted rate.

    Every row is a unit of its stratum, counted in the denominator only when it
    is a member, so the bootstrap keeps each stratum's size.
    """
    units = [
        _stats.StratumUnit(r.stratum, float(member(r) and hit(r)), float(member(r))) for r in rows
    ]
    n = sum(member(r) for r in rows)
    hits = sum(member(r) and hit(r) for r in rows)
    return {
        'n': n,
        'hits': hits,
        'rate': _stats.rate(hits, n),
        'wilson95': _stats.wilson95(hits, n),
        'population_rate': _stats.stratified_ratio(units, weights) if n else None,
        'population_interval95': _bootstrap(units, weights) if n else None,
    }


def _bootstrap(units: Sequence[_stats.StratumUnit], weights: Mapping[str, float]) -> list[float] | None:
    return _stats.stratified_bootstrap95(
        units, weights, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
    )


def _judge_wrong_home(r: _Replayed) -> bool:
    return Cell.WRONG_HOME in r.truth.triage_cells


def _contested(r: _Replayed) -> bool:
    return r.outcome == OUTCOME_CONTESTED


def _every(_: _Replayed) -> bool:
    return True


def _repair_clears_wrong_home(r: _Replayed) -> bool:
    repaired = r.repaired_decided
    return r.action is not None and repaired is not None and Cell.WRONG_HOME not in repaired.triage_cells


def _detached(r: _Replayed) -> bool:
    return r.action is not None and r.action.decision is Decision.DETACH


def _correct(r: _Replayed) -> bool:
    return r.correct


def _misfile_detection(rows: Sequence[_Replayed], weights: Mapping[str, float]) -> dict[str, Any]:
    """Among the judge's wrong homes, the share recon's repair leaves with no wrong home."""
    return _rate_block(rows, _judge_wrong_home, _repair_clears_wrong_home, weights)


def _wrong_detach_rate(rows: Sequence[_Replayed], weights: Mapping[str, float]) -> dict[str, Any]:
    """Among children rightly homed, the share recon detached, and the true contests among them."""
    def rightly_homed(r: _Replayed) -> bool:
        return not _judge_wrong_home(r)

    return {
        **_rate_block(rows, rightly_homed, _detached, weights),
        'correct_contests_detached': sum(
            rightly_homed(r) and _contested(r) and r.truth.conflict and _detached(r) for r in rows
        ),
    }


def _contest_resolution_accuracy(
    rows: Sequence[_Replayed], weights: Mapping[str, float],
) -> dict[str, Any]:
    """Among contested children, the share whose repair is exactly the expected one."""
    return _rate_block(rows, _contested, _correct, weights)


def _action_accuracy(rows: Sequence[_Replayed], weights: Mapping[str, float]) -> dict[str, Any]:
    """The share of children whose repair is exactly the expected one."""
    return _rate_block(rows, _every, _correct, weights)


def _judge_wrong_home_all(rows: Sequence[_Replayed], weights: Mapping[str, float]) -> dict[str, Any]:
    """The judge's wrong-home share over every child (rubric v2.2 M4 on these rows)."""
    return _rate_block(rows, _every, _judge_wrong_home, weights)


def _judge_wrong_home_contested(
    rows: Sequence[_Replayed], weights: Mapping[str, float],
) -> dict[str, Any]:
    """The judge's wrong-home share over contested children."""
    return _rate_block(rows, _contested, _judge_wrong_home, weights)


_METRICS: Mapping[str, Callable[[Sequence[_Replayed], Mapping[str, float]], dict[str, Any]]] = (
    MappingProxyType({
        'misfile_detection': _misfile_detection,
        'wrong_detach_rate': _wrong_detach_rate,
        'contest_resolution_accuracy': _contest_resolution_accuracy,
        'action_accuracy': _action_accuracy,
        'judge_wrong_home_all': _judge_wrong_home_all,
        'judge_wrong_home_contested': _judge_wrong_home_contested,
    })
)
METRIC_NAMES = tuple(_METRICS)


def _metric_blocks(rows: Sequence[_Replayed], weights: Mapping[str, float]) -> dict[str, Any]:
    return {name: metric(rows, weights) for name, metric in _METRICS.items()}


def _recon_label(r: _Replayed) -> str:
    return NO_DECISION if r.action is None else r.action.label


def _confusion(rows: Sequence[_Replayed]) -> dict[str, dict[str, int]]:
    """How many children of each expected repair recon gave each repair."""
    counts = Counter((r.expected.label, _recon_label(r)) for r in rows)
    return {
        expected: {
            recon: count for (row_expected, recon), count in sorted(counts.items())
            if row_expected == expected
        }
        for expected in sorted({expected for expected, _ in counts})
    }


def _write_harm(cells: frozenset[_rubric.Cell], weights: Mapping[str, float]) -> float:
    return sum(weights.get(cell, 0) for cell in sorted(cells))


def _harm(
    rows: Sequence[_Replayed],
    weights: Mapping[str, float],
    stratum_weights: Mapping[str, float],
    projects: Sequence[str],
) -> dict[str, Any]:
    """Harm per write as the judge filed, as recon repaired, and stored flat, over priced rows."""
    priced = [r for r in rows if r.repaired_decided is not None]
    return {
        **_harm_figures(priced, weights, stratum_weights),
        'unpriced': [
            {'entry_id': r.entry_id, 'action': _recon_label(r),
             'derivation': _DERIVATION_KINDS[type(r.repaired)]}
            for r in rows if r.repaired_decided is None
        ],
        'by_project': {
            project: _harm_figures(
                [r for r in priced if r.project == project], weights, stratum_weights,
            )
            for project in projects
        },
    }


_REGIMES: Mapping[str, Callable[[_Replayed], frozenset[_rubric.Cell]]] = MappingProxyType({
    'judge': lambda r: r.truth.triage_cells,
    'recon': lambda r: _priced(r).triage_cells,
    'status_quo': lambda r: r.truth.status_quo_cells,
})


def _priced(r: _Replayed) -> _rubric.Decided:
    repaired = r.repaired_decided
    if repaired is None:
        raise ValueError(f'child {r.entry_id!r}: recon\'s repair has no priced derivation')
    return repaired


def _harm_figures(
    priced: Sequence[_Replayed], weights: Mapping[str, float], stratum_weights: Mapping[str, float],
) -> dict[str, Any]:
    harms = {
        regime: [_stats.StratumUnit(r.stratum, _write_harm(cells(r), weights), 1.0) for r in priced]
        for regime, cells in _REGIMES.items()
    }
    delta = [
        _stats.StratumUnit(r.stratum, mine.numerator - judge.numerator, 1.0)
        for r, mine, judge in zip(priced, harms['recon'], harms['judge'], strict=True)
    ]
    uniform = {stratum: 1.0 for stratum in stratum_weights}
    empty = not priced
    return {
        'n': len(priced),
        'per_write': {regime: _stats.stratified_ratio(units, uniform) for regime, units in harms.items()},
        'delta_per_write': _stats.stratified_ratio(delta, uniform),
        'delta_interval95': None if empty else _bootstrap(delta, uniform),
        'population_per_write': {
            regime: _stats.stratified_ratio(units, stratum_weights) for regime, units in harms.items()
        },
        'population_delta_per_write': _stats.stratified_ratio(delta, stratum_weights),
        'population_delta_interval95': None if empty else _bootstrap(delta, stratum_weights),
    }


def _no_decision(rows: Sequence[_Replayed]) -> dict[str, Any]:
    missing = [r.entry_id for r in rows if r.action is None]
    return {'count': len(missing), 'entry_ids': missing}


def _touched_row(r: _Replayed, weights: Mapping[str, float]) -> dict[str, Any]:
    repaired = r.repaired_decided
    return {
        'entry_id': r.entry_id,
        'project': r.project,
        'outcome': r.outcome,
        'headline': r.headline,
        'live_recon_state': r.live_recon_state,
        'truth_cells': sorted(str(cell) for cell in r.truth.triage_cells),
        'expected': r.expected.label,
        'recon': None if r.action is None else r.action.label,
        'correct': r.correct,
        'judge_harm': _write_harm(r.truth.triage_cells, weights),
        'recon_harm': None if repaired is None else _write_harm(repaired.triage_cells, weights),
    }


_RATE_KEYS = ('n', 'hits', 'rate', 'wilson95', 'population_rate', 'population_interval95')
_RATE_HEADER = ('n', 'hits', 'rate', 'Wilson 95%', 'population rate', 'population 95%')
_HARM_REGIMES = ('judge', 'recon', 'status_quo')


def render_markdown(report: Mapping[str, Any]) -> str:
    """*report* as Markdown tables; the same report always renders to the same text."""
    sections = [
        _summary_markdown(report),
        _headline_markdown(report),
        _by_outcome_markdown(report),
        _by_project_markdown(report),
        _confusion_markdown(report),
        _harm_markdown(report),
        _missing_markdown(report),
        _touched_markdown(report),
    ]
    return '\n\n'.join(sections) + '\n'


def _shown(value: Any) -> str:
    if value is None:
        return '—'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    return str(value)


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [f'| {" | ".join(header)} |', f'|{"---|" * len(header)}']
    lines += [f'| {" | ".join(_shown(cell) for cell in row)} |' for row in rows]
    return '\n'.join(lines)


def _listed(counts: Mapping[str, Any]) -> str:
    return ', '.join(f'{key} {value}' for key, value in counts.items())


def _rate_cells(block: Mapping[str, Any]) -> list[Any]:
    return [block[key] for key in _RATE_KEYS]


def _summary_markdown(report: Mapping[str, Any]) -> str:
    sample = report['sample']
    return '\n'.join([
        f'# Recon triage replay scores (rubric {report["rubric"]})',
        '',
        f'Headline children: {sample["n_headline"]}; report-only: {sample["n_report_only"]};'
        f' touched (live recon state known): {sample["n_touched"]}.',
        f'Sampled per stratum: {_listed(sample["by_stratum"])}.',
        f'Stratum weights: {_listed(sample["stratum_weights"])}.',
        f'Bootstrap: {report["bootstrap"]["resamples"]} resamples, seed {report["bootstrap"]["seed"]}.',
    ])


def _headline_markdown(report: Mapping[str, Any]) -> str:
    rows = [
        [metric, *_rate_cells(block), block.get('correct_contests_detached')]
        for metric, block in report['headline'].items()
    ]
    return '\n'.join([
        '## Headline',
        '',
        _table(['metric', *_RATE_HEADER, 'correct contests detached'], rows),
    ])


def _by_outcome_markdown(report: Mapping[str, Any]) -> str:
    rows = [[outcome, *_rate_cells(block)] for outcome, block in report['action_accuracy_by_outcome'].items()]
    return '\n'.join([
        '## Action accuracy by outcome',
        '',
        _table(['outcome', *_RATE_HEADER], rows),
    ])


def _by_project_markdown(report: Mapping[str, Any]) -> str:
    rows = [
        [project, metric, *_rate_cells(block), block.get('correct_contests_detached')]
        for project, metrics in report['by_project'].items()
        for metric, block in metrics.items()
    ]
    return '\n'.join([
        '## By project',
        '',
        _table(['project', 'metric', *_RATE_HEADER, 'correct contests detached'], rows),
    ])


def _confusion_markdown(report: Mapping[str, Any]) -> str:
    confusion = report['confusion']
    columns = sorted({recon for row in confusion.values() for recon in row})
    rows = [
        [expected, *(row.get(column, 0) for column in columns)]
        for expected, row in confusion.items()
    ]
    return '\n'.join([
        '## Confusion (expected repair by recon repair)',
        '',
        _table(['expected', *columns], rows),
    ])


def _harm_markdown(report: Mapping[str, Any]) -> str:
    harm = report['harm']
    rows: list[list[Any]] = []
    for scope, block in [('all', harm), *harm['by_project'].items()]:
        rows.append([
            scope, 'sample', block['n'], *(block['per_write'][regime] for regime in _HARM_REGIMES),
            block['delta_per_write'], block['delta_interval95'],
        ])
        rows.append([
            scope, 'population', block['n'],
            *(block['population_per_write'][regime] for regime in _HARM_REGIMES),
            block['population_delta_per_write'], block['population_delta_interval95'],
        ])
    return '\n'.join([
        f'## Harm per write (weights: {report["harm_weights_status"]})',
        '',
        _table(
            ['scope', 'figure', 'n priced', 'judge', 'recon', 'status quo', 'recon − judge',
             'recon − judge 95%'],
            rows,
        ),
        '',
        f'Weights: {_listed(report["harm_weights"])}.',
    ])


def _missing_markdown(report: Mapping[str, Any]) -> str:
    missing = report['no_decision']
    unpriced = report['harm']['unpriced']
    return '\n'.join([
        '## Missing and unpriced',
        '',
        _table(['kind', 'count', 'children'], [
            ['no_decision', missing['count'], ', '.join(missing['entry_ids'])],
            ['unpriced', len(unpriced),
             ', '.join(f'{row["entry_id"]} ({row["action"]}, {row["derivation"]})' for row in unpriced)],
        ]),
    ])


def _touched_markdown(report: Mapping[str, Any]) -> str:
    rows = [
        [row['entry_id'], row['project'], row['outcome'], row['headline'], row['live_recon_state'],
         ', '.join(row['truth_cells']) or None, row['expected'], row['recon'], row['correct'],
         row['judge_harm'], row['recon_harm']]
        for row in report['touched']
    ]
    return '\n'.join([
        '## Touched children (recon or an agent acted on them live)',
        '',
        _table(['entry', 'project', 'outcome', 'headline', 'live recon state', 'truth cells',
                'expected', 'recon', 'correct', 'judge harm', 'recon harm'], rows),
    ])
