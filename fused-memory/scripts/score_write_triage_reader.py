#!/usr/bin/env python3
"""Score write-triage outcomes by what a reader of the corpus meets (rubric v2).

An *item* is one write and the outcome triage gave it: attached under a
target as restated, amended or contested, or stored standalone with the
earlier records it might have joined. Raters judge each (write, record) pair
on rubric v2's binaries; ``write_triage_reader_rubric.py`` resolves their
verdict rows and derives each write's wrong cells under triage and under the
status quo, where every write is stored flat. This module turns the cells
into the reader-consequence metrics M1-M6, rater agreement, and the
comparative harm per write.

The metric definitions, populations, sample sizes and gates are
``fused-memory/calibration/write_triage_reader_metrics.md``. This scorer
reports and never gates.

Items
-----
One JSON row per write; :meth:`ReaderItem.from_row` states and checks the
contract. A ``strata`` map ``{stratum: population}`` makes every proportion
also report a population rate, each sampled item weighted N_h / n_h.

Refusal
-------
Every binary the derivation needs on the ``current`` basis must be rated and
not tied. Otherwise :func:`score_reader` raises
:class:`IncompleteJudgementError`, naming each gap, and scores nothing: an
incomplete corpus must not read as a result. A binary the derivation never
asks for needs no rating, and a missing ``attach_time`` judgement never
refuses (M6 is report-only).
"""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple

from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    TRIAGE_OUTCOMES,
)

_SCRIPTS = Path(__file__).resolve().parent


def _load_script(path: Path, mod_name: str) -> types.ModuleType:
    """Load a ``scripts/`` sibling by path, cached in ``sys.modules``."""
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


if TYPE_CHECKING:
    import write_triage_reader_rubric as _rubric
    import write_triage_stats as _stats
else:
    _rubric = _load_script(_SCRIPTS / 'write_triage_reader_rubric.py', 'write_triage_reader_rubric')
    _stats = _load_script(_SCRIPTS / 'write_triage_stats.py', 'write_triage_stats')

Cell = _rubric.Cell

KAPPA_FLOOR = 0.70
KAPPA_MIN_OVERLAP = 60
"""A binary's primary/second agreement is good enough to score when kappa >= KAPPA_FLOOR
on at least KAPPA_MIN_OVERLAP doubly rated pairs."""

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 7157

PROVISIONAL_HARM_WEIGHTS: Mapping[str, float] = MappingProxyType({
    Cell.HIDDEN_CONFLICT: 3,
    Cell.DIGEST_CONFLICT: 2,
    Cell.FALSE_CONFLICT: 2,
    Cell.ORPHANED_CONFLICT: 1.5,
    Cell.HIDDEN_DETAIL: 1,
    Cell.WRONG_HOME: 1,
    Cell.ATTACHED_SUPERSESSION: 1,
    Cell.DUPLICATE_CLUTTER: 0.5,
})
"""The re-based harm of each cell, provisional until Leo ratifies; a cell left out weighs 0.

The basis of each weight is ``calibration/write_triage_reader_metrics.md`` §weights.
"""

HARM_WEIGHTS_STATUS = 'provisional'

_M1_BAND = OUTCOME_RESTATED
_BANDS = frozenset({OUTCOME_RESTATED, OUTCOME_JUDGE, OUTCOME_STORED})
_ATTACH_OUTCOMES = TRIAGE_OUTCOMES - {OUTCOME_STORED}
_REQUIRED_ITEM_KEYS = ('entry_id', 'outcome', 'band', 'project', 'stratum')
_FLAG_KEYS = ('via_sibling', 'cross_category', 'parent_edited_after_attach')
_ENDINGS = (OUTCOME_RESTATED, OUTCOME_AMENDED, 'stored_unlinked', OUTCOME_CONTESTED, 'stored_linked')
"""Where a CONFLICT write can end up: its attach outcome, or stored with or without a mark."""
_HIDDEN_ENDINGS = frozenset({OUTCOME_RESTATED, OUTCOME_AMENDED, 'stored_unlinked'})
_SCOPE_LABELS: Mapping[str, str] = MappingProxyType({
    _rubric.Resolved.CENTRAL: 'central', _rubric.Resolved.SECONDARY: 'secondary',
})


@dataclass(frozen=True)
class ReaderItem:
    """One write, the outcome triage gave it, and the facts the metrics split it by.

    An attach outcome names its ``target_id`` (the parent) and no
    ``earlier_ids``; a stored write names no target and lists ``earlier_ids``,
    the earlier records it might have joined. ``band`` is the triage band
    (None when unknown). The three flags are None when unknown.
    ``supersedes_target`` (the write declares it supersedes its parent) and
    ``superseded_marks`` (earlier records marked superseded by this write)
    are metadata facts for ruling 5c.
    """

    entry_id: str
    outcome: str
    target_id: str | None
    earlier_ids: tuple[str, ...]
    band: str | None
    project: str
    stratum: str
    via_sibling: bool | None
    cross_category: bool | None
    parent_edited_after_attach: bool | None
    supersedes_target: bool
    superseded_marks: frozenset[str]

    @classmethod
    def from_row(cls, index: int, row: Mapping[str, Any]) -> ReaderItem:
        violation = _item_violation(row)
        if violation is not None:
            raise ValueError(f'item row {index} (entry_id {row.get("entry_id")!r}): {violation}')
        return cls(
            entry_id=row['entry_id'],
            outcome=row['outcome'],
            target_id=row.get('target_id'),
            earlier_ids=tuple(row.get('earlier_ids') or ()),
            band=row['band'],
            project=row['project'],
            stratum=row['stratum'],
            via_sibling=row.get('via_sibling'),
            cross_category=row.get('cross_category'),
            parent_edited_after_attach=row.get('parent_edited_after_attach'),
            supersedes_target=row.get('supersedes_target', False),
            superseded_marks=frozenset(row.get('superseded_marks') or ()),
        )

    @property
    def targets(self) -> tuple[str, ...]:
        """The records this write is judged against."""
        if self.target_id is None:
            return self.earlier_ids
        return (self.target_id,)


def _item_violation(row: Mapping[str, Any]) -> str | None:
    """The first item-row invariant *row* breaks, naming the offending value; None if none."""
    missing = [key for key in _REQUIRED_ITEM_KEYS if key not in row]
    if missing:
        return f'lacks {", ".join(missing)}'
    for key in ('entry_id', 'project', 'stratum'):
        if not isinstance(row[key], str) or not row[key]:
            return f'{key} {row[key]!r} is not a non-empty string'
    if row['outcome'] not in TRIAGE_OUTCOMES:
        return f'outcome {row["outcome"]!r} is not one of {sorted(TRIAGE_OUTCOMES)}'
    if row['band'] is not None and row['band'] not in _BANDS:
        return f'band {row["band"]!r} is not one of {sorted(_BANDS)} or null'
    for key in (*_FLAG_KEYS, 'supersedes_target'):
        if row.get(key) is not None and not isinstance(row[key], bool):
            return f'{key} {row[key]!r} is not a bool or null'
    earlier = row.get('earlier_ids') or []
    marks = row.get('superseded_marks') or []
    if not all(isinstance(record, str) and record for record in (*earlier, *marks)):
        return f'earlier_ids {earlier!r} and superseded_marks {marks!r} must list record ids'
    target = row.get('target_id')
    if row['outcome'] == OUTCOME_STORED:
        if target is not None:
            return f'a stored write has no target_id, got {target!r}'
        if row.get('supersedes_target'):
            return 'a stored write has no target to supersede, so supersedes_target must be false'
        if not set(marks) <= set(earlier):
            return f'superseded_marks {sorted(set(marks) - set(earlier))} are not earlier_ids'
        return None
    if not isinstance(target, str) or not target:
        return f'a {row["outcome"]} write needs a target_id, got {target!r}'
    if earlier or marks:
        return f'a {row["outcome"]} write has no earlier_ids or superseded_marks, got {earlier!r}'
    return None


class JudgementGap(NamedTuple):
    """A decisive binary of one (write, record) pair the verdicts leave unrated or tied."""

    entry_id: str
    target_id: str
    binary: str
    kind: str


class IncompleteJudgementError(ValueError):
    """The verdicts leave decisive binaries unrated or tied, so nothing is scored."""

    def __init__(self, gaps: tuple[JudgementGap, ...]) -> None:
        self.gaps = gaps
        header = (
            f'verdicts are incomplete: {len(gaps)} decisive binaries unrated or tied'
            ' on the current basis'
        )
        lines = [f'{gap.kind} {gap.entry_id} {gap.target_id} {gap.binary}' for gap in gaps]
        super().__init__('\n'.join([header, *lines]))


@dataclass(frozen=True)
class _Scored:
    """One item with its derivation on each basis and its resolved SCOPE (attach items)."""

    item: ReaderItem
    current: _rubric.Derivation
    attach_time: _rubric.Derivation | None
    scope: _rubric.Resolved | None

    @property
    def decided(self) -> _rubric.Decided | None:
        return self.current if isinstance(self.current, _rubric.Decided) else None


_Predicate = Callable[[_Scored, '_rubric.Decided'], bool]


def score_reader(
    items: Iterable[Mapping[str, Any]],
    verdicts: Iterable[Mapping[str, Any]],
    *,
    strata: Mapping[str, int] | None = None,
    harm_weights: Mapping[str, float] = PROVISIONAL_HARM_WEIGHTS,
) -> dict[str, Any]:
    """Score every item against the verdict corpus; refuse an incomplete one.

    The report carries no timestamps, so scoring the same inputs twice gives
    the same report.
    """
    weights = _validated_weights(harm_weights)
    reader_items = _reader_items(items)
    stratum_weights = _stratum_weights(reader_items, strata)
    judgements = _rubric.resolve_judgements(verdicts)
    scored = [_score(item, judgements) for item in reader_items]
    _refuse_incomplete(scored)
    return {
        'rubric': 'v2',
        'cell_rules': {str(cell): rule for cell, rule in _rubric.CELL_RULES.items()},
        'harm_weights': {str(cell): weight for cell, weight in weights.items()},
        'harm_weights_status': (
            HARM_WEIGHTS_STATUS if dict(weights) == dict(PROVISIONAL_HARM_WEIGHTS) else 'custom'
        ),
        'bootstrap': {'resamples': BOOTSTRAP_RESAMPLES, 'seed': BOOTSTRAP_SEED},
        'population': _population(scored, strata),
        'kappa': _kappa(judgements),
        'cells': _cell_counts(scored),
        'metrics': {
            'm1': _m1(scored, stratum_weights),
            'm2': _m2(scored, stratum_weights),
            'm3': _m3(scored, stratum_weights),
            'm4': _m4(scored, stratum_weights),
            'm5': _m5(scored, stratum_weights),
            'm6': _m6(scored),
        },
        'harm': _harm(scored, weights, stratum_weights),
    }


def _validated_weights(harm_weights: Mapping[str, float]) -> Mapping[_rubric.Cell, float]:
    cells = [cell.value for cell in Cell]
    validated: dict[_rubric.Cell, float] = {}
    for key, weight in harm_weights.items():
        if key not in cells:
            raise ValueError(f'harm weight {key!r}: {weight!r} names no cell; the cells are {cells}')
        if isinstance(weight, bool) or not isinstance(weight, Real) or not (
            math.isfinite(weight) and weight >= 0
        ):
            raise ValueError(
                f'harm weight {key!r}: {weight!r} is not a finite non-negative number;'
                f' the cells are {cells}'
            )
        validated[Cell(key)] = weight
    return MappingProxyType(validated)


def _reader_items(rows: Iterable[Mapping[str, Any]]) -> list[ReaderItem]:
    items: list[ReaderItem] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        item = ReaderItem.from_row(index, row)
        if item.entry_id in seen:
            raise ValueError(f'item row {index} (entry_id {item.entry_id!r}): duplicate entry_id')
        seen.add(item.entry_id)
        items.append(item)
    return sorted(items, key=lambda item: item.entry_id)


def _stratum_weights(
    items: Sequence[ReaderItem], strata: Mapping[str, int] | None,
) -> Mapping[str, float] | None:
    """Each sampled stratum's weight N_h / n_h; None without a strata map."""
    if strata is None:
        return None
    sampled = Counter(item.stratum for item in items)
    for stratum, count in sorted(sampled.items()):
        if stratum not in strata:
            raise ValueError(
                f'item stratum {stratum!r} has no population; strata: {sorted(strata)}'
            )
        population = strata[stratum]
        if isinstance(population, bool) or not isinstance(population, int) or population < count:
            raise ValueError(
                f'stratum {stratum!r}: population {population!r} is not an integer at least'
                f' its {count} sampled items'
            )
    return MappingProxyType({stratum: strata[stratum] / count for stratum, count in sampled.items()})


def _score(item: ReaderItem, judgements: Mapping[_rubric.PairKey, _rubric.Judgement]) -> _Scored:
    def by_target(basis: _rubric.Basis) -> dict[str, _rubric.Judgement | None]:
        return {
            target: judgements.get(_rubric.PairKey(item.entry_id, target, basis))
            for target in item.targets
        }

    current = by_target(_rubric.Basis.CURRENT)
    attach_time = by_target(_rubric.Basis.ATTACH_TIME)
    parent = current.get(item.target_id) if item.target_id is not None else None
    return _Scored(
        item=item,
        current=_derive(item, current, in_m1_band=item.band == _M1_BAND),
        attach_time=(
            _derive(item, attach_time, in_m1_band=False)
            if any(judgement is not None for judgement in attach_time.values()) else None
        ),
        scope=None if parent is None else parent.answer(_rubric.Binary.SCOPE),
    )


def _derive(
    item: ReaderItem, by_target: Mapping[str, _rubric.Judgement | None], *, in_m1_band: bool,
) -> _rubric.Derivation:
    return _rubric.derive(
        item.outcome, by_target, supersedes_target=item.supersedes_target,
        superseded_marks=item.superseded_marks, in_m1_band=in_m1_band,
    )


def _refuse_incomplete(scored: Sequence[_Scored]) -> None:
    gaps = sorted(
        JudgementGap(s.item.entry_id, gap.target_id, str(gap.binary), str(gap.kind))
        for s in scored if isinstance(s.current, _rubric.Incomplete)
        for gap in s.current.missing
    )
    if gaps:
        raise IncompleteJudgementError(tuple(gaps))


def _population(scored: Sequence[_Scored], strata: Mapping[str, int] | None) -> dict[str, Any]:
    outcomes = Counter(s.item.outcome for s in scored)
    unclear = sorted(s.item.entry_id for s in scored if s.decided is None)
    return {
        'n_items': len(scored),
        'by_outcome': {outcome: outcomes[outcome] for outcome in sorted(TRIAGE_OUTCOMES)},
        'by_stratum': dict(sorted(Counter(s.item.stratum for s in scored).items())),
        'strata': None if strata is None else dict(sorted(strata.items())),
        'decided': len(scored) - len(unclear),
        'unclear': len(unclear),
        'unclear_entry_ids': unclear,
    }


def _kappa(judgements: Mapping[_rubric.PairKey, _rubric.Judgement]) -> dict[str, Any]:
    """Cohen's kappa per binary between the primary and second rater of each pair key."""
    pairs: dict[_rubric.Binary, list[tuple[str, str]]] = {binary: [] for binary in _rubric.Binary}
    for judgement in judgements.values():
        primary = [vote for vote in judgement.votes if vote.role is _rubric.Role.PRIMARY]
        second = [vote for vote in judgement.votes if vote.role is _rubric.Role.SECOND]
        if len(primary) != 1 or len(second) != 1:
            continue
        for binary, overlap in pairs.items():
            answers = (primary[0].answers.get(binary), second[0].answers.get(binary))
            if all(answer is not None and answer != 'UNCLEAR' for answer in answers):
                overlap.append((str(answers[0]), str(answers[1])))
    return {
        'floor': KAPPA_FLOOR,
        'min_overlap': KAPPA_MIN_OVERLAP,
        'by_binary': {str(binary): _kappa_block(overlap) for binary, overlap in pairs.items()},
    }


def _kappa_block(overlap: Sequence[tuple[str, str]]) -> dict[str, Any]:
    kappa = _stats.cohen_kappa(overlap)
    return {
        'n_overlap': len(overlap),
        'kappa': kappa,
        'floor_met': (
            kappa is not None and len(overlap) >= KAPPA_MIN_OVERLAP and kappa >= KAPPA_FLOOR
        ),
    }


def _cell_counts(scored: Sequence[_Scored]) -> dict[str, dict[str, int]]:
    decided = [s.decided for s in scored if s.decided is not None]
    return {
        'triage': {str(cell): sum(cell in d.triage_cells for d in decided) for cell in Cell},
        'status_quo': {str(cell): sum(cell in d.status_quo_cells for d in decided) for cell in Cell},
    }


def _proportion(
    scored: Sequence[_Scored],
    member: Callable[[_Scored], bool],
    hit: _Predicate,
    weights: Mapping[str, float] | None,
    *,
    within: _Predicate | None = None,
) -> dict[str, Any]:
    """One metric block: *hit* among the decided *member* items (narrowed by *within*).

    ``unclear`` counts the member items whose derivation was UNCLEAR; they are
    outside n. The population figures are None without stratum weights.
    """
    members = [s for s in scored if member(s)]
    decided = [
        (s, s.decided) for s in members
        if s.decided is not None and (within is None or within(s, s.decided))
    ]
    hits = [hit(s, d) for s, d in decided]
    units = [
        _stats.StratumUnit(s.item.stratum, float(is_hit), 1.0)
        for (s, _), is_hit in zip(decided, hits, strict=True)
    ]
    return {
        'n': len(decided),
        'hits': sum(hits),
        'rate': _rate(sum(hits), len(decided)),
        'wilson95': _stats.wilson95(sum(hits), len(decided)),
        'unclear': sum(s.decided is None for s in members),
        'population_rate': None if weights is None else _stats.stratified_ratio(units, weights),
        'population_interval95': None if weights is None else _bootstrap(units, weights),
    }


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def _bootstrap(
    units: Sequence[_stats.StratumUnit], weights: Mapping[str, float],
) -> list[float] | None:
    return _stats.stratified_bootstrap95(
        units, weights, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
    )


def _m1(scored: Sequence[_Scored], weights: Mapping[str, float] | None) -> dict[str, Any]:
    block = _proportion(
        scored, lambda s: s.item.band == _M1_BAND, lambda _, d: bool(d.m1_hidden), weights,
    )
    return {**block, 'band_unknown': sum(s.item.band is None for s in scored)}


def _m2(scored: Sequence[_Scored], weights: Mapping[str, float] | None) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for outcome in sorted(TRIAGE_OUTCOMES):
        def of_outcome(s: _Scored, outcome: str = outcome) -> bool:
            return s.item.outcome == outcome

        projects: dict[str, Counter[str]] = {}
        for s in scored:
            if of_outcome(s) and s.decided is not None:
                tally = projects.setdefault(s.item.project, Counter())
                tally['n'] += 1
                tally['hits'] += not s.decided.triage_cells
        report[outcome] = {
            **_proportion(scored, of_outcome, lambda _, d: not d.triage_cells, weights),
            'by_project': {
                project: {'n': tally['n'], 'hits': tally['hits']}
                for project, tally in sorted(projects.items())
            },
        }
    return report


def _ending(s: _Scored, decided: _rubric.Decided) -> str:
    """Where a CONFLICT write ended up, as M3 splits it."""
    if s.item.outcome != OUTCOME_STORED:
        return s.item.outcome
    return 'stored_unlinked' if Cell.ORPHANED_CONFLICT in decided.triage_cells else 'stored_linked'


def _m3(scored: Sequence[_Scored], weights: Mapping[str, float] | None) -> dict[str, Any]:
    endings = Counter(
        _ending(s, s.decided) for s in scored if s.decided is not None and s.decided.conflict
    )
    block = _proportion(
        scored, lambda _: True, lambda s, d: _ending(s, d) in _HIDDEN_ENDINGS, weights,
        within=lambda _, d: d.conflict,
    )
    return {**block, 'by_ending': {ending: endings[ending] for ending in _ENDINGS}}


def _m4(scored: Sequence[_Scored], weights: Mapping[str, float] | None) -> dict[str, Any]:
    def attach(s: _Scored) -> bool:
        return s.item.outcome in _ATTACH_OUTCOMES

    def wrong_home(_: _Scored, decided: _rubric.Decided) -> bool:
        return Cell.WRONG_HOME in decided.triage_cells

    def split(flag: Callable[[ReaderItem], bool | None], labels: Sequence[str]) -> dict[str, Any]:
        return {
            label: _proportion(
                scored, lambda s, value=value: attach(s) and flag(s.item) is value,
                wrong_home, weights,
            )
            for label, value in zip(labels, (False, True, None), strict=True)
        }

    return {
        **_proportion(scored, attach, wrong_home, weights),
        'by_route': split(lambda item: item.via_sibling, ('direct', 'via_sibling', 'unknown')),
        'by_category': split(lambda item: item.cross_category, ('same', 'cross', 'unknown')),
    }


def _m5(scored: Sequence[_Scored], weights: Mapping[str, float] | None) -> dict[str, Any]:
    def contested(s: _Scored) -> bool:
        return s.item.outcome == OUTCOME_CONTESTED

    scopes = Counter(
        _SCOPE_LABELS.get(str(s.scope), 'unknown') for s in scored
        if contested(s) and s.decided is not None and s.decided.conflict
    )
    return {
        **_proportion(scored, contested, lambda _, d: d.conflict, weights),
        'by_scope': {label: scopes[label] for label in ('central', 'secondary', 'unknown')},
    }


def _m6(scored: Sequence[_Scored]) -> dict[str, Any]:
    rated = [
        (s, s.attach_time, s.decided) for s in scored
        if isinstance(s.attach_time, _rubric.Decided) and s.decided is not None
    ]
    drifted = sorted(
        s.item.entry_id for s, then, now in rated if then.triage_cells != now.triage_cells
    )
    return {
        'parent_edited': sum(bool(s.item.parent_edited_after_attach) for s in scored),
        'attach_time_rated': len(rated),
        'drifted': len(drifted),
        'drifted_entry_ids': drifted,
        'attach_time_undecided': sum(
            s.attach_time is not None and not isinstance(s.attach_time, _rubric.Decided)
            for s in scored
        ),
        'attach_time_unrated': sum(
            bool(s.item.parent_edited_after_attach) and s.attach_time is None for s in scored
        ),
    }


class _HarmFigures(NamedTuple):
    per_write: dict[str, float | None] | None
    delta_per_write: float | None
    delta_interval95: list[float] | None


def _harm(
    scored: Sequence[_Scored],
    weights: Mapping[_rubric.Cell, float],
    stratum_weights: Mapping[str, float] | None,
) -> dict[str, Any]:
    """Harm per write under each regime and triage's delta, per sample and per population."""
    harms = [
        (s.item.stratum, _write_harm(d.triage_cells, weights), _write_harm(d.status_quo_cells, weights))
        for s in scored if (d := s.decided) is not None
    ]
    sample = _harm_figures(harms, {stratum: 1.0 for stratum, _, _ in harms})
    population = _harm_figures(harms, stratum_weights)
    return {
        'n': len(harms),
        'unclear': len(scored) - len(harms),
        'per_write': sample.per_write,
        'delta_per_write': sample.delta_per_write,
        'delta_interval95': sample.delta_interval95,
        'population_per_write': population.per_write,
        'population_delta_per_write': population.delta_per_write,
        'population_delta_interval95': population.delta_interval95,
    }


def _write_harm(cells: frozenset[_rubric.Cell], weights: Mapping[_rubric.Cell, float]) -> float:
    return sum(weights.get(cell, 0) for cell in cells)


def _harm_figures(
    harms: Sequence[tuple[str, float, float]], weights: Mapping[str, float] | None,
) -> _HarmFigures:
    if weights is None:
        return _HarmFigures(None, None, None)
    triage = [_stats.StratumUnit(stratum, harm, 1.0) for stratum, harm, _ in harms]
    status_quo = [_stats.StratumUnit(stratum, harm, 1.0) for stratum, _, harm in harms]
    delta = [_stats.StratumUnit(stratum, mine - flat, 1.0) for stratum, mine, flat in harms]
    return _HarmFigures(
        per_write={
            'triage': _stats.stratified_ratio(triage, weights),
            'status_quo': _stats.stratified_ratio(status_quo, weights),
        },
        delta_per_write=_stats.stratified_ratio(delta, weights),
        delta_interval95=_bootstrap(delta, weights),
    )
