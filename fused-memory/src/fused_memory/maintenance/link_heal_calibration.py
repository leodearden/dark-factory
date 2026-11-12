"""Scoring the link adjudicator against blind-rated pairs, and the gate on that score.

The population is the write-triage soak's rated pairs: a ratings file keyed by
``child_intent_id`` joined to an items file holding the texts the raters read.
The adjudicator is scored against each pair's FINAL rater verdict on two
binaries, misfile and contradiction, whose verdict sets come from
``link_heal`` and are never respelled here.

The floor for each binary is the raters' own agreement on it, recomputed from
the same ratings file (the second rater against the primary, over the rows both
rated), less a margin. The gate holds when every gated scope clears its floors
and few enough pairs went unanswered. The pool is always gated; a project is
gated once it has enough rated items to make its kappa meaningful.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

from fused_memory.maintenance.link_heal import MISFILE_VERDICTS, Verdict

POOLED = 'pooled'
DEFAULT_MAX_UNANSWERED_SHARE = 0.05


class CalibrationInputError(ValueError):
    """A soak ratings or items file this module refuses to score from."""


@dataclass(frozen=True)
class CalibrationItem:
    """One rated pair: the texts the raters read, and their verdicts on it.

    ``final`` is the verdict the pair is scored against; ``primary`` and
    ``second`` are the two raters', ``second`` None where only one rated it.
    """

    key: str
    project: str
    child_id: str
    parent_id: str
    child_text: str
    parent_text: str
    final: Verdict
    primary: Verdict
    second: Verdict | None


class Binary(StrEnum):
    """The two yes/no questions a verdict answers, each scored on its own."""

    MISFILE = 'misfile'
    CONTRADICTION = 'contradiction'

    def holds(self, verdict: Verdict) -> bool:
        return verdict in _BINARY_VERDICTS[self]


_BINARY_VERDICTS: Mapping[Binary, frozenset[Verdict]] = MappingProxyType({
    Binary.MISFILE: MISFILE_VERDICTS,
    Binary.CONTRADICTION: frozenset({Verdict.CORRECTS}),
})


@dataclass(frozen=True)
class BinaryAgreement:
    """The 2x2 confusion of a judge under test against a reference rater, and its kappa."""

    both_yes: int
    adjudicator_only: int
    rater_only: int
    both_no: int

    @property
    def n(self) -> int:
        return self.both_yes + self.adjudicator_only + self.rater_only + self.both_no

    @property
    def kappa(self) -> float | None:
        """Cohen's kappa; None when chance agreement is 1, so kappa is undefined."""
        n = self.n
        if n == 0:
            return None
        observed = (self.both_yes + self.both_no) / n
        adjudicator_yes = (self.both_yes + self.adjudicator_only) / n
        rater_yes = (self.both_yes + self.rater_only) / n
        chance = adjudicator_yes * rater_yes + (1 - adjudicator_yes) * (1 - rater_yes)
        return None if chance >= 1 else (observed - chance) / (1 - chance)

    def as_json(self) -> dict[str, Any]:
        return {
            'n': self.n,
            'kappa': self.kappa,
            'both_yes': self.both_yes,
            'adjudicator_only': self.adjudicator_only,
            'rater_only': self.rater_only,
            'both_no': self.both_no,
        }


def binary_agreement(pairs: Iterable[tuple[bool, bool]]) -> BinaryAgreement:
    """The agreement of (adjudicator, rater) answers to one yes/no question."""
    cells = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for adjudicator, rater in pairs:
        cells[(bool(adjudicator), bool(rater))] += 1
    return BinaryAgreement(
        both_yes=cells[(True, True)],
        adjudicator_only=cells[(True, False)],
        rater_only=cells[(False, True)],
        both_no=cells[(False, False)],
    )


def _score(binary: Binary, judged: Iterable[tuple[Verdict, Verdict]]) -> BinaryAgreement:
    return binary_agreement((binary.holds(judge), binary.holds(reference)) for judge, reference in judged)


def rater_agreement(items: Iterable[CalibrationItem]) -> Mapping[Binary, BinaryAgreement]:
    """The raters' own agreement per binary: the second rater, in the adjudicator's
    seat, against the primary, over the rows both rated."""
    overlap = [(item.second, item.primary) for item in items if item.second is not None]
    return MappingProxyType({binary: _score(binary, overlap) for binary in Binary})


@dataclass(frozen=True)
class BinaryScore:
    """The adjudicator's agreement on one binary within one scope, and that binary's floor."""

    binary: Binary
    agreement: BinaryAgreement
    floor: float | None

    @property
    def clears_floor(self) -> bool:
        kappa = self.agreement.kappa
        return kappa is not None and self.floor is not None and kappa >= self.floor

    def as_json(self) -> dict[str, Any]:
        return {**self.agreement.as_json(), 'floor': self.floor, 'clears_floor': self.clears_floor}


@dataclass(frozen=True)
class ScopeScore:
    """The pool, or one project: its size, whether it is gated, and its score per binary."""

    scope: str
    items: int
    answered: int
    gated: bool
    binaries: Mapping[Binary, BinaryScore]

    def as_json(self) -> dict[str, Any]:
        return {
            'scope': self.scope,
            'items': self.items,
            'answered': self.answered,
            'gated': self.gated,
            'binaries': {binary.value: score.as_json() for binary, score in self.binaries.items()},
        }


@dataclass(frozen=True)
class KappaBelowFloor:
    """A gated scope whose kappa on *binary* is under its floor, or undefined."""

    binary: Binary
    scope: str
    kappa: float | None
    floor: float | None

    def describe(self) -> str:
        return f'{self.binary} kappa {_number(self.kappa)} < floor {_number(self.floor)} ({self.scope})'

    def as_json(self) -> dict[str, Any]:
        return {
            'gate': 'kappa_floor',
            'binary': self.binary.value,
            'scope': self.scope,
            'kappa': self.kappa,
            'floor': self.floor,
        }


@dataclass(frozen=True)
class TooManyUnanswered:
    """More of the population went unanswered than the bound allows."""

    share: float
    bound: float

    def describe(self) -> str:
        return f'unanswered share {self.share:.3f} > bound {self.bound:.3f}'

    def as_json(self) -> dict[str, Any]:
        return {'gate': 'unanswered_share', 'share': self.share, 'bound': self.bound}


GateFailure: TypeAlias = KappaBelowFloor | TooManyUnanswered


def _number(value: float | None) -> str:
    return 'undefined' if value is None else f'{value:.3f}'


@dataclass(frozen=True)
class CalibrationResult:
    """The adjudicator's score on a rated population, and whether it clears the gate."""

    pooled: ScopeScore
    projects: Mapping[str, ScopeScore]
    rater: Mapping[Binary, BinaryAgreement]
    margin: float
    min_project_items: int
    unanswered_share: float
    max_unanswered_share: float
    failures: tuple[GateFailure, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_json(self) -> dict[str, Any]:
        """The result as JSON-ready data, projects in name order."""
        return {
            'passed': self.passed,
            'failures': [failure.as_json() for failure in self.failures],
            'margin': self.margin,
            'min_project_items': self.min_project_items,
            'unanswered_share': self.unanswered_share,
            'max_unanswered_share': self.max_unanswered_share,
            'rater': {binary.value: agreement.as_json() for binary, agreement in self.rater.items()},
            'pooled': self.pooled.as_json(),
            'projects': [self.projects[project].as_json() for project in sorted(self.projects)],
        }


def assess_calibration(
    items: Sequence[CalibrationItem],
    verdicts: Mapping[str, Verdict],
    *,
    margin: float,
    min_project_items: int,
    max_unanswered_share: float = DEFAULT_MAX_UNANSWERED_SHARE,
) -> CalibrationResult:
    """Score *verdicts*, the adjudicator's answer per item key, against *items*.

    An item with no entry in *verdicts* is unanswered: it counts against the
    unanswered bound and is left out of every kappa.
    """
    rated = {item.key for item in items}
    unrated = sorted(set(verdicts) - rated)
    if unrated:
        raise ValueError(f'verdicts for keys no item carries: {", ".join(unrated)}')
    rater = rater_agreement(items)
    floors = {binary: _floor(rater[binary], margin) for binary in Binary}
    pooled = _scope_score(POOLED, items, verdicts, floors, gated=True)
    projects = {
        project: _scope_score(
            project, members, verdicts, floors, gated=len(members) >= min_project_items,
        )
        for project, members in _by_project(items).items()
    }
    unanswered_share = (len(items) - pooled.answered) / len(items) if items else 0.0
    failures = _failures(pooled, projects, unanswered_share, max_unanswered_share)
    return CalibrationResult(
        pooled=pooled,
        projects=MappingProxyType(projects),
        rater=rater,
        margin=margin,
        min_project_items=min_project_items,
        unanswered_share=unanswered_share,
        max_unanswered_share=max_unanswered_share,
        failures=failures,
    )


def _floor(rater: BinaryAgreement, margin: float) -> float | None:
    kappa = rater.kappa
    return None if kappa is None else kappa - margin


def _by_project(items: Iterable[CalibrationItem]) -> dict[str, list[CalibrationItem]]:
    projects: dict[str, list[CalibrationItem]] = {}
    for item in sorted(items, key=lambda item: (item.project, item.key)):
        projects.setdefault(item.project, []).append(item)
    return projects


def _scope_score(
    scope: str,
    items: Sequence[CalibrationItem],
    verdicts: Mapping[str, Verdict],
    floors: Mapping[Binary, float | None],
    *,
    gated: bool,
) -> ScopeScore:
    judged = [(verdicts[item.key], item.final) for item in items if item.key in verdicts]
    return ScopeScore(
        scope=scope,
        items=len(items),
        answered=len(judged),
        gated=gated,
        binaries=MappingProxyType({
            binary: BinaryScore(binary, _score(binary, judged), floors[binary])
            for binary in Binary
        }),
    )


def _failures(
    pooled: ScopeScore,
    projects: Mapping[str, ScopeScore],
    unanswered_share: float,
    max_unanswered_share: float,
) -> tuple[GateFailure, ...]:
    failures: list[GateFailure] = [
        KappaBelowFloor(score.binary, scope.scope, score.agreement.kappa, score.floor)
        for scope in (pooled, *(projects[project] for project in sorted(projects)))
        if scope.gated
        for score in scope.binaries.values()
        if not score.clears_floor
    ]
    if unanswered_share > max_unanswered_share:
        failures.append(TooManyUnanswered(unanswered_share, max_unanswered_share))
    return tuple(failures)


_RATING_TEXT_FIELDS = ('child_intent_id', 'child_memory_id', 'parent_id', 'project')
_ITEM_TEXT_FIELDS = ('intent_id', 'child_text', 'parent_text')


def load_soak_population(ratings_path: Path, items_path: Path) -> tuple[CalibrationItem, ...]:
    """Every rated pair in *ratings_path*, in its order, joined to its texts in *items_path*."""
    texts = _items_by_intent(items_path)
    population: list[CalibrationItem] = []
    seen: set[str] = set()
    lines = ratings_path.read_text(encoding='utf-8').splitlines()
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        where = f'{ratings_path.name} line {line_no}'
        item = _calibration_item(_rating_row(line, where), texts, where, items_path)
        if item.key in seen:
            raise CalibrationInputError(f'{where}: {item.key} rated twice')
        seen.add(item.key)
        population.append(item)
    return tuple(population)


def _items_by_intent(items_path: Path) -> dict[str, Mapping[str, Any]]:
    try:
        items = json.loads(items_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise CalibrationInputError(f'{items_path.name}: not JSON ({exc})') from exc
    if not isinstance(items, list):
        raise CalibrationInputError(f'{items_path.name}: not a JSON list')
    by_intent: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(items):
        where = f'{items_path.name} item {index}'
        if not isinstance(item, dict):
            raise CalibrationInputError(f'{where}: not a JSON object')
        for field_name in _ITEM_TEXT_FIELDS:
            _text_field(item, field_name, where)
        by_intent[item['intent_id']] = item
    return by_intent


def _rating_row(line: str, where: str) -> Mapping[str, Any]:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise CalibrationInputError(f'{where}: not JSON ({exc})') from exc
    if not isinstance(row, dict):
        raise CalibrationInputError(f'{where}: not a JSON object')
    for field_name in _RATING_TEXT_FIELDS:
        _text_field(row, field_name, where)
    return row


def _calibration_item(
    row: Mapping[str, Any],
    texts: Mapping[str, Mapping[str, Any]],
    where: str,
    items_path: Path,
) -> CalibrationItem:
    key = row['child_intent_id']
    item = texts.get(key)
    if item is None:
        raise CalibrationInputError(
            f'{where}: child_intent_id {key!r} has no item in {items_path.name}',
        )
    second = row.get('second_verdict')
    return CalibrationItem(
        key=key,
        project=row['project'],
        child_id=row['child_memory_id'],
        parent_id=row['parent_id'],
        child_text=item['child_text'],
        parent_text=item['parent_text'],
        final=_verdict(row, 'rater_verdict', where),
        primary=_verdict(row, 'primary_verdict', where),
        second=None if second is None else _verdict(row, 'second_verdict', where),
    )


def _text_field(row: Mapping[str, Any], field_name: str, where: str) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or not value:
        raise CalibrationInputError(f'{where}: {field_name} is not a non-empty string ({value!r})')
    return value


def _verdict(row: Mapping[str, Any], field_name: str, where: str) -> Verdict:
    value = row.get(field_name)
    try:
        return Verdict(value)
    except ValueError as exc:
        raise CalibrationInputError(
            f'{where}: {field_name} {value!r} is not one of the brief\'s verdict words',
        ) from exc
