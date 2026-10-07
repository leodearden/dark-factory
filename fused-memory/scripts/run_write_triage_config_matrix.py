#!/usr/bin/env python3
"""Score every π judge arm with ι and name the configuration Γ3 is asked about.

Task 5807 (μ), plans/write-triage-flip-readiness-prd.md §11 C2''. π
(``run_write_triage_population_arms.py``) ran each judge arm over one frozen
population and published ``calibration/write_triage_population.json``. λ rated
every pair any arm's judge named (``calibration/write_triage_pair_verdicts.jsonl``).
This module scores every arm with ι (``score_write_triage_pairs.py::score_pairs``,
the one definition of the flip metrics), measures it against Γ3's bounds with
Γ3's own predicate, and picks the winner.

Inputs
------
- The π population artifact, whose ``arms`` rows name each arm's settings and
  its gitignored case file (``cases_path``, ``cases_sha256``).
- The λ verdict corpus.
- The checkout whose ``data/`` holds π's case files (``--data-root``).
- Γ3's bounds, each ``--require PATH OP VALUE`` as task 5808's before_done
  args spell them, without the ``best`` report handle.
- The shipped judge configuration, resolved from the live fused-memory config.

Outputs
-------
- ``calibration/write_triage_config_matrix.json``: one row per π arm with ι's
  blocks and the bound checks, the arms no route reached, the wording
  attribution and the winner.
- ``calibration/write_triage_best_config.json``: the winner's doc, the one Γ3
  reads.

Usage
-----
Run from ``fused-memory/``::

    uv run python scripts/run_write_triage_config_matrix.py \\
        --data-root /home/leo/src/dark-factory \\
        --require quality.misfile_rate_of_attaches '<=' 0.08 \\
        --require quality.unrated_pairs '<=' 0
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import types
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Any

from fused_memory.server.write_triage_judge import (
    resolve_judge_candidate_count,
    resolve_judge_field_chars,
    resolve_judge_model,
    resolve_judge_provider,
    resolve_judge_reasoning_effort,
)

_SCRIPTS = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]


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


_scorer = _load_script(_SCRIPTS / 'score_write_triage_pairs.py', 'score_write_triage_pairs')
_wording = _load_script(_SCRIPTS / 'write_triage_judge_wording.py', 'write_triage_judge_wording')
_gate = _load_script(
    _REPO_ROOT / 'scripts' / 'check_write_triage_readiness_gate.py',
    'check_write_triage_readiness_gate',
)

POPULATION_KEYS: tuple[str, ...] = (
    'n_writes', 'n_judge_band', 'projects', 'frozen_at', 'snapshot_sha256',
)
"""The C2'' subset of π's population block that μ's artifacts carry."""


class MatrixInputError(ValueError):
    """An input does not match what π published, so nothing is scored."""


@dataclass(frozen=True)
class ArmSettings:
    """What a judge arm is configured with, independent of how it scored."""

    provider: str
    model: str
    reasoning_effort: str | None
    candidate_count: int
    wording: str
    field_chars: int

    @classmethod
    def from_provenance(cls, row: Mapping[str, Any]) -> ArmSettings:
        """The settings of one π ``arms`` row (``_arm_row``'s keys)."""
        return cls(
            provider=row['provider'],
            model=row['model'],
            reasoning_effort=row['reasoning_effort'],
            candidate_count=row['width'],
            wording=row['wording'],
            field_chars=row['field_chars'],
        )

    def selection_keys(self) -> dict[str, Any]:
        return {
            'judge_provider': self.provider,
            'judge_model': self.model,
            'judge_reasoning_effort': self.reasoning_effort,
            'judge_candidate_count': self.candidate_count,
            'wording': self.wording,
            'field_chars': self.field_chars,
        }


def shipped_settings(service: Any) -> ArmSettings:
    """The judge arm the live config ships, through the shipped resolvers."""
    return ArmSettings(
        provider=resolve_judge_provider(service),
        model=resolve_judge_model(service),
        reasoning_effort=resolve_judge_reasoning_effort(service),
        candidate_count=resolve_judge_candidate_count(service),
        wording=_wording.WORDING_SHIPPED,
        field_chars=resolve_judge_field_chars(service),
    )


@dataclass(frozen=True)
class PiArm:
    """One arm π ran, with the case file it published."""

    name: str
    settings: ArmSettings
    cases_path: str
    cases_sha256: str


@dataclass(frozen=True)
class Population:
    """π's published population: its population block and its arms, in π's order."""

    block: Mapping[str, Any]
    arms: tuple[PiArm, ...]

    @classmethod
    def from_artifact(cls, doc: Mapping[str, Any]) -> Population:
        arms = tuple(
            PiArm(
                name=row['arm'],
                settings=ArmSettings.from_provenance(row),
                cases_path=row['cases_path'],
                cases_sha256=row['cases_sha256'],
            )
            for row in doc['arms']
        )
        names = [arm.name for arm in arms]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f'the population artifact names arms more than once: {repeated}')
        return cls(block=MappingProxyType(dict(doc['population'])), arms=arms)

    @property
    def names(self) -> list[str]:
        return [arm.name for arm in self.arms]

    def c2_block(self) -> dict[str, Any]:
        return {key: self.block[key] for key in POPULATION_KEYS}


def read_arm_cases(population: Population, data_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Every π arm's case rows, refusing a file that is not the one π published."""
    return {arm.name: _read_arm(arm, population, Path(data_root)) for arm in population.arms}


def _read_arm(arm: PiArm, population: Population, data_root: Path) -> list[dict[str, Any]]:
    path = data_root / arm.cases_path
    try:
        body = path.read_bytes()
    except FileNotFoundError:
        raise MatrixInputError(
            f'arm {arm.name}: {path} does not exist; pass the checkout holding π\'s'
            ' data/ (e.g. the main checkout) as --data-root'
        ) from None
    actual = hashlib.sha256(body).hexdigest()
    if actual != arm.cases_sha256:
        raise MatrixInputError(
            f'arm {arm.name}: {path} has sha256 {actual}, but π published {arm.cases_sha256}'
        )
    rows = _read_jsonl(path)
    _refuse_foreign_rows(arm, path, rows, 'arm', arm.name)
    _refuse_foreign_rows(arm, path, rows, 'snapshot_sha256', population.block['snapshot_sha256'])
    return rows


def _refuse_foreign_rows(
    arm: PiArm, path: Path, rows: Sequence[Mapping[str, Any]], key: str, expected: str,
) -> None:
    foreign = sorted({str(row.get(key)) for row in rows} - {expected})
    if foreign:
        raise MatrixInputError(
            f'arm {arm.name}: {path} holds rows whose {key} is {foreign}, not {expected!r}'
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding='utf-8') as lines:
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MatrixInputError(f'{path}:{number}: not a JSON line ({exc.msg})') from exc
            if not isinstance(row, dict):
                raise MatrixInputError(
                    f'{path}:{number}: not a JSON object line (got {type(row).__name__})'
                )
            rows.append(row)
    return rows


@dataclass(frozen=True)
class Bound:
    """One Γ3 requirement on the best-config doc, spelled as the gate's ``--require``."""

    path: str
    op: str
    value: str


@dataclass(frozen=True)
class SkippedArm:
    """A judge arm the PRD names that no route reached, so π never ran it."""

    arm: str
    provider: str
    reason: str

    def row(self) -> dict[str, Any]:
        return {'arm': self.arm, 'status': 'skipped', 'provider': self.provider,
                'reason': self.reason}


SKIPPED_ARMS: tuple[SkippedArm, ...] = (
    SkippedArm(
        'claude-haiku', 'anthropic',
        "no route today: the deployment's ANTHROPIC_API_KEY is rejected (401), PRD §11.1;"
        ' dropped from μ by §11.6',
    ),
    SkippedArm(
        'local-model', 'local',
        "no route today: σ dropped from μ's deps by PRD §11.6; π ran no local endpoint",
    ),
)
"""The PRD's arms that are recorded as skipped rather than silently omitted."""

WINNER_RULE = (
    'meets every bound (scripts/check_write_triage_readiness_gate.py::check_require);'
    ' fewest quality.contested_decision_errors; ties to lower'
    ' selection.cost_per_write_usd, then arm name'
)


def build_matrix(
    population: Population,
    cases_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    verdicts: Sequence[Mapping[str, Any]],
    *,
    shipped: ArmSettings,
    bounds: Sequence[Bound],
) -> dict[str, Any]:
    """Score every π arm with ι, check it against *bounds*, and name the winner.

    Every arm is paired against the π arm whose settings are *shipped*. An arm
    meets a bound exactly when Γ3's ``check_require`` passes on the arm's
    :func:`candidate_doc`, so the winner's best_config passes Γ3 exactly when
    it met every bound here.
    """
    if not bounds:
        raise ValueError(
            "no bounds given: pass Γ3's bounds as --require PATH OP VALUE; an empty"
            ' list would let every arm qualify'
        )
    reference = _reference_arm(population, shipped)
    _refuse_unmatched_cases(population, cases_by_arm)
    _refuse_scored_and_skipped(population)
    scores = _scorer.score_pairs(
        chain.from_iterable(cases_by_arm.values()), verdicts, reference_arm=reference.name,
    )
    _refuse_judge_band_drift(population, scores['arms'])
    block = population.c2_block()
    scored = [
        _with_bounds(_scored_row(arm, scores['arms'][arm.name]), block, bounds)
        for arm in population.arms
    ]
    winner, fallback_applied = select_winner(scored)
    return {
        'reference_arm': reference.name,
        'reference_settings': shipped.selection_keys(),
        'population': block,
        'verdict_corpus': scores['verdict_corpus'],
        'list_prices': scores['list_prices'],
        'arms': scored + [skipped.row() for skipped in SKIPPED_ARMS],
        'selection_rule': {
            'bounds': [{'path': b.path, 'op': b.op, 'value': b.value} for b in bounds],
            'rule': WINNER_RULE,
            'fallback_applied': fallback_applied,
        },
        'winner': winner,
    }


def candidate_doc(row: Mapping[str, Any], population: Mapping[str, Any]) -> dict[str, Any]:
    """The doc Γ3 reads for one scored arm; the one builder of it."""
    return {'quality': row['quality'], 'selection': row['selection'], 'population': population}


def best_config(matrix: Mapping[str, Any]) -> dict[str, Any]:
    """The winner's :func:`candidate_doc`, the doc its bounds were checked on, plus provenance.

    Γ3 reads ``quality``, ``selection`` and ``population``; ``provenance``
    names the arm and the bounds it failed, which are the checks Γ3 will fail.
    """
    winners = [
        row for row in matrix['arms']
        if row['status'] == 'scored' and row['arm'] == matrix['winner']
    ]
    if len(winners) != 1:
        raise ValueError(f'the matrix winner {matrix["winner"]!r} names no single scored arm')
    [row] = winners
    checks = row['bounds']['checks']
    return candidate_doc(row, matrix['population']) | {
        'provenance': {
            'arm': row['arm'],
            'reference_arm': matrix['reference_arm'],
            'meets_every_bound': row['bounds']['met'],
            'failed_bounds': [check['check'] for check in checks if not check['ok']],
        },
    }


def select_winner(rows: Sequence[Mapping[str, Any]]) -> tuple[str, bool]:
    """The winning arm by :data:`WINNER_RULE`, and whether no arm met every bound.

    When no scored arm meets every bound, the same ranking runs over every
    scored arm and the fallback flag is True. Skipped rows are never chosen.
    """
    scored = [row for row in rows if row['status'] == 'scored']
    if not scored:
        raise ValueError('no scored arm to choose a winner from')
    qualified = [row for row in scored if row['bounds']['met']]
    pool = qualified or scored
    return min(pool, key=_rank)['arm'], not qualified


def _rank(row: Mapping[str, Any]) -> tuple[int, float, str]:
    cost = row['selection']['cost_per_write_usd']
    return (
        row['quality']['contested_decision_errors'],
        math.inf if cost is None else cost,
        row['arm'],
    )


def _reference_arm(population: Population, shipped: ArmSettings) -> PiArm:
    matches = [arm for arm in population.arms if arm.settings == shipped]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f'the shipped configuration {shipped!r} matches no π arm; the arms are {population.names}'
        )
    raise ValueError(
        f'the shipped configuration {shipped!r} matches several π arms:'
        f' {[arm.name for arm in matches]}'
    )


def _refuse_unmatched_cases(
    population: Population, cases_by_arm: Mapping[str, Iterable[Mapping[str, Any]]],
) -> None:
    if sorted(cases_by_arm) != sorted(population.names):
        raise MatrixInputError(
            f'case rows are given for arms {sorted(cases_by_arm)} but the population'
            f' publishes {sorted(population.names)}'
        )


def _refuse_judge_band_drift(population: Population, scored: Mapping[str, Any]) -> None:
    published = population.block['n_judge_band']
    for arm in population.arms:
        held = scored[arm.name]['population']['n_judge_band']
        if held != published:
            raise MatrixInputError(
                f'arm {arm.name} holds {held} judge-band writes, but the population'
                f' publishes n_judge_band {published}'
            )


def _refuse_scored_and_skipped(population: Population) -> None:
    both = sorted(set(population.names) & {skipped.arm for skipped in SKIPPED_ARMS})
    if both:
        raise ValueError(f'arms {both} are both run by π and recorded as skipped')


def _with_bounds(
    row: dict[str, Any], population: Mapping[str, Any], bounds: Sequence[Bound],
) -> dict[str, Any]:
    doc = {'best': candidate_doc(row, population)}
    checks = [_gate.check_require(doc, 'best', b.path, b.op, b.value) for b in bounds]
    return row | {'bounds': {'met': all(check['ok'] for check in checks), 'checks': checks}}


def _scored_row(arm: PiArm, iota: Mapping[str, Any]) -> dict[str, Any]:
    runtime = iota['runtime']
    return {
        'arm': arm.name,
        'status': 'scored',
        'selection': arm.settings.selection_keys() | {
            'p95_judge_seconds': runtime['p95_judge_seconds'],
            'cost_per_write_usd': runtime['cost_per_write_usd'],
        },
        'quality': iota['quality'],
        'runtime': runtime,
        'paired_vs_reference': iota['paired_vs_reference'],
    }

