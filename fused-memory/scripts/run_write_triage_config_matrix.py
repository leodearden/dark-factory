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
Run from ``fused-memory/``, with one ``--require`` per ``--require best …``
of task 5808's metadata.before_done.args (two are shown)::

    uv run python scripts/run_write_triage_config_matrix.py \\
        --data-root /home/leo/src/dark-factory \\
        --require quality.misfile_rate_of_attaches '<=' 0.08 \\
        --require quality.unrated_pairs '<=' 0

An unrated or tied pair, an arm file that differs from what π published, or
a population artifact that lacks a key μ reads exits 1 with the reason on
stderr and writes neither artifact. A failure writing them exits 1 too.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import types
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Any

from shared.cli_boundary import LoudArgumentParser, run_cli

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.server.write_triage_judge import (
    resolve_judge_candidate_count,
    resolve_judge_field_chars,
    resolve_judge_model,
    resolve_judge_provider,
    resolve_judge_reasoning_effort,
)

_SCRIPTS = Path(__file__).resolve().parent
_PACKAGE_ROOT = _SCRIPTS.parent
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
        """π's artifact, refused as a :class:`MatrixInputError` if it lacks a key μ reads."""
        try:
            arms = tuple(_pi_arm(index, row) for index, row in enumerate(doc['arms']))
            block = MappingProxyType(dict(doc['population']))
        except KeyError as missing:
            raise MatrixInputError(f'the population artifact lacks key {missing}') from None
        except TypeError as malformed:
            raise MatrixInputError(
                f'the population artifact is not shaped as π publishes it: {malformed}'
            ) from None
        absent = [key for key in POPULATION_KEYS if key not in block]
        if absent:
            raise MatrixInputError(f'the population artifact\'s population block lacks {absent}')
        names = [arm.name for arm in arms]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f'the population artifact names arms more than once: {repeated}')
        return cls(block=block, arms=arms)

    @property
    def names(self) -> list[str]:
        return [arm.name for arm in self.arms]

    def c2_block(self) -> dict[str, Any]:
        return {key: self.block[key] for key in POPULATION_KEYS}


def _pi_arm(index: int, row: Mapping[str, Any]) -> PiArm:
    try:
        return PiArm(
            name=row['arm'],
            settings=ArmSettings.from_provenance(row),
            cases_path=row['cases_path'],
            cases_sha256=row['cases_sha256'],
        )
    except KeyError as missing:
        raise MatrixInputError(
            f'the population artifact\'s arms[{index}] lacks key {missing}'
        ) from None


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
    'fewest failed bounds (scripts/check_write_triage_readiness_gate.py::check_require),'
    ' so an arm meeting every bound beats every arm that does not; then fewest'
    ' quality.contested_decision_errors; then lower selection.cost_per_write_usd;'
    ' then arm name'
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
        'wording_attribution': _wording_attribution(population, cases_by_arm, verdicts),
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
    return candidate_doc(row, matrix['population']) | {
        'provenance': {
            'arm': row['arm'],
            'reference_arm': matrix['reference_arm'],
            'meets_every_bound': row['bounds']['met'],
            'failed_bounds': _failed_bounds(row),
        },
    }


def select_winner(rows: Sequence[Mapping[str, Any]]) -> tuple[str, bool]:
    """The winning arm by :data:`WINNER_RULE`, and whether it is a fallback.

    The fallback flag is True when the winner fails a bound, which the
    ranking allows only when no scored arm meets every bound. Skipped rows are
    never chosen.
    """
    scored = [row for row in rows if row['status'] == 'scored']
    if not scored:
        raise ValueError('no scored arm to choose a winner from')
    winner = min(scored, key=_rank)
    return winner['arm'], not winner['bounds']['met']


def _rank(row: Mapping[str, Any]) -> tuple[int, int, float, str]:
    cost = row['selection']['cost_per_write_usd']
    return (
        len(_failed_bounds(row)),
        row['quality']['contested_decision_errors'],
        math.inf if cost is None else cost,
        row['arm'],
    )


def _failed_bounds(row: Mapping[str, Any]) -> list[str]:
    return [check['check'] for check in row['bounds']['checks'] if not check['ok']]


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


def _wording_attribution(
    population: Population,
    cases_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    verdicts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Each non-shipped-wording arm paired by ι against its shipped twin at matched settings."""
    by_settings = {arm.settings: arm for arm in population.arms}
    attribution: dict[str, Any] = {}
    for arm in population.arms:
        if arm.settings.wording == _wording.WORDING_SHIPPED:
            continue
        twin = by_settings.get(replace(arm.settings, wording=_wording.WORDING_SHIPPED))
        attribution[arm.name] = (
            {'shipped_twin': None, 'paired': None} if twin is None
            else {'shipped_twin': twin.name,
                  'paired': _paired_against(arm, twin, cases_by_arm, verdicts)}
        )
    return attribution


def _paired_against(
    arm: PiArm,
    twin: PiArm,
    cases_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    verdicts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scores = _scorer.score_pairs(
        chain(cases_by_arm[arm.name], cases_by_arm[twin.name]), verdicts,
        reference_arm=twin.name,
    )
    return scores['arms'][arm.name]['paired_vs_reference']


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


# --- CLI -----------------------------------------------------------------------

MATRIX_NAME = 'write_triage_config_matrix.json'
BEST_CONFIG_NAME = 'write_triage_best_config.json'
_CALIBRATION = _PACKAGE_ROOT / 'calibration'


def _build_parser() -> LoudArgumentParser:
    parser = LoudArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--population', type=Path, default=_CALIBRATION / 'write_triage_population.json',
        help="π's published population artifact",
    )
    parser.add_argument(
        '--verdicts', type=Path, default=_CALIBRATION / 'write_triage_pair_verdicts.jsonl',
        help="λ's verdict corpus",
    )
    parser.add_argument(
        '--data-root', type=Path, required=True,
        help="the checkout whose gitignored data/ holds π's arm files, e.g. the main checkout",
    )
    parser.add_argument(
        '--require', action='append', nargs=3, required=True,
        metavar=('PATH', 'OP', 'VALUE'),
        help="one Γ3 bound: copy each `--require best …` of task 5808's before_done.args"
             " without the 'best' handle; repeat for every bound",
    )
    parser.add_argument(
        '--out-dir', type=Path, default=_CALIBRATION,
        help='where both artifacts are written (default: calibration/)',
    )
    return parser


def _compute(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    population = Population.from_artifact(json.loads(args.population.read_text(encoding='utf-8')))
    matrix = build_matrix(
        population,
        read_arm_cases(population, args.data_root),
        _read_jsonl(args.verdicts),
        shipped=shipped_settings(types.SimpleNamespace(config=FusedMemoryConfig())),
        bounds=[Bound(*spec) for spec in args.require],
    ) | {'inputs': _inputs(args.population, args.verdicts)}
    return matrix, best_config(matrix)


def _inputs(population: Path, verdicts: Path) -> dict[str, str]:
    return {
        'population_path': _repo_relative(population),
        'population_sha256': hashlib.sha256(population.read_bytes()).hexdigest(),
        'verdicts_path': _repo_relative(verdicts),
        'verdicts_sha256': hashlib.sha256(verdicts.read_bytes()).hexdigest(),
    }


def _repo_relative(path: Path) -> str:
    """*path* relative to the checkout that holds it (the first parent with ``.git``)."""
    resolved = Path(path).resolve()
    for parent in resolved.parents:
        if (parent / '.git').exists():
            return resolved.relative_to(parent).as_posix()
    raise ValueError(f'{path} is not inside a git checkout')


def _json_body(doc: Mapping[str, Any]) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + '\n'


def _write_staged(bodies: Mapping[Path, str]) -> None:
    """Write every body beside its path before moving any into place.

    A body that cannot be written therefore replaces none of the files. The
    moves run one after another, so a move that fails after the first leaves
    the earlier files replaced and the later ones not: the artifacts then
    disagree until the next successful run.
    """
    staged = {path: path.with_name(f'{path.name}.tmp') for path in bodies}
    try:
        for path, body in bodies.items():
            staged[path].write_text(body, encoding='utf-8')
        for path, temporary in staged.items():
            os.replace(temporary, path)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        matrix, best = _compute(args)
    except (OSError, ValueError) as refusal:
        print(refusal, file=sys.stderr)
        return 1
    matrix_out = args.out_dir / MATRIX_NAME
    best_out = args.out_dir / BEST_CONFIG_NAME
    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _write_staged({matrix_out: _json_body(matrix), best_out: _json_body(best)})
    except OSError as failure:
        print(f'cannot write the artifacts under {args.out_dir}: {failure}', file=sys.stderr)
        return 1
    print(json.dumps({
        'winner': matrix['winner'],
        'reference_arm': matrix['reference_arm'],
        'meets_every_bound': best['provenance']['meets_every_bound'],
        'fallback_applied': matrix['selection_rule']['fallback_applied'],
        'matrix_out': str(matrix_out),
        'best_config_out': str(best_out),
    }, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(run_cli(main))
