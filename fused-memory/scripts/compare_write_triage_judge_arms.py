#!/usr/bin/env python3
"""Compare the judge arms of a prompt-by-model grid, such as task 9045's 2x2.

Each arm is one ``eval_write_triage_judge.py`` report with its cases dump
beside it. The comparison reads each arm's figures from its own report,
refuses arms that did not judge one population or that do not fill the grid
of their wordings by their models, and separates the prompt effect (one
model, two prompts) from the model effect (one prompt, two models) as arm
differences against the baseline arm. Pure apart from reading the arms and
writing the comparison JSON with its markdown render beside it.

Usage
-----
  python scripts/compare_write_triage_judge_arms.py \\
      --arm calibration/write_triage_judge_accuracy_report.json \\
      --arm calibration/write_triage_judge_arms/psi__gpt-6.1-sol.json \\
      --arm calibration/write_triage_judge_arms/shipped__gpt-6.2-sol.json \\
      --arm calibration/write_triage_judge_arms/psi__gpt-6.2-sol.json \\
      --out calibration/write_triage_judge_arms/comparison.json
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fused_memory.server.write_triage import OUTCOME_CONTESTED
from fused_memory.utils.script_loader import load_script

_SCRIPTS = Path(__file__).resolve().parent

#: The fused-memory package root, which a report's relative ``cases_path`` is relative to.
_PACKAGE_ROOT = _SCRIPTS.parent


# Pyright cannot follow the by-path loader, so it alone resolves `_scoring` and
# `_wording` through the scripts/ extraPath; at runtime each is the by-path load.
if TYPE_CHECKING:
    import eval_write_triage_scoring as _scoring
    import write_triage_judge_wording as _wording
else:
    _scoring = load_script(
        _SCRIPTS / 'eval_write_triage_scoring.py', 'eval_write_triage_scoring',
    )
    _wording = load_script(
        _SCRIPTS / 'write_triage_judge_wording.py', 'write_triage_judge_wording',
    )


#: The provenance fields that say which population an arm judged.
POPULATION_FIELDS: tuple[str, ...] = (
    'fixture_path', 'slate_mode', 'project_id', 't_high', 't_low', 'candidate_k',
    'judge_candidate_count', 'field_chars', 'case_count', 'canonical_aliases_path',
)

#: What one case of a cases dump judged: arms compared must agree on every case.
_CASE_KEYS: tuple[str, ...] = ('memory_id', 'band', 'routed', 'candidates')

#: Each figure an effect is taken of, and its path into :func:`arm_figures`.
EFFECT_FIGURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('middle_band_accuracy', ('middle_band', 'accuracy')),
    ('restate_band_accuracy', ('restate_band', 'accuracy')),
    ('judged_accuracy', ('judged', 'accuracy')),
    ('false_contested', ('false_contested',)),
    ('pseudo_contradiction_contested', ('pseudo_contradiction_contested',)),
    ('duplicate_contested', ('duplicate_contested',)),
)

_SHA_PREFIX = 12


@dataclass(frozen=True)
class Arm:
    """One judged run: its report, its cases dump, and the grid cell it fills."""

    path: str
    report: Mapping[str, Any]
    rows: list[dict[str, Any]]
    wording: str
    model: str


def _wording_of(sha: str) -> str:
    """The selectable wording whose prompt hashes to *sha*, else the sha's prefix."""
    for wording in _wording.SELECTABLE_WORDINGS:
        if hashlib.sha256(_wording.system_prompt(wording).encode()).hexdigest() == sha:
            return wording
    return sha[:_SHA_PREFIX]


def _cases_file(report_path: str, provenance: Mapping[str, Any]) -> Path:
    cases_path = provenance.get('cases_path')
    if not isinstance(cases_path, str):
        raise ValueError(f'{report_path}: names no cases_path, so its cases cannot be compared')
    path = Path(cases_path)
    return path if path.is_absolute() else _PACKAGE_ROOT / path


def load_arms(report_paths: Sequence[str | Path]) -> list[Arm]:
    """Each report at *report_paths* with its cases dump, named by its wording and model."""
    arms = []
    for report_path in map(str, report_paths):
        report = json.loads(Path(report_path).read_text())
        if report.get('routing') is None:
            raise ValueError(
                f'{report_path}: has no routing block, so it predates route_band '
                'routing (task 9045) and cannot be compared',
            )
        provenance = report['provenance']
        lines = _cases_file(report_path, provenance).read_text().splitlines()
        arms.append(Arm(
            path=report_path,
            report=report,
            rows=[json.loads(line) for line in lines if line.strip()],
            wording=_wording_of(str(provenance['judge_system_prompt_sha256'])),
            model=str(provenance['judge_model']),
        ))
    return arms


def _cases_of(arm: Arm) -> list[dict[str, Any]]:
    return [{key: row[key] for key in _CASE_KEYS} for row in arm.rows]


def check_same_population(arms: Sequence[Arm]) -> None:
    """Refuse arms that did not all judge the whole of one and the same population."""
    for arm in arms:
        limit = arm.report['provenance'].get('limit')
        if limit is not None:
            raise ValueError(f'{arm.path}: limit {limit!r}; a --limit run judged part of the population')
    for field in POPULATION_FIELDS:
        values = [arm.report['provenance'].get(field) for arm in arms]
        if any(value != values[0] for value in values):
            named = ', '.join(f'{arm.path}: {value!r}' for arm, value in zip(arms, values, strict=True))
            raise ValueError(f'the arms judged different populations: {field} differs ({named})')
    first = arms[0]
    expected = _cases_of(first)
    for arm in arms[1:]:
        pairs = itertools.zip_longest(expected, _cases_of(arm))
        for index, (theirs, mine) in enumerate(pairs, 1):
            if theirs != mine:
                raise ValueError(
                    f'{arm.path} judged another population than {first.path}: case '
                    f'{index} is {mine} there and {theirs} here',
                )


def _ordered_wordings(wordings: set[str]) -> list[str]:
    """Selectable wordings in their declared order, then any unrecognised prompt."""
    known = _wording.SELECTABLE_WORDINGS
    return sorted(wordings, key=lambda w: (known.index(w) if w in known else len(known), w))


def check_full_grid(arms: Sequence[Arm]) -> tuple[list[str], list[str]]:
    """The grid's wordings and models, refusing a doubled or a missing cell."""
    seen: set[tuple[str, str]] = set()
    for arm in arms:
        cell = (arm.wording, arm.model)
        if cell in seen:
            raise ValueError(f'two arms fill the cell {arm.wording} x {arm.model}; {arm.path} is one')
        seen.add(cell)
    wordings = _ordered_wordings({wording for wording, _ in seen})
    models = sorted({model for _, model in seen})
    missing = [f'{w} x {m}' for w in wordings for m in models if (w, m) not in seen]
    if missing:
        raise ValueError(
            f'the arms do not fill the grid of {wordings} by {models}: missing '
            f'{", ".join(missing)}',
        )
    return wordings, models


def arm_figures(arm: Arm) -> dict[str, Any]:
    """The figures one arm is compared on, each read from its own report."""
    report = arm.report
    confusion = report['confusion']
    return {
        'middle_band': report['production_shape']['middle_band'],
        'restate_band': report['routing']['restate_band'],
        'judged': report['routing']['judged'],
        'false_contested': report['false_contested'],
        'pseudo_contradiction_contested': (
            confusion[_scoring.LABEL_PSEUDO_CONTRADICTION][OUTCOME_CONTESTED]
        ),
        'duplicate_contested': confusion[_scoring.LABEL_DUPLICATE][OUTCOME_CONTESTED],
    }


def _read(figures: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    return functools.reduce(lambda node, key: node[key], path, figures)


def _difference(base: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    """*other* minus *base* on every :data:`EFFECT_FIGURES` entry; ``None`` where either is unmeasured."""
    differences: dict[str, Any] = {}
    for name, path in EFFECT_FIGURES:
        theirs, ours = _read(base, path), _read(other, path)
        differences[name] = None if theirs is None or ours is None else round(ours - theirs, 4)
    return differences


def effects(
    figures: Mapping[tuple[str, str], Mapping[str, Any]],
    wordings: Sequence[str],
    models: Sequence[str],
) -> dict[str, Any]:
    """The prompt effect at each model and the model effect at each prompt.

    Each is an arm's figures minus the baseline arm's in its row or column: the
    first wording at the same model, or the first model under the same wording.
    """
    base_wording, base_model = wordings[0], models[0]
    return {
        'prompt': {
            model: {
                wording: _difference(figures[(base_wording, model)], figures[(wording, model)])
                for wording in wordings[1:]
            }
            for model in models
        },
        'model': {
            wording: {
                model: _difference(figures[(wording, base_model)], figures[(wording, model)])
                for model in models[1:]
            }
            for wording in wordings
        },
    }


def build_comparison(arms: Sequence[Arm]) -> dict[str, Any]:
    """The JSON-serializable comparison of *arms*, refused unless they form one grid over one population."""
    if not arms:
        raise ValueError('no arms to compare')
    check_same_population(arms)
    wordings, models = check_full_grid(arms)
    by_cell = {(arm.wording, arm.model): arm for arm in arms}
    figures = {cell: arm_figures(arm) for cell, arm in by_cell.items()}
    provenance = arms[0].report['provenance']
    return {
        'wordings': wordings,
        'models': models,
        'baseline': {'wording': wordings[0], 'model': models[0]},
        'population': {field: provenance.get(field) for field in POPULATION_FIELDS},
        'arms': [
            {
                'wording': wording,
                'model': model,
                'report_path': by_cell[(wording, model)].path,
                'judge_system_prompt_sha256': (
                    by_cell[(wording, model)].report['provenance']['judge_system_prompt_sha256']
                ),
                'figures': figures[(wording, model)],
            }
            for wording in wordings for model in models
        ],
        'effects': effects(figures, wordings, models),
    }


def _rate_cell(block: Mapping[str, Any], denominator: str = 'n') -> str:
    return f'{block["correct"]}/{block[denominator]} = `{block["accuracy"]}`'


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    return [
        '| ' + ' | '.join(header) + ' |',
        '|---' * len(header) + '|',
        *('| ' + ' | '.join(str(cell) for cell in row) + ' |' for row in rows),
    ]


def render_markdown(comparison: Mapping[str, Any]) -> str:
    """The human-readable sibling of *comparison*."""
    base = comparison['baseline']
    names = [name for name, _ in EFFECT_FIGURES]
    arms = [
        (
            arm['wording'], arm['model'],
            _rate_cell(arm['figures']['middle_band']),
            _rate_cell(arm['figures']['restate_band'], 'judged'),
            _rate_cell(arm['figures']['judged']),
            arm['figures']['false_contested'],
            arm['figures']['pseudo_contradiction_contested'],
            arm['figures']['duplicate_contested'],
        )
        for arm in comparison['arms']
    ]
    prompt = [
        (model, wording, *(difference[name] for name in names))
        for model, by_wording in comparison['effects']['prompt'].items()
        for wording, difference in by_wording.items()
    ]
    model_effect = [
        (wording, model, *(difference[name] for name in names))
        for wording, by_model in comparison['effects']['model'].items()
        for model, difference in by_model.items()
    ]
    lines = [
        '# Write-triage judge arms: prompt by model',
        '',
        f'Baseline arm: `{base["wording"]}` x `{base["model"]}`. Every arm judged '
        'the same population:',
        '',
        *(f'- `{field}`: `{value}`' for field, value in comparison['population'].items()),
        '',
        '## Arms',
        '',
        *_table(
            ['wording', 'model', 'cosine middle band', 'judged restate band',
             'every judged case', 'false contested', 'pseudo_contradiction contested',
             'duplicate contested'],
            arms,
        ),
        '',
        f'## Prompt effect at each model (arm minus the `{base["wording"]}` arm)',
        '',
        *_table(['model', 'wording', *names], prompt),
        '',
        f'## Model effect at each prompt (arm minus the `{base["model"]}` arm)',
        '',
        *_table(['wording', 'model', *names], model_effect),
    ]
    return '\n'.join(lines) + '\n'


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--arm', action='append', required=True,
                        help='one arm report written by eval_write_triage_judge.py; '
                             'repeat once per grid cell')
    parser.add_argument('--out', required=True,
                        help='the comparison JSON; its markdown render is written '
                             'beside it as <stem>.md')
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.suffix != '.json':
        raise ValueError(
            f'--out {args.out!r} must end in .json: its markdown render is written '
            'beside it as <stem>.md',
        )
    comparison = build_comparison(load_arms(args.arm))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comparison, indent=2) + '\n')
    out.with_suffix('.md').write_text(render_markdown(comparison))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
