"""Tests for compare_write_triage_judge_arms.py — the 2x2 of judge prompt by judge model.

Every arm is a synthetic report in ``tmp_path``, assembled by the eval's own
scoring (``build_report`` over ``score_attachments`` and ``score_routing`` of
its rows) with its cases dump beside it, so the comparison is driven through
what an eval run writes. No network, key or store.
"""
from __future__ import annotations

import functools
import json
import os
import types
from pathlib import Path

import pytest
from _eval_write_triage_fixtures import SCRIPTS
from _eval_write_triage_fixtures import scoring_mod as _scoring
from _eval_write_triage_fixtures import wording_mod as _wording
from _fm_helpers import load_script_module

from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
)

PACKAGE_ROOT = SCRIPTS.parent

MODELS = ('gpt-6.1-sol', 'gpt-6.2-sol')
WORDINGS = ('shipped', 'psi')

#: The measured population every arm shares: (memory_id, class, band, route).
POPULATION = (
    ('d-restate-judged', 'duplicate', OUTCOME_RESTATED, OUTCOME_JUDGE),
    ('d-restate-verbatim', 'duplicate', OUTCOME_RESTATED, OUTCOME_RESTATED),
    ('d-middle', 'duplicate', OUTCOME_JUDGE, OUTCOME_JUDGE),
    ('p-middle', 'pseudo_contradiction', OUTCOME_JUDGE, OUTCOME_JUDGE),
    ('x-floor', 'distinct', OUTCOME_STORED, OUTCOME_STORED),
)

#: What each arm's judge answered on the three cases routed to it.
VERDICTS = {
    ('shipped', 'gpt-6.1-sol'): (OUTCOME_AMENDED, OUTCOME_AMENDED, OUTCOME_CONTESTED),
    ('psi', 'gpt-6.1-sol'): (OUTCOME_CONTESTED, OUTCOME_AMENDED, OUTCOME_CONTESTED),
    ('shipped', 'gpt-6.2-sol'): (OUTCOME_AMENDED, OUTCOME_AMENDED, OUTCOME_AMENDED),
    ('psi', 'gpt-6.2-sol'): (OUTCOME_CONTESTED, OUTCOME_CONTESTED, OUTCOME_AMENDED),
}

#: Each arm's figures, counted by hand from POPULATION and VERDICTS: correct
#: answers in the cosine middle band (of 2), the judged restate band (of 1) and
#: every judged case (of 3), then contested verdicts on duplicates and on
#: pseudo-contradictions.
EXPECTED_FIGURES = {
    ('shipped', 'gpt-6.1-sol'): (1, 1, 2, 0, 1),
    ('psi', 'gpt-6.1-sol'): (1, 0, 1, 1, 1),
    ('shipped', 'gpt-6.2-sol'): (2, 1, 3, 0, 0),
    ('psi', 'gpt-6.2-sol'): (1, 0, 1, 2, 0),
}

#: Every provenance field that says WHICH population an arm judged.
POPULATION_FIELDS = (
    'fixture_path', 'slate_mode', 'project_id', 't_high', 't_low', 'candidate_k',
    'judge_candidate_count', 'field_chars', 'case_count', 'canonical_aliases_path',
)


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'compare_write_triage_judge_arms.py', 'compare_write_triage_judge_arms',
    )


def _sha(wording: str) -> str:
    return _wording().prompt_sha256(wording)


def _rows(verdicts) -> list[dict]:
    """The cases dump of one arm: the shared population, answered by *verdicts*."""
    answer = iter(verdicts)
    rows = []
    for index, (memory_id, expected_class, band, routed) in enumerate(POPULATION, 1):
        outcome = next(answer) if routed == OUTCOME_JUDGE else routed
        acceptable = _scoring().acceptable_outcomes_for(expected_class)
        rows.append({
            'index': index, 'memory_id': memory_id, 'cluster_id': 'c',
            'expected_class': expected_class, 'acceptable_outcomes': sorted(acceptable),
            'candidates': ['c', 'z'], 'band_winner_id': 'c', 'attach_target_id': 'c',
            'attach_target_cluster_id': 'c', 'attach_target_category': None,
            'attach_target_label': 'canonical', 'attach_target_is_canonical': True,
            'canonical_in_slate': True, 'canonical_present': True,
            'band': band, 'routed': routed, 'similarity': 0.9,
            'verdict': outcome if routed == OUTCOME_JUDGE else None,
            'outcome': outcome, 'correct': outcome in acceptable,
            'entry_elided': False, 'candidates_elided': {}, 'usage': None,
        })
    return rows


def _write_arm(
    directory: Path, wording: str, model: str, *,
    rows: list[dict] | None = None, cases_path: str | None = None,
    sha: str | None = None, routing: bool = True, **provenance,
) -> Path:
    """One arm's report and cases dump, as `run_judge_eval` writes them."""
    rows = _rows(VERDICTS[(wording, model)]) if rows is None else rows
    report_path = directory / f'{wording}__{model}.json'
    dump = report_path.parent / (report_path.stem + '.cases.jsonl')
    dump.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    cases = [
        {'memory_id': r['memory_id'], 'expected_class': r['expected_class'],
         'acceptable_outcomes': frozenset(r['acceptable_outcomes'])}
        for r in rows
    ]
    report = _scoring().build_report(
        scored=_scoring().score_cases(cases, [r['outcome'] for r in rows]),
        provenance={
            'fixture_path': 'tests/fixtures/write_triage_calibration.jsonl',
            'slate_mode': _scoring().SLATE_RETRIEVED, 'project_id': 'reify',
            't_high': 0.9, 't_low': 0.5, 'candidate_k': 10,
            'judge_candidate_count': 5, 'field_chars': 1200,
            'case_count': len(rows), 'limit': None,
            'canonical_aliases_path': 'tests/fixtures/aliases.json',
            'judge_provider': 'openai', 'judge_model': model,
            'judge_system_prompt_sha256': sha or _sha(wording),
            'cases_path': str(dump) if cases_path is None else cases_path,
            **provenance,
        },
        production_shape=_scoring().score_attachments(rows),
        routing=_scoring().score_routing(rows) if routing else None,
    )
    report_path.write_text(json.dumps(report))
    return report_path


def _grid(directory: Path) -> list[Path]:
    return [_write_arm(directory, wording, model) for model in MODELS for wording in WORDINGS]


def _report(path: Path) -> dict:
    return json.loads(path.read_text())


def _sections(markdown: str) -> dict[str, list[dict[str, str]]]:
    """Each ``## `` section's table, one ``{column: cell}`` per row."""
    tables: dict[str, list[list[str]]] = {}
    heading = None
    for line in markdown.splitlines():
        if line.startswith('## '):
            heading = line[3:]
            tables[heading] = []
        elif heading is not None and line.startswith('| '):
            tables[heading].append([cell.strip() for cell in line.strip('|').split('|')])
    return {
        heading: [dict(zip(header, row, strict=True)) for row in rows]
        for heading, (header, *rows) in tables.items()
    }


def _section(sections: dict[str, list[dict[str, str]]], prefix: str) -> tuple[str, list]:
    [(heading, rows)] = [(h, rows) for h, rows in sections.items() if h.startswith(prefix)]
    return heading, rows


def _rate_cell(block: dict, denominator: str = 'n') -> str:
    return f'{block["correct"]}/{block[denominator]} = `{block["accuracy"]}`'


class TestTheArmsAreReadFromTheirOwnReports:
    def test_each_arm_reads_its_own_figures(self, tmp_path: Path) -> None:
        for arm in _mod().load_arms(_grid(tmp_path)):
            figures = _mod().arm_figures(arm)
            middle, restate, judged, duplicate, pseudo = EXPECTED_FIGURES[(arm.wording, arm.model)]
            assert (figures['middle_band']['n'], figures['middle_band']['correct']) == (2, middle)
            assert (
                figures['restate_band']['judged'], figures['restate_band']['correct'],
            ) == (1, restate)
            assert (figures['judged']['n'], figures['judged']['correct']) == (3, judged)
            assert (
                figures['duplicate_contested'], figures['pseudo_contradiction_contested'],
                figures['false_contested'],
            ) == (duplicate, pseudo, duplicate + pseudo)

    def test_the_arms_do_not_all_read_the_same(self, tmp_path: Path) -> None:
        figures = [_mod().arm_figures(arm) for arm in _mod().load_arms(_grid(tmp_path))]
        assert len({json.dumps(f, sort_keys=True) for f in figures}) == len(figures)

    @pytest.mark.parametrize('wording', WORDINGS)
    def test_an_arm_is_named_by_the_wording_its_prompt_sha_names(
        self, tmp_path: Path, wording: str,
    ) -> None:
        [arm] = _mod().load_arms([_write_arm(tmp_path, wording, MODELS[0])])
        assert (arm.wording, arm.model) == (wording, MODELS[0])

    def test_an_unknown_prompt_is_named_by_its_sha_prefix(self, tmp_path: Path) -> None:
        sha = 'f' * 64
        [arm] = _mod().load_arms([_write_arm(tmp_path, 'shipped', MODELS[0], sha=sha)])
        assert arm.wording == sha[:12]

    def test_a_relative_cases_path_resolves_against_the_package_root(
        self, tmp_path: Path,
    ) -> None:
        dump = tmp_path / f'shipped__{MODELS[0]}.cases.jsonl'
        relative = os.path.relpath(dump, PACKAGE_ROOT)
        [arm] = _mod().load_arms([_write_arm(tmp_path, 'shipped', MODELS[0], cases_path=relative)])
        assert arm.rows == _rows(VERDICTS[('shipped', MODELS[0])])

    def test_an_arm_without_a_routing_block_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='routing'):
            _mod().load_arms([_write_arm(tmp_path, 'shipped', MODELS[0], routing=False)])


class TestTheEffects:
    def test_the_prompt_and_model_effects_are_the_arm_differences(self, tmp_path: Path) -> None:
        comparison = _mod().build_comparison(_mod().load_arms(_grid(tmp_path)))
        figures = {
            (arm['wording'], arm['model']): arm['figures'] for arm in comparison['arms']
        }

        def judged(wording: str, model: str) -> float:
            return figures[(wording, model)]['judged']['accuracy']

        for model in MODELS:
            assert comparison['effects']['prompt'][model]['psi']['judged_accuracy'] == round(
                judged('psi', model) - judged('shipped', model), 4,
            )
        for wording in WORDINGS:
            assert comparison['effects']['model'][wording][MODELS[1]]['duplicate_contested'] == (
                figures[(wording, MODELS[1])]['duplicate_contested']
                - figures[(wording, MODELS[0])]['duplicate_contested']
            )
        assert (comparison['baseline']['wording'], comparison['baseline']['model']) == (
            'shipped', MODELS[0],
        )

    def test_an_unmeasured_side_makes_no_effect(self) -> None:
        def figures(accuracy):
            block = {'n': 1, 'correct': 1, 'accuracy': accuracy}
            return {
                'middle_band': block, 'restate_band': block, 'judged': block,
                'false_contested': 0, 'pseudo_contradiction_contested': 0,
                'duplicate_contested': 0,
            }

        by_cell = {
            ('shipped', 'm1'): figures(0.5), ('psi', 'm1'): figures(None),
            ('shipped', 'm2'): figures(None), ('psi', 'm2'): figures(None),
        }
        effects = _mod().effects(by_cell, ['shipped', 'psi'], ['m1', 'm2'])
        assert effects['prompt']['m1']['psi']['judged_accuracy'] is None
        assert effects['model']['shipped']['m2']['judged_accuracy'] is None
        assert effects['prompt']['m1']['psi']['false_contested'] == 0


class TestArmsThatDidNotJudgeOnePopulationAreRefused:
    @pytest.mark.parametrize('field', POPULATION_FIELDS)
    def test_a_population_field_that_differs_is_named(self, tmp_path: Path, field: str) -> None:
        arms = _grid(tmp_path)
        report = _report(arms[-1])
        value = report['provenance'][field]
        report['provenance'][field] = (value + 1) if isinstance(value, int | float) else f'{value}-x'
        arms[-1].write_text(json.dumps(report))
        out = tmp_path / 'out' / 'comparison.json'
        with pytest.raises(ValueError, match=field):
            _mod().main([*(a for path in arms for a in ('--arm', str(path))), '--out', str(out)])
        assert not out.parent.exists()

    def test_a_limited_arm_is_refused(self, tmp_path: Path) -> None:
        arms = _grid(tmp_path)
        arms[0] = _write_arm(tmp_path, WORDINGS[0], MODELS[0], limit=3)
        with pytest.raises(ValueError, match='limit'):
            _mod().build_comparison(_mod().load_arms(arms))

    @pytest.mark.parametrize(('key', 'value'), [
        ('memory_id', 'd-middle-other'),
        ('band', OUTCOME_STORED),
        ('routed', OUTCOME_STORED),
        ('candidates', ['z', 'c']),
    ])
    def test_a_cases_dump_that_differs_names_the_case(
        self, tmp_path: Path, key: str, value,
    ) -> None:
        arms = _grid(tmp_path)
        rows = _rows(VERDICTS[(WORDINGS[1], MODELS[1])])
        rows[2][key] = value
        arms[-1] = _write_arm(tmp_path, WORDINGS[1], MODELS[1], rows=rows)
        with pytest.raises(ValueError, match='d-middle') as refused:
            _mod().build_comparison(_mod().load_arms(arms))
        assert f'case 3: {arms[0]} has ' in str(refused.value)
        assert f', {arms[-1]} has ' in str(refused.value)

    def test_two_arms_of_one_cell_are_refused(self, tmp_path: Path) -> None:
        arms = _grid(tmp_path)
        twin = tmp_path / 'twin'
        twin.mkdir()
        arms.append(_write_arm(twin, WORDINGS[0], MODELS[0]))
        with pytest.raises(ValueError, match=MODELS[0]):
            _mod().build_comparison(_mod().load_arms(arms))

    def test_a_grid_with_a_missing_cell_is_refused_naming_it(self, tmp_path: Path) -> None:
        arms = _grid(tmp_path)[:-1]
        with pytest.raises(ValueError, match=f'{WORDINGS[1]}.*{MODELS[1]}'):
            _mod().build_comparison(_mod().load_arms(arms))


class TestRenderMarkdown:
    def test_each_arm_row_carries_that_arms_figures(self, tmp_path: Path) -> None:
        comparison = _mod().build_comparison(_mod().load_arms(_grid(tmp_path)))
        rows = _sections(_mod().render_markdown(comparison))['Arms']
        by_cell = {(row['wording'], row['model']): row for row in rows}
        assert len(by_cell) == len(rows) == len(EXPECTED_FIGURES)
        for arm in comparison['arms']:
            row, figures = by_cell[(arm['wording'], arm['model'])], arm['figures']
            assert row['cosine middle band'] == _rate_cell(figures['middle_band'])
            assert row['judged restate band'] == _rate_cell(figures['restate_band'], 'judged')
            assert row['every judged case'] == _rate_cell(figures['judged'])
            middle, restate, judged, _, _ = EXPECTED_FIGURES[(arm['wording'], arm['model'])]
            assert row['cosine middle band'].startswith(f'{middle}/2 ')
            assert row['judged restate band'].startswith(f'{restate}/1 ')
            assert row['every judged case'].startswith(f'{judged}/3 ')

    def test_each_effect_row_carries_its_difference_against_the_named_baseline(
        self, tmp_path: Path,
    ) -> None:
        comparison = _mod().build_comparison(_mod().load_arms(_grid(tmp_path)))
        baseline = comparison['baseline']
        sections = _sections(_mod().render_markdown(comparison))
        prompt_heading, prompt_rows = _section(sections, 'Prompt effect')
        model_heading, model_rows = _section(sections, 'Model effect')
        assert f'`{baseline["wording"]}`' in prompt_heading
        assert f'`{baseline["model"]}`' in model_heading
        assert {(row['model'], row['wording']) for row in prompt_rows} == {
            (model, wording) for model in MODELS for wording in WORDINGS[1:]
        }
        for row in prompt_rows:
            effect = comparison['effects']['prompt'][row['model']][row['wording']]
            assert row['judged_accuracy'] == str(effect['judged_accuracy'])
        assert {(row['wording'], row['model']) for row in model_rows} == {
            (wording, model) for wording in WORDINGS for model in MODELS[1:]
        }
        for row in model_rows:
            effect = comparison['effects']['model'][row['wording']][row['model']]
            assert row['judged_accuracy'] == str(effect['judged_accuracy'])


class TestMain:
    def test_it_writes_the_comparison_and_its_markdown_render(self, tmp_path: Path) -> None:
        out = tmp_path / 'arms' / 'comparison.json'
        argv = [a for path in _grid(tmp_path) for a in ('--arm', str(path))]
        assert _mod().main([*argv, '--out', str(out)]) == 0
        comparison = json.loads(out.read_text())
        assert comparison == _mod().build_comparison(_mod().load_arms(_grid(tmp_path)))
        assert (out.parent / 'comparison.md').read_text() == _mod().render_markdown(comparison)

    def test_an_out_path_that_is_not_json_is_refused_before_anything_is_read(
        self, tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match='--out'):
            _mod().main([
                '--arm', str(tmp_path / 'absent.json'), '--out', str(tmp_path / 'comparison.md'),
            ])
        assert list(tmp_path.iterdir()) == []
