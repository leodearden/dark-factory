"""Tests for score_write_triage_m3_levers.py — classify the M3 hits and score each lever arm (task 7519).

Small synthetic items, verdicts, strata, features and arm rows; the
expectations are worked by hand from the strata weights. No LLM, no soak data.
"""
from __future__ import annotations

import copy
import functools
import json
import types
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

SCRIPTS = Path(__file__).parent.parent / 'scripts'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'score_write_triage_m3_levers.py', mod_name='score_write_triage_m3_levers',
    )


@functools.cache
def _scorer() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'score_write_triage_reader.py', mod_name='score_write_triage_reader',
    )


@functools.cache
def _cases() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'write_triage_m3_replay_cases.py', mod_name='write_triage_m3_replay_cases',
    )


_STRATA = {'amended-sample': 10, 'contested-census': 3, 'restated-sample': 4}
"""Weights: amended 10/2 = 5, contested 3/3 = 1, restated 4/1 = 4."""


def _item(entry: str, outcome: str, stratum: str, *, band: str = 'judge') -> dict[str, Any]:
    return {
        'entry_id': entry, 'outcome': outcome, 'target_id': f'P{entry}', 'band': band,
        'project': 'dark_factory', 'stratum': stratum, 'earlier_ids': [],
        'superseded_marks': [], 'supersedes_target': False, 'via_sibling': False,
        'cross_category': False, 'parent_edited_after_attach': False,
    }


def _vote(entry: str, *, rater: str = 'opus', role: str = 'primary', **answers: Any) -> dict[str, Any]:
    row = {
        'entry_id': entry, 'target_id': f'P{entry}', 'basis': 'current', 'rater': rater,
        'role': role, 'batch': 'b1', **answers,
    }
    if answers.get('conflict') == 'YES':
        row['conflict_quote'] = {'parent': 'the record says A', 'child': 'it says B'}
    return row


def _corpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items = [
        _item('A1', 'amended', 'amended-sample'),      # a CONFLICT amended: an M3 hit
        _item('A2', 'amended', 'amended-sample'),      # correctly amended
        _item('C1', 'contested', 'contested-census'),  # correctly contested
        _item('C2', 'contested', 'contested-census'),  # a false contest
        _item('C3', 'contested', 'contested-census'),  # CONFLICT, HOME tied
        _item('R1', 'restated', 'restated-sample', band='restated'),  # an M3 hit
    ]
    verdicts = [
        _vote('A1', home='YES', conflict='YES', scope='SECONDARY'),
        _vote('A2', home='YES', conflict='NO', loss='NO'),
        _vote('C1', home='YES', conflict='YES', scope='CENTRAL', action_changing='YES'),
        _vote('C2', home='YES', conflict='NO', loss='NO'),
        _vote('C3', home='YES', conflict='YES', scope='CENTRAL', action_changing='NO'),
        _vote('C3', rater='sonnet', role='second', home='NO', conflict='YES',
              scope='CENTRAL', action_changing='NO'),
        _vote('R1', home='YES', conflict='YES', scope='SECONDARY'),
    ]
    return items, verdicts


def _features(items: list[dict[str, Any]]) -> list[Any]:
    cases = _cases()
    return [
        cases.CaseFeatures(
            entry_id=item['entry_id'], target_id=item['target_id'], stratum=item['stratum'],
            project=item['project'], soak_outcome=item['outcome'], soak_band=item['band'],
            via_sibling=False, cross_category=False, parent_edited_after_attach=False,
            supersedes_target=False,
            route=cases.Route.NEAR_VERBATIM if item['entry_id'] == 'R1' else cases.Route.JUDGE,
            near_verbatim=item['entry_id'] == 'R1', cos_parent=0.75, source_cosine=0.75, metadata_recovered=True,
            child_chars=10, parent_chars=10,
        )
        for item in items
    ]


def _row(entry: str, outcome: str, *, arm: str = 'f3p', marked: bool | None = None,
         seconds: float | None = 1.0, usd: float = 0.01, error: str | None = None,
         judged: str | None = None) -> dict[str, Any]:
    return {
        'arm': arm, 'entry_id': entry, 'target_id': f'P{entry}', 'routed_outcome': outcome,
        'judge_outcome': judged if judged is not None else outcome, 'dispute_marked': marked,
        'calls': 0 if seconds is None else 1, 'seconds': seconds, 'usd': usd,
        'error_type': error, 'prompt_sha256': 'p' * 64, 'model': 'gpt-6.1-sol',
        'effort': 'low', 'cases_sha256': 'c' * 64, 'parent': None, 'parent_sha256': None,
    }


def _soak_rows(arm: str = 'f3p', **overrides: str) -> list[dict[str, Any]]:
    items, _ = _corpus()
    return [_row(item['entry_id'], overrides.get(item['entry_id'], str(item['outcome'])), arm=arm)
            for item in items]


def _arm_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items, _ = _corpus()
    return _mod().arm_items(items, _features(items), rows)


def _labels() -> list[dict[str, Any]]:
    return [
        {'entry_id': 'A1', 'target_id': 'PA1', 'dispute_kind': 'outdated_state',
         'self_declared_correction': True, 'dispute_in_child_core': False,
         'evidence': 'the record says A'},
        {'entry_id': 'R1', 'target_id': 'PR1', 'dispute_kind': 'contradiction',
         'self_declared_correction': False, 'dispute_in_child_core': True,
         'evidence': 'the record says A'},
    ]


class TestArmItems:
    def test_substitutes_each_rows_routed_outcome(self) -> None:
        items = _arm_items(_soak_rows(A1='stored'))
        by_id = {item['entry_id']: item for item in items}
        assert by_id['A1']['outcome'] == 'stored'
        assert by_id['A1']['earlier_ids'] == ['PA1']
        assert by_id['R1']['band'] == 'restated'
        for item in items:
            _scorer().ReaderItem.from_row(0, item)

    def test_a_dispute_marked_write_is_scored_as_contested(self) -> None:
        rows = _soak_rows()
        rows[0]['dispute_marked'] = True
        by_id = {item['entry_id']: item for item in _arm_items(rows)}
        assert by_id['A1']['outcome'] == 'contested'
        assert by_id['A1']['target_id'] == 'PA1'

    def test_an_arm_missing_a_case_is_refused_naming_it(self) -> None:
        with pytest.raises(ValueError, match='C2'):
            _arm_items([row for row in _soak_rows() if row['entry_id'] != 'C2'])


class TestScoreArm:
    def test_reports_the_scorers_own_blocks(self) -> None:
        items, verdicts = _corpus()
        scored = _mod().score_arm(items, verdicts, _STRATA)
        report = _scorer().score_reader(items, verdicts, brief=_mod().BRIEF, strata=_STRATA)
        metrics = report['metrics']
        assert scored['m1'] == metrics['m1']
        assert scored['m3'] == metrics['m3']
        assert scored['m5'] == metrics['m5']
        assert scored['m2_contested'] == metrics['m2']['contested']
        assert scored['m2_amended'] == metrics['m2']['amended']
        assert scored['m4'] == metrics['m4']
        assert scored['harm'] == report['harm']
        assert scored['closure'] is None

    def test_m3_is_the_population_hidden_share_of_conflict_writes(self) -> None:
        items, verdicts = _corpus()
        m3 = _mod().score_arm(items, verdicts, _STRATA)['m3']
        assert m3['hits'] == 2
        assert m3['population_rate'] == pytest.approx((5 + 4) / (5 + 1 + 1 + 4), abs=1e-4)


class TestTheHomeTieClosure:
    def test_a_tied_home_is_closed_both_ways(self) -> None:
        items, verdicts = _corpus()
        before = copy.deepcopy(verdicts)
        arm = _arm_items(_soak_rows(C3='amended'))
        with pytest.raises(_scorer().IncompleteJudgementError):
            _scorer().score_reader(arm, verdicts, brief=_mod().BRIEF, strata=_STRATA)
        scored = _mod().score_arm(arm, verdicts, _STRATA)
        assert verdicts == before
        assert scored['closure'] == {
            'pairs': [['C3', 'PC3']], 'raters': list(_mod().BOUND_RATERS),
        }
        assert scored['m3']['hits'] == 3
        m4 = scored['m4']['population_rate']
        assert m4[0] < m4[1]
        assert set(scored['m4']['by_closure']) == set(_mod().BOUND_RATERS)
        assert len(scored['harm']['population_delta_per_write']) == 2
        assert len(scored['m2_amended']['population_rate']) == 2

    def test_another_gap_refuses(self) -> None:
        items, verdicts = _corpus()
        unrated = [row for row in verdicts if row['entry_id'] != 'A2']
        with pytest.raises(_scorer().IncompleteJudgementError):
            _mod().score_arm(items, unrated, _STRATA)


class TestPairedDelta:
    def test_flipping_one_amended_conflict_to_contested(self) -> None:
        items, verdicts = _corpus()
        base = _arm_items(_soak_rows())
        arm = _arm_items(_soak_rows(A1='contested'))
        delta = _mod().paired_delta(arm, base, verdicts, _STRATA)
        assert delta['m3_delta'] == pytest.approx(-5 / (5 + 1 + 1 + 4), abs=1e-4)
        assert delta['m3_delta_interval95'][0] <= delta['m3_delta'] <= delta['m3_delta_interval95'][1]
        assert delta['transitions'] == {'amended-sample': {'amended->contested': 1}}
        assert delta['false_conflict_change'] == {'sample': 0, 'population': 0}

    def test_contesting_a_correct_amendment_adds_population_false_contests(self) -> None:
        items, verdicts = _corpus()
        base = _arm_items(_soak_rows())
        arm = _arm_items(_soak_rows(A2='contested'))
        delta = _mod().paired_delta(arm, base, verdicts, _STRATA)
        assert delta['m3_delta'] == 0
        assert delta['false_conflict_change'] == {'sample': 1, 'population': 5.0}

    def test_uses_the_scorers_bootstrap_settings(self) -> None:
        items, verdicts = _corpus()
        base = _arm_items(_soak_rows())
        delta = _mod().paired_delta(base, base, verdicts, _STRATA)
        assert delta['bootstrap'] == {
            'resamples': _scorer().BOOTSTRAP_RESAMPLES, 'seed': _scorer().BOOTSTRAP_SEED,
        }
        assert delta['m3_delta'] == 0


class TestReproAgreement:
    def test_confusion_over_the_soak_judge_band(self) -> None:
        items, _ = _corpus()
        rows = [_row(item['entry_id'], 'stored', arm='psi-repro', judged=judged)
                for item, judged in zip(items, ['amended', 'contested', 'contested',
                                                'stored', 'contested', 'restated'], strict=True)
                if item['band'] == 'judge']
        agreement = _mod().repro_agreement(rows, items)
        assert agreement['n'] == 5
        assert agreement['agree'] == 3
        assert agreement['rate'] == pytest.approx(0.6)
        assert agreement['confusion']['amended'] == {'amended': 1, 'contested': 1}
        assert agreement['confusion']['contested'] == {'contested': 2, 'stored': 1}


class TestCost:
    def test_calls_latency_fail_opens_and_spend(self) -> None:
        rows = [_row('A1', 'amended', seconds=1.0, usd=0.01),
                _row('A2', 'amended', seconds=3.0, usd=0.03),
                _row('C1', 'stored', seconds=2.0, usd=0.0, error='TimeoutError'),
                _row('R1', 'restated', seconds=None, usd=0.0)]
        figures = _mod().cost(rows)
        assert figures['writes'] == 4
        assert figures['calls_per_write'] == pytest.approx(0.75)
        assert figures['fail_open'] == 1
        assert (figures['p50_seconds'], figures['p95_seconds']) == (2.0, 3.0)
        assert figures['usd'] == pytest.approx(0.04)
        assert figures['usd_per_1000_writes'] == pytest.approx(10.0)
        assert figures['usd_per_1000_judged_writes'] == pytest.approx(0.04 / 3 * 1000)


class TestClassify:
    def _classify(self, labels: list[dict[str, Any]]) -> dict[str, Any]:
        items, verdicts = _corpus()
        return _mod().classify(items, verdicts, _STRATA, _features(items), labels)

    def test_refuses_labels_that_miss_a_hit_or_label_a_non_hit(self) -> None:
        with pytest.raises(ValueError, match='R1'):
            self._classify(_labels()[:1])
        extra = {**_labels()[0], 'entry_id': 'A2', 'target_id': 'PA2'}
        with pytest.raises(ValueError, match='A2'):
            self._classify([*_labels(), extra])

    def test_refuses_a_second_label_row_for_one_entry(self) -> None:
        relabelled = {**_labels()[0], 'dispute_kind': 'contradiction'}
        with pytest.raises(ValueError, match="'A1'.*already labels"):
            self._classify([*_labels(), relabelled])

    def test_refuses_a_label_outside_the_vocabulary(self) -> None:
        bad = [{**_labels()[0], 'dispute_kind': 'vibes'}, _labels()[1]]
        with pytest.raises(ValueError, match='vibes'):
            self._classify(bad)

    def test_the_dispute_kinds_are_a_closed_vocabulary(self) -> None:
        kinds = _mod().DISPUTE_KINDS
        assert set(kinds) == {'outdated_state', 'status_overtaken', 'newer_measurement',
                              'contradiction'}

    def test_accepts_every_kind_in_the_vocabulary(self) -> None:
        for kind in _mod().DISPUTE_KINDS:
            table = self._classify([{**_labels()[0], 'dispute_kind': kind}, _labels()[1]])
            assert kind in table['dimensions']['dispute_kind']

    def test_counts_and_weighs_the_hits_on_every_dimension(self) -> None:
        table = self._classify(_labels())
        assert (table['hits'], table['population_hits']) == (2, 9.0)
        dims = table['dimensions']
        assert dims['stratum'] == {
            'amended-sample': {'n': 1, 'population': 5.0},
            'restated-sample': {'n': 1, 'population': 4.0},
        }
        assert dims['scope'] == {'secondary': {'n': 2, 'population': 9.0}}
        assert dims['home'] == {'yes': {'n': 2, 'population': 9.0}}
        assert dims['route'] == {'judge': {'n': 1, 'population': 5.0},
                                 'near_verbatim': {'n': 1, 'population': 4.0}}
        assert dims['dispute_kind'] == {'contradiction': {'n': 1, 'population': 4.0},
                                        'outdated_state': {'n': 1, 'population': 5.0}}
        assert dims['cos_parent_bucket'] == {'0.70-0.80': {'n': 2, 'population': 9.0}}
        for dimension in ('soak_band', 'action_changing', 'via_sibling', 'cross_category',
                          'project', 'self_declared_correction', 'dispute_in_child_core'):
            assert sum(cell['n'] for cell in dims[dimension].values()) == 2


class TestRender:
    def _report(self) -> dict[str, Any]:
        items, verdicts = _corpus()
        arms = {'f3p': _soak_rows(), 'lever-x': _soak_rows('lever-x', A1='contested')}
        return _mod().build_report(
            soak_items=items, verdicts=verdicts, strata=_STRATA, features=_features(items),
            labels=_labels(), arms=arms, baseline='f3p', partial={},
            provenance={'shipped_prompt_commit': 'abc'},
        )

    def test_is_deterministic_and_names_every_arm(self) -> None:
        report = self._report()
        markdown = _mod().render_markdown(report)
        assert markdown == _mod().render_markdown(self._report())
        assert 'f3p' in markdown
        assert 'lever-x' in markdown

    def test_the_report_pairs_every_arm_with_the_baseline(self) -> None:
        report = self._report()
        assert report['arms']['lever-x']['paired_vs_baseline']['m3_delta'] < 0
        assert report['arms']['f3p']['paired_vs_baseline'] is None
        assert report['arms']['lever-x']['provenance'] == {
            'prompt_sha256': 'p' * 64, 'model': 'gpt-6.1-sol', 'effort': 'low',
            'cases_sha256': 'c' * 64, 'parent': None, 'parent_sha256': None, 'rows': 6,
        }


def _child_rows(parent: list[dict[str, Any]], arm: str, *, calls: int = 0,
                **marked: bool) -> list[dict[str, Any]]:
    """A child arm's rows over *parent*'s: its verdicts, its own calls, and any marks."""
    return [
        {**row, 'arm': arm, 'parent': parent[0]['arm'], 'parent_sha256': 'd' * 64,
         'prompt_sha256': None, 'model': None, 'effort': None, 'calls': calls,
         'seconds': 2.0 if calls else None, 'usd': 0.05 if calls else 0.0,
         'dispute_marked': marked.get(row['entry_id'], False)}
        for row in parent
    ]


class TestArmsBuiltOnAParent:
    def _arm_report(self, name: str, arms: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        items, verdicts = _corpus()
        return _mod().arm_report(
            name, arms, baseline='f3p', soak_items=items, verdicts=verdicts, strata=_STRATA,
            features=_features(items),
        )

    def test_a_transforms_cumulative_cost_is_its_parents(self) -> None:
        f3p = _soak_rows()
        arms = {'f3p': f3p, 'mark': _child_rows(f3p, 'mark', A1=True)}
        report = self._arm_report('mark', arms)
        assert report['cost']['calls_per_write'] == 0
        assert report['cost']['usd'] == 0
        assert report['cumulative_cost'] == _mod().cost(f3p)
        assert report['paired_vs_baseline']['m3_delta'] < 0

    def test_a_chain_sums_every_ancestors_calls_latency_and_spend(self) -> None:
        f3p = _soak_rows()
        second = _child_rows(f3p, 'second', calls=1)
        arms = {'f3p': f3p, 'second': second, 'mark': _child_rows(second, 'mark')}
        cumulative = self._arm_report('mark', arms)['cumulative_cost']
        assert cumulative['calls_per_write'] == 2
        assert cumulative['usd'] == pytest.approx(6 * (0.01 + 0.05))
        assert (cumulative['p50_seconds'], cumulative['p95_seconds']) == (3.0, 3.0)

    def test_inherited_verdicts_have_no_agreement_of_their_own(self) -> None:
        f3p = _soak_rows()
        arms = {'f3p': f3p, 'mark': _child_rows(f3p, 'mark'), 'lever-x': _soak_rows('lever-x')}
        assert self._arm_report('mark', arms)['outcome_agreement_vs_baseline'] is None
        assert self._arm_report('lever-x', arms)['outcome_agreement_vs_baseline']['n'] == 6
        assert self._arm_report('f3p', arms)['cumulative_cost'] == self._arm_report(
            'f3p', arms)['cost']

    def test_a_child_whose_verdicts_are_not_its_parents_is_refused(self) -> None:
        f3p = _soak_rows()
        stale = _child_rows(_soak_rows(A1='contested'), 'mark')
        stale = [{**row, 'parent': 'f3p'} for row in stale]
        with pytest.raises(ValueError, match="'A1'.*'f3p'"):
            self._arm_report('mark', {'f3p': f3p, 'mark': stale})

    def test_a_child_whose_parent_is_not_scored_is_refused(self) -> None:
        orphan = _child_rows(_soak_rows('d1'), 'mark')
        with pytest.raises(ValueError, match="'d1'"):
            self._arm_report('mark', {'f3p': _soak_rows(), 'mark': orphan})

    def test_the_markdown_costs_a_child_in_full_and_lists_its_own_cost(self) -> None:
        items, verdicts = _corpus()
        f3p = _soak_rows()
        report = _mod().build_report(
            soak_items=items, verdicts=verdicts, strata=_STRATA, features=_features(items),
            labels=_labels(), arms={'f3p': f3p, 'mark': _child_rows(f3p, 'mark', A1=True)},
            baseline='f3p', partial={}, provenance={'shipped_prompt_commit': 'abc'},
        )
        markdown = _mod().render_markdown(report)
        own = markdown.split('## Own cost of the arms built on a parent')[1].split('##')[0]
        assert [line.split(' | ')[:3] for line in own.splitlines() if line.startswith('| ')][1:] == [
            ['| mark', 'f3p', '0.0000'],
        ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


class TestAssembleReport:
    """The report assembled from files, as the ``report`` command writes it."""

    def _inputs(self, root: Path) -> list[str]:
        items, verdicts = _corpus()
        paths = {name: root / f'{name}.jsonl' for name in ('items', 'verdicts', 'features', 'labels')}
        _write_jsonl(paths['items'], items)
        _write_jsonl(paths['verdicts'], verdicts)
        _write_jsonl(paths['features'], [feature.to_row() for feature in _features(items)])
        _write_jsonl(paths['labels'], _labels())
        strata = root / 'strata.json'
        strata.write_text(json.dumps(_STRATA))
        return [*(arg for name, path in paths.items() for arg in (f'--{name}', str(path))),
                '--strata', str(strata)]

    def _arms_dir(self, root: Path) -> Path:
        items, _ = _corpus()
        arms_dir = root / 'arms'
        arms_dir.mkdir()
        _write_jsonl(arms_dir / 'f3p.jsonl', _soak_rows())
        short = [_row(entry, 'amended', arm='lever-y', usd=0.02) for entry in ('A1', 'A2', 'C1')]
        _write_jsonl(arms_dir / 'lever-y.jsonl', short)
        repro = [_row(item['entry_id'], 'stored', arm='psi', judged=str(item['outcome']))
                 for item in items if item['band'] == 'judge']
        _write_jsonl(arms_dir / 'psi.jsonl', repro)
        return arms_dir

    def _report(self, tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
        out_json, out_md = tmp_path / 'report.json', tmp_path / 'report.md'
        code = _mod().main([
            'report', *self._inputs(tmp_path), '--arms-dir', str(self._arms_dir(tmp_path)),
            '--arms', 'f3p,lever-y,never-run', '--repro-arms', 'psi', '--prompt-commit', 'abc1234',
            '--out-json', str(out_json), '--out-md', str(out_md),
        ])
        assert code == 0
        return json.loads(out_json.read_text()), out_json, out_md

    def test_writes_the_json_and_its_rendered_markdown(self, tmp_path: Path) -> None:
        report, _, out_md = self._report(tmp_path)
        assert out_md.read_text() == _mod().render_markdown(report)
        assert report['provenance']['shipped_prompt_commit'] == 'abc1234'
        assert report['arm_order'] == ['f3p']

    def test_an_arm_short_of_cases_is_partial_with_its_projected_spend(
        self, tmp_path: Path,
    ) -> None:
        report, _, _ = self._report(tmp_path)
        assert report['partial'] == {
            'lever-y': {'rows': 3, 'cases': 6, 'usd': 0.06, 'projected_usd': 0.12},
            'never-run': {'rows': 0, 'cases': 6, 'usd': 0.0, 'projected_usd': None},
        }
        assert set(report['arms']) == {'f3p'}

    def test_reads_each_repro_arm_against_the_soak(self, tmp_path: Path) -> None:
        report, _, _ = self._report(tmp_path)
        agreement = report['repro']['psi']['agreement']
        assert (agreement['n'], agreement['agree']) == (5, 5)
        assert report['repro']['psi']['provenance']['rows'] == 5
