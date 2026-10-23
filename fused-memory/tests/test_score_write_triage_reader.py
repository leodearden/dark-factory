"""Tests for score_write_triage_reader.py — the rubric v2 reader-consequence scorer.

Reached only through the public surface: ``score_reader``, its refusal
``IncompleteJudgementError``, ``render_markdown`` and the ``main`` CLI. The
scripts are loaded by path (``scripts/`` is not a package), lazily through
``_mod()``.

The synthetic corpus's expectations were worked by hand from the rubric's
derivation table; each item's comment states its cells.
"""
from __future__ import annotations

import functools
import json
import math
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

SCRIPTS = Path(__file__).parent.parent / 'scripts'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'score_write_triage_reader.py', mod_name='score_write_triage_reader',
    )


@functools.cache
def _stats() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'write_triage_stats.py', mod_name='write_triage_stats')


_FLAT = {'home': 'YES', 'loss': 'NO', 'conflict': 'NO'}
_MORE = {'home': 'YES', 'loss': 'YES', 'conflict': 'NO'}


def _item(
    entry: str, outcome: str, target: str | None = None, *, band: str | None = 'judge',
    project: str = 'df', stratum: str | None = None, **extra: Any,
) -> dict[str, Any]:
    return {
        'entry_id': entry, 'outcome': outcome, 'target_id': target, 'band': band,
        'project': project, 'stratum': stratum or outcome[0], **extra,
    }


def _vote(
    entry: str, target: str, *, rater: str = 'opus', role: str = 'primary',
    basis: str = 'current', **answers: str,
) -> dict[str, Any]:
    return {
        'entry_id': entry, 'target_id': target, 'basis': basis, 'rater': rater,
        'role': role, 'batch': 'b1', **answers,
    }


def _corpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items = [
        # triage {} / status quo {duplicate_clutter}; M1 not hidden; drifted at attach time
        _item('R1', 'restated', 'P1', band='restated', via_sibling=False, cross_category=False,
              parent_edited_after_attach=True),
        # {hidden_detail} / {}; M1 hidden
        _item('R2', 'restated', 'P2', band='restated', via_sibling=True, cross_category=False),
        # {hidden_conflict, wrong_home} / {orphaned_conflict}; M1 hidden
        _item('R3', 'restated', 'P3', band='restated', project='reify', via_sibling=False,
              cross_category=True),
        # {wrong_home} / {}; band unknown
        _item('R4', 'restated', 'P4', band=None),
        # {} / {}; parent edited, no attach-time rating
        _item('A1', 'amended', 'P5', via_sibling=False, cross_category=False,
              parent_edited_after_attach=True),
        # {digest_conflict} / {orphaned_conflict}
        _item('A2', 'amended', 'P6', via_sibling=False, cross_category=True),
        # {} / {orphaned_conflict}; central
        _item('C1', 'contested', 'P7', via_sibling=False, cross_category=False),
        # {} / {orphaned_conflict}; secondary, ¬HOME (ruling 5b)
        _item('C2', 'contested', 'P8', project='reify', via_sibling=True, cross_category=False),
        # {false_conflict, wrong_home} / {}
        _item('C3', 'contested', 'P9', via_sibling=False, cross_category=False),
        # {attached_supersession} / {orphaned_conflict}; scope unknown
        _item('C4', 'contested', 'P10', via_sibling=False, cross_category=False,
              supersedes_target=True),
        # decisive UNCLEAR: the status quo needs LOSS
        _item('C5', 'contested', 'P11', via_sibling=False, cross_category=False),
        # {duplicate_clutter}; a CONFLICT against a marked record (linked)
        _item('S1', 'stored', band='stored', earlier_ids=['E1', 'E2'], superseded_marks=['E1']),
        # {orphaned_conflict}; unlinked
        _item('S2', 'stored', band='stored', earlier_ids=['E3']),
        # {}; no earlier records
        _item('S3', 'stored', band='stored', earlier_ids=[]),
    ]
    verdicts = [
        _vote('R1', 'P1', **_FLAT),
        _vote('R1', 'P1', basis='attach_time', **_MORE),
        _vote('R2', 'P2', **_MORE),
        _vote('R3', 'P3', home='NO', conflict='YES', scope='CENTRAL'),
        _vote('R4', 'P4', home='NO', conflict='NO'),
        _vote('A1', 'P5', **_MORE),
        _vote('A2', 'P6', home='YES', conflict='YES', scope='SECONDARY'),
        _vote('C1', 'P7', conflict='YES', scope='CENTRAL'),
        _vote('C2', 'P8', home='NO', conflict='YES', scope='SECONDARY'),
        _vote('C3', 'P9', home='NO', conflict='NO'),
        _vote('C4', 'P10', conflict='YES'),
        _vote('C5', 'P11', home='YES', conflict='NO', loss='UNCLEAR',
              unclear_reason='the parent is cut at the deciding threshold'),
        _vote('S1', 'E1', conflict='YES'),
        _vote('S1', 'E2', **_FLAT),
        _vote('S2', 'E3', conflict='YES'),
    ]
    return items, verdicts


_STRATA = {'r': 40, 'a': 20, 'c': 5, 's': 30}


@functools.cache
def _report() -> dict[str, Any]:
    items, verdicts = _corpus()
    return _mod().score_reader(items, verdicts)


@functools.cache
def _weighted_report() -> dict[str, Any]:
    items, verdicts = _corpus()
    return _mod().score_reader(items, verdicts, strata=_STRATA)


def _block(hits: int, n: int, *, unclear: int = 0) -> dict[str, Any]:
    return {
        'n': n, 'hits': hits, 'rate': round(hits / n, 4) if n else None,
        'wilson95': _stats().wilson95(hits, n), 'unclear': unclear,
        'population_rate': None, 'population_interval95': None,
    }


class TestTheItemRowContract:
    @pytest.mark.parametrize(('row', 'match'), [
        (_item('X', 'restated', None), 'row 0.*X.*target_id'),
        (_item('X', 'stored', 'P1', band='stored'), 'row 0.*X.*target_id'),
        (_item('X', 'filed', 'P1'), "row 0.*'filed'"),
        (_item('X', 'amended', 'P1', band='middle'), "row 0.*'middle'"),
        (_item('X', 'amended', 'P1', earlier_ids=['E1']), 'row 0.*earlier_ids'),
        (_item('X', 'stored', band='stored', supersedes_target=True), 'row 0.*supersedes_target'),
        ({'entry_id': 'X', 'outcome': 'amended', 'target_id': 'P1'}, 'row 0.*X.*lacks'),
    ])
    def test_a_malformed_item_is_refused_naming_its_row(
        self, row: dict[str, Any], match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            _mod().score_reader([row], [])

    def test_a_duplicate_entry_is_refused(self) -> None:
        rows = [_item('X', 'contested', 'P1'), _item('X', 'contested', 'P2')]
        with pytest.raises(ValueError, match="row 1.*'X'"):
            _mod().score_reader(rows, [_vote('X', 'P1', conflict='YES')])


class TestTheReport:
    def test_states_its_rubric_rules_and_weights(self) -> None:
        report = _report()
        assert report['rubric'] == 'v2'
        assert set(report['cell_rules']) == {
            'hidden_detail', 'hidden_conflict', 'digest_conflict', 'wrong_home',
            'false_conflict', 'duplicate_clutter', 'orphaned_conflict', 'attached_supersession',
        }
        assert report['harm_weights'] == {
            'hidden_conflict': 3, 'digest_conflict': 2, 'false_conflict': 2,
            'orphaned_conflict': 1.5, 'hidden_detail': 1, 'wrong_home': 1,
            'attached_supersession': 1, 'duplicate_clutter': 0.5,
        }
        assert report['harm_weights_status'] == 'provisional'
        assert report['bootstrap'] == {'resamples': 10_000, 'seed': 7157}

    def test_population(self) -> None:
        assert _report()['population'] == {
            'n_items': 14,
            'by_outcome': {'amended': 2, 'contested': 5, 'restated': 4, 'stored': 3},
            'by_stratum': {'a': 2, 'c': 5, 'r': 4, 's': 3},
            'strata': None,
            'decided': 13,
            'unclear': 1,
            'unclear_entry_ids': ['C5'],
        }

    def test_cells_per_regime(self) -> None:
        assert _report()['cells'] == {
            'triage': {
                'hidden_detail': 1, 'hidden_conflict': 1, 'digest_conflict': 1,
                'wrong_home': 3, 'false_conflict': 1, 'duplicate_clutter': 1,
                'orphaned_conflict': 1, 'attached_supersession': 1,
            },
            'status_quo': {
                'hidden_detail': 0, 'hidden_conflict': 0, 'digest_conflict': 0,
                'wrong_home': 0, 'false_conflict': 0, 'duplicate_clutter': 2,
                'orphaned_conflict': 6, 'attached_supersession': 0,
            },
        }

    def test_m1_hidden_information_in_the_restated_band(self) -> None:
        assert _report()['metrics']['m1'] == {**_block(2, 3), 'band_unknown': 1}

    def test_m2_ack_truth_per_outcome(self) -> None:
        m2 = _report()['metrics']['m2']
        assert m2['restated'] == {
            **_block(1, 4),
            'by_project': {'df': {'n': 3, 'hits': 1}, 'reify': {'n': 1, 'hits': 0}},
        }
        assert m2['amended'] == {**_block(1, 2), 'by_project': {'df': {'n': 2, 'hits': 1}}}
        assert m2['contested'] == {
            **_block(2, 4, unclear=1),
            'by_project': {'df': {'n': 3, 'hits': 1}, 'reify': {'n': 1, 'hits': 1}},
        }
        assert m2['stored'] == {**_block(1, 3), 'by_project': {'df': {'n': 3, 'hits': 1}}}

    def test_m3_hidden_corrections(self) -> None:
        assert _report()['metrics']['m3'] == {
            **_block(3, 7, unclear=1),
            'by_ending': {
                'restated': 1, 'amended': 1, 'stored_unlinked': 1,
                'contested': 3, 'stored_linked': 1,
            },
        }

    def test_m4_wrong_home_over_decided_attaches(self) -> None:
        assert _report()['metrics']['m4'] == {
            **_block(3, 10, unclear=1),
            'by_route': {
                'direct': _block(2, 7, unclear=1),
                'via_sibling': _block(0, 2),
                'unknown': _block(1, 1),
            },
            'by_category': {
                'same': _block(1, 7, unclear=1),
                'cross': _block(1, 2),
                'unknown': _block(1, 1),
            },
        }

    def test_m5_conflict_flag_precision_with_its_scope_split(self) -> None:
        assert _report()['metrics']['m5'] == {
            **_block(3, 4, unclear=1),
            'by_scope': {'central': 1, 'secondary': 1, 'unknown': 1},
        }

    def test_m6_read_time_drift(self) -> None:
        assert _report()['metrics']['m6'] == {
            'parent_edited': 2,
            'attach_time_rated': 1,
            'drifted': 1,
            'drifted_entry_ids': ['R1'],
            'attach_time_undecided': 0,
            'attach_time_unrated': 1,
        }

    def test_harm_per_write_in_each_regime(self) -> None:
        harm = _report()['harm']
        assert harm['n'] == 13
        assert harm['unclear'] == 1
        assert harm['per_write'] == {'triage': round(14 / 13, 4), 'status_quo': round(10 / 13, 4)}
        assert harm['delta_per_write'] == round(4 / 13, 4)
        low, high = harm['delta_interval95']
        assert low <= high
        assert harm['population_per_write'] is None
        assert harm['population_delta_per_write'] is None
        assert harm['population_delta_interval95'] is None

    def test_a_rerun_gives_the_same_report(self) -> None:
        items, verdicts = _corpus()
        assert _mod().score_reader(items, verdicts) == _report()


class TestHarmWeights:
    def test_a_custom_map_changes_the_totals(self) -> None:
        items, verdicts = _corpus()
        report = _mod().score_reader(items, verdicts, harm_weights={'wrong_home': 10})
        assert report['harm_weights'] == {'wrong_home': 10}
        assert report['harm_weights_status'] == 'custom'
        assert report['harm']['per_write'] == {'triage': round(30 / 13, 4), 'status_quo': 0.0}

    @pytest.mark.parametrize('weights', [
        {'misfile_attach': 1},
        {'wrong_home': -1},
        {'wrong_home': math.nan},
        {'wrong_home': True},
        {'wrong_home': '1'},
    ])
    def test_a_bad_weight_is_refused_listing_the_cells(self, weights: dict[str, Any]) -> None:
        items, verdicts = _corpus()
        with pytest.raises(ValueError, match='wrong_home.*hidden_conflict|hidden_detail'):
            _mod().score_reader(items, verdicts, harm_weights=weights)


class TestStrata:
    def test_a_proportion_reports_its_population_rate(self) -> None:
        m4 = _weighted_report()['metrics']['m4']
        # weights N/n: r 40/4, a 20/2, c 5/5 → (10 + 10 + 1) / (6 * 10 + 4 * 1)
        assert m4['population_rate'] == round(21 / 64, 4)
        low, high = m4['population_interval95']
        assert low <= high
        assert _weighted_report()['metrics']['m2']['restated']['population_rate'] == 0.25

    def test_harm_reports_population_figures(self) -> None:
        harm = _weighted_report()['harm']
        assert harm['population_per_write'] == {
            'triage': round(104 / 94, 4), 'status_quo': round(59.5 / 94, 4),
        }
        assert harm['population_delta_per_write'] == round(44.5 / 94, 4)
        low, high = harm['population_delta_interval95']
        assert low <= high

    def test_a_domain_interval_varies_with_how_many_sampled_items_fall_in_the_domain(self) -> None:
        # M3's domain is the CONFLICT writes. Every restated one is hidden and no
        # contested one is, so only the restated sample's share of CONFLICT
        # writes can move the rate; the interval must not collapse to a point.
        conflict = {'home': 'YES', 'loss': 'NO', 'conflict': 'YES', 'scope': 'CENTRAL'}
        items = [_item(f'R{n}', 'restated', f'P{n}') for n in range(1, 5)]
        items += [_item(f'C{n}', 'contested', f'Q{n}') for n in range(1, 5)]
        verdicts = [_vote('R1', 'P1', **conflict), _vote('R2', 'P2', **conflict),
                    _vote('R3', 'P3', **_FLAT), _vote('R4', 'P4', **_FLAT)]
        verdicts += [_vote(f'C{n}', f'Q{n}', **conflict) for n in range(1, 5)]
        m3 = _mod().score_reader(items, verdicts, strata={'r': 40, 'c': 4})['metrics']['m3']
        assert (m3['n'], m3['hits']) == (6, 2)
        assert m3['population_rate'] == round(20 / 24, 4)
        low, high = m3['population_interval95']
        assert low < m3['population_rate'] < high

    def test_the_strata_are_echoed(self) -> None:
        assert _weighted_report()['population']['strata'] == _STRATA

    def test_an_item_outside_every_stratum_is_refused(self) -> None:
        items, verdicts = _corpus()
        with pytest.raises(ValueError, match="'s'"):
            _mod().score_reader(items, verdicts, strata={'r': 40, 'a': 20, 'c': 5})

    def test_a_stratum_smaller_than_its_sample_is_refused(self) -> None:
        items, verdicts = _corpus()
        with pytest.raises(ValueError, match="'c'.*5"):
            _mod().score_reader(items, verdicts, strata={**_STRATA, 'c': 4})


class TestKappa:
    def _contested(self, home_pairs: list[tuple[str, str]], **extra: Any) -> dict[str, Any]:
        items, verdicts = [], []
        for index, (first, second) in enumerate(home_pairs):
            entry = f'K{index:03d}'
            items.append(_item(entry, 'contested', 'P'))
            verdicts.append(_vote(entry, 'P', home=first, conflict='YES', **extra))
            verdicts.append(_vote(entry, 'P', rater='sonnet', role='second', home=second,
                                  conflict='YES', **extra))
        return _mod().score_reader(items, verdicts)['kappa']

    def test_kappa_between_primary_and_second_per_binary(self) -> None:
        kappa = self._contested([('YES', 'YES'), ('NO', 'NO'), ('YES', 'NO'), ('NO', 'YES')])
        assert kappa['floor'] == 0.70
        assert kappa['min_overlap'] == 60
        assert kappa['by_binary']['home'] == {'n_overlap': 4, 'kappa': 0.0, 'floor_met': False}
        assert kappa['by_binary']['conflict'] == {
            'n_overlap': 4, 'kappa': None, 'floor_met': False,
        }
        assert kappa['by_binary']['loss'] == {'n_overlap': 0, 'kappa': None, 'floor_met': False}

    def test_scope_kappa_reads_central_and_secondary(self) -> None:
        items, verdicts = [], []
        for index, (first, second) in enumerate(
            [('CENTRAL', 'CENTRAL'), ('SECONDARY', 'SECONDARY'), ('CENTRAL', 'SECONDARY')],
        ):
            entry = f'K{index}'
            items.append(_item(entry, 'contested', 'P'))
            verdicts.append(_vote(entry, 'P', conflict='YES', scope=first))
            verdicts.append(_vote(entry, 'P', rater='sonnet', role='second', conflict='YES',
                                  scope=second))
        scope = _mod().score_reader(items, verdicts)['kappa']['by_binary']['scope']
        assert scope == {'n_overlap': 3, 'kappa': 0.4, 'floor_met': False}

    def test_the_floor_needs_both_agreement_and_overlap(self) -> None:
        kappa = self._contested([('YES', 'YES')] * 30 + [('NO', 'NO')] * 30)
        assert kappa['by_binary']['home'] == {'n_overlap': 60, 'kappa': 1.0, 'floor_met': True}


class TestRefusal:
    def test_a_decisive_unrated_binary_refuses_naming_every_gap(self) -> None:
        items = [
            _item('X2', 'restated', 'P2', band='restated'),
            _item('X1', 'restated', 'P1', band='restated'),
        ]
        verdicts = [_vote('X2', 'P2', home='YES', conflict='NO')]
        with pytest.raises(_mod().IncompleteJudgementError) as refusal:
            _mod().score_reader(items, verdicts)
        assert refusal.value.gaps == (
            ('X1', 'P1', 'conflict', 'unrated'), ('X2', 'P2', 'loss', 'unrated'),
        )
        lines = str(refusal.value).splitlines()
        assert len(lines) == 3
        assert 'X1' in lines[1] and 'conflict' in lines[1] and 'unrated' in lines[1]

    def test_a_decisive_tie_refuses(self) -> None:
        verdicts = [_vote('X', 'P', conflict='YES'),
                    _vote('X', 'P', rater='sonnet', role='second', conflict='NO', home='YES',
                          loss='NO')]
        with pytest.raises(_mod().IncompleteJudgementError) as refusal:
            _mod().score_reader([_item('X', 'contested', 'P')], verdicts)
        assert refusal.value.gaps == (('X', 'P', 'conflict', 'tied'),)

    def test_a_non_decisive_tie_is_accepted(self) -> None:
        verdicts = [_vote('X', 'P', conflict='YES', home='YES'),
                    _vote('X', 'P', rater='sonnet', role='second', conflict='YES', home='NO')]
        report = _mod().score_reader([_item('X', 'contested', 'P')], verdicts)
        assert report['metrics']['m5']['hits'] == 1

    def test_a_missing_attach_time_judgement_never_refuses(self) -> None:
        item = _item('X', 'contested', 'P', parent_edited_after_attach=True)
        report = _mod().score_reader([item], [_vote('X', 'P', conflict='YES')])
        assert report['metrics']['m6']['attach_time_unrated'] == 1

    def test_an_undecided_attach_time_judgement_is_counted_not_refused(self) -> None:
        item = _item('X', 'contested', 'P', parent_edited_after_attach=True)
        verdicts = [_vote('X', 'P', conflict='YES'),
                    _vote('X', 'P', basis='attach_time', home='YES')]
        report = _mod().score_reader([item], verdicts)
        assert report['metrics']['m6']['attach_time_undecided'] == 1
        assert report['metrics']['m6']['attach_time_rated'] == 0


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path


class TestTheCli:
    def _inputs(self, tmp_path: Path) -> list[str]:
        items, verdicts = _corpus()
        strata = tmp_path / 'strata.json'
        strata.write_text(json.dumps(_STRATA))
        return [
            '--items', str(_write_jsonl(tmp_path / 'items.jsonl', items)),
            '--verdicts', str(_write_jsonl(tmp_path / 'v1.jsonl', verdicts[:5])),
            str(_write_jsonl(tmp_path / 'v2.jsonl', verdicts[5:])),
            '--strata', str(strata),
        ]

    def test_prints_the_report_and_writes_both_files(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        out = tmp_path / 'deep' / 'report.json'
        markdown = tmp_path / 'other' / 'report.md'
        argv = [*self._inputs(tmp_path), '--out', str(out), '--markdown', str(markdown)]
        assert _mod().main(argv) == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed == _weighted_report()
        assert json.loads(out.read_text()) == printed
        assert markdown.read_text() == _mod().render_markdown(printed)

    def test_takes_a_harm_weight_map(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        weights = tmp_path / 'weights.json'
        weights.write_text(json.dumps({'wrong_home': 10}))
        assert _mod().main([*self._inputs(tmp_path), '--harm-weights', str(weights)]) == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed['harm_weights'] == {'wrong_home': 10}
        assert printed['harm']['per_write']['triage'] == round(30 / 13, 4)

    def test_a_refusal_exits_1_names_each_gap_and_writes_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        items = _write_jsonl(tmp_path / 'items.jsonl', [
            _item('X1', 'contested', 'P1'), _item('X2', 'contested', 'P2'),
        ])
        verdicts = _write_jsonl(tmp_path / 'verdicts.jsonl', [_vote('X0', 'P0', home='YES')])
        out, markdown = tmp_path / 'report.json', tmp_path / 'report.md'
        code = _mod().main([
            '--items', str(items), '--verdicts', str(verdicts),
            '--out', str(out), '--markdown', str(markdown),
        ])
        assert code == 1
        captured = capsys.readouterr()
        assert captured.out == ''
        lines = captured.err.strip().splitlines()
        assert len(lines) == 3
        assert 'X1 P1 conflict' in lines[1] and 'X2 P2 conflict' in lines[2]
        assert not out.exists() and not markdown.exists()

    def test_a_malformed_line_names_its_path_and_line(self, tmp_path: Path) -> None:
        items = tmp_path / 'items.jsonl'
        items.write_text(json.dumps(_item('X', 'contested', 'P')) + '\n{not json\n')
        verdicts = _write_jsonl(tmp_path / 'verdicts.jsonl', [_vote('X', 'P', conflict='YES')])
        with pytest.raises(ValueError, match=f'{items}:2'):
            _mod().main(['--items', str(items), '--verdicts', str(verdicts)])


def _table_rows(markdown: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in markdown.splitlines():
        if line.startswith('|') and not set(line) <= set('|-: '):
            cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
            assert cells[0] not in rows, f'two table rows are labelled {cells[0]!r}'
            rows[cells[0]] = cells[1:]
    return rows


def _shown(value: Any) -> str:
    return '—' if value is None else str(value)


class TestRenderMarkdown:
    def test_is_deterministic(self) -> None:
        assert _mod().render_markdown(_weighted_report()) == _mod().render_markdown(
            _weighted_report(),
        )

    def test_has_one_row_per_metric_with_the_reports_numbers(self) -> None:
        report = _weighted_report()
        rows = _table_rows(_mod().render_markdown(report))
        metrics = report['metrics']
        blocks = {
            'M1': metrics['m1'],
            **{f'M2 {outcome}': block for outcome, block in metrics['m2'].items()},
            'M3': metrics['m3'],
            'M4': metrics['m4'],
            'M5': metrics['m5'],
        }
        for label, block in blocks.items():
            assert rows[label][:7] == [
                _shown(block[key]) for key in (
                    'n', 'hits', 'rate', 'wilson95', 'population_rate',
                    'population_interval95', 'unclear',
                )
            ], label
        m6 = metrics['m6']
        assert rows['M6'][:2] == [str(m6['attach_time_rated']), str(m6['drifted'])]

    def test_has_a_cells_table_per_regime_with_the_weights(self) -> None:
        report = _weighted_report()
        rows = _table_rows(_mod().render_markdown(report))
        for cell, rule_count in report['cells']['triage'].items():
            assert rows[cell][:3] == [
                str(rule_count),
                str(report['cells']['status_quo'][cell]),
                _shown(report['harm_weights'].get(cell)),
            ]

    def test_has_the_harm_block_and_its_weight_status(self) -> None:
        report = _weighted_report()
        markdown = _mod().render_markdown(report)
        rows = _table_rows(markdown)
        harm = report['harm']
        assert rows['per write (sample)'] == [
            str(harm['per_write']['triage']), str(harm['per_write']['status_quo']),
            str(harm['delta_per_write']), str(harm['delta_interval95']),
        ]
        assert rows['per write (population)'] == [
            str(harm['population_per_write']['triage']),
            str(harm['population_per_write']['status_quo']),
            str(harm['population_delta_per_write']),
            str(harm['population_delta_interval95']),
        ]
        assert report['harm_weights_status'] in markdown

    def test_has_the_kappa_table(self) -> None:
        report = _weighted_report()
        rows = _table_rows(_mod().render_markdown(report))
        for binary, block in report['kappa']['by_binary'].items():
            assert rows[binary] == [
                str(block['n_overlap']), _shown(block['kappa']),
                'yes' if block['floor_met'] else 'no',
            ]


_LAYERING_PROBE = """
import importlib.util
import json
import sys
from pathlib import Path

scripts = Path('scripts')
stems = {path.stem for path in scripts.glob('*.py')}
stem = sys.argv[1]
spec = importlib.util.spec_from_file_location(stem, scripts / f'{stem}.py')
module = importlib.util.module_from_spec(spec)
sys.modules[stem] = module
spec.loader.exec_module(module)
print(json.dumps(sorted(stems & set(sys.modules))))
"""


class TestTheV2ScriptsImportOnlyDownward:
    """Each v2 module loads exactly its downward closure of ``scripts/`` stems.

    The loads go by path, so no linter sees an upward or sideways one. Each
    module is loaded in its OWN fresh interpreter, so a sibling cannot hide
    behind a ``sys.modules`` entry an earlier load left.
    """

    @pytest.mark.parametrize(('stem', 'closure'), [
        ('write_triage_stats', {'write_triage_stats'}),
        ('write_triage_reader_rubric', {'write_triage_reader_rubric'}),
        ('score_write_triage_reader', {
            'score_write_triage_reader', 'write_triage_reader_rubric', 'write_triage_stats',
        }),
        ('build_write_triage_v2_soak_baseline', {
            'build_write_triage_v2_soak_baseline', 'write_triage_reader_rubric',
        }),
    ])
    def test_a_module_loads_exactly_its_downward_closure(
        self, stem: str, closure: set[str],
    ) -> None:
        result = subprocess.run(
            [sys.executable, '-c', _LAYERING_PROBE, stem],
            cwd=SCRIPTS.parent, capture_output=True, text=True, check=False, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        loaded = set(json.loads(result.stdout))
        assert loaded == closure, (
            f'loading {stem} also loaded {sorted(loaded - closure)} '
            f'and not {sorted(closure - loaded)}'
        )


CALIBRATION = Path(__file__).parent.parent / 'calibration'
_BASELINE_STEM = 'write_triage_v2_soak'


def _calibration(suffix: str) -> Path:
    return CALIBRATION / f'{_BASELINE_STEM}_{suffix}'


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@functools.cache
def _committed_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    items = _jsonl_rows(_calibration('items.jsonl'))
    verdicts = _jsonl_rows(_calibration('verdicts.jsonl'))
    strata = json.loads(_calibration('strata.json').read_text())
    return items, verdicts, strata


@functools.cache
def _rescored_baseline() -> dict[str, Any]:
    items, verdicts, strata = _committed_inputs()
    return _mod().score_reader(items, verdicts, strata=strata)


@pytest.mark.xdist_group('committed_v2_soak_baseline')
class TestTheCommittedSoakBaseline:
    """The committed v2 soak baseline cannot drift from its committed inputs.

    Built by ``scripts/build_write_triage_v2_soak_baseline.py`` from the
    gitignored 2026-10-19 soak bundle. The invariants below are fixed by the
    soak's v1 labels whatever the new ratings say. The class runs on one xdist
    worker, so the cached re-scoring is paid once per run.
    """

    def test_rescoring_the_inputs_reproduces_the_report(self) -> None:
        committed = json.loads(_calibration('baseline.json').read_text())
        assert _rescored_baseline() == committed

    def test_the_markdown_is_the_rendered_report(self) -> None:
        committed = json.loads(_calibration('baseline.json').read_text())
        assert _calibration('baseline.md').read_text() == _mod().render_markdown(committed)

    def test_holds_the_soaks_553_adjudicated_attaches(self) -> None:
        population = _rescored_baseline()['population']
        assert population['n_items'] == 553
        assert population['by_outcome'] == {
            'amended': 40, 'contested': 473, 'restated': 40, 'stored': 0,
        }
        assert sum(_committed_inputs()[2].values()) == 1822

    def test_every_v1_action_changing_correction_is_a_flagged_conflict(self) -> None:
        items, verdicts, _ = _committed_inputs()
        contested = {item['entry_id'] for item in items if item['outcome'] == 'contested'}
        derived_conflicts = {
            row['entry_id'] for row in verdicts
            if row['role'] == 'derived' and row.get('conflict') == 'YES'
            and row['entry_id'] in contested
        }
        assert len(derived_conflicts) == 225
        assert _rescored_baseline()['metrics']['m5']['hits'] >= 225

    def test_counts_the_13_parents_edited_after_attach(self) -> None:
        assert _rescored_baseline()['metrics']['m6']['parent_edited'] == 13
