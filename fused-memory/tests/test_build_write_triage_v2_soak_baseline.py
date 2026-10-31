"""Tests for build_write_triage_v2_soak_baseline.py — the one-shot v2 soak baseline builder.

Synthetic rows only: the soak bundle is gitignored and lives in the main
checkout, so no test reads it. The script is loaded by path (``scripts/`` is
not a package), lazily through ``_mod()``. The round protocol it builds on has
its own tests in ``test_write_triage_rating_rounds.py``; the shards and
rating-round tests here drive it end to end through the soak CLI.
"""
from __future__ import annotations

import functools
import hashlib
import json
import sqlite3
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

SCRIPTS = Path(__file__).parent.parent / 'scripts'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'build_write_triage_v2_soak_baseline.py',
        mod_name='build_write_triage_v2_soak_baseline',
    )


@functools.cache
def _rounds() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'write_triage_rating_rounds.py', mod_name='write_triage_rating_rounds',
    )


@functools.cache
def _rubric() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'write_triage_reader_rubric.py', mod_name='write_triage_reader_rubric',
    )


def _adjudication(
    word: str, *, outcome: str = 'contested', new_detail: bool = True,
    dispute: str | None = None, action_changing: bool | None = None, **extra: Any,
) -> dict[str, Any]:
    return {
        'child_memory_id': 'c1', 'parent_id': 'p1', 'project': 'df',
        'category': 'procedural_knowledge', 'outcome': outcome, 'rater_verdict': word,
        'new_detail': new_detail, 'dispute': dispute, 'action_changing': action_changing,
        'likely_via_sibling': False, 'parent_edited_after_attach': False, **extra,
    }


CANDIDATE_ROWS = {
    'related_unrelated_home_no': 'home',
    'same_subsumed_home_yes': 'home',
    'same_subsumed_unrelated_conflict_no': 'conflict',
    'corrects_action_changing_conflict_yes': 'conflict',
    'corrects_action_changing_scope': 'scope',
    'corrects_noise_conflict_no': 'conflict',
    'no_new_detail_loss_no': 'loss',
}
RETIRED_ROWS = {
    'extends_home_yes': 'home',
    'corrects_not_action_changing_conflict_no': 'conflict',
}


def _determined(row: dict[str, Any], held: frozenset[str] | None = None) -> dict[str, str]:
    builder = _mod()
    determined = builder.v1_determined(row) if held is None else builder.v1_determined(row, held=held)
    return {str(binary): str(answer) for binary, answer in determined.items()}


class TestV1Rows:
    def test_names_each_candidate_and_retired_row_with_its_binary(self) -> None:
        builder = _mod()
        assert {row.name: str(row.binary) for row in builder.V1_ROWS} == CANDIDATE_ROWS
        assert {row.name: str(row.binary) for row in builder.RETIRED_V1_ROWS} == RETIRED_ROWS

    @pytest.mark.parametrize(('row', 'fixed'), [
        (_adjudication('EXTENDS'), {'extends_home_yes': 'YES'}),
        (_adjudication('CORRECTS', dispute='one_detail', action_changing=False),
         {'corrects_not_action_changing_conflict_no': 'NO'}),
        (_adjudication('CORRECTS', dispute='genuine_correction', action_changing=True,
                       new_detail=False),
         {'corrects_action_changing_conflict_yes': 'YES',
          'corrects_action_changing_scope': 'CENTRAL', 'no_new_detail_loss_no': 'NO'}),
        (_adjudication('UNRELATED', new_detail=False),
         {'related_unrelated_home_no': 'NO', 'same_subsumed_unrelated_conflict_no': 'NO',
          'no_new_detail_loss_no': 'NO'}),
        (_adjudication('UNCLEAR', new_detail=False), {}),
    ])
    def test_each_row_fixes_one_answer_for_the_adjudications_it_applies_to(
        self, row: dict[str, Any], fixed: dict[str, str],
    ) -> None:
        builder = _mod()
        assert {
            v1.name: str(answer) for v1 in (*builder.V1_ROWS, *builder.RETIRED_V1_ROWS)
            if (answer := v1.fixed(row)) is not None
        } == fixed


class TestV1Determined:
    @pytest.mark.parametrize(('row', 'expected'), [
        (_adjudication('SAME', new_detail=False), {'home': 'YES', 'loss': 'NO', 'conflict': 'NO'}),
        (_adjudication('SAME'), {'home': 'YES', 'conflict': 'NO'}),
        (_adjudication('SUBSUMED'), {'home': 'YES', 'conflict': 'NO'}),
        (_adjudication('EXTENDS'), {}),
        (_adjudication('RELATED'), {'home': 'NO'}),
        (_adjudication('UNRELATED'), {'home': 'NO', 'conflict': 'NO'}),
        (_adjudication('CORRECTS', dispute='genuine_correction', action_changing=True),
         {'conflict': 'YES', 'scope': 'CENTRAL'}),
        (_adjudication('CORRECTS', dispute='one_detail', action_changing=True),
         {'conflict': 'YES', 'scope': 'SECONDARY'}),
        (_adjudication('CORRECTS', dispute='genuine_correction', action_changing=False), {}),
        (_adjudication('CORRECTS', dispute='noise', action_changing=True), {'conflict': 'NO'}),
        (_adjudication('CORRECTS', dispute='one_detail', action_changing=False, new_detail=False),
         {}),
        (_adjudication('RELATED', new_detail=False), {'home': 'NO'}),
        (_adjudication('UNCLEAR', new_detail=False), {}),
    ])
    def test_applies_every_candidate_row_by_default(
        self, row: dict[str, Any], expected: dict[str, str],
    ) -> None:
        assert _determined(row) == expected

    def test_a_loss_answer_without_home_yes_is_pruned(self) -> None:
        held = frozenset(CANDIDATE_ROWS) - {'same_subsumed_home_yes'}
        assert _determined(_adjudication('SAME', new_detail=False), held) == {'conflict': 'NO'}

    def test_a_scope_answer_without_conflict_yes_is_pruned(self) -> None:
        held = frozenset(CANDIDATE_ROWS) - {'corrects_action_changing_conflict_yes'}
        row = _adjudication('CORRECTS', dispute='genuine_correction', action_changing=True)
        assert _determined(row, held) == {}

    def test_an_unknown_v1_word_is_refused(self) -> None:
        with pytest.raises(ValueError, match="'MAYBE'"):
            _mod().v1_determined(_adjudication('MAYBE'))


def _rated(entry: str, *, role: str = 'primary', basis: str = 'current', **answers: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        'entry_id': entry, 'target_id': 'p1', 'basis': basis, 'rater': f'opus-{role}',
        'role': role, 'batch': f'{role}_00', 'reason': 'two claims', **answers,
    }
    if answers.get('conflict') == 'YES':
        row['conflict_quote'] = {'parent': 'parent line', 'child': 'child line'}
    if answers.get('loss') == 'YES':
        row['loss_spans'] = ['child line']
    if 'UNCLEAR' in answers.values():
        row['unclear_reason'] = 'cannot tell'
    return row


def _check(word: str, answers: list[dict[str, str]], **adjudication: Any) -> Any:
    adjudications = {
        (f'c{index}', 'p1'): _adjudication(word, child_memory_id=f'c{index}', **adjudication)
        for index in range(len(answers))
    }
    rows = [_rated(f'c{index}', **answer) for index, answer in enumerate(answers)]
    return _mod().derivation_check(adjudications, rows)


def _measured(check: Any) -> tuple[int, int, int, bool]:
    return (check.n_items, check.n_compared, check.agree, check.held)


class TestDerivationCheck:
    def test_the_thresholds(self) -> None:
        builder = _mod()
        assert builder.DERIVATION_MIN_COMPARED == 20
        assert builder.DERIVATION_MIN_AGREEMENT == 0.90

    @pytest.mark.parametrize(('n_compared', 'agree', 'held'), [
        (20, 18, True), (19, 19, False), (20, 17, False),
    ])
    def test_a_row_holds_at_enough_compared_answers_and_agreement(
        self, n_compared: int, agree: int, held: bool,
    ) -> None:
        answers = [{'home': 'NO'}] * agree + [{'home': 'YES'}] * (n_compared - agree)
        check = _check('RELATED', answers)['related_unrelated_home_no']
        assert str(check.binary) == 'home'
        assert _measured(check) == (n_compared, n_compared, agree, held)

    def test_counts_only_primary_and_second_answers_on_the_current_basis(self) -> None:
        adjudications = {
            (f'c{index}', 'p1'): _adjudication('RELATED', child_memory_id=f'c{index}')
            for index in range(11)
        }
        rows = []
        for index in range(10):
            rows += [
                _rated(f'c{index}', role='primary', home='NO'),
                _rated(f'c{index}', role='second', home='NO'),
                _rated(f'c{index}', role='third', home='YES'),
                _rated(f'c{index}', role='primary', basis='attach_time', home='YES'),
            ]
        rows.append(_rated('c10', role='primary', home='UNCLEAR'))
        builder = _mod()
        checks = builder.derivation_check(adjudications, rows)
        assert _measured(checks['related_unrelated_home_no']) == (11, 20, 20, True)
        assert builder.held_rows(checks) == frozenset({'related_unrelated_home_no'})

    def test_a_retired_row_is_measured_and_never_held(self) -> None:
        check = _check('EXTENDS', [{'home': 'YES'}] * 20)['extends_home_yes']
        assert _measured(check) == (20, 20, 20, False)
        assert check.retired is True

    @pytest.mark.parametrize(('dispute', 'fixed', 'other'), [
        ('genuine_correction', 'CENTRAL', 'SECONDARY'), ('one_detail', 'SECONDARY', 'CENTRAL'),
    ])
    def test_the_scope_row_compares_central_and_secondary(
        self, dispute: str, fixed: str, other: str,
    ) -> None:
        answers = ([{'conflict': 'YES', 'scope': fixed}] * 18
                   + [{'conflict': 'YES', 'scope': other}] * 2)
        checks = _check('CORRECTS', answers, dispute=dispute, action_changing=True)
        assert str(checks['corrects_action_changing_scope'].binary) == 'scope'
        assert _measured(checks['corrects_action_changing_scope']) == (20, 20, 18, True)
        assert _measured(checks['corrects_action_changing_conflict_yes']) == (20, 20, 20, True)

    def test_the_loss_row_compares_only_the_loss_answers_raters_gave(self) -> None:
        answers = ([{'home': 'YES', 'conflict': 'NO', 'loss': 'NO'}] * 10
                   + [{'home': 'YES', 'conflict': 'NO', 'loss': 'YES'}] * 5
                   + [{'home': 'NO'}] * 5)
        checks = _check('SAME', answers, new_detail=False)
        assert _measured(checks['no_new_detail_loss_no']) == (20, 15, 10, False)
        assert _measured(checks['same_subsumed_home_yes']) == (20, 20, 15, False)
        assert _measured(checks['same_subsumed_unrelated_conflict_no']) == (20, 15, 15, False)

    def test_reports_every_candidate_and_retired_row(self) -> None:
        checks = _mod().derivation_check({}, [])
        assert set(checks) == set(CANDIDATE_ROWS) | set(RETIRED_ROWS)
        assert {name for name, check in checks.items() if check.retired} == set(RETIRED_ROWS)
        assert all(_measured(check) == (0, 0, 0, False) for check in checks.values())


def _item(**overrides: Any) -> dict[str, Any]:
    row = {
        'entry_id': 'c1', 'outcome': 'contested', 'target_id': 'p1', 'earlier_ids': [],
        'band': 'judge', 'project': 'df', 'stratum': 'contested-census',
        'via_sibling': False, 'cross_category': False, 'parent_edited_after_attach': False,
        'supersedes_target': False, 'superseded_marks': [],
    }
    return {**row, **overrides}


def _history_db(path: Path, rows: list[tuple[str, str | None, str, str, str]]) -> Path:
    connection = sqlite3.connect(path)
    connection.execute(
        'CREATE TABLE history (id TEXT PRIMARY KEY, memory_id TEXT, old_memory TEXT,'
        ' new_memory TEXT, event TEXT, created_at DATETIME, updated_at DATETIME,'
        ' is_deleted INTEGER, actor_id TEXT, role TEXT)'
    )
    for index, (memory_id, old, new, event, updated_at) in enumerate(rows):
        connection.execute(
            'INSERT INTO history VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL)',
            (f'h{index}', memory_id, old, new, event, '2026-10-01T00:00:00+00:00', updated_at),
        )
    connection.commit()
    connection.close()
    return path


def _without(row: dict[str, Any], field: str) -> dict[str, Any]:
    return {name: value for name, value in row.items() if name != field}


EARLY = 'Early detail sits here.'
LATE = 'Late detail sits here.'
LONG_CHILD = f'{EARLY} {"x" * 300} {LATE}'


def _with_spans(row: dict[str, Any], *spans: str) -> dict[str, Any]:
    return {**row, 'loss_spans': list(spans)}


class TestAssemble:
    """Ten UNRELATED pairs and one RELATED pair hold the HOME NO and CONFLICT NO rows.

    r10's raters both say HOME YES against its held row, s0's SAME row fails the
    check, and x0 is an extension item v1 never labelled.
    """

    def _adjudications(self) -> dict[tuple[str, str], dict[str, Any] | None]:
        adjudications: dict[tuple[str, str], dict[str, Any] | None] = {
            (f'r{index}', 'p1'): _adjudication('UNRELATED', child_memory_id=f'r{index}')
            for index in range(10)
        }
        adjudications['r10', 'p1'] = _adjudication('RELATED', child_memory_id='r10')
        adjudications['s0', 'p1'] = _adjudication('SAME', child_memory_id='s0')
        adjudications['x0', 'p1'] = None
        return adjudications

    def _rated_rows(self) -> list[dict[str, Any]]:
        rows = [
            _rated(f'r{index}', role=role, home='NO', conflict='NO')
            for index in range(10) for role in ('primary', 'second')
        ]
        rows += [
            _with_spans(_rated('r10', role='primary', home='YES', conflict='NO', loss='YES'), EARLY),
            _with_spans(_rated('r10', role='second', home='YES', conflict='NO', loss='YES'),
                        EARLY, LATE),
            _rated('s0', role='primary', home='NO', conflict='NO'),
            _rated('s0', role='second', home='NO', conflict='UNCLEAR'),
            _rated('x0', role='primary', home='YES', conflict='YES', scope='CENTRAL',
                   action_changing='YES'),
            _rated('x0', role='second', home='YES', conflict='YES', scope='SECONDARY',
                   action_changing='NO'),
            _with_spans(_rated('r0', role='primary', basis='attach_time', home='YES',
                               conflict='NO', loss='YES'), EARLY),
            _rated('r0', role='second', basis='attach_time', home='YES', conflict='NO', loss='NO'),
        ]
        return rows

    def _assembly(self) -> Any:
        child_texts = {entry_id: LONG_CHILD for entry_id, _ in self._adjudications()}
        return _mod().assemble(self._adjudications(), self._rated_rows(), child_texts)

    def _rows(self) -> dict[tuple[str, str, str], dict[str, Any]]:
        return {
            (row['entry_id'], row['basis'], row['role']): row for row in self._assembly().verdicts
        }

    def _judgement(self, entry_id: str) -> Any:
        rubric = _rubric()
        judgements = rubric.resolve_judgements(self._assembly().verdicts)
        return judgements[rubric.PairKey(entry_id, 'p1', rubric.Basis.CURRENT)]

    def test_one_derived_row_per_adjudicated_pair_from_the_held_rows(self) -> None:
        derived = {
            row['entry_id']: {key: row[key] for key in ('home', 'loss', 'conflict', 'scope')
                              if key in row}
            for row in self._assembly().verdicts if row['role'] == 'derived'
        }
        assert derived == {
            **{f'r{index}': {'home': 'NO', 'conflict': 'NO'} for index in range(10)},
            'r10': {'home': 'NO'}, 's0': {'conflict': 'NO'},
        }

    def test_every_rated_row_is_kept_whole(self) -> None:
        rows = {
            (row['entry_id'], row['basis'], row['rater']): row for row in self._assembly().verdicts
        }
        for rated in self._rated_rows():
            kept = rows[rated['entry_id'], rated['basis'], rated['rater']]
            assert {key: kept[key] for key in rated} == rated
            assert set(kept) - set(rated) <= {'non_consumed', 'digest'}

    def test_answers_a_derived_row_fixes_are_marked_non_consumed(self) -> None:
        rows = self._rows()
        assert rows['r0', 'current', 'primary']['non_consumed'] == ['conflict', 'home']
        assert rows['r10', 'current', 'second']['non_consumed'] == ['home']
        assert rows['s0', 'current', 'second']['non_consumed'] == ['conflict']
        assert 'non_consumed' not in rows['x0', 'current', 'primary']
        assert 'non_consumed' not in rows['r0', 'attach_time', 'primary']
        assert 'non_consumed' not in rows['r0', 'attach_time', 'second']

    def test_digest_is_computed_from_the_loss_spans(self) -> None:
        rows = self._rows()
        assert rows['r10', 'current', 'primary']['digest'] == 'YES'
        assert rows['r10', 'current', 'second']['digest'] == 'NO'
        assert rows['r0', 'attach_time', 'primary']['digest'] == 'YES'
        assert [row for row in self._assembly().verdicts
                if 'digest' in row and row.get('loss') != 'YES'] == []
        assert 'digest' not in rows['r0', 'attach_time', 'second']

    def test_the_corpus_is_a_valid_verdict_corpus_in_order(self) -> None:
        verdicts = self._assembly().verdicts
        _rubric().resolve_judgements(verdicts)
        order = [(row['entry_id'], row['target_id'], row['basis'], row['rater']) for row in verdicts]
        assert order == sorted(order)

    def test_a_held_row_outvotes_both_raters(self) -> None:
        assert self._judgement('r10').answer(_rubric().Binary.HOME) == 'NO'

    def test_a_pair_whose_row_failed_the_check_resolves_from_the_raters(self) -> None:
        judgement = self._judgement('s0')
        assert judgement.answer(_rubric().Binary.HOME) == 'NO'
        assert judgement.answer(_rubric().Binary.CONFLICT) == 'NO'
        assert self._judgement('x0').answer(_rubric().Binary.CONFLICT) == 'YES'

    def test_the_derivation_reports_each_row_check_and_the_held_rows(self) -> None:
        derivation = self._assembly().derivation
        assert derivation['check'] == {
            'min_compared': 20, 'min_agreement': 0.90, 'roles': ['primary', 'second'],
        }
        assert set(derivation['rows']) == set(CANDIDATE_ROWS) | set(RETIRED_ROWS)
        assert derivation['rows']['related_unrelated_home_no'] == {
            'binary': 'home', 'n_items': 11, 'n_compared': 22, 'agree': 20, 'held': True,
            'retired': False,
        }
        assert derivation['rows']['same_subsumed_unrelated_conflict_no'] == {
            'binary': 'conflict', 'n_items': 11, 'n_compared': 21, 'agree': 21, 'held': True,
            'retired': False,
        }
        assert derivation['rows']['same_subsumed_home_yes']['held'] is False
        assert derivation['held_rows'] == [
            'related_unrelated_home_no', 'same_subsumed_unrelated_conflict_no',
        ]

    def test_the_derivation_names_each_follow_up_a_held_row_left_unreached(self) -> None:
        assert self._assembly().derivation['unreached_follow_ups'] == []
        adjudications = {**self._adjudications(),
                         ('s1', 'p1'): _adjudication('SAME', child_memory_id='s1')}
        rated = self._rated_rows() + [
            _rated('s1', role=role, home='YES', conflict='YES', scope='CENTRAL',
                   action_changing='YES')
            for role in ('primary', 'second')
        ]
        child_texts = {entry_id: LONG_CHILD for entry_id, _ in adjudications}
        derivation = _mod().assemble(adjudications, rated, child_texts).derivation
        assert 'same_subsumed_unrelated_conflict_no' in derivation['held_rows']
        assert derivation['unreached_follow_ups'] == [
            {'entry_id': 's1', 'target_id': 'p1', 'basis': 'current', 'binary': 'loss'},
        ]

    def test_held_rows_are_decided_by_the_checked_rows(self) -> None:
        adjudications: dict[tuple[str, str], dict[str, Any] | None] = {
            (f'u{index}', 'p1'): _adjudication('UNRELATED', child_memory_id=f'u{index}')
            for index in range(10)
        }
        rated = [_rated(f'u{index}', role=role, home='YES', conflict='NO', loss='NO')
                 for index in range(10) for role in ('primary', 'second')]
        checked = [_rated(f'u{index}', role=role, home='NO', conflict='NO')
                   for index in range(10) for role in ('primary', 'second')]
        child_texts = {entry_id: LONG_CHILD for entry_id, _ in adjudications}
        builder = _mod()
        assembly = builder.assemble(adjudications, rated, child_texts, checked_rows=checked)

        derivation = assembly.derivation
        assert derivation['held_rows'] == [
            'related_unrelated_home_no', 'same_subsumed_unrelated_conflict_no']
        derived = {row['entry_id']: (row['home'], row['conflict'])
                   for row in assembly.verdicts if row['role'] == 'derived'}
        assert derived == {f'u{index}': ('NO', 'NO') for index in range(10)}
        assert {tuple(row['non_consumed']) for row in assembly.verdicts
                if row['role'] != 'derived'} == {('conflict', 'home')}
        home_no = {'binary': 'home', 'n_items': 10, 'n_compared': 20, 'retired': False}
        assert derivation['rows']['related_unrelated_home_no'] == {
            **home_no, 'agree': 20, 'held': True}
        assert derivation['effective_lines_check']['related_unrelated_home_no'] == {
            **home_no, 'agree': 0, 'held': False}
        assert set(derivation['effective_lines_check']) == set(derivation['rows'])

        unchecked = builder.assemble(adjudications, rated, child_texts).derivation
        assert unchecked['held_rows'] == ['same_subsumed_unrelated_conflict_no']
        assert unchecked['rows'] == unchecked['effective_lines_check']


CREATED = '2026-10-10T00:00:00+00:00'


def _attach(
    intent: str, outcome: str, *, triage_sig: bool = True, category: str = 'procedural_knowledge',
    **meta: Any,
) -> dict[str, Any]:
    """An attaches_soak.json row: the child's text, and its parent and supersessions in ``meta``."""
    return {
        'intent_id': intent, 'write_op_id': f'op-{intent}', 'project': 'df',
        'agent': 'claude-test', 'category': category, 'content': f'child text of {intent}',
        'meta': {'parent_id': f'p-{intent}', 'kind': 'amendment', 'category': category, **meta},
        'created_at': CREATED, 'source': 'mcp_tool', 'op': 'add_memory', 'outcome': outcome,
        'triage_sig': triage_sig,
    }


def _item_row(attach: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    flags: dict[str, Any] = {
        'child_memory_id': 'c1', 'high_band': False, 'via_sibling': False,
        'parent_category': 'procedural_knowledge', 'parent_edited_after_attach': False,
    }
    return _mod().item_row(attach, **{**flags, **overrides})


class TestItemRow:
    @pytest.mark.parametrize(('high_band', 'band'), [(True, 'restated'), (False, 'judge'),
                                                     (None, None)])
    def test_band_comes_from_the_high_band_flag(self, high_band: bool | None, band: str | None) -> None:
        assert _item_row(_attach('i1', 'restated'), high_band=high_band)['band'] == band

    def test_carries_the_scorer_contract(self) -> None:
        attach = _attach('i1', 'amended', parent_id='p1', supersedes=['p1', 'p9'])
        assert _item_row(
            attach, via_sibling=True, parent_category='observations_and_summaries',
            parent_edited_after_attach=True,
        ) == {
            'entry_id': 'c1', 'outcome': 'amended', 'target_id': 'p1', 'earlier_ids': [],
            'band': 'judge', 'project': 'df', 'stratum': 'amended-sample',
            'via_sibling': True, 'cross_category': True, 'parent_edited_after_attach': True,
            'supersedes_target': True, 'superseded_marks': [],
        }

    @pytest.mark.parametrize(('outcome', 'stratum'), [
        ('restated', 'restated-sample'), ('amended', 'amended-sample'),
        ('contested', 'contested-census'),
    ])
    def test_a_stratum_per_outcome(self, outcome: str, stratum: str) -> None:
        row = _item_row(_attach('i1', outcome), parent_category=None, via_sibling=None)
        assert row['stratum'] == stratum
        assert row['via_sibling'] is None
        assert row['cross_category'] is None
        assert row['supersedes_target'] is False

    def test_the_strata_are_the_soak_populations(self) -> None:
        assert _mod().STRATA == {
            'restated-sample': 125, 'amended-sample': 1224, 'contested-census': 473,
        }


POPULATION = frozenset(f'i{index:02d}' for index in range(30))
SAMPLED = frozenset({'i00', 'i01', 'i02'})


class TestExtendSample:
    def _draw(self, total: int, *, sampled: frozenset[str] = SAMPLED, seed: str = 's') -> list[str]:
        return _mod().extend_sample(POPULATION, sampled, total, seed=seed)

    def test_draws_new_ids_up_to_the_total(self) -> None:
        drawn = self._draw(10)
        assert len(drawn) == 7
        assert len(set(drawn)) == 7
        assert set(drawn) <= POPULATION - SAMPLED

    def test_counts_only_sampled_ids_inside_the_population(self) -> None:
        assert len(self._draw(5, sampled=frozenset({'i00', 'elsewhere'}))) == 4

    def test_the_same_seed_gives_the_same_draw(self) -> None:
        assert self._draw(10) == self._draw(10)
        assert self._draw(10) != self._draw(10, seed='other')

    def test_a_larger_total_extends_the_smaller_draw(self) -> None:
        assert self._draw(15)[:7] == self._draw(10)

    @pytest.mark.parametrize('total', [3, 2, 0])
    def test_a_total_already_sampled_draws_nothing(self, total: int) -> None:
        assert self._draw(total) == []

    def test_a_total_beyond_the_population_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r'31.*30|30.*31'):
            self._draw(31)


ADJUDICATED = {'i-r0': 'restated', 'i-c0': 'contested'}
RESTATED = ['i-r1', 'i-r2', 'i-r3', 'i-r4']
AMENDED = ['i-a1', 'i-a2', 'i-a3', 'i-a4']
NOT_TRIAGED = 'i-x'
ATTACH_TIME_PARENT = 'the parent as it read when the child was written'


def _drawn(outcome: str, total: int) -> list[str]:
    """The extension the builder must draw: ``extend_sample`` seeded with ``'<seed>:<outcome>'``."""
    population = {'restated': {'i-r0', *RESTATED}, 'amended': set(AMENDED)}[outcome]
    return _mod().extend_sample(population, set(ADJUDICATED), total, seed=f'7:{outcome}')


BRIEF_SHA256 = hashlib.sha256(
    (SCRIPTS.parent / 'calibration' / 'write_triage_rater_brief_v2.md').read_bytes()).hexdigest()
"""The committed brief's sha256, which every shard a default run writes records."""


class _Bundle:
    """A synthetic soak bundle: two adjudicated attaches and eight that may extend the sample.

    The first drawn restated extension item's parent was edited after attach,
    and the first drawn amended one has no write op in the journal.
    """

    def __init__(self, root: Path) -> None:
        self.edited = _drawn('restated', 3)[0]
        self.missing = _drawn('amended', 3)[0]
        self.dir = root / 'bundle'
        self.dir.mkdir()
        attaches = [_attach(intent, outcome) for intent, outcome in ADJUDICATED.items()]
        attaches += [_attach(intent, 'restated') for intent in RESTATED]
        attaches += [_attach(intent, 'amended') for intent in AMENDED]
        attaches.append(_attach(NOT_TRIAGED, 'restated', triage_sig=False))
        adjudications = [
            _adjudication('SAME', outcome='restated', **self._identity('i-r0')),
            _adjudication('CORRECTS', dispute='genuine_correction', action_changing=True,
                          **self._identity('i-c0')),
        ]
        (self.dir / 'adjudications.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in adjudications))
        self._json('attaches_soak.json', attaches)
        self._json('parents.json', {
            f'p-{row["intent_id"]}': [{'collection': 'fused_df', 'payload': {
                'data': f'parent text of {row["intent_id"]}', 'category': 'procedural_knowledge',
            }}] for row in attaches
        })
        self._json('hoist.json', {
            row['intent_id']: {'high_band': row['outcome'] == 'restated', 'likely_via_sibling': False}
            for row in attaches
        })
        self._json('parent_edited_after_attach.json', [self.edited])
        self.journal = self._journal(root / 'journal.db', [
            row for row in attaches if row['intent_id'] != self.missing
        ])
        self.history = _history_db(root / 'history.db', [
            (f'p-{self.edited}', ATTACH_TIME_PARENT, f'parent text of {self.edited}', 'UPDATE',
             '2026-10-12T00:00:00+00:00'),
        ])
        self.scratch = root / 'scratch'

    @staticmethod
    def _identity(intent: str) -> dict[str, Any]:
        return {'child_intent_id': intent, 'child_memory_id': f'c-{intent}',
                'parent_id': f'p-{intent}', 'created_at': CREATED}

    def _json(self, name: str, value: Any) -> None:
        (self.dir / name).write_text(json.dumps(value))

    @staticmethod
    def _journal(path: Path, attaches: list[dict[str, Any]]) -> Path:
        connection = sqlite3.connect(path)
        connection.execute('CREATE TABLE write_ops (id TEXT PRIMARY KEY, result_summary TEXT)')
        for row in attaches:
            connection.execute('INSERT INTO write_ops VALUES (?, ?)', (
                row['write_op_id'], json.dumps({'memory_ids': [f'c-{row["intent_id"]}'],
                                                'stores': ['mem0']}),
            ))
        connection.commit()
        connection.close()
        return path

    def shards(self, *extra: str, restated: int = 3, amended: int = 3) -> int:
        return _mod().main([
            'shards', '--bundle', str(self.dir), '--history-db', str(self.history),
            '--journal-db', str(self.journal), '--scratch', str(self.scratch), '--seed', '7',
            '--restated-total', str(restated), '--amended-total', str(amended), *extra,
        ])

    def key(self) -> dict[str, Any]:
        return json.loads((self.scratch / 'key.json').read_text())

    def shard_files(self) -> dict[str, bytes]:
        return {path.name: path.read_bytes() for path in (self.scratch / 'shards').iterdir()}

    def run(self, command: str, *extra: str) -> int:
        return _mod().main([command, '--bundle', str(self.dir), '--scratch', str(self.scratch),
                            *extra])


def _triple(item: dict[str, str]) -> tuple[str, str, str]:
    return (item['entry_id'], item['target_id'], item['basis'])


def _keyed(key: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {_triple(item) for item in key['items'].values()}


def _shard_keyed(key: dict[str, Any], name: str) -> set[tuple[str, str, str]]:
    return {_triple(key['items'][item]) for item in key['shards'][name]['items']}


def _shards_holding(key: dict[str, Any], keyed: tuple[str, str, str]) -> list[str]:
    return sorted(name for name in key['shards'] if keyed in _shard_keyed(key, name))


def _current(intent: str) -> tuple[str, str, str]:
    return (f'c-{intent}', f'p-{intent}', 'current')


class TestTheShardsCommand:
    @pytest.fixture
    def bundle(self, tmp_path: Path) -> _Bundle:
        bundle = _Bundle(tmp_path)
        assert bundle.shards() == 0
        return bundle

    def test_records_each_drawn_extension_item(self, bundle: _Bundle) -> None:
        assert bundle.key()['extension'] == {
            **{intent: {'outcome': 'restated', 'child_memory_id': f'c-{intent}'}
               for intent in _drawn('restated', 3)},
            **{intent: {'outcome': 'amended',
                        'child_memory_id': None if intent == bundle.missing else f'c-{intent}'}
               for intent in _drawn('amended', 3)},
        }
        assert NOT_TRIAGED not in bundle.key()['extension']

    def test_an_unrecoverable_item_is_recorded_and_not_replaced(self, bundle: _Bundle) -> None:
        key = bundle.key()
        assert [row['intent_id'] for row in key['unrecoverable']] == [bundle.missing]
        assert key['unrecoverable'][0]['reason']
        keyed_targets = {target for _, target, _ in _keyed(key)}
        undrawn = set(AMENDED) - set(_drawn('amended', 3))
        assert {f'p-{intent}' for intent in {bundle.missing} | undrawn} & keyed_targets == set()

    def test_every_pair_and_basis_is_in_one_primary_and_one_second_shard(
        self, bundle: _Bundle,
    ) -> None:
        key = bundle.key()
        recovered = [*ADJUDICATED, *_drawn('restated', 3),
                     *(intent for intent in _drawn('amended', 3) if intent != bundle.missing)]
        attach_time = (f'c-{bundle.edited}', f'p-{bundle.edited}', 'attach_time')
        assert _keyed(key) == {_current(intent) for intent in recovered} | {attach_time}
        for keyed in _keyed(key):
            roles = sorted(key['shards'][name]['role'] for name in _shards_holding(key, keyed))
            assert roles == ['primary', 'second'], keyed
        for spec in key['shards'].values():
            assert spec == {'role': spec['role'], 'rater': f'opus-{spec["role"]}', 'round': 0,
                            'items': spec['items'], 'brief_sha256': BRIEF_SHA256}

    def test_attach_time_pairs_sit_only_in_attach_shards(self, bundle: _Bundle) -> None:
        key = bundle.key()
        for keyed in _keyed(key):
            attach_shards = [name.startswith('attach_') for name in _shards_holding(key, keyed)]
            assert attach_shards == [keyed[2] == 'attach_time'] * 2, keyed

    def test_texts_hold_each_pair_and_basis_as_its_raters_saw_it(self, bundle: _Bundle) -> None:
        texts = json.loads((bundle.scratch / 'texts.json').read_text())
        by_keyed = {(row['entry_id'], row['target_id'], row['basis']): row for row in texts}
        assert len(texts) == len(by_keyed)
        assert set(by_keyed) == _keyed(bundle.key())
        assert by_keyed[_current('i-c0')] == {
            'entry_id': 'c-i-c0', 'target_id': 'p-i-c0', 'basis': 'current',
            'child_text': 'child text of i-c0', 'parent_text': 'parent text of i-c0',
        }
        edited = by_keyed[f'c-{bundle.edited}', f'p-{bundle.edited}', 'attach_time']
        assert edited['parent_text'] == ATTACH_TIME_PARENT
        assert edited['child_text'] == f'child text of {bundle.edited}'

    def test_a_rerun_writes_nothing_new(self, bundle: _Bundle) -> None:
        shard_files, key_bytes = bundle.shard_files(), (bundle.scratch / 'key.json').read_bytes()
        assert bundle.shards() == 0
        assert bundle.shard_files() == shard_files
        assert (bundle.scratch / 'key.json').read_bytes() == key_bytes

    def test_a_larger_total_adds_shards_only_for_the_new_pairs(self, bundle: _Bundle) -> None:
        shard_files, earlier = bundle.shard_files(), bundle.key()
        assert bundle.shards(amended=4) == 0
        key = bundle.key()
        added = _drawn('amended', 4)[3]
        assert set(key['extension']) == set(earlier['extension']) | {added}
        files = bundle.shard_files()
        assert {name: files[name] for name in shard_files} == shard_files
        assert set(files) - set(shard_files) == {'primary_01.txt', 'second_01.txt'}
        for name in ('primary_01', 'second_01'):
            assert _shard_keyed(key, name) == {_current(added)}

    def test_a_smaller_total_than_already_drawn_is_refused(self, bundle: _Bundle) -> None:
        key_bytes = (bundle.scratch / 'key.json').read_bytes()
        with pytest.raises(ValueError, match='extension'):
            bundle.shards(amended=2)
        assert (bundle.scratch / 'key.json').read_bytes() == key_bytes

    def test_shards_renders_and_records_the_brief_it_is_given(self, tmp_path: Path) -> None:
        old_brief = tmp_path / 'brief_v2_1.md'
        old_brief.write_text('# Rater brief v2.1\n\nThe older LOSS wording.\n')
        built = _Bundle(tmp_path)
        assert built.shards('--brief', str(old_brief)) == 0
        sha = hashlib.sha256(old_brief.read_bytes()).hexdigest()
        assert {spec['brief_sha256'] for spec in built.key()['shards'].values()} == {sha}
        assert (built.scratch / 'rater_prompt.md').read_text() == _rounds().rater_prompt(old_brief)

    def test_the_rater_prompt_asks_for_evidence_and_not_digest(self, bundle: _Bundle) -> None:
        prompt = (bundle.scratch / 'rater_prompt.md').read_text()
        brief = _rounds().BRIEF_PATH.read_text()
        assert brief in prompt
        output = prompt.split(brief, 1)[1]
        for field in ('conflict_quote', 'loss_spans', 'action_changing'):
            assert field in output
        assert 'digest' not in output.lower()


Answer = dict[str, Any]
Rule = Callable[[dict[str, str]], Answer]


def _conflict_yes(text: dict[str, str]) -> Answer:
    return {'home': 'YES', 'conflict': 'YES', 'scope': 'CENTRAL', 'action_changing': 'YES',
            'conflict_quote': {'parent': text['parent_text'], 'child': text['child_text']}}


def _loss_yes(text: dict[str, str]) -> Answer:
    return {'home': 'YES', 'conflict': 'NO', 'loss': 'YES', 'loss_spans': [text['child_text']]}


def _no_loss(_: dict[str, str]) -> Answer:
    return {'home': 'YES', 'conflict': 'NO', 'loss': 'NO'}


def _misfiled(_: dict[str, str]) -> Answer:
    return {'home': 'NO', 'conflict': 'NO'}


def _conflict_unclear(_: dict[str, str]) -> Answer:
    return {'home': 'YES', 'conflict': 'UNCLEAR', 'unclear_reason': 'cut where the claims meet'}


def _home_unclear(_: dict[str, str]) -> Answer:
    return {'home': 'UNCLEAR', 'conflict': 'NO', 'unclear_reason': 'cut at the subject'}


def _lacks_loss(_: dict[str, str]) -> Answer:
    return {'home': 'YES', 'conflict': 'NO'}


def _conflict_unquoted(text: dict[str, str]) -> Answer:
    return _without(_conflict_yes(text), 'conflict_quote')


def _loss_unspanned(_: dict[str, str]) -> Answer:
    return {'home': 'YES', 'conflict': 'NO', 'loss': 'YES'}


HONEST: dict[str, Rule] = {'c-i-c0': _conflict_yes, 'c-i-r0': _loss_yes}
"""Each entry's answer when a test does not override it; any other entry is misfiled."""


def _rate(bundle: _Bundle, overrides: dict[tuple[str, tuple[str, str, str]], Rule] | None = None) -> None:
    """Write a fake rater's JSONL for each shard without one, quoting verbatim from texts.json.

    *overrides* maps a (role, (entry, target, basis)) to the rule its line follows.
    """
    key = bundle.key()
    texts = {_triple(row): row for row in json.loads((bundle.scratch / 'texts.json').read_text())}
    ratings = bundle.scratch / 'ratings'
    ratings.mkdir(exist_ok=True)
    for name, spec in key['shards'].items():
        path = ratings / f'{name}.jsonl'
        if path.exists():
            continue
        lines = []
        for item_id in spec['items']:
            keyed = _triple(key['items'][item_id])
            rule = (overrides or {}).get((spec['role'], keyed)) or HONEST.get(keyed[0], _misfiled)
            lines.append({'id': item_id, **rule(texts[keyed]), 'reason': 'names both claims'})
        path.write_text(''.join(json.dumps(line) + '\n' for line in lines))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _item_holding(key: dict[str, Any], shard_name: str, keyed: tuple[str, str, str]) -> str:
    [item_id] = [item for item in key['shards'][shard_name]['items']
                 if _triple(key['items'][item]) == keyed]
    return item_id


def _primary_shard_of(key: dict[str, Any], keyed: tuple[str, str, str]) -> str:
    [name] = [name for name in _shards_holding(key, keyed)
              if key['shards'][name]['role'] == 'primary']
    return name


CORPUS_FILES = ('items.jsonl', 'verdicts.jsonl', 'strata.json', 'derivation.json')


def _score(out: Path, *, verdicts: str = 'verdicts.jsonl', rubric: str = 'v2.1') -> int:
    """The scorer's exit code over the corpus ``assemble`` wrote into *out*, its verdicts
    read from ``write_triage_v2_soak_<verdicts>``."""
    score = load_script_module(SCRIPTS / 'score_write_triage_reader.py',
                               mod_name='score_write_triage_reader')
    paths = {name: out / f'write_triage_v2_soak_{name}' for name in ('items.jsonl', verdicts, 'strata.json')}
    return score.main([
        '--items', str(paths['items.jsonl']), '--verdicts', str(paths[verdicts]),
        '--strata', str(paths['strata.json']), '--rubric', rubric,
    ])


class TestTheRatingRounds:
    @pytest.fixture
    def bundle(self, tmp_path: Path) -> _Bundle:
        bundle = _Bundle(tmp_path)
        assert bundle.shards() == 0
        return bundle

    @pytest.fixture
    def out(self, tmp_path: Path) -> Path:
        return tmp_path / 'out'

    def _assemble(self, bundle: _Bundle, out: Path) -> int:
        return bundle.run('assemble', '--out-dir', str(out))

    def test_assemble_refuses_a_shard_without_ratings(self, bundle: _Bundle, out: Path) -> None:
        _rate(bundle)
        (bundle.scratch / 'ratings' / 'second_00.jsonl').unlink()
        with pytest.raises(ValueError, match='second_00'):
            self._assemble(bundle, out)
        assert not out.exists()

    def test_assemble_refuses_an_item_without_its_line(self, bundle: _Bundle, out: Path) -> None:
        _rate(bundle)
        key = bundle.key()
        item_id = _item_holding(key, 'second_00', _current('i-c0'))
        path = bundle.scratch / 'ratings' / 'second_00.jsonl'
        path.write_text(''.join(json.dumps(line) + '\n' for line in _jsonl(path)
                                if line['id'] != item_id))
        with pytest.raises(ValueError, match=rf'second_00.*{item_id}'):
            self._assemble(bundle, out)

    def test_an_unfound_quote_is_reasked_and_the_new_line_supersedes_it(
        self, bundle: _Bundle, out: Path,
    ) -> None:
        keyed = _current('i-c0')

        def unfound(text: dict[str, str]) -> Answer:
            answer = _conflict_yes(text)
            return {**answer, 'conflict_quote': {**answer['conflict_quote'], 'child': 'not in it'}}

        _rate(bundle, {('primary', keyed): unfound})
        key = bundle.key()
        item_id = _item_holding(key, _primary_shard_of(key, keyed), keyed)
        with pytest.raises(ValueError, match=rf'{item_id}.*conflict_quote\.child|conflict_quote\.child.*{item_id}'):
            self._assemble(bundle, out)

        assert bundle.run('reask') == 0
        key = bundle.key()
        reasks = {name: spec for name, spec in key['shards'].items() if name.startswith('reask_')}
        assert list(reasks) == ['reask_01_primary']
        spec = reasks['reask_01_primary']
        assert (spec['role'], spec['rater'], spec['round'], len(spec['items'])) == (
            'primary', 'opus-primary', 1, 1,
        )
        assert _shard_keyed(key, 'reask_01_primary') == {keyed}

        _rate(bundle)
        shard_files = bundle.shard_files()
        assert bundle.run('reask') == 0
        assert bundle.shard_files() == shard_files
        assert self._assemble(bundle, out) == 0
        [verdict] = [row for row in _jsonl(out / 'write_triage_v2_soak_verdicts.jsonl')
                     if _triple(row) == keyed and row['rater'] == 'opus-primary']
        assert verdict['batch'] == 'reask_01_primary'
        assert verdict['conflict_quote'] == {'parent': 'parent text of i-c0',
                                             'child': 'child text of i-c0'}
        assert json.loads((out / 'write_triage_v2_soak_derivation.json').read_text())['reasked'] == 1

    @pytest.mark.parametrize(('broken', 'fragment'), [
        (_lacks_loss, 'lacks loss'), (_conflict_unquoted, 'conflict_quote'),
        (_loss_unspanned, 'loss_spans'),
    ])
    def test_a_line_breaking_the_row_contract_is_reasked_and_the_new_line_supersedes_it(
        self, bundle: _Bundle, out: Path, broken: Rule, fragment: str,
    ) -> None:
        keyed = _current('i-c0')
        _rate(bundle, {('primary', keyed): broken})
        key = bundle.key()
        item_id = _item_holding(key, _primary_shard_of(key, keyed), keyed)
        for refused in (lambda: bundle.run('third'), lambda: self._assemble(bundle, out)):
            with pytest.raises(ValueError) as refusal:
                refused()
            assert item_id in str(refusal.value) and fragment in str(refusal.value)
        assert not out.exists()

        assert bundle.run('reask') == 0
        key = bundle.key()
        reasks = {name: spec for name, spec in key['shards'].items() if name.startswith('reask_')}
        assert list(reasks) == ['reask_01_primary']
        spec = reasks['reask_01_primary']
        assert (spec['role'], spec['rater'], spec['round']) == ('primary', 'opus-primary', 1)
        assert _shard_keyed(key, 'reask_01_primary') == {keyed}

        _rate(bundle)
        shard_files = bundle.shard_files()
        assert bundle.run('reask') == 0
        assert bundle.shard_files() == shard_files
        assert self._assemble(bundle, out) == 0
        [verdict] = [row for row in _jsonl(out / 'write_triage_v2_soak_verdicts.jsonl')
                     if _triple(row) == keyed and row['rater'] == 'opus-primary']
        assert verdict['batch'] == 'reask_01_primary'
        assert json.loads((out / 'write_triage_v2_soak_derivation.json').read_text())['reasked'] == 1

    def test_third_rounds_name_a_fresh_rater_until_no_decisive_tie_is_left(
        self, bundle: _Bundle, out: Path,
    ) -> None:
        tied = _current(_drawn('restated', 3)[1])
        overrides = {('primary', tied): _conflict_yes, ('second', tied): _loss_yes,
                     ('third', tied): _no_loss}
        _rate(bundle, overrides)
        for index in range(2):
            assert bundle.run('third') == 0
            key = bundle.key()
            name = f'third_{index:02d}'
            assert {**key['shards'][name], 'items': len(key['shards'][name]['items'])} == {
                'role': 'third', 'rater': f'opus-third-{index:02d}', 'round': 0, 'items': 1,
                'brief_sha256': BRIEF_SHA256,
            }
            assert _shard_keyed(key, name) == {tied}
            _rate(bundle, overrides)
        shard_files = bundle.shard_files()
        assert bundle.run('third') == 0
        assert bundle.shard_files() == shard_files
        assert self._assemble(bundle, out) == 0

    def test_an_open_follow_up_goes_to_a_third_round_and_closes(
        self, bundle: _Bundle, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        open_pair = _current(_drawn('restated', 3)[1])
        overrides = {('primary', open_pair): _conflict_unclear,
                     ('second', open_pair): _home_unclear, ('third', open_pair): _no_loss}
        _rate(bundle, overrides)
        before, after = tmp_path / 'before', tmp_path / 'after'
        assert self._assemble(bundle, before) == 0
        capsys.readouterr()
        assert _score(before) == 1
        assert capsys.readouterr().err.splitlines()[1:] == [
            f'open {open_pair[0]} {open_pair[1]} loss',
        ]

        assert bundle.run('third') == 0
        key = bundle.key()
        assert {**key['shards']['third_00'], 'items': len(key['shards']['third_00']['items'])} == {
            'role': 'third', 'rater': 'opus-third-00', 'round': 0, 'items': 1,
            'brief_sha256': BRIEF_SHA256,
        }
        assert _shard_keyed(key, 'third_00') == {open_pair}

        _rate(bundle, overrides)
        shard_files = bundle.shard_files()
        assert bundle.run('third') == 0
        assert bundle.shard_files() == shard_files
        assert self._assemble(bundle, after) == 0
        assert _score(after) == 0
        [third] = [row for row in _jsonl(after / 'write_triage_v2_soak_verdicts.jsonl')
                   if _triple(row) == open_pair and row['rater'] == 'opus-third-00']
        assert third['loss'] == 'NO'

    def test_assemble_writes_the_corpus_the_scorer_reads(self, bundle: _Bundle, out: Path) -> None:
        _rate(bundle)
        assert self._assemble(bundle, out) == 0
        paths = {name: out / f'write_triage_v2_soak_{name}' for name in CORPUS_FILES}
        assert sorted(path.name for path in out.iterdir()) == sorted(
            path.name for path in paths.values())
        recovered = [*ADJUDICATED, *_drawn('restated', 3),
                     *(intent for intent in _drawn('amended', 3) if intent != bundle.missing)]
        assert sorted(row['entry_id'] for row in _jsonl(paths['items.jsonl'])) == sorted(
            f'c-{intent}' for intent in recovered)

        key = bundle.key()
        verdicts = _jsonl(paths['verdicts.jsonl'])
        _rubric().resolve_judgements(verdicts)
        assert sorted((*_triple(row), row['role']) for row in verdicts) == sorted(
            (*keyed, role) for keyed in _keyed(key) for role in ('primary', 'second'))
        loss_rows = [row for row in verdicts if row.get('loss') == 'YES']
        assert loss_rows and all(row['digest'] == 'YES' for row in loss_rows)
        assert [row for row in verdicts if 'non_consumed' in row] == []

        assert json.loads(paths['strata.json'].read_text()) == dict(_mod().STRATA)
        derivation = json.loads(paths['derivation.json'].read_text())
        assert set(derivation) == {
            'check', 'rows', 'held_rows', 'unreached_follow_ups', 'rating_volume', 'sample',
            'unrecoverable', 'attach_time_unrecoverable', 'reasked', 'shards', 'rerates',
            'effective_lines_check',
        }
        assert derivation['check']['brief_sha256'] == BRIEF_SHA256
        assert derivation['rerates'] == []
        assert derivation['effective_lines_check'] == derivation['rows']
        assert derivation['held_rows'] == []
        assert derivation['unreached_follow_ups'] == []
        assert derivation['rating_volume'] == {
            'current': {'primary': 6, 'second': 6}, 'attach_time': {'primary': 1, 'second': 1},
        }
        assert derivation['sample'] == {
            'restated-sample': {'adjudicated': 1, 'extension': 2, 'unrecoverable': 0},
            'amended-sample': {'adjudicated': 0, 'extension': 2, 'unrecoverable': 1},
            'contested-census': {'adjudicated': 1, 'extension': 0, 'unrecoverable': 0},
        }
        assert derivation['unrecoverable'] == key['unrecoverable']
        assert derivation['attach_time_unrecoverable'] == []
        assert derivation['reasked'] == 0
        assert derivation['shards'] == {
            name: {'rater': spec['rater'], 'role': spec['role'], 'round': spec['round'],
                   'items': len(spec['items']), 'brief_sha256': BRIEF_SHA256}
            for name, spec in key['shards'].items()
        }

        assert _score(out) == 0


RERATED = _current('i-r0')
"""The one pair whose derivation reads LOSS when every line follows HONEST."""
V2_2_CORPUS = ('items.jsonl', 'strata.json', 'verdicts_v2_2.jsonl', 'derivation_v2_2.json')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestTheRerate:
    @pytest.fixture
    def bundle(self, tmp_path: Path) -> _Bundle:
        bundle = _Bundle(tmp_path)
        assert bundle.shards() == 0
        return bundle

    @pytest.fixture
    def brief(self, tmp_path: Path) -> Path:
        path = tmp_path / 'brief_v2_2.md'
        path.write_text('# Rater brief v2.2 (test copy)\n\nAnswer LOSS the revised way.\n')
        assert _sha(path) != BRIEF_SHA256
        return path

    @pytest.fixture
    def rerated(self, bundle: _Bundle, brief: Path) -> _Bundle:
        _rate(bundle)
        assert self._rerate(bundle, brief) == 0
        return bundle

    @staticmethod
    def _rerate(bundle: _Bundle, brief: Path) -> int:
        return bundle.run('rerate', '--brief', str(brief), '--brief-version', 'v2.2')

    @staticmethod
    def _assemble(bundle: _Bundle, out: Path, brief: Path) -> int:
        return bundle.run('assemble', '--out-dir', str(out), '--brief', str(brief))

    def test_keys_one_round_of_both_raters_over_the_pairs_reading_loss(
        self, rerated: _Bundle, brief: Path,
    ) -> None:
        key = rerated.key()
        [rerate] = key['rerates']
        assert rerate == {
            'version': 'v2.2', 'round': 1, 'brief_sha256': _sha(brief), 'pairs': [list(RERATED[:2])],
            'shards': ['rerate_01_primary_00', 'rerate_01_second_00'], 'retires_third_raters': [],
        }
        for name in rerate['shards']:
            spec = key['shards'][name]
            assert (spec['rater'], spec['round'], spec['brief_sha256']) == (
                f'opus-{spec["role"]}', 1, _sha(brief))
            assert _shard_keyed(key, name) == {RERATED}
        assert (rerated.scratch / 'rater_prompt.md').read_text() == _rounds().rater_prompt(brief)

    def test_the_rerated_pairs_are_those_the_assembled_corpus_reads_loss_on(
        self, bundle: _Bundle, brief: Path, tmp_path: Path,
    ) -> None:
        _rate(bundle)
        before = tmp_path / 'before'
        assert bundle.run('assemble', '--out-dir', str(before)) == 0
        derivations = _mod().current_derivations(
            _jsonl(before / 'write_triage_v2_soak_items.jsonl'),
            _jsonl(before / 'write_triage_v2_soak_verdicts.jsonl'))
        assert _rounds().pairs_reading_loss(derivations) == {RERATED[:2]}
        assert self._rerate(bundle, brief) == 0
        assert bundle.key()['rerates'][0]['pairs'] == [list(RERATED[:2])]

    def test_refuses_while_a_keyed_shard_has_no_ratings(self, bundle: _Bundle, brief: Path) -> None:
        with pytest.raises(ValueError, match='no ratings'):
            self._rerate(bundle, brief)
        assert 'rerates' not in bundle.key()

    def test_refuses_while_a_reask_is_pending(self, bundle: _Bundle, brief: Path) -> None:
        _rate(bundle, {('primary', RERATED): _lacks_loss})
        with pytest.raises(ValueError, match='reask'):
            self._rerate(bundle, brief)
        assert 'rerates' not in bundle.key()

    def test_refuses_while_a_pair_awaits_a_third_rater(self, bundle: _Bundle, brief: Path) -> None:
        _rate(bundle, {('primary', RERATED): _conflict_yes})
        with pytest.raises(ValueError, match=r"run third.*\('c-i-r0', 'p-i-r0'\)"):
            self._rerate(bundle, brief)
        assert 'rerates' not in bundle.key()

    def test_refuses_a_brief_version_that_is_not_a_v_numbered_label(
        self, bundle: _Bundle, brief: Path,
    ) -> None:
        _rate(bundle)
        with pytest.raises(ValueError, match=r"brief version label '\.\./v2\.2'"):
            bundle.run('rerate', '--brief', str(brief), '--brief-version', '../v2.2')
        assert 'rerates' not in bundle.key()

    def test_assemble_refuses_a_hand_edited_rerate_label_writing_nothing(
        self, rerated: _Bundle, brief: Path, tmp_path: Path,
    ) -> None:
        key = rerated.key()
        key['rerates'][0]['version'] = '../../escaped'
        (rerated.scratch / 'key.json').write_text(json.dumps(key))
        _rate(rerated)
        out = tmp_path / 'nested' / 'out'
        with pytest.raises(ValueError, match=r"brief version label '\.\./\.\./escaped'"):
            self._assemble(rerated, out, brief)
        assert not (tmp_path / 'nested').exists()

    def test_a_rerun_keys_nothing_new_and_renders_a_deleted_shard_again(
        self, rerated: _Bundle, brief: Path,
    ) -> None:
        key_bytes, shard_files = (rerated.scratch / 'key.json').read_bytes(), rerated.shard_files()
        (rerated.scratch / 'shards' / 'rerate_01_second_00.txt').unlink()
        assert self._rerate(rerated, brief) == 0
        assert (rerated.scratch / 'key.json').read_bytes() == key_bytes
        assert rerated.shard_files() == shard_files

    def test_assemble_writes_the_v2_2_verdicts_and_derivation_beside_the_base_items_and_strata(
        self, rerated: _Bundle, brief: Path, tmp_path: Path,
    ) -> None:
        _rate(rerated)
        out = tmp_path / 'out'
        assert self._assemble(rerated, out, brief) == 0
        assert sorted(path.name for path in out.iterdir()) == sorted(
            f'write_triage_v2_soak_{name}' for name in V2_2_CORPUS)

        key = rerated.key()
        [rerate] = key['rerates']
        verdicts = _jsonl(out / 'write_triage_v2_soak_verdicts_v2_2.jsonl')
        batches = {row['rater']: row['batch'] for row in verdicts if _triple(row) == RERATED}
        assert batches == {'opus-primary': 'rerate_01_primary_00', 'opus-second': 'rerate_01_second_00'}
        derivation = json.loads((out / 'write_triage_v2_soak_derivation_v2_2.json').read_text())
        assert derivation['rerates'] == [
            {'version': 'v2.2', 'round': 1, 'pairs': 1, 'brief_sha256': _sha(brief)}]
        assert {name: spec['brief_sha256'] for name, spec in derivation['shards'].items()} == {
            name: _sha(brief) if name in rerate['shards'] else BRIEF_SHA256 for name in key['shards']}
        assert derivation['check']['brief_sha256'] == BRIEF_SHA256
        assert _score(out, verdicts='verdicts_v2_2.jsonl', rubric='v2.2') == 0

    def test_assemble_refuses_a_pair_reading_loss_outside_the_latest_rerate(
        self, rerated: _Bundle, brief: Path, tmp_path: Path,
    ) -> None:
        key = rerated.key()
        rerate = key['rerates'][0]
        rerate['pairs'].remove(list(RERATED[:2]))
        for name in rerate['shards']:
            key['shards'][name]['items'] = []
        (rerated.scratch / 'key.json').write_text(json.dumps(key))
        _rate(rerated)
        out = tmp_path / 'out'
        with pytest.raises(ValueError, match=r'(?s)latest re-rate \(v2\.2\).*c-i-r0 p-i-r0'):
            self._assemble(rerated, out, brief)
        assert not out.exists()

    def test_a_tie_on_a_rerated_pair_goes_to_a_fresh_third_rater_under_the_new_brief(
        self, bundle: _Bundle, brief: Path, tmp_path: Path,
    ) -> None:
        tie = {('primary', RERATED): _conflict_yes, ('third', RERATED): _loss_yes}
        _rate(bundle, tie)
        assert bundle.run('third') == 0
        _rate(bundle, tie)
        assert self._rerate(bundle, brief) == 0
        [rerate] = bundle.key()['rerates']
        assert (rerate['pairs'], rerate['retires_third_raters']) == ([list(RERATED[:2])], ['opus-third-00'])

        _rate(bundle, tie)
        assert bundle.run('third', '--brief', str(brief)) == 0
        key = bundle.key()
        fresh = {name: spec for name, spec in key['shards'].items()
                 if spec['role'] == 'third' and spec['rater'] != 'opus-third-00'}
        assert [(spec['rater'], spec['brief_sha256']) for spec in fresh.values()] == [
            ('opus-third-01', _sha(brief))]
        assert {keyed for name in fresh for keyed in _shard_keyed(key, name)} == {RERATED}

        _rate(bundle, tie)
        out = tmp_path / 'out'
        assert self._assemble(bundle, out, brief) == 0
        assert {row['rater'] for row in _jsonl(out / 'write_triage_v2_soak_verdicts_v2_2.jsonl')
                if _triple(row) == RERATED} == {'opus-primary', 'opus-second', 'opus-third-01'}


class TestTheCli:
    def test_refuses_to_write_inside_the_bundle(self, tmp_path: Path) -> None:
        bundle = tmp_path / 'bundle'
        bundle.mkdir()
        with pytest.raises(SystemExit):
            _mod().main([
                'shards', '--bundle', str(bundle), '--history-db', str(tmp_path / 'h.db'),
                '--journal-db', str(tmp_path / 'j.db'), '--scratch', str(bundle / 'scratch'),
            ])
        assert list(bundle.iterdir()) == []


class TestThirdRoundPairs:
    def _vote(self, entry: str, rater: str, role: str, **answers: str) -> dict[str, Any]:
        return {**_rated(entry, role=role, **answers), 'rater': rater, 'batch': 'b'}

    def test_names_the_pairs_a_tie_or_an_open_follow_up_blocks(self) -> None:
        items = [_item(entry_id='tied'), _item(entry_id='calm'), _item(entry_id='harmless'),
                 _item(entry_id='open', outcome='restated')]
        verdicts = [
            self._vote('tied', 'opus', 'primary', conflict='YES'),
            self._vote('tied', 'sonnet', 'second', conflict='NO', home='YES', loss='NO'),
            self._vote('calm', 'opus', 'primary', conflict='YES'),
            self._vote('harmless', 'opus', 'primary', conflict='YES', home='YES'),
            self._vote('harmless', 'sonnet', 'second', conflict='YES', home='NO'),
            self._vote('open', 'opus', 'primary', home='YES', conflict='UNCLEAR'),
            self._vote('open', 'sonnet', 'second', home='UNCLEAR', conflict='NO'),
        ]
        assert _mod().third_round_pairs(items, verdicts) == [('open', 'p1'), ('tied', 'p1')]

    def test_an_unrated_decisive_binary_is_refused(self) -> None:
        with pytest.raises(ValueError, match="'unrated-item'"):
            _mod().third_round_pairs([_item(entry_id='unrated-item')], [])


class TestCurrentDerivations:
    def test_maps_each_item_pair_to_its_current_basis_derivation(self) -> None:
        rubric = _rubric()
        items = [_item(entry_id='kept', outcome='restated'), _item(entry_id='tied'),
                 _item(entry_id='misfiled', outcome='amended')]
        verdicts = [
            *(_rated('kept', role=role, home='YES', conflict='NO', loss='NO')
              for role in ('primary', 'second')),
            _rated('kept', role='primary', basis='attach_time', home='NO', conflict='NO'),
            _rated('tied', role='primary', home='YES', conflict='YES', scope='CENTRAL',
                   action_changing='YES'),
            _rated('tied', role='second', home='YES', conflict='NO', loss='NO'),
            *(_rated('misfiled', role=role, home='NO', conflict='NO') for role in ('primary', 'second')),
        ]
        derivations = _mod().current_derivations(items, verdicts)
        assert set(derivations) == {('kept', 'p1'), ('tied', 'p1'), ('misfiled', 'p1')}
        assert ('p1', rubric.Binary.LOSS) in derivations['kept', 'p1'].consumed
        assert isinstance(derivations['misfiled', 'p1'], rubric.Decided)
        assert [(str(gap.binary), str(gap.kind)) for gap in derivations['tied', 'p1'].missing] == [
            ('conflict', 'tied')]
        assert _rounds().pairs_reading_loss(derivations) == {('kept', 'p1')}
