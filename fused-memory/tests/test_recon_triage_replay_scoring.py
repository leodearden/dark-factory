"""Tests for recon_triage_replay_scoring.py — how a recon repair of a triage attach is judged.

Reached only through the public surface. The scripts are loaded by path
(``scripts/`` is not a package), lazily through ``_mod()``. Every Decided is
built from synthetic verdict rows through the rubric's own
``resolve_judgements`` and ``derive``; none is hand-constructed.
"""
from __future__ import annotations

import copy
import functools
import json
import math
import re
import types
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
)

SCRIPTS = Path(__file__).parent.parent / 'scripts'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'recon_triage_replay_scoring.py', mod_name='recon_triage_replay_scoring',
    )


@functools.cache
def _rubric() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'write_triage_reader_rubric.py', mod_name='write_triage_reader_rubric',
    )


@functools.cache
def _reader() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'score_write_triage_reader.py', mod_name='score_write_triage_reader',
    )


def _vote(
    entry: str, target: str, *, rater: str = 'opus-primary', role: str = 'primary',
    **answers: Any,
) -> dict[str, Any]:
    """A current-basis verdict row; a YES conflict and a YES loss carry default evidence."""
    row = {
        'entry_id': entry, 'target_id': target, 'basis': 'current', 'rater': rater,
        'role': role, 'batch': 'b1', **answers,
    }
    if answers.get('conflict') == 'YES':
        row.setdefault('conflict_quote', {'parent': 'the record says A', 'child': 'it says B'})
    if answers.get('loss') == 'YES':
        row.setdefault('loss_spans', ['a detail the reader needs'])
    return row


def _decided(outcome: str, *, supersedes_target: bool = False, **answers: Any) -> Any:
    """The rubric's Decided for one write under *outcome*, rated once with *answers*."""
    judgements = _rubric().resolve_judgements([_vote('E', 'P', **answers)])
    derivation = _rubric().derive(
        outcome, {'P': judgements[_rubric().PairKey('E', 'P', _rubric().Basis.CURRENT)]},
        supersedes_target=supersedes_target,
    )
    assert isinstance(derivation, _rubric().Decided), derivation
    return derivation


def _action(decision: str, relabel_to: str | None = None) -> Any:
    return _mod().Action(_mod().Decision(decision), relabel_to)


def _item_row(outcome: str = OUTCOME_AMENDED, **extra: Any) -> dict[str, Any]:
    return {
        'band': 'judge', 'cross_category': False, 'earlier_ids': [], 'entry_id': 'E',
        'outcome': outcome, 'parent_edited_after_attach': False, 'project': 'dark_factory',
        'stratum': f'{outcome}-sample', 'superseded_marks': [], 'supersedes_target': False,
        'target_id': 'P', 'via_sibling': False, **extra,
    }


class TestTheActionVocabulary:
    @pytest.mark.parametrize(('outcome', 'expected'), [
        (OUTCOME_CONTESTED, {('contest_upheld', None), ('not_a_contest', None), ('detach', None),
                             ('superseded_parent', None)}),
        (OUTCOME_AMENDED, {('keep', None), ('relabel', OUTCOME_CONTESTED), ('detach', None),
                           ('superseded_parent', None)}),
        (OUTCOME_RESTATED, {('keep', None), ('relabel', OUTCOME_AMENDED),
                            ('relabel', OUTCOME_CONTESTED), ('detach', None),
                            ('superseded_parent', None)}),
    ])
    def test_each_outcome_allows_exactly_its_repairs(
        self, outcome: str, expected: set[tuple[str, str | None]],
    ) -> None:
        allowed = _mod().allowed_actions(outcome)
        assert isinstance(allowed, frozenset)
        assert {(str(action.decision), action.relabel_to) for action in allowed} == expected

    def test_a_stored_write_has_no_repair_vocabulary(self) -> None:
        with pytest.raises(ValueError, match=OUTCOME_STORED):
            _mod().allowed_actions(OUTCOME_STORED)

    def test_every_decision_is_named(self) -> None:
        assert {str(decision) for decision in _mod().Decision} == {
            'keep', 'detach', 'relabel', 'contest_upheld', 'superseded_parent', 'not_a_contest',
        }

    @pytest.mark.parametrize(('decision', 'relabel_to'), [
        ('relabel', None),
        ('relabel', 'stored'),
        ('keep', OUTCOME_AMENDED),
    ])
    def test_a_malformed_action_is_refused_naming_its_values(
        self, decision: str, relabel_to: str | None,
    ) -> None:
        with pytest.raises(ValueError, match=f'{decision}.*{relabel_to}'):
            _action(decision, relabel_to)

    @pytest.mark.parametrize(('outcome', 'decision', 'relabel_to'), [
        (OUTCOME_RESTATED, 'relabel', None),
        (OUTCOME_AMENDED, 'relabel', OUTCOME_AMENDED),
        (OUTCOME_CONTESTED, 'relabel', OUTCOME_AMENDED),
        (OUTCOME_CONTESTED, 'keep', None),
        (OUTCOME_AMENDED, 'contest_upheld', None),
        (OUTCOME_RESTATED, 'not_a_contest', None),
    ])
    def test_an_action_the_outcome_does_not_allow_is_refused_naming_outcome_and_values(
        self, outcome: str, decision: str, relabel_to: str | None,
    ) -> None:
        with pytest.raises(ValueError) as refused:
            _mod().action_for(outcome, decision, relabel_to)
        message = str(refused.value)
        assert outcome in message
        assert decision in message
        assert str(relabel_to) in message

    def test_an_unknown_decision_is_refused_naming_it(self) -> None:
        with pytest.raises(ValueError, match='merge'):
            _mod().action_for(OUTCOME_AMENDED, 'merge')

    def test_an_allowed_action_is_built(self) -> None:
        action = _mod().action_for(OUTCOME_RESTATED, 'relabel', OUTCOME_AMENDED)
        assert action == _action('relabel', OUTCOME_AMENDED)
        assert action in _mod().allowed_actions(OUTCOME_RESTATED)

    def test_an_action_labels_itself(self) -> None:
        assert _action('relabel', OUTCOME_CONTESTED).label == 'relabel->contested'
        assert _action('detach').label == 'detach'


class TestTheExpectedRepair:
    """expected_action reads the rubric's cells with a fixed precedence."""

    @pytest.mark.parametrize(('outcome', 'supersedes', 'answers', 'expected'), [
        # attached_supersession outranks everything, even an upheld conflict
        (OUTCOME_CONTESTED, True, {'conflict': 'YES'}, ('superseded_parent', None)),
        (OUTCOME_AMENDED, True, {'home': 'YES', 'conflict': 'YES'}, ('superseded_parent', None)),
        (OUTCOME_RESTATED, True, {'home': 'NO', 'conflict': 'NO'}, ('superseded_parent', None)),
        # a true dispute is a correct contest, whatever HOME says (ruling 5b)
        (OUTCOME_CONTESTED, False, {'conflict': 'YES'}, ('contest_upheld', None)),
        (OUTCOME_CONTESTED, False, {'home': 'NO', 'conflict': 'YES'}, ('contest_upheld', None)),
        # {false_conflict, wrong_home}
        (OUTCOME_CONTESTED, False, {'home': 'NO', 'conflict': 'NO'}, ('detach', None)),
        # {false_conflict}
        (OUTCOME_CONTESTED, False, {'home': 'YES', 'conflict': 'NO', 'loss': 'NO'},
         ('not_a_contest', None)),
        # {digest_conflict}, and {digest_conflict, wrong_home}
        (OUTCOME_AMENDED, False, {'home': 'YES', 'conflict': 'YES'},
         ('relabel', OUTCOME_CONTESTED)),
        (OUTCOME_AMENDED, False, {'home': 'NO', 'conflict': 'YES'},
         ('relabel', OUTCOME_CONTESTED)),
        # {hidden_conflict}, and {hidden_conflict, wrong_home}
        (OUTCOME_RESTATED, False, {'home': 'YES', 'conflict': 'YES'},
         ('relabel', OUTCOME_CONTESTED)),
        (OUTCOME_RESTATED, False, {'home': 'NO', 'conflict': 'YES'},
         ('relabel', OUTCOME_CONTESTED)),
        # {wrong_home}
        (OUTCOME_AMENDED, False, {'home': 'NO', 'conflict': 'NO'}, ('detach', None)),
        (OUTCOME_RESTATED, False, {'home': 'NO', 'conflict': 'NO'}, ('detach', None)),
        # {hidden_detail}
        (OUTCOME_RESTATED, False, {'home': 'YES', 'conflict': 'NO', 'loss': 'YES'},
         ('relabel', OUTCOME_AMENDED)),
        # {}
        (OUTCOME_RESTATED, False, {'home': 'YES', 'conflict': 'NO', 'loss': 'NO'}, ('keep', None)),
        (OUTCOME_AMENDED, False, {'home': 'YES', 'conflict': 'NO', 'loss': 'NO'}, ('keep', None)),
        (OUTCOME_AMENDED, False, {'home': 'YES', 'conflict': 'NO', 'loss': 'YES'}, ('keep', None)),
    ])
    def test_the_truth_table(
        self, outcome: str, supersedes: bool, answers: dict[str, str],
        expected: tuple[str, str | None],
    ) -> None:
        decided = _decided(outcome, supersedes_target=supersedes, **answers)
        action = _mod().expected_action(outcome, decided)
        assert action == _action(*expected)
        assert action in _mod().allowed_actions(outcome)


class TestTheRepairedItem:
    @pytest.mark.parametrize(('outcome', 'decision'), [
        (OUTCOME_AMENDED, 'keep'),
        (OUTCOME_RESTATED, 'keep'),
        (OUTCOME_CONTESTED, 'contest_upheld'),
    ])
    def test_keeping_the_outcome_leaves_the_row_unchanged(self, outcome: str, decision: str) -> None:
        row = _item_row(outcome)
        repaired = _mod().repaired_item(row, _action(decision))
        assert repaired == row
        assert repaired is not row

    def test_a_detach_stores_the_write_against_its_old_parent(self) -> None:
        repaired = _mod().repaired_item(_item_row(OUTCOME_CONTESTED), _action('detach'))
        assert repaired['outcome'] == OUTCOME_STORED
        assert repaired['target_id'] is None
        assert repaired['earlier_ids'] == ['P']
        assert repaired['superseded_marks'] == []
        assert repaired['supersedes_target'] is False

    def test_a_supersession_stores_the_write_and_marks_the_old_parent(self) -> None:
        row = _item_row(OUTCOME_AMENDED, supersedes_target=True)
        repaired = _mod().repaired_item(row, _action('superseded_parent'))
        assert repaired['outcome'] == OUTCOME_STORED
        assert repaired['target_id'] is None
        assert repaired['earlier_ids'] == ['P']
        assert repaired['superseded_marks'] == ['P']
        assert repaired['supersedes_target'] is False

    @pytest.mark.parametrize(('outcome', 'relabel_to'), [
        (OUTCOME_RESTATED, OUTCOME_AMENDED),
        (OUTCOME_RESTATED, OUTCOME_CONTESTED),
        (OUTCOME_AMENDED, OUTCOME_CONTESTED),
    ])
    def test_a_relabel_changes_only_the_outcome(self, outcome: str, relabel_to: str) -> None:
        row = _item_row(outcome)
        repaired = _mod().repaired_item(row, _action('relabel', relabel_to))
        assert repaired == {**row, 'outcome': relabel_to}

    def test_not_a_contest_reads_the_write_as_an_amendment(self) -> None:
        row = _item_row(OUTCOME_CONTESTED)
        assert _mod().repaired_item(row, _action('not_a_contest')) == {
            **row, 'outcome': OUTCOME_AMENDED,
        }

    def test_every_allowed_repair_gives_a_valid_row_and_never_mutates_its_input(self) -> None:
        for outcome in (OUTCOME_CONTESTED, OUTCOME_AMENDED, OUTCOME_RESTATED):
            for action in _mod().allowed_actions(outcome):
                row = _item_row(outcome)
                before = copy.deepcopy(row)
                repaired = _mod().repaired_item(row, action)
                assert row == before, (outcome, action)
                _reader().ReaderItem.from_row(0, repaired)

    def test_a_repair_the_outcome_does_not_allow_is_refused(self) -> None:
        with pytest.raises(ValueError, match=f'{OUTCOME_CONTESTED}.*keep'):
            _mod().repaired_item(_item_row(OUTCOME_CONTESTED), _action('keep'))


# ── score_replay over a hand-worked corpus ─────────────────────────────────
#
# Corpus strata (populations in _STRATA): contested-census holds C1 C2 C3 T1 X1
# (df) and C4 C5 C6 (rf); amended-sample A1 A2 (df), A3 A4 X2 (rf);
# restated-sample R1 (df), R2 R3 (rf). X1 and X2 are never sampled; T1 is the
# report-only touched row. Harm figures use the provisional weights.

_STRATA = {'contested-census': 16, 'amended-sample': 50, 'restated-sample': 30}

_CORPUS: tuple[tuple[str, str, str, str, dict[str, Any], dict[str, Any]], ...] = (
    # (entry, outcome, target, project, answers, item extras)
    # {} / sq {orphaned_conflict}; expected contest_upheld; recon contest_upheld
    ('C1', OUTCOME_CONTESTED, 'P1', 'df', {'conflict': 'YES'}, {}),
    # {false_conflict, wrong_home} / {}; expected detach; recon detach (misfile detected)
    ('C2', OUTCOME_CONTESTED, 'P2', 'df', {'home': 'NO', 'conflict': 'NO'}, {}),
    # {false_conflict} / {duplicate_clutter}; expected not_a_contest; recon detach (wrong detach)
    ('C3', OUTCOME_CONTESTED, 'P3', 'df', {'home': 'YES', 'conflict': 'NO', 'loss': 'NO'}, {}),
    # {digest_conflict, wrong_home} / {orphaned_conflict}; expected relabel->contested; recon
    # relabel->contested, which clears both cells (misfile detected)
    ('A1', OUTCOME_AMENDED, 'P4', 'df', {'home': 'NO', 'conflict': 'YES'}, {}),
    # {} / {duplicate_clutter}; expected keep; recon gave no decision
    ('A2', OUTCOME_AMENDED, 'P5', 'df', {'home': 'YES', 'conflict': 'NO', 'loss': 'NO'}, {}),
    # {hidden_detail} / {}; expected relabel->amended; recon relabel->amended
    ('R1', OUTCOME_RESTATED, 'P6', 'df',
     {'home': 'YES', 'conflict': 'NO', 'loss': 'YES', 'digest': 'YES'}, {'band': 'restated'}),
    # {} / {orphaned_conflict}; expected contest_upheld; recon detach (a correct contest detached)
    ('C4', OUTCOME_CONTESTED, 'P7', 'rf', {'home': 'NO', 'conflict': 'YES'}, {}),
    # {false_conflict, wrong_home} / {}; expected detach; recon not_a_contest leaves
    # {wrong_home} (misfile missed)
    ('C5', OUTCOME_CONTESTED, 'P8', 'rf', {'home': 'NO', 'conflict': 'NO'}, {}),
    # {} / {orphaned_conflict}; expected contest_upheld; recon not_a_contest reads as an
    # amendment, which needs the HOME no rater gave: unpriced
    ('C6', OUTCOME_CONTESTED, 'P13', 'rf', {'conflict': 'YES'}, {}),
    # {wrong_home} / {}; expected detach; recon detach (misfile detected)
    ('A3', OUTCOME_AMENDED, 'P9', 'rf', {'home': 'NO', 'conflict': 'NO'}, {}),
    # {digest_conflict, attached_supersession} / {orphaned_conflict}; expected
    # superseded_parent; recon keep
    ('A4', OUTCOME_AMENDED, 'P10', 'rf', {'home': 'YES', 'conflict': 'YES'},
     {'supersedes_target': True}),
    # {wrong_home} / {}; expected detach; recon gave no decision (misfile missed)
    ('R2', OUTCOME_RESTATED, 'P11', 'rf', {'home': 'NO', 'conflict': 'NO'}, {'band': 'restated'}),
    # {hidden_conflict} / {orphaned_conflict}; expected relabel->contested; recon relabel->contested
    ('R3', OUTCOME_RESTATED, 'P12', 'rf', {'home': 'YES', 'conflict': 'YES'}, {'band': 'restated'}),
    # touched, report-only: {false_conflict, wrong_home}; expected detach; recon detach
    ('T1', OUTCOME_CONTESTED, 'P14', 'df', {'home': 'NO', 'conflict': 'NO'}, {}),
    # never sampled
    ('X1', OUTCOME_CONTESTED, 'P15', 'df', {'conflict': 'YES'}, {}),
    ('X2', OUTCOME_AMENDED, 'P16', 'rf', {'home': 'NO', 'conflict': 'NO'}, {}),
)

_RECON: dict[str, tuple[str, str | None]] = {
    'C1': ('contest_upheld', None), 'C2': ('detach', None), 'C3': ('detach', None),
    'A1': ('relabel', OUTCOME_CONTESTED), 'R1': ('relabel', OUTCOME_AMENDED),
    'C4': ('detach', None), 'C5': ('not_a_contest', None), 'C6': ('not_a_contest', None),
    'A3': ('detach', None), 'A4': ('keep', None), 'R3': ('relabel', OUTCOME_CONTESTED),
    'T1': ('detach', None),
}
_LIVE = {'C2': 'missing', 'T1': 'relinked'}
_REPORT_ONLY = frozenset({'T1'})
_UNSAMPLED = frozenset({'X1', 'X2'})
_CORPUS_STRATUM = {
    OUTCOME_CONTESTED: 'contested-census', OUTCOME_AMENDED: 'amended-sample',
    OUTCOME_RESTATED: 'restated-sample',
}


def _corpus_items() -> list[dict[str, Any]]:
    return [
        {**_item_row(outcome, entry_id=entry, target_id=target, project=project,
                     stratum=_CORPUS_STRATUM[outcome]), **extras}
        for entry, outcome, target, project, _answers, extras in _CORPUS
    ]


def _verdicts() -> list[dict[str, Any]]:
    return [
        _vote(entry, target, **answers)
        for entry, _outcome, target, _project, answers, _extras in _CORPUS
    ]


def _sample_rows() -> list[dict[str, Any]]:
    return [
        {'entry_id': entry, 'target_id': target, 'outcome': outcome, 'project': project,
         'stratum': f'{outcome}|{project}', 'headline': entry not in _REPORT_ONLY,
         'live_recon_state': _LIVE.get(entry)}
        for entry, outcome, target, project, _answers, _extras in _CORPUS
        if entry not in _UNSAMPLED
    ]


def _decision_rows(recon: dict[str, tuple[str, str | None]] = _RECON) -> list[dict[str, Any]]:
    targets = {entry: target for entry, _outcome, target, *_rest in _CORPUS}
    return [
        {'entry_id': entry, 'target_id': targets[entry], 'batch': 'b1', 'model': 'fake',
         'decision': decision, 'relabel_to': relabel_to, 'reason': 'r', 'evidence': [],
         'cost_usd': 0.0}
        for entry, (decision, relabel_to) in sorted(recon.items())
    ]


def _score(**overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        'sample_rows': _sample_rows(), 'corpus_items': _corpus_items(), 'verdicts': _verdicts(),
        'decisions': _decision_rows(), 'strata_populations': _STRATA,
    }
    arguments.update(overrides)
    return _mod().score_replay(
        arguments.pop('sample_rows'), arguments.pop('corpus_items'), arguments.pop('verdicts'),
        arguments.pop('decisions'), rubric='v2.2', **arguments,
    )


@functools.cache
def _report() -> dict[str, Any]:
    return _score()


@functools.cache
def _stats() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'write_triage_stats.py', mod_name='write_triage_stats')


def _counts(block: dict[str, Any]) -> tuple[int, int]:
    return block['hits'], block['n']


class TestTheHeadlineFigures:
    def test_misfile_detection_is_the_share_of_wrong_homes_the_repair_clears(self) -> None:
        # C2 A1 A3 cleared; C5 left wrong, R2 had no decision
        assert _counts(_report()['headline']['misfile_detection']) == (3, 5)

    def test_wrong_detaches_are_detaches_of_rightly_homed_children(self) -> None:
        block = _report()['headline']['wrong_detach_rate']
        # over C1 C3 A2 R1 C4 C6 A4 R3: C3 and C4 detached
        assert _counts(block) == (2, 8)
        assert block['correct_contests_detached'] == 1

    def test_contest_resolution_is_exact_agreement_on_contested_children(self) -> None:
        # C1 C2 right; C3 C4 C5 C6 wrong
        assert _counts(_report()['headline']['contest_resolution_accuracy']) == (2, 6)

    def test_action_accuracy_overall_and_by_outcome(self) -> None:
        report = _report()
        assert _counts(report['headline']['action_accuracy']) == (6, 13)
        assert {
            outcome: _counts(block) for outcome, block in report['action_accuracy_by_outcome'].items()
        } == {OUTCOME_AMENDED: (2, 4), OUTCOME_CONTESTED: (2, 6), OUTCOME_RESTATED: (2, 3)}

    def test_the_confusion_matrix_keys_expected_by_recon(self) -> None:
        assert _report()['confusion'] == {
            'contest_upheld': {'contest_upheld': 1, 'detach': 1, 'not_a_contest': 1},
            'detach': {'detach': 2, 'no_decision': 1, 'not_a_contest': 1},
            'keep': {'no_decision': 1},
            'not_a_contest': {'detach': 1},
            'relabel->amended': {'relabel->amended': 1},
            'relabel->contested': {'relabel->contested': 2},
            'superseded_parent': {'keep': 1},
        }

    def test_the_judges_wrong_home_rates_on_the_same_rows(self) -> None:
        headline = _report()['headline']
        assert _counts(headline['judge_wrong_home_all']) == (5, 13)
        assert _counts(headline['judge_wrong_home_contested']) == (2, 6)


class TestPerProject:
    def test_every_metric_is_reported_per_project(self) -> None:
        by_project = _report()['by_project']
        assert sorted(by_project) == ['df', 'rf']
        assert {
            project: {metric: _counts(block) for metric, block in metrics.items()}
            for project, metrics in by_project.items()
        } == {
            'df': {
                'misfile_detection': (2, 2), 'wrong_detach_rate': (1, 4),
                'contest_resolution_accuracy': (2, 3), 'action_accuracy': (4, 6),
                'judge_wrong_home_all': (2, 6), 'judge_wrong_home_contested': (1, 3),
            },
            'rf': {
                'misfile_detection': (1, 3), 'wrong_detach_rate': (1, 4),
                'contest_resolution_accuracy': (0, 3), 'action_accuracy': (2, 7),
                'judge_wrong_home_all': (3, 7), 'judge_wrong_home_contested': (1, 3),
            },
        }
        assert by_project['df']['wrong_detach_rate']['correct_contests_detached'] == 0
        assert by_project['rf']['wrong_detach_rate']['correct_contests_detached'] == 1


class TestAMissingDecision:
    def test_is_reported(self) -> None:
        assert _report()['no_decision'] == {'count': 2, 'entry_ids': ['A2', 'R2']}

    def test_is_a_miss_for_detection_and_never_a_detach(self) -> None:
        everything = dict(_RECON, A2=('detach', None), R2=('detach', None))
        report = _score(decisions=_decision_rows(everything))
        assert _counts(report['headline']['misfile_detection']) == (4, 5)
        assert _counts(report['headline']['wrong_detach_rate']) == (3, 8)
        assert _counts(_report()['headline']['misfile_detection']) == (3, 5)
        assert _counts(_report()['headline']['wrong_detach_rate']) == (2, 8)

    def test_is_priced_at_the_judges_harm(self) -> None:
        # R2's judge harm is its wrong_home, 1; the paired recon total is 7 over 12
        assert _report()['harm']['per_write']['recon'] == round(7 / 12, 4)


class TestHarm:
    def test_harm_per_write_is_paired_over_priced_items(self) -> None:
        harm = _report()['harm']
        assert harm['n'] == 12
        assert harm['per_write'] == {
            'judge': round(20 / 12, 4), 'recon': round(7 / 12, 4), 'status_quo': round(8.5 / 12, 4),
        }
        assert harm['delta_per_write'] == round(-13 / 12, 4)
        low, high = harm['delta_interval95']
        assert low <= harm['delta_per_write'] <= high

    def test_an_unpriceable_repair_is_listed_not_raised(self) -> None:
        assert _report()['harm']['unpriced'] == [
            {'entry_id': 'C6', 'action': 'not_a_contest', 'derivation': 'incomplete'},
        ]

    def test_population_harm_weighs_each_stratum(self) -> None:
        harm = _report()['harm']
        # judge harm weighted by the stratum weights over the 12 priced rows: 488/3 over 94
        assert harm['population_per_write']['judge'] == round(488 / 282, 4)
        low, high = harm['population_delta_interval95']
        assert low <= harm['population_delta_per_write'] <= high

    def test_harm_is_reported_per_project(self) -> None:
        by_project = _report()['harm']['by_project']
        assert {project: (figures['n'], figures['per_write'], figures['delta_per_write'])
                for project, figures in by_project.items()} == {
            'df': (6, {'judge': 1.5, 'recon': round(0.5 / 6, 4), 'status_quo': round(4 / 6, 4)},
                   round(-8.5 / 6, 4)),
            'rf': (6, {'judge': round(11 / 6, 4), 'recon': round(6.5 / 6, 4), 'status_quo': 0.75},
                   -0.75),
        }

    def test_the_report_records_the_weights_and_their_status(self) -> None:
        report = _report()
        assert report['harm_weights'] == {
            str(cell): weight for cell, weight in _reader().PROVISIONAL_HARM_WEIGHTS.items()
        }
        assert report['harm_weights_status'] == 'provisional'
        assert report['rubric'] == 'v2.2'
        assert report['bootstrap'] == {'resamples': 10_000, 'seed': 7161}


class TestTheHarmWeights:
    def test_other_weights_are_recorded_as_custom_and_price_the_harm(self) -> None:
        provisional = {str(cell): weight for cell, weight in _reader().PROVISIONAL_HARM_WEIGHTS.items()}
        doubled = {cell: 2 * weight for cell, weight in provisional.items()}
        report, base = _score(harm_weights=doubled), _report()
        assert report['harm_weights'] == doubled
        assert report['harm_weights_status'] == 'custom'
        for arm, harm in base['harm']['per_write'].items():
            assert report['harm']['per_write'][arm] == pytest.approx(2 * harm, abs=2e-4)

    def test_a_cell_left_out_weighs_nothing(self) -> None:
        report = _score(harm_weights={})
        assert report['harm_weights_status'] == 'custom'
        assert report['harm']['per_write'] == {'judge': 0, 'recon': 0, 'status_quo': 0}

    @pytest.mark.parametrize(('weights', 'named'), [
        pytest.param({'wrong_hme': 1.0}, "harm weight 'wrong_hme': 1.0 names no cell", id='unknown-cell'),
        pytest.param({'wrong_home': -1.0}, "harm weight 'wrong_home': -1.0 is not", id='negative'),
        pytest.param({'wrong_home': math.inf}, "harm weight 'wrong_home': inf is not", id='infinite'),
        pytest.param({'wrong_home': math.nan}, "harm weight 'wrong_home': nan is not", id='nan'),
        pytest.param({'wrong_home': True}, "harm weight 'wrong_home': True is not", id='bool'),
        pytest.param({'wrong_home': '1'}, "harm weight 'wrong_home': '1' is not", id='string'),
    ])
    def test_a_weight_that_prices_no_cell_finitely_is_refused_naming_it(
        self, weights: dict[str, Any], named: str,
    ) -> None:
        with pytest.raises(ValueError, match=re.escape(named)):
            _score(harm_weights=weights)


class TestTheTouchedSubset:
    def test_report_only_rows_stay_out_of_every_headline_figure(self) -> None:
        report = _report()
        assert report['sample']['n_headline'] == 13
        assert report['sample']['n_report_only'] == 1
        assert sum(block['n'] for block in report['action_accuracy_by_outcome'].values()) == 13
        assert 'T1' not in report['no_decision']['entry_ids']

    def test_every_touched_row_is_tabled_beside_its_live_state(self) -> None:
        assert _report()['touched'] == [
            {'entry_id': 'C2', 'project': 'df', 'outcome': OUTCOME_CONTESTED, 'headline': True,
             'live_recon_state': 'missing', 'truth_cells': ['false_conflict', 'wrong_home'],
             'expected': 'detach', 'recon': 'detach', 'correct': True,
             'judge_harm': 3, 'recon_harm': 0},
            {'entry_id': 'T1', 'project': 'df', 'outcome': OUTCOME_CONTESTED, 'headline': False,
             'live_recon_state': 'relinked', 'truth_cells': ['false_conflict', 'wrong_home'],
             'expected': 'detach', 'recon': 'detach', 'correct': True,
             'judge_harm': 3, 'recon_harm': 0},
        ]


class TestIntervalsAndWeights:
    def test_a_rate_carries_its_sample_and_population_intervals(self) -> None:
        block = _report()['headline']['misfile_detection']
        assert block['rate'] == 0.6
        assert block['wilson95'] == _stats().wilson95(3, 5)
        # C2 (10/3) A1 (10) A3 (15) of C2 A1 C5 (2) A3 R2 (10): 85/3 over 121/3
        assert block['population_rate'] == round(85 / 121, 4)
        low, high = block['population_interval95']
        assert low <= block['population_rate'] <= high

    def test_stratum_weights_come_from_the_corpus_strata(self) -> None:
        assert _report()['sample']['stratum_weights'] == {
            'amended|df': 10.0, 'amended|rf': 15.0, 'contested|df': round(10 / 3, 4),
            'contested|rf': 2.0, 'restated|df': 10.0, 'restated|rf': 10.0,
        }

    def test_a_sample_stratum_without_a_population_is_refused_by_name(self) -> None:
        strata = {key: value for key, value in _STRATA.items() if key != 'restated-sample'}
        with pytest.raises(ValueError, match=r'restated\|df'):
            _score(strata_populations=strata)


class TestRefusals:
    def test_a_decision_its_outcome_does_not_allow_is_refused_naming_the_entry(self) -> None:
        with pytest.raises(ValueError, match='C1'):
            _score(decisions=_decision_rows(dict(_RECON, C1=('keep', None))))

    def test_a_decision_for_a_child_outside_the_sample_is_refused(self) -> None:
        rows = _decision_rows()
        rows.append({**rows[0], 'entry_id': 'X1', 'target_id': 'P15'})
        with pytest.raises(ValueError, match='X1'):
            _score(decisions=rows)

    def test_a_duplicate_decision_is_refused(self) -> None:
        rows = _decision_rows()
        with pytest.raises(ValueError, match=rows[0]['entry_id']):
            _score(decisions=[*rows, rows[0]])

    def test_a_sample_row_disagreeing_with_its_corpus_item_is_refused(self) -> None:
        rows = _sample_rows()
        rows[0] = {**rows[0], 'outcome': OUTCOME_AMENDED, 'stratum': f'{OUTCOME_AMENDED}|df'}
        with pytest.raises(ValueError, match=rows[0]['entry_id']):
            _score(sample_rows=rows)


class TestDeterminism:
    def test_the_same_inputs_give_a_byte_identical_report(self) -> None:
        assert json.dumps(_score(), sort_keys=False) == json.dumps(_report(), sort_keys=False)


# ── render_markdown ────────────────────────────────────────────────────────

_RATE_KEYS = ('n', 'hits', 'rate', 'wilson95', 'population_rate', 'population_interval95')


def _cell(value: Any) -> str:
    """How a report value reads in a table cell."""
    if value is None:
        return '—'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    return str(value)


def _section(markdown: str, heading: str) -> str:
    """The text under the ``## `` heading starting with *heading*, up to the next one."""
    parts = markdown.split('\n## ')
    found = [part for part in parts if part.startswith(heading)]
    assert len(found) == 1, (heading, [part.splitlines()[0] for part in parts])
    return found[0]


def _rows(section: str) -> list[list[str]]:
    """Each table body row of *section*, as its cells."""
    lines = [line for line in section.splitlines() if line.startswith('| ')]
    return [line[2:-2].split(' | ') for line in lines[1:]]


def _row(section: str, *key: str) -> list[str]:
    found = [row for row in _rows(section) if tuple(row[:len(key)]) == key]
    assert len(found) == 1, (key, _rows(section))
    return found[0][len(key):]


@functools.cache
def _markdown() -> str:
    return _mod().render_markdown(_report())


class TestTheRenderedScores:
    def test_the_headline_table_carries_every_metric_block(self) -> None:
        section = _section(_markdown(), 'Headline')
        for metric, block in _report()['headline'].items():
            assert _row(section, metric) == [
                *(_cell(block[key]) for key in _RATE_KEYS),
                _cell(block.get('correct_contests_detached')),
            ]

    def test_action_accuracy_by_outcome(self) -> None:
        section = _section(_markdown(), 'Action accuracy by outcome')
        for outcome, block in _report()['action_accuracy_by_outcome'].items():
            assert _row(section, outcome) == [_cell(block[key]) for key in _RATE_KEYS]

    def test_the_per_project_table(self) -> None:
        section = _section(_markdown(), 'By project')
        for project, metrics in _report()['by_project'].items():
            for metric, block in metrics.items():
                assert _row(section, project, metric) == [
                    *(_cell(block[key]) for key in _RATE_KEYS),
                    _cell(block.get('correct_contests_detached')),
                ]

    def test_the_confusion_matrix(self) -> None:
        confusion = _report()['confusion']
        columns = sorted({recon for row in confusion.values() for recon in row})
        section = _section(_markdown(), 'Confusion')
        header = [line for line in section.splitlines() if line.startswith('| ')][0]
        assert header[2:-2].split(' | ')[1:] == columns
        for expected, row in confusion.items():
            assert _row(section, expected) == [_cell(row.get(column, 0)) for column in columns]

    def test_the_harm_table_is_labelled_with_the_weights_status(self) -> None:
        report = _report()
        harm = report['harm']
        section = _section(_markdown(), 'Harm per write')
        assert section.splitlines()[0] == f'Harm per write (weights: {report["harm_weights_status"]})'
        regimes = ('judge', 'recon', 'status_quo')
        figures = [('all', harm), *harm['by_project'].items()]
        for scope, block in figures:
            assert _row(section, scope, 'sample') == [
                _cell(block['n']), *(_cell(block['per_write'][regime]) for regime in regimes),
                _cell(block['delta_per_write']), _cell(block['delta_interval95']),
            ]
            assert _row(section, scope, 'population') == [
                _cell(block['n']),
                *(_cell(block['population_per_write'][regime]) for regime in regimes),
                _cell(block['population_delta_per_write']),
                _cell(block['population_delta_interval95']),
            ]

    def test_the_no_decision_and_unpriced_counts(self) -> None:
        report = _report()
        section = _section(_markdown(), 'Missing and unpriced')
        assert _row(section, 'no_decision') == [
            _cell(report['no_decision']['count']), ', '.join(report['no_decision']['entry_ids']),
        ]
        unpriced = report['harm']['unpriced']
        assert _row(section, 'unpriced') == [
            _cell(len(unpriced)),
            ', '.join(f'{row["entry_id"]} ({row["action"]}, {row["derivation"]})' for row in unpriced),
        ]

    def test_the_touched_table(self) -> None:
        section = _section(_markdown(), 'Touched')
        for row in _report()['touched']:
            assert _row(section, row['entry_id']) == [
                _cell(row['project']), _cell(row['outcome']), _cell(row['headline']),
                _cell(row['live_recon_state']), ', '.join(row['truth_cells']) or '—',
                _cell(row['expected']), _cell(row['recon']), _cell(row['correct']),
                _cell(row['judge_harm']), _cell(row['recon_harm']),
            ]

    def test_it_titles_the_rubric(self) -> None:
        assert _markdown().splitlines()[0].endswith(f'(rubric {_report()["rubric"]})')

    def test_it_re_renders_byte_for_byte_from_the_json_report(self) -> None:
        assert _mod().render_markdown(json.loads(json.dumps(_report()))) == _markdown()
        assert _markdown().endswith('\n')
