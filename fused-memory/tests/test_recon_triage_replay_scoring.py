"""Tests for recon_triage_replay_scoring.py — how a recon repair of a triage attach is judged.

Reached only through the public surface. The scripts are loaded by path
(``scripts/`` is not a package), lazily through ``_mod()``. Every Decided is
built from synthetic verdict rows through the rubric's own
``resolve_judgements`` and ``derive``; none is hand-constructed.
"""
from __future__ import annotations

import copy
import functools
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
