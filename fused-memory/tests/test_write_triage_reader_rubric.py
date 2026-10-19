"""Tests for write_triage_reader_rubric.py — the one definition of write-triage rubric v2.

Reached only through the public surface: ``resolve_judgements`` (the verdict
row contract and per-binary majority) and ``derive`` (the derivation table).
The script is loaded by path (``scripts/`` is not a package), lazily through
``_mod()``.
"""
from __future__ import annotations

import functools
import types
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'write_triage_reader_rubric.py'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPT_PATH, mod_name='write_triage_reader_rubric')


def _row(
    entry: str = 'e1', target: str = 't1', *, basis: str = 'current', rater: str = 'r1',
    role: str = 'primary', batch: str = 'b1', **answers: Any,
) -> dict[str, Any]:
    return {
        'entry_id': entry, 'target_id': target, 'basis': basis, 'rater': rater,
        'role': role, 'batch': batch, **answers,
    }


def _resolved(rows: list[dict[str, Any]], binary: str, *, basis: str = 'current') -> Any:
    rubric = _mod()
    judgements = rubric.resolve_judgements(rows)
    return judgements[rubric.PairKey('e1', 't1', rubric.Basis(basis))].answer(rubric.Binary(binary))


class TestTheVerdictRowContract:
    @pytest.mark.parametrize('key', ['entry_id', 'target_id', 'basis', 'rater', 'role', 'batch'])
    def test_a_missing_required_key_is_refused(self, key: str) -> None:
        row = _row(home='YES')
        del row[key]
        with pytest.raises(ValueError, match=f'row 0.*{key}'):
            _mod().resolve_judgements([row])

    @pytest.mark.parametrize(('field', 'value'), [
        ('basis', 'yesterday'),
        ('role', 'fourth'),
        ('home', 'MAYBE'),
        ('loss', 'yes'),
        ('conflict', 'CENTRAL'),
        ('scope', 'YES'),
        ('digest', 'SECONDARY'),
    ])
    def test_an_unknown_word_is_refused_naming_it(self, field: str, value: str) -> None:
        row = _row(home='YES', loss='YES', conflict='YES')
        row[field] = value
        with pytest.raises(ValueError, match=f"row 0.*e1.*t1.*{field}.*'{value}'"):
            _mod().resolve_judgements([row])

    def test_scope_without_a_yes_conflict_is_refused(self) -> None:
        with pytest.raises(ValueError, match="row 1.*scope.*conflict.*'NO'"):
            _mod().resolve_judgements([
                _row(home='YES'),
                _row(rater='r2', conflict='NO', scope='CENTRAL'),
            ])

    def test_scope_on_a_row_carrying_no_conflict_is_refused(self) -> None:
        with pytest.raises(ValueError, match='row 0.*scope.*conflict'):
            _mod().resolve_judgements([_row(home='YES', scope='SECONDARY')])

    def test_digest_without_a_yes_loss_is_refused(self) -> None:
        with pytest.raises(ValueError, match="row 0.*digest.*loss.*'UNCLEAR'"):
            _mod().resolve_judgements([
                _row(loss='UNCLEAR', unclear_reason='cut mid-number', digest='NO'),
            ])

    @pytest.mark.parametrize('reason', [None, '', '   '])
    def test_an_unclear_answer_needs_a_reason(self, reason: str | None) -> None:
        row = _row(home='UNCLEAR')
        if reason is not None:
            row['unclear_reason'] = reason
        with pytest.raises(ValueError, match='row 0.*UNCLEAR.*unclear_reason'):
            _mod().resolve_judgements([row])

    def test_a_row_carrying_no_binary_is_refused(self) -> None:
        with pytest.raises(ValueError, match='row 0.*no binary'):
            _mod().resolve_judgements([_row(reason='nothing to say')])

    def test_a_rater_voting_twice_on_one_pair_and_basis_is_refused(self) -> None:
        with pytest.raises(ValueError, match="row 1.*'r1'"):
            _mod().resolve_judgements([_row(home='YES'), _row(loss='NO')])

    def test_one_rater_may_rate_both_bases(self) -> None:
        rubric = _mod()
        judgements = rubric.resolve_judgements([
            _row(home='YES'), _row(basis='attach_time', home='NO'),
        ])
        assert len(judgements) == 2


class TestPerBinaryResolution:
    def test_a_single_vote_resolves_to_it(self) -> None:
        assert _resolved([_row(home='YES')], 'home') == _mod().Resolved.YES

    def test_a_strict_majority_wins(self) -> None:
        rows = [_row(conflict='NO'), _row(rater='r2', conflict='YES'),
                _row(rater='r3', conflict='NO')]
        assert _resolved(rows, 'conflict') == _mod().Resolved.NO

    def test_an_even_split_is_tied(self) -> None:
        rows = [_row(home='YES'), _row(rater='r2', home='NO')]
        assert _resolved(rows, 'home') == _mod().Resolved.TIED

    def test_unclear_abstains(self) -> None:
        rows = [_row(loss='YES'),
                _row(rater='r2', loss='UNCLEAR', unclear_reason='cut at the number')]
        assert _resolved(rows, 'loss') == _mod().Resolved.YES

    def test_all_unclear_resolves_unclear(self) -> None:
        rows = [_row(home='UNCLEAR', unclear_reason='cut'),
                _row(rater='r2', home='UNCLEAR', unclear_reason='cut')]
        assert _resolved(rows, 'home') == _mod().Resolved.UNCLEAR

    def test_a_binary_no_row_carries_is_absent(self) -> None:
        assert _resolved([_row(home='YES')], 'conflict') is None

    def test_only_rows_carrying_a_binary_vote_on_it(self) -> None:
        rows = [_row(home='YES'), _row(rater='r2', loss='NO')]
        assert _resolved(rows, 'home') == _mod().Resolved.YES

    def test_scope_resolves_central_or_secondary(self) -> None:
        rows = [_row(conflict='YES', scope='SECONDARY'),
                _row(rater='r2', conflict='YES', scope='CENTRAL'),
                _row(rater='r3', conflict='YES', scope='SECONDARY')]
        assert _resolved(rows, 'scope') == _mod().Resolved.SECONDARY

    def test_a_scope_split_is_tied(self) -> None:
        rows = [_row(conflict='YES', scope='SECONDARY'),
                _row(rater='r2', conflict='YES', scope='CENTRAL')]
        assert _resolved(rows, 'scope') == _mod().Resolved.TIED

    def test_digest_resolves_like_any_binary(self) -> None:
        rows = [_row(loss='YES', digest='NO')]
        assert _resolved(rows, 'digest') == _mod().Resolved.NO

    def test_a_derived_row_counts_as_one_vote(self) -> None:
        rows = [_row(rater='v1', role='derived', home='YES'), _row(home='NO')]
        assert _resolved(rows, 'home') == _mod().Resolved.TIED

    def test_the_two_bases_never_mix(self) -> None:
        rows = [_row(home='YES'), _row(rater='r2', home='YES'),
                _row(basis='attach_time', home='NO')]
        assert _resolved(rows, 'home') == _mod().Resolved.YES
        assert _resolved(rows, 'home', basis='attach_time') == _mod().Resolved.NO


def _judged(target: str, *votes: dict[str, str]) -> Any:
    """The Judgement of write 'w' read against *target*, one rater per vote; None if no votes."""
    if not votes:
        return None
    rubric = _mod()
    rows = [_row('w', target, rater=f'r{index}', **vote) for index, vote in enumerate(votes)]
    return rubric.resolve_judgements(rows)[rubric.PairKey('w', target, rubric.Basis.CURRENT)]


def _derive(outcome: str, *votes: dict[str, str], **options: Any) -> Any:
    return _mod().derive(outcome, {'p': _judged('p', *votes)}, **options)


def _decided(result: Any) -> Any:
    assert isinstance(result, _mod().Decided), result
    return result


_HOME_FLAT = {'home': 'YES', 'loss': 'NO', 'conflict': 'NO'}
_HOME_MORE = {'home': 'YES', 'loss': 'YES', 'conflict': 'NO'}


class TestTheDerivationTable:
    @pytest.mark.parametrize(('outcome', 'vote', 'triage', 'status_quo'), [
        ('restated', _HOME_FLAT, set(), {'duplicate_clutter'}),
        ('restated', _HOME_MORE, {'hidden_detail'}, set()),
        ('restated', {'home': 'YES', 'conflict': 'YES'}, {'hidden_conflict'}, {'orphaned_conflict'}),
        ('restated', {'home': 'NO', 'conflict': 'YES'},
         {'hidden_conflict', 'wrong_home'}, {'orphaned_conflict'}),
        ('restated', {'home': 'NO', 'conflict': 'NO'}, {'wrong_home'}, set()),
        ('amended', _HOME_MORE, set(), set()),
        ('amended', _HOME_FLAT, set(), {'duplicate_clutter'}),
        ('amended', {'home': 'YES', 'conflict': 'YES'}, {'digest_conflict'}, {'orphaned_conflict'}),
        ('amended', {'home': 'NO', 'conflict': 'YES'},
         {'digest_conflict', 'wrong_home'}, {'orphaned_conflict'}),
        ('amended', {'home': 'NO', 'conflict': 'NO'}, {'wrong_home'}, set()),
        ('contested', {'conflict': 'YES'}, set(), {'orphaned_conflict'}),
        ('contested', {'home': 'NO', 'conflict': 'YES', 'scope': 'SECONDARY'},
         set(), {'orphaned_conflict'}),
        ('contested', {'home': 'NO', 'conflict': 'NO'}, {'false_conflict', 'wrong_home'}, set()),
        ('contested', _HOME_FLAT, {'false_conflict'}, {'duplicate_clutter'}),
        ('contested', _HOME_MORE, {'false_conflict'}, set()),
    ])
    def test_an_attach_outcome_derives_its_cells_in_both_regimes(
        self, outcome: str, vote: dict[str, str], triage: set[str], status_quo: set[str],
    ) -> None:
        result = _decided(_derive(outcome, vote))
        assert result.triage_cells == triage
        assert result.status_quo_cells == status_quo
        assert result.m1_hidden is None

    @pytest.mark.parametrize(('outcome', 'vote', 'conflict'), [
        ('restated', {'home': 'NO', 'conflict': 'YES'}, True),
        ('amended', _HOME_FLAT, False),
        ('contested', {'conflict': 'YES'}, True),
        ('contested', _HOME_FLAT, False),
    ])
    def test_a_decided_attach_says_whether_the_write_conflicts(
        self, outcome: str, vote: dict[str, str], conflict: bool,
    ) -> None:
        assert _decided(_derive(outcome, vote)).conflict is conflict

    def test_a_supersession_filed_as_an_attach_is_always_wrong(self) -> None:
        for outcome, vote in (('restated', _HOME_FLAT), ('contested', {'conflict': 'YES'})):
            result = _decided(_derive(outcome, vote, supersedes_target=True))
            assert result.triage_cells == {'attached_supersession'}

    def test_cell_rules_state_one_rule_per_cell(self) -> None:
        rubric = _mod()
        assert set(rubric.CELL_RULES) == set(rubric.Cell)
        assert len(rubric.Cell) == 8
        assert all(isinstance(rule, str) and rule.strip() for rule in rubric.CELL_RULES.values())


class TestTheStoredRule:
    def _derive(self, by_target: dict[str, list[dict[str, str]]], **options: Any) -> Any:
        rubric = _mod()
        return rubric.derive(
            'stored', {target: _judged(target, *votes) for target, votes in by_target.items()},
            **options,
        )

    @pytest.mark.parametrize(('by_target', 'marks', 'cells', 'conflict'), [
        ({}, frozenset(), set(), False),
        ({'e1': [{'conflict': 'YES'}]}, frozenset(), {'orphaned_conflict'}, True),
        ({'e1': [{'conflict': 'YES'}]}, frozenset({'e1'}), set(), True),
        ({'e1': [_HOME_FLAT]}, frozenset(), {'duplicate_clutter'}, False),
        ({'e1': [_HOME_MORE]}, frozenset(), set(), False),
        ({'e1': [{'home': 'NO', 'conflict': 'NO'}]}, frozenset(), set(), False),
        ({'e1': [{'conflict': 'YES'}], 'e2': [_HOME_FLAT]}, frozenset({'e1'}),
         {'duplicate_clutter'}, True),
        ({'e1': [{'conflict': 'YES'}], 'e2': [{'conflict': 'YES'}]}, frozenset({'e1'}),
         {'orphaned_conflict'}, True),
    ])
    def test_a_stored_write_is_judged_against_every_earlier_record(
        self, by_target: dict[str, list[dict[str, str]]], marks: frozenset[str],
        cells: set[str], conflict: bool,
    ) -> None:
        result = _decided(self._derive(by_target, superseded_marks=marks))
        assert result.triage_cells == cells
        assert result.status_quo_cells == cells
        assert result.conflict is conflict

    def test_a_refusal_names_every_earlier_records_gap_at_once(self) -> None:
        result = self._derive({'e1': [{'conflict': 'NO'}], 'e2': []})
        assert isinstance(result, _mod().Incomplete)
        assert [(gap.target_id, gap.binary, gap.kind) for gap in result.missing] == [
            ('e1', 'home', 'unrated'), ('e2', 'conflict', 'unrated'),
        ]

    def test_a_stored_write_has_no_target_to_supersede(self) -> None:
        with pytest.raises(ValueError, match='supersedes_target'):
            self._derive({}, supersedes_target=True)


class TestDecisiveness:
    def _gaps(self, result: Any) -> list[tuple[str, str, str]]:
        assert isinstance(result, _mod().Incomplete), result
        return [(gap.target_id, gap.binary, gap.kind) for gap in result.missing]

    def test_a_decisive_unrated_binary_is_incomplete(self) -> None:
        assert self._gaps(_derive('restated', {'home': 'YES', 'conflict': 'NO'})) == [
            ('p', 'loss', 'unrated'),
        ]

    def test_a_decisive_tie_is_incomplete(self) -> None:
        result = _derive('restated', {'home': 'YES', 'conflict': 'YES'},
                         {'home': 'YES', 'conflict': 'NO'})
        assert self._gaps(result) == [('p', 'conflict', 'tied')]

    def test_an_unrated_pair_names_its_first_decisive_binary(self) -> None:
        assert self._gaps(_derive('contested')) == [('p', 'conflict', 'unrated')]

    def test_the_status_quo_regime_can_be_the_decisive_one(self) -> None:
        assert self._gaps(_derive('amended', {'home': 'YES', 'conflict': 'NO'})) == [
            ('p', 'loss', 'unrated'),
        ]

    def test_a_decisive_unclear_binary_makes_the_item_unclear(self) -> None:
        result = _derive('restated', {'home': 'YES', 'conflict': 'NO', 'loss': 'UNCLEAR',
                                      'unclear_reason': 'cut at the threshold'})
        assert isinstance(result, _mod().Unclear)
        assert result.binaries == (('p', 'loss'),)

    def test_a_non_decisive_tie_never_blocks(self) -> None:
        result = _derive('contested', {'home': 'YES', 'conflict': 'YES'},
                         {'home': 'NO', 'conflict': 'YES'})
        assert _decided(result).triage_cells == set()

    def test_a_non_decisive_absent_binary_never_blocks(self) -> None:
        assert _decided(_derive('restated', {'home': 'NO', 'conflict': 'NO'})).triage_cells == {
            'wrong_home',
        }

    def test_an_attach_outcome_needs_exactly_one_target(self) -> None:
        with pytest.raises(ValueError, match='restated.*one target'):
            _mod().derive('restated', {})

    def test_an_unknown_outcome_is_refused(self) -> None:
        with pytest.raises(ValueError, match="'filed'"):
            _mod().derive('filed', {})


class TestM1Hiddenness:
    @pytest.mark.parametrize(('outcome', 'vote', 'hidden'), [
        ('restated', {'conflict': 'YES', 'home': 'YES'}, True),
        ('restated', _HOME_MORE, True),
        ('restated', _HOME_FLAT, False),
        ('restated', {'home': 'NO', 'conflict': 'NO'}, False),
        ('amended', {**_HOME_MORE, 'digest': 'NO'}, True),
        ('amended', {**_HOME_MORE, 'digest': 'YES'}, False),
        ('amended', {'home': 'YES', 'conflict': 'YES'}, True),
        ('amended', _HOME_FLAT, False),
        ('contested', {'conflict': 'NO', 'home': 'NO'}, False),
    ])
    def test_hiddenness_in_the_band(
        self, outcome: str, vote: dict[str, str], hidden: bool,
    ) -> None:
        assert _decided(_derive(outcome, vote, in_m1_band=True)).m1_hidden is hidden

    def test_a_stored_write_in_the_band_is_never_hidden(self) -> None:
        result = _mod().derive('stored', {}, in_m1_band=True)
        assert _decided(result).m1_hidden is False

    def test_digest_is_decisive_only_in_the_band(self) -> None:
        in_band = _derive('amended', _HOME_MORE, in_m1_band=True)
        assert isinstance(in_band, _mod().Incomplete)
        assert [(g.target_id, g.binary, g.kind) for g in in_band.missing] == [
            ('p', 'digest', 'unrated'),
        ]
        assert _decided(_derive('amended', _HOME_MORE)).m1_hidden is None
