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


def _resolved(rows: list[dict[str, Any]], binary: str, *, basis: str = 'current') -> object:
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
