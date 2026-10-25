"""Tests for write_triage_reader_rubric.py — the one definition of write-triage rubric v2.1.

Reached only through the public surface: ``resolve_judgements`` (the verdict
row contract and per-binary resolution), ``derive`` (the derivation table),
``quote_found`` and ``digest_visible`` (the evidence checks) and ``is_abstain``
(the one abstain word).
The script is loaded by path (``scripts/`` is not a package), lazily through
``_mod()``.
"""
from __future__ import annotations

import functools
import re
import types
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'write_triage_reader_rubric.py'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPT_PATH, mod_name='write_triage_reader_rubric')


_QUOTE = {'parent': 'the record says A', 'child': 'the write says B'}
_SPANS = ['a detail the reader needs']


def _row(
    entry: str = 'e1', target: str = 't1', *, basis: str = 'current', rater: str = 'r1',
    role: str = 'primary', batch: str = 'b1', **answers: Any,
) -> dict[str, Any]:
    """A verdict row; a rater row's YES conflict and YES loss carry default evidence."""
    row = {
        'entry_id': entry, 'target_id': target, 'basis': basis, 'rater': rater,
        'role': role, 'batch': batch, **answers,
    }
    if role != 'derived':
        if answers.get('conflict') == 'YES':
            row.setdefault('conflict_quote', dict(_QUOTE))
        if answers.get('loss') == 'YES':
            row.setdefault('loss_spans', list(_SPANS))
    return row


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
        ('action_changing', 'CENTRAL'),
        ('digest', 'SECONDARY'),
    ])
    def test_an_unknown_word_is_refused_naming_it(self, field: str, value: str) -> None:
        row = _row(home='YES', conflict='NO', loss='YES')
        row[field] = value
        with pytest.raises(ValueError, match=f"row 0.*e1.*t1.*{field}.*'{value}'"):
            _mod().resolve_judgements([row])

    @pytest.mark.parametrize(('field', 'value'), [
        ('home', ['YES']),
        ('scope', {'CENTRAL': True}),
        ('basis', ['current']),
        ('role', {'primary': 1}),
    ])
    def test_a_word_that_is_not_a_string_is_refused_naming_it(self, field: str, value: object) -> None:
        row = _row(home='YES', conflict='YES', scope='CENTRAL')
        row[field] = value
        with pytest.raises(ValueError, match=f'row 0.*{field} {re.escape(repr(value))}'):
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
                _row(home='YES', conflict='NO', loss='UNCLEAR', unclear_reason='cut mid-number',
                     digest='NO'),
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
            _mod().resolve_judgements([_row(home='YES'), _row(conflict='NO')])

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
        rows = [_row(home='YES', conflict='NO', loss='YES'),
                _row(rater='r2', home='YES', conflict='NO', loss='UNCLEAR',
                     unclear_reason='cut at the number')]
        assert _resolved(rows, 'loss') == _mod().Resolved.YES

    def test_all_unclear_resolves_unclear(self) -> None:
        rows = [_row(home='UNCLEAR', unclear_reason='cut'),
                _row(rater='r2', home='UNCLEAR', unclear_reason='cut')]
        assert _resolved(rows, 'home') == _mod().Resolved.UNCLEAR

    def test_a_binary_no_row_carries_is_absent(self) -> None:
        assert _resolved([_row(home='YES')], 'conflict') is None

    def test_only_rows_carrying_a_binary_vote_on_it(self) -> None:
        rows = [_row(home='YES'), _row(rater='r2', conflict='NO')]
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

    def test_a_derived_row_is_the_only_vote_on_the_binary_it_fixes(self) -> None:
        rows = [_row(rater='v1', role='derived', home='YES'),
                _row(home='NO', non_consumed=['home'])]
        assert _resolved(rows, 'home') == _mod().Resolved.YES

    def test_action_changing_resolves_by_majority(self) -> None:
        rows = [_row(conflict='YES', action_changing='YES'),
                _row(rater='r2', conflict='YES', action_changing='NO'),
                _row(rater='r3', conflict='YES', action_changing='YES')]
        assert _resolved(rows, 'action_changing') == _mod().Resolved.YES

    def test_an_action_changing_split_is_tied(self) -> None:
        rows = [_row(conflict='YES', action_changing='YES'),
                _row(rater='r2', conflict='YES', action_changing='NO')]
        assert _resolved(rows, 'action_changing') == _mod().Resolved.TIED

    def test_the_two_bases_never_mix(self) -> None:
        rows = [_row(home='YES'), _row(rater='r2', home='YES'),
                _row(basis='attach_time', home='NO')]
        assert _resolved(rows, 'home') == _mod().Resolved.YES
        assert _resolved(rows, 'home', basis='attach_time') == _mod().Resolved.NO


class TestTheFollowUps:
    def test_each_follow_up_names_the_answers_it_needs_on_its_row(self) -> None:
        rubric = _mod()
        binary, answer = rubric.Binary, rubric.Answer
        assert {follow_up: dict(needs) for follow_up, needs in rubric.FOLLOW_UPS.items()} == {
            binary.LOSS: {binary.HOME: answer.YES, binary.CONFLICT: answer.NO},
            binary.SCOPE: {binary.CONFLICT: answer.YES},
            binary.ACTION_CHANGING: {binary.CONFLICT: answer.YES},
            binary.DIGEST: {binary.LOSS: answer.YES},
        }

    @pytest.mark.parametrize(('answers', 'needed', 'found'), [
        ({'conflict': 'NO'}, "home 'YES'", 'None'),
        ({'home': 'NO', 'conflict': 'NO'}, "home 'YES'", "'NO'"),
        ({'home': 'YES', 'conflict': 'YES'}, "conflict 'NO'", "'YES'"),
        ({'home': 'YES'}, "conflict 'NO'", 'None'),
    ])
    def test_loss_is_asked_only_after_home_yes_and_conflict_no(
        self, answers: dict[str, str], needed: str, found: str,
    ) -> None:
        with pytest.raises(ValueError, match=f"row 0.*e1.*t1.*loss 'NO' needs {needed}, got {found}"):
            _mod().resolve_judgements([_row(**answers, loss='NO')])

    @pytest.mark.parametrize('loss', ['YES', 'NO', 'UNCLEAR'])
    def test_loss_on_a_row_with_home_yes_and_conflict_no_is_accepted(self, loss: str) -> None:
        rows = [_row(home='YES', conflict='NO', loss=loss, unclear_reason='cut at the number')]
        assert _resolved(rows, 'loss') == _mod().Resolved(loss)

    @pytest.mark.parametrize('conflict', [None, 'NO', 'UNCLEAR'])
    def test_action_changing_without_a_yes_conflict_is_refused(self, conflict: str | None) -> None:
        answers = {} if conflict is None else {'conflict': conflict}
        row = _row(home='YES', **answers, action_changing='YES', unclear_reason='cut')
        with pytest.raises(ValueError, match=f"row 0.*action_changing 'YES' needs conflict 'YES', got {conflict!r}"):
            _mod().resolve_judgements([row])


class TestTheEvidenceContract:
    @pytest.mark.parametrize('role', ['primary', 'second', 'third'])
    def test_a_rater_yes_conflict_needs_a_conflict_quote(self, role: str) -> None:
        row = _row(role=role, conflict='YES')
        del row['conflict_quote']
        with pytest.raises(ValueError, match='row 0.*e1.*t1.*conflict.*conflict_quote'):
            _mod().resolve_judgements([row])

    @pytest.mark.parametrize('quote', [
        'the record says A',
        {},
        {'parent': 'the record says A'},
        {'child': 'the write says B'},
        {'parent': '', 'child': 'the write says B'},
        {'parent': 'the record says A', 'child': '   '},
        {'parent': 'the record says A', 'child': 3},
        None,
    ])
    def test_a_malformed_conflict_quote_is_refused(self, quote: Any) -> None:
        with pytest.raises(ValueError, match='row 0.*conflict_quote'):
            _mod().resolve_judgements([_row(conflict='YES', conflict_quote=quote)])

    @pytest.mark.parametrize('role', ['primary', 'second', 'third'])
    def test_a_rater_yes_loss_needs_loss_spans(self, role: str) -> None:
        row = _row(role=role, home='YES', conflict='NO', loss='YES')
        del row['loss_spans']
        with pytest.raises(ValueError, match='row 0.*e1.*t1.*loss.*loss_spans'):
            _mod().resolve_judgements([row])

    @pytest.mark.parametrize('spans', [
        'a detail', [], [''], ['a detail', '  '], ['a detail', 3], None, {'span': 'a detail'},
    ])
    def test_malformed_loss_spans_are_refused(self, spans: Any) -> None:
        row = _row(home='YES', conflict='NO', loss='YES', loss_spans=spans)
        with pytest.raises(ValueError, match='row 0.*loss_spans'):
            _mod().resolve_judgements([row])

    def test_a_digest_needs_loss_spans_on_its_row(self) -> None:
        row = _row(rater='v1', role='derived', home='YES', conflict='NO', loss='YES', digest='YES')
        with pytest.raises(ValueError, match='row 0.*digest.*loss_spans'):
            _mod().resolve_judgements([row])

    def test_a_derived_row_needs_no_evidence(self) -> None:
        rows = [_row(rater='v1', role='derived', conflict='YES'),
                _row('e2', rater='v1', role='derived', home='YES', conflict='NO', loss='YES')]
        judgements = _mod().resolve_judgements(rows)
        assert len(judgements) == 2

    def test_rater_evidence_is_accepted(self) -> None:
        rows = [_row(conflict='YES'), _row('e2', home='YES', conflict='NO', loss='YES', digest='NO')]
        assert len(_mod().resolve_judgements(rows)) == 2


class TestNonConsumedAnswers:
    @pytest.mark.parametrize(('non_consumed', 'named'), [
        (['house'], 'house'),
        (['conflict'], 'conflict'),
        (['home', 'home'], 'home'),
        ('home', 'home'),
    ])
    def test_a_malformed_non_consumed_list_is_refused_naming_the_binary(
        self, non_consumed: Any, named: str,
    ) -> None:
        rows = [_row(rater='v1', role='derived', home='YES', conflict='NO'),
                _row(home='YES', non_consumed=non_consumed)]
        with pytest.raises(ValueError, match=f"row 1.*e1.*t1.*non_consumed.*'{named}'"):
            _mod().resolve_judgements(rows)

    def test_non_consumed_on_a_derived_row_is_refused(self) -> None:
        row = _row(rater='v1', role='derived', home='YES', non_consumed=['home'])
        with pytest.raises(ValueError, match="row 0.*non_consumed.*'home'.*derived"):
            _mod().resolve_judgements([row])

    def test_a_non_consumed_binary_no_derived_row_fixes_is_refused(self) -> None:
        rows = [_row(rater='v1', role='derived', conflict='NO'),
                _row(home='NO', conflict='NO', non_consumed=['home', 'conflict'])]
        with pytest.raises(ValueError, match="row 1.*e1.*t1.*'r1'.*non_consumed.*'home'"):
            _mod().resolve_judgements(rows)

    def test_a_voting_rater_answer_on_a_derived_binary_is_refused(self) -> None:
        rows = [_row(home='NO', conflict='NO', non_consumed=['conflict']),
                _row(rater='v1', role='derived', home='YES', conflict='NO')]
        with pytest.raises(ValueError, match="row 0.*e1.*t1.*'r1'.*home.*derived"):
            _mod().resolve_judgements(rows)

    def test_the_cross_row_rules_read_only_the_same_pair_key(self) -> None:
        rows = [_row(rater='v1', role='derived', home='YES'),
                _row(basis='attach_time', home='NO'),
                _row('e2', home='NO')]
        assert _resolved(rows, 'home', basis='attach_time') == _mod().Resolved.NO

    def test_a_non_consumed_answer_is_recorded_and_does_not_vote(self) -> None:
        rubric = _mod()
        rows = [_row(rater='v1', role='derived', home='YES', conflict='NO'),
                _row(home='NO', conflict='NO', non_consumed=['home', 'conflict'])]
        judgement = rubric.resolve_judgements(rows)[rubric.PairKey('e1', 't1', rubric.Basis.CURRENT)]
        rated = next(vote for vote in judgement.votes if vote.rater == 'r1')
        assert rated.non_consumed == {rubric.Binary.HOME, rubric.Binary.CONFLICT}
        assert dict(rated.answers) == {}
        assert judgement.answer(rubric.Binary.HOME) == rubric.Resolved.YES
        assert judgement.answer(rubric.Binary.CONFLICT) == rubric.Resolved.NO

    def test_a_partly_non_consumed_row_votes_with_the_rest(self) -> None:
        rows = [_row(rater='v1', role='derived', home='NO'),
                _row(home='YES', conflict='NO', loss='YES', non_consumed=['home']),
                _row(rater='r2', home='YES', conflict='NO', loss='YES', non_consumed=['home'])]
        assert _resolved(rows, 'loss') == _mod().Resolved.YES
        assert _resolved(rows, 'conflict') == _mod().Resolved.NO
        assert _resolved(rows, 'home') == _mod().Resolved.NO


_DISPUTED = {'home': 'YES', 'conflict': 'YES', 'scope': 'CENTRAL', 'action_changing': 'YES'}


class TestUnreachedFollowUps:
    """A follow-up no rater reached because a derived row overrode its precondition (esc-7174-4)."""

    def _judgement(self, rows: list[dict[str, Any]]) -> Any:
        rubric = _mod()
        return rubric.resolve_judgements(rows)[rubric.PairKey('e1', 't1', rubric.Basis.CURRENT)]

    def _overridden(self) -> list[dict[str, Any]]:
        """A held row fixing CONFLICT NO beside two raters who both said CONFLICT YES."""
        return [
            _row(rater='v1', role='derived', conflict='NO'),
            _row(**_DISPUTED, non_consumed=['conflict']),
            _row(rater='r2', role='second', **_DISPUTED, non_consumed=['conflict']),
        ]

    def test_loss_resolves_unclear_when_a_derived_conflict_no_overrode_the_raters(self) -> None:
        rubric = _mod()
        judgement = self._judgement(self._overridden())
        assert judgement.answer(rubric.Binary.HOME) == rubric.Resolved.YES
        assert judgement.answer(rubric.Binary.CONFLICT) == rubric.Resolved.NO
        assert judgement.answer(rubric.Binary.LOSS) == rubric.Resolved.UNCLEAR
        assert judgement.unreached_follow_ups == {rubric.Binary.LOSS}

    @pytest.mark.parametrize('outcome', ['restated', 'contested'])
    def test_an_item_reading_the_unreached_loss_is_unclear_not_incomplete(self, outcome: str) -> None:
        rubric = _mod()
        result = rubric.derive(outcome, {'t1': self._judgement(self._overridden())})
        assert isinstance(result, rubric.Unclear), result
        assert result.binaries == (('t1', 'loss'),)

    def test_a_transitive_follow_up_is_unreached_the_same_way(self) -> None:
        rubric = _mod()
        judgement = self._judgement([
            _row(rater='v1', role='derived', home='YES', conflict='NO', loss='YES'),
            _row(home='YES', conflict='NO', loss='NO', non_consumed=['home', 'conflict', 'loss']),
        ])
        assert judgement.answer(rubric.Binary.DIGEST) == rubric.Resolved.UNCLEAR
        assert judgement.unreached_follow_ups == {rubric.Binary.DIGEST}

    @pytest.mark.parametrize('role', ['primary', 'second', 'third'])
    def test_a_rater_row_meeting_the_preconditions_without_loss_is_refused(self, role: str) -> None:
        rows = [_row(rater='v1', role='derived', conflict='NO'),
                _row(rater='r2', role=role, home='YES', conflict='NO', non_consumed=['conflict'])]
        with pytest.raises(ValueError, match="row 1.*e1.*t1.*lacks loss.*home 'YES'.*conflict 'NO'"):
            _mod().resolve_judgements(rows)

    def test_a_rater_row_with_home_and_conflict_on_separate_rows_leaves_loss_open(self) -> None:
        rubric = _mod()
        judgement = self._judgement([_row(home='YES'), _row(rater='r2', role='second', conflict='NO')])
        assert judgement.answer(rubric.Binary.LOSS) == rubric.Resolved.OPEN
        assert judgement.unreached_follow_ups == frozenset()
        result = rubric.derive('restated', {'t1': judgement})
        assert isinstance(result, rubric.Incomplete), result
        assert [(gap.target_id, gap.binary, gap.kind) for gap in result.missing] == [
            ('t1', 'loss', 'open'),
        ]

    def test_a_derived_row_meeting_the_preconditions_needs_no_loss(self) -> None:
        rubric = _mod()
        judgement = self._judgement([_row(rater='v1', role='derived', home='YES', conflict='NO')])
        assert judgement.answer(rubric.Binary.LOSS) == rubric.Resolved.UNCLEAR
        assert judgement.unreached_follow_ups == {rubric.Binary.LOSS}

    @pytest.mark.parametrize('answers', [
        {'conflict': 'YES'}, {'home': 'YES', 'conflict': 'YES'},
    ])
    def test_a_rater_row_may_leave_out_the_report_only_follow_ups(
        self, answers: dict[str, str],
    ) -> None:
        rubric = _mod()
        judgement = self._judgement([_row(**answers)])
        assert judgement.answer(rubric.Binary.SCOPE) is None
        assert judgement.answer(rubric.Binary.ACTION_CHANGING) is None

    def test_a_tied_precondition_leaves_the_follow_up_missing(self) -> None:
        rubric = _mod()
        judgement = self._judgement([
            _row(rater='v1', role='derived', conflict='NO'),
            _row(**_DISPUTED, non_consumed=['conflict']),
            _row(rater='r2', role='second', **{**_DISPUTED, 'home': 'NO'}, non_consumed=['conflict']),
        ])
        assert judgement.answer(rubric.Binary.HOME) == rubric.Resolved.TIED
        assert judgement.answer(rubric.Binary.LOSS) is None
        assert judgement.unreached_follow_ups == frozenset()

    def test_a_derived_precondition_the_raters_also_met_resolves_from_their_votes(self) -> None:
        rubric = _mod()
        judgement = self._judgement([
            _row(rater='v1', role='derived', conflict='NO'),
            _row(home='YES', conflict='NO', loss='YES', non_consumed=['conflict']),
            _row(rater='r2', role='second', home='YES', conflict='NO', loss='YES',
                 non_consumed=['conflict']),
        ])
        assert judgement.answer(rubric.Binary.LOSS) == rubric.Resolved.YES
        assert judgement.unreached_follow_ups == frozenset()
        result = _decided(rubric.derive('restated', {'t1': judgement}))
        assert result.triage_cells == {'hidden_detail'}


class TestOpenFollowUps:
    """A decisive follow-up whose preconditions rater votes alone resolve, though no rater
    row met them all, goes to a third round (F3, task 7412)."""

    def _judgement(self, rows: list[dict[str, Any]]) -> Any:
        rubric = _mod()
        return rubric.resolve_judgements(rows)[rubric.PairKey('e1', 't1', rubric.Basis.CURRENT)]

    def _split(self) -> list[dict[str, Any]]:
        """A primary deciding HOME beside a second deciding CONFLICT, each abstaining on the other."""
        return [
            _row(home='YES', conflict='UNCLEAR', unclear_reason='cut where the claims meet'),
            _row(rater='r2', role='second', home='UNCLEAR', conflict='NO',
                 unclear_reason='cut at the subject'),
        ]

    def test_loss_resolves_open_when_the_votes_meet_its_preconditions_but_no_row_does(self) -> None:
        rubric = _mod()
        judgement = self._judgement(self._split())
        assert judgement.answer(rubric.Binary.HOME) == rubric.Resolved.YES
        assert judgement.answer(rubric.Binary.CONFLICT) == rubric.Resolved.NO
        assert judgement.answer(rubric.Binary.LOSS) == rubric.Resolved.OPEN
        assert judgement.unreached_follow_ups == frozenset()

    @pytest.mark.parametrize('outcome', ['restated', 'amended', 'contested'])
    def test_an_item_reading_the_open_loss_is_incomplete(self, outcome: str) -> None:
        rubric = _mod()
        result = rubric.derive(outcome, {'t1': self._judgement(self._split())})
        assert isinstance(result, rubric.Incomplete), result
        assert [(gap.target_id, gap.binary, gap.kind) for gap in result.missing] == [
            ('t1', 'loss', 'open'),
        ]

    def test_a_third_rater_reaching_loss_closes_the_gap(self) -> None:
        rubric = _mod()
        judgement = self._judgement([
            *self._split(),
            _row(rater='r3', role='third', home='YES', conflict='NO', loss='NO'),
        ])
        assert judgement.answer(rubric.Binary.LOSS) == rubric.Resolved.NO
        result = _decided(rubric.derive('restated', {'t1': judgement}))
        assert result.status_quo_cells == {'duplicate_clutter'}

    def test_a_split_that_is_not_an_abstention_opens_loss_too(self) -> None:
        rubric = _mod()
        judgement = self._judgement([
            _row(home='YES', conflict='YES'),
            _row(rater='r2', role='second', home='YES', conflict='UNCLEAR',
                 unclear_reason='cut where the claims meet'),
            _row(rater='r3', role='third', home='NO', conflict='NO'),
            _row(rater='r4', role='third', home='UNCLEAR', conflict='NO',
                 unclear_reason='cut at the subject'),
        ])
        assert judgement.answer(rubric.Binary.HOME) == rubric.Resolved.YES
        assert judgement.answer(rubric.Binary.CONFLICT) == rubric.Resolved.NO
        assert judgement.answer(rubric.Binary.LOSS) == rubric.Resolved.OPEN


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
        assert self._gaps(_derive('restated', {'home': 'YES'})) == [
            ('p', 'conflict', 'unrated'),
        ]

    def test_a_decisive_tie_is_incomplete(self) -> None:
        result = _derive('restated', {'home': 'YES', 'conflict': 'YES'}, {'conflict': 'NO'})
        assert self._gaps(result) == [('p', 'conflict', 'tied')]

    def test_an_unrated_pair_names_its_first_decisive_binary(self) -> None:
        assert self._gaps(_derive('contested')) == [('p', 'conflict', 'unrated')]

    def test_the_status_quo_regime_can_be_the_decisive_one(self) -> None:
        assert self._gaps(_derive('amended', {'home': 'YES'}, {'conflict': 'NO'})) == [
            ('p', 'loss', 'open'),
        ]

    def test_a_decisive_unclear_binary_makes_the_item_unclear(self) -> None:
        result = _derive('restated', {'home': 'YES', 'conflict': 'NO', 'loss': 'UNCLEAR',
                                      'unclear_reason': 'cut at the threshold'})
        assert isinstance(result, _mod().Unclear)
        assert result.binaries == (('p', 'loss'),)

    def test_an_unclear_status_quo_binary_leaves_out_a_decided_triage_reading(self) -> None:
        result = _derive('contested', {'home': 'YES', 'conflict': 'NO', 'loss': 'UNCLEAR',
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
        ('restated', {**_HOME_MORE, 'digest': 'NO'}, True),
        ('restated', {**_HOME_MORE, 'digest': 'YES'}, True),
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

    @pytest.mark.parametrize('outcome', ['restated', 'amended', 'contested'])
    def test_digest_is_decisive_only_in_the_band(self, outcome: str) -> None:
        in_band = _derive(outcome, _HOME_MORE, in_m1_band=True)
        assert isinstance(in_band, _mod().Incomplete)
        assert [(g.target_id, g.binary, g.kind) for g in in_band.missing] == [
            ('p', 'digest', 'unrated'),
        ]
        assert _decided(_derive(outcome, _HOME_MORE)).m1_hidden is None


class TestTheCounterfactualM1:
    @pytest.mark.parametrize(('outcome', 'vote', 'hidden', 'as_amended'), [
        ('restated', {**_HOME_MORE, 'digest': 'YES'}, True, False),
        ('restated', {**_HOME_MORE, 'digest': 'NO'}, True, True),
        ('restated', {'home': 'YES', 'conflict': 'YES'}, True, True),
        ('contested', {'home': 'YES', 'conflict': 'YES'}, False, True),
        ('restated', _HOME_FLAT, False, False),
        ('contested', _HOME_FLAT, False, False),
        ('amended', {**_HOME_MORE, 'digest': 'NO'}, True, True),
    ])
    def test_a_band_attach_is_read_as_if_amended(
        self, outcome: str, vote: dict[str, str], hidden: bool, as_amended: bool,
    ) -> None:
        result = _decided(_derive(outcome, vote, in_m1_band=True))
        assert result.m1_hidden is hidden
        assert result.m1_hidden_as_amended is as_amended

    def test_is_none_outside_the_band(self) -> None:
        assert _decided(_derive('restated', _HOME_FLAT)).m1_hidden_as_amended is None

    def test_is_none_for_a_stored_write(self) -> None:
        result = _decided(_mod().derive('stored', {}, in_m1_band=True))
        assert result.m1_hidden_as_amended is None


class TestConsumption:
    @pytest.mark.parametrize(('outcome', 'vote', 'options', 'consumed'), [
        ('contested', {'home': 'NO', 'conflict': 'YES'}, {}, {('p', 'conflict')}),
        ('contested', _HOME_FLAT, {}, {('p', 'conflict'), ('p', 'home'), ('p', 'loss')}),
        ('restated', {'home': 'NO', 'conflict': 'NO'}, {}, {('p', 'conflict'), ('p', 'home')}),
        ('amended', {**_HOME_MORE, 'digest': 'NO'}, {'in_m1_band': True},
         {('p', 'conflict'), ('p', 'home'), ('p', 'loss'), ('p', 'digest')}),
        ('amended', {**_HOME_MORE, 'digest': 'NO'}, {},
         {('p', 'conflict'), ('p', 'home'), ('p', 'loss')}),
    ])
    def test_names_every_binary_the_derivation_read(
        self, outcome: str, vote: dict[str, str], options: dict[str, Any],
        consumed: set[tuple[str, str]],
    ) -> None:
        assert _decided(_derive(outcome, vote, **options)).consumed == consumed

    def test_a_stored_write_names_every_earlier_record(self) -> None:
        result = _decided(_mod().derive('stored', {
            'e1': _judged('e1', {'conflict': 'YES'}),
            'e2': _judged('e2', _HOME_FLAT),
        }))
        assert result.consumed == {
            ('e1', 'conflict'), ('e2', 'conflict'), ('e2', 'home'), ('e2', 'loss'),
        }


class TestQuoteFound:
    @pytest.mark.parametrize(('quote', 'text'), [
        ('needle here', 'x needle here y'),
        ('needle   here', 'x needle\nhere y'),
        ('  needle here\n', 'x needle \t here y'),
    ])
    def test_finds_a_quote_up_to_whitespace(self, quote: str, text: str) -> None:
        assert _mod().quote_found(quote, text)

    @pytest.mark.parametrize('quote', ['the needle is here', 'Needle here', 'needle-here', '', '   '])
    def test_does_not_find_a_paraphrase_a_case_change_or_an_empty_quote(self, quote: str) -> None:
        assert not _mod().quote_found(quote, 'x needle here y')


_TAIL_WORDS = ' '.join(f'w{index:03d}' for index in range(60))
_LONG_CHILD = 'a' * 100 + ' needle here ' + _TAIL_WORDS
"""A child cut at 240: 'w010' lies wholly before the cut, 'w025' straddles it, 'w040' lies after."""


class TestDigestVisible:
    def test_the_fixture_straddles_the_cut_where_it_says(self) -> None:
        start = _LONG_CHILD.index('w025')
        assert start < 240 < start + len('w025')
        assert _LONG_CHILD.index('w040') > 240
        assert _LONG_CHILD.index('w010') + len('w010') < 240

    @pytest.mark.parametrize(('spans', 'visible'), [
        (['needle here'], True),
        (['needle   here', 'w010'], True),
        (['w040'], False),
        (['w025'], False),
        (['w024 w025'], False),
        (['needle here', 'w040'], False),
    ])
    def test_a_long_childs_spans_are_visible_only_before_the_cut(
        self, spans: list[str], visible: bool,
    ) -> None:
        assert _mod().digest_visible(spans, _LONG_CHILD) is visible

    @pytest.mark.parametrize('child', ['short text with a needle in it', 'b' * 236 + ' end'])
    def test_a_child_of_240_characters_or_fewer_shows_every_span(self, child: str) -> None:
        assert len(child) <= 240
        assert _mod().digest_visible([child.split()[-1]], child)

    def test_an_empty_span_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match='span'):
            _mod().digest_visible([], _LONG_CHILD)


class TestTheAllVisibleDigest:
    def _digest(self, *answers: str) -> Any:
        rows = [
            _row(rater=f'r{index}', home='YES', conflict='NO', loss='YES', digest=answer,
                 unclear_reason='spans not checked')
            for index, answer in enumerate(answers)
        ]
        return _resolved(rows, 'digest')

    @pytest.mark.parametrize(('answers', 'resolved'), [
        (('YES', 'NO'), 'NO'),
        (('NO', 'YES'), 'NO'),
        (('YES', 'YES'), 'YES'),
        (('NO',), 'NO'),
        (('YES',), 'YES'),
        (('UNCLEAR', 'YES'), 'YES'),
        (('UNCLEAR', 'NO'), 'NO'),
        (('UNCLEAR',), 'UNCLEAR'),
    ])
    def test_digest_is_yes_only_when_every_decided_vote_is_yes(
        self, answers: tuple[str, ...], resolved: str,
    ) -> None:
        assert self._digest(*answers) == _mod().Resolved(resolved)


class TestTheAbstainWord:
    @pytest.mark.parametrize('word', ['UNCLEAR', 'Answer', 'Scope', 'Resolved'])
    def test_is_one_word_across_every_vocabulary(self, word: str) -> None:
        rubric = _mod()
        value = word if word == 'UNCLEAR' else getattr(rubric, word).UNCLEAR
        assert rubric.is_abstain(value)

    @pytest.mark.parametrize('word', ['YES', 'NO', 'CENTRAL', 'SECONDARY', 'TIED', None, 'unclear'])
    def test_no_other_word_abstains(self, word: str | None) -> None:
        assert not _mod().is_abstain(word)
