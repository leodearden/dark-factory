"""Tests for write_triage_rating_rounds.py — the blind, double-rated rating rounds.

Synthetic rows and a tmp scratch only. The module is loaded by path
(``scripts/`` is not a package), lazily through ``_mod()``.
"""
from __future__ import annotations

import functools
import hashlib
import json
import sqlite3
import types
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

SCRIPTS = Path(__file__).parent.parent / 'scripts'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'write_triage_rating_rounds.py', mod_name='write_triage_rating_rounds',
    )


@functools.cache
def _rubric() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'write_triage_reader_rubric.py', mod_name='write_triage_reader_rubric',
    )


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


@pytest.fixture
def history(tmp_path: Path) -> Path:
    """p1 edited twice, p2 once, and p3 once by an UPDATE that kept no old text."""
    return _history_db(tmp_path / 'history.db', [
        ('p1', None, 'first', 'ADD', '2026-10-01T00:00:00+00:00'),
        ('p1', 'first', 'second', 'UPDATE', '2026-10-05T00:00:00+00:00'),
        ('p1', 'second', 'third', 'UPDATE', '2026-10-09T00:00:00+00:00'),
        ('p2', 'other', 'other v2', 'UPDATE', '2026-10-03T00:00:00+00:00'),
        ('p3', None, 'lost', 'UPDATE', '2026-10-03T00:00:00+00:00'),
    ])


class TestTextAt:

    @pytest.mark.parametrize(('instant', 'expected'), [
        ('2026-10-02T12:00:00.000001+00:00', 'first'),
        ('2026-10-07T00:00:00+00:00', 'second'),
        ('2026-10-10T00:00:00+00:00', 'third, current'),
    ])
    def test_is_the_text_before_the_first_edit_after_the_instant(
        self, history: Path, instant: str, expected: str,
    ) -> None:
        at = datetime.fromisoformat(instant)
        assert _mod().text_at(history, 'p1', at, 'third, current') == expected

    def test_an_edit_without_its_old_text_is_unrecoverable(self, history: Path) -> None:
        at = datetime.fromisoformat('2026-10-02T00:00:00+00:00')
        assert _mod().text_at(history, 'p3', at, 'lost') is None


class TestMemoryHistory:
    """MemoryHistory reads the UPDATE events once and answers as text_at does."""

    @pytest.mark.parametrize(('instant', 'expected'), [
        ('2026-10-02T12:00:00.000001+00:00', 'first'),
        ('2026-10-07T00:00:00+00:00', 'second'),
        ('2026-10-10T00:00:00+00:00', 'third, current'),
    ])
    def test_answers_as_text_at_does(self, history: Path, instant: str, expected: str) -> None:
        at = datetime.fromisoformat(instant)
        rounds = _mod()
        assert rounds.MemoryHistory.read(history).text_at('p1', at, 'third, current') == expected
        assert rounds.text_at(history, 'p1', at, 'third, current') == expected

    def test_an_edit_without_its_old_text_is_unrecoverable(self, history: Path) -> None:
        at = datetime.fromisoformat('2026-10-02T00:00:00+00:00')
        assert _mod().MemoryHistory.read(history).text_at('p3', at, 'lost') is None

    @pytest.mark.parametrize(('memory_id', 'instant', 'edited'), [
        ('p1', '2026-10-07T00:00:00+00:00', True),
        ('p1', '2026-10-10T00:00:00+00:00', False),
        ('p3', '2026-10-02T00:00:00+00:00', True),
        ('never-edited', '2026-10-01T00:00:00+00:00', False),
    ])
    def test_edited_after_is_whether_an_update_follows_the_instant(
        self, history: Path, memory_id: str, instant: str, edited: bool,
    ) -> None:
        at = datetime.fromisoformat(instant)
        assert _mod().MemoryHistory.read(history).edited_after(memory_id, at) is edited

    def test_a_history_read_for_some_memories_refuses_any_other(self, history: Path) -> None:
        partial = _mod().MemoryHistory.read(history, memory_ids=['p2'])
        at = datetime.fromisoformat('2026-10-02T00:00:00+00:00')
        assert partial.text_at('p2', at, 'other v2') == 'other'
        with pytest.raises(ValueError, match='p1'):
            partial.text_at('p1', at, 'third, current')
        with pytest.raises(ValueError, match='p1'):
            partial.edited_after('p1', at)


class TestCapped:
    def test_a_text_within_the_cap_is_unchanged(self) -> None:
        assert _mod().capped('x' * 4000) == 'x' * 4000

    def test_a_longer_text_is_cut_at_the_cap_and_marked_with_its_length(self) -> None:
        assert _mod().capped('x' * 4500) == 'x' * 4000 + '…[truncated, 4500 chars total]'

    def test_capping_a_capped_text_changes_nothing(self) -> None:
        once = _mod().capped('y' * 9000)
        assert _mod().capped(once) == once

    def test_a_shard_of_a_capped_text_shows_the_original_length(self) -> None:
        rounds = _mod()
        entry = rounds.ShardEntry('e', 't', 'current', 'df', 'child', rounds.capped('p' * 4500))
        text = rounds.shard([entry], name='primary_00').text
        assert '…[truncated, 4500 chars total]' in text
        assert text.count('…[truncated,') == 1


CHILD = 'The deploy uses blue-green.\nRollback takes   five minutes.'
PARENT = 'The deploy uses canary releases. Rollback is manual.'


def _line(**fields: Any) -> dict[str, Any]:
    return {'id': 'primary_00-000', 'reason': 'two claims', **fields}


class TestValidateRating:
    def _problems(self, line: dict[str, Any]) -> list[str]:
        return _mod().validate_rating(line, CHILD, PARENT)

    def test_verbatim_quotes_and_spans_pass(self) -> None:
        assert self._problems(_line(
            home='YES', conflict='YES', scope='CENTRAL', action_changing='YES',
            conflict_quote={'parent': 'The deploy uses canary releases.',
                            'child': 'The deploy uses blue-green.'},
        )) == []
        assert self._problems(_line(
            home='YES', conflict='NO', loss='YES', loss_spans=['Rollback takes five minutes.'],
        )) == []
        assert self._problems(_line(home='NO')) == []

    def test_a_quote_differing_only_in_whitespace_passes(self) -> None:
        assert self._problems(_line(
            home='YES', conflict='YES', scope='CENTRAL',
            conflict_quote={'parent': 'canary\n  releases.', 'child': 'blue-green. Rollback takes five'},
        )) == []

    @pytest.mark.parametrize(('field', 'quote', 'conflict_quote'), [
        ('conflict_quote.child', 'The deploy is blue-green.',
         {'parent': 'canary releases', 'child': 'The deploy is blue-green.'}),
        ('conflict_quote.child', 'Rollback is manual.',
         {'parent': 'canary releases', 'child': 'Rollback is manual.'}),
        ('conflict_quote.parent', 'The deploy uses blue-green.',
         {'parent': 'The deploy uses blue-green.', 'child': 'blue-green'}),
    ])
    def test_a_conflict_quote_not_in_its_text_fails(
        self, field: str, quote: str, conflict_quote: dict[str, str],
    ) -> None:
        problems = self._problems(_line(
            home='YES', conflict='YES', scope='CENTRAL', conflict_quote=conflict_quote,
        ))
        assert len(problems) == 1
        assert field in problems[0] and quote in problems[0]

    def test_each_loss_span_must_be_in_child(self) -> None:
        problems = self._problems(_line(
            home='YES', conflict='NO', loss='YES',
            loss_spans=['Rollback takes five minutes.', 'Rollback is manual.', 'uses canary'],
        ))
        assert len(problems) == 2
        assert 'loss_spans' in problems[0] and 'Rollback is manual.' in problems[0]
        assert 'loss_spans' in problems[1] and 'uses canary' in problems[1]

    def test_a_rater_line_carrying_digest_fails(self) -> None:
        problems = self._problems(_line(
            home='YES', conflict='NO', loss='YES', loss_spans=['five minutes'], digest='YES',
        ))
        assert len(problems) == 1
        assert 'digest is computed from loss_spans' in problems[0]


def _quoting(row: dict[str, Any], **evidence: Any) -> dict[str, Any]:
    """*row* with its evidence fields set to *evidence*."""
    return {**row, **evidence}


def _without(row: dict[str, Any], field: str) -> dict[str, Any]:
    return {name: value for name, value in row.items() if name != field}


_VERBATIM_QUOTE = {'parent': 'The deploy uses canary releases.', 'child': 'The deploy uses blue-green.'}
_DISPUTED = {'home': 'YES', 'conflict': 'YES', 'scope': 'CENTRAL', 'action_changing': 'YES'}


class TestRatingProblems:
    def _problems(self, row: dict[str, Any]) -> list[str]:
        return _mod().rating_problems(row, CHILD, PARENT)

    def test_a_valid_line_with_verbatim_evidence_has_none(self) -> None:
        assert self._problems(_quoting(_rated('e1', **_DISPUTED), conflict_quote=_VERBATIM_QUOTE)) == []
        assert self._problems(_quoting(
            _rated('e1', home='YES', conflict='NO', loss='YES'),
            loss_spans=['Rollback takes five minutes.'],
        )) == []

    @pytest.mark.parametrize(('row', 'fragment'), [
        (_rated('e1', home='YES', conflict='NO'), 'lacks loss'),
        (_without(_rated('e1', conflict='YES', scope='CENTRAL', action_changing='YES'),
                  'conflict_quote'), 'conflict_quote'),
        (_without(_rated('e1', home='YES', conflict='NO', loss='YES'), 'loss_spans'),
         'loss_spans'),
        ({**_rated('e1', conflict='NO'), 'home': ['YES']}, "home ['YES']"),
    ])
    def test_a_line_breaking_the_row_contract_names_that_alone(
        self, row: dict[str, Any], fragment: str,
    ) -> None:
        problems = self._problems(row)
        assert len(problems) == 1
        assert fragment in problems[0]

    def test_a_contract_valid_line_with_unfound_evidence_names_it(self) -> None:
        problems = self._problems(_quoting(
            _rated('e1', **_DISPUTED),
            conflict_quote={**_VERBATIM_QUOTE, 'child': 'The deploy is blue-green.'},
        ))
        assert len(problems) == 1
        assert 'conflict_quote.child' in problems[0]

    def test_a_cross_family_line_is_checked_against_the_same_contract(self) -> None:
        line = {**_rated('e1', home='YES', conflict='NO'), 'role': 'cross_family'}
        assert ['lacks loss' in problem for problem in self._problems(line)] == [True]
        assert self._problems({**line, 'loss': 'NO'}) == []

    def test_a_contract_violation_hides_the_unfound_evidence_beside_it(self) -> None:
        problems = self._problems(_quoting(
            _rated('e1', home='YES', conflict='NO'), loss_spans=['not in the child'],
        ))
        assert len(problems) == 1
        assert 'lacks loss' in problems[0]
        assert 'not in the child' not in problems[0]


class TestEffectiveRatings:
    def test_the_highest_round_of_each_rater_on_each_pair_and_basis_survives(self) -> None:
        reasked = {**_rated('e1', role='primary', home='YES'), 'batch': 'reask_01_primary'}
        rows = [
            {**reasked, 'round': 1},
            {**_rated('e1', role='primary', home='NO'), 'round': 0},
            {**_rated('e1', role='second', home='NO'), 'round': 0},
            {**_rated('e1', role='primary', basis='attach_time', home='NO'), 'round': 0},
        ]
        effective = _mod().effective_ratings(rows)
        assert sorted(effective, key=lambda row: (row['basis'], row['rater'])) == [
            _rated('e1', role='primary', basis='attach_time', home='NO'),
            reasked,
            _rated('e1', role='second', home='NO'),
        ]

    def test_two_lines_of_one_rater_in_one_round_are_refused(self) -> None:
        row = {**_rated('e1', role='primary', home='NO'), 'round': 0}
        with pytest.raises(ValueError, match='twice in round 0'):
            _mod().effective_ratings([row, row])


EARLY = 'Early detail sits here.'
LATE = 'Late detail sits here.'
LONG_CHILD = f'{EARLY} {"x" * 300} {LATE}'


class TestWithDigest:
    def _loss_row(self, *spans: str) -> dict[str, Any]:
        return {**_rated('e1', home='YES', conflict='NO', loss='YES'), 'loss_spans': list(spans)}

    def test_a_loss_yes_row_whose_spans_the_digest_shows_gets_digest_yes(self) -> None:
        row = self._loss_row(EARLY)
        assert _mod().with_digest(row, LONG_CHILD) == {**row, 'digest': 'YES'}

    def test_a_loss_yes_row_with_a_span_past_the_digest_gets_digest_no(self) -> None:
        row = self._loss_row(EARLY, LATE)
        assert _mod().with_digest(row, LONG_CHILD) == {**row, 'digest': 'NO'}

    @pytest.mark.parametrize('row', [
        _rated('e1', home='YES', conflict='NO', loss='NO'),
        _rated('e1', home='NO'),
        _rated('e1', home='YES', conflict='YES', scope='CENTRAL', action_changing='YES'),
    ])
    def test_any_other_row_is_left_alone(self, row: dict[str, Any]) -> None:
        assert _mod().with_digest(row, LONG_CHILD) == row

    def test_the_given_row_is_not_changed(self) -> None:
        row = self._loss_row(EARLY)
        before = dict(row)
        _mod().with_digest(row, LONG_CHILD)
        assert row == before


def _gap(target: str, binary: str, kind: str) -> Any:
    rubric = _rubric()
    return rubric.Gap(target, rubric.Binary(binary), rubric.GapKind(kind))


def _incomplete(*gaps: Any) -> Any:
    return _rubric().Incomplete(tuple(sorted(gaps)))


def _decided() -> Any:
    return _rubric().Decided(
        triage_cells=frozenset(), status_quo_cells=frozenset(), m1_hidden=None,
        conflict=False, m1_hidden_as_amended=None, consumed=frozenset(),
    )


class TestThirdRoundBlockers:
    def test_names_each_pair_a_tie_or_an_open_follow_up_alone_blocks(self) -> None:
        derivations = [
            ('tied', _incomplete(_gap('p1', 'conflict', 'tied'))),
            ('open', _incomplete(_gap('p2', 'loss', 'open'))),
            ('both', _incomplete(_gap('p3', 'home', 'tied'), _gap('p3', 'loss', 'open'))),
            ('decided', _decided()),
            ('unclear', _rubric().Unclear(binaries=(('p4', _rubric().Binary.HOME),))),
        ]
        assert _mod().third_round_blockers(derivations) == [
            ('both', 'p3'), ('open', 'p2'), ('tied', 'p1'),
        ]

    def test_a_multi_target_derivation_blocks_only_its_gap_targets(self) -> None:
        derivations = [('stored', _incomplete(_gap('t2', 'home', 'tied'), _gap('t5', 'conflict', 'tied')))]
        assert _mod().third_round_blockers(derivations) == [('stored', 't2'), ('stored', 't5')]

    def test_one_entry_may_come_once_per_pair(self) -> None:
        derivations = [('w', _incomplete(_gap('t1', 'home', 'tied'))), ('w', _decided()),
                       ('w', _incomplete(_gap('t3', 'loss', 'open')))]
        assert _mod().third_round_blockers(derivations) == [('w', 't1'), ('w', 't3')]

    def test_an_unrated_decisive_binary_is_refused_naming_the_entry_and_binaries(self) -> None:
        derivations = [
            ('fine', _incomplete(_gap('p1', 'conflict', 'tied'))),
            ('unrated-entry', _incomplete(_gap('p1', 'home', 'tied'), _gap('p1', 'conflict', 'unrated'))),
        ]
        with pytest.raises(ValueError, match=r"'unrated-entry'.*\['conflict'\]"):
            _mod().third_round_blockers(derivations)

    def test_an_unrated_gap_after_a_tie_on_one_pair_is_refused(self) -> None:
        derivation = _incomplete(_gap('p1', 'conflict', 'tied'), _gap('p1', 'loss', 'unrated'))
        assert [str(gap.kind) for gap in derivation.missing] == ['tied', 'unrated']
        with pytest.raises(ValueError, match=r"'w'.*\['loss'\]"):
            _mod().third_round_blockers([('w', derivation)])

    def test_nothing_blocked_is_an_empty_list(self) -> None:
        assert _mod().third_round_blockers([('decided', _decided())]) == []


class TestLossConsumingPairs:
    def test_names_each_current_pair_resolving_home_yes_and_conflict_no(self) -> None:
        kept = {'home': 'YES', 'conflict': 'NO', 'loss': 'NO'}
        rows = [
            _rated('agree', role='primary', **kept), _rated('agree', role='second', **kept),
            _rated('third-resolved', role='primary', home='YES', conflict='YES'),
            _rated('third-resolved', role='second', **kept),
            _rated('third-resolved', role='third', home='YES', conflict='NO', loss='YES'),
            _rated('conflict', role='primary', home='YES', conflict='YES'),
            _rated('conflict', role='second', home='YES', conflict='YES'),
            _rated('tied', role='primary', home='YES', conflict='YES'),
            _rated('tied', role='second', **kept),
            _rated('home-no', role='primary', home='NO', conflict='NO'),
            _rated('home-no', role='second', home='NO', conflict='NO'),
            _rated('attach', role='primary', basis='attach_time', **kept),
            _rated('attach', role='second', basis='attach_time', **kept),
        ]
        assert _mod().loss_consuming_pairs(rows) == [('agree', 'p1'), ('third-resolved', 'p1')]


class TestShard:
    def _shard(self) -> tuple[Any, str, str]:
        child = 'c' * 300 + 'tail of the child ' * 300
        parent = 'p' * 4500
        rounds = _mod()
        shard = rounds.shard([
            rounds.ShardEntry('entry-one', 'target-one', 'current', 'df', child, parent),
            rounds.ShardEntry('entry-two', 'target-two', 'attach_time', 'reify', 'short', 'old'),
        ], name='primary_00')
        return shard, child, parent

    def test_caps_each_text_and_marks_the_cut(self) -> None:
        assert _mod().TEXT_CAP == 4000
        shard, child, parent = self._shard()
        assert parent[:4000] + '…[truncated, 4500 chars total]' in shard.text
        assert parent[:4001] not in shard.text
        assert child[:4000] + f'…[truncated, {len(child)} chars total]' in shard.text

    def test_shows_no_digest_because_raters_are_not_asked_digest(self) -> None:
        shard, _, _ = self._shard()
        assert 'DIGEST' not in shard.text

    def test_item_ids_are_opaque_and_the_key_maps_them_back(self) -> None:
        shard, _, _ = self._shard()
        assert 'entry-one' not in shard.text and 'target-one' not in shard.text
        assert sorted(shard.key.values(), key=lambda entry: entry['entry_id']) == [
            {'entry_id': 'entry-one', 'target_id': 'target-one', 'basis': 'current'},
            {'entry_id': 'entry-two', 'target_id': 'target-two', 'basis': 'attach_time'},
        ]
        for item_id in shard.key:
            assert f'=== ITEM id={item_id}' in shard.text

    def test_an_attach_time_item_says_so(self) -> None:
        shard, _, _ = self._shard()
        assert 'basis=attach_time' in shard.text


def _entries(count: int) -> list[Any]:
    rounds = _mod()
    return [
        rounds.ShardEntry(f'e{index:02d}', f't{index:02d}', 'current', 'df', 'child', 'parent')
        for index in range(count)
    ]


class TestPlanShards:
    def _plan(self, entries: list[Any], role: str = 'primary', **overrides: Any) -> list[Any]:
        options: dict[str, Any] = {'prefix': '', 'seed': 7, 'first_index': 0, **overrides}
        return _mod().plan_shards(entries, role=role, **options)

    def test_every_entry_lands_in_exactly_one_shard_of_at_most_shard_size(self) -> None:
        rounds = _mod()
        assert rounds.SHARD_SIZE == 40
        entries = _entries(95)
        plan = self._plan(entries)
        planned = [entry for _, members in plan for entry in members]
        assert sorted(planned, key=lambda entry: entry.entry_id) == entries
        assert all(len(members) <= rounds.SHARD_SIZE for _, members in plan)

    def test_names_count_from_the_first_index(self) -> None:
        assert [name for name, _ in self._plan(_entries(50))] == ['primary_00', 'primary_01']
        assert [name for name, _ in self._plan(_entries(50), 'second', prefix='attach_',
                                               first_index=3)] == [
            'attach_second_03', 'attach_second_04',
        ]

    def test_each_role_reads_the_entries_in_its_own_order(self) -> None:
        def order(role: str) -> list[str]:
            return [entry.entry_id for _, members in self._plan(_entries(50), role)
                    for entry in members]

        assert order('primary') != order('second')

    def test_the_plan_is_deterministic(self) -> None:
        assert self._plan(_entries(50)) == self._plan(list(reversed(_entries(50))))

    def test_every_rater_is_opus(self) -> None:
        assert dict(_mod().RATER_MODELS) == {'primary': 'opus', 'second': 'opus', 'third': 'opus'}


IDENTITY = {'entry_id': 'e1', 'target_id': 'p1', 'basis': 'current', 'rater': 'opus-primary',
            'role': 'primary', 'batch': 'primary_00'}


class TestRaterRow:
    def test_is_the_answer_under_its_identity(self) -> None:
        answer = {'home': 'NO', 'conflict': 'NO', 'reason': 'r'}
        assert _mod().rater_row(IDENTITY, answer) == {**answer, **IDENTITY}

    @pytest.mark.parametrize('key', sorted(IDENTITY))
    def test_an_answer_carrying_an_identity_key_is_refused_naming_it(self, key: str) -> None:
        with pytest.raises(ValueError, match=rf"\['{key}'\]"):
            _mod().rater_row(IDENTITY, {'home': 'NO', key: 'x'})


class TestRatingVolume:
    def test_counts_distinct_pairs_per_basis_and_role(self) -> None:
        def item(entry: str, basis: str = 'current') -> dict[str, str]:
            return {'entry_id': entry, 'target_id': 'p1', 'basis': basis}

        key = {
            'shards': {
                'primary_00': {'role': 'primary', 'items': ['a', 'b']},
                'reask_01_primary': {'role': 'primary', 'items': ['c']},
                'second_00': {'role': 'second', 'items': ['d']},
                'attach_primary_00': {'role': 'primary', 'items': ['e']},
                'third_00': {'role': 'third', 'items': ['f']},
            },
            'items': {'a': item('e1'), 'b': item('e2'), 'c': item('e1'), 'd': item('e1'),
                      'e': item('e1', 'attach_time'), 'f': item('e2')},
        }
        assert _mod().rating_volume(key) == {
            'current': {'primary': 2, 'second': 1, 'third': 1}, 'attach_time': {'primary': 1},
        }


Triple = tuple[str, str, str]
Rule = Callable[[Any], dict[str, Any]]


def _honest(entry: Any) -> dict[str, Any]:
    return {'home': 'YES', 'conflict': 'YES', 'scope': 'CENTRAL', 'action_changing': 'YES',
            'conflict_quote': {'parent': entry.parent_text, 'child': entry.child_text}}


def _lacks_loss(_: Any) -> dict[str, Any]:
    return {'home': 'YES', 'conflict': 'NO'}


def _unfound_quote(entry: Any) -> dict[str, Any]:
    return {**_honest(entry), 'conflict_quote': {'parent': entry.parent_text, 'child': 'not in it'}}


PROJECTS = {'e0': 'df', 'e1': 'reify', 'e2': 'df'}


def _scratch_entries() -> list[Any]:
    rounds = _mod()
    entries = [
        rounds.ShardEntry(entry_id, 'p1', 'current', project, f'child of {entry_id}',
                          f'parent of {entry_id}')
        for entry_id, project in PROJECTS.items()
    ]
    entries.append(rounds.ShardEntry('e0', 'p1', 'attach_time', 'df', 'child of e0',
                                     'parent of e0 as it read then'))
    return entries


def _triple(item: dict[str, str]) -> Triple:
    return (item['entry_id'], item['target_id'], item['basis'])


class _Rounds:
    """A tmp scratch with every scratch entry sharded, and a fake rater for its shards."""

    def __init__(self, root: Path) -> None:
        rounds = _mod()
        self.root = root / 'scratch'
        self.scratch = rounds.RatingScratch(self.root)
        key = self.scratch.load_key()
        self.added = self.scratch.add_shards(key, _scratch_entries(), seed=7)
        self.scratch.save_key(key)
        self.scratch.save_texts(_scratch_entries())

    def key(self) -> dict[str, Any]:
        return self.scratch.load_key()

    def entries(self) -> dict[Triple, Any]:
        return self.scratch.entries(PROJECTS)

    def shard_files(self) -> dict[str, bytes]:
        return {path.name: path.read_bytes() for path in (self.root / 'shards').iterdir()}

    def holding(self, triple: Triple) -> list[str]:
        key = self.key()
        return sorted(name for name, spec in key['shards'].items()
                      if any(_triple(key['items'][item]) == triple for item in spec['items']))

    def rate(self, overrides: dict[tuple[str, Triple], Rule] | None = None) -> None:
        """Write a ratings file for each shard without one; *overrides* maps (role, triple) to a rule."""
        key, entries = self.key(), self.entries()
        ratings = self.root / 'ratings'
        ratings.mkdir(exist_ok=True)
        for name, spec in key['shards'].items():
            path = ratings / f'{name}.jsonl'
            if path.exists():
                continue
            lines = []
            for item_id in spec['items']:
                triple = _triple(key['items'][item_id])
                rule = (overrides or {}).get((spec['role'], triple), _honest)
                lines.append({'id': item_id, **rule(entries[triple]), 'reason': 'names both claims'})
            path.write_text(''.join(json.dumps(line) + '\n' for line in lines))


ALL_TRIPLES = {('e0', 'p1', 'current'), ('e1', 'p1', 'current'), ('e2', 'p1', 'current'),
               ('e0', 'p1', 'attach_time')}


class TestRatingScratch:
    @pytest.fixture
    def rounds(self, tmp_path: Path) -> _Rounds:
        return _Rounds(tmp_path)

    def test_add_shards_deals_every_entry_into_one_primary_and_one_second_shard(
        self, rounds: _Rounds,
    ) -> None:
        key = rounds.key()
        assert {_triple(item) for item in key['items'].values()} == ALL_TRIPLES
        for triple in ALL_TRIPLES:
            roles = sorted(key['shards'][name]['role'] for name in rounds.holding(triple))
            assert roles == ['primary', 'second'], triple
            attach = [name.startswith('attach_') for name in rounds.holding(triple)]
            assert attach == [triple[2] == 'attach_time'] * 2, triple
        for name, spec in key['shards'].items():
            assert spec == {'role': spec['role'], 'rater': f'opus-{spec["role"]}', 'round': 0,
                            'items': spec['items'], 'brief_sha256': _sha(_mod().BRIEF_PATH)}
            assert (rounds.root / 'shards' / f'{name}.txt').exists()
        assert sorted(rounds.added) == sorted(key['shards'])

    def test_a_rerun_adds_nothing(self, rounds: _Rounds) -> None:
        key, shard_files = rounds.key(), rounds.shard_files()
        assert rounds.scratch.add_shards(key, _scratch_entries(), seed=7) == []
        assert key == rounds.key()
        assert rounds.shard_files() == shard_files

    def test_a_new_entry_gets_shards_of_its_own(self, rounds: _Rounds) -> None:
        key = rounds.key()
        extra = _mod().ShardEntry('e9', 'p1', 'current', 'df', 'child of e9', 'parent of e9')
        assert rounds.scratch.add_shards(key, [*_scratch_entries(), extra], seed=7) == [
            'primary_01', 'second_01',
        ]
        assert [_triple(key['items'][item]) for item in key['shards']['primary_01']['items']] == [
            ('e9', 'p1', 'current'),
        ]

    def test_texts_hold_each_entry_as_its_raters_saw_it(self, rounds: _Rounds) -> None:
        entries = rounds.entries()
        assert set(entries) == ALL_TRIPLES
        assert entries['e1', 'p1', 'current'] == _mod().ShardEntry(
            'e1', 'p1', 'current', 'reify', 'child of e1', 'parent of e1')
        assert entries['e0', 'p1', 'attach_time'].parent_text == 'parent of e0 as it read then'

    def test_a_deleted_shard_text_is_rendered_again_byte_for_byte(self, rounds: _Rounds) -> None:
        shard_files = rounds.shard_files()
        (rounds.root / 'shards' / 'primary_00.txt').unlink()
        (rounds.root / 'shards' / 'attach_second_00.txt').unlink()
        assert rounds.scratch.render_missing(rounds.key(), rounds.entries()) == [
            'attach_second_00', 'primary_00',
        ]
        assert rounds.shard_files() == shard_files
        assert rounds.scratch.render_missing(rounds.key(), rounds.entries()) == []

    def test_rendering_a_shard_whose_entry_is_gone_is_refused(self, rounds: _Rounds) -> None:
        (rounds.root / 'shards' / 'primary_00.txt').unlink()
        entries = {triple: entry for triple, entry in rounds.entries().items() if triple[0] != 'e1'}
        with pytest.raises(ValueError, match=r'primary_00.*e1'):
            rounds.scratch.render_missing(rounds.key(), entries)

    def test_ratings_are_the_effective_lines_and_those_to_ask_again(self, rounds: _Rounds) -> None:
        broken, unfound = ('e1', 'p1', 'current'), ('e2', 'p1', 'current')
        rounds.rate({('primary', broken): _lacks_loss, ('second', unfound): _unfound_quote})
        rated = rounds.scratch.ratings(rounds.key(), rounds.entries())
        assert sorted((_triple(row), row['role']) for row in rated.rows) == sorted(
            (triple, role) for triple in ALL_TRIPLES for role in ('primary', 'second'))
        assert rated.reasked == 0
        reasks = {(_triple(reask.row), reask.row['role']): reask for reask in rated.reask}
        assert set(reasks) == {(broken, 'primary'), (unfound, 'second')}
        assert 'lacks loss' in reasks[broken, 'primary'].problems[0]
        assert 'conflict_quote.child' in reasks[unfound, 'second'].problems[0]
        assert reasks[broken, 'primary'].item_id in str(reasks[broken, 'primary'])

    def test_a_rating_line_carrying_an_identity_key_is_refused_naming_it(
        self, rounds: _Rounds,
    ) -> None:
        rounds.rate()
        path = rounds.root / 'ratings' / 'primary_00.jsonl'
        first, *rest = path.read_text().splitlines()
        path.write_text('\n'.join([json.dumps({**json.loads(first), 'role': 'second'}), *rest]) + '\n')
        with pytest.raises(ValueError, match=r"primary_00\.jsonl:1.*\['role'\]"):
            rounds.scratch.ratings(rounds.key(), rounds.entries())

    def test_a_shard_without_ratings_is_refused(self, rounds: _Rounds) -> None:
        rounds.rate()
        (rounds.root / 'ratings' / 'second_00.jsonl').unlink()
        with pytest.raises(ValueError, match='second_00'):
            rounds.scratch.ratings(rounds.key(), rounds.entries())

    def test_a_reask_round_writes_one_shard_per_rater_and_the_later_round_supersedes(
        self, rounds: _Rounds,
    ) -> None:
        broken, unfound = ('e1', 'p1', 'current'), ('e2', 'p1', 'current')
        rounds.rate({('primary', broken): _lacks_loss, ('second', unfound): _unfound_quote})
        key = rounds.key()
        assert _mod().reask_round(rounds.scratch, key, rounds.entries()) == {
            'lines': 2, 'shards': {'reask_01_primary': 1, 'reask_02_second': 1},
        }
        key = rounds.key()
        assert {name: (spec['role'], spec['rater'], spec['round'])
                for name, spec in key['shards'].items() if name.startswith('reask_')} == {
            'reask_01_primary': ('primary', 'opus-primary', 1),
            'reask_02_second': ('second', 'opus-second', 2),
        }
        rounds.rate()
        rated = rounds.scratch.ratings(rounds.key(), rounds.entries())
        assert rated.reask == []
        assert rated.reasked == 2
        batches = {(_triple(row), row['role']): row['batch'] for row in rated.rows}
        assert batches[broken, 'primary'] == 'reask_01_primary'
        assert batches[unfound, 'second'] == 'reask_02_second'
        shard_files = rounds.shard_files()
        assert _mod().reask_round(rounds.scratch, rounds.key(), rounds.entries()) == {
            'lines': 0, 'shards': {},
        }
        assert rounds.shard_files() == shard_files

    def test_refuse_unfinished_names_lines_awaiting_a_reask(self, rounds: _Rounds) -> None:
        rounds.rate({('primary', ('e1', 'p1', 'current')): _lacks_loss})
        rated = rounds.scratch.ratings(rounds.key(), rounds.entries())
        [reask] = rated.reask
        with pytest.raises(ValueError, match='reask') as refusal:
            _mod().refuse_unfinished(rated, rounds.entries())
        assert reask.item_id in str(refusal.value) and 'lacks loss' in str(refusal.value)

    def test_refuse_unfinished_names_pairs_lacking_a_primary_or_second_line(
        self, rounds: _Rounds,
    ) -> None:
        rounds.rate()
        rated = rounds.scratch.ratings(rounds.key(), rounds.entries())
        _mod().refuse_unfinished(rated, rounds.entries())
        extra = _mod().ShardEntry('e9', 'p1', 'current', 'df', 'child of e9', 'parent of e9')
        entries = {**rounds.entries(), ('e9', 'p1', 'current'): extra}
        with pytest.raises(ValueError, match=r"'e9'.*'primary'.*'e9'.*'second'"):
            _mod().refuse_unfinished(rated, entries)

    def test_a_third_round_shards_the_blocked_current_pairs_for_a_new_rater(
        self, rounds: _Rounds,
    ) -> None:
        third = _mod().third_round
        key = rounds.key()
        assert third(rounds.scratch, key, rounds.entries(), {('e0', 'p1')}) == {
            'pairs': 1, 'shards': ['third_00'],
        }
        key = rounds.key()
        assert {**key['shards']['third_00'], 'items': len(key['shards']['third_00']['items'])} == {
            'role': 'third', 'rater': 'opus-third-00', 'round': 0, 'items': 1,
            'brief_sha256': _sha(_mod().BRIEF_PATH),
        }
        assert rounds.holding(('e0', 'p1', 'current'))[-1] == 'third_00'
        assert 'third_00' not in rounds.holding(('e0', 'p1', 'attach_time'))
        assert third(rounds.scratch, key, rounds.entries(), {('e1', 'p1'), ('e2', 'p1')}) == {
            'pairs': 2, 'shards': ['third_01'],
        }
        assert rounds.key()['shards']['third_01']['rater'] == 'opus-third-01'

    def test_a_third_round_with_nothing_blocked_writes_nothing(self, rounds: _Rounds) -> None:
        key_bytes = (rounds.root / 'key.json').read_bytes()
        assert _mod().third_round(rounds.scratch, rounds.key(), rounds.entries(), set()) == {
            'pairs': 0, 'shards': [],
        }
        assert (rounds.root / 'key.json').read_bytes() == key_bytes


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def old_brief(tmp_path: Path) -> Path:
    path = tmp_path / 'brief_v2_1.md'
    path.write_text('# Rater brief v2.1: an older wording\n\nAnswer LOSS the old way.\n')
    return path


class TestTheBrief:
    def test_the_contract_and_the_prompt_render_the_brief_they_are_given(
        self, old_brief: Path,
    ) -> None:
        rounds = _mod()
        contract = rounds.answer_contract(old_brief)
        assert old_brief.read_text() in contract
        assert rounds.BRIEF_PATH.read_text() not in contract
        assert contract in rounds.rater_prompt(old_brief)
        assert rounds.answer_contract() == rounds.answer_contract(rounds.BRIEF_PATH)
        assert rounds.brief_sha256(old_brief) == _sha(old_brief)

    def test_every_shard_records_the_sha256_of_its_scratchs_brief(
        self, tmp_path: Path, old_brief: Path,
    ) -> None:
        scratch = _mod().RatingScratch(tmp_path / 'scratch', brief=old_brief)
        key = scratch.load_key()
        scratch.add_shards(key, _scratch_entries(), seed=7)
        assert {spec['brief_sha256'] for spec in key['shards'].values()} == {_sha(old_brief)}

    def test_a_line_under_a_brief_neither_round_0s_nor_a_covering_rerates_is_refused(
        self, tmp_path: Path, old_brief: Path,
    ) -> None:
        rounds = _Rounds(tmp_path)
        rounds.rate()
        other = _mod().RatingScratch(rounds.root, brief=old_brief)
        _mod().third_round(other, rounds.key(), rounds.entries(), {('e0', 'p1')})
        rounds.rate()
        with pytest.raises(ValueError, match=r'third_00 e0 p1'):
            rounds.scratch.effective_lines(rounds.key())
        with pytest.raises(ValueError, match=r'third_00 e0 p1'):
            rounds.scratch.ratings(rounds.key(), rounds.entries())

    def test_a_shard_recording_no_brief_is_refused_naming_it(self, tmp_path: Path) -> None:
        rounds = _Rounds(tmp_path)
        rounds.rate()
        key = rounds.key()
        del key['shards']['second_00']['brief_sha256']
        with pytest.raises(ValueError, match='second_00'):
            rounds.scratch.effective_lines(key)

    def test_round_0_must_record_one_brief(self, tmp_path: Path) -> None:
        rounds = _Rounds(tmp_path)
        rounds.rate()
        key = rounds.key()
        key['shards']['second_00']['brief_sha256'] = '0' * 64
        with pytest.raises(ValueError, match='round-0'):
            rounds.scratch.effective_lines(key)

    def test_the_effective_lines_need_no_texts(self, tmp_path: Path) -> None:
        rounds = _Rounds(tmp_path)
        rounds.rate()
        (rounds.root / 'texts.json').unlink()
        lines = rounds.scratch.effective_lines(rounds.key())
        assert sorted((_triple(row), row['role']) for row in lines) == sorted(
            (triple, role) for triple in ALL_TRIPLES for role in ('primary', 'second'))


class TestTheRaterContract:
    def test_the_answer_contract_is_the_brief_verbatim_and_the_output_contract(self) -> None:
        rounds = _mod()
        brief = rounds.BRIEF_PATH.read_text()
        contract = rounds.answer_contract()
        assert brief in contract
        output = contract.split(brief, 1)[1]
        for field in ('conflict_quote', 'loss_spans', 'action_changing', 'unclear_reason'):
            assert field in output
        assert 'digest' not in output.lower()

    def test_the_rater_prompt_embeds_the_contract_and_keeps_the_rater_blind(self) -> None:
        rounds = _mod()
        prompt = rounds.rater_prompt()
        assert rounds.answer_contract() in prompt
        assert 'blind rater' in prompt
        assert 'Do not open any other file' in prompt
        assert 'exactly one line' in prompt


class TestRoundsLargerThanAShard:
    """A re-ask or third round over more than SHARD_SIZE items is dealt into shards of one rater."""

    COUNT = 45

    @pytest.fixture
    def scratch(self, tmp_path: Path) -> tuple[Any, dict[Triple, Any]]:
        rounds = _mod()
        entries = [rounds.ShardEntry(f'e{index:02d}', 'p1', 'current', 'df', f'child {index}',
                                     f'parent {index}') for index in range(self.COUNT)]
        scratch = rounds.RatingScratch(tmp_path / 'scratch')
        key = scratch.load_key()
        scratch.add_shards(key, entries, seed=7)
        scratch.save_key(key)
        return scratch, {(entry.entry_id, entry.target_id, entry.basis): entry for entry in entries}

    @staticmethod
    def _rate(scratch: Any, rule: Rule) -> None:
        key = scratch.load_key()
        (scratch.root / 'ratings').mkdir(exist_ok=True)
        for name, spec in key['shards'].items():
            path = scratch.root / 'ratings' / f'{name}.jsonl'
            if not path.exists():
                path.write_text(''.join(
                    json.dumps({'id': item, **rule(None), 'reason': 'r'}) + '\n'
                    for item in spec['items']))

    def test_a_large_reask_round_is_split_under_one_rater_and_round(
        self, scratch: tuple[Any, dict[Triple, Any]],
    ) -> None:
        rounds, (store, entries) = _mod(), scratch
        self._rate(store, _lacks_loss)
        written = rounds.reask_round(store, store.load_key(), entries)
        assert written['lines'] == 2 * self.COUNT
        key = store.load_key()
        reasks = {name: spec for name, spec in key['shards'].items() if name.startswith('reask_')}
        assert sorted(reasks) == ['reask_01_primary_00', 'reask_01_primary_01',
                                  'reask_02_second_00', 'reask_02_second_01']
        assert written['shards'] == {name: len(spec['items']) for name, spec in reasks.items()}
        assert all(len(spec['items']) <= rounds.SHARD_SIZE for spec in reasks.values())
        assert {(spec['role'], spec['rater'], spec['round']) for spec in reasks.values()} == {
            ('primary', 'opus-primary', 1), ('second', 'opus-second', 2)}

    def test_a_large_third_round_is_split_under_one_new_rater(
        self, scratch: tuple[Any, dict[Triple, Any]],
    ) -> None:
        rounds, (store, entries) = _mod(), scratch
        blocked = {(entry_id, target_id) for entry_id, target_id, _ in entries}
        assert rounds.third_round(store, store.load_key(), entries, blocked) == {
            'pairs': self.COUNT, 'shards': ['third_00_00', 'third_00_01']}
        key = store.load_key()
        thirds = {name: spec for name, spec in key['shards'].items() if spec['role'] == 'third'}
        assert sorted(thirds) == ['third_00_00', 'third_00_01']
        assert {spec['rater'] for spec in thirds.values()} == {'opus-third-00'}
        assert sum(len(spec['items']) for spec in thirds.values()) == self.COUNT
        assert rounds.third_round(store, key, entries, {('e00', 'p1')}) == {
            'pairs': 1, 'shards': ['third_01']}
        assert store.load_key()['shards']['third_01']['rater'] == 'opus-third-01'
