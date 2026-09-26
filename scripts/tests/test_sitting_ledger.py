"""Tests for scripts/sitting/ledger.py — session-stable numbering and answer bookkeeping (task 5376)."""
from __future__ import annotations

import random
import re
from dataclasses import replace

import pytest
from sitting import ledger as mod
from sitting.inventory import decision_key, escalation_key, key_str
from sitting.preparation import EscalationClosed, Manual, Standing, TaskStatusIs

QUEUE = '/src/dark-factory/data/escalations'
T0 = '2026-09-26T08:00:00+00:00'
T1 = '2026-09-26T08:30:00+00:00'
T2 = '2026-09-26T09:00:00+00:00'
T3 = '2026-09-26T09:30:00+00:00'
HOLD = Standing('hold', 'Leo', TaskStatusIs('4811', ('done',)), 'Leo HOLD 2026-08-27')
SEVERITY_RANK = {'critical': 0, 'blocking': 1, 'info': 2}


def esc(n: int) -> str:
    return key_str(escalation_key(QUEUE, f'esc-{n}-1'))


A, B, C, D = esc(1), esc(2), esc(3), key_str(decision_key('df-esc-9-9'))
OPTIONS = {
    A: {'A': 'close as ruled', 'B': 'hold for a re-ruling'},
    C: {'A': 'retarget', 'B': 'cancel', 'C': 'split'},
    D: {'A': 'close the decision'},
}


def _flat(key: str) -> int:
    return 0


def _assign(ledger: mod.Ledger, keys, now: str = T0, sort_key=_flat) -> mod.Ledger:
    return mod.assign(ledger, keys, sort_key, now)


def _numbers(ledger: mod.Ledger) -> dict[str, int]:
    return {key: entry.number for key, entry in ledger.entries.items()}


def _answer(item_ref: str, option_ref: str = '', *, note: str = '', at: str = T1, source: str = 'terminal'):
    return mod.Answer(item_ref, option_ref, note, at, source)


def _sitting() -> mod.Ledger:
    """Items 1 (A), 3 (C) and 4 (D) open; item 2 (B) done."""
    return _assign(_assign(mod.new_sitting(T0), [A, B, C, D]), [A, C, D], now=T1)


class TestNewSitting:
    def test_without_a_seed_there_are_no_entries(self):
        ledger = mod.new_sitting(T0)

        assert ledger.entries == {}
        assert ledger.started_at == T0

    def test_started_at_is_a_timestamp(self):
        with pytest.raises(ValueError):
            mod.new_sitting('this morning')


class TestAssign:
    def test_new_keys_are_numbered_from_one_in_sort_order(self):
        severities = {A: 'info', B: 'critical', C: 'blocking'}

        ledger = _assign(mod.new_sitting(T0), [A, B, C], sort_key=lambda key: SEVERITY_RANK[severities[key]])

        assert _numbers(ledger) == {B: 1, C: 2, A: 3}

    def test_the_key_is_the_final_tiebreak(self):
        ledger = _assign(mod.new_sitting(T0), [C, A, B])

        assert _numbers(ledger) == {A: 1, B: 2, C: 3}

    def test_existing_numbers_are_kept_and_new_keys_take_max_plus_one(self):
        ledger = _assign(mod.new_sitting(T0), [B, C])

        ledger = _assign(ledger, [A, B, C], now=T1)

        assert _numbers(ledger) == {B: 1, C: 2, A: 3}
        assert ledger.entries[A].first_presented_at == T1
        assert ledger.entries[B].first_presented_at == T0

    def test_a_key_that_leaves_goes_done_keeping_its_number(self):
        ledger = _sitting()

        assert _numbers(ledger) == {A: 1, B: 2, C: 3, D: 4}
        assert ledger.entries[B].state == 'done'
        assert ledger.entries[B].done_at == T1
        assert [key for key, _ in ledger.in_state('open')] == [A, C, D]
        assert [key for key, _ in ledger.in_state('done')] == [B]

    def test_a_number_is_never_reused(self):
        ledger = _assign(_sitting(), [A, C, D, esc(5)], now=T2)

        assert ledger.entries[esc(5)].number == 5
        assert ledger.entries[B].number == 2

    def test_numbering_is_stable_over_a_shuffled_equal_key_set(self):
        keys = [esc(n) for n in range(1, 9)]
        shuffled = keys[:]
        random.Random(7).shuffle(shuffled)

        first = _assign(mod.new_sitting(T0), keys)
        second = _assign(_assign(mod.new_sitting(T0), shuffled), list(reversed(keys)), now=T1)

        assert _numbers(first) == _numbers(second)

    def test_a_done_key_that_returns_reopens_under_its_old_number(self):
        ledger = _assign(_sitting(), [A, B, C, D], now=T2)

        assert ledger.entries[B].number == 2
        assert ledger.entries[B].state == 'open'
        assert ledger.entries[B].done_at is None

    def test_a_key_must_be_an_open_item_key(self):
        with pytest.raises(ValueError):
            _assign(mod.new_sitting(T0), ['esc-1-1'])

    def test_now_is_a_timestamp(self):
        with pytest.raises(ValueError):
            _assign(mod.new_sitting(T0), [A], now='later')


class TestSeed:
    def test_the_seeds_open_numbers_are_the_sessions_numbers(self):
        nightly = _assign(mod.new_sitting(T0), [A, B, C])

        session = _assign(mod.new_sitting(T2, seed=nightly), [A, B, C, D], now=T2)

        assert _numbers(session) == {A: 1, B: 2, C: 3, D: 4}
        assert session.started_at == T2
        assert session.sitting_id != nightly.sitting_id

    def test_a_number_the_seed_already_retired_is_not_reissued(self):
        nightly = _assign(_assign(mod.new_sitting(T0), [A, B, C]), [A, B], now=T1)

        session = _assign(mod.new_sitting(T2, seed=nightly), [A, B, D], now=T2)

        assert C not in session.entries
        assert _numbers(session) == {A: 1, B: 2, D: 4}

    def test_standing_is_carried_and_answer_rounds_start_again(self):
        seed = mod.set_standing(_sitting(), C, HOLD)
        seed, _ = mod.resolve_answers(seed, [_answer('1', 'A')], OPTIONS)

        session = mod.new_sitting(T2, seed=seed)

        assert session.entries[C].standing == HOLD
        assert session.entries[A].answer_rounds == 0
        assert session.entries[A].last_answered_at is None


class TestStanding:
    def test_standing_leaves_the_numbered_set_but_keeps_its_number(self):
        ledger = mod.set_standing(_sitting(), C, HOLD)

        assert ledger.entries[C].state == 'standing'
        assert ledger.entries[C].number == 3
        assert [key for key, _ in ledger.in_state('open')] == [A, D]
        assert [(key, entry.standing) for key, entry in ledger.in_state('standing')] == [(C, HOLD)]

    def test_standing_persists_across_reassigns(self):
        ledger = _assign(mod.set_standing(_sitting(), C, HOLD), [A, C, D, esc(5)], now=T2)

        assert ledger.entries[C].standing == HOLD
        assert ledger.entries[esc(5)].number == 5

    def test_a_standing_key_that_leaves_goes_done(self):
        ledger = _assign(mod.set_standing(_sitting(), C, HOLD), [A, D], now=T2)

        assert ledger.entries[C].state == 'done'
        assert ledger.entries[C].number == 3

    def test_clearing_standing_returns_the_item_to_the_numbered_set(self):
        ledger = mod.set_standing(mod.set_standing(_sitting(), C, HOLD), C, None)

        assert ledger.entries[C].state == 'open'
        assert ledger.entries[C].standing is None

    def test_a_key_not_in_the_ledger_is_refused(self):
        with pytest.raises(ValueError):
            mod.set_standing(_sitting(), esc(7), HOLD)

    def test_a_done_item_takes_no_standing(self):
        with pytest.raises(ValueError):
            mod.set_standing(_sitting(), B, HOLD)


class TestResolveAnswers:
    def test_each_resolved_answer_echoes_item_record_and_option(self):
        answers = [_answer('1', 'B'), _answer('3', 'C', note='split it | keep the cap', source='docket')]

        _, resolved = mod.resolve_answers(_sitting(), answers, OPTIONS)

        assert resolved.rows == (
            mod.EchoRow(1, A, 'esc-1-1', 'B', 'hold for a re-ruling', '', T1),
            mod.EchoRow(3, C, 'esc-3-1', 'C', 'split', 'split it | keep the cap', T1),
        )
        assert resolved.unresolved == ()

    def test_a_decision_item_echoes_its_decision_id(self):
        _, resolved = mod.resolve_answers(_sitting(), [_answer(' 4 ', 'A')], OPTIONS)

        assert [row.record_id for row in resolved.rows] == ['df-esc-9-9']

    @pytest.mark.parametrize(('item_ref', 'option_ref', 'reason'), [
        ('C', '', 'not an item number'),
        ('1', '', 'no option given'),
        ('9', 'A', 'no item 9'),
        ('2', 'A', 'already done'),
        ('3', 'D', "no option 'D'"),
        ('3.0', 'A', 'not an item number'),
        ('-1', 'A', 'not an item number'),
    ])
    def test_unresolvable_tokens_are_asked_back_with_a_reason(self, item_ref, option_ref, reason):
        _, resolved = mod.resolve_answers(_sitting(), [_answer(item_ref, option_ref)], OPTIONS)

        assert resolved.rows == ()
        assert [(u.answer.item_ref, u.answer.option_ref) for u in resolved.unresolved] == [(item_ref, option_ref)]
        assert reason in resolved.unresolved[0].reason

    def test_the_two_corpus_misreads_are_never_guessed(self):
        _, resolved = mod.resolve_answers(_sitting(), [_answer('C'), _answer('1')], OPTIONS)

        assert resolved.rows == ()
        assert len(resolved.unresolved) == 2

    def test_a_standing_item_is_asked_back_not_applied(self):
        ledger = mod.set_standing(_sitting(), C, HOLD)

        _, resolved = mod.resolve_answers(ledger, [_answer('3', 'A')], OPTIONS)

        assert resolved.rows == ()
        assert 'standing' in resolved.unresolved[0].reason

    def test_an_item_with_no_options_on_record_is_asked_back(self):
        _, resolved = mod.resolve_answers(_sitting(), [_answer('1', 'A')], {})

        assert 'no options on record' in resolved.unresolved[0].reason

    def test_answer_rounds_count_each_touched_item_once_per_round(self):
        answers = [_answer('1', 'B'), _answer('1', 'B', note='again'), _answer('3', 'Z'), _answer('2', 'A')]

        ledger, _ = mod.resolve_answers(_sitting(), answers, OPTIONS)
        ledger, _ = mod.resolve_answers(ledger, [_answer('3', 'C', at=T2)], OPTIONS)

        assert {key: entry.answer_rounds for key, entry in ledger.entries.items()} == {A: 1, B: 0, C: 2, D: 0}

    def test_tokens_that_name_no_live_item_touch_nothing(self):
        before = _sitting()

        after, _ = mod.resolve_answers(before, [_answer('9', 'A'), _answer('C', 'A')], OPTIONS)

        assert after == before


class TestResolutionTurns:
    def test_resolution_turns_is_the_items_answer_rounds_counting_ask_backs(self):
        ledger, _ = mod.resolve_answers(_sitting(), [_answer('3', 'Z')], OPTIONS)
        ledger, _ = mod.resolve_answers(ledger, [_answer('3', 'A', at=T2)], OPTIONS)

        assert mod.resolution_turns(ledger, C) == 2
        assert mod.resolution_turns(ledger, A) == 0

    def test_an_unknown_key_is_refused(self):
        with pytest.raises(ValueError):
            mod.resolution_turns(_sitting(), esc(7))


class TestSittingSummary:
    def test_the_sitting_time_instrument_inputs(self):
        ledger, _ = mod.resolve_answers(_sitting(), [_answer('1', 'A', at=T1)], OPTIONS)
        ledger, _ = mod.resolve_answers(ledger, [_answer('3', 'Z', at=T2)], OPTIONS)
        ledger, _ = mod.resolve_answers(ledger, [_answer('3', 'B', at=T3)], OPTIONS)

        assert mod.sitting_summary(ledger) == mod.SittingSummary(
            started_at=T0, items_presented=4, items_answered=2,
            first_answered_at=T1, last_answered_at=T3, answer_rounds=3,
        )

    def test_standing_items_are_not_counted_as_presented(self):
        summary = mod.sitting_summary(mod.set_standing(_sitting(), C, HOLD))

        assert summary.items_presented == 3

    def test_an_unanswered_sitting(self):
        assert mod.sitting_summary(mod.new_sitting(T0)) == mod.SittingSummary(
            started_at=T0, items_presented=0, items_answered=0,
            first_answered_at=None, last_answered_at=None, answer_rounds=0,
        )


class TestEchoTable:
    def test_one_row_per_resolved_answer_and_no_ask_back_when_all_resolve(self):
        _, resolved = mod.resolve_answers(_sitting(), [_answer('1', 'B'), _answer('3', 'C')], OPTIONS)

        text = mod.render_echo_table(resolved)

        rows = [line for line in text.splitlines() if re.match(r'\|\s*\d+\s*\|', line)]
        assert len(rows) == 2
        assert 'esc-1-1' in rows[0] and 'hold for a re-ruling' in rows[0]
        assert 'esc-3-1' in rows[1] and 'split' in rows[1]
        assert 'ASK BACK:' not in text

    def test_the_ask_back_block_lists_each_token_and_its_reason(self):
        _, resolved = mod.resolve_answers(_sitting(), [_answer('1', 'B'), _answer('C'), _answer('3', 'Z')], OPTIONS)

        text = mod.render_echo_table(resolved)
        ask_back = text.split('ASK BACK:', 1)[1]

        assert "'C'" in ask_back and 'not an item number' in ask_back
        assert "'Z'" in ask_back and "no option 'Z'" in ask_back

    def test_a_pipe_or_newline_in_a_note_does_not_break_the_row(self):
        _, resolved = mod.resolve_answers(_sitting(), [_answer('1', 'B', note='a | b\nc')], OPTIONS)

        lines = mod.render_echo_table(resolved).splitlines()
        header = next(line for line in lines if line.startswith('|'))
        row = next(line for line in lines if re.match(r'\|\s*1\s*\|', line))

        def cells(line: str) -> int:
            return len(re.findall(r'(?<!\\)\|', line))

        assert cells(row) == cells(header)


class TestInvariants:
    def test_an_answer_names_its_source(self):
        with pytest.raises(ValueError):
            _answer('1', 'A', source='email')

    def test_an_answer_is_timestamped(self):
        with pytest.raises(ValueError):
            _answer('1', 'A', at='just now')

    @pytest.mark.parametrize('overrides', [
        {'number': 0},
        {'state': 'parked'},
        {'answer_rounds': -1},
        {'state': 'standing'},
        {'state': 'done'},
        {'standing': Standing('pin', 'esc-3105-5', EscalationClosed('esc-3105-5'), 'veto pin')},
    ])
    def test_an_entry_keeps_its_state_consistent(self, overrides):
        entry = _sitting().entries[A]

        with pytest.raises(ValueError):
            replace(entry, **overrides)

    def test_numbers_are_unique_in_a_ledger(self):
        ledger = _sitting()

        with pytest.raises(ValueError):
            replace(ledger, entries={**ledger.entries, C: replace(ledger.entries[C], number=1)})


class TestPersistence:
    def test_save_then_load_round_trips(self, tmp_path):
        path = tmp_path / 'sitting' / 'ledger.json'
        ledger = mod.set_standing(_sitting(), D, Standing('leo_owned', 'Leo', Manual('Leo says when'), 'handover'))
        ledger, _ = mod.resolve_answers(ledger, [_answer('1', 'B')], OPTIONS)

        mod.save(path, ledger)

        assert mod.load(path) == ledger

    def test_save_is_atomic_and_leaves_no_temp_files(self, tmp_path):
        path = tmp_path / 'ledger.json'

        mod.save(path, _sitting())
        mod.save(path, _sitting())

        assert [p.name for p in tmp_path.iterdir()] == ['ledger.json']

    def test_a_missing_file_is_no_ledger(self, tmp_path):
        assert mod.load(tmp_path / 'ledger.json') is None

    @pytest.mark.parametrize('body', [
        '{"sitting_id": ',
        '[]',
        '{"sitting_id": "s"}',
        '{"sitting_id": "s", "started_at": "2026-09-26T08:00:00+00:00", "next_number": 1, "entries": [{"key": ["esc"]}]}',
    ])
    def test_a_corrupt_file_is_loud_and_names_the_path(self, tmp_path, body):
        path = tmp_path / 'ledger.json'
        path.write_text(body)

        with pytest.raises(mod.LedgerCorrupt) as caught:
            mod.load(path)

        assert caught.value.path == path
        assert str(path) in str(caught.value)
