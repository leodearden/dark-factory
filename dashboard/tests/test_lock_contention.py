"""The one module-lock counting rule the Scheduler tab and the PRD view's contention line share."""

from __future__ import annotations

import sys
from typing import Any

import pytest
from shared.scheduler_state import SchedulerStateSnapshot

from dashboard.data.lock_contention import KeyHolders, key_holders_in, lock_wanters

UNLIMITED = sys.maxsize


def _held(key: str, **snapshot: Any) -> KeyHolders:
    return key_holders_in(SchedulerStateSnapshot.from_json(snapshot))(key)


def test_each_member_wants_a_key_once_however_often_its_lock_set_names_it():
    assert lock_wanters([['a/b', 'c/d'], ['a/b'], ['a/b', 'a/b']]) == {'a/b': 3, 'c/d': 1}


def test_an_empty_population_wants_nothing():
    assert lock_wanters([]) == {}


class TestWhoHoldsAKey:
    def test_every_holder_of_the_exact_key_is_named_in_sorted_order(self):
        held = _held(
            'OPERATIONS.md',
            held_by_task={'B': ['OPERATIONS.md'], 'A': ['OPERATIONS.md']},
            holder_limits={'OPERATIONS.md': UNLIMITED},
        )
        assert held.holders == ('A', 'B')

    def test_a_holder_of_a_sub_depth_parent_key_holds_its_child(self):
        assert _held('crates/reify-types', held_by_task={'4': ['crates']}).holders == ('4',)

    def test_a_key_no_holder_conflicts_with_is_unheld(self):
        held = _held('a/b', held_by_task={'1': ['a/bc'], '2': ['c/d']})
        assert held == KeyHolders(holders=(), at_limit=False)

    def test_holders_of_a_parent_and_of_the_exact_key_are_unioned(self):
        held = _held('a/b', held_by_task={'2': ['a/b'], '1': ['a']})
        assert held.holders == ('1', '2')


class TestWhetherTheHoldersFillTheLimit:
    def test_an_unlimited_key_is_never_full(self):
        held = _held(
            'OPERATIONS.md',
            held_by_task={'A': ['OPERATIONS.md'], 'B': ['OPERATIONS.md']},
            holder_limits={'OPERATIONS.md': UNLIMITED},
        )
        assert held.at_limit is False

    def test_one_holder_fills_an_exclusive_key(self):
        held = _held('src/a.py', held_by_task={'A': ['src/a.py']}, holder_limits={'src/a.py': 1})
        assert held.at_limit is True

    def test_a_limit_two_key_fills_at_its_second_holder(self):
        one = _held('k', held_by_task={'A': ['k']}, holder_limits={'k': 2})
        two = _held('k', held_by_task={'A': ['k'], 'B': ['k']}, holder_limits={'k': 2})
        assert one.at_limit is False
        assert two.at_limit is True

    def test_an_unheld_key_is_not_full(self):
        assert _held('k', holder_limits={'k': 1}).at_limit is False


class TestWhichLimitApplies:
    def test_a_held_key_reads_its_own_limit(self):
        held = _held(
            'docs/x.md',
            held_by_task={'1': ['docs'], '2': ['docs/x.md']},
            holder_limits={'docs': 1, 'docs/x.md': UNLIMITED},
        )
        assert held.holders == ('1', '2')
        assert held.at_limit is False

    def test_an_unheld_key_under_an_unlimited_held_parent_is_not_full(self):
        held = _held('docs/x.md', held_by_task={'1': ['docs']}, holder_limits={'docs': UNLIMITED})
        assert held.at_limit is False

    def test_an_unheld_key_under_an_exclusive_held_parent_is_full(self):
        held = _held('src/a.py', held_by_task={'1': ['src']}, holder_limits={'src': 1})
        assert held.at_limit is True

    def test_an_unheld_key_reads_the_smallest_conflicting_limit(self):
        held = _held(
            'a/b',
            held_by_task={'1': ['a'], '2': ['a/b/c']},
            holder_limits={'a': UNLIMITED, 'a/b/c': 1},
        )
        assert held.at_limit is True


def test_an_older_snapshot_without_limits_reads_each_held_key_as_exclusive():
    held = _held('m', current_holders={'m': '9'})
    assert held == KeyHolders(holders=('9',), at_limit=True)


class TestBlocks:
    def test_a_full_key_blocks_a_task_that_is_not_a_holder(self):
        assert KeyHolders(holders=('A',), at_limit=True).blocks('B') is True

    def test_a_full_key_does_not_block_its_own_holder(self):
        assert KeyHolders(holders=('A', 'B'), at_limit=True).blocks('B') is False

    def test_a_key_below_its_limit_blocks_no_task(self):
        assert KeyHolders(holders=('A',), at_limit=False).blocks('B') is False

    def test_an_unheld_key_blocks_no_task(self):
        assert KeyHolders(holders=(), at_limit=False).blocks('B') is False


class TestKeyHoldersInvariants:
    def test_no_holders_cannot_fill_a_limit(self):
        with pytest.raises(ValueError, match=r'holders=\(\), at_limit=True'):
            KeyHolders(holders=(), at_limit=True)

    def test_holders_out_of_order_are_refused(self):
        with pytest.raises(ValueError, match=r"\('B', 'A'\)"):
            KeyHolders(holders=('B', 'A'), at_limit=True)

    def test_a_repeated_holder_is_refused(self):
        with pytest.raises(ValueError, match=r"\('A', 'A'\)"):
            KeyHolders(holders=('A', 'A'), at_limit=False)


def test_one_snapshot_answers_every_key_it_is_asked_about():
    holders_of = key_holders_in(SchedulerStateSnapshot.from_json({
        'held_by_task': {'A': ['src/a.py'], 'B': ['OPERATIONS.md']},
        'holder_limits': {'src/a.py': 1, 'OPERATIONS.md': UNLIMITED},
    }))
    assert holders_of('src/a.py') == KeyHolders(holders=('A',), at_limit=True)
    assert holders_of('OPERATIONS.md') == KeyHolders(holders=('B',), at_limit=False)
    assert holders_of('src/b.py') == KeyHolders(holders=(), at_limit=False)
