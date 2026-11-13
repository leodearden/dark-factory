"""The one module-lock counting rule the Scheduler tab and the PRD view's contention line share."""

from __future__ import annotations

from dashboard.data.lock_contention import holder_of, lock_wanters


def test_each_member_wants_a_key_once_however_often_its_lock_set_names_it():
    assert lock_wanters([['a/b', 'c/d'], ['a/b'], ['a/b', 'a/b']]) == {'a/b': 3, 'c/d': 1}


def test_an_empty_population_wants_nothing():
    assert lock_wanters([]) == {}


def test_a_holder_of_the_exact_key_holds_it():
    assert holder_of('a/b', {'c/d': '7', 'a/b': '9'}) == '9'


def test_a_holder_of_a_sub_depth_parent_key_holds_its_child():
    assert holder_of('crates/reify-types', {'crates': '4'}) == '4'


def test_a_key_no_holder_conflicts_with_is_unheld():
    assert holder_of('a/b', {'a/bc': '1', 'c/d': '2'}) is None


def test_of_several_conflicting_holders_the_first_in_holder_order_holds_it():
    assert holder_of('a/b', {'a': '1', 'a/b': '2'}) == '1'
    assert holder_of('a/b', {'a/b': '2', 'a': '1'}) == '2'
