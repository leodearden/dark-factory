"""The tested tree of an in-flight entry: its values, its invariants, its base table.

``orchestrator.merge_lane.tested_tree`` is the pure value behind
``InflightEntry.tree`` (plans/merge-pool-batches-prd.md §6.1): T1 is held at
construction, T2 and T1's owner at the use point, and a base resolves by
commit through the three-row table.
"""

from __future__ import annotations

import pytest

from orchestrator.merge_lane import tested_tree

_MAIN = 'c-main'


def _member(
    name: str, commit: str, *, own: bool = True, tip: str | None = 'tip'
) -> tested_tree.TreeMember:
    return tested_tree.TreeMember(
        request_id=f'r-{name}',
        task_id=name,
        merge_commit=commit,
        tip=f'{tip}-{name}' if tip is not None else None,
        own=own,
    )


def _tree(
    *members: tested_tree.TreeMember,
    base: str = _MAIN,
    root: str | None = _MAIN,
) -> tested_tree.TestedTree:
    return tested_tree.TestedTree(base_commit=base, root_commit=root, members=members)


class TestT1HeldAtConstruction:
    def test_a_request_listed_twice_is_refused(self) -> None:
        with pytest.raises(tested_tree.TestedTreeViolation) as caught:
            _tree(_member('a', 'c1'), _member('a', 'c2'))
        assert str(caught.value).startswith('T1')
        assert 'r-a' in str(caught.value)

    def test_a_tree_with_no_own_member_is_refused(self) -> None:
        with pytest.raises(tested_tree.TestedTreeViolation) as caught:
            _tree(_member('a', 'c1', own=False))
        assert str(caught.value).startswith('T1')
        assert 'r-a' in str(caught.value)

    def test_a_built_on_member_after_an_own_one_is_refused(self) -> None:
        with pytest.raises(tested_tree.TestedTreeViolation) as caught:
            _tree(_member('a', 'c1'), _member('b', 'c2', own=False))
        assert str(caught.value).startswith('T1')
        assert 'r-b' in str(caught.value)
        assert '1' in str(caught.value)

    def test_the_violation_is_a_value_error(self) -> None:
        assert issubclass(tested_tree.TestedTreeViolation, ValueError)

    def test_a_member_with_no_merge_commit_is_refused(self) -> None:
        with pytest.raises(tested_tree.TestedTreeViolation):
            _member('a', '')


class TestTheBaseTable:
    def test_a_base_on_main_has_no_built_on_members_and_is_its_own_root(self) -> None:
        own = (_member('o', 'c-o'),)

        tree = tested_tree.resolve_tested_tree(
            base_commit=_MAIN, base_on_main=True, live_trees=(), own_members=own
        )

        assert tree.base_commit == _MAIN
        assert tree.root_commit == _MAIN
        assert tree.members == own

    def test_main_is_checked_before_the_live_trees(self) -> None:
        live = _tree(_member('a', 'c-a'), root='c-older')
        own = (_member('o', 'c-o'),)

        tree = tested_tree.resolve_tested_tree(
            base_commit='c-a', base_on_main=True, live_trees=(live,), own_members=own
        )

        assert tree.root_commit == 'c-a'
        assert tree.members == own

    def test_a_base_on_a_live_tip_is_built_on_every_member_of_that_tree(self) -> None:
        live = _tree(_member('a', 'c-a'), _member('k', 'c-k'))
        own = (_member('o', 'c-o'),)

        tree = tested_tree.resolve_tested_tree(
            base_commit='c-k', base_on_main=False, live_trees=(live,), own_members=own
        )

        assert tree.base_commit == 'c-k'
        assert tree.root_commit == live.root_commit
        assert tree.members == (
            _member('a', 'c-a', own=False),
            _member('k', 'c-k', own=False),
            *own,
        )

    def test_a_base_on_a_live_owner_before_its_links_is_built_on_that_owner_alone(
        self,
    ) -> None:
        live = _tree(_member('a', 'c-a'), _member('k', 'c-k'))
        own = (_member('o', 'c-o'),)

        tree = tested_tree.resolve_tested_tree(
            base_commit='c-a', base_on_main=False, live_trees=(live,), own_members=own
        )

        assert tree.root_commit == live.root_commit
        assert tree.members == (_member('a', 'c-a', own=False), *own)

    def test_a_live_owner_brings_the_members_it_was_built_on(self) -> None:
        live = _tree(
            _member('b', 'c-b', own=False),
            _member('a', 'c-a'),
            _member('k', 'c-k'),
            base='c-b',
            root='c-root',
        )
        own = (_member('o', 'c-o'),)

        tree = tested_tree.resolve_tested_tree(
            base_commit='c-a', base_on_main=False, live_trees=(live,), own_members=own
        )

        assert tree.root_commit == 'c-root'
        assert tree.members == (
            _member('b', 'c-b', own=False),
            _member('a', 'c-a', own=False),
            *own,
        )

    def test_a_base_no_live_tree_holds_has_no_root(self) -> None:
        live = _tree(_member('a', 'c-a'))
        own = (_member('o', 'c-o'), _member('k', 'c-k'))

        tree = tested_tree.resolve_tested_tree(
            base_commit='c-elsewhere',
            base_on_main=False,
            live_trees=(live,),
            own_members=own,
        )

        assert tree.base_commit == 'c-elsewhere'
        assert tree.root_commit is None
        assert tree.members == own


class TestTheOwnerAndTheVerifiedCommit:
    def _tree(self) -> tested_tree.TestedTree:
        return _tree(
            _member('b', 'c-b', own=False),
            _member('o', 'c-o'),
            _member('k', 'c-k'),
            base='c-b',
        )

    def test_the_owner_is_the_first_own_member(self) -> None:
        assert self._tree().owner == _member('o', 'c-o')

    def test_the_verified_commit_is_the_last_own_members_merge_commit(self) -> None:
        assert self._tree().verified_commit == 'c-k'

    def test_a_matching_owner_and_commit_pass(self) -> None:
        assert self._tree().require_verifies('r-o', 'c-k') is None

    def test_a_different_verified_commit_breaches_t2(self) -> None:
        with pytest.raises(tested_tree.TestedTreeViolation) as caught:
            self._tree().require_verifies('r-o', 'c-other')
        message = str(caught.value)
        assert message.startswith('T2')
        assert 'c-k' in message
        assert 'c-other' in message

    def test_a_different_owner_breaches_t1(self) -> None:
        with pytest.raises(tested_tree.TestedTreeViolation) as caught:
            self._tree().require_verifies('r-someone', 'c-k')
        message = str(caught.value)
        assert message.startswith('T1')
        assert 'r-o' in message
        assert 'r-someone' in message


def test_the_wire_form_lists_members_in_tree_order() -> None:
    tree = _tree(
        _member('b', 'c-b', own=False),
        _member('o', 'c-o', tip=None),
        base='c-b',
        root='c-root',
    )

    assert tree.to_wire() == {
        'base_commit': 'c-b',
        'root_commit': 'c-root',
        'members': [
            {
                'request_id': 'r-b',
                'task_id': 'b',
                'merge_commit': 'c-b',
                'tip': 'tip-b',
                'own': False,
            },
            {
                'request_id': 'r-o',
                'task_id': 'o',
                'merge_commit': 'c-o',
                'tip': None,
                'own': True,
            },
        ],
    }
