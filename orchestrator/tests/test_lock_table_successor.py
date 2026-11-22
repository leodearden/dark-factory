"""The successor relation in ModuleLockTable (early start, PRD section 5.2).

``plans/early-start-on-predecessor-branch-prd.md`` sections 5.2 and 7: a
dependent D declared the successor of its predecessor P may share a key that
allows one holder with P, and P may widen onto such a key D holds.  Every other
requester counts both, two-holder keys keep the ordinary count, reservations
keep the plain rule, and the relation lives until either task releases.

P is the predecessor, D the dependent, X a third task and U an unrelated
holder.  Every test drives ModuleLockTable's public API only, on the code
defaults: pydantic-settings would otherwise merge the operational yaml's own
``module_overrides`` into each table's holder limits.
"""

from __future__ import annotations

import itertools

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.lock_table import (
    ModuleLockTable,
    SuccessorRelation,
    UncontestedAcquisition,
)
from orchestrator.pin_reservation import Blocker, BlockerKind

pytestmark = pytest.mark.usefixtures('code_default_config')

RELATION = SuccessorRelation(successor='D', predecessor='P')


def _table() -> ModuleLockTable:
    return ModuleLockTable(
        OrchestratorConfig(max_per_module=1, lock_depth=12, module_overrides={'two.py': 2})
    )


def _co_held() -> ModuleLockTable:
    """P holds a.py; D, its successor, co-holds a.py and alone holds c.py."""
    table = _table()
    assert table.try_acquire('P', ['a.py'])
    assert table.try_acquire_as_successor('D', ['a.py', 'c.py'], predecessor='P')
    return table


def _held(task_id: str, *modules: str) -> list[Blocker]:
    return [Blocker(module, task_id, BlockerKind.HELD) for module in modules]


class TestSuccessorRelationValue:
    def test_partner_of_names_the_other_member(self):
        assert RELATION.partner_of('D') == 'P'
        assert RELATION.partner_of('P') == 'D'
        assert RELATION.partner_of('X') is None

    def test_a_task_cannot_succeed_itself(self):
        with pytest.raises(ValueError, match="'P'"):
            SuccessorRelation(successor='P', predecessor='P')


class TestCoHold:
    def test_the_successor_shares_a_single_holder_key_with_its_predecessor(self):
        table = _co_held()

        assert table.snapshot_held_by_task() == {'P': ('a.py',), 'D': ('a.py', 'c.py')}
        assert table.successor_relation('D') == RELATION
        assert table.successor_relation('P') == RELATION
        assert table.snapshot_successor_of() == {'D': 'P'}

    def test_snapshot_successor_of_is_a_fresh_dict(self):
        table = _co_held()

        table.snapshot_successor_of()['D'] = 'tampered'

        assert table.snapshot_successor_of() == {'D': 'P'}

    def test_a_third_task_counts_both_holders(self):
        table = _co_held()

        assert not table.try_acquire('X', ['a.py'])
        assert table.blocking_holders(['a.py'], exclude_task='X') == {'a.py': ['D', 'P']}
        assert table.blockers('X', ['a.py']) == _held('D', 'a.py') + _held('P', 'a.py')

    def test_the_partners_name_no_blocker_on_each_others_keys(self):
        table = _co_held()

        assert table.blocking_holders(['a.py', 'c.py'], exclude_task='P') == {}
        assert table.blockers('P', ['c.py']) == []
        assert table.blockers('D', ['a.py']) == []


class TestPredecessorWidens:
    def test_the_predecessor_widens_onto_a_key_its_successor_holds(self):
        table = _co_held()

        assert table.try_acquire_additional('P', ['c.py'])
        assert table.held_modules('P') == {'a.py', 'c.py'}

    def test_the_in_place_widening_takes_the_successors_key_uncontested(self):
        table = _co_held()

        result = table.acquire_uncontested('P', ['c.py'])

        assert result == UncontestedAcquisition(acquired=('c.py',), contested=())
        assert table.held_modules('P') == {'a.py', 'c.py'}


class TestTwoHolderKey:
    def test_the_successor_is_refused_when_the_key_already_has_two_holders(self):
        table = _table()
        assert table.try_acquire('P', ['two.py'])
        assert table.try_acquire('U', ['two.py'])

        assert not table.try_acquire_as_successor('D', ['two.py'], predecessor='P')
        assert not table.is_held('D')
        assert table.held_modules('D') == frozenset()
        assert table.successor_relation('P') is None
        assert table.successor_relation('D') is None
        assert table.snapshot_successor_of() == {}

    def test_the_successor_fills_the_second_slot_and_a_third_task_is_refused(self):
        table = _table()
        assert table.try_acquire('P', ['two.py'])

        assert table.try_acquire_as_successor('D', ['two.py'], predecessor='P')
        assert not table.try_acquire('X', ['two.py'])
        assert table.blockers('X', ['two.py']) == _held('D', 'two.py') + _held('P', 'two.py')

    def test_the_predecessor_is_refused_on_a_two_holder_key_its_successor_shares(self):
        """The accepted residual: two holders already, so the ordinary count refuses P."""
        table = _table()
        assert table.try_acquire('P', ['a.py'])
        assert table.try_acquire_as_successor('D', ['two.py'], predecessor='P')
        assert table.try_acquire('U', ['two.py'])

        assert not table.try_acquire_additional('P', ['two.py'])
        assert table.held_modules('P') == {'a.py'}


class TestReservationsKeepThePlainRule:
    def test_a_foreign_park_on_a_free_key_refuses_the_successor_like_anyone(self):
        table = _table()
        assert table.try_acquire('P', ['a.py'])
        table.install_parks('R', ['c.py'], 'high')

        assert not table.try_acquire('X', ['c.py'])
        assert not table.try_acquire_as_successor('D', ['a.py', 'c.py'], predecessor='P')
        assert not table.is_held('D')
        assert table.successor_relation('P') is None

    def test_a_foreign_park_on_the_predecessors_key_refuses_the_successor(self):
        table = _table()
        assert table.try_acquire('P', ['a.py'])
        table.install_parks('R', ['a.py'], 'high')

        assert not table.try_acquire_as_successor('D', ['a.py'], predecessor='P')
        assert not table.is_held('D')
        assert table.successor_relation('P') is None


class TestReleaseDropsTheRelation:
    def test_the_predecessors_release_leaves_the_successor_sole_holder(self):
        table = _co_held()

        assert table.release('P') == ['a.py']

        assert table.snapshot_held_by_task() == {'D': ('a.py', 'c.py')}
        assert table.successor_relation('D') is None
        assert table.snapshot_successor_of() == {}
        assert not table.try_acquire('X', ['a.py'])

    def test_a_requeued_predecessor_does_not_skip_its_former_successor(self):
        table = _co_held()
        table.release('P')

        assert not table.try_acquire('P', ['a.py'])

    def test_the_key_frees_once_both_have_released(self):
        table = _co_held()
        table.release('P')
        table.release('D')

        assert table.try_acquire('X', ['a.py'])

    def test_the_successors_release_frees_the_predecessor_for_a_new_successor(self):
        table = _co_held()

        table.release('D')

        assert table.successor_relation('P') is None
        assert table.try_acquire_as_successor('D2', ['a.py'], predecessor='P')
        assert table.snapshot_successor_of() == {'D2': 'P'}

    def test_narrowing_keeps_the_relation(self):
        table = _table()
        assert table.try_acquire('P', ['a.py', 'b.py'])
        assert table.try_acquire_as_successor('D', ['a.py'], predecessor='P')

        assert table.release_subset('P', ['a.py']) == ['a.py']

        assert table.successor_relation('D') == RELATION
        assert table.try_acquire_additional('P', ['a.py'])


class TestDeclarationGuards:
    """Invariant 1: a predecessor has at most one successor, a successor one predecessor."""

    def _assert_raises_naming(self, table, successor, modules, predecessor, *ids):
        before_held = table.snapshot_held_by_task()
        before_relations = table.snapshot_successor_of()
        with pytest.raises(ValueError) as raised:
            table.try_acquire_as_successor(successor, modules, predecessor=predecessor)
        for task_id in ids:
            assert repr(task_id) in str(raised.value)
        assert table.snapshot_held_by_task() == before_held
        assert table.snapshot_successor_of() == before_relations

    def test_a_second_successor_for_the_same_predecessor(self):
        self._assert_raises_naming(_co_held(), 'D2', ['b.py'], 'P', 'D2', 'P')

    def test_the_same_successor_for_a_second_predecessor(self):
        table = _co_held()
        assert table.try_acquire('Q', ['q.py'])

        self._assert_raises_naming(table, 'D', ['q.py'], 'Q', 'D', 'Q')

    def test_a_successor_as_predecessor_of_another(self):
        self._assert_raises_naming(_co_held(), 'E', ['e.py'], 'D', 'E', 'D')

    def test_a_predecessor_that_holds_nothing(self):
        self._assert_raises_naming(_table(), 'D', ['a.py'], 'NOPE', 'NOPE')

    def test_a_task_as_its_own_predecessor(self):
        table = _table()
        assert table.try_acquire('P', ['a.py'])

        self._assert_raises_naming(table, 'P', ['b.py'], 'P', 'P')

    def test_a_successor_that_already_holds_locks(self):
        table = _table()
        assert table.try_acquire('P', ['a.py'])
        assert table.try_acquire('D', ['b.py'])

        self._assert_raises_naming(table, 'D', ['a.py'], 'P', 'D')


class _LimitReadFails(OrchestratorConfig):
    """Fails the holder-limit read for ``broken.py``, as a malformed module config would."""

    def for_module(self, module_path):
        if module_path == 'broken.py':
            raise RuntimeError('module config read failed')
        return super().for_module(module_path)


def test_an_error_in_the_admit_check_declares_no_relation():
    table = ModuleLockTable(_LimitReadFails(max_per_module=1, lock_depth=12))
    assert table.try_acquire('P', ['a.py'])

    with pytest.raises(RuntimeError, match='module config read failed'):
        table.try_acquire_as_successor('D', ['broken.py'], predecessor='P')

    assert table.successor_relation('P') is None
    assert table.snapshot_successor_of() == {}
    assert table.snapshot_held_by_task() == {'P': ('a.py',)}


GRID_KEYS: tuple[str, ...] = ('a.py', 'c.py', 'two.py', 'b.py', 'u.py')
GRID_KEY_SETS: list[list[str]] = [
    list(keys)
    for size in range(1, len(GRID_KEYS) + 1)
    for keys in itertools.combinations(GRID_KEYS, size)
]


def _grid_table() -> ModuleLockTable:
    """P holds a.py and two.py; D, its successor, co-holds both and alone holds c.py; U holds u.py."""
    table = _table()
    assert table.try_acquire('P', ['a.py', 'two.py'])
    assert table.try_acquire_as_successor('D', ['a.py', 'c.py', 'two.py'], predecessor='P')
    assert table.try_acquire('U', ['u.py'])
    return table


@pytest.mark.parametrize('requester', ['P', 'D', 'X'])
def test_the_acquires_and_the_blocker_views_agree(requester):
    """Invariant 3: a request names no blocker exactly when its acquire succeeds."""
    disagreements = []
    for keys in GRID_KEY_SETS:
        named = _grid_table().blockers(requester, keys)
        table = _grid_table()
        if requester == 'X':
            acquired = table.try_acquire(requester, keys)
        else:
            acquired = table.try_acquire_additional(requester, keys)
        if acquired != (named == []):
            disagreements.append((keys, acquired, named))

    assert disagreements == []


def test_the_grid_reaches_both_answers_for_every_requester():
    """Guards the agreement test above against a grid that only ever says yes or no."""
    for requester in ('P', 'D', 'X'):
        answers = {_grid_table().blockers(requester, keys) == [] for keys in GRID_KEY_SETS}
        assert answers == {True, False}, requester
