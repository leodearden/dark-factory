"""ModuleLockTable.acquire_uncontested: the live-slot partial widen (task 3090).

Takes every free key, names each contested one with its holder, and never
releases anything the task already holds.
"""

from __future__ import annotations

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.lock_table import ModuleLockTable, UncontestedAcquisition
from orchestrator.pin_reservation import Blocker, BlockerKind


def _lt() -> ModuleLockTable:
    return ModuleLockTable(OrchestratorConfig(max_per_module=1, lock_depth=12))


@pytest.fixture
def lt() -> ModuleLockTable:
    table = _lt()
    assert table.try_acquire('t1', ['a.py'])
    return table


def test_all_free_keys_are_acquired(lt):
    result = lt.acquire_uncontested('t1', ['b.py', 'c.py'])

    assert result == UncontestedAcquisition(acquired=('b.py', 'c.py'), contested=())
    assert lt.held_modules('t1') == {'a.py', 'b.py', 'c.py'}


def test_a_held_key_is_contested_and_nothing_is_released(lt):
    assert lt.try_acquire('t2', ['c.py'])

    result = lt.acquire_uncontested('t1', ['b.py', 'c.py'])

    assert result.acquired == ('b.py',)
    assert result.contested == (Blocker('c.py', 't2', BlockerKind.HELD),)
    assert lt.held_modules('t1') == {'a.py', 'b.py'}
    assert lt.held_modules('t2') == {'c.py'}


def test_a_key_the_task_already_holds_is_neither_acquired_nor_contested(lt):
    result = lt.acquire_uncontested('t1', ['a.py', 'b.py'])

    assert result.acquired == ('b.py',)
    assert result.contested == ()
    assert lt.held_modules('t1') == {'a.py', 'b.py'}


def test_all_contested_acquires_nothing(lt):
    assert lt.try_acquire('t2', ['b.py', 'c.py'])

    result = lt.acquire_uncontested('t1', ['b.py', 'c.py'])

    assert result.acquired == ()
    assert result.contested == (
        Blocker('b.py', 't2', BlockerKind.HELD),
        Blocker('c.py', 't2', BlockerKind.HELD),
    )
    assert lt.held_modules('t1') == {'a.py'}


def test_held_modules_is_an_immutable_snapshot(lt):
    held = lt.held_modules('t1')

    assert isinstance(held, frozenset)
    lt.acquire_uncontested('t1', ['b.py'])
    assert held == {'a.py'}
    assert lt.held_modules('unknown') == frozenset()


def test_a_key_cannot_be_both_acquired_and_contested():
    with pytest.raises(ValueError, match='x.py'):
        UncontestedAcquisition(
            acquired=('x.py',),
            contested=(Blocker('x.py', 't9', BlockerKind.HELD),),
        )
