"""Park precedence: a fairness-park cycle between pending owners is impossible by construction (task 7799).

The solar_challenge park-cycle-787-945 ruling (Leo, 2026-10-30): a deadlock
between parked owners must be impossible by construction, not broken by hand.
Every park entry has one key, rank first and then the dispatch tie-break
(older numeric id first, non-numeric last).  A foreign park blocks a requester
that owns parks only when the park's key precedes the requester's best key,
so every park wait points at an owner with a strictly smaller best key and
the wait-for graph between park owners can hold no cycle.  A requester that
owns no park is blocked by every park, as before.

Every module in the scheduler-level test is a single-segment ``.py`` file, so
its lock key is its own name at any ``lock_depth``.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _park_test_helpers import event_data_for, make_task, park_world

from orchestrator.config import OrchestratorConfig
from orchestrator.lock_table import ModuleLockTable
from orchestrator.pin_reservation import Blocker, BlockerKind, ParkPriority, PinOrder


def _lt(**config_overrides: object) -> ModuleLockTable:
    return ModuleLockTable(OrchestratorConfig(**{'max_per_module': 1, 'lock_depth': 2, **config_overrides}))


class TestTheGateOrdersParkOwners:
    def test_a_followers_same_tier_park_never_blocks_its_predecessor(self):
        lt = _lt()
        lt.install_parks('945', ['config.py', 'test_sweep.py'], 'high')
        lt.install_parks('787', ['domain_doc.py', 'other.py'], 'high')
        footprint_787 = ['domain_doc.py', 'other.py', 'config.py']

        assert lt.park_owners_blocking('787', footprint_787) == []
        assert lt.try_acquire('787', footprint_787) is True
        lt.release('787')
        assert lt.park_owners_blocking('945', ['config.py', 'test_sweep.py', 'domain_doc.py']) == ['787']

    def test_a_park_owner_passes_a_lower_tier_park_on_a_key_it_never_parked(self):
        lt = _lt()
        lt.install_parks('H', ['m1'], 'high')
        lt.install_parks('L', ['m2'], 'low')

        assert lt.park_owners_blocking('H', ['m2']) == []
        assert lt.park_owners_blocking('L', ['m1']) == ['H']

    def test_a_parkless_requester_is_still_blocked_by_every_park(self):
        lt = _lt()
        lt.install_parks('L', ['m1'], 'low')

        assert lt.try_acquire('X', ['m1']) is False
        assert lt.blockers('X', ['m1']) == [Blocker('m1', 'L', BlockerKind.PARKED)]

    def test_the_id_tie_break_is_numeric_and_puts_non_numeric_ids_last(self):
        lt = _lt()
        lt.install_parks('10', ['m1'], 'high')
        lt.install_parks('9', ['m2'], 'high')
        lt.install_parks('x', ['m3'], 'high')

        assert lt.park_owners_blocking('9', ['m1', 'm2']) == []
        assert lt.park_owners_blocking('10', ['m1', 'm2']) == ['9']
        assert lt.park_owners_blocking('x', ['m1', 'm3']) == ['10']
        assert lt.park_owners_blocking('10', ['m1', 'm3']) == []

    def test_a_pin_owner_passes_a_critical_park(self):
        lt = _lt()
        lt.install_parks('P', ['m1'], PinOrder(3))
        lt.install_parks('C', ['m2'], 'critical')

        assert lt.park_owners_blocking('P', ['m2']) == []
        assert lt.park_owners_blocking('C', ['m1']) == ['P']


def _stack_owners(lt: ModuleLockTable, module: str) -> list[str]:
    """*module*'s park stack owners, bottom to top."""
    return [entry['owner'] for entry in lt.snapshot_park_stacks().get(module, [])]


class TestInstallFollowsTheSameOrder:
    def test_a_same_tier_predecessor_shadows_a_followers_park(self):
        lt = _lt()
        lt.install_parks('945', ['m1'], 'high')

        assert lt.install_parks('787', ['m1'], 'high') == (['m1'], [('945', ['m1'])])
        assert _stack_owners(lt, 'm1') == ['945', '787']
        assert lt.clear_parks_for('787') == [('945', ['m1'])]

    def test_a_same_tier_follower_cannot_shadow_its_predecessor(self):
        lt = _lt()
        lt.install_parks('787', ['m1'], 'high')

        assert lt.install_parks('945', ['m1'], 'high') == ([], [])
        assert _stack_owners(lt, 'm1') == ['787']

        numeric = _lt()
        numeric.install_parks('9', ['m1'], 'high')
        assert numeric.install_parks('10', ['m1'], 'high') == ([], [])
        reversed_order = _lt()
        reversed_order.install_parks('10', ['m1'], 'high')
        assert reversed_order.install_parks('9', ['m1'], 'high') == (['m1'], [('10', ['m1'])])

    def test_tied_pin_orders_resolve_by_id(self):
        lt = _lt()
        lt.install_parks('20', ['m1'], PinOrder(0))
        assert lt.install_parks('7', ['m1'], PinOrder(0)) == (['m1'], [('20', ['m1'])])

        reversed_order = _lt()
        reversed_order.install_parks('7', ['m1'], PinOrder(0))
        assert reversed_order.install_parks('20', ['m1'], PinOrder(0)) == ([], [])
        assert _stack_owners(reversed_order, 'm1') == ['7']


_WALK_MODULES = ['a', 'a/x', 'a/y', 'b', 'b/x', 'c', 'd']
_WALK_OWNERS = ['1', '2', '3', '10', 'x', 'y']
_WALK_PRIORITIES: list[ParkPriority] = [
    'critical', 'high', 'high', 'medium', 'low', 'polish', PinOrder(1), PinOrder(2),
]
_WALK_STEPS = 30


def _find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    """A cycle in *graph* as its node path, or None."""
    done: set[str] = set()

    def visit(node: str, path: list[str]) -> list[str] | None:
        if node in path:
            return path[path.index(node):] + [node]
        if node in done:
            return None
        for successor in sorted(graph.get(node, ())):
            cycle = visit(successor, path + [node])
            if cycle:
                return cycle
        done.add(node)
        return None

    for start in sorted(graph):
        cycle = visit(start, [])
        if cycle:
            return cycle
    return None


class _ParkWalk:
    """A seeded random walk over the lock table's public park and lock operations."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.lt = _lt(module_overrides={'b': 2})
        self.footprints = {
            owner: self.rng.sample(_WALK_MODULES, self.rng.randint(1, 4)) for owner in _WALK_OWNERS
        }
        self.holding: set[str] = set()

    def step(self) -> None:
        op = self.rng.choice(['install', 'acquire', 'release', 'clear'])
        waiting = [owner for owner in _WALK_OWNERS if owner not in self.holding]
        if op == 'install' and waiting:
            owner = self.rng.choice(waiting)
            footprint = self.footprints[owner]
            modules = self.rng.sample(footprint, self.rng.randint(1, len(footprint)))
            self.lt.install_parks(owner, modules, self.rng.choice(_WALK_PRIORITIES))
        elif op == 'acquire' and waiting:
            owner = self.rng.choice(waiting)
            if self.lt.try_acquire(owner, self.footprints[owner]):
                self.holding.add(owner)
                self.lt.clear_parks_for(owner)
        elif op == 'release' and self.holding:
            owner = self.rng.choice(sorted(self.holding))
            self.lt.release(owner)
            self.holding.discard(owner)
        elif op == 'clear':
            self.lt.clear_parks_for(self.rng.choice(_WALK_OWNERS))

    def wait_for_graph(self) -> dict[str, set[str]]:
        return {
            owner: set(self.lt.park_owners_blocking(owner, self.footprints[owner]))
            for owner in _WALK_OWNERS
            if owner not in self.holding and self.lt.has_parks(owner)
        }


@pytest.mark.parametrize('seed', range(40))
def test_no_reachable_park_state_has_a_wait_cycle(seed: int):
    walk = _ParkWalk(seed)
    for step in range(_WALK_STEPS):
        walk.step()
        graph = walk.wait_for_graph()
        cycle = _find_cycle(graph)
        assert cycle is None, f'seed {seed} step {step}: park wait cycle {cycle} in {graph}'
        if graph:
            assert any(not blockers for blockers in graph.values()), (
                f'seed {seed} step {step}: every park owner waits on a park: {graph}'
            )


_SHAPES = {
    'two_owner': {
        '787': (['domain_doc.py', 'other.py'], ['config.py']),
        '945': (['config.py', 'test_sweep.py'], ['domain_doc.py']),
    },
    'three_owner': {
        '787': (['domain_doc.py', 'validate.py', 'config.py'], ['home.py', 'test_fleet.py']),
        '894': (['home.py'], ['domain_doc.py']),
        '919': (['test_fleet.py', 'test_web.py'], ['domain_doc.py', 'config.py']),
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize('orientation', ['head_top', 'follower_top'])
@pytest.mark.parametrize('shape', sorted(_SHAPES))
async def test_partially_parked_owners_in_a_cycle_dispatch_the_precedence_head(
    tmp_path: Path, shape: str, orientation: str,
):
    owners = _SHAPES[shape]
    follower = max(owners, key=int)
    score_top = '787' if orientation == 'head_top' else follower
    now = datetime.now(UTC)
    world = park_world(tmp_path)
    footprints: dict[str, list[str]] = {}
    tasks: list[dict] = []
    for owner, (parked, needed) in owners.items():
        footprints[owner] = parked + needed
        world.scheduler.lock_table.install_parks(owner, parked, 'high')
        task = make_task(owner, 'high', footprints[owner])
        hours = 30 if owner == score_top else 1
        task['metadata']['pending_since'] = (now - timedelta(hours=hours)).isoformat()
        tasks.append(task)
    world.scheduler.get_tasks = AsyncMock(return_value=tasks)

    assignment = None
    for _ in range(3):
        assignment = await world.scheduler.acquire_next()
        if assignment is not None:
            break

    assert assignment is not None, f'{shape}/{orientation}: no owner dispatched in 3 ticks'
    assert assignment.task_id == '787', f'{shape}/{orientation}: the precedence head dispatches'
    if orientation == 'follower_top':
        assert event_data_for(world.store, 'task_skipped', follower), (
            f'{shape}/{orientation}: premise: the follower {follower} was the score-top'
        )
    for owner, footprint in footprints.items():
        if owner == '787':
            continue
        kinds = {blocker.kind for blocker in world.scheduler.lock_table.blockers(owner, footprint)}
        assert kinds <= {BlockerKind.HELD}, f'{shape}/{orientation}: {owner} still waits on a park'
