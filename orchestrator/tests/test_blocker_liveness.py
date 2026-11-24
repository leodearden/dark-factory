"""Whether a lock blocker's holder is live: in-flight slots and claimant heartbeat for holders, parks followed."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Any

import pytest

from orchestrator.blocker_liveness import (
    PARK_FOLLOW_DEPTH,
    HolderState,
    LockWorld,
    NonLiveHolder,
    non_live_holder,
)
from orchestrator.pin_reservation import Blocker, BlockerKind

NOW = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)
TTL = timedelta(seconds=300)
HELD, PARKED = BlockerKind.HELD, BlockerKind.PARKED


def _task(
    status: str, *,
    claimant: str | None = None,
    beat_secs_ago: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    beat = None if beat_secs_ago is None else (NOW - timedelta(seconds=beat_secs_ago)).isoformat()
    record: dict[str, Any] = {'status': status, 'claimant_run_id': claimant, 'heartbeat_at': beat}
    if metadata is not None:
        record['metadata'] = metadata
    return record


def _running(beat_secs_ago: float = 30) -> dict[str, Any]:
    return _task('in-progress', claimant='run-x', beat_secs_ago=beat_secs_ago)


def _world(
    tasks: Mapping[str, Mapping[str, Any]],
    blockers: Mapping[str, Sequence[Blocker]] | None = None,
    in_flight: frozenset[str] | None = None,
) -> LockWorld:
    """In production every lock holder has a slot, so *in_flight* defaults to every task."""
    table = dict(blockers or {})
    return LockWorld(
        tasks=tasks,
        blockers_of=lambda tid: tuple(table.get(tid, ())),
        now=NOW,
        claimant_ttl=TTL,
        in_flight=frozenset(tasks) if in_flight is None else in_flight,
    )


def _held(owner: str, module: str = 'hot/a.py') -> Blocker:
    return Blocker(module=module, owner=owner, kind=HELD)


def _parked(owner: str, module: str = 'hot/a.py') -> Blocker:
    return Blocker(module=module, owner=owner, kind=PARKED)


class TestHeldBlocker:
    """A held lock's owner is judged against this process's in-flight slots."""

    @pytest.mark.parametrize('record', [
        _running(),
        _task('pending'),
        _task('pending', claimant='run-x', beat_secs_ago=10),
        _task('blocked', claimant='run-x', beat_secs_ago=10),
        _task('done'),
        _task('cancelled'),
        _task('in-progress', claimant='run-x', beat_secs_ago=900, metadata={'infra_hold': True}),
        _task('blocked', metadata={'infra_hold': True}),
    ], ids=[
        'heartbeating', 'dispatched-not-yet-claimed', 'requeued-before-its-claim-clears',
        'blocked-on-its-way-out', 'done-before-release', 'cancelled-before-release',
        'legacy-infra-held', 'legacy-infra-held-blocked',
    ])
    def test_an_in_flight_owner_with_progress_or_unwinding_is_live(
        self, record: dict[str, Any],
    ) -> None:
        assert non_live_holder(_held('H'), _world({'H': record})) is None

    def test_an_in_flight_owner_gone_from_the_task_read_is_live(self) -> None:
        assert non_live_holder(_held('H'), _world({}, in_flight=frozenset({'H'}))) is None

    @pytest.mark.parametrize(('record', 'reason'), [
        (_running(beat_secs_ago=301), HolderState.NO_LIVE_CLAIMANT),
        (_task('in-progress', claimant=None, beat_secs_ago=10), HolderState.NO_LIVE_CLAIMANT),
        (_task('in-progress', claimant='run-x'), HolderState.NO_LIVE_CLAIMANT),
        (_task('blocked', claimant='run-x', beat_secs_ago=301), HolderState.BLOCKED),
        (_task('blocked'), HolderState.BLOCKED),
        (_task('pending', claimant='run-x', beat_secs_ago=900), HolderState.NO_LIVE_CLAIMANT),
        (_task('pending', claimant='run-x'), HolderState.NO_LIVE_CLAIMANT),
        (_task('merge-deferred', claimant='run-x', beat_secs_ago=900), HolderState.NO_LIVE_CLAIMANT),
        (_task('infra-hold', claimant='run-x', beat_secs_ago=900), HolderState.NO_LIVE_CLAIMANT),
    ], ids=[
        'stale-heartbeat', 'no-claimant', 'no-heartbeat', 'blocked-stale-claimant',
        'blocked-no-claimant', 'requeue-exit-hung', 'requeued-no-heartbeat',
        'merge-deferred-exit-hung', 'infra-hold-status-exit-hung',
    ])
    def test_an_in_flight_owner_without_a_live_claimant_names_its_blocker(
        self, record: dict[str, Any], reason: HolderState,
    ) -> None:
        blocker = _held('H')

        assert non_live_holder(blocker, _world({'H': record})) == NonLiveHolder(blocker, reason)

    @pytest.mark.parametrize('tasks', [{'H': _running()}, {}], ids=['heartbeating', 'absent'])
    def test_an_owner_with_no_slot_holds_a_lock_without_a_slot(
        self, tasks: dict[str, Mapping[str, Any]],
    ) -> None:
        blocker = _held('H')

        assert non_live_holder(blocker, _world(tasks, in_flight=frozenset())) == NonLiveHolder(
            blocker, HolderState.LOCK_WITHOUT_SLOT,
        )


class TestParkedBlocker:
    """Specimen esc-__starvation_auto_pin_cap__-34: 9945 waits on a park whose owner waits on live holders."""

    ENGINE = 'gui/src-tauri/src/engine.rs'
    PARK = Blocker(module=ENGINE, owner='5519', kind=PARKED)
    OWNER_BLOCKERS = (
        Blocker(module='gui/src-tauri/src/lib.rs', owner='5499', kind=HELD),
        Blocker(module='gui/src/App.tsx', owner='10029', kind=HELD),
    )

    def _specimen(self, *, stale_10029: bool = False) -> LockWorld:
        return _world(
            {
                '9945': _task('pending'),
                '5519': _task('pending'),
                '5499': _running(),
                '10029': _running(beat_secs_ago=900 if stale_10029 else 30),
            },
            {'5519': self.OWNER_BLOCKERS},
        )

    def test_a_park_whose_owner_waits_on_live_holders_is_live(self) -> None:
        assert non_live_holder(self.PARK, self._specimen()) is None

    def test_the_deepest_non_live_holder_is_named_not_the_park(self) -> None:
        assert non_live_holder(self.PARK, self._specimen(stale_10029=True)) == NonLiveHolder(
            self.OWNER_BLOCKERS[1], HolderState.NO_LIVE_CLAIMANT,
        )

    def test_a_park_owner_with_no_blockers_waits_only_for_a_slot(self) -> None:
        assert non_live_holder(_parked('P'), _world({'P': _task('pending')})) is None

    def test_two_parks_waiting_on_each_other_are_a_cycle(self) -> None:
        a_waits_on_b = _parked('B', 'hot/b.py')
        b_waits_on_a = _parked('A', 'hot/c.py')
        world = _world(
            {'A': _task('pending'), 'B': _task('pending')},
            {'A': [a_waits_on_b], 'B': [b_waits_on_a]},
        )

        assert non_live_holder(_parked('A'), world) == NonLiveHolder(
            b_waits_on_a, HolderState.PARK_CYCLE,
        )

    def test_a_chain_of_parks_deeper_than_the_bound_is_too_deep(self) -> None:
        chain = [f'P{i}' for i in range(PARK_FOLLOW_DEPTH + 2)]
        tasks = {owner: _task('pending') for owner in chain}
        links = {
            owner: [_parked(nxt, f'hot/{nxt}.py')] for owner, nxt in pairwise(chain)
        }
        too_deep = _parked(chain[PARK_FOLLOW_DEPTH], f'hot/{chain[PARK_FOLLOW_DEPTH]}.py')

        assert non_live_holder(_parked(chain[0]), _world(tasks, links)) == NonLiveHolder(
            too_deep, HolderState.PARK_TOO_DEEP,
        )

    def test_a_chain_of_parks_within_the_bound_reaches_its_holder(self) -> None:
        chain = [f'P{i}' for i in range(PARK_FOLLOW_DEPTH)]
        tasks: dict[str, Mapping[str, Any]] = {owner: _task('pending') for owner in chain}
        tasks['H'] = _running()
        links = {
            owner: [_parked(nxt, f'hot/{nxt}.py')] for owner, nxt in pairwise(chain)
        }
        links[chain[-1]] = [_held('H', 'hot/H.py')]

        assert non_live_holder(_parked(chain[0]), _world(tasks, links)) is None

    def test_a_park_owner_absent_from_the_task_read_is_not_active(self) -> None:
        park = _parked('ghost')

        assert non_live_holder(park, _world({})) == NonLiveHolder(park, HolderState.NOT_ACTIVE)
