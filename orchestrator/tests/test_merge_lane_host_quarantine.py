"""The contract of ``merge_lane/host_quarantine.py::HostQuarantine`` (PRD κ, task 5040).

``HostQuarantine`` is the merge lane's only writer of the remote verify
hosts' shared quarantine set and of the per-host unavailability tracker.
It is driven ALONE here, through its public methods, over the
collaborators the worker composes it with: a ``FakeClock``, a real
``EscalationQueue`` on a scratch directory (so ``has_open_l1`` dedups
for real), a recording ``EventStore``, and a real ``HostAllocator`` built
over the shared set by ``HostQuarantine.allocator_over``, reached through a
zero-argument resolver the way the worker hands over its live allocator.

Every expected value is today's ``SpeculativeMergeWorker`` behaviour, read
off the methods this class takes over; none is new.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest
from _gate_fixtures import TEST_DEPLOYED_SHA
from _merge_lane_fakes import FakeClock
from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.merge_lane.host_quarantine import HostQuarantine
from orchestrator.remote_gate import ReapStamp, RunnerUnavailableKind
from orchestrator.verify_runner import HostAllocator, RunnerUnavailable

_HOST = 'laptop'
_OTHER = 'desktop'
_LANE_LOGGER = 'orchestrator.merge_lane'
_STILL_DOWN = 'still unavailable'


class _Remote:
    """A remote verify host whose probe answers the test sets."""

    is_local: ClassVar[bool] = False

    def __init__(self, name: str) -> None:
        self.name = name
        self.healthy = True
        self.probes: list[str] = []
        self.reaps = 0
        self.reap_reading: ReapStamp | None = None

    async def health(self) -> bool:
        self.probes.append('health')
        return self.healthy

    async def ensure_gate(self, *, event_store: Any = None, task_id: str | None = None) -> None:
        self.probes.append('ensure')

    async def probe_clean(self) -> bool:
        return True

    async def reap_idle_gates(self) -> ReapStamp | None:
        self.reaps += 1
        return self.reap_reading


class _Events(EventStore):
    """An EventStore that keeps what is emitted instead of writing it."""

    def __init__(self) -> None:
        object.__init__(self)
        self.events: list[tuple[Any, dict[str, Any]]] = []

    def emit(self, event_type: Any, *, data: dict[str, Any] | None = None, **_: Any) -> None:
        self.events.append((event_type, dict(data or {})))

    def of(self, event_type: Any) -> list[dict[str, Any]]:
        return [data for kind, data in self.events if kind == event_type]


class _Scene:
    def __init__(
        self, tmp_path: Path, *, hosts: tuple[str, ...] = (_HOST,),
        shared_set: set[str] | None = None,
        on_host_readmitted: Callable[[str], None] | None = None,
    ) -> None:
        self.tmp_path = tmp_path
        self.clock = FakeClock()
        self.events = _Events()
        self.escalations = EscalationQueue(tmp_path / 'escalations')
        self.remotes = {name: _Remote(name) for name in hosts}
        self.allocator: HostAllocator | None = None
        self.holders: dict[str, str] = {}
        self.quarantine = HostQuarantine(
            clock=self.clock,
            allocator=lambda: self.allocator,
            event_store=lambda: self.events,
            escalation_queue=lambda: self.escalations,
            deployed_sha=TEST_DEPLOYED_SHA,
            verify_hosts=hosts,
            quarantine=shared_set,
            host_holder_reader=lambda: self.holders,
            on_host_readmitted=on_host_readmitted,
        )
        self.allocator = self.quarantine.allocator_over(list(self.remotes.values()))

    def adopt(self, *, after_n: int, after_secs: float) -> None:
        self.quarantine.adopt_config(OrchestratorConfig(
            project_root=self.tmp_path,
            verify_host_unreachable_escalate_after_n=after_n,
            verify_host_unreachable_escalate_after_secs=after_secs,
        ))

    def bench(self, kind: RunnerUnavailableKind = 'transport', *, host: str = _HOST) -> None:
        self.quarantine.bench(host, RunnerUnavailable('boom', kind=kind))

    async def sweep(self, *, advance: float = 120.0) -> None:
        self.clock.time += advance
        await self.quarantine.sweep(self.clock.now())

    def handed_out(self) -> str | None:
        assert self.allocator is not None
        lease = self.allocator.acquire_remote()
        return None if lease is None else lease.name

    def open_l1s(self) -> list[Any]:
        return [esc for esc in self.escalations.get_pending() if esc.level == 1]


@pytest.fixture
def scene(tmp_path: Path) -> _Scene:
    return _Scene(tmp_path)


def _still_down_levels(caplog: pytest.LogCaptureFixture) -> list[int]:
    return [
        record.levelno for record in caplog.records
        if record.name == _LANE_LOGGER and _STILL_DOWN in record.getMessage()
    ]


# ---------------------------------------------------------------------------
# (a) Benching a remote for a transport RunnerUnavailable
# ---------------------------------------------------------------------------


def test_a_transport_bench_quarantines_the_host_and_opens_an_episode(scene: _Scene) -> None:
    started = scene.clock.now()
    scene.bench()

    assert scene.quarantine.is_quarantined(_HOST)
    assert scene.handed_out() is None, 'the allocator over the shared set must refuse the host'
    episode = scene.quarantine.unavailability(_HOST)
    assert episode is not None
    assert (episode.streak, episode.first_unavailable_at) == (1, started)


def test_a_second_bench_extends_the_streak_and_keeps_the_episode_start(scene: _Scene) -> None:
    started = scene.clock.now()
    scene.bench()
    scene.clock.time += 45.0
    scene.bench()

    episode = scene.quarantine.unavailability(_HOST)
    assert episode is not None
    assert (episode.streak, episode.first_unavailable_at) == (2, started)


def test_unavailability_is_a_copy_not_the_live_tracker_entry(scene: _Scene) -> None:
    scene.bench()
    episode = scene.quarantine.unavailability(_HOST)
    assert episode is not None
    episode.streak = 99

    again = scene.quarantine.unavailability(_HOST)
    assert again is not None and again.streak == 1


# ---------------------------------------------------------------------------
# (b) The local trust anchor
# ---------------------------------------------------------------------------


def test_the_local_trust_anchor_is_never_quarantined_but_says_so(
    scene: _Scene, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=_LANE_LOGGER)
    scene.quarantine.quarantine_unreachable('local', 'local said no', scene.clock.now())

    assert not scene.quarantine.is_quarantined('local')
    assert scene.quarantine.unavailability('local') is None
    [warning] = [r for r in caplog.records if r.name == _LANE_LOGGER and 'LOCAL' in r.getMessage()]
    assert warning.levelno == logging.WARNING
    assert 'local said no' in warning.getMessage()


# ---------------------------------------------------------------------------
# (c) The reprobe sweep
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_passing_probe_readmits_and_closes_the_episode(scene: _Scene) -> None:
    scene.bench()

    await scene.sweep()

    assert scene.remotes[_HOST].probes == ['health']
    assert not scene.quarantine.is_quarantined(_HOST)
    assert scene.quarantine.unavailability(_HOST) is None
    assert scene.handed_out() == _HOST


@pytest.mark.asyncio
async def test_a_failing_probe_keeps_the_host_out_with_a_throttled_info_record(
    scene: _Scene, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger=_LANE_LOGGER)
    scene.remotes[_HOST].healthy = False
    scene.bench()
    sweeps = HostQuarantine.REPROBE_STILL_DOWN_INFO_SWEEPS + 2

    for _ in range(sweeps):
        await scene.sweep()

    assert scene.quarantine.is_quarantined(_HOST)
    assert scene.handed_out() is None
    assert _still_down_levels(caplog) == (
        [logging.INFO] * HostQuarantine.REPROBE_STILL_DOWN_INFO_SWEEPS + [logging.DEBUG] * 2
    )

    caplog.clear()
    await scene.sweep(advance=HostQuarantine.REPROBE_STILL_DOWN_INFO_PERIOD_SECS)
    assert _still_down_levels(caplog) == [logging.INFO], 'one INFO per period thereafter'

    caplog.clear()
    scene.bench()
    await scene.sweep()
    assert _still_down_levels(caplog) == [logging.INFO], 'a streak change logs at INFO'


@pytest.mark.asyncio
async def test_the_throttle_restarts_with_each_new_episode(
    scene: _Scene, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger=_LANE_LOGGER)
    remote = scene.remotes[_HOST]
    remote.healthy = False
    scene.bench()
    for _ in range(HostQuarantine.REPROBE_STILL_DOWN_INFO_SWEEPS + 1):
        await scene.sweep()
    remote.healthy = True
    await scene.sweep()

    caplog.clear()
    remote.healthy = False
    scene.bench()
    await scene.sweep()
    assert _still_down_levels(caplog) == [logging.INFO]


@pytest.mark.asyncio
async def test_the_sweep_reaps_idle_gates_at_most_hourly_and_skips_benched_hosts(
    tmp_path: Path,
) -> None:
    scene = _Scene(tmp_path, hosts=(_HOST, _OTHER))
    scene.remotes[_OTHER].healthy = False
    scene.bench(host=_OTHER)

    await scene.sweep()
    await scene.sweep()
    assert (scene.remotes[_HOST].reaps, scene.remotes[_OTHER].reaps) == (1, 0)

    await scene.sweep(advance=HostQuarantine.GATE_REAP_INTERVAL_SECS)
    assert (scene.remotes[_HOST].reaps, scene.remotes[_OTHER].reaps) == (2, 0)


@pytest.mark.asyncio
async def test_the_sweep_files_an_l1_when_a_hosts_gate_reaper_stops_completing_runs(
    scene: _Scene,
) -> None:
    remote = scene.remotes[_HOST]
    remote.reap_reading = ReapStamp(path='/gates/.last-reap', gate_root=True, age_secs=None)

    await scene.sweep()
    for _ in range(2):
        await scene.sweep(advance=HostQuarantine.GATE_REAP_INTERVAL_SECS)

    [alarm] = scene.escalations.get_pending()
    assert (alarm.category, alarm.task_id) == ('gate_reaper_stale', '__gate_reaper_stale__laptop')

    remote.reap_reading = ReapStamp(path='/gates/.last-reap', gate_root=True, age_secs=0)
    await scene.sweep(advance=HostQuarantine.GATE_REAP_INTERVAL_SECS)

    assert scene.escalations.get_by_task('__gate_reaper_stale__laptop', status='pending') == []


# ---------------------------------------------------------------------------
# (d) The sticky skew bench
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_second_skew_bench_on_one_sha_is_sticky_and_not_reprobed(scene: _Scene) -> None:
    scene.bench('gate')
    await scene.sweep()
    assert not scene.quarantine.is_quarantined(_HOST), 'a first skew bench readmits on ensure-gate'

    scene.bench('gate')
    episode = scene.quarantine.unavailability(_HOST)
    assert episode is not None and episode.sticky
    await scene.sweep()

    assert scene.quarantine.is_quarantined(_HOST)
    assert scene.remotes[_HOST].probes == ['ensure'], 'a sticky bench is never probed'
    assert [e['sticky'] for e in scene.events.of(EventType.verify_host_skew)] == [False, True]
    assert all(e['code_sha'] == TEST_DEPLOYED_SHA.sha for e in scene.events.of(EventType.verify_host_skew))


# ---------------------------------------------------------------------------
# (e) The alarm thresholds adopted from config
# ---------------------------------------------------------------------------


def test_the_streak_threshold_from_config_files_one_deduped_l1(scene: _Scene) -> None:
    scene.adopt(after_n=2, after_secs=0.0)

    scene.bench()
    assert scene.open_l1s() == []
    scene.bench()
    scene.bench()

    [l1] = scene.open_l1s()
    assert (l1.category, l1.task_id) == ('verify_host_unreachable', f'__verify_host_unreachable__{_HOST}')
    assert '(2 consecutive RunnerUnavailable failures)' in l1.summary


@pytest.mark.asyncio
async def test_the_downtime_threshold_from_config_fires_from_the_sweep(scene: _Scene) -> None:
    scene.adopt(after_n=10, after_secs=600.0)
    scene.remotes[_HOST].healthy = False
    scene.bench()

    await scene.sweep(advance=599.0)
    assert scene.open_l1s() == []
    await scene.sweep(advance=1.0)
    await scene.sweep()

    [l1] = scene.open_l1s()
    assert l1.category == 'verify_host_unreachable'


# ---------------------------------------------------------------------------
# (f) The operator clear
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_operator_clear_of_a_benched_host(scene: _Scene) -> None:
    scene.bench()

    assert await scene.quarantine.clear(_HOST) == {
        'host': _HOST, 'cleared': True, 'quarantine_class': 'ru', 'was_sticky': False,
    }
    assert not scene.quarantine.is_quarantined(_HOST)
    assert scene.quarantine.unavailability(_HOST) is None


@pytest.mark.asyncio
async def test_the_operator_clear_of_a_host_this_lane_does_not_run(scene: _Scene) -> None:
    assert await scene.quarantine.clear('nowhere') == {
        'host': 'nowhere', 'cleared': False, 'quarantine_class': None, 'was_sticky': False,
        'reason': "'nowhere' is not a remote verify host of this worker",
    }


@pytest.mark.asyncio
async def test_the_operator_clear_of_a_host_that_is_not_benched(scene: _Scene) -> None:
    assert await scene.quarantine.clear(_HOST) == {
        'host': _HOST, 'cleared': False, 'quarantine_class': None, 'was_sticky': False,
        'reason': f'{_HOST!r} is not benched',
    }


# ---------------------------------------------------------------------------
# (g) The snapshot's per-host block
# ---------------------------------------------------------------------------

_UNTRACKED = {
    'unavailable_since': None, 'unavailable_secs': None, 'streak': None, 'reason': None,
    'kind': None, 'sticky': None, 'identity': None, 'task_held_by': None,
}


def test_host_states_is_the_snapshot_hosts_block(tmp_path: Path) -> None:
    scene = _Scene(tmp_path, hosts=(_HOST, _OTHER))
    started = scene.clock.now()
    scene.bench()
    scene.quarantine.quarantine_divergent(_OTHER)
    scene.quarantine.quarantine_unreachable('gone', 'renamed away', started)
    now = started + 30.0

    assert scene.quarantine.host_states(now) == [
        {'name': 'local', 'is_local': True, 'slot_state': 'free', 'quarantined': False,
         'quarantine_class': None, **_UNTRACKED},
        {'name': _HOST, 'is_local': False, 'slot_state': 'free', 'quarantined': True,
         'quarantine_class': 'ru', 'unavailable_since': started, 'unavailable_secs': 30.0,
         'streak': 1, 'reason': 'boom', 'kind': 'transport', 'sticky': False, 'identity': None,
         'task_held_by': None},
        {'name': _OTHER, 'is_local': False, 'slot_state': 'free', 'quarantined': True,
         'quarantine_class': 'divergence', **_UNTRACKED},
        {'name': 'gone', 'is_local': False, 'slot_state': None, 'quarantined': True,
         'quarantine_class': 'ru', 'unavailable_since': started, 'unavailable_secs': 30.0,
         'streak': 1, 'reason': 'renamed away', 'kind': 'transport', 'sticky': False,
         'identity': None, 'task_held_by': None},
    ]


def test_host_states_name_the_task_holding_each_host(scene: _Scene) -> None:
    scene.holders[_HOST] = 'task:42'

    held_by = {row['name']: row['task_held_by'] for row in scene.quarantine.host_states(0.0)}

    assert held_by == {'local': None, _HOST: 'task:42'}


def test_host_states_is_empty_before_an_allocator_exists(tmp_path: Path) -> None:
    unbuilt = HostQuarantine(
        clock=FakeClock(), allocator=lambda: None, event_store=lambda: None,
        escalation_queue=lambda: None, deployed_sha=TEST_DEPLOYED_SHA,
    )
    unbuilt.quarantine_unreachable(_HOST, 'boom', 0.0)

    assert unbuilt.host_states(0.0) == []


# ---------------------------------------------------------------------------
# (h) The verdict-divergence sink
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_divergence_quarantine_is_lifted_by_the_operator_and_never_by_reprobe(
    scene: _Scene,
) -> None:
    scene.quarantine.quarantine_divergent(_HOST)

    assert scene.quarantine.is_quarantined(_HOST)
    assert scene.handed_out() is None
    assert scene.quarantine.unavailability(_HOST) is None, 'a divergence opens no episode'

    await scene.sweep()
    assert scene.quarantine.is_quarantined(_HOST), 'reachability must not lift a divergence'
    assert scene.remotes[_HOST].probes == []

    outcome = await scene.quarantine.clear(_HOST)
    assert (outcome['cleared'], outcome['quarantine_class']) == (True, 'divergence')
    assert scene.handed_out() == _HOST


# ---------------------------------------------------------------------------
# (i) The harness's quarantine set and readmission hook (task 6670)
# ---------------------------------------------------------------------------


def test_an_injected_quarantine_set_is_the_one_written(tmp_path: Path) -> None:
    harness_set: set[str] = set()
    scene = _Scene(tmp_path, shared_set=harness_set)

    scene.bench()

    assert harness_set == {_HOST}
    assert scene.handed_out() is None


@pytest.mark.asyncio
async def test_a_readmission_tells_the_harness(tmp_path: Path) -> None:
    readmitted: list[str] = []
    scene = _Scene(tmp_path, on_host_readmitted=readmitted.append)
    scene.bench()

    await scene.sweep()

    assert readmitted == [_HOST]


@pytest.mark.asyncio
async def test_a_failing_readmission_hook_never_undoes_the_readmission(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    def refuse(name: str) -> None:
        raise RuntimeError(f'task pool refused {name}')

    scene = _Scene(tmp_path, on_host_readmitted=refuse)
    scene.bench()

    with caplog.at_level(logging.ERROR, logger=_LANE_LOGGER):
        await scene.sweep()

    assert not scene.quarantine.is_quarantined(_HOST)
    assert scene.handed_out() == _HOST
    assert any('on_host_readmitted hook raised' in r.getMessage() for r in caplog.records)
