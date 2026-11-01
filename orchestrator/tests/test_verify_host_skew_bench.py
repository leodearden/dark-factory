"""Task 6565, design D6: how the merge worker benches, re-probes and re-admits a remote.

One table decides it.  A 'transport' RunnerUnavailable (the host was not
reached) is an 'ru' bench, re-probed by ``ssh true`` and re-admitted without
bound.  Every other kind (the host was reached but could not run this
dispatcher's gate, or gave no usable verdict) is a 'skew' bench, re-probed by
ensure-gate; its second bench on one (host, deployed sha) — necessarily after
a passing probe — is sticky until an operator clear or a restart.  A
'wire_shape' bench and a sticky bench file their L1 at once; the rest use the
3-consecutive / 600 s ladder.

The worker is real and so are its HostAllocator and escalation queue; the
remote is a fake whose health and ensure-gate answers the test sets.  The
reprobe tick is driven through ``_reprobe_quarantined_hosts``, the one method
``_reprobe_loop`` runs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from _gate_fixtures import TEST_DEPLOYED_SHA, gated_remote_runner
from _merge_lane_fakes import FakeClock
from _xdist_crash_fixtures import XDIST_SESSION_ABORTED_OUTPUT
from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.merge_lane.worker import SpeculativeMergeWorker
from orchestrator.remote_gate import GateIdentity, GateIdentityPair, RunnerUnavailableKind
from orchestrator.verify import WORKER_DEATH_SUMMARY_MARKER, VerifyResult
from orchestrator.verify_runner import (
    MergeVerifySpec,
    RunnerUnavailable,
    UnscopedTypecheckSpec,
    result_to_json,
)

_HOST = 'laptop'
_SKEW_KINDS: tuple[RunnerUnavailableKind, ...] = ('gate', 'unparseable', 'no_verdict', 'wire_shape')


class _Remote:
    """A remote verify host whose probe answers are the test's to set."""

    is_local: ClassVar[bool] = False

    def __init__(self, name: str = _HOST) -> None:
        self.name = name
        self.healthy = True
        self.ensure_error: RunnerUnavailable | None = None
        self.cancel_rc = 0
        self.clean = True
        self.probes: list[str] = []
        self.reaps = 0

    async def health(self) -> bool:
        self.probes.append('health')
        return self.healthy

    async def ensure_gate(self, *, event_store: Any = None, task_id: str | None = None) -> None:
        self.probes.append('ensure')
        if self.ensure_error is not None:
            raise self.ensure_error

    async def probe_clean(self) -> bool:
        return self.clean

    async def cancel_verify(self) -> int:
        return self.cancel_rc

    async def reap_idle_gates(self) -> None:
        self.reaps += 1

    async def run_merge_verify(self, merge_sha: str, spec: MergeVerifySpec) -> VerifyResult:
        raise AssertionError('these tests never dispatch a verify')


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
    def __init__(self, tmp_path: Path, *, hosts: tuple[str, ...] = (_HOST,)) -> None:
        self.clock = FakeClock()
        self.remotes = [_Remote(name) for name in hosts]
        self.remote = self.remotes[0]
        self.events = _Events()
        self.escalations = EscalationQueue(tmp_path / 'escalations')
        git_ops = MagicMock()
        git_ops.project_root = tmp_path
        self.worker = SpeculativeMergeWorker(
            git_ops=git_ops, queue=asyncio.Queue(), event_store=self.events,
            escalation_queue=self.escalations, clock=self.clock,
            remote_runners=self.remotes, deployed_sha=TEST_DEPLOYED_SHA,
            verify_hosts=hosts,
        )
        self.allocator = self.worker._ensure_host_allocator(OrchestratorConfig(
            project_root=tmp_path,
            verify_host_unreachable_escalate_after_n=3,
            verify_host_unreachable_escalate_after_secs=600.0,
            verify_host_reprobe_interval_s=120.0,
        ))

    def bench(
        self, kind: RunnerUnavailableKind, reason: str = 'boom', *, host: str = _HOST, **kw: Any,
    ) -> None:
        self.worker.bench_unavailable_runner(host, RunnerUnavailable(reason, kind=kind, **kw))

    async def tick(self, *, advance: float = 120.0) -> None:
        self.clock.time += advance
        await self.worker._reprobe_quarantined_hosts(self.clock.now())

    def host(self) -> dict[str, Any]:
        [entry] = [h for h in self.worker.snapshot()['hosts'] if h['name'] == _HOST]
        return entry

    def in_pool(self) -> bool:
        return not self.allocator.is_quarantined(_HOST)

    def open_l1s(self) -> list[Any]:
        return [esc for esc in self.escalations.get_pending() if esc.level == 1]


@pytest.fixture
def scene(tmp_path: Path) -> _Scene:
    return _Scene(tmp_path)


def _spec() -> MergeVerifySpec:
    return MergeVerifySpec(
        verify_commands=(),
        unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
        task_files=None,
        verify_env={},
        cold_timeout_secs=60.0,
    )


# ---------------------------------------------------------------------------
# Acceptance 2 of task 6565
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unknown_keyword_verdict_benches_as_skew_with_event_and_l1_at_once(scene: _Scene):
    """The real codec's wire_shape RunnerUnavailable, through the worker's bench."""
    verdict = result_to_json(VerifyResult(
        passed=True, test_output='', lint_output='', type_output='', summary='ok',
    ))
    newer_remote = verdict[:-1] + ', "field_from_a_newer_gate": 1}'

    async def run(argv: list[str], *, cwd: object = None) -> tuple[int, str, str]:
        return (0, newer_remote if argv[0] == 'ssh' else '', '')

    runner = gated_remote_runner(_HOST, 'leo-laptop', 'leo-laptop', '/project', run=run)
    with pytest.raises(RunnerUnavailable) as excinfo:
        await runner.run_merge_verify('mergesha', _spec())
    assert excinfo.value.kind == 'wire_shape'

    scene.worker.bench_unavailable_runner(_HOST, excinfo.value)

    host = scene.host()
    assert (host['quarantine_class'], host['kind'], host['sticky']) == ('skew', 'wire_shape', False)
    assert [e['kind'] for e in scene.events.of(EventType.verify_host_skew)] == ['wire_shape']
    [l1] = scene.open_l1s()
    assert l1.category == 'verify_host_skew'
    assert l1.task_id == f'__verify_host_skew__{_HOST}'
    assert TEST_DEPLOYED_SHA.sha in l1.detail


@pytest.mark.asyncio
async def test_a_skew_bench_stays_out_while_ensure_keeps_failing(scene: _Scene):
    scene.bench('wire_shape')
    scene.remote.ensure_error = RunnerUnavailable('handshake mismatch', kind='wire_shape')

    for _ in range(3):
        await scene.tick()

    assert not scene.in_pool()
    assert scene.remote.probes == ['ensure'] * 3
    assert scene.host()['sticky'] is False, 'a failing reprobe never counts toward sticky'


@pytest.mark.asyncio
async def test_a_passing_ensure_readmits_and_the_second_failure_goes_sticky(scene: _Scene):
    scene.bench('unparseable')
    await scene.tick()
    assert scene.in_pool()
    assert scene.open_l1s() == []

    scene.bench('no_verdict')

    host = scene.host()
    assert (host['quarantine_class'], host['sticky']) == ('skew', True)
    [l1] = scene.open_l1s()
    assert 'STICKY' in l1.detail and 'clear_verify_host_quarantine' in l1.detail
    await scene.tick()
    assert not scene.in_pool()
    assert scene.remote.probes == ['ensure'], 'a sticky bench is not probed at all'


@pytest.mark.asyncio
async def test_a_no_verdict_bench_record_names_the_archived_output(
    scene: _Scene, tmp_path: Path,
):
    """A real no-verdict RunnerUnavailable leads the operator to its archived test log."""
    wire = result_to_json(VerifyResult(
        passed=False, test_output=XDIST_SESSION_ABORTED_OUTPUT, lint_output='', type_output='',
        summary=f'Failures: test leg {WORKER_DEATH_SUMMARY_MARKER}', category='test_failure',
    ))

    async def run(argv: list[str], *, cwd: object = None) -> tuple[int, str, str]:
        return (0, wire, 'LAPTOP LOG') if argv[0] == 'ssh' else (0, '', '')

    runner = gated_remote_runner(_HOST, 'leo-laptop', 'leo-laptop', '/project', run=run)
    archive_root = tmp_path / 'verify-logs'

    async def dispatch_and_bench(task_id: str) -> Path:
        with pytest.raises(RunnerUnavailable) as excinfo:
            await runner.run_merge_verify(
                'mergesha', _spec(), task_id=task_id, archive_root=archive_root,
            )
        assert excinfo.value.kind == 'no_verdict'
        scene.worker.bench_unavailable_runner(_HOST, excinfo.value)
        return next((archive_root / task_id).glob('attempt-1.remote-laptop.test-*.log'))

    first_test_log = await dispatch_and_bench('3821')
    assert str(first_test_log) in scene.host()['reason']
    assert str(first_test_log) in scene.events.of(EventType.verify_host_skew)[0]['reason']

    await scene.tick()
    assert scene.in_pool()

    second_test_log = await dispatch_and_bench('6215')
    [l1] = scene.open_l1s()
    assert str(second_test_log) in l1.detail


# ---------------------------------------------------------------------------
# The D6 table, kind by kind
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['transport', *_SKEW_KINDS])
@pytest.mark.parametrize('probe_passes', [True, False])
async def test_each_kind_is_probed_by_its_class_and_readmitted_only_on_a_pass(
    scene: _Scene, kind: RunnerUnavailableKind, probe_passes: bool,
):
    scene.remote.healthy = probe_passes
    if not probe_passes:
        scene.remote.ensure_error = RunnerUnavailable('still broken', kind='gate')

    scene.bench(kind)
    immediate = [esc.category for esc in scene.open_l1s()]
    await scene.tick()

    assert immediate == (['verify_host_skew'] if kind == 'wire_shape' else [])
    assert scene.remote.probes == ['health' if kind == 'transport' else 'ensure']
    assert scene.in_pool() is probe_passes


@pytest.mark.asyncio
@pytest.mark.parametrize(('kind', 'category'), [
    ('transport', 'verify_host_unreachable'), ('gate', 'verify_host_skew'),
])
async def test_the_ladder_files_after_three_benches_in_one_episode(scene: _Scene, kind, category):
    for _ in range(2):
        scene.bench(kind)
    assert scene.open_l1s() == []

    scene.bench(kind)

    assert [esc.category for esc in scene.open_l1s()] == [category]


@pytest.mark.asyncio
async def test_the_ladder_files_a_skew_l1_after_600s_of_failing_ensures(scene: _Scene):
    scene.bench('gate', reason="ensure-gate on 'leo-laptop' exited 70: uv sync failed")
    scene.remote.ensure_error = RunnerUnavailable('uv sync failed', kind='gate')

    await scene.tick(advance=599.0)
    assert scene.open_l1s() == []
    await scene.tick(advance=2.0)

    [l1] = scene.open_l1s()
    assert l1.category == 'verify_host_skew'
    assert 'uv sync failed' in l1.summary


@pytest.mark.asyncio
async def test_the_skew_l1_names_the_shim_when_the_remote_answered_another_commit(scene: _Scene):
    local = GateIdentity(TEST_DEPLOYED_SHA.sha, 'a' * 64)
    pair = GateIdentityPair(local, GateIdentity('0' * 40, 'a' * 64))

    scene.bench('wire_shape', reason='gate answered another commit', identity=pair)

    [l1] = scene.open_l1s()
    assert 'install_laptop_gate_shim.sh' in l1.suggested_action
    assert '0' * 40 in l1.detail and TEST_DEPLOYED_SHA.sha in l1.detail


@pytest.mark.asyncio
async def test_a_restarted_process_resolves_the_skew_l1s_its_predecessor_left_open(
    tmp_path: Path,
):
    """A restart clears every bench, sticky included; its L1s must not be orphaned.

    Resolved when the new worker starts, before any merge request arrives,
    for every verify host in config.
    """
    hosts = (_HOST, 'desktop')
    before = _Scene(tmp_path, hosts=hosts)
    before.bench('gate')
    await before.tick()
    before.bench('gate')
    before.bench('wire_shape', host='desktop')
    assert sorted(esc.task_id for esc in before.open_l1s()) == [
        '__verify_host_skew__desktop', f'__verify_host_skew__{_HOST}',
    ]

    restarted = _Scene(tmp_path, hosts=hosts)
    running = asyncio.create_task(restarted.worker.run())
    try:
        for _ in range(500):
            if not restarted.open_l1s():
                break
            await asyncio.sleep(0.01)
    finally:
        await restarted.worker.stop()
        await running

    assert restarted.open_l1s() == []
    restarted.bench('gate')
    assert restarted.host()['sticky'] is False


@pytest.mark.asyncio
async def test_two_skew_failures_in_one_episode_are_not_sticky(scene: _Scene):
    """No passing probe came between them, so the cause has not survived one."""
    scene.bench('gate')
    scene.bench('unparseable')

    assert (scene.host()['kind'], scene.host()['sticky']) == ('unparseable', False)


@pytest.mark.asyncio
async def test_a_later_transport_failure_never_downgrades_a_skew_bench(scene: _Scene):
    scene.bench('gate', reason='uv sync failed')
    scene.bench('transport', reason='ssh: connect to host leo-laptop: Connection timed out')

    host = scene.host()
    assert (host['quarantine_class'], host['kind'], host['reason']) == ('skew', 'gate', 'uv sync failed')
    await scene.tick()
    assert scene.remote.probes == ['ensure']


@pytest.mark.asyncio
async def test_a_transport_failure_never_goes_sticky(scene: _Scene):
    """A sleeping laptop wakes up as often as it likes."""
    for _ in range(3):
        scene.bench('transport')
        await scene.tick()
        assert scene.in_pool()

    assert scene.events.of(EventType.verify_host_skew) == []


@pytest.mark.asyncio
async def test_readmit_resolves_the_skew_l1(scene: _Scene):
    scene.bench('wire_shape')
    assert len(scene.open_l1s()) == 1

    await scene.tick()

    assert scene.open_l1s() == []
    assert scene.events.of(EventType.verify_host_recovered)[-1]['quarantine_class'] == 'skew'


# ---------------------------------------------------------------------------
# The operator clear (escalation tool clear_verify_host_quarantine)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_operator_clear_lifts_a_sticky_bench(scene: _Scene):
    scene.bench('gate')
    await scene.tick()
    scene.bench('gate')
    assert scene.host()['sticky'] is True

    outcome = await scene.worker.clear_host_quarantine(_HOST)

    assert outcome == {
        'host': _HOST, 'cleared': True, 'quarantine_class': 'skew', 'was_sticky': True,
    }
    assert scene.in_pool()
    assert scene.open_l1s() == []
    scene.bench('gate')
    assert scene.host()['sticky'] is False, 'a clear resets the count, as a restart does'


@pytest.mark.asyncio
async def test_the_operator_clear_lifts_a_divergence_quarantine(scene: _Scene):
    lease = scene.allocator.acquire_remote()
    assert lease is not None
    await scene.allocator.quarantine_and_release(lease)

    outcome = await scene.worker.clear_host_quarantine(_HOST)

    assert (outcome['cleared'], outcome['quarantine_class']) == (True, 'divergence')
    assert scene.in_pool()


@pytest.mark.asyncio
@pytest.mark.parametrize('host', [_HOST, 'nowhere'])
async def test_the_operator_clear_says_why_it_cleared_nothing(scene: _Scene, host: str):
    outcome = await scene.worker.clear_host_quarantine(host)

    assert outcome['cleared'] is False
    assert outcome['reason']


# ---------------------------------------------------------------------------
# Neighbours: the 3310 parked-slot backstop and the gate reaper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_parked_slot_backstop_leaves_a_skew_bench_to_the_tracker(scene: _Scene):
    lease = scene.allocator.acquire_remote()
    assert lease is not None
    scene.remote.cancel_rc, scene.remote.clean = 1, False
    await scene.allocator.cancel_and_release(lease, max_attempts=1)
    assert scene.allocator.is_parked(_HOST)
    scene.bench('gate')
    scene.remote.clean = True

    await scene.worker._reprobe_untracked_parked_slots()

    assert scene.allocator.is_parked(_HOST)
    assert not scene.in_pool()


@pytest.mark.asyncio
async def test_the_gate_reaper_runs_at_most_hourly_and_only_on_hosts_in_the_pool(scene: _Scene):
    now = scene.clock.now()
    await scene.worker._reap_idle_gates(now)
    await scene.worker._reap_idle_gates(now + 1800)
    assert scene.remote.reaps == 1

    scene.bench('transport')
    await scene.worker._reap_idle_gates(now + 3601)
    assert scene.remote.reaps == 1


@pytest.mark.asyncio
async def test_the_gate_reaper_skips_a_divergence_quarantined_host(scene: _Scene):
    lease = scene.allocator.acquire_remote()
    assert lease is not None
    await scene.allocator.quarantine_and_release(lease)

    await scene.worker._reap_idle_gates(scene.clock.now())

    assert scene.remote.reaps == 0
