"""DISPATCH-FILL redispatch-drain tests for task 3276.

Pins two invariants of the DISPATCH-FILL loop's early-stop guard
(``orchestrator/src/orchestrator/merge_queue.py::SpeculativeMergeWorker._verifier_loop``):

* A re-dispatch that drains the lane's redispatch park must not end the fill
  pass while another item is ready to verify and a host slot is free --
  whether that item was already waiting before the re-dispatch, or arrives a
  moment after.
* Once the park and the ready queue are genuinely both empty, the loop must
  still fall through to FINALIZE-HEAD rather than hang -- the anti-deadlock
  property a since-deleted redispatch-specific special case used to provide
  is now structural, coming from the fill loop's other fall-through paths.

See merge_queue.py's comment on the guard (immediately above
``allocator = self._ensure_host_allocator(...)`` in the DISPATCH-FILL tail)
for the current predicate and the traced fall-through argument for why the
anti-deadlock property holds without a redispatch-specific special case.

HOW THESE DRIVE THE LANE (task 5030, PRD ``plans/merge-lane-quality-prd.md``
task γ7). Every test here runs a REAL lane over a REAL git repo through its
public surface only: requests go in on the public queue, verify outcomes are
scripted on the injected ``FakeVerifier``/fake remote runner, and every
observation is read off ``MergeLane.snapshot()``. Nothing reaches into the
lane's internals, so a restructuring of the fill loop that preserves these
invariants does not break these tests.

The redispatch park is produced the way production produces it: the head's
verify FAILS, and the head-failure cascade re-merges its in-flight follower
and parks it for re-dispatch (``snapshot()['entries']`` reports that item as
``awaiting_host``). That leaves both host slots free with one parked item, so
the next fill pass is exactly the redispatch-sourced-dispatch window the guard
governs. Verified as a genuine fence, not just a green test: restoring the
pre-3276 clause (``if not is_from_verifier_queue and not self._redispatch:
fill_done = True``) makes
``test_second_host_dispatched_from_ready_queue_in_same_fill_pass`` and
``test_late_arrival_dispatched_to_free_host_while_head_verify_runs`` fail with
the follower alone in flight and the second item stranded at
``awaiting_verify`` beside a free host.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import AsyncIterator, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# COMPAT RE-EXPORT, task 5030. The white-box drive harness these tests used to
# run on now lives in _fill_drive_harness.py, but
# test_merge_queue_verifier_raw_cancel.py (a different γ group, not editable
# from here) still imports it FROM THIS MODULE. Re-exported so that import
# keeps resolving; delete both this block and _fill_drive_harness.py when that
# file's group migrates or retires it.
from _fill_drive_harness import (  # noqa: F401,E402
    _drive_fill,
    _make_real_item,
    _teardown_fill_drive,
)
from _merge_lane_fakes import FakeVerifier, VerifyScript, fails, hangs_until, passes
from _orch_helpers import MERGE_RESULT_TIMEOUT, wait_responsive
from test_merge_queue_concurrent_verify import (
    _inject_two_host_allocator,
    _make_branch_with_file,
    _make_request,
    config,  # noqa: F401 — pytest fixture re-exported from γ harness
    git_config,  # noqa: F401 — pytest fixture re-exported from γ harness
    git_ops,  # noqa: F401 — pytest fixture re-exported from γ harness
    git_repo,  # noqa: F401 — pytest fixture re-exported from γ harness
)

from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import MergeLane
from orchestrator.verify_runner import HostAllocator, HostLease, RunnerUnavailable

#: The item whose verify fails, triggering the head-failure cascade.
HEAD = 'rd-head'
#: The in-flight follower the cascade re-merges and parks for re-dispatch.
FOLLOWER = 'rd-follower'
#: The second item, ready (or arriving) while the parked follower re-dispatches.
SECOND = 'rd-second'

_POLL_INTERVAL = 0.02
_STOP_TIMEOUT = 30.0


def _hanging_remote(release: asyncio.Event, name: str = 'laptop') -> MagicMock:
    """A fake RemoteRunner whose every verify blocks until *release* is set.

    Deliberately not ``test_merge_queue_concurrent_verify._gated_runner``,
    which blocks only its FIRST call: these tests need the second host's
    verify to stay in flight too, so that both leases are observably held at
    the same time.
    """
    async def _run_merge_verify(*args: Any, **kwargs: Any) -> Any:
        await release.wait()
        return passes().result

    runner = MagicMock()
    runner.name = name
    runner.is_local = False
    runner.run_merge_verify = AsyncMock(side_effect=_run_merge_verify)
    runner.cancel_verify = AsyncMock(return_value=0)  # 0 = clean cancel
    runner.probe_clean = AsyncMock(return_value=True)
    return runner


@dataclasses.dataclass
class _Lane:
    """A running two-host :class:`MergeLane` plus the gates scripting its verifies.

    ``head_gate``   : release -> ``HEAD``'s local verify FAILS (the cascade trigger).
    ``local_gate``  : release -> every other LOCAL verify passes.
    ``remote_gate`` : release -> every REMOTE verify passes.

    Until a gate is released the corresponding verify hangs, so a dispatched
    item stays in flight holding its host lease -- which is what makes
    "both slots held at once" a measurement rather than a call count.
    """

    worker: Any
    queue: asyncio.Queue
    git_ops: GitOps
    config: OrchestratorConfig
    head_gate: asyncio.Event
    local_gate: asyncio.Event
    remote_gate: asyncio.Event
    allocator: HostAllocator
    remote: Any
    requests: dict[str, Any] = dataclasses.field(default_factory=dict)
    run_task: asyncio.Task | None = None  # type: ignore[type-arg]

    async def enqueue(self, task_id: str) -> Any:
        """Commit a one-file branch for *task_id* and submit it on the public queue."""
        worktree = await _make_branch_with_file(
            self.git_ops, f'task/{task_id}', f'{task_id.replace("-", "_")}.py',
            f'# {task_id}\n',
        )
        request = _make_request(task_id, f'task/{task_id}', worktree, self.config)
        self.requests[task_id] = request
        await self.queue.put(request)
        return request

    def inflight_ids(self) -> set[str]:
        """Task ids holding a host lease right now, from the public snapshot."""
        by_host = self.worker.snapshot()['occupancy']['inflight_by_host']
        return {task_id for ids in by_host.values() for task_id in ids}

    def states(self) -> dict[str, str]:
        """{task_id: pipeline state} for every item the lane is tracking."""
        return {e['task_id']: e['state'] for e in self.worker.snapshot()['entries']}

    def free_hosts(self) -> int:
        """Host slots the allocator reports FREE."""
        return sum(1 for h in self.worker.snapshot()['hosts'] if h['slot_state'] == 'free')

    def observed(self) -> str:
        return (
            f'states={self.states()!r} inflight={sorted(self.inflight_ids())!r} '
            f'free_hosts={self.free_hosts()}'
        )

    async def settle(self, predicate: Callable[[], bool], *, expected: str, why: str) -> None:
        """Wait for *predicate* to hold of the public snapshot, or fail with *why*."""
        async def _poll() -> None:
            while not predicate():
                await asyncio.sleep(_POLL_INTERVAL)

        try:
            await asyncio.wait_for(_poll(), timeout=MERGE_RESULT_TIMEOUT)
        except TimeoutError:
            pytest.fail(
                f'{expected} was never observed within {MERGE_RESULT_TIMEOUT}s.\n'
                f'{why}\nlast observation: {self.observed()}'
            )


@contextlib.asynccontextmanager
async def _running_lane(
    git_ops: GitOps,
    config: OrchestratorConfig,
    *,
    remote: Callable[[asyncio.Event], Any] = _hanging_remote,
) -> AsyncIterator[_Lane]:
    """Start a two-host lane with scripted verifies; stop it on the way out.

    *remote* builds the second host's runner from the lane's ``remote_gate``.

    Teardown releases every gate before stopping, so a failing test leaves no
    verify hanging and no dangling task behind, and goes through
    ``worker.stop()`` -- the lane's own shutdown protocol, which resolves
    in-flight request futures, drains its queues, cleans merge worktrees and
    releases leases, and is internally bounded so it cannot hang this helper.
    """
    head_gate = asyncio.Event()
    local_gate = asyncio.Event()
    remote_gate = asyncio.Event()
    verifier = FakeVerifier(
        default=hangs_until(local_gate),
        scripts={
            HEAD: VerifyScript(
                result=fails(category='test_failure', summary='head verify failed').result,
                release=head_gate,
            ),
        },
    )
    queue: asyncio.Queue = asyncio.Queue()
    worker = MergeLane(git_ops, queue, speculation_depth=2, verifier=verifier)
    remote_runner = remote(remote_gate)
    allocator = _inject_two_host_allocator(worker, remote_runner)
    lane = _Lane(
        worker=worker, queue=queue, git_ops=git_ops, config=config,
        head_gate=head_gate, local_gate=local_gate, remote_gate=remote_gate,
        allocator=allocator, remote=remote_runner,
    )
    worker_task = asyncio.ensure_future(worker.run())
    lane.run_task = worker_task
    try:
        yield lane
    finally:
        head_gate.set()
        local_gate.set()
        remote_gate.set()
        # Exception, not BaseException: this must not swallow a CancelledError
        # aimed at the enclosing test task (or a KeyboardInterrupt) -- only
        # absorb worker.stop()'s own failures, so a genuine stop() regression
        # is never silently masked into a falsely-clean teardown.
        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker.stop(), timeout=_STOP_TIMEOUT)
        worker_task.cancel()
        await asyncio.gather(worker_task, return_exceptions=True)


async def _dispatch_head_and_follower(lane: _Lane) -> None:
    """Submit HEAD and FOLLOWER and wait until both hold a host lease."""
    await lane.enqueue(HEAD)
    await lane.enqueue(FOLLOWER)
    await lane.settle(
        lambda: lane.inflight_ids() == {HEAD, FOLLOWER},
        expected=f'{HEAD} and {FOLLOWER} both verifying on their own host',
        why=(
            'precondition for every test here: the head-failure cascade only '
            'parks a follower that was itself in flight.'
        ),
    )


async def _fail_head_and_park_follower(lane: _Lane) -> None:
    """Fail HEAD's verify and wait for the cascade to park FOLLOWER.

    The cascade cancels FOLLOWER's verify, re-merges it against main and parks
    it for re-dispatch -- publicly, FOLLOWER reads ``awaiting_host`` and both
    host slots come free. That is the state the DISPATCH-FILL guard governs.
    """
    lane.head_gate.set()
    outcome = await wait_responsive(
        lane.requests[HEAD].result, label=f'{HEAD} verify-fail cascade'
    )
    assert outcome.status not in ('done', 'already_merged'), (
        f'expected {HEAD} to fail its verify and cascade, got {outcome.status!r}'
    )


@pytest.mark.asyncio
class TestRedispatchDrainDoesNotEndFillPass:
    """A re-dispatch that drains the redispatch park must not end the
    DISPATCH-FILL pass while another item is ready to verify and a host slot
    is free -- task 3276.

    Two shapes are pinned: the item already ready before the guard evaluates
    (below), and the item arriving a moment after (see
    ``test_late_arrival_dispatched_to_free_host_while_head_verify_runs``).
    """

    async def test_second_host_dispatched_from_ready_queue_in_same_fill_pass(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """A re-dispatch that drains the park must NOT end the fill pass while
        a ready item and a free host slot remain.

        SECOND is dispatched to the free second host in the SAME fill pass,
        immediately after the parked FOLLOWER -- both leases held
        simultaneously. A regression here means the fill pass ends as soon as
        the park drains, leaving SECOND stranded at ``awaiting_verify`` and the
        second host slot idle while the loop blocks in FINALIZE-HEAD awaiting
        FOLLOWER's verify instead.
        """
        async with _running_lane(git_ops, config) as lane:
            await _dispatch_head_and_follower(lane)

            await lane.enqueue(SECOND)
            await lane.settle(
                lambda: lane.states().get(SECOND) == 'awaiting_verify',
                expected=f'{SECOND} merged and ready to verify',
                why=(
                    'the guard only bites when a ready item is already waiting '
                    'at the moment the park drains.'
                ),
            )

            await _fail_head_and_park_follower(lane)

            await lane.settle(
                lambda: lane.inflight_ids() == {FOLLOWER, SECOND},
                expected=(
                    f'{FOLLOWER} (re-dispatched from the park) and {SECOND} '
                    f'(from the ready queue) both verifying'
                ),
                why=(
                    'This means the DISPATCH-FILL guard is ending the fill pass '
                    'as soon as the redispatch park drains, even though a ready '
                    'item and a free host slot remain -- the loop falls through '
                    "to FINALIZE-HEAD and blocks on the re-dispatched item's "
                    'verify instead of filling the free second host in the same '
                    'fill pass.'
                ),
            )
            assert lane.free_hosts() == 0, (
                'expected both host slots held simultaneously; '
                f'{lane.observed()}'
            )

    async def test_late_arrival_dispatched_to_free_host_while_head_verify_runs(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """A late-arriving item -- one that becomes ready a moment AFTER the
        DISPATCH-FILL guard has already evaluated -- must still be dispatched
        to a free host while the head's verify is still running, not left to
        wait out the entire head verify.

        Same shape as
        ``test_second_host_dispatched_from_ready_queue_in_same_fill_pass``
        except nothing is ready when the park drains: SECOND is submitted only
        after FOLLOWER has already been re-dispatched and the guard has already
        run.

        Dispatching FOLLOWER falls through to the QueueEmpty multi-host
        fill-ahead race, which picks SECOND up the moment it is ready and
        dispatches it to the free second host. A guard that merely re-checks
        whether anything is ready at decision time is NOT sufficient to pass
        this test: nothing is ready at the instant the guard evaluates, so a
        snapshot-based guard would still end the fill pass there instead of
        letting the fall-through paths keep filling.
        """
        async with _running_lane(git_ops, config) as lane:
            await _dispatch_head_and_follower(lane)
            await _fail_head_and_park_follower(lane)

            await lane.settle(
                lambda: lane.inflight_ids() == {FOLLOWER},
                expected=f'{FOLLOWER} re-dispatched from the park, alone',
                why=(
                    'the late-arrival shape starts from a re-dispatched item '
                    'verifying with nothing else ready.'
                ),
            )

            # The guard has now evaluated with nothing ready. SECOND arrives
            # NOW, a moment later, with FOLLOWER's verify still running and a
            # host slot still free.
            await lane.enqueue(SECOND)

            await lane.settle(
                lambda: lane.inflight_ids() == {FOLLOWER, SECOND},
                expected=f'{SECOND} dispatched to the free host beside {FOLLOWER}',
                why=(
                    'This means the DISPATCH-FILL guard is stopping the fill '
                    'pass on an empty-queue snapshot taken before SECOND '
                    'arrived, so the loop commits to FINALIZE-HEAD and blocks '
                    "on the re-dispatched item's verify -- SECOND arriving "
                    'moments later changes nothing. A free host must keep '
                    'falling through to the QueueEmpty fill-ahead race instead '
                    'of a special-cased early exit.'
                ),
            )
            assert lane.free_hosts() == 0, (
                'expected both host slots held simultaneously; '
                f'{lane.observed()}'
            )


@pytest.mark.asyncio
class TestCascadeAntiDeadlockPreserved:
    """Fences the anti-deadlock property the original DISPATCH-FILL guard's
    comment claimed to provide -- task 3276.

    Not a regression pin on a specific fix, but a permanent invariant fence:
    this test must stay green regardless of how the DISPATCH-FILL guard is
    implemented, proving that draining the redispatch park does not
    reintroduce the "blocking on the ready queue ... would deadlock when the
    queue is empty after a cascade" hazard the original guard's comment named
    -- see merge_queue.py's comment on the guard for the traced fall-through
    argument this test backs up empirically.
    """

    async def test_empty_queue_after_redispatch_drain_still_finalizes_head(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """The park drains to empty, nothing else is ever submitted, and both
        host slots are free.

        FINALIZE-HEAD must still be reached and the re-dispatched item's
        merge() caller must still get its result once that item's verify
        completes on its own -- i.e. the loop must not hang forever waiting for
        a queue arrival that never comes.
        """
        async with _running_lane(git_ops, config) as lane:
            await _dispatch_head_and_follower(lane)
            await _fail_head_and_park_follower(lane)

            await lane.settle(
                lambda: lane.inflight_ids() == {FOLLOWER},
                expected=f'{FOLLOWER} re-dispatched from the park, alone',
                why=(
                    'this is the exact cascade-drain shape the original '
                    "guard's comment described: nothing follows it."
                ),
            )

            # FOLLOWER's re-dispatched verify completes on its own now. Nothing
            # is ready behind it and nothing ever arrives.
            lane.local_gate.set()
            lane.remote_gate.set()

            try:
                outcome = await asyncio.wait_for(
                    lane.requests[FOLLOWER].result, timeout=MERGE_RESULT_TIMEOUT
                )
            except TimeoutError:
                pytest.fail(
                    f"{FOLLOWER}'s result was never resolved within "
                    f'{MERGE_RESULT_TIMEOUT}s of its verify completing, with '
                    'nothing ready behind it the whole time. FINALIZE-HEAD was '
                    'never reached -- the anti-deadlock property the original '
                    f'guard was written to provide has been lost. '
                    f'last observation: {lane.observed()}'
                )

            assert outcome.status == 'done', f'expected a done outcome, got {outcome!r}'


# ---------------------------------------------------------------------------
# task 5303 (3277 + 4929): a host that becomes usable while the head verifies
# ---------------------------------------------------------------------------

#: A head whose local verify hangs until ``local_gate``, which these tests never
#: release -- so any dispatch they observe happened while it was still verifying.
LONG_HEAD = 'hf-head'
#: The item waiting for the second host.
WAITER = 'hf-waiter'
#: Short enough that "within one poll slice" is far inside MERGE_RESULT_TIMEOUT.
_FAST_POLL_SECS = 0.05


def _remote_lease(lane: _Lane) -> HostLease:
    return HostLease(name=lane.remote.name, runner=lane.remote, is_local=False)


def _hosts_of(lane: _Lane) -> dict[str, list[str]]:
    return lane.worker.snapshot()['occupancy']['inflight_by_host']


def _unavailable_once_remote(release: asyncio.Event, name: str = 'laptop') -> MagicMock:
    """A fake RemoteRunner whose FIRST verify raises RunnerUnavailable once the
    runner's ``unavailable_gate`` is set; every later verify hangs until
    *release*, like :func:`_hanging_remote`.
    """
    calls = 0
    unavailable_gate = asyncio.Event()

    async def _run_merge_verify(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            await unavailable_gate.wait()
            raise RunnerUnavailable('laptop dropped off the network')
        await release.wait()
        return passes().result

    runner = _hanging_remote(release, name)
    runner.run_merge_verify = AsyncMock(side_effect=_run_merge_verify)
    runner.unavailable_gate = unavailable_gate
    return runner


async def _assert_still_waiting(lane: _Lane, task_id: str) -> None:
    """*task_id* stays off every host across several poll slices."""
    for _ in range(10):
        await asyncio.sleep(_FAST_POLL_SECS)
        assert task_id not in lane.inflight_ids(), (
            f'{task_id} was dispatched although no host was acquirable; '
            f'{lane.observed()}'
        )


@pytest.mark.asyncio
class TestHostFreedWhileHeadVerifies:
    """A host that becomes usable mid-head-verify is used within one poll
    slice, not after the head's verify finishes -- task 5303 (3277, 4929).

    Every head here verifies on local and is never released, so a dispatch
    observed at all is a dispatch bounded by ``VERIFY_ABANDON_POLL_SECS``
    rather than by the head's verify.
    """

    async def test_cleared_quarantine_dispatches_waiting_item_beside_verifying_head(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        async with _running_lane(git_ops, config) as lane:
            lane.worker.VERIFY_ABANDON_POLL_SECS = _FAST_POLL_SECS
            await lane.allocator.quarantine_and_release(_remote_lease(lane))

            await lane.enqueue(LONG_HEAD)
            await lane.settle(
                lambda: _hosts_of(lane) == {'local': [LONG_HEAD]},
                expected=f'{LONG_HEAD} verifying on local',
                why='the second host is quarantined, so the head must take local.',
            )
            await lane.enqueue(WAITER)
            await lane.settle(
                lambda: lane.states().get(WAITER) in ('awaiting_verify', 'awaiting_host'),
                expected=f'{WAITER} merged and waiting for a host',
                why='the merger merges ahead while the head verifies.',
            )
            await _assert_still_waiting(lane, WAITER)

            lane.allocator.clear_quarantine(lane.remote.name)

            await lane.settle(
                lambda: _hosts_of(lane).get(lane.remote.name) == [WAITER],
                expected=f'{WAITER} dispatched to the un-quarantined second host',
                why=(
                    'FINALIZE-HEAD is waiting out the head verify without '
                    're-entering DISPATCH-FILL, so a host that becomes usable '
                    'mid-verify idles until the head finishes.'
                ),
            )
            assert lane.states()[LONG_HEAD] == 'verifying', lane.observed()
            assert not lane.requests[LONG_HEAD].result.done(), lane.observed()

    async def test_parked_item_dispatched_when_readmitted_host_frees_mid_head_verify(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """The awaiting_host shape: prefer_remote puts LONG_HEAD on the laptop
        and WAITER on local; the laptop's verify raises RunnerUnavailable, so
        LONG_HEAD's finalize quarantines the laptop and the cascade parks both.
        LONG_HEAD re-dispatches to local and WAITER parks ``awaiting_host``
        until the reprobe primitive readmits the laptop.
        """
        prefer_remote = config.model_copy(update={'verify_host_policy': 'prefer_remote'})
        async with _running_lane(
            git_ops, prefer_remote, remote=_unavailable_once_remote,
        ) as lane:
            lane.worker.VERIFY_ABANDON_POLL_SECS = _FAST_POLL_SECS
            await lane.enqueue(LONG_HEAD)
            await lane.enqueue(WAITER)
            await lane.settle(
                lambda: _hosts_of(lane) == {lane.remote.name: [LONG_HEAD], 'local': [WAITER]},
                expected=f'{LONG_HEAD} on the laptop and {WAITER} on local',
                why='prefer_remote fills the laptop first, then overflows to local.',
            )

            lane.remote.unavailable_gate.set()
            await lane.settle(
                lambda: (
                    _hosts_of(lane) == {'local': [LONG_HEAD]}
                    and lane.states().get(WAITER) == 'awaiting_host'
                ),
                expected=f'{LONG_HEAD} re-dispatched to local, {WAITER} parked awaiting_host',
                why='RunnerUnavailable quarantines the laptop and the cascade parks WAITER.',
            )
            await _assert_still_waiting(lane, WAITER)

            lane.allocator.readmit(lane.remote.name)

            await lane.settle(
                lambda: _hosts_of(lane).get(lane.remote.name) == [WAITER],
                expected=f'{WAITER} dispatched from the park to the readmitted laptop',
                why=(
                    'nothing drains the redispatch park while FINALIZE-HEAD waits '
                    'out the head verify.'
                ),
            )
            assert lane.states()[LONG_HEAD] == 'verifying', lane.observed()

    async def test_parked_item_dispatched_when_host_frees_with_nothing_in_flight(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Nothing in flight: the laptop is quarantined and local is held
        through the public allocator, as merge_drift.py::_run_drift_check
        holds it.  The single item parks ``awaiting_host``; returning local
        must dispatch it with no further submission to wake the loop.
        """
        async with _running_lane(git_ops, config) as lane:
            lane.worker.VERIFY_ABANDON_POLL_SECS = _FAST_POLL_SECS
            await lane.allocator.quarantine_and_release(_remote_lease(lane))
            drift_check_lease = lane.allocator.acquire_local(lambda: lane.remote)
            assert drift_check_lease is not None

            await lane.enqueue(WAITER)
            await lane.settle(
                lambda: lane.states().get(WAITER) == 'awaiting_host',
                expected=f'{WAITER} parked awaiting_host',
                why='no host is acquirable, so the dispatch attempt must park it.',
            )
            assert lane.inflight_ids() == set(), lane.observed()
            await _assert_still_waiting(lane, WAITER)

            await lane.allocator.release(drift_check_lease)

            await lane.settle(
                lambda: _hosts_of(lane) == {'local': [WAITER]},
                expected=f'{WAITER} dispatched to the returned local slot',
                why=(
                    'with nothing in flight the loop is blocked on the verifier '
                    'queue, so a host freeing up never wakes it.'
                ),
            )

    async def test_stop_ends_the_parked_item_poll_promptly(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """stop() during the parked-item poll must not wait out a poll slice.

        The default VERIFY_ABANDON_POLL_SECS (10 s) outlasts stop()'s 5 s
        shutdown wait, after which stop()'s re-drain consumes the None
        sentinel -- so a poll that only notices stop() at its next slice
        leaves the verifier blocked on an empty queue for good.
        """
        async with _running_lane(git_ops, config) as lane:
            await lane.allocator.quarantine_and_release(_remote_lease(lane))
            assert lane.allocator.acquire_local(lambda: lane.remote) is not None
            await lane.enqueue(WAITER)
            await lane.settle(
                lambda: lane.states().get(WAITER) == 'awaiting_host',
                expected=f'{WAITER} parked awaiting_host with nothing in flight',
                why='no host is acquirable, so the dispatch attempt must park it.',
            )
            assert lane.inflight_ids() == set(), lane.observed()
            assert lane.run_task is not None

            loop = asyncio.get_running_loop()
            started = loop.time()
            await lane.worker.stop()
            stop_secs = loop.time() - started

            try:
                await asyncio.wait_for(asyncio.shield(lane.run_task), timeout=5.0)
            except TimeoutError:
                pytest.fail(
                    'the lane never finished after stop(): the verifier loop '
                    'slept through stop() in the parked-item poll and is now '
                    'blocked on a verifier queue whose sentinel was drained.'
                )
            assert stop_secs < 4.0, (
                f'stop() took {stop_secs:.2f}s -- it waited out its shutdown '
                'timeout instead of the verifier loop exiting on its sentinel'
            )
