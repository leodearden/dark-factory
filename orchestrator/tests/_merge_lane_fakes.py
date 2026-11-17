"""Fakes for the merge lane's ports (PRD ``plans/merge-lane-quality-prd.md`` task β).

Injected in place of the production adapters through
``MergeLane(..., verifier=FakeVerifier(...), clock=FakeClock(...))``.
``FakeVerifier`` scripts the verify outcome per task id or in call order,
and records each scoped verify as a ``ScopedVerifyCall``;
``FakeRemoteRunner`` is a remote verify host handed to the lane through
``remote_runners=``; ``FakeClock`` is a
hand-advanced clock whose ``sleep`` advances it and waits at most
``sleep_cap`` of real time, and ``healthy_verify_clock`` one on which an
in-flight verify keeps writing;
``VerifyTasks`` builds a verify-frontier entry's ``verify_task`` in each
state it can be in (pending, finished, raised, cancelled), and its
``verifies`` fixture cancels the pending ones at teardown -- re-export it
into a test module as ``from _merge_lane_fakes import verifies as verifies``;
``RecordingEscalations`` stands in for the escalation queue and keeps what
the lane filed; ``lane_state``/``lane_entry`` read an item's state back off
the lane's public ``snapshot()`` census, and ``slot_states`` each verify
host's slot state. ``make_lane`` builds a lane on all three at once, so a test
that owns its worker never falls back to a production adapter by omission;
``running_lane`` runs such a lane for the body of a block and
``merge_through_lane`` until it resolves one request; ``drive_merge`` plays
the merger for a caller that enqueues onto a queue
nothing is draining. ``main_health_probe_spawned`` reads off a red
``MergeOutcome`` whether it left a detached main-health probe running, and
``lane_scene_config`` builds a scene's ``OrchestratorConfig`` with that probe
off. ``branch_merge_request`` commits one root-level file on a fresh task
branch and returns the ``MergeRequest`` that merges it.

Since task 5037 every merge-lane module, and ``suffix_graph``, logs under
exactly ``'orchestrator.merge_lane'``; filter caplog by that name. A
NEGATIVE assertion over records filtered by a logger name nothing emits
under passes vacuously, so pair it with a positive capture.

Imported by bare module name (``from _merge_lane_fakes import ...``), like
``_orch_helpers`` -- ``orchestrator/tests/`` has no ``__init__.py``.
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
from collections.abc import AsyncIterator, Collection, Coroutine, Iterable, Mapping
from pathlib import Path
from typing import Any, ClassVar

import pytest_asyncio
from _merge_worker_drive import make_merge_request

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import MergeLane
from orchestrator.merge_lane.gates import PostMergePyrightResult
from orchestrator.merge_lane.ports import ClockPort, VerifyPort
from orchestrator.merge_lane.remerge_base import RemergeClaim
from orchestrator.merge_lane.reverify_gate_decision import GateClearance
from orchestrator.merge_lane.types import (
    DiskGuardOutcome,
    InflightVerifyResult,
    MergeOutcome,
    MergeRequest,
    RemergeBase,
    RemergeReason,
    SpeculativeItem,
)
from orchestrator.merge_lane.worker import MAIN_HEALTH_PROBE_PENDING_NOTE, SpeculativeMergeWorker
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import RunnerUnavailable


async def remerge_on_main(
    worker: SpeculativeMergeWorker, req: MergeRequest, started_monotonic: float | None = None,
) -> SpeculativeItem:
    """Re-merge *req* under the claim a site makes with no frontier tip:
    onto main, holding no permit."""
    claim = RemergeClaim(RemergeBase(None, RemergeReason.NO_TIP), None)
    return await worker._remerge(req, started_monotonic, claim=claim)


def lane_entry(lane: MergeLane, request_id: str) -> dict[str, Any] | None:
    """The ``snapshot()['entries']`` dict for *request_id*, or None.

    The lane's own public census, looked up by request id -- what a
    dashboard, a heartbeat line or a liveness probe sees.
    """
    for entry in lane.snapshot()['entries']:
        if entry['request_id'] == request_id:
            return entry
    return None


def lane_state(lane: MergeLane, request_id: str) -> str | None:
    """The wire state ``snapshot()`` reports for *request_id*.

    ``None`` means the lane no longer tracks it at all -- retired, or never
    seen. Outside the lane those are the same observation ("done with it"),
    which is why this is the public expression of a retired item.
    """
    entry = lane_entry(lane, request_id)
    return None if entry is None else entry['state']


def slot_states(lane: MergeLane) -> dict[str, str]:
    """Each verify host's slot state, keyed by host name, as the lane's
    public ``snapshot()['hosts']`` census reports it."""
    return {host['name']: host['slot_state'] for host in lane.snapshot()['hosts']}


def lane_finalizing(lane: MergeLane) -> list[dict[str, Any]]:
    """Census entries in the post-verify ``finalizing`` window.

    At most one entry at a time, so ``[]`` is the public reading of "no
    entry is stuck mid-finalize".
    """
    return [e for e in lane.snapshot()['entries'] if e['state'] == 'finalizing']


def main_health_probe_spawned(outcome: MergeOutcome) -> bool:
    """Whether a red *outcome* left a detached main-health probe in flight.

    Reads ``merge_queue.py::MAIN_HEALTH_PROBE_PENDING_NOTE``, which
    ``_run_post_merge_verify`` appends to a red outcome's reason whenever
    ``_spawn_main_health_probe`` put a probe in flight -- the only trace the
    probe leaves on the outcome.

    A lane scene over a real tmp repo must keep this False. The probe adds
    and removes a ``_mainprobe-*`` worktree in the scene's repo while the
    scene carries on, so a concurrent ``git worktree add`` in the scene can
    die with "fatal: Invalid path '<repo>/.git/worktrees/_mainprobe-<hex>'"
    (task 5870); and the probe is a project-wide verify of the tmp repo that
    can outlive the verdict (task 5811). A scene switches it off by building
    its config with ``lane_scene_config``.
    """
    return MAIN_HEALTH_PROBE_PENDING_NOTE in outcome.reason


def lane_scene_config(
    project_root: Path, git: GitConfig, **overrides: Any,
) -> OrchestratorConfig:
    """The ``OrchestratorConfig`` for a merge-lane test scene.

    The main-health probe is off (``escalate_preexisting_main_break=False``);
    ``main_health_probe_spawned`` says why. *overrides* are any further
    ``OrchestratorConfig`` fields the scene needs. A scene that exercises the
    probe builds ``OrchestratorConfig`` itself with
    ``escalate_preexisting_main_break`` named, and
    ``test_lane_scene_config_guard.py`` holds every worker-constructing module
    to that.
    """
    return OrchestratorConfig(
        project_root=project_root, git=git,
        escalate_preexisting_main_break=False, **overrides,
    )


async def branch_merge_request(
    git_ops: GitOps, config: OrchestratorConfig, task_id: str, *, filename: str,
) -> MergeRequest:
    """The request merging task *task_id*'s new branch, one commit adding root-level *filename*.

    A root-level file lies outside every service-restart watched prefix, so
    landing it restarts nothing.
    """
    worktree = (await git_ops.create_worktree(task_id)).path
    (worktree / filename).write_text(f'# {task_id}\n')
    await git_ops.commit(worktree, f'Add {filename}')
    return make_merge_request(task_id, f'{config.git.branch_prefix}{task_id}', worktree, config)


@dataclasses.dataclass(frozen=True)
class VerifyScript:
    """What one ``FakeVerifier.run_scoped`` call does.

    Exactly one of the shapes below applies: return ``result``, raise
    ``error``, or wait for ``release`` first and then return ``result``.
    *entered*, when given, is set the moment a call reaches this script,
    before any release wait; it is the arrival half of a release gate.
    """

    result: VerifyResult
    error: BaseException | None = None
    release: asyncio.Event | None = None
    entered: asyncio.Event | None = None


@dataclasses.dataclass(frozen=True)
class ScopedVerifyCall:
    """What one ``FakeVerifier.run_scoped`` call was handed."""

    task_id: str | None
    worktree: Path
    module_configs: tuple[Any, ...]


def passes(summary: str = 'fake verify passed') -> VerifyScript:
    return VerifyScript(result=VerifyResult(
        passed=True, test_output='', lint_output='', type_output='', summary=summary,
    ))


def fails(*, category: str, summary: str) -> VerifyScript:
    return VerifyScript(result=VerifyResult(
        passed=False, test_output='', lint_output='', type_output='',
        summary=summary, category=category,
    ))


def raises(error: BaseException) -> VerifyScript:
    return VerifyScript(result=passes().result, error=error)


def hangs_until(
    release: asyncio.Event, *, entered: asyncio.Event | None = None,
) -> VerifyScript:
    return VerifyScript(result=passes().result, release=release, entered=entered)


def times_out(summary: str = 'Verification timed out') -> VerifyScript:
    return VerifyScript(result=VerifyResult(
        passed=False, test_output='', lint_output='', type_output='',
        summary=summary, timed_out=True,
    ))


class FakeVerifier:
    """``VerifyPort`` scripted per task id or in call order.

    Each ``run_scoped`` call follows the next of the ordered ``sequence``
    scripts while any remain, then ``scripts[task_id]``, then ``default``.
    It records every task id it was asked about in ``verified`` and what each
    call was handed in ``verify_calls``, one ``ScopedVerifyCall`` per entry
    into the BASE ``run_scoped`` (an override that does not delegate records
    only ``verified``/``entered_count``, through ``_note_entry``).
    ``await_entry(n)`` waits for the *n*-th entry into ``run_scoped``, which
    is how a test waits for a scripted hang to be genuinely under way before
    it probes the lane -- per CALL, so a test that drives two verifies can
    wait for the SECOND one instead of being let through early by the first.
    The gates a merge passes through after a green scoped verify all report
    clean, the disk guard reports *disk_reason* (``None``, the default, being
    "proceed"), and dry-run investigations are recorded in ``investigations``
    rather than run.

    Script a per-CALL sequence with ``sequence=``; a subclass that must vary
    the selection further overrides ``next_script``. A legacy subclass that
    overrides ``run_scoped`` itself must still call ``_note_entry``.
    """

    def __init__(
        self,
        default: VerifyScript | None = None,
        scripts: Mapping[str | None, VerifyScript] | None = None,
        *,
        sequence: Iterable[VerifyScript] = (),
        disk_reason: str | None = None,
    ) -> None:
        self.default = passes() if default is None else default
        self.scripts: dict[str | None, VerifyScript] = dict(scripts or {})
        self._ordered = collections.deque(sequence)
        self.disk_reason = disk_reason
        self.verified: list[str | None] = []
        self.verify_calls: list[ScopedVerifyCall] = []
        self.investigations: list[dict[str, Any]] = []
        self.entered_count = 0
        self._entry_bell = asyncio.Event()

    def _note_entry(self, task_id: str | None) -> None:
        """Record one entry into ``run_scoped`` and wake its waiters."""
        self.verified.append(task_id)
        self.entered_count += 1
        self._entry_bell.set()

    async def await_entry(self, n: int = 1) -> None:
        """Wait until ``run_scoped`` has been entered at least *n* times.

        Nothing here is one-shot: the bell is cleared and re-awaited until
        the COUNT says so, so waiting for the n-th verify cannot be
        satisfied by an earlier one. Callers bound the wait themselves
        (``asyncio.wait_for``), since how long the lane may legitimately take
        to get there is the caller's knowledge, not this fake's.
        """
        while self.entered_count < n:
            self._entry_bell.clear()
            await self._entry_bell.wait()

    def next_script(self, task_id: str | None) -> VerifyScript:
        """The script the current ``run_scoped`` call for *task_id* follows.

        The selection a subclass overrides to vary which script a call
        follows; recording the call and playing the script stay in
        ``run_scoped``.
        """
        if self._ordered:
            return self._ordered.popleft()
        return self.scripts.get(task_id, self.default)

    async def run_scoped(
        self,
        worktree: Path,
        config: Any,
        module_configs: list[Any],
        task_files: list[str] | None = None,
        **options: Any,
    ) -> VerifyResult:
        task_id = options.get('task_id')
        self.verify_calls.append(ScopedVerifyCall(
            task_id=task_id, worktree=worktree, module_configs=tuple(module_configs),
        ))
        self._note_entry(task_id)
        script = self.next_script(task_id)
        if script.entered is not None:
            script.entered.set()
        if script.release is not None:
            await script.release.wait()
        if script.error is not None:
            raise script.error
        return script.result

    async def run_unscoped_typechecks(
        self, worktree: Path, config: Any, module_configs: list[Any], **options: Any,
    ) -> PostMergePyrightResult:
        return PostMergePyrightResult()

    async def check_post_merge_pyright(
        self, advanced_sha: str, git_ops: Any, config: Any, module_configs: list[Any],
        **options: Any,
    ) -> PostMergePyrightResult:
        return PostMergePyrightResult()

    async def check_post_merge_equivalence(
        self, task_worktree: Path, advanced_sha: str, git_ops: Any, main_sha: str,
        **options: Any,
    ) -> list[str]:
        return []

    async def ensure_disk_space(
        self,
        git_ops: Any,
        merge_wt: Path,
        min_free_bytes: int,
        task_id: str,
        keep_worktrees: Collection[Path] | None = None,
    ) -> DiskGuardOutcome:
        return DiskGuardOutcome(reason=self.disk_reason)

    async def cold_shadow(
        self, git_ops: Any, req: Any, merge_commit: str, event_store: Any,
    ) -> dict[str, str]:
        return {}

    def dry_run_unblock(self, **investigation: Any) -> Coroutine[Any, Any, None]:
        self.investigations.append(investigation)
        return _nothing()


async def _nothing() -> None:
    return None


class FakeRemoteRunner:
    """A remote verify host, duck-typed to ``verify_runner.py::VerifyRunner``.

    Not a ``RemoteRunner``, so ``VerifyRunnerPool.dispatch`` calls it with
    ``(merge_sha, spec)`` alone and skips the INV-2 code sync. ``calls`` keeps
    every merge SHA it was asked to verify. A verify waits for *release* when
    one is given, setting *entered* first. An *unreachable* host raises
    ``RunnerUnavailable`` and never probes healthy or clean, so the reprobe
    sweep cannot re-admit it during a test. ``cancel_verify`` answers
    *cancel_rc*. ``live`` is set while a verify runs, as ``RemoteRunner``'s
    in-flight id is; ``cancels`` records whether each remote cancel found a
    verify still live to abort, and ``clean_probes`` counts clean-probes.
    ``reap_idle_gates`` is the no-op a running lane's reprobe loop calls on
    every remote.

    It exposes no ``dispatch_in_flight``, which the lane reads as a live
    dispatch (``orchestrator/merge_lane/types.py::remote_dispatch_live``), so
    a verify held on it is never no-progress-aborted.
    """

    is_local: ClassVar[bool] = False

    def __init__(
        self,
        name: str = 'laptop',
        *,
        release: asyncio.Event | None = None,
        entered: asyncio.Event | None = None,
        unreachable: bool = False,
        cancel_rc: int = 0,
    ) -> None:
        self.name = name
        self.release = release
        self.entered = entered
        self.unreachable = unreachable
        self.cancel_rc = cancel_rc
        self.calls: list[str] = []
        self.live = False
        self.cancels: list[bool] = []
        self.clean_probes = 0

    async def health(self) -> bool:
        return not self.unreachable

    async def probe_clean(self) -> bool:
        self.clean_probes += 1
        return not self.unreachable

    async def cancel_verify(self) -> int:
        self.cancels.append(self.live)
        return self.cancel_rc

    async def reap_idle_gates(self) -> None:
        return None

    async def run_merge_verify(self, merge_sha: str, spec: Any) -> VerifyResult:
        self.calls.append(merge_sha)
        if self.unreachable:
            raise RunnerUnavailable(f'{self.name} is unreachable')
        self.live = True
        try:
            if self.entered is not None:
                self.entered.set()
            if self.release is not None:
                await self.release.wait()
            return VerifyResult(
                passed=True, test_output='', lint_output='', type_output='',
                summary=f'green on {self.name}',
            )
        finally:
            self.live = False


class FakeClock:
    """``ClockPort`` that moves only when told to.

    Two independent readings, exactly as production has them: ``now`` reads
    the wall-clock ``time`` (what the lane stamps with) and ``monotonic``
    reads ``mono`` (what it measures durations against). ``sleep`` advances
    BOTH by the requested seconds and really waits ``min(secs, sleep_cap)``,
    keeping every requested sleep in ``sleeps``. The real wait stops a
    started lane's periodic loops (``_heartbeat_loop``, ``_reprobe_loop``)
    from spinning a core; the cap keeps their production-sized periods off
    the wall clock. ``sleep_cap=0`` restores the bare yield, for a test that
    drives a single loop through a production-sized poll count with nothing
    else needing real time.

    ``wait_for_any`` is the lane's poll cadence, and it charges ``mono``
    the FULL requested timeout while really waiting only ``wait_cap``.
    So a loop that polls on this clock and measures its budget off
    ``monotonic()`` -- the in-flight verify abort-poll is the one that does
    -- reaches that budget with no real time elapsed. Every requested
    timeout lands in ``waits``.

    How much real time the polled task gets before that is NOT dependable.
    A started lane's loops charge 150s (heartbeat 30s + reprobe 120s) per
    ``sleep_cap`` of real sleep and the abort poll charges its full poll
    interval per ``wait_cap``, so at the defaults a parked verify's
    no-progress budget runs out in under a second of real time. Load
    only lengthens that, but it is still no load-safe slack: a scene that must not call a verify dead shows the
    content progress a healthy verify makes -- ``healthy_verify_clock`` --
    and only a scene exercising the dead-verify abort leaves
    ``content_mtime`` frozen.

    ``tick`` additionally advances ``mono`` on every ``monotonic()`` read,
    for a lane loop that measures elapsed time off this clock but waits on
    something else. Keeping the two counters apart is what makes both safe:
    a test that ticks an hour per duration reading does not thereby drag
    every ``now()`` stamp an hour into the future as a side effect of
    however many durations the lane happened to read -- which is also why
    ``wait_for_any`` moves ``mono`` alone while ``sleep``, being real
    elapsed time, moves both.

    ``newest_content_mtime`` reports ``content_mtime`` -- ``None`` or a
    frozen value being a merge worktree nothing is writing to -- and
    advances it by ``content_tick`` per probe, so a ``content_tick`` above
    zero is a verify that keeps writing. Every probed root is kept in
    ``content_probes``, the way ``sleeps`` keeps every requested sleep.
    ``verify_cpu_ticks`` mirrors it for the verify process tree's CPU time:
    ``cpu_ticks`` advanced by ``cpu_tick`` per probe, roots in ``cpu_probes``.
    """

    def __init__(
        self,
        *,
        time: float = 1_000_000.0,
        tick: float = 0.0,
        content_mtime: float | None = None,
        content_tick: float = 0.0,
        cpu_ticks: int | None = None,
        cpu_tick: int = 0,
        wait_cap: float = 0.02,
        sleep_cap: float = 0.02,
    ) -> None:
        self.time = time
        self.mono = time
        self.tick = tick
        self.content_mtime = content_mtime
        self.content_tick = content_tick
        self.cpu_ticks = cpu_ticks
        self.cpu_tick = cpu_tick
        self.wait_cap = wait_cap
        self.sleep_cap = sleep_cap
        self.sleeps: list[float] = []
        self.waits: list[float] = []
        self.content_probes: list[Path] = []
        self.cpu_probes: list[Path] = []

    def now(self) -> float:
        return self.time

    def monotonic(self) -> float:
        reading = self.mono
        self.mono += self.tick
        return reading

    def newest_content_mtime(self, root: Path) -> float | None:
        self.content_probes.append(root)
        reading = self.content_mtime
        if reading is not None:
            self.content_mtime = reading + self.content_tick
        return reading

    def verify_cpu_ticks(self, root: Path) -> int | None:
        self.cpu_probes.append(root)
        reading = self.cpu_ticks
        if reading is not None:
            self.cpu_ticks = reading + self.cpu_tick
        return reading

    async def sleep(self, secs: float) -> None:
        self.sleeps.append(secs)
        self.time += secs
        self.mono += secs
        await asyncio.sleep(min(secs, self.sleep_cap))

    async def wait_for_any(
        self, aws: Collection[asyncio.Future[Any]], timeout: float,
    ) -> set[asyncio.Future[Any]]:
        self.waits.append(timeout)
        # A REAL bounded wait, never a bare yield: a yield reschedules on the
        # ready queue without letting any timer fire, which starves whatever
        # is being polled. The cap is what makes the fake -- not the wall
        # clock -- own the cadence of a production-sized poll.
        done, _ = await asyncio.wait(aws, timeout=min(timeout, self.wait_cap))
        self.mono += timeout
        return done


_FAKE_CLOCK_IS_A_CLOCK_PORT: ClockPort = FakeClock()


def healthy_verify_clock() -> FakeClock:
    """A ``FakeClock`` on which an in-flight verify keeps writing to its merge
    worktree, as a healthy one does, so the lane's no-progress abort never
    calls it dead however far the lane's loops have run ``monotonic()``."""
    return FakeClock(content_mtime=1.0, content_tick=1.0)


VerifyTask = asyncio.Task[InflightVerifyResult]
GateTask = asyncio.Task[MergeOutcome | GateClearance]


class VerifyTasks:
    """Builds verify tasks in each state a frontier owner's verify can be in,
    and cancels whichever are still pending when the test ends."""

    def __init__(self) -> None:
        self._made: list[VerifyTask | GateTask] = []

    def pending(self) -> VerifyTask:
        task = asyncio.create_task(_never_verifies())
        self._made.append(task)
        return task

    def pending_gate(self) -> GateTask:
        task = asyncio.create_task(_never_clears())
        self._made.append(task)
        return task

    async def settled(
        self, coro: Coroutine[Any, Any, InflightVerifyResult],
    ) -> VerifyTask:
        task = asyncio.create_task(coro)
        await asyncio.wait([task])
        return task

    async def finished(self, result: InflightVerifyResult) -> VerifyTask:
        return await self.settled(_returns(result))

    async def raised(self) -> VerifyTask:
        return await self.settled(_raises())

    async def cancelled(self) -> VerifyTask:
        task = self.pending()
        task.cancel()
        await asyncio.wait([task])
        return task

    async def close(self) -> None:
        for task in self._made:
            task.cancel()
        if self._made:
            await asyncio.wait(self._made)


async def _never_verifies() -> InflightVerifyResult:
    await asyncio.Event().wait()
    raise AssertionError('a pending verify never completes')


async def _never_clears() -> MergeOutcome | GateClearance:
    await asyncio.Event().wait()
    raise AssertionError('a pending gate never clears')


async def _returns(result: InflightVerifyResult) -> InflightVerifyResult:
    return result


async def _raises() -> InflightVerifyResult:
    raise RuntimeError('verify blew up')


@pytest_asyncio.fixture
async def verifies() -> AsyncIterator[VerifyTasks]:
    made = VerifyTasks()
    yield made
    await made.close()


class RecordingEscalations:
    """The escalation queue a lane files through, recording instead of filing.

    Duck-types the surface ``ports.ProductionEscalations`` uses on an
    injected ``escalation_queue`` -- ``make_id`` and ``submit`` -- plus the
    ``has_open_l1`` dedup probe the lane asks before filing. Every submitted
    ``Escalation`` lands in ``filed`` in order; ``open_it`` makes the next
    probe report a prior open L1.
    """

    def __init__(self, *, open_l1: bool = False) -> None:
        self.open_l1 = open_l1
        self.filed: list[Any] = []

    def has_open_l1(self, task_id: str, *, category: str | None = None) -> bool:
        return self.open_l1

    def make_id(self, task_id: str) -> str:
        return f'esc-{task_id}-{len(self.filed) + 1}'

    def submit(self, escalation: Any) -> None:
        self.filed.append(escalation)

    def open_it(self) -> None:
        self.open_l1 = True


def make_lane(
    git_ops: Any,
    queue: asyncio.Queue[Any] | None = None,
    *,
    verifier: VerifyPort | None = None,
    clock: ClockPort | None = None,
    escalation_queue: Any = None,
    **kwargs: Any,
) -> MergeLane:
    """A ``MergeLane`` on the fakes, for a test that drives its own worker.

    The verifier and clock default to fresh fakes rather than to the
    production adapters, so a test that forgets to pass one still gets a
    lane that neither shells out nor waits out a production-sized sleep.
    *queue* defaults to a fresh ``asyncio.Queue`` on the running loop.
    Anything else ``MergeLane`` accepts passes through in *kwargs*.
    """
    return MergeLane(
        git_ops,
        asyncio.Queue() if queue is None else queue,
        escalation_queue=escalation_queue,
        verifier=FakeVerifier() if verifier is None else verifier,
        clock=FakeClock() if clock is None else clock,
        **kwargs,
    )


@dataclasses.dataclass(frozen=True)
class LaneRun:
    """A lane's ``run()`` task, as ``running_lane`` yields it."""

    task: asyncio.Task[None]

    async def outcome(self, request: Any, *, timeout: float = 120.0) -> MergeOutcome:
        """The outcome the lane resolves *request* with.

        Watches the ``run()`` task while waiting, so a lane that crashes or
        returns first raises that, instead of surfacing as a timeout.
        """
        done, _ = await asyncio.wait(
            {self.task, request.result}, timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if request.result in done:
            return request.result.result()
        if self.task in done:
            self.task.result()
            raise AssertionError('the lane stopped before resolving the request')
        raise TimeoutError(f'the lane did not resolve the request within {timeout}s')


@contextlib.asynccontextmanager
async def running_lane(
    lane: MergeLane, *, stop_timeout: float = 60.0,
) -> AsyncIterator[LaneRun]:
    """Run *lane* for the body of the block, then stop it.

    On the way out, whatever happened, the lane is stopped and its ``run()``
    awaited, so a failure of ``run()`` itself is raised rather than lost.
    """
    run = asyncio.create_task(lane.run())
    try:
        yield LaneRun(run)
    finally:
        await lane.stop()
        await asyncio.wait_for(run, stop_timeout)


async def merge_through_lane(
    lane: MergeLane,
    queue: asyncio.Queue[Any],
    request: Any,
    *,
    timeout: float = 120.0,
) -> MergeOutcome:
    """Run *lane* until it resolves *request*, then stop it; return the outcome.

    *queue* is the one *lane* was built on. The request -- a single
    ``MergeRequest`` or a train's ``GroupMergeRequest`` -- goes in through the
    lane's queue like any submission, so it takes the production dispatch
    route rather than a test-chosen internal entry point.
    """
    async with running_lane(lane) as run:
        await queue.put(request)
        return await run.outcome(request, timeout=timeout)


@dataclasses.dataclass(frozen=True)
class DrivenMerge:
    """What one ``drive_merge`` produced.

    *result* is what the driven coroutine returned; *request* is the REAL
    ``MergeRequest`` the production enqueue path parked on the queue, for a
    caller that wants to read what was actually enqueued.
    """

    result: Any
    request: Any


async def drive_merge(
    submit: Coroutine[Any, Any, Any],
    queue: asyncio.Queue[Any],
    outcome: Any,
    *,
    timeout: float = 10.0,
) -> DrivenMerge:
    """Run *submit* to completion, playing the merger for what it enqueues.

    Nothing drains a queue a test owns, so the request the production
    enqueue path parks there would wait forever: this takes it off and
    resolves it with *outcome*, delivering the result through the REAL
    ``MergeRequest`` future the caller is awaiting rather than substituting
    the lane's entry point.

    The wait for that request RACES the driven coroutine, so a production
    path that raises BEFORE it ever enqueues -- the common way these break
    -- surfaces its own traceback here instead of a bare ``TimeoutError``
    from an empty queue. Both tasks are awaited after cancellation, so a
    failure never escapes as cross-test "Task exception was never
    retrieved" noise.
    """
    driven = asyncio.ensure_future(submit)
    getter = asyncio.ensure_future(queue.get())
    try:
        done, _pending = await asyncio.wait(
            (driven, getter), timeout=timeout, return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            raise AssertionError(
                f'nothing was enqueued within {timeout}s and the driven '
                f'coroutine is still running'
            )
        if driven in done and not getter.done():
            driven.result()  # re-raises the production failure, if there was one
            raise AssertionError(
                'the driven coroutine finished without enqueuing a merge request'
            )
        request = await getter
        request.result.set_result(outcome)
        return DrivenMerge(
            result=await asyncio.wait_for(driven, timeout=timeout), request=request,
        )
    finally:
        for task in (driven, getter):
            task.cancel()
            # Retrieve whatever each task settled on -- including an
            # exception already re-raised above, which must not replace the
            # one propagating out of the try block.
            with contextlib.suppress(BaseException):
                await task
