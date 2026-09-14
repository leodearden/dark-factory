"""Fakes for the merge lane's ports (PRD ``plans/merge-lane-quality-prd.md`` task β).

Injected in place of the production adapters through
``MergeLane(..., verifier=FakeVerifier(...), clock=FakeClock(...))``.
``FakeVerifier`` scripts the verify outcome per task id; ``FakeClock`` is a
hand-advanced clock whose ``sleep`` advances it instead of waiting;
``RecordingEscalations`` stands in for the escalation queue and keeps what
the lane filed; ``lane_state``/``lane_entry`` read an item's state back off
the lane's public ``snapshot()`` census. ``make_lane`` builds a lane on all three at once, so a test
that owns its worker never falls back to a production adapter by omission,
and ``drive_merge`` plays the merger for a caller that enqueues onto a queue
nothing is draining.

Imported by bare module name (``from _merge_lane_fakes import ...``), like
``_orch_helpers`` -- ``orchestrator/tests/`` has no ``__init__.py``.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import Collection, Coroutine, Mapping
from pathlib import Path
from typing import Any

from orchestrator.merge_gates import PostMergePyrightResult
from orchestrator.merge_lane import MergeLane
from orchestrator.merge_lane.types import DiskGuardOutcome
from orchestrator.verify import VerifyResult


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


def lane_finalizing(lane: MergeLane) -> list[dict[str, Any]]:
    """Census entries in the post-verify ``finalizing`` window.

    At most one entry at a time, so ``[]`` is the public reading of "no
    entry is stuck mid-finalize".
    """
    return [e for e in lane.snapshot()['entries'] if e['state'] == 'finalizing']


@dataclasses.dataclass(frozen=True)
class VerifyScript:
    """What ``FakeVerifier.run_scoped`` does for one task.

    Exactly one of the shapes below applies: return ``result``, raise
    ``error``, or wait for ``release`` first and then return ``result``.
    """

    result: VerifyResult
    error: BaseException | None = None
    release: asyncio.Event | None = None


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


def hangs_until(release: asyncio.Event) -> VerifyScript:
    return VerifyScript(result=passes().result, release=release)


class FakeVerifier:
    """``VerifyPort`` scripted per task id.

    ``run_scoped`` follows ``scripts[task_id]``, or ``default`` for a task
    without a script, and records every task id it was asked about in
    ``verified``. ``await_entry(n)`` waits for the *n*-th entry into
    ``run_scoped``, which is how a test waits for a scripted hang to be
    genuinely under way before it probes the lane -- per CALL, so a test
    that drives two verifies can wait for the SECOND one instead of being
    let through early by the first. The gates a merge passes through after a
    green scoped verify all report clean, the disk guard reports
    *disk_reason* (``None``, the default, being "proceed"), and dry-run
    investigations are recorded in ``investigations`` rather than run.

    A subclass that overrides ``run_scoped`` to script a per-CALL sequence
    calls ``_note_entry`` itself, so ``verified`` and ``entered_count`` stay
    truthful for it too.
    """

    def __init__(
        self,
        default: VerifyScript | None = None,
        scripts: Mapping[str | None, VerifyScript] | None = None,
        *,
        disk_reason: str | None = None,
    ) -> None:
        self.default = passes() if default is None else default
        self.scripts: dict[str | None, VerifyScript] = dict(scripts or {})
        self.disk_reason = disk_reason
        self.verified: list[str | None] = []
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

    async def run_scoped(
        self,
        worktree: Path,
        config: Any,
        module_configs: list[Any],
        task_files: list[str] | None = None,
        **options: Any,
    ) -> VerifyResult:
        task_id = options.get('task_id')
        self._note_entry(task_id)
        script = self.scripts.get(task_id, self.default)
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


class FakeClock:
    """``ClockPort`` that moves only when told to.

    Two independent readings, exactly as production has them: ``now`` reads
    the wall-clock ``time`` (what the lane stamps with) and ``monotonic``
    reads ``mono`` (what it measures durations against). ``sleep`` advances
    BOTH by the requested seconds and yields once so other tasks run,
    keeping every requested sleep in ``sleeps``.

    ``tick`` additionally advances ``mono`` on every ``monotonic()`` read.
    That is how a test drives a lane loop which measures elapsed time off
    this clock but waits on something else -- the in-flight verify
    abort-poll waits on ``asyncio.wait(timeout=VERIFY_ABANDON_POLL_SECS)``
    and only READS ``monotonic()``, so with the default ``tick`` of 0 its
    no-progress budget can never elapse. Keeping the two counters apart is
    what makes that safe: a test that ticks an hour per duration reading
    does not thereby drag every ``now()`` stamp an hour into the future as a
    side effect of however many durations the lane happened to read.

    ``newest_content_mtime`` reports ``content_mtime`` -- ``None`` or a
    frozen value being a merge worktree nothing is writing to -- and
    advances it by ``content_tick`` per probe, so a ``content_tick`` above
    zero is a verify that keeps writing. Every probed root is kept in
    ``content_probes``, the way ``sleeps`` keeps every requested sleep.
    """

    def __init__(
        self,
        *,
        time: float = 1_000_000.0,
        tick: float = 0.0,
        content_mtime: float | None = None,
        content_tick: float = 0.0,
    ) -> None:
        self.time = time
        self.mono = time
        self.tick = tick
        self.content_mtime = content_mtime
        self.content_tick = content_tick
        self.sleeps: list[float] = []
        self.content_probes: list[Path] = []

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

    async def sleep(self, secs: float) -> None:
        self.sleeps.append(secs)
        self.time += secs
        self.mono += secs
        await asyncio.sleep(0)


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
    verifier: Any = None,
    clock: Any = None,
    escalation_queue: Any = None,
    **kwargs: Any,
) -> MergeLane:
    """A ``MergeLane`` on the fakes, for a test that drives its own worker.

    The verifier and clock default to fresh fakes rather than to the
    production adapters, so a test that forgets to pass one still gets a
    lane that neither shells out nor waits on the wall clock. *queue*
    defaults to a fresh ``asyncio.Queue`` on the running loop. Anything else
    ``MergeLane`` accepts passes through in *kwargs*.
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
