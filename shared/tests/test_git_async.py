"""Tests for shared.git_async — the repo's single async subprocess spawn primitive.

The contract pinned here is what BOTH consumers need:

  - orchestrator ``git_ops._run`` (the donor: LC_ALL=C child env, optional
    stdin feeding, task-2608 cancellation kill+reap), which keeps its own
    ``WorktreeMissing`` pre-flight and re-classification on top; and
  - the fused-memory live-workflow probes (task 3778), which additionally
    need a per-call TIMEOUT that degrades fail-open rather than raising, and
    a CONCURRENCY BOUND — because ``create_subprocess_exec`` still forks
    INLINE on the event-loop thread, so unbounded fan-out occupies the loop
    even though every call is "async".

``asyncio_mode = "auto"`` is set in shared/pyproject.toml, so async test
bodies need no ``@pytest.mark.asyncio`` marker.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from shared.git_async import MAX_CONCURRENT_SPAWNS, GitResult, run_git


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """A minimal real git repository, for probes that must shell out for real."""
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    return tmp_path


# ---------------------------------------------------------------------------
# (a) result shape
# ---------------------------------------------------------------------------


async def test_returns_returncode_stdout_stderr_with_stdout_stripped(tmp_repo: Path) -> None:
    result = await run_git(['git', 'rev-parse', '--show-toplevel'], cwd=tmp_repo)

    assert isinstance(result, GitResult)
    assert result.returncode == 0
    assert result.ok is True
    assert result.timed_out is False
    # stdout is stripped: no trailing newline for the caller to remember to eat.
    assert result.stdout == result.stdout.strip()
    assert Path(result.stdout).resolve() == tmp_repo.resolve()
    assert result.stderr == ''


# ---------------------------------------------------------------------------
# (b) fail-open on non-zero rc
# ---------------------------------------------------------------------------


async def test_nonzero_returncode_is_returned_never_raised(tmp_repo: Path) -> None:
    """Every fused-memory probe reads rc!=0 as "signal absent", so it must RETURN."""
    result = await run_git(['git', 'rev-parse', '--verify', 'refs/heads/nope'], cwd=tmp_repo)

    assert result.returncode != 0
    assert result.ok is False
    assert result.timed_out is False


# ---------------------------------------------------------------------------
# (c) timeout: fail-open result, child killed AND reaped, loudly logged
# ---------------------------------------------------------------------------


async def test_timeout_returns_timed_out_result_and_kills_and_reaps_child(
    caplog: pytest.LogCaptureFixture,
) -> None:
    spawned: list[asyncio.subprocess.Process] = []
    real_spawn = asyncio.create_subprocess_exec

    async def _recording_spawn(*args: object, **kwargs: object):
        proc = await real_spawn(*args, **kwargs)  # type: ignore[arg-type]
        spawned.append(proc)
        return proc

    with (
        mock.patch.object(asyncio, 'create_subprocess_exec', _recording_spawn),
        caplog.at_level(logging.WARNING, logger='shared.git_async'),
    ):
        result = await run_git(['sleep', '5'], timeout=0.3)

    assert result.timed_out is True
    assert result.returncode != 0, 'a timed-out call must not look like success'
    assert result.ok is False

    assert len(spawned) == 1
    proc = spawned[0]
    assert proc.returncode is not None, 'child was killed but never reaped (zombie)'
    with pytest.raises(ProcessLookupError):
        os.kill(proc.pid, 0)

    assert caplog.records, 'a degraded (timed-out) call must WARN, not fail silently'


async def test_timeout_not_reached_returns_normally() -> None:
    result = await run_git(['sh', '-c', 'exit 0'], timeout=30)

    assert result.timed_out is False
    assert result.ok is True


# ---------------------------------------------------------------------------
# (d) C locale is forced in the child
# ---------------------------------------------------------------------------


async def test_child_env_forces_c_locale() -> None:
    """LC_ALL=C is load-bearing: git_ops substring-matches English git output."""
    result = await run_git(['sh', '-c', 'echo "$LC_ALL:$LANG"'])

    assert result.stdout == 'C:C'


# ---------------------------------------------------------------------------
# (e) optional stdin feeding
# ---------------------------------------------------------------------------


async def test_input_text_round_trips_through_stdin() -> None:
    result = await run_git(['cat'], input_text='x')

    assert result.returncode == 0
    assert result.stdout == 'x'


# ---------------------------------------------------------------------------
# (f) FileNotFoundError propagates (git_ops re-classifies it into WorktreeMissing)
# ---------------------------------------------------------------------------


async def test_missing_binary_raises_file_not_found(tmp_repo: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await run_git(['definitely-not-a-real-binary-3778'], cwd=tmp_repo)


# ---------------------------------------------------------------------------
# (g) concurrency is bounded
# ---------------------------------------------------------------------------


class _FakeProc:
    """A spawned child that never really exists — only its timing matters."""

    pid = -1

    def __init__(self, on_enter, on_exit) -> None:
        self.returncode = 0
        self._on_enter = on_enter
        self._on_exit = on_exit

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:  # noqa: A002
        self._on_enter()
        try:
            await asyncio.sleep(0.02)
        finally:
            self._on_exit()
        return b'', b''

    def kill(self) -> None:
        return None

    async def wait(self) -> int:
        return 0


async def test_concurrency_is_bounded_by_the_module_cap() -> None:
    """The fork happens ON the loop thread, so simultaneous spawns must be capped."""
    state = {'in_flight': 0, 'peak': 0}

    def _enter() -> None:
        state['in_flight'] += 1
        state['peak'] = max(state['peak'], state['in_flight'])

    def _exit() -> None:
        state['in_flight'] -= 1

    async def _fake_spawn(*args: object, **kwargs: object) -> _FakeProc:
        return _FakeProc(_enter, _exit)

    with mock.patch.object(asyncio, 'create_subprocess_exec', _fake_spawn):
        await asyncio.gather(*(run_git(['git', 'status']) for _ in range(20)))

    assert MAX_CONCURRENT_SPAWNS < 20, 'the cap must actually bind at this fan-out'
    assert state['peak'] <= MAX_CONCURRENT_SPAWNS
    assert state['peak'] > 1, 'the helper must still run calls concurrently, not serialise them'


# ---------------------------------------------------------------------------
# (h) the semaphore is resolved PER RUNNING LOOP
# ---------------------------------------------------------------------------


def test_semaphore_is_resolved_per_running_loop() -> None:
    """A module-global Semaphore would bind to the first loop and then raise.

    Since 3.10 asyncio primitives bind lazily on first use and thereafter
    raise ``RuntimeError: ... is bound to a different event loop``. pytest
    makes a fresh loop per async test, so a singleton would pass exactly once.
    """

    async def _once() -> GitResult:
        return await run_git(['sh', '-c', 'exit 0'])

    first = asyncio.run(_once())
    second = asyncio.run(_once())

    assert first.returncode == 0
    assert second.returncode == 0


# ---------------------------------------------------------------------------
# (i) the bound is opt-OUT-able, and opting out shares no queue with it
# ---------------------------------------------------------------------------


async def _settle_until(predicate, timeout: float = 2.0) -> None:
    """Yield to the loop until *predicate* holds, or *timeout* elapses.

    Bounded so that a REGRESSION (a call that queues when it should not)
    reports as the assertion that follows, rather than parking the loop
    until pytest-timeout kills the whole test.
    """
    async def _spin() -> None:
        while not predicate():
            await asyncio.sleep(0)

    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(_spin(), timeout=timeout)


class _BlockingProc:
    """A spawned child that stays alive until *released* is set."""

    pid = -1

    def __init__(self, released: asyncio.Event, entered: list[str], tag: str) -> None:
        self.returncode = 0
        self._released = released
        self._entered = entered
        self._tag = tag

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:  # noqa: A002
        self._entered.append(self._tag)
        await self._released.wait()
        return b'', b''

    def kill(self) -> None:
        return None

    async def wait(self) -> int:
        return 0


async def test_bounded_false_spawns_without_taking_a_slot() -> None:
    """``bounded=False`` must not queue behind a saturated bound.

    Causal, not wall-clock: the bounded blockers are held open by an Event
    that is never set until the assertion has been made, so the unbounded
    call can only have reached the spawn by NOT having queued.  If it took a
    slot this would hang rather than fail slowly.
    """
    released = asyncio.Event()
    entered: list[str] = []

    async def _fake_spawn(*args: object, **kwargs: object) -> _BlockingProc:
        argv = args[0] if args else ''
        return _BlockingProc(released, entered, 'free' if argv == 'free' else 'blocker')

    with mock.patch.object(asyncio, 'create_subprocess_exec', _fake_spawn):
        blockers = [
            asyncio.create_task(run_git(['git', 'status']))
            for _ in range(MAX_CONCURRENT_SPAWNS)
        ]
        free = asyncio.create_task(run_git(['free'], bounded=False))
        # One more BOUNDED call, to pin that the bound still binds for it.
        queued = asyncio.create_task(run_git(['git', 'status']))
        pending = [free, queued, *blockers]
        try:
            await _settle_until(
                lambda: entered.count('blocker') >= MAX_CONCURRENT_SPAWNS,
            )
            await asyncio.wait_for(asyncio.sleep(0.05), timeout=1)

            assert 'free' in entered, (
                'the bounded=False call never spawned: it queued on the semaphore'
            )
            assert entered.count('blocker') == MAX_CONCURRENT_SPAWNS, (
                'the bound stopped binding for ordinary bounded callers'
            )
            assert not queued.done()
        finally:
            # Unblock every fake child unconditionally, so an assertion
            # failure fails FAST instead of leaving the loop parked on tasks
            # that can never complete.
            released.set()
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True), timeout=5,
            )


async def test_bounded_calls_do_not_queue_behind_unbounded_ones() -> None:
    """The converse: an unbounded long-lived child must not consume the bound.

    Otherwise opting out would merely move the starvation — an operator
    script held open forever would still eat a probe's slot.
    """
    released = asyncio.Event()
    entered: list[str] = []

    async def _fake_spawn(*args: object, **kwargs: object) -> _BlockingProc:
        return _BlockingProc(released, entered, 'any')

    with mock.patch.object(asyncio, 'create_subprocess_exec', _fake_spawn):
        unbounded = [
            asyncio.create_task(run_git(['slow-script'], bounded=False))
            for _ in range(MAX_CONCURRENT_SPAWNS * 2)
        ]
        probes = [
            asyncio.create_task(run_git(['git', 'status']))
            for _ in range(MAX_CONCURRENT_SPAWNS)
        ]
        try:
            await _settle_until(lambda: len(entered) >= MAX_CONCURRENT_SPAWNS * 2)
            await asyncio.wait_for(asyncio.sleep(0.05), timeout=1)

            assert len(entered) == MAX_CONCURRENT_SPAWNS * 3, (
                'bounded probes were starved by unbounded children holding slots'
            )
        finally:
            released.set()
            await asyncio.wait_for(
                asyncio.gather(*unbounded, *probes, return_exceptions=True),
                timeout=5,
            )
