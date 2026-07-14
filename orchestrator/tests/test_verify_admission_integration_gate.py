"""T4: verify-admission INTEGRATION GATE (task 2392; PRD
``plans/verify-oversubscription-control-prd.md`` task T4; capability
manifest ``plans/verify-oversubscription-control-prd.capability-manifest.md``
§T4).

Executable pytest suite proving the six admission boundary scenarios from
the PRD's §Boundary-test sketch compose end-to-end through the REAL
``orchestrator.verify`` seam — T1 (``shared.verify_admission``'s flock N-slot
semaphore + role nice tiers), T2 (``orchestrator.verify._admission_slot`` /
``_resolve_nice_prefix`` wiring), and T3 (``role`` threaded through
``run_verification`` / ``run_full_verification`` / ``run_main_tip_sweep``) —
rather than re-testing any of them in isolation.  See
``test_verify_admission_wiring.py`` for the per-seam unit/wiring coverage
this suite builds ON TOP OF.

SIX SCENARIOS (one class each below):

1. Global cap — the N=1 flock semaphore serializes concurrent task-role
   pytest legs; the rest block on the slot.
2. Merge never blocks — merge bypasses the slot entirely (no acquire, no
   wait), even while the sole task slot is held.
3. Sweep yields + interleaves — a background-role sweep's per-subproject
   acquire/release never monopolizes the slot for its full duration; a
   contending task verify can acquire BETWEEN two sweep subprojects.
4. Untimed wait / no requeue — slot-wait time is excluded from
   ``verify_command_timeout_secs`` by construction (the clock starts inside
   ``_admission_slot``, after acquisition).
5. Self-heal — a SIGKILLed slot holder frees its flock immediately (kernel
   behaviour — no canary/daemon involved).
6. Fail-open — admission disabled, or an unmkdirable ``slots_dir``, never
   blocks a verify.

Every test is marked ``@pytest.mark.real_verify_admission`` to opt out of
the autouse ``_neutralize_verify_admission`` conftest fixture
(conftest.py:407), which otherwise force-patches
``orchestrator.verify._verify_admission_active`` to ``False`` for every
other test in the suite (task 2390 pre-1) — this gate must exercise the REAL
seam.

Helpers are kept MODULE-LOCAL (never conftest.py) — a conftest.py edit trips
``verify.py``'s ``has_conftest`` heuristic and forces merge-time scoped
verify to fall back to running the full owning-package suite instead of a
scoped subset (mirrors ``test_verify_admission_wiring.py``'s stated
rationale).

TEST-ONLY integration gate: every scenario drives already-shipped T1/T2/T3
seams (patch ``orchestrator.verify._run_cmd``, instrument
``orchestrator.verify.acquire_task_slot``, inject
``config._module_configs``, real subprocess slot holders) — no production
change to ``verify.py`` is expected or made by this suite.  Assertions are
STRUCTURAL/instrumented facts only (max concurrently-held slot, exact
wrapped-argv strings, acquire/release counts, captured ``timeout`` equality,
``VerifyResult.passed``) — never wall-clock thresholds or flock-race
outcomes, per the PRD's "direction, not frozen thresholds" (G6).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify import run_full_verification, run_verification

# ---------------------------------------------------------------------------
# Shared sentinels + labelling (adapted from test_verify_admission_wiring.py)
# ---------------------------------------------------------------------------

# module_config commands are chosen to be uniquely identifiable by substring,
# so a spy `_run_cmd` can label which leg is running without needing `label`
# (which is a `_run_or_skip_timed`-local closure variable, never passed down
# to `_run_cmd`).
_TEST_CMD = 'pytest tests/'
_LINT_CMD = 'ruff'
_TYPE_CMD = 'pyright'


def _leg_for_cmd(cmd: str) -> str:
    """Label which leg *cmd* belongs to by substring, not exact match — an
    active admission gate nice-wraps the test leg (``<nice argv> /bin/bash -c
    <shlex.quote(cmd)>``), so its captured cmd still CONTAINS ``_TEST_CMD``
    but is no longer equal to it. lint/type are never wrapped either way.
    """
    if _TEST_CMD in cmd:
        return 'test'
    if _LINT_CMD in cmd:
        return 'lint'
    if _TYPE_CMD in cmd:
        return 'type'
    return cmd


def _module_config(**overrides: Any) -> ModuleConfig:
    kwargs: dict[str, Any] = dict(
        prefix='pkg',
        test_command=_TEST_CMD,
        lint_command=_LINT_CMD,
        type_check_command=_TYPE_CMD,
        # Sequential so the three legs run strictly test -> lint -> type,
        # making ordering/labelling assertions deterministic (no gather
        # interleaving between legs themselves).
        concurrent_verify=False,
    )
    kwargs.update(overrides)
    return ModuleConfig(**kwargs)


# ---------------------------------------------------------------------------
# Configurable spy for orchestrator.verify._run_cmd
# ---------------------------------------------------------------------------


class _RunCmdSpy:
    """Configurable spy standing in for ``orchestrator.verify._run_cmd``.

    Records every captured call (cmd/cwd/timeout/leg, in call order) and
    tracks live/max concurrency for the leg named by *count_leg* (default
    ``'test'`` — the only leg the admission gate ever wraps) by holding each
    matching call open for *hold_secs* before returning, generalizing
    ``test_real_acquire_serializes_test_leg_across_concurrent_verifies``'s
    ``asyncio.sleep(0.05)`` pattern to an arbitrary number of concurrent
    callers.

    A call can additionally be gated on a test-controlled ``asyncio.Event``
    via :meth:`gate`, keyed by ``(leg, occurrence)`` — the 0-based count of
    prior calls to that same leg — so a test can hold a specific call open
    (e.g. "the second background subproject's test leg") until it explicitly
    releases it, without depending on wall-clock timing.
    """

    def __init__(self, *, count_leg: str | None = 'test', hold_secs: float = 0.05):
        self.calls: list[dict[str, Any]] = []
        self.current = 0
        self.max_seen = 0
        self._count_leg = count_leg
        self._hold_secs = hold_secs
        self._leg_seen: dict[str, int] = {}
        self._gates: dict[tuple[str, int], asyncio.Event] = {}

    def gate(self, leg: str, occurrence: int) -> asyncio.Event:
        """Return (creating if needed) the Event gating *leg*'s *occurrence*-th call."""
        event = self._gates.get((leg, occurrence))
        if event is None:
            event = asyncio.Event()
            self._gates[(leg, occurrence)] = event
        return event

    async def __call__(
        self,
        cmd: str,
        cwd: Path,
        timeout: float,
        env: dict[str, str] | None = None,
        log_path: Path | None = None,
        **kwargs: Any,
    ) -> tuple[int, str, bool]:
        leg = _leg_for_cmd(cmd)
        occurrence = self._leg_seen.get(leg, 0)
        self._leg_seen[leg] = occurrence + 1
        self.calls.append({'cmd': cmd, 'cwd': cwd, 'timeout': timeout, 'leg': leg})

        gate = self._gates.get((leg, occurrence))
        if gate is not None:
            await gate.wait()

        if leg == self._count_leg:
            self.current += 1
            self.max_seen = max(self.max_seen, self.current)
            try:
                await asyncio.sleep(self._hold_secs)
            finally:
                self.current -= 1
        return 0, '', False


# ---------------------------------------------------------------------------
# Deterministic instrumented stand-in for shared.verify_admission.acquire_task_slot
# ---------------------------------------------------------------------------


class _DeterministicAcquire:
    """Stand-in for ``orchestrator.verify.acquire_task_slot`` (T1's real flock
    semaphore) that replaces the real poll race with a test-controlled,
    one-at-a-time grant order — used only where the real race's winner is
    nondeterministic and the test must PROVE a specific interleave (PRD
    scenario 3), rather than merely observe *a* valid one.

    ``__call__`` matches T1's ``acquire_task_slot(role, *, slots_dir, n,
    wait=True)`` signature and, like it, no-ops (yields ``False``
    immediately, no bookkeeping) for any role other than ``'task'``/
    ``'background'``. For those two roles, each call is assigned the next
    0-based *occurrence* number for its role (mirroring ``_RunCmdSpy``'s
    per-leg occurrence tracking) and returns a **sync** context manager —
    matching T1's contract, since ``orchestrator.verify._admission_slot``
    drives ``cm.__enter__``/``cm.__exit__`` from a worker thread via
    ``run_in_executor`` — whose ``__enter__`` blocks the calling thread on
    a ``threading.Event`` (not ``asyncio.Event``: this blocks in a real OS
    thread, not on the event loop) obtained via :meth:`gate`. The test
    drives the interleave by calling ``.set()`` on each occurrence's gate
    in whatever order it wants grants to happen, one at a time, waiting for
    each grant's acquire+release to appear in :attr:`log` before opening
    the next — ``max_seen`` is real, honest accounting (incremented on
    acquire, decremented on release, under :attr:`_lock`, since worker
    threads call concurrently), not asserted-by-construction, so a bug that
    opens two gates too early is still caught.
    """

    def __init__(self) -> None:
        self.log: list[tuple[str, str]] = []
        self.current = 0
        self.max_seen = 0
        self._lock = threading.Lock()
        self._seen: dict[str, int] = {}
        self._gates: dict[tuple[str, int], threading.Event] = {}

    def gate(self, role: str, occurrence: int) -> threading.Event:
        """Return (creating if needed) the Event gating *role*'s *occurrence*-th grant."""
        event = self._gates.get((role, occurrence))
        if event is None:
            event = threading.Event()
            self._gates[(role, occurrence)] = event
        return event

    def __call__(self, role: str, *, slots_dir: Path, n: int, wait: bool = True):
        if role not in {'task', 'background'}:
            @contextlib.contextmanager
            def _noop():
                yield False
            return _noop()

        occurrence = self._seen.get(role, 0)
        self._seen[role] = occurrence + 1
        gate = self.gate(role, occurrence)
        return self._held(role, gate)

    @contextlib.contextmanager
    def _held(self, role: str, gate: threading.Event):
        gate.wait()  # blocks the executor worker thread until the test releases it
        with self._lock:
            self.current += 1
            self.max_seen = max(self.max_seen, self.current)
            self.log.append((role, 'acquire'))
        try:
            yield True
        finally:
            with self._lock:
                self.log.append((role, 'release'))
                self.current -= 1


async def _wait_for_log_len(acq: _DeterministicAcquire, want_len: int, *, timeout: float = 5.0) -> None:
    """Poll (test-side, bounded) until *acq*'s log reaches *want_len* entries."""
    deadline = time.monotonic() + timeout
    while len(acq.log) < want_len and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert len(acq.log) >= want_len, (
        f'expected at least {want_len} acquire-log entries within {timeout}s; '
        f'got {acq.log!r}'
    )


# ---------------------------------------------------------------------------
# Real-subprocess slot holder (adapted from shared/tests/test_verify_admission.py)
# ---------------------------------------------------------------------------

# Inline stdlib child: flocks slots_dir/slot-<n> (matching T1's slot-file
# naming), signals readiness via a marker file, then sleeps so the parent can
# observe it holding the slot until deliberately killed/terminated.
_SLOT_HOLDER_CHILD_SRC = (
    'import fcntl, os, sys, time\n'
    "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o644)\n"
    'fcntl.flock(fd, fcntl.LOCK_EX)\n'
    "open(sys.argv[2], 'w').write('ready')\n"
    'time.sleep(60)\n'
)


def _spawn_slot_holder(slot_path: Path, marker_path: Path) -> subprocess.Popen:
    """Spawn a real subprocess that flocks *slot_path* and signals readiness
    via *marker_path*. Caller owns the process lifecycle — kill (SIGKILL for
    self-heal, or terminate for teardown) and ``.wait()`` it, ideally in a
    ``finally`` block so a failed assertion never leaks the child.
    """
    return subprocess.Popen(
        [sys.executable, '-c', _SLOT_HOLDER_CHILD_SRC, str(slot_path), str(marker_path)],
    )


def _wait_for_marker(marker_path: Path, timeout: float = 5.0) -> bool:
    """Poll for *marker_path* to appear; return False on timeout.

    Test-side synchronization only (bounded, unlike the production module's
    untimed wait) so a child process that fails to start doesn't hang the
    suite forever.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker_path.exists():
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------------------
# Scenario 1 — GLOBAL CAP
# ---------------------------------------------------------------------------


class TestGlobalCap:
    """PRD Boundary-test sketch scenario 1: dispatching M concurrent
    task-role verifies against a single N=1 slot never lets more than one
    'test' leg run at a time — the rest block on the real T1 flock semaphore
    until it frees up, and all eventually complete successfully.
    """

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    async def test_n1_slot_serializes_m_concurrent_task_verifies(self, tmp_path):
        m = 3
        slots_dir = tmp_path / 'slots'
        config = OrchestratorConfig(
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
        )
        worktrees = []
        for i in range(m):
            worktree = tmp_path / f'wt-{i}'
            worktree.mkdir()
            worktrees.append(worktree)

        spy = _RunCmdSpy()
        # REAL acquire_task_slot (not mocked) — only _run_cmd is patched, so
        # this exercises T1's actual flock semaphore + T2's off-loop wiring,
        # generalizing the wiring test's 2-verify serialize test to M>=3.
        # `new=spy` (not `side_effect=spy`): AsyncMock only awaits a
        # side_effect it recognizes via asyncio.iscoroutinefunction, which is
        # False for a callable *instance* like `spy` (only true functions/
        # bound methods qualify) — side_effect=spy would return spy(...)'s
        # coroutine object un-awaited. `new=spy` substitutes the callable
        # directly, so `await _run_cmd(...)` awaits spy.__call__ itself.
        with patch('orchestrator.verify._run_cmd', new=spy):
            results = await asyncio.gather(*(
                run_verification(
                    worktree=worktree,
                    config=config,
                    module_config=_module_config(),
                    role='task',
                    attempt_id=None,
                )
                for worktree in worktrees
            ))

        assert spy.max_seen == 1, (
            f'expected the N=1 semaphore to serialize all {m} task-role test '
            f'legs (max concurrently-running test leg == 1); got '
            f'max_seen={spy.max_seen}'
        )
        assert len(results) == m
        assert all(r.passed for r in results), (
            f'expected all {m} verifies to pass; got '
            f'{[r.passed for r in results]!r}'
        )


# ---------------------------------------------------------------------------
# Scenario 2 — MERGE NEVER BLOCKS
# ---------------------------------------------------------------------------


class TestMergeNeverBlocks:
    """PRD Boundary-test sketch scenario 2: a merge-role verify's test leg
    runs immediately — no acquire, no wait — even while the sole task slot
    is held by a task-role verify (T1's merge no-op: ``acquire_task_slot``
    never attempts acquisition for ``role='merge'``, always yielding
    ``held=False``).
    """

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    async def test_merge_verify_runs_unblocked_while_task_slot_held(self, tmp_path):
        slots_dir = tmp_path / 'slots'
        config = OrchestratorConfig(
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
        )
        holder_worktree = tmp_path / 'wt-holder'
        holder_worktree.mkdir()
        merge_worktree = tmp_path / 'wt-merge'
        merge_worktree.mkdir()

        spy = _RunCmdSpy()
        # Gate the holder's (leg='test', occurrence 0) call open indefinitely
        # so it keeps the sole task slot held for the test's duration. The
        # call is only recorded — proving acquisition already happened,
        # since _run_cmd is invoked strictly after _admission_slot acquires
        # — once it reaches this gate, so it blocks *after* acquiring, never
        # before.
        holder_gate = spy.gate('test', 0)

        with patch('orchestrator.verify._run_cmd', new=spy):
            holder_task = asyncio.create_task(
                run_verification(
                    worktree=holder_worktree,
                    config=config,
                    module_config=_module_config(),
                    role='task',
                    attempt_id=None,
                )
            )
            try:
                deadline = time.monotonic() + 5.0
                while len(spy.calls) < 1 and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
                assert len(spy.calls) == 1, (
                    'expected the task-role holder to acquire the sole slot '
                    'and reach its gated test leg within 5s'
                )

                merge_result = await run_verification(
                    worktree=merge_worktree,
                    config=config,
                    module_config=_module_config(),
                    role='merge',
                    attempt_id=None,
                )

                # run_verification(role='merge') runs test -> lint -> type
                # sequentially (concurrent_verify=False) before returning, so
                # spy.calls also holds merge's lint/type entries by now;
                # isolate the 'test' leg specifically. Its call is the 2nd
                # 'test'-leg entry (occurrence 1) — proven to have completed
                # *while the holder is still gated/blocked*, by construction:
                # holder_gate is not released until after this assertion
                # block, in the `finally` below.
                test_calls = [c for c in spy.calls if c['leg'] == 'test']
                assert len(test_calls) == 2
                merge_call = test_calls[1]
                expected_cmd = (
                    shlex.join(['nice', '-n', '5'])
                    + ' /bin/bash -c '
                    + shlex.quote(_TEST_CMD)
                )
                assert merge_call['cmd'] == expected_cmd, (
                    f'expected the merge test leg wrapped with the nice -n 5 '
                    f'tier; got {merge_call["cmd"]!r}'
                )
                assert merge_result.passed
            finally:
                holder_gate.set()
                holder_result = await holder_task
            assert holder_result.passed


# ---------------------------------------------------------------------------
# Scenario 3 — SWEEP YIELDS + INTERLEAVES
# ---------------------------------------------------------------------------


class TestSweepYieldsAndInterleaves:
    """PRD Boundary-test sketch scenario 3: a background-role sweep
    (``run_full_verification`` fanning out over ``config._module_configs``)
    never monopolizes the sole slot for its full duration — each
    subproject's admission is its own acquire/release pair — and a
    contending task-role verify can be granted the slot BETWEEN two
    sweep subprojects.

    Uses :class:`_DeterministicAcquire` in place of the real T1 flock
    semaphore: under the real poll race, which of several concurrent
    waiters wins any given release is nondeterministic, so proving a
    SPECIFIC interleave (background -> task -> background) against it
    would be flaky (design_decisions[3]). The deterministic stand-in still
    does real, honest max-held accounting — only the grant ORDER is
    test-controlled.
    """

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    async def test_task_verify_interleaves_between_two_background_subprojects(self, tmp_path):
        project_root = tmp_path / 'proj'
        project_root.mkdir()
        task_worktree = tmp_path / 'wt-task'
        task_worktree.mkdir()
        slots_dir = tmp_path / 'slots'

        config = OrchestratorConfig(
            project_root=project_root,
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
        )
        # Reuse path (task 2390/2391): run_full_verification(project_root, ...)
        # only skips a fresh filesystem walk when project_root.resolve() ==
        # config.project_root, which the field_validator above already
        # resolved — so passing the same `project_root` object to both
        # satisfies the reuse guard.
        config._module_configs = {
            'a': _module_config(prefix='a'),
            'b': _module_config(prefix='b'),
        }

        acq = _DeterministicAcquire()
        spy = _RunCmdSpy()
        # Pre-register all three expected grants' gates up front — creating a
        # gate does not release it, so this is safe regardless of which
        # subproject's acquire call is assigned occurrence 0 vs 1.
        bg_gate_0 = acq.gate('background', 0)
        bg_gate_1 = acq.gate('background', 1)
        task_gate_0 = acq.gate('task', 0)

        with (
            patch('orchestrator.verify.acquire_task_slot', new=acq),
            patch('orchestrator.verify._run_cmd', new=spy),
        ):
            sweep_task = asyncio.create_task(
                run_full_verification(project_root, config, role='background')
            )

            # 1st background subproject: grant, then wait for its full
            # acquire+release pair before letting anything else proceed.
            bg_gate_0.set()
            await _wait_for_log_len(acq, 2)

            # Contending task-role verify, started only now so its acquire
            # attempt lands after the 1st background release.
            task_task = asyncio.create_task(
                run_verification(
                    worktree=task_worktree,
                    config=config,
                    module_config=_module_config(),
                    role='task',
                    attempt_id=None,
                )
            )
            task_gate_0.set()
            await _wait_for_log_len(acq, 4)

            # 2nd background subproject, granted last.
            bg_gate_1.set()
            await _wait_for_log_len(acq, 6)

            sweep_result = await sweep_task
            task_result = await task_task

        # (c) interleave: task's grant lands BETWEEN the two background grants.
        acquire_order = [role for role, event in acq.log if event == 'acquire']
        assert acquire_order == ['background', 'task', 'background'], (
            f'expected background -> task -> background grant order; got {acquire_order!r}'
        )

        # (b) per-subproject release — one acquire/release pair per
        # subproject, never monopolized for the sweep's full duration — and
        # the shared n=1 budget (task + background) never over-granted.
        assert acq.log == [
            ('background', 'acquire'), ('background', 'release'),
            ('task', 'acquire'), ('task', 'release'),
            ('background', 'acquire'), ('background', 'release'),
        ]
        assert acq.max_seen == 1, (
            f'expected the shared n=1 slot to never be held by more than one '
            f'role at a time; got max_seen={acq.max_seen}'
        )

        # (a) every sweep subproject's test leg is wrapped with the
        # background nice tier. Both subprojects share worktree=project_root
        # (run_full_verification's fan-out contract), while the contending
        # task verify uses a distinct worktree — cwd distinguishes them.
        background_test_calls = [
            c for c in spy.calls
            if c['leg'] == 'test' and c['cwd'] == project_root
        ]
        assert len(background_test_calls) == 2
        expected_background_cmd = (
            shlex.join(['nice', '-n', '19', 'ionice', '-c3'])
            + ' /bin/bash -c '
            + shlex.quote(_TEST_CMD)
        )
        assert all(c['cmd'] == expected_background_cmd for c in background_test_calls), (
            f'expected both sweep subproject test legs wrapped with the '
            f'background nice tier; got {[c["cmd"] for c in background_test_calls]!r}'
        )

        assert sweep_result.passed
        assert task_result.passed


# ---------------------------------------------------------------------------
# Scenario 4 — UNTIMED WAIT / NO REQUEUE
# ---------------------------------------------------------------------------


class TestUntimedWaitNoRequeue:
    """PRD Boundary-test sketch scenario 4: slot-wait time is excluded from
    ``verify_command_timeout_secs`` by construction — ``_run_or_skip_timed``
    resolves the timeout and starts its clock (``t0``/``started_at``) INSIDE
    ``_admission_slot``, after acquisition. A task-role verify that must
    wait a real, nonzero interval for a contended slot therefore receives
    the exact same timeout budget as one with a free slot, and completes
    successfully with no timeout/requeue.
    """

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    async def test_contended_wait_gets_same_timeout_budget_as_free_slot(self, tmp_path):
        slots_dir = tmp_path / 'slots'
        config = OrchestratorConfig(
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
        )

        # --- Baseline: free slot, no contention. ---
        baseline_worktree = tmp_path / 'wt-baseline'
        baseline_worktree.mkdir()
        baseline_spy = _RunCmdSpy()
        with patch('orchestrator.verify._run_cmd', new=baseline_spy):
            baseline_result = await run_verification(
                worktree=baseline_worktree,
                config=config,
                module_config=_module_config(),
                role='task',
                attempt_id=None,
            )
        baseline_test_calls = [c for c in baseline_spy.calls if c['leg'] == 'test']
        assert len(baseline_test_calls) == 1
        baseline_timeout = baseline_test_calls[0]['timeout']
        assert baseline_result.passed

        # --- Contended: a gated holder saturates the sole slot; a second
        # task verify must wait a real, nonzero interval for it. ---
        holder_worktree = tmp_path / 'wt-holder'
        holder_worktree.mkdir()
        waiter_worktree = tmp_path / 'wt-waiter'
        waiter_worktree.mkdir()

        spy = _RunCmdSpy()
        holder_gate = spy.gate('test', 0)

        with patch('orchestrator.verify._run_cmd', new=spy):
            holder_task = asyncio.create_task(
                run_verification(
                    worktree=holder_worktree,
                    config=config,
                    module_config=_module_config(),
                    role='task',
                    attempt_id=None,
                )
            )
            try:
                deadline = time.monotonic() + 5.0
                while len(spy.calls) < 1 and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
                assert len(spy.calls) == 1, (
                    'expected the holder to acquire the sole slot and reach '
                    'its gated test leg within 5s'
                )

                waiter_task = asyncio.create_task(
                    run_verification(
                        worktree=waiter_worktree,
                        config=config,
                        module_config=_module_config(),
                        role='task',
                        attempt_id=None,
                    )
                )
                # Test-side pacing only (never asserted on): give the waiter
                # a real, nonzero interval to genuinely contend for — and
                # poll against, per T1's 0.1s poll loop — the still-held
                # slot before releasing the holder.
                await asyncio.sleep(0.3)
            finally:
                holder_gate.set()
                holder_result = await holder_task
            waiter_result = await waiter_task

        assert holder_result.passed

        waiter_test_calls = [
            c for c in spy.calls if c['leg'] == 'test' and c['cwd'] == waiter_worktree
        ]
        assert len(waiter_test_calls) == 1
        waiter_timeout = waiter_test_calls[0]['timeout']
        assert waiter_timeout == baseline_timeout, (
            f'expected the contended waiter to receive the same timeout '
            f'budget as the free-slot baseline (slot-wait excluded from '
            f'verify_command_timeout_secs); got waiter={waiter_timeout!r} '
            f'baseline={baseline_timeout!r}'
        )
        assert waiter_result.passed
        assert not waiter_result.timed_out


# ---------------------------------------------------------------------------
# Scenario 5 — SELF-HEAL
# ---------------------------------------------------------------------------


class TestSelfHeal:
    """PRD Boundary-test sketch scenario 5: a SIGKILLed slot holder's flock
    is freed immediately by the kernel on process death — no canary/daemon
    involved (T1's C-daemonless self-heal). A task-role verify blocked
    waiting (untimed) for the sole slot proceeds as soon as the holder dies.
    """

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    async def test_task_verify_proceeds_after_holder_is_sigkilled(self, tmp_path):
        slots_dir = tmp_path / 'slots'
        # mkdir'd up front: the real subprocess holder must os.open a slot
        # file inside it before run_verification ever starts (T1 never
        # creates slots_dir itself; _admission_slot's own mkdir would race
        # the holder otherwise).
        slots_dir.mkdir()
        config = OrchestratorConfig(
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
        )
        worktree = tmp_path / 'wt-task'
        worktree.mkdir()
        marker = tmp_path / 'ready.marker'
        slot_path = slots_dir / 'slot-1'

        holder: subprocess.Popen | None = _spawn_slot_holder(slot_path, marker)
        try:
            assert _wait_for_marker(marker), 'holder never signaled readiness'

            spy = _RunCmdSpy()
            # REAL acquire_task_slot (not mocked) — only _run_cmd is patched,
            # so the block-then-self-heal is proven against T1's actual
            # flock semaphore, not a stand-in.
            with patch('orchestrator.verify._run_cmd', new=spy):
                task_verify = asyncio.create_task(
                    run_verification(
                        worktree=worktree,
                        config=config,
                        module_config=_module_config(),
                        role='task',
                        attempt_id=None,
                    )
                )

                # Bounded poll proving the verify is genuinely blocked on the
                # held slot (T1's untimed acquire wait) — its test leg never
                # reaches _run_cmd while the real holder lives.
                blocked_deadline = time.monotonic() + 1.0
                while time.monotonic() < blocked_deadline:
                    assert not task_verify.done(), (
                        'task verify completed while the sole slot was '
                        'still held by the real subprocess holder'
                    )
                    assert len(spy.calls) == 0, (
                        f'expected no leg to run while the slot is held; '
                        f'got {spy.calls!r}'
                    )
                    await asyncio.sleep(0.05)

                os.kill(holder.pid, signal.SIGKILL)
                holder.wait(timeout=5)
                holder = None

                result = await asyncio.wait_for(task_verify, timeout=10.0)

            assert result.passed, (
                f'expected the task verify to proceed once the kernel freed '
                f"the SIGKILLed holder's flock; got passed={result.passed!r}"
            )
            assert not result.timed_out
            test_calls = [c for c in spy.calls if c['leg'] == 'test']
            assert len(test_calls) == 1, (
                f'expected exactly one test-leg call after self-heal; got '
                f'{test_calls!r}'
            )
        finally:
            if holder is not None:
                holder.kill()
                holder.wait(timeout=5)


# ---------------------------------------------------------------------------
# Scenario 6 — FAIL-OPEN
# ---------------------------------------------------------------------------


class TestFailOpen:
    """PRD Boundary-test sketch scenario 6: neither admission being disabled
    nor a ``slots_dir`` that cannot be created ever blocks a verify —
    T2's ``_verify_admission_active`` / ``_admission_slot`` fail-open paths
    (C-fail-open, layered on top of T1's own acquire fail-open contract).
    """

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    async def test_admission_disabled_never_acquires_and_cmd_is_unwrapped(self, tmp_path):
        slots_dir = tmp_path / 'slots'
        config = OrchestratorConfig(
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
            verify_admission_enabled=False,
        )
        worktree = tmp_path / 'wt'
        worktree.mkdir()

        spy = _RunCmdSpy()
        with (
            patch('orchestrator.verify._run_cmd', new=spy),
            patch('orchestrator.verify.acquire_task_slot') as mock_acquire,
        ):
            result = await run_verification(
                worktree=worktree,
                config=config,
                module_config=_module_config(),
                role='task',
                attempt_id=None,
            )

        mock_acquire.assert_not_called()
        assert not slots_dir.exists(), (
            'expected slots_dir to never be created when admission is disabled'
        )

        legs = {c['leg'] for c in spy.calls}
        assert legs == {'test', 'lint', 'type'}, (
            f'expected all three legs to run; got legs={legs!r}'
        )
        test_calls = [c for c in spy.calls if c['leg'] == 'test']
        assert len(test_calls) == 1
        assert test_calls[0]['cmd'] == _TEST_CMD, (
            f'expected the byte-identical, unwrapped (ungated) test-leg cmd; '
            f'got {test_calls[0]["cmd"]!r}'
        )
        assert result.passed

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    async def test_unmkdirable_slots_dir_fails_open_and_verify_still_passes(self, tmp_path):
        # slots_dir's PARENT path component is a regular file, not a missing
        # directory — `slots_dir.mkdir(parents=True, exist_ok=True)` inside
        # _admission_slot raises NotADirectoryError (an OSError), exercising
        # the same fail-open `except OSError: cm = None` path a real
        # permission error or read-only filesystem would hit.
        blocker = tmp_path / 'blocker'
        blocker.write_text('not a directory')
        slots_dir = blocker / 'slots'
        config = OrchestratorConfig(
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
        )
        worktree = tmp_path / 'wt'
        worktree.mkdir()

        spy = _RunCmdSpy()
        # REAL acquire_task_slot (not mocked) — only _run_cmd is patched, so
        # the mkdir failure is exercised against the real _admission_slot
        # seam, not a stand-in.
        with patch('orchestrator.verify._run_cmd', new=spy):
            result = await run_verification(
                worktree=worktree,
                config=config,
                module_config=_module_config(),
                role='task',
                attempt_id=None,
            )

        legs = {c['leg'] for c in spy.calls}
        assert legs == {'test', 'lint', 'type'}, (
            f'expected all three legs to still run despite the unmkdirable '
            f'slots_dir; got legs={legs!r}'
        )
        test_calls = [c for c in spy.calls if c['leg'] == 'test']
        assert len(test_calls) == 1
        expected_cmd = (
            shlex.join(['nice', '-n', '15', 'ionice', '-c2', '-n7'])
            + ' /bin/bash -c '
            + shlex.quote(_TEST_CMD)
        )
        assert test_calls[0]['cmd'] == expected_cmd, (
            f'expected the task nice-tier wrap to still apply even though '
            f'acquisition failed open; got {test_calls[0]["cmd"]!r}'
        )
        assert result.passed
