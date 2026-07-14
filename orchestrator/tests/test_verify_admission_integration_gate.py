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
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify import run_verification

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
