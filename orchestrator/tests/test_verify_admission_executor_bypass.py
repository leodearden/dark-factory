"""Task 5424: ``orchestrator.verify._admission_slot`` must not queue a role
that ``shared.verify_admission.acquire_task_slot`` never gates behind its
shared executor.

Sibling to ``test_verify_admission_wiring.py`` (per-seam acquire / nice-prefix
wiring) and ``test_verify_admission_integration_gate.py`` (the PRD's six
boundary scenarios): the property guarded here is the CM's own executor hop,
not whether T1 grants a slot — so it gets a module named for that seam rather
than a slot in the 10k-line ``test_verify.py`` grab-bag.

Exercises the REAL admission machinery (a real flock'd slots dir), hence
``@pytest.mark.real_verify_admission`` as in both modules above. Helpers stay
MODULE-LOCAL (never conftest.py) — a conftest.py edit trips verify.py's
has_conftest and forces the merge-time verify to fall back to running the full
owning-package suite.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import time

import pytest

from orchestrator import verify
from orchestrator.config import OrchestratorConfig

# Post-fix the merge acquire enters in ~0.000s with the pool saturated, so this
# is three orders of magnitude of headroom, not a tuned threshold.
_ENTER_BUDGET_SECS = 1.0
_SATURATION_DEADLINE_SECS = 5.0
_PROBE_SETTLE_SECS = 0.05


async def _await_executor_saturation() -> None:
    """Block until ``_admission_executor`` has no free worker left.

    Polled rather than inferred from a fixed sleep: the worker threads have to
    be scheduled and reach T1's flock poll loop, and under this package's ``-n
    auto --dist loadgroup`` that start-up can slip past any fixed budget on a
    loaded host — redding the caller with the 'not saturated' message instead
    of a real regression.

    A probe job still queued after ``_PROBE_SETTLE_SECS`` proves the pool is
    full; a probe that ran is a free worker observed, so poll again. The settle
    margin is job-pickup slack, not a race: a saturated pool leaves the probe
    queued until the caller releases its slot, so there is no upper bound to be
    tight against. The last probe drains harmlessly at executor shutdown.
    """
    deadline = time.monotonic() + _SATURATION_DEADLINE_SECS
    while True:
        probe = verify._admission_executor().submit(lambda: None)
        await asyncio.sleep(_PROBE_SETTLE_SECS)
        if not probe.done():
            return
        if time.monotonic() >= deadline:
            pytest.fail(
                f'admission executor was not saturated within '
                f'{_SATURATION_DEADLINE_SECS}s — this guard would pass vacuously; '
                f'check _ADMISSION_EXECUTOR_MAX_WORKERS patching and that both '
                f'task-role acquisitions are blocked on the held slot'
            )


class TestAdmissionSlotBypassesExecutorForUngatedRoles:
    """Task 5424 regression guard.

    ``_admission_slot`` used to route EVERY role's ``cm.__enter__()`` through
    ``_admission_executor``, so once that fixed pool was pinned by queued
    task-role acquisitions a merge verify's admission check queued FIFO behind
    them despite never needing a slot at all — 90+ minutes of observed
    head-of-line merge starvation.

    Pre-fix the merge acquire does not merely run slow, it never returns: its
    ``__enter__`` waits on a pool whose workers only free when ``holder_fd``
    closes, and that close lives in the same ``finally`` the blocked ``async
    with`` prevents reaching. A bare ``assert elapsed < budget`` after the
    acquire would therefore be unreachable in exactly the scenario it exists to
    catch, so the acquire is bounded by ``asyncio.timeout`` — which turns that
    deadlock into THIS test's failure rather than a suite-level pytest-timeout
    that, under ``timeout_method = "thread"`` plus ``--max-worker-restart=0``,
    reds a shifting victim (orchestrator/pyproject.toml).

    The polled probe before the clock starts asserts the saturation
    precondition directly, so a setup that stops saturating fails loudly here
    instead of passing vacuously forever.
    """

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    async def test_merge_role_enters_immediately_when_executor_saturated(self, tmp_path, monkeypatch):
        slots_dir = tmp_path / 'slots'
        slots_dir.mkdir()
        config = OrchestratorConfig(
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
        )

        # A tiny pool so two blocked task-role acquisitions fully saturate it
        # (a fresh pool, decoupled from any singleton other tests created).
        monkeypatch.setattr(verify, '_ADMISSION_EXECUTOR_MAX_WORKERS', 2)
        monkeypatch.setattr(verify, '_admission_executor_singleton', None)

        # Hold the sole slot externally so every task-role acquisition below
        # blocks forever in T1's flock poll loop until this fd is closed.
        holder_fd = os.open(str(slots_dir / 'slot-1'), os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        async def _saturate():
            async with verify._admission_slot('task', config):
                pass  # pragma: no cover - never reached while the slot is held

        saturating_tasks = [asyncio.create_task(_saturate()) for _ in range(2)]
        try:
            await _await_executor_saturation()

            try:
                async with asyncio.timeout(_ENTER_BUDGET_SECS), verify._admission_slot('merge', config):
                    pass
            except TimeoutError:
                pytest.fail(
                    f'merge-role _admission_slot did not enter within '
                    f'{_ENTER_BUDGET_SECS}s with the admission executor fully '
                    f'saturated by task-role acquisitions; a role '
                    f'acquire_task_slot never gates must never queue behind the '
                    f'shared executor'
                )
        finally:
            os.close(holder_fd)
            await asyncio.wait_for(asyncio.gather(*saturating_tasks), timeout=5.0)
            executor = verify._admission_executor_singleton
            if executor is not None:
                executor.shutdown(wait=True)
