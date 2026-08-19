"""Cancellation-mid-acquire slot-release coverage for
``orchestrator.verify._admission_slot`` (task 4071).

TEST-ONLY: this module pins existing, already-correct behaviour in
``_admission_slot``. No production behaviour change is expected or made
here — see ``_admission_slot``'s docstring in ``verify.py`` for the
cross-reference back to this module.

Helpers are kept MODULE-LOCAL (never conftest.py) — mirrors
``test_verify_admission_wiring.py`` and
``test_verify_admission_integration_gate.py``'s stated rationale: a
conftest.py edit trips ``verify.py``'s ``has_conftest`` heuristic and forces
merge-time scoped verify to fall back to running the whole owning package
instead of a scoped subset.

Every test here is marked ``@pytest.mark.real_verify_admission`` to opt out
of the autouse ``_neutralize_verify_admission`` conftest fixture, which
otherwise force-patches ``orchestrator.verify._verify_admission_active`` to
False for every other test in the suite. Strictly speaking neither test in
this module calls ``_verify_admission_active`` (both drive
``_admission_slot`` directly, not ``run_verification``), so the fixture
would be a no-op here either way — the marker is applied regardless, to
document that this module exercises the real admission seam and to
future-proof it against that fixture's patch surface growing.

THE FAILURE MODE BEING PINNED: ``_admission_slot`` hands the blocking,
untimed flock poll loop (``acquire_task_slot(...).__enter__``) to a
dedicated executor thread and awaits it under ``asyncio.shield``. If the
awaiting coroutine is cancelled while the thread is still polling, the
thread cannot be interrupted mid-wait — it keeps running and may go on to
acquire a slot AFTER the coroutine stopped waiting. The cancellation path's
``enter_future.add_done_callback(_release_if_acquired)`` is the ONLY thing
that then releases that slot. With ``verify_admission_task_slots`` defaulting
to 1, a single leaked slot wedges every subsequent task/background-role
verify until an orchestrator restart. Test 1 below pins the release itself
(the done-callback fires and frees the slot the thread later wins); test 2
pins the adjacent guard that must NOT call ``__exit__`` on a context manager
whose ``__enter__`` never completed.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from shared.verify_admission import acquire_task_slot

from orchestrator.config import OrchestratorConfig
from orchestrator.verify import _admission_slot


async def _await_flag(event: threading.Event, *, timeout: float = 5.0, msg: str) -> None:
    """Bounded async poll for *event*.

    NEVER ``event.wait()`` — that would block the event-loop thread, and
    ``_release_if_acquired`` is scheduled ON the loop (a done-callback on an
    asyncio future), so blocking the loop would prevent the very signal
    under test from ever running. Polling asynchronously keeps the loop
    live; the deadline turns a genuine leak into a named assertion failure
    instead of a hang.
    """
    deadline = time.monotonic() + timeout
    while not event.is_set():
        assert time.monotonic() < deadline, msg
        await asyncio.sleep(0.01)


class _SpyAcquire:
    """Recording wrapper around the REAL ``shared.verify_admission.acquire_task_slot``.

    Callable with T1's own signature so it can replace
    ``orchestrator.verify.acquire_task_slot`` via ``patch()``. Unlike a fake
    context manager, this wraps the real flock semaphore end-to-end — the
    failure mode this module pins is an OS-level flock leak, so only a real
    flock can prove the closing "a subsequent acquire succeeds" assertion.
    The wrapper adds observability (``started``/``entered``/``exited``
    events, ``held_values``, ``exit_calls``) without changing semantics.

    ``_admission_slot`` never opens this via ``with cm:`` — it drives
    ``__enter__``/``__exit__`` directly, from a worker thread, via
    ``run_in_executor`` — so a single instance doubling as both the
    callable and the returned context manager (``__call__`` returns
    ``self``) is enough; this module never needs two concurrent
    acquisitions through one spy.
    """

    def __init__(self) -> None:
        self.started = threading.Event()
        self.entered = threading.Event()
        self.exited = threading.Event()
        self.held_values: list[bool] = []
        self.exit_calls = 0
        self._cm: Any = None

    def __call__(self, role: str, *, slots_dir: Path, n: int, wait: bool = True) -> _SpyAcquire:
        self._cm = acquire_task_slot(role, slots_dir=slots_dir, n=n, wait=wait)
        return self

    def __enter__(self) -> bool:
        self.started.set()
        held = self._cm.__enter__()
        self.held_values.append(held)
        self.entered.set()
        return held

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool | None:
        self.exit_calls += 1
        try:
            return self._cm.__exit__(exc_type, exc, tb)
        finally:
            self.exited.set()


@pytest.mark.real_verify_admission
@pytest.mark.asyncio
async def test_cancel_mid_acquire_releases_the_slot_the_thread_later_wins(tmp_path):
    slots_dir = tmp_path / 'slots'
    slots_dir.mkdir(parents=True)
    config = OrchestratorConfig(
        verify_admission_slots_dir=str(slots_dir),
        verify_admission_task_slots=1,
    )
    spy = _SpyAcquire()
    body_ran = False

    async def _run() -> None:
        nonlocal body_ran
        async with _admission_slot('task', config):
            body_ran = True

    with contextlib.ExitStack() as stack:
        held = stack.enter_context(
            acquire_task_slot('task', slots_dir=slots_dir, n=1, wait=False),
        )
        assert held is True, 'test setup: must own the only slot before racing the CM'

        with patch('orchestrator.verify.acquire_task_slot', spy):
            task = asyncio.create_task(_run())
            await _await_flag(
                spy.started, msg='worker thread never started polling for a slot',
            )
            assert not spy.entered.is_set(), (
                'worker thread must still be polling, not past __enter__'
            )

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert body_ran is False, 'CM body must never run on the cancelled-mid-acquire path'
            assert not spy.entered.is_set()

            stack.close()  # release the external holder; worker thread wins the slot next poll

            await _await_flag(
                spy.entered,
                msg='worker thread never acquired after the external holder released',
            )
            assert spy.held_values == [True], (
                'worker thread must genuinely win a slot after we stopped waiting'
            )

            await _await_flag(
                spy.exited,
                msg='LEAK: the cancellation done-callback never released the slot',
            )
            assert spy.exit_calls == 1, 'slot must be released exactly once (no double release)'

    with acquire_task_slot('task', slots_dir=slots_dir, n=1, wait=False) as held:
        assert held is True, 'LEAK: a subsequent acquire is blocked by a never-released slot'
