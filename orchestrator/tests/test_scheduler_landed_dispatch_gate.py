"""Tests for the scheduler's landed-outbox consult-before-dispatch gate
(task 2156, W1 δ — PRD merge-queue-reliability §8.2 SD-1, boundary B5).

Covers:
  step-3  (RED)  ``Scheduler.acquire_next()`` consults the injected
                 ``_landed_outbox_gate`` hook per candidate BEFORE either
                 dispatch loop commits: hook unset (default no-op,
                 byte-identical to today), hook True (gate — the task is
                 NOT dispatched, not added to ``_dispatched``), hook False
                 (task dispatches normally), and pin-loop parity (a gated
                 PINNED candidate is also skipped, not just the scored
                 loop's candidate list).

Mirrors test_scheduler.py's TestAcquireNextNoDuplicates / TestPinDispatch
conventions: ``scheduler.get_tasks = AsyncMock(return_value=[task])`` with
plain pending-task dicts carrying ``dependencies``/``metadata.files``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import Scheduler


def _pending_task(
    task_id: str,
    *,
    priority: str = 'medium',
    files: list[str] | None = None,
) -> dict:
    """Minimal pending-task dict carrying every field acquire_next reads."""
    return {
        'id': task_id,
        'title': f'Task {task_id}',
        'status': 'pending',
        'priority': priority,
        'dependencies': [],
        'metadata': {'files': files or [f'mod{task_id}']},
    }


# ---------------------------------------------------------------------------
# Hook unset (None) — byte-identical to pre-gate behavior.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLandedOutboxGateDefaultNoop:
    """`_landed_outbox_gate` unset (None) => dispatch proceeds exactly as before."""

    async def test_unset_gate_dispatches_normally(self) -> None:
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        scheduler.finish_startup()
        task = _pending_task('Z')
        scheduler.get_tasks = AsyncMock(return_value=[task])

        result = await scheduler.acquire_next()

        assert result is not None
        assert result.task_id == 'Z'
        assert 'Z' in scheduler._dispatched


# ---------------------------------------------------------------------------
# B5 — hook True gates; hook False lets the task through.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLandedOutboxGateB5Gates:
    """B5: the gate is consulted per candidate before either dispatch loop commits."""

    async def test_gate_true_withholds_dispatch(self) -> None:
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        scheduler.finish_startup()
        task = _pending_task('Z')
        scheduler.get_tasks = AsyncMock(return_value=[task])
        scheduler._landed_outbox_gate = AsyncMock(return_value=True)

        result = await scheduler.acquire_next()

        assert result is None, 'a gated task must not be dispatched this tick'
        assert 'Z' not in scheduler._dispatched
        scheduler._landed_outbox_gate.assert_awaited_once_with('Z')

    async def test_gate_false_dispatches_normally(self) -> None:
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        scheduler.finish_startup()
        task = _pending_task('Z')
        scheduler.get_tasks = AsyncMock(return_value=[task])
        scheduler._landed_outbox_gate = AsyncMock(return_value=False)

        result = await scheduler.acquire_next()

        assert result is not None
        assert result.task_id == 'Z'
        assert 'Z' in scheduler._dispatched
        scheduler._landed_outbox_gate.assert_awaited_once_with('Z')


# ---------------------------------------------------------------------------
# Pin-loop parity — a gated PINNED candidate must also be skipped.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLandedOutboxGatePinnedParity:
    """A pinned candidate the gate gates must be skipped by the pin loop too.

    The pin-dispatch loop re-derives its own candidate list from the pin
    queue (independent of the scored loop's ``candidates``), so the gate
    must be consulted there as well — this is exactly the parity the task's
    design decision calls for (single ``gated_ids`` source honored by BOTH
    dispatch paths).
    """

    async def test_gated_pinned_candidate_is_not_dispatched(self, tmp_path) -> None:
        from orchestrator.overrides import OverrideStore

        config = OrchestratorConfig(max_per_module=1)
        store = OverrideStore(tmp_path / 'o.db')
        store.set_override('/proj', 'Z', pinned=True)

        scheduler = Scheduler(config, override_store=store)
        scheduler.finish_startup()
        scheduler._project_root = '/proj'
        # Gate only 'Z' (the pinned task) — 'Y' must remain freely dispatchable
        # so the run doesn't short-circuit on an empty candidate list before
        # ever reaching the pin loop.
        scheduler._landed_outbox_gate = AsyncMock(side_effect=lambda task_id: task_id == 'Z')

        task_z = _pending_task('Z', files=['z/src'])
        task_y = _pending_task('Y', files=['y/src'])
        scheduler.get_tasks = AsyncMock(return_value=[task_z, task_y])

        result = await scheduler.acquire_next()

        assert result is not None
        assert result.task_id == 'Y', 'Y (not gated) should dispatch; Z (gated+pinned) must not'
        assert 'Z' not in scheduler._dispatched
        assert 'Y' in scheduler._dispatched
        awaited_ids = {call.args[0] for call in scheduler._landed_outbox_gate.await_args_list}
        assert awaited_ids == {'Z', 'Y'}, 'the gate must be consulted for every candidate'


# ---------------------------------------------------------------------------
# PROPERTY 1 — a raising gate hook must never propagate out of acquire_next()
# and must never abort the rest of the tick's consult.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLandedOutboxGateFailsOpen:
    """A gate hook that raises must fail OPEN, per-candidate, never crash the tick.

    ``Harness.run()``'s dispatch loop wraps ``acquire_next()`` in try/finally
    with NO except clause, so any exception that escapes here crashes the
    orchestrator process. The injected hook performs unguarded git I/O
    (``get_main_sha``/``is_ancestor``) and scheduler writes
    (``get_status``/``mark_done``) via ``reconcile_landed_task`` — any of
    which can transiently fail — so a raising hook must be treated as "not
    gated" for that candidate only, not propagated and not allowed to skip
    consulting the remaining candidates.
    """

    async def test_gate_raises_fails_open_and_dispatches(self, caplog) -> None:
        caplog.set_level('WARNING')
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        scheduler.finish_startup()
        task = _pending_task('Z')
        scheduler.get_tasks = AsyncMock(return_value=[task])
        scheduler._landed_outbox_gate = AsyncMock(side_effect=RuntimeError('git hiccup'))

        result = await scheduler.acquire_next()

        assert result is not None, 'a raising consult must fail open, not withhold dispatch'
        assert result.task_id == 'Z'
        assert 'Z' in scheduler._dispatched
        assert any(record.levelname == 'WARNING' for record in caplog.records), (
            'a raising gate hook should be logged, not silently swallowed'
        )

    async def test_gate_raises_for_one_candidate_does_not_abort_others(self) -> None:
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        scheduler.finish_startup()
        task_z = _pending_task('Z', files=['z/src'])
        task_y = _pending_task('Y', files=['y/src'])
        scheduler.get_tasks = AsyncMock(return_value=[task_z, task_y])

        async def _gate_side_effect(task_id: str) -> bool:
            if task_id == 'Z':
                raise RuntimeError('git hiccup')
            return task_id == 'Y'

        scheduler._landed_outbox_gate = AsyncMock(side_effect=_gate_side_effect)

        result = await scheduler.acquire_next()

        assert result is not None
        assert result.task_id == 'Z', 'Z fails open (its consult raised) so it is the only candidate left'
        assert 'Z' in scheduler._dispatched
        assert 'Y' not in scheduler._dispatched, 'Y was genuinely gated (its consult returned True)'
        awaited_ids = {call.args[0] for call in scheduler._landed_outbox_gate.await_args_list}
        assert awaited_ids == {'Z', 'Y'}, "Z's exception must not abort Y's consult"
