"""Tests for the generalized escalation-revalidation sweep — task 2114.

Background: task 2074 built escalation-revalidation machinery scoped to the
blocked-deterministic subset only (Source A/B inside
``_run_deterministic_recon_sweep``).  Motivating evidence (2026-07-06): six L2
escalations were manually ``close_only``'d that a general sweep would have
auto-closed — five because their SUBJECT task became terminal (done or
cancelled), one because the cited defect landed on main (a main-tip-sweep
integrity gate whose fix commits are now ancestors of main).

This file generalizes the revalidation sweep to auto-close ALL stale open L2
escalations using only POSITIVE, fail-safe evidence, split across two homes:

  Source C — terminal-subject closure (criterion a, general, all
  categories/task_kinds): hosted inside the EXISTING
  ``_run_deterministic_recon_sweep`` pass.  For each pending L2 escalation,
  reads the subject task's status via ``scheduler.get_statuses`` and closes
  when the status is terminal (``done`` or ``cancelled``).

  Main-tip-sweep self-heal (criterion b, the main-tip integrity cluster):
  hosted inside ``_run_main_tip_sweep``'s ``vr.passed`` branch.  Closes any
  pending ``orchestrator-main-sweep`` escalation whose swept SHA is a strict
  ancestor of the just-verified clean tip.

This file covers:
  step-1: test_config_defaults_escalation_revalidation
  step-3: TestRevalidateOpenL2
  step-5: TestReconSweepSourceC
  step-7: TestCloseSupersededMainSweepEscalations
  step-9: TestMainTipSweepSelfHealWiring
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import _init_harness_state_for_test
from escalation.models import Escalation

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import EventType, Harness

# ---------------------------------------------------------------------------
# step-1: Config field presence and default
# ---------------------------------------------------------------------------


def test_config_defaults_escalation_revalidation() -> None:
    """OrchestratorConfig exposes escalation_revalidation_enabled (True) —
    the single operator kill-switch for both new auto-closure paths."""
    config = OrchestratorConfig()
    assert config.escalation_revalidation_enabled is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_harness() -> Harness:
    """Build a minimal bare Harness for escalation-revalidation-sweep tests."""
    h = Harness.__new__(Harness)
    _init_harness_state_for_test(h)
    h.config = OrchestratorConfig()
    h.scheduler = MagicMock()
    h._escalation_queue = MagicMock()
    h._escalation_queue.make_id = MagicMock(return_value='esc-revalidation-1')
    h.event_store = MagicMock()
    return h


def _make_esc(
    *,
    level: int = 2,
    category: str = 'design_concern',
    agent_role: str = 'orchestrator-scheduler',
    esc_id: str = 'esc-tid-1',
    task_id: str = 'tid',
    status: str = 'pending',
) -> Escalation:
    return Escalation(
        id=esc_id,
        task_id=task_id,
        agent_role=agent_role,
        severity='critical',
        category=category,
        summary='test escalation',
        level=level,
        status=status,
    )


# ---------------------------------------------------------------------------
# step-3: Harness._revalidate_open_l2 — terminal-subject closure (criterion a)
# ---------------------------------------------------------------------------


class TestRevalidateOpenL2:
    """step-3: _revalidate_open_l2 closes an open L2 whose subject task is
    terminal (done/cancelled), and leaves everything else untouched."""

    @pytest.mark.asyncio
    async def test_done_subject_closes_l2(self) -> None:
        h = _make_harness()
        esc = _make_esc(level=2, task_id='tid-done')
        statuses = {'tid-done': 'done'}

        result = await h._revalidate_open_l2(esc, statuses)

        assert result is True
        h._escalation_queue.resolve.assert_called_once()  # type: ignore[union-attr, attr-defined]
        args, kwargs = h._escalation_queue.resolve.call_args  # type: ignore[union-attr, attr-defined]
        assert args[0] == esc.id
        resolution = args[1] if len(args) > 1 else kwargs.get('resolution')
        assert resolution and isinstance(resolution, str)
        assert kwargs.get('dismiss') is True
        assert kwargs.get('resolved_by') == 'harness-escalation-revalidation-sweep'
        h.event_store.emit.assert_called_once()  # type: ignore[attr-defined]
        emit_args, emit_kwargs = h.event_store.emit.call_args  # type: ignore[attr-defined]
        assert emit_args[0] == EventType.escalation_resolved
        data = emit_kwargs.get('data') or {}
        assert data.get('escalation_id') == esc.id
        assert data.get('subject_status') == 'done'
        assert data.get('reason') == 'terminal-subject'
        h.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_cancelled_subject_closes_l2(self) -> None:
        h = _make_harness()
        esc = _make_esc(level=2, task_id='tid-cancelled')
        statuses = {'tid-cancelled': 'cancelled'}

        result = await h._revalidate_open_l2(esc, statuses)

        assert result is True
        h._escalation_queue.resolve.assert_called_once()  # type: ignore[union-attr, attr-defined]
        _args, kwargs = h._escalation_queue.resolve.call_args  # type: ignore[union-attr, attr-defined]
        assert kwargs.get('dismiss') is True
        assert kwargs.get('resolved_by') == 'harness-escalation-revalidation-sweep'
        h.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_blocked_subject_leaves_open(self) -> None:
        h = _make_harness()
        esc = _make_esc(level=2, task_id='tid-live')
        statuses = {'tid-live': 'blocked'}

        result = await h._revalidate_open_l2(esc, statuses)

        assert result is False
        h._escalation_queue.resolve.assert_not_called()  # type: ignore[union-attr, attr-defined]

    @pytest.mark.asyncio
    async def test_in_progress_subject_leaves_open(self) -> None:
        h = _make_harness()
        esc = _make_esc(level=2, task_id='tid-live')
        statuses = {'tid-live': 'in-progress'}

        result = await h._revalidate_open_l2(esc, statuses)

        assert result is False
        h._escalation_queue.resolve.assert_not_called()  # type: ignore[union-attr, attr-defined]

    @pytest.mark.asyncio
    async def test_subject_absent_from_statuses_leaves_open(self) -> None:
        h = _make_harness()
        esc = _make_esc(level=2, task_id='tid-unknown')
        statuses: dict[str, str] = {}

        result = await h._revalidate_open_l2(esc, statuses)

        assert result is False
        h._escalation_queue.resolve.assert_not_called()  # type: ignore[union-attr, attr-defined]

    @pytest.mark.asyncio
    async def test_non_l2_escalation_ignored(self) -> None:
        h = _make_harness()
        esc = _make_esc(level=1, task_id='tid-done')
        statuses = {'tid-done': 'done'}

        result = await h._revalidate_open_l2(esc, statuses)

        assert result is False
        h._escalation_queue.resolve.assert_not_called()  # type: ignore[union-attr, attr-defined]

    @pytest.mark.asyncio
    async def test_disabled_by_config_kill_switch(self) -> None:
        h = _make_harness()
        h.config = h.config.model_copy(update={'escalation_revalidation_enabled': False})
        esc = _make_esc(level=2, task_id='tid-done')
        statuses = {'tid-done': 'done'}

        result = await h._revalidate_open_l2(esc, statuses)

        assert result is False
        h._escalation_queue.resolve.assert_not_called()  # type: ignore[union-attr, attr-defined]


# ---------------------------------------------------------------------------
# step-5: _run_deterministic_recon_sweep — Source-C wiring
# ---------------------------------------------------------------------------


class TestReconSweepSourceC:
    """step-5: _run_deterministic_recon_sweep's new Source-C wiring — a batch
    get_statuses read over pending L2s feeds _revalidate_open_l2 per pending
    escalation, and Source B is skipped for whichever escalation Source C
    just closed."""

    @pytest.mark.asyncio
    async def test_closed_l2_skips_source_b(self) -> None:
        h = _make_harness()
        esc = _make_esc(level=2, task_id='tid-done', esc_id='esc-l2-done')
        h.scheduler.get_tasks = AsyncMock(return_value=[])  # type: ignore[method-assign]
        h.scheduler.get_statuses = AsyncMock(  # type: ignore[method-assign]
            return_value=({esc.task_id: 'done'}, None)
        )
        h._escalation_queue.get_pending = MagicMock(return_value=[esc])  # type: ignore[union-attr]
        h._escalation_queue.get_by_task = MagicMock(return_value=[])  # type: ignore[union-attr]
        h._recover_stranded_deterministic_task = AsyncMock()  # type: ignore[method-assign]
        h._revalidate_open_l2 = AsyncMock(return_value=True)  # type: ignore[method-assign]
        h._revalidate_open_deterministic_escalation = AsyncMock()  # type: ignore[method-assign]

        await h._run_deterministic_recon_sweep()

        h.scheduler.get_statuses.assert_awaited_once_with([esc.task_id])  # type: ignore[attr-defined]
        h._revalidate_open_l2.assert_awaited_once_with(esc, {esc.task_id: 'done'})
        h._revalidate_open_deterministic_escalation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_open_l2_falls_through_to_source_b(self) -> None:
        h = _make_harness()
        esc = _make_esc(
            level=2, category='infra_issue', agent_role='orchestrator-deterministic',
            task_id='tid-live', esc_id='esc-l2-live',
        )
        task = {
            'id': 'tid-live', 'status': 'in-progress',
            'metadata': {'task_kind': 'deterministic'},
        }
        h.scheduler.get_tasks = AsyncMock(return_value=[])  # type: ignore[method-assign]
        h.scheduler.get_task = AsyncMock(return_value=task)  # type: ignore[method-assign]
        h.scheduler.get_statuses = AsyncMock(  # type: ignore[method-assign]
            return_value=({esc.task_id: 'in-progress'}, None)
        )
        h._escalation_queue.get_pending = MagicMock(return_value=[esc])  # type: ignore[union-attr]
        h._escalation_queue.get_by_task = MagicMock(return_value=[])  # type: ignore[union-attr]
        h._recover_stranded_deterministic_task = AsyncMock()  # type: ignore[method-assign]
        h._revalidate_open_l2 = AsyncMock(return_value=False)  # type: ignore[method-assign]
        h._revalidate_open_deterministic_escalation = AsyncMock()  # type: ignore[method-assign]

        await h._run_deterministic_recon_sweep()

        h._revalidate_open_l2.assert_awaited_once_with(esc, {esc.task_id: 'in-progress'})
        h._revalidate_open_deterministic_escalation.assert_awaited_once_with(
            esc, task, task['metadata']
        )

    @pytest.mark.asyncio
    async def test_get_statuses_error_degrades_to_empty_and_source_b_still_runs(self) -> None:
        """DEFENSIVE: get_statuses raising must not abort the pass — statuses
        degrades to {} and Source B still runs.  This is the property that
        keeps the existing 2074 suite green (its bare MagicMock scheduler has
        no get_statuses configured, so the real call raises TypeError)."""
        h = _make_harness()
        esc = _make_esc(
            level=2, category='infra_issue', agent_role='orchestrator-deterministic',
            task_id='tid-live', esc_id='esc-l2-live',
        )
        task = {
            'id': 'tid-live', 'status': 'in-progress',
            'metadata': {'task_kind': 'deterministic'},
        }
        h.scheduler.get_tasks = AsyncMock(return_value=[])  # type: ignore[method-assign]
        h.scheduler.get_task = AsyncMock(return_value=task)  # type: ignore[method-assign]
        h.scheduler.get_statuses = AsyncMock(side_effect=RuntimeError('boom'))  # type: ignore[method-assign]
        h._escalation_queue.get_pending = MagicMock(return_value=[esc])  # type: ignore[union-attr]
        h._escalation_queue.get_by_task = MagicMock(return_value=[])  # type: ignore[union-attr]
        h._recover_stranded_deterministic_task = AsyncMock()  # type: ignore[method-assign]
        h._revalidate_open_l2 = AsyncMock(return_value=False)  # type: ignore[method-assign]
        h._revalidate_open_deterministic_escalation = AsyncMock()  # type: ignore[method-assign]

        await h._run_deterministic_recon_sweep()  # must not raise

        h._revalidate_open_l2.assert_awaited_once_with(esc, {})
        h._revalidate_open_deterministic_escalation.assert_awaited_once_with(
            esc, task, task['metadata']
        )
