"""Tests for the periodic deterministic-strand reconciliation sweep — task 2074.

Background: task 2066 hardened DeterministicRunner against NEW strands but
explicitly scoped OUT recovery of tasks already stranded by a PAST occurrence
(e.g. task 2059): a deterministic gate/deploy task left BLOCKED with
``before_done_ran_at`` stamped but ``before_done_verified_at`` /
``gate_escalated_at`` / ``done_provenance`` all absent, and an EMPTY pending
escalation queue.  Root cause: the cross-unit deploy severed the
orchestrator's own fused-memory connection, so the runner could neither write
back success nor file an escalation.

This file adds a new, cross-cutting background sweep (mirroring
``_main_tip_sweep_loop`` / ``_stranded_reconcile_loop``) that:

  Source A — detects ABSENT-escalation strands (blocked deterministic task +
  empty pending queue) and RE-FILES an L1 escalation whose category depends
  on live systemd unit health (never flips status itself).

  Source B — re-validates OPEN deterministic-deploy ``infra_issue``
  escalations against live unit health, auto-resolving when the stated
  failure is now contradicted by a healthy unit.

This file covers:
  step-1:  test_config_defaults_deterministic_recon_sweep
  step-3:  TestDeterministicDeployHealthVerdict / TestRevalidateDeployHealth
  step-5:  TestIsStrandedDeterministicShape
  step-7:  TestRecoverStrandedDeterministicTask
  step-9:  TestRevalidateOpenDeterministicEscalation
  step-11: TestRunDeterministicReconSweep
  step-13: TestDeterministicReconSweepLifecycle
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import _init_harness_state_for_test
from escalation.models import Escalation

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import (
    Harness,
    _deterministic_deploy_health_verdict,
    _is_stranded_deterministic_shape,
)

# ---------------------------------------------------------------------------
# step-1: Config field presence and defaults
# ---------------------------------------------------------------------------


def test_config_defaults_deterministic_recon_sweep() -> None:
    """OrchestratorConfig exposes deterministic_recon_sweep_enabled (True) and
    deterministic_recon_sweep_interval_secs (900.0) with the correct defaults."""
    config = OrchestratorConfig()
    assert config.deterministic_recon_sweep_enabled is True
    assert config.deterministic_recon_sweep_interval_secs == 900.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_recon_harness() -> Harness:
    """Build a minimal bare Harness for deterministic-recon-sweep tests."""
    h = Harness.__new__(Harness)
    _init_harness_state_for_test(h)
    h.config = OrchestratorConfig()
    h.scheduler = MagicMock()
    h._escalation_queue = MagicMock()
    h._escalation_queue.make_id = MagicMock(return_value='esc-recon-1')
    h._recon_unit_inspector = None
    h._deterministic_recon_sweep_task = None
    return h


# ---------------------------------------------------------------------------
# step-3: pure health-verdict fn + Harness._revalidate_deterministic_deploy_health
# ---------------------------------------------------------------------------


class TestDeterministicDeployHealthVerdict:
    """step-3: harness._deterministic_deploy_health_verdict pure function."""

    def test_healthy_when_pid_positive_and_active(self) -> None:
        result = _deterministic_deploy_health_verdict({'MainPID': 1234, 'ActiveState': 'active'})
        assert result == 'healthy'

    def test_unconfirmed_when_pid_zero(self) -> None:
        result = _deterministic_deploy_health_verdict({'MainPID': 0, 'ActiveState': 'active'})
        assert result == 'unconfirmed'

    def test_unconfirmed_when_not_active(self) -> None:
        result = _deterministic_deploy_health_verdict({'MainPID': 1234, 'ActiveState': 'failed'})
        assert result == 'unconfirmed'

    def test_unconfirmed_when_inspect_result_is_none(self) -> None:
        assert _deterministic_deploy_health_verdict(None) == 'unconfirmed'

    def test_unconfirmed_when_inspect_result_is_empty(self) -> None:
        assert _deterministic_deploy_health_verdict({}) == 'unconfirmed'


class TestRevalidateDeployHealth:
    """step-3: Harness._revalidate_deterministic_deploy_health."""

    @pytest.mark.asyncio
    async def test_revalidate_calls_injected_inspector_and_returns_verdict(self) -> None:
        h = _make_recon_harness()
        h._recon_unit_inspector = AsyncMock(
            return_value={'MainPID': 4321, 'ActiveState': 'active'}
        )
        metadata = {'before_done': {'target_unit': 'fused-memory.service'}}

        verdict = await h._revalidate_deterministic_deploy_health(metadata)

        assert verdict == 'healthy'
        h._recon_unit_inspector.assert_awaited_once_with('fused-memory.service')

    @pytest.mark.asyncio
    async def test_revalidate_unconfirmed_when_no_before_done(self) -> None:
        h = _make_recon_harness()
        h._recon_unit_inspector = AsyncMock()

        verdict = await h._revalidate_deterministic_deploy_health({})

        assert verdict == 'unconfirmed'
        h._recon_unit_inspector.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_revalidate_unconfirmed_when_no_target_unit(self) -> None:
        h = _make_recon_harness()
        h._recon_unit_inspector = AsyncMock()

        verdict = await h._revalidate_deterministic_deploy_health({'before_done': {}})

        assert verdict == 'unconfirmed'
        h._recon_unit_inspector.assert_not_awaited()


# ---------------------------------------------------------------------------
# step-5: pure fn _is_stranded_deterministic_shape
# ---------------------------------------------------------------------------


def _strand_metadata(**overrides) -> dict:
    """Canonical stranded-deterministic-shape metadata (task 2059 EVIDENCE)."""
    base = {
        'task_kind': 'deterministic',
        'before_done': {'target_unit': 'fused-memory.service'},
        'before_done_ran_at': '2026-07-01T00:00:00+00:00',
        'before_done_verified_at': None,
        'gate_escalated_at': None,
        'done_provenance': None,
    }
    base.update(overrides)
    return base


class TestIsStrandedDeterministicShape:
    """step-5: harness._is_stranded_deterministic_shape pure classifier."""

    def test_true_for_canonical_strand_shape(self) -> None:
        assert _is_stranded_deterministic_shape(_strand_metadata()) is True

    def test_false_when_task_kind_not_deterministic(self) -> None:
        assert _is_stranded_deterministic_shape(_strand_metadata(task_kind='normal')) is False

    def test_false_when_before_done_ran_at_missing(self) -> None:
        assert _is_stranded_deterministic_shape(_strand_metadata(before_done_ran_at=None)) is False

    def test_false_when_before_done_verified_at_present(self) -> None:
        assert _is_stranded_deterministic_shape(
            _strand_metadata(before_done_verified_at='2026-07-02T00:00:00+00:00')
        ) is False

    def test_false_when_gate_escalated_at_present(self) -> None:
        assert _is_stranded_deterministic_shape(
            _strand_metadata(gate_escalated_at='2026-07-02T00:00:00+00:00')
        ) is False

    def test_false_when_done_provenance_present(self) -> None:
        assert _is_stranded_deterministic_shape(
            _strand_metadata(done_provenance={'kind': 'deterministic-deploy'})
        ) is False

    def test_false_when_before_done_is_none(self) -> None:
        assert _is_stranded_deterministic_shape(_strand_metadata(before_done=None)) is False

    def test_false_when_before_done_lacks_target_unit(self) -> None:
        assert _is_stranded_deterministic_shape(_strand_metadata(before_done={})) is False

    def test_false_for_empty_metadata(self) -> None:
        assert _is_stranded_deterministic_shape({}) is False

    def test_false_for_none_metadata(self) -> None:
        assert _is_stranded_deterministic_shape(None) is False


# ---------------------------------------------------------------------------
# step-7: Harness._recover_stranded_deterministic_task (Source A)
# ---------------------------------------------------------------------------


class TestRecoverStrandedDeterministicTask:
    """step-7: RE-FILE-NEVER-FLIP strand recovery for absent-escalation strands."""

    @pytest.mark.asyncio
    async def test_healthy_verdict_files_stranded_blocked_resume(self) -> None:
        h = _make_recon_harness()
        h.event_store = MagicMock()
        h._escalation_queue.get_by_task = MagicMock(return_value=[])
        h._recon_unit_inspector = AsyncMock(
            return_value={'MainPID': 4321, 'ActiveState': 'active'}
        )
        tid = 'task-2059'
        task = {'id': tid, 'description': 'Deploy fused-memory restart'}
        metadata = _strand_metadata()

        await h._recover_stranded_deterministic_task(tid, task, metadata)

        h._escalation_queue.submit.assert_called_once()
        esc = h._escalation_queue.submit.call_args[0][0]
        assert esc.category == 'stranded_blocked'
        assert esc.level == 1
        assert esc.severity == 'blocking'
        assert esc.suggested_action == 'resume'
        assert esc.agent_role == 'harness-deterministic-recon-sweep'
        assert esc.task_id == tid
        h.scheduler.set_task_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_unconfirmed_verdict_files_infra_issue_manual(self) -> None:
        h = _make_recon_harness()
        h.event_store = MagicMock()
        h._escalation_queue.get_by_task = MagicMock(return_value=[])
        h._recon_unit_inspector = AsyncMock(
            return_value={'MainPID': 0, 'ActiveState': 'failed'}
        )
        tid = 'task-2059'
        task = {'id': tid, 'description': 'Deploy fused-memory restart'}
        metadata = _strand_metadata()

        await h._recover_stranded_deterministic_task(tid, task, metadata)

        h._escalation_queue.submit.assert_called_once()
        esc = h._escalation_queue.submit.call_args[0][0]
        assert esc.category == 'infra_issue'
        assert esc.level == 1
        assert esc.severity == 'blocking'
        assert esc.suggested_action == 'manual_intervention'
        assert esc.agent_role == 'harness-deterministic-recon-sweep'
        h.scheduler.set_task_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_dedup_skips_when_pending_escalation_exists(self) -> None:
        h = _make_recon_harness()
        h.event_store = MagicMock()
        h._escalation_queue.get_by_task = MagicMock(return_value=[MagicMock()])
        h._recon_unit_inspector = AsyncMock(
            return_value={'MainPID': 4321, 'ActiveState': 'active'}
        )
        tid = 'task-2059'
        task = {'id': tid, 'description': 'Deploy fused-memory restart'}
        metadata = _strand_metadata()

        await h._recover_stranded_deterministic_task(tid, task, metadata)

        h._escalation_queue.submit.assert_not_called()
        h.scheduler.set_task_status.assert_not_called()


# ---------------------------------------------------------------------------
# step-9: Harness._revalidate_open_deterministic_escalation (Source B)
# ---------------------------------------------------------------------------


def _make_esc(*, category: str, agent_role: str, esc_id: str = 'esc-tid-1', task_id: str = 'tid') -> Escalation:
    return Escalation(
        id=esc_id,
        task_id=task_id,
        agent_role=agent_role,
        severity='critical',
        category=category,
        summary='test escalation',
        level=2,
    )


class TestRevalidateOpenDeterministicEscalation:
    """step-9: re-validates OPEN deterministic-deploy escalations against live state."""

    @pytest.mark.asyncio
    async def test_resolves_when_infra_issue_orchestrator_deterministic_and_healthy(self) -> None:
        h = _make_recon_harness()
        h._recon_unit_inspector = AsyncMock(
            return_value={'MainPID': 111, 'ActiveState': 'active'}
        )
        esc = _make_esc(category='infra_issue', agent_role='orchestrator-deterministic')
        metadata = _strand_metadata()

        await h._revalidate_open_deterministic_escalation(esc, {'id': 'tid'}, metadata)

        h._escalation_queue.resolve.assert_called_once()
        args, kwargs = h._escalation_queue.resolve.call_args
        assert args[0] == esc.id
        resolution = args[1] if len(args) > 1 else kwargs.get('resolution')
        assert resolution and isinstance(resolution, str)
        assert kwargs.get('resolved_by') == 'harness-deterministic-recon-sweep'

    @pytest.mark.asyncio
    async def test_does_not_resolve_when_unconfirmed(self) -> None:
        h = _make_recon_harness()
        h._recon_unit_inspector = AsyncMock(
            return_value={'MainPID': 0, 'ActiveState': 'failed'}
        )
        esc = _make_esc(category='infra_issue', agent_role='orchestrator-deterministic')
        metadata = _strand_metadata()

        await h._revalidate_open_deterministic_escalation(esc, {'id': 'tid'}, metadata)

        h._escalation_queue.resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_never_touches_milestone_gate(self) -> None:
        h = _make_recon_harness()
        h._recon_unit_inspector = AsyncMock(
            return_value={'MainPID': 111, 'ActiveState': 'active'}
        )
        esc = _make_esc(category='milestone_gate', agent_role='orchestrator-deterministic')
        metadata = _strand_metadata()

        await h._revalidate_open_deterministic_escalation(esc, {'id': 'tid'}, metadata)

        h._escalation_queue.resolve.assert_not_called()
        h._recon_unit_inspector.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ignores_non_sentinel_agent_role(self) -> None:
        h = _make_recon_harness()
        h._recon_unit_inspector = AsyncMock(
            return_value={'MainPID': 111, 'ActiveState': 'active'}
        )
        esc = _make_esc(category='infra_issue', agent_role='orchestrator-scheduler')
        metadata = _strand_metadata()

        await h._revalidate_open_deterministic_escalation(esc, {'id': 'tid'}, metadata)

        h._escalation_queue.resolve.assert_not_called()
        h._recon_unit_inspector.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_resolves_own_sweep_filed_escalation_when_healthy(self) -> None:
        h = _make_recon_harness()
        h._recon_unit_inspector = AsyncMock(
            return_value={'MainPID': 111, 'ActiveState': 'active'}
        )
        esc = _make_esc(
            category='infra_issue', agent_role='harness-deterministic-recon-sweep',
        )
        metadata = _strand_metadata()

        await h._revalidate_open_deterministic_escalation(esc, {'id': 'tid'}, metadata)

        h._escalation_queue.resolve.assert_called_once()


# ---------------------------------------------------------------------------
# step-11: Harness._run_deterministic_recon_sweep — single testable pass
# ---------------------------------------------------------------------------


class TestRunDeterministicReconSweep:
    """step-11: single-pass orchestration of Source A + Source B."""

    @pytest.mark.asyncio
    async def test_source_a_recovers_absent_escalation_strand(self) -> None:
        h = _make_recon_harness()
        metadata = _strand_metadata()
        task = {'id': 'tid-a', 'status': 'blocked', 'metadata': metadata}
        h.scheduler.get_tasks = AsyncMock(return_value=[task])
        h._escalation_queue.get_by_task = MagicMock(return_value=[])
        h._escalation_queue.get_pending = MagicMock(return_value=[])
        h._recover_stranded_deterministic_task = AsyncMock()  # type: ignore[method-assign]
        h._revalidate_open_deterministic_escalation = AsyncMock()  # type: ignore[method-assign]

        await h._run_deterministic_recon_sweep()

        h._recover_stranded_deterministic_task.assert_awaited_once_with(
            'tid-a', task, metadata
        )
        h._revalidate_open_deterministic_escalation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_source_b_revalidates_open_escalation_not_source_a(self) -> None:
        h = _make_recon_harness()
        metadata = _strand_metadata()
        task = {'id': 'tid-b', 'status': 'blocked', 'metadata': metadata}
        esc = _make_esc(
            category='infra_issue', agent_role='orchestrator-deterministic', task_id='tid-b',
        )
        h.scheduler.get_tasks = AsyncMock(return_value=[task])
        h._escalation_queue.get_by_task = MagicMock(return_value=[esc])
        h._escalation_queue.get_pending = MagicMock(return_value=[esc])
        h._recover_stranded_deterministic_task = AsyncMock()  # type: ignore[method-assign]
        h._revalidate_open_deterministic_escalation = AsyncMock()  # type: ignore[method-assign]

        await h._run_deterministic_recon_sweep()

        h._recover_stranded_deterministic_task.assert_not_awaited()
        h._revalidate_open_deterministic_escalation.assert_awaited_once_with(
            esc, task, metadata
        )

    @pytest.mark.asyncio
    async def test_ignores_non_deterministic_and_done_tasks(self) -> None:
        h = _make_recon_harness()
        normal_task = {
            'id': 'tid-normal', 'status': 'blocked', 'metadata': {'task_kind': 'normal'},
        }
        done_task = {'id': 'tid-done', 'status': 'done', 'metadata': _strand_metadata()}
        h.scheduler.get_tasks = AsyncMock(return_value=[normal_task, done_task])
        h._escalation_queue.get_by_task = MagicMock(return_value=[])
        h._escalation_queue.get_pending = MagicMock(return_value=[])
        h._recover_stranded_deterministic_task = AsyncMock()  # type: ignore[method-assign]
        h._revalidate_open_deterministic_escalation = AsyncMock()  # type: ignore[method-assign]

        await h._run_deterministic_recon_sweep()

        h._recover_stranded_deterministic_task.assert_not_awaited()
        h._revalidate_open_deterministic_escalation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_noop_when_no_escalation_queue(self) -> None:
        h = _make_recon_harness()
        h._escalation_queue = None
        h.scheduler.get_tasks = AsyncMock(side_effect=AssertionError('should not be called'))

        await h._run_deterministic_recon_sweep()  # must not raise

    @pytest.mark.asyncio
    async def test_per_item_fail_soft(self) -> None:
        h = _make_recon_harness()
        metadata = _strand_metadata()
        bad_task = {'id': 'tid-bad', 'status': 'blocked', 'metadata': metadata}
        good_task = {'id': 'tid-good', 'status': 'blocked', 'metadata': metadata}
        h.scheduler.get_tasks = AsyncMock(return_value=[bad_task, good_task])
        h._escalation_queue.get_by_task = MagicMock(return_value=[])
        h._escalation_queue.get_pending = MagicMock(return_value=[])

        calls: list[str] = []

        async def _recover(tid, task, meta):
            calls.append(tid)
            if tid == 'tid-bad':
                raise RuntimeError('boom')

        h._recover_stranded_deterministic_task = _recover  # type: ignore[method-assign]
        h._revalidate_open_deterministic_escalation = AsyncMock()  # type: ignore[method-assign]

        await h._run_deterministic_recon_sweep()  # must not raise

        assert calls == ['tid-bad', 'tid-good']
