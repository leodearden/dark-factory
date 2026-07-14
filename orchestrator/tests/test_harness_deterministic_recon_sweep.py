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
  step-5:  TestDeterministicDeployStranded (ζ/task 2240: phase-based
           classifier replacing the deleted stamp-archaeology
           TestIsStrandedDeterministicShape)
  step-7:  TestRecoverStrandedDeterministicTask
  step-9:  TestRevalidateOpenDeterministicEscalation
  step-11: TestRunDeterministicReconSweep
  step-13: TestDeterministicReconSweepLifecycle

Amendment pass (post-review) additionally covers:
  - TestReconInspectUnit: the module-level I/O inspector's systemctl-output
    parsing (previously only exercised via the injected-mock seam).
  - TestRunDeterministicReconSweep: Source-B's scheduler.get_task fallback
    (escalation task not in the blocked-tasks map) and its None-guard, plus
    the literal same-pass mutual-exclusion between Source A and Source B.
  - TestGenericStrandedBlockedReaperSkipsDeterministic: the pre-existing
    generic stranded-blocked reaper (_reconcile_one_stranded) now excludes
    task_kind=='deterministic' tasks, delegating them exclusively to this
    sweep so the health check is never bypassed by the generic backstop.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import _init_harness_state_for_test, wire_scheduler_liveness_mock
from escalation.models import Escalation

from orchestrator.config import OrchestratorConfig
from orchestrator.deploy_state import DeployPhase
from orchestrator.harness import (
    Harness,
    _deterministic_deploy_health_verdict,
    _deterministic_deploy_stranded,
    _recon_inspect_unit,
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

    # --- ζ/task 2240 DS-3: deploy_state.verify_baseline pass-through --------

    @pytest.mark.asyncio
    async def test_revalidate_passes_deploy_state_verify_baseline_to_verdict(self) -> None:
        """_revalidate_deterministic_deploy_health must read
        deploy_state.verify_baseline from metadata and thread it through to
        the verdict fn — proven behaviourally: an inspect result with
        ActiveState NOT 'active' (so the liveness-only branch would read
        'unconfirmed') still verdicts 'healthy' once the live monotonic has
        advanced past a persisted baseline, because the freshness branch
        does not require ActiveState=='active'."""
        h = _make_recon_harness()
        h._recon_unit_inspector = AsyncMock(return_value={
            'MainPID': 4321, 'ActiveState': 'deactivating',
            'ActiveEnterTimestampMonotonic': 500,
        })
        metadata = {
            'before_done': {'target_unit': 'fused-memory.service'},
            'deploy_state': {
                'phase': 'ran',
                'verify_baseline': {'active_enter_timestamp_monotonic': 100, 'main_pid': 999},
            },
        }

        verdict = await h._revalidate_deterministic_deploy_health(metadata)

        assert verdict == 'healthy'

    @pytest.mark.asyncio
    async def test_revalidate_unconfirmed_with_baseline_when_monotonic_stale(self) -> None:
        """The live unit reports 'active' (the OLD liveness-only check would
        call this 'healthy'), but the freshness branch overrides it to
        'unconfirmed' because the monotonic did NOT advance past baseline."""
        h = _make_recon_harness()
        h._recon_unit_inspector = AsyncMock(return_value={
            'MainPID': 4321, 'ActiveState': 'active',
            'ActiveEnterTimestampMonotonic': 100,
        })
        metadata = {
            'before_done': {'target_unit': 'fused-memory.service'},
            'deploy_state': {
                'phase': 'ran',
                'verify_baseline': {'active_enter_timestamp_monotonic': 100, 'main_pid': 999},
            },
        }

        verdict = await h._revalidate_deterministic_deploy_health(metadata)

        assert verdict == 'unconfirmed'

    @pytest.mark.asyncio
    async def test_revalidate_no_baseline_falls_back_to_liveness_only(self) -> None:
        """When deploy_state has no verify_baseline, the pre-existing
        liveness-only verdict is used (backward compat)."""
        h = _make_recon_harness()
        h._recon_unit_inspector = AsyncMock(return_value={
            'MainPID': 4321, 'ActiveState': 'active', 'ActiveEnterTimestampMonotonic': 5,
        })
        metadata = {
            'before_done': {'target_unit': 'fused-memory.service'},
            'deploy_state': {'phase': 'ran'},
        }

        verdict = await h._revalidate_deterministic_deploy_health(metadata)

        assert verdict == 'healthy'


# ---------------------------------------------------------------------------
# Amendment: _recon_inspect_unit — the real I/O default inspector.
#
# Every test above (and below) exercises the sweep exclusively via the
# injected ``_recon_unit_inspector`` seam.  ``_recon_inspect_unit`` itself —
# the only real-I/O function, containing the actual systemctl-output
# '='-partition parsing and int-coercion fallback — previously had zero
# direct coverage.  A regression here (e.g. MainPID left as a string) would
# silently make ``_deterministic_deploy_health_verdict`` return 'unconfirmed'
# always, defeating the whole sweep, with nothing to catch it.
# ---------------------------------------------------------------------------


class TestReconInspectUnit:
    """Amendment: cover _recon_inspect_unit's systemctl-output parsing."""

    def _mock_proc(self, stdout: bytes) -> AsyncMock:
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(stdout, b''))
        return proc

    @pytest.mark.asyncio
    async def test_parses_healthy_unit_output_and_coerces_ints(self) -> None:
        stdout = (
            b'MainPID=1234\n'
            b'ActiveState=active\n'
            b'ActiveEnterTimestamp=Sat 2026-07-04 12:00:00 UTC\n'
            b'ActiveEnterTimestampMonotonic=123456789\n'
        )
        with patch(
            'orchestrator.harness.asyncio.create_subprocess_exec',
            AsyncMock(return_value=self._mock_proc(stdout)),
        ):
            result = await _recon_inspect_unit('fused-memory.service')

        assert result['MainPID'] == 1234
        assert isinstance(result['MainPID'], int)
        assert result['ActiveState'] == 'active'
        assert result['ActiveEnterTimestampMonotonic'] == 123456789
        assert isinstance(result['ActiveEnterTimestampMonotonic'], int)
        assert _deterministic_deploy_health_verdict(result) == 'healthy'

    @pytest.mark.asyncio
    async def test_malformed_numeric_field_coerces_to_zero_not_raise(self) -> None:
        """A non-numeric MainPID (unexpected systemctl output shape) must
        coerce to 0 rather than raising, and classify 'unconfirmed'."""
        stdout = b'MainPID=not-a-number\nActiveState=failed\n'
        with patch(
            'orchestrator.harness.asyncio.create_subprocess_exec',
            AsyncMock(return_value=self._mock_proc(stdout)),
        ):
            result = await _recon_inspect_unit('unknown.service')

        assert result['MainPID'] == 0
        assert isinstance(result['MainPID'], int)
        assert _deterministic_deploy_health_verdict(result) == 'unconfirmed'

    @pytest.mark.asyncio
    async def test_empty_stdout_coerces_to_zero_and_unconfirmed(self) -> None:
        """A gone/unknown unit (empty systemctl show output) must not raise
        and must classify 'unconfirmed'."""
        with patch(
            'orchestrator.harness.asyncio.create_subprocess_exec',
            AsyncMock(return_value=self._mock_proc(b'')),
        ):
            result = await _recon_inspect_unit('gone.service')

        assert result.get('MainPID') == 0
        assert isinstance(result['MainPID'], int)
        assert _deterministic_deploy_health_verdict(result) == 'unconfirmed'


# ---------------------------------------------------------------------------
# step-5: pure fn _is_stranded_deterministic_shape
# ---------------------------------------------------------------------------


def _strand_metadata(
    phase: str | None = None, verify_baseline: dict | None = None, **overrides
) -> dict:
    """Canonical stranded-deterministic-shape metadata (task 2059 EVIDENCE).

    ``phase`` (+ optional ``verify_baseline``), when given, seeds a ζ
    ``metadata['deploy_state']`` slice (``{'phase': phase}``, merging in
    ``verify_baseline`` when provided) — the phase-bearing shape used by the
    phase-based strand classifier tests. Omitting ``phase`` (the default)
    keeps the pre-ζ stamp-only shape — no ``deploy_state`` key at all — so
    existing backward-compat parity tests can still build stamp-only tasks.
    """
    base: dict = {
        'task_kind': 'deterministic',
        'before_done': {'target_unit': 'fused-memory.service'},
        'before_done_ran_at': '2026-07-01T00:00:00+00:00',
        'before_done_verified_at': None,
        'gate_escalated_at': None,
        'done_provenance': None,
    }
    if phase is not None:
        deploy_state: dict = {'phase': phase}
        if verify_baseline is not None:
            deploy_state['verify_baseline'] = verify_baseline
        base['deploy_state'] = deploy_state
    base.update(overrides)
    return base


class TestDeterministicDeployStranded:
    """step-5 (ζ/task 2240): harness._deterministic_deploy_stranded — the
    phase-based classifier replacing the deleted 4-stamp combinatorial
    ``_is_stranded_deterministic_shape`` (DS-4). Because ζ writes
    ``deploy_state.phase`` atomically with every stamp, the old shape
    collapses to a single enum compare: phase == RAN.
    """

    # --- phase-authoritative (deploy_state present) --------------------------

    def test_true_when_phase_ran(self) -> None:
        assert _deterministic_deploy_stranded(_strand_metadata(phase=DeployPhase.RAN)) is True

    def test_false_when_phase_verified(self) -> None:
        assert _deterministic_deploy_stranded(_strand_metadata(phase=DeployPhase.VERIFIED)) is False

    def test_false_when_phase_scheduled(self) -> None:
        assert _deterministic_deploy_stranded(_strand_metadata(phase=DeployPhase.SCHEDULED)) is False

    def test_false_when_phase_escalated(self) -> None:
        assert _deterministic_deploy_stranded(_strand_metadata(phase=DeployPhase.ESCALATED)) is False

    def test_false_when_phase_done(self) -> None:
        assert _deterministic_deploy_stranded(_strand_metadata(phase=DeployPhase.DONE)) is False

    # --- backward-compat migration shim (no deploy_state at all) -------------

    def test_true_for_legacy_shape_no_deploy_state(self) -> None:
        """A deploy that began before ζ activated has before_done_ran_at
        stamped but no deploy_state key at all — treated as a RAN-strand
        (bounded, documented migration shim) so it isn't silently un-stranded
        the moment ζ ships."""
        metadata = _strand_metadata()
        assert 'deploy_state' not in metadata
        assert _deterministic_deploy_stranded(metadata) is True

    def test_false_for_legacy_shape_when_before_done_ran_at_missing(self) -> None:
        assert _deterministic_deploy_stranded(_strand_metadata(before_done_ran_at=None)) is False

    # --- gating checks unaffected by the phase/stamp swap --------------------

    def test_false_when_task_kind_not_deterministic(self) -> None:
        assert _deterministic_deploy_stranded(
            _strand_metadata(phase=DeployPhase.RAN, task_kind='normal')
        ) is False

    def test_false_when_before_done_is_none(self) -> None:
        assert _deterministic_deploy_stranded(
            _strand_metadata(phase=DeployPhase.RAN, before_done=None)
        ) is False

    def test_false_when_before_done_lacks_target_unit(self) -> None:
        assert _deterministic_deploy_stranded(
            _strand_metadata(phase=DeployPhase.RAN, before_done={})
        ) is False

    def test_false_for_empty_metadata(self) -> None:
        assert _deterministic_deploy_stranded({}) is False

    def test_false_for_none_metadata(self) -> None:
        assert _deterministic_deploy_stranded(None) is False


def test_is_stranded_deterministic_shape_no_longer_exists() -> None:
    """Grep-delete proof: the old stamp-archaeology classifier is gone from
    orchestrator.harness — DS-4 collapses it to _deterministic_deploy_stranded."""
    import orchestrator.harness as harness_module

    assert not hasattr(harness_module, '_is_stranded_deterministic_shape')


# ---------------------------------------------------------------------------
# step-7: Harness._recover_stranded_deterministic_task (Source A)
# ---------------------------------------------------------------------------


class TestRecoverStrandedDeterministicTask:
    """step-7: RE-FILE-NEVER-FLIP strand recovery for absent-escalation strands."""

    @pytest.mark.asyncio
    async def test_healthy_verdict_files_stranded_blocked_resume(self) -> None:
        h = _make_recon_harness()
        h.event_store = MagicMock()
        h._escalation_queue.get_by_task = MagicMock(return_value=[])  # type: ignore[union-attr]
        h._recon_unit_inspector = AsyncMock(
            return_value={'MainPID': 4321, 'ActiveState': 'active'}
        )
        tid = 'task-2059'
        task = {'id': tid, 'description': 'Deploy fused-memory restart'}
        metadata = _strand_metadata()

        await h._recover_stranded_deterministic_task(tid, task, metadata)

        h._escalation_queue.submit.assert_called_once()  # type: ignore[union-attr, attr-defined]
        esc = h._escalation_queue.submit.call_args[0][0]  # type: ignore[union-attr, attr-defined]
        assert esc.category == 'stranded_blocked'
        assert esc.level == 1
        assert esc.severity == 'blocking'
        assert esc.suggested_action == 'resume'
        assert esc.agent_role == 'harness-deterministic-recon-sweep'
        assert esc.task_id == tid
        h.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_unconfirmed_verdict_files_infra_issue_manual(self) -> None:
        h = _make_recon_harness()
        h.event_store = MagicMock()
        h._escalation_queue.get_by_task = MagicMock(return_value=[])  # type: ignore[union-attr]
        h._recon_unit_inspector = AsyncMock(
            return_value={'MainPID': 0, 'ActiveState': 'failed'}
        )
        tid = 'task-2059'
        task = {'id': tid, 'description': 'Deploy fused-memory restart'}
        metadata = _strand_metadata()

        await h._recover_stranded_deterministic_task(tid, task, metadata)

        h._escalation_queue.submit.assert_called_once()  # type: ignore[union-attr, attr-defined]
        esc = h._escalation_queue.submit.call_args[0][0]  # type: ignore[union-attr, attr-defined]
        assert esc.category == 'infra_issue'
        assert esc.level == 1
        assert esc.severity == 'blocking'
        assert esc.suggested_action == 'manual_intervention'
        assert esc.agent_role == 'harness-deterministic-recon-sweep'
        h.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_dedup_skips_when_pending_escalation_exists(self) -> None:
        h = _make_recon_harness()
        h.event_store = MagicMock()
        h._escalation_queue.get_by_task = MagicMock(return_value=[MagicMock()])  # type: ignore[union-attr]
        h._recon_unit_inspector = AsyncMock(
            return_value={'MainPID': 4321, 'ActiveState': 'active'}
        )
        tid = 'task-2059'
        task = {'id': tid, 'description': 'Deploy fused-memory restart'}
        metadata = _strand_metadata()

        await h._recover_stranded_deterministic_task(tid, task, metadata)

        h._escalation_queue.submit.assert_not_called()  # type: ignore[union-attr, attr-defined]
        h.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]


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

        h._escalation_queue.resolve.assert_called_once()  # type: ignore[union-attr, attr-defined]
        args, kwargs = h._escalation_queue.resolve.call_args  # type: ignore[union-attr, attr-defined]
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

        h._escalation_queue.resolve.assert_not_called()  # type: ignore[union-attr, attr-defined]

    @pytest.mark.asyncio
    async def test_never_touches_milestone_gate(self) -> None:
        h = _make_recon_harness()
        h._recon_unit_inspector = AsyncMock(
            return_value={'MainPID': 111, 'ActiveState': 'active'}
        )
        esc = _make_esc(category='milestone_gate', agent_role='orchestrator-deterministic')
        metadata = _strand_metadata()

        await h._revalidate_open_deterministic_escalation(esc, {'id': 'tid'}, metadata)

        h._escalation_queue.resolve.assert_not_called()  # type: ignore[union-attr, attr-defined]
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

        h._escalation_queue.resolve.assert_not_called()  # type: ignore[union-attr, attr-defined]
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

        h._escalation_queue.resolve.assert_called_once()  # type: ignore[union-attr, attr-defined]


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
        h._escalation_queue.get_by_task = MagicMock(return_value=[])  # type: ignore[union-attr]
        h._escalation_queue.get_pending = MagicMock(return_value=[])  # type: ignore[union-attr]
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
        h._escalation_queue.get_by_task = MagicMock(return_value=[esc])  # type: ignore[union-attr]
        h._escalation_queue.get_pending = MagicMock(return_value=[esc])  # type: ignore[union-attr]
        h._recover_stranded_deterministic_task = AsyncMock()  # type: ignore[method-assign]
        h._revalidate_open_deterministic_escalation = AsyncMock()  # type: ignore[method-assign]

        await h._run_deterministic_recon_sweep()

        h._recover_stranded_deterministic_task.assert_not_awaited()
        h._revalidate_open_deterministic_escalation.assert_awaited_once_with(
            esc, task, metadata
        )

    @pytest.mark.asyncio
    async def test_source_b_falls_back_to_scheduler_get_task(self) -> None:
        """Amendment: an open escalation whose task_id is NOT in the
        blocked-tasks map (e.g. the task has since moved to another status)
        must fall back to scheduler.get_task — previously untested."""
        h = _make_recon_harness()
        metadata = _strand_metadata()
        # A blocked-but-irrelevant task occupies the blocked-tasks map so
        # Source A has something to (correctly) ignore, keeping this test
        # isolated to the Source-B fallback path.
        other_task = {
            'id': 'tid-other', 'status': 'blocked', 'metadata': {'task_kind': 'normal'},
        }
        esc = _make_esc(
            category='infra_issue', agent_role='orchestrator-deterministic',
            task_id='tid-not-blocked',
        )
        fetched_task = {'id': 'tid-not-blocked', 'status': 'in-progress', 'metadata': metadata}
        h.scheduler.get_tasks = AsyncMock(return_value=[other_task])
        h.scheduler.get_task = AsyncMock(return_value=fetched_task)
        h._escalation_queue.get_by_task = MagicMock(return_value=[])  # type: ignore[union-attr]
        h._escalation_queue.get_pending = MagicMock(return_value=[esc])  # type: ignore[union-attr]
        h._recover_stranded_deterministic_task = AsyncMock()  # type: ignore[method-assign]
        h._revalidate_open_deterministic_escalation = AsyncMock()  # type: ignore[method-assign]

        await h._run_deterministic_recon_sweep()

        h.scheduler.get_task.assert_awaited_once_with('tid-not-blocked')  # type: ignore[attr-defined]
        h._revalidate_open_deterministic_escalation.assert_awaited_once_with(
            esc, fetched_task, metadata
        )

    @pytest.mark.asyncio
    async def test_source_b_skips_when_task_not_found_anywhere(self) -> None:
        """Amendment: when neither the blocked-tasks map nor
        scheduler.get_task can locate the escalation's task (e.g. a deleted/
        cancelled task), Source B must skip it without raising — the
        ``task is None: continue`` guard, previously untested."""
        h = _make_recon_harness()
        esc = _make_esc(
            category='infra_issue', agent_role='orchestrator-deterministic',
            task_id='tid-missing',
        )
        h.scheduler.get_tasks = AsyncMock(return_value=[])
        h.scheduler.get_task = AsyncMock(return_value=None)
        h._escalation_queue.get_by_task = MagicMock(return_value=[])  # type: ignore[union-attr]
        h._escalation_queue.get_pending = MagicMock(return_value=[esc])  # type: ignore[union-attr]
        h._recover_stranded_deterministic_task = AsyncMock()  # type: ignore[method-assign]
        h._revalidate_open_deterministic_escalation = AsyncMock()  # type: ignore[method-assign]

        await h._run_deterministic_recon_sweep()  # must not raise

        h.scheduler.get_task.assert_awaited_once_with('tid-missing')  # type: ignore[attr-defined]
        h._revalidate_open_deterministic_escalation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_source_a_recovered_tid_excluded_from_source_b_same_pass(self) -> None:
        """Amendment (reviewer finding #4): a tid Source A recovers in THIS
        pass must be skipped by the Source-B loop even though get_pending()
        already reflects the just-filed escalation — making the "A and B
        never double-handle one task" invariant literally true rather than
        an incidental no-op of today's category/verdict values."""
        h = _make_recon_harness()
        metadata = _strand_metadata()
        task = {'id': 'tid-a', 'status': 'blocked', 'metadata': metadata}
        # The escalation Source A "just filed" is already visible via
        # get_pending() within the same pass (queue re-globs the directory).
        filed_esc = _make_esc(
            category='infra_issue', agent_role='harness-deterministic-recon-sweep',
            task_id='tid-a',
        )
        h.scheduler.get_tasks = AsyncMock(return_value=[task])
        h._escalation_queue.get_by_task = MagicMock(return_value=[])  # type: ignore[union-attr]
        h._escalation_queue.get_pending = MagicMock(return_value=[filed_esc])  # type: ignore[union-attr]
        h._recover_stranded_deterministic_task = AsyncMock()  # type: ignore[method-assign]
        h._revalidate_open_deterministic_escalation = AsyncMock()  # type: ignore[method-assign]

        await h._run_deterministic_recon_sweep()

        h._recover_stranded_deterministic_task.assert_awaited_once_with(
            'tid-a', task, metadata
        )
        h._revalidate_open_deterministic_escalation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ignores_non_deterministic_and_done_tasks(self) -> None:
        h = _make_recon_harness()
        normal_task = {
            'id': 'tid-normal', 'status': 'blocked', 'metadata': {'task_kind': 'normal'},
        }
        done_task = {'id': 'tid-done', 'status': 'done', 'metadata': _strand_metadata()}
        h.scheduler.get_tasks = AsyncMock(return_value=[normal_task, done_task])
        h._escalation_queue.get_by_task = MagicMock(return_value=[])  # type: ignore[union-attr]
        h._escalation_queue.get_pending = MagicMock(return_value=[])  # type: ignore[union-attr]
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
        h._escalation_queue.get_by_task = MagicMock(return_value=[])  # type: ignore[union-attr]
        h._escalation_queue.get_pending = MagicMock(return_value=[])  # type: ignore[union-attr]

        calls: list[str] = []

        async def _recover(tid, task, meta):
            calls.append(tid)
            if tid == 'tid-bad':
                raise RuntimeError('boom')

        h._recover_stranded_deterministic_task = _recover  # type: ignore[method-assign]
        h._revalidate_open_deterministic_escalation = AsyncMock()  # type: ignore[method-assign]

        await h._run_deterministic_recon_sweep()  # must not raise

        assert calls == ['tid-bad', 'tid-good']


# ---------------------------------------------------------------------------
# step-13: start/stop/loop lifecycle wiring
# ---------------------------------------------------------------------------


def _make_lifecycle_harness(*, enabled: bool = True) -> Harness:
    """Build a minimal Harness for deterministic-recon-sweep lifecycle tests."""
    h = Harness.__new__(Harness)
    _init_harness_state_for_test(h)
    config = OrchestratorConfig()
    config = config.model_copy(update={'deterministic_recon_sweep_enabled': enabled})
    h.config = config
    h._deterministic_recon_sweep_task = None
    h._recon_unit_inspector = None
    return h


class TestDeterministicReconSweepLifecycle:
    """step-13: _start/_stop_deterministic_recon_sweep lifecycle wiring."""

    def test_start_deterministic_recon_sweep_creates_task(self) -> None:
        """When deterministic_recon_sweep_enabled=True, _start_deterministic_recon_sweep
        creates an asyncio.Task named 'deterministic-recon-sweep' and stores it."""
        h = _make_lifecycle_harness(enabled=True)
        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.done.return_value = False
        mock_task.get_name.return_value = 'deterministic-recon-sweep'

        def _create_task(coro, *, name=None):
            coro.close()
            return mock_task

        with patch('orchestrator.harness.asyncio.create_task', side_effect=_create_task) as mock_ct:
            h._start_deterministic_recon_sweep()

        assert h._deterministic_recon_sweep_task is mock_task
        _, kwargs = mock_ct.call_args
        assert kwargs.get('name') == 'deterministic-recon-sweep', (
            f'Expected task name deterministic-recon-sweep, got {kwargs.get("name")!r}'
        )

    def test_start_deterministic_recon_sweep_disabled(self) -> None:
        """When deterministic_recon_sweep_enabled=False, the start call is a no-op."""
        h = _make_lifecycle_harness(enabled=False)
        with patch('orchestrator.harness.asyncio.create_task') as mock_ct:
            h._start_deterministic_recon_sweep()
            mock_ct.assert_not_called()
        assert h._deterministic_recon_sweep_task is None

    def test_start_deterministic_recon_sweep_idempotent(self) -> None:
        """A live (not-done) existing task is not replaced by a second start call."""
        h = _make_lifecycle_harness(enabled=True)
        live_task = MagicMock(spec=asyncio.Task)
        live_task.done.return_value = False
        h._deterministic_recon_sweep_task = live_task

        with patch('orchestrator.harness.asyncio.create_task') as mock_ct:
            h._start_deterministic_recon_sweep()
            mock_ct.assert_not_called()

        assert h._deterministic_recon_sweep_task is live_task

    @pytest.mark.asyncio
    async def test_stop_deterministic_recon_sweep_cancels(self) -> None:
        """After _start_, _stop_deterministic_recon_sweep cancels the task and
        resets h._deterministic_recon_sweep_task to None."""
        h = _make_lifecycle_harness(enabled=True)
        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.done.return_value = False

        def _create_task(coro, *, name=None):
            coro.close()
            return mock_task

        with patch('orchestrator.harness.asyncio.create_task', side_effect=_create_task):
            h._start_deterministic_recon_sweep()

        assert h._deterministic_recon_sweep_task is mock_task

        mock_task.cancel.return_value = True

        async def _await_cancelled():
            raise asyncio.CancelledError()

        mock_task.__await__ = lambda self: _await_cancelled().__await__()

        await h._stop_deterministic_recon_sweep()

        mock_task.cancel.assert_called_once()
        assert h._deterministic_recon_sweep_task is None


class TestDeterministicReconSweepLoopFailureHandling:
    """step-13: the sweep loop logs a bounded summary and backs off on failure,
    mirroring _main_tip_sweep_loop's task-1907 bounded-failure discipline."""

    @pytest.mark.asyncio
    async def test_loop_failure_is_bounded_and_backs_off_not_spin(self) -> None:
        """A pass that fails immediately must back off (no tight-spin) and log a
        bounded summary via logger.error — NOT logger.exception."""
        h = _make_lifecycle_harness(enabled=True)
        # Interval 0 so the top-of-loop sleep returns immediately: without the
        # backoff guarantee, a failing pass would spin without yielding.
        h.config = h.config.model_copy(update={'deterministic_recon_sweep_interval_secs': 0.0})

        calls = {'n': 0}

        async def _failing_pass() -> None:
            calls['n'] += 1
            raise RuntimeError('sweep boom')

        h._run_deterministic_recon_sweep = _failing_pass  # type: ignore[method-assign]

        sleeps: list[float] = []
        real_sleep = asyncio.sleep

        async def _tracking_sleep(delay: float, *a, **k):
            sleeps.append(delay)
            # Stop the loop after a couple of failing iterations so the test
            # terminates deterministically without relying on wall-clock.
            if calls['n'] >= 2:
                raise asyncio.CancelledError()
            return await real_sleep(0)

        with patch('orchestrator.harness.asyncio.sleep', side_effect=_tracking_sleep), \
                patch('orchestrator.harness.logger') as mock_logger, \
                pytest.raises(asyncio.CancelledError):
            await h._deterministic_recon_sweep_loop()

        # logger.exception must NOT be used (pathological-traceback footgun).
        mock_logger.exception.assert_not_called()
        # A bounded one-line summary is logged via logger.error per failure.
        assert mock_logger.error.call_count >= 1, 'expected bounded logger.error summary'
        # The backoff sleep (60.0s) was requested after a failure — proving the
        # loop yields a real quiescent gap instead of tight-spinning.
        assert 60.0 in sleeps, f'expected a 60s backoff sleep, got {sleeps!r}'


# ---------------------------------------------------------------------------
# Amendment: the generic stranded-blocked reaper (_reconcile_one_stranded)
# must exclude task_kind=='deterministic' tasks, delegating them exclusively
# to _recover_stranded_deterministic_task (reviewer finding #2).
#
# Both reapers write to the same escalation queue and dedup on the same
# get_by_task(tid, status='pending') check, so whichever fires first wins.
# escalation-watcher-auto's `stranded_blocked` routing
# (skills/escalation-watcher-auto/SKILL.md) auto-resumes ANY
# category='stranded_blocked' L1 once its predicate holds, regardless of the
# filer's suggested_action or role — so if the generic reaper won the race
# for a deterministic-deploy task whose unit is actually down, it would file
# stranded_blocked/manual_intervention, which still gets auto-resumed,
# silently bypassing the live-health check.  Excluding deterministic tasks
# from the generic reaper means only the health-aware sweep ever files
# their first escalation.
# ---------------------------------------------------------------------------


def _make_stranded_reaper_harness() -> Harness:
    """Minimal bare Harness sufficient to drive _reconcile_one_stranded down
    the 'blocked, no on-main evidence' path, without the full git-ops-backed
    fixture in test_reconcile_stranded.py (out of this task's locked scope).
    is_ancestor=False + find_merge_marker=None short-circuit every on-main
    fast-path so the method falls straight through to the final
    status=='blocked' reaper block under test."""
    h = Harness.__new__(Harness)
    _init_harness_state_for_test(h)
    h.config = OrchestratorConfig()
    h.config.stranded_blocked_escalate_enabled = True
    h.git_ops = MagicMock()
    h.git_ops.config.branch_prefix = 'task/'
    h.git_ops.config.main_branch = 'main'
    h.git_ops.is_ancestor = AsyncMock(return_value=False)
    h.git_ops.find_merge_marker = AsyncMock(return_value=None)
    h.scheduler = MagicMock()
    # Task 2235: harness._reconcile_one_stranded's stranded-blocked gate now
    # calls self.scheduler.workflow_cancel_recent(tid) instead of reading
    # self._workflow_cancel_at directly. A bare MagicMock auto-mocks that
    # call to a truthy Mock regardless of state, which makes the `not
    # self.scheduler.workflow_cancel_recent(tid)` guard always False and
    # silently suppresses the escalation this fixture is meant to exercise.
    wire_scheduler_liveness_mock(h.scheduler)
    h._escalation_queue = MagicMock()
    h._escalation_queue.make_id = MagicMock(return_value='esc-reaper-1')
    h._escalation_queue.get_by_task = MagicMock(return_value=[])
    h._escalation_events = {}
    h._workflow_cancel_at = {}
    h.event_store = None
    return h


class TestGenericStrandedBlockedReaperSkipsDeterministic:
    """Amendment (reviewer finding #2): _reconcile_one_stranded's generic
    backstop must not race the health-aware deterministic-recon sweep."""

    @pytest.mark.asyncio
    async def test_deterministic_task_not_escalated_by_generic_reaper(self) -> None:
        h = _make_stranded_reaper_harness()
        h.scheduler.get_task = AsyncMock(return_value={  # type: ignore[attr-defined]
            'metadata': {
                'task_kind': 'deterministic',
                'before_done': {'target_unit': 'fused-memory.service'},
                'before_done_ran_at': '2026-07-01T00:00:00+00:00',
            },
        })

        result = await h._reconcile_one_stranded('tid-det', 'blocked', mid_run=False)

        assert result is None
        h._escalation_queue.submit.assert_not_called()  # type: ignore[union-attr, attr-defined]

    @pytest.mark.asyncio
    async def test_non_deterministic_task_still_escalated_by_generic_reaper(self) -> None:
        """Control case: the exclusion is scoped to task_kind=='deterministic'
        only — an ordinary blocked task is unaffected and still gets the
        generic backstop escalation."""
        h = _make_stranded_reaper_harness()
        h.scheduler.get_task = AsyncMock(return_value={'metadata': {}})  # type: ignore[attr-defined]

        result = await h._reconcile_one_stranded('tid-normal', 'blocked', mid_run=False)

        assert result is None
        h._escalation_queue.submit.assert_called_once()  # type: ignore[union-attr, attr-defined]
        esc = h._escalation_queue.submit.call_args[0][0]  # type: ignore[union-attr, attr-defined]
        assert esc.category == 'stranded_blocked'
        assert esc.agent_role == 'harness-stranded-blocked-reaper'
