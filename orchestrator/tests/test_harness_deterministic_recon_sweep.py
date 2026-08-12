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

ζ/task 2240 (DS-3 freshness verdict) additionally covers:
  - TestRevalidateDeployHealth: Harness._revalidate_deterministic_deploy_health
    reads ``deploy_state.verify_baseline`` and threads it through to
    ``_deterministic_deploy_health_verdict`` — see test_systemd_inspect.py
    for the canonical verdict-function coverage of the freshness branch
    itself (live monotonic advanced past baseline AND MainPID>0).
  - TestRunDeterministicReconSweep (D1 composition): a FULL sweep pass —
    Source A's real ``_recover_stranded_deterministic_task``, not mocked
    out — over a ``deploy_state.phase=='ran'`` strand with a persisted
    baseline, proving the freshness verdict (not phantom-done) drives the
    recovery escalation's category end-to-end.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import _init_harness_state_for_test, wire_scheduler_liveness_mock
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig
from orchestrator.deploy_state import DeployPhase
from orchestrator.event_store import EventStore, EventType
from orchestrator.harness import (
    Harness,
    _deterministic_deploy_health_verdict,
    _deterministic_deploy_stranded,
    _deterministic_gate_stranded,
    _recon_inspect_unit,
)
from orchestrator.recovery_emission import (
    RECOVERY_VETO_STREAK_SENTINEL_PREFIX,
    RecoverySite,
    RecoverySweepTally,
    RecoveryVetoStreakTracker,
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

    def test_false_for_legacy_shape_with_verified_at_set(self) -> None:
        """Reviewer amendment (task 2240): a pre-ζ deploy that already
        reached VERIFIED (before_done_verified_at stamped) is a terminal
        outcome, not a RAN-strand, even though it predates deploy_state and
        still carries before_done_ran_at. Preserves the exclusion the
        deleted ``_is_stranded_deterministic_shape`` enforced."""
        metadata = _strand_metadata(before_done_verified_at='2026-07-01T00:05:00+00:00')
        assert 'deploy_state' not in metadata
        assert _deterministic_deploy_stranded(metadata) is False

    def test_false_for_legacy_shape_with_gate_escalated_at_set(self) -> None:
        """Reviewer amendment (task 2240): a pre-ζ act-then-ask deploy
        blocked at its gate (gate_escalated_at stamped) is a terminal
        outcome, not a RAN-strand — e.g. its escalation was just resolved
        but the task has not yet been re-dispatched to done."""
        metadata = _strand_metadata(gate_escalated_at='2026-07-01T00:05:00+00:00')
        assert 'deploy_state' not in metadata
        assert _deterministic_deploy_stranded(metadata) is False

    def test_false_for_legacy_shape_with_done_provenance_set(self) -> None:
        """Reviewer amendment (task 2240): a pre-ζ deploy that already
        recorded done_provenance is a terminal outcome, not a RAN-strand."""
        metadata = _strand_metadata(done_provenance={'kind': 'deterministic-deploy'})
        assert 'deploy_state' not in metadata
        assert _deterministic_deploy_stranded(metadata) is False

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
# step-1/step-2 (task 2954): harness._deterministic_gate_stranded — the
# pure-gate / always_escalates GATE-strand detector, disjoint from the deploy
# detector above.
# ---------------------------------------------------------------------------


class TestDeterministicGateStranded:
    """step-1: harness._deterministic_gate_stranded — the pure-gate /
    ``always_escalates`` GATE-strand detector.

    Keys solely on ``task_kind=='deterministic'`` + a truthy
    ``gate_escalated_at`` (the sole signal a human-decision gate was supposed
    to be filed).  Deliberately DISJOINT from ``_deterministic_deploy_stranded``
    (which requires ``deploy_state.phase==RAN`` and explicitly excludes
    ``gate_escalated_at``) so no task is ever matched by both.  Metadata-only —
    the empty-escalation-queue I/O check is the caller's job, mirroring the
    deploy predicate's contract.
    """

    _ISO = '2026-07-01T00:05:00+00:00'

    def test_true_for_pure_gate_with_gate_escalated_at(self) -> None:
        """(a) pure gate that stamped its gate escalation → True."""
        assert _deterministic_gate_stranded({
            'task_kind': 'deterministic',
            'always_escalates': True,
            'before_done': None,
            'gate_escalated_at': self._ISO,
        }) is True

    def test_false_for_pure_gate_without_gate_escalated_at(self) -> None:
        """(b) pure gate that never stamped (no gate filed yet) → False."""
        assert _deterministic_gate_stranded({
            'task_kind': 'deterministic',
            'always_escalates': True,
            'before_done': None,
        }) is False

    def test_true_for_act_then_ask_deploy_gate_that_stamped(self) -> None:
        """(c) an act-then-ask deploy that reached its human gate
        (gate_escalated_at stamped, phase escalated) is a GATE strand — and is
        NOT a deploy RAN-strand, so the two detectors stay disjoint here too."""
        md = {
            'task_kind': 'deterministic',
            'before_done': {'target_unit': 'fused-memory.service'},
            'gate_escalated_at': self._ISO,
            'deploy_state': {'phase': DeployPhase.ESCALATED},
        }
        assert _deterministic_gate_stranded(md) is True
        assert _deterministic_deploy_stranded(md) is False

    def test_deploy_ran_strand_is_disjoint_from_gate_strand(self) -> None:
        """(d) a deploy RAN-strand (target_unit set, phase==RAN, NO
        gate_escalated_at) is owned by the deploy detector ONLY — pins that the
        two detectors never both match the same task."""
        ran_strand = _strand_metadata(phase=DeployPhase.RAN)
        assert _deterministic_gate_stranded(ran_strand) is False
        assert _deterministic_deploy_stranded(ran_strand) is True

    def test_false_when_task_kind_not_deterministic(self) -> None:
        """(e) task_kind != 'deterministic' → False even with gate_escalated_at."""
        assert _deterministic_gate_stranded({
            'task_kind': 'normal',
            'gate_escalated_at': self._ISO,
        }) is False

    def test_false_for_none_metadata(self) -> None:
        """(f) None metadata → False (no raise)."""
        assert _deterministic_gate_stranded(None) is False

    def test_false_for_non_dict_metadata(self) -> None:
        """(f) non-dict metadata → False (no raise)."""
        # Deliberately pass types the ``dict | None`` signature forbids to
        # exercise the runtime ``isinstance`` guard documented as "non-dict
        # metadata is treated as non-matching rather than raising".
        assert _deterministic_gate_stranded('not-a-dict') is False  # type: ignore[arg-type]
        assert _deterministic_gate_stranded(123) is False  # type: ignore[arg-type]


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
# step-3/step-4 (task 2954): Harness._recover_stranded_deterministic_gate
# (Source A — pure-gate / always_escalates GATE strand recovery)
# ---------------------------------------------------------------------------


class TestRecoverStrandedDeterministicGate:
    """step-3 (task 2954): RE-FILE-NEVER-FLIP recovery for pure-gate /
    always_escalates GATE strands.

    Uses a REAL EscalationQueue on tmp_path so the re-filed record is
    queryable. Re-files a born-at-L2 milestone_gate escalation mirroring what
    DeterministicRunner._file_milestone_gate_and_block itself would have filed
    (agent_role='orchestrator-deterministic', level=2, severity='critical',
    category='milestone_gate') so the runner's resume quiescence/resolve-to-done
    machinery (which scopes on that agent_role) integrates cleanly — never flips
    the task status (the gate is a human decision).
    """

    @staticmethod
    def _gate_task(tid: str = 'task-2954') -> tuple[dict, dict]:
        metadata = {
            'task_kind': 'deterministic',
            'always_escalates': True,
            'before_done': None,
            'gate_escalated_at': '2026-07-01T00:05:00+00:00',
            'gate_options': ['approve', 'reject'],
        }
        task = {
            'id': tid,
            'title': 'Human milestone gate for reify 5330',
            'description': 'Approve the reify milestone before the run proceeds.',
            'metadata': metadata,
        }
        return task, metadata

    @pytest.mark.asyncio
    async def test_refiles_l2_milestone_gate_never_flips_status(self, tmp_path: Path) -> None:
        h = _make_recon_harness()
        h.event_store = MagicMock()
        h.scheduler.set_task_status = AsyncMock()
        queue = EscalationQueue(tmp_path)
        h._escalation_queue = queue
        tid = 'task-2954'
        task, metadata = self._gate_task(tid)

        await h._recover_stranded_deterministic_gate(tid, task, metadata)

        # (a) exactly one born-at-L2 milestone_gate escalation, correct fields
        pending = queue.get_by_task(tid, status='pending')
        assert len(pending) == 1
        esc = pending[0]
        assert esc.agent_role == 'orchestrator-deterministic'
        assert esc.level == 2
        assert esc.severity == 'critical'
        assert esc.category == 'milestone_gate'
        assert esc.summary == task['title']
        assert task['description'] in esc.detail
        assert esc.options == metadata['gate_options']
        assert esc.task_id == tid
        # (b) RE-FILE-NEVER-FLIP: the task is already blocked — no status change.
        h.scheduler.set_task_status.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_emits_escalation_created_event(self, tmp_path: Path) -> None:
        h = _make_recon_harness()
        h.event_store = MagicMock()
        queue = EscalationQueue(tmp_path)
        h._escalation_queue = queue
        tid = 'task-2954'
        task, metadata = self._gate_task(tid)

        await h._recover_stranded_deterministic_gate(tid, task, metadata)

        # (c) emits an escalation_created event with the gate-strand reason
        h.event_store.emit.assert_called_once()
        args, kwargs = h.event_store.emit.call_args
        assert args[0] == EventType.escalation_created
        assert kwargs['task_id'] == tid
        assert kwargs['data']['reason'] == 'deterministic-recon-sweep-gate-strand-recovery'
        assert kwargs['data']['level'] == 2
        assert kwargs['data']['category'] == 'milestone_gate'

    @pytest.mark.asyncio
    async def test_noop_when_no_escalation_queue(self, tmp_path: Path) -> None:
        h = _make_recon_harness()
        h.event_store = MagicMock()
        h._escalation_queue = None
        h.scheduler.set_task_status = AsyncMock()
        tid = 'task-2954'
        task, metadata = self._gate_task(tid)

        # (d) graceful no-op when the queue is unavailable
        await h._recover_stranded_deterministic_gate(tid, task, metadata)  # must not raise

        h.event_store.emit.assert_not_called()
        h.scheduler.set_task_status.assert_not_awaited()


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
        assert kwargs.get('resolution_class') == 'benign'

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
        _args, kwargs = h._escalation_queue.resolve.call_args  # type: ignore[union-attr, attr-defined]
        assert kwargs.get('resolution_class') == 'benign'


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
            # tally=: the pass's recovery accumulator, threaded so this site's
            # holds reach the sweep's streak release (task 3535 S28).  ANY, not
            # a literal: what is asserted here is the POSITIONAL contract.
            'tid-a', task, metadata, tally=ANY,
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
            # tally=: the pass's recovery accumulator, threaded so this site's
            # holds reach the sweep's streak release (task 3535 S28).  ANY, not
            # a literal: what is asserted here is the POSITIONAL contract.
            'tid-a', task, metadata, tally=ANY,
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

        async def _recover(tid, task, meta, *, tally=None):
            calls.append(tid)
            if tid == 'tid-bad':
                raise RuntimeError('boom')

        h._recover_stranded_deterministic_task = _recover  # type: ignore[method-assign]
        h._revalidate_open_deterministic_escalation = AsyncMock()  # type: ignore[method-assign]

        await h._run_deterministic_recon_sweep()  # must not raise

        assert calls == ['tid-bad', 'tid-good']

    # -----------------------------------------------------------------------
    # ζ/task 2240 D1 composition: a FULL sweep pass (Source A's real
    # _recover_stranded_deterministic_task, NOT mocked out this time) over a
    # blocked deterministic deploy at deploy_state.phase=='ran' with a
    # persisted verify_baseline, proving the freshness verdict — not
    # phantom-done — drives the recovery escalation's category end-to-end.
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_d1_ran_phase_with_advanced_baseline_files_stranded_blocked_resume(self) -> None:
        h = _make_recon_harness()
        h.event_store = None
        metadata = _strand_metadata(
            phase=DeployPhase.RAN,
            verify_baseline={'active_enter_timestamp_monotonic': 100, 'main_pid': 999},
        )
        task = {
            'id': 'tid-fresh', 'status': 'blocked',
            'description': 'Deploy fused-memory restart', 'metadata': metadata,
        }
        h.scheduler.get_tasks = AsyncMock(return_value=[task])
        h._escalation_queue.get_by_task = MagicMock(return_value=[])  # type: ignore[union-attr]
        h._escalation_queue.get_pending = MagicMock(return_value=[])  # type: ignore[union-attr]
        # ActiveState deliberately NOT 'active' -- proves the FRESHNESS
        # branch (monotonic advance), not the liveness-only fallback, drove
        # the verdict.
        h._recon_unit_inspector = AsyncMock(return_value={
            'MainPID': 4321, 'ActiveState': 'deactivating',
            'ActiveEnterTimestampMonotonic': 500,
        })

        await h._run_deterministic_recon_sweep()

        h._escalation_queue.submit.assert_called_once()  # type: ignore[union-attr, attr-defined]
        esc = h._escalation_queue.submit.call_args[0][0]  # type: ignore[union-attr, attr-defined]
        assert esc.category == 'stranded_blocked'
        assert esc.suggested_action == 'resume'
        h.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_d1_ran_phase_with_stale_baseline_files_infra_issue_never_done(self) -> None:
        h = _make_recon_harness()
        h.event_store = None
        metadata = _strand_metadata(
            phase=DeployPhase.RAN,
            verify_baseline={'active_enter_timestamp_monotonic': 500, 'main_pid': 999},
        )
        task = {
            'id': 'tid-stale', 'status': 'blocked',
            'description': 'Deploy fused-memory restart', 'metadata': metadata,
        }
        h.scheduler.get_tasks = AsyncMock(return_value=[task])
        h._escalation_queue.get_by_task = MagicMock(return_value=[])  # type: ignore[union-attr]
        h._escalation_queue.get_pending = MagicMock(return_value=[])  # type: ignore[union-attr]
        # Live unit reports 'active' -- the OLD liveness-only check would
        # call this 'healthy'; the freshness branch must override it to
        # 'unconfirmed' because the monotonic did NOT advance past baseline
        # (D1: a crash between ran and verified recovers at phase==ran with
        # the defined re-escalate action, NEVER phantom-done).
        h._recon_unit_inspector = AsyncMock(return_value={
            'MainPID': 4321, 'ActiveState': 'active',
            'ActiveEnterTimestampMonotonic': 500,
        })

        await h._run_deterministic_recon_sweep()

        h._escalation_queue.submit.assert_called_once()  # type: ignore[union-attr, attr-defined]
        esc = h._escalation_queue.submit.call_args[0][0]  # type: ignore[union-attr, attr-defined]
        assert esc.category == 'infra_issue'
        assert esc.suggested_action == 'manual_intervention'
        h.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    # -----------------------------------------------------------------------
    # task 2954: pure-gate / always_escalates GATE-strand recovery wired into
    # Source A — FULL sweep passes over a REAL EscalationQueue on tmp_path so
    # the re-filed (or NOT re-filed) record is queryable end-to-end.
    # -----------------------------------------------------------------------

    @staticmethod
    def _gate_metadata() -> dict:
        return {
            'task_kind': 'deterministic',
            'always_escalates': True,
            'before_done': None,
            'gate_escalated_at': '2026-07-01T00:05:00+00:00',
            'gate_options': ['approve', 'reject'],
        }

    @pytest.mark.asyncio
    async def test_gate_strand_refired_when_no_escalation_anywhere(self, tmp_path: Path) -> None:
        """(a) blocked pure-gate task, gate_escalated_at set, NO record anywhere
        (empty queue + empty archive) → sweep re-fires exactly ONE L2
        milestone_gate escalation, queryable and scoped to the runner's role."""
        h = _make_recon_harness()
        h.event_store = None
        queue = EscalationQueue(tmp_path)
        h._escalation_queue = queue
        tid = 'tid-gate'
        task = {
            'id': tid, 'status': 'blocked',
            'title': 'Human gate 5330', 'description': 'Approve reify 5330.',
            'metadata': self._gate_metadata(),
        }
        h.scheduler.get_tasks = AsyncMock(return_value=[task])
        h.scheduler.get_statuses = AsyncMock(return_value=({}, None))

        await h._run_deterministic_recon_sweep()

        pending = queue.get_by_task(tid, status='pending')
        assert len(pending) == 1
        assert pending[0].category == 'milestone_gate'
        assert pending[0].level == 2
        assert pending[0].agent_role == 'orchestrator-deterministic'

    @pytest.mark.asyncio
    async def test_gate_strand_not_refired_when_pending_exists(self, tmp_path: Path) -> None:
        """(b) same task but a PENDING deterministic escalation already exists →
        sweep does NOT re-fire (quiescent, still exactly one)."""
        h = _make_recon_harness()
        h.event_store = None
        queue = EscalationQueue(tmp_path)
        h._escalation_queue = queue
        tid = 'tid-gate'
        existing = Escalation(
            id=queue.make_id(tid), task_id=tid,
            agent_role='orchestrator-deterministic', severity='critical',
            category='milestone_gate', summary='Human gate 5330', level=2,
        )
        queue.submit(existing)
        task = {
            'id': tid, 'status': 'blocked',
            'title': 'Human gate 5330', 'description': 'Approve reify 5330.',
            'metadata': self._gate_metadata(),
        }
        h.scheduler.get_tasks = AsyncMock(return_value=[task])
        h.scheduler.get_statuses = AsyncMock(return_value=({}, None))

        await h._run_deterministic_recon_sweep()

        pending = queue.get_by_task(tid, status='pending')
        assert len(pending) == 1  # still exactly one — no duplicate re-fired
        assert pending[0].id == existing.id

    @pytest.mark.asyncio
    async def test_gate_strand_not_refired_when_resolved_archived(self, tmp_path: Path) -> None:
        """(c) same task but the only escalation is RESOLVED+ARCHIVED → the
        archive-inclusive discriminator recognizes a genuinely-resolved gate,
        NOT a strand, so the sweep does NOT re-fire."""
        h = _make_recon_harness()
        h.event_store = None
        queue = EscalationQueue(tmp_path)
        h._escalation_queue = queue
        tid = 'tid-gate'
        existing = Escalation(
            id=queue.make_id(tid), task_id=tid,
            agent_role='orchestrator-deterministic', severity='critical',
            category='milestone_gate', summary='Human gate 5330', level=2,
        )
        queue.submit(existing)
        queue.resolve(existing.id, 'human approved the gate', resolved_by='human')
        # Sanity: nothing pending, but the archive holds the resolved record.
        assert queue.get_by_task(tid, status='pending') == []
        assert len(queue.get_by_task(tid, agent_role='orchestrator-deterministic')) == 1

        task = {
            'id': tid, 'status': 'blocked',
            'title': 'Human gate 5330', 'description': 'Approve reify 5330.',
            'metadata': self._gate_metadata(),
        }
        h.scheduler.get_tasks = AsyncMock(return_value=[task])
        h.scheduler.get_statuses = AsyncMock(return_value=({}, None))

        await h._run_deterministic_recon_sweep()

        # No NEW pending re-fired; the archived resolved record is untouched.
        assert queue.get_by_task(tid, status='pending') == []
        assert len(queue.get_by_task(tid, agent_role='orchestrator-deterministic')) == 1

    @pytest.mark.asyncio
    async def test_gate_and_deploy_strands_disjoint_in_same_pass(self, tmp_path: Path) -> None:
        """(d) mutual-exclusion: a deploy RAN-strand and a gate strand in the
        SAME pass are each handled by their OWN recovery — the gate task is not
        double-handled by the deploy path, and vice-versa."""
        h = _make_recon_harness()
        h.event_store = None
        queue = EscalationQueue(tmp_path)
        h._escalation_queue = queue
        h._recon_unit_inspector = AsyncMock(
            return_value={'MainPID': 4321, 'ActiveState': 'active'}
        )
        gate_tid = 'tid-gate'
        deploy_tid = 'tid-deploy'
        gate_task = {
            'id': gate_tid, 'status': 'blocked',
            'title': 'Human gate 5330', 'description': 'Approve reify 5330.',
            'metadata': self._gate_metadata(),
        }
        deploy_task = {
            'id': deploy_tid, 'status': 'blocked',
            'description': 'Deploy fused-memory restart',
            'metadata': _strand_metadata(phase=DeployPhase.RAN),
        }
        h.scheduler.get_tasks = AsyncMock(return_value=[gate_task, deploy_task])
        h.scheduler.get_statuses = AsyncMock(return_value=({}, None))

        await h._run_deterministic_recon_sweep()

        # Gate task → exactly one L2 milestone_gate from the runner's sentinel role.
        gate_escs = queue.get_by_task(gate_tid, status='pending')
        assert len(gate_escs) == 1
        assert gate_escs[0].category == 'milestone_gate'
        assert gate_escs[0].agent_role == 'orchestrator-deterministic'
        # Deploy task → exactly one L1 deploy-strand recovery from the SWEEP
        # role; NOT a milestone_gate, and NOT double-handled by the gate path.
        deploy_escs = queue.get_by_task(deploy_tid, status='pending')
        assert len(deploy_escs) == 1
        assert deploy_escs[0].category != 'milestone_gate'
        assert deploy_escs[0].agent_role == 'harness-deterministic-recon-sweep'


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


def _make_stranded_reaper_harness(tmp_path: Path) -> Harness:
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
    # TaskGroundTruth._resolve_branch_state (task 2243, W10-θ2) awaits
    # resolve_branch_sha unconditionally as its "does the branch still
    # exist" check — a bare MagicMock attribute isn't awaitable, so this
    # must be an AsyncMock. None preserves this fixture's "no on-main
    # evidence" invariant (docstring above).
    h.git_ops.resolve_branch_sha = AsyncMock(return_value=None)
    # TaskGroundTruth.derive_truth (task 2243, W10-θ2) resolves a worktree
    # path via _resolve_task_worktree for both _resolve_live_claimant's
    # plan.lock read and worktree_present — a bare MagicMock chain there
    # isn't a real Path, so json.loads(<MagicMock>.read_text()) blows up
    # with TypeError instead of gracefully reading "no lock file". Route
    # through a real (never-created) tmp_path so `.exists()` cleanly
    # returns False, matching this fixture's "no worktree" invariant.
    h.git_ops.warm_lane_pool = None
    h.git_ops.worktree_base = tmp_path / 'worktrees'
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
    async def test_deterministic_task_not_escalated_by_generic_reaper(
        self, tmp_path: Path,
    ) -> None:
        h = _make_stranded_reaper_harness(tmp_path)
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
    async def test_non_deterministic_task_still_escalated_by_generic_reaper(
        self, tmp_path: Path,
    ) -> None:
        """Control case: the exclusion is scoped to task_kind=='deterministic'
        only — an ordinary blocked task is unaffected and still gets the
        generic backstop escalation."""
        h = _make_stranded_reaper_harness(tmp_path)
        # task 2243, W10-θ2: recovery_for's db_status comes from this same
        # get_task() mock (TaskGroundTruth.derive_truth) — it must reflect
        # 'blocked' for the RE_FILE_ESCALATION table row (g) to match; a
        # status-less dict makes db_status='' and the resolver defaults to
        # LEAVE, starving this test's positive-control escalation.
        h.scheduler.get_task = AsyncMock(return_value={'status': 'blocked', 'metadata': {}})  # type: ignore[attr-defined]

        result = await h._reconcile_one_stranded('tid-normal', 'blocked', mid_run=False)

        assert result is None
        h._escalation_queue.submit.assert_called_once()  # type: ignore[union-attr, attr-defined]
        esc = h._escalation_queue.submit.call_args[0][0]  # type: ignore[union-attr, attr-defined]
        assert esc.category == 'stranded_blocked'
        assert esc.agent_role == 'harness-stranded-blocked-reaper'


# ---------------------------------------------------------------------------
# Task 3535 (PRD task beta, D5) — STRUCTURED EMISSION at the deterministic
# recon pair.
#
# The same "already has a pending escalation" predicate is implemented TWICE,
# a few hundred lines apart: once in _run_deterministic_recon_sweep's Source-A
# deploy branch (a completely SILENT `continue`) and once at the head of
# _recover_stranded_deterministic_task (info-log-only).  Collapsing the pair
# is task eta's (3541); this task makes BOTH emit, under DISTINCT site labels,
# so the duplication becomes a measurable fact in the event store rather than
# an assumption.  These tests must therefore NOT assert that only one fires.
# ---------------------------------------------------------------------------

def _pinning_esc(esc_id: str = 'esc-pin-1', *, task_id: str = 'tid-pinned') -> Escalation:
    """A REAL pending record — the shape ``get_by_task`` actually returns."""
    return Escalation(
        id=esc_id,
        task_id=task_id,
        agent_role='harness-deterministic-recon-sweep',
        severity='blocking',
        category='infra_issue',
        summary=f'{esc_id} summary',
        level=1,
        timestamp='2026-07-01T00:00:00+00:00',
    )


def _emitting_recon_harness(tmp_path: Path) -> Harness:
    """``_make_recon_harness`` plus a REAL EventStore on tmp_path.

    Real, not a MagicMock, so the payload's JSON round-trip through the store
    is genuinely exercised — an unserialisable member would drop the whole row
    in production and a mock would happily accept it.
    """
    h = _make_recon_harness()
    h.event_store = EventStore(tmp_path / 'runs.db', 'run-test')
    return h


def _recon_recovery_rows(h: Harness) -> list[dict]:
    """Every recovery event row, oldest first, ``data`` JSON-decoded."""
    conn = sqlite3.connect(str(h.event_store.db_path))  # type: ignore[union-attr]
    try:
        return [
            {'task_id': tid, 'event_type': et, 'data': json.loads(raw or '{}')}
            for tid, et, raw in conn.execute(
                'SELECT task_id, event_type, data FROM events '
                'WHERE event_type IN (?, ?) ORDER BY id',
                (EventType.recovery_vetoed.value, EventType.recovery_left.value),
            )
        ]
    finally:
        conn.close()


def _pinned_deploy_strand(tid: str = 'tid-pinned') -> dict:
    """A blocked deploy RAN-strand whose recovery is already deduped away."""
    return {
        'id': tid, 'status': 'blocked',
        'description': 'Deploy fused-memory restart',
        'metadata': _strand_metadata(phase=DeployPhase.RAN),
    }


class TestDeterministicReconSweepSiteEmits:
    """The sweep's Source-A deploy dedup skip — the SILENT half of the pair."""

    @pytest.mark.asyncio
    async def test_dedup_skip_emits_recovery_vetoed(self, tmp_path: Path) -> None:
        h = _emitting_recon_harness(tmp_path)
        task = _pinned_deploy_strand()
        h.scheduler.get_tasks = AsyncMock(return_value=[task])
        h._escalation_queue.get_by_task = MagicMock(return_value=[_pinning_esc()])  # type: ignore[union-attr]
        h._escalation_queue.get_pending = MagicMock(return_value=[])  # type: ignore[union-attr]

        await h._run_deterministic_recon_sweep()

        rows = _recon_recovery_rows(h)
        assert len(rows) == 1, f'expected exactly one row, got {rows}'
        assert rows[0]['event_type'] == EventType.recovery_vetoed.value
        assert rows[0]['task_id'] == 'tid-pinned'
        data = rows[0]['data']
        assert data['site'] == 'deterministic_recon_sweep'
        assert data['reason'] == 'escalation_pinned'
        assert data['store_unavailable'] is False

    @pytest.mark.asyncio
    async def test_payload_names_the_pinning_id_and_its_age(self, tmp_path: Path) -> None:
        h = _emitting_recon_harness(tmp_path)
        h.scheduler.get_tasks = AsyncMock(return_value=[_pinned_deploy_strand()])
        h._escalation_queue.get_by_task = MagicMock(return_value=[_pinning_esc()])  # type: ignore[union-attr]
        h._escalation_queue.get_pending = MagicMock(return_value=[])  # type: ignore[union-attr]

        await h._run_deterministic_recon_sweep()

        data = _recon_recovery_rows(h)[0]['data']
        ids = [i for bucket in data['escalation_ids'].values() for i in bucket]
        assert ids == ['esc-pin-1']
        assert set(data['ages_secs']) == {'esc-pin-1'}, (
            'ages_secs is a MAPPING keyed by id — an id can never silently '
            'vanish from the payload'
        )

    @pytest.mark.asyncio
    async def test_payload_names_the_deploy_phase_it_knows(self, tmp_path: Path) -> None:
        """This site genuinely knows the deploy phase, so it states it; the
        elements it does not resolve render 'unknown' rather than a guess."""
        h = _emitting_recon_harness(tmp_path)
        h.scheduler.get_tasks = AsyncMock(return_value=[_pinned_deploy_strand()])
        h._escalation_queue.get_by_task = MagicMock(return_value=[_pinning_esc()])  # type: ignore[union-attr]
        h._escalation_queue.get_pending = MagicMock(return_value=[])  # type: ignore[union-attr]

        await h._run_deterministic_recon_sweep()

        assert _recon_recovery_rows(h)[0]['data']['shape'] == (
            'blocked|unknown|unknown|true|ran'
        )

    @pytest.mark.asyncio
    async def test_dedup_skip_still_files_and_flips_nothing(self, tmp_path: Path) -> None:
        """The disposition half: RE-FILE-NEVER-FLIP, and the dedup still wins."""
        h = _emitting_recon_harness(tmp_path)
        h.scheduler.get_tasks = AsyncMock(return_value=[_pinned_deploy_strand()])
        h._escalation_queue.get_by_task = MagicMock(return_value=[_pinning_esc()])  # type: ignore[union-attr]
        h._escalation_queue.get_pending = MagicMock(return_value=[])  # type: ignore[union-attr]

        await h._run_deterministic_recon_sweep()

        h._escalation_queue.submit.assert_not_called()  # type: ignore[union-attr, attr-defined]
        h.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_a_recovered_strand_emits_nothing(self, tmp_path: Path) -> None:
        """An un-deduped strand is ACTED on, not held — no row."""
        h = _emitting_recon_harness(tmp_path)
        h.scheduler.get_tasks = AsyncMock(return_value=[_pinned_deploy_strand()])
        h._escalation_queue.get_by_task = MagicMock(return_value=[])  # type: ignore[union-attr]
        h._escalation_queue.get_pending = MagicMock(return_value=[])  # type: ignore[union-attr]
        h._recon_unit_inspector = AsyncMock(return_value={
            'MainPID': 4321, 'ActiveState': 'active',
        })

        await h._run_deterministic_recon_sweep()

        h._escalation_queue.submit.assert_called_once()  # type: ignore[union-attr, attr-defined]
        assert _recon_recovery_rows(h) == [], 'an ACTION is not a hold'

    @pytest.mark.asyncio
    async def test_repeat_passes_with_an_identical_hold_stay_bounded(
        self, tmp_path: Path,
    ) -> None:
        """Two passes over the same unchanged hold re-state it once."""
        h = _emitting_recon_harness(tmp_path)
        h.scheduler.get_tasks = AsyncMock(return_value=[_pinned_deploy_strand()])
        h._escalation_queue.get_by_task = MagicMock(return_value=[_pinning_esc()])  # type: ignore[union-attr]
        h._escalation_queue.get_pending = MagicMock(return_value=[])  # type: ignore[union-attr]

        await h._run_deterministic_recon_sweep()
        await h._run_deterministic_recon_sweep()

        assert len(_recon_recovery_rows(h)) == 1


class TestDeterministicReconDeploySiteEmits:
    """``_recover_stranded_deterministic_task``'s dedup skip — the
    INFO-log-only half.  Exercised by calling the method DIRECTLY, since a
    real sweep pass short-circuits at its twin above."""

    @pytest.mark.asyncio
    async def test_dedup_skip_emits_recovery_vetoed_under_its_own_site(
        self, tmp_path: Path,
    ) -> None:
        h = _emitting_recon_harness(tmp_path)
        task = _pinned_deploy_strand()
        h._escalation_queue.get_by_task = MagicMock(return_value=[_pinning_esc()])  # type: ignore[union-attr]

        await h._recover_stranded_deterministic_task(
            'tid-pinned', task, task['metadata'],
        )

        rows = _recon_recovery_rows(h)
        assert len(rows) == 1, f'expected exactly one row, got {rows}'
        assert rows[0]['event_type'] == EventType.recovery_vetoed.value
        data = rows[0]['data']
        assert data['site'] == 'deterministic_recon_deploy'
        assert data['reason'] == 'escalation_pinned'
        assert [i for b in data['escalation_ids'].values() for i in b] == ['esc-pin-1']

    @pytest.mark.asyncio
    async def test_the_existing_info_log_line_is_kept(
        self, tmp_path: Path, caplog,
    ) -> None:
        """The structured event is the MACHINE-readable record; the log line
        stays the human one.  Replacing it would break existing greps."""
        h = _emitting_recon_harness(tmp_path)
        task = _pinned_deploy_strand()
        h._escalation_queue.get_by_task = MagicMock(return_value=[_pinning_esc()])  # type: ignore[union-attr]

        with caplog.at_level('INFO', logger='orchestrator.harness'):
            await h._recover_stranded_deterministic_task(
                'tid-pinned', task, task['metadata'],
            )

        rendered = [r.getMessage() for r in caplog.records]
        assert any('already has a pending escalation' in m for m in rendered), (
            f'the dedup INFO line must survive: {rendered}'
        )

    @pytest.mark.asyncio
    async def test_dedup_skip_files_nothing(self, tmp_path: Path) -> None:
        h = _emitting_recon_harness(tmp_path)
        task = _pinned_deploy_strand()
        h._escalation_queue.get_by_task = MagicMock(return_value=[_pinning_esc()])  # type: ignore[union-attr]

        await h._recover_stranded_deterministic_task(
            'tid-pinned', task, task['metadata'],
        )

        h._escalation_queue.submit.assert_not_called()  # type: ignore[union-attr, attr-defined]
        h.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]


class TestDeterministicReconPairIsMeasurable:
    """The point of the two labels: the duplication becomes a FACT."""

    @pytest.mark.asyncio
    async def test_the_pair_speaks_under_two_distinct_site_labels(
        self, tmp_path: Path,
    ) -> None:
        """Deliberately NOT asserting that only one fires — de-duplicating
        the predicate is task eta's (3541), and a shared label would hide
        exactly the duplication eta needs to measure."""
        h = _emitting_recon_harness(tmp_path)
        task = _pinned_deploy_strand()
        h.scheduler.get_tasks = AsyncMock(return_value=[task])
        h._escalation_queue.get_by_task = MagicMock(return_value=[_pinning_esc()])  # type: ignore[union-attr]
        h._escalation_queue.get_pending = MagicMock(return_value=[])  # type: ignore[union-attr]

        await h._run_deterministic_recon_sweep()
        await h._recover_stranded_deterministic_task(
            'tid-pinned', task, task['metadata'],
        )

        sites = [r['data']['site'] for r in _recon_recovery_rows(h)]
        assert sites == ['deterministic_recon_sweep', 'deterministic_recon_deploy'], (
            'both halves of the duplicated predicate must be independently '
            'attributable in the event store'
        )


class TestDeterministicReconQueueAbsent:
    """A missing queue is a PROCESS-scoped fact, not a per-task one."""

    @pytest.mark.asyncio
    async def test_sweep_queue_absent_emits_one_recovery_left(
        self, tmp_path: Path,
    ) -> None:
        h = _emitting_recon_harness(tmp_path)
        h._escalation_queue = None

        for _ in range(3):
            await h._run_deterministic_recon_sweep()

        rows = _recon_recovery_rows(h)
        assert len(rows) == 1, 'once per PROCESS, not once per pass'
        assert rows[0]['event_type'] == EventType.recovery_left.value
        assert rows[0]['task_id'] is None, (
            'the whole sweep is degraded — no single task is the subject'
        )
        data = rows[0]['data']
        assert data['reason'] == 'escalation_store_unavailable'
        assert data['store_unavailable'] is True
        assert data['site'] == 'deterministic_recon_sweep'

    @pytest.mark.asyncio
    async def test_recover_queue_absent_emits_one_recovery_left(
        self, tmp_path: Path,
    ) -> None:
        h = _emitting_recon_harness(tmp_path)
        task = _pinned_deploy_strand()
        h._escalation_queue = None

        for _ in range(3):
            await h._recover_stranded_deterministic_task(
                'tid-pinned', task, task['metadata'],
            )

        rows = _recon_recovery_rows(h)
        assert len(rows) == 1
        assert rows[0]['task_id'] is None
        assert rows[0]['data']['site'] == 'deterministic_recon_deploy', (
            'each half latches its own notice, so silencing one never '
            'silences the other'
        )

    @pytest.mark.asyncio
    async def test_the_two_halves_latch_independently(self, tmp_path: Path) -> None:
        h = _emitting_recon_harness(tmp_path)
        task = _pinned_deploy_strand()
        h._escalation_queue = None

        await h._run_deterministic_recon_sweep()
        await h._recover_stranded_deterministic_task(
            'tid-pinned', task, task['metadata'],
        )

        assert [r['data']['site'] for r in _recon_recovery_rows(h)] == [
            'deterministic_recon_sweep', 'deterministic_recon_deploy',
        ]


class TestDeterministicReconGateCheckStaysSilent:
    """PRD D3 carve-out: the archive-INCLUSIVE, role-scoped gate check is one
    of the predicates the PRD names as deliberately different and documented
    as staying separate.  Emitting there would blur a boundary drawn on
    purpose, so its silence is asserted — a later reader must see the
    omission as intentional rather than a missed site."""

    @pytest.mark.asyncio
    async def test_gate_strand_skip_emits_nothing(self, tmp_path: Path) -> None:
        h = _emitting_recon_harness(tmp_path)
        task = {
            'id': 'tid-gate', 'status': 'blocked', 'description': 'gate task',
            'metadata': _strand_metadata(
                gate_escalated_at='2026-07-01T01:00:00+00:00',
                always_escalates=True,
            ),
        }
        h.scheduler.get_tasks = AsyncMock(return_value=[task])
        h._escalation_queue.get_by_task = MagicMock(return_value=[_pinning_esc()])  # type: ignore[union-attr]
        h._escalation_queue.get_pending = MagicMock(return_value=[])  # type: ignore[union-attr]

        await h._run_deterministic_recon_sweep()

        h._escalation_queue.submit.assert_not_called()  # type: ignore[union-attr, attr-defined]
        assert _recon_recovery_rows(h) == [], (
            'the archive-inclusive role-scoped gate check is DELIBERATELY '
            'untouched by task 3535 (PRD D3)'
        )


class TestDeterministicReconZeroBehaviorChange:
    """Emission may not move a single disposition."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('pinned', [True, False], ids=['held', 'recovered'])
    async def test_writes_are_identical_with_emission_on_and_off(
        self, tmp_path: Path, pinned: bool,
    ) -> None:
        outcomes = []
        for enabled in (True, False):
            h = _emitting_recon_harness(tmp_path / f'on-{enabled}')
            h.config.recovery_emission.enabled = enabled
            h.scheduler.get_tasks = AsyncMock(return_value=[_pinned_deploy_strand()])
            h._escalation_queue.get_by_task = MagicMock(  # type: ignore[union-attr]
                return_value=[_pinning_esc()] if pinned else [],
            )
            h._escalation_queue.get_pending = MagicMock(return_value=[])  # type: ignore[union-attr]
            h._recon_unit_inspector = AsyncMock(return_value={
                'MainPID': 4321, 'ActiveState': 'active',
            })

            await h._run_deterministic_recon_sweep()

            submitted = [
                (c[0][0].task_id, c[0][0].category, c[0][0].suggested_action)
                for c in h._escalation_queue.submit.call_args_list  # type: ignore[union-attr, attr-defined]
            ]
            outcomes.append((
                submitted,
                h.scheduler.set_task_status.call_args_list,  # type: ignore[attr-defined]
            ))
            if not enabled:
                assert _recon_recovery_rows(h) == []

        assert outcomes[0] == outcomes[1], (
            'emission changed a disposition — the one thing this task may not do'
        )
        assert outcomes[0][1] == [], 'RE-FILE-NEVER-FLIP: no status write, ever'

    @pytest.mark.asyncio
    async def test_source_b_still_skips_a_tid_source_a_recovered(
        self, tmp_path: Path,
    ) -> None:
        """``recovered_this_pass`` bookkeeping is untouched: a tid Source A
        recovered is still never handled by Source B in the same pass."""
        h = _emitting_recon_harness(tmp_path)
        task = _pinned_deploy_strand()
        h.scheduler.get_tasks = AsyncMock(return_value=[task])
        h._escalation_queue.get_by_task = MagicMock(return_value=[])  # type: ignore[union-attr]
        h._escalation_queue.get_pending = MagicMock(  # type: ignore[union-attr]
            return_value=[_pinning_esc(task_id='tid-pinned')],
        )
        h._recon_unit_inspector = AsyncMock(return_value={
            'MainPID': 4321, 'ActiveState': 'active',
        })
        h._revalidate_open_deterministic_escalation = AsyncMock()  # type: ignore[method-assign]

        await h._run_deterministic_recon_sweep()

        h._revalidate_open_deterministic_escalation.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_an_emission_failure_never_aborts_the_pass(
        self, tmp_path: Path,
    ) -> None:
        """Telemetry sits inside the per-task fail-soft guard, so a sibling
        sorted after a failing task is still swept."""
        h = _emitting_recon_harness(tmp_path)
        h._emit_recovery_disposition = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError('telemetry exploded'),
        )
        held = _pinned_deploy_strand('tid-held')
        free = _pinned_deploy_strand('tid-free')
        h.scheduler.get_tasks = AsyncMock(return_value=[held, free])
        h._escalation_queue.get_by_task = MagicMock(  # type: ignore[union-attr]
            side_effect=lambda tid, **kw: [_pinning_esc()] if tid == 'tid-held' else [],
        )
        h._escalation_queue.get_pending = MagicMock(return_value=[])  # type: ignore[union-attr]
        h._recon_unit_inspector = AsyncMock(return_value={
            'MainPID': 4321, 'ActiveState': 'active',
        })

        await h._run_deterministic_recon_sweep()

        submitted = [
            c[0][0].task_id
            for c in h._escalation_queue.submit.call_args_list  # type: ignore[union-attr, attr-defined]
        ]
        assert submitted == ['tid-free'], (
            'a telemetry failure on one task must not collateral-strand its sibling'
        )


# ---------------------------------------------------------------------------
# Task 3535 S27 — the deterministic-recon sweep's MISSING RELEASE HALF.
#
# Both deterministic sites are in ``STREAK_CHARGING_SITES``, so both can file a
# blocking L1 against ``__recovery_veto_streak__<tid>``.  The only release path
# is site-filtered to ``reconcile_sweep`` and driven only from
# ``_reconcile_stranded_in_progress``, so a deterministic hold that CLEARS
# leaves (a) an operator holding a blocking alarm for a resolved condition and
# (b) the detector permanently silenced for that task, because the emitter
# dedups on ``has_open_l1`` — both failures ``resolve_recovery_veto_streak_
# escalation``'s own docstring calls "NOT optional polish".  The tracker and the
# filed_at memo also grow one permanent entry per deterministic task ever held,
# against ``RecoveryVetoStreakTracker``'s stated "proportional to the tasks
# CURRENTLY held" footprint contract.
#
# The sentinel and the memo are keyed on task_id ALONE while the tracker is
# keyed on ``(site, task_id)``, so two charging sites SHARE one alarm.  That is
# why the release must be guarded on no OTHER charging site still holding the
# tid: an unguarded per-site release would resolve an alarm the other sweep's
# live hold still justifies, and the next pass would re-file it — an alarm flap
# on an operator's queue, strictly worse than the stale alarm it fixed.
# ---------------------------------------------------------------------------

class _FakeClock:
    """A monotonic clock, so the alarm's SPAN dimension is drivable.

    The predicate is two-dimensional (streak AND elapsed span); a test that
    could not control time could only ever exercise one half of it.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, secs: float) -> None:
        self.now += secs


#: ``deterministic_recon_sweep_interval_secs`` — the real pass cadence, and
#: half the published derivation for ``veto_streak_min_span_secs``.
_RECON_INTERVAL = 900.0


def _streak_recon_harness(tmp_path: Path) -> tuple[Harness, _FakeClock]:
    """A recon harness on a REAL EscalationQueue, with a drivable clock.

    Real queue, not a mock: the sentinel alarm's ``has_open_l1`` dedup, its
    resolution and the re-file-after-recurrence all live in the queue's own
    archive handling, and a mock would assert none of it.
    """
    h = _make_recon_harness()
    h.event_store = EventStore(tmp_path / 'runs.db', 'run-test')
    h._escalation_queue = EscalationQueue(tmp_path / 'queue')
    clock = _FakeClock()
    h._recovery_veto_tracker = RecoveryVetoStreakTracker(clock=clock)
    h._recon_unit_inspector = AsyncMock(return_value={
        'MainPID': 4321, 'ActiveState': 'active',
    })
    # Sources B and C are orthogonal to the release half; keep them inert so a
    # test failure here can only mean the release misbehaved.
    h.scheduler.get_task = AsyncMock(return_value=None)
    h._revalidate_open_l2 = AsyncMock(return_value=False)  # type: ignore[method-assign]
    h._revalidate_open_deterministic_escalation = AsyncMock()  # type: ignore[method-assign]
    return h, clock


def _hold(h: Harness, tid: str = 'tid-pinned', esc_id: str = 'esc-pin-1') -> Escalation:
    """Put a REAL pending record on *tid*, so the sweep's dedup skips it."""
    esc = _pinning_esc(esc_id, task_id=tid)
    h._escalation_queue.submit(esc)  # type: ignore[union-attr]
    return esc


def _sentinel_alarms(h: Harness, tid: str = 'tid-pinned') -> list:
    """Pending veto-streak alarms filed against *tid*'s SENTINEL id."""
    return h._escalation_queue.get_by_task(  # type: ignore[union-attr]
        f'{RECOVERY_VETO_STREAK_SENTINEL_PREFIX}{tid}', status='pending',
    )


async def _recon_passes(
    h: Harness, clock: _FakeClock, count: int, *, interval: float = _RECON_INTERVAL,
) -> None:
    """Run *count* deterministic passes *interval* apart, as the loop would."""
    for i in range(count):
        if i:
            clock.advance(interval)
        await h._run_deterministic_recon_sweep()


def _tracked_for(h: Harness, tid: str) -> set[str]:
    """Every SITE currently tracking a veto streak for *tid*."""
    return {
        str(site) for site, tracked_tid in h._recovery_veto_tracker.tracked()
        if tracked_tid == tid
    }


class TestDeterministicReconStreakRelease:
    """A deterministic hold that ENDS must stand its alarm down."""

    @pytest.mark.asyncio
    async def test_a_sustained_hold_files_one_alarm(self, tmp_path: Path) -> None:
        """The charge half — already true, pinned here as this section's premise."""
        h, clock = _streak_recon_harness(tmp_path)
        h.scheduler.get_tasks = AsyncMock(return_value=[_pinned_deploy_strand()])
        _hold(h)

        await _recon_passes(h, clock, 3)

        alarms = _sentinel_alarms(h)
        assert len(alarms) == 1, f'expected one sentinel alarm, got {alarms}'
        assert alarms[0].level == 1

    @pytest.mark.asyncio
    async def test_the_next_pass_after_the_hold_clears_resolves_the_alarm(
        self, tmp_path: Path,
    ) -> None:
        """The missing half: today this alarm stays pending forever."""
        h, clock = _streak_recon_harness(tmp_path)
        h.scheduler.get_tasks = AsyncMock(return_value=[_pinned_deploy_strand()])
        held = _hold(h)
        await _recon_passes(h, clock, 3)
        assert len(_sentinel_alarms(h)) == 1

        h._escalation_queue.resolve(held.id, 'unblocked')  # type: ignore[union-attr]
        clock.advance(_RECON_INTERVAL)
        await h._run_deterministic_recon_sweep()

        assert _sentinel_alarms(h) == [], (
            'the hold ended, so its alarm must stand down — otherwise an '
            'operator holds a blocking L1 for a resolved condition'
        )

    @pytest.mark.asyncio
    async def test_the_release_drops_the_tracker_and_memo_entries(
        self, tmp_path: Path,
    ) -> None:
        """The footprint contract: proportional to CURRENTLY-held tasks.

        This process runs for weeks; one permanent entry per deterministic task
        ever held is an unbounded leak, not a rounding error.
        """
        h, clock = _streak_recon_harness(tmp_path)
        h.scheduler.get_tasks = AsyncMock(return_value=[_pinned_deploy_strand()])
        held = _hold(h)
        await _recon_passes(h, clock, 3)
        assert _tracked_for(h, 'tid-pinned') == {'deterministic_recon_sweep'}

        h._escalation_queue.resolve(held.id, 'unblocked')  # type: ignore[union-attr]
        clock.advance(_RECON_INTERVAL)
        await h._run_deterministic_recon_sweep()

        assert _tracked_for(h, 'tid-pinned') == set()
        assert 'tid-pinned' not in getattr(h, '_recovery_streak_filed_at', {})

    @pytest.mark.asyncio
    async def test_a_later_recurrence_files_a_new_alarm(self, tmp_path: Path) -> None:
        """Without the release, ``has_open_l1`` silences this detector for the
        task for the life of the queue — one incident per task, ever."""
        h, clock = _streak_recon_harness(tmp_path)
        h.scheduler.get_tasks = AsyncMock(return_value=[_pinned_deploy_strand()])
        held = _hold(h)
        await _recon_passes(h, clock, 3)
        first = _sentinel_alarms(h)[0].id

        # The hold clears; the next pass releases and RE-FILES on the real tid
        # (RE-FILE-NEVER-FLIP), which is what holds the task again below.
        h._escalation_queue.resolve(held.id, 'unblocked')  # type: ignore[union-attr]
        clock.advance(_RECON_INTERVAL)
        await h._run_deterministic_recon_sweep()
        assert _sentinel_alarms(h) == []

        await _recon_passes(h, clock, 3)

        alarms = _sentinel_alarms(h)
        assert len(alarms) == 1, f'the recurrence must alarm again, got {alarms}'
        assert alarms[0].id != first, 'a NEW alarm, not the resurrected old one'

    @pytest.mark.asyncio
    async def test_the_release_writes_nothing_to_the_subject_task(
        self, tmp_path: Path,
    ) -> None:
        """Zero behavior change still binds: the release is bookkeeping."""
        h, clock = _streak_recon_harness(tmp_path)
        h.scheduler.get_tasks = AsyncMock(return_value=[_pinned_deploy_strand()])
        held = _hold(h)
        await _recon_passes(h, clock, 3)
        h._escalation_queue.resolve(held.id, 'unblocked')  # type: ignore[union-attr]
        clock.advance(_RECON_INTERVAL)

        await h._run_deterministic_recon_sweep()

        h.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        # The pass RE-FILED for the real tid (unchanged behaviour); the release
        # itself contributed no record of its own to the subject.
        real = h._escalation_queue.get_by_task('tid-pinned', status='pending')  # type: ignore[union-attr]
        assert [e.category for e in real] == ['stranded_blocked']


class TestDeterministicReconStreakReleaseIsSiteScoped:
    """The two sweeps have DIFFERENT candidate sets, so neither may stand down
    the other's live alarm."""

    @pytest.mark.asyncio
    async def test_a_deterministic_pass_leaves_a_reconcile_streak_alone(
        self, tmp_path: Path,
    ) -> None:
        """``T-other`` is not in the deterministic candidate set at all."""
        h, clock = _streak_recon_harness(tmp_path)
        h.scheduler.get_tasks = AsyncMock(return_value=[_pinned_deploy_strand()])
        _hold(h)
        h._recovery_veto_tracker.observe(
            RecoverySite.reconcile_sweep, 'T-other', 'sig',
        )

        await _recon_passes(h, clock, 3)

        assert _tracked_for(h, 'T-other') == {'reconcile_sweep'}, (
            'a sweep may only release streaks for the sites IT drives'
        )

    @pytest.mark.asyncio
    async def test_a_deterministic_pass_will_not_resolve_a_still_held_alarm(
        self, tmp_path: Path,
    ) -> None:
        """CROSS-SITE SUPPRESSION — the flap this guard exists to prevent.

        The sentinel and the memo are task-scoped while the tracker is
        (site, task_id)-scoped, so one alarm covers both sites.  Resolving it
        while the reconcile sweep still holds the task would simply re-file on
        the next reconcile pass: an alarm flap, worse than the stale alarm.
        """
        h, clock = _streak_recon_harness(tmp_path)
        h.scheduler.get_tasks = AsyncMock(return_value=[_pinned_deploy_strand()])
        held = _hold(h)
        await _recon_passes(h, clock, 3)
        assert len(_sentinel_alarms(h)) == 1
        for _ in range(3):
            h._recovery_veto_tracker.observe(
                RecoverySite.reconcile_sweep, 'tid-pinned', 'sig',
            )

        h._escalation_queue.resolve(held.id, 'unblocked')  # type: ignore[union-attr]
        clock.advance(_RECON_INTERVAL)
        await h._run_deterministic_recon_sweep()

        assert _tracked_for(h, 'tid-pinned') == {'reconcile_sweep'}, (
            'the deterministic entry pops; the reconcile one is not this '
            "pass's to touch"
        )
        assert len(_sentinel_alarms(h)) == 1, (
            'still held at another charging site — the alarm stays up'
        )

    @pytest.mark.asyncio
    async def test_a_reconcile_release_will_not_resolve_a_deterministic_hold(
        self, tmp_path: Path,
    ) -> None:
        """The mirror direction, on the PRE-EXISTING reconcile-only path — the
        same latent hazard, which the cross-site guard closes for both."""
        h, clock = _streak_recon_harness(tmp_path)
        h.scheduler.get_tasks = AsyncMock(return_value=[_pinned_deploy_strand()])
        _hold(h)
        await _recon_passes(h, clock, 3)
        assert len(_sentinel_alarms(h)) == 1
        for _ in range(3):
            h._recovery_veto_tracker.observe(
                RecoverySite.reconcile_sweep, 'tid-pinned', 'sig',
            )

        # A reconcile pass that swept nothing: its own entry is stale.
        h._release_recovery_veto_streaks(RecoverySweepTally())

        assert _tracked_for(h, 'tid-pinned') == {'deterministic_recon_sweep'}
        assert len(_sentinel_alarms(h)) == 1, (
            'the deterministic sweep still holds this task — its shared alarm '
            'must not be stood down by the other sweep'
        )
