"""Wiring tests for task 3535's recovery emission at the reconcile sweep.

The pure half of the mechanism (payload primitives, streak tracker, streak-L1
filer) is pinned in ``test_recovery_emission.py``.  THIS module pins the
harness WIRING: that ``Harness._reconcile_stranded_in_progress`` /
``_reconcile_one_stranded`` actually describe the dispositions they reach, at
the right cadence, WITHOUT moving any of them.

The load-bearing test in here is not any single emission assertion — it is
``TestZeroDispositionChange``, which re-runs each matrix row with
``recovery_emission.enabled`` flipped and demands byte-identical writes.  Task
3535's whole contract is emission BEFORE behavior (PRD
plans/task-escalation-state-graph-prd.md D5); an emission that changed a
disposition would be a defect no amount of correct payload could redeem.

Facts are injected through a stub ground-truth resolver, but the CLASSIFIER is
the real ``classify_recovery`` — so every row's disposition is the production
one and the ``_RECOVERY`` table is never mocked away.
"""

import json
import shutil
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import wire_scheduler_liveness_mock
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from shared.deploy_state import DeployPhase
from shared.task_statuses import TaskStatus

from orchestrator.config import RecoveryEmissionConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.harness import Harness
from orchestrator.landed_outbox import MergeProvenance
from orchestrator.recovery_emission import (
    RECOVERY_VETO_STREAK_SENTINEL_PREFIX,
    RecoveryVetoStreakTracker,
)
from orchestrator.task_ground_truth import (
    BranchState,
    BranchStateKind,
    Claimant,
    ClaimantSource,
    EscalationRef,
    RecoveryAction,
    TruthReport,
    classify_recovery,
)


@pytest.fixture(autouse=True)
def _reset_merge_provenance():
    """MergeProvenance._outbox is a process-global — never leak a bound outbox
    into another test (mirrors the identically-named fixture in
    test_reconcile_stranded.py / test_task_ground_truth.py)."""
    MergeProvenance._outbox = None
    yield
    MergeProvenance._outbox = None


# ---------------------------------------------------------------------------
# Report builders — the injected FACTS.  classify_recovery stays real.
# ---------------------------------------------------------------------------

def _iso(*, secs_ago: float) -> str:
    """An ISO-8601 UTC stamp *secs_ago* seconds in the past."""
    return (datetime.now(UTC) - timedelta(seconds=secs_ago)).isoformat()


def _ref(
    esc_id: str,
    *,
    level: int = 2,
    severity: str = 'blocking',
    category: str = 'needs_human',
    age_secs: float = 300.0,
) -> EscalationRef:
    return EscalationRef(
        id=esc_id,
        level=level,
        category=category,
        severity=severity,
        created_at=_iso(secs_ago=age_secs),
    )


def _report(
    *,
    db_status: str = TaskStatus.IN_PROGRESS,
    claimant: Claimant | None = None,
    branch: BranchStateKind = BranchStateKind.ON_MAIN,
    sha: str | None = 'a' * 40,
    escalations: list[EscalationRef] | None = None,
    deploy_phase: DeployPhase | None = None,
    store_unavailable: bool = False,
) -> TruthReport:
    return TruthReport(
        db_status=db_status,
        live_claimant=claimant,
        branch_state=BranchState(branch, sha),
        worktree_present=True,
        open_escalations=list(escalations or []),
        deploy_phase=deploy_phase,
        escalation_store_unavailable=store_unavailable,
    )


def _live_claimant() -> Claimant:
    return Claimant(
        run_id='run-live', heartbeat_at=_iso(secs_ago=5.0),
        source=ClaimantSource.IN_MEMORY,
    )


class _StubGroundTruth:
    """Stands in for TaskGroundTruth so a test pins the exact TruthReport.

    ``recovery_for`` composes the REAL ``classify_recovery`` over the injected
    report, exactly as the production resolver does — so a disposition asserted
    here is the one the ``_RECOVERY`` table actually produces, not a mocked
    stand-in.  Only the facts are injected.
    """

    def __init__(self, reports: dict[str, TruthReport]):
        self.escalation_queue = None
        self.reports = reports

    async def recovery_for(self, tid: str) -> tuple[TruthReport, RecoveryAction]:
        report = self.reports[tid]
        return report, classify_recovery(report)


def _bind_reports(harness: Harness, reports: dict[str, TruthReport]) -> _StubGroundTruth:
    """Point *harness* at a stub resolver holding *reports*, and drive the sweep
    over exactly those task ids."""
    stub = _StubGroundTruth(reports)
    harness._get_ground_truth = lambda: stub  # type: ignore[method-assign]
    statuses = {tid: str(r.db_status) for tid, r in reports.items()}
    harness.scheduler.get_statuses = AsyncMock(return_value=(statuses, None))
    harness.scheduler.get_task = AsyncMock(
        side_effect=lambda tid: {'status': statuses.get(tid), 'metadata': {}},
    )
    return stub


# ---------------------------------------------------------------------------
# Event-store readback
# ---------------------------------------------------------------------------

def _rows(harness: Harness, event_type: EventType | None = None) -> list[dict]:
    """Every event row, oldest first, with ``data`` JSON-decoded."""
    conn = sqlite3.connect(str(harness.event_store.db_path))  # type: ignore[union-attr]
    try:
        sql = 'SELECT task_id, event_type, data FROM events'
        params: tuple = ()
        if event_type is not None:
            sql += ' WHERE event_type = ?'
            params = (event_type.value,)
        return [
            {'task_id': tid, 'event_type': et, 'data': json.loads(raw or '{}')}
            for tid, et, raw in conn.execute(sql + ' ORDER BY id', params)
        ]
    finally:
        conn.close()


def _recovery_rows(harness: Harness) -> list[dict]:
    """Both recovery event types, in emission order."""
    return [
        r for r in _rows(harness)
        if r['event_type'] in (
            EventType.recovery_vetoed.value, EventType.recovery_left.value,
        )
    ]


# ---------------------------------------------------------------------------
# Harness fixture — the reconcile-sweep subset of test_reconcile_stranded.py's,
# plus a REAL EventStore and a REAL EscalationQueue on tmp_path (so JSON
# round-tripping and queue dedup are exercised for real, not mocked).
# ---------------------------------------------------------------------------

@pytest.fixture
def harness(tmp_path: Path, mock_orch_config):
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    wire_scheduler_liveness_mock(h.scheduler)
    h.scheduler.get_tasks = AsyncMock(return_value=[])
    h.scheduler.get_statuses = AsyncMock(return_value=({}, None))
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.mark_done = AsyncMock()

    h.git_ops.worktree_base = (tmp_path / '.worktrees').resolve()
    h.git_ops.cleanup_worktree = AsyncMock(
        side_effect=lambda path, tid: shutil.rmtree(path, ignore_errors=True),
    )
    h.git_ops.is_ancestor = AsyncMock(return_value=False)
    h.git_ops.resolve_branch_sha = AsyncMock(return_value='b' * 40)
    h.git_ops.find_merge_marker = AsyncMock(return_value=None)
    h.git_ops.find_task_citation_commit = AsyncMock(return_value='b' * 40)
    h.git_ops.warm_lane_ref_is_degenerate = AsyncMock(return_value=False)
    h.git_ops.commit_effect_present_in_main = AsyncMock(return_value=True)
    h._branch_is_degenerate = AsyncMock(return_value=False)

    # Real collaborators — the emission path writes to both.
    h.event_store = EventStore(tmp_path / 'runs.db', 'run-test')
    h._escalation_queue = EscalationQueue(tmp_path / 'escalations')

    # A REAL config model: mock_orch_config's spec'd MagicMock would hand the
    # streak predicate a MagicMock threshold, which the emitter's fail-open
    # guard would swallow — hiding whatever this suite meant to assert.
    h.config.recovery_emission = RecoveryEmissionConfig()
    h.config.stranded_blocked_escalate_enabled = False

    return h


# ---------------------------------------------------------------------------
# Boundary #9 — row (f): IN_PROGRESS + no claimant + ON_MAIN + open escalation
# classifies LEAVE.  The MARK_DONE the evidence would otherwise justify is
# VETOED, and until now that veto was completely silent.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBoundary9VetoIsDescribed:
    async def test_pinned_on_main_task_is_left_alone(self, harness):
        """The disposition half: nothing is written, exactly as before."""
        _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})

        await harness._reconcile_stranded_in_progress(mid_run=False)

        assert harness.scheduler.set_task_status.await_count == 0, (
            'boundary #9 must remain a LEAVE — emission may not flip a status'
        )
        assert harness.scheduler.mark_done.await_count == 0

    async def test_pinned_on_main_task_emits_one_vetoed_row(self, harness):
        _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})

        await harness._reconcile_stranded_in_progress(mid_run=False)

        rows = _recovery_rows(harness)
        assert len(rows) == 1, f'expected exactly one recovery row, got {rows}'
        assert rows[0]['event_type'] == EventType.recovery_vetoed.value
        assert rows[0]['task_id'] == 'T1', (
            'task_id must be a first-class column so the row joins against '
            'task_completed / escalation_created'
        )

    async def test_vetoed_payload_carries_the_full_key_set(self, harness):
        _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})

        await harness._reconcile_stranded_in_progress(mid_run=False)

        data = _recovery_rows(harness)[0]['data']
        assert set(data) == {
            'task_id', 'site', 'shape', 'reason', 'escalation_ids',
            'ages_secs', 'measured_at', 'store_unavailable', 'streak',
        }
        assert data['task_id'] == 'T1'
        assert data['site'] == 'reconcile_sweep'
        assert data['reason'] == 'escalation_pinned'
        assert data['store_unavailable'] is False
        assert data['streak'] == 1

    async def test_vetoed_payload_names_the_shape_that_was_classified(self, harness):
        _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})

        await harness._reconcile_stranded_in_progress(mid_run=False)

        # Exactly _RECOVERY row (f)'s key, rendered: no live claimant, on main,
        # escalation present, no deploy phase.
        assert _recovery_rows(harness)[0]['data']['shape'] == (
            'in-progress|false|on_main|true|-'
        )

    async def test_vetoed_payload_names_the_pinning_ids(self, harness):
        _bind_reports(harness, {
            'T1': _report(escalations=[_ref('esc-2'), _ref('esc-1')]),
        })

        await harness._reconcile_stranded_in_progress(mid_run=False)

        buckets = _recovery_rows(harness)[0]['data']['escalation_ids']
        flat = sorted(i for ids in buckets.values() for i in ids)
        assert flat == ['esc-1', 'esc-2'], (
            'the operator-visible payoff of this task is naming WHICH records '
            'held the strand'
        )

    async def test_vetoed_payload_ages_each_pinning_id_by_id(self, harness):
        _bind_reports(harness, {
            'T1': _report(escalations=[
                _ref('esc-1', age_secs=300.0), _ref('esc-2', age_secs=90.0),
            ]),
        })

        await harness._reconcile_stranded_in_progress(mid_run=False)

        ages = _recovery_rows(harness)[0]['data']['ages_secs']
        assert set(ages) == {'esc-1', 'esc-2'}, 'ages join by id, never by position'
        assert 250.0 < ages['esc-1'] < 400.0
        assert 60.0 < ages['esc-2'] < 200.0

    async def test_vetoed_payload_stamps_an_iso_utc_measured_at(self, harness):
        _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})

        await harness._reconcile_stranded_in_progress(mid_run=False)

        stamp = datetime.fromisoformat(
            _recovery_rows(harness)[0]['data']['measured_at'],
        )
        assert stamp.tzinfo is not None, 'measured_at must be tz-aware'
        assert abs((datetime.now(UTC) - stamp).total_seconds()) < 120

    async def test_one_pass_emits_once_despite_two_veto_sites(self, harness):
        """The LEAVE chokepoint and the open-escalation early-return both see
        this task in the SAME pass.  Emitting at both would double every
        boundary-#9 row and silently double any consumer's counts."""
        _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})

        await harness._reconcile_stranded_in_progress(mid_run=False)

        assert len(_recovery_rows(harness)) == 1


# ---------------------------------------------------------------------------
# The healthy majority must stay silent, or it buries the strand signal.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestNoStormForHealthyTasks:
    async def test_live_claimant_leave_emits_nothing(self, harness):
        _bind_reports(harness, {'T1': _report(claimant=_live_claimant())})

        await harness._reconcile_stranded_in_progress(mid_run=False)

        assert _recovery_rows(harness) == [], (
            'a running task is the healthy majority of every sweep; emitting '
            'for it would drown the strands this task exists to surface'
        )

    async def test_live_claimant_with_open_escalation_still_emits_nothing(self, harness):
        """Held AND pinned is still just "running" — link 1 of the precedence
        chain outranks the veto for exactly this reason."""
        _bind_reports(harness, {
            'T1': _report(claimant=_live_claimant(), escalations=[_ref('esc-1')]),
        })

        await harness._reconcile_stranded_in_progress(mid_run=False)

        assert _recovery_rows(harness) == []

    async def test_acted_on_tasks_emit_nothing(self, harness):
        """REVERT_TO_PENDING is an action, not a hold — leave_reason returns
        None for it, so no row may claim the task was held."""
        _bind_reports(harness, {
            'T1': _report(branch=BranchStateKind.EXISTS_OFF_MAIN, sha=None),
        })

        await harness._reconcile_stranded_in_progress(mid_run=False)

        assert _recovery_rows(harness) == []


# ---------------------------------------------------------------------------
# Cadence: emission is signature-transition-gated, not one-row-per-observation.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestQuietRepeats:
    async def test_second_sweep_with_identical_signature_adds_no_row(self, harness):
        harness.config.recovery_emission = RecoveryEmissionConfig(
            veto_streak_threshold=3,
        )
        _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})

        await harness._reconcile_stranded_in_progress(mid_run=False)
        await harness._reconcile_stranded_in_progress(mid_run=False)

        assert len(_recovery_rows(harness)) == 1, (
            'an unchanged hold is the SAME fact re-observed; re-emitting it '
            'every sweep is the INV-4 storm this gating exists to prevent'
        )

    async def test_the_threshold_crossing_sweep_speaks_once_more(self, harness):
        """should_emit_event's SECOND clause: the event store must record the
        moment the streak reached the alarm threshold, even though the
        signature has not changed since sweep 1."""
        harness.config.recovery_emission = RecoveryEmissionConfig(
            veto_streak_threshold=3,
        )
        _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})

        for _ in range(5):
            await harness._reconcile_stranded_in_progress(mid_run=False)

        rows = _recovery_rows(harness)
        assert [r['data']['streak'] for r in rows] == [1, 3], (
            'exactly two rows: the new hold, and the threshold crossing — '
            'sweeps 4 and 5 must stay quiet forever after'
        )

    async def test_changed_pinning_id_set_re_emits(self, harness):
        stub = _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})
        await harness._reconcile_stranded_in_progress(mid_run=False)

        # A different record now holds the task — a genuinely new fact.
        stub.reports['T1'] = _report(escalations=[_ref('esc-1'), _ref('esc-9')])
        await harness._reconcile_stranded_in_progress(mid_run=False)

        rows = _recovery_rows(harness)
        assert len(rows) == 2
        second = sorted(
            i for ids in rows[1]['data']['escalation_ids'].values() for i in ids
        )
        assert second == ['esc-1', 'esc-9']

    async def test_repeat_sweeps_still_charge_the_streak(self, harness):
        """Quiet does not mean unobserved: the streak keeps climbing so the L1
        detector can fire even while no event row is written."""
        _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})

        for _ in range(3):
            await harness._reconcile_stranded_in_progress(mid_run=False)

        assert harness._recovery_veto_tracker.streak('reconcile_sweep', 'T1') == 3


# ---------------------------------------------------------------------------
# The store-correctness third state, made visible.
# ---------------------------------------------------------------------------

def _unreadable_store_leave() -> TruthReport:
    """A LEAVE reached while the escalation store could not be READ.

    FAILED is one of the deliberately-unmapped deploy phases, so this shape is
    absent from ``_RECOVERY`` and falls through to LEAVE while carrying an
    empty open_escalations list — the exact shape a store outage produces.
    """
    return _report(
        branch=BranchStateKind.EXISTS_OFF_MAIN, sha=None,
        deploy_phase=DeployPhase.FAILED, store_unavailable=True,
    )


@pytest.mark.asyncio
class TestStoreUnavailableIsNamed:
    async def test_unreadable_store_emits_recovery_left(self, harness):
        _bind_reports(harness, {'T1': _unreadable_store_leave()})

        await harness._reconcile_stranded_in_progress(mid_run=False)

        rows = _recovery_rows(harness)
        assert len(rows) == 1
        assert rows[0]['event_type'] == EventType.recovery_left.value, (
            'nothing VETOED this task — the store simply could not be read, '
            'which is a fall-through, not a hold'
        )

    async def test_unreadable_store_payload_says_so(self, harness):
        _bind_reports(harness, {'T1': _unreadable_store_leave()})

        await harness._reconcile_stranded_in_progress(mid_run=False)

        data = _recovery_rows(harness)[0]['data']
        assert data['reason'] == 'escalation_store_unavailable'
        assert data['store_unavailable'] is True
        assert data['escalation_ids'] == {} or not any(
            data['escalation_ids'].values()
        ), 'an unreadable store yields no ids — never a falsely-empty veto'

    async def test_unreadable_store_does_not_change_the_disposition(self, harness):
        """The flag is EMITTED, never folded into the shape key — a store
        outage must not turn a revert into a hold."""
        _bind_reports(harness, {
            'T1': _report(
                branch=BranchStateKind.EXISTS_OFF_MAIN, sha=None,
                store_unavailable=True,
            ),
        })
        harness._revert_in_progress_if_no_live_claimant = AsyncMock(
            return_value='reverted',
        )

        await harness._reconcile_stranded_in_progress(mid_run=False)

        harness._revert_in_progress_if_no_live_claimant.assert_awaited_once()
        assert _recovery_rows(harness) == [], 'an ACTION is not a hold'


# ---------------------------------------------------------------------------
# THE load-bearing contract.  Emission before behavior: flipping the kill
# switch must change what is OBSERVED and nothing else.
# ---------------------------------------------------------------------------

def _matrix_rows() -> dict[str, tuple[TruthReport, RecoveryAction]]:
    """One report per recovery action, each asserted to classify as named."""
    rows = {
        'leave_pinned': (
            _report(escalations=[_ref('esc-1')]), RecoveryAction.LEAVE,
        ),
        'revert_to_pending': (
            _report(branch=BranchStateKind.EXISTS_OFF_MAIN, sha=None),
            RecoveryAction.REVERT_TO_PENDING,
        ),
        'mark_done_with_provenance': (
            _report(), RecoveryAction.MARK_DONE_WITH_PROVENANCE,
        ),
        're_file_escalation': (
            _report(
                branch=BranchStateKind.GONE_NO_MARKER, sha=None,
                deploy_phase=DeployPhase.RAN,
            ),
            RecoveryAction.RE_FILE_ESCALATION,
        ),
    }
    for name, (report, expected) in rows.items():
        assert classify_recovery(report) is expected, (
            f'matrix row {name!r} no longer classifies {expected} — the row is '
            f'stale, not the emission code'
        )
    return rows


@pytest.mark.asyncio
class TestZeroDispositionChange:
    @staticmethod
    async def _run(harness, report: TruthReport) -> dict:
        """Run one sweep and return every write the harness performed.

        The two APPLIERS are spied rather than executed: this test is about
        which disposition was reached, and running the real appliers would drag
        in git archaeology that has nothing to do with emission.
        """
        _bind_reports(harness, {'T1': report})
        harness._revert_in_progress_if_no_live_claimant = AsyncMock(
            return_value='reverted',
        )
        harness._mark_in_progress_done = AsyncMock(return_value=True)
        harness._delivered_checks_block = AsyncMock(return_value=None)

        with patch(
            'orchestrator.harness.validate_landing_evidence',
            AsyncMock(return_value=MagicMock(accepted=True)),
        ):
            result = await harness._reconcile_stranded_in_progress(mid_run=False)

        return {
            'result': result,
            'reverts': harness._revert_in_progress_if_no_live_claimant.await_count,
            'mark_dones': harness._mark_in_progress_done.await_count,
            'status_writes': [
                (c.args, sorted(c.kwargs))
                for c in harness.scheduler.set_task_status.await_args_list
            ],
            'escalations': sorted(
                e.id for e in harness._escalation_queue.get_by_task(
                    'T1', status='pending',
                )
            ),
        }

    @pytest.mark.parametrize('row_name', list(_matrix_rows()))
    async def test_writes_are_identical_with_emission_on_and_off(self, harness, row_name):
        report, _expected = _matrix_rows()[row_name]

        harness.config.recovery_emission = RecoveryEmissionConfig(enabled=True)
        with_emission = await TestZeroDispositionChange._run(harness, report)

        # Fresh recorders + a fresh tracker, so run two is not influenced by
        # run one's bookkeeping.
        harness.scheduler.set_task_status = AsyncMock()
        harness._recovery_veto_tracker.clear('reconcile_sweep', 'T1')
        harness.config.recovery_emission = RecoveryEmissionConfig(enabled=False)
        without_emission = await TestZeroDispositionChange._run(harness, report)

        assert with_emission == without_emission, (
            f'row {row_name!r}: emission moved a disposition — this is the one '
            f'failure mode task 3535 may not have'
        )

    async def test_disabled_emission_writes_no_rows(self, harness):
        harness.config.recovery_emission = RecoveryEmissionConfig(enabled=False)
        _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})

        await harness._reconcile_stranded_in_progress(mid_run=False)

        assert _recovery_rows(harness) == [], (
            'the kill switch must actually silence the detector'
        )

    async def test_emission_never_files_against_the_real_task(self, harness):
        """A record on the real task id would be read by every veto predicate —
        an observability signal mutating into a disposition change."""
        _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})

        await harness._reconcile_stranded_in_progress(mid_run=False)

        assert harness._escalation_queue.get_by_task('T1', status='pending') == []

    async def test_a_broken_event_store_never_disturbs_the_sweep(self, harness):
        """Telemetry is fire-and-forget: a store fault must not strand a task
        the sweep would otherwise have recovered."""
        harness.event_store = MagicMock()
        harness.event_store.emit.side_effect = OSError('disk full')
        _bind_reports(harness, {
            'T1': _report(branch=BranchStateKind.EXISTS_OFF_MAIN, sha=None),
        })
        harness._revert_in_progress_if_no_live_claimant = AsyncMock(
            return_value='reverted',
        )

        result = await harness._reconcile_stranded_in_progress(mid_run=False)

        assert result == 1
        harness._revert_in_progress_if_no_live_claimant.assert_awaited_once()


# ---------------------------------------------------------------------------
# BOUNDARY #17 — the N-consecutive-vetoes alarm, end to end.
#
# The streak L1 is the only part of task 3535 that WRITES to the escalation
# queue, so it is also the only part that could break the zero-behavior-change
# contract.  It writes against a SENTINEL id for exactly that reason: a record
# on the real task id would be read back by every veto predicate — including
# the one that observed the hold — turning an observability signal into a
# disposition change.  ``test_the_alarm_never_touches_the_subject_task`` is the
# load-bearing assertion of this whole section.
# ---------------------------------------------------------------------------

class _FakeClock:
    """A monotonic clock the sweep cadence can be driven by, deterministically.

    The alarm predicate is two-dimensional (streak AND elapsed span), so a test
    that could not control time could only ever exercise one half of it.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, secs: float) -> None:
        self.now += secs


#: The production sweep interval both charging sites run at
#: (``stranded_reconcile_interval_secs`` / ``deterministic_recon_sweep_interval_secs``).
_SWEEP_INTERVAL = 900.0


@pytest.fixture
def clocked(harness):
    """(harness, clock) whose veto tracker rides an injected monotonic clock."""
    clock = _FakeClock()
    harness._recovery_veto_tracker = RecoveryVetoStreakTracker(clock=clock)
    return harness, clock


async def _sweeps(harness, clock, count: int, *, interval: float = _SWEEP_INTERVAL):
    """Run *count* sweeps *interval* seconds apart, as the real loop would."""
    for i in range(count):
        if i:
            clock.advance(interval)
        await harness._reconcile_stranded_in_progress(mid_run=False)


def _alarms(harness, tid: str = 'T1') -> list:
    """Pending veto-streak alarms filed against *tid*'s SENTINEL id."""
    return harness._escalation_queue.get_by_task(
        f'{RECOVERY_VETO_STREAK_SENTINEL_PREFIX}{tid}', status='pending',
    )


@pytest.mark.asyncio
class TestVetoStreakAlarm:
    async def test_the_first_two_sweeps_file_nothing(self, clocked):
        """Two sweeps is contention, not an incident."""
        harness, clock = clocked
        _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})

        await _sweeps(harness, clock, 2)

        assert _alarms(harness) == []

    async def test_the_threshold_crossing_sweep_files_exactly_one_l1(self, clocked):
        harness, clock = clocked
        _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})

        await _sweeps(harness, clock, 3)

        alarms = _alarms(harness)
        assert len(alarms) == 1, f'expected one alarm, got {alarms}'
        assert alarms[0].level == 1, 'the streak alarm is an L1, not a page'
        assert alarms[0].severity == 'blocking'

    async def test_further_sweeps_never_file_a_second(self, clocked):
        """Boundary #17, "no storm": a hold that lasts a week is ONE alarm."""
        harness, clock = clocked
        _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})

        await _sweeps(harness, clock, 8)

        assert len(_alarms(harness)) == 1

    async def test_the_alarm_names_the_task_the_site_and_the_hold(self, clocked):
        """An operator must be able to act on the alarm without a second query."""
        harness, clock = clocked
        _bind_reports(harness, {
            'T1': _report(escalations=[_ref('esc-1', age_secs=4200.0)]),
        })

        await _sweeps(harness, clock, 3)

        body = ' '.join((_alarms(harness)[0].summary, _alarms(harness)[0].detail))
        assert 'T1' in body, 'the REAL task id, not just the sentinel'
        assert 'reconcile_sweep' in body, 'which site observed the streak'
        assert 'esc-1' in body, 'which record has been holding it'
        assert '3' in body, 'the streak that tripped the alarm'
        assert 'escalation_pinned' in body

    async def test_the_alarm_never_touches_the_subject_task(self, clocked):
        """THE zero-behavior-change assertion of this section.

        A record filed on the real id would be returned by
        ``_resolve_open_escalations`` on the very next sweep, deepening the
        hold that filing it merely observed.
        """
        harness, clock = clocked
        report = _report(escalations=[_ref('esc-1')])
        _bind_reports(harness, {'T1': report})

        await _sweeps(harness, clock, 5)

        assert len(_alarms(harness)) == 1, 'precondition: the alarm did fire'
        assert harness._escalation_queue.get_by_task('T1', status='pending') == [], (
            'the subject task\'s own pending set must be untouched by the alarm'
        )
        assert classify_recovery(report) is RecoveryAction.LEAVE, (
            'and its classification must be exactly what it was before'
        )
        assert harness.scheduler.set_task_status.await_count == 0
        assert harness.scheduler.mark_done.await_count == 0

    async def test_the_crossing_sweep_also_writes_its_event_row(self, clocked):
        """The event store must record the MOMENT the alarm was filed, even
        though the veto signature has not changed since sweep 1 — otherwise the
        L1 and the event stream disagree about when the hold became an
        incident."""
        harness, clock = clocked
        _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})

        await _sweeps(harness, clock, 3)

        assert [r['data']['streak'] for r in _recovery_rows(harness)] == [1, 3]
        assert len(_alarms(harness)) == 1

    async def test_a_changed_signature_restarts_the_streak(self, clocked):
        """An alarm may only ever describe N genuinely IDENTICAL vetoes.

        Four consecutive held sweeps, but the pinning set changed midway, so
        the longest identical run is two — no alarm.
        """
        harness, clock = clocked
        stub = _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})

        await _sweeps(harness, clock, 2)
        stub.reports['T1'] = _report(escalations=[_ref('esc-9')])
        await _sweeps(harness, clock, 2)

        assert _alarms(harness) == []
        assert harness._recovery_veto_tracker.streak('reconcile_sweep', 'T1') == 2

    async def test_a_fast_streak_alone_does_not_alarm(self, clocked):
        """The SECOND dimension.  Three sweeps seconds apart is a retry loop or
        a test fixture, not a task stranded for half an hour — and a per-tick
        site that ever charged this counter must not be able to page anyone
        seconds after a strand appears."""
        harness, clock = clocked
        _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})

        await _sweeps(harness, clock, 6, interval=1.0)

        assert _alarms(harness) == [], (
            'streak >= threshold is necessary but not sufficient'
        )

    async def test_the_threshold_is_honoured_from_config(self, clocked):
        """The crossing moves when an operator retunes the (green-tier) knob."""
        harness, clock = clocked
        harness.config.recovery_emission = RecoveryEmissionConfig(
            veto_streak_threshold=2,
        )
        _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})

        await _sweeps(harness, clock, 1, interval=1600.0)
        assert _alarms(harness) == []

        await _sweeps(harness, clock, 2, interval=1600.0)
        assert len(_alarms(harness)) == 1

    async def test_the_kill_switch_stops_the_l1_but_not_the_events(self, clocked):
        """``streak_escalation_enabled`` is the narrow switch for the only part
        that writes to the queue — silencing it must not blind the sweep."""
        harness, clock = clocked
        harness.config.recovery_emission = RecoveryEmissionConfig(
            streak_escalation_enabled=False,
        )
        _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})

        await _sweeps(harness, clock, 5)

        assert _alarms(harness) == []
        assert [r['data']['streak'] for r in _recovery_rows(harness)] == [1, 3], (
            'the events are a separate, cheaper signal and keep flowing'
        )


@pytest.mark.asyncio
class TestVetoStreakAlarmRecovery:
    """The resolve half is not polish: without it the ``has_open_l1`` dedup
    would silence this detector for that task for the life of the queue."""

    async def test_the_alarm_auto_resolves_when_the_veto_stops(self, clocked):
        harness, clock = clocked
        stub = _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})
        await _sweeps(harness, clock, 3)
        assert len(_alarms(harness)) == 1, 'precondition: the alarm did fire'

        # The operator resolves the pinning record; the next sweep finds the
        # task recoverable and acts on it as it always would have.
        stub.reports['T1'] = _report(
            branch=BranchStateKind.EXISTS_OFF_MAIN, sha=None,
        )
        harness._revert_in_progress_if_no_live_claimant = AsyncMock(
            return_value='reverted',
        )
        await _sweeps(harness, clock, 1)

        assert _alarms(harness) == [], 'a resolved condition must not hold a blocker'
        assert 'T1' not in harness._recovery_streak_filed_at, (
            'the memo must be dropped with it, or a later recurrence would be '
            'short-circuited before it ever reached the queue'
        )
        assert harness._recovery_veto_tracker.streak('reconcile_sweep', 'T1') == 0, (
            'the tracker entry is POPPED, so the dict stays proportional to '
            'currently-held tasks over a weeks-long run'
        )

    async def test_a_later_recurrence_files_again(self, clocked):
        """The whole point of the resolve half."""
        harness, clock = clocked
        stub = _bind_reports(harness, {'T1': _report(escalations=[_ref('esc-1')])})
        await _sweeps(harness, clock, 3)
        stub.reports['T1'] = _report(branch=BranchStateKind.EXISTS_OFF_MAIN, sha=None)
        harness._revert_in_progress_if_no_live_claimant = AsyncMock(
            return_value='reverted',
        )
        await _sweeps(harness, clock, 1)
        assert _alarms(harness) == [], 'precondition: the first alarm resolved'

        stub.reports['T1'] = _report(escalations=[_ref('esc-1')])
        await _sweeps(harness, clock, 3)

        assert len(_alarms(harness)) == 1, (
            'a task that strands again must alarm again — a detector that '
            'fires once per task per queue lifetime is not a detector'
        )

    async def test_a_healthy_task_never_pays_for_the_resolve_check(self, clocked):
        """The release path runs for EVERY swept task, so it must not put a
        pending-queue scan on the healthy path."""
        harness, clock = clocked
        _bind_reports(harness, {'T1': _report(claimant=_live_claimant())})
        harness._escalation_queue = MagicMock(wraps=harness._escalation_queue)

        await _sweeps(harness, clock, 3)

        assert harness._escalation_queue.has_open_l1.call_count == 0
        assert harness._escalation_queue.submit.call_count == 0


@pytest.mark.asyncio
class TestPerTickSitesNeverCharge:
    """Only the SWEEP-frequency sites charge the counter.

    The dispatch gate and the scheduler's blocked-redispatch phase run per
    dispatch TICK — seconds apart, not 900s apart — so charging them would file
    a blocking L1 within seconds of a hold appearing.  The span dimension is
    the belt-and-braces backstop; this is the primary guard.
    """

    async def test_hammering_the_dispatch_gate_never_files_an_l1(self, harness):
        harness._escalation_queue.submit(Escalation(
            id='esc-T1-1', task_id='T1', agent_role='implementer',
            severity='blocking', category='design_concern',
            summary='an open handoff on task T1', level=2,
        ))
        harness.scheduler.get_task = AsyncMock(
            return_value={'id': 'T1', 'metadata': {}},
        )
        harness._mark_in_progress_done = AsyncMock()

        for _ in range(30):
            assert await harness._already_landed_dispatch_gate('T1') is False

        assert _alarms(harness) == [], (
            'a per-tick site charging this counter would page an operator '
            'seconds after a hold appeared, not after three 900s sweeps'
        )
