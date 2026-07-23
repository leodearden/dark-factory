"""Tests for the ζ growth-freshness sweep + streak escalation (task 2899).

Covers the Stage-2-tail growth sweep helper
(:func:`_sweep_entity_standing_decision_growth`), the pure streak helpers, the
module-level escalation filer, and the ``TaskKnowledgeSync`` orchestrator method
that wires them into the Stage-2 tail.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore
from fused_memory.reconciliation.standing_decision_constants import (
    EXPIRY_REASON_GROWTH,
    GROUNDS_STRUCTURAL_SIZE_CONFLATION,
    RECORD_KIND_ENTITY_STANDING_DECISION,
    STATE_ACTIVE,
    STATE_EXPIRED,
)
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    ENTITY_STANDING_DECISION_GROWTH_SWEEP_ESCALATION_CATEGORY,
    ENTITY_STANDING_DECISION_GROWTH_SWEEP_FAILED_STAT_KEY,
    _compute_growth_sweep_failure_streak,
    _evaluate_growth_sweep_escalation,
    _extract_growth_sweep_failed,
    _maybe_escalate_growth_sweep_failures,
    _sweep_entity_standing_decision_growth,
)

_PROJECT_ID = 'dark_factory'
_DECIDED_AT = '2026-07-01T00:00:00+00:00'
_EXPIRES_AT = '2026-09-29T00:00:00+00:00'


@pytest_asyncio.fixture
async def ledger(tmp_path):
    """A real initialized ReconLedgerStore on tmp_path."""
    store = ReconLedgerStore(tmp_path / 'reconciliation.db')
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


async def _seed(store, *, entity_uuid, edge_count, state=STATE_ACTIVE, extra_payload=None):
    """Seed one entity_standing_decision row (optionally with extra payload keys).

    Uses the low-level ``upsert`` so a per-record ``growth_factor`` /
    ``growth_abs_delta`` override can be injected into ``payload_json`` — the
    row-schema override path α's upsert helper does not itself write.
    """
    payload = {
        'grounds': GROUNDS_STRUCTURAL_SIZE_CONFLATION,
        'decided_at': _DECIDED_AT,
        'edge_count_at_decision': edge_count,
        'evidence': [],
    }
    if extra_payload:
        payload.update(extra_payload)
    await store.upsert(
        ReconLedgerRecord(
            project_id=_PROJECT_ID,
            record_kind=RECORD_KIND_ENTITY_STANDING_DECISION,
            task_id='',
            flag_type=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
            run_id=entity_uuid,
            entity_uuid=entity_uuid,
            payload_json=json.dumps(payload),
            state=state,
            created_at=_DECIDED_AT,
            expires_at=_EXPIRES_AT,
        )
    )


def _svc(store, *, edges=0, raises=False):
    """MagicMock memory_service: real ledger + AsyncMock graphiti edge fetch."""
    service = MagicMock()
    service.recon_ledger = store
    if raises:
        service.graphiti.get_valid_edges_for_node = AsyncMock(
            side_effect=RuntimeError('graphiti down')
        )
    else:
        service.graphiti.get_valid_edges_for_node = AsyncMock(
            return_value=[{'uuid': f'e{i}'} for i in range(edges)]
        )
    return service


async def _states(store):
    active = await store.list_entity_standing_decisions(_PROJECT_ID, state=STATE_ACTIVE)
    expired = await store.list_entity_standing_decisions(_PROJECT_ID, state=STATE_EXPIRED)
    return active, expired


class TestGrowthSweep:
    """Closed-form growth thresholds over seeded decision count + mocked live edges."""

    @pytest.mark.asyncio
    async def test_factor_rule_trips(self, ledger):
        """decision=10, live=13 → 13 > 12.5 (factor rule) ⇒ expired/growth."""
        await _seed(ledger, entity_uuid='u-a', edge_count=10)
        service = _svc(ledger, edges=13)
        result = await _sweep_entity_standing_decision_growth(service, _PROJECT_ID, 'run-1')
        assert result['checked'] == 1
        assert result['expired'] == 1
        assert result['failed'] is False
        active, expired = await _states(ledger)
        assert active == []
        assert len(expired) == 1
        assert json.loads(expired[0].payload_json)['expiry_reason'] == EXPIRY_REASON_GROWTH
        service.graphiti.get_valid_edges_for_node.assert_called_once_with(
            'u-a', group_id=_PROJECT_ID
        )

    @pytest.mark.asyncio
    async def test_abs_rule_trips(self, ledger):
        """decision=100, live=116 → 116 ≥ 115 (abs rule; 116 < 125 factor) ⇒ growth."""
        await _seed(ledger, entity_uuid='u-b', edge_count=100)
        service = _svc(ledger, edges=116)
        result = await _sweep_entity_standing_decision_growth(service, _PROJECT_ID, 'run-1')
        assert result['expired'] == 1
        assert result['failed'] is False
        active, expired = await _states(ledger)
        assert active == []
        assert json.loads(expired[0].payload_json)['expiry_reason'] == EXPIRY_REASON_GROWTH

    @pytest.mark.asyncio
    async def test_neither_rule_keeps_active(self, ledger):
        """decision=10, live=12 → 12 ≤ 12.5 and 12 < 25 ⇒ row stays active."""
        await _seed(ledger, entity_uuid='u-c', edge_count=10)
        service = _svc(ledger, edges=12)
        result = await _sweep_entity_standing_decision_growth(service, _PROJECT_ID, 'run-1')
        assert result['expired'] == 0
        assert result['failed'] is False
        active, expired = await _states(ledger)
        assert len(active) == 1
        assert expired == []

    @pytest.mark.asyncio
    async def test_per_record_factor_override_trips(self, ledger):
        """A payload growth_factor=1.1 override trips (12 > 11) where the default
        1.25 would keep (12 ≤ 12.5)."""
        await _seed(
            ledger, entity_uuid='u-d', edge_count=10, extra_payload={'growth_factor': 1.1}
        )
        service = _svc(ledger, edges=12)
        result = await _sweep_entity_standing_decision_growth(service, _PROJECT_ID, 'run-1')
        assert result['expired'] == 1
        active, _ = await _states(ledger)
        assert active == []

    @pytest.mark.asyncio
    async def test_graphiti_error_leaves_row_active_failed_true(self, ledger):
        """A per-row Graphiti error ⇒ row stays active (fail-safe) and failed=True."""
        await _seed(ledger, entity_uuid='u-e', edge_count=10)
        service = _svc(ledger, raises=True)
        result = await _sweep_entity_standing_decision_growth(service, _PROJECT_ID, 'run-1')
        assert result['failed'] is True
        assert result['expired'] == 0
        active, expired = await _states(ledger)
        assert len(active) == 1
        assert expired == []

    @pytest.mark.asyncio
    async def test_none_ledger_noop(self):
        """recon_ledger None ⇒ {checked:0, expired:0, failed:False}, never raises."""
        service = MagicMock()
        service.recon_ledger = None
        result = await _sweep_entity_standing_decision_growth(service, _PROJECT_ID, 'run-1')
        assert result == {'checked': 0, 'expired': 0, 'failed': False}

    @pytest.mark.asyncio
    async def test_expired_row_not_swept(self, ledger):
        """Only ACTIVE rows are swept — an already-expired row is ignored (no
        graphiti fetch, not re-flipped)."""
        await _seed(ledger, entity_uuid='u-g', edge_count=10, state=STATE_EXPIRED)
        service = _svc(ledger, edges=100)  # would trip if it were swept
        result = await _sweep_entity_standing_decision_growth(service, _PROJECT_ID, 'run-1')
        assert result['checked'] == 0
        assert result['expired'] == 0
        service.graphiti.get_valid_edges_for_node.assert_not_called()
        active, expired = await _states(ledger)
        assert active == []
        assert len(expired) == 1


class TestGrowthSweepStreakPure:
    """The pure streak helpers (streak-on-FAILURE, mirroring the snapshot cadence
    inverted: the counted flag is True=failed, not False=miss)."""

    def test_extract_failed_flag_true(self):
        report = SimpleNamespace(
            stats={ENTITY_STANDING_DECISION_GROWTH_SWEEP_FAILED_STAT_KEY: 1}
        )
        assert _extract_growth_sweep_failed(report) is True

    def test_extract_failed_flag_false(self):
        report = SimpleNamespace(
            stats={ENTITY_STANDING_DECISION_GROWTH_SWEEP_FAILED_STAT_KEY: 0}
        )
        assert _extract_growth_sweep_failed(report) is False

    def test_extract_accepts_dict_shape(self):
        report = {'stats': {ENTITY_STANDING_DECISION_GROWTH_SWEEP_FAILED_STAT_KEY: 1}}
        assert _extract_growth_sweep_failed(report) is True

    def test_extract_absent_key_is_none(self):
        assert _extract_growth_sweep_failed(SimpleNamespace(stats={})) is None

    def test_extract_none_report_is_none(self):
        assert _extract_growth_sweep_failed(None) is None

    def test_compute_streak_stops_at_reset(self):
        # Leading run of True (failures), stops at the first False (a success
        # cycle resets the streak).
        assert _compute_growth_sweep_failure_streak([True, True, False]) == 2

    def test_compute_streak_stops_at_unknown(self):
        # None (inconclusive/fail-safe) also stops the run — never counted.
        assert _compute_growth_sweep_failure_streak([True, None, True]) == 1

    def test_compute_streak_leading_false_is_zero(self):
        assert _compute_growth_sweep_failure_streak([False, True]) == 0

    def test_compute_streak_empty_is_zero(self):
        assert _compute_growth_sweep_failure_streak([]) == 0

    def test_evaluate_current_not_failed_short_circuits(self):
        # current_failed False or None ⇒ never escalate (only a CONFIRMED
        # current failure can trigger).
        assert _evaluate_growth_sweep_escalation(False, [True, True]) == {
            'streak': 0, 'escalate': False,
        }
        assert _evaluate_growth_sweep_escalation(None, [True, True]) == {
            'streak': 0, 'escalate': False,
        }

    def test_evaluate_reaches_threshold(self):
        # current failure + two prior failures ⇒ streak 3 ⇒ escalate.
        assert _evaluate_growth_sweep_escalation(True, [True, True]) == {
            'streak': 3, 'escalate': True,
        }

    def test_evaluate_below_threshold(self):
        # current failure + one prior failure ⇒ streak 2 ⇒ no escalation yet.
        assert _evaluate_growth_sweep_escalation(True, [True]) == {
            'streak': 2, 'escalate': False,
        }


# ---------------------------------------------------------------------------
# _maybe_escalate_growth_sweep_failures — journal-recomputed streak + filing
# ---------------------------------------------------------------------------

_ESC_KEY = f'recon-esd-growth-sweep:{_PROJECT_ID}'


def _run(*, run_id, started, failed=True, run_type='full', status='completed'):
    """A ReconciliationRun-like double (duck-typed, attribute-access shape)."""
    stats = {ENTITY_STANDING_DECISION_GROWTH_SWEEP_FAILED_STAT_KEY: (1 if failed else 0)}
    return SimpleNamespace(
        id=run_id,
        run_type=run_type,
        status=status,
        started_at=started,
        stage_reports={'task_knowledge_sync': SimpleNamespace(stats=stats)},
    )


class _FakeJournal:
    def __init__(self, runs):
        self._runs = runs
        self.calls: list = []

    async def get_recent_runs(self, project_id, limit=None):
        self.calls.append((project_id, limit))
        return list(self._runs)


def _queue(*, has_open=False):
    q = MagicMock()
    q.has_open_l1 = MagicMock(return_value=has_open)
    q.make_id = MagicMock(return_value='esc-generated-id')
    q.submit = MagicMock()
    return q


class TestGrowthSweepEscalationFiler:
    """The module-level filer: streak recomputed from journalled full+completed
    prior runs (excluding the current run + running/remediation runs), deduped
    on a stable per-project synthetic key."""

    @pytest.mark.asyncio
    async def test_confirmed_streak_files_one_escalation(self):
        """Two prior failed full+completed runs + a confirmed current failure ⇒
        streak 3 ⇒ exactly one level-1 Escalation with the ζ category. Noise runs
        (the current run_id, a remediation run, a running run) — all more recent
        and 'failed' — are EXCLUDED, so the streak is exactly 3."""
        runs = [
            # Noise that must be excluded — all more recent than the genuine priors.
            _run(run_id='run-current', started=datetime(2026, 7, 10, tzinfo=UTC)),
            _run(run_id='run-rem', started=datetime(2026, 7, 9, tzinfo=UTC),
                 run_type='remediation'),
            _run(run_id='run-running', started=datetime(2026, 7, 8, tzinfo=UTC),
                 status='running'),
            # Genuine prior failures (most-recent-first when filtered).
            _run(run_id='run-p1', started=datetime(2026, 7, 3, tzinfo=UTC)),
            _run(run_id='run-p2', started=datetime(2026, 7, 2, tzinfo=UTC)),
        ]
        journal = _FakeJournal(runs)
        queue = _queue(has_open=False)

        filed = await _maybe_escalate_growth_sweep_failures(
            queue, journal, _PROJECT_ID, 'run-current', True,
        )
        assert filed is True
        queue.has_open_l1.assert_called_once_with(_ESC_KEY)
        queue.make_id.assert_called_once_with(_ESC_KEY)
        queue.submit.assert_called_once()
        esc = queue.submit.call_args.args[0]
        assert esc.task_id == _ESC_KEY
        assert esc.category == ENTITY_STANDING_DECISION_GROWTH_SWEEP_ESCALATION_CATEGORY
        assert esc.level == 1
        assert esc.agent_role == 'reconciliation-stage2'
        # Streak is exactly 3 (the two genuine priors + current) — proves the
        # noise runs were excluded.
        assert '3' in esc.summary

    @pytest.mark.asyncio
    async def test_open_l1_dedup_suppresses_submit(self):
        """An already-open level-1 escalation on the key ⇒ no new submit."""
        runs = [
            _run(run_id='run-p1', started=datetime(2026, 7, 3, tzinfo=UTC)),
            _run(run_id='run-p2', started=datetime(2026, 7, 2, tzinfo=UTC)),
        ]
        queue = _queue(has_open=True)
        filed = await _maybe_escalate_growth_sweep_failures(
            queue, _FakeJournal(runs), _PROJECT_ID, 'run-current', True,
        )
        assert filed is False
        queue.submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_current_not_failed_no_submit(self):
        """current_failed=False ⇒ no escalation regardless of prior history."""
        runs = [
            _run(run_id='run-p1', started=datetime(2026, 7, 3, tzinfo=UTC)),
            _run(run_id='run-p2', started=datetime(2026, 7, 2, tzinfo=UTC)),
        ]
        queue = _queue()
        filed = await _maybe_escalate_growth_sweep_failures(
            queue, _FakeJournal(runs), _PROJECT_ID, 'run-current', False,
        )
        assert filed is False
        queue.submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_queue_is_noop(self):
        """A None escalation_queue ⇒ no-op, returns False, never raises."""
        filed = await _maybe_escalate_growth_sweep_failures(
            None, _FakeJournal([]), _PROJECT_ID, 'run-current', True,
        )
        assert filed is False
