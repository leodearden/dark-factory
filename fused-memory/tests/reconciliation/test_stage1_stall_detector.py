"""Tests for stage1_stall_detector module — task 1201.

Covers:
- TestExtractHumanOperatorTaskIds  (step-1/2)
- TestTrackHumanOperatorStalls     (step-3/4)
- TestComputeStalledTaskIds        (step-5/6)
- TestMaybeEscalateStalledTasks    (step-7/8)

Gate-backlog age check (task 3017):
- TestGateBacklogConstants
- TestGateEscalatedAgeSecs
- TestExtractStalledGateBacklogTaskIds
- TestMaybeEscalateStalledGateBacklog
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.reconciliation.stage1_stall_detector import (
    _GATE_BACKLOG_ESCALATION_CATEGORY,
    _STAGE1_HUMAN_OPERATOR_STALL_MARKER_SOURCE,
    STAGE1_GATE_BACKLOG_STALL_THRESHOLD_SECS,
    STAGE1_HUMAN_OPERATOR_STALL_THRESHOLD,
    compute_stalled_task_ids,
    extract_human_operator_task_ids,
    extract_stalled_gate_backlog_task_ids,
    gate_escalated_age_secs,
    maybe_escalate_stalled_gate_backlog,
    maybe_escalate_stalled_tasks,
    track_human_operator_stalls,
)

# A fixed, timezone-aware "now" for the gate-backlog age tests (task 3017).
# gate_escalated_at values are built relative to this instant so the age
# assertions are deterministic regardless of wall-clock time.
_GATE_NOW = datetime(2026, 7, 24, 12, 0, 0, tzinfo=UTC)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Assert module-level constants have expected values."""

    def test_threshold_is_five(self):
        assert STAGE1_HUMAN_OPERATOR_STALL_THRESHOLD == 5

    def test_marker_source_value(self):
        assert _STAGE1_HUMAN_OPERATOR_STALL_MARKER_SOURCE == 'stage1_human_operator_stall_marker'


# ---------------------------------------------------------------------------
# extract_human_operator_task_ids
# ---------------------------------------------------------------------------


class TestExtractHumanOperatorTaskIds:
    """extract_human_operator_task_ids filters and deduplicates task_ids."""

    def test_returns_task_ids_of_hor_flags(self):
        """(a) Returns task_ids where resolution_status == 'human_operator_required'."""
        flags = [
            {'task_id': '42', 'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required'},
        ]
        result = extract_human_operator_task_ids(flags)
        assert result == ['42']

    def test_skips_other_resolution_statuses(self):
        """(b) Flags with other resolution_status values are skipped."""
        flags = [
            {'task_id': '1', 'resolution_status': 'automated'},
            {'task_id': '2', 'resolution_status': 'agent_retry'},
            {'task_id': '3', 'resolution_status': 'human_operator_required'},
        ]
        result = extract_human_operator_task_ids(flags)
        assert result == ['3']

    def test_skips_flags_without_resolution_status(self):
        """(c) Flags missing the resolution_status key are skipped."""
        flags = [
            {'task_id': '7', 'flag_type': 'missing_deliverable'},
            {'task_id': '8', 'resolution_status': 'human_operator_required'},
        ]
        result = extract_human_operator_task_ids(flags)
        assert result == ['8']

    def test_deduplicates_same_task_id_preserving_first_seen_order(self):
        """(d) Multiple flags for the same task_id deduplicate, preserving first-seen order."""
        flags = [
            {'task_id': '5', 'flag_type': 'A', 'resolution_status': 'human_operator_required'},
            {'task_id': '3', 'flag_type': 'B', 'resolution_status': 'human_operator_required'},
            {'task_id': '5', 'flag_type': 'C', 'resolution_status': 'human_operator_required'},
        ]
        result = extract_human_operator_task_ids(flags)
        # '5' seen first, '3' second; '5' not duplicated
        assert result == ['5', '3']

    def test_coerces_int_task_id_to_str(self):
        """(e) Int task_id is coerced to str; int 0 and str '0' collapse."""
        flags = [
            {'task_id': 0, 'resolution_status': 'human_operator_required'},
            {'task_id': '0', 'resolution_status': 'human_operator_required'},
        ]
        result = extract_human_operator_task_ids(flags)
        assert result == ['0']

    def test_ignores_flags_with_none_task_id(self):
        """(f) Flags with task_id=None are skipped."""
        flags = [
            {'task_id': None, 'resolution_status': 'human_operator_required'},
            {'task_id': '99', 'resolution_status': 'human_operator_required'},
        ]
        result = extract_human_operator_task_ids(flags)
        assert result == ['99']

    def test_ignores_flags_with_missing_task_id(self):
        """(f) Flags missing the task_id key entirely are skipped."""
        flags = [
            {'resolution_status': 'human_operator_required'},
            {'task_id': '10', 'resolution_status': 'human_operator_required'},
        ]
        result = extract_human_operator_task_ids(flags)
        assert result == ['10']

    def test_empty_input_returns_empty_list(self):
        """(g) Empty input returns []."""
        assert extract_human_operator_task_ids([]) == []

    def test_multiple_hor_flags_in_order(self):
        """Multiple distinct HOR task_ids are returned in input order."""
        flags = [
            {'task_id': 'b', 'resolution_status': 'human_operator_required'},
            {'task_id': 'a', 'resolution_status': 'human_operator_required'},
            {'task_id': 'c', 'resolution_status': 'human_operator_required'},
        ]
        result = extract_human_operator_task_ids(flags)
        assert result == ['b', 'a', 'c']


# ---------------------------------------------------------------------------
# track_human_operator_stalls
# ---------------------------------------------------------------------------


class TestTrackHumanOperatorStalls:
    """track_human_operator_stalls counts and writes Mem0 stall markers."""

    @pytest.mark.asyncio
    async def test_happy_path_single_task(self):
        """(a) prior_count=2 → returns {'42': 3}; calls count then add_memory with correct args."""
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(return_value=2)
        memory_service.add_memory = AsyncMock(return_value={'memory_ids': ['m1']})

        result = await track_human_operator_stalls(
            memory_service=memory_service,
            project_id='proj',
            run_id='run-001',
            task_ids=['42'],
        )

        assert result == {'42': 3}

        # count called with correct filters
        memory_service.count_memories_by_metadata.assert_awaited_once_with(
            project_id='proj',
            filters={
                'source': 'stage1_human_operator_stall_marker',
                'task_id': '42',
            },
        )

        # add_memory called with correct payload
        memory_service.add_memory.assert_awaited_once()
        add_call = memory_service.add_memory.await_args
        assert add_call is not None
        assert add_call.kwargs.get('metadata', {}).get('source') == 'stage1_human_operator_stall_marker'
        assert add_call.kwargs.get('metadata', {}).get('task_id') == '42'
        assert add_call.kwargs.get('metadata', {}).get('run_id') == 'run-001'

        # search must NOT have been awaited (deterministic count only)
        memory_service.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_count_failure_treats_prior_as_zero(self):
        """(b) count failure → returns {'42': 1} (prior=0) and logs WARNING."""
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(
            side_effect=RuntimeError('Qdrant timeout')
        )
        memory_service.add_memory = AsyncMock(return_value={'memory_ids': ['m2']})

        result = await track_human_operator_stalls(
            memory_service=memory_service,
            project_id='proj',
            run_id='run-002',
            task_ids=['42'],
        )

        # Should gracefully default prior=0 and still return count=1
        assert result == {'42': 1}
        # add_memory should still be called (write phase continues)
        memory_service.add_memory.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_memory_failure_still_returns_count(self):
        """(c) add_memory failure → still returns prior+1, logs WARNING."""
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(return_value=4)
        memory_service.add_memory = AsyncMock(side_effect=RuntimeError('Qdrant write failed'))

        result = await track_human_operator_stalls(
            memory_service=memory_service,
            project_id='proj',
            run_id='run-003',
            task_ids=['42'],
        )

        # count still returned despite write failure
        assert result == {'42': 5}

    @pytest.mark.asyncio
    async def test_empty_task_ids_returns_empty_dict(self):
        """(d) empty task_ids=[] → returns {} with no I/O."""
        memory_service = AsyncMock()

        result = await track_human_operator_stalls(
            memory_service=memory_service,
            project_id='proj',
            run_id='run-004',
            task_ids=[],
        )

        assert result == {}
        memory_service.count_memories_by_metadata.assert_not_awaited()
        memory_service.add_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_multiple_task_ids_parallel(self):
        """(e) multiple task_ids → two count calls and two add_memory calls."""
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(return_value=1)
        memory_service.add_memory = AsyncMock(return_value={'memory_ids': ['mx']})

        result = await track_human_operator_stalls(
            memory_service=memory_service,
            project_id='proj',
            run_id='run-005',
            task_ids=['10', '20'],
        )

        assert set(result.keys()) == {'10', '20'}
        assert result['10'] == 2
        assert result['20'] == 2
        assert memory_service.count_memories_by_metadata.await_count == 2
        assert memory_service.add_memory.await_count == 2


# ---------------------------------------------------------------------------
# compute_stalled_task_ids
# ---------------------------------------------------------------------------


class TestComputeStalledTaskIds:
    """compute_stalled_task_ids returns sorted task_ids that meet threshold."""

    def test_empty_dict_returns_empty(self):
        """(a) empty dict → []."""
        assert compute_stalled_task_ids({}) == []

    def test_all_below_threshold_returns_empty(self):
        """(b) all below threshold → []."""
        assert compute_stalled_task_ids({'a': 1, 'b': 4}, threshold=5) == []

    def test_at_threshold_included(self):
        """(c) count == threshold → included."""
        assert compute_stalled_task_ids({'a': 5}, threshold=5) == ['a']

    def test_above_threshold_included(self):
        """(d) count > threshold → included."""
        assert compute_stalled_task_ids({'a': 9}, threshold=5) == ['a']

    def test_result_sorted_lexicographically(self):
        """(e) result is lexicographically sorted."""
        counts = {'z': 7, 'a': 6, 'm': 5}
        result = compute_stalled_task_ids(counts, threshold=5)
        assert result == ['a', 'm', 'z']

    def test_custom_threshold_respected(self):
        """(f) custom threshold overrides default."""
        counts = {'x': 3, 'y': 2}
        assert compute_stalled_task_ids(counts, threshold=3) == ['x']
        assert compute_stalled_task_ids(counts, threshold=2) == ['x', 'y']

    def test_default_threshold_is_five(self):
        """(g) calling without explicit threshold uses STAGE1_HUMAN_OPERATOR_STALL_THRESHOLD (5)."""
        counts = {'a': 4, 'b': 5}
        result = compute_stalled_task_ids(counts)
        assert result == ['b']


# ---------------------------------------------------------------------------
# maybe_escalate_stalled_tasks
# ---------------------------------------------------------------------------


class TestMaybeEscalateStalledTasks:
    """maybe_escalate_stalled_tasks submits level-1 escalations for stalled tasks."""

    def _make_queue(self, has_open_l1_return: bool = False) -> MagicMock:
        """Build a fake escalation queue."""
        q = MagicMock()
        q.has_open_l1.return_value = has_open_l1_return
        q.make_id.return_value = 'esc-1155-001'
        q.submit.return_value = 'esc-1155-001'
        return q

    @pytest.mark.asyncio
    async def test_happy_path_submits_level1_escalation(self):
        """(a) happy path: submits Escalation with correct attributes; returns ['1155']."""
        queue = self._make_queue(has_open_l1_return=False)
        flags = [
            {
                'task_id': '1155',
                'flag_type': 'assumption_invalid',
                'resolution_status': 'human_operator_required',
                'description': 'Task stalled waiting for human review',
            }
        ]

        result = await maybe_escalate_stalled_tasks(
            escalation_queue=queue,
            project_id='dark_factory',
            run_id='run-abc',
            stalled_task_ids=['1155'],
            stall_counts={'1155': 7},
            flags=flags,
        )

        assert result == ['1155']
        queue.submit.assert_called_once()
        submitted = queue.submit.call_args[0][0]

        # Verify escalation fields
        assert submitted.level == 1
        assert submitted.severity == 'blocking'
        assert submitted.category == 'reconciliation_stale_human_operator'
        assert submitted.task_id == '1155'
        assert submitted.agent_role == 'reconciliation-stage1'
        assert '1155' in submitted.summary
        assert '7' in submitted.summary
        # detail includes flag description and run_id
        assert 'Task stalled waiting for human review' in submitted.detail
        assert 'run-abc' in submitted.detail

    @pytest.mark.asyncio
    async def test_skips_when_open_l1_exists(self):
        """(b) has_open_l1=True → no submit call; returns []."""
        queue = self._make_queue(has_open_l1_return=True)

        result = await maybe_escalate_stalled_tasks(
            escalation_queue=queue,
            project_id='dark_factory',
            run_id='run-abc',
            stalled_task_ids=['1155'],
            stall_counts={'1155': 7},
            flags=[],
        )

        assert result == []
        queue.submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_mixed_already_escalated_and_new(self):
        """(c) task A already escalated, task B new → only B submitted; returns ['B']."""
        queue = MagicMock()
        queue.has_open_l1.side_effect = lambda tid: tid == 'A'
        queue.make_id.return_value = 'esc-B-001'
        queue.submit.return_value = 'esc-B-001'

        result = await maybe_escalate_stalled_tasks(
            escalation_queue=queue,
            project_id='dark_factory',
            run_id='run-mixed',
            stalled_task_ids=['A', 'B'],
            stall_counts={'A': 8, 'B': 6},
            flags=[],
        )

        assert result == ['B']
        queue.submit.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_stalled_list_returns_empty(self):
        """(d) empty stalled_task_ids → no calls, returns []."""
        queue = self._make_queue()

        result = await maybe_escalate_stalled_tasks(
            escalation_queue=queue,
            project_id='dark_factory',
            run_id='run-empty',
            stalled_task_ids=[],
            stall_counts={},
            flags=[],
        )

        assert result == []
        queue.has_open_l1.assert_not_called()
        queue.submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_submit_failure_logs_warning_and_excludes_task(self):
        """(e) submit raises RuntimeError → logs WARNING; failed task excluded from return."""
        queue = self._make_queue(has_open_l1_return=False)
        queue.submit.side_effect = RuntimeError('queue full')

        result = await maybe_escalate_stalled_tasks(
            escalation_queue=queue,
            project_id='dark_factory',
            run_id='run-fail',
            stalled_task_ids=['1155'],
            stall_counts={'1155': 7},
            flags=[],
        )

        # Should NOT raise; failed task excluded
        assert result == []


# ---------------------------------------------------------------------------
# TestEscalationBinding
# ---------------------------------------------------------------------------


class TestEscalationBinding:
    """Verify Escalation is unconditionally bound on the module namespace.

    Two behavioral checks replace the previous importlib.reload approach:

    1. ``test_escalation_attribute_is_bound`` — runtime hasattr on the loaded
       module; catches regressions where the symbol is dropped entirely.
    2. ``test_except_branch_binds_escalation_to_none`` — loads a fresh isolated
       copy of the production module from source via
       ``importlib.util.spec_from_file_location`` with ``sys.modules['escalation']``
       and ``sys.modules['escalation.models']`` stubbed to ``None``; asserts the
       except branch binds ``Escalation = None`` and ``_HAS_ESCALATION = False``
       as actual runtime behavior even when CI has escalation installed.
    """

    def test_escalation_attribute_is_bound(self):
        """Escalation is always bound on the module namespace.

        When the escalation package is installed (normal CI), also asserts the
        success-branch contract: ``Escalation`` is the real class and
        ``_HAS_ESCALATION`` is ``True``.
        """
        import fused_memory.reconciliation.stage1_stall_detector as mod

        assert hasattr(mod, 'Escalation'), (
            'Escalation must be bound on the module namespace at all times'
        )
        # In this repo the escalation package is always installed, so the
        # try-branch runs and _HAS_ESCALATION must be True.
        assert mod._HAS_ESCALATION is True, (
            '_HAS_ESCALATION must be True when the escalation package is installed'
        )

    def test_except_branch_binds_escalation_to_none(self):
        """Behavioral check: except ImportError branch sets Escalation=None and _HAS_ESCALATION=False.

        Loads an isolated copy of the production module from source via
        ``importlib.util.spec_from_file_location`` against a synthetic module
        name, with ``sys.modules['escalation']`` and
        ``sys.modules['escalation.models']`` stubbed to ``None`` so Python
        treats them as failed imports and falls through to the except branch.
        The live ``fused_memory.reconciliation.stage1_stall_detector`` cache
        entry is never overwritten — no reload blast radius.
        """
        import fused_memory.reconciliation.stage1_stall_detector as live_mod

        source_path = live_mod.__file__
        _MISSING = object()

        esc_saved = sys.modules.get('escalation', _MISSING)
        esc_models_saved = sys.modules.get('escalation.models', _MISSING)

        try:
            sys.modules['escalation'] = None  # type: ignore[assignment]
            sys.modules['escalation.models'] = None  # type: ignore[assignment]

            spec = importlib.util.spec_from_file_location(
                '_test_stage1_stall_detector_isolated', source_path
            )
            assert spec is not None and spec.loader is not None, source_path
            fresh_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fresh_mod)  # type: ignore[union-attr]

            assert fresh_mod.Escalation is None, (
                'When escalation package is unavailable, '
                'the except ImportError branch must bind Escalation = None'
            )
            assert fresh_mod._HAS_ESCALATION is False, (
                'When escalation package is unavailable, '
                'the except ImportError branch must set _HAS_ESCALATION = False'
            )
        finally:
            if esc_saved is _MISSING:
                sys.modules.pop('escalation', None)
            else:
                sys.modules['escalation'] = esc_saved  # type: ignore[assignment]
            if esc_models_saved is _MISSING:
                sys.modules.pop('escalation.models', None)
            else:
                sys.modules['escalation.models'] = esc_models_saved  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Gate-backlog age check — constants (task 3017)
# ---------------------------------------------------------------------------


class TestGateBacklogConstants:
    """Assert the gate-backlog module-level constants have expected values."""

    def test_threshold_secs_is_48h(self):
        assert STAGE1_GATE_BACKLOG_STALL_THRESHOLD_SECS == 48 * 3600

    def test_escalation_category_value(self):
        assert _GATE_BACKLOG_ESCALATION_CATEGORY == 'reconciliation_stale_gate_backlog'


# ---------------------------------------------------------------------------
# gate_escalated_age_secs (task 3017)
# ---------------------------------------------------------------------------


class TestGateEscalatedAgeSecs:
    """gate_escalated_age_secs returns the age (secs) of metadata.gate_escalated_at.

    Pure helper — never raises; returns None on any malformed / non-gate input.
    ``now`` is injected so the age is deterministic.
    """

    def test_blocked_gate_metadata_returns_age(self):
        """(a) operational_mode='gate', gate_escalated_at = now-49h → ~49*3600s."""
        meta = {
            'operational_mode': 'gate',
            'gate_escalated_at': (_GATE_NOW - timedelta(hours=49)).isoformat(),
        }
        age = gate_escalated_age_secs(meta, now=_GATE_NOW)
        assert age == pytest.approx(49 * 3600, abs=1.0)

    def test_operational_mode_absent_treated_as_gate(self):
        """(b) operational_mode absent → treated as 'gate' (returns an age)."""
        meta = {'gate_escalated_at': (_GATE_NOW - timedelta(hours=10)).isoformat()}
        age = gate_escalated_age_secs(meta, now=_GATE_NOW)
        assert age == pytest.approx(10 * 3600, abs=1.0)

    def test_operational_mode_llm_still_returns_age(self):
        """(c) operational_mode='llm' → age (llm gates still await a human today).

        Reviewer amendment (task 3017): the age check keys purely on the
        authoritative ``gate_escalated_at`` stamp, NOT on ``operational_mode``.
        Per the operational-routing contract a ``decision``/``operational`` +
        ``llm`` submission is coerced into a pure human gate (the LLM-operational
        lane is future work), so an ``operational_mode=='llm'`` blocked gate is
        exactly the population this safety-net must cover.
        """
        meta = {
            'operational_mode': 'llm',
            'gate_escalated_at': (_GATE_NOW - timedelta(hours=49)).isoformat(),
        }
        age = gate_escalated_age_secs(meta, now=_GATE_NOW)
        assert age == pytest.approx(49 * 3600, abs=1.0)

    def test_metadata_not_a_dict_returns_none(self):
        """(d) metadata not a dict (None / str / int) → None, no raise."""
        assert gate_escalated_age_secs(None, now=_GATE_NOW) is None
        assert gate_escalated_age_secs('str', now=_GATE_NOW) is None  # type: ignore[arg-type]
        assert gate_escalated_age_secs(123, now=_GATE_NOW) is None  # type: ignore[arg-type]

    def test_gate_escalated_at_missing_or_nonstr_returns_none(self):
        """(e) gate_escalated_at missing / None / '' / non-str → None."""
        assert gate_escalated_age_secs({'operational_mode': 'gate'}, now=_GATE_NOW) is None
        assert gate_escalated_age_secs({'gate_escalated_at': None}, now=_GATE_NOW) is None
        assert gate_escalated_age_secs({'gate_escalated_at': ''}, now=_GATE_NOW) is None
        assert gate_escalated_age_secs({'gate_escalated_at': 123}, now=_GATE_NOW) is None

    def test_unparseable_gate_escalated_at_returns_none(self):
        """(f) unparseable ISO-8601 → None (ValueError swallowed)."""
        assert gate_escalated_age_secs(
            {'gate_escalated_at': 'not-a-date'}, now=_GATE_NOW
        ) is None

    def test_naive_gate_escalated_at_returns_none(self):
        """(g) naive (tz-less) stamp → None (aware-vs-naive TypeError swallowed)."""
        naive = datetime(2026, 7, 22, 12, 0, 0).isoformat()  # no tzinfo
        assert gate_escalated_age_secs({'gate_escalated_at': naive}, now=_GATE_NOW) is None

    def test_future_gate_escalated_at_returns_negative_age(self):
        """(h) future stamp (now+2h) → negative age (helper does not clamp)."""
        meta = {'gate_escalated_at': (_GATE_NOW + timedelta(hours=2)).isoformat()}
        age = gate_escalated_age_secs(meta, now=_GATE_NOW)
        assert age is not None
        assert age == pytest.approx(-2 * 3600, abs=1.0)


# ---------------------------------------------------------------------------
# extract_stalled_gate_backlog_task_ids (task 3017)
# ---------------------------------------------------------------------------


def _gate_task(
    tid,
    *,
    hours_ago: float = 49,
    status: str = 'blocked',
    operational_mode: str | None = 'gate',
    with_stamp: bool = True,
) -> dict:
    """Build a task dict shaped like a blocked human-decision gate."""
    meta: dict = {}
    if operational_mode is not None:
        meta['operational_mode'] = operational_mode
    if with_stamp:
        meta['gate_escalated_at'] = (_GATE_NOW - timedelta(hours=hours_ago)).isoformat()
    return {'id': tid, 'status': status, 'metadata': meta}


class TestExtractStalledGateBacklogTaskIds:
    """extract_stalled_gate_backlog_task_ids selects blocked, aged gate tasks."""

    def test_empty_list_returns_empty(self):
        """(a) empty list → []."""
        assert extract_stalled_gate_backlog_task_ids([], now=_GATE_NOW) == []

    def test_blocked_stale_gate_included_id_coerced_to_str(self):
        """(b) blocked gate aged 49h (> 48h) → ['645'] (int id coerced to str)."""
        tasks = [_gate_task(645, hours_ago=49)]
        assert extract_stalled_gate_backlog_task_ids(tasks, now=_GATE_NOW) == ['645']

    def test_blocked_fresh_gate_excluded(self):
        """(c) blocked gate aged 47h (< 48h) → excluded."""
        tasks = [_gate_task(645, hours_ago=47)]
        assert extract_stalled_gate_backlog_task_ids(tasks, now=_GATE_NOW) == []

    def test_non_blocked_status_excluded(self):
        """(d) stale gate stamp but status != 'blocked' → excluded."""
        tasks = [
            _gate_task(1, hours_ago=49, status='in-progress'),
            _gate_task(2, hours_ago=49, status='pending'),
        ]
        assert extract_stalled_gate_backlog_task_ids(tasks, now=_GATE_NOW) == []

    def test_operational_mode_llm_included(self):
        """(e) operational_mode='llm' with stale stamp → included.

        Reviewer amendment (task 3017): a blocked ``operational_mode=='llm'``
        gate still awaits a human decision today (the LLM-operational lane is
        future work), so it must be selected — the check keys on the
        ``gate_escalated_at`` stamp, not ``operational_mode``.
        """
        tasks = [_gate_task(3, hours_ago=49, operational_mode='llm')]
        assert extract_stalled_gate_backlog_task_ids(tasks, now=_GATE_NOW) == ['3']

    def test_operational_llm_pure_gate_with_marker_included(self):
        """operational+llm pure gate (carries x_operational_llm_gate) → included.

        The exact real-world shape the operational-routing guard produces for an
        execution_class='operational' + operational_mode='llm' submission: a
        blocked pure gate stamped with gate_escalated_at AND the
        x_operational_llm_gate marker.  It awaits a human decision today, so the
        gate-backlog safety-net must select it (task 3017 reviewer amendment).
        """
        stamp = (_GATE_NOW - timedelta(hours=49)).isoformat()
        tasks = [
            {
                'id': 646,
                'status': 'blocked',
                'metadata': {
                    'operational_mode': 'llm',
                    'x_operational_llm_gate': True,
                    'gate_escalated_at': stamp,
                },
            }
        ]
        assert extract_stalled_gate_backlog_task_ids(tasks, now=_GATE_NOW) == ['646']

    def test_no_gate_escalated_at_excluded(self):
        """(f) blocked task without a gate_escalated_at stamp → excluded."""
        tasks = [_gate_task(4, with_stamp=False)]
        assert extract_stalled_gate_backlog_task_ids(tasks, now=_GATE_NOW) == []

    def test_missing_or_none_id_skipped(self):
        """(g) blocked, stale gate tasks with missing/None id → skipped (no raise)."""
        stamp = (_GATE_NOW - timedelta(hours=49)).isoformat()
        tasks = [
            {'status': 'blocked', 'metadata': {'operational_mode': 'gate', 'gate_escalated_at': stamp}},
            {'id': None, 'status': 'blocked',
             'metadata': {'operational_mode': 'gate', 'gate_escalated_at': stamp}},
        ]
        assert extract_stalled_gate_backlog_task_ids(tasks, now=_GATE_NOW) == []

    def test_multiple_stalled_sorted_and_deduped(self):
        """(h) two stalled tasks → sorted; the same id twice collapses to once."""
        tasks = [
            _gate_task(650, hours_ago=50),
            _gate_task(648, hours_ago=60),
            _gate_task(650, hours_ago=70),  # duplicate id
        ]
        assert extract_stalled_gate_backlog_task_ids(tasks, now=_GATE_NOW) == ['648', '650']

    def test_custom_threshold_secs_respected(self):
        """(i) custom threshold_secs overrides the 48h default."""
        tasks = [_gate_task(5, hours_ago=10)]  # 10h old
        assert extract_stalled_gate_backlog_task_ids(tasks, now=_GATE_NOW) == []
        assert extract_stalled_gate_backlog_task_ids(
            tasks, now=_GATE_NOW, threshold_secs=5 * 3600
        ) == ['5']


# ---------------------------------------------------------------------------
# maybe_escalate_stalled_gate_backlog (task 3017)
# ---------------------------------------------------------------------------


def _gate_task_record(tid, *, hours_ago: float = 49) -> dict:
    """A blocked gate task record for task_by_id lookups (aged `hours_ago`)."""
    stamp = (_GATE_NOW - timedelta(hours=hours_ago)).isoformat()
    return {
        'id': tid,
        'status': 'blocked',
        'title': f'Gate task {tid}',
        'metadata': {'operational_mode': 'gate', 'gate_escalated_at': stamp},
    }


class TestMaybeEscalateStalledGateBacklog:
    """maybe_escalate_stalled_gate_backlog files per-task level-1 gate-backlog L1s."""

    def _make_queue(self, has_open_l1_return: bool = False) -> MagicMock:
        q = MagicMock()
        q.has_open_l1.return_value = has_open_l1_return
        q.make_id.return_value = 'esc-645-001'
        q.submit.return_value = 'esc-645-001'
        # REQUIRED once the path routes through submit_or_dedupe: find_dedupe_parent
        # iterates queue.get_pending(), and `for parent in <bare MagicMock>` raises
        # TypeError, which the caller's broad `except Exception` swallows into a
        # WARNING — silently flipping every shape test below to result == [] with no
        # visible error.
        q.get_pending.return_value = []
        # Pins observed_submit_response to its documented fail-open 'queued' branch
        # instead of leaning on MagicMock attribute truthiness.
        q.get.return_value = None
        return q

    @pytest.mark.asyncio
    async def test_happy_path_submits_level1_gate_backlog(self):
        """(a) submits one Escalation with the gate-backlog fields; returns ['645']."""
        queue = self._make_queue(has_open_l1_return=False)
        stamp = (_GATE_NOW - timedelta(hours=49)).isoformat()
        task_by_id = {'645': _gate_task_record(645, hours_ago=49)}

        result = await maybe_escalate_stalled_gate_backlog(
            escalation_queue=queue,
            project_id='autopilot_video',
            run_id='run-xyz',
            stalled_task_ids=['645'],
            task_by_id=task_by_id,
            now=_GATE_NOW,
        )

        assert result == ['645']
        queue.submit.assert_called_once()
        submitted = queue.submit.call_args[0][0]
        assert submitted.level == 1
        assert submitted.severity == 'blocking'
        assert submitted.category == 'reconciliation_stale_gate_backlog'
        assert submitted.task_id == '645'
        assert submitted.agent_role == 'reconciliation-stage1'

        combined = f'{submitted.summary}\n{submitted.detail}'
        assert '645' in combined
        assert 'run-xyz' in combined
        assert stamp in combined  # the gate_escalated_at value

        # The summary must state a live-truthful anchor (the gate_escalated_at
        # stamp) plus the static, never-staling threshold fact — not a
        # filing-time-computed elapsed-hours figure that goes stale while the
        # escalation sits open (task 3520). Naming the threshold alongside the
        # anchor keeps a compact-drain (summary-only) read legible without a
        # mental diff against "now".
        assert 'has awaited a human decision since' in submitted.summary
        assert stamp in submitted.summary
        assert 'past the 48h gate-backlog threshold' in submitted.summary
        # Concrete negative (not a shape-based regex): pins that the stale,
        # filing-time-computed age (49h, from hours_ago=49 above) never leaks
        # into the summary. Scoped to `summary`, not `combined` — the detail
        # block legitimately retains a filing-time hours value
        # (age_hours_at_filing), and the summary now legitimately names the
        # static 48h threshold, so a shape regex over `\d+h` would false-
        # positive on that intentional, truthful content.
        assert '49' not in submitted.summary

        # The filing-time age is still retained in detail, but relabeled
        # `age_hours_at_filing` so it no longer reads as a live counter.
        assert 'age_hours_at_filing: 49.0' in submitted.detail
        assert 'age_hours:' not in submitted.detail

    @pytest.mark.asyncio
    async def test_unparseable_stamp_uses_threshold_summary_and_at_filing_label(self):
        """(a2) unparseable gate_escalated_at → fallback summary; detail still relabeled."""
        queue = self._make_queue(has_open_l1_return=False)
        task = _gate_task_record(645, hours_ago=49)
        task['metadata']['gate_escalated_at'] = 'not-a-timestamp'
        task_by_id = {'645': task}

        result = await maybe_escalate_stalled_gate_backlog(
            escalation_queue=queue,
            project_id='autopilot_video',
            run_id='run-xyz',
            stalled_task_ids=['645'],
            task_by_id=task_by_id,
            now=_GATE_NOW,
        )

        assert result == ['645']
        queue.submit.assert_called_once()
        submitted = queue.submit.call_args[0][0]

        # The out-of-scope fallback branch is unchanged: it still names the
        # threshold, not a computed age, and never echoes the unparseable
        # stamp as if it were a trustworthy anchor.
        assert 'beyond the 48h gate-backlog threshold' in submitted.summary
        assert 'not-a-timestamp' not in submitted.summary

        # The relabel applies to the "unknown" fallback value too.
        assert 'age_hours_at_filing: unknown' in submitted.detail
        assert 'age_hours:' not in submitted.detail

    @pytest.mark.asyncio
    async def test_stamps_expected_dedupe_fingerprint(self):
        """(a3) The filed escalation carries the (category, project, task) content key.

        The fingerprint is what ``DedupeConfig.for_gate_backlog()`` folds on, so
        its exact composition is a contract, not an implementation detail.
        """
        from escalation.dedupe import compute_content_fingerprint

        queue = self._make_queue(has_open_l1_return=False)
        task_by_id = {'645': _gate_task_record(645, hours_ago=49)}

        await maybe_escalate_stalled_gate_backlog(
            escalation_queue=queue,
            project_id='autopilot_video',
            run_id='run-xyz',
            stalled_task_ids=['645'],
            task_by_id=task_by_id,
            now=_GATE_NOW,
        )

        submitted = queue.submit.call_args[0][0]
        expected = compute_content_fingerprint(
            'reconciliation_stale_gate_backlog', '', ['autopilot_video:645'], ''
        )
        assert submitted.dedupe_fingerprint == expected

        # NON-EMPTY is load-bearing on its own: find_dedupe_parent short-circuits
        # to None on a falsy key, so an empty fingerprint means "never fold" —
        # silently re-pinning dedupe_count at 0, the exact defect being fixed.
        assert submitted.dedupe_fingerprint, (
            'fingerprint must be non-empty; a falsy key makes find_dedupe_parent '
            'never fold and silently mints a duplicate record every cycle'
        )

    @pytest.mark.asyncio
    async def test_fingerprint_stable_across_cycles(self):
        """(a4) 100h apart, drifting prose, same gate → identical non-empty fingerprint."""
        task_by_id = {'645': _gate_task_record(645, hours_ago=49)}

        queue1 = self._make_queue(has_open_l1_return=False)
        await maybe_escalate_stalled_gate_backlog(
            escalation_queue=queue1,
            project_id='autopilot_video',
            run_id='run-cycle-1',
            stalled_task_ids=['645'],
            task_by_id=task_by_id,
            now=_GATE_NOW,
        )
        first = queue1.submit.call_args[0][0]

        queue2 = self._make_queue(has_open_l1_return=False)
        await maybe_escalate_stalled_gate_backlog(
            escalation_queue=queue2,
            project_id='autopilot_video',
            run_id='run-cycle-2',
            stalled_task_ids=['645'],
            task_by_id=task_by_id,
            now=_GATE_NOW + timedelta(hours=100),
        )
        second = queue2.submit.call_args[0][0]

        # Precondition: the prose really did drift between the two cycles, so
        # equality below is proving stability, not just re-reading a constant.
        assert first.detail != second.detail
        assert 'age_hours_at_filing: 49.0' in first.detail
        assert 'age_hours_at_filing: 149.0' in second.detail

        # Truthiness is REQUIRED here: today both fingerprints are None, so a
        # bare `first == second` would pass vacuously green.
        assert first.dedupe_fingerprint, 'cycle-1 fingerprint must be non-empty'
        assert second.dedupe_fingerprint, 'cycle-2 fingerprint must be non-empty'
        assert first.dedupe_fingerprint == second.dedupe_fingerprint, (
            'fingerprint must not drift with age/run_id/prose — a drifting key '
            'never folds and re-pins dedupe_count at 0'
        )

    @pytest.mark.asyncio
    async def test_fingerprint_distinct_per_task(self):
        """(a5) Two gates in the same project get distinct fingerprints (no cross-fold)."""
        queue = MagicMock()
        queue.has_open_l1.return_value = False
        queue.make_id.side_effect = lambda tid: f'esc-{tid}-001'
        queue.submit.side_effect = lambda esc: esc.id
        task_by_id = {
            'A': _gate_task_record('A', hours_ago=50),
            'B': _gate_task_record('B', hours_ago=50),
        }

        await maybe_escalate_stalled_gate_backlog(
            escalation_queue=queue,
            project_id='autopilot_video',
            run_id='r',
            stalled_task_ids=['A', 'B'],
            task_by_id=task_by_id,
            now=_GATE_NOW,
        )

        fps = [call[0][0].dedupe_fingerprint for call in queue.submit.call_args_list]
        assert len(fps) == 2
        assert all(fps), f'both fingerprints must be non-empty; got {fps}'
        assert fps[0] != fps[1], 'distinct gates must not share a fold key'

    @pytest.mark.asyncio
    async def test_has_open_l1_no_longer_consulted(self):
        """(b0) The categorized has_open_l1 skip is deliberately gone.

        Dedup for this path is now the submit_or_dedupe fold, not a suppression
        check — so consulting has_open_l1 at all would be a regression toward
        the old "second cycle vanishes" behaviour.
        """
        queue = self._make_queue(has_open_l1_return=False)

        result = await maybe_escalate_stalled_gate_backlog(
            escalation_queue=queue,
            project_id='autopilot_video',
            run_id='run-xyz',
            stalled_task_ids=['645'],
            task_by_id={'645': _gate_task_record(645, hours_ago=49)},
            now=_GATE_NOW,
        )

        assert result == ['645']
        queue.has_open_l1.assert_not_called()

    @pytest.mark.asyncio
    async def test_folds_into_existing_pending_gate_backlog_parent(self, tmp_path):
        """(b) Repeat cycles over the same stale gate fold into ONE parent record.

        Uses a REAL EscalationQueue: the load-bearing claim is "exactly one
        pending record on disk, same id, dedupe_count incremented" — on-disk
        state a MagicMock cannot express (it could only prove a call happened).
        """
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        task_by_id = {'645': _gate_task_record(645, hours_ago=49)}

        async def _cycle(now, run_id):
            return await maybe_escalate_stalled_gate_backlog(
                escalation_queue=queue,
                project_id='autopilot_video',
                run_id=run_id,
                stalled_task_ids=['645'],
                task_by_id=task_by_id,
                now=now,
            )

        # Cycle 1 — gate 49h stale, first filing.
        assert await _cycle(_GATE_NOW, 'run-1') == ['645']
        pending = queue.get_pending()
        assert len(pending) == 1
        parent_id = pending[0].id
        first_summary = pending[0].summary
        first_detail = pending[0].detail

        # Cycle 2 — same gate, now 149h stale. Must fold, not mint a second record.
        assert await _cycle(_GATE_NOW + timedelta(hours=100), 'run-2') == [], (
            'a folded cycle files no NEW escalation, so it reports no task_ids'
        )

        pending = queue.get_pending()
        assert len(pending) == 1, (
            f'fold must leave exactly ONE pending record; got {len(pending)}: '
            f'{[e.id for e in pending]}'
        )
        parent = pending[0]
        assert parent.id == parent_id, 'the fold target must be the SAME record'
        assert parent.dedupe_count == 1, (
            'dedupe_count is the recurrence signal a steward triages on; it must '
            f'advance once per stale cycle. got {parent.dedupe_count}'
        )
        assert len(parent.dedupe_children) == 1
        assert parent.dedupe_children[0] != parent_id, (
            'the folded child id must be distinct from the parent id'
        )
        # No accidental ladder/severity drift: _max_severity is a no-op here
        # because both records are blocking L1s.
        assert parent.severity == 'blocking'
        assert parent.level == 1

        # attach_dedupe_child does NOT rewrite the text. That is SAFE only
        # because task 3520 made the summary an absolute `since <ISO>` anchor,
        # which never goes stale — unlike the elapsed-hours phrasing it replaced.
        assert parent.summary == first_summary
        assert parent.detail == first_detail

        # Cycle 3 — still stale: the count keeps climbing on the same record.
        assert await _cycle(_GATE_NOW + timedelta(hours=200), 'run-3') == []
        assert len(queue.get_pending()) == 1
        reread = queue.get(parent_id)
        assert reread is not None
        assert reread.dedupe_count == 2

    @pytest.mark.asyncio
    async def test_mixed_already_escalated_and_new(self, tmp_path):
        """(c) Gate A already has a pending record, gate B is new → only B is new."""
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        task_by_id = {
            'A': _gate_task_record('A', hours_ago=50),
            'B': _gate_task_record('B', hours_ago=50),
        }

        # Pre-file A's parent in an earlier cycle.
        assert await maybe_escalate_stalled_gate_backlog(
            escalation_queue=queue,
            project_id='p',
            run_id='r0',
            stalled_task_ids=['A'],
            task_by_id=task_by_id,
            now=_GATE_NOW,
        ) == ['A']
        a_parent_id = queue.get_pending()[0].id

        result = await maybe_escalate_stalled_gate_backlog(
            escalation_queue=queue,
            project_id='p',
            run_id='r1',
            stalled_task_ids=['A', 'B'],
            task_by_id=task_by_id,
            now=_GATE_NOW + timedelta(hours=1),
        )

        assert result == ['B'], f'only the NEW filing is reported; got {result}'
        pending = {e.task_id: e for e in queue.get_pending()}
        assert set(pending) == {'A', 'B'}, f'expected one record per gate; got {pending}'
        assert pending['A'].id == a_parent_id
        assert pending['A'].dedupe_count == 1
        assert pending['B'].dedupe_count == 0

    @pytest.mark.asyncio
    async def test_same_task_id_in_two_projects_does_not_fold(self, tmp_path):
        """(c2) The headline fix over the old skip: dedup is scoped BY PROJECT.

        The queue is shared across projects and task ids are small per-project
        integers, so gate 166 exists in more than one project at once.  The
        categorized ``has_open_l1(task_id, category=...)`` skip this replaced
        keyed on task_id ALONE, so project B's stalled gate 166 was silently
        suppressed whenever project A's gate 166 already had a pending L1 —
        losing an escalation a human was waiting on.

        Asserted end-to-end through the emitter against a real queue, not only
        at the pure-key level (``TestGateBacklogFingerprintKey``): the key being
        distinct is worth nothing if the emitter does not thread ``project_id``
        into it.
        """
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        task_by_id = {'166': _gate_task_record('166', hours_ago=50)}

        async def _cycle(project_id, run_id):
            return await maybe_escalate_stalled_gate_backlog(
                escalation_queue=queue,
                project_id=project_id,
                run_id=run_id,
                stalled_task_ids=['166'],
                task_by_id=task_by_id,
                now=_GATE_NOW,
            )

        assert await _cycle('dark_factory', 'run-df') == ['166']
        assert await _cycle('reify', 'run-reify') == ['166'], (
            'the same task_id in a DIFFERENT project is a different gate and must '
            'still file — under the old has_open_l1(task_id) skip it vanished'
        )

        pending = queue.get_pending()
        assert len(pending) == 2, (
            'two projects\' gate 166 are two distinct escalations; got '
            f'{len(pending)}: {[(e.id, e.summary) for e in pending]}'
        )
        assert {e.dedupe_count for e in pending} == {0}, (
            'neither record folded into the other, so both stay at dedupe_count 0; '
            f'got {[(e.id, e.dedupe_count) for e in pending]}'
        )
        assert len({e.id for e in pending}) == 2
        # Each record names its own project, so a steward can tell them apart.
        details = sorted(e.detail.split('\n', 1)[0] for e in pending)
        assert details == ['project_id: dark_factory', 'project_id: reify']

    @pytest.mark.asyncio
    async def test_folds_into_legacy_fingerprintless_parent(self, tmp_path):
        """MIGRATION: a pre-stamp parent (dedupe_fingerprint=None) is still a fold target.

        Every gate-backlog record filed before the fingerprint stamp landed
        carries ``dedupe_fingerprint: None`` — see
        ``escalation.dedupe.gate_backlog_fingerprint_key`` for the live-queue
        measurement of how much of the pending backlog that covers (kept in one
        place, since a point-in-time census goes stale as records fold and
        resolve).  If those cannot be folded into, the FIRST cycle after this
        change mints a second pending record per stalled gate at
        ``dedupe_count 0``, and the legacy record — whose key stays falsy
        forever — never becomes a fold target again.  That is the exact defect
        this task exists to remove, so it is pinned here as a test rather than
        left to an operator backfill someone has to remember to run.

        The parent is hand-filed in the PRE-change shape actually observed on
        disk: relative-age summary (pre-3520) and the ``age_hours:`` detail key
        (pre-3520), so the recovery must not depend on anything but detail's
        first line.
        """
        from escalation.models import Escalation
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        stamp = (_GATE_NOW - timedelta(hours=49)).isoformat()

        # --- hand-file the legacy parent exactly as the pre-change code did ---
        legacy = Escalation(
            id=queue.make_id('645'),
            task_id='645',
            agent_role='reconciliation-stage1',
            severity='blocking',
            category='reconciliation_stale_gate_backlog',
            summary='Gate task 645 has awaited a human decision for 48.7h',
            detail='\n'.join([
                'project_id: autopilot_video',
                'run_id: run-legacy',
                'task_id: 645',
                f'gate_escalated_at: {stamp}',
                'age_hours: 48.7',
                'title: Gate task 645',
            ]),
            level=1,
        )
        assert legacy.dedupe_fingerprint is None, 'fixture must reproduce the legacy shape'
        legacy_id = queue.submit(legacy)

        task_by_id = {'645': _gate_task_record(645, hours_ago=49)}

        async def _cycle(now, run_id):
            return await maybe_escalate_stalled_gate_backlog(
                escalation_queue=queue,
                project_id='autopilot_video',
                run_id=run_id,
                stalled_task_ids=['645'],
                task_by_id=task_by_id,
                now=now,
            )

        assert await _cycle(_GATE_NOW, 'run-1') == [], (
            'folding into the legacy parent is not a NEW filing, so '
            "stage1_gate_backlog_escalated must stay 0"
        )

        pending = queue.get_pending()
        assert len(pending) == 1, (
            'the legacy parent must absorb the new filing, not sit beside it; '
            f'got {len(pending)}: {[e.id for e in pending]}'
        )
        parent = pending[0]
        assert parent.id == legacy_id, 'no new record may be minted for this gate'
        assert parent.dedupe_count == 1, (
            f'the legacy record must start counting recurrences; got {parent.dedupe_count}'
        )
        assert len(parent.dedupe_children) == 1
        assert parent.dedupe_children[0] != legacy_id

        # Steady state: the recompute is deterministic, so the unstamped parent
        # keeps resolving through the fallback every cycle.
        assert await _cycle(_GATE_NOW + timedelta(hours=100), 'run-2') == []
        assert len(queue.get_pending()) == 1
        reread = queue.get(legacy_id)
        assert reread is not None
        assert reread.dedupe_count == 2

    @pytest.mark.asyncio
    async def test_empty_stalled_list_returns_empty(self):
        """(d) empty stalled_task_ids → no calls, returns []."""
        queue = self._make_queue()

        result = await maybe_escalate_stalled_gate_backlog(
            escalation_queue=queue,
            project_id='p',
            run_id='r',
            stalled_task_ids=[],
            task_by_id={},
            now=_GATE_NOW,
        )

        assert result == []
        queue.has_open_l1.assert_not_called()
        queue.submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_fingerprint_is_not_filed(self, monkeypatch, caplog):
        """(e2) A falsy fingerprint fails CLOSED — nothing is filed, task excluded.

        Not reachable through the real callee (``compute_content_fingerprint``
        returns a sha256 hexdigest, never empty), so it is driven by patching
        that callee: the guard's whole purpose is to survive a future change to
        it.  Without the guard, a falsy key would sail past
        ``find_dedupe_parent``'s "never fold" short-circuit and mint a SECOND
        visible pending record for a gate that already has one — the exact
        single-record invariant this task exists to establish.

        Dropping the task_id is safe because the gate is still stalled on the
        next Stage-1 cycle, so the detector retries.
        """
        import fused_memory.reconciliation.stage1_stall_detector as _s1sd

        queue = self._make_queue(has_open_l1_return=False)
        monkeypatch.setattr(_s1sd, 'compute_content_fingerprint', lambda *a, **kw: '')

        with caplog.at_level('WARNING'):
            result = await maybe_escalate_stalled_gate_backlog(
                escalation_queue=queue,
                project_id='autopilot_video',
                run_id='run-xyz',
                stalled_task_ids=['645'],
                task_by_id={'645': _gate_task_record(645, hours_ago=49)},
                now=_GATE_NOW,
            )

        assert result == [], 'a gate whose fold key could not be built is not reported'
        queue.submit.assert_not_called()
        queue.attach_dedupe_child.assert_not_called()
        assert 'empty dedupe_fingerprint' in caplog.text, (
            f'the drop must be loud, not silent; got: {caplog.text!r}'
        )
        assert '645' in caplog.text, 'the warning must name the affected task_id'

    @pytest.mark.asyncio
    async def test_submit_failure_logs_warning_and_excludes(self):
        """(e) submit raises RuntimeError → no raise; failed tid excluded from return."""
        queue = self._make_queue(has_open_l1_return=False)
        queue.submit.side_effect = RuntimeError('queue full')
        task_by_id = {'645': _gate_task_record(645)}

        result = await maybe_escalate_stalled_gate_backlog(
            escalation_queue=queue,
            project_id='p',
            run_id='r',
            stalled_task_ids=['645'],
            task_by_id=task_by_id,
            now=_GATE_NOW,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_escalation_unavailable_returns_empty(self, monkeypatch):
        """(f) Escalation package unavailable (module Escalation=None) → [] with no submit."""
        import fused_memory.reconciliation.stage1_stall_detector as mod

        monkeypatch.setattr(mod, 'Escalation', None)
        queue = self._make_queue()

        result = await maybe_escalate_stalled_gate_backlog(
            escalation_queue=queue,
            project_id='p',
            run_id='r',
            stalled_task_ids=['645'],
            task_by_id={'645': _gate_task_record(645)},
            now=_GATE_NOW,
        )

        assert result == []
        queue.submit.assert_not_called()
