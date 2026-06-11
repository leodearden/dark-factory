"""Tests for the per-task retry cap on WorkflowOutcome.REQUEUED loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _orch_helpers import _init_harness_state_for_test
from pydantic import ValidationError

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.scheduler import RequeueRecord, Scheduler, is_transient_api_requeue

# --- Fixtures -----------------------------------------------------------------


@pytest.fixture
def config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        max_per_module=1,
        requeue_cap=3,
        project_root=tmp_path,
    )


@pytest.fixture
def scheduler(config: OrchestratorConfig) -> Scheduler:
    return Scheduler(config)


@dataclass
class _StubEscalationQueue:
    """Records submit() calls without touching the filesystem."""

    submitted: list = field(default_factory=list)
    _next_id: int = 0

    def make_id(self, task_id: str) -> str:
        self._next_id += 1
        return f'esc-{task_id}-{self._next_id}'

    def submit(self, escalation) -> str:
        self.submitted.append(escalation)
        return escalation.id


def _record_one(
    scheduler: Scheduler,
    task_id: str = 't1',
    *,
    reason: str = 'architect returned empty output',
    phase: str = 'plan',
    run_id: str = 'run-abc',
    cost_usd: float = 7.5,
) -> int:
    return scheduler.record_requeue(
        task_id,
        phase=phase,
        reason=reason,
        detail=reason + ' (see iteration log)',
        run_id=run_id,
        cost_usd=cost_usd,
    )


# --- Config: transient_requeue_cap field -------------------------------------


class TestTransientRequeueCapConfig:
    def test_default_is_10(self, tmp_path: Path):
        cfg = OrchestratorConfig(project_root=tmp_path)
        assert cfg.transient_requeue_cap == 10

    def test_zero_raises_validation_error(self, tmp_path: Path):
        with pytest.raises(ValidationError):
            OrchestratorConfig(project_root=tmp_path, transient_requeue_cap=0)


# --- is_transient_api_requeue predicate --------------------------------------


class TestIsTransientApiRequeue:
    @pytest.mark.parametrize('reason', [
        'Planning failed: agent API error: HTTP 529',
        'agent API error: HTTP 500',
        'agent API error: HTTP 503',
        'agent API error: HTTP 599',
    ])
    def test_true_for_5xx(self, reason: str):
        assert is_transient_api_requeue(reason) is True

    @pytest.mark.parametrize('reason', [
        'agent API error: HTTP 400',
        'agent API error: HTTP 401',
        'agent API error: HTTP 429',
        'architect returned empty output',
        None,
        '',
    ])
    def test_false_for_non_transient(self, reason):
        assert is_transient_api_requeue(reason) is False

    def test_producer_classifier_contract(self):
        """Pin the cross-module marker: classify_agent_failure(HTTP 529) -> is_transient."""
        from shared.cli_invoke import AgentResult, classify_agent_failure
        result = AgentResult(success=False, output='', api_error_status=529)
        cls = classify_agent_failure(result)
        assert is_transient_api_requeue(f'Planning failed: {cls.summary}') is True


# --- Counter mechanics --------------------------------------------------------


class TestDualCounter:
    """record_requeue routes transient (5xx API) and genuine requeues to separate counters."""

    _529_REASON = 'Planning failed: agent API error: HTTP 529'

    def _record_transient(self, scheduler: Scheduler, task_id: str = 't1') -> int:
        return scheduler.record_requeue(
            task_id,
            phase='plan',
            reason=self._529_REASON,
            detail=self._529_REASON,
            run_id='run-abc',
            cost_usd=1.0,
        )

    def test_transient_does_not_touch_genuine_counter(self, scheduler: Scheduler):
        count = self._record_transient(scheduler)
        assert count == 0  # genuine count is 0
        assert 't1' not in scheduler._requeue_counts
        assert scheduler._transient_requeue_counts['t1'] == 1
        assert len(scheduler._requeue_history['t1']) == 1
        assert scheduler.transient_requeue_count('t1') == 1

    def test_mixed_sequence_genuine_transient_genuine(self, scheduler: Scheduler):
        # genuine → transient → genuine  →  returns 1, 1, 2
        c1 = _record_one(scheduler)
        c2 = self._record_transient(scheduler)
        c3 = _record_one(scheduler)
        assert (c1, c2, c3) == (1, 1, 2)
        assert scheduler._requeue_counts['t1'] == 2
        assert scheduler._transient_requeue_counts['t1'] == 1
        history = scheduler._requeue_history['t1']
        assert len(history) == 3
        assert [r.attempt for r in history] == [1, 2, 3]

    def test_transient_requeue_count_unknown_task_returns_zero(self, scheduler: Scheduler):
        assert scheduler.transient_requeue_count('no-such-task') == 0


class TestCounter:
    def test_record_requeue_increments_per_task(self, scheduler: Scheduler):
        assert _record_one(scheduler, 't1') == 1
        assert _record_one(scheduler, 't1') == 2
        assert _record_one(scheduler, 't2') == 1
        assert _record_one(scheduler, 't1') == 3
        assert scheduler._requeue_counts == {'t1': 3, 't2': 1}
        assert len(scheduler._requeue_history['t1']) == 3
        assert len(scheduler._requeue_history['t2']) == 1

    def test_history_captures_record_fields(self, scheduler: Scheduler):
        _record_one(
            scheduler,
            't1',
            reason='empty_output',
            phase='plan',
            run_id='run-xyz',
            cost_usd=4.25,
        )
        record = scheduler._requeue_history['t1'][0]
        assert isinstance(record, RequeueRecord)
        assert record.attempt == 1
        assert record.phase == 'plan'
        assert record.reason == 'empty_output'
        assert record.run_id == 'run-xyz'
        assert record.cost_usd == pytest.approx(4.25)
        assert record.timestamp > 0

    def test_clear_requeue_count(self, scheduler: Scheduler):
        _record_one(scheduler, 't1')
        _record_one(scheduler, 't1')
        scheduler.clear_requeue_count('t1')
        assert 't1' not in scheduler._requeue_counts
        assert 't1' not in scheduler._requeue_history
        # Next requeue starts from 1 again.
        assert _record_one(scheduler, 't1') == 1

    def test_clear_also_clears_transient_counter(self, scheduler: Scheduler):
        _529 = 'Planning failed: agent API error: HTTP 529'
        scheduler.record_requeue(
            't1', phase='plan', reason=_529, detail=_529, run_id='r', cost_usd=1.0,
        )
        _record_one(scheduler, 't1')
        assert scheduler.transient_requeue_count('t1') == 1
        scheduler.clear_requeue_count('t1')
        assert 't1' not in scheduler._requeue_counts
        assert 't1' not in scheduler._requeue_history
        assert 't1' not in scheduler._transient_requeue_counts
        assert scheduler.transient_requeue_count('t1') == 0


# --- trigger_retry_cap_exhausted ---------------------------------------------


class TestTriggerRetryCapExhausted:
    @pytest.mark.asyncio
    async def test_submits_l1_escalation(
        self, scheduler: Scheduler, tmp_path: Path, monkeypatch
    ):
        scheduler.set_task_status = AsyncMock()
        queue = _StubEscalationQueue()
        for i in range(3):
            _record_one(scheduler, 't1', reason=f'attempt-{i}', cost_usd=2.0)

        report_path = await scheduler.trigger_retry_cap_exhausted(
            't1',
            run_id='run-abc',
            cost_usd=6.0,
            escalation_queue=queue,
            reports_dir=tmp_path / 'reports',
        )

        assert len(queue.submitted) == 1
        esc = queue.submitted[0]
        assert esc.level == 1
        assert esc.task_id == 't1'
        assert esc.severity == 'blocking'
        assert esc.category == 'retry_cap_exhausted'
        assert 'cap=3' in esc.summary or '(cap=3)' in esc.summary
        # Compact summary includes the count, reason, and report path pointer.
        assert '3 REQUEUED' in esc.summary or '3 REQUEUED iterations' in esc.summary
        assert 'attempt-2' in esc.summary  # last reason
        assert '$6.00' in esc.detail
        assert report_path is not None
        assert str(report_path) in esc.detail

    @pytest.mark.asyncio
    async def test_writes_report_artifact(
        self, scheduler: Scheduler, tmp_path: Path
    ):
        scheduler.set_task_status = AsyncMock()
        reports_dir = tmp_path / 'reports'
        for i in range(3):
            _record_one(
                scheduler,
                't1',
                reason=f'attempt-{i}-reason',
                phase='plan' if i % 2 == 0 else 'execute',
                cost_usd=1.5,
            )

        report_path = await scheduler.trigger_retry_cap_exhausted(
            't1',
            run_id='run-abc',
            cost_usd=4.5,
            escalation_queue=None,
            reports_dir=reports_dir,
        )

        assert report_path is not None and report_path.exists()
        body = report_path.read_text()
        assert '# Retry Cap Exhausted: task t1' in body
        assert 'Run ID:** run-abc' in body
        assert 'Cap:** 3' in body
        assert 'Attempts recorded:** 3' in body
        assert '$4.50' in body
        # Every attempt appears in the timeline.
        for i in range(3):
            assert f'attempt-{i}-reason' in body
        # Dig-deeper SQL snippet is present with the right identifiers.
        assert 'sqlite3 data/orchestrator/runs.db' in body
        assert "task_id='t1' AND run_id='run-abc'" in body

    @pytest.mark.asyncio
    async def test_sets_blocked_and_clears_counter(
        self, scheduler: Scheduler, tmp_path: Path
    ):
        scheduler.set_task_status = AsyncMock()
        for _ in range(3):
            _record_one(scheduler, 't1')

        await scheduler.trigger_retry_cap_exhausted(
            't1',
            run_id='run-abc',
            cost_usd=0.0,
            escalation_queue=None,
            reports_dir=tmp_path / 'reports',
        )

        scheduler.set_task_status.assert_any_await('t1', 'blocked')
        assert 't1' not in scheduler._requeue_counts
        assert 't1' not in scheduler._requeue_history

    @pytest.mark.asyncio
    async def test_emits_event_store_row(
        self, scheduler: Scheduler, tmp_path: Path
    ):
        scheduler.set_task_status = AsyncMock()
        emitted: list[tuple] = []

        class _FakeEventStore:
            def emit(self, event_type, **kwargs) -> None:
                emitted.append((event_type, kwargs))

        scheduler.event_store = _FakeEventStore()  # type: ignore[assignment]
        for i in range(3):
            _record_one(scheduler, 't1', reason=f'r{i}', cost_usd=1.0)

        await scheduler.trigger_retry_cap_exhausted(
            't1',
            run_id='run-abc',
            cost_usd=3.0,
            escalation_queue=None,
            reports_dir=tmp_path / 'reports',
        )

        cap_events = [
            kw for (et, kw) in emitted if et == EventType.retry_cap_exhausted
        ]
        assert len(cap_events) == 1
        data = cap_events[0]['data']
        assert data['cap'] == 3
        assert data['requeue_count'] == 3
        assert data['last_reason'] == 'r2'
        assert data['report_path']  # non-empty

    @pytest.mark.asyncio
    async def test_still_escalates_when_set_status_fails(
        self, scheduler: Scheduler, tmp_path: Path
    ):
        """If set_task_status raises, the L1 escalation must still be submitted.

        A human must be alerted even when fused-memory is unreachable.
        """
        scheduler.set_task_status = AsyncMock(
            side_effect=RuntimeError('memory unreachable')
        )
        queue = _StubEscalationQueue()
        _record_one(scheduler, 't1')

        await scheduler.trigger_retry_cap_exhausted(
            't1',
            run_id='run-abc',
            cost_usd=1.0,
            escalation_queue=queue,
            reports_dir=tmp_path / 'reports',
        )

        assert len(queue.submitted) == 1

    @pytest.mark.asyncio
    async def test_explicit_cap_param_overrides_config(
        self, scheduler: Scheduler, tmp_path: Path
    ):
        """Passing cap=10 overrides config.requeue_cap (3) in report, event, escalation."""
        scheduler.set_task_status = AsyncMock()
        emitted: list[tuple] = []

        class _FakeEventStore:
            def emit(self, event_type, **kwargs) -> None:
                emitted.append((event_type, kwargs))

        scheduler.event_store = _FakeEventStore()  # type: ignore[assignment]
        queue = _StubEscalationQueue()
        for i in range(3):
            _record_one(scheduler, 't1', reason=f'r{i}', cost_usd=1.0)

        report_path = await scheduler.trigger_retry_cap_exhausted(
            't1',
            run_id='run-abc',
            cost_usd=3.0,
            escalation_queue=queue,
            reports_dir=tmp_path / 'reports',
            cap=10,
        )

        # Report file must show cap=10
        assert report_path is not None and report_path.exists()
        body = report_path.read_text()
        assert 'Cap:** 10' in body

        # Event data must carry cap=10
        cap_events = [kw for (et, kw) in emitted if et == EventType.retry_cap_exhausted]
        assert len(cap_events) == 1
        assert cap_events[0]['data']['cap'] == 10

        # Escalation summary must reference cap=10
        assert len(queue.submitted) == 1
        assert 'cap=10' in queue.submitted[0].summary


# --- Harness-level integration -----------------------------------------------


def _build_report(outcome, *, cost_usd: float = 5.0, steward_cost_usd: float = 2.5):
    from orchestrator.harness import TaskReport

    return TaskReport(
        task_id='t1',
        title='T1',
        outcome=outcome,
        cost_usd=cost_usd,
        steward_cost_usd=steward_cost_usd,
        block_reason='architect returned empty output',
        block_detail='see iteration log',
        block_phase='plan',
    )


def _build_529_report(outcome, *, cost_usd: float = 5.0, steward_cost_usd: float = 2.5):
    from orchestrator.harness import TaskReport

    return TaskReport(
        task_id='t1',
        title='T1',
        outcome=outcome,
        cost_usd=cost_usd,
        steward_cost_usd=steward_cost_usd,
        block_reason='Planning failed: agent API error: HTTP 529',
        block_detail='transient provider overload',
        block_phase='plan',
    )


def _make_harness(config: OrchestratorConfig):
    """Minimal Harness instance for calling _apply_retry_cap directly."""
    from orchestrator.harness import Harness

    harness = Harness.__new__(Harness)
    _init_harness_state_for_test(harness)  # task 1449: initialise digest counters
    harness.config = config
    harness.scheduler = Scheduler(config)
    harness._escalation_queue = None
    harness._run_id = 'run-test'
    return harness


class TestHarnessIntegration:
    """Drive the real _apply_retry_cap helper with a stubbed Scheduler."""

    @pytest.mark.asyncio
    async def test_cap_fires_on_nth_requeue_not_before(
        self, config: OrchestratorConfig,
    ):
        """With requeue_cap=3, cap fires on the 3rd REQUEUED, not the 2nd."""
        from orchestrator.workflow import WorkflowOutcome

        harness = _make_harness(config)
        trigger = AsyncMock()
        harness.scheduler.trigger_retry_cap_exhausted = trigger  # type: ignore[method-assign]

        # After 2 REQUEUEDs: no trigger call.
        requeued_after_1 = await harness._apply_retry_cap(
            't1', _build_report(WorkflowOutcome.REQUEUED), True,
        )
        requeued_after_2 = await harness._apply_retry_cap(
            't1', _build_report(WorkflowOutcome.REQUEUED), True,
        )
        assert requeued_after_1 is True
        assert requeued_after_2 is True
        assert trigger.await_count == 0
        assert harness.scheduler._requeue_counts['t1'] == 2

        # 3rd REQUEUED fires the trigger exactly once and returns requeued=False.
        requeued_after_3 = await harness._apply_retry_cap(
            't1', _build_report(WorkflowOutcome.REQUEUED), True,
        )
        assert requeued_after_3 is False  # cap path suppresses cooldown
        assert trigger.await_count == 1
        call = trigger.await_args
        assert call is not None
        assert call.args == ('t1',)
        assert call.kwargs['run_id'] == 'run-test'
        # cumulative cost summed across all 3 attempts
        assert call.kwargs['cost_usd'] == pytest.approx(3 * (5.0 + 2.5))

    @pytest.mark.asyncio
    async def test_done_clears_counter(self, config: OrchestratorConfig):
        from orchestrator.workflow import WorkflowOutcome

        harness = _make_harness(config)
        harness.scheduler.trigger_retry_cap_exhausted = AsyncMock()  # type: ignore[method-assign]

        await harness._apply_retry_cap(
            't1', _build_report(WorkflowOutcome.REQUEUED), True,
        )
        await harness._apply_retry_cap(
            't1', _build_report(WorkflowOutcome.REQUEUED), True,
        )
        assert harness.scheduler._requeue_counts['t1'] == 2

        await harness._apply_retry_cap(
            't1', _build_report(WorkflowOutcome.DONE), False,
        )
        assert 't1' not in harness.scheduler._requeue_counts

        # Subsequent REQUEUED starts fresh at 1.
        await harness._apply_retry_cap(
            't1', _build_report(WorkflowOutcome.REQUEUED), True,
        )
        assert harness.scheduler._requeue_counts['t1'] == 1

    @pytest.mark.asyncio
    async def test_cap_action_returns_false_even_when_caller_requeued_true(
        self, config: OrchestratorConfig,
    ):
        """When cap fires, _apply_retry_cap returns False so release() skips cooldown."""
        from orchestrator.workflow import WorkflowOutcome

        harness = _make_harness(config)
        harness.scheduler.trigger_retry_cap_exhausted = AsyncMock()  # type: ignore[method-assign]

        # Pre-seed 2 attempts (not yet at cap).
        await harness._apply_retry_cap(
            't1', _build_report(WorkflowOutcome.REQUEUED), True,
        )
        await harness._apply_retry_cap(
            't1', _build_report(WorkflowOutcome.REQUEUED), True,
        )

        # Cap-firing attempt must return False regardless of caller's input.
        result = await harness._apply_retry_cap(
            't1', _build_report(WorkflowOutcome.REQUEUED), True,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_trigger_exception_does_not_leak(
        self, config: OrchestratorConfig,
    ):
        """A raising trigger must not propagate past _apply_retry_cap.

        The finally block in _run_slot must still reach scheduler.release().
        """
        from orchestrator.workflow import WorkflowOutcome

        harness = _make_harness(config)
        harness.scheduler.trigger_retry_cap_exhausted = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError('boom')
        )

        result: bool | None = None
        for _ in range(3):
            result = await harness._apply_retry_cap(
                't1', _build_report(WorkflowOutcome.REQUEUED), True,
            )
        # Cap fired on the 3rd call — returned False despite the exception.
        assert result is False


# --- Harness dual-ceiling integration ----------------------------------------


class TestHarnessDualCeiling:
    """_apply_retry_cap enforces both genuine and transient ceilings."""

    def _make(self, tmp_path):
        from orchestrator.config import OrchestratorConfig
        cfg = OrchestratorConfig(
            project_root=tmp_path,
            max_per_module=1,
            requeue_cap=3,
            transient_requeue_cap=2,
        )
        return _make_harness(cfg)

    @pytest.mark.asyncio
    async def test_transient_does_not_fire_genuine_cap(self, tmp_path):
        from orchestrator.workflow import WorkflowOutcome
        harness = self._make(tmp_path)
        trigger = AsyncMock()
        harness.scheduler.trigger_retry_cap_exhausted = trigger  # type: ignore[method-assign]

        # 1st transient REQUEUED — no trigger, genuine counter stays empty
        r1 = await harness._apply_retry_cap(
            't1', _build_529_report(WorkflowOutcome.REQUEUED), True,
        )
        assert r1 is True
        assert trigger.await_count == 0
        assert 't1' not in harness.scheduler._requeue_counts
        assert harness.scheduler._transient_requeue_counts['t1'] == 1

    @pytest.mark.asyncio
    async def test_transient_cap_fires_at_transient_ceiling(self, tmp_path):
        from orchestrator.workflow import WorkflowOutcome
        harness = self._make(tmp_path)
        trigger = AsyncMock()
        harness.scheduler.trigger_retry_cap_exhausted = trigger  # type: ignore[method-assign]

        # 1st transient — no trigger
        await harness._apply_retry_cap(
            't1', _build_529_report(WorkflowOutcome.REQUEUED), True,
        )
        assert trigger.await_count == 0

        # 2nd transient — hits transient_requeue_cap=2, trigger fires with cap=2
        r2 = await harness._apply_retry_cap(
            't1', _build_529_report(WorkflowOutcome.REQUEUED), True,
        )
        assert r2 is False
        assert trigger.await_count == 1
        call = trigger.await_args
        assert call is not None
        assert call.kwargs.get('cap') == 2

    @pytest.mark.asyncio
    async def test_genuine_cap_unaffected_by_transient_counts(self, tmp_path):
        from orchestrator.workflow import WorkflowOutcome
        harness = self._make(tmp_path)
        trigger = AsyncMock()
        harness.scheduler.trigger_retry_cap_exhausted = trigger  # type: ignore[method-assign]

        # 3 genuine requeues must fire at cap=3 regardless of transient state
        for _ in range(2):
            await harness._apply_retry_cap(
                't2', _build_report(WorkflowOutcome.REQUEUED), True,
            )
        assert trigger.await_count == 0

        r3 = await harness._apply_retry_cap(
            't2', _build_report(WorkflowOutcome.REQUEUED), True,
        )
        assert r3 is False
        assert trigger.await_count == 1
        call = trigger.await_args
        assert call is not None
        assert call.kwargs.get('cap') == 3
