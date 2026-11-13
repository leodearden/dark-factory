"""Unit tests for the startup-step runner (orchestrator/src/orchestrator/startup_steps.py).

Driven through the module's public API against a real EscalationQueue and a
real EventStore, so the filed L1 and the emitted event are read back exactly
as an operator would see them.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.event_store import EventStore, EventType
from orchestrator.startup_steps import (
    EntryFailure,
    EntryFailures,
    StartupJournal,
    StartupStep,
    StepOutcome,
    StepStatus,
    run_startup_step,
    sentinel_task_id,
)

RUN_ID = 'run-x'
CLAIMANT = 'claimant-run-x'
LOGGER_NAME = 'orchestrator.startup_steps'


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'escalations')


@pytest.fixture
def event_store(tmp_path: Path) -> EventStore:
    return EventStore(tmp_path / 'runs.db', RUN_ID)


@pytest.fixture
def journal() -> StartupJournal:
    return StartupJournal()


async def _run(
    step: StartupStep,
    journal: StartupJournal,
    event_store: EventStore | None,
    queue: EscalationQueue | None,
) -> StepOutcome:
    return await run_startup_step(
        step,
        journal=journal,
        event_store=event_store,
        escalation_queue=queue,
        filing_claimant_run_id=CLAIMANT,
    )


def _pending_l1s(queue: EscalationQueue, step_name: str) -> list[Escalation]:
    return queue.get_by_task(sentinel_task_id(step_name), status='pending', level=1)


def _failed_events(event_store: EventStore) -> list[dict]:
    return event_store.fetch_events_by_type(EventType.startup_step_failed)


def _raiser(exc: BaseException):
    async def run() -> None:
        raise exc
    return run


class TestStartupStepName:
    @pytest.mark.parametrize('bad', ['Diff 7', '', '2C', '-2c', '2c_x'])
    def test_rejects_non_slug_name(self, bad: str) -> None:
        with pytest.raises(ValueError, match=repr(bad)):
            StartupStep(bad, lambda: None)

    @pytest.mark.parametrize('good', ['1b', '1c0-dirty', '1c0-config-keys', '2c2', 'diff7'])
    def test_accepts_slug_name(self, good: str) -> None:
        assert StartupStep(good, lambda: None).name == good

    def test_sentinel_task_id_carries_step_name(self) -> None:
        assert sentinel_task_id('2c') == 'orchestrator-startup-2c'


class TestHealthyStep:
    @pytest.mark.asyncio
    async def test_none_result_is_ok_and_files_nothing(
        self, journal, event_store, queue,
    ) -> None:
        seen_in_flight = []

        async def body() -> None:
            seen_in_flight.append(journal.in_flight)

        outcome = await _run(StartupStep('2c', body), journal, event_store, queue)

        assert outcome.status is StepStatus.OK
        assert outcome.step == '2c'
        assert outcome.failed_entries == ()
        assert _failed_events(event_store) == []
        assert queue.get_pending() == []
        assert seen_in_flight[0] is not None
        assert seen_in_flight[0].step == '2c'
        assert seen_in_flight[0].started_at is not None
        assert journal.in_flight is not None
        assert journal.in_flight.step == '2c'

    @pytest.mark.asyncio
    async def test_sync_callable_is_supported(self, journal, event_store, queue) -> None:
        calls = []
        outcome = await _run(
            StartupStep('1c0', lambda: calls.append('ran') or 'incidental-str'),
            journal, event_store, queue,
        )
        assert calls == ['ran']
        assert outcome.status is StepStatus.OK

    @pytest.mark.asyncio
    async def test_sync_callable_raising_is_failed(self, journal, event_store, queue) -> None:
        def body() -> None:
            raise ValueError('sync boom')

        outcome = await _run(StartupStep('1c0b', body), journal, event_store, queue)
        assert outcome.status is StepStatus.FAILED
        assert outcome.exception_type == 'ValueError'


class TestFailedStep:
    @pytest.mark.asyncio
    async def test_raise_yields_failed_outcome_log_event_and_l1(
        self, journal, event_store, queue, caplog,
    ) -> None:
        with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
            outcome = await _run(
                StartupStep('2c', _raiser(RuntimeError('injected'))),
                journal, event_store, queue,
            )

        assert outcome.status is StepStatus.FAILED
        assert outcome.exception_type == 'RuntimeError'
        assert outcome.message == 'injected'

        errors = [
            r for r in caplog.records
            if r.name == LOGGER_NAME and r.levelno == logging.ERROR
            and 'startup step 2c failed' in r.getMessage()
        ]
        assert len(errors) == 1
        assert errors[0].exc_info is not None

        events = _failed_events(event_store)
        assert len(events) == 1
        data = events[0]['data']
        assert data['step'] == '2c'
        assert data['status'] == 'failed'
        assert data['exception_type'] == 'RuntimeError'
        assert data['message'] == 'injected'

        l1s = _pending_l1s(queue, '2c')
        assert len(l1s) == 1
        esc = l1s[0]
        assert esc.level == 1
        assert esc.severity == 'blocking'
        assert esc.category == 'infra_issue'
        assert esc.agent_role == 'orchestrator-startup'
        assert esc.filing_claimant_run_id == CLAIMANT
        assert esc.task_id == sentinel_task_id('2c')
        assert esc.suggested_action == 'investigate_startup_step'
        assert '2c' in esc.summary
        assert 'RuntimeError' in esc.summary

    @pytest.mark.asyncio
    async def test_repeat_failure_refreshes_the_same_l1(
        self, journal, event_store, queue,
    ) -> None:
        step = StartupStep('2e', _raiser(RuntimeError('first')))
        await _run(step, journal, event_store, queue)
        [first] = _pending_l1s(queue, '2e')

        second_step = StartupStep('2e', _raiser(RuntimeError('second')))
        await _run(second_step, journal, event_store, queue)

        l1s = _pending_l1s(queue, '2e')
        assert len(l1s) == 1
        refreshed = l1s[0]
        assert refreshed.id == first.id
        assert refreshed.dedupe_count == 1
        assert 'second' in refreshed.detail
        assert refreshed.timestamp >= first.timestamp
        assert len(queue.get_pending()) == 1

    @pytest.mark.asyncio
    async def test_repeat_failure_refreshes_summary_and_keeps_the_first_filer(
        self, journal, event_store, queue,
    ) -> None:
        for claimant, message in (('claimant-run-a', 'first'), ('claimant-run-b', 'second')):
            await run_startup_step(
                StartupStep('2f', _raiser(RuntimeError(message))),
                journal=journal,
                event_store=event_store,
                escalation_queue=queue,
                filing_claimant_run_id=claimant,
            )

        [refreshed] = _pending_l1s(queue, '2f')
        assert 'second' in refreshed.summary
        assert 'first' not in refreshed.summary
        assert refreshed.filing_claimant_run_id == 'claimant-run-a'


class TestPartialStep:
    @pytest.mark.asyncio
    async def test_returned_entry_failures_yield_partial(
        self, journal, event_store, queue,
    ) -> None:
        async def body() -> EntryFailures:
            failures = EntryFailures('Recovery')
            failures.record('41', OSError('boom'))
            return failures

        outcome = await _run(StartupStep('2c', body), journal, event_store, queue)

        assert outcome.status is StepStatus.PARTIAL
        assert outcome.failed_entries == (EntryFailure('41', 'OSError', 'boom'),)

        events = _failed_events(event_store)
        assert len(events) == 1
        data = events[0]['data']
        assert data['status'] == 'partial'
        assert data['failed_entries'] == [
            {'entry': '41', 'exception_type': 'OSError', 'message': 'boom'},
        ]

        [esc] = _pending_l1s(queue, '2c')
        assert '41' in esc.summary or '41' in esc.detail

    @pytest.mark.asyncio
    async def test_empty_entry_failures_is_ok(self, journal, event_store, queue) -> None:
        async def body() -> EntryFailures:
            return EntryFailures('Recovery')

        outcome = await _run(StartupStep('2c', body), journal, event_store, queue)

        assert outcome.status is StepStatus.OK
        assert _failed_events(event_store) == []
        assert queue.get_pending() == []

    def test_record_logs_error_naming_sweep_and_entry(self, caplog) -> None:
        failures = EntryFailures('Orphan reaper')
        with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
            failures.record('500', OSError('boom'))

        [record] = [r for r in caplog.records if r.name == LOGGER_NAME]
        assert record.levelno == logging.ERROR
        assert 'Orphan reaper' in record.getMessage()
        assert '500' in record.getMessage()
        assert record.exc_info is not None
        assert failures.failures == (EntryFailure('500', 'OSError', 'boom'),)


class TestFatalStep:
    @pytest.mark.asyncio
    async def test_fatal_reraises_the_same_exception_after_reporting(
        self, journal, event_store, queue,
    ) -> None:
        exc = RuntimeError('liveness over budget')
        with pytest.raises(RuntimeError) as raised:
            await _run(StartupStep('1b', _raiser(exc), fatal=True), journal, event_store, queue)

        assert raised.value is exc
        [event] = _failed_events(event_store)
        assert event['data']['step'] == '1b'
        assert len(_pending_l1s(queue, '1b')) == 1


class TestBaseExceptionPassthrough:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'exc', [asyncio.CancelledError(), SystemExit(3), KeyboardInterrupt()],
        ids=['cancelled', 'system-exit', 'keyboard-interrupt'],
    )
    async def test_base_exception_propagates_untouched(
        self, exc, journal, event_store, queue,
    ) -> None:
        with pytest.raises(type(exc)) as raised:
            await _run(StartupStep('2d', _raiser(exc)), journal, event_store, queue)

        assert raised.value is exc
        assert _failed_events(event_store) == []
        assert queue.get_pending() == []
        assert journal.in_flight is not None
        assert journal.in_flight.step == '2d'


class TestNoneSafety:
    @pytest.mark.asyncio
    async def test_failure_without_store_or_queue_returns_outcome(self, journal) -> None:
        outcome = await _run(
            StartupStep('1c', _raiser(RuntimeError('no sinks'))), journal, None, None,
        )
        assert outcome.status is StepStatus.FAILED
        assert outcome.message == 'no sinks'


class _SubmitFailsQueue(EscalationQueue):
    def submit(self, escalation: Escalation) -> str:
        raise OSError('disk full')


class TestFilingIsBestEffort:
    @pytest.mark.asyncio
    async def test_filing_fault_keeps_non_fatal_step_non_fatal(
        self, tmp_path, journal, event_store, caplog,
    ) -> None:
        queue = _SubmitFailsQueue(tmp_path / 'escalations')
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            outcome = await _run(
                StartupStep('2c', _raiser(RuntimeError('injected'))),
                journal, event_store, queue,
            )

        assert outcome.status is StepStatus.FAILED
        assert any(
            r.levelno == logging.WARNING and r.name == LOGGER_NAME
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_filing_fault_does_not_mask_fatal_exception(
        self, tmp_path, journal, event_store,
    ) -> None:
        queue = _SubmitFailsQueue(tmp_path / 'escalations')
        exc = RuntimeError('original')
        with pytest.raises(RuntimeError) as raised:
            await _run(StartupStep('1b', _raiser(exc), fatal=True), journal, event_store, queue)
        assert raised.value is exc
