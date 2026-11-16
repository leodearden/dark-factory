"""Tests for durable server-side retry of transiently failed Mem0 writes (task 7397)."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import openai
import pytest
import pytest_asyncio
from _mem0_intent_fixtures import (
    EMBEDDINGS_REQUEST,
    mem0_intent_row,
    openai_status_error,
    rate_limit_error,
    seed_mem0_intent,
)

from fused_memory.services.mem0_add_outcome import Mem0AddOutcome
from fused_memory.services.mem0_write_retry import (
    DEFAULT_MEM0_RETRY_POLICY,
    Mem0RetryPassSummary,
    Mem0RetryPolicy,
    TransientProviderFailure,
    classify_transient_provider_failure,
    retry_due_mem0_intents,
    retry_reason,
    run_periodic_mem0_retry,
)
from fused_memory.services.write_journal import Mem0IntentStatus, WriteJournal


class TestClassifyTransientProviderFailure:
    @pytest.mark.parametrize(
        ('error', 'expected'),
        [
            (rate_limit_error(), TransientProviderFailure.RATE_LIMITED),
            (
                openai_status_error(openai.InternalServerError, 500),
                TransientProviderFailure.SERVER_ERROR,
            ),
            (
                openai_status_error(openai.APIStatusError, 503),
                TransientProviderFailure.SERVER_ERROR,
            ),
            (
                openai.APIConnectionError(request=EMBEDDINGS_REQUEST),
                TransientProviderFailure.CONNECTION,
            ),
            (
                openai.APITimeoutError(request=EMBEDDINGS_REQUEST),
                TransientProviderFailure.CONNECTION,
            ),
        ],
        ids=['429', '500', '503', 'connection', 'sdk-timeout'],
    )
    def test_openai_transient_errors_are_classified(self, error, expected):
        assert classify_transient_provider_failure(error) is expected

    @pytest.mark.parametrize(
        'error',
        [
            TimeoutError('Mem0 add timed out after 30.0s'),
            openai_status_error(openai.AuthenticationError, 401),
            openai_status_error(openai.BadRequestError, 400),
            RuntimeError(
                "Error code: 429 - {'error': {'message': 'You have no credits remaining'}}"
            ),
            ValueError('bad vector'),
            asyncio.CancelledError(),
        ],
        ids=['write-budget-timeout', '401', '400', '429-text-only', 'value-error', 'cancelled'],
    )
    def test_everything_else_is_not_transient(self, error):
        assert classify_transient_provider_failure(error) is None


class TestMem0RetryPolicy:
    _POLICY = Mem0RetryPolicy(
        initial_delay_seconds=30,
        max_delay_seconds=900,
        poll_interval_seconds=30,
        batch_size=50,
    )

    def test_is_frozen(self):
        for field in dataclasses.fields(self._POLICY):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(self._POLICY, field.name, 1)

    def test_backoff_doubles_from_the_initial_delay_and_is_capped(self):
        now = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
        assert self._POLICY.retry_at(attempts=1, now=now) == now + timedelta(seconds=30)
        assert self._POLICY.retry_at(attempts=2, now=now) == now + timedelta(seconds=60)
        assert self._POLICY.retry_at(attempts=3, now=now) == now + timedelta(seconds=120)
        assert self._POLICY.retry_at(attempts=20, now=now) == now + timedelta(seconds=900)

    def test_backoff_stays_capped_after_unbounded_attempts(self):
        now = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
        assert self._POLICY.retry_at(attempts=5000, now=now) == now + timedelta(seconds=900)

    def test_default_policy_exists(self):
        assert isinstance(DEFAULT_MEM0_RETRY_POLICY, Mem0RetryPolicy)

    def test_the_default_backlog_alarm_age_is_an_hour(self):
        assert DEFAULT_MEM0_RETRY_POLICY.backlog_alarm_age_seconds == 3600

    def test_the_default_backlog_is_checked_every_five_minutes(self):
        assert DEFAULT_MEM0_RETRY_POLICY.backlog_check_interval_seconds == 300

    def test_the_backlog_is_never_checked_more_often_than_a_retry_pass_runs(self):
        policy = Mem0RetryPolicy(
            initial_delay_seconds=30,
            max_delay_seconds=900,
            poll_interval_seconds=30,
            batch_size=50,
            backlog_alarm_age_seconds=60,
        )

        assert policy.backlog_check_interval_seconds == 30

    @pytest.mark.parametrize('age', [0, -1])
    def test_a_non_positive_backlog_alarm_age_is_rejected(self, age):
        with pytest.raises(ValueError, match='backlog_alarm_age_seconds'):
            Mem0RetryPolicy(
                initial_delay_seconds=30,
                max_delay_seconds=900,
                poll_interval_seconds=30,
                batch_size=50,
                backlog_alarm_age_seconds=age,
            )

    def test_initial_delay_above_the_cap_is_rejected(self):
        with pytest.raises(ValueError, match=r'901') as excinfo:
            Mem0RetryPolicy(
                initial_delay_seconds=901,
                max_delay_seconds=900,
                poll_interval_seconds=30,
                batch_size=50,
            )
        assert '900' in str(excinfo.value)


def test_retry_reason_names_the_failure_and_the_exception_type():
    reason = retry_reason(TransientProviderFailure.RATE_LIMITED, rate_limit_error())
    assert reason.startswith(TransientProviderFailure.RATE_LIMITED.value)
    assert 'RateLimitError' in reason


_T0 = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
_NOW = _T0 + timedelta(seconds=1)
_PASS_POLICY = Mem0RetryPolicy(
    initial_delay_seconds=10,
    max_delay_seconds=100,
    poll_interval_seconds=1,
    batch_size=2,
)


@pytest_asyncio.fixture
async def journal(tmp_path):
    j = WriteJournal(tmp_path / 'wj')
    await j.initialize()
    yield j
    await j.close()


async def _seed_retrying(
    journal: WriteJournal,
    *,
    retry_at: datetime = _T0,
    content: str = 'Never run git stash in a dark-factory checkout. ' * 8,
    metadata: dict | None = None,
) -> tuple[str, str]:
    write_op_id = str(uuid.uuid4())
    intent_id = await seed_mem0_intent(
        journal,
        status=Mem0IntentStatus.RETRYING,
        reason='rate_limited: RateLimitError: 429',
        retry_at=retry_at,
        write_op_id=write_op_id,
        content=content,
        metadata=metadata if metadata is not None else {'category': 'procedural_knowledge'},
        project_id='dark_factory',
        agent_id='agent-a',
        session_id='sess-1',
        category='procedural_knowledge',
    )
    return intent_id, write_op_id


_LANDED_LATE = Mem0AddOutcome(memory_ids=('mem0-late',), empty_result_error=None)


class _Reissue:
    def __init__(self, error: BaseException | None = None) -> None:
        self.calls: list[dict] = []
        self._error = error

    async def __call__(self, intent: dict) -> Mem0AddOutcome:
        self.calls.append(intent)
        if self._error is not None:
            raise self._error
        return _LANDED_LATE


class _StampedElsewhere(_Reissue):
    """A re-issue during which another writer stamps the intent out of ``retrying``."""

    def __init__(self, journal: WriteJournal, error: BaseException | None = None) -> None:
        super().__init__(error)
        self._journal = journal

    async def __call__(self, intent: dict) -> Mem0AddOutcome:
        await self._journal.resolve_mem0_intent(
            intent['id'], Mem0IntentStatus.FAILED, reason='stamped elsewhere'
        )
        return await super().__call__(intent)


class _OnLanded:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[Mapping[str, Any], Mem0AddOutcome]] = []
        self._error = error

    async def __call__(self, intent: Mapping[str, Any], outcome: Mem0AddOutcome) -> None:
        self.calls.append((intent, outcome))
        if self._error is not None:
            raise self._error


async def _run_pass(
    journal: WriteJournal, reissue: _Reissue, on_landed: _OnLanded | None = None
) -> Mem0RetryPassSummary:
    hook = {} if on_landed is None else {'on_landed': on_landed}
    return await retry_due_mem0_intents(
        journal, reissue, policy=_PASS_POLICY, now=_NOW, **hook
    )


class TestRetryDueMem0Intents:
    @pytest.mark.asyncio
    async def test_a_landed_reissue_completes_the_intent(self, journal):
        content = 'the full content, longer than the write_ops prefix. ' * 6
        metadata = {'category': 'procedural_knowledge', 'parent_id': 'p', 'n': [1, 2]}
        intent_id, _ = await _seed_retrying(journal, content=content, metadata=metadata)
        reissue = _Reissue()

        summary = await _run_pass(journal, reissue)

        (call,) = reissue.calls
        assert call['id'] == intent_id
        assert call['content'] == content
        assert json.loads(call['metadata']) == metadata
        row = await mem0_intent_row(journal, intent_id)
        assert row['status'] == 'completed'
        assert 'attempt 2' in row['reason']
        assert summary == Mem0RetryPassSummary(due=1, landed=1)

    @pytest.mark.asyncio
    async def test_a_retry_not_yet_due_is_left_alone(self, journal):
        intent_id, _ = await _seed_retrying(journal, retry_at=_NOW + timedelta(seconds=1))
        reissue = _Reissue()

        summary = await _run_pass(journal, reissue)

        assert reissue.calls == []
        assert (await mem0_intent_row(journal, intent_id))['status'] == 'retrying'
        assert summary == Mem0RetryPassSummary()

    @pytest.mark.asyncio
    async def test_a_transient_failure_reschedules_with_backoff(self, journal):
        intent_id, _ = await _seed_retrying(journal)

        summary = await _run_pass(journal, _Reissue(rate_limit_error()))

        row = await mem0_intent_row(journal, intent_id)
        assert row['status'] == 'retrying'
        assert row['attempts'] == 2
        assert row['next_attempt_at'] == _PASS_POLICY.retry_at(
            attempts=2, now=_NOW
        ).isoformat(timespec='microseconds')
        assert summary == Mem0RetryPassSummary(due=1, rescheduled=1)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'error',
        [RuntimeError('qdrant schema mismatch'), TimeoutError('Mem0 add timed out')],
        ids=['runtime-error', 'write-budget-timeout'],
    )
    async def test_a_non_transient_failure_dead_letters_loudly(
        self, journal, caplog, error
    ):
        intent_id, write_op_id = await _seed_retrying(journal)

        with caplog.at_level(logging.ERROR):
            summary = await _run_pass(journal, _Reissue(error))

        row = await mem0_intent_row(journal, intent_id)
        assert row['status'] == 'dead'
        assert type(error).__name__ in row['reason']
        assert 'attempt 2' in row['reason']
        assert any(
            r.levelno == logging.ERROR and write_op_id in r.getMessage()
            for r in caplog.records
        )
        assert summary == Mem0RetryPassSummary(due=1, dead_lettered=1)

    @pytest.mark.asyncio
    async def test_a_retry_that_already_landed_is_reconciled_not_reissued(self, journal):
        intent_id, write_op_id = await _seed_retrying(journal)
        await journal.log_backend_op(
            write_op_id=write_op_id, backend='mem0', operation='add', success=True
        )
        reissue = _Reissue()

        summary = await _run_pass(journal, reissue)

        assert reissue.calls == []
        assert (await mem0_intent_row(journal, intent_id))['status'] == 'completed'
        assert summary == Mem0RetryPassSummary(due=1, reconciled=1)

    @pytest.mark.asyncio
    async def test_a_failed_backend_op_does_not_count_as_landed(self, journal):
        intent_id, write_op_id = await _seed_retrying(journal)
        await journal.log_backend_op(
            write_op_id=write_op_id,
            backend='mem0',
            operation='add',
            success=False,
            error='429',
        )
        reissue = _Reissue()

        summary = await _run_pass(journal, reissue)

        assert [c['id'] for c in reissue.calls] == [intent_id]
        assert summary == Mem0RetryPassSummary(due=1, landed=1)

    @pytest.mark.asyncio
    async def test_one_pass_handles_at_most_a_batch(self, journal):
        for _ in range(3):
            await _seed_retrying(journal)
        reissue = _Reissue()

        summary = await _run_pass(journal, reissue)

        assert len(reissue.calls) == _PASS_POLICY.batch_size == 2
        assert summary.due == 2
        assert len(await journal.get_mem0_intents(status='retrying')) == 1

    @pytest.mark.asyncio
    async def test_cancellation_propagates_and_leaves_the_row_retrying(self, journal):
        intent_id, _ = await _seed_retrying(journal)

        with pytest.raises(asyncio.CancelledError):
            await _run_pass(journal, _Reissue(asyncio.CancelledError()))

        assert (await mem0_intent_row(journal, intent_id))['status'] == 'retrying'

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('error', 'target_status'),
        [
            (None, Mem0IntentStatus.COMPLETED),
            (rate_limit_error(), Mem0IntentStatus.RETRYING),
            (RuntimeError('qdrant schema mismatch'), Mem0IntentStatus.DEAD),
        ],
        ids=['landed', 'rescheduled', 'dead-lettered'],
    )
    async def test_a_stamp_that_misses_is_counted_unstamped_and_warned(
        self, journal, caplog, error, target_status
    ):
        intent_id, write_op_id = await _seed_retrying(journal)

        with caplog.at_level(logging.WARNING):
            summary = await _run_pass(journal, _StampedElsewhere(journal, error))

        assert (await mem0_intent_row(journal, intent_id))['status'] == 'failed'
        assert summary == Mem0RetryPassSummary(due=1, unstamped=1)
        assert any(
            r.levelno == logging.WARNING
            and intent_id in r.getMessage()
            and repr(target_status.value) in r.getMessage()
            and write_op_id in r.getMessage()
            for r in caplog.records
        )



class TestTheLandingHook:
    """``on_landed`` is told about each re-issue that lands, so a deferred
    write's follow-on work (its supersedes marks) runs once the id exists."""

    @pytest.mark.asyncio
    async def test_a_landed_reissue_hands_the_intent_and_outcome_to_the_hook(self, journal):
        intent_id, _ = await _seed_retrying(journal)
        on_landed = _OnLanded()

        summary = await _run_pass(journal, _Reissue(), on_landed)

        ((intent, outcome),) = on_landed.calls
        assert intent['id'] == intent_id
        assert outcome == _LANDED_LATE
        assert (await mem0_intent_row(journal, intent_id))['status'] == 'completed'
        assert summary == Mem0RetryPassSummary(due=1, landed=1)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'error',
        [rate_limit_error(), RuntimeError('qdrant schema mismatch')],
        ids=['rescheduled', 'dead-lettered'],
    )
    async def test_a_reissue_that_did_not_land_is_not_reported(self, journal, error):
        await _seed_retrying(journal)
        on_landed = _OnLanded()

        await _run_pass(journal, _Reissue(error), on_landed)

        assert on_landed.calls == []

    @pytest.mark.asyncio
    async def test_a_reconciled_intent_is_not_reported(self, journal):
        _, write_op_id = await _seed_retrying(journal)
        await journal.log_backend_op(
            write_op_id=write_op_id, backend='mem0', operation='add', success=True
        )
        on_landed = _OnLanded()

        summary = await _run_pass(journal, _Reissue(), on_landed)

        assert on_landed.calls == []
        assert summary == Mem0RetryPassSummary(due=1, reconciled=1)

    @pytest.mark.asyncio
    async def test_a_raising_hook_is_logged_and_the_write_still_counts_landed(
        self, journal, caplog
    ):
        intent_id, write_op_id = await _seed_retrying(journal)

        with caplog.at_level(logging.ERROR):
            summary = await _run_pass(
                journal, _Reissue(), _OnLanded(RuntimeError('mark failed'))
            )

        assert summary == Mem0RetryPassSummary(due=1, landed=1)
        assert (await mem0_intent_row(journal, intent_id))['status'] == 'completed'
        assert any(
            r.levelno == logging.ERROR
            and intent_id in r.getMessage()
            and write_op_id in r.getMessage()
            and r.exc_info is not None
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_a_landed_reissue_whose_stamp_missed_is_still_reported(self, journal):
        """The record exists whatever the journal row says, so its marks are due."""
        intent_id, _ = await _seed_retrying(journal)
        on_landed = _OnLanded()

        summary = await _run_pass(journal, _StampedElsewhere(journal), on_landed)

        assert [intent['id'] for intent, _ in on_landed.calls] == [intent_id]
        assert summary == Mem0RetryPassSummary(due=1, unstamped=1)

@pytest.mark.asyncio
async def test_periodic_loop_survives_a_failing_pass_and_ends_on_cancel(caplog):
    calls = 0
    third_call = asyncio.Event()

    async def pass_fn() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            third_call.set()
        if calls == 1:
            raise RuntimeError('journal locked')

    with caplog.at_level(logging.ERROR):
        task = asyncio.create_task(
            run_periodic_mem0_retry(pass_fn, interval_seconds=0.01, what='retry pass')
        )
        # A genuine hang is bounded by the suite's per-test timeout.
        await third_call.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert calls >= 3
    assert any(
        r.exc_info is not None and 'journal locked' in str(r.exc_info[1])
        for r in caplog.records
    )
