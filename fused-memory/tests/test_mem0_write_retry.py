"""Tests for durable server-side retry of transiently failed Mem0 writes (task 7397)."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import openai
import pytest
import pytest_asyncio

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
from fused_memory.services.write_journal import WriteJournal

_EMBEDDINGS_REQUEST = httpx.Request('POST', 'https://api.openai.com/v1/embeddings')


def _status_error(cls: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    return cls(
        f'Error code: {status} - no credits',
        response=httpx.Response(status, request=_EMBEDDINGS_REQUEST),
        body=None,
    )


def rate_limit_error() -> openai.RateLimitError:
    error = _status_error(openai.RateLimitError, 429)
    assert isinstance(error, openai.RateLimitError)
    return error


class TestClassifyTransientProviderFailure:
    @pytest.mark.parametrize(
        ('error', 'expected'),
        [
            (rate_limit_error(), TransientProviderFailure.RATE_LIMITED),
            (
                _status_error(openai.InternalServerError, 500),
                TransientProviderFailure.SERVER_ERROR,
            ),
            (
                _status_error(openai.APIStatusError, 503),
                TransientProviderFailure.SERVER_ERROR,
            ),
            (
                openai.APIConnectionError(request=_EMBEDDINGS_REQUEST),
                TransientProviderFailure.CONNECTION,
            ),
            (
                openai.APITimeoutError(request=_EMBEDDINGS_REQUEST),
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
            _status_error(openai.AuthenticationError, 401),
            _status_error(openai.BadRequestError, 400),
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
    intent_id, write_op_id = str(uuid.uuid4()), str(uuid.uuid4())
    await journal.log_mem0_intent(
        intent_id=intent_id,
        write_op_id=write_op_id,
        project_id='dark_factory',
        agent_id='agent-a',
        session_id='sess-1',
        category='procedural_knowledge',
        content=content,
        metadata=metadata if metadata is not None else {'category': 'procedural_knowledge'},
        payload_digest='d',
    )
    assert await journal.schedule_mem0_retry(
        intent_id, reason='rate_limited: RateLimitError: 429', retry_at=retry_at
    )
    return intent_id, write_op_id


async def _intent(journal: WriteJournal, intent_id: str) -> dict:
    (row,) = [r for r in await journal.get_mem0_intents() if r['id'] == intent_id]
    return row


class _Reissue:
    def __init__(self, error: BaseException | None = None) -> None:
        self.calls: list[dict] = []
        self._error = error

    async def __call__(self, intent: dict) -> object:
        self.calls.append(intent)
        if self._error is not None:
            raise self._error
        return {'results': [{'id': 'mem0-late'}]}


async def _run_pass(journal: WriteJournal, reissue: _Reissue) -> Mem0RetryPassSummary:
    return await retry_due_mem0_intents(journal, reissue, policy=_PASS_POLICY, now=_NOW)


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
        row = await _intent(journal, intent_id)
        assert row['status'] == 'completed'
        assert 'attempt 2' in row['reason']
        assert summary == Mem0RetryPassSummary(due=1, landed=1)

    @pytest.mark.asyncio
    async def test_a_retry_not_yet_due_is_left_alone(self, journal):
        intent_id, _ = await _seed_retrying(journal, retry_at=_NOW + timedelta(seconds=1))
        reissue = _Reissue()

        summary = await _run_pass(journal, reissue)

        assert reissue.calls == []
        assert (await _intent(journal, intent_id))['status'] == 'retrying'
        assert summary == Mem0RetryPassSummary()

    @pytest.mark.asyncio
    async def test_a_transient_failure_reschedules_with_backoff(self, journal):
        intent_id, _ = await _seed_retrying(journal)

        summary = await _run_pass(journal, _Reissue(rate_limit_error()))

        row = await _intent(journal, intent_id)
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

        row = await _intent(journal, intent_id)
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
        assert (await _intent(journal, intent_id))['status'] == 'completed'
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

        assert (await _intent(journal, intent_id))['status'] == 'retrying'


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
        task = asyncio.create_task(run_periodic_mem0_retry(pass_fn, interval_seconds=0.01))
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
