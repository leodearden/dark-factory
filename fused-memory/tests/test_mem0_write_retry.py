"""Tests for durable server-side retry of transiently failed Mem0 writes (task 7397)."""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import UTC, datetime, timedelta

import httpx
import openai
import pytest

from fused_memory.services.mem0_write_retry import (
    DEFAULT_MEM0_RETRY_POLICY,
    Mem0RetryPolicy,
    TransientProviderFailure,
    classify_transient_provider_failure,
    retry_reason,
)

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
