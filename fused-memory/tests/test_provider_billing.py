"""Contract for provider_billing.classify_billing_exhaustion (task 10922).

An exhausted OpenAI credit balance (``insufficient_quota``) must be told
apart from a transient too-many-requests 429, including when graphiti_core
has re-raised the provider error so the structured fields sit only on
``__cause__``.
"""

import asyncio
import dataclasses

import httpx
import openai
import pytest
from _fm_helpers import (
    make_insufficient_quota_error,
    make_transient_rate_limit_error,
    wrap_like_graphiti,
)

from fused_memory.services.provider_billing import (
    BillingExhaustion,
    BillingSource,
    classify_billing_exhaustion,
)


def _assert_api_key_exhaustion(result: BillingExhaustion | None) -> BillingExhaustion:
    assert result is not None
    assert result.source is BillingSource.API_KEY_CREDITS
    assert result.provider == 'openai'
    assert 'OPENAI_API_KEY' in result.key_scope
    return result


class TestExhaustionIsRecognised:
    def test_incident_body_classifies_as_api_key_credits(self):
        result = _assert_api_key_exhaustion(
            classify_billing_exhaustion(make_insufficient_quota_error())
        )
        assert result.error_type == 'RateLimitError'
        assert 'no credits remaining' in result.provider_message

    def test_type_only_body_classifies_as_exhaustion(self):
        exc = make_insufficient_quota_error(with_code=False)
        _assert_api_key_exhaustion(classify_billing_exhaustion(exc))

    def test_graphiti_wrapper_classifies_through_cause(self):
        inner = make_insufficient_quota_error()
        result = _assert_api_key_exhaustion(classify_billing_exhaustion(wrap_like_graphiti(inner)))
        assert result.error_type == type(inner).__name__
        assert 'no credits remaining' in result.provider_message

    def test_unstructured_message_falls_back_to_substring(self):
        exc = Exception('insufficient_quota: no credits left')
        _assert_api_key_exhaustion(classify_billing_exhaustion(exc))


class TestEverythingElseIsNotExhaustion:
    def test_transient_rate_limit_is_not_exhaustion(self):
        assert classify_billing_exhaustion(make_transient_rate_limit_error()) is None

    def test_wrapped_transient_rate_limit_is_not_exhaustion(self):
        wrapped = wrap_like_graphiti(make_transient_rate_limit_error())
        assert classify_billing_exhaustion(wrapped) is None

    def test_server_error_is_not_exhaustion(self):
        request = httpx.Request('POST', 'https://api.openai.com/v1/embeddings')
        response = httpx.Response(500, request=request)
        exc = openai.InternalServerError('server error', response=response, body=None)
        assert classify_billing_exhaustion(exc) is None

    def test_runtime_error_is_not_exhaustion(self):
        assert classify_billing_exhaustion(RuntimeError('boom')) is None

    def test_cancelled_error_is_not_exhaustion(self):
        assert classify_billing_exhaustion(asyncio.CancelledError()) is None

    def test_structured_status_beats_message_quote(self):
        class WrappedError(Exception):
            status_code = 500

        exc = WrappedError('upstream said: insufficient_quota, but this is a 500')
        assert classify_billing_exhaustion(exc) is None

    def test_cyclic_cause_chain_terminates(self):
        first = RuntimeError('first')
        second = RuntimeError('second')
        first.__cause__ = second
        second.__cause__ = first
        assert classify_billing_exhaustion(first) is None


class TestBillingExhaustionValue:
    def test_provider_message_is_bounded(self):
        exc = make_insufficient_quota_error('x' * 10_000)
        result = _assert_api_key_exhaustion(classify_billing_exhaustion(exc))
        assert len(result.provider_message) <= 400

    def test_value_is_frozen(self):
        result = _assert_api_key_exhaustion(
            classify_billing_exhaustion(make_insufficient_quota_error())
        )
        for field in dataclasses.fields(result):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(result, field.name, 'other')
