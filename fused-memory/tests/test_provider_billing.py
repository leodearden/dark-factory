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
    make_usage_limit_reached_error,
    wrap_like_graphiti,
)

from fused_memory.services.provider_billing import (
    BillingExhaustion,
    BillingSource,
    billing_source_of,
    classify_billing_exhaustion,
    is_plan_cap_status,
    plan_cap_of,
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
        assert result.code == 'insufficient_quota'
        assert 'no credits remaining' in result.provider_message

    def test_type_only_body_classifies_as_exhaustion(self):
        exc = make_insufficient_quota_error(with_code=False)
        result = _assert_api_key_exhaustion(classify_billing_exhaustion(exc))
        assert result.code == 'insufficient_quota'

    def test_live_credit_balance_code_with_quota_type_classifies(self):
        exc = make_insufficient_quota_error(code='credit_balance_exhausted')
        for result in (
            classify_billing_exhaustion(exc),
            classify_billing_exhaustion(wrap_like_graphiti(exc)),
        ):
            assert _assert_api_key_exhaustion(result).code == 'credit_balance_exhausted'

    def test_live_credit_balance_code_alone_classifies(self):
        exc = make_insufficient_quota_error(code='credit_balance_exhausted', error_type=None)
        _assert_api_key_exhaustion(classify_billing_exhaustion(exc))

    def test_graphiti_wrapper_classifies_through_cause(self):
        inner = make_insufficient_quota_error()
        result = _assert_api_key_exhaustion(classify_billing_exhaustion(wrap_like_graphiti(inner)))
        assert result.error_type == type(inner).__name__
        assert 'no credits remaining' in result.provider_message

    def test_unstructured_message_falls_back_to_substring(self):
        exc = Exception('insufficient_quota: no credits left')
        result = _assert_api_key_exhaustion(classify_billing_exhaustion(exc))
        assert result.code == 'insufficient_quota'


class TestBillingSourceOf:
    def test_names_the_exhausted_source(self):
        assert billing_source_of(make_insufficient_quota_error()) == 'api_key_credits'

    @pytest.mark.parametrize(
        'exc', [None, make_transient_rate_limit_error(), RuntimeError('boom')],
        ids=['none', 'transient-429', 'other'],
    )
    def test_is_none_for_anything_else(self, exc):
        assert billing_source_of(exc) is None


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


_USAGE_LIMIT_BODY = (
    b'{"error": {"type": "usage_limit_reached", "code": "usage_limit_reached",'
    b' "message": "The usage limit has been reached", "resets_in_seconds": 3600}}'
)


def _assert_plan_cap(result: BillingExhaustion | None) -> BillingExhaustion:
    assert result is not None
    assert result.source is BillingSource.PLAN_CAP
    assert result.provider == 'chatgpt-plan'
    assert 'OPENAI_PLAN_PROXY_URL' in result.key_scope
    assert 'OPENAI_PLAN_PROXY_KEY' in result.key_scope
    assert 'OPENAI_API_KEY' not in result.key_scope
    return result


class TestPlanCapIsRecognised:
    """Task 9944: a capped ChatGPT plan is a second billing source."""

    def test_usage_limit_reached_classifies_as_plan_cap(self):
        result = _assert_plan_cap(classify_billing_exhaustion(make_usage_limit_reached_error()))
        assert result.code == 'usage_limit_reached'
        assert result.error_type == 'RateLimitError'

    def test_graphiti_wrapper_classifies_through_cause(self):
        wrapped = wrap_like_graphiti(make_usage_limit_reached_error())
        assert _assert_plan_cap(classify_billing_exhaustion(wrapped)).code == 'usage_limit_reached'

    def test_names_the_plan_cap_source(self):
        assert billing_source_of(make_usage_limit_reached_error()) == 'plan_cap'

    def test_insufficient_quota_is_still_api_key_credits(self):
        result = classify_billing_exhaustion(make_insufficient_quota_error())
        assert result is not None
        assert result.source is BillingSource.API_KEY_CREDITS

    def test_server_error_from_the_proxy_is_not_a_cap(self):
        request = httpx.Request('POST', 'http://127.0.0.1:8317/v1/responses')
        response = httpx.Response(503, request=request)
        exc = openai.InternalServerError('proxy unavailable', response=response, body=None)
        assert classify_billing_exhaustion(exc) is None

    def test_timeout_is_not_a_cap(self):
        request = httpx.Request('POST', 'http://127.0.0.1:8317/v1/responses')
        assert classify_billing_exhaustion(openai.APITimeoutError(request=request)) is None


class TestPlanCapOfAProxyResponse:
    """The plan-route rule: any 429 from the plan proxy is a cap."""

    def test_usage_limit_body_names_its_code(self):
        result = _assert_plan_cap(plan_cap_of(_USAGE_LIMIT_BODY))
        assert result.code == 'usage_limit_reached'
        assert result.error_type == 'HTTP 429'
        assert 'The usage limit has been reached' in result.provider_message

    def test_top_level_type_names_its_code(self):
        body = b'{"type": "usage_limit_reached", "message": "capped"}'
        assert _assert_plan_cap(plan_cap_of(body)).code == 'usage_limit_reached'

    def test_bodyless_429_is_a_cap(self):
        assert _assert_plan_cap(plan_cap_of(b'')).code == 'http_429'

    def test_unparseable_429_is_a_cap(self):
        result = _assert_plan_cap(plan_cap_of(b'<html>slow down</html>'))
        assert result.code == 'http_429'
        assert 'slow down' in result.provider_message

    def test_provider_message_is_bounded(self):
        assert len(_assert_plan_cap(plan_cap_of(b'x' * 10_000)).provider_message) <= 400

    def test_a_429_is_a_cap(self):
        assert is_plan_cap_status(429)

    @pytest.mark.parametrize('status', [200, 400, 500, 503])
    def test_any_other_status_is_not_a_cap(self, status):
        assert not is_plan_cap_status(status)


class TestEachSourceNamesItsOwnAlarm:
    def test_api_key_credits_alarm_category(self):
        result = classify_billing_exhaustion(make_insufficient_quota_error())
        assert result is not None
        assert result.alarm_category == 'provider_billing_exhausted'

    def test_plan_cap_alarm_category(self):
        result = _assert_plan_cap(classify_billing_exhaustion(make_usage_limit_reached_error()))
        assert result.alarm_category == 'plan_route_capped'
