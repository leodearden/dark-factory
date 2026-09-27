"""Tests for the Z.ai GLM Coding Plan endpoint probe (task 5384).

GLM access is a GLM Coding Plan, and a Coding Plan key answers only on the
Plan's coding endpoints: the general ``/api/paas/v4`` errors. Without a probe,
a wrong endpoint or key degrades into a 4xx on EVERY eval cell rather than one
loud failure (INV-11).

Hermetic: every request goes to an ``httpx.MockTransport``, never the network,
and no real ZAI_API_KEY is used. Per-test local imports (the sibling eval
suites' convention) so an absent symbol fails the one test that needs it, not
collection of the file.
"""

from __future__ import annotations

import json
import math

import httpx
import pytest


class _Recorder:
    """An httpx handler answering every request with one canned reply, keeping the requests."""

    def __init__(self, status_code: int = 200, **reply) -> None:
        self.status_code = status_code
        self.reply = reply or {'json': {'choices': [{'message': {'content': 'pong'}}]}}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status_code, **self.reply)


def _require(handler, **kwargs):
    """Run the probe against *handler* through a client with no timeout of its own."""
    from orchestrator.evals.zai_probe import require_zai_coding_endpoint

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=None)
    return require_zai_coding_endpoint(client=client, **kwargs)


class TestZaiCodingPlanConstants:
    def test_the_coding_base_url_is_the_openai_protocol_coding_endpoint(self):
        from orchestrator.evals.configs import ZAI_CODING_BASE_URL

        assert ZAI_CODING_BASE_URL == 'https://api.z.ai/api/coding/paas/v4'

    def test_the_glm_base_url_is_the_anthropic_protocol_coding_endpoint(self):
        from orchestrator.evals.configs import GLM_BASE_URL

        assert GLM_BASE_URL == 'https://api.z.ai/api/anthropic'


class TestRequireZaiCodingEndpointAnswers:
    def test_a_2xx_answer_returns_an_ok_outcome_for_the_coding_endpoint(self):
        from orchestrator.evals import configs, zai_probe

        outcome = _require(_Recorder(200), auth_token='tk-test')

        assert isinstance(outcome, zai_probe.ZaiProbeOutcome)
        assert outcome.base_url == configs.ZAI_CODING_BASE_URL
        assert outcome.status_code == 200
        assert outcome.ok is True

    def test_the_probe_is_one_minimal_openai_protocol_chat_completion(self):
        """The mock client sets no timeout, so a finite one can only come from the probe."""
        from orchestrator.evals.configs import ZAI_CODING_BASE_URL

        recorder = _Recorder(200)
        _require(recorder, auth_token='tk-test')

        [request] = recorder.requests
        assert request.method == 'POST'
        assert str(request.url) == f'{ZAI_CODING_BASE_URL}/chat/completions'
        assert request.headers['Authorization'] == 'Bearer tk-test'
        body = json.loads(request.content)
        assert body['model'] == 'glm-5.3-flash'
        assert body['max_tokens'] == 1
        [message] = body['messages']
        assert message['role'] == 'user'
        timeouts = request.extensions['timeout'].values()
        assert all(t is not None and math.isfinite(t) and t > 0 for t in timeouts)

    def test_the_token_defaults_to_the_zai_api_key_env_var(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv('ZAI_API_KEY', 'tk-env')
        recorder = _Recorder(200)

        _require(recorder)

        [request] = recorder.requests
        assert request.headers['Authorization'] == 'Bearer tk-env'

    def test_an_explicit_token_overrides_the_env_var(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv('ZAI_API_KEY', 'tk-env')
        recorder = _Recorder(200)

        _require(recorder, auth_token='tk-test')

        [request] = recorder.requests
        assert request.headers['Authorization'] == 'Bearer tk-test'
