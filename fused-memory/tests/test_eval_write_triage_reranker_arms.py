"""Tests for eval_write_triage_reranker_arms.py — the ρ1 reranker adapters.

No network, no torch: each vendor's client is a fake built from its documented
response shape, and a local model is a fake with ``predict``.
"""
from __future__ import annotations

import functools
import math
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from _fm_helpers import load_script_module

SCRIPTS = Path(__file__).parent.parent / 'scripts'


@functools.cache
def _arms() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'eval_write_triage_reranker_arms.py', 'eval_write_triage_reranker_arms',
    )


def _context():
    return _arms().ArmContext(
        device='auto', local_batch_size=4, vram_cap_gib=8.0, pairwise_concurrency=20,
    )


class TestYesProbability:
    def test_yes_and_no_spellings_are_normalised_and_summed(self) -> None:
        top = [
            ('Yes', math.log(0.6)), (' yes', math.log(0.1)),
            ('No', math.log(0.2)), ('maybe', math.log(0.1)),
        ]
        assert _arms().yes_probability(top) == pytest.approx(0.7 / 0.9)

    def test_only_yes_is_certainty(self) -> None:
        assert _arms().yes_probability([('yes', math.log(0.8)), ('sure', math.log(0.2))]) == 1.0

    def test_neither_answer_present_is_refused(self) -> None:
        with pytest.raises(ValueError):
            _arms().yes_probability([('maybe', math.log(0.9))])


class _FakeChatClient:
    """A sync OpenAI client whose calls for one slate must all be in flight together."""

    def __init__(self, p_yes: dict[str, float]) -> None:
        self.p_yes = p_yes
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        self._barrier = threading.Barrier(len(p_yes), timeout=5)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        with self._lock:
            self.requests.append(kwargs)
        self._barrier.wait()
        user = kwargs['messages'][-1]['content']
        [(position, p)] = [
            (i, p) for i, (text, p) in enumerate(self.p_yes.items()) if text in user
        ]
        time.sleep(0.02 * (len(self.p_yes) - position))
        top = [
            SimpleNamespace(token='yes', logprob=math.log(p)),
            SimpleNamespace(token='no', logprob=math.log(1 - p)),
        ]
        return SimpleNamespace(
            choices=[SimpleNamespace(logprobs=SimpleNamespace(
                content=[SimpleNamespace(top_logprobs=top)],
            ))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=1),
        )


class TestPairwiseScorer:
    P_YES = {'alpha claim': 0.9, 'beta claim': 0.25, 'gamma claim': 0.5}

    def _score(self, client: _FakeChatClient):
        scorer = _arms().PairwiseScorer(client, model='gpt-4o-mini', concurrency=3)
        try:
            return scorer, scorer.score('the entry', list(self.P_YES))
        finally:
            scorer.close()

    def test_scores_come_back_in_candidate_order(self) -> None:
        _, slate = self._score(_FakeChatClient(self.P_YES))
        assert slate.scores == pytest.approx((0.9, 0.25, 0.5))
        assert slate.pairs_over_max_length is None

    def test_one_single_token_logprob_request_per_candidate(self) -> None:
        client = _FakeChatClient(self.P_YES)
        self._score(client)
        assert len(client.requests) == 3
        for request in client.requests:
            assert request['model'] == 'gpt-4o-mini'
            assert (request['temperature'], request['max_tokens']) == (0, 1)
            assert (request['logprobs'], request['top_logprobs']) == (True, 5)
        prompts = [
            '\n'.join(message['content'] for message in request['messages'])
            for request in client.requests
        ]
        assert all('the entry' in prompt for prompt in prompts)
        for text in self.P_YES:
            assert sum(text in prompt for prompt in prompts) == 1

    def test_cost_is_the_token_usage_at_list_price(self) -> None:
        _, slate = self._score(_FakeChatClient(self.P_YES))
        assert slate.cost_usd == pytest.approx(3 * (100 * 0.15 + 1 * 0.60) / 1e6)

    def test_facts_name_the_remote_host(self) -> None:
        scorer, _ = self._score(_FakeChatClient(self.P_YES))
        facts = scorer.facts()
        assert (facts.device, facts.vram_peak_mib, facts.max_length) == (
            'remote:api.openai.com', None, None,
        )


class TestOpenPairwise:
    def test_a_missing_key_is_an_unavailable_arm_naming_the_variable(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        arms = _arms()
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        with pytest.raises(arms.ArmUnavailable) as caught, arms.open_pairwise(_context()):
            pass
        assert caught.value.reason == arms.SkipReason.no_credential
        assert 'OPENAI_API_KEY' in caught.value.detail
