"""Tests for eval_write_triage_reranker_arms.py — the ρ1 reranker adapters.

No network, no torch: each vendor's client is a fake built from its documented
response shape, and a local model is a fake with ``predict``.
"""
from __future__ import annotations

import functools
import itertools
import json
import math
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from _fm_helpers import load_script_module

SCRIPTS = Path(__file__).parent.parent / 'scripts'


@functools.cache
def _arms() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'eval_write_triage_reranker_arms.py', 'eval_write_triage_reranker_arms',
    )


@functools.cache
def _core() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'eval_write_triage_reranker.py', 'eval_write_triage_reranker')


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


HOSTED_NAMES = ('jina-reranker', 'voyage-rerank', 'cohere-rerank')


def _hosted(name: str):
    [api] = [api for api in _arms().HOSTED_APIS if api.name == name]
    return api


class _Recorder:
    """An httpx handler answering every request with *body*, keeping the requests."""

    def __init__(self, body: dict) -> None:
        self.body = body
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=self.body)


def _results_body(api, results: list[dict]) -> dict:
    """A rerank response under *api*'s results key, carrying every vendor's usage block."""
    return {
        api.results_key: results,
        'usage': {'total_tokens': 1000},
        'meta': {'billed_units': {'search_units': 1}},
    }


def _ranked(*pairs: tuple) -> list[dict]:
    return [{'index': index, 'relevance_score': score} for index, score in pairs]


def _hosted_score(api, body: dict, candidates: list[str]):
    recorder = _Recorder(body)
    client = httpx.Client(transport=httpx.MockTransport(recorder))
    slate = _arms().HostedRerankScorer(api, client, 'sk-test').score('the entry', candidates)
    return slate, recorder


class TestHostedRerankScorer:
    def test_the_three_vendors_are_registered(self) -> None:
        assert {api.name for api in _arms().HOSTED_APIS} == set(HOSTED_NAMES)

    @pytest.mark.parametrize('name', HOSTED_NAMES)
    def test_the_request_carries_the_entry_and_every_candidate_in_order(self, name: str) -> None:
        api = _hosted(name)
        _, recorder = _hosted_score(
            api, _results_body(api, _ranked((0, 0.1), (1, 0.2), (2, 0.3))), ['a', 'b', 'c'],
        )
        [request] = recorder.requests
        assert (request.method, str(request.url)) == ('POST', api.url)
        assert request.headers['Authorization'] == 'Bearer sk-test'
        body = json.loads(request.content)
        assert (body['model'], body['query'], body['documents']) == (
            api.model, 'the entry', ['a', 'b', 'c'],
        )
        assert body[api.top_n_field] == 3

    @pytest.mark.parametrize('name', HOSTED_NAMES)
    def test_shuffled_results_map_back_to_candidate_order(self, name: str) -> None:
        api = _hosted(name)
        slate, _ = _hosted_score(
            api, _results_body(api, _ranked((2, 0.3), (0, 0.9), (1, 0.5))), ['a', 'b', 'c'],
        )
        assert slate.scores == (0.9, 0.5, 0.3)
        assert slate.pairs_over_max_length is None

    @pytest.mark.parametrize('name', HOSTED_NAMES)
    @pytest.mark.parametrize('results', [
        pytest.param([{'relevance_score': 0.9}, *_ranked((1, 0.5), (2, 0.3))], id='no-index'),
        pytest.param(_ranked((0, 0.9), (0, 0.5), (1, 0.3)), id='duplicate'),
        pytest.param(_ranked((0, 0.9), (1, 0.5), (3, 0.3)), id='out-of-range'),
        pytest.param(_ranked((0, 0.9), (1, 0.5)), id='one-never-returned'),
        pytest.param(_ranked((0, 'high'), (1, 0.5), (2, 0.3)), id='non-numeric'),
        pytest.param(_ranked((0, True), (1, 0.5), (2, 0.3)), id='boolean'),
    ])
    def test_a_malformed_ranking_is_refused(self, name: str, results: list) -> None:
        api = _hosted(name)
        with pytest.raises(ValueError):
            _hosted_score(api, _results_body(api, results), ['a', 'b', 'c'])

    @pytest.mark.parametrize('name', HOSTED_NAMES)
    def test_cost_is_the_specs_price_over_the_response(self, name: str) -> None:
        api = _hosted(name)
        body = _results_body(api, _ranked((0, 0.9), (1, 0.5)))
        slate, _ = _hosted_score(api, body, ['a', 'b'])
        assert slate.cost_usd == api.cost(body)

    @pytest.mark.parametrize('name', HOSTED_NAMES)
    def test_with_no_key_set_the_arm_is_unavailable(
        self, name: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        arms, api = _arms(), _hosted(name)
        for variable in api.env_vars:
            monkeypatch.delenv(variable, raising=False)
        with pytest.raises(arms.ArmUnavailable) as caught, arms.open_hosted(api)(_context()):
            pass
        assert caught.value.reason == arms.SkipReason.no_credential


class _JevEndpoint:
    """Answers a single-choice question from the request itself, option names shuffled.

    *p_by_text* gives each candidate text's probability; *reshape* edits the
    answer's name -> probability mapping before it is returned.
    """

    def __init__(self, p_by_text: dict[str, float], *, reshape=None, usage=None) -> None:
        self.p_by_text = p_by_text
        self.reshape = reshape or (lambda probabilities: probabilities)
        self.usage = usage if usage is not None else {'input_tokens': 1000, 'output_tokens': 3}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        [(key, question)] = json.loads(request.content)['questions'].items()
        names = list(question['criteria'])
        probabilities = {
            name: self.p_by_text[question['criteria'][name]] for name in reversed(names)
        }
        return httpx.Response(200, json={
            'model': 'jev-1.13.0',
            'answers': {key: {
                'type': 'choice', 'choice': names[0],
                'probabilities': self.reshape(probabilities), 'confidence': 0.5,
            }},
            'usage': self.usage,
        })


class TestJevChoiceScorer:
    P_BY_TEXT = {'alpha': 0.7, 'beta': 0.1, 'gamma': 0.2}

    def _score(self, endpoint: _JevEndpoint, candidates: list[str] | None = None):
        client = httpx.Client(transport=httpx.MockTransport(endpoint))
        return _arms().JevChoiceScorer(client, 'tk-test').score(
            'the entry', candidates if candidates is not None else list(self.P_BY_TEXT),
        )

    def test_the_entry_is_the_state_and_the_candidates_one_choice_in_order(self) -> None:
        endpoint = _JevEndpoint(self.P_BY_TEXT)
        self._score(endpoint)
        [request] = endpoint.requests
        assert (request.method, str(request.url)) == (
            'POST', 'https://api.typesafe.ai/v1/systemone',
        )
        assert request.headers['Authorization'] == 'Bearer tk-test'
        body = json.loads(request.content)
        assert body['state'] == 'the entry'
        [question] = body['questions'].values()
        assert question['type'] == 'choice'
        assert list(question['criteria'].values()) == list(self.P_BY_TEXT)

    def test_the_distribution_becomes_scores_in_candidate_order(self) -> None:
        slate = self._score(_JevEndpoint(self.P_BY_TEXT))
        assert slate.scores == (0.7, 0.1, 0.2)
        assert slate.pairs_over_max_length is None

    @pytest.mark.parametrize('reshape', [
        pytest.param(lambda p: dict(list(p.items())[1:]), id='an-option-missing'),
        pytest.param(lambda p: {**p, 'stranger': 0.0}, id='an-unknown-option'),
        pytest.param(lambda p: {name: 'high' for name in p}, id='non-numeric'),
    ])
    def test_a_distribution_not_matching_the_options_is_refused(self, reshape) -> None:
        with pytest.raises(ValueError):
            self._score(_JevEndpoint(self.P_BY_TEXT, reshape=reshape))

    def test_cost_is_the_input_tokens_at_list_price(self) -> None:
        slate = self._score(_JevEndpoint(self.P_BY_TEXT))
        assert slate.cost_usd == pytest.approx(1000 * 0.042 / 1e6)

    def test_unreported_input_tokens_leave_the_cost_unmeasured(self) -> None:
        slate = self._score(_JevEndpoint(self.P_BY_TEXT, usage={'input_tokens': None}))
        assert slate.cost_usd is None

    def test_more_options_than_a_choice_allows_is_refused_before_any_request(self) -> None:
        endpoint = _JevEndpoint({})
        with pytest.raises(ValueError):
            self._score(endpoint, [f'candidate {i}' for i in range(256)])
        assert endpoint.requests == []

    def test_with_no_key_set_the_arm_is_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        arms = _arms()
        monkeypatch.delenv('TYPESAFE_API_KEY', raising=False)
        with pytest.raises(arms.ArmUnavailable) as caught, arms.open_jev(_context()):
            pass
        assert caught.value.reason == arms.SkipReason.no_credential
        assert 'TYPESAFE_API_KEY' in caught.value.detail


class TestOpenCrossEncoder:
    @pytest.mark.parametrize('missing', ['sentence_transformers', 'torch'])
    def test_a_missing_dependency_is_an_unavailable_arm_with_the_install_hint(
        self, missing: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        arms = _arms()
        monkeypatch.setitem(sys.modules, missing, None)
        with (
            pytest.raises(arms.ArmUnavailable) as caught,
            arms.open_cross_encoder('some/model', 512)(_context()),
        ):
            pass
        assert caught.value.reason == arms.SkipReason.dependency_unavailable
        assert missing in caught.value.detail
        assert '--group reranker' in caught.value.detail


class _FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[list, dict]] = []

    def predict(self, pairs, **kwargs):
        self.calls.append((list(pairs), kwargs))
        return list(self.scores)


def _word_count(text: str) -> int:
    return len(text.split())


class TestCrossEncoderScorer:

    def _scorer(self, model: _FakeCrossEncoder):
        arms = _arms()
        return arms.CrossEncoderScorer(
            model, _word_count, batch_size=4, max_length=5,
            device_facts=lambda: arms.ScorerFacts(
                device='cuda:0 (fake)', vram_peak_mib=100.0, max_length=None,
            ),
        )

    def test_each_candidate_is_paired_with_the_entry_in_candidate_order(self) -> None:
        model = _FakeCrossEncoder([0.2, 0.9, 0.5])
        slate = self._scorer(model).score('a b c', ['d', 'e f g h', 'i j'])
        [(pairs, kwargs)] = model.calls
        assert pairs == [('a b c', 'd'), ('a b c', 'e f g h'), ('a b c', 'i j')]
        assert kwargs['batch_size'] == 4
        assert slate.scores == (0.2, 0.9, 0.5)
        assert slate.cost_usd == 0.0

    def test_only_pairs_longer_than_max_length_count_as_over(self) -> None:
        slate = self._scorer(_FakeCrossEncoder([0.2, 0.9, 0.5])).score(
            'a b c', ['d', 'e f g h', 'i j'],
        )
        assert slate.pairs_over_max_length == 1

    def test_facts_carry_the_device_and_the_configured_max_length(self) -> None:
        facts = self._scorer(_FakeCrossEncoder([])).facts()
        assert (facts.device, facts.vram_peak_mib, facts.max_length) == (
            'cuda:0 (fake)', 100.0, 5,
        )


def _fake_torch(*, available: bool, **cuda) -> SimpleNamespace:
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available, **cuda))


class TestCudaFacts:
    def test_without_cuda_the_device_is_the_cpu(self) -> None:
        facts = _arms().cuda_facts(_fake_torch(available=False))
        assert (facts.device, facts.vram_peak_mib) == ('cpu', None)

    def test_with_cuda_the_gpu_and_its_peak_allocation_are_named(self) -> None:
        facts = _arms().cuda_facts(_fake_torch(
            available=True, current_device=lambda: 0,
            get_device_name=lambda index: 'NVIDIA GeForce RTX 3090',
            max_memory_allocated=lambda index: 3 * 2**30,
        ))
        assert 'RTX 3090' in facts.device
        assert facts.vram_peak_mib == 3072.0


class TestApplyVramCap:
    def _cuda(self, calls: list) -> SimpleNamespace:
        return _fake_torch(
            available=True, current_device=lambda: 0,
            get_device_properties=lambda index: SimpleNamespace(total_memory=24 * 2**30),
            set_per_process_memory_fraction=lambda fraction, device=None: calls.append(fraction),
        )

    def test_the_cap_is_a_fraction_of_the_devices_memory(self) -> None:
        calls: list[float] = []
        _arms().apply_vram_cap(self._cuda(calls), 8.0)
        assert calls == [pytest.approx(1 / 3)]

    def test_a_cap_above_the_device_is_the_whole_device(self) -> None:
        calls: list[float] = []
        _arms().apply_vram_cap(self._cuda(calls), 30.0)
        assert calls == [1.0]

    def test_on_the_cpu_nothing_is_capped(self) -> None:
        _arms().apply_vram_cap(_fake_torch(available=False), 8.0)


D1_NAMES = (
    'qwen3-reranker-0.6b', 'mxbai-rerank-base-v2', 'bge-reranker-v2-m3',
    'gpt-4o-mini-pairwise', 'jina-reranker', 'voyage-rerank', 'cohere-rerank', 'jev-choice',
)
CREDENTIAL_VARIABLES = (
    'OPENAI_API_KEY', 'JINA_API_KEY', 'VOYAGE_API_KEY', 'COHERE_API_KEY', 'CO_API_KEY',
    'TYPESAFE_API_KEY',
)


@pytest.fixture
def bare_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """No credential set and neither torch nor sentence-transformers importable."""
    hosted = {variable for api in _arms().HOSTED_APIS for variable in api.env_vars}
    for variable in {*CREDENTIAL_VARIABLES, *hosted}:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setitem(sys.modules, 'torch', None)
    monkeypatch.setitem(sys.modules, 'sentence_transformers', None)


class TestD1Registry:
    def test_the_eight_arms_in_report_order(self) -> None:
        assert tuple(spec.name for spec in _arms().D1_ARMS) == D1_NAMES

    def test_every_arm_class_is_represented(self) -> None:
        arms = _arms()
        assert {spec.arm_class for spec in arms.D1_ARMS} == set(arms.ArmClass)

    def test_the_local_arms_name_their_model_repos(self) -> None:
        assert [spec.model for spec in _arms().D1_ARMS[:3]] == [
            'Qwen/Qwen3-Reranker-0.6B', 'mixedbread-ai/mxbai-rerank-base-v2',
            'BAAI/bge-reranker-v2-m3',
        ]

    @pytest.mark.usefixtures('bare_host')
    @pytest.mark.parametrize('name', D1_NAMES)
    def test_on_a_bare_host_every_arm_is_unavailable_never_a_crash(self, name: str) -> None:
        arms = _arms()
        [spec] = [spec for spec in arms.D1_ARMS if spec.name == name]
        with pytest.raises(arms.ArmUnavailable), spec.open(_context()):
            pass

    @pytest.mark.usefixtures('bare_host')
    def test_a_bare_host_run_writes_skipped_rows_and_a_null_best(self, tmp_path: Path) -> None:
        arms, core = _arms(), _core()
        cases = [
            core.RerankCase(
                memory_id=f'd{i}', label='duplicate', entry=f'entry {i}', canonical_id='c1',
                canonical_present=True, candidate_ids=('c1', 'x'),
                candidate_texts=('canonical', 'other'), candidate_cosines=(0.9, 0.5),
                candidate_parents={}, degraded=False, self_retrieved=False,
            )
            for i in range(2)
        ]
        path = tmp_path / 'bare.json'
        report = core.run_reranker_eval(
            cases=cases, arms=arms.D1_ARMS, aliases=None, context=_context(),
            provenance={}, report_path=path, clock=itertools.count(0.0, 0.5).__next__,
            max_spend_usd=1.0, p95_ceiling_seconds=3.0,
        )
        assert {row['status'] for row in report['arms']} == {'skipped'}
        assert {row['skip_reason'] for row in report['arms']} <= {
            'no_credential', 'dependency_unavailable',
        }
        assert report['best'] == {
            'arm': None, 'rank1_rate': None, 'p95_seconds': None,
            'qualified': False, 'p95_ceiling_seconds': 3.0,
        }
        assert '"best":' in path.read_text()
