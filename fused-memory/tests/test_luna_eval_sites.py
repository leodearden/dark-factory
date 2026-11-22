"""Tests for scripts/luna_eval_sites.py (task 9945): one fixture of one call site under one arm.

The graphiti-site runners call graphiti_core's own resolve_extracted_edge and
extract_attributes_from_nodes, so these tests drive the real functions with a
fake LLM that answers by prompt name, as test_replay_overreach_extraction.py does.
No driver and no network exists anywhere here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.prompts.models import Message
from pydantic import BaseModel

from fused_memory.backends.llm_token_usage import AttributingTokenUsageTracker

SCRIPTS = Path(__file__).parent.parent / 'scripts'
sites = load_script_module(SCRIPTS / 'luna_eval_sites.py', mod_name='luna_eval_sites')
replay = load_script_module(SCRIPTS / 'replay_overreach_extraction.py', mod_name='replay_overreach_extraction')

RESOLVE_PROMPT, SUMMARY_PROMPT = 'dedupe_edges.resolve_edge', 'extract_nodes.extract_summaries_batch'
LUNA = replay.LUNA_MODEL
DF = 'dark_factory'


class _SiteLLM(LLMClient):
    """Answers each prompt name with *answers*' entry, records every call and its tokens, and
    declares the model, effort and endpoint it was built on."""

    def __init__(self, answers: dict[str, dict[str, Any]], *, model: str = 'gpt-4o-mini',
                 reasoning_effort: str | None = None, endpoint: str = 'configured') -> None:
        super().__init__(GraphitiLLMConfig(model=model))
        self.token_tracker = AttributingTokenUsageTracker()
        self.reasoning_effort = reasoning_effort
        self.endpoint = endpoint
        self.answers = answers
        self.calls: list[tuple[str | None, list[Message]]] = []

    async def _generate_response(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError('the runners call generate_response, never the private hook')

    async def generate_response(
        self, messages: list[Message], response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None, model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None, prompt_name: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append((prompt_name, messages))
        self.token_tracker.record(prompt_name or '', 120, 30)
        return self.answers[prompt_name or '']

    def prompt(self, prompt_name: str) -> str:
        (messages,) = [messages for name, messages in self.calls if name == prompt_name]
        return '\n'.join(message.content for message in messages)


def _luna(answers: dict[str, dict[str, Any]]) -> _SiteLLM:
    return _SiteLLM(answers, model=LUNA, reasoning_effort='low', endpoint='plan_proxy')


def _stored(uuid: str, fact: str, source: str, target: str, valid_at: str) -> dict[str, Any]:
    return {'uuid': uuid, 'fact': fact, 'source_node_uuid': source, 'target_node_uuid': target,
            'created_at': valid_at, 'valid_at': valid_at, 'invalid_at': None, 'expired_at': None}


RESOLVE_FIXTURE = {
    'graph': DF, 'edge_uuid': 'new', 'fact': 'Alpha adopts Beta for the queue',
    'source_node_uuid': 'A', 'target_node_uuid': 'B',
    'created_at': '2026-10-06T00:00:00+00:00', 'valid_at': '2026-10-06T00:00:00+00:00',
    'invalid_at': None, 'expired_at': None,
    'episode_uuid': 'ep-1', 'episode_valid_at': '2026-10-06T00:00:00+00:00',
    'related': [_stored('sib', 'Alpha uses Beta for the queue', 'A', 'B', '2026-10-01T00:00:00+00:00')],
    'invalidation_candidates': [
        _stored('old-x', 'Alpha uses Xi for the queue', 'A', 'X', '2026-09-01T00:00:00+00:00'),
        _stored('old-y', 'Psi feeds Beta', 'Y', 'B', '2026-09-02T00:00:00+00:00'),
    ],
}


def _resolve_answer(duplicates: list[int], contradicted: list[int]) -> dict[str, dict[str, Any]]:
    return {RESOLVE_PROMPT: {'duplicate_facts': duplicates, 'contradicted_facts': contradicted}}


class TestTheArms:
    def test_the_site_arms_are_the_replays_own(self) -> None:
        assert set(sites.SITE_ARMS) == {'control', 'luna-low'}
        assert sites.SITE_ARMS['control'] is replay.ARMS['control']
        assert sites.SITE_ARMS['luna-low'] is replay.ARMS['luna-low']


class TestRunResolveEdge:
    @pytest.mark.asyncio
    async def test_a_duplicate_answer_merges_into_the_sibling(self) -> None:
        llm = _luna(_resolve_answer([0], []))
        record = await sites.run_resolve_edge(llm, RESOLVE_FIXTURE, sites.SITE_ARMS['luna-low'])
        assert record['duplicate_uuids'] == ['sib']
        assert record['merged_into'] == 'sib'
        assert record['invalidated_uuids'] == []

    @pytest.mark.asyncio
    async def test_a_contradiction_invalidates_an_older_candidate_and_merges_nothing(self) -> None:
        llm = _SiteLLM(_resolve_answer([], [1]))
        record = await sites.run_resolve_edge(llm, RESOLVE_FIXTURE, sites.SITE_ARMS['control'])
        assert record['duplicate_uuids'] == [] and record['merged_into'] is None
        assert record['invalidated_uuids'] == ['old-x']

    @pytest.mark.asyncio
    async def test_the_prompt_offers_the_siblings_and_the_invalidation_candidates(self) -> None:
        llm = _SiteLLM(_resolve_answer([], []))
        await sites.run_resolve_edge(llm, RESOLVE_FIXTURE, sites.SITE_ARMS['control'])
        prompt = llm.prompt(RESOLVE_PROMPT)
        assert all(fact in prompt for fact in (
            'Alpha adopts Beta for the queue', 'Alpha uses Beta for the queue', 'Psi feeds Beta',
        ))

    @pytest.mark.asyncio
    async def test_the_record_names_its_site_arm_client_fixture_and_tokens(self) -> None:
        record = await sites.run_resolve_edge(
            _luna(_resolve_answer([], [])), RESOLVE_FIXTURE, sites.SITE_ARMS['luna-low'],
        )
        assert {k: record[k] for k in ('site', 'arm', 'model', 'reasoning_effort', 'endpoint', 'fixture_id')} == {
            'site': 'resolve_edge', 'arm': 'luna-low', 'model': LUNA, 'reasoning_effort': 'low',
            'endpoint': 'plan_proxy', 'fixture_id': f'{DF}/new',
        }
        assert record['tokens'] == {'resolve_edge': {'input_tokens': 120, 'output_tokens': 30, 'llm_calls': 1}}
        assert sites.record_key(record) == ('resolve_edge', 'luna-low', f'{DF}/new')


SUMMARY_FIXTURE = {
    'graph': DF, 'episode_uuid': 'ep-1', 'episode_valid_at': '2026-10-06T00:00:00+00:00',
    'episode_content': 'Alpha now runs the merge queue for Beta.',
    'nodes': [{'node_uuid': 'A', 'node_name': 'Alpha', 'summary': 'Alpha. ' * 300},
              {'node_uuid': 'B', 'node_name': 'Beta', 'summary': 'Beta. ' * 400}],
    'edges': [{'uuid': 'r1', 'source_node_uuid': 'A', 'target_node_uuid': 'B', 'fact': 'Alpha runs the queue for Beta'}],
}
PREVIOUS = ('Earlier Alpha weighed Gamma.', 'Later Beta shipped.')
SUMMARY_ANSWER = {SUMMARY_PROMPT: {'summaries': [
    {'name': 'Alpha', 'summary': 'Alpha runs the merge queue.'},
    {'name': 'beta', 'summary': 'Beta is served by the queue.'},
]}}


class TestRunSummaries:
    @pytest.mark.asyncio
    async def test_one_summary_per_node(self) -> None:
        record = await sites.run_summaries(_luna(SUMMARY_ANSWER), SUMMARY_FIXTURE, PREVIOUS, sites.SITE_ARMS['luna-low'])
        assert record['summaries'] == {'A': 'Alpha runs the merge queue.', 'B': 'Beta is served by the queue.'}

    @pytest.mark.asyncio
    async def test_a_node_the_answer_omits_records_none(self) -> None:
        answer = {SUMMARY_PROMPT: {'summaries': [{'name': 'Alpha', 'summary': 'Alpha runs the merge queue.'}]}}
        record = await sites.run_summaries(_luna(answer), SUMMARY_FIXTURE, PREVIOUS, sites.SITE_ARMS['luna-low'])
        assert record['summaries'] == {'A': 'Alpha runs the merge queue.', 'B': None}

    @pytest.mark.asyncio
    async def test_the_episode_its_previous_episodes_and_the_summaries_reach_the_prompt(self) -> None:
        llm = _SiteLLM(SUMMARY_ANSWER)
        await sites.run_summaries(llm, SUMMARY_FIXTURE, PREVIOUS, sites.SITE_ARMS['control'])
        prompt = llm.prompt(SUMMARY_PROMPT)
        assert 'Alpha now runs the merge queue for Beta.' in prompt
        assert all(previous in prompt for previous in PREVIOUS)
        assert 'Alpha. Alpha.' in prompt and 'Beta. Beta.' in prompt
        assert [name for name, _ in llm.calls] == [SUMMARY_PROMPT]

    @pytest.mark.asyncio
    async def test_the_record_names_its_site_arm_client_fixture_and_tokens(self) -> None:
        record = await sites.run_summaries(_SiteLLM(SUMMARY_ANSWER), SUMMARY_FIXTURE, PREVIOUS, sites.SITE_ARMS['control'])
        assert {k: record[k] for k in ('site', 'arm', 'model', 'reasoning_effort', 'endpoint', 'fixture_id')} == {
            'site': 'summaries', 'arm': 'control', 'model': 'gpt-4o-mini', 'reasoning_effort': None,
            'endpoint': 'configured', 'fixture_id': f'{DF}/ep-1',
        }
        assert record['tokens'] == {'summaries': {'input_tokens': 120, 'output_tokens': 30, 'llm_calls': 1}}


class TestTheNullEmbedder:
    @pytest.mark.asyncio
    async def test_it_is_a_graphiti_embedder_returning_zero_vectors(self) -> None:
        embedder = sites.NullEmbedder()
        assert isinstance(embedder, EmbedderClient)
        assert await embedder.create('anything') == [0.0] * sites.NULL_EMBEDDING_DIM
        assert await embedder.create_batch(['a', 'b']) == [[0.0] * sites.NULL_EMBEDDING_DIM] * 2


WRONG_CLIENTS = [
    pytest.param({'model': 'gpt-4o-mini', 'reasoning_effort': 'low', 'endpoint': 'plan_proxy'}, id='model'),
    pytest.param({'model': LUNA, 'reasoning_effort': 'medium', 'endpoint': 'plan_proxy'}, id='effort'),
    pytest.param({'model': LUNA, 'reasoning_effort': 'low', 'endpoint': 'configured'}, id='endpoint'),
]


class TestAWrongClientIsRefusedBeforeAnyCall:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('client', WRONG_CLIENTS)
    async def test_resolve_edge(self, client: dict[str, Any]) -> None:
        llm = _SiteLLM(_resolve_answer([], []), **client)
        with pytest.raises(replay.ArmClientMismatch):
            await sites.run_resolve_edge(llm, RESOLVE_FIXTURE, sites.SITE_ARMS['luna-low'])
        assert llm.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize('client', WRONG_CLIENTS)
    async def test_summaries(self, client: dict[str, Any]) -> None:
        llm = _SiteLLM(SUMMARY_ANSWER, **client)
        with pytest.raises(replay.ArmClientMismatch):
            await sites.run_summaries(llm, SUMMARY_FIXTURE, PREVIOUS, sites.SITE_ARMS['luna-low'])
        assert llm.calls == []


# --------------------------------------------------------------------------- #
# The routing sites: the production classifier and router per arm (step 15)
# --------------------------------------------------------------------------- #

PROXY_URL, PROXY_KEY = 'http://127.0.0.1:8317/v1', 'proxy-test-key'
CLASSIFIER_ANSWER = '{"primary": "temporal_facts", "secondary": "decisions_and_rationale", "confidence": 0.8}'
ROUTER_ANSWER = '{"query_type": "procedural", "stores": ["mem0"], "primary_store": "mem0"}'


@pytest.fixture
def plan_proxy_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    monkeypatch.setenv('OPENAI_PLAN_PROXY_URL', PROXY_URL)
    monkeypatch.setenv('OPENAI_PLAN_PROXY_KEY', PROXY_KEY)
    return {'OPENAI_PLAN_PROXY_URL': PROXY_URL, 'OPENAI_PLAN_PROXY_KEY': PROXY_KEY}


@pytest.fixture
def routing_cfg(mock_config: Any) -> Any:
    mock_config.routing.llm_fallback = True
    return mock_config


class _Chat:
    """An httpx MockTransport handler answering every chat completion with *content* and usage 11/7."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: Any) -> Any:
        import json

        import httpx

        self.bodies.append(json.loads(request.content))
        return httpx.Response(200, json={
            'id': 'c', 'object': 'chat.completion', 'created': 0, 'model': 'm',
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': self.content}}],
            'usage': {'prompt_tokens': 11, 'completion_tokens': 7, 'total_tokens': 18},
        })


def _transport(chat: _Chat) -> Any:
    import httpx

    return httpx.MockTransport(chat)


class TestRoutingArm:
    def test_the_control_routes_on_the_production_endpoint_unchanged(self, routing_cfg: Any, plan_proxy_env: dict[str, str]) -> None:
        from fused_memory.routing.openai_client import routing_endpoint

        arm_cfg, client = sites.routing_arm(routing_cfg, sites.SITE_ARMS['control'], plan_proxy_env)
        assert arm_cfg.routing == routing_cfg.routing
        assert (client.api_key, str(client.base_url).rstrip('/')) == routing_endpoint(routing_cfg)
        assert replay.client_endpoint(client) == 'configured'

    def test_the_luna_arm_is_the_flip_the_routing_layer_would_ship(
        self, routing_cfg: Any, plan_proxy_env: dict[str, str],
    ) -> None:
        before = routing_cfg.model_dump()
        arm_cfg, client = sites.routing_arm(routing_cfg, sites.SITE_ARMS['luna-low'], plan_proxy_env)
        assert arm_cfg.routing == routing_cfg.routing.model_copy(update={
            'model': LUNA, 'reasoning_effort': 'low', 'openai_api_url': PROXY_URL, 'openai_api_key': PROXY_KEY,
        })
        assert (client.api_key, str(client.base_url).rstrip('/')) == (PROXY_KEY, PROXY_URL)
        assert replay.client_endpoint(client) == 'plan_proxy'
        assert routing_cfg.model_dump() == before

    @pytest.mark.parametrize('missing', ['OPENAI_PLAN_PROXY_URL', 'OPENAI_PLAN_PROXY_KEY'])
    def test_an_unset_proxy_variable_is_refused_by_name(self, routing_cfg: Any, plan_proxy_env: dict[str, str], missing: str) -> None:
        environ = {name: value for name, value in plan_proxy_env.items() if name != missing}
        with pytest.raises(replay.ArmClientMismatch, match=missing):
            sites.routing_arm(routing_cfg, sites.SITE_ARMS['luna-low'], environ)


class TestTheRoutingRequests:
    @pytest.mark.asyncio
    async def test_the_luna_request_sends_the_effort_and_no_temperature(self, routing_cfg: Any, plan_proxy_env: dict[str, str]) -> None:
        chat = _Chat(CLASSIFIER_ANSWER)
        arm = sites.SITE_ARMS['luna-low']
        arm_cfg, client = sites.routing_arm(routing_cfg, arm, plan_proxy_env, transport=_transport(chat))
        await sites.run_classifier(arm_cfg, client, {'id': 'w-1', 'input': 'Hello world'}, arm)
        (body,) = chat.bodies
        assert (body['model'], body['reasoning_effort'], body['max_completion_tokens']) == (LUNA, 'low', 256)
        assert 'temperature' not in body and 'max_tokens' not in body

    @pytest.mark.asyncio
    async def test_the_control_request_is_todays(self, routing_cfg: Any, plan_proxy_env: dict[str, str]) -> None:
        chat = _Chat(ROUTER_ANSWER)
        arm = sites.SITE_ARMS['control']
        arm_cfg, client = sites.routing_arm(routing_cfg, arm, plan_proxy_env, transport=_transport(chat))
        await sites.run_router(arm_cfg, client, {'id': 's-1', 'query': 'foo bar baz'}, arm)
        (body,) = chat.bodies
        assert (body['model'], body['temperature'], body['max_tokens']) == ('gpt-4o-mini', 0.0, 256)
        assert 'reasoning_effort' not in body


class TestRunClassifierAndRouter:
    @pytest.mark.asyncio
    async def test_a_classifier_record_carries_its_label_secondary_fallback_and_tokens(
        self, routing_cfg: Any, plan_proxy_env: dict[str, str],
    ) -> None:
        arm = sites.SITE_ARMS['luna-low']
        arm_cfg, client = sites.routing_arm(routing_cfg, arm, plan_proxy_env, transport=_transport(_Chat(CLASSIFIER_ANSWER)))
        record = await sites.run_classifier(arm_cfg, client, {'id': 'w-1', 'input': 'Hello world'}, arm)
        assert {k: record[k] for k in ('site', 'arm', 'model', 'reasoning_effort', 'endpoint', 'fixture_id')} == {
            'site': 'classifier', 'arm': 'luna-low', 'model': LUNA, 'reasoning_effort': 'low',
            'endpoint': 'plan_proxy', 'fixture_id': 'w-1',
        }
        assert (record['label'], record['secondary'], record['fallback']) == (
            'temporal_facts', 'decisions_and_rationale', None,
        )
        assert record['tokens'] == {'classifier': {'input_tokens': 11, 'output_tokens': 7, 'llm_calls': 1}}

    @pytest.mark.asyncio
    async def test_a_classifier_fallback_is_recorded(self, routing_cfg: Any, plan_proxy_env: dict[str, str]) -> None:
        arm = sites.SITE_ARMS['control']
        arm_cfg, client = sites.routing_arm(routing_cfg, arm, plan_proxy_env, transport=_transport(_Chat('no json here')))
        record = await sites.run_classifier(arm_cfg, client, {'id': 'w-1', 'input': 'Hello world'}, arm)
        assert (record['label'], record['fallback']) == ('observations_and_summaries', 'llm_no_json')

    @pytest.mark.asyncio
    async def test_a_router_record_carries_its_query_type_stores_and_tokens(
        self, routing_cfg: Any, plan_proxy_env: dict[str, str],
    ) -> None:
        arm = sites.SITE_ARMS['control']
        arm_cfg, client = sites.routing_arm(routing_cfg, arm, plan_proxy_env, transport=_transport(_Chat(ROUTER_ANSWER)))
        record = await sites.run_router(arm_cfg, client, {'id': 's-1', 'query': 'foo bar baz'}, arm)
        assert (record['site'], record['fixture_id'], record['endpoint']) == ('router', 's-1', 'configured')
        assert (record['label'], record['stores'], record['primary_store']) == ('procedural', ['mem0'], 'mem0')
        assert record['tokens'] == {'router': {'input_tokens': 11, 'output_tokens': 7, 'llm_calls': 1}}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('run', 'fixture'), [
        ('run_classifier', {'id': 'w-1', 'input': 'Hello world'}),
        ('run_router', {'id': 's-1', 'query': 'foo bar baz'}),
    ])
    async def test_a_call_no_completion_answers_raises_rather_than_score_the_fallback(
        self, routing_cfg: Any, plan_proxy_env: dict[str, str], run: str, fixture: dict[str, str],
    ) -> None:
        import httpx

        def refused(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={'error': {'message': 'Unsupported parameter', 'type': 'invalid_request'}})

        arm = sites.SITE_ARMS['luna-low']
        arm_cfg, client = sites.routing_arm(routing_cfg, arm, plan_proxy_env, transport=httpx.MockTransport(refused))
        with pytest.raises(sites.RoutingCallUnanswered, match=fixture['id']):
            await getattr(sites, run)(arm_cfg, client, fixture, arm)


class TestResumableRuns:
    FIXTURES = [{'id': 'f-1'}, {'id': 'f-2'}, {'id': 'f-3'}]

    def _runner(self, calls: list[tuple[str, str]], failing: frozenset[str] = frozenset()) -> Any:
        async def runner(arm: Any, fixture: dict[str, Any]) -> dict[str, Any]:
            calls.append((arm.name, fixture['id']))
            if fixture['id'] in failing:
                raise RuntimeError('proxy down')
            return {'kind': 'replayed', 'site': 'router', 'arm': arm.name, 'fixture_id': fixture['id']}
        return runner

    async def _run(self, out: Path, calls: list[tuple[str, str]], failing: frozenset[str] = frozenset()) -> int:
        return await sites.run_site_fixtures(
            sites.Site.router, self.FIXTURES, list(sites.SITE_ARMS.values()), self._runner(calls, failing), out,
        )

    @staticmethod
    def _lines(out: Path) -> list[dict[str, Any]]:
        import json

        return [json.loads(line) for line in out.read_text().splitlines()]

    @pytest.mark.asyncio
    async def test_a_failure_is_a_line_and_a_rerun_retries_only_it(self, tmp_path: Path) -> None:
        out, first, second = tmp_path / 'runs.jsonl', [], []
        assert await self._run(out, first, frozenset({'f-2'})) == 2
        failed = [line for line in self._lines(out) if line['kind'] == 'failed']
        assert {(line['arm'], line['fixture_id'], line['error_type']) for line in failed} == {
            ('control', 'f-2', 'RuntimeError'), ('luna-low', 'f-2', 'RuntimeError'),
        }
        assert await self._run(out, second) == 0
        assert sorted(second) == [('control', 'f-2'), ('luna-low', 'f-2')]
        lines = self._lines(out)
        assert len(lines) == 6 and all(line['kind'] == 'replayed' for line in lines)

    @pytest.mark.asyncio
    async def test_recorded_keys_are_skipped(self, tmp_path: Path) -> None:
        out, first, second = tmp_path / 'runs.jsonl', [], []
        assert await self._run(out, first) == 0
        assert await self._run(out, second) == 0
        assert second == [] and len(self._lines(out)) == 6
