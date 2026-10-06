"""Runnable serving-dependent checks (boundary rows 1, 4, 6), their logic proven without serving."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from _mock_openai_server import mock_openai_server, responses_body
from redis.exceptions import ResponseError
from shared.memory_eval_metrics import Metric

from arm_harness._fakes import (
    UNREACHABLE_BASE_URL,
    incumbent_control_spec,
    llm_spec,
    run_manifest_for,
)
from fused_memory.arm_harness.checks import (
    INDEX_PROBE_ATTEMPTS,
    INDEX_PROBE_INTERVAL_S,
    check_index_configuration,
    control_variance_check,
    smoke_endpoint,
)
from fused_memory.arm_harness.comparison import RunComparabilityError
from fused_memory.arm_harness.instrument_checks import InstrumentCheckId
from fused_memory.arm_harness.metrics_record import (
    IndexConfiguration,
    LlmMetricId,
    MetricsRecord,
    record_for,
)
from fused_memory.arm_harness.replay_types import EpisodeOutcome
from fused_memory.arm_harness.scratch_guard import GuardCheckpoint, ScratchGuardError
from fused_memory.backends.llm_clients import TokenRecordingOpenAIGenericClient

VALID_ENTITIES = json.dumps({'entities': [{'name': 'Alice'}, {'name': 'Bob'}]})
OFF_SCHEMA = json.dumps({'unexpected': 1})
SCRATCH = 'evalmem_index_probe'


def _accept_everything(result: object, response_model: object) -> None:
    return None


def _chat_requests(server) -> list[dict[str, Any]]:
    return [request['json_body'] for request in server.requests_to('/chat/completions')]


# --- boundary row 1: endpoint conformance smoke -------------------------------------


@pytest.mark.asyncio
async def test_smoke_passes_against_a_schema_valid_endpoint(mock_config):
    with mock_openai_server() as server:
        server.chat_content = VALID_ENTITIES
        verdict = await smoke_endpoint(llm_spec(base_url=server.base_url), mock_config)
        requests = _chat_requests(server)

    assert verdict.passed is True
    assert verdict.positive.passed and verdict.negative_control.passed
    assert verdict.counts.schema_valid >= 1
    assert requests, 'the arm base_url received no traffic'
    response_format = requests[0]['response_format']
    assert response_format['type'] == 'json_schema'
    assert '$defs' in json.dumps(response_format), 'the smoke schema must carry $defs/$ref'


@pytest.mark.asyncio
async def test_smoke_detects_off_schema_json_on_every_attempt(mock_config):
    with mock_openai_server() as server:
        server.chat_content = OFF_SCHEMA
        verdict = await smoke_endpoint(llm_spec(base_url=server.base_url), mock_config)

    assert verdict.passed is False
    assert verdict.positive.check_id is InstrumentCheckId.ENDPOINT_CONFORMANCE
    assert verdict.positive.passed is False
    assert 'ValidationError' in verdict.positive.detail
    assert verdict.counts.schema_valid == 0
    assert verdict.counts.schema_invalid == TokenRecordingOpenAIGenericClient.MAX_RETRIES + 1
    assert verdict.negative_control.passed is True


@pytest.mark.asyncio
async def test_a_validator_that_accepts_everything_fails_the_negative_control(mock_config):
    with mock_openai_server() as server:
        server.chat_content = OFF_SCHEMA
        verdict = await smoke_endpoint(
            llm_spec(base_url=server.base_url), mock_config, validator=_accept_everything
        )

    assert verdict.positive.passed is True, 'an inert validator lets off-schema JSON through'
    assert verdict.negative_control.check_id is InstrumentCheckId.VALIDATOR_NEGATIVE_CONTROL
    assert verdict.negative_control.passed is False
    assert verdict.passed is False


@pytest.mark.asyncio
async def test_smoke_of_a_json_object_arm(mock_config):
    with mock_openai_server() as server:
        server.chat_content = VALID_ENTITIES
        spec = llm_spec(base_url=server.base_url, structured_output_mode='json_object')
        verdict = await smoke_endpoint(spec, mock_config)
        requests = _chat_requests(server)

    assert verdict.passed is True
    assert requests[0]['response_format'] == {'type': 'json_object'}


@pytest.mark.asyncio
async def test_an_unreachable_endpoint_fails_the_smoke_as_a_transport_error(mock_config):
    verdict = await smoke_endpoint(llm_spec(base_url=UNREACHABLE_BASE_URL), mock_config)

    assert verdict.passed is False
    assert 'APIConnectionError' in verdict.positive.detail
    assert verdict.counts.transport_errors >= 1


def _incumbent_on(server):
    return incumbent_control_spec(serving={'stack': 'openai', 'base_url': server.base_url})


@pytest.mark.asyncio
async def test_smoke_passes_on_the_incumbent_control_arm(mock_config):
    with mock_openai_server() as server:
        server.set_response('/responses', responses_body(VALID_ENTITIES))
        verdict = await smoke_endpoint(_incumbent_on(server), mock_config)

    assert verdict.passed is True
    assert verdict.counts.schema_valid >= 1


@pytest.mark.asyncio
async def test_smoke_detects_off_schema_json_on_the_incumbent_control_arm(mock_config):
    with mock_openai_server() as server:
        server.set_response('/responses', responses_body(OFF_SCHEMA))
        verdict = await smoke_endpoint(_incumbent_on(server), mock_config)

    assert verdict.passed is False
    assert 'ValidationError' in verdict.positive.detail


# --- boundary row 6: index-configuration reality -------------------------------------


class FakeProbeGraph:
    """An AsyncGraph double: writes recorded, each fulltext probe answered from a script.

    Each scripted answer is a row count, or an exception instance to raise.
    """

    def __init__(self, probe_answers: list[int | Exception]) -> None:
        self._answers = list(probe_answers)
        self.writes: list[tuple[str, dict | None]] = []
        self.probes: list[tuple[str, dict | None]] = []

    async def query(self, cypher: str, params: dict | None = None) -> Any:
        self.writes.append((cypher, params))
        return SimpleNamespace(result_set=[])

    async def ro_query(self, cypher: str, params: dict | None = None) -> Any:
        self.probes.append((cypher, params))
        answer = self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(result_set=[['uuid']] * answer)


class RecordingSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


async def _check(graph, expected, sleep=None, name=SCRATCH):
    return await check_index_configuration(graph, name, expected, sleep=sleep or RecordingSleep())


@pytest.mark.asyncio
async def test_index_check_refuses_a_protected_graph_before_any_query():
    graph = FakeProbeGraph([1])

    with pytest.raises(ScratchGuardError) as caught:
        await _check(graph, IndexConfiguration.WITH_INDICES, name='dark_factory')

    assert caught.value.checkpoint is GuardCheckpoint.INDEX_BUILD
    assert graph.writes == [] and graph.probes == []


@pytest.mark.asyncio
async def test_with_indices_passes_when_the_seeded_token_is_found():
    graph = FakeProbeGraph([1])

    result = await _check(graph, IndexConfiguration.WITH_INDICES)

    assert result.check_id is InstrumentCheckId.INDEX_CONFIGURATION
    assert result.passed is True, result.detail
    seed_cypher, seed_params = graph.writes[0]
    assert 'Entity' in seed_cypher
    probe_cypher, probe_params = graph.probes[0]
    assert "db.idx.fulltext.queryNodes('Entity'" in probe_cypher
    assert seed_params is not None and probe_params is not None
    assert seed_params['name'] in probe_params.values()


@pytest.mark.asyncio
async def test_the_seeded_probe_node_is_removed_afterwards():
    graph = FakeProbeGraph([1])

    await _check(graph, IndexConfiguration.WITH_INDICES)

    assert len(graph.writes) == 2
    (_, seed_params), (cleanup_cypher, cleanup_params) = graph.writes
    assert 'DELETE' in cleanup_cypher
    assert seed_params is not None
    assert cleanup_params == {'uuid': seed_params['uuid']}


@pytest.mark.asyncio
async def test_with_indices_polls_until_the_index_catches_up():
    graph, sleep = FakeProbeGraph([0, 0, 1]), RecordingSleep()

    result = await _check(graph, IndexConfiguration.WITH_INDICES, sleep=sleep)

    assert result.passed is True
    assert len(graph.probes) == 3
    assert sleep.calls == [INDEX_PROBE_INTERVAL_S] * 2


@pytest.mark.asyncio
async def test_with_indices_fails_when_no_row_ever_arrives_within_the_bound():
    graph, sleep = FakeProbeGraph([0]), RecordingSleep()

    result = await _check(graph, IndexConfiguration.WITH_INDICES, sleep=sleep)

    assert result.passed is False
    assert 'does not differ in fact' in result.detail
    assert len(graph.probes) == INDEX_PROBE_ATTEMPTS
    assert sum(sleep.calls) <= 5.0


@pytest.mark.asyncio
async def test_with_indices_fails_on_a_query_error():
    graph = FakeProbeGraph([ResponseError('no such index')])

    result = await _check(graph, IndexConfiguration.WITH_INDICES)

    assert result.passed is False
    assert 'does not differ in fact' in result.detail
    assert 'no such index' in result.detail


@pytest.mark.asyncio
async def test_embedding_only_passes_when_no_row_arrives():
    graph = FakeProbeGraph([0])

    result = await _check(graph, IndexConfiguration.EMBEDDING_ONLY)

    assert result.passed is True, result.detail
    assert len(graph.probes) == INDEX_PROBE_ATTEMPTS


@pytest.mark.asyncio
async def test_embedding_only_passes_on_a_missing_index_error():
    graph = FakeProbeGraph([ResponseError('no such index')])

    result = await _check(graph, IndexConfiguration.EMBEDDING_ONLY)

    assert result.passed is True, result.detail
    assert len(graph.probes) == 1


@pytest.mark.asyncio
async def test_embedding_only_fails_when_the_graph_answers_a_fulltext_query():
    graph = FakeProbeGraph([0, 2])

    result = await _check(graph, IndexConfiguration.EMBEDDING_ONLY)

    assert result.passed is False
    assert 'embedding-only' in result.detail


@pytest.mark.asyncio
async def test_a_connection_error_is_not_mistaken_for_a_missing_index():
    graph = FakeProbeGraph([ConnectionError('falkordb is down')])

    with pytest.raises(ConnectionError):
        await _check(graph, IndexConfiguration.EMBEDDING_ONLY)


# --- boundary row 4: control variance ------------------------------------------------

MEASURED_AT = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)


def _control(arm_id: str, scratch: str, **overrides):
    return incumbent_control_spec(arm_id=arm_id, scratch_group_id=scratch, **overrides)


def _accounted(spec) -> list[MetricsRecord]:
    def scalar(metric_id: LlmMetricId, value: float) -> MetricsRecord:
        metric = Metric(metric_id=metric_id, kind='scalar', value=value, n=3)
        return record_for(spec, metric, measured_at=MEASURED_AT, incomplete=False)

    return [scalar(LlmMetricId.TOKENS_PER_EPISODE, 900.0), scalar(LlmMetricId.USD_PER_EPISODE, 0.001)]


def _reference() -> list[EpisodeOutcome]:
    return [
        EpisodeOutcome(
            episode_id='e1',
            ok=True,
            error_class=None,
            duration_ms=10.0,
            tokens=None,
            replay_episode_uuid='r1',
            entity_names=('alice',),
            edge_triples=(),
        )
    ]


def test_two_symmetric_control_runs_pass_every_check():
    spec_a, spec_b = _control('ctrl-a', 'evalmem_ctrl_a'), _control('ctrl-b', 'evalmem_ctrl_b')
    runs = [run_manifest_for(spec_a), run_manifest_for(spec_b)]
    records = {'ctrl-a': _accounted(spec_a), 'ctrl-b': _accounted(spec_b)}

    results = control_variance_check(runs, records, reference=_reference())

    assert all(result.passed for result in results), results
    assert [result.check_id for result in results] == [
        InstrumentCheckId.ARM_CONFIG_SYMMETRY,
        InstrumentCheckId.SINGLE_CODE_SHA,
        InstrumentCheckId.TOKEN_COST_ACCOUNTING,
        InstrumentCheckId.TOKEN_COST_ACCOUNTING,
        InstrumentCheckId.REFERENCE_NONEMPTY,
    ]


def test_a_temperature_difference_is_a_symmetry_failure_naming_temperature():
    spec_a = _control('ctrl-a', 'evalmem_ctrl_a')
    spec_b = _control('ctrl-b', 'evalmem_ctrl_b', params={'temperature': 0.7, 'max_tokens': 4096})
    runs = [run_manifest_for(spec_a), run_manifest_for(spec_b)]
    records = {'ctrl-a': _accounted(spec_a), 'ctrl-b': _accounted(spec_b)}

    results = control_variance_check(runs, records)

    (symmetry,) = [r for r in results if r.check_id is InstrumentCheckId.ARM_CONFIG_SYMMETRY]
    assert symmetry.passed is False
    assert 'temperature' in symmetry.detail


def test_an_arm_without_records_fails_its_token_accounting():
    spec_a, spec_b = _control('ctrl-a', 'evalmem_ctrl_a'), _control('ctrl-b', 'evalmem_ctrl_b')
    runs = [run_manifest_for(spec_a), run_manifest_for(spec_b)]

    results = control_variance_check(runs, {'ctrl-a': _accounted(spec_a)})

    token_checks = [r for r in results if r.check_id is InstrumentCheckId.TOKEN_COST_ACCOUNTING]
    assert [check.passed for check in token_checks] == [True, False]
    assert 'ctrl-b' in token_checks[1].detail


def test_two_runs_of_one_arm_are_refused_before_their_records_collapse():
    spec = _control('ctrl-a', 'evalmem_ctrl_a')
    runs = [run_manifest_for(spec), run_manifest_for(spec)]

    with pytest.raises(RunComparabilityError, match='repeat arm ids'):
        control_variance_check(runs, {'ctrl-a': _accounted(spec)})


@pytest.mark.parametrize(
    ('overrides', 'message'),
    [
        ({'episode_ids': ('e1', 'e2')}, "lacks ['e3']"),
        ({'incomplete': True}, 'incomplete'),
    ],
    ids=['a-limited-run', 'an-incomplete-run'],
)
def test_control_runs_that_are_not_comparable_are_refused(overrides, message):
    spec_a, spec_b = _control('ctrl-a', 'evalmem_ctrl_a'), _control('ctrl-b', 'evalmem_ctrl_b')
    runs = [run_manifest_for(spec_a), run_manifest_for(spec_b, **overrides)]
    records = {'ctrl-a': _accounted(spec_a), 'ctrl-b': _accounted(spec_b)}

    with pytest.raises(RunComparabilityError) as raised:
        control_variance_check(runs, records)

    assert message in str(raised.value)
