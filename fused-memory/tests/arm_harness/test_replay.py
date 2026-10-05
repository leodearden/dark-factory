"""The replay engine: episodes onto an arm graph, INV-4 abort, journaled telemetry."""

import sqlite3
from datetime import UTC, datetime

import pytest
from _mock_openai_server import mock_openai_server
from graphiti_core.nodes import EpisodeType
from graphiti_core.prompts.models import Message

import fused_memory.arm_harness.replay as replay_module
from arm_harness._fakes import (
    FakeArmGraph,
    RecordingJournal,
    fail,
    fake_add_result,
    hang,
    llm_spec,
    slow,
    succeed,
)
from fused_memory.arm_harness.arm_config import llm_arm_config
from fused_memory.arm_harness.arm_spec import LlmArmSpec
from fused_memory.arm_harness.metrics_record import IndexConfiguration
from fused_memory.arm_harness.replay import (
    MAX_CONSECUTIVE_FAILURES,
    ArmAbort,
    ConsecutiveFailureBreaker,
    ReplayItem,
    ReplaySettings,
    default_replay_settings,
    open_arm_backend,
    replay_arm,
)
from fused_memory.arm_harness.scratch_guard import GuardCheckpoint, ScratchGuardError
from fused_memory.backends.graphiti_client import build_llm_client
from fused_memory.services.write_journal import OPERATOR_TELEMETRY_QUERY, WriteJournal

REFERENCE_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def _items(count: int) -> list[ReplayItem]:
    return [
        ReplayItem(
            episode_id=f'ep-{i}',
            name=f'ep-{i}',
            content=f'content of episode {i}',
            source_description='lme corpus',
            reference_time=REFERENCE_TIME,
        )
        for i in range(count)
    ]


def _settings(concurrency: int = 1, **overrides) -> ReplaySettings:
    fields = {
        'concurrency': concurrency,
        'episode_timeout_s': 120.0,
        'index_configuration': IndexConfiguration.WITH_INDICES,
    }
    return ReplaySettings(**(fields | overrides))


async def _replay(graph, items, *, spec=None, journal=None, **settings):
    return await replay_arm(
        spec or llm_spec(),
        items,
        graph=graph,
        journal=journal or RecordingJournal(),
        settings=_settings(**settings),
    )


# ── INV-4 consecutive-failure abort ──


@pytest.mark.asyncio
async def test_five_consecutive_failures_abort_and_nothing_after_is_dispatched():
    graph = FakeArmGraph(default=fail)
    spec = llm_spec()

    result = await _replay(graph, _items(20), spec=spec)

    assert len(graph.add_calls) == MAX_CONSECUTIVE_FAILURES == 5
    assert result.incomplete is True
    assert result.abort == ArmAbort(
        arm_id=spec.arm_id,
        item_ids=tuple(f'ep-{i}' for i in range(5)),
        error_classes=('RuntimeError',) * 5,
    )
    assert result.cancelled_ids == ()


@pytest.mark.asyncio
async def test_abort_under_concurrency_cancels_and_lists_in_flight_work():
    behaviours = {f'ep-{i}': hang for i in range(5, 20)}
    graph = FakeArmGraph(behaviours, default=fail)

    result = await _replay(graph, _items(20), concurrency=4)

    assert result.abort is not None
    assert result.abort.item_ids == tuple(f'ep-{i}' for i in range(5))
    assert set(result.cancelled_ids) == {'ep-5', 'ep-6', 'ep-7'}
    assert len(graph.add_calls) <= 5 + 3
    assert {o.episode_id for o in result.outcomes}.isdisjoint(result.cancelled_ids)


@pytest.mark.asyncio
async def test_a_success_resets_the_consecutive_count():
    behaviours = {f'ep-{i}': fail for i in range(4)} | {f'ep-{i}': fail for i in range(5, 9)}
    graph = FakeArmGraph(behaviours)

    result = await _replay(graph, _items(10))

    assert result.abort is None
    assert result.incomplete is False
    assert len(result.outcomes) == 10


@pytest.mark.asyncio
async def test_alternating_failures_never_abort():
    behaviours = {f'ep-{i}': fail for i in range(0, 20, 2)}
    graph = FakeArmGraph(behaviours)

    result = await _replay(graph, _items(20))

    assert result.abort is None
    assert sum(not o.ok for o in result.outcomes) == 10


@pytest.mark.asyncio
async def test_failures_are_counted_in_completion_order():
    items = _items(6)
    behaviours = {'ep-0': fail, 'ep-1': slow(succeed, 5.0)}
    graph = FakeArmGraph(behaviours, default=fail)

    result = await _replay(graph, items, concurrency=2)

    assert result.abort is not None
    assert result.abort.item_ids == ('ep-0', 'ep-2', 'ep-3', 'ep-4', 'ep-5')
    assert result.cancelled_ids == ('ep-1',)


def test_breaker_trips_on_the_fifth_consecutive_failure():
    breaker = ConsecutiveFailureBreaker(arm_id='arm-a')

    trips = [breaker.observe(f'ep-{i}', 'TimeoutError') for i in range(5)]

    assert trips[:4] == [None] * 4
    assert trips[4] == ArmAbort(
        arm_id='arm-a',
        item_ids=tuple(f'ep-{i}' for i in range(5)),
        error_classes=('TimeoutError',) * 5,
    )


def test_breaker_resets_on_success_and_honours_its_limit():
    breaker = ConsecutiveFailureBreaker(arm_id='arm-a', limit=2)

    assert breaker.observe('ep-0', 'RuntimeError') is None
    assert breaker.observe('ep-1', None) is None
    assert breaker.observe('ep-2', 'RuntimeError') is None
    abort = breaker.observe('ep-3', 'ValueError')

    assert abort is not None
    assert abort.item_ids == ('ep-2', 'ep-3')
    assert abort.error_classes == ('RuntimeError', 'ValueError')


def test_breaker_rejects_a_non_positive_limit():
    with pytest.raises(ValueError, match='limit'):
        ConsecutiveFailureBreaker(arm_id='arm-a', limit=0)


# ── timeouts ──


@pytest.mark.asyncio
async def test_timeout_error_is_a_failure():
    async def time_out(call):
        raise TimeoutError

    graph = FakeArmGraph({'ep-0': time_out})

    result = await _replay(graph, _items(1))

    (outcome,) = result.outcomes
    assert (outcome.ok, outcome.error_class) == (False, 'TimeoutError')


@pytest.mark.asyncio
async def test_a_success_over_the_episode_budget_is_a_failure():
    ticks = iter([0.0, 121.0, 200.0, 200.5])
    graph = FakeArmGraph()

    result = await _replay(graph, _items(2), clock=lambda: next(ticks))

    over, within = result.outcomes
    assert (over.ok, over.error_class, over.duration_ms) == (False, 'EpisodeOverBudget', 121000.0)
    assert (within.ok, within.error_class, within.duration_ms) == (True, None, 500.0)


def test_default_settings_read_the_production_write_timeout(mock_config):
    settings = default_replay_settings(
        mock_config, concurrency=3, index_configuration=IndexConfiguration.EMBEDDING_ONLY
    )

    assert settings.episode_timeout_s == mock_config.queue.backend_write_timeout_seconds == 120
    assert settings.concurrency == 3
    assert settings.index_configuration is IndexConfiguration.EMBEDDING_ONLY


def test_default_settings_follow_a_changed_write_timeout(mock_config):
    config = mock_config.model_copy(deep=True)
    config.queue.backend_write_timeout_seconds = 7.0

    settings = default_replay_settings(
        config, concurrency=1, index_configuration=IndexConfiguration.WITH_INDICES
    )

    assert settings.episode_timeout_s == 7.0


def test_settings_reject_zero_concurrency():
    with pytest.raises(ValueError, match='concurrency'):
        _settings(concurrency=0)


# ── concurrency and writes ──


@pytest.mark.asyncio
async def test_in_flight_writes_never_exceed_the_concurrency():
    graph = FakeArmGraph(default=slow(succeed, 0.01))

    result = await _replay(graph, _items(12), concurrency=3)

    assert graph.max_in_flight == 3
    assert len(result.outcomes) == 12


@pytest.mark.asyncio
async def test_every_write_targets_the_scratch_graph_with_the_item_fields():
    spec = llm_spec()
    graph = FakeArmGraph()
    items = _items(3)

    await _replay(graph, items, spec=spec)

    assert len(graph.add_calls) == 3
    for item, call in zip(items, graph.add_calls, strict=True):
        assert call == {
            'name': item.name,
            'content': item.content,
            'source': EpisodeType.text,
            'group_id': spec.scratch_group_id,
            'source_description': item.source_description,
            'reference_time': item.reference_time,
        }


@pytest.mark.asyncio
async def test_each_episode_is_journaled_as_a_write_op_and_a_backend_op():
    spec = llm_spec()
    journal = RecordingJournal()
    graph = FakeArmGraph({'ep-1': fail})

    await _replay(graph, _items(2), spec=spec, journal=journal)

    names = [name for name, _ in journal.calls]
    assert names == ['log_write_op', 'log_backend_op'] * 2
    write_op, backend_op = journal.calls[2][1], journal.calls[3][1]
    assert write_op['operation'] == 'arm_replay_episode'
    assert write_op['project_id'] == spec.scratch_group_id
    assert write_op['params'] == {'arm_id': spec.arm_id, 'episode_id': 'ep-1'}
    assert backend_op['write_op_id'] == write_op['write_op_id']
    assert backend_op['backend'] == 'graphiti'
    assert backend_op['operation'] == 'add_episode'
    assert backend_op['success'] is False
    assert backend_op['error'] == 'RuntimeError'


# ── outcome extraction ──


@pytest.mark.asyncio
async def test_outcome_carries_normalised_entities_edge_triples_and_the_episode_uuid():
    async def extracted(call):
        return fake_add_result(
            'replay-uuid-7',
            entity_names=('  Alice   Smith ', 'BOB', 'alice smith'),
            edges=(('  Alice   Smith ', 'KNOWS', 'BOB'),),
        )

    graph = FakeArmGraph({'ep-0': extracted})

    result = await _replay(graph, _items(1))

    (outcome,) = result.outcomes
    assert outcome.ok is True
    assert outcome.replay_episode_uuid == 'replay-uuid-7'
    assert outcome.entity_names == ('alice smith', 'bob')
    assert outcome.edge_triples == (('alice smith', 'KNOWS', 'bob'),)
    assert outcome.tokens is not None
    assert outcome.tokens.total_tokens == 40


@pytest.mark.asyncio
async def test_failed_outcome_carries_no_graph_facts():
    graph = FakeArmGraph({'ep-0': fail})

    result = await _replay(graph, _items(1))

    (outcome,) = result.outcomes
    assert (outcome.replay_episode_uuid, outcome.entity_names, outcome.edge_triples) == (
        None,
        (),
        (),
    )


# ── BOUNDARY ROW 2: redundant scratch-guard enforcement ──


def _bypassed_spec() -> LlmArmSpec:
    valid = llm_spec()
    return LlmArmSpec.model_construct(
        **(dict(valid) | {'scratch_group_id': 'dark_factory'})
    )


@pytest.mark.asyncio
async def test_replay_refuses_a_validation_bypassed_spec_before_any_call():
    graph = FakeArmGraph()
    journal = RecordingJournal()

    with pytest.raises(ScratchGuardError) as caught:
        await _replay(graph, _items(3), spec=_bypassed_spec(), journal=journal)

    assert caught.value.checkpoint is GuardCheckpoint.REPLAY
    assert graph.add_calls == []
    assert graph.events == []
    assert journal.calls == []


@pytest.mark.asyncio
async def test_open_arm_backend_refuses_before_building_anything(mock_config, monkeypatch):
    constructed: list[object] = []

    class SpyBackend:
        def __init__(self, *args, **kwargs):
            constructed.append((args, kwargs))

    monkeypatch.setattr(replay_module, 'GraphitiBackend', SpyBackend)

    with pytest.raises(ScratchGuardError) as caught:
        async with open_arm_backend(_bypassed_spec(), mock_config, _settings()):
            pytest.fail('a guarded backend must never be yielded')

    assert caught.value.checkpoint is GuardCheckpoint.REPLAY
    assert constructed == []


# ── BOUNDARY ROW 7: telemetry presence ──


@pytest.mark.asyncio
async def test_replayed_episode_journal_row_carries_duration_and_tokens(mock_config, tmp_path):
    journal = WriteJournal(tmp_path / 'journal')
    await journal.initialize()
    try:
        with mock_openai_server() as server:
            spec = llm_spec(base_url=server.base_url)
            client = build_llm_client(llm_arm_config(spec, mock_config))
            assert client is not None

            async def extract_with_real_llm(call):
                await client.generate_response([Message(role='user', content=call['content'])])
                return fake_add_result('replay-uuid-0', entity_names=('alice',))

            graph = FakeArmGraph({'ep-0': extract_with_real_llm}, usage=None, llm_client=client)

            result = await replay_arm(
                spec, _items(1), graph=graph, journal=journal, settings=_settings()
            )
    finally:
        await journal.close()

    with sqlite3.connect(tmp_path / 'journal' / 'write_journal.db') as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(OPERATOR_TELEMETRY_QUERY, ('1970-01-01', 10)).fetchall()

    (outcome,) = result.outcomes
    assert outcome.tokens is not None
    assert outcome.tokens.total_tokens > 0
    (row,) = [r for r in rows if r['operation'] == 'arm_replay_episode']
    assert row['project_id'] == spec.scratch_group_id
    assert row['duration_ms'] > 0
    assert row['total_tokens'] > 0
