"""Per-write LLM token attribution (task 3716).

One graphiti LLM client, and so one token tracker, is shared by every
concurrent write. These tests pin that a measurement window counts exactly
the tokens spent by its own asyncio context and nothing else.

No test here builds a ``Graphiti``, a ``FalkorDriver`` or calls
``GraphitiBackend.initialize()``, and none touches the network.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from graphiti_core.llm_client.token_tracker import TokenUsageTracker

from fused_memory.backends.graphiti_client import GraphitiBackend
from fused_memory.backends.llm_token_usage import (
    AttributingTokenUsageTracker,
    LlmTokenUsage,
    measure_llm_tokens,
)


def _attributing_client() -> SimpleNamespace:
    return SimpleNamespace(token_tracker=AttributingTokenUsageTracker())


@pytest.mark.asyncio
async def test_window_reports_the_tokens_recorded_inside_it():
    client = _attributing_client()

    async with measure_llm_tokens(client) as m:
        client.token_tracker.record('extract_nodes', 30, 10)

    assert m.usage == LlmTokenUsage(input_tokens=30, output_tokens=10, llm_calls=1)
    assert m.usage.total_tokens == 40
    assert m.usage.as_journal_dict() == {
        'input_tokens': 30,
        'output_tokens': 10,
        'total_tokens': 40,
        'llm_calls': 1,
    }


@pytest.mark.asyncio
async def test_records_outside_the_window_are_excluded_but_still_cumulative():
    client = _attributing_client()
    client.token_tracker.record('before', 1000, 100)

    async with measure_llm_tokens(client) as m:
        client.token_tracker.record('inside', 30, 10)

    client.token_tracker.record('after', 2000, 200)

    assert m.usage == LlmTokenUsage(30, 10, 1)
    total = client.token_tracker.get_total_usage()
    assert (total.input_tokens, total.output_tokens) == (3030, 310)


@pytest.mark.asyncio
async def test_concurrent_windows_over_one_tracker_each_count_only_their_own():
    client = _attributing_client()
    record = client.token_tracker.record
    a_open, b_open = asyncio.Event(), asyncio.Event()
    a_first, b_recorded, a_second = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def window_a():
        async with measure_llm_tokens(client) as m:
            a_open.set()
            await b_open.wait()
            record('extract_nodes', 100, 40)
            a_first.set()
            await b_recorded.wait()
            record('extract_edges', 5, 5)
            a_second.set()
        return m

    async def window_b():
        await a_open.wait()
        async with measure_llm_tokens(client) as m:
            b_open.set()
            await a_first.wait()
            record('dedupe_nodes', 7, 3)
            b_recorded.set()
            await a_second.wait()
        return m

    a, b = await asyncio.wait_for(
        asyncio.gather(asyncio.create_task(window_a()), asyncio.create_task(window_b())),
        10,
    )

    assert a.usage == LlmTokenUsage(105, 45, 2)
    assert b.usage == LlmTokenUsage(7, 3, 1)


@pytest.mark.asyncio
async def test_records_from_child_tasks_spawned_in_the_window_are_attributed():
    client = _attributing_client()

    async def _llm_call(prompt_name: str, input_tokens: int, output_tokens: int):
        await asyncio.sleep(0)
        client.token_tracker.record(prompt_name, input_tokens, output_tokens)

    async with measure_llm_tokens(client) as m:
        await asyncio.gather(
            _llm_call('extract_nodes', 10, 1),
            _llm_call('extract_edges', 20, 2),
        )

    assert m.usage == LlmTokenUsage(30, 3, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'llm_client',
    [None, SimpleNamespace(token_tracker=TokenUsageTracker())],
    ids=['no-client', 'non-attributing-tracker'],
)
async def test_unmeasurable_client_yields_no_usage_and_still_runs_the_body(llm_client):
    body_ran = False

    async with measure_llm_tokens(llm_client) as m:
        body_ran = True

    assert body_ran
    assert m.usage is None


@pytest.mark.asyncio
async def test_measured_window_without_llm_calls_is_a_real_zero():
    async with measure_llm_tokens(_attributing_client()) as m:
        pass

    assert m.usage == LlmTokenUsage(0, 0, 0)


@pytest.mark.asyncio
async def test_raising_body_propagates_and_still_freezes_the_tokens_it_burned():
    client = _attributing_client()

    with pytest.raises(RuntimeError, match='extraction failed'):
        async with measure_llm_tokens(client) as m:
            client.token_tracker.record('extract_nodes', 10, 2)
            raise RuntimeError('extraction failed')

    assert m.usage == LlmTokenUsage(10, 2, 1)


class TestGraphitiBackendTokenProbe:
    @pytest.mark.asyncio
    async def test_without_an_llm_client_the_probe_measures_nothing(self, mock_config):
        backend = GraphitiBackend(mock_config)

        async with backend.token_probe() as m:
            pass

        assert m.usage is None

    @pytest.mark.asyncio
    async def test_probe_counts_tokens_on_the_backend_llm_client(self, mock_config):
        backend = GraphitiBackend(mock_config)
        client = _attributing_client()
        backend._llm_client = client

        async with backend.token_probe() as m:
            client.token_tracker.record('extract_nodes', 12, 3)

        assert m.usage == LlmTokenUsage(12, 3, 1)
