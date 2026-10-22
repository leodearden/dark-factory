"""The overreach shadow's wiring into MemoryService and the config (task 6137).

Write path, end to end through the public surface: add_memory enqueues onto a
real durable queue, the queue's workers run the write, and a guarded write is
judged into the real sidecar that initialize() opened. Only the judge LLM and
the graph are doubles. Lifecycle: the shadow opens in initialize() and closes
between the durable queue and graphiti. Config: the overreach_guard leaves
hot-reload.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from _fm_helpers import install_identity_mocks, poll_until

from fused_memory.config.reload import RELOADABLE_FIELDS, apply_reload
from fused_memory.server.main import _MEMORY_CLOSE_STEP_TIMEOUT
from fused_memory.services.memory_service import MemoryService
from fused_memory.services.overreach_shadow import (
    OverreachShadow,
    ShadowJob,
    open_overreach_shadow,
)

RULING = 'RULING (Leo, 2026-09-01, esc-1-2): the angular gate narrows to ANGLE only.'
GROUP = 'dark_factory'


def _service(config) -> MemoryService:
    svc = MemoryService(config)
    svc.graphiti = MagicMock()
    svc.graphiti.initialize = AsyncMock()
    svc.graphiti.close = AsyncMock()
    svc.graphiti.add_episode = AsyncMock(return_value=SimpleNamespace(
        episode=SimpleNamespace(uuid='ep-1'), edges=[SimpleNamespace(uuid='e1')],
    ))
    install_identity_mocks(svc.graphiti)
    svc.mem0 = MagicMock()
    svc.mem0.add = AsyncMock(return_value={'results': [{'id': 'mem0-1'}]})
    svc.mem0.close = AsyncMock()
    return svc


@pytest.fixture
def service(mock_config):
    svc = _service(mock_config)
    svc.durable_queue = MagicMock()
    svc.durable_queue.enqueue = AsyncMock(return_value=1)
    svc.durable_queue.close = AsyncMock()
    return svc


class _ClosingShadow(OverreachShadow):
    """Records its close into *order*, or raises from submit.

    Deliberately skips OverreachShadow.__init__: it has no store and no LLM.
    """

    def __init__(self, *, order: list[str] | None = None, error: Exception | None = None) -> None:
        self.order, self.error = order, error

    def submit(self, job: ShadowJob) -> bool:
        raise self.error or AssertionError('this shadow is never submitted to')

    async def close(self) -> None:
        if self.order is not None:
            self.order.append('overreach_shadow')


class _Graph:
    """Each write mints one edge; the survivor read returns it, live."""

    def __init__(self) -> None:
        self.minted: dict[str, str] = {}

    async def add_episode(self, *args: Any, **kwargs: Any) -> Any:
        number = len(self.minted) + 1
        episode, edge = f'ep-{number}', f'edge-{number}'
        self.minted[edge] = episode
        return SimpleNamespace(episode=SimpleNamespace(uuid=episode), edges=[SimpleNamespace(uuid=edge)])

    async def get_edges_by_uuids(self, uuids, *, group_id):
        return [
            {'uuid': uuid, 'fact': f'fact of {uuid}', 'source_name': 'S', 'target_name': 'T',
             'episodes': [self.minted[uuid]], 'active': True}
            for uuid in uuids if uuid in self.minted
        ]


class _Judge:
    """Labels every edge overreach, but only once released; records each call."""

    def __init__(self) -> None:
        self.calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def generate_response(self, messages, response_model=None, max_tokens=None, **kwargs):
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        return {'judgements': [{'edge': 'E1', 'label': 'overreach', 'rationale': 'the whole span'}]}


@pytest_asyncio.fixture
async def live(mock_config):
    """An initialized MemoryService: real queue, real shadow, real sidecar."""
    mock_config.overreach_guard.shadow_enabled = True
    mock_config.queue.workers_per_group = 2
    svc = _service(mock_config)
    graph, judge = _Graph(), _Judge()
    svc.graphiti.add_episode = AsyncMock(side_effect=graph.add_episode)
    svc.graphiti.get_edges_by_uuids = AsyncMock(side_effect=graph.get_edges_by_uuids)
    with patch('fused_memory.services.overreach_shadow.build_llm_client', return_value=judge):
        await svc.initialize()
    yield SimpleNamespace(service=svc, judge=judge)
    judge.release.set()
    await svc.close()


async def _completed(svc: MemoryService, count: int) -> None:
    async def done() -> bool:
        assert svc.durable_queue is not None
        stats = await svc.durable_queue.get_stats()
        return stats['counts'].get('completed') == count
    await poll_until(done, message=f'the queue never completed {count} write(s)')


async def _flagged(svc: MemoryService) -> list[str]:
    assert svc.overreach_shadow is not None
    await svc.overreach_shadow.drain()
    page = await svc.overreach_shadow.overreach_flag_page(GROUP, 10)
    return [flag.episode_uuid for flag in page.flags]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


class TestHotReload:
    def test_both_leaves_are_green_tier(self) -> None:
        assert {'overreach_guard.shadow_enabled', 'overreach_guard.judge_timeout_seconds'} <= RELOADABLE_FIELDS

    def test_the_kill_switch_applies_without_a_restart(self, mock_config) -> None:
        mock_config.overreach_guard.shadow_enabled = True
        fresh = mock_config.model_copy(deep=True)
        fresh.overreach_guard.shadow_enabled = False
        report = apply_reload(mock_config, fresh)
        assert 'overreach_guard.shadow_enabled' in report['applied']
        assert 'overreach_guard.shadow_enabled' not in report['restart_required']
        assert mock_config.overreach_guard.shadow_enabled is False

    @pytest.mark.asyncio
    async def test_a_live_runner_observes_the_reload(self, mock_config, tmp_path) -> None:
        mock_config.queue.data_dir = str(tmp_path)
        mock_config.overreach_guard.shadow_enabled = True
        no_survivors = AsyncMock(return_value=[])
        runner = await open_overreach_shadow(mock_config, read_edges=no_survivors)
        assert runner is not None
        job = ShadowJob(graph='g', episode_uuid='ep-1', content='x', edge_uuids=('e1',), classifiers=())
        try:
            assert runner.submit(job) is True
            await runner.drain()
            fresh = mock_config.model_copy(deep=True)
            fresh.overreach_guard.shadow_enabled = False
            apply_reload(mock_config, fresh)
            assert runner.submit(job) is False
        finally:
            await runner.close()


class TestMockConfigPinsTheShadowOff:
    def test_the_shared_fixture_never_reaches_a_live_judge(self, mock_config) -> None:
        assert mock_config.overreach_guard.shadow_enabled is False


# --------------------------------------------------------------------------- #
# The write path
# --------------------------------------------------------------------------- #


class TestTheWritePath:
    @pytest.mark.asyncio
    async def test_a_decision_write_is_judged_into_the_sidecar(self, live) -> None:
        live.judge.release.set()
        await live.service.add_memory(
            content=RULING, category='decisions_and_rationale', project_id=GROUP,
        )
        await _completed(live.service, 1)
        assert await _flagged(live.service) == ['ep-1']
        assert 'memory_category' not in live.service.graphiti.add_episode.await_args.kwargs

    @pytest.mark.asyncio
    async def test_a_slow_judge_holds_up_neither_its_write_nor_a_same_group_one(self, live) -> None:
        svc = live.service
        await svc.add_memory(content=RULING, category='decisions_and_rationale', project_id=GROUP)
        await poll_until(live.judge.entered.is_set, message='the first judge never started')
        await svc.add_memory(content=RULING + ' Again.', category='decisions_and_rationale',
                             project_id=GROUP)
        await poll_until(lambda: svc.graphiti.add_episode.await_count == 2,
                         message='the second write was serialized behind the first judge')
        await _completed(svc, 2)
        assert not live.judge.release.is_set()
        live.judge.release.set()
        assert await _flagged(svc) == ['ep-1', 'ep-2']

    @pytest.mark.asyncio
    async def test_another_category_is_not_judged(self, live) -> None:
        await live.service.add_memory(content=RULING, category='temporal_facts', project_id=GROUP)
        await _completed(live.service, 1)
        assert (await _flagged(live.service), live.judge.calls) == ([], 0)

    @pytest.mark.asyncio
    async def test_add_episode_is_not_judged(self, live) -> None:
        await live.service.add_episode(content=RULING, project_id=GROUP)
        await _completed(live.service, 1)
        assert (await _flagged(live.service), live.judge.calls) == ([], 0)

    @pytest.mark.parametrize('category', [
        pytest.param(None, id='enqueued-before-deploy'),
        pytest.param('bogus', id='unknown-value'),
    ])
    @pytest.mark.asyncio
    async def test_a_queued_row_without_a_known_category_is_not_judged(self, live, category) -> None:
        payload: dict[str, Any] = {
            'name': 'memory_decisions_and_rationale', 'content': RULING, 'source': 'text',
            'group_id': GROUP, 'source_description': 'add_memory:decisions_and_rationale',
        }
        if category is not None:
            payload['memory_category'] = category
        assert live.service.durable_queue is not None
        await live.service.durable_queue.enqueue(
            group_id=GROUP, operation='add_memory_graphiti', payload=payload,
        )
        await _completed(live.service, 1)
        assert (await _flagged(live.service), live.judge.calls) == ([], 0)

    @pytest.mark.asyncio
    async def test_a_result_naming_no_episode_is_not_judged(self, live) -> None:
        live.service.graphiti.add_episode = AsyncMock(return_value=None)
        await live.service.add_memory(
            content=RULING, category='decisions_and_rationale', project_id=GROUP,
        )
        await _completed(live.service, 1)
        assert (await _flagged(live.service), live.judge.calls) == ([], 0)

    @pytest.mark.parametrize('shadow', [
        pytest.param(None, id='no-shadow'),
        pytest.param(_ClosingShadow(error=RuntimeError('shadow down')), id='raising-submit'),
    ])
    @pytest.mark.asyncio
    async def test_the_write_lands_whatever_the_shadow_does(self, live, shadow) -> None:
        opened = live.service.overreach_shadow
        live.service.overreach_shadow = shadow
        try:
            await live.service.add_memory(
                content=RULING, category='decisions_and_rationale', project_id=GROUP,
            )
            await _completed(live.service, 1)
        finally:
            live.service.overreach_shadow = opened
        assert live.service.graphiti.add_episode.await_count == 1


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_the_shadow_closes_after_the_queue_and_before_graphiti(self, service) -> None:
        order: list[str] = []
        service.durable_queue.close = AsyncMock(side_effect=lambda: order.append('durable_queue'))
        service.graphiti.close = AsyncMock(side_effect=lambda: order.append('graphiti'))
        service.overreach_shadow = _ClosingShadow(order=order)
        await service.close()
        assert order == ['durable_queue', 'overreach_shadow', 'graphiti']

    def test_the_close_budget_still_fits_the_server_step(self, mock_config) -> None:
        assert MemoryService(mock_config).close_budget_secs() <= _MEMORY_CLOSE_STEP_TIMEOUT

    @pytest.mark.asyncio
    async def test_no_shadow_is_skipped_on_close(self, service) -> None:
        service.overreach_shadow = None
        await service.close()

    @pytest.mark.asyncio
    async def test_initialize_opens_the_shadow(self, service) -> None:
        opened = MagicMock()
        opener = AsyncMock(return_value=opened)
        await self._initialize(service, opener)
        assert service.overreach_shadow is opened
        opener.assert_awaited_once_with(
            service.config, read_edges=service.graphiti.get_edges_by_uuids,
        )

    @pytest.mark.asyncio
    async def test_a_failing_opener_leaves_no_shadow_and_initialize_completes(self, service) -> None:
        queue = await self._initialize(service, AsyncMock(side_effect=OSError('read-only fs')))
        assert service.overreach_shadow is None
        queue.initialize.assert_awaited_once()

    @staticmethod
    async def _initialize(service: MemoryService, opener: AsyncMock) -> MagicMock:
        queue = MagicMock()
        queue.initialize = AsyncMock()
        registry = MagicMock()
        registry.initialize = AsyncMock()
        service.graphiti.initialize = AsyncMock()
        with (
            patch('fused_memory.services.memory_service.DurableWriteQueue', return_value=queue),
            patch('fused_memory.services.planned_episode_registry.PlannedEpisodeRegistry',
                  MagicMock(return_value=registry)),
            patch('fused_memory.services.memory_service.open_overreach_shadow', opener),
        ):
            await service.initialize()
        return queue
