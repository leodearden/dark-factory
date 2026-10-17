"""The overreach shadow's wiring into MemoryService and the config (task 6137).

Write path: add_memory carries the resolved category on the durable-queue
payload, _execute_graphiti_write pops it, and a guarded write submits ONE
shadow job after the identity lock is released (design.md §6 'Lock
placement'). Lifecycle: the shadow opens in initialize() and closes between the
durable queue and graphiti. Config: the overreach_guard leaves hot-reload.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _fm_helpers import install_identity_mocks

from fused_memory.config.reload import RELOADABLE_FIELDS, apply_reload
from fused_memory.server.main import _MEMORY_CLOSE_STEP_TIMEOUT
from fused_memory.services.memory_service import MemoryService
from fused_memory.services.overreach_shadow import (
    OverreachShadow,
    ShadowJob,
    open_overreach_shadow,
)

RULING = 'RULING (Leo, 2026-09-01, esc-1-2): the angular gate narrows to ANGLE only.'


@pytest.fixture
def service(mock_config):
    svc = MemoryService(mock_config)
    svc.graphiti = MagicMock()
    svc.graphiti.add_episode = AsyncMock(return_value=SimpleNamespace(
        episode=SimpleNamespace(uuid='ep-1'), edges=[SimpleNamespace(uuid='e1')],
    ))
    svc.graphiti._require_client = MagicMock()
    svc.graphiti.close = AsyncMock()
    install_identity_mocks(svc.graphiti)
    svc.mem0 = MagicMock()
    svc.mem0.add = AsyncMock(return_value={'results': [{'id': 'mem0-1'}]})
    svc.mem0.close = AsyncMock()
    svc.durable_queue = MagicMock()
    svc.durable_queue.enqueue = AsyncMock(return_value=1)
    svc.durable_queue.enqueue_batch = AsyncMock(return_value=[])
    svc.durable_queue.close = AsyncMock()
    return svc


class _RecordingShadow(OverreachShadow):
    """Records each submit and whether the group's identity lock was held at the time.

    Deliberately skips OverreachShadow.__init__: it has no store and no LLM.
    """

    def __init__(self, svc: MemoryService, *, error: Exception | None = None,
                 order: list[str] | None = None) -> None:
        self.svc, self.error, self.order = svc, error, order
        self.jobs: list[ShadowJob] = []
        self.lock_held: list[bool] = []

    def submit(self, job: ShadowJob) -> bool:
        self.lock_held.append(self.svc.graphiti._identity_lock_for(job.graph).locked())
        self.jobs.append(job)
        if self.error is not None:
            raise self.error
        return True

    async def close(self) -> None:
        if self.order is not None:
            self.order.append('overreach_shadow')


def _payload(**overrides):
    payload = {
        'name': 'memory_decisions_and_rationale', 'content': RULING, 'source': 'text',
        'group_id': 'dark_factory', 'source_description': 'add_memory:decisions_and_rationale',
        'memory_category': 'decisions_and_rationale',
    }
    payload.update(overrides)
    return payload


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


class TestAddMemoryCarriesTheCategory:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('category', ['decisions_and_rationale', 'temporal_facts'])
    async def test_the_payload_carries_the_resolved_category(self, service, category) -> None:
        await service.add_memory(content=RULING, category=category, project_id='dark_factory')
        payload = service.durable_queue.enqueue.call_args.kwargs['payload']
        assert service.durable_queue.enqueue.call_args.kwargs['operation'] == 'add_memory_graphiti'
        assert payload['memory_category'] == category

    @pytest.mark.asyncio
    async def test_add_episode_carries_no_category(self, service) -> None:
        await service.add_episode(content=RULING, project_id='dark_factory')
        assert 'memory_category' not in service.durable_queue.enqueue.call_args.kwargs['payload']


class TestExecuteGraphitiWrite:
    @pytest.mark.asyncio
    async def test_a_guarded_write_submits_one_job_after_the_lock(self, service) -> None:
        shadow = _RecordingShadow(service)
        service.overreach_shadow = shadow
        await service._execute_graphiti_write('add_memory_graphiti', _payload())
        (job,) = shadow.jobs
        assert (job.graph, job.episode_uuid, job.edge_uuids, job.content) == (
            'dark_factory', 'ep-1', ('e1',), RULING,
        )
        assert 'category_decisions' in job.classifiers
        assert shadow.lock_held == [False]

    @pytest.mark.asyncio
    async def test_the_category_never_reaches_the_backend(self, service) -> None:
        service.overreach_shadow = _RecordingShadow(service)
        payload = _payload()
        await service._execute_graphiti_write('add_memory_graphiti', payload)
        assert 'memory_category' not in payload
        assert 'memory_category' not in service.graphiti.add_episode.call_args.kwargs

    @pytest.mark.asyncio
    @pytest.mark.parametrize('payload', [
        pytest.param(_payload(memory_category='temporal_facts'), id='not-guarded'),
        pytest.param({k: v for k, v in _payload().items() if k != 'memory_category'}, id='absent-key'),
        pytest.param(_payload(memory_category='bogus'), id='unknown-value'),
    ])
    async def test_an_out_of_class_write_submits_nothing(self, service, payload) -> None:
        shadow = _RecordingShadow(service)
        service.overreach_shadow = shadow
        await service._execute_graphiti_write('add_memory_graphiti', dict(payload))
        assert shadow.jobs == []

    @pytest.mark.asyncio
    async def test_no_shadow_means_nothing_to_submit(self, service) -> None:
        service.overreach_shadow = None
        assert await service._execute_graphiti_write('add_memory_graphiti', _payload()) is not None

    @pytest.mark.asyncio
    async def test_a_none_result_submits_nothing(self, service) -> None:
        shadow = _RecordingShadow(service)
        service.overreach_shadow = shadow
        service.graphiti.add_episode = AsyncMock(return_value=None)
        await service._execute_graphiti_write('add_memory_graphiti', _payload())
        assert shadow.jobs == []

    @pytest.mark.asyncio
    async def test_a_raising_submit_never_fails_the_write(self, service) -> None:
        service.overreach_shadow = _RecordingShadow(service, error=RuntimeError('shadow down'))
        result = await service._execute_graphiti_write('add_memory_graphiti', _payload())
        assert result.episode.uuid == 'ep-1'


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_the_shadow_closes_after_the_queue_and_before_graphiti(self, service) -> None:
        order: list[str] = []
        service.durable_queue.close = AsyncMock(side_effect=lambda: order.append('durable_queue'))
        service.graphiti.close = AsyncMock(side_effect=lambda: order.append('graphiti'))
        service.overreach_shadow = _RecordingShadow(service, order=order)
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
