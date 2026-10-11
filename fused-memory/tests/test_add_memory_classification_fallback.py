"""A WriteClassifier fallback is visible on add_memory and feeds the storm alarm (task 6624).

Drives a REAL MemoryService with mocked backends and a REAL WriteClassifier
whose LLM client is a stub, so the classify -> add_memory -> response chain is
the production one.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from _fm_helpers import poll_until
from _openai_stubs import completion_client, raising_client

from fused_memory.middleware import _folded_escalation
from fused_memory.models.enums import ClassificationFallback, MemoryCategory
from fused_memory.routing.classifier import WriteClassifier
from fused_memory.services.classification_fallback_alarm import (
    ANCHOR_TASK_ID,
    DEFAULT_THRESHOLD,
    JOURNAL_PARAM_KEY,
)
from fused_memory.services.memory_service import MemoryService

_NO_HEURISTIC_MATCH = 'Hello world'

_needs_escalation = pytest.mark.skipif(
    not _folded_escalation.HAS_ESCALATION,
    reason='escalation package unavailable (minimal env); filing is a logged no-op there',
)


def _journal() -> MagicMock:
    journal = MagicMock()
    journal.log_write_op = AsyncMock()
    journal.log_backend_op = AsyncMock()
    journal.log_mem0_intent = AsyncMock()
    journal.resolve_mem0_intent = AsyncMock()
    journal.record_terminal_outcome = AsyncMock()
    journal.close = AsyncMock()
    return journal


def _service(config, journal: MagicMock, openai_client: MagicMock | None) -> MemoryService:
    svc = MemoryService(config)
    svc.mem0 = MagicMock()
    svc.mem0.add = AsyncMock(return_value={'results': [{'id': 'mem0-1'}]})
    svc.mem0.count_by_metadata = AsyncMock(return_value=0)
    svc.mem0.scroll_by_metadata = AsyncMock(return_value=[])
    svc.durable_queue = MagicMock()
    svc.durable_queue.enqueue = AsyncMock(return_value=1)
    svc.classifier = WriteClassifier(config, openai_client=openai_client)
    svc.set_write_journal(journal)  # type: ignore[arg-type]
    return svc


def _extracting(*facts: str) -> MagicMock:
    """A Graphiti add_episode result whose extraction yielded *facts*."""
    result = MagicMock()
    result.edges = [
        MagicMock(fact=fact, source_node_uuid='', target_node_uuid='') for fact in facts
    ]
    return result


def _fails_once_per_fact() -> AsyncMock:
    """A Mem0 add that fails each fact's first attempt, so the queue retries it."""
    attempted: set[str] = set()

    async def add(*, content: str, **_: object) -> dict:
        if content not in attempted:
            attempted.add(content)
            raise RuntimeError('mem0 unavailable')
        return {'results': [{'id': 'mem0-1'}]}

    return AsyncMock(side_effect=add)


async def _drain(service: MemoryService, *, completed: int) -> None:
    queue = service.durable_queue
    assert queue is not None

    async def _done() -> bool:
        stats = await queue.get_stats()
        return stats['counts'].get('completed') == completed

    await poll_until(_done, message=f'the queue never completed {completed} item(s)')


def _add_memory_params(journal: MagicMock) -> list[dict]:
    return [
        call.kwargs['params']
        for call in journal.log_write_op.await_args_list
        if call.kwargs.get('operation') == 'add_memory'
    ]


def _pending(tmp_path: Path) -> list:
    from escalation.queue import EscalationQueue  # noqa: PLC0415

    return EscalationQueue(tmp_path / 'data' / 'escalations').get_pending()


@pytest.fixture
def llm_config(mock_config):
    mock_config.routing.llm_fallback = True
    return mock_config


@pytest.fixture
def journal() -> MagicMock:
    return _journal()


@pytest.fixture
def failing_client() -> MagicMock:
    return raising_client(RuntimeError('connection refused'))


@pytest.fixture
def service(llm_config, journal, failing_client) -> MemoryService:
    return _service(llm_config, journal, failing_client)


@pytest_asyncio.fixture
async def queued_service(llm_config, journal, failing_client, tmp_path):
    """The service over a REAL durable queue, so add_episode's facts run as queued writes."""
    svc = _service(llm_config, journal, failing_client)
    svc.graphiti = MagicMock()
    svc.graphiti.initialize = AsyncMock()
    svc.graphiti.add_episode = AsyncMock(return_value=None)
    svc.graphiti.close = AsyncMock()
    svc.graphiti._require_client = MagicMock()
    svc.mem0.close = AsyncMock()
    svc.set_known_projects({'p1': str(tmp_path)})
    await svc.initialize()
    yield svc
    await svc.close()


class TestTheResponseReportsTheFallback:
    @pytest.mark.asyncio
    async def test_an_llm_failure_is_reported_on_the_response_and_journal(
        self, service, journal,
    ):
        response = await service.add_memory(
            _NO_HEURISTIC_MATCH, category=None, project_id='p1',
        )

        assert response.category == MemoryCategory.observations_and_summaries
        assert response.classification_fallback is ClassificationFallback.llm_error
        assert response.model_dump()['classification_fallback'] == 'llm_error'
        assert '[classification_fallback: llm_error]' in response.message
        [params] = _add_memory_params(journal)
        assert params[JOURNAL_PARAM_KEY] == 'llm_error'
        assert params['content'] == _NO_HEURISTIC_MATCH
        assert params['category'] == MemoryCategory.observations_and_summaries.value

    @pytest.mark.asyncio
    async def test_an_explicit_category_is_never_a_fallback(
        self, service, journal, failing_client,
    ):
        response = await service.add_memory(
            _NO_HEURISTIC_MATCH, category='decisions_and_rationale', project_id='p1',
        )

        assert response.classification_fallback is None
        assert 'classification_fallback' not in response.message
        [params] = _add_memory_params(journal)
        assert JOURNAL_PARAM_KEY not in params
        failing_client.chat.completions.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_real_classification_is_not_a_fallback(self, llm_config, journal):
        service = _service(
            llm_config,
            journal,
            completion_client(
                '{"primary": "decisions_and_rationale", "secondary": null, '
                '"confidence": 0.9, "reasoning": "a choice"}'
            ),
        )

        response = await service.add_memory(
            _NO_HEURISTIC_MATCH, category=None, project_id='p1',
        )

        assert response.category == MemoryCategory.decisions_and_rationale
        assert response.classification_fallback is None
        [params] = _add_memory_params(journal)
        assert JOURNAL_PARAM_KEY not in params


class TestHeuristicOnlyMode:
    @pytest.mark.asyncio
    async def test_no_confident_match_is_reported(self, mock_config, journal):
        service = _service(mock_config, journal, openai_client=None)

        response = await service.add_memory(
            _NO_HEURISTIC_MATCH, category=None, project_id='p1',
        )

        assert response.memory_ids == ['mem0-1']
        assert response.classification_fallback is ClassificationFallback.no_confident_match
        [params] = _add_memory_params(journal)
        assert params[JOURNAL_PARAM_KEY] == 'no_confident_match'

    @_needs_escalation
    @pytest.mark.asyncio
    async def test_no_confident_match_never_escalates(self, mock_config, journal, tmp_path):
        service = _service(mock_config, journal, openai_client=None)
        service.set_known_projects({'p1': str(tmp_path)})

        for _ in range(DEFAULT_THRESHOLD + 1):
            await service.add_memory(_NO_HEURISTIC_MATCH, category=None, project_id='p1')

        assert _pending(tmp_path) == []


@_needs_escalation
class TestTheStormAlarmIsWired:
    @pytest.mark.asyncio
    async def test_an_add_memory_burst_files_one_escalation_without_blocking(
        self, service, tmp_path,
    ):
        service.set_known_projects({'p1': str(tmp_path)})

        responses = [
            await service.add_memory(_NO_HEURISTIC_MATCH, category=None, project_id='p1')
            for _ in range(DEFAULT_THRESHOLD)
        ]

        assert [r.memory_ids for r in responses] == [['mem0-1']] * DEFAULT_THRESHOLD
        [esc] = [e for e in _pending(tmp_path) if e.task_id == ANCHOR_TASK_ID]
        assert 'p1' in esc.summary


@_needs_escalation
class TestTheEpisodePathFeedsTheStormAlarm:
    @pytest.mark.asyncio
    async def test_an_extracted_fact_burst_files_one_escalation(
        self, queued_service, journal, tmp_path,
    ):
        facts = [f'{_NO_HEURISTIC_MATCH} {n}' for n in range(DEFAULT_THRESHOLD)]
        queued_service.graphiti.add_episode.return_value = _extracting(*facts)

        await queued_service.add_episode('an episode', project_id='p1')
        await _drain(queued_service, completed=1 + len(facts))

        assert len([e for e in _pending(tmp_path) if e.task_id == ANCHOR_TASK_ID]) == 1
        params = _add_memory_params(journal)
        assert len(params) == DEFAULT_THRESHOLD
        assert all(p[JOURNAL_PARAM_KEY] == 'llm_error' for p in params)

    @pytest.mark.asyncio
    async def test_a_retried_fact_is_counted_once(self, queued_service, tmp_path):
        facts = [f'{_NO_HEURISTIC_MATCH} {n}' for n in range(DEFAULT_THRESHOLD - 1)]
        queued_service.graphiti.add_episode.return_value = _extracting(*facts)
        queued_service.mem0.add = _fails_once_per_fact()

        await queued_service.add_episode('an episode', project_id='p1')
        await _drain(queued_service, completed=1 + len(facts))

        assert queued_service.mem0.add.await_count == 2 * len(facts)
        assert _pending(tmp_path) == []
