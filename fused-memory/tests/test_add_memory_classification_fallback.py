"""A WriteClassifier fallback is visible on add_memory and feeds the storm alarm (task 6624).

Drives a REAL MemoryService with mocked backends and a REAL WriteClassifier
whose LLM client is a stub, so the classify -> add_memory -> response chain is
the production one.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.middleware import _folded_escalation
from fused_memory.models.enums import ClassificationFallback, MemoryCategory
from fused_memory.routing.classifier import WriteClassifier
from fused_memory.services.classification_fallback_alarm import (
    _ANCHOR_TASK_ID,
    DEFAULT_THRESHOLD,
    JOURNAL_PARAM_KEY,
)
from fused_memory.services.memory_service import MemoryService

_NO_HEURISTIC_MATCH = 'Hello world'

_needs_escalation = pytest.mark.skipif(
    not _folded_escalation.HAS_ESCALATION,
    reason='escalation package unavailable (minimal env); filing is a logged no-op there',
)


def _raising_client() -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError('connection refused'))
    return client


def _answering_client(content: str) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def _journal() -> MagicMock:
    journal = MagicMock()
    journal.log_write_op = AsyncMock()
    journal.log_backend_op = AsyncMock()
    journal.log_mem0_intent = AsyncMock()
    journal.resolve_mem0_intent = AsyncMock()
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
    return _raising_client()


@pytest.fixture
def service(llm_config, journal, failing_client) -> MemoryService:
    return _service(llm_config, journal, failing_client)


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
            _answering_client(
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
        [esc] = [e for e in _pending(tmp_path) if e.task_id == _ANCHOR_TASK_ID]
        assert 'p1' in esc.summary

    @pytest.mark.asyncio
    async def test_an_extracted_fact_burst_files_one_escalation(
        self, service, journal, tmp_path,
    ):
        service.set_known_projects({'p1': str(tmp_path)})

        for _ in range(DEFAULT_THRESHOLD):
            await service._execute_mem0_classify_and_add(
                {'fact_text': _NO_HEURISTIC_MATCH, 'project_id': 'p1'},
            )

        assert len([e for e in _pending(tmp_path) if e.task_id == _ANCHOR_TASK_ID]) == 1
        params = _add_memory_params(journal)
        assert len(params) == DEFAULT_THRESHOLD
        assert all(p[JOURNAL_PARAM_KEY] == 'llm_error' for p in params)
