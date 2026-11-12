"""Orchestrator completion records are never write-triage children (task 6956).

Each task's completion record restates the same template as every other
task's, so triage would file task B's record as a child of task A's. Two routes
keep it standalone: the orchestrator declares ``kind='task_completion'``, which
``write_triage.declares_attach_keys`` already force-stores, and a server-side
backstop force-stores a legacy write that carries only the completion source.

Harness: ``tests/server/write_triage_harness.py`` and the
``server._tool_manager.call_tool('add_memory', ...)`` idiom of
``tests/server/test_write_triage_supersession.py``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.memory_metadata import ORCHESTRATOR_COMPLETION_SOURCE
from fused_memory.server import write_triage
from fused_memory.server.write_triage import (
    OUTCOME_STORED,
    TriageFailOpenCounter,
    triage_write,
)
from server.write_triage_harness import configure_triage_config, search_candidate

_HIGH_BAND = 0.99


class TestIsOrchestratorCompletionWrite:
    @pytest.mark.parametrize(
        'metadata',
        [
            {'task_id': '4102', 'source': ORCHESTRATOR_COMPLETION_SOURCE},
            {'task_id': '4102', 'source': ORCHESTRATOR_COMPLETION_SOURCE, 'kind': 'task_completion'},
        ],
        ids=['unkinded', 'kinded'],
    )
    def test_a_completion_write_counts(self, metadata) -> None:
        assert write_triage.is_orchestrator_completion_write(metadata) is True

    @pytest.mark.parametrize(
        'metadata',
        [None, {}, ORCHESTRATOR_COMPLETION_SOURCE, {'task_id': '4102'},
         {'task_id': '4102', 'source': 'agent_note'}],
        ids=['none', 'empty', 'not-a-dict', 'task-id-only', 'other-source'],
    )
    def test_anything_else_does_not_count(self, metadata) -> None:
        assert write_triage.is_orchestrator_completion_write(metadata) is False


class TestTriageWriteForceStoresACompletionRecord:
    @pytest.mark.asyncio
    async def test_a_completion_record_stores_before_retrieval(self) -> None:
        mock_service = AsyncMock()
        configure_triage_config(mock_service)
        mock_service.search.return_value = [search_candidate('completion-4101', _HIGH_BAND)]
        counter = TriageFailOpenCounter(time_provider=lambda: 1000.0)

        decision = await triage_write(
            mock_service, content='c', project_id='p', counter=counter,
            caller_is_completion_record=True,
        )

        assert decision.outcome == OUTCOME_STORED
        assert decision.canonical_id is None
        mock_service.search.assert_not_awaited()
        assert counter.live_count() == 0, 'a force-store is not a fail-open'
