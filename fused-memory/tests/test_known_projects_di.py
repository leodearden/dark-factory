"""Cross-consumer DI test: a single known_projects registry propagates to both
ReconciliationHarness and TicketJanitor (task 1164).

Pins the 'no silent divergence' acceptance criterion: when the same dict is
passed to both constructors, both observers see the same registry — independent
of the env var and of each other's mutations after construction.
"""
from __future__ import annotations

import pytest
import pytest_asyncio

from fused_memory.middleware.ticket_janitor import TicketJanitor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.journal import ReconciliationJournal


@pytest_asyncio.fixture
async def journal(tmp_path):
    j = ReconciliationJournal(tmp_path / 'di_test_journal')
    await j.initialize()
    yield j
    await j.close()


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(
        db_path=tmp_path / 'di_test_eb.db',
        buffer_size_threshold=2,
        max_staleness_seconds=3600,
    )
    await buf.initialize()
    yield buf
    await buf.close()


@pytest_asyncio.fixture
async def store(tmp_path):
    s = TicketStore(tmp_path / 'di_test_tickets.db')
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
def mock_memory_service():
    from unittest.mock import AsyncMock
    svc = AsyncMock()
    svc.search = AsyncMock(return_value=[])
    svc.get_episodes = AsyncMock(return_value=[])
    svc.get_status = AsyncMock(
        return_value={'graphiti': {'connected': True}, 'mem0': {'connected': True}, 'projects': {}}
    )
    svc.get_entity = AsyncMock(return_value={'nodes': [], 'edges': []})
    svc.mem0 = AsyncMock()
    svc.mem0.get_all = AsyncMock(return_value={'results': []})
    return svc


@pytest.mark.asyncio
async def test_single_registry_propagates_to_harness_and_janitor(
    journal, event_buffer, store, mock_memory_service
):
    """One dict injected into both constructors → both consumers see the same registry.

    Verifies:
    (a) harness._known_projects == registry
    (b) janitor._known_projects == registry
    (c) harness._known_project_scope_for('pid_a').project_root returns the right path
    (d) janitor._known_projects.get('pid_a') returns the right path
    (e) Defensive-copy contract: mutating registry after construction does NOT
        affect either harness or janitor.
    """
    from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
    from fused_memory.reconciliation.harness import ReconciliationHarness

    registry = {'pid_a': '/path/a', 'pid_b': '/path/b'}

    config = FusedMemoryConfig(
        reconciliation=ReconciliationConfig(
            enabled=True,
            explore_codebase_root='/tmp/test',
            agent_llm_provider='anthropic',
            agent_llm_model='claude-sonnet-4-20250514',
        )
    )

    harness = ReconciliationHarness(
        memory_service=mock_memory_service,
        taskmaster=None,
        journal=journal,
        event_buffer=event_buffer,
        config=config,
        known_projects=registry,
    )

    janitor = TicketJanitor(
        store,
        primary_project_root='',
        known_projects=registry,
    )

    # (a) harness stores the injected content
    assert harness._known_projects == {'pid_a': '/path/a', 'pid_b': '/path/b'}

    # (b) janitor stores the injected content
    assert janitor._known_projects == {'pid_a': '/path/a', 'pid_b': '/path/b'}

    # (c) harness lookup method works
    assert harness._known_project_scope_for('pid_a').project_root == '/path/a'

    # (d) janitor direct dict access works
    assert janitor._known_projects.get('pid_a') == '/path/a'

    # (e) defensive-copy contract: mutating the original dict after construction
    #     must NOT affect the harness or janitor registries.
    registry['pid_a'] = '/mutated/path'
    registry['pid_c'] = '/new/path'
    assert harness._known_projects == {'pid_a': '/path/a', 'pid_b': '/path/b'}, (
        'harness must hold an independent copy — external mutation must not propagate'
    )
    assert janitor._known_projects == {'pid_a': '/path/a', 'pid_b': '/path/b'}, (
        'janitor must hold an independent copy — external mutation must not propagate'
    )
