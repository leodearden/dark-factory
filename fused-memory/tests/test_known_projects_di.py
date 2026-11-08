"""Cross-consumer DI test: a single known_projects registry propagates to both
ReconciliationHarness and TicketJanitor (task 1164).

Pins the 'no silent divergence' acceptance criterion: when the same dict is
passed to both constructors, both observers see the same registry — independent
of the env var and of each other's mutations after construction.
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from reconciliation.harness_fixtures import (
    make_harness,
    make_memory_service,
    open_event_buffer,
    open_journal,
)

from fused_memory.middleware.ticket_janitor import TicketJanitor
from fused_memory.middleware.ticket_store import TicketStore


@pytest_asyncio.fixture
async def journal(tmp_path):
    async with open_journal(tmp_path / 'di_test_journal') as j:
        yield j


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    async with open_event_buffer(tmp_path / 'di_test_eb.db') as b:
        yield b


@pytest_asyncio.fixture
async def store(tmp_path):
    s = TicketStore(tmp_path / 'di_test_tickets.db')
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
def mock_memory_service():
    return make_memory_service()


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
    registry = {'pid_a': '/path/a', 'pid_b': '/path/b'}

    harness = make_harness(
        journal=journal,
        event_buffer=event_buffer,
        memory_service=mock_memory_service,
        taskmaster=None,
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
