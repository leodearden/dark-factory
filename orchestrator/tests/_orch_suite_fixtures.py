"""Suite-wide pytest fixtures whose definitions need first-party imports.

``conftest.py::pytest_configure`` registers this module as a plugin, so every
test in the suite can request these fixtures by name and no module binds them.
They are not in ``conftest.py`` itself because a new first-party import there
would follow its ``sys.path`` setup and need an ``E402`` suppression. The rule
for where a shared fixture lives is in ``CONTRIBUTING.md`` §2.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from shared.cost_store import CostStore

from orchestrator.artifacts import TaskArtifacts


@pytest.fixture()
def plan_artifacts(tmp_path):
    """TaskArtifacts over a temp worktree — mirrors ``test_plan_tools_server``."""
    a = TaskArtifacts(tmp_path)
    a.init('test-1', 'Test task', 'A test')
    return a


@pytest_asyncio.fixture
async def _cost_store_factory(tmp_path: Path):
    """Fixture factory that opens CostStore instances and closes them on teardown.

    Usage within a test::

        store = await _cost_store_factory()
        store2 = await _cost_store_factory('other.db')

    All opened stores are closed in a ``finally`` block when the test finishes,
    preventing aiosqlite connection and background-thread leaks.
    """
    stores: list[CostStore] = []

    async def _open(filename: str = 'cost.db') -> CostStore:
        store = CostStore(tmp_path / filename)
        await store.open()
        stores.append(store)
        return store

    yield _open

    for store in stores:
        await store.close()
