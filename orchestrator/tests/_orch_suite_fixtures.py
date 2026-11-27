"""Suite-wide pytest fixtures whose definitions need first-party imports.

``conftest.py::pytest_configure`` registers this module as a plugin, so every
test in the suite can request these fixtures by name and no module binds them.
They are not in ``conftest.py`` itself because a new first-party import there
would follow its ``sys.path`` setup and need an ``E402`` suppression. The rule
for where a shared fixture lives is in ``CONTRIBUTING.md`` §2.

A plugin's fixtures are session-global, not scoped to ``orchestrator/tests``
as conftest fixtures are: a session that also collects another workspace
member's tests offers them there too. A name hosted here must therefore be
unambiguous across the whole workspace.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
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

    Every opened store is closed when the test finishes, even if closing
    another one raises, preventing aiosqlite connection and background-thread
    leaks.
    """
    async with AsyncExitStack() as opened_stores:

        async def _open(filename: str = 'cost.db') -> CostStore:
            store = CostStore(tmp_path / filename)
            await store.open()
            opened_stores.push_async_callback(store.close)
            return store

        yield _open
