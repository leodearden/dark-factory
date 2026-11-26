"""The served task stack rig's own contract: a served stack answers, and a startup that fails raises instead of hanging."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from _served_task_stack import ServedTaskStackDidNotStart, served_task_stack, serving
from shared.mcp_session import SyncMcpSession
from starlette.applications import Starlette


class LifespanRefused(RuntimeError):
    """The app's lifespan refusing to start, as a store that fails to open would."""


@contextlib.asynccontextmanager
async def _refusing_lifespan(app: Starlette) -> AsyncIterator[None]:
    raise LifespanRefused(app)
    yield


@pytest.fixture
def loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def test_a_served_stack_answers_a_tool_call_on_its_own_store(tmp_path: Path) -> None:
    with served_task_stack(tmp_path) as served, SyncMcpSession(served.url, client_name='served-task-stack-test') as session:
        reply = session.call_tool('get_tasks', {'project_root': str(served.root)})

    assert reply['tasks'] == []
    assert served.residue_spool.status()['spooled_files'] == 0


def test_an_app_whose_lifespan_refuses_to_start_raises_did_not_start(loop: asyncio.AbstractEventLoop) -> None:
    entered = False

    with pytest.raises(ServedTaskStackDidNotStart), serving(loop, Starlette(lifespan=_refusing_lifespan)):
        entered = True

    assert not entered
