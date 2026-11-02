"""client-feed's escalation-queue reader against the REAL producer (task 7280).

The client-feed suite cannot import the escalation package, so its fixture records are
hand-copied and cannot catch producer drift. Here the escalation server's own tools file,
promote, park and resolve on a tmp root, and client-feed reads the queue they leave.
A renamed `level`, `status` or `members`, or a change in archival or cascade, turns this red.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from client_feed.contract import TaskStatus, WaitingOn
from client_feed.sources.escalations import read_open_escalations
from client_feed.sources.tasks import waiting_on
from escalation.queue import EscalationQueue
from escalation.server import create_server
from fastmcp import FastMCP
from fastmcp.tools import FunctionTool
from shared.project_layout import escalation_queue_dir

TASK_ID = 9


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def server(root: Path) -> FastMCP:
    return create_server(EscalationQueue(escalation_queue_dir(root)))


async def _tool(server: FastMCP, name: str) -> FunctionTool:
    tool = await server.get_tool(name)
    assert isinstance(tool, FunctionTool)
    return tool


async def _filed(server: FastMCP, name: str, **kwargs: Any) -> dict[str, Any]:
    """An async filing tool: escalate_blocker or promote_to_l2."""
    return await (await _tool(server, name)).fn(**kwargs)


async def _resolved(server: FastMCP, **kwargs: Any) -> dict[str, Any]:
    """resolve_issue, a sync tool."""
    return (await _tool(server, 'resolve_issue')).fn(**kwargs)


async def _l1(server: FastMCP) -> str:
    filed = await _filed(
        server, 'escalate_blocker',
        task_id=str(TASK_ID), agent_role='steward', category='design_concern',
        summary='SECRET', level=1,
    )
    return filed['id']


def _blocked_waits_on(root: Path) -> WaitingOn | None:
    return waiting_on(TaskStatus.BLOCKED, read_open_escalations(root).levels(TASK_ID))


async def test_a_promoted_then_resolved_l2_waits_on_leo_then_on_the_factory(
    root: Path, server: FastMCP
) -> None:
    l1_id = await _l1(server)

    assert _blocked_waits_on(root) is WaitingOn.FACTORY

    promoted = await _filed(
        server, 'promote_to_l2',
        task_id=str(TASK_ID), agent_role='escalation-watcher-auto', member_ids=[l1_id],
        root_cause='rc', evidence='e', options=['A: x'], summary='s',
    )

    assert _blocked_waits_on(root) is WaitingOn.LEO

    resolved = await _resolved(
        server, escalation_id=promoted['id'], resolution='decided', action='close_only'
    )

    assert resolved['status'] in {'resolved', 'dismissed'}
    assert _blocked_waits_on(root) is WaitingOn.FACTORY
    assert read_open_escalations(root).levels(TASK_ID) == ()


async def test_a_parked_l1_waits_on_leo(root: Path, server: FastMCP) -> None:
    l1_id = await _l1(server)

    parked = await _resolved(server, escalation_id=l1_id, resolution='parked', action='park')

    assert (parked['status'], parked['level']) == ('pending', 2)
    assert _blocked_waits_on(root) is WaitingOn.LEO
