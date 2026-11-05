"""client-feed's escalation-queue reader against the REAL producer (task 7280).

The client-feed suite cannot import the escalation package, so its fixture records are
hand-copied and cannot catch producer drift. Here the escalation server's own tools file,
promote, park and resolve on a tmp root, and client-feed reads the queue they leave. Checked:
`level` in every test; that every record the producer leaves parses, in every read, so a
status or type the reader refuses turns this red instead of reading as a close; `members`
through an L1 promoted under another task's L2; `status` through a resolution whose archival
fails and leaves the closed records at the root; and archival and the L2 cascade.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from client_feed.contract import TaskStatus, WaitingOn
from client_feed.sources.escalations import OpenEscalations, read_open_escalations
from client_feed.sources.tasks import waiting_on
from escalation.queue import EscalationQueue
from escalation.server import create_server
from fastmcp import FastMCP
from fastmcp.tools import FunctionTool
from shared.project_layout import escalation_queue_dir

TASK_ID = 9
MEMBER_TASK_ID = 10


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


async def _l1(server: FastMCP, task_id: int) -> str:
    filed = await _filed(
        server, 'escalate_blocker',
        task_id=str(task_id), agent_role='steward', category='design_concern',
        summary='SECRET', level=1,
    )
    return filed['id']


async def _promoted(server: FastMCP, l1_id: str) -> str:
    promoted = await _filed(
        server, 'promote_to_l2',
        task_id=str(TASK_ID), agent_role='escalation-watcher-auto', member_ids=[l1_id],
        root_cause='rc', evidence='e', options=['A: x'], summary='s',
    )
    return promoted['id']


async def _cluster(server: FastMCP) -> tuple[str, str]:
    """An L1 on MEMBER_TASK_ID promoted under an L2 on TASK_ID: (l1_id, l2_id)."""
    l1_id = await _l1(server, MEMBER_TASK_ID)
    return l1_id, await _promoted(server, l1_id)


async def _closed(server: FastMCP, l2_id: str) -> None:
    resolved = await _resolved(
        server, escalation_id=l2_id, resolution='decided', action='close_only'
    )
    assert resolved['status'] in {'resolved', 'dismissed'}


def _open_escalations(root: Path) -> OpenEscalations:
    escalations = read_open_escalations(root)
    assert escalations.skipped == ()
    return escalations


def _blocked_waits_on(root: Path, task_id: int) -> WaitingOn | None:
    return waiting_on(TaskStatus.BLOCKED, _open_escalations(root).levels(task_id))


async def test_a_promoted_then_resolved_l2_waits_on_leo_then_on_the_factory(
    root: Path, server: FastMCP
) -> None:
    l1_id = await _l1(server, TASK_ID)

    assert _blocked_waits_on(root, TASK_ID) is WaitingOn.FACTORY

    l2_id = await _promoted(server, l1_id)

    assert _blocked_waits_on(root, TASK_ID) is WaitingOn.LEO

    await _closed(server, l2_id)

    assert _blocked_waits_on(root, TASK_ID) is WaitingOn.FACTORY
    assert _open_escalations(root).levels(TASK_ID) == ()


async def test_a_promoted_l2_makes_its_member_task_wait_on_leo_until_the_cascade(
    root: Path, server: FastMCP
) -> None:
    l1_id = await _l1(server, MEMBER_TASK_ID)

    assert _blocked_waits_on(root, MEMBER_TASK_ID) is WaitingOn.FACTORY

    l2_id = await _promoted(server, l1_id)

    assert _blocked_waits_on(root, MEMBER_TASK_ID) is WaitingOn.LEO
    assert _blocked_waits_on(root, TASK_ID) is WaitingOn.LEO

    await _closed(server, l2_id)

    assert _blocked_waits_on(root, MEMBER_TASK_ID) is WaitingOn.FACTORY
    assert _blocked_waits_on(root, TASK_ID) is WaitingOn.FACTORY
    assert _open_escalations(root).levels(MEMBER_TASK_ID) == ()


async def test_a_resolution_left_at_the_root_by_a_failed_archival_is_not_open(
    root: Path, server: FastMCP
) -> None:
    queue = escalation_queue_dir(root)
    l1_id, l2_id = await _cluster(server)
    (queue / 'archive').write_text('a file where the dated archive dir would go')

    await _closed(server, l2_id)

    assert {path.stem for path in queue.glob('esc-*.json')} == {l1_id, l2_id}
    assert _blocked_waits_on(root, TASK_ID) is WaitingOn.FACTORY
    assert _blocked_waits_on(root, MEMBER_TASK_ID) is WaitingOn.FACTORY
    assert _open_escalations(root).levels(TASK_ID) == ()
    assert _open_escalations(root).levels(MEMBER_TASK_ID) == ()


async def test_a_parked_l1_waits_on_leo(root: Path, server: FastMCP) -> None:
    l1_id = await _l1(server, TASK_ID)

    parked = await _resolved(server, escalation_id=l1_id, resolution='parked', action='park')

    assert (parked['status'], parked['level']) == ('pending', 2)
    assert _blocked_waits_on(root, TASK_ID) is WaitingOn.LEO
