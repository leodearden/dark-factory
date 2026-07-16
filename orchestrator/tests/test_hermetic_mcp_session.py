"""Guard test for HermeticMcpSession — proves injected-stub dispatch is hermetic.

Task 2644: several harness tests build a real ``Scheduler`` with no injected
``mcp_session`` and no ``McpLifecycle.start()``. In that shape,
``Scheduler.dispatch_tool`` (scheduler.py:1787-1794) falls through to the LIVE
``mcp_call(...)`` HTTP branch against ``http://localhost:8002`` — which, under
parallel test load, can saturate past the per-test timeout and ``os._exit()``
the xdist worker (timeout_method='thread').

Injecting a ``HermeticMcpSession`` makes ``dispatch_tool`` take the
``self._mcp_session.call_tool`` branch instead. This test is the guard that
proves that routing actually happens: it asserts both on the returned values
(seeded task list / empty statuses) AND — via the ``forbid_live_mcp`` fixture
— that zero live ``McpSession`` round-trips occurred.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from _orch_helpers import HermeticMcpSession

from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import Scheduler


class TestHermeticMcpSessionRoutesDispatchTool:
    @pytest.mark.asyncio
    async def test_get_tasks_and_get_statuses_route_through_injected_stub(
        self, tmp_path: Path, forbid_live_mcp,
    ) -> None:
        """get_tasks/get_statuses on a stub-injected Scheduler never touch the network.

        RED (pre-implementation): ``from _orch_helpers import HermeticMcpSession``
        raises ImportError — the helper does not exist yet.
        GREEN: dispatch_tool routes to the stub, get_tasks parses the seeded
        list, get_statuses parses to ({}, None), and forbid_live_mcp's
        teardown confirms zero live McpSession round-trips.
        """
        config = OrchestratorConfig(project_root=tmp_path)
        seeded_tasks = [{'id': '1', 'status': 'pending', 'dependencies': []}]
        scheduler = Scheduler(
            config, mcp_session=HermeticMcpSession(tasks=seeded_tasks),
        )
        scheduler.finish_startup()

        tasks = await scheduler.get_tasks(statuses=None)
        assert tasks == seeded_tasks, (
            f'Expected get_tasks to parse the seeded list via the injected '
            f'stub; got {tasks!r}'
        )

        statuses, err = await scheduler.get_statuses()
        assert statuses == {}, f'Expected empty statuses dict; got {statuses!r}'
        assert err is None, f'Expected no error; got {err!r}'
