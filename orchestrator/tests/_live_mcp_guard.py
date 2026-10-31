"""Fail a test that attempts a live fused-memory MCP round-trip.

``shared/src/shared/mcp_session.py::McpSession`` is the one network chokepoint
every alias funnels through: the ``mcp_call`` fallback of
``orchestrator/src/orchestrator/scheduler.py::Scheduler.dispatch_tool``,
``orchestrator/src/orchestrator/mcp_lifecycle.py::mcp_call`` and
``orchestrator/src/orchestrator/agents/memory_recall.py``. Each imports
``mcp_call`` as its own binding, so the guard patches ``initialize`` and
``call`` on the class rather than any one binding.

A refusal is ``pytest.fail``. Its ``Failed`` is a ``BaseException``, so the
``except Exception`` retry and swallow arms cannot absorb it; under an
``AssertionError`` the fm-retry loop of ``Scheduler.set_task_status`` slept
about 120s before the refusal surfaced. Every attempt is also recorded, so the
fixture's teardown still fails a test whose caller swallows ``BaseException``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn

import pytest
from shared.mcp_session import McpSession


def install_live_mcp_guard(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Patch McpSession to refuse every round-trip; return the attempt log."""
    attempts: list[tuple[str, str]] = []

    def refuse(session: McpSession, what: str) -> NoReturn:
        attempts.append((session.base_url, what))
        pytest.fail(
            f'forbid_live_mcp: live fused-memory MCP round-trip attempted: {what} '
            f'at {session.base_url!r} — inject a HermeticMcpSession into the '
            'Scheduler under test (see _orch_helpers.HermeticMcpSession)'
        )

    async def initialize(self: McpSession) -> None:
        refuse(self, 'initialize')

    async def call(
        self: McpSession,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout: float = 30,
    ) -> dict:
        refuse(self, _attempt_label(method, params))

    monkeypatch.setattr(McpSession, 'initialize', initialize)
    monkeypatch.setattr(McpSession, 'call', call)
    return attempts


def _attempt_label(method: str, params: Mapping[str, Any] | None) -> str:
    if method == 'tools/call' and params and 'name' in params:
        return f'{method} {params["name"]!r}'
    return method
