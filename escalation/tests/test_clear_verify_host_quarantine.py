"""The ``clear_verify_host_quarantine`` MCP tool: the operator clear of a benched verify host.

Task 6565, design D6.  The tool is a thin wrapper over
``Harness.clear_verify_host_quarantine``, which delegates to the merge worker
(its behaviour is pinned in orchestrator/tests/test_verify_host_skew_bench.py).
Exercised through the FastMCP tool registry, as test_release_workflow.py does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from escalation.queue import EscalationQueue
from escalation.server import create_server


class _FakeHarness:
    def __init__(self, outcome: dict[str, Any]) -> None:
        self.outcome = outcome
        self.cleared: list[str] = []

    async def clear_verify_host_quarantine(self, host: str) -> dict[str, Any]:
        self.cleared.append(host)
        return self.outcome


async def _call(server: Any, **kwargs: Any) -> dict[str, Any]:
    tool = await server.get_tool('clear_verify_host_quarantine')
    return await tool.fn(**kwargs)


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'esc')


@pytest.mark.asyncio
async def test_the_tool_hands_the_host_to_the_orchestrator_and_returns_its_outcome(queue):
    outcome = {'host': 'laptop', 'cleared': True, 'quarantine_class': 'skew', 'was_sticky': True}
    harness = _FakeHarness(outcome)

    result = await _call(create_server(queue, harness=harness), host='laptop')

    assert result == outcome
    assert harness.cleared == ['laptop']


@pytest.mark.asyncio
async def test_without_an_orchestrator_the_tool_clears_nothing_and_says_why(queue):
    result = await _call(create_server(queue, harness=None), host='laptop')

    assert result['cleared'] is False
    assert result['host'] == 'laptop'
    assert 'standalone' in result['reason']
