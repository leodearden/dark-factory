"""Tests for L2 severity-gated born-at-L2 path and get_pending_escalations level filter.

Uses the async FastMCP unit-test pattern from test_server_chokepoint.py:
    tool = await server.get_tool(name)
    result = await tool.fn(...)

and tmp_path isolation with EscalationQueue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from escalation.models import Escalation
from escalation.queue import EscalationQueue
from escalation.server import create_server

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMMON_KWARGS: dict[str, Any] = {
    'task_id': 'task-999',
    'agent_role': 'implementer',
    'category': 'scope_violation',
    'summary': 'born-at-L2 test',
}


async def _blocker(server, **kwargs: Any) -> dict[str, Any]:
    tool = await server.get_tool('escalate_blocker')
    return await tool.fn(**kwargs)


async def _info(server, **kwargs: Any) -> dict[str, Any]:
    tool = await server.get_tool('escalate_info')
    return await tool.fn(**kwargs)


async def _get_pending(server, **kwargs: Any) -> list[dict[str, Any]]:
    tool = await server.get_tool('get_pending_escalations')
    return await tool.fn(**kwargs)


# ---------------------------------------------------------------------------
# TestBornAtL2: severity-gated born-at-L2 path
# ---------------------------------------------------------------------------


class TestBornAtL2:
    """escalate_blocker/escalate_info with critical/urgent severity → on-disk level==2."""

    @pytest.mark.asyncio
    async def test_blocker_critical_severity_returns_queued(self, tmp_path: Path):
        """escalate_blocker(severity='critical') → result['status']=='queued'."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(server, severity='critical', **_COMMON_KWARGS)

        assert result['status'] == 'queued', f"Expected 'queued', got: {result}"

    @pytest.mark.asyncio
    async def test_blocker_critical_severity_on_disk_level2(self, tmp_path: Path):
        """escalate_blocker(severity='critical') → on-disk escalation has level==2."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(server, severity='critical', **_COMMON_KWARGS)

        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.level == 2, f"Expected level==2, got: {esc.level}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize('severity', ['critical', 'urgent'])
    async def test_blocker_l2_severities_yield_level2(self, tmp_path: Path, severity: str):
        """escalate_blocker with critical/urgent severity → on-disk level==2."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(server, severity=severity, **_COMMON_KWARGS)

        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.level == 2, f"severity={severity!r}: expected level==2, got: {esc.level}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize('severity', ['critical', 'urgent'])
    async def test_info_l2_severities_yield_level2(self, tmp_path: Path, severity: str):
        """escalate_info with critical/urgent severity → on-disk level==2."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _info(server, severity=severity, **_COMMON_KWARGS)

        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.level == 2, f"severity={severity!r}: expected level==2, got: {esc.level}"

    # Regression: default severity → level==0

    @pytest.mark.asyncio
    async def test_blocker_default_severity_is_level0(self, tmp_path: Path):
        """escalate_blocker() with default severity → on-disk level==0."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(server, **_COMMON_KWARGS)

        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.level == 0, f"Expected level==0, got: {esc.level}"

    @pytest.mark.asyncio
    async def test_info_default_severity_is_level0(self, tmp_path: Path):
        """escalate_info() with default severity → on-disk level==0."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _info(server, **_COMMON_KWARGS)

        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.level == 0, f"Expected level==0, got: {esc.level}"

    @pytest.mark.asyncio
    async def test_blocker_explicit_blocking_severity_is_level0(self, tmp_path: Path):
        """escalate_blocker(severity='blocking') → on-disk level==0 (gate is BORN_AT_L2_SEVERITIES-bound)."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(server, severity='blocking', **_COMMON_KWARGS)

        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.level == 0, f"Expected level==0, got: {esc.level}"

    @pytest.mark.asyncio
    async def test_info_explicit_info_severity_is_level0(self, tmp_path: Path):
        """escalate_info(severity='info') → on-disk level==0 (gate is BORN_AT_L2_SEVERITIES-bound)."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _info(server, severity='info', **_COMMON_KWARGS)

        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.level == 0, f"Expected level==0, got: {esc.level}"


# ---------------------------------------------------------------------------
# TestGetPendingLevelFilter: get_pending_escalations level filter
# ---------------------------------------------------------------------------


class TestGetPendingLevelFilter:
    """get_pending_escalations(level=N) filters by escalation level."""

    def _seed_esc(self, queue: EscalationQueue, task_id: str, level: int) -> Escalation:
        """Seed a pending escalation at the given level directly via queue.submit()."""
        esc = Escalation(
            id=queue.make_id(task_id),
            task_id=task_id,
            agent_role='implementer',
            severity='blocking',
            category='scope_violation',
            summary=f'level={level} test escalation',
            level=level,
        )
        queue.submit(esc)
        return esc

    @pytest.mark.asyncio
    async def test_filter_level2_returns_only_l2(self, tmp_path: Path):
        """get_pending_escalations(level=2) returns only the L2 escalation."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        self._seed_esc(queue, 'task-A', level=0)
        self._seed_esc(queue, 'task-A', level=1)
        l2 = self._seed_esc(queue, 'task-A', level=2)

        result = await _get_pending(server, level=2)

        assert len(result) == 1, f"Expected 1 result, got {len(result)}: {result}"
        assert result[0]['level'] == 2
        assert result[0]['id'] == l2.id

    @pytest.mark.asyncio
    async def test_filter_level1_returns_only_l1(self, tmp_path: Path):
        """get_pending_escalations(level=1) returns only the L1 escalation."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        self._seed_esc(queue, 'task-A', level=0)
        l1 = self._seed_esc(queue, 'task-A', level=1)
        self._seed_esc(queue, 'task-A', level=2)

        result = await _get_pending(server, level=1)

        assert len(result) == 1, f"Expected 1 result, got {len(result)}: {result}"
        assert result[0]['level'] == 1
        assert result[0]['id'] == l1.id

    @pytest.mark.asyncio
    async def test_filter_level0_returns_only_l0(self, tmp_path: Path):
        """get_pending_escalations(level=0) returns only the L0 escalation."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        l0 = self._seed_esc(queue, 'task-A', level=0)
        self._seed_esc(queue, 'task-A', level=1)
        self._seed_esc(queue, 'task-A', level=2)

        result = await _get_pending(server, level=0)

        assert len(result) == 1, f"Expected 1 result, got {len(result)}: {result}"
        assert result[0]['level'] == 0
        assert result[0]['id'] == l0.id

    @pytest.mark.asyncio
    async def test_no_level_filter_returns_all(self, tmp_path: Path):
        """get_pending_escalations() with no level filter returns all escalations."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        self._seed_esc(queue, 'task-A', level=0)
        self._seed_esc(queue, 'task-A', level=1)
        self._seed_esc(queue, 'task-A', level=2)

        result = await _get_pending(server)

        assert len(result) == 3, f"Expected 3 results (no filter), got {len(result)}: {result}"

    @pytest.mark.asyncio
    async def test_combined_task_id_and_level_filter(self, tmp_path: Path):
        """get_pending_escalations(task_id='task-A', level=2) returns only L2 for task-A."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        # L0 and L1 for task-A
        self._seed_esc(queue, 'task-A', level=0)
        self._seed_esc(queue, 'task-A', level=1)
        # L2 for task-A (the one we want)
        l2_a = self._seed_esc(queue, 'task-A', level=2)
        # L2 for a different task
        self._seed_esc(queue, 'task-B', level=2)

        result = await _get_pending(server, task_id='task-A', level=2)

        assert len(result) == 1, f"Expected 1 result, got {len(result)}: {result}"
        assert result[0]['id'] == l2_a.id
        assert result[0]['task_id'] == 'task-A'
        assert result[0]['level'] == 2
