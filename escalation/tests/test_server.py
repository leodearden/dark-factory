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

from escalation.dedupe import DedupeConfig
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
    # get_pending_escalations is a sync def, so tool.fn(...) returns directly
    return tool.fn(**kwargs)


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


# ---------------------------------------------------------------------------
# TestL2DedupeBypass: born-at-L2 escalations bypass deduplication
# ---------------------------------------------------------------------------


class TestL2DedupeBypass:
    """Born-at-L2 escalations bypass deduplication and get their own on-disk record."""

    @pytest.mark.asyncio
    async def test_l2_infra_issue_bypasses_dedupe_gets_own_record(self, tmp_path: Path):
        """escalate_info(severity='critical', category='infra_issue') is NOT folded into parent.

        Even with a matching pending infra_issue parent (same summary prefix),
        a critical-severity child must produce its own 'queued' record at level=2
        rather than returning 'dedup_skipped' and losing its L2 routing.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue, dedupe_config=DedupeConfig())

        # Pre-seed a pending infra_issue parent whose summary key will match
        parent = Escalation(
            id=queue.make_id('task-999'),
            task_id='task-999',
            agent_role='implementer',
            severity='info',
            category='infra_issue',
            summary='infra connection timeout',
        )
        queue.submit(parent)

        # Child call: same category, overlapping summary, but born-at-L2
        result = await _info(
            server,
            task_id='task-999',
            agent_role='implementer',
            category='infra_issue',
            summary='infra connection timeout on port 8002',
            severity='critical',
        )

        # Must be queued (not dedup_skipped) — dedupe bypassed for L2
        assert result['status'] == 'queued', (
            f"Expected 'queued' (L2 bypasses dedupe), got: {result}"
        )
        # The child's on-disk record must exist independently and carry level=2
        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.level == 2, f"Expected level==2 on own record, got: {esc.level}"

    @pytest.mark.asyncio
    async def test_l2_dedupe_bypass_does_not_modify_parent(self, tmp_path: Path):
        """When L2 bypasses dedupe, the pre-seeded parent's dedupe_count stays 0."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue, dedupe_config=DedupeConfig())

        parent = Escalation(
            id=queue.make_id('task-999'),
            task_id='task-999',
            agent_role='implementer',
            severity='info',
            category='infra_issue',
            summary='infra connection timeout',
        )
        queue.submit(parent)
        parent_id = parent.id

        await _info(
            server,
            task_id='task-999',
            agent_role='implementer',
            category='infra_issue',
            summary='infra connection timeout on port 8002',
            severity='critical',
        )

        updated_parent = queue.get(parent_id)
        assert updated_parent is not None
        assert updated_parent.dedupe_count == 0, (
            f"Expected dedupe_count==0 (L2 bypasses dedupe), got: {updated_parent.dedupe_count}"
        )
        assert len(updated_parent.dedupe_children) == 0, (
            f"Expected no dedupe_children (L2 bypasses dedupe), got: {updated_parent.dedupe_children}"
        )

    @pytest.mark.asyncio
    async def test_l2_blocker_terminal_task_resolved_record_has_level2(self, tmp_path: Path):
        """born-at-L2 severity + terminal task: auto-resolved on-disk record has level==2.

        The L2 gate stamps esc.level=2 before Gate 4 (terminal-task auto-resolve),
        so submit_resolved writes the record with level=2 even when the task is done.
        """
        async def _lookup(task_id: str) -> str:
            return 'done'

        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue, task_status_lookup=_lookup)

        result = await _blocker(server, severity='critical', **_COMMON_KWARGS)

        assert result['status'] == 'resolved', (
            f"Expected 'resolved' (terminal task auto-resolve), got: {result['status']}"
        )
        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.level == 2, (
            f"Expected level==2 on auto-resolved record, got: {esc.level}"
        )
        assert esc.status == 'resolved'

    @pytest.mark.asyncio
    async def test_l2_urgent_blocker_bypasses_dedupe(self, tmp_path: Path):
        """escalate_blocker(severity='urgent') with infra_issue also bypasses dedupe."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue, dedupe_config=DedupeConfig())

        # Pre-seed a matching pending parent
        parent = Escalation(
            id=queue.make_id('task-999'),
            task_id='task-999',
            agent_role='implementer',
            severity='blocking',
            category='infra_issue',
            summary='infra connection timeout',
        )
        queue.submit(parent)

        result = await _blocker(
            server,
            task_id='task-999',
            agent_role='implementer',
            category='infra_issue',
            summary='infra connection timeout on port 8002',
            severity='urgent',
        )

        assert result['status'] == 'queued', (
            f"Expected 'queued' (L2 urgent bypasses dedupe), got: {result}"
        )
        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.level == 2


# ---------------------------------------------------------------------------
# TestSeverityValidation: escalate_blocker/escalate_info validate severity
# ---------------------------------------------------------------------------


class TestSeverityValidation:
    """escalate_blocker/escalate_info reject unknown severity strings."""

    @pytest.mark.asyncio
    async def test_blocker_uppercase_critical_returns_error(self, tmp_path: Path):
        """escalate_blocker(severity='CRITICAL') returns error — gate is case-sensitive."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(server, severity='CRITICAL', **_COMMON_KWARGS)

        assert 'error' in result, f"Expected error for 'CRITICAL' (case-sensitive), got: {result}"

    @pytest.mark.asyncio
    async def test_info_uppercase_urgent_returns_error(self, tmp_path: Path):
        """escalate_info(severity='Urgent') returns error — gate is case-sensitive."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _info(server, severity='Urgent', **_COMMON_KWARGS)

        assert 'error' in result, f"Expected error for 'Urgent' (case-sensitive), got: {result}"

    @pytest.mark.asyncio
    async def test_blocker_typo_severity_returns_error_and_nothing_queued(self, tmp_path: Path):
        """escalate_blocker with typo severity returns error and queues nothing."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(server, severity='criticial', **_COMMON_KWARGS)

        assert 'error' in result, f"Expected error for typo severity 'criticial', got: {result}"
        assert len(queue.get_pending()) == 0, (
            "Expected nothing queued when severity is invalid"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('severity', ['CRITICAL', 'Urgent', 'criticial', 'INFO', 'BLOCKING'])
    async def test_case_sensitive_gate_rejects_variants(self, tmp_path: Path, severity: str):
        """The gate is case-sensitive; mixed-case/upper-case severities are rejected."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(server, severity=severity, **_COMMON_KWARGS)

        assert 'error' in result, (
            f"Expected error for severity={severity!r} (case-sensitive gate), got: {result}"
        )

    @pytest.mark.asyncio
    async def test_known_severities_all_accepted(self, tmp_path: Path):
        """All KNOWN_SEVERITIES values are accepted without error by escalate_blocker."""
        from escalation.models import KNOWN_SEVERITIES
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        for sev in KNOWN_SEVERITIES:
            result = await _blocker(
                server,
                task_id=f'task-{sev}',
                agent_role='implementer',
                category='scope_violation',
                summary=f'severity={sev} acceptance test',
                severity=sev,
            )
            assert 'error' not in result, (
                f"Known severity {sev!r} should be accepted, got error: {result}"
            )
