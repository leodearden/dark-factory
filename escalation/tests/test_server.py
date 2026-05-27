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


# ---------------------------------------------------------------------------
# Helpers for promote_to_l2 tests
# ---------------------------------------------------------------------------

async def _promote_to_l2(server, **kwargs: Any) -> dict[str, Any]:
    """Invoke the promote_to_l2 MCP tool directly."""
    tool = await server.get_tool('promote_to_l2')
    return await tool.fn(**kwargs)


async def _resolve_issue(server, **kwargs: Any) -> dict[str, Any]:
    """Invoke the resolve_issue MCP tool directly (sync tool)."""
    tool = await server.get_tool('resolve_issue')
    # resolve_issue is a sync def, so tool.fn(...) returns directly
    return tool.fn(**kwargs)


_L2_DEFAULTS: dict[str, Any] = {
    'task_id': 't-1',
    'agent_role': 'escalation-watcher-auto',
    'member_ids': [],  # override in each test
    'root_cause': 'Bad merge strategy',
    'evidence': 'Multiple L1s pointing to the same merge regression.',
    'options': ['A: rollback merge', 'B: fix forward', 'C: something else'],
    'summary': 'Merge regression cluster',
}


# ---------------------------------------------------------------------------
# TestPromoteToL2Create: create path
# ---------------------------------------------------------------------------


class TestPromoteToL2Create:
    """promote_to_l2 creates a new L2 record (create path)."""

    @pytest.mark.asyncio
    async def test_returns_created_with_id_and_members(self, tmp_path: Path):
        """(a) promote_to_l2 returns {id, status='created', members=[l1_id]}."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _promote_to_l2(
            server, **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-1']},
        )

        assert 'error' not in result, f'Unexpected error: {result}'
        assert result['status'] == 'created', f"Expected 'created', got: {result}"
        assert 'id' in result
        assert result['members'] == ['esc-l1-1']

    @pytest.mark.asyncio
    async def test_on_disk_record_has_correct_fields(self, tmp_path: Path):
        """(b) On-disk record has level=2, status='pending', members, root_cause, options, detail."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _promote_to_l2(
            server,
            task_id='t-1',
            agent_role='watcher-auto',
            member_ids=['esc-l1-1'],
            root_cause='Bad merge strategy',
            evidence='Evidence text here',
            options=['A: fix', 'B: rollback'],
            summary='Merge cluster',
        )

        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.level == 2, f'Expected level=2, got {esc.level}'
        assert esc.status == 'pending', f'Expected pending, got {esc.status!r}'
        assert esc.members == ['esc-l1-1']
        assert esc.root_cause == 'Bad merge strategy'
        assert esc.options == ['A: fix', 'B: rollback']
        assert esc.detail == 'Evidence text here', f'Expected evidence in detail, got {esc.detail!r}'

    @pytest.mark.asyncio
    async def test_single_member_l2_works(self, tmp_path: Path):
        """(c) 1-member L2 works exactly like multi-member (no special case)."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _promote_to_l2(
            server, **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-solo']},
        )

        assert result['status'] == 'created'
        assert result['members'] == ['esc-l1-solo']
        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.level == 2
        assert esc.members == ['esc-l1-solo']

    @pytest.mark.asyncio
    async def test_empty_member_ids_returns_error(self, tmp_path: Path):
        """(d) member_ids=[] returns {'error': ...} and nothing is queued."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _promote_to_l2(
            server, **{**_L2_DEFAULTS, 'member_ids': []},
        )

        assert 'error' in result, f'Expected error for empty member_ids, got: {result}'
        assert len(queue.get_pending()) == 0, 'Nothing should be queued on error'

    @pytest.mark.asyncio
    async def test_empty_root_cause_returns_error(self, tmp_path: Path):
        """(e) root_cause='' returns {'error': ...} and nothing is queued."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _promote_to_l2(
            server,
            **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-1'], 'root_cause': ''},
        )

        assert 'error' in result, f'Expected error for empty root_cause, got: {result}'
        assert len(queue.get_pending()) == 0, 'Nothing should be queued on error'

    @pytest.mark.asyncio
    async def test_whitespace_only_root_cause_returns_error(self, tmp_path: Path):
        """(e cont.) root_cause='  ' (whitespace-only) returns error."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _promote_to_l2(
            server,
            **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-1'], 'root_cause': '   '},
        )

        assert 'error' in result, f'Expected error for whitespace root_cause, got: {result}'
        assert len(queue.get_pending()) == 0, 'Nothing should be queued on error'

    @pytest.mark.asyncio
    async def test_invalid_severity_returns_error(self, tmp_path: Path):
        """Suggestion-1: unknown severity returns {'error': ...} and nothing is queued.

        Unlike escalate_blocker/escalate_info, promote_to_l2 previously lacked
        this guard.  A misconfigured caller (e.g. severity='CRITICAL' due to
        case error, or severity='warn') would silently create an L2 with an
        arbitrary severity tag.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _promote_to_l2(
            server,
            **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-1'], 'severity': 'CRITICAL'},
        )

        assert 'error' in result, (
            f"Expected error for unknown severity 'CRITICAL' (case-sensitive gate), got: {result}"
        )
        assert len(queue.get_pending()) == 0, 'Nothing should be queued when severity is invalid'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('severity', ['CRITICAL', 'Urgent', 'warn', 'INFO', 'BLOCKING'])
    async def test_known_severity_variants_are_rejected(self, tmp_path: Path, severity: str):
        """Suggestion-1: severity gate is case-sensitive; mixed-case/unknown values rejected."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _promote_to_l2(
            server,
            **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-1'], 'severity': severity},
        )

        assert 'error' in result, (
            f"Expected error for severity={severity!r} (case-sensitive gate), got: {result}"
        )

    @pytest.mark.asyncio
    async def test_all_known_severities_accepted(self, tmp_path: Path):
        """Suggestion-1: all KNOWN_SEVERITIES values are accepted by promote_to_l2."""
        from escalation.models import KNOWN_SEVERITIES
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        for sev in KNOWN_SEVERITIES:
            result = await _promote_to_l2(
                server,
                task_id=f'task-{sev}',
                agent_role='watcher',
                member_ids=[f'esc-l1-{sev}'],
                root_cause=f'root cause for {sev}',
                evidence='e',
                options=['A'],
                summary='s',
                severity=sev,
            )
            assert 'error' not in result, (
                f"Known severity {sev!r} should be accepted, got error: {result}"
            )

    @pytest.mark.asyncio
    async def test_duplicate_member_ids_in_create_path_are_deduplicated(self, tmp_path: Path):
        """Suggestion-3: duplicate member_ids in the create path produce a single entry.

        Passing member_ids=['a', 'a', 'b'] must result in members=['a', 'b'] on
        the on-disk record — the create path must apply dict.fromkeys dedup.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _promote_to_l2(
            server,
            **{
                **_L2_DEFAULTS,
                'member_ids': ['esc-l1-1', 'esc-l1-1', 'esc-l1-2'],
                'root_cause': 'dedup create test',
            },
        )

        assert result['status'] == 'created'
        assert result['members'].count('esc-l1-1') == 1, (
            f"Duplicate 'esc-l1-1' must appear once, got members={result['members']}"
        )
        assert 'esc-l1-2' in result['members']
        assert len(result['members']) == 2, f"Expected 2 unique members, got {result['members']}"

        # Verify durability: on-disk record also has deduped members
        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.members.count('esc-l1-1') == 1
        assert len(esc.members) == 2


# ---------------------------------------------------------------------------
# TestPromoteToL2Dedup: dedup / update path
# ---------------------------------------------------------------------------


class TestPromoteToL2Dedup:
    """promote_to_l2 dedup: second call with same root_cause updates the existing L2."""

    @pytest.mark.asyncio
    async def test_second_call_same_root_cause_returns_same_id_and_updated_status(
        self, tmp_path: Path,
    ):
        """(a) Two calls with the same root_cause return the SAME id; second has status='updated'."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        first = await _promote_to_l2(
            server, **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-1']},
        )
        second = await _promote_to_l2(
            server, **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-2']},
        )

        assert first['status'] == 'created'
        assert second['status'] == 'updated', f"Expected 'updated', got: {second}"
        assert second['id'] == first['id'], (
            f"Expected same id on dedup; first={first['id']!r}, second={second['id']!r}"
        )

    @pytest.mark.asyncio
    async def test_second_call_appends_members_set_union(self, tmp_path: Path):
        """(b) Second call appends new members (set-union: existing preserved, new added once)."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        await _promote_to_l2(
            server, **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-1']},
        )
        # Pass both the existing member (should not duplicate) and a new one
        second = await _promote_to_l2(
            server, **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-1', 'esc-l1-2']},
        )

        assert second['status'] == 'updated'
        members = second['members']
        assert 'esc-l1-1' in members, f"Original member must be preserved: {members}"
        assert 'esc-l1-2' in members, f"New member must be added: {members}"
        assert members.count('esc-l1-1') == 1, f"No duplicate for existing member: {members}"

    @pytest.mark.asyncio
    async def test_only_one_l2_file_on_disk(self, tmp_path: Path):
        """(c) Only ONE L2 file exists after two calls with the same root_cause."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        await _promote_to_l2(server, **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-1']})
        await _promote_to_l2(server, **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-2']})

        pending_l2 = [e for e in queue.get_pending() if e.level == 2]
        assert len(pending_l2) == 1, (
            f'Expected exactly 1 pending L2, got {len(pending_l2)}: '
            f'{[e.id for e in pending_l2]}'
        )

    @pytest.mark.asyncio
    async def test_second_call_does_not_modify_root_cause_options_summary_detail(
        self, tmp_path: Path,
    ):
        """(d) Existing L2's root_cause, options, summary, detail are NOT modified by the second call."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        first = await _promote_to_l2(
            server,
            task_id='t-1',
            agent_role='watcher',
            member_ids=['esc-l1-1'],
            root_cause='Bad merge strategy',
            evidence='First evidence',
            options=['A: fix'],
            summary='First summary',
        )
        # Second call with different evidence, options, summary — should NOT overwrite
        await _promote_to_l2(
            server,
            task_id='t-2',
            agent_role='watcher',
            member_ids=['esc-l1-2'],
            root_cause='Bad merge strategy',
            evidence='Second evidence (should not overwrite)',
            options=['X: different option'],
            summary='Different summary',
        )

        esc = queue.get(first['id'])
        assert esc is not None
        assert esc.root_cause == 'Bad merge strategy'
        assert esc.options == ['A: fix'], f'Options must be preserved: {esc.options}'
        assert esc.summary == 'First summary', f'Summary must be preserved: {esc.summary}'
        assert esc.detail == 'First evidence', f'Detail must be preserved: {esc.detail}'

    @pytest.mark.asyncio
    async def test_different_root_cause_creates_new_l2(self, tmp_path: Path):
        """(e) Different root_cause produces a new L2 (status='created' again)."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        first = await _promote_to_l2(
            server,
            **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-1'], 'root_cause': 'root cause A'},
        )
        second = await _promote_to_l2(
            server,
            **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-2'], 'root_cause': 'root cause B'},
        )

        assert first['status'] == 'created'
        assert second['status'] == 'created', f"Expected 'created' for different root_cause: {second}"
        assert first['id'] != second['id'], 'Different root causes must produce different L2s'
        pending_l2 = [e for e in queue.get_pending() if e.level == 2]
        assert len(pending_l2) == 2, f'Expected 2 pending L2s: {[e.id for e in pending_l2]}'

    @pytest.mark.asyncio
    async def test_resolved_l2_does_not_block_new_l2_same_root_cause(
        self, tmp_path: Path,
    ):
        """(f) A resolved L2 with the same root_cause does NOT block creating a fresh L2."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        # Create and resolve the first L2
        first = await _promote_to_l2(
            server, **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-1']},
        )
        queue.resolve(first['id'], 'Previous resolution')

        # Second call: same root_cause, but first is resolved → new L2 (status='created')
        second = await _promote_to_l2(
            server, **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-2']},
        )

        assert second['status'] == 'created', (
            f"Expected 'created' (resolved L2 should not block new one): {second}"
        )
        assert second['id'] != first['id'], 'New L2 must have a different id'

    @pytest.mark.asyncio
    async def test_race_archived_l2_falls_through_to_create(
        self, tmp_path: Path, monkeypatch,
    ):
        """Suggestion-2: when add_members_to_l2 returns None (race — L2 archived between
        find and update), the tool falls through to create a new L2 rather than returning
        a misleading {'status': 'updated', 'members': []}.

        The race: find_pending_l2_by_root_cause returns a stale id (L2 was pending
        at scan time), but by the time add_members_to_l2 runs the L2 has been
        resolved/archived and the queue root file is gone — so add_members_to_l2
        returns None.  The correct response is to create a fresh L2 (status='created')
        rather than lying to the caller.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        # Simulate the race via monkeypatching:
        # find_pending_l2_by_root_cause returns a stale id,
        # but add_members_to_l2 returns None (archived between calls).
        monkeypatch.setattr(queue, 'find_pending_l2_by_root_cause', lambda rc: 'esc-stale-id')
        monkeypatch.setattr(queue, 'add_members_to_l2', lambda esc_id, ids: None)

        result = await _promote_to_l2(
            server, **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-1']},
        )

        # Must NOT return {'status': 'updated', 'members': []}  — that was the bug.
        assert result.get('status') == 'created', (
            f"Expected 'created' on race fallthrough, got: {result}"
        )
        assert 'id' in result
        assert result['id'] != 'esc-stale-id', (
            'New L2 must have a different id from the stale one'
        )
        assert result['members'] == ['esc-l1-1']


# ---------------------------------------------------------------------------
# TestPromoteToL2Cascade: end-to-end integration through MCP tools
# ---------------------------------------------------------------------------


class TestPromoteToL2Cascade:
    """End-to-end: promote_to_l2 + resolve_issue (via MCP tools) cascades to members.

    Verification strategy:
    - Seed L1 escalations via queue.submit() directly (bypasses MCP, mirrors _seed_esc helper).
    - Call promote_to_l2 via MCP tool to file an L2 referencing both.
    - Verify L1s remain pending at L1 after promote_to_l2 (members stay at L1).
    - Call resolve_issue on the L2 via MCP tool.
    - Verify members are now resolved with the cascade attribution.
    """

    def _seed_l1(self, queue: EscalationQueue, esc_id: str, task_id: str) -> Escalation:
        """Seed a pending L1 escalation directly via queue.submit()."""
        esc = Escalation(
            id=esc_id,
            task_id=task_id,
            agent_role='steward',
            severity='blocking',
            category='design_concern',
            summary='L1 cluster member',
            level=1,
        )
        queue.submit(esc)
        return esc

    @pytest.mark.asyncio
    async def test_members_stay_pending_after_promote(self, tmp_path: Path):
        """(c) After promote_to_l2, L1 members remain pending — not pulled to L2."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        self._seed_l1(queue, 'esc-l1-1', 'task-1')
        self._seed_l1(queue, 'esc-l1-2', 'task-2')

        await _promote_to_l2(
            server,
            **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-1', 'esc-l1-2']},
        )

        # Both L1s must still be pending
        all_pending_l1 = await _get_pending(server, level=1)
        pending_l1 = [e for e in all_pending_l1 if e['id'] in {'esc-l1-1', 'esc-l1-2'}]
        assert len(pending_l1) == 2, (
            f'Expected both L1s still pending, got {[e["id"] for e in pending_l1]}'
        )
        for e in pending_l1:
            assert e['status'] == 'pending', f'Expected pending, got {e["status"]!r}'

    @pytest.mark.asyncio
    async def test_resolve_l2_cascades_to_members_via_mcp(self, tmp_path: Path):
        """(d) resolve_issue on the L2 cascades resolution to members."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        self._seed_l1(queue, 'esc-l1-1', 'task-1')
        self._seed_l1(queue, 'esc-l1-2', 'task-2')

        l2_result = await _promote_to_l2(
            server,
            **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-1', 'esc-l1-2']},
        )
        l2_id = l2_result['id']

        # Resolve the L2 via MCP tool
        resolve_result = await _resolve_issue(server, escalation_id=l2_id, resolution='Root cause fixed')

        assert resolve_result.get('status') == 'resolved', (
            f'Expected L2 resolved, got: {resolve_result}'
        )

        # Members must now be resolved
        m1 = queue.get('esc-l1-1')
        m2 = queue.get('esc-l1-2')
        assert m1 is not None
        assert m2 is not None
        assert m1.status == 'resolved', f'Expected m1 resolved, got {m1.status!r}'
        assert m2.status == 'resolved', f'Expected m2 resolved, got {m2.status!r}'

    @pytest.mark.asyncio
    async def test_cascade_preserves_resolution_text_in_members(self, tmp_path: Path):
        """(d cont.) Cascaded members have the same resolution text as the L2."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        self._seed_l1(queue, 'esc-l1-1', 'task-1')
        l2_result = await _promote_to_l2(
            server,
            **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-1']},
        )

        await _resolve_issue(server, escalation_id=l2_result['id'], resolution='Fix confirmed in prod')

        m1 = queue.get('esc-l1-1')
        assert m1 is not None
        assert m1.resolution == 'Fix confirmed in prod', (
            f"Expected resolution text propagated to member, got {m1.resolution!r}"
        )

    @pytest.mark.asyncio
    async def test_cascade_sets_resolved_by_audit_attribution(self, tmp_path: Path):
        """(e) Members carry resolved_by='l2-cascade:{l2_id}' for audit attribution."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        self._seed_l1(queue, 'esc-l1-1', 'task-1')
        l2_result = await _promote_to_l2(
            server,
            **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-1']},
        )
        l2_id = l2_result['id']

        await _resolve_issue(server, escalation_id=l2_id, resolution='Fixed')

        m1 = queue.get('esc-l1-1')
        assert m1 is not None
        assert m1.resolved_by == f'l2-cascade:{l2_id}', (
            f"Expected l2-cascade attribution, got {m1.resolved_by!r}"
        )

    @pytest.mark.asyncio
    async def test_dismiss_l2_cascades_dismiss_to_members(self, tmp_path: Path):
        """(f) Dismiss the L2 (terminate=True) → members are dismissed."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        self._seed_l1(queue, 'esc-l1-1', 'task-1')
        self._seed_l1(queue, 'esc-l1-2', 'task-2')
        l2_result = await _promote_to_l2(
            server,
            **{**_L2_DEFAULTS, 'member_ids': ['esc-l1-1', 'esc-l1-2']},
        )

        await _resolve_issue(
            server,
            escalation_id=l2_result['id'],
            resolution='Not actionable',
            terminate=True,
        )

        m1 = queue.get('esc-l1-1')
        m2 = queue.get('esc-l1-2')
        assert m1 is not None
        assert m2 is not None
        assert m1.status == 'dismissed', f'Expected m1 dismissed, got {m1.status!r}'
        assert m2.status == 'dismissed', f'Expected m2 dismissed, got {m2.status!r}'
