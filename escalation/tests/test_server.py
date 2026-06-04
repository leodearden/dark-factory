"""Tests for L2 severity-gated born-at-L2 path and get_pending_escalations level filter.

Uses the async FastMCP unit-test pattern from test_server_chokepoint.py:
    tool = await server.get_tool(name)
    result = await tool.fn(...)

and tmp_path isolation with EscalationQueue.
"""

from __future__ import annotations

import asyncio
import time
import types
from pathlib import Path
from typing import Any

import pytest

from escalation.dedupe import DedupeConfig, summary_dedupe_key
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from escalation.server import create_server

# ---------------------------------------------------------------------------
# Cross-package orchestrator imports — used by TestMergeStatus.
# Guarded so the rest of the file is still collected when the orchestrator
# package is absent (e.g. in an escalation-only install).
# ---------------------------------------------------------------------------
try:
    from orchestrator.config import OrchestratorConfig  # type: ignore[reportMissingImports]
    from orchestrator.event_store import EventStore, EventType  # type: ignore[reportMissingImports]
    from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
        MergeRequest,
        SpeculativeItem,
        SpeculativeMergeWorker,
        TerminalOutcomeRecord,
        TerminalOutcomeRetention,
    )
    _ORCHESTRATOR_AVAILABLE = True
except ImportError:
    _ORCHESTRATOR_AVAILABLE = False
    # Satisfy pyright's definite-assignment check.  The TestMergeStatus class
    # is guarded by @pytest.mark.skipif(not _ORCHESTRATOR_AVAILABLE) so these
    # stubs are never exercised at runtime.  Annotating as Any lets pyright
    # treat every subsequent use (calls, attribute access) as valid.
    OrchestratorConfig: Any = None
    EventStore: Any = None
    EventType: Any = None
    MergeRequest: Any = None
    SpeculativeItem: Any = None
    SpeculativeMergeWorker: Any = None
    TerminalOutcomeRecord: Any = None
    TerminalOutcomeRetention: Any = None

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
        """escalate_blocker(severity='critical') → on-disk escalation has level==2.

        Uses sentinel agent_role ('orchestrator-watcher-supervisor') so the
        C4/D3 downgrade is bypassed and born-at-L2 stamping is still covered.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(
            server, severity='critical',
            **{**_COMMON_KWARGS, 'agent_role': 'orchestrator-watcher-supervisor'},
        )

        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.level == 2, f"Expected level==2, got: {esc.level}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize('severity', ['critical', 'urgent'])
    async def test_blocker_l2_severities_yield_level2(self, tmp_path: Path, severity: str):
        """escalate_blocker with critical/urgent severity → on-disk level==2.

        Uses sentinel agent_role so the C4/D3 downgrade is bypassed — this
        test covers born-at-L2 stamping AND the sentinel exemption regression.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(
            server, severity=severity,
            **{**_COMMON_KWARGS, 'agent_role': 'orchestrator-watcher-supervisor'},
        )

        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.level == 2, f"severity={severity!r}: expected level==2, got: {esc.level}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize('severity', ['critical', 'urgent'])
    async def test_info_l2_severities_yield_level2(self, tmp_path: Path, severity: str):
        """escalate_info with critical/urgent severity → on-disk level==2.

        Uses sentinel agent_role so the C4/D3 downgrade is bypassed — this
        test covers born-at-L2 stamping AND the sentinel exemption regression.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _info(
            server, severity=severity,
            **{**_COMMON_KWARGS, 'agent_role': 'orchestrator-watcher-supervisor'},
        )

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
# TestGetPendingCompact: get_pending_escalations(compact=True) projection
# ---------------------------------------------------------------------------


class TestGetPendingCompact:
    """get_pending_escalations(compact=True) returns only the triage-relevant fields."""

    _COMPACT_KEYS = {
        'id', 'task_id', 'category', 'severity', 'level', 'status',
        'summary', 'suggested_action', 'timestamp',
    }
    # Heavy fields that compact mode must omit.
    _HEAVY_KEYS = {
        'detail', 'members', 'options', 'root_cause', 'train_state',
        'workflow_state', 'worktree', 'dedupe_children', 'dedupe_fingerprint',
        'resolution',
    }

    def _seed_heavy(self, queue: EscalationQueue, task_id: str, level: int) -> Escalation:
        """Seed a pending escalation populated with heavy fields, at the given level."""
        esc = Escalation(
            id=queue.make_id(task_id),
            task_id=task_id,
            agent_role='implementer',
            severity='blocking',
            category='design_concern',
            summary=f'level={level} compact test',
            detail='x' * 4000,  # large free-text — must be dropped in compact mode
            suggested_action='manual_intervention',
            level=level,
            members=['esc-1-1', 'esc-1-2'],
            root_cause='shared root cause hypothesis',
            options=['A: rollback', 'B: fix forward'],
            workflow_state='REVIEW',
            worktree='/abs/path/.worktrees/1',
        )
        queue.submit(esc)
        return esc

    @pytest.mark.asyncio
    async def test_compact_projects_only_light_fields(self, tmp_path: Path):
        """compact=True returns exactly the triage keys and omits the heavy ones."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        seeded = self._seed_heavy(queue, 'task-A', level=2)

        result = await _get_pending(server, level=2, compact=True)

        assert len(result) == 1, f"Expected 1 result, got {len(result)}: {result}"
        keys = set(result[0].keys())
        assert keys == self._COMPACT_KEYS, f"Unexpected compact key set: {keys}"
        assert not (keys & self._HEAVY_KEYS), f"Heavy fields leaked: {keys & self._HEAVY_KEYS}"
        # Values that ARE projected must be faithful.
        assert result[0]['id'] == seeded.id
        assert result[0]['suggested_action'] == 'manual_intervention'

    @pytest.mark.asyncio
    async def test_compact_false_is_default_full_dict(self, tmp_path: Path):
        """Default (compact omitted) preserves the full to_dict() shape — back-compat."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        self._seed_heavy(queue, 'task-A', level=2)

        result = await _get_pending(server, level=2)

        assert len(result) == 1
        # Heavy fields present in the full shape.
        assert result[0]['detail'].startswith('x')
        assert result[0]['members'] == ['esc-1-1', 'esc-1-2']
        assert result[0]['root_cause'] == 'shared root cause hypothesis'

    @pytest.mark.asyncio
    async def test_compact_preserves_filter_and_count(self, tmp_path: Path):
        """compact=True still honours the level filter and returns one row per match."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        self._seed_heavy(queue, 'task-A', level=0)
        self._seed_heavy(queue, 'task-A', level=1)
        self._seed_heavy(queue, 'task-A', level=2)
        self._seed_heavy(queue, 'task-B', level=2)

        result = await _get_pending(server, level=2, compact=True)

        assert len(result) == 2, f"Expected 2 L2 rows, got {len(result)}: {result}"
        assert all(r['level'] == 2 for r in result)
        assert all(set(r.keys()) == self._COMPACT_KEYS for r in result)


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

        # Child call: same category, overlapping summary, but born-at-L2.
        # Uses sentinel agent_role so the C4/D3 downgrade is bypassed —
        # dedupe-bypass coverage requires level==2 on the child record.
        result = await _info(
            server,
            task_id='task-999',
            agent_role='orchestrator-watcher-supervisor',
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
            agent_role='orchestrator-watcher-supervisor',
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

        # Uses sentinel agent_role so the C4/D3 downgrade is bypassed and
        # the born-at-L2 + terminal-task auto-resolve combination is covered.
        result = await _blocker(
            server, severity='critical',
            **{**_COMMON_KWARGS, 'agent_role': 'orchestrator-watcher-supervisor'},
        )

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

        # Uses sentinel agent_role so the C4/D3 downgrade is bypassed —
        # dedupe-bypass coverage requires the urgent escalation to stay at L2.
        result = await _blocker(
            server,
            task_id='task-999',
            agent_role='orchestrator-watcher-supervisor',
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
# TestStrandedBlockedCategory: 'stranded_blocked' is registered in CATEGORIES
# ---------------------------------------------------------------------------


class TestStrandedBlockedCategory:
    """CATEGORIES constant must include 'stranded_blocked' (PRD-3 D5 data-contract)."""

    def test_stranded_blocked_in_categories(self):
        """'stranded_blocked' must be a member of CATEGORIES at runtime.

        This is a data-contract assertion on a runtime constant — not a prose
        test.  CATEGORIES is the canonical list downstream consumers (task ε,
        task ι, task ζ) rely on to recognise the category.
        """
        from escalation.server import CATEGORIES
        assert 'stranded_blocked' in CATEGORIES, (
            f"'stranded_blocked' missing from CATEGORIES; current list: {CATEGORIES}"
        )


# ---------------------------------------------------------------------------
# TestResolveIssueActionEnum: C1/C2 action enum contract + terminate= migration guard
# ---------------------------------------------------------------------------


class TestResolveIssueActionEnum:
    """resolve_issue action enum: C1 semantics table + terminate= migration guard (C2)."""

    def _seed_pending(self, queue: EscalationQueue, esc_id: str = 'esc-t1-0001') -> Escalation:
        esc = Escalation(
            id=esc_id,
            task_id='t-1',
            agent_role='implementer',
            severity='blocking',
            category='scope_violation',
            summary='action enum test escalation',
        )
        queue.submit(esc)
        return esc

    # --- (a) terminate= sentinel ---

    @pytest.mark.asyncio
    async def test_terminate_true_returns_migration_error(self, tmp_path: Path):
        """terminate=True returns {'error': ...} naming all five actions; record stays pending."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue)

        result = await _resolve_issue(
            server, escalation_id=esc.id, resolution='done', terminate=True,
        )

        assert 'error' in result, f"Expected error dict, got: {result}"
        msg = result['error']
        for action in ('resume', 'restart', 'park', 'abandon', 'close_only'):
            assert action in msg, f"Migration error must name '{action}'; got: {msg!r}"
        # Record must be unchanged
        record = queue.get(esc.id)
        assert record is not None
        assert record.status == 'pending', f"Record must stay pending; got {record.status!r}"

    @pytest.mark.asyncio
    async def test_terminate_false_returns_migration_error(self, tmp_path: Path):
        """terminate=False also returns {'error': ...} (any non-None value triggers guard)."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue)

        result = await _resolve_issue(
            server, escalation_id=esc.id, resolution='done', terminate=False,
        )

        assert 'error' in result, f"Expected error dict for terminate=False, got: {result}"
        msg = result['error']
        for action in ('resume', 'restart', 'park', 'abandon', 'close_only'):
            assert action in msg, f"Migration error must name '{action}'; got: {msg!r}"
        record = queue.get(esc.id)
        assert record is not None
        assert record.status == 'pending', f"Record must stay pending; got {record.status!r}"

    # --- (b) default + 'resume' → resolved ---

    @pytest.mark.asyncio
    async def test_default_action_resolves(self, tmp_path: Path):
        """Default call (no action kwarg) resolves the escalation."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue)

        result = await _resolve_issue(server, escalation_id=esc.id, resolution='fixed')

        assert result.get('status') == 'resolved', f"Expected resolved; got: {result}"

    @pytest.mark.asyncio
    async def test_action_resume_resolves(self, tmp_path: Path):
        """action='resume' resolves the escalation."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue)

        result = await _resolve_issue(
            server, escalation_id=esc.id, resolution='fixed', action='resume',
        )

        assert result.get('status') == 'resolved', f"Expected resolved; got: {result}"

    # --- (c) restart → resolved ---

    @pytest.mark.asyncio
    async def test_action_restart_resolves(self, tmp_path: Path):
        """action='restart' resolves the escalation (dismiss=False path)."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue)

        result = await _resolve_issue(
            server, escalation_id=esc.id, resolution='restart approved', action='restart',
        )

        assert result.get('status') == 'resolved', f"Expected resolved; got: {result}"

    # --- (d) park / abandon / close_only → dismissed ---

    @pytest.mark.asyncio
    async def test_action_park_dismisses(self, tmp_path: Path):
        """action='park' dismisses the escalation."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue)

        result = await _resolve_issue(
            server, escalation_id=esc.id, resolution='parked', action='park',
        )

        assert result.get('status') == 'dismissed', f"Expected dismissed; got: {result}"

    @pytest.mark.asyncio
    async def test_action_abandon_dismisses(self, tmp_path: Path):
        """action='abandon' dismisses the escalation."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue)

        result = await _resolve_issue(
            server, escalation_id=esc.id, resolution='abandoned', action='abandon',
        )

        assert result.get('status') == 'dismissed', f"Expected dismissed; got: {result}"

    @pytest.mark.asyncio
    async def test_action_close_only_dismisses(self, tmp_path: Path):
        """action='close_only' dismisses the escalation."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue)

        result = await _resolve_issue(
            server, escalation_id=esc.id, resolution='closed', action='close_only',
        )

        assert result.get('status') == 'dismissed', f"Expected dismissed; got: {result}"

    # --- (e) invalid action → error, record unchanged ---

    @pytest.mark.asyncio
    async def test_action_bogus_returns_error_record_unchanged(self, tmp_path: Path):
        """action='bogus' returns {'error': ...} and record remains pending."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue)

        result = await _resolve_issue(
            server, escalation_id=esc.id, resolution='fixed', action='bogus',
        )

        assert 'error' in result, f"Expected error dict for invalid action; got: {result}"
        record = queue.get(esc.id)
        assert record is not None
        assert record.status == 'pending', f"Record must stay pending; got {record.status!r}"


# ---------------------------------------------------------------------------
# TestResolveIssueResolutionActionPersisted: C1 resolution_action persisted to disk
# ---------------------------------------------------------------------------


class TestResolveIssueResolutionActionPersisted:
    """resolve_issue stamps resolution_action on the resolved record (C1 persistence)."""

    def _seed_pending(self, queue: EscalationQueue, esc_id: str = 'esc-p1-0001') -> Escalation:
        esc = Escalation(
            id=esc_id,
            task_id='t-persist',
            agent_role='implementer',
            severity='blocking',
            category='scope_violation',
            summary='persistence test escalation',
        )
        queue.submit(esc)
        return esc

    @pytest.mark.asyncio
    async def test_park_resolution_action_persisted(self, tmp_path: Path):
        """action='park' → queue.get(id).resolution_action == 'park' with status 'dismissed'."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue)

        await _resolve_issue(
            server, escalation_id=esc.id, resolution='parked', action='park',
        )

        record = queue.get(esc.id)
        assert record is not None
        assert record.status == 'dismissed', f"Expected dismissed; got {record.status!r}"
        assert record.resolution_action == 'park', (
            f"Expected resolution_action='park'; got {record.resolution_action!r}"
        )

    @pytest.mark.asyncio
    async def test_resume_resolution_action_persisted(self, tmp_path: Path):
        """action='resume' → queue.get(id).resolution_action == 'resume' with status 'resolved'."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue)

        await _resolve_issue(
            server, escalation_id=esc.id, resolution='fixed', action='resume',
        )

        record = queue.get(esc.id)
        assert record is not None
        assert record.status == 'resolved', f"Expected resolved; got {record.status!r}"
        assert record.resolution_action == 'resume', (
            f"Expected resolution_action='resume'; got {record.resolution_action!r}"
        )

    @pytest.mark.asyncio
    async def test_resolve_issue_return_dict_includes_resolution_action(self, tmp_path: Path):
        """resolve_issue return dict includes resolution_action key with the chosen action."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue)

        result = await _resolve_issue(
            server, escalation_id=esc.id, resolution='closed', action='close_only',
        )

        assert 'resolution_action' in result, f"Return dict missing resolution_action; got: {result}"
        assert result['resolution_action'] == 'close_only', (
            f"Expected resolution_action='close_only'; got {result['resolution_action']!r}"
        )



# ---------------------------------------------------------------------------
# TestResolveIssueTerminateMcpPath: terminate= guard reachable via MCP dispatch
# ---------------------------------------------------------------------------


class TestResolveIssueTerminateMcpPath:
    """terminate= migration guard is reachable through server.call_tool() (MCP dispatch path).

    The terminate parameter is annotated Any so FastMCP's pydantic validation
    does NOT reject non-null values at the schema layer.  Without this the
    caller would get an opaque ValidationError instead of the friendly
    migration error naming the five replacement actions.
    """

    def _seed_pending(self, queue: EscalationQueue, esc_id: str = 'esc-mcp-0001') -> Escalation:
        esc = Escalation(
            id=esc_id,
            task_id='t-mcp',
            agent_role='implementer',
            severity='blocking',
            category='scope_violation',
            summary='mcp path terminate test',
        )
        queue.submit(esc)
        return esc

    @pytest.mark.asyncio
    async def test_terminate_true_via_call_tool_returns_friendly_error(self, tmp_path: Path):
        """terminate=True through server.call_tool() returns the migration error, not a ValidationError.

        This exercises the full MCP dispatch path (pydantic schema validation +
        middleware) rather than the bare tool.fn() shortcut, confirming the
        guard is actually reachable end-to-end.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue)

        tool_result = await server.call_tool('resolve_issue', {
            'escalation_id': esc.id,
            'resolution': 'done',
            'terminate': True,
        })

        result = tool_result.structured_content
        assert result is not None, (
            "call_tool returned no structured_content — "
            "terminate=True may have been rejected at the schema layer"
        )
        assert 'error' in result, (
            f"Expected friendly migration error dict via MCP path, got: {result}"
        )
        msg = result['error']
        for action in ('resume', 'restart', 'park', 'abandon', 'close_only'):
            assert action in msg, (
                f"Migration error must name '{action}'; got: {msg!r}"
            )
        # Record must be unchanged
        record = queue.get(esc.id)
        assert record is not None
        assert record.status == 'pending', (
            f"Record must stay pending after migration error; got {record.status!r}"
        )


# ---------------------------------------------------------------------------
# TestChokepointSeverityDowngrade: C4/D3 agent-role downgrade + sentinel exemption
# ---------------------------------------------------------------------------


class TestChokepointSeverityDowngrade:
    """Critical/urgent escalations from AGENT roles are downgraded to 'blocking' (C4/D3).

    (A) AGENT-ROLE DOWNGRADE — agent_role='implementer', severity in critical/urgent:
        - result['status'] == 'queued'
        - on-disk record: severity=='blocking', level==0
        - summary has '[downgraded:critical]' / '[downgraded:urgent]' appended as suffix
          (appended so summary_dedupe_key's first-three-token slice is preserved)
        - a WARNING is emitted on logger 'escalation.server'

    (B) SENTINEL EXEMPTION — harness-/orchestrator- prefixed roles keep born-at-L2:
        - on-disk record: severity unchanged, level==2
        - summary NOT modified (no marker appended)
        - NO WARNING logged
    """

    # --- (A) Agent-role downgrade via escalate_blocker ---

    @pytest.mark.asyncio
    @pytest.mark.parametrize('severity', ['critical', 'urgent'])
    async def test_blocker_agent_role_downgrade_returns_queued(
        self, tmp_path: Path, severity: str,
    ):
        """escalate_blocker(severity='critical'/'urgent', agent_role='implementer') → status=='queued'."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(
            server,
            task_id='task-999',
            agent_role='implementer',
            category='scope_violation',
            summary=f'original {severity} summary',
            severity=severity,
        )

        assert result['status'] == 'queued', (
            f"severity={severity!r}: expected 'queued' after downgrade, got: {result}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('severity', ['critical', 'urgent'])
    async def test_blocker_agent_role_downgrade_on_disk_severity_blocking_level0(
        self, tmp_path: Path, severity: str,
    ):
        """escalate_blocker agent downgrade: on-disk record has severity=='blocking', level==0."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(
            server,
            task_id='task-999',
            agent_role='implementer',
            category='scope_violation',
            summary=f'original {severity} summary',
            severity=severity,
        )

        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.severity == 'blocking', (
            f"severity={severity!r}: expected on-disk severity=='blocking', got: {esc.severity!r}"
        )
        assert esc.level == 0, (
            f"severity={severity!r}: expected on-disk level==0, got: {esc.level}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('severity', ['critical', 'urgent'])
    async def test_blocker_agent_role_downgrade_summary_suffixed(
        self, tmp_path: Path, severity: str,
    ):
        """escalate_blocker agent downgrade: summary has '[downgraded:<original>]' appended as suffix.

        The marker is appended (not prepended) so summary_dedupe_key's first-three-token
        slice stays equal to the original summary's key (PRD C4 — marker visible on the
        summary line, placed at the suffix to preserve the dedupe key).
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        original_summary = f'original {severity} summary'

        result = await _blocker(
            server,
            task_id='task-999',
            agent_role='implementer',
            category='scope_violation',
            summary=original_summary,
            severity=severity,
        )

        esc = queue.get(result['id'])
        assert esc is not None
        expected_suffix = f' [downgraded:{severity}]'
        assert esc.summary.startswith(original_summary), (
            f"severity={severity!r}: expected summary to start with original text, got: {esc.summary!r}"
        )
        assert esc.summary.endswith(expected_suffix), (
            f"severity={severity!r}: expected summary to end with {expected_suffix!r}, got: {esc.summary!r}"
        )
        assert summary_dedupe_key(esc.summary) == summary_dedupe_key(original_summary), (
            f"severity={severity!r}: dedupe key must be preserved after appending marker"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('severity', ['critical', 'urgent'])
    async def test_blocker_agent_role_downgrade_emits_warning(
        self, tmp_path: Path, severity: str, caplog,
    ):
        """escalate_blocker agent downgrade: WARNING emitted on logger 'escalation.server'."""
        import logging
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        with caplog.at_level(logging.WARNING, logger='escalation.server'):
            await _blocker(
                server,
                task_id='task-999',
                agent_role='implementer',
                category='scope_violation',
                summary=f'original {severity} summary',
                severity=severity,
            )

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING
                    and r.name == 'escalation.server']
        assert warnings, (
            f"severity={severity!r}: expected a WARNING on logger 'escalation.server', got: {caplog.records}"
        )

    # --- (A) Agent-role downgrade via escalate_info ---

    @pytest.mark.asyncio
    @pytest.mark.parametrize('severity', ['critical', 'urgent'])
    async def test_info_agent_role_downgrade_on_disk_severity_blocking_level0(
        self, tmp_path: Path, severity: str,
    ):
        """escalate_info agent downgrade: on-disk record has severity=='blocking', level==0."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _info(
            server,
            task_id='task-999',
            agent_role='implementer',
            category='scope_violation',
            summary=f'original {severity} info summary',
            severity=severity,
        )

        assert result['status'] == 'queued', (
            f"severity={severity!r}: expected 'queued', got: {result}"
        )
        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.severity == 'blocking', (
            f"severity={severity!r}: expected severity=='blocking', got: {esc.severity!r}"
        )
        assert esc.level == 0, (
            f"severity={severity!r}: expected level==0, got: {esc.level}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('severity', ['critical', 'urgent'])
    async def test_info_agent_role_downgrade_summary_suffixed(
        self, tmp_path: Path, severity: str,
    ):
        """escalate_info agent downgrade: summary has '[downgraded:<original>]' appended as suffix.

        The marker is appended (not prepended) so summary_dedupe_key's first-three-token
        slice stays equal to the original summary's key (PRD C4 — marker visible on the
        summary line, placed at the suffix to preserve the dedupe key).
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        original_summary = f'original {severity} info summary'

        result = await _info(
            server,
            task_id='task-999',
            agent_role='implementer',
            category='scope_violation',
            summary=original_summary,
            severity=severity,
        )

        esc = queue.get(result['id'])
        assert esc is not None
        expected_suffix = f' [downgraded:{severity}]'
        assert esc.summary.startswith(original_summary), (
            f"severity={severity!r}: expected summary to start with original text, got: {esc.summary!r}"
        )
        assert esc.summary.endswith(expected_suffix), (
            f"severity={severity!r}: expected summary to end with {expected_suffix!r}, got: {esc.summary!r}"
        )

    # --- (B) Sentinel exemption (harness- and orchestrator- prefixes) ---

    @pytest.mark.asyncio
    @pytest.mark.parametrize('sentinel_role', [
        'harness-stranded-blocked-reaper',
        'orchestrator-watcher-supervisor',
    ])
    async def test_sentinel_role_keeps_level2_severity_unchanged(
        self, tmp_path: Path, sentinel_role: str,
    ):
        """Sentinel roles are exempt from downgrade: severity stays 'critical', level stays 2."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(
            server,
            task_id='task-999',
            agent_role=sentinel_role,
            category='scope_violation',
            summary='sentinel critical escalation',
            severity='critical',
        )

        assert result['status'] == 'queued', (
            f"sentinel_role={sentinel_role!r}: expected 'queued', got: {result}"
        )
        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.severity == 'critical', (
            f"sentinel_role={sentinel_role!r}: severity must NOT be downgraded, got: {esc.severity!r}"
        )
        assert esc.level == 2, (
            f"sentinel_role={sentinel_role!r}: level must stay 2 (born-at-L2), got: {esc.level}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('sentinel_role', [
        'harness-stranded-blocked-reaper',
        'orchestrator-watcher-supervisor',
    ])
    async def test_sentinel_role_summary_not_prefixed(
        self, tmp_path: Path, sentinel_role: str,
    ):
        """Sentinel roles are exempt: summary is NOT prefixed with '[downgraded:...]'."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        original_summary = 'sentinel critical escalation'

        result = await _blocker(
            server,
            task_id='task-999',
            agent_role=sentinel_role,
            category='scope_violation',
            summary=original_summary,
            severity='critical',
        )

        esc = queue.get(result['id'])
        assert esc is not None
        assert not esc.summary.startswith('[downgraded:'), (
            f"sentinel_role={sentinel_role!r}: summary must NOT be prefixed, got: {esc.summary!r}"
        )
        assert esc.summary == original_summary, (
            f"sentinel_role={sentinel_role!r}: summary must be unchanged, got: {esc.summary!r}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('sentinel_role', [
        'harness-stranded-blocked-reaper',
        'orchestrator-watcher-supervisor',
    ])
    async def test_sentinel_role_no_warning_logged(
        self, tmp_path: Path, sentinel_role: str, caplog,
    ):
        """Sentinel roles are exempt: NO downgrade WARNING is logged."""
        import logging
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        with caplog.at_level(logging.WARNING, logger='escalation.server'):
            await _blocker(
                server,
                task_id='task-999',
                agent_role=sentinel_role,
                category='scope_violation',
                summary='sentinel critical escalation',
                severity='critical',
            )

        # Filter out the task_status_lookup warning (not present here); only
        # check that no "downgrad" warning was emitted.
        downgrade_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and 'downgrad' in r.message.lower()
        ]
        assert not downgrade_warnings, (
            f"sentinel_role={sentinel_role!r}: expected NO downgrade WARNING, got: {downgrade_warnings}"
        )

    # --- (C) Robustness: None/empty agent_role falls through to downgrade ---

    @pytest.mark.parametrize('agent_role', [None, ''])
    def test_none_or_empty_agent_role_is_not_sentinel(self, agent_role):
        """_is_harness_sentinel_role(None/'') returns False, not AttributeError.

        The MCP tool validates agent_role as str, but legacy/deserialized records
        may carry None or empty strings.  The defensive ``(agent_role or '')``
        guard ensures these fall through to the downgrade (non-sentinel) path
        rather than crashing with AttributeError inside the hot submit path.
        """
        from escalation.server import _is_harness_sentinel_role
        # Must not raise, must return False (non-sentinel → downgrade path)
        result = _is_harness_sentinel_role(agent_role)
        assert result is False, (
            f"agent_role={agent_role!r}: expected False (non-sentinel), got: {result!r}"
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
        """(f) Dismiss the L2 (action='abandon') → members are dismissed."""
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
            action='abandon',
        )

        m1 = queue.get('esc-l1-1')
        m2 = queue.get('esc-l1-2')
        assert m1 is not None
        assert m2 is not None
        assert m1.status == 'dismissed', f'Expected m1 dismissed, got {m1.status!r}'
        assert m2.status == 'dismissed', f'Expected m2 dismissed, got {m2.status!r}'


# ---------------------------------------------------------------------------
# TestMergeRequestDedup — step-11: merge_request de-dup wiring (server-level)
# ---------------------------------------------------------------------------


async def _call_merge_request(server, **kwargs: Any) -> dict[str, Any]:
    """Invoke the merge_request MCP tool directly."""
    tool = await server.get_tool('merge_request')
    return await tool.fn(**kwargs)


@pytest.mark.asyncio
class TestMergeRequestDedup:
    """Server-level de-dup tests for merge_request.

    The disk-scan path is covered at the merge_queue/git_ops level (steps 7/9).
    These tests focus on the registry-only path, using an injected
    InFlightMergeRegistry (new optional create_server param) so no real git
    repo or worker is needed.
    """

    def _make_orch_config(self, tmp_path: Path):
        """Create a minimal OrchestratorConfig without a git remote."""
        from orchestrator.config import OrchestratorConfig  # type: ignore[reportMissingImports]
        return OrchestratorConfig(project_root=tmp_path)

    def _make_registry(self):
        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            InFlightMergeRegistry,
        )
        return InFlightMergeRegistry()

    async def test_in_flight_branch_returns_immediately(self, tmp_path: Path):
        """merge_request for an already-in-flight branch returns {status:'in_flight'}
        immediately (no blocking await) and leaves the queue empty.

        Simulates the /unblock-spam scenario: the registry is pre-seeded with
        branch 'X' via a never-resolving future, so the second call should
        coalesce rather than enqueue.
        """
        import asyncio

        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = self._make_orch_config(tmp_path / 'repo')
        registry = self._make_registry()

        # Pre-seed the registry: acquire branch 'X' with a never-resolving future
        never_future: asyncio.Future = asyncio.get_running_loop().create_future()
        acquired = registry.acquire('X', 'existing-task', never_future)
        assert acquired, 'Prerequisite: registry must accept first acquire'

        # Create server with injected registry (new param added in step-12)
        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
        )

        # Call merge_request for branch 'X' — should return in_flight immediately
        # (asyncio.wait_for with 2s proves it does NOT block on the future)
        result = await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='X',
                branch='X',
                worktree=str(tmp_path / 'wt'),
                description='',
            ),
            timeout=2.0,
        )

        assert result.get('status') == 'attached', (
            f'Expected status attached, got: {result}'
        )
        assert mq.empty(), (
            f'Queue should be empty (no enqueue on attached/coalesce), qsize={mq.qsize()}'
        )
        # Clean up the never-resolving future to avoid ResourceWarning
        never_future.cancel()

    async def test_dispatch_resolves_and_releases_registry(self, tmp_path: Path):
        """merge_request with empty registry enqueues, awaits outcome, and releases the slot.

        A background task dequeues the MergeRequest and resolves its future with
        MergeOutcome('done'), proving the full dispatch path through the coalesce fn.
        After resolution, registry.is_inflight(branch) must be False (auto-released
        via future.add_done_callback).
        """
        import asyncio

        from orchestrator.merge_queue import MergeOutcome  # type: ignore[reportMissingImports]

        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = self._make_orch_config(tmp_path / 'repo')
        registry = self._make_registry()

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
        )

        async def _worker():
            """Dequeue the first MergeRequest and resolve its future with 'done'."""
            req = await mq.get()
            req.result.set_result(
                MergeOutcome('done', merge_sha='abc123', reason='test merge')
            )

        # Start worker before calling merge_request so it can drain the queue
        worker_task = asyncio.create_task(_worker())

        result = await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='Y',
                branch='Y',
                worktree=str(tmp_path / 'wt'),
                description='',
                wait_secs=100,
            ),
            timeout=5.0,
        )

        await worker_task  # ensure the worker finished cleanly

        assert result.get('status') == 'done', (
            f'Expected status done, got: {result}'
        )
        # After future resolves, the registry slot should be auto-released
        await asyncio.sleep(0)  # let add_done_callback fire
        assert not registry.is_inflight('Y'), (
            'Registry slot should be released after merge completes'
        )

    async def test_inflight_with_workflow_path_registration(self, tmp_path: Path):
        """Acceptance #1: a branch registered via the workflow-path helper is
        visible to the MCP merge_request coalesce gate.

        Simulates the dedup blind-spot fix: the workflow enqueues via
        register_and_enqueue_merge_request (which registers in the shared
        registry), then a subsequent MCP merge_request call for the same
        branch returns status='in_flight' (coalesced from registry) with
        inflight_task_id='workflow-task' and NO duplicate queue entry.
        """
        import asyncio
        from pathlib import Path as _Path

        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            MergeRequest,
            register_and_enqueue_merge_request,
        )

        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = self._make_orch_config(tmp_path / 'repo')
        registry = self._make_registry()

        # Simulate the workflow-path enqueue: build a MergeRequest inline
        # (no _make_request helper in test_server.py — it lives in test_merge_queue.py)
        never_future: asyncio.Future = asyncio.get_running_loop().create_future()
        workflow_req = MergeRequest(
            task_id='workflow-task',
            branch='B',
            worktree=_Path(str(tmp_path / 'wf-wt')),
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=orch_config,
            result=never_future,
        )

        # Workflow-path enqueue registers in the shared registry
        acquired = await register_and_enqueue_merge_request(mq, workflow_req, None, registry)
        assert acquired is True, 'Prerequisite: workflow must acquire the registry slot'
        assert registry.is_inflight('B'), 'Prerequisite: branch must be registered'

        # MCP caller: create server with the SAME shared registry
        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
        )

        # MCP merge_request for the same branch 'B' must coalesce
        result = await _call_merge_request(
            server,
            task_id='mcp-caller',
            branch='B',
            worktree=str(tmp_path / 'mcp-wt'),
            description='',
        )

        # Acceptance assertions
        assert result.get('status') == 'attached', (
            f'Expected status attached (coalesced from registry), got: {result}'
        )
        assert result.get('inflight_task_id') == 'workflow-task', (
            f'Expected inflight_task_id=workflow-task (from registry entry), got: {result}'
        )
        # Only the workflow's entry in the queue — MCP call must NOT enqueue a duplicate
        assert mq.qsize() == 1, (
            f'Expected queue size 1 (workflow entry only), got qsize={mq.qsize()}'
        )

        # Cleanup
        never_future.cancel()


# ---------------------------------------------------------------------------
# TestGetMergeQueue — live merge-queue snapshot via get_merge_queue MCP tool
# ---------------------------------------------------------------------------


async def _call_get_merge_queue(server) -> dict[str, Any]:
    """Invoke the get_merge_queue MCP tool directly (sync tool)."""
    tool = await server.get_tool('get_merge_queue')
    return tool.fn()


@pytest.mark.asyncio
class TestGetMergeQueue:
    """Tests for the get_merge_queue escalation MCP tool."""

    def _make_orch_config(self, tmp_path: Path):
        from orchestrator.config import OrchestratorConfig  # type: ignore[reportMissingImports]
        return OrchestratorConfig(project_root=tmp_path)

    # ── step-1: standalone error ──────────────────────────────────────────

    async def test_standalone_returns_error(self, tmp_path: Path):
        """get_merge_queue with no merge_queue returns an error dict, no exception."""
        esc_queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(esc_queue)  # NO merge_queue — standalone

        result = await _call_get_merge_queue(server)

        assert isinstance(result, dict), f'Expected dict, got {type(result)}'
        assert 'error' in result, f'Expected error key, got: {result}'

    # ── step-3: MergeRequest.enqueued_at default ──────────────────────────

    async def test_merge_request_has_enqueued_at_default(self, tmp_path: Path):
        """MergeRequest has enqueued_at float default; GroupMergeRequest still builds."""
        import asyncio
        import time

        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            GroupMergeRequest,
            MergeRequest,
        )

        loop = asyncio.get_running_loop()
        config = self._make_orch_config(tmp_path / 'repo')

        req = MergeRequest(
            task_id='T1',
            branch='T1',
            worktree=tmp_path / 'wt',
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
        )

        assert isinstance(req.enqueued_at, float), (
            f'enqueued_at must be float, got {type(req.enqueued_at)}'
        )
        assert abs(req.enqueued_at - time.time()) < 60, (
            f'enqueued_at={req.enqueued_at!r} not within 60s of now={time.time()!r}'
        )

        # Regression guard: GroupMergeRequest must still construct without TypeError
        async def _noop_status(ids):
            return {}

        async def _noop_done(tid, sha):
            pass

        grq = GroupMergeRequest(
            task_id='G1',
            branch='G1',
            worktree=tmp_path / 'wt2',
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
            train_id='train-1',
            member_task_ids=['G1'],
            tip_branch='G1',
            tip_task_id='G1',
            status_check=_noop_status,
            mark_member_done=_noop_done,
        )
        assert isinstance(grq.enqueued_at, float)

    # ── step-5: queued blind-spot entry visible ───────────────────────────

    async def test_queued_entry_is_visible(self, tmp_path: Path):
        """A queued MergeRequest (no merge worktree) appears in snapshot — incident blind spot."""
        import asyncio
        import types

        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            MergeRequest,
            SpeculativeMergeWorker,
        )

        loop = asyncio.get_running_loop()
        config = self._make_orch_config(tmp_path / 'repo')
        mq: asyncio.Queue = asyncio.Queue()

        git_ops_stub = types.SimpleNamespace()
        worker = SpeculativeMergeWorker(git_ops=git_ops_stub, queue=mq)  # type: ignore[reportArgumentType]

        req = MergeRequest(
            task_id='Q',
            branch='Q',
            worktree=tmp_path / 'wt',
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
        )
        await mq.put(req)

        # Stub harness exposing _merge_worker
        stub_harness = types.SimpleNamespace(_merge_worker=worker)
        esc_queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(esc_queue, merge_queue=mq, harness=stub_harness)

        result = await _call_get_merge_queue(server)

        assert isinstance(result, dict), f'Expected dict, got: {result}'
        assert 'error' not in result, f'Unexpected error: {result}'
        assert result.get('depth', 0) >= 1, f'Expected depth >= 1, got: {result}'

        entries = result.get('entries', [])
        q_entries = [e for e in entries if e.get('task_id') == 'Q']
        assert q_entries, f'Entry for task_id Q not found in entries: {entries}'
        entry = q_entries[0]
        assert entry['state'] == 'queued', f'Expected queued, got: {entry["state"]}'
        assert entry['worktree'] is None, f'Expected worktree=None, got: {entry["worktree"]}'
        assert entry['pre_rebased'] is False
        assert isinstance(entry['age_secs'], (int, float))
        assert entry['age_secs'] >= 0

    # ── step-7: waiter_alive reflects cancelled future ────────────────────

    async def test_waiter_alive_reflects_cancelled_future(self, tmp_path: Path):
        """waiter_alive=True for a live future, False for a cancelled future."""
        import asyncio
        import types

        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            MergeRequest,
            SpeculativeMergeWorker,
        )

        loop = asyncio.get_running_loop()
        config = self._make_orch_config(tmp_path / 'repo')
        mq: asyncio.Queue = asyncio.Queue()

        git_ops_stub = types.SimpleNamespace()
        worker = SpeculativeMergeWorker(git_ops=git_ops_stub, queue=mq)  # type: ignore[reportArgumentType]

        live_fut = loop.create_future()
        cancelled_fut = loop.create_future()
        cancelled_fut.cancel()

        await mq.put(MergeRequest(
            task_id='LIVE', branch='LIVE', worktree=tmp_path / 'wt1',
            pre_rebased=False, task_files=None, module_configs=[], config=config,
            result=live_fut,
        ))
        await mq.put(MergeRequest(
            task_id='DEAD', branch='DEAD', worktree=tmp_path / 'wt2',
            pre_rebased=False, task_files=None, module_configs=[], config=config,
            result=cancelled_fut,
        ))

        stub_harness = types.SimpleNamespace(_merge_worker=worker)
        esc_queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(esc_queue, merge_queue=mq, harness=stub_harness)

        result = await _call_get_merge_queue(server)
        entries = result.get('entries', [])

        live_entry = next((e for e in entries if e['task_id'] == 'LIVE'), None)
        dead_entry = next((e for e in entries if e['task_id'] == 'DEAD'), None)

        assert live_entry is not None, 'LIVE entry missing from snapshot'
        assert dead_entry is not None, 'DEAD entry missing from snapshot'
        assert live_entry['waiter_alive'] is True, (
            f'Expected waiter_alive=True for live future, got: {live_entry}'
        )
        assert dead_entry['waiter_alive'] is False, (
            f'Expected waiter_alive=False for cancelled future, got: {dead_entry}'
        )

    # ── step-9: verifier/queue-level state mapping ────────────────────────

    @pytest.mark.parametrize('verify_phase', ['verifying', 'gate_reverify', 'finalizing'])
    async def test_verifier_and_queue_level_state_mapping(
        self, tmp_path: Path, verify_phase: str
    ):
        """snapshot() maps _verify_item/_inflight_req/_verifier_queue to correct states."""
        import asyncio
        import types

        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            MergeRequest,
            SpeculativeItem,
            SpeculativeMergeWorker,
        )

        loop = asyncio.get_running_loop()
        config = self._make_orch_config(tmp_path / 'repo')
        mq: asyncio.Queue = asyncio.Queue()
        git_ops_stub = types.SimpleNamespace()
        worker = SpeculativeMergeWorker(git_ops=git_ops_stub, queue=mq)  # type: ignore[reportArgumentType]

        def _req(tid: str):
            return MergeRequest(
                task_id=tid, branch=tid,
                worktree=tmp_path / f'wt-{tid}',
                pre_rebased=False, task_files=None, module_configs=[],
                config=config, result=loop.create_future(),
            )

        # M — in the merger (merging)
        req_M = _req('M')
        worker._inflight_req = req_M

        # A — in the verifier queue (awaiting_verify)
        merge_wt_A = tmp_path / 'mergeA'
        merge_wt_A.mkdir()
        item_A = SpeculativeItem(
            request=_req('A'),
            merge_result=None, merge_wt=merge_wt_A,
            base_sha='base', speculative=False, skip_verify=False,
        )
        await worker._verifier_queue.put(item_A)

        # V — currently being verified (phase = verify_phase param)
        merge_wt_V = tmp_path / 'mergeV'
        merge_wt_V.mkdir()
        item_V = SpeculativeItem(
            request=_req('V'),
            merge_result=None, merge_wt=merge_wt_V,
            base_sha='base', speculative=False, skip_verify=False,
        )
        worker._verify_item = item_V
        worker._verify_phase = verify_phase

        # WIP halt
        worker.halt_for_wip('test-wip')
        worker.set_halt_owner('esc-9')

        stub_harness = types.SimpleNamespace(_merge_worker=worker)
        esc_queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(esc_queue, merge_queue=mq, harness=stub_harness)

        result = await _call_get_merge_queue(server)

        assert 'error' not in result, f'Unexpected error: {result}'
        entries = result['entries']

        # find entries by task_id
        by_id = {e['task_id']: e for e in entries}
        assert 'M' in by_id, f'merging entry missing: {by_id}'
        assert 'A' in by_id, f'awaiting_verify entry missing: {by_id}'
        assert 'V' in by_id, f'verifying entry missing: {by_id}'

        assert by_id['M']['state'] == 'merging'
        assert by_id['A']['state'] == 'awaiting_verify'
        assert by_id['A']['worktree'] == str(merge_wt_A)
        assert by_id['V']['state'] == verify_phase
        assert by_id['V']['worktree'] == str(merge_wt_V)

        # head-of-line = V (verifier-current has lowest position)
        assert result['head_of_line'] == 'V', (
            f'Expected head_of_line=V, got: {result["head_of_line"]}'
        )
        assert by_id['V']['position'] == 0

        # verify_in_progress reflects the current item
        vip = result.get('verify_in_progress')
        assert vip is not None, 'verify_in_progress should be set'
        assert vip['task_id'] == 'V'

        # WIP halt state
        assert result['is_wip_halted'] is True
        assert result['halt_owner_esc_id'] == 'esc-9'

    # ── step-11: pipeline instrumentation sets/clears verify_phase ────────

    async def test_pipeline_sets_and_clears_verify_phase(self, tmp_path: Path):
        """_verify_and_advance sets _verify_phase='verifying' before verify and
        'finalizing' before advance_main; worker._verify_item is None after.
        """
        import asyncio
        import types

        from orchestrator.git_ops import MergeResult  # type: ignore[reportMissingImports]
        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            MergeRequest,
            SpeculativeItem,
            SpeculativeMergeWorker,
        )

        loop = asyncio.get_running_loop()
        config = self._make_orch_config(tmp_path / 'repo')
        mq: asyncio.Queue = asyncio.Queue()

        merge_wt = tmp_path / 'merge'
        merge_wt.mkdir()

        # Tracking lists so we can record the phase at call time
        captured_phases: list[str | None] = []

        async def fake_run_scoped_verification(*args, **kwargs):
            captured_phases.append(worker._verify_phase)
            return types.SimpleNamespace(passed=True, timed_out=False, enospc=False)

        async def fake_advance_main(*args, **kwargs):
            captured_phases.append(worker._verify_phase)
            # Return 'not_descendant' — terminal non-advanced path
            return 'not_descendant'

        async def fake_cleanup_merge_worktree(path):
            pass

        git_ops_stub = types.SimpleNamespace(
            advance_main=fake_advance_main,
            cleanup_merge_worktree=fake_cleanup_merge_worktree,
            config=config,
        )
        worker = SpeculativeMergeWorker(git_ops=git_ops_stub, queue=mq)  # type: ignore[reportArgumentType]

        req = MergeRequest(
            task_id='P',
            branch='P',
            worktree=tmp_path / 'wt',
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
        )
        merge_result = MergeResult(success=True, merge_commit='deadbeef0000000', merge_worktree=merge_wt)
        item = SpeculativeItem(
            request=req,
            merge_result=merge_result,
            merge_wt=merge_wt,
            base_sha='base000',
            speculative=False,
            skip_verify=False,
        )

        import orchestrator.merge_queue as mq_module  # type: ignore[reportMissingImports]
        original_rsv = mq_module.run_scoped_verification
        mq_module.run_scoped_verification = fake_run_scoped_verification
        try:
            await worker._verify_and_advance(item)
        finally:
            mq_module.run_scoped_verification = original_rsv

        # Phases: first call (verify) should be 'verifying', second (advance_main) 'finalizing'
        assert len(captured_phases) >= 2, (
            f'Expected at least 2 phase captures, got: {captured_phases}'
        )
        assert captured_phases[0] == 'verifying', (
            f'Expected verifying before verify, got: {captured_phases[0]!r}'
        )
        assert captured_phases[1] == 'finalizing', (
            f'Expected finalizing before advance_main, got: {captured_phases[1]!r}'
        )

        # Request future should be resolved (not_descendant → blocked)
        assert req.result.done(), 'request future should be resolved after terminal advance_main'

        # _verify_item is None — _verify_and_advance doesn't set it
        # (that's _verifier_loop's job); just confirm it was never set
        assert worker._verify_item is None

        # snapshot has no verifying/finalizing entries
        snap = worker.snapshot()
        bad_states = {e['state'] for e in snap['entries']} & {'verifying', 'finalizing', 'gate_reverify'}
        assert not bad_states, f'Snapshot should have no active verifier states, got: {bad_states}'

    # ── amend: gate_reverify phase set/cleared by production code ─────────

    async def test_gate_reverify_phase_set_and_cleared(self, tmp_path: Path):
        """_verify_and_advance sets _verify_phase='gate_reverify' when advance_main
        returns 'rebased_pending_reverify', and resets to 'finalizing' after the gate
        clears (so subsequent advance_main retries report the correct phase).
        """
        import asyncio
        import types

        from orchestrator.git_ops import MergeResult  # type: ignore[reportMissingImports]
        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            MergeRequest,
            SpeculativeItem,
            SpeculativeMergeWorker,
        )

        loop = asyncio.get_running_loop()
        config = self._make_orch_config(tmp_path / 'repo')
        mq: asyncio.Queue = asyncio.Queue()

        merge_wt = tmp_path / 'merge'
        merge_wt.mkdir()

        advance_calls: list[int] = []
        captured_phases_reverify: list[str | None] = []
        captured_phases_advance2: list[str | None] = []

        async def fake_advance_main(*args, **kwargs):
            advance_calls.append(len(advance_calls) + 1)
            if len(advance_calls) == 1:
                # First call: trigger rebase path
                return 'rebased_pending_reverify'
            else:
                # Second call (after gate cleared): terminal failure
                captured_phases_advance2.append(worker._verify_phase)
                return 'not_descendant'

        async def fake_cleanup_merge_worktree(path):
            pass

        # Side-channel attributes read by _verify_and_advance after
        # 'rebased_pending_reverify' to extract the post-rebase SHAs.
        git_ops_stub = types.SimpleNamespace(
            advance_main=fake_advance_main,
            cleanup_merge_worktree=fake_cleanup_merge_worktree,
            config=config,
            _last_advanced_sha='rebased0abc',
            _rebased_from='from0sha',
            _rebased_onto='onto0sha',
        )
        worker = SpeculativeMergeWorker(git_ops=git_ops_stub, queue=mq)  # type: ignore[reportArgumentType]

        req = MergeRequest(
            task_id='GR',
            branch='GR',
            worktree=tmp_path / 'wt',
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
        )
        merge_result = MergeResult(
            success=True,
            merge_commit='deadbeef00000001',
            merge_worktree=merge_wt,
        )
        # skip_verify=True: bypass Step 4 and go straight to the advance_main loop,
        # which is where 'gate_reverify' is triggered.
        item = SpeculativeItem(
            request=req,
            merge_result=merge_result,
            merge_wt=merge_wt,
            base_sha='base0sha',
            speculative=False,
            skip_verify=True,
        )

        import orchestrator.merge_queue as mq_module  # type: ignore[reportMissingImports]

        async def fake_reverify_rebased_tree(*args, **kwargs):
            # Capture the phase at the moment _reverify_rebased_tree is invoked.
            captured_phases_reverify.append(worker._verify_phase)
            # Return None → gate cleared (disjoint/green), advance proceeds.
            return None

        original_reverify = mq_module._reverify_rebased_tree
        mq_module._reverify_rebased_tree = fake_reverify_rebased_tree  # type: ignore[attr-defined]
        try:
            await worker._verify_and_advance(item)
        finally:
            mq_module._reverify_rebased_tree = original_reverify  # type: ignore[attr-defined]

        # _reverify_rebased_tree must have been called exactly once
        assert len(captured_phases_reverify) == 1, (
            f'Expected _reverify_rebased_tree called once, got: {len(captured_phases_reverify)}'
        )
        # Phase must be 'gate_reverify' when _reverify_rebased_tree is invoked
        assert captured_phases_reverify[0] == 'gate_reverify', (
            f'Expected gate_reverify at reverify call, got: {captured_phases_reverify[0]!r}'
        )

        # advance_main must have been called twice
        assert len(advance_calls) == 2, (
            f'Expected 2 advance_main calls, got: {advance_calls}'
        )
        # After gate cleared, phase must be 'finalizing' when the second advance_main runs
        assert len(captured_phases_advance2) == 1, (
            f'Expected phase captured for second advance_main, got: {captured_phases_advance2}'
        )
        assert captured_phases_advance2[0] == 'finalizing', (
            f'Expected finalizing after gate cleared, got: {captured_phases_advance2[0]!r}'
        )

        # Request future is resolved (not_descendant → blocked)
        assert req.result.done(), 'request future should be resolved after terminal advance_main'


# ---------------------------------------------------------------------------
# TestDowngradeDedupeCorrectness — C4/D3 review fix: appended marker vs. dedupe key
# ---------------------------------------------------------------------------


class TestDowngradeDedupeCorrectness:
    """Downgraded marker must not corrupt the summary_dedupe_key.

    summary_dedupe_key() keys on the FIRST three whitespace tokens.  When the
    marker was PREPENDED ('[downgraded:critical] {summary}'), every token
    shifted: downgraded criticals never matched equivalent normally-filed
    'blocking' parents (defeating dedupe), and unrelated criticals whose first
    two words matched could false-merge on the constant leading token.

    The fix (step-6) APPENDS the marker so the leading tokens are unchanged.
    All four tests below fail on current (prepend) code and pass once the
    marker is appended:

    (1) PRISTINE KEY — key(esc.summary) == key(original)
    (2) FOLDS INTO BLOCKING PARENT — downgraded critical matches blocking parent
    (3) TWO AGENT-CRITICALS DEDUPE — second downgraded critical folds into first
    (4) DISTINCT SUMMARIES DO NOT MERGE — different 3rd token keeps them separate
    """

    # IMPORTANT: DedupeConfig() folds only category='infra_issue' (default
    # infra_dedupe_categories=('infra_issue',)).  The downgrade tests above use
    # category='scope_violation' which never dedupes.  These tests use
    # category='infra_issue' so submit_or_dedupe actually attempts a fold.

    _SUMMARY_A = 'fused memory connection timeout'
    _SUMMARY_B = 'fused memory disk full'

    @pytest.mark.asyncio
    async def test_pristine_key_equals_original_summary_key(self, tmp_path: Path):
        """(1) PRISTINE KEY — downgraded record key == original summary key.

        After downgrade, summary_dedupe_key(esc.summary) must equal
        summary_dedupe_key(original_summary) so the record can fold into a
        normally-filed 'blocking' parent with the same summary.

        On current (prepend) code this FAILS because the key shifts to
        ('downgradedcritical', first, second) instead of (first, second, third).
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue, dedupe_config=DedupeConfig())
        S = self._SUMMARY_A

        result = await _blocker(
            server,
            task_id='task-999',
            agent_role='implementer',
            category='infra_issue',
            summary=S,
            severity='critical',
        )

        assert result['status'] == 'queued', f'Expected queued, got: {result}'
        esc = queue.get(result['id'])
        assert esc is not None
        # The marker must be present somewhere in the summary (not stripped)
        assert '[downgraded:critical]' in esc.summary, (
            f'Expected downgrade marker in summary, got: {esc.summary!r}'
        )
        # KEY INVARIANT: downgraded summary's key must equal the original key
        assert summary_dedupe_key(esc.summary) == summary_dedupe_key(S), (
            f'summary_dedupe_key mismatch after downgrade:\n'
            f'  downgraded key: {summary_dedupe_key(esc.summary)}\n'
            f'  original key:   {summary_dedupe_key(S)}\n'
            f'  esc.summary: {esc.summary!r}'
        )
        # The original text must appear FIRST (not after the marker)
        assert esc.summary.startswith(S), (
            f'Expected summary to start with original text {S!r}, got: {esc.summary!r}'
        )

    @pytest.mark.asyncio
    async def test_downgraded_critical_folds_into_blocking_parent(self, tmp_path: Path):
        """(2) FOLDS INTO BLOCKING PARENT — downgraded critical matches blocking parent.

        Pre-seed a pending severity='blocking', category='infra_issue' parent
        with summary S; then file an agent critical infra_issue with the SAME
        summary S via escalate_blocker; assert result['status']=='dedup_skipped'.

        On current (prepend) code this FAILS because the downgraded key
        ('downgradedcritical', ...) never matches the parent's ('fused', ...).
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue, dedupe_config=DedupeConfig())
        S = self._SUMMARY_A

        # Pre-seed a blocking parent directly via queue.submit()
        parent = Escalation(
            id=queue.make_id('task-100'),
            task_id='task-100',
            agent_role='implementer',
            severity='blocking',
            category='infra_issue',
            summary=S,
        )
        queue.submit(parent)

        # File an agent critical with the SAME summary — should fold into parent
        result = await _blocker(
            server,
            task_id='task-999',
            agent_role='implementer',
            category='infra_issue',
            summary=S,
            severity='critical',
        )

        assert result['status'] == 'dedup_skipped', (
            f'Expected dedup_skipped (downgraded key should match blocking parent), got: {result}'
        )
        assert result.get('parent_id') == parent.id, (
            f'Expected parent_id={parent.id!r}, got parent_id={result.get("parent_id")!r}'
        )

    @pytest.mark.asyncio
    async def test_two_agent_criticals_dedupe(self, tmp_path: Path):
        """(3) TWO AGENT-CRITICALS DEDUPE — second folds into first (regression guard).

        File two agent critical infra_issue escalations with the same summary S:
        first must be 'queued', second must be 'dedup_skipped' with parent_id
        pointing at the first.

        This passes on BOTH current (prepend) and fixed (append) code because
        both records get the same marker → same key.  It locks the invariant
        so any future change that breaks the folding fails this test.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue, dedupe_config=DedupeConfig())
        S = self._SUMMARY_A

        first = await _blocker(
            server,
            task_id='task-1',
            agent_role='implementer',
            category='infra_issue',
            summary=S,
            severity='critical',
        )
        assert first['status'] == 'queued', f'Expected first to be queued, got: {first}'

        second = await _blocker(
            server,
            task_id='task-2',
            agent_role='implementer',
            category='infra_issue',
            summary=S,
            severity='critical',
        )
        assert second['status'] == 'dedup_skipped', (
            f'Expected second agent-critical to fold into first, got: {second}'
        )
        assert second.get('parent_id') == first['id'], (
            f'Expected parent_id={first["id"]!r}, got: {second.get("parent_id")!r}'
        )

    @pytest.mark.asyncio
    async def test_distinct_summaries_do_not_merge(self, tmp_path: Path):
        """(4) DISTINCT SUMMARIES DO NOT MERGE — different 3rd token keeps them separate.

        File two agent critical infra_issue escalations whose summaries share
        the first two real tokens but differ on the third:
          _SUMMARY_A = 'fused memory connection timeout'
          _SUMMARY_B = 'fused memory disk full'
        Both must be 'queued' — their keys differ at the 3rd token.

        On current (prepend) code this FAILS because both summaries collapse
        to ('downgradedcritical', 'fused', 'memory'), causing the second to
        return 'dedup_skipped' instead of 'queued'.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue, dedupe_config=DedupeConfig())

        first = await _blocker(
            server,
            task_id='task-1',
            agent_role='implementer',
            category='infra_issue',
            summary=self._SUMMARY_A,
            severity='critical',
        )
        assert first['status'] == 'queued', (
            f'Expected first (SUMMARY_A) to be queued, got: {first}'
        )

        second = await _blocker(
            server,
            task_id='task-2',
            agent_role='implementer',
            category='infra_issue',
            summary=self._SUMMARY_B,
            severity='critical',
        )
        assert second['status'] == 'queued', (
            f'Expected second (SUMMARY_B) to be queued (distinct 3rd token), got: {second}'
        )
        assert second['id'] != first['id'], (
            'SUMMARY_A and SUMMARY_B must produce separate records'
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "<3-token summaries: appended '[downgraded:...]' marker leaks into the "
            "dedupe key, so the downgraded record won't fold into its blocking parent. "
            "Fix requires stripping the trailing marker inside summary_dedupe_key "
            "(dedupe.py) before taking the 3-token slice — outside α2 scope."
        ),
    )
    @pytest.mark.asyncio
    @pytest.mark.parametrize('summary', ['merge', 'lost link'])
    async def test_short_summary_folds_into_blocking_parent(
        self, tmp_path: Path, summary: str,
    ):
        """KNOWN LIMITATION: <3-token summaries don't fold into equivalently-worded parents.

        summary_dedupe_key() takes the first 3 normalised tokens.  For a summary
        with fewer than 3 real tokens (e.g. 'merge' or 'lost link'), the appended
        '[downgraded:critical]' marker becomes a key token:
          'merge'      → key ('merge',)                  (original)
          'merge [downgraded:critical]' → key ('merge','downgradedcritical')  (downgraded)
          'lost link'  → key ('lost','link')              (original)
          'lost link [downgraded:critical]' → key ('lost','link','downgradedcritical')

        These keys differ, so the downgraded record is filed as a new escalation
        instead of folding into the existing blocking parent.

        To make these tests green, strip a trailing '[downgraded:...]' inside
        summary_dedupe_key (or use a dedicated severity field) — see dedupe.py.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue, dedupe_config=DedupeConfig())

        # Pre-seed a pending blocking parent with the short summary
        parent = Escalation(
            id=queue.make_id('task-100'),
            task_id='task-100',
            agent_role='implementer',
            severity='blocking',
            category='infra_issue',
            summary=summary,
        )
        queue.submit(parent)

        # File an agent critical with the same summary — should fold into parent
        result = await _blocker(
            server,
            task_id='task-999',
            agent_role='implementer',
            category='infra_issue',
            summary=summary,
            severity='critical',
        )

        assert result['status'] == 'dedup_skipped', (
            f'summary={summary!r}: expected dedup_skipped (short-summary fold), got: {result}'
        )
        assert result.get('parent_id') == parent.id, (
            f'summary={summary!r}: expected parent_id={parent.id!r}, '
            f'got parent_id={result.get("parent_id")!r}'
        )


# ---------------------------------------------------------------------------
# TestMergeStatus — merge_status MCP tool (task 1630 α3)
# ---------------------------------------------------------------------------


async def _call_merge_status(server, **kwargs) -> dict:
    """Invoke the merge_status MCP tool (async tool)."""
    tool = await server.get_tool('merge_status')
    return await tool.fn(**kwargs)


@pytest.mark.asyncio
@pytest.mark.skipif(not _ORCHESTRATOR_AVAILABLE, reason='orchestrator package not installed')
class TestMergeStatus:
    """Tests for the merge_status escalation MCP tool (α3)."""

    # ── step-5a: no key returns error ────────────────────────────────────────

    async def test_no_key_returns_error(self, tmp_path: Path) -> None:
        """Calling merge_status with no request_id/branch/task_id returns {'error': ...}."""
        esc_queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(esc_queue)

        result = await _call_merge_status(server)

        assert isinstance(result, dict), f'Expected dict, got {type(result)}'
        assert 'error' in result, f'Expected error key, got: {result}'

    # ── step-5b: standalone unknown + hint ───────────────────────────────────

    async def test_standalone_returns_unknown_with_hint(self, tmp_path: Path) -> None:
        """On a server with no event_store/harness, merge_status returns unknown + hint."""
        esc_queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(esc_queue)  # standalone: no event_store, no harness

        result = await _call_merge_status(server, request_id='mr-deadbeef')

        assert isinstance(result, dict), f'Expected dict, got {type(result)}'
        assert result.get('state') == 'unknown', f'Expected state=unknown, got: {result}'
        assert result.get('generation') == 1, f'Expected generation=1, got: {result}'
        assert result.get('request_id') == 'mr-deadbeef', (
            f'Expected request_id echoed, got: {result}'
        )
        assert 'hint' in result, f'Expected hint key in result: {result}'
        assert 'git log' in result['hint'].lower() or 'git' in result['hint'], (
            f'Expected hint to mention git, got: {result["hint"]!r}'
        )

    # ── step-7: event-store tier + state mapping ──────────────────────────────

    def _make_event_store(self, tmp_path: Path):
        return EventStore(tmp_path / 'runs.db', 'run-ms-test'), EventType

    def _emit_finalized(self, store, EventType, *, request_id, task_id, branch, state):
        store.emit(
            EventType.merge_finalized,
            task_id=task_id,
            data={'request_id': request_id, 'branch': branch, 'state': state},
        )

    @pytest.mark.parametrize('raw_state,expected_coarse', [
        ('done', 'done'),
        ('done_wip_recovery', 'done'),
        ('already_merged', 'done'),
        ('conflict', 'conflict'),
        ('blocked', 'blocked'),
        ('wip_halted', 'blocked'),
        ('wip_recovery_no_advance', 'blocked'),
        ('unmerged_state', 'blocked'),
        ('unknown_branch', 'blocked'),
        ('error', 'blocked'),
        ('abandoned', 'abandoned'),
    ])
    async def test_event_store_state_mapping(
        self, tmp_path: Path, raw_state: str, expected_coarse: str
    ) -> None:
        """Event-store tier maps raw terminal states to the public coarse vocabulary."""
        event_store, EventType = self._make_event_store(tmp_path)
        self._emit_finalized(
            event_store, EventType,
            request_id='mr-statetest',
            task_id='T-map',
            branch='branch-map',
            state=raw_state,
        )

        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = types.SimpleNamespace(_merge_worker=None)
        server = create_server(esc_queue, harness=stub_harness, event_store=event_store)

        result = await _call_merge_status(server, request_id='mr-statetest')

        assert result.get('state') == expected_coarse, (
            f'raw={raw_state!r}: expected coarse={expected_coarse!r}, got state={result.get("state")!r}'
        )
        assert result.get('outcome') == raw_state, (
            f'raw={raw_state!r}: expected outcome preserved, got outcome={result.get("outcome")!r}'
        )
        assert result.get('generation') == 1
        assert result.get('request_id') == 'mr-statetest'
        # finished_at must be an ISO-8601 string (same type as ring tier after normalisation)
        fa = result.get('finished_at')
        assert isinstance(fa, str), f'Expected ISO-8601 string for finished_at, got {type(fa)}: {fa!r}'
        assert fa.startswith('20'), f'Expected ISO-8601 date string, got: {fa!r}'

    async def test_event_store_lookup_by_branch(self, tmp_path: Path) -> None:
        """branch= lookup resolves to the most-recent row and echoes request_id."""
        event_store, EventType = self._make_event_store(tmp_path)
        # Emit two rows for the same branch; the second should win
        self._emit_finalized(
            event_store, EventType,
            request_id='mr-old', task_id='T1', branch='feat-branch', state='conflict'
        )
        self._emit_finalized(
            event_store, EventType,
            request_id='mr-new', task_id='T1', branch='feat-branch', state='done'
        )

        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = types.SimpleNamespace(_merge_worker=None)
        server = create_server(esc_queue, harness=stub_harness, event_store=event_store)

        result = await _call_merge_status(server, branch='feat-branch')

        assert result.get('state') == 'done', f'Expected done (most-recent), got: {result}'
        assert result.get('request_id') == 'mr-new', (
            f'Expected resolved request_id=mr-new, got: {result.get("request_id")!r}'
        )

    async def test_event_store_lookup_by_task_id(self, tmp_path: Path) -> None:
        """task_id= lookup resolves to the most-recent row and echoes request_id."""
        event_store, EventType = self._make_event_store(tmp_path)
        self._emit_finalized(
            event_store, EventType,
            request_id='mr-earlier', task_id='T-tid', branch='b1', state='blocked'
        )
        self._emit_finalized(
            event_store, EventType,
            request_id='mr-later', task_id='T-tid', branch='b2', state='done'
        )

        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = types.SimpleNamespace(_merge_worker=None)
        server = create_server(esc_queue, harness=stub_harness, event_store=event_store)

        result = await _call_merge_status(server, task_id='T-tid')

        assert result.get('state') == 'done', f'Expected done (most-recent), got: {result}'
        assert result.get('request_id') == 'mr-later', (
            f'Expected resolved request_id=mr-later, got: {result.get("request_id")!r}'
        )

    async def test_event_store_miss_falls_through_to_unknown(self, tmp_path: Path) -> None:
        """An id not in the event store falls through to unknown+hint."""
        event_store, EventType = self._make_event_store(tmp_path)

        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = types.SimpleNamespace(_merge_worker=None)
        server = create_server(esc_queue, harness=stub_harness, event_store=event_store)

        result = await _call_merge_status(server, request_id='mr-nothere')

        assert result.get('state') == 'unknown', f'Expected unknown, got: {result}'
        assert 'hint' in result

    # ── step-9: ring tier ─────────────────────────────────────────────────────

    async def test_ring_tier_returns_terminal_record(self, tmp_path: Path) -> None:
        """Ring tier: a request_id in the ring returns the recorded terminal state."""
        ring = TerminalOutcomeRetention()
        finished = time.time() - 5.0
        ring.record(TerminalOutcomeRecord(
            request_id='mr-aaaa1111',
            task_id='T-ring',
            branch='branch-ring',
            state='done',
            finished_at=finished,
        ))

        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = types.SimpleNamespace(_merge_worker=None, _terminal_retention=ring)
        server = create_server(esc_queue, harness=stub_harness)

        result = await _call_merge_status(server, request_id='mr-aaaa1111')

        assert result.get('state') == 'done', f'Expected done from ring, got: {result}'
        assert result.get('request_id') == 'mr-aaaa1111'
        assert result.get('generation') == 1
        assert result.get('outcome') == 'done'
        # finished_at is normalised to ISO-8601 string (same type as event-store tier)
        fa = result.get('finished_at')
        assert isinstance(fa, str), f'Expected ISO-8601 string for finished_at, got {type(fa)}: {fa!r}'
        assert fa.startswith('20'), f'Expected ISO-8601 date string, got: {fa!r}'

    async def test_ring_wins_over_event_store(self, tmp_path: Path) -> None:
        """Ring tier precedes event store: ring record wins when both have the same request_id."""
        event_store = EventStore(tmp_path / 'runs.db', 'run-ring-vs-ev')
        event_store.emit(
            EventType.merge_finalized,
            task_id='T-rv',
            data={'request_id': 'mr-ringwin', 'branch': 'b-rv', 'state': 'done'},
        )

        # Ring says 'blocked' — this should win over event store's 'done'
        ring = TerminalOutcomeRetention()
        ring.record(TerminalOutcomeRecord(
            request_id='mr-ringwin',
            task_id='T-rv',
            branch='b-rv',
            state='blocked',
        ))

        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = types.SimpleNamespace(_merge_worker=None, _terminal_retention=ring)
        server = create_server(esc_queue, harness=stub_harness, event_store=event_store)

        result = await _call_merge_status(server, request_id='mr-ringwin')

        assert result.get('state') == 'blocked', (
            f'Expected ring value (blocked) to beat event store (done), got: {result}'
        )
        assert result.get('outcome') == 'blocked'

    async def test_ring_miss_falls_through_to_event_store(self, tmp_path: Path) -> None:
        """Ring miss falls through to the event store tier."""
        event_store = EventStore(tmp_path / 'runs.db', 'run-ring-miss')
        event_store.emit(
            EventType.merge_finalized,
            task_id='T-rm',
            data={'request_id': 'mr-in-ev-only', 'branch': 'b-rm', 'state': 'conflict'},
        )

        # Ring is empty — miss should fall through to event store
        ring = TerminalOutcomeRetention()

        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = types.SimpleNamespace(_merge_worker=None, _terminal_retention=ring)
        server = create_server(esc_queue, harness=stub_harness, event_store=event_store)

        result = await _call_merge_status(server, request_id='mr-in-ev-only')

        assert result.get('state') == 'conflict', (
            f'Expected conflict from event-store fallback, got: {result}'
        )

    # ── step-11: live-snapshot tier ───────────────────────────────────────────

    def _make_orch_config(self, tmp_path: Path):
        return OrchestratorConfig(project_root=tmp_path)

    async def test_live_snapshot_state_mapping(self, tmp_path: Path) -> None:
        """Live snapshot states map to the public vocabulary."""
        loop = asyncio.get_running_loop()
        config = self._make_orch_config(tmp_path / 'repo')
        mq: asyncio.Queue = asyncio.Queue()
        git_ops_stub = types.SimpleNamespace()
        worker = SpeculativeMergeWorker(
            git_ops=git_ops_stub,  # type: ignore[reportArgumentType]
            queue=mq,
        )
        req = MergeRequest(
            task_id='T-live', branch='branch-live',
            worktree=tmp_path / 'wt',
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(),
        )
        await mq.put(req)

        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = types.SimpleNamespace(_merge_worker=worker)
        server = create_server(esc_queue, harness=stub_harness)

        result = await _call_merge_status(server, request_id=req.request_id)

        assert result.get('state') == 'queued', f'Expected queued, got: {result}'
        assert result.get('request_id') == req.request_id
        assert result.get('generation') == 1
        assert 'position' in result, f'Expected position key, got: {result}'
        assert isinstance(result.get('position'), int)
        assert 'enqueued_at' in result, f'Expected enqueued_at key, got: {result}'
        assert result.get('enqueued_at') == req.enqueued_at
        assert 'eta_seconds' in result, f'Expected eta_seconds key, got: {result}'

    async def test_live_snapshot_lookup_by_branch(self, tmp_path: Path) -> None:
        """branch= lookup resolves to the live entry's request_id."""
        loop = asyncio.get_running_loop()
        config = self._make_orch_config(tmp_path / 'repo')
        mq: asyncio.Queue = asyncio.Queue()
        git_ops_stub = types.SimpleNamespace()
        worker = SpeculativeMergeWorker(git_ops=git_ops_stub, queue=mq)  # type: ignore
        req = MergeRequest(
            task_id='T-lbranch', branch='branch-bylookup',
            worktree=tmp_path / 'wt', pre_rebased=False, task_files=None,
            module_configs=[], config=config, result=loop.create_future(),
        )
        await mq.put(req)

        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = types.SimpleNamespace(_merge_worker=worker)
        server = create_server(esc_queue, harness=stub_harness)

        result = await _call_merge_status(server, branch='branch-bylookup')

        assert result.get('state') == 'queued', f'Expected queued, got: {result}'
        assert result.get('request_id') == req.request_id, (
            f'Expected resolved request_id={req.request_id!r}, got: {result.get("request_id")!r}'
        )

    async def test_live_snapshot_lookup_by_task_id(self, tmp_path: Path) -> None:
        """task_id= lookup resolves to the live entry's request_id."""
        loop = asyncio.get_running_loop()
        config = self._make_orch_config(tmp_path / 'repo')
        mq: asyncio.Queue = asyncio.Queue()
        git_ops_stub = types.SimpleNamespace()
        worker = SpeculativeMergeWorker(git_ops=git_ops_stub, queue=mq)  # type: ignore
        req = MergeRequest(
            task_id='T-ltask', branch='branch-ltask',
            worktree=tmp_path / 'wt', pre_rebased=False, task_files=None,
            module_configs=[], config=config, result=loop.create_future(),
        )
        await mq.put(req)

        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = types.SimpleNamespace(_merge_worker=worker)
        server = create_server(esc_queue, harness=stub_harness)

        result = await _call_merge_status(server, task_id='T-ltask')

        assert result.get('state') == 'queued', f'Expected queued, got: {result}'
        assert result.get('request_id') == req.request_id, (
            f'Expected resolved request_id={req.request_id!r}, got: {result.get("request_id")!r}'
        )

    async def test_live_snapshot_beats_ring_and_event_store(self, tmp_path: Path) -> None:
        """Live snapshot tier wins over ring and event store for the same request_id."""
        loop = asyncio.get_running_loop()
        config = self._make_orch_config(tmp_path / 'repo')
        mq: asyncio.Queue = asyncio.Queue()
        git_ops_stub = types.SimpleNamespace()
        worker = SpeculativeMergeWorker(git_ops=git_ops_stub, queue=mq)  # type: ignore
        req = MergeRequest(
            task_id='T-prec', branch='branch-prec',
            worktree=tmp_path / 'wt', pre_rebased=False, task_files=None,
            module_configs=[], config=config, result=loop.create_future(),
        )
        await mq.put(req)

        # Ring says 'done'
        ring = TerminalOutcomeRetention()
        ring.record(TerminalOutcomeRecord(
            request_id=req.request_id,
            task_id=req.task_id,
            branch=req.branch,
            state='done',
        ))

        # Event store also says 'done'
        event_store = EventStore(tmp_path / 'runs.db', 'run-prec')
        event_store.emit(
            EventType.merge_finalized,
            task_id=req.task_id,
            data={'request_id': req.request_id, 'branch': req.branch, 'state': 'done'},
        )

        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = types.SimpleNamespace(_merge_worker=worker, _terminal_retention=ring)
        server = create_server(esc_queue, harness=stub_harness, event_store=event_store)

        result = await _call_merge_status(server, request_id=req.request_id)

        # Live snapshot is 'queued'; ring+event store have 'done'
        # Live must win
        assert result.get('state') == 'queued', (
            f'Expected live state (queued) to beat ring/ev (done), got: {result}'
        )
        assert 'position' in result, 'Live entry should carry position'

    @pytest.mark.parametrize('verify_phase,expected', [
        ('merging', 'verifying'),
        ('awaiting_verify', 'verifying'),
        ('verifying', 'verifying'),
        ('gate_reverify', 'gate'),
        ('finalizing', 'finalizing'),
        ('queued', 'queued'),
    ])
    async def test_live_snapshot_phase_mapping(
        self, tmp_path: Path, verify_phase: str, expected: str
    ) -> None:
        """Live snapshot state mapping covers all documented phase values."""
        loop = asyncio.get_running_loop()
        config = self._make_orch_config(tmp_path / 'repo')
        mq: asyncio.Queue = asyncio.Queue()
        git_ops_stub = types.SimpleNamespace()
        worker = SpeculativeMergeWorker(git_ops=git_ops_stub, queue=mq)  # type: ignore

        req = MergeRequest(
            task_id='T-phase', branch='branch-phase',
            worktree=tmp_path / 'wt', pre_rebased=False, task_files=None,
            module_configs=[], config=config, result=loop.create_future(),
        )

        if verify_phase == 'queued':
            await mq.put(req)
        else:
            # Put req through the verify item path
            merge_wt = tmp_path / 'merge-wt'
            merge_wt.mkdir()
            item = SpeculativeItem(
                request=req,
                merge_result=None, merge_wt=merge_wt,
                base_sha='base', speculative=False, skip_verify=False,
            )
            if verify_phase in ('merging',):
                worker._inflight_req = req
            elif verify_phase == 'awaiting_verify':
                await worker._verifier_queue.put(item)
            else:
                worker._verify_item = item
                worker._verify_phase = verify_phase

        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = types.SimpleNamespace(_merge_worker=worker)
        server = create_server(esc_queue, harness=stub_harness)

        result = await _call_merge_status(server, request_id=req.request_id)

        assert result.get('state') == expected, (
            f'verify_phase={verify_phase!r}: expected {expected!r}, got {result.get("state")!r}'
        )

    # ── Tier-1 degradation: snapshot() failure falls through to durable tiers ─

    async def test_snapshot_failure_degrades_to_durable_tier(self, tmp_path: Path) -> None:
        """Tier-1 fire-safe wrapper: a snapshot() exception falls through to ring/event-store.

        This covers the most safety-critical branch: merge_status must return a
        durable-tier result (not propagate the exception) when the live worker
        is present but snapshot() itself raises.
        """
        # Populate the event store so the durable tier has a result to serve.
        event_store = EventStore(tmp_path / 'runs.db', 'run-snap-fail')
        event_store.emit(
            EventType.merge_finalized,
            task_id='T-snapfail',
            data={'request_id': 'mr-snapfail', 'branch': 'b-snapfail', 'state': 'done'},
        )

        # Stub worker whose snapshot() always raises — simulates a transient
        # introspection failure (e.g. queue internals in a bad state).
        broken_worker = types.SimpleNamespace(
            snapshot=lambda: (_ for _ in ()).throw(RuntimeError('simulated snapshot failure'))
        )

        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = types.SimpleNamespace(
            _merge_worker=broken_worker,
            _terminal_retention=None,
        )
        server = create_server(esc_queue, harness=stub_harness, event_store=event_store)

        # Should NOT raise; should fall through to the event-store tier.
        result = await _call_merge_status(server, request_id='mr-snapfail')

        assert result.get('state') == 'done', (
            f'Expected durable-tier result after snapshot() failure, got: {result}'
        )
        assert result.get('request_id') == 'mr-snapfail'
        assert result.get('outcome') == 'done'
        assert 'finished_at' in result


# ---------------------------------------------------------------------------
# TestCreateServerStartupSweep: startup_sweep wiring in create_server
# ---------------------------------------------------------------------------


class TestCreateServerStartupSweep:
    """create_server runs run_startup_sweep on construction when startup_sweep=True."""

    def test_startup_sweep_true_archives_orphan_at_construction(
        self, tmp_path: Path, caplog
    ):
        """(a) create_server() with startup_sweep=True (default) archives a resolved orphan."""
        import logging

        from escalation.models import Escalation

        queue = EscalationQueue(tmp_path / 'esc')
        # Seed a resolved orphan DIRECTLY — NOT via queue.submit_resolved (which
        # would auto-archive the file itself via _archive_resolved, defeating this test).
        esc = Escalation(
            id='esc-9-9',
            task_id='9',
            agent_role='test',
            severity='info',
            category='cleanup_needed',
            summary='orphan',
            status='resolved',
            resolved_at='2026-05-20T10:00:00+00:00',
        )
        (queue.queue_dir / 'esc-9-9.json').write_text(esc.to_json())

        with caplog.at_level(logging.INFO, logger='escalation.sweep'):
            create_server(queue)  # default startup_sweep=True

        # Orphan was archived
        archive_path = queue.queue_dir / 'archive' / '2026-05-20' / 'esc-9-9.json'
        assert archive_path.exists(), (
            f'Expected orphan archived at {archive_path}; '
            f'still in root: {(queue.queue_dir / "esc-9-9.json").exists()}'
        )
        assert not (queue.queue_dir / 'esc-9-9.json').exists()

        # INFO report line logged on escalation.sweep logger
        assert any(
            r.name == 'escalation.sweep' and r.levelno == logging.INFO
            for r in caplog.records
        ), f'Expected INFO sweep report; got: {[r.getMessage() for r in caplog.records]}'

    def test_startup_sweep_false_leaves_orphan_untouched(self, tmp_path: Path):
        """(b) create_server(startup_sweep=False) leaves a pre-seeded orphan in root."""
        from escalation.models import Escalation

        queue = EscalationQueue(tmp_path / 'esc')
        esc = Escalation(
            id='esc-9-9',
            task_id='9',
            agent_role='test',
            severity='info',
            category='cleanup_needed',
            summary='orphan',
            status='resolved',
            resolved_at='2026-05-20T10:00:00+00:00',
        )
        (queue.queue_dir / 'esc-9-9.json').write_text(esc.to_json())

        create_server(queue, startup_sweep=False)

        # File still in root — startup sweep was skipped
        assert (queue.queue_dir / 'esc-9-9.json').exists(), (
            'Orphan was archived even with startup_sweep=False'
        )
        assert not (queue.queue_dir / 'archive').exists()
