"""Tests for L2 severity-gated born-at-L2 path and get_pending_escalations level filter.

Uses the async FastMCP unit-test pattern from test_server_chokepoint.py:
    tool = await server.get_tool(name)
    result = await tool.fn(...)

and tmp_path isolation with EscalationQueue.
"""

from __future__ import annotations

import asyncio
import json
import time
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from escalation.dedupe import DedupeConfig, summary_dedupe_key
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from escalation.server import _COMPACT_ESCALATION_FIELDS, create_server

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
        QueuedBranch,
        RealMergeItem,
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
    QueuedBranch: Any = None
    RealMergeItem: Any = None
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


async def _stamp_triage(server, **kwargs: Any) -> dict[str, Any]:
    tool = await server.get_tool('stamp_triage')
    # stamp_triage is a sync def, so tool.fn(...) returns directly
    return tool.fn(**kwargs)


async def _get_task_escalations(server, **kwargs: Any) -> list[dict[str, Any]]:
    tool = await server.get_tool('get_task_escalations')
    # get_task_escalations is a sync def, so tool.fn(...) returns directly
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
        'triaged_at', 'triaged_by', 'triage_note', 'updated_at',
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
# TestGetTaskEscalations: archive-inclusive task-scoped lookup (task 3023)
# ---------------------------------------------------------------------------


class TestGetTaskEscalations:
    """get_task_escalations(task_id=...) is ARCHIVE-INCLUSIVE by default.

    The recurring recon false positive this tool exists to disconfirm: an
    auditor probes ``get_pending_escalations(task_id=...)``, gets ``[]``
    because the human-resolved born-at-L2 gate record was archived to
    ``data/escalations/archive/<date>/``, and concludes the record was never
    written.  ``get_task_escalations`` sees the archived record, so an empty
    result THERE (and only there) is evidence of absence.
    """

    def _seed(
        self,
        queue: EscalationQueue,
        esc_id: str,
        *,
        task_id: str = '42',
        level: int = 0,
        agent_role: str = 'implementer',
    ) -> Escalation:
        """Submit a pending escalation with an explicit id."""
        esc = Escalation(
            id=esc_id,
            task_id=task_id,
            agent_role=agent_role,
            severity='blocking',
            category='task_failure',
            summary=f'{esc_id} test escalation',
            level=level,
        )
        queue.submit(esc)
        return esc

    def _mixed_queue(self, tmp_path: Path) -> EscalationQueue:
        """One resolved+archived record and one still-pending record for task '42'.

        ``esc-42-1`` mirrors a human-resolved born-at-L2 deterministic gate:
        submitted at level 2 by the deterministic runner, then resolved (which
        moves the file out of the queue root into ``archive/<date>/``).
        ``esc-42-2`` stays in the queue root.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        self._seed(queue, 'esc-42-1', level=2, agent_role='deterministic')
        queue.resolve('esc-42-1', 'Human reviewed the gate')
        self._seed(queue, 'esc-42-2', level=0, agent_role='implementer')
        return queue

    # -- (a) THE REGRESSION -------------------------------------------------

    @pytest.mark.asyncio
    async def test_pending_probe_misses_archived_record_but_task_scoped_finds_it(
        self, tmp_path: Path,
    ):
        """get_pending_escalations misses the archived record; get_task_escalations doesn't."""
        queue = self._mixed_queue(tmp_path)
        server = create_server(queue)

        pending = await _get_pending(server, task_id='42')
        assert {e['id'] for e in pending} == {'esc-42-2'}, (
            f'get_pending_escalations must stay root-only, got {pending}'
        )

        result = await _get_task_escalations(server, task_id='42')

        assert {e['id'] for e in result} == {'esc-42-1', 'esc-42-2'}, (
            f'Expected both the archived and the pending record, got {result}'
        )
        archived = next(e for e in result if e['id'] == 'esc-42-1')
        assert archived['status'] == 'resolved'

    # -- (b) status filter --------------------------------------------------

    @pytest.mark.asyncio
    async def test_status_resolved_returns_only_the_archived_record(self, tmp_path: Path):
        """status='resolved' → only the archived record."""
        queue = self._mixed_queue(tmp_path)
        server = create_server(queue)

        result = await _get_task_escalations(server, task_id='42', status='resolved')

        assert {e['id'] for e in result} == {'esc-42-1'}, f'got {result}'

    @pytest.mark.asyncio
    async def test_status_pending_returns_only_the_root_record(self, tmp_path: Path):
        """status='pending' → get_by_task's root-only fast path."""
        queue = self._mixed_queue(tmp_path)
        server = create_server(queue)

        result = await _get_task_escalations(server, task_id='42', status='pending')

        assert {e['id'] for e in result} == {'esc-42-2'}, f'got {result}'

    # -- (c) level / agent_role passthrough ---------------------------------

    @pytest.mark.asyncio
    async def test_level_filter_is_passed_through(self, tmp_path: Path):
        """level=2 narrows to the archived L2 gate record."""
        queue = self._mixed_queue(tmp_path)
        server = create_server(queue)

        result = await _get_task_escalations(server, task_id='42', level=2)

        assert {e['id'] for e in result} == {'esc-42-1'}, f'got {result}'
        assert result[0]['level'] == 2

    @pytest.mark.asyncio
    async def test_agent_role_filter_is_passed_through(self, tmp_path: Path):
        """agent_role='deterministic' narrows to the record filed by that role."""
        queue = self._mixed_queue(tmp_path)
        server = create_server(queue)

        result = await _get_task_escalations(
            server, task_id='42', agent_role='deterministic',
        )

        assert {e['id'] for e in result} == {'esc-42-1'}, f'got {result}'
        assert result[0]['agent_role'] == 'deterministic'

    # -- (d) compact projection --------------------------------------------

    @pytest.mark.asyncio
    async def test_compact_projects_to_the_shared_compact_fields(self, tmp_path: Path):
        """compact=True projects each dict to exactly _COMPACT_ESCALATION_FIELDS."""
        queue = self._mixed_queue(tmp_path)
        server = create_server(queue)

        result = await _get_task_escalations(server, task_id='42', compact=True)

        assert len(result) == 2, f'Expected both records, got {result}'
        for row in result:
            assert set(row.keys()) == set(_COMPACT_ESCALATION_FIELDS), (
                f'compact row keys {sorted(row.keys())} != '
                f'{sorted(_COMPACT_ESCALATION_FIELDS)}'
            )

    # -- (e) unknown task ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_unknown_task_id_returns_empty_list(self, tmp_path: Path):
        """An unknown task id returns [] — the ONLY sound evidence of absence."""
        queue = self._mixed_queue(tmp_path)
        server = create_server(queue)

        result = await _get_task_escalations(server, task_id='no-such-task')

        assert result == [], f'Expected [], got {result}'


# ---------------------------------------------------------------------------
# TestStampTriageTool: stamp_triage MCP tool (triage-ack annotation, ungated)
# ---------------------------------------------------------------------------


class TestStampTriageTool:
    """stamp_triage MCP tool stamps a triage-ack annotation (in-process: headers == {}).

    In-process tool.fn() calls see empty headers, so triaged_by comes from the
    passed arg here — identity-header attribution is covered separately by the
    live-HTTP acceptance test (test_capability_guard_http.py).
    """

    def _seed_pending_l2(self, queue: EscalationQueue, esc_id: str = 'esc-t1-0001') -> Escalation:
        esc = Escalation(
            id=esc_id,
            task_id='t-1',
            agent_role='escalation-watcher-auto',
            severity='blocking',
            category='design_concern',
            summary='pending L2 for stamp_triage tool test',
            level=2,
        )
        queue.submit(esc)
        return esc

    @pytest.mark.asyncio
    async def test_stamps_pending_l2_full_dict(self, tmp_path: Path):
        """(a) Stamping returns a full dict; triaged_at/triaged_by/triage_note set, status/level unchanged."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending_l2(queue)

        result = await _stamp_triage(
            server, escalation_id=esc.id,
            triaged_by='orchestrator-escalation-watcher-auto',
            triage_note='task-604 status==done | probe: get_task 604 -> status=done',
        )

        assert 'error' not in result, f"Unexpected error: {result}"
        assert result['triaged_at'] is not None
        assert result['triaged_by'] == 'orchestrator-escalation-watcher-auto'
        assert result['triage_note'] == 'task-604 status==done | probe: get_task 604 -> status=done'
        assert result['status'] == 'pending'
        assert result['level'] == 2

    @pytest.mark.asyncio
    async def test_unknown_id_returns_error(self, tmp_path: Path):
        """(b) An unknown escalation id returns an {'error': ...} dict."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _stamp_triage(
            server, escalation_id='esc-does-not-exist',
            triaged_by='watcher', triage_note='note',
        )

        assert 'error' in result, f"Expected error dict, got: {result}"

    @pytest.mark.asyncio
    async def test_resolved_archived_id_returns_error(self, tmp_path: Path):
        """(c) A resolved/archived id returns an {'error': ...} dict (queue method returned None)."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending_l2(queue)
        queue.resolve(esc.id, 'Fixed')

        result = await _stamp_triage(
            server, escalation_id=esc.id,
            triaged_by='watcher', triage_note='note',
        )

        assert 'error' in result, f"Expected error dict, got: {result}"


# ---------------------------------------------------------------------------
# TestGetPendingSurfacesTriageFields: triage fields surfaced in full + compact
# ---------------------------------------------------------------------------


class TestGetPendingSurfacesTriageFields:
    """get_pending_escalations surfaces triaged_at/triaged_by/triage_note/updated_at.

    Full mode already surfaces them via to_dict() once the model fields exist
    (models.py). Compact mode requires _COMPACT_ESCALATION_FIELDS to be
    widened to include them — that widening is the RED driver here.
    """

    def _seed_and_stamp(self, queue: EscalationQueue, esc_id: str = 'esc-t1-0001') -> Escalation:
        esc = Escalation(
            id=esc_id,
            task_id='t-1',
            agent_role='escalation-watcher-auto',
            severity='blocking',
            category='design_concern',
            summary='pending L2 for triage-field surfacing test',
            level=2,
        )
        queue.submit(esc)
        queue.stamp_triage(
            esc_id, triaged_by='orchestrator-escalation-watcher-auto',
            triage_note='task-604 status==done | probe: get_task 604 -> status=done',
        )
        # A real L2 member append bumps updated_at — exercise that path too so
        # both timestamp fields have faithful non-default values to assert on.
        queue.add_members_to_l2(esc_id, ['esc-m-0001'])
        return esc

    @pytest.mark.asyncio
    async def test_full_mode_surfaces_triage_fields(self, tmp_path: Path):
        """(a) FULL mode (compact omitted) returns the four triage fields faithfully."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        self._seed_and_stamp(queue)

        result = await _get_pending(server, level=2)

        assert len(result) == 1, f"Expected 1 result, got {len(result)}: {result}"
        row = result[0]
        assert row['triaged_at'] is not None
        assert row['triaged_by'] == 'orchestrator-escalation-watcher-auto'
        assert row['triage_note'] == 'task-604 status==done | probe: get_task 604 -> status=done'
        assert row['updated_at'] is not None

    @pytest.mark.asyncio
    async def test_compact_mode_includes_triage_fields(self, tmp_path: Path):
        """(b) COMPACT mode includes triaged_at/triaged_by/triage_note/updated_at per row."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        self._seed_and_stamp(queue)

        result = await _get_pending(server, level=2, compact=True)

        assert len(result) == 1, f"Expected 1 result, got {len(result)}: {result}"
        row = result[0]
        for key in ('triaged_at', 'triaged_by', 'triage_note', 'updated_at'):
            assert key in row, f"Missing {key!r} in compact projection: {row}"
        assert row['triaged_at'] is not None
        assert row['triaged_by'] == 'orchestrator-escalation-watcher-auto'
        assert row['triage_note'] == 'task-604 status==done | probe: get_task 604 -> status=done'
        assert row['updated_at'] is not None


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
    async def test_terminal_task_auto_resolve_stamps_resolution_class_benign(self, tmp_path: Path):
        """The task-already-terminal auto-resolve chokepoint (submit_resolved,
        resolved_by='escalation-mcp-pre-submit-check') stamps resolution_class=
        'benign' explicitly — this resolver isn't reaper-sweep tier, so leaving
        it unstamped would have the effective_benign() proxy misread the closed
        record as 'actionable'."""
        async def _lookup(task_id: str) -> str:
            return 'done'

        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue, task_status_lookup=_lookup)

        result = await _info(
            server,
            task_id='task-999',
            agent_role='implementer',
            category='infra_issue',
            summary='infra connection timeout',
        )

        assert result['status'] == 'resolved'
        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.resolution_class == 'benign', (
            f"Expected resolution_class='benign', got: {esc.resolution_class!r}"
        )

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

    # --- (d) park → kept OPEN at L2; abandon / close_only → dismissed ---

    @pytest.mark.asyncio
    async def test_action_park_keeps_open_l2(self, tmp_path: Path):
        """action='park' keeps the escalation open (status='pending') at level=2 with resolution_action='park'.

        Version-a: park does NOT dismiss; the open L2 is the mechanism for sweep-quiescence.
        Fails: park currently calls queue.resolve(dismiss=True) → status='dismissed'.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue)

        result = await _resolve_issue(
            server, escalation_id=esc.id, resolution='parked', action='park',
        )

        assert result.get('status') == 'pending', f"Expected pending (kept open); got: {result}"
        assert result.get('level') == 2, f"Expected level=2 (promoted); got: {result}"
        assert result.get('resolution_action') == 'park', f"Expected resolution_action='park'; got: {result}"

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
    #
    # Folded into TestResolveIssueTableBGate.
    # test_bogus_action_returns_illegal_transition_code_record_unchanged below,
    # which asserts everything this case covered (error dict + record stays
    # pending) plus the typed code='illegal_transition' and the
    # resolution_action stamp guard. Kept as a single test to avoid two
    # near-identical bogus-action tests drifting apart.


# ---------------------------------------------------------------------------
# TestResolveIssueTableBGate: Table B (escalation.action_effects) is the SINGLE
# legality authority for resolve_issue (plans/task-status-authority-prd.md
# contract C5 / decisions D1, D2).
# ---------------------------------------------------------------------------


class TestResolveIssueTableBGate:
    """resolve_issue consults Table B (escalation.action_effects) before any mutation."""

    def _seed_pending(
        self,
        queue: EscalationQueue,
        esc_id: str = 'esc-tb-0001',
        category: str = 'scope_violation',
    ) -> Escalation:
        esc = Escalation(
            id=esc_id,
            task_id='t-table-b',
            agent_role='implementer',
            severity='blocking',
            category=category,
            summary='Table B gate test escalation',
        )
        queue.submit(esc)
        return esc

    @pytest.mark.asyncio
    async def test_bogus_action_returns_illegal_transition_code_record_unchanged(
        self, tmp_path: Path
    ):
        """action='bogus' returns a typed code='illegal_transition' and makes no record change.

        Fails today: resolve_issue's bare `action not in RESOLVE_ACTIONS` check
        returns {'error': ...} with no 'code' key.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue)

        result = await _resolve_issue(
            server, escalation_id=esc.id, resolution='fixed', action='bogus',
        )

        assert 'error' in result, f"Expected error dict for invalid action; got: {result}"
        assert result.get('code') == 'illegal_transition', (
            f"Expected code='illegal_transition'; got: {result}"
        )
        record = queue.get(esc.id)
        assert record is not None
        assert record.status == 'pending', f"Record must stay pending; got {record.status!r}"
        assert record.resolution_action is None, (
            f"Record must not be stamped; got {record.resolution_action!r}"
        )

    @pytest.mark.asyncio
    async def test_novel_category_does_not_narrow_legality(self, tmp_path: Path):
        """A category absent from server.CATEGORIES (e.g. 'milestone_gate') still resolves.

        Proves the Table B gate does not narrow legality by category — the G6
        archive verification found live resolutions across a wide, open
        category vocabulary that server.CATEGORIES (inert/unvalidated) does
        not even fully list.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue, esc_id='esc-tb-0002', category='milestone_gate')

        result = await _resolve_issue(
            server, escalation_id=esc.id, resolution='fixed', action='resume',
        )

        assert result.get('status') == 'resolved', f"Expected resolved; got: {result}"


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
        """action='park' → queue.get(id) has status='pending', level=2, resolution_action='park'.

        Version-a: the parked record stays live (not archived), resolution text is persisted,
        and the escalation remains open at L2.
        Fails: park currently resolves with dismiss=True → status='dismissed'.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue)

        await _resolve_issue(
            server, escalation_id=esc.id, resolution='parked for human review', action='park',
        )

        record = queue.get(esc.id)
        assert record is not None
        assert record.status == 'pending', f"Expected pending (kept open); got {record.status!r}"
        assert record.level == 2, f"Expected level=2 (promoted); got {record.level!r}"
        assert record.resolution_action == 'park', (
            f"Expected resolution_action='park'; got {record.resolution_action!r}"
        )
        assert record.resolution == 'parked for human review', (
            f"Expected resolution text persisted; got {record.resolution!r}"
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


class TestResolveIssueResolutionClass:
    """resolve_issue accepts an optional resolution_class, validated before any record mutation."""

    def _seed_pending(self, queue: EscalationQueue, esc_id: str = 'esc-rc-0001') -> Escalation:
        esc = Escalation(
            id=esc_id,
            task_id='t-resolution-class',
            agent_role='implementer',
            severity='blocking',
            category='scope_violation',
            summary='resolution_class test escalation',
        )
        queue.submit(esc)
        return esc

    @pytest.mark.asyncio
    async def test_actionable_round_trips_to_queue_get_and_archive(self, tmp_path: Path):
        """(a) resolution_class='actionable' -> queue.get() and archived JSON both carry it (boundary row 1)."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue)

        await _resolve_issue(
            server, escalation_id=esc.id, resolution='done', action='close_only',
            resolution_class='actionable',
        )

        record = queue.get(esc.id)
        assert record is not None
        assert record.resolution_class == 'actionable', (
            f"Expected resolution_class='actionable'; got {record.resolution_class!r}"
        )

        archived_files = list((queue.queue_dir / 'archive').rglob(f'{esc.id}.json'))
        assert len(archived_files) == 1
        data = json.loads(archived_files[0].read_text())
        assert data['resolution_class'] == 'actionable', (
            f"Expected archived JSON resolution_class='actionable'; got {data.get('resolution_class')!r}"
        )

    @pytest.mark.asyncio
    async def test_no_resolution_class_leaves_none(self, tmp_path: Path):
        """(b) resolve_issue with no resolution_class -> record has resolution_class None (boundary row 3)."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue)

        await _resolve_issue(
            server, escalation_id=esc.id, resolution='fixed', action='resume',
        )

        record = queue.get(esc.id)
        assert record is not None
        assert record.resolution_class is None

    @pytest.mark.asyncio
    async def test_invalid_class_returns_error_code_record_unchanged(self, tmp_path: Path):
        """(c) resolution_class='meh' -> error naming benign/actionable, code='invalid_resolution_class', record still pending (boundary row 4)."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue)

        result = await _resolve_issue(
            server, escalation_id=esc.id, resolution='fixed', action='resume',
            resolution_class='meh',
        )

        assert 'error' in result, f"Expected error dict for invalid resolution_class; got: {result}"
        assert 'benign' in result['error'], f"Error must name 'benign'; got: {result}"
        assert 'actionable' in result['error'], f"Error must name 'actionable'; got: {result}"
        assert result.get('code') == 'invalid_resolution_class', (
            f"Expected code='invalid_resolution_class'; got: {result}"
        )

        record = queue.get(esc.id)
        assert record is not None
        assert record.status == 'pending', f"Record must stay pending; got {record.status!r}"
        assert record.resolution_class is None, (
            f"Record must remain unstamped; got {record.resolution_class!r}"
        )


class TestResolveIssueGrantedFiles:
    """resolve_issue accepts an optional granted_files scope-expansion grant (task 2505)."""

    def _seed_pending(self, queue: EscalationQueue, esc_id: str = 'esc-gf-0001') -> Escalation:
        esc = Escalation(
            id=esc_id,
            task_id='t-granted-files',
            agent_role='implementer',
            severity='blocking',
            category='scope_violation',
            summary='granted_files test escalation',
        )
        queue.submit(esc)
        return esc

    @pytest.mark.asyncio
    async def test_granted_files_round_trips_to_queue_get(self, tmp_path: Path):
        """action='resume', granted_files=[...] -> queue.get(id).granted_files carries it."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue)

        await _resolve_issue(
            server, escalation_id=esc.id, resolution='granted', action='resume',
            granted_files=['crate/Cargo.toml'],
        )

        record = queue.get(esc.id)
        assert record is not None
        assert record.granted_files == ['crate/Cargo.toml'], (
            f"Expected granted_files=['crate/Cargo.toml']; got {record.granted_files!r}"
        )

    @pytest.mark.asyncio
    async def test_no_granted_files_leaves_it_empty(self, tmp_path: Path):
        """resolve_issue with no granted_files argument -> record.granted_files == []."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        esc = self._seed_pending(queue)

        await _resolve_issue(
            server, escalation_id=esc.id, resolution='fixed', action='resume',
        )

        record = queue.get(esc.id)
        assert record is not None
        assert record.granted_files == []


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

    async def test_a_worktree_attach_marks_itself_unpollable(
        self, tmp_path: Path, monkeypatch,
    ):
        """task 3148: a disk-scan attach must disclose that it is NOT pollable.

        The incident's misleading shape: merge_request returned a
        documented-durable `attached` whose `request_id` was the SUBMITTING
        request's own never-enqueued id, because the disk-scan arm registers no
        retention alias, and the `_waiters` registration sits AFTER the
        `if dispatch.in_flight: return base` early-return — so no waiter either.
        Every merge_status resolution tier misses and the id resolves 'unknown';
        /unblock and unblock-low-risk submit-then-poll on it and hang.
        """
        import asyncio

        from orchestrator import merge_queue as _mq_mod  # type: ignore[reportMissingImports]
        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            MergeDispatchResult,
        )

        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = self._make_orch_config(tmp_path / 'repo')
        registry = self._make_registry()

        # The disk-scan arm needs a real git repo + worktree to reach, so return
        # its exact result shape directly: no inflight_task_id, no
        # inflight_request_id, source='worktree'.
        async def _fake_coalesce(*args, **kwargs):
            return MergeDispatchResult(
                dispatched=False, in_flight=True, branch='X', source='worktree',
            )

        monkeypatch.setattr(
            _mq_mod, 'coalesce_or_enqueue_merge_request', _fake_coalesce,
        )

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
        )
        result = await asyncio.wait_for(
            _call_merge_request(
                server, task_id='X', branch='X',
                worktree=str(tmp_path / 'wt'), description='',
            ),
            timeout=2.0,
        )

        assert result.get('status') == 'attached', f'got: {result}'
        assert result.get('source') == 'worktree', f'got: {result}'
        assert result.get('inflight_request_id') is None
        assert result.get('inflight_task_id') is None
        assert result.get('pollable') is False, (
            'a worktree attach carries no alias and no waiter, so its '
            f'request_id is not a poll handle: {result}'
        )
        # NEITHER handle is present, so the remedy is named explicitly rather
        # than left for the caller to infer: poll by branch / get_merge_queue.
        assert result.get('poll_by') == 'branch', f'got: {result}'

    async def test_b_registry_attach_is_pollable(self, tmp_path: Path):
        """A registry attach DOES carry a pollable handle (alias + task_id)."""
        import asyncio

        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = self._make_orch_config(tmp_path / 'repo')
        registry = self._make_registry()

        never_future: asyncio.Future = asyncio.get_running_loop().create_future()
        # NB: request_id must be passed EXPLICITLY — the sibling
        # test_in_flight_branch_returns_immediately omits it (the legacy case,
        # covered by test_c below).
        assert registry.acquire('X', '5566', never_future, request_id='mr-primary')

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
        )
        try:
            result = await asyncio.wait_for(
                _call_merge_request(
                    server, task_id='X', branch='X',
                    worktree=str(tmp_path / 'wt'), description='',
                ),
                timeout=2.0,
            )

            assert result.get('status') == 'attached', f'got: {result}'
            assert result.get('source') == 'registry', f'got: {result}'
            assert result.get('inflight_task_id') == '5566'
            assert result.get('inflight_request_id') == 'mr-primary'
            assert result.get('pollable') is True, f'got: {result}'
            # The returned request_id IS the in-flight entry's id (D8 override),
            # so it is the handle to poll.
            assert result.get('poll_by') == 'request_id', f'got: {result}'
            assert result.get('request_id') == 'mr-primary', f'got: {result}'
        finally:
            never_future.cancel()

    async def test_c_legacy_registry_entry_polls_by_task_id(
        self, tmp_path: Path,
    ):
        """A legacy entry (no request_id) is routed to the handle it DOES have.

        `_nonblocking_state_response` falls back to the SUBMITTING call's id when
        `req_id_override` is None, so the returned `request_id` is not a real
        poll handle even though the attach came from the registry.  But the
        entry still carries a task_id, and `merge_status` accepts task_id (D10),
        so this attach IS pollable — just not by `request_id`.  Deriving the
        verdict from the handles actually present (rather than from `source`, or
        from `inflight_request_id` alone) is what makes this case come out
        right: `poll_by='task_id'` names the usable handle instead of writing
        the attach off as unpollable and sending the caller to the branch tier.
        """
        import asyncio

        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = self._make_orch_config(tmp_path / 'repo')
        registry = self._make_registry()

        never_future: asyncio.Future = asyncio.get_running_loop().create_future()
        assert registry.acquire('X', 'existing-task', never_future)  # NO request_id

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
        )
        try:
            result = await asyncio.wait_for(
                _call_merge_request(
                    server, task_id='X', branch='X',
                    worktree=str(tmp_path / 'wt'), description='',
                ),
                timeout=2.0,
            )

            assert result.get('status') == 'attached', f'got: {result}'
            assert result.get('source') == 'registry', f'got: {result}'
            assert result.get('inflight_request_id') is None
            assert result.get('inflight_task_id') == 'existing-task'
            assert result.get('poll_by') == 'task_id', f'got: {result}'
            assert result.get('pollable') is True, (
                'a task_id IS a merge_status handle (D10) — reporting this '
                f'attach as unpollable would send the caller to git: {result}'
            )
            # ...and the returned request_id is explicitly NOT the handle here.
            assert result.get('request_id') != result.get('inflight_request_id')
        finally:
            never_future.cancel()

    async def test_d_dispatched_unaffected(self, tmp_path: Path):
        """A freshly-dispatched submission still returns the `queued` shape.

        Pre-existing keys asserted as a SUBSET, not an exact-dict equality, so
        this does not become brittle against the very additive convention the
        three new `attached` keys follow.
        """
        import asyncio

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
        result = await asyncio.wait_for(
            _call_merge_request(
                server, task_id='D', branch='D',
                worktree=str(tmp_path / 'wt'), description='',
            ),
            timeout=2.0,
        )

        assert result.get('status') == 'queued', f'got: {result}'
        assert set(result) >= {
            'request_id', 'generation', 'position', 'queue_depth', 'snapshot_tip',
        }, f'pre-existing queued keys must be unchanged: {sorted(result)}'

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

    async def test_retry_failed_only_reaches_worker(self, tmp_path: Path):
        """merge_request(retry_failed_only=True) threads the flag onto the
        MergeRequest the worker dequeues (PRD task D1).

        The background worker captures the dequeued MergeRequest so the test
        can assert ``req.retry_failed_only is True`` — i.e. the caller-vouched
        bool reaches the worker's retry path (``req`` is the same object
        ``_run_post_merge_verify`` reads).
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

        captured: dict[str, Any] = {}

        async def _worker():
            req = await mq.get()
            captured['req'] = req
            req.result.set_result(
                MergeOutcome('done', merge_sha='abc123', reason='test merge')
            )

        worker_task = asyncio.create_task(_worker())

        result = await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='RFO',
                branch='RFO',
                worktree=str(tmp_path / 'wt'),
                description='',
                wait_secs=100,
                retry_failed_only=True,
            ),
            timeout=5.0,
        )

        await worker_task

        assert result.get('status') == 'done', f'Expected status done, got: {result}'
        assert 'req' in captured, 'Worker did not dequeue a MergeRequest'
        assert captured['req'].retry_failed_only is True, (
            'retry_failed_only=True must be threaded onto the dequeued MergeRequest'
        )

    async def test_retry_failed_only_defaults_false(self, tmp_path: Path):
        """Omitting retry_failed_only leaves the dequeued MergeRequest's flag
        False — the default is a strict no-op (unchanged behavior)."""
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

        captured: dict[str, Any] = {}

        async def _worker():
            req = await mq.get()
            captured['req'] = req
            req.result.set_result(
                MergeOutcome('done', merge_sha='abc123', reason='test merge')
            )

        worker_task = asyncio.create_task(_worker())

        await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='RFO2',
                branch='RFO2',
                worktree=str(tmp_path / 'wt'),
                description='',
                wait_secs=100,
            ),
            timeout=5.0,
        )

        await worker_task

        assert 'req' in captured, 'Worker did not dequeue a MergeRequest'
        assert captured['req'].retry_failed_only is False, (
            'retry_failed_only must default to False on the dequeued MergeRequest'
        )

    async def test_inflight_with_workflow_path_registration(self, tmp_path: Path):
        """Acceptance #1: a branch registered via the workflow-path helper is
        visible to the MCP merge_request coalesce gate.

        Simulates the dedup blind-spot fix: the workflow enqueues via
        register_and_enqueue_merge_request (which registers in the shared
        registry), then a subsequent MCP merge_request call for the same
        branch returns status='attached' (coalesced from registry, existing
        entry's id) with inflight_task_id='workflow-task' and NO duplicate
        queue entry.
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
            branch=QueuedBranch.parse('B', 'task/'),
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

    async def test_already_merged_fast_path_dedups_prefixed_branch(
        self, tmp_path: Path
    ) -> None:
        """merge_request already_merged fast-path resolves the SAME ref for a
        bare and a pre-prefixed branch — regression guard for the
        double-prefix bug (PRD I4 fast-path used to build the full ref via a
        raw f-string concat with no startswith guard, so an already-prefixed
        branch like 'task/123' became 'task/task/123' and missed the
        git_ops lookup, falling through to a normal enqueue instead of the
        already_merged short-circuit).
        """
        from unittest.mock import AsyncMock

        tip = 't' * 40
        resolve_branch_sha = AsyncMock(
            side_effect=lambda b: tip if b == 'task/123' else None
        )
        is_ancestor = AsyncMock(return_value=True)
        stub_git = types.SimpleNamespace(
            resolve_branch_sha=resolve_branch_sha,
            is_ancestor=is_ancestor,
            # Only exercised today (RED) when the double-prefix bug causes the
            # pre-prefixed call to miss the fast-path and fall through to the
            # coalesce/enqueue disk-scan; stubbed so that fallthrough resolves
            # cleanly to 'queued' (a clean assertion mismatch) instead of an
            # unrelated AttributeError.
            find_inflight_merge_worktree=AsyncMock(return_value=None),
        )
        stub_harness = types.SimpleNamespace(
            git_ops=stub_git, _merge_worker=None, _terminal_retention=None
        )
        esc_queue = EscalationQueue(tmp_path / 'esc')
        orch_config = self._make_orch_config(tmp_path / 'repo')

        server = create_server(
            esc_queue,
            harness=stub_harness,
            orch_config=orch_config,
            merge_queue=asyncio.Queue(),
        )

        bare_result = await _call_merge_request(
            server,
            task_id='123',
            branch='123',
            worktree=str(tmp_path / 'wt-bare'),
            description='',
        )
        prefixed_result = await _call_merge_request(
            server,
            task_id='123',
            branch='task/123',
            worktree=str(tmp_path / 'wt-prefixed'),
            description='',
        )

        assert bare_result.get('status') == 'already_merged', (
            f'Expected status already_merged for bare branch, got: {bare_result}'
        )
        assert bare_result.get('commit') == tip
        assert prefixed_result.get('status') == 'already_merged', (
            f'Expected status already_merged for pre-prefixed branch, got: {prefixed_result}'
        )
        assert prefixed_result.get('commit') == tip

        called_with = [call.args[0] for call in resolve_branch_sha.call_args_list]
        assert called_with == ['task/123', 'task/123'], (
            "Expected both calls to resolve the same ref 'task/123' "
            f"(never a double-prefixed 'task/task/123'), got: {called_with}"
        )


# ---------------------------------------------------------------------------
# TestMergeRequestWorkflowVerifyEmission — step-1/3 (2411): verified_green
# ---------------------------------------------------------------------------
# merge_request(verified_green=True) emits EventType.workflow_verify so the
# merge-skew classifier's I5 branch-green fact (merge_disposition.
# _branch_pre_merge_verify_green, keyed by task_id, reads only data['passed'])
# can source from non-orchestrator submission pathways (/merge-queue,
# /unblock, /do) — mirroring the orchestrator's own emission at
# workflow.py:1724-1733 (task 2381 alpha / 2383 beta).


async def _call_merge_cancel(server, **kwargs: Any) -> dict[str, Any]:
    """Invoke the merge_cancel MCP tool directly."""
    tool = await server.get_tool('merge_cancel')
    return await tool.fn(**kwargs)


@pytest.mark.asyncio
class TestMergeRequestWorkflowVerifyEmission:
    """merge_request(verified_green=True) emits a workflow_verify row."""

    def _make_orch_config(self, tmp_path: Path):
        """Create a minimal OrchestratorConfig without a git remote."""
        from orchestrator.config import OrchestratorConfig  # type: ignore[reportMissingImports]
        return OrchestratorConfig(project_root=tmp_path)

    async def test_verified_green_emits_workflow_verify(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A fresh dispatch with verified_green=True emits EXACTLY ONE
        workflow_verify row shaped like the orchestrator's own emission:
        task_id-keyed, data['passed'] is True, data['branch'] the full
        queued ref, data['base_sha'] the best-effort dispatch-time merge base.

        RED (pre step-2): merge_request has no verified_green parameter, so
        this call raises TypeError; even once accepted, no row is emitted.
        """
        import asyncio
        from unittest.mock import AsyncMock

        import orchestrator.merge_queue as orchestrator_merge_queue  # type: ignore[reportMissingImports]

        tip = 't' * 40
        sentinel_base_sha = 's' * 40
        monkeypatch.setattr(
            orchestrator_merge_queue,
            '_resolve_dispatch_time_merge_base',
            AsyncMock(return_value=sentinel_base_sha),
        )
        # is_ancestor is stubbed False to reach the fresh-dispatch path; the
        # task-2945 patch-id backstop then runs after that miss.  Stub it False
        # (this SimpleNamespace git_ops has no project_root for a real
        # `git cherry`, and a genuine fresh dispatch is not patch-id-contained).
        monkeypatch.setattr(
            orchestrator_merge_queue,
            'patch_content_contained',
            AsyncMock(return_value=False),
        )

        stub_git = types.SimpleNamespace(
            resolve_branch_sha=AsyncMock(
                side_effect=lambda b: tip if b == 'task/777' else None
            ),
            # False so the already_merged fast-path is NOT taken — the
            # request must reach the fresh-dispatch/coalesce path below.
            is_ancestor=AsyncMock(return_value=False),
            find_inflight_merge_worktree=AsyncMock(return_value=None),
        )
        stub_harness = types.SimpleNamespace(
            git_ops=stub_git, _merge_worker=None, _terminal_retention=None,
        )
        esc_queue = EscalationQueue(tmp_path / 'esc')
        orch_config = self._make_orch_config(tmp_path / 'repo')
        event_store = EventStore(tmp_path / 'runs.db', 'run-wf-verify')

        server = create_server(
            esc_queue,
            harness=stub_harness,
            orch_config=orch_config,
            event_store=event_store,
            merge_queue=asyncio.Queue(),
        )

        result = await _call_merge_request(
            server,
            task_id='777',
            branch='777',
            worktree=str(tmp_path / 'wt'),
            description='',
            verified_green=True,
            wait_secs=0,
        )

        try:
            rows = event_store.fetch_events_by_type(EventType.workflow_verify)
            assert len(rows) == 1, (
                f'Expected exactly one workflow_verify row, got: {rows}'
            )
            row = rows[0]
            assert row['task_id'] == '777', f'Expected task_id 777, got: {row}'
            assert row['data']['passed'] is True, (
                f"Expected data['passed'] is True, got: {row}"
            )
            assert row['data']['branch'] == 'task/777', (
                f"Expected data['branch']=='task/777', got: {row}"
            )
            assert row['data']['base_sha'] == sentinel_base_sha, (
                f"Expected data['base_sha']=={sentinel_base_sha!r}, got: {row}"
            )
        finally:
            # Fresh dispatch with no worker draining the queue leaves the
            # waiter future pending — cancel it via merge_cancel (mirrors the
            # never_future.cancel() cleanup in TestMergeRequestDedup).
            assert result.get('status') == 'queued', (
                f'Expected status queued, got: {result}'
            )
            await _call_merge_cancel(server, request_id=result['request_id'])

    async def test_default_verified_green_emits_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Calling merge_request WITHOUT verified_green (implicit default
        False) emits zero workflow_verify rows — the honest
        INDETERMINATE-preserving gate for pathways that cannot vouch a
        verify actually ran.
        """
        from unittest.mock import AsyncMock

        import orchestrator.merge_queue as orchestrator_merge_queue  # type: ignore[reportMissingImports]

        tip = 't' * 40
        monkeypatch.setattr(
            orchestrator_merge_queue,
            '_resolve_dispatch_time_merge_base',
            AsyncMock(return_value='s' * 40),
        )
        # is_ancestor False → reach the dispatch path; the task-2945 patch-id
        # backstop then runs.  Stub it False (SimpleNamespace git_ops has no
        # project_root for `git cherry`; a fresh dispatch is not contained).
        monkeypatch.setattr(
            orchestrator_merge_queue,
            'patch_content_contained',
            AsyncMock(return_value=False),
        )

        stub_git = types.SimpleNamespace(
            resolve_branch_sha=AsyncMock(
                side_effect=lambda b: tip if b == 'task/778' else None
            ),
            is_ancestor=AsyncMock(return_value=False),
            find_inflight_merge_worktree=AsyncMock(return_value=None),
        )
        stub_harness = types.SimpleNamespace(
            git_ops=stub_git, _merge_worker=None, _terminal_retention=None,
        )
        orch_config = self._make_orch_config(tmp_path / 'repo')
        event_store = EventStore(tmp_path / 'runs.db', 'run-wf-verify-default')

        server = create_server(
            EscalationQueue(tmp_path / 'esc'),
            harness=stub_harness,
            orch_config=orch_config,
            event_store=event_store,
            merge_queue=asyncio.Queue(),
        )

        result = await _call_merge_request(
            server,
            task_id='778',
            branch='778',
            worktree=str(tmp_path / 'wt'),
            description='',
            wait_secs=0,
        )

        try:
            rows = event_store.fetch_events_by_type(EventType.workflow_verify)
            assert rows == [], f'Expected zero workflow_verify rows, got: {rows}'
        finally:
            assert result.get('status') == 'queued', (
                f'Expected status queued, got: {result}'
            )
            await _call_merge_cancel(server, request_id=result['request_id'])

    async def test_verified_green_false_explicit_emits_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """verified_green=False (explicit) emits zero workflow_verify rows —
        pinned separately from the implicit-default case above since the
        gate is ``if verified_green:``, not an ``is None`` check.
        """
        from unittest.mock import AsyncMock

        import orchestrator.merge_queue as orchestrator_merge_queue  # type: ignore[reportMissingImports]

        tip = 't' * 40
        monkeypatch.setattr(
            orchestrator_merge_queue,
            '_resolve_dispatch_time_merge_base',
            AsyncMock(return_value='s' * 40),
        )
        # is_ancestor False → reach the dispatch path; the task-2945 patch-id
        # backstop then runs.  Stub it False (SimpleNamespace git_ops has no
        # project_root for `git cherry`; a fresh dispatch is not contained).
        monkeypatch.setattr(
            orchestrator_merge_queue,
            'patch_content_contained',
            AsyncMock(return_value=False),
        )

        stub_git = types.SimpleNamespace(
            resolve_branch_sha=AsyncMock(
                side_effect=lambda b: tip if b == 'task/779' else None
            ),
            is_ancestor=AsyncMock(return_value=False),
            find_inflight_merge_worktree=AsyncMock(return_value=None),
        )
        stub_harness = types.SimpleNamespace(
            git_ops=stub_git, _merge_worker=None, _terminal_retention=None,
        )
        orch_config = self._make_orch_config(tmp_path / 'repo')
        event_store = EventStore(tmp_path / 'runs.db', 'run-wf-verify-explicit-false')

        server = create_server(
            EscalationQueue(tmp_path / 'esc'),
            harness=stub_harness,
            orch_config=orch_config,
            event_store=event_store,
            merge_queue=asyncio.Queue(),
        )

        result = await _call_merge_request(
            server,
            task_id='779',
            branch='779',
            worktree=str(tmp_path / 'wt'),
            description='',
            verified_green=False,
            wait_secs=0,
        )

        try:
            rows = event_store.fetch_events_by_type(EventType.workflow_verify)
            assert rows == [], f'Expected zero workflow_verify rows, got: {rows}'
        finally:
            assert result.get('status') == 'queued', (
                f'Expected status queued, got: {result}'
            )
            await _call_merge_cancel(server, request_id=result['request_id'])

    async def test_verified_green_true_but_already_merged_emits_nothing(
        self, tmp_path: Path,
    ) -> None:
        """verified_green=True on a branch whose tip is already an ancestor
        of main hits the already_merged fast-path and returns BEFORE the
        emission block is reached — a branch already on main has no pending
        merge to attribute, so this correctly emits nothing.
        """
        from unittest.mock import AsyncMock

        tip = 't' * 40
        stub_git = types.SimpleNamespace(
            resolve_branch_sha=AsyncMock(
                side_effect=lambda b: tip if b == 'task/780' else None
            ),
            is_ancestor=AsyncMock(return_value=True),
            find_inflight_merge_worktree=AsyncMock(return_value=None),
        )
        stub_harness = types.SimpleNamespace(
            git_ops=stub_git, _merge_worker=None, _terminal_retention=None,
        )
        orch_config = self._make_orch_config(tmp_path / 'repo')
        event_store = EventStore(tmp_path / 'runs.db', 'run-wf-verify-already-merged')

        server = create_server(
            EscalationQueue(tmp_path / 'esc'),
            harness=stub_harness,
            orch_config=orch_config,
            event_store=event_store,
            merge_queue=asyncio.Queue(),
        )

        result = await _call_merge_request(
            server,
            task_id='780',
            branch='780',
            worktree=str(tmp_path / 'wt'),
            description='',
            verified_green=True,
            wait_secs=0,
        )

        # Fast-path short-circuits before any MergeRequest/future is created —
        # no queue entry, no waiter, so no merge_cancel cleanup is needed
        # (mirrors test_already_merged_fast_path_dedups_prefixed_branch).
        assert result.get('status') == 'already_merged', (
            f'Expected status already_merged, got: {result}'
        )
        rows = event_store.fetch_events_by_type(EventType.workflow_verify)
        assert rows == [], f'Expected zero workflow_verify rows, got: {rows}'

    async def test_verified_green_true_with_no_event_store_does_not_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """verified_green=True on a server with no event_store wired (e.g.
        standalone escalation without an orchestrator) must NOT raise — it
        degrades to no attribution, the same fail-open guarantee as a git
        error inside the base_sha helper, and still returns a normal
        queued/attached response.

        RED (pre step-4): the emission block unconditionally calls
        event_store.emit(...) whenever verified_green is True, so a None
        event_store raises AttributeError here instead of degrading
        gracefully.
        """
        from unittest.mock import AsyncMock

        import orchestrator.merge_queue as orchestrator_merge_queue  # type: ignore[reportMissingImports]

        tip = 't' * 40
        # Monkeypatched for determinism even though the post-step-4 gate
        # short-circuits before this would ever be called when event_store
        # is None — avoids relying on a real git subprocess pre-step-4.
        monkeypatch.setattr(
            orchestrator_merge_queue,
            '_resolve_dispatch_time_merge_base',
            AsyncMock(return_value='s' * 40),
        )
        # is_ancestor False → reach the gate path; the task-2945 patch-id
        # backstop then runs.  Stub it False (SimpleNamespace git_ops has no
        # project_root for `git cherry`; a fresh dispatch is not contained).
        monkeypatch.setattr(
            orchestrator_merge_queue,
            'patch_content_contained',
            AsyncMock(return_value=False),
        )

        stub_git = types.SimpleNamespace(
            resolve_branch_sha=AsyncMock(
                side_effect=lambda b: tip if b == 'task/781' else None
            ),
            is_ancestor=AsyncMock(return_value=False),
            find_inflight_merge_worktree=AsyncMock(return_value=None),
        )
        stub_harness = types.SimpleNamespace(
            git_ops=stub_git, _merge_worker=None, _terminal_retention=None,
        )
        orch_config = self._make_orch_config(tmp_path / 'repo')

        server = create_server(
            EscalationQueue(tmp_path / 'esc'),
            harness=stub_harness,
            orch_config=orch_config,
            event_store=None,
            merge_queue=asyncio.Queue(),
        )

        result = await _call_merge_request(
            server,
            task_id='781',
            branch='781',
            worktree=str(tmp_path / 'wt'),
            description='',
            verified_green=True,
            wait_secs=0,
        )

        try:
            assert result.get('status') in ('queued', 'attached'), (
                f'Expected a normal queued/attached response, got: {result}'
            )
        finally:
            if result.get('status') == 'queued':
                await _call_merge_cancel(server, request_id=result['request_id'])

    async def test_verified_green_true_on_resubmit_attach_emits_second_row(
        self, tmp_path: Path,
    ) -> None:
        """A verified_green=True resubmission for a branch already in flight
        (same task_id) coalesces onto the 'attached' path — which is reached
        via the SAME emission block, placed before the coalesce/enqueue call.
        This pins two previously-unverified facts (reviewer follow-up on
        step-4): (1) the coalesce/'attached' path emits a row too, not just
        fresh dispatch — harmless duplication since the classifier
        (_branch_pre_merge_verify_green) is any-prior-green keyed by
        task_id; and (2) with no harness/git_ops wired, resolved_tip stays
        None so data['base_sha'] is None — the informational field degrades
        gracefully instead of the emission being skipped.
        """
        esc_queue = EscalationQueue(tmp_path / 'esc')
        orch_config = self._make_orch_config(tmp_path / 'repo')
        event_store = EventStore(tmp_path / 'runs.db', 'run-wf-verify-attach')

        server = create_server(
            esc_queue,
            # No harness passed — git_ops_for_scan stays None, so resolved_tip
            # is None on both calls (documents the base_sha=None case) and the
            # coalesce gate's tip classifier is skipped (back-compat,
            # registry-only comparison), guaranteeing the resubmit attaches.
            orch_config=orch_config,
            event_store=event_store,
            merge_queue=asyncio.Queue(),
        )

        first = await _call_merge_request(
            server,
            task_id='790',
            branch='790',
            worktree=str(tmp_path / 'wt'),
            description='',
            verified_green=True,
            wait_secs=0,
        )
        assert first.get('status') == 'queued', (
            f'Expected first submission status queued, got: {first}'
        )

        try:
            second = await _call_merge_request(
                server,
                task_id='790',
                branch='790',
                worktree=str(tmp_path / 'wt'),
                description='',
                verified_green=True,
                wait_secs=0,
            )
            assert second.get('status') == 'attached', (
                f'Expected resubmit status attached (coalesced), got: {second}'
            )

            rows = event_store.fetch_events_by_type(EventType.workflow_verify)
            assert len(rows) == 2, (
                f'Expected two workflow_verify rows (dispatch + attach), got: {rows}'
            )
            for row in rows:
                assert row['task_id'] == '790', f'Expected task_id 790, got: {row}'
                assert row['data']['passed'] is True, (
                    f"Expected data['passed'] is True, got: {row}"
                )
                assert row['data']['base_sha'] is None, (
                    f"Expected data['base_sha'] is None (no git_ops wired), got: {row}"
                )
                assert row['data']['branch'] == 'task/790', (
                    f"Expected data['branch']=='task/790', got: {row}"
                )
        finally:
            await _call_merge_cancel(server, request_id=first['request_id'])


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

    def _make_finalize_fixture(
        self,
        tmp_path: Path,
        task_id: str,
        advance_outcome,
        merge_commit: str,
    ):
        """Shared setup for the ``_finalize_inflight`` retirement-oracle
        sibling tests (``test_gate_reverify_failure_retires_registry_entry``,
        ``test_wip_overlap_finalize_retires_head_and_preserves_halt``):
        builds a ``SpeculativeMergeWorker`` wired to a stub ``git_ops`` whose
        ``advance_main`` always returns ``advance_outcome``, plus a
        registered sole ``RealMergeItem``/``InflightEntry`` (state
        ``VERIFYING``) ready for ``await worker._finalize_inflight(entry)``.

        Returns ``(worker, req, entry, advance_call_args)`` — the last
        element accumulates each ``advance_main`` call's ``(args, kwargs)``
        for callers that assert on call/retry counts.
        """
        import asyncio
        import types

        from orchestrator.git_ops import MergeResult  # type: ignore[reportMissingImports]
        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            InflightEntry,
            ItemLifecycleState,
            MergeRequest,
            RealMergeItem,
            SpeculativeMergeWorker,
        )

        loop = asyncio.get_running_loop()
        config = self._make_orch_config(tmp_path / 'repo')
        mq: asyncio.Queue = asyncio.Queue()

        merge_wt = tmp_path / 'merge'
        merge_wt.mkdir()

        advance_call_args: list[tuple[tuple, dict]] = []

        async def fake_advance_main(*args, **kwargs):
            advance_call_args.append((args, kwargs))
            return advance_outcome

        async def fake_cleanup_merge_worktree(path):
            pass

        git_ops_stub = types.SimpleNamespace(
            advance_main=fake_advance_main,
            cleanup_merge_worktree=fake_cleanup_merge_worktree,
            config=config,
        )
        worker = SpeculativeMergeWorker(git_ops=git_ops_stub, queue=mq)  # type: ignore[reportArgumentType]

        req = MergeRequest(
            task_id=task_id,
            branch=QueuedBranch.parse(task_id, 'task/'),
            worktree=tmp_path / 'wt',
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
        )
        merge_result = MergeResult(
            success=True,
            merge_commit=merge_commit,
            merge_worktree=merge_wt,
        )
        item = RealMergeItem(
            request=req,
            merge_result=merge_result,
            merge_wt=merge_wt,
            base_sha='base0sha',
            speculative=False,
        )
        entry = InflightEntry(
            item=item,
            lease=None,
            verify_task=None,
            merge_wt=merge_wt,
            was_speculative=False,
        )
        worker._register_item(item, initial=ItemLifecycleState.VERIFYING)

        return worker, req, entry, advance_call_args

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
            branch=QueuedBranch.parse('T1', 'task/'),
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
            branch=QueuedBranch.parse('G1', 'task/'),
            worktree=tmp_path / 'wt2',
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
            train_id='train-1',
            member_task_ids=['G1'],
            tip_branch=QueuedBranch.parse('G1', 'task/'),
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
            branch=QueuedBranch.parse('Q', 'task/'),
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
            task_id='LIVE', branch=QueuedBranch.parse('LIVE', 'task/'), worktree=tmp_path / 'wt1',
            pre_rebased=False, task_files=None, module_configs=[], config=config,
            result=live_fut,
        ))
        await mq.put(MergeRequest(
            task_id='DEAD', branch=QueuedBranch.parse('DEAD', 'task/'), worktree=tmp_path / 'wt2',
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
        """snapshot() maps the ItemLifecycle registry (merging via _register_item) + _inflight verify entries + _verifier_queue to correct wire states."""
        import asyncio
        import types

        from orchestrator.git_ops import MergeResult  # type: ignore[reportMissingImports]
        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            InflightEntry,
            ItemLifecycleState,
            MergeRequest,
            RealMergeItem,
            SpeculativeMergeWorker,
        )

        loop = asyncio.get_running_loop()
        config = self._make_orch_config(tmp_path / 'repo')
        mq: asyncio.Queue = asyncio.Queue()
        git_ops_stub = types.SimpleNamespace()
        worker = SpeculativeMergeWorker(git_ops=git_ops_stub, queue=mq)  # type: ignore[reportArgumentType]

        def _req(tid: str):
            return MergeRequest(
                task_id=tid, branch=QueuedBranch.parse(tid, 'task/'),
                worktree=tmp_path / f'wt-{tid}',
                pre_rebased=False, task_files=None, module_configs=[],
                config=config, result=loop.create_future(),
            )

        # M — in the merger (merging)
        req_M = _req('M')
        worker._register_item(req_M, initial=ItemLifecycleState.MERGING)

        # A — in the verifier queue (awaiting_verify)
        merge_wt_A = tmp_path / 'mergeA'
        merge_wt_A.mkdir()
        merge_result_A = MergeResult(success=True, merge_commit='deadbeefA0000000', merge_worktree=merge_wt_A)
        item_A = RealMergeItem(
            request=_req('A'),
            merge_result=merge_result_A, merge_wt=merge_wt_A,
            base_sha='base', speculative=False,
        )
        await worker._verifier_queue.put(item_A)

        # V — currently being verified (phase = verify_phase param).
        # The entry lives in worker._inflight (the singular
        # _verify_item/_verify_phase fields were retired in task 1736; the
        # free-form InflightEntry.phase field was retired in task lambda / 2173).
        # snapshot() derives the entry phase from the ItemLifecycle registry via
        # _entry_phase() and iterates _inflight head-first (section 1), so this
        # entry is head-of-line at position 0, ahead of the awaiting_verify (A)
        # and merging (M) sections.
        merge_wt_V = tmp_path / 'mergeV'
        merge_wt_V.mkdir()
        merge_result_V = MergeResult(success=True, merge_commit='deadbeefV0000000', merge_worktree=merge_wt_V)
        item_V = RealMergeItem(
            request=_req('V'),
            merge_result=merge_result_V, merge_wt=merge_wt_V,
            base_sha='base', speculative=False,
        )
        worker._inflight.append(InflightEntry(
            item=item_V,
            lease=None,
            verify_task=None,
            merge_wt=merge_wt_V,
            was_speculative=False,
        ))
        # Per-entry phase now DERIVES from the ItemLifecycle registry
        # (task lambda / 2173 deleted InflightEntry.phase) — drive the registry
        # to the parametrized state so snapshot()/_entry_phase surface it.
        worker._register_item(item_V, initial=ItemLifecycleState.VERIFYING)
        if verify_phase == 'gate_reverify':
            worker._note_transition(
                item_V.request.request_id,
                ItemLifecycleState.VERIFYING, ItemLifecycleState.GATE_REVERIFY,
            )
        elif verify_phase == 'finalizing':
            worker._note_transition(
                item_V.request.request_id,
                ItemLifecycleState.VERIFYING, ItemLifecycleState.FINALIZING,
            )

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
        """_finalize_inflight sets phase='finalizing' before advance_main and
        clears the finalize-head window afterwards.

        The CAS/advance/gate finalize half moved out of _verify_and_advance into
        _finalize_inflight (task 1735, commit 2a9db6ac83); the VERIFY half (which
        sets phase='verifying') now lives in _run_inflight_verify/_dispatch_item.
        This test drives the real finalize path: an InflightEntry whose verify
        already passed (verify_task=None) is finalized, and we capture the live
        phase that snapshot() would surface at the moment advance_main runs.
        """
        import asyncio
        import types

        from orchestrator.git_ops import (  # type: ignore[reportMissingImports]
            AdvanceOutcome,
            MergeResult,
        )
        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            InflightEntry,
            ItemLifecycleState,
            MergeRequest,
            RealMergeItem,
            SpeculativeMergeWorker,
        )

        loop = asyncio.get_running_loop()
        config = self._make_orch_config(tmp_path / 'repo')
        mq: asyncio.Queue = asyncio.Queue()

        merge_wt = tmp_path / 'merge'
        merge_wt.mkdir()

        # Capture the LIVE phase that snapshot() would surface at advance_main
        # time: read it through the live finalize-head entry's per-entry phase
        # (worker._finalizing_head_entry(), set by _finalize_inflight — the
        # production object, not a test-held reference) and snapshot()'s
        # verify_in_progress (the real merge_status observability path).
        captured_phases: list[str | None] = []
        captured_snapshot_phases: list[str | None] = []

        # Explicit return annotation: breaks the pyright inference cycle between
        # this closure (which reads `worker`) and the git_ops_stub/worker bindings
        # defined below — without it, embedding fake_advance_main in git_ops_stub
        # makes pyright report it as self-referential.
        async def fake_advance_main(*args, **kwargs) -> AdvanceOutcome:
            fh = worker._finalizing_head_entry()
            captured_phases.append(worker._entry_phase(fh) if fh is not None else None)
            snap = worker.snapshot()
            vip = snap.get('verify_in_progress')
            captured_snapshot_phases.append(vip['phase'] if vip else None)
            # Return 'not_descendant' — terminal non-advanced path
            return AdvanceOutcome('not_descendant')

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
            branch=QueuedBranch.parse('P', 'task/'),
            worktree=tmp_path / 'wt',
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
        )
        merge_result = MergeResult(success=True, merge_commit='deadbeef0000000', merge_worktree=merge_wt)
        item = RealMergeItem(
            request=req,
            merge_result=merge_result,
            merge_wt=merge_wt,
            base_sha='base000',
            speculative=False,
        )
        # verify_task=None → verify already passed (no fail/skip); _finalize_inflight
        # goes straight to the CAS advance_main loop where it sets phase='finalizing'.
        entry = InflightEntry(
            item=item,
            lease=None,
            verify_task=None,
            merge_wt=merge_wt,
            was_speculative=False,
        )
        # Phase now derives from the registry (task lambda / 2173 deleted
        # InflightEntry.phase); register at VERIFYING so _finalize_inflight's
        # production VERIFYING->FINALIZING transition takes effect.
        worker._register_item(item, initial=ItemLifecycleState.VERIFYING)

        await worker._finalize_inflight(entry)

        # advance_main ran; phase was 'finalizing' at that point — on the
        # finalize-head entry AND surfaced via snapshot()'s live
        # verify_in_progress (the real merge_status observability path).
        assert captured_phases == ['finalizing'], (
            f'Expected finalizing on entry.phase at '
            f'advance_main, got: {captured_phases}'
        )
        assert captured_snapshot_phases == ['finalizing'], (
            f'Expected snapshot().verify_in_progress.phase==finalizing during '
            f'finalize, got: {captured_snapshot_phases}'
        )

        # Request future should be resolved (not_descendant → blocked)
        assert req.result.done(), 'request future should be resolved after terminal advance_main'

        # Finalize-head window cleared after _finalize_inflight returns.
        assert worker._finalizing_head_entry() is None, (
            f'Expected _finalizing_head_entry() cleared, got: {worker._finalizing_head_entry()!r}'
        )

        # snapshot has no active verifier states (entry was popped/finalized).
        snap = worker.snapshot()
        bad_states = {e['state'] for e in snap['entries']} & {'verifying', 'finalizing', 'gate_reverify'}
        assert not bad_states, f'Snapshot should have no active verifier states, got: {bad_states}'

    # ── amend: gate_reverify phase set/cleared by production code ─────────

    async def test_gate_reverify_phase_set_and_cleared(self, tmp_path: Path):
        """_finalize_inflight sets phase='gate_reverify' when advance_main returns
        'rebased_pending_reverify', and resets to 'finalizing' after the gate clears
        (so subsequent advance_main retries report the correct phase).

        The CAS/advance/gate loop — including the gate_reverify phase and the
        _reverify_rebased_tree call — moved out of _verify_and_advance into
        _finalize_inflight (task 1735, commit 2a9db6ac83).  This test drives the
        real finalize path via an already-passed InflightEntry (verify_task=None),
        and captures the LIVE phase surfaced by snapshot() at the reverify call.

        task 1997: the git_ops stub deliberately does NOT set
        _last_advanced_sha/_rebased_from/_rebased_onto — the post-rebase SHAs
        are sourced entirely from the AdvanceOutcome return value (the getattr
        side channel is retired; the CAS loop reads
        adv_outcome.advanced_sha/rebased_from/rebased_onto directly).
        """
        import asyncio
        import types

        from orchestrator.git_ops import (  # type: ignore[reportMissingImports]
            AdvanceOutcome,
            MergeResult,
        )
        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            InflightEntry,
            ItemLifecycleState,
            MergeRequest,
            RealMergeItem,
            SpeculativeMergeWorker,
        )

        loop = asyncio.get_running_loop()
        config = self._make_orch_config(tmp_path / 'repo')
        mq: asyncio.Queue = asyncio.Queue()

        merge_wt = tmp_path / 'merge'
        merge_wt.mkdir()

        REBASED_SHA = 'rebased0abc'
        REBASED_FROM = 'from0sha'
        REBASED_ONTO = 'onto0sha'

        advance_call_args: list[tuple[tuple, dict]] = []
        captured_phases_reverify: list[str | None] = []
        captured_snapshot_reverify: list[str | None] = []
        captured_phases_advance2: list[str | None] = []

        # Explicit return annotation breaks the pyright inference cycle between
        # this closure (reads `worker`) and git_ops_stub/worker below.
        async def fake_advance_main(*args, **kwargs) -> AdvanceOutcome:
            advance_call_args.append((args, kwargs))
            if len(advance_call_args) == 1:
                # First call: trigger rebase path.  The SHA fields are carried
                # SOLELY by the return value — the git_ops stub below has no
                # _last_advanced_sha/_rebased_from/_rebased_onto attributes.
                return AdvanceOutcome(
                    'rebased_pending_reverify',
                    advanced_sha=REBASED_SHA,
                    rebased_from=REBASED_FROM,
                    rebased_onto=REBASED_ONTO,
                )
            else:
                # Second call (after gate cleared): terminal failure.
                # Read the live finalize-head entry (production object set by
                # _finalize_inflight), not a test-held reference, to avoid a
                # forward closure ref to the later-bound `entry`.
                fh = worker._finalizing_head_entry()
                captured_phases_advance2.append(
                    worker._entry_phase(fh) if fh is not None else None
                )
                return AdvanceOutcome('not_descendant')

        async def fake_cleanup_merge_worktree(path):
            pass

        # Deliberately NO _last_advanced_sha/_rebased_from/_rebased_onto —
        # the SHAs must ride the AdvanceOutcome return value alone (task 1997).
        git_ops_stub = types.SimpleNamespace(
            advance_main=fake_advance_main,
            cleanup_merge_worktree=fake_cleanup_merge_worktree,
            config=config,
        )
        worker = SpeculativeMergeWorker(git_ops=git_ops_stub, queue=mq)  # type: ignore[reportArgumentType]

        req = MergeRequest(
            task_id='GR',
            branch=QueuedBranch.parse('GR', 'task/'),
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
        item = RealMergeItem(
            request=req,
            merge_result=merge_result,
            merge_wt=merge_wt,
            base_sha='base0sha',
            speculative=False,
        )
        # verify_task=None → verify already passed; _finalize_inflight runs the
        # CAS advance_main loop that owns the gate_reverify phase transition.
        entry = InflightEntry(
            item=item,
            lease=None,
            verify_task=None,
            merge_wt=merge_wt,
            was_speculative=False,
        )
        # Phase now derives from the registry (task lambda / 2173 deleted
        # InflightEntry.phase); register at VERIFYING so _finalize_inflight's
        # production VERIFYING->FINALIZING->GATE_REVERIFY transitions take effect.
        worker._register_item(item, initial=ItemLifecycleState.VERIFYING)

        import orchestrator.merge_queue as mq_module  # type: ignore[reportMissingImports]

        captured_reverify_kwargs: dict = {}

        async def fake_reverify_rebased_tree(*args, **kwargs):
            # Capture the phase at the moment _reverify_rebased_tree is invoked,
            # both on the entry and via the live snapshot() observability path.
            captured_phases_reverify.append(worker._entry_phase(entry))
            snap = worker.snapshot()
            vip = snap.get('verify_in_progress')
            captured_snapshot_reverify.append(vip['phase'] if vip else None)
            # Capture the SHA-bearing kwargs forwarded to the reverify gate —
            # must come from the AdvanceOutcome return value, not getattr.
            captured_reverify_kwargs.update(
                rebased_from=kwargs.get('rebased_from'),
                rebased_onto=kwargs.get('rebased_onto'),
                merge_sha=kwargs.get('merge_sha'),
            )
            # Return None → gate cleared (disjoint/green), advance proceeds.
            return None

        original_reverify = mq_module._reverify_rebased_tree
        mq_module._reverify_rebased_tree = fake_reverify_rebased_tree  # type: ignore[attr-defined]
        try:
            await worker._finalize_inflight(entry)
        finally:
            mq_module._reverify_rebased_tree = original_reverify  # type: ignore[attr-defined]

        # _reverify_rebased_tree must have been called exactly once
        assert len(captured_phases_reverify) == 1, (
            f'Expected _reverify_rebased_tree called once, got: {len(captured_phases_reverify)}'
        )
        # Phase must be 'gate_reverify' when _reverify_rebased_tree is invoked —
        # on the entry AND as surfaced by snapshot()'s live verify_in_progress.
        assert captured_phases_reverify[0] == 'gate_reverify', (
            f'Expected gate_reverify at reverify call, got: {captured_phases_reverify[0]!r}'
        )
        assert captured_snapshot_reverify == ['gate_reverify'], (
            f'Expected snapshot().verify_in_progress.phase==gate_reverify during '
            f'reverify, got: {captured_snapshot_reverify}'
        )

        # The SHAs forwarded to the reverify gate must come from the
        # AdvanceOutcome return value — the stub has no getattr side channel.
        assert captured_reverify_kwargs == {
            'rebased_from': REBASED_FROM,
            'rebased_onto': REBASED_ONTO,
            'merge_sha': REBASED_SHA,
        }, (
            f'_reverify_rebased_tree must be called with the AdvanceOutcome '
            f'fields, not the (unset) getattr side channel; got '
            f'{captured_reverify_kwargs!r}'
        )

        # advance_main must have been called twice
        assert len(advance_call_args) == 2, (
            f'Expected 2 advance_main calls, got: {len(advance_call_args)}'
        )
        # The second (retry) call must resume from the rebased SHA/base —
        # correct rebased_from -> item.base_sha rebuild (task 1990 replace-only
        # rebuild) sourced from the return object, not the getattr side channel.
        second_args, second_kwargs = advance_call_args[1]
        assert second_args[0] == REBASED_SHA, (
            f'second advance_main call must retry with current_sha sourced '
            f'from adv_outcome.advanced_sha; got {second_args[0]!r}'
        )
        assert second_kwargs.get('expected_main') == REBASED_ONTO, (
            f'second advance_main call must use item.base_sha rebuilt from '
            f'rebased_onto; got {second_kwargs.get("expected_main")!r}'
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

    # ── task 2604 review fix: gate FAILURE retires the registry entry ────────

    async def test_gate_reverify_failure_retires_registry_entry(self, tmp_path: Path):
        """When the post-rebase reverify gate FAILS (``_reverify_rebased_tree``
        returns a non-None MergeOutcome), the ``rebased_pending_reverify``
        gate-fail branch must route the terminal outcome through the unified
        chokepoint (``_resolve_or_drop_abandoned``) so ``_retire_item`` fires.

        Regression for the task-2604 review finding: the branch used to call a
        raw ``req.result.set_result(gate)`` and ``return False`` WITHOUT
        retiring, leaving the entry non-terminal at GATE_REVERIFY in
        ``_live_items`` forever — a ghost 'gate_reverify' item surfaced by
        ``_finalizing_head_entry()`` / ``snapshot()``.  On the unfixed code
        this test fails: ``_finalizing_head_entry()`` returns the ghost entry
        and snapshot() reports an active 'gate_reverify' state.
        """
        from orchestrator.git_ops import AdvanceOutcome  # type: ignore[reportMissingImports]
        from orchestrator.merge_queue import MergeOutcome  # type: ignore[reportMissingImports]

        REBASED_SHA = 'rebased0abc'
        REBASED_FROM = 'from0sha'
        REBASED_ONTO = 'onto0sha'

        # Single call expected: trigger the rebase/reverify path.  The gate
        # then fails, so advance_main is never called a second time.
        worker, req, entry, advance_call_args = self._make_finalize_fixture(
            tmp_path,
            task_id='GRF',
            advance_outcome=AdvanceOutcome(
                'rebased_pending_reverify',
                advanced_sha=REBASED_SHA,
                rebased_from=REBASED_FROM,
                rebased_onto=REBASED_ONTO,
            ),
            merge_commit='deadbeef00000002',
        )

        import orchestrator.merge_queue as mq_module  # type: ignore[reportMissingImports]

        GATE_FAIL = MergeOutcome('blocked', reason='post-rebase reverify gate failed')

        async def fake_reverify_rebased_tree(*args, **kwargs):
            # Non-None return → gate FAILED; the gate-fail branch resolves the
            # future and must retire the registry entry.
            return GATE_FAIL

        original_reverify = mq_module._reverify_rebased_tree
        mq_module._reverify_rebased_tree = fake_reverify_rebased_tree  # type: ignore[attr-defined]
        try:
            await worker._finalize_inflight(entry)
        finally:
            mq_module._reverify_rebased_tree = original_reverify  # type: ignore[attr-defined]

        # advance_main called exactly once (gate failed → no retry).
        assert len(advance_call_args) == 1, (
            f'Expected 1 advance_main call on gate failure, got: {len(advance_call_args)}'
        )

        # Future resolved with the gate outcome.
        assert req.result.done(), 'request future should be resolved on gate failure'
        assert req.result.result() is GATE_FAIL, (
            f'Expected the gate MergeOutcome delivered to the waiter, '
            f'got: {req.result.result()!r}'
        )

        # KEY REGRESSION: registry entry retired — no ghost finalize head.
        assert worker._finalizing_head_entry() is None, (
            f'Expected _finalizing_head_entry() cleared after gate failure, '
            f'got: {worker._finalizing_head_entry()!r}'
        )

        # snapshot() surfaces no active verifier states (no lingering
        # gate_reverify ghost in _live_items).
        snap = worker.snapshot()
        bad_states = {e['state'] for e in snap['entries']} & {'verifying', 'finalizing', 'gate_reverify'}
        assert not bad_states, (
            f'Snapshot should have no active verifier states after gate '
            f'failure, got: {bad_states}'
        )

    # ── wip_overlap: retire preserves the recoverable WIP halt-then-retry ──

    async def test_wip_overlap_finalize_retires_head_and_preserves_halt(self, tmp_path: Path):
        """When ``advance_main`` returns ``'wip_overlap'``, the terminal
        ``if result != 'cas_failed':`` branch (merge_queue.py:11952) must
        retire the registry entry through ``_resolve_or_drop_abandoned``
        WITHOUT clearing the recoverable WIP queue-halt.

        Regression for task 2609 (leak-fix pre-empted on main by task 2604):
        2604 converted this branch's raw ``req.result.set_result(outcome)``
        to ``self._resolve_or_drop_abandoned(req, outcome)`` (commit
        bdbd56ecf7), fixing the registry leak, but shipped with no test
        locking in the non-mechanical subtlety this test guards — that
        retiring the sole head after a wip_overlap outcome PRESERVES
        ``is_wip_halted`` so the halt-then-retry recovery flow still works.
        ``_map_advance_failure`` (merge_gates.py:699-736) calls
        ``halt('advance_main: wip_overlap')`` BEFORE returning the
        ``MergeOutcome('wip_halted', ...)``; per merge_queue.py's own comment
        on the coalesce re-drive path, "wip_halted has no awaiter to re-fire
        it ... each fresh workflow will hit the WIP-halt barrier
        independently and wait correctly" — i.e. the halt is recovered by a
        FRESH request (new request_id/future) hitting the barrier again, not
        by resuming this one, so retiring THIS request_id is correct and
        expected.

        The load-bearing regression guards here are assertions (3)-(4): they
        would RED if the terminal branch regressed to a raw ``set_result()``
        (the pre-2604 leak) that leaves the entry stuck non-terminal in
        ``_live_items``. Assertions (2)&(5) are weaker than a "proves
        untouched by retirement" framing implies — ``_retire_item``/
        ``_resolve_or_drop_abandoned`` mutate only ``_live_items``/
        ``_lifecycle``, and the lane-halt flag is mutated exclusively by the
        halt-lane methods, so retirement structurally cannot reach it; a
        hypothetical retirement-clears-halt bug is not what these two
        assertions are positioned to catch. They instead document the
        combined halted+retired end-state and confirm the halt reverses
        cleanly via ``unhalt_wip()``.
        """
        from orchestrator.git_ops import AdvanceOutcome  # type: ignore[reportMissingImports]
        from orchestrator.merge_queue import (
            ItemLifecycleState,  # type: ignore[reportMissingImports]
        )

        # wip_overlap: a recoverable halt, not a terminal advance failure —
        # the real _map_advance_failure (merge_gates.py:699) owns this
        # mapping, so no _reverify_rebased_tree monkeypatch is needed
        # (unlike the rebased_pending_reverify gate-fail sibling test
        # above).
        worker, req, entry, _advance_call_args = self._make_finalize_fixture(
            tmp_path,
            task_id='WIP',
            advance_outcome=AdvanceOutcome('wip_overlap'),
            merge_commit='deadbeef00000003',
        )

        assert not worker.is_wip_halted, 'sanity: queue should start un-halted'

        await worker._finalize_inflight(entry)

        # (1) Future resolved with a wip_halted MergeOutcome.
        assert req.result.done(), 'request future should be resolved on wip_overlap'
        delivered = req.result.result()
        assert delivered.status == 'wip_halted', (
            f'Expected a wip_halted MergeOutcome delivered to the waiter, '
            f'got: {delivered!r}'
        )

        # (2) The recoverable halt-then-retry barrier is engaged — retiring
        # the head must NOT clear it.
        assert worker.is_wip_halted, (
            'Expected the WIP halt to be engaged after a wip_overlap outcome'
        )

        # (3) HEAD RETIRED (no leak) — same retirement oracle as
        # test_gate_reverify_failure_retires_registry_entry above.
        assert worker._finalizing_head_entry() is None, (
            f'Expected _finalizing_head_entry() cleared after wip_overlap, '
            f'got: {worker._finalizing_head_entry()!r}'
        )
        assert req.request_id not in worker._live_items, (
            f'Expected {req.request_id!r} retired from _live_items, '
            f'got: {worker._live_items!r}'
        )
        assert worker._lifecycle.current(req.request_id) == ItemLifecycleState.TERMINAL, (
            f'Expected {req.request_id!r} at TERMINAL after retirement, '
            f'got: {worker._lifecycle.current(req.request_id)!r}'
        )

        # (4) snapshot() surfaces no active verifier states (no lingering
        # ghost entry in _live_items).
        snap = worker.snapshot()
        bad_states = {e['state'] for e in snap['entries']} & {'verifying', 'finalizing', 'gate_reverify'}
        assert not bad_states, (
            f'Snapshot should have no active verifier states after '
            f'wip_overlap, got: {bad_states}'
        )

        # (5) The halt reverses cleanly — retirement did not corrupt
        # _lane_halt.
        worker.unhalt_wip()
        assert not worker.is_wip_halted, (
            'Expected is_wip_halted False after unhalt_wip() — retirement '
            'must not have corrupted _lane_halt'
        )


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

    # ── step-7 (2554): failure reason surfaces through both durable tiers ────

    async def test_reason_surfaces_through_both_durable_tiers(
        self, tmp_path: Path,
    ) -> None:
        """A terminal 'blocked' outcome's failure reason surfaces via merge_status.

        (i) Ring survivor: a TerminalOutcomeRecord carrying reason= is served
            directly from the retention ring (Tier 2).
        (ii) Post-restart event-store: an empty ring + a merge_finalized event
            carrying reason= in its data dict is served from the event-store
            tier (Tier 3) — simulating a restart that dropped the ring.

        RED: _OPTIONAL_TERMINAL_META_FIELDS omits 'reason' and the durable
        resp builder has no reason plumbing, so resp['reason'] is absent in
        both cases (the event-store tier additionally relies on step-6's
        latest_merge_finalized row already carrying 'reason').
        """
        REASON = 'verify failed: gui_tsc'

        # (i) Ring survivor
        ring = TerminalOutcomeRetention()
        ring.record(TerminalOutcomeRecord(
            request_id='mr-reason-ring',
            task_id='r1',
            branch='r1',
            state='blocked',
            reason=REASON,
        ))
        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = types.SimpleNamespace(_merge_worker=None, _terminal_retention=ring)
        server = create_server(esc_queue, harness=stub_harness)

        result_ring = await _call_merge_status(server, request_id='mr-reason-ring')

        assert result_ring.get('state') == 'blocked', (
            f'Expected state=blocked from ring, got: {result_ring}'
        )
        assert result_ring.get('outcome') == 'blocked', (
            f'Expected outcome=blocked from ring, got: {result_ring}'
        )
        assert result_ring.get('reason') == REASON, (
            f'Expected reason={REASON!r} surfaced from ring tier, got: {result_ring}'
        )

        # (ii) Post-restart event-store: empty ring, same reason via event data.
        event_store = EventStore(tmp_path / 'runs.db', 'run-reason-ev')
        event_store.emit(
            EventType.merge_finalized,
            task_id='r2',
            data={
                'request_id': 'mr-reason-ev',
                'branch': 'r2',
                'state': 'blocked',
                'reason': REASON,
            },
        )
        empty_ring = TerminalOutcomeRetention()
        esc_queue2 = EscalationQueue(tmp_path / 'esc2')
        stub_harness2 = types.SimpleNamespace(
            _merge_worker=None, _terminal_retention=empty_ring
        )
        server2 = create_server(
            esc_queue2, harness=stub_harness2, event_store=event_store
        )

        result_ev = await _call_merge_status(server2, request_id='mr-reason-ev')

        assert result_ev.get('state') == 'blocked', (
            f'Expected state=blocked from event store, got: {result_ev}'
        )
        assert result_ev.get('reason') == REASON, (
            f'Expected reason={REASON!r} surfaced from event-store tier, got: {result_ev}'
        )

    # ── step-9 (1749): ring tier branch/task_id/alias lookup ─────────────────

    async def test_ring_tier_returns_terminal_record_by_branch(
        self, tmp_path: Path,
    ) -> None:
        """Ring tier: a branch= poll resolves to the ring's recorded terminal state."""
        ring = TerminalOutcomeRetention()
        ring.record(TerminalOutcomeRecord(
            request_id='mr-branchpoll',
            task_id='T-bpoll',
            branch='branch-bylookup',
            state='done',
        ))

        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = types.SimpleNamespace(_merge_worker=None, _terminal_retention=ring)
        server = create_server(esc_queue, harness=stub_harness)

        result = await _call_merge_status(server, branch='branch-bylookup')

        assert result.get('state') == 'done', (
            f'Expected done from ring branch lookup, got: {result}'
        )
        assert result.get('request_id') == 'mr-branchpoll', (
            f'Expected resolved record request_id, got: {result}'
        )
        assert result.get('outcome') == 'done', (
            f'Expected outcome=done, got: {result}'
        )

    async def test_ring_tier_returns_terminal_record_by_task_id(
        self, tmp_path: Path,
    ) -> None:
        """Ring tier: a task_id= poll resolves to the ring's recorded terminal state."""
        ring = TerminalOutcomeRetention()
        ring.record(TerminalOutcomeRecord(
            request_id='mr-taskpoll',
            task_id='T-ltask',
            branch='branch-ltask',
            state='blocked',
        ))

        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = types.SimpleNamespace(_merge_worker=None, _terminal_retention=ring)
        server = create_server(esc_queue, harness=stub_harness)

        result = await _call_merge_status(server, task_id='T-ltask')

        assert result.get('state') == 'blocked', (
            f'Expected blocked from ring task_id lookup, got: {result}'
        )
        assert result.get('request_id') == 'mr-taskpoll', (
            f'Expected resolved record request_id, got: {result}'
        )

    async def test_ring_wins_over_event_store_by_branch(self, tmp_path: Path) -> None:
        """Ring branch= record wins over event store when both have the same branch."""
        event_store = EventStore(tmp_path / 'runs.db', 'run-ring-vs-ev-branch')
        event_store.emit(
            EventType.merge_finalized,
            task_id='T-rb',
            data={'request_id': 'mr-evbranch', 'branch': 'branch-ring-wins', 'state': 'done'},
        )

        # Ring says 'blocked' for this branch — ring must win over event store's 'done'
        ring = TerminalOutcomeRetention()
        ring.record(TerminalOutcomeRecord(
            request_id='mr-ringbranch',
            task_id='T-rb',
            branch='branch-ring-wins',
            state='blocked',
        ))

        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = types.SimpleNamespace(_merge_worker=None, _terminal_retention=ring)
        server = create_server(esc_queue, harness=stub_harness, event_store=event_store)

        result = await _call_merge_status(server, branch='branch-ring-wins')

        assert result.get('state') == 'blocked', (
            f'Expected ring value (blocked) to beat event store (done) on branch= poll, '
            f'got: {result}'
        )

    async def test_ring_tier_coalesced_id_via_alias(self, tmp_path: Path) -> None:
        """Coalesced id resolves to primary's terminal record via alias."""
        ring = TerminalOutcomeRetention()
        primary = TerminalOutcomeRecord(
            request_id='mr-primary-alias',
            task_id='T-alias',
            branch='branch-alias',
            state='done',
        )
        ring.record(primary)
        ring.record_alias('mr-coalesced-alias', 'mr-primary-alias')

        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = types.SimpleNamespace(_merge_worker=None, _terminal_retention=ring)
        server = create_server(esc_queue, harness=stub_harness)

        result = await _call_merge_status(server, request_id='mr-coalesced-alias')

        assert result.get('state') == 'done', (
            f'Expected done from coalesced-id alias resolution, got: {result}'
        )
        assert result.get('request_id') == 'mr-primary-alias', (
            f'Expected resolved primary request_id, got: {result}'
        )

    async def test_ring_request_id_over_branch_precedence(self, tmp_path: Path) -> None:
        """When request_id is supplied and misses the ring, branch is NOT consulted.

        The Tier-2 elif chain applies request_id > branch > task_id precedence:
        if request_id is given (even if it misses the ring), the branch key is
        never checked in the ring.  This pins the deliberate precedence choice
        so a regression that accidentally fell through to get_by_branch on a
        request_id miss would be caught.
        """
        # Ring has a record for branch 'prec-branch' but NOT for request_id 'mr-miss'
        ring = TerminalOutcomeRetention()
        ring.record(TerminalOutcomeRecord(
            request_id='mr-other-prec',
            task_id='T-prec',
            branch='prec-branch',
            state='done',
        ))

        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = types.SimpleNamespace(_merge_worker=None, _terminal_retention=ring)
        server = create_server(esc_queue, harness=stub_harness)

        # Supply both request_id (ring-miss) and branch (ring-hit).
        # The ring tier must NOT resolve via branch because request_id is present.
        result = await _call_merge_status(server, request_id='mr-miss', branch='prec-branch')

        # Should fall through to unknown (not 'done' from the ring's branch record)
        assert result.get('state') == 'unknown', (
            f'Expected unknown (request_id-only lookup; branch ignored on ring-miss), '
            f'got: {result}'
        )

    async def test_ring_wins_over_event_store_by_task_id(self, tmp_path: Path) -> None:
        """Ring task_id= record wins over event store when both have the same task_id."""
        event_store = EventStore(tmp_path / 'runs.db', 'run-ring-vs-ev-task')
        event_store.emit(
            EventType.merge_finalized,
            task_id='T-rtask',
            data={'request_id': 'mr-evtask', 'branch': 'b-rtask', 'state': 'done'},
        )

        # Ring says 'conflict' for this task_id — ring must win over event store's 'done'
        ring = TerminalOutcomeRetention()
        ring.record(TerminalOutcomeRecord(
            request_id='mr-ringtask',
            task_id='T-rtask',
            branch='b-rtask',
            state='conflict',
        ))

        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = types.SimpleNamespace(_merge_worker=None, _terminal_retention=ring)
        server = create_server(esc_queue, harness=stub_harness, event_store=event_store)

        result = await _call_merge_status(server, task_id='T-rtask')

        assert result.get('state') == 'conflict', (
            f'Expected ring value (conflict) to beat event store (done) on task_id= poll, '
            f'got: {result}'
        )
        assert result.get('outcome') == 'conflict', (
            f'Expected outcome=conflict from ring, got: {result}'
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
            task_id='T-live', branch=QueuedBranch.parse('branch-live', 'task/'),
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
            task_id='T-lbranch', branch=QueuedBranch.parse('branch-bylookup', 'task/'),
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
            task_id='T-ltask', branch=QueuedBranch.parse('branch-ltask', 'task/'),
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
            task_id='T-prec', branch=QueuedBranch.parse('branch-prec', 'task/'),
            worktree=tmp_path / 'wt', pre_rebased=False, task_files=None,
            module_configs=[], config=config, result=loop.create_future(),
        )
        await mq.put(req)

        # Ring says 'done'
        ring = TerminalOutcomeRetention()
        ring.record(TerminalOutcomeRecord(
            request_id=req.request_id,
            task_id=req.task_id,
            branch=req.branch.bare_id,
            state='done',
        ))

        # Event store also says 'done'
        event_store = EventStore(tmp_path / 'runs.db', 'run-prec')
        event_store.emit(
            EventType.merge_finalized,
            task_id=req.task_id,
            data={'request_id': req.request_id, 'branch': req.branch.bare_id, 'state': 'done'},
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
            task_id='T-phase', branch=QueuedBranch.parse('branch-phase', 'task/'),
            worktree=tmp_path / 'wt', pre_rebased=False, task_files=None,
            module_configs=[], config=config, result=loop.create_future(),
        )

        from orchestrator.git_ops import MergeResult  # type: ignore[reportMissingImports]
        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            InflightEntry,
            ItemLifecycleState,
        )

        if verify_phase == 'queued':
            await mq.put(req)
        else:
            # Put req through the verify item path
            merge_wt = tmp_path / 'merge-wt'
            merge_wt.mkdir()
            merge_result = MergeResult(success=True, merge_commit='deadbeefP0000000', merge_worktree=merge_wt)
            item = RealMergeItem(
                request=req,
                merge_result=merge_result, merge_wt=merge_wt,
                base_sha='base', speculative=False,
            )
            if verify_phase in ('merging',):
                worker._register_item(req, initial=ItemLifecycleState.MERGING)
            elif verify_phase == 'awaiting_verify':
                await worker._verifier_queue.put(item)
            else:
                # verifying / gate_reverify / finalizing: the entry lives in
                # worker._inflight (the singular _verify_item/_verify_phase
                # fields were retired in task 1736; the free-form
                # InflightEntry.phase field was retired in task lambda / 2173).
                # snapshot() derives the entry 'state' from the ItemLifecycle
                # registry via _entry_phase(), which the server maps through
                # _map_live_state.
                worker._inflight.append(InflightEntry(
                    item=item,
                    lease=None,
                    verify_task=None,
                    merge_wt=merge_wt,
                    was_speculative=False,
                ))
                # Phase now DERIVES from the ItemLifecycle registry
                # (task lambda / 2173 deleted InflightEntry.phase) — drive the
                # registry to the parametrized verifying/gate_reverify/finalizing
                # state so snapshot()/_entry_phase surface it.
                worker._register_item(item, initial=ItemLifecycleState.VERIFYING)
                if verify_phase == 'gate_reverify':
                    worker._note_transition(
                        req.request_id,
                        ItemLifecycleState.VERIFYING, ItemLifecycleState.GATE_REVERIFY,
                    )
                elif verify_phase == 'finalizing':
                    worker._note_transition(
                        req.request_id,
                        ItemLifecycleState.VERIFYING, ItemLifecycleState.FINALIZING,
                    )

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


def _seed_resolved_orphan(tmp_path: Path) -> EscalationQueue:
    """Create an EscalationQueue and seed a resolved orphan file directly.

    Seeds 'esc-9-9' with resolved_at='2026-05-20T10:00:00+00:00' directly to
    disk (NOT via queue.submit_resolved, which would auto-archive it via
    _archive_resolved and defeat startup-sweep tests).  Returns the queue with
    the orphan already in place.
    """
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
    return queue


class TestCreateServerStartupSweep:
    """create_server runs run_startup_sweep on construction when startup_sweep=True."""

    @pytest.mark.parametrize(
        'startup_sweep_now, expect_pruned',
        [
            (datetime(2026, 6, 4, tzinfo=UTC), False),
            (datetime(2026, 8, 1, tzinfo=UTC), True),
        ],
        ids=['survive', 'prune'],
    )
    def test_startup_sweep_prune_direction(
        self,
        tmp_path: Path,
        caplog,
        startup_sweep_now: datetime,
        expect_pruned: bool,
    ):
        """Pinned now controls whether the freshly-archived dir survives or is pruned.

        survive: now=2026-06-04 → cutoff=2026-05-05 → 2026-05-20 > cutoff → NOT pruned
        prune:   now=2026-08-01 → cutoff=2026-07-02 → 2026-05-20 < cutoff → pruned

        The prune variant proves the injected now genuinely reaches prune_archive
        through create_server — guards against a future refactor that accepts
        startup_sweep_now but drops the threading.
        """
        import logging

        queue = _seed_resolved_orphan(tmp_path)

        with caplog.at_level(logging.INFO, logger='escalation.sweep'):
            create_server(queue, startup_sweep_now=startup_sweep_now)

        # Orphan always swept from root
        assert not (queue.queue_dir / 'esc-9-9.json').exists()

        archive_dir = queue.queue_dir / 'archive' / '2026-05-20'
        if expect_pruned:
            assert not archive_dir.exists(), (
                'Expected archive/2026-05-20 to be pruned at construction '
                '(now far past retention window)'
            )
        else:
            assert (archive_dir / 'esc-9-9.json').exists(), (
                f'Expected orphan archived at {archive_dir / "esc-9-9.json"}; '
                f'still in root: {(queue.queue_dir / "esc-9-9.json").exists()}'
            )

        # Sweep log always emitted — confirms run_startup_sweep ran in both cases
        assert any(
            r.name == 'escalation.sweep' and r.levelno == logging.INFO
            for r in caplog.records
        ), f'Expected INFO sweep report; got: {[r.getMessage() for r in caplog.records]}'

    def test_startup_sweep_false_leaves_orphan_untouched(self, tmp_path: Path):
        """(b) create_server(startup_sweep=False) leaves a pre-seeded orphan in root."""
        queue = _seed_resolved_orphan(tmp_path)

        create_server(queue, startup_sweep=False)

        # File still in root — startup sweep was skipped
        assert (queue.queue_dir / 'esc-9-9.json').exists(), (
            'Orphan was archived even with startup_sweep=False'
        )
        assert not (queue.queue_dir / 'archive').exists()


async def _call_tool(server: Any, name: str, **kwargs: Any) -> Any:
    """Invoke an async MCP tool by name.

    *server* is intentionally typed ``Any`` so pyright does not reach into
    FastMCP's ``Tool`` internals (``.fn``) — mirroring the ``_blocker`` /
    ``_info`` / ``_call_merge_status`` helpers above, which the type-check gate
    accepts for exactly this reason.
    """
    tool = await server.get_tool(name)
    return await tool.fn(**kwargs)


@pytest.mark.asyncio
class TestOperatorHaltTools:
    """halt_merge_queue / halt_scheduler — operator-only halt-direction tools.

    Siblings of unhalt_merge_queue / resume_scheduler.  Each guards the
    standalone (no-harness) case, requires a non-empty reason, and forwards the
    stripped reason to the harness.  Restriction from autonomous agents is by
    allow-list omission (see test_roles_operator_tools.py), not a server gate.
    """

    async def test_tools_registered(self, tmp_path: Path) -> None:
        queue = EscalationQueue(tmp_path / 'esc')
        server: Any = create_server(queue)
        assert await server.get_tool('halt_merge_queue') is not None
        assert await server.get_tool('halt_scheduler') is not None

    # ── halt_merge_queue ──────────────────────────────────────────────────

    async def test_halt_merge_queue_standalone_returns_wired_error(
        self, tmp_path: Path,
    ) -> None:
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)  # no harness wired
        result = await _call_tool(server, 'halt_merge_queue', reason='bad main')
        assert result['halted'] is False
        assert 'standalone' in result['error']

    async def test_halt_merge_queue_forwards_stripped_reason(
        self, tmp_path: Path,
    ) -> None:
        queue = EscalationQueue(tmp_path / 'esc')
        calls: list[str] = []

        def _halt(reason: str) -> dict[str, Any]:
            calls.append(reason)
            return {'halted': True, 'reason': reason}

        stub_harness = types.SimpleNamespace(halt_merge_queue=_halt)
        server = create_server(queue, harness=stub_harness)

        result = await _call_tool(server, 'halt_merge_queue', reason='  infra incident  ')

        assert calls == ['infra incident'], 'reason must be forwarded stripped'
        assert result == {'halted': True, 'reason': 'infra incident'}

    async def test_halt_merge_queue_rejects_empty_reason(
        self, tmp_path: Path,
    ) -> None:
        queue = EscalationQueue(tmp_path / 'esc')
        called: list[str] = []
        stub_harness = types.SimpleNamespace(
            halt_merge_queue=lambda r: called.append(r),
        )
        server = create_server(queue, harness=stub_harness)

        result = await _call_tool(server, 'halt_merge_queue', reason='   ')

        assert result['halted'] is False
        assert 'reason' in result['error']
        assert called == [], 'harness must NOT be invoked on an empty reason'

    # ── halt_scheduler ────────────────────────────────────────────────────

    async def test_halt_scheduler_standalone_returns_wired_error(
        self, tmp_path: Path,
    ) -> None:
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)  # no harness wired
        result = await _call_tool(server, 'halt_scheduler', reason='runaway')
        assert result['halted'] is False
        assert 'standalone' in result['error']

    async def test_halt_scheduler_forwards_stripped_reason(
        self, tmp_path: Path,
    ) -> None:
        queue = EscalationQueue(tmp_path / 'esc')
        calls: list[str] = []

        async def _force_halt(reason: str) -> dict[str, Any]:
            calls.append(reason)
            return {
                'halted': True, 'was_paused': False,
                'prior_reason': None, 'reason': reason,
            }

        stub_harness = types.SimpleNamespace(force_halt_scheduler=_force_halt)
        server = create_server(queue, harness=stub_harness)

        result = await _call_tool(server, 'halt_scheduler', reason='  bad deploy  ')

        assert calls == ['bad deploy'], 'reason must be forwarded stripped'
        assert result['halted'] is True
        assert result['reason'] == 'bad deploy'

    async def test_halt_scheduler_rejects_empty_reason(
        self, tmp_path: Path,
    ) -> None:
        queue = EscalationQueue(tmp_path / 'esc')
        called: list[str] = []

        async def _force_halt(reason: str) -> dict[str, Any]:
            called.append(reason)
            return {}

        stub_harness = types.SimpleNamespace(force_halt_scheduler=_force_halt)
        server = create_server(queue, harness=stub_harness)

        result = await _call_tool(server, 'halt_scheduler', reason='')

        assert result['halted'] is False
        assert 'reason' in result['error']
        assert called == [], 'harness must NOT be invoked on an empty reason'


@pytest.mark.asyncio
class TestReloadConfigTool:
    """reload_config — operator-only config hot-reload trigger (task 2007, PRD gamma).

    Unlike halt_scheduler / halt_merge_queue, this tool takes NO parameters: it
    always re-reads the process's own ORCH_CONFIG_PATH, so a reload can never
    retarget the orchestrator at another project.  It is a thin standalone-guard
    + verbatim delegate to ``harness.reload_config()`` (task 2006), which does
    the actual load/apply/audit work.  Restriction from autonomous agents is by
    allow-list omission (see test_roles_operator_tools.py), not a server gate.
    """

    async def test_tool_registered(self, tmp_path: Path) -> None:
        queue = EscalationQueue(tmp_path / 'esc')
        server: Any = create_server(queue)
        assert await server.get_tool('reload_config') is not None

    async def test_standalone_returns_wired_error(self, tmp_path: Path) -> None:
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)  # no harness wired

        result = await _call_tool(server, 'reload_config')

        assert result['reloaded'] is False
        assert 'standalone' in result['error']
        assert 'no harness wired' in result['error']

    async def test_wired_path_returns_report_verbatim(self, tmp_path: Path) -> None:
        queue = EscalationQueue(tmp_path / 'esc')
        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        canned_report = {
            'reloaded': True,
            'config_path': '/x/orchestrator.yaml',
            'applied': {'some_field': 'new_value'},
            'restart_required': {},
            'unchanged': 3,
            'error': None,
        }

        async def _reload_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append((args, kwargs))
            return canned_report

        stub_harness = types.SimpleNamespace(reload_config=_reload_config)
        server = create_server(queue, harness=stub_harness)

        result = await _call_tool(server, 'reload_config')

        assert result == canned_report, 'must return the harness report verbatim'
        assert calls == [((), {})], 'harness.reload_config must be awaited exactly once with no arguments'


# ---------------------------------------------------------------------------
# TestEscalateEvidenceParam: escalate_blocker/escalate_info structured evidence
# ---------------------------------------------------------------------------


class TestEscalateEvidenceParam:
    """escalate_blocker/escalate_info accept an optional `evidence` list stored verbatim (task 2558)."""

    _EVIDENCE: list[dict[str, Any]] = [
        {'observation': 'exit code 134', 'measured_at': 'sha=abc123', 'ref': 'rerun#2'},
    ]

    @pytest.mark.asyncio
    async def test_blocker_stores_evidence_verbatim(self, tmp_path: Path):
        """escalate_blocker(evidence=[...]) persists an escalation whose evidence == that list verbatim."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(server, evidence=self._EVIDENCE, **_COMMON_KWARGS)

        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.evidence == self._EVIDENCE, f"Expected evidence stored verbatim, got: {esc.evidence!r}"

    @pytest.mark.asyncio
    async def test_info_stores_evidence_verbatim(self, tmp_path: Path):
        """escalate_info(evidence=[...]) persists an escalation whose evidence == that list verbatim."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _info(server, evidence=self._EVIDENCE, **_COMMON_KWARGS)

        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.evidence == self._EVIDENCE, f"Expected evidence stored verbatim, got: {esc.evidence!r}"

    @pytest.mark.asyncio
    async def test_blocker_without_evidence_defaults_empty(self, tmp_path: Path):
        """escalate_blocker() with no evidence kwarg → on-disk evidence == [] (free-form callers unaffected)."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(server, **_COMMON_KWARGS)

        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.evidence == [], f"Expected evidence=[], got: {esc.evidence!r}"

    @pytest.mark.asyncio
    async def test_info_without_evidence_defaults_empty(self, tmp_path: Path):
        """escalate_info() with no evidence kwarg → on-disk evidence == [] (free-form callers unaffected)."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _info(server, **_COMMON_KWARGS)

        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.evidence == [], f"Expected evidence=[], got: {esc.evidence!r}"


class TestResolveIssueEscalateModel:
    """resolve_issue(escalate_model=True) pre-bumps the task's routing tier via
    harness.pre_increment_routing_tier (task μ, trigger 3) — so the task's NEXT
    dispatch routes one ladder rung stronger via the retry-tier-up rule.

    Fires ONLY for resume/restart (the actions that lead to a next dispatch);
    park keeps the task blocked with no re-dispatch (early return before the
    hook), so it never bumps. Best-effort + off-loop: the write is delegated to
    the harness (which owns the metadata write path and the loop); resolve_issue
    stays sync.
    """

    def _seed_pending(self, queue: EscalationQueue, esc_id: str = 'esc-em-0001') -> Escalation:
        esc = Escalation(
            id=esc_id,
            task_id='t-em-1',
            agent_role='implementer',
            severity='blocking',
            category='scope_violation',
            summary='escalate_model test escalation',
        )
        queue.submit(esc)
        return esc

    @staticmethod
    def _harness():
        from unittest.mock import Mock
        return types.SimpleNamespace(pre_increment_routing_tier=Mock())

    @pytest.mark.asyncio
    async def test_resume_with_flag_bumps(self, tmp_path: Path):
        queue = EscalationQueue(tmp_path / 'esc')
        harness = self._harness()
        server = create_server(queue, harness=harness)
        esc = self._seed_pending(queue)

        result = await _resolve_issue(
            server, escalation_id=esc.id, resolution='fixed',
            action='resume', escalate_model=True,
        )

        assert 'error' not in result, f'Unexpected error: {result}'
        harness.pre_increment_routing_tier.assert_called_once_with(esc.task_id)

    @pytest.mark.asyncio
    async def test_restart_with_flag_bumps(self, tmp_path: Path):
        queue = EscalationQueue(tmp_path / 'esc')
        harness = self._harness()
        server = create_server(queue, harness=harness)
        esc = self._seed_pending(queue)

        result = await _resolve_issue(
            server, escalation_id=esc.id, resolution='fixed',
            action='restart', escalate_model=True,
        )

        assert 'error' not in result, f'Unexpected error: {result}'
        harness.pre_increment_routing_tier.assert_called_once_with(esc.task_id)

    @pytest.mark.asyncio
    async def test_resume_without_flag_does_not_bump(self, tmp_path: Path):
        queue = EscalationQueue(tmp_path / 'esc')
        harness = self._harness()
        server = create_server(queue, harness=harness)
        esc = self._seed_pending(queue)

        result = await _resolve_issue(
            server, escalation_id=esc.id, resolution='fixed',
            action='resume', escalate_model=False,
        )

        assert 'error' not in result, f'Unexpected error: {result}'
        harness.pre_increment_routing_tier.assert_not_called()

    @pytest.mark.asyncio
    async def test_park_with_flag_does_not_bump(self, tmp_path: Path):
        """park has no next dispatch (task stays blocked) — the hook sits after
        park's early return, so escalate_model is a no-op there."""
        queue = EscalationQueue(tmp_path / 'esc')
        harness = self._harness()
        server = create_server(queue, harness=harness)
        esc = self._seed_pending(queue)

        await _resolve_issue(
            server, escalation_id=esc.id, resolution='needs human',
            action='park', escalate_model=True,
        )

        harness.pre_increment_routing_tier.assert_not_called()


# ---------------------------------------------------------------------------
# TestEscalateBlockerLevelParam: explicit `level` argument (task 3236)
# ---------------------------------------------------------------------------


class TestEscalateBlockerLevelParam:
    """escalate_blocker accepts an explicit ``level``, restricted to {0, 1}.

    roles.py instructs the steward to "re-escalate to level-1 via
    escalate_blocker" in several places, but the tool had no ``level``
    parameter: every agent filing landed at level=0, where the auto-watcher
    (an L1 consumer filtering on level) never reads it and the level=0-scoped
    workflow sweeps are entitled to dismiss it.  These tests pin the
    parameter's runtime behaviour, reading the persisted record back through
    the queue rather than trusting the response.
    """

    @pytest.mark.asyncio
    async def test_level1_persists_level1(self, tmp_path: Path):
        """escalate_blocker(level=1) → the ON-DISK record has level == 1."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(server, level=1, **_COMMON_KWARGS)

        assert 'error' not in result, f'Unexpected error: {result}'
        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.level == 1, f'Expected on-disk level==1, got: {esc.level}'

    @pytest.mark.asyncio
    async def test_default_still_persists_level0(self, tmp_path: Path):
        """No ``level`` passed → level == 0, so every existing caller is unchanged."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(server, **_COMMON_KWARGS)

        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.level == 0, f'Expected on-disk level==0, got: {esc.level}'

    @pytest.mark.asyncio
    async def test_explicit_level0_persists_level0(self, tmp_path: Path):
        """An explicit level=0 is accepted and behaves like the default."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(server, level=0, **_COMMON_KWARGS)

        assert 'error' not in result, f'Unexpected error: {result}'
        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.level == 0, f'Expected on-disk level==0, got: {esc.level}'

    @pytest.mark.asyncio
    async def test_level2_rejected_and_nothing_submitted(self, tmp_path: Path):
        """level=2 is rejected — agents must not self-mint L2 — and nothing is written.

        Mirrors the existing agent-role severity downgrade policy: the
        legitimate routes to L2 are a born-at-L2 severity from a harness
        sentinel role, or promote_to_l2.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(server, level=2, **_COMMON_KWARGS)

        assert 'error' in result, f'Expected an error response, got: {result}'
        assert 'id' not in result, f'Nothing should be submitted, got: {result}'
        assert queue.get_pending() == [], 'level=2 must not submit any record'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('bad_level', [-1, 3, 99])
    async def test_out_of_range_level_rejected(self, tmp_path: Path, bad_level: int):
        """An out-of-range level is rejected with {'error': ...} and submits nothing."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(server, level=bad_level, **_COMMON_KWARGS)

        assert 'error' in result, f'level={bad_level}: expected error, got: {result}'
        assert queue.get_pending() == [], f'level={bad_level} must not submit any record'

    @pytest.mark.asyncio
    async def test_born_at_l2_severity_keeps_precedence_over_level1(self, tmp_path: Path):
        """A sentinel-filed critical still persists level==2 even when level=1 is passed.

        The born-at-L2 stamp runs inside _chokepoint_or_submit, i.e. AFTER
        construction, so ``esc.level = 2`` naturally overrides the explicitly
        passed level.  Pinning it keeps that ordering dependency explicit.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(
            server, level=1, severity='critical',
            **{**_COMMON_KWARGS, 'agent_role': 'orchestrator-watcher-supervisor'},
        )

        assert 'error' not in result, f'Unexpected error: {result}'
        esc = queue.get(result['id'])
        assert esc is not None
        assert esc.level == 2, f'Expected born-at-L2 to win, got level={esc.level}'
