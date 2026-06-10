"""Tests for terminal-task chokepoint in escalation MCP handlers.

Exercises the auto-resolve path: when the target task is already 'done' or
'cancelled', escalate_blocker / escalate_info should auto-resolve the
escalation immediately rather than leaving it pending.

Uses the same async FastMCP unit-test pattern as test_release_workflow.py:
    tool = await server.get_tool('escalate_blocker')
    result = await tool.fn(...)

and the same tmp_path isolation as test_server_dedupe.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import types
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
    'summary': 'target task already done',
}


async def _make_lookup(status: str | None):
    """Build a simple async stub that always returns *status*."""

    async def _lookup(task_id: str) -> str | None:
        return status

    return _lookup


async def _blocker(server, **kwargs: Any) -> dict[str, Any]:
    tool = await server.get_tool('escalate_blocker')
    return await tool.fn(**kwargs)


async def _info(server, **kwargs: Any) -> dict[str, Any]:
    tool = await server.get_tool('escalate_info')
    return await tool.fn(**kwargs)


# ---------------------------------------------------------------------------
# Step-1 RED tests: terminal auto-resolve (done / cancelled)
# ---------------------------------------------------------------------------


class TestTerminalAutoResolve:
    """escalate_blocker / escalate_info auto-resolve when task is terminal."""

    @pytest.mark.asyncio
    async def test_blocker_done_task_returns_resolved(self, tmp_path: Path):
        """escalate_blocker against a 'done' task returns status='resolved'."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('done')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _blocker(server, **_COMMON_KWARGS)

        assert result['status'] == 'resolved', f"Expected 'resolved', got: {result}"

    @pytest.mark.asyncio
    async def test_blocker_done_task_resolution_text(self, tmp_path: Path):
        """escalate_blocker response includes auto-resolve resolution text."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('done')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _blocker(server, **_COMMON_KWARGS)

        assert 'resolution' in result, f"No 'resolution' key in: {result}"
        assert 'auto-resolved' in result['resolution'], f"Unexpected resolution: {result['resolution']}"
        assert 'done' in result['resolution'], f"Status not in resolution: {result['resolution']}"

    @pytest.mark.asyncio
    async def test_blocker_done_task_resolved_by(self, tmp_path: Path):
        """escalate_blocker response includes resolved_by='escalation-mcp-pre-submit-check'."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('done')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _blocker(server, **_COMMON_KWARGS)

        assert result.get('resolved_by') == 'escalation-mcp-pre-submit-check', (
            f"Expected 'escalation-mcp-pre-submit-check', got: {result.get('resolved_by')}"
        )

    @pytest.mark.asyncio
    async def test_blocker_done_task_action_terminate(self, tmp_path: Path):
        """escalate_blocker response retains action='terminate_cleanly' on auto-resolve path."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('done')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _blocker(server, **_COMMON_KWARGS)

        assert result.get('action') == 'terminate_cleanly', (
            f"Expected 'terminate_cleanly', got: {result.get('action')}"
        )

    @pytest.mark.asyncio
    async def test_blocker_done_task_on_disk_resolved(self, tmp_path: Path):
        """After auto-resolve, the escalation file on disk has status='resolved'."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('done')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _blocker(server, **_COMMON_KWARGS)

        esc_id = result['id']
        esc = queue.get(esc_id)
        assert esc is not None, f"Escalation {esc_id} not found in queue"
        assert esc.status == 'resolved', f"Expected 'resolved', got: {esc.status}"

    @pytest.mark.asyncio
    async def test_info_cancelled_task_returns_resolved(self, tmp_path: Path):
        """escalate_info against a 'cancelled' task returns status='resolved'."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('cancelled')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _info(server, **_COMMON_KWARGS)

        assert result['status'] == 'resolved', f"Expected 'resolved', got: {result}"

    @pytest.mark.asyncio
    async def test_info_cancelled_task_resolution_text(self, tmp_path: Path):
        """escalate_info response includes 'cancelled' in resolution text."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('cancelled')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _info(server, **_COMMON_KWARGS)

        assert 'resolution' in result
        assert 'auto-resolved' in result['resolution']
        assert 'cancelled' in result['resolution']

    @pytest.mark.asyncio
    async def test_info_cancelled_task_resolved_by(self, tmp_path: Path):
        """escalate_info response includes resolved_by='escalation-mcp-pre-submit-check'."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('cancelled')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _info(server, **_COMMON_KWARGS)

        assert result.get('resolved_by') == 'escalation-mcp-pre-submit-check'

    @pytest.mark.asyncio
    async def test_info_has_no_action_key(self, tmp_path: Path):
        """escalate_info response must NOT include 'action' key (blocker-only contract)."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('cancelled')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _info(server, **_COMMON_KWARGS)

        assert 'action' not in result, f"Unexpected 'action' key in info result: {result}"

    @pytest.mark.asyncio
    async def test_info_cancelled_task_on_disk_resolved(self, tmp_path: Path):
        """After auto-resolve via escalate_info, the on-disk record is resolved."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('cancelled')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _info(server, **_COMMON_KWARGS)

        esc_id = result['id']
        esc = queue.get(esc_id)
        assert esc is not None
        assert esc.status == 'resolved'

    # --- Item 4: dedupe-vs-terminal gate-priority characterization ---

    @pytest.mark.asyncio
    async def test_dedupe_eligible_terminal_task_auto_resolves(self, tmp_path: Path):
        """dedupe-eligible infra_issue + terminal task → auto-resolve wins over dedupe.

        Characterization test: the explicit _submit_or_dedupe bypass at
        server.py:153-155 (inside the status-in-done/cancelled branch) ensures
        that even when DedupeConfig is active and a matching pending parent exists,
        Gate 4 auto-resolve fires first and the child is NOT folded into the parent.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('done')
        server = create_server(queue, dedupe_config=DedupeConfig(), task_status_lookup=lookup)

        # Pre-seed a pending infra_issue parent for the same task
        parent = Escalation(
            id=queue.make_id('task-999'),
            task_id='task-999',
            agent_role='implementer',
            severity='info',
            category='infra_issue',
            summary='parent infra issue',
        )
        queue.submit(parent)

        # Child call: same category, same task_id — would dedupe if task were not terminal
        result = await _info(
            server,
            task_id='task-999',
            agent_role='implementer',
            category='infra_issue',
            summary='child infra issue similar',
        )

        assert result['status'] == 'resolved', (
            f"Expected 'resolved' (auto-resolve wins over dedupe), got: {result}"
        )

    @pytest.mark.asyncio
    async def test_dedupe_eligible_terminal_task_parent_unchanged(self, tmp_path: Path):
        """dedupe-eligible + terminal → pre-seeded parent's dedupe_count stays 0 (not folded).

        Characterization test: the bypass ensures the child is auto-resolved via
        submit_resolved (or submit+resolve), never routed through _submit_or_dedupe,
        so the parent's dedupe_children and dedupe_count remain at their initial values.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('done')
        server = create_server(queue, dedupe_config=DedupeConfig(), task_status_lookup=lookup)

        parent = Escalation(
            id=queue.make_id('task-999'),
            task_id='task-999',
            agent_role='implementer',
            severity='info',
            category='infra_issue',
            summary='parent infra issue',
        )
        queue.submit(parent)
        parent_id = parent.id

        await _info(
            server,
            task_id='task-999',
            agent_role='implementer',
            category='infra_issue',
            summary='child infra issue similar',
        )

        # Parent must be unmodified — child was NOT folded into it
        updated_parent = queue.get(parent_id)
        assert updated_parent is not None, f"Parent {parent_id} not found after call"
        assert updated_parent.dedupe_count == 0, (
            f"Expected dedupe_count==0 (auto-resolve bypassed dedupe), "
            f"got: {updated_parent.dedupe_count}"
        )
        assert len(updated_parent.dedupe_children) == 0, (
            f"Expected no dedupe_children (auto-resolve bypassed dedupe), "
            f"got: {updated_parent.dedupe_children}"
        )

    # --- Item 2: minimal-key-set contract (RED until step-2 impl) ---

    @pytest.mark.asyncio
    async def test_blocker_response_has_only_minimal_keys(self, tmp_path: Path):
        """escalate_blocker auto-resolve returns exactly {id,status,resolution,resolved_by,action}.

        RED on current code: the auto-resolve path returns (resolved or esc).to_dict()
        which is a 20-field Escalation dump. After step-2, the shape is normalized to
        the minimal four-field contract plus 'action' from the blocker wrapper.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('done')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _blocker(server, **_COMMON_KWARGS)

        expected_keys = {'id', 'status', 'resolution', 'resolved_by', 'action'}
        assert set(result.keys()) == expected_keys, (
            f"Expected minimal keys {expected_keys}, got: {set(result.keys())}"
        )

    @pytest.mark.asyncio
    async def test_info_response_has_only_minimal_keys(self, tmp_path: Path):
        """escalate_info auto-resolve returns exactly {id,status,resolution,resolved_by} (no 'action').

        RED on current code: the auto-resolve path returns (resolved or esc).to_dict()
        which is a 20-field dump. After step-2, the shape is normalized to the minimal
        four-field contract — no 'action' key (that is blocker-only).
        """
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('cancelled')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _info(server, **_COMMON_KWARGS)

        expected_keys = {'id', 'status', 'resolution', 'resolved_by'}
        assert set(result.keys()) == expected_keys, (
            f"Expected minimal keys {expected_keys} (no 'action'), got: {set(result.keys())}"
        )


# ---------------------------------------------------------------------------
# Step-3 characterization tests: non-terminal / disabled → kept open (queued)
# ---------------------------------------------------------------------------


class TestKeptOpenPaths:
    """Non-terminal statuses and disabled lookup → escalation stays queued."""

    @pytest.mark.asyncio
    async def test_deferred_status_stays_queued(self, tmp_path: Path):
        """task status 'deferred' → escalation is NOT auto-resolved (status='queued')."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('deferred')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _blocker(server, **_COMMON_KWARGS)

        assert result['status'] == 'queued', f"Expected 'queued', got: {result['status']}"

    @pytest.mark.asyncio
    async def test_blocked_status_stays_queued(self, tmp_path: Path):
        """task status 'blocked' → escalation stays queued (not auto-resolved)."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('blocked')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _blocker(server, **_COMMON_KWARGS)

        assert result['status'] == 'queued'

    @pytest.mark.asyncio
    async def test_in_progress_status_stays_queued(self, tmp_path: Path):
        """task status 'in-progress' → escalation stays queued."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('in-progress')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _blocker(server, **_COMMON_KWARGS)

        assert result['status'] == 'queued'

    @pytest.mark.asyncio
    async def test_pending_status_stays_queued(self, tmp_path: Path):
        """task status 'pending' → escalation stays queued."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('pending')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _info(server, **_COMMON_KWARGS)

        assert result['status'] == 'queued'

    @pytest.mark.asyncio
    async def test_none_status_stays_queued(self, tmp_path: Path):
        """task_status_lookup returning None → fail-open, escalation stays queued."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup(None)
        server = create_server(queue, task_status_lookup=lookup)

        result = await _blocker(server, **_COMMON_KWARGS)

        assert result['status'] == 'queued'

    @pytest.mark.asyncio
    async def test_no_lookup_stays_queued(self, tmp_path: Path):
        """create_server without task_status_lookup → chokepoint disabled, stays queued."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)  # no task_status_lookup

        result = await _blocker(server, **_COMMON_KWARGS)

        assert result['status'] == 'queued'

    @pytest.mark.asyncio
    async def test_non_terminal_no_resolved_record(self, tmp_path: Path):
        """Non-terminal status → no resolved record in queue (pending file stays pending)."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('in-progress')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _blocker(server, **_COMMON_KWARGS)

        esc_id = result['id']
        esc = queue.get(esc_id)
        assert esc is not None
        assert esc.status == 'pending', f"Expected 'pending', got: {esc.status}"

    @pytest.mark.asyncio
    async def test_lookup_raises_stays_queued(self, tmp_path: Path):
        """lookup raising an exception → fail-open, escalation stays queued."""
        queue = EscalationQueue(tmp_path / 'esc')

        async def _raising_lookup(task_id: str) -> str:
            raise RuntimeError('scheduler unavailable')

        server = create_server(queue, task_status_lookup=_raising_lookup)

        result = await _blocker(server, **_COMMON_KWARGS)

        assert result['status'] == 'queued', f"Expected fail-open 'queued', got: {result['status']}"


# ---------------------------------------------------------------------------
# Step-5 RED tests: bypass gates (terminal_state_is_the_bug + review_suggestions)
# ---------------------------------------------------------------------------


class TestBypassGates:
    """Two bypass gates skip the auto-resolve chokepoint entirely."""

    @pytest.mark.asyncio
    async def test_terminal_state_is_the_bug_skips_auto_resolve(self, tmp_path: Path):
        """escalate_blocker with terminal_state_is_the_bug=True → NOT auto-resolved (queued)."""
        queue = EscalationQueue(tmp_path / 'esc')
        spy_calls: list[str] = []

        async def _spy_lookup(task_id: str) -> str:
            spy_calls.append(task_id)
            return 'done'

        server = create_server(queue, task_status_lookup=_spy_lookup)

        result = await _blocker(
            server,
            terminal_state_is_the_bug=True,
            **_COMMON_KWARGS,
        )

        assert result['status'] == 'queued', (
            f"Expected 'queued' (bypass), got: {result['status']}"
        )

    @pytest.mark.asyncio
    async def test_terminal_state_is_the_bug_lookup_not_called(self, tmp_path: Path):
        """With terminal_state_is_the_bug=True, the lookup spy is NOT consulted."""
        queue = EscalationQueue(tmp_path / 'esc')
        spy_calls: list[str] = []

        async def _spy_lookup(task_id: str) -> str:
            spy_calls.append(task_id)
            return 'done'

        server = create_server(queue, task_status_lookup=_spy_lookup)

        await _blocker(server, terminal_state_is_the_bug=True, **_COMMON_KWARGS)

        assert spy_calls == [], f"Lookup was unexpectedly called: {spy_calls}"

    @pytest.mark.asyncio
    async def test_terminal_state_is_the_bug_open_on_disk(self, tmp_path: Path):
        """With terminal_state_is_the_bug=True, escalation file on disk is 'pending'."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('done')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _blocker(server, terminal_state_is_the_bug=True, **_COMMON_KWARGS)

        esc_id = result['id']
        esc = queue.get(esc_id)
        assert esc is not None
        assert esc.status == 'pending', f"Expected 'pending', got: {esc.status}"

    @pytest.mark.asyncio
    async def test_review_suggestions_category_skips_auto_resolve(self, tmp_path: Path):
        """escalate_info with category='review_suggestions' → NOT auto-resolved (queued)."""
        queue = EscalationQueue(tmp_path / 'esc')
        spy_calls: list[str] = []

        async def _spy_lookup(task_id: str) -> str:
            spy_calls.append(task_id)
            return 'done'

        server = create_server(queue, task_status_lookup=_spy_lookup)

        result = await _info(
            server,
            task_id='task-999',
            agent_role='implementer',
            category='review_suggestions',
            summary='review suggestions bypass test',
        )

        assert result['status'] == 'queued', (
            f"Expected 'queued' (bypass), got: {result['status']}"
        )

    @pytest.mark.asyncio
    async def test_review_suggestions_lookup_not_called(self, tmp_path: Path):
        """With category='review_suggestions', the lookup spy is NOT consulted."""
        queue = EscalationQueue(tmp_path / 'esc')
        spy_calls: list[str] = []

        async def _spy_lookup(task_id: str) -> str:
            spy_calls.append(task_id)
            return 'done'

        server = create_server(queue, task_status_lookup=_spy_lookup)

        await _info(
            server,
            task_id='task-999',
            agent_role='implementer',
            category='review_suggestions',
            summary='review suggestions bypass test',
        )

        assert spy_calls == [], f"Lookup was unexpectedly called: {spy_calls}"

    @pytest.mark.asyncio
    async def test_review_suggestions_open_on_disk(self, tmp_path: Path):
        """With category='review_suggestions', escalation file on disk is 'pending'."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('done')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _info(
            server,
            task_id='task-999',
            agent_role='implementer',
            category='review_suggestions',
            summary='review suggestions bypass test',
        )

        esc_id = result['id']
        esc = queue.get(esc_id)
        assert esc is not None
        assert esc.status == 'pending', f"Expected 'pending', got: {esc.status}"


# ---------------------------------------------------------------------------
# Step-5 RED tests: auto-resolve must fire only _resolve_callback, not _notify
# ---------------------------------------------------------------------------


class TestAutoResolveSingleNotification:
    """Auto-resolve fires _resolve_callback once and _notify_callback zero times.

    The old two-call path (submit then resolve) fires _notify_callback once
    and _resolve_callback once — spurious double-count.  After step-6 wires
    queue.submit_resolved, only _resolve_callback fires.

    Both tests FAIL until step-6 switches the server to submit_resolved.
    """

    @pytest.mark.asyncio
    async def test_auto_resolve_fires_resolve_callback_only_no_notify(self, tmp_path: Path):
        """escalate_blocker auto-resolve: _resolve_callback fires once, _notify_callback zero times."""
        queue = EscalationQueue(tmp_path / 'esc')

        notify_fired: list[str] = []
        resolve_fired: list[str] = []
        queue.set_notify_callback(lambda e: notify_fired.append(e.id))
        queue.set_resolve_callback(lambda e: resolve_fired.append(e.id))

        lookup = await _make_lookup('done')
        server = create_server(queue, task_status_lookup=lookup)

        await _blocker(server, **_COMMON_KWARGS)

        assert notify_fired == [], (
            f"Expected _notify_callback NOT fired on auto-resolve, got: {notify_fired}"
        )
        assert len(resolve_fired) == 1, (
            f"Expected _resolve_callback fired exactly once, got: {resolve_fired}"
        )

    @pytest.mark.asyncio
    async def test_auto_resolve_info_path_fires_resolve_callback_only(self, tmp_path: Path):
        """escalate_info auto-resolve: _resolve_callback fires once, _notify_callback zero times."""
        queue = EscalationQueue(tmp_path / 'esc')

        notify_fired: list[str] = []
        resolve_fired: list[str] = []
        queue.set_notify_callback(lambda e: notify_fired.append(e.id))
        queue.set_resolve_callback(lambda e: resolve_fired.append(e.id))

        lookup = await _make_lookup('cancelled')
        server = create_server(queue, task_status_lookup=lookup)

        await _info(server, **_COMMON_KWARGS)

        assert notify_fired == [], (
            f"Expected _notify_callback NOT fired on auto-resolve, got: {notify_fired}"
        )
        assert len(resolve_fired) == 1, (
            f"Expected _resolve_callback fired exactly once, got: {resolve_fired}"
        )


# TestResolveNoneFallback was removed in step-6 (commit: switch to submit_resolved).
#
# Those tests characterised the queue.submit(esc) + queue.resolve()→None fallback
# at server.py.  With submit_resolved, the server no longer calls queue.resolve at
# all — the monkeypatch on queue.resolve became a no-op and the asserted fallback
# behaviour (status='pending' + warning log) was unreachable.  Leaving vacuous-green
# tests would mislead future readers into thinking the fallback is covered.
# See design decision: "Remove TestResolveNoneFallback … in the impl step that
# switches the server to submit_resolved."


# ---------------------------------------------------------------------------
# Helpers for merge_request tests
# ---------------------------------------------------------------------------


async def _call_merge_request(server, **kwargs: Any) -> dict[str, Any]:
    """Invoke the merge_request MCP tool directly."""
    tool = await server.get_tool('merge_request')
    return await tool.fn(**kwargs)


def _make_orch_config(tmp_path: Path):
    """Create a minimal OrchestratorConfig without a git remote."""
    from orchestrator.config import OrchestratorConfig  # type: ignore[reportMissingImports]
    return OrchestratorConfig(project_root=tmp_path)


def _make_registry():
    from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
        InFlightMergeRegistry,
    )
    return InFlightMergeRegistry()


# ---------------------------------------------------------------------------
# Step-1 RED test: dispatched response includes request_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeRequestRequestId:
    """merge_request dispatched and in_flight responses include request_id."""

    async def test_dispatched_response_includes_request_id(self, tmp_path: Path):
        """Dispatched (terminal) response includes request_id from the MergeRequest.

        RED until step-2 impl: the current dispatched path returns only
        {status, reason, conflict_details, push_status} — no request_id key.

        Setup: server with merge_queue + orch_config + injected registry and
        NO harness (git_ops=None — also proves the fast-path is skipped
        gracefully when git_ops is absent).  A background worker dequeues the
        MergeRequest, captures req.request_id, and resolves req.result with
        MergeOutcome('done').
        """
        from orchestrator.merge_queue import MergeOutcome  # type: ignore[reportMissingImports]

        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = _make_orch_config(tmp_path / 'repo')
        registry = _make_registry()

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
            # no harness → git_ops=None → fast-path skipped
        )

        captured_request_id: list[str] = []

        async def _worker():
            """Dequeue the MergeRequest, capture its request_id, resolve done."""
            req = await mq.get()
            captured_request_id.append(req.request_id)
            req.result.set_result(MergeOutcome('done', reason='test done'))

        worker_task = asyncio.create_task(_worker())

        result = await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='591',
                branch='591',
                worktree=str(tmp_path / 'wt'),
                description='',
                wait_secs=100,
            ),
            timeout=5.0,
        )

        await worker_task

        assert result['status'] == 'done', (
            f"Expected status 'done', got: {result}"
        )
        assert len(captured_request_id) == 1, (
            f"Worker did not capture request_id: {captured_request_id}"
        )
        assert 'request_id' in result, (
            f"Expected 'request_id' key in dispatched result, got keys: {set(result.keys())}"
        )
        assert result['request_id'] == captured_request_id[0], (
            f"Expected request_id={captured_request_id[0]!r}, got: {result.get('request_id')!r}"
        )
        assert result['request_id'].startswith('mr-'), (
            f"Expected request_id to start with 'mr-', got: {result['request_id']!r}"
        )

    # test_in_flight_response_includes_request_id was removed at β8 (default flip).
    # The legacy in_flight response (status='in_flight' + submitting request's id) is
    # retired; coalesce now always returns 'attached' with the existing entry's id.
    # Attached coverage lives in TestMergeRequestWaitSecsZeroAttached and
    # TestMergeRequestDefaultFlip.test_default_call_inflight_branch_returns_attached.

    async def test_submit_time_already_merged_fast_path(self, tmp_path: Path):
        """When branch tip is already an ancestor of main, merge_request returns
        {status:'already_merged', commit} immediately — no enqueue, no merge_queued.

        RED until step-6 impl: without the fast-path, the code proceeds to
        coalesce+enqueue (emits merge_queued, mq non-empty) then blocks on the
        future → asyncio.wait_for raises TimeoutError.

        Verifies PRD invariant I4: guaranteed-redundant submission is killed at
        the door.  Also verifies that resolve_branch_sha is called with the
        prefixed ref 'task/591' (not bare '591') per the worker convention.
        """
        from orchestrator.event_store import EventType  # type: ignore[reportMissingImports]

        FAKE_TIP = 'deadbeef12345678'

        # Recording event_store stub
        class _RecordingEventStore:
            def __init__(self):
                self.events: list = []

            def emit(self, event_type, **kwargs) -> None:  # type: ignore[override]
                self.events.append(event_type)

        recording_event_store = _RecordingEventStore()

        # git_ops stub recording calls
        resolve_calls: list[str] = []
        ancestor_calls: list[tuple] = []

        async def _resolve_branch_sha(name: str) -> str:
            resolve_calls.append(name)
            return FAKE_TIP

        async def _is_ancestor(ancestor: str, descendant: str) -> bool:
            ancestor_calls.append((ancestor, descendant))
            return True

        async def _find_inflight_merge_worktree(branch: str):
            return None

        git_ops_stub = types.SimpleNamespace(
            resolve_branch_sha=_resolve_branch_sha,
            is_ancestor=_is_ancestor,
            find_inflight_merge_worktree=_find_inflight_merge_worktree,
        )
        harness_stub = types.SimpleNamespace(git_ops=git_ops_stub)

        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = _make_orch_config(tmp_path / 'repo')
        registry = _make_registry()

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            event_store=recording_event_store,
            harness=harness_stub,
            merge_inflight_registry=registry,
        )

        result = await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='591',
                branch='591',
                worktree=str(tmp_path / 'wt'),
                description='',
            ),
            timeout=2.0,
        )

        # Fast-path result: {status, commit, reason, conflict_details, push_status}.
        # request_id is absent: fast-path fires before any MergeRequest entry exists.
        assert result == {
            'status': 'already_merged',
            'commit': FAKE_TIP,
            'reason': '',
            'conflict_details': '',
            'push_status': None,
        }, (
            f"Expected converged already_merged fast-path result, got: {result}"
        )
        assert mq.empty(), (
            f"Expected empty queue (no enqueue on already_merged), qsize={mq.qsize()}"
        )
        assert EventType.merge_queued not in recording_event_store.events, (
            f"Expected no merge_queued event, got: {recording_event_store.events}"
        )
        # resolve_branch_sha must be called with the prefixed ref 'task/591'
        assert resolve_calls == ['task/591'], (
            f"Expected resolve_branch_sha('task/591'), got: {resolve_calls}"
        )
        # is_ancestor must be called with (tip, main_branch)
        assert len(ancestor_calls) == 1, (
            f"Expected is_ancestor called once, got: {ancestor_calls}"
        )
        assert ancestor_calls[0] == (FAKE_TIP, 'main'), (
            f"Expected is_ancestor('{FAKE_TIP}', 'main'), got: {ancestor_calls[0]}"
        )


# ---------------------------------------------------------------------------
# Fast-path fall-through tests (suggestion 2 — test_coverage)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeRequestFastPathFallThrough:
    """Fast-path fall-throughs: missing branch and non-ancestor branch both enqueue."""

    @staticmethod
    def _make_git_ops_stub(tip_sha: str | None, is_anc: bool):
        """Build a minimal git_ops stub recording resolve/ancestor calls."""
        resolve_calls: list[str] = []
        ancestor_calls: list[tuple] = []

        async def _resolve_branch_sha(name: str) -> str | None:
            resolve_calls.append(name)
            return tip_sha

        async def _is_ancestor(ancestor: str, descendant: str) -> bool:
            ancestor_calls.append((ancestor, descendant))
            return is_anc

        async def _find_inflight_merge_worktree(branch: str):
            return None

        stub = types.SimpleNamespace(
            resolve_branch_sha=_resolve_branch_sha,
            is_ancestor=_is_ancestor,
            find_inflight_merge_worktree=_find_inflight_merge_worktree,
        )
        return stub, resolve_calls, ancestor_calls

    async def _run_until_blocked(self, server, tmp_path: Path, branch: str = '591'):
        """Start a merge_request call and yield control until it blocks on await future.

        Returns the asyncio.Task; caller must cancel it when done.
        """
        task = asyncio.create_task(
            _call_merge_request(
                server,
                task_id=branch,
                branch=branch,
                worktree=str(tmp_path / 'wt'),
                description='',
            )
        )
        # All stub coroutines complete without yielding, so a few sleep(0) rounds
        # are enough to advance the task to the blocking `await future`.
        for _ in range(5):
            await asyncio.sleep(0)
        return task

    async def test_tip_none_falls_through_to_enqueue(self, tmp_path: Path):
        """resolve_branch_sha returns None → fast-path skipped; request enqueues.

        When the branch ref is absent, tip=None makes the fast-path condition
        False (short-circuit `tip is not None and ...`), so is_ancestor is never
        called and the normal coalesce/enqueue path runs instead.

        This preserves existing semantics: the worker will detect the missing
        branch and emit its unknown_branch outcome.
        """
        git_ops_stub, resolve_calls, ancestor_calls = self._make_git_ops_stub(
            tip_sha=None, is_anc=False  # is_anc value irrelevant — never called
        )
        harness_stub = types.SimpleNamespace(git_ops=git_ops_stub)

        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = _make_orch_config(tmp_path / 'repo')
        registry = _make_registry()

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            harness=harness_stub,
            merge_inflight_registry=registry,
        )

        task = await self._run_until_blocked(server, tmp_path)

        try:
            # Branch tip was None → must have enqueued
            assert mq.qsize() == 1, (
                f"Expected one item enqueued (tip=None falls through), qsize={mq.qsize()}"
            )
            # resolve_branch_sha still called with prefixed ref
            assert resolve_calls == ['task/591'], (
                f"Expected resolve_branch_sha('task/591'), got: {resolve_calls}"
            )
            # is_ancestor must NOT be called when tip is None (short-circuit)
            assert ancestor_calls == [], (
                f"is_ancestor must not be called when tip=None, got: {ancestor_calls}"
            )
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def test_non_ancestor_tip_falls_through_to_enqueue(self, tmp_path: Path):
        """resolve_branch_sha returns a SHA but is_ancestor returns False → enqueues.

        When the branch exists but its tip is NOT yet an ancestor of main
        (i.e. the branch has not been merged), is_ancestor returns False and
        the fast-path condition is False, so the request proceeds to the normal
        coalesce/enqueue path rather than returning already_merged.

        A regression that skipped enqueue on any non-None tip would violate this.
        """
        FAKE_TIP = 'aabbccdd11223344'
        git_ops_stub, resolve_calls, ancestor_calls = self._make_git_ops_stub(
            tip_sha=FAKE_TIP, is_anc=False
        )
        harness_stub = types.SimpleNamespace(git_ops=git_ops_stub)

        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = _make_orch_config(tmp_path / 'repo')
        registry = _make_registry()

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            harness=harness_stub,
            merge_inflight_registry=registry,
        )

        task = await self._run_until_blocked(server, tmp_path)

        try:
            # Non-ancestor tip → must enqueue rather than fast-path return
            assert mq.qsize() == 1, (
                f"Expected one item enqueued (non-ancestor falls through), qsize={mq.qsize()}"
            )
            # is_ancestor WAS called (the check ran, returned False, fell through)
            assert len(ancestor_calls) == 1, (
                f"Expected is_ancestor called once, got: {ancestor_calls}"
            )
            assert ancestor_calls[0] == (FAKE_TIP, 'main'), (
                f"Expected is_ancestor('{FAKE_TIP}', 'main'), got: {ancestor_calls[0]}"
            )
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


# ---------------------------------------------------------------------------
# β1 Step-7 RED: merge_request(wait_secs=0) free branch → 'queued' immediately
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeRequestWaitSecsZeroFree:
    """β1 step-7 RED: wait_secs=0 on a free branch returns 'queued' without blocking.

    RED until step-8 impl: wait_secs param does not exist yet → TypeError,
    and with no wait_secs the call blocks on `await future` with no worker
    → asyncio.wait_for raises TimeoutError.
    """

    async def test_wait_secs_zero_dispatched_returns_queued(self, tmp_path: Path):
        """wait_secs=0 on a free branch: immediate return, status='queued'.

        Setup: server with merge_queue + orch_config + injected registry,
        NO worker resolving the future.  Call merge_request(wait_secs=0)
        under wait_for(timeout=2).  The call must return before the timeout.
        """
        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = _make_orch_config(tmp_path / 'repo')
        registry = _make_registry()

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
        )

        result = await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='free-b',
                branch='free-b',
                worktree=str(tmp_path / 'wt'),
                wait_secs=0,
            ),
            timeout=2.0,
        )

        # Status must be 'queued'
        assert result.get('status') == 'queued', (
            f"Expected status='queued', got: {result}"
        )
        # request_id present and well-formed
        assert 'request_id' in result, f'Missing request_id: {result}'
        assert result['request_id'].startswith('mr-'), (
            f"Expected request_id to start with 'mr-', got: {result['request_id']!r}"
        )
        # Required non-blocking shape keys
        for key in ('snapshot_tip', 'generation', 'position', 'queue_depth', 'eta_seconds'):
            assert key in result, f'Missing key {key!r} in result: {result}'
        # generation is always 0 in β1
        assert result['generation'] == 0, f"Expected generation=0, got: {result['generation']}"
        # position is an int
        assert isinstance(result['position'], int), (
            f"Expected position to be int, got: {type(result['position'])}"
        )
        # queue_depth >= 1 (the request was enqueued)
        assert result['queue_depth'] >= 1, (
            f"Expected queue_depth >= 1, got: {result['queue_depth']}"
        )
        # The request must actually be enqueued
        assert mq.qsize() == 1, f'Expected mq.qsize()==1, got: {mq.qsize()}'

        # Clean up enqueued future to avoid ResourceWarning
        req = mq.get_nowait()
        req.result.cancel()


# ---------------------------------------------------------------------------
# β1 Step-9 RED: merge_request(wait_secs=0) in-flight branch → 'attached'
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeRequestWaitSecsZeroAttached:
    """β1 step-9 RED: wait_secs=0 on an already-in-flight branch returns 'attached'.

    RED until step-10 impl: the in_flight block currently ignores wait_secs
    and either falls through to 'queued' (current fall-through path added in
    step-8) with the submitting request's id, or returns 'in_flight' on the
    legacy path — neither matches the expected 'attached' shape with the
    EXISTING entry's request_id.
    """

    async def test_wait_secs_zero_inflight_returns_attached(self, tmp_path: Path):
        """wait_secs=0 for an already-in-flight branch: returns 'attached' with existing id.

        Pre-seed the registry with branch 'X' and request_id='mr-existing'.
        Then call merge_request(task_id='X', branch='X', wait_secs=0).
        Must return immediately (no blocking), status='attached', and the
        request_id must be the existing entry's id ('mr-existing'), NOT the
        submitting request's id.
        """
        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = _make_orch_config(tmp_path / 'repo')
        registry = _make_registry()

        # Pre-seed the registry: acquire branch 'X' with a known request_id
        # and a never-resolving future to simulate an in-flight merge.
        never_future: asyncio.Future = asyncio.get_running_loop().create_future()
        acquired = registry.acquire('X', 'existing-task', never_future, request_id='mr-existing')
        assert acquired, 'Prerequisite: registry must accept first acquire'

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
        )

        result = await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='X',
                branch='X',
                worktree=str(tmp_path / 'wt'),
                wait_secs=0,
            ),
            timeout=2.0,
        )

        # Must be 'attached' (not 'in_flight' or 'queued')
        assert result.get('status') == 'attached', (
            f"Expected status='attached', got: {result}"
        )
        # request_id must be the EXISTING entry's id, not the submitting request's
        assert result.get('request_id') == 'mr-existing', (
            f"Expected request_id='mr-existing', got: {result.get('request_id')!r}"
        )
        # Required non-blocking shape keys
        for key in ('snapshot_tip', 'generation', 'eta_seconds', 'position', 'queue_depth'):
            assert key in result, f'Missing key {key!r} in result: {result}'
        # generation is always 0 in β1
        assert result['generation'] == 0, f"Expected generation=0, got: {result['generation']}"

        # Clean up the never-resolving future to avoid ResourceWarning
        never_future.cancel()


# ---------------------------------------------------------------------------
# β8 Step-1 RED: default call (no wait_secs) must return immediately
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeRequestDefaultFlip:
    """β8 step-1 RED: merge_request with NO wait_secs must return immediately.

    RED today: the None default blocks unboundedly → (a) wait_for raises
    TimeoutError; (b) routes through the legacy 'in_flight' shape → status
    mismatch.  GREEN after step-2 impl flips the default to 0.
    """

    async def test_default_call_free_branch_returns_queued(self, tmp_path: Path):
        """Default call on a free branch returns 'queued' without blocking.

        Setup: server with merge_queue + orch_config + injected registry, NO
        worker.  Call merge_request with NO wait_secs under
        asyncio.wait_for(timeout=2.0).  Must return before the timeout with
        status=='queued' and the non-blocking shape keys.
        """
        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = _make_orch_config(tmp_path / 'repo')
        registry = _make_registry()

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
        )

        result = await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='def-free',
                branch='def-free',
                worktree=str(tmp_path / 'wt'),
            ),
            timeout=2.0,
        )

        assert result.get('status') == 'queued', (
            f"Expected status='queued', got: {result}"
        )
        assert 'request_id' in result, f'Missing request_id: {result}'
        assert result['request_id'].startswith('mr-'), (
            f"Expected request_id to start with 'mr-', got: {result['request_id']!r}"
        )
        for key in ('snapshot_tip', 'generation', 'position', 'queue_depth', 'eta_seconds'):
            assert key in result, f'Missing key {key!r} in result: {result}'
        assert result['generation'] == 0, f"Expected generation=0, got: {result['generation']}"
        assert isinstance(result['position'], int), (
            f"Expected position to be int, got: {type(result['position'])}"
        )
        assert result['queue_depth'] >= 1, (
            f"Expected queue_depth >= 1, got: {result['queue_depth']}"
        )
        assert mq.qsize() == 1, f'Expected mq.qsize()==1, got: {mq.qsize()}'

        # Clean up enqueued future to avoid ResourceWarning
        req = mq.get_nowait()
        req.result.cancel()

    async def test_default_call_inflight_branch_returns_attached(self, tmp_path: Path):
        """Default call on an already-in-flight branch returns 'attached' with existing id.

        Pre-seed the registry with branch 'X' and request_id='mr-existing'.
        Call merge_request with NO wait_secs for branch 'X'.  Must return
        immediately (no blocking), status=='attached', and request_id must be
        the existing entry's id ('mr-existing'), NOT the submitting request's id.
        """
        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = _make_orch_config(tmp_path / 'repo')
        registry = _make_registry()

        # Pre-seed the registry: acquire branch 'X' with a known request_id
        never_future: asyncio.Future = asyncio.get_running_loop().create_future()
        acquired = registry.acquire('X', 'existing-task', never_future, request_id='mr-existing')
        assert acquired, 'Prerequisite: registry must accept first acquire'

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
        )

        result = await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='X',
                branch='X',
                worktree=str(tmp_path / 'wt'),
            ),
            timeout=2.0,
        )

        assert result.get('status') == 'attached', (
            f"Expected status='attached', got: {result}"
        )
        assert result.get('request_id') == 'mr-existing', (
            f"Expected request_id='mr-existing', got: {result.get('request_id')!r}"
        )
        assert result['generation'] == 0, f"Expected generation=0, got: {result['generation']}"
        for key in ('snapshot_tip', 'generation', 'eta_seconds', 'position', 'queue_depth'):
            assert key in result, f'Missing key {key!r} in result: {result}'

        # Clean up the never-resolving future to avoid ResourceWarning
        never_future.cancel()


# ---------------------------------------------------------------------------
# β1 Step-11 RED: wait_secs>0 bounded wait — happy path + clamp/timeout/shield
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeRequestWaitSecsPositive:
    """β1 step-11 RED: wait_secs>0 bounded wait with clamp, timeout, and shield.

    RED until step-12 impl: wait_secs>0 falls through to the unbounded
    asyncio.shield(future) → blocks forever → wait_for(timeout=2) raises
    TimeoutError for scenario (b).  Scenario (a) happens to pass (worker
    resolves quickly) but (b) is the regression driver.
    """

    async def test_wait_secs_positive_happy_path(self, tmp_path: Path):
        """wait_secs=5: worker resolves quickly → terminal 'done' shape returned.

        A background worker dequeues the MergeRequest and resolves it with
        MergeOutcome('done').  The call must return the terminal outcome shape
        before the timeout.
        """
        from orchestrator.merge_queue import MergeOutcome  # type: ignore[reportMissingImports]

        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = _make_orch_config(tmp_path / 'repo')
        registry = _make_registry()

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
        )

        captured_req: list = []

        async def _worker():
            req = await mq.get()
            captured_req.append(req)
            req.result.set_result(MergeOutcome('done', reason='fast worker'))

        worker_task = asyncio.create_task(_worker())

        result = await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='task-wp',
                branch='task-wp',
                worktree=str(tmp_path / 'wt'),
                wait_secs=5,
            ),
            timeout=5.0,
        )
        await worker_task

        # Must return the terminal outcome shape
        assert result.get('status') == 'done', (
            f"Expected status='done', got: {result}"
        )
        assert 'request_id' in result, f'Missing request_id: {result}'
        assert result['request_id'].startswith('mr-'), (
            f"Expected request_id to start with 'mr-', got: {result['request_id']!r}"
        )
        for key in ('reason', 'conflict_details', 'push_status', 'commit'):
            assert key in result, f'Missing key {key!r} in result: {result}'

    async def test_wait_secs_clamp_timeout_shield(self, tmp_path: Path, monkeypatch):
        """wait_secs=600 clamped to _MAX_WAIT_SECS (monkeypatched to 0.1):
        times out → returns 'queued' shape; entry survives (not cancelled).

        Monkeypatch escalation.server._MAX_WAIT_SECS to 0.1 so the clamp
        fires immediately.  NO worker resolves the future.  The call must
        return the non-terminal 'queued' shape within the outer wait_for
        timeout (2 s).  The enqueued entry's future must NOT be cancelled
        (asyncio.shield ensures the timeout cancels only the outer wait,
        not req.result).
        """
        import escalation.server as _srv

        monkeypatch.setattr(_srv, '_MAX_WAIT_SECS', 0.1)

        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = _make_orch_config(tmp_path / 'repo')
        registry = _make_registry()

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
        )

        result = await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='task-ct',
                branch='task-ct',
                worktree=str(tmp_path / 'wt'),
                wait_secs=600,
            ),
            timeout=2.0,
        )

        # Must return the non-terminal 'queued' shape
        assert result.get('status') == 'queued', (
            f"Expected status='queued' on timeout, got: {result}"
        )
        for key in ('request_id', 'snapshot_tip', 'generation', 'position', 'queue_depth', 'eta_seconds'):
            assert key in result, f'Missing key {key!r} in result: {result}'
        assert result['generation'] == 0, f"Expected generation=0, got: {result['generation']}"

        # Entry must be enqueued (not swallowed)
        assert mq.qsize() == 1, f'Expected mq.qsize()==1, got: {mq.qsize()}'

        # Shield: the entry's future must NOT be cancelled
        req = mq.get_nowait()
        assert not req.result.cancelled(), (
            'Entry future must NOT be cancelled — asyncio.shield should protect it'
        )
        # Clean up
        req.result.cancel()


# ---------------------------------------------------------------------------
# β8 Step-3 RED: explicit None treated as 'queued' (I1 regression guard)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeRequestExplicitNoneQueued:
    """β8 step-3 RED: explicit wait_secs=None must NOT block (PRD invariant I1).

    RED after step-2: explicit None still falls into the legacy
    `else: outcome = await asyncio.shield(future)` unbounded branch → blocks
    → wait_for raises TimeoutError.
    GREEN after step-4 impl deletes the unbounded branch.
    """

    async def test_explicit_none_wait_secs_does_not_block(self, tmp_path: Path):
        """Passing wait_secs=None explicitly returns 'queued' without blocking.

        Setup: server with merge_queue + orch_config + injected registry,
        NO worker.  Call merge_request(wait_secs=None) (type: ignore intentional —
        exercises the retired sentinel) under asyncio.wait_for(timeout=2.0).
        Must return before the timeout with status=='queued'.
        """
        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = _make_orch_config(tmp_path / 'repo')
        registry = _make_registry()

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
        )

        result = await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='none-free',
                branch='none-free',
                worktree=str(tmp_path / 'wt'),
                wait_secs=None,  # type: ignore[reportArgumentType]
            ),
            timeout=2.0,
        )

        assert result.get('status') == 'queued', (
            f"Expected status='queued' for explicit None, got: {result}"
        )
        assert 'request_id' in result, f'Missing request_id: {result}'
        assert result['request_id'].startswith('mr-'), (
            f"Expected request_id to start with 'mr-', got: {result['request_id']!r}"
        )

        # Clean up
        req = mq.get_nowait()
        req.result.cancel()


# ---------------------------------------------------------------------------
# β8 Step-5 RED: explicit None on coalesced branch must return 'attached'
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeRequestExplicitNoneAttached:
    """β8 step-5 RED: explicit wait_secs=None on an in-flight branch must
    return 'attached' (not the legacy 'in_flight' shape).

    RED after step-4: explicit None no longer blocks (step-4 greened that),
    but the `if wait_secs is None:` sub-branch inside dispatch.in_flight
    still returns status=='in_flight' with the submitting request's id.
    GREEN after step-6 impl deletes that sub-branch.
    """

    async def test_explicit_none_wait_secs_coalesce_returns_attached(self, tmp_path: Path):
        """Explicit None on an already-in-flight branch returns 'attached' with existing id.

        Pre-seed the registry with branch 'X' and request_id='mr-existing'.
        Call merge_request(wait_secs=None) (type: ignore intentional).
        Must return status=='attached' and request_id=='mr-existing'.
        """
        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = _make_orch_config(tmp_path / 'repo')
        registry = _make_registry()

        never_future: asyncio.Future = asyncio.get_running_loop().create_future()
        acquired = registry.acquire('X', 'existing-task', never_future, request_id='mr-existing')
        assert acquired, 'Prerequisite: registry must accept first acquire'

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
        )

        result = await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='X',
                branch='X',
                worktree=str(tmp_path / 'wt'),
                wait_secs=None,  # type: ignore[reportArgumentType]
            ),
            timeout=2.0,
        )

        assert result.get('status') == 'attached', (
            f"Expected status='attached' for explicit None on coalesced branch, got: {result}"
        )
        assert result.get('request_id') == 'mr-existing', (
            f"Expected request_id='mr-existing', got: {result.get('request_id')!r}"
        )

        never_future.cancel()


# ---------------------------------------------------------------------------
# β1 Step-13 RED: legacy (wait_secs=None) — client disconnect doesn't cancel entry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeRequestDurableIntent:
    """β1 step-13: durable intent — client disconnect (Task cancel) does NOT cancel entry.

    Originally RED: without asyncio.shield the Task cancel propagates to
    req.result.  Step-12 pre-implemented the shield on the legacy path, so
    this test is GREEN on write.  Step-14 updates the docstring and runs the
    full suite.
    """

    async def test_bounded_wait_disconnect_does_not_cancel_entry(self, tmp_path: Path):
        """Bounded-wait path (wait_secs>0): cancelling the Task mid-wait does not
        cancel the enqueued entry's future (durable intent, PRD D2/I5).

        Start merge_request(wait_secs=30) as a Task (no worker → blocks on the
        bounded asyncio.wait_for).  Advance the loop until it blocks.
        Retrieve the enqueued req.  Cancel the Task.  Assert that req.result is
        NOT cancelled (asyncio.shield protects it from the outer cancellation).
        """
        from orchestrator.merge_queue import MergeOutcome  # type: ignore[reportMissingImports]

        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = _make_orch_config(tmp_path / 'repo')
        registry = _make_registry()

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
        )

        # Start the merge_request call as a Task (simulates an MCP session lifetime).
        # wait_secs=30 → bounded wait; no worker → blocks inside wait_for.
        merge_task = asyncio.create_task(
            _call_merge_request(
                server,
                task_id='task-di',
                branch='task-di',
                worktree=str(tmp_path / 'wt'),
                wait_secs=30,
            )
        )

        # Advance the event loop until the task blocks on the shielded await.
        for _ in range(5):
            await asyncio.sleep(0)

        # The entry must be in the queue at this point.
        assert mq.qsize() == 1, (
            f'Expected entry enqueued before cancel, qsize={mq.qsize()}'
        )
        # Retrieve the enqueued entry.
        req = mq.get_nowait()

        # Simulate client disconnect: cancel the merge_request Task.
        merge_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await merge_task

        # Durable intent: the entry's future must NOT be cancelled.
        assert not req.result.cancelled(), (
            'Entry future must NOT be cancelled after Task cancel — '
            'asyncio.shield should protect req.result from the disconnect'
        )

        # The entry is still usable: the worker can resolve it normally.
        req.result.set_result(MergeOutcome('done', reason='late resolve'))
        assert req.result.done() and not req.result.cancelled()


# ---------------------------------------------------------------------------
# Helpers for merge_cancel tests (β2)
# ---------------------------------------------------------------------------


async def _call_merge_cancel(server, **kwargs: Any) -> dict[str, Any]:
    """Invoke the merge_cancel MCP tool directly."""
    tool = await server.get_tool('merge_cancel')
    return await tool.fn(**kwargs)


# ---------------------------------------------------------------------------
# Step-1 RED test: success path — pending waiter → cancel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeCancel:
    """β2 merge_cancel MCP tool — cancels a pending waiter future."""

    async def test_cancel_pending_waiter_returns_cancelled_true(self, tmp_path: Path):
        """Cancelling a live pending waiter returns cancelled=True, state='abandoned'.

        Setup: server with merge_queue + orch_config + injected registry, NO harness.
        Submit merge_request(wait_secs=0) on a free branch to register a durable-intent
        waiter and enqueue the MergeRequest.  Capture request_id.  Call merge_cancel
        with that request_id and assert the success shape.  Then drain mq and assert
        the underlying future was actually cancelled.

        RED until step-2 impl: server.get_tool('merge_cancel') fails because the tool
        does not exist yet.
        """
        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = _make_orch_config(tmp_path / 'repo')
        registry = _make_registry()

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
        )

        # Submit via merge_request(wait_secs=0) — registers a WaiterRecord in _waiters
        result_mr = await _call_merge_request(
            server,
            task_id='c1',
            branch='c1',
            worktree=str(tmp_path / 'wt-c1'),
            wait_secs=0,
        )
        assert result_mr['status'] == 'queued', f'Unexpected merge_request status: {result_mr}'
        rid = result_mr['request_id']
        assert rid, 'Expected non-empty request_id from merge_request'

        # Cancel the in-flight waiter
        result_cancel = await _call_merge_cancel(server, request_id=rid)

        assert result_cancel.get('cancelled') is True, (
            f"Expected cancelled=True, got: {result_cancel}"
        )
        assert result_cancel.get('state') == 'abandoned', (
            f"Expected state='abandoned', got: {result_cancel}"
        )
        assert 'reason' in result_cancel, f"Expected 'reason' key, got: {result_cancel}"
        assert result_cancel.get('reason') is None, (
            f"Expected reason=None on success, got: {result_cancel['reason']}"
        )

        # Drain mq and assert the underlying future was actually cancelled
        req = mq.get_nowait()
        assert req.result.cancelled(), (
            f'Expected future to be cancelled after merge_cancel, got: '
            f'done={req.result.done()} cancelled={req.result.cancelled()}'
        )

    async def test_cancel_unknown_request_id_returns_unknown(self, tmp_path: Path):
        """Cancelling a request_id with no live waiter returns cancelled=False, state='unknown'.

        Fresh server with no submissions.  Call merge_cancel with a random id.
        Must return a dict (no raise) with cancelled=False, state='unknown', and
        a non-empty reason string.

        RED until step-4 impl: the current minimal body calls rec.future on a None rec,
        raising AttributeError.
        """
        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = _make_orch_config(tmp_path / 'repo')
        registry = _make_registry()

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
        )

        result = await _call_merge_cancel(server, request_id='mr-doesnotexist')

        assert result.get('cancelled') is False, (
            f"Expected cancelled=False for unknown id, got: {result}"
        )
        assert result.get('state') == 'unknown', (
            f"Expected state='unknown' for unknown id, got: {result}"
        )
        assert result.get('reason'), (
            f"Expected non-empty reason string for unknown id, got: {result}"
        )

    async def test_idempotent_double_cancel_returns_already_cancelled(self, tmp_path: Path):
        """Calling merge_cancel twice on the same request_id returns cancelled=False on the second call.

        Acquires the merge_cancel tool up front (single tool lookup).  Submits
        merge_request(wait_secs=0) to register a waiter; first cancel → {cancelled: True, ...}.
        Immediately (no awaited suspension) calls merge_cancel again — the
        _waiters.pop done-callback is call_soon-scheduled and has not run yet, so
        the waiter is still present with future.cancelled()==True.  Second call must
        return {cancelled: False, state: 'abandoned', reason: <non-empty>} and not raise.

        RED until step-6 impl: the current body has no future.cancelled() branch.
        It falls into the pending path, calls fut.cancel() on an already-cancelled
        future (returns False), and still reports cancelled=True (incorrect).
        """
        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = _make_orch_config(tmp_path / 'repo')
        registry = _make_registry()

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
        )

        # Acquire the tool up front — no awaited suspension between here and .fn() calls
        tool = await server.get_tool('merge_cancel')

        result_mr = await _call_merge_request(
            server,
            task_id='c2',
            branch='c2',
            worktree=str(tmp_path / 'wt-c2'),
            wait_secs=0,
        )
        rid = result_mr['request_id']

        # First cancel — must succeed
        first = await tool.fn(request_id=rid)  # type: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
        assert first.get('cancelled') is True, f'First cancel must succeed: {first}'

        # Second cancel — no awaited suspension here, so _waiters.pop callback hasn't run
        second = await tool.fn(request_id=rid)  # type: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]

        assert second.get('cancelled') is False, (
            f"Expected cancelled=False on double-cancel, got: {second}"
        )
        assert second.get('state') == 'abandoned', (
            f"Expected state='abandoned' on double-cancel, got: {second}"
        )
        assert second.get('reason'), (
            f"Expected non-empty reason on double-cancel, got: {second}"
        )

        # Clean up the drained-not-consumed entry
        req = mq.get_nowait()
        assert req.result.cancelled()

    async def test_cancel_mid_finalize_window_returns_coarse_terminal(self, tmp_path: Path):
        """Cancelling a waiter whose future is resolved-but-not-yet-popped returns cancelled=False.

        Simulates the mid-finalize window: merge_request has resolved the future (e.g.
        the worker delivered MergeOutcome('done')) but the call_soon-scheduled _waiters.pop
        done-callback has not run yet (the loop has not regained control).

        Steps: acquire merge_cancel tool up front; submit merge_request(wait_secs=0);
        drain mq; resolve req.result.set_result(MergeOutcome('done', reason='late resolve'));
        immediately call merge_cancel (no intervening awaited suspension).  The waiter is
        still present (future.done()==True, not cancelled()), so the call must return
        {cancelled: False, state: 'done', reason: <non-empty>}.

        RED until step-8 impl: current body has no future.done() (non-cancelled terminal)
        branch; falls into the pending path, calls fut.cancel() on a done future (returns
        False), and mis-reports cancelled=True.
        """
        from orchestrator.merge_queue import MergeOutcome  # type: ignore[reportMissingImports]

        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = _make_orch_config(tmp_path / 'repo')
        registry = _make_registry()

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
        )

        # Acquire the tool up front — no awaited suspension until .fn() call
        tool = await server.get_tool('merge_cancel')

        result_mr = await _call_merge_request(
            server,
            task_id='c3',
            branch='c3',
            worktree=str(tmp_path / 'wt-c3'),
            wait_secs=0,
        )
        rid = result_mr['request_id']

        # Drain the queue and resolve the future (simulate worker delivering outcome)
        req = mq.get_nowait()
        req.result.set_result(MergeOutcome('done', reason='late resolve'))
        # Confirm we're in the mid-finalize window: done but not cancelled
        assert req.result.done() and not req.result.cancelled(), (
            'Prerequisite: future must be done and not cancelled for mid-finalize test'
        )

        # Immediately cancel (no intervening awaited suspension)
        result = await tool.fn(request_id=rid)  # type: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]

        assert result.get('cancelled') is False, (
            f"Expected cancelled=False for mid-finalize window, got: {result}"
        )
        assert result.get('state') == 'done', (
            f"Expected state='done' (via _map_terminal_state), got: {result}"
        )
        assert result.get('reason'), (
            f"Expected non-empty reason for mid-finalize window, got: {result}"
        )

    async def test_cancel_mid_finalize_window_excepted_future_returns_blocked(
        self, tmp_path: Path
    ):
        """Mid-finalize window: excepted future (abnormal) returns state='blocked'.

        Exercises the defensive sub-path in the mid-finalize branch:
        ``rec.future.exception() is not None → state='blocked'``.  This case
        cannot arise via the normal MergeWorker path, but the branch exists to
        keep the tool exception-free when a future is resolved with an exception
        rather than a MergeOutcome result.

        Steps: acquire merge_cancel tool up front; submit merge_request(wait_secs=0);
        drain mq; call req.result.set_exception(RuntimeError(...)); immediately call
        merge_cancel (no awaited suspension).  Must return
        {cancelled: False, state: 'blocked', reason: <non-empty>}.

        Previously untested: the existing test only covers set_result(MergeOutcome)
        (the normal-outcome sub-path).  This test exercises the excepted-future branch.
        """
        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = _make_orch_config(tmp_path / 'repo')
        registry = _make_registry()

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
        )

        # Acquire the tool up front — no awaited suspension until .fn() call
        tool = await server.get_tool('merge_cancel')

        result_mr = await _call_merge_request(
            server,
            task_id='c4',
            branch='c4',
            worktree=str(tmp_path / 'wt-c4'),
            wait_secs=0,
        )
        rid = result_mr['request_id']

        # Drain and set an exception on the future (abnormal mid-finalize window)
        req = mq.get_nowait()
        req.result.set_exception(RuntimeError('worker exploded'))
        assert req.result.done() and not req.result.cancelled(), (
            'Prerequisite: future must be done (excepted) for this test'
        )

        # Immediately cancel (no awaited suspension between set_exception and .fn())
        result = await tool.fn(request_id=rid)  # type: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]

        assert result.get('cancelled') is False, (
            f"Expected cancelled=False for excepted-future mid-finalize, got: {result}"
        )
        assert result.get('state') == 'blocked', (
            f"Expected state='blocked' for excepted future, got: {result}"
        )
        assert result.get('reason'), (
            f"Expected non-empty reason for excepted-future mid-finalize, got: {result}"
        )

    async def test_cancel_finalized_popped_id_resolves_via_durable_tier(self, tmp_path: Path):
        """Cancelling a finalized+popped request_id returns the durable terminal state.

        Injects a tiny fake event_store that returns a finalized row for 'mr-finalized'
        and None for other ids.  No waiter is registered (simulating finalized+popped).
        merge_cancel('mr-finalized') must return {cancelled: False, state: 'done', reason:
        <non-empty>} — not 'unknown'.  merge_cancel('mr-never') must still return state='unknown'.

        RED until step-10 impl: the None-rec branch always returns 'unknown' and never
        consults event_store.
        """

        class FakeEventStore:
            def latest_merge_finalized(self, request_id=None, branch=None, task_id=None):
                if request_id == 'mr-finalized':
                    return {
                        'request_id': request_id,
                        'state': 'done',
                        'finished_at': '2026-01-01T00:00:00+00:00',
                    }
                return None

        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = _make_orch_config(tmp_path / 'repo')
        registry = _make_registry()

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
            event_store=FakeEventStore(),
        )

        # No waiter registered for 'mr-finalized' (simulates finalized+popped)
        result_finalized = await _call_merge_cancel(server, request_id='mr-finalized')

        assert result_finalized.get('cancelled') is False, (
            f"Expected cancelled=False for finalized id, got: {result_finalized}"
        )
        assert result_finalized.get('state') == 'done', (
            f"Expected state='done' from durable tier, got: {result_finalized}"
        )
        assert result_finalized.get('reason'), (
            f"Expected non-empty reason for finalized id, got: {result_finalized}"
        )

        # An id not in the event_store must still return 'unknown'
        result_never = await _call_merge_cancel(server, request_id='mr-never')
        assert result_never.get('state') == 'unknown', (
            f"Expected state='unknown' for truly unknown id, got: {result_never}"
        )

    async def test_cancel_finalized_popped_id_resolves_via_retention_ring(
        self, tmp_path: Path
    ):
        """Cancelling a finalized+popped id resolves via the retention ring (Tier 2).

        Injects a fake harness whose ``_terminal_retention`` ring returns a record for
        'mr-ring-hit' (state='done') and a fake event_store that would return 'conflict'
        for the same id.  The ring (Tier 2) must take precedence over the event_store
        (Tier 3).

        Also asserts:
          - 'mr-store-only' (ring miss, event_store hit) returns state='conflict'.
          - 'mr-never' (both miss) returns state='unknown'.

        Previously untested: test_cancel_finalized_popped_id_resolves_via_durable_tier
        covers the event_store-only path but not the retention-ring path.  A regression
        in the ring branch (e.g. wrong attribute name, wrong precedence) would go
        undetected without this test.
        """

        class FakeRetentionRecord:
            def __init__(self, request_id: str, state: str) -> None:
                self.request_id = request_id
                self.state = state
                self.finished_at = 1_700_000_000.0  # epoch float — normalised by _epoch_to_iso8601
                self.superseded_by = None

        class FakeRetentionRing:
            def get(self, request_id: str):
                if request_id == 'mr-ring-hit':
                    return FakeRetentionRecord('mr-ring-hit', 'done')
                return None

        class FakeHarness:
            _terminal_retention = FakeRetentionRing()

        class FakeEventStore:
            def latest_merge_finalized(self, request_id=None, branch=None, task_id=None):
                if request_id == 'mr-ring-hit':
                    # Ring takes precedence — this row must NOT be returned
                    return {
                        'request_id': request_id,
                        'state': 'conflict',
                        'finished_at': '2026-01-01T00:00:00+00:00',
                    }
                if request_id == 'mr-store-only':
                    return {
                        'request_id': request_id,
                        'state': 'conflict',
                        'finished_at': '2026-01-01T00:00:00+00:00',
                    }
                return None

        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = _make_orch_config(tmp_path / 'repo')
        registry = _make_registry()

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
            harness=FakeHarness(),
            event_store=FakeEventStore(),
        )

        # Ring hit: returns state='done' from ring (not 'conflict' from event_store)
        result_ring = await _call_merge_cancel(server, request_id='mr-ring-hit')
        assert result_ring.get('cancelled') is False, (
            f"Expected cancelled=False for finalized ring-hit id, got: {result_ring}"
        )
        assert result_ring.get('state') == 'done', (
            f"Expected state='done' from retention ring (Tier 2), got: {result_ring}"
        )
        assert result_ring.get('reason'), (
            f"Expected non-empty reason for finalized ring-hit id, got: {result_ring}"
        )

        # Event-store only: ring misses, falls through to event_store
        result_store = await _call_merge_cancel(server, request_id='mr-store-only')
        assert result_store.get('state') == 'conflict', (
            f"Expected state='conflict' from event_store (Tier 3), got: {result_store}"
        )

        # Truly unknown: both tiers miss
        result_never = await _call_merge_cancel(server, request_id='mr-never')
        assert result_never.get('state') == 'unknown', (
            f"Expected state='unknown' when both durable tiers miss, got: {result_never}"
        )


# ---------------------------------------------------------------------------
# Step-11 regression guard: cancellation releases the branch slot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeCancelSlotRelease:
    """β2 regression guard — cancel unwinds in-flight state at MCP boundary.

    Confirms that merge_cancel releases the branch slot via the α1/β1 done-callbacks
    so a subsequent dispatch on the same branch returns 'queued' (fresh dispatch)
    rather than blocking on a stuck/leaked slot.

    This is a regression guard over the α1/β1 done-callback wiring only.
    Worker-drop-without-halt (_request_abandoned) and retention-records-abandoned
    consequences are covered by test_merge_queue.py:4520 / :4601 and are NOT
    re-tested here.
    """

    async def test_cancel_releases_branch_slot_for_resubmit(self, tmp_path: Path):
        """After merge_cancel, re-submitting the same branch dispatches fresh ('queued').

        Build server with merge_queue + orch_config + injected registry (no harness).
        Submit merge_request(branch='re', wait_secs=0) — dispatches and acquires the
        registry slot; capture request_id.  Cancel via merge_cancel (assert cancelled=True).
        Yield the event loop (await asyncio.sleep(0)) enough times for the acquire-time
        _release done-callback (merge_queue.py:1633) to fire.  Re-submit merge_request
        on the same branch and assert status=='queued' — proves the slot was released and
        the dispatch path runs cleanly without a stuck/halted slot.
        """
        esc_queue = EscalationQueue(tmp_path / 'esc')
        mq: asyncio.Queue = asyncio.Queue()
        orch_config = _make_orch_config(tmp_path / 'repo')
        registry = _make_registry()

        server = create_server(
            esc_queue,
            merge_queue=mq,
            orch_config=orch_config,
            merge_inflight_registry=registry,
        )

        # First dispatch: acquires the registry slot for branch 're'
        result1 = await _call_merge_request(
            server,
            task_id='re',
            branch='re',
            worktree=str(tmp_path / 'wt-re'),
            wait_secs=0,
        )
        assert result1['status'] == 'queued', f'First dispatch must be queued: {result1}'
        rid = result1['request_id']

        # Cancel the in-flight waiter
        cancel_result = await _call_merge_cancel(server, request_id=rid)
        assert cancel_result.get('cancelled') is True, (
            f'merge_cancel must succeed: {cancel_result}'
        )

        # Yield the event loop so the acquire-time _release done-callback fires
        for _ in range(5):
            await asyncio.sleep(0)

        # Re-submit on the same branch — must dispatch fresh (not 'in_flight'/'attached')
        result2 = await _call_merge_request(
            server,
            task_id='re',
            branch='re',
            worktree=str(tmp_path / 'wt-re2'),
            wait_secs=0,
        )
        assert result2['status'] == 'queued', (
            f"Expected 'queued' after slot release, got: {result2}"
        )

        # Clean up enqueued future from second submission
        req2 = mq.get_nowait()
        req2.result.cancel()


# ---------------------------------------------------------------------------
# Boundary-test pre-1 helpers
# ---------------------------------------------------------------------------


class _FakeMergeWorker:
    """Controllable merge-worker stand-in whose snapshot() returns caller-set state."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def set_entries(self, entries: list[dict[str, Any]]) -> None:
        self._entries = list(entries)

    def snapshot(self) -> dict[str, Any]:
        return {'entries': list(self._entries), 'depth': len(self._entries)}


class _FakeHarness:
    """Minimal harness stub for boundary-test scenarios 4-8."""

    def __init__(
        self,
        worker: _FakeMergeWorker | None = None,
        retention: Any = None,
        git_ops: Any = None,
    ) -> None:
        self._merge_worker = worker
        self._terminal_retention = retention
        self.git_ops = git_ops


def _build_merge_server(
    tmp_path: Path,
    *,
    worker: _FakeMergeWorker | None = None,
    retention: Any = None,
    event_store: Any = None,
    registry: Any = None,
    git_ops: Any = None,
) -> tuple:
    """Wire EscalationQueue + asyncio.Queue + registry + _FakeHarness into create_server.

    Returns (server, mq, registry, event_store, harness).
    """
    from orchestrator.merge_queue import InFlightMergeRegistry  # type: ignore[reportMissingImports]

    esc_queue = EscalationQueue(tmp_path / 'esc')
    mq: asyncio.Queue = asyncio.Queue()
    orch_config = _make_orch_config(tmp_path / 'repo')
    reg = registry if registry is not None else InFlightMergeRegistry()

    harness = _FakeHarness(worker=worker, retention=retention, git_ops=git_ops)

    server = create_server(
        esc_queue,
        merge_queue=mq,
        orch_config=orch_config,
        event_store=event_store,
        harness=harness,
        merge_inflight_registry=reg,
        startup_sweep=False,
    )
    return server, mq, reg, event_store, harness


async def _call_merge_status(server: Any, **kwargs: Any) -> dict[str, Any]:
    """Invoke the merge_status MCP tool directly."""
    tool = await server.get_tool('merge_status')
    return await tool.fn(**kwargs)


# ---------------------------------------------------------------------------
# TestBoundaryTableMcpSurface — §8 rows 1-8 at the MCP/skill seam
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBoundaryTableMcpSurface:
    """PRD §8 boundary-test table: scenarios 1-8 at the MCP/skill seam.

    Each method is one row, asserting the FULL postcondition.
    Reuses _build_merge_server / _call_merge_request / _call_merge_status /
    _call_merge_cancel helpers from pre-1.
    """

    async def test_scenario_1_non_blocking_busy_queue(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Row 1: non-blocking submit against a busy queue.

        Pre-load the queue with a dummy request so branch Y queues behind it
        (queue_depth=2, position=1).  Submit merge_request(branch=Y, wait_secs=0).
        Assert status='queued', position>=1, merge_queued event emitted.
        Extends TestMergeRequestWaitSecsZeroFree with the full §8-row-1 postcondition.
        """
        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            MergeRequest,
        )

        queued_spy: list[int] = []
        monkeypatch.setattr(
            'orchestrator.merge_queue._emit_merge_queued',
            lambda *a, **kw: queued_spy.append(1),
        )

        server, mq, _reg, _, _ = _build_merge_server(tmp_path)

        # Pre-load a dummy MergeRequest for branch X so Y queues behind it
        x_future: asyncio.Future = asyncio.get_running_loop().create_future()
        dummy_req = MergeRequest(
            task_id='X',
            branch='X',
            worktree=tmp_path / 'wt-x',
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=_make_orch_config(tmp_path / 'repo'),
            result=x_future,
        )
        await mq.put(dummy_req)

        result = await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='Y',
                branch='Y',
                worktree=str(tmp_path / 'wt-Y'),
                wait_secs=0,
            ),
            timeout=3.0,
        )

        assert result.get('status') == 'queued', f"Expected 'queued', got: {result}"
        assert 'request_id' in result and result['request_id'].startswith('mr-'), (
            f"Expected valid request_id, got: {result}"
        )
        assert isinstance(result.get('position'), int), (
            f"Expected int position, got type {type(result.get('position'))}: {result}"
        )
        assert result['position'] >= 1, (
            f"Expected position>=1 (Y behind X), got: {result['position']}"
        )
        assert 'queue_depth' in result, f"Missing queue_depth: {result}"
        assert result['queue_depth'] >= 1, f"Expected queue_depth>=1: {result}"
        assert len(queued_spy) == 1, (
            f"Expected exactly 1 merge_queued event, got: {len(queued_spy)}"
        )

        # Cleanup: drain all entries (X was pre-loaded, Y was enqueued by merge_request)
        while not mq.empty():
            mq.get_nowait().result.cancel()

    async def test_scenario_2_bounded_wait_expiry_entry_intact(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Row 2: bounded-wait expiry leaves the entry intact.

        monkeypatch _MAX_WAIT_SECS=0.1, call merge_request(wait_secs=600).
        Assert returns with 'queued' shape, mq still has the entry, and the
        entry future is NOT cancelled (shield held).
        Extends test_wait_secs_clamp_timeout_shield with full §8-row-2 postcondition.
        """
        import escalation.server as _srv  # type: ignore[reportMissingImports]

        monkeypatch.setattr(_srv, '_MAX_WAIT_SECS', 0.1)

        server, mq, _, _, _ = _build_merge_server(tmp_path)

        result = await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='task-sc2',
                branch='task-sc2',
                worktree=str(tmp_path / 'wt'),
                wait_secs=600,
            ),
            timeout=2.0,
        )

        assert result.get('status') == 'queued', (
            f"Expected status='queued' on clamp-timeout, got: {result}"
        )
        for key in ('request_id', 'snapshot_tip', 'generation', 'position', 'queue_depth', 'eta_seconds'):
            assert key in result, f"Missing key {key!r}: {result}"

        assert mq.qsize() == 1, f"Expected entry still enqueued (qsize=1), got: {mq.qsize()}"

        req = mq.get_nowait()
        assert not req.result.cancelled(), (
            'Entry future must NOT be cancelled after timeout — asyncio.shield must hold'
        )

        req.result.cancel()

    async def test_scenario_3_submit_time_already_merged(
        self, tmp_path: Path,
    ) -> None:
        """Row 3: submit-time already_merged fast-path.

        Wire git_ops with is_ancestor=True.  Assert status='already_merged',
        commit returned, NO request_id, queue untouched, NO merge_queued event.
        Extends test_submit_time_already_merged_fast_path with queue-untouched
        + no-event assertions.
        """
        from orchestrator.event_store import EventType  # type: ignore[reportMissingImports]

        FAKE_TIP = 'deadbeef12345678sc3'

        events_recorded: list = []

        class _RecordingES:
            def emit(self, event_type, **kwargs) -> None:
                events_recorded.append(event_type)

            def latest_merge_finalized(self, **kwargs):
                return None

        async def _resolve(name: str) -> str:
            return FAKE_TIP

        async def _is_anc(ancestor: str, descendant: str) -> bool:
            return True

        async def _find_wt(branch: str):
            return None

        git_ops_stub = types.SimpleNamespace(
            resolve_branch_sha=_resolve,
            is_ancestor=_is_anc,
            find_inflight_merge_worktree=_find_wt,
        )

        server3, mq3, _, _, _ = _build_merge_server(
            tmp_path,
            event_store=_RecordingES(),
            git_ops=git_ops_stub,
        )

        result = await asyncio.wait_for(
            _call_merge_request(
                server3,
                task_id='sc3',
                branch='sc3',
                worktree=str(tmp_path / 'wt-sc3'),
            ),
            timeout=2.0,
        )

        assert result.get('status') == 'already_merged', (
            f"Expected status='already_merged', got: {result}"
        )
        assert result.get('commit') == FAKE_TIP, (
            f"Expected commit={FAKE_TIP!r}, got: {result.get('commit')!r}"
        )
        assert 'request_id' not in result, (
            f"request_id must be absent on fast-path: {result}"
        )
        assert mq3.empty(), (
            f"Queue must be untouched (empty), qsize={mq3.qsize()}"
        )
        assert not any(e == EventType.merge_queued for e in events_recorded), (
            f"No merge_queued event must be emitted on fast-path, got: {events_recorded}"
        )

    async def test_scenario_4_merge_status_lifecycle(
        self, tmp_path: Path,
    ) -> None:
        """Row 4: merge_status across queued → verifying → done lifecycle.

        Drive states strictly at the worker/verify boundary (fake snapshot()).
        Records a terminal TerminalOutcomeRecord into harness._terminal_retention
        for the 'done' transition.
        """
        import time

        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            TerminalOutcomeRecord,
            TerminalOutcomeRetention,
        )

        worker = _FakeMergeWorker()
        retention = TerminalOutcomeRetention()
        server, mq, _, _, harness = _build_merge_server(
            tmp_path,
            worker=worker,
            retention=retention,
        )

        # Submit to get a real request_id
        result_submit = await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='sc4',
                branch='sc4',
                worktree=str(tmp_path / 'wt-sc4'),
                wait_secs=0,
            ),
            timeout=2.0,
        )
        rid = result_submit['request_id']

        # ── State 1: queued ────────────────────────────────────────────────
        worker.set_entries([{
            'request_id': rid,
            'branch': 'sc4',
            'task_id': 'sc4',
            'state': 'queued',
            'position': 0,
            'enqueued_at': time.time(),
        }])
        status_queued = await _call_merge_status(server, request_id=rid)
        assert status_queued['state'] == 'queued', (
            f"Expected 'queued' state, got: {status_queued}"
        )
        assert 'position' in status_queued or 'enqueued_at' in status_queued, (
            f"Live entry must include position or enqueued_at: {status_queued}"
        )

        # ── State 2: verifying ─────────────────────────────────────────────
        worker.set_entries([{
            'request_id': rid,
            'branch': 'sc4',
            'task_id': 'sc4',
            'state': 'verifying',
            'position': 0,
            'enqueued_at': time.time(),
        }])
        status_verifying = await _call_merge_status(server, request_id=rid)
        assert status_verifying['state'] == 'verifying', (
            f"Expected 'verifying' state, got: {status_verifying}"
        )

        # ── State 3: done (from retention ring) ────────────────────────────
        worker.set_entries([])  # worker no longer tracking it
        retention.record(TerminalOutcomeRecord(
            request_id=rid,
            task_id='sc4',
            branch='sc4',
            state='done',
            merge_sha='abc123sc4',
        ))

        status_done = await _call_merge_status(server, request_id=rid)
        assert status_done['state'] == 'done', (
            f"Expected 'done' from retention ring, got: {status_done}"
        )
        assert 'outcome' in status_done, (
            f"Terminal entry must include outcome: {status_done}"
        )

        # Cleanup
        req = mq.get_nowait()
        req.result.cancel()

    async def test_scenario_5_merge_status_restart_and_unknown(
        self, tmp_path: Path,
    ) -> None:
        """Row 5: merge_status after restart uses event store; unknown returns hint.

        (a) Emit merge_finalized directly to a real EventStore.  Build a fresh
        server with empty retention ring but same EventStore; assert merge_status
        returns the terminal state from the event store.
        (b) merge_status('mr-doesnotexist') → {state:'unknown', hint:'check git log main'}.
        """
        from orchestrator.event_store import (  # type: ignore[reportMissingImports]
            EventStore,
            EventType,
        )
        from orchestrator.merge_queue import (
            TerminalOutcomeRetention,  # type: ignore[reportMissingImports]
        )

        db_path = tmp_path / 'events-sc5.db'
        event_store = EventStore(db_path=db_path, run_id='sc5')

        KNOWN_RID = 'mr-sc5known01'

        # Emit a merge_finalized record directly into the event store
        event_store.emit(
            EventType.merge_finalized,
            task_id='sc5-task',
            phase='merge',
            data={
                'request_id': KNOWN_RID,
                'branch': 'sc5',
                'state': 'done',
                'snapshot_tip': None,
                'merge_sha': 'sha-sc5',
                'superseded_by': None,
                'generation': 1,
            },
        )

        # Build a FRESH server with empty retention (simulating restart)
        # but the same event_store
        from orchestrator.merge_queue import (
            InFlightMergeRegistry,  # type: ignore[reportMissingImports]
        )

        from escalation.queue import EscalationQueue  # type: ignore[reportMissingImports]

        fresh_retention = TerminalOutcomeRetention()
        fresh_harness = _FakeHarness(retention=fresh_retention)
        fresh_server = create_server(
            EscalationQueue(tmp_path / 'esc-sc5'),
            merge_queue=asyncio.Queue(),
            orch_config=_make_orch_config(tmp_path / 'repo-sc5'),
            event_store=event_store,
            harness=fresh_harness,
            merge_inflight_registry=InFlightMergeRegistry(),
            startup_sweep=False,
        )

        # (a) Known request_id → should come from event store
        status_known = await _call_merge_status(fresh_server, request_id=KNOWN_RID)
        assert status_known.get('state') == 'done', (
            f"Expected state='done' from event store after restart, got: {status_known}"
        )
        assert status_known.get('request_id') == KNOWN_RID, (
            f"Expected request_id={KNOWN_RID!r}, got: {status_known}"
        )

        # (b) Unknown id → {state:'unknown', hint:...}
        status_unknown = await _call_merge_status(fresh_server, request_id='mr-doesnotexist')
        assert status_unknown.get('state') == 'unknown', (
            f"Expected state='unknown' for unknown id, got: {status_unknown}"
        )
        assert 'hint' in status_unknown, f"Expected 'hint' key: {status_unknown}"
        assert 'check git log main' in status_unknown['hint'], (
            f"Expected 'check git log main' in hint, got: {status_unknown['hint']!r}"
        )

    async def test_scenario_6_explicit_cancel(
        self, tmp_path: Path,
    ) -> None:
        """Row 6: explicit cancel.

        Submit (wait_secs=0) to register a waiter.  Call merge_cancel(request_id).
        Assert {cancelled:True, state:'abandoned', reason:None}.  Yield event loop
        for _on_finalized to fire.  Assert merge_status(request_id) returns 'abandoned'
        (via event_store Tier 3).  Assert the queue is not halted (second branch queues).
        Extends test_cancel_pending_waiter_returns_cancelled_true with
        merge_status+queue-not-halted assertions.
        """
        from orchestrator.event_store import EventStore  # type: ignore[reportMissingImports]

        es = EventStore(db_path=tmp_path / 'events-sc6.db', run_id='sc6')
        server, mq, _, _, _ = _build_merge_server(tmp_path, event_store=es)

        result_mr = await _call_merge_request(
            server,
            task_id='sc6',
            branch='sc6',
            worktree=str(tmp_path / 'wt-sc6'),
            wait_secs=0,
        )
        assert result_mr['status'] == 'queued', f"Unexpected status: {result_mr}"
        rid = result_mr['request_id']

        result_cancel = await _call_merge_cancel(server, request_id=rid)
        assert result_cancel.get('cancelled') is True, (
            f"Expected cancelled=True, got: {result_cancel}"
        )
        assert result_cancel.get('state') == 'abandoned', (
            f"Expected state='abandoned', got: {result_cancel}"
        )
        assert result_cancel.get('reason') is None, (
            f"Expected reason=None, got: {result_cancel.get('reason')!r}"
        )

        # Yield event loop for _on_finalized done_callback to fire
        for _ in range(5):
            await asyncio.sleep(0)

        # merge_status must report 'abandoned' from event store (Tier 3)
        status = await _call_merge_status(server, request_id=rid)
        assert status.get('state') == 'abandoned', (
            f"Expected 'abandoned' from event store after cancel, got: {status}"
        )

        # Queue not halted: a second distinct branch can still be submitted and queued
        result2 = await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='sc6-b',
                branch='sc6-b',
                worktree=str(tmp_path / 'wt-sc6b'),
                wait_secs=0,
            ),
            timeout=2.0,
        )
        assert result2.get('status') == 'queued', (
            f"Expected 'queued' for second branch (queue not halted), got: {result2}"
        )

        # Cleanup
        req = mq.get_nowait()
        req.result.cancel()

    async def test_scenario_7_disconnect_is_not_cancel(
        self, tmp_path: Path,
    ) -> None:
        """Row 7: MCP disconnect does not cancel the entry (durable intent).

        Submit with wait_secs=30, advance loop, drain queue for request_id,
        cancel the merge_request Task (simulates disconnect).  Assert the entry
        future is NOT cancelled.  Manually record 'done' into retention ring.
        Build a NEW server with same retention; assert merge_status returns 'done'.
        Extends test_bounded_wait_disconnect_does_not_cancel_entry with
        new-session merge_status cross-check.
        """
        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            InFlightMergeRegistry,  # type: ignore[reportMissingImports]
            MergeOutcome,
            TerminalOutcomeRecord,
            TerminalOutcomeRetention,
        )

        from escalation.queue import EscalationQueue  # type: ignore[reportMissingImports]

        retention = TerminalOutcomeRetention()
        server, mq, _, _, _ = _build_merge_server(tmp_path, retention=retention)

        merge_task = asyncio.create_task(
            _call_merge_request(
                server,
                task_id='sc7',
                branch='sc7',
                worktree=str(tmp_path / 'wt-sc7'),
                wait_secs=30,
            )
        )

        # Advance event loop until the task blocks on the shielded await
        for _ in range(5):
            await asyncio.sleep(0)

        assert mq.qsize() == 1, f"Expected 1 entry in queue before disconnect, got {mq.qsize()}"
        req = mq.get_nowait()
        rid = req.request_id

        # Simulate client disconnect: cancel the merge_request Task
        merge_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await merge_task

        # Durable intent: the entry future must NOT be cancelled
        assert not req.result.cancelled(), (
            'Entry future must NOT be cancelled after MCP disconnect — shield protects it'
        )

        # Manually record the terminal 'done' outcome into the retention ring
        retention.record(TerminalOutcomeRecord(
            request_id=rid,
            task_id='sc7',
            branch='sc7',
            state='done',
            merge_sha='sha-sc7',
        ))
        # Resolve the entry future (worker finished)
        req.result.set_result(MergeOutcome('done', reason='late resolve'))

        # Build a fresh server sharing the same retention ring (simulating new session)
        fresh_server = create_server(
            EscalationQueue(tmp_path / 'esc-sc7-fresh'),
            merge_queue=asyncio.Queue(),
            orch_config=_make_orch_config(tmp_path / 'repo-sc7'),
            harness=_FakeHarness(retention=retention),
            merge_inflight_registry=InFlightMergeRegistry(),
            startup_sweep=False,
        )

        # New session: merge_status must find 'done' from retention ring
        status = await _call_merge_status(fresh_server, request_id=rid)
        assert status.get('state') == 'done', (
            f"Expected 'done' from retention ring in new session, got: {status}"
        )

    async def test_scenario_8_coalesce_returns_existing_request_id(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Row 8: coalesce returns the existing request_id (D8).

        First submit for branch B → 'queued', R1 in registry.
        Second submit for B at same tip → 'attached', request_id==R1.
        merge_coalesced event emitted on second submit.
        Extends test_wait_secs_zero_inflight_returns_attached /
        test_default_call_inflight_branch_returns_attached with the
        request_id-equality + merge_coalesced-event assertions.
        """
        coalesced_spy: list[int] = []
        monkeypatch.setattr(
            'orchestrator.merge_queue._emit_merge_coalesced',
            lambda *a, **kw: coalesced_spy.append(1),
        )

        server, mq, _, _, _ = _build_merge_server(tmp_path)

        # First submit: acquires the registry slot for branch B
        result1 = await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='sc8',
                branch='sc8',
                worktree=str(tmp_path / 'wt-sc8'),
                wait_secs=0,
            ),
            timeout=2.0,
        )
        assert result1.get('status') == 'queued', f"First submit must be 'queued': {result1}"
        r1 = result1['request_id']
        assert r1.startswith('mr-'), f"Expected mr- prefix, got: {r1!r}"

        # Second submit for same branch B: must return 'attached' with R1
        result2 = await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='sc8',
                branch='sc8',
                worktree=str(tmp_path / 'wt-sc8b'),
                wait_secs=0,
            ),
            timeout=2.0,
        )
        assert result2.get('status') == 'attached', (
            f"Expected status='attached' on coalesce, got: {result2}"
        )
        assert result2.get('request_id') == r1, (
            f"Expected request_id={r1!r} (existing entry's id, D8), got: {result2.get('request_id')!r}"
        )
        assert len(coalesced_spy) == 1, (
            f"Expected exactly 1 merge_coalesced event, got: {len(coalesced_spy)}"
        )

        # Cleanup
        req = mq.get_nowait()
        req.result.cancel()

    async def test_scenario_14_submit_then_poll_protocol(
        self, tmp_path: Path,
    ) -> None:
        """Row 14: submit-then-poll protocol exercised end-to-end on the MCP surface.

        §7.3 runtime invariant — the four merge-calling skills document that
        completion is awaited only via merge_status (not via a blocking wait on
        the merge_request return).  This test exercises that pattern:

        (a) SUBMIT half: merge_request(branch=B, wait_secs=0) returns promptly
            with status=='queued' and a valid 'mr-' request_id R.
        (b) POLL half: set the fake worker snapshot so it echoes the submitted
            entry, then call merge_status(request_id=R) (and merge_status(branch=B));
            assert both resolve to a coherent non-'unknown' state whose
            request_id and branch match the submission.

        This is the runtime analogue of the §7.3 invariant — exercised end-to-end
        through the real MCP tool layer rather than via prose-pinning assertions.
        Reuses _build_merge_server + _FakeMergeWorker + _call_merge_request +
        _call_merge_status from pre-1.
        """
        worker = _FakeMergeWorker()
        server, mq, _, _, _ = _build_merge_server(tmp_path, worker=worker)

        # (a) SUBMIT: non-blocking, must return promptly with 'queued' + valid R
        result = await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='sc14',
                branch='sc14',
                worktree=str(tmp_path / 'wt-sc14'),
                wait_secs=0,
            ),
            timeout=3.0,
        )
        assert result.get('status') == 'queued', (
            f"SUBMIT half: expected status='queued', got: {result}"
        )
        R = result.get('request_id', '')
        assert R.startswith('mr-'), (
            f"SUBMIT half: expected 'mr-' request_id, got: {R!r}"
        )

        # (b) POLL: set the worker snapshot to echo the submitted entry, then
        # call merge_status(request_id=R) and merge_status(branch='sc14')
        worker.set_entries([{
            'request_id': R,
            'branch': 'sc14',
            'task_id': 'sc14',
            'state': 'queued',
            'position': 0,
            'enqueued_at': 0,
        }])

        # Poll by request_id
        status_by_rid = await asyncio.wait_for(
            _call_merge_status(server, request_id=R),
            timeout=2.0,
        )
        assert status_by_rid.get('state') != 'unknown', (
            f"POLL by request_id: expected non-'unknown' state, got: {status_by_rid}"
        )
        assert status_by_rid.get('request_id') == R, (
            f"POLL by request_id: expected request_id={R!r}, got: {status_by_rid.get('request_id')!r}"
        )

        # Poll by branch
        status_by_branch = await asyncio.wait_for(
            _call_merge_status(server, branch='sc14'),
            timeout=2.0,
        )
        assert status_by_branch.get('state') != 'unknown', (
            f"POLL by branch: expected non-'unknown' state, got: {status_by_branch}"
        )
        assert status_by_branch.get('request_id') == R, (
            f"POLL by branch: expected request_id={R!r}, got: {status_by_branch.get('request_id')!r}"
        )

        # Cleanup
        req = mq.get_nowait()
        req.result.cancel()

    async def test_scenario_superseded_ring_tier(
        self, tmp_path: Path,
    ) -> None:
        """Superseded surface via retention ring (Tier 2).

        Record two TerminalOutcomeRecords directly into the ring:
          1. absorbed request (state='superseded', superseded_by='mr-train')
          2. train request    (state='done', merge_sha='sha-train')

        Assert:
          - merge_status(request_id='mr-absorbed') → state='superseded', outcome='superseded',
            superseded_by='mr-train'
          - merge_status(request_id='mr-train')    → state='done'

        RED: _map_terminal_state collapses 'superseded' → 'blocked', and the ring meta
        never threads superseded_by, so neither assertion passes until step-2 impl.
        """
        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            TerminalOutcomeRecord,
            TerminalOutcomeRetention,
        )

        retention = TerminalOutcomeRetention()
        server, _, _, _, _ = _build_merge_server(tmp_path, retention=retention)

        # Record absorbed request into ring
        retention.record(TerminalOutcomeRecord(
            request_id='mr-absorbed',
            task_id='task-absorbed',
            branch='branch-absorbed',
            state='superseded',
            superseded_by='mr-train',
        ))
        # Record train request into ring
        retention.record(TerminalOutcomeRecord(
            request_id='mr-train',
            task_id='task-train',
            branch='branch-train',
            state='done',
            merge_sha='sha-train',
        ))

        # Assert absorbed → state='superseded', outcome='superseded', superseded_by='mr-train'
        status_absorbed = await _call_merge_status(server, request_id='mr-absorbed')
        assert status_absorbed.get('state') == 'superseded', (
            f"Expected state='superseded' for absorbed request, got: {status_absorbed}"
        )
        assert status_absorbed.get('outcome') == 'superseded', (
            f"Expected outcome='superseded' for absorbed request, got: {status_absorbed}"
        )
        assert status_absorbed.get('superseded_by') == 'mr-train', (
            f"Expected superseded_by='mr-train', got: {status_absorbed}"
        )

        # Assert train → state='done'
        status_train = await _call_merge_status(server, request_id='mr-train')
        assert status_train.get('state') == 'done', (
            f"Expected state='done' for train request, got: {status_train}"
        )

    async def test_scenario_superseded_event_store_tier(
        self, tmp_path: Path,
    ) -> None:
        """Superseded surface via event store (Tier 3) — post-restart durability.

        Emit a real EventStore merge_finalized event with state='superseded' and
        superseded_by='mr-train2'.  Build a FRESH server with an EMPTY
        TerminalOutcomeRetention but the SAME event_store (simulating a restart /
        ring eviction).

        Assert merge_status(request_id='mr-absorbed2') returns:
          - state == 'superseded'
          - outcome == 'superseded'
          - superseded_by == 'mr-train2'

        RED: event_store.latest_merge_finalized drops superseded_by from its
        returned dict, so _durable_terminal_state Tier 3 never threads it into
        meta and the response never carries superseded_by.
        """
        from orchestrator.event_store import (  # type: ignore[reportMissingImports]
            EventStore,
            EventType,
        )
        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            InFlightMergeRegistry,
            TerminalOutcomeRetention,
        )

        db_path = tmp_path / 'events-sup3.db'
        event_store = EventStore(db_path=db_path, run_id='sup3')

        ABSORBED_RID = 'mr-absorbed2'
        TRAIN_RID = 'mr-train2'

        # Emit a merge_finalized record for the absorbed request
        event_store.emit(
            EventType.merge_finalized,
            task_id='task-absorbed2',
            phase='merge',
            data={
                'request_id': ABSORBED_RID,
                'branch': 'branch-absorbed2',
                'state': 'superseded',
                'snapshot_tip': None,
                'merge_sha': None,
                'superseded_by': TRAIN_RID,
                'generation': 1,
            },
        )

        # Build a FRESH server with empty retention (simulating restart / ring eviction)
        # but the same event_store
        fresh_retention = TerminalOutcomeRetention()
        fresh_harness = _FakeHarness(retention=fresh_retention)
        fresh_server = create_server(
            EscalationQueue(tmp_path / 'esc-sup3'),
            merge_queue=asyncio.Queue(),
            orch_config=_make_orch_config(tmp_path / 'repo-sup3'),
            event_store=event_store,
            harness=fresh_harness,
            merge_inflight_registry=InFlightMergeRegistry(),
            startup_sweep=False,
        )

        # Assert absorbed request resolved from event store post-restart
        status = await _call_merge_status(fresh_server, request_id=ABSORBED_RID)
        assert status.get('state') == 'superseded', (
            f"Expected state='superseded' from event store after restart, got: {status}"
        )
        assert status.get('outcome') == 'superseded', (
            f"Expected outcome='superseded', got: {status}"
        )
        assert status.get('superseded_by') == TRAIN_RID, (
            f"Expected superseded_by={TRAIN_RID!r}, got: {status}"
        )

    async def test_scenario_superseded_merge_request_bounded_wait(
        self, tmp_path: Path,
    ) -> None:
        """merge_request bounded-wait returns status='superseded' + superseded_by.

        Submit merge_request(wait_secs=5) while a background worker dequeues the
        entry and resolves req.result with MergeOutcome('superseded',
        superseded_by='mr-train3').

        Assert the returned dict has:
          - status == 'superseded'
          - superseded_by == 'mr-train3'

        RED: the bounded-wait terminal response (server.py ~905-915) builds
        status/request_id/reason/conflict_details/push_status/commit and the
        failure_diagnostic conditional, but never includes superseded_by.
        """
        from orchestrator.merge_queue import MergeOutcome  # type: ignore[reportMissingImports]

        server, mq, _, _, _ = _build_merge_server(tmp_path)

        async def _worker():
            req = await mq.get()
            req.result.set_result(MergeOutcome('superseded', superseded_by='mr-train3'))

        worker_task = asyncio.create_task(_worker())

        result = await asyncio.wait_for(
            _call_merge_request(
                server,
                task_id='sc-sup5',
                branch='sc-sup5',
                worktree=str(tmp_path / 'wt-sup5'),
                wait_secs=5,
            ),
            timeout=5.0,
        )
        await worker_task

        assert result.get('status') == 'superseded', (
            f"Expected status='superseded', got: {result}"
        )
        assert result.get('superseded_by') == 'mr-train3', (
            f"Expected superseded_by='mr-train3', got: {result}"
        )
