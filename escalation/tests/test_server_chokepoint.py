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
