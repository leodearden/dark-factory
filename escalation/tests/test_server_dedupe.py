"""Tests for escalation deduplication via the MCP server tools.

Mirrors the FastMCP unit-test pattern from test_release_workflow.py:
    tool = await server.get_tool('escalate_blocker')
    result = await tool.fn(...)

All tests use tmp_path-backed EscalationQueue so they are fully isolated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from escalation.dedupe import DedupeConfig
from escalation.queue import EscalationQueue
from escalation.server import create_server

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server(queue: EscalationQueue, dedupe_config: DedupeConfig | None = None):
    return create_server(queue, dedupe_config=dedupe_config)


async def _blocker(server, **kwargs: Any) -> dict[str, Any]:
    tool = await server.get_tool('escalate_blocker')
    # escalate_blocker is a sync tool — tool.fn() returns dict directly
    return tool.fn(**kwargs)


async def _info(server, **kwargs: Any) -> dict[str, Any]:
    tool = await server.get_tool('escalate_info')
    # escalate_info is a sync tool — tool.fn() returns dict directly
    return tool.fn(**kwargs)


def _queue_root_files(queue: EscalationQueue) -> list[Path]:
    """Return all esc-*.json files in the queue root (excludes archive)."""
    return sorted(queue.queue_dir.glob('esc-*.json'))


# ---------------------------------------------------------------------------
# TestEscalateBlockerDedupe
# ---------------------------------------------------------------------------


class TestEscalateBlockerDedupe:
    """escalate_blocker deduplication — two infra_issue calls with similar summaries."""

    @pytest.mark.asyncio
    async def test_first_call_creates_file(self, tmp_path: Path):
        """(a) First escalate_blocker creates exactly one esc-*.json."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = _make_server(queue)

        result = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )

        assert result['status'] == 'queued'
        assert result['action'] == 'terminate_cleanly'
        assert 'id' in result

        files = _queue_root_files(queue)
        assert len(files) == 1, f'Expected exactly 1 file, got: {files}'

    @pytest.mark.asyncio
    async def test_second_call_dedupes_to_parent(self, tmp_path: Path):
        """(b) Second call with similar summary dedupes: no new file, parent bumped."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = _make_server(queue)

        first = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )
        parent_id = first['id']

        # Different tail / casing — same first 3 tokens after normalisation
        second = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='Fused-memory  CONNECTION timeout!',
        )

        assert second['status'] == 'dedup_skipped'
        assert second['parent_id'] == parent_id
        assert second['action'] == 'terminate_cleanly'

        # Still exactly one file in queue root
        files = _queue_root_files(queue)
        assert len(files) == 1, f'Expected 1 file after dedupe, got: {files}'

        # Parent on disk has dedupe_count == 1 and the child id in dedupe_children
        from escalation.models import Escalation
        parent = Escalation.from_json(files[0].read_text())
        assert parent.dedupe_count == 1
        assert len(parent.dedupe_children) == 1
        # child_id in the response must match what was stored in dedupe_children
        assert 'child_id' in second
        assert second['child_id'] == parent.dedupe_children[0]

    @pytest.mark.asyncio
    async def test_notify_callback_fires_once_for_parent_not_for_dedupe(self, tmp_path: Path):
        """(c) notify_callback fires for the parent submit but NOT for the deduped call."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = _make_server(queue)

        fired_ids: list[str] = []
        queue.set_notify_callback(lambda esc: fired_ids.append(esc.id))

        first = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )
        parent_id = first['id']

        await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='Fused-memory  CONNECTION timeout!',
        )

        # Callback fires exactly once — only for the original submit
        assert fired_ids == [parent_id], (
            f'Expected notify callback to fire exactly once for parent; '
            f'got: {fired_ids}'
        )


# ---------------------------------------------------------------------------
# TestEscalateInfoDedupe
# ---------------------------------------------------------------------------


class TestEscalateInfoDedupe:
    """escalate_info also gains dedupe behaviour via the same _submit_or_dedupe helper."""

    @pytest.mark.asyncio
    async def test_two_info_calls_fold(self, tmp_path: Path):
        """(a) Two escalate_info calls with the same infra_issue summary fold to one file.

        Second response: {id: parent_id, status: dedup_skipped, parent_id: parent_id}
        Note: NO 'action' key — that is only on the blocker path.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = _make_server(queue)

        first = await _info(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )
        parent_id = first['id']
        assert first['status'] == 'queued'
        assert 'action' not in first, 'escalate_info must NOT return action key'

        second = await _info(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='Fused-memory  CONNECTION timeout!',
        )

        assert second['status'] == 'dedup_skipped'
        assert second['parent_id'] == parent_id
        assert 'action' not in second, 'escalate_info must NOT return action key'

        # One file on disk; parent.dedupe_count == 1
        files = _queue_root_files(queue)
        assert len(files) == 1
        from escalation.models import Escalation
        parent = Escalation.from_json(files[0].read_text())
        assert parent.dedupe_count == 1

    @pytest.mark.asyncio
    async def test_cross_severity_info_then_blocker(self, tmp_path: Path):
        """(b) Cross-severity: info creates parent, blocker dedupes against it.

        Parent's severity must be PROMOTED to 'blocking' — absorbing a blocker
        child into an info parent must escalate urgency so the steward UI
        treats the parent with blocker-level urgency.

        This test FAILS on current main because attach_dedupe_child never
        mutates parent.severity.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = _make_server(queue)

        first = await _info(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )
        parent_id = first['id']
        assert first['status'] == 'queued'

        second = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='Fused-memory  CONNECTION timeout!',
        )

        assert second['status'] == 'dedup_skipped'
        assert second['parent_id'] == parent_id
        assert second['action'] == 'terminate_cleanly'
        assert 'child_id' in second

        from escalation.models import Escalation
        files = _queue_root_files(queue)
        parent = Escalation.from_json(files[0].read_text())
        assert parent.severity == 'blocking', (
            'Parent severity must be promoted from info to blocking after '
            'absorbing a blocker child'
        )
        assert parent.dedupe_count == 1
        assert len(parent.dedupe_children) == 1
        assert second['child_id'] == parent.dedupe_children[0]

    @pytest.mark.asyncio
    async def test_cross_severity_blocker_then_info(self, tmp_path: Path):
        """(c) Cross-severity: blocker creates parent, info dedupes against it.

        Parent's severity stays 'blocking' — info call does not demote it.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = _make_server(queue)

        first = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )
        parent_id = first['id']

        second = await _info(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='Fused-memory  CONNECTION timeout!',
        )

        assert second['status'] == 'dedup_skipped'
        assert second['parent_id'] == parent_id

        from escalation.models import Escalation
        files = _queue_root_files(queue)
        parent = Escalation.from_json(files[0].read_text())
        assert parent.severity == 'blocking', (
            'Parent severity must remain blocking after info dedupe'
        )
        assert parent.dedupe_count == 1


# ---------------------------------------------------------------------------
# TestCrossTaskDedupe
# ---------------------------------------------------------------------------


class TestCrossTaskDedupe:
    """Cross-task infra_issue folding: two different task_ids, same summary -> same parent."""

    @pytest.mark.asyncio
    async def test_cross_task_blocker_dedupes(self, tmp_path: Path):
        """escalate_blocker from task_id='42' then task_id='99' with same infra_issue summary:
        second call dedupes to the parent id from task_id='42'.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = _make_server(queue)

        first = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )
        parent_id = first['id']

        second = await _blocker(
            server,
            task_id='99',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 9999',
        )

        assert second['status'] == 'dedup_skipped'
        assert second['parent_id'] == parent_id

        # Still exactly one file — the task-42 parent
        files = _queue_root_files(queue)
        assert len(files) == 1

        from escalation.models import Escalation
        parent = Escalation.from_json(files[0].read_text())
        assert parent.task_id == '42'
        assert parent.dedupe_count == 1


# ---------------------------------------------------------------------------
# TestDedupeGates
# ---------------------------------------------------------------------------


class TestDedupeGates:
    """Short-circuit gates: disabled flag and non-member category produce two files."""

    @pytest.mark.asyncio
    async def test_disabled_flag_produces_two_files(self, tmp_path: Path):
        """(a) DedupeConfig(infra_dedupe_enabled=False): two identical infra_issue calls
        produce two esc-*.json files and neither response has status='dedup_skipped'."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = _make_server(queue, dedupe_config=DedupeConfig(infra_dedupe_enabled=False))

        first = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )
        second = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 9999',
        )

        assert first['status'] == 'queued'
        assert second['status'] == 'queued'
        assert first.get('status') != 'dedup_skipped'
        assert second.get('status') != 'dedup_skipped'

        files = _queue_root_files(queue)
        assert len(files) == 2, (
            f'Expected 2 files when dedupe disabled; got: {files}'
        )

    @pytest.mark.asyncio
    async def test_non_infra_category_produces_two_files(self, tmp_path: Path):
        """(b) risk_identified (not in infra_dedupe_categories) with same summary
        produces two esc-*.json files, no dedup."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = _make_server(queue)  # default config: only infra_issue is in scope

        first = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='risk_identified',
            summary='fused-memory connection timeout on port 8002',
        )
        second = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='risk_identified',
            summary='fused-memory connection timeout on port 9999',
        )

        assert first['status'] == 'queued'
        assert second['status'] == 'queued'
        assert first.get('status') != 'dedup_skipped'
        assert second.get('status') != 'dedup_skipped'

        files = _queue_root_files(queue)
        assert len(files) == 2, (
            f'Expected 2 files for non-infra category; got: {files}'
        )


# ---------------------------------------------------------------------------
# TestDedupeTOCTOURace
# ---------------------------------------------------------------------------


class TestDedupeTOCTOURace:
    """TOCTOU race: parent resolved between find_dedupe_parent() and attach_dedupe_child().

    Simulates a concurrent resolve() that archives the parent file in the narrow window
    between find_dedupe_parent() returning a parent_id and attach_dedupe_child() trying
    to load that parent from queue_dir.  In this window attach_dedupe_child returns None
    because the file no longer exists in the queue root.

    The escalation must NOT be silently dropped — it must fall through to queue.submit().
    """

    @pytest.mark.asyncio
    async def test_toctou_falls_through_to_submit(self, tmp_path: Path, monkeypatch):
        """Parent resolved between find and attach → second escalation is queued normally.

        Setup:
        (1) Submit a first infra_issue blocker → parent_id.
        (2) Monkeypatch escalation.server.find_dedupe_parent to a wrapper that resolves
            the parent (archiving the file) and then returns parent_id as if it were still
            pending — mimicking the TOCTOU window.
        (3) Call escalate_blocker a second time with a similar infra_issue summary.

        Assertions:
        (a) The second result has status='queued' and NOT 'dedup_skipped' — the escalation
            must NOT be silently dropped.
        (b) Exactly one new esc-*.json exists in the queue root (the parent was archived,
            so the root holds only the newly submitted child).
        (c) The second result['id'] matches the filename id of that new file — confirming
            queue.submit(esc) was actually invoked with the candidate escalation.

        This test FAILS on the current implementation because attach_dedupe_child returns
        None silently when the parent is already archived, but the helper still returns
        {'status': 'dedup_skipped'} unconditionally, dropping the escalation.
        """
        import escalation.server as server_module

        queue = EscalationQueue(tmp_path / 'esc')
        server = _make_server(queue)

        # (1) Submit first infra_issue blocker — establishes the parent.
        first = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )
        assert first['status'] == 'queued'

        # (2) Monkeypatch find_dedupe_parent so that after finding the parent it
        #     concurrently resolves it (archiving the file) before returning.
        #     This replicates the race where the parent is resolved between the
        #     queue scan and the attach call.
        _original_find = server_module.find_dedupe_parent

        def _racing_find(q, esc, cfg, now=None):
            result = _original_find(q, esc, cfg, now=now)
            if result is not None:
                # Concurrent resolve: parent is archived before attach_dedupe_child runs.
                q.resolve(result, resolution='raced')
            return result  # still returns the id — simulating the stale read

        monkeypatch.setattr(server_module, 'find_dedupe_parent', _racing_find)

        # (3) Second blocker with a similar summary — hits the race window.
        second = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='Fused-memory  CONNECTION timeout!',
        )

        # (a) Must NOT be silently dropped as 'dedup_skipped' — must be queued.
        assert second['status'] == 'queued', (
            f'Expected status=queued after TOCTOU race, got: {second}'
        )

        # (b) A new esc-*.json must exist in the queue root.
        #     The parent was archived → root holds only the newly submitted escalation.
        files = _queue_root_files(queue)
        assert len(files) == 1, (
            f'Expected exactly 1 file (the new escalation) in queue root after TOCTOU; '
            f'got: {files}'
        )

        # (c) The returned id must match the file on disk — queue.submit() was invoked.
        new_file_id = files[0].stem  # e.g. 'esc-42-2'
        assert second['id'] == new_file_id, (
            f"result id {second['id']!r} does not match file stem {new_file_id!r}"
        )


# ---------------------------------------------------------------------------
# TestDedupeChildIdContract
# ---------------------------------------------------------------------------


class TestDedupeChildIdContract:
    """End-to-end tests that pin the child_id audit trail at the server tier."""

    @pytest.mark.asyncio
    async def test_child_id_in_response_matches_dedupe_children(self, tmp_path: Path):
        """child_id in the dedup_skipped response equals dedupe_children[0] on disk."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = _make_server(queue)

        first = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )
        parent_id = first['id']

        second = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='Fused-memory  CONNECTION timeout!',
        )

        assert second['status'] == 'dedup_skipped'
        assert second['parent_id'] == parent_id
        assert 'child_id' in second

        from escalation.models import Escalation
        files = _queue_root_files(queue)
        parent = Escalation.from_json(files[0].read_text())
        assert parent.dedupe_children == [second['child_id']]

    @pytest.mark.asyncio
    async def test_two_deduped_children_stored_in_submission_order(self, tmp_path: Path):
        """Three calls: first is parent, second and third are folded.

        dedupe_children on disk must list the children in submission order,
        and each response's child_id must correspond to its entry in the list.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = _make_server(queue)

        await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )

        second = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='Fused-memory  CONNECTION timeout!',
        )
        third = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 9999',
        )

        assert second['status'] == 'dedup_skipped'
        assert third['status'] == 'dedup_skipped'

        from escalation.models import Escalation
        files = _queue_root_files(queue)
        assert len(files) == 1
        parent = Escalation.from_json(files[0].read_text())

        assert parent.dedupe_count == 2
        assert len(parent.dedupe_children) == 2
        # Submission order must be preserved
        assert parent.dedupe_children[0] == second['child_id']
        assert parent.dedupe_children[1] == third['child_id']
