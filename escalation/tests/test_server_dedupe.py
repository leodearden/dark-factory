"""Tests for escalation deduplication via the MCP server tools.

Uses the FastMCP async unit-test pattern from test_release_workflow.py:
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
    return await tool.fn(**kwargs)


async def _info(server, **kwargs: Any) -> dict[str, Any]:
    tool = await server.get_tool('escalate_info')
    return await tool.fn(**kwargs)


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
        (2) Monkeypatch escalation.dedupe.find_dedupe_parent (NOT escalation.server) to
            a wrapper that resolves the parent (archiving the file) and then returns
            parent_id as if it were still pending — mimicking the TOCTOU window.
            After step-14 the server delegates to dedupe.submit_or_dedupe, which calls
            find_dedupe_parent from escalation.dedupe; patching the dedupe module is
            therefore the correct target post-delegation.
        (3) Call escalate_blocker a second time with a similar infra_issue summary.

        Assertions:
        (a) The second result has status='queued' and NOT 'dedup_skipped' — the escalation
            must NOT be silently dropped.
        (b) Exactly one new esc-*.json exists in the queue root (the parent was archived,
            so the root holds only the newly submitted child).
        (c) The second result['id'] matches the filename id of that new file — confirming
            queue.submit(esc) was actually invoked with the candidate escalation.
        """
        import escalation.dedupe as dedupe_module

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

        # (2) Monkeypatch escalation.dedupe.find_dedupe_parent (the call site after
        #     step-14 delegation) so that after finding the parent it concurrently
        #     resolves it (archiving the file) before returning.
        #     This replicates the race where the parent is resolved between the
        #     queue scan and the attach call.
        _original_find = dedupe_module.find_dedupe_parent

        def _racing_find(q, esc, cfg, now=None):
            result = _original_find(q, esc, cfg, now=now)
            if result is not None:
                # Concurrent resolve: parent is archived before attach_dedupe_child runs.
                q.resolve(result, resolution='raced')
            return result  # still returns the id — simulating the stale read

        monkeypatch.setattr(dedupe_module, 'find_dedupe_parent', _racing_find)

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


# ---------------------------------------------------------------------------
# TestCrossTaskChildResumeContract
# ---------------------------------------------------------------------------


class TestCrossTaskChildResumeContract:
    """Characterizes the re-run-on-next-invocation contract for cross-task deduped children.

    A child whose task_id differs from the parent's receives 'dedup_skipped' and
    has NO on-disk pending escalation under its own task_id.  The child workflow's
    wait predicate (get_by_task(child_task_id, status='pending', level=0)) therefore
    returns [] immediately — the child does not block waiting on the parent's resolution.
    The orchestrator's normal re-invocation of the child task on its next workflow cycle
    is the resume mechanism.

    See DESIGN.md "Escalation cross-task dedupe: re-run-on-next-invocation contract".
    """

    @pytest.mark.asyncio
    async def test_cross_task_child_resume_contract(self, tmp_path: Path):
        """Cross-task deduped child has no on-disk escalation; resolve introduces no obstacle.

        Sequence:
        (1) Parent blocker from task 42.
        (2) Deduped blocker from task 99 — same infra_issue summary.
        (3) get_by_task('99', pending, level=0) == [] before resolve (contract: no obstacle).
        (4) get_by_task('42', pending, level=0) contains the parent.
        (5) Resolve the parent via resolve_issue tool.
        (6) get_by_task('99', pending, level=0) == [] after resolve (no ghost left behind).
        (7) Parent is now resolved/archived.
        (8) A fresh escalation from task 99 with a different summary submits cleanly
            (status='queued') — no lingering dedupe block from the resolved parent.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = _make_server(queue)

        # (1) Submit parent blocker from task 42.
        first = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )
        assert first['status'] == 'queued'
        parent_id = first['id']

        # (2) Submit cross-task blocker from task 99 — dedupes into parent under task 42.
        second = await _blocker(
            server,
            task_id='99',
            agent_role='implementer',
            category='infra_issue',
            summary='Fused-memory  CONNECTION timeout!',
        )
        assert second['status'] == 'dedup_skipped'
        assert second['parent_id'] == parent_id
        assert second['action'] == 'terminate_cleanly'

        # (3) No on-disk pending escalation under task 99's own task_id.
        #     This is the key contract: the child's workflow wait predicate returns []
        #     immediately, so the child does not block waiting on the parent's resolution.
        pending_99_before = queue.get_by_task('99', status='pending', level=0)
        assert pending_99_before == [], (
            'Cross-task deduped child must have no on-disk pending escalation '
            'under its own task_id — the child workflow must not block.'
        )

        # (4) The parent is still pending under task 42.
        pending_42 = queue.get_by_task('42', status='pending', level=0)
        assert len(pending_42) == 1
        assert pending_42[0].id == parent_id

        # (5) Resolve the parent.
        resolve_tool = await server.get_tool('resolve_issue')
        assert resolve_tool is not None
        resolved = resolve_tool.fn(escalation_id=parent_id, resolution='fused-memory restarted')  # type: ignore[attr-defined]
        assert resolved.get('status') in ('resolved', 'dismissed')

        # (6) Still no pending escalation under task 99 after the parent resolves.
        pending_99_after = queue.get_by_task('99', status='pending', level=0)
        assert pending_99_after == [], (
            'Resolving the parent must not create any pending escalation under '
            'the child task_id — no ghost left behind.'
        )

        # (7) The parent is now archived/resolved.
        resolved_parent = queue.get(parent_id)
        assert resolved_parent is not None
        assert resolved_parent.status in ('resolved', 'dismissed')

        # (8a) A fresh infra_issue from task 99 with a clearly distinct summary submits
        #      cleanly — this sub-case exercises the full dedupe code path.
        #      Both dedupe gates pass (infra_dedupe_enabled=True by default;
        #      'infra_issue' is in infra_dedupe_categories), so find_dedupe_parent IS
        #      called.  find_dedupe_parent scans only queue.get_pending(), so the
        #      resolved/archived parent from step (5) is invisible to it — no stale
        #      dedupe block.  The candidate's distinct dedupe key ('qdrant','connection',
        #      'refused') vs the parent's ('fusedmemory','connection','timeout') provides
        #      defense-in-depth: even if the similarity heuristic is later tuned, two
        #      clearly disjoint keys will never fold.  status='queued' proves there is no
        #      lingering block from the archived parent (no-ghost contract).
        fresh_infra = await _blocker(
            server,
            task_id='99',
            agent_role='implementer',
            category='infra_issue',
            summary='qdrant connection refused on port 6333',
        )
        assert fresh_infra['status'] == 'queued', (
            'infra_issue dedupe path: archived parent must not match — '
            'find_dedupe_parent scans only pending escalations; '
            f'got: {fresh_infra}'
        )

        # (8b) A fresh scope_violation from task 99 also submits cleanly, covering the
        #      orthogonal gate-2 short-circuit path.  'scope_violation' is not in
        #      cfg.infra_dedupe_categories, so _submit_or_dedupe short-circuits before
        #      find_dedupe_parent is called.  The escalation is queued normally regardless
        #      of any archived parents — the two status='queued' outcomes (gate-2
        #      short-circuit vs. dedupe-path-with-no-match) have meaningfully different
        #      causes and are asserted separately to document both contracts.
        fresh_sv = await _blocker(
            server,
            task_id='99',
            agent_role='implementer',
            category='scope_violation',
            summary='task 99 needs to write outside its locked modules',
        )
        assert fresh_sv['status'] == 'queued', (
            'scope_violation: gate-2 short-circuit must yield queued — '
            'category not in infra_dedupe_categories bypasses find_dedupe_parent; '
            f'got: {fresh_sv}'
        )
