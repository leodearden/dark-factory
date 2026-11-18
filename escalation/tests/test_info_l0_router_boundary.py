"""The escalation-server rows of the info-L0 router boundary table.

plans/info-l0-disposition-router-prd.md §Boundary-test sketch rows #9, #10 and
the park half of #18; the orchestrator rows live in
orchestrator/tests/test_info_l0_router_boundary.py.  The esc-4314-2 cluster
member is unrecoverable, so ``_member`` rebuilds its shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _filing_tools import call_promote
from _resolve_issue_call import call_resolve_issue
from _tool_calls import call_tool

from escalation.models import Escalation
from escalation.pins import is_queue_handoff
from escalation.queue import EscalationQueue
from escalation.server import create_server

TASK_ID = '4314'
WATCHER = 'escalation-watcher-auto'
DIGEST = 'Closed at L1 by the info-only digest: informational notes, no human action needed.'


def _member(queue: EscalationQueue, *, severity: str = 'info') -> Escalation:
    """The esc-4314-2 shape: the pre-router reaper's verbatim info promotion of an architect note."""
    member = Escalation(
        id=queue.make_id(TASK_ID), task_id=TASK_ID, agent_role='harness-orphan-reaper',
        severity=severity, category='design_concern',
        summary='Orphan L0 (900s old, no active workflow): plan touches a module with a pending refactor',
        detail='The refactor of lib.py in flight elsewhere may conflict with this plan.',
        suggested_action='manual_intervention', level=1,
    )
    queue.submit(member)
    return member


def _promote_args(members: list[Escalation], **overrides) -> dict:
    return {
        'task_id': TASK_ID,
        'agent_role': WATCHER,
        'member_ids': [m.id for m in members],
        'root_cause': 'informational architect notes about one module',
        'evidence': 'every member is an informational note',
        'options': ['A: close the members at L1', 'B: convert to a task'],
        'summary': 'info-only cluster',
        **overrides,
    }


def _level_two_records(queue: EscalationQueue) -> list[Escalation]:
    """Every level-2 record on disk, archive included."""
    records = (Escalation.from_json(path.read_text()) for path in queue.queue_dir.rglob('esc-*.json'))
    return [record for record in records if record.level == 2]


def _file_bytes(queue: EscalationQueue, esc_id: str) -> bytes:
    return (queue.queue_dir / f'{esc_id}.json').read_bytes()


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'escalations')


@pytest.fixture
def server(queue: EscalationQueue):
    return create_server(queue, startup_sweep=False)


@pytest.mark.asyncio
class TestRow09ExplicitSeverityStillPromotes:
    async def test_a_fresh_all_info_cluster_promoted_blocking_gates(self, queue, server):
        members = [_member(queue), _member(queue)]

        result = await call_promote(server, **_promote_args(members, severity='blocking'))

        assert result['status'] == 'created', result
        minted = queue.get(result['id'])
        assert minted is not None
        assert (minted.level, minted.severity) == (2, 'blocking')
        assert minted.members == [m.id for m in members]
        assert is_queue_handoff(minted) is True


@pytest.mark.asyncio
class TestRow10SilentAllInfoPromoteIsRefused:
    """The server half of row #10.  Which rung the watcher picks and the wording
    of its digest are an LLM judgement in skills/escalation-watcher-auto/SKILL.md,
    filed as finding esc-5227-1; this row runs the exact server calls the rung
    prescribes."""

    async def test_the_refusal_writes_nothing_and_the_rung_closes_the_members_at_l1(
        self, queue, server,
    ):
        members = [_member(queue), _member(queue)]

        result = await call_promote(server, **_promote_args(members))

        assert result == {
            'status': 'refused',
            'code': 'info_only_cluster',
            'members': [m.id for m in members],
            'unusable': [],
            'derived_severity': 'info',
        }
        assert _level_two_records(queue) == []

        for member in members:
            await call_resolve_issue(
                server, escalation_id=member.id, resolution=DIGEST, action='close_only',
                resolved_by=WATCHER, resolution_class='benign',
            )

        for member in members:
            closed = queue.get(member.id)
            assert closed is not None
            assert (closed.status, closed.level, closed.resolution_class) == ('dismissed', 1, 'benign')
        assert _level_two_records(queue) == []
        assert await call_tool(server, 'get_pending_escalations', level=2) == []


@pytest.mark.asyncio
class TestRow18ParkedHoldSurvivesARefusedPromote:
    async def test_the_parked_record_is_untouched_and_the_refused_members_stay_at_l1(
        self, queue, server,
    ):
        blocker = _member(queue, severity='blocking')
        parked = await call_resolve_issue(
            server, escalation_id=blocker.id, resolution='parked for a human ruling',
            action='park', resolved_by='interactive',
        )
        assert 'error' not in parked, parked
        held = queue.get(blocker.id)
        assert held is not None
        assert (held.status, held.level, held.resolution_action) == ('pending', 2, 'park')
        held_bytes = _file_bytes(queue, blocker.id)
        members = [_member(queue), _member(queue)]

        result = await call_promote(server, **_promote_args(members))

        assert (result['status'], result['code']) == ('refused', 'info_only_cluster')
        assert _file_bytes(queue, blocker.id) == held_bytes
        for member in members:
            still = queue.get(member.id)
            assert still is not None
            assert (still.status, still.level) == ('pending', 1)
