"""``_mark_blocked(evidence=...)`` lands the entries on every escalation it files.

The L0 the steward reads, the fall-through L1 a human reads when the steward
cannot resolve the block, and the ``escalate_to_human`` short-circuit L1 all
carry the same structured observations (task 5107). Without evidence, the
records keep ``Escalation.evidence``'s empty default.

Driven against a real ``EscalationQueue`` and a real ``OrchestratorConfig``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from escalation.models import EvidenceEntry
from test_workflow_mark_blocked_l0_coverage import TASK_ID, _make_workflow

ENTRY = EvidenceEntry(
    observation='merger kind=empty_output',
    measured_at='t0',
    ref='merger session_id=s1',
)


@pytest.mark.asyncio
async def test_evidence_lands_on_the_l0_and_the_fall_through_l1(tmp_path: Path):
    workflow, queue, _ = _make_workflow(tmp_path)

    await workflow._mark_blocked('r', evidence=[ENTRY])

    filed = queue.get_by_task(TASK_ID)
    assert {e.level for e in filed} == {0, 1}, [(e.id, e.level) for e in filed]
    assert all(e.evidence == [ENTRY] for e in filed), [(e.id, e.evidence) for e in filed]


@pytest.mark.asyncio
async def test_evidence_lands_on_the_escalate_to_human_l1(tmp_path: Path):
    workflow, queue, _ = _make_workflow(tmp_path)

    await workflow._mark_blocked('r', escalate_to_human=True, evidence=[ENTRY])

    filed = queue.get_by_task(TASK_ID, level=1)
    assert len(filed) == 1, [(e.id, e.level) for e in queue.get_by_task(TASK_ID)]
    assert filed[0].evidence == [ENTRY]


@pytest.mark.asyncio
async def test_no_evidence_keeps_the_empty_default(tmp_path: Path):
    workflow, queue, _ = _make_workflow(tmp_path)

    await workflow._mark_blocked('r')

    filed = queue.get_by_task(TASK_ID)
    assert filed
    assert all(e.evidence == [] for e in filed), [(e.id, e.evidence) for e in filed]
