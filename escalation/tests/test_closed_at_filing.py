"""``closed_at_filing`` against the records the real filing paths persist.

Three writers file a record already resolved, through
``EscalationQueue.submit_resolved``: a note to file, the server's eval-lane
containment, and the server's terminal-task auto-close.  Nobody ruled on any
of them, so ``closed_at_filing`` must recognise each one as persisted, and
must not recognise a record a human resolved after it was filed pending.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _filing_tools import call_blocker, call_info
from fastmcp import FastMCP

from escalation.models import (
    EVAL_LANE_CONTAINMENT_RESOLVER,
    PRE_SUBMIT_CHECK_RESOLVER,
    Escalation,
    closed_at_filing,
)
from escalation.queue import EscalationQueue
from escalation.server import create_server

LIVE_TASK_ID = '9952'
EVAL_FIXTURE_ID = 'df_task_2430_adv_plan'

_BLOCKER: dict[str, Any] = {
    'agent_role': 'implementer',
    'category': 'design_concern',
    'summary': 'step-2 conflicts with the frozen wrap decision',
}

_NOTE: dict[str, Any] = {
    'agent_role': 'implementer',
    'category': 'design_concern',
    'summary': 'deviated from step-1: kept the helper private',
    'severity': 'info',
    'note_to_file': True,
}


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'esc')


def _server(queue: EscalationQueue, *, status: str = 'in-progress') -> FastMCP:
    return create_server(
        queue,
        startup_sweep=False,
        task_status_lookup=AsyncMock(return_value=status),
        task_claimant_lookup=AsyncMock(return_value='run-live'),
    )


def _persisted(queue: EscalationQueue, result: dict[str, Any]) -> Escalation:
    record = queue.get(result['id'])
    assert record is not None, result
    return record


class TestRecordsFiledResolvedAreClosedAtFiling:
    @pytest.mark.asyncio
    async def test_a_note_to_file(self, queue: EscalationQueue) -> None:
        result = await call_info(_server(queue), task_id=LIVE_TASK_ID, **_NOTE)

        record = _persisted(queue, result)
        assert record.status == 'resolved'
        assert closed_at_filing(record)

    @pytest.mark.asyncio
    async def test_an_eval_lane_containment(self, queue: EscalationQueue) -> None:
        result = await call_blocker(_server(queue), task_id=EVAL_FIXTURE_ID, **_BLOCKER)

        record = _persisted(queue, result)
        assert record.resolved_by == EVAL_LANE_CONTAINMENT_RESOLVER
        assert closed_at_filing(record)

    @pytest.mark.asyncio
    async def test_a_terminal_task_pre_submit_auto_close(self, queue: EscalationQueue) -> None:
        result = await call_blocker(
            _server(queue, status='done'), task_id=LIVE_TASK_ID, **_BLOCKER
        )

        record = _persisted(queue, result)
        assert record.resolved_by == PRE_SUBMIT_CHECK_RESOLVER
        assert closed_at_filing(record)

    @pytest.mark.asyncio
    async def test_an_eval_lane_note_to_file(self, queue: EscalationQueue) -> None:
        result = await call_info(_server(queue), task_id=EVAL_FIXTURE_ID, **_NOTE)

        record = _persisted(queue, result)
        assert record.status == 'resolved'
        assert closed_at_filing(record)


class TestRecordsFiledPendingAreNotClosedAtFiling:
    @pytest.mark.asyncio
    async def test_a_blocker_a_human_resolved(self, queue: EscalationQueue) -> None:
        result = await call_blocker(_server(queue), task_id=LIVE_TASK_ID, **_BLOCKER)
        assert queue.resolve(result['id'], 'KEEP the wrap', resolved_by='interactive')

        record = _persisted(queue, result)
        assert record.status == 'resolved'
        assert not closed_at_filing(record)

    @pytest.mark.asyncio
    async def test_a_pending_blocker(self, queue: EscalationQueue) -> None:
        result = await call_blocker(_server(queue), task_id=LIVE_TASK_ID, **_BLOCKER)

        record = _persisted(queue, result)
        assert record.status == 'pending'
        assert not closed_at_filing(record)
