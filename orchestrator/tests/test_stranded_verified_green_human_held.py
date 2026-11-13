"""A human-held ``stranded_blocked`` record holds the verified-green auto-merge.

Task 6145.  The stranded sweep's blocked-arm EXISTS_OFF_MAIN upgrade submits a
verified-green lane branch straight to the merge queue, so its remediation
never meets the workflow gate.  Its veto is
``orchestrator.recovery_pins.records_pin_blocked_recovery``, which relaxes a
set made up entirely of ``stranded_blocked`` records only while none of them
needs a human (``escalation.pins.needs_human``).  A human-held one stays a
veto: merging past it is what repend PRD D4 forbids.

The harness wiring is ``test_stranded_verified_green.py``'s, whose
``TestVerifiedGreenVetoRelaxExistsOffMain`` covers the relaxed counterpart.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from test_stranded_verified_green import (
    _TID,
    _arm_escalation_queue,
    _asserted_no_repend,
    _drain_background,
    _match,
    _merge_queue_depth,
    _wire_exists_off_main,
)
from test_stranded_verified_green import _reset_merge_provenance as _reset_merge_provenance
from test_stranded_verified_green import harness as harness

from orchestrator.harness import Harness


def _seed_stranded_blocked(
    queue: EscalationQueue, tid: str, *, severity: str, level: int,
) -> str:
    esc_id = f'esc-seed-human-held-{tid}'
    queue.submit(Escalation(
        id=esc_id, task_id=tid, agent_role='steward', severity=severity,
        category='stranded_blocked', summary='human-held stranded_blocked',
        level=level,
    ))
    return esc_id


@pytest.mark.asyncio
class TestHumanHeldStrandedBlockedHoldsTheAutoMerge:

    @pytest.mark.parametrize(
        ('severity', 'level'), [('blocking', 2), ('critical', 1)],
        ids=['parked-l2', 'critical-l1'],
    )
    async def test_the_task_is_left_and_no_merge_request_is_submitted(
        self, harness: Harness, tmp_path: Path, severity: str, level: int,
    ) -> None:
        tid = _TID
        queue = _arm_escalation_queue(harness, tmp_path, 'human-held')
        _wire_exists_off_main(harness, tid)
        seeded = _seed_stranded_blocked(queue, tid, severity=severity, level=level)
        detect = AsyncMock(return_value=_match(tmp_path))

        with patch('orchestrator.harness.detect_verified_green', detect):
            await harness._reconcile_stranded_in_progress()
        await _drain_background(harness)

        detect.assert_not_awaited()
        assert _merge_queue_depth(harness) == 0
        _asserted_no_repend(harness, tid)
        assert [e.id for e in queue.get_by_task(tid, status='pending')] == [seeded]
