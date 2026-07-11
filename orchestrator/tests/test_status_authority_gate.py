"""ζ integration gate — B1/B2/B3-harness/B4/B6 + comp-2 harness face.

PRD ``plans/task-status-authority-prd.md`` §Boundary-test sketch. This
module realizes the escalation<->harness HARNESS-side rows of the 17-cell
boundary matrix: B1 (memberless born-at-L2 resume), B2 (esc-2073-15
stranded-in-progress revert), B3-harness (park's harness face, two-way with
the escalation suite's B3-server), B4 (abandon/close_only), B6 (infra-hold
first-class: resume-at-verify + reconcile-skip), and the harness face of
comp-2 (an unrecognised resolution_action -> ``effect_for`` returns None ->
no ``set_task_status`` call — the SAME table the escalation server rejects
on, proven from the other side of the seam).

Every drive goes through ``harness._on_escalation_resolved(esc)`` (a SYNC
call that schedules background coros into ``harness._background_tasks``)
followed by draining those coros, then asserts the scheduler's
``set_task_status`` target — the product's own read path on the harness
side (mirrors test_harness_action_dispatch.py / test_harness_infra_hold_repend.py).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from escalation.action_effects import effect_for
from escalation.models import Escalation
from orchestrator.harness import Harness
from orchestrator.task_status import is_infra_held

__all__ = ['effect_for', 'is_infra_held']

_BASE_SHA = 'a' * 40  # branch_base_sha in metadata — non-degenerate-branch fixtures
_TIP_SHA = 'b' * 40   # tip SHA different from base -> non-degenerate


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config) -> Harness:
    """Harness with mocked internals for status-authority-gate dispatch tests.

    Mirrors test_harness_action_dispatch.py:32-55 — McpLifecycle/
    OverrideStore/Scheduler/BriefingAssembler patched at construction, then
    the scheduler replaced with a MagicMock carrying AsyncMock
    get_status/set_task_status/get_task/update_task so
    ``_on_escalation_resolved``'s background coros can be driven and
    asserted through the scheduler's own call surface (the harness's
    "product read path" for a task-status write).
    """
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler.get_status = AsyncMock(return_value='blocked')
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.get_task = AsyncMock(return_value={'id': 'task', 'metadata': {}})
    h.scheduler.update_task = AsyncMock(return_value=True)

    # _merge_worker stays None — unhalt branch skipped in all tests here.
    return h


def _make_esc(
    task_id: str = 'task-1',
    status: str = 'resolved',
    resolved_by: str = 'steward',
    level: int = 1,
    resolution_action: str | None = None,
) -> Escalation:
    """Mirrors test_harness_action_dispatch.py:58-76 — a generic resolved
    escalation for the B1/B3-harness/B4/comp-2 dispatch cells."""
    return Escalation(
        id=f'esc-{task_id}-1',
        task_id=task_id,
        agent_role='workflow',
        severity='blocking',
        category='infra_issue',
        summary='status-authority-gate dispatch test',
        level=level,
        status=status,
        resolved_by=resolved_by,
        resolution_action=resolution_action,
    )


def _make_infra_esc(
    task_id: str = '1883',
    status: str = 'resolved',
    level: int = 1,
    resolved_by: str | None = None,
) -> Escalation:
    """Mirrors test_harness_infra_hold_repend.py:204-221 — a minimal resolved
    infra_issue escalation for the B6 resume-at-verify cell."""
    return Escalation(
        id=f'esc-{task_id}-99',
        task_id=task_id,
        agent_role='workflow',
        severity='blocking',
        category='infra_issue',
        summary='ENOSPC during verify warm marker write',
        level=level,
        status=status,
        resolved_by=resolved_by,
    )
