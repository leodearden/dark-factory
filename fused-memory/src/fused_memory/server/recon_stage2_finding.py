"""File a code-detected recon-stage finding on the project's active Stage-2 run,
or drop it with a warning when none is active.

Exceptions from ``ReconReportState.add_finding`` propagate; the calling guard
owns the never-break-the-write containment.
"""

from __future__ import annotations

import logging

from fused_memory.models.reconciliation import StageId
from fused_memory.server.recon_report import ReconReportState

logger = logging.getLogger(__name__)


def file_stage2_finding(
    recon_report_state: ReconReportState,
    *,
    project_id: str,
    task_id: str,
    flag_type: str,
    description: str,
    suggested_action: str,
) -> None:
    run_id = recon_report_state.active_run_for_stage(StageId.task_knowledge_sync, project_id)
    if run_id is None:
        logger.warning(
            'recon Stage-2 finding dropped: no active %s run for project_id=%r; '
            'flag_type=%s task_id=%r',
            StageId.task_knowledge_sync.value,
            project_id,
            flag_type,
            task_id,
        )
        return
    recon_report_state.add_finding(
        run_id=run_id,
        severity='serious',
        category='task_memory_mismatch',
        description=description,
        suggested_action=suggested_action,
        actionable=True,
        task_id=task_id,
        flag_type=flag_type,
    )
