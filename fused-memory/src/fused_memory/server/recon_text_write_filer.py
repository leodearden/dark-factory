"""Adapter: ``TextWriteUnconfirmedFinding`` -> ``ReconReportState.add_finding`` (task 4650).

Wired at bootstrap into ``TaskInterceptor.set_text_write_unconfirmed_filer``.
With no active Stage-2 run for the finding's project it is a logged no-op.
Exceptions are not swallowed here:
``middleware/task_text_write_guard.py::verify_text_write_landed`` owns the
never-break-the-write contract.
"""

from __future__ import annotations

import logging

from fused_memory.middleware.task_text_write_guard import (
    FileTextFindingFn,
    TextWriteUnconfirmedFinding,
)
from fused_memory.server.recon_report import ReconReportState

logger = logging.getLogger(__name__)

_STAGE2_STAGE = 'task_knowledge_sync'

_SUGGESTED_ACTION = (
    'Do not count this amendment as applied. Re-read the task with get_task, '
    'recompose the amendment against the text it returns, and resend it with '
    'expected_text_fp set to that read\'s task_text_fp; count it only when the '
    'response carries text_verified: true.'
)


def _describe(finding: TextWriteUnconfirmedFinding) -> str:
    return (
        f'Recon-stage {finding.op!r} on task {finding.task_id!r} reported success, '
        'but a fresh read under the write lock does not contain the written text '
        f'in: {", ".join(finding.unconfirmed_fields)}.'
    )


def build_text_write_unconfirmed_filer(
    recon_report_state: ReconReportState,
) -> FileTextFindingFn:
    async def _file(finding: TextWriteUnconfirmedFinding) -> None:
        run_id = recon_report_state.active_run_for_stage(_STAGE2_STAGE, finding.project_id)
        if run_id is None:
            logger.info(
                'recon_text_write_filer: no active %s run for project_id=%r; '
                'dropping %s finding for task_id=%r',
                _STAGE2_STAGE,
                finding.project_id,
                finding.flag_type,
                finding.task_id,
            )
            return
        recon_report_state.add_finding(
            run_id=run_id,
            severity='serious',
            category='task_memory_mismatch',
            description=_describe(finding),
            suggested_action=_SUGGESTED_ACTION,
            actionable=True,
            task_id=finding.task_id,
            flag_type=finding.flag_type,
        )

    return _file
