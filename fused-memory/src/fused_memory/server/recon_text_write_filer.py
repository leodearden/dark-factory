"""Adapter: ``TextWriteUnconfirmedFinding`` -> ``ReconReportState.add_finding`` (task 4650).

Wired at bootstrap into ``TaskInterceptor.set_text_write_unconfirmed_filer``.
With no active Stage-2 run for the finding's project it drops the finding with
a warning.
Exceptions are not swallowed here:
``middleware/task_text_write_guard.py::verify_text_write_landed`` owns the
never-break-the-write contract.
"""

from __future__ import annotations

import logging

from fused_memory.middleware.task_text_write_guard import (
    TASK_TEXT_FP_KEY,
    TEXT_VERIFIED_KEY,
    FileTextFindingFn,
    TextFieldVerification,
    TextWriteUnconfirmedFinding,
)
from fused_memory.server.recon_report import ReconReportState

logger = logging.getLogger(__name__)

_STAGE2_STAGE = 'task_knowledge_sync'

_SUGGESTED_ACTION = (
    'Do not count this amendment as applied. Re-read the task with get_task, '
    'recompose the amendment against the text it returns, and resend it with '
    f'expected_text_fp set to that read\'s {TASK_TEXT_FP_KEY}; count it only when '
    f'the response carries {TEXT_VERIFIED_KEY}: true.'
)


def _why_unconfirmed(field: TextFieldVerification) -> str:
    if field.readback_error:
        return f'{field.field} (the read failed, so whether the write landed is unknown)'
    if field.appended:
        return f'{field.field} (append: the stored text does not end with the written text)'
    return f'{field.field} (replace: the stored text does not equal the written text)'


def _describe(finding: TextWriteUnconfirmedFinding) -> str:
    return (
        f'Recon-stage {finding.op!r} on task {finding.task_id!r} reported success, '
        'but a fresh read under the write lock does not confirm the written text '
        f'landed in: {"; ".join(_why_unconfirmed(field) for field in finding.unconfirmed)}.'
    )


def build_text_write_unconfirmed_filer(
    recon_report_state: ReconReportState,
) -> FileTextFindingFn:
    async def _file(finding: TextWriteUnconfirmedFinding) -> None:
        run_id = recon_report_state.active_run_for_stage(_STAGE2_STAGE, finding.project_id)
        if run_id is None:
            logger.warning(
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
