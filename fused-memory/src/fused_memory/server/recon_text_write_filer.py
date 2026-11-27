"""Adapter: ``TextWriteUnconfirmedFinding`` -> ``ReconReportState.add_finding`` (task 4650).

Wired at bootstrap into ``TaskInterceptor.set_text_write_unconfirmed_filer``.
The drop policy for a finding with no active Stage-2 run lives in
``server/recon_stage2_finding.py::file_stage2_finding``.
Exceptions are not swallowed here:
``middleware/task_text_write_guard.py::verify_text_write_landed`` owns the
never-break-the-write contract.
"""

from __future__ import annotations

from fused_memory.middleware.task_text_write_guard import (
    TASK_TEXT_FP_KEY,
    TEXT_VERIFIED_KEY,
    FileTextFindingFn,
    TextFieldVerification,
    TextWriteUnconfirmedFinding,
)
from fused_memory.server.recon_report import ReconReportState
from fused_memory.server.recon_stage2_finding import file_stage2_finding

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
        file_stage2_finding(
            recon_report_state,
            project_id=finding.project_id,
            task_id=finding.task_id,
            flag_type=finding.flag_type,
            description=_describe(finding),
            suggested_action=_SUGGESTED_ACTION,
        )

    return _file
