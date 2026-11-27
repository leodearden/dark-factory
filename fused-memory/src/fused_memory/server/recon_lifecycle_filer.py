"""Adapter: ``LifecycleResetFinding`` -> ``ReconReportState.add_finding`` (task 2624).

Maps a :class:`fused_memory.middleware.live_task_write_guard.LifecycleResetFinding`
onto the recon_report channel — the sanctioned programmatic API for
task-lifecycle findings (see ``reconciliation/prompts/stage2.py``). Built
once at server bootstrap, right after ``ReconReportState`` is constructed
(see ``server/main.py``), and wired into
``TaskInterceptor.set_lifecycle_reset_filer``.

The drop policy for a finding with no active Stage-2 run lives in
``server/recon_stage2_finding.py::file_stage2_finding``.
"""

from __future__ import annotations

from fused_memory.middleware.live_task_write_guard import (
    FileFindingFn,
    LifecycleResetFinding,
)
from fused_memory.server.recon_report import ReconReportState
from fused_memory.server.recon_stage2_finding import file_stage2_finding

_SUGGESTED_ACTION = (
    'Investigate a concurrent write racing this recon-stage write on the '
    'same task — a second live claimant, a stranded-task sweep un-claiming '
    'it, or a manual status edit — around the time of this write.'
)


def _describe(finding: LifecycleResetFinding) -> str:
    return (
        f'Unexpected lifecycle divergence on task {finding.task_id!r} during '
        f'recon-stage {finding.op!r} write: before={finding.before!r} '
        f'after={finding.after!r}; diverged fields: '
        f'{", ".join(finding.diverged_fields)}.'
    )


def build_lifecycle_reset_filer(recon_report_state: ReconReportState) -> FileFindingFn:
    """Build the async ``file_finding`` callback for ``set_lifecycle_reset_filer``."""

    async def _file(finding: LifecycleResetFinding) -> None:
        file_stage2_finding(
            recon_report_state,
            project_id=finding.project_id,
            task_id=finding.task_id,
            flag_type=finding.flag_type,
            description=_describe(finding),
            suggested_action=_SUGGESTED_ACTION,
        )

    return _file
