"""Adapter: ``LifecycleResetFinding`` -> ``ReconReportState.add_finding`` (task 2624).

Maps a :class:`fused_memory.middleware.live_task_write_guard.LifecycleResetFinding`
onto the recon_report channel — the sanctioned programmatic API for
task-lifecycle findings (see ``reconciliation/prompts/stage2.py``). Built
once at server bootstrap, right after ``ReconReportState`` is constructed
(see ``server/main.py``), and wired into
``TaskInterceptor.set_lifecycle_reset_filer``.

Best-effort: when no Stage-2 (``task_knowledge_sync``) run is currently
active for the finding's project, there is nowhere sanctioned to land the
finding — this is a logged no-op, not a raise, matching
``guarded_recon_task_write``'s never-break-the-write contract (the
before/after write it is guarding has already succeeded by the time this
runs).
"""

from __future__ import annotations

import logging

from fused_memory.middleware.live_task_write_guard import (
    FileFindingFn,
    LifecycleResetFinding,
)
from fused_memory.server.recon_report import ReconReportState

logger = logging.getLogger(__name__)

# Stage-2's stage name in ReconReportState — the only stage this filer ever
# resolves against, since the guard it backs only wraps Stage-2 recon-stage
# task writes (agent_id.startswith('recon-stage-')).
_STAGE2_STAGE = 'task_knowledge_sync'

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
        run_id = recon_report_state.active_run_for_stage(_STAGE2_STAGE, finding.project_id)
        if run_id is None:
            logger.info(
                'recon_lifecycle_filer: no active %s run for project_id=%r; '
                'dropping task_lifecycle_reset_detected finding for task_id=%r',
                _STAGE2_STAGE,
                finding.project_id,
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
