"""A shared driver for ``ReconciliationHarness._run_remediation_pass``.

Builds a harness over a real ``EscalationQueue``, seeds the integrity-finding
persistence the live-workflow gate needs, and stubs every pipeline stage so the
pass sees exactly the findings a test hands it. Suites about different
properties of that pass (loop responsiveness, liveness-probe failure) import
from here rather than from each other.

Import as ``from reconciliation.remediation_pass_fixtures import ...``; the
package-import mechanics are those described in ``consolidator_fixtures.py``'s
docstring.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

from escalation.queue import EscalationQueue
from shared.scheduler_state import SchedulerStateSnapshot

import fused_memory.reconciliation.harness as harness_module
from fused_memory.models.reconciliation import (
    ReconciliationRun,
    RunStatus,
    RunType,
    StageReport,
)
from fused_memory.reconciliation.harness import (
    _INTEGRITY_FINDING_RECURRENCE_THRESHOLD,
    TierConfig,
)
from fused_memory.reconciliation.stages.task_knowledge_sync import IntegrityCheck
from fused_memory.reconciliation.task_filter import FilteredTaskTree
from reconciliation.consolidator_fixtures import make_scope
from reconciliation.harness_fixtures import (
    make_harness,
    make_harness_config,
    stub_stage_runs,
)

PROJECT_ID = 'test-project'
PROJECT_ROOT = '/tmp/test-project'


def make_remediation_harness(
    journal, event_buffer, memory_service, *, project_root: str = PROJECT_ROOT,
    **harness_kwargs,
):
    """A harness that knows ``PROJECT_ID`` as *project_root*."""
    return make_harness(
        journal=journal,
        event_buffer=event_buffer,
        memory_service=memory_service,
        taskmaster=AsyncMock(),
        config=make_harness_config(judge_enabled=False),
        known_projects={PROJECT_ID: project_root},
        **harness_kwargs,
    )


def finding_citing(
    *task_ids: str, description: str | None = None, routed_to: str | None = None,
) -> dict:
    """An actionable stranded-work finding citing each of *task_ids*.

    *routed_to* sets the bare ``task_id`` field, which
    ``resolve_finding_task_target`` prefers over ``cited_tasks`` — the way to
    build a finding whose ROUTED target is not one of its cited ids, and so the
    only way to reach the routed-target liveness gate.
    """
    cited = ','.join(task_ids)
    finding = {
        'description': description or f'Implementation-complete but not merged: {cited}',
        'severity': 'urgent',
        'actionable': True,
        'category': 'stranded_work',
        'cited_tasks': [
            {'project_id': PROJECT_ID, 'task_id': tid} for tid in task_ids
        ],
        'suggested_action': 'Manual merge required',
    }
    if routed_to is not None:
        finding['task_id'] = routed_to
        finding['description'] = description or (
            f'Implementation-complete but not merged: {cited} (routed to {routed_to})'
        )
    return finding


def in_progress_task(task_id: str) -> dict:
    return {
        'id': int(task_id),
        'title': f'In-progress task {task_id}',
        'status': 'in-progress',
        'claimant_run_id': f'run-{task_id}',
        'heartbeat_at': datetime.now(UTC).isoformat(),
        'metadata': {'task_kind': 'normal'},
        'dependencies': [],
    }


async def seed_persistence(journal, findings: list[dict]) -> None:
    """Seed threshold-1 prior completed runs carrying *findings*.

    Together with this pass's own persisted S3 report that reaches
    ``_INTEGRITY_FINDING_RECURRENCE_THRESHOLD`` so the live-workflow gate fires.
    Uses the journal's PUBLIC async API (start_run / update_run_stage_reports /
    complete_run), never its storage internals.
    """
    n_seed = _INTEGRITY_FINDING_RECURRENCE_THRESHOLD - 1
    base_time = datetime.now(UTC) - timedelta(minutes=n_seed + 1)
    for i in range(n_seed):
        rid = str(uuid.uuid4())
        await journal.start_run(
            ReconciliationRun(
                id=rid,
                project_id=PROJECT_ID,
                run_type=RunType.full,
                trigger_reason='buffer_size:1',
                started_at=base_time + timedelta(minutes=i),
                events_processed=1,
                status=RunStatus.running,
            )
        )
        await journal.update_run_stage_reports(
            rid, {'integrity_check': {'items_flagged': findings}}
        )
        await journal.complete_run(rid, 'completed')


async def prepare_remediation_pass(
    *, journal, event_buffer, memory_service, tmp_path, monkeypatch,
    findings: list[dict], cited_tasks: list[dict] | None = None,
    project_root: str = PROJECT_ROOT,
):
    """Wire a harness to drive ``_run_remediation_pass`` over *findings*.

    The pass runs for ``PROJECT_ID`` rooted at *project_root*, which is where
    the routed-finding filer writes its orchestrator queue.

    Returns ``(harness, esc_queue, run_pass)`` where ``run_pass()`` returns a
    fresh coroutine for the pass, so a caller can await it or hand it to a
    measuring wrapper.
    """
    esc_queue = EscalationQueue(tmp_path / 'esc')
    harness = make_remediation_harness(
        journal, event_buffer, memory_service,
        project_root=project_root, escalation_queue=esc_queue,
    )

    # Pin the two on-disk corroboration inputs to their empty forms so neither
    # of the other corroboration signals can fire against whatever lives at
    # the project root (by default the shared, pytest-unmanaged PROJECT_ROOT).
    monkeypatch.setattr(
        harness_module, 'read_scheduler_state',
        lambda _root: SchedulerStateSnapshot.empty(),
    )
    monkeypatch.setattr(harness_module, 'orchestrator_started_at', lambda _root: None)

    await seed_persistence(journal, findings)

    async def fake_run(
        self, events, watermark, prior_reports, run_id, model=None, resume_session_id=None,
    ):
        now = datetime.now(UTC)
        items_flagged = findings if isinstance(self, IntegrityCheck) else []
        return StageReport(
            stage=self.stage_id, started_at=now, completed_at=now, items_flagged=items_flagged,
        )

    stub_stage_runs(monkeypatch, fake_run)

    tree = FilteredTaskTree(
        active_tasks=list(cited_tasks or []), total_count=len(cited_tasks or []),
    )
    tier = TierConfig(model='sonnet', episode_limit=100, memory_limit=200)

    def run_pass():
        return harness._run_remediation_pass(
            PROJECT_ID, 'parent-run-id', findings, tier,
            scope=make_scope(PROJECT_ID, project_root),
            filtered_task_tree=tree,
            filtered_task_tree_fetched_at=datetime.now(UTC),
        )

    return harness, esc_queue, run_pass
