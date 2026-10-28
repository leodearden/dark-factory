"""IntegrityCheck.run() appends the cross-run dangling-citation findings (task 5749).

Driven through the REAL ``BaseStage`` assembly path: ``stages.base.run_stage_via_cli``
is patched, not ``BaseStage.run``, the same way
``test_task_knowledge_sync_citation_verify.py`` does it. The stage's journal is a
REAL ``ReconciliationJournal``, because the scan reads settled runs back out of
it. Helpers are copied rather than imported across test modules (house norm).
"""

from __future__ import annotations

import sqlite3
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import (
    ReconciliationRun,
    RunStatus,
    RunType,
    StageId,
    StageReport,
    Watermark,
)
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.citation_verifier import STAGE_STAT_PREFIX
from fused_memory.reconciliation.cli_stage_runner import StageResult
from fused_memory.reconciliation.dangling_citation import (
    DanglingCitation,
    dangling_citation_from_finding,
)
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.stages.task_knowledge_sync import IntegrityCheck

PROJECT = 'test_project'
PRIOR_RUN = '06a4466d-cdc0-49ac-8e99-e6723be39392'
CURRENT_RUN = '9b8a7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d'
PRIOR_FINDING = 'a1a1a1a1-0000-4000-8000-000000000001'
CURRENT_FINDING = 'a1a1a1a1-0000-4000-8000-000000000002'
ABSENT = 'beacf7fc-b76a-4c0b-876d-f4cf6d906d42'
ABSENT_IN_CURRENT_RUN = 'e1c2d3b4-a5f6-4e7d-8c9b-0a1b2c3d4e5f'
LIVE = '746b4ab9-ca3c-418b-982a-32b85bfcf94b'

PREFIX = STAGE_STAT_PREFIX[StageId.integrity_check]

LLM_FINDING: dict[str, Any] = {
    'description': 'a stage-3 finding of its own',
    'severity': 'minor',
    'category': 'other',
    'actionable': False,
    'task_id': '99',
}

_STAGE3_FINDING_FILTERS = (
    'filter_blocked_snapshot_findings',
    'filter_contamination_ceiling_findings',
)


def _deps(journal: Any) -> dict:
    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[])
    memory_service.count_memories_by_metadata = AsyncMock(return_value=0)

    async def _get(project_id: str, memory_id: str):
        if memory_id in {ABSENT, ABSENT_IN_CURRENT_RUN}:
            return None
        return {'id': memory_id, 'content': 'x'}

    memory_service.get_memory_by_id = AsyncMock(side_effect=_get)
    taskmaster = AsyncMock()
    taskmaster.get_tasks = AsyncMock(return_value={'tasks': []})
    return {
        'memory_service': memory_service,
        'taskmaster': taskmaster,
        'journal': journal,
        'config': ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test'),
        'scope': ProjectScope(ProjectId(PROJECT), ProjectRoot('/tmp/test')),
    }


def _stage_result(flagged: list[dict[str, Any]]) -> StageResult:
    return StageResult(
        report={'summary': '', 'stats': {}, 'flagged_items': flagged},
        llm_calls=1,
        tokens_used=10,
        cost_usd=0.0,
        model='test-model',
        success=True,
        error='',
    )


def _citing(finding_id: str, *memory_ids: str) -> dict[str, Any]:
    return {
        'finding_id': finding_id,
        'description': f'finding {finding_id}',
        'severity': 'minor',
        'cited_memories': [{'memory_id': m, 'store': 'mem0'} for m in memory_ids],
    }


async def _seed_run(
    journal: ReconciliationJournal,
    run_id: str,
    *,
    started_at: datetime,
    status: str,
    findings: list[dict[str, Any]],
) -> None:
    await journal.start_run(
        ReconciliationRun(
            id=run_id,
            project_id=PROJECT,
            run_type=RunType.full,
            trigger_reason='test',
            started_at=started_at,
            completed_at=None,
            status=RunStatus.running,
        )
    )
    await journal.update_run_stage_reports(
        run_id,
        {
            'memory_consolidator': StageReport(
                stage=StageId.memory_consolidator,
                started_at=started_at,
                completed_at=started_at,
                items_flagged=findings,
            )
        },
    )
    if status != 'running':
        await journal.complete_run(run_id, status)


@pytest_asyncio.fixture
async def journal(tmp_path: Any):
    journal = ReconciliationJournal(tmp_path)
    await journal.initialize()
    now = datetime.now(UTC)
    await _seed_run(
        journal, PRIOR_RUN, started_at=now - timedelta(hours=1), status='completed',
        findings=[_citing(PRIOR_FINDING, ABSENT, LIVE)],
    )
    await _seed_run(
        journal, CURRENT_RUN, started_at=now, status='running',
        findings=[_citing(CURRENT_FINDING, ABSENT_IN_CURRENT_RUN)],
    )
    yield journal
    await journal.close()


async def _run_stage3(deps: dict, flagged: list[dict[str, Any]]) -> StageReport:
    stage = IntegrityCheck(StageId.integrity_check, **deps, recon_report_state=None)
    stack = [
        patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(return_value=_stage_result(flagged)),
        ),
        patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')),
        patch(
            'fused_memory.reconciliation.stages.task_knowledge_sync'
            '.filter_false_phantom_task_creation_flags',
            new=AsyncMock(side_effect=lambda *, taskmaster, known_projects, flags: flags),
        ),
    ]
    stack += [
        patch(
            f'fused_memory.reconciliation.stages.task_knowledge_sync.{name}',
            side_effect=lambda items, **_kw: items,
        )
        for name in _STAGE3_FINDING_FILTERS
    ]
    with ExitStack() as patches:
        for manager in stack:
            patches.enter_context(manager)
        return await stage.run([], Watermark(project_id=PROJECT), [], run_id=CURRENT_RUN)


def _dangling_records(report: StageReport) -> list[DanglingCitation]:
    return [
        record for record in map(dangling_citation_from_finding, report.items_flagged)
        if record is not None
    ]


@pytest.mark.asyncio
async def test_scan_appends_one_finding_beside_the_llms_own(journal: Any) -> None:
    report = await _run_stage3(_deps(journal), [dict(LLM_FINDING)])

    assert LLM_FINDING in report.items_flagged
    assert _dangling_records(report) == [
        DanglingCitation(PRIOR_RUN, PRIOR_FINDING, ABSENT, 'mem0'),
    ]


@pytest.mark.asyncio
async def test_scan_stats_land_under_the_stage3_prefix(journal: Any) -> None:
    report = await _run_stage3(_deps(journal), [dict(LLM_FINDING)])

    assert report.stats[f'{PREFIX}_cross_run_dangling_citations'] == 1
    assert report.stats[f'{PREFIX}_cross_run_runs_scanned'] == 1
    assert report.stats[f'{PREFIX}_cross_run_citation_scan_failed'] == 0


@pytest.mark.asyncio
async def test_the_current_running_run_is_not_scanned(journal: Any) -> None:
    report = await _run_stage3(_deps(journal), [dict(LLM_FINDING)])

    named = {(r.target_run_id, r.memory_id) for r in _dangling_records(report)}
    assert all(run_id != CURRENT_RUN for run_id, _ in named)
    assert all(memory_id != ABSENT_IN_CURRENT_RUN for _, memory_id in named)


class _RecentRunsRaise:
    """Delegates to a real journal except that the recent-runs read raises."""

    def __init__(self, inner: ReconciliationJournal) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def get_recent_runs(self, project_id: str, limit: int = 10) -> Any:
        raise sqlite3.OperationalError('database is locked')


@pytest.mark.asyncio
async def test_a_failed_journal_read_degrades_to_scan_failed(journal: Any) -> None:
    report = await _run_stage3(_deps(_RecentRunsRaise(journal)), [dict(LLM_FINDING)])

    assert report.items_flagged == [LLM_FINDING]
    assert _dangling_records(report) == []
    assert report.stats[f'{PREFIX}_cross_run_citation_scan_failed'] == 1
