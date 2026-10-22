"""``MemoryConsolidator.run`` drives the REAL unlanded-path check (task 5626).

Only the LLM seam (``BaseStage.run``) and ``dedup_flags`` are patched, the way
``test_stage1.py::TestCuratorGateResolutionSweepWiring`` does it. The check
runs against a real merged git repo and MemoryService-shaped reader mocks, so a
drift in either the call site or the reader signature fails here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from _fm_helpers import _init_git_repo
from _merged_branch_repo import INCIDENT_BRIEFING, INCIDENT_GUARD, incident_merge

from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
    StageId,
    StageReport,
    Watermark,
)
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from reconciliation.consolidator_fixtures import make_consolidator

FLAG_TYPE = 'memory_cites_unlanded_path'
TASK_FILTER = {'task_id': '107'}
INCIDENT_MEMORIES = {
    'm1': f'Task 107 delivered {INCIDENT_GUARD} and a briefing edit.',
    'm2': f'Task 107 edited {INCIDENT_BRIEFING}.',
}
STAT_KEYS = (
    'unlanded_path_check_checked',
    'unlanded_path_check_skipped',
    'unlanded_path_check_errors',
    'unlanded_path_check_flags_emitted',
)


def _list_memories(project_id: str, filters: dict, limit: int = 1000) -> list[dict]:
    if filters != TASK_FILTER:
        return []
    return [
        {'id': memory_id, 'created_at': '2026-09-18T13:08:50+00:00', 'metadata': dict(TASK_FILTER)}
        for memory_id in INCIDENT_MEMORIES
    ]


def _read_memory(project_id: str, memory_id: str) -> dict | None:
    content = INCIDENT_MEMORIES.get(memory_id)
    return None if content is None else {'id': memory_id, 'content': content, 'metadata': {}}


@pytest.fixture
def merged_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    """The repo, and task 107's incident merge in it."""
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path))
    root = tmp_path / 'repo'
    root.mkdir()
    _init_git_repo(root)
    return root, incident_merge(root, '107').merge


def _done_event(merge_sha: str) -> ReconciliationEvent:
    return ReconciliationEvent(
        id='evt-107-done',
        type=EventType.task_status_changed,
        source=EventSource.agent,
        project_id='test_project',
        timestamp=datetime.now(UTC),
        payload={
            'task_id': '107',
            'old_status': 'in-progress',
            'new_status': 'done',
            'done_provenance': {'kind': 'merged', 'commit': merge_sha},
        },
    )


def _stage(repo: Path) -> tuple[MemoryConsolidator, AsyncMock]:
    """The stage, and the get_memories_by_metadata mock its memory service carries."""
    stage = make_consolidator(project_root=str(repo))
    list_memories = AsyncMock(side_effect=_list_memories)
    stage.memory.get_memories_by_metadata = list_memories
    stage.memory.get_memory_by_id = AsyncMock(side_effect=_read_memory)
    return stage, list_memories


def _base_report(items_flagged: list[dict] | None = None) -> StageReport:
    return StageReport(
        stage=StageId.memory_consolidator,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        items_flagged=items_flagged if items_flagged is not None else [],
        stats={},
    )


async def _run(
    stage: MemoryConsolidator,
    events: list[ReconciliationEvent],
    base_report: StageReport,
    dedup_mock: AsyncMock,
) -> StageReport:
    with (
        patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
        patch(
            'fused_memory.reconciliation.stages.memory_consolidator.dedup_flags',
            new=dedup_mock,
        ),
    ):
        return await stage.run(
            events=events,
            watermark=Watermark(project_id='test_project'),
            prior_reports=[],
            run_id='run-5626',
        )


def _identity_dedup() -> AsyncMock:
    return AsyncMock(side_effect=lambda **kw: kw['flags'])


def _task_filtered_listings(list_memories: AsyncMock) -> list:
    return [call for call in list_memories.await_args_list if call.kwargs.get('filters') == TASK_FILTER]


@pytest.mark.asyncio
async def test_full_cycle_appends_the_flag_above_dedup(merged_repo: tuple[Path, str]) -> None:
    repo, merge = merged_repo
    stage, list_memories = _stage(repo)
    llm_flag = {
        'description': 'an unrelated LLM finding',
        'severity': 'minor',
        'task_id': '999',
        'flag_type': 'stale_memory',
        'category': 'memory_stale',
    }
    dedup_mock = _identity_dedup()

    report = await _run(stage, [_done_event(merge)], _base_report([llm_flag]), dedup_mock)

    flag_types = [f.get('flag_type') for f in report.items_flagged]
    assert FLAG_TYPE in flag_types
    assert 'stale_memory' in flag_types
    assert dedup_mock.await_args is not None
    assert FLAG_TYPE in [f.get('flag_type') for f in dedup_mock.await_args.kwargs['flags']]
    assert report.stats['unlanded_path_check_checked'] == 1
    assert report.stats['unlanded_path_check_flags_emitted'] == 1
    assert report.stats['unlanded_path_check_errors'] == 0
    list_memories.assert_any_await(project_id='test_project', filters=TASK_FILTER)


@pytest.mark.asyncio
async def test_filter_chain_fires_when_the_llm_flagged_nothing(merged_repo: tuple[Path, str]) -> None:
    repo, merge = merged_repo
    stage, _ = _stage(repo)
    dedup_mock = _identity_dedup()

    report = await _run(stage, [_done_event(merge)], _base_report([]), dedup_mock)

    dedup_mock.assert_awaited_once()
    assert [f.get('flag_type') for f in report.items_flagged] == [FLAG_TYPE]


@pytest.mark.asyncio
async def test_remediation_pass_publishes_zero_stats_and_reads_nothing(merged_repo: tuple[Path, str]) -> None:
    repo, merge = merged_repo
    stage, list_memories = _stage(repo)
    stage.remediation_findings = []

    report = await _run(stage, [_done_event(merge)], _base_report([]), _identity_dedup())

    assert {key: report.stats.get(key) for key in STAT_KEYS} == dict.fromkeys(STAT_KEYS, 0)
    assert _task_filtered_listings(list_memories) == []


@pytest.mark.asyncio
async def test_full_cycle_without_events_publishes_zero_stats(merged_repo: tuple[Path, str]) -> None:
    repo, _ = merged_repo
    stage, _ = _stage(repo)

    report = await _run(stage, [], _base_report([]), _identity_dedup())

    assert {key: report.stats.get(key) for key in STAT_KEYS} == dict.fromkeys(STAT_KEYS, 0)


@pytest.mark.asyncio
async def test_flag_survives_the_done_task_filters(merged_repo: tuple[Path, str]) -> None:
    """Task 107 IS done, which the terminal-task filter acts on: a stale_metadata control flag drops, ours stays."""
    repo, merge = merged_repo
    stage, _ = _stage(repo)
    assert stage.taskmaster is not None
    stage.taskmaster.get_task = AsyncMock(return_value={'id': '107', 'status': 'done'})
    control = {
        'description': 'task 107 metadata is stale',
        'severity': 'minor',
        'task_id': '107',
        'flag_type': 'stale_metadata',
        'category': 'memory_stale',
    }

    report = await _run(stage, [_done_event(merge)], _base_report([control]), _identity_dedup())

    flag_types = [f.get('flag_type') for f in report.items_flagged]
    assert 'stale_metadata' not in flag_types
    assert flag_types == [FLAG_TYPE]
