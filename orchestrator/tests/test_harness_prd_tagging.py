"""Tests for Harness._tag_prd_metadata()."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness


@pytest.fixture
def config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=tmp_path)


@pytest.fixture
def harness(config: OrchestratorConfig) -> Harness:
    return Harness(config)


def _task(tid: str, *, prd: str | None = None) -> dict:
    metadata: dict = {}
    if prd is not None:
        metadata['prd'] = prd
    return {
        'id': tid,
        'title': f'Task {tid}',
        'status': 'pending',
        'metadata': metadata,
        'dependencies': [],
    }


@pytest.mark.asyncio
async def test_tags_new_tasks_with_prd(harness, tmp_path):
    """New tasks get PRD metadata; already-tagged tasks are skipped."""
    prd = tmp_path / 'feature.md'
    prd.touch()
    pre_ids = {'1'}  # task 1 existed before parse

    tasks = [
        _task('1'),
        _task('2'),  # new — should be tagged
        _task('3', prd='/old/prd.md'),  # already tagged — skip
    ]
    harness.scheduler.get_tasks = AsyncMock(return_value=tasks)
    harness.scheduler.update_task = AsyncMock()

    await harness._tag_prd_metadata(prd, pre_ids)

    # Only task 2 should be tagged (task 1 pre-existed, task 3 already tagged)
    assert harness.scheduler.update_task.call_count == 1
    call = harness.scheduler.update_task.call_args
    assert call.args[0] == '2'
    assert call.args[1] == {'prd': str(prd.resolve())}


@pytest.mark.asyncio
async def test_skips_tasks_not_in_new_ids(harness, tmp_path):
    """Pre-existing tasks are not re-tagged when new tasks exist."""
    prd = tmp_path / 'feature.md'
    prd.touch()
    pre_ids = {'1', '2'}

    tasks = [
        _task('1'),
        _task('2'),
        _task('3'),  # new
    ]
    harness.scheduler.get_tasks = AsyncMock(return_value=tasks)
    harness.scheduler.update_task = AsyncMock()

    await harness._tag_prd_metadata(prd, pre_ids)

    assert harness.scheduler.update_task.call_count == 1
    assert harness.scheduler.update_task.call_args.args[0] == '3'


@pytest.mark.asyncio
async def test_fallback_tags_all_untagged_when_no_new_ids(harness, tmp_path):
    """When no new task IDs detected (tree replaced), all untagged tasks are tagged."""
    prd = tmp_path / 'feature.md'
    prd.touch()
    pre_ids = {'1', '2', '3'}  # same IDs before and after

    tasks = [
        _task('1'),
        _task('2'),
        _task('3', prd='/already.md'),
    ]
    harness.scheduler.get_tasks = AsyncMock(return_value=tasks)
    harness.scheduler.update_task = AsyncMock()

    await harness._tag_prd_metadata(prd, pre_ids)

    # new_ids is empty, so all untagged (1, 2) get tagged
    assert harness.scheduler.update_task.call_count == 2
    tagged_ids = {c.args[0] for c in harness.scheduler.update_task.call_args_list}
    assert tagged_ids == {'1', '2'}


@pytest.mark.asyncio
async def test_skips_already_tagged_tasks(harness, tmp_path):
    """Never overwrites existing prd field."""
    prd = tmp_path / 'feature.md'
    prd.touch()
    pre_ids = set()  # all tasks are "new"

    tasks = [
        _task('1', prd='/other.md'),
        _task('2', prd='/another.md'),
    ]
    harness.scheduler.get_tasks = AsyncMock(return_value=tasks)
    harness.scheduler.update_task = AsyncMock()

    await harness._tag_prd_metadata(prd, pre_ids)

    harness.scheduler.update_task.assert_not_called()
