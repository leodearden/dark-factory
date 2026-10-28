"""Behaviour of ``orchestrator.lane_task_dir`` against real ``.task/`` trees."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from shared.transcript_archive import durable_archive_path, resolve_archive_root

from orchestrator.config import TranscriptArchiveConfig
from orchestrator.lane_task_dir import (
    PreviousOccupantEviction,
    archive_root_for,
    evict_previous_occupant,
)


def _plant_transcript(config_dir: Path, session_id: str) -> Path:
    transcript = config_dir / 'projects' / '-enc' / f'{session_id}.jsonl'
    transcript.parent.mkdir(parents=True)
    transcript.write_text(f'{{"session": "{session_id}"}}\n')
    return transcript


def _plant_nested_repo(config_dir: Path) -> Path:
    marketplace = config_dir / 'plugins' / 'marketplaces' / 'm'
    marketplace.mkdir(parents=True)
    subprocess.run(['git', 'init', '-q', str(marketplace)], check=True)
    return marketplace


def _plant_husk_shaped_config_dir(task_dir: Path, task_id: str, session_id: str) -> Path:
    config_dir = task_dir / f'claude-config-{task_id}'
    _plant_transcript(config_dir, session_id)
    (config_dir / '.credentials.json').write_text('{"token": "secret"}')
    _plant_nested_repo(config_dir)
    return config_dir


@pytest.mark.asyncio
async def test_a_foreign_config_dir_with_a_nested_repo_is_archived_then_removed(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / 'task'
    archive_root = tmp_path / 'archive'
    _plant_husk_shaped_config_dir(task_dir, 'A', 'sA')

    result = await evict_previous_occupant(
        task_dir, keep_task_id=None, archive_root=archive_root,
    )

    assert not os.path.lexists(task_dir / 'claude-config-A')
    assert durable_archive_path(archive_root, 'A', 'sA') is not None
    assert result == PreviousOccupantEviction(evicted=('claude-config-A',), retained=())


@pytest.mark.asyncio
async def test_an_unblock_config_dir_archives_under_its_full_suffix(tmp_path: Path) -> None:
    task_dir = tmp_path / 'task'
    archive_root = tmp_path / 'archive'
    _plant_transcript(task_dir / 'claude-config-A-unblock', 'u')

    result = await evict_previous_occupant(
        task_dir, keep_task_id=None, archive_root=archive_root,
    )

    assert not os.path.lexists(task_dir / 'claude-config-A-unblock')
    assert durable_archive_path(archive_root, 'A-unblock', 'u') is not None
    assert result.evicted == ('claude-config-A-unblock',)


@pytest.mark.asyncio
async def test_non_config_entries_are_removed_without_following_a_symlink(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / 'task'
    task_dir.mkdir()
    durable_plan = tmp_path / 'meta' / 'plan.json'
    durable_plan.parent.mkdir()
    durable_plan.write_text('{"plan": "durable"}')
    (task_dir / 'plan.json').symlink_to(durable_plan)
    (task_dir / 'iterations.jsonl').write_text('{}\n')
    (task_dir / 'verify').mkdir()
    (task_dir / 'verify' / 'out.log').write_text('log')
    (task_dir / 'reviews').mkdir()
    (task_dir / 'reviews' / 'merge.json').write_text('{}')

    result = await evict_previous_occupant(
        task_dir, keep_task_id=None, archive_root=tmp_path / 'archive',
    )

    assert task_dir.is_dir()
    assert list(task_dir.iterdir()) == []
    assert durable_plan.read_text() == '{"plan": "durable"}'
    assert result == PreviousOccupantEviction(
        evicted=('iterations.jsonl', 'plan.json', 'reviews', 'verify'), retained=(),
    )


@pytest.mark.asyncio
async def test_the_incoming_tasks_own_config_dir_is_kept_byte_for_byte(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / 'task'
    own = task_dir / 'claude-config-B'
    own_transcript = _plant_transcript(own, 'sB')
    own_credentials = own / '.credentials.json'
    own_credentials.write_text('{"token": "B"}')
    _plant_transcript(task_dir / 'claude-config-B-unblock', 'u')
    transcript_before = own_transcript.read_bytes()

    result = await evict_previous_occupant(
        task_dir, keep_task_id='B', archive_root=tmp_path / 'archive',
    )

    assert own_transcript.read_bytes() == transcript_before
    assert own_credentials.read_text() == '{"token": "B"}'
    assert not os.path.lexists(task_dir / 'claude-config-B-unblock')
    assert 'claude-config-B' not in result.evicted
    assert 'claude-config-B' not in result.retained
    assert result.evicted == ('claude-config-B-unblock',)


@pytest.mark.asyncio
async def test_without_an_archive_root_a_foreign_config_dir_is_still_removed(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / 'task'
    _plant_husk_shaped_config_dir(task_dir, 'A', 'sA')

    result = await evict_previous_occupant(task_dir, keep_task_id=None, archive_root=None)

    assert not os.path.lexists(task_dir / 'claude-config-A')
    assert sorted(p.name for p in tmp_path.iterdir()) == ['task']
    assert result.evicted == ('claude-config-A',)


@pytest.mark.asyncio
async def test_an_absent_task_dir_is_a_no_op(tmp_path: Path) -> None:
    task_dir = tmp_path / 'task'

    result = await evict_previous_occupant(
        task_dir, keep_task_id='B', archive_root=tmp_path / 'archive',
    )

    assert result == PreviousOccupantEviction(evicted=(), retained=())
    assert not os.path.lexists(task_dir)


@pytest.mark.asyncio
async def test_an_unarchivable_transcript_is_held_and_reported_retained(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / 'task'
    config_dir = _plant_husk_shaped_config_dir(task_dir, 'A', 'sA')
    archive_root = tmp_path / 'archive-is-a-file'
    archive_root.write_text('not a directory')

    result = await evict_previous_occupant(
        task_dir, keep_task_id=None, archive_root=archive_root,
    )

    assert (config_dir / 'projects' / '-enc' / 'sA.jsonl').is_file()
    assert not os.path.lexists(config_dir / '.credentials.json')
    assert result.retained == ('claude-config-A',)
    assert result.evicted == ()


@pytest.mark.asyncio
async def test_a_symlinked_config_dir_is_unlinked_and_its_target_left_alone(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / 'task'
    task_dir.mkdir()
    outside = tmp_path / 'outside-config'
    outside_transcript = _plant_transcript(outside, 'sZ')
    (task_dir / 'claude-config-Z').symlink_to(outside, target_is_directory=True)
    archive_root = tmp_path / 'archive'

    result = await evict_previous_occupant(
        task_dir, keep_task_id=None, archive_root=archive_root,
    )

    assert not os.path.lexists(task_dir / 'claude-config-Z')
    assert outside_transcript.is_file()
    assert durable_archive_path(archive_root, 'Z', 'sZ') is None
    assert result.evicted == ('claude-config-Z',)


def test_archive_root_for_no_archive_config_is_none(tmp_path: Path) -> None:
    assert archive_root_for(None, tmp_path) is None


def test_archive_root_for_a_disabled_archive_is_none(tmp_path: Path) -> None:
    assert archive_root_for(TranscriptArchiveConfig(enabled=False), tmp_path) is None


def test_archive_root_for_an_enabled_archive_resolves_against_project_root(
    tmp_path: Path,
) -> None:
    cfg = TranscriptArchiveConfig(enabled=True, root='data/agent-transcripts')

    assert archive_root_for(cfg, tmp_path) == resolve_archive_root(tmp_path, cfg.root)
