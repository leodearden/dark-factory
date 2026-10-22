"""``HistoryLegRunner`` runs only a project's declared history leg (task 7248).

The leg is a subset of the merge gate, so its result must not carry per-test
ids that a reader would take for the whole gate's.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from _merge_lane_fakes import lane_scene_config

from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.merge_lane.history_leg import HISTORY_LEG_PREFIX, HistoryLegRunner
from orchestrator.verify import run_verification
from orchestrator.verify_runner import VerifyRunner, build_merge_verify_spec

_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', worktree_dir='.worktrees',
    push_after_advance=False,
)
_ORCHESTRATOR_PROJECT = Path(__file__).resolve().parents[1]
_TASK_ID = 't-7248'


def _config(worktree: Path) -> OrchestratorConfig:
    return lane_scene_config(worktree, _GIT, merge_verify_breadth='full')


def _passing_leg(tmp_path: Path) -> tuple[Path, str]:
    """A worktree holding one passing test, and a structured pytest command over it."""
    worktree = tmp_path / 'leg'
    worktree.mkdir()
    test_file = worktree / 'test_reads_history.py'
    test_file.write_text('def test_reads_history():\n    assert True\n')
    command = (
        f'uv run --project {_ORCHESTRATOR_PROJECT} pytest {test_file} -q -p no:cacheprovider'
    )
    return worktree, command


@pytest.mark.asyncio
async def test_a_passing_leg_passes_without_the_per_test_ids_a_bare_run_collects(
    tmp_path: Path,
) -> None:
    worktree, command = _passing_leg(tmp_path)
    config = _config(worktree)
    archive_root = tmp_path / 'verify-logs'
    bare = await run_verification(
        worktree, config, ModuleConfig(prefix=HISTORY_LEG_PREFIX, test_command=command),
        max_retries=0, is_merge_verify=True, role='merge',
        task_id=_TASK_ID, archive_root=archive_root,
    )
    assert bare.passed, bare.summary
    assert None not in (
        bare.failing_test_ids, bare.skipped_test_ids, bare.failing_test_ids_by_module,
    ), bare

    runner = HistoryLegRunner(
        worktree, config, command, task_id=_TASK_ID, archive_root=archive_root,
    )
    result = await runner.run_merge_verify('a' * 40, build_merge_verify_spec(config, [], None))

    assert result.passed, result.summary
    assert result.summary.startswith('history leg'), result.summary
    assert (
        result.failing_test_ids, result.skipped_test_ids, result.failing_test_ids_by_module,
    ) == (None, None, None)


@pytest.mark.asyncio
async def test_a_history_leg_runner_is_a_healthy_local_verify_runner(tmp_path: Path) -> None:
    runner = HistoryLegRunner(
        tmp_path, _config(tmp_path), 'true',
        task_id=_TASK_ID, archive_root=tmp_path / 'verify-logs',
    )

    assert isinstance(runner, VerifyRunner)
    assert runner.is_local is True
    assert await runner.health() is True
