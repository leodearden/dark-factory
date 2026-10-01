"""Behaviour of ``reconciliation/live_workflow_section.py`` against REAL git.

Every fixture is a tmp repository: live tasks get a real ``git worktree add``
and a commit inside the detector's recent-commit window, not-live tasks get
commits dated 48 h before the injected ``now``, and landings are real
``--no-ff`` merge markers or real cherry-picked twins. No
``data/orchestrator/orchestrator.lock`` exists under the tmp root, so the
project-wide orchestrator signal is genuinely off. The one module-namespace
patch below makes the landing probe RAISE, a state no real fixture produces.

Assertions on rendered text are token-level within the task's own line, so
later fields appended to the same line do not break them.
"""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _fm_helpers import _init_git_repo

from fused_memory.models.scope import ProjectRoot
from fused_memory.reconciliation import live_workflow_section as section_module
from fused_memory.reconciliation.live_workflow_section import (
    LiveWorkflowSnapshot,
    build_live_workflow_snapshot,
    render_live_workflow_section,
)
from fused_memory.services.landed_on_main import LandingEvidence, LandingVerdict

NOW = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
#: Outside the detector's recent-commit window, so a commit here is not live.
STALE = NOW - timedelta(hours=48)
#: Inside the recent-commit window.
RECENT = NOW - timedelta(hours=1)


def _git(repo: Path, *args: str, when: datetime | None = None) -> str:
    env = dict(os.environ)
    if when is not None:
        stamp = f'@{int(when.timestamp())} +0000'
        env['GIT_AUTHOR_DATE'] = stamp
        env['GIT_COMMITTER_DATE'] = stamp
    return subprocess.run(
        ['git', '-C', str(repo), *args], check=True, capture_output=True, text=True, env=env,
    ).stdout.strip()


def _branch_with_commits(repo: Path, task_id: int, count: int, *, when: datetime) -> list[str]:
    """Create ``task/<id>`` at main's tip carrying *count* own commits; return their shas."""
    _git(repo, 'checkout', '-q', '-b', f'task/{task_id}', 'main')
    shas = []
    for n in range(count):
        name = f'{task_id}-{n}.txt'
        (repo / name).write_text(f'{name}\n')
        _git(repo, 'add', name)
        _git(repo, 'commit', '-q', '-m', f'task {task_id} change {n}', when=when + timedelta(minutes=n))
        shas.append(_git(repo, 'rev-parse', 'HEAD'))
    _git(repo, 'checkout', '-q', 'main')
    return shas


def _merge_and_delete(repo: Path, task_id: int, *, when: datetime) -> str:
    branch = f'task/{task_id}'
    _git(repo, 'merge', '-q', '--no-ff', '-m', f'Merge {branch} into main', branch, when=when)
    merge_sha = _git(repo, 'rev-parse', 'HEAD')
    _git(repo, 'branch', '-q', '-D', branch)
    return merge_sha


def _cherry_pick_onto_main(repo: Path, shas: list[str], *, when: datetime) -> None:
    """Cherry-pick keeps author date, email and subject; only the committer date is pinned."""
    env = dict(os.environ)
    env['GIT_COMMITTER_DATE'] = f'@{int(when.timestamp())} +0000'
    for sha in shas:
        subprocess.run(
            ['git', '-C', str(repo), 'cherry-pick', sha], check=True, capture_output=True, env=env,
        )


def _landed_by_marker(repo: Path, task_id: int) -> str:
    _branch_with_commits(repo, task_id, 1, when=STALE - timedelta(hours=1))
    return _merge_and_delete(repo, task_id, when=STALE)


def _live_with_worktree(repo: Path, task_id: int) -> None:
    _branch_with_commits(repo, task_id, 1, when=RECENT)
    _git(repo, 'worktree', 'add', '-q', str(repo.parent / f'wt-{task_id}'), f'task/{task_id}')


def _task(task_id: int, *, status: str = 'in-progress', **metadata: object) -> dict:
    return {'id': task_id, 'status': status, 'metadata': dict(metadata)}


async def _snapshot(repo: Path, *tasks: dict) -> LiveWorkflowSnapshot:
    return await build_live_workflow_snapshot(list(tasks), ProjectRoot(str(repo)), now=NOW)


def _line_for(rendered: str, branch: str) -> str:
    lines = [line for line in rendered.splitlines() if line.startswith(f'- {branch}:')]
    assert len(lines) == 1, f'expected exactly one row for {branch} in:\n{rendered}'
    return lines[0]


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path))
    root = tmp_path / 'repo'
    root.mkdir()
    _init_git_repo(root)
    return root


class TestLandedOnMainRendering:
    @pytest.mark.asyncio
    async def test_deleted_branch_with_merge_marker_renders_landed(self, repo: Path) -> None:
        merge_sha = _landed_by_marker(repo, 101)

        snapshot = await _snapshot(repo, _task(101))

        row = snapshot.row_for('101')
        assert row is not None, 'a landed task gets a row even though nothing about it is live'
        assert row.landing.landed is True
        assert row.landing.evidence is LandingEvidence.MERGE_MARKER
        assert row.is_live is False
        line = _line_for(snapshot.render(), 'task/101')
        assert 'landed=true' in line
        assert 'merge-marker' in line
        assert merge_sha[:10] in line
        assert 'not live' in line

    @pytest.mark.asyncio
    async def test_live_branch_with_commits_off_main_renders_not_landed(self, repo: Path) -> None:
        _live_with_worktree(repo, 102)

        snapshot = await _snapshot(repo, _task(102))

        row = snapshot.row_for('102')
        assert row is not None and row.is_live is True
        assert row.landing.landed is False
        assert 'landed=false' in _line_for(snapshot.render(), 'task/102')

    @pytest.mark.asyncio
    async def test_genuinely_stranded_task_stays_absent(self, repo: Path) -> None:
        _landed_by_marker(repo, 101)
        _branch_with_commits(repo, 108, 1, when=STALE)

        snapshot = await _snapshot(repo, _task(101), _task(108))

        assert snapshot.row_for('108') is None, (
            'unlanded work, no worktree and no recent commit must stay absent: absence marks a stranded candidate'
        )
        rendered = snapshot.render()
        assert 'task/101' in rendered
        assert 'task/108' not in rendered

    @pytest.mark.asyncio
    async def test_rebased_twins_render_landed_for_the_task_3838_shape(self, repo: Path) -> None:
        shas = _branch_with_commits(repo, 103, 2, when=STALE)
        _cherry_pick_onto_main(repo, shas, when=STALE + timedelta(hours=1))

        snapshot = await _snapshot(repo, _task(103))

        row = snapshot.row_for('103')
        assert row is not None and row.is_live is False
        assert 'landed=true (rebased-twins)' in _line_for(snapshot.render(), 'task/103')

    @pytest.mark.asyncio
    async def test_marker_older_than_reopen_does_not_land_the_reopened_task(self, repo: Path) -> None:
        _landed_by_marker(repo, 101)
        _landed_by_marker(repo, 109)
        reopened = _task(109, reopen_at=(STALE + timedelta(hours=1)).isoformat())

        snapshot = await _snapshot(repo, _task(101), reopened)

        assert snapshot.row_for('109') is None, 'the marker predates the reopen, so it is the previous run'
        assert 'task/109' not in snapshot.render()

    @pytest.mark.asyncio
    async def test_failed_landing_probe_renders_unknown_and_warns(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        _live_with_worktree(repo, 102)
        _landed_by_marker(repo, 101)

        async def _raising_probe(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError('probe exploded')

        monkeypatch.setattr(section_module, 'probe_landing', _raising_probe)
        with caplog.at_level(logging.WARNING, logger=section_module.__name__):
            snapshot = await _snapshot(repo, _task(101), _task(102))

        live_row = snapshot.row_for('102')
        assert live_row is not None and live_row.landing == LandingVerdict.unknown()
        assert 'landed=unknown' in _line_for(snapshot.render(), 'task/102')
        assert snapshot.row_for('101') is None, 'an unknown landing never earns a not-live task a row'
        assert any(
            record.levelno == logging.WARNING and str(repo) in record.getMessage()
            for record in caplog.records
            if record.name == section_module.__name__
        ), 'a failed landing probe is logged at WARNING naming the project root'

    @pytest.mark.asyncio
    async def test_render_live_workflow_section_is_the_snapshot_rendered(self, repo: Path) -> None:
        _landed_by_marker(repo, 101)
        _live_with_worktree(repo, 102)
        tasks = [_task(101), _task(102)]

        snapshot = await _snapshot(repo, *tasks)

        rendered = await render_live_workflow_section(tasks, ProjectRoot(str(repo)), now=NOW)
        assert rendered == snapshot.render()
