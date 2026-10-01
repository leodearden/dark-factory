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
import re
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _fm_helpers import _init_git_repo

from fused_memory.models.reconciliation import Watermark
from fused_memory.models.scope import ProjectRoot
from fused_memory.reconciliation import live_workflow_section as section_module
from fused_memory.reconciliation.live_workflow_section import (
    LIVE_WORKFLOW_SECTION_HEADER,
    NOT_LIVE_TOKEN,
    LandedToken,
    LiveWorkflowSnapshot,
    build_live_workflow_snapshot,
    render_live_workflow_section,
)
from fused_memory.reconciliation.task_filter import FilteredTaskTree
from fused_memory.services.landed_on_main import LandingEvidence, LandingVerdict
from fused_memory.services.live_workflow_detector import ClaimantLabel

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


_TASK_REF = re.compile(r'task/\d')


def _pending_with_worktree(repo: Path, task_id: int, *, when: datetime = STALE) -> None:
    """A registered worktree on a non-bare branch whose commit is outside the recent window."""
    _branch_with_commits(repo, task_id, 1, when=when)
    _git(repo, 'worktree', 'add', '-q', str(repo.parent / f'wt-{task_id}'), f'task/{task_id}')


def _hold_project_lock(repo: Path) -> None:
    """A real orchestrator lock naming THIS process, so the lock genuinely reads as held."""
    lock = repo / 'data' / 'orchestrator' / 'orchestrator.lock'
    lock.parent.mkdir(parents=True)
    lock.write_text(f'PID {os.getpid()} started {(NOW - timedelta(days=3)).isoformat()}\n')


def _claimed(task_id: int, *, heartbeat_age: timedelta | None) -> dict:
    heartbeat = None if heartbeat_age is None else (NOW - heartbeat_age).isoformat()
    return {
        **_task(task_id, status='pending'),
        'claimant_run_id': f'run-a1d3b5dba75a/{task_id}-011fcff1/pid=1807449',
        'heartbeat_at': heartbeat,
    }


def _unclaimed(task_id: int) -> dict:
    return {**_task(task_id, status='pending'), 'claimant_run_id': None, 'heartbeat_at': None}


def _body_lines(rendered: str) -> list[str]:
    lines = rendered.splitlines()
    assert lines and lines[0].startswith(LIVE_WORKFLOW_SECTION_HEADER), rendered
    return lines[1:]


def _project_lines(rendered: str) -> list[str]:
    return [
        line for line in _body_lines(rendered)
        if not line.startswith('- ') and 'orchestrator lock' in line
    ]


def _legend_lines(rendered: str) -> list[str]:
    return [
        line for line in _body_lines(rendered)
        if not line.startswith('- ') and LandedToken.TRUE in line
    ]


def _section_of(payload: str) -> str:
    start = payload.index(LIVE_WORKFLOW_SECTION_HEADER)
    end = payload.find('\n#', start + 1)
    return payload[start:] if end == -1 else payload[start:end]


def _tree(tasks: list[dict]) -> FilteredTaskTree:
    return FilteredTaskTree(
        active_tasks=tasks,
        done_tasks=[],
        cancelled_tasks=[],
        done_count=0,
        cancelled_count=0,
        other_count=0,
        total_count=len(tasks),
        max_task_id=max(t['id'] for t in tasks),
    )


def _assert_rows_are_honest(section: str) -> None:
    rows = [line for line in section.splitlines() if line.startswith('- task/')]
    assert rows, section
    assert all('claimant=' in row for row in rows), section
    assert not any('orchestrator' in row for row in rows), section


class TestLiveWorkflowSectionIsHonest:
    @pytest.mark.asyncio
    async def test_claimant_discriminates_the_3254_shape_from_the_3879_shape(self, repo: Path) -> None:
        _hold_project_lock(repo)
        _pending_with_worktree(repo, 3254)
        _pending_with_worktree(repo, 3879)

        snapshot = await _snapshot(
            repo, _unclaimed(3254), _claimed(3879, heartbeat_age=timedelta(seconds=30)),
        )

        unclaimed, claimed = snapshot.row_for('3254'), snapshot.row_for('3879')
        assert unclaimed is not None and claimed is not None
        assert unclaimed.claimant is ClaimantLabel.NONE
        assert claimed.claimant is ClaimantLabel.LIVE
        rendered = snapshot.render()
        unclaimed_line = _line_for(rendered, 'task/3254')
        claimed_line = _line_for(rendered, 'task/3879')
        assert unclaimed_line.removeprefix('- task/3254') != claimed_line.removeprefix('- task/3879')
        assert f'claimant={ClaimantLabel.NONE}' in unclaimed_line
        assert f'claimant={ClaimantLabel.LIVE}' in claimed_line

    @pytest.mark.asyncio
    async def test_no_row_carries_the_project_wide_orchestrator_token(self, repo: Path) -> None:
        _hold_project_lock(repo)
        _pending_with_worktree(repo, 3254)

        snapshot = await _snapshot(repo, _unclaimed(3254))

        rows = [line for line in snapshot.render().splitlines() if line.startswith('- task/')]
        assert rows
        assert not any('orchestrator' in row for row in rows)

    @pytest.mark.asyncio
    async def test_a_held_lock_is_one_project_line(self, repo: Path) -> None:
        _hold_project_lock(repo)
        _pending_with_worktree(repo, 3254)
        _pending_with_worktree(repo, 3879)

        snapshot = await _snapshot(repo, _unclaimed(3254), _unclaimed(3879))

        assert snapshot.project_orchestrator_live is True
        project_lines = _project_lines(snapshot.render())
        assert len(project_lines) == 1, snapshot.render()
        line = project_lines[0]
        assert not line.startswith('#'), 'payload tests slice the section on a newline-hash'
        assert not _TASK_REF.search(line), 'payload tests assert no task/<id> outside its row'

    @pytest.mark.asyncio
    async def test_an_unheld_lock_renders_no_project_line(self, repo: Path) -> None:
        _pending_with_worktree(repo, 3254)

        snapshot = await _snapshot(repo, _unclaimed(3254))

        assert snapshot.project_orchestrator_live is False
        rendered = snapshot.render()
        assert _project_lines(rendered) == []
        assert 'claimant=' in _line_for(rendered, 'task/3254')

    @pytest.mark.asyncio
    async def test_a_failed_lock_hoist_renders_unknown_not_silence(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _pending_with_worktree(repo, 3254)

        def _raising(_root: object) -> bool:
            raise OSError('lock unreadable')

        monkeypatch.setattr(section_module, 'is_orchestrator_live_for', _raising)
        snapshot = await _snapshot(repo, _unclaimed(3254))

        assert snapshot.project_orchestrator_live is None
        project_lines = _project_lines(snapshot.render())
        assert len(project_lines) == 1 and 'UNKNOWN' in project_lines[0]

    @pytest.mark.asyncio
    async def test_a_task_listed_only_through_the_lock_says_it_has_no_per_task_signal(
        self, repo: Path,
    ) -> None:
        _hold_project_lock(repo)

        snapshot = await _snapshot(repo, _unclaimed(7001))

        row = snapshot.row_for('7001')
        assert row is not None and row.is_live is True
        line = _line_for(snapshot.render(), 'task/7001')
        assert 'no per-task signal' in line
        assert not re.search(r'\blive\b', line), 'a lock-only row must not read as live'
        assert 'orchestrator' not in line

    @pytest.mark.asyncio
    async def test_one_legend_line_names_every_rendered_token(self, repo: Path) -> None:
        _pending_with_worktree(repo, 3254)

        rendered = (await _snapshot(repo, _unclaimed(3254))).render()

        legends = _legend_lines(rendered)
        assert len(legends) == 1, rendered
        legend = legends[0]
        assert not legend.startswith('#') and not _TASK_REF.search(legend)
        for label in ClaimantLabel:
            assert f'claimant={label}' in legend
        for token in LandedToken:
            assert token in legend
        assert NOT_LIVE_TOKEN in legend

    @pytest.mark.asyncio
    async def test_a_lingering_claimant_renders_stale(self, repo: Path) -> None:
        _pending_with_worktree(repo, 3879)

        snapshot = await _snapshot(repo, _claimed(3879, heartbeat_age=timedelta(minutes=45)))

        assert f'claimant={ClaimantLabel.STALE}' in _line_for(snapshot.render(), 'task/3879')

    @pytest.mark.asyncio
    async def test_stage2_payload_delivers_the_honest_section(self, repo: Path) -> None:
        from test_stages import _mock_stage_deps, make_configured_task_knowledge_sync_stage

        _hold_project_lock(repo)
        _pending_with_worktree(repo, 3254, when=datetime.now(UTC) - timedelta(hours=48))
        stage = make_configured_task_knowledge_sync_stage(
            _mock_stage_deps(), project_id='dark_factory', project_root=str(repo),
        )
        stage.filtered_task_tree = _tree([_unclaimed(3254)])

        payload = await stage.assemble_payload([], Watermark(project_id='dark_factory'), [])

        _assert_rows_are_honest(_section_of(payload))

    @pytest.mark.asyncio
    async def test_stage1_payload_delivers_the_honest_section(self, repo: Path) -> None:
        from reconciliation.consolidator_fixtures import make_consolidator

        _hold_project_lock(repo)
        _pending_with_worktree(repo, 3254, when=datetime.now(UTC) - timedelta(hours=48))
        stage = make_consolidator(project_root=str(repo))
        stage.filtered_task_tree = _tree([_unclaimed(3254)])

        payload = await stage.assemble_payload(
            events=[], watermark=Watermark(project_id='dark_factory'), prior_reports=[],
        )

        _assert_rows_are_honest(_section_of(payload))
