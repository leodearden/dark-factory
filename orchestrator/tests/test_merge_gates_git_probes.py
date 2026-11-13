"""The post-advance finalize path and its git-history probes, driven over real git.

Every scene is a real repository seeded by ``_git_fixtures.seed_repo``, a real
:class:`GitOps` and a real task worktree, so the git probes' own answers decide
each outcome.  Nothing here patches the equivalence or pyright gates out.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from pathlib import Path
from typing import NamedTuple

import pytest
from _git_fixtures import seed_repo
from _merge_lane_fakes import lane_scene_config
from _orch_helpers import git_env_with_ceiling
from df_pytest_isolation import git_redirect_env

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import gates
from orchestrator.merge_lane.gates import (
    POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
    Gate,
    GateVerdict,
    _finalize_advanced_merge,
    is_cross_repo_task,
    resolve_already_landed_branch,
)
from orchestrator.merge_lane.types import MergeOutcome, MergeRequest, OutcomeKind, QueuedBranch

TASK_ID = '6411'
TASK_BRANCH = f'task/{TASK_ID}'
GATES_LOGGER = 'orchestrator.merge_lane'


class Scene(NamedTuple):
    """A landing on main next to the task worktree it came from."""

    worktree: Path
    base: str
    landed: str
    branch_tip: str


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo')


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main', branch_prefix='task/', remote='origin',
        worktree_dir='.worktrees', push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, repo: Path) -> GitOps:
    return GitOps(git_config, repo)


@pytest.fixture
def lane_config(repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return lane_scene_config(repo, git_config)


def _git(cwd: Path, *args: str) -> str:
    env = git_env_with_ceiling(cwd)
    for key in git_redirect_env(env):
        del env[key]
    completed = subprocess.run(
        ['git', *args], cwd=cwd, env=env, check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def _commit(cwd: Path, files: dict[str, str], message: str) -> str:
    for relpath, text in files.items():
        target = cwd / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    _git(cwd, 'add', '-A')
    _git(cwd, 'commit', '-m', message)
    return _git(cwd, 'rev-parse', 'HEAD')


def _task_worktree(repo: Path, tmp_path: Path) -> Path:
    worktree = tmp_path / 'wt'
    _git(repo, 'worktree', 'add', '-b', TASK_BRANCH, str(worktree), 'main')
    return worktree


def _linear_landing_with_drifted_worktree(repo: Path, tmp_path: Path) -> Scene:
    """Main lands ``feature.txt`` as a single-parent commit; the worktree drifts.

    After the landing's snapshot the task worktree committed different content
    to the same file, so its live HEAD disagrees with what main holds.
    """
    base = _git(repo, 'rev-parse', 'HEAD')
    worktree = _task_worktree(repo, tmp_path)
    drifted = _commit(worktree, {'feature.txt': 'drifted after snapshot\n'}, 'drift after snapshot')
    landed = _commit(repo, {'feature.txt': 'landed\n'}, f'impl({TASK_ID}): land linearly')
    return Scene(worktree=worktree, base=base, landed=landed, branch_tip=drifted)


def _merge_landing(repo: Path, tmp_path: Path) -> Scene:
    """Main lands the task branch as a ``--no-ff`` merge whose second parent is the branch tip."""
    base = _git(repo, 'rev-parse', 'HEAD')
    worktree = _task_worktree(repo, tmp_path)
    branch_tip = _commit(worktree, {'feature.txt': 'branch work\n'}, f'impl({TASK_ID}): branch work')
    _git(repo, 'merge', '--no-ff', '-m', f'Merge {TASK_BRANCH} into main', TASK_BRANCH)
    landed = _git(repo, 'rev-parse', 'HEAD')
    return Scene(worktree=worktree, base=base, landed=landed, branch_tip=branch_tip)


def _request(
    config: OrchestratorConfig, worktree: Path, *, snapshot_tip: str | None = None,
) -> MergeRequest:
    return MergeRequest(
        task_id=TASK_ID,
        branch=QueuedBranch.parse(TASK_BRANCH, config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        snapshot_tip=snapshot_tip,
    )


async def _finalize(
    git_ops: GitOps,
    req: MergeRequest,
    *,
    advanced_sha: str,
    base_sha: str,
    event_store: EventStore | None = None,
    started_monotonic: float | None = None,
    **kwargs,
) -> MergeOutcome:
    return await _finalize_advanced_merge(
        git_ops, req, event_store,
        merge_commit_fallback=advanced_sha,
        base_sha=base_sha,
        started_monotonic=started_monotonic,
        cas_retries={},
        timeouts={},
        enospc_retries={},
        **kwargs,
    )


@pytest.mark.asyncio
async def test_linear_landing_without_a_recoverable_tip_does_not_consult_the_drifted_worktree(
    repo: Path, tmp_path: Path, git_ops: GitOps, lane_config: OrchestratorConfig,
) -> None:
    """A positively linear landing with no tip on record lands without reading the worktree.

    No second parent, no caller-supplied tip and no snapshot tip: the landed
    commit is the branch's work by construction, so the task-1928 fail-safe
    suppresses the live worktree-HEAD comparison that would otherwise report
    the post-snapshot drift as a dropped file.
    """
    scene = _linear_landing_with_drifted_worktree(repo, tmp_path)

    outcome = await _finalize(
        git_ops, _request(lane_config, scene.worktree),
        advanced_sha=scene.landed, base_sha=scene.base,
    )

    assert (outcome.status, outcome.merge_sha, outcome.push_status) == (
        'done', scene.landed, 'noop',
    )


@pytest.mark.asyncio
async def test_snapshot_tip_is_the_last_tip_tier(
    repo: Path, tmp_path: Path, git_ops: GitOps, lane_config: OrchestratorConfig,
) -> None:
    """With no second parent and no caller tip, the request's snapshot tip is compared.

    The same drift the fail-safe ignores is caught as soon as a tip is on
    record, which is what makes the linear-landing test above meaningful.
    """
    scene = _linear_landing_with_drifted_worktree(repo, tmp_path)

    outcome = await _finalize(
        git_ops, _request(lane_config, scene.worktree, snapshot_tip=scene.branch_tip),
        advanced_sha=scene.landed, base_sha=scene.base,
    )

    assert outcome.status == 'blocked'
    assert outcome.reason.startswith(POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX)
    assert 'feature.txt' in outcome.reason
    assert outcome.merge_sha == scene.landed


# The landing is timed from a start a fixed 1_000_000 s in the past, so its
# duration is that offset plus finalize's real elapsed time.  The real part is
# bounded by the per-test timeout (540 s ini default, 300 s under verify), and
# the ceiling stays well short of the offset itself, so a duration measured in
# the wrong unit or scale cannot land inside the window.
PRETEND_ELAPSED_S = 1_000_000.0
PRETEND_ELAPSED_MS = 1_000_000_000
REAL_ELAPSED_CEILING_MS = 600_000
TRAIN_TAGS = {'train_id': 'train-6411', 'member_task_ids': ['6411', '6412']}


@pytest.fixture
def event_store(tmp_path: Path) -> EventStore:
    return EventStore(tmp_path / 'runs.db', run_id='run-6411')


def _assert_single_train_tagged_row(event_store: EventStore, outcome: OutcomeKind) -> None:
    rows = event_store.fetch_events_by_type(EventType.merge_attempt)
    assert len(rows) == 1, rows
    row = rows[0]
    assert (row['task_id'], row['phase'], row['data']) == (
        TASK_ID, 'merge', {'outcome': outcome, **TRAIN_TAGS},
    )
    assert row['duration_ms'] is not None
    assert PRETEND_ELAPSED_MS <= row['duration_ms'] <= PRETEND_ELAPSED_MS + REAL_ELAPSED_CEILING_MS


@pytest.mark.asyncio
async def test_done_landing_records_one_merge_attempt_row(
    repo: Path, tmp_path: Path, git_ops: GitOps, lane_config: OrchestratorConfig,
    event_store: EventStore,
) -> None:
    """A clean landing persists exactly one train-tagged ``done`` row, timed from the caller's start."""
    scene = _merge_landing(repo, tmp_path)

    outcome = await _finalize(
        git_ops, _request(lane_config, scene.worktree),
        advanced_sha=scene.landed, base_sha=scene.base,
        event_store=event_store,
        started_monotonic=time.monotonic() - PRETEND_ELAPSED_S,
        **TRAIN_TAGS,
    )

    assert outcome.status == 'done'
    _assert_single_train_tagged_row(event_store, OutcomeKind.done)


@pytest.mark.asyncio
async def test_blocked_landing_records_the_gate_outcome_row(
    repo: Path, tmp_path: Path, git_ops: GitOps, lane_config: OrchestratorConfig,
    event_store: EventStore,
) -> None:
    """A gate-blocked landing persists exactly one train-tagged row naming the gate's outcome.

    No ``done`` row follows it: the blocked landing's row is the whole record.
    """
    scene = _linear_landing_with_drifted_worktree(repo, tmp_path)

    outcome = await _finalize(
        git_ops, _request(lane_config, scene.worktree, snapshot_tip=scene.branch_tip),
        advanced_sha=scene.landed, base_sha=scene.base,
        event_store=event_store,
        started_monotonic=time.monotonic() - PRETEND_ELAPSED_S,
        **TRAIN_TAGS,
    )

    assert outcome.status == 'blocked'
    _assert_single_train_tagged_row(event_store, OutcomeKind.post_merge_equivalence_failed)


@pytest.mark.asyncio
async def test_unlabelled_finalize_names_the_bare_task_in_its_gate_run_line(
    repo: Path, tmp_path: Path, git_ops: GitOps, lane_config: OrchestratorConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Without a log label the gate-run line names the task alone, then lists every gate run."""
    scene = _merge_landing(repo, tmp_path)

    with caplog.at_level(logging.INFO, logger=GATES_LOGGER):
        await _finalize(
            git_ops, _request(lane_config, scene.worktree),
            advanced_sha=scene.landed, base_sha=scene.base,
        )

    gate_run_lines = [
        record for record in caplog.records
        if record.getMessage() == f'Task {TASK_ID}: post-advance gates run: equivalence, pyright'
    ]
    assert len(gate_run_lines) == 1, [record.getMessage() for record in caplog.records]


@pytest.mark.asyncio
async def test_log_label_reaches_the_equivalence_failure_warning(
    repo: Path, tmp_path: Path, git_ops: GitOps, lane_config: OrchestratorConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The caller's log label follows the task id in the equivalence gate's failure warning."""
    scene = _linear_landing_with_drifted_worktree(repo, tmp_path)

    with caplog.at_level(logging.INFO, logger=GATES_LOGGER):
        await _finalize(
            git_ops, _request(lane_config, scene.worktree, snapshot_tip=scene.branch_tip),
            advanced_sha=scene.landed, base_sha=scene.base,
            log_label=' (speculative)',
        )

    warnings = [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert any(
        message.startswith(f'Task {TASK_ID} (speculative): post-merge equivalence failed')
        for message in warnings
    ), warnings


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('reason', 'emit_subtype'),
    [
        (None, OutcomeKind.post_merge_equivalence_failed),
        ('hand-built reason', None),
    ],
    ids=['reason-unset', 'emit-subtype-unset'],
)
async def test_a_gate_returning_a_hand_built_blocked_verdict_is_refused_loudly(
    reason: str | None,
    emit_subtype: OutcomeKind | None,
    repo: Path, tmp_path: Path, git_ops: GitOps, lane_config: OrchestratorConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blocked verdict missing either its reason or its outcome kind stops finalize outright.

    Only ``GateVerdict.block()`` guarantees both; a registered gate that builds
    the verdict by hand must fail loudly, naming itself, rather than land a
    row or an outcome with a hole in it.
    """
    async def run(_ctx: object) -> GateVerdict:
        return GateVerdict(passed=False, reason=reason, emit_subtype=emit_subtype)

    monkeypatch.setattr(gates, 'POST_ADVANCE_GATES', [Gate('misbuilt', run)])
    scene = _merge_landing(repo, tmp_path)

    with pytest.raises(AssertionError, match=r"gate 'misbuilt' returned a blocked GateVerdict"):
        await _finalize(
            git_ops, _request(lane_config, scene.worktree),
            advanced_sha=scene.landed, base_sha=scene.base,
        )


LANDED_TASK_ID = '9101'
LANDED_BRANCH = f'task/{LANDED_TASK_ID}'


@pytest.fixture
def landed_and_reparked(repo: Path) -> str:
    """Task 9101's work is merged into main, and its branch re-parked on that merge.

    The branch range is then empty, the merge subject attributes the landing
    to the task, and the landing touched ``pkg/a.py``, ``pkg/b.py`` and
    ``docs/INDEX/a.md``.  Returns the merge commit.
    """
    _git(repo, 'checkout', '-b', LANDED_BRANCH)
    _commit(
        repo,
        {'pkg/a.py': 'A = 1\n', 'pkg/b.py': 'B = 2\n', 'docs/INDEX/a.md': '# a\n'},
        f'impl({LANDED_TASK_ID}): deliver',
    )
    _git(repo, 'checkout', 'main')
    _git(repo, 'merge', '--no-ff', '-m', f'Merge {LANDED_BRANCH} into main', LANDED_BRANCH)
    merge = _git(repo, 'rev-parse', 'HEAD')
    _git(repo, 'branch', '-f', LANDED_BRANCH, merge)
    return merge


async def _matched_files(git_ops: GitOps, landed: str, plan_files: list[str]) -> list[str] | None:
    result = await resolve_already_landed_branch(
        plan_files, landed, landed, git_ops, task_id=LANDED_TASK_ID, branch=LANDED_BRANCH,
    )
    return None if result is None else result.matched_files


@pytest.mark.asyncio
async def test_a_declared_directory_is_covered_by_a_file_beneath_it(
    git_ops: GitOps, landed_and_reparked: str,
) -> None:
    """A declared directory counts as delivered when the landing touched a file inside it."""
    assert await _matched_files(git_ops, landed_and_reparked, ['pkg']) == ['pkg']


@pytest.mark.asyncio
async def test_only_the_trailing_separator_is_trimmed_from_a_declared_directory(
    git_ops: GitOps, landed_and_reparked: str,
) -> None:
    """Only a trailing separator is trimmed from a declared directory, never its own last letters."""
    assert await _matched_files(git_ops, landed_and_reparked, ['docs/INDEX']) == ['docs/INDEX']


@pytest.mark.asyncio
async def test_trailing_whitespace_is_not_trimmed_into_a_different_directory(
    git_ops: GitOps, landed_and_reparked: str,
) -> None:
    """``'pkg '`` names a different path from ``pkg``, so the landing does not cover it."""
    assert await _matched_files(git_ops, landed_and_reparked, ['pkg ']) is None


@pytest.mark.asyncio
async def test_a_leading_slash_entry_is_not_rerooted_onto_the_repo(
    git_ops: GitOps, landed_and_reparked: str,
) -> None:
    """``'/pkg'`` is an absolute path, not the repo's ``pkg``, so the landing does not cover it."""
    assert await _matched_files(git_ops, landed_and_reparked, ['/pkg']) is None


@pytest.mark.asyncio
async def test_every_declared_entry_is_matched(
    git_ops: GitOps, landed_and_reparked: str,
) -> None:
    """Coverage keeps checking after a hit, and reports every declared entry it matched."""
    matched = await _matched_files(git_ops, landed_and_reparked, ['pkg/a.py', 'pkg/b.py'])

    assert matched == ['pkg/a.py', 'pkg/b.py']


@pytest.mark.asyncio
async def test_an_uncovered_entry_after_a_covered_one_fails_closed(
    git_ops: GitOps, landed_and_reparked: str,
) -> None:
    """A partial landing never excuses the branch: one uncovered entry declines the whole set."""
    assert await _matched_files(git_ops, landed_and_reparked, ['pkg/a.py', 'other/c.py']) is None


@pytest.mark.asyncio
async def test_a_blank_entry_is_skipped_not_terminal(
    git_ops: GitOps, landed_and_reparked: str,
) -> None:
    """A blank declared entry is ignored and the entries after it are still checked."""
    assert await _matched_files(git_ops, landed_and_reparked, ['', 'pkg/a.py']) == ['pkg/a.py']


def test_plan_files_of_only_blank_entries_are_not_cross_repo(tmp_path: Path) -> None:
    """Declared entries that are all blank name no foreign project, so the task is not cross-repo."""
    assert is_cross_repo_task(['', ''], tmp_path, None) is False
