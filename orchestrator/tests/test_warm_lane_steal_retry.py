"""Tests for task 4930 — the warm-lane reclaim-on-exhaustion STEAL path must
retry a DIFFERENT lane instead of hard-BLOCKing on a hostile one.

The 2026-08-29 incident: ~2% of the ~65 daily reclaim-on-exhaustion steals land
on a lane that is hostile in one of three unrelated ways (a conflicted index
from a quarantined lane, a branch already checked out at another worktree, a
30s ``<lane>.lock`` wait lost to a concurrent GC reseed → rc=124). All three
collapse into ``WarmLaneUnavailable.FAULT`` — the ONE warm-lane discriminant
that is not a :class:`WarmLaneRequeue` — so ``create_worktree`` raises a bare
``RuntimeError``, ``workflow.py``'s ``except WarmLaneRequeue`` misses it, and
the task lands BLOCKED + L1 with ``agent_invocations=0``. Every neighbouring
condition requeues; only this one strands a task, ~1-2/day.

Test classes (added across the plan's TDD steps):
  step-03: TestReclaimExclude — ``_try_reclaim_lane_for(exclude=...)``
  step-05: TestStealRetriesAnotherLane — the bounded retry serves lane B
  step-07: TestStealFailedSentinel — retries exhausted → STEAL_FAILED, and no
           spurious structural-exhaustion signal
  step-09: TestCreateWorktreeMapping — the two new sentinels map to typed
           requeue exceptions, and the fall-through message stops lying

Convention note: this module keeps its OWN local copies of the ``wl_git_repo``
fixture and the warm-lane script/pool helpers rather than cross-importing from
``test_warm_lane_pool.py`` / ``test_git_ops.py`` — the same local-copy
convention ``test_warm_lane_structural_exhaustion.py`` follows (see its
``_init_repo`` / ``wl_git_repo`` at module scope).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
from _orch_helpers import assert_isolated_git_repo

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, WorktreeInfo, _run

# ---------------------------------------------------------------------------
# Local fixtures / helpers (see module docstring for the local-copy convention)
# ---------------------------------------------------------------------------


async def _init_repo(repo: Path) -> None:
    """Minimal git repo with one commit (mirrors test_warm_lane_pool.py).

    esc-3072-3: the :func:`assert_isolated_git_repo` pre-flight cannot run
    before ``git init`` — it refuses any directory that is not ALREADY a repo
    root, which is precisely what that call creates.  So the guard runs
    immediately AFTER the init and ahead of every other subprocess, which is
    the earliest point at which it is evaluable and the point that matters:
    ``git init`` only ever writes into its own ``cwd`` (it never adopts an
    enclosing repo), whereas the ``git add -A`` + ``git commit`` below WOULD
    retarget whatever encloses *repo* if *repo* were not a root.  Same
    ordering rationale as ``test_branch_work_landed.py::_Repo.init``.
    """
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    assert_isolated_git_repo(repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def wl_git_repo(tmp_path: Path) -> Path:
    """A git repo whose default warm-lane base + pool-storage sentinel exist so
    ``acquire_warm_lane``'s pre-acquire base-health gate sees
    ``WarmBaseHealth.OK`` and the mount-presence guard does not false-trip
    (mirrors test_warm_lane_structural_exhaustion.py::wl_git_repo)."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    # esc-3072-3 guard lives inside _init_repo, immediately after `git init`
    # (the earliest point the pre-flight is evaluable) — see its docstring.
    asyncio.run(_init_repo(repo))
    default_base = repo / '.worktrees' / '_merge-verify' / 'target'
    default_base.mkdir(parents=True, exist_ok=True)
    (default_base / '.keep').write_text('warm base sentinel\n')
    (repo / '.worktrees' / '.pool-root').touch()
    return repo


async def _add_warm_lane_scripts(repo: Path, port: int = 39411) -> None:
    """Commit stub seed-warm-lane.sh + setup-worktree-debug-port.sh into repo.

    Local copy of ``test_git_ops.py::_add_warm_lane_scripts``; the
    ``assert_isolated_git_repo`` guard is the FIRST statement for the same
    esc-3072-3 reason documented there (this helper runs a real ``git add -A``
    + ``git commit``, so an escaping call would commit into a live worktree).
    """
    assert_isolated_git_repo(repo)
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    seed_script = scripts_dir / 'seed-warm-lane.sh'
    seed_script.write_text(
        '#!/usr/bin/env bash\nmkdir -p "$2/target"\necho "seeded" > "$2/target/seeded.bin"\n'
    )
    seed_script.chmod(0o755)
    debug_script = scripts_dir / 'setup-worktree-debug-port.sh'
    debug_script.write_text(f'#!/usr/bin/env bash\necho {port}\n')
    debug_script.chmod(0o755)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add warm-lane scripts'], cwd=repo)


def _warm_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=True,
    )


async def _setup_pool(
    repo: Path, size: int, *, wire_reclaim: bool = True, exhaust: bool = True,
) -> tuple[GitOps, list[Path], str]:
    """Build a GitOps with a size-*size* warm-lane pool.

    When *exhaust* (the default), acquires EVERY lane for a distinct victim
    branch (``V0``..``V{size-1}``) so the pool is EXHAUSTED and the next
    acquire must take the reclaim-on-exhaustion steal path.

    When *wire_reclaim* (the default), wires the two reclaim callbacks the way
    ``test_git_ops.py::TestAcquireWarmLaneReclaimOnExhaustion`` does — by direct
    attribute assignment — so every victim is an eligible, non-dispatched
    steal candidate.

    Returns ``(git_ops, lanes_in_acquisition_order, start_ref)``.
    """
    await _add_warm_lane_scripts(repo)
    git_ops = GitOps(_warm_config(), repo, warm_lane_pool_size=size)
    assert git_ops.warm_lane_pool is not None

    _, sha_raw, _ = await _run(['git', 'rev-parse', 'main'], cwd=repo)
    start_ref = sha_raw.strip()

    lanes: list[Path] = []
    if exhaust:
        for k in range(size):
            victim = f'V{k}'
            result = await git_ops.acquire_warm_lane(victim, start_ref)
            assert isinstance(result, WorktreeInfo), (
                f'setup: expected WorktreeInfo for {victim}, got {result!r}'
            )
            lanes.append(result.path)
        assert await git_ops.warm_lane_pool.acquire_for('SETUP_PROBE') is None, (
            'setup: pool must be EXHAUSTED after acquiring every lane'
        )
    else:
        lanes = list(git_ops.warm_lane_pool.lane_paths())

    if wire_reclaim:
        async def _provider(candidates):
            return set(candidates)

        git_ops.warm_lane_reclaim_candidate_provider = _provider
        git_ops.warm_lane_dispatched_predicate = lambda b: False

    return git_ops, lanes, start_ref



# ---------------------------------------------------------------------------
# step-03: _try_reclaim_lane_for(exclude=...)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReclaimExclude:
    """``GitOps._try_reclaim_lane_for`` must skip lanes this acquire already
    stole-and-failed on.

    Half of a two-part exclusion (the other half is the veto-and-release in
    ``_acquire_warm_lane_impl``): without it the steal-path retry is a silent
    no-op, because a failed attempt unwinds through ``_abort_lane_acquisition``
    whose final ``pool.release(lane)`` returns the hostile lane to FREE — where
    it is the lowest-index candidate ``acquire_for`` hands straight back.
    """

    async def test_excluded_lane_is_never_stolen(self, wl_git_repo: Path):
        """With _lane-0 excluded, the valve must steal _lane-1 instead."""
        git_ops, lanes, _start_ref = await _setup_pool(wl_git_repo, size=2)
        lane0, lane1 = lanes
        pool = git_ops.warm_lane_pool
        assert pool is not None

        stolen = await git_ops._try_reclaim_lane_for(
            'Z', exclude=frozenset({lane0}),
        )

        assert stolen == lane1, (
            f'Excluded lane {lane0} must never be stolen; expected {lane1}, '
            f'got {stolen!r}'
        )
        assert pool.assignment_for('Z') == lane1, (
            'the thief must be re-keyed onto the non-excluded lane'
        )

    async def test_all_candidates_excluded_returns_none(self, wl_git_repo: Path):
        """Excluding every lane returns None WITHOUT disturbing any victim.

        The filter must short-circuit BEFORE ``reclaim_victim`` re-keys
        anything — a valve that steals and then discovers the lane is excluded
        would strand the victim's assignment.
        """
        git_ops, lanes, _start_ref = await _setup_pool(wl_git_repo, size=2)
        lane0, lane1 = lanes
        pool = git_ops.warm_lane_pool
        assert pool is not None

        stolen = await git_ops._try_reclaim_lane_for(
            'Z', exclude=frozenset({lane0, lane1}),
        )

        assert stolen is None, f'expected None with every lane excluded, got {stolen!r}'
        assert pool.assignment_for('Z') is None, (
            'no assignment may be created when nothing is eligible'
        )
        assert pool.assignment_for('V0') == lane0, (
            'victim V0 must be left untouched — the filter short-circuits '
            'before reclaim_victim re-keys anything'
        )
        assert pool.assignment_for('V1') == lane1, (
            'victim V1 must be left untouched'
        )

    async def test_exclude_omitted_is_byte_identical(
        self, wl_git_repo: Path, caplog,
    ):
        """Calling with no ``exclude`` kwarg at all is unchanged behaviour.

        Pins that all four existing call shapes are unaffected by the new
        keyword-only parameter's default.
        """
        git_ops, lanes, _start_ref = await _setup_pool(wl_git_repo, size=2)
        lane0 = lanes[0]
        pool = git_ops.warm_lane_pool
        assert pool is not None

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            stolen = await git_ops._try_reclaim_lane_for('Z')

        assert stolen == lane0, (
            f'with no exclusion the valve still steals the lowest-index '
            f'eligible lane {lane0}; got {stolen!r}'
        )
        assert pool.assignment_for('Z') == lane0
        assert any(
            'reclaim-on-exhaustion — stole lane' in r.getMessage()
            for r in caplog.records
        ), (
            f'the existing steal WARNING must still fire; got: '
            f'{[r.getMessage() for r in caplog.records]}'
        )
