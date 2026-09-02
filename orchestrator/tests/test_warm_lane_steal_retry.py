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


# ---------------------------------------------------------------------------
# step-05: the bounded steal-path retry actually serves a DIFFERENT lane
# ---------------------------------------------------------------------------


def _install_selective_seed_failure(
    git_ops: GitOps, *, failing_rc: int = 1, fail_lanes: set[Path] | None = None,
) -> list[Path]:
    """Make ``_seed_warm_lane`` hostile for a chosen set of lanes.

    ``_seed_warm_lane`` is the established injection seam for warm-lane
    hostility (cf. ``test_harness_warm_lane_wiring.py``'s
    ``git_ops._seed_warm_lane = AsyncMock(return_value=1)``); wrapping the
    BOUND method rather than replacing it keeps every non-hostile lane on the
    real implementation, so the surviving lane is provisioned for real.

    When *fail_lanes* is None the FIRST lane this wrapper is asked to seed
    becomes the hostile one — the lane the steal valve happens to pick — which
    keeps the test independent of ``reclaim_victim``'s victim ordering.

    Returns the (live) list of lane dirs the wrapper was called with, in order.
    """
    real_seed = git_ops._seed_warm_lane
    seen: list[Path] = []
    hostile: set[Path] = set() if fail_lanes is None else set(fail_lanes)
    pick_first = fail_lanes is None

    async def _seed(lane_dir: Path, mode: str, *, take_lane_lock: bool = True) -> int:
        lane = Path(lane_dir)
        seen.append(lane)
        if pick_first and not hostile:
            hostile.add(lane)
        if lane in hostile:
            return failing_rc
        return await real_seed(lane_dir, mode, take_lane_lock=take_lane_lock)

    git_ops._seed_warm_lane = _seed  # type: ignore[method-assign]
    return seen


@pytest.mark.asyncio
class TestStealRetriesAnotherLane:
    """A steal that lands on a hostile lane must retry a DIFFERENT lane.

    The 2026-08-29 incident shape: ``_try_reclaim_lane_for`` hands out whatever
    ``reclaim_victim`` re-keys — conflicted index and all — with no validation
    of the victim lane's state, and the resulting FAULT hard-BLOCKs the task.
    """

    async def test_second_lane_is_served_after_first_steal_faults(
        self, wl_git_repo: Path,
    ):
        git_ops, lanes, start_ref = await _setup_pool(wl_git_repo, size=2)
        pool = git_ops.warm_lane_pool
        assert pool is not None
        seen = _install_selective_seed_failure(git_ops)

        result = await git_ops.acquire_warm_lane('Z', start_ref)

        assert isinstance(result, WorktreeInfo), (
            f'a hostile first steal must be retried on another lane, not '
            f'returned as a failure sentinel; got {result!r}'
        )
        hostile = seen[0]
        served = result.path
        assert served != hostile, (
            f'the retry must serve a DIFFERENT lane; served the hostile '
            f'{hostile} again'
        )
        assert served in lanes, f'served lane {served} is not a pool lane'

        _, branch_raw, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=served,
        )
        assert branch_raw.strip() == 'task/Z', (
            f'served lane HEAD must be on task/Z, got {branch_raw.strip()!r}'
        )
        assert pool.assignment_for('Z') == served, (
            'the pool must map Z to the lane it was actually served'
        )

    async def test_failed_stolen_lane_is_not_re_handed(self, wl_git_repo: Path):
        """The hostile lane must be seeded exactly ONCE.

        This is the assertion that fails if the exclusion is dropped: a failed
        attempt unwinds through ``_abort_lane_acquisition`` whose final
        ``pool.release(lane)`` returns the hostile lane to FREE, making it the
        lowest-index lane ``acquire_for`` hands straight back — so without the
        exclusion + veto the retry re-picks the same lane and the loop is a
        no-op.
        """
        git_ops, _lanes, start_ref = await _setup_pool(wl_git_repo, size=2)
        seen = _install_selective_seed_failure(git_ops)

        result = await git_ops.acquire_warm_lane('Z', start_ref)

        assert isinstance(result, WorktreeInfo), f'expected success, got {result!r}'
        hostile = seen[0]
        assert seen.count(hostile) == 1, (
            f'the hostile lane {hostile} was re-handed to the retry '
            f'({seen.count(hostile)} seed attempts); the exclusion + veto did '
            f'not hold. Full seed order: {seen}'
        )

    async def test_retry_is_logged(self, wl_git_repo: Path, caplog):
        """The retry must be greppable in the journal alongside the existing
        ``reclaim-on-exhaustion — stole lane`` line."""
        git_ops, _lanes, start_ref = await _setup_pool(wl_git_repo, size=2)
        seen = _install_selective_seed_failure(git_ops)

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            result = await git_ops.acquire_warm_lane('Z', start_ref)

        assert isinstance(result, WorktreeInfo), f'expected success, got {result!r}'
        hostile = seen[0]
        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        retry_lines = [m for m in messages if 'steal-retry' in m]
        assert retry_lines, (
            f'expected a WARNING naming the steal-retry; got: {messages}'
        )
        line = retry_lines[0]
        assert str(hostile) in line, (
            f'the retry WARNING must name the failed lane {hostile}: {line!r}'
        )
        assert 'attempt' in line, (
            f'the retry WARNING must name the attempt number: {line!r}'
        )
        assert 'fault' in line, (
            f'the retry WARNING must name the retryable sentinel: {line!r}'
        )

    async def test_free_lane_fault_is_not_retried(self, wl_git_repo: Path):
        """Retry is scoped to the STEAL path.

        A size-1 pool with no victim leaves the lane FREE, so ``acquire_for``
        succeeds and the steal path is never entered. A FREE lane's failure is
        not evidence that a different lane is healthier, so its disposition
        must stay exactly as it is today: FAULT after exactly ONE seed attempt.
        """
        from orchestrator.git_ops import WarmLaneUnavailable

        git_ops, _lanes, start_ref = await _setup_pool(
            wl_git_repo, size=1, exhaust=False,
        )
        seen = _install_selective_seed_failure(git_ops)

        result = await git_ops.acquire_warm_lane('Z', start_ref)

        assert result is WarmLaneUnavailable.FAULT, (
            f'a FREE-lane seed fault must keep its existing disposition; '
            f'got {result!r}'
        )
        assert len(seen) == 1, (
            f'the FREE-lane path must not be retried; saw {len(seen)} seed '
            f'attempts: {seen}'
        )


# ---------------------------------------------------------------------------
# step-07: exhausting the retries yields STEAL_FAILED, without manufacturing a
#          false structural-exhaustion signal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStealFailedSentinel:
    """When every retryable steal attempt fails, the acquire must report a
    DISTINCT requeue-class sentinel — and must not corrupt the pool-level
    structural-exhaustion signal on the way there.
    """

    async def test_all_attempts_fault_returns_steal_failed(self, wl_git_repo: Path):
        from orchestrator.git_ops import (
            _WARM_LANE_STEAL_MAX_ATTEMPTS,
            WarmLaneUnavailable,
        )
        from orchestrator.warm_lane_pool import LaneState

        git_ops, lanes, start_ref = await _setup_pool(wl_git_repo, size=3)
        pool = git_ops.warm_lane_pool
        assert pool is not None
        seen = _install_selective_seed_failure(git_ops, fail_lanes=set(lanes))

        result = await git_ops.acquire_warm_lane('Z', start_ref)

        assert result is WarmLaneUnavailable.STEAL_FAILED, (
            f'every stolen lane failing is a POOL-PRESSURE condition with its '
            f'own requeue class, not a per-task FAULT; got {result!r}'
        )
        assert len(seen) == _WARM_LANE_STEAL_MAX_ATTEMPTS, (
            f'expected exactly {_WARM_LANE_STEAL_MAX_ATTEMPTS} steal attempts, '
            f'saw {len(seen)}: {seen}'
        )
        assert len(set(seen)) == _WARM_LANE_STEAL_MAX_ATTEMPTS, (
            f'each attempt must steal a DIFFERENT lane; got {seen}'
        )
        # No ASSIGNED leak: every attempted lane is back to FREE and the thief
        # holds nothing.
        assert pool.assignment_for('Z') is None, (
            'a failed acquire must not leave the thief holding an assignment'
        )
        for lane in lanes:
            assert pool.state(lane) is LaneState.FREE, (
                f'lane {lane} leaked as {pool.state(lane)!r} after a failed '
                f'steal-retry; every attempted lane must be released'
            )

    async def test_no_further_victim_on_retry_returns_steal_failed_without_exhaustion_signal(
        self, wl_git_repo: Path,
    ):
        """A retry that finds no further eligible victim is STEAL_FAILED — and
        must NOT walk the census / _note_structural_exhaustion path.

        Today one acquire reaches the reclaim-returned-None branch at most
        once, so it bumps ``_consecutive_exhausted`` at most once. With up to
        _WARM_LANE_STEAL_MAX_ATTEMPTS attempts per acquire, a single
        hostile-lane event on a small or heavily-pinned pool would bump the
        counter on every retry too, driving it toward the L2 threshold and
        firing spurious born-at-L2 structural-exhaustion escalations — trading
        the per-task L2s this task removes for pool-level L2s that are equally
        wrong.
        """
        from orchestrator.git_ops import WarmLaneUnavailable

        git_ops, lanes, start_ref = await _setup_pool(wl_git_repo, size=1)
        fires: list = []
        git_ops._on_structural_exhaustion = (
            lambda count, census: fires.append((count, census))
        )
        git_ops._consecutive_exhausted = 0
        seen = _install_selective_seed_failure(git_ops, fail_lanes=set(lanes))

        result = await git_ops.acquire_warm_lane('Z', start_ref)

        assert result is WarmLaneUnavailable.STEAL_FAILED, (
            f'a retry that finds no further victim is a steal failure, not '
            f'structural exhaustion; got {result!r}'
        )
        assert len(seen) == 1, (
            f'the sole lane can only be stolen once; saw {seen}'
        )
        assert fires == [], (
            f'the retry must NOT fire the structural-exhaustion callback; '
            f'fired {fires!r}'
        )
        assert git_ops._consecutive_exhausted == 0, (
            f'the retry must NOT bump the consecutive-exhausted counter; it '
            f'is {git_ops._consecutive_exhausted}'
        )

    async def test_first_attempt_no_victim_still_exhausted(
        self, wl_git_repo: Path, caplog,
    ):
        """Regression: a genuine FIRST-attempt exhaustion is untouched.

        With no eligible victim at all the steal path never fires, so the
        existing backpressure signal — EXHAUSTED, the census WARNING, and the
        _consecutive_exhausted bump — must survive the new retry loop intact.
        """
        from orchestrator.git_ops import WarmLaneUnavailable

        git_ops, _lanes, start_ref = await _setup_pool(wl_git_repo, size=1)

        async def _empty_provider(candidates):
            return set()

        git_ops.warm_lane_reclaim_candidate_provider = _empty_provider
        fires: list = []
        git_ops._on_structural_exhaustion = (
            lambda count, census: fires.append((count, census))
        )
        git_ops._consecutive_exhausted = 0

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            result = await git_ops.acquire_warm_lane('Z', start_ref)

        assert result is WarmLaneUnavailable.EXHAUSTED, (
            f'a genuine first-attempt exhaustion must still be EXHAUSTED; '
            f'got {result!r}'
        )
        assert git_ops._consecutive_exhausted == 1, (
            f'the existing backpressure counter must still increment; it is '
            f'{git_ops._consecutive_exhausted}'
        )
        messages = [r.getMessage() for r in caplog.records]
        assert any(
            'warm-lane pool EXHAUSTED' in m for m in messages
        ), f'the census WARNING must still be logged; got: {messages}'
