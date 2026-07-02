"""Tests for GitOps.create_interactive_worktree — task α (2010).

Isolated interactive warm-worktree primitive: mints a fresh worktree in a new
_iact-* band and CoW-seeds its build cache by reusing _seed_warm_lane, WITHOUT
ever touching WarmLanePool (isolation invariant I1 — strictly disjoint from
the _lane-* dispatch pool and the _spec-* merge-speculation pool).

Mirrors test_warm_lane_integration_gate.py conventions: a committed seed stub
that creates <lane>/target/seeded.bin (orchestration-observable seededness —
literal filefrag/CoW extent-sharing is a reify/filesystem guarantee, NOT
asserted on a CI tmpfs), an ig-style git-repo fixture, and WarmLanePool
FREE-count / assignments_snapshot / is_lane assertions for the I1 signal.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest

from orchestrator import config as orchestrator_config
from orchestrator.config import GitConfig
from orchestrator.git_ops import (
    GitOps,
    InteractiveWorktreeInfo,
    InteractiveWorktreeLimitError,
    _run,
)
from orchestrator.warm_lane_pool import LaneState

# ---------------------------------------------------------------------------
# Repo fixture + seed-stub helper (mirrors test_warm_lane_integration_gate.py)
# ---------------------------------------------------------------------------


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def iw_git_repo(tmp_path: Path) -> Path:
    """Temporary git repo with an initial commit for interactive-worktree tests."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


async def _add_seed_script(repo: Path, *, exit_code: int = 0) -> None:
    """Commit a seed-warm-lane.sh stub into repo/scripts/.

    On exit_code == 0: creates <lane>/target/seeded.bin (orchestration-observable
    seededness — mirrors _add_all_warm_lane_scripts' seed stub in
    test_warm_lane_integration_gate.py).  On non-zero: exits with that code
    immediately (no target/ created).
    """
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    seed = scripts_dir / 'seed-warm-lane.sh'
    if exit_code == 0:
        seed.write_text(
            '#!/usr/bin/env bash\n'
            'mkdir -p "$2/target"\n'
            'echo "seeded" > "$2/target/seeded.bin"\n'
        )
    else:
        seed.write_text(
            f'#!/usr/bin/env bash\necho "seed failure" >&2\nexit {exit_code}\n'
        )
    seed.chmod(0o755)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add seed-warm-lane.sh stub'], cwd=repo)


async def _commit_file(repo: Path, name: str, content: str, message: str) -> str:
    """Write+commit a file on the repo's current branch; return the new commit SHA."""
    (repo / name).write_text(content)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', message], cwd=repo)
    rc, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0, f'rev-parse HEAD failed after committing {name!r}'
    return sha.strip()


# ---------------------------------------------------------------------------
# Step-01: config knobs
# ---------------------------------------------------------------------------


class TestInteractiveWorktreeConfigDefaults:
    """RED — config knobs: GitConfig + packaged defaults.yaml carry the three new knobs."""

    def test_gitconfig_defaults(self) -> None:
        """GitConfig() exposes max_interactive_worktrees/interactive_worktree_ttl/iact_prefix."""
        config = GitConfig()
        assert config.max_interactive_worktrees == 2, (
            f'expected default max_interactive_worktrees == 2, '
            f'got {config.max_interactive_worktrees!r}'
        )
        assert config.interactive_worktree_ttl == 86400.0, (
            f'expected default interactive_worktree_ttl == 86400.0, '
            f'got {config.interactive_worktree_ttl!r}'
        )
        assert config.iact_prefix == '_iact-', (
            f"expected default iact_prefix == '_iact-', got {config.iact_prefix!r}"
        )

    def test_packaged_defaults_carry_knobs(self) -> None:
        """orchestrator.config._load_defaults()['git'] carries the same three knobs."""
        defaults = orchestrator_config._load_defaults()
        git_defaults = defaults['git']
        assert git_defaults['iact_prefix'] == '_iact-', (
            f"expected packaged defaults.yaml git.iact_prefix == '_iact-', "
            f"got {git_defaults.get('iact_prefix')!r}"
        )
        assert git_defaults['max_interactive_worktrees'] == 2, (
            f"expected packaged defaults.yaml git.max_interactive_worktrees == 2, "
            f"got {git_defaults.get('max_interactive_worktrees')!r}"
        )
        assert git_defaults['interactive_worktree_ttl'] == 86400.0, (
            f"expected packaged defaults.yaml git.interactive_worktree_ttl == 86400.0, "
            f"got {git_defaults.get('interactive_worktree_ttl')!r}"
        )


# ---------------------------------------------------------------------------
# Step-03: primitive core happy path (pool DISABLED)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCreateInteractiveWorktreeHappyPath:
    """RED — create_interactive_worktree core happy path, no WarmLanePool involved."""

    async def test_creates_worktree_on_iact_branch_warm_and_base_ref(
        self, iw_git_repo: Path,
    ) -> None:
        """Fresh create: path/branch/warm/base_ref all match the documented contract."""
        await _add_seed_script(iw_git_repo)
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)

        rc, main_sha, _ = await _run(['git', 'rev-parse', 'main'], cwd=iw_git_repo)
        assert rc == 0, 'setup: rev-parse main failed'
        main_sha = main_sha.strip()

        info = await git_ops.create_interactive_worktree('demo-slug')

        assert isinstance(info, InteractiveWorktreeInfo), (
            f'create_interactive_worktree must return InteractiveWorktreeInfo, '
            f'got {info!r}'
        )
        assert info.path == git_ops.worktree_base / '_iact-demo-slug', (
            f'expected path worktree_base/_iact-demo-slug, got {info.path}'
        )
        assert info.branch == 'task/demo-slug', (
            f"expected branch 'task/demo-slug', got {info.branch!r}"
        )
        assert info.warm is True, f'expected warm is True (seed ran), got {info.warm!r}'
        assert info.base_ref == main_sha, (
            f'expected base_ref == main tip {main_sha!r}, got {info.base_ref!r}'
        )

        # Registered worktree, HEAD on task/demo-slug
        rc_branch, branch_out, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=info.path,
        )
        assert rc_branch == 0 and branch_out.strip() == 'task/demo-slug', (
            f'expected HEAD on task/demo-slug, got {branch_out.strip()!r} (rc={rc_branch})'
        )
        rc_list, wt_list, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'], cwd=iw_git_repo,
        )
        assert rc_list == 0 and str(info.path.resolve()) in wt_list, (
            f'expected {info.path} to be a registered git worktree; '
            f'worktree list: {wt_list!r}'
        )

        # Seed ran: target/seeded.bin present (CoW-populated, orchestration-observable)
        assert (info.path / 'target' / 'seeded.bin').exists(), (
            'expected target/seeded.bin to exist after seed run (CoW-populated)'
        )

    async def test_explicit_start_ref_pins_base_ref(self, iw_git_repo: Path) -> None:
        """A caller-supplied start_ref (a prior commit SHA) pins base_ref exactly."""
        await _add_seed_script(iw_git_repo)
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)

        # Create a distinct prior commit to pin, then advance main further so
        # main's tip != prior_sha (proves the pin — not the fresh tip — was used).
        prior_sha = await _commit_file(
            iw_git_repo, 'second.txt', 'second\n', 'second commit',
        )
        await _commit_file(iw_git_repo, 'third.txt', 'third\n', 'third commit')

        info = await git_ops.create_interactive_worktree(
            'pinned-slug', start_ref=prior_sha,
        )
        assert isinstance(info, InteractiveWorktreeInfo), (
            f'create_interactive_worktree must return InteractiveWorktreeInfo, '
            f'got {info!r}'
        )
        assert info.base_ref == prior_sha, (
            f'expected base_ref == pinned start_ref {prior_sha!r}, got {info.base_ref!r}'
        )


# ---------------------------------------------------------------------------
# Step-05: .task/interactive.json stamp
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInteractiveWorktreeStamp:
    """RED — .task/interactive.json stamp written for the δ reaper to consume."""

    async def test_stamp_written_and_gitignored(self, iw_git_repo: Path) -> None:
        """The stamp records owner/created_at and is never git-tracked."""
        await _add_seed_script(iw_git_repo)
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)

        info = await git_ops.create_interactive_worktree('stamp-slug')

        stamp_path = info.path / '.task' / 'interactive.json'
        assert stamp_path.exists(), f'expected stamp to exist at {stamp_path}'

        stamp = json.loads(stamp_path.read_text())
        assert stamp['owner'] == 'stamp-slug', (
            f"expected stamp['owner'] == 'stamp-slug', got {stamp.get('owner')!r}"
        )
        # Machine-parseable ISO8601 — raises ValueError on failure.
        datetime.fromisoformat(stamp['created_at'])

        # Not git-tracked: ls-files --error-unmatch must fail (non-zero rc),
        # and `git status --porcelain` must not list it (gitignored via
        # .task/.gitignore, written by _ensure_task_gitignore).
        rc_ls, _, _ = await _run(
            ['git', 'ls-files', '--error-unmatch', '.task/interactive.json'],
            cwd=info.path,
        )
        assert rc_ls != 0, (
            'expected .task/interactive.json to NOT be git-tracked '
            '(git ls-files --error-unmatch unexpectedly succeeded)'
        )
        rc_status, status_out, _ = await _run(
            ['git', 'status', '--porcelain'], cwd=info.path,
        )
        assert rc_status == 0 and 'interactive.json' not in status_out, (
            f'expected .task/interactive.json to be absent from git status '
            f'(gitignored); got status: {status_out!r}'
        )


# ---------------------------------------------------------------------------
# Step-07: fail-soft warmth (the key deviation from acquire_warm_lane)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCreateInteractiveWorktreeFailSoftWarmth:
    """RED — a seed fault must be fail-soft: worktree retained, warm=False, no raise."""

    async def test_absent_seed_script_returns_cold_but_usable(
        self, iw_git_repo: Path,
    ) -> None:
        """No scripts/seed-warm-lane.sh committed (rc=127) → cold, usable, stamped."""
        # Deliberately do NOT commit a seed script.
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)

        info = await git_ops.create_interactive_worktree('cold-a')

        assert info.warm is False, (
            f'expected warm is False (no seed script present), got {info.warm!r}'
        )
        assert info.path.exists(), f'expected {info.path} to exist'

        rc_list, wt_list, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'], cwd=iw_git_repo,
        )
        assert rc_list == 0 and str(info.path.resolve()) in wt_list, (
            f'expected {info.path} to be a registered git worktree even though '
            f'cold; worktree list: {wt_list!r}'
        )
        assert (info.path / '.task' / 'interactive.json').exists(), (
            'expected the stamp to be present even though the worktree is cold '
            '(worktree is usable)'
        )

    async def test_seed_script_failure_retains_worktree(self, iw_git_repo: Path) -> None:
        """Seed script exiting 1 → worktree RETAINED (not removed), warm=False, no raise."""
        await _add_seed_script(iw_git_repo, exit_code=1)
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)

        info = await git_ops.create_interactive_worktree('cold-b')

        assert info.warm is False, (
            f'expected warm is False (seed script exited 1), got {info.warm!r}'
        )
        assert info.path.exists(), (
            f'expected {info.path} to exist — fail-soft retention, NOT removed'
        )

        rc_list, wt_list, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'], cwd=iw_git_repo,
        )
        assert rc_list == 0 and str(info.path.resolve()) in wt_list, (
            f'expected {info.path} to remain a REGISTERED git worktree after a '
            f'seed failure (fail-soft retention, not the acquire_warm_lane '
            f'remove-on-failure behaviour); worktree list: {wt_list!r}'
        )


# ---------------------------------------------------------------------------
# Step-09: max_interactive_worktrees cap (REJECT)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCreateInteractiveWorktreeCap:
    """RED — max_interactive_worktrees enforces a REJECT cap before any git op."""

    async def test_cap_rejects_third_worktree_then_frees_on_removal(
        self, iw_git_repo: Path,
    ) -> None:
        """Cap of 2: 'a'/'b' succeed, 'c' is rejected, 'd' succeeds once freed."""
        await _add_seed_script(iw_git_repo)
        config = GitConfig(max_interactive_worktrees=2)
        git_ops = GitOps(config, iw_git_repo)

        info_a = await git_ops.create_interactive_worktree('a')
        info_b = await git_ops.create_interactive_worktree('b')
        assert info_a.path.exists() and info_b.path.exists(), (
            'setup: both a and b should be created within the cap'
        )

        with pytest.raises(InteractiveWorktreeLimitError):
            await git_ops.create_interactive_worktree('c')

        c_path = git_ops.worktree_base / '_iact-c'
        assert not c_path.exists(), (
            f'expected {c_path} to NOT exist after a cap-rejected create'
        )
        rc_branch, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'task/c'], cwd=iw_git_repo,
        )
        assert rc_branch != 0, (
            'expected no task/c branch to have been created by the rejected call'
        )

        # Free a slot by removing one of the existing interactive worktrees.
        rc_remove, _, remove_err = await _run(
            ['git', 'worktree', 'remove', '--force', str(info_a.path)],
            cwd=iw_git_repo,
        )
        assert rc_remove == 0, f'setup: failed to remove {info_a.path}: {remove_err}'

        info_d = await git_ops.create_interactive_worktree('d')
        assert info_d.path.exists(), (
            'expected create_interactive_worktree("d") to succeed once a slot freed'
        )


# ---------------------------------------------------------------------------
# Step-11: I1 isolation (the α headline signal)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCreateInteractiveWorktreeIsolation:
    """RED — I1: create_interactive_worktree must never touch WarmLanePool state."""

    async def test_pool_untouched_by_create_and_remove(self, iw_git_repo: Path) -> None:
        """FREE lane count / assignments_snapshot unchanged by create AND by removal."""
        await _add_seed_script(iw_git_repo)
        config = GitConfig(warm_lane_pool=True)
        git_ops = GitOps(config, iw_git_repo, warm_lane_pool_size=3)
        assert git_ops.warm_lane_pool is not None, 'setup: pool must be enabled'
        pool = git_ops.warm_lane_pool

        def _free_count() -> int:
            return sum(1 for s in pool._lanes.values() if s == LaneState.FREE)

        free_before = _free_count()
        assert free_before == 3, f'setup: expected 3 FREE lanes, got {free_before}'
        assert pool.assignments_snapshot() == {}, 'setup: expected no assignments yet'

        info = await git_ops.create_interactive_worktree('iso')

        assert info.warm is True, f'expected warm is True (seed ran), got {info.warm!r}'
        assert (info.path / 'target' / 'seeded.bin').exists(), (
            'expected target/seeded.bin to exist (CoW-populated)'
        )

        # I1: pool state is completely untouched by create.
        assert _free_count() == 3, (
            f'I1 VIOLATION: create_interactive_worktree changed the FREE lane '
            f'count (before={free_before}, after={_free_count()})'
        )
        assert pool.assignments_snapshot() == {}, (
            'I1 VIOLATION: create_interactive_worktree left an entry in '
            'assignments_snapshot()'
        )
        assert pool.is_lane(info.path) is False, (
            'I1 VIOLATION: the interactive worktree path is registered as a pool lane'
        )
        assert pool.state(info.path) is None, (
            'I1 VIOLATION: the pool reports a LaneState for the interactive '
            'worktree path'
        )
        lane_dirs_after_create = sorted(
            p.name for p in git_ops.worktree_base.iterdir()
            if p.name.startswith('_lane-')
        )
        assert lane_dirs_after_create == [], (
            f'I1 VIOLATION: a _lane-* dir was created on disk: {lane_dirs_after_create}'
        )

        # Remove the interactive worktree directly (no pool-aware release verb
        # exists yet — that is β) and re-assert the pool is still untouched.
        rc_remove, _, remove_err = await _run(
            ['git', 'worktree', 'remove', '--force', str(info.path)],
            cwd=iw_git_repo,
        )
        assert rc_remove == 0, f'setup: failed to remove {info.path}: {remove_err}'

        assert _free_count() == 3, (
            f'I1 VIOLATION: removing the interactive worktree changed the FREE '
            f'lane count (expected 3, got {_free_count()})'
        )
        assert pool.assignments_snapshot() == {}, (
            'I1 VIOLATION: removing the interactive worktree left an '
            'assignments entry behind'
        )


# ---------------------------------------------------------------------------
# Amendment: slug validation (reviewer_comprehensive robustness #2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCreateInteractiveWorktreeSlugValidation:
    """Amendment — slug is validated against a safe charset before any state
    is touched, so a hostile/malformed slug can't escape worktree_base or
    produce an opaque git ref error."""

    @pytest.mark.parametrize(
        'bad_slug',
        [
            'has/slash',
            '../escape',
            'a..b',
            'has space',
            '',
            '.leading-dot',
            '-leading-dash',
            'trailing/../traversal',
        ],
    )
    async def test_invalid_slug_raises_value_error_with_no_side_effects(
        self, iw_git_repo: Path, bad_slug: str,
    ) -> None:
        """An invalid slug raises ValueError and creates nothing on disk or in git."""
        await _add_seed_script(iw_git_repo)
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)

        with pytest.raises(ValueError):
            await git_ops.create_interactive_worktree(bad_slug)

        # No worktree_base side effect at all (validation runs before any I/O).
        assert not git_ops.worktree_base.exists(), (
            f'expected no worktree_base side effects for invalid slug {bad_slug!r}, '
            f'but {git_ops.worktree_base} exists'
        )

    async def test_valid_slug_with_dots_and_dashes_still_works(
        self, iw_git_repo: Path,
    ) -> None:
        """A slug using the allowed charset (letters/digits/./_/-) is accepted."""
        await _add_seed_script(iw_git_repo)
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)

        info = await git_ops.create_interactive_worktree('feature.v2_final-1')
        assert info.path == git_ops.worktree_base / '_iact-feature.v2_final-1', (
            f'expected a valid dotted/dashed slug to be accepted, got {info.path}'
        )


# ---------------------------------------------------------------------------
# Amendment: branch-namespace collision self-heal (reviewer_comprehensive
# robustness #1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCreateInteractiveWorktreeBranchCollision:
    """Amendment — a leftover branch_prefix-namespaced branch for the same
    slug (worktree dir removed, branch not deleted) is cleaned up when
    provably safe, or retained with a RuntimeError when it carries work."""

    async def test_leftover_empty_branch_is_cleaned_up_and_recreate_succeeds(
        self, iw_git_repo: Path,
    ) -> None:
        """A branch with 0 commits beyond main is deleted; recreate succeeds."""
        await _add_seed_script(iw_git_repo)
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)

        info1 = await git_ops.create_interactive_worktree('reuse-me')
        rc_remove, _, remove_err = await _run(
            ['git', 'worktree', 'remove', '--force', str(info1.path)],
            cwd=iw_git_repo,
        )
        assert rc_remove == 0, f'setup: failed to remove {info1.path}: {remove_err}'

        # `git worktree remove` never deletes branches — confirm the
        # collision setup is real before recreating.
        rc_branch, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'task/reuse-me'], cwd=iw_git_repo,
        )
        assert rc_branch == 0, 'setup: expected leftover task/reuse-me branch to exist'

        info2 = await git_ops.create_interactive_worktree('reuse-me')
        assert info2.branch == 'task/reuse-me', (
            f"expected reuse to still land on 'task/reuse-me', got {info2.branch!r}"
        )
        rc_list, wt_list, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'], cwd=iw_git_repo,
        )
        assert rc_list == 0 and str(info2.path.resolve()) in wt_list, (
            f'expected {info2.path} to be a registered worktree after reuse; '
            f'worktree list: {wt_list!r}'
        )

    async def test_leftover_branch_with_commits_is_retained_and_raises(
        self, iw_git_repo: Path,
    ) -> None:
        """A branch carrying commits beyond main is NEVER deleted; create raises."""
        await _add_seed_script(iw_git_repo)
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)

        info1 = await git_ops.create_interactive_worktree('has-work')
        wip_sha = await _commit_file(
            info1.path, 'wip.txt', 'wip\n', 'uncommitted work on the branch',
        )
        rc_remove, _, remove_err = await _run(
            ['git', 'worktree', 'remove', '--force', str(info1.path)],
            cwd=iw_git_repo,
        )
        assert rc_remove == 0, f'setup: failed to remove {info1.path}: {remove_err}'

        with pytest.raises(RuntimeError):
            await git_ops.create_interactive_worktree('has-work')

        # Fail-safe retain: the branch (and its commit) must survive the
        # rejected create — this is the "never destroy work" contract.
        rc_branch, branch_sha, _ = await _run(
            ['git', 'rev-parse', '--verify', 'task/has-work'], cwd=iw_git_repo,
        )
        assert rc_branch == 0 and branch_sha.strip() == wip_sha, (
            f'expected task/has-work to survive the rejected create and still '
            f'point at {wip_sha!r}, got rc={rc_branch}, sha={branch_sha.strip()!r}'
        )


# ---------------------------------------------------------------------------
# Amendment: cap enforcement is TOCTOU-safe under concurrency
# (reviewer_comprehensive concurrency #3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCreateInteractiveWorktreeConcurrency:
    """Amendment — concurrent creates on one GitOps instance cannot both slip
    past the REJECT cap (the count-then-create span is lock-serialized)."""

    async def test_concurrent_creates_do_not_overrun_the_cap(
        self, iw_git_repo: Path,
    ) -> None:
        """Two concurrent create calls at cap=1 yield exactly 1 success, 1 reject."""
        await _add_seed_script(iw_git_repo)
        config = GitConfig(max_interactive_worktrees=1)
        git_ops = GitOps(config, iw_git_repo)

        results = await asyncio.gather(
            git_ops.create_interactive_worktree('race-a'),
            git_ops.create_interactive_worktree('race-b'),
            return_exceptions=True,
        )

        successes = [r for r in results if isinstance(r, InteractiveWorktreeInfo)]
        limit_errors = [
            r for r in results if isinstance(r, InteractiveWorktreeLimitError)
        ]
        assert len(successes) == 1, (
            f'expected exactly 1 success under max_interactive_worktrees=1 '
            f'(TOCTOU guard), got {results!r}'
        )
        assert len(limit_errors) == 1, (
            f'expected exactly 1 InteractiveWorktreeLimitError (TOCTOU guard), '
            f'got {results!r}'
        )

        iact_dirs = [
            p for p in git_ops.worktree_base.iterdir()
            if p.is_dir() and p.name.startswith('_iact-')
        ]
        assert len(iact_dirs) == 1, (
            f'expected exactly 1 _iact-* dir on disk (cap held under race), '
            f'got {[p.name for p in iact_dirs]!r}'
        )
