"""Real-git fixtures plus a stub of reify's ``seed-warm-lane.sh`` locking contract.

Shared by the lane-lock re-entrancy suite and the main-tip-sweep seed-mode
suite, so the one model of how today's seed script takes, refuses and accepts
an asserted ``<lane_dir>.lock`` lives in one place.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from orchestrator.config import GitConfig
from orchestrator.git_ops import _run

# Models TODAY's reify script (post-5354 + post-5568): it takes
# ${LANE_DIR}.lock itself unless told the caller already holds it, and refuses
# with 77 under --distinct-lock-refusal-rc.  Stricter than the real script, it
# VERIFIES an --assume-lane-lock-held assertion (exit 3 if the lock is in fact
# free), so a caller asserting a lock it does not hold fails loudly instead of
# silently dropping inv.2 exclusivity.
CURRENT_LOCKING_SEED_SCRIPT = """#!/usr/bin/env bash
# Supported flags include --assume-lane-lock-held and --distinct-lock-refusal-rc.
set -u
lane_dir="$2"
refusal_rc=75
assume_held=""
for a in "$@"; do
    case "$a" in
        --distinct-lock-refusal-rc) refusal_rc=77 ;;
        --assume-lane-lock-held) assume_held=1 ;;
    esac
done
if [ -z "$assume_held" ]; then
    exec 9>"${lane_dir}.lock"
    if ! flock -n 9; then
        echo "LANE_LOCK_CONTENDED: ${lane_dir}.lock held by a live consumer" >&2
        exit "$refusal_rc"
    fi
elif flock -n "${lane_dir}.lock" true; then
    echo "asserted lane lock ${lane_dir}.lock is not actually held" >&2
    exit 3
fi
mkdir -p "$lane_dir/target"
echo seeded > "$lane_dir/target/seeded.bin"
exit 0
"""


async def init_git_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


def warm_pool_git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=True,
        merge_spec_warm_lane_pool=True,
    )


def make_seed_test_repo(tmp_path: Path, *, warm_base: bool = True) -> Path:
    """A committed repo under ``tmp_path``, optionally with a resolvable warm base."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(init_git_repo(repo))
    if warm_base:
        base = repo / '.worktrees' / '_merge-verify' / 'target'
        base.mkdir(parents=True, exist_ok=True)
        (base / '.keep').write_text('warm base sentinel\n')
    return repo


async def commit_seed_script(repo: Path, script_body: str) -> None:
    """Commit ``script_body`` as the repo's seed script so POOL lanes carry it.

    Unlike a manually-registered lane (which gets the script written into it),
    ``acquire_warm_lane`` creates its own ``_lane-N`` worktrees, so the script
    has to be in the committed tree for the lane checkout to pick it up.
    """
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    seed = scripts_dir / 'seed-warm-lane.sh'
    seed.write_text(script_body)
    seed.chmod(0o755)
    debug = scripts_dir / 'setup-worktree-debug-port.sh'
    debug.write_text('#!/usr/bin/env bash\necho 39411\n')
    debug.chmod(0o755)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add seed + debug-port scripts'], cwd=repo)
