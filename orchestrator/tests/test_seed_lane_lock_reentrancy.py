"""Regression fence: _seed_warm_lane must not self-refuse against its own lane lock.

reify 5556 (root cause). ``GitOps._seed_warm_lane`` wraps the seed subprocess in
its OWN outer ``flock -x <lane_dir>.lock`` (task 2599).  As of reify commit
7b20d010c6 (task 5354), ``seed-warm-lane.sh`` ALSO acquires that same lock by
default under ``--fresh-checkout`` — previously opt-in via ``--lane-lock``.
flock is not re-entrant across a process tree, so the script's ``flock -n``
self-refused against dark-factory's own held lock and exited 75.

75 is ``_seed_rc_to_unavailable``'s disk-pressure code, so every dispatch
requeued as ``WarmLaneDiskPressure`` with ``agent_invocations=0``, released the
lane, and the next dispatch re-picked the same lowest-index free lane — a
fleet-wide dispatch livelock that ran for ~46h at 349 requeues to 4 completions
per day, with no agent ever starting.

The fix passes reify's ``--assume-lane-lock-held`` opt-out (reify db9ea9387b,
same task) whenever dark-factory holds the outer lock.  Because the seed script
is read from the LANE's own checkout, its vintage varies per lane, so the flag
is capability-probed rather than passed blind: a pre-5354 script would reject
the unknown flag as a usage error and turn a working seed into a hard fault.

The stub scripts below mimic the REAL script's locking contract rather than
asserting on argv, so these tests fail against the pre-fix implementation for
the same reason production did.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run

# Mirrors the real script's lock stage: refuse (75) when ${LANE_DIR}.lock is
# already held, UNLESS the caller asserts it holds the lock itself.
_LOCKING_SEED_SCRIPT = """#!/usr/bin/env bash
set -u
lane_dir="$2"
assume_held=""
for a in "$@"; do
    [ "$a" = "--assume-lane-lock-held" ] && assume_held=1
done
if [ -z "$assume_held" ]; then
    exec 9>"${lane_dir}.lock"
    if ! flock -n 9; then
        echo "Lane lock held by a live consumer (flock -n failed)" >&2
        exit 75
    fi
fi
mkdir -p "$lane_dir/target"
echo seeded > "$lane_dir/target/seeded.bin"
exit 0
"""

# A pre-reify-5354 script: never self-locks, and rejects ANY unrecognised flag
# with exit 2.  Faithful to the real pre-5354 parser, which had a generic
# `-*) err "Unknown flag: $1"; exit 2` arm and did not mention
# --assume-lane-lock-held anywhere (verified against 7b20d010c6^).  It must NOT
# name the flag: the capability probe is a text search, so a script that spells
# the flag out in order to reject it is indistinguishable from one that
# supports it — and no real script does that.
_LEGACY_SEED_SCRIPT = """#!/usr/bin/env bash
set -u
lane_dir="$2"
shift 3
for a in "$@"; do
    case "$a" in
        -*)
            echo "Unknown flag: $a" >&2
            exit 2
            ;;
    esac
done
mkdir -p "$lane_dir/target"
echo seeded > "$lane_dir/target/seeded.bin"
exit 0
"""


# ── task 4211 stubs: reify 5568's --distinct-lock-refusal-rc ──────────────
#
# A post-5568 script that ADVERTISES the flag (the probe is a text search over
# the script body, exactly as for --assume-lane-lock-held) and records the argv
# it was actually handed, so the plumbing can be asserted at the wire.
_ARGV_RECORDING_SEED_SCRIPT = """#!/usr/bin/env bash
# Supported flags include --distinct-lock-refusal-rc (reify task 5568).
set -u
lane_dir="$2"
printf '%s\n' "$@" > "${lane_dir}.argv"
mkdir -p "$lane_dir/target"
echo seeded > "$lane_dir/target/seeded.bin"
exit 0
"""

# A pre-5568 script: records argv, then rejects ANY unrecognised flag with
# exit 2, like the real generic `-*) err "Unknown flag: $1"; exit 2` arm.  As
# with _LEGACY_SEED_SCRIPT above it must NOT spell the flag out anywhere — the
# probe is a text search, so a script naming the flag in order to reject it
# would be indistinguishable from one that supports it.
_LEGACY_ARGV_RECORDING_SEED_SCRIPT = """#!/usr/bin/env bash
set -u
lane_dir="$2"
printf '%s\n' "$@" > "${lane_dir}.argv"
shift 3
for a in "$@"; do
    case "$a" in
        -*)
            echo "Unknown flag: $a" >&2
            exit 2
            ;;
    esac
done
mkdir -p "$lane_dir/target"
echo seeded > "$lane_dir/target/seeded.bin"
exit 0
"""


def _recorded_argv(lane: Path) -> list[str]:
    """The argv the stub seed script was actually handed."""
    return Path(f'{lane}.argv').read_text().split()


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


def _config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=True,
        merge_spec_warm_lane_pool=True,
    )


@pytest.fixture
def seed_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    base = repo / '.worktrees' / '_merge-verify' / 'target'
    base.mkdir(parents=True, exist_ok=True)
    (base / '.keep').write_text('warm base sentinel\n')
    return repo


async def _make_lane(repo: Path, git_ops: GitOps, script_body: str) -> Path:
    """Register a real git worktree lane carrying ``script_body`` as its seed script."""
    _, head, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    lane = git_ops.worktree_base / 'manual-lane'
    rc, _, err = await _run(
        ['git', 'worktree', 'add', '--detach', str(lane), head.strip()], cwd=repo,
    )
    assert rc == 0, f'setup: worktree add failed: {err}'
    scripts_dir = lane / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / 'seed-warm-lane.sh'
    script.write_text(script_body)
    script.chmod(0o755)
    return lane


@pytest.mark.asyncio
class TestSeedLaneLockReentrancy:
    async def test_fresh_checkout_seed_does_not_self_refuse_on_own_lane_lock(
        self, seed_repo: Path,
    ):
        """The reify-5556 livelock: DF's outer lock must not defeat its own seed.

        Pre-fix this returns 75 (disk pressure) and the caller requeues forever.
        """
        git_ops = GitOps(_config(), seed_repo)
        lane = await _make_lane(seed_repo, git_ops, _LOCKING_SEED_SCRIPT)

        rc = await git_ops._seed_warm_lane(lane, '--fresh-checkout')

        assert rc == 0, (
            f'seed must succeed while DF holds the outer lane lock, got rc={rc}. '
            'rc=75 is the reify-5556 self-refusal: DF holds <lane>.lock and the '
            'seed script re-flocks the same file, so every dispatch requeues as '
            'WarmLaneDiskPressure with agent_invocations=0.'
        )
        assert (lane / 'target' / 'seeded.bin').exists(), 'seed did not run'

    async def test_legacy_seed_script_is_not_passed_the_unknown_flag(
        self, seed_repo: Path,
    ):
        """A pre-5354 lane checkout must not be handed a flag it will reject.

        The seed script comes from the LANE's own tree, so its vintage varies.
        Passing the flag blind would turn a working seed into a usage error.
        """
        git_ops = GitOps(_config(), seed_repo)
        lane = await _make_lane(seed_repo, git_ops, _LEGACY_SEED_SCRIPT)

        rc = await git_ops._seed_warm_lane(lane, '--fresh-checkout')

        assert rc == 0, (
            f'a pre-5354 seed script must still seed cleanly, got rc={rc} '
            '(rc=2 means the unknown flag was passed without a capability probe)'
        )
        assert (lane / 'target' / 'seeded.bin').exists(), 'seed did not run'

    async def test_flag_omitted_when_caller_does_not_take_the_lane_lock(
        self, seed_repo: Path,
    ):
        """take_lane_lock=False: the script owns the lock, so it must still take it.

        Callers that already hold the lock some other way pass take_lane_lock=False;
        suppressing the script's own acquire there would drop inv.2 exclusivity
        entirely rather than relocate it.
        """
        git_ops = GitOps(_config(), seed_repo)
        lane = await _make_lane(seed_repo, git_ops, _LOCKING_SEED_SCRIPT)

        # Nobody holds the lock -> the script's own flock -n succeeds.
        rc = await git_ops._seed_warm_lane(
            lane, '--fresh-checkout', take_lane_lock=False,
        )
        assert rc == 0, f'unlocked seed should succeed, got rc={rc}'

        # With the lock genuinely held by a foreign consumer, the script MUST
        # still refuse — proving the flag was not passed on this path.
        lock_path = Path(f'{lane}.lock')
        lock_path.touch()
        proc = await asyncio.create_subprocess_exec(
            'flock', '-x', str(lock_path), 'sleep', '10',
        )
        try:
            await asyncio.sleep(0.5)
            rc_locked = await git_ops._seed_warm_lane(
                lane, '--fresh-checkout', take_lane_lock=False,
            )
        finally:
            proc.kill()
            await proc.wait()

        assert rc_locked == 75, (
            'with take_lane_lock=False and a foreign holder, the seed script must '
            f'still self-refuse (75) — got rc={rc_locked}, meaning inv.2 '
            'single-consumer exclusivity was silently dropped'
        )


# ===========================================================================
# Task 4211: --distinct-lock-refusal-rc capability probe + flag plumbing
# ===========================================================================
#
# reify's seed-warm-lane.sh exits 75 from BOTH its lane-lock refusal arms
# (flock -n immediate refusal; flock -w queue timeout) and has no
# disk-pressure exit-75 path at all, so DF rendered every lock refusal as
# WarmLaneDiskPressure.  reify task 5568 added the OPT-IN
# --distinct-lock-refusal-rc flag, under which those arms exit 77 instead.
# DF must pass it — capability-probed, for the same per-lane-vintage reason
# --assume-lane-lock-held is probed.


class TestDistinctLockRefusalRcProbe:
    """_seed_script_supports_distinct_lock_refusal_rc — text-grep, fail CLOSED."""

    def _probe(self):
        from orchestrator.git_ops import (
            _seed_script_supports_distinct_lock_refusal_rc,
        )
        # lru_cache'd per resolved path; clear so a prior test's answer for a
        # same-named path can never leak in.
        _seed_script_supports_distinct_lock_refusal_rc.cache_clear()
        return _seed_script_supports_distinct_lock_refusal_rc

    def test_true_when_the_script_advertises_the_flag(self, tmp_path: Path):
        script = tmp_path / 'seed-warm-lane.sh'
        script.write_text(_ARGV_RECORDING_SEED_SCRIPT)
        assert self._probe()(script) is True

    def test_false_for_a_pre_5568_script(self, tmp_path: Path):
        script = tmp_path / 'seed-warm-lane.sh'
        script.write_text(_LEGACY_ARGV_RECORDING_SEED_SCRIPT)
        assert self._probe()(script) is False

    def test_false_when_the_script_is_absent(self, tmp_path: Path):
        """Fails CLOSED on a read error, mirroring the sibling probe's OSError arm.

        A false negative simply omits the flag, so the script exits 75 and the
        condition surfaces as DISK_PRESSURE — byte-identical to the behaviour
        before this fix existed, which is the safe degradation.
        """
        assert self._probe()(tmp_path / 'does-not-exist.sh') is False

    def test_false_when_the_path_is_a_directory(self, tmp_path: Path):
        """IsADirectoryError is an OSError — the same fail-closed arm."""
        d = tmp_path / 'seed-warm-lane.sh'
        d.mkdir()
        assert self._probe()(d) is False


@pytest.mark.asyncio
class TestDistinctLockRefusalRcPlumbing:
    async def test_flag_is_passed_when_the_script_advertises_it(
        self, seed_repo: Path,
    ):
        git_ops = GitOps(_config(), seed_repo)
        lane = await _make_lane(seed_repo, git_ops, _ARGV_RECORDING_SEED_SCRIPT)

        rc = await git_ops._seed_warm_lane(lane, '--fresh-checkout')

        assert rc == 0, f'seed must succeed, got rc={rc}'
        assert '--distinct-lock-refusal-rc' in _recorded_argv(lane), (
            'DF must opt in to reify 5568 rc-77 disambiguation, otherwise a '
            'lane-lock refusal keeps arriving as 75 and rendering as disk '
            'pressure'
        )

    async def test_flag_is_not_passed_to_a_pre_5568_script(
        self, seed_repo: Path,
    ):
        """A legacy lane must keep seeding cleanly — the flag is a usage error there."""
        git_ops = GitOps(_config(), seed_repo)
        lane = await _make_lane(
            seed_repo, git_ops, _LEGACY_ARGV_RECORDING_SEED_SCRIPT,
        )

        rc = await git_ops._seed_warm_lane(lane, '--fresh-checkout')

        assert rc == 0, (
            f'a pre-5568 seed script must still seed cleanly, got rc={rc} '
            '(rc=2 means the unknown flag was passed without a capability probe)'
        )
        assert '--distinct-lock-refusal-rc' not in _recorded_argv(lane)

    async def test_flag_is_passed_regardless_of_take_lane_lock(
        self, seed_repo: Path,
    ):
        """The gating fence — deliberately UNLIKE --assume-lane-lock-held.

        That flag is gated on ``take_lane_lock`` (it only matters when DF holds
        the outer lock), which
        test_flag_omitted_when_caller_does_not_take_the_lane_lock pins.  This
        one must NOT be, because seed's refusal arms are reachable precisely
        when the SCRIPT self-locks: take_lane_lock=False (the ephemeral-worktree
        CM caller, which holds the lock itself), and take_lane_lock=True against
        a pre-5354 script that ignores --assume-lane-lock-held — the original
        esc-5556-1 self-refusal shape.  Gating it would make it inert in exactly
        the cases it exists for.
        """
        git_ops = GitOps(_config(), seed_repo)
        lane = await _make_lane(seed_repo, git_ops, _ARGV_RECORDING_SEED_SCRIPT)

        rc_held = await git_ops._seed_warm_lane(
            lane, '--fresh-checkout', take_lane_lock=True,
        )
        assert rc_held == 0, f'take_lane_lock=True seed failed: rc={rc_held}'
        assert '--distinct-lock-refusal-rc' in _recorded_argv(lane)

        rc_free = await git_ops._seed_warm_lane(
            lane, '--fresh-checkout', take_lane_lock=False,
        )
        assert rc_free == 0, f'take_lane_lock=False seed failed: rc={rc_free}'
        assert '--distinct-lock-refusal-rc' in _recorded_argv(lane), (
            'the flag must survive take_lane_lock=False — that is the caller '
            'shape where the script self-locks and can therefore refuse'
        )
