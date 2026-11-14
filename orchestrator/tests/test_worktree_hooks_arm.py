"""A worktree's own hooks-armed-guard.sh ``arm`` run, classified by its exit contract."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest
from _git_fixtures import seed_repo
from _lane_test_helpers import make_warm_lane_repo
from _orch_helpers import assert_isolated_git_repo
from _seed_script_stubs import commit_seed_script, warm_pool_git_config
from shared.git_async import run_git

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps
from orchestrator.worktree_hooks_arm import HooksArmOutcome, arm_worktree_hooks

_ARM_LOGGER = 'orchestrator.worktree_hooks_arm'
_GUARD_STDERR = 'hooks-armed-guard: stub stderr marker'


def _write_guard(wt: Path, body: str, mode: int = 0o755) -> Path:
    guard = wt / 'scripts' / 'hooks-armed-guard.sh'
    guard.parent.mkdir(parents=True, exist_ok=True)
    guard.write_text(body)
    guard.chmod(mode)
    return guard


def _recording_guard_body(log: Path, rc: int) -> str:
    return (
        '#!/usr/bin/env bash\n'
        f"printf '%s\\n' \"$*\" >> '{log}'\n"
        f"echo '[warn] {_GUARD_STDERR}' >&2\n"
        f'exit {rc}\n'
    )


def _arm_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage() for record in caplog.records
        if record.name == _ARM_LOGGER and record.levelno == logging.WARNING
    ]


@pytest.fixture
def wt(tmp_path: Path) -> Path:
    worktree = tmp_path / 'wt'
    worktree.mkdir()
    return worktree


@pytest.fixture
def log(tmp_path: Path) -> Path:
    return tmp_path / 'arm-calls.log'


@pytest.mark.asyncio
class TestArmWorktreeHooks:
    async def test_rc0_is_armed_and_passes_arm_and_the_worktree(
        self, wt: Path, log: Path, caplog: pytest.LogCaptureFixture,
    ):
        _write_guard(wt, _recording_guard_body(log, 0))

        with caplog.at_level(logging.DEBUG, logger=_ARM_LOGGER):
            outcome = await arm_worktree_hooks(wt)

        assert outcome is HooksArmOutcome.ARMED
        assert log.read_text().splitlines() == [f'arm {wt}']
        assert _arm_warnings(caplog) == []

    async def test_rc2_is_pin_cannot_fix_and_warns_with_stderr(
        self, wt: Path, log: Path, caplog: pytest.LogCaptureFixture,
    ):
        _write_guard(wt, _recording_guard_body(log, 2))

        with caplog.at_level(logging.DEBUG, logger=_ARM_LOGGER):
            outcome = await arm_worktree_hooks(wt)

        assert outcome is HooksArmOutcome.PIN_CANNOT_FIX
        warnings = _arm_warnings(caplog)
        assert len(warnings) == 1
        assert str(wt) in warnings[0]
        assert _GUARD_STDERR in warnings[0]

    @pytest.mark.parametrize('rc', [1, 7])
    async def test_any_other_nonzero_rc_is_failed_and_warns_with_rc_and_stderr(
        self, wt: Path, log: Path, caplog: pytest.LogCaptureFixture, rc: int,
    ):
        _write_guard(wt, _recording_guard_body(log, rc))

        with caplog.at_level(logging.DEBUG, logger=_ARM_LOGGER):
            outcome = await arm_worktree_hooks(wt)

        assert outcome is HooksArmOutcome.FAILED
        warnings = _arm_warnings(caplog)
        assert len(warnings) == 1
        assert f'rc={rc}' in warnings[0]
        assert str(wt) in warnings[0]
        assert _GUARD_STDERR in warnings[0]

    async def test_worktree_without_the_guard_is_a_silent_no_op(
        self, wt: Path, log: Path, caplog: pytest.LogCaptureFixture,
    ):
        with caplog.at_level(logging.DEBUG, logger=_ARM_LOGGER):
            outcome = await arm_worktree_hooks(wt)

        assert outcome is HooksArmOutcome.NO_SCRIPT
        assert _arm_warnings(caplog) == []
        assert not log.exists()

    async def test_nonexistent_worktree_is_a_silent_no_op(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        with caplog.at_level(logging.DEBUG, logger=_ARM_LOGGER):
            outcome = await arm_worktree_hooks(tmp_path / 'no-such-worktree')

        assert outcome is HooksArmOutcome.NO_SCRIPT
        assert _arm_warnings(caplog) == []

    async def test_non_executable_guard_warns_and_spawns_nothing(
        self, wt: Path, log: Path, caplog: pytest.LogCaptureFixture,
    ):
        guard = _write_guard(wt, _recording_guard_body(log, 0), mode=0o644)

        with caplog.at_level(logging.DEBUG, logger=_ARM_LOGGER):
            outcome = await arm_worktree_hooks(wt)

        assert outcome is HooksArmOutcome.NOT_EXECUTABLE
        warnings = _arm_warnings(caplog)
        assert len(warnings) == 1
        assert str(guard) in warnings[0]
        assert not log.exists()

    async def test_unexecutable_guard_without_shebang_is_failed_not_raised(
        self, wt: Path, log: Path, caplog: pytest.LogCaptureFixture,
    ):
        _write_guard(wt, f"printf '%s\\n' \"$*\" >> '{log}'\n")

        with caplog.at_level(logging.DEBUG, logger=_ARM_LOGGER):
            outcome = await arm_worktree_hooks(wt)

        assert outcome is HooksArmOutcome.FAILED
        assert len(_arm_warnings(caplog)) == 1
        assert not log.exists()

    async def test_wedged_guard_is_timed_out_and_warns_with_the_timeout(
        self, wt: Path, caplog: pytest.LogCaptureFixture,
    ):
        _write_guard(wt, '#!/usr/bin/env bash\nsleep 30\n')

        started = time.monotonic()
        with caplog.at_level(logging.DEBUG, logger=_ARM_LOGGER):
            outcome = await arm_worktree_hooks(wt, timeout_secs=0.5)
        elapsed = time.monotonic() - started

        assert outcome is HooksArmOutcome.TIMED_OUT
        assert elapsed < 10
        warnings = _arm_warnings(caplog)
        assert len(warnings) == 1
        assert str(wt) in warnings[0]
        assert 'timed out after 0.5s' in warnings[0]


_PLAIN_SEED_SCRIPT = '#!/usr/bin/env bash\nmkdir -p "$2/target"\nexit 0\n'


def _cold_git_ops(repo: Path) -> GitOps:
    config = GitConfig(
        main_branch='main', branch_prefix='task/', remote='origin',
        worktree_dir='.worktrees', push_after_advance=False,
    )
    return GitOps(config, repo)


def _logged_calls(log: Path) -> list[str]:
    return log.read_text().splitlines() if log.exists() else []


async def _git(repo: Path, *args: str) -> None:
    result = await run_git(['git', *args], cwd=repo)
    assert result.ok, f'git {args} failed: {result.stderr}'


async def _commit_guard_stub(repo: Path, log: Path, rc: int) -> None:
    assert_isolated_git_repo(repo)
    _write_guard(repo, _recording_guard_body(log, rc))
    await _git(repo, 'add', 'scripts/hooks-armed-guard.sh')
    await _git(repo, 'commit', '-m', 'add hooks-armed-guard stub')


@pytest.mark.asyncio
class TestWorktreeHandoffArmsHooks:
    async def test_cold_fresh_create_arms_the_new_worktree(
        self, tmp_path: Path, log: Path,
    ):
        repo = seed_repo(tmp_path / 'repo')
        await _commit_guard_stub(repo, log, 0)

        info = await _cold_git_ops(repo).create_worktree('t1')

        assert _logged_calls(log) == [f'arm {info.path}']

    async def test_cold_reuse_by_path_arms_again(self, tmp_path: Path, log: Path):
        repo = seed_repo(tmp_path / 'repo')
        await _commit_guard_stub(repo, log, 0)
        git_ops = _cold_git_ops(repo)

        first = await git_ops.create_worktree('t1')
        again = await git_ops.create_worktree('t1')

        assert again.path == first.path
        assert _logged_calls(log) == [f'arm {again.path}', f'arm {again.path}']

    async def test_failed_arm_still_hands_the_worktree_over(
        self, tmp_path: Path, log: Path, caplog: pytest.LogCaptureFixture,
    ):
        repo = seed_repo(tmp_path / 'repo')
        await _commit_guard_stub(repo, log, 1)

        with caplog.at_level(logging.DEBUG, logger=_ARM_LOGGER):
            info = await _cold_git_ops(repo).create_worktree('t2')

        assert info.path.is_dir()
        warnings = _arm_warnings(caplog)
        assert len(warnings) == 1
        assert str(info.path) in warnings[0]
        assert 'rc=1' in warnings[0]

    async def test_warm_live_requeue_arms_but_the_seeded_acquire_leaves_it_to_the_seed(
        self, tmp_path: Path, log: Path,
    ):
        repo = make_warm_lane_repo(tmp_path)
        _write_guard(repo, _recording_guard_body(log, 0))
        await commit_seed_script(repo, _PLAIN_SEED_SCRIPT)
        git_ops = GitOps(warm_pool_git_config(), repo, warm_lane_pool_size=1)

        first = await git_ops.create_worktree('A')
        assert _logged_calls(log) == []

        again = await git_ops.create_worktree('A')

        assert again.path == first.path
        assert _logged_calls(log) == [f'arm {again.path}']
