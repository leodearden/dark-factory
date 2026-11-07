"""A worktree's own hooks-armed-guard.sh ``arm`` run, classified by its exit contract."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

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

    async def test_wedged_guard_times_out_as_failed(
        self, wt: Path, caplog: pytest.LogCaptureFixture,
    ):
        _write_guard(wt, '#!/usr/bin/env bash\nsleep 30\n')

        started = time.monotonic()
        with caplog.at_level(logging.DEBUG, logger=_ARM_LOGGER):
            outcome = await arm_worktree_hooks(wt, timeout_secs=0.5)
        elapsed = time.monotonic() - started

        assert outcome is HooksArmOutcome.FAILED
        assert elapsed < 10
        warnings = _arm_warnings(caplog)
        assert len(warnings) == 1
        assert 'timed out' in warnings[0]
