"""The env-recovery retry gate reads the TEST leg's own classification.

``run_verification``'s bounded env-recovery retry re-runs only the test
command, serially, to recover from a shared-venv mutation. Guard 3 makes
ENV_TRANSIENT reachable from a type leg too (task 3367's mis-resolved pyright
interpreter), so a gate reading the run-wide worst-of-legs category fired a
pointless serial re-run over a genuine test failure, and the merge lane held
the branch as transient infra (task 4843, #4103).

Mirrors ``test_verify_env_transient.TestRunVerificationEnvRecovery``'s seam:
``_run_cmd`` is patched, and every invoked command is recorded so the gate's
own decision is asserted, not only the aggregate it feeds.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator import verify
from orchestrator.config import OrchestratorConfig
from orchestrator.verify_categories import INFRA_TRANSIENT_CATEGORIES

_TEST_FAILURE_OUTPUT = (
    'FAILED tests/test_x.py::test_y - assert 1 == 2\n'
    '1 failed, 3 passed in 0.50s\n'
)

_XDIST_VANISHED_OUTPUT = (
    'usage: pytest [options] [file_or_dir] [file_or_dir] [...]\n'
    'pytest: error: unrecognized arguments: -n --dist --max-worker-restart=0\n'
)

# Task 3367's mis-resolved-interpreter signature, in pyright's real line
# shape: a false guard-3 ENV_TRANSIENT on the type leg.
_PYRIGHT_INTERPRETER_MISRESOLVED_OUTPUT = '\n'.join(
    f'  /repo/src/fused_memory/server.py:{line}:8 - error: Import "{module}" '
    'could not be resolved (reportMissingImports)'
    for line, module in enumerate(
        ('pytest', 'pydantic', 'aiosqlite', 'openai', 'qdrant_client'), start=8,
    )
)

_PYRIGHT_TYPE_ERROR_OUTPUT = (
    '/repo/fused-memory/src/x.py:1:1 - error: "foo" is not defined '
    '(reportUndefinedVariable)\n'
    '1 error, 0 warnings, 0 informations \n'
)

_GREEN = (0, '', False)


def _config(project_root: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=project_root,
        test_command='uv run pytest tests/',
        lint_command='echo lint',
        type_check_command='uv run --directory fused-memory pyright src/',
    )


def _is_serial_recovery(cmd: str) -> bool:
    return '-o addopts=' in cmd


async def _verify(
    tmp_path: Path, *, first_test, recovery_test=_GREEN, type_leg=_GREEN,
):
    """Run verification over scripted legs; return (result, pytest cmds)."""
    invoked: list[str] = []

    async def fake_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
        invoked.append(cmd)
        if 'pyright' in cmd:
            return type_leg
        if 'pytest' in cmd:
            return recovery_test if _is_serial_recovery(cmd) else first_test
        return _GREEN

    with patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
        result = await verify.run_verification(tmp_path, _config(tmp_path))
    return result, [c for c in invoked if 'pytest' in c]


class TestEnvRecoveryGateReadsTheTestLeg:

    @pytest.mark.asyncio
    async def test_test_failure_beside_type_env_transient_is_not_rerun(self, tmp_path):
        """The acceptance case: a genuine test failure is reported as one,
        with no serial re-run and no transient-infra hold."""
        result, pytest_cmds = await _verify(
            tmp_path,
            first_test=(1, _TEST_FAILURE_OUTPUT, False),
            type_leg=(1, _PYRIGHT_INTERPRETER_MISRESOLVED_OUTPUT, False),
        )

        assert len(pytest_cmds) == 1, pytest_cmds
        assert not any(_is_serial_recovery(c) for c in pytest_cmds), pytest_cmds
        assert result.category == 'test_failure'
        assert result.category not in INFRA_TRANSIENT_CATEGORIES, (
            'the merge lane would hold this branch as transient infra'
        )
        assert result.failing_leg_categories == ['test_failure', 'env_transient']

    @pytest.mark.asyncio
    async def test_test_leg_env_transient_still_recovers(self, tmp_path):
        """Positive control: a vanished-xdist test leg gets its serial
        re-run, and the re-run's green is the verdict."""
        result, pytest_cmds = await _verify(
            tmp_path, first_test=(4, _XDIST_VANISHED_OUTPUT, False),
        )

        assert len(pytest_cmds) == 2, pytest_cmds
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_test_leg_env_transient_recovers_beside_type_env_transient(
        self, tmp_path,
    ):
        """The gate reads the test leg, not 'any leg': a false type-leg
        ENV_TRANSIENT neither causes nor suppresses the re-run."""
        _, pytest_cmds = await _verify(
            tmp_path,
            first_test=(4, _XDIST_VANISHED_OUTPUT, False),
            type_leg=(1, _PYRIGHT_INTERPRETER_MISRESOLVED_OUTPUT, False),
        )

        assert len(pytest_cmds) == 2, pytest_cmds

    @pytest.mark.asyncio
    async def test_test_leg_env_transient_recovers_beside_a_real_type_error(
        self, tmp_path,
    ):
        """The gate reads the test leg, not the run-wide worst: a real type
        error outranks ENV_TRANSIENT in the aggregate, and must not cost the
        test leg its recovery."""
        result, pytest_cmds = await _verify(
            tmp_path,
            first_test=(4, _XDIST_VANISHED_OUTPUT, False),
            type_leg=(1, _PYRIGHT_TYPE_ERROR_OUTPUT, False),
        )

        assert len(pytest_cmds) == 2, pytest_cmds
        assert result.category == 'type_error'
        assert result.failing_leg_categories == ['type_error']

    @pytest.mark.asyncio
    async def test_green_test_leg_is_never_rerun(self, tmp_path):
        """Task 3367's esc-3359-1 case: re-running a passing test leg can
        never recover anything."""
        _, pytest_cmds = await _verify(
            tmp_path,
            first_test=_GREEN,
            type_leg=(1, _PYRIGHT_INTERPRETER_MISRESOLVED_OUTPUT, False),
        )

        assert len(pytest_cmds) == 1, pytest_cmds
