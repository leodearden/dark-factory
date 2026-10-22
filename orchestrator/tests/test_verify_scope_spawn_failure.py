"""A verify leg whose systemd-run scope never started is host infra, not a branch fault.

Pinned from the classifier, through the public ``run_verification`` spawn path,
to the merge lane's bounded retry and transient-infra hold.

Fixtures are kept MODULE-LOCAL (not conftest.py): a conftest.py edit widens the
merge-time verify (see test_verify_scope_cpu_weight.py's docstring).
"""

from __future__ import annotations

import os
import shlex
import stat
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _merge_lane_fakes import FakeVerifier, VerifyScript

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.merge_lane.worker import (
    TRANSIENT_INFRA_REASON_PREFIX,
    _DryRunInvestigationHandles,
    _run_post_merge_verify,
)
from orchestrator.verify import VerifyResult, run_verification
from orchestrator.verify_categories import INFRA_TRANSIENT_CATEGORIES, FailureCategory
from orchestrator.verify_classify import classify_failure
from orchestrator.verify_cmd import ToolKind

# Byte-identical to data/verify-logs/7217/attempt-1.cockpit.{test,lint,type}-20261020T090729_536074Z.log (rc=1, ~0.19s).
_SCOPE_START_REFUSED = 'Failed to start transient scope unit: Transport endpoint is not connected\n'
# Measured 2026-10-22, systemd 255.4-1ubuntu8.18, `systemd-run --user --scope --quiet --collect` with an unreachable user bus; rc=1, entire output.
_BUS_CONNECT_REFUSED = 'Failed to connect to bus: No such file or directory\n'

_SAMPLES = [
    pytest.param(_SCOPE_START_REFUSED, id='scope-start-refused'),
    pytest.param(_BUS_CONNECT_REFUSED, id='bus-connect-refused'),
]


class TestClassifiesScopeSpawnFailureAsEnvTransient:

    @pytest.mark.parametrize('tool', list(ToolKind))
    @pytest.mark.parametrize('sample', _SAMPLES)
    def test_sole_refusal_is_env_transient_for_every_tool(self, sample, tool):
        result = classify_failure(tool, 1, sample, False)
        assert result == FailureCategory.ENV_TRANSIENT
        assert result in INFRA_TRANSIENT_CATEGORIES

    @pytest.mark.parametrize('sample', _SAMPLES)
    def test_refusal_without_trailing_newline_is_env_transient(self, sample):
        assert classify_failure(
            ToolKind.PYTEST, 1, sample.rstrip('\n'), False,
        ) == FailureCategory.ENV_TRANSIENT


class TestCommandThatRanKeepsItsOwnVerdict:

    def test_refusal_followed_by_test_verdict_is_test_failure(self):
        output = _SCOPE_START_REFUSED + 'FAILED tests/test_x.py::test_y - AssertionError\n'
        assert classify_failure(ToolKind.PYTEST, 1, output, False) == FailureCategory.TEST_FAILURE

    def test_refusal_after_real_output_is_not_infra(self):
        output = (
            '============================= test session starts ==============================\n'
            + _SCOPE_START_REFUSED
        )
        assert classify_failure(
            ToolKind.PYTEST, 1, output, False,
        ) == FailureCategory.UNKNOWN_TEST_FAILURE

    def test_refusal_quoted_in_assertion_diff_is_not_infra(self):
        output = 'E       ' + _SCOPE_START_REFUSED
        assert classify_failure(
            ToolKind.PYTEST, 1, output, False,
        ) == FailureCategory.UNKNOWN_TEST_FAILURE

    def test_bare_errno_text_is_not_infra(self):
        output = 'OSError: [Errno 107] Transport endpoint is not connected\n'
        assert classify_failure(
            ToolKind.OPAQUE, 1, output, False,
        ) == FailureCategory.UNKNOWN_TEST_FAILURE


@pytest.fixture(params=_SAMPLES)
def unreachable_bus(request, tmp_path, monkeypatch) -> str:
    """Put a ``systemd-run`` first on PATH that refuses, as with no user bus.

    It prints the grounded refusal to stderr and exits 1 without exec'ing its
    payload, so the real ``shutil.which`` + ``create_subprocess_exec`` spawn
    path produces the production leg output. Returns the refusal printed.
    """
    refusal: str = request.param
    fakebin = tmp_path / 'fakebin'
    fakebin.mkdir()
    script = fakebin / 'systemd-run'
    script.write_text(
        '#!/bin/sh\n'
        f"printf '%s' {shlex.quote(refusal)} >&2\n"
        'exit 1\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv('PATH', f'{fakebin}{os.pathsep}' + os.environ['PATH'])
    return refusal


@pytest.fixture
def worktree(tmp_path) -> Path:
    wt = tmp_path / 'wt'
    wt.mkdir()
    return wt


def _cockpit_module_config() -> ModuleConfig:
    return ModuleConfig(
        prefix='cockpit',
        test_command='uv run --directory cockpit pytest tests/ --tb=short -q',
        lint_command='python3 scripts/lint_cockpit.py',
        type_check_command='uv run --directory cockpit pyright src/ tests/',
    )


async def _verify_with_unreachable_bus(worktree: Path) -> VerifyResult:
    return await run_verification(
        worktree,
        OrchestratorConfig(verify_use_cgroup_scope=True),
        module_config=_cockpit_module_config(),
        max_retries=0,
        role='merge',
    )


@pytest.mark.usefixtures('code_default_config', 'unreachable_bus')
class TestScopedVerifyWithUnreachableBus:

    @pytest.mark.asyncio
    async def test_every_leg_refused_is_env_transient(self, worktree):
        result = await _verify_with_unreachable_bus(worktree)
        assert result.passed is False
        assert result.category == FailureCategory.ENV_TRANSIENT, result.summary


def _make_git_ops() -> MagicMock:
    git_ops = MagicMock()
    git_ops.cleanup_merge_worktree = AsyncMock()
    git_ops.prune_stale_merge_worktrees = AsyncMock(return_value=[])
    return git_ops


def _make_req() -> MagicMock:
    req = MagicMock()
    req.task_id = 'task-scope-spawn'
    req.task_files = None
    req.module_configs = []
    req.config.merge_verify_min_free_disk_bytes = 1024
    req.config.merge_verify_workspace = False
    return req


@pytest.mark.usefixtures('code_default_config', 'unreachable_bus')
class TestMergeLaneHoldsScopeSpawnFailure:

    @pytest.mark.asyncio
    async def test_persistent_refusal_holds_as_transient_infra(self, worktree):
        spawn_failed = await _verify_with_unreachable_bus(worktree)
        verifier = FakeVerifier(default=VerifyScript(result=spawn_failed))
        timeouts: dict[str, int] = {}

        outcome = await _run_post_merge_verify(
            _make_git_ops(), _make_req(), MagicMock(),
            timeouts=timeouts, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            dry_run_handles=_DryRunInvestigationHandles(scheduler=MagicMock()),
            verifier=verifier,
        )

        assert outcome is not None
        assert outcome.status == 'blocked'
        assert outcome.reason.startswith(TRANSIENT_INFRA_REASON_PREFIX), outcome.reason
        assert len(verifier.verified) == 2
        assert verifier.investigations == []
        assert timeouts == {}
        assert outcome.failure_category == ''

    @pytest.mark.asyncio
    async def test_refusal_that_clears_on_retry_lets_the_merge_proceed(self, worktree):
        spawn_failed = await _verify_with_unreachable_bus(worktree)
        verifier = FakeVerifier(sequence=[VerifyScript(result=spawn_failed)])

        outcome = await _run_post_merge_verify(
            _make_git_ops(), _make_req(), MagicMock(),
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            verifier=verifier,
        )

        assert outcome is None
        assert len(verifier.verified) == 2
