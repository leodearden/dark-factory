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
from _fake_systemd_run import fake_systemd_run as fake_systemd_run
from _merge_lane_fakes import FakeVerifier, VerifyScript

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.merge_lane.worker import (
    TRANSIENT_INFRA_REASON_PREFIX,
    _DryRunInvestigationHandles,
    _run_post_merge_verify,
)
from orchestrator.verify import CheckRun, LegSlot, VerifyResult, classify_leg, run_verification
from orchestrator.verify_categories import INFRA_TRANSIENT_CATEGORIES, FailureCategory
from orchestrator.verify_classify import classify_failure
from orchestrator.verify_cmd import ToolKind

# Byte-identical to data/verify-logs/7217/attempt-1.cockpit.{test,lint,type}-20261020T090729_536074Z.log (rc=1, ~0.19s).
_SCOPE_START_REFUSED = 'Failed to start transient scope unit: Transport endpoint is not connected\n'
# Measured 2026-10-22, systemd 255.4-1ubuntu8.18, `systemd-run --user --scope --quiet --collect` with an unreachable user bus; rc=1, entire output.
_BUS_CONNECT_REFUSED = 'Failed to connect to bus: No such file or directory\n'

# Never observed from a verify launch: no text rule could have named it in advance.
_UNOBSERVED_REFUSAL = 'Failed to connect to bus: Permission denied\n'

_SAMPLES = [
    pytest.param(_SCOPE_START_REFUSED, id='scope-start-refused'),
    pytest.param(_BUS_CONNECT_REFUSED, id='bus-connect-refused'),
]
_ALL_REFUSALS = [*_SAMPLES, pytest.param(_UNOBSERVED_REFUSAL, id='unobserved-reason')]


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

    def test_scope_unit_refusal_is_env_transient_whatever_the_reason(self):
        output = 'Failed to start transient scope unit: Connection reset by peer\n'
        assert classify_failure(ToolKind.OPAQUE, 1, output, False) == FailureCategory.ENV_TRANSIENT


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

    def test_bus_connect_refusal_with_unobserved_reason_is_not_infra(self):
        output = 'Failed to connect to bus: Permission denied\n'
        assert classify_failure(
            ToolKind.OPAQUE, 1, output, False,
        ) == FailureCategory.UNKNOWN_TEST_FAILURE

    def test_bare_errno_text_is_not_infra(self):
        output = 'OSError: [Errno 107] Transport endpoint is not connected\n'
        assert classify_failure(
            ToolKind.OPAQUE, 1, output, False,
        ) == FailureCategory.UNKNOWN_TEST_FAILURE


class TestRefusedLaunchIsDecidedByStructure:

    @pytest.mark.parametrize('slot', list(LegSlot))
    @pytest.mark.parametrize('output', ['anything', 'FAILED tests/x.py::t - AssertionError\n'])
    def test_refused_leg_is_env_transient_whatever_it_printed(self, slot, output):
        category = classify_leg(slot, 1, output, False, 'pytest tests/', scope_launch_refused=True)
        assert category == FailureCategory.ENV_TRANSIENT

    @pytest.mark.parametrize(
        ('rc', 'timed_out', 'cmd'),
        [(0, False, 'pytest'), (1, True, 'pytest'), (1, False, None)],
        ids=['green', 'timed-out', 'no-command'],
    )
    def test_check_run_refuses_a_refusal_that_is_not_a_red_run(self, rc, timed_out, cmd):
        with pytest.raises(ValueError, match='scope_launch_refused') as raised:
            CheckRun(
                label=LegSlot.TEST, cmd=cmd, rc=rc, output='', timed_out=timed_out,
                started_at=None, duration_secs=0.0, scope_launch_refused=True,
            )
        message = str(raised.value)
        assert f'rc={rc}' in message
        assert f'timed_out={timed_out}' in message
        assert f'cmd={cmd!r}' in message

    @pytest.mark.parametrize(('rc', 'timed_out'), [(0, False), (1, True)], ids=['green', 'timed-out'])
    def test_classify_leg_refuses_the_same_impossible_states(self, rc, timed_out):
        with pytest.raises(ValueError, match='scope_launch_refused') as raised:
            classify_leg(LegSlot.LINT, rc, '', timed_out, 'ruff check', scope_launch_refused=True)
        assert f'rc={rc}' in str(raised.value)
        assert f'timed_out={timed_out}' in str(raised.value)


def _install_refusing_systemd_run(tmp_path: Path, monkeypatch, refusal: str) -> None:
    """Put a ``systemd-run`` first on PATH that refuses, as with no user bus.

    It prints *refusal* to stderr and exits 1 without exec'ing its payload, so
    the real ``shutil.which`` + ``create_subprocess_exec`` spawn path produces
    the production leg output.
    """
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


@pytest.fixture(params=_SAMPLES)
def unreachable_bus(request, tmp_path, monkeypatch) -> str:
    """A refusing ``systemd-run`` printing a grounded refusal; returns the refusal."""
    _install_refusing_systemd_run(tmp_path, monkeypatch, request.param)
    return request.param


@pytest.fixture(params=_ALL_REFUSALS)
def refusing_launcher(request, tmp_path, monkeypatch) -> str:
    """A refusing ``systemd-run`` for every reason, observed or not; returns the refusal."""
    _install_refusing_systemd_run(tmp_path, monkeypatch, request.param)
    return request.param


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


@pytest.mark.usefixtures('code_default_config', 'refusing_launcher')
class TestRefusalReasonDoesNotMatter:

    @pytest.mark.asyncio
    async def test_every_leg_refused_is_env_transient_for_any_reason(self, worktree):
        result = await _verify_with_unreachable_bus(worktree)
        assert result.passed is False
        assert result.category == FailureCategory.ENV_TRANSIENT, result.summary


async def _verify_in_started_scope(worktree: Path, test_command: str) -> VerifyResult:
    return await run_verification(
        worktree,
        OrchestratorConfig(verify_use_cgroup_scope=True),
        module_config=ModuleConfig(
            prefix='pkg', test_command=test_command,
            lint_command=None, type_check_command=None,
        ),
        max_retries=0,
        role='task',
    )


@pytest.mark.usefixtures('code_default_config', 'fake_systemd_run')
class TestStartedScopeKeepsItsOwnVerdict:

    @pytest.mark.asyncio
    @pytest.mark.parametrize('test_command', ['exit 1', 'if then fi'], ids=['red', 'syntax-error'])
    async def test_red_command_in_a_started_scope_is_not_env_transient(
        self, worktree, test_command,
    ):
        result = await _verify_in_started_scope(worktree, test_command)
        assert result.passed is False
        assert result.category != FailureCategory.ENV_TRANSIENT, result.summary


def _make_git_ops() -> MagicMock:
    git_ops = MagicMock()
    git_ops.cleanup_merge_worktree = AsyncMock()
    git_ops.prune_stale_merge_worktrees = AsyncMock(return_value=[])
    return git_ops


def _make_req(project_root: Path) -> MagicMock:
    req = MagicMock()
    req.task_id = 'task-scope-spawn'
    req.config.project_root = project_root
    req.task_files = None
    req.module_configs = []
    req.config.merge_verify_min_free_disk_bytes = 1024
    req.config.merge_verify_workspace = False
    return req


@pytest.mark.usefixtures('code_default_config', 'unreachable_bus')
class TestMergeLaneHoldsScopeSpawnFailure:

    @pytest.mark.asyncio
    async def test_persistent_refusal_holds_as_transient_infra(self, worktree, tmp_path):
        spawn_failed = await _verify_with_unreachable_bus(worktree)
        verifier = FakeVerifier(default=VerifyScript(result=spawn_failed))
        timeouts: dict[str, int] = {}

        outcome = await _run_post_merge_verify(
            _make_git_ops(), _make_req(tmp_path), MagicMock(),
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
    async def test_refusal_that_clears_on_retry_lets_the_merge_proceed(self, worktree, tmp_path):
        spawn_failed = await _verify_with_unreachable_bus(worktree)
        verifier = FakeVerifier(sequence=[VerifyScript(result=spawn_failed)])

        outcome = await _run_post_merge_verify(
            _make_git_ops(), _make_req(tmp_path), MagicMock(),
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            verifier=verifier,
        )

        assert outcome is None
        assert len(verifier.verified) == 2
