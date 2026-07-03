"""Tests for env-transient shared-venv-mutation detection — task 2048.

A concurrent `uv sync` from another orchestrator process on the shared
/home/leo/src/dark-factory/.venv can transiently mutate that venv (a
non-atomic remove-then-readd window) WHILE a consumer is mid-pytest against
it. Observed on task 2045: an identical `pytest -n auto` that first passed
later failed with `unrecognized arguments: -n --dist --max-worker-restart=0`;
`python -c "import xdist"` raised ModuleNotFoundError; `python -m pip`
reported "No module named pip". A serial run (`-o addopts=""`) passed,
confirming it was environmental, not a code regression.

These signatures are test-harness-infrastructure-absence signatures
(xdist/pip vanished) that application code does not normally emit, so they
must classify as 'env_transient' — an infra transient, NOT a code
regression — so verify auto-recovers via a bounded serial retry instead of
misattributing the loss as test drift (see the module's
INVERSE-MISATTRIBUTION GUARDRAIL discussion).

Test coverage:
  step-1: _classify_failure grounded env_transient signatures + negative guards
  step-3: non-misattribution wiring (_worst_category, PREEXISTING_BREAK_SKIP_CATEGORIES,
          _should_archive_category, _CATEGORY_PRIORITY)
  step-5: _force_serial_pytest pure helper
  step-7: run_verification env-recovery retry (integration, mocked _run_cmd)
  step-9: run_main_tip_sweep env_transient infra sentinel (mocked run_full_verification)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator import verify
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.verify import VerifyResult


class TestClassifyFailureEnvTransient:
    """step-1: _classify_failure recognises grounded shared-venv-mutation signatures.

    Each test imports nothing extra — `_classify_failure` already exists;
    only the 'env_transient' category is new. RED today because no pattern
    ladder branch produces 'env_transient' yet.
    """

    def _classify(self, output: str, rc: int, timed_out: bool) -> str:
        return verify._classify_failure(output, rc, timed_out)

    def test_xdist_usage_error_is_env_transient(self):
        """pytest usage error (rc=4) when the xdist plugin vanished mid-run.

        Grounded in task 2045's exact observed line: an identical
        `pytest -n auto` invocation that had just passed failed with this
        usage error once a concurrent `uv sync` removed the xdist plugin.
        """
        output = (
            'usage: pytest [options] [file_or_dir] [file_or_dir] [...]\n'
            'pytest: error: unrecognized arguments: -n --dist --max-worker-restart=0\n'
        )
        assert self._classify(output, rc=4, timed_out=False) == 'env_transient'

    def test_no_module_named_pip_is_env_transient(self):
        """`python -m pip` reports this when pip itself vanished from the venv.

        Grounded in task 2045's observation: `python -m pip` -> "No module
        named pip" during the same concurrent uv sync window.
        """
        output = '/home/leo/src/dark-factory/.venv/bin/python3.12: No module named pip\n'
        assert self._classify(output, rc=1, timed_out=False) == 'env_transient'

    def test_modulenotfounderror_xdist_is_env_transient(self):
        """`import xdist` raises ModuleNotFoundError when the plugin vanished.

        Grounded in task 2045's observation: `python -c "import xdist"` ->
        ModuleNotFoundError during the same concurrent uv sync window.
        """
        output = (
            'Traceback (most recent call last):\n'
            '  File "<string>", line 1, in <module>\n'
            "ModuleNotFoundError: No module named 'xdist'\n"
        )
        assert self._classify(output, rc=1, timed_out=False) == 'env_transient'

    def test_modulenotfounderror_pytest_xdist_is_env_transient(self):
        """`import pytest_xdist` raises ModuleNotFoundError when the plugin vanished.

        Separate module-name spelling from the 'xdist' case above — pytest's
        own plugin bookkeeping sometimes surfaces the distribution name
        ('pytest_xdist') rather than the import name ('xdist').
        """
        output = (
            'Traceback (most recent call last):\n'
            '  File "<string>", line 1, in <module>\n'
            "ModuleNotFoundError: No module named 'pytest_xdist'\n"
        )
        assert self._classify(output, rc=1, timed_out=False) == 'env_transient'

    def test_plain_test_failure_not_env_transient(self):
        """A genuine test failure (no xdist/pip absence signature) stays test_failure.

        Negative guard: env_transient patterns must be narrow enough that an
        ordinary FAILED line is not swept up into the infra bucket.
        """
        output = 'FAILED orchestrator/tests/test_x.py::test_y - AssertionError\n'
        assert self._classify(output, rc=1, timed_out=False) == 'test_failure'

    def test_ordinary_error_not_env_transient(self):
        """A generic non-xdist `error:` line must not be misclassified as env_transient.

        Negative guard: application/cargo-style 'error: ...' lines unrelated
        to xdist/pip absence must not be swept into env_transient.
        """
        output = 'error: something went wrong unrelated to xdist or pip\n'
        assert self._classify(output, rc=1, timed_out=False) == 'unknown_test_failure'


class TestEnvTransientNonMisattributionWiring:
    """step-3: env_transient must be wired into every non-misattribution collection.

    Classifying the failure correctly (step 1/2) is not enough on its own —
    downstream selection (_worst_category), the preexisting-main-break probe,
    and archival must all treat env_transient as an infra category, not a
    human-triage-worthy or drift-worthy one. RED today: env_transient is
    absent from all three collections.
    """

    def test_worst_category_ranks_env_transient_above_test_failure(self):
        """A mixed-category run resolves to env_transient, not test_failure.

        This mirrors how a xdist-death run whose output contains BOTH the
        env_transient signature and collateral FAILED lines must resolve to
        the infra category, not the test-drift category.
        """
        assert verify._worst_category(['test_failure', 'env_transient']) == 'env_transient'

    def test_env_transient_skips_preexisting_break_probe(self):
        """env_transient is non-deterministic to re-probe on main (like infra_timeout)."""
        assert 'env_transient' in verify.PREEXISTING_BREAK_SKIP_CATEGORIES

    def test_env_transient_not_archived(self):
        """env_transient is infra, not human-triage-worthy — must not be archived."""
        assert verify._should_archive_category('env_transient') is False

    def test_env_transient_ranked_above_test_failure_in_category_priority(self):
        """env_transient must outrank test_failure in the severity list."""
        assert 'env_transient' in verify._CATEGORY_PRIORITY
        assert (
            verify._CATEGORY_PRIORITY.index('env_transient')
            < verify._CATEGORY_PRIORITY.index('test_failure')
        )


class TestForceSerialPytest:
    """step-5: _force_serial_pytest(cmd) is a pure string-rewrite helper.

    Appends ` -p no:xdist -o addopts=''` to every `pytest` invocation in a
    `&&`-chain — reproducing task 2045's proven `-o addopts=""` serial
    workaround (clears the pyproject `-n auto`/xdist addopts) plus a
    belt-and-suspenders `-p no:xdist` plugin disable. Non-pytest commands and
    None pass through unchanged. RED today: _force_serial_pytest does not exist.
    """

    # The real multi-module test_command from orchestrator/config.yaml —
    # five chained `cd <module> && uv run pytest tests/` invocations.
    REAL_CONFIG_TEST_COMMAND = (
        'cd shared && uv run pytest tests/ && '
        'cd ../escalation && uv run pytest tests/ && '
        'cd ../orchestrator && uv run pytest tests/ && '
        'cd ../fused-memory && uv run pytest tests/ && '
        'cd ../dashboard && uv run pytest tests/'
    )

    def test_rewrites_every_pytest_invocation_in_chained_command(self):
        """Each of the 5 chained `uv run pytest tests/` segments gains the flags."""
        result = verify._force_serial_pytest(self.REAL_CONFIG_TEST_COMMAND)

        expected = (
            "cd shared && uv run pytest tests/ -p no:xdist -o addopts='' && "
            "cd ../escalation && uv run pytest tests/ -p no:xdist -o addopts='' && "
            "cd ../orchestrator && uv run pytest tests/ -p no:xdist -o addopts='' && "
            "cd ../fused-memory && uv run pytest tests/ -p no:xdist -o addopts='' && "
            "cd ../dashboard && uv run pytest tests/ -p no:xdist -o addopts=''"
        )
        assert result == expected

        # Number of rewrites equals the number of pytest invocations (5).
        assert self.REAL_CONFIG_TEST_COMMAND.count('pytest') == 5
        assert result.count("-p no:xdist -o addopts=''") == 5

    def test_non_pytest_command_returned_unchanged(self):
        """A command with no `pytest` token (e.g. cargo) passes through unchanged."""
        cmd = 'cargo test --workspace'
        assert verify._force_serial_pytest(cmd) == cmd

    def test_none_returns_none(self):
        """A None command (skipped check) stays None."""
        assert verify._force_serial_pytest(None) is None


_XDIST_VANISHED_OUTPUT = (
    'usage: pytest [options] [file_or_dir] [file_or_dir] [...]\n'
    'pytest: error: unrecognized arguments: -n --dist --max-worker-restart=0\n'
)


class TestRunVerificationEnvRecovery:
    """step-7: run_verification auto-recovers from a shared-venv-mutation
    transient via a single bounded serial retry (_force_serial_pytest)
    instead of misattributing it as test drift.

    Mirrors TestRunVerificationColdFirstUse's `_run_cmd` mocking pattern.
    RED today: no env-recovery retry exists, so the rewritten (recovery)
    command is never invoked and a still-transient retry can't be
    distinguished from "no retry happened at all".
    """

    def _make_config(self, tmp_path: Path) -> OrchestratorConfig:
        # lint/type are non-Optional str fields on OrchestratorConfig (unlike
        # ModuleConfig, which allows None to mean "skip") — use harmless
        # always-succeeding placeholders so only the 'test' check is under
        # test here.
        return OrchestratorConfig(
            project_root=tmp_path,
            test_command='uv run pytest tests/',
            lint_command='echo lint',
            type_check_command='echo type',
        )

    @pytest.mark.asyncio
    async def test_env_transient_recovers_on_serial_retry(self, tmp_path: Path):
        """Case A: the original run hits the xdist usage-error; the serial
        (_force_serial_pytest-rewritten) retry passes -> GREEN, env recovered."""
        config = self._make_config(tmp_path)
        invoked_cmds: list[str] = []

        async def fake_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            invoked_cmds.append(cmd)
            if 'pytest' in cmd and "-o addopts=''" not in cmd:
                return 4, _XDIST_VANISHED_OUTPUT, False
            return 0, '', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
            result = await verify.run_verification(tmp_path, config)

        assert result.passed is True, f'Expected recovery to pass, got {result!r}'
        assert result.category == 'passed'
        pytest_invocations = [c for c in invoked_cmds if 'pytest' in c]
        assert len(pytest_invocations) == 2, (
            f'Expected exactly 2 pytest invocations (original + one bounded '
            f'recovery retry), got {len(pytest_invocations)}: {pytest_invocations!r}'
        )
        assert any("-o addopts=''" in c for c in pytest_invocations), (
            f"Expected a recovery command containing -o addopts='' to be "
            f'invoked, got {pytest_invocations!r}'
        )

    @pytest.mark.asyncio
    async def test_env_transient_preserved_when_retry_still_transient(self, tmp_path: Path):
        """Case B: both the original run AND the serial retry hit the
        xdist-vanished signature -> category stays 'env_transient' (NOT
        misattributed as unknown_test_failure/test_failure), and the
        recovery retry must actually have been attempted."""
        config = self._make_config(tmp_path)
        invoked_cmds: list[str] = []

        async def fake_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            invoked_cmds.append(cmd)
            if 'pytest' in cmd:
                return 4, _XDIST_VANISHED_OUTPUT, False
            return 0, '', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
            result = await verify.run_verification(tmp_path, config)

        assert result.passed is False
        assert result.category == 'env_transient', (
            f'Expected env_transient to be preserved (not misattributed), '
            f'got category={result.category!r}'
        )
        pytest_invocations = [c for c in invoked_cmds if 'pytest' in c]
        assert len(pytest_invocations) == 2, (
            f'Expected exactly 2 pytest invocations (original + one bounded '
            f'recovery retry — the retry must actually fire, not just leave '
            f'the first-pass category untouched), got {len(pytest_invocations)}: '
            f'{pytest_invocations!r}'
        )
        assert any("-o addopts=''" in c for c in pytest_invocations), (
            f"Expected the bounded recovery retry to invoke a command "
            f"containing -o addopts='', got {pytest_invocations!r}"
        )


class TestRunMainTipSweepEnvTransient:
    """step-9: run_main_tip_sweep treats env_transient exactly like
    pytest_internalerror — an infra crash, not drift — so it returns the
    None sentinel (retry next tick, no false-positive L1 drift escalation)
    instead of surfacing a false main-branch break.

    Mirrors test_verify_main_tip_sweep.py's INTERNALERROR test structure
    (mock strategy: monkeypatch orchestrator.git_ops._run for worktree
    add/remove, and orchestrator.verify.run_full_verification for the sweep
    itself). RED today: run_main_tip_sweep only special-cases
    'pytest_internalerror', so category='env_transient' falls through to a
    normal (sha, result) drift return instead of None.
    """

    MAIN_SHA = 'b' * 40

    ENV_TRANSIENT_RESULT = VerifyResult(
        passed=False,
        test_output=_XDIST_VANISHED_OUTPUT,
        lint_output='',
        type_output='',
        summary='env_transient',
        cause_hint='pytest: error: unrecognized arguments: -n --dist --max-worker-restart=0',
        category='env_transient',
    )

    FAILING_RESULT = VerifyResult(
        passed=False,
        test_output='FAILED test_x',
        lint_output='',
        type_output='',
        summary='test_failure',
        cause_hint='FAILED test_x',
        category='test_failure',
    )

    def _make_config(self, tmp_path: Path) -> OrchestratorConfig:
        return OrchestratorConfig(
            project_root=tmp_path,
            max_concurrent_tasks=1,
            git=GitConfig(
                main_branch='main',
                branch_prefix='task/',
                remote='origin',
                worktree_dir='.worktrees',
            ),
        )

    def _make_git_ops(self, tmp_path: Path) -> MagicMock:
        mock = MagicMock()
        mock.get_main_sha = AsyncMock(return_value=self.MAIN_SHA)
        worktree_base = tmp_path / '.worktrees'
        worktree_base.mkdir(parents=True, exist_ok=True)
        mock.worktree_base = worktree_base
        return mock

    def test_env_transient_first_pass_returns_none(self, tmp_path: Path):
        """First-pass env_transient -> None (infra sentinel); cleanup still runs."""
        config = self._make_config(tmp_path)
        git_ops = self._make_git_ops(tmp_path)
        run_calls: list = []

        async def _fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            return (0, '', '')

        async def _fake_full_verify(*args, **kwargs):
            return self.ENV_TRANSIENT_RESULT

        with (
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
            patch.object(verify, 'run_full_verification', side_effect=_fake_full_verify),
        ):
            result = asyncio.run(verify.run_main_tip_sweep(config, git_ops))

        assert result is None, (
            f'Expected None (infra sentinel) when category=env_transient, got {result!r}'
        )
        remove_calls = [c for c in run_calls if 'worktree' in c and 'remove' in c]
        assert remove_calls, (
            'git worktree remove --force must run even when returning None for '
            'env_transient (cleanup-in-finally guarantee)'
        )

    def test_env_transient_on_retry_returns_none(self, tmp_path: Path):
        """First-pass FAIL (non-transient category) + retry env_transient -> None.

        Mirrors test_run_main_tip_sweep_internalerror_on_retry_returns_none:
        the retry, not just the first pass, must also be checked for the
        infra sentinel category.
        """
        config = self._make_config(tmp_path)
        git_ops = self._make_git_ops(tmp_path)
        run_calls: list = []

        async def _fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            return (0, '', '')

        rfv = AsyncMock(side_effect=[self.FAILING_RESULT, self.ENV_TRANSIENT_RESULT])

        with (
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
            patch.object(verify, 'run_full_verification', rfv),
        ):
            result = asyncio.run(verify.run_main_tip_sweep(config, git_ops))

        assert result is None, (
            f'Expected None (infra sentinel) when retry returns env_transient, '
            f'got {result!r}'
        )
        remove_calls = [c for c in run_calls if 'worktree' in c and 'remove' in c]
        assert remove_calls, (
            'git worktree remove --force must run even when retry returns '
            'env_transient (cleanup-in-finally guarantee)'
        )
