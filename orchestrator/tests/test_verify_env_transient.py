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
  step-11: pip-absence word-boundary regression guard (negative + positive preservation)
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
    transient via a single bounded serial retry (serial_pytest, task 2125
    step-25/26 — the VerifyCmd mutator that replaced _force_serial_pytest)
    instead of misattributing it as test drift.

    Mirrors TestRunVerificationColdFirstUse's `_run_cmd` mocking pattern.
    RED today: the env-recovery retry still renders through the deleted
    _force_serial_pytest's format (flags appended after targets, quoted
    ``addopts=''``); the VerifyCmd model's structured render puts flags
    before targets and does not quote a bare-safe ``addopts=`` — see
    RECOVERED_TEST_COMMAND, computed the same way
    test_verify_cmd.TestSerialPytest.test_structured_single_invocation_appends_flags
    pins it for the mutator in isolation.
    """

    # The exact recovery command for `_make_config`'s test_command
    # ('uv run pytest tests/'), rendered via
    # render(serial_pytest(parse_config_command(test_command))): the
    # structured (non-chained) render path puts base_flags before targets
    # and shlex.quote leaves the safe 'addopts=' token unquoted — distinct
    # from the deleted _force_serial_pytest's append-at-end, quoted
    # ``addopts=''`` form.
    RECOVERED_TEST_COMMAND = 'uv run pytest -p no:xdist -o addopts= tests/'

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
        (serial_pytest-rewritten) retry passes -> GREEN, env recovered."""
        config = self._make_config(tmp_path)
        invoked_cmds: list[str] = []

        async def fake_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            invoked_cmds.append(cmd)
            if 'pytest' in cmd and '-o addopts=' not in cmd:
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
        assert any(c == self.RECOVERED_TEST_COMMAND for c in pytest_invocations), (
            f'Expected the serial_pytest()-rendered recovery command '
            f'{self.RECOVERED_TEST_COMMAND!r} to be invoked, got {pytest_invocations!r}'
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
        assert any(c == self.RECOVERED_TEST_COMMAND for c in pytest_invocations), (
            f'Expected the bounded recovery retry to invoke the '
            f'serial_pytest()-rendered command {self.RECOVERED_TEST_COMMAND!r}, '
            f'got {pytest_invocations!r}'
        )

    @pytest.mark.asyncio
    async def test_env_transient_recovery_logs_addopts_clearing_note(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        """Amendment regression (reviewer_comprehensive robustness_false_signal,
        verify.py:637): serial_pytest's `-o addopts=` clears ALL
        pyproject addopts, not just `-n auto` — including any marker filters
        (e.g. `-m 'not integration'`) — so the single recovery run can
        exercise a materially different test selection than the original.
        At minimum this must be observable in the log so an operator
        investigating a recovery-path result knows a broader test selection
        may have run.
        """
        import logging

        config = self._make_config(tmp_path)

        async def fake_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            if 'pytest' in cmd and '-o addopts=' not in cmd:
                return 4, _XDIST_VANISHED_OUTPUT, False
            return 0, '', False

        with (
            caplog.at_level(logging.WARNING, logger='orchestrator.verify'),
            patch('orchestrator.verify._run_cmd', side_effect=fake_cmd),
        ):
            result = await verify.run_verification(tmp_path, config)

        assert result.passed is True
        addopts_warnings = [
            r for r in caplog.records
            if 'shared-venv transient' in r.getMessage() and 'addopts' in r.getMessage()
        ]
        assert addopts_warnings, (
            f'Expected the recovery WARNING to mention addopts/marker-filter '
            f'clearing; got: {[r.getMessage() for r in caplog.records]}'
        )

    @pytest.mark.asyncio
    async def test_env_transient_recovery_retry_itself_times_out(
        self, tmp_path: Path,
    ):
        """Case C: the env-recovery retry itself hits the wall-clock timeout.

        Amendment regression (reviewer_comprehensive robustness_stale_state,
        verify.py:2681): the recovery block recomputed `passed`/`category`/
        `cause_hint`/`summary` from the retried test result but NOT
        `timed_out` (set once, before the recovery run, from the first
        pass).  Before the fix this left an inconsistent VerifyResult —
        category='infra_timeout' but timed_out=False — which both (a) wrongly
        marked the worktree warm (`not result.timed_out` at the warm-marking
        check) despite the recovery run having timed out, and (b) hid the
        timeout from callers that special-case `result.timed_out`
        (merge_queue.py, workflow.py).
        """
        (tmp_path / '.task').mkdir()
        config = self._make_config(tmp_path)
        invoked_cmds: list[str] = []

        async def fake_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            invoked_cmds.append(cmd)
            if '-o addopts=' in cmd:
                return 1, f'Command timed out after {timeout}s: {cmd}', True
            if 'pytest' in cmd:
                return 4, _XDIST_VANISHED_OUTPUT, False
            return 0, '', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
            result = await verify.run_verification(tmp_path, config)

        assert result.passed is False
        assert result.category == 'infra_timeout'
        assert result.timed_out is True, (
            f'Expected timed_out=True to stay consistent with category='
            f'{result.category!r} after the recovery retry itself timed out '
            f'(got timed_out={result.timed_out!r})'
        )
        assert not (tmp_path / '.task' / 'verify_warmed').exists(), (
            'A recovery run that times out must not mark the worktree warm'
        )
        pytest_invocations = [c for c in invoked_cmds if 'pytest' in c]
        assert len(pytest_invocations) == 2, (
            f'Expected exactly 2 pytest invocations (original + one bounded '
            f'recovery retry), got {len(pytest_invocations)}: {pytest_invocations!r}'
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


class TestClassifyFailurePipWordBoundary:
    """step-11: word-boundary regression guard for the pip-absence pattern
    (fixes reviewer_comprehensive robustness_misclassification at verify.py:440).

    The original `_ENV_TRANSIENT_PATTERNS` pip pattern
    (``No module named ['"]?pip['"]?``) left the trailing quote optional and
    had no boundary after 'pip', so `re.search` substring-matched any
    ModuleNotFoundError whose module name merely STARTS with 'pip'
    (pipeline, pipx, pipenv, pip_audit, pip-tools). That reproduces the exact
    INVERSE misattribution this feature forbids: a genuine import/code
    regression would be silently relabelled 'env_transient', auto-retried via
    _force_serial_pytest, kept off the archive deny-list, and treated as a
    None infra sentinel by run_main_tip_sweep — i.e. never surfaced for human
    triage. RED today: all five pip-prefixed variants currently match.
    """

    def _classify(self, output: str, rc: int, timed_out: bool) -> str:
        return verify._classify_failure(output, rc, timed_out)

    @pytest.mark.parametrize(
        'module_name',
        ['pipx', 'pipenv', 'pip_audit', 'pip-tools'],
    )
    def test_pip_prefixed_module_name_not_env_transient(self, module_name: str):
        """A ModuleNotFoundError whose module name merely starts with 'pip'
        must NOT be swept into env_transient — it's a real import/code
        regression, not a pip-absence signature."""
        output = f"ModuleNotFoundError: No module named '{module_name}'\n"
        assert self._classify(output, rc=1, timed_out=False) != 'env_transient'

    def test_pipeline_module_not_found_is_unknown_test_failure(self):
        """A genuine 'No module named pipeline' import regression must stay
        RED and be surfaced for human triage — 'unknown_test_failure'
        specifically — not silently relabelled environmental, auto-retried,
        and archive-denied."""
        output = "ModuleNotFoundError: No module named 'pipeline'\n"
        assert self._classify(output, rc=1, timed_out=False) == 'unknown_test_failure'

    def test_unquoted_runpy_pip_absence_still_env_transient(self):
        """The grounded task-2045 runpy line — unquoted, no
        'ModuleNotFoundError:' prefix (emitted by `python -m pip` itself, not
        a bare `import pip`) — must still classify as env_transient."""
        output = '/home/leo/src/dark-factory/.venv/bin/python3.12: No module named pip\n'
        assert self._classify(output, rc=1, timed_out=False) == 'env_transient'

    def test_single_quoted_pip_still_env_transient(self):
        """The quoted 'ModuleNotFoundError: No module named 'pip'' form must
        still classify as env_transient after the word-boundary fix."""
        output = "ModuleNotFoundError: No module named 'pip'\n"
        assert self._classify(output, rc=1, timed_out=False) == 'env_transient'

    def test_double_quoted_pip_still_env_transient(self):
        """The double-quoted form must still classify as env_transient after
        the word-boundary fix."""
        output = 'ModuleNotFoundError: No module named "pip"\n'
        assert self._classify(output, rc=1, timed_out=False) == 'env_transient'


class TestClassifyFailureXdistUsageErrorPytestScoped:
    """Amendment regression guard for reviewer_comprehensive
    misclassification_risk at verify.py:427.

    The original xdist usage-error pattern
    (``^.*unrecognized arguments:.*(?:-n\\b|--dist\\b|--max-worker-restart\\b).*$``)
    matched ANY line containing 'unrecognized arguments:' followed by
    -n/--dist/--max-worker-restart — not just pytest's own argparse usage
    error. A different CLI tool emitting a similarly shaped
    'unrecognized arguments: -n ...' line would have been swept into
    'env_transient', auto-retried via _force_serial_pytest, and kept off the
    archive deny-list — the same inverse-misattribution hazard the pip
    pattern's word-boundary fix (TestClassifyFailurePipWordBoundary) guards
    against. RED before the fix: the non-pytest sample below matched
    env_transient; the fix anchors the pattern to the literal
    'pytest: error: unrecognized arguments:' prefix argparse emits for
    pytest's own CLI (prog='pytest').
    """

    def _classify(self, output: str, rc: int, timed_out: bool) -> str:
        return verify._classify_failure(output, rc, timed_out)

    def test_non_pytest_unrecognized_arguments_not_env_transient(self):
        """A different CLI tool's usage error must not be swept into
        env_transient, even though it shares the same -n/--dist/
        --max-worker-restart flag vocabulary as the grounded pytest
        signature — the '{prog}: error: ...' prefix names a different,
        unrelated tool, so this is a genuine CLI usage failure to surface
        for human triage, not the xdist-vanished signature."""
        output = (
            'usage: sometool [options]\n'
            'sometool: error: unrecognized arguments: -n --dist --max-worker-restart=0\n'
        )
        assert self._classify(output, rc=4, timed_out=False) == 'unknown_test_failure'

    def test_pytest_usage_error_still_env_transient_after_scoping(self):
        """The grounded task-2045 pytest usage error must still classify as
        env_transient after anchoring to the 'pytest: error:' prefix."""
        output = (
            'usage: pytest [options] [file_or_dir] [file_or_dir] [...]\n'
            'pytest: error: unrecognized arguments: -n --dist --max-worker-restart=0\n'
        )
        assert self._classify(output, rc=4, timed_out=False) == 'env_transient'


# task 2365: bare pytest-xdist worker-crash signature. Grounded in
# config.yaml's task-2361 comment (under host CPU oversubscription a starved
# xdist worker crosses the per-test wall-clock ceiling, gets os._exit()'d by
# pytest-timeout's thread method, and --max-worker-restart=0 turns that into
# a false-failing per-test "node down" on whatever test happens to be
# running) and test_cli.py's grounded wording:
# ``[gwN] node down: Not properly terminated``. NO ^E  /^FAILED /failed-summary
# lines — a hard os._exit() worker kill produces no assertion traceback.
_XDIST_WORKER_CRASH_OUTPUT = (
    'orchestrator/tests/test_config.py ....\n'
    '[gw3] node down: Not properly terminated\n'
    "worker gw3 crashed while running 'orchestrator/tests/test_config.py::TestFoo::test_bar'\n"
)

# Same crash signature, but WITH a genuine failure alongside it (collateral
# FAILED line from the dead worker's run) — the conservative discriminator
# must treat this as a real failure and NOT reclassify it.
_XDIST_CRASH_WITH_REAL_FAILURE_OUTPUT = (
    _XDIST_WORKER_CRASH_OUTPUT
    + 'E   AssertionError: expected 3, got 4\n'
    + 'FAILED orchestrator/tests/test_x.py::test_real - AssertionError\n'
    + '========== 1 failed, 2 passed in 5.00s ==========\n'
)


class TestBareXdistWorkerCrashDetector:
    """task 2365 step-1: verify._is_bare_xdist_worker_crash(output) pure helper.

    RED today: the helper does not exist yet (AttributeError).
    """

    def test_bare_worker_crash_is_true(self):
        """A bare node-down/worker-crash signature with no real failure marker -> True."""
        assert verify._is_bare_xdist_worker_crash(_XDIST_WORKER_CRASH_OUTPUT) is True

    def test_crash_with_real_failure_markers_is_false(self):
        """The same crash signature PLUS genuine E/FAILED/summary lines -> False.

        Never mask a genuine failure: any real pytest failure marker alongside
        the crash signature suppresses reclassification.
        """
        assert verify._is_bare_xdist_worker_crash(_XDIST_CRASH_WITH_REAL_FAILURE_OUTPUT) is False

    def test_plain_failure_with_no_crash_signature_is_false(self):
        """An ordinary FAILED line with no crash signature at all -> False."""
        output = 'FAILED orchestrator/tests/test_y.py::test_y - AssertionError\n'
        assert verify._is_bare_xdist_worker_crash(output) is False

    def test_crash_signature_with_traceback_e_line_is_false(self):
        """Crash signature plus a lone `E   ...` traceback line -> False.

        Guards the narrower case where only the traceback E-line (not a
        FAILED line or failure summary) accompanies the crash signature.
        """
        output = (
            '[gw3] node down: Not properly terminated\n'
            'E   TypeError: unexpected keyword argument\n'
        )
        assert verify._is_bare_xdist_worker_crash(output) is False


class TestRunVerificationXdistWorkerCrashRetry:
    """task 2365 step-3/4: run_verification raises VerifyInfraError on a bare
    pytest-xdist worker crash so the task-path retry wrapper
    (_run_scoped_verification_with_infra_retry) retries the whole scoped
    verification with backoff instead of the debugfix loop invoking the
    DEBUGGER on a non-reproducible infra flake (esc-2286-21).

    Mirrors TestRunVerificationEnvRecovery's `_run_cmd` mocking pattern.
    RED today (before step-4): case (a) returns a failed VerifyResult
    (category='unknown_test_failure') instead of raising; cases (b)/(c) are
    guards that must hold both before and after step-4.
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
    async def test_bare_worker_crash_raises_verify_infra_error(self, tmp_path: Path):
        """Case (a): a bare worker crash raises VerifyInfraError(phase='xdist_worker_crash')."""
        config = self._make_config(tmp_path)

        async def fake_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            if 'pytest' in cmd:
                return 1, _XDIST_WORKER_CRASH_OUTPUT, False
            return 0, '', False

        with (
            patch('orchestrator.verify._run_cmd', side_effect=fake_cmd),
            pytest.raises(verify.VerifyInfraError) as exc_info,
        ):
            await verify.run_verification(tmp_path, config)

        assert exc_info.value.phase == 'xdist_worker_crash'
        assert exc_info.value.errno is None

    @pytest.mark.asyncio
    async def test_crash_with_real_failure_does_not_raise(self, tmp_path: Path):
        """Case (b): a real failure alongside the crash signature must NOT be
        masked — run_verification returns a failed VerifyResult instead of
        raising."""
        config = self._make_config(tmp_path)

        async def fake_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            if 'pytest' in cmd:
                return 1, _XDIST_CRASH_WITH_REAL_FAILURE_OUTPUT, False
            return 0, '', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
            result = await verify.run_verification(tmp_path, config)

        assert result.passed is False
        assert result.category == 'test_failure'

    @pytest.mark.asyncio
    async def test_merge_verify_does_not_raise(self, tmp_path: Path):
        """Case (c): the merge path (is_merge_verify=True) has no VerifyInfraError
        handler — an uncaught raise there would stall the merge queue, so the
        gate must not fire."""
        config = self._make_config(tmp_path)

        async def fake_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            if 'pytest' in cmd:
                return 1, _XDIST_WORKER_CRASH_OUTPUT, False
            return 0, '', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
            result = await verify.run_verification(tmp_path, config, is_merge_verify=True)

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_crash_with_concurrent_lint_failure_does_not_raise(self, tmp_path: Path):
        """Co-occurrence guard: a bare worker crash on the test leg must not
        divert a genuine, simultaneous lint failure onto the infra-retry path.

        _is_bare_xdist_worker_crash only inspects test_out, so a real lint
        regression happening at the same time as the crash would otherwise be
        silently discarded by the raise (routing to infra_hold/retry instead
        of being surfaced). The gate additionally requires lint_rc == 0 and
        type_rc == 0 so it only fires when the crashed test leg is the ONLY
        non-zero check.
        """
        config = self._make_config(tmp_path)

        async def fake_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            if 'pytest' in cmd:
                return 1, _XDIST_WORKER_CRASH_OUTPUT, False
            if 'lint' in cmd:
                return 1, 'file.py:10:1: E501 line too long\n', False
            return 0, '', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
            result = await verify.run_verification(tmp_path, config)

        assert result.passed is False
        assert 'lint issues' in result.summary

    @pytest.mark.asyncio
    async def test_crash_with_concurrent_type_failure_does_not_raise(self, tmp_path: Path):
        """Same co-occurrence guard as above, for the type-check leg."""
        config = self._make_config(tmp_path)

        async def fake_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            if 'pytest' in cmd:
                return 1, _XDIST_WORKER_CRASH_OUTPUT, False
            if 'type' in cmd:
                return 1, 'file.py:3: error: Incompatible return value type\n', False
            return 0, '', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
            result = await verify.run_verification(tmp_path, config)

        assert result.passed is False
        assert 'type errors' in result.summary
