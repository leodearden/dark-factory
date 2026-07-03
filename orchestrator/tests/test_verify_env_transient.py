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

from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator import verify
from orchestrator.config import ModuleConfig, OrchestratorConfig


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
