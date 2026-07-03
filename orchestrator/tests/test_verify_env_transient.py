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

from orchestrator import verify


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
