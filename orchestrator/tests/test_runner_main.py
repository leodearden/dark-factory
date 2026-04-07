"""Tests for `python -m orchestrator.evals.runner` entry-point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Root of the installed package (the parent of src/)
_PACKAGE_ROOT = Path(__file__).parent.parent / 'src'


def test_runner_module_invocable_as_main_help():
    """`python -m orchestrator.evals.runner --help` must exit 0 and print usage."""
    result = subprocess.run(
        [sys.executable, '-m', 'orchestrator.evals.runner', '--help'],
        capture_output=True,
        text=True,
        cwd=str(_PACKAGE_ROOT),
    )
    assert result.returncode == 0, (
        f'Expected exit 0 from --help, got {result.returncode}.\n'
        f'stdout: {result.stdout!r}\n'
        f'stderr: {result.stderr!r}'
    )
    combined = (result.stdout + result.stderr).lower()
    # The click eval_cmd exposes --judge, --trials, and --vllm-url options
    assert '--judge' in combined or '--trials' in combined, (
        f'Expected usage text in output, got:\n'
        f'stdout: {result.stdout!r}\n'
        f'stderr: {result.stderr!r}'
    )
