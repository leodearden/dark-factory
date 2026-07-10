"""Milestone ε: end-to-end integration gate (task 2338).

B1–B12 boundary-test suite (docs/prds/milestone-tasks.md §7) that COMPOSES
the already-landed β (scheduler.py time-gate/sweep), γ (deterministic_runner.py
predicate mode), and α (shared.task_metadata.Milestone) substrate end-to-end,
rather than re-deriving their own unit assertions.  Follows the established
test_*_integration_gate.py convention (coalesce/config_reload/warm_lane).

The one genuinely new production artifact is the exemplar predicate fixture,
scripts/check_merge_flakiness.sh — a dependency-free, executable check script
that owns the threshold and the exit-code verdict contract (PRD §5.5: the
orchestrator parses nothing).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# B... — exemplar check-script contract (self-authored: ε owns both the
# script and this test — no external numeric premise).
# ---------------------------------------------------------------------------


class TestExemplarCheckScript:
    """Contract test for the ε exemplar predicate: scripts/check_merge_flakiness.sh.

    RED until step-2 authors the script: pytest.fail on the missing-script
    sentinel (the repo_root fixture's documented contract — the .git sentinel
    exists but a required file within the repo is absent, so this must not
    silently skip).
    """

    SCRIPT_REL = 'scripts/check_merge_flakiness.sh'

    def _script_path(self, repo_root: Path) -> Path:
        script = repo_root / self.SCRIPT_REL
        if not script.exists():
            pytest.fail(
                f'{self.SCRIPT_REL} does not exist at {script} — the ε exemplar '
                f'predicate script has not been authored yet'
            )
        return script

    def test_script_exists_and_is_executable(self, repo_root: Path | None):
        if repo_root is None:
            pytest.skip('not running inside a git checkout')
        script = self._script_path(repo_root)
        assert os.access(script, os.X_OK), f'{script} is not executable (missing +x bit)'

    def test_script_exits_0_and_reports_holds_when_value_below_threshold(
        self, repo_root: Path | None,
    ):
        if repo_root is None:
            pytest.skip('not running inside a git checkout')
        script = self._script_path(repo_root)
        result = subprocess.run(
            [str(script), '--window-days', '7', '--threshold', '0.05', '--value', '0.03'],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, (
            f'expected rc=0 (invariant holds); got rc={result.returncode}, '
            f'stdout={result.stdout!r}, stderr={result.stderr!r}'
        )
        tail = result.stdout.strip()
        assert 'holds' in tail, f'expected "holds" in stdout tail; got {tail!r}'

    def test_script_exits_1_and_reports_violated_when_value_at_or_above_threshold(
        self, repo_root: Path | None,
    ):
        if repo_root is None:
            pytest.skip('not running inside a git checkout')
        script = self._script_path(repo_root)
        result = subprocess.run(
            [str(script), '--value', '0.08', '--threshold', '0.05'],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 1, (
            f'expected rc=1 (invariant VIOLATED); got rc={result.returncode}, '
            f'stdout={result.stdout!r}, stderr={result.stderr!r}'
        )
        tail = result.stdout.strip()
        assert 'VIOLATED' in tail, f'expected "VIOLATED" in stdout tail; got {tail!r}'
