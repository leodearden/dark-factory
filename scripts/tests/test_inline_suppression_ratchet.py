"""Boundary tests for ``scripts/inline_suppressions.py`` — PRD γ1, scenarios 1-11.

WHAT IS UNDER TEST.  The inline-suppression scanner: the kind table and its
COMMENT-token scan, D7's content-addressed multiset key, D8's consumer model,
D9's ratified classes, and the ``--check`` / ``--seed`` / ``--tighten`` /
``--json`` verbs with the 0/1/2 exit ladder the PRD Contract fixes.
``plans/inv12-exceptions-owned-or-ratified-prd.md``, boundary-test sketch rows
1-11.

HOW IT IS TESTED, and why the shape differs between families.  Almost every
test builds a throwaway git repository under ``tmp_path`` and calls
``inline_suppressions.main([...])`` IN-PROCESS, capturing stdout/stderr with
``capsys``: the verbs' whole contract is an exit code plus rendered lines, and
an in-process call gets both without paying a subprocess per scenario.  The two
exceptions are deliberate and each is a case an in-process call cannot prove:

* the import-failure test (Contract: "an ImportError is 2, never 1") needs a
  real interpreter, because the fault it pins happens before ``main`` exists;
* the first-party code drift guard runs
  ``fused-memory/scripts/check_bare_magicmock_config.py`` as a subprocess and
  reads the violation messages it EMITS, rather than importing its private
  ``_RULE_A_CODE`` / ``_RULE_B_CODE`` constants — ``docs/code-quality.md``'s
  Tests stance (a test that reads another module's private attributes pins
  implementation rather than behaviour).

A REAL GIT REPOSITORY, not a bare directory.  The scanner enumerates its corpus
from ``git ls-files``, so trackedness is a property the fixtures must exercise
rather than one the tests assert by reading the code — the same argument
``scripts/tests/test_design_invariants_consistency.py::_write_scan_tree`` gives.

NO WALL-CLOCK ASSERTION APPEARS IN THIS MODULE, on purpose.  The PRD's ≤10 s
scan budget is enforced here as COUNTED WORK (``files_enumerated`` versus
``files_tokenized``), never as ``assert elapsed < N``.
``orchestrator/tests/test_merge_lane_ratchet.py`` already ruled on exactly this
for the sibling ratchet — "count WORK, never wall-clock" — and the measurement
behind that ruling holds here: this scan costs ~5 CPU-seconds but took 6.5-6.8 s
of wall clock on a box at load 105, and the merge gate runs on that same box.
Subprocesses therefore get a generous ``timeout=`` rather than a clock guard.

XDIST-SAFE.  The merge gate runs ``pytest … -n auto --dist loadgroup``, so every
test here keeps its mutable state inside ``tmp_path`` and changes no process-wide
state outside ``monkeypatch``.
"""

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import inline_suppressions

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / 'scripts' / 'inline_suppressions.py'

#: Where a fixture tree's baseline lives.  The real one is
#: ``scripts/inline_suppression_baseline.json`` and is κ1's to seed; fixtures
#: keep theirs inside ``tmp_path`` so no test can reach the committed file.
_BASELINE_NAME = 'inline_suppression_baseline.json'

#: Generous, and not a performance assertion — see the module docstring.  A
#: subprocess that blows through this has hung, which is worth a loud
#: ``TimeoutExpired`` rather than a silent wait.
_SUBPROCESS_TIMEOUT_SECS = 120


def _scrubbed_git_env() -> dict[str, str]:
    """``os.environ`` with every ``GIT_*`` override removed.

    ``GIT_DIR``, ``GIT_WORK_TREE``, ``GIT_INDEX_FILE`` and
    ``GIT_CEILING_DIRECTORIES`` are inherited by default, and any one of them
    silently retargets a git invocation at a different repository than its
    ``cwd`` implies — ``git -C <path>`` READS as "act on <path>" but only
    changes directory, while ``GIT_DIR`` skips repository discovery outright.
    A fixture that lost that race would ``git add`` into the live checkout.
    Spelled as in ``test_design_invariants_consistency.py::_scrubbed_git_env``;
    ``df_pytest_isolation._df_git_env_hermetic`` is the suite-wide second line
    of the same defence.
    """
    return {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}


def _run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run ``git`` *args* against *cwd* under :func:`_scrubbed_git_env`."""
    return subprocess.run(
        ['git', *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        timeout=_SUBPROCESS_TIMEOUT_SECS,
        env=_scrubbed_git_env(),
    )


def _write_fixture_tree(
    root: Path, files: Mapping[str, str], *, baseline: bool = False
) -> Path:
    """Build a git repo at *root* holding *files*, and return its baseline path.

    *files* maps a repo-relative path to its whole content — source modules and
    ``pyproject.toml`` alike, since the scanner reads the nearest pyproject as
    ordinary tracked content and a second parameter for it would only be a
    second way to write a file.

    Every path is TRACKED: ``git init -q`` then ``git add -A -f``.  No commit,
    and no ``user.name`` / ``user.email`` — ``git ls-files`` reads the INDEX,
    not history, so a commit would be ceremony the scanner never looks at.  The
    ``-f`` defeats any ambient global gitignore.  A test that needs an UNTRACKED
    file writes it itself after this call.

    With ``baseline=True`` the scanner's own ``--seed`` verb seeds the returned
    path from this very tree, so a fixture baseline can never drift from the key
    format the scanner emits.  The path is returned either way: the tests that
    need an ABSENT baseline, or a deliberately corrupt one, need somewhere to
    point ``--baseline`` at just as much as the seeded ones do.
    """
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')

    for args in (['init', '-q'], ['add', '-A', '-f']):
        _run_git(args, cwd=root)

    baseline_path = root / _BASELINE_NAME
    if baseline:
        seeded = inline_suppressions.main(
            ['--seed', '--root', str(root), '--baseline', str(baseline_path)]
        )
        assert seeded == 0, f'fixture --seed failed with exit {seeded}'
    return baseline_path


def _check(root: Path, baseline_path: Path, *paths: str) -> int:
    """Run ``--check`` over *root*, optionally scoped to *paths*."""
    return inline_suppressions.main(
        ['--check', '--root', str(root), '--baseline', str(baseline_path), *paths]
    )


def _python_env_without_shared() -> dict[str, str]:
    """An environment in which ``import shared`` cannot resolve.

    ``PYTHONPATH`` is emptied AND ``PYTHONNOUSERSITE`` is set, because the
    scanner is meant to fail on a missing ``shared`` and not on the ambient
    editable install that ``sys.path`` would otherwise supply.
    """
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
    env['PYTHONNOUSERSITE'] = '1'
    return env


def _run_script(args: list[str], *, cwd: Path, script: Path, env: dict[str, str] | None = None):
    """Run *script* as a real subprocess with ``sys.executable``."""
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_SECS,
        env=env,
    )
