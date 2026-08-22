"""Meta-guard: every whole-tree AST-scanning test in this directory must carry a
module-level ``pytest.mark.timeout`` (task 4215).

THE FAILURE MODE, END TO END.  A dozen guard tests under ``orchestrator/tests/``
sweep the ENTIRE repo -- ``rglob('*.py')`` over ~500 files -- and ``ast.parse``
each one.  Unloaded and serial that costs 6-9s per call here; the orchestrator
suite however runs under ``-n auto`` (``orchestrator/pyproject.toml:178``), so on
a 32-core box those sweeps run concurrently with everything else and the same
call has been MEASURED at 17.85 / 21.32 / 30.75s at loadavg 120-176 -- a ~4.8x
load inflation.  Worker deaths were then observed at loadavg 250-423, one
further inflation step past the 60s default at ``pyproject.toml:152``.

What makes the breach so expensive is the two settings around it:

* ``timeout_method = "thread"`` (``pyproject.toml:153``) means a breach does NOT
  fail the test -- pytest-timeout's thread method ``os._exit()``s the whole xdist
  worker ("node down: Not properly terminated");
* ``--max-worker-restart=0`` (``pyproject.toml:178``, task 1907) then declines to
  replace that worker, degrading the run to a TRUNCATED whole-suite session whose
  surviving failure names some innocent guard that merely happened to share the
  dead worker.

So a single slow tree-scan does not cost one red test -- it costs the whole
verify run, and it misattributes the blame.  Escalations esc-3980-1 and
esc-3787-1 are both instances.

THE FIX the pyproject itself sanctions ("Slow tests opt out with
``@pytest.mark.timeout(N)``", ``pyproject.toml:154``) is a module-level
``pytestmark`` on each scanner.  This module pins the two constants that mark
uses, and (from step-5 on) recomputes the census of scanners from source on
every run so the Nth new scanner is caught at commit time rather than by another
truncated verify.

NOT widening the global ``timeout``: that would blunt the 60s ceiling for the
other ~16000 tests, which is exactly the ceiling that catches real hangs.
"""
from __future__ import annotations

import ast
import tomllib

import pytest
from _orch_helpers import (
    ORCH_PYPROJECT,
    PYPROJECT_DEFAULT_TIMEOUT,
    WHOLE_TREE_SCAN_TEST_TIMEOUT,
)

# This module is ITSELF a member of the family it polices -- the invariant
# below rglob()s every *.py under this directory and ast.parse()s each one --
# so it carries the very mark it demands instead of exempting itself.  The
# deliberate omission of the sibling guards' `_THIS_FILE`/`continue` skip-self
# idiom is spelled out in that test's docstring; skipping itself would exempt
# the one file most certain to need the mark.
pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

# Resolved from _orch_helpers' shared anchor, which is itself resolved from
# THAT file and never from the process CWD: merge-verify runs pytest from the
# orchestrator/ cwd while a plain `pytest orchestrator/tests` runs from the
# repo root, and both must resolve identically.  Aliased to a private name so
# the assertion messages below read as this module's own.
_ORCH_PYPROJECT = ORCH_PYPROJECT


def _pytest_ini_options() -> dict[str, object]:
    """``[tool.pytest.ini_options]`` parsed from the REAL orchestrator pyproject."""
    data = tomllib.loads(_ORCH_PYPROJECT.read_text(encoding='utf-8'))
    return data['tool']['pytest']['ini_options']


class TestTimeoutConstants:
    """The two constants this guard's remediation advice depends on."""

    def test_pyproject_default_timeout_mirrors_pyproject(self) -> None:
        """``PYPROJECT_DEFAULT_TIMEOUT`` must equal the REAL configured default.

        The constant is a mirror of ``[tool.pytest.ini_options].timeout``, and
        its previous incarnation (test_merge_queue_concurrent_verify.py, task
        3492) admitted the defect in its own comment: "it is still a literal and
        CAN drift if that setting changes without a matching edit here; there is
        no automated link between the two."  This test IS that link -- it reads
        orchestrator/pyproject.toml with ``tomllib`` at runtime rather than
        citing a line number in a comment, which is precisely how the earlier
        mirror was free to go stale.

        ``timeout_method`` is asserted alongside it because the entire cost model
        above rests on it: under thread mode a breach ``os._exit()``s the xdist
        worker instead of failing the test.  If someone switches back to signal
        mode, the rationale for every mark this guard demands changes, and that
        should surface loudly here rather than be discovered later.
        """
        ini_options = _pytest_ini_options()

        assert ini_options['timeout'] == PYPROJECT_DEFAULT_TIMEOUT, (
            f'PYPROJECT_DEFAULT_TIMEOUT ({PYPROJECT_DEFAULT_TIMEOUT}) no longer '
            f"mirrors [tool.pytest.ini_options].timeout ({ini_options['timeout']}) "
            f'in {_ORCH_PYPROJECT}. Update the constant in '
            'orchestrator/tests/_orch_helpers.py -- and re-check that '
            'WHOLE_TREE_SCAN_TEST_TIMEOUT, which is derived from it, still '
            'clears the measured worst case.'
        )
        assert ini_options['timeout_method'] == 'thread', (
            'timeout_method is no longer "thread" in '
            f'{_ORCH_PYPROJECT} (got {ini_options["timeout_method"]!r}). The '
            'whole rationale for this guard -- a breach os._exit()s the xdist '
            'worker rather than failing the test, and --max-worker-restart=0 '
            'turns that into a truncated whole-suite run -- is specific to '
            'thread mode. Revisit this module before changing it.'
        )

    def test_whole_tree_scan_timeout_clears_default_with_margin(self) -> None:
        """``WHOLE_TREE_SCAN_TEST_TIMEOUT`` must clear the default several times over.

        Measured basis, all first-hand except where the task record is cited:

        * 8.25s / 6.70s / 6.46s per call unloaded and serial (``-n0``) on a
          32-core box for test_merge_queue_reachback_patch_guard,
          test_event_loop_antipattern_guard and
          test_serial_merge_worker_import_guard respectively;
        * 17.85 / 21.32 / 30.75s per call for test_serial_merge_worker_import_guard
          at loadavg 120-176 (task 4215's record) -- ~4.8x its unloaded figure;
        * xdist worker deaths observed at loadavg 250-423 (esc-3980-1,
          esc-3787-1), i.e. past the 60s default.

        A 5x multiple leaves ~36x headroom over the unloaded worst case and ~10x
        over the measured-under-load worst case.  Asserted as ``>=`` rather than
        ``==`` so raising the constant later is never blocked by this test --
        the never-narrow polarity the neighbouring shared timeouts use.
        """
        assert WHOLE_TREE_SCAN_TEST_TIMEOUT >= 5 * PYPROJECT_DEFAULT_TIMEOUT, (
            f'WHOLE_TREE_SCAN_TEST_TIMEOUT ({WHOLE_TREE_SCAN_TEST_TIMEOUT}) must '
            f'clear 5x the pyproject default ({5 * PYPROJECT_DEFAULT_TIMEOUT}s). '
            'A whole-tree AST sweep measured 30.75s at loadavg 120-176 and the '
            'xdist worker deaths this guard exists to prevent were seen at '
            'loadavg 250-423; anything tighter re-arms that cliff.'
        )


#: Path methods that expand a glob pattern over a directory tree.
_GLOB_METHODS = frozenset({'glob', 'rglob'})


def _scans_whole_tree_py(source: str) -> bool:
    """True if *source* sweeps a directory tree for Python files.

    Matches an :class:`ast.Call` whose ``func`` is an :class:`ast.Attribute`
    with ``attr`` in ``{'glob', 'rglob'}`` and whose FIRST positional argument
    is a string literal that both contains ``*`` and ends in ``.py`` -- i.e. a
    pattern that expands to "every Python file under here" rather than to one
    named file or to some other suffix.

    DELIBERATE SCOPE, and why each half of it is drawn where it is:

    * **AST, not a text grep** -- the same justification the sibling guards
      give (test_raw_semaphore_access_guard.py:52-54,
      test_prune_chokepoint_guard.py). A docstring or comment merely
      MENTIONING ``rglob('*.py')`` cannot trip it, which matters more here
      than usual: this module's own docstring is full of such mentions, and so
      are the per-file comments step-6 adds to the twelve files it marks.

    * **Matched on the ATTRIBUTE name only**, never on the receiver. All three
      scan-root idioms in this directory therefore hit the same code path --
      ``_TESTS_DIR.rglob(...)``, ``_SRC_DIR.rglob(...)``, ``REPO_ROOT.rglob(...)``,
      ``_orchestrator_src_root().rglob(...)`` -- with no allowlist of receiver
      names to keep in sync.

    * **Wildcard required.** ``glob('conftest.py')`` names at most one file
      per directory; the hazard being guarded is ``ast.parse`` over hundreds
      of modules, not a lookup.

    * **``.py`` suffix required.** ``glob('*/src/*')`` (a real pattern, at
      test_killpg_frozen_pgid_guard.py:433-434) walks directories, not Python
      sources. Restricting to ``.py`` is what keeps that file's genuine
      ``rglob('*.py')`` at :447 caught without its directory walks being
      misread as three more scans.

    KNOWN LIMITATION, stated rather than papered over: an f-string, a
    variable, or a pattern built at runtime is NOT matched, and neither is a
    sweep whose per-file cost comes from something other than a ``.py`` glob.
    This detector is therefore a FLOOR on family coverage, not a proof of
    totality -- the same conservative-under-demand stance task 3492's auditor
    took ("it can under-demand ... but must never over-demand"). The
    anti-vacuity assertion in the family invariant below is what stops that
    concession from quietly becoming total.

    Fails SOFT: an unparseable *source* yields ``False`` rather than raising,
    matching test_prune_chokepoint_guard.py:138-141 and
    test_raw_semaphore_access_guard.py:76-79. A file mid-edit, or a fixture
    that is malformed on purpose, must not turn this guard red.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in _GLOB_METHODS):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        if first.value.endswith('.py') and '*' in first.value:
            return True
    return False


# ---------------------------------------------------------------------------
# _scans_whole_tree_py(source) -- inline-fixture unit tests.
#
# Same shape as the sibling guards this module polices
# (test_raw_semaphore_access_guard.py, test_prune_chokepoint_guard.py): a pure
# `_find_X(source)` detector exercised against synthetic snippets, so each
# branch is pinned directly instead of only ever being reached via the real
# tree -- which goes green by construction once step-6 lands and would
# otherwise leave the detector's negative cases untested forever.
# ---------------------------------------------------------------------------


def test_detector_flags_rglob_py() -> None:
    """The canonical whole-tree sweep: ``rglob('*.py')``."""
    source = (
        "def scan(root):\n"
        "    return sorted(root.rglob('*.py'))\n"
    )
    assert _scans_whole_tree_py(source) is True


def test_detector_flags_glob_py() -> None:
    """Non-recursive ``glob('*.py')`` counts too -- ``orchestrator/src`` alone
    is already ~250 files, and the cost model is per-file ``ast.parse``, not
    recursion depth."""
    source = (
        "def scan(root):\n"
        "    return sorted(root.glob('*.py'))\n"
    )
    assert _scans_whole_tree_py(source) is True


def test_detector_ignores_non_py_directory_sweep() -> None:
    """``glob('*/src/*')`` is NOT a Python-source sweep.

    This is the real pattern at test_killpg_frozen_pgid_guard.py:433-434, and
    it is what stops the detector reading that file's directory walks as the
    thing being guarded -- while its genuine ``rglob('*.py')`` at :447 is still
    caught by the case above.
    """
    source = (
        "def packages(root):\n"
        "    return sorted(root.glob('*/src/*'))\n"
    )
    assert _scans_whole_tree_py(source) is False


def test_detector_ignores_non_py_suffix() -> None:
    """A wildcard sweep over some other suffix is not this hazard."""
    source = (
        "def fixtures(root):\n"
        "    return sorted(root.rglob('*.txt'))\n"
    )
    assert _scans_whole_tree_py(source) is False


def test_detector_ignores_literal_filename_without_wildcard() -> None:
    """``glob('conftest.py')`` names ONE file per directory, not a tree sweep.

    The wildcard requirement is what separates "find this specific file" from
    "parse every module in the repo".
    """
    source = (
        "def conftests(root):\n"
        "    return sorted(root.glob('conftest.py'))\n"
    )
    assert _scans_whole_tree_py(source) is False


def test_detector_ignores_non_literal_pattern() -> None:
    """A variable pattern is unknowable statically, so it is NOT flagged.

    Fail-soft direction, matching the fail-soft ``SyntaxError`` polarity below:
    this guard is a FLOOR on coverage, never a claim of totality (see
    ``_scans_whole_tree_py``'s "known limitation" note).
    """
    source = (
        "def scan(root, pattern):\n"
        "    return sorted(root.rglob(pattern))\n"
    )
    assert _scans_whole_tree_py(source) is False


def test_detector_ignores_zero_arg_glob_without_raising() -> None:
    """``rglob()`` with no args must return False, not IndexError.

    Not valid at runtime, but it is valid SYNTAX -- and a detector that blows
    up on a half-typed call would turn every unrelated guard red mid-edit.
    """
    source = (
        "def broken(root):\n"
        "    return sorted(root.rglob())\n"
    )
    assert _scans_whole_tree_py(source) is False


def test_detector_ignores_module_with_no_glob_at_all() -> None:
    """The overwhelmingly common case: a module that never sweeps anything."""
    source = (
        "import json\n"
        "\n"
        "def load(path):\n"
        "    return json.loads(path.read_text())\n"
    )
    assert _scans_whole_tree_py(source) is False


def test_detector_ignores_docstring_mention() -> None:
    """A docstring or comment merely MENTIONING the pattern is not flagged --
    proves the detector is AST-based, not a text grep. This module's own
    docstring is full of such mentions."""
    source = (
        '"""Sweeps the tree with rglob(\'*.py\') and parses each file."""\n'
        "# root.rglob('*.py') -- example only, not a real call\n"
    )
    assert _scans_whole_tree_py(source) is False


def test_detector_fails_soft_on_syntax_error() -> None:
    """An unparseable file yields False rather than propagating.

    Deliberately the polarity of the sibling guards (test_prune_chokepoint_guard.py:138-141,
    test_raw_semaphore_access_guard.py:76-79 both ``except SyntaxError: return []``)
    and deliberately NOT test_marker_registration_drift.py's loud polarity: a
    mid-edit or intentionally-malformed fixture file under this directory must
    not turn a timeout-coverage guard red.
    """
    assert _scans_whole_tree_py('def f(:\n') is False


# ---------------------------------------------------------------------------
# Required fixture: demonstrate fail-on-new.
#
# The real tree is green by construction after step-6, so without this the
# detector could silently rot into always-False and the family invariant would
# pass vacuously. (The invariant carries its own anti-vacuity floor as well.)
# ---------------------------------------------------------------------------


def test_detector_flags_a_synthetic_new_scanner() -> None:
    """A brand-new whole-tree scanner -- the 13th member of the family -- is
    caught by the detector, demonstrated against a synthetic module rather
    than waiting for a real one to be added and crash someone's verify run."""
    synthetic_source = (
        "import ast\n"
        "from pathlib import Path\n"
        "\n"
        "_TESTS_DIR = Path(__file__).parent\n"
        "\n"
        "def test_some_new_invariant() -> None:\n"
        "    for py_file in sorted(_TESTS_DIR.rglob('*.py')):\n"
        "        ast.parse(py_file.read_text(encoding='utf-8'))\n"
    )
    assert _scans_whole_tree_py(synthetic_source) is True, (
        'the detector must flag a newly-added whole-tree scanner; if this '
        'fails, the family invariant below is passing vacuously'
    )
