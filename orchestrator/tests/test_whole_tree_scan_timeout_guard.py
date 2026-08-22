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
from pathlib import Path

import pytest
from _orch_helpers import (
    ORCH_PYPROJECT,
    PYPROJECT_DEFAULT_TIMEOUT,
    WHOLE_TREE_SCAN_TEST_TIMEOUT,
)

from orchestrator.pytest_markers import (
    _marker_name,
    _pytestmark_value,
    module_level_marker_names,
)

# This module is ITSELF a member of the family it polices -- the invariant
# below rglob()s every *.py under this directory and ast.parse()s each one --
# so it carries the very mark it demands instead of exempting itself.  The
# deliberate omission of the sibling guards' `_THIS_FILE`/`continue` skip-self
# idiom is spelled out in that test's docstring; skipping itself would exempt
# the one file most certain to need the mark.
pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

# This directory, resolved from THIS FILE and never from the process CWD:
# merge-verify runs pytest from the orchestrator/ cwd while a plain
# `pytest orchestrator/tests` runs from the repo root, and the census must come
# out identical under both.  Same idiom as
# test_marker_registration_drift.py:597-601.
_TESTS_DIR = Path(__file__).resolve().parent

# Anti-vacuity FLOORS, not equalities -- 535 files and 12 scanners measured at
# authorship time -- so the guard survives the tree growing while still failing
# loudly if the sweep itself ever breaks (a wrong _TESTS_DIR, a read that
# silently yields nothing, a detector rotted to always-False).  The house
# pattern for exactly this risk: test_marker_registration_drift.py:602-604's
# _MIN_EXPECTED_TEST_FILES, test_killpg_frozen_pgid_guard.py's measured file
# floor, test_serial_merge_worker_import_guard.py::test_allowlist_has_no_stale_entries.
_MIN_EXPECTED_TEST_FILES = 400
_MIN_EXPECTED_SCANNERS = 10

# Worst per-call wall clock MEASURED for a member of this family under REAL
# load: 30.75s for test_serial_merge_worker_import_guard at loadavg 120-176
# (task 4215's record; ~4.8x its 6.46s unloaded figure).  Named rather than
# left in prose so the floor below is anchored to a measurement instead of only
# to a ratio against a setting that can itself move.
_MEASURED_UNDER_LOAD_WORST_CASE = 30.75

# Headroom demanded over that measurement.  8x rather than 2x because the
# xdist worker deaths this guard exists to prevent were observed at loadavg
# 250-423 -- one further inflation step PAST the load at which the 30.75s was
# taken -- so the ceiling has to clear a figure nobody has managed to measure.
_REQUIRED_HEADROOM_FACTOR = 8

# ABSOLUTE floor in seconds, deliberately independent of
# PYPROJECT_DEFAULT_TIMEOUT.  WHOLE_TREE_SCAN_TEST_TIMEOUT is DERIVED
# (`5 * PYPROJECT_DEFAULT_TIMEOUT`), which is right for tracking the hazard
# UPWARD but would also track the ini default DOWNWARD in silence: tightening
# `[tool.pytest.ini_options].timeout` to 20s shrinks the family ceiling to
# 100s -- inside ~3x of the measured-under-load worst case, and well inside the
# further inflation seen at loadavg 250-423 -- while a ratio-only assertion
# (`>= 5 * PYPROJECT_DEFAULT_TIMEOUT`) stays green because it is an identity.
# This floor is what makes that scenario fail loudly.  Never-narrow.
_ABSOLUTE_FLOOR_SECONDS = 300

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

        TWO INDEPENDENT ASSERTIONS, because the ratio alone pins nothing.
        ``WHOLE_TREE_SCAN_TEST_TIMEOUT`` is *defined* as
        ``5 * PYPROJECT_DEFAULT_TIMEOUT``, so a ratio-only check is an identity
        that cannot fail -- and worse, the derivation tracks the ini default
        DOWNWARD in silence: tightening ``[tool.pytest.ini_options].timeout`` to
        20s would auto-shrink this ceiling to 100s with both this test and
        :meth:`test_pyproject_default_timeout_mirrors_pyproject` still green.
        The ABSOLUTE floor is therefore asserted first, and the floor itself is
        justified against the MEASUREMENT rather than written as a bare literal.
        """
        assert (
            _ABSOLUTE_FLOOR_SECONDS
            >= _MEASURED_UNDER_LOAD_WORST_CASE * _REQUIRED_HEADROOM_FACTOR
        ), (
            f'the absolute floor ({_ABSOLUTE_FLOOR_SECONDS}s) no longer clears '
            f'{_REQUIRED_HEADROOM_FACTOR}x the measured-under-load worst case '
            f'({_MEASURED_UNDER_LOAD_WORST_CASE}s = '
            f'{_MEASURED_UNDER_LOAD_WORST_CASE * _REQUIRED_HEADROOM_FACTOR}s). '
            'Either a slower measurement has landed or the floor was lowered; '
            'raise the floor rather than relaxing this arithmetic.'
        )
        assert WHOLE_TREE_SCAN_TEST_TIMEOUT >= _ABSOLUTE_FLOOR_SECONDS, (
            f'WHOLE_TREE_SCAN_TEST_TIMEOUT ({WHOLE_TREE_SCAN_TEST_TIMEOUT}) has '
            f'fallen below the absolute floor ({_ABSOLUTE_FLOOR_SECONDS}s). It '
            f'is derived as 5 * PYPROJECT_DEFAULT_TIMEOUT '
            f'({PYPROJECT_DEFAULT_TIMEOUT}), so the likeliest cause is that the '
            "pyproject's per-test default was TIGHTENED and dragged this "
            'ceiling down with it. The family ceiling must stay anchored to the '
            f'measured cost ({_MEASURED_UNDER_LOAD_WORST_CASE}s per call at '
            'loadavg 120-176, with worker deaths at loadavg 250-423), not to '
            'the setting it exists to clear -- pin it explicitly rather than '
            'lowering this floor.'
        )
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


#: Names that resolve to ``WHOLE_TREE_SCAN_TEST_TIMEOUT`` at import time.  A
#: bare ``WHOLE_TREE_SCAN_TEST_TIMEOUT`` (the house `from _orch_helpers import`
#: idiom, used by all 13 members) and the dotted
#: ``_orch_helpers.WHOLE_TREE_SCAN_TEST_TIMEOUT`` both land here, since only the
#: trailing name is compared.
_SANCTIONED_CEILING_NAMES = frozenset({'WHOLE_TREE_SCAN_TEST_TIMEOUT'})


def _timeout_call_arg(call: ast.Call) -> ast.expr | None:
    """The seconds expression a ``pytest.mark.timeout(...)`` *call* pins.

    Both spellings pytest-timeout accepts are read: the positional
    ``timeout(300)`` and the keyword ``timeout(timeout=300)``.  A call with
    neither (``timeout()``, or one passing only ``method=``) yields None.
    """
    if call.args:
        return call.args[0]
    for keyword in call.keywords:
        if keyword.arg == 'timeout':
            return keyword.value
    return None


def _module_level_timeout_ceiling(source: str) -> float | None:
    """Seconds pinned by *source*'s module-level ``pytest.mark.timeout(...)``, if knowable.

    WHY THIS EXISTS SEPARATELY from :func:`module_level_marker_names`: that
    helper answers "is the marker NAME present", which is the wrong question on
    its own.  A new scanner added with ``pytestmark = pytest.mark.timeout(30)``
    carries the name and would sail through a name-only check while sitting
    BELOW the very 60s cliff this module exists to clear -- the guard's own
    remediation text says to use ``WHOLE_TREE_SCAN_TEST_TIMEOUT``, and until
    this function existed nothing enforced it.

    RESOLUTION, deliberately tiny:

    * a numeric literal (``timeout(30)``) resolves to itself;
    * a name resolving to ``WHOLE_TREE_SCAN_TEST_TIMEOUT`` -- bare or dotted,
      compared on the trailing name only -- resolves to that constant's real
      runtime value, so the family's own spelling is understood without
      hard-coding 300 here;
    * ANY other expression (a different constant, arithmetic, an f-string) is
      UNKNOWABLE and yields None.

    None means "no opinion", never "too small": the caller must not fail on it.
    This is the same fail-soft polarity as :func:`_scans_whole_tree_py`, and for
    the same reason -- an unparseable or unfamiliar module must not turn a
    timeout-coverage guard red.  The consequence, stated rather than hidden: a
    scanner that pins a small ceiling through an indirection this grammar
    cannot follow is NOT caught.  Like the detector, this is a floor.

    If several module-level ``timeout`` marks are present (pathological, but
    legal), the SMALLEST resolvable one wins -- the ambiguity is real and the
    conservative reading is the one that surfaces it.

    ``_pytestmark_value`` and ``_marker_name`` are imported from
    :mod:`orchestrator.pytest_markers` rather than re-derived, for the same
    reason the family invariant imports ``module_level_marker_names``: the
    grammar of a module-level ``pytestmark`` (``Assign`` vs ``AnnAssign``,
    last-assignment-wins, list/tuple element forms) belongs in exactly one
    place, and a rename there should break this import loudly at collection
    rather than let two readings of the same syntax drift apart.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None

    value: ast.expr | None = None
    for statement in tree.body:
        bound = _pytestmark_value(statement)
        if bound is not None:
            value = bound
    if value is None:
        return None

    elements = list(value.elts) if isinstance(value, ast.List | ast.Tuple) else [value]
    resolved: list[float] = []
    for element in elements:
        if not isinstance(element, ast.Call) or _marker_name(element) != 'timeout':
            continue
        arg = _timeout_call_arg(element)
        if arg is None:
            continue
        if (
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, int | float)
            and not isinstance(arg.value, bool)
        ):
            resolved.append(float(arg.value))
            continue
        name: str | None = None
        if isinstance(arg, ast.Name):
            name = arg.id
        elif isinstance(arg, ast.Attribute):
            name = arg.attr
        if name in _SANCTIONED_CEILING_NAMES:
            resolved.append(float(WHOLE_TREE_SCAN_TEST_TIMEOUT))
    return min(resolved) if resolved else None


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


# ---------------------------------------------------------------------------
# _module_level_timeout_ceiling(source) -- inline-fixture unit tests.
#
# The value half of the invariant. Every real member of the family spells its
# mark `pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)`, so the tree exercises
# exactly ONE of these branches; the offending spellings below can only be
# pinned synthetically, which is precisely why they are pinned here.
# ---------------------------------------------------------------------------


def test_ceiling_reads_a_bare_numeric_literal() -> None:
    """``pytest.mark.timeout(30)`` -- the hole this function closes.

    A name-only check reads this module as "has a timeout mark" and passes it
    green while it sits BELOW the 60s default the family exists to clear.
    """
    source = 'import pytest\npytestmark = pytest.mark.timeout(30)\n'
    assert _module_level_timeout_ceiling(source) == 30.0


def test_ceiling_resolves_the_sanctioned_constant_name() -> None:
    """The house spelling resolves to the constant's real runtime value."""
    source = (
        'import pytest\n'
        'from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT\n'
        'pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)\n'
    )
    assert _module_level_timeout_ceiling(source) == float(WHOLE_TREE_SCAN_TEST_TIMEOUT)


def test_ceiling_resolves_the_dotted_constant_name() -> None:
    """``_orch_helpers.WHOLE_TREE_SCAN_TEST_TIMEOUT`` resolves too.

    No file in the family uses the dotted form today (130 use the bare-name
    import idiom, zero use this one), but reading it costs one branch and
    saves a future author a false red.
    """
    source = (
        'import pytest\n'
        'import _orch_helpers\n'
        'pytestmark = pytest.mark.timeout(_orch_helpers.WHOLE_TREE_SCAN_TEST_TIMEOUT)\n'
    )
    assert _module_level_timeout_ceiling(source) == float(WHOLE_TREE_SCAN_TEST_TIMEOUT)


def test_ceiling_reads_the_keyword_spelling() -> None:
    """pytest-timeout accepts ``timeout(timeout=N)``; so does this."""
    source = 'import pytest\npytestmark = pytest.mark.timeout(timeout=30)\n'
    assert _module_level_timeout_ceiling(source) == 30.0


def test_ceiling_reads_a_timeout_inside_a_pytestmark_list() -> None:
    """The list form is legal and must not hide the value."""
    source = (
        'import pytest\n'
        'pytestmark = [pytest.mark.slow, pytest.mark.timeout(30)]\n'
    )
    assert _module_level_timeout_ceiling(source) == 30.0


def test_ceiling_takes_the_smallest_of_several_timeout_marks() -> None:
    """Pathological but legal; the conservative reading surfaces the ambiguity."""
    source = (
        'import pytest\n'
        'pytestmark = [pytest.mark.timeout(300), pytest.mark.timeout(30)]\n'
    )
    assert _module_level_timeout_ceiling(source) == 30.0


def test_ceiling_is_unknown_for_an_unresolvable_expression() -> None:
    """An unfamiliar constant yields None -- "no opinion", not "too small".

    Fail-soft, matching ``_scans_whole_tree_py``: this guard is a floor on
    coverage, and a module whose ceiling it cannot read must not go red for
    that reason alone.
    """
    source = (
        'import pytest\n'
        'from somewhere import SOME_OTHER_TIMEOUT\n'
        'pytestmark = pytest.mark.timeout(SOME_OTHER_TIMEOUT)\n'
    )
    assert _module_level_timeout_ceiling(source) is None


def test_ceiling_is_unknown_for_a_zero_arg_timeout_mark() -> None:
    """``timeout()`` pins nothing and must not IndexError."""
    source = 'import pytest\npytestmark = pytest.mark.timeout()\n'
    assert _module_level_timeout_ceiling(source) is None


def test_ceiling_is_unknown_when_there_is_no_timeout_mark() -> None:
    """A module-level ``pytestmark`` carrying some OTHER marker pins nothing."""
    source = 'import pytest\npytestmark = pytest.mark.asyncio\n'
    assert _module_level_timeout_ceiling(source) is None


def test_ceiling_is_unknown_with_no_pytestmark_at_all() -> None:
    """The common case. None here is what makes the NAME check the primary gate."""
    assert _module_level_timeout_ceiling('import pytest\n') is None


def test_ceiling_ignores_a_function_local_pytestmark() -> None:
    """Only ``tree.body`` is walked, inheriting ``_pytestmark_value``'s contract.

    A ``pytestmark`` bound inside a function body applies to nothing, so
    reading it would be worse than reading nothing.
    """
    source = (
        'import pytest\n'
        'def f():\n'
        '    pytestmark = pytest.mark.timeout(30)\n'
        '    return pytestmark\n'
    )
    assert _module_level_timeout_ceiling(source) is None


def test_ceiling_fails_soft_on_syntax_error() -> None:
    """Same polarity as the detector: a mid-edit file yields None, not a raise."""
    assert _module_level_timeout_ceiling('def f(:\n') is None


def test_a_synthetic_new_scanner_marked_too_tight_is_an_offender() -> None:
    """END TO END on the two halves: detected as a scanner AND read as too tight.

    The 13th family member added with ``pytest.mark.timeout(30)`` is the
    scenario this pair of checks exists for -- it satisfies a name-only check
    while sitting below even the 60s default. Proven synthetically because the
    real tree is green by construction and cannot demonstrate it.
    """
    synthetic_source = (
        'import ast\n'
        'import pytest\n'
        'from pathlib import Path\n'
        '\n'
        'pytestmark = pytest.mark.timeout(30)\n'
        '\n'
        '_TESTS_DIR = Path(__file__).parent\n'
        '\n'
        'def test_some_new_invariant() -> None:\n'
        "    for py_file in sorted(_TESTS_DIR.rglob('*.py')):\n"
        "        ast.parse(py_file.read_text(encoding='utf-8'))\n"
    )
    assert _scans_whole_tree_py(synthetic_source) is True
    assert 'timeout' in module_level_marker_names(synthetic_source), (
        'the name-only check PASSES this module -- which is the whole point'
    )
    ceiling = _module_level_timeout_ceiling(synthetic_source)
    assert ceiling is not None and ceiling < WHOLE_TREE_SCAN_TEST_TIMEOUT, (
        'a scanner pinned at 30s must be readable as below the family ceiling; '
        'otherwise the value half of the invariant below is dead code'
    )


# ---------------------------------------------------------------------------
# The family invariant: the actual ratchet.
#
# Allowlist-free and TOTAL -- every whole-tree scanner under this directory
# must carry the mark, with no frozen residual to rot. The census is recomputed
# from source on every run, so the Nth new scanner fails at commit time instead
# of os._exit()ing an xdist worker in someone else's verify.
# ---------------------------------------------------------------------------


def test_whole_tree_scanners_carry_module_level_timeout_mark() -> None:
    """Every whole-tree ``*.py`` scanner here needs a MODULE-LEVEL timeout mark
    AT the family ceiling.

    TWO HALVES, because either alone is insufficient. The NAME half
    (:func:`module_level_marker_names`) catches a scanner with no mark; the
    VALUE half (:func:`_module_level_timeout_ceiling`) catches one marked
    ``pytest.mark.timeout(30)``, which carries the name yet sits BELOW even the
    60s default -- so a name-only check would pass it green while it re-arms the
    exact cliff this module exists to clear. Only a value the grammar can
    actually resolve can offend: an unresolvable expression is skipped, keeping
    the guard a floor rather than a proof.

    Why module-level rather than "a timeout mark somewhere in the file": only a
    module-level ``pytestmark`` is a statically sound LOWER BOUND on every
    collected item's marker set -- the documented contract of
    :func:`orchestrator.pytest_markers.module_level_marker_names`, which is
    IMPORTED here rather than re-derived. A decorator-aware sweep (e.g.
    test_marker_registration_drift.py's ``_applied_marker_names``) is an
    explicit UPPER bound: a file with three tests where only one is decorated
    would read as "has timeout" while the other two sit at the 60s cliff, which
    is precisely the defect this guard exists to stop. An existing tighter
    per-test decorator still WINS where present -- verified empirically: with
    ``pytestmark = pytest.mark.timeout(300)`` plus ``@pytest.mark.timeout(7)``
    on one test, ``get_closest_marker('timeout').args`` is ``(7,)`` for the
    decorated test and ``(300,)`` for the bare one -- so the module-level mark
    acts purely as a FLOOR and narrows nothing.

    NO SKIP-SELF, deliberately, and stated here because it inverts a convention
    a reviewer will expect. Five sibling guards (test_prune_chokepoint_guard.py:164,
    test_event_loop_antipattern_guard.py:81, and the serial-merge-worker,
    reachback and lock-release guards) carry a ``_THIS_FILE``/``continue``
    skip-self because their forbidden pattern necessarily appears in their own
    detector code, making self-inclusion a guaranteed false positive. The
    opposite holds here: this module genuinely IS a whole-tree AST scanner and
    genuinely IS exposed to the same worker-death cliff, so skipping itself
    would exempt the one file most certain to need the mark. Its own
    ``pytestmark`` is at the top of this file.
    """
    scanners: list[Path] = []
    offenders: list[str] = []
    unreadable: list[str] = []
    examined = 0

    for py_file in sorted(_TESTS_DIR.rglob('*.py')):
        # Fail-soft on the READ for the same reason _scans_whole_tree_py fails
        # soft on the PARSE: a non-UTF-8 source, a deliberately-malformed
        # encoding fixture or a broken symlink under this directory would
        # otherwise raise UnicodeDecodeError/OSError straight out of a
        # TIMEOUT-COVERAGE check, turning it red for a reason that has nothing
        # to do with timeout coverage -- the very class of misattributed
        # failure this module exists to prevent. Skipped files are NOT counted
        # as examined (they were not), and the count is surfaced below so a
        # sweep that silently stops reading anything cannot hide here.
        try:
            source = py_file.read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError):
            unreadable.append(py_file.name)
            continue
        examined += 1
        if not _scans_whole_tree_py(source):
            continue
        scanners.append(py_file)
        if 'timeout' not in module_level_marker_names(source):
            offenders.append(f'{py_file.name}: no module-level pytestmark timeout')
            continue
        # NAME present, but at what value? A mark of `timeout(30)` satisfies the
        # check above while sitting BELOW even the 60s default -- see
        # _module_level_timeout_ceiling. None means unresolvable, which is NOT
        # an offence (fail-soft; the guard stays a floor).
        ceiling = _module_level_timeout_ceiling(source)
        if ceiling is not None and ceiling < WHOLE_TREE_SCAN_TEST_TIMEOUT:
            offenders.append(
                f'{py_file.name}: module-level timeout pins {ceiling:g}s, below '
                f'the {WHOLE_TREE_SCAN_TEST_TIMEOUT}s family ceiling'
            )

    assert examined >= _MIN_EXPECTED_TEST_FILES, (
        f'only {examined} .py files examined under {_TESTS_DIR} (expected at '
        f'least {_MIN_EXPECTED_TEST_FILES}; {len(unreadable)} skipped as '
        f'unreadable: {sorted(unreadable)}) -- the sweep itself is broken, so '
        'this guard would pass vacuously rather than because the family is '
        'clean.'
    )
    assert len(scanners) >= _MIN_EXPECTED_SCANNERS, (
        f'only {len(scanners)} whole-tree scanner(s) detected among {examined} '
        f'files (expected at least {_MIN_EXPECTED_SCANNERS}) -- '
        '_scans_whole_tree_py has probably stopped matching, so this guard '
        'would pass vacuously. Found: '
        f'{sorted(f.name for f in scanners)}'
    )

    if offenders:
        offender_list = '\n  '.join(offenders)
        raise AssertionError(
            f'{len(offenders)} whole-tree AST-scanning test module(s) under '
            f'{_TESTS_DIR.name}/ lack a module-level timeout mark AT THE FAMILY '
            'CEILING (either no mark at all, or one pinned lower). Add\n\n'
            '    from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT\n'
            '    pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)\n\n'
            'at module level in each. These modules rglob() every *.py in the '
            'repo and ast.parse() each one; under `-n auto` that has been '
            'MEASURED at 30.75s per call at loadavg 120-176, against the 60s '
            f'default in {ORCH_PYPROJECT.name}. Exceeding it does NOT fail the '
            "test: pytest-timeout's thread method os._exit()s the whole xdist "
            'worker, and --max-worker-restart=0 then truncates the ENTIRE suite '
            'run and reports the failure against some innocent guard that '
            'merely shared the dead worker (esc-3980-1, esc-3787-1).\n\n'
            'Use WHOLE_TREE_SCAN_TEST_TIMEOUT rather than a hand-picked number: '
            'a smaller literal carries the marker NAME while leaving the test at '
            'or under the cliff, which is why the value is checked and not just '
            'the name.'
            f'\n\nOffending scanners:\n  {offender_list}'
        )
