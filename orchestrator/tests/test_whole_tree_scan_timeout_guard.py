"""Meta-guard: every whole-tree AST-scanning test in this directory must carry a
module-level ``pytest.mark.timeout`` (task 4215).

THE FAILURE MODE, END TO END.  Whole-tree guard tests under ``orchestrator/tests/``
sweep the ENTIRE repo -- ``rglob('*.py')`` over ~500 files -- and ``ast.parse``
each one.  Unloaded and serial that costs 6-9s per call here; the orchestrator
suite however runs under ``-n auto`` (``[tool.pytest.ini_options].addopts`` in
``orchestrator/pyproject.toml``), so on a 32-core box those sweeps run
concurrently with everything else and the same call has been MEASURED at
17.85 / 21.32 / 30.75s at loadavg 120-176 -- a ~4.8x load inflation.  Worker
deaths were then observed at loadavg 250-423, one further inflation step past
the 60s ``[tool.pytest.ini_options].timeout`` default THEN IN FORCE.  That
default has moved since, for reasons of its own (shared/pyproject.toml); the
marks this module demands guarantee the family never drops BELOW it, which is
what a floor is for.  The value history of both lives in
plans/pytest-per-test-timeout-measurement-2026-09-17.md.

Until task 6354 what made the breach so expensive was the two settings around
it:

* ``[tool.pytest.ini_options].timeout_method = "thread"`` meant a breach did
  NOT fail the test -- pytest-timeout's thread method ``os._exit()``ed the whole
  xdist worker ("node down: Not properly terminated");
* ``--max-worker-restart=0`` (in ``[tool.pytest.ini_options].addopts``, task
  1907) then declined to replace that worker, degrading the run to a TRUNCATED
  whole-suite session whose surviving failure named some innocent guard that
  merely happened to share the dead worker.

So a single slow tree-scan cost the whole verify run, and misattributed the
blame.  Escalations esc-3980-1 and esc-3787-1 are both instances.  Since task
6354 the method is ``signal``: a breach fails the one scanner, with its own
traceback, and the worker lives on.  That is still a false red on a
healthy-but-slow test, and the marks still prevent it, so the census below
stays as it is.

THE FIX the pyproject itself sanctions ("Slow tests opt out with
``@pytest.mark.timeout(N)``", the comment on
``[tool.pytest.ini_options].timeout``) is a module-level
``pytestmark`` on each scanner.  This module pins the two constants that mark
uses, and recomputes the census from source on every run so the Nth new guard
is caught at commit time rather than by another false red in verify.

THE CENSUS is every module that sweeps a tree for ``*.py`` itself plus every
module that consumes the one shared tree pass ``_tree_scan.py`` provides, and
each of them must be a member of the family task-role verify runs.  A consumer
carries the mark because the shared pass is billed to whichever consumer item
first requests it, and pins the shared xdist group so that only one worker
parses.

NOT widening the global ``timeout`` *as the remedy for THIS hazard*: doing it
for that reason would blunt the hang-catching ceiling for the other ~16000
tests to buy headroom only the family needs.  (The global default did later
move, but for a different hazard -- wall-clock CPU starvation false-redding a
shifting victim -- and the per-module marks stayed, because a sweep's cost and
a host's contention are separate things to size against.)

WHERE THE RATIONALE LIVES.  The mechanism is restated here because this is the
module a failing run points at, but the CANONICAL home -- the derivation of the
ceiling, the full measurement record, and the escalation references -- is the
``WHOLE_TREE_SCAN_TEST_TIMEOUT`` comment in
``shared/src/shared/whole_tree_guard_family.py``.  The modules that carry the
mark point THERE rather than repeating it, so retuning the ceiling or switching
``timeout_method`` is one edit plus this file, not one per member.
"""
from __future__ import annotations

import ast
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pytest
from _orch_helpers import (
    ORCH_PYPROJECT,
    PYPROJECT_DEFAULT_TIMEOUT,
    UNDER_LOAD_HEADROOM_FACTOR,
    VERIFY_CLI_PER_TEST_TIMEOUT,
)
from _tree_scan import (
    ORCHESTRATOR_TESTS,
    TESTS_DIR,
    TREE_PASS_FIXTURE,
    TREE_PASS_XDIST_GROUP,
    ParsedModule,
    TreeAnalysis,
    TreePass,
)
from shared.governed_exceptions import Debt, TaskRef, TicketRef, governed_exceptions
from shared.pytest_mark_grammar import marker_name, pytestmark_elements, scope_binding
from shared.testing_timeout_markers import SiteKind, timeout_marker_sites
from shared.whole_tree_guard_family import (
    WHOLE_TREE_SCAN_TEST_TIMEOUT,
    WHOLE_TREE_SCAN_TIMEOUT_NAME,
    is_whole_tree_guard_module,
    module_level_timeout_arguments,
)

from orchestrator.pytest_markers import module_level_marker_names

# This module is ITSELF a member of the family it polices -- its census is a
# consumer of the shared tree pass, so any of its items may be the one billed
# the pass -- and it carries the very mark and group it demands instead of
# exempting itself.  The deliberate omission of the sibling guards'
# `_THIS_FILE`/`continue` skip-self idiom is spelled out in that test's
# docstring; skipping itself would exempt the one file most certain to need it.
pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP),
]

# Anti-vacuity FLOORS, not equalities -- 535 files and 12 scanners measured at
# authorship time -- so the guard survives the tree growing while still failing
# loudly if the sweep itself ever breaks (a wrong TESTS_DIR, a read that
# silently yields nothing, a detector rotted to always-False).  The house
# pattern for exactly this risk:
# test_marker_registration_drift.py::_MIN_EXPECTED_TEST_FILES and
# test_killpg_frozen_pgid_guard.py's measured file floor.
_MIN_EXPECTED_TEST_FILES = 400
_MIN_EXPECTED_FAMILY = 10

# The shared tree pass's home sweeps by design but is a helper, not a test
# module: a pytestmark there binds no item, and its cost is billed to the
# consumer item that first requests the tree_pass fixture.
_SHARED_PASS_MODULE = '_tree_scan.py'

# Every whole-tree guard joins the shared pass unless it is named below.
_PERMANENT_PRIVATE_SWEEPS = MappingProxyType({
    'test_cited_test_class_drift.py': (
        'it sweeps the whole uv workspace and builds its phase-2 prefilter from '
        'phase-1 results, so it parses only the few files defining a cited class; '
        'one streaming pass cannot express a phase that depends on another'
    ),
})
_OUTSIDE_THE_SHARED_PASS = MappingProxyType({
    'test_merge_lane_ratchet.py': (
        'a complexipy whole-repo measurement through _merge_lane_census, not an '
        'AST sweep of a tree; task 6986 owns separating it from the family'
    ),
})

PERMANENT_PRIVATE_SWEEPS_DECLARATION = governed_exceptions(
    'orchestrator.tests.whole_tree_permanent_private_sweeps',
    'A whole-tree guard module that keeps its own private *.py sweep because the '
    'shared tree pass cannot express its analysis.',
    _PERMANENT_PRIVATE_SWEEPS.keys(),
    default=Debt(TicketRef('tkt_0RW782Z6RY3X8W8Q3NX8J1EKD0')),
    default_covers=1,
)
OUTSIDE_THE_SHARED_PASS_DECLARATION = governed_exceptions(
    'orchestrator.tests.whole_tree_outside_the_shared_pass',
    'A whole-tree guard family member that neither sweeps a tree privately nor '
    'consumes the shared tree pass.',
    _OUTSIDE_THE_SHARED_PASS.keys(),
    default=Debt(TaskRef(6986)),
    default_covers=1,
)

# Worst wall clock MEASURED for a member of this family under REAL load: a
# per-test TOTAL (setup + call + teardown, what a timeout mark bounds; almost all
# of it module-fixture setup) of 148.65s for test_merge_lane_ratchet.py::
# TestLanePatchTargets::test_real_tree_union_anchor, from task 3165's
# orchestrator verify (2026-10-08, `-n 8`, loadavg 64 -> 79).  The family is
# every module marked with WHOLE_TREE_SCAN_TEST_TIMEOUT, not only the ones the
# detector below finds.  The worst run is kept, never the freshest; a figure
# measured on code since replaced retires with that code.  Corpus:
# plans/pytest-per-test-timeout-measurement-2026-09-17.md.  That fixture is a
# whole-repo complexity measurement rather than an AST sweep of this tree, so
# one member's cost sets the ceiling for the whole family; task 6986 owns
# separating it.
_MEASURED_UNDER_LOAD_WORST_CASE = 148.65

# ABSOLUTE floor in seconds: the measured worst case above times
# UNDER_LOAD_HEADROOM_FACTOR, rounded up to a whole minute.  Deliberately
# independent of PYPROJECT_DEFAULT_TIMEOUT, and never-narrow.
#
# WHOLE_TREE_SCAN_TEST_TIMEOUT used to be DERIVED (`5 *
# PYPROJECT_DEFAULT_TIMEOUT`), which tracked the hazard UPWARD but also tracked
# the ini default DOWNWARD in silence: tightening
# `[tool.pytest.ini_options].timeout` to 20s would shrink the family ceiling to
# 100s -- well inside the measured-under-load worst case -- while a ratio-only
# assertion (`>= 5 * PYPROJECT_DEFAULT_TIMEOUT`) stayed green because it was an
# identity.  This floor was written to make that scenario fail loudly.  The
# derivation was dropped when the ini default was raised 60 -> 300 (for
# host-contention reasons of its own), because carrying the multiple forward
# would have set the family ceiling to 1500s on no measurement at all.
_ABSOLUTE_FLOOR_SECONDS = 1200

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

    def test_a_hung_test_dumps_its_stacks_below_the_verify_cap(self) -> None:
        """``faulthandler_timeout`` must fire below the tightest cap a test meets.

        The dump writes every thread's stack to STDERR, which an xdist worker
        inherits from the controller, and verify merges stderr into its log.
        The bound serves two hangs.

        A main thread blocked in a C call that never returns is one that
        ``timeout_method = "signal"`` cannot interrupt: the cap never fires,
        the worker hangs until verify's outer kill, and this dump is the only
        witness.  Any positive value below the cap lands it well inside the
        verify run.

        A hang the cap DOES interrupt (a main thread waiting on a deadlocked
        peer, say) fails with the main thread's traceback alone.
        pytest-timeout writes the other threads' stacks to the terminal, and
        on an xdist worker that is stdout, which execnet points at /dev/null.
        pytest also cancels the pending dump when the test ends.  So the peer's
        stack reaches the verify log only if this dump fires before the cap,
        and the tightest cap an UNMARKED test can meet is verify's CLI
        ``--timeout=300``, not this file's ini default.
        """
        ini_options = _pytest_ini_options()

        dump_after = ini_options['faulthandler_timeout']
        assert isinstance(dump_after, int | float), (
            f'faulthandler_timeout must be a TOML number, got {dump_after!r}'
        )
        assert 0 < dump_after < VERIFY_CLI_PER_TEST_TIMEOUT, (
            f'faulthandler_timeout ({dump_after}) must be positive and below '
            f'VERIFY_CLI_PER_TEST_TIMEOUT ({VERIFY_CLI_PER_TEST_TIMEOUT}), or a '
            "hang the cap interrupts fails without its other threads' stacks: "
            "pytest-timeout writes those to the xdist worker's discarded stdout, "
            'and the pending dump is cancelled with the test'
        )

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

        ``timeout_method`` is asserted alongside it because the cost model above
        rests on it.  Under signal mode a breach fails the one test, with its
        traceback, and the worker lives on; a breach inside a long C or Rust
        call (complexipy's analysis) is raised only once that call returns; and
        a C call that never returns is not interrupted at all, which is what
        ``faulthandler_timeout``'s dump is for.  If someone switches the method,
        the cost of every breach this guard's marks prevent changes, and that
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
        assert ini_options['timeout_method'] == 'signal', (
            'timeout_method is no longer "signal" in '
            f'{_ORCH_PYPROJECT} (got {ini_options["timeout_method"]!r}). This '
            "guard's cost model is signal's: a breach fails the one test, with "
            'its traceback, and the worker lives on; a breach inside a long C or '
            'Rust call (complexipy) is raised only once that call returns; and a '
            'C call that never returns is not interrupted, which is what '
            "faulthandler_timeout's dump is for. The trade is decided in "
            'tests/scripts/test_timeout_method_policy.py; revisit this module '
            'before changing the method.'
        )

    def test_whole_tree_scan_timeout_clears_default_with_margin(self) -> None:
        """``WHOLE_TREE_SCAN_TEST_TIMEOUT`` must clear the default several times over.

        Measured basis: ``_MEASURED_UNDER_LOAD_WORST_CASE``, whose corpus is the
        task 6556 addendum of plans/pytest-per-test-timeout-measurement-2026-09-17.md.

        The floor clears 8 x the anchor, and the constant clears both the floor
        and the ini default.  Asserted as ``>=`` rather than ``==`` so raising the
        constant later is never blocked by this test -- the never-narrow
        polarity the neighbouring shared timeouts use.

        THREE ASSERTIONS, each pinning something the others do not:

        1. the ABSOLUTE floor still clears the MEASUREMENT with the demanded
           headroom -- the arithmetic that justifies the literal floor rather
           than leaving it a bare number;
        2. the constant clears that floor -- the substantive check;
        3. the constant has not fallen below the ini default, which would turn
           the module-level marks from a floor into a tightening.

        Assertion 3 REPLACED a ``>= 5 * PYPROJECT_DEFAULT_TIMEOUT`` check on
        2026-09-12.  While the constant was *defined* as
        ``5 * PYPROJECT_DEFAULT_TIMEOUT`` that check was an identity that could
        not fail; the derivation was then dropped (the ini default rose 60 ->
        300 for host-contention reasons, and carrying the multiple would have
        set this ceiling to 1500s on no measurement), so the relationship worth
        pinning is the never-narrow one, which has teeth in both directions.
        """
        assert (
            _ABSOLUTE_FLOOR_SECONDS
            >= _MEASURED_UNDER_LOAD_WORST_CASE * UNDER_LOAD_HEADROOM_FACTOR
        ), (
            f'the absolute floor ({_ABSOLUTE_FLOOR_SECONDS}s) no longer clears '
            f'{UNDER_LOAD_HEADROOM_FACTOR}x the measured-under-load worst case '
            f'({_MEASURED_UNDER_LOAD_WORST_CASE}s = '
            f'{_MEASURED_UNDER_LOAD_WORST_CASE * UNDER_LOAD_HEADROOM_FACTOR}s). '
            'Either a slower measurement has landed or the floor was lowered; '
            'raise the floor rather than relaxing this arithmetic.'
        )
        assert WHOLE_TREE_SCAN_TEST_TIMEOUT >= _ABSOLUTE_FLOOR_SECONDS, (
            f'WHOLE_TREE_SCAN_TEST_TIMEOUT ({WHOLE_TREE_SCAN_TEST_TIMEOUT}) has '
            f'fallen below the absolute floor ({_ABSOLUTE_FLOOR_SECONDS}s). It '
            'is a LITERAL, derived from nothing: the `5 * '
            'PYPROJECT_DEFAULT_TIMEOUT` derivation was DROPPED on 2026-09-12 '
            "(this test's docstring says why), so do not read the two as "
            'tracking each other. What binds the constant is never-narrow '
            'against TWO independent bounds -- this floor, and the ini default '
            f'({PYPROJECT_DEFAULT_TIMEOUT}s, asserted separately below) -- so '
            'the likeliest cause of this failure is the constant being lowered '
            'by hand. The family ceiling must stay anchored to the measured '
            f'cost ({_MEASURED_UNDER_LOAD_WORST_CASE}s under load, with worker '
            'deaths at loadavg 250-423), not to whichever '
            'setting it happens to clear -- pin it explicitly rather than '
            'lowering this floor.'
        )
        assert WHOLE_TREE_SCAN_TEST_TIMEOUT >= PYPROJECT_DEFAULT_TIMEOUT, (
            f'WHOLE_TREE_SCAN_TEST_TIMEOUT ({WHOLE_TREE_SCAN_TEST_TIMEOUT}) has '
            f'fallen BELOW the pyproject default ({PYPROJECT_DEFAULT_TIMEOUT}s). '
            'The module-level marks this guard demands are meant to be a FLOOR '
            'that narrows nothing; below the default they would instead TIGHTEN '
            'every test in the family, which is the opposite of the intent. '
            'Raise the constant, or lower the ini default to match.'
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
      give (test_raw_semaphore_access_guard.py::_raw_semaphore_accesses,
      test_prune_chokepoint_guard.py). A docstring or comment merely
      MENTIONING ``rglob('*.py')`` cannot trip it, which matters more here
      than usual: this module's own docstring is full of such mentions.

    * **Matched on the ATTRIBUTE name only**, never on the receiver, so every
      scan-root idiom -- a constant (``REPO_ROOT.rglob(...)``), a parameter
      (``root.rglob(...)``) or a call's result (``_src_root().rglob(...)``) --
      hits the same code path, with no allowlist of receiver names to keep in
      sync.

    * **Wildcard required.** ``glob('conftest.py')`` names at most one file
      per directory; the hazard being guarded is ``ast.parse`` over hundreds
      of modules, not a lookup.

    * **``.py`` suffix required.** ``glob('*/src/*')`` (a real pattern, in
      _tree_scan.py::_package_root_python_files) walks directories,
      not Python sources. Restricting to ``.py`` is what keeps that same
      function's genuine ``glob('*.py')`` caught without its directory walks
      being misread as more scans.

    KNOWN LIMITATION, stated rather than papered over: an f-string, a
    variable, or a pattern built at runtime is NOT matched, and neither is a
    sweep whose per-file cost comes from something other than a ``.py`` glob.
    This detector is therefore a FLOOR on family coverage, not a proof of
    totality -- the same conservative-under-demand stance task 3492's auditor
    took ("it can under-demand ... but must never over-demand"). The
    anti-vacuity assertion in the family invariant below is what stops that
    concession from quietly becoming total.

    Fails SOFT: an unparseable *source* yields ``False`` rather than raising,
    matching test_prune_chokepoint_guard.py::_find_prune_argv_literals and
    test_raw_semaphore_access_guard.py::_raw_semaphore_accesses. A file
    mid-edit, or a fixture that is malformed on purpose, must not turn this
    guard red.
    """
    tree = _parsed(source)
    return tree is not None and _sweeps_whole_tree_in(tree)


def _parsed(source: str) -> ast.Module | None:
    """*source* parsed, or None when it is not valid Python: every detector here fails soft."""
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError):
        return None


def _sweeps_whole_tree_in(tree: ast.Module) -> bool:
    """:func:`_scans_whole_tree_py` over an already-parsed module."""
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


#: The module-level name a shared-pass consumer declares its analyses under,
#: the one ``_tree_scan.analyses_requested_by`` reads.
_TREE_ANALYSES_NAME = 'TREE_ANALYSES'


def _consumes_tree_pass(source: str) -> bool:
    """True if *source* binds a module-level ``TREE_ANALYSES`` or requests the ``tree_pass`` fixture.

    Either one makes the module a consumer of the shared pass in
    ``_tree_scan.py``.  Read from the AST, so a docstring or comment mention
    cannot trip it, and a ``TREE_ANALYSES`` bound inside a function declares
    nothing the fixture discovers.  The request is any ``def`` or ``async def``
    taking a parameter named ``tree_pass``, the spelling pytest injects a
    fixture by.  Fails soft, like :func:`_scans_whole_tree_py`.
    """
    tree = _parsed(source)
    return tree is not None and _consumes_tree_pass_in(tree)


def _consumes_tree_pass_in(tree: ast.Module) -> bool:
    """:func:`_consumes_tree_pass` over an already-parsed module."""
    if scope_binding(tree.body, _TREE_ANALYSES_NAME) is not None:
        return True
    return any(
        TREE_PASS_FIXTURE in _parameter_names(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    )


def _parameter_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    parameters = function.args
    return frozenset(
        parameter.arg
        for parameter in (*parameters.posonlyargs, *parameters.args, *parameters.kwonlyargs)
    )


def _trailing_name(expression: ast.expr | None) -> str | None:
    """The identifier a bare ``NAME`` or a dotted ``module.NAME`` ends in, else None."""
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        return expression.attr
    return None


#: The name that resolves to ``TREE_PASS_XDIST_GROUP`` at import time, bare or dotted.
_SANCTIONED_GROUP_NAME = 'TREE_PASS_XDIST_GROUP'


def _module_xdist_group(source: str) -> str | None:
    """The group *source*'s module-level ``pytestmark`` pins with ``pytest.mark.xdist_group``, if knowable.

    The group is read positionally or as ``name=``, as pytest-xdist reads it.
    A string literal resolves to itself, and a bare or dotted
    ``TREE_PASS_XDIST_GROUP`` to that constant's runtime value; anything else is
    None, as is a module with no module-level group.  Class-level marks do not
    count: the pass is shared only if every item of a consumer lands on the
    pinned worker.  Fails soft.
    """
    tree = _parsed(source)
    return None if tree is None else _module_xdist_group_in(tree)


def _module_xdist_group_in(tree: ast.Module) -> str | None:
    """:func:`_module_xdist_group` over an already-parsed module."""
    return next(
        (
            _xdist_group_name(element)
            for element in pytestmark_elements(tree.body)
            if marker_name(element) == 'xdist_group'
        ),
        None,
    )


def _xdist_group_name(element: ast.expr) -> str | None:
    if not isinstance(element, ast.Call):
        return None
    argument = element.args[0] if element.args else next(
        (keyword.value for keyword in element.keywords if keyword.arg == 'name'), None,
    )
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return argument.value
    if _trailing_name(argument) == _SANCTIONED_GROUP_NAME:
        return TREE_PASS_XDIST_GROUP
    return None


#: Each name that resolves to ``WHOLE_TREE_SCAN_TEST_TIMEOUT`` at import time,
#: with that constant's runtime value.  A bare ``WHOLE_TREE_SCAN_TEST_TIMEOUT``
#: (the house `from shared.whole_tree_guard_family import` idiom) and a dotted
#: ``<module>.WHOLE_TREE_SCAN_TEST_TIMEOUT`` both land here, since only the
#: trailing name is compared.  The spelling itself is
#: ``shared.whole_tree_guard_family``'s -- the same one its family predicate reads.
#: Module-level and class- or test-level marks resolve through this one table.
_SANCTIONED_CEILING_SECONDS = MappingProxyType(
    {WHOLE_TREE_SCAN_TIMEOUT_NAME: float(WHOLE_TREE_SCAN_TEST_TIMEOUT)},
)


def _module_level_timeout_ceiling(source: str) -> float | None:
    """Seconds pinned by *source*'s module-level ``pytest.mark.timeout(...)``, if knowable.

    WHY THIS EXISTS SEPARATELY from :func:`module_level_marker_names`: that
    helper answers "is the marker NAME present", which is the wrong question on
    its own.  A new scanner added with ``pytestmark = pytest.mark.timeout(30)``
    carries the name and would sail through a name-only check while sitting
    BELOW the family ceiling this module exists to enforce -- the guard's own
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

    The seconds expressions come from
    :func:`shared.whole_tree_guard_family.module_level_timeout_arguments` rather
    than a fold of this module's own, for the same reason the family invariant
    imports ``module_level_marker_names``: the grammar of a module-level
    ``pytestmark`` (``Assign`` vs ``AnnAssign``, last-assignment-wins,
    list/tuple element forms, positional vs ``timeout=``) belongs in exactly one
    place, and a rename there should break this import loudly at collection
    rather than let two readings of the same syntax drift apart.
    """
    resolved = [
        seconds
        for seconds in map(_resolved_seconds, module_level_timeout_arguments(source))
        if seconds is not None
    ]
    return min(resolved) if resolved else None


def _resolved_seconds(argument: ast.expr) -> float | None:
    """A timeout mark's seconds *argument* as a number, or None when it is unknowable.

    A numeric literal resolves to itself (``bool`` excluded: ``timeout(True)``
    is not 1s); a sanctioned name resolves through
    :data:`_SANCTIONED_CEILING_SECONDS`; anything else is no opinion.
    """
    if (
        isinstance(argument, ast.Constant)
        and isinstance(argument.value, int | float)
        and not isinstance(argument.value, bool)
    ):
        return float(argument.value)
    name = _trailing_name(argument)
    return None if name is None else _SANCTIONED_CEILING_SECONDS.get(name)


def _timeout_marks_below_ceiling(source: str) -> tuple[tuple[str, float], ...]:
    """``(qualname, seconds)`` of every class- or test-level timeout mark in *source* below the family ceiling.

    Covers test and method decorators, class decorators and a class-body
    ``pytestmark``, in classes at any depth, keyed as
    :class:`shared.testing_timeout_markers.TimeoutSite` keys them
    (``TestA::test_a``, ``TestA::<pytestmark>``).  The walk is that module's
    :func:`~shared.testing_timeout_markers.timeout_marker_sites`, and seconds
    resolve through the same literal rule and :data:`_SANCTIONED_CEILING_SECONDS`
    table as :func:`_module_level_timeout_ceiling`, so an unresolvable value is
    no opinion.  The module-level mark is that function's concern, not this
    one's.  Fails soft.
    """
    tree = _parsed(source)
    return () if tree is None else _timeout_marks_below_ceiling_in(tree)


def _timeout_marks_below_ceiling_in(tree: ast.Module) -> tuple[tuple[str, float], ...]:
    """:func:`_timeout_marks_below_ceiling` over an already-parsed module."""
    return tuple(
        (site.qualname, site.seconds)
        for site in timeout_marker_sites(tree, _SANCTIONED_CEILING_SECONDS)
        if site.kind is not SiteKind.MODULE_PYTESTMARK
        and site.seconds is not None
        and site.seconds < WHOLE_TREE_SCAN_TEST_TIMEOUT
    )


def _scanner_offence(name: str, source: str) -> str | None:
    """Why the whole-tree scanner *name* (content *source*) breaks the family invariant, else None.

    The three halves in order: a module-level timeout NAME, a resolvable VALUE
    not below the family ceiling (an unresolvable one is no opinion), and
    MEMBERSHIP of the family task-role verify enumerates.
    """
    if 'timeout' not in module_level_marker_names(source):
        return f'{name}: no module-level pytestmark timeout'
    ceiling = _module_level_timeout_ceiling(source)
    if ceiling is not None and ceiling < WHOLE_TREE_SCAN_TEST_TIMEOUT:
        return (
            f'{name}: module-level timeout pins {ceiling:g}s, below '
            f'the {WHOLE_TREE_SCAN_TEST_TIMEOUT}s family ceiling'
        )
    if not is_whole_tree_guard_module(source):
        return (
            f'{name}: module-level timeout is not spelled '
            'pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT), so verify_plan\'s '
            'task-role guard family will not run it'
        )
    return None


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

    This is the real pattern in
    _tree_scan.py::_package_root_python_files, and it is what stops
    the detector reading that function's directory walks as the thing being
    guarded -- while its genuine ``glob('*.py')`` is still caught by the case
    above.
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

    Deliberately the polarity of the sibling guards
    (test_prune_chokepoint_guard.py::_find_prune_argv_literals and
    test_raw_semaphore_access_guard.py::_raw_semaphore_accesses both
    ``except SyntaxError: return []``)
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
    green while it sits BELOW the family ceiling.
    """
    source = 'import pytest\npytestmark = pytest.mark.timeout(30)\n'
    assert _module_level_timeout_ceiling(source) == 30.0


def test_ceiling_resolves_the_sanctioned_constant_name() -> None:
    """The house spelling resolves to the constant's real runtime value."""
    source = (
        'import pytest\n'
        'from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT\n'
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
    """Only ``tree.body`` is walked, inheriting ``pytestmark_value``'s contract.

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
    while sitting below even the global ini default. Proven synthetically because the
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
# Membership of this module itself. The family predicate's own unit tests live
# with it, in shared/tests/test_whole_tree_guard_family.py.
# ---------------------------------------------------------------------------


def test_this_guard_module_is_itself_a_guard_family_member() -> None:
    """The real-name coupling: this module carries the house mark, so renaming
    the constant or its spelling in shared.whole_tree_guard_family reds here."""
    assert is_whole_tree_guard_module(Path(__file__).read_text(encoding='utf-8')) is True


# ---------------------------------------------------------------------------
# _scanner_offence(name, source) -- the per-scanner verdict the family
# invariant applies to each detected scanner, pinned synthetically because the
# real tree is green by construction and cannot demonstrate any offence.
# ---------------------------------------------------------------------------


def _synthetic_scanner(mark_lines: str) -> str:
    """A whole-tree scanner module whose module-level marks are *mark_lines*."""
    return (
        'import ast\n'
        'import pytest\n'
        'from pathlib import Path\n'
        'from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT\n'
        f'{mark_lines}'
        '\n'
        '_TESTS_DIR = Path(__file__).parent\n'
        '\n'
        'def test_some_new_invariant() -> None:\n'
        "    for py_file in sorted(_TESTS_DIR.rglob('*.py')):\n"
        "        ast.parse(py_file.read_text(encoding='utf-8'))\n"
    )


def test_offence_is_none_for_the_house_spelling() -> None:
    source = _synthetic_scanner('pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)\n')
    assert _scanner_offence('test_new.py', source) is None


def test_offence_names_a_missing_module_level_timeout() -> None:
    offence = _scanner_offence('test_new.py', _synthetic_scanner(''))
    assert offence is not None
    assert offence.startswith('test_new.py: ')
    assert 'no module-level pytestmark timeout' in offence


def test_offence_names_a_ceiling_below_the_family_ceiling() -> None:
    offence = _scanner_offence(
        'test_new.py', _synthetic_scanner('pytestmark = pytest.mark.timeout(30)\n'),
    )
    assert offence is not None
    assert 'pins 30s' in offence
    assert 'below the' in offence and 'family ceiling' in offence


@pytest.mark.parametrize(
    'mark_line',
    [
        pytest.param('pytestmark = pytest.mark.timeout(1500)\n', id='literal-above-the-ceiling'),
        pytest.param(
            'from somewhere import SOME_OTHER_TIMEOUT\n'
            'pytestmark = pytest.mark.timeout(SOME_OTHER_TIMEOUT)\n',
            id='unresolvable-constant',
        ),
    ],
)
def test_offence_names_a_mark_outside_the_guard_family(mark_line: str) -> None:
    """Passing the name and value halves is not enough: task-role verify runs
    only the family ``is_whole_tree_guard_module`` admits, so a scanner spelled
    any other way would stop running before the merge gate."""
    offence = _scanner_offence('test_new.py', _synthetic_scanner(mark_line))
    assert offence is not None
    assert 'WHOLE_TREE_SCAN_TEST_TIMEOUT' in offence
    assert 'guard family' in offence


def test_a_synthetic_new_scanner_marked_by_a_literal_is_not_a_family_member() -> None:
    """END TO END on the third half: detected, named, AT the ceiling by value --
    and still an offender, proving the membership half is not dead code."""
    synthetic_source = _synthetic_scanner('pytestmark = pytest.mark.timeout(1500)\n')
    assert _scans_whole_tree_py(synthetic_source) is True
    assert 'timeout' in module_level_marker_names(synthetic_source)
    ceiling = _module_level_timeout_ceiling(synthetic_source)
    assert ceiling is not None and ceiling >= WHOLE_TREE_SCAN_TEST_TIMEOUT
    assert _scanner_offence('test_new.py', synthetic_source) is not None


# ---------------------------------------------------------------------------
# _consumes_tree_pass(source) -- a guard that joined the shared pass no longer
# globs, so the census must recognise it by what it declares instead.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'source',
    [
        pytest.param('TREE_ANALYSES = (ANALYSIS,)\n', id='module-level-analyses'),
        pytest.param(
            'TREE_ANALYSES: tuple[TreeAnalysis, ...] = (ANALYSIS,)\n',
            id='annotated-module-level-analyses',
        ),
        pytest.param(f'def test_x({TREE_PASS_FIXTURE}):\n    pass\n', id='test-function'),
        pytest.param(
            f'async def test_x({TREE_PASS_FIXTURE}):\n    pass\n', id='async-test-function',
        ),
        pytest.param(
            f'class TestX:\n    def test_x(self, {TREE_PASS_FIXTURE}):\n        pass\n',
            id='test-method',
        ),
        pytest.param(
            f'import pytest\n@pytest.fixture\ndef census(*, {TREE_PASS_FIXTURE}):\n    pass\n',
            id='keyword-only-fixture-parameter',
        ),
    ],
)
def test_consumer_detector_admits_a_declaration_or_a_fixture_request(source: str) -> None:
    assert _consumes_tree_pass(source) is True


@pytest.mark.parametrize(
    'source',
    [
        pytest.param(
            f'"""Declares TREE_ANALYSES and takes {TREE_PASS_FIXTURE}."""\n'
            f'# def test_x({TREE_PASS_FIXTURE}): ...\n',
            id='docstring-and-comment-only',
        ),
        pytest.param(
            'def build():\n    TREE_ANALYSES = (ANALYSIS,)\n    return TREE_ANALYSES\n',
            id='function-local-analyses',
        ),
        pytest.param(f'def test_x(tmp_path):\n    return {TREE_PASS_FIXTURE}\n', id='name-not-parameter'),
        pytest.param('import json\n', id='unrelated-module'),
        pytest.param(f'def test_x({TREE_PASS_FIXTURE}:\n', id='syntax-error'),
    ],
)
def test_consumer_detector_rejects_everything_else_without_raising(source: str) -> None:
    assert _consumes_tree_pass(source) is False


# ---------------------------------------------------------------------------
# _module_xdist_group(source) -- the shared pass is paid once only if every
# consumer lands on the one worker its xdist group pins.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('mark', 'expected'),
    [
        pytest.param("pytest.mark.xdist_group('some_group')", 'some_group', id='literal'),
        pytest.param("pytest.mark.xdist_group(name='some_group')", 'some_group', id='keyword'),
        pytest.param(
            'pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP)', TREE_PASS_XDIST_GROUP,
            id='sanctioned-name',
        ),
        pytest.param(
            'pytest.mark.xdist_group(_tree_scan.TREE_PASS_XDIST_GROUP)', TREE_PASS_XDIST_GROUP,
            id='dotted-sanctioned-name',
        ),
        pytest.param(
            '[pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT), '
            'pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP)]',
            TREE_PASS_XDIST_GROUP,
            id='inside-a-pytestmark-list',
        ),
        pytest.param('pytest.mark.xdist_group(SOME_OTHER_GROUP)', None, id='unresolvable-name'),
        pytest.param('pytest.mark.xdist_group()', None, id='zero-arg'),
        pytest.param('pytest.mark.timeout(30)', None, id='no-xdist-group'),
    ],
)
def test_xdist_group_reads_the_module_level_pytestmark(mark: str, expected: str | None) -> None:
    assert _module_xdist_group(f'import pytest\npytestmark = {mark}\n') == expected


@pytest.mark.parametrize(
    'source',
    [
        pytest.param('import pytest\n', id='no-pytestmark'),
        pytest.param(
            "import pytest\nclass TestX:\n    pytestmark = pytest.mark.xdist_group('g')\n",
            id='class-level-only',
        ),
        pytest.param(
            "import pytest\ndef f():\n    pytestmark = pytest.mark.xdist_group('g')\n",
            id='function-local',
        ),
        pytest.param("pytestmark = pytest.mark.xdist_group('g'\n", id='syntax-error'),
    ],
)
def test_xdist_group_is_none_without_a_module_level_group(source: str) -> None:
    assert _module_xdist_group(source) is None


# ---------------------------------------------------------------------------
# _timeout_marks_below_ceiling(source) -- any consumer item may be the one
# billed the shared pass, so none may narrow the family ceiling below the
# module level.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('source', 'expected'),
    [
        pytest.param(
            'import pytest\n@pytest.mark.timeout(30)\ndef test_a():\n    pass\n',
            (('test_a', 30.0),),
            id='function-decorator',
        ),
        pytest.param(
            'import pytest\n@pytest.mark.timeout(timeout=30)\ndef test_a():\n    pass\n',
            (('test_a', 30.0),),
            id='keyword-spelling',
        ),
        pytest.param(
            'import pytest\n@pytest.mark.timeout(30)\nclass TestA:\n    pass\n',
            (('TestA', 30.0),),
            id='class-decorator',
        ),
        pytest.param(
            'import pytest\nclass TestA:\n'
            '    @pytest.mark.timeout(30)\n    def test_a(self):\n        pass\n',
            (('TestA::test_a', 30.0),),
            id='method-decorator',
        ),
        pytest.param(
            'import pytest\nclass TestA:\n    pytestmark = pytest.mark.timeout(30)\n',
            (('TestA::<pytestmark>', 30.0),),
            id='class-body-pytestmark',
        ),
        pytest.param(
            'import pytest\nclass TestA:\n    class TestB:\n'
            '        @pytest.mark.timeout(30)\n        def test_a(self):\n            pass\n',
            (('TestA::TestB::test_a', 30.0),),
            id='nested-class-method',
        ),
        pytest.param(
            'import pytest\n'
            '@pytest.mark.timeout(30)\ndef test_a():\n    pass\n'
            '@pytest.mark.timeout(45)\ndef test_b():\n    pass\n',
            (('test_a', 30.0), ('test_b', 45.0)),
            id='every-site-in-source-order',
        ),
    ],
)
def test_sub_ceiling_marks_are_read_below_module_level(
    source: str, expected: tuple[tuple[str, float], ...],
) -> None:
    assert _timeout_marks_below_ceiling(source) == expected


@pytest.mark.parametrize(
    'source',
    [
        pytest.param(
            'import pytest\nfrom shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT\n'
            '@pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)\ndef test_a():\n    pass\n',
            id='at-the-ceiling-by-name',
        ),
        pytest.param(
            'import pytest\n@pytest.mark.timeout(1500)\ndef test_a():\n    pass\n',
            id='above-the-ceiling',
        ),
        pytest.param(
            'import pytest\nfrom somewhere import SOME_OTHER_TIMEOUT\n'
            '@pytest.mark.timeout(SOME_OTHER_TIMEOUT)\ndef test_a():\n    pass\n',
            id='unresolvable-is-no-opinion',
        ),
        pytest.param(
            'import pytest\npytestmark = pytest.mark.timeout(30)\n',
            id='module-level-is-the-ceiling-check-not-this-one',
        ),
        pytest.param('import pytest\n', id='no-marks'),
        pytest.param('@pytest.mark.timeout(30)\ndef test_a(:\n', id='syntax-error'),
    ],
)
def test_sub_ceiling_marks_ignore_everything_else_without_raising(source: str) -> None:
    assert _timeout_marks_below_ceiling(source) == ()


# ---------------------------------------------------------------------------
# The family invariant: the actual ratchet.
#
# The census is detected private sweepers plus shared-pass consumers, recomputed
# from source on every run, so the Nth new guard fails at commit time instead
# of os._exit()ing an xdist worker in someone else's verify. Its only residuals
# are the reasoned allowlists at the top of this module, each checked for
# staleness.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ModuleCensus:
    """What the family invariant needs to know about one module under this directory.

    An unreadable module is ``readable=False`` with every other field at its
    default: it was not examined.
    """

    readable: bool
    sweeps_privately: bool = False
    consumes: bool = False
    is_family_member: bool = False
    offence: str | None = None
    xdist_group: str | None = None
    low_timeout_marks: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class _TreeCensus:
    """Every readable module under this directory by :func:`_census_key`, and the keys that could not be read."""

    modules: Mapping[str, _ModuleCensus]
    unreadable: tuple[str, ...]


def _module_census(module: ParsedModule) -> _ModuleCensus:
    """Census one module from the shared pass; ``_tree_scan.py`` is a helper, not a test module, so it cannot offend.

    Fail-soft on the READ for the same reason _scans_whole_tree_py fails soft
    on the PARSE: a non-UTF-8 source, a deliberately-malformed encoding fixture
    or a broken symlink under this directory would otherwise fail a
    TIMEOUT-COVERAGE check for a reason that has nothing to do with timeout
    coverage -- the very class of misattributed failure this module exists to
    prevent.  An unreadable module is not counted as examined (it was not), and
    the family test surfaces the count, so a sweep that silently stops reading
    cannot hide.

    A module naming none of the identifiers the two detectors match on is
    neither a sweeper nor a consumer, so its tree is never walked.
    """
    try:
        source = module.source
    except (UnicodeDecodeError, OSError):
        return _ModuleCensus(readable=False)
    is_family_member = is_whole_tree_guard_module(source)
    if not module.may_name(*_GLOB_METHODS, _TREE_ANALYSES_NAME, TREE_PASS_FIXTURE):
        return _ModuleCensus(readable=True, is_family_member=is_family_member)
    try:
        tree = module.tree
    except (SyntaxError, ValueError):
        return _ModuleCensus(readable=True, is_family_member=is_family_member)
    sweeps = _sweeps_whole_tree_in(tree)
    consumes = _consumes_tree_pass_in(tree)
    judged = (sweeps or consumes) and module.path.name.startswith('test_')
    return _ModuleCensus(
        readable=True,
        sweeps_privately=sweeps,
        consumes=consumes,
        is_family_member=is_family_member,
        offence=_scanner_offence(_census_key(module.path), source) if judged else None,
        xdist_group=_module_xdist_group_in(tree) if consumes else None,
        low_timeout_marks=_timeout_marks_below_ceiling_in(tree) if consumes else (),
    )


def _census_key(path: Path) -> str:
    """*path* as the census keys it: POSIX, relative to this directory, so a nested module cannot shadow a top-level one."""
    return path.relative_to(TESTS_DIR).as_posix()


ANALYSIS = TreeAnalysis(name=Path(__file__).stem, files=ORCHESTRATOR_TESTS, extract=_module_census)
TREE_ANALYSES = (ANALYSIS,)


def _tests_tree_census(tree_pass: TreePass) -> _TreeCensus:
    """This directory's census by :func:`_census_key`, projected from the shared tree pass."""
    facts = {_census_key(path): fact for path, fact in tree_pass.facts(ANALYSIS).items()}
    return _TreeCensus(
        MappingProxyType({key: fact for key, fact in facts.items() if fact.readable}),
        tuple(key for key, fact in facts.items() if not fact.readable),
    )


def _sweepers(modules: Mapping[str, _ModuleCensus]) -> frozenset[str]:
    return frozenset(name for name, module in modules.items() if module.sweeps_privately)


def _consumers(modules: Mapping[str, _ModuleCensus]) -> frozenset[str]:
    return frozenset(name for name, module in modules.items() if module.consumes)


def _family(modules: Mapping[str, _ModuleCensus]) -> frozenset[str]:
    return frozenset(name for name, module in modules.items() if module.is_family_member)


def _family_mark_offences(modules: Mapping[str, _ModuleCensus]) -> list[str]:
    """(i) Every guard's :func:`_scanner_offence`, so each one is a family member at the ceiling."""
    return [module.offence for module in modules.values() if module.offence is not None]


def _ungrouped_consumers(modules: Mapping[str, _ModuleCensus]) -> list[str]:
    """(ii) Consumers whose module-level xdist group is not the shared one."""
    return [
        f'{name}: pins xdist_group {module.xdist_group!r}, not {TREE_PASS_XDIST_GROUP!r}'
        for name, module in modules.items()
        if module.consumes and module.xdist_group != TREE_PASS_XDIST_GROUP
    ]


def _sub_ceiling_marks_in_consumers(modules: Mapping[str, _ModuleCensus]) -> list[str]:
    """(iii) Class- or test-level timeouts below the ceiling inside a consumer."""
    return [
        f'{name}::{qualname}: timeout {seconds:g}s'
        for name, module in modules.items()
        if module.consumes
        for qualname, seconds in module.low_timeout_marks
    ]


def _unsanctioned_private_sweeps(modules: Mapping[str, _ModuleCensus]) -> list[str]:
    """(iv) The private sweepers must be exactly the shared pass's home plus the reasoned exemptions.

    An exemption that stopped sweeping is (vi)'s stale entry, so only the home's
    absence is reported here.
    """
    sweepers = _sweepers(modules)
    sanctioned = {_SHARED_PASS_MODULE} | _PERMANENT_PRIVATE_SWEEPS.keys()
    return [
        *(
            f'{name}: sweeps a directory tree for *.py itself'
            for name in sorted(sweepers - sanctioned)
        ),
        *(
            [f'{_SHARED_PASS_MODULE}: the shared pass no longer sweeps a tree for *.py']
            if _SHARED_PASS_MODULE not in sweepers else []
        ),
    ]


def _members_outside_the_shared_pass(modules: Mapping[str, _ModuleCensus]) -> list[str]:
    """(v) Family members that neither consume the pass nor carry a reasoned exemption."""
    reasoned = _PERMANENT_PRIVATE_SWEEPS.keys() | _OUTSIDE_THE_SHARED_PASS.keys()
    return [
        f'{name}: is a whole-tree guard family member that does not consume the shared pass'
        for name in sorted(_family(modules) - _consumers(modules) - reasoned)
    ]


def _stale_allowlist_entries(modules: Mapping[str, _ModuleCensus]) -> list[str]:
    """(vi) Allowlist entries whose module no longer has the shape the entry excuses."""
    family, sweepers, consumers = _family(modules), _sweepers(modules), _consumers(modules)
    return [
        *(
            f'_PERMANENT_PRIVATE_SWEEPS: {name} no longer sweeps privately'
            for name in sorted(_PERMANENT_PRIVATE_SWEEPS)
            if name not in sweepers
        ),
        *(
            f'_OUTSIDE_THE_SHARED_PASS: {name} is not a family member that '
            'neither sweeps nor consumes'
            for name in sorted(_OUTSIDE_THE_SHARED_PASS)
            if name not in family or name in sweepers or name in consumers
        ),
    ]


_TIMEOUT_MARK_REMEDY = (
    f'Whole-tree guard module(s) under {TESTS_DIR.name}/ lack a module-level '
    'timeout mark AT THE FAMILY CEILING (no mark at all, one pinned lower, or '
    'one not spelled via WHOLE_TREE_SCAN_TEST_TIMEOUT). For a shared-pass '
    'consumer, make the module-level pytestmark\n\n'
    '    from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT\n'
    '    from _tree_scan import TREE_PASS_XDIST_GROUP\n\n'
    '    pytestmark = [\n'
    '        pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),\n'
    '        pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP),\n'
    '    ]\n\n'
    'A sanctioned private sweeper needs only the timeout entry. These modules '
    'ast.parse() every *.py in a tree, privately or as the item billed the '
    'shared pass; under `-n auto` that has been MEASURED at up to '
    f'{_MEASURED_UNDER_LOAD_WORST_CASE}s under load, against the per-test '
    f'default in {ORCH_PYPROJECT.name}. With timeout_method = "signal" a breach '
    'fails that one item, with its own traceback: a false red on a '
    'healthy-but-slow guard, which the ceiling exists to prevent.\n\n'
    'Use WHOLE_TREE_SCAN_TEST_TIMEOUT rather than a hand-picked number: a '
    'smaller literal carries the marker NAME while leaving the test at or under '
    'the cliff, which is why the value is checked and not just the name. Any '
    'other spelling, even a larger literal, also leaves the module out of the '
    'guard family task-role verify runs.'
)
_GROUP_PIN_REMEDY = (
    'Shared-pass consumer(s) not pinned to the shared xdist group. Make each '
    'module-level pytestmark\n\n'
    '    [pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),\n'
    '     pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP)]\n\n'
    'so `--dist loadgroup` puts every consumer on one worker and only that '
    'worker parses the tree (_tree_scan.py).'
)
_SUB_CEILING_REMEDY = (
    'Class- or test-level timeout mark(s) below the family ceiling inside a '
    'shared-pass consumer. Any consumer item may be the one billed the whole '
    'shared pass, so remove these marks and let the module-level ceiling bound '
    'every item.'
)
_JOIN_THE_PASS_REMEDY = (
    'Whole-tree guard(s) outside the shared tree pass. Declare a TreeAnalysis '
    "in the module's TREE_ANALYSES, take the tree_pass fixture, and pin "
    'pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP) beside the family timeout '
    '(_tree_scan.py), instead of reaching the tree some other way.'
)
_STALE_ENTRY_REMEDY = (
    'Stale allowlist entries at the top of this module: each module no longer '
    'has the shape its entry excuses. Delete each one.'
)


def _failure_report(checks: Iterable[tuple[str, list[str]]]) -> str:
    """One section per failing check, its remedy then its offenders; empty when every check passes."""
    return '\n\n'.join(
        f'{remedy}\n\nOffenders:\n  ' + '\n  '.join(offences)
        for remedy, offences in checks
        if offences
    )


def test_whole_tree_scanners_carry_module_level_timeout_mark(tree_pass: TreePass) -> None:
    """Every whole-tree guard here -- a private ``*.py`` sweeper or a consumer of
    the shared tree pass -- needs a MODULE-LEVEL timeout mark AT the family
    ceiling, spelled via ``WHOLE_TREE_SCAN_TEST_TIMEOUT``, and every family
    member must reach the tree through the shared pass unless an allowlist at
    the top of this module says why not.

    SEVEN CHECKS, liveness first: (vii) the sweep read enough files and found
    enough guards to mean anything; (i) every guard passes
    :func:`_scanner_offence`; (ii) every consumer pins the shared xdist group,
    so one worker parses; (iii) no consumer narrows the ceiling below module
    level, since any consumer item may be billed the pass; (iv) nothing but
    ``_tree_scan.py`` and the reasoned exemptions sweeps privately; (v) every
    family member is a consumer or a reasoned exemption; (vi) no allowlist
    entry is stale. Consumers are in the census because a guard that joined the
    pass no longer globs, and would otherwise drop out of the family exactly
    when it starts bearing the pass's cost.

    THREE HALVES, applied per guard by :func:`_scanner_offence`. The NAME half
    (:func:`module_level_marker_names`) catches a scanner with no mark; the
    VALUE half (:func:`_module_level_timeout_ceiling`) catches one marked
    ``pytest.mark.timeout(30)``, which carries the name yet sits BELOW even the
    global ini default -- so a name-only check would pass it green while it re-arms the
    exact cliff this module exists to clear. Only a value the grammar can
    actually resolve can offend: an unresolvable expression is skipped, keeping
    the guard a floor rather than a proof.

    The MEMBERSHIP half exists because task 7253 (the gate-5702 ruling) narrowed
    the task-role verify floor to the touched tests plus this family, enumerated
    by :func:`shared.whole_tree_guard_family.is_whole_tree_guard_module`. A scanner
    outside the family -- marked by a literal, or by another constant -- would
    silently stop running before the merge gate, so membership is not fail-soft.

    Why module-level rather than "a timeout mark somewhere in the file": only a
    module-level ``pytestmark`` is a statically sound LOWER BOUND on every
    collected item's marker set -- the documented contract of
    :func:`orchestrator.pytest_markers.module_level_marker_names`, which is
    IMPORTED here rather than re-derived. A decorator-aware sweep (e.g.
    test_marker_registration_drift.py's ``_applied_marker_names``) is an
    explicit UPPER bound: a file with three tests where only one is decorated
    would read as "has timeout" while the other two sit at the bare ini default, which
    is precisely the defect this guard exists to stop. An existing tighter
    per-test decorator still WINS where present -- verified empirically: with
    ``pytestmark = pytest.mark.timeout(300)`` plus ``@pytest.mark.timeout(7)``
    on one test, ``get_closest_marker('timeout').args`` is ``(7,)`` for the
    decorated test and ``(300,)`` for the bare one -- so the module-level mark
    acts purely as a FLOOR and narrows nothing.

    NO SKIP-SELF, deliberately, and stated here because it inverts a convention
    a reviewer will expect. Sibling guards such as
    test_prune_chokepoint_guard.py::test_exactly_one_prune_argv_literal_inside_chokepoint
    and the lock-release guard carry a ``_THIS_FILE``/``continue``
    skip-self because their forbidden pattern necessarily appears in their own
    detector code, making self-inclusion a guaranteed false positive. The
    opposite holds here: this module genuinely IS a whole-tree guard -- a
    consumer of the shared pass, so any of its items may be billed the pass --
    and genuinely IS exposed to the same worker-death cliff, so skipping itself
    would exempt the one file most certain to need the mark. Its own
    ``pytestmark`` is at the top of this file.
    """
    census = _tests_tree_census(tree_pass)
    modules = census.modules
    guards = _sweepers(modules) | _consumers(modules)

    assert len(modules) >= _MIN_EXPECTED_TEST_FILES, (
        f'only {len(modules)} .py files examined under {TESTS_DIR} (expected at '
        f'least {_MIN_EXPECTED_TEST_FILES}; {len(census.unreadable)} skipped as '
        f'unreadable: {sorted(census.unreadable)}) -- the sweep itself is broken, '
        'so this guard would pass vacuously rather than because the family is '
        'clean.'
    )
    assert len(guards) >= _MIN_EXPECTED_FAMILY, (
        f'only {len(guards)} whole-tree guard(s) detected among {len(modules)} '
        f'files (expected at least {_MIN_EXPECTED_FAMILY}) -- '
        '_scans_whole_tree_py or _consumes_tree_pass has probably stopped '
        f'matching, so this guard would pass vacuously. Found: {sorted(guards)}'
    )

    report = _failure_report((
        (_TIMEOUT_MARK_REMEDY, _family_mark_offences(modules)),
        (_GROUP_PIN_REMEDY, _ungrouped_consumers(modules)),
        (_SUB_CEILING_REMEDY, _sub_ceiling_marks_in_consumers(modules)),
        (_JOIN_THE_PASS_REMEDY, _unsanctioned_private_sweeps(modules)),
        (_JOIN_THE_PASS_REMEDY, _members_outside_the_shared_pass(modules)),
        (_STALE_ENTRY_REMEDY, _stale_allowlist_entries(modules)),
    ))
    assert not report, report


# ---------------------------------------------------------------------------
# Formerly-mixed modules stay out of the family (task 4600).
#
# Each of these once carried the family mark only because ONE incidental
# whole-tree scan sat beside many unrelated tests. Each scan now lives in a
# dedicated *_guard.py module, and these modules must not drift back.
# ---------------------------------------------------------------------------

_FORMERLY_MIXED_MODULES = (
    'test_lane_lifecycle_gitops.py',
    'test_scheduler_hold_history.py',
    'test_workflow_factory.py',
    'test_eval_boundary_suite.py',
    'test_mcp_post_transport.py',
    'test_review_checkpoint_working_tree.py',
)


@pytest.mark.parametrize('module_name', _FORMERLY_MIXED_MODULES)
def test_formerly_mixed_module_is_not_a_family_member(
    module_name: str, tree_pass: TreePass,
) -> None:
    tree_census = _tests_tree_census(tree_pass)
    census = tree_census.modules.get(module_name)
    unreadable = ' (it could not be read)' if module_name in tree_census.unreadable else ''
    assert census is not None, (
        f'{module_name} is not a readable module under {TESTS_DIR}{unreadable}. '
        'If it was renamed or deleted, update _FORMERLY_MIXED_MODULES to match.'
    )
    reason = (
        'Move the incidental whole-tree scan into a dedicated *_guard.py module. '
        'A module-level family mark widens the hang ceiling of every unrelated '
        f'test in {module_name} to WHOLE_TREE_SCAN_TEST_TIMEOUT '
        f'({WHOLE_TREE_SCAN_TEST_TIMEOUT}s), and drags those tests into every '
        'task-role verify (docs/task-role-pytest-floor.md).'
    )
    assert census.is_family_member is False, (
        f'{module_name} carries the whole-tree guard family mark. {reason}'
    )
    assert census.sweeps_privately is False, (
        f'{module_name} sweeps a directory tree for *.py files. {reason}'
    )


# ---------------------------------------------------------------------------
# Every scan of the original family now reaches the tree through the shared
# pass (task 4600, acceptance 4(c)).
#
# The family task 4215 marked -- its twelve scanners plus this meta-guard -- by
# the module that now hosts each one's scan. test_merge_queue_reachback_patch_guard.py,
# test_serial_merge_worker_import_guard.py and test_event_loop_antipattern_guard.py
# are omitted: tasks 5036, 5034 and 8395 deleted them.
# ---------------------------------------------------------------------------

_ORIGINAL_FAMILY_HOMES = MappingProxyType({
    'test_eval_boundary_suite.py': 'test_taskworkflow_construction_point_guard.py',
    'test_killpg_frozen_pgid_guard.py': 'test_killpg_frozen_pgid_guard.py',
    'test_lane_lifecycle_gitops.py': 'test_lane_note_assigned_single_writer_guard.py',
    'test_lock_release_single_writer_guard.py': 'test_lock_release_single_writer_guard.py',
    'test_marker_registration_drift.py': 'test_marker_registration_drift.py',
    'test_prune_chokepoint_guard.py': 'test_prune_chokepoint_guard.py',
    'test_raw_semaphore_access_guard.py': 'test_raw_semaphore_access_guard.py',
    'test_scheduler_hold_history.py': 'test_lock_event_emit_site_guard.py',
    'test_whole_tree_scan_timeout_guard.py': 'test_whole_tree_scan_timeout_guard.py',
    'test_workflow_factory.py': 'test_taskworkflow_construction_point_guard.py',
})


def test_every_original_family_scan_is_served_by_the_shared_pass(tree_pass: TreePass) -> None:
    modules = _tests_tree_census(tree_pass).modules
    outside = [
        f'{original}: its scan lives in {home}, which does not consume the shared pass'
        for original, home in sorted(_ORIGINAL_FAMILY_HOMES.items())
        if not (home in modules and modules[home].consumes)
    ]
    assert outside == [], (
        'Each scan of the original whole-tree guard family must resolve to the one '
        'shared tree pass (_tree_scan.py), not a private sweep of its own:\n  '
        + '\n  '.join(outside)
    )
