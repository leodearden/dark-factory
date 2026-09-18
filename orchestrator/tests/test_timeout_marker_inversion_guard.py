"""Guard: no ``@pytest.mark.timeout(N)`` may INVERT into a tighter clamp under verify.

THE RULE, in one line: a marker at ``DELIBERATE_TIGHT_BOUND_CEILING < N <
VERIFY_CLI_PER_TEST_TIMEOUT`` is too large to read as a deliberate tight bound
and still sits below the budget verify passes, so the loosening its author
meant for a slow test silently TIGHTENS the verify run that gates their merge,
because a pytest-timeout marker is a two-way override and never a floor.

The derivation -- the ``pytest_timeout._get_item_settings`` precedence it
follows from, the three regimes the two budgets carve out, and why a breach
costs a whole truncated session rather than one red test -- has ONE home: the
``VERIFY_CLI_PER_TEST_TIMEOUT`` comment block in _orch_helpers.py.  Read it
there.  This module points at it rather than restating it, so a pytest-timeout
upgrade that moves that precedence invalidates one copy and not five.  The sole
deliberate exception is :func:`test_no_new_inverting_timeout_marker`'s failure
message, where the reader is looking at a traceback and not at the source.

A RATCHET, NOT A SWEEP.  61 pre-existing in-band sites are grandfathered in
:data:`_GRANDFATHERED`; see its comment for why they were not migrated here and
:func:`test_grandfather_allowlist_has_no_stale_entries` for what forces that
list to shrink.

SCOPE IS ``orchestrator/tests`` ONLY, and the sibling packages are KNOWINGLY
UNGUARDED -- do not read this module as tree-wide coverage.  The defect is a
property of the ``(ini timeout, verify --timeout)`` PAIR, not of this package,
and all eight segments of dark-factory-orchestrator.yaml's fleet chain pass
``--timeout=300``.  MEASURED by running this module's own extractor over the
siblings: fused-memory has 15 in-band sites of 41, under the same
``timeout_method = "thread"`` / ``-n auto`` settings that make a breach here
cost a worker; shared has 1 of 21; escalation, dashboard,
sampler and tests/scripts have none.  Covering them means lifting the extractor
and :func:`_inverts` into a shared home -- orchestrator.pytest_markers already
owns the marker grammar this module imports -- and instantiating the guard per
package.  Filed as follow-up work rather than silently implied here:
agent-followup ticket tkt_0RTGWEF0FDSZ9QFZZYDFZVYSF5.

WHAT THIS DOES NOT DUPLICATE.  test_whole_tree_scan_timeout_guard.py polices a
per-FILE family invariant using MODULE-LEVEL marks only; test_marker_
registration_drift.py sweeps marker NAMES and never argument values;
tests/scripts/test_fallback_verify_config.py pins the CONFIG side (that every
pytest segment really does carry ``--timeout``).  Nothing before this checked a
per-test marker's VALUE against the verify budget.

Task 5147.
"""

from __future__ import annotations

import ast
import functools
import math
import re
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import _orch_helpers
import pytest
import yaml
from _orch_helpers import (
    DEEP_GATE_SCENE_TEST_TIMEOUT,
    DELIBERATE_TIGHT_BOUND_CEILING,
    ORCH_DIR,
    PYPROJECT_DEFAULT_TIMEOUT,
    VERIFY_CLI_PER_TEST_TIMEOUT,
    WHOLE_TREE_SCAN_TEST_TIMEOUT,
)

from orchestrator.pytest_markers import _marker_name, _pytestmark_value

# This module is ITSELF a whole-tree AST scanner -- it rglob()s every *.py
# under this directory and ast.parse()s each one -- so it is a member of the
# family test_whole_tree_scan_timeout_guard.py polices and carries the mark
# that guard demands.  WHOLE_TREE_SCAN_TEST_TIMEOUT is 300, at the inversion
# band's OPEN upper edge, so this module does not flag its own mark.
pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

#: The per-module merge-verify config whose ``test_command`` carries the
#: ``--timeout=N`` that verify actually passes to pytest.  Resolved from
#: ``ORCH_DIR`` (itself resolved from ``_orch_helpers.__file__``) and never
#: from the process CWD: merge-verify runs pytest from the ``orchestrator/``
#: cwd while a plain ``pytest orchestrator/tests`` runs from the repo root,
#: and this pin must read identically under both.
_ORCH_YAML = ORCH_DIR / 'orchestrator.yaml'

#: This directory, resolved from THIS FILE and never from the process CWD, for
#: the same reason ``_ORCH_YAML`` is: the census must come out identical under
#: merge-verify (cwd ``orchestrator/``) and a bare ``pytest orchestrator/tests``
#: (cwd the repo root).  Same idiom as
#: test_whole_tree_scan_timeout_guard.py::_TESTS_DIR and
#: test_marker_registration_drift.py::TESTS_DIR.
_TESTS_DIR = Path(__file__).resolve().parent

#: The module task 5333 sizes and ratchets: eight real-git classes that all
#: carried an identical bare ``300`` regardless of weight, which is what made
#: TestRow7KillSwitchByteIdentity's under-sizing invisible.
_DEEP_GATE_MODULE = 'test_merge_queue_deep_integration_gate.py'

#: The class whose marker is DERIVED from a measured spawn count (task 5333).
_ROW7_CLASS = 'TestRow7KillSwitchByteIdentity'

#: The only spellings a timeout marker in :data:`_DEEP_GATE_MODULE` may use.
#: Two, not one, because the asymmetry is DELIBERATE: Row 7's marker is
#: derived from its own measured cost, and its neighbours stay at the verify
#: CLI budget because none of them has been measured.
_DEEP_GATE_SPELLINGS = frozenset({
    'VERIFY_CLI_PER_TEST_TIMEOUT',
    'DEEP_GATE_SCENE_TEST_TIMEOUT',
})

#: Non-vacuity floor for the sweep of that module: it had eight real-git
#: classes when the ratchet was written, and a sweep that found none would
#: pass by finding nothing rather than by finding nothing wrong.
_MIN_DEEP_GATE_MARKER_SITES = 8

#: Same spelling as tests/scripts/test_fallback_verify_config.py, which pins
#: the FLEET-chain side of this same budget (``--timeout > 60`` on every
#: pytest segment of dark-factory-orchestrator.yaml, and ``--timeout >= 300``
#: on every per-module orchestrator.yaml).  Both the ``--timeout=300`` and
#: ``--timeout 300`` spellings are accepted, exactly as it does.
_TIMEOUT_FLAG_RE = re.compile(r'--timeout[=\s](\d+)')

#: Names a ``pytest.mark.timeout(...)`` argument may resolve to, and the
#: seconds each one carries.  A literal MAP rather than an import, because none
#: of the three is importable from ``_orch_helpers``: two are defined
#: FILE-LOCALLY in the modules that use them --
#: ``HEAVY_BARRIER_TEST_TIMEOUT = 5 * MERGE_RESULT_TIMEOUT + 75  # 300s``
#: (test_merge_queue_concurrent_verify.py) and ``PYTEST_TIMEOUT = 960``
#: (test_warm_lane_bash_suite.py -- test_pytest_marker_deselection.py's
#: identical line is inside a string FIXTURE, so it is not a binding, which
#: :class:`TestSanctionedNameMirrors` measures rather than assumes).
#: This is where the shape departs from its template:
#: ``_SANCTIONED_CEILING_NAMES`` in test_whole_tree_scan_timeout_guard.py is a
#: one-element frozenset paired with a single hard-coded
#: ``float(WHOLE_TREE_SCAN_TEST_TIMEOUT)``, which does not generalise to three
#: names at two distinct values.
#:
#: These are cross-module MIRRORS, and a stale one is NOT self-announcing.  A
#: mirror that reads too HIGH fails SILENTLY, which is the dangerous
#: direction: retune ``MERGE_RESULT_TIMEOUT`` to 30 and every
#: ``timeout(HEAVY_BARRIER_TEST_TIMEOUT)`` site really pins 225s -- squarely
#: inside the band -- while this map still answers 300 and the ratchet stays
#: green.  Matching on the TRAILING name only widens that hole: a new
#: file-local ``PYTEST_TIMEOUT = 120`` would be waved through at 960.
#:
#: So every entry carries the same EXECUTABLE link the YAML pin above
#: demonstrates -- :class:`TestSanctionedNameMirrors` re-reads each name's REAL
#: definition out of the tree and fails if the two disagree.  This map is a
#: cache of those definitions, never a claim about them.
_SANCTIONED_TIMEOUT_NAMES: dict[str, float] = {
    'WHOLE_TREE_SCAN_TEST_TIMEOUT': 540.0,
    'HEAVY_BARRIER_TEST_TIMEOUT': 300.0,
    'PYTEST_TIMEOUT': 960.0,
    'VERIFY_CLI_PER_TEST_TIMEOUT': float(VERIFY_CLI_PER_TEST_TIMEOUT),
    'DEEP_GATE_SCENE_TEST_TIMEOUT': float(DEEP_GATE_SCENE_TEST_TIMEOUT),
}

#: Qualname suffix for a ``pytestmark`` binding inside a class body, and the
#: whole qualname for a module-level one.  Angle brackets because no Python
#: identifier can contain them, so these can never collide with a real
#: function or class name in the allowlist's ``(module, qualname)`` key.
_MODULE_QUALNAME = '<module>'
_PYTESTMARK_QUALNAME = '<pytestmark>'


class _Site(NamedTuple):
    """One ``pytest.mark.timeout(...)`` occurrence found in a source file.

    ``seconds`` is None when the argument is present but UNRESOLVABLE, or
    absent entirely -- "no opinion", never "too small".  See
    :func:`_timeout_marker_sites`.

    ``spelling`` is the argument's unparsed SOURCE (``'300'``,
    ``'VERIFY_CLI_PER_TEST_TIMEOUT'``), empty when there is no argument.  It
    answers the question ``seconds`` cannot: a bare literal and the constant
    NAMING that same number resolve identically, so only the spelling
    distinguishes a marker that moves with a re-derivation from one that has
    to be found and hand-edited.
    """

    qualname: str
    kind: str
    seconds: float | None
    lineno: int
    spelling: str


def _timeout_call_arg(call: ast.Call) -> ast.expr | None:
    """The seconds expression a ``pytest.mark.timeout(...)`` *call* pins.

    Both spellings pytest-timeout accepts are read: positional ``timeout(300)``
    and keyword ``timeout(timeout=300)``.  A call with neither (``timeout()``,
    or one passing only ``method=``) yields None.  Same contract as
    test_whole_tree_scan_timeout_guard.py's function of this name; kept
    separate rather than imported because that module is a guard, not a
    helper library, and importing across two guards would couple their
    collection order.
    """
    if call.args:
        return call.args[0]
    for keyword in call.keywords:
        if keyword.arg == 'timeout':
            return keyword.value
    return None


def _resolve_seconds(arg: ast.expr | None) -> float | None:
    """Seconds *arg* pins, if statically knowable.

    RESOLUTION, deliberately tiny -- generalised from
    ``_module_level_timeout_ceiling``'s rules
    (test_whole_tree_scan_timeout_guard.py):

    * a numeric literal resolves to itself.  ``bool`` is excluded explicitly:
      it is an ``int`` subclass, so ``timeout(True)`` would otherwise resolve
      to 1.0 and read as an absurdly tight bound;
    * a name in :data:`_SANCTIONED_TIMEOUT_NAMES`, bare (``ast.Name``, the
      house ``from _orch_helpers import`` idiom) or dotted (``ast.Attribute``,
      compared on the trailing name only), resolves to that constant's value;
    * ANYTHING else -- arithmetic, an ``int(...)`` call, an f-string, an
      unfamiliar constant -- is UNKNOWABLE and yields None.

    None means "no opinion", never "too small": :func:`_inverts` must not treat
    it as an offence.  The consequence, stated rather than hidden: a marker
    that pins an in-band value through an indirection this grammar cannot
    follow is NOT caught.  Like the whole sweep, this is a FLOOR.
    """
    if arg is None:
        return None
    if (
        isinstance(arg, ast.Constant)
        and isinstance(arg.value, int | float)
        and not isinstance(arg.value, bool)
    ):
        return float(arg.value)
    name: str | None = None
    if isinstance(arg, ast.Name):
        name = arg.id
    elif isinstance(arg, ast.Attribute):
        name = arg.attr
    if name is None:
        return None
    return _SANCTIONED_TIMEOUT_NAMES.get(name)


def _parse(source: str) -> ast.Module | None:
    """*source* as a tree, or None when it does not parse.

    FAIL-SOFT by design: :func:`_tree_scan` reads every ``*.py`` under this
    directory, deliberately-malformed fixtures included, and a parse failure
    must not turn a TIMEOUT-COVERAGE guard red for a reason unrelated to
    timeout coverage -- the very class of misattributed failure this module
    exists to prevent.
    """
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError):
        return None


def _timeout_sites_in(elements: list[ast.expr], qualname: str, kind: str) -> list[_Site]:
    """Every ``timeout`` mark among *elements*, as sites keyed *qualname*/*kind*.

    *elements* is a decorator list or the unpacked value of a ``pytestmark``
    binding.  Non-``timeout`` marks yield no site at all (rather than a site
    with None seconds), so an ``@pytest.mark.asyncio`` never shows up in a
    census of timeout coverage.
    """
    sites: list[_Site] = []
    for element in elements:
        if not isinstance(element, ast.Call) or _marker_name(element) != 'timeout':
            continue
        arg = _timeout_call_arg(element)
        sites.append(
            _Site(
                qualname=qualname,
                kind=kind,
                seconds=_resolve_seconds(arg),
                lineno=element.lineno,
                spelling='' if arg is None else ast.unparse(arg),
            )
        )
    return sites


def _mark_elements(value: ast.expr) -> list[ast.expr]:
    """A ``pytestmark`` binding's marks, unwrapping the list/tuple form."""
    return list(value.elts) if isinstance(value, ast.List | ast.Tuple) else [value]


def _timeout_marker_sites(source: str) -> tuple[_Site, ...]:
    """Every ``pytest.mark.timeout(...)`` site in *source*, with its resolved seconds.

    WHY THIS EXISTS SEPARATELY from
    ``test_whole_tree_scan_timeout_guard.py::_module_level_timeout_ceiling``:
    that helper answers "what MODULE-LEVEL ceiling does this file pin", which
    is the right question for a per-FILE family invariant and the wrong one
    here.  A module-level ``pytestmark`` is the only form that is a sound LOWER
    bound on every collected item, which is exactly why that guard reads it and
    nothing else -- and exactly why it is blind to the population this module
    polices.  The measured census found 62 in-band markers and only a handful
    were module-level; the dominant spellings are the per-test DECORATOR and
    the per-CLASS decorator, neither of which that helper can see.  So this
    generalises its value resolution rather than replacing it: the existing
    guard keeps its narrower, stricter family invariant untouched.

    FOUR BINDING FORMS are collected, each keyed by a qualname that identifies
    the site stably across ordinary edits (the allowlist is keyed on
    ``(module, qualname)``, never a line number):

    * a function/method decorator -> ``test_a`` or ``TestThing::test_a``;
    * a class decorator -> ``TestThing``;
    * a module-level ``pytestmark`` -> ``<module>``;
    * a class-level ``pytestmark`` -> ``TestThing::<pytestmark>``.

    Nesting deeper than one class is walked for classes but not for closures:
    a decorator on a function defined INSIDE another function is not a
    collected pytest item, so it is not a site.

    ``_marker_name`` and ``_pytestmark_value`` are imported from
    :mod:`orchestrator.pytest_markers` rather than re-derived, for the same
    reason test_whole_tree_scan_timeout_guard.py imports them: the grammar of a
    ``pytest.mark.NAME`` element and of a ``pytestmark`` binding (``Assign`` vs
    ``AnnAssign``, list/tuple element forms) belongs in exactly one place, and
    a rename there should break this import loudly at collection rather than
    let two readings of the same syntax drift apart.

    FAIL-SOFT: unparseable source yields an EMPTY tuple and never raises.  The
    sweep reads every ``*.py`` under this directory, deliberately-malformed
    fixtures included, and a parse failure must not turn a timeout-coverage
    guard red for a reason unrelated to timeout coverage.
    """
    tree = _parse(source)
    return () if tree is None else _timeout_marker_sites_in(tree)


def _timeout_marker_sites_in(tree: ast.Module) -> tuple[_Site, ...]:
    """:func:`_timeout_marker_sites` over an already-parsed *tree*.

    Split out so :func:`_tree_scan` can ``ast.parse`` each file ONCE and feed
    both this and :func:`_sanctioned_bindings_in`; parsing dominates the sweep.
    """
    sites: list[_Site] = []

    def walk(body: list[ast.stmt], prefix: str) -> None:
        for statement in body:
            bound = _pytestmark_value(statement)
            if bound is not None:
                qualname = f'{prefix}{_PYTESTMARK_QUALNAME}' if prefix else _MODULE_QUALNAME
                kind = 'class-pytestmark' if prefix else 'module-pytestmark'
                sites.extend(_timeout_sites_in(_mark_elements(bound), qualname, kind))
            if isinstance(statement, ast.ClassDef):
                qualname = f'{prefix}{statement.name}'
                sites.extend(
                    _timeout_sites_in(statement.decorator_list, qualname, 'class-decorator')
                )
                walk(statement.body, f'{qualname}::')
            elif isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
                sites.extend(
                    _timeout_sites_in(
                        statement.decorator_list, f'{prefix}{statement.name}', 'decorator'
                    )
                )

    walk(tree.body, '')
    return tuple(sites)


def _inverts(seconds: float | None) -> bool:
    """True iff a marker at *seconds* TIGHTENS verify while reading as a loosening.

    The band is ``(DELIBERATE_TIGHT_BOUND_CEILING,
    VERIFY_CLI_PER_TEST_TIMEOUT)``, open at both ends -- a mark AT the ceiling
    is still a deliberate tight bound, and one AT the CLI budget is the
    recommended remediation.  Named
    against the constants rather than their numbers, and why those two edges
    and not the task text's literal ``N < 300``: the
    ``VERIFY_CLI_PER_TEST_TIMEOUT`` comment block in _orch_helpers.py.

    None is not a number and cannot invert: unresolvable means "no opinion",
    never "too small".
    """
    return (
        seconds is not None
        and DELIBERATE_TIGHT_BOUND_CEILING < seconds < VERIFY_CLI_PER_TEST_TIMEOUT
    )


def _class_def(tree: ast.Module, name: str) -> ast.ClassDef | None:
    """The top-level class *name*, or None if the module defines no such class.

    Top-level only: a pytest test class is collected only at module scope, so
    a nested definition of the same name would not be the thing under pin.
    """
    return next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name),
        None,
    )


def _autouse_fixtures(node: ast.ClassDef) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
    """*node*'s own ``@pytest.fixture(autouse=True)`` functions.

    Its OWN body only, never a walk: an autouse fixture bound anywhere else --
    at module scope, or in a sibling class -- applies to a different set of
    tests, and a pin that accepted one would pass while the class it names
    went unguarded.

    ``autouse`` is matched as the literal ``True`` rather than for mere
    presence, since ``autouse=False`` is a fixture that never runs.
    """
    return tuple(
        statement
        for statement in node.body
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
        and any(
            isinstance(decorator, ast.Call)
            and ast.unparse(decorator.func).endswith('fixture')
            and any(
                keyword.arg == 'autouse'
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in decorator.keywords
            )
            for decorator in statement.decorator_list
        )
    )


def _assert_enforced_call_names(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
) -> frozenset[str]:
    """Names *fn* CALLS whose result an ``assert`` in *fn* actually tests.

    Strictly stronger than "the name appears somewhere in the body", and the
    difference is the whole value of the pin that uses it (task 5333 reviewer
    amendment).  A fixture that computes a verdict and drops the assert, or
    that merely mentions the name in a keyword default, enforces NOTHING while
    satisfying a bare ``ast.Name`` walk -- so a pin built on one would be
    weaker than the claim it is there to carry.

    TWO enforcing shapes are accepted, because both really do fail the test:
    the call written inside the ``assert``'s own test, and the call bound to a
    name that an ``assert``'s test then reads (what the Row 7 fixture does, so
    its message can carry the verdict text).  The assert MESSAGE deliberately
    does not count: a call evaluated only to build the text of a failure that
    some other condition decides is not enforcement.

    Intra-procedural and syntactic, so it does not chase reassignment or prove
    a branch is live.  It pins the shape, not the semantics -- which is the
    honest limit of an AST pin and is why it is paired with the real fixture
    actually running.
    """
    asserted_names = {
        name.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Assert)
        for name in ast.walk(node.test)
        if isinstance(name, ast.Name)
    }

    def _called_name(value: ast.expr | None) -> str | None:
        return (
            value.func.id
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            else None
        )

    enforced: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assert):
            enforced.update(
                call.func.id
                for call in ast.walk(node.test)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            )
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            called = _called_name(node.value)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if called is not None and any(
                isinstance(target, ast.Name) and target.id in asserted_names
                for target in targets
            ):
                enforced.add(called)
    return frozenset(enforced)


# ---------------------------------------------------------------------------
# The REAL definitions behind _SANCTIONED_TIMEOUT_NAMES.
#
# The map is a mirror; these read the originals, so the mirror can be checked
# rather than trusted.  See _SANCTIONED_TIMEOUT_NAMES' comment for the two
# silent failures this closes.
# ---------------------------------------------------------------------------

#: Numbers a sanctioned constant's definition may be written in terms of.  All
#: four real definitions are literal arithmetic over _orch_helpers' own numeric
#: globals -- ``5 * PYPROJECT_DEFAULT_TIMEOUT`` and ``5 * MERGE_RESULT_TIMEOUT
#: + 75``, the latter defined in a test module that imports that name from
#: there -- so ONE namespace resolves every case with no test module imported.
#: Importing one for a constant would be a bad trade: test modules run code at
#: import, and test_warm_lane_bash_suite.py asserts at module scope right below
#: the PYTEST_TIMEOUT this map mirrors.
#:
#: STATED LIMIT: a definition written over a FILE-LOCAL name that shadows an
#: _orch_helpers global of the same spelling would resolve against the wrong
#: one.  No such shadow exists (all four definitions were read and resolved),
#: and the check is a floor, not a proof -- so this is recorded rather than
#: engineered around.
_MIRROR_NAMESPACE: Mapping[str, float] = {
    name: float(value)
    for name, value in vars(_orch_helpers).items()
    if isinstance(value, int | float) and not isinstance(value, bool)
}


class _Binding(NamedTuple):
    """One assignment of a :data:`_SANCTIONED_TIMEOUT_NAMES` name found in the tree.

    ``seconds`` is None when the right-hand side is outside
    :func:`_binding_value`'s grammar -- "no longer statically verifiable",
    which :class:`TestSanctionedNameMirrors` reports rather than passes over.
    ``expression`` is the unparsed source, so a failure can show what it read.
    """

    module: str
    name: str
    lineno: int
    expression: str
    seconds: float | None


def _binding_value(node: ast.expr, namespace: Mapping[str, float]) -> float | None:
    """*node*'s value, if it is literal arithmetic over *namespace*'s numbers.

    Deliberately tiny: numeric literals, names from *namespace*, and ``+ - *``
    over them is the entire grammar the four real definitions use.  ``bool`` is
    excluded from the literal case for the same reason :func:`_resolve_seconds`
    excludes it.  Anything else -- a call, an attribute, a name this namespace
    does not carry -- yields None.

    OPPOSITE POLARITY TO :func:`_resolve_seconds`, deliberately.  There, None
    means "no opinion" and is waved through; here it means the mirror can no
    longer be checked against its original, which is a finding, not a pass.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float) and not isinstance(node.value, bool):
            return float(node.value)
        return None
    if isinstance(node, ast.Name):
        return namespace.get(node.id)
    if isinstance(node, ast.BinOp):
        left = _binding_value(node.left, namespace)
        right = _binding_value(node.right, namespace)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
    return None


def _sanctioned_bindings_in(tree: ast.Module, module: str) -> list[_Binding]:
    """Every assignment in *tree* that binds a :data:`_SANCTIONED_TIMEOUT_NAMES` name.

    ``ast.walk`` rather than module-level only, so the rule stays one sentence:
    ANY binding of a sanctioned ceiling name under this tree must carry the
    mirrored value.  A class-body binding really can reach a method decorator
    (decorators are evaluated in the class namespace), and a function-local one
    cannot -- but flagging that one anyway errs toward a LOUD false offender,
    and a local shadowing a tree-wide ceiling name at a different value is
    confusing enough to be worth the noise either way.
    """
    bindings: list[_Binding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value: ast.expr | None = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
            value = node.value
        else:
            continue
        if value is None:
            continue
        bindings.extend(
            _Binding(
                module=module,
                name=target,
                lineno=node.lineno,
                expression=ast.unparse(value),
                seconds=_binding_value(value, _MIRROR_NAMESPACE),
            )
            for target in targets
            if target in _SANCTIONED_TIMEOUT_NAMES
        )
    return bindings


class TestVerifyCliBudgetConstant:
    """``VERIFY_CLI_PER_TEST_TIMEOUT`` -- the budget the whole guard is built on.

    Pinned here for the same reason ``PYPROJECT_DEFAULT_TIMEOUT`` is pinned by
    test_whole_tree_scan_timeout_guard.py::TestTimeoutConstants: a shared
    timeout constant that merely *claims* in a comment to mirror a config file
    is drift-prone, and the executable link is what makes the claim true.  This
    class is that shape one anchor over -- it reads the REAL
    orchestrator/orchestrator.yaml at runtime instead of citing a line number.
    """

    def test_the_inversion_band_is_non_empty(self) -> None:
        """The band ``(DELIBERATE_TIGHT_BOUND_CEILING, VERIFY_CLI_PER_TEST_TIMEOUT)`` must be real.

        The entire guard is the predicate "N sits strictly between the
        deliberate-tight-bound ceiling and the verify CLI budget".  If the two
        ever converge or invert, that predicate becomes unsatisfiable and every
        sweep below would pass VACUOUSLY -- green because nothing can offend,
        not because nothing does.  Asserted rather than assumed: the upper edge
        mirrors a config file that moves independently.

        THIS EXACT COLLAPSE HAPPENED ONCE (2026-09-12, commit 64e24b547f), back
        when the lower edge mirrored the ini default and that default was
        raised 60 -> 300 fleet-wide.  That default has since moved AGAIN, to 540
        (2026-09-17, task 5442, this time derived from measurement rather than
        picked) -- so had the edge stayed a mirror it would now sit ABOVE the
        upper one and invert the band rather than merely emptying it.  See
        DELIBERATE_TIGHT_BOUND_CEILING's comment in _orch_helpers.py for why the
        edge is a literal.
        """
        assert DELIBERATE_TIGHT_BOUND_CEILING < VERIFY_CLI_PER_TEST_TIMEOUT, (
            f'the inversion band ({DELIBERATE_TIGHT_BOUND_CEILING}, '
            f'{VERIFY_CLI_PER_TEST_TIMEOUT}) is empty -- the tight-bound '
            'ceiling has caught up with the verify CLI budget, so nothing can '
            'invert and this whole module would pass vacuously. Revisit it '
            'before changing either constant.'
        )

    def test_constant_mirrors_the_real_verify_test_command(self) -> None:
        """``VERIFY_CLI_PER_TEST_TIMEOUT`` must equal the ``--timeout`` verify really passes.

        THE DRIFT PIN.  The constant is deliberately a literal ``300`` rather
        than an expression over ``PYPROJECT_DEFAULT_TIMEOUT`` (see its comment
        in _orch_helpers.py), which means nothing about the Python source keeps
        it honest.  THIS test is what does: it re-reads
        ``orchestrator/orchestrator.yaml``'s ``test_command`` and extracts the
        ``--timeout`` token that pytest-timeout will actually see as
        ``config._env_timeout``.  If an operator retunes that flag, the band's
        upper edge moves with it and this fails loudly instead of leaving the
        guard silently policing a budget nobody passes any more.
        """
        test_command = yaml.safe_load(_ORCH_YAML.read_text(encoding='utf-8'))['test_command']

        match = _TIMEOUT_FLAG_RE.search(test_command)
        assert match, (
            f'{_ORCH_YAML} test_command carries no --timeout override '
            f'(got: {test_command!r}). VERIFY_CLI_PER_TEST_TIMEOUT models that '
            'flag, so without it the constant models nothing -- and every test '
            f'here silently falls back to the {PYPROJECT_DEFAULT_TIMEOUT}s '
            'pyproject default. This is also pinned from the other side by '
            'tests/scripts/test_fallback_verify_config.py::'
            'test_per_module_merge_verify_raises_per_test_timeout.'
        )
        configured = int(match.group(1))
        assert configured == VERIFY_CLI_PER_TEST_TIMEOUT, (
            f'VERIFY_CLI_PER_TEST_TIMEOUT ({VERIFY_CLI_PER_TEST_TIMEOUT}) no '
            f'longer mirrors --timeout={configured} in {_ORCH_YAML}. Update the '
            'constant in orchestrator/tests/_orch_helpers.py -- the inversion '
            'band this module polices is (DELIBERATE_TIGHT_BOUND_CEILING, '
            'VERIFY_CLI_PER_TEST_TIMEOUT), so a stale upper edge either lets a '
            'genuinely-inverting marker through or manufactures false '
            'offenders.'
        )


class TestSanctionedNameMirrors:
    """Every :data:`_SANCTIONED_TIMEOUT_NAMES` value must match its REAL definition.

    THE EXECUTABLE LINK the mirror map would otherwise lack -- the same shape
    :class:`TestVerifyCliBudgetConstant` applies to the YAML, one anchor over.
    Without it the map is an unchecked claim about four constants defined
    elsewhere, and a wrong entry that reads too HIGH is silent: it resolves a
    marker to a safe number while the real constant sits inside the band, so
    the ratchet stays green over a live inversion.

    THE MEASURED CASE this closes:
    ``HEAVY_BARRIER_TEST_TIMEOUT = 5 * MERGE_RESULT_TIMEOUT + 75``
    (test_merge_queue_concurrent_verify.py) is not a literal -- retune
    ``MERGE_RESULT_TIMEOUT`` from 45 to 30 and every site marked with it really
    pins 225s.  And because :func:`_resolve_seconds` matches on the TRAILING
    name only, a NEW file-local ``PYTEST_TIMEOUT = 120`` anywhere under this
    directory would resolve at 960.  Both are caught here, at their source.
    """

    def test_every_mirror_matches_its_real_definition(self) -> None:
        """No binding of a sanctioned name may disagree with the mirrored value.

        Checked over EVERY binding in the tree, not just the one the map was
        written from, because the trailing-name match makes a second definition
        at a different value indistinguishable from the first.
        """
        wrong = [
            binding
            for binding in _tree_scan().bindings
            if binding.seconds != _SANCTIONED_TIMEOUT_NAMES[binding.name]
        ]

        assert not wrong, (
            f'{len(wrong)} binding(s) of a sanctioned timeout name disagree '
            'with _SANCTIONED_TIMEOUT_NAMES in '
            'test_timeout_marker_inversion_guard.py.\n\n'
            'That map is how a `@pytest.mark.timeout(NAME)` site is resolved, '
            'and it matches on the TRAILING name only. An entry reading HIGHER '
            'than the real constant fails SILENTLY: the site is waved through '
            'as safe while it really pins a value inside the inversion band '
            f'({DELIBERATE_TIGHT_BOUND_CEILING} < N < {VERIFY_CLI_PER_TEST_TIMEOUT}). '
            'Update the map to the real value -- and if the real value has '
            'moved INTO the band, fix the constant instead, not the mirror.\n\n'
            'A `seconds` of None means the definition left the literal '
            'arithmetic _binding_value understands, so the mirror can no '
            'longer be checked at all; give the constant a statically '
            'resolvable definition or drop it from the map.\n\n'
            + '\n'.join(
                f'  {b.module}:{b.lineno} {b.name} = {b.expression} -> '
                f'{b.seconds} (mirror says '
                f'{_SANCTIONED_TIMEOUT_NAMES[b.name]})'
                for b in sorted(wrong, key=lambda b: (b.module, b.lineno))
            )
        )

    def test_every_mirrored_name_is_really_defined(self) -> None:
        """The map may not outlive the constants it mirrors.

        Same hygiene as
        :func:`test_grandfather_allowlist_has_no_stale_entries`: an entry for a
        deleted constant is dead weight that silently re-sanctions the name if
        someone later reintroduces it at an arbitrary value.  It is also the
        anti-vacuity floor for the twin above, which passes trivially over an
        empty binding list.
        """
        defined = {binding.name for binding in _tree_scan().bindings}
        undefined = sorted(set(_SANCTIONED_TIMEOUT_NAMES) - defined)

        assert not undefined, (
            f'_SANCTIONED_TIMEOUT_NAMES mirrors {undefined}, which no longer '
            f'name a constant defined anywhere under {_TESTS_DIR}. Drop the '
            'entr(y/ies) -- a mirror for a deleted constant sanctions the bare '
            'NAME, so reintroducing it at any value at all would be resolved '
            'to the stale number instead of read as new.'
        )


class TestSpawnBoundSizingModel:
    """Task 4203's spawn-bound sizing model must have exactly ONE home.

    THE MODEL sizes a ``@pytest.mark.timeout`` OVERRIDE for a test whose cost
    is dominated by real subprocess spawns rather than by bounded waits.  Its
    derivation -- what each term means, why the additive term exists, and the
    scope limit to marker-carrying overrides -- lives with
    :func:`_orch_helpers.required_timeout_secs`.  Read it there; this class
    pins the ARITHMETIC, not the reasoning.

    WHY IT IS PINNED HERE rather than where it was born.  Task 4203 wrote the
    model inside test_offline_lane_integration.py, where only that module
    could reach it.  Task 5333 needed the same arithmetic for
    test_merge_queue_deep_integration_gate.py -- which cannot import a test
    module without coupling the two suites' collection order -- so it moved to
    _orch_helpers.py, this package's established home for the
    timeout-constant family.  The last test below is what stops the old copy
    growing back.
    """

    def test_the_measured_spawn_latency_is_the_one_task_3451_measured(self) -> None:
        """The per-spawn price is a MEASUREMENT, and it is that one.

        Pinned as a literal because every timeout this model derives is a
        MULTIPLE of it: re-measuring it re-prices every marker sized against
        it at once.  That must be a deliberate edit carrying fresh numbers,
        never a passing tweak.
        """
        assert _orch_helpers.MEASURED_SPAWN_LATENCY_SECS == 4.71, (
            f'MEASURED_SPAWN_LATENCY_SECS is '
            f'{_orch_helpers.MEASURED_SPAWN_LATENCY_SECS}, not the 4.71 task '
            '3451 measured (n=3: 2.13/3.10/4.71, load-per-core 6.6). Every '
            'timeout derived through required_timeout_secs is a multiple of '
            'this number, so moving it silently re-prices '
            'DEEP_GATE_SCENE_TEST_TIMEOUT and every offline-lane marker at '
            'once. If it really has been re-measured, re-derive those '
            'constants in the SAME commit and record what was measured.'
        )

    def test_the_model_is_the_bounded_sum_plus_its_priced_spawns(self) -> None:
        """``required = bounded_secs + out_of_bound_spawns x latency``.

        Two of the three rows are numbers the model's callers ALREADY
        shipped, so the lifted copy is pinned to reproduce its origin rather
        than merely to be self-consistent: a move that quietly changed the
        arithmetic would leave every marker derived before it mis-sized, and
        nothing else in the tree would say so.
        """
        latency = _orch_helpers.MEASURED_SPAWN_LATENCY_SECS
        table = (
            (0.0, 0, 0.0, 'the degenerate case -- no waits and no spawns cost nothing'),
            (
                15.5,
                21,
                114.41,
                'the derivation worked in test_offline_lane_integration.py::'
                'test_out_of_bound_spawn_counts_are_measured_not_asserted\'s '
                '@pytest.mark.timeout(120) comment, the model\'s clearest '
                'shipped instance',
            ),
            (
                0.0,
                260,
                1224.6,
                'task 5333 -- DEEP_GATE_SCENE_SPAWN_BUDGET priced, the figure '
                'DEEP_GATE_SCENE_TEST_TIMEOUT rounds up from',
            ),
        )

        for bounded, spawns, expected, provenance in table:
            required = _orch_helpers.required_timeout_secs(bounded, spawns)

            assert required == pytest.approx(expected), (
                f'required_timeout_secs({bounded}, {spawns}) = {required}, '
                f'expected {expected} -- {provenance}. The model priced this '
                'row differently when the marker derived from it was written, '
                'so that marker is now mis-sized. Re-derive every constant '
                'built on this model before changing it.'
            )
            assert required == bounded + spawns * latency, (
                f'required_timeout_secs({bounded}, {spawns}) = {required}, but '
                f'the model it states is bounded_secs + out_of_bound_spawns x '
                f'MEASURED_SPAWN_LATENCY_SECS = {bounded + spawns * latency}. '
                'The function and its documented model have diverged; see its '
                'docstring in _orch_helpers.py.'
            )

    def test_the_offline_lane_module_no_longer_defines_its_own_copy(self) -> None:
        """The model's ORIGIN must import it, not redeclare it.

        Reads the source TEXT rather than the imported module, because that is
        the only way to tell an import apart from a redefinition: a module
        doing both would still answer every attribute lookup correctly while
        shipping a second, independently-editable copy.

        BOTH SPELLINGS are rejected.  The private one is what task 4203
        shipped and what a revert would restore; the PUBLIC one is what the
        module imports today and is therefore the likelier shape for a copy to
        come back in -- an author retuning a value "just for this module"
        would shadow the imported name, and every call site would keep
        reading.
        """
        offline_lane = _TESTS_DIR / 'test_offline_lane_integration.py'
        source = offline_lane.read_text(encoding='utf-8')

        redefinitions = [
            f'  {offline_lane.name}:{source.count(chr(10), 0, match.start()) + 1}'
            f'  {match.group(0)!r}'
            for pattern in (
                r'^_?MEASURED_SPAWN_LATENCY_SECS\s*(?::[^=\n]+)?=',
                r'^def _?required_timeout_secs\b',
            )
            for match in re.finditer(pattern, source, re.MULTILINE)
        ]

        assert not redefinitions, (
            f'{len(redefinitions)} definition(s) of the spawn-bound sizing '
            'model remain in test_offline_lane_integration.py, which must '
            'IMPORT it from _orch_helpers.py instead.\n\n'
            'Task 4203 wrote the model there, where only that module could '
            'reach it; task 5333 moved it to _orch_helpers.py so '
            'test_merge_queue_deep_integration_gate.py could size its own '
            'marker from the same arithmetic without importing a test module. '
            'A second definition here would not be a redundancy but a FORK -- '
            'two copies pricing spawns independently -- and 4203\'s own '
            'docstring records that per-callsite copies of a spawn count had '
            'ALREADY drifted apart once, inconsistently, before it '
            'consolidated them.\n\n' + chr(10).join(redefinitions)
        )


class TestDeepGateSceneBudget:
    """The two constants that size TestRow7KillSwitchByteIdentity (task 5333).

    The DERIVATION -- the measured spawn counts, the headroom, the rounding,
    and the two ceilings it sits under -- has ONE home: the
    ``DEEP_GATE_SCENE_TEST_TIMEOUT`` comment block in _orch_helpers.py.  This
    class is that comment's EXECUTABLE link, the same shape
    :class:`TestVerifyCliBudgetConstant` gives the YAML pin and
    :class:`TestSanctionedNameMirrors` gives the mirror map: the literals stay
    literals, and a runtime re-derivation is what keeps them honest.

    WHY A LITERAL AND NOT AN IMPORT-TIME EXPRESSION -- the argument
    ``VERIFY_CLI_PER_TEST_TIMEOUT``'s own comment makes, plus a mechanical
    one: :func:`_resolve_seconds` resolves only bare or dotted NAMES present
    in :data:`_SANCTIONED_TIMEOUT_NAMES`, so a marker spelled as arithmetic
    yields None -- "no opinion" -- and the ratchet would stop having a view of
    this marker at all.
    """

    def test_the_budget_covers_the_measured_worst_case(self) -> None:
        """The budget may never be tightened below what the class really costs.

        A budget BELOW the measurement would fail every run, loaded or not.
        The headroom above it is deliberate, and its size is argued in the
        constant's comment rather than here.
        """
        budget = _orch_helpers.DEEP_GATE_SCENE_SPAWN_BUDGET

        assert budget >= 234, (
            f'DEEP_GATE_SCENE_SPAWN_BUDGET is {budget}, below the 234 git '
            'spawns TestRow7KillSwitchByteIdentity\'s heaviest test '
            '(test_the_same_sequence_at_cap_six_moves_every_deep_field) was '
            'MEASURED to make. Measured 2026-09-14 at main 99ab62335a by '
            'counting asyncio.create_subprocess_exec/_shell per test, '
            'IDENTICAL across two independent runs (the other two tests cost '
            '113 each; 460 total, against 0.00s of asyncio.sleep -- which is '
            'what makes this class spawn-bound rather than sleep- or '
            'CPU-bound). A budget below the measurement fails every run, not '
            'just a loaded one. Re-measure before lowering it, and re-derive '
            'DEEP_GATE_SCENE_TEST_TIMEOUT in the same commit.'
        )

    def test_the_timeout_is_the_budget_priced_and_rounded_to_the_grid(self) -> None:
        """``ceil(required_timeout_secs(0.0, budget) / 60) * 60``, re-derived here.

        ``bounded_secs`` is 0.0 because the class was measured to perform
        0.00s of ``asyncio.sleep``: it has no bounded waits, so its whole cost
        is the spawn term.

        Sized against the BUDGET and not against the raw measurement, which is
        what lets the autouse fixture in
        test_merge_queue_deep_integration_gate.py keep the marker honest: the
        marker can only be wrong if the budget is breached, and a breach fails
        loudly, in-process, on the test that caused it.
        """
        budget = _orch_helpers.DEEP_GATE_SCENE_SPAWN_BUDGET
        timeout = _orch_helpers.DEEP_GATE_SCENE_TEST_TIMEOUT
        required = _orch_helpers.required_timeout_secs(0.0, budget)
        expected = math.ceil(required / 60) * 60

        assert timeout == expected, (
            f'DEEP_GATE_SCENE_TEST_TIMEOUT is {timeout}, but '
            f'DEEP_GATE_SCENE_SPAWN_BUDGET ({budget} spawns x '
            f'{_orch_helpers.MEASURED_SPAWN_LATENCY_SECS}s = {required}s) '
            f'rounds up the 60s pyproject grid to {expected}. The two '
            'constants are a PAIR -- moving the budget without re-deriving '
            'the timeout leaves the marker sized for a scene that no longer '
            'exists, which is the exact failure this task was filed to fix.'
        )

    def test_the_timeout_never_inverts_the_verify_budget(self) -> None:
        """A marker may only ever LOOSEN verify's per-test budget, never tighten it.

        The general rule and its cost are argued at
        ``VERIFY_CLI_PER_TEST_TIMEOUT`` in _orch_helpers.py.  Asserted here
        directly rather than left to the tree sweep because this marker is
        DERIVED: a future re-derivation could walk it into the band without
        anyone writing an in-band number by hand.
        """
        timeout = _orch_helpers.DEEP_GATE_SCENE_TEST_TIMEOUT

        assert timeout >= VERIFY_CLI_PER_TEST_TIMEOUT, (
            f'DEEP_GATE_SCENE_TEST_TIMEOUT ({timeout}) is below the verify '
            f'CLI budget ({VERIFY_CLI_PER_TEST_TIMEOUT}), so the marker meant '
            'to LOOSEN a slow class would instead TIGHTEN the run that gates '
            'the merge. Re-derive the budget upward, or re-measure the spawn '
            'latency -- do not simply clamp this constant.'
        )

    def test_the_timeout_fits_inside_the_whole_verify_run_budget(self) -> None:
        """A per-test backstop larger than the whole verify's budget is no backstop.

        Read from the REAL orchestrator.yaml at runtime, through the same
        :data:`_ORCH_YAML` and in the same shape
        :class:`TestVerifyCliBudgetConstant` reads the ``--timeout`` token --
        so an
        operator retuning the run budget down past this marker fails here
        loudly instead of leaving a marker that can never fire.
        """
        timeout = _orch_helpers.DEEP_GATE_SCENE_TEST_TIMEOUT
        config = yaml.safe_load(_ORCH_YAML.read_text(encoding='utf-8'))
        run_budget = config.get('verify_command_timeout_secs')

        assert run_budget is not None, (
            f'{_ORCH_YAML} carries no verify_command_timeout_secs, which '
            'DEEP_GATE_SCENE_TEST_TIMEOUT is bounded by. Without it this pin '
            'checks nothing; restore the key or re-argue the bound.'
        )
        assert timeout <= run_budget, (
            f'DEEP_GATE_SCENE_TEST_TIMEOUT ({timeout}s) exceeds '
            f'verify_command_timeout_secs ({run_budget}s) in {_ORCH_YAML}. A '
            'per-test backstop larger than the budget for the WHOLE verify '
            'run can never fire: verify kills the run first, and the class '
            'this marker protects goes back to dying as an unattributed '
            'worker crash. Shrink the scene or raise the run budget -- '
            'raising this constant alone buys nothing.'
        )

class TestSpawnBudgetVerdict:
    """``deep_gate_spawn_budget_violation`` -- the budget check as a pure verdict.

    Split out as a FUNCTION rather than written inline in the autouse fixture
    that calls it, so the fixture stays a thin wire and every branch below is
    reachable from a test.  A budget check buried in a fixture teardown is
    exercised only when it PASSES; these are the cases that matter and they
    are the ones a fixture-only implementation would never run.
    """

    def test_a_count_within_budget_is_no_violation(self) -> None:
        """The ordinary case, across the range the class really spans."""
        budget = _orch_helpers.DEEP_GATE_SCENE_SPAWN_BUDGET

        for count in (1, 113, 234, budget):
            assert _orch_helpers.deep_gate_spawn_budget_violation(count, 'm.py::t') is None, (
                f'{count} spawns against a budget of {budget} was reported as '
                'a violation. Only a count ABOVE the budget (or a zero count, '
                'which means the counting seam saw no git at all) is one.'
            )

    def test_the_budget_itself_is_inside_the_budget(self) -> None:
        """``count == budget`` passes -- the boundary is inclusive.

        Called out separately from the range above because an off-by-one here
        would fail a run that is exactly at the figure
        DEEP_GATE_SCENE_TEST_TIMEOUT was derived from, which is the one count
        the pair is guaranteed to be correctly sized for.
        """
        budget = _orch_helpers.DEEP_GATE_SCENE_SPAWN_BUDGET

        assert _orch_helpers.deep_gate_spawn_budget_violation(budget, 'm.py::t') is None, (
            f'a count of exactly {budget} -- the budget itself -- was reported '
            'as a violation. DEEP_GATE_SCENE_TEST_TIMEOUT is derived from this '
            'exact number, so it is the one count that must pass.'
        )

    def test_going_over_budget_names_everything_the_reader_needs(self) -> None:
        """The failure must say what got heavier, by how much, and what to re-derive.

        A bare "too many spawns" would leave the reader to discover on their
        own that a marker is sized from this number.  Each fragment is
        asserted individually so a message that drops one fails naming the
        fragment it dropped.
        """
        budget = _orch_helpers.DEEP_GATE_SCENE_SPAWN_BUDGET
        count = budget + 1
        nodeid = 'tests/test_merge_queue_deep_integration_gate.py::TestRow7::test_x'

        message = _orch_helpers.deep_gate_spawn_budget_violation(count, nodeid)

        assert message is not None, (
            f'{count} spawns against a budget of {budget} was not reported as '
            'a violation. One spawn over is over.'
        )
        for fragment, why in (
            (nodeid, 'the node id, so the reader knows WHICH test got heavier'),
            (str(count), 'the observed count'),
            (str(budget), 'the budget it broke'),
            (
                'DEEP_GATE_SCENE_TEST_TIMEOUT',
                'the constant to re-derive -- the point of the check is that a '
                'heavier scene needs a re-sized marker, not merely a raised budget',
            ),
        ):
            assert fragment in message, (
                f'the over-budget message omits {fragment!r} -- {why}.\n\n'
                f'got: {message}'
            )

    def test_a_zero_count_is_a_violation_although_it_is_within_budget(self) -> None:
        """Zero means the counting seam went blind, which is worse than over-budget.

        A guard that passes because it has been silently DISCONNECTED is worse
        than no guard: it reports green forever while measuring nothing, and
        the budget it appears to enforce becomes vacuous.  Zero is inside any
        budget, so nothing else in the check would catch it.
        """
        budget = _orch_helpers.DEEP_GATE_SCENE_SPAWN_BUDGET
        nodeid = 'tests/test_merge_queue_deep_integration_gate.py::TestRow7::test_x'

        message = _orch_helpers.deep_gate_spawn_budget_violation(0, nodeid)

        assert message is not None, (
            'a count of ZERO was accepted as within budget. It is arithmetically '
            'within any budget, and that is exactly the problem: it means the '
            'counting seam saw no git subprocess at all, so the budget is '
            'enforcing nothing. Report it rather than passing.'
        )
        assert nodeid in message, (
            f'the zero-count message omits the node id.\n\ngot: {message}'
        )
        assert message != _orch_helpers.deep_gate_spawn_budget_violation(budget + 1, nodeid), (
            'the zero-count message is identical to the over-budget message, so '
            'a reader cannot tell "this scene got heavier" (re-derive the '
            'constants) from "the counting seam broke" (fix the fixture). They '
            'are different failures with different remedies.'
        )


class TestRow7SceneIsGuarded:
    """What test_merge_queue_deep_integration_gate.py must carry for task 5333.

    TWO HALVES OF ONE MECHANISM, pinned together because either alone decays.
    The widened marker without the budget fixture is a number that silently
    goes stale the next time the scene grows -- which is exactly how it got
    stale in the first place.  The budget fixture without the widened marker
    guards a class that still dies on a loaded host before the fixture can
    report anything.
    """

    def _row7_marker(self) -> _Site:
        """The ``timeout`` marker site on the Row 7 class, or fail saying it is gone."""
        sites = [
            site
            for module, site in _tree_scan().sites
            if module == _DEEP_GATE_MODULE and site.qualname == _ROW7_CLASS
        ]

        assert sites, (
            f'{_DEEP_GATE_MODULE}::{_ROW7_CLASS} carries no @pytest.mark.timeout '
            'at all. It is the heaviest class in that file (234 git spawns in '
            'its worst test) and falls back to the pyproject default without '
            'one, which is how it came to die as an unattributed xdist worker '
            'crash. Restore the marker.'
        )
        assert len(sites) == 1, (
            f'{_DEEP_GATE_MODULE}::{_ROW7_CLASS} carries {len(sites)} timeout '
            f'markers at lines {[s.lineno for s in sites]}; the effective '
            'budget is then whichever pytest-timeout reads last, which no '
            'reader can predict. Keep exactly one.'
        )
        return sites[0]

    def test_the_row7_marker_is_the_derived_constant_not_a_literal(self) -> None:
        """Spelled as the NAME, so a re-derivation moves exactly one number.

        The value alone is not enough: a bare ``1260`` resolves identically,
        and would leave the constant and the marker as two independent copies
        of one figure -- the state this task found the file in, where eight
        classes of very different weight all carried an identical bare 300.
        """
        marker = self._row7_marker()

        assert marker.spelling == 'DEEP_GATE_SCENE_TEST_TIMEOUT', (
            f'{_DEEP_GATE_MODULE}::{_ROW7_CLASS} pins its timeout as '
            f'{marker.spelling!r} (line {marker.lineno}) rather than as the '
            'name DEEP_GATE_SCENE_TEST_TIMEOUT. That marker is DERIVED from a '
            'measured spawn count, so it must move when the derivation does; '
            'a literal here is a second copy of the number that no '
            're-derivation can reach. Import the constant from _orch_helpers.'
        )
        assert marker.seconds == DEEP_GATE_SCENE_TEST_TIMEOUT, (
            f'{_DEEP_GATE_MODULE}::{_ROW7_CLASS} resolves to '
            f'{marker.seconds}, not DEEP_GATE_SCENE_TEST_TIMEOUT '
            f'({DEEP_GATE_SCENE_TEST_TIMEOUT}). _SANCTIONED_TIMEOUT_NAMES has '
            'drifted from the real constant, which would leave this ratchet '
            'reasoning about a number the file does not pin.'
        )

    def test_the_row7_class_binds_an_autouse_spawn_budget_fixture(self) -> None:
        """The marker is only honest while the budget it was sized from is enforced.

        Read from the class BODY rather than from module text, so a fixture
        that drifted out to module scope -- where it would silently apply to
        every class in the file, or to none -- does not read as coverage of
        this one.

        ENFORCEMENT, not mention: the verdict must be CALLED and the call's
        result must reach an ``assert`` (see `_assert_enforced_call_names`).
        An earlier revision accepted the bare name anywhere in the fixture
        body, which a fixture that computed the verdict and dropped the assert
        would have satisfied while guarding nothing -- a pin weaker than the
        claim it carries.
        """
        tree = _parse((_TESTS_DIR / _DEEP_GATE_MODULE).read_text(encoding='utf-8'))
        assert tree is not None, f'{_DEEP_GATE_MODULE} did not parse'

        row7 = _class_def(tree, _ROW7_CLASS)
        assert row7 is not None, (
            f'{_DEEP_GATE_MODULE} defines no top-level class {_ROW7_CLASS}. If '
            'it was renamed, rename it here and in DEEP_GATE_SCENE_TEST_TIMEOUT'
            "'s comment too -- those constants are sized for THIS class's "
            'measured cost and mean nothing detached from it.'
        )

        verdict = 'deep_gate_spawn_budget_violation'
        guarded = [
            fixture.name
            for fixture in _autouse_fixtures(row7)
            if verdict in _assert_enforced_call_names(fixture)
        ]

        assert guarded, (
            f'{_DEEP_GATE_MODULE}::{_ROW7_CLASS} binds no autouse fixture that '
            f'calls {verdict} AND asserts on the result.\n\n'
            'DEEP_GATE_SCENE_TEST_TIMEOUT is sized against '
            'DEEP_GATE_SCENE_SPAWN_BUDGET rather than against the raw '
            'measurement precisely so that the budget, not a comment, is what '
            'keeps the marker honest -- a marker with no enforced budget is a '
            'number that decays silently. The argument for that coupling, and '
            'the measured decay behind it, are in _orch_helpers.py::'
            'DEEP_GATE_SCENE_TEST_TIMEOUT. Restore the fixture rather than '
            'widening the marker further.'
        )


# ---------------------------------------------------------------------------
# _assert_enforced_call_names(fn) -- inline-fixture unit tests.
#
# Same rationale as the extractor tests below: against the real tree this
# detector is green by construction, so its NEGATIVE cases -- the ones that
# carry its entire value over a bare `ast.Name` walk -- would otherwise never
# be exercised.  Each rejection below is a fixture shape that would pass the
# weaker pin while enforcing nothing (task 5333 reviewer amendment).
# ---------------------------------------------------------------------------


def _enforced(body: str) -> frozenset[str]:
    """``_assert_enforced_call_names`` over one dedented function snippet."""
    fn = ast.parse(textwrap.dedent(body)).body[0]
    assert isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef)
    return _assert_enforced_call_names(fn)


def test_enforced_reads_the_bind_then_assert_shape() -> None:
    """The Row 7 fixture's own spelling: bind the verdict, assert on the name."""
    assert 'verdict' in _enforced(
        """
        def _fixture():
            result = verdict(7)
            assert result is None, result
        """
    )


def test_enforced_reads_the_call_written_inside_the_assert() -> None:
    """The other honest spelling, where nothing is bound first."""
    assert 'verdict' in _enforced(
        """
        def _fixture():
            assert verdict(7) is None
        """
    )


def test_enforced_rejects_a_verdict_computed_and_dropped() -> None:
    """The exact regression this detector exists for.

    A fixture that still CALLS the verdict but no longer asserts on it guards
    nothing, while a bare-name walk reports it as covered.
    """
    assert 'verdict' not in _enforced(
        """
        def _fixture():
            result = verdict(7)
        """
    )


def test_enforced_rejects_a_bare_mention() -> None:
    """A name that is never called -- a keyword default, an alias, a comment's
    worth of code -- is not enforcement however prominently it appears."""
    assert 'verdict' not in _enforced(
        """
        def _fixture(check=verdict):
            handler = verdict
            assert True
        """
    )


def test_enforced_rejects_a_call_reached_only_by_the_assert_message() -> None:
    """A verdict evaluated only to TEXT a different condition decides to raise.

    The message is built after the test has already failed, so the call never
    determines the outcome -- which is why only the assert's test is walked.
    """
    assert 'verdict' not in _enforced(
        """
        def _fixture():
            assert spawns < 10, verdict(spawns)
        """
    )


# ---------------------------------------------------------------------------
# _timeout_marker_sites(source) -- inline-fixture unit tests.
#
# Same shape as the sibling guards' pure-detector tests
# (test_whole_tree_scan_timeout_guard.py::test_detector_flags_rglob_py,
# test_raw_semaphore_access_guard.py, test_prune_chokepoint_guard.py): the
# extractor is exercised against synthetic snippets so every resolution rule is
# pinned DIRECTLY, rather than only ever being reached through the real tree --
# which goes green by construction once the allowlist lands and would otherwise
# leave the negative cases untested forever.
# ---------------------------------------------------------------------------


def _sites(source: str) -> dict[str, float | None]:
    """``{qualname: seconds}`` for *source*, dedented so fixtures can be indented."""
    return {site.qualname: site.seconds for site in _timeout_marker_sites(textwrap.dedent(source))}


def test_extractor_reads_a_bare_literal_function_decorator() -> None:
    """The canonical spelling, and the one the named regression instance used."""
    assert _sites(
        """
        import pytest

        @pytest.mark.timeout(120)
        def test_slow() -> None:
            pass
        """
    ) == {'test_slow': 120.0}


def test_extractor_reads_the_keyword_spelling() -> None:
    """``timeout(timeout=120)`` -- the second spelling pytest-timeout accepts.

    Read for the same reason ``_timeout_call_arg`` reads it: a marker written
    this way clamps exactly as hard as the positional form, so skipping it
    would leave a silent hole in the sweep.
    """
    assert _sites(
        """
        import pytest

        @pytest.mark.timeout(timeout=120)
        def test_slow() -> None:
            pass
        """
    ) == {'test_slow': 120.0}


def test_extractor_reads_module_level_pytestmark() -> None:
    """A module-level ``pytestmark`` binds every item in the file."""
    assert _sites(
        """
        import pytest

        pytestmark = pytest.mark.timeout(120)
        """
    ) == {'<module>': 120.0}


def test_extractor_reads_list_form_pytestmark() -> None:
    """The list form, where the timeout mark sits among unrelated siblings."""
    assert _sites(
        """
        import pytest

        pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(120)]
        """
    ) == {'<module>': 120.0}


def test_extractor_reads_a_class_decorator() -> None:
    """A class decorator binds every method in the class at once.

    The dominant in-band spelling in this repo by count: the merge-queue and
    crash-recovery modules carry ~34 of them at 180s, so an extractor blind to
    class decorators would miss most of the population.
    """
    assert _sites(
        """
        import pytest

        @pytest.mark.timeout(180)
        class TestThing:
            def test_a(self) -> None:
                pass
        """
    ) == {'TestThing': 180.0}


def test_extractor_reads_class_level_pytestmark() -> None:
    """``pytestmark`` inside a class body -- same binding, different syntax."""
    assert _sites(
        """
        import pytest

        class TestThing:
            pytestmark = pytest.mark.timeout(180)

            def test_a(self) -> None:
                pass
        """
    ) == {'TestThing::<pytestmark>': 180.0}


def test_extractor_gives_a_method_a_dotted_qualname() -> None:
    """A decorated METHOD is keyed ``TestClass::test_method``.

    The allowlist is keyed on ``(module, qualname)`` rather than a line number
    so it survives ordinary edits, which means the qualname must actually
    disambiguate: two classes in one module routinely carry same-named
    methods, and a bare ``test_a`` would silently collapse them into one entry.
    """
    assert _sites(
        """
        import pytest

        class TestOne:
            @pytest.mark.timeout(120)
            def test_a(self) -> None:
                pass

        class TestTwo:
            @pytest.mark.timeout(150)
            def test_a(self) -> None:
                pass
        """
    ) == {'TestOne::test_a': 120.0, 'TestTwo::test_a': 150.0}


def test_extractor_resolves_a_sanctioned_constant_name() -> None:
    """Bare and dotted sanctioned names both resolve to their seconds.

    The house idiom is a bare ``from _orch_helpers import ...``; the dotted
    form is accepted too, comparing on the trailing name only, exactly as
    ``_module_level_timeout_ceiling`` does.
    """
    assert _sites(
        """
        import pytest

        @pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)
        def test_bare() -> None:
            pass

        @pytest.mark.timeout(_orch_helpers.WHOLE_TREE_SCAN_TEST_TIMEOUT)
        def test_dotted() -> None:
            pass

        @pytest.mark.timeout(PYTEST_TIMEOUT)
        def test_warm_lane() -> None:
            pass
        """
    ) == {'test_bare': 540.0, 'test_dotted': 540.0, 'test_warm_lane': 960.0}


def test_extractor_yields_none_for_an_unresolvable_expression() -> None:
    """A computed argument is UNKNOWABLE -- None, never a number.

    "None means no opinion, never too small" is the polarity
    ``_module_level_timeout_ceiling`` already establishes, and it matters here
    for a concrete population: test_laptop_warm_verify_boundary.py's five
    ``int(...)``-derived marks. Those modules carry their own file-local
    derived-budget adequacy guards, so resolving them to a guess and failing
    them would manufacture offenders whose only "fix" is deleting a sounder,
    more specific guard.
    """
    assert _sites(
        """
        import pytest

        @pytest.mark.timeout(int(ROW_BUDGET * 2))
        def test_derived() -> None:
            pass
        """
    ) == {'test_derived': None}


def test_extractor_yields_none_for_a_zero_arg_timeout_mark() -> None:
    """``timeout()`` (or one passing only ``method=``) pins no seconds."""
    assert _sites(
        """
        import pytest

        @pytest.mark.timeout()
        def test_a() -> None:
            pass

        @pytest.mark.timeout(method='signal')
        def test_b() -> None:
            pass
        """
    ) == {'test_a': None, 'test_b': None}


def test_extractor_ignores_marks_that_are_not_timeout() -> None:
    """A non-``timeout`` mark is not a site at all -- not a site with None."""
    assert (
        _sites(
            """
            import pytest

            @pytest.mark.asyncio
            @pytest.mark.slow
            def test_a() -> None:
                pass
            """
        )
        == {}
    )


def test_extractor_fails_soft_on_a_syntax_error() -> None:
    """Unparseable source yields an EMPTY tuple and never raises.

    The sweep below reads every ``*.py`` under this directory, which includes
    deliberately-malformed fixtures. A parse failure must not turn a
    TIMEOUT-COVERAGE guard red for a reason unrelated to timeout coverage --
    the very class of misattributed failure this module exists to prevent.
    """
    assert _timeout_marker_sites('def test_a(:\n') == ()


# ---------------------------------------------------------------------------
# _inverts(seconds) -- boundary table.
# ---------------------------------------------------------------------------


def test_the_band_edges_are_exactly_where_the_design_puts_them() -> None:
    """``_inverts`` is True only strictly inside ``(60, 300)``.

    THE WHOLE DESIGN LIVES IN THESE EDGES, so every one is pinned rather than
    left to a spot check.  Why the band is CLOSED at the bottom (a mark at or
    under the ini default tightens under both budgets, so it is a deliberate
    tight bound rather than an accident) and OPEN at the top (a mark at or over
    the CLI budget loosens under both): the ``VERIFY_CLI_PER_TEST_TIMEOUT``
    comment block in _orch_helpers.py.  The per-row comments record which REAL
    population each edge was chosen to admit or exclude, which is the part that
    belongs here.

    None -> False is the fail-soft polarity from :func:`_resolve_seconds`:
    "no opinion", never "too small".
    """
    assert [
        _inverts(seconds)
        for seconds in (None, 15, 60, 61, 90, 120, 150, 180, 299, 300, 360, 960)
    ] == [
        False,  # None       -- unresolvable: no opinion, never an offence
        False,  # 15         -- test_verify_clock_stop.py's watchdog marks
        False,  # 60         -- exactly the ceiling: a deliberate tight bound
        True,  # 61          -- first inverting value
        True,  # 90          -- measured, test_merge_queue.py
        True,  # 120         -- measured, the named regression instance
        True,  # 150         -- measured, test_offline_lane_integration.py
        True,  # 180         -- measured, the most common in-band value (34 sites)
        True,  # 299         -- last inverting value
        False,  # 300        -- WHOLE_TREE_SCAN / HEAVY_BARRIER / the CLI budget
        False,  # 360        -- loosens under both
        False,  # 960        -- PYTEST_TIMEOUT (warm-lane bash bucket)
    ]


# ---------------------------------------------------------------------------
# The tree-wide RATCHET.
# ---------------------------------------------------------------------------

#: Anti-vacuity FLOORS, not equalities -- 563 files and 148 marker sites
#: MEASURED at authorship time -- so the guard survives the tree growing while
#: still failing loudly if the sweep itself ever breaks (a wrong _TESTS_DIR, a
#: read that silently yields nothing, an extractor rotted to always-empty).
#: Without them a broken sweep reports zero offenders and passes, which is
#: indistinguishable from a clean tree.  The house pattern for exactly this
#: risk: test_whole_tree_scan_timeout_guard.py::_MIN_EXPECTED_TEST_FILES,
#: test_marker_registration_drift.py::_MIN_EXPECTED_TEST_FILES,
#: test_serial_merge_worker_import_guard.py::test_allowlist_has_no_stale_entries.
_MIN_EXPECTED_TEST_FILES = 400
_MIN_EXPECTED_MARKER_SITES = 100

#: The pre-existing in-band sites, MEASURED at authorship time: 61 across 18
#: modules, at 90/120/150/180s.  Entries may only ever be REMOVED, never added
#: -- a new marker in the band is what this module exists to reject, and
#: `test_no_new_inverting_timeout_marker`'s failure message says so outright.
#:
#: Not mechanically migrated in task 5147, deliberately: these are the hottest
#: files in the repo (test_merge_queue.py and its deep_* siblings,
#: test_crash_recovery.py), so rewriting ~20 of them at once would have taken
#: concurrency locks on nearly every file in-flight fleet tasks were editing
#: and would itself have risked destabilising the very verify path the task
#: existed to de-flake.  Stopping the bleeding is what prevents a fourth task
#: being blamed; the migration is ordinary follow-up work, filed as
#: agent-followup ticket tkt_0RTCC80EM92A7WD08D6RF6ZZPY.
#:
#: Keyed on ``(module, qualname)`` -- *module* being the path RELATIVE to this
#: directory (``test_cli.py``, and ``fixtures/x.py`` for anything nested), NOT
#: the basename.  The sweep rglob()s subdirectories, so a basename key would
#: silently hand a future ``tests/<subdir>/test_cli.py`` the top-level
#: test_cli.py's exemption -- exactly the collapse that per-SITE keying exists
#: to avoid.  Every entry below is top-level, so the two spellings agree today.
#: Keyed on a name rather than a line number so an entry
#: survives ordinary edits above it, and per-SITE rather than a per-file COUNT
#: because a count nets to zero when one marker is added and another removed in
#: the same file -- a hole in a guard whose entire purpose is catching
#: accidental additions.  Verbosity is cheap; a hole in the ratchet is not.
#: :func:`test_grandfather_allowlist_has_no_stale_entries` is what stops this
#: list rotting into a permanent blanket exemption.
_GRANDFATHERED: frozenset[tuple[str, str]] = frozenset(
    {
    # test_cli.py -- 1 site at 120s
    ('test_cli.py', 'test_verify_merge_cancel_end_to_end'),
    # test_crash_recovery.py -- 6 sites at 180s
    ('test_crash_recovery.py', 'TestRecoverCrashedTasksWarmLane'),
    ('test_crash_recovery.py', 'TestRecoverCrashedTasksWarmLaneEdgeCases'),
    ('test_crash_recovery.py', 'TestRecoverCrashedTasksPoolStorageAbsentGuard'),
    ('test_crash_recovery.py', 'TestRecoverCrashedTasksNoPoolConfiguredNoOp'),
    ('test_crash_recovery.py', 'TestRecordDrivenRecovery'),
    ('test_crash_recovery.py', 'TestRecordDrivenRecoveryCompatAndRelocation'),
    # test_laptop_warm_verify_boundary.py -- 2 sites at 180s
    ('test_laptop_warm_verify_boundary.py', 'test_flock_wait_env_override_speeds_up_contention_result'),
    ('test_laptop_warm_verify_boundary.py', 'test_watchdog_timeout_env_override_fires_fast_without_heartbeat'),
    # test_marker_registration_drift.py -- 2 sites at 120s
    ('test_marker_registration_drift.py', 'TestMarkerRegistrationDrift::test_every_marker_applied_under_tests_is_registered'),
    ('test_marker_registration_drift.py', 'TestMarkerRegistrationDrift::test_the_sweep_is_not_vacuous'),
    # test_merge_queue.py -- 12 sites at 90s/120s
    ('test_merge_queue.py', 'TestMergeWorker::test_cas_retry_limit_exhausted'),
    ('test_merge_queue.py', 'TestMergeWorker::test_merge_worker_emits_duration_ms_on_non_done_outcomes'),
    ('test_merge_queue.py', 'TestSpeculativeMergeWorker::test_speculative_chain_invalidation_propagates'),
    ('test_merge_queue.py', 'TestSpeculativeMergeWorker::test_speculative_merger_phase_emits_duration_ms'),
    ('test_merge_queue.py', 'TestSpeculativeMergeWorker::test_speculative_follower_chain_invalidated_after_pickup_rebase'),
    ('test_merge_queue.py', 'TestSpeculativeMergeWorker::test_chain_invalidated_pre_rebased_n2_verify_runs'),
    ('test_merge_queue.py', 'TestSpeculativeMergeWorker::test_chain_invalidated_pre_rebased_n2_red_tree_blocked'),
    ('test_merge_queue.py', 'TestBoundaryTableWorkerEntry::test_scenario_11_generation_chain_escalation'),
    ('test_merge_queue.py', 'TestSpeculationSlotSemaphoreDepth::test_k2_builds_two_speculative_ahead'),
    ('test_merge_queue.py', 'TestSpeculationPermitLeakOnMergerError::test_worktree_missing_releases_speculation_permit'),
    ('test_merge_queue.py', 'TestSpeculationPermitLeakOnMergerError::test_merger_exception_releases_speculation_permit'),
    ('test_merge_queue.py', 'TestSpeculationPermitLeakOnMergerError::test_abandoned_speculative_releases_speculation_permit'),
    # test_merge_queue_build_chain.py -- 7 sites at 180s
    ('test_merge_queue_build_chain.py', 'TestMergeBranchIntoWorktree'),
    ('test_merge_queue_build_chain.py', 'TestChainBuildLane'),
    ('test_merge_queue_build_chain.py', 'TestChainSnapshot'),
    ('test_merge_queue_build_chain.py', 'TestBuildChainDegenerate'),
    ('test_merge_queue_build_chain.py', 'TestBuildChainClean'),
    ('test_merge_queue_build_chain.py', 'TestBuildChainTruncation'),
    ('test_merge_queue_build_chain.py', 'TestMergeBranchIntoWorktreeRevParseGuard'),
    # test_merge_queue_deep_dispatch.py -- 4 sites at 180s
    ('test_merge_queue_deep_dispatch.py', 'TestDeepChainPlacementBuild'),
    ('test_merge_queue_deep_dispatch.py', 'TestRunInflightVerifyChainRedirect'),
    ('test_merge_queue_deep_dispatch.py', 'TestDeepTipVerifyNeverAdopts'),
    ('test_merge_queue_deep_dispatch.py', 'TestDeepDispatchRoundsIntegration'),
    # test_merge_queue_deep_landing.py -- 9 sites at 180s
    ('test_merge_queue_deep_landing.py', 'TestTipPassAdoptionSignal'),
    ('test_merge_queue_deep_landing.py', 'TestInOrderCasWalk'),
    ('test_merge_queue_deep_landing.py', 'TestStaleCasAbortLeavesTheRestAlone'),
    ('test_merge_queue_deep_landing.py', 'TestContendedLeaseDeferInheritance'),
    ('test_merge_queue_deep_landing.py', 'TestHeadCancelOnAdoption'),
    ('test_merge_queue_deep_landing.py', 'TestHeadCancelLeavesTheLaneIdle'),
    ('test_merge_queue_deep_landing.py', 'TestAdoptedHeadLandsWithThePostVerifyWorktree'),
    ('test_merge_queue_deep_landing.py', 'TestChainWalkConsumesNoPermits'),
    ('test_merge_queue_deep_landing.py', 'TestDeepLandingEndToEnd'),
    # test_merge_queue_request_liveness.py -- 1 site at 180s
    ('test_merge_queue_request_liveness.py', 'TestDeadVerifyAbortSelfHealsEndToEnd'),
    # test_merge_queue_restart_hook.py -- 1 site at 180s
    ('test_merge_queue_restart_hook.py', 'test_stop_does_not_preempt_finalizing_head_mid_advance'),
    # test_merge_verify_survivor_barrier.py -- 3 sites at 90s/120s
    ('test_merge_verify_survivor_barrier.py', 'TestReapMergeVerifySurvivors::test_knob_on_reaps_real_survivor_and_excludes_own_group'),
    ('test_merge_verify_survivor_barrier.py', 'TestReapMergeVerifySurvivors::test_residual_survivor_returns_false_and_logs_error'),
    ('test_merge_verify_survivor_barrier.py', 'TestReapMergeVerifySurvivors::test_integration_reap_then_reset_succeeds_on_clear_tree'),
    # test_merge_worktree_lifecycle_integration_gate.py -- 1 site at 180s
    ('test_merge_worktree_lifecycle_integration_gate.py', 'TestFiveThreeTwoSixReplayGate'),
    # test_offline_lane_infra_integration.py -- 3 sites at 120s/150s
    ('test_offline_lane_infra_integration.py', 'test_out_of_bound_spawn_counts_are_measured_not_asserted'),
    ('test_offline_lane_infra_integration.py', 'test_ib2_infra_run_in_flight_never_gates_merge'),
    ('test_offline_lane_infra_integration.py', 'test_ib4_same_infra_set_recurrence_updates_not_duplicates'),
    # test_offline_lane_integration.py -- 3 sites at 120s/150s
    ('test_offline_lane_integration.py', 'test_out_of_bound_spawn_counts_are_measured_not_asserted'),
    ('test_offline_lane_integration.py', 'test_b3_never_a_gate'),
    ('test_offline_lane_integration.py', 'test_b5_same_set_recurrence_updates_not_duplicates'),
    # test_plan_tools_startup_load.py -- 1 site at 120s
    ('test_plan_tools_startup_load.py', 'test_concurrent_startup_no_hang'),
    # test_shutdown.py -- 1 site at 120s
    ('test_shutdown.py', 'test_sigterm_exits_within_deadline'),
    # test_warm_lane_bash_bucket_placement.py -- 1 site at 120s
    ('test_warm_lane_bash_bucket_placement.py', 'test_the_configured_lane_command_actually_collects_the_bucket'),
    # test_workflow_cancellation.py -- 3 sites at 180s
    ('test_workflow_cancellation.py', 'TestRunSingleCatchHardCancel'),
    ('test_workflow_cancellation.py', 'TestSoftCancelCoversNewAwait'),
    ('test_workflow_cancellation.py', 'TestHarnessSyntheticCancelRetirement'),
    }
)


class _TreeScan(NamedTuple):
    """One pass over every ``*.py`` under :data:`_TESTS_DIR`, with sweep-health counters.

    All fields are IMMUTABLE because the scan is memoised and shared: a caller
    that mutated a list here would corrupt every later caller's view of the
    tree.

    ``sites`` pairs each timeout marker site with its module -- the path
    relative to :data:`_TESTS_DIR`, which is the first half of the
    ``(module, qualname)`` key :data:`_GRANDFATHERED` is written in.
    ``bindings`` is every assignment of a :data:`_SANCTIONED_TIMEOUT_NAMES`
    name, for :class:`TestSanctionedNameMirrors`.  Files skipped as
    ``unreadable`` are NOT counted as ``examined`` (they were not).
    """

    sites: tuple[tuple[str, _Site], ...]
    bindings: tuple[_Binding, ...]
    examined: int
    unreadable: tuple[str, ...]


@functools.cache
def _tree_scan() -> _TreeScan:
    """Read, parse and extract from every ``*.py`` under this directory -- ONCE.

    MEMOISED because FOUR tests need it and one pass is not cheap: MEASURED
    15.03s over 569 files on this loaded machine (6.06s measured unloaded).
    Uncached that is four passes where one does, and the surplus lands as
    contention on the very ``-n auto`` verify run this module exists to
    de-flake -- the same cost that motivated WHOLE_TREE_SCAN_TEST_TIMEOUT.
    Under xdist the tests may land on different workers, so the win is partial;
    it is never negative.

    Parsing dominates, which is why each file is parsed once here and the tree
    handed to BOTH extractors, rather than each extractor re-reading the file.

    Fail-soft on the READ for the same reason :func:`_parse` fails soft on the
    PARSE: a non-UTF-8 source, a deliberately-malformed encoding fixture or a
    broken symlink under this directory would otherwise raise straight out of a
    TIMEOUT-COVERAGE check. Skipped files are surfaced to the caller, so a
    sweep that silently stops reading anything cannot hide here.
    """
    sites: list[tuple[str, _Site]] = []
    bindings: list[_Binding] = []
    unreadable: list[str] = []
    examined = 0

    for py_file in sorted(_TESTS_DIR.rglob('*.py')):
        module = py_file.relative_to(_TESTS_DIR).as_posix()
        try:
            source = py_file.read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError):
            unreadable.append(module)
            continue
        examined += 1
        tree = _parse(source)
        if tree is None:
            continue
        sites.extend((module, site) for site in _timeout_marker_sites_in(tree))
        bindings.extend(_sanctioned_bindings_in(tree, module))

    return _TreeScan(tuple(sites), tuple(bindings), examined, tuple(unreadable))


def _in_band_sites() -> tuple[tuple[str, _Site], ...]:
    """:func:`_tree_scan`'s sites, narrowed to the ones that actually invert."""
    return tuple(pair for pair in _tree_scan().sites if _inverts(pair[1].seconds))


def _sweep_is_healthy(scan: _TreeScan) -> str:
    """'' when *scan* cleared :data:`_MIN_EXPECTED_TEST_FILES`, else why not.

    SPOT for the anti-vacuity floor both ratchet tests need: a broken sweep
    reports zero offenders AND reads every allowlist entry as stale, and the
    two failures want the same measurement stated the same way.
    """
    if scan.examined >= _MIN_EXPECTED_TEST_FILES:
        return ''
    return (
        f'only {scan.examined} .py files examined under {_TESTS_DIR} (expected '
        f'at least {_MIN_EXPECTED_TEST_FILES}; {len(scan.unreadable)} skipped '
        f'as unreadable: {sorted(scan.unreadable)}) -- the sweep itself is '
        'broken'
    )


def test_no_new_inverting_timeout_marker() -> None:
    """No marker in the inversion band, except the grandfathered census.

    THE RATCHET.  A marker at ``DELIBERATE_TIGHT_BOUND_CEILING < N <
    VERIFY_CLI_PER_TEST_TIMEOUT`` is too large to read as a deliberate tight
    bound, so it was written to give a slow test room -- and it silently
    becomes a TIGHTENING under verify's CLI budget.  Under ``timeout_method = "thread"`` a breach is not a red
    test: pytest-timeout ``os._exit()``s the xdist worker,
    ``--max-worker-restart=0`` declines to replace it, and the session is
    truncated with the blame landing on whatever innocent test shared the dead
    worker.  Three tasks (4176, 4384, 4405) were failed that way by ONE such
    marker.

    A RATCHET AND NOT A SWEEP, deliberately.  The 61 surviving in-band sites
    span ~20 modules, most of them the hottest files in the repo
    (test_merge_queue.py, test_merge_queue_deep_landing.py,
    test_merge_queue_build_chain.py, test_crash_recovery.py).  Rewriting them
    here would take a concurrency lock on nearly every file in-flight fleet
    tasks are editing, and would risk destabilising the very verify path this
    guard exists to de-flake.  Blocking NEW instances at commit time is what
    actually stops a fourth task being blamed; migrating the existing ones is
    ordinary follow-up work, and the stale-entry twin below is what forces the
    list to shrink as that happens.

    The SITES are grandfathered, not the FILES.  A per-file count would net to
    zero when one marker is added and another removed in the same file,
    leaving a hole in a guard whose entire purpose is catching accidental
    additions.  Verbosity is cheap; a hole in the ratchet is not.
    """
    broken = _sweep_is_healthy(_tree_scan())
    assert not broken, (
        f'{broken}, so this guard would pass vacuously rather than because '
        'the tree is clean.'
    )

    new_offenders = [
        (module, site)
        for module, site in _in_band_sites()
        if (module, site.qualname) not in _GRANDFATHERED
    ]
    if new_offenders:
        offender_list = '\n  '.join(
            f'{module}::{site.qualname} ({site.kind}, line {site.lineno}) pins '
            f'{site.seconds:g}s'
            for module, site in sorted(new_offenders, key=lambda pair: (pair[0], pair[1].qualname))
        )
        raise AssertionError(
            f'{len(new_offenders)} NEW timeout marker(s) in the inversion band '
            f'({DELIBERATE_TIGHT_BOUND_CEILING} < N < {VERIFY_CLI_PER_TEST_TIMEOUT}).\n\n'
            'A marker there is a TWO-WAY override, not a floor: it REPLACES '
            'the ambient budget in both directions, so a number big enough to '
            'give a slow test room, yet below the '
            f'--timeout={VERIFY_CLI_PER_TEST_TIMEOUT} verify passes, silently '
            'tightens the run that actually gates your merge. Exceeding it '
            "does NOT fail the test: pytest-timeout's thread method os._exit()s "
            'the xdist worker, --max-worker-restart=0 declines to replace it, '
            'and the truncated session blames an innocent test that merely '
            'shared it.\n\n'
            'Write one of:\n\n'
            '    from _orch_helpers import VERIFY_CLI_PER_TEST_TIMEOUT\n'
            '    @pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)   # slow test\n\n'
            f'    @pytest.mark.timeout(N)  # N <= {DELIBERATE_TIGHT_BOUND_CEILING}, a '
            'DELIBERATE tight bound\n\n'
            'The second is for a test that asserts something happens FAST (see '
            "test_verify_clock_stop.py's 15s watchdog marks); it is small "
            'enough to read as that deliberate bound, which is why it is '
            'allowed. Anything in between '
            'inverts. Full rationale: the VERIFY_CLI_PER_TEST_TIMEOUT comment '
            'block in _orch_helpers.py.\n\n'
            '_GRANDFATHERED is a shrinking census of pre-existing sites and may '
            'only ever have entries REMOVED -- do not add yours to it.'
            f'\n\nOffending sites:\n  {offender_list}'
        )


def test_grandfather_allowlist_has_no_stale_entries() -> None:
    """Every ``_GRANDFATHERED`` entry must still name a real in-band site.

    The ratchet self-tightens: as the follow-up migration raises these markers,
    their entries stop matching and must be deleted, so the list can never rot
    into a permanent blanket exemption that silently re-admits a site someone
    later re-adds under the same name.  Same shape, and same reason, as
    test_serial_merge_worker_import_guard.py::test_allowlist_has_no_stale_entries.
    """
    broken = _sweep_is_healthy(_tree_scan())
    assert not broken, f'{broken}, so EVERY allowlist entry would read as stale.'

    live = {(module, site.qualname) for module, site in _in_band_sites()}
    stale = sorted(_GRANDFATHERED - live)

    assert not stale, (
        f'{len(stale)} _GRANDFATHERED entr(y/ies) no longer correspond to an '
        'in-band timeout marker -- delete them, the ratchet is supposed to '
        'shrink. (The marker was raised, removed, or its test renamed; in the '
        'rename case re-add nothing, the new name must stand on its own.)\n  '
        + '\n  '.join(f'{module}::{qualname}' for module, qualname in stale)
    )


def test_the_marker_census_is_not_vacuous() -> None:
    """The sweep must find a substantial population of markers, in-band or not.

    Distinct from the file floor above and load-bearing in a way it is not: a
    correct ``_TESTS_DIR`` with an EXTRACTOR rotted to always-empty would
    examine 563 files, find zero sites, report zero offenders and pass. This is
    the floor that catches that. 148 sites measured at authorship time.
    """
    scan = _tree_scan()

    assert len(scan.sites) >= _MIN_EXPECTED_MARKER_SITES, (
        f'only {len(scan.sites)} timeout marker site(s) found across '
        f'{scan.examined} files (expected at least '
        f'{_MIN_EXPECTED_MARKER_SITES}) -- '
        '_timeout_marker_sites has probably stopped matching, so the ratchet '
        'would pass vacuously. Check it against the inline fixtures above.'
    )


def test_the_deep_gate_module_pins_every_timeout_by_name() -> None:
    """No bare number may pin a timeout in the deep merge-queue gate.

    WHY THIS MODULE GETS A RATCHET ITS NEIGHBOURS DO NOT.  All eight real-git
    classes in it carried an IDENTICAL bare ``300`` -- classes whose measured
    costs differ by more than 2x.  That uniformity is precisely what made
    TestRow7KillSwitchByteIdentity's under-sizing invisible: a reader had no
    way to see that one of the eight was roughly twice the weight of its
    neighbours, because the file said the same thing about all of them.

    A NAME fixes both halves of that.  It makes the next re-derivation ONE
    edit instead of eight, and it makes a divergent class conspicuous -- the
    file now states which classes are sized by the verify budget and which by
    their own measurement, in the marker itself rather than in a comment that
    can drift from it.

    A FLOOR, not a proof: this checks the SPELLING. The value behind a
    sanctioned name is held honest separately, by
    :class:`TestSanctionedNameMirrors`.
    """
    sites = [site for module, site in _tree_scan().sites if module == _DEEP_GATE_MODULE]

    assert len(sites) >= _MIN_DEEP_GATE_MARKER_SITES, (
        f'only {len(sites)} timeout marker site(s) found in '
        f'{_DEEP_GATE_MODULE} (expected at least '
        f'{_MIN_DEEP_GATE_MARKER_SITES}). Either the module was renamed -- '
        'update _DEEP_GATE_MODULE -- or its real-git classes lost their '
        'markers, which is the condition this sweep exists to prevent and '
        'would otherwise pass here VACUOUSLY, green because it found nothing '
        'rather than because it found nothing wrong.'
    )

    unnamed = sorted(
        (site for site in sites if site.spelling not in _DEEP_GATE_SPELLINGS),
        key=lambda site: site.lineno,
    )

    assert not unnamed, (
        f'{len(unnamed)} timeout marker(s) in {_DEEP_GATE_MODULE} pin a value '
        'that is not one of the sanctioned constants '
        f'{sorted(_DEEP_GATE_SPELLINGS)}.\n\n'
        'Every real-git class in that file once carried an identical bare '
        '300, and that uniformity is what hid TestRow7KillSwitchByteIdentity '
        'being roughly twice the weight of its neighbours until it had cost '
        'five recorded xdist worker crashes. Spelling the budget as a NAME '
        'makes a re-derivation one edit rather than eight, and makes a class '
        'whose budget genuinely differs conspicuous instead of invisible.\n\n'
        'Import the constant from _orch_helpers rather than writing the '
        'number: VERIFY_CLI_PER_TEST_TIMEOUT for a class sized by the verify '
        'budget, DEEP_GATE_SCENE_TEST_TIMEOUT for one sized by its own '
        'MEASURED spawn count. If a new class needs a third budget, measure '
        'it and add a named constant -- do not reach for a literal.\n\n'
        + '\n'.join(
            f'  {_DEEP_GATE_MODULE}:{site.lineno} {site.qualname} '
            f'({site.kind}) pins {site.spelling or "<no argument>"}'
            for site in unnamed
        )
    )
