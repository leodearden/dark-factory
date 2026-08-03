"""Static guard: every pytest marker applied under orchestrator/tests/ must be
registered, so a typo fails loudly instead of silently degrading to an
unknown-marker warning forever (task 3532, follow-up to task 3506).

This module is built incrementally. This stage unit-tests the pure marker-
application extractor, ``_applied_marker_names``, against synthetic source
strings only — no disk I/O, no pytest config.
"""
from __future__ import annotations

import ast

import pytest


def _marker_name(element: ast.expr) -> str | None:
    """The marker name in a ``pytest.mark.NAME`` / ``pytest.mark.NAME(...)`` element.

    Mirrors ``orchestrator.pytest_markers._marker_name`` BY CONSTRUCTION, not
    by import — see the module docstring's note on the two modules'
    deliberately opposite fail-safe polarities. Anything else — a bare
    constant, a local name, an unrelated attribute chain — yields None and is
    skipped silently, without suppressing its siblings.
    """
    if isinstance(element, ast.Call):
        element = element.func
    if not isinstance(element, ast.Attribute):
        return None
    owner = element.value
    if (
        isinstance(owner, ast.Attribute)
        and owner.attr == 'mark'
        and isinstance(owner.value, ast.Name)
        and owner.value.id == 'pytest'
    ):
        return element.attr
    return None


def _is_pytestmark_target(node: ast.expr) -> bool:
    """True iff *node* is the bare name ``pytestmark``."""
    return isinstance(node, ast.Name) and node.id == 'pytestmark'


def _pytestmark_value(statement: ast.stmt) -> ast.expr | None:
    """The value *statement* binds to ``pytestmark``, else None.

    Covers both the plain ``pytestmark = ...`` and the annotated
    ``pytestmark: list = ...`` spellings; an annotation with no value binds
    nothing. Anything other than an ``Assign``/``AnnAssign`` — including
    every ``ast.expr`` node ``ast.walk`` also yields — falls through to None.
    """
    if isinstance(statement, ast.Assign):
        if any(_is_pytestmark_target(target) for target in statement.targets):
            return statement.value
        return None
    if isinstance(statement, ast.AnnAssign) and _is_pytestmark_target(statement.target):
        return statement.value
    return None


def _names_from_marker_value(value: ast.expr) -> frozenset[str]:
    """Accepted value shapes: a bare element, or a list/tuple of elements."""
    elements = list(value.elts) if isinstance(value, ast.List | ast.Tuple) else [value]
    return frozenset(
        name for name in (_marker_name(element) for element in elements) if name is not None
    )


def _applied_marker_names(source: str) -> frozenset[str]:
    """Every pytest marker name *source* APPLIES via a decorator or ``pytestmark``.

    An UPPER-BOUND sweep: walked with ``ast.walk`` (not just ``tree.body``),
    so class-level ``pytestmark`` and decorators on nested classes are
    reached — the deliberate difference from
    ``orchestrator.pytest_markers.module_level_marker_names``, which walks
    only ``tree.body`` because its contract is a module-wide LOWER bound.
    Collects from two node kinds: the ``decorator_list`` of every
    ``FunctionDef``/``AsyncFunctionDef``/``ClassDef``, and the value of any
    ``Assign``/``AnnAssign`` binding the name ``pytestmark`` (unpacking a
    ``List``/``Tuple`` value into its elements).

    KNOWN LIMITATIONS, acknowledged rather than silently omitted:
    ``pytest.param(marks=...)`` and ``item.add_marker(...)`` are not swept —
    zero uses under ``orchestrator/tests/`` today (grep-verified); nor is a
    marker applied through an aliased pytest import
    (``import pytest as _pytest``) — two such aliases exist today
    (test_overlap_footprint.py, test_multihost_verify_integration.py) but
    both use the alias only for ``.raises``, never ``.mark``, so this is a
    real but currently-inert gap.

    LOUD ON FAILURE — the OPPOSITE of ``pytest_markers``' fail-safe polarity:
    a ``SyntaxError`` from ``ast.parse`` is left to propagate rather than
    being swallowed into an empty set, because a silent empty set here would
    make a drift guard built on top of this silently vacuous.
    """
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            for decorator in node.decorator_list:
                name = _marker_name(decorator)
                if name is not None:
                    names.add(name)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            value = _pytestmark_value(node)
            if value is not None:
                names |= _names_from_marker_value(value)
    return frozenset(names)


class TestAppliedMarkerNames:
    """``_applied_marker_names(source: str) -> frozenset[str]``.

    Every marker name a module APPLIES via a decorator (function or class) or
    a ``pytestmark`` assignment (module- or class-level). Pure: parses a
    source string, touches no filesystem. Unlike
    ``orchestrator.pytest_markers.module_level_marker_names`` — a LOWER-BOUND
    walk of ``tree.body`` only, by design, because it feeds a widen-only
    decision — this is an UPPER-BOUND sweep that also reaches class-level
    ``pytestmark`` and decorators on nested classes, because for a drift
    guard under-sweeping is the unsafe direction: a missed marker is missed
    drift.
    """

    # -- application forms that occur for real under orchestrator/tests/ ------

    def test_bare_function_decorator(self):
        """``@pytest.mark.slow`` — mirrors
        test_verify_plan_integration.py::test_scoped_pytest_producer_and_runner_agree_on_scope.
        """
        source = (
            'import pytest\n\n\n'
            '@pytest.mark.slow\n'
            'def test_scoped_pytest_producer_and_runner_agree_on_scope():\n'
            '    pass\n'
        )
        assert _applied_marker_names(source) == frozenset({'slow'})

    def test_call_form_function_decorator(self):
        """``@pytest.mark.timeout(120)`` — mirrors
        test_warm_lane_bash_bucket_placement.py:228."""
        source = (
            'import pytest\n\n\n'
            '@pytest.mark.timeout(120)\n'
            'def test_the_configured_lane_command_actually_collects_the_bucket():\n'
            '    pass\n'
        )
        assert _applied_marker_names(source) == frozenset({'timeout'})

    def test_class_decorator(self):
        """A marker decorator applied directly to a test class."""
        source = (
            'import pytest\n\n\n'
            "@pytest.mark.usefixtures('x')\n"
            'class TestThing:\n'
            '    def test_it(self):\n'
            '        pass\n'
        )
        assert _applied_marker_names(source) == frozenset({'usefixtures'})

    def test_module_level_pytestmark_bare(self):
        """``pytestmark = pytest.mark.asyncio`` — mirrors
        test_provenance_gate_integration.py:46."""
        source = 'import pytest\n\npytestmark = pytest.mark.asyncio\n'
        assert _applied_marker_names(source) == frozenset({'asyncio'})

    def test_module_level_pytestmark_list(self):
        """Verbatim shape of test_warm_lane_bash_suite.py:240-244."""
        source = (
            'import pytest\n\n'
            'pytestmark = [\n'
            '    pytest.mark.warm_lane_bash,\n'
            '    pytest.mark.timeout(120),\n'
            "    pytest.mark.xdist_group('warm_lane_bash'),\n"
            ']\n'
        )
        assert _applied_marker_names(source) == frozenset({
            'warm_lane_bash', 'timeout', 'xdist_group',
        })

    def test_module_level_pytestmark_tuple(self):
        source = 'import pytest\n\npytestmark = (pytest.mark.asyncio, pytest.mark.slow)\n'
        assert _applied_marker_names(source) == frozenset({'asyncio', 'slow'})

    def test_class_level_pytestmark(self):
        """``pytestmark = pytest.mark.asyncio`` bound INSIDE a class body —
        mirrors test_stranded_verified_green.py:267
        (``class TestDetectVerifiedGreen: pytestmark = pytest.mark.asyncio``).
        This is precisely the form ``module_level_marker_names`` deliberately
        excludes (``tree.body``-only); this extractor must NOT exclude it.
        """
        source = (
            'import pytest\n\n\n'
            'class TestDetectVerifiedGreen:\n'
            '    pytestmark = pytest.mark.asyncio\n\n'
            '    async def test_positive_match(self):\n'
            '        pass\n'
        )
        assert _applied_marker_names(source) == frozenset({'asyncio'})

    def test_multiple_stacked_decorators(self):
        source = (
            'import pytest\n\n\n'
            '@pytest.mark.slow\n'
            '@pytest.mark.timeout(30)\n'
            'def test_thing():\n'
            '    pass\n'
        )
        assert _applied_marker_names(source) == frozenset({'slow', 'timeout'})

    # -- negative cases: must yield NO names -----------------------------------

    def test_marker_name_inside_a_string_literal_is_ignored(self):
        """The exact trap AST protects against. test_pytest_marker_deselection.py
        embeds real marker shapes as SYNTHETIC SOURCE STRING CONSTANTS at its
        lines 376/481/661 (e.g. ``_MARKED_SOURCE = 'import pytest\\npytestmark
        = pytest.mark.warm_lane_bash\\n...'``). A grep sweep would misread
        that line as an applied marker; ast sees an ``ast.Constant`` string
        and correctly ignores it — this is why AST is used instead of grep.
        """
        source = (
            "_MARKED_SOURCE = 'import pytest\\npytestmark = "
            "pytest.mark.warm_lane_bash\\n\\n\\ndef test_x():\\n    pass\\n'\n"
        )
        assert _applied_marker_names(source) == frozenset()

    def test_unrelated_attribute_chain_is_ignored(self):
        source = (
            'import other\n\n\n'
            '@other.mark.thing\n'
            'def test_thing():\n'
            '    pass\n'
        )
        assert _applied_marker_names(source) == frozenset()

    def test_functools_wraps_decorator_is_ignored(self):
        source = (
            'import functools\n\n\n'
            'def _identity(f):\n'
            '    return f\n\n\n'
            '@functools.wraps(_identity)\n'
            'def test_thing():\n'
            '    pass\n'
        )
        assert _applied_marker_names(source) == frozenset()

    def test_pytestmark_named_local_with_non_marker_value_is_ignored(self):
        """A ``pytestmark``-named local whose value is not a marker access."""
        source = (
            'def f():\n'
            "    pytestmark = 'not a real marker'\n"
            '    return pytestmark\n'
        )
        assert _applied_marker_names(source) == frozenset()

    # -- loud failure: the OPPOSITE of pytest_markers.py's swallow policy -----

    def test_syntax_error_raises_not_swallowed(self):
        """Opposite of ``orchestrator.pytest_markers``' fail-safe polarity: a
        silent empty set here would be a silently-vacuous drift guard, so an
        unparseable file must RAISE rather than resolve to "no markers".
        """
        with pytest.raises(SyntaxError):
            _applied_marker_names('def f(:\n    pass\n')
