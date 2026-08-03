"""Static guard: every pytest marker applied under orchestrator/tests/ must be
registered, so a typo fails loudly instead of silently degrading to an
unknown-marker warning forever (task 3532, follow-up to task 3506).

This module is built incrementally. This stage unit-tests the pure marker-
application extractor, ``_applied_marker_names``, against synthetic source
strings only — no disk I/O, no pytest config.
"""
from __future__ import annotations

import pytest


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
