"""Tests for scripts/check_method_param_wiring.py (task 3364).

The script exists because ``orchestrator/delivered_checks.py::_run_grep_check``
shells to ``git grep -E`` — single-line POSIX ERE — so a file-scoped pattern
cannot be pinned to one multi-line ``def``. These tests pin the AST analyzer
that replaces it for capability ``qdrant-vector-access-for-ann`` (task 3210).

Imports the module under test bare (``import check_method_param_wiring``);
``scripts/tests/conftest.py`` already puts ``scripts/`` on ``sys.path``. No
first-party package is imported here — that is load-bearing for the fallback
verify chain (see the conftest docstring).
"""
import ast

import pytest

import check_method_param_wiring as cmpw

# Mirrors mem0_client.py's real shape: a class with two async methods, each
# signature spread one parameter per line — the exact layout that defeats a
# single-line grep. `scroll_by_metadata` declares keyword-only
# `with_vectors: bool = False`; `get_point_by_id` declares no such parameter
# and its only `with_vectors=False` is a hardcoded literal on a different call.
REAL_SHAPE = '''
import asyncio


class Mem0Client:
    async def scroll_by_metadata(
        self,
        scope: Scope,
        filters: dict[str, Any],
        limit: int = 1000,
        *,
        with_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        """Deterministic enumeration."""
        client = await self._get_async_qdrant()
        points, _next_offset = await asyncio.wait_for(
            client.scroll(
                collection_name=collection_name,
                scroll_filter=qdrant_filter,
                with_payload=True,
                with_vectors=with_vectors,
                limit=limit,
            ),
            timeout=self._read_timeout,
        )
        return points

    async def get_point_by_id(self, memory_id: str, scope: Scope) -> dict | None:
        """Direct point-fetch by id."""
        client = await self._get_async_qdrant()
        records = await asyncio.wait_for(
            client.retrieve(
                collection_name=collection_name,
                ids=[memory_id],
                with_payload=True,
                with_vectors=False,
            ),
            timeout=self._read_timeout,
        )
        return records[0] if records else None
'''


@pytest.fixture
def real_tree() -> ast.Module:
    return cmpw.parse_module(REAL_SHAPE)


def _resolved(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Resolve *name*, failing loudly when the fixture does not declare it.

    Also narrows away `resolve_function`'s `| None` for pyright, which
    `scripts/orchestrator.yaml` runs over this directory.
    """
    fn = cmpw.resolve_function(tree, name)
    assert fn is not None, f'fixture module declares no function {name!r}'
    return fn


class TestResolveFunction:
    """`resolve_function` walks the whole module, not just its top level."""

    def test_finds_async_method_nested_in_class(self, real_tree):
        fn = cmpw.resolve_function(real_tree, 'scroll_by_metadata')
        assert isinstance(fn, ast.AsyncFunctionDef)
        assert fn.name == 'scroll_by_metadata'

    def test_finds_plain_def_at_module_level(self):
        tree = cmpw.parse_module('def helper(a, b):\n    return a + b\n')
        fn = cmpw.resolve_function(tree, 'helper')
        assert isinstance(fn, ast.FunctionDef)
        assert fn.name == 'helper'

    def test_absent_function_returns_none(self, real_tree):
        assert cmpw.resolve_function(real_tree, 'no_such_name') is None

    def test_ambiguity_is_explicit_not_silent_first_match(self):
        """Two same-named defs must NOT silently resolve to one of them.

        A check that picks arbitrarily between two candidates is not
        method-scoped; it is grep with extra steps.
        """
        tree = cmpw.parse_module(
            'class A:\n'
            '    def dup(self, with_vectors: bool = False):\n'
            '        pass\n'
            '\n'
            '\n'
            'class B:\n'
            '    def dup(self, other: int = 0):\n'
            '        pass\n'
        )
        with pytest.raises(cmpw.AmbiguousFunction) as excinfo:
            cmpw.resolve_function(tree, 'dup')
        assert 'dup' in str(excinfo.value)


class TestDeclaresParam:
    """`declares_param` inspects the resolved function's own signature."""

    def test_true_for_keyword_only_declaration(self, real_tree):
        fn = _resolved(real_tree, 'scroll_by_metadata')
        assert cmpw.declares_param(fn, 'with_vectors') is True

    def test_false_for_the_method_that_does_not_declare_it(self, real_tree):
        """`get_point_by_id` has no `with_vectors` parameter.

        Its `with_vectors=False` is a literal keyword on `client.retrieve`.
        The superseded whole-file grep could not tell these two apart.
        """
        fn = _resolved(real_tree, 'get_point_by_id')
        assert cmpw.declares_param(fn, 'with_vectors') is False

    def test_finds_positional_or_keyword_param(self):
        """Must not silently depend on the `*` in today's signature."""
        tree = cmpw.parse_module('def f(self, with_vectors: bool = False):\n    pass\n')
        fn = _resolved(tree, 'f')
        assert cmpw.declares_param(fn, 'with_vectors') is True

    def test_finds_positional_only_param(self):
        tree = cmpw.parse_module('def f(with_vectors: bool = False, /):\n    pass\n')
        fn = _resolved(tree, 'f')
        assert cmpw.declares_param(fn, 'with_vectors') is True


class TestDeclaresParamAnnotation:
    """Annotation matching preserves what the grep pattern asserted."""

    def test_matching_annotation_is_true(self, real_tree):
        fn = _resolved(real_tree, 'scroll_by_metadata')
        assert cmpw.declares_param(fn, 'with_vectors', annotation='bool') is True

    def test_mismatched_annotation_is_false(self, real_tree):
        fn = _resolved(real_tree, 'scroll_by_metadata')
        assert cmpw.declares_param(fn, 'with_vectors', annotation='int') is False

    def test_unannotated_param_fails_a_required_annotation(self):
        tree = cmpw.parse_module('def f(self, with_vectors=False):\n    pass\n')
        fn = _resolved(tree, 'f')
        assert cmpw.declares_param(fn, 'with_vectors', annotation='bool') is False

    def test_unannotated_param_passes_when_no_annotation_required(self):
        tree = cmpw.parse_module('def f(self, with_vectors=False):\n    pass\n')
        fn = _resolved(tree, 'f')
        assert cmpw.declares_param(fn, 'with_vectors', annotation=None) is True


class TestForwardsParamTo:
    """Declaration alone is NOT the capability.

    `qdrant-vector-access-for-ann` is vector *access*, not a signature shape. A
    stub `with_vectors: bool = False` that is accepted and then dropped would
    satisfy a declaration-only check while delivering nothing — the same
    hollow-DELIVERED failure mode as the file-scoped grep, one level in.
    """

    def test_true_for_the_real_forward(self, real_tree):
        """`client.scroll(..., with_vectors=with_vectors, ...)` on main."""
        fn = _resolved(real_tree, 'scroll_by_metadata')
        assert cmpw.forwards_param_to(fn, 'with_vectors', 'scroll') is True

    def test_false_for_a_hardcoded_literal(self, real_tree):
        """`client.retrieve(..., with_vectors=False)` is NOT a forward.

        This is exactly `get_point_by_id`'s shape on main (mem0_client.py:839)
        and is the discrimination the whole task turns on: an `ast.Constant`
        value means the parameter is not reaching the call.
        """
        fn = _resolved(real_tree, 'get_point_by_id')
        assert cmpw.forwards_param_to(fn, 'with_vectors', 'retrieve') is False

    def test_false_when_keyword_forwards_a_different_name(self):
        tree = cmpw.parse_module(
            'def f(with_vectors: bool = False):\n'
            '    return client.scroll(with_vectors=want_vectors)\n'
        )
        fn = _resolved(tree, 'f')
        assert cmpw.forwards_param_to(fn, 'with_vectors', 'scroll') is False

    def test_false_when_the_right_forward_goes_to_the_wrong_callee(self):
        tree = cmpw.parse_module(
            'def f(with_vectors: bool = False):\n'
            '    return client.query_points(with_vectors=with_vectors)\n'
        )
        fn = _resolved(tree, 'f')
        assert cmpw.forwards_param_to(fn, 'with_vectors', 'scroll') is False

    def test_matches_on_attribute_name_regardless_of_receiver(self):
        """`other.scroll(...)` still matches `callee='scroll'`.

        The check binds to the method NAME, not to the receiver expression —
        resolving the receiver would need type inference the AST cannot give.
        """
        tree = cmpw.parse_module(
            'def f(with_vectors: bool = False):\n'
            '    return other.scroll(with_vectors=with_vectors)\n'
        )
        fn = _resolved(tree, 'f')
        assert cmpw.forwards_param_to(fn, 'with_vectors', 'scroll') is True

    def test_true_for_a_bare_call_callee(self):
        """A plain `ast.Name` func, not just an `ast.Attribute`."""
        tree = cmpw.parse_module(
            'def f(with_vectors: bool = False):\n'
            '    return scroll(with_vectors=with_vectors)\n'
        )
        fn = _resolved(tree, 'f')
        assert cmpw.forwards_param_to(fn, 'with_vectors', 'scroll') is True

    def test_true_when_the_call_is_nested_in_another_expression(self):
        """On main the real call sits inside `await asyncio.wait_for(...)`.

        A body scan inspecting only top-level statements would miss it. The
        `real_tree` fixture carries that exact nesting; this pins the property
        in isolation.
        """
        tree = cmpw.parse_module(
            'async def f(with_vectors: bool = False):\n'
            '    points, _ = await asyncio.wait_for(\n'
            '        client.scroll(with_vectors=with_vectors),\n'
            '        timeout=30,\n'
            '    )\n'
            '    return points\n'
        )
        fn = _resolved(tree, 'f')
        assert cmpw.forwards_param_to(fn, 'with_vectors', 'scroll') is True

    def test_a_forward_in_a_different_function_does_not_count(self):
        """Walk `fn`, never the module — this is what keeps it method-scoped."""
        tree = cmpw.parse_module(
            'def target(with_vectors: bool = False):\n'
            '    return client.scroll(with_payload=True)\n'
            '\n'
            '\n'
            'def neighbour(with_vectors: bool = False):\n'
            '    return client.scroll(with_vectors=with_vectors)\n'
        )
        fn = _resolved(tree, 'target')
        assert cmpw.forwards_param_to(fn, 'with_vectors', 'scroll') is False
