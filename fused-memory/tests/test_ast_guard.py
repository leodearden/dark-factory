"""Unit coverage for the shared AST machinery in _ast_guard.py."""

import ast

import pytest
from _ast_guard import calls_named, imported_names_from, parse_python_module

# ---------------------------------------------------------------------------
# Tests for the shared AST migration-guard machinery (task 3502 / 3574)
# ---------------------------------------------------------------------------
# parse_python_module() and calls_named() are the single parse/search
# implementation behind three migration guards — test_falkor_probe_routing_guard,
# test_falkor_index_barrier_guard and test_gather_idiom_helper_routing. Those
# guards are themselves the enforcement layer for two false-green mechanisms
# (an unbarriered async index build; a hand-rolled gather Pass-2), so a silent
# weakening HERE disarms all three at once while every one of them still
# reports green. The semantics they rest on are pinned below.
#
# NOTE the sample identifiers are deliberately inert (`widget`, `mod.widget`)
# and no snippet contains an index-creating literal. test_falkor_index_barrier_guard
# discovers its parametrized module set by selecting every tests/test_*.py that
# holds BOTH a /CREATE\s+INDEX|createNodeIndex/i string constant AND a real
# ast.Call to select_graph. This file already mentions select_graph in prose, so
# a sample that combined a select_graph(...) call with an index-creation string
# would drag THIS file into that guard's scope and fail it — and the failure
# would read as a barrier-guard bug rather than a bad fixture here.
# ---------------------------------------------------------------------------

# A bare call, an attribute call, an unrelated callee, and the SAME token
# appearing only inside a string constant.
_CALLS_SNIPPET = """
widget(1)
mod.widget(2)
gadget(3)
note = 'widget(4) is described here but never called'
"""

# Calls in three distinct positions so a node-scoped query can be shown to see
# only its own subtree.
_NESTED_SNIPPET = """
@decorate(widget('in-decorator'))
def decorated():
    pass

assigned = [widget('in-assignment'), gadget('sibling')]

def elsewhere():
    return widget('in-function-body')
"""


class TestSharedAstGuardMachinery:
    """Unit coverage for parse_python_module() / calls_named()."""

    def test_parses_arbitrary_python_source(self, tmp_path):
        """Any Python file parses — which is why the name says module, not TEST module.

        test_gather_idiom_helper_routing asserts over src/fused_memory/*.py, so
        a parse that only claimed to handle test modules would be lying at one
        of its three call sites. The helper is path-agnostic (read_text +
        ast.parse), so a module written here pins that property without
        coupling this file to some production module's current location.
        """
        source = tmp_path / 'prod.py'
        source.write_text('def f():\n    return 1\n')

        tree = parse_python_module(source)

        assert isinstance(tree, ast.Module)
        assert tree.body, 'parsed an empty module — the parse silently read nothing'

    def test_repeated_parses_return_the_identical_tree(self, tmp_path):
        """Memoised: the same path yields the SAME object, not an equal copy.

        Consequence every consumer must respect: the tree is SHARED across
        guards and across tests within a session, so no caller may mutate it
        (no ast.NodeTransformer in place, no attribute assignment on nodes).
        A mutation here would silently corrupt an unrelated guard's view of the
        same file.
        """
        source = tmp_path / 'memoised.py'
        source.write_text('value = 1\n')

        assert parse_python_module(source) is parse_python_module(source)

    def test_missing_path_raises_assertion_error(self, tmp_path):
        """A path that does not exist fails loudly rather than parsing empty.

        A guard whose target file was renamed must break, not quietly assert
        over an empty tree and report green having checked nothing.
        """
        missing = tmp_path / 'nope.py'

        with pytest.raises(AssertionError):
            parse_python_module(missing)

    def test_finds_bare_and_attribute_calls_but_not_string_mentions(self):
        """Bare `widget()` and `mod.widget()` match; the same token in a string does not.

        The attribute form is what makes `asyncio.gather(...)` and
        `db.select_graph(...)` match in the real guards. The string clause is
        the AST-not-grep property every guard docstring claims: prose that
        merely *describes* the idiom being migrated must not satisfy or trip a
        check.
        """
        tree = ast.parse(_CALLS_SNIPPET)

        found = calls_named(tree, 'widget')

        assert len(found) == 2, (
            f'expected the bare and attribute calls only, got {len(found)} — a '
            f'string constant mentioning the name must not count.'
        )
        assert calls_named(tree, 'gadget'), 'an unrelated callee should still match its own name'
        assert calls_named(tree, 'absent') == [], 'a name that is never called must match nothing'

    def test_returns_real_call_nodes_with_keywords_and_lineno(self):
        """Results keep `.keywords` and `.lineno` — both are load-bearing.

        The gather guard filters results on `kw.arg == 'return_exceptions'`, and
        all three guards print `.lineno` in their failure messages so a
        developer can find the offending call.
        """
        tree = ast.parse('x = widget(payload, return_exceptions=True)\n')

        (call,) = calls_named(tree, 'widget')

        assert isinstance(call, ast.Call)
        assert [kw.arg for kw in call.keywords] == ['return_exceptions']
        assert call.lineno == 1

    def test_accepts_a_non_module_node_and_searches_only_that_subtree(self):
        """Any ast.AST, not just a Module — so a guard can ask "is it called *here*".

        The probe-routing guard uses exactly this to distinguish a marker call
        in a gating position (a decorator, or the value of a `pytestmark`
        assignment) from one that merely builds a marker object and drops it.
        """
        tree = ast.parse(_NESTED_SNIPPET)
        (decorated,) = [
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'decorated'
        ]
        (decorator,) = decorated.decorator_list

        scoped = calls_named(decorator, 'widget')

        assert len(scoped) == 1, f'node-scoped search leaked outside its subtree: {scoped}'
        marker = scoped[0].args[0]
        assert isinstance(marker, ast.Constant) and marker.value == 'in-decorator'
        assert len(calls_named(tree, 'widget')) == 3, (
            'the whole-module search should still see every call — the narrowing '
            'must come from the node passed in, not from the search itself.'
        )


# The two properties below are what the barrier guard's "must import the shared
# barrier" clause and the gather guard's "must import a named helper" clause
# both rest on, so they are pinned rather than left to the two former copies to
# agree by luck.
_IMPORTS_SNIPPET = """
from pkg.mod import alpha, beta
from pkg.other import gamma
from pkg.mod import delta as renamed
import pkg.mod
from . import relative
"""

# The same module imported inside two sibling function bodies, so a node-scoped
# query can be shown to see only its own subtree.
_SCOPED_IMPORTS_SNIPPET = """
def inner():
    from pkg.mod import scoped

def sibling():
    from pkg.mod import elsewhere
"""


class TestImportedNamesFrom:
    """Unit coverage for imported_names_from(node, module)."""

    def test_collects_names_imported_from_the_named_module(self):
        assert imported_names_from(ast.parse(_IMPORTS_SNIPPET), 'pkg.mod') == {
            'alpha',
            'beta',
            'delta',
        }

    def test_excludes_names_imported_from_other_modules(self):
        """`gamma` comes from pkg.other and must not leak into a pkg.mod query."""
        assert 'gamma' not in imported_names_from(ast.parse(_IMPORTS_SNIPPET), 'pkg.mod')
        assert imported_names_from(ast.parse(_IMPORTS_SNIPPET), 'pkg.other') == {'gamma'}

    def test_unimported_module_returns_an_empty_set(self):
        """Empty, not None — both call sites render it as `imported or "nothing"`.

        A None return would print "None" in a guard's failure message and break
        the set intersection the gather guard performs on the result.
        """
        result = imported_names_from(ast.parse(_IMPORTS_SNIPPET), 'pkg.absent')

        assert result == set()
        assert not result

    def test_plain_import_statement_contributes_nothing(self):
        """`import pkg.mod` is an ast.Import, not an ast.ImportFrom.

        It binds the module, never a name FROM it, so it cannot satisfy a
        guard demanding that a specific helper be imported.
        """
        assert imported_names_from(ast.parse('import pkg.mod\n'), 'pkg.mod') == set()

    def test_aliased_import_reports_the_source_side_name(self):
        """`from pkg.mod import delta as renamed` reports `delta`, not `renamed`.

        Load-bearing: both guards compare the result against SOURCE-side names
        (`await_index_operational`, `gather_or_raise`), so reporting the local
        binding instead would make a legitimately-aliased import look missing.
        """
        tree = ast.parse('from pkg.mod import delta as renamed\n')

        assert imported_names_from(tree, 'pkg.mod') == {'delta'}

    def test_relative_import_does_not_match_a_named_module(self):
        """`from . import relative` has `node.module is None` — must not match.

        Guards against a None/str comparison crash or a spurious match when a
        parsed module happens to use relative imports.
        """
        assert imported_names_from(ast.parse('from . import relative\n'), 'pkg.mod') == set()
        assert 'relative' not in imported_names_from(ast.parse(_IMPORTS_SNIPPET), 'pkg.mod')

    def test_star_import_reports_the_literal_star(self):
        """`from pkg.mod import *` reports `'*'` — an edge, pinned as deliberate.

        `alias.name` is the literal `'*'` for a star import, so both former
        forks behaved this way and the extraction preserved it rather than
        quietly filtering it. The consequence is visible at the call sites: a
        star-importing module reads as importing `['*']`, which satisfies no
        guard demanding a NAMED helper. Pinned so the behaviour is a documented
        choice rather than a surprise in a failure message.
        """
        assert imported_names_from(ast.parse('from pkg.mod import *\n'), 'pkg.mod') == {'*'}

    def test_accepts_a_non_module_node_and_searches_only_that_subtree(self):
        """Any ast.AST, not just a Module — the symmetry with calls_named the docstring claims.

        Pinned rather than left as an unverified promise: without this, the
        annotation could be narrowed to ast.Module tomorrow with every test
        still green.
        """
        tree = ast.parse(_SCOPED_IMPORTS_SNIPPET)
        inner, sibling = [n for n in tree.body if isinstance(n, ast.FunctionDef)]

        assert imported_names_from(inner, 'pkg.mod') == {'scoped'}
        assert imported_names_from(sibling, 'pkg.mod') == {'elsewhere'}
        assert imported_names_from(tree, 'pkg.mod') == {'scoped', 'elsewhere'}, (
            'the whole-module search should still see every import — the narrowing '
            'must come from the node passed in, not from the search itself.'
        )
