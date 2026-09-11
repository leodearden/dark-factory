"""Shared Python-AST machinery for the fused-memory migration guards.

The migration guards assert over PARSED source rather than raw text, so that
prose which merely mentions the idiom being migrated cannot satisfy or trip a
check.  The parse and node-search primitives all of them rest on live here, in
one module, so the guards agree on those semantics by construction rather than
by copy.

Lives outside conftest.py — for the same reason `_fm_helpers` does — to avoid
the `sys.modules['conftest']` collision that arises when root-level pytest
loads multiple subprojects' conftests in the same process.  Each subproject
exports its helpers under a unique module name so test files can
`from _ast_guard import X` without colliding with sibling subprojects'
helpers.
"""

import ast
import functools
import pathlib

# ---------------------------------------------------------------------------
# Shared AST migration-guard machinery (task 3502; consumers widened by 3574)
# ---------------------------------------------------------------------------
#
# Migration guards (tests/test_falkor_probe_routing_guard.py;
# tests/test_falkor_index_barrier_guard.py;
# tests/test_gather_idiom_helper_routing.py) assert over the source of a set of
# migrated modules. The target is not always a TEST module: the gather-idiom
# guard asserts over production source under src/fused_memory/, which is why
# the parse is named for Python modules generally rather than for test ones.
# Parsing is shared and memoised here so several guards asserting over
# overlapping module sets pay one parse per file per session rather than one
# each.
# ---------------------------------------------------------------------------


@functools.cache
def parse_python_module(path: pathlib.Path) -> ast.Module:
    """Parse *path* into an ``ast.Module``, memoised per session.

    Accepts any Python module — a test module or production source under
    ``src/fused_memory/``. Sources do not change mid-session, so several guards
    asserting over overlapping module sets can share one parse per file.

    The returned tree is SHARED between callers, so no consumer may mutate it.
    """
    assert path.exists(), f'{path} not found'
    # The read is deliberately not memoised separately: this function is its
    # only caller and is itself cached on the same key, so a source cache would
    # score zero hits while pinning every parsed file's full text for the
    # session on top of its tree. The tree is the one retained artefact.
    return ast.parse(path.read_text(), filename=str(path))


def calls_named(node: ast.AST, name: str) -> list[ast.Call]:
    """Every ``ast.Call`` under *node* whose callee is ``name(...)`` / ``….name(...)``.

    AST, not string grep: prose that merely mentions the name — a docstring
    describing the helper a guard enforces — must not satisfy or trip a check.

    Takes any node, not only a module, so a guard can ask the narrower question
    "is it called *here*" — inside a decorator, or inside the value assigned to
    ``pytestmark`` — rather than only "is it called anywhere in the file".
    """
    found: list[ast.Call] = []
    for descendant in ast.walk(node):
        if not isinstance(descendant, ast.Call):
            continue
        func = descendant.func
        if (isinstance(func, ast.Name) and func.id == name) or (
            isinstance(func, ast.Attribute) and func.attr == name
        ):
            found.append(descendant)
    return found


def imported_names_from(node: ast.AST, module: str) -> set[str]:
    """Every name a ``from <module> import …`` under *node* brings in.

    Reports the SOURCE-side name, so ``from m import x as y`` is reported as
    ``x``. Guards compare the result against the name as the helper module
    exports it, so an aliased import must not read as a missing one.

    Only ``from <module> import …`` counts: a plain ``import <module>`` binds
    the module and never a name from it, so it cannot satisfy a guard demanding
    a specific helper be imported. A star import reports the literal ``'*'``
    (that is its ``alias.name``), which likewise satisfies no guard asking for
    a NAMED helper. Returns an empty set — never None — when
    nothing is imported from *module*, which callers rely on both for set
    algebra and to render ``imported or 'nothing'`` in a failure message.

    Takes any node rather than only a module, for symmetry with
    :func:`calls_named`.
    """
    names: set[str] = set()
    for descendant in ast.walk(node):
        if isinstance(descendant, ast.ImportFrom) and descendant.module == module:
            names.update(alias.name for alias in descendant.names)
    return names
