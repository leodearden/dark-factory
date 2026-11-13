"""Import scans for the fused-memory leaf-module guards.

A leaf guard asserts that a module imports only the stdlib and a declared set
of first-party modules. Imported by name (``from _leaf_import_guard import
imported_modules``), like ``_ast_guard``, whose ``parse_python_module`` supplies
the tree.
"""

import ast
import sys
from collections.abc import Set


def imported_modules(tree: ast.AST) -> set[str]:
    """Every module an import under *tree* names.

    A relative import is reported with its leading dots (``from . import x`` as
    ``.x``, ``from ..m import y`` as ``..m``), which no stdlib name and no
    absolute allowed name matches, so a relative sibling import fails a guard.
    """
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            dots = '.' * node.level
            if node.module is not None:
                modules.add(dots + node.module)
            else:
                modules.update(dots + alias.name for alias in node.names)
    return modules


def stdlib_or_allowed(module: str, allowed: Set[str]) -> bool:
    top_level = module.partition('.')[0]
    return top_level in sys.stdlib_module_names or top_level == '__future__' or module in allowed
