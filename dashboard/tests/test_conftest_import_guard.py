"""AST-based guard: sys.path.insert must precede all non-stdlib imports in conftest.py.

This test enforces the structural invariant that the sys.path manipulation
block in conftest.py appears BEFORE any third-party imports.  Without this
ordering, a maintainer who adds `from dashboard import ...` above the
sys.path block would silently resolve against the installed editable package
rather than the local worktree src/, causing hard-to-diagnose test failures.
"""

import ast
import sys
from pathlib import Path

_CONFTEST = Path(__file__).parent / "conftest.py"


def _find_syspath_insert_line(tree: ast.Module) -> int | None:
    """Return the line number of the first sys.path.insert() call, or None.

    Walks the entire AST (including nested bodies such as if-blocks) to locate
    a call of the form ``sys.path.insert(...)``.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "insert"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "path"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "sys"
            ):
                return node.lineno
    return None


def _first_non_stdlib_import_line(tree: ast.Module) -> int | None:
    """Return the line number of the first top-level non-stdlib import, or None.

    Iterates through ``tree.body`` in source order so that the first
    occurrence (by line number) is returned.  Only top-level import statements
    are examined; imports nested inside functions or classes are ignored.
    """
    stdlib = sys.stdlib_module_names
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level = alias.name.split(".")[0]
                if top_level not in stdlib:
                    return node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] not in stdlib:
            return node.lineno
    return None


def test_relative_import_detected_as_non_stdlib() -> None:
    """from . import helpers — level>0, module=None — must be returned as non-stdlib."""
    source = "import os\nfrom . import helpers\n"
    tree = ast.parse(source)
    assert _first_non_stdlib_import_line(tree) == 2


def test_dotted_relative_import_detected_as_non_stdlib() -> None:
    """from .utils import helper — level=1, module='utils' — must be returned as non-stdlib."""
    source = "import sys\nfrom .utils import helper\n"
    tree = ast.parse(source)
    assert _first_non_stdlib_import_line(tree) == 2


def test_double_dot_relative_import_detected() -> None:
    """from .. import config — level=2, module=None — must be returned as non-stdlib."""
    source = "from .. import config\n"
    tree = ast.parse(source)
    assert _first_non_stdlib_import_line(tree) == 1


def test_syspath_insert_precedes_non_stdlib_imports() -> None:
    """sys.path.insert() in conftest.py must appear before any non-stdlib import.

    The sys.path block must be positioned between the stdlib imports (which are
    safe to import before path manipulation) and the third-party imports.  This
    prevents silent wrong-package resolution if `from dashboard import ...` is
    ever added above the block.

    The test will FAIL if the sys.path.insert call appears on a line that is
    greater than or equal to the first non-stdlib import line.
    """
    source = _CONFTEST.read_text()
    tree = ast.parse(source, filename=str(_CONFTEST))

    syspath_line = _find_syspath_insert_line(tree)
    first_non_stdlib_line = _first_non_stdlib_import_line(tree)

    assert syspath_line is not None, (
        "No sys.path.insert() call found in conftest.py — the guard block may "
        "have been removed or renamed."
    )
    assert first_non_stdlib_line is not None, (
        "No non-stdlib imports found in conftest.py — test setup may be wrong "
        "or all imports are stdlib."
    )
    assert syspath_line < first_non_stdlib_line, (
        f"Ordering violation in conftest.py: sys.path.insert() is on line "
        f"{syspath_line} but the first non-stdlib import is on line "
        f"{first_non_stdlib_line}.  The sys.path block MUST appear before any "
        f"non-stdlib imports to prevent silent wrong-package resolution against "
        f"the installed editable package instead of the local worktree src/."
    )
