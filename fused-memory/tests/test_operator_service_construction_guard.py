"""No operator script constructs a MemoryService itself (C3 / INV-9).

Every operator entry point under ``fused-memory/scripts/`` and the repo-root
``scripts/`` builds its service through
``fused_memory.maintenance._utils.maintenance_service``, so construction,
opening and close are one policy.  The guard runs on every verify because the
delivered checks run once, and no pyright gate covers the root ``scripts/``.

AST, not grep: prose naming the class, an import of it and an
``isinstance(x, MemoryService)`` are not construction.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
from _ast_guard import calls_named, parse_python_module
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OPERATOR_SCRIPT_ROOTS = (REPO_ROOT / 'fused-memory' / 'scripts', REPO_ROOT / 'scripts')
CLASS_NAME = 'MemoryService'
RULE = 'construct the service through fused_memory.maintenance._utils.maintenance_service'


def _names_the_class(node: ast.expr | None) -> bool:
    return (isinstance(node, ast.Name) and node.id == CLASS_NAME) or (
        isinstance(node, ast.Attribute) and node.attr == CLASS_NAME
    )


def construction_sites(tree: ast.Module) -> list[int]:
    """Lines that call the class, or take it as a parameter default."""
    lines = [call.lineno for call in calls_named(tree, CLASS_NAME)]
    for arguments in ast.walk(tree):
        if not isinstance(arguments, ast.arguments):
            continue
        defaults = [*arguments.defaults, *arguments.kw_defaults]
        lines.extend(default.lineno for default in defaults if _names_the_class(default))
    return sorted(lines)


def operator_scripts() -> list[pathlib.Path]:
    return sorted(
        path
        for root in OPERATOR_SCRIPT_ROOTS
        for path in root.rglob('*.py')
        if '__pycache__' not in path.parts
    )


def test_no_operator_script_constructs_a_memory_service():
    offenders = [
        f'{path.relative_to(REPO_ROOT)}:{line}'
        for path in operator_scripts()
        for line in construction_sites(parse_python_module(path))
    ]

    assert offenders == [], f'{RULE}; found {len(offenders)}:\n' + '\n'.join(offenders)


def test_the_sweep_reaches_both_script_roots():
    scanned = {path.relative_to(REPO_ROOT).parts[0] for path in operator_scripts()}

    assert scanned == {'fused-memory', 'scripts'}


_FLAGGED = """
def build(c):
    m = MemoryService(c)
    return m

def build_dotted(c):
    return mod.MemoryService(c)

def factory(service=MemoryService):
    return service

def keyword_factory(*, service=memory_service.MemoryService):
    return service
"""

_IGNORED = '''
"""A docstring naming MemoryService(config)."""

from fused_memory.services.memory_service import MemoryService

# MemoryService(config) in a comment


def check(x):
    return isinstance(x, MemoryService)


def annotated(service: MemoryService | None = None):
    return service
'''


def _sites_in(tmp_path: pathlib.Path, source: str) -> list[int]:
    path = tmp_path / 'script.py'
    path.write_text(source)
    return construction_sites(parse_python_module(path))


def test_the_classifier_flags_calls_and_parameter_defaults(tmp_path):
    assert _sites_in(tmp_path, _FLAGGED) == [3, 7, 9, 12]


def test_the_classifier_ignores_prose_imports_isinstance_and_annotations(tmp_path):
    assert _sites_in(tmp_path, _IGNORED) == []
