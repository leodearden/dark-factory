"""No operator script constructs a MemoryService itself (C3 / INV-9).

Every operator entry point under ``fused-memory/scripts/`` and the repo-root
``scripts/`` builds its service through
``fused_memory.maintenance._utils.maintenance_service``, so construction,
opening and close are one policy.  The guard runs on every verify because the
delivered checks run once, and no pyright gate covers the root ``scripts/``.

AST, not grep. Any reference to the class as a value can build a service, so
each one is flagged: a call, a parameter default, an alias
(``cls = MemoryService``), an argument (``functools.partial(MemoryService, c)``,
``open_memory(c, service=MemoryService)``). Prose, an import, an annotation and
the type argument of ``isinstance``/``issubclass`` name the class without being
able to build one, so they are not flagged.
"""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Iterator
from typing import TypeGuard

import pytest
from _ast_guard import parse_python_module
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OPERATOR_SCRIPT_ROOTS = (REPO_ROOT / 'fused-memory' / 'scripts', REPO_ROOT / 'scripts')
CLASS_NAME = 'MemoryService'
RULE = 'construct the service through fused_memory.maintenance._utils.maintenance_service'
_TYPE_TESTS = frozenset({'isinstance', 'issubclass'})


def _loads_the_class(node: ast.AST) -> TypeGuard[ast.Name | ast.Attribute]:
    if isinstance(node, ast.Name):
        return node.id == CLASS_NAME and isinstance(node.ctx, ast.Load)
    if isinstance(node, ast.Attribute):
        return node.attr == CLASS_NAME and isinstance(node.ctx, ast.Load)
    return False


def _naming_only_subtrees(tree: ast.Module) -> Iterator[ast.expr]:
    """Annotations and type-test arguments: where the class is named, never used."""
    for node in ast.walk(tree):
        if isinstance(node, ast.arg | ast.AnnAssign) and node.annotation is not None:
            yield node.annotation
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.returns is not None:
            yield node.returns
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _TYPE_TESTS
        ):
            yield from node.args[1:]


def construction_sites(tree: ast.Module) -> list[int]:
    """Lines that reference the class as a value."""
    naming_only = {
        id(node) for subtree in _naming_only_subtrees(tree) for node in ast.walk(subtree)
    }
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if _loads_the_class(node) and id(node) not in naming_only
    )


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

def aliased(c):
    cls = MemoryService
    return cls(c)

def partial_factory(c):
    return functools.partial(memory_service.MemoryService, c)

def keyword_argument(c):
    return open_memory(c, service=MemoryService)

def alternate_constructor(c):
    return MemoryService.from_config(c)
"""

_IGNORED = '''
"""A docstring naming MemoryService(config)."""

from fused_memory.services.memory_service import MemoryService

# MemoryService(config) in a comment


def check(x):
    return isinstance(x, MemoryService)


def annotated(service: MemoryService | None = None) -> MemoryService | None:
    held: type[MemoryService] | None = None
    return service or held


def type_tests(x, cls):
    return isinstance(x, (MemoryService, int)) or issubclass(cls, memory_service.MemoryService)
'''


def _sites_in(tmp_path: pathlib.Path, source: str) -> list[int]:
    path = tmp_path / 'script.py'
    path.write_text(source)
    return construction_sites(parse_python_module(path))


def test_the_classifier_flags_every_reference_to_the_class_as_a_value(tmp_path):
    assert _sites_in(tmp_path, _FLAGGED) == [3, 7, 9, 12, 16, 20, 23, 26]


def test_the_classifier_ignores_prose_imports_annotations_and_type_tests(tmp_path):
    assert _sites_in(tmp_path, _IGNORED) == []
