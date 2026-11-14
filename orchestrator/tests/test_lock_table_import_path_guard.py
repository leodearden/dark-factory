"""Guard: nothing under orchestrator/ imports a name the lock-table split moved
out of ``orchestrator.scheduler`` by its old path.

scheduler.py still imports each of these names to use them, so the old path
keeps resolving and no type checker objects to it. Dropping them from
``__all__`` (test_lock_table.py::TestModuleBoundary) only declares the
surface; this sweep is what stops an importer from re-coupling to the
scheduler for a lock-table, tie-break or clock name.
"""
from __future__ import annotations

import ast
import functools
from pathlib import Path

import pytest
from _tree_scan import TREE_PASS_XDIST_GROUP, ParsedModule, TreeAnalysis, TreePass, python_files
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

# The family mark and shared group: shared/src/shared/whole_tree_guard_family.py::WHOLE_TREE_SCAN_TEST_TIMEOUT, _tree_scan.py.
pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP),
]

_ORCH_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _ORCH_ROOT / 'src'
_TESTS_DIR = _ORCH_ROOT / 'tests'
_OLD_HOME = 'orchestrator.scheduler'
_HOME_OF = {
    'ModuleLockTable': 'orchestrator.lock_table',
    'id_tie_break': 'orchestrator.id_tie_break',
    'resolve_monotonic_time_source': 'orchestrator.time_sources',
    'resolve_wall_time_source': 'orchestrator.time_sources',
}


def _resolved_module(node: ast.ImportFrom, package: tuple[str, ...]) -> str:
    base = package[: len(package) - node.level + 1] if node.level else ()
    return '.'.join([*base, *([node.module] if node.module else [])])


def _old_path_uses_in(tree: ast.Module, package: tuple[str, ...]) -> tuple[tuple[int, str], ...]:
    uses: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and _resolved_module(node, package) == _OLD_HOME:
            uses.extend((node.lineno, alias.name) for alias in node.names if alias.name in _HOME_OF)
        elif isinstance(node, ast.Attribute) and node.attr in _HOME_OF and ast.unparse(node.value) == _OLD_HOME:
            uses.append((node.lineno, node.attr))
    return tuple(uses)


def _old_path_uses(source: str, package: tuple[str, ...] = ()) -> list[tuple[int, str]]:
    """``(lineno, name)`` for each moved name *source* reaches through ``orchestrator.scheduler``."""
    if not any(name in source for name in _HOME_OF):
        return []
    return list(_old_path_uses_in(ast.parse(source), package))


def _package_of(path: Path) -> tuple[str, ...]:
    return path.relative_to(_SRC_DIR).parent.parts if path.is_relative_to(_SRC_DIR) else ()


def _module_old_path_uses(module: ParsedModule) -> tuple[tuple[int, str], ...]:
    if not module.may_name(*_HOME_OF):
        return ()
    return _old_path_uses_in(module.tree, _package_of(module.path))


ANALYSIS = TreeAnalysis(
    name=Path(__file__).stem,
    files=functools.partial(python_files, _SRC_DIR, _TESTS_DIR),
    extract=_module_old_path_uses,
)
TREE_ANALYSES = (ANALYSIS,)


def test_no_moved_name_is_reached_through_the_scheduler(tree_pass: TreePass) -> None:
    facts = tree_pass.facts(ANALYSIS)
    swept = {path.relative_to(_ORCH_ROOT).as_posix() for path in facts}
    assert {'src/orchestrator/scheduler.py', 'tests/test_lock_table.py'} <= swept
    offenders = [
        f'{path.relative_to(_ORCH_ROOT)}:{lineno}: {name} -> import it from {_HOME_OF[name]}'
        for path, uses in facts.items()
        for lineno, name in uses
    ]
    assert not offenders, 'old-path imports of moved names:\n  ' + '\n  '.join(offenders)


def test_a_source_file_resolves_relative_imports_against_its_package() -> None:
    assert _package_of(_SRC_DIR / 'orchestrator' / 'evals' / 'x.py') == ('orchestrator', 'evals')
    assert _package_of(_TESTS_DIR / 'test_x.py') == ()


@pytest.mark.parametrize(
    'source',
    [
        'from orchestrator.scheduler import ModuleLockTable\n',
        'from orchestrator.scheduler import (\n    Scheduler,\n    ModuleLockTable as MLT,\n)\n',
        'import orchestrator.scheduler\nlt = orchestrator.scheduler.ModuleLockTable(cfg)\n',
        'from orchestrator.scheduler import id_tie_break\n',
    ],
)
def test_detector_flags_an_old_path_use(source: str) -> None:
    assert _old_path_uses(source)


def test_detector_resolves_a_relative_import_against_its_package() -> None:
    source = 'from ..scheduler import resolve_wall_time_source\n'
    assert _old_path_uses(source, ('orchestrator', 'evals')) == [(1, 'resolve_wall_time_source')]


@pytest.mark.parametrize(
    'source',
    [
        'from orchestrator.lock_table import ModuleLockTable\n',
        'from orchestrator.scheduler import Scheduler\n',
        "patch('orchestrator.scheduler.ModuleLockTable')\n",
        '"""from orchestrator.scheduler import ModuleLockTable"""\n',
    ],
)
def test_detector_ignores_a_new_path_or_non_import_mention(source: str) -> None:
    assert _old_path_uses(source) == []
