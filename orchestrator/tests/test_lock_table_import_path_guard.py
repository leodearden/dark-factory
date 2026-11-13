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
from pathlib import Path

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

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


def _old_path_uses(source: str, package: tuple[str, ...] = ()) -> list[tuple[int, str]]:
    """``(lineno, name)`` for each moved name *source* reaches through ``orchestrator.scheduler``."""
    if not any(name in source for name in _HOME_OF):
        return []
    uses: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and _resolved_module(node, package) == _OLD_HOME:
            uses.extend((node.lineno, alias.name) for alias in node.names if alias.name in _HOME_OF)
        elif isinstance(node, ast.Attribute) and node.attr in _HOME_OF and ast.unparse(node.value) == _OLD_HOME:
            uses.append((node.lineno, node.attr))
    return uses


def _swept_files() -> list[tuple[Path, tuple[str, ...]]]:
    src = [(path, path.relative_to(_SRC_DIR).parent.parts) for path in sorted(_SRC_DIR.rglob('*.py'))]
    tests = [(path, ()) for path in sorted(_TESTS_DIR.rglob('*.py'))]
    return src + tests


def test_no_moved_name_is_reached_through_the_scheduler() -> None:
    offenders = [
        f'{path.relative_to(_ORCH_ROOT)}:{lineno}: {name} -> import it from {_HOME_OF[name]}'
        for path, package in _swept_files()
        for lineno, name in _old_path_uses(path.read_text(encoding='utf-8'), package)
    ]
    assert not offenders, 'old-path imports of moved names:\n  ' + '\n  '.join(offenders)


def test_the_sweep_covers_the_scheduler_and_the_lock_table_tests() -> None:
    swept = {path.relative_to(_ORCH_ROOT).as_posix() for path, _ in _swept_files()}
    assert {'src/orchestrator/scheduler.py', 'tests/test_lock_table.py'} <= swept


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
