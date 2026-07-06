"""Migration guard: the 5 hand-rolled gather Pass-2 idiom sites must route
through the named async_utils helpers (gather_or_raise / gather_collect).

PRD plans/fm-cancellederror-convention-prd.md, task β. Before this task, three
source files hand-copied the two-tier "asyncio.gather(return_exceptions=True)
→ propagate_cancellations (Pass 1) → hand-rolled Pass-2" idiom at 5 call
sites. Task α (dep 2130) landed two named Pass-2 helpers in
fused_memory.utils.async_utils — gather_or_raise (log-all-then-raise-first)
and gather_collect (return per-item value-or-Exception) — so each call site
can pick its Pass-2 policy BY NAME instead of hand-rolling it.

This guard encodes the task's acceptance criterion: after migration, none of
the 3 touched files may call propagate_cancellations directly (it remains
exported for the rare bespoke site, but is otherwise an async_utils-internal
implementation detail of the two named helpers), none may hand-roll the old
gather(..., return_exceptions=...) two-tier idiom, and each must import at
least one of the two named helpers.

AST (not string grep) so a docstring/comment that still *describes* the old
idiom — e.g. "see async_utils.propagate_cancellations for the Pass 1
contract" — does not false-positive; only actual Call nodes count.

Scoped to EXACTLY the 3 files this task touches — NOT all of fused_memory/.
There are ~15 additional recon-stage swallow sites elsewhere in the codebase
that a separate task (task γ) owns and converts later; this guard must not
pre-empt that broader sweep.

Expected RED on current main (all 5 sites still hand-rolled); expected fully
GREEN once the last site (graphiti_client.detect_stale_dry_run, step-7)
migrates.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC_ROOT = pathlib.Path(__file__).parents[1] / 'src' / 'fused_memory'

ASYNC_UTILS_MODULE = 'fused_memory.utils.async_utils'
NAMED_HELPERS = {'gather_or_raise', 'gather_collect'}

TOUCHED_FILES = [
    SRC_ROOT / 'services' / 'memory_service.py',
    SRC_ROOT / 'backends' / 'graphiti_client.py',
    SRC_ROOT / 'reconciliation' / 'context_assembler.py',
]


def _parse(path: pathlib.Path) -> ast.Module:
    assert path.exists(), f'{path} not found'
    return ast.parse(path.read_text(), filename=str(path))


def _calls_named(tree: ast.Module, name: str) -> list[ast.Call]:
    """Every ast.Call node whose callee is a bare `name(...)` or an attribute
    access ending in `.name(...)` (e.g. `asyncio.gather(...)`)."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Name) and func.id == name) or (
            isinstance(func, ast.Attribute) and func.attr == name
        ):
            found.append(node)
    return found


def _gather_calls_with_return_exceptions(tree: ast.Module) -> list[ast.Call]:
    """Every gather(...) call (bare or asyncio.gather) carrying a
    `return_exceptions` keyword — the hand-rolled two-tier idiom's preamble."""
    return [
        node
        for node in _calls_named(tree, 'gather')
        if any(kw.arg == 'return_exceptions' for kw in node.keywords)
    ]


def _names_imported_from(tree: ast.Module, module: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize('path', TOUCHED_FILES, ids=lambda p: p.name)
class TestGatherIdiomMigratedToNamedHelpers:
    """Zero hand-rolled Pass-2 loops remain at the 5 sites; direct
    propagate_cancellations calls remain only inside async_utils itself."""

    def test_no_direct_propagate_cancellations_calls(self, path):
        tree = _parse(path)
        calls = _calls_named(tree, 'propagate_cancellations')
        assert not calls, (
            f'{path}: found {len(calls)} direct propagate_cancellations() call(s) '
            f'at line(s) {[c.lineno for c in calls]}. propagate_cancellations is Pass 1 '
            f'of the hand-rolled two-tier idiom being migrated away — this call site must '
            f'route through a named Pass-2 helper (gather_or_raise / gather_collect) from '
            f'{ASYNC_UTILS_MODULE} instead. Direct propagate_cancellations() calls may '
            f'only remain inside async_utils itself.'
        )

    def test_no_raw_gather_return_exceptions_calls(self, path):
        tree = _parse(path)
        calls = _gather_calls_with_return_exceptions(tree)
        assert not calls, (
            f'{path}: found {len(calls)} raw gather(..., return_exceptions=...) call(s) '
            f'at line(s) {[c.lineno for c in calls]}. This is the preamble of the '
            f'hand-rolled two-tier idiom being migrated — replace it with a call to '
            f'gather_or_raise / gather_collect ({ASYNC_UTILS_MODULE}).'
        )

    def test_imports_a_named_helper(self, path):
        tree = _parse(path)
        imported = _names_imported_from(tree, ASYNC_UTILS_MODULE)
        assert imported & NAMED_HELPERS, (
            f'{path}: imports {imported or "nothing"} from {ASYNC_UTILS_MODULE}, but '
            f'expected at least one of {sorted(NAMED_HELPERS)} to be in use after '
            f'migrating this file\'s gather Pass-2 idiom onto a named helper.'
        )
