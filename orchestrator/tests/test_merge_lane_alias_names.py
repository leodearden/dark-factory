"""Every name reached THROUGH a pre-package merge-lane alias module still exists.

PRD ``plans/merge-lane-quality-prd.md`` task zeta2. The fourteen old module
paths (``scripts/merge_lane_metrics.py::ALIAS_MODULES``) are aliases: each one
IS its ``orchestrator.merge_lane`` submodule. Pyright checks a name imported
through an alias only as far as the alias's typing allows, and the aliases that
external code reaches private names through keep an ``Any``-typed module
``__getattr__``, under which pyright accepts any name at all. So a stale
reference through one of those paths would pass every static gate and fail
only when the line runs.

This test closes that gap for the three ways a tracked file reaches a name
through an alias, each checked with ``hasattr`` against the live module:

* ``from <alias> import N``;
* ``m.N`` where ``m`` is bound to an alias module in that scope
  (``import <alias> as m``, ``from orchestrator import <leaf> as m``, or the
  bare ``orchestrator.<leaf>.N`` chain);
* a string constant ``'<alias>.N…'``, a ``patch`` or ``monkeypatch`` target,
  checked on its first segment after the alias.

The facade ``orchestrator.merge_lane`` has the same blind spot (its export
table is a module ``__getattr__``), so ``from orchestrator.merge_lane import N``
is checked too: ``N`` must be an export or a submodule.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import re
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType

import pytest
from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT

# Sweeps every tracked .py and ast-parses it, so it carries the whole-tree
# scan ceiling; WHY lives at _orch_helpers.py::WHOLE_TREE_SCAN_TEST_TIMEOUT.
pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

# Same import of the metrics script as test_merge_lane_ratchet.py: its
# ALIAS_MODULES is the single list of alias paths.
_SCRIPTS = Path(__file__).parents[2] / 'scripts'
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import merge_lane_metrics as metrics  # type: ignore[import-not-found]  # noqa: E402

_REPO_ROOT = Path(__file__).parents[2]

_FACADE = 'orchestrator.merge_lane'

#: Where a ``from <module> import N`` is checked: every alias, and the facade.
_FROM_IMPORT_CHECKED = frozenset({*metrics.ALIAS_MODULES, _FACADE})

_DOTTED_NAME = re.compile(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*')

#: Longest alias first, so ``orchestrator.merge_queue_store.X`` is never read
#: as ``orchestrator.merge_queue`` + ``_store``.
_ALIASES_LONGEST_FIRST = tuple(sorted(metrics.ALIAS_MODULES, key=len, reverse=True))


def _alias_and_first_segment(dotted: str) -> tuple[str, str] | None:
    for alias in _ALIASES_LONGEST_FIRST:
        prefix = alias + '.'
        if dotted.startswith(prefix):
            return alias, dotted[len(prefix):].split('.', 1)[0]
    return None


def _alias_bindings(statements: list[ast.stmt]) -> dict[str, str]:
    """Names bound to an alias module by import statements directly in *statements*."""
    bound: dict[str, str] = {}
    for node in statements:
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name in metrics.ALIAS_MODULES and imported.asname:
                    bound[imported.asname] = imported.name
        elif isinstance(node, ast.ImportFrom) and node.module == 'orchestrator' and not node.level:
            for imported in node.names:
                dotted = f'orchestrator.{imported.name}'
                if dotted in metrics.ALIAS_MODULES:
                    bound[imported.asname or imported.name] = dotted
    return bound


def _rebound_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Names a function binds to something else: its parameters and assignments."""
    arguments = function.args
    names = {
        argument.arg
        for argument in (
            *arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
            *(a for a in (arguments.vararg, arguments.kwarg) if a is not None),
        )
    }
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def _flat_imports(scope: ast.AST) -> list[ast.stmt]:
    """Import statements anywhere in *scope* that are not inside a nested function."""
    found: list[ast.stmt] = []
    pending = list(ast.iter_child_nodes(scope))
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Import | ast.ImportFrom):
            found.append(node)
        elif not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            pending.extend(ast.iter_child_nodes(node))
    return found


def _attribute_references(tree: ast.Module) -> Iterator[tuple[int, str, str]]:
    """``(lineno, alias, name)`` for every attribute load through an alias-bound name."""

    def visit(scope: ast.AST, inherited: dict[str, str]) -> Iterator[tuple[int, str, str]]:
        bound = dict(inherited)
        if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef):
            for name in _rebound_names(scope):
                bound.pop(name, None)
        bound.update(_alias_bindings(_flat_imports(scope)))
        pending = list(ast.iter_child_nodes(scope))
        while pending:
            node = pending.pop()
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                yield from visit(node, bound)
                continue
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                receiver = node.value
                if isinstance(receiver, ast.Name) and receiver.id in bound:
                    yield node.lineno, bound[receiver.id], node.attr
                elif (
                    isinstance(receiver, ast.Attribute)
                    and isinstance(receiver.value, ast.Name)
                    and receiver.value.id == 'orchestrator'
                    and f'orchestrator.{receiver.attr}' in metrics.ALIAS_MODULES
                ):
                    yield node.lineno, f'orchestrator.{receiver.attr}', node.attr
            pending.extend(ast.iter_child_nodes(node))

    yield from visit(tree, {})


def _is_submodule(module: str, name: str) -> bool:
    """``from <package> import <submodule>`` imports it even when no attribute exists yet."""
    return module == _FACADE and importlib.util.find_spec(f'{module}.{name}') is not None


def stale_alias_references(
    source: str, *, path: str, live: Mapping[str, ModuleType],
) -> list[str]:
    """Every name *source* reaches through an alias that the live alias module lacks."""
    tree = ast.parse(source, filename=path)
    stale: list[str] = []

    def check(lineno: int, kind: str, alias: str, name: str) -> None:
        if not hasattr(live[alias], name) and not _is_submodule(alias, name):
            stale.append(f'{path}:{lineno}: {kind} {alias}.{name}')

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module in _FROM_IMPORT_CHECKED
        ):
            for imported in node.names:
                if imported.name != '*':
                    check(node.lineno, 'import', node.module, imported.name)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _DOTTED_NAME.fullmatch(node.value)
        ):
            target = _alias_and_first_segment(node.value)
            if target is not None:
                check(node.lineno, 'string', *target)
    for lineno, alias, name in _attribute_references(tree):
        check(lineno, 'attribute', alias, name)
    return stale


@pytest.fixture(scope='module')
def live_aliases() -> dict[str, ModuleType]:
    return {module: importlib.import_module(module) for module in _FROM_IMPORT_CHECKED}


def test_no_tracked_file_reaches_a_missing_name_through_an_alias(
    live_aliases: dict[str, ModuleType],
) -> None:
    scanned = 0
    stale: list[str] = []
    for relpath in metrics.tracked_python_files(_REPO_ROOT):
        if not metrics.is_external_importer_source(relpath):
            continue
        source = (_REPO_ROOT / relpath).read_text(encoding='utf-8')
        if not any(alias.rsplit('.', 1)[-1] in source for alias in metrics.ALIAS_MODULES):
            continue
        scanned += 1
        stale.extend(stale_alias_references(source, path=relpath, live=live_aliases))
    assert scanned > 100, f'only {scanned} files mention an alias leaf; the sweep is not reading the tree'
    assert not stale, 'names reached through an alias module that no longer exist:\n' + '\n'.join(stale)


class TestStaleAliasReferences:
    """Seeded fixtures: each reference shape is flagged when stale and passes when live."""

    @pytest.mark.parametrize('source', [
        'from orchestrator.merge_queue import NoSuchName\n',
        'import orchestrator.merge_queue as mq\nmq.NoSuchName\n',
        'from orchestrator import merge_gates as g\ng.NoSuchName\n',
        'import orchestrator.merge_queue\norchestrator.merge_queue.NoSuchName\n',
        "patch('orchestrator.merge_queue.NoSuchName.attr', 1)\n",
        "monkeypatch.setattr('orchestrator.merge_queue_store.NoSuchName', 1)\n",
        'def f():\n    import orchestrator.merge_queue as mq\n    return mq.NoSuchName\n',
        'from orchestrator.merge_lane import NoSuchName\n',
    ])
    def test_a_stale_reference_is_flagged(
        self, source: str, live_aliases: dict[str, ModuleType],
    ) -> None:
        stale = stale_alias_references(source, path='fixture.py', live=live_aliases)
        assert len(stale) == 1 and 'NoSuchName' in stale[0], stale

    @pytest.mark.parametrize('source', [
        'from orchestrator.merge_queue import SpeculativeMergeWorker\n',
        'from orchestrator.merge_gates import _check_plan_targets_in_tree\n',
        'import orchestrator.merge_queue as mq\nmq.SpeculativeMergeWorker\n',
        "patch('orchestrator.merge_queue_store.MergeQueueStore.load', 1)\n",
        "patch('orchestrator.merge_lane.gates.NotAnAliasPath', 1)\n",
        'def f(mq):\n    return mq.put\nimport orchestrator.merge_queue as mq\n',
        'mq = object()\nmq.anything\n',
        'from orchestrator.merge_lane import MergeLane, WaiterRecord\n',
        'from orchestrator.merge_lane import landed_outbox\n',
    ])
    def test_a_live_or_unrelated_reference_is_not_flagged(
        self, source: str, live_aliases: dict[str, ModuleType],
    ) -> None:
        assert stale_alias_references(source, path='fixture.py', live=live_aliases) == []
