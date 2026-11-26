"""Guard: each lock event is emitted from exactly one site, ``_emit_lock_event``.

A single-writer guard (task 3822, INV-5), modelled on
test_lock_release_single_writer_guard.py and split out of
test_scheduler_hold_history.py so its whole-tree sweep no longer widens that
module's behavioural tests to the guard-family ceiling.  A behavioural test can
only prove today's emit sites feed the hold-history predictor; this proves a
new one cannot be added that doesn't.

AST, not text grep: an ``emit(`` call can span lines, and the enclosing
function is exactly what the invariant is about.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
from pathlib import Path

import pytest
from _tree_scan import (
    ORCHESTRATOR_SRC_DIR,
    TREE_PASS_XDIST_GROUP,
    ParsedModule,
    TreeAnalysis,
    TreePass,
    python_files,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

# The family mark and shared group: shared/src/shared/whole_tree_guard_family.py::WHOLE_TREE_SCAN_TEST_TIMEOUT, _tree_scan.py.
pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP),
]

_SRC_DIR = ORCHESTRATOR_SRC_DIR.parent
_CHOKEPOINT = '_emit_lock_event'
_GUARDED_EVENTS = ('lock_acquired', 'lock_released')
_EVENT_ENUM = 'EventType'


def _enclosing_function_ranges(tree: ast.Module) -> list[tuple[str, int, int]]:
    """``(name, lineno, end_lineno)`` for every function/method in *tree*."""
    ranges: list[tuple[str, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ranges.append((node.name, node.lineno, node.end_lineno or node.lineno))
    return ranges


def _emit_sites_in(tree: ast.Module) -> tuple[tuple[str, int, str], ...]:
    """``(event, line, enclosing function)`` for each ``…emit(EventType.<event>…)`` in *tree*."""
    ranges = _enclosing_function_ranges(tree)
    sites: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == 'emit'):
            continue
        args = list(node.args) + [kw.value for kw in node.keywords if kw.arg == 'event_type']
        for arg in args:
            if (
                isinstance(arg, ast.Attribute)
                and isinstance(arg.value, ast.Name)
                and arg.value.id == _EVENT_ENUM
            ):
                enclosing = [
                    name for name, lo, hi in ranges if lo <= node.lineno <= hi
                ]
                sites.append((arg.attr, node.lineno, enclosing[-1] if enclosing else '<module>'))
    return tuple(sites)


def _emit_sites(module: ParsedModule) -> tuple[tuple[str, int, str], ...]:
    """:func:`_emit_sites_in`, loud on a parse error; a module never naming the enum has no site."""
    tree = module.tree
    if not module.may_name(_EVENT_ENUM):
        return ()
    return _emit_sites_in(tree)


ANALYSIS = TreeAnalysis(
    name=Path(__file__).stem, files=functools.partial(python_files, _SRC_DIR), extract=_emit_sites,
)
TREE_ANALYSES = (ANALYSIS,)


@pytest.mark.parametrize('event_name', _GUARDED_EVENTS)
def test_lock_events_are_emitted_from_exactly_one_site(event_name: str, tree_pass: TreePass):
    """One emit site per lock event, and it is ``_emit_lock_event``.

    A second site would emit an event the predictor never sees — and the gap
    is invisible: the event stream stays complete, so nothing looks broken
    while the in-process history silently drifts from what the durable seed
    would reconstruct.  Scoped to ``orchestrator/src`` deliberately; the tests
    tree legitimately constructs these events in fixtures.
    """
    sites = [
        (path.name, line, enclosing)
        for path, emits in tree_pass.facts(ANALYSIS).items()
        for event, line, enclosing in emits
        if event == event_name
    ]

    assert len(sites) == 1, f'expected exactly one EventType.{event_name} emit site, got {sites}'
    assert sites[0][2] == _CHOKEPOINT, (
        f'EventType.{event_name} emitted from {sites[0][2]}, not {_CHOKEPOINT}: '
        f'a lock event emitted outside the chokepoint never reaches the predictor'
    )


def test_the_guard_can_actually_find_emit_sites(tmp_path):
    """Meta-test: a scanner that finds nothing would pass the guard vacuously.

    The probe goes in ``tmp_path``, never in the tests tree: this suite runs
    under pytest-xdist, so a fixed path shared by every worker is a race, and a
    hard interrupt mid-test would strand an untracked .py file in
    ``orchestrator/tests/`` — where the tests-tree guards would then find it.
    """
    source = 'from x import EventType\ndef f(self):\n    self.event_store.emit(\n' \
             '        EventType.lock_acquired, task_id="T1", data={})\n'
    (tmp_path / '_guard_meta_probe.py').write_text(source)
    probe = dataclasses.replace(ANALYSIS, files=functools.partial(python_files, tmp_path))

    found = TreePass.run((probe,)).facts(probe)

    assert list(found.values()) == [(('lock_acquired', 3, 'f'),)]
