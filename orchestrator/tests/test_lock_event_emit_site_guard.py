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
from pathlib import Path

import pytest
from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT

# Ceiling rationale: _orch_helpers.py::WHOLE_TREE_SCAN_TEST_TIMEOUT.
pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_SRC_DIR = Path(__file__).parent.parent / 'src'
_CHOKEPOINT = '_emit_lock_event'
_GUARDED_EVENTS = ('lock_acquired', 'lock_released')


def _enclosing_function_ranges(tree: ast.Module) -> list[tuple[str, int, int]]:
    """``(name, lineno, end_lineno)`` for every function/method in *tree*."""
    ranges: list[tuple[str, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ranges.append((node.name, node.lineno, node.end_lineno or node.lineno))
    return ranges


def _emit_sites(path: Path, event_name: str) -> list[tuple[str, int, str]]:
    """``(file, line, enclosing function)`` for each ``…emit(EventType.<event>…)``."""
    tree = ast.parse(path.read_text())
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
                and arg.attr == event_name
                and isinstance(arg.value, ast.Name)
                and arg.value.id == 'EventType'
            ):
                enclosing = [
                    name for name, lo, hi in ranges if lo <= node.lineno <= hi
                ]
                sites.append((path.name, node.lineno, enclosing[-1] if enclosing else '<module>'))
    return sites


@pytest.mark.parametrize('event_name', _GUARDED_EVENTS)
def test_lock_events_are_emitted_from_exactly_one_site(event_name: str):
    """One emit site per lock event, and it is ``_emit_lock_event``.

    A second site would emit an event the predictor never sees — and the gap
    is invisible: the event stream stays complete, so nothing looks broken
    while the in-process history silently drifts from what the durable seed
    would reconstruct.  Scoped to ``orchestrator/src`` deliberately; the tests
    tree legitimately constructs these events in fixtures.
    """
    sites = [
        site
        for path in sorted(_SRC_DIR.rglob('*.py'))
        for site in _emit_sites(path, event_name)
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
    ``orchestrator/tests/`` — where the guard's own ``rglob`` would then find it.
    """
    source = 'from x import EventType\ndef f(self):\n    self.event_store.emit(\n' \
             '        EventType.lock_acquired, task_id="T1", data={})\n'
    probe = tmp_path / '_guard_meta_probe.py'
    probe.write_text(source)

    found = _emit_sites(probe, 'lock_acquired')

    assert [(line, fn) for _f, line, fn in found] == [(3, 'f')]
