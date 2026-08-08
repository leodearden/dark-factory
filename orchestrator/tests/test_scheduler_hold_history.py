"""Scheduler wiring for the module hold-history predictor.

Task 3822 / plans/scheduler-dispatch-scoring-and-lock-layer-prd.md task ζ
(D11, INV-5).

Two things are asserted here, and they are different in kind:

1. BEHAVIOUR — the Scheduler builds a ``HoldHistory`` unconditionally, and
   ``finish_startup()`` seeds it from the event store when one is attached.
   The harness attaches the store AFTER construction (harness.py:2121-2122)
   and calls ``finish_startup()`` later (harness.py:2445), so a seed in
   ``__init__`` would silently read ``None`` forever.

2. STRUCTURE — a single-writer guard, modelled on
   test_lock_release_single_writer_guard.py.  Both lock events must be
   emitted from exactly one call site each, inside ``_emit_lock_event``, so
   a future emit site cannot record a lock event without also feeding the
   predictor.  A behavioural test can only prove today's four sites feed it;
   this proves a fifth cannot be added that doesn't.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import _hold_history_fixtures as F
import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.hold_history import HoldHistory
from orchestrator.scheduler import Scheduler

FIXED_DT = datetime(2026, 8, 1, 0, 0, 0, tzinfo=UTC)


class _StubEventStore:
    """Fetch-only stand-in returning the canonical fixture trace by type.

    ``_RecordingEventStore`` implements ``emit()`` and no fetch API, so it
    cannot exercise the seed; this is its mirror image — enough fetch surface
    for ``seed_from_event_store``, plus an ``emit`` recorder so the same object
    can also witness the four lock-event sites.
    """

    def __init__(self, rows=None):
        self._rows = list(rows if rows is not None else F.build_trace())
        self.emitted: list[tuple] = []

    def fetch_events_by_type_all_runs(self, event_type, *, task_id=None):
        wanted = getattr(event_type, 'value', event_type)
        return [r for r in self._rows if r['event_type'] == wanted]

    def emit(self, event_type, *, task_id=None, data=None, **kwargs):
        self.emitted.append((getattr(event_type, 'value', event_type), task_id, data))


def _make_scheduler(**config_overrides) -> Scheduler:
    config = OrchestratorConfig(max_per_module=1, **config_overrides)
    return Scheduler(config, wall_time_source=lambda: FIXED_DT)


# --- (a) construction and the seed point -----------------------------------


def test_scheduler_builds_a_hold_history_without_an_event_store():
    """The harness constructs the Scheduler with event_store=None
    (harness.py:1498-1514) and injects the store later.  The predictor must
    exist from the start regardless, so the live feed has somewhere to go."""
    scheduler = _make_scheduler()

    assert isinstance(scheduler._hold_history, HoldHistory)


def test_finish_startup_without_an_event_store_is_a_no_op():
    scheduler = _make_scheduler()

    scheduler.finish_startup()

    assert scheduler._hold_history.predicted_hold(['orchestrator/src']) is None
    assert scheduler.started is True


def test_finish_startup_seeds_from_an_event_store_attached_after_construction():
    """The harness order: construct (no store) -> attach store -> finish_startup.

    Seeding in ``__init__`` would read ``self.event_store is None`` and leave
    the predictor permanently empty in production while every
    store-in-constructor unit test looked fine.
    """
    scheduler = _make_scheduler()
    assert scheduler._hold_history.predicted_hold(['orchestrator/src']) is None

    scheduler.event_store = _StubEventStore()  # harness.py:2121-2122
    scheduler.finish_startup()                 # harness.py:2445

    for module, expected in F.EXPECTED_MEDIANS.items():
        assert scheduler._hold_history.predicted_hold([module]) == pytest.approx(
            expected
        ), module


def test_finish_startup_is_idempotent_for_the_seed_too():
    """``finish_startup`` documents itself as idempotent; the seed must be too."""
    scheduler = _make_scheduler()
    scheduler.event_store = _StubEventStore()

    scheduler.finish_startup()
    scheduler.finish_startup()

    assert scheduler._hold_history.sample_count(
        list(F.EXPECTED_SAMPLES)
    ) == F.EXPECTED_SPAN_COUNT


# --- (b) the single-writer structural guard --------------------------------
#
# AST, not text grep: an ``emit(`` call can span lines, and the enclosing
# function is exactly what the invariant is about.

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


def test_the_guard_can_actually_find_emit_sites():
    """Meta-test: a scanner that finds nothing would pass the guard vacuously."""
    source = 'from x import EventType\ndef f(self):\n    self.event_store.emit(\n' \
             '        EventType.lock_acquired, task_id="T1", data={})\n'
    tmp = Path(__file__).parent / '_guard_meta_probe.py'
    tmp.write_text(source)
    try:
        found = _emit_sites(tmp, 'lock_acquired')
    finally:
        tmp.unlink()

    assert [(line, fn) for _f, line, fn in found] == [(3, 'f')]
