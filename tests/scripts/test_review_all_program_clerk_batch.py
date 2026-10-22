"""Behavioural tests for ``skills/review-all/scripts/program/clerk_batch.py``.

``impact_scan`` is driven through the module loaded by path. ``write`` and
``ingest`` run as ``-I -S`` subprocesses from a temp cwd outside the repo, over
homes seeded from copies of the committed fixture, which is never edited.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from review_all_program_fixture import FIXTURE_DIR, PROGRAM_DIR, load_state

state = load_state()

CLERK_BATCH_PATH = PROGRAM_DIR / 'clerk_batch.py'
CLERK_BATCH_MODULE_NAME = 'review_all_program_clerk_batch'


def _load_clerk_batch() -> types.ModuleType:
    """Load ``clerk_batch.py`` by path, source-authoritative, as ``load_state`` loads state.py."""
    assert CLERK_BATCH_PATH.is_file(), f'module under test is missing: {CLERK_BATCH_PATH}'
    Path(importlib.util.cache_from_source(str(CLERK_BATCH_PATH))).unlink(missing_ok=True)
    spec = importlib.util.spec_from_file_location(CLERK_BATCH_MODULE_NAME, CLERK_BATCH_PATH)
    assert spec is not None and spec.loader is not None, f'no spec for {CLERK_BATCH_PATH}'
    module = importlib.util.module_from_spec(spec)
    sys.modules[CLERK_BATCH_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


clerk = _load_clerk_batch()

OLD = 'src/fixture/old_name.py'
NEW = 'src/fixture/new_name.py'
REMOVED = 'src/fixture/removed.py'


def _row(task_id: int, checks: Any, status: str = 'pending') -> Any:
    return state.TaskRow(
        task_id, f'task {task_id}', status, 'medium', '2026-10-19T00:00:00Z',
        metadata={'delivered_checks': checks}, dependencies=(),
    )


def _move(frm: str, to: str | None, *symbols: str) -> Any:
    return state.Move(from_path=frm, to_path=to, symbols=symbols)


def _check(
    name: str, *, kind: str = 'path', paths: Sequence[str] | None = (), pattern: str | None = None,
) -> dict[str, Any]:
    return {
        'name': name, 'kind': kind, 'paths': None if paths is None else list(paths),
        'pattern': pattern, 'expect': 'present',
    }


def _fixture_checks(task_id: int) -> list[Any]:
    tasks = json.loads((FIXTURE_DIR / 'store.json').read_text())['tasks']
    return next(t for t in tasks if t['id'] == task_id)['metadata']['delivered_checks']


def _impacts(moves: Sequence[Any], rows: Sequence[Any]) -> list[tuple[Any, ...]]:
    found = clerk.impact_scan(moves, rows)
    assert isinstance(found, tuple)
    return [(i.task, i.check, i.kind, i.from_path, i.to_path) for i in found]


def test_a_check_naming_the_moved_path_is_one_impact() -> None:
    row = _row(203, [_check('old-thing-defined', paths=[OLD])])
    assert _impacts([_move(OLD, NEW)], [row]) == [(203, 'old-thing-defined', 'path', OLD, NEW)]


def test_a_pattern_containing_a_moved_symbol_is_one_impact() -> None:
    check = _check('old-thing-defined', kind='grep', paths=['src/other.py'],
                   pattern=r'^class OldThing\b')
    impacts = _impacts([_move(OLD, NEW, 'OldThing')], [_row(203, [check])])
    assert impacts == [(203, 'old-thing-defined', 'grep', OLD, NEW)]


def test_the_fixture_check_matching_by_path_and_by_symbol_is_one_impact() -> None:
    impacts = _impacts([_move(OLD, NEW, 'OldThing')], [_row(203, _fixture_checks(203))])
    assert impacts == [(203, 'old-thing-defined', 'grep', OLD, NEW)]


def test_a_deleted_path_is_an_impact_with_no_new_path() -> None:
    row = _row(204, _fixture_checks(204), status='in-progress')
    assert _impacts([_move(REMOVED, None)], [row]) == [
        (204, 'removed-module-present', 'path', REMOVED, None),
    ]


@pytest.mark.parametrize(('check', 'move'), [
    (_check('unrelated-path', paths=['src/other.py']), _move(OLD, NEW)),
    (_check('no-pattern', paths=['src/other.py']), _move(OLD, NEW, 'OldThing')),
    (_check('no-paths', kind='script', paths=None), _move(OLD, NEW)),
    (_check('unrelated-pattern', kind='grep', pattern=r'^def unrelated\('),
     _move(OLD, NEW, 'OldThing')),
    (_check('empty-symbol', kind='grep', pattern=r'^def seam_contract\('), _move(OLD, NEW, '')),
], ids=['unrelated-paths', 'pattern-none', 'paths-none', 'unrelated-pattern', 'empty-symbol'])
def test_a_check_that_names_neither_the_path_nor_a_symbol_is_no_impact(
    check: dict[str, Any], move: Any,
) -> None:
    assert _impacts([move], [_row(203, [check])]) == []


def test_kind_is_passed_through_and_never_filtered() -> None:
    substrate = _check('substrate-check', kind='substrate', paths=[OLD])
    kindless = {'name': 'kindless-check', 'paths': [OLD], 'pattern': None}
    assert _impacts([_move(OLD, NEW)], [_row(203, [substrate, kindless])]) == [
        (203, 'substrate-check', 'substrate', OLD, NEW),
        (203, 'kindless-check', None, OLD, NEW),
    ]


def test_each_matching_check_of_one_task_is_its_own_impact() -> None:
    checks = [
        _check('first', paths=[OLD]),
        {'kind': 'grep', 'paths': [], 'pattern': 'OldThing()'},
        {'name': '', 'kind': 'path', 'paths': [OLD]},
    ]
    assert _impacts([_move(OLD, NEW, 'OldThing')], [_row(203, checks)]) == [
        (203, 'first', 'path', OLD, NEW),
        (203, 'delivered_checks[1]', 'grep', OLD, NEW),
        (203, 'delivered_checks[2]', 'path', OLD, NEW),
    ]


def test_impacts_are_in_move_order_then_task_id_then_check_index() -> None:
    rows = [
        _row(209, [_check('d0', paths=[REMOVED, OLD])]),
        _row(203, [_check('c0', paths=[OLD]), _check('c1', paths=[OLD, REMOVED])]),
    ]
    assert _impacts([_move(REMOVED, None), _move(OLD, NEW)], rows) == [
        (203, 'c1', 'path', REMOVED, None),
        (209, 'd0', 'path', REMOVED, None),
        (203, 'c0', 'path', OLD, NEW),
        (203, 'c1', 'path', OLD, NEW),
        (209, 'd0', 'path', OLD, NEW),
    ]


def test_a_check_entry_that_is_not_an_object_is_skipped_not_raised() -> None:
    rows = [
        _row(203, ['old-thing-defined', None, _check('kept', paths=[OLD])]),
        _row(204, None),
    ]
    assert _impacts([_move(OLD, NEW)], rows) == [(203, 'kept', 'path', OLD, NEW)]
