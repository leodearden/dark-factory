"""The scheduler payload states whether each lock's holder is still a live task.

``holder_live`` is how the Scheduler heatmap tells a DEPARTED holder (a lock no
live task can release, with a waiter queued behind it) from an ordinary held
module. It is three-state: ``None`` means unknown, never departed, so a caller
that supplies no live set cannot make every held module look stranded.
"""

from __future__ import annotations

from dashboard.data.scheduler import _merge_module_lists, _module_contention_counts


def _entry(modules: list[dict], path: str) -> dict:
    (match,) = [m for m in modules if m['path'] == path]
    return match


def test_holder_in_the_live_set_is_live():
    modules = _module_contention_counts(
        [{'task_id': '1', 'lock_set': ['m']}],
        {'m': '1'},
        project='p',
        live_task_ids={'1'},
    )
    assert _entry(modules, 'm')['holder_live'] is True


def test_holder_absent_from_the_live_set_has_departed_and_only_the_waiter_counts():
    modules = _module_contention_counts(
        [{'task_id': '2', 'lock_set': ['m']}],
        {'m': '9'},
        project='p',
        live_task_ids={'2'},
    )
    entry = _entry(modules, 'm')
    assert entry['holder'] == '9'
    assert entry['contention'] == 1
    assert entry['holder_live'] is False


def test_module_without_a_holder_has_no_holder_liveness():
    modules = _module_contention_counts(
        [{'task_id': '1', 'lock_set': ['m']}],
        {},
        project='p',
        live_task_ids={'1'},
    )
    assert _entry(modules, 'm')['holder_live'] is None


def test_unknown_liveness_never_reads_as_departed():
    modules = _module_contention_counts(
        [{'task_id': '1', 'lock_set': ['m']}],
        {'m': '9'},
        project='p',
    )
    assert _entry(modules, 'm')['holder_live'] is None


def test_an_explicitly_empty_live_set_is_known_and_departs_the_holder():
    modules = _module_contention_counts(
        [{'task_id': '1', 'lock_set': ['m']}],
        {'m': '9'},
        project='p',
        live_task_ids=set(),
    )
    assert _entry(modules, 'm')['holder_live'] is False


def _module(*, holder: str | None, holder_live: bool | None) -> dict:
    return {
        'project': 'p',
        'path': 'm',
        'contention': 1,
        'holder': holder,
        'holder_project': 'p' if holder else None,
        'holder_live': holder_live,
    }


def test_merge_carries_holder_live_with_the_holder_it_describes():
    (merged,) = _merge_module_lists(
        [_module(holder=None, holder_live=None)],
        [_module(holder='5', holder_live=False)],
    )
    assert merged['holder'] == '5'
    assert merged['holder_live'] is False
    assert merged['contention'] == 2


def test_merge_keeps_the_existing_holder_live_when_incoming_has_no_holder():
    (merged,) = _merge_module_lists(
        [_module(holder='5', holder_live=False)],
        [_module(holder=None, holder_live=None)],
    )
    assert merged['holder'] == '5'
    assert merged['holder_live'] is False
    assert merged['contention'] == 2
