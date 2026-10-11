"""Tests for ``task_metadata_selection.select_task_ids_by_metadata``, the one gate-selector walk.

Both gate selectors — ``curator_gate_resolution_sweep.extract_open_gate_task_ids``
and ``gate_owned_finding_phrasing.extract_human_gated_task_ids`` — walk the
task list through this function and differ only in the metadata predicate
they pass, so the traversal contract (malformed shapes skipped, ``str`` id
coercion, sorted dedup, empty input) is pinned once, here.

Rows mixing in non-``dict`` elements are passed as plain lists: the helper
takes ``Iterable[object]``, so such an element is in-contract for it, whereas
``FilteredTaskTree.active_tasks`` is declared ``list[dict]``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from fused_memory.reconciliation.curator_gate_resolution_sweep import extract_open_gate_task_ids
from fused_memory.reconciliation.gate_owned_finding_phrasing import extract_human_gated_task_ids
from fused_memory.reconciliation.task_filter import FilteredTaskTree
from fused_memory.reconciliation.task_metadata_selection import select_task_ids_by_metadata


class TestSelectTaskIdsByMetadata:
    """select_task_ids_by_metadata(tasks, metadata_predicate) walks the task list once."""

    def test_predicate_decides_membership(self):
        """A task is selected exactly when its metadata satisfies the predicate."""
        tree = FilteredTaskTree(active_tasks=[
            {'id': 1, 'metadata': {'kind': 'x'}},
            {'id': 2, 'metadata': {'kind': 'y'}},
        ])

        result = select_task_ids_by_metadata(tree.active_tasks, lambda m: m.get('kind') == 'x')

        assert result == ['1'], (
            f'only tasks whose metadata satisfies the predicate may be selected, got {result!r}'
        )

    def test_predicate_only_ever_receives_a_dict_metadata(self):
        """The predicate is never fed a non-dict, and receives the task's own metadata object."""
        m5: dict[str, Any] = {'operational_mode': 'gate'}
        recorded: list[object] = []

        def record(metadata: dict[str, Any]) -> bool:
            recorded.append(metadata)
            return True

        select_task_ids_by_metadata(
            [
                'not-a-dict',
                None,
                {'id': 1, 'metadata': 'str'},
                {'id': 2, 'metadata': ['x']},
                {'id': 3, 'metadata': None},
                {'id': 4},
                {'id': 5, 'metadata': m5},
            ],
            record,
        )

        assert len(recorded) == 1 and recorded[0] is m5, (
            'the walk must skip non-dict tasks and non-dict/absent metadata BEFORE '
            'calling the predicate, so a predicate may .get() freely; got '
            f'{recorded!r}'
        )

    def test_skips_malformed_shapes_without_raising(self):
        """Non-dict elements, non-dict/absent metadata and missing/None ids contribute nothing."""
        result = select_task_ids_by_metadata(
            [
                'not-a-dict',
                None,
                {'id': 1, 'metadata': 'operational_mode=gate'},
                {'id': 2, 'metadata': ['gate']},
                {'id': 3, 'metadata': None},
                {'id': 4},
                {'id': None, 'metadata': {'operational_mode': 'gate'}},
                {'metadata': {'operational_mode': 'gate'}},
                {'id': 12, 'metadata': {'operational_mode': 'gate'}},
            ],
            lambda m: True,
        )

        assert result == ['12'], (
            'malformed elements must be skipped without raising — a spurious id '
            f'would query the wrong key or rewrite an unrelated finding; got {result!r}'
        )

    def test_coerces_ids_to_str_and_returns_sorted_deduped(self):
        """int ids coerce to str; int and str spellings of one id collapse; the result is sorted."""
        tree = FilteredTaskTree(active_tasks=[
            {'id': 5563, 'metadata': {'operational_mode': 'gate'}},
            {'id': '5561', 'metadata': {'operational_mode': 'gate'}},
            {'id': 5561, 'metadata': {'operational_mode': 'gate'}},
        ])

        result = select_task_ids_by_metadata(tree.active_tasks, lambda m: True)

        assert result == ['5561', '5563'], (
            'int and str spellings of one id must collapse, and the result must '
            f'be sorted; got {result!r}'
        )

    def test_empty_input_returns_empty_list(self):
        """No tasks -> []."""
        assert select_task_ids_by_metadata(FilteredTaskTree().active_tasks, lambda m: True) == []
        assert select_task_ids_by_metadata([], lambda m: True) == []


def _gate_rows_with_malformed_shapes() -> list[dict]:
    """Malformed shapes plus valid rows that BOTH gate predicates match.

    Every valid row carries ``operational_mode == 'gate'``, so the two
    selectors must return the same ids from it.
    """
    return FilteredTaskTree(active_tasks=[
        'not-a-dict',  # type: ignore[list-item]
        None,  # type: ignore[list-item]
        {'id': 1, 'metadata': 'operational_mode=gate'},
        {'id': 2, 'metadata': ['gate']},
        {'id': 3, 'metadata': None},
        {'id': 4},
        {'id': None, 'metadata': {'operational_mode': 'gate'}},
        {'metadata': {'operational_mode': 'gate'}},
        {'id': 5563, 'metadata': {'operational_mode': 'gate'}},
        {'id': '5561', 'metadata': {'operational_mode': 'gate'}},
        {'id': 5561, 'metadata': {'operational_mode': 'gate'}},
    ]).active_tasks


_GATE_SELECTORS = pytest.mark.parametrize(
    'selector',
    [extract_open_gate_task_ids, extract_human_gated_task_ids],
    ids=['open_gate', 'human_gated'],
)


class TestGateSelectorsShareTheTraversal:
    """Both public gate selectors honour the traversal contract pinned above.

    Driven through their public seams with one shared fixture, so each
    selector's documented "malformed shapes are skipped" contract stays
    covered without either suite restating it.
    """

    @_GATE_SELECTORS
    def test_skips_malformed_shapes_and_returns_sorted_deduped_str_ids(
        self, selector: Callable[[list[dict]], list[str]],
    ):
        """Malformed shapes contribute nothing; int and str ids collapse; the result is sorted."""
        result = selector(_gate_rows_with_malformed_shapes())

        assert result == ['5561', '5563'], (
            'malformed elements must be skipped without raising, int and str '
            'spellings of one id must collapse, and the result must be sorted; '
            f'got {result!r}'
        )

    @_GATE_SELECTORS
    def test_empty_input_returns_empty_list(
        self, selector: Callable[[list[dict]], list[str]],
    ):
        """No active tasks -> []."""
        assert selector(FilteredTaskTree().active_tasks) == []
        assert selector([]) == []
