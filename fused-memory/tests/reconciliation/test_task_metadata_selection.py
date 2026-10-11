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

from typing import Any

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
