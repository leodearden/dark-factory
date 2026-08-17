"""Tests for the resolved-curator-gate Mem0 source sweep (task 3084).

Stage 1 (``MemoryConsolidator``) has no deterministic sweep that notices when
a human-curator gate task (``metadata.operational_mode == 'gate'``) has in
fact been resolved.  The resolution evidence is already deterministic and
already in Mem0 — the reify curator writes its ruling stamped
``metadata.source == f'curator_gate_{task_id}'`` (independently documented at
``fused-memory/tests/fixtures/README.md``) — but nothing reads it back, so
detection is an ad-hoc Stage-3 spot-check that misses roughly a quarter of
cases (reify run ec45eed0: gates 5561 and 5563 were resolved-but-stale and
went undetected).

Covers:
- curator_gate_source / CURATOR_GATE_SOURCE_TEMPLATE: the single owner of the
  ``curator_gate_{task_id}`` source-key spelling — the one load-bearing
  string of this task.
- extract_open_gate_task_ids: pure selector that reads
  ``FilteredTaskTree.active_tasks`` and returns the sorted, deduped str ids
  of tasks carrying ``metadata.operational_mode == 'gate'``.
"""

from __future__ import annotations

from fused_memory.reconciliation.curator_gate_resolution_sweep import (
    CURATOR_GATE_SOURCE_TEMPLATE,
    curator_gate_source,
    extract_open_gate_task_ids,
)
from fused_memory.reconciliation.task_filter import FilteredTaskTree


class TestCuratorGateSource:
    """curator_gate_source(task_id) owns the ``curator_gate_{task_id}`` spelling.

    Every Mem0 read this module performs filters on exactly this string, so a
    divergence here is silently a zero-recall sweep — the failure mode this
    task exists to fix.  These tests pin the spelling, the int coercion, and
    the template/helper identity (INV-5: one copy in the tree).
    """

    def test_str_task_id_yields_curator_gate_key(self):
        """A str task id formats to the exact observed key (reify gate 5561)."""
        assert curator_gate_source('5561') == 'curator_gate_5561', (
            'source key must be the exact curator-written spelling; '
            f'got {curator_gate_source("5561")!r}'
        )

    def test_int_task_id_is_coerced_to_the_same_key(self):
        """An int task id must not silently produce a different key than the str form."""
        assert curator_gate_source(5561) == 'curator_gate_5561', (
            'int task ids must be coerced to str before formatting, else an '
            f'int-typed caller queries a different key; got {curator_gate_source(5561)!r}'
        )
        assert curator_gate_source(5561) == curator_gate_source('5561'), (
            'int and str spellings of the same task id must collapse to one key'
        )

    def test_template_and_helper_are_one_definition(self):
        """CURATOR_GATE_SOURCE_TEMPLATE.format(...) equals the helper's output (INV-5)."""
        assert CURATOR_GATE_SOURCE_TEMPLATE.format(task_id='5561') == curator_gate_source('5561'), (
            'the template and the helper must be one definition, not two copies '
            'that can drift apart'
        )
        assert CURATOR_GATE_SOURCE_TEMPLATE == 'curator_gate_{task_id}', (
            f'template spelling drifted; got {CURATOR_GATE_SOURCE_TEMPLATE!r}'
        )


class TestExtractOpenGateTaskIds:
    """extract_open_gate_task_ids(tasks) selects operational_mode == 'gate' tasks.

    Inputs are built as ``FilteredTaskTree(active_tasks=[...])`` and the
    ``active_tasks`` field is passed, mirroring how
    ``extract_stalled_gate_backlog_task_ids`` is exercised — that helper is
    the live proof that ``task['metadata']`` is reachable as a dict on those
    dicts.  Restricting to ``active_tasks`` excludes done/cancelled gates for
    free.  The match is deliberately value-sensitive and exact, mirroring
    ``TaskInterceptor._is_gate_metadata``.
    """

    def test_selects_only_operational_mode_gate_tasks(self):
        """A gate task is selected; a non-gate task alongside it is not."""
        tree = FilteredTaskTree(active_tasks=[
            {'id': 5561, 'status': 'blocked', 'metadata': {'operational_mode': 'gate'}},
            {'id': 4242, 'status': 'pending', 'metadata': {'operational_mode': 'llm'}},
        ])

        result = extract_open_gate_task_ids(tree.active_tasks)

        assert result == ['5561'], (
            f'only operational_mode == "gate" tasks may be selected, got {result!r}'
        )

    def test_excludes_missing_empty_and_modeless_metadata(self):
        """No metadata key, empty metadata, and metadata without operational_mode all skip."""
        tree = FilteredTaskTree(active_tasks=[
            {'id': 1, 'status': 'blocked'},
            {'id': 2, 'status': 'blocked', 'metadata': {}},
            {'id': 3, 'status': 'blocked', 'metadata': {'execution_class': 'operational'}},
            {'id': 4, 'status': 'blocked', 'metadata': {'operational_mode': 'gate'}},
        ])

        result = extract_open_gate_task_ids(tree.active_tasks)

        assert result == ['4'], (
            'absent/empty/mode-less metadata must never select a task, got '
            f'{result!r}'
        )

    def test_operational_mode_match_is_value_sensitive(self):
        """operational_mode='llm' is NOT a gate — the match is exact, not truthy."""
        tree = FilteredTaskTree(active_tasks=[
            {'id': 7, 'status': 'pending', 'metadata': {'operational_mode': 'llm'}},
            {'id': 8, 'status': 'pending', 'metadata': {'operational_mode': 'GATE'}},
            {'id': 9, 'status': 'pending', 'metadata': {'operational_mode': None}},
        ])

        result = extract_open_gate_task_ids(tree.active_tasks)

        assert result == [], (
            'the operational_mode comparison must be an exact == "gate" match '
            f'(mirroring TaskInterceptor._is_gate_metadata), got {result!r}'
        )

    def test_skips_non_dict_metadata_without_raising(self):
        """A str/list/None metadata value is skipped, not fed to .get()."""
        tree = FilteredTaskTree(active_tasks=[
            {'id': 1, 'metadata': 'operational_mode=gate'},
            {'id': 2, 'metadata': ['gate']},
            {'id': 3, 'metadata': None},
            {'id': 4, 'metadata': {'operational_mode': 'gate'}},
        ])

        result = extract_open_gate_task_ids(tree.active_tasks)

        assert result == ['4'], (
            f'non-dict metadata must be skipped without raising, got {result!r}'
        )

    def test_skips_non_dict_tasks_and_none_ids(self):
        """A non-dict element, and a gate task whose id is None, contribute nothing."""
        tree = FilteredTaskTree(active_tasks=[
            'not-a-dict',  # type: ignore[list-item]
            None,  # type: ignore[list-item]
            {'id': None, 'metadata': {'operational_mode': 'gate'}},
            {'metadata': {'operational_mode': 'gate'}},
            {'id': 12, 'metadata': {'operational_mode': 'gate'}},
        ])

        result = extract_open_gate_task_ids(tree.active_tasks)

        assert result == ['12'], (
            'non-dict tasks and gates with a missing/None id must be skipped — a '
            f'spurious id would query the wrong source key; got {result!r}'
        )

    def test_coerces_int_ids_and_returns_sorted_deduped(self):
        """int ids coerce to str; the result is sorted and deduped."""
        tree = FilteredTaskTree(active_tasks=[
            {'id': 5563, 'metadata': {'operational_mode': 'gate'}},
            {'id': '5561', 'metadata': {'operational_mode': 'gate'}},
            {'id': 5561, 'metadata': {'operational_mode': 'gate'}},
        ])

        result = extract_open_gate_task_ids(tree.active_tasks)

        assert result == ['5561', '5563'], (
            'int and str spellings of one id must collapse, and the result must '
            f'be sorted; got {result!r}'
        )

    def test_empty_input_returns_empty_list(self):
        """No active tasks -> []."""
        assert extract_open_gate_task_ids(FilteredTaskTree().active_tasks) == []
        assert extract_open_gate_task_ids([]) == []
