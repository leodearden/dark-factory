"""Tests for gate-owned finding phrasing — Stage 1 norm + code gate (task 4814).

Stage 1 (``memory_consolidator``) sometimes writes a finding's
``suggested_action`` in words that read as authorizing Stage 2 to DECIDE
("Stage 2 or operator should decide", "enumerate", "extend") when the
finding's cited task is a HUMAN GATE (``metadata.operational_mode == 'gate'``
and/or ``metadata.always_escalates == true``).  Stage 2 holds
``update_task``/``set_task_status``, so that phrasing invites a stage to close
a question only a human may answer.  Evidence: autopilot_video run 8b2d3371,
Stage 1 findings 640d4ceb (task 645) and 66af9601 (task 652) — Stage 2 caught
both that cycle, and the recurring cost is that it must catch them EVERY
cycle.

Covers:
- extract_human_gated_task_ids: pure selector over
  ``FilteredTaskTree.active_tasks`` returning the sorted, deduped str ids of
  human-gate-owned tasks.  Deliberately a WIDER population than its sibling
  ``curator_gate_resolution_sweep.extract_open_gate_task_ids`` (which filters
  on mode alone), and deliberately STRICTER than a truthy read.
"""

from __future__ import annotations

from fused_memory.reconciliation.gate_owned_finding_phrasing import (
    extract_human_gated_task_ids,
)
from fused_memory.reconciliation.task_filter import FilteredTaskTree


class TestExtractHumanGatedTaskIds:
    """extract_human_gated_task_ids(tasks) selects human-gate-owned tasks.

    Inputs are built as ``FilteredTaskTree(active_tasks=[...])`` and the
    ``active_tasks`` field is passed, mirroring how the sibling selector
    ``extract_open_gate_task_ids`` is exercised in
    ``test_curator_gate_resolution_sweep.py`` — that suite is the live proof
    that ``task['metadata']`` is reachable as a dict on those dicts.

    The predicate is ``operational_mode == 'gate'`` OR ``always_escalates is
    True``, mirroring the strictness of the canonical gate predicate
    ``TaskInterceptor._is_gate_metadata`` (which this module deliberately does
    not import — see the module docstring).  Strictness errs fail-safe: a
    strict predicate can only UNDER-select, so a non-gate finding is never
    rewritten.
    """

    def test_selects_operational_mode_gate_task(self):
        """metadata.operational_mode == 'gate' selects (the run-8b2d3371 task 645 shape)."""
        tree = FilteredTaskTree(active_tasks=[
            {'id': 645, 'status': 'blocked', 'metadata': {'operational_mode': 'gate'}},
        ])

        result = extract_human_gated_task_ids(tree.active_tasks)

        assert result == ['645'], (
            f'operational_mode == "gate" must select the task, got {result!r}'
        )

    def test_selects_always_escalates_task_without_operational_mode(self):
        """always_escalates=True with NO operational_mode selects (the task 652 shape).

        This is the population the shipped sibling ``extract_open_gate_task_ids``
        provably misses — it filters on mode alone — which is why task 4814
        needs its own selector rather than reusing that one.
        """
        tree = FilteredTaskTree(active_tasks=[
            {'id': 652, 'status': 'blocked', 'metadata': {'always_escalates': True}},
        ])

        result = extract_human_gated_task_ids(tree.active_tasks)

        assert result == ['652'], (
            'always_escalates is True must select even with no operational_mode '
            f'key — the operational-routing coercion produces exactly that shape; got {result!r}'
        )

    def test_task_carrying_both_markers_is_selected_exactly_once(self):
        """A task with both gate markers contributes one id, not two (the return is deduped)."""
        tree = FilteredTaskTree(active_tasks=[
            {
                'id': 645,
                'status': 'blocked',
                'metadata': {'operational_mode': 'gate', 'always_escalates': True},
            },
        ])

        result = extract_human_gated_task_ids(tree.active_tasks)

        assert result == ['645'], (
            f'a task matching both clauses must appear exactly once, got {result!r}'
        )

    def test_non_gate_task_is_not_selected(self):
        """operational_mode='llm' with no always_escalates is not a gate."""
        tree = FilteredTaskTree(active_tasks=[
            {'id': 4242, 'status': 'pending', 'metadata': {'operational_mode': 'llm'}},
            {'id': 4243, 'status': 'pending', 'metadata': {'always_escalates': False}},
            {'id': 4244, 'status': 'pending', 'metadata': {'execution_class': 'operational'}},
        ])

        result = extract_human_gated_task_ids(tree.active_tasks)

        assert result == [], (
            'a non-gate task must never be selected — over-selection would rewrite '
            f'the suggested_action of a finding no human owns; got {result!r}'
        )

    def test_execution_class_operational_alone_is_not_a_gate(self):
        """execution_class='operational' is the THIRD _is_gate_metadata clause, deliberately not adopted.

        ``TaskInterceptor._is_gate_metadata`` answers True on that clause too.
        Task 4814 asks only for ``operational_mode='gate'`` and/or
        ``always_escalates=true``, so adopting it would over-select
        operational-class tasks that are not human gates.
        """
        tree = FilteredTaskTree(active_tasks=[
            {'id': 99, 'metadata': {'execution_class': 'operational'}},
        ])

        assert extract_human_gated_task_ids(tree.active_tasks) == [], (
            'execution_class == "operational" must NOT select — that third '
            '_is_gate_metadata clause is deliberately not adopted by task 4814'
        )

    def test_matches_are_value_and_type_sensitive(self):
        """'GATE' is not 'gate', and the string 'true' is not True.

        Mirrors ``TaskInterceptor._is_gate_metadata``'s stated rationale:
        ``bool('false')`` is True, so a loose read would silently accept the
        opposite of the caller's intent.
        """
        tree = FilteredTaskTree(active_tasks=[
            {'id': 1, 'metadata': {'operational_mode': 'GATE'}},
            {'id': 2, 'metadata': {'operational_mode': None}},
            {'id': 3, 'metadata': {'always_escalates': 'true'}},
            {'id': 4, 'metadata': {'always_escalates': 'false'}},
            {'id': 5, 'metadata': {'always_escalates': 1}},
        ])

        result = extract_human_gated_task_ids(tree.active_tasks)

        assert result == [], (
            'the operational_mode match must be exact == "gate" and the '
            'always_escalates match must be strict `is True`, so non-bool '
            f'truthy values do not satisfy the gate; got {result!r}'
        )

    def test_skips_non_dict_tasks_metadata_and_none_ids_without_raising(self):
        """Non-dict elements, non-dict metadata, and None ids are all skipped."""
        tree = FilteredTaskTree(active_tasks=[
            'not-a-dict',  # type: ignore[list-item]
            None,  # type: ignore[list-item]
            {'id': 1, 'metadata': 'operational_mode=gate'},
            {'id': 2, 'metadata': ['gate']},
            {'id': 3, 'metadata': None},
            {'id': 4},
            {'id': None, 'metadata': {'operational_mode': 'gate'}},
            {'metadata': {'always_escalates': True}},
            {'id': 12, 'metadata': {'operational_mode': 'gate'}},
        ])

        result = extract_human_gated_task_ids(tree.active_tasks)

        assert result == ['12'], (
            'malformed elements must be skipped without raising — a spurious id '
            f'would rewrite an unrelated finding; got {result!r}'
        )

    def test_coerces_int_ids_and_returns_sorted_deduped(self):
        """int ids coerce to str; the result is sorted and deduped."""
        tree = FilteredTaskTree(active_tasks=[
            {'id': 652, 'metadata': {'always_escalates': True}},
            {'id': '645', 'metadata': {'operational_mode': 'gate'}},
            {'id': 645, 'metadata': {'operational_mode': 'gate'}},
        ])

        result = extract_human_gated_task_ids(tree.active_tasks)

        assert result == ['645', '652'], (
            'int and str spellings of one id must collapse, and the result must '
            f'be sorted; got {result!r}'
        )

    def test_empty_input_returns_empty_list(self):
        """No active tasks -> []."""
        assert extract_human_gated_task_ids(FilteredTaskTree().active_tasks) == []
        assert extract_human_gated_task_ids([]) == []
