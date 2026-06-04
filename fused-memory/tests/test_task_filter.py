"""Tests for reconciliation/task_filter.py — FilteredTaskTree, filter_task_tree, format_filtered_task_tree."""

from __future__ import annotations

import re

from _fm_helpers import assert_id_title_pairing, make_8df8_scenario

from fused_memory.reconciliation.task_filter import (
    _STATUS_PRIORITY,
    MAX_ACTIVE_TASKS_RENDERED,
    MAX_CANCELLED_TASKS_RETAINED,
    MAX_DONE_TASKS_RETAINED,
    FilteredTaskTree,
    _render_task_line,
    detect_census_inconsistency,
    filter_task_tree,
    format_filtered_task_tree,
    format_task_list,
    id_key,
    render_active_section,
    select_visible_active,
)


def _make_task(tid: int, status: str, title: str | None = None, deps: list | None = None) -> dict:
    return {
        'id': tid,
        'title': title or f'Task {tid}',
        'status': status,
        'dependencies': deps or [],
    }


class TestFilterTaskTree:
    """Tests for filter_task_tree()."""

    def test_partitions_active_done_cancelled_and_other(self):
        """filter_task_tree partitions 8 tasks correctly: 5 active, 1 done, 1 cancelled, 1 other."""
        tasks_data = {
            'tasks': [
                _make_task(1, 'pending'),
                _make_task(2, 'in-progress'),
                _make_task(3, 'blocked'),
                _make_task(4, 'deferred'),
                _make_task(5, 'review'),
                _make_task(6, 'done'),
                _make_task(7, 'cancelled'),
                _make_task(8, 'stalled'),  # unknown status → other
            ]
        }
        result = filter_task_tree(tasks_data)

        assert isinstance(result, FilteredTaskTree)
        active_statuses = {t['status'] for t in result.active_tasks}
        assert active_statuses == {'pending', 'in-progress', 'blocked', 'deferred', 'review'}
        assert len(result.active_tasks) == 5
        assert result.done_count == 1
        assert result.cancelled_count == 1
        assert result.other_count == 1
        assert result.total_count == 8

    def test_handles_empty_and_malformed_inputs(self):
        """filter_task_tree returns empty FilteredTaskTree for empty/malformed inputs."""
        # Empty dict
        result = filter_task_tree({})
        assert result.active_tasks == []
        assert result.done_count == 0
        assert result.cancelled_count == 0
        assert result.other_count == 0
        assert result.total_count == 0

        # tasks is None
        result = filter_task_tree({'tasks': None})
        assert result.active_tasks == []
        assert result.total_count == 0

        # tasks is a list with non-dict elements
        result = filter_task_tree({'tasks': ['not-a-dict', 42, None]})
        assert result.active_tasks == []
        assert result.total_count == 0

        # tasks has task with missing status → other_count
        result = filter_task_tree({'tasks': [{'id': 1}]})
        assert result.active_tasks == []
        assert result.other_count == 1
        assert result.total_count == 1

        # Non-dict top-level inputs
        result = filter_task_tree(None)
        assert result.active_tasks == []
        assert result.done_tasks == []
        assert result.cancelled_tasks == []
        assert result.done_count == 0
        assert result.cancelled_count == 0
        assert result.other_count == 0
        assert result.total_count == 0

        result = filter_task_tree([{'id': 1, 'status': 'pending'}])
        assert result.active_tasks == []
        assert result.done_tasks == []
        assert result.cancelled_tasks == []
        assert result.done_count == 0
        assert result.cancelled_count == 0
        assert result.other_count == 0
        assert result.total_count == 0

        result = filter_task_tree('bad')
        assert result.active_tasks == []
        assert result.done_tasks == []
        assert result.cancelled_tasks == []
        assert result.done_count == 0
        assert result.cancelled_count == 0
        assert result.other_count == 0
        assert result.total_count == 0

    def test_done_tasks_field_defaults_to_empty_list(self):
        """FilteredTaskTree.done_tasks defaults to [] and is independent per instance."""
        tree1 = FilteredTaskTree()
        tree2 = FilteredTaskTree()

        assert hasattr(tree1, 'done_tasks')
        assert tree1.done_tasks == []

        # Mutating one instance's done_tasks must not affect the other
        tree1.done_tasks.append({'id': 99, 'status': 'done'})
        assert tree2.done_tasks == [], (
            'Mutable default arg regression: tree2.done_tasks was affected by tree1 mutation'
        )

    def test_filter_task_tree_populates_done_tasks(self):
        """filter_task_tree captures done task dicts in done_tasks (not just counted)."""
        done6 = _make_task(6, 'done', 'Done task 6')
        done7 = _make_task(7, 'done', 'Done task 7')
        done8 = _make_task(8, 'done', 'Done task 8')
        tasks_data = {
            'tasks': [
                _make_task(1, 'pending'),
                _make_task(2, 'in-progress'),
                done6,
                done7,
                done8,
            ]
        }
        result = filter_task_tree(tasks_data)

        assert len(result.done_tasks) == 3
        assert result.done_count == 3

        # Verify original dict objects are retained (identity-preserving, no copies)
        done_ids = {t['id'] for t in result.done_tasks}
        assert done_ids == {6, 7, 8}

    def test_filter_task_tree_caps_done_tasks_at_max_retained(self):
        """filter_task_tree caps done_tasks at MAX_DONE_TASKS_RETAINED (30) while preserving done_count."""
        tasks_data = {
            'tasks': [_make_task(i, 'done') for i in range(1, 51)]  # 50 done tasks
        }
        result = filter_task_tree(tasks_data)

        assert len(result.done_tasks) == 30, (
            f'Expected 30 done tasks retained, got {len(result.done_tasks)}'
        )
        assert result.done_count == 50, (
            f'done_count must reflect full input count (50), got {result.done_count}'
        )
        assert len(result.done_tasks) < result.done_count, (
            'Consumers should detect overflow via len(done_tasks) < done_count'
        )

    def test_filter_task_tree_caps_cancelled_tasks_at_max_retained(self):
        """filter_task_tree caps cancelled_tasks at MAX_CANCELLED_TASKS_RETAINED while preserving cancelled_count."""
        tasks_data = {
            'tasks': [_make_task(i, 'cancelled') for i in range(1, 51)]  # 50 cancelled tasks
        }
        result = filter_task_tree(tasks_data)

        assert len(result.cancelled_tasks) == MAX_CANCELLED_TASKS_RETAINED, (
            f'Expected {MAX_CANCELLED_TASKS_RETAINED} cancelled tasks retained, '
            f'got {len(result.cancelled_tasks)}'
        )
        assert result.cancelled_count == 50, (
            f'cancelled_count must reflect full input count (50), got {result.cancelled_count}'
        )
        assert len(result.cancelled_tasks) < result.cancelled_count, (
            'Consumers can detect overflow via len(cancelled_tasks) < cancelled_count'
        )
        # Highest IDs (most recent) are retained
        retained_ids = [t['id'] for t in result.cancelled_tasks]
        expected_ids = list(range(50, 50 - MAX_CANCELLED_TASKS_RETAINED, -1))
        assert retained_ids == expected_ids, (
            f'Expected top-{MAX_CANCELLED_TASKS_RETAINED} ids descending {expected_ids}, '
            f'got {retained_ids}'
        )

    def test_filter_task_tree_done_tasks_sorted_by_id_desc(self):
        """filter_task_tree returns done_tasks sorted by id desc, highest-30 retained."""
        import random
        ids = list(range(1, 51))
        random.shuffle(ids)
        tasks_data = {
            'tasks': [_make_task(i, 'done') for i in ids]
        }
        result = filter_task_tree(tasks_data)

        assert len(result.done_tasks) == 30
        result_ids = [t['id'] for t in result.done_tasks]
        assert result_ids == list(range(50, 20, -1)), (
            f'Expected ids 50..21 descending, got {result_ids}'
        )

        # Non-int id must not crash sorting (fallback to 0 in sort key)
        tasks_with_bad_id = {
            'tasks': [
                _make_task(5, 'done'),
                {'id': '5x', 'title': 'Bad id task', 'status': 'done', 'dependencies': []},
                _make_task(3, 'done'),
            ]
        }
        result2 = filter_task_tree(tasks_with_bad_id)
        assert len(result2.done_tasks) == 3  # All three retained (under cap)
        # Must not raise — just verify it ran without error

    def test_sorts_active_by_priority_and_id_desc(self):
        """filter_task_tree sorts active tasks by _STATUS_PRIORITY then ID descending."""
        tasks_data = {
            'tasks': [
                _make_task(10, 'deferred'),
                _make_task(20, 'pending'),
                _make_task(5, 'review'),
                _make_task(15, 'blocked'),
                _make_task(8, 'in-progress'),
                _make_task(3, 'in-progress'),
                _make_task(12, 'pending'),
                _make_task(7, 'blocked'),
            ]
        }
        result = filter_task_tree(tasks_data)

        # All active tasks are present
        assert len(result.active_tasks) == 8

        # Verify sort by priority groups first
        # Priority: in-progress(0) < blocked(1) < review(2) < pending(3) < deferred(4+)
        statuses = [t['status'] for t in result.active_tasks]
        # in-progress tasks come first
        assert statuses[0] == 'in-progress'
        assert statuses[1] == 'in-progress'
        # blocked tasks come after in-progress
        assert statuses[2] == 'blocked'
        assert statuses[3] == 'blocked'
        # review comes after blocked
        assert statuses[4] == 'review'
        # pending comes after review
        assert statuses[5] == 'pending'
        assert statuses[6] == 'pending'
        # deferred comes last
        assert statuses[7] == 'deferred'

        # Within same priority, higher IDs sort first
        in_progress_ids = [t['id'] for t in result.active_tasks if t['status'] == 'in-progress']
        assert in_progress_ids == sorted(in_progress_ids, reverse=True)

        blocked_ids = [t['id'] for t in result.active_tasks if t['status'] == 'blocked']
        assert blocked_ids == sorted(blocked_ids, reverse=True)

        pending_ids = [t['id'] for t in result.active_tasks if t['status'] == 'pending']
        assert pending_ids == sorted(pending_ids, reverse=True)

    def test_filter_does_not_descend_into_subtasks(self):
        """filter_task_tree does NOT descend into 'subtasks' — only top-level tasks are counted.

        Post-DF-D behaviour: the scheduler is top-level-only; subtask entries in the
        wire dict are ignored.  A parent task with two nested subtasks yields exactly
        ONE active task (the parent) and zero done/total for the subtasks.

        This is a RED test against pre-DF-D code (which calls _flatten_with_subtasks
        and would yield 2 active tasks and done_count=1).
        """
        tasks_data = {
            'tasks': [
                {
                    'id': '1',
                    'title': 'Parent Task',
                    'status': 'pending',
                    'dependencies': [],
                    'subtasks': [
                        {'id': '1.1', 'title': 'Sub active', 'status': 'pending', 'dependencies': []},
                    ],
                }
            ]
        }
        result = filter_task_tree(tasks_data)

        assert len(result.active_tasks) == 1, (
            f'Expected exactly 1 active task (the parent); '
            f'filter_task_tree must NOT descend into subtasks. '
            f'Got {len(result.active_tasks)} active tasks: {[t["id"] for t in result.active_tasks]}'
        )
        assert result.total_count == 1, (
            f'Expected total_count=1 (top-level only); got {result.total_count}'
        )
        assert str(result.active_tasks[0]['id']) == '1', (
            f"Expected the single active task to be the parent (id='1'), "
            f"got id={result.active_tasks[0]['id']!r}"
        )

    # --- Step: docstring dict-invariant contract (task-709) ---

    def test_filtered_task_tree_docstring_documents_dict_invariant(self):
        """FilteredTaskTree.__doc__ must exist and reference the dict-only invariant.

        Guards against accidental docstring removal: downstream consumers of
        FilteredTaskTree fields (e.g. _select_proactive_sample) omit per-element
        isinstance checks and rely on this documented contract.
        """
        doc = FilteredTaskTree.__doc__
        assert doc is not None, 'FilteredTaskTree must have a docstring'
        doc_lower = doc.lower()
        assert 'dict' in doc_lower or 'dictionary' in doc_lower, (
            "FilteredTaskTree docstring must mention 'dict'/'dictionary' to document the element-type contract"
        )
        assert 'invariant' in doc_lower or 'contract' in doc_lower, (
            "FilteredTaskTree docstring must mention 'invariant' or 'contract' to document the element-type guarantee"
        )


class TestFormatFilteredTaskTree:
    """Tests for format_filtered_task_tree()."""

    def _make_tree(self, active_count: int = 3, done_count: int = 5, cancelled_count: int = 2, other_count: int = 1) -> FilteredTaskTree:
        active = [_make_task(i + 1, 'pending', f'Task title {i + 1}') for i in range(active_count)]
        total = active_count + done_count + cancelled_count + other_count
        return FilteredTaskTree(
            active_tasks=active,
            done_count=done_count,
            cancelled_count=cancelled_count,
            other_count=other_count,
            total_count=total,
        )

    def test_includes_active_list_and_summary(self):
        """format_filtered_task_tree includes each task and the summary line."""
        tree = self._make_tree(active_count=3, done_count=5, cancelled_count=2, other_count=1)
        output = format_filtered_task_tree(tree)

        # Each active task must appear
        for i in range(1, 4):
            assert f'Task title {i}' in output

        # Summary line with em dash
        assert '5 done, 2 cancelled \u2014 omitted' in output

    def test_caps_at_max_tasks_and_under_budget(self):
        """Regression: format_filtered_task_tree must honour max_chars and emit the
        max_tasks-cap header phrase when active tasks exceed max_tasks.

        With 500 active tasks and the default max_tasks=50 cap, 450 tasks are omitted.
        The header must contain a phrase with '450 more active' and 'max_tasks'
        (the format emitted at task_filter.py when omitted_active > 0) and the
        total output must not exceed max_chars (default 50,000 chars).

        The regex pins the count (450) and intent (max_tasks cap) while tolerating
        benign preposition rewording (e.g. 'omitted due to' vs 'omitted by').
        """
        # 500 active tasks with plausible-length titles
        active = [
            _make_task(i, 'pending', f'Task title {i}')
            for i in range(1, 501)
        ]
        tree = FilteredTaskTree(
            active_tasks=active,
            done_count=340,
            cancelled_count=20,
            other_count=0,
            total_count=860,
        )

        output = format_filtered_task_tree(tree)

        # Output must not exceed max_chars budget (default 50,000)
        assert len(output) <= 50_000

        # Task 51 is beyond the max_tasks=50 cap — structural match immune to ID-range changes
        assert '\n- [51] ' not in output

        # Header must contain the max_tasks-cap omission phrase: pins count + verb + intent,
        # tolerates preposition rewording (e.g. 'omitted by' vs 'omitted due to')
        assert re.search(r'450\s+more active omitted (by|due to) max_tasks cap', output)

    def test_char_budget_clamps_below_max_tasks(self):
        """When max_chars forces truncation below the max_tasks cap, the truncation notice
        reflects post-cap survivors, not total active.

        Regression guard for task 480 (esc-480-107).
        """
        tree = self._make_tree(active_count=10, done_count=0, cancelled_count=0, other_count=0)

        # max_chars is chosen tight enough to exercise three regimes in one pass:
        #   (1) max_tasks=5 caps the 10 active tasks to 5 post-cap survivors,
        #   (2) the first-pass accumulator overflows, admitting only some task lines,
        #   (3) the lazy pop loop fires, dropping at least one line to bring the result
        #       within budget and emitting the truncation notice.
        # The exact byte counts are intentionally not pinned here; the assertions below
        # validate the invariants directly from the rendered output.
        # Note: max_chars was raised from 240 to 270 when the header grew by the
        # 'highest task id: N' token (~20 chars) added in step-4 of task 1516.
        max_tasks = 5
        max_chars = 270
        output = format_filtered_task_tree(tree, max_tasks=max_tasks, max_chars=max_chars)

        # Output must honour the char budget
        assert len(output) <= max_chars

        # The char-budget clamp branch must have fired — look for the truncation notice
        match = re.search(r'\.\.\. and (\d+) more active \(truncated for budget\)', output)
        assert match is not None, f'Expected truncation notice in output: {output!r}'
        trimmed_count = int(match.group(1))
        # Count surviving task lines dynamically: each line rendered by _render_task_line
        # starts with '- [N]' at the beginning of a line.
        kept_count = len(re.findall(r'^- \[\d+\]', output, re.MULTILINE))
        # Sanity bound: made explicit here (also implied by the task-1 regex below) so
        # that a failure is diagnosed in terms of kept_count before reaching the
        # anchored-line check.
        assert kept_count >= 1, (
            f'kept_count={kept_count}: no task lines survived — the lazy pop loop may have '
            f'over-truncated or _render_task_line format changed'
        )

        # Lower bound: at least one task was dropped by the char-budget clamp, confirming
        # the lazy pop loop genuinely fired (not just the initial accumulator cycle).
        # If trimmed_count=0, the budget arithmetic has drifted and the pop regime is
        # no longer being exercised.
        assert trimmed_count >= 1, (
            f'trimmed_count={trimmed_count} should be >= 1; '
            f'the lazy pop loop did not fire — budget may be too loose or derivation drifted'
        )

        # At least one task line must survive the lazy pop loop — guards against the
        # regression where the notice fires but kept_lines ends up empty.
        # Anchored on the full task-line prefix format from _render_task_line
        # (f'- [{tid}] ({status}) {title}') to avoid false-positive matches from
        # bracketed numbers that may appear in the header or from higher task IDs
        # whose string representation contains '1' as a substring.
        assert re.search(r'- \[1\] \(pending\) Task title 1', output), (
            'Task 1 line should survive the lazy pop loop; '
            'if missing, the budget accounting has regressed'
        )

        # trimmed_count must equal max_tasks minus the kept task lines.  Exact equality
        # catches: (a) the total_active bug where buggy trimmed_count = 10 - kept instead
        # of 5 - kept, which fails because 10-kept != 5-kept; (b) subtler off-by-one
        # errors in truncation accounting that an upper bound alone would not catch.
        # Using kept_count (parsed from the output) decouples from byte-level arithmetic
        # while preserving the same regression-detection strength.
        assert trimmed_count == max_tasks - kept_count, (
            f'trimmed_count={trimmed_count} should be {max_tasks} - {kept_count} = '
            f'{max_tasks - kept_count} (max_tasks minus surviving task lines); '
            f'bug: trimmed_count may track total_active instead of len(active[:max_tasks])'
        )

    def test_empty_active_and_empty_tree(self):
        """format_filtered_task_tree handles empty FilteredTaskTree gracefully."""
        # Completely empty tree
        empty_tree = FilteredTaskTree(
            active_tasks=[],
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=0,
        )
        output = format_filtered_task_tree(empty_tree)
        assert '0 active' in output
        assert 'No active tasks.' in output

        # Tree with done/cancelled but 0 active
        done_only = FilteredTaskTree(
            active_tasks=[],
            done_count=10,
            cancelled_count=3,
            other_count=0,
            total_count=13,
        )
        output2 = format_filtered_task_tree(done_only)
        # Summary line should still be present
        assert '10 done, 3 cancelled \u2014 omitted' in output2
        assert 'No active tasks.' in output2

    def test_trimmed_count_relative_to_max_tasks_cap(self):
        """Regression: trimmed_count in the truncation notice must be bounded by max_tasks,
        not by total_active.

        With 200 active tasks, max_tasks=50, and max_chars=300 (tiny budget), the notice
        must report a count <= 50 (tasks dropped from the 50-task render cap), never a
        count anywhere near 200 (total_active). The implementation uses
        `trimmed_count = len(active) - len(kept_lines)` where
        `active = tree.active_tasks[:max_tasks]`, so trimmed_count is always <= max_tasks.
        """
        active = [_make_task(i, 'pending', f'Task {i}') for i in range(1, 201)]
        tree = FilteredTaskTree(
            active_tasks=active,
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=200,
        )
        output = format_filtered_task_tree(tree, max_tasks=50, max_chars=300)

        # The truncation notice must appear because max_chars=300 is tiny
        assert 'truncated for budget' in output, 'Expected truncation notice in output'

        # Extract the trimmed_count from '... and N more active (truncated for budget)'
        match = re.search(r'\.\.\. and (\d+) more active \(truncated for budget\)', output)
        assert match is not None, f'Could not find truncation notice in: {output!r}'
        trimmed_count = int(match.group(1))

        # Must be <= max_tasks (50), not inflated to ~200
        assert trimmed_count <= 50, (
            f'trimmed_count={trimmed_count} exceeds max_tasks=50; '
            f'bug: using total_active instead of len(active[:max_tasks])'
        )

    def test_deps_none_normalized_to_empty_list(self):
        """deps=None (explicitly set) must render as [] not None.

        When a task dict has 'dependencies': None (key present, value None), the formatter
        must treat it as an empty list. Bug: `t.get('dependencies', [])` returns None
        when the key is present with value None; should use `t.get('dependencies') or []`.
        """
        task_with_none_deps = {
            'id': 1,
            'title': 'Task with None deps',
            'status': 'pending',
            'dependencies': None,  # explicit None, not missing
        }
        tree = FilteredTaskTree(
            active_tasks=[task_with_none_deps],
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=1,
        )
        output = format_filtered_task_tree(tree)

        # Must render deps as empty list, not None
        assert 'deps=[]' in output, f'Expected deps=[] in output, got: {output!r}'
        assert 'deps=None' not in output, f'Found deps=None in output: {output!r}'

    def test_negative_budget_returns_header_plus_summary(self):
        """Regression: when header+summary exceeds max_chars so that the remaining budget
        for task lines is <= 0, format_filtered_task_tree must early-return
        header + summary_line without appending a truncation notice.

        With max_chars=50 and 5 active tasks, header+summary alone exceed 50 chars, the
        budget goes non-positive, and the `budget <= 0` guard in format_filtered_task_tree
        must short-circuit before any truncation notice is appended.
        """
        active = [_make_task(i, 'pending', f'Task {i}') for i in range(1, 6)]
        tree = FilteredTaskTree(
            active_tasks=active,
            done_count=2,
            cancelled_count=1,
            other_count=0,
            total_count=8,
        )
        output = format_filtered_task_tree(tree, max_chars=50)

        # Must not crash
        assert isinstance(output, str)

        # Must contain the header marker
        assert '### Active Task Tree' in output

        # Must contain the summary em-dash line
        assert '\u2014 omitted' in output

        # Must NOT contain the truncation notice when budget is exhausted with no kept lines
        assert 'truncated for budget' not in output, (
            f'Found "truncated for budget" in output when budget was exhausted: {output!r}'
        )

    def test_budget_reserve_matches_actual_notice_length(self):
        """Regression: format_filtered_task_tree must enforce len(output) <= max_chars even
        when the truncation-notice length is not known until after line accumulation.

        The implementation computes budget = max_chars - len(header) - len(summary_line)
        (no magic fixed reserve) and then uses a lazy verification loop that pops kept
        lines until the realized notice length fits within max_chars. This test exercises
        that path with a large input (N=10,000 via repeated-reference trick) and a tight
        budget that forces truncation and at least one pop iteration.

        Uses repeated-reference trick ([same_dict]*N) to keep allocations under 1 MB
        instead of creating N full task dicts.
        """
        # Pre-clamp (budget arithmetic, unchanged):
        # Task line: "- [1] (pending) T deps=[]" = 25 chars. Pre-clamp header:
        #   "### Active Task Tree\n(10000 active shown, 0 done, 0 cancelled, 0 other, 10000 total)\n"
        #   = 85 chars. summary_line = "0 done, 0 cancelled — omitted" = 29 chars.
        # budget = 500 - 85 - 29 = 386. Each line costs 26 chars (25 + newline sep).
        # 14 fit (14×26=364 ≤ 386; 15×26=390 > 386). trimmed=9986, notice=49 chars.
        # Initial result = 85 + (14×25+13) + 49 + 29 = 526 > 500.
        # Lazy loop pops 1 line → kept=13, trimmed=9987, notice=49;
        # intermediate result = 85 + (13×25+12) + 49 + 29 = 500 ≤ 500. Loop exits.
        #
        # Post-clamp rebuild (returned in final output):
        # _select_visible_active_with_body rebuilds the header using final_shown = len(kept_lines) = 13.
        # New header: "### Active Task Tree\n(13 active shown, 0 done, 0 cancelled, 0 other, 10000 total)\n"
        #           = 82 chars (5-digit shown → 2-digit, saves 3 chars).
        # Final returned output = 82 + (13×25+12) + 49 + 29 = 497 chars ≤ 500.
        # The pre-clamp intermediate (500) is not the assertion target — because the rebuilt
        # header is monotonically ≤ the pre-clamp header, the lazy-loop invariant carries through.

        single_task = {'id': 1, 'title': 'T', 'status': 'pending', 'dependencies': []}
        n = 10_000
        # Repeated-reference trick: list of n pointers to same dict — keeps memory < 1 MB.
        # Safe only because format_filtered_task_tree treats task dicts as read-only; any future in-place mutation in the formatter (e.g. dep normalization) would alias across all N entries and skew results.
        active_large = [single_task] * n
        tree_large = FilteredTaskTree(
            active_tasks=active_large,
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=n,
        )

        max_chars = 500  # Tight budget: forces truncation and exercises the lazy pop loop
        output = format_filtered_task_tree(tree_large, max_tasks=n, max_chars=max_chars)

        assert len(output) <= max_chars, (
            f'Output length {len(output)} exceeds max_chars={max_chars}; '
            f'the lazy verification loop must pop task lines until the output fits'
        )

    def test_budget_lazy_loop_handles_7_digit_trimmed_count(self, monkeypatch):
        """Regression: format_filtered_task_tree must enforce len(output) <= max_chars even
        when trimmed_count reaches 7+ digits, where a fixed-width reserve approach would
        have under-allocated space for the truncation notice.

        The implementation uses a lazy verification loop that re-measures the realized
        notice length after each pop iteration. This test exercises the 7+ digit path
        where a fixed-width reserve keyed on 4-digit trimmed counts would overflow.

        Performance: _render_task_line is monkeypatched to return 'X' (1 char) instead of
        the real ~25-char line. This collapses per-line cost from ~25 chars to 1 char,
        reducing peak memory from ~100 MB to ~20 MB and wall time from seconds to
        sub-second, while preserving lazy-loop + 7-digit-trimmed_count coverage.

        N=1_100_000 ensures trimmed_count is always 7+ digits regardless of how many lines
        the lazy loop retains (1,100,000 - any_kept ≈ 1,099,500+). max_chars=500 provides
        330 chars of headroom above the minimum viable output (header+notice+summary=170),
        making the test insensitive to header format changes while still exercising the
        lazy loop (initial result=551 > 500).

        Failure mode guarded: if the implementation ever switches to a fixed-width reserve
        (e.g. reserving 49 chars for a notice with a 4-digit count), then a 7-digit
        trimmed_count would produce a notice 3 chars longer, causing output to exceed
        max_chars. This test fails loudly in that case.
        """
        # Monkeypatch _render_task_line to return 'X' (1 char) for all tasks.
        # format_filtered_task_tree calls the function via the module-local reference,
        # so we patch at the module level to ensure the stub is used during the call.
        monkeypatch.setattr(
            'fused_memory.reconciliation.task_filter._render_task_line',
            lambda task: 'X',
        )

        # Pre-clamp (budget arithmetic, unchanged):
        # Stub lines are 1 char each. Pre-clamp header:
        #   "### Active Task Tree\n(1100000 active shown, 0 done, 0 cancelled, 0 other, 1100000 total)\n"
        #   = 89 chars. summary_line = "0 done, 0 cancelled — omitted" = 29 chars.
        # budget = 500 - 89 - 29 = 382. Each stub line costs 2 chars (1 char + newline sep).
        # initial kept = floor(383/2) = 191. trimmed_count = 1_100_000 - 191 = 1_099_809 (7 digits).
        # Notice = "... and 1099809 more active (truncated for budget)" framed by \n = 52 chars.
        # Initial result = 89 + 381 + 52 + 29 = 551 > 500. Lazy loop fires, pops ~26 lines.
        # After ~26 pops: kept=165, body=329 (165+164), trimmed=1_099_835 (7 digits — regression preserved),
        # notice=52; intermediate result = 89 + 329 + 52 + 29 = 499 ≤ 500. Loop exits.
        #
        # Post-clamp rebuild (returned in final output):
        # Rebuilt header: "### Active Task Tree\n(165 active shown, 0 done, 0 cancelled, 0 other, 1100000 total)\n"
        #              = 85 chars (7-digit shown → 3-digit, saves 4 chars).
        # Final returned output = 85 + 329 + 52 + 29 = 495 chars ≤ 500.
        # The 7-digit trimmed_count regex assertion below still matches: trimmed=1,099,835 has 7 digits.

        single_task = {'id': 1, 'title': 'T', 'status': 'pending', 'dependencies': []}
        n = 1_100_000
        # Repeated-reference trick: list of n pointers to same dict — keeps memory < 2 MB.
        # Safe only because format_filtered_task_tree treats task dicts as read-only;
        # any future in-place mutation in the formatter (e.g. dep normalization) would
        # alias across all N entries and invalidate the trick.
        active_large = [single_task] * n
        tree_large = FilteredTaskTree(
            active_tasks=active_large,
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=n,
        )

        max_chars = 500  # Wide enough to avoid header-format sensitivity; lazy loop still fires
        output = format_filtered_task_tree(tree_large, max_tasks=n, max_chars=max_chars)

        assert len(output) <= max_chars, (
            f'Output length {len(output)} exceeds max_chars={max_chars}; '
            f'the lazy verification loop must handle 7-digit trimmed_count correctly'
        )

        # Extract trimmed_count from the truncation notice and verify 7+ digit path
        m = re.search(r'\.\.\. and (\d+) more active \(truncated for budget\)', output)
        assert m is not None, (
            f'Truncation notice not found in output; got: {output!r}'
        )
        trimmed_count = int(m.group(1))
        assert len(str(trimmed_count)) >= 7, (
            f'trimmed_count={trimmed_count} has fewer than 7 digits; '
            f'the 7+ digit path was not exercised (short-circuited?)'
        )

    def test_deps_more_than_5_renders_truncated(self):
        """Task with >5 deps renders first 5 items with '...' suffix, not the full list.

        When a task has deps=[1,2,3,4,5,6,7,8], the formatter must emit
        'deps=[1, 2, 3, 4, 5]...' and must NOT emit the full list.
        Bug: current code uses `deps={deps}` which renders the full list repr.
        """
        task = {
            'id': 42,
            'title': 'Many deps task',
            'status': 'pending',
            'dependencies': [1, 2, 3, 4, 5, 6, 7, 8],
        }
        tree = FilteredTaskTree(
            active_tasks=[task],
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=1,
        )
        output = format_filtered_task_tree(tree)

        # Must show first 5 items with '...' suffix
        assert 'deps=[1, 2, 3, 4, 5]...' in output, (
            f'Expected truncated deps repr in output, got: {output!r}'
        )
        # Must NOT show the full list
        assert 'deps=[1, 2, 3, 4, 5, 6, 7, 8]' not in output, (
            f'Found full deps list (not truncated) in output: {output!r}'
        )

    def test_deps_exactly_5_renders_full_list_without_ellipsis(self):
        """Task with exactly 5 deps renders the full list without trailing '...'.

        When a task has deps=[10,20,30,40,50], the formatter must emit
        'deps=[10, 20, 30, 40, 50]' with no trailing '...'.
        """
        task = {
            'id': 99,
            'title': 'Five deps task',
            'status': 'pending',
            'dependencies': [10, 20, 30, 40, 50],
        }
        tree = FilteredTaskTree(
            active_tasks=[task],
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=1,
        )
        output = format_filtered_task_tree(tree)

        # Must show full list
        assert 'deps=[10, 20, 30, 40, 50]' in output, (
            f'Expected full deps repr in output, got: {output!r}'
        )
        # Must NOT have trailing '...' after the closing bracket
        assert 'deps=[10, 20, 30, 40, 50]...' not in output, (
            f'Found unexpected ellipsis in output: {output!r}'
        )

    def test_structural_match_resists_id_range_extension(self):
        """Canary: proves old substring assertion 'Task title 51' is fragile
        while structural match '\\n- [51] ' correctly detects omission.

        With 600 active tasks having IDs 510..1109, max_tasks=50 shows
        tasks 510..559.  Task 51 is absent (correctly omitted — it is not
        in the task set at all).  Titles 'Task title 510'..'Task title 519'
        each contain 'Task title 51' as a leading substring, so the old
        assertion::

            assert 'Task title 51' not in output

        would raise AssertionError even though task 51 is correctly absent
        — a false failure.  The structural assertion::

            assert '\\n- [51] ' not in output

        passes correctly because '[51] ' (bracket-51-space) is a different
        token from '[510] ' (bracket-510-space) and cannot be confused.

        This test PASSES and documents precisely WHY the structural form was
        chosen in test_caps_at_max_tasks_and_under_budget.
        """
        # 600 tasks, IDs 510..1109.  active[:50] shows IDs 510..559.
        # Task 51 is not in this set — correctly excluded.
        active = [
            _make_task(i, 'pending', f'Task title {i}')
            for i in range(510, 1110)  # 600 tasks, none is task 51
        ]
        tree = FilteredTaskTree(
            active_tasks=active,
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=600,
        )
        output = format_filtered_task_tree(tree, max_tasks=50)

        # (a) Old-style substring: 'Task title 51' IS present because
        #     'Task title 510'..'Task title 519' are shown and each contains
        #     'Task title 51' as a prefix substring.
        #     The old assertion `assert 'Task title 51' not in output` would
        #     raise AssertionError here — a false failure.
        assert 'Task title 51' in output, (
            "'Task title 51' must be found as a substring of a shown task title "
            "(e.g. 'Task title 510'); if absent, the canary is not exercised correctly"
        )

        # (b) Structural match: '\n- [51] ' is NOT present because task 51 is
        #     correctly absent.  '[51] ' cannot match '[510] ' — different tokens.
        assert '\n- [51] ' not in output, (
            "Task 51 is not in the task set; its rendered line prefix "
            "'\\n- [51] ' must not appear in output"
        )

        # (c) End-to-end header check: 600 active tasks minus 50 shown = 550 omitted.
        #     Confirms the max_tasks cap logic fires correctly for this range of IDs.
        assert re.search(r'550\s+more active omitted (by|due to) max_tasks cap', output), (
            "Expected header phrase '550 more active omitted … max_tasks cap' "
            "(600 tasks − 50 shown = 550 omitted); output was:\n" + output
        )

    def test_many_deps_per_task_stays_under_budget(self):
        """Tasks with many deps (200) do not exhaust the char budget.

        With 50 tasks each having 200 deps, and max_chars=5000, the output
        must stay under budget. Bug: unbounded deps repr inflates each line
        by hundreds of chars, blowing through the budget before other tasks
        are shown.
        """
        big_deps = list(range(1, 201))  # 200 deps per task
        active = [
            _make_task(i, 'pending', f'Task {i}', deps=big_deps)
            for i in range(1, 51)
        ]
        tree = FilteredTaskTree(
            active_tasks=active,
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=50,
        )
        output = format_filtered_task_tree(tree, max_chars=5000)

        assert len(output) <= 5000, (
            f'Output length {len(output)} exceeds max_chars=5000; '
            f'deps display is not being truncated'
        )

    def test_format_filtered_task_tree_renders_onlyselect_visible_active_return(
        self, monkeypatch
    ):
        """format_filtered_task_tree must delegate visible-window selection to
        select_visible_active_with_body (single source of truth requirement).

        A monkeypatched stub makes select_visible_active_with_body return only
        the first 3 tasks (and their pre-rendered body) regardless of input.
        format_filtered_task_tree calls the worker for its visible window and
        reuses the returned body, so only tasks 1-3 are rendered.

        Uses the same module-level monkeypatch pattern as
        test_budget_lazy_loop_handles_7_digit_trimmed_count (line ~643).
        """
        tasks = [_make_task(i, 'pending', f'Task {i}') for i in range(1, 11)]
        tree = FilteredTaskTree(
            active_tasks=tasks,
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=10,
        )

        # Stub: return only the first 3 task dicts and their pre-rendered body,
        # ignoring budget args.  The four extra fields (header, cancelled_section,
        # summary_line) are empty strings — the test only checks task-line presence,
        # not the surrounding strings.
        def _stub_with_body(t, max_tasks=50, max_chars=50_000):  # noqa: ARG001
            first_three = t.active_tasks[:3]
            body = '\n'.join(_render_task_line(task) for task in first_three) + '\n'
            return first_three, body, '', '', ''

        monkeypatch.setattr(
            'fused_memory.reconciliation.task_filter._select_visible_active_with_body',
            _stub_with_body,
        )

        output = format_filtered_task_tree(tree)

        # Tasks 1-3 must be present.
        for tid in (1, 2, 3):
            assert f'- [{tid}] ' in output, (
                f'Task {tid} must appear in output when stub returns first 3 tasks.\n'
                f'Output: {output!r}'
            )

        # Tasks 4-10 must be absent.
        for tid in range(4, 11):
            assert f'- [{tid}] ' not in output, (
                f'Task {tid} must NOT appear when stub limits visible window to 3.\n'
                f'Output: {output!r}'
            )

    def test_render_task_line_invoked_once_per_task_in_format_call(self, monkeypatch):
        """format_filtered_task_tree must render each task line at most once.

        Counts _render_task_line calls and asserts the total equals
        len(visible) + len(cancelled_tasks) — the behavioural invariant that
        guards against double-rendering.  The pre-refactor code triggered
        2 * (active + cancelled) renders; post-refactor it is 1 * (active +
        cancelled).

        Using _render_task_line as the counter rather than _build_surrounding
        pins *behaviour* (output lines rendered) rather than internal structure
        (call count of a private helper), so this test won't falsely fail if the
        implementation is reorganised while the render-once invariant is maintained.

        Uses the same module-level monkeypatch pattern as
        test_budget_lazy_loop_handles_7_digit_trimmed_count (line ~643).
        """
        tasks = [_make_task(i, 'pending', f'Task {i}') for i in range(1, 4)]
        cancelled = [_make_task(10 + i, 'cancelled', f'Cancelled {i}') for i in range(1, 3)]
        tree = FilteredTaskTree(
            active_tasks=tasks,
            done_count=5,
            cancelled_count=2,
            cancelled_tasks=cancelled,
            other_count=0,
            total_count=10,
        )

        # Compute expected count before patching so this call is unaffected.
        visible = select_visible_active(tree)
        expected_count = len(visible) + len(tree.cancelled_tasks)

        call_count = [0]

        def _counting_render_task_line(task):
            call_count[0] += 1
            return _render_task_line(task)

        monkeypatch.setattr(
            'fused_memory.reconciliation.task_filter._render_task_line',
            _counting_render_task_line,
        )

        format_filtered_task_tree(tree)

        assert call_count[0] == expected_count, (
            f'_render_task_line was called {call_count[0]} time(s); '
            f'expected {expected_count} '
            f'({len(visible)} visible active + {len(tree.cancelled_tasks)} cancelled). '
            f'Each task line must be rendered at most once per format_filtered_task_tree call '
            f'(task 1311 refactor guard).'
        )

    def test_header_counts_consistent_under_max_chars_clamp(self):
        """Regression (task 1312): the header 'shown' count must equal the number of task
        lines actually rendered in the body when the max_chars clamp fires below the
        max_tasks cap.

        Before task 1312, _build_surrounding computed 'shown' from the pre-clamp
        tree.active_tasks[:max_tasks] slice.  When _select_visible_active_with_body's
        lazy-pop loop trimmed the body further, the header still reported the pre-clamp
        count (e.g. 50), while the body contained fewer lines (e.g. 13).

        Setup:
          - 100 active tasks, each with a 400-char title → each rendered line ~422 chars.
          - max_tasks=50  → max_tasks cap fires (100 > 50), omitted_active = 50.
          - max_chars=6000 → max_chars clamp fires (50 tasks × ~422 chars >> ~5850 budget).
          - Expected body survivors ≈ 13 (well within the 1..49 partial-truncation regime).
        """
        title_len = 400
        title = 'A' * title_len
        active = [_make_task(i, 'pending', f'{title}-{i}') for i in range(1, 101)]
        tree = FilteredTaskTree(
            active_tasks=active,
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=100,
        )

        max_tasks = 50
        max_chars = 6000
        output = format_filtered_task_tree(tree, max_tasks=max_tasks, max_chars=max_chars)

        # ── Sanity: verify the partial-truncation regime was actually exercised ── #
        kept_count = len(re.findall(r'^- \[\d+\]', output, re.MULTILINE))
        assert 0 < kept_count < max_tasks, (
            f'kept_count={kept_count} is not in the partial-truncation regime (1..{max_tasks - 1}). '
            f'Tune title_len or max_chars so the max_chars clamp fires below the max_tasks cap. '
            f'Current title_len={title_len}, max_chars={max_chars}, max_tasks={max_tasks}.'
        )

        # ── Core regression: header 'shown' count must equal body task-line count ── #
        # Before fix: shown = pre-clamp len(active[:max_tasks]) = 50 ≠ kept_count ≈ 13.
        shown_match = re.search(r'\((\d+) active shown', output)
        assert shown_match is not None, (
            f'Header "(N active shown" phrase not found in output: {output!r}'
        )
        shown = int(shown_match.group(1))
        assert shown == kept_count, (
            f'Header shown={shown} disagrees with body task-line count={kept_count}. '
            f'The header must report the post-max_chars-clamp visible count, '
            f'not the pre-clamp len(active[:max_tasks])={max_tasks}.'
        )

        # ── Semantics pin: omitted_active must reflect max_tasks-cap omissions only ── #
        # omitted_active = len(active) - max_tasks = 100 - 50 = 50.
        # It must NOT be len(active) - kept_count (option B), which would misattribute
        # max_chars-truncated tasks to the "max_tasks cap" phrase.
        omitted_match = re.search(r'(\d+) more active omitted by max_tasks cap', output)
        assert omitted_match is not None, (
            f'Header "N more active omitted by max_tasks cap" phrase not found in output: {output!r}'
        )
        omitted = int(omitted_match.group(1))
        expected_omitted = len(active) - max_tasks  # 100 - 50 = 50
        assert omitted == expected_omitted, (
            f'omitted_active={omitted} should be len(active)-max_tasks={expected_omitted} '
            f'(max_tasks-cap-only semantics). '
            f'If omitted={len(active) - kept_count}, option B was mistakenly applied '
            f'(misattributes max_chars-truncated tasks to the max_tasks cap phrase).'
        )

    def test_header_shown_zero_when_no_line_fits_positive_budget(self):
        """Regression (task 1319): when budget > 0 but no single task line fits, the
        header must report '0 active shown', not the pre-clamp active count.

        This test exercises the lazy-drain branch of _select_visible_active_with_body
        via the greedy-break route: budget is small and positive (> 0), but every
        rendered task line is wider than the budget, so the greedy fill loop breaks
        immediately without adding any line.  kept_lines ends up empty, which should
        trigger a header rebuild reporting shown=0.

        This branch is distinct from:
          - budget <= 0: the entire budget block is skipped.
          - partial-clamp: some lines fit, and kept_count > 0.

        The test acts as a safety net for the step-2 consolidation (task 1319): if
        the refactor accidentally drops the header rebuild for the empty-kept_lines
        path, or if the single tail rebuild misroutes final_shown, this test will
        fail with a non-zero shown count.

        Setup:
          - 5 active pending tasks, each with a 200-char title.
            Rendered line format: '- [i] (pending) AAA...A deps=[]'
            Each rendered line ≈ 224 chars (3+'i'+2+1+7+2+200+1+7).
          - max_chars=120:
            overhead = len(header) + len(summary_line) = 78 + 30 = 108
            budget   = 120 - 108 = 12  (positive, but < 224-char task line)
          Greedy loop: used(0) + len(line)(224) + 1 = 225 > 12 → break.
          kept_lines = [] → lazy-drain branch → shown must be 0.
        """
        title = 'A' * 200
        active = [_make_task(i, 'pending', title) for i in range(1, 6)]
        tree = FilteredTaskTree(
            active_tasks=active,
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=5,
        )

        max_chars = 120
        output = format_filtered_task_tree(tree, max_tasks=50, max_chars=max_chars)

        # ── Sanity: verify that no task line was rendered in the body ── #
        # Guards against accidentally falling into the partial-clamp or fast-path
        # branches if surrounding-string lengths drift from the expected ~108-char
        # overhead.
        task_lines = re.findall(r'^- \[\d+\]', output, re.MULTILINE)
        assert task_lines == [], (
            f'Expected no task lines in output but found {len(task_lines)}. '
            f'The test parameters may need adjusting if surrounding-string lengths changed. '
            f'output={output!r}'
        )

        # ── Core regression: header shown count must be 0 when no line fits ── #
        shown_match = re.search(r'\((\d+) active shown', output)
        assert shown_match is not None, (
            f'Header "(N active shown" phrase not found in output: {output!r}'
        )
        shown = int(shown_match.group(1))
        assert shown == 0, (
            f'Header shown={shown} but expected 0 for the lazy-drain branch '
            f'(budget > 0 but no task line fits). '
            f'The header must report the post-clamp visible count, '
            f'not the pre-clamp active count.'
        )


class TestRenderActiveSection:
    """Tests for the render_active_section(tree) -> (list[dict], str) public helper.

    render_active_section is the single-call API that returns BOTH the
    visible-task list (for hint-section consumption) AND the fully assembled
    prompt string (for the Active Task Tree slot).  These tests pin its contract
    and confirm the render-once invariant.
    """

    def test_render_active_section_returns_visible_list_and_assembled_string(self):
        """render_active_section must return a (list[dict], str) tuple whose elements
        match select_visible_active and format_filtered_task_tree respectively.

        Parity check: for the same tree, the two halves of the returned tuple
        must be byte-identical to what the existing public helpers return.
        """
        tasks = [_make_task(i, 'pending', f'Task {i}') for i in range(1, 4)]
        cancelled = [_make_task(20 + i, 'cancelled', f'Cancelled {i}') for i in range(1, 3)]
        tree = FilteredTaskTree(
            active_tasks=tasks,
            done_count=4,
            cancelled_count=2,
            cancelled_tasks=cancelled,
            other_count=0,
            total_count=9,
        )

        result = render_active_section(tree)

        assert len(result) == 2, (
            f'render_active_section must return a 2-tuple; got {len(result)}-tuple'
        )
        visible_list, assembled_str = result

        expected_visible = select_visible_active(tree)
        assert visible_list == expected_visible, (
            f'render_active_section visible list must match select_visible_active(tree).\n'
            f'Got:      {visible_list!r}\n'
            f'Expected: {expected_visible!r}'
        )

        expected_str = format_filtered_task_tree(tree)
        assert assembled_str == expected_str, (
            f'render_active_section assembled string must match format_filtered_task_tree(tree).\n'
            f'Got:      {assembled_str!r}\n'
            f'Expected: {expected_str!r}'
        )

    def test_render_active_section_renders_each_visible_task_once(self, monkeypatch):
        """render_active_section must render each candidate task line at most once.

        Simulates the assemble_payload payload-assembly pattern: call
        render_active_section once and consume BOTH return values (visible list
        for the hint section, assembled string for the prompt slot).  The
        _render_task_line counter must equal len(candidate_active) +
        len(tree.cancelled_tasks) — at most one render per candidate active task
        (tasks iterated by the worker before budget trimming) plus one per
        cancelled task in the cancelled section.

        ``candidate_active`` is ``tree.active_tasks[:max_tasks]`` — the slice the
        worker considers before any budget trimming.  On the fast path (all tasks
        fit the budget), candidate_active == visible_list; on budget-capped paths
        candidate_active may be larger than visible_list.  The invariant is that
        each candidate is rendered *at most once*, not exactly once — a future
        improvement that renders only kept lines would still satisfy this bound.

        The legacy select_visible_active + format_filtered_task_tree pair would
        produce 2 * len(candidate_active) + 2 * len(tree.cancelled_tasks) invocations.
        """
        call_count = [0]

        def _counting_render_task_line(task):
            call_count[0] += 1
            return _render_task_line(task)

        monkeypatch.setattr(
            'fused_memory.reconciliation.task_filter._render_task_line',
            _counting_render_task_line,
        )

        tasks = [_make_task(i, 'pending', f'Task {i}') for i in range(1, 11)]
        cancelled = [_make_task(100 + i, 'cancelled', f'Cancelled {i}') for i in range(1, 4)]
        tree = FilteredTaskTree(
            active_tasks=tasks,
            done_count=2,
            cancelled_count=3,
            cancelled_tasks=cancelled,
            other_count=0,
            total_count=15,
        )

        # Single call — simulates assemble_payload using both outputs.
        visible_list, _assembled_str = render_active_section(tree)

        # The worker iterates all candidates (active_tasks[:max_tasks]) before
        # trimming to kept_lines, so the correct upper bound is the full candidate
        # slice, not just the returned visible list.  Using visible_list here would
        # produce a false regression if the worker is later optimised to skip
        # rendering lines it knows will be trimmed.
        candidate_active = tree.active_tasks[:50]  # same cap as MAX_ACTIVE_TASKS_RENDERED
        expected_count = len(candidate_active) + len(tree.cancelled_tasks)
        assert call_count[0] == expected_count, (
            f'_render_task_line was called {call_count[0]} time(s); '
            f'expected {expected_count} '
            f'({len(candidate_active)} candidate active + {len(tree.cancelled_tasks)} cancelled). '
            f'Each candidate task must be rendered at most once per render_active_section call.'
        )


class TestFilterTaskTreeDoneAndCancelledLists:
    """Tests for done_tasks and cancelled_tasks list fields on FilteredTaskTree."""

    def test_filter_task_tree_exposes_done_tasks_list(self):
        """filter_task_tree populates done_tasks as a list sorted by id descending."""
        tasks_data = {
            'tasks': [
                {'id': 5, 'title': 'Done 5', 'status': 'done', 'dependencies': []},
                {'id': 10, 'title': 'Done 10', 'status': 'done', 'dependencies': []},
                {'id': 3, 'title': 'Done 3', 'status': 'done', 'dependencies': []},
            ]
        }
        result = filter_task_tree(tasks_data)

        assert isinstance(result.done_tasks, list)
        assert len(result.done_tasks) == 3
        ids = [t['id'] for t in result.done_tasks]
        assert ids == [10, 5, 3], f'Expected [10, 5, 3] (descending), got {ids}'
        assert result.done_count == 3

    def test_filter_task_tree_exposes_cancelled_tasks_list(self):
        """filter_task_tree populates cancelled_tasks as a list sorted by id descending."""
        tasks_data = {
            'tasks': [
                {'id': 7, 'title': 'Cancelled 7', 'status': 'cancelled', 'dependencies': []},
                {'id': 2, 'title': 'Cancelled 2', 'status': 'cancelled', 'dependencies': []},
            ]
        }
        result = filter_task_tree(tasks_data)

        assert isinstance(result.cancelled_tasks, list)
        assert len(result.cancelled_tasks) == 2
        ids = [t['id'] for t in result.cancelled_tasks]
        assert ids == [7, 2], f'Expected [7, 2] (descending), got {ids}'
        assert result.cancelled_count == 2

    def test_filter_task_tree_done_and_cancelled_empty_by_default(self):
        """Empty input yields empty done_tasks and cancelled_tasks lists."""
        result = filter_task_tree({})

        assert result.done_tasks == []
        assert result.cancelled_tasks == []
        assert result.done_count == 0
        assert result.cancelled_count == 0

    def test_done_and_cancelled_lists_sort_with_non_int_ids(self):
        """id_key covers both branches: string-digit ids coerce to int (happy path) and non-parseable ids fall back to 0.

        String-digit id '4' exercises the int()-coercion branch of id_key; it must sort between
        int id=5 and int id=3 (i.e. 4 > 3) confirming coercion drives sort order, not input position.
        Non-parseable ids 'abc' and 'def' exercise the ValueError-fallback branch (id_key=0) and
        sort last in descending order; the -index tiebreaker preserves their input order: abc before def.
        """
        tasks_data = {
            'tasks': [
                # done: two non-int ids ('abc' then 'def') and one string-digit id ('4') interleaved
                # with ints; no literal id=0 to avoid sort-stability ambiguity with the fallback key.
                {'id': 10, 'title': 'Done 10', 'status': 'done', 'dependencies': []},
                {'id': 'abc', 'title': 'Done abc', 'status': 'done', 'dependencies': []},
                {'id': 5, 'title': 'Done 5', 'status': 'done', 'dependencies': []},
                {'id': 3, 'title': 'Done 3', 'status': 'done', 'dependencies': []},
                # id='4' is placed AFTER id=3 in input order so that a correct sort
                # (4 > 3) cannot be confused with a no-op passthrough of input position.
                # It exercises the int()-coercion (not fallback) path of id_key.
                {'id': '4', 'title': 'Done 4', 'status': 'done', 'dependencies': []},
                {'id': 'def', 'title': 'Done def', 'status': 'done', 'dependencies': []},
                # cancelled: mix of int and non-int ids
                {'id': 7, 'title': 'Cancelled 7', 'status': 'cancelled', 'dependencies': []},
                {'id': 'xyz', 'title': 'Cancelled xyz', 'status': 'cancelled', 'dependencies': []},
                {'id': 2, 'title': 'Cancelled 2', 'status': 'cancelled', 'dependencies': []},
            ]
        }
        result = filter_task_tree(tasks_data)

        done_ids = [t['id'] for t in result.done_tasks]
        cancelled_ids = [t['id'] for t in result.cancelled_tasks]

        # '4' has id_key=4 via int() coercion (happy path), so it sorts between 5 and 3.
        # 'abc' and 'def' both have id_key=0 (int() fallback), so they sort last after
        # all int ids (10 > 5 > 4 > 3 > 0). The -index tiebreaker preserves their input order: 'abc' before 'def'.
        assert done_ids == [10, 5, '4', 3, 'abc', 'def'], (
            f"Expected done_tasks id order [10, 5, '4', 3, 'abc', 'def'] — "
            f"string-digit id '4' has id_key=4 via successful int() coercion and sorts between 5 and 3; "
            f"non-int ids 'abc' and 'def' both have id_key=0 via the int() fallback, sort last "
            f"(0 < 3 < 4 < 5 < 10 descending), and the -index tiebreaker preserves their input order "
            f"(abc before def). Got: {done_ids}"
        )

        # 'xyz' has id_key=0 (int() fallback), so it sorts last after 7, 2 (both > 0)
        assert cancelled_ids == [7, 2, 'xyz'], (
            f"Expected cancelled_tasks id order [7, 2, 'xyz'] — non-int 'xyz' has id_key=0 "
            f"via the int() fallback and sorts last (0 < 2 < 7 descending). Got: {cancelled_ids}"
        )

    def test_active_done_cancelled_lists_are_disjoint(self):
        """Every task is routed to exactly one bucket; active/done/cancelled are mutually exclusive."""
        tasks_data = {
            'tasks': [
                # active statuses
                _make_task(1, 'pending'),
                _make_task(2, 'in-progress'),
                _make_task(3, 'blocked'),
                _make_task(4, 'deferred'),
                _make_task(5, 'review'),
                # done
                _make_task(6, 'done'),
                _make_task(7, 'done'),
                # cancelled
                _make_task(8, 'cancelled'),
                _make_task(9, 'cancelled'),
                # unknown/other — must NOT appear in any of the three lists
                _make_task(10, 'stalled'),
            ]
        }
        result = filter_task_tree(tasks_data)

        active_ids = {t['id'] for t in result.active_tasks}
        done_ids = {t['id'] for t in result.done_tasks}
        cancelled_ids = {t['id'] for t in result.cancelled_tasks}

        # Pairwise disjointness — the primary regression guard of this test
        assert active_ids.isdisjoint(done_ids), (
            f"active_tasks and done_tasks overlap: {active_ids & done_ids}"
        )
        assert active_ids.isdisjoint(cancelled_ids), (
            f"active_tasks and cancelled_tasks overlap: {active_ids & cancelled_ids}"
        )
        assert done_ids.isdisjoint(cancelled_ids), (
            f"done_tasks and cancelled_tasks overlap: {done_ids & cancelled_ids}"
        )

        # id=10 (status='stalled') must NOT appear in any list — it goes to other_count
        # (bucket-content checks are already covered by test_partitions_active_done_cancelled_and_other)
        all_listed_ids = active_ids | done_ids | cancelled_ids
        assert 10 not in all_listed_ids, (
            f"Task id=10 (status='stalled') should route to other_count only, "
            f"not appear in active/done/cancelled. Found in: {all_listed_ids}"
        )
        assert result.other_count >= 1, (
            f"Expected other_count >= 1 for the 'stalled' task, got {result.other_count}"
        )


class TestStatusPriorityIncludesDone:
    """Tests that _STATUS_PRIORITY includes 'done' and all expected keys."""

    def test_status_priority_includes_done(self):
        """_STATUS_PRIORITY must contain 'done': 4."""
        assert 'done' in _STATUS_PRIORITY, (
            "_STATUS_PRIORITY is missing 'done' key; task_filter is the source of truth"
        )
        assert _STATUS_PRIORITY['done'] == 4, (
            f"Expected _STATUS_PRIORITY['done'] == 4, got {_STATUS_PRIORITY['done']}"
        )

    def test_status_priority_has_all_expected_keys(self):
        """_STATUS_PRIORITY must contain all six status keys with correct priority values."""
        expected = {
            'in-progress': 0,
            'blocked': 1,
            'review': 2,
            'pending': 3,
            'done': 4,
            'deferred': 5,
        }
        for status, priority in expected.items():
            assert status in _STATUS_PRIORITY, (
                f"_STATUS_PRIORITY missing key '{status}'"
            )
            assert _STATUS_PRIORITY[status] == priority, (
                f"_STATUS_PRIORITY['{status}'] = {_STATUS_PRIORITY[status]}, expected {priority}"
            )

    def test_status_priority_includes_merge_deferred(self):
        """_STATUS_PRIORITY must contain 'merge-deferred': 6.

        merge-deferred is a non-terminal holding state for atomic-train members
        (PRD orchestrator-atomic-train-merge §9.2, task 1519). Priority 6 places
        it at the bottom of the active task list (below deferred=5).
        """
        assert 'merge-deferred' in _STATUS_PRIORITY, (
            "_STATUS_PRIORITY is missing 'merge-deferred' key; task_filter is the source of truth"
        )
        assert _STATUS_PRIORITY['merge-deferred'] == 6, (
            f"Expected _STATUS_PRIORITY['merge-deferred'] == 6, got {_STATUS_PRIORITY['merge-deferred']}"
        )


class TestFilterTaskTreeClassifiesMergeDeferred:
    """Tests that filter_task_tree correctly classifies 'merge-deferred' as active."""

    def test_filter_task_tree_classifies_merge_deferred_as_active(self):
        """filter_task_tree must count merge-deferred tasks as active (not other_count).

        merge-deferred is a non-terminal in-flight state — it should appear in
        active_tasks so operators and reconciliation prompts see holding-state
        members. Treating it as other_count would under-report in-flight work.
        PRD orchestrator-atomic-train-merge §9.2, task 1519.
        """
        task_tree = {
            'tasks': [
                {
                    'id': '42',
                    'status': 'merge-deferred',
                    'title': 'Atomic-train member holding',
                    'dependencies': [],
                }
            ]
        }
        result = filter_task_tree(task_tree)
        assert len(result.active_tasks) == 1, (
            f"Expected 1 active task for merge-deferred, got {len(result.active_tasks)}"
        )
        assert result.active_tasks[0]['id'] == '42'
        assert result.done_count == 0, (
            f"Expected done_count=0, got {result.done_count}"
        )
        assert result.cancelled_count == 0, (
            f"Expected cancelled_count=0, got {result.cancelled_count}"
        )
        assert result.other_count == 0, (
            f"Expected other_count=0 for merge-deferred (it is active), got {result.other_count}"
        )


class TestRenderTaskLineAndFormatTaskList:
    """Tests for _render_task_line and format_task_list helpers."""

    def test_render_task_line_basic(self):
        """_render_task_line produces '- [id] (status) title deps=[]' format."""
        task = {'id': 1, 'status': 'pending', 'title': 'X', 'dependencies': []}
        result = _render_task_line(task)
        assert result == '- [1] (pending) X deps=[]'

    def test_render_task_line_truncates_deps_over_5(self):
        """_render_task_line truncates deps to first 5 items with '...' suffix."""
        task = {'id': 2, 'status': 'in-progress', 'title': 'Y', 'dependencies': list(range(1, 9))}
        result = _render_task_line(task)
        assert result.endswith('deps=[1, 2, 3, 4, 5]...')

    def test_render_task_line_deps_none_normalized(self):
        """_render_task_line treats deps=None as empty list."""
        task = {'id': 3, 'status': 'review', 'title': 'Z', 'dependencies': None}
        result = _render_task_line(task)
        assert 'deps=[]' in result
        assert 'deps=None' not in result

    def test_render_task_line_missing_fields(self):
        """_render_task_line uses '?' defaults for missing id/status/title."""
        result = _render_task_line({})
        assert result == '- [?] (?) ? deps=[]'

    def test_format_task_list_empty_returns_no_tasks(self):
        """format_task_list([]) returns 'No tasks.'."""
        assert format_task_list([]) == 'No tasks.'

    def test_format_task_list_joins_rendered_lines(self):
        """format_task_list of 2 tasks returns their rendered lines joined by newline."""
        t1 = {'id': 1, 'status': 'pending', 'title': 'Alpha', 'dependencies': []}
        t2 = {'id': 2, 'status': 'done', 'title': 'Beta', 'dependencies': []}
        result = format_task_list([t1, t2])
        expected = _render_task_line(t1) + '\n' + _render_task_line(t2)
        assert result == expected

    def test_render_task_line_deps_truthy_non_list(self):
        """_render_task_line treats any truthy non-list deps value as empty list.

        When a task has 'dependencies' set to a truthy non-list value (int like 42,
        dict like {'a': 1}, string like 'bad'), the formatter must treat it as [].
        Bug: `task.get('dependencies') or []` passes truthy non-list values through,
        causing TypeError on deps[:5] for int/dict, or garbled output for string.
        """
        cases = [
            (42, 'int deps'),
            ({'a': 1}, 'dict deps'),
            ('bad', 'string deps'),
        ]
        for deps_value, label in cases:
            task = {'id': 1, 'status': 'pending', 'title': 'X', 'dependencies': deps_value}
            result = _render_task_line(task)
            assert 'deps=[]' in result, (
                f'Expected deps=[] for {label} ({deps_value!r}), got: {result!r}'
            )

    def test_format_task_list_filters_non_dict_items(self):
        """format_task_list skips non-dict elements and renders only valid task dicts.

        Cases:
        (a) mixed list [valid_dict, 42, None, 'bad', valid_dict2] renders only the two dicts;
        (b) all-non-dict [42, None, 'bad'] returns 'No tasks.';
        (c) empty [] still returns 'No tasks.' (regression guard).

        Bug: current code calls _render_task_line(t) unconditionally, which calls t.get()
        and crashes with AttributeError when t is not a dict.
        """
        t1 = {'id': 1, 'status': 'pending', 'title': 'Alpha', 'dependencies': []}
        t2 = {'id': 2, 'status': 'done', 'title': 'Beta', 'dependencies': []}

        # (a) mixed list — only valid dicts rendered
        result_a = format_task_list([t1, 42, None, 'bad', t2])
        expected_a = _render_task_line(t1) + '\n' + _render_task_line(t2)
        assert result_a == expected_a, (
            f'Expected only valid dicts rendered, got: {result_a!r}'
        )

        # (b) all non-dicts → 'No tasks.'
        result_b = format_task_list([42, None, 'bad'])
        assert result_b == 'No tasks.', (
            f"Expected 'No tasks.' for all-non-dict input, got: {result_b!r}"
        )

        # (c) empty list → 'No tasks.' (regression guard)
        result_c = format_task_list([])
        assert result_c == 'No tasks.', (
            f"Expected 'No tasks.' for empty input, got: {result_c!r}"
        )


class TestFormatCancelledSection:
    """Tests for the '### Recently Cancelled Tasks' section in format_filtered_task_tree."""

    def test_format_cancelled_tasks_section(self):
        """format_filtered_task_tree renders a '### Recently Cancelled Tasks' section
        when cancelled_tasks is non-empty, and updates the summary line accordingly.
        """
        tree = FilteredTaskTree(
            active_tasks=[_make_task(1, 'pending'), _make_task(2, 'in-progress')],
            cancelled_tasks=[_make_task(8, 'cancelled'), _make_task(9, 'cancelled')],
            done_count=3,
            cancelled_count=2,
            other_count=0,
            total_count=7,
        )
        output = format_filtered_task_tree(tree)

        # (a) Section header must be present
        assert '### Recently Cancelled Tasks' in output, (
            f'Expected "### Recently Cancelled Tasks" section in output, got:\n{output!r}'
        )

        # (b) Task lines for both cancelled tasks must appear
        assert '- [8] (cancelled)' in output, (
            f'Expected cancelled task line "- [8] (cancelled)" in output, got:\n{output!r}'
        )
        assert '- [9] (cancelled)' in output, (
            f'Expected cancelled task line "- [9] (cancelled)" in output, got:\n{output!r}'
        )

        # (c) Summary line omits 'cancelled' since they're now shown
        assert '3 done \u2014 omitted' in output, (
            f'Expected summary "3 done \u2014 omitted" in output, got:\n{output!r}'
        )

        # (d) Old summary line format must NOT appear when cancelled section is rendered.
        # Note: the header always contains '3 done, 2 cancelled' in the stats line; the
        # assertion checks for the full old summary string (with em dash) which is the
        # actual old format that must be replaced.
        assert '3 done, 2 cancelled \u2014 omitted' not in output, (
            f'Old summary line "3 done, 2 cancelled \u2014 omitted" must not appear when '
            f'cancelled section is rendered, got:\n{output!r}'
        )

    def test_format_cancelled_section_omitted_when_empty(self):
        """When cancelled_tasks=[] (empty), no cancelled section is rendered and
        the summary line retains the original 'N done, N cancelled — omitted' format.

        This guards backward compatibility: all existing budget tests have
        cancelled_tasks=[] and must not be affected by the conditional rendering.
        """
        tree = FilteredTaskTree(
            active_tasks=[_make_task(1, 'pending'), _make_task(2, 'in-progress')],
            # cancelled_tasks left as default empty list
            done_count=3,
            cancelled_count=5,
            other_count=0,
            total_count=10,
        )
        output = format_filtered_task_tree(tree)

        # (a) Section must be absent when cancelled_tasks is empty
        assert '### Recently Cancelled Tasks' not in output, (
            f'Section "### Recently Cancelled Tasks" must not appear when '
            f'cancelled_tasks=[], got:\n{output!r}'
        )

        # (b) Summary line retains original format (backward compatibility)
        assert '3 done, 5 cancelled \u2014 omitted' in output, (
            f'Expected original summary "3 done, 5 cancelled \u2014 omitted" '
            f'when cancelled_tasks=[], got:\n{output!r}'
        )

    def test_format_cancelled_section_budget_accounting(self):
        """When cancelled_tasks is non-empty and max_chars forces truncation, the
        cancelled section must survive intact and only active task lines are trimmed.

        The budget calculation subtracts len(cancelled_section) before computing
        available space for active task lines, so truncation never cuts the
        cancelled section.
        """
        active = [_make_task(i, 'pending', f'Task {i}') for i in range(1, 21)]
        cancelled = [
            _make_task(101, 'cancelled', 'Cancelled A'),
            _make_task(102, 'cancelled', 'Cancelled B'),
            _make_task(103, 'cancelled', 'Cancelled C'),
        ]
        tree = FilteredTaskTree(
            active_tasks=active,
            cancelled_tasks=cancelled,
            done_count=5,
            cancelled_count=3,
            other_count=0,
            total_count=28,
        )

        max_chars = 500  # Tight enough to force active-task truncation
        output = format_filtered_task_tree(tree, max_chars=max_chars)

        # (a) Output must honour the char budget
        assert len(output) <= max_chars, (
            f'Output length {len(output)} exceeds max_chars={max_chars}; '
            f'budget accounting with cancelled section is broken'
        )

        # (b) Cancelled section must survive budget truncation
        assert '### Recently Cancelled Tasks' in output, (
            f'Expected "### Recently Cancelled Tasks" to survive budget clamp, '
            f'got:\n{output!r}'
        )

        # (c) All 3 cancelled task ids must appear (cancelled section is never truncated)
        assert '- [101]' in output, f'Cancelled task 101 missing from output:\n{output!r}'
        assert '- [102]' in output, f'Cancelled task 102 missing from output:\n{output!r}'
        assert '- [103]' in output, f'Cancelled task 103 missing from output:\n{output!r}'

    def test_format_cancelled_section_ordering(self):
        """Sections must appear in order: header stats → active body → cancelled → summary.

        Verifies positional ordering via index() rather than just substring presence —
        a malformed concatenation order would pass presence-only assertions.
        """
        tree = FilteredTaskTree(
            active_tasks=[_make_task(1, 'pending'), _make_task(2, 'in-progress')],
            cancelled_tasks=[_make_task(8, 'cancelled'), _make_task(9, 'cancelled')],
            done_count=3,
            cancelled_count=2,
            other_count=0,
            total_count=7,
        )
        output = format_filtered_task_tree(tree)

        pos_active = output.index('- [1]')            # first active task line
        pos_cancelled_header = output.index('### Recently Cancelled Tasks')
        pos_summary = output.index('done \u2014 omitted')

        assert pos_active < pos_cancelled_header, (
            f'Active task lines (pos {pos_active}) must appear before '
            f'the cancelled section header (pos {pos_cancelled_header})'
        )
        assert pos_cancelled_header < pos_summary, (
            f'Cancelled section header (pos {pos_cancelled_header}) must appear '
            f'before the summary line (pos {pos_summary})'
        )

    def test_format_cancelled_section_large_accepted_overflow(self):
        """Documents accepted behavior: when the cancelled section alone fills the budget,
        the formatter returns header + cancelled_section + summary_line even if that
        exceeds max_chars.  With MAX_CANCELLED_TASKS_RETAINED capping the list,
        this degenerate case only occurs under unrealistically tight budgets.

        The fallback path `return header + cancelled_section + summary_line` (triggered
        when budget <= 0) is the documented safe exit — it never silently drops the
        cancelled section or the summary.
        """
        # Build the maximum retained cancelled tasks with long titles to ensure
        # the section is large enough to exhaust a tiny budget.
        cancelled = [
            _make_task(
                100 + i, 'cancelled',
                'A very long task title that consumes character budget space',
            )
            for i in range(MAX_CANCELLED_TASKS_RETAINED)
        ]
        tree = FilteredTaskTree(
            active_tasks=[_make_task(1, 'pending')],
            cancelled_tasks=cancelled,
            done_count=2,
            cancelled_count=MAX_CANCELLED_TASKS_RETAINED,
            other_count=0,
            total_count=MAX_CANCELLED_TASKS_RETAINED + 3,
        )

        # A budget far below the size of the cancelled section alone.
        tiny_budget = 50
        output = format_filtered_task_tree(tree, max_chars=tiny_budget)

        # The output exceeds tiny_budget — this is the documented accepted behavior.
        assert len(output) > tiny_budget, (
            f'Expected output ({len(output)} chars) to exceed tiny_budget={tiny_budget}; '
            f'the fallback path emits the full cancelled section regardless of budget'
        )
        # The formatter never silently drops the cancelled section or summary line.
        assert '### Recently Cancelled Tasks' in output
        assert 'done \u2014 omitted' in output


class TestIdKey:
    """Direct unit tests for the module-level id_key() helper."""

    def test_int_id_returns_int(self):
        """id_key returns the int value when 'id' is already an int."""
        assert id_key({'id': 42}) == 42

    def test_string_parseable_id_returns_int(self):
        """id_key converts a string-encoded integer to int."""
        assert id_key({'id': '42'}) == 42

    def test_non_parseable_string_returns_zero(self):
        """id_key returns 0 for a non-parseable string like 'abc'."""
        assert id_key({'id': 'abc'}) == 0

    def test_none_id_returns_zero(self):
        """id_key returns 0 when 'id' is explicitly None."""
        assert id_key({'id': None}) == 0

    def test_missingid_key_returns_zero(self):
        """id_key returns 0 when the 'id' key is absent from the dict."""
        assert id_key({}) == 0

    def test_float_id_is_truncated_to_int(self):
        """id_key returns the int truncation of a float (int(3.9) == 3)."""
        assert id_key({'id': 3.9}) == 3

    def test_dotted_id_returns_zero_not_parent_segment(self):
        """id_key returns 0 for dotted ids — the first-dot-segment rule is removed (DF-D).

        Post-DF-D: id_key only parses bare integers; dotted ids like '450.2' are
        non-parseable and fall back to 0 just like 'abc'.  This is a RED test
        against pre-DF-D code that returned 450 for '450.2'.
        """
        assert id_key({'id': '450.2'}) == 0, (
            "id_key({'id': '450.2'}) must return 0 after DF-D; "
            'dotted ids are no longer special-cased (first-dot-segment rule removed). '
            'Got non-zero — the dotted branch is still present.'
        )
        assert id_key({'id': '7'}) == 7, (
            "id_key({'id': '7'}) must still return 7 (bare-integer string parse path)."
        )
        assert id_key({'id': '450.2.1'}) == 0, (
            "id_key({'id': '450.2.1'}) must return 0 after DF-D (deep dotted id)."
        )


class TestSelectVisibleActive:
    """Unit tests for select_visible_active(tree, max_tasks, max_chars).

    Covers: empty input, under-cap, max_chars-clamp prefix, budget<=0, and
    cancelled-section budget accounting.  All five tests fail with ImportError
    until select_visible_active is defined in task_filter.py (step-2).
    """

    def _make_active_task(self, tid: int, title_len: int = 20) -> dict:
        """Build a task dict with title padded to title_len chars."""
        return {
            'id': tid,
            'title': f'T{tid}'.ljust(title_len, 'x'),
            'status': 'pending',
            'dependencies': [],
        }

    def test_empty_active_returns_empty_list(self):
        """empty tree.active_tasks -> []."""
        tree = FilteredTaskTree(
            active_tasks=[],
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=0,
        )
        result = select_visible_active(tree, max_tasks=50, max_chars=50_000)
        assert result == []

    def test_under_cap_returns_full_slice(self):
        """Under-cap with huge max_chars returns full active_tasks[:max_tasks]."""
        tasks = [self._make_active_task(i) for i in range(1, 6)]
        tree = FilteredTaskTree(
            active_tasks=tasks,
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=5,
        )
        result = select_visible_active(tree, max_tasks=50, max_chars=999_999)
        assert result == tasks

    def test_max_chars_clamp_returns_prefix_matching_formatter(self):
        """max_chars clamp fires -> prefix length matches format_filtered_task_tree output count.

        Uses format_filtered_task_tree as ground truth so the assertion stays
        accurate if the rendering format changes.
        """
        # 20 tasks each with a 400-char title to force the max_chars clamp.
        tasks = [self._make_active_task(i, title_len=400) for i in range(1, 21)]
        tree = FilteredTaskTree(
            active_tasks=tasks,
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=20,
        )
        max_chars = 4000

        # Ground-truth: count task lines rendered by format_filtered_task_tree.
        output = format_filtered_task_tree(tree, max_tasks=20, max_chars=max_chars)
        task_lines_in_output = [ln for ln in output.splitlines() if ln.startswith('- [')]
        expected_count = len(task_lines_in_output)

        # Verify the clamp actually fired — if not, the test is vacuous.
        assert expected_count < 20, (
            'Clamp did not fire (all 20 tasks visible); lower max_chars or raise title_len.'
        )

        result = select_visible_active(tree, max_tasks=20, max_chars=max_chars)

        assert len(result) == expected_count, (
            f'Expected {expected_count} tasks, got {len(result)}'
        )
        # Must be a strict prefix of the input slice.
        assert result == tasks[:expected_count], (
            'Result is not a prefix of the original task list'
        )

    def test_budget_zero_or_negative_returns_empty(self):
        """budget <= 0 (header+summary alone exceed max_chars) -> returns [].

        With 1 active task: header ~ 78 chars, summary_line ~ 29 chars.
        max_chars=100 makes budget = 100 - 78 - 29 = -7 <= 0.
        """
        tasks = [self._make_active_task(1)]
        tree = FilteredTaskTree(
            active_tasks=tasks,
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=1,
        )
        result = select_visible_active(tree, max_tasks=50, max_chars=100)
        assert result == [], (
            f'Expected [] when budget <= 0, got {result!r}'
        )

    def test_cancelled_section_reduces_active_budget(self):
        """Non-empty cancelled_tasks subtracts the cancelled-section length from the budget.

        Builds two trees with identical active tasks: one with five cancelled
        tasks and one without.  At a tight max_chars, the cancelled section
        consumes part of the budget so fewer active tasks fit.  Both results
        must match what format_filtered_task_tree renders for the same tree.
        """
        active_tasks = [self._make_active_task(i, title_len=200) for i in range(1, 11)]
        cancelled_tasks = [
            {'id': 100 + i, 'title': 'C' * 50, 'status': 'cancelled', 'dependencies': []}
            for i in range(1, 6)
        ]
        tree_with = FilteredTaskTree(
            active_tasks=active_tasks,
            cancelled_tasks=cancelled_tasks,
            done_count=0,
            cancelled_count=5,
            other_count=0,
            total_count=15,
        )
        tree_without = FilteredTaskTree(
            active_tasks=active_tasks,
            cancelled_tasks=[],
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=10,
        )
        max_chars = 2500

        # Ground-truth visible counts from the formatter.
        # For tree_with the cancelled section also contains '- [' lines — strip it
        # before counting so we only measure the active-task lines.
        out_with = format_filtered_task_tree(tree_with, max_tasks=10, max_chars=max_chars)
        out_without = format_filtered_task_tree(tree_without, max_tasks=10, max_chars=max_chars)
        active_part_with = out_with.split('\n### Recently Cancelled Tasks')[0]
        expected_with = len([ln for ln in active_part_with.splitlines() if ln.startswith('- [')])
        expected_without = len([ln for ln in out_without.splitlines() if ln.startswith('- [')])

        # Precondition: cancelled section must have reduced the active budget.
        assert expected_with < expected_without, (
            f'Precondition failed: cancelled section did not reduce active task count '
            f'({expected_with} vs {expected_without}). Adjust max_chars or title_len.'
        )

        result_with = select_visible_active(tree_with, max_tasks=10, max_chars=max_chars)
        result_without = select_visible_active(tree_without, max_tasks=10, max_chars=max_chars)

        assert len(result_with) == expected_with, (
            f'With cancelled section: expected {expected_with}, got {len(result_with)}'
        )
        assert len(result_without) == expected_without, (
            f'Without cancelled section: expected {expected_without}, got {len(result_without)}'
        )


# ── Regression: cycle 8df8bdcd title↔task_id contract (task 1379) ──────────
#
# Cycle 8df8bdcd: tasks 1355/1361/1369 appeared in Stage 1 output each
# carrying the NEXT task's title in the sorted completion sequence.
# Scenario data and parse helpers centralised in _fm_helpers.make_8df8_scenario /
# assert_id_title_pairing so all four suites share one source of truth.


class TestCompletionOrderVsIdOrderPreservesIdTitlePairing:
    """Stage 1 formatter: id↔title pairing survives completion-order ≠ id-order scenarios."""

    # Canonical 8df8bdcd scenario: int ids, status='done', completion order 1369→1355→1361
    _TASKS, _TITLE_BY_ID = make_8df8_scenario(id_type=int, status='done')
    # in-progress variant for the active-rendering path tests
    _TASKS_ACTIVE, _TITLE_BY_ID_ACTIVE = make_8df8_scenario(id_type=int, status='in-progress')

    def test_format_task_list_preserves_id_title_pairing(self):
        """format_task_list: every rendered line pairs each id with its OWN title.

        Uses the 8df8bdcd done-task set in completion order (1369→1355→1361).
        Anti-vacuity guard is bundled into assert_id_title_pairing (zero matches → fail).
        """
        rendered = format_task_list(list(self._TASKS))
        assert rendered != 'No tasks.', 'format_task_list returned empty output'
        assert_id_title_pairing(
            rendered, self._TITLE_BY_ID, kind='active',
            expected_ids={1369, 1355, 1361},
        )

    def test_format_filtered_task_tree_active_tasks_preserve_id_title_pairing(self):
        """format_filtered_task_tree: active task lines pair each id with its OWN title.

        Pins the REAL format_filtered_task_tree active-rendering path for the
        8df8bdcd scenario: tasks are fed as in-progress (active) so they are
        individually line-rendered, with completion order ≠ id order.
        Anti-vacuity guard is bundled into assert_id_title_pairing.
        """
        result = filter_task_tree({'tasks': list(self._TASKS_ACTIVE)})
        rendered = format_filtered_task_tree(result)
        assert_id_title_pairing(
            rendered, self._TITLE_BY_ID_ACTIVE, kind='active',
            expected_ids={1369, 1355, 1361},
        )

    def test_format_filtered_task_tree_done_tasks_appear_only_in_summary_count(self):
        """format_filtered_task_tree: done tasks appear ONLY in the summary count, never as task lines.

        Pins the true done-task contract: format_filtered_task_tree never
        line-renders done tasks individually. Done tasks contribute to the
        '{done_count} done … — omitted' summary line, and none of their titles
        or individual '[id] (' fragments appear in the rendered output.
        """
        tree = FilteredTaskTree(
            active_tasks=[],
            done_tasks=list(self._TASKS),
            done_count=3,
            cancelled_tasks=[],
            cancelled_count=0,
            other_count=0,
            total_count=3,
        )
        rendered = format_filtered_task_tree(tree)

        # No individual task lines should exist for done tasks
        task_line_pattern = re.compile(r'- \[\d+\] \(')
        assert not task_line_pattern.search(rendered), (
            f'format_filtered_task_tree rendered individual task lines for done tasks '
            f'(done tasks should only appear in the summary count).\n'
            f'Rendered output:\n{rendered}'
        )

        # None of the done-task titles should appear in the output
        for task in self._TASKS:
            assert task['title'] not in rendered, (
                f'Done task title {task["title"]!r} (id={task["id"]}) found in '
                f'format_filtered_task_tree output — should only appear in count summary.\n'
                f'Rendered output:\n{rendered}'
            )

    def test_stage1_data_block_id_title_pairs_internally_consistent(self):
        """Stage-1 data block: filter_task_tree→format_filtered_task_tree preserves id↔title.

        harness.py:434 builds the Stage-1-bound structured data block as:
            format_filtered_task_tree(filter_task_tree(tasks_data))
        This test reproduces that exact path for a MIXED active+done tree
        (some in-progress active, some done) and asserts that every rendered
        active-task line pairs each id with its OWN title — no neighbor bleed.

        This is the highest-signal deterministic assertion reachable with zero
        production change: it guards the actual Stage-1 input path against a
        reintroduction of the cycle-8df8bdcd title-swap bug.
        """
        # Mixed tree: 1369 and 1355 are active (in-progress); 1361 is done
        tasks_data = {
            'tasks': [
                {'id': 1369, 'title': 'Refactor event dispatch to async', 'status': 'in-progress', 'dependencies': []},
                {'id': 1355, 'title': 'Implement rate limiter middleware', 'status': 'in-progress', 'dependencies': []},
                {'id': 1361, 'title': 'Add retry logic for database connections', 'status': 'done', 'dependencies': []},
            ]
        }
        # Build the Stage-1 data block exactly as harness.py does
        filtered = filter_task_tree(tasks_data)
        rendered = format_filtered_task_tree(filtered)

        # Active tasks (1369, 1355) must each carry their OWN title in rendered lines
        active_title_by_id = {1369: 'Refactor event dispatch to async', 1355: 'Implement rate limiter middleware'}
        assert_id_title_pairing(
            rendered, active_title_by_id, kind='active',
            expected_ids={1369, 1355},
        )

        # Done task (1361) must NOT appear as an individual line — only in the summary count
        assert 'Add retry logic for database connections' not in rendered, (
            f'Done task 1361 title leaked into rendered Stage-1 block:\n{rendered}'
        )

    def test_filter_task_tree_done_tasks_preserve_id_title_pairing_after_nlargest(self):
        """filter_task_tree: each done_tasks entry carries its OWN title after heapq.nlargest reorder.

        Pins the regression locus at task_filter.py:235-242:
            heapq.nlargest(MAX_DONE_TASKS_RETAINED, enumerate(done),
                           key=lambda p: (id_key(p[1]), -p[0]))
        The 8df8bdcd scenario has completion order [1369, 1355, 1361].  nlargest
        reorders by id-descending to [1369, 1361, 1355].  This test asserts:
        1. Anti-vacuity: exactly 3 done_tasks are returned.
        2. Each entry's title == the title for THAT entry's id (no neighbor bleed).
        3. The resulting id order [1369, 1361, 1355] differs from completion order
           [1369, 1355, 1361], proving the nlargest reorder path is exercised.

        Expected to PASS on addition — production filter_task_tree is already correct
        and must NOT be modified (task 1403 is test-suite-only).
        """
        result = filter_task_tree({'tasks': list(self._TASKS)})

        # Anti-vacuity: all three done tasks returned
        assert len(result.done_tasks) == 3, (
            f'Expected 3 done_tasks, got {len(result.done_tasks)}: {result.done_tasks}'
        )

        # Each entry carries its OWN title (the cycle-8df8bdcd regression contract)
        title_by_id = {int(k): v for k, v in self._TITLE_BY_ID.items()}
        for entry in result.done_tasks:
            eid = int(entry['id'])
            assert entry['title'] == title_by_id[eid], (
                f'id {eid}: got title {entry["title"]!r}, expected {title_by_id[eid]!r}'
            )

        # nlargest reorders completion order [1369,1355,1361] → id-desc [1369,1361,1355]
        id_order = [int(t['id']) for t in result.done_tasks]
        assert id_order == [1369, 1361, 1355], (
            f'Expected id-desc order [1369,1361,1355], got {id_order}'
        )
        # Confirm this differs from completion order, proving the reorder path is exercised
        completion_order = [1369, 1355, 1361]
        assert id_order != completion_order, (
            'done_tasks id order matches completion order — nlargest reorder path not exercised'
        )


# ---------------------------------------------------------------------------
# Step 1: max_task_id field (RED tests)
# ---------------------------------------------------------------------------


class TestFilterTaskTreeMaxTaskId:
    """RED tests for FilteredTaskTree.max_task_id (step-1).

    max_task_id must be the global maximum TOP-LEVEL task id across the FULL
    input, independent of the active/done/cancelled render caps.
    """

    def test_max_task_id_populated_across_full_input_ignoring_render_caps(self):
        """max_task_id equals the global max id even when done/cancelled/active lists are capped.

        Build a tasks_data with:
          - MAX_DONE_TASKS_RETAINED + 5 done tasks (ids 1..35), caps at 30
          - MAX_CANCELLED_TASKS_RETAINED + 5 cancelled tasks (ids 36..55), caps at 15
          - MAX_ACTIVE_TASKS_RENDERED + 5 active (pending) tasks (ids 56..110)
          - One active task with id 4044 (the global max, bare integer)

        Expected: result.max_task_id == 4044 even though none of these tasks
        necessarily appear in the capped done_tasks/cancelled_tasks lists.
        """
        # Build done tasks: ids 1..MAX_DONE_TASKS_RETAINED+5 (capped at MAX_DONE_TASKS_RETAINED=30)
        done_tasks = [_make_task(i, 'done') for i in range(1, MAX_DONE_TASKS_RETAINED + 6)]

        # Build cancelled tasks: ids starting after done tasks
        # (capped at MAX_CANCELLED_TASKS_RETAINED=15)
        cancelled_start = MAX_DONE_TASKS_RETAINED + 6
        cancelled_tasks = [
            _make_task(cancelled_start + i, 'cancelled')
            for i in range(MAX_CANCELLED_TASKS_RETAINED + 5)
        ]

        # Build active tasks: MAX_ACTIVE_TASKS_RENDERED + 5 = 55 active tasks
        active_start = cancelled_start + MAX_CANCELLED_TASKS_RETAINED + 5
        active_tasks = [
            _make_task(active_start + i, 'pending')
            for i in range(MAX_ACTIVE_TASKS_RENDERED + 5)
        ]

        # Add the highest-id task: id=4044 as active (bare integer, no dotted ids post-DF-D)
        high_id_task = _make_task(4044, 'pending', 'High ID task')
        active_tasks.append(high_id_task)

        all_tasks = done_tasks + cancelled_tasks + active_tasks
        tasks_data = {'tasks': all_tasks}

        result = filter_task_tree(tasks_data)

        # The done/cancelled lists are capped — verify caps are active
        assert len(result.done_tasks) == MAX_DONE_TASKS_RETAINED, (
            f'done_tasks should be capped at {MAX_DONE_TASKS_RETAINED}'
        )
        assert len(result.cancelled_tasks) == MAX_CANCELLED_TASKS_RETAINED, (
            f'cancelled_tasks should be capped at {MAX_CANCELLED_TASKS_RETAINED}'
        )

        # max_task_id must equal 4044 — the global max across the FULL input
        assert result.max_task_id == 4044, (
            f'max_task_id should be 4044 (global max), got {result.max_task_id}'
        )

    def test_max_task_id_zero_for_empty_input(self):
        """filter_task_tree returns max_task_id == 0 for empty/non-dict inputs."""
        # Empty dict (no 'tasks' key)
        result = filter_task_tree({})
        assert result.max_task_id == 0, (
            f'Empty dict input: expected max_task_id==0, got {result.max_task_id}'
        )

        # tasks is an empty list
        result = filter_task_tree({'tasks': []})
        assert result.max_task_id == 0, (
            f'Empty tasks list: expected max_task_id==0, got {result.max_task_id}'
        )

        # Non-dict input
        result = filter_task_tree(None)
        assert result.max_task_id == 0, (
            f'None input: expected max_task_id==0, got {result.max_task_id}'
        )

        result = filter_task_tree('bad')
        assert result.max_task_id == 0, (
            f'String input: expected max_task_id==0, got {result.max_task_id}'
        )

        result = filter_task_tree([{'id': 1, 'status': 'pending'}])
        assert result.max_task_id == 0, (
            f'List input: expected max_task_id==0, got {result.max_task_id}'
        )


# ---------------------------------------------------------------------------
# Step 3: format_filtered_task_tree header includes 'highest task id' (RED)
# ---------------------------------------------------------------------------


class TestFormatHeaderHighestTaskId:
    """RED tests for 'highest task id' token in format_filtered_task_tree header (step-3).

    The header must render an authoritative 'highest task id: N' token derived
    from tree.max_task_id, even when max_task_id does NOT appear in any rendered
    active task line (e.g. because it belongs to a capped or done task).
    """

    def test_header_includes_highest_task_id_token(self):
        """format_filtered_task_tree header includes 'highest task id: N' from tree.max_task_id.

        Build a tree where max_task_id=4044 but only small-id active tasks are
        present (id=1..3), so 4044 never appears as a rendered task line.
        Assert the header contains 'highest task id: 4044'.
        """
        active = [_make_task(i, 'pending') for i in range(1, 4)]
        tree = FilteredTaskTree(
            active_tasks=active,
            done_count=100,
            cancelled_count=10,
            other_count=0,
            total_count=113,
            max_task_id=4044,
        )
        output = format_filtered_task_tree(tree)

        assert 'highest task id: 4044' in output, (
            f'Header must contain "highest task id: 4044" (from tree.max_task_id=4044). '
            f'Got:\n{output!r}'
        )

    def test_header_highest_task_id_equals_max_task_id_not_max_rendered(self):
        """Rendered highest task id equals tree.max_task_id, not the max id among rendered lines.

        Set max_task_id=4044 but the ONLY rendered active task has id=10.
        The header must show 'highest task id: 4044', not 'highest task id: 10'.
        """
        tree = FilteredTaskTree(
            active_tasks=[_make_task(10, 'pending')],
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=1,
            max_task_id=4044,
        )
        output = format_filtered_task_tree(tree)

        assert 'highest task id: 4044' in output, (
            f'Expected "highest task id: 4044" (from max_task_id), not the rendered max id. '
            f'Got:\n{output!r}'
        )
        assert 'highest task id: 10' not in output, (
            f'Header must not show the rendered max id (10) as highest task id. '
            f'Got:\n{output!r}'
        )

    def test_header_highest_task_id_zero_when_max_task_id_zero(self):
        """When tree.max_task_id=0 (empty input), header shows 'highest task id: 0'."""
        tree = FilteredTaskTree(
            active_tasks=[],
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=0,
            max_task_id=0,
        )
        output = format_filtered_task_tree(tree)

        assert 'highest task id: 0' in output, (
            f'Expected "highest task id: 0" for empty tree. Got:\n{output!r}'
        )

    def test_existing_header_count_fields_still_present(self):
        """Existing count fields (active shown, done, cancelled, total) are preserved alongside highest task id."""
        active = [_make_task(i, 'pending') for i in range(1, 4)]
        tree = FilteredTaskTree(
            active_tasks=active,
            done_count=7,
            cancelled_count=2,
            other_count=1,
            total_count=13,
            max_task_id=500,
        )
        output = format_filtered_task_tree(tree)

        # Existing fields must still be present
        assert '3 active shown' in output, (
            f'Expected "3 active shown" in header. Got:\n{output!r}'
        )
        assert '7 done' in output, f'Expected "7 done" in header. Got:\n{output!r}'
        assert '2 cancelled' in output, f'Expected "2 cancelled" in header. Got:\n{output!r}'
        assert '13 total' in output, f'Expected "13 total" in header. Got:\n{output!r}'

        # New field must also be present
        assert 'highest task id: 500' in output, (
            f'Expected "highest task id: 500" in header. Got:\n{output!r}'
        )


# ---------------------------------------------------------------------------
# Step 5: detect_census_inconsistency (RED tests)
# ---------------------------------------------------------------------------


class TestDetectCensusInconsistency:
    """RED tests for detect_census_inconsistency(max_task_id, referenced_ids) (step-5).

    Returns sorted, deduplicated list of ids that STRICTLY exceed max_task_id.
    """

    def test_returns_ids_exceeding_max_task_id(self):
        """detect_census_inconsistency returns sorted list of ids strictly > max_task_id.

        With max_task_id=1515 and referenced ids [3438, '4026', '12', '4044.1', 'x']:
        - 3438 > 1515 → included
        - '4026' → 4026 > 1515 → included
        - '12' → 12 ≤ 1515 → excluded
        - '4044.1' → first-segment 4044 > 1515 → included
        - 'x' → unparseable → silently ignored

        Expected: [3438, 4026, 4044] (sorted ascending, deduplicated)
        """
        result = detect_census_inconsistency(1515, [3438, '4026', '12', '4044.1', 'x'])
        assert result == [3438, 4026, 4044], (
            f'Expected [3438, 4026, 4044], got {result}'
        )

    def test_returns_empty_when_no_ids_exceed_max(self):
        """Returns [] when all referenced ids are <= max_task_id."""
        result = detect_census_inconsistency(5000, [1, 100, '2000', '5000'])
        assert result == [], (
            f'Expected [], got {result}'
        )

    def test_strictly_exceeds_not_equal(self):
        """IDs equal to max_task_id are NOT returned (strictly greater)."""
        result = detect_census_inconsistency(1515, [1515, 1516, 1514])
        assert result == [1516], (
            f'Expected [1516] (1515 excluded as equal, 1514 excluded as lesser), got {result}'
        )

    def test_deduplicates_repeated_ids(self):
        """Duplicate ids in referenced_ids appear only once in the result."""
        result = detect_census_inconsistency(100, [200, 200, 300, 200])
        assert result == [200, 300], (
            f'Expected [200, 300] (deduplicated), got {result}'
        )

    def test_parses_dotted_ids_via_first_segment_rule(self):
        """Dotted subtask ids use the first dot-segment as the int value."""
        # '4044.2' → first segment 4044; '500.1.1' → first segment 500
        result = detect_census_inconsistency(1000, ['4044.2', '500.1.1', '999.9'])
        # 4044 > 1000 → included; 500 ≤ 1000 → excluded; 999 ≤ 1000 → excluded
        assert result == [4044], (
            f'Expected [4044], got {result}'
        )

    def test_silently_ignores_unparseable_entries(self):
        """Non-parseable entries (non-int, non-dotted-int) are silently ignored."""
        result = detect_census_inconsistency(100, ['x', None, {}, [], 'abc', '3000'])
        # '3000' → 3000 > 100; others → ignored
        assert result == [3000], (
            f'Expected [3000] (only parseable id 3000 exceeds 100), got {result}'
        )

    def test_returns_sorted_ascending(self):
        """Result is always sorted ascending regardless of input order."""
        result = detect_census_inconsistency(0, [300, 100, 200, 50, 1])
        assert result == [1, 50, 100, 200, 300], (
            f'Expected [1, 50, 100, 200, 300] (ascending), got {result}'
        )

    def test_empty_referenced_ids_returns_empty(self):
        """Empty referenced_ids returns []."""
        result = detect_census_inconsistency(1000, [])
        assert result == [], (
            f'Expected [] for empty referenced_ids, got {result}'
        )


# ---------------------------------------------------------------------------
# COUNT_SNAPSHOT_RE / is_count_snapshot / strip_snapshot_lines  (task 1547)
# ---------------------------------------------------------------------------


class TestCountSnapshotPrimitives:
    """Tests for COUNT_SNAPSHOT_RE, is_count_snapshot, and strip_snapshot_lines.

    These primitives detect and strip lines that contain count-snapshot text
    (e.g. '1505 done / 148 cancelled tasks') from reconciliation payloads.
    """

    # ------------------------------------------------------------------ #
    # is_count_snapshot — positive fixtures
    # ------------------------------------------------------------------ #

    def test_positive_full_status_snapshot(self):
        """Full status snapshot (pending / in-progress / blocked / deferred / done / cancelled)
        must be detected as a count-snapshot.
        """
        from fused_memory.reconciliation.task_filter import is_count_snapshot

        text = (
            'As of 2026-05-28, project reify has 2 pending / 2 in-progress / '
            '0 blocked / 1 deferred / 1505 done / 148 cancelled tasks'
        )
        assert is_count_snapshot(text) is True, (
            f'Expected is_count_snapshot to return True for full-status snapshot, got False.\n'
            f'text={text!r}'
        )

    def test_positive_partial_snapshot_done_cancelled_total(self):
        """Partial snapshot (done, cancelled, total) must be detected."""
        from fused_memory.reconciliation.task_filter import is_count_snapshot

        text = '...3355 done, 290 cancelled, 3358 total...'
        assert is_count_snapshot(text) is True, (
            f'Expected is_count_snapshot to return True for partial snapshot, got False.\n'
            f'text={text!r}'
        )

    # ------------------------------------------------------------------ #
    # is_count_snapshot — negative fixtures
    # ------------------------------------------------------------------ #

    def test_negative_single_done_mention(self):
        """Single incidental 'done' mention must NOT be detected as a snapshot."""
        from fused_memory.reconciliation.task_filter import is_count_snapshot

        text = 'Task 42 done via commit abc'
        assert is_count_snapshot(text) is False, (
            f'Expected is_count_snapshot to return False for single done mention, got True.\n'
            f'text={text!r}'
        )

    def test_negative_legitimate_temporal_fact(self):
        """Legitimate temporal fact mentioning done in passing must NOT be detected."""
        from fused_memory.reconciliation.task_filter import is_count_snapshot

        text = 'Decision: mark the rollout done once QA signs off'
        assert is_count_snapshot(text) is False, (
            f'Expected is_count_snapshot to return False for legitimate temporal fact, got True.\n'
            f'text={text!r}'
        )

    # ------------------------------------------------------------------ #
    # COUNT_SNAPSHOT_RE — basic contract
    # ------------------------------------------------------------------ #

    def test_count_snapshot_re_case_insensitive(self):
        """is_count_snapshot must match uppercase, lowercase, and mixed-case status words.

        Behavioral assertion replacing the former implementation-detail check on
        COUNT_SNAPSHOT_RE.flags.  Pins the case-insensitivity contract via the
        public function rather than the compiled regex object's attributes.
        """
        from fused_memory.reconciliation.task_filter import is_count_snapshot

        assert is_count_snapshot('1505 DONE / 148 CANCELLED tasks') is True, (
            'is_count_snapshot must match uppercase status words (DONE, CANCELLED)'
        )
        assert is_count_snapshot('1505 done / 148 cancelled tasks') is True, (
            'is_count_snapshot must match lowercase status words (done, cancelled)'
        )
        assert is_count_snapshot('1505 Done / 148 Cancelled tasks') is True, (
            'is_count_snapshot must match mixed-case status words (Done, Cancelled)'
        )

    # ------------------------------------------------------------------ #
    # strip_snapshot_lines — removes only the snapshot line
    # ------------------------------------------------------------------ #

    def test_strip_removes_snapshot_line_preserves_benign(self):
        """strip_snapshot_lines drops the snapshot line and keeps benign lines verbatim."""
        from fused_memory.reconciliation.task_filter import strip_snapshot_lines

        block = (
            'Entity summary line A\n'
            'As of 2026-05-28, project reify has 1505 done / 148 cancelled tasks\n'
            'Entity summary line B'
        )
        result_text, count = strip_snapshot_lines(block)

        assert '1505 done' not in result_text, (
            f'Snapshot line must be stripped; got result_text={result_text!r}'
        )
        assert 'Entity summary line A' in result_text, (
            f'"Entity summary line A" must be preserved; got result_text={result_text!r}'
        )
        assert 'Entity summary line B' in result_text, (
            f'"Entity summary line B" must be preserved; got result_text={result_text!r}'
        )
        # Verify order is preserved
        pos_a = result_text.index('Entity summary line A')
        pos_b = result_text.index('Entity summary line B')
        assert pos_a < pos_b, (
            f'"Entity summary line A" must appear before "Entity summary line B" in result; '
            f'got result_text={result_text!r}'
        )
        assert count == 1, (
            f'Expected count=1 (one snapshot line stripped), got count={count}'
        )

    def test_strip_returns_unchanged_text_and_zero_when_no_snapshot(self):
        """strip_snapshot_lines returns (unchanged_text, 0) when no snapshot lines are present."""
        from fused_memory.reconciliation.task_filter import strip_snapshot_lines

        block = 'Benign line one\nBenign line two\nNo snapshots here'
        result_text, count = strip_snapshot_lines(block)

        assert result_text == block, (
            f'Text must be unchanged when no snapshot lines present; '
            f'got result_text={result_text!r}'
        )
        assert count == 0, (
            f'Expected count=0 when no snapshot lines stripped, got count={count}'
        )

    # ------------------------------------------------------------------ #
    # is_count_snapshot — negative fixture: incidental two-token sentences
    # ------------------------------------------------------------------ #

    def test_negative_incidental_two_token_no_separator(self):
        """Sentences with two digit+status tokens but no ',' or '/' delimiter must NOT
        be detected as count-snapshots.

        Pins the regex delimiter requirement as intentional: COUNT_SNAPSHOT_RE requires
        at least one ',' or '/' between the two count+status tokens so that natural-
        language sentences like '2 review comments and 3 pending follow-ups' are not
        stripped from context-item or episode payloads.
        """
        from fused_memory.reconciliation.task_filter import is_count_snapshot

        assert is_count_snapshot('2 review comments and 3 pending follow-ups') is False, (
            'Incidental two-token sentence without , or / separator must NOT be '
            'classified as a count-snapshot; got True. '
            'The regex delimiter requirement may have been removed.'
        )
        assert is_count_snapshot('1 blocked ticket, please keep 2 pending items') is True, (
            'Sentence with , separator and two tokens must still be detected; '
            'regression check for the delimiter requirement.'
        )
