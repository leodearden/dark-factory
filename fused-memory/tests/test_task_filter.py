"""Tests for reconciliation/task_filter.py — FilteredTaskTree, filter_task_tree, format_filtered_task_tree."""

from __future__ import annotations

import re

from fused_memory.reconciliation.task_filter import (
    _STATUS_PRIORITY,
    FilteredTaskTree,
    _render_task_line,
    filter_task_tree,
    format_filtered_task_tree,
    format_task_list,
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
        """Regression: format_filtered_task_tree must honour max_chars and emit the exact
        max_tasks-cap header phrase when active tasks exceed max_tasks.

        With 500 active tasks and the default max_tasks=50 cap, 450 tasks are omitted.
        The header must contain the exact phrase '450 more active omitted by max_tasks cap'
        (the format emitted at task_filter.py line 232 when omitted_active > 0) and the
        total output must not exceed max_chars (default 50,000 chars).
        """
        # 500 active tasks with plausible-length titles
        active = [
            _make_task(i, 'pending', f'This is a moderately long title for task number {i} in the queue')
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

        # Header must contain the exact max_tasks-cap omission phrase
        assert '450 more active omitted by max_tasks cap' in output

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
        """Output length must not exceed max_chars regardless of truncation notice length.

        The magic -50 reserve fails when trimmed_count has 7+ digits (≥ 1,000,000).
        The actual notice '\n... and NNNNNNN more active (truncated for budget)\n' is
        52+ chars, exceeding the 50-char reserve by 1+, causing an off-by-one violation.

        With 1,000,001 tasks all having title='T', budget=25, exactly 1 task line fits
        (used=budget). trimmed_count = 1,000,000, notice = 52 chars > 50 reserve.
        len(result) = 195 > max_chars=194. Bug confirmed.

        Uses repeated-reference trick ([same_dict]*N) to keep memory at ~8 MB instead
        of creating N full task dicts (~200 MB).
        """
        # Compute max_chars so that exactly 1 task line fits and used == budget exactly.
        # Task line for title='T', id=1: "- [1] (pending) T deps=[]" = 24 chars.
        # Loop condition: used + len(line) + 1 ≤ budget.
        # For 1 line to fit: 0 + 24 + 1 = 25 ≤ budget → budget ≥ 25.
        # For 2nd line to not fit: 25 + 24 + 1 = 50 > budget → budget < 50. So budget = 25.
        #
        # With 1_000_001 tasks shown, header is:
        # "### Active Task Tree\n(1000001 active shown, 0 done, 0 cancelled, 0 other, 1000001 total)\n"
        # = 90 chars. summary_line = "0 done, 0 cancelled — omitted" = 29 chars.
        # budget = max_chars - 90 - 29 - 50 = max_chars - 169.
        # For budget = 25: max_chars = 25 + 169 = 194.
        #
        # trimmed_count = 1_000_001 - 1 = 1_000_000 (7 digits).
        # notice = "\n... and 1000000 more active (truncated for budget)\n" = 52 chars.
        # len(result) = 90 + (25-1) + 52 + 29 = 195 > 194 = max_chars. BUG!

        single_task = {'id': 1, 'title': 'T', 'status': 'pending', 'dependencies': []}
        n = 1_000_001
        # Repeated-reference trick: list of n pointers to same dict — uses ~8 MB, not ~200 MB.
        active_large = [single_task] * n
        tree_large = FilteredTaskTree(
            active_tasks=active_large,
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=n,
        )

        max_chars = 194  # Precisely computed to make used == budget == 25, notice 52 chars
        output = format_filtered_task_tree(tree_large, max_tasks=n, max_chars=max_chars)

        # With the magic -50 reserve, len(output) = 195 > 194 = max_chars.
        assert len(output) <= max_chars, (
            f'Output length {len(output)} exceeds max_chars={max_chars}; '
            f'bug: magic -50 reserve is insufficient when trimmed_count has 7+ digits '
            f'(notice is 52 chars, not ≤ 50)'
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
        assert 'deps=[1, 2, 3, 4, 5]...' in result

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
