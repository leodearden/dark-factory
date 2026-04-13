"""Tests for reconciliation/task_filter.py — FilteredTaskTree, filter_task_tree, format_filtered_task_tree."""

from __future__ import annotations

import re

from fused_memory.reconciliation.task_filter import (
    _STATUS_PRIORITY,
    MAX_CANCELLED_TASKS_RETAINED,
    FilteredTaskTree,
    _flatten_with_subtasks,
    _id_key,
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

    def test_nested_subtasks_are_included_in_filter(self):
        """filter_task_tree walks subtasks nested under parent tasks.

        Parent (id='450', in-progress) has two subtasks:
          - id=2 (pending) → active
          - id=3 (done)    → done

        Expects: 2 active tasks (parent + pending subtask), done_count=1,
        total_count=3.
        """
        tasks_data = {
            'tasks': [
                {
                    'id': '450',
                    'title': 'Parent Task',
                    'status': 'in-progress',
                    'dependencies': [],
                    'subtasks': [
                        {'id': 2, 'title': 'Sub active', 'status': 'pending', 'dependencies': []},
                        {'id': 3, 'title': 'Sub done', 'status': 'done', 'dependencies': []},
                    ],
                }
            ]
        }
        result = filter_task_tree(tasks_data)

        assert result.total_count == 3
        assert len(result.active_tasks) == 2
        assert result.done_count == 1

        active_ids = {str(t['id']) for t in result.active_tasks}
        assert '450' in active_ids  # parent
        # Subtask bare-int id=2 should be qualified as '450.2'
        assert '450.2' in active_ids


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
        max_tasks = 5
        max_chars = 240
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
        # Task line for title='T', id=1: "- [1] (pending) T deps=[]" = 25 chars.
        # With N=10_000, max_tasks=N, header is:
        # "### Active Task Tree\n(10000 active shown, 0 done, 0 cancelled, 0 other, 10000 total)\n"
        # = 85 chars. summary_line = "0 done, 0 cancelled — omitted" = 29 chars.
        # budget = max_chars - 85 - 29 = max_chars - 114.
        # For max_chars=500: budget=386. Each line costs 26 chars (25 + newline separator).
        # 14 lines fit (14×26=364 ≤ 386, 15×26=390 > 386). trimmed=9986, notice=49 chars.
        # Initial result = 85 + (14×25+13) + 49 + 29 = 85+363+49+29 = 526 > 500.
        # Lazy loop pops 1 line → 13 lines, trimmed=9987, result=500 ≤ 500. Loop exits.

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

        # Stub lines are 1 char each. With N=1_100_000, max_tasks=N, max_chars=500:
        # header = "### Active Task Tree\n(1100000 active shown, 0 done, 0 cancelled, 0 other, 1100000 total)\n"
        #        = 89 chars.
        # summary_line = "0 done, 0 cancelled — omitted" = 29 chars.
        # budget = 500 - 89 - 29 = 382. Each stub line costs 2 chars (1 char + newline sep).
        # initial kept = floor(383/2) = 191. trimmed_count = 1_100_000 - 191 = 1_099_809 (7 digits).
        # Notice = "... and 1099809 more active (truncated for budget)" = 52 chars.
        # Initial result = 89 + 381 + 52 + 29 = 551 > 500. Lazy loop fires, pops ~26 lines.
        # After ~26 pops: kept=165, trimmed=1_099_835 (still 7 digits), result=499 ≤ 500. ✓

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
        """Non-int ids fall back to _id_key=0 and sort last in descending order; stable sort preserves their mutual order."""
        tasks_data = {
            'tasks': [
                # done: two non-int ids ('abc' then 'def') interleaved with ints;
                # no literal id=0 to avoid sort-stability ambiguity with the fallback key.
                {'id': 10, 'title': 'Done 10', 'status': 'done', 'dependencies': []},
                {'id': 'abc', 'title': 'Done abc', 'status': 'done', 'dependencies': []},
                {'id': 5, 'title': 'Done 5', 'status': 'done', 'dependencies': []},
                {'id': 3, 'title': 'Done 3', 'status': 'done', 'dependencies': []},
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

        # 'abc' and 'def' both have _id_key=0 (int() fallback), so they sort last after
        # all int ids (10 > 5 > 3 > 0). Stable sort preserves their input order: 'abc' before 'def'.
        assert done_ids == [10, 5, 3, 'abc', 'def'], (
            f"Expected done_tasks id order [10, 5, 3, 'abc', 'def'] — non-int ids 'abc' and 'def' "
            f"both have _id_key=0 via the int() fallback, sort last (0 < 3 < 5 < 10 descending), "
            f"and preserve input order relative to each other (stable sort). Got: {done_ids}"
        )

        # 'xyz' has _id_key=0 (int() fallback), so it sorts last after 7, 2 (both > 0)
        assert cancelled_ids == [7, 2, 'xyz'], (
            f"Expected cancelled_tasks id order [7, 2, 'xyz'] — non-int 'xyz' has _id_key=0 "
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
    """Direct unit tests for the module-level _id_key() helper."""

    def test_int_id_returns_int(self):
        """_id_key returns the int value when 'id' is already an int."""
        assert _id_key({'id': 42}) == 42

    def test_string_parseable_id_returns_int(self):
        """_id_key converts a string-encoded integer to int."""
        assert _id_key({'id': '42'}) == 42

    def test_non_parseable_string_returns_zero(self):
        """_id_key returns 0 for a non-parseable string like 'abc'."""
        assert _id_key({'id': 'abc'}) == 0

    def test_none_id_returns_zero(self):
        """_id_key returns 0 when 'id' is explicitly None."""
        assert _id_key({'id': None}) == 0

    def test_missing_id_key_returns_zero(self):
        """_id_key returns 0 when the 'id' key is absent from the dict."""
        assert _id_key({}) == 0

    def test_float_id_is_truncated_to_int(self):
        """_id_key returns the int truncation of a float (int(3.9) == 3)."""
        assert _id_key({'id': 3.9}) == 3

    def test_dotted_subtask_id_returns_parent_component(self):
        """_id_key returns the parent (first dot-segment) as int for '450.2'."""
        assert _id_key({'id': '450.2'}) == 450

    def test_deep_dotted_id_returns_parent_component(self):
        """_id_key returns the parent (first dot-segment) as int for '450.2.1'."""
        assert _id_key({'id': '450.2.1'}) == 450


class TestFlattenWithSubtasks:
    """Direct unit tests for _flatten_with_subtasks()."""

    def test_bare_subtask_id_is_qualified(self):
        """Bare-integer subtask id=2 under parent id='450' becomes '450.2'."""
        raw = [
            {
                'id': '450',
                'title': 'Parent',
                'status': 'in-progress',
                'dependencies': [],
                'subtasks': [
                    {'id': 2, 'title': 'Sub', 'status': 'pending', 'dependencies': []},
                ],
            }
        ]
        flat = _flatten_with_subtasks(raw)
        ids = [str(t['id']) for t in flat]
        assert '450' in ids
        assert '450.2' in ids

    def test_already_qualified_id_is_not_double_qualified(self):
        """Subtask with id='450.2' (already dotted) stays '450.2', not '450.450.2'."""
        raw = [
            {
                'id': '450',
                'title': 'Parent',
                'status': 'in-progress',
                'dependencies': [],
                'subtasks': [
                    {'id': '450.2', 'title': 'Already qualified', 'status': 'pending', 'dependencies': []},
                ],
            }
        ]
        flat = _flatten_with_subtasks(raw)
        ids = [str(t['id']) for t in flat]
        assert '450.2' in ids
        assert '450.450.2' not in ids

    def test_no_subtasks_returns_flat_list(self):
        """Tasks without 'subtasks' key pass through unchanged."""
        raw = [
            {'id': 1, 'title': 'A', 'status': 'pending', 'dependencies': []},
            {'id': 2, 'title': 'B', 'status': 'done', 'dependencies': []},
        ]
        flat = _flatten_with_subtasks(raw)
        assert len(flat) == 2
        assert flat[0]['id'] == 1
        assert flat[1]['id'] == 2

    def test_original_subtask_dict_not_mutated(self):
        """_flatten_with_subtasks creates a shallow copy — original subtask dict is unchanged."""
        subtask = {'id': 3, 'title': 'Sub', 'status': 'pending', 'dependencies': []}
        raw = [
            {
                'id': '100',
                'title': 'Parent',
                'status': 'in-progress',
                'dependencies': [],
                'subtasks': [subtask],
            }
        ]
        _flatten_with_subtasks(raw)
        # Original must still have bare int id, not '100.3'
        assert subtask['id'] == 3

    def test_recursive_subtask_nesting(self):
        """_flatten_with_subtasks handles multi-level nesting.

        parent id='100'
          subtask id=1  → becomes '100.1'
            sub-subtask id=1 → becomes '100.1.1'

        All three should appear in flattened output with correct statuses.
        """
        raw = [
            {
                'id': '100',
                'title': 'Parent',
                'status': 'in-progress',
                'dependencies': [],
                'subtasks': [
                    {
                        'id': 1,
                        'title': 'Child',
                        'status': 'pending',
                        'dependencies': [],
                        'subtasks': [
                            {
                                'id': 1,
                                'title': 'Grandchild',
                                'status': 'done',
                                'dependencies': [],
                            }
                        ],
                    }
                ],
            }
        ]
        flat = _flatten_with_subtasks(raw)
        ids = [str(t['id']) for t in flat]
        assert '100' in ids
        assert '100.1' in ids
        assert '100.1.1' in ids
        assert len(flat) == 3

        by_id = {str(t['id']): t for t in flat}
        assert by_id['100']['status'] == 'in-progress'
        assert by_id['100.1']['status'] == 'pending'
        assert by_id['100.1.1']['status'] == 'done'


class TestSortOrderWithSubtasks:
    """Sort order tests for filter_task_tree when subtasks are present."""

    def test_sort_order_with_mixed_parent_and_subtask_ids(self):
        """Tasks with mixed parent and subtask IDs sort by _STATUS_PRIORITY then -_id_key.

        All tasks are 'pending' (same priority), so sort is purely by -_id_key:
          451 → _id_key=451 → sort position 0  (highest, first shown)
          '450'  → _id_key=450 → sort position 1
          '450.2' → _id_key=450 → sort position 2 (stable: same as parent)
          '450.1' → _id_key=450 → sort position 3 (stable: same as parent)

        With Python's stable sort, tasks sharing _id_key=450 preserve input order
        relative to each other: 450, 450.2, 450.1 (as given in input).
        """
        tasks_data = {
            'tasks': [
                {'id': '450', 'title': 'Parent 450', 'status': 'pending', 'dependencies': [],
                 'subtasks': [
                     {'id': '450.2', 'title': 'Sub 450.2', 'status': 'pending', 'dependencies': []},
                     {'id': '450.1', 'title': 'Sub 450.1', 'status': 'pending', 'dependencies': []},
                 ]},
                {'id': 451, 'title': 'Task 451', 'status': 'pending', 'dependencies': []},
            ]
        }
        result = filter_task_tree(tasks_data)

        ids = [str(t['id']) for t in result.active_tasks]
        # 451 must come before 450 group (higher _id_key → negated → lower sort value)
        assert ids.index('451') < ids.index('450')
        assert ids.index('451') < ids.index('450.2')
        assert ids.index('451') < ids.index('450.1')
        assert len(ids) == 4
