"""Shared utility module for filtering and formatting the Taskmaster task tree.

Extracts active tasks from the full raw get_tasks response, partitions them by
status, sorts by priority, and provides a budget-capped formatter.

Design decision: active status set = {pending, in-progress, blocked, deferred, review}.
The task description says 'pending, in-progress, blocked, deferred' explicitly, but
existing code (_select_proactive_sample, old Stage 2 filter) treats 'review' as active.
Excluding it would regress proactive-sampling tests.  The task's intent is
'exclude done/cancelled', so widening to 'not done/cancelled' preserves that
intent without regressions. (ref: task 455)
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Status constants
# --------------------------------------------------------------------------- #

ACTIVE_TASK_STATUSES: frozenset[str] = frozenset({
    'pending',
    'in-progress',
    'blocked',
    'deferred',
    'review',
})

INACTIVE_TASK_STATUSES: frozenset[str] = frozenset({
    'done',
    'cancelled',
})

# Status priority for sorting: lower value = higher priority.
# Matches _select_proactive_sample in task_knowledge_sync.py, with 'deferred'
# added at priority 5 (below 'pending') since it was missing there.
_STATUS_PRIORITY: dict[str, int] = {
    'in-progress': 0,
    'blocked': 1,
    'review': 2,
    'pending': 3,
    'deferred': 5,
}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class FilteredTaskTree:
    """Result of filter_task_tree(): active tasks plus aggregate counts."""

    active_tasks: list[dict] = field(default_factory=list)
    done_count: int = 0
    cancelled_count: int = 0
    other_count: int = 0
    total_count: int = 0


# --------------------------------------------------------------------------- #
# Core filter
# --------------------------------------------------------------------------- #

def filter_task_tree(tasks_data: dict) -> FilteredTaskTree:
    """Partition a raw get_tasks response into active vs. inactive tasks.

    Args:
        tasks_data: Dict returned by taskmaster.get_tasks(), expected to contain
            a 'tasks' key with a list of task dicts.

    Returns:
        FilteredTaskTree with active_tasks sorted by (_STATUS_PRIORITY, -id) and
        aggregate counts for done, cancelled, and other (unknown) statuses.
    """
    raw_tasks = tasks_data.get('tasks') if isinstance(tasks_data, dict) else None
    if not isinstance(raw_tasks, list):
        return FilteredTaskTree()

    active: list[dict] = []
    done_count = 0
    cancelled_count = 0
    other_count = 0

    for task in raw_tasks:
        if not isinstance(task, dict):
            continue  # Skip non-dict elements defensively

        status = task.get('status')  # May be None if key is missing

        if status in ACTIVE_TASK_STATUSES:
            active.append(task)
        elif status == 'done':
            done_count += 1
        elif status == 'cancelled':
            cancelled_count += 1
        else:
            # Unknown status (or None) → other
            other_count += 1

    # Sort active tasks: by priority ascending, then by ID descending (higher = more recent)
    def sort_key(t: dict) -> tuple[int, int]:
        status = t.get('status', 'pending')
        priority = _STATUS_PRIORITY.get(status, len(_STATUS_PRIORITY))
        tid = t.get('id', 0)
        try:
            tid_int = int(tid)
        except (TypeError, ValueError):
            tid_int = 0
        return (priority, -tid_int)

    active.sort(key=sort_key)

    total = len(active) + done_count + cancelled_count + other_count
    return FilteredTaskTree(
        active_tasks=active,
        done_count=done_count,
        cancelled_count=cancelled_count,
        other_count=other_count,
        total_count=total,
    )


# --------------------------------------------------------------------------- #
# Formatter
# --------------------------------------------------------------------------- #

def format_filtered_task_tree(
    tree: FilteredTaskTree,
    max_tasks: int = 50,
    max_chars: int = 50_000,
) -> str:
    """Render a FilteredTaskTree as a prompt-ready string.

    Enforces two limits:
    1. max_tasks — at most this many active tasks are rendered (default 50,
       matching the existing active_tasks[:50] cap in Stage 2).
    2. max_chars — secondary safety clamp; if the rendered string still exceeds
       this after applying max_tasks, lines are trimmed and a truncation notice
       is appended. (ref: task 455)

    The summary line uses the exact format
    '{done_count} done, {cancelled_count} cancelled — omitted' (em dash)
    as specified in the task description.

    Args:
        tree: FilteredTaskTree to render.
        max_tasks: Maximum number of active tasks to include.
        max_chars: Maximum total character budget for the output string.

    Returns:
        Formatted string suitable for injection into a reconciliation prompt.
    """
    active = tree.active_tasks[:max_tasks]
    shown = len(active)
    total_active = len(tree.active_tasks)
    omitted_active = total_active - shown

    header = (
        f'### Active Task Tree\n'
        f'({shown} active shown'
        + (f', {omitted_active} more active omitted by max_tasks cap' if omitted_active > 0 else '')
        + f', {tree.done_count} done, {tree.cancelled_count} cancelled, '
        f'{tree.other_count} other, {tree.total_count} total)\n'
    )

    summary_line = (
        f'{tree.done_count} done, {tree.cancelled_count} cancelled \u2014 omitted'
    )

    if not active:
        body = 'No active tasks.\n'
    else:
        lines = []
        for t in active:
            tid = t.get('id', '?')
            title = t.get('title', '?')
            status = t.get('status', '?')
            deps = t.get('dependencies', [])
            lines.append(f'- [{tid}] ({status}) {title} deps={deps}')
        body = '\n'.join(lines) + '\n'

    result = header + body + summary_line

    # Secondary max_chars clamp
    if len(result) > max_chars:
        # Trim body lines one by one until we fit
        if active:
            task_lines = body.rstrip('\n').split('\n')
            budget = max_chars - len(header) - len(summary_line) - 50  # 50 chars for truncation notice
            kept_lines: list[str] = []
            used = 0
            for line in task_lines:
                if used + len(line) + 1 > budget:
                    break
                kept_lines.append(line)
                used += len(line) + 1

            trimmed_count = total_active - len(kept_lines)
            trunc_notice = f'\n... and {trimmed_count} more active (truncated for budget)\n'
            body = '\n'.join(kept_lines) + trunc_notice
            result = header + body + summary_line

    return result
