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
# Matches _select_proactive_sample in task_knowledge_sync.py.
# 'done': 4 is included so that _select_proactive_sample (which sorts ALL tasks
# including done) can import this map directly instead of redefining it.
# 'deferred': 5 (below 'pending') since it was missing from the original stage dict.
_STATUS_PRIORITY: dict[str, int] = {
    'in-progress': 0,
    'blocked': 1,
    'review': 2,
    'pending': 3,
    'done': 4,
    'deferred': 5,
}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class FilteredTaskTree:
    """Result of filter_task_tree(): active tasks plus aggregate counts."""

    active_tasks: list[dict] = field(default_factory=list)
    done_tasks: list[dict] = field(default_factory=list)
    cancelled_tasks: list[dict] = field(default_factory=list)
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
        FilteredTaskTree with active_tasks sorted by (_STATUS_PRIORITY, -id),
        done_tasks sorted by id descending, cancelled_tasks sorted by id descending,
        and aggregate counts for done, cancelled, and other (unknown) statuses.
    """
    raw_tasks = tasks_data.get('tasks') if isinstance(tasks_data, dict) else None
    if not isinstance(raw_tasks, list):
        return FilteredTaskTree()

    active: list[dict] = []
    done: list[dict] = []
    cancelled: list[dict] = []
    other_count = 0

    for task in raw_tasks:
        if not isinstance(task, dict):
            continue  # Skip non-dict elements defensively

        status = task.get('status')  # May be None if key is missing

        if status in ACTIVE_TASK_STATUSES:
            active.append(task)
        elif status == 'done':
            done.append(task)
        elif status == 'cancelled':
            cancelled.append(task)
        else:
            # Unknown status (or None) → other
            other_count += 1

    def _id_key(t: dict) -> int:
        """Return task id as int for sorting, defaulting to 0 on error."""
        tid = t.get('id', 0)
        try:
            return int(tid)
        except (TypeError, ValueError):
            return 0

    # Sort active tasks: by priority ascending, then by ID descending (higher = more recent)
    def sort_key(t: dict) -> tuple[int, int]:
        status = t.get('status', 'pending')
        priority = _STATUS_PRIORITY.get(status, len(_STATUS_PRIORITY))
        return (priority, -_id_key(t))

    active.sort(key=sort_key)

    # Sort done/cancelled by id descending (recency proxy — higher id = more recently created)
    done.sort(key=_id_key, reverse=True)
    cancelled.sort(key=_id_key, reverse=True)

    total = len(active) + len(done) + len(cancelled) + other_count
    return FilteredTaskTree(
        active_tasks=active,
        done_tasks=done,
        cancelled_tasks=cancelled,
        done_count=len(done),
        cancelled_count=len(cancelled),
        other_count=other_count,
        total_count=total,
    )


# --------------------------------------------------------------------------- #
# Task-line rendering helpers
# --------------------------------------------------------------------------- #

def _render_task_line(task: dict) -> str:
    """Render a single task dict as a prompt-ready line string.

    Format: '- [id] (status) title deps=[...]'
    deps are truncated to first 5 items with '...' suffix when len > 5.
    Missing fields fall back to '?' for id/status/title and [] for deps.

    Args:
        task: Task dict (expected keys: id, status, title, dependencies).

    Returns:
        Formatted string for one task line.
    """
    tid = task.get('id', '?')
    title = task.get('title', '?')
    status = task.get('status', '?')
    deps = task.get('dependencies') or []
    deps_str = str(deps[:5]) + ('...' if len(deps) > 5 else '')
    return f'- [{tid}] ({status}) {title} deps={deps_str}'


def format_task_list(tasks: list[dict]) -> str:
    """Render a list of task dicts as a newline-joined string.

    Returns 'No tasks.' for an empty list; otherwise joins
    _render_task_line(t) for each task with newlines.

    Args:
        tasks: List of task dicts to render.

    Returns:
        Formatted string suitable for injection into a reconciliation prompt.
    """
    if not tasks:
        return 'No tasks.'
    return '\n'.join(_render_task_line(t) for t in tasks)


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
        lines = [_render_task_line(t) for t in active]
        body = '\n'.join(lines) + '\n'

    result = header + body + summary_line

    # Secondary max_chars clamp
    if len(result) > max_chars and active:
        task_lines = body.rstrip('\n').split('\n')
        # Use the full remaining budget — no fixed reserve.  The actual truncation
        # notice length is computed lazily after line accumulation and verified below.
        budget = max_chars - len(header) - len(summary_line)
        if budget <= 0:
            return header + summary_line
        kept_lines: list[str] = []
        used = 0
        for line in task_lines:
            if used + len(line) + 1 > budget:
                break
            kept_lines.append(line)
            used += len(line) + 1

        # Lazy: compute the real notice length and verify the max_chars contract.
        # Pop task lines until result fits or kept_lines is exhausted.
        trimmed_count = len(active) - len(kept_lines)
        trunc_notice = f'\n... and {trimmed_count} more active (truncated for budget)\n'
        body = '\n'.join(kept_lines) + trunc_notice
        result = header + body + summary_line
        while len(result) > max_chars and kept_lines:
            kept_lines.pop()
            trimmed_count = len(active) - len(kept_lines)
            trunc_notice = f'\n... and {trimmed_count} more active (truncated for budget)\n'
            body = '\n'.join(kept_lines) + trunc_notice
            result = header + body + summary_line
        if len(result) > max_chars:
            return header + summary_line

    return result
