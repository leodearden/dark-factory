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

import heapq
from dataclasses import dataclass, field
from typing import Any

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

# Maximum number of done task dicts to retain in FilteredTaskTree.done_tasks.
# This is the sole cap on done_tasks; consumers rely on filter_task_tree to enforce it.
MAX_DONE_TASKS_RETAINED: int = 30

# Maximum number of cancelled task dicts to retain in FilteredTaskTree.cancelled_tasks.
# Caps the list to prevent the '### Recently Cancelled Tasks' section in
# format_filtered_task_tree from growing unbounded — that section is exempt from
# active-task truncation and would single-handedly exceed max_chars without this cap.
MAX_CANCELLED_TASKS_RETAINED: int = 15

# Maximum number of active tasks rendered to the LLM prompt in a single Stage 2 cycle.
# Consumed by both format_filtered_task_tree (as the max_tasks default) AND
# task_knowledge_sync's hint-attention section slice, ensuring the two sections
# always reference the same set of tasks — see task_knowledge_sync.assemble_payload().
MAX_ACTIVE_TASKS_RENDERED: int = 50

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


def id_key(t: dict) -> int:
    """Return task id as int for sorting, defaulting to 0 on error.

    For dotted subtask IDs like '450.2' or '450.2.1', returns the parent
    (first dot-segment) as int, so subtasks sort alongside their parent.
    """
    tid = t.get('id', 0)
    try:
        tid_str = str(tid)
        # For dotted IDs like '450.2', use only the first segment
        return int(tid_str.split('.')[0])
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class FilteredTaskTree:
    """Result of filter_task_tree(): active tasks plus aggregate counts.

    All list fields (active_tasks, done_tasks, cancelled_tasks) are expected to
    contain only ``dict`` elements.  ``filter_task_tree`` enforces this via
    ``isinstance`` checks; direct constructors (e.g. in tests) must honour the
    same invariant — downstream consumers omit per-element type guards.
    """

    active_tasks: list[dict] = field(default_factory=list)
    done_tasks: list[dict] = field(default_factory=list)
    # cancelled_tasks is consumed by two callers:
    #   1. format_filtered_task_tree — renders '### Recently Cancelled Tasks' section
    #      when non-empty, giving reconciliation agents visibility into recent cancellations.
    #   2. _select_proactive_sample in task_knowledge_sync.py — concatenates
    #      active_tasks + done_tasks + cancelled_tasks for proactive sampling.
    cancelled_tasks: list[dict] = field(default_factory=list)
    done_count: int = 0
    cancelled_count: int = 0
    other_count: int = 0
    total_count: int = 0


# --------------------------------------------------------------------------- #
# Subtask flattening
# --------------------------------------------------------------------------- #

def _flatten_with_subtasks(raw_tasks: list[dict]) -> list[dict]:
    """Flatten a nested task list into a single flat list including all subtasks.

    Recursively walks each task's 'subtasks' array.  Bare-integer subtask IDs
    are qualified as 'parent_id.subtask_id' using a shallow copy (originals are
    never mutated).  Already-qualified IDs (containing a dot) are left unchanged.
    The 'subtasks' key is stripped from each emitted dict so the result is a
    genuinely flat list (no nested arrays on any entry).

    Note on intentional divergence: ``task_curator._flatten_task_tree`` (line 948)
    performs a similar recursive walk but does *not* qualify subtask IDs — its
    purpose is shape normalisation, not ID canonicalisation.  Extracting a shared
    helper would require an ID-qualification callback and add coupling across
    package boundaries for minimal gain; the implementations are kept separate.

    Args:
        raw_tasks: Top-level list of task dicts from a get_tasks response.

    Returns:
        Flat list of task dicts (parents first, subtasks immediately following),
        with subtask IDs qualified to 'parent_id.subtask_id' and the 'subtasks'
        key removed from every entry.
    """
    result: list[dict] = []

    def _walk(tasks: list[dict], parent_id: str | None) -> None:
        for task in tasks:
            if not isinstance(task, dict):
                continue
            if parent_id is not None:
                tid = str(task.get('id', ''))
                if '.' not in tid:
                    # Bare integer subtask id — qualify it
                    task = {**task, 'id': f'{parent_id}.{tid}'}
            # Capture subtasks before stripping, then emit a clean flat entry
            subtasks = task.get('subtasks')
            result.append({k: v for k, v in task.items() if k != 'subtasks'})
            if isinstance(subtasks, list) and subtasks:
                # Use the (possibly newly qualified) id as parent for next level
                _walk(subtasks, str(task.get('id', '')))

    _walk(raw_tasks, None)
    return result


# --------------------------------------------------------------------------- #
# Core filter
# --------------------------------------------------------------------------- #

def filter_task_tree(tasks_data: object) -> FilteredTaskTree:
    """Partition a raw get_tasks response into active vs. inactive tasks.

    Args:
        tasks_data: Value returned by taskmaster.get_tasks(), expected to be a
            dict containing a 'tasks' key with a list of task dicts.  Any
            non-dict value (None, list, str, …) is treated as missing input and
            returns an empty FilteredTaskTree.

    Returns:
        FilteredTaskTree with active_tasks sorted by (_STATUS_PRIORITY, -id),
        done_tasks sorted by id descending and capped at MAX_DONE_TASKS_RETAINED,
        cancelled_tasks sorted by id descending and capped at MAX_CANCELLED_TASKS_RETAINED,
        and aggregate counts for done,
        cancelled, and other (unknown) statuses. done_count/cancelled_count
        reflect the full input counts, not the (possibly capped) list lengths —
        consumers can detect overflow via `len(done_tasks) < done_count`.
    """
    raw_tasks = tasks_data.get('tasks') if isinstance(tasks_data, dict) else None
    if not isinstance(raw_tasks, list):
        return FilteredTaskTree()

    # Flatten subtasks into the top-level list, qualifying bare-integer IDs
    all_tasks = _flatten_with_subtasks(raw_tasks)

    active: list[dict] = []
    done: list[dict] = []
    cancelled: list[dict] = []
    done_count = 0
    cancelled_count = 0
    other_count = 0

    for task in all_tasks:
        if not isinstance(task, dict):
            continue  # Skip non-dict elements defensively

        status = task.get('status')  # May be None if key is missing

        if status in ACTIVE_TASK_STATUSES:
            active.append(task)
        elif status == 'done':
            done_count += 1
            done.append(task)
        elif status == 'cancelled':
            cancelled_count += 1
            cancelled.append(task)
        else:
            # Unknown status (or None) → other
            other_count += 1

    # Sort active tasks: by priority ascending, then by ID descending (higher = more recent)
    def sort_key(t: dict) -> tuple[int, int]:
        status = t.get('status', 'pending')
        priority = _STATUS_PRIORITY.get(status, len(_STATUS_PRIORITY))
        return (priority, -id_key(t))

    active.sort(key=sort_key)

    # Select top-MAX_DONE_TASKS_RETAINED done tasks by id descending (recency proxy).
    # heapq.nlargest is O(n + k log n) heap selection vs O(n log n) sort+slice —
    # effectively O(n) for constant k=MAX_DONE_TASKS_RETAINED=30.
    # Composite key (id_key, -original_index) adds the original list position as a
    # tiebreaker, guaranteeing stable selection for equal id_key values (mirrors
    # Python's stable sort: earlier-appearing tasks win ties).
    done_retained = [
        t for _, t in heapq.nlargest(
            MAX_DONE_TASKS_RETAINED,
            enumerate(done),
            key=lambda pair: (id_key(pair[1]), -pair[0]),
        )
    ]

    # Select top-MAX_CANCELLED_TASKS_RETAINED cancelled tasks by id descending.
    # Cap prevents the '### Recently Cancelled Tasks' section from growing unbounded
    # (that section is exempt from active-task truncation in format_filtered_task_tree).
    # Composite key tiebreaker guarantees stable selection for equal id_key values.
    cancelled_retained = [
        t for _, t in heapq.nlargest(
            MAX_CANCELLED_TASKS_RETAINED,
            enumerate(cancelled),
            key=lambda pair: (id_key(pair[1]), -pair[0]),
        )
    ]

    total = len(active) + done_count + cancelled_count + other_count
    return FilteredTaskTree(
        active_tasks=active,
        done_tasks=done_retained,
        cancelled_tasks=cancelled_retained,
        done_count=done_count,
        cancelled_count=cancelled_count,
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
    deps = task.get('dependencies')
    deps = deps if isinstance(deps, list) else []
    deps_str = str(deps[:5]) + ('...' if len(deps) > 5 else '')
    return f'- [{tid}] ({status}) {title} deps={deps_str}'


def format_task_list(tasks: list[Any]) -> str:
    """Render a list of task dicts as a newline-joined string.

    Non-dict elements (e.g. None, int, string) are silently skipped.
    Returns 'No tasks.' for an empty list, when all elements are non-dict,
    or when the input contains no valid dict items.

    Args:
        tasks: List of task dicts to render.  Non-dict items are ignored.

    Returns:
        Formatted string suitable for injection into a reconciliation prompt.
    """
    return '\n'.join(_render_task_line(t) for t in tasks if isinstance(t, dict)) or 'No tasks.'


# --------------------------------------------------------------------------- #
# Surrounding-strings helper
# --------------------------------------------------------------------------- #

def _build_surrounding(
    tree: FilteredTaskTree,
    max_tasks: int,
) -> tuple[str, str, str]:
    """Return (header, cancelled_section, summary_line) for the Active Task Tree block.

    Centralises the header / cancelled-section / summary-line construction used
    by both _select_visible_active_with_body (for budget arithmetic) and
    format_filtered_task_tree (for the rendered output).  Any future change to
    the surrounding format — em-dash style, header phrasing, etc. — need only
    be made here.

    Args:
        tree: FilteredTaskTree whose metadata drives the strings.
        max_tasks: Cap used to compute the 'shown' and 'omitted' counts in
            the header.

    Returns:
        (header, cancelled_section, summary_line) as three strings.
        cancelled_section is '' when tree.cancelled_tasks is empty.
    """
    active = tree.active_tasks[:max_tasks]
    shown = len(active)
    omitted_active = len(tree.active_tasks) - shown

    header = (
        f'### Active Task Tree\n'
        f'({shown} active shown'
        + (f', {omitted_active} more active omitted by max_tasks cap' if omitted_active > 0 else '')
        + f', {tree.done_count} done, {tree.cancelled_count} cancelled, '
        f'{tree.other_count} other, {tree.total_count} total)\n'
    )

    if tree.cancelled_tasks:
        cancelled_lines = '\n'.join(_render_task_line(t) for t in tree.cancelled_tasks)
        cancelled_section = f'\n### Recently Cancelled Tasks\n{cancelled_lines}\n'
        summary_line = f'{tree.done_count} done — omitted'
    else:
        cancelled_section = ''
        summary_line = (
            f'{tree.done_count} done, {tree.cancelled_count} cancelled — omitted'
        )

    return header, cancelled_section, summary_line


# --------------------------------------------------------------------------- #
# Visible-active selection helpers
# --------------------------------------------------------------------------- #

def _select_visible_active_with_body(
    tree: FilteredTaskTree,
    max_tasks: int = MAX_ACTIVE_TASKS_RENDERED,
    max_chars: int = 50_000,
) -> tuple[list[dict], str | None]:
    """Internal worker: select visible active tasks and return the pre-rendered body.

    Returns (visible_tasks, body_str) where body_str is the fully rendered body
    string (with truncation notice when partial), or None when no tasks fit the
    budget.  Both the public select_visible_active helper and
    format_filtered_task_tree call this worker, so task lines are rendered once
    and reused rather than being computed independently in each caller.

    Algorithm:
      1. Slice active = tree.active_tasks[:max_tasks]; return ([], None) if empty.
      2. Build surroundings via _build_surrounding.
      3. Render all lines; if full result fits, return (active, body).
      4. Compute budget = max_chars - overhead; return ([], None) if budget <= 0.
      5. Greedy fill kept_lines while cumulative cost <= budget.
      6. Lazy verification: pop lines until the realised result fits or
         kept_lines is drained.
      7. Return (active[:kept], body_with_trunc_notice), or ([], None) if drained.
    """
    active = tree.active_tasks[:max_tasks]
    if not active:
        return [], None

    header, cancelled_section, summary_line = _build_surrounding(tree, max_tasks)

    # ── Fast path: full result fits in budget ── #
    lines = [_render_task_line(t) for t in active]
    body = '\n'.join(lines) + '\n'
    full = header + body + cancelled_section + summary_line
    if len(full) <= max_chars:
        return active, body

    # ── Budget-capped path ── #
    budget = max_chars - len(header) - len(cancelled_section) - len(summary_line)
    if budget <= 0:
        return [], None

    # Greedy fill.
    kept_lines: list[str] = []
    used = 0
    for line in lines:
        if used + len(line) + 1 > budget:
            break
        kept_lines.append(line)
        used += len(line) + 1

    # Lazy verification: recompute real truncation-notice length and pop until
    # the realised result fits or kept_lines is exhausted.
    trimmed_count = len(active) - len(kept_lines)
    trunc_notice = f'\n... and {trimmed_count} more active (truncated for budget)\n'
    result_body = '\n'.join(kept_lines) + trunc_notice
    result = header + result_body + cancelled_section + summary_line
    while len(result) > max_chars and kept_lines:
        kept_lines.pop()
        trimmed_count = len(active) - len(kept_lines)
        trunc_notice = f'\n... and {trimmed_count} more active (truncated for budget)\n'
        result_body = '\n'.join(kept_lines) + trunc_notice
        result = header + result_body + cancelled_section + summary_line

    if not kept_lines:
        return [], None
    return active[: len(kept_lines)], result_body


def select_visible_active(
    tree: FilteredTaskTree,
    max_tasks: int = MAX_ACTIVE_TASKS_RENDERED,
    max_chars: int = 50_000,
) -> list[dict]:
    """Return the prefix of tree.active_tasks[:max_tasks] that survives the max_chars clamp.

    This is the shared **public** helper backing both format_filtered_task_tree
    and the hint-attention section in assemble_payload.  Both consumers use the
    same _select_visible_active_with_body worker so that the rendered Active
    Task Tree and the hint section always reference exactly the same set of tasks.

    Algorithm:
      1. Slice active = tree.active_tasks[:max_tasks]; return [] if empty.
      2. Build header / cancelled_section / summary_line via _build_surrounding.
      3. Render all lines; if the full result fits within max_chars, return active.
      4. Otherwise compute budget = max_chars - overhead; return [] if budget <= 0.
      5. Greedy fill kept_lines while cumulative cost <= budget.
      6. Lazy verification pop loop to account for truncation-notice length.
      7. Return active[:len(kept_lines)] (may be [] if loop drained everything).

    Args:
        tree: FilteredTaskTree whose active_tasks are the candidates.
        max_tasks: Maximum number of active tasks to consider (same default as
            format_filtered_task_tree).
        max_chars: Character budget for the full rendered output (same default
            as format_filtered_task_tree).

    Returns:
        A (possibly empty) prefix list of task dicts that will all appear in
        the output of format_filtered_task_tree(tree, max_tasks, max_chars).
    """
    visible, _ = _select_visible_active_with_body(tree, max_tasks, max_chars)
    return visible


# --------------------------------------------------------------------------- #
# Formatter
# --------------------------------------------------------------------------- #

def format_filtered_task_tree(
    tree: FilteredTaskTree,
    max_tasks: int = MAX_ACTIVE_TASKS_RENDERED,
    max_chars: int = 50_000,
) -> str:
    """Render a FilteredTaskTree as a prompt-ready string.

    Enforces two limits:
    1. max_tasks — at most this many active tasks are rendered (default 50,
       matching the existing active_tasks[:50] cap in Stage 2).
    2. max_chars — secondary safety clamp; if the rendered string still exceeds
       this after applying max_tasks, active task lines are trimmed and a
       truncation notice is appended. (ref: task 455)

    The summary line format depends on whether cancelled tasks are present:
    - When tree.cancelled_tasks is non-empty, a '### Recently Cancelled Tasks'
      section is rendered between the active body and the summary, and the
      summary becomes '{done_count} done — omitted' (cancelled no longer
      omitted since they are displayed in the section).
    - When tree.cancelled_tasks is empty, the section is omitted and the
      summary retains the format '{done_count} done, {cancelled_count}
      cancelled — omitted' for backward compatibility.

    The cancelled section is never truncated by the max_chars clamp — only
    active task lines are trimmed. The cancelled section length is subtracted
    from the available budget so that active-task truncation accounts for it.

    Note: cancelled_tasks serves two consumers:
      1. This formatter — renders the '### Recently Cancelled Tasks' section.
      2. _select_proactive_sample in task_knowledge_sync.py — concatenates
         active_tasks + done_tasks + cancelled_tasks for proactive sampling.

    Args:
        tree: FilteredTaskTree to render.
        max_tasks: Maximum number of active tasks to include.
        max_chars: Maximum total character budget for the output string.

    Returns:
        Formatted string suitable for injection into a reconciliation prompt.
    """
    # active_slice drives the header's "shown" count — reflects the max_tasks cap,
    # NOT the secondary max_chars clamp (same as the pre-refactor behaviour).
    active_slice = tree.active_tasks[:max_tasks]
    # _build_surrounding is also called inside _select_visible_active_with_body;
    # both calls are cheap (pure string concatenation) relative to task-line rendering.
    header, cancelled_section, summary_line = _build_surrounding(tree, max_tasks)

    if not active_slice:
        return header + 'No active tasks.\n' + cancelled_section + summary_line

    # Delegate to the shared internal worker — single source of truth for the
    # visible-window decision.  The worker returns the pre-rendered body so that
    # task lines are not rendered a second time here.
    visible, body = _select_visible_active_with_body(tree, max_tasks, max_chars)

    if not visible or body is None:
        # Budget too tight for any task lines (budget<=0 or lazy loop drained all).
        return header + cancelled_section + summary_line

    result = header + body + cancelled_section + summary_line
    # Defensive guard: if _select_visible_active_with_body's budget algorithm ever
    # drifts from what the formatter assembles (e.g. a truncation-notice format
    # change applied in one place only), fall back to the safe header-only result
    # rather than silently returning an oversized string.
    if len(result) > max_chars:
        return header + cancelled_section + summary_line
    return result
