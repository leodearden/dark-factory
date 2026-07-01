"""Shared utility module for filtering and formatting the Taskmaster task tree.

Extracts active tasks from the full raw get_tasks response, partitions them by
status, sorts by priority, and provides a budget-capped formatter.

Design decision: active status set = {pending, in-progress, blocked, deferred, review,
merge-deferred}.
The task description says 'pending, in-progress, blocked, deferred' explicitly, but
existing code (_select_proactive_sample, old Stage 2 filter) treats 'review' as active.
Excluding it would regress proactive-sampling tests.  The task's intent is
'exclude done/cancelled', so widening to 'not done/cancelled' preserves that
intent without regressions. (ref: task 455)
merge-deferred is a non-terminal holding state for atomic-train members that have
passed own-verify-green and are awaiting the group merge; classifying it as active
keeps holding-state members visible in reconciliation prompts until done.
(ref: PRD orchestrator-atomic-train-merge §9.2, task 1519)
"""

from __future__ import annotations

import heapq
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# Count-snapshot detection
# --------------------------------------------------------------------------- #

# Matches lines containing >=2 occurrences of `\d+ <status>` on a single line,
# separated by at least one ',' or '/' delimiter.
# Non-DOTALL so the two tokens cannot span a newline.
# The delimiter requirement distinguishes count-snapshot lines (e.g. '1505 done / 148 cancelled',
# '3355 done, 290 cancelled') from incidental two-token sentences that lack delimiters
# (e.g. '2 review comments and 3 pending follow-ups').
COUNT_SNAPSHOT_RE: re.Pattern[str] = re.compile(
    r'\b\d+\s+(?:done|cancell?ed|pending|in[-_ ]?progress|blocked|deferred|review|total|merge[-_ ]?deferred)\b'
    r'[^,/\n]*[,/][^,/\n]*'
    r'\b\d+\s+(?:done|cancell?ed|pending|in[-_ ]?progress|blocked|deferred|review|total|merge[-_ ]?deferred)\b',
    re.IGNORECASE,
)


def is_count_snapshot(text: str) -> bool:
    """Return True when text contains a count-snapshot pattern (>=2 digit+status tokens)."""
    return bool(COUNT_SNAPSHOT_RE.search(text))


def strip_snapshot_lines(text: str) -> tuple[str, int]:
    """Remove count-snapshot lines from text.

    Splits on newline, drops lines where is_count_snapshot is True, rejoins,
    and returns (filtered_text, num_dropped).
    """
    lines = text.split('\n')
    kept = [line for line in lines if not is_count_snapshot(line)]
    dropped = len(lines) - len(kept)
    return '\n'.join(kept), dropped


# --------------------------------------------------------------------------- #
# Status constants
# --------------------------------------------------------------------------- #

ACTIVE_TASK_STATUSES: frozenset[str] = frozenset(
    {
        'pending',
        'in-progress',
        'blocked',
        'deferred',
        'review',
        # Non-terminal holding state for atomic-train members that have passed
        # own-verify-green and are awaiting the group merge.
        # Deliberately excluded from TERMINAL_STATUSES and STATUS_TRIGGERS —
        # see PRD orchestrator-atomic-train-merge §9.2 and task 1519.
        'merge-deferred',
    }
)

INACTIVE_TASK_STATUSES: frozenset[str] = frozenset(
    {
        'done',
        'cancelled',
    }
)

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
    # Priority 6: below deferred since merge-deferred members have completed their
    # own work and need no active operator attention until the group merge.
    # (PRD orchestrator-atomic-train-merge §9.2, task 1519)
    'merge-deferred': 6,
}


def id_key(t: dict) -> int:
    """Return task id as int for sorting, defaulting to 0 on error.

    Bare-integer ids (int or numeric string like '42') are coerced directly
    to int.  Non-parseable values (None, 'abc', dotted strings like '450.2')
    return 0.  Top-level tasks only; the scheduler is top-level-only (DF-D).
    """
    tid = t.get('id', 0)
    try:
        return int(tid)
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

    max_task_id: Comprehensive (uncapped) maximum top-level task id across the
        FULL input, computed by id_key (bare-integer parse only; non-parseable
        ids return 0).  Defaults to 0 when the input is empty or contains no
        parseable ids.  This field is independent of the done/cancelled/active
        render caps and provides ground truth for detecting partial/wrong-source
        bulk reads.
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
    max_task_id: int = 0


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

    active: list[dict] = []
    done: list[dict] = []
    cancelled: list[dict] = []
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
        t
        for _, t in heapq.nlargest(
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
        t
        for _, t in heapq.nlargest(
            MAX_CANCELLED_TASKS_RETAINED,
            enumerate(cancelled),
            key=lambda pair: (id_key(pair[1]), -pair[0]),
        )
    ]

    total = len(active) + done_count + cancelled_count + other_count

    # Compute comprehensive max task id over the FULL top-level list (not the
    # capped done_retained / cancelled_retained slices).  Only bare-integer ids
    # contribute a positive value; non-parseable ids return 0 via id_key.
    # Defaults to 0 when raw_tasks is empty or every id_key returns 0.
    max_task_id = max((id_key(t) for t in raw_tasks if isinstance(t, dict)), default=0)

    return FilteredTaskTree(
        active_tasks=active,
        done_tasks=done_retained,
        cancelled_tasks=cancelled_retained,
        done_count=done_count,
        cancelled_count=cancelled_count,
        other_count=other_count,
        total_count=total,
        max_task_id=max_task_id,
    )


# --------------------------------------------------------------------------- #
# Census-inconsistency helper
# --------------------------------------------------------------------------- #


def summarize_statuses(statuses: dict[str, str]) -> dict:
    """Return a census dict from an authoritative {id: status} map.

    Classifies each status using the project-wide ACTIVE_TASK_STATUSES and
    INACTIVE_TASK_STATUSES sets so census semantics stay identical to
    filter_task_tree — single source of truth for status classification.

    Returns:
        dict with integer counts for 'total', 'done', 'cancelled', 'active',
        and 'other' (unknown/unrecognised statuses).

    Pure: no I/O, no side effects.
    """
    total = 0
    done = 0
    cancelled = 0
    active = 0
    other = 0

    for status in statuses.values():
        total += 1
        if status == 'done':
            done += 1
        elif status == 'cancelled':
            cancelled += 1
        elif status in ACTIVE_TASK_STATUSES:
            active += 1
        else:
            other += 1

    return {
        'total': total,
        'done': done,
        'cancelled': cancelled,
        'active': active,
        'other': other,
    }


def cross_verify_task_counts(tree: FilteredTaskTree, statuses: dict[str, str] | None) -> dict:
    """Compare authoritative get_statuses census against the get_tasks-derived tree.

    When the status map is empty or None (taskmaster unavailable), returns
    available=False, consistent=True — fail-open: no spurious mismatch warning.
    When available, compares census total/done against tree.total_count and
    tree.done_count and flags any discrepancy.

    Args:
        tree: FilteredTaskTree from filter_task_tree() (the get_tasks read path).
        statuses: Compact {id: status} map from get_statuses(), or None/empty.

    Returns:
        dict with:
            available (bool): False when statuses is empty/None.
            consistent (bool): True when available and no mismatches.
            total_mismatch (bool): set when available and totals differ.
            done_mismatch (bool): set when available and done counts differ.
            authoritative (dict): census from statuses (when available).
            tree (dict): {'total': tree.total_count, 'done': tree.done_count}.

    Note on read-skew: the census and the tree are produced by two independent reads
    (get_tasks then get_statuses) taken a moment apart. A status transition committed
    between the two reads can produce a transient total_mismatch/done_mismatch that
    resolves itself next cycle — these are read-skew artefacts rather than the
    truncation incident this diagnostic is designed to catch. Single-cycle divergences
    should be treated as advisory; only divergence that persists across consecutive
    cycles warrants escalation.

    Pure: no I/O, no side effects.
    """
    _unavailable = {
        'available': False,
        'consistent': True,
        'total_mismatch': False,
        'done_mismatch': False,
        'authoritative': None,
        'tree': {'total': tree.total_count, 'done': tree.done_count},
    }

    if not statuses:
        return _unavailable

    census = summarize_statuses(statuses)
    total_mismatch = census['total'] != tree.total_count
    done_mismatch = census['done'] != tree.done_count
    consistent = not total_mismatch and not done_mismatch

    return {
        'available': True,
        'consistent': consistent,
        'total_mismatch': total_mismatch,
        'done_mismatch': done_mismatch,
        'authoritative': census,
        'tree': {'total': tree.total_count, 'done': tree.done_count},
    }


def _coerce_id_set(values: Iterable) -> set[int]:
    """Coerce a mixed int/str id iterable into a canonical int set.

    Reuses the first-dot-segment rule from detect_census_inconsistency: ints
    are used directly, strings are split on '.' and the first segment is
    coerced to int, and unparseable entries are silently ignored.
    """
    result: set[int] = set()
    for ref in values:
        try:
            result.add(int(str(ref).split('.')[0]))
        except (TypeError, ValueError, AttributeError):
            continue  # silently ignore unparseable entries
    return result


def diff_status_correction(metadata: dict, statuses: dict[str, str] | None) -> dict:
    """Diff a cached Mem0 ``project_status_correction`` memory against the live census.

    ``project_status_correction`` memories are written at runtime by agents
    (metadata ``kind='project_status_correction'``, fields ``task_count_done``/
    ``task_count_total``/``active_tasks``) and can silently go stale.  This pure
    helper compares that cached snapshot against the authoritative get_statuses
    census so a caller (``ReconciliationHarness._reconcile_status_correction``)
    can decide whether to supersede it.

    Args:
        metadata: The cached memory's metadata dict (or falsy for a missing/
            malformed record).  ``task_count_done``/``task_count_total``/
            ``active_tasks`` are read from it; missing fields become None.
        statuses: Compact {id: status} map from get_statuses(), or None/empty
            when the live census is unavailable.

    Returns:
        dict with:
            available (bool): False when statuses is empty/None — fail-open,
                a caller must NOT supersede against an unavailable census.
            diverged (bool): True when available and any field mismatches.
            done_mismatch / total_mismatch / active_mismatch (bool): per-field
                divergence flags (available only when `available` is True).
            cached (dict): {'done', 'total', 'active_tasks'} echoed from metadata,
                always populated so callers can log the pre-diff snapshot.
            live (dict | None): {'done', 'total', 'active_tasks'} recomputed from
                statuses (active_tasks as a sorted int list), or None when
                unavailable.

    active_tasks comparison is order-insensitive and coerces both cached ints
    and live string keys via the first-dot-segment rule (see
    detect_census_inconsistency), so ordering or int-vs-str differences never
    cause a false mismatch.

    Pure: no I/O, no side effects.
    """
    metadata = metadata or {}
    cached = {
        'done': metadata.get('task_count_done'),
        'total': metadata.get('task_count_total'),
        'active_tasks': metadata.get('active_tasks'),
    }

    if not statuses:
        return {
            'available': False,
            'diverged': False,
            'done_mismatch': False,
            'total_mismatch': False,
            'active_mismatch': False,
            'cached': cached,
            'live': None,
        }

    census = summarize_statuses(statuses)
    live_active_ids = _coerce_id_set(
        tid for tid, status in statuses.items() if status in ACTIVE_TASK_STATUSES
    )
    live = {
        'done': census['done'],
        'total': census['total'],
        'active_tasks': sorted(live_active_ids),
    }

    # Missing/None cached fields count as a mismatch so a malformed memory is rewritten.
    done_mismatch = cached['done'] is None or cached['done'] != live['done']
    total_mismatch = cached['total'] is None or cached['total'] != live['total']
    active_mismatch = (
        cached['active_tasks'] is None
        or _coerce_id_set(cached['active_tasks']) != live_active_ids
    )
    diverged = done_mismatch or total_mismatch or active_mismatch

    return {
        'available': True,
        'diverged': diverged,
        'done_mismatch': done_mismatch,
        'total_mismatch': total_mismatch,
        'active_mismatch': active_mismatch,
        'cached': cached,
        'live': live,
    }


def detect_census_inconsistency(max_task_id: int, referenced_ids: Iterable) -> list[int]:
    """Return referenced task ids that strictly exceed the census maximum.

    Surfaces the 'highest-ID below known task IDs' inconsistency signature
    produced by a partial or wrong-source bulk task read: when an event or flag
    references a task id higher than the FilteredTaskTree's max_task_id, the
    bulk read did not cover that id — it may have been truncated or pointed at
    a different project_root.

    Args:
        max_task_id: The authoritative maximum task id from FilteredTaskTree
            (computed over the full flattened input, not the capped lists).
        referenced_ids: Iterable of candidate ids to check.  Each entry is
            parsed using the id_key first-dot-segment-to-int rule:
            integers are used directly; strings are split on '.' and the first
            segment is coerced to int; unparseable entries are silently ignored.

    Returns:
        Sorted, deduplicated list of ints that strictly exceed max_task_id.
        Returns [] when referenced_ids is empty or no entry exceeds max_task_id.
    """
    exceeding: set[int] = set()
    for ref in referenced_ids:
        try:
            ref_str = str(ref)
            val = int(ref_str.split('.')[0])
        except (TypeError, ValueError, AttributeError):
            continue  # silently ignore unparseable entries
        if val > max_task_id:
            exceeding.add(val)
    return sorted(exceeding)


# --------------------------------------------------------------------------- #
# Task-dump contamination detector
# --------------------------------------------------------------------------- #
# Complements detect_census_inconsistency (numeric census) with two additional
# project-identity + title-plausibility signals:
#
#  1. Stamped project_id mismatch — the primary signal.  Added to get_tasks /
#     get_task results by server/tools.py (task 1661) so any consumer can verify
#     at a glance which project a dump came from.
#
#  2. Decompose-PRD step-pattern title — corroborating heuristic that fired in
#     the 2026-06 incident: reify's 'impl(step-N)' subtask titles appearing for
#     dark_factory task IDs.  Exact substring matching is deterministic and
#     testable; no fragile numeric threshold.
#
# detect_task_dump_contamination is purely a read/classification helper — it
# never writes or modifies its inputs.  The caller decides what to do with a
# contaminated result (log a WARNING, record a stat, raise a finding).


# Substrings that identify decompose-PRD step-pattern titles (lower-cased check).
#
# Heuristic note: plain substring matching may occasionally false-positive on
# legitimate task titles that happen to contain these substrings (e.g. a task
# discussing the decompose-PRD pattern, or "fix impl(step parser) crash").
# Since this signal is observability-only (records a stat / WARNING; never mutates
# state), the impact is a spurious operator-triage alert rather than a correctness
# issue.  If false positives become noisy, tighten to anchor on the title start or
# require the full 'impl(step-<n>)' shape via a regex.
_STEP_PATTERN_MARKERS: tuple[str, ...] = ('impl(step', 'test(step')


def detect_task_dump_contamination(
    tasks_data: object,
    expected_project_id: str,
) -> dict:
    """Detect cross-project contamination in a raw get_tasks dump.

    Returns a dict with the following fields:

    - stamped_project_id (str | None): tasks_data.get('project_id') when
      tasks_data is a dict, else None.  None means the dump is unstamped
      (legacy or pre-task-1661 server).

    - project_mismatch (str | None): the stamped_project_id when it is
      present AND differs from expected_project_id.  None otherwise (clean
      or unstamped).

    - step_pattern_title_ids (list): ids of tasks whose title, lowercased and
      stripped, contains 'impl(step' or 'test(step' — the decompose-PRD
      signature from the 2026-06 incident.  Empty list when none match.

    - contaminated (bool): True when project_mismatch is non-empty OR
      step_pattern_title_ids is non-empty.  False otherwise (including on
      empty/None input).

    Defensive handling:
    - Non-dict tasks_data (None, list, …) → all fields empty/False, no crash.
    - Missing 'tasks' key → same.
    - Non-dict task entries inside the list → silently skipped.
    """
    result: dict = {
        'stamped_project_id': None,
        'project_mismatch': None,
        'step_pattern_title_ids': [],
        'contaminated': False,
    }

    if not isinstance(tasks_data, dict):
        return result

    stamped = tasks_data.get('project_id')
    result['stamped_project_id'] = stamped

    if stamped is not None and stamped != expected_project_id:
        result['project_mismatch'] = stamped

    raw_tasks = tasks_data.get('tasks')
    if isinstance(raw_tasks, list):
        offending: list = []
        for task in raw_tasks:
            if not isinstance(task, dict):
                continue
            title = task.get('title')
            if not isinstance(title, str):
                continue
            lowered = title.lower().strip()
            if any(marker in lowered for marker in _STEP_PATTERN_MARKERS):
                tid = task.get('id')
                if tid is not None:
                    offending.append(tid)
        result['step_pattern_title_ids'] = offending

    result['contaminated'] = bool(result['project_mismatch']) or bool(
        result['step_pattern_title_ids']
    )
    return result


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


def _format_header(shown: int, omitted_active: int, tree: FilteredTaskTree) -> str:
    """Return just the header line for the Active Task Tree block.

    Extracted so that _select_visible_active_with_body can rebuild the header
    cheaply (varying only 'shown') without re-running the cancelled-section /
    summary-line logic inside _build_surrounding.

    Includes an authoritative 'highest task id: N' token derived from
    tree.max_task_id so the LLM receives ground truth instead of inferring
    the maximum from the (possibly capped or wrong-source) rendered body.

    Args:
        shown: Number of active tasks actually rendered in the body.
        omitted_active: Count of tasks omitted by the max_tasks cap only (not
            by the max_chars clamp — those are reported via the truncation notice).
        tree: FilteredTaskTree supplying the done/cancelled/other/total counts.
    """
    return (
        f'### Active Task Tree\n'
        f'({shown} active shown'
        + (f', {omitted_active} more active omitted by max_tasks cap' if omitted_active > 0 else '')
        + f', {tree.done_count} done, {tree.cancelled_count} cancelled, '
        f'{tree.other_count} other, {tree.total_count} total, '
        f'highest task id: {tree.max_task_id})\n'
    )


def _build_surrounding(
    tree: FilteredTaskTree,
    max_tasks: int,
    shown_override: int | None = None,
) -> tuple[str, str, str]:
    """Return (header, cancelled_section, summary_line) for the Active Task Tree block.

    Centralises the header / cancelled-section / summary-line construction used
    by both _select_visible_active_with_body (for budget arithmetic) and
    format_filtered_task_tree (for the rendered output).  Any future change to
    the surrounding format — em-dash style, header phrasing, etc. — need only
    be made here.

    Args:
        tree: FilteredTaskTree whose metadata drives the strings.
        max_tasks: Cap used to compute the 'omitted' count in the header.
        shown_override: When provided, use this value for the 'shown' count in
            the header instead of the default len(tree.active_tasks[:max_tasks]).
            Used by _select_visible_active_with_body to report the
            post-max_chars-clamp visible count rather than the pre-clamp
            len(tree.active_tasks[:max_tasks]).

    Returns:
        (header, cancelled_section, summary_line) as three strings.
        cancelled_section is '' when tree.cancelled_tasks is empty.
    """
    active = tree.active_tasks[:max_tasks]
    # 'shown' may reflect the post-clamp count when an override is supplied.
    # 'omitted_active' retains its max_tasks-cap-only meaning; the body's
    # truncation notice reports max_chars-clamped count separately.
    shown = shown_override if shown_override is not None else len(active)
    omitted_active = len(tree.active_tasks) - len(active)

    header = _format_header(shown, omitted_active, tree)

    if tree.cancelled_tasks:
        cancelled_lines = '\n'.join(_render_task_line(t) for t in tree.cancelled_tasks)
        cancelled_section = f'\n### Recently Cancelled Tasks\n{cancelled_lines}\n'
        summary_line = f'{tree.done_count} done — omitted'
    else:
        cancelled_section = ''
        summary_line = f'{tree.done_count} done, {tree.cancelled_count} cancelled — omitted'

    return header, cancelled_section, summary_line


# --------------------------------------------------------------------------- #
# Visible-active selection helpers
# --------------------------------------------------------------------------- #


def _select_visible_active_with_body(
    tree: FilteredTaskTree,
    max_tasks: int = MAX_ACTIVE_TASKS_RENDERED,
    max_chars: int = 50_000,
) -> tuple[list[dict], str | None, str, str, str]:
    """Internal worker: select visible active tasks and return the pre-rendered body.

    Returns (visible_tasks, body_str, header, cancelled_section, summary_line)
    where body_str is the fully rendered body string (with truncation notice when
    partial), or None when no tasks fit the budget.  The surroundings tuple
    (header, cancelled_section, summary_line) is always populated so callers
    never need to call _build_surrounding independently.

    Both the public select_visible_active helper and format_filtered_task_tree
    (via render_active_section) call this worker, so task lines are rendered once
    and reused rather than being computed independently in each caller.

    Algorithm:
      1. Build surroundings via _build_surrounding (always, even for empty active).
      2. Slice active = tree.active_tasks[:max_tasks]; return ([], None, ...) if empty.
      3. Render all lines; if full result fits, return (active, body, ...).
      4. Compute budget = max_chars - overhead; greedy fill + lazy pop run only
         when budget > 0 (kept_lines stays empty when budget <= 0).
      5. Greedy fill kept_lines while cumulative cost <= budget.
      6. Lazy verification: pop lines until the realised result fits or
         kept_lines is drained.
      7. Single tail rebuild: compute final_shown = len(kept_lines) (0 on both
         empty-kept_lines paths; partial-clamp count otherwise) and call
         _format_header exactly once.  Two terminal returns share the rebuilt
         header.
    """
    header, cancelled_section, summary_line = _build_surrounding(tree, max_tasks)

    active = tree.active_tasks[:max_tasks]
    # omitted_active is constant across this function (depends only on the
    # max_tasks cap, not on how many lines survive the max_chars clamp).
    # Pre-computed here so the single tail _format_header call can use it
    # without re-running the cancelled-section / summary-line logic inside
    # _build_surrounding.
    omitted_active = len(tree.active_tasks) - len(active)

    if not active:
        # Empty-active early return: shown count is unchanged (0 either way).
        return [], None, header, cancelled_section, summary_line

    # ── Fast path: full result fits in budget ── #
    lines = [_render_task_line(t) for t in active]
    body = '\n'.join(lines) + '\n'
    full = header + body + cancelled_section + summary_line
    if len(full) <= max_chars:
        # visible == active: shown count is unchanged; no header rebuild needed.
        return active, body, header, cancelled_section, summary_line

    # ── Budget-capped path ── #
    kept_lines: list[str] = []
    result_body: str | None = None
    budget = max_chars - len(header) - len(cancelled_section) - len(summary_line)

    if budget > 0:
        # Greedy fill.
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

    # Single tail rebuild: final_shown = len(kept_lines) is 0 on both
    # empty-kept_lines paths (budget <= 0 and lazy-drain), and equals the
    # partial-clamp count otherwise.  The rebuilt header is monotonically <=
    # the pre-clamp header (post-clamp shown <= pre-clamp shown, so digit count
    # can only stay the same or decrease); for the partial-clamp case the
    # lazy-pop loop already established len(result) <= max_chars with the
    # strictly longer pre-clamp header, so no additional pop iterations are
    # needed after the rebuild.
    final_shown = len(kept_lines)
    header = _format_header(final_shown, omitted_active, tree)

    if not kept_lines:
        return [], None, header, cancelled_section, summary_line
    return active[:len(kept_lines)], result_body, header, cancelled_section, summary_line


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

    .. deprecated::
        Prefer ``render_active_section`` when both the visible-task list *and*
        the assembled prompt string are needed — that avoids a second call to
        ``_select_visible_active_with_body`` (and the duplicate task-line
        rendering it implies).  ``select_visible_active`` is retained for
        callers that genuinely need *only* the visible list without the
        assembled string (e.g. unit tests that assert on task membership rather
        than rendered output).

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
    visible, _body, _header, _cancelled, _summary = _select_visible_active_with_body(
        tree, max_tasks, max_chars
    )
    return visible


def render_active_section(
    tree: FilteredTaskTree,
    max_tasks: int = MAX_ACTIVE_TASKS_RENDERED,
    max_chars: int = 50_000,
) -> tuple[list[dict], str]:
    """Return (visible_tasks, assembled_string) for the Active Task Tree prompt slot.

    Single-call API that returns BOTH the visible-task list (for hint-section
    consumption) AND the fully assembled prompt string (for the Active Task Tree
    slot), calling _select_visible_active_with_body exactly once.

    This eliminates the double rendering that occurs when a caller invokes
    select_visible_active and format_filtered_task_tree separately — each call
    invoked the worker internally, rendering every task line twice and calling
    _build_surrounding twice.

    The returned assembled_string is byte-identical to format_filtered_task_tree(
    tree, max_tasks, max_chars).  The returned visible list is byte-identical to
    select_visible_active(tree, max_tasks, max_chars).

    Args:
        tree: FilteredTaskTree to render.
        max_tasks: Maximum number of active tasks to include.
        max_chars: Maximum total character budget for the output string.

    Returns:
        (visible_tasks, assembled_string) tuple.  visible_tasks is a (possibly
        empty) prefix list of task dicts; assembled_string is the fully rendered
        prompt string suitable for injection into a reconciliation prompt.
    """
    active_slice = tree.active_tasks[:max_tasks]
    visible, body, header, cancelled_section, summary_line = _select_visible_active_with_body(
        tree, max_tasks, max_chars
    )

    if not active_slice:
        return visible, header + 'No active tasks.\n' + cancelled_section + summary_line

    if not visible or body is None:
        # Budget too tight for any task lines (budget<=0 or lazy loop drained all).
        # visible is always [] in this branch (worker guarantees body is None iff
        # visible is empty), but return [] explicitly to keep the contract clear:
        # an empty assembled string → an empty visible list.
        return [], header + cancelled_section + summary_line

    assembled = header + body + cancelled_section + summary_line
    # Defensive guard: mirrors the guard in the former format_filtered_task_tree
    # body — if the budget algorithm drifts from the assembly, fall back safely.
    # Return [] for visible so callers never see tasks claimed present in a string
    # that doesn't actually contain their rendered lines.
    if len(assembled) > max_chars:
        return [], header + cancelled_section + summary_line
    return visible, assembled


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
    _, assembled_str = render_active_section(tree, max_tasks, max_chars)
    return assembled_str
