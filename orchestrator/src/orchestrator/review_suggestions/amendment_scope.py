"""Scope a post-amendment review to the amendment delta (task 2750).

After an in-workflow amendment round, the review re-runs all reviewers
against the full task diff, surfacing fresh suggestion-tier findings on code
the amendment never touched.  :func:`partition_suggestions_by_delta` is the
deterministic, enforceable half of the fix: it partitions the aggregated
reviewer suggestions by whether each finding's ``location`` (``file:line``)
falls within the amendment's NEW-side changed line ranges.  In-delta
suggestions stay in the verdict; out-of-delta suggestions are routed to the
curator by the caller instead of re-arming the amendment loop.

Sibling of :mod:`orchestrator.review_suggestions.dedup`.  Pure functions —
no git invocation, no workflow state.  The location-parse convention mirrors
``TaskWorkflow._suggestions_in_scope`` (``(location or '').strip()`` then
split on ``':'``) and the range-membership check reuses the closed-interval
overlap idea from ``workflow._line_ranges_stackable`` (``s1 <= e2 and
s2 <= e1``).
"""

from __future__ import annotations


def _parse_location(location: object) -> tuple[str, int | None] | None:
    """Parse a suggestion ``location`` into ``(file, line)``.

    Tolerant of the malformed inputs reviewers occasionally emit (mirrors the
    convention in ``_suggestions_in_scope``): never raises.

    Returns:
        - ``None`` when the location is unusable: missing, empty, whitespace
          only, an empty file part (e.g. ``":"`` or ``":42"``), or a ``:line``
          suffix that does not parse as an integer.
        - ``(file, None)`` for a file-level location (no ``:`` at all).
        - ``(file, line)`` for a well-formed ``file:line``.
    """
    if not isinstance(location, str):
        return None
    loc = location.strip()
    if not loc:
        return None
    parts = loc.split(':', 1)
    file_path = parts[0].strip()
    if not file_path:
        return None
    if len(parts) == 1:
        # No ':' at all — file-level location.
        return (file_path, None)
    line_str = parts[1].strip()
    try:
        return (file_path, int(line_str))
    except ValueError:
        # ':line' present but not an integer — unusable (out of delta).
        return None


def _line_in_ranges(
    line: int, ranges: list[tuple[int, int]], context_lines: int,
) -> bool:
    """Return True iff *line* falls within any range (± ``context_lines``).

    Closed-interval membership: a point ``[line, line]`` overlaps a range
    ``[start, end]`` widened by ``context_lines`` on each side iff
    ``start - context_lines <= line <= end + context_lines`` — the same
    ``s1 <= e2 and s2 <= e1`` intersection test used by
    ``workflow._line_ranges_stackable``.
    """
    return any(
        start - context_lines <= line <= end + context_lines
        for start, end in ranges
    )


def partition_suggestions_by_delta(
    suggestions: list[dict],
    delta_ranges: dict[str, list[tuple[int, int]]],
    *,
    context_lines: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Partition *suggestions* into (in_delta, out_of_delta) by amendment delta.

    *delta_ranges* is the amendment's NEW-side changed line ranges, keyed by
    new file path (the shape :func:`orchestrator.git_ops.parse_diff_added_line_ranges`
    produces).

    A suggestion is **in-delta** iff its location's file is a key of
    *delta_ranges* AND (the location is file-level — no ``:line`` — OR its line
    falls within one of that file's ranges widened by ``context_lines`` on each
    side).  Everything else — a different file, a line outside every range, or
    an unparseable/empty location — is **out-of-delta**.

    Input order is preserved within each partition.  Never raises on malformed
    locations.
    """
    in_delta: list[dict] = []
    out_of_delta: list[dict] = []
    for s in suggestions:
        parsed = _parse_location(s.get('location'))
        if parsed is None:
            out_of_delta.append(s)
            continue
        file_path, line = parsed
        ranges = delta_ranges.get(file_path)
        if ranges is None:
            # File not touched by the amendment.
            out_of_delta.append(s)
            continue
        if line is None:
            # File-level location on a touched file — in delta by construction.
            in_delta.append(s)
            continue
        if _line_in_ranges(line, ranges, context_lines):
            in_delta.append(s)
        else:
            out_of_delta.append(s)
    return in_delta, out_of_delta
