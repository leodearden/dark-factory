"""Which task branches can ride one merge train, judged by their changed line ranges.

Pure functions over ``{file: [(start, end), ...]}`` maps of lines changed
relative to main. The workflow's train former and the worker's coalesce pass
both choose members with :func:`select_train_members`.
"""


def line_ranges_stackable(
    ranges_a: dict[str, list[tuple[int, int]]],
    ranges_b: dict[str, list[tuple[int, int]]],
) -> bool:
    """Return True iff tasks A and B are line-level stackable.

    Two tasks are stackable iff no file they both touch has overlapping changed
    line ranges relative to BASE (main).  Crate/file-disjointness is NOT
    required for stackability — same file, different lines is fine (PRD §A.2).

    Uses closed-interval intersection: range (s1,e1) intersects (s2,e2)
    iff s1 <= e2 and s2 <= e1.
    """
    shared_files = set(ranges_a) & set(ranges_b)
    for fname in shared_files:
        for s1, e1 in ranges_a[fname]:
            for s2, e2 in ranges_b[fname]:
                if s1 <= e2 and s2 <= e1:
                    return False
    return True


def select_train_members(
    anchor_id: str,
    candidate_ids: list[str],
    ranges_by_id: dict[str, dict[str, list[tuple[int, int]]]],
    max_members: int,
) -> list[str]:
    """Select a mutually-stackable subset of candidates capped at max_members.

    Greedy selection: start with the anchor, then iterate candidates in
    deterministic (id-sorted) order, adding each that is line_ranges_stackable
    against ALL already-selected members.  Stops when len(selected) == max_members.

    Returns [] when the result has fewer than 2 members — the sentinel for
    "no viable train" (a single-member train is meaningless).

    The anchor is always first (order-0) in the returned list.
    """
    selected: list[str] = [anchor_id]

    for candidate_id in sorted(candidate_ids):
        if len(selected) >= max_members:
            break
        candidate_ranges = ranges_by_id.get(candidate_id, {})
        # Must be stackable with every already-selected member (mutual stackability).
        stackable_with_all = all(
            line_ranges_stackable(candidate_ranges, ranges_by_id.get(sel_id, {}))
            for sel_id in selected
        )
        if stackable_with_all:
            selected.append(candidate_id)

    # A single-member "train" is a no-op — return the empty sentinel.
    if len(selected) < 2:
        return []
    return selected
