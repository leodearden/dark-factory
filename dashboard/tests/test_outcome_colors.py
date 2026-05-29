"""Tests for dashboard.data.outcome_colors — outcome-family color assignment."""

from __future__ import annotations

from dashboard.data.outcome_colors import assign_outcome_colors


# ---------------------------------------------------------------------------
# Step-1: reify leaf-signal — five cross-family codes all get distinct colors
# ---------------------------------------------------------------------------

# outcome_distribution sorts non-canonical codes alphabetically; the five reify
# codes are all non-canonical so they appear sorted:
REIFY_LABELS = [
    'cas_retry',
    'dropped_plan_targets',
    'plan_files_narrowed',
    'plan_files_not_touched',
    'post_merge_equivalence_failed',
]


def test_reify_colors_length():
    """assign_outcome_colors returns one color per label."""
    colors = assign_outcome_colors(REIFY_LABELS)
    assert len(colors) == len(REIFY_LABELS)


def test_reify_colors_all_distinct():
    """All five reify codes map to pairwise-distinct colors."""
    colors = assign_outcome_colors(REIFY_LABELS)
    assert len(set(colors)) == len(REIFY_LABELS), (
        f"Expected 5 distinct colors, got {len(set(colors))}: {colors}"
    )


def test_reify_no_adjacent_collision():
    """No two adjacent colors are equal (visual slice separation)."""
    colors = assign_outcome_colors(REIFY_LABELS)
    for i in range(len(colors) - 1):
        assert colors[i] != colors[i + 1], (
            f"Adjacent colors equal at position {i} and {i+1}: {colors[i]!r}"
        )
