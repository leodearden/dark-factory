"""Tests for dashboard.data.outcome_colors — outcome-family color assignment."""

from __future__ import annotations

from dashboard.data.outcome_colors import assign_outcome_colors, classify_outcome

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


# ---------------------------------------------------------------------------
# Step-3: unknown/unmapped-code fallback and determinism
# ---------------------------------------------------------------------------

# The frontend defensive fallback from charts.jsx PALETTE.accent.
_CP_ACCENT = 'oklch(0.72 0.14 230)'

# First shade from each named family — a genuinely-new code must not collide.
_FAMILY_BASE_COLORS = {
    'oklch(0.74 0.14 155)',  # terminal_success[0]
    'oklch(0.70 0.20 15)',   # cas[0]
    'oklch(0.78 0.16 55)',   # plan[0]
    'oklch(0.82 0.14 80)',   # post_merge[0]
    'oklch(0.62 0.20 305)',  # train[0]
    'oklch(0.68 0.18 25)',   # terminal_failure[0]
}


def test_unknown_code_not_accent_and_not_family_base():
    """A genuinely-new code returns a color distinct from accent and family bases."""
    colors = assign_outcome_colors(['brand_new_code_xyz'])
    assert len(colors) == 1
    color = colors[0]
    assert color != _CP_ACCENT, f"Unknown code must not collide with CP.accent, got {color!r}"
    for base in _FAMILY_BASE_COLORS:
        assert color != base, f"Unknown code collides with family base {base!r}"


def test_two_distinct_unknown_codes_get_distinct_colors():
    """Two different unmapped codes in the same input get distinct colors."""
    labels = ['done', 'brand_new_a', 'brand_new_b']
    done_c, a_c, b_c = assign_outcome_colors(labels)
    assert a_c != b_c, "Two distinct unmapped codes must map to different colors"
    assert a_c != done_c, "Unmapped code must differ from known-family 'done'"
    assert b_c != done_c, "Unmapped code must differ from known-family 'done'"


def test_assign_outcome_colors_is_deterministic():
    """Same input always yields the same output (pure function, no randomness)."""
    labels = ['done', 'brand_new_a', 'cas_retry', 'conflict']
    assert assign_outcome_colors(labels) == assign_outcome_colors(labels)


def test_single_known_code_is_stable():
    """The same single-element list maps to the same color across calls."""
    assert assign_outcome_colors(['done']) == assign_outcome_colors(['done'])


def test_empty_input_returns_empty_list():
    """assign_outcome_colors([]) returns []."""
    assert assign_outcome_colors([]) == []


def test_classify_outcome_unknown_literal():
    """The literal string 'unknown' routes to the 'unknown' family."""
    assert classify_outcome('unknown') == 'unknown'


def test_classify_outcome_genuinely_new_code():
    """A code not in the map routes to the 'unknown' family."""
    assert classify_outcome('some_brand_new_code_xyz') == 'unknown'


# ---------------------------------------------------------------------------
# Step-5: comprehensive coverage — every enumerated code maps to a real family,
# and the realistic ordered label set has no adjacent color collisions.
# ---------------------------------------------------------------------------

# Full enumerated code set from the orchestrator, grouped by family.
_ALL_ENUMERATED_CODES: dict[str, list[str]] = {
    'terminal_success': [
        'done', 'already_merged', 'requeued',
        'done_wip_recovery', 'bypass_done', 'done_provenance',
    ],
    'cas': ['cas_retry', 'cas_exhausted', 'cas_failed'],
    'plan': [
        'dropped_plan_targets', 'plan_files_narrowed',
        'plan_files_not_touched', 'plan_blast_radius_lock_conflict',
    ],
    'post_merge': [
        'post_merge_equivalence_failed', 'post_merge_pyright_broken',
        'post_rebase_verify_fail',
    ],
    'train': ['train_incomplete', 'train_rebase_conflict', 'train_partial_flip'],
    'terminal_failure': [
        'blocked', 'conflict', 'merge_failed', 'verify_failed',
        'advance_failed', 'abandoned_verify_timeouts', 'unknown_branch',
        'not_descendant', 'contaminated', 'stash_failed',
    ],
}

# Canonical codes always emitted first (in this order) by outcome_distribution.
_CANONICAL_ORDER = ['done', 'conflict', 'blocked', 'already_merged']


def _build_realistic_label_list() -> list[str]:
    """Reproduce outcome_distribution ordering: canonical first, then sorted."""
    canonical_set = set(_CANONICAL_ORDER)
    all_codes = [c for codes in _ALL_ENUMERATED_CODES.values() for c in codes]
    remaining = sorted(c for c in all_codes if c not in canonical_set)
    return _CANONICAL_ORDER + remaining


def test_every_enumerated_code_classifies_to_known_family():
    """No enumerated outcome code falls through to the 'unknown' fallback."""
    for family, codes in _ALL_ENUMERATED_CODES.items():
        for code in codes:
            got = classify_outcome(code)
            assert got != 'unknown', (
                f"Code {code!r} (expected family {family!r}) fell through to 'unknown'"
            )
            assert got == family, (
                f"Code {code!r} classified as {got!r}, expected {family!r}"
            )


def test_realistic_label_set_no_adjacent_color_collision():
    """The realistic outcome_distribution label ordering has no adjacent color collision."""
    labels = _build_realistic_label_list()
    colors = assign_outcome_colors(labels)
    assert len(colors) == len(labels)
    for i in range(len(colors) - 1):
        assert colors[i] != colors[i + 1], (
            f"Adjacent color collision at positions {i} ({labels[i]!r}) "
            f"and {i+1} ({labels[i+1]!r}): {colors[i]!r}"
        )
