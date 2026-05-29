"""Outcome-family color assignment for the merge-queue doughnut chart.

Colors are resolved server-side and shipped in the MERGE_QUEUE payload as
``outcomes.colors`` (parallel to ``outcomes.labels``).  This keeps all color
logic in tested Python rather than in the JS layer, and guarantees that the
donut slice (tabs.jsx:880) and the legend swatch (tabs.jsx:884) always agree.

Design
------
Each outcome code belongs to a *semantic family*.  Every family has a list of
distinct oklch shade strings whose hues mirror ``charts.jsx`` PALETTE
conventions:

  terminal_success  green   ~155
  post_merge        yellow  ~80
  terminal_failure  red     ~25
  train             violet  ~305
  cas               crimson ~15   (distinct from terminal_failure red)
  plan              orange  ~55
  unknown           neutral ~250  (fallback for unmapped codes)

Within a family, shades differ in lightness so adjacent same-family slices are
visually distinguishable.  Cross-family slices differ by hue.  Together this
guarantees "no two adjacent slices share a color" by construction.

Unknown / genuinely-new codes are routed to the neutral ``unknown`` family
ramp and assigned by order-of-appearance, so multiple unmapped codes each get
a distinct color.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Shade lists — one oklch string per ordinal within each family.
# Vary lightness (and occasionally chroma) so consecutive shades always differ.
# terminal_failure needs >= 7 entries (many terminal codes).
# ---------------------------------------------------------------------------

_FAMILY_SHADES: dict[str, list[str]] = {
    # Terminal success — greens
    'terminal_success': [
        'oklch(0.74 0.14 155)',
        'oklch(0.65 0.17 155)',
        'oklch(0.82 0.11 155)',
        'oklch(0.56 0.19 155)',
        'oklch(0.88 0.09 155)',
        'oklch(0.48 0.21 155)',
    ],
    # CAS — distinct crimson (different hue from terminal_failure)
    'cas': [
        'oklch(0.70 0.20 15)',
        'oklch(0.61 0.22 15)',
        'oklch(0.79 0.17 15)',
    ],
    # Plan targeting — orange
    'plan': [
        'oklch(0.78 0.16 55)',
        'oklch(0.68 0.19 55)',
        'oklch(0.86 0.13 55)',
        'oklch(0.58 0.21 55)',
    ],
    # Post-merge / post-rebase — yellow
    'post_merge': [
        'oklch(0.82 0.14 80)',
        'oklch(0.72 0.17 80)',
        'oklch(0.90 0.11 80)',
    ],
    # Train — violet
    'train': [
        'oklch(0.62 0.20 305)',
        'oklch(0.72 0.18 305)',
        'oklch(0.52 0.22 305)',
        'oklch(0.80 0.15 305)',
    ],
    # Terminal failure — red (needs >= 7 shades for the large code set)
    'terminal_failure': [
        'oklch(0.68 0.18 25)',
        'oklch(0.58 0.21 25)',
        'oklch(0.76 0.16 25)',
        'oklch(0.50 0.23 25)',
        'oklch(0.84 0.13 25)',
        'oklch(0.45 0.25 25)',
        'oklch(0.78 0.20 25)',
        'oklch(0.42 0.26 25)',
    ],
    # Unknown / fallback — neutral ramp.
    # Distinct from CP.accent 'oklch(0.72 0.14 230)' (different chroma & hue)
    # and from every family base color.  Multiple unmapped codes cycle here.
    'unknown': [
        'oklch(0.60 0.04 250)',
        'oklch(0.50 0.06 250)',
        'oklch(0.70 0.03 250)',
        'oklch(0.42 0.07 250)',
        'oklch(0.78 0.02 250)',
        'oklch(0.38 0.08 250)',
    ],
}

# ---------------------------------------------------------------------------
# Outcome-code → family mapping
# Covers every code emitted by the orchestrator (merge_queue.py + workflow.py).
# ---------------------------------------------------------------------------

_CODE_FAMILY: dict[str, str] = {
    # terminal_success
    'done': 'terminal_success',
    'already_merged': 'terminal_success',
    'requeued': 'terminal_success',
    'done_wip_recovery': 'terminal_success',
    'bypass_done': 'terminal_success',
    'done_provenance': 'terminal_success',
    # cas
    'cas_retry': 'cas',
    'cas_exhausted': 'cas',
    'cas_failed': 'cas',
    # plan
    'dropped_plan_targets': 'plan',
    'plan_files_narrowed': 'plan',
    'plan_files_not_touched': 'plan',
    'plan_blast_radius_lock_conflict': 'plan',
    # post_merge
    'post_merge_equivalence_failed': 'post_merge',
    'post_merge_pyright_broken': 'post_merge',
    'post_rebase_verify_fail': 'post_merge',
    # train
    'train_incomplete': 'train',
    'train_rebase_conflict': 'train',
    'train_partial_flip': 'train',
    # terminal_failure
    'blocked': 'terminal_failure',
    'conflict': 'terminal_failure',
    'merge_failed': 'terminal_failure',
    'verify_failed': 'terminal_failure',
    'advance_failed': 'terminal_failure',
    'abandoned_verify_timeouts': 'terminal_failure',
    'unknown_branch': 'terminal_failure',
    'not_descendant': 'terminal_failure',
    'contaminated': 'terminal_failure',
    'stash_failed': 'terminal_failure',
}


def classify_outcome(code: str) -> str:
    """Return the semantic family for *code*.

    Returns ``'unknown'`` for any code not in the enumerated set (including
    the literal ``'unknown'`` string itself).
    """
    return _CODE_FAMILY.get(code, 'unknown')


def assign_outcome_colors(labels: list[str]) -> list[str]:
    """Return one oklch color string for each label, in order.

    The function is a pure deterministic mapping of its input:

    * Each label is classified to a semantic family via :func:`classify_outcome`.
    * Within a family, shades are assigned by order-of-appearance ordinal
      (0, 1, 2, …), cycling through the family's shade list as needed.
    * Cross-family slices differ by hue; within-family adjacent shades differ
      by lightness — so no two adjacent slices share a color by construction.
    * Unmapped codes route to the ``'unknown'`` neutral ramp, also by
      order-of-appearance, so multiple unmapped codes get distinct colors.
    """
    family_ordinals: dict[str, int] = {}
    colors: list[str] = []
    for label in labels:
        family = classify_outcome(label)
        shades = _FAMILY_SHADES[family]
        ordinal = family_ordinals.get(family, 0)
        colors.append(shades[ordinal % len(shades)])
        family_ordinals[family] = ordinal + 1
    return colors
