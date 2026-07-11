"""cockpit.panes.weight_editor — pure weight-edit merge logic (Fleet Cockpit C9b, PRD §9).

merge_weight_edits is pure edit-merge logic -- fast/deterministic to unit
test directly (no pilot, no event loop): it parses raw (Input-widget)
strings into a new Priorities via dataclasses.replace, touching ONLY
category_weights/project_weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from cockpit.priority import Priorities


def merge_weight_edits(
    base: Priorities,
    *,
    category_edits: Mapping[str, str],
    project_edits: Mapping[str, str],
) -> Priorities:
    """Build a new Priorities from *base* with *category_edits*/*project_edits* applied.

    Only category_weights/project_weights are touched (via
    dataclasses.replace) -- severity_weights/defaults/age_curve/manual_boost
    pass through from *base* unchanged. Each raw edit string (as read from a
    WeightEditorScreen Input) is parsed with float(); an unparseable or
    empty value is fail-soft skipped (PRD §2), keeping *base*'s existing
    value for that key rather than raising or clearing it -- a weight is
    cleared by typing 0, not by blanking a field.
    """
    category_weights = dict(base.category_weights)
    for name, raw in category_edits.items():
        try:
            category_weights[name] = float(raw)
        except (TypeError, ValueError):
            continue

    project_weights = dict(base.project_weights)
    for name, raw in project_edits.items():
        try:
            project_weights[name] = float(raw)
        except (TypeError, ValueError):
            continue

    return replace(base, category_weights=category_weights, project_weights=project_weights)
