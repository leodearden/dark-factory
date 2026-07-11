"""cockpit.panes.weight_editor — pure weight-edit merge logic (Fleet Cockpit C9b, PRD §9).

merge_weight_edits and known_projects are pure helpers -- fast/deterministic
to unit test directly (no pilot, no event loop). merge_weight_edits parses
raw (Input-widget) strings into a new Priorities via dataclasses.replace,
touching ONLY category_weights/project_weights. known_projects mirrors
decision_queue.known_project_roots' "sorted distinct candidates, fail-soft
on empties" convention, but over each record's/decision's .project rather
than .cwd.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace

from orchestrator.session_registry import DecisionRecord, SessionRecord

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


def known_projects(
    records: Sequence[SessionRecord],
    decisions: Sequence[DecisionRecord],
    existing: Iterable[str] = (),
) -> list[str]:
    """Sorted distinct project-name candidates for the weight editor's project fields.

    Unions each SessionRecord's/DecisionRecord's own .project with
    *existing* (typically the current Priorities.project_weights' own
    keys, so an already-weighted project is always offered even if it
    currently has no live session/decision). Deduped and sorted for a
    stable, deterministic picker order -- mirrors known_project_roots.
    Fail-soft (PRD §2): an empty/falsy project name is simply excluded,
    never raises.
    """
    projects = {record.project for record in records if record.project}
    projects.update(decision.project for decision in decisions if decision.project)
    projects.update(name for name in existing if name)
    return sorted(projects)
