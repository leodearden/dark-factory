"""cockpit.panes.weight_editor — the `w` in-cockpit weight editor (Fleet Cockpit C9b, PRD §9).

merge_weight_edits and known_projects are pure helpers -- fast/deterministic
to unit test directly (no pilot, no event loop). merge_weight_edits parses
raw (Input-widget) strings into a new Priorities via dataclasses.replace,
touching ONLY category_weights/project_weights. known_projects mirrors
decision_queue.known_project_roots' "sorted distinct candidates, fail-soft
on empties" convention, but over each record's/decision's .project rather
than .cwd. WeightEditorScreen(ModalScreen) is the editor widget itself --
mirroring spawn_bar.SpawnScreen, it only ever calls its injected on_submit
callback (app.apply_priorities in production) with a new Priorities built
via merge_weight_edits; it never persists to priorities.yaml or touches
the DecisionQueue directly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace

from orchestrator.session_registry import DecisionRecord, SessionRecord
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label

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


class WeightEditorScreen(ModalScreen[None]):
    """The 'w' weight editor (PRD §9 C9b) -- edit category/project weights in-cockpit.

    A minimal modal form -- one labeled Input per existing category weight
    (id ``cat-<name>``) and one per known project name (id ``proj-<name>``),
    each seeded with its current value: a category Input is seeded directly
    from *priorities*.category_weights (every entry there already has a
    value); a project Input is seeded from priorities.project_weights.get
    (name, priorities.defaults.project) -- the same fallback score() itself
    uses (see priority.score), so an as-yet-unweighted known project shows
    its real effective weight, not a fake placeholder. Submitting (the
    button, or Enter in any Input) reads every Input's current value into
    category_edits/project_edits, builds a new Priorities via the pure
    merge_weight_edits, dismisses, then calls the injected *on_submit* with
    that new Priorities -- this widget never persists to priorities.yaml or
    touches the DecisionQueue itself; that's app.apply_priorities' job (see
    cockpit.app.CockpitApp). Escape dismisses without submitting.
    """

    DEFAULT_CSS = """
    WeightEditorScreen {
        align: center middle;
    }

    WeightEditorScreen > Vertical {
        width: 60;
        height: auto;
        border: round $accent;
        padding: 1 2;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding('escape', 'cancel', 'Cancel', show=False),
    ]

    def __init__(
        self,
        priorities: Priorities,
        project_names: Sequence[str],
        on_submit: Callable[[Priorities], None],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._priorities = priorities
        self._project_names = list(project_names)
        self._on_submit = on_submit

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label('Edit priority weights')
            yield Label('Category weights')
            for name, value in self._priorities.category_weights.items():
                yield Input(value=str(value), placeholder=name, id=f'cat-{name}')
            yield Label('Project weights')
            for name in self._project_names:
                current = self._priorities.project_weights.get(name, self._priorities.defaults.project)
                yield Input(value=str(current), placeholder=name, id=f'proj-{name}')
            yield Button('Apply', id='weight-submit', variant='primary')

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == 'weight-submit':
            self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def _submit(self) -> None:
        category_edits = {
            name: self.query_one(f'#cat-{name}', Input).value
            for name in self._priorities.category_weights
        }
        project_edits = {
            name: self.query_one(f'#proj-{name}', Input).value for name in self._project_names
        }
        new_priorities = merge_weight_edits(
            self._priorities, category_edits=category_edits, project_edits=project_edits
        )
        self.dismiss()
        self._on_submit(new_priorities)

    def action_cancel(self) -> None:
        self.dismiss()
