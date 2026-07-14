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

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace

from orchestrator.session_registry import DecisionRecord, SessionRecord
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label

from cockpit.priority import Priorities


def _parse_weight(raw: str) -> float | None:
    """Parse *raw* as a finite float, or None if unparseable/empty/non-finite.

    float() itself accepts 'inf'/'-inf'/'nan', which priority.score() cannot
    tolerate: an inf project/category weight makes raw=inf and
    urgency=inf/(1+inf)=nan, and a nan weight makes urgency nan outright --
    nan comparisons are undefined, so the DecisionQueue sort order would
    become non-deterministic garbage for that item. Fail-soft (PRD §2) means
    REJECTING a bad edit, not admitting a non-finite one, so this is the one
    shared parse+finite-check site for both merge_weight_edits loops and
    WeightEditorScreen._submit's seeded-value comparison below.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


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
    WeightEditorScreen Input) is parsed via the shared _parse_weight
    (float() plus a finite check); an unparseable, empty, or non-finite
    ('inf'/'-inf'/'nan') value is fail-soft skipped (PRD §2), keeping
    *base*'s existing value for that key rather than raising or clearing it
    -- a weight is cleared by typing 0, not by blanking a field.
    """
    category_weights = dict(base.category_weights)
    for name, raw in category_edits.items():
        value = _parse_weight(raw)
        if value is None:
            continue
        category_weights[name] = value

    project_weights = dict(base.project_weights)
    for name, raw in project_edits.items():
        value = _parse_weight(raw)
        if value is None:
            continue
        project_weights[name] = value

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
    (id ``cat-<index>``) and one per known project name (id
    ``proj-<index>``), each seeded with its current value: a category Input
    is seeded directly from *priorities*.category_weights (every entry
    there already has a value); a project Input is seeded from
    priorities.project_weights.get(name, priorities.defaults.project) --
    the same fallback score() itself uses (see priority.score), so an
    as-yet-unweighted known project shows its real effective weight, not a
    fake placeholder. Ids are positional (enumerate index), not derived from
    the category/project name itself: names are arbitrary
    operator-editable strings (priorities.yaml keys / on-disk project
    names) that may contain '.', '/', ':' or spaces, which Textual's
    identifier rules reject (BadIdentifier) -- a name-derived id would crash
    the whole screen on compose, violating the cockpit's fail-soft
    invariant (PRD §2). compose() and _submit() enumerate the same
    sequences (self._priorities.category_weights / self._project_names) in
    the same order, so the index correspondence is stable within one screen
    instance. self._project_names is additionally deduped in __init__
    (order-preserving): _submit()'s project_edits dict is keyed by NAME, not
    index, so a duplicate project name would otherwise render two
    indistinguishable rows and let the later-index row's read silently
    clobber the earlier one's entry. known_projects() (the production
    caller) already returns a deduped sorted union, so this only guards a
    directly-constructed screen -- e.g. in tests, or a future caller --
    against passing a non-deduped Sequence.

    Category rows are, deliberately, NOT symmetric with project rows: a
    project row is seeded from known_projects, a union of
    project_weights' own keys with project names observed on live
    records/decisions, so an as-yet-unweighted-but-active project still
    gets an editable row. There is no analogous union for categories today
    -- SessionRecord/DecisionRecord carry no per-item category field yet
    (decision_queue.decision_to_scoring_item/session_to_scoring_item
    hardcode category='' -- see those docstrings), so there is no live
    "categories observed on records/decisions" set to union against.
    Category rows are therefore seeded ONLY from priorities.category_weights'
    own keys: a category scoring purely via defaults.category (no explicit
    priorities.yaml entry) has no row here and isn't editable from this
    screen -- add it to priorities.yaml directly to expose it.

    Submitting (the button, or Enter in any Input) reads every
    Input's current value into category_edits/project_edits -- an edit is
    only included when it differs from that Input's seeded value, so
    accepting the defaults without touching any field never materializes
    an explicit priorities.yaml entry for it -- builds a new Priorities via
    the pure merge_weight_edits, dismisses, then calls the injected
    *on_submit* with that new Priorities -- this widget never persists to
    priorities.yaml or touches the DecisionQueue itself; that's
    app.apply_priorities' job (see cockpit.app.CockpitApp). Escape
    dismisses without submitting.
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
        # dict.fromkeys dedupes while preserving first-seen order. Defensive:
        # known_projects() (the production caller) always returns an
        # already-deduped sorted union, but this screen accepts an arbitrary
        # Sequence[str] directly (and is constructed with raw, possibly
        # non-deduped, lists in tests) -- see the class docstring's
        # "self._project_names is additionally deduped" note for why a
        # duplicate would otherwise corrupt _submit()'s name-keyed edits.
        self._project_names = list(dict.fromkeys(project_names))
        self._on_submit = on_submit

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label('Edit priority weights')
            yield Label('Category weights')
            # Only pre-existing category_weights keys get a row -- there is
            # no known_projects-style union for categories; see the class
            # docstring's "NOT symmetric with project rows" note.
            for index, (name, value) in enumerate(self._priorities.category_weights.items()):
                yield Input(value=str(value), placeholder=name, id=f'cat-{index}')
            yield Label('Project weights')
            for index, name in enumerate(self._project_names):
                current = self._priorities.project_weights.get(
                    name, self._priorities.defaults.project
                )
                yield Input(value=str(current), placeholder=name, id=f'proj-{index}')
            yield Button('Apply', id='weight-submit', variant='primary')

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == 'weight-submit':
            self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def _submit(self) -> None:
        """Read every Input, merge, dismiss, then hand off to *on_submit*.

        A category or project Input holding an unparseable/empty/non-finite
        value is fail-soft dropped by merge_weight_edits (base's prior value
        is kept, never raised -- see its docstring). That drop is currently
        SILENT: dismiss() (below) tears this screen down immediately after,
        before the operator could see any re-seeded/restyled Input, so
        visible feedback for a rejected edit would need to land somewhere
        that outlives the modal (e.g. an app-level notification) rather
        than on the Input itself -- a new Textual pattern with no existing
        precedent in this codebase. Reviewed and intentionally deferred as
        a UX nicety, not a correctness gap: see
        test_submit_keeps_prior_value_for_a_rejected_category_edit in
        test_app.py for the pinned current behavior.
        """
        category_edits: dict[str, str] = {}
        for index, name in enumerate(self._priorities.category_weights):
            raw = self.query_one(f'#cat-{index}', Input).value
            seeded = self._priorities.category_weights[name]
            value = _parse_weight(raw)
            if value is None or value == seeded:
                continue
            category_edits[name] = raw

        project_edits: dict[str, str] = {}
        for index, name in enumerate(self._project_names):
            raw = self.query_one(f'#proj-{index}', Input).value
            seeded = self._priorities.project_weights.get(name, self._priorities.defaults.project)
            value = _parse_weight(raw)
            if value is None or value == seeded:
                continue
            project_edits[name] = raw
        new_priorities = merge_weight_edits(
            self._priorities, category_edits=category_edits, project_edits=project_edits
        )
        self.dismiss()
        self._on_submit(new_priorities)

    def action_cancel(self) -> None:
        self.dismiss()
