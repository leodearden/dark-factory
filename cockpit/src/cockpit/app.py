"""cockpit.app — CockpitApp: the C5a/C5b TUI (decision queue + session table + detail pane).

Fleet Cockpit C5a+C5b (plans/fleet-cockpit-prd.md §9). C5a scope: session
table + detail pane + poll refresh. C5b extends in place: a score-ordered
decision queue, prioritization/focus keybindings, a spawn bar, and the
explicit-action focus wiring that enforces "signal-don't-move" (PRD §6.2/B3)
-- the refresh/diff path may call only backend.set_urgency/reorder, never
backend.focus/tile.

Pure-consumer discipline (PRD §2/§5 hard invariant): every refresh/scan/
select code path here is strictly READ-ONLY over the session and decision
registries -- nothing in the refresh path calls write_record/write_decision/
update_decision_state/set_manual_boost. The cockpit's ONLY unconditional
write target is its own cockpit-ui.json (via cockpit.ui_config); C5b's
explicit-action keybindings (boost/drop) add sanctioned, action-only writes
to a DECISION's manual_boost/state via C1's set_manual_boost/
update_decision_state -- see test_app.py's TestWriteDiscipline for the
end-to-end proof of the refresh-path half of this contract.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from orchestrator.session_registry import DecisionRecord, SessionRecord, list_decisions
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable

from cockpit.backends import FocusArrangeBackend, TmuxBackend, WmBackend
from cockpit.panes.decision_queue import DecisionQueue, order_queue
from cockpit.panes.detail_pane import DetailPane
from cockpit.panes.session_table import SessionTable, order_sessions
from cockpit.priority import Priorities, load_priorities
from cockpit.registry_reader import build_snapshot, scan_sessions, snapshot_changed
from cockpit.ui_config import CockpitUIConfig, load_ui_config, save_ui_config

# Decision fields that feed the queue's scoring/display -- excludes
# escalation_id/options (order_queue/format_queue_row never read them), so a
# change to those never triggers a rebuild. Mirrors registry_reader's
# _SNAPSHOT_FIELDS convention: keyed for a cheap equality diff, not identity.
_DECISION_SNAPSHOT_FIELDS = (
    'project',
    'text',
    'filed_at',
    'session_id',
    'task_id',
    'manual_boost',
    'state',
)


def _decisions_snapshot(decisions: list[DecisionRecord]) -> dict[str, tuple]:
    """Map decision id -> a hashable tuple of its queue-relevant fields.

    Mirrors registry_reader.build_snapshot for SessionRecords: lets
    refresh_registry's poll-tick short-circuit account for a decision being
    filed/boosted/dropped by another process (e.g. a C8 watcher), not just a
    session state change, without rebuilding on every purely-time-passing
    tick.
    """
    return {
        decision.id: tuple(getattr(decision, field) for field in _DECISION_SNAPSHOT_FIELDS)
        for decision in decisions
    }


class CockpitApp(App):
    """Fleet Cockpit TUI: decision queue + session table + detail pane, polling for changes."""

    def __init__(
        self,
        *,
        fleet_root: Path | str | None = None,
        poll_interval: float = 1.5,
        now_fn: Callable[[], datetime] | None = None,
        backend: FocusArrangeBackend | None = None,
        spawn_runner: Callable[[list[str]], None] | None = None,
        spawn_script: Path | str | None = None,
        priorities: Priorities | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.fleet_root = fleet_root
        self.poll_interval = poll_interval
        self._now_fn = now_fn if now_fn is not None else lambda: datetime.now(UTC)
        self._backend = backend
        self._default_backends: dict[str, FocusArrangeBackend] = {
            'wm': WmBackend(),
            'tmux': TmuxBackend(),
        }
        self._spawn_runner = spawn_runner
        self._spawn_script = spawn_script
        self._priorities = priorities if priorities is not None else load_priorities()
        self._records: list[SessionRecord] = []
        self._decisions: list[DecisionRecord] = []
        self._snapshot: dict[str, tuple] = {}
        self._decisions_snapshot: dict[str, tuple] = {}
        self._has_scanned = False
        self._selected_slug: str | None = None

    def compose(self) -> ComposeResult:
        yield Vertical(
            DecisionQueue(id='decision-queue'),
            Horizontal(
                SessionTable(id='session-table'),
                DetailPane(id='detail'),
            ),
        )

    def on_mount(self) -> None:
        selected_slug = load_ui_config(self.fleet_root).selected_slug
        self.refresh_registry()
        if selected_slug is not None:
            self.query_one('#session-table', SessionTable).select_slug(selected_slug)
        self.set_interval(self.poll_interval, self.refresh_registry)

    def on_unmount(self) -> None:
        self._persist_ui_config()

    def refresh_registry(self) -> None:
        """Scan the registry and rebuild the SessionTable/DecisionQueue only when something changed.

        The in-memory snapshot diff (build_snapshot/snapshot_changed plus
        _decisions_snapshot) keys on substantive fields only (never
        start_ts/age), so a purely-time-passing poll tick is a no-op -- no
        flicker, and each table's own replace_rows() re-locates the
        previously-highlighted key so the cursor survives a rebuild.
        """
        records = scan_sessions(self.fleet_root)
        decisions = list_decisions(self.fleet_root)
        new_snapshot = build_snapshot(records)
        new_decisions_snapshot = _decisions_snapshot(decisions)
        if (
            self._has_scanned
            and not snapshot_changed(self._snapshot, new_snapshot)
            and new_decisions_snapshot == self._decisions_snapshot
        ):
            return
        self._has_scanned = True
        self._snapshot = new_snapshot
        self._decisions_snapshot = new_decisions_snapshot
        self._decisions = decisions
        self._records = order_sessions(records)
        table = self.query_one('#session-table', SessionTable)
        table.replace_rows(self._records, self._now_fn())
        self._sync_detail_pane(table.highlighted_slug())

        now = self._now_fn()
        queue_items = order_queue(decisions, records, self._priorities, now)
        queue = self.query_one('#decision-queue', DecisionQueue)
        queue.replace_rows(queue_items, now)

    def _sync_detail_pane(self, slug: str | None) -> None:
        """Render *slug*'s record (or the empty placeholder) into the detail pane.

        Looked up against self._records -- the ordered set from the most
        recent scan -- so this always reflects current data, not whatever
        object identity a stale event might carry. Also remembers *slug* for
        _persist_ui_config, since on_unmount runs after the DataTable itself
        has already been torn down and can no longer be queried.
        """
        self._selected_slug = slug
        record = next((r for r in self._records if r.session_slug == slug), None)
        detail = self.query_one('#detail', DetailPane)
        detail.show_record(record, self._records, self._now_fn())

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Keep the detail pane in sync with the SessionTable's highlighted row.

        Covers interactive cursor moves (e.g. arrow keys, or a test/caller
        calling move_cursor directly). The complementary rebuild-time sync
        lives in refresh_registry -- clear()'s cursor reset only reposts
        this message when the highlighted row index actually changes, so a
        same-row-different-content rebuild needs its own explicit sync.

        SessionTable and DecisionQueue are both DataTable subclasses, so
        Textual routes both their RowHighlighted messages through this same
        handler (dispatch is by the base DataTable message namespace, not
        the subclass) -- event.data_table disambiguates so moving the
        DecisionQueue's cursor never clobbers the session detail sync.
        """
        table = self.query_one('#session-table', SessionTable)
        if event.data_table is not table:
            return
        self._sync_detail_pane(event.row_key.value)
        self._persist_ui_config()

    def _persist_ui_config(self) -> None:
        """Write the cockpit's own UI state -- its ONLY write target (PRD §2/§5).

        Persists just enough to restore the operator's place across a
        restart (selected_slug) plus the poll_interval currently in effect.
        Never touches sessions/ or decisions/. Reads self._selected_slug
        (not the DataTable) so this is safe to call from on_unmount, after
        the table has already been torn down.
        """
        cfg = CockpitUIConfig(selected_slug=self._selected_slug, poll_interval=self.poll_interval)
        save_ui_config(cfg, self.fleet_root)
