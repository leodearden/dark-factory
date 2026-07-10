"""cockpit.app — CockpitApp: the C5a TUI skeleton (session table + detail pane + poll refresh).

Fleet Cockpit C5a (plans/fleet-cockpit-prd.md §9). Scope: session table +
detail pane + poll refresh only -- the decision queue, keybindings, and
spawn bar are C5b.

Pure-consumer discipline (PRD §2/§5 hard invariant): every refresh/scan/
select code path here is strictly READ-ONLY over the session and decision
registries -- nothing in this module calls write_record/write_decision/
update_decision_state/set_manual_boost. The cockpit's ONLY write target is
its own cockpit-ui.json (via cockpit.ui_config), used solely to remember
the last-selected session across restarts. See test_app.py's
TestWriteDiscipline for the end-to-end proof.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from orchestrator.session_registry import SessionRecord
from textual.app import App, ComposeResult
from textual.containers import Horizontal

from cockpit.panes.detail_pane import DetailPane
from cockpit.panes.session_table import SessionTable, order_sessions
from cockpit.registry_reader import build_snapshot, scan_sessions, snapshot_changed
from cockpit.ui_config import CockpitUIConfig, load_ui_config, save_ui_config


class CockpitApp(App):
    """Fleet Cockpit TUI: a session table + a detail pane, polling the registry for changes."""

    def __init__(
        self,
        *,
        fleet_root: Path | str | None = None,
        poll_interval: float = 1.5,
        now_fn: Callable[[], datetime] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.fleet_root = fleet_root
        self.poll_interval = poll_interval
        self._now_fn = now_fn if now_fn is not None else lambda: datetime.now(UTC)
        self._records: list[SessionRecord] = []
        self._snapshot: dict[str, tuple] = {}
        self._has_scanned = False
        self._selected_slug: str | None = None

    def compose(self) -> ComposeResult:
        yield Horizontal(
            SessionTable(id='session-table'),
            DetailPane(id='detail'),
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
        """Scan the registry and rebuild the SessionTable only when something changed.

        The in-memory snapshot diff (build_snapshot/snapshot_changed) keys on
        substantive fields only (never start_ts/age), so a purely-time-passing
        poll tick is a no-op -- no flicker, and the table's own
        replace_rows() re-locates the previously-highlighted slug so the
        cursor survives a rebuild.
        """
        records = scan_sessions(self.fleet_root)
        new_snapshot = build_snapshot(records)
        if self._has_scanned and not snapshot_changed(self._snapshot, new_snapshot):
            return
        self._has_scanned = True
        self._snapshot = new_snapshot
        self._records = order_sessions(records)
        table = self.query_one('#session-table', SessionTable)
        table.replace_rows(self._records, self._now_fn())
        self._sync_detail_pane(table.highlighted_slug())

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

    def on_data_table_row_highlighted(self, event: SessionTable.RowHighlighted) -> None:
        """Keep the detail pane in sync with the DataTable's highlighted row.

        Covers interactive cursor moves (e.g. arrow keys, or a test/caller
        calling move_cursor directly). The complementary rebuild-time sync
        lives in refresh_registry -- clear()'s cursor reset only reposts
        this message when the highlighted row index actually changes, so a
        same-row-different-content rebuild needs its own explicit sync.
        """
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
