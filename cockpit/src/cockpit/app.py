"""cockpit.app — CockpitApp: the C5a TUI skeleton (session table + detail pane + poll refresh).

Fleet Cockpit C5a (plans/fleet-cockpit-prd.md §9). Scope: session table +
detail pane + poll refresh only -- the decision queue, keybindings, and
spawn bar are C5b.
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

    def compose(self) -> ComposeResult:
        yield Horizontal(
            SessionTable(id='session-table'),
            DetailPane(id='detail'),
        )

    def on_mount(self) -> None:
        self.refresh_registry()
        self.set_interval(self.poll_interval, self.refresh_registry)

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
        object identity a stale event might carry.
        """
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
