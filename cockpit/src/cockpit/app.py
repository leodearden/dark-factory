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
