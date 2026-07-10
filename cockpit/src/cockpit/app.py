"""cockpit.app — CockpitApp: the C5a TUI skeleton (session table + detail pane + poll refresh).

Fleet Cockpit C5a (plans/fleet-cockpit-prd.md §9). Scope: session table +
detail pane + poll refresh only -- the decision queue, keybindings, and
spawn bar are C5b.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from textual.app import App, ComposeResult
from textual.containers import Horizontal

from cockpit.panes.detail_pane import DetailPane
from cockpit.panes.session_table import SessionTable, order_sessions
from cockpit.registry_reader import scan_sessions


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

    def compose(self) -> ComposeResult:
        yield Horizontal(
            SessionTable(id='session-table'),
            DetailPane(id='detail'),
        )

    def on_mount(self) -> None:
        self.refresh_registry()

    def refresh_registry(self) -> None:
        """Scan the registry and (re)populate the SessionTable."""
        records = scan_sessions(self.fleet_root)
        ordered = order_sessions(records)
        table = self.query_one('#session-table', SessionTable)
        table.replace_rows(ordered, self._now_fn())
