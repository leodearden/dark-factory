"""cockpit.app — CockpitApp: the C5a/C5b TUI (decision queue + session table + detail pane).

Fleet Cockpit C5a+C5b (plans/fleet-cockpit-prd.md §9). C5a scope: session
table + detail pane + poll refresh. C5b extends in place: a score-ordered
decision queue, prioritization/focus keybindings, a spawn bar, and the
explicit-action focus wiring that enforces "signal-don't-move" (PRD §6.2/B3)
-- the refresh/diff path may call only backend.set_urgency/reorder, never
backend.focus/tile.

Pure-consumer discipline (PRD §2/§5 hard invariant): every refresh/scan/
select code path here is strictly READ-ONLY over the session and decision
registries -- nothing in refresh_registry, _rebuild_queue, or
_update_attention (the C5b queue-build/reorder/set_urgency path) calls
write_record/write_decision/update_decision_state/set_manual_boost. The
cockpit's ONLY unconditional write target is its own cockpit-ui.json (via
cockpit.ui_config); C5b's explicit-action keybindings (boost/drop) add
sanctioned, action-only writes to a DECISION's manual_boost/state via C1's
set_manual_boost/update_decision_state -- see test_app.py's
TestWriteDiscipline (the C5a session/detail path) and
TestRefreshWriteDiscipline (the C5b queue/attention path) for the
end-to-end proof of the refresh-path half of this contract.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from orchestrator.session_registry import (
    DecisionRecord,
    DecisionState,
    SessionRecord,
    list_decisions,
    set_manual_boost,
    update_decision_state,
)
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable

from cockpit.backends import DisplayTarget, FocusArrangeBackend, TmuxBackend, WmBackend
from cockpit.panes.decision_queue import (
    DecisionQueue,
    QueueItem,
    known_project_roots,
    order_queue,
    resolve_target,
)
from cockpit.panes.detail_pane import DetailPane
from cockpit.panes.session_table import SessionTable, order_sessions
from cockpit.panes.spawn_bar import SpawnScreen, build_spawn_argv
from cockpit.panes.spawn_tree import SpawnTreeScreen
from cockpit.priority import Priorities, load_priorities
from cockpit.registry_reader import (
    SessionScanner,
    SessionScannerProtocol,
    build_snapshot,
    snapshot_changed,
)
from cockpit.ui_config import CockpitUIConfig, load_ui_config, save_ui_config

_log = logging.getLogger(__name__)

# skills/spawn/spawn-claude.sh, relative to this repo checkout -- this file
# lives at <repo>/cockpit/src/cockpit/app.py, so the repo root is three
# parents up. Only used as a last-resort fallback (see _default_spawn_script);
# an installed cockpit package has no such repo layout at all.
_REPO_RELATIVE_SPAWN_SCRIPT = (
    Path(__file__).resolve().parents[3] / 'skills' / 'spawn' / 'spawn-claude.sh'
)


def _default_spawn_script() -> Path | None:
    """Best-effort resolution of spawn-claude.sh's path (PRD §9 C5b spawn bar).

    Prefers $CLAUDE_SPAWN_SCRIPT when set -- an installed cockpit package
    has no guaranteed relative path to a checkout's skills/ directory.
    Otherwise falls back to the path relative to this repo checkout, when
    it actually exists there. Degrades to None (fail-soft, PRD §2) when
    neither resolves -- spawn_session then simply no-ops rather than
    raising: a view/action is never a hard dependency.
    """
    override = os.environ.get('CLAUDE_SPAWN_SCRIPT')
    if override:
        return Path(override)
    if _REPO_RELATIVE_SPAWN_SCRIPT.is_file():
        return _REPO_RELATIVE_SPAWN_SCRIPT
    return None


def _default_spawn_runner(argv: list[str]) -> None:
    """Fire-and-forget default spawn runner: subprocess.Popen, fail-soft.

    Mirrors the cockpit's fail-soft discipline (PRD §2): a launch failure
    (missing/non-executable script, no terminal emulator found, etc.) is
    logged, never raised -- the cockpit itself must keep running regardless
    of whether a spawn attempt succeeded.
    """
    try:
        subprocess.Popen(argv)  # noqa: S603 -- argv is built by build_spawn_argv, not shell text
    except OSError:
        _log.exception('spawn_session: failed to launch %r', argv)


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

    BINDINGS = [
        # Fallback only: in normal operation Enter is consumed by whichever
        # DataTable (SessionTable or DecisionQueue) currently holds focus --
        # both bind 'enter' to their own select_cursor action, which posts
        # RowSelected and is resolved from the focused widget UP the DOM, so
        # it wins before this app-level binding is ever consulted (see
        # on_data_table_row_selected, which is what actually routes a
        # DecisionQueue selection to action_focus_selected). This entry only
        # fires when no DataTable holds focus at all.
        Binding('enter', 'focus_selected', 'Focus', show=False),
        Binding('b', 'boost', 'Boost', show=False),
        Binding('B', 'big_boost', 'Big boost', show=False),
        Binding('x', 'drop', 'Drop', show=False),
        Binding('d', 'defer', 'Defer', show=False),
        Binding('n', 'new_session', 'New session', show=False),
        Binding('t', 'toggle_tree', 'Spawn tree', show=False),
        # All ten digits are bound, but a digit's SCORE effect saturates
        # once it exceeds the active Priorities.manual_boost.max (default 5
        # -- see priority.py's score(), which clamps manual_boost into
        # [manual_boost.min, .max] before weighting it). Under the bundled
        # default config, '6'..'9' therefore reorder identically to '5'.
        # The persisted/in-memory boost still records the exact digit
        # pressed (see action_set_priority) -- only the resulting score
        # saturates.
        *(Binding(str(d), f'set_priority({d})', f'Priority {d}', show=False) for d in range(10)),
    ]

    _BOOST_STEP = 1
    _BIG_BOOST_STEP = 3

    def __init__(
        self,
        *,
        fleet_root: Path | str | None = None,
        poll_interval: float = 1.5,
        scanner: SessionScannerProtocol | None = None,
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
        self._scanner = scanner if scanner is not None else SessionScanner(self.fleet_root)
        self._now_fn = now_fn if now_fn is not None else lambda: datetime.now(UTC)
        self._backend = backend
        self._default_backends: dict[str, FocusArrangeBackend] = {
            'wm': WmBackend(),
            'tmux': TmuxBackend(),
        }
        self._spawn_runner = spawn_runner if spawn_runner is not None else _default_spawn_runner
        self._spawn_script = spawn_script if spawn_script is not None else _default_spawn_script()
        self._priorities = priorities if priorities is not None else load_priorities()
        self._records: list[SessionRecord] = []
        self._decisions: list[DecisionRecord] = []
        self._snapshot: dict[str, tuple] = {}
        self._decisions_snapshot: dict[str, tuple] = {}
        self._has_scanned = False
        self._selected_slug: str | None = None
        # Keyed by resolved DisplayTarget, NOT by item key -- see
        # _update_attention's docstring: a decision and the AWAITING_INPUT
        # session it links to can resolve to the SAME target, so urgency
        # must be a function of "is any live item still pointing here", not
        # of any one item's key.
        self._attention_targets: set[DisplayTarget] = set()
        self._queue_items_by_key: dict[str, QueueItem] = {}
        # In-memory "already acted on" marker, set by action_focus_selected.
        # Pruned to the live queue on every rebuild (_rebuild_queue) so a
        # key never keeps reporting "handling" past the ask it was set for.
        self._handling: set[str] = set()
        self._boosts: dict[str, int] = {}
        self._dropped: set[str] = set()
        self._deferred: dict[str, datetime] = {}

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
        self.set_interval(self.poll_interval, self._poll_registry)

    def on_unmount(self) -> None:
        self._persist_ui_config()

    def refresh_registry(self) -> None:
        """Scan the registry and rebuild the SessionTable/DecisionQueue, synchronously.

        A thin synchronous wrapper around _scan_registry (the I/O half) and
        _apply_scan (the UI half) -- used for on_mount's initial paint and
        by every test that calls it directly, so a same-tick rebuild stays
        deterministic. The recurring poll timer goes through _poll_registry
        instead, which runs the scan on a background thread (see
        _scan_registry_worker) so a slow scan never blocks the event loop
        (C10 tour F1, esc-2303-1).
        """
        self._apply_scan(*self._scan_registry())

    def _scan_registry(self) -> tuple[list[SessionRecord], list[DecisionRecord]]:
        """I/O half of a refresh: scan disk for the current records/decisions.

        Touches only self._scanner/self.fleet_root -- no widget access -- so
        this is safe to call off the main/UI thread (see
        _scan_registry_worker, which does exactly that).
        """
        records = self._scanner.scan()
        decisions = list_decisions(self.fleet_root)
        return records, decisions

    def _apply_scan(
        self, records: list[SessionRecord], decisions: list[DecisionRecord]
    ) -> None:
        """UI half of a refresh: rebuild the SessionTable/DecisionQueue only when something changed.

        The in-memory snapshot diff (build_snapshot/snapshot_changed plus
        _decisions_snapshot) keys on substantive fields only (never
        start_ts/age), so a purely-time-passing poll tick is a no-op -- no
        flicker, and each table's own replace_rows() re-locates the
        previously-highlighted key so the cursor survives a rebuild. This
        method touches widgets, so it must only ever run on the main/UI
        thread: refresh_registry calls it directly (already on that
        thread); _scan_registry_worker marshals back onto it via
        call_from_thread instead of calling it directly.
        """
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
        self._rebuild_queue()

    def _poll_registry(self) -> None:
        """on_mount's set_interval callback: launch the threaded scan worker.

        Never scans synchronously itself -- see _scan_registry_worker, which
        runs the actual I/O off the UI thread so a slow scan (10k+ sessions,
        ~4.5s) never freezes input (C10 tour F1, esc-2303-1).
        """
        self._scan_registry_worker()

    @work(thread=True, group='registry-scan')
    def _scan_registry_worker(self) -> None:
        """Threaded poll worker: run the scan off-thread, marshal the UI update back.

        Runs _scan_registry() (pure I/O, no widget access) on a background
        thread via Textual's @work(thread=True), then hands the result to
        _apply_scan on the main/UI thread via call_from_thread -- Textual
        widgets are never safe to touch directly from a worker thread.
        """
        records, decisions = self._scan_registry()
        self.call_from_thread(self._apply_scan, records, decisions)

    def _rebuild_queue(self) -> None:
        """Re-score and re-render the DecisionQueue from current in-memory state, right now.

        Unlike refresh_registry (which only re-scans disk and rebuilds when
        the on-disk snapshot actually changed), this always re-runs
        order_queue and replace_rows against self._decisions/self._records
        as they stand -- the live-reorder half of a boost/drop/defer
        keypress (PRD §9 C5b). A SESSION-row boost/drop/defer is in-memory
        only (self._boosts, and later self._dropped/self._deferred) and
        never touches disk, so it would never be picked up by
        refresh_registry's snapshot-diff short-circuit otherwise. Action
        handlers that mutate self._decisions directly (a persisted
        decision-row boost) call this afterward instead of waiting for the
        next poll tick.

        Also prunes self._handling down to the keys still present in the
        freshly-built queue: a key whose item left (resolved/dropped, or a
        session moved off AWAITING_INPUT) stops being "handling" so a later,
        unrelated ask that happens to reuse the same key (e.g. the same
        session asking a brand new question) starts out unmarked.
        """
        now = self._now_fn()
        queue_items = order_queue(
            self._decisions,
            self._records,
            self._priorities,
            now,
            boosts=self._boosts,
            handling=self._handling,
            dropped=self._dropped,
            deferred=self._deferred,
        )
        queue = self.query_one('#decision-queue', DecisionQueue)
        queue.replace_rows(queue_items, now)
        self._queue_items_by_key = {item.key: item for item in queue_items}
        self._handling &= self._queue_items_by_key.keys()
        self._update_attention(queue_items)

    def _backend_for(self, kind: str) -> FocusArrangeBackend:
        """Resolve the focus/arrange backend for *kind* ('wm'/'tmux').

        A single injected `backend` (constructor param) overrides routing
        entirely and is used uniformly for every target regardless of kind
        -- this is what lets a test assert against one FakeBackend spy
        (PRD §2 design decisions). Otherwise routes by kind through the
        default {'wm': WmBackend(), 'tmux': TmuxBackend()} map, falling
        back to the wm backend for an unrecognized/foreign kind (fail-soft
        -- a view must never crash on a target shape it didn't control).
        """
        if self._backend is not None:
            return self._backend
        return self._default_backends.get(kind, self._default_backends['wm'])

    def _update_attention(self, queue_items: list[QueueItem]) -> None:
        """Signal (never move) on attention transitions -- PRD B3, this task's leaf signal.

        Diffs the current queue's resolvable targets against the previous
        tick's set of urgent TARGETS -- not item keys: a target newly
        appearing (a session just flipped to awaiting-input, or a new open
        decision was filed) gets set_urgency(True); a target that no longer
        has ANY live item pointing at it (every item that resolved to it
        left the queue -- answered/dropped/resolved, or its session
        vanished) gets set_urgency(False). Diffing by resolved target
        rather than by item key matters because a DecisionRecord and the
        AWAITING_INPUT session it links to (via session_id) can resolve to
        the exact SAME target (see resolve_target) -- keying by item key
        would clear urgency on a still-live target the moment just ONE of
        its two items (e.g. the decision, answered elsewhere) left the
        queue, even though the other (the session) is still asking. Then
        reorders each backend's own targets -- grouped by kind so a wm
        target never pollutes a tmux backend's positional reorder (whose
        index IS the destination window position) and vice versa. This
        method and its caller are the ONLY place the refresh path may call
        backend.*; it calls ONLY set_urgency/reorder, NEVER focus/tile
        (those live exclusively in explicit-action handlers, e.g.
        action_focus_selected).
        """
        new_targets = {item.target for item in queue_items if item.target is not None}
        for target in new_targets - self._attention_targets:
            self._backend_for(target.kind).set_urgency(target, True)
        for target in self._attention_targets - new_targets:
            self._backend_for(target.kind).set_urgency(target, False)
        self._attention_targets = new_targets

        targets_by_kind: dict[str, list[DisplayTarget]] = {}
        for item in queue_items:
            if item.target is not None:
                targets_by_kind.setdefault(item.target.kind, []).append(item.target)
        for kind, targets in targets_by_kind.items():
            self._backend_for(kind).reorder(targets)

    def action_focus_selected(self) -> None:
        """Explicit-action focus (PRD B3/§6.2, §9 C5b) -- the ONLY place a
        focus() call may originate from. Bound to Enter (via BINDINGS) and
        routed here from a DecisionQueue row-selected message too (Enter
        AND a same-row click both post DataTable.RowSelected -- see
        on_data_table_row_selected), so both triggers funnel through this
        one handler. _update_attention (the refresh/diff path) is the
        structural complement: it may call ONLY set_urgency/reorder, never
        focus/tile -- this method is where focus/tile live instead.

        Resolves the DecisionQueue's currently-highlighted row to its
        QueueItem and, when a target is resolvable, calls
        backend.focus(target) exactly once. Always marks the row's key
        "handling" (an in-memory, best-effort "already acted on" marker,
        fed back into the next order_queue call) -- even when no target
        resolved, since the operator's acknowledgement is real regardless
        of whether a live window was found. This mark is not permanent:
        _rebuild_queue prunes self._handling down to whatever is still in
        the queue on every rebuild, so the flag tracks the CURRENT ask, not
        any historical acknowledgement -- a key that later leaves the queue
        and resurfaces (e.g. the same session filing a brand new question)
        starts unmarked again. Fail-soft: an empty queue (no highlighted
        key) or a key not present in the last-built queue no-ops without
        raising -- a gone/unlinked lead is simply not focusable, never a
        crash.
        """
        queue = self.query_one('#decision-queue', DecisionQueue)
        key = queue.highlighted_key()
        if key is None:
            return
        self._handling.add(key)
        item = self._queue_items_by_key.get(key)
        if item is None or item.target is None:
            return
        self._backend_for(item.target.kind).focus(item.target)

    def _decision_by_id(self, decision_id: str) -> DecisionRecord | None:
        return next((d for d in self._decisions if d.id == decision_id), None)

    def _apply_boost(self, *, delta: int | None = None, absolute: int | None = None) -> None:
        """Shared boost logic for action_boost/action_big_boost/action_set_priority.

        Exactly one of *delta* (additive -- b/B) or *absolute* (a direct
        set -- a digit key) is given. A DECISION-backed highlighted row
        persists its new manual_boost via C1's set_manual_boost -- the
        cockpit's own sanctioned decision write -- then re-scans decisions
        so self._decisions (and its snapshot, so a later poll tick doesn't
        redundantly re-detect this same change as external) reflect the
        persisted value directly; no in-memory overlay is needed once a
        decision's boost is on disk. A SESSION-backed row has no
        persisted priority field at all (PRD §2 design decisions), so its
        boost lives ONLY in self._boosts, an ephemeral overlay order_queue
        layers on top of the item's base (always-0) manual_boost. Either
        way, the queue is re-scored and re-rendered immediately after --
        never waits for the next poll tick. Fail-soft: no highlighted row,
        or a key not present in the last-built queue, no-ops.
        """
        queue = self.query_one('#decision-queue', DecisionQueue)
        key = queue.highlighted_key()
        if key is None:
            return
        item = self._queue_items_by_key.get(key)
        if item is None:
            return
        if item.kind == 'decision' and item.decision_id is not None:
            current = self._decision_by_id(item.decision_id)
            current_boost = current.manual_boost if current is not None else 0
            if absolute is not None:
                new_boost = absolute
            else:
                assert delta is not None, 'exactly one of delta/absolute must be given'
                new_boost = current_boost + delta
            set_manual_boost(item.decision_id, new_boost, root=self.fleet_root)
            self._decisions = list_decisions(self.fleet_root)
            self._decisions_snapshot = _decisions_snapshot(self._decisions)
        else:
            current_boost = self._boosts.get(key, 0)
            if absolute is not None:
                new_boost = absolute
            else:
                assert delta is not None, 'exactly one of delta/absolute must be given'
                new_boost = current_boost + delta
            self._boosts[key] = new_boost
        self._rebuild_queue()

    def action_boost(self) -> None:
        """'b' -- nudge the highlighted row's priority up by one point. See _apply_boost."""
        self._apply_boost(delta=self._BOOST_STEP)

    def action_big_boost(self) -> None:
        """'B' -- nudge the highlighted row's priority up by a bigger step. See _apply_boost."""
        self._apply_boost(delta=self._BIG_BOOST_STEP)

    def action_set_priority(self, priority: int) -> None:
        """A digit key (0-9) -- set the highlighted row's priority to that EXACT value.

        Absolute, not additive -- distinct from action_boost/action_big_boost.
        The stored manual_boost is set to *priority* exactly (unclamped --
        see _apply_boost/TestBoostReordersAndPersists), but cockpit.priority
        .score() clamps manual_boost into [Priorities.manual_boost.min,
        .max] (default [-5, 5]) before scoring it, so under the default
        config digits above 5 persist distinctly yet reorder identically to
        '5' (see the BINDINGS comment above). See _apply_boost.
        """
        self._apply_boost(absolute=priority)

    def action_drop(self) -> None:
        """'x' -- drop the highlighted row (PRD §9 C5b).

        A DECISION-backed row persists via C1's update_decision_state --
        the cockpit's other sanctioned decision write -- then re-scans
        decisions so self._decisions (and its snapshot) reflect the
        persisted state directly; order_queue's own state=='open' filter
        then excludes it, exactly like _apply_boost's re-scan. A
        SESSION-backed row has no cockpit-writable state field at all
        (PRD §2 design decisions), so it is instead added to
        self._dropped, an in-memory overlay order_queue filters out by
        key -- never written to the session's own record. Either way the
        queue is re-scored and re-rendered immediately -- never waits for
        the next poll tick. Fail-soft: no highlighted row, or a key not
        present in the last-built queue, no-ops.
        """
        queue = self.query_one('#decision-queue', DecisionQueue)
        key = queue.highlighted_key()
        if key is None:
            return
        item = self._queue_items_by_key.get(key)
        if item is None:
            return
        if item.kind == 'decision' and item.decision_id is not None:
            update_decision_state(item.decision_id, DecisionState.DROPPED, root=self.fleet_root)
            self._decisions = list_decisions(self.fleet_root)
            self._decisions_snapshot = _decisions_snapshot(self._decisions)
        else:
            self._dropped.add(key)
        self._rebuild_queue()

    def action_defer(self) -> None:
        """'d' -- defer the highlighted row: reset its EFFECTIVE age to ~0 (PRD §9 C5b).

        Stamps self._deferred[key] = now(), an in-memory overlay
        order_queue's *deferred* param uses to override that item's
        scoring filed_at -- NEVER a rewrite of the underlying record's
        real filed_at/asked_at/start_ts (that field is provenance -- when
        it was actually filed -- not a UI knob). Applies uniformly to
        both a decision- and a session-backed row, since neither the C1
        registry nor a session record has a cockpit-writable "deferred"
        field (PRD §2 design decisions): a defer is a pure, ephemeral
        "sink this for now" applied on top of whatever order_queue reads.
        Re-scores and re-renders the queue immediately -- never waits for
        the next poll tick. Fail-soft: no highlighted row, or a key not
        present in the last-built queue, no-ops.
        """
        queue = self.query_one('#decision-queue', DecisionQueue)
        key = queue.highlighted_key()
        if key is None:
            return
        if key not in self._queue_items_by_key:
            return
        self._deferred[key] = self._now_fn()
        self._rebuild_queue()

    def action_new_session(self) -> None:
        """'n' -- push the spawn bar's project/role/prompt picker (PRD §9 C5b).

        Seeds the picker's project-root field from known_project_roots over
        the most recently scanned session records (self._records) -- a
        picker source, not a queue filter. Submitting the screen calls back
        into spawn_session, this task's leaf signal; the screen itself never
        builds argv or launches a process.
        """
        self.push_screen(SpawnScreen(known_project_roots(self._records), self.spawn_session))

    def spawn_session(self, project_root: str, role: str, prompt: str) -> None:
        """Spawn a new Claude session (PRD §9 C5b spawn bar) -- the `n` action's leaf signal.

        Builds spawn-claude.sh's exact positional argv (build_spawn_argv)
        from the picked project (cwd), a role-derived title (mirroring
        session_table's own "role:project#task" shape, minus the task
        segment a not-yet-spawned session doesn't have), and the prompt --
        then hands it to the injected spawn_runner (default:
        _default_spawn_runner, a fail-soft subprocess.Popen wrapper).
        Fail-soft (PRD §2): with no resolvable spawn_script (no injected
        path, no $CLAUDE_SPAWN_SCRIPT, and the repo-relative default not
        found), this simply no-ops -- a view/action is never a hard
        dependency, never an exception.
        """
        script = self._spawn_script
        if script is None:
            _log.warning('spawn_session: no spawn_script resolved, dropping spawn request')
            return
        title = f'{role}:{Path(project_root).name}'
        argv = build_spawn_argv(script, project_root, title, prompt)
        self._spawn_runner(argv)

    def action_toggle_tree(self) -> None:
        """'t' -- open the spawn-tree modal (Fleet Cockpit C9a, PRD §9).

        Snapshots self._records (the most recently scanned/ordered set)
        into a fresh SpawnTreeScreen, which renders it as a point-in-time
        parent->child forest -- see cockpit.panes.spawn_tree. That SAME
        snapshot -- a local `records`, not the `self._records` attribute --
        is also closed over for the screen's on_focus callback, so a later
        Enter always resolves the selected slug against the exact list the
        operator is looking at, never whatever list self._records has been
        reassigned to by an intervening poll tick (see refresh_registry,
        which replaces self._records wholesale on every changed scan) --
        see _focus_slug's docstring for why that distinction matters. The
        pop branch below guards against ever stacking a second
        SpawnTreeScreen on top of one already open; it is defensive, not
        how an interactive 't' keypress actually closes the tree though --
        Textual's ModalScreen truncates the non-priority key-binding chain
        at itself, so this app-level 't' binding is never consulted while
        a SpawnTreeScreen is on top of the stack. SpawnTreeScreen therefore
        binds 't' (and Escape) itself, straight to its own dismiss()-based
        action_cancel -- see SpawnTreeScreen.action_cancel for the full
        explanation.
        """
        if isinstance(self.screen, SpawnTreeScreen):
            self.pop_screen()
            return
        records = self._records
        self.push_screen(SpawnTreeScreen(records, lambda slug: self._focus_slug(slug, records)))

    def _focus_slug(self, slug: str, records: list[SessionRecord]) -> None:
        """Resolve *slug* within *records* to a focus target and route it to a backend (Fleet Cockpit C9a).

        *records* is the exact snapshot the open SpawnTreeScreen was built
        from -- action_toggle_tree closes over it and passes it back in on
        every call, rather than this method reading self._records fresh.
        self._records is reassigned wholesale (not mutated) by every
        changed refresh_registry tick, so resolving against the live
        attribute could silently miss a slug that's still visible in the
        (unchanged) open tree if a poll tick lands between opening the tree
        and pressing Enter -- fail-soft in effect, but a surprising one,
        since the operator selected a row they could plainly see. Reusing
        the same snapshot the tree renders makes Enter's behavior match
        what's on screen exactly.

        Reuses decision_queue.resolve_target, exactly mirroring
        action_focus_selected's own resolve-then-route step, so the tree's
        Enter path and the decision queue's Enter path share one focus
        discipline. resolve_target's sessions_by_slug param is only ever
        read for a DecisionRecord (a SessionRecord resolves straight from
        its own .display -- see resolve_target) -- *record* here is always
        a SessionRecord, so an empty mapping is passed rather than
        rebuilding one from *records* on every keypress. Fail-soft (PRD
        §2): a slug with no matching record (a gone/stale snapshot entry)
        or a record with no resolvable target (no Display) both no-op
        rather than raising.
        """
        record = next((r for r in records if r.session_slug == slug), None)
        if record is None:
            return
        target = resolve_target(record, {})
        if target is None:
            return
        self._backend_for(target.kind).focus(target)

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

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Route a DecisionQueue selection (Enter, via DataTable's own
        'enter' binding, or a same-row click) to action_focus_selected.

        SessionTable also posts RowSelected (same message namespace, same
        disambiguation idiom as on_data_table_row_highlighted above) --
        selecting a session-table row has no focus action of its own, so
        that case is simply ignored here.
        """
        queue = self.query_one('#decision-queue', DecisionQueue)
        if event.data_table is not queue:
            return
        self.action_focus_selected()

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
