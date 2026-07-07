# PRD: Fleet Cockpit (attention UI over the Session Attention Rail)

**Status:** ratified (Leo, 2026-07-07 — "LGTM. Do it.") — authorized to author, decompose, and queue.
**Brief:** `~/.claude/spawn-briefs/fleet-cockpit-2026-07-07/brief.md` (ratified scope C1–C10, three design forks, gate pre-answers, queueing authorization).
**Output artifacts:** this PRD + `fleet-cockpit-prd.capability-manifest.md` (beside it).
**Namespace:** `project_id=dark_factory`, `project_root=/home/leo/src/dark-factory`, `agent_id=claude-prd-fleet-cockpit`.
**Depends on (in-flight):** Session Attention Rail batch — `plans/session-attention-rail-prd.md`, df 2283–2289 (statuses at author time: 2283/2284 done, 2285 in-progress, 2286/2287/2288/2289 pending). This PRD **is** the "attention manifest / dashboard renderer" the rail PRD names as its future consumer (rail §1, §5, §6) — it closes rail's forward-looking G1.

## 1. Consumer + user-observable surface (G1 / G2)

The **fleet cockpit** is the human interface over the Session Attention Rail's data plane
(the global session registry at `~/.claude/fleet/`). Its **primary consumer is Leo** — the
daily driver. Usage pattern: at his desk a few hours most days plus 1–2 brief drop-ins nearly
every day; on arrival he wants whole-dev-system state in one place at a glance. Secondary
consumers: the two watcher loops (tmux lane + park-to-registry), the `/prd` skill (sibling
handoff), and `spawn-claude.sh` callers (spawn modes).

Leo's verbatim priorities (descending) and the mechanism each maps to:

| Leo's priority | Mechanism (component) | User-observable signal (G2) |
|---|---|---|
| Start a new terminal | Spawn bar (C5) → spawn-claude.sh | Pressing `n` opens a project/role/prompt picker; a new session launches and appears in the registry + cockpit |
| See working/idle/blocked at a glance | Session table (C5) | Cockpit shows every live session with a state glyph (⚙ working / ✓ idle / ⏸ blocked-on-you / ☠ dead), blocked-first |
| Restack/rearrange to focus one; restack when states change | Focus/arrange backends (C4) + list reorder (C5) | Pressing Enter on a row **raises that terminal** (explicit action); when a session's state changes the cockpit **reorders its lists** and sets a WM urgency hint — but **never** raises/moves a window on its own |
| Summary/priority queue of open questions | Decision queue (C5) + park-to-registry (C8) | A blocked/awaiting-input session appears in a score-ordered queue **with its actual question text within seconds** |
| Dynamically alter prioritization (per-request + heuristics) | Priority scoring lib + priorities.yaml (C3), boost/defer/drop keys (C5), weight editor (C9b) | `b`/`d`/`x`/digits on a queue row reorder it immediately; editing a project/category weight reorders the whole queue |
| Access full context per question | Detail pane (C5) + Enter-to-focus (C4) | Selecting a queue row shows the full question text, task/esc ids, result-file tail, parent/children; Enter focuses that terminal |
| View the spawn tree | Spawn-tree view (C9a) + `parent_session_id` (C1) | A tree toggle shows the parent→child session tree with outstanding children highlighted; Enter jumps to a child |
| Spawn a SIBLING (parented by its parent) | Sibling spawn + /prd handoff (C7) | A `/prd` author session, once the PRD is committed, spawns the decompose session as its **sibling** and exits cleanly; the decompose session's record shows the same parent, not the author session |
| Long-runners survive crashes / reattach | tmux lane (C6) | escalation-watcher + recon-watcher default to a reattachable tmux lane; a crashed watcher is reattachable and its record persists |

**Anti-requirements (Leo, explicit):** a live-updating **file** is not an acceptable UI (two-step,
fights external edits — the afk-digest.md problem); in-terminal summaries are ideal; the cockpit
must **lift** the mental load of session/decision management, not add a viewer to babysit.

## 2. Sketch of approach

A single new **`cockpit/`** uv package (sibling to `escalation/`, `orchestrator/`) plus additive
extensions to the rail-owned registry substrate and the shared spawn/watcher tooling:

- **C1 — registry schema extensions** (additive, migration-free) to the rail T3 helper
  (`session_registry.py`): `parent_session_id`, `spawn_mode`, `display`, `question` on the
  session record, plus a small **decision-record** family (one file per pending human decision).
  `spawn-claude.sh` exports `CLAUDE_SPAWN_SESSION_ID` / `CLAUDE_SPAWN_PARENT_ID` so hooks stamp actuals.
- **C2 — hook enrichment**: the rail hooks (SessionStart/Notification/Stop) additionally capture
  the question text and stamp parent linkage + display target.
- **C3 — priority scoring lib** (`cockpit.priority`) + `~/.claude/fleet/priorities.yaml`: pure,
  property-tested scoring shared by the cockpit (and any future digest consumer).
- **C4 — focus/arrange backends**: one `FocusArrangeBackend` interface, a wm/X11 implementation
  and a tmux implementation, both testable against fakes.
- **C5 — the cockpit TUI** (Textual): decision queue, session table, detail pane, spawn bar,
  poll-based refresh. A **pure consumer** of the registry (writes only manual-boost + decision-state
  + its own UI config); the system stays fully functional when the cockpit isn't running.
- **C6 — tmux lane** for long-runners; **C7 — sibling spawn + /prd handoff**;
  **C8 — park-to-registry** for watchers; **C9 — v1.1** tree view + weight editor (deferrable);
  **C10 — operator acceptance gate** (deterministic pure gate) + a **live smoke test** on the real host.

**Fail-soft everywhere (hard constraint):** a cockpit / backend / hook / registry fault emits loud
logging but **never** breaks a session, a spawn, a watcher, or the orchestrator. The cockpit is a
**view, never a dependency**.

## 3. Ratified design forks (G5 — decided by Leo 2026-07-07, do not relitigate)

1. **Hybrid window model.** Long-lived loop sessions (escalation-watcher, recon-escalation-watcher)
   run in a **tmux lane** (crash-survivable, reattachable); interactive work sessions (`/unblock`,
   `/prd`, `/deb`, ad-hoc) stay as **WM terminal windows**. The cockpit abstracts both behind one
   focus/arrange backend interface (C4).
2. **Signal, don't move** (HARD invariant). The system **NEVER** takes keyboard focus and **never**
   moves/raises windows automatically. State changes may ONLY: reorder the cockpit's lists, update
   tab titles, and set WM urgency hints (taskbar flash). Window raise/focus/tile happens **exclusively**
   on an explicit user action (keypress/click in the cockpit). Any automatic focus/raise/move —
   including a "helpful" exception — is a design violation.
3. **Cockpit form = TUI in a terminal** (Python **Textual**). Mouse click/drag, live refresh,
   headless-testable via the pilot API. Runs in a dedicated always-visible terminal Leo positions
   himself. **NOT** a GUI app, NOT a web page, NOT a file.

## 4. Pre-conditions — substrate verified 2026-07-07 (G3)

Everything below was **actually run** during authoring (not assumed):

| Assumed capability | Verified | Evidence (2026-07-07) |
|---|---|---|
| Textual installs in a fresh `cockpit/` uv package; pilot API present | ✅ | `uv sync` of a scratch pkg with `textual>=0.60` → `textual 8.2.8`; `import textual.app, textual.pilot; Pilot` OK |
| WM focus-by-title (no captured window id required) | ✅ | `wmctrl -a <title>` (activate by title) + `xdotool search --name <title> windowactivate` present |
| WM **urgency hint** — the invariant's *allowed* automatic attention action | ✅ | `xdotool set_window --urgency 1\|0` present; `wmctrl -r <win> -b add,demands_attention` state-change present |
| WM title/tile control | ✅ | `wmctrl -r <win> -T <title>` (retitle), `-e <MVARG>` (move/resize into a zone) |
| tmux focus + reorder | ✅ | `tmux select-window` + `switch-client` (focus); `move-window` (reorder tab strip) |
| X11 reachable, real targets exist | ✅ | `DISPLAY=:0`, `XDG_SESSION_TYPE=x11`; `wmctrl -l` → 44 live windows |
| Registry substrate `~/.claude/fleet/` exists | ⏳ produced by rail T3 (2285) | `~/.claude/fleet` absent today → **hard dep on 2285**; every schema consumer gates on C1 which gates on 2285 |
| `spawn-claude.sh` positional/lifecycle contract + per-emulator title branches | ✅ | rail PRD §3 (verified there); `skills/spawn/spawn-claude.sh` |
| Rail hooks trio (SessionStart/Notification/Stop) exist to enrich | ⏳ produced by rail T6 (2288) | C2 gates on 2288 |

**Two substrate findings, both resolved without a gate pause:**

- **`inotifywait` is NOT installed** (inotify-tools absent). The brief already sanctions "inotify **or**
  1–2s poll". **Resolution:** the cockpit's refresh is **poll-based by default** (1–2s stat of
  `~/.claude/fleet`); inotify is an optional enhancement only if a pure-Python inotify path is added
  later. No prerequisite task; no dependency on the missing CLI.
- **WM window-id capture at spawn is fragile** (terminals launch async; matching a new X11 window to a
  pid is best-effort). **Resolution:** the wm backend's **focus path is by title** (`wmctrl -a`), which
  the rail title convention (`<role>:<project>#<task> <slug>`) makes reliably targetable;
  `display.wm_window_id` is a best-effort optimization, never required for correctness.

**G3 verdict:** all novel substrate the mechanisms invoke is verified present, or (registry dir / rail
hooks) queued as a hard dependency on a tracked rail task. No fictions.

## 5. Resolved design decisions

Beyond the three ratified forks (§3):

1. **Cockpit is its own `cockpit/` uv package** (own `pyproject.toml`, `src/cockpit/`, `tests/`, `uv.lock`)
   in the dark-factory repo, mirroring `escalation/`. Holds C3 (scoring), C4 (backends), C5 (TUI),
   C-smoke (live test). This keeps the daily-driver UI decoupled and independently installable.
2. **C1 schema extensions live in the rail-owned helper** (`session_registry.py`, rail §4.9) — additive
   fields on the existing record type + a new `DecisionRecord` type & writer. Additive-only; if any
   extension turns out to be *breaking* to the rail base schema, **PAUSE and escalate** rather than fork
   the schema (G4).
3. **Focus is by title; urgency is the only automatic attention signal.** `wmctrl -a <title>` /
   `xdotool windowactivate` on explicit action; `xdotool set_window --urgency 1` (+ clear on focus) as
   the sole automatic state-change-driven signal. Tile-blocked-set uses `wmctrl -e` into a configurable
   zone, **explicit action only**.
4. **tmux reorder IS allowed automatically** (the tmux-lane analogue of cockpit list reordering) because
   `move-window` reorders the tab strip **without** disturbing focus — it does not violate signal-don't-move.
   tmux **focus** (`select-window`/`switch-client`) remains explicit-action-only.
5. **Refresh = 1–2s poll** (see §4). The cockpit reads the whole `~/.claude/fleet` tree each tick and
   diffs in memory; single-writer-per-file discipline (rail T3) means no torn reads under `os.replace`.
6. **Decision records are per-file, no global index** (`~/.claude/fleet/decisions/<id>.json`), preserving
   the rail single-writer-per-file rule. The cockpit's only writes are `manual_boost` and `state`
   (open→answered/dropped) on a decision file it owns the update to, plus its own UI config
   (`~/.claude/fleet/cockpit-ui.json`). No writer rewrites a shared index.
7. **Backend interface is X11-first but Wayland-ready.** The `FocusArrangeBackend` ABC is clean enough
   that a `kdotool`/KWin-script Wayland backend can be added later without touching the cockpit. No
   dependency on an emulator switch (kitty/wezterm adapter stays deferred, rail §5).

## 6. Contract section (B+H — the seams consumers import, never re-derive)

This PRD is high-stakes/complex (new package + 4 shared seams: registry schema, spawn-claude.sh,
watcher SKILL.mds, hooks) → **approach B+H**. The two load-bearing seams are the **registry schema
extensions** and the **`FocusArrangeBackend` interface**. Both are specified here so backend/consumer
tasks implement against a frozen contract rather than deriving it, and land as first-class tasks
instead of starving under the narrow-lock orchestrator.

### 6.1 Registry schema extensions (C1) — additive on the rail T3 record

Added to the existing session `record.json` shape (all optional/defaulted → migration-free):

```
parent_session_id: str | None   # spawner's session-slug; None for human-launched roots
spawn_mode:        "child" | "sibling" | "detached"   # default "child" (current semantics)
display: {                       # how to focus this session
    kind:          "wm" | "tmux"
    wm_title:      str            # the convention title (primary focus key)
    wm_window_id:  str | None     # best-effort X11 id (optimization only)
    tmux_target:   str | None     # "<session>:<window>" when kind == "tmux"
} | None
question: { text: str, asked_at: <iso8601> } | None   # latest awaiting-input question
```

New **decision record** family — one file per pending human decision under
`~/.claude/fleet/decisions/<id>.json` (same single-writer-per-file discipline):

```
DecisionRecord {
    id:            str            # stable, filename-safe
    session_id:    str | None     # originating session-slug
    project:       str
    task_id:       str | None
    escalation_id: str | None
    text:          str            # the question/decision prose
    options:       [str] | None
    filed_at:      <iso8601>
    manual_boost:  int            # cockpit-writable
    state:         "open" | "answered" | "dropped"   # cockpit-writable
}
```

**C1 also provides the writer/reader helpers** (`write_decision`, `list_decisions`,
`update_decision_state`, `set_manual_boost`) so C8 (watchers) and C5 (cockpit) share one code path;
each uses tmp-file + `os.replace` atomic write, and **fail-soft** (a write fault logs and returns,
never raises into a watcher/spawn). `SCHEMA_VERSION` bumps its minor.

### 6.2 `FocusArrangeBackend` interface (C4) — the focus/arrange seam

```python
class FocusArrangeBackend(Protocol):
    def focus(self, target: DisplayTarget) -> FocusResult: ...      # EXPLICIT action only
    def set_urgency(self, target: DisplayTarget, on: bool) -> None: # automatic-allowed
    def reorder(self, ordered_targets: list[DisplayTarget]) -> None:# tmux: move-window; wm: NO-OP
    def tile(self, targets: list[DisplayTarget], zone: Zone) -> None: # EXPLICIT action only (wm)
    def is_alive(self, target: DisplayTarget) -> bool: ...
```

Invariants (enforced by boundary tests §7):
- `focus`/`tile` are **only ever** called from an explicit-user-action handler — never from the
  refresh/diff path. The refresh path may call **only** `set_urgency` and `reorder`.
- `reorder` on the **wm** backend is a **no-op with a debug log** (WM windows are not reordered
  automatically); on the **tmux** backend it issues `move-window` (allowed — focus-preserving).
- Any operation on a gone window/target is a **no-op with a warning**, never an exception.
- `wm` backend focus = `wmctrl -a <wm_title>` (fallback `xdotool search --name`); urgency =
  `xdotool set_window --urgency {1,0}`. `tmux` backend focus = `select-window` + `switch-client`.

### 6.3 Priority scoring (C3) — the ordering seam

```python
def score(item: DecisionRecord | SessionRecord, weights: Priorities, now: <ts>) -> float
# f(severity, category_weight, project_weight, age_curve, manual_boost); higher = more urgent
```

Property-tested invariants: monotonic non-decreasing in `manual_boost`; monotonic in age under the
configured curve; `drop`ped items score below all `open` items; identical inputs → identical score
(pure). `priorities.yaml` carries per-project weights, per-category weights, age-curve params.

## 7. Boundary-test sketch (B+H — cross-seam scenarios, both sides)

Each row is an integration-gate scenario; the tasks named build the two-way test.

| # | Scenario | Preconditions | Postcondition (asserted) | Faces |
|---|---|---|---|---|
| B1 | Schema round-trip | C1 landed; a record written with the new fields | Reading it back yields `parent_session_id`/`spawn_mode`/`display`/`question` unchanged; a rail-vintage record (no new fields) still parses (migration-free) | C1 writer ↔ C5 reader |
| B2 | Decision record write→read→update | C1 helpers | `write_decision` → `list_decisions` finds it; `update_decision_state(answered)` and `set_manual_boost` persist; concurrent writers touch only their own file | C8 writer ↔ C5 reader |
| B3 | **Signal-don't-move (headless)** | C5b + fake backend; a synthetic record flips `running`→`awaiting-input` on the refresh tick | The cockpit **reorders** its list and calls `set_urgency(on=True)` **but issues ZERO `focus`/`tile`/`raise` calls**; a fake-backend spy asserts the forbidden calls never occurred | C5 refresh ↔ C4 backend |
| B4 | **Signal-don't-move (live)** | C-smoke on real host; a disposable test window; simulate a state change | Same as B3 against the **real** wm backend: window stack + input focus **unchanged** after the state change; urgency hint observably set | C5/C4 ↔ real X11 |
| B5 | Explicit focus raises the right window | C4 wm backend; a disposable titled window; press Enter path | `focus(target)` raises exactly that window (title match); a gone target → no-op + warning, no raise, no exception | C5 action ↔ C4 wm |
| B6 | tmux reorder preserves focus | C4 tmux backend; ≥2 tmux windows, one focused | `reorder` moves the tab strip to match priority; the focused window is **still focused** afterward | C5 refresh ↔ C4 tmux |
| B7 | Sibling parentage | C7; a spawner with a known `CLAUDE_SPAWN_PARENT_ID` | A `spawn_mode=sibling` spawn writes a child record whose `parent_session_id` == the spawner's **own** parent (not the spawner); the spawner does **not** block on a sentinel | C7 spawn ↔ C1 schema |
| B8 | Question text surfaces fast | C2 + C1; a session hits a Notification | Within the poll window the record's `question.text` is populated and the cockpit decision queue shows that exact one-liner | C2 hook ↔ C5 queue |

## 8. Cross-PRD relationship + seam ownership (G4)

| Seam | Owner | Resolution |
|---|---|---|
| Base registry record **schema + key contract** | **Rail PRD (T3 / 2285)** | This PRD's C1 extensions are **additive/migration-free** on top; if any extension is breaking → PAUSE + escalate (do not fork) |
| Attention manifest / dashboard renderer | **This PRD (the cockpit)** | Closes the rail PRD's named "future consumer"; rail §1/§5/§6 anticipated this exact consumer |
| `spawn-claude.sh` (spawn modes, env exports, tmux mode) | **This PRD (C1 env, C6 tmux, C7 sibling)** | Serializes after the last rail spawn-claude.sh writer (T5 / 2287) via dep edges |
| Watcher SKILL.mds (lane switch, park-to-registry) | **This PRD (C6, C8)** | Serializes after the last rail watcher-SKILL.md writer (T7 / 2289) via dep edges |
| Hooks trio (question capture, parent stamping) | **Rail owns creation (T6 / 2288); this PRD enriches (C2)** | C2 merges into the rail-created hook scripts; dep on 2288 |
| Emulator adapter (kitty/wezterm) | Deferred future task | Decoupled; nothing here depends on an emulator switch (rail §5) |
| Wayland wm backend | Named future work | Sits behind the `FocusArrangeBackend` interface (§6.2); no cockpit change to add it |
| Auto-watcher supervisor | Orchestrator (untouched) | Lease API is interactive-only; the cockpit never contends interactive leases |

No reciprocal-ownership ambiguity: every seam has exactly one owner; the schema seam is additive with
an explicit escalate-don't-fork rule.

## 9. Decomposition plan

13 tasks. `Cx` labels map to brief components. Deps use component labels + rail task ids.
Signals are user-observable (leaf) or name a downstream consumer (intermediate). Same-file writers
are serialized by dependency onto the last in-flight rail writer of that file (spawn-claude.sh → 2287;
watcher SKILL.mds + session_registry.py lease additions → 2289; hooks → 2288; registry substrate → 2285).

- **C1 — Registry schema extensions + decision records + spawn env exports** *(deps: 2285, 2287).*
  Add `parent_session_id`/`spawn_mode`/`display`/`question` to the record; add `DecisionRecord` type +
  atomic fail-soft writer/reader helpers under `~/.claude/fleet/decisions/`; bump `SCHEMA_VERSION`;
  export `CLAUDE_SPAWN_SESSION_ID`/`CLAUDE_SPAWN_PARENT_ID` from `spawn-claude.sh`.
  *Modules:* `orchestrator/src/orchestrator/session_registry.py`, `skills/spawn/spawn-claude.sh`.
  **Signal (intermediate):** schema round-trip test (B1) + decision write→read→update test (B2) pass;
  a rail-vintage record still parses (migration-free). **Consumers:** C2, C4, C5, C6, C7, C8.

- **C2 — Hook enrichment: capture question + stamp parentage/display** *(deps: 2288, C1).*
  Notification hook additionally writes `question={text,asked_at}`; SessionStart hook stamps
  `parent_session_id` from `CLAUDE_SPAWN_*` and best-effort `display` (WINDOWID/TMUX env).
  Merge-not-clobber; fast + fail-soft. *Modules:* `skills/spawn/hooks/*`.
  **Signal (intermediate):** a session that hits a Notification gets `question.text` populated in its
  record (B8 producer side); existing hooks untouched. **Consumer:** C5b decision queue / detail pane.

- **C3 — Priority scoring lib + priorities.yaml** *(no deps).*
  `cockpit.priority.score(...)` + `~/.claude/fleet/priorities.yaml`; property tests for the §6.3
  invariants. *Modules:* `cockpit/src/cockpit/priority.py`, `cockpit/tests/`.
  **Signal (intermediate):** property tests hold (monotonic in boost/age, pure, dropped<open).
  **Consumer:** C5b (queue ordering), C9b (weight editor).

- **C4 — Focus/arrange backends (wm + tmux) behind one interface** *(deps: C1).*
  Implement the §6.2 `FocusArrangeBackend`: wm/X11 (`wmctrl`/`xdotool`) + tmux impls + fakes; obey
  signal-don't-move (focus/tile explicit-only; reorder = tmux move-window / wm no-op; urgency
  automatic; gone-target = no-op+warn). *Modules:* `cockpit/src/cockpit/backends/`.
  **Signal (intermediate):** unit tests against fakes assert the call-discipline invariants (B3-shape,
  headless) and gone-target no-ops. **Consumers:** C5b, C-smoke.

- **C5a — Cockpit TUI skeleton: session table + detail pane + poll refresh** *(deps: C1).*
  Textual app; 1–2s poll of `~/.claude/fleet`; **session table** (state glyph, title, age, project,
  outstanding-children badge; blocked-first) + **detail pane** (full record); pure-consumer write
  discipline + UI config file. *Modules:* `cockpit/src/cockpit/app.py`, `.../panes/`.
  **Signal (intermediate):** Textual pilot test — a record added/changed on disk appears/updates in the
  table within the poll window; selecting a row renders its detail. **Consumer:** C5b.

- **C5b — Decision queue + keybindings + spawn bar + focus wiring** *(deps: C5a, C2, C3, C4).*
  **Decision queue** (score-ordered via C3; rows `[score][age][project#task][question]`); keys/mouse:
  Enter/click = `backend.focus` + mark handling; `b`/`B` boost, `d` defer, `x` drop, digits = manual
  priority; **spawn bar** (`n` → project/role/prompt picker → spawn-claude.sh); refresh path calls only
  `set_urgency`/`reorder`. *Modules:* `cockpit/src/cockpit/app.py`, `.../panes/`.
  **Signal (leaf-ish):** Textual pilot — a synthetic record flips to `awaiting-input`; the cockpit
  reorders + sets urgency and issues **zero** focus/raise calls (B3); Enter calls `focus` on the right
  target; the queue orders by score and reorders live on `b`/`d`/`x`. **Consumers:** C-smoke, C9a, C9b, C10.

- **C6 — tmux lane for long-runners** *(deps: 2287, 2289, C1).*
  `spawn-claude.sh` tmux mode (`CLAUDE_SPAWN_BACKEND=tmux`): `new-window` in a per-project tmux session,
  convention title, record `display.kind=tmux`, sentinel/result/verify semantics preserved; switch
  escalation-watcher + recon-escalation-watcher SKILL.md **default lane** to tmux (interactive skills
  unchanged). *Modules:* `skills/spawn/spawn-claude.sh`, `skills/escalation-watcher/SKILL.md`,
  `skills/recon-escalation-watcher/SKILL.md`.
  **Signal (intermediate):** a long-runner spawned in tmux mode gets a `display.kind=tmux` record and a
  reattachable tmux window; a killed watcher is reattachable and its record persists. **Consumer:** C10.

- **C7 — Sibling spawn + /prd handoff** *(deps: 2287, C1).*
  `spawn-claude.sh spawn_mode=sibling`: parent-of-record = spawner's OWN parent
  (`CLAUDE_SPAWN_PARENT_ID`; null→root), **no** blocking sentinel wait (join via registry status +
  result file). `/prd` author mode: after commit, spawn the decompose session as a sibling pointing at
  the committed PRD, write own `result.md` (`outcome=handed-off`), exit. Document the generalized
  author→execute / investigate→fix handoff in spawn SKILL.md. *Modules:* `skills/spawn/spawn-claude.sh`,
  `skills/spawn/SKILL.md`, `skills/prd/**` (handoff step).
  **Signal (intermediate):** a `spawn_mode=sibling` spawn writes a child record whose `parent_session_id`
  is the spawner's parent (B7), and the spawner exits without a sentinel wait. **Consumer:** C10.

- **C8 — Park-to-registry for watchers** *(deps: 2289, C1; `complexity=simple`).*
  escalation-watcher (and recon-watcher) "tell the human / park pending decision" moments additionally
  call C1's `write_decision` to file a decision record; afk-digest.md **retained** but demoted to a
  generated history view (do **not** remove it). *Modules:* `skills/escalation-watcher/SKILL.md`
  (+ recon-watcher guidance), invoking the C1 helper.
  **Signal (intermediate):** a watcher park moment writes a `DecisionRecord` (state=open) the cockpit
  queue then shows; the digest still generates. **Consumer:** C5b queue + C10.

- **C-smoke — Live smoke test on the real X11/tmux host** *(deps: C4, C5b).*
  A scripted live test (its own disposable window + disposable tmux window): wm backend **raises** the
  real test window on an explicit call (B5 live); a simulated state change sets urgency and moves **no**
  window / steals **no** focus (B4 live, signal-don't-move); tmux `reorder` preserves focus (B6 live).
  Runs on the real host, tears down its own artifacts. *Modules:* `cockpit/tests/smoke/`.
  **Signal (leaf):** on the real host, `focus` raises exactly the disposable test window; after a
  simulated blocked-state change the window stack + input focus are unchanged and the urgency hint is
  observably set. **Consumer:** C10.

- **C9a — v1.1 spawn-tree view** *(deps: C5b; deferrable).*
  Tree toggle: `parent_session_id` tree, outstanding children highlighted, Enter jumps to a child.
  **Signal (leaf):** pilot test — the tree renders the parent→child structure and Enter on a child
  invokes `focus` on that child's target.

- **C9b — v1.1 in-cockpit priorities.yaml editor** *(deps: C3, C5b; deferrable).*
  Edit project/category weights without leaving the TUI; persists to `priorities.yaml`.
  **Signal (leaf):** pilot test — bumping a project weight in-cockpit reorders the decision queue and
  the change persists to `priorities.yaml`.

- **C10 — Operator acceptance gate (capstone)** *(deps: C5b, C6, C7, C8, C-smoke;
  `task_kind=deterministic`, `always_escalates=true`, no `before_done` → pure gate).*
  Born-at-L2 escalation asking Leo to drive the cockpit live for a session and accept/adjust. This is
  the **G2 leaf for the whole batch**. **Signal (leaf):** Leo accepts after live use (task goes `done`
  on `resume` after his sign-off).

**DAG (critical path in bold):** C3 (free) ‖ C1←{2285,2287}. **C1 → C4 → C5b**; C5a←C1, **C5b←{C5a,C2,C3,C4}**;
C2←{2288,C1}; C6←{2287,2289,C1}; C7←{2287,C1}; C8←{2289,C1}; **C-smoke←{C4,C5b}**;
C9a←C5b, C9b←{C3,C5b} (deferrable); **C10←{C5b,C6,C7,C8,C-smoke}**.

## 10. Out of scope

- **Emulator adapter** (kitty `kitty @` / WezTerm `wezterm cli`) — deferred (rail §5); nothing here
  couples to an emulator switch.
- **Wayland wm backend** — named future work behind the `FocusArrangeBackend` interface.
- **Retiring afk-digest.md** — demoted to a history view here (C8), but not removed; retirement is a
  later decision.
- **push/ntfy reachability** — excluded by standing user directive (AFK-by-design: optimize autonomous
  handling + clean RETURN trail, not reachability).
- **A global registry index file** — forbidden (breaks single-writer-per-file); the cockpit scans the dir.
- **Cross-host / remote fleet** — single-host (Leo's workstation) only.

## 11. Open questions (tactical — resolved at implementation time)

1. **Exact keybindings** beyond the brief's set (b/B/d/x/digits/n/Enter) and mouse drag-to-reorder if
   Textual makes it cheap. *Suggested:* start with the brief's keys; add drag only if trivial. Decide in C5b.
2. **priorities.yaml exact schema** (weight ranges, age-curve form — linear vs log vs sigmoid).
   *Suggested:* simple bounded linear age curve + per-project/per-category float weights. Decide in C3.
3. **tmux session naming** for the lane (per-project `fleet-<project>` vs one shared). *Suggested:*
   per-project session, window-per-runner. Decide in C6.
4. **Failed spawn/gone-window glyph set** finalization (⚙/✓/⏸/☠ suggested). Decide in C5a.
5. **State-glyph poll cost** — if 1–2s full-dir scans get heavy at 30+ sessions, add mtime short-circuit
   or the optional inotify path. *Suggested:* ship the poll; optimize only if measured. Decide in C5a.

---
*Metadata note for the orchestrator: tasks carry `user_observable_signal`, `consumer_ref`, and a
substrate-confirmed flag in metadata. The orchestrator does not currently read these fields — they are
substrate for a future tracking-infra session.*
