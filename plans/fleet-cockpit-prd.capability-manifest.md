# Capability Manifest — Fleet Cockpit PRD

Beside `fleet-cockpit-prd.md`. Mechanizes **G3** (assumed-substrate verified) + **G6** (premise
validity) per leaf/load-bearing task: every capability a signal asserts is bound to evidence.
Any binding resolving to a FAIL value (`declared-only`/`test-only`/`producer-absent`/
`producer-downstream`/`producer-extent-short`/`bound≤floor`/`rejection-absent`) **blocks the batch**.

**Domain flag:** tooling/UI (not numerical). G6 branches **1 (numeric floor)** and **2 (closed-form
exactness)** do **not fire** — no signal asserts a number or an exactness claim. The load-bearing
branches here are **3 (end-to-end capability + field-population twin)** and **4 (negative assertion /
rejection-mechanism)** — the latter is the signal-don't-move invariant.

**Result: all bindings PASS. No FAIL. Batch clears the manifest gate.**

## A. Substrate bindings (G3) — verified LIVE at authoring 2026-07-07

| Capability | Binding | Evidence | Verdict |
|---|---|---|---|
| Textual + pilot API in a fresh `cockpit/` uv pkg | `substrate:verified` | `uv sync` scratch pkg `textual>=0.60` → `textual 8.2.8`; `import textual.app, textual.pilot; Pilot` OK | PASS |
| wm focus-by-title | `substrate:verified` | `wmctrl -a <WIN>` (activate by title) + `xdotool search --name … windowactivate` present | PASS |
| **wm urgency hint** (the invariant's *allowed* automatic action) | `substrate:verified` | `xdotool set_window --urgency 1\|0` present; `wmctrl -r <win> -b add,demands_attention` present | PASS |
| wm retitle / tile-to-zone | `substrate:verified` | `wmctrl -r -T` (retitle), `-e <MVARG>` (move/resize) | PASS |
| tmux focus + reorder | `substrate:verified` | `tmux select-window` + `switch-client` (focus); `move-window` (reorder) | PASS |
| X11 reachable + real targets | `substrate:verified` | `DISPLAY=:0`, `XDG_SESSION_TYPE=x11`, `wmctrl -l` → 44 windows | PASS |
| `~/.claude/fleet/` registry substrate | `producer:rail-2285` (upstream via C1) | dir absent today; created by rail T3 (2285, in-progress); C1 deps 2285 | PASS (gated) |
| rail hooks trio to enrich | `producer:rail-2288` (upstream via C2) | rail T6 (2288) creates SessionStart/Notification/Stop; C2 deps 2288 | PASS (gated) |
| `spawn-claude.sh` contract + emulator title branches | `substrate:verified` (rail §3) | `skills/spawn/spawn-claude.sh`; rail PRD §3 evidence table | PASS |

*Resolved non-blockers (see PRD §4):* `inotifywait` **absent** → refresh is **poll-based** (brief-sanctioned
alternative), no dep on the missing CLI. `wm_window_id` capture fragile → **focus-by-title is primary**,
window-id best-effort only.

## B. Per-task capability→evidence bindings

DAG-direction verified for every producer link: the producer task is **upstream** of the consumer
(no `producer-downstream`). Field-population sentinel for this project = a **declared-but-unwritten**
record field; each field below names the task that **writes a real value** on the production path.

### C1 — schema extensions + decision records + env exports (intermediate; consumers C2/C4/C5/C6/C7/C8)
| Capability asserted (B1/B2) | Binding | Verdict |
|---|---|---|
| record gains `parent_session_id`/`spawn_mode`/`display`/`question` (round-trips; rail-vintage still parses) | `producer:C1` self; `substrate:session_registry.py` from rail-2285 upstream | PASS |
| `DecisionRecord` + atomic fail-soft writer/reader helpers | `producer:C1` self | PASS |
| `CLAUDE_SPAWN_SESSION_ID/PARENT_ID` exported from spawn-claude.sh | `producer:C1` self; serialized after rail-2287 | PASS |

### C2 — hook enrichment (intermediate; consumer C5b)
| Capability | Binding | Verdict |
|---|---|---|
| `question={text,asked_at}` **populated** on the Notification path | `producer:C2` (field-population of C1's declared field); hooks from `producer:rail-2288` upstream | PASS |
| `parent_session_id`/`display` **populated** on SessionStart from `CLAUDE_SPAWN_*` | `producer:C2`; env from `producer:C1` upstream | PASS |

### C4 — focus/arrange backends (intermediate; consumers C5b, C-smoke)
| Capability | Binding | Verdict |
|---|---|---|
| `focus`/`set_urgency`/`reorder`/`tile`/`is_alive` per §6.2 | `producer:C4` self; each verb `substrate:verified` (§A) | PASS |
| call-discipline invariants (focus/tile explicit-only; reorder=tmux-move/wm-noop; gone→noop+warn) | `rejection-check:C4 fake-backend unit tests` (headless B3-shape) | PASS |

### C5b — decision queue + keys + spawn bar + focus wiring (leaf-ish; consumers C-smoke/C9/C10)
| Capability asserted | Binding | Verdict |
|---|---|---|
| queue orders by score; live reorder on b/d/x | `producer:C3` (score, upstream) + `producer:C5b` | PASS |
| Enter focuses the right terminal (end-to-end) | `producer:C4` focus (upstream) + `display` from `producer:C1/C2` (upstream) + `substrate:` real window | PASS |
| **signal-don't-move (headless, B3):** state-change tick → reorder + urgency, **ZERO** focus/raise/tile | `rejection-check:B3` — fake-backend spy asserts forbidden calls absent under a triggering flip → `awaiting-input` | PASS (rejection observed-absent under trigger) |
| decision queue shows the real question one-liner (B8) | `producer:C2` question-population (upstream) + `producer:C5b` | PASS |

### C6 — tmux lane (intermediate; consumer C10)
| Capability | Binding | Verdict |
|---|---|---|
| tmux-mode spawn writes `display.kind=tmux`; reattachable window; record persists on crash | `producer:C6` (populates C1's `display` field); serialized after rail-2287/2289 | PASS |

### C7 — sibling spawn + /prd handoff (intermediate; consumer C10)
| Capability asserted (B7) | Binding | Verdict |
|---|---|---|
| `spawn_mode=sibling` writes child record whose `parent_session_id` = spawner's OWN parent; no sentinel wait | `producer:C7` (populates C1's `spawn_mode`/`parent_session_id`); env from `producer:C1` upstream | PASS |

### C8 — park-to-registry (intermediate; consumers C5b/C10)
| Capability | Binding | Verdict |
|---|---|---|
| watcher park moment writes a real `DecisionRecord(state=open)` the queue then shows | `producer:C8` (field-population — writes real records via C1's `write_decision`, upstream); digest retained | PASS |

### C-smoke — live smoke on real host (leaf; consumer C10)
| Capability asserted | Binding | Verdict |
|---|---|---|
| **B5 live:** `focus` raises exactly the disposable test window | `producer:C4` focus (upstream) + `substrate:` real X11 (§A) | PASS |
| **B4 live signal-don't-move:** simulated state change → window stack + input focus **unchanged**; urgency set | `rejection-check:B4` — real `wmctrl -l` stack + `xdotool getactivewindow` unchanged after trigger; `--urgency` observed set | PASS (rejection observed-absent live) |
| **B6 live:** tmux `reorder` preserves focus | `producer:C4` tmux backend (upstream) + `substrate:` real tmux (§A) | PASS |

### C9a — spawn-tree view (leaf; deferrable)
| Capability | Binding | Verdict |
|---|---|---|
| tree renders parent→child; Enter jumps to a child | `producer:C1` `parent_session_id` (upstream, populated by C2/C7) + `producer:C4/C5b` focus (upstream) | PASS |

### C9b — in-cockpit weight editor (leaf; deferrable)
| Capability | Binding | Verdict |
|---|---|---|
| weight edit reorders queue + persists to priorities.yaml | `producer:C3` scoring+yaml (upstream) + `producer:C5b` (upstream) | PASS |

### C10 — operator acceptance gate (leaf; batch G2 leaf)
| Capability asserted | Binding | Verdict |
|---|---|---|
| Leo drives the cockpit live end-to-end and accepts | end-to-end capability fully delivered by deps `{C5b,C6,C7,C8,C-smoke}` — **all upstream**, none downstream | PASS (no DAG inversion) |

## C. G6 branch summary

- **Branch 1 (numeric floor):** not fired — no numeric bounds asserted anywhere.
- **Branch 2 (closed-form exactness):** not fired — no exactness claims.
- **Branch 3 (end-to-end + field-population):** every end-to-end signal traces its capabilities to
  the task's dependency set (upstream only); every asserted record field names a producer that
  **writes** it (C2 question, C6 display, C7 spawn_mode/parent, C8 decision records) — no
  declared-but-unwritten field reaches a consumer signal.
- **Branch 4 (rejection / negative assertion):** the signal-don't-move invariant is bound twice —
  headless (B3, fake-backend spy) and live (B4, real X11 stack/focus) — each **triggers** a
  state change and **observes the forbidden action's absence**. Not a bare claim; a fired-and-observed
  rejection binding.

No binding resolves to a FAIL value → the batch is clear to queue.
