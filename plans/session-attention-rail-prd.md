# PRD: Session Attention Rail

**Status:** ratified (Leo, 2026-07-07) — authorized to author, decompose, and queue.
**Brief:** `~/.claude/spawn-briefs/attention-rail-2026-07-07/brief.md` (evidence, gate pre-answers, scope).
**Output artifacts:** this PRD + `session-attention-rail-prd.capability-manifest.md` (beside it).
**Namespace:** `project_id=dark_factory`, `project_root=/home/leo/src/dark-factory`.

## 1. Consumer + user-observable surface (G1 / G2)

Leo runs 4–30 durable, multi-day interactive Claude Code sessions across dark-factory,
reify, and siblings (open-duration p50 ~6h / p90 ~68h / max ~15d; ~3–5 interactive
starts/day/project). The dark-factory checkout is the **fleet cockpit** — sessions there
routinely operate on reify/solar-challenge and vice-versa, so **all identity keys on
role / task / escalation, never cwd**. Five documented pain classes (P1–P5, brief §Premise)
motivate seven mechanisms, each with a named consumer and a user-observable completion signal:

| Mechanism | Consumer (G1) | User-observable signal (G2) |
|---|---|---|
| Mandatory spawn titles | Leo-on-return (WM window list); *future attention manifest* | Every spawned window carries a `<role>:<project>#<task> <slug>` title in the window manager |
| Enriched `/unblock` prompts | cold human reader in the spawned window | Spawned prompt shows esc-id + category + severity + summary after the leading `/unblock <id>`; `/unblock` still triggers |
| Session registry | T4/T5/T6/T7 (this PRD) + *future manifest renderer* | A `record.json` appears per session with status transitions and exit code |
| Verify-spawn-started | calling background task / escalation-watcher; spawn-claude.sh | A spawn that never starts Claude surfaces a **loud** caller-visible line + registry `failed-to-start` + a distinct exit code, within the grace window |
| Result-handback protocol | parent spawner sessions (fan-out/join) | `result.md` appears next to the record; the parent quotes it instead of exploring |
| Hooks trio (SessionStart/Notification/Stop) | Leo-on-return (tab titles); registry (captures hand-launched sessions) | Tab retitles to `⏸ AWAITING …` when blocked; hand-launched sessions also get records |
| Role leases | escalation-watcher startup; `/unblock` startup | A duplicate watcher prints `lease held by <session> (alive, heartbeat 42s ago) — standing down` and exits |

**Named future consumer (out of scope):** the **attention manifest / dashboard renderer**
(Leo is writing the UX notes). The registry record schema is designed here so that a manifest
renderer can be added later **without migration** — that is the primary reason the registry is
a first-class, versioned, documented artifact rather than an ad-hoc file.

## 2. Sketch of approach

A single durable substrate — a **global, cross-project session registry** under
`~/.claude/fleet/` — plus the writers and readers that populate and consume it:

- **`spawn-claude.sh`** (the one programmatic-spawn chokepoint) writes a per-session record at
  launch and updates it at exit; allocates a result file; verifies the session actually started.
- **Claude Code hooks** (SessionStart/Notification/Stop) refresh the record for *every* session
  (including Leo's hand-launched ones) and retitle terminal tabs on attention-state changes.
- **Role leases** (atomic files in the same substrate) give single-owner-per-role semantics,
  replacing the pgrep/ps-tree archaeology ritual.
- **Skill docs** (escalation-watcher, factory-init, spawn, /unblock) are updated to pass titles,
  enrich prompts, claim leases first, and read result files on join.

Everything is **fail-soft**: a registry / lease / hook / verify fault emits loud logging but
**never** fails a spawn, a watcher cycle, or a session start. Exit codes are **demoted to
liveness-only** (present-vs-died-silent); the semantic result channel is `result.md`.

## 3. Pre-conditions — substrate verified 2026-07-07 (G3)

| Assumed capability | Verified | Evidence |
|---|---|---|
| `spawn-claude.sh` positional contract `<cwd> <skip_perms> <title\|""> <prompt>`, sentinel-wait lifecycle | ✅ | `skills/spawn/spawn-claude.sh:29-37,42-99` |
| Per-emulator title plumbing already present | ✅ | `spawn-claude.sh:162` (gnome `--title`), `:170` (xterm `-T`), `:178` (kitty `--title`), `:186` (konsole `-p tabtitle=`) |
| `finish()` is the single exit chokepoint (exit-code capture point) | ✅ | `spawn-claude.sh:92-99` |
| `inner` payload is the env-injection / prompt-trailer point | ✅ | `spawn-claude.sh:69-72` |
| Real test harness for the script (fake-terminal + fake-claude + `SPAWN_LAUNCH_GRACE_SECS`) | ✅ | `tests/scripts/test_spawn_claude.py` (505 lines) |
| Existing exit codes: 0–125 (claude), 126 no-emulator, 127 launcher-fail, 129 window-closed, 143 SIGTERM, 2 usage | ✅ | `spawn-claude.sh:20-25` header + code |
| `.task-meta/<name>/interactive.json` single-writer stamp + TTL/pid reaper idiom to mirror | ✅ | `git_ops.py:1983-1991` (stamp), `reap_interactive_worktrees` `:5947`, `ReapedInteractiveWorktree` `:555` |
| Transcript path encoding `/home/leo/src/dark-factory` → `~/.claude/projects/-home-leo-src-dark-factory/` (both `/` and `.` → `-`) | ✅ | dir exists; `-home-leo--openclaw-workspace` confirms `.`→`-` |
| Five `/spawn`→`/unblock` call sites | ✅ | `skills/escalation-watcher/SKILL.md:395,403,422,456,518` |
| factory-init spawn invocation | ✅ | `skills/factory-init/SKILL.md:110` (positional `<cwd> <skip_perms> '<title>' '<prompt>'`) |
| spawn SKILL.md `terminal_title` arg + exit-code table | ✅ | `skills/spawn/SKILL.md:45,71-77` |
| `/unblock` Step 0 extracts task-id from message; Step 1 re-derives all context from `TASK_ID` | ✅ | `skills/unblock/SKILL.md:23-60` — validates the item-2 premise (enrichment is additive, no /unblock change) |

**Two brief-contradictions found and resolved (see §4):** `~/.claude/sessions/` is **not**
greenfield (Claude Code itself writes `<pid>.json` records there), and `~/.claude/settings.json`
**already has a populated `hooks` key** (`PreToolUse`/`PostToolUse`). Both are handled by design
decisions below rather than blocking — the brief granted registry-layout latitude and both fixes
are additive.

## 4. Resolved design decisions

1. **Registry lives OFF the harness-owned dir.** `~/.claude/sessions/` is written and managed by
   Claude Code itself (`<pid>.json` with `sessionId`/`status`/`name`/`version`/`peerProtocol`).
   Co-tenanting our reaper there risks the harness cleaning our entries or our reaper deleting
   harness files. **Decision:** the registry lives at **`~/.claude/fleet/sessions/<session-slug>/record.json`**
   (global, cross-project, one subdir per record — single-writer-per-record). Leases live at
   **`~/.claude/fleet/leases/<lease-name>.lease`**. The harness `~/.claude/sessions/` record is
   available as an *optional enrichment source* (sessionId/name) but is never written or reaped by us.

2. **Session-slug = `<role>-<project>[-<taskid>]-<pid>`** (filesystem-safe). Pid guarantees
   record-level uniqueness across concurrent spawns that share role+project+task (the near-dup
   incident class); **single-ownership is enforced separately by leases**, not by the record key.

3. **Result format = markdown with a small structured header** (`result.md`). Parent readers are
   LLMs. Header fields: `outcome` (`done|blocked|abandoned|handed-off`), `changed`
   (commits/branches/task-ids), `action_needed` (what the spawner must do). Prose body below.

4. **Status enum:** `launching → running → awaiting-input → idle → exited | failed-to-start`.
   `launching`/`exited` written by spawn-claude.sh; `running`/`failed-to-start` by the verify step;
   `awaiting-input`/`idle` by the Notification/Stop hooks; `running`-refresh also by SessionStart.

5. **Hooks MERGE, never clobber.** `~/.claude/settings.json` already carries `PreToolUse` (Bash,
   EnterWorktree) and `PostToolUse` (ExitWorktree) hooks. The new `SessionStart`/`Notification`/`Stop`
   event keys are **added to the existing `hooks` object**; no existing event, matcher, or top-level
   key (`env`, `permissions`, `statusLine`, …) is disturbed. Hook **scripts live in the dark-factory
   repo** (version-controlled) and are referenced by absolute path — distinct from the existing
   non-repo `~/.claude/hooks/` scripts. Suggested home: `skills/spawn/hooks/`.

6. **Terminal retitle = emulator-agnostic OSC only** (`\033]0;<title>\007`) — no konsole DBus.
   Default glyphs (implementer may finalize): running `⚙`, awaiting-input `⏸ AWAITING`, idle `✅`.

7. **Lease semantics.** Atomic `O_EXCL` create; content = holder identity (session-slug, pid,
   start-ts). Heartbeat = mtime touch on each watcher cycle. Stale = heartbeat-age > TTL **and**
   pid not alive. Second claimant: report holder + liveness, then **stand down** (watchers) or
   **warn-and-proceed** (configurable per role, e.g. `/unblock`). **Interactive-only:** the
   headless supervised auto-watcher (`escalation-watcher-auto`, resolution label
   `escalation-watcher-L2`) is **forbidden from spawning and must never claim or contend an
   interactive lease** — the lease API is invoked only from interactive skills, so auto rotations
   simply never call it. The lease namespace (`watcher-<project>`, `recon-watcher-<project>`,
   `unblock-<project>#<task>`) is for interactive owners only.

8. **Reaper TTLs (defaults, hot-tunable later):** session records reaped when status is terminal
   (`exited`/`failed-to-start`) and age > 24h, **or** non-terminal but pid dead and heartbeat
   age > 1h. Leases reaped on the stale rule in (7). Mirror the `interactive.json` reaper: derive
   identity from the path, never depend on the record body being intact.

9. **Testable logic factored into a Python helper** under the orchestrator tree (e.g.
   `orchestrator/src/orchestrator/session_registry.py`) that spawn-claude.sh and the hook scripts
   both call; the bash stays thin. This keeps the verify chain real (unit-testable record/lease/reaper
   logic) and reuses the existing `tests/scripts/test_spawn_claude.py` harness for the bash seam.

**Explicitly delegated to implementation-time (brief G5, "pick sensible defaults, don't park"):**
exact glyphs, exact reaper TTLs, exact registry sub-layout, and the failed-to-start exit-code
integer (see Open Questions).

## 5. Out of scope

- **Attention manifest / dashboard renderer** — future PRD (Leo's UX notes). Schema anticipated
  here; no migration required to add it.
- **Emulator adapter** (kitty `kitty @` / WezTerm `wezterm cli`) for focus-jump + ground-truth tab
  enumeration — optional, deferrable follow-up. Core hooks **must not couple** to any emulator.
  Leo is weighing a terminal switch (WezTerm) — this stays decoupled so that decision is free.
- **VibeTunnel / session-manager overlays** — out entirely.
- **Orchestrator-internal auto-watcher supervisor** — untouched beyond lease-namespace awareness.
- **push/ntfy reachability** — deliberately excluded; Leo is AFK-by-design (optimize autonomous
  handling + a clean RETURN trail, not reachability).

## 6. Cross-PRD relationship + seam ownership (G4)

| Seam | Owner | Resolution |
|---|---|---|
| Registry record **schema + key contract** | **This PRD (T3)** | Shared contract consumed by T4/T5/T6/T7; a documented schema + a two-way boundary test on the write→refresh→reap path (G5 / B+H) |
| Attention manifest / dashboard | Future PRD (Leo) | Registry schema owned here, migration-free extension point |
| Emulator adapter | Optional future task | Deferrable; hooks decoupled from emulators |
| Auto-watcher supervisor | Orchestrator (untouched) | Lease API is interactive-only; auto rotations never call it |
| VibeTunnel / overlays | — | Out of scope |

**High-stakes seam (G5 → approach B + H):** the registry record schema/key is written by
spawn-claude.sh, refreshed by the SessionStart hook, and read by the lease/verify/result paths.
T3 must ship it as a **documented contract** (Python dataclass/`TypedDict` + JSON shape + a
`SCHEMA_VERSION`) and a **two-way boundary test**: (a) a record written by the spawn writer is
found and refreshed by the hook under the same key; (b) a hook-refreshed record is still reaped
correctly. Consumers import the contract; they do not re-derive the shape.

## 7. Decomposition plan (7 tasks → 7 items; signals are user-observable)

Grouped by edit-locus to respect the orchestrator's narrow-file-lock model (same-file writers are
serialized by dependency, so no task starves on a lock). **T3 is the spine root.**

- **T1 — Harden the 5 escalation-watcher spawn call sites** *(items 1-watcher + 2; `complexity=simple`; no deps).*
  Add a mandatory `terminal_title` (`<role>:<project>#<task-id> <slug>`) **and** extend each spawned
  prompt from bare `/unblock <task_id>` to `/unblock <task_id>` + trailing esc-id/category/severity/
  one-line-summary at `skills/escalation-watcher/SKILL.md:395,403,422,456,518`.
  **Signal:** each of the 5 call sites passes a non-empty convention-shaped title and an enriched
  prompt whose leading token is still `/unblock <id>`; a note records that `/unblock` Step-1
  re-derivation stays authoritative (verified: `skills/unblock/SKILL.md:23-60`).

- **T2 — Title convention doc + factory-init titles** *(item 1 remainder; `complexity=simple`; no deps).*
  Update `skills/spawn/SKILL.md` to make `terminal_title` effectively required for programmatic
  callers and document the `<role>:<project>#<task> <slug>` convention; apply a convention-shaped
  title to the `skills/factory-init/SKILL.md:110-119` spawn steps.
  **Signal:** spawn SKILL.md documents the convention + marks title required for programmatic
  callers; factory-init spawn steps pass a non-empty convention-shaped title.

- **T3 — Session registry substrate** *(item 3; NOT simple; no deps — SPINE ROOT).*
  Python helper (`session_registry.py`): documented record schema (`SCHEMA_VERSION`, title, role,
  project, task_id/esc_id, prompt, cwd, launcher_pid, start_ts, status-enum, exit_code, result_file,
  transcript_path), single-writer atomic write (tmp + `os.replace`), and a TTL/pid stale-record
  reaper mirroring the `interactive.json` idiom. Wire into `spawn-claude.sh`: write `launching`
  after arg-parse, update `exit_code`+`exited` in `finish()`. **Fail-soft:** a forced registry
  fault must not change the spawn's exit code.
  **Signal:** after a spawn, `~/.claude/fleet/sessions/<slug>/record.json` exists with
  `launching→…→exited` and populated `exit_code`; a stale record (dead pid, past TTL) is reaped;
  a forced registry-write failure leaves the spawn's exit code unchanged (fail-soft test).

- **T4 — Verify-spawn-started + failed-to-start** *(item 4; NOT simple; deps: T3, T2).*
  After launch, a **backgrounded bounded poll** (60–120s grace) for the transcript under
  `~/.claude/projects/<encoded-cwd>/` newer than spawn-ts, **or** a claude child of the spawned
  terminal — able to break the unbounded `await_sentinel` hang (`spawn-claude.sh:126`, the P4
  silent-no-transcript path). On failure: registry `failed-to-start`, a **loud** stderr/stdout line
  the calling background task sees, and a **distinct documented exit code**. Document the encoding
  and the new code in the spawn SKILL.md exit-code table. *(Dep T2 serializes spawn/SKILL.md edits.)*
  **Signal:** a spawn whose Claude never starts (no transcript; launcher exits 0) is detected within
  grace → registry `failed-to-start` + loud caller-visible line + the distinct exit code; a normal
  spawn is **not** flagged.

- **T5 — Result-handback protocol** *(item 5′; NOT simple; deps: T4, T1).*
  spawn-claude.sh allocates `<record-dir>/result.md`, exports `CLAUDE_SPAWN_RESULT_FILE` into the
  spawned env (in `inner` before `claude`), stores the path in the record, and appends a standard
  prompt trailer (before ending, write outcome/what-changed/action-needed). Demote exit codes to
  liveness-only and document the observed **129-on-clean-exit race** (`spawn-claude.sh:53-68`) in
  spawn SKILL.md. Update `skills/escalation-watcher/SKILL.md` guidance: on spawn completion, **read
  `result.md`** instead of exploring. *(Dep T4 serializes spawn-claude.sh + spawn/SKILL.md; dep T1
  serializes escalation-watcher/SKILL.md.)*
  **Signal:** a spawned session writes `result.md` with the outcome header; the record's
  `result_file` points to it; spawn SKILL.md documents exit-codes-as-liveness + the 129 race;
  escalation-watcher guidance says read `result.md` on join.

- **T6 — Hooks trio (SessionStart / Notification / Stop)** *(item 6; NOT simple; deps: T3).*
  Hook scripts (in-repo, absolute-path-referenced) **merged** into the existing `~/.claude/settings.json`
  `hooks` object (must not disturb `PreToolUse`/`PostToolUse` or other top-level keys). SessionStart
  upserts/refreshes the registry record via the T3 helper (reading `CLAUDE_SPAWN_*` env when present;
  **also captures Leo's hand-launched sessions**). Notification → status `awaiting-input` + OSC
  retitle `⏸ AWAITING <title>`. Stop → status `idle` + OSC retitle. Fail-soft + fast (runs every event).
  **Signal:** a hand-launched session (no `CLAUDE_SPAWN_*`) gets a registry record via SessionStart;
  when it awaits input, its tab retitles to `⏸ AWAITING …` and the record shows `awaiting-input`;
  existing `PreToolUse`/`PostToolUse` hooks remain intact.

- **T7 — Role leases (single-owner-per-role)** *(item 7; NOT simple; deps: T3, T5).*
  Lease API in the T3 helper: atomic `O_EXCL` claim under `~/.claude/fleet/leases/`, holder identity,
  heartbeat (mtime touch), stale detection (heartbeat-age + pid-liveness), deterministic
  second-claimant behaviour (stand-down for watchers / warn-and-proceed configurable). Wire into
  `skills/escalation-watcher/SKILL.md` startup (**claim FIRST, replacing pgrep archaeology**; release
  on clean exit; reap stale) and `/unblock` startup (claim `unblock-<proj>#<task>`; second `/unblock`
  on the same task is the reify 06-28 near-dup class). Auto-watcher rotations never claim interactive
  leases. *(Dep T5 serializes escalation-watcher/SKILL.md; dep T3 for the helper + substrate dir.)*
  **Signal:** a second watcher for a held role prints `lease held by <session> (alive, heartbeat Ns
  ago) — standing down` and exits; a second `/unblock <same task>` reports the holder; a stale lease
  (dead holder) is reaped and re-claimable.

**DAG:** `T1, T2, T3` have no deps. `T4 → {T3, T2}`. `T5 → {T4, T1}`. `T6 → {T3}`. `T7 → {T3, T5}`.
Critical path: `T3 → T4 → T5 → T7` (T2 feeds T4, T1 feeds T5, T6 parallel off T3).

## 8. Open questions (tactical — resolved at implementation time)

- **Failed-to-start exit code (T4):** must be distinct from `{0–125, 126, 127, 129, 143, 2}` and
  documented in the exit-code table. Candidates: a value in `144–199` (above the common `128+signo`
  band), or ceding a code from the launcher-condition set. Any distinct, documented choice is
  acceptable — exit codes are demoted to liveness-only by T5, so this is not load-bearing.
- **Hook script language (T6):** thin bash entrypoints calling the Python helper, vs. Python hooks
  directly. Either works; pick per fail-soft + speed (hooks run on every event).
- **Exact glyphs / reaper TTLs / registry sub-layout** — defaults given in §4; finalize in code.

---
*Metadata note for the orchestrator: tasks carry `user_observable_signal`, `consumer_ref`, and a
substrate-confirmed flag in metadata. The orchestrator does not currently read these fields — they
are substrate for a future tracking-infra session.*
