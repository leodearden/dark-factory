# Capability Manifest — session-attention-rail

Mechanizes G3+G6 for `plans/session-attention-rail-prd.md`. One block per task; each capability
the task's signal asserts is bound to evidence verified 2026-07-07. Any FAIL value blocks the batch.
All bindings below resolve **PASS** — the change is new substrate (registry/leases/hooks) plus
wiring into an existing spawn chokepoint whose seams were all confirmed present.

Task labels T1–T7 match the PRD §7 decomposition. Task IDs filled in after filing (see the
decompose hand-back / `prd_task_label` metadata).

## T1 — Harden 5 escalation-watcher spawn call sites  *(LEAF; `simple`)*
- title plumbing exists at spawn substrate → `spawn-claude.sh:162,170,178,186` (gnome/xterm/kitty/konsole) — **PASS**
- 5 call sites present to edit → `skills/escalation-watcher/SKILL.md:395,403,422,456,518` — **PASS**
- enriched prompt keeps `/unblock` triggering (anti-false-premise, G6) → `/unblock` Step 0 extracts the id, Step 1 re-derives from `TASK_ID` (`skills/unblock/SKILL.md:23-60`); leading `/unblock <id>` preserved — **PASS**
- esc-id/category/severity/summary available to the watcher at spawn time → escalation payload fields (`get_pending_escalations`) — **PASS**
- no numeric/exactness claim asserted → G6 branches 1/2 N/A — **PASS**

## T2 — Title convention doc + factory-init titles  *(LEAF; `simple`)*
- `terminal_title` arg exists to document/require → `skills/spawn/SKILL.md:45`, positional `'<title>'` in `:53` — **PASS**
- factory-init spawn steps present to edit → `skills/factory-init/SKILL.md:110-119` — **PASS**
- docs-only; no runtime capability asserted beyond existing title plumbing (see T1) — N/A

## T3 — Session registry substrate  *(spine root; has its own signal)*
- registry dir is writable and NOT harness-owned (anti-collision) → `~/.claude/fleet/` new namespace; `~/.claude/sessions/` confirmed harness-owned (`<pid>.json` w/ `sessionId`/`peerProtocol`) and deliberately avoided — **PASS**
- single-writer atomic-write + TTL/pid reaper idiom to mirror → `git_ops.py:1983-1991` (stamp), `reap_interactive_worktrees:5947`, `ReapedInteractiveWorktree:555` — **PASS**
- write/update injection points in the chokepoint → after arg-parse `spawn-claude.sh:37`; exit at `finish()` `:92-99` — **PASS**
- fail-soft is achievable without altering exit contract → `finish()` reads sentinel independently of any record write; registry write is additive — **PASS**
- testable via existing harness → `tests/scripts/test_spawn_claude.py` (fake-terminal + fake-claude + `SPAWN_LAUNCH_GRACE_SECS`) — **PASS**
- field-population (anti-sentinel): record carries real status transitions + exit_code, not a stub → asserted by the fail-soft + status-transition signal, exercised end-to-end in test — **PASS**
- schema/key **contract** is the shared seam (G5/B+H) → owned here; documented dataclass + `SCHEMA_VERSION` + two-way write→refresh→reap boundary test — **PASS**

## T4 — Verify-spawn-started + failed-to-start  *(LEAF; consumes T3)*
- registry status-write capability → `producer:T3` upstream (dep) — **PASS**
- transcript-appearance signal is real (anti-false-premise, G6) → encoding `/home/leo/src/dark-factory`→`~/.claude/projects/-home-leo-src-dark-factory/` confirmed (dir exists; `.`→`-` via `-home-leo--openclaw-workspace`) — **PASS**
- the unbounded-hang path it must break exists (motivating P4) → `await_sentinel` `spawn-claude.sh:126` blocks unbounded on detached launch_rc==0 no-sentinel — **PASS**
- distinct exit code is producible (unused space exists) → `{0–125,126,127,129,143,2}` in use; `144–199` free — **PASS**
- loud caller-visible surfacing is observable → background-task stdout/stderr reaches the calling session (per `/spawn` completion contract, `skills/spawn/SKILL.md:8,49`) — **PASS**
- DAG-direction: T4 depends-on T3,T2; no producer downstream — **PASS**

## T5 — Result-handback protocol  *(LEAF; consumes T3, T4)*
- env-injection + prompt-trailer point → `inner` payload `spawn-claude.sh:69-72` (export before `claude`) — **PASS**
- record holds `result_file` path → `producer:T3` schema field (upstream) — **PASS**
- 129-on-clean-exit race is real and documentable (G6 truth) → `spawn-claude.sh:53-68` header on signal dispositions; brief P4 observation — **PASS**
- escalation-watcher guidance surface exists to update → `skills/escalation-watcher/SKILL.md` (guidance/tracking sections, e.g. `:564`) — **PASS**
- exit-code demotion does not break existing callers (backward-compat) → codes remain returned; only *semantics* documented as liveness — **PASS**
- DAG-direction: T5 depends-on T4,T1; no producer downstream — **PASS**

## T6 — Hooks trio (SessionStart / Notification / Stop)  *(LEAF; consumes T3)*
- hooks substrate is available and events unused → `SessionStart`/`Notification`/`Stop` absent from `~/.claude/settings.json` `hooks` (only `PreToolUse`/`PostToolUse` present) — **PASS**
- MERGE-not-clobber is required and feasible (anti-regression, G3 contradiction fix) → existing `hooks` object has `PreToolUse`(Bash,EnterWorktree)+`PostToolUse`(ExitWorktree); new event keys are additive — **PASS**
- SessionStart can upsert the record via T3 helper → `producer:T3` upstream — **PASS**
- captures hand-launched sessions (no `CLAUDE_SPAWN_*`) → SessionStart fires for all sessions; helper keys on session identity — **PASS**
- OSC retitle is emulator-agnostic → `\033]0;…\007` honored by all four target emulators (no DBus) — **PASS**
- DAG-direction: T6 depends-on T3 — **PASS**

## T7 — Role leases (single-owner-per-role)  *(LEAF; consumes T3, T5)*
- atomic-claim primitive → `O_EXCL` create under `~/.claude/fleet/leases/` (T3 substrate dir) — **PASS**
- heartbeat + stale (mtime + pid-liveness) idiom → mirrors reaper pattern (`git_ops.py` reaper, T3) — **PASS**
- escalation-watcher + `/unblock` startup surfaces exist to wire → `skills/escalation-watcher/SKILL.md` startup; `skills/unblock/SKILL.md:21-33` Step 0 — **PASS**
- auto-watcher exclusion is enforceable (anti-false-premise on lease scope) → lease API invoked only from interactive skills; `escalation-watcher-auto` already forbidden from spawning (`skills/spawn/SKILL.md:85`) so it never calls the API — **PASS**
- second-claimant signal is observable (stand-down line) → deterministic report from holder-record read — **PASS**
- DAG-direction: T7 depends-on T3,T5; no producer downstream — **PASS**

---
**Anti-orphan check:** every mechanism has a wired consumer within this batch or a named future
consumer (attention manifest — schema owned by T3, migration-free). No producer task lacks a
downstream reader. **Integration seam (T3 schema/key)** carries a two-way boundary test per G5.
