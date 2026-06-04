# PRD-1 — Escalation watcher & queue operations hardening

**Status:** active · 2026-06-04
**Origin:** Brief 1 of `plans/escalation-flow-2026-06-04-prd-briefs.md` (16-agent verified audit, all anchors re-verified against the working tree in this authoring session).
**Seam register:** the static register in the briefs file governs file ownership; newly discovered seams logged in `plans/escalation-flow-gaps-prd1.md`.

## Goal

Increase factory throughput by cutting issue-handling latency and rotation cost — without touching correctness of code reaching main:

- L1 watcher rotations exit **cleanly with digests** instead of being force-killed every cycle (live: 10/10 unclean in journal, kill at exactly `rotation_hours*3600 + 300s` = 14702s observed; up to ~$40 burned per quiet rotation; post-kill backoff 30s·2ⁿ adds L1 pickup latency).
- An L1 born during watcher arming is picked up **immediately**, not on the next unrelated escalation (the L2 analog sat 21h — esc-1573-8).
- `get_pending()` stops paying a **683-orphan tax** (~28ms warm, paid per drain/submission/promote/steward tick; live-verified this session: 445 resolved + 238 dismissed in queue root, 11 loose files at archive top level).
- Concurrent member-appends to an L2 **cannot lose data** (queue.py:478 documents the RMW clobber).
- Born-at-L2 `critical`/`urgent` pushes **actually interrupt** an AFK human (`watcher.py:29` maps only `blocking`→urgent today — priority inversion against the L2 skill's promise at escalation-watcher/SKILL.md:194-195).

## Background

3-tier ladder: L0 agent→steward, L1 steward→escalation-watcher-auto (harness-internal supervisor running `claude --print` rotations, harness.py:3621-3676 / supervisor loop :3678+), L2 auto-watcher→human. The rotation agent is told to block on `python -m escalation.watcher --level 1` as a single foreground Bash call (escalation-watcher-auto/SKILL.md:221-230), but the Bash tool caps at 10 min and no override is injected (`shared/src/shared/cli_invoke.py:822` copies `os.environ` verbatim minus the API key). The watcher CLI has no `--timeout` and no initial scan (watcher.py — verified). `escalation.sweep --apply` fully implements root↔archive reconciliation but is wired nowhere; `escalation.archive` `prune_archive` is manual-CLI-only and skips loose top-level files.

## Sketch of approach (issue → mechanism)

| # | Issue | Mechanism |
|---|---|---|
| 1 | Rotation force-killed | `watcher.py --timeout <secs>` (inotify `read(timeout=ms)` — verified supported) + auto-watcher SKILL loop re-arms bounded waits against a rotation deadline + harness injects `BASH_MAX_TIMEOUT_MS` sized to the rotation via existing `env_overrides` kwargs pass-through (no shared/ edit) |
| 2 | Drain-before-up race | Initial scan inside watcher.py: arm inotify watch FIRST, then scandir for already-pending matches, emit-and-exit if found — closes the gap for every consumer (L1 skill, L2 skill, ad-hoc) |
| 3 | Queue-root hygiene | Server-start hook (escalation pkg, at/near `create_server()` — the pre-serving single-writer window) runs sweep + extended reaper + retention prune; one-time `--apply` relief run on first deploy |
| 4 | Member-append clobber | Per-id **sidecar** flock adopted by ALL queue-root read-modify-write mutators |
| 5 | ntfy priority inversion | `urgent` iff `severity ∈ BORN_AT_L2_SEVERITIES ∪ {'blocking'}`; title tag = severity upper-cased |
| 6 | Doc truth | Drop obsolete fd-pool rationale (escalation-watcher-auto/SKILL.md:230 — contradicts escalation-watcher/SKILL.md:577 "historical — no longer expected"); add watcher.py module docstring stating the two load-bearing invariants |

## Resolved design decisions

- **D1 — Rotation fix is `--timeout` + env belt (both).** `--timeout` is primary (bounded waits, rotation-clock recheck, clean digest exit, also caps issue 2's residual window); `BASH_MAX_TIMEOUT_MS` injection at the `_run_watcher_rotation` call site is the belt so a future mis-sized wait never hits the 10-min cliff. Zero shared/ changes — `invoke_with_cap_retry(**invoke_kwargs)` forwards `env_overrides` (verified).
- **D2 — Sweep wiring lives in the escalation package at server start.** Pre-serving is the one guaranteed single-writer window; runs every orchestrator start (sufficient at current volumes). NOT in harness startup (~743 is PRD-3-owned) and NOT a systemd timer (installed units are local-only per project convention; a timer would race a live server).
- **D3 — flock ALL mutators via per-id sidecar lockfiles** (`esc-<id>.json.lock`, never renamed). Locking the data file itself is defeated by the atomic tmp+rename writer invariant — the lock binds to the **old inode** after a rename-replace, so a second writer locking the new path races anyway. Sidecar + `fcntl.flock(LOCK_EX)` around read→modify→tmp+rename. Adopted by every queue-root mutator (resolve, dismiss, `add_members_to_l2`, `attach_dedupe_child`, submit collision paths) **and** by ε's sweep relocations — encodes the invariant once rather than patching the documented call site only.
- **D4 — Approach: bare-B slices + one contract section** for the watcher CLI (the only real seam: three consumer classes read it). No full boundary-test sketch — the six issues are near-independent hardening, not a coupled state machine (the audit reserved design-first for PRD-3).
- **D5 — Initial scan is default-on, no flag.** Arm-then-scan ordering guarantees no gap (files created before `add_watch` are seen by scandir; after, by inotify; overlap is deduped by exit-on-first-match). Safe for all existing consumers under the wake-signal-only invariant — an instant fire on a pre-existing pending item is exactly a legitimate wake.
- **D6 — The sweep/reaper glob stays `esc-*.json` and is never widened.** Protects PRD-2's `b3-state.json` and the existing `afk-digest.md` (gaps file, PRD-2 entry 1 — acknowledged in `plans/escalation-flow-gaps-prd1.md` entry 4). ε carries a regression assertion: non-`esc-*` queue-root files are untouched by sweep/reaper.
- **D7 — Watcher exit-code contract:** `0` = one matching escalation printed to stdout; `124` = `--timeout` expired, nothing printed (coreutils convention); SIGTERM → exit 0 (unchanged); argparse errors keep argparse semantics.

## Watcher CLI contract (the seam — consumed by L1 skill loop, L2 skill flows, ad-hoc humans)

```
python -m escalation.watcher --queue-dir <path> [--task-id <id>] [--level <n>]
                             [--ntfy-url <url>] [--timeout <secs>]
```

- `--timeout <secs>`: maximum blocking wait. Absent → block indefinitely (current behaviour preserved for existing callers).
- **Startup order:** arm inotify watch → scandir queue root for already-pending matches (oldest `timestamp` first) → if found, print it and exit 0 → else enter the event loop.
- **Exit codes:** see D7. On exit 124 stdout is empty — consumers re-check their own deadline and re-arm or exit.
- **ntfy:** `Priority: urgent` iff `severity ∈ BORN_AT_L2_SEVERITIES ∪ {'blocking'}`; title tag `[<SEVERITY-UPPER>]`; failures logged to stderr, never fatal (unchanged).
- **Module-docstring invariants (load-bearing, to be stated verbatim):**
  1. All escalation-queue writers are atomic tmp+rename (incl. fused-memory recon) — a partial read off an inotify event is impossible.
  2. The watcher is **wake-signal-only; drains are authoritative.** Consumers MUST re-drain after every watcher return and treat spurious wakes as normal. (This is what makes D5 safe.)

## Pre-conditions for activating

None — all assumed substrate verified to exist on main, 2026-06-04, this session: `inotify_simple.INotify.read(timeout=…)`; `env_overrides` plumbing through `invoke_with_cap_retry`; `sweep.py --apply`; `archive.prune_archive`; `BORN_AT_L2_SEVERITIES` (models/server chokepoint). `BASH_MAX_TIMEOUT_MS` is a documented Claude Code settings env var; it is belt-only (D1) — harmless if a future CLI ignores it.

## Cross-PRD relationship

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| PRD-2 (B3 hardening) | constrains this | `data/escalations/` non-escalation residents (`b3-state.json`, `afk-digest.md`) vs ε's reaper | this-prd (D6) | wired (glob invariant + regression assertion) |
| PRD-3 (re-pend state machine) | informs other | `server.py` same-file/different-section: ε adds a startup-sweep region; PRD-3 owns CATEGORIES + `resolve_issue` docstring | each-its-section | logged (gaps entry 2) |
| PRD-3 | informs other | watcher initial-scan makes instant-fire-on-pending guaranteed; L2 SKILL.md launch prose (PRD-3-owned) stays correct, optional one-liner | other-prd | logged (gaps entry 3) |
| PRD-3 | respects | No new escalation categories or re-pend semantics introduced here (register) | other-prd | n/a |
| (register: nobody) | respects | `merge_queue.py` verify path treated as existing invariant, untouched | — | n/a |

PRD-1 makes **zero edits** to `skills/escalation-watcher/SKILL.md` (PRD-2/PRD-3 territory) and **zero edits** to `shared/` (gaps entry 1).

## Decomposition plan

Per project convention each task touches a single package. Greek letters = intra-batch deps.

- **α — Harden `escalation.watcher` CLI: `--timeout`, initial scan, ntfy severity, invariants docstring** *(escalation pkg; watcher.py only)* — **intermediate**, unlocks γ.
  Observable signals: (1) with a pending L1 already in the queue, launching the watcher emits it immediately and exits 0 (no inotify event needed); (2) on an empty queue, `--timeout 2` exits 124 within ~2s with empty stdout; (3) a `critical` born-at-L2 with `--ntfy-url` pointed at a local HTTP sink receives `Priority: urgent` + `[CRITICAL]` title.
- **β — Per-id sidecar flock for all queue-root mutators** *(escalation pkg; queue.py)* — **leaf**.
  Observable signal: a CI fixture runs two concurrent OS processes appending different member sets to the same pending L2 via the real queue API; the on-disk record read back through the queue read path contains the union (today it provably loses one set). `add_members_to_l2`'s "Not concurrency-safe" docstring replaced by the lock contract.
- **γ — Rewrite auto-watcher Main Loop for bounded waits + clean rotation exit** *(skills pkg; escalation-watcher-auto/SKILL.md only)* — depends **α** — **leaf**.
  Loop computes the rotation deadline at startup, waits `min(timeout_default, remaining)` per arm, re-drains after every return (incl. 124), exits cleanly with digest when the deadline or escalation count trips; deletes the obsolete fd-pool rationale (:230); documents the initial-scan semantics.
  Observable signal: post-deploy journal over a rotation window shows clean rotation exits with digests and **zero** force-kills at the 14700s signature; a synthetic L1 filed between skill startup and watcher launch is drained without a second L1 being needed.
- **δ — Inject `BASH_MAX_TIMEOUT_MS` into watcher rotations** *(orchestrator pkg; `_run_watcher_rotation` call site only)* — **leaf**, independent (belt works regardless of α/γ).
  Observable signal: dispatch log line records the injected value (`rotation_hours*3600 + grace`, in ms) at each rotation start; a >10-min single blocking Bash wait inside a rotation survives.
- **ε — Wire sweep + reaper into escalation server start; one-time relief** *(escalation pkg; sweep.py/archive.py + a startup hook at/near `create_server`)* — depends **β** (relocations take the sidecar lock) — **leaf**.
  Extends the reaper to root orphans (sweep already does) and loose `esc-*.json` at archive top level (relocate into dated subdirs by `resolved_at`); retention prune; revises sweep.py's "single manual deploy run only" warning to "server-start single-writer window only"; keeps the `esc-*.json` glob (D6) with a regression assertion that non-`esc-*` root files are untouched.
  Observable signal: after the next orchestrator restart, the server startup log emits the sweep report line and the queue root contains only pending `esc-*.json` (683 orphans at authoring time → 0); archive top level contains no loose `esc-*.json` files.

DAG: α→γ; β→ε; δ independent. No cross-package task; no task touches more than one SKILL.md.

## Capability manifest (draft bindings — committed beside this PRD at decompose)

| Leaf | Capability | Evidence |
|---|---|---|
| β | queue-root mutators enumerable; atomic tmp+rename writers | grep: queue.py mutator set; briefs "all writers atomic" (audit-verified) |
| β | `fcntl.flock` sidecar viable | stdlib; D3 inode analysis |
| γ | `--timeout`/exit-124/initial-scan | producer: task-α (upstream ✓) |
| γ | turn budget fits bounded waits | `watcher_max_turns=400` (config.py:983) vs ~27 waits/quiet 4h rotation at 540s |
| δ | `env_overrides` reaches subprocess env | cli_invoke.py:377,405,825 + `**invoke_kwargs` pass-through (verified) |
| δ | `BASH_MAX_TIMEOUT_MS` honoured by Claude Code Bash tool | documented settings env var (belt-only if drift) |
| ε | sweep/prune logic exists | sweep.py:116 (`--apply`), archive.py:46 |
| ε | single-writer window at server start | harness.py:610 starts serving after create; MCP-only external writers |
| ε | 683→0 is achievable | live dry-run counts match brief; signal phrased as "only pending remain" to survive drift |

## Out of scope

- New escalation categories, re-pend semantics, gate predicates, `resolve_issue` docstring (PRD-3; register).
- Any edit to `skills/escalation-watcher/SKILL.md` or `dry_run_unblock.py` / B3 mechanics (PRD-2/PRD-3).
- Any edit to `shared/cli_invoke.py` (gaps entry 1).
- `merge_queue.py` verify path (register: nobody).
- **Verified non-issues — do not re-open:** inotify partial-read (atomic writers); queue.py archive-scan TODOs (~2ms at live volume).

## Open questions (tactical — decide at the named task)

1. **`--timeout` default used by the skill loop.** Suggested 540s (margin under the 600s cap; ~27 waits per quiet 4h rotation ≪ 400 max turns). Decide in γ.
2. **Also set `BASH_DEFAULT_TIMEOUT_MS`?** Probably not — the skill passes an explicit per-call timeout; only MAX matters. Decide in δ.
3. **Retention days for the server-start prune.** Reuse `escalation.archive` CLI default vs a config field. Decide in ε.
4. **Whether α emits the matched escalation's id to stderr on exit 124 paths for debugging.** Cosmetic. Decide in α.
