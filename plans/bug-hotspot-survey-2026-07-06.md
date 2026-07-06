# Bug Hotspot Survey — 2026-07-06

Method: mined 45,861 commits since 2026-01-01 (5,225 fix-flavored, 11%), the 1,117-task
tracker, and plans/ postmortems for recurring fix themes (50 found); 12 deep architectural
reviews (one per hotspot cluster); every finding adversarially verified against the code
(75 findings: 72 confirmed, 3 weakened, 0 refuted); cross-system synthesis on top.
Full machine-readable findings (file:line evidence per finding):
`plans/bug-hotspot-survey-2026-07-06-full-findings.json`.

## Ranked hotspots

| # | Hotspot | Evidence | Root structural cause |
|---|---------|----------|----------------------|
| 1 | merge-queue (`merge_queue.py` 9.4k lines + 12 satellites) | 248 fix commits; still #1 churn (180 changes) in last 3 weeks, *after* the 17-task refactor | Lifecycle & permits tracked by census/flags, not ownership; refactor extracted the wrong seam |
| 2 | workflow (`workflow.py` 8.8k) | 59% fix ratio (highest of any big file) | Terminal state quadruplicated; steward outcome inferred from side effects; "already merged?" answered heuristically 3 ways |
| 3 | harness (`harness.py` 9k, `service_restart.py`) | 195 fix commits; shutdown-hang + restart incident arcs (2064, 2105, 2091) | 4 parallel restart mechanisms; 11 hand-rolled loop triplets; 7 sweeps re-deriving task ground truth |
| 4 | git-worktrees (`git_ops.py` 6.8k, `warm_lane_pool.py`) | 135 fix commits; two CRITICAL live incidents (Jul 3–5) | Lane lifecycle has no durable authoritative state; invariants enforced by convention |
| 5 | fm-recon (`task_knowledge_sync.py` 4.2k, recon `harness.py` 2.9k) | Hottest fused-memory fix surface (~871 recon commits since April) | Control-plane ledger stored in an eventually-consistent vector store; self-model lives in prompt prose |
| 6 | fm-task-layer (`task_interceptor.py` 4.1k, `tools.py` 4k) | Dedup defect corrupted its own tracker (dup ID pairs 999/1000, 1026/1028…) | Metadata is an unversioned cross-process contract; dedup fails open to CREATE |
| 7 | shared-infra (`usage_gate.py`, `cli_invoke.py`) | 68% fix ratio — highest in repo | 6-flag account state bag; outcome classification duplicated 5×; steward forked the retry loop |
| 8 | verify (`verify.py` 3.6k, `b3_gate.py`) | 9× "wrong shell command" theme; 3× classifier re-grounding | Commands are strings mutated by textual surgery; scope policy computed twice; classifier is tool-blind |
| 9 | scheduler (`scheduler.py` 4.7k, `deterministic_runner.py`) | Stranded-task + external-dep arcs (1854/1855/1799, esc-2073-15) | No status transition authority; deploy state = stamp archaeology |
| 10 | fm-memory (`memory_service.py`, `graphiti_client.py`) | ~30-task rebuild/refresh hardening chain | No write-time entity identity guarantee; edge-uuid uniqueness deliberately broken |
| 11 | escalation (`server.py`, `queue.py`) | resume/restart semantic traps; L1→L2 overstep | Action semantics split across 2 files; level gate is caller-declared |
| 12 | dashboard (`data/*.py`) | 46% fix ratio | Scrapes orchestrator internals with no shared vocabulary |

## Latent bugs found during the survey (small, immediately actionable)

1. `harness.py:366` `_recon_inspect_unit` is a diverged copy of the runner's inspector
   **missing the task-2091 timeout guard** — a wedged `systemctl --user show` silently
   stops deterministic-strand recovery forever.
2. `git worktree prune` has **5–6 raw call sites bypassing the 2099 registration-wipe
   guard** (git_ops.py:3824, 4341, 4426, 5365, 5725; harness.py:4765) — the Jul-4
   incident can recur through the merge-lane paths during a mount-down window.
3. `acquire_warm_lane`'s top-level `except Exception` (git_ops.py:3060-3065) does a bare
   `pool.release()` with no HEAD detach — **re-creates the task-2062 stale-checkout
   collision** on any post-checkout fault.
4. `deterministic_runner.py:952`: `ever_escalated = bool(get_by_task(task_id))` with no
   `agent_role` filter and archive included — **any unrelated escalation (e.g. starvation
   watchdog) aliases as "human resolved the deploy"** and can drive the task to done.
5. Dashboard clock-skew race fixed in `merge_queue.py` (task 692) is **still latent in
   `burndown.py:509` and `costs.py:33/440`** — same asyncio.gather fan-out shape.
6. Steward's hand-rolled cap-retry loop (steward.py:536-607) lacks the zero-output wedge
   guard — a wedged steward session is re-resumed forever.
7. Scheduler substrate-probe predicate duplicated with divergent semantics — malformed
   probe **fails open at the real dispatch gate**.
8. Dashboard `_TERMINAL_MERGE_OUTCOMES` frozen ~2 months ago while orchestrator kept
   adding outcome strings (latest 2 days ago) — in-flight classification silently stale.

## Per-hotspot proposals

### 1. merge-queue
- **SpecPermit token + PermitLedger**: permits are anonymous semaphore decrements, so the
  conservation audit must census five containers (+ `_finalizing_head`, `_dispatching_item`
  — each added by an incident: 2063/2068/2096, I4 alarm storms). Own the semaphore behind
  a ledger; token travels on the item; release idempotent. Conservation becomes structural.
- **ItemLifecycle registry**: item state = container membership + 4 nullable side-fields +
  3 parallel status mechanisms (free-form `phase` str, `InflightStatus`, vestigial
  dual-written `_verify_phase`). One state enum + legal-transition table; every put/pop
  calls `transition()`; snapshot/audit/liveness become single reads.
- **Landed-outbox in MergeQueueStore** (see cross-system chain #1): write (task_id,
  branch_tip_sha, advanced_sha) fsynced **before** the CAS main advance; startup
  reconciler drives done-with-provenance from unconsumed rows.
- Unfreeze the satellite seams: satellites are function-bags over worker privates, wired
  through circular reach-back imports frozen by test monkeypatch paths; retired serial
  worker + compat shims are load-bearing test anchors. One-time monkeypatch-path migration.
- **QueuedBranch** typed branch identity parsed once at the boundary.

### 2. workflow
- **One durable merge-provenance lookup** replaces the three deliberately-divergent
  "already merged?" guards (workflow.py:7291, :1776-1816, merge-phase ancestor check) and
  the 90-line `_has_prior_implementation` heuristic (tasks 846/851/882/883/954/1141).
- **StewardOutcome sum type** (RESOLVED | REESCALATED_L1 | TERMINAL_DECISION |
  INTERRUPTED(wip_present) | BUDGET_EXHAUSTED) returned by the steward, replacing five
  probes over escalation-queue side effects + timestamp windows. INTERRUPTED encodes the
  "triage WIP health before restarting" lesson structurally.
- **WorkflowStateMachine + TerminalReport**: legal-transition table with DONE/CANCELLED
  absorbing; replace the `_last_block_reason/phase/detail` attr side-channel to harness
  with a returned dataclass; exit-assert WorkflowState/Outcome/status-row consistency.
- **classify_failure → BlockDisposition table** replaces run()'s seven-clause exception
  ladder re-implementing other subsystems' failure classification.

### 3. harness
- **`proc_supervision.py` with one RestartPlan/execute()**: 4 restart mechanisms each
  re-discovered a different subset of {absolute path, cwd, on-failure wrapper, fresh-PID
  verify, own-unit detection}. Derive detached-vs-blocking from own-unit comparison; fail
  closed on unknown own-unit. Would have prevented 2064 and 2105 outright.
- **Hoist `inspect_systemd_unit(unit, timeout)`** (fixes latent bug #1 above).
- **BackgroundService/LifecycleRegistry**: 11 `_start/_stop/_loop` triplets collapse to
  registrations; stop_all() in reverse order with per-service timeout kills the
  shutdown-hang class (tasks 108, 161/162/169, 875, 1080).
- **TaskGroundTruth resolver**: one `derive_truth(tid) → TruthReport` + one
  classification table replaces 7 sweeps re-deriving state from 5 substrates; new crash
  shapes become table rows, not new 400-line sweeps.
- **DeployState typed schema** (phase enum + persisted fresh-PID verify baseline) replaces
  stamp-combination archaeology (1900 phantom-done, 2059 strand, 2066 lost writeback).

### 4. git-worktrees
- **`_prune_registrations(context)` chokepoint** — only place the argv literal may appear;
  refuse when pool in use and pool storage absent; enforce via grep-guard test (same
  pattern as existing .task defense tests). Fixes latent bug #2.
- **`_abort_lane_acquisition` teardown primitive** routed from every fault exit including
  the top-level except (fixes latent bug #3); test injects post-checkout exception and
  asserts detached HEAD.
- **LaneLifecycle single-writer + durable per-lane record** `<mount>/.lane-state/<lane>.json`
  (state, task_id, branch, seeded_from_sha) — record lives on the pool mount so it
  vanishes with the mount, coherent with the 2099 sentinel. Startup recovery = read record,
  verify git reality, quarantine on divergence (replaces the heuristic adopt/clean/re-pin
  tree; kills the 2097/2098 class).
- **Move `.task/` out of the git tree** to `<worktree_base>/.task-meta/<worktree>/`
  (quarantine_base sibling pattern). The entire 6-guard defense layer (scrubs, pathspec
  exclusions, post-staging nets, per-incident rmtrees) becomes dead code; contamination of
  main becomes structurally impossible; metadata survives `git clean -xfd`.

### 5. fm-recon
- **ReconLedgerStore (SQLite)** for markers/suppressions/counters: marker replacement
  becomes one UPSERT; kills flag_dedup's 7 stacked compensations (241-line docstring) and
  collapses 4 GC sweeps to one DELETE. Mem0 keeps only a searchable narrative mirror.
- **Dedup-exempt system-write path** in MemoryService for recon agent_ids: deletes the
  summary_nonce/retry_nonce arms race (1590/1796/1821, 2777f2b227) and the entire
  verify→repair→reconstruct chain (tasks 1963/1964). Cycle summaries written once,
  deterministically, by Python.
- **ProjectScope frozen dataclass** (NewType'd ProjectId/ProjectRoot), constructed only at
  `_known_project_root_for`, required in BaseStage.__init__ — kills the 10× swap class
  (156, 186, 927, 930-963) at pyright time.
- **ReconWritePolicy at the interceptor write boundary** (reject terminal update_task,
  live-workflow status writes, stale snapshots — structured errors the LLM reads mid-run)
  replaces the ~650-line post-hoc `_apply_post_flight_guards` forensics.
- **`recon_self_model.py`** renders prompt sections from code-owned constants (prompt/code
  drift impossible) + require `metadata.execution_class ∈ {code_tdd, operational, decision}`
  on recon-filed tasks — kills the false-premise class (2083/2092/2093) and the TDD
  mis-routing class (2085).

### 6. fm-task-layer
- **`shared/task_metadata.py`**: versioned pydantic TaskMetadata with typed BeforeDone /
  DoneProvenance / MemoryHints / ExternalDep, validated at the SqliteTaskBackend write
  boundary; delete the 8 divergent parsers. Proof of need: task 1902 added provenance kind
  `deterministic-deploy-scheduled`, fused-memory's allowlist never updated → every
  self-restart deploy's done write silently rejected (1976/1982).
- **Durable `candidate_key` UNIQUE partial index** on tasks: dedup currently fails open to
  CREATE on every failure branch, hence six stacked in-memory dedup layers with different
  keys — and the tracker corrupted itself. Constraint violation resolves the ticket
  "combined"; the caches become optimizations.
- **One privileged update_task write-authority seam** instead of three drifting guard copies.

### 7. shared-infra
- **AccountPhase StrEnum + single `_transition()` writer** owning side effects and the
  global `_open` recompute (currently recomputed at 6+ sites; every transition site must
  manually clear the other 5 flags — tasks 336, 629/630, 729, 805/806 are all instances).
- **`InvocationOutcome` sum type + one `classify_invocation()`** holding all string tables;
  `InvokeSlot.report(outcome)` replaces cli_invoke's reach-ins to gate privates. Ends the
  10× cap-string whack-a-mole and 4× probe-slot leak themes at one seam.
- **Delete the steward's forked retry loop**; extend `invoke_with_cap_retry` with
  `rebuild_prompt` + `max_cap_retries` hooks (fixes latent bug #6).

### 8. verify
- **VerifyCmd structured model** parsed once at config load (unparseable → OPAQUE, never
  scoped): scoping = replace targets; reprojection = set uv_project; cd-strip = clear
  cwd_rel. Kills the find/replace helper stack (tasks 1077, 1643, 2036…).
- **`derive_verify_plan()`**: single pure scope derivation returning a declarative,
  serializable plan — the conftest and data-module bugs were each fixed twice because
  `scope_module_config` and `_build_fallback_config` duplicate the policy.
- **Tool-dispatched failure classifier** (+ structured output where tools offer it:
  `pyright --outputjson`, `ruff --output-format json`, `cargo --message-format json`) —
  ends the 3× cargo re-grounding / 6× flake-tightening arms race.
- **Typed BlockRecord with block_class enum**, constructed by workflow AND merge_queue;
  spawn dry-run investigation on the merge-verify block path — closes the "trivial
  merge-verify fixes always fall to human /unblock" gap (B3 currently aborts on the whole
  class because no proposal is ever produced there).

### 9. scheduler
- **TaskLifecycle (from, to, actor) table enforced in TaskInterceptor** + first-class
  claimant/heartbeat field: "stranded" becomes a queryable predicate; resume semantics fix
  (blocked→pending when no live claimant) becomes structural; harness sweeps shrink to a
  heartbeat check. (Table defined in shared/, consumed by escalation + workflow as thin
  validators — one table, not three.)
- **Scope deterministic-runner escalation queries by `agent_role`** and replace stamp
  combinatorics with one `metadata.deterministic_state` enum + transition table (fixes
  latent bug #4).
- Extract module-lock derivation (4+ duplicate sites), fix the substrate-probe fail-open,
  and decompose `acquire_next`'s 720-line tick into named ordered phases.

### 10. fm-memory
- **Write-time entity identity**: `_resolve_or_create_entity` chokepoint doing exact-name
  lookup before minting, guarded by a (group_id, name) uniqueness constraint or per-group
  lock. The 4 reactive post-hoc dedup sweeps fold into one legacy-cleanup helper.
- **Mint fresh uuids in `redirect_node_edges`** (carry `superseded_edge_uuid` as audit) —
  restores graph-wide edge-uuid uniqueness; the hand-rolled `WITH DISTINCT` idiom (already
  wrong twice: cfed95c706, task 2084) dies.
- Enforce the CancelledError re-raise convention (already regressed in 4 task-routing
  handlers; task 1151 pending) — extract the whole gather-cancellation idiom, not just the
  check.

### 11. escalation
- **One legality table `(action, level, category) → TaskEffect`** in escalation, imported
  by both `resolve_issue` (loud typed rejection on illegal combos) and harness's callback
  (computes the task effect). Precondition failures currently DEBUG-log and silently no-op
  — the exact esc-2073-15 trap.
- **Server-side role-derived level ceilings**: `level_forbidden` only fires when the caller
  self-declares the header. Map identified automation roles to ceilings; **header-less
  sessions must remain full-capability human channel** (preserves the deployed L2-closure
  convention). Gate `promote_to_l2`'s create side the same way.

### 12. dashboard
- **OutcomeKind StrEnum owned by orchestrator** (merge_types), imported by dashboard — or
  invert to terminal-unless-listed so drift fails safe (fixes latent bug #8).
- Thread request-scoped `now` through burndown/costs aggregators as merge_queue already
  does (fixes latent bug #5); grep-based CI check against bare `datetime.now(UTC)` in
  gather fan-outs.
- Extract shared MCP fan-out-with-failover + TTL cache helpers (reimplemented 4-5×).

## Cross-system defect→patch chains

1. **Merge-landed vs task-done atomicity gap (the ghost-loop web)** — merge queue advances
   main in-process; task done status written after via a lossy callback chain. Six+
   independent detectors compensate across workflow (2 guards), merge_queue, scheduler,
   harness sweeps, and fused-memory server gates; each has its own false-positive history.
   Fundamental fix: the landed-outbox write-ahead journal (priority 1).
2. **No task-status transition authority** — dead-end states (resume→in-progress,
   blocked-with-no-re-eval) are recovered by heuristic sweeps two subsystems away.
   Fix: transition table at the TaskInterceptor chokepoint + claimant/heartbeat (priority 3).
3. **Task metadata as unversioned cross-process schema** — drift ships silently in both
   directions (provenance-kind rejections, before_done.cwd 127s, memory_hints repair
   shims). Fix: shared versioned TaskMetadata (priority 4).
4. **Escalation queue as untyped cross-process RPC** — each consumer reconstructs action
   semantics; steward outcome inferred from queue side effects. Fix: legality table +
   StewardOutcome type.
5. **Claude CLI's unstructured contract** — cap/wedge classification scattered across 5
   sites and 2 retry loops in shared-infra, workflow, fm-recon. Fix: InvocationOutcome
   classifier (priority 5).
6. **Missing write-time guarantees in the memory stores** — recon and memory_service both
   patch with post-hoc sweeps and an LLM nonce arms race. Fix: dedup-exempt system writes +
   write-time entity identity (priority 6).
7. **Git-substrate invariants enforced by convention** — unguarded prune, divergent lane
   teardown, .task/ in-tree; defended by guard stacks in git-worktrees, verify, harness,
   merge-queue, fm-task-layer. Fix: chokepoints + lane state machine + .task/ relocation
   (priority 2).
8. **Merge-queue permits/lifecycle by census** — false-positive detector storms consuming
   human triage (I4 alarms). Fix: SpecPermit ledger + ItemLifecycle (priority 7).
9. **Dashboard scrapes orchestrator internals** — silent staleness by design. Fix: shared
   outcome vocabulary + served formats.

## Ranked priorities (payoff × feasibility)

1. **Landed-outbox merge journal** in MergeQueueStore, fsynced before the CAS advance +
   startup reconciler — collapses ~6 ghost-loop/phantom-done/false-done detectors across 4
   subsystems (20+ historical fix tasks).
2. **Cheap git_ops chokepoint bundle** (days): guarded `_prune_registrations` + grep-guard
   test, `_abort_lane_acquisition` primitive, PROTECTED_PREFIXES registry — directly
   prevents recurrence of two already-lived CRITICAL incident classes at near-zero risk.
3. **Server-side task-status transition authority** (shared table, enforced in
   TaskInterceptor) + claimant/heartbeat field — kills the stranded-task class and shrinks
   harness's seven sweeps toward one heartbeat check.
4. **Versioned shared TaskMetadata schema** validated at the backend write boundary —
   kills bidirectional silent schema drift.
5. **One InvocationOutcome classifier + InvokeSlot.report + AccountPhase transitions**;
   delete the steward fork; collapse workflow's exception ladder — ends the cap-string
   (10×), probe-slot-leak (4×), and AllAccountsCapped-escape (4×) themes at one seam.
6. **Recon control-plane off Mem0** (SQLite ReconLedgerStore + dedup-exempt system writes)
   — deletes the nonce arms race and verify/repair/reconstruct chain in the repo's hottest
   fix surface; companion: graphiti write-time exact-name resolution.
7. **SpecPermit ledger + ItemLifecycle registry** in merge-queue (after the one-time
   monkeypatch-path migration unfreezes seams) — ends the invisible-window false-positive
   class (2063/2068/2096, I4 alarm storms).
8. **"Parse, don't validate" typed-value sweep** — each independently landable:
   ProjectScope (kills the 10× project_id/project_root swap at pyright time), QueuedBranch,
   FailureCategory enum + policy table, AccountLease.

## Contradiction resolutions (from synthesis)

- **Transition-table home**: three reviewers proposed three homes (workflow client-side,
  TaskInterceptor, escalation). Resolution: enforcement floor at the single durable write
  chokepoint (TaskInterceptor); table defined in shared/; escalation + workflow consume
  the same table as thin validators.
- **Merge-provenance substrate**: MergeQueueStore (already durable, already journals the
  accept side) with write-ahead ordering — write-after would re-open the crash window.
- **Escalation level gate**: naive default-deny on unknown callers would lock humans out
  of L2 closure (the esc-2087-2 pain). Role-mapped ceilings for identified automation;
  header-less stays full-capability human.
- **Duplication doctrine**: task_interceptor.py:115's "duplication is cheaper than
  cross-package coupling" policy contradicts five reviewers' shared/-based proposals and
  is already violated by the CI drift-guard test. Retire the doctrine; shared/ is the
  proven sanctioned home (shared.usage_gate, shared.locking precedents).
