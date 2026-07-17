# Fused-Memory Restart Survey — 2026-07-17

**Goal assessed:** make fused-memory restarts (routine, for code deploys) data-safe,
efficient (no wasted in-progress full reconciliation runs), undisruptive to clients,
and suitable for automatic action.

**Method:** 6-dimension multi-agent survey (shutdown path, on-disk kill-safety,
reconciliation lifecycle, restart/deploy mechanics, client impact, operational
records), every finding independently adversarially verified against the code /
journal / records. 33 findings confirmed, 0 uncertain, 10 refuted-or-folded
(duplicates or overstated variants of confirmed findings). Raw findings JSON:
session scratchpad `tasks/fm_survey.json` (workflow run `wf_98bf61ab-603`).

## Headline verdict

Restarts today are **clean but wasteful and only accidentally safe**:

- The two real restarts in the retained journal window were deliberate,
  attributable, and fast (14–15s stop→started). No crash-loop is active.
- **Both** of those restarts cancelled an in-flight full reconciliation cycle
  mid-stage (`Reconciliation run … cancelled … (stage: memory_consolidator)`),
  restoring 251 / 4 drained events for full re-runs. The efficiency goal is
  currently failing on 2-of-2 observed restarts.
- The one "graceful" path (`restart-fused-memory.sh --drain`) is a documented
  no-op-that-hangs (task 2090) which every deploy convention explicitly bypasses.
- The in-process graceful shutdown (`_graceful_shutdown`) is well-engineered but
  sits **behind an unbounded uvicorn wait**, and its `memory_service.close()`
  step exhausted its 5s budget on both observed restarts — so the careful
  cleanup ordering is not reliably reached/completed in practice.
- Several client-side edges are **fail-dangerous, not fail-safe** during an
  fm outage (park-eviction guard bypass; retry budgets ≪ restart window).
- There is **no watchdog, no staleness-driven deploy, no hot-reload** for
  fused-memory — nothing to hang automation on yet.

---

## Findings by goal

### 1. Efficiency — in-progress full reconciliation runs (worst area)

| # | Sev | Finding |
|---|-----|---------|
| E1 | HIGH | **Every observed restart cancels an in-flight full cycle mid-stage.** Journal 2026-07-16 12:37 and 2026-07-17 08:12: SIGTERM → subprocess process-group kill → drained events restored → run cancelled. (ops-records) |
| E2 | HIGH | **No per-stage checkpoint/resume.** `run_full_cycle` persists watermark/completed only after all 3 stages; `CancelledError` handler marks the run `failed` and `restore_drained(project_id)` flips everything back to `buffered` — next trigger redoes Stage 1→2→3 from scratch. A cycle measures ~9–30 min (run 97b49a64 = 29:58) with per-stage budgets up to $5 (≈$15/cycle). `harness.py:1889-1929`, `harness.py:1970-2001`, `event_buffer.py:593-604`. |
| E3 | HIGH | **`drain()` cannot converge on an active project** — the task-2090 hang root cause. `_draining` is only checked in `run_loop` when spawning *new* per-project loops (`harness.py:1580`); `_project_loop` (`harness.py:1639-1786`) never checks it, so a busy project keeps launching successive cycles and `_no_active_loops()` never becomes true. |
| E4 | MED | **Every deploy convention explicitly bypasses `--drain`** ("hung per task 2090" — `deploy-w5-recon-reliability.sh:16-24`, repeated across ≥5 PRDs/RCAs). The bare `systemctl --user restart` path cancels the harness loop after 25s (`_HARNESS_CANCEL_TIMEOUT`) unconditionally. |
| E5 | MED | **Even a working `--drain` is mistimed:** `DRAIN_TIMEOUT=120` (`restart-fused-memory.sh:12`) was sized for the already-idle fast path (<5s per the `drain()` docstring); a genuinely in-flight cycle runs 9–30+ min, then the script "proceeds with restart anyway". |
| E6 | MED | **Orphan-run detection after an ungraceful kill takes up to 30 min**, wedging the project the whole time: `get_stale_runs` cutoff is age-from-`started_at` (`journal.py:629-638`, default `stale_run_recovery_seconds=1800`), not owner-death time; the project lock is only released by that same reaper pass. A kill 2 min into a run ⇒ ~28 min of wedge. |
| E7 | LOW | **`restore_drained` is project-scoped, not run-scoped** — the accepted-gap remnant of the 2026-05-28 stale-run RCA. A future live-run-misclassified-as-stale would again force duplicate reprocessing of a *live* cycle's events. `event_buffer.py:593-604`, RCA §"Not strictly required". |

### 2. Data safety on restart

| # | Sev | Finding |
|---|-----|---------|
| D1 | HIGH | **The graceful-cleanup path is gated behind an unbounded uvicorn wait.** `uvicorn.Config` never sets `timeout_graceful_shutdown` (default `None` ⇒ `asyncio.wait_for(_wait_tasks_to_complete(), timeout=None)` never times out), and that await happens *inside* `server.serve()` — before `run_server()`'s `finally`, `_shutdown_with_watchdog`, and the 75s force-exit timer are ever armed. Worse, fm monkeypatches away uvicorn's signal handlers and its replacement only sets `should_exit`, never `force_exit`, removing uvicorn's own second-signal escape hatch. A stuck in-flight request ⇒ systemd `TimeoutStopSec=90` SIGKILLs the cgroup and **none** of the ordered shielded cleanup (interceptor drain, event-queue flush, harness cancel+restore, memory close, journal/ledger close) runs. `main.py:980-987`, `main.py:1632-1699`, uvicorn `server.py:271-319`. |
| D2 | HIGH | **TaskInterceptor reconciliation events are in-memory until drained.** `event_queue.enqueue` is fire-and-forget into an `asyncio.Queue` (`event_queue.py:128,229-252`); durability only comes from the background drainer / bounded `close()` flush — which is behind D1's gate. Hard kill ⇒ silent event loss. Additionally, the EventQueue's dead-letter **JSONL has no replay path at all**: `replay_dead_letters` (`tools.py:2504-2525`) only replays the *separate* Graphiti `durable_queue`; recon dead letters can only be read, never re-fed. |
| D3 | HIGH | **`add_memory` dual-store write is non-atomic with no repair path.** Graphiti half is durably enqueued (SQLite, synchronous commit) while the Mem0 half runs synchronously in-request, journaled only *after* the awaited call returns (`memory_service.py:2085-2146`, `:831-866`). Kill mid-Mem0-call ⇒ orphaned Graphiti write, zero trace ("attempt started" is never recorded), and `_recover_in_flight` resumes the Graphiti half unconditionally on restart. |
| D4 | MED | **`memory_service.close()` exhausted its 5s budget on both observed restarts** (2-for-2 in journal since 07-10) — the close path may never complete cleanly, so final flushes ride on WAL/pragma durability rather than orderly close. |
| D5 | MED | **No idempotency key on the MCP write path.** Client retry loops re-send on ambiguous timeout/connect failures; `write_ops` ids are server-generated uuid4s (`task_interceptor.py:528`), so a retried `update_task`/`add_dependency` can double-apply. (`submit_task` alone has the R4 escalation-idempotency short-circuit.) |
| D6 | MED | **Manifest sidecar stamping is a torn-write hazard:** `stamp_capability_manifests` plain `write_text`s a git-tracked YAML (`manifest_stamping.py:246-250`), violating the codebase's own temp+`os.replace` convention (`curator_escalator.py:182-224`). |
| D7 | MED | **Judge review is fire-and-forget:** bare `asyncio.create_task(self._run_judge(run_id))` (`harness.py:1952,2911`), untracked by `run_loop`'s shutdown finally (only `self._project_tasks`), no `CancelledError` handling — a restart between cycle-end and judge-end silently drops the verdict and any halt decision, no re-run marker. |
| D8 | — | **History rhymes:** the two worst past incidents are both restart-adjacent — the on-loop watchdog heartbeat SIGABRT crash-loop (task 1731, fixed: OS-thread ping + `WatchdogSec=120`) and the stale-run reaper's unscoped project-lock delete stealing a live run's lock (2026-05-28 RCA, fixed: instance_id-scoped release). Both fixed and holding, but they mark this lock/instance bookkeeping as historically fragile. |

### 3. Client disruption

| # | Sev | Finding |
|---|-----|---------|
| C1 | HIGH | **A restart is a whole-fleet event.** All 7 running orchestrator units point at the single shared instance on 127.0.0.1:8002 (all carry the identical `wait-for-port.py --timeout 280` ExecStartPre). There is no per-project isolation and no fleet quiesce step in any fm restart path. |
| C2 | HIGH | **Client retry budgets ≪ documented restart window.** `mcp_lifecycle.py` `_MCP_MAX_RETRIES=3` / backoff ≈7s total; scheduler transient retries ≈4.5s; deterministic-runner writeback ≈62s — vs. the unit's own `TimeoutStopSec=90` + `TimeoutStartSec=300` ("init routinely runs 30-60s"). Observed restarts were 14–15s (inside almost nothing but the 62s budget), worst case is minutes. A *successful* restart converts into client-visible hard failures. |
| C3 | HIGH | **`get_tasks()` failure is indistinguishable from "zero tasks" and defeats the anti-destructive-eviction guard.** `scheduler.py:2004-2027` catches everything → `[]`; `acquire_next` then drains park-eviction requests with empty maps (`scheduler.py:5964-5973`), and `_owner_is_live_dispatchable` returns False for unknown ids (`:4597-4619`) — so during exactly an fm hiccup, queued evictions force-evict live owners. Fail-dangerous (the guard was built against the df-1865 destructive-eviction starvation). |
| C4 | MED | **Retryable-error set misses what a restart actually emits.** `_RETRYABLE_STATUS={502,503,504}` but fm's `_ASGIExceptionShield` returns bare **500** for exceptions escaping mid-shutdown (`main.py:329-354`) ⇒ not retried, hard failure. Session-reset only fires on the enumerated set; currently masked solely by `stateless_http: true` in config. |
| C5 | MED | **The one code path that *anticipates* an fm self-restart (task 2066 deploy writeback) is under-budgeted for it** — 62s vs. worst-case window; exhaustion files a durable `infra_issue` L2 and BLOCKS a task whose deploy actually succeeded. `deterministic_runner.py:278-295, 1117-1153`. |
| C6 | LOW | Dashboard / orchestrator / interactive sessions each run an independent, divergently-behaved client; the recon-watcher skill's documented response to 8103-down is "tell the human" — nothing auto-recovers. |

### 4. Suitability for automatic action

| # | Sev | Finding |
|---|-----|---------|
| A1 | HIGH | **No liveness watchdog exists for fused-memory.** `orchestrator-watchdog.py` watches only `orchestrator-*` units and paths; the in-process systemd watchdog thread pings `WATCHDOG=1` unconditionally from an OS thread — by design (task 1731), but it means a wedged event loop stays "healthy" forever. Nothing port-probes 8002 or `/health`. |
| A2 | MED | **No staleness signal → no automatic redeploy.** Merged fm changes sit undeployed until a human/PRD files a deterministic deploy task. (Compare orchestrator staleness backstop + shared 8h deploy clock.) |
| A3 | MED | **No config hot-reload** — every knob change is a full restart; a prior partial dynamic-reread was deliberately removed (task 1164, `ticket_janitor.py:100-134`), while `schema.py` still says "hot-reloadable" in places. |
| A4 | LOW | **Unit-parity drift gate checks only 2 directives** (`MEM0_TELEMETRY`, `WatchdogSec`); `Restart=`, `RestartSec=`, `TimeoutStartSec/StopSec`, ExecStartPre ordering can silently drift template↔installed. |
| A5 | LOW | **Journal volume (≈9M lines / ~2 days, 2–3 log lines per MCP request) breaks naive post-restart verification** — full-unit journalctl scans time out at 280s; automation must use narrow `--since` / field-scoped queries. |
| A6 | — | Positive: restarts in the observed window were infrequent, deliberate, attributable, and clean (14–15s). The trigger mechanism itself is not unstable today. |

---

## Recommendations

### Phase 0 — small, high-leverage fixes (each ≤ a simple task)

1. **Bound uvicorn's shutdown wait** (D1): set `timeout_graceful_shutdown` (~10–15s)
   on `uvicorn.Config` so `_graceful_shutdown` is always reached inside the
   `TimeoutStopSec=90` budget; optionally make a second SIGTERM set `force_exit`.
   This single line converts most "hard-kill" data-safety hazards (D2 in-memory
   loss, D4 unflushed close) back into the engineered shielded-cleanup path.
2. **Fix the drain hang at its root** (E3): check `self._draining` inside
   `_project_loop` between cycles (don't start a new cycle while draining; exit
   the loop when idle). This un-breaks `--drain` (task 2090) cheaply.
3. **Make the restart script cycle-aware** (E4/E5): expose "full cycle in
   flight?" (project, run_id, stage, started_at) via `/health` or a tiny MCP
   tool, and have `restart-fused-memory.sh` poll it with a cycle-scale budget
   (default ~35 min, cap configurable) before `systemctl restart` — i.e. restart
   *between* cycles, mirroring `restart-all-orchestrators.sh --drain` semantics
   (defer-if-busy, force after cap). With cycles ≤30 min and restarts rare, this
   alone recovers nearly all the wasted recon spend.
4. **Fail-safe `get_tasks()`** (C3): distinguish RPC failure from empty (return
   `None`/raise), and skip `_drain_park_eviction_requests` on failure ticks.
5. **Retry alignment** (C2/C4/C5): add 500 to `_RETRYABLE_STATUS` (or better,
   have the ASGI shield return 503 once `should_exit` is set), and size one
   shared "fm-restart-window" retry budget (~120s with decorrelated backoff)
   used by scheduler transients and the task-2066 writeback.
6. **Atomic sidecar writes** (D6): temp + `os.replace` in `manifest_stamping.py`.
7. **Track judge tasks** (D7): keep `_run_judge` tasks in a tracked set that the
   shutdown finally cancels/awaits, and persist a "judge pending for run X"
   marker so a restart re-runs it (it's read-only over the run record).

### Phase 1 — durability + detection

8. **Journal recon events at enqueue** (D2): synchronous SQLite append (same
   durability pragmas as `write_journal`) before/at `enqueue`, making the
   in-memory queue a read cache rather than the source of truth; add a real
   replay path for the EventQueue dead-letter JSONL (mirror
   `replay_dead_letters`).
9. **Intent records for dual-store writes** (D3): journal "mem0 write intended
   (op_id)" *before* the awaited call; on startup, reconcile intents with no
   completion record (re-issue or dead-letter). Alternatively route the Mem0
   half through the durable_queue like the Graphiti half.
10. **Instance-aware stale-run recovery** (E6): on startup, immediately reap
    runs whose `instance_id` belongs to this unit's dead predecessor (the
    restarting process *knows* the old instance died) instead of waiting out
    the 1800s age cutoff; keep the age cutoff as the cross-instance backstop.
    Also scope `restore_drained` by run/claim id (E7).
11. **Client idempotency keys** (D5): accept a client-supplied `op_id` on
    mutating task tools, dedup in `write_journal` (unique index), and have
    `McpSession._raw_call` send one per logical call so ambiguous-timeout
    retries are safe.

### Phase 2 — automation substrate

12. **Watch fused-memory** (A1/A2): add `fused-memory.service` (port 8002 +
    `/health`) to `orchestrator-watchdog.py`'s liveness pass, and add
    `fused-memory/src/` to a staleness pass whose deploy action is the
    cycle-aware restart from rec. 3 — gated by its own deploy clock. That makes
    "merged ⇒ deployed" automatic and safe, closing the loop the orchestrator
    fleet already has. Extend the parity gate (A4) to the restart-relevant
    directives so template tuning actually propagates.
13. **Session-resume for interrupted recon stage agents** (E2) — *revised
    2026-07-17 after mapping the orchestrator's crash-recovery resume seam;
    supersedes the original per-stage-checkpoint sketch.* The resume machinery
    lives in `shared/src/shared/cli_invoke.py` — the module recon already
    invokes through — so this is persistence + recovery-path work, not new
    dispatch code. Design (decisions ratified by Leo 2026-07-17):
    - **Mint-before-spawn:** mint a uuid4 per stage attempt, pass
      `--session-id`, persist `session_id` + stage cursor + attempt counter in
      the SQLite run journal (`runs` table) *before* awaiting the subprocess
      (orchestrator pattern, `workflow.py:8334-8365`, but journal-backed
      instead of a sidecar file).
    - **Dedicated transcript store:** pass a per-run `config_dir` under
      fused-memory's data dir (today stage runs use the ambient default —
      cwd-dependent `--resume` lookup, the steward's dominant resume-failure
      mode). Bonus: activates `_run_subprocess`'s transcript-liveness timeout
      logic, currently inert for recon stages. GC the dir with the run.
    - **Real resume prompt:** use `resume_delivers_prompt=True` with a
      purpose-written recovery prompt ("interrupted by a server restart; MCP
      connections were reset; check what you already wrote via
      write_journal/causation_id and continue") and keep the full stage prompt
      as `original_prompt` so fallback-to-fresh degrades to a clean stage
      re-run. Do NOT copy the orchestrator's `'continue'` placeholder
      (`CRASH_RECOVERY_RESUME_PROMPT`, known design-concern f1bad303/task 1462
      — being handled orchestrator-side separately).
    - **ReconReportState → SQLite** (decision: SQLite persistence, not
      prompt-mitigation): persist the `recon-report` findings state keyed by
      (run_id, stage) so a restart doesn't lose partially-filed findings the
      resumed transcript believes were already reported.
    - **Run-state machine:** new `interrupted` run status; on
      shutdown-cancellation persist session/stage state and do **not**
      `restore_drained` (events must stay drained so the resumed run's work
      isn't double-processed); on startup an instance-scoped adopt-and-resume
      pass (own dead predecessor only — also closes the ba3d8b75
      duplicate-spawn class) runs before the stale-run reaper; the reaper
      remains the backstop converting unresumable runs to today's
      failed+restore path. Completed stages are skipped via their
      already-persisted `stage_reports` entries (`harness.py:1913-1917`,
      `:2037`).
    - **Guard rails:** one resume per stage, small per-run cap, freshness
      window (~1h), then fall back to failed+restore; fallback-to-fresh and
      the zero-output-wedge session-clear come free from
      `invoke_with_cap_retry`. A `resume_after_restart` config knob opts a
      deploy out of resume when the deploy changes the recon prompts/tooling
      themselves (resumed sessions finish under the old system prompt by
      construction — `--resume` skips it).
    - Note: kill-then-resume is the right shape (not detached-child
      survival) because stage agents connect over HTTP to the restarting
      server itself (fused-memory :8002, recon-report :8003) — a resumed CLI
      opens fresh connections; a detached child's tool calls would all error
      mid-run.
14. **Green-tier hot-reload** (A3): a narrow `reload_config` MCP tool for
    genuinely reload-safe knobs (thresholds, budgets, timeouts), mirroring the
    escalation server's applied/restart_required disposition report — this
    removes a whole class of restarts outright.

### Sequencing note

Recs 1–3 are the core: (1) makes every restart *safe*, (2)+(3) make every
restart *cheap*. Rec 12 then makes restarts *automatic*. Most other items are
independent hardening and can ride as `complexity=simple` tasks or fold into a
single "fm restart safety" PRD with recs 8–11.

### Decisions & spin-outs (2026-07-17, Leo)

- Phases 0–2 ratified for filing as fix tasks (this batch), including the
  revised rec 13 above. Decisions: ReconReportState persists to **SQLite**;
  run-state machine per the `interrupted`/adopt-and-resume sketch.
- **Spun out to a separate interactive session** (not in this batch): three
  orchestrator-side questions surfaced while mapping the resume seam —
  (a) are task-agent session transcripts destroyed with the worktree on task
  completion (per-task `CLAUDE_CONFIG_DIR` lives inside the worktree)?
  (b) can the legibility/confusion-reduction infra mine them, or is the
  fleet's own agent activity invisible to it? (c) is the crash-recovery
  `'continue'` resume prompt (f1bad303 / task 1462) a bug to fix, and how?
  Brief: `~/.claude/spawn-briefs/orch-session-transcripts-continue-prompt-2026-07-17.md`.

## Decomposition plan (2026-07-17)

19 tasks, phases 0–2. Leaves declare user-observable signals (G2); π/ρ are
intermediates unlocked into integration-gate σ. Deps: β←α, δ←γ, ο←δ+ξ,
σ←π+ρ+μ. No cross-PRD seams (orchestrator spin-outs excluded above, owned by
a separate session). G5: B+H — the survey findings + Decisions section are
the contract; σ is the resume cluster's integration gate. G7: walked, no
waivers (structured-signal, corroboration, storm-escape, single-chokepoint
requirements folded into task details below).

| L | Task | Deps | Observable signal |
|---|---|---|---|
| α | Bound uvicorn graceful-shutdown wait; second-signal force-exit | — | `systemctl stop` with a stuck in-flight request: journal shows ordered `_graceful_shutdown` steps before exit, no SIGKILL |
| β | Fix `memory_service.close()` budget exhaustion | α | restart journal shows close completing, no "timed out after 5.0s" |
| γ | Drain converges: honor `_draining` in `_project_loop`; `drain_ack` log token | — | SIGUSR1 mid-cycle → "Harness fully drained" after current cycle; `drain_ack` per exited loop |
| δ | Structured `recon_busy` health field + defer-if-busy restart script | γ | restart issued mid-cycle defers with structured wait status; proceeds after cycle or 35-min cap |
| ε | Scheduler distinguishes fm-failure from empty tasks; eviction guard fail-safe + streak escalation | — | simulated fm outage: `park_eviction_deferred_fm_unavailable` event, no eviction on empty maps |
| ζ | fm returns 503+Retry-After during shutdown (not bare 500) | — | request in shutdown window receives 503 |
| η | Shared restart-window retry budget across orchestrator fm-clients | — | fm restart mid-operation: calls succeed after retry, no spurious infra_issue |
| θ | Atomic temp+rename sidecar stamping | — | injected mid-write failure leaves prior sidecar intact |
| ι | Track judge tasks; judge-pending marker re-runs verdicts | — | restart between cycle-end and judge-end → verdict present after startup |
| κ | Durable-at-enqueue recon events + event-queue dead-letter replay | — | kill -9 pre-drain: event survives restart; replay returns count, events re-enter buffer |
| λ | Write-ahead intent for dual-store add_memory | — | kill mid-Mem0-write: intent reconciled on startup (re-issued or dead-lettered), no silent orphan |
| μ | Instance-aware stale-run recovery; run-scoped `restore_drained` | — | kill 2 min into run → recovered first reaper tick after restart, not ~30 min |
| ν | Idempotency keys for mutating task writes | — | duplicate retry with same op_id applies once, returns recorded result |
| ξ | fm liveness in orchestrator-watchdog + unit-parity gate extension | — | `--report` shows fused-memory row; port-down → revive |
| ο | Staleness-driven fm auto-redeploy via δ's gate | δ, ξ | merged fm change → watchdog redeploys through defer-if-busy chokepoint within clock window |
| π | Session capture substrate: mint ids, journal columns, per-run config_dir | — | in-flight run row carries session_id+stage (read tool); unlocks σ |
| ρ | ReconReportState persisted to SQLite (run_id, stage) | — | restart mid-stage: previously filed findings readable after restart; unlocks σ |
| σ | Resume interrupted recon runs (`interrupted` status, adopt-and-resume, guard rails, `resume_after_restart` knob) | π, ρ, μ | restart mid-stage-2 → same run_id resumes at stage 2, stage 1 not re-run, drained events not reprocessed |
| τ | Green-tier config hot-reload tool (applied/restart_required dispositions) | — | edit green knob + call tool → named in `applied`, behavior changes without restart |

## Refuted / folded findings

10 candidate findings were refuted by verifiers — all were duplicates or
overstated framings of confirmed findings above (e.g. "drain is partial" folded
into E3/E4; "no busy signal exists" narrowed into rec. 3's gap; "health-wait
undersized" superseded by C2's broader budget analysis). No refutation
contradicted a confirmed finding.
