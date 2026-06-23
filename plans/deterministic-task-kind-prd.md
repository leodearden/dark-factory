# PRD: First-class `deterministic` task kind (auto-deploy + milestone gates)

**Status:** active — ready to decompose
**Date:** 2026-06-23
**Type:** infrastructure extension (gives the existing-but-inert `task_kind` metadata its first consumer)
**Approach:** B + H (contract + two-way boundary tests) — blast radius ≥ 3 (task model + scheduler/harness + escalation + the reify deploy script), load-bearing seams.

---

## 1. Goal

Make the two non-code task patterns Leo queues regularly — **auto-deploy capstones** (restart a long-lived service so a just-merged fix takes effect) and **milestone human-review/design gates** — first-class, instead of smuggling them through the TDD pipeline.

**User-observable surface (what changes for Leo):**

- A **gate** task queued with deps on a milestone's tasks, when those land, produces **exactly one born-at-L2 escalation in his queue with no agent ever spawned** — zero LLM turns, no L0→L1→L2 climb. He resolves it; the line proceeds (`resume`→`done`→dependents dispatch) or he drives a no-go.
- A **deploy** task gated on a fix, when the fix lands, **restarts the target service and goes `done`** — for the common cross-unit case (e.g. a `dark_factory` task restarting `orchestrator-reify`) `done` means *deployed-and-verified* (fresh PID confirmed); for a self-restart it means *scheduled*, with an `OnFailure` escalation if the detached restart doesn't land.
- Neither task manufactures a markdown diff, spins an architect, takes a worktree, or risks the degenerate-branch / no-diff strand paths.

## 2. Background

Today (verified in the study that motivated this PRD):

- The orchestrator has **one task shape** — `PLAN→EXECUTE→VERIFY→REVIEW→MERGE→DONE` (`orchestrator/src/orchestrator/workflow.py:291-301`) — whose definition of success is *a diff lands on main*. There is **no `task_kind` consumer anywhere** (grep-confirmed); `task_kind` appears in metadata (task 1680) and is read by nothing.
- **Auto-deploy** is bifurcated: an automatic post-merge hook `StaleServiceRestartCoordinator` (`orchestrator/src/orchestrator/service_restart.py`) covers leaf services (fused-memory, dashboard), but the **orchestrator process itself is deliberately excluded** (highest blast radius / self-kill). So orchestrator restarts are done as hand-authored **DEPLOY capstone tasks** (exemplars 1738, 1793, 1800, 1858, 1863, 1866, 1875) that disguise themselves as code tasks: their "diff" is a `plans/<id>-deploy-*.md` record (so they don't strand as degenerate branches), and the real restart fires out-of-band via `systemd-run` after the agent exits (`reify/scripts/orchestrator-redeploy-restart.sh`). Each capstone is near-verbatim re-typed boilerplate (deps + authorization + steps + self-kill note).
- **Milestone gates** have no pre-queueable form. A "decide-and-implement-if-needed" task that auto-acts works (task 1680, merged), but a standalone no-code go/no-go node **churns** (the architect has nothing to plan, re-blocks) and gets cancelled (task 1703, the "A′ gate"). Reaching a human is only possible *reactively* at runtime via the escalation ladder, and an ordinary agent's `escalate_blocker(critical)` is **downgraded to blocking L1** (`escalation/src/escalation/server.py:246-267`) — so the climb is L0 steward → L1 auto-watcher → L2, costing redundant LLM turns before it reaches the one place Leo actually reads.

The fix is one primitive: a `deterministic` task kind that **skips the LLM workflow entirely** (no worktree, no branch, no agent, no merge), runs an optional committed action, and escalates **born-at-L2 directly** (the scheduler/harness files as a sentinel, which is already exempt from the downgrade). Both patterns become field combinations on this one kind.

## 3. Sketch of approach

A `deterministic` task is dispatched by the **existing** scheduler eligibility check (`_eligible_for_dispatch`/`_deps_satisfied`, `scheduler.py:2197-2266,2372-2441`) — same "when ready" gate as every task — but at the dispatch point where the harness currently builds a `TaskWorkflow`, it routes to a new **`DeterministicRunner`** instead.

The runner is a small state machine over three fields:

| `before_done` | `always_escalates` | Behaviour | Pattern |
|---|---|---|---|
| present | false | run action once; escalate **only** on non-zero/verify-fail; else `done` | **auto-deploy** |
| present | true | run action once; then escalate (born-at-L2) regardless; `done` only after `resume` | act-then-ask |
| none | true | escalate (born-at-L2) immediately; `done` only after `resume` | **pure gate** |
| none | false | **rejected at `submit_task`** (ill-formed no-op) | — |

Escalations are **born-at-L2** (`severity ∈ {critical,urgent}`, filed with a sentinel `agent_role='orchestrator-*'` so the server keeps `level=2` rather than downgrading — `server.py:215-276`), submitted in-process via the harness's existing `EscalationQueue.submit(Escalation(...))` path (already used ~15× in `harness.py`). On escalate the task goes `blocked`; the open L2 is a **quiescence guard** so a parked gate never churns (`harness.py:2289,2344-2352`).

The human's resolution flows back through the **existing** `resolve_issue`/`RESOLVE_ACTIONS`→`harness._on_escalation_resolved` machinery (`server.py:75,450-538`; `harness.py:5218-5345`); the runner adds the deterministic-task interpretation of those actions (below).

**Blocking vs detached is *determined*, not a knob:** if `before_done.target_unit` is the unit running this orchestrator (self-kill risk) → detached `systemd-run` with `--on-failure` (done = scheduled); otherwise → blocking subprocess + fresh-PID verify (done = deployed-and-verified). The reify exec-restart script is used as a **dumb payload** (`--exec-restart`); all new wiring stays in dark-factory, including a tiny script-callable **`escalation submit` CLI** for the `OnFailure` path (the queue is file-backed, so a detached unit can submit without the MCP server).

## 4. Resolved design decisions

1. **`task_kind` is a first-class `submit_task` parameter** (`'normal'` default | `'deterministic'`), persisted to metadata; the action fields are structured metadata. No DB migration. *(Decision: "Promote task_kind to a submit_task param".)*
2. **`before_done` is a committed reference, never inline bash:** `{script: <repo-relative path>, args: [str], env: {str:str}, cwd: str, timeout_secs: int, target_unit: str|None}`. Auditable, version-controlled, gives the action its execution context.
3. **Validity invariant, enforced at `submit_task` (block at creation):** for `task_kind='deterministic'`, `before_done=None ⟹ always_escalates=true`; reject otherwise. Also reject `before_done` set on a `normal` task (fold-in onto normal tasks is out of scope, §10).
4. **Once-only action.** The runner stamps `metadata.before_done_ran_at` before running, and never re-runs a stamped action — so a `resume` that drives the task to `done`, or a reaper re-pass, cannot restart a service twice.
5. **Loop-free gate.** The runner stamps `metadata.gate_escalated_at` when it files the gate L2. On re-dispatch (after `resume`), `gate_escalated_at` set + no open escalation ⟹ proceed to `done` (do **not** re-escalate). This is the load-bearing idempotency.
6. **All deterministic-task escalations are born-at-L2.** First cut: one level, no climb. (Revisit only if L2 noise appears.)
7. **The gate is a normal task with normal statuses**; `→blocked` on escalate. The **handler is accountable** for post-decision action. `proceed` = `resume`→`done`→dependents flow. `no-go` = handler cancels dependents and/or files new design tasks, **re-depends the gate task on the new tasks**, and resolves `resume` → the gate re-pends and re-gates on the new work (survives; re-fires when they land).
8. **`done` = scheduled with `OnFailure` escalation is accepted for self-restart**; cross-unit restart blocks and verifies fresh PID so `done` = deployed-and-verified.
9. **Runner owns the detached scheduling + `OnFailure` wiring** (no reify edit); ships an `escalation submit` CLI. The reify `orchestrator-redeploy-restart.sh --exec-restart` is the payload.
10. **Deterministic tasks take no module locks and create no branch/worktree**, so they are invisible to the degenerate-branch / strand reapers (`harness.py:1982-2004,2362-2404`) — they cannot trigger a false revert.
11. **Retire the obsolete dark_factory-internal-dep rule** for deploy capstones. It was a workaround for the external-dep gate bug (now fixed by tasks 1854/1855/1799); deterministic deploys/gates use normal deps, **including cross-project `project_id:task_id` deps**.

### Deterministic-task resolution semantics (extends `RESOLVE_ACTIONS`)

| Resolution | Effect on a deterministic task |
|---|---|
| `resume` | `blocked`→`pending`; re-dispatch → runner sees `gate_escalated_at` (and/or `before_done_ran_at`) → drives to `done` without re-running the action or re-escalating. |
| `restart` | `blocked`→`pending`; runner **clears** `before_done_ran_at`/`gate_escalated_at` and re-runs from scratch. |
| `park` | stays `blocked`, L2 kept open (re-ask later) — unchanged from existing `park`. |
| `abandon` | `cancelled` (note: a `cancelled` dep is *terminal* and would satisfy dependents — the no-go path uses `resume`+re-depend, not `abandon`, to stop the line). |
| `close_only` | stays `blocked`, no status change — unchanged. |

## 5. Pre-conditions for activating

- **No novel substrate** beyond what's verified below — G3 is effectively N/A except the one new deliverable (the `escalation submit` CLI, which is itself a task in this batch, δ).
- Verified substrate this PRD leans on (all confirmed in the study):
  - Programmatic born-at-L2 submit from a sentinel context — `EscalationQueue.submit(Escalation(...))` used throughout `harness.py`; `level=2` stamping + sentinel exemption at `server.py:45-52,215-276`.
  - Dispatch eligibility unchanged — `scheduler.py:2197-2266,2372-2441`.
  - Resolution round-trip — `server.py:75,450-538`; `harness.py:5218-5345`.
  - Open-L2 quiescence guard — `harness.py:2289,2344-2352`.
  - File-backed escalation queue (script-submittable) — `EscalationQueue(queue_dir)`, `harness.py:4293`.
  - Reify exec-restart payload — `reify/scripts/orchestrator-redeploy-restart.sh` (`--exec-restart`).
- **Bootstrap note:** the first deployment of *this* feature cannot use a deterministic deploy task (it doesn't exist yet) — after the batch lands on main, the orchestrator must be restarted once by the old manual/capstone path. Called out in the hand-back.

## 6. Cross-PRD / cross-project relationship (G4)

| Other artifact | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `reify/scripts/orchestrator-redeploy-restart.sh` | dark_factory **consumes** | `--exec-restart` payload (no behaviour change) | reify (unchanged) | wired (decision 9 keeps all new wiring dark_factory-side) |
| `escalation` package | this PRD **produces** | new `escalation submit` CLI entrypoint (file-backed queue) | this PRD (task δ) | queued |
| `fused-memory` `submit_task` | this PRD **produces** | `task_kind` param + structured `before_done`/`always_escalates` + validation | this PRD (task α) | queued |
| `service_restart.py` coordinator | **orthogonal** | unchanged; continues to own watched-path leaf services (fused-memory/dashboard) | unchanged | N/A |

No reciprocal "the other owns it" ambiguity. The deterministic kind does **not** replace the post-merge coordinator — it owns the bespoke orchestrator restarts the coordinator can't safely do.

## 7. Decomposition plan

Greek labels; task IDs assigned at decompose. Boundary-test scenarios (§12) are referenced as B*n*.

**Phase 1 — Foundation (task model)**

- **α — `task_kind` param + structured action fields + `submit_task` validation.**
  Modules: `fused-memory/src/fused_memory/server/tools.py`, the task interceptor/backend write path.
  Signal (observable, incl. rejection): `submit_task(task_kind='deterministic', always_escalates=False)` with no `before_done` returns a **validation error naming the invariant** (B10); a valid deterministic task round-trips — `get_task` shows `task_kind` + `metadata.before_done`/`always_escalates`.
  Unlocks: β, γ, δ.

**Phase 2 — Vertical slice (pure gate; the integration gate)**

- **β — Deterministic dispatch + `DeterministicRunner` + pure-gate end-to-end.** Depends α.
  Modules: `orchestrator/src/orchestrator/scheduler.py` (route by `task_kind` at dispatch), new `orchestrator/src/orchestrator/deterministic_runner.py`, `harness.py` (resume interpretation).
  Signal (user-observable leaf): a queued gate task (`before_done=None, always_escalates=true`) whose deps are satisfied produces **exactly one pending born-at-L2** (visible via `get_pending_escalations(level=2)`) with **no agent invocation, no worktree, no branch** in the run log; it stays quiescent across ≥2 sweeps (B3); on `resolve_issue(resume)` the task goes `done` **without re-escalating** (B4) and a dependent task dispatches.
  This is the B+H integration-gate leaf for the gate path; its signal is the B1–B5 + B11–B12 slice of the boundary table.

**Phase 3 — Cross-unit blocking deploy**

- **γ — `before_done` blocking execution + fresh-PID verify + escalate-on-fail.** Depends β.
  Modules: `deterministic_runner.py`.
  Signal (leaf): a deterministic deploy task gated on an already-`done` fix, when dispatched, runs the committed script **once** (B-stamp), restarts the cross-unit target, **verifies** a fresh `MainPID` + `ActiveEnterTimestamp` after a pre-captured baseline, and sets `done` with `done_provenance.kind='deterministic-deploy'` carrying the new PID/timestamp (B6); an injected failure (bad unit / stale PID) lands a **born-at-L2** and leaves the task `blocked` with `before_done_ran_at` stamped (not re-run) (B7).

**Phase 4 — Self-restart (detached + OnFailure)**

- **δ — `escalation submit` CLI.** Depends α.
  Modules: `escalation/src/escalation/` (new submit entrypoint + pyproject script), constructs an `Escalation` and `submit()`s to the queue_dir.
  Signal (observable): `python -m escalation submit --task <id> --severity critical --category … --summary …` writes a pending **L2** visible via `get_pending_escalations`. Unlocks ε.
- **ε — Runner detached self-restart path.** Depends γ, δ.
  Modules: `deterministic_runner.py`.
  Signal (leaf): a deterministic deploy task whose `target_unit` **is** the dispatching orchestrator's own unit schedules a detached `systemd-run --user --on-active=… --on-failure=<δ unit> <exec-restart payload>`, stamps `before_done_ran_at`, and sets `done` with `done_provenance.kind='deterministic-deploy-scheduled'` — **the dispatching orchestrator is not killed** (B8); a forced fire-time failure triggers `OnFailure`→δ→a born-at-L2 (B9).

**Phase 5 — Integration gate + companion corrections**

- **θ — Boundary-test suite B1–B12 end-to-end.** Depends β, γ, ε. *(C-as-integration-gate.)*
  Modules: `orchestrator/tests/test_deterministic_task.py`.
  Signal (leaf): all of §12 pass, including the cross-phase scenarios — once-only across a simulated orchestrator restart (B11), no-go re-pend with new deps (B5), no-lock/no-strand invisibility (B12).
- **ζ — Docs + convention correction.** Depends θ.
  Modules: `CLAUDE.md` (and the deploy-task guidance prose).
  Signal (leaf): `CLAUDE.md` documents the deterministic task kind (param, fields, born-at-L2, blocking-vs-detached rule) **and removes** the obsolete "file deploy capstones in dark_factory with an internal dep" rule, replacing it with "use normal incl. cross-project deps." (Sanctioned companion correction-task per author-mode Stage 7.)

DAG: α → {β, δ}; β → γ; {γ, δ} → ε; {β, γ, ε} → θ → ζ.

## 8. Out of scope

- **`before_done` on `normal` (code) tasks** (the true "fold the deploy onto the fix task" form / a `deploy_on_merge` field). The deterministic deploy task downstream of the fix already collapses authoring to ~4 fields and reuses the gate machinery; adding a side-effect to the merge-completion path is a separate, higher-risk change. Revisit only if even the small deterministic task proves friction at volume.
- **Changing `service_restart.py`** — it stays as-is for watched-path leaf services.
- **A `/deploy-gate` authoring skill / template** to generate these tasks. Worth doing once the primitive exists; not part of this PRD.
- **Multi-host / remote-unit restarts** beyond the local cross-unit case (reify's orchestrator runs locally against the dark-factory checkout, so the local case covers the exemplars).
- **Making a `cancelled` dependency not satisfy dependents** (the alternative no-go mechanism). First cut uses the handler-accountable `resume`+re-depend path per decision 7; the scheduler change is a possible future hardening.

## 9. Open questions (tactical — decide at impl)

1. **Escalation `category` for gates.** Reuse `design_concern` vs add a `milestone_gate` category (`server.py:79-96`). Suggested: add `milestone_gate` for clean dashboard filtering. Decide in β.
2. **Gate L2 payload shape.** The runner should include the task title + description + the landed dep IDs/links; optionally read a `metadata.gate_options` list (mirroring `promote_to_l2`'s `options`) into the L2 `options`. Suggested: support `gate_options` if present, else summary-only. Decide in β.
3. **`on-active` delay + transient-unit naming** for the detached path (mirror the existing 60s / `orch-redeploy-restart`). Decide in ε.
4. **Where the runner learns "its own unit"** for the self-kill determination (env `ORCH_UNIT` vs systemd `$INVOCATION_ID` lookup vs config). Decide in ε.

## 10. (B+H) Contract section — see §11; Boundary-test sketch — see §12.

---

## 11. Contract

### 11.1 `submit_task` surface delta

```
submit_task(..., task_kind: str = 'normal', metadata: {...})
  task_kind ∈ {'normal','deterministic'}            # first-class param, persisted to metadata.task_kind
  metadata.before_done: BeforeDone | None
  metadata.always_escalates: bool = False
  # runner-written (not author-supplied): before_done_ran_at, gate_escalated_at, done_provenance

BeforeDone = {
  script: str,            # repo-relative path under project_root; MUST exist & be executable
  args: list[str] = [],
  env: dict[str,str] = {},
  cwd: str = project_root,
  timeout_secs: int,      # required; runner kills + escalates on timeout
  target_unit: str | None # systemd unit the action restarts; None ⇒ treated as cross-unit/no-self-kill
}
```

**Validation (block at `submit_task`):**
- `task_kind='deterministic'` ∧ `before_done=None` ∧ `always_escalates=false` → **reject** ("ill-formed no-op: a deterministic task must run an action or always escalate").
- `task_kind='normal'` ∧ `before_done≠None` → **reject** ("before_done is only valid on deterministic tasks").
- `before_done.script` not present/executable under `project_root` → **reject**.

### 11.2 `DeterministicRunner.run(task)` — invariants

```
# dispatched only when _deps_satisfied (unchanged). No worktree, no branch, no module lock.
if before_done and not metadata.before_done_ran_at:
    stamp before_done_ran_at = now
    self_target = (before_done.target_unit == this_orchestrator_unit)
    if self_target:
        schedule detached: systemd-run --user --on-active=D --on-failure=<escalation-submit unit> <script --exec-restart>
        if schedule fails: born_at_L2('infra_issue', …); set_status(blocked); return
        # done = scheduled
    else:
        baseline = systemctl show target_unit (MainPID, ActiveEnterTimestamp)
        rc, out = run(script, args, env, cwd, timeout_secs)         # blocking, async subprocess
        if rc != 0: born_at_L2('infra_issue', detail=tail(out)); set_status(blocked); return
        if not verified_fresh(target_unit, baseline): born_at_L2('infra_issue', …); set_status(blocked); return
        # done = deployed-and-verified; done_provenance.kind='deterministic-deploy'
if always_escalates and not metadata.gate_escalated_at:
    stamp gate_escalated_at = now
    born_at_L2(category, summary=title, detail=description+deps[+gate_options]); set_status(blocked); return
set_status(done, done_provenance)        # reached only when action done (if any) AND gate opened/absent
```

**Invariants:**
- I1 (once-only action): `before_done_ran_at` set ⟹ action never re-runs.
- I2 (loop-free gate): `gate_escalated_at` set ∧ no open escalation ⟹ next dispatch goes to `done`, never re-escalates.
- I3 (born-at-L2): all escalations filed with `severity ∈ {critical,urgent}` and a sentinel `agent_role` (`orchestrator-deterministic`) so `level=2` is retained.
- I4 (no diff subsystem): no worktree/branch/lock created ⟹ degenerate-branch & strand reapers never act on a deterministic task.
- I5 (quiescence): a `blocked` deterministic task with an open L2 is not re-dispatched (existing guard).

## 12. Boundary-test sketch (faces both task-model/scheduler and escalation sides)

| # | Scenario | Preconditions | Postconditions (asserted) |
|---|---|---|---|
| B1 | Gate, deps unsatisfied | gate task, one dep pending | not dispatched; **no** escalation filed |
| B2 | Gate, deps satisfied | gate task, deps done | exactly **one** pending L2 (`level=2`); task `blocked`; **no** worktree/branch/agent/lock created |
| B3 | Gate quiescence | B2 state, run ≥2 sweeps | no re-dispatch, no second escalation (no churn) |
| B4 | Proceed | B2 state, `resolve_issue(resume)` | task `done`; action **not** run; escalation **not** re-filed; a dependent dispatches |
| B5 | No-go re-pend | B2 state; handler cancels dependents, files new design task, re-depends gate on it, `resume` | gate → `pending`, re-gated on the new task; no immediate re-escalate; re-fires when new task lands |
| B6 | Cross-unit deploy success | deterministic task, `target_unit`≠self, fix done | script runs once; fresh PID verified; `done` w/ `done_provenance.kind='deterministic-deploy'` + non-sentinel PID |
| B7 | Cross-unit deploy failure | as B6 but script rc≠0 or stale PID | **born-at-L2**; task `blocked`; `before_done_ran_at` stamped; not re-run on a reaper pass |
| B8 | Self-restart scheduled | deterministic task, `target_unit`==self | transient unit scheduled w/ `--on-failure`; task `done` (`…-scheduled`); **dispatching orchestrator not killed** |
| B9 | Self-restart fire-time failure | B8, forced exec failure | `OnFailure`→`escalation submit`→ a pending **L2** |
| B10 | Validation | `submit_task` of the no-op corner / before_done-on-normal | **rejected** with the invariant message (rejection mechanism fires) |
| B11 | Restart-window replay | blocked gate, simulate orchestrator restart | stamps persist; on rehydrate no re-escalate, no action re-run |
| B12 | No-lock / no-strand | deterministic task in flight | takes no module lock; degenerate-branch & strand reapers do **not** revert/escalate it |
