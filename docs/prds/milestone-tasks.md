# PRD: Dated / delayed milestone tasks

**Milestone:** orchestrator scheduling infra
**Status:** active — greenfield feature on existing substrate
**Approach:** B + H (contract + two-way boundary tests) — touches the load-bearing scheduler dispatch gate across 3 packages
**Date:** 2026-07-08

---

## 1. Goal

Let an operator file a task that **does nothing until a point in time**, then runs a
check and either completes or escalates to a human. Two time modes:

- **Dated** — fires at an absolute wall-clock instant (`at: 2026-08-01T00:00:00Z`).
- **Delayed** — fires a duration **after its dependencies are satisfied**
  (`after_secs: 604800` = one week *after* the deps land). The timer starts when
  the deps go `done`, not when the task is filed.

Motivating exemplar (fully autonomous, no human unless the check fails):

> "One week after the merge-queue-reliability work lands, check the logs and see
> whether average merge flakiness < 5%. If so, done; else escalate."

A milestone is a **time-based dispatch gate that is orthogonal to `task_kind`**:
it holds *any* task — a normal LLM agent task, a deterministic predicate check,
or a pure human gate — out of dispatch until the trigger fires, then the task
dispatches through its normal path.

## 2. Background — substrate that exists, and the gaps

This feature is a **thin composition PRD** over three existing subsystems, plus
one genuinely new runner mode. The audit that produced this PRD confirmed:

**Reusable substrate (verified on `main`):**

- **`metadata.deferred_watch` + `trigger_met` dispatch gate** — `Scheduler._deferred_watch_gated`
  (`orchestrator/src/orchestrator/scheduler.py:2824`, task 2234) withholds a task
  from **both** dispatch paths (scored + pin) until un-gated. It is **manual-only
  by design**: the scheduler deliberately never evaluates a trigger, because a
  human-readable prose trigger is unjudgeable. *No time evaluation exists.*
- **`task_kind='deterministic'` + `DeterministicRunner`** (`deterministic_runner.py`)
  — runs a committed `before_done` script, files born-at-L2 escalations with the
  sentinel role `orchestrator-deterministic`, and is fully crash-safe (I1
  once-only, quiescence guard, resume-to-done on escalation resolution). This is
  the "act, then done-or-escalate" engine.
- **Typed metadata schema** — `shared/src/shared/task_metadata.py`
  (`task_metadata.enforce=true` is **live** as of 2026-07-08 / commit ee97613a96):
  `TaskMetadata` (`extra='allow'`), the `register_metadata_submodel` extension
  point, `BeforeDone` (`extra='allow'`), and `DoneProvenance.kind` (a **closed**
  `Literal[...]`). Submit-time validation lives in
  `fused-memory/.../middleware/deterministic_task_guard.py`.
- Escalation `category` is an **open `str`** (`escalation/src/escalation/models.py:55`)
  — a new category costs no enum change.

**The four gaps this PRD fills:**

1. **Time-trigger evaluation** — auto-satisfy a dispatch gate on a
   *machine-evaluable* time condition (dated absolute / delayed relative).
2. **deps-satisfied stamping** — for delayed mode, persist the first instant all
   deps (local **and** `external_deps`) are `done`. Confirmed: the scheduler's
   `self._time_source()` is `time.monotonic` (a float that resets across
   restarts) — so the anchor **must** be a persisted **wall-clock**
   `datetime.now(UTC)` ISO string, and dispatch-eligibility must stay a pure
   predicate (stamp in a per-tick sweep, mirroring `_gc_expired_cooldowns`).
3. **Predicate runner mode (new behaviour).** The existing
   `before_done`+`always_escalates=False` "auto" preset is **deploy-shaped**: it
   demands a fresh systemd `MainPID` verify after the script and *fails the
   baseline inspect when `target_unit` is empty* (`deterministic_runner.py:1227`,
   `:1322`). "Run a check script, exit-code decides done-vs-escalate, no unit
   restart" **does not exist today** — it is a new runner branch + a new
   `DoneProvenance.kind`.
4. **submit_task validation + typed schema** for the milestone metadata and the
   predicate `before_done` discriminator.

## 3. Consumer + user-observable surface (G1)

Every mechanism this PRD introduces has a named consumer on an **existing**
surface:

| Mechanism introduced | Consumer |
|---|---|
| `metadata.milestone` time-gate | `Scheduler.acquire_next` dispatch path (holds → releases the task); observable via `get_scheduler_state` (task not in `current_holders` while gated, then dispatched) |
| `milestone_deps_satisfied_at` stamp | The delayed-mode gate predicate; observable via `get_task` metadata |
| `before_done.kind='predicate'` | `DeterministicRunner.run` predicate branch |
| `deterministic-milestone` provenance | `get_task` / `get_statuses` (task shows `done` with that provenance kind) |
| `milestone_check_failed` L2 escalation | `get_pending_escalations` + the **escalation-watcher** (the existing L2 human consumer) |
| `Milestone` typed sub-model | The `submit_task` fused-memory author API (accepts / rejects the spec) |

No orphan mechanisms: the time-gate is consumed by the scheduler; the predicate
mode by the runner; the verdict by task-status and the escalation queue.

## 4. Sketch of approach

**One concept: a milestone is an orthogonal time-based dispatch gate.**

```
                         ┌─────────────────────────────────────────────┐
  submit_task(           │  Scheduler.acquire_next (per tick)           │
    metadata.milestone,  │                                              │
    [task_kind, …]       │  1. _stamp_milestone_deps_satisfied() sweep  │
  )                      │     → delayed tasks: stamp deps_satisfied_at │
     │                   │       the first tick all deps are `done`     │
     ▼                   │  2. _eligible_for_dispatch (pure predicate): │
  δ validates &          │     … _deferred_watch_gated? …               │
  persists the spec      │     … _milestone_time_gated(now_wall)? ──────┼─ withhold until
                         │     … _deps_satisfied? …                     │  time fires
                         └───────────────────┬──────────────────────────┘
                                             │ gate opens → dispatch normally
                    ┌────────────────────────┼───────────────────────────┐
                    ▼                        ▼                           ▼
       task_kind='normal'        task_kind='deterministic'    task_kind='deterministic'
       → LLM agent runs at        before_done.kind='predicate'  before_done=None,
         fire time (existing      → γ: run check;               always_escalates=True
         path, no change)           rc==0 → done                → pure human gate at
                                     (deterministic-milestone)     fire time (existing
                                     rc!=0 → milestone_check_       pure-gate path)
                                     failed L2 + blocked
```

The **time-gate (§2 gaps 1,2,4)** is the core new mechanism and is what makes
*any* task a milestone. The **predicate runner mode (§2 gap 3)** is what makes the
autonomous exemplar work without an agent. A **pure human-gate milestone**
(dated + `always_escalates=True`) falls out for free from composition.

## 5. Resolved design decisions

1. **Milestone = orthogonal time-gate, not a new `task_kind`.** `metadata.milestone`
   is allowed on `normal` **and** `deterministic` tasks. (Per the "both / orthogonal"
   choice.) This is why the exemplar needs only a *dispatch gate* plus the
   existing deterministic machinery — no bespoke milestone runner.

2. **Separate scheduler gate `_milestone_time_gated`, NOT reuse of
   `deferred_watch`/`trigger_met`.** Rationale: `deferred_watch`'s stated
   invariant is "the scheduler never evaluates a trigger; un-defer is a human
   action." Auto-setting `trigger_met` from a clock would reverse that invariant
   and conflate human-un-defer provenance with clock-un-defer. A sibling
   predicate keeps eligibility a **pure read** (it reads `milestone_deps_satisfied_at`;
   it never writes `trigger_met` as a signalling side effect) and leaves
   `deferred_watch` semantically untouched.

3. **Wall-clock, persisted, restart-safe.** Dated `at` and the delayed anchor
   `milestone_deps_satisfied_at` are ISO-8601 UTC strings compared against
   `datetime.now(UTC)` — **not** `self._time_source()` (monotonic, resets on
   restart). A "1 week" delayed timer must survive orchestrator restarts; a
   persisted wall-clock anchor + recompute-on-tick achieves this with no in-memory
   timer. New `datetime.now(UTC)` call sites carry the project's clock-guard
   treatment (fold into task 2281's consolidation or tag `# clock-exempt`).

4. **Deps-satisfied stamping is a per-tick sweep, frozen-once.** `_deps_satisfied`
   (the same evaluator dispatch uses, `scheduler.py:2596`) determines the anchor.
   The stamp is written **once** (never overwritten), so a later dependency
   regression does not restart the timer. Dispatch **still** re-checks live
   `_deps_satisfied` at eligibility, so a regressed dep withholds dispatch even
   after the timer elapses — the milestone fires only when *both* the timer has
   elapsed *and* deps are currently satisfied. A no-dep delayed milestone has its
   deps trivially satisfied at `pending` time → timer starts immediately →
   fires `after_secs` from filing.

5. **Predicate verdict is exit-code only.** `rc==0` → `done` (provenance
   `kind='deterministic-milestone'`, carrying the script's stdout tail as `note`);
   `rc!=0` → born-at-L2 `milestone_check_failed` escalation (detail carries `rc`
   + stdout tail) + task `blocked`; timeout → escalate (reusing the runner's
   existing timeout→escalate path). The **script owns the threshold** ("< 5%");
   the orchestrator parses nothing and therefore asserts **no numeric premise**
   (clean under G6). Resolving the escalation (`resume`) drives the task `done`,
   reusing the runner's section-1 resume path.

6. **One-shot.** A milestone fires exactly once. No re-check loop, no interval /
   deadline state. Recurrence is explicitly out of scope (§10).

7. **`before_done.kind` discriminator.** `BeforeDone.kind: Literal['deploy','predicate'] = 'deploy'`
   (default preserves every existing deterministic task byte-identically).
   Predicate mode **forbids** `target_unit` (no unit to verify) and **forbids**
   `always_escalates=True` (predicate escalation is inherently conditional on
   `rc!=0`; "always ask even on pass" is the existing `kind='deploy'`+act-then-ask
   feature). The submit guard enforces both.

8. **Duration is `after_secs: int` (seconds), not a human string.** A duration
   string ("7d", "P7D") would require a parser Python's stdlib does not provide —
   novel substrate we decline to build. `datetime.fromisoformat` (stdlib, exists)
   parses the dated `at`. A human-duration convenience is a deferred tactical
   nicety (§11).

## 6. Contract section (H)

### 6.1 Milestone metadata schema

```jsonc
metadata = {
  // ── author-supplied ────────────────────────────────────────────────
  "milestone": {
    "mode": "dated" | "delayed",
    "at":  "2026-08-01T00:00:00+00:00",   // required iff mode=="dated"; ISO-8601 (datetime.fromisoformat-parseable)
    "after_secs": 604800                    // required iff mode=="delayed"; int > 0
  },

  // ── for the autonomous exemplar (optional; orthogonal to milestone) ──
  "task_kind": "deterministic",
  "before_done": {
    "kind": "predicate",                    // NEW; default "deploy"
    "script": "scripts/check_merge_flakiness.sh",
    "args": ["--window-days", "7", "--threshold", "0.05"],
    "timeout_secs": 120
    // target_unit FORBIDDEN in predicate mode; always_escalates FORBIDDEN
  },

  // ── scheduler-stamped provenance (NEVER author-supplied) ────────────
  "milestone_deps_satisfied_at": "2026-07-25T09:14:03+00:00",  // delayed only; frozen-once
  "milestone_fired_at":          "2026-08-01T00:00:11+00:00"   // audit; stamped when the gate opens
}
```

Typed shapes (`shared/task_metadata.py`):
- New `Milestone(BaseModel, extra='allow')`: `mode: Literal['dated','delayed']`,
  `at: str | None`, `after_secs: int | None`, with a model-validator enforcing
  `at` present-and-parseable iff dated, `after_secs` present-and-`>0` iff delayed.
  Registered via `register_metadata_submodel('milestone', Milestone)`.
- `BeforeDone.kind: Literal['deploy','predicate'] = 'deploy'`.
- `DoneProvenance.kind` Literal gains `'deterministic-milestone'`.

### 6.2 Scheduler-gate invariants

`Scheduler._milestone_time_gated(task, now_wall) -> bool` — **pure predicate**,
`True` means *withhold*:

- No `metadata.milestone` → `False` (not a milestone; unaffected).
- `mode=='dated'` → withhold while `now_wall < parse(at)`.
- `mode=='delayed'` → withhold while `milestone_deps_satisfied_at` is unset **or**
  `now_wall < parse(milestone_deps_satisfied_at) + after_secs`.
- Any malformed/unknown value → **fail-safe withhold** (`True`) and log
  (defense-in-depth; the submit guard already rejects malformed specs at write).

Wired into `_eligible_for_dispatch` immediately after `_deferred_watch_gated`, so
it applies to **both** the scored and pin dispatch loops automatically.

`Scheduler._stamp_milestone_deps_satisfied(...)` — a **per-tick sweep** called
from `acquire_next` alongside `_gc_expired_cooldowns` (never from
`_eligible_for_dispatch`, preserving its purity). For each `pending`,
`mode=='delayed'`, milestone task **without** `milestone_deps_satisfied_at`
whose deps `_deps_satisfied(...)` → stamp `datetime.now(UTC).isoformat()` once
via `update_task(..., metadata_mode='merge')`.

### 6.3 Predicate verdict contract

`DeterministicRunner.run`, when `before_done.kind == 'predicate'` (branch taken
**before** the deploy path at `deterministic_runner.py:919`):

| Outcome | Action |
|---|---|
| script `rc == 0` | `set_task_status(done, done_provenance=kind='deterministic-milestone', note=<stdout tail>)` → `DONE` |
| script `rc != 0` | file born-at-L2 `milestone_check_failed` (detail = `rc` + stdout tail), stamp `gate_escalated_at`, `set_task_status(blocked)` → `BLOCKED` |
| timeout | existing timeout path → escalate + `BLOCKED` |
| crash mid-check | re-run on resume is **safe** (a predicate is a read-only check — unlike a deploy it is idempotent), so predicate mode needs no I1 once-only stamp; the escalation quiescence guard alone prevents churn |
| escalation resolved (`resume`) | section-1 resume drives to `done` — with a predicate-aware provenance kind (`deterministic-milestone`, not `deterministic-deploy`) |

No systemd inspect, no baseline, no fresh-PID verify on this branch.

## 7. Boundary-test sketch (H)

Facing **both** the scheduler-gate side and the runner-verdict side:

| # | Scenario | Preconditions | Postconditions (asserted) |
|---|---|---|---|
| B1 | Dated gate holds then releases | milestone `mode=dated`, `at`=T+Δ; injected clock | task withheld from `_eligible_for_dispatch` while `now<at`; eligible once `now≥at` |
| B2 | Delayed anchor stamped on deps-done | `mode=delayed`, dep X pending | no `milestone_deps_satisfied_at` while X pending; stamped exactly once the tick X→`done` |
| B3 | Delayed timer from anchor | anchor stamped at S, `after_secs`=Δ | withheld while `now<S+Δ`; eligible at `now≥S+Δ` |
| B4 | Anchor frozen-once | anchor stamped; sweep runs again | stamp value unchanged across subsequent ticks |
| B5 | Dep regression after elapse | timer elapsed, dep X reopened→`pending` | withheld (live `_deps_satisfied` fails) even though timer elapsed |
| B6 | Restart survival | anchor persisted, process restarted | recomputed gate from persisted anchor — no timer reset |
| B7 | Predicate pass → done | `before_done.kind=predicate`, script exits 0 | task `done`, `done_provenance.kind='deterministic-milestone'`, stdout tail in `note` |
| B8 | Predicate fail → escalate | script exits 1 | pending L2 `milestone_check_failed` for the task; task `blocked`; detail carries rc+tail |
| B9 | Predicate fail resume → done | B8 escalation resolved; task re-dispatched | task `done` (predicate provenance); check NOT re-run needlessly / no phantom-done |
| B10 | No unit inspect on predicate | predicate task, no `target_unit` | `unit_inspector` / systemd never invoked (would previously fail baseline) |
| B11 | Submit rejects malformed spec | `mode=delayed` with no `after_secs`; `mode=dated` with unparseable `at`; predicate with `target_unit` or `always_escalates=True` | `submit_task` returns a structured `ValidationError`; nothing persisted |
| B12 | Normal-agent milestone | `task_kind=normal`, milestone dated | held until `at`, then dispatched as a normal agent task (existing path) |

## 8. Pre-conditions for activating

None external. All substrate (deferred-watch gate, DeterministicRunner,
typed-metadata schema, escalation queue) is on `main`. This PRD is
immediately decomposable.

## 9. Cross-PRD relationship (G4)

No contested cross-PRD seams. This PRD is standalone orchestrator scheduling
infra. It **shares files** with two landed/in-flight areas — flagged for merge
awareness, not ownership disputes:

| Other work | Direction | Shared surface | Owner | Status |
|---|---|---|---|---|
| task 2234 (deferred-watch gate) | extends | `scheduler._eligible_for_dispatch` gate chain | this-prd | landed on main |
| `dispatch-admission-load-cap.md` | co-touches | `scheduler.acquire_next` per-tick sweeps | this-prd (additive sweep) | independent |
| deterministic-task-kind PRD | extends | `DeterministicRunner.run`, `BeforeDone`, `deterministic_task_guard` | this-prd (additive predicate branch) | landed on main |

## 10. Decomposition plan

B+H shape: foundation → parallel vertical slices → integration gate → docs.
(Greek labels; real task IDs assigned at decompose.)

- **α — Typed milestone + predicate + provenance schema** *(shared)*.
  Add `Milestone` sub-model + `register_metadata_submodel('milestone', …)`;
  `BeforeDone.kind: Literal['deploy','predicate']='deploy'`; add
  `'deterministic-milestone'` to `DoneProvenance.kind`.
  *Signal (intermediate → unlocks β,γ,δ):* a milestone spec + a
  `kind='predicate'` before_done round-trip through `parse_metadata` with **zero**
  warnings; `mode='delayed'` without `after_secs` and an unknown provenance kind
  each raise `ValidationError`.

- **β — Scheduler milestone time-gate** *(orchestrator)*. `_milestone_time_gated`
  pure predicate + `_stamp_milestone_deps_satisfied` per-tick sweep, wired into
  `_eligible_for_dispatch` + `acquire_next`. Uses `datetime.now(UTC)`.
  *Signal (user-observable):* with an injected clock, a dated milestone task is
  absent from dispatch (`get_scheduler_state`) while `now<at` and dispatches once
  `now≥at`; a delayed task stamps `milestone_deps_satisfied_at` the tick its dep
  goes `done` and dispatches `after_secs` later. Depends **α**.

- **γ — Predicate deterministic mode** *(orchestrator)*. New `before_done.kind=='predicate'`
  branch in `DeterministicRunner.run` (exit-code verdict, no systemd verify),
  `deterministic-milestone` provenance, `milestone_check_failed` escalation,
  predicate-aware section-1 resume.
  *Signal (user-observable):* a deterministic predicate task whose script exits 0
  goes `done` with `done_provenance.kind='deterministic-milestone'` (via
  `get_task`); a script exiting 1 files a pending `milestone_check_failed` L2
  (via `get_pending_escalations`), and resolving it drives the task `done`.
  Depends **α**.

- **δ — submit_task validation** *(fused-memory)*. Extend `deterministic_task_guard`
  / `tools.submit_task` to validate `metadata.milestone` (dated needs parseable
  `at`; delayed needs `after_secs>0`) and predicate `before_done`
  (forbid `target_unit`, forbid `always_escalates=True`); allow `metadata.milestone`
  on both `normal` and `deterministic` tasks.
  *Signal (user-observable):* the B11 malformed submissions each return a
  structured `ValidationError` dict; a well-formed milestone persists and is
  retrievable via `get_task`. Depends **α**.

- **ε — End-to-end integration gate** *(orchestrator)* — **the leaf**
  (C-as-integration-gate). The boundary-test sketch (§7) as executable tests:
  the exemplar wired end-to-end (delayed deterministic predicate milestone
  depending on task X → anchor stamps on X→done → check runs `after_secs` later →
  exit 0 → done / exit 1 → `milestone_check_failed` L2), plus the normal-agent
  variant (B12) and restart survival (B6).
  *Signal:* the §7 boundary suite passes. Depends **β, γ, δ**.

- **ζ — Operator docs** *(docs)*. Add a "Milestone tasks (dated / delayed)"
  subsection to `CLAUDE.md`'s Task Routing: the `metadata.milestone` schema,
  orthogonality to `task_kind`, the predicate exit-code contract, and the
  frozen-once delayed-anchor semantics.
  *Signal:* `CLAUDE.md` documents the shipped milestone metadata + the exit-code
  verdict contract. Depends **ε** (documents shipped behaviour).

## 11. Open questions (tactical — deferred, not design-blocking)

1. **Human-duration convenience.** Accept `"7d"` / ISO-8601 `"P7D"` alongside
   `after_secs`? Needs a parser (no stdlib support). *Suggested:* ship
   `after_secs` only; add a convenience later if authoring friction shows up.
   Decide during δ.
2. **`milestone_fired_at` audit stamp.** Nice-to-have provenance; the task's own
   status history already records dispatch time. *Suggested:* stamp it in the
   dispatch bookkeeping if cheap, else drop. Decide during β.
3. **Escalation `options` on `milestone_check_failed`.** Should the fail
   escalation carry `gate_options` (e.g. "accept anyway" / "re-run" / "extend")
   for the human? *Suggested:* start with a plain escalation; add `gate_options`
   if the escalation-watcher workflow wants them. Decide during γ.
4. **Sweep cost at scale.** The per-tick deps-satisfied sweep scans pending
   delayed milestones. *Suggested:* fine at current task volumes; if it shows on
   a profile, index milestone tasks separately. Decide during ε.

## 12. Out of scope

- **Recurrence / polling** — one-shot only (§5.6). A "re-check weekly until it
  passes" mode is a separate future PRD.
- **Metric computation inside the orchestrator** — the predicate script owns the
  threshold; the orchestrator parses nothing (§5.5).
- **Cron / calendar schedules** — this is a single fire per task, not a recurring
  schedule. A general scheduled-task system is a distinct PRD.
- **Structured-stdout verdicts** — exit-code only (§5.5); a JSON verdict contract
  is a future extension.
- **Automatic un-defer of prose `deferred_watch` triggers** — untouched; remains
  manual by design (§5.2).
