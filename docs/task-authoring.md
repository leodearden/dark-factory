# Task Authoring Reference

Reference for anyone — human operator or agent — filing tasks into the
dark-factory orchestrator: the metadata fields a task can carry, what each
does, how the scheduler interprets it, and the validation rules enforced
at write time. Field shapes and behavior, not day-to-day operation (see
`OPERATIONS.md`), overall system shape (see `ARCHITECTURE.md`), or repo
orientation (see `README.md`). For decomposing a PRD into a task batch,
see `skills/prd/SKILL.md` — the primary producer of the batch-submission
pattern in §1 and §9.

Every field shape below is load-bearing: it mirrors the validated Pydantic
models in `shared/src/shared/task_metadata.py` (and, for delivered-checks,
`shared/src/shared/capability_manifest.py`) and the fused-memory/scheduler
code that reads them. Treat each table and fenced shape as the contract,
not a paraphrase.

---

## 1. How tasks enter the system

**All task operations go through fused-memory MCP tools** — never the
Taskmaster CLI or Taskmaster MCP server directly. Routing through
fused-memory is what makes the `TaskInterceptor` emit reconciliation
events for every state transition; a task created or transitioned any
other way is invisible to reconciliation and the legibility tooling that
depends on it.

### Two creation paths

**1. Ticket → curator (default, single-task path).** `submit_task`
persists a ticket (`{"ticket": "tkt_<id>"}`) and returns immediately. The
`TaskCurator` (an async worker) decides — dedupe, combine, or create — and
lands the result in the SQLite task backend. Callers follow up with
`resolve_ticket` to obtain the final `task_id`, polling or blocking.

**2. `planning_mode=True` → `commit_planning` (batch path).** For
decomposing a PRD into many cross-dependent tasks (`/prd` decompose),
`submit_task(planning_mode=True)` bypasses the curator and writes the task
directly in `deferred` status — one committed write, no transient
`pending` state — so the scheduler can't claim a sibling before the rest
of the batch and its dependency wiring exist. Once every task and
dependency is in place, `commit_planning(task_ids="42,43,44",
target_status="pending")` atomically flips the whole batch from
`deferred` to `pending` (or to `cancelled` to discard it) under one
per-project write lock, so the scheduler sees a coherent batch on its next
poll rather than picking up siblings one at a time. A `pending` commit
also indexes the batch into the curator's search corpus;
`deferred`/`cancelled` commits are never indexed.

Workflow agent roles (architect, implementer, steward, deep_reviewer) file
their own follow-up tasks the same way, fire-and-forget, via `submit_task`.

### Write-tagging convention

Every write operation (task or memory) should carry:

- **`project_id`** — the project's canonical id (e.g. `"dark_factory"`).
- **`agent_id`** — a descriptive identifier for the writer, e.g.
  `"claude-interactive"`, `"claude-task-7"`, `"reconciliation-stage-1"`.

### `project_root` threading

Every task tool call takes `project_root` — the absolute path to the
target project's checkout (e.g. `/home/leo/src/dark-factory`). It's how
fused-memory locates the right per-project task backend and write lock;
pass the same value consistently for a given project across a session.

---

## 2. Task statuses & transitions

**Vocabulary** (`shared/src/shared/task_statuses.py`):

```
pending, in-progress, blocked, deferred, review, merge-deferred, infra-hold, done, cancelled
```

- `TERMINAL = {done, cancelled}`
- `WORKFLOW_PRESERVE = TERMINAL ∪ {deferred, blocked, merge-deferred}`

Status transitions `done`, `blocked`, `cancelled`, and `deferred` trigger
targeted reconciliation automatically.

**Legal transitions** (`shared/src/shared/task_transitions.py`; every pair
is annotated with its call site in code — this is a summary, not the full
state machine. See `ARCHITECTURE.md` for the complete machine and the
`WorkflowStateMachine`/`TaskInterceptor` enforcement layers):

```
pending ─dispatch→ in-progress ─merge→ done
in-progress ─block→ blocked ─re-pend→ pending
in-progress ─park(train)→ merge-deferred → done/blocked/cancelled/pending
in-progress ─requeue→ pending
pending → deferred (planning) → pending/done/blocked/cancelled
any non-terminal → cancelled
in-progress ⇄ infra-hold; in-progress ⇄ review
blocked ─resume(infra)→ in-progress
done/cancelled ─reopen (audit, requires reopen_reason)→ *
```

Load-bearing enforcement of who may perform which transition lives in
fused-memory's `TaskInterceptor`, keyed by actor class — for example, a
reconciliation actor may never transition a task away from `in-progress`.
Within a live agent workflow, `set_task_status` is restricted to the
steward role; every other role drives status changes indirectly through
the workflow's own state machine.

### `done_provenance` requirement

Every `done` write must carry a `metadata.done_provenance` object naming
its `kind`. This is server-side schema-enforced and backstopped by a
`git merge-base --is-ancestor` check where applicable — a task cannot be
marked `done` on the strength of an unverified claim.

| `kind` | Meaning | Conditional requirements |
|---|---|---|
| `merged` | Landed via the normal merge queue | `commit` required |
| `found_on_main` | Discovered already on `main` (e.g. stranded-task recovery) | `commit` **and** `note` required; `stamped_at` written server-side (see below) |
| `deterministic-deploy` | `DeterministicRunner` cross-unit deploy completed | — |
| `deterministic-deploy-scheduled` | `DeterministicRunner` self-restart scheduled via detached `systemd-run` | — |
| `deterministic-gate` | A pure deterministic gate (no `before_done`) resolved | — |
| `deterministic-milestone` | A `kind='predicate'` milestone check exited `0` | — |
| `operational-verified` | A `normal`-task no-code operational ask closed via a resolved escalation, not a merge | `escalation_id` **and** `note` required |

`operational-verified` is commitless like the `deterministic-*` kinds, so
it is likewise exempt from the reopen-freshness gate (which only inspects
`merged`/`found_on_main`). It is accepted only from non-recon-stage
callers, on both the fresh `done` transition and the same-status
`done`→`done` repair path — a recon stage may never self-authorize an
operational close.

#### `stamped_at` (server-written, `found_on_main` only)

`found_on_main` stamps additionally carry `stamped_at`: an ISO-8601 UTC
timestamp recording **when the attribution was asserted**.

- **Never supply it.** It is written server-side by fused-memory's
  `_validate_done_provenance` chokepoint, which every `found_on_main`
  producer funnels through (the fresh `done` transition, the same-status
  repair seam, agent `set_task_status` calls, and the orchestrator's
  `Scheduler.mark_done`). A caller-supplied value is **discarded with a
  warning**, not rejected — rejecting would break the repair seam, which
  re-submits an already-stamped blob. A repair **refreshes** the stamp,
  which is correct: a repair is a fresh assertion of the attribution.
- **Scoped to `found_on_main`.** No other kind carries it. `merged`
  already has independent landing evidence (merge-queue journal +
  ancestor check + commit citation); `found_on_main` is the
  attribution-by-inference kind that lacked any record of *when* the
  inference was made.
- **Its absence is load-bearing.** `updatedAt` cannot answer "when was
  this asserted?" — any later write to the task bumps it. So stamps
  written before task 3576 landed have no `stamped_at`, and because every
  write through the chokepoint since then populates it, **absence proves
  the stamp predates 3576**. The soak-gate predicate
  (`fused-memory/scripts/check_found_on_main_spurious_rate.py`) uses
  exactly this to separate legacy backlog (`stamp_class=legacy` —
  reported but never gating) from genuinely new stamps
  (`stamp_class=fresh` — which gate). For that reason the field is
  optional, not conditionally required, and must stay that way.
- **Absent ≠ unparseable.** That argument turns on the field being
  *missing*. A `stamped_at` that is *present* but unparseable proves the
  opposite — a post-3576 write whose freshness merely cannot be read — so
  the predicate classes it `stamp_class=corrupt`, logs it at `WARNING`,
  and **gates** on it. Only `legacy` is exempt from gating; silently
  demoting a corrupt stamp would be a fail-open in exactly the direction
  the gate exists to catch.

### Terminal-task write freeze (recon-stage boundary)

Reconciliation-stage agents (`agent_id` prefixed `recon-stage-`) may **not**
write to a terminal (`done`/`cancelled`) task via `update_task`. The
server-side guard (`fused_memory.middleware.recon_write_policy`) rejects such a
write with `ReconTerminalWriteRejected`. Only three narrow, sanctioned
exemptions exist:

- **The same-status `done_provenance` repair seam** — `set_task_status(id,
  'done', done_provenance={...})` on an already-`done` task, routed to
  `TaskInterceptor._repair_done_provenance_same_status`. It writes
  `done_provenance` **only**, and validates it (`kind` enum + `git merge-base
  --is-ancestor`); it cannot touch any other field.
- **`CLEARABLE_ANNOTATION_KEYS` clears** — a merge-mode clear of an advisory
  annotation (e.g. `possible_scope_mismatch`). These stay clearable, but
  `possible_scope_mismatch` is no longer inert: in its containment-**confirmed**
  form it can block a dispatch (§3.2.1), so clearing it removes that evidence.
  The independent `metadata.files` and `metadata.cross_repo` legs of that gate
  are unaffected by a clear.
- **`x_`-prefixed annotation adds** — a merge-mode add of forward-compat
  `x_`-namespaced annotation keys.

There is intentionally **no** recon-stage corrective path for load-bearing
string content fields (`details`, `description`, `title`, `prompt`,
`priority`, `dependencies`). This is a permanent, intentional **human-gate
boundary, not a bug**: unlike `done_provenance` — a structured, schema- and
git-ancestor-validated evidence field with a mechanical ground truth (the
commit SHA) whose same-status repair seam can therefore be safely
auto-authorized even for a recon-stage caller — these fields are free-form
prose whose "correct" replacement is a human judgment call, exactly what the
terminal-write human gate protects. A recon stage must never self-authorize
such a correction.

When a terminal task's `details` (or another content field) is stale or wrong,
**file a human-gated workaround task** to correct it rather than re-diagnosing
the `ReconTerminalWriteRejected` rejection. Both prior recurrences were
resolved this way: `autopilot_video` 544 was documented-as-artifact, and the
644/650 pair went through a human-gated workaround task.

---

## 3. Dependencies

### 3.1 Local dependencies

A bare integer `depends_on` (e.g. `"5"`) records the dependency in the
task's integer `dependencies` table — the original, same-project
mechanism. A task dispatches only once every local dependency reaches a
TERMINAL status (`done`/`cancelled`); an intra-atomic-train
`merge-deferred` sibling is also accepted as satisfying.

### 3.2 Cross-project external dependencies

A task can declare a dependency on a task in **another** project using the
qualified `"project_id:task_id"` form (e.g. `"dark_factory:42"`). When
`add_dependency` receives a `depends_on` value containing `:`, it routes
the dep to `metadata.external_deps` (a list of canonical
`"project_id:task_id"` strings) instead of the integer `dependencies`
table — no schema migration required.

```python
# Qualified form → appended to metadata.external_deps
add_dependency(
    id="<dependent_task_id>",
    depends_on="dark_factory:42",   # project_id:task_id
    project_root="<project_root>",
)
# Bare integer → existing integer dependencies table (unchanged)
add_dependency(id="<id>", depends_on=13, project_root="<project_root>")
```

The foreign target is **not** verified at write time; existence is
resolved at gate time.

**Resolution: `get_external_statuses`**

The scheduler resolves `metadata.external_deps` at each dispatch tick via
the read-only fused-memory tool `get_external_statuses(deps: list[str]) ->
dict[str, str]`. It takes a list of `"project_id:task_id"` strings, looks
each up in the shared fused-memory registry, and returns a status per dep.
Unresolvable deps return explicit sentinels:

| Sentinel | Meaning |
|---|---|
| `"unknown_project"` | `project_id` not in the registry |
| `"unknown_task"` | Project known; no top-level task with that id |
| `"malformed"` | Not parseable as `project_id:task_id` |

**Dispatch-time policy**

The gate lives in the **dependent's** scheduler only — it does not affect
the upstream project's orchestrator. External deps are checked at dispatch
time; they are not re-evaluated after a task has been dispatched.

| Resolved status | Scheduler action |
|---|---|
| `done` | Satisfied — counts toward dispatch |
| `cancelled` | Not satisfied → `_mark_blocked(escalate_to_human=True)` immediately |
| `unknown_project` / `unknown_task` / `malformed` | Not satisfied; grace period then escalate after repeated unresolved cycles |
| Any other live status (`pending`, `in-progress`, …) | Not satisfied; keep waiting |
| Resolver error (transient timeout / server hiccup) | Not satisfied this tick — fail-safe wait, no grace counter increment |

A task is dispatched only when **all** local deps **and** all
`metadata.external_deps` are `done`.

Deterministic deploy and gate tasks (§5) use this same dependency
mechanism, including cross-project deps. The older convention of filing
deploy capstones in `dark_factory` with a `dark_factory`-internal
dependency — a workaround for an external-dep gate bug since fixed — is
retired; use a `task_kind='deterministic'` deploy or gate task with normal
deps instead.

### 3.2.1 Cross-repo deliverables (`metadata.cross_repo`)

A **cross-repo deliverable** is a task filed under one project whose
declared `metadata.files` are **all** owned by **one other** project — so
the filing project's own branch is legitimately **empty**, because the
actual code change lands on the owning project's branch (the reify-task
5308 shape: a `reify` task declaring only `orchestrator/…` dark_factory
paths). Left unmarked, the orchestrator's pre-merge Decision-1 gate would
see every declared file "not touched" on the (empty) local branch and
false-flag the task as undelivered.

`metadata.cross_repo` (boolean) marks such a task, with the companion
`metadata.cross_repo_project` naming the owning project. You normally do
**not** set these by hand: the **fused-memory submit path auto-sets** them
when it detects the all-foreign shape, i.e. when **every** `metadata.files`
entry is owned by **one** other **registered** project **and the filing
project is itself registered**. Both sides of that relationship must be
registered projects — a cross-repo deliverable is a relationship between two
**known** projects, not a blanket allowance for any namespace to declare
another project's files. A submission from an **unregistered** filer, or one
that **mixes** local and foreign files, is **not** tagged and stays a hard
path-scope reject (the task-2206 anti-bypass guard is preserved).

When the marker is present, the orchestrator pre-merge narrowing gate treats
the task as a cross-repo deliverable: instead of flagging "files not
touched" and forcing a dishonest plan-narrowing pass, it routes the merge
attempt to the terminal outcome `OutcomeKind.plan_files_cross_repo` and
blocks it on the **normal** ladder with `category='cross_repo_deliverable'`
and `suggested_action='verify_external_landing'` — an honest state naming
the real situation (verify the deliverable landed on the owning project's
branch) rather than the false "implementation has not delivered". The
orchestrator additionally honors an explicitly-set `metadata.cross_repo` for
the absolute-path-foreign shape it can classify without the registry.

**A marked task no longer reaches merge time at all.** As of task 3121 the
orchestrator also runs a **dispatch-time cross-repo admission gate**
(`orchestrator/src/orchestrator/cross_repo_gate.py`), in `Harness._run_slot`
just ahead of the D4 substrate gate and **before any agent spins up**. A task
is blocked there with `block_reason='cross_repo_misfile'` and an **L1
`scope_violation`** naming the owning project when **any** of:

- `metadata.cross_repo` is truthy — the marker leg;
- every entry in `metadata.files` is an **absolute** path resolving outside
  the orchestrator's `project_root` — the containment leg, which needs no
  registry;
- a `metadata.possible_scope_mismatch` stamp whose `matched_paths` are
  **confirmed** foreign by that same containment test.

An advisory stamp alone does **not** block — see the prose-citation rules
below; the prose scan over-fires, so confirmation by path containment is
required before it can stop a dispatch. Unreadable metadata is a logged SKIP,
never a silent pass. The fix for a blocked task is to **refile it under the
owning project**, not to unblock it in place.

The merge-time `OutcomeKind.plan_files_cross_repo` route above is unchanged
and still applies to a task that reaches merge by another path. It was
previously the *only* consumer of the cross-repo signal, which is why a task
whose branch never got built — the reify-5638 shape — used to reach the
architect, burn an agent, and land an L2 anyway.

The dispatch gate reads the task **as it stands at dispatch**, so a marker or
a `metadata.files` list stamped **after** creation (the `files_tagged_at`
shape) is honoured too — a submit-time gate structurally cannot see those.

**Merely CITING another project's paths in prose is not a scope error.** A
task whose description references `orchestrator/src/foo.py` as evidence, as
a thing to mirror, or as prior art is created normally — the citation never
blocks anything. It can, however, attract the advisory annotation
`metadata.possible_scope_mismatch` (plus a non-blocking `scope_violation`
escalation an operator then has to read), because a prose scan cannot tell
"modifies X" from "mentions X".

The annotation stays advisory. Its one behavioural consumer is the
dispatch-time gate above, and only in its **confirmed** form: the stamp blocks
a dispatch just when its `matched_paths` independently resolve foreign by path
containment. An unconfirmed prose stamp still blocks nothing — turning a
false-positive advisory into a stalled task plus a spurious L1 would be
strictly worse than leaving it advisory.

A **declared deliverable** can, so it is what the advisory is attributed on.
The annotation is suppressed when your declared deliverables attest local
work — meaning **both** of:

- at least one entry across `metadata.files`, `metadata.files_to_modify` or
  `metadata.modules` is owned by the project you are filing into, **and**
- **no** entry across those three keys is owned by a *different* project.

When both hold, the prose citation is treated as incidental and neither the
annotation nor the escalation fires. **Supplying accurate deliverable
metadata is therefore the supported way to keep a legitimately
cross-repo-*referencing* task quiet** — and it is worth doing regardless,
since the same fields drive scope assignment and the pre-merge delivery
gate. Declaring only *unowned* paths (`README.md`, `docs/x.md`) does **not**
count: that is no evidence of local work, so the advisory still fires.

For the residual case — a task that genuinely belongs here but can declare
no filer-owned deliverable at all — pass `metadata.routing_override_reason`
(a non-empty string). It is the pre-existing explicit bypass and skips the
path-scope guards entirely, both the advisory **and** the hard reject, so
use it only when you are sure the task belongs to the submitting project.

None of this relaxes the rule above. A submission whose **`metadata.files`**
mix local and foreign entries is still a hard reject — attribution is
consulted only after that check has already passed, so a locally-owned entry
can never buy a mixed `files` list past it. A foreign entry that the reject
never classified (one in `metadata.modules`, or in `metadata.files_to_modify`
when `metadata.files` is also present and takes precedence) does not reject,
but it *does* fail the second condition above, so such a submission keeps the
annotation and the escalation rather than being silently suppressed.

### 3.3 Delivered-check dependency gate (`metadata.delivered_checks`)

A **local** (same-project) dependency can additionally carry
`metadata.delivered_checks` — a list of check descriptors asserting that
the capability the dependency claims to deliver is actually present on the
committed `main` tree, not just that the task record reached a terminal
status. Every scheduler tick, `Scheduler._compute_delivered_check_cache`
sweeps every distinct TERMINAL (`done`/`cancelled`) local dependency
carrying this metadata against `main` and caches the result per
`(dep_task_id, main_sha)` — a capability landing on `main` self-heals the
very next tick with no operator action, since a new SHA prunes the stale
cache entry.

**Descriptor shape** (`shared.capability_manifest.DeliveredCheckMeta`; the
`grep`/`script` fields are mutually exclusive and cross-validated):

```
{
  name: str,                      # required; names the capability in escalations
  kind: "grep" | "script",

  # kind="grep" — evaluated against the COMMITTED tree at `main` via
  # `git grep -E -e <pattern> <ref> [-- <paths...>]`
  pattern: str,                    # required iff kind="grep"
  expect: "present" | "absent",    # required iff kind="grep"
  paths: [str],                    # optional, kind="grep" only

  # kind="script" — evaluated against the WORKING CHECKOUT via
  # `<project_root>/<script> <args>`, bounded by timeout_secs
  script: str,                     # required iff kind="script"
  args: [str],                     # optional, kind="script" only
  timeout_secs: int,                # required (>0) iff kind="script"
}
```

`grep` is the primary kind (reads exactly what's on `main`, immune to
working-checkout dirtiness); `script` is the escape hatch for capabilities
that can't be expressed as a pattern, at the cost of running against the
working checkout rather than a materialized `main` tree.

**Dispatch-time policy**

| Check outcome | Scheduler action |
|---|---|
| All checks DELIVERED on `main@SHA` | Satisfied — dep counts toward dispatch |
| ≥1 check FAILED, consecutive-fail streak `< grace_cycles` | Withheld; `delivered_check_gate_held` event emitted each held tick (hold-visibility only) |
| ≥1 check FAILED, streak reaches `grace_cycles` | Born-at-L2 `dependency_capability` escalation naming the failed check/pattern-or-script/dep id/`main` SHA; dependent → `blocked`; streak cleared (re-fires on a later re-crossing) |
| A check ERRORS (git/script failure) or exceeds `check_timeout_secs` | Fail-safe wait — dep left uncached, **no** streak bump on either counter, retried next tick |
| `delivered_checks.enabled = false` | Gate entirely inert — no sweep, no cache, no streaks, no escalation; the dep gate takes its legacy arm-off path as if the metadata didn't exist |

The born-at-L2 escalation (`agent_role='orchestrator-scheduler'`,
`severity='critical'`) bypasses the auto-watcher and routes straight to a
human, mirroring the cross-project external-dep gate's L1 filer but one
level up — a persistently false capability claim is a "someone must look
at this now" condition, not a routine triage item.

**Config knobs** (`delivered_checks.*`, all green-tier hot-reloadable):

| Knob | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Kill switch — set `false` to disable the gate entirely |
| `grace_cycles` | `3` | Consecutive FAILED ticks (per dependent, dep pair) before the born-at-L2 escalation fires |
| `check_timeout_secs` | `120` | Per-check wall-clock timeout; a hung check maps to the same fail-safe outcome as a runner error |
| `max_checks_per_tick` | `50` | Per-tick fan-out budget across all checked deps |

**Manual re-pend recipe:** exactly like the external-dep gate, a dependent
blocked by this escalation is **not auto-re-pended**. Once the underlying
capability actually lands on `main` (or the check itself is fixed), an
operator must manually set the dependent back to `pending` to reopen it —
resolving the escalation records the human decision but does not by itself
unblock the task.

---

## 4. Fast path (`metadata.complexity`)

Set `metadata.complexity = "simple"` to route a task to the single-agent
SIMPLE_TASK fast path (one Sonnet agent explores, plans, edits, and
commits; the architect+implementer pair is skipped, but verify/review/merge
still run). The only meaningful value is `"simple"` — absent or any other
value routes to the full architect path.

**When to declare `"simple"`:** the change is a single coherent edit — docs
or comments, a rename, a localized behaviour-preserving refactor, a
typo/wording fix, a one-spot bug fix — that needs **no new abstraction and
no cross-module design**, and you can name the target file(s). A `simple`
task may be high-priority and may touch several files/modules, as long as
the *change* is mechanically simple. **When unsure, omit it** — the full
path is the safe default, and a mis-declared task simply falls back to the
architect.

**Hard-blocker veto:** if the task description contains a hard-blocker
token (`migration`, `architecture`, `integration test`, `design ... new`,
`implement ... new feature`), the fast path is vetoed even if
`complexity='simple'` is set.

**Hard escape:** `metadata.force_full_path = true` always forces the full
architect path regardless of `complexity`.

---

## 5. Deterministic tasks (`task_kind='deterministic'`)

Set `task_kind='deterministic'` on `submit_task` to skip the LLM pipeline
entirely (no worktree, no branch, no agent, no diff) and route to the
**`DeterministicRunner`** — a small state machine that runs an optional
committed action, escalates born-at-L2 when required, and marks the task
`done` once both are satisfied. Dispatch eligibility uses the same
dependency gate as every other task (§3).

`task_kind` is a first-class `submit_task` parameter (`'normal'` default |
`'deterministic'`), persisted to `metadata.task_kind`.

**`metadata.before_done`** — committed-script reference (set at
`submit_task`):

```
{
  script: "<repo-relative path>",  # must exist & be executable
  args: [],                         # list[str], default []
  env: {},                          # dict[str,str], default {}
  cwd: "<project_root>",            # default: project_root
  timeout_secs: 120,                # int, required; runner kills + escalates on timeout
  target_unit: None,                # str|None; None → cross-unit (no self-kill)
  kind: "deploy",                   # "deploy" | "predicate"; default "deploy" — see §6 for "predicate"
}
```

**`metadata.always_escalates`** (`bool`, default `false`) — file a
born-at-L2 escalation after the action completes (or immediately if no
action); task goes `blocked` until resolved.

**Field-combo presets:**

| `before_done` | `always_escalates` | Behaviour | Use for |
|---|---|---|---|
| present | `false` | run action; escalate only on failure; else `done` | **auto-deploy** |
| present | `true` | run action; then escalate born-at-L2; `done` after `resume` | act-then-ask (incompatible with `human_curator_gate`) |
| absent | `true` | escalate born-at-L2 immediately; `done` after `resume` | **pure gate** |
| absent | `false` | **rejected** at `submit_task` (ill-formed no-op) | — |

**Validation (enforced at `submit_task`):** `task_kind='deterministic'`
with `before_done=None` and `always_escalates=false` is **rejected**
("ill-formed no-op"). `before_done` set on a `normal` task is also
**rejected** ("before_done is only valid on deterministic tasks"). A truthy
`human_curator_gate` together with a non-null `before_done` is likewise
**rejected** ("human_curator_gate is only valid on a pure gate") — see
[The human-curator-gate contract](#the-human-curator-gate-contract).

**Born-at-L2 escalations:** all filed with `severity ∈ {critical, urgent}`
and sentinel `agent_role='orchestrator-deterministic'`; the server retains
`level=2` (no L0→L1→L2 climb). The task goes `blocked` while the L2 is
open (quiescence guard — no re-dispatch, no churn).

**Blocking vs detached self-kill — *determined*, not a knob:**

- `before_done.target_unit` equals this orchestrator's own unit → detached
  `systemd-run --user` with `--on-failure` (done = `scheduled`; the
  dispatching orchestrator is **not** killed).
- `before_done.target_unit` differs from own unit (or is `None`) →
  blocking subprocess + fresh `MainPID`/`ActiveEnterTimestamp` verify
  against a pre-run baseline (done = `deployed-and-verified`).

**Runner stamps** (written by `DeterministicRunner`, never
author-supplied): `before_done_ran_at`, `before_done_verified_at`,
`gate_escalated_at`, and `done_provenance` (stamped for all four
`deterministic-*` kinds — see the requirement table in §2 for
per-kind semantics; `deterministic-milestone` is stamped on both the
first-pass check and the post-escalation re-check described in §6).

**`done_provenance.kind='operational-verified'`** — a related but distinct
closure path (see §2), used for `normal`-task no-code operational asks
(e.g. a restart/redeploy/confirm) closed out via a resolved escalation
rather than a `DeterministicRunner` action or a code merge.

**Dep convention:** deterministic deploys and gates use **normal**
dependencies — including cross-project `project_id:task_id` deps (§3.2).

---

## 6. Milestone tasks (dated / delayed)

`metadata.milestone` is an orthogonal, time-based dispatch gate — **not**
a new `task_kind`. Setting it holds a task out of dispatch until its time
trigger fires; once fired, the task dispatches through its normal path
unchanged. It is allowed on **both** `normal` and `deterministic` tasks
(orthogonal to `task_kind`), so it can gate a normal LLM-agent task, a
deterministic predicate check (below), or a pure human gate (a `dated`
milestone on a deterministic task with `always_escalates=True` and no
`before_done`).

Two time modes:

- **`dated`** — fires at an explicit wall-clock instant.
- **`delayed`** — fires `after_secs` seconds after the task's own
  dependencies (local deps **and** `metadata.external_deps`, §3) are
  satisfied.

**`metadata.milestone`** (set at `submit_task`; validated by the shared
`Milestone` model):

```
{
  mode: "dated" | "delayed",
  at: "<ISO-8601 datetime>",  # required iff mode="dated"; datetime.fromisoformat-parseable; forbidden iff delayed
  after_secs: <int>,          # required and > 0 iff mode="delayed"; forbidden iff dated
}
```

The `mode`-conditional fields are a strict *iff*: a `dated` milestone must
not also carry `after_secs`, and a `delayed` milestone must not also carry
`at`. This is enforced by the shared `Milestone` model
(`shared/src/shared/task_metadata.py`) and re-checked by the `submit_task`
guard — a malformed spec (`delayed` with no `after_secs`, `dated` with an
unparseable `at`) is rejected at submit with a structured `ValidationError`
and never persisted.

**Scheduler-stamped fields** (never author-supplied):
`milestone_deps_satisfied_at` (`delayed` mode only — the frozen-once
wall-clock UTC anchor, stamped the first tick all of the task's
dependencies go `done`) and, only on a predicate check failure,
`gate_escalated_at` (shared with §5).  A task author never sets either
field.

**Predicate exit-code contract (`before_done.kind`):** `before_done` gains
a discriminator, `'deploy' | 'predicate'` (default `'deploy'` — every
existing deterministic task is byte-identical). `kind='deploy'` is the
existing act-then-done/ask behaviour in §5; `kind='predicate'` is
check-then-done-or-escalate, and is only meaningful paired with a
milestone. In `kind='predicate'` mode the `DeterministicRunner` runs the
script and decides by **exit code only** — it parses no output:

| Exit code | Outcome |
|---|---|
| `0` | task `done`, `done_provenance.kind='deterministic-milestone'` (a **bounded structured verdict** carried as `note` — see below) |
| non-`0` | born-at-L2 `milestone_check_failed` escalation (detail carries the exit code + stdout tail) + task `blocked` |
| timeout | born-at-L2 `infra_issue` escalation (existing timeout path) + task `blocked`, **no** `gate_escalated_at` stamp |

**What the `rc == 0` `note` carries (task 3286):** `predicate check passed
(rc=0)`, plus — when the script emitted one — a single extracted payload:
either a **trailing JSON block** (re-dumped compactly) or the **last output
line if it is clean**, capped at 400 chars. Log-shaped lines (a leading
timestamp, or a standalone `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`
token) are dropped, and an over-cap payload is replaced wholesale by an
elision marker rather than sliced mid-structure. An unrecognized shape
yields the verdict prefix alone.

The two extraction tiers give **different guarantees**. The trailing-JSON
tier is a true allowlist — only a parseable payload survives, and every
preceding log line is excluded structurally. The last-clean-line tier is a
best-effort heuristic with a log-shape *denylist*, so a log line under a
bare `name message` formatter (no timestamp, no level token) can still
reach the note; the cap bounds that to one ≤400-char line, and the
recurrence guard shares the same blind spot.

**If you want your predicate's verdict preserved, emit it as a trailing
JSON object** (the only tier with a real guarantee) **or as one clean final
line.** The reason for the bound is that
`note` is not a private field — fused-memory's reconciliation
`_format_outcome_echo` appends it to a Mem0 completion-summary write, so
whatever lands there is ingested into memory. Task 2902 is the specimen: a
chatty script's server-log noise reached the knowledge graph this way.
Nothing is lost to debugging — the orchestrator logs the raw output in full
at INFO before summarizing, and the non-`0` row below still carries it
verbatim in the escalation detail (an escalation is read by a human, never
memory-ingested). `scripts/scan_provenance_note_log_leaks.py` is the
read-only recurrence guard.

A predicate is **read-only**, so resolving the escalation (`resume`)
safely **re-runs** the check rather than trusting the resolution blindly —
it drives to `done` or re-files `milestone_check_failed` from whatever the
check reports this time. `kind='predicate'` **forbids**
`before_done.target_unit` (no unit to verify — no systemd inspect /
PID-verify on a read-only check) and forbids top-level
`always_escalates=True` (predicate escalation is already conditional on a
non-zero exit); the `submit_task` guard enforces both. A predicate never
stamps `before_done_ran_at` / `before_done_verified_at` — re-running a
read-only check on crash-resume is harmless.

**Frozen-once delayed-anchor semantics:** for `mode='delayed'`, the
scheduler stamps `milestone_deps_satisfied_at` in a per-tick sweep the
first tick ALL of the task's dependencies (local and external) are `done`;
the timer then runs `after_secs` from that anchor. The anchor is a
**persisted wall-clock UTC ISO string**, not an in-memory/monotonic timer,
so a multi-day delay **survives orchestrator restarts** — it's recomputed
each tick from the persisted value, with no in-memory timer to lose. It is
frozen-once: a later dependency regression never rewrites the anchor or
restarts the timer. Dispatch still re-checks *live* deps at eligibility,
though — a milestone fires only when both the timer has elapsed **and**
deps are currently satisfied, so a regressed dependency still withholds
dispatch even after the timer elapses. A `delayed` milestone with no
dependencies has them trivially satisfied at filing, so its timer starts
immediately. `mode='dated'` simply withholds while `now < at`. A malformed
or unrecognized `metadata.milestone` value fails safe — withhold dispatch
indefinitely, with a one-time WARNING log — rather than dispatch early or
crash the tick.

**Exemplar** (autonomous — no human involvement unless the check fails):
"one week after task X lands, check merge flakiness is under 5%, else
escalate." A deterministic task depending on X, with:

```
{
  "milestone": {"mode": "delayed", "after_secs": 604800},  # 7 days
  "before_done": {
    "kind": "predicate",
    "script": "scripts/check_merge_flakiness.sh",
    "args": ["--window-days", "7", "--threshold", "0.05", "--value", "0.03"],
    "timeout_secs": 120
  },
  "always_escalates": false
}
```

(`--value` is normally populated by a wrapper that computes the measured
rate from CI logs — out of scope here; the script only owns the threshold
comparison, per its header.) The anchor stamps when X reaches `done`;
`after_secs` later the check runs. Exit `0` → `done`
(`deterministic-milestone`); non-zero → `milestone_check_failed` at L2
(same `agent_role` / severity / level as the born-at-L2 escalations in
§5) — predicate mode reuses the `DeterministicRunner` (§5), and the
delayed anchor waits on the same dependency gate described in §3.

---

## 7. Per-task model pins (`metadata.model_overrides`)

Set `metadata.model_overrides` on a task to pin specific agent roles to a
specific model for that task only — the highest-precedence layer in the
orchestrator's route resolver (see `ARCHITECTURE.md`'s routing section for
the full layered precedence and the operator-facing `routing.*` config
block).

**Shape:** an object mapping full role name → model string:

```
{
  "implementer": "opus",
  "reviewer_comprehensive": "haiku"
}
```

**Validation split** — submit-time SHAPE guard vs. resolve-time model
STRING check, because fused-memory does not know the orchestrator's
allowlist:

- `submit_task`/`update_task` shape-validate at write time via
  `shared.task_metadata.validate_model_overrides` against
  `KNOWN_ROLE_NAMES` — an unknown role name, or a non-string/empty-string
  model value, raises `ValidationError` and the write is rejected. The
  fused-memory `model_overrides_guard` mirrors this at both `submit_task`
  and `update_task`.
- The orchestrator resolver separately validates the model *string*
  against `routing.allowed_models` (and any configured per-model ceiling)
  at resolve time, fail-safe.

**Fail-safe semantics:** a well-formed override naming a model outside
`routing.allowed_models` (or past its configured daily ceiling) is skipped
at resolve time, recorded in `RoutingDecision.rejected`, WARN-logged, and
never blocks dispatch — the override is the resolver's highest-precedence
layer (`metadata_override`) and sets `model` only (never
effort/budget_usd/max_turns).

**Role-name caveat:** keys must be the full dispatch role name from
`orchestrator.agents.roles.ROLES` (e.g. `implementer`,
`reviewer_comprehensive`), not a collapsed config key. `reviewer`,
`triage`, and `module_tagger` are accepted by the shape guard
(`KNOWN_ROLE_NAMES` is a superset covering both `ROLES` and
`ModelsConfig`'s collapsed keys) but are accepted-but-**inert** as
override keys — the resolver's layer-1 reader keys strictly on the literal
`role_name` it was invoked with, never the collapsed config key, so an
override authored under one of these three collapsed keys silently never
matches at resolve time.

The full known-role set today (`shared.task_metadata.KNOWN_ROLE_NAMES`):

```
architect, implementer, debugger, reviewer, reviewer_comprehensive,
merger, steward, triage, module_tagger, deep_reviewer, judge, simple_task
```

Of these, `reviewer`, `triage`, and `module_tagger` are the accepted-but-
inert collapsed-config-key names — never use them as a `model_overrides`
key if you want the pin to actually take effect.

---

## 8. Task metadata vocabulary & census

`parse_metadata` (`shared/src/shared/task_metadata.py`) validates every
task's `metadata` blob on read and write and returns a list of
`SchemaWarning`s for anything it can't reconcile; the write-boundary
backend logs one WARNING line per warning via `_emit_schema_warning`
(`fused-memory/src/fused_memory/backends/sqlite_task_backend.py`). This is
the schema-drift census that enforce-gate runbooks grep for.

**Census line and the `code=` discriminator:**

```
task_metadata.schema_warning task_id=<id> code=<class> field=<key> error=<message>
```

The literal `task_metadata.schema_warning` token is the stable grep
anchor — never move or rename it. `code=` is the warning's class
discriminator (`unknown_key`, `invalid_field`, `invalid_submodel`,
`unparseable_json`, `not_an_object`, `invalid_metadata`). Separate
enforcement-relevant classes from routine vocabulary noise with:

```
grep 'task_metadata.schema_warning' | grep -v code=unknown_key
```

### Tier-A: blessed keys

A frozenset of load-bearing conventional metadata keys — already relied on
by real writers (orchestrator, curator, `DeterministicRunner`, escalation
flows) — is exempted from the `unknown_key` scan even though these are not
(yet) typed `TaskMetadata` fields. This is `_BLESSED_METADATA_KEYS` in
`shared/src/shared/task_metadata.py`:

```
source, modules, spawn_context, complexity, force_full_path,
branch_base_sha, _causation_id, dry_run_proposals, reblock_guard,
agent_id, escalation_id, suggestion_hash, prd_path, prd_task_label,
user_observable_signal, consumer_ref, substrate_confirmed,
human_decomposed, grammar_confirmed, invariants, optimistic_path,
capability_manifest, curator_action, curator_justification, combined_at,
gate_escalated_at, before_done_ran_at, before_done_verified_at,
before_done_verified_pid, files_tagged_at, origin_finding_id,
spawned_from, program, program_stream, stream, cross_repo,
cross_repo_project, human_curator_gate,
human_curator_adjudicated_at, last_blocked_at
```

`cross_repo` + `cross_repo_project` are the cross-repo deliverable marker
(§3.2.1): auto-set by the fused-memory submit path when a task's
`metadata.files` are all owned by one other registered project (and the
filer is itself registered), and read by **both** the orchestrator's
dispatch-time cross-repo admission gate (which blocks the task before any
agent spins up) and its pre-merge narrowing gate.

Two unrelated curators appear in this list, and the prefixes keep them
apart: `curator_action` / `curator_justification` / `combined_at` are
written by the **automated task curator**'s combine flow (fused-memory
`task_interceptor`), while the `human_curator_*` keys below belong to a
**human content curator** adjudicating a deterministic gate.

#### The human-curator-gate contract

`human_curator_gate` + `human_curator_adjudicated_at` mark, and then
discharge, a deterministic pure gate that only a human's **content
judgement** can close.

**`metadata.human_curator_gate`** (truthy) on a `task_kind='deterministic'`
pure gate (`before_done` absent, `always_escalates=true`) declares that the
gate asks for per-entry human review — not merely a decision. Resolving the
born-at-L2 `milestone_gate` escalation is **not by itself sufficient** to
close such a task: closing an escalation record and performing the review
are different propositions.

The marker belongs on a **pure** gate only. A `before_done` action is a
machine step that closes the task, which contradicts "only a human's content
judgement closes this" — a task carrying both takes the act-then-ask path
with the marker unread, and `DeterministicRunner` logs a warning naming the
authoring defect on every dispatch. Do not combine them.

The combination is **rejected at `submit_task`/`update_task`** by
`shared.task_metadata`'s cross-field validator whenever `task_metadata.enforce`
is true (its production setting), so the contradiction cannot land in the first
place. In warn-mode it degrades to a single `task_metadata.schema_warning`
census line (whole-blob field `<metadata>`, code `invalid_metadata`) and the
write proceeds. Note the write boundary treats that whole-blob field as
*always* fatal regardless of which keys the delta names — so an `update_task`
supplying only `human_curator_gate` is rejected even when `before_done` is an
untouched legacy field. The `DeterministicRunner` warning above is deliberately
**retained** as a defence-in-depth backstop for records that did not pass
through that boundary: `task_metadata.enforce` is a red-tier restart-only flag,
records predate the validator, and not every writer goes through
`SqliteTaskBackend`.

**Repairing a row that already carries both.** That same always-fatal rule cuts
the other way once a contradictory row exists — written in warn-mode, or by a
writer that bypassed `SqliteTaskBackend`. Every later `metadata_mode='merge'`
`update_task` on it is rejected, including writes with nothing to do with the
contradiction (`retry_ledger` updates, `files` edits), because the merged whole
is what gets validated. To unstick such a row, clear the contradiction *in the
same write*: merge an explicitly falsy `human_curator_gate` (a cleared marker no
longer contradicts `before_done`), or pass `metadata_mode='replace'` with a blob
that omits one of the two keys. No task on the live store is in this state today
— the four rows carrying the marker all have `before_done` absent — so this is a
forward hazard of a warn→enforce flip, not a live cleanup.

**`metadata.human_curator_adjudicated_at`** (ISO-8601 string) is the required
content-adjudication signal, stamped via `update_task` (with
`metadata_mode='merge'`) by whoever actually performed the review.

`DeterministicRunner`'s pure-gate resume enforces this as the second rung of
a two-rung proof ladder — rung one being the archive-inclusive escalation
record it already required. When the marker is set and the stamp is absent
or is not a non-empty string, the runner files a born-at-L2
`curator_adjudication_missing` escalation naming the remediation and leaves
the task **blocked** rather than driving it to `done`. Both checks fail
closed: a truthy-but-not-`true` marker still trips the guard, and a
`bool`/`int`/blank stamp is not accepted as proof. A curator gate that does
close carries a `done_provenance.note` naming the adjudication stamp
(truncated if oversized — the note is memory-ingested downstream) and a
`done_provenance.escalation_id` naming the `milestone_gate` record that
proved rung one — **not** the `curator_adjudication_missing` re-ask, which
proves nothing about the gate. So a genuine closure is distinguishable from
a phantom one in the audit trail.

The originating incident is task 3181, whose gate escalation `esc-3181-1`
was resolved by the automated `escalation-watcher` — with the resolution's
own text stating the curator content work was deliberately not executed —
after which the resume path nonetheless closed the task.

### Tier-B: canonical keys, not aliases

These aliases are deliberately *not* on the Tier-A allowlist, so each still
emits `code=unknown_key` as a greppable drift signal until the caller is
fixed to use the canonical spelling:

| Canonical | Aliases to avoid |
|---|---|
| `prd_path` + `prd_task_label` | `prd`, `prd_ref`, `prd_leaf` |
| `invariants` | `inv` |
| `related_tasks` | `related_task`, `related_df_tasks`, `related_task_examples` |

### Tier-C: ad-hoc keys

One-off, timestamped, or id-suffixed annotation keys (e.g.
`reconciliation_with_5123`) must never be filed as a bespoke top-level
metadata key — that just adds another `code=unknown_key` census line. Use
the `x_`-prefixed forward-compat namespace instead (e.g.
`x_reconciliation_note`) — silently allowed, no warning — or fold the
value into a single `annotations` field.

### `allow_mcp_markup`: a write-time flag, not a metadata key

`metadata={'allow_mcp_markup': True}` is the sanctioned move for a write
that **deliberately quotes the MCP envelope literals** — documenting the
leak, pasting a specimen, quoting a rejection's `matched_pattern`. It is
honoured at all four guarded write boundaries: `submit_task`,
`update_task`, `add_memory`, `add_episode`.

**Why it belongs in this section.** The convention that grew up instead —
paraphrase the literals, then park the evidence under a bespoke
timestamped metadata key such as `markup_tripwire_rejections_<date>` —
manufactures *both* failure classes at once: a `code=unknown_key` census
line (Tier-C, above) for every such key, and a bounced write for every
author who quotes the literals without the flag. It was self-perpetuating
because it was documented inside task 3083's own `details`, so each reader
learned the workaround rather than the flag. Task 3697 retires it.

**Scope of the gate.** The guard reads only the caller's *text* arguments,
never the metadata blob: `title` / `description` / `details` / `prompt` on
`submit_task` and `update_task`, and `content` on `add_memory` and
`add_episode`. Metadata is handed to the guard for exactly one purpose —
reading this flag. So it is a description that trips the tripwire, not a
metadata *value* that happens to contain the literals. `update_task`'s own
docstring states the same contract
(`fused-memory/src/fused_memory/server/tools.py`).

**Fail-closed: only a literal boolean `True`.** `'yes'`, `1` and `'true'`
do not enable it — `markup_override_requested` tests
`parsed.get(...) is True` (`markup_tripwire.py`). That strictness is the
containment argument, not pedantry: the failure being contained is an
accidental harness serialization leak, and an accidental leak never sets
an explicit flag. A deliberate author can.

**Write-time-only — it never persists.** `strip_markup_override`
(`markup_tripwire.py`) removes the key at every one of the four
boundaries before the metadata reaches storage, so the flag never lands in
stored metadata and never mints an `unknown_key` census line of its own.
It is non-mutating (the caller's own dict is left intact) and honours both
accepted metadata shapes — dict in / dict out, JSON string in / JSON
string out. Pinned by
`fused-memory/tests/server/test_markup_tripwire_gate.py::test_override_flag_is_stripped_before_persistence`
and `::test_override_flag_is_not_persisted_into_task_metadata`.

**This is routine, not exotic.** The 2026-08-05 decompose session that
filed the toolcall-markup-containment batch had its *first* `submit_task`
rejected by this tripwire for quoting the literals, and needed the flag on
seven of the nine tasks it filed. The guard was working as designed; the
missing piece was the convention around it.

**What it is not: an escape hatch for an accidental leak.** If you did not
mean to quote the markup, strip the fragment and resubmit — do not reword
the payload to sneak it past the guard, and report a recurrence per the
hint the rejection itself carries. Using the flag to push an accidental
leak through the boundary defeats the containment it exists to provide.

The authoritative enumeration of the literals lives in exactly one place,
`fused-memory/src/fused_memory/server/markup_tripwire.py`, and is
deliberately **not** repeated here: restating them in this file would
oblige every future task write quoting this section to set the override,
which is precisely the loop this section exists to break. For the full
picture see `docs/mcp-toolcall-xml-leak.md` §4, "The boundary rejection".

### Promoting a convention

A key only stops warning once it's added to the `_BLESSED_METADATA_KEYS`
frozenset in `shared/src/shared/task_metadata.py` (the Tier-A load-bearing
allowlist — an allowlist rather than typed optional fields, so
`model_dump()` doesn't grow `None`-valued noise on every task). Add a key
there only for a genuinely load-bearing, stable convention; Tier-B/C drift
should be fixed by renaming to the canonical key or moving under `x_`, not
by blessing it.

### Known gaps (measured 2026-08-06 — not fixed)

Three `unknown_key` sources are known, measured, and deliberately left
open. They are recorded here so the next reader does not re-measure them.
All counts are a snapshot of a **growing** corpus (3553 tasks carried dict
metadata at measurement), not an invariant.

| Gap | Measured | Owner |
|---|---|---|
| `execution_class` is read by two live guards but is neither blessed nor typed | 272 tasks | `tkt_0RS4XDWJQ9PR8MFXY5DKW950WS` |
| Ad-hoc reify/escalation keys unmigrated corpus-wide | `origin_escalation` 19, `related_reify_tasks` 8, `origin_reify_task` 4, `related_reify_memories` 1 | `tkt_0RS4XDWJQ9PR8MFXY5DKW950WS` |
| Task 3083 still emits 6 `unknown_key` lines — the write path is blocked | 6 of an original 7 | `tkt_0RS4WVMH1RSTSY88N781E70F5S` |

**`execution_class`** is not in `_BLESSED_METADATA_KEYS`, is not a typed
`TaskMetadata` field and is not a registered submodel — yet
`execution_class_guard` and `routing_intent_guard` both read it, so every
one of those 272 tasks plausibly emits this same warning class. It was
deliberately **not** blessed by task 3697: 272 tasks and two live guards
make it a broader vocabulary decision (bless / promote to a typed field /
retire) than a single-task cleanup should settle. Originally recorded as
Finding 5 of the toolcall-markup-containment capability manifest. The
count is still climbing — 253 at that PRD's decompose, 272 here.

**The `x_` sweep** was scoped to task 3083 alone, not the corpus, because
a ~30-task metadata rewrite has a very different blast radius from one
reserved task. `x_`-prefixed precedents for these same spellings already
exist (`x_origin_escalation` 1, `x_related_reify_tasks` 1,
`x_related_df_tasks` 5), so the target spelling is not in doubt and the
sweep is a mechanical per-task re-run of
`fused-memory/scripts/migrate_task_metadata_to_x_namespace.py`. That
script's "no reader anywhere" grep argument covers only its six built-in
default keys, so when you re-run it with your own `--keys` it validates
them first and refuses an already-`x_`-prefixed key, a typed
`TaskMetadata` field or a Tier-A blessed key — its read-back proves the
rename *landed*, never that the rename was *safe*.

**The write-path blocker** is why the third row is still open, and it
bounds both of the others: `update_task` rejects any metadata payload
containing `done_provenance` — a presence-only write-authority floor
evaluated *before* `metadata_mode` is resolved — and `'merge'` mode cannot
retire a key at all, since `_merge_metadata` is a shallow `{**old, **new}`
with no deletion sentinel. A whole-blob `'replace'` is therefore
structurally impossible on any `done`/merged task, which is most of the
corpus above. Check a target task's status before assuming its metadata is
writable.

---

## 9. Practical recipes

**Wire cross-project deps via `planning_mode`, not `submit_task` +
`add_dependency` back to back.** A plain `submit_task` followed
immediately by `add_dependency` races the scheduler — the task can be
picked up for dispatch in the window between the two calls, before the
dependency is attached. Use `submit_task(planning_mode=True)` →
`add_dependency` → `commit_planning` instead: the task sits in `deferred`
(unreachable by the scheduler) until every dependency is wired, and only
then is it committed to `pending` in one atomic flip.

**Verify a batch is filed with `get_task`, not `search_tasks` /
grep-equivalents.** `search_tasks` queries the curator's semantic-search
corpus, which is only populated by tasks the curator has actually indexed
— `deferred` and still-uncommitted `planning_mode` tasks are excluded, so
a batch you just planned will not show up in `search_tasks` results even
though it exists. Confirm a specific id landed with `get_task(id=...)`
directly.

**Change a filed task's kind in place with `update_task`, not
remove+refile.** If a task's `task_kind`, `before_done`, or other metadata
needs correcting after filing, update it in place via `update_task` rather
than removing and resubmitting — resubmission loses the task's id (and
anything depending on that id), history, and any escalations already
attached to it. To retire a task instead of correcting it, cancel it
(`status="cancelled"`) rather than removing it, so the record and its id
remain resolvable by anything that referenced it.

---

## Cross-references

- **`README.md`** — repo orientation and top-level entry points.
- **`OPERATIONS.md`** — day-to-day operator workflows: running the
  orchestrator, resolving escalations, config reload, fleet redeploy.
- **`ARCHITECTURE.md`** — the full task status state machine, the
  scheduler's per-tick phase pipeline, agent roles, the merge lane, and
  model routing's layered precedence.
- **`skills/prd/SKILL.md`** — the PRD authoring and decomposition pipeline;
  the primary producer of `planning_mode` batch submissions (§1, §9) and
  of `metadata.delivered_checks`-bearing capability tasks (§3.3).
- **`docs/legibility/design-invariants.md`** — the design invariants gating
  `/prd` decompose and `/review` phase 2.
