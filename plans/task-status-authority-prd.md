# PRD — Task-status authority (bug-hotspot remediation stream W2)

**Status:** active — authored 2026-07-06. Program: `plans/bug-hotspot-remediation-program-2026-07-06.md`
stream **W2** (wave 1). Findings: `plans/bug-hotspot-survey-2026-07-06-full-findings.json`
(scheduler 4.1, fm-task-layer 6.4, escalation 10.0-10.5, workflow 1.2).
**Approach:** **B + H** (design-first; one contract + two-way boundary tests) — HIGH-STAKES: this
installs the transition authority on the single durable task-status write chokepoint of a **running**
factory. A bad table can brick dispatch fleet-wide, so the contract is decided here and enforcement is
gated behind a log-mode soak.
**Supersedes/completes:** `plans/escalation-repend-state-machine-prd.md` (D1-D11, landed via tasks
**1617-1626, 1622**) and `plans/escalation-connection-capability-guard-prd.md` (landed via **2041-2043**).
Those PRDs built the *mechanism* (action enum, harness dispatch, gates, stranded sweep, reblock guard,
header capability gate); they left the *authority* diffuse — two files re-derive the same enum's effect,
the store enforces no vocabulary, the level gate is a caller-declared header opt-in. Drift followed
(task 1883 bolted an `infra_hold` branch onto `_cascade_unblock_member` outside the design), exactly the
class those PRDs meant to close. W2 installs the single authority.

## Goal (user-observable behaviour)

One authority for every task-status transition and every escalation-resolution action:

1. **Illegal status writes are rejected loudly at the store.** A typo'd or novel status
   (`in_progress` vs `in-progress`) — today persisted silently and then dropped from the active fetch,
   the worst failure shape in this system — returns a typed `TaskmasterError` from fused-memory instead
   of stranding the task.
2. **Task-status legality is one table, enforced at the fused-memory TaskInterceptor.** The
   `(from, to, actor_class) → allowed` table lives in `shared/`; the interceptor (which already gates
   terminal-exit / done-provenance) consults it. Off by default (log-only) at first; enforcement flips
   after a production soak proves zero false rejections.
3. **"Stranded" is a queryable predicate, not `plan.lock`/owner_pid forensics.** Tasks carry
   first-class `claimant_run_id` + `heartbeat_at`, stamped at dispatch, cleared at release. W10's
   `TaskGroundTruth` reads these instead of grepping `plan.lock`.
4. **Escalation-resolution intent is one `(action, level, category) → TaskEffect` table.** Imported by
   BOTH `resolve_issue` (rejects illegal combos loudly — no more DEBUG-level silent no-op) and the
   harness `_on_escalation_resolved` callback (computes the effect, replacing the ad hoc if/elif chain +
   the `_ACTION_TARGETS` dict). `infra_hold` gets a first-class status instead of overloading
   `in-progress` across two hand-synced sites.
5. **`resume` on a task with no live claimant re-pends** (the esc-2073-15 class), enforced via the
   table + the claimant predicate, not by the brittle `status == 'blocked'` precondition.
6. **Level ceilings are server-derived from caller identity**, not a caller-declared header opt-in.
   Identified automation gets a role-mapped ceiling; **header-less sessions remain the full-capability
   human channel** (a naive default-deny would repeat the esc-2087-2 lockout). `promote_to_l2`'s create
   side is gated the same way.
7. **Minor authority leaks closed:** escalation IDs come from one authoritative counter, not a
   directory re-scan; the `datetime.UTC` flip-flop is settled by a convention note.

## Consumers (G1) — every mechanism has a named consumer

| Mechanism introduced | Consumers |
|---|---|
| `shared/task_statuses.py` StrEnum + derived frozensets | fused-memory (interceptor, backend, task_filter), orchestrator (task_status.py, scheduler), **W3** (metadata schema references status as a column) |
| `shared` `(from,to,actor_class)` transition table + `is_legal_transition` + `outcome_allows_status` | fused-memory TaskInterceptor (enforcement floor); **W9** `WorkflowStateMachine` (imports it as its outcome→status validator — brief: "you own the TABLE, W9 owns the client-side machine") |
| `claimant_run_id` / `heartbeat_at` columns + `is_stranded()` | orchestrator dispatch/release wiring; **W10** `TaskGroundTruth` resolver + the harness reconcile sweeps (which W10 makes thin) |
| escalation `(action,level,category)→TaskEffect` table | `resolve_issue` (server); harness `_on_escalation_resolved` |
| `infra-hold` first-class status | `_cascade_unblock_member` (write site) + `_revert_in_progress_if_no_live_claimant` (skip site) + workflow verify-infra stamp — one accessor, three consumers |
| server-side `ROLE_LEVEL_ALLOWLIST` + `PROMOTE_ALLOWED` | `resolve_issue` + `promote_to_l2`; the orchestrator-spawned auto-watcher connection (unchanged behaviour) |
| escalation ID counter | `queue.make_id()` / `get_by_task()` |

User-observable surface: every row of the §Boundary-test sketch is a signal asserted through the
product's own read paths (task status via fused-memory, escalation state via the escalation APIs,
workflow outcomes via harness tests).

## Background — what landed, why the defects persist

The escalation-repend PRD (2026-06) and the capability-guard PRD (2026-07-03) both landed and are green
on main. The 2026-07-06 survey re-verified the code and still confirms the defects, because the prior
work built mechanism without a single authority:

- **10.0 (confirmed, high):** `resolve_issue` (server.py:506-629) validates `action` as a bare string;
  `queue.resolve()/park()` see only `status`. The *effect* on the task is re-derived independently in
  the harness (`_on_escalation_resolved` :8189-8316, `_ACTION_TARGETS` :8300, `_cascade_unblock_member`
  :8659-8756, `_action_teardown_and_set_status` :8318-8468). Any precondition miss is a DEBUG-level
  silent no-op. Task 1883's `infra_hold` branch was added here with no shared table → drift.
- **4.1 (confirmed, high) / 6.4 (weakened, medium):** `task_status.py` is three frozensets; the store
  column is `TEXT NOT NULL` with no CHECK (sqlite_task_backend.py:62) and `set_task_status` writes any
  string verbatim (:658-692). The legal vocabulary is declared **four** times (orchestrator
  task_status.py:11-44 w/ an explicit "DRIFT RISK" comment; task_filter.py:217; task_interceptor.py:126
  w/ "duplication is cheaper than cross-package coupling"; `_VALID_TASK_STATUSES` composed at
  tools.py:636). No `(from,to,actor)` legality table anywhere. Dead-end states are recovered by
  heuristic compensation sweeps in another subsystem (harness.py:2795+). *(6.4 weakened → the store
  already has partial floors — the `update_task(status=)` reject at three layers incl. the backend floor,
  task 1664 — but the two live gaps stand: no vocabulary CHECK on `set_task_status`/`add_task`, and no
  transition table. W2 targets exactly those two, not the already-closed `update_task` bypass.)*
- **10.1 (confirmed, high):** the level gate (server.py:565-592) fires only when the caller sends
  `X-Escalation-Levels`; default-open for header-less/stdio. The only caller that sends it is the
  orchestrator's own auto-watcher (`_WATCHER_ESCALATION_HEADERS`, harness.py:270-273). Restriction is
  self-declared, not identity-derived.
- **10.2 (confirmed, medium):** `promote_to_l2` (server.py:683-807) has **no** role/capability check on
  the create side — `agent_role` is a free-form param stored verbatim.
- **10.3 (confirmed, medium):** `in-progress` is overloaded — `_cascade_unblock_member` writes it for an
  `infra_hold` task (harness.py:8704) and `_revert_in_progress_if_no_live_claimant` re-checks the same
  `metadata.infra_hold` flag to skip reverting (harness.py:3462). Two hand-written flag checks in two
  functions, no shared type.
- **10.4 (confirmed, medium):** `make_id` (queue.py:1001-1039) re-derives the next sequence by globbing
  the queue dir + a single-writer-only `_archive_max_seq_cache`; densest fix-history in the mined themes.
- **10.5 (confirmed, low):** `datetime.UTC`↔`timezone.utc` flip-flopped 6× on a false "<3.11 compat"
  premise; `escalation/pyproject.toml` pins `>=3.11,<4` and selects ruff `UP` → the rationale was always
  false. Code is already on `datetime.UTC`; the fix is procedural.
- **1.2 (confirmed, medium) — workflow, W2 owns the TABLE only:** terminal state is quadruplicated
  (`WorkflowState`, `WorkflowOutcome`, the status row, metadata stamps). W2 provides the shared
  transition table + `outcome_allows_status` consistency map; **W9** builds the `WorkflowStateMachine`
  value object on top. W2 does not touch workflow.py's `_last_block_*` side channel (W9's territory).

## Substrate reality check (G3) — all anchors re-verified against the working tree 2026-07-06

| Assumed capability | Status | Evidence |
|---|---|---|
| `shared/` is importable by BOTH orchestrator and fused-memory with no new wiring | **verified** | `dark-factory-shared` is a `{workspace=true}` dep of both pyprojects; ~34/~18 `from shared` import sites. New `shared/src/shared/task_statuses.py` needs zero wiring. All three pin `>=3.11,<4` → `StrEnum` OK. |
| Single durable status write chokepoint exists | **verified** | `task_interceptor._apply_status_transition` (:619-857) is the only path; backend `update_task` hard-rejects `status`, MCP `update_task` rejects it, `commit_planning` routes through the interceptor. |
| Interceptor can attribute a write to an actor | **absent — must be built (additive)** | `_apply_status_transition` receives no `agent_id`; task tools take no `ctx` (memory tools do, via `_resolve_identity` tools.py:409-441). Threading `ctx→agent_id→actor_class` is task ρ1b. UNKNOWN actor fails **safe-open** (see D5). |
| tasks.db can gain columns | **verified** | migration mechanism exists (`_SCHEMA_VERSION`/`_migrate`/`PRAGMA user_version`, sqlite_task_backend.py:39/165-238); bump to v2 + ALTER is the established idiom. |
| escalation `Escalation` has `resolution_action`/`level`/`category`/`agent_role` | **verified** | models.py:53/55/66/86. `BORN_AT_L2_SEVERITIES` = `frozenset({'critical','urgent'})` models.py:41. |
| `resolve_issue` reads `X-Escalation-Levels`+`X-Escalation-Identity`; identity overrides `resolved_by` | **verified** | server.py:565-598. Role→ceiling map (ε2) reuses this identity channel — no new transport. |
| `queue.make_id`/`get_by_task` derive by directory scan; `escalation_id_lock` sidecar exists | **verified** | queue.py:1001-1039 / :309-390 / :24-69. The counter (ε3) fsyncs under the existing lock. |
| escalation `CATEGORIES` list is inert (no validation) | **verified** | server.py:134-151; grep shows no `category not in CATEGORIES` check. `stranded_blocked`/`infra_hold` category strings pass through unvalidated → adding a status value is the vocabulary change, not a category one. |

No unbacked numeric premise. The two premises that *could* be wrong — "the encoded transition set covers
all live transitions" (τ2) and "the illegal `(action,level,category)` combos are unused" (ε1) — are
de-risked empirically: τ2 by the log-mode soak (Γ) before enforcement; ε1 by validating against the
`data/escalations` archive (G6, ε1 manifest binding).

## Resolved design decisions

**D1 — two tables, two homes, one authority each.**
- **Table A (task lifecycle):** `shared/src/shared/task_statuses.py` — `StrEnum TaskStatus`, the derived
  frozensets (`TERMINAL`/`ACTIVE`/`WORKFLOW_PRESERVE`/`STATUS_TRIGGERS`), `ActorClass`,
  `derive_actor_class(agent_id)`, and `TRANSITIONS: dict[(TaskStatus, TaskStatus, ActorClass), bool]`
  with `is_legal_transition()` + `outcome_allows_status()`. Home = `shared/` because fused-memory
  enforces it and must not depend on orchestrator/escalation (program decision #4: `shared/` is the
  sanctioned home for cross-process contracts).
- **Table B (escalation intent):** `escalation/src/escalation/action_effects.py` —
  `ACTION_EFFECTS: dict[(action, level_class, category), TaskEffect]` where `TaskEffect` names the target
  status + live-workflow disposition. Home = `escalation/` (only the server and harness use it; both
  already import `escalation.*`). Table B's `TaskEffect.target` is a `TaskStatus` from Table A.

  These compose: Table B computes the intended `TaskEffect`; the actual write goes through the
  fused-memory chokepoint where Table A validates its legality. **Never three tables** (program G4).

**D2 — Table B encodes CURRENT semantics, behaviour-preserving.** `resume→pending`,
`restart→pending`, `park→blocked` **+ keep an open L2** (task 1792's version-a — NOT the repend PRD's
stale `deferred`), `abandon→cancelled`, `close_only→no-op`. The only new behaviour is that combos the
harness previously **silently no-op'd** now return a typed rejection from `resolve_issue`. Validated: no
combo marked illegal appears in the `data/escalations` archive (ε1 G6 step).

**D3 — `infra_hold` becomes a first-class status `infra-hold`.** A distinct, non-terminal, ACTIVE (but
non-dispatchable) status replaces the `metadata.infra_hold`-boolean overload of `in-progress`. Both the
write site (harness.py:8704) and the reconcile skip site (harness.py:3462) key on the status via one
accessor `is_infra_held(task)`; the workflow verify-infra path (workflow.py:4842) writes the status
instead of the metadata flag. This is the "first-class representation" the brief mandates. *(Alternative
— keep the flag but read it through a single typed accessor — is recorded under Open questions; a
distinct status is chosen because it makes the exemption visible to every status consumer, not just the
two that remember to check the flag.)*

**D4 — claimant/heartbeat are COLUMNS, not metadata.** `claimant_run_id TEXT` + `heartbeat_at TEXT`
(ISO-8601) added to `tasks` via a v2 migration. Columns (parallel to `status`), not
`metadata.*` — keeps the W3 boundary clean (W3 owns metadata; status/claimant are columns). Stamped at
the pending→in-progress dispatch write (workflow.py:1510 region, carrying the workflow's `session_id` +
`owner_pid` + process `run_id`), refreshed on a lightweight heartbeat, set NULL at release
(scheduler.release / `_run_slot` finally). `is_stranded(task, now, ttl)` = `status == 'in-progress'` AND
(`claimant_run_id IS NULL` OR `heartbeat_at` older than ttl) AND NOT `is_infra_held`. **Claimant writes
must fail-safe if the column is absent** (a routine orchestrator restart ahead of the fm deploy must not
error) — feature-detect / guarded write.

**D5 — actor_class is derived, coarse, and fails safe-open.** `derive_actor_class(agent_id)` maps the
existing `agent_id`/client-name prefixes: `recon-stage-*→RECONCILIATION`,
`orchestrator-deterministic→DETERMINISTIC`, `orchestrator*/harness*/steward*→ORCHESTRATOR`,
`escalation*→ESCALATION`, `None`/other→`HUMAN`. The table only ever *adds* restrictions for a **known**
actor (e.g. RECONCILIATION may not revert a claimed in-progress task — the task-1655 defect); an UNKNOWN
actor is permitted the **union** of all actors' transitions. So an unattributed write can never be
bricked by the actor dimension — it can only be rejected for a truly-illegal `(from,to)` (or an unknown
status). Precise actor granularity is validated during the soak (Open questions).

**D6 — enforcement is log-first, flipped behind a soak gate.** The interceptor consults Table A gated on
a fused-memory config `task_status.enforce_transitions` (default **false** = log-only: emit a WARNING
`illegal_transition would-reject <from>→<to> actor=<x>`, then write). Vocabulary rejection
(status ∉ enum) is **not** gated — a novel status is unambiguously a bug and rejects immediately. The
transition-legality flip to reject happens only after a production soak (Γ) confirms zero would-reject
WARNINGs. This is the "additive + logging mode first, enforcement flip as its own gated task" the brief
requires — a bad table cannot brick dispatch before a human has seen the soak.

**D7 — level authority is identity-derived server-side; header-less stays full.** Add
`ROLE_LEVEL_ALLOWLIST: dict[identity, frozenset[int]]` in escalation (e.g.
`{'orchestrator-escalation-watcher-auto': {0,1}}`). In `resolve_issue`: if `X-Escalation-Identity` maps
to a role, the **role ceiling is authoritative** (a present `X-Escalation-Levels` may only *narrow*
within it, never widen); if identity is absent (header-less) → **full authority, unchanged** (the
esc-2087-2 human-channel guarantee); if an identified caller is unmapped → the existing header-opt-in
behaviour (2041) is preserved as a fallback. `promote_to_l2` gets `PROMOTE_ALLOWED` (identities allowed
to mint L2): identity present ∧ ∉ set → `level_forbidden`; identity absent → allowed. The deployed
auto-watcher (identity + levels `{0,1}` + promote-allowed) is a **no-op** under this — same ceiling,
now identity-derived; the new enforcement catches *other* identified callers and a watcher that drops
its levels header. This supersedes the capability-guard PRD's decision #1 (which barred an in-server
identity→policy map on separation-of-concerns grounds): the map is **data**, not an
`if resolved_by == watcher` fork, and header-less remains open — per program resolved decision #3.

**D8 — deploy is a human operator gate, not an automated `before_done` script.** This deploy restarts
fused-memory itself (severs the runner's own MCP connection — task 2066 class) **and** orchestrators
(self-kill — task 2004/2105 class) **and** the escalation servers, fleet-wide, changing live dispatch
semantics — the highest-risk deploy shape, where unattended automation has repeatedly broken. A
`before_done` referencing a not-yet-existing multi-service restart script would also be rejected at
`submit_task`. So Δ1/Γ are **deterministic pure gates** (`task_kind='deterministic'`,
`always_escalates=true`, no `before_done`) carrying a precise out-of-cgroup restart runbook (fused-memory
via `systemctl --user restart fused-memory.service`, **never** `--drain` — program decision #6). An
automated deterministic deploy is deferred until a vetted `restart-all` script exists (Open questions).

## Contract (B+H §1)

### C1 — Table A: `shared/task_statuses.py`
- `class TaskStatus(StrEnum)` enumerating exactly the union of the four current copies:
  `pending, in-progress, blocked, deferred, review, merge-deferred, infra-hold (new, D3), done, cancelled`.
- Derived: `TERMINAL = {done, cancelled}`, `ACTIVE = {pending, in-progress, blocked, deferred, review,
  merge-deferred, infra-hold}`, `WORKFLOW_PRESERVE = {done, cancelled, deferred, blocked, merge-deferred}`,
  `STATUS_TRIGGERS = {done, blocked, cancelled, deferred}`. `infra-hold` ∈ ACTIVE, ∉ dispatchable
  (scheduler dispatches only `pending`), ∉ TERMINAL.
- `class ActorClass(StrEnum)`: `ORCHESTRATOR, RECONCILIATION, ESCALATION, DETERMINISTIC, HUMAN`.
- `derive_actor_class(agent_id: str | None) -> ActorClass` (D5 prefix rules).
- `TRANSITIONS` legal set + `is_legal_transition(frm, to, actor) -> bool` (UNKNOWN/HUMAN = union;
  same-status no-op is always legal; terminal→non-terminal illegal for every actor except a
  reopen-carrying write, matching the existing terminal-exit gate).
- `outcome_allows_status(outcome, status) -> bool` — the one map W9's `WorkflowStateMachine` consults
  (WorkflowOutcome → allowed final statuses).
- The three orchestrator/interceptor/task_filter copies are **deleted** and re-imported from here; the
  hardcoded-mirror CI test (test_scheduler.py:8992) becomes a real `from shared…` import equality (or is
  removed as tautological).

### C2 — store-level vocabulary rejection (fused-memory)
`SqliteTaskBackend.set_task_status` / `add_task` reject `status ∉ TaskStatus` with a loud
`TaskmasterError` (optionally a CHECK at the next schema bump). Immediate, not gated (D6).

### C3 — Table A enforcement at the interceptor (fused-memory)
`_apply_status_transition` (:619-857), inside the existing gauntlet (after terminal-exit/done-provenance,
around the write at :806): resolve `actor_class` from the threaded `agent_id`; if
`not is_legal_transition(old, new, actor)` → **if `enforce_transitions`**: return a typed
`illegal_transition` rejection; **else**: emit the would-reject WARNING and proceed (D6). `agent_id` is
threaded `ctx → _resolve_identity → set_task_status → _apply_status_transition` (additive; absent ctx →
HUMAN, safe-open).

### C4 — claimant/heartbeat columns (fused-memory + orchestrator)
Schema v2 (`claimant_run_id`, `heartbeat_at`); backend accessors; `is_stranded()` in `shared`
(D4). Orchestrator stamps at dispatch, refreshes, clears at release; writes fail-safe if the column is
absent.

### C5 — Table B: `escalation/action_effects.py` + `resolve_issue`
`ACTION_EFFECTS[(action, level_class, category)] → TaskEffect(target_status, workflow_disposition)`
encoding D2. `resolve_issue` looks it up **before** any mutation; an absent/illegal key → typed
`{'error':…, 'code':'illegal_transition'}`, no record change (parallels the existing capability-gate
early-return). The docstring reproduces the table. Validated against `data/escalations` archive (G6).

### C6 — harness consumes Table B
`_on_escalation_resolved` / `_resolve_escalation_action` / `_cascade_unblock_member` /
`_action_teardown_and_set_status` compute the effect from `ACTION_EFFECTS` (delete the local
`_ACTION_TARGETS` dict + the if/elif chain). Behaviour-preserving by construction; the C3 ordering
invariants (status-write-precedes-kill; teardown-suppression) are retained.

### C7 — `infra-hold` first-class (orchestrator, D3)
`is_infra_held(task) -> bool` (single accessor); the write site, the reconcile skip site, and the
workflow stamp all route through it / the status. `resume` on an infra-held task → `infra-hold`
(resume-at-verify); the reconcile sweep leaves `infra-hold` untouched.

### C8 — level authority (escalation, D7)
`ROLE_LEVEL_ALLOWLIST` + `PROMOTE_ALLOWED`; `resolve_issue` derives the ceiling from
`X-Escalation-Identity` (role ceiling authoritative, header narrows within, header-less = full);
`promote_to_l2` gated by `PROMOTE_ALLOWED` (header-less = allowed).

### C9 — escalation ID counter (escalation, 10.4)
One durable per-`task_id` counter file, fsync-incremented under `escalation_id_lock`, is the sole source
of the next sequence in `make_id`; the archive-scan + `_archive_max_seq_cache` (task 1879) is retired.
`get_by_task` correctness follows from the counter.

### C10 — datetime convention (escalation, 10.5)
Code stays on `datetime.UTC` (already true). Add a one-line convention note (escalation package
`AGENTS.md`/`CLAUDE.md` or the verify skill): fix-agents trust `ruff` UP017 over ad hoc `<3.11 compat`
claims; grep `requires-python` before asserting a version rationale.

## Boundary-test sketch (B+H §2) — the ζ integration-gate signal

Each row faces **both** sides of a seam; postconditions assert through product read paths.

| # | Scenario | Pre | Post |
|---|---|---|---|
| A1 | Unknown status rejected at store | `set_task_status('in_progress')` | typed `TaskmasterError`; row unchanged; task still fetched/dispatchable |
| A2 | Legal transition passes (log & enforce) | `pending→in-progress`, ORCHESTRATOR actor | accepted both modes; no WARNING |
| A3 | Illegal transition: log-mode warns, enforce-mode rejects | e.g. RECONCILIATION `in-progress(claimed)→pending` | log-mode: WARNING + write proceeds; enforce-mode: typed reject, no write |
| A4 | UNKNOWN actor never bricked | header-less/no-agent-id status write of any legal `(from,to)` | accepted in enforce mode (D5 safe-open) |
| A5 | Claimant stamped at dispatch, cleared at release | dispatch then terminal | `claimant_run_id`/`heartbeat_at` set at in-progress, NULL after release |
| A6 | `is_stranded` true iff no live claimant | in-progress, `heartbeat_at` older than ttl, not infra-held | predicate true; W10-style query finds it |
| B1 | `resume` re-pends a memberless born-at-L2 (esc-2073-15 class) | blocked task, level-2 record resolved `resume` | task `pending` |
| B2 | `resume` on stranded in-progress w/ no live claimant | in-progress, no claimant | `pending` (was: strands as in-progress) |
| B3 | `park` keeps L2 open, status blocked (task 1792) | live task, `park` | status `blocked`; L2 kept open; not `deferred` |
| B4 | `abandon`→cancelled, `close_only`→no-op | resp. | matches Table B |
| B5 | illegal `(action,level,category)` combo rejected loudly | a combo absent from ACTION_EFFECTS | `resolve_issue` returns `illegal_transition`; record unchanged (was: silent DEBUG no-op) |
| B6 | `infra-hold` first-class | verify-infra window exhausted | status `infra-hold`; reconcile sweep skips it; `resume`→`infra-hold` |
| C1 | header-less human keeps full L2 authority | no identity header; L2 record | `resolve_issue(park|close_only|resume)` succeeds |
| C2 | identified auto-watcher ceiling enforced server-side | identity `…watcher-auto`, no levels header; L2 record | `level_forbidden` (ceiling `{0,1}` derived from identity) |
| C3 | header levels narrow but cannot widen | identity ceiling `{0,1}` + header `0,1,2` | still denied L2 |
| C4 | `promote_to_l2` gated | identity ∉ PROMOTE_ALLOWED | `level_forbidden`; header-less caller → allowed |
| D1 | escalation ID uniqueness under the counter | rapid submits + archive present | ids strictly increasing, no collision, no directory rescan dependence |

## Cross-PRD relationship (G4)

| Seam / artifact | Direction | Owner | Notes |
|---|---|---|---|
| `shared/task_statuses.py` table A + transition table + claimant fields | W2 **produces** | **W2** | W9 + escalation server consume the SAME table as thin validators — never three tables (program G4) |
| escalation `(action,level,category)→TaskEffect` | W2 **produces** | **W2** | harness `_on_escalation_resolved` consumes |
| `WorkflowStateMachine` / `TerminalReport` | W2 **feeds** (table only) | W9 | W2 exports `outcome_allows_status`; W9 builds the client-side machine + owns workflow.py's `_last_block_*`. W9 depends on W2 (wave 2 wires it) |
| `TaskGroundTruth` / `DeployState` / sweep rewrites | W2 **feeds** (claimant fields) | W10 | W10 reads `claimant_run_id`/`heartbeat_at`/`is_stranded`; W10 depends on W2 |
| `shared/task_metadata.py` (TaskMetadata typed sub-models) | sibling | W3 | **status/claimant are columns (W2); metadata is W3.** Both edit `sqlite_task_backend.py`+`task_interceptor.py` in disjoint concerns (columns vs metadata JSON) — no hard dep; the per-project write lock serializes; second-to-land rebases |
| escalation `server.py` / `queue.py` (fleet-wide package) | W2 edits | W2 | changes affect reify + every project on restart; deploy is fleet-wide (D8) |
| capability-guard header path (2041-2043) | W2 **extends** | W2 | keeps the header transport; adds the identity→role ceiling on top (D7) |

No reciprocal "the other owns it" seams. W2 has **no upstream deps** (brief); W9/W10 declare deps on
W2's real task ids in wave 2.

## Migration & deploy sequencing

1. **Foundation (additive, inert):** τ1 (vocabulary), τ2 (actor + transition table), ρ2 (claimant
   columns) — nothing enforces yet.
2. **Consolidate + log-mode + escalation authority:** ρ1a (store vocabulary reject — the only
   immediate enforcement, safe), ρ1b (interceptor log-mode + actor threading), ω1 (claimant stamping),
   ω2 (orchestrator consolidation), ε1/ω3/ω4 (Table B + infra-hold), ε2 (level ceilings), ε3/ε4 (minors).
3. **Integration gate ζ** proves the whole matrix green in CI (log-mode).
4. **Δ1 (operator gate):** restart fused-memory (out-of-cgroup) → escalation servers → orchestrators, in
   that order (fm first so the v2 migration runs before orchestrators write claimant columns). Enforcement
   stays log-only.
5. **Γ (operator gate):** confirm the transition log shows zero would-reject WARNINGs over the soak
   window; then set `task_status.enforce_transitions=true` and restart fused-memory. Enforcement live.

## Out of scope

- **W9:** `WorkflowStateMachine`, `TerminalReport`, `StewardOutcome`, `BlockDisposition`, workflow.py's
  `_last_block_*` side channel.
- **W3:** `TaskMetadata` schema / typed metadata sub-models (status is a column, not metadata).
- **W10:** `DeployState`, `proc_supervision`, and the reconcile-sweep rewrites that *consume* the
  claimant predicate (W2 provides the fields + predicate; W10 makes the sweeps thin).
- The already-closed `update_task(status=)` bypass (tasks 1664 et al.) — W2 targets the two open 6.4 gaps
  only (vocabulary CHECK on set/add; transition table).
- Re-deriving the reblock guard / stranded sweep / action enum (landed 1617-1626, 1622) — W2 consolidates
  their effect onto the shared tables, it does not rebuild them.
- An automated `restart-all` deploy script (D8) and a general escalation authn/authz scheme.

## Open questions (tactical — safe defaults taken; operator AFK)

1. **Deploy shape (decided: pure operator gates).** Δ1/Γ are deterministic pure gates with a restart
   runbook, not automated `before_done` scripts — rationale in D8 (fm-self-restart / orchestrator
   self-kill / not-yet-existing-script rejection). Revisit once a vetted `scripts/restart-all-*.sh`
   exists and is smoke-tested against a live service.
2. **Enforcement flip mechanism (decided: config `task_status.enforce_transitions`, default false).**
   fused-memory has no hot-reload, so the flip is a config change applied by the Γ restart. If a
   hot-reload path lands, the flip can become a `reload_config`.
3. **`infra_hold` representation (decided: first-class `infra-hold` status, D3).** Alternative — keep the
   metadata flag behind a single `is_infra_held` accessor — is the fallback if the status value proves to
   touch more sweep sites than expected; decide during ω4.
4. **Actor granularity (decided: coarse, safe-open, D5).** The soak (Γ) reveals whether the orchestrator
   process's single fused-memory connection can distinguish pipeline vs reconcile writes by `agent_id`;
   if not, the RECONCILIATION-specific restrictions degrade to comment-documented and the table keeps only
   the actor-independent `(from,to)` legality. Decide after the soak.
5. **Heartbeat cadence (tactical).** Stamp at dispatch + refresh at each phase transition, or a dedicated
   lightweight interval touch; the stranded-ttl is a config, not a hard-coded number. Decide during ω1.
6. **CI mirror test (decided: convert to real import).** test_scheduler.py:8992's hardcoded mirror
   becomes a `from shared.task_statuses import …` equality (or is deleted as tautological) in ω2.
