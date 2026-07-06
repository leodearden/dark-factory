# PRD — workflow-state-machine: type the TaskWorkflow control plane (stream W9)

**Status:** active · **wave 2** · authored 2026-07-06 · bug-hotspot remediation program
(`plans/bug-hotspot-remediation-program-2026-07-06.md`, **workflow** cluster of
`plans/bug-hotspot-survey-2026-07-06.md`; full evidence
`plans/bug-hotspot-survey-2026-07-06-full-findings.json` cluster 1, findings 0–6).
**Brief:** `/home/leo/.claude/spawn-briefs/df-hotspot-2026-07-06/W9-workflow-state-machine.md`.
**Approach:** **B + H** (HIGH-STAKES — the terminal-decision / merge-provenance / steward-RPC
/ capability-gate seams are load-bearing for a *running* factory; contract + two-way boundary
tests below). **Write-tag:** `agent_id="claude-prd-workflow-state-machine"`.
**Owns (G4 authoritative, program seam map):** `WorkflowStateMachine`, `TerminalReport`,
`StewardOutcome`, `BlockDisposition` (+ `WorkflowCancelled`), all homed in a new
`orchestrator/src/orchestrator/workflow_types.py`.

**Upstream (all VERIFIED FILED — deps wired to real ids):** W1 α `MergeProvenance.lookup`
(**2153**); W2 τ2 shared transition table / `outcome_allows_status` (**2168**, →τ1 **2163**);
W4 α `InvocationOutcome`/`classify_invocation` (**2127**) + β `AgentFailureKind` projection
(**2129**); W7 α `FailureCategory` (**2123**) + ζ `BlockRecord` (**2138**); W3 α `RetryLedger`
sub-model / schema home (**2158**).

---

## 1. Goal

`TaskWorkflow` is a single **8,835-line god-class** whose `run()` drives
PLAN→EXECUTE→VERIFY→REVIEW→MERGE inline over **53 mutable instance attributes acting as
inter-phase side channels**. Its "state machine" is nominal: `WorkflowState` is set by
`_enter_phase` (workflow.py:1463) which *emits events but enforces no legality*, with 19 call
sites jumping states freely; `WorkflowOutcome`, the fused-memory status row (22 `set_task_status`
sites), and metadata stamps are three further, drift-prone representations of the same lifecycle.
This PRD replaces the ad-hoc control plane with **typed, enforced seams** so the survey's
confirmed regression classes become impossible by construction rather than patched forever.

After this PRD lands:

1. **"Did my work already land?" is answered by one durable record, not three divergent
   heuristics.** The pre-PLAN, pre-EXECUTE, and merge-phase guards collapse onto a single
   `MergeProvenance.lookup(task_id)` (W1's journal); `_has_prior_implementation` survives only as
   the journal-miss fallback. A test proves **no code path returns `WorkflowOutcome.DONE` without
   journal provenance or an explicit fallback marker** — closing the 6× false-done class
   (846/851/882/883/954/1141; task-2911 incident).
2. **Terminal state has one owner with an enforced transition table.** A `WorkflowStateMachine`
   value object owns `state`, raises on illegal transitions, and makes `DONE`/`CANCELLED`
   *absorbing* (the hand-rolled "already DONE, ignoring late blocked" guard at workflow.py:7744
   becomes a property of the machine, not one method's memory).
3. **The workflow↔harness contract is a return value, not three stashed attributes.** `run()`
   returns a `TerminalReport(outcome, reason, phase, detail, category)`; the harness consumes it,
   deleting the `_last_block_reason/_phase/_detail` side channel (workflow.py:775-777 →
   harness.py:5189-5191). A runtime assert on `run()` exit proves `WorkflowState`,
   `WorkflowOutcome`, and the last status row are mutually consistent (via W2's
   `outcome_allows_status`).
4. **Steward outcome is a typed result, not forensic re-reads of the escalation queue.** A
   `StewardOutcome` sum type flows from the steward to the owning workflow on an in-process
   channel; the workflow branches on it in **one** function; the timestamp-window dismissal
   heuristic (workflow.py:7913/7986) dies. `INTERRUPTED(reason, wip_commits_present)` routes to a
   resume-plan path — encoding the "triage WIP health before restarting" lesson structurally
   (task 2060).
5. **`run()`'s seven-clause exception ladder becomes one `classify_failure → BlockDisposition`
   table** consuming W4's `InvocationOutcome`/`AgentFailureKind` and W7's
   `BlockRecord`/`FailureCategory`. New failure types become table rows with a completeness test;
   the four independent `AllAccountsCappedException` catch sites (workflow/steward/review/dry-run)
   consult the same table.
6. **Anti-thrash counters become one typed `RetryLedger`** (W3's sub-model), with guards as pure
   functions and **persist-failure that escalates** rather than silently losing an increment.
7. **Agent capability wiring is a property of the role**, not name-string tuples: a role that
   forgets to declare its MCP family **fails at import** (the SIMPLE_TASK silent-fallthrough class
   becomes impossible).
8. **Cancellation is one seam.** A `CancellationScope` supervisor converts harness hard-cancel and
   the soft `_cancel_event` into a single `WorkflowCancelled(kind)` caught at one place, running an
   ordered terminal-cleanup list — replacing the `sys.exc_info()` sniffing and the two-file B1/B2
   comment contract.

**Observability at the user/operator surface:** ghost-loop / phantom-done incidents that today
need manual `/unblock` stop (guard collapse + provenance test); healthy-WIP-misdiagnosed-as-failed
steward runs route to resume instead of a wasted restart; a role wired without its MCP family fails
loudly at import instead of silently falling through weeks later in a watched project; every
BLOCKED path carries correct retry-cap accounting because the report is a return type, not an attr
that a path can forget to stash.

Finding 7 (`_SchedulerLike` shadow protocol) is **out of scope — owned by W10** (SchedulerCallbacks
seam; program seam map). This PRD does not touch it.

---

## 2. Background — what the survey confirmed (code-verified 2026-07-06, re-verified here)

`workflow.py` is the #1 fix/amend hotspot (604 commits, 226 fix-class). Cluster 1's
`architecture_notes` name the real decomposition seams as **terminal-decision/status ownership,
steward RPC, merge-provenance lookup, and the agent-invocation capability gate** — exactly the
seams below. `cross_system_notes` document four cross-system-patching cases this PRD closes: (1)
workflow reverse-engineers fused-memory's server-side status legality (`SetTaskStatusRejected`
handlers) — W2's shared table gives workflow a client-side validator that consumes the *same*
authority; (2) `_has_prior_implementation`'s dual-signal heuristic compensates for a missing "main
only advances via merge queue" guarantee — W1's journal supplies it; (3) `AllAccountsCappedException`
escapes into four independent catch sites — the `BlockDisposition` table gives one boundary-level
disposition; (4) warm-lane faults are triaged in two subsystems — the table centralises the
workflow half.

**All anchors re-verified against current main** (see §3). None of the prior remediation
(escalation-repend 1617-1626, born-at-L2 gates 1619, stranded sweep 1622) is redone — this PRD
consolidates their *effect* onto typed seams. This is a **consumer** stream: it wires against the
already-filed wave-1 substrates (W1/W2/W3/W4/W7) rather than reinventing them.

---

## 3. Substrate reality check (G3) — all anchors re-verified against current main

**Upstream substrates (VERIFIED FILED; wired as deps):**

| Assumed capability | Owner / status | Evidence |
|---|---|---|
| `MergeProvenance.lookup(task_id) -> LandedRow \| None` | W1 α **task 2153** — filed (pending) | Landed-outbox store + public façade; the survey named it as the substrate W9 consumes. `_has_prior_implementation` becomes the journal-miss fallback. |
| `outcome_allows_status(outcome, status)` + `is_legal_transition` + `TaskStatus` StrEnum | W2 τ2 **task 2168** (→τ1 **2163**) — filed | `shared/task_transitions.py` exports the map "W9's WorkflowStateMachine consumes" (task 2168 desc verbatim). |
| `InvocationOutcome` sum type + `classify_invocation` | W4 α **task 2127** — filed | `shared/invocation_outcome.py`. |
| `AgentFailureKind`/`AgentFailureClass` (projection of `InvocationOutcome`) | W4 β **task 2129** — filed | "AgentFailureKind/AgentFailureClass remain the public types W9 consumes" (task 2129 desc). |
| `FailureCategory` StrEnum + `CATEGORY_POLICY` | W7 α **task 2123** — filed | `verify_categories.py`; rewires workflow.py:4894/4939 to the enum. |
| typed `BlockRecord` + `BlockClass` (+ b3_gate `block_class` branch) | W7 ζ **task 2138** — filed | `unblock_types.py`; "W9 `_mark_blocked` consumes BlockRecord" (task 2138 desc + seam map). |
| `RetryLedger` typed sub-model + `shared/task_metadata.py` schema home | W3 α **task 2158** — filed | task 2158 lists `RetryLedger (the 8 workflow anti-thrash counters)` among its sub-models. See §Open-Q Q2 on extent. |

**In-repo anchors (grep-verified on current main — line drift within ±20 of the survey, main moves fast):**

| Anchor | Site (current main) |
|---|---|
| Three already-merged guards | pre-PLAN `_recover_if_already_merged` workflow.py:**7291**; pre-EXECUTE `already_on_main` **1777-1816**; merge-phase `is_ancestor` **2208-2224**. Shared heuristic `_has_prior_implementation` **7202**. |
| Terminal state representations | `WorkflowState` enum **324**; `WorkflowOutcome` enum **337**; `_enter_phase` **1463** (sets `self.state` **1479**); "already DONE, ignoring late blocked" hand-guard **7744**. |
| Block side channel | `_last_block_reason/_detail/_phase` defn workflow.py:**775-777**; written 2065-2067 / 2800-2802 / 7751-7753; **read harness.py:5189-5191**. |
| Steward RPC | steward is a decoupled loop: `TaskSteward` steward.py:76, `_run_loop` 140, `_handle_escalation(esc) -> None` **294**; workflow waits via `_await_steward_completion` **8739**; forensic probes / dismissal-window in `_mark_blocked` **7913 / 7980-7998**. |
| Exception ladder | run() catches AllAccountsCapped **1919**, _SessionBudgetExhausted **1929**, SetTaskStatusRejected **1957**, WarmLaneRequeue (3-subclass) **2014**, VerifyInfraError **2070**, `_is_infra_oserror` **2098**, broad Exception **2114**; sibling catches steward.py:653 / review_checkpoint.py:192 / dry_run_unblock.py:341. |
| Anti-thrash counters (8) | `consecutive_no_plan_failures`/`total_no_plan_failures`/`last_no_plan_main_sha` (3303-3330); `consecutive_infra_resume_failures`/`last_infra_resume_iteration_count` (3424-3454, persist-best-effort 3456-3464); `consecutive_merge_thrash`/`last_merge_outcome_signature` (3523-3536); `merge_first_enqueued_at` (1030). Merge helper `_merge_fresh_metadata` **3233**; signature `_normalize_cause_hint` **367** / `_compute_merge_outcome_signature` **394**. |
| Capability wiring | `_MCP_CONFIG_ROLES`/`_PLAN_TOOLS_ROLES` **105-114**; `_invoke` **6876** (role gates 6892/6913/6925/6938; hardcoded `role.name in (...)` 6867/6913). `AgentRole` dataclass roles.py:8 (`allowed_tools` 11). |
| Cancellation | `_cancel_event` **727**; `_await_cancellable` **8642**; `_handle_soft_cancel` **8691**; `sys.exc_info()` hard-cancel sniff **2154**; harness B2 `synthetic_cancel` harness.py:489 / **5239**. |

**No novel substrate is assumed that does not exist.** The four W9-owned types
(`WorkflowStateMachine`, `TerminalReport`, `StewardOutcome`, `BlockDisposition`, plus
`WorkflowCancelled`) are this PRD's own deliverables in a clean new `workflow_types.py`. G3 passes.

---

## 4. Sketch of approach

`workflow.py` is a **single `metadata.files` file-level lock** (Lock-charter Contract-1), so — like
the 15-task merge-queue chain (df 1985-2002) and W1/W7 — the eight mechanisms decompose into a
**strictly linear spine** on `workflow.py` to prevent rebase thrash on the god-file, plus one B+H
integration-gate leaf that branches off the spine (a new test module, its own lock).

- **α — Guard collapse onto `MergeProvenance.lookup`.** The three merge guards call W1's lookup
  first; `_has_prior_implementation` becomes the journal-miss fallback; a test pins the
  no-DONE-without-provenance-or-fallback-marker invariant. *(the early, clearly-titled leaf; kills
  the #1 false-done class; consumes W1 α = 2153.)*
- **β — `WorkflowStateMachine` + `workflow_types.py`.** New types module; relocate
  `WorkflowState`/`WorkflowOutcome` there (re-export shim in workflow.py); the machine owns `state`
  with a legal-transition table, `DONE`/`CANCELLED` absorbing, raising on illegal moves. Consumes
  W2's `is_legal_transition`/`outcome_allows_status` (= 2168).
- **γ — `TerminalReport` + delete the `_last_block_*` side channel.** `_mark_blocked` constructs
  the report; `run()` returns it; harness consumes it. Runtime consistency assert on `run()` exit
  via `outcome_allows_status`. `TerminalReport.category: FailureCategory` (W7 α = 2123).
- **δ — `StewardOutcome` sum type.** Steward `_handle_escalation` publishes it on an in-process
  per-task channel; `_await_steward_completion` returns it; `_mark_blocked` branches in one place;
  the dismissal-window heuristic dies; `INTERRUPTED`+wip → resume-plan.
- **ε — `classify_failure → BlockDisposition` table.** Replaces the seven-clause ladder; consumes
  W4 `InvocationOutcome`/`AgentFailureKind` (2127/2129) and W7 `FailureCategory`/`BlockRecord`
  (2123/2138); the four cap-catch sites consult it; completeness test.
- **ζ — `RetryLedger`.** Migrate the 8 counters + `_merge_fresh_metadata` RMW + signature helpers
  onto W3's typed sub-model (2158); guards → pure functions; persist-failure escalates.
- **η — Capability wiring as a role property.** `AgentRole.mcp_families`/`sandboxed`; `_invoke`
  derives every gate from the role; the tuples + hardcoded name checks die; import-time assertion.
- **θ — `CancellationScope` supervisor.** One typed `WorkflowCancelled(kind)` caught once; ordered
  terminal-cleanup list; replaces exc_info sniffing + B1/B2. Highest-effort leaf (see §Open-Q Q1
  for the in-batch-vs-follow-on decision + deferrability).
- **ι — B+H integration gate.** One new test module facing both sides of seams 1-7 (§9); the sole
  non-cancellation leaf. θ carries its own cancellation boundary tests so ι does **not** depend on
  θ (deferrability).

---

## 5. Resolved design decisions (do not relitigate)

1. **Linear `workflow.py` spine.** `metadata.files` is file-level (Contract-1); every task touching
   `workflow.py` serialises on one lock, so the spine is strictly linear (df 1985-2002 / W1 / W7
   precedent). Semantic order respects the accretion: guard-collapse → state-machine →
   terminal-report → steward → block-disposition → retry-ledger → capability → cancellation.
2. **Types home = a new `orchestrator/src/orchestrator/workflow_types.py`** (precedent:
   `merge_types.py`, `unblock_types.py`). `WorkflowStateMachine`, `TerminalReport`,
   `StewardOutcome`, `BlockDisposition`, `WorkflowCancelled`, and the relocated
   `WorkflowState`/`WorkflowOutcome` enums live there; `workflow.py` keeps a re-export shim
   (`from orchestrator.workflow_types import WorkflowState, WorkflowOutcome`) so the 19 in-file call
   sites and external importers (`orchestrator.workflow.WorkflowOutcome`) are unbroken. Rationale:
   `harness.py`/`steward.py`/`review_checkpoint.py`/`dry_run_unblock.py` must import these types;
   importing from the 8.8k-line god-module invites circular imports.
3. **Provenance-first, fallback-preserved (α).** Each guard calls `MergeProvenance.lookup(task_id)`
   first; a hit → DONE with that provenance; a miss → `_has_prior_implementation` (the SHA +
   iteration-log heuristic) as the **explicit** legacy fallback, which sets a fallback marker. The
   enforcement test asserts `run()` cannot reach `WorkflowOutcome.DONE` unless (journal hit) OR
   (explicit fallback marker) OR (human `done_provenance`). The collapse is **behaviourally safe
   with an empty journal** (lookup misses → fallback), so W9 α depends only on W1 α's lookup API
   (2153), not on W1's write-side chain (§Open-Q Q3).
4. **`WorkflowStateMachine` consumes W2's authority — never a fourth table.** The machine's
   transition legality and its `outcome → allowed-status` consistency both come from
   `shared/task_transitions.py` (`is_legal_transition` / `outcome_allows_status`, task 2168). The
   client-side machine is a **thin validator** over the shared table (program G4 decision #1: the
   escalation server, fused-memory interceptor, and W9 machine all consume the SAME table). DONE and
   CANCELLED are absorbing; the 7744 hand-guard is deleted in favour of the machine property.
5. **`TerminalReport` replaces the attr side channel; `run()` returns it.** `TerminalReport(outcome:
   WorkflowOutcome, reason: str, phase: WorkflowState, detail: str, category: FailureCategory |
   None)`. `_mark_blocked` builds it; `run()` returns it; the harness reads `report.block_reason`
   etc. (deleting the `_last_block_*` reads at harness.py:5189-5191). A runtime assert on `run()`
   exit checks `outcome_allows_status(report.outcome, last_status_row)` and that `report.phase ==
   self.machine.state` — a mismatch is a loud bug, not a silent misattribution.
6. **`StewardOutcome` travels on an in-process typed channel; the escalation queue stops being the
   RPC.** The steward is a decoupled asyncio object (`_run_loop`), so `_handle_escalation` resolves
   a per-task `Future`/`Queue` (registered by the owning workflow keyed by task/escalation id) with
   `StewardOutcome = RESOLVED(resolution_text) | REESCALATED_L1(esc_id) |
   TERMINAL_DECISION(new_status: TaskStatus) | INTERRUPTED(reason: timeout|attempt_cap,
   wip_commits_present: bool) | BUDGET_EXHAUSTED`. `_await_steward_completion` returns it; the
   escalation queue remains persistence/notification. The five timestamp-window/queue probes in
   `_mark_blocked` (7913-7998) are deleted. `INTERRUPTED(wip_commits_present=True)` routes to a
   resume-plan path, **not** L1 re-escalation (structural encoding of
   `feedback_unblock_steward_reescalated_is_incomplete_impl_triage`, task 2060). `wip_commits_present`
   is derived from the worktree, not guessed.
7. **`BlockDisposition` table replaces the ladder; sibling catch sites consult it.** `classify_failure(exc)
   -> BlockDisposition(category: FailureCategory, escalate_to_human: bool, requeue_kind, counts_against_requeue_cap:
   bool, reason_prefix: str, block_class: BlockClass)`. `_mark_blocked` accepts a `BlockDisposition`
   and constructs a `BlockRecord` (W7 ζ) from it. `run()` keeps one `except` per outcome-kind
   (requeue / block / cancel) that consults the table. A completeness test asserts every exported
   exception type in git_ops/verify/cli_invoke/usage_gate has a row. Cap/agent failures map through
   W4's `AgentFailureKind`/`InvocationOutcome`. Home = `workflow_types.py` (orchestrator package —
   importable by the steward/review/dry-run satellites; the disposition *policy* is orchestrator-level
   even though `AllAccountsCappedException` originates in shared/cli_invoke).
8. **`RetryLedger` is W3's sub-model; W9 wires the workflow-side consumers.** W3 α (2158) owns the
   `RetryLedger` shape; W9 ζ migrates the 8 top-level counter keys + `_merge_fresh_metadata` RMW +
   the signature helpers onto it under one `retry_ledger` key, with guards as pure `RetryLedger ->
   GuardVerdict` functions. `load()` reads the new key **and** (back-compat) the 8 legacy top-level
   keys so in-flight tasks aren't reset. **Persist failure escalates** (the guard exists to stop
   money-burning loops; silently losing an increment defeats it). Extent fallback in §Open-Q Q2.
9. **Capability wiring is a role property (η).** `AgentRole` gains `mcp_families:
   frozenset[Literal['orchestrator','plan_tools']]` and `sandboxed: bool`; `_invoke` derives every
   gate from the role object; `_MCP_CONFIG_ROLES`/`_PLAN_TOOLS_ROLES` + the hardcoded
   `role.name in (...)` checks die. A roles.py import-time assertion: any role whose `allowed_tools`
   reference fused-memory/escalation/plan-tools names must declare the matching family (a forgetful
   role fails at import — the SIMPLE_TASK class made structurally impossible).
10. **One cancellation seam (θ).** A `CancellationScope` runs the `run()` body; harness hard-cancel
    (`asyncio` task.cancel) and the soft `_cancel_event` both surface as one
    `WorkflowCancelled(kind=hard|soft)` caught at exactly one place; an ordered `on_terminal`
    cleanup list (lane release, config-dir cleanup, steward stop) runs with the cancel kind as
    input, replacing the `sys.exc_info()` sniff (2154) and the B1/B2 two-file comment contract
    (harness.py:5239). Every future long await is covered by construction. Cancellation produces a
    `TerminalReport(outcome=CANCELLED, ...)` — so θ depends on γ.
11. **No deploy capstone.** These are library/refactor changes to the orchestrator, dormant on main
    until the next restart (harmless). Per W1/W7 and program decision #6, fleet activation is a
    program-level restart after several streams land — W9 files no deploy task (§Open-Q Q4).

---

## 6. Pre-conditions for activating

- **Upstream (all filed; wired as hard deps):** W1 α (2153), W2 τ2 (2168, →2163), W4 α/β
  (2127/2129), W7 α/ζ (2123/2138), W3 α (2158). W9 is wave 2; every consumed substrate is filed
  (not necessarily merged) — the deps gate dispatch until each lands.
- **Runtime activation is out of scope** (dormant on main until a program-level restart; §Open-Q Q4).

---

## 7. Cross-PRD relationship (G4 — per program seam map)

| Other stream / PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| **W1** merge-queue-reliability | **W9 consumes W1** | `MergeProvenance.lookup(task_id)` → collapse the three already-merged guards | **W1 owns the journal + API; W9 owns the collapse** | dep on **2153** (α) |
| **W2** task-status-authority | **W9 consumes W2** | `outcome_allows_status` / `is_legal_transition` / `TaskStatus` — the state-machine's validator | **W2 owns the table; W9 owns the client-side machine** | dep on **2168** (→2163) |
| **W4** invocation-outcome | **W9 consumes W4** | `InvocationOutcome`, `AgentFailureKind` feed `BlockDisposition` | **W4 owns classification; W9 consumes** | dep on **2127** (α), **2129** (β) |
| **W7** verify-plan | **W9 consumes W7** | `FailureCategory` (TerminalReport/BlockDisposition category), `BlockRecord` (built in `_mark_blocked`) | **W7 owns the types; W9 consumes** | dep on **2123** (α), **2138** (ζ) |
| **W3** task-metadata-schema | **W9 consumes W3** | `RetryLedger` typed sub-model / schema home | **W3 owns the sub-model; W9 wires workflow-side consumers** | dep on **2158** (α); extent fallback Q2 |
| **W4** invocation-outcome (η) | co-touch (no consumed type) | `steward.py` — W4 η reroutes the steward *invocation* loop; W9 δ retypes the steward *escalation-outcome* (disjoint methods: `_invoke_with_session`/`_invoke_steward` vs `_handle_escalation`) | **W4 owns invocation; W9 owns outcome** | wave-1 W4 lands first; file lock serialises; **no hard dep** |
| **W10** harness-supervision | co-touch (no consumed type) | `harness.py` — W9 γ deletes `_last_block_*` reads (5189-5191) and θ removes the B2 cancel dual-guard (5239); W10 rewrites the supervision/sweep layer | **disjoint concerns** | both wave-2; file lock serialises; **no hard dep** |
| **W10** harness-supervision | **W10 may consume W9** | `TerminalReport` (replaces the `_last_block_*` attr pokes W10's sweeps read) | **W9 produces; W10 consumes if it lands after** | program seam map; W10 wires if needed |
| finding 7 `_SchedulerLike` | **NOT W9** | SchedulerCallbacks seam | **W10** | explicitly excluded (brief) |

**No reciprocal-ownership ambiguity.** Every consumed-substrate edge is a clean "they produce, W9
consumes." The two co-touch edges (W4 η on `steward.py`, W10 on `harness.py`) are disjoint-method
file-lock coordinations, not contested seams — resolved by wave ordering + the per-file lock, no
hard dep.

---

## 8. Contract section (B + H)

### 8.1 `workflow_types.py` value types (W9-owned)

```python
# Relocated here (workflow.py keeps a re-export shim):
class WorkflowState(Enum): ...        # PLAN, EXECUTE, VERIFY, REVIEW, MERGE, ...
class WorkflowOutcome(Enum): ...      # DONE, BLOCKED, ESCALATED, CANCELLED, MERGE_DEFERRED, ...

class WorkflowStateMachine:
    """Owns `state`; enforces legal transitions; DONE/CANCELLED absorbing."""
    state: WorkflowState
    def transition(self, to: WorkflowState) -> None: ...   # raises IllegalTransition on illegal / post-absorbing
    def is_terminal(self) -> bool: ...
    # legality + outcome→status consistency delegate to shared.task_transitions (W2 τ2)

@dataclass(frozen=True)
class TerminalReport:
    outcome: WorkflowOutcome
    reason: str
    phase: WorkflowState
    detail: str
    category: FailureCategory | None          # W7 α (2123)

# StewardOutcome — a tagged union / frozen dataclasses:
#   RESOLVED(resolution_text) | REESCALATED_L1(esc_id) | TERMINAL_DECISION(new_status: TaskStatus)
#   | INTERRUPTED(reason: Literal['timeout','attempt_cap'], wip_commits_present: bool) | BUDGET_EXHAUSTED

@dataclass(frozen=True)
class BlockDisposition:
    category: FailureCategory                 # W7 α (2123)
    escalate_to_human: bool
    requeue_kind: RequeueKind                 # requeue / block / cancel
    counts_against_requeue_cap: bool
    reason_prefix: str
    block_class: BlockClass                   # W7 ζ (2138) — for the BlockRecord built in _mark_blocked

def classify_failure(exc: BaseException) -> BlockDisposition: ...   # table; completeness-tested

@dataclass(frozen=True)
class WorkflowCancelled(Exception):
    kind: Literal['hard', 'soft']
```

### 8.2 Invariants (prose contract → boundary tests below make them executable)

- **SM-1 (legality).** `WorkflowStateMachine.transition(to)` raises `IllegalTransition` for any move
  not in the legal set; `DONE`/`CANCELLED` are absorbing (any transition out raises). The 7744
  "already DONE, ignoring late blocked" case is this property, not a method's memory.
- **SM-2 (consistency).** On `run()` exit, `outcome_allows_status(report.outcome, last_status_row)`
  is True and `report.phase == machine.state`; a mismatch raises loudly (via W2's map, 2168).
- **MP-1 (provenance-first).** Each of the three guards calls `MergeProvenance.lookup(task_id)`
  first; hit → DONE with that provenance; miss → `_has_prior_implementation` fallback marker.
- **MP-2 (no unprovenanced DONE).** No `run()` path returns `WorkflowOutcome.DONE` unless a journal
  row exists, or an explicit fallback marker is set, or a human `done_provenance` is present.
- **TR-1 (report is the channel).** The harness reads terminal context from the returned
  `TerminalReport`; the `_last_block_reason/_phase/_detail` attributes are deleted; no path can
  "forget to stash."
- **SO-1 (typed steward outcome).** `_await_steward_completion` returns a `StewardOutcome`; the
  timestamp-window dismissal heuristic is deleted; `_mark_blocked` branches on the union in one
  place. `INTERRUPTED(wip_commits_present=True)` → resume-plan (not L1).
- **BD-1 (one classifier).** `run()` consults `classify_failure` for every failure; the four
  `AllAccountsCappedException` catch sites (workflow/steward/review/dry-run) use the same table.
- **BD-2 (completeness).** A test asserts every exported exception type in
  git_ops/verify/cli_invoke/usage_gate maps to a `BlockDisposition` row; a new exception with no row
  fails the test (not silent default behaviour).
- **RL-1 (typed ledger, loud persist).** Counters live under one typed `retry_ledger`; guards are
  pure `RetryLedger -> GuardVerdict`; `load()` is back-compat with the 8 legacy keys; a persist
  failure **escalates** (never silently drops an increment).
- **CW-1 (wiring is a role property).** `_invoke` derives MCP-family + sandbox gating from the
  `AgentRole` object; a role declaring a plan-tools/fused-memory/escalation tool without the
  matching `mcp_families` entry fails a roles.py **import-time** assertion.
- **CX-1 (one cancel seam).** Hard-cancel and soft `_cancel_event` both surface as one
  `WorkflowCancelled(kind)` caught at exactly one place; the ordered `on_terminal` list runs with
  the kind; no `sys.exc_info()` sniffing, no B1/B2 two-file contract; a long await added anywhere
  inside the scope is cancellable by construction.

---

## 9. Boundary-test sketch (B + H) — faces both producer and consumer sides

The **guard-collapse equivalence suite** (rows 1-4) and the **state-machine property suite** (rows
5-6) are the headline G5 two-way artifacts, run against the *historical incident shapes*, not
invented inputs (G6). Rows 1-12 (seams 1-7) live in the ι integration module; rows 13-15
(cancellation) live with θ so ι does not depend on θ.

| # | Scenario | Preconditions | Postconditions (asserted) | Faces |
|---|---|---|---|---|
| 1 | Journal-hit → DONE | `MergeProvenance.lookup(Z)` returns a row on main | all three guards resolve DONE via provenance; `_has_prior_implementation` not consulted | provenance + workflow |
| 2 | Journal-miss → fallback, no phantom DONE | empty journal, ghost-loop shape (rebased HEAD == base_commit) | lookup misses → `_has_prior_implementation` fallback; **no DONE** without a fallback marker (MP-2); task re-dispatchable | workflow + reconciler |
| 3 | `.task/` contamination (task 954 shape) | worktree carries `.task/` artifacts but no real work | fallback returns `has_work=False`; not marked DONE | workflow |
| 4 | No-DONE-without-provenance enforcement | exhaustive: every `run()` path that can return DONE | test proves each is gated by journal-hit ∨ fallback-marker ∨ human done_provenance (MP-2) | workflow (property) |
| 5 | Illegal transition raises | machine at DONE, attempt `→ BLOCKED` | `IllegalTransition` raised; state unchanged (SM-1; the 7744 case) | state machine |
| 6 | Outcome↔status consistency on exit | random legal run sequences (property test) | `outcome_allows_status(outcome, status_row)` holds and `phase == machine.state` after every run (SM-2, via 2168) | machine + W2 table |
| 7 | Harness consumes `TerminalReport` | a BLOCKED run | harness reads `report.block_reason/_detail/_phase`; grep shows no `_last_block_*` attr on `TaskWorkflow` (TR-1) | workflow + harness |
| 8 | Steward RESOLVED / TERMINAL_DECISION | steward resolves; steward defers | `_await_steward_completion` returns the typed outcome; `_mark_blocked` branches once; no timestamp-window read | workflow + steward |
| 9 | Steward INTERRUPTED+wip → resume | steward killed at attempt cap with WIP commits present | outcome `INTERRUPTED(attempt_cap, wip_commits_present=True)` → resume-plan path, **not** L1 (task 2060) | workflow + steward |
| 10 | One classifier, all cap sites | `AllAccountsCappedException` raised in each of workflow/steward/review/dry-run | each consults `classify_failure`; identical `BlockDisposition` (BD-1) | workflow + 3 satellites |
| 11 | Classifier completeness | a synthetic new exception type with no table row | completeness test fails (BD-2) | classifier (property) |
| 12 | Capability wiring import assert | a role declaring a plan-tools tool without `mcp_families` | roles.py import-time assertion fires (CW-1); SIMPLE_TASK-shape regression covered | roles + `_invoke` |
| 13 | Retry-ledger persist escalates | ledger persist raises mid-increment | escalation fired; increment not silently lost (RL-1); legacy 8-key load round-trips | workflow + metadata |
| 14 | Hard-cancel → single seam | harness `task.cancel()` mid-VERIFY | one `WorkflowCancelled(kind='hard')` caught once; `on_terminal` cleanups run ordered; no exc_info sniff; `TerminalReport(CANCELLED)` (CX-1) | workflow + harness |
| 15 | Soft-cancel covers a new await | `_cancel_event` set during a long steward wait | the await inside the scope is cancelled; `WorkflowCancelled(kind='soft')`; lane released (CX-1) | workflow (scope) |

Rows 1-4 draw their incident shapes from tasks 846/954/1141 and the task-2911 incident
(workflow.py:1745-1754); row 9 from task 2060; row 12 from the SIMPLE_TASK fallthrough
(reify esc-4943-54). Crashes/kills are simulated by injected fault points, not real process kills.

---

## 10. Decomposition plan (the linear spine; Greek labels → task IDs at decompose)

Every leaf carries `force_full_path=true` (each touches the workflow god-module with cross-cutting
design implications — none is a safe fast-path candidate). `metadata.files` is file-level
throughout. Priorities: α–ε and θ **high** (correctness seams + false-done class + highest-stakes
control-flow); ζ, η **medium**; ι **high** (the G5 gate).

### Spine (linear on `workflow.py` / `workflow_types.py`)

- **α — Guard collapse onto `MergeProvenance.lookup`; `_has_prior_implementation` = journal-miss
  fallback.** Files: `orchestrator/src/orchestrator/workflow.py`,
  `orchestrator/src/orchestrator/artifacts.py`, `orchestrator/tests/test_workflow_merge_provenance.py`.
  Signal: the guard-collapse equivalence suite (boundary rows 1-4) is green — journal-hit → DONE via
  provenance; journal-miss → fallback with **no phantom DONE**; `.task/` contamination and
  rebased-HEAD==base shapes do not mark DONE; the no-DONE-without-provenance-or-marker property
  holds. Prereq: **W1 α = 2153**.
- **β — `WorkflowStateMachine` + `workflow_types.py`; relocate the enums.** Files: `workflow.py`,
  `orchestrator/src/orchestrator/workflow_types.py` (new), `orchestrator/tests/test_workflow_state_machine.py`.
  Signal: legal transitions succeed; illegal transitions and post-DONE/CANCELLED moves raise
  (boundary row 5); `outcome_allows_status`/`is_legal_transition` are imported from
  `shared.task_transitions` (2168), not redefined. Prereqs: α, **W2 τ2 = 2168**.
- **γ — `TerminalReport`; delete `_last_block_*`; run()-exit consistency assert.** Files:
  `workflow.py`, `workflow_types.py`, `orchestrator/src/orchestrator/harness.py`,
  `orchestrator/tests/test_workflow_terminal_report.py`. Signal: harness consumes the returned
  report; grep shows no `_last_block_*` attribute on `TaskWorkflow`; the exit consistency assert
  (boundary rows 6-7) is green. Prereqs: β, **W7 α = 2123** (FailureCategory).
- **δ — `StewardOutcome` sum type on an in-process channel; kill the dismissal-window heuristic.**
  Files: `workflow.py`, `workflow_types.py`, `orchestrator/src/orchestrator/steward.py`,
  `orchestrator/tests/test_steward_outcome.py`. Signal: `_await_steward_completion` returns a typed
  `StewardOutcome`; `_mark_blocked` branches once; the 7913/7986 timestamp-window probes are
  deleted; `INTERRUPTED`+wip → resume-plan (boundary rows 8-9). Prereq: γ. *(steward.py co-touch
  with W4 η — wave-1, disjoint methods; no hard dep, see §7.)*
- **ε — `classify_failure → BlockDisposition` table; consume W4/W7 types; unify the four cap
  sites.** Files: `workflow.py`, `workflow_types.py`, `steward.py`,
  `orchestrator/src/orchestrator/review_checkpoint.py`,
  `orchestrator/src/orchestrator/dry_run_unblock.py`,
  `orchestrator/tests/test_block_disposition.py`. Signal: the seven-clause ladder collapses to one
  `except`-per-outcome-kind consulting the table; all four cap-catch sites use it; completeness test
  (boundary rows 10-11); `_mark_blocked` builds a `BlockRecord` from the disposition. Prereqs: δ,
  **W4 α = 2127**, **W4 β = 2129**, **W7 α = 2123**, **W7 ζ = 2138**.
- **ζ — `RetryLedger`: migrate the 8 counters onto W3's sub-model; guards → pure fns; persist
  escalates.** Files: `workflow.py`, `orchestrator/tests/test_retry_ledger.py`. Signal: the 8
  top-level counter keys + `_merge_fresh_metadata` RMW + signature helpers route through one typed
  `retry_ledger` (back-compat 8-key load); a persist failure escalates instead of silently dropping
  an increment (boundary row 13). Prereqs: ε, **W3 α = 2158**.
- **η — Capability wiring as a role property; delete the tuples + hardcoded name checks; import
  assertion.** Files: `workflow.py`, `orchestrator/src/orchestrator/agents/roles.py`,
  `orchestrator/tests/test_agent_capability_wiring.py`. Signal: `_invoke` derives every gate from
  `AgentRole.mcp_families`/`sandboxed`; a role missing its family fails a roles.py import assert
  (boundary row 12); the SIMPLE_TASK fallthrough shape is covered. Prereq: ζ.

### Cancellation (spine tip; workflow.py + harness.py)

- **θ — `CancellationScope` supervisor; one `WorkflowCancelled(kind)`; ordered `on_terminal`.**
  Files: `workflow.py`, `workflow_types.py`, `harness.py`,
  `orchestrator/tests/test_workflow_cancellation.py`. Signal: hard-cancel and soft `_cancel_event`
  surface as one typed control signal caught at one place; the `on_terminal` cleanup list runs
  ordered; `sys.exc_info()` sniffing (2154) and the B1/B2 dual-guard (harness.py:5239) are gone; a
  soft-cancel during a long await is honoured; cancellation yields `TerminalReport(CANCELLED)`
  (boundary rows 14-15, carried by **this** task). Prereqs: η (spine-lock order) + γ (TerminalReport)
  + ε (WorkflowCancelled distinct from the failure ladder). Highest-effort leaf — §Open-Q Q1.

### Integration gate (B + H; new test module — branches off η, NOT θ)

- **ι — Two-way boundary suite for seams 1-7.** Files:
  `orchestrator/tests/test_workflow_state_machine_boundary.py` (new). Signal (**the leaf**): boundary
  rows 1-12 pass facing both producer and consumer sides — guard-collapse equivalence, state-machine
  legality/consistency, TerminalReport harness consumption, StewardOutcome routing, BlockDisposition
  completeness, capability import-assert. Prereq: η (all seven seams landed). Does **not** depend on
  θ — cancellation's boundary rows travel with θ, so θ is independently deferrable (§Open-Q Q1).

**DAG.** Spine: α → β → γ → δ → ε → ζ → η. Then θ ← {η, γ, ε}; ι ← η (parallel to θ; separate lock).
Cross-batch: α←2153; β←2168; γ←2123; δ←(none cross-batch); ε←{2127,2129,2123,2138}; ζ←2158; η,θ,ι
none cross-batch. Sole leaves: ι and θ (both carry observable signals).

---

## 11. Out of scope

- **Finding 7 — `_SchedulerLike` shadow protocol** (the three-facade split). **W10-owned**
  (SchedulerCallbacks seam; brief + program seam map).
- **The W1 journal WRITE side** (write-ahead ordering, startup reconciler, scheduler consult) — W1
  owns it; W9 consumes only `MergeProvenance.lookup`.
- **The W2 transition-table definition / interceptor enforcement** — W2 owns it; W9's machine is a
  thin client-side validator over it.
- **W4's steward invocation-loop deletion (η) / cap-retry internals** — W4 owns them; W9 δ retypes
  only the steward escalation-*outcome* channel (disjoint method).
- **W7's `classify_failure`** (verify-failure → FailureCategory) — a *different* classifier from
  W9's `classify_failure` (agent-invocation exception → BlockDisposition); W9 consumes W7's
  `FailureCategory`/`BlockRecord` types, not W7's function.
- **The `RetryLedger` sub-model definition** — W3 owns it (2158); W9 wires the workflow-side
  consumers.
- **W10's harness supervision/sweep rewrites** — disjoint harness.py concerns; W9 touches only the
  `_last_block_*` read (γ) and the B2 cancel dual-guard (θ).
- **Fleet deploy / orchestrator restart** — program-level (§Open-Q Q4).

---

## 12. Open questions (surfaced but not decided — tactical; AFK-safe defaults recorded)

1. **Cancellation in-batch vs its own follow-on PRD (θ).** The brief permits splitting the
   highest-effort leaf. **Default taken: keep θ in-batch as the spine tip, with structural
   deferrability** — ι (integration gate) depends on η, **not** θ, and θ carries its own cancellation
   boundary tests (rows 14-15). Rationale: θ is on the same `workflow.py` lock, so a separate PRD
   yields **zero** parallelism and forces a second `run()`-control-flow design pass that must be
   reconciled with the exception-ladder (ε) and terminal-report (γ) rewrites. If θ's implementation
   balloons past the batch budget at dispatch, it can be lifted into a follow-on PRD **then** without
   disturbing seams 1-7 (ι already gates them). Decide-at: dispatch of θ.
2. **`RetryLedger` W3 extent (ζ).** W3 α (2158) lists a `RetryLedger` sub-model, but the program
   seam map's row for `shared/task_metadata.py` enumerates only BeforeDone/DoneProvenance/
   MemoryHints/ExternalDep (RetryLedger not listed) — a possible `producer-extent-short`. **Default
   taken (the brief's own fallback): if W3's `RetryLedger` covers all 8 fields, consume it; else
   define the typed `RetryLedger` accessor in `workflow_types.py` and register it via W3's schema
   extension point.** Dep on 2158 holds either way. Decide-at: ζ impl (grep W3's landed sub-model).
3. **W1 journal-write dependency for the guard collapse (α).** The collapse is behaviourally safe
   with an empty journal (lookup misses → fallback), so **default taken: α depends on W1 α's lookup
   API (2153) only, not on W1's write-side chain (β/γ/δ).** Full benefit (journal hits) materialises
   when W1 β/γ land; W9 does not hard-block on them because the fallback preserves correctness.
   Decide-at: post-W1 review (whether to add a soft dep once W1 β lands).
4. **Deploy/activation capstone.** **Default taken: W9 files NO deploy capstone** — changes are
   dormant on main until a program-level orchestrator restart (per W1/W7; program decision #6:
   deterministic self-restart 2064/2105). A per-stream restart would thrash the fleet. Decide-at:
   program-level deploy planning.
5. **`classify_failure` / `BlockDisposition` home (ε).** **Default taken: `workflow_types.py`
   (orchestrator package)** so the steward/review/dry-run satellites import it without touching
   shared/; the disposition *policy* is orchestrator-level even though `AllAccountsCappedException`
   originates in shared/cli_invoke. An architect may relocate to a dedicated
   `orchestrator/block_disposition.py` if `workflow_types.py` grows unwieldy. Decide-at: ε impl.
6. **`StewardOutcome` channel primitive (δ).** In-process typed transport is decided (D6); whether
   it is an `asyncio.Future` registered per task or a small `asyncio.Queue` keyed by escalation id is
   tactical. **Default: a per-task `Future` the workflow registers before `_await_steward_completion`
   and the steward resolves in `_handle_escalation`.** Decide-at: δ impl.

---

## 13. Note on tracking metadata

Per the prd skill: the orchestrator does **not** currently read the `user_observable_signal` /
`consumer_ref` / substrate-confirmed metadata fields these tasks carry — they are substrate for a
future tracking-infra session. The capability manifest beside this PRD
(`plans/workflow-state-machine-prd.capability-manifest.md`) is the artifact a dispatch-time
architect or downstream verifier diffs against substrate.
