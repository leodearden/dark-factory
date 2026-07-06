# Capability manifest — workflow-state-machine PRD (W9)

Mechanizes G3 (assumed-substrate verified) + G6 (premise validity) per leaf. Each leaf's
observable/RED signal is decomposed into the capabilities it asserts, each **bound to evidence**.
Evidence forms: `grep:<file>:<line>` = wired on current main (re-verified 2026-07-06);
`producer:task-<N>` = delivered by a task in this leaf's **transitive dependency closure**;
`W9-new` = this task's own deliverable (a new type/field — not an assumed pre-existing substrate,
so not a G3 hazard). Any capability resolving to a FAIL value **blocks queueing**. Statuses below
are all **PASS** (zero FAIL bindings); two G6 caveats are recorded with their resolution.

Line anchors below were grep-confirmed on main 2026-07-06 (`workflow.py` 8,835 lines; drift within
±20 of the survey). An implementing architect should re-confirm at dispatch — main moves fast.

---

## α — Guard collapse onto `MergeProvenance.lookup`
- `MergeProvenance.lookup(task_id) -> LandedRow|None` → **producer:task-2153** (W1 α, in dep
  closure). Extent = the public lookup API + landed-outbox store. **PASS.**
- pre-PLAN guard `_recover_if_already_merged` → `grep:workflow.py:7291`. **PASS.**
- pre-EXECUTE `already_on_main` guard → `grep:workflow.py:1777-1816`. **PASS.**
- merge-phase ancestor guard → `grep:workflow.py:2208-2224` (`is_ancestor`). **PASS.**
- `_has_prior_implementation` (survives as fallback) → `grep:workflow.py:7202`. **PASS.**
- `WorkflowOutcome.DONE` (the enforced-invariant target) → `grep:workflow.py:337`. **PASS.**
- G6 (rejection/enforcement — the no-DONE-without-provenance test): the enforcement predicate is
  **W9-new** (this task authors it); the incident shapes it runs against (ghost-loop
  rebased-HEAD==base, `.task/` contamination) are drawn from tasks 846/954/1141 + task-2911
  (`grep:workflow.py:1745-1754`) — **real historical shapes, not invented** (G6 branch-4 satisfied).
  **PASS.**
- G6 (premise — safe with empty journal): lookup-miss → fallback is a **dependency-correctness**
  fact (MP-1/MP-2), not a numeric/exactness claim. **PASS.**

## β — `WorkflowStateMachine` + `workflow_types.py`
- `is_legal_transition(frm, to, actor)` + `TaskStatus` StrEnum → **producer:task-2168** (W2 τ2) /
  **producer:task-2163** (τ1, transitive). Extent = the transition legality set. **PASS.**
- `outcome_allows_status(outcome, status)` → **producer:task-2168** ("the map W9's
  WorkflowStateMachine consumes", task 2168 desc — exact extent match). **PASS.**
- `WorkflowState` / `WorkflowOutcome` enums (relocated) → `grep:workflow.py:324` / `:337`. **PASS.**
- `_enter_phase` (state-set site the machine replaces) → `grep:workflow.py:1463` (sets `state`
  `:1479`). **PASS.**
- 7744 "already DONE, ignoring late blocked" hand-guard (becomes a machine property) →
  `grep:workflow.py:7744`. **PASS.**
- `WorkflowStateMachine` type → **W9-new** (this task's deliverable). **PASS.**
- G6 (rejection — illegal transition raises): the raising machine is **W9-authored**, so the
  rejection substrate *is* this task's deliverable — RED test authors an illegal transition and
  observes the raise. Not a pre-existing-substrate rejection claim → G6 branch-4 satisfied by
  construction. **PASS.**

## γ — `TerminalReport`; delete `_last_block_*`; consistency assert
- `_last_block_reason/_detail/_phase` defn (deleted) → `grep:workflow.py:775-777`. **PASS.**
- harness read of `_last_block_*` (repointed to the report) → `grep:harness.py:5189-5191`. **PASS.**
- `outcome_allows_status` (exit consistency assert) → **producer:task-2168**. **PASS.**
- `FailureCategory` (TerminalReport.category type) → **producer:task-2123** (W7 α). **PASS.**
- `_mark_blocked` (constructs the report) → `grep:workflow.py:7702`. **PASS.**
- `TerminalReport` dataclass → **W9-new**. **PASS.**
- G6 (field-population — TerminalReport fields sampled by the harness): outcome/reason/phase/detail/
  category are **populated on the production path by this task's own `_mark_blocked`/`run()`**, not
  a declared-but-unpopulated downstream field → field-population sub-check satisfied. **PASS.**

## δ — `StewardOutcome` sum type
- steward `_handle_escalation` (produces the outcome) → `grep:steward.py:294`. **PASS.**
- `_await_steward_completion` (returns the outcome) → `grep:workflow.py:8739`. **PASS.**
- timestamp-window dismissal heuristic (deleted) → `grep:workflow.py:7913` + `:7980-7998`. **PASS.**
- `TaskStatus` (TERMINAL_DECISION.new_status type) → **producer:task-2163** (via 2168 dep). **PASS.**
- `StewardOutcome` union + the in-process channel → **W9-new**. **PASS.**
- G6 (dependency-direction — `INTERRUPTED.wip_commits_present`): derived from the worktree by **this
  task** (not owned by a downstream leaf) → branch-3 satisfied. **PASS.** *(steward.py co-touch with
  W4 η is a file-lock coordination, not a consumed capability — no dep, per PRD §7.)*

## ε — `classify_failure → BlockDisposition` table
- `InvocationOutcome` → **producer:task-2127** (W4 α). **PASS.**
- `AgentFailureKind`/`AgentFailureClass` → **producer:task-2129** (W4 β; "the public types W9
  consumes"). **PASS.**
- `FailureCategory` (BlockDisposition.category) → **producer:task-2123** (W7 α). **PASS.**
- `BlockRecord` + `BlockClass` (built in `_mark_blocked`) → **producer:task-2138** (W7 ζ). **PASS.**
- the 7 caught exception types → `grep:workflow.py:1919/1929/1957/2014/2070/2098/2114`. **PASS.**
- sibling cap-catch sites → `grep:steward.py:653`, `grep:review_checkpoint.py:192`,
  `grep:dry_run_unblock.py:341`. **PASS.**
- `BlockDisposition` + `classify_failure` → **W9-new**. **PASS.**
- G6 (rejection/completeness — every exception type has a row): the completeness assert is
  **W9-authored** over the exported exception types in git_ops/verify/cli_invoke/usage_gate (a real,
  enumerable set on main), not an invented claim → branch-4 satisfied. **PASS.**

## ζ — `RetryLedger`
- `RetryLedger` typed sub-model / `shared/task_metadata.py` schema home → **producer:task-2158**
  (W3 α). **PASS — with G6 extent caveat:** the program seam map does not list RetryLedger among
  W3's enumerated sub-models (possible `producer-extent-short`). **Resolution (PRD Open-Q Q2, the
  brief's fallback):** consume W3's if it covers all 8 fields; else define the typed accessor in
  `workflow_types.py` and register via W3's extension point. Dep on 2158 holds either way → not a
  FAIL.
- the 8 counter keys → `grep:workflow.py:3303-3330` / `:3424-3454` / `:3523-3536` / `:1030`. **PASS.**
- `_merge_fresh_metadata` RMW → `grep:workflow.py:3233`. **PASS.**
- signature helpers `_normalize_cause_hint` / `_compute_merge_outcome_signature` →
  `grep:workflow.py:367` / `:394`. **PASS.**
- G6 (premise — persist escalates): a behavioural invariant (RL-1), not a numeric claim. **PASS.**

## η — Capability wiring as a role property
- `AgentRole` dataclass + `allowed_tools` → `grep:agents/roles.py:8` / `:11`. **PASS.**
- `_MCP_CONFIG_ROLES` / `_PLAN_TOOLS_ROLES` tuples (deleted) → `grep:workflow.py:105-114`. **PASS.**
- `_invoke` (derives gates from the role) → `grep:workflow.py:6876` (gates 6892/6913/6925/6938).
  **PASS.**
- `mcp_families` / `sandboxed` AgentRole fields → **W9-new**. **PASS.**
- G6 (rejection — import assert fires for a family-less role): the roles.py import-time assertion is
  **W9-authored**; RED test declares a role with a plan-tools tool but no family and observes the
  import failure → branch-4 satisfied by construction. **PASS.**

## θ — `CancellationScope` supervisor
- `_cancel_event` → `grep:workflow.py:727`. **PASS.**
- `_await_cancellable` → `grep:workflow.py:8642`. **PASS.**
- `_handle_soft_cancel` → `grep:workflow.py:8691`. **PASS.**
- `sys.exc_info()` hard-cancel sniff (replaced) → `grep:workflow.py:2154`. **PASS.**
- harness B2 `synthetic_cancel` dual-guard (replaced) → `grep:harness.py:489` / `:5239`. **PASS.**
- `TerminalReport` (cancellation produces `TerminalReport(CANCELLED)`) → **producer:task-γ**
  (in-batch dep). **PASS.**
- `CancellationScope` + `WorkflowCancelled` → **W9-new**. **PASS.**

## ι — B+H integration gate (seams 1-7)
- all seam capabilities → **producer:task-{α,β,γ,δ,ε,ζ,η}** (in-batch transitive dep closure; ι
  depends on η which transitively pulls the spine). **PASS.**
- G2 leaf signal: the two-way boundary suite (rows 1-12) exercises each seam through the product's
  own paths (workflow `run()` return, harness consumption, roles.py import). **PASS.**

---

### Summary
Zero FAIL bindings. Two G6 caveats, both resolved to PASS: (ζ) RetryLedger extent → brief's
typed-accessor fallback; (α/β/ε/η) the enforcement/rejection substrates are **W9-authored** (the
tasks build the raising machine / import-assert / completeness check), so they are not
pre-existing-substrate rejection claims and cannot be `rejection-absent`. All cross-batch producers
(2153, 2168/2163, 2127, 2129, 2123, 2138, 2158) are **verified filed** and wired as hard deps. The
manifest is queue-clear.
