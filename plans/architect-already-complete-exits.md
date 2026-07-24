# PRD: Architect exits for already-complete work (A1 skip-EXECUTE + C deterministic merge)

**Milestone:** orchestrator ergonomics · **Status:** deferred (design complete, awaiting decompose) · **Approach:** B+H · **Date:** 2026-07-24
**Origin:** esc-3011-1 (human-gate memory-consolidation of the `architect_report_task_already_done_main_reachability` cluster). While resolving that gate with Leo, two mechanism gaps surfaced that the canonical SOP memories *document* but do not *fix*. This PRD fixes them.

## Goal

When an architect is re-invoked on a task whose branch already carries the **complete, green** implementation, the orchestrator should take the cheapest correct path to main instead of burning agent turns or a human round-trip:

- **A1 — skip the implementer turn.** Today a re-invoked architect on an already-committed-green branch must emit a normal plan whose steps are all `pending`, so EXECUTE runs one **near-noop implementer LLM turn** (it is briefed by `_detect_tip_wip_commits` to *attribute* the existing commit, not re-derive it) before VERIFY→MERGE. Give the architect authority to mark those steps satisfied at authoring time so EXECUTE is skipped entirely.
- **C — land the merge-landing desync deterministically.** When the branch is a **clean fast-forward of main + verify already PASSED + review PASS**, the only thing missing is the physical merge. Today the architect's advised exit is `report_unactionable_task` → opens an L1 → a human lands the fast-forward (~100k tokens + latency). Worse, that open L1 **vetoes** the *existing* verified-green auto-merge reaper (`harness._maybe_submit_stranded_verified_green` fires only when there is no open escalation). Give the architect a deterministic merge-submit exit, and relax the reaper veto so a merge-remediable escalation no longer blocks the auto-merge.

**User-observable surfaces:** (A1) a re-invoked architect on an already-green branch produces a plan with all steps `done`, the workflow event log shows **zero EXECUTE iterations**, and the branch lands via the normal VERIFY→MERGE flow. (C) a clean-FF/verified/reviewed task **lands on main via the merge queue with no L1 escalation filed** — the exact task-3000 shape (which was sitting `blocked` awaiting a human at the time this PRD was written) auto-lands.

## Background

Resolved in esc-3011-1, grounded in current code (all citations verified 2026-07-24):

- `report_task_already_done` requires the cited commit be reachable from main (`_handle_already_done_report`, `workflow.py:5211`, `git merge-base --is-ancestor`); a branch-only commit hard-fails as an architect mistake.
- EXECUTE is a `while self.artifacts.get_pending_steps():` loop (`workflow.py:5931`); with **no pending steps** the body never runs and it returns immediately (`workflow.py:6229`) → VERIFY. But the architect's plan-tools only ever create steps with `status:'pending'` (`mcp/plan_tools.py:184`); only the implementer's `mark_step_done` sets `done`. **So today the architect has no authority to emit an all-satisfied plan.** (The empty-skip substrate exists; the authoring surface does not — that is A1's deliverable.)
- `enqueue_merge_request` / `register_and_enqueue_merge_request` (`merge_queue.py:3679`/`:3784`) are **stage-agnostic** free functions; the merge worker is a separate asyncio task with **no scheduler-pause gate** (`harness.py:8397`, `merge_queue.py:9371`) and **re-runs scoped verify before landing** (`merge_queue.py:2392`). `harness._maybe_submit_stranded_verified_green` (`harness.py:4324`) already uses them for exactly this shape (`source='stranded-reaper'`) — but only from the stranded reaper, and only when `action == RE_FILE_ESCALATION`, which the resolver produces **only when `not report.open_escalations`** (`harness.py:4792`/`4812`, call-site `:5087`). So `report_unactionable_task`'s L1 (`workflow.py:5320`) is precisely what vetoes the auto-merge.

The canonical SOP for this decision now lives in three Mem0 procedural_knowledge canonicals under topic `architect_report_task_already_done_main_reachability`: `7a9b4757` (`complete_not_on_main_replan`), `74542b31` (`partial_uncommitted_wip_restore_and_replan`), `8c7b99dc` (`merge_landing_desync_deterministic_merge`). Both `7a9b4757` and `8c7b99dc` carry an explicit **"KNOWN IMPROVEMENT IN FLIGHT (esc-3011-1 follow-up /prd)"** pointer that this PRD's Phase 4 removes once the mechanisms land.

## Activation status

Ready to decompose now. No blocking prerequisites — every substrate the mechanisms build on (EXECUTE empty-skip, `enqueue_merge_request` under pause + re-verify, `detect_verified_green`) exists on main today. The only gates on landing are internal to this PRD's DAG.

## Sketch of approach

**A1 (pre-satisfied plan steps).** Add a plan-tool `mark_step_committed(step_id, sha)` that sets a step's `status='done'` at authoring time and tags its description `[COMMITTED <shortsha>]`. A plan whose every step is thus marked yields `get_pending_steps() == []`, so the existing EXECUTE loop is a total no-op and the branch flows PLAN→VERIFY→REVIEW→MERGE. VERIFY remains the real gate — a falsely pre-satisfied step surfaces as a VERIFY failure, not a silent skip. Provenance is retained per-step via the `[COMMITTED <sha>]` tag (mirrors the `lost_plan_reconstruction` canonical `974b0adb`).

**C (new merge-submit exit + scoped veto relax).** Two coordinated changes:
1. A new architect exit `report_ready_to_merge(commit, evidence)` writing `.task/ready_to_merge.json`; handler `_handle_ready_to_merge_report` **validates the desync predicate first-hand** (clean-FF both directions + latest `workflow_verify` passed + review PASS in routing history), then builds a `MergeRequest` and calls `enqueue_merge_request(..., source='architect-desync')`. The merge worker's own scoped re-verify is the sole safety gate (identical posture to the stranded reaper). Merge-success → mark done via the existing `found_on_main` path; merge-fail / predicate-fail → `_mark_blocked` (architect mistake, no bypass).
2. **Relax the reaper veto** so `_maybe_submit_stranded_verified_green` (and its `RE_FILE_ESCALATION` gate) fires when the *only* open escalation(s) belong to a **merge-remediable class** — the auto-merge should not be blocked by the very escalation that exists to request the merge. Escalations that signal a *human concern* (design_concern, task_failure, review_issues, operator-action, …) continue to veto. This closes the anti-synergy for tasks that reach `blocked` via any path, not just the architect exit.

## Resolved design decisions

1. **A1 mechanism = pre-satisfied plan steps** (not a `verify_only` plan flag, not a separate `report_ready_for_verify` exit). Reuses the existing empty-pending-steps EXECUTE skip, keeps per-step `[COMMITTED <sha>]` provenance, and leaves VERIFY as the gate. (Leo, 2026-07-24.)
2. **C = new `report_ready_to_merge` exit + scoped veto relax** (not exit-only; not folded into `report_unactionable_task`). The exit keeps `report_unactionable`'s "no valid plan" semantics clean; the scoped veto relax fixes the anti-synergy for the reaper path too. **Scope of the relax is deliberately narrow** — merge-remediable escalation classes only; a human-concern escalation still vetoes (preserves the stranding PRD's "never bypasses" safety posture). (Leo, 2026-07-24.)
3. **One PRD, B+H, with a post-landing SOP phase.** A1 + C share the diagnosis and the SOP updates; the merge lane is a load-bearing seam (blast radius ≥3: `plan_tools`, `workflow`, `harness`, `merge_queue`) so contracts + two-way boundary tests are authored up front. (Leo, 2026-07-24.)

## Pre-conditions for activating

None external. Internal DAG only (see Decomposition plan). The Phase-4 SOP-update leaves are hard-gated on the corresponding implementation leaves landing (Leo's requirement: "update the SOP memories and prompts/skills **when the implementations have landed**").

## Cross-PRD relationship

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/stranding-remediation-scheduler-ergonomics-prd.md` | extends | `stranded_verified_green.detect_verified_green` fire-condition + `harness._maybe_submit_stranded_verified_green` open-escalation veto (`harness.py:4792`/`5087`) | **this-prd** | queued (leaf δ) |
| `plans/async-merge-request-prd.md` / merge-queue family | consumes | `enqueue_merge_request` / `register_and_enqueue_merge_request` (`merge_queue.py:3679`) — used as-is, no change to the queue contract | merge-queue PRDs (unchanged) | wired (existing) |

No reciprocal-ownership ambiguity: this PRD unambiguously owns both the new architect exit and the veto-predicate change; the merge-queue enqueue API is consumed unchanged.

## Contract section (B+H)

### A1 — `mark_step_committed`
- **Signature (plan-tools MCP):** `mark_step_committed(step_id: str, sha: str) -> {ok: bool, step_id, status}`.
- **Effect:** sets the named step's `status = 'done'` and prepends `[COMMITTED <sha[:12]>]` to its description. Idempotent per (step_id, sha).
- **Authoring guard:** `sha` must resolve on the current branch (`git cat-file -e <sha>`), else the tool errors (architect cannot pre-satisfy against a non-existent commit). It does **not** attempt to prove the step's *semantics* — VERIFY is the semantic gate.
- **Invariant:** a plan is valid iff (existing rules) non-empty `steps` + `_finalized_at` (via `confirm_plan`) + non-empty `files`. `mark_step_committed` does not relax any of these; it only flips `status`. `confirm_plan` must accept an all-`done` plan.
- **Downstream invariant (unchanged code):** `get_pending_steps()` (`artifacts.py:706`) excludes `done` steps ⇒ all-done plan ⇒ EXECUTE no-op (`workflow.py:5931→6229`) ⇒ VERIFY. The post-PLAN `_check_branch_on_main` guard and the phantom-done gate still apply (branch modifies declared files vs base ⇒ pass).

### C — `report_ready_to_merge` + handler
- **Exit signature (architect MCP):** `report_ready_to_merge(commit: str, evidence: str)` → writes `.task/ready_to_merge.json`.
- **Handler `_handle_ready_to_merge_report` (workflow.py, sibling to `_handle_already_done_report`):** validates ALL of, first-hand:
  - clean FF: `git merge-base --is-ancestor main HEAD` **true** AND `git merge-base --is-ancestor HEAD main` **false**;
  - latest `workflow_verify` event for the task has `passed` truthy with a `tip_sha` == branch tip (reuse `stranded_verified_green.last_verified_green_tip`);
  - review PASS present in routing history.
  On pass → build `MergeRequest(task_id, branch, worktree, pre_rebased=False, snapshot_tip=<tip>, …)`, `enqueue_merge_request(..., source='architect-desync')`, register a done-callback: merge-success → `scheduler.mark_done(kind='found_on_main', sha=<landed>)`; merge-fail/transient → `_mark_blocked` (no human bypass beyond the queue's own verify). On any validation-fail → `_mark_blocked` (architect mistake, mirrors `already_done` reject path). **Idempotency:** a durable `metadata.architect_merge_request` marker (mirrors `stranded_merge_request`, `harness.py:4372`) prevents a re-invoked architect from double-enqueuing the same tip.
- **Invariant:** the exit NEVER lands main itself — it only enqueues; the merge worker's scoped re-verify (`merge_queue.py:2392`) is the sole gate that advances main.

### C — veto-predicate relax
- **New predicate:** define `MERGE_REMEDIABLE_ESC_CATEGORIES` (e.g. `{stranded_blocked, merge_landing_desync}` — exact set finalized at impl time from the live category vocabulary). The `RE_FILE_ESCALATION` classification (`harness.py:4792`/`4812`) and the `_maybe_submit_stranded_verified_green` call-site (`:5087`) fire when open escalations are **empty OR all of a merge-remediable class**. Any open escalation of a non-remediable class ⇒ unchanged veto (`LEAVE`).
- **Invariant (safety-preserving):** the relax NEVER auto-merges a task carrying a human-concern escalation; and even for a remediable-class match, `detect_verified_green`'s three-part shape check (ASSIGNED lane + tip==last-passed-verify + plan all-DONE) and the merge queue's re-verify remain the gates. "Never bypasses" (stranding PRD §2.2) is preserved.

## Boundary-test sketch (B+H)

| # | Scenario | Preconditions | Postconditions (asserted) |
|---|---|---|---|
| A1-1 | All-done plan skips EXECUTE | branch carries committed complete impl; architect emits plan, all steps `mark_step_committed` | 0 EXECUTE iterations in event log; task advances to VERIFY; branch lands |
| A1-2 | Partial plan still implements | some steps committed, some pending | implementer runs only the pending steps; committed steps not re-derived |
| A1-3 | False pre-satisfy is caught | a step marked committed but code doesn't satisfy it | VERIFY fails ⇒ task blocks (no silent green) |
| C-1 | Architect exit lands desync | clean-FF + verify passed + review PASS | `MergeRequest` enqueued `source='architect-desync'`; branch lands on main; **no L1 filed**; task done via found_on_main |
| C-2 | Exit rejects non-desync | NOT clean-FF (or verify not passed, or no review PASS) | `_mark_blocked`; no MergeRequest enqueued |
| C-3 | Queue re-verify still gates | architect exit fires but branch fails scoped re-verify | main does NOT advance; task blocked; failure surfaced |
| C-4 | Idempotent exit | architect re-invoked on same tip after enqueue | no second MergeRequest for the same tip (marker) |
| D-1 | Veto relaxes for remediable esc | verified-green blocked task; only open esc ∈ merge-remediable class | `_maybe_submit_stranded_verified_green` submits (`source='stranded-reaper'`) instead of re-filing |
| D-2 | Veto holds for human concern | verified-green blocked task; open esc = design_concern/task_failure | NO auto-submit; unchanged re-file/wait path |

## Decomposition plan

**Phase 1 — foundation + contracts**
- **α** `mark_step_committed` plan-tool (`orchestrator/src/orchestrator/mcp/plan_tools.py`, `artifacts.py`). *Signal (intermediate, unlocks γ):* calling it flips step `status→done` + tags `[COMMITTED <sha>]`; `get_pending_steps()` excludes it; `confirm_plan` accepts an all-done plan.
- **β** `report_ready_to_merge` exit + `.task/ready_to_merge.json` + `_handle_ready_to_merge_report` (`workflow.py`, artifact schema). *Signal (leaf / integration-gate):* on a clean-FF+verify-passed+review-PASS task, the exit enqueues a `MergeRequest` (`source='architect-desync'`) that lands the branch on main with **no L1 filed** (boundary tests C-1..C-4).

**Phase 2 — A1 vertical slice**
- **γ** Architect behavior wiring: a re-invoked architect on an already-committed-green branch emits an all-`mark_step_committed` plan → EXECUTE no-op → VERIFY→MERGE. Depends **α**. *Signal (leaf):* end-to-end run shows **0 EXECUTE iterations** in the event log and the branch lands (boundary tests A1-1..A1-3).

**Phase 3 — C vertical slice (reaper side)**
- **δ** Veto relax: `detect_verified_green` `RE_FILE_ESCALATION`/`_maybe_submit_stranded_verified_green` fire under merge-remediable-only open escalations (`harness.py`). Relates **β**. *Signal (leaf):* a verified-green blocked task with only a merge-remediable esc auto-submits (`source='stranded-reaper'`); with a human-concern esc it still vetoes (boundary tests D-1, D-2).

**Phase 4 — post-landing SOP correction (hard-gated on the impl leaves)**
- **ε** Update the canonical SOP memories: rewrite `7a9b4757` (`complete_not_on_main_replan`) to state EXECUTE is now skipped via `mark_step_committed` (drop "near-noop / improvement in flight"), and `8c7b99dc` (`merge_landing_desync_deterministic_merge`) to prescribe `report_ready_to_merge` + note the relaxed veto (drop the report_unactionable→L1 caveat). `execution_class=operational` (Mem0 write, no code). Depends **γ, β, δ**. *Signal (leaf):* `get_memory_by_id` shows the updated canonical text referencing the shipped exits with no "in flight" pointer.
- **ζ** Update the architect prompt/skill exit enumeration to teach: use `mark_step_committed` for already-committed-green steps (skip implement); use `report_ready_to_merge` for the merge-landing desync (not `report_unactionable`). Depends **α, β**. *Signal (leaf):* the architect prompt/skill source enumerates both new surfaces.

DAG: α→γ; α→ζ, β→ζ; β,γ,δ→ε. β and δ are the two halves of C. α, β, δ are independently landable.

## Out of scope

- Changing the merge-queue contract or its scoped-verify gate (consumed unchanged).
- A `verify_only` plan flag or a separate `report_ready_for_verify` architect exit (explicitly rejected in favor of `mark_step_committed`).
- Broadening the veto relax beyond merge-remediable escalation classes (human-concern escalations must still veto).
- The A2-partial / restore-and-replan path (`74542b31`) — unaffected; genuinely-partial WIP still plans RED-first.

## Open questions (tactical — decide at impl time)

1. **Task status while the architect-exit merge is in flight (β).** Mirror the stranded path (leave `blocked`, mark done via found_on_main callback) vs a dedicated transient status. *Suggested:* reuse the stranded path's leave-blocked + found_on_main callback for symmetry. Decide during β.
2. **Exact membership of `MERGE_REMEDIABLE_ESC_CATEGORIES` (δ).** Derive from the live category vocabulary at impl time; start minimal (`stranded_blocked`, and a `merge_landing_desync` category if β introduces one) and widen only with evidence. Decide during δ.
3. **Whether β should emit its own `merge_landing_desync` escalation category** (so δ's predicate has a clean class to match) or reuse `stranded_blocked`. Decide during β/δ jointly.

## Design-invariants (G7 advisory walk — docs/legibility/design-invariants.md)

- `corroborate-before-acting`: **satisfied** — both new paths act only after first-hand corroboration (β validates the desync predicate directly; the merge queue re-verifies; δ retains `detect_verified_green`'s three-part shape check) rather than on a snapshot claim.
- `contracts-machine-checked`: **satisfied** — the new exits are typed JSON artifacts with schemas, not prose; the veto predicate is a code-level category set.
- `no-lockstep-duplication`: **satisfied** — β reuses `enqueue_merge_request` and `last_verified_green_tip`; α reuses the existing EXECUTE empty-skip; no merge logic is duplicated.
- `storm-escape-required`: **addressed** — β's durable `architect_merge_request` idempotency marker + the queue's own dedup prevent a re-invoked architect from storming duplicate merge requests.
- `structured-facts-at-failure`: to confirm at impl — handlers emit structured events (enqueue source, validation-fail reason) rather than log-scrape.
