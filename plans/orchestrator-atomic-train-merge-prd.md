# PRD — Orchestrator support for atomic multi-crate "stacked train" landings

**Status:** active (greenfield design), 2026-05-28
**Slug:** `orchestrator-atomic-train-merge`
**Approach:** B + H (contract + two-way boundary-tests). See § Contract and § Boundary-test sketch.

## 1. Goal

The dark-factory orchestrator can land a declared, linearly-ordered group of tasks ("train") onto `main` as a **single green→green merge** — never reding main intermediately, never requiring manual task-graph surgery — while preserving per-member granularity (each member is independently planned, executed, and reviewed).

User-observable surface, in one sentence: a PRD whose decomposition declares α/β/γ as `train_id = "T1"` with `train_order` 0/1/2 lands as one merge commit on `main`, with all three tasks flipping to `done` in lockstep with valid `kind: merged` provenance, and no intermediate red-main window.

## 2. Background — the 3997 incident

Reify task 3997 (escalation `esc-3997-12`, 2026-05-27) attempted a foundation refactor that removed a widely-imported type (`reify-ir::OptimizationObjective` → `ObjectiveSet`). The work was decomposed across three crates:

- **α** — `reify-ir` type change (the removal).
- **β** — `reify-compiler` lowering update (must compile against α).
- **γ** — `reify-eval` / `reify-constraints` solver thread-through (must compile against β).

The chain was intentionally main-breaking *in the middle*: α alone reds `main` because β/γ's crates still import the removed symbol; the workspace only compiles green again once all three land together.

The orchestrator's current model produced a circular deadlock:

- **α** built green own-crate (`cargo test -p reify-ir`: 687 pass) but could not reach `done` — the done-provenance gate (correctly) refuses to fabricate main-provenance for a branch that isn't an ancestor of `main`.
- **α** could not merge alone — that would red `main` across all worktrees (high blast radius).
- **β** could not dispatch — its dependency gate requires α to be `done` (or `cancelled`).
- ∴ neither could move; the verify harness re-dispatched α 4–5× on the same pre-accepted workspace failure, burning compute.

The incident was resolved by **collapsing α+β+γ into one task** (Option C: rewire 8 dependents, cancel β/γ, run the combined work as a single workspace-green unit). Correct, but it sacrificed granularity and required manual surgery — the kind of escape hatch one keeps but doesn't celebrate.

Reify's memory note `procedural_atomic_multicrate_refactor_collapse` documents the collapse playbook. This PRD makes the **architecturally-preferred path** — Option A, the "stacked train" — feasible.

## 3. Non-goals and constraints (the hard rules this PRD honours)

- **Do NOT weaken the done-provenance gate** for normal (non-trained) tasks. The gate stays bit-identical for non-train work.
- **Never leave `main` red** at any committed point. The whole value-proposition vs. the rejected "accept-red-main" alternative.
- **Degrade gracefully.** Absent a declared train, every orchestrator behaviour is byte-identical to today.
- **Linear only for v1.** Arbitrary N-deep DAG / fan-out / fan-in trains are out of scope.

## 4. The key architectural insight (why this is tractable)

Members are **stacked** — β's branch is created off α's branch tip, γ off β's tip. By construction, β's branch contains α's commits as ancestors, and γ's branch contains both α's and β's. Therefore:

1. The **"group-tip workspace"** at any point during the holding window IS the latest member's own branch tip with `--workspace`. No separate integration branch is needed for linear-stacked v1 — the stack lineage *is* the integration history.

2. The **group merge** is mechanically a single `git merge --no-ff <train-tip-branch>` of the *tip* member's branch into main. That single merge brings α+β+γ's commits onto main in one CAS-advance. Atomicity follows from the existing `advance_main` compare-and-swap (`orchestrator/src/orchestrator/git_ops.py:1216-1428`).

3. After the merge, every member's tip commit IS an ancestor of `main`. The done-provenance ancestor check (`git_ops.py:1273-1291`) passes for all members with normal `kind: merged` provenance — **the gate is satisfied, not bypassed**. We are deferring entry to `done`, not weakening the gate.

This insight collapses what looks like a five-capability change into a much smaller set of focused mechanisms.

## 5. Sketch of approach

A train is declared by attaching `metadata.train = {id, order, members?}` to each member task at filing time. The orchestrator reads the train metadata and applies five focused changes:

1. **New `merge-deferred` status** in fused-memory's vocabulary — the holding state between own-verify-green and group-merge.
2. **Train-aware dispatch gate** — `_deps_satisfied` accepts `merge-deferred` predecessors *if and only if* the dependent shares the same `train_id`.
3. **Sibling-tip worktree base** — when `train_order > 0`, the worktree branches from the prior member's branch tip rather than `origin/main`.
4. **Per-member group-tip verify** — train members run workspace-wide verify (`--workspace`) against their own branch tip, overriding any per-module file-scoping.
5. **Group merge** — a new `GroupMergeRequest` carrying the train_id + ordered member task_ids. The merge worker merges the tip branch, runs the existing post-merge workspace verify, CAS-advances, then flips all members to `done` with `kind: merged` provenance pointing at the merge commit.

Plus one cross-cutting loop-guard fix (§ 11.5) that benefits non-train tasks too.

## 6. Resolved design decisions

The three load-bearing decisions made during this design session:

### D1 — `merge-deferred` is a first-class status in fused-memory (not a metadata marker on `blocked`)

A new status value `merge-deferred` is added to fused-memory's accepted task-status vocabulary. It is **not terminal** (not in `TERMINAL_STATUSES`) and **is** workflow-preserved (added to `WORKFLOW_PRESERVE_STATUSES`).

Trade-off accepted: the change has high blast radius into fused-memory (TaskInterceptor, reconciliation, terminal-status sets, the dashboard, the done-provenance gate's "current status" classification). We accept the cost in exchange for clean semantics — no conflation with failure-`blocked` in the steward, reaper, dashboard, or escalation paths, all of which today treat `blocked` as needs-attention.

### D2 — Per-member verify gate is the group-tip workspace (not own-crate)

Each train member runs workspace-wide verify against its own branch tip. In linear-stacked v1 this is the cheapest implementation of "group-tip workspace" because the branch tip already contains the prefix.

Trade-off accepted: a full workspace build per member is slower than per-crate, but it catches cross-crate breakage at the earliest member that could detect it, and the workspace-green guarantee at every step lines up with the architectural invariant we want (the merge gate isn't the only place quality is checked).

### D3 — Derail policy is "park prefix, escalate member only" (not auto-collapse, not whole-train escalate)

When a member hits a hard, un-fixable failure (loop-guard tripped), only that member is escalated to L2. Upstream `merge-deferred` siblings stay parked — their worktrees and branches are preserved by reaper guards. The train resumes once the failing member is resolved (re-verified to `merge-deferred`).

Trade-off accepted: less aggressive than auto-collapse, more surgical than whole-train escalation. A long-parked train can hold worktrees indefinitely; the operator (via escalation triage) is responsible for deciding "collapse this train" if it stays parked too long.

## 7. Activation status / pre-conditions

This PRD is **self-contained** — it has no external pre-requisite PRDs.

Substrate it depends on (all verified to exist):

- `submit_task` accepts arbitrary `metadata` dicts — confirmed (no schema change needed for train metadata).
- `set_task_status` writes status + done_provenance — confirmed (`scheduler.py:1112-1240`).
- Merge queue is a single-worker queue accepting MergeRequest objects with futures — confirmed (`orchestrator/src/orchestrator/merge_queue.py:1-31`, `603-626`).
- Post-merge workspace verify runs inside the merge queue before CAS-advance — confirmed (memory: "Cargo workspace crate scoping should not apply to post-merge verify in merge_queue.py", 2026-04-08).
- `advance_main` is a CAS ref-update — confirmed (`git_ops.py:1216-1428`).
- `is_ancestor` check uses `git merge-base --is-ancestor` — confirmed (`git_ops.py:896-902`).
- Worktree creation can take an explicit start-ref at the `git worktree add` level — confirmed (the orchestrator's `_freshen_main` currently hardcodes main but `git worktree add <path> -b <branch> <start-ref>` accepts any ref).
- Per-module `test_command` overrides exist (`ModuleConfig.test_command`, `config.py:427-443`) — confirmed; the train-member override layers above this.

**G3 substrate verifier (manual mode):** no novel substrate is assumed. Every capability above is either present today or is the explicit deliverable of one of this PRD's tasks. The single "new vocabulary entry" — the `merge-deferred` status string — is a value-add to existing machinery (the fused-memory status enum), not a fictional capability.

## 8. Cross-PRD relationship

| Other PRD / surface | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| Reify refactor PRDs (e.g. constraint-solver-completion successors) | Consumes | `metadata.train = {id, order}` convention on member tasks | this PRD (defines the schema; consumer PRDs follow) | wired-on-publish |
| `/prd` decompose mode (skill) | Consumes (optional) | Ergonomic emission of train metadata when a PRD declares a train | future PRD | out-of-scope (see § 12) |
| fused-memory status vocabulary | Produces | New status value `merge-deferred` + workflow-preserve membership | this PRD (intra-system; not a separate PRD) | queued (Phase 1) |
| Dashboard | Consumes | New status pill / column for `merge-deferred` train members | this PRD (companion task) | queued (Phase 6) |

No reciprocal ownership ambiguity. The metadata-schema is published by this PRD; downstream consumer PRDs (reify refactors) reference it by convention.

## 9. Contract — signatures and invariants

### 9.1 Train metadata schema (on member tasks)

```python
# stored in fused-memory task.metadata["train"]
TrainMembership = TypedDict("TrainMembership", {
    "id": str,            # train identifier, kebab-case, e.g. "objectiveset-refactor"
    "order": int,         # 0-indexed position in linear stack (0 = root member)
    "members": list[str] | None,  # OPTIONAL cache of member task IDs in order;
                                  # absent → orchestrator discovers via status_map sweep
})
```

**Invariants:**

- All members of a single train share the same `id`.
- Within a train, `order` values are 0..N-1 with no gaps and no duplicates.
- Order is purely positional; intra-train dependencies are declared via the existing `dependencies` field — member `order=k` MUST `depends_on` exactly the member with `order=k-1` (no transitive shortcuts).
- A task carrying `metadata.train` MUST NOT have non-train tasks as dependencies whose status is `merge-deferred` (only intra-train deps may rely on the merge-deferred state).

### 9.2 Status vocabulary extension (fused-memory)

```python
# in fused-memory's status validator:
ACCEPTED_STATUSES = {"pending", "in-progress", "done", "blocked",
                     "cancelled", "deferred", "merge-deferred"}

# in orchestrator/src/orchestrator/task_status.py:
TERMINAL_STATUSES = frozenset({"done", "cancelled"})              # UNCHANGED
WORKFLOW_PRESERVE_STATUSES = frozenset(
    {"done", "cancelled", "deferred", "blocked", "merge-deferred"}  # ADD
)
```

**Invariants:**

- `merge-deferred` is **not terminal** — it transitions to `done` (via group-merge) or back to `in-progress` (via re-dispatch after a sibling-driven failure).
- `merge-deferred` is **workflow-preserved** — the workflow does not re-execute a `merge-deferred` task on its own.
- Entering `merge-deferred` does **not** require done-provenance — the gate only fires on transitions TO `done`.

### 9.3 Dispatch gate (`scheduler._deps_satisfied`)

```python
def _deps_satisfied(self, task: dict, status_map: dict[str, str]) -> bool:
    task_train_id = (task.get("metadata") or {}).get("train", {}).get("id")
    deps = task.get("dependencies", [])
    for d in deps:
        dep_id = _dep_id(d)
        dep_status = status_map.get(dep_id, "unknown")
        if dep_status in TERMINAL_STATUSES:
            continue
        # NEW: intra-train allowance
        if dep_status == "merge-deferred" and task_train_id is not None:
            dep_task = self._task_lookup(dep_id)              # cached
            dep_train_id = (dep_task.get("metadata") or {})\
                              .get("train", {}).get("id")
            if dep_train_id == task_train_id:
                continue
        return False
    return True
```

**Invariants:**

- A non-train task with a train-member predecessor in `merge-deferred` does NOT dispatch — it waits for the predecessor to reach `done`.
- An intra-train dependent dispatches when its immediate predecessor (order = self.order - 1) is in `merge-deferred` OR `done`/`cancelled`.

### 9.4 Worktree base-ref (`git_ops.create_worktree`)

```python
async def create_worktree(
    self, branch_name: str, *, train: TrainMembership | None = None,
) -> WorktreeHandle:
    if train is not None and train["order"] > 0:
        predecessor = await self._train_predecessor(train["id"], train["order"])
        start_ref = await self._resolve_branch_tip(predecessor.branch)
        base_kind = "train-sibling"
    else:
        start_ref, stale_commits = await self._freshen_main()
        base_kind = "main"
    # ... rest unchanged (git worktree add <path> -b <full_branch> <start_ref>)
```

**Invariants:**

- Non-train workflows: identical behaviour to today (branch from `origin/main` via `_freshen_main`).
- Train members with `order=0`: branch from `origin/main` (same as non-train).
- Train members with `order>0`: branch from the predecessor's branch tip; predecessor MUST exist and be in `merge-deferred` or later (`_train_predecessor` raises otherwise — this is a workflow invariant, enforced by the dispatch gate).

### 9.5 Per-member verify command override

When a task's `metadata.train` is set, the verify step:

1. Bypasses any per-module file-scoping that would narrow the verify command.
2. Runs the project's workspace-wide verify command against the worktree's branch tip (e.g. `cargo test --workspace`).
3. Reports failure via the existing `run_scoped_verification` result protocol — the loop-guard (§ 9.7) reads `category` + `cause_hint` to detect signature repetition.

**Invariant:** the verify command for a train member is *not* the per-task `test_command` override (which exists for narrowing); train members use the project's verify command verbatim.

### 9.6 Group merge (`merge_queue.GroupMergeRequest`)

```python
@dataclass
class GroupMergeRequest(MergeRequest):
    """A request to merge an atomic train as one green→green unit.

    Only the tip member's branch is merged into main; by stacking,
    that single merge brings all member commits onto main atomically.
    """
    train_id: str
    member_task_ids: list[str]    # ordered, root → tip
    tip_branch: str               # same as MergeRequest.branch; tip member's
    tip_task_id: str              # convenience for callbacks
```

**Worker behaviour:**

1. Resolve all `member_task_ids` to current status. If any is NOT `merge-deferred`, reject with reason `train_incomplete` (caller resolves).
2. Rebase the tip branch onto current `main` if `main` advanced during the holding window. If rebase conflicts, reject with `train_rebase_conflict` — the failing member's worktree is the resolution surface (existing pattern).
3. `git merge --no-ff <tip_branch>` into the merge worktree; standard post-merge `.task/` scrub; existing workspace post-merge verify (already workspace-wide per the memory note).
4. CAS-advance `main` to the merge SHA via `advance_main`.
5. On success: for each `member_task_id`, call `scheduler.mark_done(task_id, kind="merged", sha=<merge_sha>, note=f"train {train_id}")`. The `note` field carries the train context for forensics.

**Invariants:**

- Members flip to `done` only after `advance_main` succeeds — never speculatively.
- If `advance_main` CAS-fails (external actor moved main), the worker re-enqueues the request front-of-queue; tip-branch may need re-rebasing.
- All members share the same `done_provenance.commit` (the merge SHA). This is intentional — they ARE on main via the same merge.

### 9.7 Loop-guard (`workflow._verify_debugfix_loop`)

Track per-task verify failure signatures across attempts; escalate to L1 after N (default 3) consecutive identical signatures with no actionable cause-hint.

```python
# new state on workflow instance:
self._failure_signature_history: list[tuple[str, str]] = []  # (category, cause_hint_norm)

# in the loop, after capturing `result`:
sig = (result.category, _normalize_cause_hint(result.cause_hint))
self._failure_signature_history.append(sig)
if (
    len(self._failure_signature_history) >= self.config.max_failure_signature_repeat
    and all(s == sig for s in self._failure_signature_history[-self.config.max_failure_signature_repeat:])
):
    logger.warning(
        "Task %s: %d consecutive identical verify failures (sig=%r) — escalating to L1",
        self.task_id, self.config.max_failure_signature_repeat, sig,
    )
    return WorkflowOutcome.BLOCKED
```

**Invariants:**

- Applies to **all** tasks, not just train members — non-train tasks benefit from the same loop-guard.
- Default threshold: `max_failure_signature_repeat = 3` (mirrors the existing `_check_*_thrash` pattern documented in memory `feedback_check_thrash_helper_pattern.md`).
- Cause-hint normalisation strips file:line numbers and timestamps so semantically identical failures match.

### 9.8 Reaper / steward / escalation guards

- **Reaper:** skip tasks where `status == "merge-deferred"`. Mirrors the existing `/unblock` worktree guard (`project_reaper_unblock_worktree_guard`).
- **Steward (`_check_*_thrash` family):** train members' `merge-deferred` → `merge-deferred` re-stamps don't count as thrash.
- **Escalation context (for park-prefix derail):** when a train member's verify escalates to L2, the escalation payload includes `train_state = {id, order, parked_members: [...], failing_member: <self>}` so the human operator sees the train shape, not just the failing task.

## 10. Boundary-test sketch (two-way, producer + consumer)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Happy-path linear 3-train | α/β/γ filed with `metadata.train = {id: "T1", order: 0/1/2}`, intra-train deps wired, fixture workspace is green for each member's branch tip with `--workspace` | One merge commit on `main` whose parents are `main`'s prior tip and γ's branch tip; α/β/γ all `done`; all three `done_provenance.commit == merge_sha`; `git log main` has zero red-main window (verified via `git rev-list HEAD~1..HEAD --merges` showing the single merge) |
| 2 | Sibling-tip worktree base | α `merge-deferred`, β dispatched | β's worktree's branch base SHA == α's branch tip SHA (queried via `git merge-base β-branch α-branch`); `git log β-branch..α-branch` empty |
| 3 | Intra-train dispatch | α `merge-deferred` (status), β `pending` with `depends_on=[α]` sharing `train_id` | β dispatches; scheduler log emits "intra-train dep satisfied" with `dep_id=α` `train_id=T1` |
| 4 | Extra-train dispatch blocked | α `merge-deferred`, δ `pending` with `depends_on=[α]` NOT sharing `train_id` (δ has no train metadata) | δ does NOT dispatch; `_deps_satisfied(δ)` returns False; scheduler log emits "dep α has status merge-deferred, need done or cancelled" |
| 5 | Group merge gate workspace verify | α/β/γ all `merge-deferred`, GroupMergeRequest enqueued | Workspace post-merge verify runs once on the merge worktree's HEAD; on green → `advance_main` succeeds and all 3 flip to `done`; on red → no advance, train rolled back, escalation emitted |
| 6 | Park-prefix derail | α/β `merge-deferred`; γ verify produces identical failure signature 3× | γ → `blocked` + L2 escalation with `train_state.parked_members = [α, β]`; α/β stay `merge-deferred`; α's and β's worktrees are NOT removed by the reaper sweep; no merge fires |
| 7 | Train resumes after derail fix | After scenario 6, γ is fixed and re-verified to `merge-deferred` | GroupMergeRequest enqueues automatically when γ transitions to `merge-deferred` and all members are `merge-deferred`; scenario 1's postconditions hold |
| 8 | Main advances during holding window | α/β `merge-deferred`, γ `in-progress`, an unrelated task lands a commit on `main` | When γ reaches `merge-deferred` and the GroupMergeRequest fires, the worker rebases the tip branch onto new `main`; if clean → merge proceeds; if conflict → reject with `train_rebase_conflict` and escalate |
| 9 | Loop-guard for non-train task | Plain task (no train metadata) hits identical verify failure signature 3× | Re-dispatch suppressed; task → `blocked` with L1 escalation; same code path as train member (no train-specific branch in loop-guard) |
| 10 | Done-provenance gate unweakened | Plain task (no train metadata) attempts `set_task_status("done", done_provenance={kind: "merged", commit: <sha-not-on-main>})` | Gate rejects with `ProvenanceValidationRejection` (existing behaviour) — regression check that this PRD did not weaken the gate |
| 11 | Non-train regression | A normal non-trained task runs the full pipeline | Every observable signal matches today's baseline (worktree base = `origin/main`, verify = per-module config, single-task merge, single done provenance, dispatch gate accepts only TERMINAL_STATUSES) |
| 12 | Train with `order=0` only (degenerate train of one) | Single member with `metadata.train.order=0` | Treated identically to a non-train task (no merge-deferred state needed; merges directly via existing MergeRequest path). Optional: emit a warning at PRD-decompose time |

## 11. Decomposition plan (B+H, 6 phases)

Tasks are labelled by Greek letter within phases for cross-reference; actual task IDs assigned at decompose time.

### Phase 1 — Foundation (fused-memory status vocabulary)

- **α₁** — **Add `merge-deferred` to fused-memory accepted-status vocabulary + interceptor.**
  - Modules touched: `fused-memory/.../task_validator.py` (or equivalent), `fused-memory/.../task_interceptor.py`, fused-memory reconciliation paths.
  - Observable signal: `submit_task(..., status="merge-deferred")` succeeds; `set_task_status(<task>, "merge-deferred")` returns success; `get_tasks` returns the member with the new status string; status appears in `get_statuses` output.
  - Prereqs: none.

- **α₂** — **Update orchestrator `WORKFLOW_PRESERVE_STATUSES` to include `merge-deferred`.**
  - Modules touched: `orchestrator/src/orchestrator/task_status.py`.
  - Observable signal: a `merge-deferred` task's metadata is preserved across the workflow's preserve-decision; unit test on `WORKFLOW_PRESERVE_STATUSES` membership.
  - Prereqs: α₁.

### Phase 2 — Dispatch & worktree base (orchestrator)

- **β₁** — **Train-aware `_deps_satisfied` extension.**
  - Modules touched: `orchestrator/src/orchestrator/scheduler.py`.
  - Observable signal: scenarios 3 + 4 + 10 (intra-train accept, extra-train reject, gate-unweakened regression). Test runs against synthetic status_map fixtures AND end-to-end against a fixture train.
  - Prereqs: α₁.

- **β₂** — **`create_worktree` sibling-tip base via `metadata.train`.**
  - Modules touched: `orchestrator/src/orchestrator/git_ops.py`, `orchestrator/src/orchestrator/scheduler.py` (call-site).
  - Observable signal: scenario 2 (sibling-tip worktree base SHA matches predecessor branch tip).
  - Prereqs: α₁, α₂.

### Phase 3 — Per-member verify & holding state (orchestrator)

- **γ₁** — **Train members run workspace verify; emit `merge-deferred` on green.**
  - Modules touched: `orchestrator/src/orchestrator/workflow.py` (verify path + post-verify state-transition), `orchestrator/src/orchestrator/verify.py` (train-member command-resolution).
  - Observable signal: in a fixture train, a member's verify with workspace-green outcome lands the member in `merge-deferred` (verified via `get_task` status read), NOT `done`.
  - Prereqs: α₂, β₂.

- **γ₂** — **Loop-guard: identical-signature escalation.**
  - Modules touched: `orchestrator/src/orchestrator/workflow.py` (`_verify_debugfix_loop`), `orchestrator/src/orchestrator/config.py` (`max_failure_signature_repeat` field).
  - Observable signal: scenarios 6 (train) + 9 (non-train). A task with N identical failure signatures escalates to L1; orchestrator log emits "consecutive identical verify failures … escalating".
  - Prereqs: none (independent; can land before or after the train work).

### Phase 4 — Group merge (merge queue)

- **δ₁** — **`GroupMergeRequest` dataclass + merge-worker handling.**
  - Modules touched: `orchestrator/src/orchestrator/merge_queue.py`, `orchestrator/src/orchestrator/git_ops.py` (helper for marking members done in lockstep).
  - Observable signal: scenarios 1 + 5 + 8 (happy path, gate-fail, main-advances-during-hold). Verified via fixture trains; assertions on `git log main`, member statuses, and `done_provenance` shape.
  - Prereqs: α₂, β₁ (dispatch can produce the holding state).

- **δ₂** — **Train-completion trigger: enqueue GroupMergeRequest when all members are `merge-deferred`.**
  - Modules touched: `orchestrator/src/orchestrator/scheduler.py` or `workflow.py` (the state-transition that detects "all members merge-deferred").
  - Observable signal: scenario 7 (train resumes after derail fix → merge fires).
  - Prereqs: δ₁.

### Phase 5 — Integration gate (B+H boundary test)

- **ε₁** — **End-to-end fixture train test + boundary-test harness.**
  - Modules touched: `orchestrator/tests/test_atomic_train_merge.py` (new), `orchestrator/tests/fixtures/atomic_train/` (new — three-crate cargo workspace fixture analogous to α/β/γ).
  - Observable signal: the boundary-test table in § 10 is implemented as discrete pytest cases; all 12 scenarios pass.
  - Prereqs: α₁, α₂, β₁, β₂, γ₁, γ₂, δ₁, δ₂.
  - **This is the train's own integration-gate task** — the leaf that demonstrates the user-observable signal stated in § 1.

### Phase 6 — Companion correction tasks

- **ζ₁** — **Reaper / steward / escalation guards for `merge-deferred`.**
  - Modules touched: `orchestrator/src/orchestrator/scheduler.py` (reaper sweep), `orchestrator/src/orchestrator/escalation.py` or equivalent.
  - Observable signal: scenario 6's "α/β worktrees NOT removed by reaper sweep" assertion; escalation payload contains `train_state` for park-prefix derails.
  - Prereqs: α₁, α₂.

- **ζ₂** — **Dashboard treatment for `merge-deferred`.**
  - Modules touched: `dashboard/redux/...` (the JSX/Python that renders task status).
  - Observable signal: in the dashboard, a `merge-deferred` task renders with a distinct pill (not red, not green; e.g. amber) and is grouped or annotated with its train_id.
  - Prereqs: α₁.

**Topological order:** α₁ → α₂ → {β₁, β₂, γ₂, ζ₁, ζ₂} → γ₁ → δ₁ → δ₂ → ε₁.

(γ₂, ζ₁, ζ₂ are leaves of independent sub-chains that can run in parallel with the train-state work.)

## 12. Out of scope

- **Arbitrary N-deep DAG / fan-out / fan-in trains.** Linear-only for v1. Trains with two-or-more roots, branching mid-stack, or join points are future work.
- **Cross-repo trains.** Members must live in the same git repository.
- **Distributed orchestrator trains.** Members must run on the same orchestrator instance (no train-spanning the reify ↔ dark-factory two-orchestrator deployment).
- **`/prd` decompose-skill ergonomic emission of train metadata.** Today, declaring a train requires manually passing `metadata.train` to `submit_task`. Ergonomic emission (the skill recognising "this PRD's decomposition is a train" and emitting the metadata automatically) is a follow-up enhancement, not a hard prerequisite — submit_task already accepts the metadata.
- **Weakening the done-provenance gate.** Explicit non-goal; preserved as constraint.
- **Squash-merge variant.** Members land as distinct commits via `--no-ff` of the tip; squashing into one commit loses per-member granularity in `git log`.

## 13. Open questions (tactical, deferred to implementation phase)

1. **Exact `max_failure_signature_repeat` default.** Suggested resolution: 3 (mirrors `_check_*_thrash` family in memory `feedback_check_thrash_helper_pattern`). Decide during γ₂.

2. **`GroupMergeRequest` as subclass vs flag on existing `MergeRequest`.** Suggested resolution: subclass — clearer dispatch in the merge worker; existing `MergeRequest` consumers don't need to learn a new field. Decide during δ₁.

3. **Should the train tip's branch name encode the train_id (e.g. `train/T1/γ`)?** Suggested resolution: leave branch names flat (`task/<slug>` as today). Trains are a metadata concept; encoding in branch names invites scripts to parse names and creates churn if a train is renamed. Decide during β₂.

4. **Cause-hint normalisation rules (`_normalize_cause_hint`).** What exactly counts as "identical" — strip file:line? strip ANSI? collapse whitespace? Suggested resolution: strip file:line, normalise whitespace, lowercase. Decide during γ₂.

5. **When a train member is in `merge-deferred` and the human runs `/unblock` on it (no failure, just operator inspection), what happens?** Suggested resolution: `/unblock` is a no-op on `merge-deferred` (or surfaces "no action needed; member is parked in a healthy state"). Decide during ε₁ if the boundary test exposes a real interaction; otherwise defer.

6. **GroupMergeRequest worker — should it re-verify each member's branch independently before assembly, or trust the prior `merge-deferred` signal?** Suggested resolution: trust the prior signal (the workspace verify already ran when each member transitioned to `merge-deferred`); the post-merge workspace verify on the assembled tip is the green-gate. Decide during δ₁ if there's a reason to re-verify (e.g. main moved → tip rebased → re-verify is implicit anyway).

7. **Telemetry: emit `train_started` / `train_member_deferred` / `train_merged` / `train_derailed` orchestrator events?** Suggested resolution: yes, mirror the existing `merge_queued`/`merge_dequeued` event pattern (task 940). Decide during δ₂.

## 14. Acceptance criteria (restating § 1 with measurable signals)

The PRD's success is observable when **all** of the following hold for the fixture train in ε₁:

1. A declared 3-task atomic group (α/β/γ) with `train_id="T1"` and `train_order` 0/1/2 lands on `main` as a **single merge commit** (verified: `git rev-list main~1..main --merges | wc -l == 1`).
2. **No intermediate red-main window** (verified: `cargo test --workspace` on `main` between any two adjacent commits returns 0 throughout — never reds).
3. **No manual task-graph surgery** (verified: the orchestrator run completes without human task-cancellation, dependency-rewiring, or `/unblock` intervention).
4. **Per-member granularity preserved in history** (verified: `git log main` shows α/β/γ's commits as distinct ancestors of the merge commit).
5. **Each member is independently planned, executed, and reviewed** (verified: each member has its own plan, debug, and review artefacts on its branch).
6. **All members flip to `done` with valid `kind: merged` provenance** (verified: `get_task(α/β/γ)` returns `status="done"`, `done_provenance.kind=="merged"`, `done_provenance.commit==<merge_sha>`).
7. **No debugger re-dispatch loop on intermediate workspace failures** — the loop-guard catches identical-signature repetition (verified: scenarios 6 + 9 of the boundary-test table pass).
8. **Non-train regression**: a parallel non-trained task during the same orchestrator run completes with byte-identical behaviour to today (verified: scenario 11).

## 15. References

- **Incident:** reify task 3997, escalation `esc-3997-12` (2026-05-27); resolved via collapse (Option C).
- **Reify memory note:** `procedural_atomic_multicrate_refactor_collapse` (the collapse playbook this PRD replaces for the preferred path).
- **Orchestrator code anchors:**
  - Task status enum & preserve set: `orchestrator/src/orchestrator/task_status.py:11-18`.
  - Workflow state machine: `orchestrator/src/orchestrator/workflow.py:183-200`.
  - Dispatch gate `_deps_satisfied`: `orchestrator/src/orchestrator/scheduler.py:1402-1433`.
  - Worktree creation & base ref: `orchestrator/src/orchestrator/git_ops.py:464-607`, `_freshen_main:368-451`.
  - GitConfig (branch_prefix, main_branch, remote): `orchestrator/src/orchestrator/config.py:381-406`.
  - Verify loop + retry caps: `orchestrator/src/orchestrator/workflow.py:3023-3115`.
  - Per-module verify command: `orchestrator/src/orchestrator/config.py:427-443`.
  - Done-provenance construction: `orchestrator/src/orchestrator/scheduler.py:1215-1240`.
  - Ancestor check: `orchestrator/src/orchestrator/git_ops.py:1273-1291` and `:896-902`.
  - Merge queue + MergeRequest: `orchestrator/src/orchestrator/merge_queue.py:1-31, 603-626`.
  - Merge invocation & advance_main: `orchestrator/src/orchestrator/git_ops.py:981-1071, 1216-1428`.
- **Relevant memory notes:**
  - `procedural_atomic_multicrate_refactor_collapse` (reify side — the collapse this PRD makes optional).
  - `project_reaper_unblock_worktree_guard` (the existing worktree-spare pattern this PRD mirrors).
  - `feedback_check_thrash_helper_pattern` (the loop-guard pattern this PRD adopts).
  - `feedback_split_multi_package_tasks` (the symptom this PRD partially fixes — multi-package scopes that exceed architect budgets).
  - "Cargo workspace crate scoping should not apply to post-merge verify in merge_queue.py" (the existing workspace post-merge verify this PRD reuses).
