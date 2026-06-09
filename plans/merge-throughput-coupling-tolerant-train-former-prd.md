# PRD — Coupling-tolerant train former (Lever A′)

**Status:** active, **experiment-gated** · dark-factory orchestrator capability, first consumer = reify · authored 2026-06-09.
**Source design:** `plans/merge-throughput-disjoint-former-design.md` §7 (measurement-backed). A′ is the *secondary* throughput lever (~45% P(improves), design §5) — sibling of the primary `plans/merge-throughput-multihost-verify-prd.md` (Lever C). The two **compose** (a train's single union verify can dispatch to C's runner pool) but neither blocks the other (design §8).
**Approach:** **B + H**. A′ touches the scheduler (former), the merge queue (train verify scope), and git_ops (branch stacking) — blast radius ≥3 — and carries a load-bearing correctness invariant (union-scope verify; under-verification = unverified code on `main`). Contract §A + boundary tests §B.
**Gating (load-bearing):** the *build* phase is gated behind a **go/no-go decision** (A-gate) fed by two de-risking experiments (A-exp1, A-exp2). The design's economics only close if **union ≈ single** verify cost AND combined-verify success **s(N) > 1/N** (design §7.3). The experiments measure both; A′ is built only if they clear. (The union-scope *correctness fix*, A-α, lands regardless — it hardens the existing train path.)

---

## 1. Goal — amortize one full verify across N coupled tasks

For a tightly-coupled workspace, batching **coupled** tasks saves *more* than batching disjoint ones (design §7.1, the inversion): members' rdep closures overlap, so the **union closure ≈ a single task's closure** — one union verify costs ≈ one single verify but lands N tasks. reify's coupling (M3: rdep median ~22, `reify-core` a dep of 23/32) — which *kills* disjoint verify-skip — makes amortization *attractive*, if you select by **line-level stackability** and scope to the **union** closure.

**User-observable end state (consumer = the orchestrator merge worker + reify throughput + the 31% CAS-retry churn):**

| | Today (no trains form for reify) | After A′ |
|---|---:|---:|
| Trains formed for reify | 0 (M3/design §7.2) | ≥1 (former selects ≥2 stackable merge-ready tasks) |
| Verifies per N landed tasks | N | ~1 union verify (N tasks land together) |
| CAS-retry churn | 31% (M4) | falls (fewer separate advances) |
| Verify scope on a train | tip-only (latent under-verify bug) | **union over all members** (correctness fix, A-α) |

All deltas are **expectations, not gated thresholds** (G6): each task asserts a *measured improvement direction + recorded delta vs baseline*.

## 2. Background — the existing train path under-verifies; reify never hit it

The orchestrator already has a full atomic-train-merge path (`plans/orchestrator-atomic-train-merge-prd.md`, landed): `metadata.train.{id,order,members}` (workflow.py `_train` ≈:499), the δ₂ trigger `_maybe_enqueue_group_merge` (workflow.py ≈:547), `GroupMergeRequest` (workflow.py ≈:579/648), `_do_train_merge` (merge_queue.py), and sibling-predecessor branch stacking (`_train_predecessor`, git_ops.py ≈:500). **What is missing is a *former*** — something that *creates* trains by selecting merge-ready tasks and assigning `metadata.train`. Today nothing forms trains for reify, so the path is inert (0 trains, M3).

**The latent correctness bug (G3, mandatory fix).** `GroupMergeRequest` is built from the **tip's** `task_files` / `module_configs` only (workflow.py ≈:653–654: `task_files=self._task_files, module_configs=self._module_configs`). When `merge_verify_workspace=false` (reify's setting, config.py ≈:811), scoped verify would then cover **only the tip's crates**, leaving lower members' crates **unverified**. reify never hit this (zero trains) — but the moment a former creates a multi-crate train, this is unverified code on `main`. A′ MUST set the train's scope to the **union over all members** (design §7.2). The union of closures is the sound minimum and, given overlap, ≈ the tip alone.

## 3. Sketch of approach — a scheduler-side former + the union-scope fix

A scheduler-side former (sibling of δ₂, workflow.py ≈:547) that, when ≥2 tasks are merge-ready, selects a small (N≤3) **line-level-stackable** subset, stacks their branches (rebase b2→b1→main; reuse `rebase_onto_main` + `_train_predecessor` git_ops.py ≈:500), assigns synthetic `metadata.train.{id,order,members}`, and lets the tip enqueue a `GroupMergeRequest` whose verify is **union-scoped**.

Mechanisms (each with a named consumer — G1):

| # | Mechanism | Consumer |
|---|---|---|
| 1 | Union-scope on `GroupMergeRequest` (task_files/module_configs = ∪ members) | `_do_train_merge` verify (the correctness gate) |
| 2 | Line-level-stackability test (same crate, different lines = OK) | the former's selection |
| 3 | The former (select N≤3 merge-ready, assign `metadata.train`) | the existing δ₂ → `GroupMergeRequest` path |
| 4 | Branch stacking + conflict-eject fallback | the former → `_do_train_merge` |
| 5 | Failure attribution (re-verify members as singles on train-fail) | the merge worker's failure path |
| 6 | Config: `merge.train_max_members` (default 2), former enable knob | operator |

## 4. Resolved design decisions

- **D1 — dark-factory PRD; all tasks `dark_factory:`.** A′ is entirely DF orchestrator code (scheduler + merge queue + git_ops). reify is the first consumer; it opts in via a config knob. *(Leo, 2026-06-09.)*
- **D2 — Union-scope, not force-workspace (the G3 mandatory fix).** Set the train's `task_files`/`module_configs` to the **union over all members**; `merge_verify_workspace=true` remains the config fallback (force a full workspace verify) for cases where the union closure can't be trusted/computed. Union = sound minimum and, given coupling overlap, ≈ tip-alone cost. *(Leo, 2026-06-09.)*
- **D3 — N ≤ 3, config `merge.train_max_members` default 2.** Start conservative (2); raise to 3 only after A-exp2 confirms `s(3) > 1/3`. Small N bounds the failure-attribution cost (≤1 extra verify, design §7.1). *(Leo, 2026-06-09.)*
- **D4 — Conflict fallback = drop the conflicting member, keep forming.** On a line-level stacking conflict, **eject** the conflicting member (it merges solo via the normal path) and keep the rest; abandon train formation only if <2 members remain. Maximizes formation under the 8–12% single-merge conflict floor (M4). *(Leo, 2026-06-09.)*
- **D5 — Failure attribution = re-verify members as singles, not bisect.** On a train union-verify failure, fall back to the **existing single-task merge path** for each member (re-verify singly), landing the members that pass and blocking/escalating the offender. At N≤3 this is cheaper and less ambiguous than bisection. *(Leo, 2026-06-09.)*
- **D6 — A-α (union-scope fix) lands regardless of the go/no-go.** It is a correctness fix to the *existing* train path, valuable even if the former is never built. Only the former (A-β…A-ε) is gated behind A-gate. *(Leo, 2026-06-09.)*

## 5. Pre-conditions for activating

- **A-exp1 + A-exp2 → A-gate (go/no-go).** The former build tasks (A-β, A-γ, A-δ, A-ε) `depends_on` A-gate. A-gate is a **human decision** task: resolved *go* only if A-exp1 shows union-verify wall-time ≈ single (design's break-even is union ≲ N×single; the *win* is union ≈ single) AND A-exp2 shows `s(N) > 1/N` for the chosen N. A *no-go* cancels the former tasks (A-α still lands).
- **A-α (union-scope fix)** has no upstream gate — substrate present (§6), correctness fix, lands now.
- **A-exp1** is an off-peak / out-of-band measurement (running a cold cargo union verify on the contended box could worsen the very livelock — same rationale as warm-builds D2). Filed with that constraint noted; it *informs* the gate, the build does not start until A-gate resolves.

## 6. Substrate verification (G3) — the train path exists; the former is net-new

Verified at authoring (2026-06-09, `main` HEAD; cite-by-symbol):

| Capability | Evidence (verified) |
|---|---|
| `metadata.train` membership + `_train` reader | present, workflow.py ≈:499–508 (`TrainMembership`) |
| δ₂ trigger `_maybe_enqueue_group_merge` | present, workflow.py ≈:547 |
| `GroupMergeRequest` (+ tip-scoped `task_files`/`module_configs`) | present, workflow.py ≈:579/648; **the tip-only scoping is the bug A-α fixes** (≈:653–654) |
| `_do_train_merge` (train execution path) | present, merge_queue.py |
| `_train_predecessor` sibling-stacking | present, git_ops.py ≈:500 |
| `rebase_onto_main` | present, git_ops.py |
| `merge_verify_workspace` force-workspace knob | present, config.py ≈:811 (default false) |
| **The former (selection + line-level-stackability + train-metadata assignment)** | **absent today — net-new, built by this PRD** (A-β…A-δ) |

No `.ri` grammar surface; no DB schema. The only *assumed* premises are the two economic ones (union≈single; s(N)>1/N), which is exactly why they are **measured** (A-exp1/exp2) before the former is built rather than baked into a RED test (G6 — avoid the "guessed bound frozen into a RED test" failure).

## 7. Cross-PRD / cross-repo relationship (G4)

| Other | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| C = `plans/merge-throughput-multihost-verify-prd.md` | **compose** — a train's single union verify can dispatch to C's runner pool | `pool.dispatch(merge_sha, spec)` is verify-mechanism-agnostic; a train supplies merge_sha + union-spec unchanged | each owns its side; no integration task | independent — neither blocks the other |
| `plans/orchestrator-atomic-train-merge-prd.md` (landed) | A′ **builds on + fixes** its train path | `GroupMergeRequest` tip-scoping bug (A-α) + adds the missing former | A′ owns the former + the fix | A-α corrects the landed path |
| `dark_factory:1596` (`_do_train_merge` → shared post-merge core) | sibling hardening of the same path | shared post-merge verify/advance helpers | 1596 (**done**) | independent — A′ inherits parity |

No reciprocal "the other owns it."

## 8. Decomposition plan — task DAG with observable signals (G2)

Greek labels; task IDs at decompose. All `×`/rate numbers are expectations, never frozen thresholds (G6).

**Phase 0 — de-risking experiments (feed the gate; do NOT build the former yet)**
- **exp1 — Union-verify wall-time measurement.** *(leaf; off-peak / out-of-band — see §5.)* Time a cold cargo verify scoped to a 2–3-task **union closure** vs the ~170 s single baseline (M2). **Signal:** a committed bench doc records the union-closure verify wall-time, the single-task baseline, and the **union/single ratio**. *(Modules: reify measurement; no DF code.)*
- **exp2 — `s(N)` thrash proxy from history.** *(leaf.)* From `main` first-parent history, sample sets of ~3 contemporaneous landed merges and check whether their combined tree would have passed (or proxy: how often two merges landing close together produced a follow-up fix-forward). **Signal:** a committed analysis doc estimating `s(N)` (combined-verify success rate) for N=2 and N=3, and whether `s(N) > 1/N`. *(Modules: `main` history analysis; no DF code.)*
- **gate — A′ go/no-go decision.** *(intermediate → unlocks β, γ, δ, ε; human-resolved.)* Read exp1 + exp2; **go** iff union ≈ single AND `s(N) > 1/N` for the chosen N. A *no-go* cancels β–ε (α still lands). **Signal:** a recorded decision (go/no-go + the chosen N) that flips the former build tasks from blocked to runnable, or cancels them. *(Depends_on exp1, exp2.)*

**Phase 1 — correctness fix (lands regardless of the gate — D6)**
- **α — Union-scope the train verify.** *(intermediate → unlocks β; no upstream gate.)* Set `GroupMergeRequest.task_files`/`module_configs` to the **union over all members** (not the tip's only); `merge_verify_workspace=true` stays the fallback. **Signal:** a multi-member train with members in **different** crates runs a verify whose scope covers **all** members' crate closures (assert the union scope set via unit test + an event field `train_scope=union`); a deliberately-broken **lower** member is **caught** by the train verify (RED→GREEN — the exact bug today). *Modules:* `workflow.py`, `merge_queue.py`.

**Phase 2 — the former (gated behind A-gate)**
- **β — Former core: select N≤3 line-level-stackable merge-ready subset + assign `metadata.train`.** *(intermediate → unlocks γ; `depends_on` gate, α.)* A scheduler-side former (sibling of δ₂) with a line-level-stackability test (same crate, different lines = stackable). **Signal:** given ≥2 merge-ready reify tasks with non-overlapping line ranges, the former assigns `metadata.train.{id,order,members}` to a 2-member subset and emits a `train_formed` event; line-overlapping candidates are **not** co-selected. *Modules:* `workflow.py` (scheduler).
- **γ — Branch stacking + conflict-eject fallback.** *(intermediate → unlocks δ; `depends_on` β.)* Stack selected branches (rebase b2→b1→main via `rebase_onto_main` + `_train_predecessor`); on a stacking conflict **drop** the conflicting member (it merges solo) and keep forming; abandon only if <2 remain (D4). **Signal:** a 3-candidate set whose 3rd member conflicts forms a **2-member** train (the conflicting member is ejected and merges solo); the train tip enqueues a `GroupMergeRequest`. *Modules:* `git_ops.py`, `workflow.py`.
- **δ — Failure attribution: re-verify members as singles on train-fail.** *(intermediate → unlocks ε; `depends_on` γ.)* On a train union-verify failure, fall back to the existing single-task merge path per member (D5). **Signal:** a train whose combined verify **fails** (interaction bug) re-verifies its members individually; members that pass solo **land**; the offender is blocked/escalated; total verify count = N singles + 1 failed train (bounded). *Modules:* `merge_queue.py`, `workflow.py`.
- **ε — Integration gate (B+H leaf, §B).** *(leaf; `depends_on` δ.)* With the former live on reify, observe a real train land. **Signal:** the orchestrator journal shows a reify train landing **N≥2 tasks on a single union verify**, with a recorded drop in CAS-retry rate / verifies-per-landed-task vs the single-merge baseline (the §B boundary-test sketch passing end-to-end). *Modules:* integration harness, dashboard read.

**DAG:** exp1, exp2 → gate. α (independent) → β. gate → {β, γ, δ, ε}. β → γ → δ → ε. (γ also `depends_on` α via β.)

## 9. Out of scope

- **Disjoint-only batching** — dead (design §9; ~1.0–1.5 tasks/verify, M3).
- **Building the former before the experiments clear** — forbidden by A-gate (§5); avoids sinking effort into a possibly-negative-EV lever.
- **Capacity / multi-host verify** — that is C (`plans/merge-throughput-multihost-verify-prd.md`); A′ only amortizes. They compose but are independent.
- **N > 3 trains** — failure-attribution cost grows; out of scope until s(N) is measured for larger N.
- **Bisection on train failure** — rejected (D5); re-verify-singles suffices at N≤3.
- **Force-workspace as the default** — union-scope is the default (D2); force-workspace stays a config fallback only.

## 10. Open questions (tactical — surfaced, not blocking)

1. **Line-level-stackability test mechanism** (per-hunk line-range intersection vs a trial 3-way merge). **Suggested:** trial rebase/merge in a scratch worktree is the ground truth and reuses existing git_ops; line-range intersection is a cheaper pre-filter. Decide during β.
2. **Former trigger cadence** (every scheduler tick with ≥2 merge-ready vs a debounce). **Suggested:** debounce to let a small burst accumulate before forming, bounded so it never delays a lone ready task. Decide during β.
3. **`merge.train_max_members` raise to 3** — gated on A-exp2's `s(3)`. **Suggested:** keep 2 until s(3) clears. Decide post-gate.
4. **Failure-attribution escalation shape** — which member gets the block when the interaction (not a single member) is at fault. **Suggested:** land all members that pass solo; if all pass solo, escalate the *train* (interaction bug) rather than any single member. Decide during δ.

---

## §A — Contract (B+H)

The seam is the former → `GroupMergeRequest` → `_do_train_merge` verify. Invariants:

1. **Union-scope soundness (load-bearing).** A train's verify scope MUST cover the union of all members' `task_files`/`module_configs` (or force-workspace). A tip-only scope is a correctness defect (unverified lower-member crates on `main`). This is the one invariant that, if violated, lands unverified code — A-α + boundary test B1/B2 lock it.
2. **Stackability is line-level, not crate-level.** Two tasks are co-selectable iff their branches stack without conflict (same crate, different lines = OK). Crate-disjointness is **not** required (design §7.1) and is **not** used as a selection criterion.
3. **Eject, don't abort.** A stacking conflict drops the conflicting member (it merges solo); the train forms from the remainder iff ≥2 remain. Formation never blocks a lone ready task.
4. **Bounded failure cost.** A train union-verify failure costs ≤ N (re-verify singles) + 1 (the failed train) verifies. N is capped (`train_max_members`, default 2) precisely to bound this.
5. **Compose with C, don't couple.** The train supplies `merge_sha` + a union `MergeVerifySpec` to whatever verify mechanism runs (local or C's pool). A′ makes no assumption about runner count.

## §B — Boundary-test sketch (B+H) — faces the former side and the verify side

| # | Scenario | Preconditions | Postconditions (asserted) |
|---|---|---|---|
| B1 | Union-scope covers all members | a 2-member train, members in **different** crates | the verify scope set = ∪(members); event `train_scope=union` |
| B2 | Lower-member breakage is caught | a 2-member train whose **lower** member breaks its crate | the train union verify **fails** (today it falsely passes — the A-α RED→GREEN) |
| B3 | Stackable selection | ≥2 merge-ready tasks, same crate, non-overlapping lines | the former co-selects them; `train_formed` emitted |
| B4 | Non-stackable rejection | 2 merge-ready tasks, **overlapping** lines | the former does **not** co-select them; each merges solo |
| B5 | Conflict-eject | 3 candidates, the 3rd conflicts on stacking | a 2-member train forms; the 3rd merges solo |
| B6 | Bounded failure attribution | a 3-member train whose combined verify fails on an interaction | members re-verified as singles; passing members land; verify count ≤ N+1 |
| B7 | Amortization win | a 2-member coupled train (overlapping closures) lands | one union verify lands 2 tasks; recorded union/single wall-time ratio ≈ 1 (from exp1) |
| B8 | No lone-task starvation | one merge-ready task, no second candidate | it merges solo without waiting for a train to form |
