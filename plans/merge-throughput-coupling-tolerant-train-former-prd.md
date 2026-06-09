# PRD — Coupling-tolerant train former (Lever A′)

**Status:** active, **s(N)-gated** · dark-factory orchestrator capability, first consumer = reify · authored 2026-06-09 (corrected 2026-06-09 — see §4 D7).
**Source design:** `plans/merge-throughput-disjoint-former-design.md` §7 (measurement-backed). A′ is the *secondary* throughput lever (~45% P(improves), design §5) — sibling of the primary `plans/merge-throughput-multihost-verify-prd.md` (Lever C). The two **compose** (a train's single union verify can dispatch to C's runner pool) but neither blocks the other (design §8).
**Approach:** **B + H**. A′ touches the scheduler (former), the merge queue (train verify scope), and git_ops (branch stacking) — blast radius ≥3 — and carries a load-bearing correctness invariant (union-scope verify; under-verification = unverified code on `main`). Contract §A + boundary tests §B.
**Gating (load-bearing):** the *build* phase is gated behind a **human s(N) go/no-go**, and is filed **deferred** (not auto-dispatched) until that decision. The economics have **two** factors (design §7.3): (1) **union ≈ single** verify cost, and (2) combined-verify success **s(N) > 1/N**. Factor (1) is **resolved a priori** — reify's merge gate is unconditionally full-`--scope all` (verify.sh:348 contract C2; `-p` affected-crate narrowing is structurally unreachable at scope=all, verify.sh:521-527), so a 1-task and an N-task merge verify run the **byte-identical full-workspace plan** (~90 min cold). union ≡ single by construction; amortization is automatic and *large* (a train lands N tasks for ONE ~90-min verify vs N×90 min). So only factor (2), **s(N)**, gates — measured by A-exp2. (The union-scope *correctness fix*, A-α, lands regardless — it hardens the general train path.) See §4 D7 for the correction that retired the original "measure union-verify wall-time" experiment.

---

## 1. Goal — amortize one full verify across N coupled tasks

For reify, batching tasks into a train saves a near-whole verify per extra member, **unconditionally**: reify's merge gate is unconditionally full-`--scope all` (verify.sh:348 C2; no `-p` narrowing at scope=all), so *every* merge verify — 1 task or N — is the same ~90-min full-workspace build. So a train lands N tasks for **one** ~90-min verify instead of **N** of them. (The design §7.1 "coupled closures overlap so union≈single" reasoning assumed a *scoped* merge verify; reify's gate is full-scope, which makes union≡single hold even more strongly — and makes A′ a pure amortization play whose only risk is thrash, §7.3 / A-exp2.) Select by **line-level stackability** and scope the train to the **union** of members (A-α — sound for the general case, and the exact full set reify already verifies).

**User-observable end state (consumer = the orchestrator merge worker + reify throughput + the 31% CAS-retry churn):**

| | Today (no trains form for reify) | After A′ |
|---|---:|---:|
| Trains formed for reify | 0 (M3/design §7.2) | ≥1 (former selects ≥2 stackable merge-ready tasks) |
| Verifies per N landed tasks | N × ~90-min full verify | ~1 ~90-min full verify (N tasks land together) |
| CAS-retry churn | 31% (M4) | falls (fewer separate advances) |
| Verify scope on a train (orchestrator layer) | tip-only (latent bug; masked on reify by verify.sh C2) | **union over all members** (correctness fix, A-α) |

All deltas are **expectations, not gated thresholds** (G6): each task asserts a *measured improvement direction + recorded delta vs baseline*.

## 2. Background — the existing train path under-verifies; reify never hit it

The orchestrator already has a full atomic-train-merge path (`plans/orchestrator-atomic-train-merge-prd.md`, landed): `metadata.train.{id,order,members}` (workflow.py `_train` ≈:499), the δ₂ trigger `_maybe_enqueue_group_merge` (workflow.py ≈:547), `GroupMergeRequest` (workflow.py ≈:579/648), `_do_train_merge` (merge_queue.py), and sibling-predecessor branch stacking (`_train_predecessor`, git_ops.py ≈:500). **What is missing is a *former*** — something that *creates* trains by selecting merge-ready tasks and assigning `metadata.train`. Today nothing forms trains for reify, so the path is inert (0 trains, M3).

**The latent correctness bug (G3, fix at the orchestrator layer).** `GroupMergeRequest` is built from the **tip's** `task_files` / `module_configs` only (workflow.py ≈:653–654: `task_files=self._task_files, module_configs=self._module_configs`). For a target whose merge verify *honors* the orchestrator scope (`merge_verify_workspace=false` AND no force-all in its own verify script), scoped verify would cover **only the tip's crates**, leaving lower members' crates **unverified** — unverified code on `main` the moment a former creates a multi-crate train. A′ MUST set the train's scope to the **union over all members** (design §7.2).

**Reify caveat (the fix is still required, but masked here):** reify's `scripts/verify.sh` forces `--scope all` for `DF_VERIFY_ROLE=merge` (contract C2, `:348`) and disables `-p` affected-crate narrowing at scope=all (`:521-527`), so reify's merge gate already verifies the **full workspace** regardless of the orchestrator's tip-only `task_files`. So the bug cannot ship on reify today — but A-α is still required (it is correct for the general orchestrator and any target without a force-all merge gate, and removes a latent landmine if C2 is ever relaxed). For reify the fix is defense-in-depth, not a visible behavior change.

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
- **D6 — A-α (union-scope fix) lands regardless of the go/no-go.** It is a correctness fix to the *existing* train path, valuable even if the former is never built (and a general-orchestrator fix even though reify masks it — §2 caveat). Only the former (A-β…A-ε) is gated. *(Leo, 2026-06-09.)*
- **D7 — Correction (2026-06-09): union≡single is resolved a priori; the "measure union-verify wall-time" experiment is retired; the go/no-go is a human checkpoint, not a task node.** Three corrections after grounding the design doc against `runs.db` + `verify.sh`:
  1. **The design-doc M2 "~170 s, ~$0.90" verify cost is misattributed.** It is the median **verify-PHASE agent-invocation** duration (`events.invocation_end · phase=verify`, true median ~100 s, dominated by *scoped task-phase* verifies; the `$0.90` is an LLM call cost) — **not** the cold merge-gate cargo verify. The merge-gate verify is the `merge_attempt`/journal population: p90 ~81 min, **cold median ~90 min** (warm-builds `docs/prds/warmer-builds-merge-verify.md` §1).
  2. **reify's merge gate is unconditionally full-`--scope all`** (verify.sh:348 C2; `-p` narrowing structurally off at scope=all, :521-527). So a 1-task and an N-task merge verify are the **byte-identical** full plan → **union ≡ single** by construction. The original A-exp1 ("time a union closure vs a single closure") measures scoped closures reify's merge gate never uses; it cannot change an identity, so it is **retired/cancelled**. Only **s(N)** (A-exp2) gates.
  3. **The go/no-go is a human checkpoint, not a dispatchable task.** A no-code "decision" task dispatches to a coding agent and gets blocked as not-TDD-plannable (observed: the orchestrator dispatched + blocked the original A-gate task within ~17 min). So the former build (A-β…A-ε) is filed **`deferred`** and the decision is a documented human step: after A-exp2 lands, a human reviews s(N) and on **go** flips A-β…A-ε to `pending`, on **no-go** cancels them. *(Leo, 2026-06-09.)*

## 5. Pre-conditions for activating

- **A-exp2 → human s(N) go/no-go → former build.** Only `s(N)` gates (D7: union≡single is a priori). A-exp2 (the s(N) thrash proxy) runs now. When it lands, a **human** reviews s(N) and applies: **go** iff `s(N) > 1/N` for the chosen N (start N=2) — i.e. combined trains succeed often enough that the re-verify-on-fail cost (≤ N+1 full verifies) doesn't eat the (N−1)-full-verify saving. On **go**: flip A-β…A-ε from `deferred` to `pending`. On **no-go**: cancel A-β…A-ε. The former build is filed **`deferred`** so it never auto-dispatches ahead of this human call (D7.3).
- **A-α (union-scope fix)** has no upstream gate — substrate present (§6), correctness fix, lands now (`pending`).
- **A-exp1 (union-verify wall-time) is retired** (D7.2) — union≡single is an identity under reify's full-scope merge gate; no measurement applies.

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

No `.ri` grammar surface; no DB schema. One economic premise (union≈single) is **resolved a priori** by the full-scope merge gate (D7); the other (`s(N) > 1/N`) is **measured** (A-exp2) before the former is built rather than baked into a RED test (G6 — avoid the "guessed bound frozen into a RED test" failure).

## 7. Cross-PRD / cross-repo relationship (G4)

| Other | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| C = `plans/merge-throughput-multihost-verify-prd.md` | **compose** — a train's single union verify can dispatch to C's runner pool | `pool.dispatch(merge_sha, spec)` is verify-mechanism-agnostic; a train supplies merge_sha + union-spec unchanged | each owns its side; no integration task | independent — neither blocks the other |
| `plans/orchestrator-atomic-train-merge-prd.md` (landed) | A′ **builds on + fixes** its train path | `GroupMergeRequest` tip-scoping bug (A-α) + adds the missing former | A′ owns the former + the fix | A-α corrects the landed path |
| `dark_factory:1596` (`_do_train_merge` → shared post-merge core) | sibling hardening of the same path | shared post-merge verify/advance helpers | 1596 (**done**) | independent — A′ inherits parity |

No reciprocal "the other owns it."

## 8. Decomposition plan — task DAG with observable signals (G2)

Greek labels; task IDs at decompose. All `×`/rate numbers are expectations, never frozen thresholds (G6).

**Phase 0 — de-risking (feeds the human go/no-go; do NOT build the former yet)**
- **exp1 — RETIRED (D7.2).** ~~Union-verify wall-time measurement.~~ union≡single is an identity under reify's full-scope merge gate (verify.sh:348 C2; no `-p` narrowing at scope=all) — a 1-task and an N-task merge verify are the byte-identical full plan, so there is nothing to measure. *(Cancelled: reify task 4454.)*
- **exp2 — `s(N)` thrash proxy from history.** *(leaf; the ONLY economic gate.)* From `main` first-parent history, sample sets of ~3 contemporaneous landed merges and check whether their combined tree would have passed (or proxy: how often two merges landing close together produced a follow-up fix-forward). **Signal:** a committed analysis doc estimating `s(N)` (combined-verify success rate) for N=2 and N=3, and whether `s(N) > 1/N`. *(Modules: `main` history analysis; no DF code. reify task 4455.)*
- **go/no-go — a human checkpoint, NOT a task node (D7.3).** After exp2 lands, a human reads s(N) and applies: **go** iff `s(N) > 1/N` for the chosen N → flip β…ε `deferred`→`pending`; **no-go** → cancel β…ε. (α lands regardless.) Not modelled as a dispatchable task — a no-code decision task churns the orchestrator (observed: the original gate-task dispatched and was blocked as not-TDD-plannable within ~17 min).

**Phase 1 — correctness fix (lands regardless of the gate — D6)**
- **α — Union-scope the train verify.** *(intermediate → unlocks β; no upstream gate.)* Set `GroupMergeRequest.task_files`/`module_configs` to the **union over all members** (not the tip's only); `merge_verify_workspace=true` stays the fallback. **Signal:** a multi-member train with members in **different** crates runs a verify whose scope covers **all** members' crate closures (assert the union scope set via unit test + an event field `train_scope=union`); a deliberately-broken **lower** member is **caught** by the train verify (RED→GREEN — the exact bug today). *Modules:* `workflow.py`, `merge_queue.py`.

**Phase 2 — the former (filed `deferred`; a human flips to `pending` on go — D7.3)**
- **β — Former core: select N≤3 line-level-stackable merge-ready subset + assign `metadata.train`.** *(intermediate → unlocks γ; `depends_on` α; filed `deferred`, flipped on go.)* A scheduler-side former (sibling of δ₂) with a line-level-stackability test (same crate, different lines = stackable). **Signal:** given ≥2 merge-ready reify tasks with non-overlapping line ranges, the former assigns `metadata.train.{id,order,members}` to a 2-member subset and emits a `train_formed` event; line-overlapping candidates are **not** co-selected. *Modules:* `workflow.py` (scheduler).
- **γ — Branch stacking + conflict-eject fallback.** *(intermediate → unlocks δ; `depends_on` β.)* Stack selected branches (rebase b2→b1→main via `rebase_onto_main` + `_train_predecessor`); on a stacking conflict **drop** the conflicting member (it merges solo) and keep forming; abandon only if <2 remain (D4). **Signal:** a 3-candidate set whose 3rd member conflicts forms a **2-member** train (the conflicting member is ejected and merges solo); the train tip enqueues a `GroupMergeRequest`. *Modules:* `git_ops.py`, `workflow.py`.
- **δ — Failure attribution: re-verify members as singles on train-fail.** *(intermediate → unlocks ε; `depends_on` γ.)* On a train union-verify failure, fall back to the existing single-task merge path per member (D5). **Signal:** a train whose combined verify **fails** (interaction bug) re-verifies its members individually; members that pass solo **land**; the offender is blocked/escalated; total verify count = N singles + 1 failed train (bounded). *Modules:* `merge_queue.py`, `workflow.py`.
- **ε — Integration gate (B+H leaf, §B).** *(leaf; `depends_on` δ.)* With the former live on reify, observe a real train land. **Signal:** the orchestrator journal shows a reify train landing **N≥2 tasks on a single union verify**, with a recorded drop in CAS-retry rate / verifies-per-landed-task vs the single-merge baseline (the §B boundary-test sketch passing end-to-end). *Modules:* integration harness, dashboard read.

**DAG:** exp2 (reify, `pending`) feeds the human go/no-go. α (df, `pending`, independent) → β. β → γ → δ → ε (all df, filed **`deferred`**; flipped `deferred`→`pending` together on a go, or cancelled on a no-go). No "gate" task node (D7.3). exp1 retired/cancelled.

## 9. Out of scope

- **Disjoint-only batching** — dead (design §9; ~1.0–1.5 tasks/verify, M3).
- **Building the former before s(N) clears** — the former tasks are filed `deferred` and held behind the human go/no-go (§5/D7.3); avoids sinking effort into a possibly-negative-EV lever.
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
| B7 | Amortization win | a 2-member train lands | one full-scope verify lands 2 tasks; verifies-per-landed-task drops from ~1 to ~0.5 (union≡single is a priori under the full-scope gate — D7) |
| B8 | No lone-task starvation | one merge-ready task, no second candidate | it merges solo without waiting for a train to form |
