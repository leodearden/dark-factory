# Capability manifest — `plans/task-amendment-delivery-prd.md`

Authored at decompose, 2026-08-12, against `main` at `8eeb862fa5`.
Machine-readable twin: `plans/task-amendment-delivery-prd.capability-manifest.yaml`
(path strictly derived from the PRD path — never renamed to match this file's stem).

This manifest pays the G3 + G6 substrate check **once, here**, so a dispatch-time
architect diffs intent against substrate instead of re-deriving it per task.

**Anchoring discipline** (per the `resume-charter-loss-remediation` exemplar):
mechanical checks anchor on **production symbols** wherever the PRD's contracts fix
one, **never** on `file:line`, and every mechanical check was executed as
`git grep -E -c '<pattern>' -- <paths>` against `main` and is **RED today** — an
`expect: present` check misses (0 hits), an `expect: absent` check hits. Measured
counts are recorded per check in the sidecar.

Three checks are test-anchored **by necessity** (β's reconciler call site, η's two
boundary-test roots) and are labelled as such: β's production call form would be
byte-identical to the existing single call site so a `present` grep on the symbol is
vacuously green, and η's whole deliverable *is* tests.

---

## G-gate re-check summary (decompose, 2026-08-12)

| Gate | Verdict | Note |
|---|---|---|
| **G1** consumer named | PASS | every arm's consumer named below; no producer-orphan |
| **G3** substrate verified | PASS | **no novel substrate** — all 24 assumed capabilities verified present on `main` today (list below); the only prerequisite is **3651**, a *correctness* prereq, not a substrate one |
| **G4** seam ownership | PASS | 3157 is the sole contested seam; resolved by D8 + the §7b amendment performed at decompose. No reciprocal ambiguity. |
| **G5** B vs B+H | B+H | integration gate **η** exists and names §10 as its signal |
| **G2** leaf signals | PASS | every task carries a `user_observable_signal`; α/β/γ/δ/ε are intermediates with named downstream consumers *and* independent signals |
| **G6** premise validity | PASS **after one relaxation** — see ζ below | |
| **G7** design invariants | PASS **after two redesigns**, **no waivers** — see below | |

### G3 — substrate verified on `main` (`8eeb862fa5`)

`BriefingAssembler._format_task` (`briefing.py:1467`) · `_can_skip_revalidation`
(`workflow.py:4764`) · `_apply_revalidation_skip` (`:4832`) · `bump_revalidation_stamp`
(`artifacts.py:944`) · `stamp_plan_provenance` (`artifacts.py:978`, 5 production
callers) · `_reconcile_done_step_commits` (`workflow.py:8440`, **exactly one** call site
at `:7976`) · `GitOps.find_equivalent_commit` (`git_ops.py:8855`) ·
`git_ops._run` (module-level async runner at `git_ops.py:1726`, wrapping
`asyncio.create_subprocess_exec`) · `_check_plan_files_touched_in_branch`
(`merge_gates.py:1333`, called from `workflow.py:10090` with the `_try_narrow_plan`
remedy-and-recheck at `:10108`) · `register_and_enqueue_merge_request`
(`workflow.py:1740`) · `_handle_no_plan_failure` (4 call sites) +
`RetryLedger.consecutive_no_plan_failures` / `.consecutive_merge_thrash`
(`shared/task_metadata.py`) · `UpdateTaskResult` (`task_backend_types.py:18`) ·
`should_reembed` predicate (`task_interceptor.py:4467`) · `_execute_combine`
(`task_interceptor.py:2082`) and its terminal-only predicate (`:2144`) ·
`combine_eligible=(status == 'pending')` (`task_curator.py:2793`) ·
`_append_combine_audit` (`task_interceptor.py:2215`; path helper `:5914`) ·
`_normalize_content_description` (`flag_dedup.py:1375`) · `RELOADABLE_FIELDS` +
`revalidation_skip_enabled` (`config.py:3237`) · `test_plan_revalidation_skip.py`
(15 tests) · `dark-factory-shared` a workspace dep of **both** `orchestrator` and
`fused-memory` · `EventType.plan_revalidated` (`event_store.py:334`) ·
`EventType.phase_skipped` (`:363`) with `reason='revalidation_skipped_no_overlap'`
(`workflow.py:4894`) and `_stamp_optimistic_path('revalidation_skip')` (`:4886`) ·
`plan_tools._create_plan` writing `'steps': []` (`plan_tools.py:997`).

*Drift note (non-blocking):* task 3157's 2026-08-05 addendum cites
`_reconcile_done_step_commits` at `workflow.py:7915` / call site `:7451`. Those are
stale; the PRD's `8440` / `7976` are current. β should trust the PRD's anchors.

### G6 — one relaxation, recorded

**ζ's numeric premise relaxed.** PRD §7a's ζ row asserts *"the **27** measured-exposed
tasks carry the forcing sentinel"*. 27 is a measurement taken during the authoring
investigation and will have moved by the time ζ runs. Per G6 branch 1 resolution (b),
ζ's filed signal drops the literal and asserts the **re-derived** set instead — which
is also what §11 open question 3 already directs (*"the sweep should re-derive rather
than hard-code"*) and what INV-3 requires. The `27` figure is retained in ζ's task text
as a **sanity band**, not an assertion.

No other signal asserts a number, an exactness claim, or a pre-existing rejection
mechanism. β/γ/δ/ε each assert a rejection **they themselves build**, which is G6
branch 4's producer side, not its consumer side.

### G7 — two hits, both resolved by redesign; no waivers

The PRD's §12 advisory walk cleared all eight invariants. Walking every task in the
batch surfaced two the advisory walk missed, both resolved into the filed task text:

1. **γ / `storm-escape-required` (INV-4).** C6 says *"Fail-open on git error (matching
   the sibling gates' documented convention)"*. That convention, read at
   `_check_plan_files_touched_in_branch`, is *"Loud-log so regressions surface in ops"*
   — a WARNING, **no counter**. A γ fail-open that fires 100× in an hour is therefore
   silent by inheritance. **Resolved:** γ's fail-open branch carries a streak counter
   and escalates on it; γ's task text names this as a required conjunct.
2. **ε / `no-lockstep-duplication` (INV-5).** C8 as written hand-copies
   `status == 'pending'` from `task_curator.py:2793` into `task_interceptor.py:2144` —
   two sites that must agree byte-for-byte, which is precisely the INV-5 shape and
   precisely how the original divergence arose. **Resolved:** ε extracts one shared
   eligibility predicate called by **both** sites. The `claimant_run_id is None`
   conjunct (D11) stays **execution-only** and is deliberately *not* pushed into
   selection — the selection snapshot cannot know it.

Also recorded, not a hit: ε's abort→create fall-through is a fail-soft path
(INV-4's own evidence names *"curator degrade-to-create"*), and at a measured 20.2%
rate an operator must be able to compute it — so ε's refusal is required to carry a
stable machine-readable reason in the combine audit, making the rate countable from
an artifact that already exists (INV-2 + INV-4 satisfied by reuse, not a new channel).

β and γ's refusals are **holds** (INV-6/INV-7): both route through pre-existing
bounded choke points per D13 — `_handle_no_plan_failure` /
`consecutive_no_plan_failures` for β, the merge-phase `consecutive_merge_thrash`
ledger for γ — rather than inventing a second failure path.

---

## α — Task-text fingerprint gates plan-revalidation skip

**Consumer:** β, δ, ζ, η (intra-batch) + the operator-visible dispatch path (an amended
task's next dispatch takes the architect revalidation lane and the briefing carries the
amended text).

| Capability | Evidence binding | Verdict |
|---|---|---|
| `task-text-fingerprint-helper-in-shared` | capability→producer (built by α). `dark-factory-shared` is a verified workspace dep of both consumers (`orchestrator/pyproject.toml:21,26`, `fused-memory/pyproject.toml:23,27`), so D2's single-home requirement is reachable. `shared/src/shared/` holds no fingerprint helper today. | PASS |
| `fingerprint-is-explicitly-NOT-the-recon-normalizer` | INV-5 / D2. The two predicates must **differ** (recon casefolds, ours must not), so the deliverable is a non-unification note naming `_normalize_content_description` in α's own docstring — the thing that stops a future tidy-up collapsing them. | PASS |
| `skip-predicate-reads-the-fingerprint` | capability→producer (wired). `_can_skip_revalidation` (`workflow.py:4764`) already receives the plan and can read `self.task`; it has four conjuncts and reads no task text. That is the seam C3 adds the fifth conjunct to. | PASS |
| `fingerprint-mirrored-onto-task-metadata` | capability→producer (wired). D10 requires δ to read `task.metadata`, never the filesystem; `Scheduler.update_task` is metadata-only (verified across ~30 orchestrator call sites), so the mirror is legal for the orchestrator and is the DB read δ needs. | PASS |
| `revalidation-text-gate-is-green-tier` | capability→producer (wired). `RELOADABLE_FIELDS` exists and `revalidation_skip_enabled` (`config.py:3237`) is present but **red tier** today; the new gate bool must land green-tier. | PASS |
| `missing-fingerprint-is-a-structured-fault-not-a-silent-decline` | INV-2/INV-4 (D3/C3). **manual** — §11 open question 1 deliberately leaves the fault *channel* undecided (event vs counter vs `info` escalation); a grep on a guessed channel name would be a false RED at the dispatch gate rather than a check. | PASS |

## β — Done-step integrity + sanctioned descope on the revalidation path

**Consumer:** γ (reads the descope record as its `DESCOPED` set), η.
**Hard prereq:** **3651** — without it β inherits the filename-subset collapse that
stamps every rebase-orphaned step onto the WIP tip.

| Capability | Evidence binding | Verdict |
|---|---|---|
| `descope-record-promoted-to-plan-schema` | capability→producer (built by β), promoting 2971's hand-rolled `_rescoped_at`/`_rescoped_by`/`_rescoped_note` (verified on disk at `.worktrees/.task-meta/2971/plan.json`) to schema. Absent from `orchestrator/src/` today. | PASS |
| `dropped-done-steps-keyed-on-step-id-plus-commit` | capability→producer (built by β). C5's `DROPPED := D_before \ D_after` is keyed on `(step_id, commit)` + disposition, **not** count — refuted for count by 2971 (1→1, real destruction) and 3143 (19→0, legitimate wipe). | PASS |
| `revalidation-path-calls-the-existing-reconciler` | capability→producer (wired) — `_reconcile_done_step_commits` (`workflow.py:8440`) exists and is sound (2386, 2762); the gap is **phase placement**, it has exactly one call site (`:7976`, inside the implement loop). D6 forbids a second reachability mechanism (INV-5). *Test-home-anchored by necessity:* β's new production call is byte-identical to the existing one, so a `present` grep on the symbol is vacuously green. | PASS |
| `sha-unresolvable-is-expected-not-a-failure` | numeric-premise/floor. Basis: 3157's addendum — **991/1,973 (50.2%)** recorded done-step SHAs no longer exist as git objects; a bare `merge-base --is-ancestor` fires on **185/200** live branches (**39/49** non-terminal). **manual** — this is a *negative* requirement on β's classifier, provable only by the 3143-replay boundary test (§10 row 9), which η owns. | PASS |
| `duplicate-done-step-shas-refused` | capability→producer (built by β). D6's cheap companion assertion; closes the hole where a set-based check is gamed by duplicating a sha. Covered by η's §10 row 10. | PASS |
| `beta-refusal-routes-through-the-existing-bounded-hold` | INV-6/INV-7 via D13 — `_handle_no_plan_failure` (4 call sites) + `RetryLedger.consecutive_no_plan_failures`, escalating to a human at ≥2. **manual** — reuse of an existing path emits no new symbol; a `present` grep on `_handle_no_plan_failure` is vacuously green at 4 hits. | PASS |

## γ — Converse merge gate at submission: branch commits ⊆ owned ∪ descoped

**Consumer:** η + the operator-visible merge-submission refusal naming the offending sha.
**Prereq:** β (γ's `DESCOPED` set is β's `_descoped_steps`).

| Capability | Evidence binding | Verdict |
|---|---|---|
| `gate-reads-the-descope-record` | capability→producer, DAG-direction: producer is **β, upstream**. Scoped to `merge_gates.py` so β's own `workflow.py` landing cannot vacuously green it. | PASS |
| `commit-ownership-gate-at-the-sibling-gate-site` | capability→producer (wired). D9: `_check_plan_files_touched_in_branch` (`merge_gates.py:1333`) already runs inside `_submit_to_merge_queue` (`workflow.py:10090`) **before** `register_and_enqueue_merge_request`, with an in-place remedy-and-recheck precedent at `:10108`. γ goes at the same site. | PASS |
| `fail-open-branch-carries-a-storm-escape` | **G7 redesign (INV-4)** — the sibling convention is loud-log-only, no counter. **manual**: the counter's channel is not fixed by the PRD, and the sibling gates' own remedy is out of γ's scope. Pinned by γ's task text as a required conjunct. | PASS |
| `commit-walk-is-loop-safe-and-bounded` | INV-8 / C6 — async runner (`git_ops._run`, `git_ops.py:1726`), loop-invariant probes hoisted, fan-out bounded by the branch's commit count. **manual**: `subprocess.run` already has **0** occurrences in `merge_gates.py`, so an `expect: absent` check would be vacuously green rather than a measurement. | PASS |

## δ — Write-boundary amendment advisory + event

**Consumer:** η + the MCP caller that sees `amendment_advisory` in its `update_task`
result, and the operator who sees `task_text_amended`.
**Prereq:** α (δ compares against α's mirrored `metadata.plan_text_fp`).

| Capability | Evidence binding | Verdict |
|---|---|---|
| `update-task-result-carries-the-advisory` | capability→producer (wired). `UpdateTaskResult` (`task_backend_types.py:18-22`) exists with **no** advisory key; C7 adds it as `NotRequired`, so the envelope declares itself where callers see it (INV-1). | PASS |
| `advisory-emitted-at-the-should-reembed-chokepoint` | capability→producer (wired). `should_reembed` (`task_interceptor.py:4467`) already tests exactly `{prompt, title, description, details}` — the right chokepoint, no new predicate needed. | PASS |
| `structured-task-text-amended-event-carries-writer-source` | capability→producer (built by δ). Load-bearing because the **TaskCurator's `combine` bypasses `TaskInterceptor` entirely** (`tm.update_task` direct), so no advisory can reach it — the event is the only channel that keeps curator writes operator-visible. | PASS |
| `advisory-reads-task-metadata-never-the-filesystem` | **INV-8 / D10** — `plan.json` lives under `<worktree_base>/.task-meta/…`, invisible to fused-memory; a filesystem or git probe on the fused-memory loop is the exact task-3778 incident. Anchored on the mirrored key appearing in `fused-memory/src/`, which is the *positive* proof δ took the DB-read route. Pinned end-to-end by §10 row 14. | PASS |

## ε — Curator combine: execution-time predicate matches selection

**Consumer:** η + the operator-visible outcome (candidate filed as its own task; the
refusal recorded in `data/combine_audit.jsonl`, which exists and is 1.0 MB / 640 records).
**Prereq:** none — independently dispatchable today. *This PRD's cheapest win.*

| Capability | Evidence binding | Verdict |
|---|---|---|
| `terminal-only-execution-predicate-retired` | rejection-mechanism. `if target_status in {'done', 'cancelled'}` (`task_interceptor.py:2144`) is the entire status predicate at execution; nothing blocks `in-progress`. Measured: **129/640 (20.2%)** combines executed against an `in-progress` target, **128** of which rewrote the description; **23** carry an LLM justification asserting the target *"is pending"* in the same record whose `old.status` reads `in-progress`. | PASS |
| `live-claimant-checked-at-combine-execution` | capability→producer (built by ε), D11. `_execute_combine` **already re-reads the live target** (`:2126`) — this is a re-read with the wrong predicate, not a missing one — so the claimant field is one `.get` away on a dict already in hand (no extra I/O, INV-8-neutral). Anchored on the `target.get(...)` form because the bare symbol has 15 pre-existing hits in the file. | PASS |
| `one-eligibility-predicate-shared-by-selection-and-execution` | **G7 redesign (INV-5)** — `combine_eligible=(status == 'pending')` (`task_curator.py:2793`) and the new execution predicate are two sites that must agree; extraction over duplication. **manual**: both modules sit in `fused_memory.middleware`, but the helper's name and home are an implementation choice, so a guessed grep would false-RED a correct task. Pinned by ε's task text. | PASS |
| `refusal-is-countable-from-the-combine-audit` | INV-2 + INV-4. The abort path is already safe (`# combine failed → fall through to create`, `:3487`) and `_append_combine_audit` (`:2215`) already exists — the deliverable is a **stable machine-readable refusal reason** on that record so the 20.2% rate is computable without log-scraping. **manual**: the reason vocabulary is §11 open question 5. | PASS |

## ζ — One-off plan-fingerprint migration sweep

**Consumer:** α's D3 predicate (which is a *fault* on a missing fingerprint precisely
because ζ guarantees none is missing) + the operator reading the sweep's per-class counts.
**Prereq:** α (ζ uses α's shared helper — INV-5; it must not re-implement the hash).

| Capability | Evidence binding | Verdict |
|---|---|---|
| `migration-script-stamps-live-plans` | capability→producer (built by ζ), consuming α's `shared/` helper. `scripts/` holds nothing fingerprint-related today. | PASS |
| `exposed-set-is-re-derived-not-hard-coded` | **G6 relaxation (recorded above) + INV-3.** The `27` is a decompose-time measurement, not a durable constant; ζ re-derives from live state. Retained as a sanity band in ζ's task text. **manual** — a "does not hard-code a number" property is not greppable. | PASS |
| `every-live-plan-carries-the-fingerprint-after-the-sweep` | **manual** — the assertion is over `.worktrees/.task-meta/*/plan.json`, runtime state **outside the git tree**; no `git grep` can observe it. Same shape as the exemplar's out-of-tree fleet-state capability. | PASS |

## η — End-to-end amendment-delivery boundary tests (B+H integration gate)

**Consumer:** CI + the §10 boundary-test table itself. This is the G2 leaf that ropes
α–ε's foundation work into one user-observable signal.
**Prereqs:** α, β, γ, δ, ε — every capability §10 exercises is delivered **upstream**
of η, so DAG-direction is correct for all 16 rows (no signal depends on a task that
depends on η).

| Capability | Evidence binding | Verdict |
|---|---|---|
| `orchestrator-side-boundary-rows-exercised` | capability→producer: rows 1–12 (fingerprint trigger, mid-EXECUTE non-refresh, whitespace-vs-case, descope accept/refuse, 2971 + 3143 replays, duplicate-sha, γ's two ride-along rows) are all delivered by α/β/γ, upstream. *Test-anchored by construction* — η's deliverable **is** tests. | PASS |
| `fused-memory-side-boundary-rows-exercised` | capability→producer: rows 13–16 (advisory returned, `task_text_amended` emitted, δ does no filesystem/git I/O, the two combine-refusal rows) delivered by δ/ε, upstream. | PASS |
| `mid-execute-non-refresh-and-loop-safety-pinned` | the two rows the rest of the design silently depends on: row 3 (a mid-EXECUTE amendment must not self-clear via the per-iteration provenance re-stamp — the same shape that overwrote 2971's manual `_rescoped_at`) and row 14 (INV-8). **manual** — both are *negative* properties spread across α's and δ's production shapes; the positive anchors above already prove the test modules exist. | PASS |

---

## Not in this manifest

Two tasks are filed in the same planning batch but are **deliberately outside this
PRD's arms** (§7b / §8) and therefore carry no manifest entry and no `prd_path`:

- **2852's dropped "option A"** — make the state-walking merge-lifecycle poller read
  `current()` each tick instead of advancing its own cached stage. 2852 is `done`
  (merged `a0290a67`, option B only); no task owns option A today.
- **Recon Stage 2's `append=true` + `description=` clobber** (§8) — `description`
  always overwrites regardless of `append`; a real data-loss bug, adjacent but
  independent.

`commit_planning`'s label→task-id stamper skips both correctly (no
`metadata.prd_task_label`), so neither receives `metadata.delivered_checks`.
