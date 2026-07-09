# Merge-gate integration-skew attribution + pipeline-landing tripwire

**Status:** active — authored 2026-07-09 (reify verify-flakiness survey follow-up; direction ratified by Leo 2026-07-08). Approach **B+H** (light contract + boundary-test sketch — this touches the merge-landing seam).

## Goal

When a merge-gate verify fails for a reason that is neither the branch's own bug nor a flake — the branch is *semantically stale* against a change that landed on main after it was cut — the orchestrator should **say so, name the landed commit(s) to port, and route the task accordingly**, instead of emitting a generic verify failure that burns a debugfix loop, an L2 escalation, and a false "infra flake" statistic. Secondarily: when a landing touches the verify-pipeline load-bearing set, in-flight tasks likely to be invalidated should be **told proactively** rather than discovering it at their merge gate.

User-observable outcome: an operator reading a skew-classed escalation sees `disposition=integration_skew` plus the implicated landed commit sha(s) and the failing-test overlap that implicated them; the steward/debugger receives "port landed change X" context instead of "verify failed"; per-category failure stats stop counting skew as flakiness.

## Background / evidence

The 2026-07-08 reify verify-flakiness survey found that **same-day multi-task failure bursts caused by main moving under in-flight branches are the single biggest generator of misattributed verify failures** — at least six bursts in June/July (07-01 classification-manifest incident, 5 tasks; 07-07 run_all tiering flip hitting 5124/5133; 07-02/03 `REIFY_VERIFY_TEST_TIMEOUT` knob asserts, 7 tasks; 06-25 sandbox copy-list; 06-30 `ExtrudeInfinite` enum variant; 06-24 GUI pane-API type change). Each read as "flake" or "task's own bug"; retries-after-rebase passing completed the illusion. Reify memory: `project_verify_flakiness_survey_2026_07`.

**What already exists (verified in code 2026-07-09 — do NOT rebuild):**

| Mechanism | Where | What it closes |
|---|---|---|
| Freshness re-merge at verify-pickup (task 1646, Mechanism 2) | `merge_queue.py:4440-4489` (`_remerge` :4580) | Stale merge tree missing an already-landed fix (StatusBar class) |
| Disjoint-delta re-verify gate when main moves during verify (#1595) | `merge_queue.py:4832-4990` (`gate_reverify` phase :9384) | Mid-verify drift |
| `skip_verify` preserved across main-advanced re-merge (residual) | closed by task 1672 (done) | Unverified re-merged tree landing |
| Main-red (preexisting) probe at the merge gate | `verify_failure_is_preexisting_on_main` (`verify.py:3673`, cached), imported by `merge_queue.py:165`; `preexisting_main_break` category + fingerprint dedup (`merge_queue.py:675-689`) + "fix main:" escalation composer (:514) | Flavor A: main itself red — detected, dedup-escalated once |
| Broken-main contagion guard, workflow side (1645/1802) | `verify.py` preexisting-probe CATEGORY_POLICY (:723-732, :2092) | N tasks each self-patching an inherited main break |
| Periodic main-tip sweep + no-landings breaker | `harness.py:952,1305` / `:973` (`run_main_tip_sweep` `verify.py:3844` with retry-on-flake + suppressed-flake registry) | Background main-health detection; landing-stall floor |
| Inter-iteration rebase of in-flight branches | workflow ("save WIP before inter-iteration rebase" commits) | Branch worktrees pick up main between iterations |

**The remaining gap** is therefore purely *classification and routing*: today's disposition is a **bi-state** — `preexisting_main_break` or generic-failure. The generic bucket lumps together (i) the branch's own bug, (ii) transient/flake, and (iii) **integration skew** — branch content semantically incompatible with a landed change, where the failing test passes on main and the fix is "port the landed change into the branch", an *agent edit* no re-merge or retry can produce. Nothing computes (iii), nothing names the implicated landing, and the events/escalations carry no disposition field, so triage and flake statistics conflate all three.

## Sketch of approach

**M1 — skew-verdict classifier** (new, the core mechanism). On merge-gate verify failure, after the existing preexisting-on-main probe returns "not preexisting", compute a disposition:

- Extract the failing test identifiers / failing files from the verify result (cause_hint, per-test results where available).
- Map failing tests to the source/test files that define them (path heuristics + `git log --name-only`; exact mapping is tactical).
- `git log --name-only <merge_base(branch,main)>..main -- <those files + their asserted-config files>` → the **implicated landings**.
- Verdict: implicated landings exist ∧ branch's own pre-merge verify was green → `integration_skew` (+ implicated shas, overlap evidence). No implicated landings → `branch_bug`. Classifier error / ambiguity → `indeterminate` (fail-open to today's behavior).

**M2 — routing + surfacing.** The verdict travels verbatim into: the `MergeOutcome` / `merge_status` `failure_diagnostic`, the task's block reason + dry-run-proposal context (so the debugger is told "port landed commit X touching <files>, don't hunt your own diff"), the L1 escalation body, and the runs.db merge event payload (a `disposition` field). `integration_skew` failures get a distinct escalation category so watcher policy and failure statistics can separate them from flakes.

**M3 — pipeline-landing tripwire** (proactive half; cross-repo seam). Post-advance hook: for each landing, compute its changed files and consult the project-configured load-bearing oracle — for reify, `bash scripts/verify-pipeline-guard.sh requires-full-gate <files...>` (exit 0 = load-bearing; oracle exists and was exercised 2026-07-08). If load-bearing: emit one **info** escalation naming the landing and the in-flight tasks whose branch diffs overlap the load-bearing set (those are the ones whose own edits must be ported, not merely rebased), and attach a steward-visible note to those tasks' metadata. Config-gated per project; absent oracle → no-op.

**M4 — reify-side ordering check** (sibling task in the reify tree, not decomposed here; see seam table). The `/prd` overlay gains an authoring check: a task that adds a gate-test binary or infra test must carry its drift-guard/manifest registration in the same diff or as a hard upstream dependency (the 4914 A3-before-A6 class).

### Resolved design decisions

- **D1 — no new auto-retry.** The originally-discussed "rebase-then-reverify on failure" is **already implemented** (1646 + #1595 + 1672); a merge-gate failure on a freshened tree is deterministic wrt (branch, main) and a retry is pure waste. This PRD adds *attribution*, not retries. (Ratified direction 2026-07-08 was S1+S2; S1 collapsed to M1/M2 on code evidence.)
- **D2 — classifier is git-only and read-only.** No extra verify run, no worktree mutation; the main-side truth comes from the *existing cached* preexisting probe. Bounded cost: a few `git log`/`diff` calls.
- **D3 — fail-open.** `indeterminate` produces exactly today's behavior. A classifier bug must never block a landing or suppress an escalation.
- **D4 — tripwire is advisory.** It never blocks or reorders the queue (pickup re-merge already handles tree freshness); it only informs. One escalation per landing, not per task.
- **D5 — disposition vocabulary** is a closed enum: `main_red` (alias of the existing preexisting path — reported, not re-implemented), `integration_skew`, `branch_bug`, `indeterminate`. It composes with the verify-classifier rework (tasks 2123/2131) — disposition is *orthogonal to* FailureCategory (category = what failed; disposition = whose fault).

## Pre-conditions (G3 — all verified 2026-07-09)

- `verify_failure_is_preexisting_on_main` wired at the merge gate (`merge_queue.py:165`, `:675-689`) — **exists**.
- Freshness re-merge + gate_reverify + 1672 fix — **exist** (see table above).
- `branch_base_sha` present in task metadata (observed on live reify tasks) — **exists**.
- reify oracle `scripts/verify-pipeline-guard.sh requires-full-gate <files...>` — **exists, exercised** (exit-code contract confirmed; manifest `scripts/verify-pipeline-paths.txt`).
- runs.db merge events (`merge_verify`, `invocation_end`) — **exist**; adding a payload field is additive.
- No novel substrate beyond the above; remaining mapping heuristics are tactical.

## Cross-PRD / cross-repo relationship (G4)

| Other artifact | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| reify `scripts/verify-pipeline-guard.sh` | consumes (CLI) | `requires-full-gate <files...>` exit-code contract | reify owns oracle (no change needed); **this PRD owns the invocation** (M3) — the established "reify ships the primitive, DF wires the invocation" pattern | wired |
| reify `/prd` overlay (`.claude/skills/prd/project.md`) | sibling | authoring-time gate-test⇒registration ordering check (M4) | **reify task, filed separately in the reify tree** (this PRD only references it; no DF work) | queued |
| DF verify-classifier rework (`plans/verify-plan-prd.md`, tasks 2123/2131, pending) | adjacent | `FailureCategory` vs new `disposition` field | each owns its own axis; M2 must not block on 2123/2131 landing (disposition is a separate field) | wired-by-design |
| DF 2357 (verify-base enforcement / two-layer invariants) | adjacent | any skew verdict must key on **dispatch-time** base facts, never the snapshot `two_layer_invariants` surface (stale-cache artifact — see 2357 details) | 2357 | constraint noted |
| reify 5142 / DF 2358 (flaky ledger + auto-file) | consumes labels | skew-classed failures must NOT enter flake statistics/auto-filing | this PRD (M2 sets the label; 2358 filters on it) | queued |

## Contract (G5 — the seam signatures)

```python
class MergeFailureDisposition(StrEnum):
    MAIN_RED = 'main_red'                 # existing preexisting_main_break path; reported through the same field
    INTEGRATION_SKEW = 'integration_skew'
    BRANCH_BUG = 'branch_bug'
    INDETERMINATE = 'indeterminate'

@dataclass(frozen=True)
class SkewEvidence:
    implicated_commits: tuple[str, ...]   # landings on main since merge-base touching the failing tests' files
    failing_tests: tuple[str, ...]
    overlap_files: tuple[str, ...]        # the files that tie failing tests to implicated commits

async def classify_merge_failure_disposition(
    *, verify_result: VerifyResult, branch: str, merge_base_sha: str,
    main_sha: str, preexisting: bool,     # from the EXISTING probe — never re-probed here
) -> tuple[MergeFailureDisposition, SkewEvidence | None]: ...
```

Invariants:
- **I1** probe order: existing preexisting check first; classifier only refines the non-preexisting bucket.
- **I2** read-only: no worktree mutation, no verify execution; git plumbing only.
- **I3** fail-open: any internal error → (`INDETERMINATE`, None); caller behavior then byte-identical to today.
- **I4** the disposition string + implicated shas appear verbatim in: `MergeOutcome.failure_diagnostic`, the task block reason, the escalation body, and the runs.db merge event payload.
- **I5** `INTEGRATION_SKEW` requires implicated_commits non-empty AND the branch's most recent pre-merge verify green; otherwise degrade to `BRANCH_BUG`/`INDETERMINATE`.
- **I6** tripwire (M3): advisory only; ≤1 escalation per landing; oracle absent/erroring → logged no-op; never delays advance.

## Boundary-test sketch (G5 → the integration-gate signal)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Failing test reproduces on main tip | preexisting probe True | disposition `main_red`; existing fix-main escalation path unchanged (fingerprint dedup still fires) |
| 2 | Failing test green on main; its defining file changed on main since merge-base; branch verify was green | forced skew fixture (landing edits a guard test the branch's content violates) | `integration_skew`; SkewEvidence names the landing sha; block reason + escalation body contain sha + overlap files |
| 3 | Failing test green on main; no implicated landings | plain branch bug | `branch_bug`; behavior otherwise identical to today |
| 4 | Classifier raises internally | fault injection | `indeterminate`; outcome byte-identical to pre-PRD path; WARNING logged |
| 5 | Landing touches an oracle-positive file (e.g. reify `scripts/verify.sh`) | ≥1 in-flight task branch diff overlapping the load-bearing set, ≥1 not overlapping | one info escalation naming the landing + only the overlapping task; non-overlapping task absent; advance latency unaffected |
| 6 | Landing oracle-negative (e.g. `crates/foo` only) | — | no tripwire escalation |
| 7 | Event surfacing | scenario-2 run | runs.db merge event payload carries `disposition='integration_skew'` + shas; dashboard/stats query can group by it |

## Decomposition plan (G2 signals; Greek labels → task IDs at decompose)

- **α — `classify_merge_failure_disposition` + `SkewEvidence` (verify.py or a new `merge_disposition.py`) with unit tests.** Modules: orchestrator/src/orchestrator/{verify.py|merge_disposition.py}, tests. Intermediate → unlocks β, γ. Downstream consumer: β.
- **β — wire disposition into the merge-gate failure path + surfaces (THE integration gate).** Modules: merge_queue.py, workflow/escalation composers, tests. **Leaf signal:** boundary rows 1–4 as tests, and end-to-end: a forced-skew merge failure yields `merge_status.failure_diagnostic` and an escalation body containing `integration_skew` + the implicated sha (observable via the escalation MCP record). Depends: α.
- **γ — disposition in runs.db events + stats separation.** Modules: merge_queue.py event emission, dashboard/stat query if trivial. **Leaf signal:** boundary row 7 — a runs.db merge event row carries the disposition field; `sqlite3` query shown in the task proves it. Depends: α (can land parallel to β).
- **δ — pipeline-landing tripwire (M3).** Modules: merge_queue.py post-advance hook, config (per-project oracle command knob), escalation emission, tests. **Leaf signal:** boundary rows 5–6 — in the test harness, an oracle-positive landing produces exactly one info escalation naming the landing and only overlapping in-flight tasks; oracle-negative produces none. Depends: nothing in-batch (parallel to α–γ).
- **ε — docs/triage guidance.** Update the escalation-watcher guidance (disposition vocabulary, "skew ⇒ port, don't debug") — may fold into β if small. Depends: β.

*(reify-side M4 is deliberately NOT in this batch — it is a reify-tree task; see seam table.)*

Capability-manifest binding hints for decompose: α/β bind to `grep:merge_queue.py:165` (probe import), `grep:merge_queue.py:675-689` (fingerprint path), `grep:verify.py:3673`; δ binds to `oracle:scripts/verify-pipeline-paths.txt` + the exit-code check run 2026-07-08; γ binds to an existing `_emit_merge_attempt`/event-payload write site (`merge_queue.py:3338` area).

## Out of scope

- Any new retry/re-verify machinery (D1 — exists).
- Re-implementing main-red detection/dedup (exists; M2 only *reports through* the shared disposition field).
- Fixing the two-layer invariant snapshot staleness or verify-base enforcement (DF 2357).
- The reify-side /prd overlay edit (reify tree, M4).
- Per-task-verify (non-merge) skew attribution — the inter-iteration rebase + 1645 contagion guard cover the workflow side today; extend only if evidence shows a residual class there.

## Open questions (tactical)

1. **Failing-test → file mapping heuristic.** Start with: nextest test-id → crate/tests path + cause_hint file mentions; infra tests map to `tests/infra/<name>.sh`. Decide exact mapping in α; `indeterminate` is the honest fallback.
2. **Where the tripwire's "overlap" is computed** — branch diff vs load-bearing manifest paths, or vs the landing's changed files. Suggested: union, cheap either way. Decide in δ.
3. **Escalation category name** for skew (`integration_skew` vs reuse `infra_issue` with the disposition field). Suggested: distinct category; decide in β against watcher-policy compatibility.
