# Agent Legibility Survey — sources of agent confusion, 2026-06-12 → 2026-07-12

**Date:** 2026-07-13
**Method:** multi-agent mining of last-month session logs (interactive + orchestrated), clustering by shared underlying cause, adversarial verification of every high-severity cluster against current main.
**Companion artifact:** `docs/legibility/confusion-codebook.yaml` — the persistent, machine-readable cause registry seeded from this survey (for future saturation-based batches).

## Corpus and method

- 8,468 dark-factory sessions in the window (≈2.5 GB): ~7,380 reconciliation-stage agents, 342 curator/classifier runs, 338 escalation-watcher rotations, 190 orchestrated task/aux agents (warm lanes, block-time investigators, lock predictors, claim verifiers), 86 interactive sessions.
- Every session was scored with a zero-token signal pass (tool errors, self-corrections, not-found errors, guard trips, user interrupts); 584 high-signal sessions stratified across class and week were mined by 31 agents, plus a distiller that turned curated memory lessons into a known-cause taxonomy.
- 336 confusion incidents → 16 clusters (10 high / 6 medium severity) → all 10 high clusters adversarially verified against current main (targeted file reads + git log since 2026-06-12). The 6 medium clusters are **mined but unverified** — treat their details as leads, not facts.
- Cost note: the survey exposed its own inefficiency (multi-MB JSONL groped per-agent; Fable used for extraction). §5 fixes this for future runs.

## Executive summary

1. **The factory's feedback loop is working.** Of 10 verified high-severity clusters, one is *fully fixed* on main and eight are *partially fixed* with the dominant failure modes already addressed — mostly by work landed inside the survey window (tasks 2037, 1756, 5026, 2041-2044, 2131, 2265/2188, 2360, 2372, verify-plan-prd W7). The survey's chief marginal value is (a) confirming those fixes empirically, (b) pinpointing the small residuals each fix left behind, and (c) surfacing cross-cutting causes no single incident made visible.
2. **The recurring root causes are structural, not informational.** Signposts would not have prevented most of these. The five load-bearing patterns:
   - **Contracts living in prose** (task descriptions declaring routing intent, tool pseudocode in prompts, eligibility rules buried in dispatcher code) instead of in machine-checked places.
   - **Stories reconstructed by log-scraping** (block reports, failure categories, judge verdicts) instead of structured facts emitted at the point of failure.
   - **Point-in-time state never re-corroborated against ground truth** (git, DB) before agents act on it.
   - **Fail-soft defaults that absorb outages silently** (verdict fallbacks, degrade-to-create, suppression guards without storm escapes).
   - **Duplicated lock-step logic drifting apart** (branch-prefix normalization, already-merged guards, result-envelope conventions).
3. **A handful of one-line/one-file fixes carry outsized value** — e.g. `CLAUDE.md:358` points every run-review judge at a docs directory that has never existed (7/8 judge sessions in one batch died on it); `targeted.py:564` silently verifies the wrong text; the merge worker still parks WIP on the *shared* stash stack (a human's stash was consumed 6 days ago).

## 1. Verified findings (all 10 high-severity clusters)

Status legend: ✅ fixed · ◐ partially fixed (residuals listed) · ✳ substantially open.

### 1.1 ◐ One-shot subagent contract failures (`one-shot-subagent-contract`, 17 incidents)
Claim-verification and run-review judge agents were handed contracts their runtime can't honor (prose JSON shims with all tools disallowed, `max_turns=1`, a phantom docs path). Verified current state: loud-warning hardening landed (tasks 1811/1989), but four small defects are live:
- `CLAUDE.md:358` still points at `fused-memory/docs/reconciliation/` which **does not exist and never has** — and judges deliberately run with cwd that auto-loads CLAUDE.md. One batch: 8/8 June judge sessions burned their single turn trying to read it.
- `targeted.py:564` uses `task.get('details', description)` — the default never fires on empty string; should be `task.get('details') or description`.
- `verify.py:48,51` accepts a `project_id` parameter that is never read.
- `judge.py:279-309` still fabricates "Manual review recommended" on parse failure instead of escalating loudly.
**Fix: one S-effort cleanup task covering all four.** Do *not* re-architect the claude_cli AgentLoop (deliberate, already hardened).

### 1.2 ✅ Watcher capability envelope (`watcher-capability-envelope`, 18 incidents)
Autonomous watcher rotations used to rediscover their own limits (level-{0,1} cap, no Write/Edit, git-write denials) by failure, every rotation. **Fixed on main** by tasks 2041-2044 (landed 2026-07-03): the envelope is now stated in `skills/escalation-watcher-auto/SKILL.md:12-19`, enforced via server-stamped identity and the level gate at `escalation/server.py:560-635`. Residual (S): no `triaged_at`/`triaged_by` ack marker on escalation records, so consecutive rotations still re-derive identical dispositions.

### 1.3 ◐ Block-report misattribution (`block-report-misattribution`, 16 incidents)
Verify/block reports built by positional log-scraping mislabeled failures (everything read as `tree_sitter_generate_error`), duplicated summary into detail, and excerpted PASS-walls. Verified: task 2131's table-driven `FailureCategory` ladder fixed the headline. Two live residuals (both S):
- `workflow.py` ~2071 discards the `_StewardReescalated` payload → "Detail: Steward re-escalated to human" duplication.
- `verify.py:2087` excerpts a fixed `test_output[-3000:]` tail instead of FAIL/error/panic-anchored windows.
Also worth adding while there: semaphore/ENOSPC/SIGBUS/psi-gate patterns to the classifier, and command+SHA+dispatch-cycle stamping on attempt summaries.

### 1.4 ◐ Verify-scope asymmetry (`verify-scope-asymmetry`, 16 incidents)
Scoped merge-verify lands red main; full task-verify charges pre-existing/infra failures to innocent tasks; unregistered paths fall to a broken `__fallback__` lane. Verified: **this is `plans/verify-plan-prd.md` (W7), already in flight** with real landed progress (CATEGORY_POLICY, preexisting-on-main probe used by merge_queue + workflow, MERGE_VERIFY_RED dry-run proposals, fallback lane now runs ruff+pyright). **Do not re-file.** Remaining: finish the W7 spine (β/δ/ε/θ/ι), then file the explicitly out-of-scope gap separately — sibling/reverse-dep test coverage at merge-gate time (or labeling the verdict as scoped). Cheap interim signpost: the per-attempt verify plan should print "scoped to changed files; sibling tests NOT run".

### 1.5 ◐ Unverified task premises (`unverified-task-premises`, 15 incidents)
Tasks filed/dispatched on premises nobody checked against the tree: recon-invented code models, `found_on_main` phantom dones, dep `done` read as capability-delivered. Verified: the phantom-done facet is substantially fixed (tasks 1180, 2372, 2245). Open:
- `premise_lint()` exists in `recon_self_model.py` (landed 4 days ago) with **zero call sites** — recon-reliability-prd W5 is `Status: deferred`. Wiring it into the recon Stage-1/2 filing path is cheap and completes designed work.
- Dep-done ≠ capability-delivered has no fix anywhere: needs a designed dispatch-time probe (M, new concept — see open questions).

### 1.6 ◐ Warm-lane cache incoherence (`warm-lane-cache-incoherence`, 13 incidents)
Phantom compile errors from stale seeded `target/` dirs, shared caches, ENOSPC, no GC (reify lanes, but the harness is ours). Verified: heavily worked in-window (reify tasks 4419→5175: coherence guard in seed-warm-lane.sh, GC/disk-guard/Tier-3 reclaim, tree-sitter readiness gate). Residual: sccache is still one shared daemon across lanes with no isolation (small follow-up), and the base-commit coherence-guard pattern should be generalized to any other warmth-transfer path.

### 1.7 ◐ Guards assert unverified diagnoses (`guards-assert-unverified-diagnoses`, 14 incidents)
Detector text states *conclusions* computed from evidence never executed or since mutated ("last-green" SHAs never re-run — one recommended a destructive rewind to a commit that also failed; sudo failures swallowed by `2>/dev/null || true` became "no XFS magic"). Verified: the two most concrete incidents fixed (1755 storm counter, 5026 already-merged genuine-check). The generalizable residual (M): the escalation API (`escalate_blocker`/`escalate_info`, `server.py:382-458`) has **no structured evidence field** — free-form strings invite diagnosis-as-fact. Add raw-observations fields (SHA, measured_at, rerun result) and generalize `merge_liveness.py`'s consecutive-streak gate to other CRITICAL filers.

### 1.8 ◐ Merge-lane state not git-corroborated (`merge-state-not-git-corroborated`, 13 incidents)
`merge_status` "unknown" read as an outcome; dead in-flight slots; already_merged false positives. Verified: dominant modes **root-cause fixed** (task 2037 Tier-3.5 git-authority corroboration; 1756 slot release on every terminal path; 5026 shared genuine-check). Two S residuals:
- `escalation/server.py:1000` — merge_request's own already_merged fast-path still raw-concats `f'{prefix}{branch}'` instead of `canonical_queued_branch_name` (the last un-normalized site; task 1911 never touched server.py).
- `server.py:1586` — durable terminal records omit the verify-failure reason, so a post-restart poll on `blocked` is reason-less.

### 1.9 ◐ Machine-operated main checkout (`machine-operated-main-checkout`, 12 incidents)
The single main checkout is concurrently mutated by the merge worker, startup reconciler, hooks, and humans — with the invariants written nowhere. Verified: the worktree-reaping facet is fixed (2265/2182/2188 claimant + WIP-retention). Live and unmitigated:
- **Merge worker parks WIP on the shared `stash@{0}` stack** (`git_ops.py:6653/7018`); a stash-pop conflict landed markers on main 6 days ago (13674d3c68). Root-cause fix (M): private ref — `git stash create` + `git update-ref refs/dark-factory/merge-park` — never the shared LIFO stack.
- `acquire_next` (`scheduler.py:4843`) never consults claimant liveness at dispatch (task/2242 probes this, unmerged).
- CLAUDE.md has **no commit-safety section** (git commit --only, hook timeout ≥300s, "never `git stash` in project_root — park WIP as commits").
- `hooks/project-checks:60-68` runs 3× pyright on every commit including plans/-only ones (the 2-minute Bash-timeout hazard).

### 1.10 ◐ Investigator context & budget (`investigator-context-and-budget`, 14 incidents)
Every re-dispatch is amnesiac: `metadata.dry_run_proposals` and prior plan analysis never reach successor prompts (one retry architect burned $12 re-deriving persisted analysis); a wall clock used to kill deep investigations with zero output; prompts teach the two-dot diff trap. Verified: wall-clock half substantially addressed (task 2360 progress-extension + infra_failure classification, 11 commits since 07-02). Open:
- Feed `dry_run_proposals[-1]` into **retry/resume** prompt builders only (`briefing.py:63,95,785,969`) — first dispatch deliberately omits proposals to avoid anchoring (C-A1); scope the change and add a test asserting first-dispatch prompts stay proposal-free.
- `skills/unblock/SKILL.md:92` still says `git diff main..HEAD` — fix to merge-base-anchored (trivial; this exact lesson is in curated memory but never reached the prompt).

## 2. Mined-but-unverified clusters (medium severity — verify before acting)

| Cluster | Core claim | Cheapest probe |
|---|---|---|
| `recon-prompt-schema-drift` (16) | Stage prompts show pseudocode diverging from real MCP schemas; identifier naming inconsistent across sibling tools (`name` vs `entity_name`, `memory_id` vs returned `id`) | diff prompt examples against live tool schemas |
| `recon-lifecycle-state-gaps` (13) | run_id-scoped flag sweep silently drops markers stamped with the filing run's id; event buffer redelivers processed windows; remediation payload omits project_root (`memory_consolidator.py:764` one-liner) | check the 764 call site + flag-sweep scoping |
| `fused-memory-api-traps` (12) | `update_task` metadata replace-not-merge default (tasks 1827/1828 already filed), envelope asymmetry `get_statuses` vs `get_external_statuses`, silent search filters | confirm 1827/1828 status before anything |
| `watcher-loop-harness-mismatch` (15) | watcher CLI initial-scan fires on ANY pending item forcing hand-built 17–29-id exclude lists; no `--since/--baseline`; 540s wait killed by 2-min Bash default | read `escalation.watcher` CLI args |
| `subagent-runner-protocol-defects` (13) | module-tagger double-wraps StructuredOutput 9/11 sessions (param name == schema key); falsy-`[]` predictions never persisted so the same tasks re-tag forever; usage-limit resume storms (one 226-cycle stop-hook livelock) | count `{"tasks":{"tasks":` in tagger transcripts; check `_tag_task_modules` persistence |
| `prose-routing-intent` (12) | Tasks whose own text says "deterministic / DO NOT IMPLEMENT / operator --apply" still dispatch into the TDD architect pipeline; architect can only block → churn | grep recent blocked tasks for these markers |

Notable one-offs worth individual attention (full list in the codebook): the CLAUDE.md repo-map gap (`<pkg>/src/<pkg>` double-nesting and skills/-location wrong-first-path probes across *every* agent kind — a 5-line layout note removes the class); injected `currentDate` is local-time while all orchestrator timestamps are UTC; an AFK-spawned /unblock treated its own interactivity as human authorization for an irreversible live-data prune; leaked tool-call XML persisted into a task description and flowed into downstream prompts; a remediation agent hand-patched production `stage2.py` from a stale "PENDING CODE FIX" memory.

## 3. Cross-cutting root causes (what to change as a *pattern*)

1. **Contracts must live where they're consumed, machine-checked.** The three worst historical incidents in this class (simple-task fast path dead for ~7,950 tasks behind an unadvertised title regex; routing intent in prose; watcher envelope) share one shape. Standing rule for new features: any eligibility/routing/capability contract needs either an enforced schema field or a submit-time lint — not description prose or dispatcher-internal heuristics.
2. **Emit structured facts at the failure point; stop re-deriving stories from logs.** Failure category from exit code/signal + step identity; escalations carry raw observations (SHA, measured_at, rerun result) separate from diagnosis; reports quote FAIL-anchored windows. This is also what makes future confusion surveys a `GROUP BY` instead of transcript archaeology.
3. **Ground-truth corroboration before action** is steadily winning (merge Tier-3.5, already_merged genuine-check, phantom-done gates) — extend the same move to dispatch time (premise lint, dep-capability probe, claimant liveness).
4. **Every fail-soft path needs a storm-escape.** Suppression/degradation without a rate-threshold escalation hides total outages (judge fallback verdicts, curator degrade-to-create). Loud-over-silent is already the standing directive; apply it at guard design time.
5. **Dedupe lock-step logic.** `canonical_queued_branch_name` exists yet one site still concatenates; the already-merged guard was duplicated until 5026; envelope conventions differ across sibling tools. Prefer extraction over documentation (owner-ratified preference).

## 4. Action plan

### Tier 1 — quick wins (S, low risk; file as `complexity=simple` tasks)
1. **CLAUDE.md batch edit:** fix the `:358` phantom docs pointer; add a 5-line top-level repo map (`<pkg>/src/<pkg>` nesting, skills/ location, escalation/ vs fused-memory/); add a commit-safety section (git commit --only, hook timeout, "stash stack is machine-operated — park WIP as commits").
2. **fused-memory verifier/judge cleanup:** `targeted.py:564` `or`-fallback; delete dead `project_id` param in `verify.py`; `judge.py` parse-failure → loud high-severity finding.
3. **Orchestrator block-report fixes:** steward re-escalation detail passthrough (`workflow.py` ~2071); FAIL-anchored excerpting (`verify.py:2087`); add semaphore/ENOSPC/SIGBUS patterns + command/SHA stamping.
4. **Escalation server residuals:** `server.py:1000` → `canonical_queued_branch_name`; failure reason into `_OPTIONAL_TERMINAL_META_FIELDS`; `triaged_at`/`triaged_by` ack marker.
5. **skills/unblock/SKILL.md:92** merge-base-anchored diff.
6. **hooks/project-checks** path-filter: skip 3× pyright for plans/-and-docs-only commits.
7. `memory_consolidator.py:764` remediation-payload project_root directive (verify inline — from an unverified cluster).

### Tier 2 — root-cause fixes (M)
8. **Merge worker private stash ref** (`git_ops.py:6649-6666, 7004-7027`) — the one unmitigated sub-cause with a live incident this week.
9. **Feed-forward for retries:** `dry_run_proposals[-1]` into retry/resume briefing prompts with the C-A1 anti-anchoring test.
10. **Wire `premise_lint()`** into the recon Stage-1/2 filing path (revive recon-reliability W5-ξ; detector already built).
11. **Structured evidence fields on the escalation API** + generalize the streak-gate for CRITICAL filers.
12. **Claimant liveness in `acquire_next`** (finish/merge the task/2242 probe).
13. **Submit-time routing lint** for prose routing markers ("deterministic", "DO NOT IMPLEMENT", "--apply", conditional gates) → reject or re-route (extends `prose-routing-intent`; verify cluster details first).

### Tier 3 — larger / PRD-shaped
14. **Finish verify-plan-prd W7** (in flight — track; don't re-file), then file the sibling/reverse-dep merge-verify scope-widening PRD it excludes.
15. **Dispatch-time dep-capability probe** — design first (what does "dep delivered its promised capability" mean mechanically?).
16. **Recon prompt/schema regeneration** — literal tool-call examples generated from live schemas so they cannot drift + identifier aliasing across the fused-memory surface (verify cluster 10 first).

### Tier 5 → §5 — survey infrastructure (agreed 2026-07-13)
- **Codebook** seeded at `docs/legibility/confusion-codebook.yaml` (this survey). Future batches code incidents against it; novel-rate is the saturation signal.
- **Deterministic digest extractor** (`scripts/`): transcript JSONL → 5-15KB confusion digest (user corrections, error neighborhoods, self-corrections, retry loops). Zero tokens; ~20× cheaper downstream reads.
- **Nightly Haiku trickle** (cron): code yesterday's ~15 highest-signal sessions against the codebook; monthly synthesis becomes nearly free.
- **Fleet model routing:** miners/triage on Sonnet/Haiku; Fable only for clustering/synthesis. Check usage-window headroom before launching fleets.
- **Emission-time reason codes** (covered by Tier-1 #3 / Tier-2 #11) are the root-cause fix that eventually makes transcript mining the exception.

## 5. Open questions

1. **Dep-capability probe semantics** (Tier-3 #15): is "capability delivered" checkable mechanically (e.g., dep's diff touched the files/symbols the dependent's description cites), or does it need an LLM judgment at dispatch time?
2. **Merge-verify scope widening** (Tier-3 #14): appetite for running reverse-dep/sibling tests at merge-gate (wall-clock cost per merge) vs. only labeling verdicts as scoped?
3. **Anchoring vs amnesia** (Tier-2 #9): C-A1 deliberately hides prior proposals from first-dispatch architects. Is retry/resume-only feed-forward the right line, or should stewards also see prior proposals?
4. **`acquire_next` claimant gate** (Tier-2 #12): should the scheduler refuse dispatch into a claimed worktree even when the task status is `blocked` (not just `in-progress`) — i.e., is a human /unblock claim absolute?
5. **AFK authorization** (one-off): should /unblock require an explicit human-presence ack before irreversible live-data operations? Today "interactive session" is treated as consent — the mined incident says that's wrong.
6. **Unverified mediums**: verify all 6 as the next saturation batch (cheap, Sonnet), or verify lazily as each fix is picked up?
7. **Survey cadence**: nightly trickle + monthly synthesis, or trickle + on-demand synthesis only?
8. **Filing**: shall I queue Tier 1 as `complexity=simple` tasks (one per bullet) and Tier 2 via /prd gates? Nothing has been filed yet.

## Appendix — saturation evidence & cost

- 12 of 16 clusters matched causes already in the curated memory taxonomy (with new mechanisms/residuals); 4 were genuinely novel as *clusters* (one-shot subagent contract, investigator feed-forward, subagent runner protocol, prose-routing intent). The head of the distribution is saturating — consistent with the planned codebook + saturation-batch design replacing big-bang mining.
- Spend: ~7.3M subagent tokens across two mining runs (1.6M lost to a launch into an exhausted usage window), plus ~0.4M Sonnet for the 4 supplementary verifications. Verified process fixes captured in memory (`feedback_fleet_economics_tiered_models_saturation_sampling`).
- Raw artifacts: full workflow result JSON at the session task store; per-agent returns in the run journal; inventory manifest + slice lists in the session scratchpad (`legibility/`).

---

## Addendum — 2026-07-13 (same day): all 16 clusters verified; Tier 1+2 filed

Owner ratified: eager verification of the 6 medium clusters, filing of Tiers 1+2, codebook in-repo, nightly trickle acceptable, Sonnet/Haiku fleet routing.

**Medium-cluster verdicts** (all Sonnet-verified against main; codebook updated in place):
- `recon-prompt-schema-drift` — **still fully present**: `_RECON_REPORT_TOOL_GUIDANCE` examples omit the required `run_id` since 09500a38e0 (7 weeks, survived one reviewer pass) → task 2559 (generate examples from live signatures + identifier aliasing).
- `recon-lifecycle-state-gaps` — mostly mitigated; one confirmed live one-liner (third payload builder missed by task 2150) → task 2552. Re-prioritizing 2436/2437 recommended.
- `fused-memory-api-traps` — 1827/1828 landed; **planning_mode tasks are permanently invisible to search_tasks** (curator corpus never indexes them) → task 2562.
- `watcher-loop-harness-mismatch` — lease fixed duplicate watchers (07-10); hand-built exclude lists + Bash-timeout kills confirmed post-fix → task 2560.
- `subagent-runner-protocol-defects` — **still fully present**: falsy-`[]` predictions never persist (re-tag forever), no deterministic filter, schema double-wrap → task 2561.
- `prose-routing-intent` — guards exist but only on the curator path; planning_mode bypasses all of them → task 2563 (warn-first submit-boundary lint).

**Filed (17 new):** 2547 CLAUDE.md fixes · 2548 verifier/judge cleanup · 2549 verify excerpting/patterns · 2550 unblock diff recipe · 2551 hooks path-filter · 2552 remediation project_root · 2553 steward detail passthrough (dep 2248) · 2554 escalation-server merge residuals · 2555 triaged_at marker · 2556 merge-worker private stash ref · 2557 dry_run_proposals feed-forward · 2558 structured evidence + streak gate · 2559 recon guidance schema-gen · 2560 watcher CLI baseline/wrapper · 2561 module tagger · 2562 search_tasks indexing · 2563 routing lint. **Curator routing:** claimant dispatch gate combined into existing 2408; /unblock irreversibility guardrail dropped in favor of in-progress 2509 (same incident).

**Q1 finding — capability manifest** (`<prd>.capability-manifest.md`, prd skill decompose Step 2.5): right artifact for dep-capability checks; `references/decompose-mode.md:42` anticipated a dispatch-time consumer and `:143` concedes the orchestrator reads none of it today. To automate: (a) machine-readable sidecar stamped with real task ids at commit_planning; (b) per-capability `delivered_check` (pattern-anchored grep/command that must pass once the producer lands — authoring-time `file:line` anchors go stale); (c) scheduler dep-gate runs the producer's delivered_checks when a dep flips done, withhold + escalate on failure. PRD-shaped (M).

**Q2 finding — merge-gate breadth correction:** the current merge gate is *narrow*, not broad. `verify_plan.py` (~:293-330): pytest full-module-suite only when conftest/test-data touched; otherwise file-scoped to touched **test** files; **a source-only diff runs zero pytest at the gate** ("no collectable test files touched — nothing to run"). Broad coverage lives in task-level verify (pre-merge, older base) + post-merge unscoped *typechecks* + after-the-fact main sweeps. Reverse-dep/sibling test selection at the merge gate is therefore a coverage *restoration* at bounded wall-clock, not a widening luxury — the mined red-mains came exactly from this hole.
