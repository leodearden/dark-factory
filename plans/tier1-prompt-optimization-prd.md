# Tier-1 Workflow-Prompt Optimization Pilot — PRD

Apply **SkillOpt-style** iterative optimization loops (arXiv 2605.23904; MSR
Asia, May 2026) to the dark-factory workflow prompts whose rollouts are cheap
and whose scores are hard/machine-checkable — **Tier 1 only**: the **reviewer**
role prompt and the **task-curator** gate prompts. Build the reusable machinery
(a per-executor-model optimization loop with a strict-improvement, executor-
scored acceptance gate on held-out data), prove it end-to-end on a cheap
fixture, and leave the two anchor prompts optimizable-and-shippable behind a
provenance-tracked artifact loader with a pipeline-level canary.

*Design input: `~/.claude/spawn-briefs/tier1-prompt-optimization-2026-07-12.md`
(paper digest + repo prompt inventory + APO literature sweep). Companion memory:
`project_skillopt_prompt_optimization_assessment_2026_07_12.md`. Substrate
surveyed on `main` 2026-07-12; the G3 facts below were re-confirmed live while
authoring.*

## 1. Goal (user-observable surface)

When this PRD's batch lands, an operator (Leo) can:

1. **Run** a bounded, per-executor-model optimization loop over either the
   reviewer prompt (executor = opus) or the curator prompt (executor = sonnet),
   entirely offline against a mined corpus — never against live pipeline state.
2. **Read** a loop report showing whether the candidate heuristics block
   **strictly beats** the in-code baseline on a **held-out TEST split that was
   never touched during optimization**, scored on the **actual executor model**,
   reproducibly (N repeats / paired scoring), with the CONTRACT sections proven
   untouched.
3. **Ship** an accepted variant by pinning it as a provenance-stamped artifact
   the live pipeline loads (reviewer review phase; `submit_task` curator
   middleware), with a **one-command rollback** (unpin → in-code constant).
4. **Watch** a post-deploy canary over `data/orchestrator/runs.db`
   (cost-per-done-task, requeue rate, review_cycles, verify_attempts) that flags
   a pipeline-level regression even when the prompt is role-locally better.

The **consumer of every mechanism is real and existing** (G1): the live review
phase and curator middleware already load these prompts; the loop reports are
consumed by the human operator's ship decision; the canary is consumed by the
deploy runbook. Nothing here is an orphan producer.

The honest completion claim is **strict-improvement acceptance on held-out
data + no canary regression** — *not* a specific point gain. No leaf task
asserts "+X F1" (G6).

## 2. Method — SkillOpt discipline the loop must encode

- Frozen **executor** model runs rollouts; a **frontier optimizer** model
  reflects over scored failures/successes in minibatches and proposes edits.
- **Bounded edits per step** ("textual learning rate", default 4 decaying to 2)
  — never wholesale rewrites.
- Candidate accepted **only on strict improvement** of a **held-out selection
  split**, **scored on the actual executor model** (never the optimizer). Ties
  rejected — which presumes low-variance scores (see §4 variance gate).
- **Rejected-edit buffer** so failed edits aren't re-proposed.
- **Train / selection / test split** (paper default 2:1:7); tens of train +
  tens-to-~140 selection items suffice. TEST is scored **once**, at the end,
  never during optimization.
- **Protected sections**: the optimizer may only edit a designated **HEURISTICS
  block**. CONTRACT sections (JSON output schema, escalation/status-transition
  rules, curator `target_fingerprint` + `batch_target_index` + drop/combine/
  create semantics) are **FROZEN** — they encode machine contracts the pipeline
  parses. This is a hard constraint, enforced by construction (see §3, D-3).

## 3. Resolved design decisions

- **D-1 — Loop implementation: bespoke, extend `reviewer_trial`.** Generalize
  the existing `orchestrator/src/orchestrator/evals/reviewer_trial/`
  (variants + scorer + report + runner — the rollout+scoring half already
  exists) into a shared `evals/prompt_opt/` engine of a few hundred lines. No
  external dependency added to a live $615/day system; the engine is
  contract-aware (knows the CONTRACT/HEURISTICS split) and reusable for Tier 2
  later. *Rejected: `pip install skillopt` (external dep + its own loop/WebUI,
  needs adapters to our corpus/scorer/loader anyway, contract-blind).*

- **D-2 — First batch: reviewer + curator only.** The two anchors: highest value
  (reviewer, harness half-built) + cheapest-meaningful replay (curator). The
  four small classifiers (triage, module_tagger, routing, path-scope
  adjudicator) become a **follow-on batch** that reuses the proven loop;
  path-scope adjudicator is pre-flagged as likely too rare (Stage-1 hits only)
  for meaningful replay volume. See §6.

- **D-3 — Contract protection: in-code CONTRACT/HEURISTICS split.** Each prompt
  is refactored into a frozen CONTRACT constant (kept in code) + an editable
  HEURISTICS block. The artifact carries **only** the heuristics block; the
  loader concatenates `CONTRACT + artifact-heuristics` at load. The optimizer is
  handed **only** the heuristics text and can only emit a heuristics block —
  **the contract is un-editable by construction**, not by a fallible
  post-edit validator. *Rejected: whole-prompt edit + contract-token validator
  (relies on the validator catching every drift path).*

- **D-4 — Prompt-artifact loader (the crux mechanism).** A small loader in
  **`shared/`** (reachable by both `orchestrator` and `fused_memory`) resolves
  `(prompt_id, executor_model, harness_version)` → an on-disk heuristics-block
  artifact, or falls back to the in-code constant when nothing is pinned. Every
  artifact carries a **provenance sidecar**: `optimizer_model`, `corpus_hash`,
  `split_seed`, `held_out_TEST_score`, `accept_delta`, `git_sha`, `date`,
  `harness_version`. The **unpin operation is the rollback lever** — no separate
  revert path. **The `executor_model` key is the model resolved *at invocation
  time*, not a static `defaults.yaml` value** — the concurrent
  `adaptive-model-routing` PRD makes the reviewer model dynamic via
  `resolve_route`, so the loader must be called with the router-resolved model
  and the artifact set is **per-model** (a reviewer routed to sonnet must not
  load an opus-optimized block — the exact strong→weak transfer drop §2 warns
  about). Keying on `executor_model` already makes this correct by construction;
  the seam is that the *call site* passes the resolved model. See §5.

- **D-5 — Variance gate (makes strict-improvement honest, G6).** The reviewer
  scorer uses a haiku LLM matcher → noisy; bare tie-rejection would be unsound.
  Acceptance requires **paired scoring** on the held-out split (same examples,
  baseline vs candidate, paired delta) + **N repeats**, and the paired delta
  must exceed a **pre-measured repeatability band** (bootstrap CI / paired
  test). The curator scorer is a **hard signal** (replay-label action agreement)
  and needs fewer repeats, but uses the same gate shape.

- **D-6 — Corpus mining + adjudication.**
  - *Reviewer:* mine `data/orchestrator/runs.db` for diffs the reviewer **PASSed
    but a bug later surfaced** (subsequent debugger/steward/escalation activity)
    → free false-negative labels; augment with the ~10 existing annotated diffs
    and escalation records under `data/escalations/`. Target **~100** diffs
    (≥~50 to make a 2:1:7 split meaningful). Ground truth = **frontier-model
    proposes labels + human spot-check on a subset**, logged.
  - *Curator:* replay `data/reconciliation/tickets.db` (4,019 lifetime
    decisions). **Decisions ≠ ground truth** — same frontier-adjudication +
    human-spot-check protocol before use as labels.

- **D-7 — Canary + rollback (MAS net-negative guard, MAS-PromptBench 2606.23664).**
  A role-locally-better prompt can be net-negative at pipeline level (a reviewer
  that blocks more shifts cost into debugger cycles). Every shipped variant gets
  a canary spec: over a post-deploy window, compare cost-per-done-task /
  requeue-rate / review_cycles / verify_attempts against a rolling pre-deploy
  baseline from `runs.db` (all four already logged); regression beyond threshold
  → **unpin the artifact** (D-4 rollback).

## 4. Pre-conditions (verify at decompose — G3)

Confirmed live 2026-07-12:

- `orchestrator/src/orchestrator/evals/reviewer_trial/` exists with
  `variants.py`, `scorer.py` (recall/precision/F1/blocking-recall via haiku
  matcher), `report.py`, `runner.py` (rollouts via `invoke_agent()` with
  per-spec model control), `corpus/` (~10 annotated diffs, `manifest.json`).
- `data/orchestrator/runs.db` (task_results ≈ 2,086 rows w/ review_cycles &
  verify_attempts; events ≈ 178,931; invocations ≈ 12,295);
  `data/reconciliation/tickets.db` (4,019 curator decisions);
  `data/escalations/` (~2,352 files).
- Reviewer executor = **opus**, $5, 30 turns (`defaults.yaml:162/175`); curator,
  triage, module_tagger = **sonnet** (`defaults.yaml:165-166`). The in-code
  `_reviewer_role(default_model='sonnet')` is overridden per-role by
  `defaults.yaml` — the deployed reviewer executor is opus today, but see P-4:
  the reviewer model is becoming router-resolved, not static.

**Must be verified before the dependent task dispatches:**

- **P-1 (blocks D-4 loader task):** `shared/` (e.g.
  `shared/src/shared/task_metadata.py`) is importable by **both** `orchestrator`
  and `fused_memory` — the loader's home. If it is not, the loader placement is
  a genuine design revisit, not plumbing.
- **P-2 (blocks the loop-engine task):** `invoke_agent()` accepts an arbitrary
  optimizer model distinct from the executor (needed to run a frontier optimizer
  against an opus/sonnet executor). `reviewer_trial/runner.py` already sets model
  per `ReviewerSpec`; confirm the same for the optimizer call.
- **P-3 (HARD-blocks T2, and T7 transitively — reviewer contract; cross-PRD):**
  the reviewer output contract is being replaced by **verdict-servers task
  δ = 2484** (pending) — it edits `roles.py:389-408,:422,:426`, dropping "output
  pure JSON" + the fenced example and moving the verdict to an MCP
  `submit_review_verdict` tool with the verdict *content* schema preserved. T2
  must refactor onto the **post-2484** contract, so **T2 declares a cross-task
  dep on 2484** (T7 inherits it via T2). T6 (the generic engine, tested on a
  synthetic fixture) needs **no** 2484 dep. The reviewer *heuristics* being
  optimized are transport-independent (what to flag, not how to emit it), but
  `reviewer_trial`'s rollout path must mirror the live transport (or explicitly
  justify transport-independence) for score validity.
- **P-4 (shapes D-4 loader + reviewer runs; cross-PRD, soft):** the
  `adaptive-model-routing` PRD (`resolve_route`) makes the reviewer executor
  model dynamic. Confirm the loader call site can obtain the router-resolved
  model, and that a per-model artifact set is acceptable. No hard task dep (the
  loader keying on `executor_model` is forward-compatible), but the reviewer
  optimization run must target whichever model(s) the router can pick.

## 5. Cross-PRD relationship + seam ownership (G4)

Four sibling PRDs were authored/decomposed concurrently on 2026-07-12; their
task ids are live. Seams reconciled below. **The curator anchor (T3, T5) is
clean against all four; every seam is on the reviewer anchor.**

| Sibling PRD (live task ids) | Seam with this PRD | Resolution |
|---|---|---|
| **mcp-verdict-servers** — δ=**2484** (reviewer, pending), α=2481, β=2482, η=2487 | 2484 co-edits `roles.py:389-408,:422,:426` (my T2 target) and **replaces the reviewer output transport** (inline JSON → `submit_review_verdict` MCP tool). **Direct collision.** | **T2 hard-deps 2484** (T7 inherits; P-3) — refactor onto the post-migration contract. Curator is **explicitly excluded** by verdict-servers (2487 preserves the shared `--json-schema` path for `task_curator.py:2002,2093`), so **T3 is clean**. |
| **eval-framework-revival** — κ=**2476** (`reviewer_trial` refresh, pending, low-pri) | 2476 re-runs `reviewer_trial/*` (15-diff corpus, Sonnet-5 candidate). **Dual-ownership of `reviewer_trial/`.** | Ownership split: **this PRD owns `reviewer_trial` corpus-expansion + the optimization loop**; 2476 is a consuming measurement-refresh. **Wire 2476 → T4** at decompose so κ refreshes on the expanded corpus. This PRD must **not** restructure `evals/runner.py`, the benchmark suite, or the Elo judge (eval-revival's core). |
| **adaptive-model-routing** (PRD in flight; `resolve_route`) | Makes the reviewer executor model **dynamic**, breaking a static "pin to opus" assumption; also touches `evals/`, `reviewer_trial/`. | Loader keys on `executor_model` (D-4) → forward-compatible; **P-4** — call site passes the router-resolved model, artifacts are per-model. No hard dep. |
| **harness-backend-reconnect-pi** — T4 (`roles.py` pi `--tools` allowlist) | Additive `roles.py` co-edit; `invoke_agent`/backend forwarding. | Benign — no contract overlap. Opus/sonnet anchors are Claude, so backend-forwarding is irrelevant here; the **follow-on** gpt-4o-mini routing classifier will consume reconnect's backend work. |
| `shared/` prompt-artifact loader (new) | No sibling claims `shared/` or a prompt loader. | **This PRD** owns it (verify P-1); model-key interoperates with `resolve_route` (P-4). |

**Hand-off note to the eval-harness session (eval-revival)** — do **not** file
as tasks here: more benchmark tasks; sandboxed recon Stage-1/2 replay (Tier 2);
score-repeatability features *inside the workflow-eval harness* (distinct from
this PRD's §4 variance gate). One reciprocal ask: eval-revival's κ (2476) should
depend on this PRD's T4 (shared corpus) — flagged for whoever decomposes/refines
2476.

## 6. Out of scope

- **Tier 3** — architect/implementer tight loops (rollout = full $5–20 task run;
  only ~5 benchmark tasks → a selection gate would be noise).
- **Tier 2** — recon stage-1/2 prompts (blocked on sandboxed replay).
- **The four small classifiers** — triage, module_tagger, routing classifier
  (gpt-4o-mini; needs `OPENAI_API_KEY`), path-scope adjudicator. **Follow-on
  batch** reusing this loop. Path-scope adjudicator flagged as probably too rare
  for replay data; the routing classifier is the natural first classifier
  (cleanest hard label, cheapest rollout, proves cross-model-family).
- **Any change to prompt CONTRACT sections**; **any live-pipeline rollout**
  (all rollouts are offline replays); **any eval-harness restructuring**
  (`evals/runner.py` / benchmark suite / Elo judge — eval-revival's);
  **any LLM-judge-only acceptance gate** where a hard signal exists.
- **The reviewer output *transport*** (inline JSON → MCP `submit_review_verdict`)
  — owned by **mcp-verdict-servers (2484)**; this PRD refactors the reviewer
  prompt's *heuristics* onto whatever contract 2484 lands, it does not design the
  transport (P-3, §5).
- **Running the expensive real optimization loops and the ship/rollback
  decisions themselves** — these are **operator runbook** steps (§8), not
  orchestrator tasks: reviewer runs are $300–800 / overnight-to-weekend and the
  ship call is Leo's human judgment (G1 consumer). The batch builds and
  smoke-proves the machinery; the operator drives the costly runs.

## 7. Decomposition plan (one leaf per bullet; each names its G2 signal)

Dependencies in brackets. Signals are the **user-observable** proof of
completion the decompose session will attach as `user_observable_signal`.

- **T1 — Shared prompt-artifact loader + provenance schema + fallback.**
  *[deps: none; P-1]* — **Signal:** a test pins a fake heuristics-block artifact
  for `(prompt_id, executor_model, harness_version)`, and the loader returns
  `CONTRACT + artifact-heuristics` composed correctly with the full provenance
  sidecar recorded; with nothing pinned it returns the in-code constant
  verbatim; **unpinning a pinned artifact restores the in-code constant** (the
  rollback lever). Consumer: T2, T3, T6, T8.

- **T2 — Reviewer prompt CONTRACT/HEURISTICS split + loader wiring
  (`agents/roles.py`).** *[deps: T1; **dark_factory:2484** (verdict-servers δ —
  P-3)]* — **Signal:** `_reviewer_role` builds its system prompt via the loader;
  the reviewer's **live** output contract (post-2484: the `submit_review_verdict`
  tool instruction + verdict-field schema + blocking-definition) lives in a
  frozen CONTRACT constant and the editable guidance in a HEURISTICS block; a
  `reviewer_trial` rollout still produces a valid verdict via the live transport
  (parity regression); a **boundary test proves an artifact that tries to alter a
  CONTRACT token has no effect on the composed contract** (G5 two-way). Consumer:
  live review phase. *Rebase onto 2484's `roles.py` edit — do not freeze the
  pre-2484 "output pure JSON" text.*

- **T3 — Curator prompt CONTRACT/HEURISTICS split + loader wiring
  (`middleware/task_curator.py`).** *[deps: T1]* — **Signal:** `_SYSTEM_PROMPT`
  and `_BATCH_SYSTEM_PROMPT` are split so `target_fingerprint`,
  `batch_target_index`, drop/combine/create semantics, and the output schema are
  a frozen CONTRACT and the combine heuristics + positive-signals are the
  editable HEURISTICS block; the curator builds prompts via the loader; a replay
  of a fixed candidate yields the **same action** as pre-refactor (parity); the
  same CONTRACT-immutability boundary test passes. Consumer: `submit_task`
  middleware.

- **T4 — Reviewer corpus expansion to ~100 + gold labels + adjudication log.**
  *[deps: none]* — **Signal:** the corpus manifest lists ≥~50 (target ~100)
  ground-truth-annotated diffs with a recorded 2:1:7 train/selection/test split;
  mining provenance (the `runs.db` query + escalation refs behind each
  false-negative label) is captured; an adjudication log shows frontier-proposed
  labels + human spot-check on a documented subset; the existing scorer runs
  green over the expanded corpus. Consumer: the reviewer optimization run; **also
  eval-revival's κ (2476), which should dep on this task** (§5 — shared corpus).
  *Transport-independent (labels are about which bugs exist, not how the reviewer
  emits), so this task needs **no** dep on 2484 and can proceed in parallel.*

- **T5 — Curator replay corpus + curator scorer (hard signal).**
  *[deps: T3, T6]* — **Signal:** a curator replay corpus is built from
  `tickets.db` with N frontier-adjudicated + human-spot-checked labeled
  decisions, split 2:1:7; a curator scorer computes **action-match agreement**
  (drop/combine/create match + combine-target correctness) against those labels
  on held-out data, conforming to T6's scorer Protocol; scorer runs green.
  Consumer: the curator optimization run.

- **T6 — Bespoke `evals/prompt_opt/` loop engine + scorer Protocol + variance
  gate.** *[deps: T1; P-2]* — **Signal:** a dry-run integration test on a tiny
  synthetic fixture corpus shows the engine run a SkillOpt loop over a pluggable
  `(corpus, scorer, executor_model, heuristics_block)`: bounded edits
  (textual-LR 4→2), a rejected-edit buffer, frontier-optimizer minibatch
  reflection, and acceptance **only** on paired + N-repeat strict improvement of
  the **selection** split scored on the **executor** model, delta exceeding the
  measured repeatability band; the run emits a heuristics-block candidate + an
  accept/reject record + provenance, and **never mutates the CONTRACT**.
  Consumer: T5, T7, the operator runs, Tier 2.

- **T7 — End-to-end loop acceptance smoke on a fixture (cheap, hermetic).**
  *[deps: T2, T6]* (T2 already carries the 2484 dep transitively) — **Signal:** the full stack (loader → reviewer HEURISTICS →
  loop → scorer → report) runs on a ≤3-diff fixture corpus and produces a report
  with a held-out verdict, provenance, and a proof the CONTRACT was untouched —
  demonstrating the machinery works end-to-end **without** a $300–800 real run.
  Consumer: gates the operator's confidence to launch the real runs.

- **T8 — Canary metric-comparison + deploy/rollback runbook mechanism.**
  *[deps: T1]* — **Signal:** a canary script reads `runs.db` and computes
  cost-per-done-task / requeue-rate / review_cycles / verify_attempts over a
  post-deploy window vs a rolling baseline window and emits a **pass / regress**
  verdict against a documented threshold; the runbook documents pin → watch →
  **unpin-on-regress** using the T1 rollback lever. Consumer: operator deploy
  runbook (§8).

**Approach for the batch is B+H** (G5): explicit contracts (the artifact schema
and the CONTRACT/HEURISTICS boundary) + two-way boundary tests on the
high-stakes seams (loader↔reviewer and loader↔curator contract-immutability;
the `shared/` import seam exercised from both packages).

## 8. Operator runbook (out of batch — consumes the machinery)

Not filed as tasks (high-cost, human-judgment; G1 consumer = Leo):

1. Run the **reviewer** loop (needs T2, T4, T6) — offline, overnight/weekend,
   ~$300–800.
2. Run the **curator** loop (needs T3, T5, T6) — offline, ~$30–150.
3. Read each loop report; decide ship per the **held-out TEST verdict**.
4. Pin the accepted artifact (D-4); run the **canary** (T8) over the deploy
   window; **unpin on regression**.
5. **Re-validate on every model upgrade** — the optimized prompt is a build
   artifact pinned to `(prompt, model, harness)`; a model bump invalidates it.
   The loop is deliberately cheap enough to re-run at each bump.

## 9. Open (tactical / implementation-time) questions

- Exact **N repeats** and the numeric repeatability-band threshold for the
  reviewer variance gate — measure the haiku-matcher noise floor empirically in
  T4/T6 and set from data.
- Precise **mining heuristics** for reviewer false-negatives (which
  debugger/steward/escalation signals downstream of a PASS most reliably mark a
  missed bug) — refine against `runs.db` during T4.
- Curator adjudication **subset size** for human spot-check vs frontier-only —
  pick to bound human effort while keeping label confidence; decide in T5.
- Canary **window length** and per-metric regression thresholds — calibrate
  against `runs.db` baseline variance in T8.
- Artifact **on-disk layout** under `shared/` (path scheme + manifest format) —
  a T1 implementation detail, not a design fork.

## 10. Note for the decompose session

The orchestrator does **not** yet read the `user_observable_signal` /
`consumer_ref` / substrate-confirmed metadata fields — they are substrate for a
future tracking-infra session. File the batch with `planning_mode=True` on every
task, wire all deps, commit the capability manifest beside this PRD, then flip
the whole batch `deferred → pending` in one `commit_planning`.

**Intra-batch deps:** T2→T1, T3→T1, T5→T3+T6, T6→T1, T7→T2+T6, T8→T1.
**Cross-PRD dep (qualified `project_id:task_id` form, §Cross-project in
CLAUDE.md):** T2→`dark_factory:2484` (verdict-servers δ — the reviewer contract;
P-3). T7 inherits it via T2. No other cross-PRD dep is a hard blocker
(adaptive-model-routing is design-compat via the loader's model key, P-4; the
reciprocal 2476→T4 dep is eval-revival's to wire).

Re-verify **P-1** (shared-package import), **P-2** (`invoke_agent` optimizer
model), and **P-3/P-4** (§4) at decompose — P-1/P-2/P-3 are hard G3
pre-conditions; the T2→2484 edge means the reviewer sub-chain will not dispatch
until verdict-servers δ is `done` (correct — it must land first).
