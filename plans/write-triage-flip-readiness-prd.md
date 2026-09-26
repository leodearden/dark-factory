# PRD: write-triage flip readiness — honest instrument, candidate-bound attach, reranker, slate, judge arms, predicate gates

**Project:** dark-factory (fused-memory). **Status:** active, 2026-09-23. **Type:** extension of `docs/prds/memory-write-path-convergence.md` thrust A (leaves α–γ landed; decision D10's flip gate, task **3169**, held since 2026-08-27). **Approach:** B+H — the reranker amends contract C1, so the seam carries a contract and two-way boundary tests.
**Origin:** esc-3169-1 briefing (`plans/write-triage-flip-briefing-2026-09-22.md`) and Leo's rulings of 2026-09-23: option (a) filed as task **5794**; leaf ζ (3135) raised to high; F, G, H all worth pursuing; every decision gate a predicate script with numeric thresholds and its branch actions written on the gate; 5794 and the terminal gate wired as hard dependencies of 3169.
**Code anchors** verified against main `ca06d49340` (2026-09-23). Main moves fast — cite-by-symbol; re-locate at implementation time.

## 1. Goal (G1 consumer + user-observable surface)

The operator at task 3169 rules on artifacts that measure what production will do, and every remaining flip objection is either closed by a landed leaf or decided by a gate whose predicate, thresholds and branch actions are written on the gate task. Observable:

- `fused-memory/calibration/write_triage_judge_accuracy_report.{json,md}` carries `production_shape` (strict correct-attach, any-attach, wrong-record attach, false attach, band split, middle-band accuracy) measured with `--slate-mode retrieved`, plus a per-case JSONL beside it; `write_triage_calibration_report.{json,md}` carries recall@k under production retrieval parameters up to k=50 over the full 84-record population with rotated canonicals resolved. Consumers: the 3169 gate reader (D10) and gates Γ1–Γ4.
- `add_memory` with the flag on attaches to the candidate the judge named (task 5794) and, when the reranker is enabled, to the reranker's choice of target; a topic-anchored canonical enters the slate with a real score instead of being dropped (τ).
- Each gate Γ1–Γ4 is a `task_kind=deterministic` task whose `before_done` runs `scripts/check_write_triage_readiness_gate.py`; the exit code is the machine contract; the trailing compact JSON names the failed checks and the actions for the branch taken. A failing gate files `milestone_check_failed` born-at-L2 with the full verdict in its detail. Consumers: the L2 resolver and the 3169 reader.
- Terminal: Γ4 `done` satisfies a new dependency of task 3169; esc-3169-1 stays pending until the operator rules. This PRD never resolves esc-3169-1 or esc-3169-2.

## 2. Background — evidence (pointer, INV-5)

Measured 2026-09-22/23 on the reify fixture (104 records, 20 clusters, 84 non-canonical records) against the live reify store; full tables in `plans/write-triage-flip-briefing-2026-09-22.md` and the session's `out/harness/summary.md`, `out/recall/summary.md`. Load-bearing facts:

- `write_triage.py::triage_write` files every non-`stored` verdict against `decision.canonical_id`, the band's max-cosine record, regardless of which candidate the judge reasoned about. With real retrieval (k=20, three Mem0 categories, `anchor_topics=False`) the judge attaches 72/75 duplicates but **12/75 land on the right canonical (strict 0.16; 14/75 = 0.19 once rotated canonicals are resolved)**; 67/81 attaches (0.83) land on an arbitrary live memory; 8/11 deterministic `restated` attaches are wrong with no judge involved. Doubling `judge_candidate_count` to 10 left strict attach at 12/75 to the case: nothing done to the prompt moves a target decided before the prompt is built.
- Elision (`_FIELD_CHARS = 1_200`) is **refuted** as a driver: un-eliding flipped 1 of 26 misses and cost 89% more input tokens.
- Production retrieval recall over the 84 records with aliases (this is the "recall@1 = 0.19" every threshold below cites; the strict rate above is a different 0.19): @1 0.19, @5 0.38, @10 0.55, @20 0.79, @50 0.87. The calibrator's legacy mode (search default `anchor_topics=True`, i.e. task 3111's topic pin) puts the canonical at rank 1 for 48/63 records; write triage opts out because a pinned record carries no `store_score` and `decide_band` drops it.
- Seeded mode with the target present and marked: strict 49/75 = 0.653 — the judge's rate of naming the right record when it can see it.
- `t_low` (0.523) is inert: the minimum observed max-cosine over 20 neighbours is 0.674, so every triaged write is judged or attached.
- Status quo with the flag off catches ~nothing (e2 replay 0/12 at 0.92, max observed 0.899; κ 3136 ruled FAIL on inverted ANN AUC 0.25). Cosine hard negatives outrank true duplicates (`fused-memory/data/memory-evals/ann-threshold-calibration-20260811/threshold_curve.csv`).

## 3. Sketch of approach

Seven leaves and four predicate gates. κ1 lands the honest instrument. τ consumes the topic pin the triage seam discards. ρ1 measures every admissible reranker class on the retrieved-slate harness; Γ2 decides whether ρ2 integrates one behind a flag with a `band_signal` knob. σ gives the judge its own endpoint; μ runs the configuration matrix (signal × slate width × judge arm) after 5794 and writes `best_config.json`; Γ3 decides flip readiness from it; ν (Jev provider arm) is filed `deferred` and promoted only on Γ3's fail branch. Γ4 re-runs the flip preconditions and is a dependency of 3169.

## 4. Contracts (H)

### C1' — retrieval → rerank → band → judge → attach (amends convergence PRD C1)

> **Retrieval width, ranking signal, slate width and judge arm are four orthogonal knobs; the attach target is always the record a scorer named, never a positional default.**

- `retrieve_candidates` unchanged: `candidate_k` (20), the three Mem0-primary categories, `stores=['mem0']`. τ adds: the triage call passes `anchor_topics=True`; a pinned record surfaced without a `store_score` is scored explicitly (embed once, cosine against the entry) before `decide_band`, never dropped.
- New `rerank_candidates(entry: str, candidates: list[MemoryResult]) -> list[ScoredCandidate]` in `fused_memory/server/write_triage_rerank.py`, behind `write_triage.reranker.enabled` (default false). Pure in (entry text, candidate texts); returns every input candidate with `rerank_score ∈ [0, 1]`; deterministic for fixed inputs and model; bounded by `write_triage.reranker.timeout_seconds`; any failure or timeout falls open to cosine order and is counted through `write_triage.py::TriageFailOpenCounter` (INV-4, one storm anchor).
- `write_triage.band_signal ∈ {cosine, rerank}` (default cosine). Under `rerank`, `t_high`/`t_low` are order statistics of the reranker score written by `calibrate_write_triage.py --signal rerank`, and the calibration report names the signal. Under `cosine`, bands are unchanged and the reranker only orders the slate and selects the attach target.
- Slate = top `judge_candidate_count` by the signal in force; the signal's winner is always included.
- Attach target: the candidate the judge named (task 5794) when the judge ran; the signal's winner when the deterministic band answered. Never `candidates[0]`.
- Ack unchanged (`routed`, `canonical_id`); the triage decision's provenance gains `signal` and `reranker_model`.
- Config leaves added, each registered in `config/reload.py::RELOADABLE_FIELDS` with a live-read test: `write_triage.reranker.enabled`, `write_triage.reranker.model`, `write_triage.reranker.timeout_seconds`, `write_triage.band_signal` (ρ2) and `write_triage.judge_api_url` (σ; overrides `llm.providers.<provider>.api_url` for the judge only, read in `write_triage_judge.py::_provider_credentials`).

### C2' — measurement contract

> **A committed accuracy artifact describes production or says which production parameter it departs from.**

- `eval_write_triage_judge.py --slate-mode retrieved` is the arbiter for the flip; `seeded` is retained to reproduce historical artifacts and says so in its caveats.
- Provenance keys in `PROVENANCE_KEYS` order: fixture path, slate mode, field_chars, project id, t_high, t_low, candidate_k, judge_candidate_count, judge provider/model, signal, reranker model, canonical_absent, degraded/self retrievals, aliases path.
- `production_shape` is computed once, by `eval_write_triage_judge.py::score_attachments`, with numerators and denominators in the JSON. μ calls the same function per configuration and `best_config.json` **copies the winning configuration's `production_shape` block verbatim** under the same nested keys, plus a `selection` block: `signal`, `judge_candidate_count`, `judge_provider`, `judge_model`, `reranker_model`, `canonical_on_slate_rate`, `p95_write_seconds`, `cost_per_write_usd`, `residual`. The gates read those keys and nothing else.
- `residual` rule (μ): `retrieval` when `canonical_on_slate_rate < 0.60`; `judge` when `canonical_on_slate_rate ≥ 0.60` and `duplicate_attach.strict_rate / canonical_on_slate_rate < 0.60`; else `none`. Basis: recall@20 is 0.79, so a slate that shows the canonical less than 60% of the time is retrieval-bound; 0.653 is the judge's measured strict-when-present rate, so a conditional rate under 0.60 is a judge regression.

### Boundary-test sketch (integration signals for ρ2 and τ)

| scenario | preconditions | postconditions |
|---|---|---|
| reranker on, judge names candidate 3 | flag on, `reranker.enabled`, fixture cluster whose canonical is cosine rank 3 | ack `canonical_id` = that canonical; child `parent_id` = it; no write to rank-1 |
| reranker times out | `reranker.timeout_seconds` tiny | attach proceeds on cosine order; fail-open counter +1; ack shape unchanged |
| `band_signal: rerank` with rerank-calibrated bands | calibrator run with `--signal rerank` | a near-verbatim duplicate acks `restated` without a judge call; report names `signal: rerank` |
| `band_signal: cosine` with reranker on | default bands | band unchanged vs flag-off cosine; only slate order and target differ |
| topic-pinned canonical, cosine rank > 20 | record topic-tagged; pin promotes it | pinned record is in the slate with a numeric score; band computed over it |
| judge names an id outside the slate | fake judge | counted fail-open; standalone store |
| reranker disabled | default config | byte-identical decisions to main today (regression) |

## 5. Resolved design decisions (Leo, 2026-09-23)

- **D1 Admissible reranker classes for ρ1/Γ2:** local cross-encoder in-process (torch; Apache-2.0 candidates with ≥8k context: Qwen3-Reranker-0.6B, mxbai-rerank-base-v2, bge-reranker-v2-m3), LLM pairwise via gpt-4o-mini with concurrent fan-out, hosted reranker APIs (Jina, Voyage, Cohere; skipped without credentials, recorded), and TypeSafe Jev `choice` (skipped without a key, recorded). ρ1 measures every arm it can reach; Γ2 selects on measured rank-1 and latency.
- **D2 Band signal:** build the `band_signal` knob; rule at Γ2 from ρ1's measured separation. No recalibration before evidence exists.
- **D3 Gates:** all four are predicate scripts (`scripts/check_write_triage_readiness_gate.py`, committed with this PRD, mode 100755). The exit code is the machine contract; the branch actions are advisory prose the L2 resolver executes. **The home of each threshold is the gate task's `before_done.args`**; §9 records the filing-time values and the manifest points at the args. A re-base after a fail edits the args and appends a dated line to §9. A failing gate stays blocked, owned by its own born-at-L2 escalation, until its inputs change or the resolver takes the fail branch written on it. Where a fail branch must let a dependent proceed, it **removes the dependency edge** (`remove_dependency`) before cancelling anything, because a cancelled dependency still carries its `delivered_checks` against main (`scheduler.py::_deps_satisfied`).
- **D4 3169 dependencies:** 5794 and Γ4 are wired as hard dependencies of 3169 at decompose. Verified chain: `resolve_issue(resume)` → `harness.py::_on_escalation_resolved` → `pending` → `scheduler.py::_eligible_for_dispatch` → `_deps_satisfied` → only then `deterministic_runner.py::_run_predicate`. 3169's three existing deps are all `done`, so these two edges become the only mechanical hold once esc-3169-1 is resolved. Neither may ever be cancelled (esc-3169-2's warning applies). Note for the decomposer: 3169's record carries both `always_escalates: true` and `before_done.kind: predicate`, a combination the submit guard now forbids; it is a legacy record and the runner ignores `always_escalates` on the predicate path. Do not "repair" it while wiring.
- **D5 5277:** κ1 absorbs workstreams A4 (per-case persistence) and A3 (eval coverage); 5277 is amended at decompose to drop them. A2, B and C stay with 5277; μ's local-endpoint arm depends on 5277 C only softly.
- **D6 5616:** κ1 does not change the elision marker; 5616 keeps the helper unification. Elision is not a lever (§2).
- **D7 4599:** τ implements 4599's option (a) through task 3111's pin rather than the phrase-cluster guard; 4599 is closed at decompose as superseded by τ with a pointer. ζ (3135/4493) keeps its own scope; the flip objection "topic guard off with no replacement" is answered by τ + ζ together, and Γ3's detail lists ζ's status.
- **D8 Jev:** ν is filed `deferred`; promoted only by Γ3's fail branch when `residual == judge` and a key exists.
- **D9 Fixture:** the reify fixture with the alias sidecar (`fused-memory/tests/fixtures/write_triage_calibration.canonical_aliases.json`, κ1) is the population for every threshold; 5547's dark_factory fixture, when it lands, is reported alongside, never substituted silently.
- **D10 Cancellations at the flip** (unchanged from 3169's record): the reject-guard family 4729/4773/4179/4255/4739 is cancelled only when the flag is flipped, by the 3169 resolver, never by this PRD.
- **D11 Gate authoring facts** (bind every Γ task): the runner's cwd is `project_root`, so `--report` paths are repo-relative; `before_done.timeout_secs` must exceed the script's `--subcheck-timeout` (default 90 s) or the runner files `infra_issue` with no verdict; the gates carry no `metadata.milestone` (the predicate sub-path dispatches on `before_done.kind` alone; dependencies do the gating).

## 6. Pre-conditions / substrate (G3)

Verified on main `ca06d49340`: `eval_write_triage_judge.py`, `calibrate_write_triage.py` and their tests; the live reify Mem0 store; `write_triage.judge_provider`/`judge_model` in `config/reload.py::RELOADABLE_FIELDS`; `write_triage_judge.py::_provider_credentials` reads `llm.providers.<provider>.api_url`; `write_triage_judge.py::_KNOWN_PROVIDERS` = openai, anthropic; `MemoryService.search(anchor_topics=...)` (task 3111, `f62767929b`); `write_triage.py::TriageFailOpenCounter`; `scripts/check_write_triage_flip_preconditions.sh` exits 0 on main (6.3 s, 2026-09-23); ollama at localhost:11434 serving gemma4:12b / qwen3:14b (eval-only; it cannot serve a cross-encoder).

Not present, queued as prerequisites in this batch: per-case dump, `--slate-mode retrieved`, production-parity recall and aliases (κ1 — code exists on `iact/wt-option-c-harness` @ `72aba0725b` and `iact/wt-option-c-recall` @ `695f808694`, ~2,400 changed lines, both 12 commits behind main and to be rebased and re-verified); torch-class packages (ρ1 adds an optional dependency group; ρ2 promotes to runtime only if Γ2 passes for a local arm); a judge-specific endpoint (σ); a Jev provider arm (ν, deferred); the gate predicate script (committed with this PRD).

## 7. Out of scope

Flipping `write_triage.enabled`; resolving esc-3169-1/-2; ζ itself (3135/4493, high); 5547; the reject-guard family; 4916's consolidation-gate detector (it may consume `rerank_candidates` later through the service seam, INV-5); 5277 A2/B/C; 5616; ε (3131, dep-gated on 3169).

## 8. Cross-PRD / seam ownership (G4)

| Seam / mechanism | Owner | This PRD's edge |
|---|---|---|
| Judge verdict names its candidate (option a) | task 5794 (pending, high) | consumes; Γ2, μ, Γ4 depend on it; wired as a 3169 dep (D4) |
| Gate probe that the attach consumes the id | task 4949 (in-progress) | consumes; Γ4 depends on it |
| Eval per-case persistence (A4), eval coverage (A3) | 5277 → absorbed by κ1 (D5) | produces; 5277 amended |
| Contested consumer (A2), connection reuse (B), openai 400 compat (C) | 5277 (pending) | none; μ skips a 400ing endpoint and records it |
| Prompt truncation marker unification | 5616 (pending) | none (D6) |
| Topic-cluster signal at the flip | 4599 (pending) → superseded by τ (D7); ζ 3135/4493 unchanged | produces τ |
| Consolidation-gate detector | 4916 (pending) | may consume `rerank_candidates` later; no second implementation |
| Retrieval anchoring pin | task 3111 (done), `docs/prds/memory-metadata-vocabulary.md` | τ consumes through `MemoryService.search`; no re-implementation |
| Calibration fixture, per-category cutoffs | 5547 (pending) | none (D9) |
| Flip gate | task 3169 (blocked), esc-3169-1, esc-3169-2 | terminal consumer; gains deps 5794 + Γ4; never resolved here |
| Local model serving | `plans/local-memory-models-eval-prd.md` (eval-only, 7 of 10 leaves pending) | μ's local arm is a measurement, never a production pin |
| Judge provider seam | `write_triage_judge.py::_KNOWN_PROVIDERS`, `_call_llm` | σ adds an endpoint knob; ν adds an arm (deferred) |

## 9. Decomposition plan (one bullet per task; signals are the G2 gate)

Deps: Γ1←κ1; τ←κ1; ρ1←Γ1; Γ2←ρ1, 5794; ρ2←Γ2; σ←—; μ←5794, Γ1, τ, σ, ρ2; Γ3←μ; ν←Γ3 (deferred); Γ4←Γ3, 4949, 5794; 3169←5794, Γ4 (out-of-batch edges added at decompose). Acyclic.

- **κ1 — land the honest instrument (carrier).** Rebase `iact/wt-option-c-harness` @ `72aba0725b` and `iact/wt-option-c-recall` @ `695f808694` onto main, re-verify (354 + 225 tests were green on the pre-rebase trees), fresh-agent review, merge; regenerate both committed artifacts (`--slate-mode retrieved`, `--retrieval production`, aliases, k up to 50) so the two traceability excuses (`_ADDED_AFTER_THE_COMMITTED_RUN`) can be deleted; commit the per-case JSONL beside the report; amend 5277 (D5). *Signal:* committed `write_triage_judge_accuracy_report.json` has `provenance.slate_mode == retrieved` and `production_shape.duplicate_attach.n == 75`; `write_triage_calibration_report.json` has `provenance.retrieval_mode == production` and a `recall_at_k.per_k[k=20]` row.
- **Γ1 — gate: instrument landed** (deterministic, predicate; deps κ1). Checks exactly κ1's signal keys. *On pass:* ρ1 dispatches (τ dispatches on κ1 alone). *On fail:* re-open κ1; ρ1 waits. No cancellations.
- **τ — consume the topic pin in triage** (normal; deps κ1). `retrieve_candidates` asks `anchor_topics=True` and scores each pinned record explicitly so `decide_band` never drops it; contract rows 5 and 7; supersedes 4599 (D7). *Signal:* boundary rows 5 and 7 pass; a retrieved-mode report on the fixture shows canonical-on-slate at k=5 above κ1's 32/84.
- **ρ1 — measure reranker arms** (normal, eval-only; deps Γ1). New `fused-memory/scripts/eval_write_triage_reranker.py` scoring every D1 arm it can reach over the 84-record retrieved slates: rank-1 and rank-5 rate of canonical-or-alias, AUC true-vs-hard-negative on the reranker score, p50/p95 latency for 20 pairs on this host, cost per write; arms without credentials or dependencies recorded as `skipped` with the reason. Report at `fused-memory/calibration/write_triage_reranker_report.json` with per-arm rows and a `best` block. **Selection rule for `best`:** arg-max rank-1 among arms with `p95_seconds ≤ 3.0`; if no arm qualifies, the lowest-latency measured arm with `qualified: false`. Adds torch-class packages as an optional dependency group only. *Signal:* the committed report has one row per D1 arm with `status ∈ {measured, skipped}` and a `best` block carrying `rank1_rate`, `p95_seconds`, `qualified`.
- **Γ2 — gate: reranker verdict** (deterministic, predicate; deps ρ1, 5794). Requirements: `best.rank1_rate >= 0.40` and `best.p95_seconds <= 3.0`. *Basis:* cosine rank-1 is 0.19 on the same population; recall@20 (0.79) is the ceiling any reorder of the 20 can reach; 0.40 is about half that ceiling and double cosine; 3.0 s leaves the 10 s judge timeout room for a judge call. Provisional (G6); the escalation carries the observed values. *On pass:* ρ2 dispatches. *On fail:* `remove_dependency(μ, ρ2)`, then cancel ρ2, then cancel Γ2 with the verdict recorded on 3169's metadata; μ discovers the absence and runs cosine-only.
- **ρ2 — integrate the reranker behind a flag** (normal; deps Γ2). `write_triage_rerank.py::rerank_candidates`, the four config leaves in C1' (registered in `RELOADABLE_FIELDS`, live-read tests), calibrator `--signal rerank`, boundary rows 1–4 and 7, fail-open through the existing counter; promotes the chosen arm's dependency to runtime if local. *Signal:* boundary rows pass; flag-off decisions byte-identical to main.
- **σ — judge endpoint knob** (normal, small; no deps). `write_triage.judge_api_url` read by `_provider_credentials` for the judge only, registered in `RELOADABLE_FIELDS`; test that the main LLM arm is unaffected. *Signal:* judge resolves its own URL when set; `llm.providers.openai.api_url` unchanged.
- **μ — configuration matrix and best_config** (normal, eval-only; deps 5794, Γ1, τ, σ, ρ2). **Self-discovering:** runs the retrieved-mode harness over signal ∈ {cosine} ∪ {rerank iff `write_triage_rerank.py` is on main, else recorded `skipped`} × `judge_candidate_count` ∈ {5, 10, 20} × judge arm ∈ {gpt-4o-mini, one stronger OpenAI model, claude-haiku via the anthropic arm, one local model via σ (skipped if the endpoint 400s or is down, recorded)}; writes `fused-memory/calibration/write_triage_config_matrix.json` (consumer: the 3169 reader and Γ3's escalation detail) and `write_triage_best_config.json` per C2'. *Signal:* both artifacts committed; `best_config.json` carries `production_shape` and `selection` with `residual`.
- **Γ3 — gate: flip readiness** (deterministic, predicate; deps μ). Requirements on `best_config.json`: `production_shape.duplicate_attach.strict_rate >= 0.50`, `production_shape.wrong_record_attach.rate_of_attaches <= 0.50`, `selection.p95_write_seconds <= 5.0`. *Basis:* 0.79 (recall@20 ceiling) × 0.653 (seeded strict-when-present) = 0.516, so 0.50 sits at the estimate, not under it; the wrong-record bound says at least half of all attaches land on the right canonical (today 17%); the pair is jointly feasible — 38 strict of 75 with ≤ 76 attaches — and both are provisional (G6). 5.0 s is half the judge timeout. *On pass:* Γ4 dispatches; the operator applies `selection` at the flip. *On fail:* the gate's escalation is the owner of the hold; if `residual == judge` and a Jev key exists, un-defer ν, re-run μ after it lands; if `residual == retrieval`, file a reranker follow-up naming the measured gap; either way record on 3169's metadata; this gate stays blocked until `best_config.json` changes.
- **ν — Jev provider arm** (normal, filed `deferred`; deps Γ3). Third arm in `_call_llm` over `POST /v1/systemone` with a `choice` over the slate plus a relation `choice`; adapter into `parse_judge_verdict`; schema + credentials; tests. *Signal:* with `judge_provider: jev` a fixture write's verdict names a slate id; μ's matrix gains a Jev row.
- **Γ4 — gate: flip preconditions on the refreshed tree** (deterministic, predicate; deps Γ3, 4949, 5794). Sub-check `scripts/check_write_triage_flip_preconditions.sh` exits 0 (`--subcheck-timeout 90`, `before_done.timeout_secs 180`); the committed judge artifact is retrieved-mode; `best_config.json` exists with `selection.residual`. *On pass:* `done`; 3169's new dependency is satisfied and the operator rules on esc-3169-1 with the artifacts named here. *On fail:* the escalation names the failing item; nothing is cancelled.

Sizing (overlay bands): κ1 carries ~2,400 changed lines already written; τ, σ under ~300; ρ1, ρ2, μ, ν within 300–1,500; gates are deterministic and exempt from the floor.

## 10. Open questions (tactical)

1. **Where torch lives if a local arm wins Γ2.** Optional dependency group vs runtime; decide in ρ2 from the measured cold-start and VRAM contention with whisper-writer. Suggested: optional group + lazy import + a startup health check.
2. **`eval_write_triage_judge.py` is 1,924 lines on the κ1 branch** (1,080 on main). Heuristic 14 alarm: measure and name the partition at κ1's review; a split is legitimate only if every file passes heuristic 13.
3. **Pairwise LLM fan-out shape** (ρ2, if that arm wins): concurrency limit and rate-limit handling; the 10 s budget is the constraint.
4. **`reranker.timeout_seconds` default.** Set from ρ1's measured p95 with headroom, not guessed.
5. **Pinned-record scoring in τ.** Embed-and-cosine per pin vs asking Qdrant for the score; decide on measured latency.
