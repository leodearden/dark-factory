# Local memory-models eval — pre-registration

PRD-MARKER:local-memory-models-eval preregistration

**Task:** 3719 (LME-ζ) of `plans/local-memory-models-eval-prd.md`. **Committed:** 2026-10-07,
before any candidate arm has run. The decision quantities below come from
`plans/local-memory-models-eval-controls/preregistration-inputs.json`. That file is the
authority: where this doc and the file disagree, the file wins and this doc is the defect.
The contract test `fused-memory/tests/arm_harness/test_lme_control_artifacts.py`
re-derives the file byte for byte from the committed control runs.

## 1. Status, ordering and the code-sha pin

This doc is committed before any candidate arm runs. Every candidate artifact (η
screening, θ full runs, ι embedding arms) carries this doc's commit sha as
`preregistration_sha`. ε's `check_preregistration_sha` refuses a candidate whose sha does
not carry this file, and the `ArmSpec` schema refuses a candidate with none. The sha to use
is the commit on `main` that last touched this file:
`git log -1 --format=%H main -- plans/local-memory-models-eval-preregistration.md`.

**Code-sha pin.** η and θ candidate runs use the controls' code sha,
`055e9c15a0d756704017b0d77ffbe4185155402f`. That sha is on `main`, so it cannot be rebased
away. Run `harness.py` from a clean checkout pinned there, passing that checkout as
`--repo-root`. The comparison checks (`check_single_code_sha`) admit only arms run at
one code sha. Margins are judged afterwards by the committed `margins.py` on `main`; that
is offline post-processing and does not touch the replay.

If a later harness fix forces a new code sha, the following happens in this order:
1. Re-run the incumbent control pair at that sha, exactly as recorded in the controls
   README.
2. Re-derive the margins by the same committed code (`harness.py preregister`).
3. Amend this doc in a commit that predates every candidate artifact at that sha.

The formula, the floors, the directions and the envelope never change. Only their
measured inputs do.

## 2. Named control artifacts

Everything lives under `plans/local-memory-models-eval-controls/`; its README is the
provenance record.

| Artifact | Role |
|---|---|
| `runs/incumbent-generic-a/20261007T011000Z/` | Control A: GenericClient, full corpus. Its graph `evalmem_lme_ref_incumbent_a` is FROZEN as ι's reference |
| `runs/incumbent-generic-b/20261007T013800Z/` | Control B: GenericClient, full corpus, replayed with `--reference-outcomes` A, so it carries the incumbent's graph-sameness self-agreement |
| `runs/incumbent-generic-20/20261007T020600Z/` | GenericClient on the first 20 manifest episodes, which is also η's screening subset |
| `runs/incumbent-openai-20/20261007T020900Z/` | OpenAIClient on the same 20 episodes |
| `runs/incumbent-generic-20/20261007T020600Z/parity/incumbent-openai-20/metrics/` | Client-class parity deltas, GenericClient − OpenAIClient |
| `preregistration-inputs.json` | Margins, envelope and call profile: `harness.py preregister` over A and B |
| `frozen-reference.json` | Node count, edge count and topology hash of `evalmem_lme_ref_incumbent_a` |

All four runs share code sha `055e9c15a0d756704017b0d77ffbe4185155402f`, corpus sha
`850cf7c937c745b0d224a0ba975efc1c69080ad61d84c5369d06287cdd16f2e8` (δ's committed N = 200
manifest), concurrency 3 and `with-indices`.

## 3. Margin formula

    margin_m = max(2 · σ_control(m), floor_m)

The formula is implemented in `fused_memory/arm_harness/margins.py::derive_margins`. That
module also holds the gated-metric table (`GATED_METRICS`), the one home of each metric's
pass direction. `MarginEntry.admits` applies the direction.

- **σ_control.** For conformance-rate, episode-failure-rate and retrieval-utility, σ is the
  run pair's sample standard deviation, |a − b| / √2. Graph-sameness is pairwise by
  nature: B measured against A yields exactly one incumbent self-agreement value, and
  Jaccard is symmetric, so A against B adds nothing. For it, σ is that one observation's
  per-episode standard error, stdev(entity Jaccards) / √n. `preregister` recomputes those
  Jaccards from A's and B's `outcomes.jsonl` rather than trusting a file, and refuses
  unless their mean and count match B's recorded graph-sameness. That is what shows B was
  measured against A.
- **floor.** For a proportion, the floor is 1 / min(denominator), the smallest step the
  metric can express. A margin below one item's step would let a single item decide the
  verdict, which is noise rather than measurement. Floors are absent for scalars
  (graph-sameness here, mrr on the embedding axis).
- **The honest limitation.** A run pair is one degree of freedom. |a − b| / √2 is an
  unbiased but very noisy estimate of σ, and two identical controls give σ = 0. That
  happened here for episode-failure-rate and retrieval-utility. The floor is what stops
  such a margin collapsing to zero. This pre-registration accepts that weakness rather
  than buying a third replay. The margins are therefore tight where the incumbent was
  perfectly repeatable, and they are stated, not tuned.

| metric | direction | σ source | control values | reference | σ | floor | margin | admits a candidate value |
|---|---|---|---|---|---|---|---|---|
| `conformance-rate` | lower is worse | run pair | 1.0, 0.9994475138121547 | 0.9997237569060773 | 0.0003906667299373259 | 0.0005546311702717693 (1/1803) | 0.0007813334598746518 | ≥ reference − margin |
| `episode-failure-rate` | higher is worse | run pair | 0.0, 0.0 | 0.0 | 0.0 | 0.005 (1/200) | 0.005 | ≤ reference + margin |
| `graph-sameness` | lower is worse | episode SE (n = 200) | 0.8058450390324771 | 0.8058450390324771 | 0.015182768573058603 | none | 0.030365537146117207 | ≥ reference − margin |
| `retrieval-utility` | lower is worse | run pair | 0.995, 0.995 | 0.995 | 0.0 | 0.005 (1/200) | 0.005 | ≥ reference − margin |

The last column is the direction rule as `MarginEntry.admits` applies it. Computing
reference ± margin by hand gives: conformance ≥ 0.99894…, failure ≤ 0.005 (one failed
episode in 200), graph-sameness ≥ 0.77548…, retrieval ≥ 0.99 (at most two misses in 200).
`admits` is the authority for the exact boundary.

## 4. Latency envelope

The envelope bounds **episode-latency-p95, measured WARM and UNDER LOAD**:
- a full-corpus run (θ) or a screening run (η) at concurrency 3;
- the arm's engine warmed by `harness.py smoke` immediately before `run`.

It is **not** α's health-probe latency. That figure is a single sample, taken engine-warm
and prefix-cold, and it is not comparable across arms. qwen3.5-9b at `reasoning: off`
measured 2849 ms cold against ~350 ms warm there (`scripts/local-model-serving/README.md`,
the note naming the mode each figure belongs to).

    p95_bound = episode_timeout / LATENCY_HEADROOM = 120 s / 2.0 = 60000 ms

- **Anchor.** `queue.write_timeout_seconds` and `queue.backend_write_timeout_seconds` in
  `fused-memory/config/config.yaml`, both 120 s. The harness reads the latter as each run's
  `episode_timeout_s`, which is a hard failure: an episode past it counts against
  episode-failure-rate.
- **Why 2.0.** The tail between p95 and max needs room before that hard failure. A local
  server's tail has unknown shape, and its p95 says nothing about its max. Deriving the
  headroom from the incumbent's own tail was rejected, because its max is dominated by
  OpenAI network outliers that say nothing about a local server.
- **Cross-check.** The incumbent's own warm p95 under load was 42828.00681702793 ms, the
  worse of the two controls. Its slowest ok episode took 92826.04033034295 ms, a max/p95
  ratio of 2.17. The incumbent therefore sits inside the envelope with its whole tail inside
  the timeout. The ratio is also close to the 2.0 headroom: an arm sitting exactly at the
  bound with the incumbent's tail shape would push its slowest episodes to the timeout, and
  episode-failure-rate would count them. The envelope and the failure gate cover the tail
  together. `preregistration.py` refuses to produce inputs whose incumbent falls outside the
  envelope: an envelope the incumbent fails is not a valid pre-registration.
- `LatencyEnvelope.admits` is strict: p95 < 60000 ms.

## 5. LLM-axis decision rule (θ)

A candidate arm is **non-inferior** if and only if both hold:
1. every gated metric is admitted by its `MarginEntry.admits`;
2. its full-corpus warm p95 under load is inside the envelope, by `LatencyEnvelope.admits`.

Among non-inferior arms, **availability decides**: running locally rather than depending
on the metered API. That is the question the PRD exists to answer.

- **Run shape.** θ candidate runs match the controls on every symmetric setting
  `check_arm_config_symmetry` reads: concurrency 3, `with-indices`, temperature 0.0,
  max_tokens 4096, the incumbent embedder, and graphiti semaphore limit 20.
- **Reference.** Graph-sameness is measured against control A, passed as
  `--reference-outcomes plans/local-memory-models-eval-controls/runs/incumbent-generic-a/20261007T011000Z/outcomes.jsonl`.
- **Reported, not gated:** episode-latency-p50, tokens-per-episode and usd-per-episode.
- **Several non-inferior arms are all reported.** No quality ranking among them is
  pre-registered.
- **Client-class delta.** It is reported against each margin and is not gated. Every
  candidate arm runs GenericClient, and so did the controls A and B, so the margin
  comparison is like for like. The committed parity deltas say how far production's
  OpenAIClient sits from that, so a reader can see whether the client choice alone would
  have moved a verdict.

## 6. Embedding axis (ι)

The formula, floors and code are the same: `derive_margins` over two embedding-axis
control runs.

- **σ source.** σ comes from ι's own incumbent-embedder control pair: two re-embeds of the
  frozen graph `evalmem_lme_ref_incumbent_a` with `text-embedding-3-small`, `arm_role`
  control, run BEFORE any candidate embedding arm. σ between graphs A and B would include
  LLM graph-construction noise that a re-embed of one frozen topology does not have, and
  would make the embedding margins lenient. ι commits that pair's inputs beside its report.
  This doc fixes the rule now; ι produces the numbers.
- **Gated metrics**, each per index configuration: known-item-recall@5 and
  known-item-recall@10 (proportions, floor 1/n) and mrr (scalar, no floor), all lower is
  worse. If the pair's mrr values coincide, the mrr margin is 0 and the rule on mrr is
  plain "not below the incumbent". That is accepted as pre-registered.
- **with-indices decides.** embedding-only is reported with its confound stated: it is
  today's production retrieval path on the unindexed live graphs, not the configuration a
  cutover would ship.
- **query-embed-latency-p95 envelope:** `queue.search_timeout_seconds` 30 s / 2.0 =
  15000 ms, on the same warm-under-load definition.
- **Reported:** reembed-throughput, projected to full-backfill wall-clock.
- **Frozen-graph contract.** Before re-embedding, ι re-runs
  `harness.py topology --graph evalmem_lme_ref_incumbent_a` and requires output
  byte-identical to `frozen-reference.json`.

## 7. A clean negative is an acceptable verdict

A clean negative is an acceptable verdict (Leo, 2026-08-10). "No local arm is
non-inferior" is a result this eval can return and λ can record. There is no remediation
clause:
- never widen the slate to find a passing arm;
- never relax a margin, the envelope or a floor after a candidate has run;
- never re-run a candidate hoping for a better draw.

## 8. η survivor rule

The rule is authored against the **three-arm** LLM slate in
`scripts/local-model-serving/arms.yaml` (qwen3.5-9b, phi-4-14b, moe-stretch).

- **The ≤3 cap does not bind.** Three candidates against a cap of three means its
  selectivity is exactly nil, and no ranking rule is pre-registered to break a tie that
  cannot occur.
- **Four absolute gates** decide survival instead. Each is reported per arm with its
  margin, and an arm survives only if it passes all four:
  1. **Conformance smoke.** `harness.py smoke --arm-spec <arm>` exits 0: a schema-valid
     response, and the validator's negative control rejected.
  2. **VRAM fit.** α's arm-footprint verdict (`lms_vram.evaluate_budget`) passes against
     the free VRAM measured before that arm started.
  3. **Context fit.** The longest prompt the arm's own server reports on the screening
     subset, plus max_tokens 4096, must be ≤ the arm's `max_model_len` in arms.yaml. The
     4096-token output reservation is the margin. Limitation, stated: the longest prompt
     is measured on the 20-episode subset only, so a longer prompt elsewhere in the corpus
     can still surface in θ as an episode failure, where it counts against
     episode-failure-rate.
  4. **Throughput floor.** The screening run's warm p95 under load lies inside the same
     60000 ms envelope (§4).
- **Screening subset.** The first 20 episodes in manifest order, which is the client-class
  subset, so the incumbent's own numbers on it are already committed. Screening runs use
  `--limit 20 --concurrency 3 --index-configuration with-indices` at the pinned code sha.
- **Outcomes.** n ≥ 1 survivors go to θ. n = 0 means the screening report states the
  negative verdict, and λ records it. Neither is an error state. This agrees with task
  3720's DEGENERATE-OUTCOME HANDLING.
- **No rule depends on the slate growing.** A throughput failure is a result. Revisiting
  SGLang after a vLLM throughput failure is Leo's call, not an automatic step.

## 9. Reasoning mode per arm (ruled here)

The ruling is a deterministic rule over the controls' measured workload. It takes nothing
from η's results.

**Workload.** The controls made `calls_p95` = 14 LLM calls per ok episode (p50 9, max 39,
over 400 ok episodes; `call_profile` in `preregistration-inputs.json`).

**Price.** For each mode:

    admitted calls/episode = floor(60 s / per-call latency)

The per-call figure is α's upper per-call figure from 2026-08-06 for that mode
(arms.yaml comments and `scripts/local-model-serving/README.md`).

| arm | mode | per-call (α, 2026-08-06, upper) | admitted calls/episode | admissible (admitted ≥ 14)? |
|---|---|---|---|---|
| qwen3.5-9b | on | 43.5 s | 1 | no |
| moe-stretch | off | 2.9 s | 20 | yes |
| moe-stretch | on | 17.2 s | 3 | no |

**The rule:**
- An arm with one working mode is measured in it.
- An arm with two working modes is crossed only if BOTH modes are admissible. Otherwise
  it runs in the admissible mode, or in the cheaper mode if neither is.

**Ruling (not crossed):**
- **qwen3.5-9b: `on`.** It is the arm's only extracting mode; at `off` it returns
  `entities: []`. Its one working mode is measured, admissible or not. The table predicts
  that qwen-on fails the throughput floor at screening. That is a result for η to record, not a
  reason to change its mode. α's later warm figure from 2026-09-12 (33.8 s) admits the same
  1 call.
- **phi-4-14b: `off`.** It is single-mode: its chat template has no reasoning channel.
- **moe-stretch: `off`.** Off is admissible; on is not, so the factor is not crossed.

η and θ must not vary mode. Evidence against this ruling goes to escalation, not to a
re-run in a different mode. arms.yaml holds one `reasoning` value per arm and its port
block is full, so a crossed factor would need new arms. That would be a slate-shape
decision for Leo.

## 10. Corpus N

**N stays 200**, δ's committed corpus at corpus sha
`850cf7c937c745b0d224a0ba975efc1c69080ad61d84c5369d06287cdd16f2e8`. PRD Open Q4 is closed
on that basis.

- **Adequacy criterion.** N would grow only if an LLM gated margin exceeded 0.10 absolute.
  Every gated metric is on a [0, 1] scale, and a wider margin would let a candidate 10
  points worse pass. The largest measured margin is graph-sameness's
  0.030365537146117207, so N = 200 resolves every gated metric well inside that bound.
- **Wall-clock does not constrain N.** The controls took 1490.3 s (A) and 1531.8 s (B) at
  concurrency 3 (`run.json` `started_at` → `finished_at`). A θ arm's worst case inside the
  envelope is 200 × 60 s / 3 = 4000 s, about 67 min.
- **Changing N** would change corpus_sha and invalidate both controls and every margin
  here. That would be a new pre-registration, not an amendment.
