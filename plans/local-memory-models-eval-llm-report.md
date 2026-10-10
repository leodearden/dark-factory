# LME θ LLM-axis report (task 3721)

PRD-MARKER:local-memory-models-eval llm-report

PRD `plans/local-memory-models-eval-prd.md`, task θ. The decision rule is
`plans/local-memory-models-eval-preregistration.md` §5, and §7 governs a negative. Every
number below is read from a committed file named beside it. θ's own evidence is in
`plans/local-memory-models-eval-llm/`, and that directory's `README.md` records how it
was taken. Lines marked *Hypothesis:* are inferences, not measurements.

## 1. Verdict

**No local LLM arm is non-inferior. The incumbent, gpt-4o-mini, stands.**

θ ran no full-corpus arm. `plans/local-memory-models-eval-screening/screening-verdict.json`
has `survivors: []` and `outcome: negative-verdict`. Prereg §5 admits a candidate only on
its full-corpus metrics, and an empty candidate set admits nothing. So "availability
decides" never engages. Prereg §7 rules this an acceptable result: θ did not widen the
slate, re-run an eliminated arm, or relax a margin.
`fused-memory/tests/arm_harness/test_lme_llm_axis_artifacts.py` pins this premise. If η's
verdict ever gains a survivor, that test fails and this report is stale.

## 2. What θ ran, under which pins

- **Arm runs: zero.** "Every arm artifact embeds preregistration_sha + code_sha" holds
  vacuously, because θ produced no arm artifact.
- **The pins a θ run would have used**, from `screening-verdict.json`: preregistration
  `865a017d79c6a210da157474e639fd15fe51f9e0`, code
  `055e9c15a0d756704017b0d77ffbe4185155402f`, corpus
  `850cf7c937c745b0d224a0ba975efc1c69080ad61d84c5369d06287cdd16f2e8`.
- **The run shape it would have used**, from prereg §5: concurrency 3, `with-indices`,
  temperature 0.0, max_tokens 4096, GenericClient, graphiti semaphore limit 20,
  graph-sameness against control A.
- **VRAM.** η measured every arm against the free VRAM before it started: a budget of
  17020–17021 MiB (about 16.6 GiB), with whisper-writer resident at 4050 MiB
  (`plans/local-memory-models-eval-screening-report.md` §4, gate 2). That is
  consistent with the PRD's measured ~16.4 GiB (D10). θ took no VRAM reading of its own.
- **SEMAPHORE_LIMIT.** No LME run set it. All seven committed `run.json` files, the four
  ζ controls under `plans/local-memory-models-eval-controls/runs/` and the three η runs
  under `plans/local-memory-models-eval-screening/runs/`, record
  `graphiti_semaphore_limit` 20, graphiti's module default. `screen_slate.py`'s unit sets
  only `OPENAI_API_KEY`. fused-memory's queue knob `semaphore_limit:
  ${SEMAPHORE_LIMIT:3}` (`fused-memory/config/config.yaml`) was never overridden. Had a θ
  run needed a non-default graphiti limit, it would have been set only in that transient
  unit's environment (`systemd-run --setenv`), never in `config.yaml` or a shell
  profile, because the env name sets both knobs at once.

## 3. Per-metric results against the pre-registered margins

The margins, references and envelope come from
`plans/local-memory-models-eval-controls/preregistration-inputs.json`. "Admits" is
`MarginEntry.admits`: a lower-is-worse value admits at or above reference − margin, and a
higher-is-worse value admits at or below reference + margin. The envelope admits a warm
p95 under load strictly below its bound (`LatencyEnvelope.admits`).

| Gated quantity | Direction | Reference | Margin (source) | Admits iff | Full-corpus candidate |
|---|---|---|---|---|---|
| conformance-rate | lower is worse | 0.99972 | 0.00078 (run pair) | ≥ 0.99894 | no survivor |
| episode-failure-rate | higher is worse | 0.0 | 0.005 (floor) | ≤ 0.005 | no survivor |
| graph-sameness | lower is worse | 0.8058 | 0.0304 (episode SE) | ≥ 0.7755 | no survivor |
| retrieval-utility | lower is worse | 0.995 | 0.005 (floor) | ≥ 0.990 | no survivor |
| warm p95 under load | latency envelope | incumbent p95 42828 ms | timeout 120 s / headroom 2 | < 60000 ms | no survivor |

### Reported, not gated: the only candidate quality evidence

phi-4-14b is the one arm with ok episodes. Its values below come from η's 20-episode
screening run, `plans/local-memory-models-eval-screening/runs/phi-4-14b/20261007T201954Z/metrics/`,
and its graph-sameness from `screening-verdict.json`
`reported.graph_sameness_mean_entity_jaccard`. Beside them is
`plans/local-memory-models-eval-controls/runs/incumbent-generic-20/20261007T020600Z/metrics/`,
the incumbent on the same 20 episodes with the same GenericClient.

| Quantity | Admits iff | phi-4-14b (η subset) | incumbent-generic-20 (same episodes) |
|---|---|---|---|
| conformance-rate | ≥ 0.99894 | 0.987 (n 155) | 1.0 (n 165) |
| episode-failure-rate | ≤ 0.005 | 0.05 (n 20) | 0.0 (n 20) |
| retrieval-utility | ≥ 0.990 | 0.947 (n 19) | 1.0 (n 20) |
| graph-sameness vs control A | ≥ 0.7755 | 0.411 (n 19) | not measured |
| warm p95 under load | < 60000 ms | 101879 ms (n 19) | 24132 ms (n 20) |

Every phi-4-14b value lies outside its boundary. These are not a verdict:
- the run covered a 20-episode subset;
- the floors were derived for N = 200;
- the arm was eliminated on throughput before any full-corpus run;
- reading committed η records is not a re-run.

qwen3.5-9b and moe-stretch had no ok episode, so they have no quality values.

*Hypothesis:* on this evidence, fixing throughput alone (esc-3720-5, the pending SGLang
question) would not have produced a non-inferior phi-4-14b, because its subset quality
also misses every margin. The subset is too small to settle that.

## 4. Cost quantification

Every figure is from `plans/local-memory-models-eval-llm/incumbent-cost.json`, priced at
control A's gpt-4o-mini rates, $0.15 / $0.60 per 1M input / output tokens. Production
runs the same model.

**Production, measured.** The window runs from 2026-10-05T11:23:27.597318Z, the first
token-bearing graphiti attempt, to 2026-10-08T11:00:00Z: 2.98 days.

| Quantity | Value |
|---|---|
| Graphiti LLM attempts | 1850, of which 53 failed |
| Input / output tokens | 34,648,989 / 1,678,303 |
| Tokens per attempt | 19,636 |
| LLM calls | 18,104, which is 9.8 per attempt |
| Spend | $6.204 |
| Spend per ok attempt | $0.00345 |
| Spend per day | $2.079 |
| Projected spend per 30 days | **$62.38** |

By project:

| Project | Attempts | Spend |
|---|---|---|
| solar_challenge | 499 | $1.822 |
| dark_factory | 551 | $1.804 |
| solar_challenge_platform | 433 | $1.413 |
| reify | 361 | $1.155 |
| scoping_intake | 6 | $0.011 |

By day: 412 attempts and $1.415 on 2026-10-05 (from 11:23Z); 402 and $1.350 on 10-06;
842 and $2.818 on 10-07; 194 and $0.622 on 10-08 (to 11:00Z).

**Replay unit cost.** ζ's control runs replayed the 200-episode corpus. Control A cost
15,688 tokens and $0.002736 per episode; control B cost 15,836 tokens and $0.002765.

**Against the PRD's estimate.** The PRD's "$15–25/month" was an estimate. It came from
August call volume (≈300–900 calls/day) and was made when no token telemetry existed. The
measured $62.38 per 30 days replaces it. That is 2.5× the estimate's top and 4.2× its
bottom. It is a **lower bound**: on the OpenAI-shaped clients a failed or re-prompted
attempt records no tokens (OPERATIONS.md §"Per-write telemetry"). It also excludes
embedding spend.

**Volume context**, from `plans/local-memory-models-eval-llm/graphiti-write-history.jsonl`:
- **September:** 1 to 47 ok graphiti writes a day, median 18.
- **October 1–4 burst:** 1869, 4021, 6352 and 4862 attempts a day, most of them failures.
- **The measured window:** 402 to 842 attempts on each full day.

Volume, not unit price, sets the monthly figure. At September's median of 18 ok writes
a day, $0.00345 per ok attempt comes to about $0.06 a day (18 × $0.00345). At the
window's rate it is $2.08 a day.
*Hypothesis:* a three-day window just after an October backlog burst is not
representative of steady state. The 30-day projection is that window's rate, not a
forecast.

**Local arms.** Their metered spend is $0 by definition: phi-4-14b records usd-per-episode
0.0 at 11,063 tokens per ok episode on the subset. Electricity and hardware are not
priced here, and this report invents no figure for them.

## 5. Availability and independence

The incumbent's measured outage modes come from `graphiti-write-history.jsonl` and from
§3 of the evidence `README.md`.

- **Rate limiting, 2026-10-03/04:** 2179 graphiti attempts failed with `RateLimitError`
  (134 on 10-03, 2045 on 10-04).
  - 2162 of them read `Rate limit exceeded. Please try again later.`
  - 17 (15 + 2) read OpenAI's `429 … You have no credits remaining …
    insufficient_quota / credit_balance_exhausted`.
- **What became of them:** 499 distinct writes hit at least one RateLimitError. All 499
  later succeeded and are `terminal_status` `completed`. The durable queue's one dead
  letter is an unrelated 2026-09-07 `add_episode` NodeNotFoundError. No write was lost.
  The cost was delay and retry volume.
- **Which client raised the 429:** the error text does not say, so the raising client is
  unknown.
  *Hypothesis:* the raw 429 text came from the OpenAI embedder. In the installed
  graphiti_core, it is the one OpenAI path that does not re-wrap the error into
  graphiti's fixed "Rate limit exceeded" message. Some of the 2162 fixed-message
  attempts may have been the same quota exhaustion on the LLM path.
- **Cloud timeouts:** 12 `APITimeoutError: Request timed out.` attempts in October.
- **Local failures, not an LLM dependency:** in October, 8530 `ResponseError: Query timed
  out`, 252 `BusyLoadingError: Redis is loading the dataset in memory` and 50
  `ConnectionError … localhost:6379` attempts. These are the local FalkorDB, which a
  local LLM would not remove.

**The independence gain is real, but this host cannot obtain it at the measured
throughput.** All three local arms failed the 60000 ms throughput floor on the one 3090
they share with whisper-writer (`screening-verdict.json`, gate throughput-floor). The
incumbent's measured cloud failures were rate limits and an exhausted credit balance, and
they delayed writes rather than losing them. Whether to re-open the slate or revisit
SGLang is Leo's call (esc-3720-5), not θ's. The embedder is also an OpenAI dependency
(`fused-memory/config/config.yaml` `embedder.model: "text-embedding-3-small"`), which is
ι's axis. Full independence from OpenAI needs both axes.

## 6. Client-class delta context

The parity deltas are GenericClient − OpenAIClient, over the same 20 episodes:
`plans/local-memory-models-eval-controls/runs/incumbent-generic-20/20261007T020600Z/parity/incumbent-openai-20/metrics/`.

| Quantity | Delta | Margin |
|---|---|---|
| conformance-rate | 0.0 | 0.00078 |
| episode-failure-rate | 0.0 | 0.005 |
| retrieval-utility | 0.0 | 0.005 |
| episode-latency-p50 | −1806 ms | not gated |
| episode-latency-p95 | −625 ms | not gated |
| tokens-per-episode | −133 | not gated |
| usd-per-episode | −2.67e-5 | not gated |

No gated delta exceeds its margin, so the client choice alone would not have moved a
verdict. The pair measured no graph-sameness delta. That is a gap: the client's effect on
graph agreement is unmeasured.

## 7. Consumer note for λ

λ records the following for the LLM axis:
- **The negative verdict.** No local LLM arm is non-inferior, because no arm survived
  screening (§1).
- **The incumbent's measured cost.** At least $2.08/day and $62.38 per 30 days over
  2026-10-05 to 10-08, at $0.00345 per ok attempt. This replaces the PRD's estimate of
  $15–25/month (§4).
- **The availability record.** About 2.2k rate-limited attempts and 17
  credit-exhausted ones on 2026-10-03/04, with no write lost (§5).

Open human decisions for λ to surface, not decide:
- whether to revisit SGLang or re-open the slate (esc-3720-5);
- whether the measured cost, with its volume sensitivity, changes the PRD's ranking of
  cost as "minor";
- whether credit-balance monitoring for the OpenAI account is wanted, given the
  2026-10-03 exhaustion.
