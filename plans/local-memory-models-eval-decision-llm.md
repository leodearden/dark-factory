# LME λ decision record — LLM axis (task 3723)

PRD-MARKER:local-memory-models-eval decision-llm

**PRD:** `plans/local-memory-models-eval-prd.md` task λ. **Upstream:** η (task 3720),
θ (task 3721). **Consumer:** μ (task 3725), the born-at-L2 operator gate. **Status:** the
verdict in §3 is PROPOSED. Leo's ruling at μ is the decision. This record changes no
config: `fused-memory/config/config.yaml` and its `llm.*` keys are untouched.

As in `plans/fable-architect-trial-v2-decision-2026-08-30.md`, §1–2 are observation and
§3 is the sole interpretation. §4–7 are context, a named follow-up and the open human
questions. Every number is read from a committed file named beside it, and lines marked
*Hypothesis:* are inferences, not measurements.

## 1. The pre-registered rule, cited by SHA

The rule is `plans/local-memory-models-eval-preregistration.md` at
`865a017d79c6a210da157474e639fd15fe51f9e0`. That is the `preregistration_sha` every
LLM-axis candidate artifact carries
(`plans/local-memory-models-eval-screening/screening-verdict.json`). The file's later
commit `be4a44d624` adds only the esc-3722-3 framing-only erratum under §6:
`git diff 865a017d79 be4a44d624 -- plans/local-memory-models-eval-preregistration.md` is
11 inserted lines, none removed, all inside §6. So §5, §7 and §8 are identical at both SHAs.

- **§5, non-inferiority.** A candidate is non-inferior iff every gated metric is admitted
  by `MarginEntry.admits` AND its full-corpus warm p95 under load is admitted by
  `LatencyEnvelope.admits` (strictly < 60000 ms). Among non-inferior arms, availability
  decides. No quality ranking among non-inferior arms is pre-registered.
- **§8, survival to θ.** An arm reaches θ only if it passes all four absolute screening
  gates: conformance smoke, VRAM fit, context fit and the throughput floor (the same
  60000 ms envelope, on the 20-episode screening run).
- **§7, a clean negative is acceptable.** There is no remediation clause: never widen the
  slate, relax a margin or re-run a candidate for a better draw.

What "admitted" means, from `plans/local-memory-models-eval-controls/preregistration-inputs.json`
(`admits` is the authority for the exact boundary; prereg §3–4):

| Gated quantity | Admits iff |
|---|---|
| conformance-rate | ≥ 0.99894 |
| episode-failure-rate | ≤ 0.005 |
| graph-sameness vs control A | ≥ 0.7755 |
| retrieval-utility | ≥ 0.990 |
| warm p95 under load | < 60000 ms |

## 2. The measured comparison

η screened the whole three-arm slate. Per gate, from `screening-verdict.json` and
`plans/local-memory-models-eval-screening-report.md` §4:

| Gate | qwen3.5-9b | phi-4-14b | moe-stretch |
|---|---|---|---|
| conformance-smoke | PASS | PASS | PASS |
| vram-fit (margin) | PASS (2119 MiB) | PASS (720 MiB) | PASS (2392 MiB) |
| context-fit | PASS | UNMEASURED | PASS |
| throughput-floor | **FAIL**: no p95, 0/6 episodes ok | **FAIL**: p95 101879 ms, margin −41879 ms | **FAIL**: no p95, 0/6 episodes ok |

- `survivors: []`, `outcome: negative-verdict`. The ≤3 cap did not bind: three candidates
  against a cap of three (`cap.binds: false`).
- The throughput floor is each arm's only FAIL. phi-4-14b's context fit is UNMEASURED
  (one of 156 calls reported no prompt length). That alone would also block survival,
  because §8 requires all four gates to PASS.
- θ therefore ran no full-corpus arm (`plans/local-memory-models-eval-llm-report.md` §1–2).
  §5's comparison has no candidate to admit.
- Reported only, never a verdict: phi-4-14b's 20-episode screening values all sit outside
  their bounds (θ report §3: conformance 0.987, failure rate 0.05, retrieval 0.947,
  graph-sameness 0.411).

## 3. Proposed verdict (INTERPRETATION)

**Stay on the incumbent, gpt-4o-mini, for fused-memory's Graphiti write-path LLM.**

No local LLM arm is non-inferior, because none survived screening, so "availability
decides" never engages. Prereg §7 makes this an acceptable result, not a failed eval.

This proposal agrees with a ruling already made. Leo ruled **esc-3720-5, option A** at
sitting item 187, 2026-10-08 15:45Z: accept η's negative verdict and do not revisit
SGLang. Re-read with `get_escalation('esc-3720-5')` on 2026-10-10: status `dismissed`,
resolution action `close_only`, resolved 2026-10-08T15:56:46Z, level 2. θ's report
(§5 and §7) still lists esc-3720-5 as an open decision. That entry predates the ruling and
is superseded by it; this record does not re-surface it.

## 4. Context Leo should weigh (reported, not gating)

- **Measured incumbent cost: $62.38 per 30 days**, a lower bound over
  2026-10-05T11:23Z to 2026-10-08T11:00Z (`plans/local-memory-models-eval-llm/incumbent-cost.json`;
  θ report §4). It replaces the PRD's "$15–25/month" estimate. Volume sets it: at
  September's median of 18 ok writes a day the same unit price ($0.00345 per ok attempt)
  is about $0.06 a day.
- **Availability record** (θ report §5): 2179 `RateLimitError` attempts on 2026-10-03/04,
  17 of them OpenAI credit-exhausted. 499 writes were delayed and none was lost.
- **Client class** (θ report §6): every GenericClient − OpenAIClient delta is within its
  margin. The client's effect on graph-sameness is unmeasured.
- **Host scope.** The throughput failure was measured on the one RTX 3090 that the arms
  share with whisper-writer. It is a result for this host, not a property of the models.

*Hypothesis:* fixing throughput alone would not have produced a non-inferior phi-4-14b,
since its subset quality also misses every margin (θ report §3). n = 20 is too small to
settle that, and the ruling in §3 makes it moot.

## 5. Follow-up PRD this verdict warrants

The named follow-up for this axis is the **production LLM cutover PRD** (PRD
§"Out of scope" and its packaging note). Its scope would be moving the Graphiti
write-path LLM calls to a locally served model. That means adopting `OpenAIGenericClient`
(PRD D3) through β's `llm.client_class` knob, pointing `llm.providers.openai.api_url` at
the local endpoint (`fused-memory/config/config.yaml`), and standing up a permanent
serving unit in place of η's transient eval units.

**The proposed verdict does not warrant it.** It stays unfiled. A future attempt would be
a new eval under a new pre-registration, because prereg §7 forbids widening this slate.
λ files nothing either way (PRD packaging decision, Leo 2026-08-05). If Leo rules
otherwise at μ, the cutover PRD is authored after that ruling.

## 6. Open human decisions (surfaced, not decided)

- Whether the measured cost, with its volume sensitivity, changes the PRD's ranking of
  cost as "minor".
- Whether credit-balance monitoring for the OpenAI account is wanted, given the
  2026-10-03 exhaustion.

SGLang and re-opening the slate are not listed: esc-3720-5 declined option B (§3).

## 7. Premise pins

Each test re-applies the committed rule to the committed evidence. If one fails, this
record is stale.

- `fused-memory/tests/arm_harness/test_lme_decision_premises.py::test_every_llm_arm_fell_at_the_throughput_floor_alone_so_none_reached_the_section_5_comparison`
- `fused-memory/tests/arm_harness/test_lme_screening_artifacts.py::test_the_committed_verdict_is_the_rule_over_the_committed_evidence`
- `fused-memory/tests/arm_harness/test_lme_llm_axis_artifacts.py::test_theta_ran_no_arm_because_screening_left_no_survivor`

No conflation-rate metric is used (PRD D12).
