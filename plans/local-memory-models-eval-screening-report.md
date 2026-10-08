# LME η screening report (task 3720)

PRD-MARKER:local-memory-models-eval screening-report

PRD `plans/local-memory-models-eval-prd.md`, task η. The rule is
`plans/local-memory-models-eval-preregistration.md` §8. Every number below is read from
`plans/local-memory-models-eval-screening/screening-verdict.json` or from the evidence
committed beside it. That directory's `README.md` records how the evidence was taken.

## 1. Verdict

**Survivors: none. Outcome: NEGATIVE VERDICT (`negative-verdict`).**

No arm on the LLM slate passed all four absolute gates. All three were served, passed
the conformance smoke and fit in VRAM. All three failed the throughput floor: a warm
episode-latency p95 under load strictly below 60000 ms. The negative verdict is the
finding, and it is an acceptable one. Ruling 2 (Leo, 2026-08-10) and preregistration §7
make "no local arm is non-inferior" a result this eval can return, not a failure to be
remedied by adding candidates. η adds and widens nothing: no candidate, no re-run, no
mode change.

## 2. The slate

Three LLM arms, read from `scripts/local-model-serving/arms.yaml`: qwen3.5-9b and
phi-4-14b on vLLM, and moe-stretch (gemma-4-26B-A4B QAT) on llama.cpp. The PRD
commissioned four. **Mistral-Small-3.2-24B was dropped by α** (Leo, 2026-08-06,
esc-3713-10) because it cannot be served here. It is a vision-language model; vLLM 0.26
sizes a Pixtral encoder budget at startup, and the quantized repo's tokenizer encodes
the dummy `[IMG]` prompt to zero image tokens against a text count of one, so the engine
never comes up. It was not screened. The contract test ties the screened arms to
`arms.yaml`, so a slate change invalidates this verdict.

## 3. Reasoning mode screened

| Arm | Stack | Reasoning mode screened | Source |
|---|---|---|---|
| qwen3.5-9b | vllm | `on`, with `--reasoning-parser qwen3` | arms.yaml; prereg §9 (its only extracting mode) |
| phi-4-14b | vllm | `off` (single-mode: no reasoning channel) | arms.yaml; prereg §9 |
| moe-stretch | llamacpp | `off` | arms.yaml; prereg §9 (off admissible, on not) |

η did not vary the mode. The verdict records each arm's mode, the contract test requires
it to equal the manifest's, and the evidence loader refuses any α health row taken in
another mode.

## 4. Per gate

Each gate is PASS, FAIL or UNMEASURED, and UNMEASURED is never a pass. Margin is bound
minus value, positive on the passing side.

### Gate 1, conformance smoke: eliminated none

`harness.py smoke --arm-spec` exited 0 for every arm. Each returned a schema-valid
response through the tap (qwen3.5-9b and phi-4-14b in `json_schema`, moe-stretch in
`json_object`), and the validator rejected its negative control each time. This gate
has no numeric margin.

### Gate 2, VRAM fit: eliminated none

The gate is α's `lms_vram.evaluate_budget` verdict, measured beside resident
whisper-writer (4050 MiB, the baseline consumer in every reading). The budget is the
VRAM that was free before each arm started.

| Arm | Footprint (MiB) | Budget (MiB) | Margin (MiB) | Verdict |
|---|---|---|---|---|
| qwen3.5-9b | 14901 | 17020 | 2119 | PASS |
| phi-4-14b | 16301 | 17021 | 720 | PASS |
| moe-stretch | 14629 | 17021 | 2392 | PASS |

phi-4-14b is again the binding row, with 720 MiB to spare.

### Gate 3, context fit: eliminated none; phi-4-14b UNMEASURED

The value is the longest prompt the arm's own server reported (`usage.prompt_tokens`,
recorded per call by the usage tap) over the run. The bound is `max_model_len − 4096`.

| Arm | Longest reported prompt (tokens) | Bound (tokens) | Margin (tokens) | Verdict |
|---|---|---|---|---|
| qwen3.5-9b | 1291 (15 calls) | 32768 − 4096 = 28672 | 27381 | PASS |
| phi-4-14b | 4065 over the 155 reported calls | 16384 − 4096 = 12288 | — | UNMEASURED |
| moe-stretch | 1878 (12 calls) | 16384 − 4096 = 12288 | 10410 | PASS |

Every own-model call requested `max_tokens` 4096, which is the gate's "+4096" premise.
The evidence loader checks it.

phi-4-14b is UNMEASURED, not PASS. One of its 156 calls has no reported prompt length.
That call is a 502 the tap synthesized: "upstream 127.0.0.1:8412 failed:
RemoteDisconnected". It was still being served for an episode the harness had already
abandoned when the sweep's own `lms_ctl stop` stopped the server, 57.4 s after it
started. The README gives the timeline. The rule treats any in-session own-model call
without a reported prompt as making the longest prompt unknown. This record predates the
tap's session marker (end of §7), so it reads as in-session, and the rule is applied
here as written. The artifact does not change phi-4-14b's survival, because gate 4 fails
on its own.

**Answer to PRD Open Q2 (Phi-4 16K context), on this subset:** the longest prompt
phi-4's server reported was 4065 tokens. That is 8223 tokens inside the 12288 tokens
left after reserving 4096 for completion. A 4065-token prompt plus the 4096-token
reservation is 8161 tokens, about half of the 16K window.

Limitation, stated in prereg §8: the longest prompt is measured on the 20-episode subset
only. A longer prompt elsewhere in the corpus can still surface in θ as an episode
failure. The other two arms' longest prompts, 1291 and 1878 tokens, came from runs that
aborted after 6 attempted episodes. They cover less of the subset than phi-4-14b's 20.

### Gate 4, throughput floor: eliminated all three

The value is the run's warm episode-latency p95 under load, at `--concurrency 3`. The
bound is ζ's envelope, 120 s timeout / headroom 2 = 60000 ms, and the comparison is
strict (`LatencyEnvelope.admits`).

| Arm | p95 (ms) | Bound (ms) | Margin (ms) | Verdict | Why |
|---|---|---|---|---|---|
| qwen3.5-9b | none | 60000 | — | FAIL | 0 of 6 attempted episodes ok; fastest attempted 120008 ms |
| phi-4-14b | 101879 | 60000 | −41879 | FAIL | p95 over 19 ok episodes; p50 29050 ms |
| moe-stretch | none | 60000 | — | FAIL | 0 of 6 attempted episodes ok; fastest attempted 120007 ms |

qwen3.5-9b and moe-stretch were stopped by INV-4 after 5 consecutive `TimeoutError`
episodes at the 120 s budget. Each run left a complete `run.json` and `abort.json`, so
these are results, not invalid runs. α's single-sample cold health-probe latency was not
used as throughput evidence anywhere. This gate reads only the screening run's own
records.

## 5. The ≤3 cap

The cap **did not bind**: 3 candidates against a cap of 3 (`cap.binds` false in the
verdict). Every elimination came from an absolute gate, all of them from gate 4. Had
more arms passed all four gates than the cap admits, the evaluator would have refused
rather than ranked, because no ranking rule is pre-registered. That could not happen
with three candidates.

## 6. Per-arm failure modes and reported evidence (reported, NOT gated)

None of the values below gates survival. A test pins that reported values never move
it. Graph-sameness is mean per-episode entity Jaccard against control A's outcomes, over
the episodes ok in both. `top_level_entities_named` is from α's healthcheck on this
sweep, out of 4 probe entities.

| | qwen3.5-9b (on) | phi-4-14b (off) | moe-stretch (off) |
|---|---|---|---|
| Episodes attempted / ok | 6 / 0 | 20 / 19 | 6 / 0 |
| Episode failure rate | 1.0 | 0.05 | 1.0 |
| Error classes | TimeoutError ×6 | TimeoutError ×1 | TimeoutError ×6 |
| INV-4 abort (items) | yes (5) | no | yes (5) |
| Conformance rate (audited attempts) | 0.667 (n 9) | 0.987 (n 155) | 1.0 (n 6) |
| Episode latency p50 | — | 29050 ms | — |
| Tokens per ok episode | — | 11063 | — |
| LLM calls per ok episode, p50 / max | — | 8 / 11 | — |
| Own-model calls through the tap | 15 | 156 | 12 |
| Per-call duration p50 / p95 | 59404 / 128239 ms | 2489 / 11646 ms | 66363 / 174461 ms |
| Calls ending `finish_reason=length` | 9 of 15 | 2 of 156 | 4 of 12 |
| Calls without a reported prompt | 0 | 1 (the tap's 502) | 0 |
| Graph-sameness vs control A | — (no ok episode) | 0.411 (n 19) | — (no ok episode) |
| α health row / `top_level_entities_named` | PASS / 4 | PASS / 2 | PASS / 4 |

Failure modes, in short:

- **qwen3.5-9b (on):** its calls are long. 9 of 15 used the full 4096-token completion
  budget. No episode finished inside 120 s, and the run aborted after 6 episodes.
  Prereg §9 predicted this throughput failure (43.5 s per call admits 1 call per episode
  against a p95 workload of 14).
- **phi-4-14b (off):** it works, but too slowly. 19 of 20 episodes completed, and
  per-call latency was low (p50 2.5 s). The whole-episode p95 was 101879 ms against
  60000 ms. Its graph agreed with control A at 0.411 mean entity Jaccard, and it named
  2 of 4 probe entities as top-level entities.
- **moe-stretch (off):** each call took long (p50 66 s, p95 174 s), with long
  `json_object` completions (748–4096 completion tokens; 4 of 12 used all 4096). No
  episode finished inside 120 s, and the run aborted after 6 episodes.

## 7. Observations for Leo (stated, not acted on)

1. **qwen3.5-9b's pre-registered `on` mode truncates most calls at the 4096-token
   budget.** 9 of its 15 calls ended `finish_reason=length` with 4096 completion tokens.
   The other six used 311–2944. Per-call p50 was 59404 ms. Prereg §9 already rules `on`
   as this arm's only extracting mode and predicts the throughput failure. The truncation
   rate is a further fact about that mode at max_tokens 4096. Per §9 it goes to
   escalation, not to a re-run in another mode.
2. **moe-stretch's throughput looks shaped by its single llama.cpp slot.** The tap's
   per-call durations (p50 66363 ms, p95 174461 ms) are far from α's ~2.9 s
   single-request figure. The run's first three calls arrived at the same instant
   (20:27:37Z) and completed after 29973, 52751 and 84806 ms, a staircase of about 23–32
   s per step.
   *Hypothesis:* lms_serve launches this arm with `--parallel 1`
   (`LLAMACPP_PARALLEL_SLOTS`), so concurrent calls queue behind one slot under
   `--concurrency 3` and graphiti's in-episode fan-out. Long `json_object` completions
   then lengthen each turn of the queue. η measured the arm as configured and did not
   re-configure it.
3. **Both vLLM arms failed the throughput floor.** The PRD's out-of-scope note revisits
   SGLang only if vLLM fails this floor. That condition has now been met (qwen3.5-9b: no
   ok episode; phi-4-14b: p95 101879 ms). Per prereg §8, revisiting SGLang is Leo's call,
   not an automatic step.
4. **On rulings 1, 3 and 4.** All three arms passed conformance and VRAM fit, and every
   measured context gate passed. The common failure is throughput on this host's single
   card, for dense vLLM json_schema and llama.cpp json_object arms alike. η's evidence
   names no capability gap specific to the slate's composition. Whether a throughput
   failure common to all three arms is the "specific, stronger justification" rulings 1,
   3 and 4 contemplate is Leo's judgement. η makes no claim either way.

The phi-4-14b 502 is an instrument defect, not a finding about the arm. The call
started at 20:23:34Z and the run stage ended at 20:24:14Z. The tap's record landed
57.4 s after the call started, at about 20:24:31Z, inside the stop stage (20:24:20Z to
20:24:33Z). So the call outlived its tap session: the tap's handler threads are
daemons, and closing a session did not wait for them. The tap now marks a record that
lands after its session closed as `outlived_session`. The context gate does not count
such a call when it reports no prompt, but it still counts any prompt length one
reports. The committed record predates that marker and loads as an in-session call, so
the committed verdict applies the rule as written and phi-4-14b's gate 3 stays
UNMEASURED.

## 8. Consumer note

θ reads its arm list from `screening-verdict.json` `survivors`. That list is empty, so
θ has no candidate LLM arm to run. Per prereg §8, n = 0 means this report states the
negative verdict and λ records it. Neither is an error state.
