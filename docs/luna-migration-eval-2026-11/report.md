# Report: gpt-4o-mini vs gpt-6-luna, per call site

**Task 9945** (esc-9522-1, T4). The bars, tie-breaks and decision rule are
pre-registered in `design.md` §2, and the reading of these results is in
`design.md` §3 onward. Every number below comes from `report.json` in this
directory, abbreviated `r.`; the exceptions are computed from it, and the
arithmetic is shown. `report.json` was written by
`luna_migration_eval.py score` on 2026-11-22 and re-scored the same day after
one summary rating was corrected (§1, last bullet). Its sha256 is
`a1d616c50b91b3462728410f6921ba8ae8dd8cfbd953e2c23ec8c24aaee8d68b`; the first
score run is in `provenance.json` `runs[step=score]`.

The candidate arm is `luna-low`: gpt-6-luna at reasoning effort low, on the
plan proxy. The control is gpt-4o-mini at today's settings, on
`OPENAI_API_KEY`. Each site judges luna-low against the control on the
fixtures both arms ran. No arm left a failure line on any site.

## 1. Per-site verdicts

| site | check | luna-low against the control | bar (`design.md` §2.3) | check | site |
|---|---|---|---|---|---|
| extraction | E1 | upper bound 0.0653 (`r.extraction.comparison.luna-low.statistics.candidate_overreach_upper`), against the control's point 0.1405 (`…control_overreach_rate`) | ≤ 0.1405 + 0.05 = 0.1905 | pass | **PASS** (`r.extraction.comparison.luna-low.passes`) |
| | E2 | holding retention 1.2261 (`…statistics.holding_retention`) | ≥ 0.8 | pass | |
| resolve_edge | R1 | merge accuracy +0.035 [−0.0117, 0.0817] (`r.sites.resolve_edge.decision.statistics.merge_accuracy`) | lower ≥ −0.05 | pass | **FAIL** (`r.sites.resolve_edge.decision.passes`) |
| | R2 | no-false-merge −0.01 [−0.0466, 0.0266] (`…statistics.no_false_merge`) | lower ≥ −0.03 | **fail** | |
| summaries | S1 | faithful rate +0.125 [0.0309, 0.2191] (`r.sites.summaries.decision.statistics.faithful_rate`) | lower ≥ −0.10 | pass | **PASS** (`r.sites.summaries.decision.passes`) |
| classifier | C1 | label accuracy +0.0591 [−0.0189, 0.1371] (`r.sites.classifier.decision.statistics.label_accuracy`) | lower ≥ −0.05 | pass | **PASS** (`r.sites.classifier.decision.passes`) |
| router | Q1 | label accuracy −0.2667 [−0.3322, −0.2011] (`r.sites.router.decision.statistics.label_accuracy`) | lower ≥ −0.05 | **fail** | **FAIL** (`r.sites.router.decision.passes`) |

* **The failure gate holds on every site**: 0 luna failure lines
  (`r.sites.<site>.failures.candidate_failure_share` 0.0, against a maximum
  of 0.02). The extraction replay also has 0 failure lines, so `score`
  reported a verdict rather than refusing with exit 2.
* **Paired n**: resolve_edge 200, summaries 80 items over 45 fixtures,
  classifier 203, router 300 (`r.sites.<site>.decision.statistics.<metric>.n`).
  Extraction covers 54 episodes per arm (`r.extraction.arms.<arm>.episodes`).
* **One summary rating was corrected after scoring** (esc-9945-3). luna's
  answer for reify episode ea2a4400…, node 661ececa…, omitted that node, so
  graphiti kept its stored summary and the run recorded the echo: 2,248
  chars, byte-identical to the input summary, past the 500-char cap graphiti
  applies to every summary it writes. It had been rated faithful. Under
  `design.md` §2.4 it is `dropped`, and S1 moved from +0.1375 [0.0411, 0.2339]
  (b = 14) to +0.125 [0.0309, 0.2191] (b = 13). The verdict is unchanged.
  It is the only arm summary in `runs.jsonl` that matches its input's
  `summary_sha256`. `luna_eval_sites.py::run_summaries` now records None for
  an omitted node, so a re-run shows it to the rater as dropped.

## 2. The routing layer's joint verdict

**FAIL.** `r.routing_layer.checks` has `C1_classifier_label_accuracy` true
and `Q1_router_label_accuracy` false, so `r.routing_layer.passes` is false.
Under `design.md` §2.4 one pass and one fail keep both sites on gpt-4o-mini.
The classifier's own pass does not flip it, because the classifier and the
router share the one `routing.model` leaf.

## 3. Cost, per site and arm

The prices are `score_write_triage_pairs.py::LIST_PRICES` as of
`r.cost.prices.as_of` 2026-09-30, in USD per Mtok input / output:
gpt-4o-mini 0.15 / 0.60, gpt-6-luna 0.10 / 0.50. **These figures are for
comparison only.** luna runs on the plan and is not billed per token, and no
passing site keeps an API fallback. So luna's USD column is an API
equivalent, not a spend. Tokens per call are `r.cost.sites.<site>.<arm>`.
Output tokens include reasoning tokens.

| site | arm | input tok / call | output tok / call | list USD / call | calls / day | list USD / day |
|---|---|---|---|---|---|---|
| extraction | control | 2,019.2 | 217.9 | $0.000434 | 2.571 (M) | $0.00111 |
| | luna-low | 2,337.0 | 371.6 | $0.000419 | 2.571 (M) | $0.00108 |
| resolve_edge | control | 938.0 | 16.1 | $0.000150 | 5.452 (I) | $0.00082 |
| | luna-low | 1,232.2 | 22.8 | $0.000135 | 5.452 (I) | $0.00073 |
| summaries | control | 7,218.0 | 132.0 | $0.001162 | 1.286 (I) | $0.00149 |
| | luna-low | 7,512.0 | 148.3 | $0.000825 | 1.286 (I) | $0.00106 |
| classifier | control | 231.3 | 73.6 | $0.0000789 | 713.571 (I) | $0.0563 |
| | luna-low | 526.3 | 74.6 | $0.0000899 | 713.571 (I) | $0.0642 |
| router | control | 200.9 | 32.1 | $0.0000494 | 2,114.571 (M) | $0.1044 |
| | luna-low | 494.2 | 91.4 | $0.0000951 | 2,114.571 (M) | $0.2011 |

**Calls/day.** Each figure was measured read-only from the write journal over
[2026-11-15, 2026-11-22) and is tagged M (measured) or I (inferred). The
queries and derivations are in `provenance.json` `calls_per_day`.

* **extraction (M).** The journal holds 9 non-decision graphiti `add_episode`
  rows in the 7 days, 1.286/day, and the control replay made 2.0 calls per
  episode (108 over 54).
* **resolve_edge (I).** 1.286 non-decision episodes/day × 4.24 minted edges per
  episode (the control replay: 229 over 54) × 1 resolve call per minted edge.
  The last factor is an inference: it assumes every minted edge has a related
  or hybrid-search candidate and misses the exact-fact fast path.
* **summaries (I), an upper bound.** 1.286 episodes/day × 1 batched call per
  episode, as the control runs measured. An episode none of whose entities
  exceeds the LLM threshold makes no call.
* **classifier (I), an upper bound.** 4,995 of the 8,141 `add_memory` rows
  have a journalled prefix that the production classifier sends to its LLM.
  The journal records the resolved category, so caller-categorised writes,
  which skip the classifier, cannot be excluded.
* **router (M).** 14,802 of the 16,824 `search` rows have a query that the
  production router sends to its LLM. No row carried a stores override. The
  190 truncated queries were probed on their journalled prefix.

The graphiti sites are counted over non-decision episodes only, the class
the eval measures and any graphiti flip would move. Over all categories, the
journal holds 6,495 graphiti `add_episode` rows in the window (927.9/day),
6,486 of them `decisions_and_rationale`.

## 4. Plan-window share (inferred)

* **All five sites.** luna's API-equivalent across all five sites is
  $1.877/week (`r.cost.plan_window.weekly_candidate_usd`). That is 0.67% of a
  $280/week window and 0.40% of a $470/week window
  (`r.cost.plan_window.share`, tagged `inferred`).
* **The two sites that pass and would move.** Extraction and summaries
  together come to 7 × ($0.001078 + $0.001061) = $0.0150/week, which is
  0.0053% to 0.0032% of the window. The classifier does not move (§2), so
  it is not counted.
* **The window itself is inferred, not measured.** It is the $280–470/week
  range that `design.md` §2.7 pre-registered.

## 5. Residuals

Each is read in full in `design.md` §7.

* **One self-blinded adjudicator** for every site, with no inter-rater
  agreement. The conventions fixed while blind are in `provenance.json`
  `adjudication.<site>.conventions`. Two of them move a bar's numerator:
  * the extraction status-record holding reading (E2);
  * the summaries supersession reading (S1). All five of the ratings that
    rule decided fell on the control. Without the rule, S1 is still
    +0.0625 [−0.0176, 0.1426], a pass (`design.md` §4.3).
* **The summary letters follow a public rule**, sha256(fixture_id:arm), so
  the summaries' blindness rested on the rater not recomputing it. `score`
  refuses a key file that differs from the rule, and the committed key
  matches it.
* **R2 is underpowered at this n.** Only 14 fixtures split the arms on
  no-false-merge. At n = 200 with b + c = 14, even a zero difference gives a
  lower bound of −0.0367, below the −0.03 bar. So R2 could pass only with
  b − c ≥ 2; luna's was −2.
* **The router's FAIL rests mostly on the gold label `broad`.** 88 of the
  101 control-only pairs are queries the adjudicator labelled broad and luna
  gave a specific type, entity_lookup in 51 of them. That reading was fixed
  blind, but by one rater.
* **Routing parse failures are served as the production fallback.** Each
  is attributed to an arm by the request line just before it in the run log.
  * Router: 6 luna answers did not parse. 5 exhausted
    `max_completion_tokens` 256, and 1 was empty. 4 control answers named a
    store that is not a `SourceStore` ('mem1', 'error_pages'). All 10 were
    served as broad. Counting luna's 6 as luna-correct still gives
    −0.2267 [−0.2941, −0.1592], a FAIL.
  * `runs.jsonl`'s router lines carry `"fallback": null`. The runner that
    wrote them could not observe a router fallback, so that null does not
    mean the route was an LLM answer. The runner now omits the field
    (`luna_eval_sites.py::run_router`).
  * Classifier: 14 luna answers and 1 control answer sent
    `"secondary": "null"`. The production parser rejects that string, so
    each was served as `observations_and_summaries`, the `llm_error`
    fallback.
  * After scoring, `routing/classifier.py` was changed to read a secondary
    that names no category as none, keeping the parsed primary. The eval
    measured the parser as it stood at the pre-registration commit, and
    nothing is re-scored. The routing layer's FAIL rests on Q1, and the
    router's parser is untouched.
* **Proxy overhead.** On every site, luna's input exceeds the control's by
  293 to 318 tokens per call. That matches the ~300-token plan-proxy overhead
  `design.md` §2.9 stated, so luna's tokens per call overstate the model's own.
* **Graph state, the classifier's 200-character inputs and the approximate
  resolve candidates** are as stated in `design.md` §2.9.
* **luna's extraction mints 1.91× the control's edges** (437 against 229,
  `r.extraction.arms.<arm>.minted`). So a luna extraction would raise
  per-episode resolve_edge calls above the figure in §3.

## 6. Decision

The pre-registered rule is `design.md` §2.5, under Leo's 2026-11-19
amendment to esc-9522-1: there is no API fallback.

### 6.1 The graphiti sites: extraction, resolve_edge, summaries

* **resolve_edge FAILS**, on R2 (§1). It stays on gpt-4o-mini on
  `OPENAI_API_KEY`, as today. Nothing is filed for it.
* **extraction and summaries PASS. Neither flip ships in this task.** Task
  9944 owns the plan-cap classifier, latch, park and L2 escalation, and it
  was pending when this report was written. So
  `fused-memory/src/fused_memory/backends/graphiti_client.py` and
  `config.yaml`'s `llm` block are unchanged.
* **One flip follow-up was filed**, ticket `tkt_0RX9K2525CQR6J5QPN3EH7WM4B`,
  on 2026-11-22. It has `dependencies='9944'` and `spawned_from` 9945, and
  the curator assigns the task id asynchronously.
  * It moves only the measured prompts, for every category except
    `decisions_and_rationale`, to gpt-6-luna at effort low on
    `OPENAI_PLAN_PROXY_URL`/`OPENAI_PLAN_PROXY_KEY`. The extraction prompts
    are `extract_nodes.extract_text` and `extract_edges.edge`; the summaries
    prompt is `extract_nodes.extract_summaries_batch`.
  * It keeps exactly the measured configuration of `design.md` §2.2, and
    keeps no gpt-4o-mini cap fallback.
  * It routes per `prompt_name`, so `dedupe_nodes.nodes`,
    `dedupe_edges.resolve_edge` and every unmeasured prompt stay on
    gpt-4o-mini.
  * It reuses task 9943's per-prompt plan route, or reports a blocking
    dependency on 9943 if that has not landed.

### 6.2 The routing layer: classifier and router, jointly

* **The routing layer FAILS** (`r.routing_layer.passes` false). The
  classifier passes C1, but the router fails Q1 at −0.2667 [−0.3322,
  −0.2011], so both sites stay on gpt-4o-mini on `OPENAI_API_KEY`, as today.
* **No config changes.** `fused-memory/config/config.yaml`'s `routing` block
  keeps `model`, `reasoning_effort`, `openai_api_url` and `openai_api_key`
  null, which inherits `llm.providers.openai`'s gpt-4o-mini endpoint
  (`routing/openai_client.py::routing_chat_params`, `routing_endpoint`).
  Nothing is flipped, so no operational restart follow-up is filed.
* **What the user sees is unchanged.** After the next fused-memory restart,
  the 'Routing LLM: ...' line still names gpt-4o-mini on
  `https://api.openai.com/v1`.
