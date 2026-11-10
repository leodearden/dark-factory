# Re-opening the ruling-overreach lever: O1 replayed, the O3+O4 composite scored, a stronger extraction model costed

**Task 9040** · gate task 6218 (`esc-6218-1`) · instruments
`fused-memory/scripts/replay_overreach_extraction.py` and
`fused-memory/scripts/score_overreach_composite.py` · `graphiti_core` 0.28.2

**§1 and §2 are PRE-REGISTERED.** They were committed on their own, before any
replay ran, before the composite sample was drawn and before any edge was
labelled. So neither the O1 instruction text, nor the decision rules, nor the
composite window could be tuned toward the results. §3 onward are written after
adjudication. Every measured number there is cited from `report.json` or
`composite-report.json` in this directory by key path. Run provenance is in
`provenance.json`.

## 1. Question

Gate task 6218 (esc-6218-1) evaluated the §6.1 trigger of
`docs/ruling-overreach-guard-design-2026-10-01/design.md` on 2026-11-09 (that
document's §10). The offline leg failed: judge precision 0.267 against a bar of
0.80, recall 0.147 against 0.50. The judge stays in shadow, and §6.1's own
failure branch applies: "re-open the lever choice, with O1 measured by replay
of the same 48 episodes".

Two §10.2 observations widen the question. The O4 handoff does retire flagged
edges. And a 20-edge spot-check found about 0.90 of what Stage 1 retires to be
overreach, almost all of it extraction-side garbles. Leo's §10.3 decision made
that into this task's four work items:

* **W1, O1 replay.** Does `custom_extraction_instructions` for the class halve
  overreach on the 48 frozen episodes? What does it cost in holdings and in
  tokens?
* **W2, the composite.** Score O3+O4 as a lever in its own right. How precise
  is Stage 1's retirement of the judge's flags? How much flagged overreach does
  it retire, and how many holdings go with it?
* **W3, O6.** The garbles are extraction-side, and the 2026-10-01 §6 table has
  no extraction-model lever. Does a stronger extraction model belong among the
  levers, and what would it cost at about 670 class episodes/day (§10.2)?
* **W4, one lever.** Recommend one, staged, with trigger and revisit clauses in
  the §6.1 shape.

Sidecar retention (task 7487) waits on this ruling. The frozen sample of §2.5
is what releases it.

## 2. Method

### 2.1 Replay frame (W1)

* **Episodes.** The 48 episodes of the 2026-10-01 sample, as
  `audit_ruling_overreach.py::SampleDefinition()` fixes them: window
  [2026-08-25, 2026-10-01), three strata, cap 8 per graph per stratum,
  `sha256(graph:uuid)`. The input is the worksheet that
  `audit_ruling_overreach.py --emit-worksheet` writes. `replay` refuses a
  worksheet whose (graph, stratum) episode counts differ from the 4716
  `report.json` `sample.sizes`.
* **Inputs, read-only.** Each episode's `valid_at`, `source` and
  `source_description` are read over GRAPH.RO_QUERY through
  `ReadOnlyGraphReader`; the body comes from the worksheet. Its previous
  episodes follow graphiti_core 0.28.2's `retrieve_episodes` rule, as
  `Graphiti.add_episode` calls it: the last `RELEVANT_SCHEMA_LIMIT` (10)
  episodes of the same group and source whose `valid_at` is at or before the
  episode's, excluding the episode itself. The bound is
  `falkor_string_bounds.py::exact_string_upper_bound`, because FalkorDB's range
  index over-returns on a bare string `<=`
  (`replay_overreach_extraction.py::PREVIOUS_EPISODES_CYPHER`).
* **Extraction.** In process: graphiti_core's own `extract_nodes`, then
  `extract_edges`, unmodified. They run through a `GraphitiClients` that holds
  only the LLM client, so no FalkorDriver ever exists. As fused-memory calls
  them, `entity_types` is None and the edge type map is graphiti's default
  `('Entity', 'Entity')`. Then graphiti's in-batch exact-fact dedupe: the
  driver-free fast path of `resolve_extracted_edges`, keyed by source, target
  and `_normalize_string_exact(fact)`.
* **Stated divergence from production.** Production's minted set is the
  extracted set minus whatever `dedupe_nodes` and `resolve_extracted_edges`
  merge into the live graph. The replay resolves nothing against a graph. So
  its edges name endpoints as extracted, not as resolved, and an edge that
  production would have folded into an existing fact is minted here. Both arms
  share this divergence, and the control arm absorbs it (§2.4).
* **Model.** `llm.model` gpt-4o-mini at temperature 0.0
  (`cfg.llm.temperature or 0.0`; `config.yaml` sets none), built by
  `graphiti_client.py::build_llm_client`, the production seam.
* **Arms.** `control` splices nothing. `o1` splices `O1_INSTRUCTIONS`, version
  `o1-v1-99b813d73cba`. graphiti places the text in all three node
  prompts and in the edge prompt. Verbatim:

  ```text
  This is a DECISION RECORD. Extract only what it states, at the scope it states it.
  1. Keep the record's scope and qualifiers. Never generalise a decision past the cases, components, actors or conditions the record names, and never drop a qualifier (who, which, when, under what condition) that limits it.
  2. Add no implication or consequence the record does not state. Do not link two things with a vague verb such as "impacts", "affects", "relates to" or "implies" unless the record itself links them.
  3. An option the record weighed, rejected, deferred or only proposed is never adopted. State it only together with its status (for example "X was rejected in favour of Y"), never as current fact.
  4. Preserve direction exactly: who or what acts on what, what calls, imports or depends on what, what is above or below, before or after.
  5. Place each symbol, file, function, value or measurement with the entity the record places it with. Never move it to a neighbouring module, component or task.
  6. PREVIOUS_MESSAGES only help resolve references. Never attribute their content to this record.
  ```

  The model is a separate axis (`--model`). Replaying the same arms on a
  stronger model is a run, not new code.

### 2.2 Adjudication

* **Rubric.** The 2026-10-01 design.md §2.6 rubric, verbatim and unchanged:
  the labels, their precedence, the worked examples and the authored-overreach
  blind spot.
* **Blinding.** `replay_overreach_extraction.py worksheet` writes one row per
  episode: its body, and every arm's edges interleaved by sha256(edge_uuid),
  with no arm named. Edge uuids are uuid4, so they reveal no arm. Near-duplicate
  facts from the two arms sit side by side, which keeps labels consistent across
  arms.
* **Single adjudicator, self-blinded.** The agent that ran the replay labels
  it, and `replay.json` is not opened during labelling. This is a residual, not
  a guarantee.
* **Verdict file.** `verdicts.json` in this directory, in `load_verdicts`
  format, with the 4716 `SampleDefinition` as its sample block. `score`
  validates it once over the union of arms, and refuses a missing, duplicate,
  unknown-label or out-of-sample verdict.

### 2.3 Measures (W1)

Per arm, from `replay_overreach_extraction.py score`:

* **Overreach**, per minted edge and per day, population-weighted with the 4716
  cell weights: each (graph, stratum) cell's `strata.<g>.<s>.window_population`
  over its 8 sampled episodes, across `sample.window_days` (37.0), through
  `audit_ruling_overreach.py::population_weighted_rates`.
* **Holding loss.** The weighted holding share and weighted holdings/day, at
  the same weights (`weighted_rate`). O1 rewrites entity extraction for the
  whole class, so holdings/day is the price it can charge.
* **Entity extraction change.** Mean entities per episode, and the mean
  per-episode share of the control's entity names (graphiti's exact
  normalisation) that the arm also extracts.
* **Tokens**, measured per call by `llm_token_usage.py::measure_llm_tokens`
  and totalled per prompt. O1's prompt-token delta against the control is its
  cost.
* **Fidelity.** The control's weighted overreach beside the production 12.4%
  [8.8–17.2%] (the 4716 report's `a.population_weighted.overreach_rate_minted`),
  and whether the two intervals overlap. This checks the replay frame. It is
  not O1's comparator.
* Unweighted label counts per stratum, for reference.

### 2.4 The O1 decision rule

O1 is judged against the control, never against 12.4%. The control holds fixed
everything the replay changes: LLM run-to-run variance, no dedupe against the
live graph, the previous-episode set as read today, and endpoints before
resolution. So the paired difference is O1's. O1 **passes** only if all three
hold (`replay_overreach_extraction.py::lever_decision`):

1. Its population-weighted overreach is at most `OVERREACH_RATIO_MAX` = 0.5 of
   the control's, **both** per minted edge **and** per day. 0.5 is the
   2026-10-01 §6.1 revisit clause's "at least halved", so every lever is held
   to one target. The per-day figure is required too, so an O1 that only
   dilutes the rate by minting more benign edges cannot pass.
2. Its per-minted Wilson upper bound is below the control's point estimate.
3. Its weighted holdings/day are at least `HOLDING_RETENTION_MIN` = 0.8 of the
   control's.

Each arm is replayed once. If O1 fails, it is not re-tuned and replayed within
this task.

### 2.5 Composite sample (W2)

* **Population.** Every overreach flag the shadow handed to recon, in
  `dark_factory` and `reify`, read from `flag_handoffs` through
  `OverreachShadowStore.handed_overreach_flags`. The frozen window is
  **[2026-10-24T00:00:00+00:00, 2026-11-10T00:00:00+00:00)**. It runs from the
  UTC day of the first handoff in both graphs (dark_factory
  2026-10-24T20:04Z, reify 2026-10-24T18:56Z, read read-only) to the UTC
  midnight before this commit. `flag_handoffs` holds both the flags shown to
  Stage 1 and the flags dropped as already gone or retired.
* **Decision**, read from the live edge over GRAPH.RO_QUERY at sampling time
  (`score_overreach_composite.py::decision_of`):
  * **retired**: `invalid_at` set and `expired_at` null. That is the shape of
    Stage 1's `update_edge(invalid_at=now)` (`reconciliation/overreach_flags.py`).
  * **superseded**: both set, which is graphiti's own contradiction handling.
  * **kept**: `invalid_at` null, so the edge is still served.
  * **gone**: the edge no longer exists.
* **Cells**: graph × {retired, kept}. Superseded and gone flags are counted in
  the census and never sampled.
* **Sample.** The first 20 per cell by `sha256(graph:edge_uuid)`
  (`audit_ruling_overreach.py::sample_key`), frozen with the census into
  `composite-sample.json`. That file carries no episode body and no judge
  rationale. Edge state keeps moving, so the frozen file is the evidence.
* **Adjudication.** §2.2's rubric and single adjudicator, from a worksheet with
  each episode's body and its sampled flags' facts and endpoints, in hash
  order. It carries neither the decision nor the judge's rationale. Verdicts go
  to `composite-verdicts.json`, with the composite `SampleDefinition` as their
  sample block.

### 2.6 Composite measures and decision rule

Each cell's flags stand for its census population over its sampled flags
(`score_overreach_composite.py::WeightedCells`). Every ratio and Wilson bound
comes from `audit_ruling_overreach.py::weighted_rate`.

* **Retire precision**: weighted overreach among retired flags. It re-measures
  the 2026-11-09 spot-check's 0.90, this time pre-registered.
* **Flag precision**: weighted overreach over all four cells, the judge's live
  precision on handed flags.
* **Stage-1 recall on flags**: weighted retired overreach over weighted flagged
  overreach.
* **Exposure**: holdings retired/day and overreach retired/day over the window.
* **Composite recall** = offline judge recall (0.147 [0.058–0.323],
  `judge-eval-offline-2026-11-09.json` `recall`) × Stage-1 recall on flags: a
  point and a product-of-bounds range. It is labelled as a product of two
  frames, because the 48-episode window predates the shadow.

The composite **passes** if and only if its retire precision is at least
`PRECISION_MIN` = 0.80, with a Wilson lower bound of at least
`PRECISION_LOWER_MIN` = 0.65. These are §6.1's precision bars, imported from
`eval_overreach_judge.py`. Precision is the bar because a wrong retirement
hides a true fact. Recall is reported but not barred: what Stage 1 keeps is the
status quo.

### 2.7 What O6 is costed from (W3)

O6, a stronger extraction model, is costed here and not adjudicated:

* **Tokens**: the control arm's measured `extract_nodes` and `extract_edges`
  input and output tokens per episode (§2.3).
* **Volume**: about 670 class episodes/day (2026-10-01 §10.2).
* **Prices**: `fused-memory/scripts/score_write_triage_pairs.py::LIST_PRICES`,
  as of 2026-09-30 (`LIST_PRICES_SOURCE`), in USD per Mtok input/output, at
  list price with no cached-input discount: gpt-4o-mini 0.15/0.60, and the
  named candidate gpt-6.1-sol 2.00/10.00.
* **Relative cost**: against the same two calls on gpt-4o-mini, and against the
  about 10 calls per episode graphiti already makes (2026-10-01 §6). Only the
  two extraction calls would move to the stronger model.

Measuring O6 later is `replay --model <name>` into its own file, then a blind
adjudication of the same 48 episodes. It needs no new code.
