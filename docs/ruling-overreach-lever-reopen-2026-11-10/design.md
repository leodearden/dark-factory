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

**Post-registration correction (2026-11-10, after review).** This paragraph
was added after §1–§2 were committed and after the W1 replay ran. The
registered text of §2.1 and §2.7 above is left as it was. Three things were
wrong with it:

1. The `--model` flag that §2.1 and §2.7 describe recorded a model on each
   replay record, but the arm's own model field was never read.
2. A model replayed into its own file could not be blinded together with the
   control, and `lever_decision` could not judge it against the control,
   because the decision compares arms within one replay.
3. The model is now part of the arm. `ARMS['o6']` is the control's
   instructions on `replay_overreach_extraction.py::O6_MODEL`, and the
   `--model` flag is gone. O6 is measured as `replay --arm control --arm o6`
   into one file (§5).

W1 is unaffected. Control and o1 both ran with no model override, on the
configured gpt-4o-mini: `provenance.json` `runs[0].argv` has no `--model`, and
`r.arms.control.model` and `r.arms.o1.model` are both gpt-4o-mini.

---

*§3 onward were written after adjudication. `r.` abbreviates `report.json`
and `c.` abbreviates `composite-report.json`, both in this directory and both
written by the scripts, not by hand. Run provenance, including the
adjudication record and its blinding residuals, is in `provenance.json`. The
replay output is committed as `replay.jsonl`. §2.2 calls it `replay.json`,
but the file is JSONL, one record per (arm, episode).*

## 3. W1 results: O1 replayed

The replay ran once: 96 (arm, episode) records, no failures, 192 LLM calls,
and 10 previous episodes for every episode (`provenance.json`). 644 minted
edges were labelled: control 292 (`r.arms.control.minted`) and o1 352
(`r.arms.o1.minted`).

### 3.1 Fidelity

| | population-weighted overreach / minted |
|---|---|
| production, 2026-10-01 (`r.fidelity.baseline`) | 0.1238 [0.0876–0.1721] |
| replay control (`r.fidelity.control`) | 0.1355 [0.0977–0.1849] |

The intervals overlap (`r.fidelity.intervals_overlap`). So the replay frame,
with no dedupe against the live graph and endpoints left unresolved,
reproduces production's rate within its noise. The paired comparison below is
O1 against this control, as §2.4 requires.

### 3.2 O1 against the control

| measure | control | o1 | key path (`r.arms.<arm>.`) |
|---|---|---|---|
| overreach / minted, weighted | 0.1355 [0.0977–0.1849] | 0.0795 [0.0534–0.1167] | `population_weighted.overreach_rate_minted` |
| overreach / day | 12.7 | 9.2 | `population_weighted.overreach_per_day` |
| minted / day | 94.0 | 115.9 | `population_weighted.minted_per_day` |
| holding share, weighted | 0.3718 | 0.3766 | `population_weighted.holding_share` |
| holdings / day | 34.9 | 43.6 | `population_weighted.holdings_per_day` |
| episodes with ≥1 overreach | 21/48 | 19/48 | `rates.all.episode_hit_rate` |
| misbound | 7 | 8 | `rates.all.misbound_count` |
| entities / episode | 12.60 | 18.42 | `entities.mean_per_episode` |
| share of control's entity names kept | 1.0 | 0.732 | `entities.mean_share_of_control` |

**The pre-registered verdict: O1 FAILS** (`r.comparison.o1.passes` = false).

| check (§2.4) | bar | measured | holds |
|---|---|---|---|
| overreach per minted, o1 / control | ≤ 0.5 | 0.5867 | no |
| overreach per day, o1 / control | ≤ 0.5 | 0.7235 | no |
| o1's Wilson upper bound below control's point | — | 0.1167 < 0.1355 | yes |
| holdings/day retained | ≥ 0.8 | 1.2488 | yes |

Each ratio is taken over the unrounded weighted edge counts
(`population_weighted.window_counts`). The first score took the per-day
ratios from per-day rates already rounded to 0.1, and read 0.7244 and 1.2493.
The review amendment re-scored the same inputs (`provenance.json`), and no
check changed.

So O1 is a real reduction, but it does not halve overreach, which is the bar
every lever is held to. It does not dilute holdings either: it mints about 23%
more edges per day, and more of them are holdings.

Three descriptive readings, not part of the pre-registered rule:

* **What O1 changed.** By rubric class, from each verdict's rationale:
  control (ii) 32, (iii) 3, (iv) 2; o1 (i) 1, (ii) 24, (iv) 4. Its
  rejected-option clause removed every (iii) case. The garble class (ii),
  meaning inverted relations, invented links and quantities on the wrong
  thing, fell only from 32 to 24.
* **The reduction is concentrated.** One episode, dark_factory `64905a27`,
  carries 10 control overreach edges and 0 o1 ones. The control arm turned
  its list of ten re-landed tasks into chained "re-landed along with" pairs.
  Without that episode, o1 has 29 overreach edges against control's 27,
  unweighted. A rule that halves overreach should not depend on one episode.
* **O1 rewrites entity extraction.** It extracts 46% more entities per
  episode, and keeps only 73% of the control's names. Every extra entity is a
  node that graphiti's `dedupe_nodes` must resolve in production, and that
  the replay does not exercise (§7).

**O1's cost** (`r.arms.<arm>.tokens`, per episode): control
`extract_nodes` + `extract_edges` take 5,177.8 input and 633.6 output tokens.
O1 adds 588.6 input (+11.4%) and 168.7 output (+26.6%) tokens. At gpt-4o-mini
list price that is $0.00019 per episode, about $0.13/day at 670 class
episodes/day. It needs no extra calls. Cost is not why O1 fails.

As §2.4 pre-registered, O1 is not re-tuned and replayed within this task.

## 4. W2 results: the O3+O4 composite

The census (`composite-sample.json` `census`) covers the window
[2026-10-24, 2026-11-10), 17.0 days, under judge prompt
`overreach-judge-v1-ee6466c52d67`:

| graph | handed | retired | kept | superseded | gone |
|---|---|---|---|---|---|
| dark_factory | 751 | 109 (6.41/day) | 616 (36.24/day) | 12 | 14 |
| reify | 340 | 63 (3.71/day) | 263 (15.47/day) | 9 | 5 |

80 flags were sampled (20 per cell) and labelled blind.

| measure (`c.`) | weighted | unweighted |
|---|---|---|
| `retire_precision` | **0.9366** [0.8110–0.9807], effective n 37.3 | 38/40 |
| `flag_precision` | 0.4628 [0.3292–0.6019] | 52/80 |
| `stage1_recall_on_flags` | 0.3312 [0.1811–0.5259] | |
| `holding_retired_share` | 0.0317 [0.0063–0.1445] | |
| `overreach_retired_per_day` | 9.48 | |
| `holdings_retired_per_day` | 0.32 | |
| `composite_recall` | 0.0486, range [0.0106–0.1699] | |

**The pre-registered verdict: the composite PASSES** (`c.decision.passes` =
true). Retire precision 0.9366 clears 0.80, and its Wilson lower bound 0.811
clears 0.65. This re-measures the 2026-11-09 spot-check's unregistered "about
0.90" at a higher n, under a frozen window and a blind worksheet.

What the numbers say:

* **Stage 1 is the precision filter.** The judge's live flags are overreach
  less than half the time: 0.463, inside its offline interval [0.107–0.525]
  (2026-10-01 §10.1).
  Stage 1 retires almost only the true ones: 1 holding and 1 context among 40
  retired flags. By class, the retired overreach is (ii) 30 and (iii) 8, the
  same garble-dominated mix §10.2 of the 2026-10-01 design saw.
* **It retires little of what it is shown.** Stage 1 retires a third of the
  flagged overreach. The rest stays served: about 19 overreach edges/day in
  the two graphs, against 9.48 retired/day. That figure is the kept cells'
  overreach counts (`c.label_counts`) times their weights (`c.weights`), over
  17 days.
* **Its recall is low.** The composite point estimate is 0.049: the judge
  finds about 15% of overreach (offline), and Stage 1 retires about a third of
  that. The range is wide, and it is a product of two frames, because the
  judge's recall was measured on the 2026-08/09 window, before the shadow
  existed. Even its upper end, 0.17, is far from the 0.5 "at least halved"
  target in the 2026-10-01 §6.1 revisit clause.
* **Its cost in holdings is small.** 0.32 holdings retired per day, against
  9.48 overreach.

So the composite is safe and cheap, and it is already running. It does not
come near halving served overreach.

## 5. W3: O6, a stronger extraction model

**Why it belongs in the table.** Every lever in the 2026-10-01 §6 table acts
on the extraction model's output: O1 instructs it, O3 judges it, O4 retires
from it. None changes the model. The measured overreach is mostly garble
class (ii): control 32 of 37 here, and 30 of 38 retired overreach edges in
the composite. These are extraction errors: an inverted direction, a symbol
placed in the wrong module, a quantity put on the wrong thing. O1's
instructions name exactly these errors (rules 4 and 5), yet class (ii) fell
only from 32 to 24 (§3.2). The extraction model's competence, not its
instructions, is the remaining unmeasured axis. **O6** routes the class's two
extraction calls, `extract_nodes` and `extract_edges`, to a stronger model.
The other graphiti calls stay on gpt-4o-mini.

**Named candidate.** gpt-6.1-sol, the stronger model priced in
`fused-memory/scripts/score_write_triage_pairs.py::LIST_PRICES` (as of
`LIST_PRICES_AS_OF` 2026-09-30, source `LIST_PRICES_SOURCE`
https://developers.openai.com/api/docs/pricing).

**Cost**, from the control arm's measured tokens per episode (5,177.8 input,
633.6 output; `r.arms.control.tokens`), at list price with no cached-input
discount, at about 670 class episodes/day (2026-10-01 §10.2):

| model | input / output, USD per Mtok | per episode | per day | per 30 days |
|---|---|---|---|---|
| gpt-4o-mini (today) | 0.15 / 0.60 | $0.00116 | $0.78 | $23 |
| gpt-6.1-sol | 2.00 / 10.00 | $0.01669 | $11.18 | $335 |

* That is 14.4× the same two calls on gpt-4o-mini, about $10.41/day more.
* Against graphiti's whole write it moves 2 of the about 10 calls per class
  episode (2026-10-01 §6, an estimate from call sites). The other calls'
  tokens were not measured here, so no whole-write ratio is claimed.
* The figure assumes gpt-6.1-sol emits about the same tokens as gpt-4o-mini.
  Reasoning tokens are billed as output (`LIST_PRICES` docstring), so the
  real figure may be higher. The O6 replay measures it.

**How to measure it.** One run on the same 48 episodes. It needs no new
code, because O6 is an arm: `ARMS['o6']`, the control's instructions on
`replay_overreach_extraction.py::O6_MODEL` (§2.7's correction).

1. `replay_overreach_extraction.py replay --arm control --arm o6 --out <new file>`.
   The control is a fresh gpt-4o-mini replay, about $0.06 at list price from
   the per-episode cost above. The o6 replay is about $0.80.
2. `worksheet`, which interleaves both arms' edges blind.
3. Adjudicate every edge in ONE sitting under the 2026-10-01 §2.6 rubric.
4. `score`. Its `comparison.o6` is the §2.4 decision against the control
   from that same run and sitting.

**O6 measured, 2026-11-12, task 9065.** §7's temperature residual came true.
gpt-6.1-sol refuses temperature with HTTP 400, and with temperature omitted
it runs at the API's default effort. Under ruling esc-9065-2 the single arm
`o6` therefore became two declared arms, `o6-low` and `o6-medium`. Each
sends no temperature and an explicit `reasoning.effort`, and each was scored
against the same-run control under §6.2's unchanged checks.

* **Cost.** Measured tokens replace this section's estimate:
  * o6-low: 5,337.8 input and 1,681.8 output tokens per episode,
    $0.02749/episode, $18.42/day;
  * o6-medium: 2,586.5 output tokens per episode, $0.03654/episode,
    $24.48/day;
  * that is 23.1× and 30.7× the two calls on gpt-4o-mini, and 1.65× and 2.19×
    the $0.01669/episode estimated above.
* **Decision.** Both arms pass all three §6.2 checks with zero adjudicated
  overreach (0 of 905 and 0 of 1,057 minted, against control's 48 of 328),
  so the R→R+O6 trigger fires at effort low. One self-blinded adjudicator
  labelled every edge, and the arms' styles leaked which edges shared an
  arm, so that perfect separation cannot rule out adjudicator expectation.
  The new record's §8 names a second rater on an o6 sample as the cheap
  check.
* **Routing must inherit the client.** Any routing follow-up must inherit
  the omit-temperature and explicit-effort handling of
  `fused-memory/scripts/replay_overreach_extraction.py::ExplicitEffortOpenAIClient`.
  graphiti_core 0.28.2 sends temperature to, and withholds reasoning from,
  any model it does not recognise as a reasoning model.

The measurement, its residuals and the routing task are in
`docs/ruling-overreach-o6-replay-2026-11-12/`.

## 6. W4: the lever

### 6.1 The table, updated

| option | extra cost | measured effect on overreach | holdings cost | status |
|---|---|---|---|---|
| **O1** `custom_extraction_instructions` | 0 calls; +11% input and +27% output tokens on the two extraction calls, about $0.13/day | ×0.59 per minted, ×0.72 per day (§3.2), concentrated in one episode | none (holdings/day ×1.25), but entity extraction rewritten (73% name overlap) | **measured, FAILS** §2.4 |
| **O3+O4** shadow judge plus Stage-1 retirement | 1 judge call per class episode (running since 2026-10-24) plus Stage 1's existing budget | retires 9.48 overreach/day in df+reify at 0.937 precision. Composite recall about 0.05 | 0.32 holdings/day | **measured, PASSES** §2.6 |
| **O3 enforce** the judge's own marker | as O3 | judge precision 0.267 offline (2026-10-01 §10.1) and 0.463 live (§4) | would hide at least one non-overreach edge for every overreach edge it hides | **withdrawn**: neither precision is near 0.80 |
| **O6** stronger extraction model for the class | about +$10.41/day at list (§5) | unmeasured | unmeasured | **costed**, measurable as arm `o6`, no new code |
| O2 typed edges, O5 authoring lint | as 2026-10-01 §6 | 0 by construction | — | unchanged, not levers |

### 6.2 Recommendation, staged, with trigger and revisit clauses

**The O3+O4 composite is the lever. Ratify it as it runs today, and stage the
next measurement on O6.** It is the only option that passed its pre-registered
bar, and it is already deployed. O1 failed its bar. The judge's own marker
cannot pass at the precision measured on either frame.

* **Stage R (ratified, now).**
  * The judge keeps recording every class write to the sidecar, after the
    lock, as 2026-10-01 §6.1 Stage S specifies.
  * Its overreach flags keep going to recon Stage 1 as the O4 list, and
    Stage 1 keeps retiring through `update_edge(invalid_at=now)`.
  * The judge writes nothing to the graph. The 2026-10-01 §6.1 trigger from
    shadow to enforce is **withdrawn**: no marker is written and no read site
    filters on one.
  * Sidecar retention may proceed (§9).
* **Trigger clause, from R to R+O6.** It fires when the O6 measurement's
  `comparison.o6.passes` is true. The run is
  `replay_overreach_extraction.py replay --arm control --arm o6` on the same
  48 episodes, into one file. One `worksheet` interleaves both arms blind,
  every edge is adjudicated in one sitting under the 2026-10-01 §2.6 rubric,
  and then `score` runs (§5). `comparison.o6.passes` is §2.4's three checks,
  computed by `lever_decision` with `OVERREACH_RATIO_MAX` and
  `HOLDING_RETENTION_MIN`. The comparison is against the control replayed in
  the same run and labelled in the same blind sitting, and all three must
  hold:
  1. population-weighted overreach ≤ 0.5 × the control's, per minted AND per
     day;
  2. its Wilson upper bound per minted < the control's point estimate;
  3. holdings/day ≥ 0.8 × the control's.

  The clause does not use fixed numbers from this report's control, because a
  cross-sitting comparison would give up the pairing and blinding that §2.2
  and §2.4 rely on. That run's `arms.control` is reported beside this
  `r.arms.control` (0.1355 per minted, 12.7 overreach/day, 34.9
  holdings/day), as a stability reading only.

  If all three hold, file one task to route the class's `extract_nodes` and
  `extract_edges` calls to gpt-6.1-sol. That task is `always_escalates`: the
  measured cost goes to Leo for a spend ruling before anything is routed. If
  any check fails, O6 is closed as O1 is, and the composite stays the lever
  alone.
* **Revisit clause, on the composite.** Re-measure with
  `score_overreach_composite.py sample` + `score` on a fresh frozen window:
  at least 14 days, dark_factory + reify, cap 20, blind under §2.6, its window
  and cells pre-registered before the sample is drawn. Run it 90 days after
  this ruling (2027-02-08), or sooner if the judge's prompt version changes
  (the census `prompt_versions`) or recon's Stage-1 handling of the O4 list
  changes. **Stop Stage 1 retiring handed flags** (the judge keeps recording)
  and re-open the lever choice if either holds:
  * `retire_precision` < 0.80, or its Wilson lower bound < 0.65. These are
    the same bars, so a retirement keeps hiding at most one true fact per four
    overreach edges.
  * `holdings_retired_per_day` > 1.0 in the two graphs. That is about 3× the
    0.32 measured here, the volume signature of a Stage 1 that has started
    retiring holdings while its share stays under the bar.

None of the measured levers reaches the 2026-10-01 target, served overreach at
least halved. The composite trims it, and O1 dents it. That target stays open,
and O6 is the remaining measured path toward it. Two other paths were not
measured here: a higher-recall judge, and a less conservative Stage 1 (it
retires about a third of the flagged overreach it is handed,
`c.stage1_recall_on_flags` 0.33). Each would need its own pre-registered
measurement before it could be a lever.

## 7. Accepted residuals

* **Single adjudicator, self-blinded.** There is still no inter-rater
  agreement. Two blinding leaks are on record (`provenance.json`
  `adjudication.blinding_residuals`). Before labelling, the adjudicator saw
  per-arm totals, and the first control record. That record holds episode
  `64905a27`'s control edges, the same episode that carries O1's measured
  reduction (§3.2). Its ten "re-landed along with" edges were labelled by the
  same rule as every other invented link. Still, the episode the result
  depends on is the one whose arm membership was seen. Any bias from that
  could only have inflated the control's count, and so flattered O1, which
  fails anyway.
* **Replay vs production dedupe.** The replay resolves nothing against the
  live graph. The control arm absorbs this for the paired comparison, and
  §3.1 shows the frame reproduces production's rate. But O1's extra entities
  would meet production's `dedupe_nodes`, which the replay never ran.
* **The previous-episode set as read today.** Previous episodes were re-read
  on 2026-11-10, not as they stood at write time. An episode deleted since
  then, or written later with an earlier `valid_at`, changes the context.
  Both arms share it.
* **LLM run-to-run variance.** Each arm ran once, at temperature 0.0. One
  replay cannot separate O1's effect from sampling variance within that 0.59
  ratio. A failed result is not re-run, as pre-registered.
* **The composite's retired cell.** A flag is "retired" when its edge has
  `invalid_at` set and `expired_at` null. That is Stage 1's `update_edge`
  shape, but it is not proof that Stage 1 acted because of the flag. Any
  `update_edge(invalid_at=…)` of a flagged edge, for whatever reason, lands in
  that cell.
* **Two graphs.** The composite was measured on dark_factory and reify only.
  The judge runs on 10 graphs, about 670 class episodes/day.
* **Two frames in one product.** Composite recall multiplies an offline judge
  recall (2026-08/09 window) by a live Stage-1 recall (2026-10/11 window). It
  is an order of magnitude, not an estimator.
* **O6's cost uses gpt-4o-mini's token counts.** See §5.
* **O6 is sent temperature 0.0.** graphiti_core 0.28.2's
  `llm_client/openai_client.py` treats only model names starting with
  `gpt-5`, `o1` or `o3` as reasoning models, and drops `temperature` only for
  those. The default `llm.client_class: openai` builds that client, so
  gpt-6.1-sol is sent temperature 0.0 (`cfg.llm.temperature or 0.0`). This
  was read from source, not run. If the API refuses it, the o6 replay records
  failure lines and exits 1, and the measuring task escalates rather than
  patching graphiti.

## 8. Build leaves filed

Filed through `submit_task` with `spawned_from: 9040`, one per area. The
curator turned ticket `tkt_0RWTVN4F9A2APEFBN010V3RTVG` into task 9065 (B1)
and ticket `tkt_0RWTVNXGVV14Q5XZ58CCKFH24X` into task 9066 (B2). Both tasks
depend on 9040. The architect wired that on 2026-11-10, because both scripts
exist only on this branch until it merges. This file is authoritative over
each task's copy of §6.2.

* **B1, task 9065, the O6 replay measurement** (area: the replay instrument,
  no code change expected). It runs §6.2's trigger:
  `replay --arm control --arm o6` into one file, one blind worksheet over both
  arms adjudicated in one sitting, then `score`. The decision is that
  report's `comparison.o6`. On a pass it files the `always_escalates` routing
  task. Priority medium. Task 9065's description first pinned the dead
  `--model` run against fixed `r.arms.control` numbers. On 2026-11-10 it was
  replaced in full through `update_task` with the arm run above, a verbatim
  copy of §6.2's corrected trigger clause, and §7's temperature residual.
  The read-back still lists 9040 as a dependency.
* **B2, task 9066, the composite revisit gate** (area: the composite
  instrument). It runs §6.2's revisit clause on its schedule: a
  pre-registered fresh composite window, blind adjudication, score, and the
  two pull-back conditions. Priority low, with a `dated` milestone at
  2027-02-08T00:00:00+00:00.

## 9. Sidecar retention (task 7487)

The 2026-10-01 §10.3 decision held every sidecar row until this task ruled.
That hold is released:

* **The composite evidence is frozen in this directory.**
  `composite-sample.json` carries the census, and every sampled flag's graph,
  edge, episode, fact, decision and `handed_at`. With `composite-verdicts.json`
  and `composite-report.json` it reproduces §4 without the sidecar. The O1
  evidence never lived in the sidecar.
* **What retention must keep.** The §6.2 revisit clause reads
  `OverreachShadowStore.handed_overreach_flags`. That joins `edge_judgements`
  (label `overreach`), `flag_handoffs` and `judged_episodes` (for
  `prompt_version`) within a handoff window. So for the handed overreach flags
  of the most recent window the revisit will draw, at least 30 days, those
  three tables' rows must survive. Everything older, and every non-overreach
  judgement once its handoff state no longer matters to Stage 1, may be pruned
  under 7487's own design.
