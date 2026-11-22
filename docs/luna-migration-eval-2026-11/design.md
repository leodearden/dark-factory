# gpt-4o-mini vs gpt-6-luna, per call site, against a pre-registered bar

**Task 9945** (esc-9522-1, T4) · amendment (Leo, 2026-11-19): no API fallback ·
ruling `esc-9945-2` (Leo, 2026-11-22): API credit restored, embeddings not a
gate · instruments `fused-memory/scripts/luna_migration_eval.py`,
`luna_eval_frame.py`, `luna_eval_sites.py`, `luna_eval_score.py` and
`replay_overreach_extraction.py`, as of this document's pre-registration
commit · `graphiti_core` 0.28.2 · `openai` 2.31.0

The extraction leg reuses the instrument and rubric of
`docs/ruling-overreach-o6-replay-2026-11-12/design.md`, cited below as "O6",
and of `docs/ruling-overreach-guard-design-2026-10-01/design.md`, cited as
"2026-10-01".

**§1 and §2 are PRE-REGISTERED.** They were committed on their own, with the
frame and its fixtures, after the instrument's code and before any arm ran.
So neither the frame, nor the arms, nor the bars, nor the decision rule, nor
the cost method could be tuned toward the results. §3 onward are written after
adjudication and scoring. Every measured number there is cited from
`report.json` in this directory by key path. Run provenance is in
`provenance.json`.

## 1. Question

Can each fused-memory call site that runs on gpt-4o-mini today move to
gpt-6-luna, at reasoning effort low on the plan proxy, without losing quality?
Each site is judged on its own: the same frozen fixtures run under both arms,
a blind adjudication follows, and a non-inferiority bar is applied. Every bar
is fixed here, before any arm call.

The five sites, each measured per prompt:

| site | production call | function the eval runs, unmodified |
|---|---|---|
| `extraction` | graphiti's `extract_nodes` and `extract_edges`, for non-decision `add_memory` episodes | `replay_overreach_extraction.py`'s replay of `graphiti_core` `node_operations.py::extract_nodes` and `edge_operations.py::extract_edges` |
| `resolve_edge` | graphiti's `dedupe_edges.resolve_edge` prompt | `graphiti_core` `edge_operations.py::resolve_extracted_edge` |
| `summaries` | graphiti's `extract_nodes.extract_summaries_batch` prompt | `graphiti_core` `node_operations.py::extract_attributes_from_nodes`, with `entity_types=None` and `luna_eval_sites.py::NullEmbedder` |
| `classifier` | `add_memory` auto-classification | `fused_memory/routing/classifier.py::WriteClassifier.classify` |
| `router` | search routing | `fused_memory/routing/router.py::ReadRouter.route` |

**Out of scope: node dedupe (`dedupe_nodes.nodes`).** graphiti builds one
client for every prompt (`graphiti_client.py::_construct_llm_client` sets
`model = small_model = llm.model`). A whole-client flip would therefore move
node dedupe, and every other unmeasured prompt, untested. Any graphiti flip
this eval licenses routes per `prompt_name`. Everything not named above stays
on gpt-4o-mini.

**The classifier and the router share one leaf.** Both build their call from
`routing/openai_client.py::routing_chat_params` and `routing.model`. They are
therefore judged jointly, as one routing layer, and flip together or not at
all.

## 2. Method

### 2.1 Frame, as drawn

Drawn once, read-only, on 2026-11-22 between 12:23:57Z and 12:27:17Z, from
`fused-memory/` with `EVAL=../docs/luna-migration-eval-2026-11`. The argv is
exactly the one in `luna_migration_eval.py`'s usage docstring:

    uv run python scripts/luna_migration_eval.py frame --dir $EVAL \
        --graph dark_factory --graph reify --graph know_live --graph market_predictor \
        --graph autopilot_video --graph pump_web_ui --graph scoping_intake \
        --graph solar_challenge --graph solar_challenge_platform --graph stream_touch \
        --extraction-window 2026-08-01T00:00:00Z 2026-11-18T00:00:00Z --cap 3 \
        --fixture-window 2026-10-01T00:00:00Z 2026-11-18T00:00:00Z \
        --resolve-n 200 --summaries-n 80 --nodes-per-episode 2 \
        --journal /home/leo/src/dark-factory/data/reconciliation/write_journal.db \
        --journal-window 2026-11-04T00:00:00Z 2026-11-18T00:00:00Z \
        --classifier-per-category 50 --router-n 300 \
        --worksheet /tmp/9945-worksheet.jsonl --summary-inputs /tmp/9945-summary-inputs.jsonl

* **Reads.** The graphs are read over GRAPH.RO_QUERY only, through
  `fused_memory/backends/falkor_read_only.py::ReadOnlyGraphReader`. The write
  journal is opened with sqlite `mode=ro`. Nothing was written to either.
* **Code.** The frame was drawn with `luna_eval_frame.py` and
  `luna_migration_eval.py` at commit `054fb34dbf` on `task/9945`. Neither file
  changed before this commit.
* **Two earlier draws refused and wrote nothing.** Both exited 1, each on a
  different `dark_factory` summary item: the live graph rewrote an entity
  summary between the hashed read and a second read of its text.
  `054fb34dbf` builds the summary inputs from the read the manifest hashes, so
  this third draw selected and hashed the items once. That fix, and the
  routing-failure fix `40d32a178a` (§2.8), are recorded as step-23 deviations.

**Committed here** (ids, hashes and fixture text, but no episode body):

| file | sha256 | rows |
|---|---|---|
| `frame.json` | `269daaa10d5c6e8a3b5b90fea2ad839323e02458ed6d5564805ff8c98fd4dfd5` | — |
| `fixtures-resolve-edge.jsonl` | `856809b8fd1e814f66b10912249ff58a490df13d4891da6d25b81e74cccfd34d` | 200 |
| `fixtures-summaries.json` | `944c885412d3bf44491458617d139afe09c6178fa78a001bc5cce5d4e23ac3c4` | 80 items |
| `fixtures-classifier.jsonl` | `6b23e95a37c23c68555ec1993d243f87b5b0bfafab9fe04b507b50e6159e7d0f` | 203 |
| `fixtures-router.jsonl` | `dd403eaf3576880765e7f54ab1727dece8275abdf079d4f3d1fd24107544e5bd` | 300 |

**Kept private, outside this directory,** because they carry episode bodies:

| file | sha256 | rows |
|---|---|---|
| `/tmp/9945-worksheet.jsonl` | `106a67086428341396ddd02fcf5b97bc1b05095565d3cb00a8655907d7767816` | 54 episodes |
| `/tmp/9945-summary-inputs.jsonl` | `9176413a1018a870cb3734befacf36bb434b44121664f5214c1dd773a9e22879` | 45 episodes |

`replay` refuses a worksheet whose (graph, stratum) counts differ from
`frame.json` `sample.sizes`. `run --site summaries` and `worksheet --kind
summaries` refuse summary inputs that `luna_eval_frame.py::check_summary_inputs`
cannot match to `fixtures-summaries.json`.

**Extraction frame.** These are the non-decision `add_memory` episodes created
in [2026-08-01, 2026-11-18), 109 days. They are stratified by the written
category: `temporal_facts`, `entities_and_relations`, and `other_categories`
for every other non-decision category. `decisions_and_rationale` is excluded
because levers O1 and O6 measured it. The sample takes up to `--cap` 3
episodes per (graph, stratum) by sha256(graph:uuid), the 4716 audit's own hash
rule. A cell holding fewer than 3 episodes is taken whole.

Each cell is shown as drawn / window population:

| graph | temporal_facts | entities_and_relations | other_categories |
|---|---|---|---|
| dark_factory | 3 / 135 | 3 / 26 | 3 / 3 |
| reify | 3 / 65 | 3 / 11 | 2 / 2 |
| solar_challenge | 3 / 29 | 3 / 5 | 0 / 0 |
| solar_challenge_platform | 3 / 16 | 3 / 11 | 1 / 1 |
| scoping_intake | 3 / 12 | 2 / 2 | 2 / 2 |
| know_live | 3 / 9 | 1 / 1 | 0 / 0 |
| autopilot_video | 3 / 18 | 0 / 0 | 1 / 1 |
| stream_touch | 3 / 4 | 0 / 0 | 3 / 6 |
| pump_web_ui | 1 / 1 | 1 / 1 | 1 / 1 |
| market_predictor | 0 / 0 | 0 / 0 | 0 / 0 |

That is 54 episodes, of a nominal 90 (10 graphs × 3 strata × cap 3). The
shortfall of 36 is empty or small cells, chiefly `market_predictor`, which has
no in-window episode. Rates are population-weighted at the frame's cell
weights (O6 §2.1's estimators, unchanged).

**resolve_edge: 200 fixtures, as requested.** These are RELATES_TO edges
created in [2026-10-01, 2026-11-18), in sha256(graph:edge_uuid) order.

* Each fixture's `related` holds the other edges between the same node pair,
  as `EntityEdge.get_between_nodes` selects them, created before the
  candidate, newest first, at most 10.
* Its `invalidation_candidates` are other edges incident to either endpoint,
  created before the candidate, at most 10.
* An edge with no sibling is skipped. So is an edge whose exactly normalised
  fact equals a sibling's, because that is graphiti's no-LLM fast path.

| solar_challenge | dark_factory | solar_challenge_platform | scoping_intake | reify | stream_touch | pump_web_ui |
|---|---|---|---|---|---|---|
| 93 | 28 | 24 | 22 | 21 | 9 | 3 |

Related-set sizes: 117 fixtures have 1, 34 have 2, and 29 hit the cap of 10.

**summaries: 80 items over 45 episodes, as requested.** An item is an Entity
that an episode in [2026-10-01, 2026-11-18) mentions, and whose stored summary
plus that episode's minted incident facts exceeds
`4 * graphiti_core.utils.text_utils.MAX_SUMMARY_CHARS`. That is the only case
in which graphiti summarises by LLM. There are at most 2 items per episode, by
sha256(graph:node_uuid). The measured text is 2,024 to 59,744 characters
(median 4,766). By graph: solar_challenge 32, dark_factory 17,
solar_challenge_platform 10, stream_touch 8, scoping_intake 8, reify 5. Each
item's previous summary is the graph's stored summary at draw time.

**classifier: 203 fixtures, against a requested 6 × 50 = 300.** The fixtures
are `add_memory` journal rows in [2026-11-04, 2026-11-18), in sha256(row id)
order. Each is kept only when the production `WriteClassifier`, at the
`RoutingConfig()` defaults, sends it to its LLM;
`luna_eval_frame.py::LlmReachProbe` answers locally to test that. There are at
most 50 per journalled category.

| category | fixtures |
|---|---|
| decisions_and_rationale | 50 |
| observations_and_summaries | 50 |
| preferences_and_norms | 50 |
| procedural_knowledge | 50 |
| temporal_facts | 2 |
| entities_and_relations | 1 |

The shortfall is 97 fixtures: only 2 `temporal_facts` inputs and 1
`entities_and_relations` input in the window reached the LLM. The journalled
category is carried as `reference_category` and never scored. Gold comes from
adjudication (§2.6).

**router: 300 fixtures, as requested.** These are distinct, untruncated
(`query_truncated` false) `search` journal queries in [2026-11-04, 2026-11-18),
in sha256(row id) order. Each is kept only when the production `ReadRouter`
sends it to its LLM.

The live `routing` block in `fused-memory/config/config.yaml`
(`confidence_threshold` 0.7, `use_heuristics` true, `llm_fallback` true)
equals the `RoutingConfig()` defaults the probe used. So every routing fixture
reaches the LLM when it runs.

### 2.2 Arms

There are exactly two arms, `luna_eval_sites.py::SITE_ARMS`, both drawn from
`replay_overreach_extraction.py::ARMS`:

| arm | model | sampling control sent | endpoint |
|---|---|---|---|
| `control` | gpt-4o-mini: config `llm.model`; `routing.model` null keeps it for the routing sites | temperature 0.0. graphiti sends `cfg.llm.temperature or 0.0`; `routing_chat_params` sends `temperature` 0.0 and `max_tokens` 256 | configured: `llm.providers.openai`, `OPENAI_API_KEY` on `https://api.openai.com/v1`. Null routing leaves inherit it |
| `luna-low` | gpt-6-luna (`LUNA_MODEL`) | reasoning effort low, no temperature | the plan proxy: `OPENAI_PLAN_PROXY_URL` and `OPENAI_PLAN_PROXY_KEY` |

* **The graphiti sites** (extraction, resolve_edge, summaries) reach luna
  through `replay_overreach_extraction.py::ExplicitEffortOpenAIClient`. It
  wraps the client `graphiti_client.py::build_llm_client` builds, with the
  openai provider's URL and key replaced by the plan proxy's
  (`build_arm_llm`). It sends `responses.parse` with
  `reasoning={'effort': 'low'}`, no temperature and no verbosity, and
  `max_output_tokens` as graphiti passes it.
* **The routing sites** reach luna through exactly the leaves a flip would
  ship (`luna_eval_sites.py::routing_arm`): `routing.model` gpt-6-luna,
  `routing.reasoning_effort` low, and `routing.openai_api_url`/`openai_api_key`
  from the two proxy variables. `routing_chat_params` then sends
  chat.completions with `reasoning_effort` low and `max_completion_tokens`
  256, and no temperature.
  `routing/openai_client.py::routing_endpoint` resolves the endpoint, the same
  predicate production uses.
* **Instructions.** Both arms take the control's instructions:
  `NO_INSTRUCTIONS` for extraction, and the production prompts everywhere
  else.
* **Client checks.** `check_arm_client` refuses any client not on its arm's
  model, effort and endpoint, before any call. Each record names its model,
  `reasoning_effort` and endpoint.
* **One arm per candidate.** No other effort is run, and a FAIL is not retried
  at medium.

### 2.3 Metrics and bars

`luna_eval_score.py` is the code home of every bar. All bars are inclusive.
Every site judges `luna-low` against the control on the fixtures both arms ran.
The routing and graph sites use the paired non-inferiority statistic
`paired_noninferiority`:

* the difference is d = (b − c)/n, where b counts the pairs only luna got
  right and c the pairs only the control got right;
* se = √((b + c) − (b − c)²/n)/n;
* the interval is Wald 95%, d ± z·se, with z = `WILSON_Z`;
* a site passes when the lower bound ≥ −margin.

| site | check | metric | bar |
|---|---|---|---|
| extraction | E1 | luna's population-weighted overreach per minted edge, upper 95% bound | ≤ the control's point + `EXTRACTION_OVERREACH_MARGIN` (0.05) |
| extraction | E2 | luna's holdings/day ÷ the control's | ≥ `HOLDING_RETENTION_MIN` (0.8) |
| resolve_edge | R1 | paired merge accuracy: the arm merges exactly when gold holds a duplicate, and into one of gold's duplicates | lower bound ≥ −`RESOLVE_ACCURACY_MARGIN` (0.05) |
| resolve_edge | R2 | paired no-false-merge: the arm merges nothing, or merges into a gold duplicate | lower bound ≥ −`RESOLVE_FALSE_MERGE_MARGIN` (0.03) |
| summaries | S1 | paired faithful rate per (fixture, node) item | lower bound ≥ −`SUMMARY_FIDELITY_MARGIN` (0.10) |
| classifier | C1 | paired label accuracy | lower bound ≥ −`ROUTING_LABEL_MARGIN` (0.05) |
| router | Q1 | paired query-type accuracy | lower bound ≥ −`ROUTING_LABEL_MARGIN` (0.05) |
| every site except extraction | failure gate | luna failure lines left after one resume, as a share of the site's fixtures | ≤ `LUNA_FAILURE_SHARE_MAX` (0.02) |

* **Extraction** is scored by `replay_overreach_extraction.py::score_replay`
  over `frame.json`, with `luna_eval_score.py::extraction_noninferiority` as
  its `decide`. E1 compares the reported rounded rate and bound. E2 is the
  ratio of unrounded window holdings.
* **Reported, never gated:** resolve_edge's contradiction and invalidation
  agreement (`contradiction_agreement`). The invalidation candidates only
  approximate graphiti's hybrid search (§2.9).
* **Absolute accuracy is not reported** for the routing sites. Only discordant
  items are adjudicated (§2.6), and agreements move no paired difference.

### 2.4 Tie-breaks

* **Rounding.** Intervals are rounded to `SHOWN_DECIMALS` (4), and bars are
  decided on the rounded numbers. A lower bound of exactly −margin passes, as
  does an E1 bound of exactly the control's point + 0.05.
* **No pairs.** A site with no pair (n = 0) cannot pass.
* **Routing.** On an item both arms label alike, the pair moves nothing. On a
  discordant item, gold decides. A gold label neither arm gave makes both
  wrong, so that pair moves nothing either.
* **resolve_edge.** The arm's merge target is the first uuid it returns as a
  duplicate. A fixture gold marks unjudgeable leaves n.
* **Summaries.** An item is every node either arm returned a summary for. A
  node an arm returned no summary for is rated `dropped`, which is not
  faithful.
* **Failures.** A pair exists only where both arms have a run line, so a
  control failure removes that fixture from n. A luna failure does the same
  and also counts toward the failure gate. Control failures gate nothing.
* **Every check must hold.** Extraction needs E1 and E2, resolve_edge R1 and
  R2, and each site its failure gate. The routing layer passes only if C1 and
  Q1 both pass (`routing_layer_decision`). One pass and one fail keep both
  sites on gpt-4o-mini.

### 2.5 Decision rule (under the 2026-11-19 amendment)

* **A FAIL stays on gpt-4o-mini, on `OPENAI_API_KEY`, as today.**
* **A passing graphiti site** (extraction, resolve_edge or summaries) ships
  nothing in this task. Task 9944, which owns the plan-cap classifier, latch,
  park and L2 escalation, is pending. Each passing graphiti site is filed as a
  flip follow-up with `dependencies='9944'`, routed per `prompt_name`. If task
  9943's per-prompt plan route has landed, the follow-up reuses it; otherwise
  it names the blocking dependency on 9943 rather than building a second
  router.
* **A joint classifier+router PASS flips the routing layer in this task.** It
  ships `routing.model` gpt-6-luna, `routing.reasoning_effort` low, and
  `routing.openai_api_url`/`openai_api_key` as `${OPENAI_PLAN_PROXY_URL}` and
  `${OPENAI_PLAN_PROXY_KEY}`, with no API fallback. On any LLM error, including
  a plan cap, the classifier and router already degrade to their non-LLM path.
  A routing pin with an unset proxy variable raises `RoutingEndpointUnset`
  inside that same path, and never inherits `OPENAI_API_KEY`.
* **No passing site keeps gpt-4o-mini on `OPENAI_API_KEY` as a cap fallback.**
* **Cost never enters the decision.** It is reported (§2.7).

### 2.6 Adjudication

There is one adjudicator, self-blinded, as in O6 §2.3: the agent that ran the
arms. Each item gets one label and a rationale quoting the span it rests on.
Each verdict file must be accepted by its fail-closed loader, which refuses a
missing, duplicate, unknown-label, rationale-less or out-of-sample entry.

* **Extraction.** `replay_overreach_extraction.py worksheet` interleaves both
  arms' minted edges per episode by sha256(edge_uuid) and names no arm. Every
  edge is labelled in worksheet order with the episode body read in full. The
  rubric is 2026-10-01 §2.6 verbatim, with its §2.10 garble conventions, and
  one reading for non-decision records: **holding** means "states the record's
  own assertion within its scope". Output: `verdicts-extraction.json`, in
  `load_verdicts` format, with `frame.json`'s `sample.definition` as its sample
  block.
* **resolve_edge gold.** `worksheet --kind resolve_gold` shows each fixture's
  new fact, related facts and invalidation candidates from the fixture alone.
  It is labelled before any arm output is opened: the duplicate set among the
  related facts, the contradicted candidates, or unjudgeable. Output:
  `gold-resolve-edge.json`.
* **Summaries.** `worksheet --kind summaries` shows each item's inputs: the
  previous summary, the episode body, the previous episodes and the incident
  facts. Both arms' summaries are shown unattributed as A and B, with A the arm
  whose sha256(fixture_id:arm) sorts first. Each is rated `faithful`,
  `unsupported` (asserts or contradicts beyond its inputs) or `dropped` (no
  summary). The key file stays unopened until every rating is written. Output:
  `ratings-summaries.json`, then the key as `ratings-summaries-key.json`.
* **Routing.** `worksheet --kind routing` lists only the discordant items. Each
  shows its text, both labels unattributed in sha256 order, and the site's
  full vocabulary. The correct label is chosen from that vocabulary and may be
  neither arm's. Output: `verdicts-routing.json`.
* **Blinding residuals.** Per-arm run content and any score stay unopened
  during labelling. Before adjudication only counts and exit codes are read.
  What was seen is recorded in `provenance.json` under
  `adjudication.blinding_residuals`.

### 2.7 Cost

* **Tokens per call**, per site and arm, come from each record's measured
  usage: graphiti's `AttributingTokenUsageTracker` for the graphiti sites, the
  `UsageRecordingOpenAI` response hook for the routing sites. Output tokens
  include reasoning tokens.
* **API list price, for comparison only:**
  `score_write_triage_pairs.py::LIST_PRICES` as of `LIST_PRICES_AS_OF`
  2026-09-30, in USD per Mtok input / output: gpt-4o-mini 0.15 / 0.60,
  gpt-6-luna 0.10 / 0.50. There is no cached-input discount. luna runs on the
  plan and is not billed per token, and no passing site keeps an API
  fallback. So luna's list figure is an API-equivalent, not a spend.
* **Per day:** USD per call × calls/day per site. Calls/day is measured
  read-only from the write journal over the last 7 days before scoring, and
  each figure is tagged M (measured) or I (inferred).
* **Plan-window share:** luna's weekly API-equivalent USD over each end of an
  inferred plan window of $280–470/week. It is reported as an inferred range,
  tagged `inferred`, because the window is not measured.

### 2.8 Failure handling

This is O6 §2.6's rule, applied to every site.

* **Transient failures** (rate limit, connection error, 5xx) are failure lines
  in `replay.jsonl` or `runs.jsonl`. They are resumed once, by re-running the
  identical argv. That fills in the missing records and re-runs no result.
* **Routing sites.** The production classifier and router catch every LLM
  error and serve their non-LLM fallback. A routing call that gets no chat
  completion back is therefore raised as
  `luna_eval_sites.py::RoutingCallUnanswered` and recorded as a failure line
  (fix `40d32a178a`). It is not scored as the fallback label. A completion
  whose content does not parse keeps production's degraded label: the
  classifier records it in `fallback`, the router serves the broad route. That
  is the model's answer, and production serves exactly that.
* **Deterministic failures** are recorded and escalated, and never tuned
  around: an HTTP 400 on a parameter, a refusal, or the same fixture
  exhausting its output length again after a resume. A luna failure line
  that persists counts toward the failure gate (§2.3).
* **Extraction refusal.** An extraction replay left with failure lines makes
  `score` exit 2. That is a refusal, not a PASS or a FAIL. It is escalated,
  and no extraction verdict is reported.
* **Embeddings.** No leg embeds: the summaries leg passes `NullEmbedder`, and
  extraction and resolve_edge make LLM calls only. An embeddings call during
  any run is escalated as a defect (ruling `esc-9945-2`).
* **No arm is re-run or re-tuned after a result**, whichever way it falls.

### 2.9 Residuals, stated before the run

* **Classifier input.** The journal keeps a 200-character prefix of each
  `add_memory` body, and the classifier fixtures are those prefixes.
  Production sends up to 2,000 characters.
* **Graph state at draw time.** resolve_edge's siblings and candidates, and the
  summaries' previous summaries and incident facts, are the graphs as read on
  2026-11-22. That is not what each call saw when it first ran.
* **resolve_edge candidates are approximate.** They stand in for graphiti's
  hybrid search (`EDGE_HYBRID_SEARCH_RRF`) with incident edges in sha256 order,
  which is why contradiction agreement is not gated.
* **The proxy strips `max_output_tokens`**, so luna's graphiti calls run
  without the output cap the control is sent.
* **Proxy overhead.** About 300 tokens per call are included in luna's
  reported usage on the plan proxy. This is the plan's figure, not re-measured
  here, so luna's tokens per call overstate the model's own.
* **Summary inputs.** These are frozen in a private file and verified against
  the committed manifest by `check_summary_inputs` before every summaries run
  and worksheet.
* **Routing parse failures.** A router completion that does not parse is
  scored as the broad route production serves. `ReadRouteResult` carries no
  fallback marker, so such an answer cannot be told apart from an LLM answer
  of broad.
* **Classifier frame shortfall.** The frame holds 203 fixtures, not 300, and
  only 3 come from the two graph-bound categories (§2.1).
