# O6 measured: gpt-6.1-sol at reasoning effort low and medium, on the 48 frozen ruling-overreach episodes

**Task 9065**, follow-up of task 9040 · ruling `esc-9065-2` (Leo, 2026-11-12,
item 596=D) · instrument `fused-memory/scripts/replay_overreach_extraction.py`
as of this document's pre-registration commit · `graphiti_core` 0.28.2

The governing trigger is §6.2 of
`docs/ruling-overreach-lever-reopen-2026-11-10/design.md`, cited below as
"9040".

**§1 and §2 are PRE-REGISTERED.** They were committed on their own, after the
instrument's code and before any replay ran or any API call was made. So
neither the arms, nor the decision mapping, nor the cost method could be tuned
toward the results. §3 onward are written after adjudication and scoring.
Every measured number there is cited from `report.json` in this directory by
key path. Run provenance is in `provenance.json`.

## 1. Question

9040 §6.2 states the trigger this task applies, verbatim:

> * **Trigger clause, from R to R+O6.** It fires when the O6 measurement's
>   `comparison.o6.passes` is true. The run is
>   `replay_overreach_extraction.py replay --arm control --arm o6` on the same
>   48 episodes, into one file. One `worksheet` interleaves both arms blind,
>   every edge is adjudicated in one sitting under the 2026-10-01 §2.6 rubric,
>   and then `score` runs (§5). `comparison.o6.passes` is §2.4's three checks,
>   computed by `lever_decision` with `OVERREACH_RATIO_MAX` and
>   `HOLDING_RETENTION_MIN`. The comparison is against the control replayed in
>   the same run and labelled in the same blind sitting, and all three must
>   hold:
>   1. population-weighted overreach ≤ 0.5 × the control's, per minted AND per
>      day;
>   2. its Wilson upper bound per minted < the control's point estimate;
>   3. holdings/day ≥ 0.8 × the control's.
>
>   The clause does not use fixed numbers from this report's control, because a
>   cross-sitting comparison would give up the pairing and blinding that §2.2
>   and §2.4 rely on. That run's `arms.control` is reported beside this
>   `r.arms.control` (0.1355 per minted, 12.7 overreach/day, 34.9
>   holdings/day), as a stability reading only.
>
>   If all three hold, file one task to route the class's `extract_nodes` and
>   `extract_edges` calls to gpt-6.1-sol. That task is `always_escalates`: the
>   measured cost goes to Leo for a spend ruling before anything is routed. If
>   any check fails, O6 is closed as O1 is, and the composite stays the lever
>   alone.

**Why there are two o6 arms, not one.** 9040 §7 recorded a residual: graphiti
would send gpt-6.1-sol temperature 0.0, and "if the API refuses it, the o6
replay records failure lines and exits 1, and the measuring task escalates
rather than patching graphiti". It came true. The architect's live probe
(`esc-9065-2`) found:

* temperature 0.0 is refused with HTTP 400, "Unsupported parameter:
  temperature", on every attempt;
* the same call with temperature omitted succeeds, but then runs at the API's
  default effort, medium, which nobody chose;
* an explicit effort of low succeeds.

graphiti_core 0.28.2's `llm_client/openai_client.py` treats only `gpt-5`, `o1`
and `o3` model names as reasoning models. For any other model it sends
temperature and never sends `reasoning`. So no configuration of the production
seam can omit temperature or choose an effort.

Leo ruled (`esc-9065-2`, item 596=D, "test both low and medium efforts"):

* fix it inside the replay script only, with no production or graphiti change;
* omit temperature and send an explicit reasoning effort;
* run two declared arms, `o6-low` and `o6-medium`, pre-registered here;
* score each against the same-run control under the unchanged §6.2 checks;
* the trigger fires if either passes, and the routing task carries both results
  to Leo.

So 9040's single `comparison.o6` becomes `comparison.o6-low` and
`comparison.o6-medium`, and the run is `--arm control --arm o6-low --arm
o6-medium`. Low is the effort the production write-triage judge runs
gpt-6.1-sol at (`fused-memory/src/fused_memory/server/write_triage_judge.py`,
`fused-memory/calibration/write_triage_best_config.json`). Medium is the API default the
probe ran at. Arm `o6` is retired: it never ran, so no recorded result names
it.

## 2. Method

### 2.1 Frame

Identical to 9040 §2.1 and not restated here:

* the 48 episodes of the 2026-10-01 sample (`SampleDefinition()`);
* their previous episodes, read over GRAPH.RO_QUERY;
* graphiti_core's own `extract_nodes` and `extract_edges`, in process, with no
  driver;
* graphiti's in-batch exact-fact dedupe;
* the stated divergence from production: nothing is resolved against the live
  graph, which every arm shares and the control absorbs.

The worksheet is regenerated read-only by
`audit_ruling_overreach.py --emit-worksheet` for `dark_factory` and `reify`.
`replay` refuses it if its (graph, stratum) episode counts differ from the 4716
`report.json` `sample.sizes`.

### 2.2 Arms

| arm | model | client | sampling control sent | instructions |
|---|---|---|---|---|
| `control` | gpt-4o-mini (config `llm.model`) | the client `graphiti_client.py::build_llm_client` builds, as configured (graphiti `OpenAIClient`) | temperature 0.0 (`cfg.llm.temperature or 0.0`; `config.yaml` sets none) | none |
| `o6-low` | gpt-6.1-sol (`O6_MODEL`) | `replay_overreach_extraction.py::ExplicitEffortOpenAIClient` around that client | `reasoning.effort` low, no temperature | none |
| `o6-medium` | gpt-6.1-sol | the same | `reasoning.effort` medium, no temperature | none |

* **All three take the control's instructions**: no
  `custom_extraction_instructions` (`NO_INSTRUCTIONS`). O1 is not re-run.
* **The effort client** wraps the client the production seam builds, so its API
  key, base URL, SDK client, `max_tokens` and `AttributingTokenUsageTracker`
  are the seam's. It overrides only graphiti's `_create_structured_completion`
  hook, sending `reasoning={'effort': e}` on the Responses API with no
  temperature. Its JSON-mode path refuses rather than send a call without the
  effort; extraction never takes that path, because both calls pass a
  `response_model`.
* **No verbosity is sent.** graphiti's own reasoning-model branch also sends
  `text.verbosity` 'low'. Nobody chose that parameter for this model, and the
  ruling names only temperature and effort, so it is omitted and runs at the
  API default. This is a stated residual.
* **max_tokens as production sends it.** `extract_nodes` passes none, so it
  gets the client's: `llm.max_tokens` 4096. `extract_edges` passes graphiti's
  own 16384. Reasoning tokens count against `max_output_tokens`. The cap is not
  raised for the experiment: an exhaustion is itself a finding for the spend
  ruling (§2.6).
* **One run.** `replay --arm control --arm o6-low --arm o6-medium` into one
  file. That is 3 arms × 48 episodes × 2 calls = 288 LLM calls, 192 of them on
  gpt-6.1-sol. Each arm's records name its model and `reasoning_effort`.
  `check_arm_client` refuses any client that is not on the arm's model and
  effort, before any call.

### 2.3 Adjudication

As 9040 §2.2, cited:

* **One blind worksheet.** `worksheet` writes one row per episode with its body
  and every arm's minted edges interleaved by sha256(edge_uuid), naming no arm,
  model or effort. Near-duplicate facts from the three arms sit side by side.
* **One sitting, one adjudicator, self-blinded.** The agent that ran the
  replay labels every minted edge in worksheet order, reading each episode body
  in full beside its edges. It uses the 2026-10-01 design.md §2.6 rubric
  verbatim, with that document's §2.10 garble conventions: an inverted causal
  link, a misattributed actor or a quantity on the wrong thing is `overreach`
  (ii); a faithful fact on the wrong endpoint is `misbound`. Each edge gets one
  label and one rationale quoting the span it rests on. Near-duplicates are
  labelled consistently.
* **Not opened during labelling**: the replay file and anything scored from
  it. What the adjudicator saw of per-arm output before labelling is recorded
  as a blinding residual in `provenance.json`.
* **Verdict file**: `verdicts.json` here, in `load_verdicts` format, with the
  4716 default `SampleDefinition` as its sample block. `score` validates it once
  over the union of arms and refuses a missing, duplicate, unknown-label or
  out-of-sample verdict.

### 2.4 Decision

* **Per arm**: `r.comparison.o6-low` and `r.comparison.o6-medium`, each
  `lever_decision` against this run's `control`. The checks and bars are 9040
  §2.4's and §6.2's, unchanged: `OVERREACH_RATIO_MAX` 0.5 per minted and per
  day, the Wilson upper bound below the control's point, and
  `HOLDING_RETENTION_MIN` 0.8. All must hold for an arm to pass.
* **The trigger fires if either arm passes.** The routing task names the
  passing effort, or **low if both pass**: the cheaper effort at an equal
  verdict.
* **If neither passes**, O6 is closed as O1 is, and the O3+O4 composite stays
  the lever alone (9040 §6.2).
* **Stability reading only.** This run's `r.arms.control` is reported beside
  9040's `r.arms.control` (0.1355 overreach per minted, 12.7 overreach/day,
  34.9 holdings/day) and beside `r.fidelity`. It never enters the decision.

### 2.5 Cost

* **Tokens**: each arm's `extract_nodes` and `extract_edges` input and output
  totals from `r.arms.<arm>.tokens`, per episode. On the Responses API,
  `usage.output_tokens` includes reasoning tokens, which bill as output
  (`fused-memory/scripts/score_write_triage_pairs.py::LIST_PRICES` docstring),
  so no separate reasoning count is needed.
* **Prices**: `score_write_triage_pairs.py::LIST_PRICES`, as of
  `LIST_PRICES_AS_OF` 2026-09-30: gpt-4o-mini 0.15 / 0.60 and gpt-6.1-sol
  2.00 / 10.00 USD per Mtok input / output. List price, no cached-input
  discount.
* **Reported**: per episode; per day at about 670 class episodes/day (the
  2026-10-01 design.md §10.2 measured volume); per 30 days; the ratio to the
  control's same two calls; and against 9040 §5's estimate, which assumed
  gpt-4o-mini's token counts ($0.01669 per episode, $11.18/day).

### 2.6 Failure handling

* **Transient failures** (rate limit, connection error, 5xx) are failure lines
  in the replay file. They are resumed by re-running the identical argv, as the
  script documents. That resumes the missing records; it re-runs no result.
* **Deterministic failures** are escalated and never tuned around: an HTTP 400
  on a parameter, an unsupported effort, or the same episode exhausting its
  output length again after a resume.
* **No arm is re-run or re-tuned after a result**, whichever way it falls.

---

*§3 onward were written after adjudication and scoring. Every measured number
is cited from this directory's `report.json` by key path, abbreviated `r.`;
token and cost arithmetic is shown in §6. Run provenance is in
`provenance.json`.*

## 3. Run

* **One replay invocation, no resume.** `replay --arm control --arm o6-low
  --arm o6-medium` ran 2026-11-12T15:13:30Z → 16:03:44Z and exited 0. It made
  288 `POST /v1/responses` calls, all HTTP 200, with no retry.
* **144 records, 0 failure lines**: 48 per arm (`r.arms.<arm>.episodes`).
  Every record names its arm's model and effort: control (gpt-4o-mini,
  `reasoning_effort` null), o6-low (gpt-6.1-sol, low), o6-medium
  (gpt-6.1-sol, medium) (`r.arms.<arm>.model`,
  `r.arms.<arm>.reasoning_effort`).
* **No deterministic failure.** No HTTP 400, no unsupported effort and no
  output-length exhaustion, so §2.6 never fired. The largest single call
  output was 4,459 tokens: `extract_edges` on o6-medium, 27% of graphiti's
  16,384. The largest `extract_nodes` output was 735 tokens, 18% of 4,096
  (`replay.jsonl`, per-record `tokens`).
* **Minted** (`r.arms.<arm>.minted`):

  | arm | minted | per episode |
  |---|---|---|
  | control | 328 | 6.8 |
  | o6-low | 905 | 18.9 |
  | o6-medium | 1,057 | 22.0 |

  The worksheet held all 2,290. Every one was adjudicated, and `score`
  accepted `verdicts.json` on its first invocation (`r.verdicts` 2290).
* **graphiti dropped 5 edges** whose endpoint was not among the extracted
  nodes (1 source, 4 target). Every episode read exactly 10 previous
  episodes, identically across arms.

## 4. Stability reading (not part of the decision)

| | this run's `r.arms.control` | 9040 `r.arms.control` |
|---|---|---|
| overreach per minted, population-weighted | 0.1613 [0.1219, 0.2105] | 0.1355 [0.0977, 0.1849] |
| overreach / day | 17.1 | 12.7 |
| holdings / day | 43.4 | 34.9 |
| minted (unweighted) | 328 | 292 |

* The two control readings' intervals overlap.
* `r.fidelity` holds: the 4716 production baseline is 0.1238
  [0.0876, 0.1721], against this control's 0.1613, and
  `r.fidelity.intervals_overlap` is true.
* gpt-4o-mini at temperature 0.0 did not mint the same edges twice: 328 here
  against 292 in 9040, on the same 48 episodes and the same previous-episode
  reads. That repeat variance is why §6.2 compares against the same-run
  control rather than against 9040's numbers.

## 5. Results: each o6 arm against the same-run control

| `r.comparison.<arm>` | o6-low | o6-medium | bar |
|---|---|---|---|
| `overreach_ratio_minted` | 0.0 | 0.0 | ≤ 0.5 |
| `overreach_ratio_per_day` | 0.0 | 0.0 | ≤ 0.5 |
| Wilson upper bound, per minted (`r.arms.<arm>.population_weighted.overreach_rate_minted.ci[1]`) | 0.0052 | 0.0045 | < control point 0.1613 |
| `holding_retention` | 2.4224 | 2.6895 | ≥ 0.8 |
| `checks` | all four true | all four true | |
| `passes` | **true** | **true** | |

* **Overreach.** Both o6 arms minted no overreach edge in the sitting:
  0 of 905 for o6-low and 0 of 1,057 for o6-medium
  (`r.arms.<arm>.rates.all.label_counts`). Every one of the 48 overreach
  verdicts, and every one of the 7 misbound verdicts, landed on a control
  edge. Control's 48 split as 42 class (ii), 4 class (iii) and 2 class (iv),
  read from the rationales after scoring. They hit 24 of the 48 episodes
  (`r.arms.control.rates.all.episode_hit_rate` 0.5). Both o6 arms have an
  episode hit rate of 0.0 [0, 0.0741].
* **Holdings/day** is 105.2 on o6-low and 116.8 on o6-medium, against
  control's 43.4 (`r.arms.<arm>.population_weighted.holdings_per_day`). The
  retention ratio far exceeds the bar, but it is inflated by paraphrase
  multiplicity: the o6 arms state each holding several ways (§8). The
  holding SHARE is slightly lower than control's, 0.3558 and 0.3357 against
  0.4094 (`r.arms.<arm>.population_weighted.holding_share.rate`).
* **Entities.** Each o6 arm extracted about 21.6 entities per episode against
  control's 13.4. Per episode, a mean 60% (low) and 55% (medium) of control's
  entity names recur by exact normalised name in the o6 arm's set
  (`r.arms.<arm>.entities.mean_share_of_control`). The o6 arms name entities
  more descriptively, so exact-name overlap understates shared referents.
* **Unweighted label counts** (`r.arms.<arm>.rates.all.label_counts`):

  | arm | holding | context | bookkeeping | overreach | misbound | unjudgeable |
  |---|---|---|---|---|---|---|
  | control | 145 | 113 | 15 | 48 | 7 | 0 |
  | o6-low | 360 | 456 | 89 | 0 | 0 | 0 |
  | o6-medium | 402 | 555 | 100 | 0 | 0 | 0 |

* **Low against medium.** The verdict cannot separate them: both reached
  zero. Medium costs about a third more (§6) and mints about 17% more edges,
  for 11% more holdings/day.

## 6. Cost

Tokens are `r.arms.<arm>.tokens`, summed over `extract_nodes` and
`extract_edges`. On the Responses API, output tokens include reasoning
tokens. The prices are `score_write_triage_pairs.py::LIST_PRICES` as of
2026-09-30, at list price with no cached-input discount: gpt-4o-mini 0.15 /
0.60 and gpt-6.1-sol 2.00 / 10.00 USD per Mtok.

| arm | input tok / episode | output tok / episode | USD / episode | USD / day (670 episodes) | USD / 30 days | × control |
|---|---|---|---|---|---|---|
| control | 5,190.4 | 684.7 | $0.001189 | $0.80 | $24 | 1.0 |
| o6-low | 5,337.8 | 1,681.8 | $0.02749 | $18.42 | $553 | 23.1 |
| o6-medium | 5,338.4 | 2,586.5 | $0.03654 | $24.48 | $734 | 30.7 |

* **Arithmetic.**
  * Control: 249,140 input and 32,865 output tokens cost $0.05709 over 48
    episodes.
  * o6-low: 256,216 input and 80,724 output cost $1.3197.
  * o6-medium: 256,242 input and 124,151 output cost $1.7540.
  * USD/day multiplies USD/episode by 670, the 2026-10-01 §10.2 measured class
    volume.
* **Increment over today**: +$17.62/day at low and +$23.69/day at medium.
* **Against 9040 §5's estimate** of $0.01669/episode ($11.18/day), which
  assumed gpt-6.1-sol emits gpt-4o-mini's output tokens. The measured cost is
  1.65× that at low and 2.19× at medium. Output per episode is 2.46× control's
  at low and 3.78× at medium, part reasoning tokens and part the larger number
  of edges written out. These measured figures replace that estimate.
* **Scope.** This prices the two extraction calls only. The other graphiti
  calls per write stay on gpt-4o-mini. But they include per-edge resolution,
  which scales with minted edges, and the o6 arms mint 2.8× and 3.2× the
  edges. Their cost was not measured (§8).

## 7. Trigger

The pre-registered mapping (§2.4), applied:

* `r.comparison.o6-low.passes` is **true**, and
  `r.comparison.o6-medium.passes` is **true**.
* **The 9040 §6.2 R→R+O6 trigger FIRES.** Both arms pass, so the routing
  effort is **low**, the cheaper effort at an equal verdict.
* One routing task is filed, `task_kind` deterministic and
  `always_escalates` with no `before_done`: a pure gate. It carries both
  arms' results and the §6 cost to Leo for a spend ruling before anything is
  routed. Nothing is routed by this task.
  * Ticket `tkt_0RWXDK946QRQ7SVWBHCQYG49AF`, filed 2026-11-12 with
    `spawned_from` 9065. The curator assigns the task id asynchronously.
  * The routing it gates names effort low, inherits
    `ExplicitEffortOpenAIClient`'s no-temperature and explicit-effort
    handling promoted into `fused-memory/src/fused_memory/backends/llm_clients.py`,
    and keeps today's `max_output_tokens` caps (§3: no call above 27%).

## 8. Residuals

* **Single adjudicator, and a style leak.** One agent labelled all 2,290
  edges, so no inter-rater agreement exists. The worksheet named no arm, but
  the arms wrote in visibly different styles. Some edges used terse entity
  names ("task 4723", "Leo"). Others used expanded descriptive ones
  ("Product requirements document session #7882", spelled-out acronyms).
  So same-arm edges were recognisable as a group, and the stronger model's
  style was guessable.
  * Every overreach and misbound verdict landed on control. That perfect
    separation cannot rule out adjudicator expectation as a contributor.
  * Each such verdict quotes the span it contradicts and names the garble,
    so each is checkable on its own.
  * A second, independent rater on a sample of o6 edges is the cheap check
    if the spend ruling wants one.
* **What was seen before or during labelling** (`provenance.json`
  `adjudication.blinding_residuals`):
  * per-arm record counts and each arm's (model, effort);
  * the five dropped-edge warnings, by type only;
  * the worksheet's 2,290 total;
  * a running label tally over the union of arms after each episode.

  `replay.jsonl` and `report.json` were not opened until labelling ended.
* **One run per arm.** gpt-6.1-sol's run-to-run variance is unmeasured. §4
  shows the control's own repeat variance.
* **Replay is not production dedupe.** The replay applies only graphiti's
  in-batch exact-fact dedupe and resolves nothing against the live graph.
  * The o6 arms' 2.8–3.2× minted volume is largely near-paraphrase.
    Production resolution would merge some of it.
  * That merge would cost the per-edge resolution calls §6 does not price.
  * It also makes `holding_retention` an overstatement of distinct holdings
    retained. The decision-bearing quantity is the overreach ratio, which is
    0 either way.
* **Previous episodes as read today.** Every episode read 10 previous
  episodes from the live graphs on 2026-11-12, not as they stood when the
  episode was written. All arms share this.
* **No verbosity sent.** graphiti's reasoning-model branch sends
  `text.verbosity` low. The effort client sends none, so the o6 arms ran at
  the API default verbosity. A routing task that adds verbosity changes the
  measured configuration.
* **max_tokens as production.** No call exceeded 27% of its cap, so the
  production caps hold for this population. This is no evidence about longer
  episodes.
* **The effort client lives in a script.** `ExplicitEffortOpenAIClient` is
  replay-only. graphiti_core 0.28.2 sends temperature to, and withholds
  reasoning from, any model it does not recognise as a reasoning model, so
  routing must inherit this handling (§7's task names it).
* **Population.** The 48 episodes are from two graphs (dark_factory, reify)
  in the frozen 2026-08-25..10-01 window. The live class volume of about 670
  episodes/day spans ten graphs.
