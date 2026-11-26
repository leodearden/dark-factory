# Write-triage judge accuracy report

## Per-class accuracy

| class | n | correct | accuracy |
|---|---|---|---|
| duplicate | 75 | 39 | 0.52 |
| distinct | 3 | 0 | 0.0 |
| pseudo_contradiction | 6 | 1 | 0.1667 |
| distractor | 0 | 0 | None |

## Confusion — expected class by observed verdict

| class | amended | contested | restated | stored |
|---|---|---|---|---|
| duplicate | 30 | 33 | 9 | 3 |
| distinct | 1 | 0 | 2 | 0 |
| pseudo_contradiction | 0 | 5 | 1 | 0 |
| distractor | 0 | 0 | 0 | 0 |

## Where the writes attached, and which band answered

- duplicates attaching to their OWN canonical (strict): 18/75 = `0.24`
- duplicates attaching to ANYTHING: 72/75 = `0.96`
- attaches landing on a record that is NOT the case's canonical: 58/81 of all attaches = `0.716`
- attaches on a `distinct`/`pseudo_contradiction`/`distractor` case: 9/9 = `1.0`
- the case's canonical was ON the slate: 28/84 = `0.3333`
- cases whose canonical is no longer in the corpus: `21`
- judge accuracy restricted to the cosine MIDDLE band [t_low, t_high): 33/73 = `0.4521`
- LLM calls made: `84`, tokens: `257631`

### Band split — expected class by the cosine band (`decide_band`) it fell in

| class | restated | judge | stored |
|---|---|---|---|
| duplicate | 9 | 66 | 0 |
| distinct | 1 | 2 | 0 |
| pseudo_contradiction | 1 | 5 | 0 |
| distractor | 0 | 0 | 0 |

## Routing — what `route_band` sent the judge

- restate-band (cosine >= t_high) writes sent to the judge: 11/11; kept `restated` as near-verbatim repeats of their band winner: 0/11
- judge accuracy on the restate-band writes it was sent: 7/11 = `0.6364`
- judge accuracy on every case it saw: 40/84 = `0.4762`

### Route split — expected class by the route `route_band` took

| class | restated | judge | stored |
|---|---|---|---|
| duplicate | 0 | 75 | 0 |
| distinct | 0 | 3 | 0 |
| pseudo_contradiction | 0 | 6 | 0 |
| distractor | 0 | 0 | 0 |

## Duplicate attach split (a distribution, not an error term)

- `restated`: 9
- `amended`: 30

## Contested

- contested verdicts observed: **38**, all of which are FALSE POSITIVES.
- ground truth available: `False`
- `no_positive_contested_labels: the fixture carries 6 pseudo_contradiction records, every one curator-adjudicated NOT a contradiction, and 0 records labelled as a genuine contradiction. Contested recall and precision are therefore unmeasurable against this corpus; only the false-positive count below is a measurement.`

## Caveats

- No accuracy floor is asserted anywhere in this script or its tests (PRD D10). This artifact is evidence for a human decision, not a gate.
- false_contested counts EVERY contested verdict, and every one of them is a false positive — see contested_ground_truth. A judge structurally incapable of ever answering `contested` would score identically to a perfect one here, so a low number is not evidence that the contradiction detector works.
- The duplicate class accepts BOTH `restated` and `amended`, because the curator's labels do not separate a verbatim restatement from a rediscovery carrying a novel fragment. The split between them is reported as a distribution and is not scored as error. NOT SCORED IS NOT THE SAME AS NOT CONSEQUENTIAL: `_TRIAGE_ATTACH_KINDS` in `server/tools.py` files a `restated` verdict as a SIGHTING, which `grouped_read` only counts, and an `amended` one as an AMENDMENT, whose text is digested into the canonical's grouped read. A swing between the two therefore changes what an operator reads while leaving every accuracy above unmoved, so read this split as a behaviour selector rather than as noise.
- There is NO distractor control class in this mode. A retrieved slate carrying no correct attach target is the ORDINARY case here rather than one this script constructs, so the control would measure nothing the population does not already show. `production_shape.canonical_in_slate` is the measured equivalent — the share of cases whose own canonical reached the prompt at all — and `canonical_absent` counts the cases whose canonical is no longer in the corpus. Those are KEPT in the population, because production meets them.
- Every figure here is measured over the population production would actually route: the slate, the band winner, the cosine band and the route all come from a live retrieval through `retrieve_candidates` / `decide_band` / `route_band` / `select_judge_candidates` at this config's `candidate_k`, `t_high` and `t_low`, and the judge was asked for every case `route_band` sends it: the [t_low, t_high) middle band, plus each restate-band write that does not repeat its band winner, or the winner's canonical, near-verbatim. So `per_class` MIXES routes — a near-verbatim `restated` and a below-floor `stored` are counted there without an LLM having seen the case. `routing.judged` is the judge's own accuracy; `production_shape.middle_band` is its accuracy on the cosine middle band, and `routing.restate_band` its accuracy on judged high-cosine writes. A verdict that is correct for its curator label can still attach to ANOTHER RECORD, which `per_class` cannot show. `production_shape.duplicate_attach.strict` and `production_shape.wrong_record_attach` score `attach_target_id`, the record production files the write against. For a judged attach that is the candidate the judge NAMED, hoisted to its canonical (`judged_candidate_id`); otherwise it is the band winner (`band_winner_id`). So for a judged write `strict` is recall at `judge_candidate_count`, not at 1. An artifact from a judge that named no candidate (before task 5794) attached every judged write to the band winner, and its rows carry no `judged_candidate_id`.
- Under the shipped `contests` definition — the entry says a candidate is wrong, outdated or different (Leo's ruling 2026-09-30, plans/write-triage-flip-readiness-prd.md §11.3 C1'') — the judge answered `contested` on 5 of 6 pseudo_contradiction cases (of the 6 the judge saw); all outcomes: `amended` 0, `contested` 5, `restated` 1, `stored` 0. These records were adjudicated NOT contradictions under the EARLIER definition ("cannot be true at the same time"), and this report still scores `contested` on them as wrong and counts it in false_contested. The fixture is deliberately not relabelled, so this reports how the new definition reads them without asserting which reading is right.
- The judge, `gpt-6.1-sol`, answered `contested` on 33 of 75 duplicate cases (of the 75 the judge saw). The curator labelled each duplicate the same claim as its canonical, so this report scores `contested` on one as wrong; with triage enabled, each is a write this judge would flag `contested` for a human to read. The `contests` definition was settled on a frontier reasoning judge: its live boundary test, fused-memory/tests/server/test_write_triage_judge.py::TestTheShippedWordingLive, pins one, because gpt-4o-mini answers that test's outdated cases `contested` under the earlier wording too. Where this judge is not the model that test pins, the figure measures the wording on a judge it was not settled on: read it beside a run on that model before deciding which judge the flip ships.

## Provenance

- `fixture_path`: `tests/fixtures/write_triage_calibration.jsonl`
- `judge_provider`: `openai`
- `judge_model`: `gpt-6.1-sol`
- `judge_system_prompt_sha256`: `897885879ce30793ba614b837d1ba28f4732ae8ff27dea41eda65bb1d8fbded5`
- `limit`: `None`
- `canonical_aliases_path`: `tests/fixtures/write_triage_calibration.canonical_aliases.json`
- `canonical_aliases_count`: `3`
- `cases_path`: `calibration/write_triage_judge_accuracy_report.cases.jsonl`
- `slate_mode`: `retrieved`
- `field_chars`: `4000`
- `record_count`: `104`
- `case_count`: `84`
- `candidate_count`: `5`
- `candidate_count_min`: `5`
- `distractor_count`: `None`
- `distractor_count_requested`: `None`
- `judge_candidate_count`: `5`
- `judge_enabled`: `True`
- `project_id`: `reify`
- `t_high`: `0.8868282526657318`
- `t_low`: `0.5229672433064231`
- `candidate_k`: `20`
- `canonical_absent`: `21`
- `degraded_retrievals`: `0`
- `self_retrieved`: `0`
