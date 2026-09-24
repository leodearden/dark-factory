# Write-triage judge accuracy report

## Per-class accuracy

| class | n | correct | accuracy |
|---|---|---|---|
| duplicate | 75 | 71 | 0.9467 |
| distinct | 3 | 0 | 0.0 |
| pseudo_contradiction | 6 | 5 | 0.8333 |
| distractor | 0 | 0 | None |

## Confusion — expected class by observed verdict

| class | amended | contested | restated | stored |
|---|---|---|---|---|
| duplicate | 58 | 1 | 13 | 3 |
| distinct | 2 | 0 | 1 | 0 |
| pseudo_contradiction | 4 | 1 | 1 | 0 |
| distractor | 0 | 0 | 0 | 0 |

## Where the writes attached, and which band answered

- duplicates attaching to their OWN canonical (strict): 14/75 = `0.1867`
- duplicates attaching to ANYTHING: 72/75 = `0.96`
- attaches landing on a record that is NOT the case's canonical: 65/81 of all attaches = `0.8025`
- attaches on a `distinct`/`pseudo_contradiction`/`distractor` case: 9/9 = `1.0`
- the case's canonical was ON the slate: 32/84 = `0.381`
- cases whose canonical is no longer in the corpus: `21`
- judge accuracy restricted to the MIDDLE band: 66/73 = `0.9041`
- LLM calls made: `73`, tokens: `160432`

### Band split — expected class by the band that answered

| class | restated | judge | stored |
|---|---|---|---|
| duplicate | 9 | 66 | 0 |
| distinct | 1 | 2 | 0 |
| pseudo_contradiction | 1 | 5 | 0 |
| distractor | 0 | 0 | 0 |

## Duplicate attach split (a distribution, not an error term)

- `restated`: 13
- `amended`: 58

## Contested

- contested verdicts observed: **2**, all of which are FALSE POSITIVES.
- ground truth available: `False`
- `no_positive_contested_labels: the fixture carries 6 pseudo_contradiction records, every one curator-adjudicated NOT a contradiction, and 0 records labelled as a genuine contradiction. Contested recall and precision are therefore unmeasurable against this corpus; only the false-positive count below is a measurement.`

## Caveats

- No accuracy floor is asserted anywhere in this script or its tests (PRD D10). This artifact is evidence for a human decision, not a gate.
- false_contested counts EVERY contested verdict, and every one of them is a false positive — see contested_ground_truth. A judge structurally incapable of ever answering `contested` would score identically to a perfect one here, so a low number is not evidence that the contradiction detector works.
- The duplicate class accepts BOTH `restated` and `amended`, because the curator's labels do not separate a verbatim restatement from a rediscovery carrying a novel fragment. The split between them is reported as a distribution and is not scored as error. NOT SCORED IS NOT THE SAME AS NOT CONSEQUENTIAL: `_TRIAGE_ATTACH_KINDS` in `server/tools.py` files a `restated` verdict as a SIGHTING, which `grouped_read` only counts, and an `amended` one as an AMENDMENT, whose text is digested into the canonical's grouped read. A swing between the two therefore changes what an operator reads while leaving every accuracy above unmoved, so read this split as a behaviour selector rather than as noise.
- There is NO distractor control class in this mode. A retrieved slate carrying no correct attach target is the ORDINARY case here rather than one this script constructs, so the control would measure nothing the population does not already show. `production_shape.canonical_in_slate` is the measured equivalent — the share of cases whose own canonical reached the prompt at all — and `canonical_absent` counts the cases whose canonical is no longer in the corpus. Those are KEPT in the population, because production meets them.
- Every figure here is measured over the population production would actually route: the slate, the attach target and the band all come from a live retrieval through `retrieve_candidates` / `decide_band` / `select_judge_candidates` at this config's `candidate_k`, `t_high` and `t_low`, and the judge was asked ONLY for the middle band. So `per_class` MIXES bands — a deterministic `restated` and a below-floor `stored` are counted there without an LLM having seen the case — and `production_shape.middle_band` is the judge's own accuracy. A verdict that is correct for its curator label can still attach to ANOTHER RECORD: `triage_write` files every non-`stored` outcome against `decision.canonical_id`, the band's argmax, not against whatever the judge reasoned about. `production_shape.duplicate_attach.strict` is the figure that accounts for that and `per_class` is not.

## Provenance

- `fixture_path`: `tests/fixtures/write_triage_calibration.jsonl`
- `judge_provider`: `openai`
- `judge_model`: `gpt-4o-mini`
- `limit`: `None`
- `canonical_aliases_path`: `tests/fixtures/write_triage_calibration.canonical_aliases.json`
- `canonical_aliases_count`: `3`
- `cases_path`: `calibration/write_triage_judge_accuracy_report.cases.jsonl`
- `slate_mode`: `retrieved`
- `field_chars`: `1200`
- `record_count`: `104`
- `case_count`: `84`
- `candidate_count`: `5`
- `candidate_count_min`: `5`
- `distractor_count`: `4`
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
