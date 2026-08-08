# CURATION — fable-trial-v2 curated hard fixture pool

<!-- GENERATED FILE — do not edit by hand.
     Rendered from `_meta/curation.json` by
     `scripts/mint_hard_v2_fixtures.py --author`, and pinned
     byte-for-byte by `test_hard_v2_fixture_pool.py`. Edit the
     manifest and regenerate. -->

- **Task**: 3631 — fable-trial-v2 β1: mint the curated v2 hard fixture pool
- **PRD**: `plans/fable-architect-trial-v2-prd.md`
- **Cohort**: `fable-trial-v2-hard`
- **Candidates**: 41 — 39 included, 2 excluded

## Census

- **Source**: `<project_root>/data/orchestrator/runs.db (events table)`
- **Census date**: 2026-08-08

```sql
SELECT DISTINCT task_id FROM events
WHERE event_type='invocation_end' AND role='architect'
  AND ( (json_extract(data,'$.subtype')='error_max_turns'
         AND json_extract(data,'$.turns')=121)
     OR  json_extract(data,'$.subtype')='error_max_budget_usd' )
```

The turns-at-exhaustion 121 clause binds the max_turns arm ONLY. Budget exhaustion terminates at an arbitrary turn count (know-live 543 exhausted its budget at 113 turns), so applying 121 globally yields 23 candidates instead of the recorded 41. 121 is know-live production max_turns.architect=120 plus one.

| project | distinct tasks |
|---|---:|
| reify | 36 |
| dark_factory | 4 |
| know_live | 1 |
| **total** | **41** |

**Reproducibility**: The source dbs are live, so purely ADDITIVE drift (new exhaustions since the recorded census_date) is expected and harmless: the manifest pins the exact recorded task_ids and the curated pool is a dated snapshot, not a standing query. Refusing to re-author, which would silently pull uncurated candidates into the pool. Compare the live ids against census.task_ids in _meta/curation.json: if every recorded id is still present, the pool is intact and this exit is informational. If a recorded id is MISSING, the pool genuinely can no longer be re-derived from these dbs — investigate before touching the manifest.

## Curation criterion

Exclude a candidate when its brief fails to state an implementable goal. Never a length threshold: brief_chars is recorded per row as evidence for the judgement, not as the rule.

No candidate in the census is an adversarial/red-team fixture — all 41 are real product tasks from the three checkouts' task trees. The PRD's skip-adversarial rule therefore excludes nothing here; the scan is recorded so its emptiness is a finding rather than an omission.

## Candidates

`brief chars` is recorded as EVIDENCE for each judgement, never as the rule.

| task_id | project | brief chars | status | decision | mint mode | reason |
|---|---|---:|---|---|---|---|
| 2320 | reify | 224 | done | include | referenced | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 2324 | reify | 584 | done | include | planrate_only | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 2325 | reify | 137 | done | include | planrate_only | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 2330 | reify | 750 | done | include | planrate_only | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 2336 | reify | 119 | done | include | planrate_only | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 2379 | reify | 139 | done | include | planrate_only | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 2384 | reify | 285 | done | include | planrate_only | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 2531 | reify | 116 | done | include | planrate_only | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 2573 | reify | 299 | done | include | planrate_only | INCLUDE. The brief states an implementable goal, and although the architect raised unactionable escalation(s) on this task it reached status `done` — the unactionability was resolved, so this is a genuinely hard task that landed, not an ill-posed one. Terminal status is the discriminator (cf. reify 3378, cancelled AND unactionable, which is excluded). |
| 2654 | reify | 195 | done | include | referenced | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 2655 | reify | 162 | done | include | planrate_only | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 2656 | reify | 100 | done | include | referenced | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 2696 | reify | 4039 | done | include | planrate_only | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 2699 | reify | 8285 | done | include | planrate_only | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 2778 | reify | 674 | done | include | planrate_only | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 2908 | reify | 166 | done | include | planrate_only | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 2911 | reify | 254 | done | include | referenced | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 2958 | reify | 250 | done | include | referenced | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 3004 | reify | 171 | done | include | planrate_only | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 3024 | reify | 94 | done | include | referenced | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 3092 | reify | 323 | done | include | planrate_only | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 3095 | reify | 331 | done | include | referenced | INCLUDE. The brief states an implementable goal, and although the architect raised unactionable escalation(s) on this task it reached status `done` — the unactionability was resolved, so this is a genuinely hard task that landed, not an ill-posed one. Terminal status is the discriminator (cf. reify 3378, cancelled AND unactionable, which is excluded). |
| 3228 | reify | 580 | done | include | referenced | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 3378 | reify | 336 | cancelled | exclude | — | EXCLUDE. Cancelled, and abandonment was NOT benign: the architect reported the task unactionable 6 separate times (esc-3378-91/92/125/126/129/130), each naming the same cause — the required signature `fn solve_elastic_static(body, material, loads, supports, options)` references stdlib types (Body, Load, Support) and fn-param-default syntax that do not exist on main, with no sibling task creating them. The task is ill-posed AT ITS OWN BASELINE, so an exhaustion here measures the spec, not the model — precisely the confound this curation removes. |
| 3443 | reify | 268 | done | include | referenced | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 3586 | reify | 2489 | cancelled | include | planrate_only | INCLUDE. Cancelled, but abandonment WAS benign with respect to well-posedness: the only escalations on this task are esc-3586-38/40, both "Planning failed: agent failed: subtype='error_max_budget_usd'" — budget exhaustion during planning, with no unactionability claim anywhere in its history. The 2489-char brief names concrete file:line deliverables (BRepKind::Vertex at geometry.rs:54-71, OcctKernel::extract_vertices mirroring handle.rs:293-318, per-op populators in two named crates). It was abandoned for COST, not for being ill-posed, which is exactly the hard-task signal the pool measures. |
| 3779 | reify | 1060 | done | include | referenced | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 3822 | reify | 1771 | done | include | planrate_only | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 3834 | reify | 3862 | done | include | referenced | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 3845 | reify | 2201 | done | include | planrate_only | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 3883 | reify | 2631 | done | include | planrate_only | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 4026 | reify | 1577 | done | include | planrate_only | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 4086 | reify | 1298 | done | include | planrate_only | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 4370 | reify | 9422 | done | include | referenced | INCLUDE. The brief states an implementable goal, and although the architect raised unactionable escalation(s) on this task it reached status `done` — the unactionability was resolved, so this is a genuinely hard task that landed, not an ill-posed one. Terminal status is the discriminator (cf. reify 3378, cancelled AND unactionable, which is excluded). |
| 4832 | reify | 2627 | done | include | referenced | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 5208 | reify | 538 | pending | exclude | — | EXCLUDE. Status `pending`: never landed, so there is no reference diff and no ground truth that the task is completable at all. The brief is a SYMPTOM REPORT — "curated 3-arg fillet fails through the production .ri pipeline (CLI errors loudly, GUI fails silently) ... classic phantom-capability" — which states an observed failure, not an implementable goal: the deliverable is a root-cause diagnosis that did not exist at filing time. Architect exhaustion on an open-ended investigation is not the plan-quality signal the pool measures. |
| 1229 | dark_factory | 231 | done | include | referenced | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 2169 | dark_factory | 2299 | done | include | referenced | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 2260 | dark_factory | 1866 | done | include | referenced | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 882 | dark_factory | 1713 | done | include | planrate_only | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |
| 543 | know_live | 534 | done | include | referenced | INCLUDE. The brief states an implementable goal: it names the deliverable and enough of its surface for an architect to locate the work at the baseline commit. |

## Ceilings

- `max_architect_turns`: **120** — know-live production dark-factory-orchestrator.yaml carries max_turns: {architect: 120}; the census keys on exhaustion at 121 turns = that ceiling + 1. The pool must run at the same ceiling or it cannot reproduce the failures it was selected for.
- `timeout_minutes`: **180**

### Timeout derivation

- **Source**: events.duration_ms in <project_root>/data/orchestrator/runs.db, over the census population itself (the same predicate as census filter_sql)
- **Substitution**: The PRD names "v1 wall-clock dumps" as the derivation source, but data/eval-campaign/fable-architect-only-results.json carries no duration field at all — its 72 records hold only task_id / config_name / outcome / trial / plan_quality / plan_steps / cost_usd. Deriving from it is impossible, so runs.db events.duration_ms is the substituted (and richer) source. The substitution is recorded here so the threshold is not mistaken for a guess.
- **Measured** (n=46): max 28.1 min, p95 22.1 min, p50 14.9 min.
- **All architect invocations**: max 106.3 min. all_architect_max_minutes is the max over EVERY architect invocation_end in the three checkouts (n=14701), not just the census population — the strictest bound available.
- **Headroom**: 180 min is 6.4x the observed max-at-exhaustion (28.1 min, n=46) and 1.7x the all-time architect max (106.3 min), so the timeout provably cannot bind before the 120-turn or budget ceiling.
- **Why it matters**: runner.py raises the eval to outcome="timeout" on asyncio.TimeoutError and deliberately does NOT taint-exclude it, so a binding timeout scores a kept 0.0 — manufacturing an artificial failure on a set already selected for being hard.

## Merge-SHA availability — the SPLIT majority

SPLIT / direct-landed candidates are a MAJORITY (22/41), not the minority the trial manifest assumed. Only 17 of 41 have a single clean "Merge task/<id> into main" commit, so only those can carry a reference block. This caps the downstream plan_quality population and is a material fact for γ1, though it does not block β1 — D9's planRate-only mechanism handles it.

- `referenced` (single clean merge commit): **17**
- `planrate_only` (SPLIT / direct-landed): **22**

A `planrate_only` fixture carries NO `reference` key at all and instead stamps `provenance.reference_unavailable` with the cause plus `provenance.baseline_source` with the ladder rung that produced its `pre_task_commit`. An empty `reference: {}` block would be indistinguishable from a capture that silently failed; omitting the key and recording why makes it a positive, auditable fact.

## Continuity back-fill

These three fixtures are back-filled from the standing corpus so the v2 campaign shares part of its population with v1 and the two are comparable rather than merely adjacent. The v1 trial could grade plan_quality against a valid reference on only one fixture; re-banding these under references captured from their own committed pre/post SHAs closes that n=1 confound. Each record is built from the canonical fixture under evals/tasks/ — same pre_task_commit, same post_task_commit, same task_definition, same verify_commands — and the equality is asserted by test, so "reference by copy, do not duplicate content divergently" is machine-checked rather than conventional. Capturing the reference here (rather than copying a post-iota-2 fixture) is the same capture_reference call iota-2 makes for the standing corpus, so beta-1 is self-contained and creates no cross-task coupling.

| fixture | source | why |
|---|---|---|
| `reify_task_12` | `orchestrator/src/orchestrator/evals/tasks/reify_task_12.json` | The v1 trial graded plan_quality against a valid reference on exactly ONE fixture, so its plan-quality signal rested on n=1. Re-banding this fixture into the v2 cohort under a reference captured from its own committed pre/post SHAs is what closes that confound. |
| `reify_task_27` | `orchestrator/src/orchestrator/evals/tasks/reify_task_27.json` | Second reify continuity anchor: a high-complexity task from the same repo as the v1 n=1 fixture, so a v1-to-v2 delta on reify is not read off a single record. |
| `df_task_18` | `orchestrator/src/orchestrator/evals/tasks/df_task_18.json` | The dark-factory continuity anchor, so the overlap with v1 spans both repos rather than reify alone. |

These carry `provenance.baseline_source: standing_fixture_inherited` rather than a ladder rung — their baseline is inherited from the canonical fixture, not resolved here, and `df_task_18`'s `pre_task_commit` is not its post commit's first parent, so claiming `merge_first_parent` would be a false provenance. They are the ONLY fixtures whose ids overlap the standing corpus, and that overlap is pinned by test.

