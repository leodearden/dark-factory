# E1 retrieval-health — briefing re-key run (2026-09-30)

Frozen provenance for one live, read-only run of the E1 retrieval-health probe,
taken by task 4856 right after the topic registry was re-keyed. The collapsed
`g7-design-invariants` topic is retired. Each briefing query template in
`shared/src/shared/briefing_queries.py::QUERY_SPECS` is now its own topic
(`docs/prds/memory-briefing-and-fusion.md` D9, task 3660). The report also
carries the new tripwire split by census canonical presence (task 4299's
refinement). Like `plans/memory-eval-e1-first-live-run/`, this directory is
itself a valid artifact root.

| | |
|---|---|
| eval_id | `e1-retrieval-health` |
| run_stamp | `20260930T112138Z` |
| project scope | `dark_factory+reify` (the probe's default `--project-id` set) |
| registry | `fused-memory/tests/fixtures/memory_eval_topic_registry.json` as of task 4856 (34 topics) |
| census joined | `plans/memory-metadata-census-report.json` as committed (schema_version 4), read-only |
| mode | read-only: the probe searches and counts, it never writes to memory |
| files | `e1-retrieval-health/metrics-20260930T112138Z.json`, `e1-retrieval-health/report-20260930T112138Z.txt` |

Produced from the task worktree root, with the service env sourced the way
`scripts/fused-memory-flag-marker-sweep.sh` sources it, and every probe flag
at its default except `--out-root`:

```bash
OUT=/tmp/e1-rekey-4856-<UTC stamp>
( set -a; . /home/leo/src/dark-factory/.env; set +a
  export CONFIG_PATH=/home/leo/src/dark-factory/fused-memory/config/config.yaml \
         PROJECT_ROOT=/home/leo/src/dark-factory FALKORDB_URI=redis://localhost:6379
  uv run --project fused-memory python fused-memory/scripts/memory_eval_retrieval_probe.py --out-root "$OUT" )
```

The run wrote to an empty scratch root, so the report carries the initial-state
section. That is correct for the three briefing topics, which had never been
measured. The census path printed in the report is the worktree copy the run
read.

Task 4856 made two earlier runs (stamps `20260930T104959Z` and
`20260930T110107Z`). Each preceded a report-only renderer change: first naming
the passing items in the census split, then ranking failing briefing-query
phrasings. They were not kept. All six metric values were identical across the
three runs. The files here come from the committed code.

Contract-tested by `fused-memory/tests/test_memory_eval_e1_briefing_rekey_run.py`.

## Why it lives in `plans/`, and what it is not

It lives here for the reasons `plans/memory-eval-e1-first-live-run/README.md`
gives. The probe's `DEFAULT_OUT_ROOT` is gitignored, so a run worth keeping is
copied out. A committed artifact inside that root would make `is_initial_run()`
report every fresh clone as already run. The contract test asserts this root is
disjoint from it.

**This is not task 3211's baseline.** 3211's grandfather snapshot is 3211's own
first run against the live artifact root. This run never touched that root.
Treat it as a reference for operators.

## Per-briefing-topic outcome

Read off the report's metric table, the `query-surface` class of the "tripwire
split by census canonical presence" section, and "claims not recalled":

| topic | tripwire item | tuned phrasings | held-out | claim query |
|---|---|---|---|---|
| `briefing-task-semantic` (unscoped) | **passes** (named under `passed:`) | in the top 5 | in the top 5 | recalled (not listed as not recalled) |
| `briefing-conventions-area` (Mem0-scoped) | fails | one at rank 9 by content hash, one not in the top 10 | not in the top 10 | recalled |
| `briefing-conventions-generic` (Mem0-scoped) | fails | not in the top 10 | neither held-out in the top 10 | recalled |

Every phrasing of the two scoped topics was served by `mem0` only, which is the
declared `search_scope` doing its job. Task 4856 also retrieved both failing
canonicals by `last_known_id` from Qdrant, read-only, on 2026-09-30. Both
exist, and both are `procedural_knowledge`, inside the declared scope.

Hypothesis: the conventions channel's queries are generic enough that the
thousands of Mem0 conventions records outrank any single adjudicated canonical.
That would be a ranking fact about the briefing's own queries, the same failure
mode as the census-present topics below.

## The ranking-hypothesis verdict (esc-3208-1)

Source: the `canonical-present` class of the census split. It holds 19 failing
topics and 1 passing topic, and 57 per-phrasing lines.

Task 4856 added one read-only measurement that is not in the artifact. On
2026-09-30 it scrolled and retrieved from Qdrant `fused_dark_factory` and
`fused_reify`. For each of the 19 failing topics, it checked whether the
registry's canonical `content_hash` and `last_known_id` still name the record
carrying `topic=T, canonical: true`:

- **13 topics: both keys name the live canonical, so the fixture is intact.**
  They are architect-plan-files-write-set,
  background-task-reaping-and-wake-mechanisms,
  boolean-operand-consumption-edges, cargo-fmt-no-gate,
  cargo-rerun-if-changed-warm-lane-mtime,
  compiled-module-functions-user-source-only,
  eval-worktree-plan-tools-missing, nextest-less-host-simulation,
  recon-project-root-misroute-retired, reify-audit-ptodo-from-worktree,
  reify-debug-mcp-driving, task-5422-fabrication-episode-disposition and
  warm-lane-plan-json-loss-reconstruction.
- **3 topics: `last_known_id` names the live canonical but its content hash
  changed.** They are pytest-xdist-serial-override,
  reify-debug-mcp-tool-addition and tracing-interest-cache-poisoning.
- **3 topics: neither key names the live canonical.** For docs-prd-landing and
  harness-layout-gate-decision-rule, the `last_known_id` record no longer
  exists. For pkill-pgrep-self-match, it exists but is no longer
  `canonical: true`.

The 13 intact topics have 39 phrasing lines. In 5, the canonical was in the
top 5, matched by content hash. In the other 34 it was not in the top 10. 32 of
those 34 had `mem0` among the serving stores, and 2 were served by `graphiti`
only.

- **(i) Corpus-wide ranking: CONFIRMED as the dominant failure.** In 32 of the
  39 intact-fixture phrasings, four things hold at once. The census counts the
  canonical. The registry key names it. Mem0 was consulted. And the canonical
  is still not in the top 10.
- **(ii) Routing: minor.** Across all 19 topics, 3 of 57 phrasings were served
  by `graphiti` only: 2 intact, 1 hash-drifted.
- **(iii) Fixture decay: 6 of 19 topics.** These are the 3 hash-drifted and the
  3 neither-key topics. Their tripwire items do not measure retrieval until the
  registry is re-keyed, which is a registry-side repair.

**Singleton clusters.** All 13 intact-fixture failing topics have census
`records: 1`: the canonical is the only record under its topic. So does the one
passing topic, inv-geo-1-mesh-contract-mock. The singletons therefore show
(i), corpus-wide ranking. They have no sibling to outrank the canonical, so
intra-cluster ordering cannot be the cause. This answers the question memory
9c859fcd left unverified. Every multi-record census-present topic falls in the
decay group: docs-prd-landing (2), harness-layout-gate-decision-rule (30),
pkill-pgrep-self-match (10) and pytest-xdist-serial-override (20).

Hypothesis: for the singletons, reordering within a topic cannot lift the
canonical into the top K until something recalls it into the fetched list in
the first place. Candidates are a promoting pin, a grouped read or a diversity
cap (task 3111's options).

## Unpopulated topics: remedy decision

The census split's `absent-unpopulated` class names
architect-report-task-already-done-main-reachability and
eval-worktree-venv-shadowing. The committed census is v4, so their variant
counts read "not measured by this census". Task 4856 ran a read-only Qdrant
count on 2026-09-30 in `fused_dark_factory`:

- 0 records sit under either hyphenated registry slug.
- 8 sit under `architect_report_task_already_done_main_reachability`.
- 2 sit under `eval_worktree_venv_shadowing`.

**Decision:** these are legacy snake_case spellings. Their remedy is
`fused-memory/scripts/normalize_topic_slugs.py --apply` (task 4878), which
rewrites the spelling. Retro-stamping is not the remedy. These records already
carry a topic, just misspelled. And `fused-memory/scripts/retro_stamp_topics.py`
addresses records by id from its own enumerated sources (curator-gate clusters,
the `canonical: true` scroll and the calibration fixture), while both topics
are `topic_guard_cluster` entries. Running the normalization is an operator
write, and task 4856 deliberately did not perform it. It was raised as an
`escalate_info` instead. A zero-record topic with no variant spelling would
need a registry-side decision instead; none exists today.

## Stamping worklist

The split's `absent-with-records` class holds the topics whose records carry
the topic but none is canonical. Once the nightly census is schema v5, that is
`registry_coverage.stamping_worklist` in `plans/memory-metadata-census-report.json`
(see `fused-memory/scripts/census_memory_metadata.py::_build_registry_coverage`).
Read the count there; it is not restated here.

## Do not hand-edit

Both files are verbatim copies of what the probe wrote.
`test_memory_eval_e1_briefing_rekey_run.py` asserts three things:
`serialize_metric_series` re-emits the metrics file byte for byte, exactly one
run lives here, and the tripwire adjudicates every briefing template
individually with no `t-g7-design-invariants`. The run cannot be reproduced,
because the corpus moves.
