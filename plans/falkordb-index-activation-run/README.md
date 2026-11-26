# FalkorDB index provisioning ζ — activation run (2026-10-08)

Frozen provenance for the integration gate of
`docs/prds/falkordb-index-provisioning.md` (task 3711, PRD task ζ). It holds one
live, read-only run of `fused-memory/scripts/falkordb_index_activation_probe.py`
and a verbatim copy of one E1 retrieval-health probe run taken after it.

The briefing probe was re-baselined per Leo's ruling on esc-3711-5 (option A,
2026-10-08): ids 877 and 3600 were replaced, and the 4/5 floor was kept.

Contract-tested by `fused-memory/tests/test_falkordb_index_activation_run.py`.
That test recomputes every verdict from the raw rows recorded in the JSON, using
the probe script's own pure functions. It never reads a recorded boolean.

## Provenance

| | |
|---|---|
| activation record | `activation-20261008T191128Z.json` (`measured_at` 2026-10-08T19:11:28Z) |
| E1 run | `e1-retrieval-health/metrics-20261008T192015Z.json`, `e1-retrieval-health/report-20261008T192015Z.txt` (run 19:16:32Z–19:20:20Z) |
| task branch | base `3cbd626171`; probe script and tests at `e8c96609c8` when the run was taken |
| main when measured | `6551b75aba` |
| merges on main | 3708 (γ) `13a9caaca2` (2026-10-02T11:42:23Z), 3709 (δ) `72ccec46a1`, 3710 (ε) `d7e9c92abe`, 6238 (BM25 join fix) `566deee823` |
| `fused-memory.service` ActiveEnterTimestamp | Thu 2026-10-08 19:20:03 BST, after all four merges |
| registry | the unit's `DASHBOARD_KNOWN_PROJECT_ROOTS`: 10 ids, `scoping_intake` included |
| mode | read-only: FalkorDB through `GRAPH.RO_QUERY` only, the briefing probe through the live MCP `search` tool |

The activation command, run from the task worktree root:

```bash
( set -a; . /home/leo/src/dark-factory/.env; set +a
  export CONFIG_PATH=/home/leo/src/dark-factory/fused-memory/config/config.yaml \
         PROJECT_ROOT=/home/leo/src/dark-factory FALKORDB_URI=redis://localhost:6379
  export DASHBOARD_KNOWN_PROJECT_ROOTS="$(systemctl --user show fused-memory.service -p Environment --value \
    | tr ' ' '\n' | sed -n 's/^DASHBOARD_KNOWN_PROJECT_ROOTS=//p')"
  uv run --project fused-memory python fused-memory/scripts/falkordb_index_activation_probe.py \
    --out-dir plans/falkordb-index-activation-run )
```

The registry variable is exported from the unit because it is not in `.env`.
With it unset, the registry narrows to `dark_factory` alone, so the script
refuses to run (`falkordb_index_activation_probe.py::require_known_project_roots`).

The E1 command is 4856's documented command plus two scratch data dirs:

```bash
SCRATCH=$(mktemp -d /tmp/e1-3711-XXXX); OUT=$SCRATCH/out
( set -a; . /home/leo/src/dark-factory/.env; set +a
  export CONFIG_PATH=/home/leo/src/dark-factory/fused-memory/config/config.yaml \
         PROJECT_ROOT=/home/leo/src/dark-factory FALKORDB_URI=redis://localhost:6379 \
         QUEUE_DATA_DIR=$SCRATCH/queue RECONCILIATION_DATA_DIR=$SCRATCH/reconciliation
  uv run --project fused-memory python fused-memory/scripts/memory_eval_retrieval_probe.py --out-root "$OUT" )
```

## Index half (primary)

`CALL db.indexes()` was read on every registered graph and projected through
`fused_memory/backends/falkor_indices.py::normalize_index_records`. It was then
diffed against `falkor_indices.py::expected_index_set` (38 specs) with
`fused_memory/reconciliation/index_health.py::summarize_index_health`.

| graph | graph key | actual / expected | missing | unexpected | not OPERATIONAL |
|---|---|---|---|---|---|
| `autopilot_video` | present | 38 / 38 | 0 | 0 | 0 |
| `autotrade` | **absent** | — | — | — | — |
| `dark_factory` | present | 38 / 38 | 0 | 0 | 0 |
| `know_live` | present | 38 / 38 | 0 | 0 | 0 |
| `mission_control` | **absent** | — | — | — | — |
| `pump_web_ui` | present | 38 / 38 | 0 | 0 | 0 |
| `reify` | present | 38 / 38 | 0 | 0 | 0 |
| `scoping_intake` | present | 38 / 38 | 0 | 0 | 0 |
| `solar_challenge` | present | 38 / 38 | 0 | 0 | 0 |
| `solar_challenge_platform` | present | 38 / 38 | 0 | 0 | 0 |

Every registered graph that exists carries the full expected set, all
OPERATIONAL. `autotrade` and `mission_control` have no graph key yet, which the
task allows.

## The briefing probe

The probe fires the retired briefing template `task {id} context and related
decisions` at limit 5, scoped to Graphiti, against `dark_factory`. It is a
fixed instrument repeated from the PRD session, not today's briefing query.
A query counts as a hit when at least one Graphiti result mentions the queried
id **as a task** (`falkordb_index_activation_probe.py::mentions_task`, for
example "Task 3127" or "tasks 2293 and 2286"). One predicate decides both
replacement eligibility and hits.

### The original 3/5, and the drift

On 2026-10-07 (esc-3711-4), measured Graphiti-scoped, the original ids scored
**3/5**: hits for 3127, 2286 and 1157; misses for 877 and 3600. Leo's ruling
recorded why the two misses are corpus drift unrelated to the indices:

- **877**: 6 of its 7 fulltext matches were invalidated or expired.
- **"3600"**: 8 of its 9 matches used the number as seconds or a ceiling.

This run's fresh fulltext rows for the original ids, as (rows, live, live and
genuine), are in `original_id_counts`:

| id | rows | live | live + genuine |
|---|---|---|---|
| 877 | 9 | 3 | 1 |
| 2286 | 6 | 4 | 4 |
| 3127 | 7 | 7 | 6 |
| 3600 | 10 | 5 | 1 |
| 1157 | 12 | 6 | 6 |

877 and 3600 still have one genuine live edge each. The original-ids probe in
this run again scored **3/5**, missing 877 and 3600. It is recorded and not
asserted.

### Replacement rule and counts

The rule was declared before the run
(`falkordb_index_activation_probe.py::select_replacements`). From each drifted
id, walk outward: distance 1 below, then 1 above, then 2 below, and so on. Skip
the five original ids and any replacement already chosen. Pick the FIRST id
whose fresh `dark_factory` fulltext rows hold at least 2 live edges that
mention it as a task. Fail loudly beyond distance 50. Each candidate is chosen
from fulltext counts alone, never from search output.

Every examined candidate, as (rows, live, live and genuine):

| anchor | candidate | rows | live | live + genuine |
|---|---|---|---|---|
| 877 | **876** | 8 | 2 | **2** |
| 3600 | 3599 | 0 | 0 | 0 |
| 3600 | 3601 | 0 | 0 | 0 |
| 3600 | 3598 | 0 | 0 | 0 |
| 3600 | 3602 | 0 | 0 | 0 |
| 3600 | 3597 | 2 | 0 | 0 |
| 3600 | 3603 | 0 | 0 | 0 |
| 3600 | 3596 | 1 | 1 | 1 |
| 3600 | 3604 | 1 | 0 | 0 |
| 3600 | 3595 | 0 | 0 | 0 |
| 3600 | 3605 | 0 | 0 | 0 |
| 3600 | 3594 | 0 | 0 | 0 |
| 3600 | 3606 | 0 | 0 | 0 |
| 3600 | 3593 | 0 | 0 | 0 |
| 3600 | 3607 | 0 | 0 | 0 |
| 3600 | 3592 | 1 | 0 | 0 |
| 3600 | 3608 | 0 | 0 | 0 |
| 3600 | **3591** | 5 | 4 | **4** |

**Chosen: 877 → 876 (distance 1), 3600 → 3591 (distance 9).** The raw rows of
every candidate are in the JSON under `replacements[].examined`. The contract
test re-runs the rule over them and reproduces both choices.

**Caveat.** 876 and 3591 have **no pre-index baseline**. For them the check is
an absolute bound, not a before/after delta. That is defensible because the
original baseline was degenerate: on 2026-08-05 the five queries drew their 24
result slots from 10 distinct edges out of 14,041, and scored 0/24.

### Result

| probe | stores | ids | hits | asserted |
|---|---|---|---|---|
| re-baselined | `graphiti` | 876, 2286, 3127, 3591, 1157 | **4/5** (miss: 3591) | yes, floor 4 — **passes** |
| original ids | `graphiti` | 877, 2286, 3127, 3600, 1157 | 3/5 (miss: 877, 3600) | no |
| unscoped | auto-routed | 876, 2286, 3127, 3591, 1157 | 2/5 (hits: 2286, 1157) | no |

No query was degraded, and every query returned 5 results. The unscoped variant
is recorded only. Its result windows mix Graphiti and Mem0 results (3 + 2 in
every query), and 3658's cross-store merge changed how they are composed
independently of the indices, so it is not this PRD's variable.

## E1 retrieval health (secondary, recorded, not asserted)

**Pre-flight.** `memory_eval_retrieval_probe.py::_probe` calls
`MemoryService.initialize()`. That runs Graphiti startup maintenance (the
registered-graph index sweep and the identity scan, which repairs dup-uuid
edges) and opens a `DurableWriteQueue` (task 6239). The activation record's
`e1_preflight` shows both would be no-ops. Every present registered graph is
complete, and all 161 listed graphs have 0 dup-uuid edge groups:
`maintenance_noop: true`. The E1 run's own log agrees: `startup identity scan
complete: graphs_scanned=161 dup_name_groups=21 edges_repaired=0`, and no
index statement was issued. The duplicate-NAME groups it logs are reported
only, never repaired. `QUEUE_DATA_DIR` and `RECONCILIATION_DATA_DIR` pointed at
a fresh scratch dir. That stops a live queue from being recovered, and it
matches 4856's condition, a fresh, empty `./data/queue`. The log also shows
mem0's per-collection Qdrant payload-index `PUT`s (`Created index for user_id
in collection fused_dark_factory`). The live service logs the same lines on
its own start. They are not corpus writes.

**Before / after.** BEFORE is 4856's `plans/memory-eval-e1-briefing-rekey-run/`
(stamp `20260930T112138Z`). It replaces the "3660 artifact" the task text
names, because 3660 was cancelled as superseded by 4856. AFTER is this
directory's copy. Both use the probe's default `dark_factory+reify` scope and
the same 34-topic registry. No commit touched
`fused-memory/tests/fixtures/memory_eval_topic_registry.json`,
`shared/src/shared/briefing_queries.py` or the probe after 4856's run (its
last such commit, `cd971e2964`, predates that stamp). 4856's documented
command does not export `DASHBOARD_KNOWN_PROJECT_ROOTS`, and this E1 run did not
either, so both ran E1's startup sweep against the same narrowed registry.

| measure | before (4856) | after (this run) |
|---|---|---|
| `t-briefing-task-semantic` tripwire item | **passes** | **passes** |
| topic-canonical-present (failing) | 32 / 34 | 32 / 34 |
| canonical-in-top-5 | 19 / 102 (0.186) | 19 / 102 (0.186) |
| canonical-in-top-10 | 23 / 102 (0.225) | 21 / 102 (0.206) |
| canonical-in-top-5-held-out | 3 / 35 (0.086) | 3 / 35 (0.086) |
| claim-recall | 14 / 34 (0.412) | 14 / 34 (0.412) |
| contamination-share | 2 / 510 (0.004) | 2 / 510 (0.004) |
| phrasings served by graphiti / mem0 | 182 / 198 | 184 / 198 |

One input did differ between the runs: the joined census. Each run read its
own worktree's committed `plans/memory-metadata-census-report.json`. The newer
one names the three briefing topics as query surfaces and counts the two
unpopulated topics' variant spellings (8 and 2).

`briefing-task-semantic` was already passing before. Its tripwire item cannot
show an improvement. Neither report ranks a passing item's phrasings, because
the census split lists ranks for failing topics only. Neither report lists the
topic under "claims not recalled".

Observations:

- The E1 comparison shows no improvement for this topic.
- Corpus-wide, only canonical-in-top-10 moved, by −2/102. Both of the
  phrasings it lost sat at rank 9 before and are outside the top 10 now: one
  is pkill-pgrep-self-match ('why did pgrep match my own command line'), the
  other briefing-conventions-area ('conventions and gotchas for shared
  orchestrator briefing agents queries').
- Two other phrasings changed serving store from graphiti+mem0 to mem0 only,
  with the same not-in-top-10 outcome.

These were filed as `escalate_info` esc-3711-6, not as a task failure.

Hypothesis: the topic was at ceiling before the indices served, so this
binary item has no headroom to register a change.

## Case-fold tripwire (recorded, not asserted)

These are the 2026-09-26 amendment's two measurements, both read-only through
`GRAPH.RO_QUERY`.

1. **Population**: the task's verbatim query, `MATCH (n:Entity) WITH
   toLower(n.name) AS k, count(n) AS c, count(DISTINCT n.name) AS names WHERE
   c > 1 AND names > 1 RETURN count(k), sum(c)`.
2. **Splits since 3708**: `falkordb_index_activation_probe.py::CASE_FOLD_SPLIT`.
   It counts the case-fold keys of which at least two spellings each gained a
   `RELATES_TO` edge with `created_at > '2026-10-02T11:42:23+00:00'`, the
   committer time of 3708's merge `13a9caaca2`. It is edge-first because the
   endpoint-label form timed out at 120 s.

| graph | keys | nodes | split groups since 3708 |
|---|---|---|---|
| `autopilot_video` | 2 | 4 | 0 |
| `dark_factory` | 68 | 140 | 28 |
| `know_live` | 3 | 6 | 0 |
| `pump_web_ui` | 0 | 0 | 0 |
| `reify` | 96 | 194 | 21 |
| `scoping_intake` | 3 | 6 | 1 |
| `solar_challenge` | 32 | 64 | 17 |
| `solar_challenge_platform` | 11 | 23 | 5 |

The `dark_factory` baseline was 70 keys / 145 nodes on 2026-09-26. Today it is
68 / 140, which does not exceed the "~20 keys over baseline" arm. Its 28 split
groups are generic case variants: `bash`, `clear`, `cwd`, `datum`,
`escalations`, `flag`, `gate`, `graphiti`, `green`, `harness`, `info`, `judge`,
`leo`, `owner`, `path`, `pid`, `react`, `redis`, `repo`, `root`, `scheduler`,
`sha`, `spec`, `stop`, `task`, `test`, `yaml` and `γ3`. The spellings are in
the JSON.

**Observation:** the trigger's split arm holds. 28 `dark_factory` variant
groups have both spellings gaining edges since 3708's merge, which is not
trivial. This was raised to the operator as `escalate_info` esc-3711-7. No
sitting was filed: the amendment reserves that for the operator.

## Follow-ups

- `tkt_0RVJFNZTMNA6N1HE3CR9VPZ2M9`: 877's live genuine edge is not ranked in
  the top 5. The architect filed it per ruling item 4. This run's original-ids
  probe misses 877 again.
- Task 6239: the E1 instrument is not read-only. Its fix is not in scope here.
  This run proved its side effects would be no-ops (pre-flight above) and kept
  its queue and reconciliation state in scratch dirs, rather than changing the
  instrument. The PRD says ζ makes no change to the instrument.

## Do not hand-edit

`activation-*.json` is exactly what the probe script wrote. The two files under
`e1-retrieval-health/` are byte-for-byte copies of what the E1 probe wrote. The
contract test recomputes every verdict from the JSON's raw rows, and asserts
that `serialize_metric_series` re-emits the E1 metrics byte for byte. The runs
cannot be reproduced, because the corpus moves.
