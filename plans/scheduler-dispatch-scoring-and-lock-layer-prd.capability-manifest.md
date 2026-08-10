# Capability manifest — Scheduler dispatch scoring + lock layer

PRD: `plans/scheduler-dispatch-scoring-and-lock-layer-prd.md` (commit `e1c51efa8d`)
Machine-readable twin: `plans/scheduler-dispatch-scoring-and-lock-layer-prd.capability-manifest.yaml`
Built at decompose, 2026-08-06. Every binding below was checked against `main` in this
session — not copied from the PRD's §8 table.

Mechanizes G3 + G6 per task: each capability a task's user-observable signal asserts is
bound to evidence (`grep:<file>:<line> wired` / `producer:<label> upstream` / `floor:` /
`rejection-check:`). A binding resolving to `declared-only | test-only | producer-absent |
producer-downstream | producer-extent-short | bound≤floor | rejection-absent` blocks the
batch until resolved.

## Resolutions applied at decompose (three bindings did not clear as authored)

**R1 — β's literal baseline numbers are not reproducible from the committed CSV.**
The PRD's §10 β row asserts "baseline 46-distinct / 117-tie / 249-masked reproduced from
`plans/evidence/.../candidates_scored.csv`". Recomputed from that file this session over all
294 rows: **47 distinct scores, max tie group 117 at score 2999.0, 251 capped**. The 117-way
tie reproduces *exactly*; 46 and 249 come from a 291-row scoping that the CSV does not
record, so an implementer baking `46` and `249` into a RED test could not turn it green.
**Resolved (G6 (b), bound relaxed to what is achievable):** β's test *derives* the baseline
metrics from the CSV at test time and asserts new-vs-baseline improvement; the pinned
literals are the 117-way max tie group and the fixture-shape floors (≥250 candidates, ≥3
tiers, ≥20 with D≥1 — all satisfied by the committed CSV: 294 rows, 4 tiers, 51 with D≥1).

**R2 — δ's pin-source signal had no producer (`producer-absent`).**
δ's signal asserts "a `source='starvation_watchdog'` pin observable in `get_pin_queue`".
`get_pin_queue` (`fused-memory/src/fused_memory/server/tools.py:6618`) SELECTs a **named**
column list — `task_id, boost_tier, pinned, pin_order, reserve_now, ttl_until` — so the new
`overrides.source` column would be invisible to the only read path that surfaces pins to an
operator. No task in δ's dependency closure or module set produced it.
**Resolved (G6 (a) + G1):** `fused-memory/src/fused_memory/server/tools.py` added to δ's
module set; δ extends the SELECT and the returned dict with `source`. This is also what G1
requires — without it the new column is a producer-orphan with no named consumer.

**R3 — κ's signal depends on a qualifying live candidate existing at an unknown runtime.**
κ asserts a "3534-class chain head (tier `high`, D ≥ 9, wait ≥ 48h) ranked inside
`max_concurrent_tasks`". Task 3534 is pinned at `pin_order 1` right now and is expected to
dispatch well before κ runs, and no mechanism guarantees another qualifying candidate is in
the live queue at that moment — the signal would be unobservable through no fault of the
implementation.
**Resolved (G6 (c), configuration named):** κ's script takes a candidate source, defaulting
to the live queue and falling back to β's committed regenerated fixture when no live
candidate meets the (high, D≥9, ≥48h) predicate; the script reports which source it used.
The assertion is then always observable and reproducible.

## G7 addition (resolved by redesign, no waivers)

**γ / INV-4 `storm-escape-required`.** The TTL-cached branch probe is a fail-soft path with
no storm escape as authored: if `git for-each-ref` fails on every refresh, `has_work_product`
returns False for every candidate and the continuity term silently goes to zero fleet-wide —
the same per-task-safe / bulk-catastrophic shape the PRD itself identified for the age term
and fixed with D12's arming gate. Resolved by requiring a consecutive-failure streak
escalation on the probe in γ (mirroring I-6's precedent), not waived. PRD §11's "no waivers"
claim survives.

---

## α — Durable `pending_since` anchor + back-fill

| Capability | Binding | Verdict |
|---|---|---|
| Single status-write chokepoint seeing `old_status` in-transaction | `grep:fused-memory/src/fused_memory/backends/sqlite_task_backend.py:1938` (`set_task_status`), `old_status` read at `:1993` — wired production path | PASS |
| Tier-A metadata key registration point | `grep:shared/src/shared/task_metadata.py:746` `_BLESSED_METADATA_KEYS` — 41 keys today; `pending_since` confirmed **absent**, so α genuinely adds it | PASS |
| `pending_since` / `pending_since_backfilled` keys | `producer:α` — self-delivered; verified absent from the tree this session (`git grep pending_since` = 0 hits) | PASS |
| `updated_at` available as the back-fill source | `grep:fused-memory/src/fused_memory/backends/sqlite_task_backend.py:91-106` — `tasks` has `metadata TEXT` + `updated_at`, no `created_at` (so D4's rejected id-synthesis alternative is genuinely impossible) | PASS |
| Anchor observable through the product's own read path | `get_task` returns `metadata` — the G2-sanctioned read path, not storage peeking | PASS |

## β — Scoring restructure + non-saturation known-answer test

| Capability | Binding | Verdict |
|---|---|---|
| `_compute_score` / `_compute_age` / sort key are the sole ordering path | `grep:orchestrator/src/orchestrator/scheduler.py:4739` (`_compute_score`), `:4695` (`_compute_age`), `:6349` (`scored.sort(key=lambda entry: (-entry[0], entry[1]))`) — all three confirmed wired | PASS |
| `TIER_WIDTH` / `age_alpha` / `cpm_beta` config fields | `grep:orchestrator/src/orchestrator/config.py:52` (`TIER_WIDTH: int = 1000`), `:4004` (`age_alpha`), `:4014` (`cpm_beta`) | PASS |
| `pending_since` readable at score time | `producer:α` — **upstream** (β depends α); DAG-direction verified | PASS |
| I-1 budget sum below tier width | `floor: CPM_BUDGET+AGE_BUDGET+CONTINUITY_CREDIT = 920 ≤ TIER_WIDTH−1 = 998` — headroom 78, stated | PASS |
| I-3 continuity subordination | `floor: CONTINUITY_CREDIT 120 < 300·log1p(4)/log1p(32) = 138.1` — margin 18.1, stated | PASS |
| Baseline metrics reproducible from committed evidence | `grep:plans/evidence/scheduler-scoring-2026-08-06/candidates_scored.csv` — 294 rows, columns `id,tier,age,transitive_count,bonus,capped,score,rank`; recomputed 47 distinct / 117 max tie / 251 capped. **See R1** — literals relaxed to the reproducible ones | PASS (relaxed) |
| Fixture-shape floors (≥250 candidates, ≥3 tiers, ≥20 with D≥1) | `floor:` measured on the committed CSV — 294 rows, 4 tiers (critical 1 / high 56 / medium 90 / low 147), 51 rows with D≥1. Every floor cleared with margin | PASS |
| "distinct scores ≥ 0.90·n" achievable | achievability basis: `age(t) = AGE_BUDGET·a/(a+AGE_HALF_SECS)` is strictly monotone and injective in `a`, and back-filled anchors derive from `updated_at`, which carries microsecond precision per row — so distinct anchors give distinct scores by construction. The known collision source is a `commit_planning` batch stamping one identical timestamp (C1, deliberate); the three same-second bursts D2 names total 19 tasks in a 294-candidate pool, well inside the 29-collision budget | PASS |
| `RELOADABLE_FIELDS` submodel-group precedent | `grep:orchestrator/src/orchestrator/config.py:719,843,943` — existing whole-submodel groups (`session_resume` named at `:843`); confirmed no scoring key present today, so D8's premise holds | PASS |

## γ — Continuity credit via work-product detection

| Capability | Binding | Verdict |
|---|---|---|
| `git for-each-ref %(ahead-behind:main)` | `rejection-check:` run **live** this session against this repo on git 2.43.0 — emitted `task/1074 2 20457`, `task/1139 1 19857`, … Real output, not a docs claim | PASS |
| `EventType.zero_progress_requeue` | `grep:orchestrator/src/orchestrator/event_store.py:221` | PASS |
| Durable cross-run event query | `grep:orchestrator/src/orchestrator/event_store.py:697` `fetch_events_by_type_all_runs` | PASS |
| `CONTINUITY_CREDIT` knob | `producer:β` — **upstream** (γ depends β) | PASS |
| "scores exactly `CONTINUITY_CREDIT` higher" (exactness) | identity: continuity is a purely additive constant term; with tier, D, anchor and `now` held fixed the score difference is exactly the credit. Earning configuration: frozen clock + identical anchors in the test | PASS |
| Bounded subprocess occupancy | asserted as a count of git invocations per TTL window — measurable in-test; `for-each-ref` covers every branch in one process (verified above) | PASS |
| Probe-failure storm escape | `producer:γ` — **added at decompose**, see the G7 note | PASS |

## δ — Starvation watchdog repair: durable clock, park-blocked discriminator, auto-pin, emitted event

**AMENDMENT 2026-08-07 (post-csv-outage).** The last two rows were added after the
2026-08-06/07 csv main-red incident, in which the task that would have healed main sat
PARKED 16h behind a single held module and the auto-pin remedy this section certifies was
applied **twice, by two independent actors, to no effect** — priority ordering selects among
*dispatchable* candidates and does not evict a lock holder. C4 as originally written
classifies that task identically to a never-top-scored one and prescribes the inert remedy,
which converts an unhandled condition into an apparently-handled one. Both new capabilities
were confirmed `producer-absent` at amendment time (`grep park_blocked` and
`grep blocking_holder` across `orchestrator/src/` and `fused-memory/src/` each return 0),
so both bind to real gates rather than to already-satisfied greps.

| Capability | Binding | Verdict |
|---|---|---|
| Scheduler can emit events | `grep:orchestrator/src/orchestrator/scheduler.py` — ~20 `self.event_store.emit(` sites | PASS |
| `EventType.starvation_detected` | `producer:δ` — self-delivered; confirmed absent (`git grep starvation` in `event_store.py` hits only a CPU-flake comment at `:103`) | PASS |
| Override store pin + order + TTL | `grep:orchestrator/src/orchestrator/overrides.py:171` `set_override(..., pinned=, pin_order=, ttl_until=)` — keyword-only, COALESCE upsert | PASS |
| `overrides.source` column | `producer:δ` — self-delivered; schema at `overrides.py:38-49` confirmed to have **no** `source` column | PASS |
| Pin source visible in `get_pin_queue` | `producer:δ` — **added at decompose, see R2.** `tools.py:6618` SELECTs a named column list that excludes `source`; without this extension the capability was `producer-absent` | PASS (resolved) |
| `pending_since` for the durable idle clock | `producer:α` — **upstream** (δ depends α) | PASS |
| Auto-pin cap / TTL bounds | `floor:` `auto_pin_max_concurrent` default 3 against a live pin queue already holding 12 entries, 4 operator-owned — the cap cannot silently consume operator pins because `source` distinguishes them | PASS |
| Watchdog is genuinely dead today (the premise being fixed) | `grep:orchestrator/src/orchestrator/scheduler.py:3433` `_apply_starvation_watchdog`, anchor wipe at `:3480-3485`; `_bump_skip_and_maybe_park` at `:4506` called **only** at `:6425`/`:6436` for `top_id` — so a never-rank-1 task provably accrues zero skips | PASS |
| `gate='park_blocked'` discriminator | `producer:δ` — **added by amendment**; confirmed absent (`grep park_blocked` = 0). Inputs already exist: `ModuleLockTable.snapshot_parks()` + the `_held` map, joined by `shared.locking.modules_conflict` (the canonical prefix rule — INV-5, do not re-implement) | PASS |
| `starvation_detected` names its blocker | `producer:δ` — **added by amendment**; confirmed absent (`grep blocking_holder` = 0). INV-2: without `blocking_holder`/`blocking_module`/`park_age_secs` a reader must hand-join `parks[tid].modules` against `current_holders`, which is precisely the manual derivation performed during the 17h outage to establish that nine of ten modules were free and exactly one was held | PASS |

## ε — Truthful lock release

| Capability | Binding | Verdict |
|---|---|---|
| Exactly one bypassing release site | `grep:orchestrator/src/orchestrator/scheduler.py:7038` — enumerated `lock_table.release(` across `orchestrator/src/`: **exactly two** sites, `:7038` (the bypass) and `:7050` (inside `Scheduler.release`, the emitting path). PRD's "only such bypass" claim confirmed | PASS |
| The emitting path exists | `grep:orchestrator/src/orchestrator/scheduler.py:7041` `def release(self, task_id, *, requeued=False)`; `release_subset` at `:6953` already emits | PASS |
| "today it emits nothing" (negative assertion) | `rejection-check:` — read `:7036-7042` directly; the bypass is a bare call with no adjacent `emit`. Observed absence confirmed at the source, not inferred | PASS |
| Single-writer assertion is authorable | **scoping constraint recorded:** `lock_table.release(` also appears 6× in `orchestrator/tests/test_scheduler.py`, so the grep-anchored test must scope to `orchestrator/src/` and allow the one site inside `Scheduler.release` — otherwise the assertion fails against itself | PASS |

## ζ — Module hold-history predictor

| Capability | Binding | Verdict |
|---|---|---|
| `EventType.service_restart` for era-boundary span closing | `grep:orchestrator/src/orchestrator/event_store.py:399` | PASS |
| Seed source for hold history | `grep:orchestrator/src/orchestrator/event_store.py:697` `fetch_events_by_type_all_runs` | PASS |
| Truthful `lock_released` pairs | `producer:ε` — **upstream** (ζ depends ε), which is D11's whole point: a missing release reads as an infinite hold and would make η refuse every backfill | PASS |
| Predictor design basis (R², safety multipliers) | `grep:plans/evidence/scheduler-scoring-2026-08-06/PARKING_MODEL_REPORT.md:116-126` — log2-space R² table, module-history median **0.26 (DF) / 0.68 (reify)**, safety ×2.9/×2.0 at 80%. Verified present in the committed evidence | PASS |
| "reproduces the measured module medians" | **binding scoped honestly:** the test oracle is the median computed from ζ's own fixture event trace (which contains orphan releases + stuck-at-era-end spans), *not* a per-module median table — the evidence publishes R²/coverage, not per-module medians. The R² figures are the design basis, not the assertion | PASS |
| Empty-history refuses | refusal below `backfill_min_samples` is asserted directly — the predicate must **not** accept the empty case | PASS |

## η — EASY-backfill admission through parks

| Capability | Binding | Verdict |
|---|---|---|
| `predicted_hold` / `predicted_remaining` | `producer:ζ` — **upstream** (η depends ζ) | PASS |
| `try_acquire` under the lock as the acting basis | `grep:orchestrator/src/orchestrator/scheduler.py:6327` `_phase_select_scored` — scan-order acquisition confirmed; the free-module set read during scoring is never the acting basis (INV-3) | PASS |
| `park_backfill_granted` / `park_backfill_overstay` | `producer:η` — self-delivered new event types | PASS |
| `backfill_safety_factor` default 2.5 | `floor:` bracketed by the two measured 80%-coverage multipliers (DF ×2.9, reify ×2.0) at `PARKING_MODEL_REPORT.md:126`; overstay at ×2.5 measured 7–9% (`:254-255`), and the design emits that rate rather than assuming it | PASS |
| `backfill_max_park_age_secs` is load-bearing | `grep:plans/evidence/.../PARKING_MODEL_REPORT.md:256` — the model names the casualty (one reify starver flips to never-dispatched without the cutoff) | PASS |

## θ — Durable fairness state across restarts

| Capability | Binding | Verdict |
|---|---|---|
| Durable per-project orchestrator SQLite store to extend | `grep:orchestrator/src/orchestrator/run_store.py:70` — `scheduler_state` is the 4-column pause-only table (`project_id, pause_reason, pause_at, set_by_run_id`), so a **sibling** table is right and a migration would be wrong | PASS |
| No restore path exists today (the premise being fixed) | `grep:orchestrator/src/orchestrator/scheduler.py:6829` writes `scheduler_state.json`; the only reader is fused-memory's `read_scheduler_state` — confirmed absent restore | PASS |
| New snapshot keys reach `get_scheduler_state` | `producer:θ` + pass-through precedent: `read_scheduler_state()` returns the on-disk JSON verbatim via `json.loads`, so a new key surfaces without a fused-memory change (established by task 1869) | PASS |
| Restored parks are re-corroborated | owner-state park-GC runs immediately after rehydrate, emitting `reservation_expired` (INV-3/INV-7) | PASS |
| Ordering: backfill lands before park persistence | `producer:η` — **upstream** (θ depends η), enforced as a real edge per D10, not a config flag | PASS |

## ι — Below-rank-1 passed-over counting (observability only)

| Capability | Binding | Verdict |
|---|---|---|
| The scan-order pass-over is real and unrecorded | `grep:orchestrator/src/orchestrator/scheduler.py:6425,6436` — `_bump_skip_and_maybe_park` is called only for `top_id`, so ranks 2..k leave no record at all today | PASS |
| Counter surfaces through the product read path | `producer:ι` + the `read_scheduler_state` pass-through (as θ) | PASS |
| "never reaches `_bump_skip_and_maybe_park`" (negative assertion) | `rejection-check:` — asserted by a test on the call graph, since §7 rejects below-rank-1 park *installation* on measured grounds. Bound as a test property, not a grep (a plain ERE cannot express "this counter is not an argument to that function") | PASS |
| Storm escape | stride-gated `task_passed_over` emission (default 100) against ~291 candidates × 4 ticks/min (INV-4) | PASS |
| `pending_since`-derived scoring in place | `producer:β` — **upstream** (ι depends β) | PASS |

## κ — Integration gate: the whole seam, end to end

| Capability | Binding | Verdict |
|---|---|---|
| Every leg produced upstream | `producer:β,γ,δ,η,θ,ι` — all six are **upstream** of κ; DAG-direction verified, no inversion | PASS |
| `max_concurrent_tasks` is the dispatch-window bound | red-tier config field (restart-only per CLAUDE.md), read directly — the rank comparison is against a real bound | PASS |
| A qualifying chain head is observable at run time | **resolved at decompose, see R3** — live queue with a documented fallback to β's regenerated fixture; the script reports which source it used | PASS (resolved) |
| Boundary-test sketch is the named signal (G5 B+H closure) | PRD §6 rows 1–9; κ drives rows 1/2/6/7/8 through the **real** scheduler | PASS |

---

**Gate result: 10/10 task blocks PASS.** Three bindings (R1 β-baseline, R2 δ-pin-source,
R3 κ-live-candidate) did not clear as authored and were resolved before filing — by relaxing
a bound to the reproducible figure, by adding the missing producer to a task's scope, and by
naming the configuration that makes the assertion observable. One G7 addition (γ / INV-4)
resolved by redesign. No waivers.
