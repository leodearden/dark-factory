# Per-project quality dashboard: outcome and change-cost series per landing, annotated, with a deterministic quality-metrics cadence

Status: active, 2026-11-04. Authored from the ruled design note
`plans/portfolio-review-2026-11-01/followups/quality-dashboard.md` (seat evidence under
`followups/quality-dashboard-seats/`; Leo's rulings of 2026-11-04 in its §0). Approach B + H.

**Code anchors** verified against main `a26991efc9` (2026-11-04). Main moves fast — cite-by-symbol;
re-locate at implementation time.

## Goal

An operator opens the dashboard's **Quality** tab for one project and can answer "is the quality of this
project's code improving, and since when" without being misled by noise, small samples, mass events or
regime changes:

- five **outcome** tiles per 100 landings (merge-candidate verify rejections, reviewer blocking, confirmed
  flakes with real-reds beside them, genuine reverts as a count, re-verifies per landing), each a rate over
  the last 100 landings with a 95% Wilson band, an EWMA line per landing, and CUSUM alarms marked
  "explained by <marker>" or "unexplained change at landing #i";
- a **change-cost** section (all-role cost per landing normalised to a 100-line reference, the median
  baseline residual as the trend, implementer invocations per landing, turns per landing, share of landings
  with ≥2 review rounds, charter-block share), annotated with every model, effort, briefing and routing
  change derived from the ledgers plus curated markers, and a before/after effect panel per marker;
- a **per-module** drill-down over landings touching a module;
- a **quality-metrics** step series: one point per nightly snapshot of `scripts/quality_metrics_snapshot.py`
  at a pinned sha, with `/review-all` baselines and instrument-version changes marked, and a readable
  daily diff.

Every figure is a `Datum` carrying its counting rule as structured fields, renders `unknown` with the
reason when n is too small or the source has not started, and never paints a bare number for a fact not
measured.

## Background

The 2026-11-01 portfolio review (§3.4) found nothing could say whether quality was improving: one
quality-metrics snapshot per project, `review_completed` events only since 10-21, no task `created_at`.
Seat X3's own figures were then found to be wrong in ways that matter (design note §2, §10): post-merge
rejections divided by all `merge_finalized` events are two to three times low per landing; the
"flake-bearing tasks doubling" series is real reds, confirmed flakes are flat; and runs.db already holds
962 of 964 df landings once `merge_finalized done` is joined with `train_merged`, so git first-parent
merges, not runs.db, are the incomplete landing source.

The dashboard's discipline is `plans/dashboard-one-datum-one-path-prd.md` (all 14 leaves landed): one
access implementation per fact, a `Datum` envelope with `state ∈ {fresh, stale, unknown, lower_bound}`,
samplers that write value-or-gap rows, views computed in Python and never re-bucketed in JS. Nothing in
the fleet pushes to the dashboard, and the only in-process landing callback lives in the orchestrator's
merge lane, which the dashboard may not import (that PRD, decision 19). A 90-day first-parent shortstat
costs 23.8 s on df (loose objects), so git never runs on the request path.

`scripts/quality_metrics_snapshot.py` (schema 2) is already byte-identical per sha across checkout path,
locale and interpreter (measured on three runs at `922d9d3baa`, design note §6), but measuring the
live main checkout fails about 30% of the time because main moves inside the 6-minute run, and the
header records no instrument identity, so a tool bump diffs as a code change.

Measured magnitudes the treatment is sized for (design note §2–§5): df lands ~160 tasks a week (reify
~80); df post-merge rejections 7–22 per 100 landings; reviewer blocking 10–14% of reviews; cost per
landing ∝ lines^0.57, so raw cost per line tracks the size mix; at p0 ≤ 0.19 a 3σ EWMA lower limit is
negative, so an EWMA chart can never show improvement at these rates.

## Sketch of approach

1. **One landing datum.** A landing is a `merge_finalized state=done` event or a `train_merged` member;
   its time is main's advance (the event timestamp), its key `(project_id, merge_sha)`. A **verify unit**
   covers the set of landings one merge verify proved (a train's `member_task_ids`, a chain's
   `chain_items`): per-landing outcome flags are inherited by every member; verify and rejection counts
   are per unit.
2. **A shared producer** (`shared/src/shared/quality_series/`) computes, for landings newer than a
   per-project cursor, one immutable per-landing row from runs.db (events, invocations, flake tables),
   git (lines and paths of the landing vs its first parent, budgeted) and, for backfill only, the on-disk
   review counter. It also holds the pure statistics (stdlib only). The dashboard and any offline re-run
   call the same functions.
3. **A dashboard sampler loop** (600 s, beside `dashboard/loops.py::_burndown_loop`) appends rows to a
   new `quality.db` outside the retention policy and writes a value-or-gap `sample_state` row per
   project per tick. The read side marks the datum `stale` with "N landings not yet sampled" when main
   is ahead of the cursor; nothing computes on read.
4. **A marker stream**, derived (routing steps from `routing_decision`, cause from `config_reload`,
   deploy rug from `runs.code_sha`, model version from `invocations.model_id`, instrument params from
   snapshots) plus curated (`dashboard/regime-markers.yaml` gaining `project`, `kind`, `until`, `series`),
   with one reader and landing ordinals resolved server-side.
5. **A Quality tab** with `StatTile`s whose spark is the persisted rolling series, band charts with the
   marker overlay, the before/after panel, the module table and the snapshot step series; the band prop
   and marker overlay are added to `charts.jsx` once and the escalation-analytics tab consumes the lifted
   overlay.
6. **Two orchestrator fields** make two series honest at the source: a typed `block_class` written where
   the architect blocks a task, and a `prompt_digest` on every `routing_decision`.
7. **A deterministic quality-metrics cadence**: schema 3 records measurement identity; a wrapper pins
   the tree, skips an already-measured identity, writes uncommitted snapshots under
   `data/quality-metrics/<project_id>/`; a nightly per-project systemd user timer runs it; a weekly
   backfill since 10-08 seeds the series; `/review-all` Phase 1 accepts the newest nightly snapshot as
   its previous.

## Resolved design decisions

1. **Landing = runs.db, not git.** `merge_finalized done` ∪ `train_merged` members, cross-checked
   against git first-parent and labelled "runs.db landings; git first-parent disagreement: N". Rationale:
   measured 962/964 df and 484/485 reify coverage versus git's 810/964. No merge-lane change:
   `merge_finalized.absorbed_request_ids` is registry waiters (merge-status-durable-non-landed PRD D7)
   and is legitimately empty.
2. **Landing time = the event timestamp** (main's advance), never the merge commit's committer date
   (median 26 min earlier on df, 32 on reify; a +1 h commit-time cutoff loses 26% of reify's
   `merge_verify` rows). Attribution window for a landing of task T: `(previous landing time of T,
   landing time]`. For a train member the train's `train_merged.timestamp` is the anchor (leaf α verifies
   it is main's advance; if not, the member's own `merge_finalized done` timestamp).
3. **Verify units** (Leo, 11-04). Outcome flags are inherited by every landing of a unit; counts are per
   unit: `re-verifies per landing` = (verify units attributed in the window − landings-as-units) ÷
   landings; `rejections per 100 landings` counts `merge_blocked` events per unit. Both "per 100" and
   "share of landings with ≥1" are served; the chart shows the share, the tile both.
4. **Outcome series rules** (design note §2): rejections = `merge_blocked` with
   `json_extract(data,'$.category')='post_merge_verify'`, never the `reason` prefix; reviewer blocking =
   `review_completed.$.has_blocking_issues` over **reviewed** landings, `unknown` before 2026-10-21;
   confirmed flake = any `flake_occurrence` verdict `passes_in_isolation` (df cross-check:
   `merge_flake_suppressed`), real red = `fails_in_isolation`, `unconfirmable` in neither; reverts =
   subject `(?i)^revert` commits placed by the first-parent landing that carried them, counted only when
   the target landed earlier, served as a count with its Wilson band, as a late-event stream keyed to the
   target landing; re-verifies = `merge_verify` rows per unit minus one (`workflow_verify` is green-only;
   `task_results.verify_attempts` is debugger rounds).
5. **Change-cost headline = all-role cost** (Leo, 11-04): every `invocations` row with the task's id and
   `started_at ≤ landing time`, all attempts, every role; per-role split served as a stacked secondary.
   `task_results.cost_usd` is never used (−13%, 69% coverage). Landings with no invocation at all (52 of
   630 df) are excluded from the cost series with the label "orchestrated landings only" and included in
   the outcome series.
6. **Lines = code lines** (src + tests, added + deleted, excluding docs, lockfiles and data by path
   class), fixed and labelled; raw and src-only are not served (they move the level 3×). Module key =
   `<pkg>/<first component under src/<pkg>/>`; cost and lines allocated to modules by src-line share; a
   landing with no src lines goes to a bucket excluded from rankings; module cells need ≥1,000 allocated
   lines and ≥3 tasks over a 4-week window. `metadata.files` is never read (rewritten to the landed set).
7. **Effort series = implementer-class invocations per landing** (headline) with
   `task_results.execute_iterations` summed over dispatch rows as a labelled secondary; turns =
   `invocation_end.$.turns` summed over implementer-class invocations, with the right-censored share
   (`capped_reason='turns'`) served beside it.
8. **Review rounds = `review_completed.$.cycle` max per task** (the lifetime round number); for
   landings before 2026-10-21 the producer reads `.worktrees/.task-meta/<task_id>/review_state.json`
   `review_cycles_total + 1` once at backfill and stamps `source='disk_counter'` on the row (Leo, 11-04:
   do the backfill). No `task_results` stamp; task 5353 is a sibling per-dispatch defect, untouched.
9. **Charter blocks** read a typed `block_class` written at the architect block sites in
   `orchestrator/src/orchestrator/workflow.py` (the four `Architect reported …` / `Plan files not touched`
   strings): `under_specified` (unactionable, false premise), `stale_charter` (already done; Leo, 11-04:
   its own class beside `under_specified`), `plan_files_untouched`. The class is written to
   `task_results.block_class`, to the `task_blocked` event payload and to the escalation record; the
   regex over `block_reason` is used only for backfill rows, stamped `source='regex_backfill'`. Until the
   producer has written in production the series renders "no producer has written here" (INV-13).
10. **Prompt changes become derivable**: `routing_decision.data.prompt_digest` = sha256 of the role's
    rendered system prompt with task-specific text removed (the prompt constants in
    `orchestrator/src/orchestrator/agents/roles.py` plus the briefing templates), per dispatch. A step in
    the digest per role is a marker of kind `prompt`, dated at its first dispatch. Until it exists, the
    curated file is the only prompt-change source and the marker reader says so.
11. **Derived markers and de-noising** (design note §4): routing steps from `routing_decision` rows with
    `source_layer ∈ {config, role_default}` only; a step counts when it persists ≥3 decisions of that
    role and (≥24 h or the next step differs in another field); same-role steps within 24 h collapse to
    one dated at the earliest `config_reload` or commit; `config → role_default → config` is an
    availability flag, not a change; model/effort/prompt/briefing = labelled line, budget/turns = faint
    tick, deploy (`runs.code_sha`, since 10-21) = rug; a `params` change between two snapshots = marker of
    kind `instrument`.
12. **Curated markers** stay in `dashboard/regime-markers.yaml`, extended with optional `project`
    (default all), `kind` (`model | effort | budget | prompt | reviewer_policy | routing_rule | cli |
    deploy | artefact | regime`), `until` (artefact ranges) and `series` (which series the event is a
    regime event for); one reader (`dashboard/src/dashboard/data/quality_markers.py::load_curated_markers`)
    that `escalation_analytics.py` delegates to; the file's first new entry is the 2026-10-16 disk-full
    artefact (`kind: artefact`, `series: [flakes_confirmed, flakes_real_red]`). Staleness is visible: the
    marker datum reports roles.py landings since the last `prompt` entry while decision 10 is unlanded.
13. **Statistics** (design note §5; stdlib only, numpy is not a dashboard dependency): headline = last 100
    landings with a 95% Wilson band; n < 30 → `unknown` with reason; 30–99 → "k of n" with band; a source
    starting mid-window shrinks k and n, never pads. Chart = EWMA λ = 0.05 with the upper 3σ limit only.
    Alarms = two one-sided Bernoulli CUSUMs (p1 = 2p0 and p0/2) with **h = 5** (Leo, 11-04), reset at
    each series-tagged marker, onset = last zero, "explained" when within ±25 landings or ±3 days of such
    a marker. Baseline p0 = first 200 landings after the last series-tagged marker (provisional at
    100–199, absent below): a pure function of the landing list and the marker list, no persisted
    state. Counts (verifies per unit): seeded percentile bootstrap of the mean, with the share ≥2
    beside it. Costs: median on a log axis with a seeded bootstrap band (seed = sha256 of the window's
    sorted task ids); Theil–Sen log–log fit per regime baseline; reference cost at 100 lines and the
    median baseline residual; sign-of-residual CUSUM. Before/after a marker: Newcombe hybrid-Wilson
    difference for rates, bootstrap ratio of reference costs for cost; before ≤300 landings not crossing
    an earlier marker, after ≥100 before anything is shown, with n, ETA and minimum detectable effect
    until then. Mass events: all series are landing-bearing; an occurrence-cap marker at max(20, 10×
    baseline median); artefact ranges hatched and excluded from baseline, CUSUM and comparisons.
14. **Store and loop.** `quality.db` at `<project_root>/data/burndown/quality.db` (`config.quality_db`),
    owned by `_QualityStore` in `dashboard/loops.py` on the `_BurndownStore` pattern; tables `landings`
    (one row per `(project_id, merge_sha)`, immutable), `late_events` (reverts, keyed to the target
    landing), `markers_derived` (per project, role, kind, at, ordinal), `sample_state` (per project: value
    or gap with ts, reason, cursor sha, landings sampled). **Not** registered in
    `metrics.py::_DOWNSAMPLE_TABLES`; a test pins its absence from `retention.py::apply_retention`. A
    600 s `_quality_loop` added to the `_LOOPS` tuple in `dashboard/tests/test_lifespan_samplers.py`'s
    sense; per-root git budget on the `burndown.py::_SNAPSHOT_PER_ROOT_BUDGET` pattern; cold backfill
    runs inside the loop, resumable through the cursor; the loop runs `git` as the dashboard's first
    subprocess over project roots and only over roots in `config.all_project_roots` that are git
    checkouts.
15. **Endpoint and window.** `dashboard/src/dashboard/api/quality.py` registered with one
    `app.include_router`: `GET /api/v2/dashboard/quality?project=<label>&window=<spec>` returning
    `{QUALITY: Datum[QualitySeries], QUALITY_MARKERS: Datum[list[Marker]], QUALITY_SNAPSHOTS:
    Datum[list[SnapshotPoint]], WINDOW, served_at}`. Window vocabulary = landings (`100`, `300`, `all`;
    Leo, 11-04) with its own `_parse_window` vocabulary and `window_chip.js` entry; the x-axis is landing
    ordinal with date ticks.
16. **Front end.** New `static/redux/tab_quality.jsx` (`tabs.jsx` is at its ceiling); `charts.jsx::LineChart`
    gains a `band` prop (`lo`/`hi` arrays, null holes respected) and a shared `Markers` overlay taking
    server-resolved `ordinal` fractions; `tab_escalation_analytics.jsx` consumes the lifted overlay through
    an adapter from its date labels. No view is computed in JS (`deriveVelocitySeries` is the
    counter-precedent).
17. **Item 2: measurement identity.** Schema 3 `params` adds `python_version` and `instrument` (blob
    shas of `scripts/quality_metrics_snapshot.py` and `scripts/source_measures.py`); identity =
    (`as_of_sha`, `params`); a test pins byte-identity below `run_id` and `cost` for two runs at one
    identity; `diff_lines` opens with a "params differ" section and labels the deltas as not attributable
    to code (a report, never a gate: snapshot PRD decision 6). Schema 2 reads as "instrument unknown"
    with the polarity that PRD gives schema 1.
18. **Item 2: the exact-identity skip, argued against the snapshot PRD's decision 5** (Leo, 11-04:
    accepted). Decision 5 forbids "a skip when nothing changed rule" and names a blob cache as the future
    remedy; `plans/merge-lane-quality-prd.md` Correction 7's prohibition is of a content-heuristic skip
    whose "nothing changed" premise nothing proves. The wrapper's skip fires only when a snapshot with
    the **same measurement identity** already exists, and decision 17's test proves that a second run at
    that identity is byte-identical: this is not re-measuring an object already measured, not a judgement
    that nothing changed. The skip is keyed on identity alone; a changed `params` always re-measures.
    Decision 5 stands for everything else: no blob cache in this PRD.
19. **Item 2: placement and home.** `scripts/quality_snapshot_cadence.py` resolves the project's main
    sha, applies decision 18, pins the tree (`git worktree add --detach` under a state home, removed
    after), runs the instrument with the factory's `.venv/bin/python`, writes
    `data/quality-metrics/<project_id>/<as_of_sha>.json`, never commits (snapshot PRD decision 8 kept:
    `/review-all` keeps committing its own baseline to `plans/quality-metrics/`). Units
    `scripts/quality-snapshot@.timer` / `.service` on the `legibility-trickle@` template shape
    (`OnCalendar` 03:45, `Persistent=true`, `Nice=19`, low `CPUWeight=`), one instance per Python
    project (df, solar_challenge, solar_challenge_platform, scoping_intake); the Rust variant is out of
    scope until its G6 gaps are owned. Not coupled to the trickle (an LLM-bearing, committing pipeline;
    heuristic 6). A weekly backfill since 2026-10-08 for df runs once through the same wrapper with an
    explicit sha list (~200 CPU-s per point).
20. **Item 2: readers.** `/review-all` Phase 1 step 1's previous-snapshot rule gains "or the newest
    `data/quality-metrics/<project_id>/*.json` whose `as_of_sha` is an ancestor of `AS_OF`, whichever is
    newer" (`skills/review-all/SKILL.md`). The dashboard's `QUALITY_SNAPSHOTS` datum reads both
    directories, renders a sparse step series per project of the `--summary` per-member figures, the h14
    counts, cycle counts, deferred and cycle-closing imports and private patch targets, marks
    `/review-all` baselines and `params` changes, and shows no ranking, target, average or headline line
    count (contract §10; snapshot PRD decision 6).
21. **Where statistics live**: in the shared producer, not the dashboard module (Leo, 11-04:
    recommended (ii)). `shared/src/shared/memory_eval_limits.py` is left untouched; the quality
    statistics need Wilson, Newcombe, bootstrap, EWMA, CUSUM and Theil–Sen, none of which it has, and
    importing a memory-eval module from a quality module would make a confusing seam.
22. **Rule fields are structured** (heuristic 12): every served metric carries `rule = {counts,
    excludes, population, window_landings, measured_since, source}`; JS renders the label from them and
    parses nothing.

## Pre-conditions for activating

- None external. The one-datum PRD's leaves 5585–5598 are done; `Datum`, `StatTile({datum, history})`,
  `api/window.py::with_window`, `_BurndownStore`'s value-or-gap shape and the boundary suite exist on
  main `a26991efc9`.
- Producers verified writing in production (INV-13): `merge_finalized` (898 done rows df W40–W44),
  `train_merged` (74), `merge_blocked` (168 post_merge_verify), `review_completed` (since 10-21),
  `merge_verify` (1,335), `flake_occurrence` (1,653 rows in window), `routing_decision` (24,841 all
  time), `config_reload` (26), `runs.code_sha` (since 10-21), `invocation_end.$.turns` (since 04-09).
  The two producers this PRD adds (`block_class`, `prompt_digest`) have not written; their readers render
  "no producer has written here" until they do (leaf γ's signal includes that state).

## Cross-PRD relationship (G4)

| Other PRD / owner | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/dashboard-one-datum-one-path-prd.md` | consumes | `Datum` envelope, `StatTile` history, window datum, sampler value-or-gap contract, `test_datum_access_paths.py` guard | this PRD extends the guard to the quality event types; the other PRD's follow-ups 6469/7443/8187/6926/6253 do not block | wired |
| `plans/quality-metrics-snapshot-prd.md` | extends | schema 3 `params`, `diff_lines` params section, decision 5 (skip, argued in decision 18), decision 8 (kept) | this PRD (leaf η) | queued |
| `plans/escalation-lifecycle-dashboard-prd.md` (Seam 2, regime markers) | consumes and moves | `regime-markers.yaml` schema extension; `load_regime_markers` becomes a delegate of `quality_markers.py`; `RegimeMarkers`/`TimeChart` lifted into `charts.jsx` | this PRD (leaf δ, ε) | queued |
| `plans/dashboard-availability-prd.md` | consumes | request budgets: nothing computes on read; git only in the loop | this PRD honours; no change there | wired |
| `plans/merge-status-durable-non-landed-prd.md` (D7 `absorbed_request_ids`) | reads | `merge_finalized` and `train_merged` payloads as emitted | no change | wired |
| `docs/quality-findings-contract.md` §10–11; `skills/review-all/SKILL.md` Phase 1 | extends | previous-snapshot rule accepts nightly snapshots | this PRD (leaf θ) | queued |
| instruments follow-up item 8 (task 5353, `task_results.review_cycles`) | sibling | none: this PRD reads `review_completed.cycle` and never `task_results.review_cycles` | 5353 | untouched |
| flake ledger θ (3792) and 5875 | sibling | `flake_debt` is not read by this PRD | 3792/5875 | untouched |
| `plans/remote-task-verify-*` (7977) and verify-result cache | none | the sampler and the nightly snapshot consume no verify slot | — | — |

## Contract (B + H)

### Landing record (`shared/src/shared/quality_series/landings.py`)

```
Landing = {project_id, merge_sha, task_id, landed_at (UTC ISO), ordinal (int, per project, by landed_at
           then merge_sha), source ('merge_finalized' | 'train_merged'), unit_id (merge_sha of the verify
           unit's tip; == merge_sha for a plain landing), unit_size (int ≥ 1), git_first_parent (bool)}
landings(runs_db, repo, since_cursor) -> list[Landing]   # ordered; never re-emits a sha ≤ cursor
```
Invariants: one row per `(project_id, merge_sha)`; `unit_size == len(members)` for a train or chain;
every member of a unit shares `unit_id`; `git_first_parent` is False for a landing whose sha is not on
main's first-parent line (served as the disagreement count, never dropped).

### Per-landing row (`shared/src/shared/quality_series/rows.py`)

```
LandingRow = Landing + {
  outcomes: {rejected: bool|None, rejection_events: int, reviewed: bool, review_blocking: bool|None,
             flake_confirmed: bool, flake_real_red: bool, flake_rows: int, reverify_count: int,
             verify_failed_rows: int},
  cost:     {all_role_usd: float|None, by_role: {role: usd}, implementer_invocations: int,
             execute_iterations: int|None, turns: int|None, turns_capped: bool, code_lines: int,
             src_lines: int, test_lines: int, review_rounds: int|None, review_rounds_source:
             'review_completed'|'disk_counter'|None, block_class: str|None, block_class_source: ...},
  modules:  [{module, src_lines}],
  rule_vintage: {metric_id: measured_since_iso}}
rows_for(landings, runs_db, repo, git_budget_s) -> list[LandingRow]
```
Invariants: `None` means "source not started or not applicable", never 0; `rejected`/`reverify_count`
are the unit's values copied to every member; `all_role_usd is None` iff the task has no `invocations`
row; `code_lines` follows decision 6's path classes; every window is `(prev_landing_time(task),
landed_at]`.

### Pure statistics (`shared/src/shared/quality_series/stats.py`)

`wilson_interval(k, n, z=1.96)`, `newcombe_difference(k1, n1, k2, n2)`, `content_seed(task_ids)`,
`bootstrap_interval(values, stat, seed, B=1000)`, `ewma_series(xs, lam, z0)`, `ewma_limits(p0, lam, L, t)`,
`cusum_alarms(xs, p0, p1, h) -> [(alarm_idx, onset_idx)]`, `cusum_design(p0) -> (2p0, p0/2, 5)`,
`theil_sen(xs, ys)`, `reference_cost(lines, costs, ref=100, min_lines=5)`, `log_residuals(...)`,
`cap_occurrences(counts, baseline_median, floor=20)`, `regime_segments(landings, markers, series)`,
`baseline_rate(segment_xs, n_base=200, n_min=100) -> (p0, 'frozen'|'provisional'|None)`,
`block_rate_datum(xs, task_ids, n_block=100, n_min=30)`, `before_after_rate(...)`, `before_after_cost(...)`,
`explain_alarm(onset_idx, onset_at, markers, k=25, days=3)`. All deterministic for equal inputs; no
module-level state; stdlib only.

### Wire shape (`dashboard/src/dashboard/data/quality_series.py::QualitySeries.to_wire`)

```
{axis: [{ordinal, merge_sha, task_id, landed_at, unit_id}],
 metrics: {<metric_id>: {rule: {counts, excludes, population, window_landings, measured_since, source},
                         unit, points: [number|null], rolling: {center, lo, hi, n}, ewma: {line, ucl},
                         alarms: [{ordinal, onset_ordinal, direction, explained_by: marker_id|null}],
                         baseline: {p0, state: 'frozen'|'provisional'|null, since_ordinal},
                         headline: Datum(number|{k,n})}},
 cost_model: {a, b, reference_cost_100, residual_median, state},
 modules: [{module, landings, metrics: {...}}],
 before_after: [{marker_id, n_before, n_after, effect, lo, hi, state, eta_days, min_detectable}]}
Marker = {id, at, ordinal|null, kind, scope: project|'fleet', series: [..]|null, until|null, label,
          source: {type, role?, path?, file?}, ref}
SnapshotPoint = {as_of_sha, committed_at, params, complete, baseline: bool, members: {...}, h14: {...},
                 cycles: {...}, deferred_imports, cycle_closing_imports, private_patch_targets}
```
`validate_datum` runs at shaping; a metric whose `measured_since` is after the window start serves
`null` points before it and `headline.state='unknown'` with the reason naming the source date.

### Markers (`dashboard/src/dashboard/data/quality_markers.py`)

`derived_markers(runs_db, project_id, snapshots) -> list[Marker]` (decision 11 rules, idempotent, persisted
to `markers_derived` by the loop with a per-(project, role) last-seen key); `load_curated_markers(path)
-> (list[Marker], parse_failures)` fail-soft, unknown keys ignored, old entries valid with
`kind='regime'`, `scope='fleet'`, `series=None`; `resolve_ordinals(markers, axis)` sets `ordinal` to the
first landing at or after `at`.

### Orchestrator fields

- `task_results.block_class TEXT NULL` (additive column with `CREATE TABLE IF NOT EXISTS` + `ALTER`
  guard on the `_ADDED_*` pattern), value ∈ {`under_specified`, `stale_charter`, `plan_files_untouched`};
  the same key on the `task_blocked` event payload and the escalation record's `detail`.
- `routing_decision.data.prompt_digest: str` (64 hex), computed by one function
  (`orchestrator/src/orchestrator/agents/prompt_digest.py::digest_for_role`) over the role's constant
  prompt parts; absent → the marker reader reports `prompt_digest: not yet written`.

### Snapshot identity (`scripts/quality_metrics_snapshot.py`)

`params = {complexipy_version, python_version, instrument: {quality_metrics_snapshot: <blob sha>,
source_measures: <blob sha>}, h14_soft_ceiling_lines, h14_alarm_lines}`; `SCHEMA_VERSION = 3`;
`measurement_identity(snapshot) -> (as_of_sha, params)`; `diff_lines` first section `params differ: …`
when identities' `params` differ. `scripts/quality_snapshot_cadence.py --root <project_root>
--project-id <id> [--sha <sha> ...]` exits 0 with `already measured: <sha> <identity>` on a hit.

## Boundary-test sketch (B + H) — the integration gate κ's signal

| # | scenario | preconditions | postconditions |
|---|---|---|---|
| 1 | A plain landing becomes a row within one tick | df runs.db has a `merge_finalized done` row newer than the cursor; main contains its sha | `quality.db.landings` gains exactly one row keyed by that sha; `sample_state` is a value row with cursor = that sha; `/api/v2/dashboard/quality?project=dark_factory` serves it at the last ordinal with `state='fresh'` |
| 2 | A train of 3 becomes 3 landings and one verify unit | a `train_merged` row with 3 `member_task_ids` and one `merge_verify` under the tip | 3 rows share `unit_id`, `unit_size=3`; `reverify_count` equal on all three; the window's verify-unit count rises by 1, landings by 3 |
| 3 | Git disagreement is served, not dropped | a landing whose sha is a fast-forward (not a merge) on first-parent | row present with `git_first_parent=True`; a Merge-subject commit with no runs.db row raises the disagreement count in the datum's rule, no row is invented |
| 4 | A source that has not started renders unknown | window entirely before 2026-10-21 | `reviewer_blocking.points` are all `null`; `headline.state='unknown'` with reason naming 2026-10-21; nothing is 0 |
| 5 | Small n renders "k of n" | a project with 40 landings | headline value `{k, n}` with a Wilson band; no EWMA limits, no CUSUM, no before/after; state `fresh` |
| 6 | Artefact range excluded | curated marker `kind: artefact, until:` covering 2026-10-16 with `series: [flakes_confirmed]` | landings inside are served with `in_artefact=true`, excluded from `baseline`, `alarms` and `before_after`; the chart hatches them |
| 7 | Derived marker for the 7972 reviewer flip | df and reify runs.db | one marker per project, role `reviewer_comprehensive`, kind `effort`, `at` = first xhigh/90 dispatch (df 2026-11-01T08:58Z, reify 10:37Z), `ordinal` resolved; the reify marker is not dated from df's commit |
| 8 | De-noising drops a one-dispatch fallback | df merger `role_default` step of 2026-10-27 13:46 (n=1) | no marker; the availability flag is served separately |
| 9 | Alarm explained vs unexplained | synthetic Bernoulli vector with a known shift at index i and a marker at i−10 | `alarms[0].explained_by` is that marker; the same vector with the marker at i−100 yields `explained_by=null` |
| 10 | Deterministic bands | the same window served twice | `rolling.lo/hi` and `before_after.lo/hi` byte-equal (content seed) |
| 11 | Gap row on failure | a project root whose runs.db is locked or git absent | `sample_state` gap row with the verbatim reason; datum `state='stale'` with age; no zero row |
| 12 | Main ahead of cursor | 3 landings land between ticks | read side reports `state='stale'`, reason "3 landings not yet sampled"; next tick clears it |
| 13 | Retention never touches quality.db | `downsample_metrics` runs | `landings` row count unchanged; a test pins `quality.db` absent from `_DOWNSAMPLE_TABLES` |
| 14 | Block class written | an architect blocks a task as unactionable in a dispatched run | `task_results.block_class='under_specified'`, the `task_blocked` event and the escalation carry it; the charter series' next row reads it with `source='block_class'` |
| 15 | Prompt digest written and stepped | a dispatch after a `roles.py` prompt-constant edit | `routing_decision.data.prompt_digest` differs from the previous dispatch of that role; after 3 dispatches a `kind='prompt'` marker appears |
| 16 | Snapshot identity | two cadence runs at one sha with equal params | second exits 0 `already measured`; the first's file is byte-identical below `run_id`/`cost` to a forced re-run |
| 17 | Params differ labelled | diff of a schema-2 snapshot against a schema-3 one | `--diff` opens with `params differ: instrument unknown → …`; measure deltas carry the not-attributable label; exit 0 |
| 18 | Live-main hazard avoided | main moves during the nightly run | the run measures the pinned worktree and exits 0; the worktree is removed |
| 19 | `/review-all` previous = newest nightly | `data/quality-metrics/dark_factory/` holds a snapshot newer than `plans/quality-metrics/*.json` | Phase 1 names the nightly file as the previous snapshot |
| 20 | Snapshot step series | three snapshots at three shas, one with changed `params` | `QUALITY_SNAPSHOTS` serves three points ordered by `committed_at`, a `kind='instrument'` marker at the params change, baselines flagged |
| 21 | Escalation tab unchanged | existing regime markers | the escalation-analytics charts render the same three 07-15 markers through the lifted overlay |

## Decomposition plan

Sizing per the overlay's bands (300–1,500 LOC, ≤10–12 files per leaf). Labels are intra-batch; every leaf
cites `docs/code-quality.md` heuristics by name in its brief where they bind (notably *well-defined
purpose*, *structured data instead of meaningful strings*, *files make internal sense in isolation*).

- **α — Shared producer: landings, verify units, per-landing rows, statistics.**
  Modules: `shared/src/shared/quality_series/{landings,rows,stats,git_lines,__init__}.py` + tests.
  Signal (leaf-shaped CLI, run against the live df store): `python -m shared.quality_series --root
  /home/leo/src/dark-factory --since <sha>` prints one row per real landing since that sha with
  `unit_size`, outcome flags and cost terms, and reports the git first-parent disagreement count; the
  numbers for W44 reproduce the design note's §2–§3 trial tables under their stated rules (rejections
  per 100 landing-attributed, confirmed flakes, review rounds share) within the rule change from
  commit-time to event-time anchoring, which the output states. Verifies that `train_merged.timestamp`
  is main's advance (decision 2) and records the finding in its brief's return. Includes the disk-counter
  and regex backfill sources stamped by `source`. Prereqs: none.
- **β — quality.db store and the 600 s sampler loop.** Modules: `dashboard/loops.py` (`_QualityStore`,
  `_quality_loop`), `dashboard/config.py` (`quality_db`), `dashboard/app.py` (lifespan spawn, a few
  lines), `dashboard/tests/test_lifespan_samplers.py`, `test_retention.py` (pin), new store tests.
  Intermediate: unlocks γ. Signal it must still show: after a real df landing, within one tick a row
  with that sha exists in `quality.db` and a value `sample_state` row names it (sketch rows 1, 11, 13).
  Prereqs: α.
- **γ — Reader, endpoint, window vocabulary, access-path guard.** Modules:
  `dashboard/src/dashboard/data/quality_series.py`, `api/quality.py`, `api/window.py` (landing
  vocabulary), `tests/test_datum_access_paths.py` (extend `GUARDED` to the quality event types),
  boundary payload builders. Signal: `GET /api/v2/dashboard/quality?project=dark_factory&window=100`
  returns a validated `Datum[QualitySeries]` with one axis point per landing in `quality.db`, `unknown`
  headlines before each source's epoch, `stale` with "N landings not yet sampled" when main is ahead
  (sketch rows 3, 4, 5, 10, 12); the charter and prompt-derived figures render "no producer has written
  here" until ζ fires (INV-13). Prereqs: β.
- **δ — Markers: derived stream, curated schema extension, one reader, server-side ordinals.** Modules:
  `data/quality_markers.py`, `regime-markers.yaml` (+ `project/kind/until/series`; first new entry = the
  10-16 disk-full artefact), `data/escalation_analytics.py` (delegates `load_regime_markers`),
  `loops.py` (`markers_derived` refresh in the quality loop), `api/quality.py` (`QUALITY_MARKERS`).
  Signal: the endpoint serves the 7972 reviewer flip as a derived `effort` marker per project with the
  measured first-dispatch times, drops the df merger 10-27 fallback, and the disk-full artefact range
  excludes its landings from the flake baselines (sketch rows 6, 7, 8). Prereqs: γ.
- **ε — Quality tab and chart primitives.** Modules: `static/redux/tab_quality.jsx` (new),
  `charts.jsx` (`band` prop, shared `Markers` overlay), `tab_escalation_analytics.jsx` (adapter),
  `app.jsx`, `poll_scope.js`, `data.js`, `window_chip.js`, `index.html`, `tests/js/boundary_quality.test.mjs`,
  the static-asset pins. Signal: the Quality tab renders for df the five outcome tiles with bands and the
  change-cost charts with markers from the live endpoint; the escalation-analytics charts still show
  their three 07-15 markers (sketch rows 9, 21). Prereqs: δ.
- **ζ — Orchestrator: typed `block_class` and `prompt_digest`.** Modules:
  `orchestrator/src/orchestrator/workflow.py` (the four architect block sites), `event_store.py`
  (`task_results.block_class`, event payload), `escalation` record detail, `routing_dispatch.py`,
  new `agents/prompt_digest.py`, tests. Signal: in a dispatched run on main, an architect block writes
  `block_class` to `task_results`, the `task_blocked` event and the escalation; every new
  `routing_decision` row carries a 64-hex `prompt_digest` that changes when a role's prompt constant
  changes (sketch rows 14, 15). Prereqs: none (parallel to α–ε); κ depends on it.
- **η — Snapshot identity: schema 3, identity test, labelled diff, cadence wrapper.** Modules:
  `scripts/quality_metrics_snapshot.py`, `scripts/source_measures.py` (version read), new
  `scripts/quality_snapshot_cadence.py`, `tests/scripts/test_quality_metrics_snapshot.py`,
  `plans/quality-metrics-snapshot-prd.md` (correction block citing decision 18 and this PRD). Signal:
  two wrapper runs at one sha on df: the second prints `already measured` and exits 0; `--diff` of the
  10-08 schema-2 snapshot against the new one opens with `params differ` and exits 0 (sketch rows 16,
  17, 18). Prereqs: none.
- **θ — Nightly timer, weekly backfill, `/review-all` previous-snapshot rule.** Modules:
  `scripts/quality-snapshot@.timer`/`.service`, an install note in `OPERATIONS.md`, the one-off backfill
  invocation (weekly shas since 2026-10-08 for df), `skills/review-all/SKILL.md` Phase 1 step 1. Signal:
  `systemctl --user list-timers` shows `quality-snapshot@dark_factory.timer`; the morning after enabling,
  `data/quality-metrics/dark_factory/` holds a snapshot at that night's main sha and the backfill points
  exist; a dry `/review-all` Phase 1 names the nightly file as previous (sketch row 19). Prereqs: η.
- **ι — Snapshot step-series datum and tab section.** Modules: `data/quality_snapshots.py`,
  `api/quality.py` (`QUALITY_SNAPSHOTS`), `tab_quality.jsx` section, `quality_markers.py`
  (`instrument` kind). Signal: the tab shows the df step series with the 10-08 baseline and the backfill
  points, a params-change marker between schema 2 and 3, and the per-member figures with no average or
  ranking (sketch row 20). Prereqs: γ, ε, θ.
- **κ — Integration gate.** Executes the boundary-test sketch rows 1–21 against the live df and reify
  stores (rows 14–15 against a dispatched run), records the measured figures in a dated section of this
  PRD, and adds the Quality tab to `OPERATIONS.md`'s dashboard section with the counting rules pointed
  at (never restated: INV-9). Signal: the sketch table with every row's observed postcondition; an
  operator reading the tab sees df's post-merge rejection series, its rise across October, and the 7972
  marker. Prereqs: ε, ζ, ι.

DAG: α → β → γ → δ → ε; η → θ → ι (ι also after γ, ε); ζ parallel; κ after ε, ζ, ι. 10 leaves.

G7 walk (advisory; `docs/legibility/design-invariants.md`): INV-1 contracts are the wire shape validated
by `validate_datum` and the manifest's `delivered_check`s, not prose; INV-9 the landing datum has one home
(`quality.db`, derived from runs.db) and markers one reader; INV-11 a failed tick writes a gap row with the
verbatim reason; INV-13 readers render "no producer has written here" for `block_class` and `prompt_digest`
until ζ fires in production, and every other producer is verified writing (Pre-conditions); INV-14 does not
apply (no load-sampler metric). No waivers.

## Out of scope

- The reify Rust snapshot variant (`skills/review-all/scripts/rust_metrics_snapshot.py`): lizard absent,
  hard-coded versions, absolute root, no HEAD-moved check; needs an owner (snapshot PRD §Out of scope).
- A per-landing quality-metrics snapshot (design note §6 option C) and the decision-5 blob cache.
- Costing hand-carried and interactive landings from Claude session ledgers.
- A task `created_at` column (instruments follow-up item 9).
- Pushing landings to the dashboard (a new cross-process seam; the sampler's one-tick latency suffices).
- Fleet-wide (multi-project) quality views; the tab is one project at a time.
- Correcting `task_results.review_cycles` (5353) or the flake-debt drain (3792/5875).
- Bayesian online change-point detection, Western Electric rules, numpy.

## Open questions (tactical)

1. **`train_merged.timestamp` as main's advance.** Leaf α verifies against the member sha's first reach
   of main; if it is the train's build time, use each member's `merge_finalized done` timestamp.
   Decide during α.
2. **`invocation_end.$.turns` across resumed sessions** (114 `session_resume` events in October): if
   cumulative, α sums only the last row per session. Decide during α by reading
   `shared/src/shared/cli_invoke.py`'s resume path.
3. **Per-merge git cost** (inferred ~30 ms, unmeasured): β measures it and sets the per-root budget from
   the measurement. Decide during β.
4. **Module key for non-`<pkg>/src/<pkg>` layouts** (reify, the solar repos): first two path components
   when depth ≥ 3; α's brief may refine per project. Decide during α.
5. **Whether a 7-minute nightly run on the laptop is preferable** when the host is saturated: θ ships the
   workstation timer; a laptop instance is a config choice later.
6. **CPUWeight value for the timer unit**: θ picks the lowest value with a precedent in
   `orchestrator/src/orchestrator/verify.py::_resolve_scope_cpu_weight`.
