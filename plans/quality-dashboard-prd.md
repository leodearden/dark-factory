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
   its time is main's advance (the event timestamp), its key `(project_id, merge_sha, task_id)` (train
   members share one merge commit; see §Contract → Landing record, Correction). A **verify unit**
   covers the set of landings one merge verify proved (a train's `member_task_ids`; chain units are
   not formed, see §Contract → Landing record, Correction): per-landing outcome flags are inherited by every member; verify and rejection counts
   are per unit.
2. **A shared producer** (`shared/src/shared/quality_series/`) computes, for landings newer than a
   per-project cursor, one per-landing row from runs.db (events, invocations, flake tables),
   git (lines and paths of the landing vs its first parent, budgeted) and, for backfill only, the on-disk
   review counter. It also holds the pure statistics (stdlib only). The dashboard and any offline re-run
   call the same functions.
3. **A dashboard sampler loop** (600 s, beside `dashboard/loops.py::_burndown_loop`) upserts rows into a
   new `quality.db` (decision 14) outside the retention policy and writes a value-or-gap `sample_state` row per
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
6. **Two orchestrator fields** make two series honest at the source: a typed `charter_class` written where
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
9. **Charter blocks** read a typed `charter_class` (named so because `orchestrator/src/orchestrator/unblock_types.py::BlockClass` is already the unblock path's closed four-value domain, branched on by the b3 gate, and the architect-stage cause is an orthogonal dimension: heuristic 3) written at the architect block sites in
   `orchestrator/src/orchestrator/workflow.py` (the four `Architect reported …` / `Plan files not touched`
   strings): `under_specified` (unactionable, false premise), `stale_charter` (already done; Leo, 11-04:
   its own class beside `under_specified`), `plan_files_untouched`. The class is written to
   `task_results.charter_class`, to the `task_blocked` event payload and to the escalation record; the
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
    (one row per `(project_id, merge_sha, task_id)`, upserted on that key: late claims and
    off-first-parent landings may be re-emitted; see §Contract → Landing record, Correction), `late_events` (reverts, keyed to the target
    landing), `markers_derived` (per project, role, kind, at, ordinal), `sample_state` (per project: value
    or gap with ts, reason, cursor sha, landings sampled). **Not** registered in
    `metrics.py::_DOWNSAMPLE_TABLES`; a test pins its absence from `retention.py::apply_retention`. A
    600 s `_quality_loop` added to the `_LOOPS` tuple in `dashboard/tests/test_lifespan_samplers.py`'s
    sense; per-root git budget on the `burndown.py::_SNAPSHOT_PER_ROOT_BUDGET` pattern; cold backfill
    runs inside the loop, resumable through the cursor; the loop runs `git` as the dashboard's first
    subprocess over project roots and only over roots in `config.all_project_roots` that are git
    checkouts.

    **Correction (β, 2026-11-09).** `sample_state.cursor_sha` is the newest stored first-claim
    on-line landing. A tick does not resume from `LandingBatch.next_cursor`: it resumes from the newest
    stored first claim that landed at least `DISPATCH_ROW_SETTLE_SECONDS` (1800 s) before the tick
    (`dashboard/src/dashboard/data/quality_sampler.py::collect_quality`,
    `::DISPATCH_ROW_SETTLE_SECONDS`). So `landings()` re-emits a row younger than the dispatch-row lag
    and the upsert re-measures it in place (the "β must re-measure" residual of §Contract → Per-landing
    row), and a budget-truncated cold backfill resumes exactly from what was stored. The residual it
    shares with α's `next_cursor`: a first claim whose landed_at order inverts its first-parent order
    could be skipped (0 such inversions measured among 3,539 df first claims, 2026-11-09). Beside the
    cursor, a value `sample_state` row carries `landings_sampled`, `landings_deferred` (emitted but past
    that tick's git budget) and the three `GitCrossCheck` counts over the tick's emitted range; a gap
    row carries only its reason, the producer's error verbatim with two exceptions: a
    `TaskDbUnreadable` refusal is rendered from its `path` and `reason` fields, because its message's
    remedy is written for tasks.db, not runs.db; and a tick whose own write fails records a gap
    naming that failure, so every git-checkout root still gets one row per tick. The key's `project_id` is
    `shared/src/shared/project_id.py::project_id_for_root` of the root, passed to `landings()`
    explicitly. `late_events` is created with no producer yet: α ships no revert detector.
15. **Endpoint and window.** `dashboard/src/dashboard/api/quality.py` registered with one
    `app.include_router`: `GET /api/v2/dashboard/quality?project=<label>&window=<spec>` returning
    `{QUALITY: Datum[QualitySeries], QUALITY_MARKERS: Datum[list[Marker]], QUALITY_SNAPSHOTS:
    Datum[list[SnapshotPoint]], WINDOW, served_at}`. Window vocabulary = landings (`100`, `300`, `all`;
    Leo, 11-04) with its own `_parse_window` vocabulary and `window_chip.js` entry; the x-axis is landing
    ordinal with date ticks.

    **Correction (γ, 2026-11-10).** What shipped, and where each part is homed:
    - "Main ahead of the cursor" counts runs.db landings that the store lacks, never git. The reader
      takes the `(merge_sha, task_id)` keys runs.db records after the cursor landing
      (`shared/src/shared/quality_series/landings.py::recorded_landing_keys`, which reuses α's
      landing-recognition SQL) and subtracts the stored keys. A train of 3 is one commit, so it
      reads "3 landings not yet sampled". A rev-parse could not count that, and it differs from the
      cursor after any non-landing commit. When runs.db cannot be read, or holds a landing event the
      landing rule refuses (an unrecognised train_merged payload, a timestamp without an offset),
      the datum is `stale` with a reason naming the store, never `fresh`.
    - The read side is `dashboard/src/dashboard/data/quality_series.py::read_quality_series`. It is
      `unknown` until a value sample exists: with no sample it says nothing was sampled, and with
      only gaps it gives the newest gap's reason verbatim. It is `stale` when a gap follows the
      newest value, when landings are not yet sampled, or once the value sample is older than
      `::QUALITY_FRESHNESS_BOUND_SECONDS` (2 × 600 s). Otherwise it is `fresh`, as of the newest value
      sample. The two newest samples are read without loading the sample history
      (`dashboard/src/dashboard/data/quality_store.py::newest_samples`).
    - The window is a landing count with its own served type. The parser is
      `dashboard/src/dashboard/api/window.py::parse_landing_window`, which shares one decline rule
      with the day parser. It echoes `WINDOW = {requested, served, landings}`
      (`::ServedLandingWindow`, `landings: null` for `all`), never a `days` field. The vocabulary is
      `dashboard/src/dashboard/api/quality.py::_QUALITY_WINDOWS`.
    - `?project=` takes the root's label or its project id in any spelling `fold_project_token` folds
      (`dashboard/src/dashboard/api/quality.py::_project_root`), because df's label is `dark-factory`
      and its id is `dark_factory`. A missing token, or one that resolves to no configured root,
      answers 200 with an `unknown` `QUALITY` saying so, never an empty series.
    - `QUALITY_MARKERS` and `QUALITY_SNAPSHOTS` are `unknown` Datums whose reasons start "no producer
      has written here" until δ and ι land. `read_quality_series` already takes `events`
      (`RegimeEvent`s), so δ only has to supply them. Until then, alarms carry `explained_by: null`
      and `before_after` is `[]`.
    - The access-path guard got a second vocabulary rather than an extended `GUARDED`. The quality
      sources are SQL words, not Python names, so the Load finder could never see them.
      `dashboard/tests/test_datum_access_paths.py::GUARDED_QUALITY_SOURCES` matches the event types in
      non-docstring string constants. `::GUARDED_QUALITY_READS` matches Loads of α's runs.db readers
      imported from `shared.quality_series`. `::_QUALITY_GRANTS` grants the reader, the sampler's
      `_measure_root`, and `data/merge_queue.py`'s three `train_merged` reads (the merge-queue tab's
      train listing and throughput).
    - The client half is ε's. γ adds the route to the boundary suite: the
      `dashboard/tests/_boundary_payloads.py::_quality_family` scenarios, server-half tests for
      sketch rows 3, 4, 5, 10 and 12 in `dashboard/tests/test_boundary_js.py`, and the
      `qualityMetric` / `qualityPayload` builders in `dashboard/tests/js/wire_fixtures.mjs`. It does
      not touch `data.js::endpointsFor` or its `EXPECTED_ENDPOINT_KEYS` pin (now in
      `dashboard/tests/js/data_apply.test.mjs`). A polled row could not name the per-project
      `?project=`, so the registration and the window chip entry are ε's to design.
    - Measured 2026-11-10 against the live df store (3,564 stored landings, host load average ~62):
      one `read_quality_series` call took 0.54–1.30 s, the first call the slowest. A warm split was
      about 0.2 s reading quality.db, 0.06 s counting unsampled landings in runs.db, and 0.24 s for
      the pure view at `window=100`.
16. **Front end.** New `static/redux/tab_quality.jsx` (`tabs.jsx` is at its ceiling); `charts.jsx::LineChart`
    gains a `band` prop (`lo`/`hi` arrays, null holes respected) and a shared `Markers` overlay taking
    server-resolved `ordinal` fractions; `tab_escalation_analytics.jsx` consumes the lifted overlay through
    an adapter from its date labels. No view is computed in JS (`deriveVelocitySeries` is the
    counter-precedent).

    **Correction (ε, 2026-11-12).** What shipped, and where each part is homed:
    - `/api/v2/dashboard/quality` is an on-demand read, never polled: one read costs 0.5–1.3 s, its data
      moves on the sampler's 600 s tick, and a polled row cannot carry `?project=`. It is the `quality`
      row of `dashboard/src/dashboard/static/redux/data.js::ON_DEMAND_KEYS`. The param is the canonical
      query `data.js::qualityQuery` builds (project, window, module, in that order), an opaque identity
      that nothing parses back. The body's keys are fixed while their values depend on the param, so the
      row declares `fields` (`QUALITY`, `QUALITY_MARKERS`, `QUALITY_SNAPSHOTS`), each landed under
      `<FIELD>:<query>` and read from the body by `data.js::fromBody`. `data.js::onDemandReceipt` exposes
      the request's receipt and its `WINDOW` echo. `endpointsFor` and its pins are unchanged.
    - The landing window belongs to the tab, not the global chip. The vocabulary is
      `dashboard/src/dashboard/static/redux/window_chip.js::WINDOW_SETS` (`landings`, default
      `::LANDING_DEFAULT_WINDOW`), and `::TAB_WINDOWS` has no `quality` entry. Two facts force this. An
      on-demand receipt is keyed by its query, not by a path the global chip can look up. And the global
      window is also sent to the day-windowed chrome endpoints. One echo validator,
      `window_chip.js::receiptWindow`, accepts both day and landing echoes, so the label, highlight and
      pending functions are shared.
    - `?module=` drills down on the server. `dashboard/src/dashboard/data/quality_series.py::read_quality_series(module=)`
      keeps only the stored landings whose lines touch the module, and
      `dashboard/src/dashboard/api/quality.py::api_quality` passes the parameter through. The filter
      runs after the freshness check, so "N landings not yet sampled" still describes the whole
      project's store. Train members share modules, so verify units stay whole.
    - The overlay is an HTML layer over the chart (`dashboard/src/dashboard/static/redux/charts.jsx::Markers`,
      wrapped with its chart by `::MarkedChart`). It is positioned from the one `charts.jsx::PLOT_PAD`
      that `LineChart` and `StackedAreaChart` also use. Its decisions live in
      `dashboard/src/dashboard/static/redux/chart_marks.js`: the style for each marker kind, placement
      on a gapped ordinal axis or a date-label axis, the hover text and the legend. The server places
      each marker at a landing ordinal; turning that ordinal into a fraction of the served axis is
      placement (`chart_marks.js::fractionAt`), not a view. Escalation analytics uses the overlay through
      `chart_marks.js::dateLabelMarks`, and its regime marks render exactly the DOM its old
      `RegimeMarkers` did (sketch row 21).
    - The tab is `dashboard/src/dashboard/static/redux/tab_quality.jsx::QualityTab`. Its quality-specific
      decisions live in `dashboard/src/dashboard/static/redux/quality_view.js`: the layout, the text of a
      headline, rule, alarm verdict, before/after comparison and cost-model reading, the chart arrays
      (served verbatim) and each metric's marks. While the tab is open it polls only `/orchestrators`,
      for its project selector (`poll_scope.js::TAB_ENDPOINTS`).
    - Two wire limits are rendered around, not computed. The wire carries no per-landing residual
      series: the cost model is shown as `cost_model.reference_cost_100` and
      `cost_model.residual_median`. The wire carries no occurrence-cap field: mass events reach the
      tab only as the served `artefact` markers, drawn as hatched ranges.
    - Tests: the client half of quality sketch rows 3, 4, 5, 9, 10 and 12 is
      `dashboard/tests/js/boundary_quality.test.mjs`. The tab's render cases are in
      `dashboard/tests/js/jsx_render_quality.test.mjs`.
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
  The two producers this PRD adds (`charter_class`, `prompt_digest`) have not written; their readers render
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
Landing = {project_id, merge_sha, task_id, landed_at (tz-aware UTC), ordinal (int, per project, by landed_at
           then merge_sha), source ('merge_finalized' | 'train_merged' | 'train_attributed'), unit_id (merge_sha of the verify
           unit's tip; == merge_sha for a plain landing), unit_size (int ≥ 1), git_first_parent (bool)}
           + late_claim, prev_landed_at, lines_base, base_sha (see the Correction below)
landings(runs_db, repo, since_cursor) -> LandingBatch(landings, git_check, next_cursor)
    # ordered; since_cursor=None is the full history; may re-emit late claims and off-first-parent
    # landings, so consumers upsert on (project_id, merge_sha, task_id); the next call's
    # since_cursor is next_cursor, never main's HEAD
```
Invariants: one row per `(project_id, merge_sha, task_id)`; `unit_size == len(members)` for a train;
every member of a unit shares `unit_id`; `git_first_parent` is False for a landing whose sha is not on
main's first-parent line (served as the disagreement count, never dropped).

**Correction (α, 2026-11-04).** Measured on the df runs.db and git; the shipped shape is
`shared/src/shared/quality_series/landings.py::landings`.
- The key is `(project_id, merge_sha, task_id)`. A train lands as one merge commit and its members have
  no `merge_finalized done` row (0 of 287), so every member carries `merge_sha == unit_id ==
  train_merged.merge_commit_sha`; per-member lines are not separable, so members share the merge
  commit's lines and modules (consumers pairing cost with lines aggregate per `unit_id`).
- Chain units are not formed: `merge_verify.$.chain_items` is a count, not member ids, and 4 of the 5
  chain-verify shas never reached main. A chain landing is a unit of one.
- Lines are measured against a typed `lines_base`: a merge's first parent; for a single-parent
  (fast-forward) landing, the nearest older first-parent commit that is itself a landing (its own first
  parent is the task's previous commit); off the first-parent line, unmeasurable (None).
- The signature is `landings(runs_db, repo, since_cursor, *, project_id=None, ...) -> LandingBatch(landings,
  git_check)`; `project_id` defaults to the repo root's project id (its basename), and the batch carries
  the git cross-check counts. `since_cursor` has no default; `None` means the full history (every
  landing in the store, ordered by landed_at then merge_sha), the promise
  `plans/instruments-honesty-prd.md` decision 2 relies on.
- (2026-11-09) A **late claim** is a `merge_finalized done` row naming a sha that an earlier event
  already claimed for another task. The full df history has 5: 2980@d397ce031f (first claimed by 2913),
  5267 and 6042@100e526e15 (5963), 5149@0c8596de4c (4152), 6677@18fd0e0047 (3723). For 5149, 5267
  and 6677, `git log -1 --format=%P` shows the task's own `Merge task/<id> into main` commit as the
  claimed sha's first parent, and no runs.db landing names that commit. 2980's own merge
  (95abbf6df8) is a separate landing. 6042's own merge (51fcedc570) is on main and no runs.db
  landing names it. A late claim stays a landing, because the task left the merge queue with main
  advanced. It forms its own unit of one (`late_claim`, with an opaque `unit_id` compared only for
  equality) with unmeasurable lines, and `git_check.late_claims` counts it in the disagreement. See
  `shared/src/shared/quality_series/landings.py::landings`.
- (2026-11-09) Late claims and off-first-parent landings are emitted by `landed_at`, after the newest
  first-claim landing at or below the cursor, so a later cursor may re-emit one. Consumers upsert on
  `(project_id, merge_sha, task_id)`. This replaced the original sketch's "never re-emits a sha ≤ cursor".
- (2026-11-09) **Which sha is the cursor.** A consumer resumes from `LandingBatch.next_cursor`: the
  merge_sha of the newest first-claim landing emitted, or the call's cursor in full when none was. It
  must never store main's HEAD. A landing's runs.db event trails main's ref update by 38 s to 3m52s
  (measured above), so with a 600 s loop a commit at or below HEAD is routinely not yet a recorded
  landing, and a HEAD cursor would skip it for good. Until its event is written, such a merge counts in
  `git_check.merges_without_landing`, so that count is transient at the top of main. The Sketch's
  "stale when main is ahead of the cursor" therefore means recorded landings above the cursor, not
  first-parent commits. See `shared/src/shared/quality_series/landings.py::landings`.
- (2026-11-09) A unit's members are consecutive in ordinal order, and
  `shared/src/shared/quality_series/rows.py::rows_for` refuses input that splits, shortens or repeats
  a unit, before any git work.
- (2026-11-09) `train_merged` has two payload shapes. A coalesced train carries `member_task_ids` and
  `merge_commit_sha`. A member that passes solo after the train's union verify fails
  (`orchestrator/src/orchestrator/workflow.py::_attribute_train_failure`) carries `verdict =
  'attributed'` and `merge_sha` under the member's own `task_id`. It advanced main alone, so it is a
  landing with source `train_attributed` and is its own unit. Any other shape raises
  `UnrecognisedTrainEvent`, which names the event ids; the store is never read past it. The full df
  history had 0 attributed events on 2026-11-09.

### Per-landing row (`shared/src/shared/quality_series/rows.py`)

```
LandingRow = Landing + {
  outcomes: {rejected: bool|None, rejection_events: int|None, reviewed: bool|None,
             review_blocking: bool|None, flake_confirmed: bool|None, flake_real_red: bool|None,
             flake_rows: int|None, reverify_count: int|None, verify_failed_rows: int|None},
  cost:     {all_role_usd: float|None, by_role: ((role, usd), ...), implementer_invocations: int|None,
             execute_iterations: int|None, turns: int|None, turns_capped: bool, code_lines: int|None,
             src_lines: int|None, test_lines: int|None, review_rounds: int|None, review_rounds_source:
             'review_completed'|'disk_counter'|None, charter_class: str|None, charter_class_source: ...},
  modules:  [{module, src_lines}],
  rule_vintage: SourceEpochs   # one first-row timestamp per source, not per metric}
rows_for(landings, runs_db, repo, git_budget_s) -> list[LandingRow]
```
Invariants: `None` means "source not started or not applicable", never 0; `rejected`/`reverify_count`
are the unit's values copied to every member; `all_role_usd is None` iff the task has no `invocations`
row; `code_lines` follows decision 6's path classes; every window is `(prev_landing_time(task),
landed_at]`.

**Correction (α, 2026-11-09).** The block above was widened in place to the shipped shape. Every
outcome field, `implementer_invocations` and the three line fields are Optional, because one epoch rule
applies uniformly: a field is None when its source has not started at the landing (lines: when the
base is unmeasurable). So quality.db columns for them are nullable. `rule_vintage` is keyed by
source, not by metric: a consumer maps each metric to the source it reads. `by_role` is a tuple of
(role, usd) pairs. The single homes are `shared/src/shared/quality_series/rows.py::Outcomes`,
`::Cost` and `::SourceEpochs`, with their `__post_init__` invariants. `SourceEpochs` and the
Landing's `landed_at`/`prev_landed_at` are timezone-aware datetimes; ISO text is rendered only where a
row is written to quality.db or printed.
- (2026-11-09) The `rejections` epoch is the first categorised `merge_blocked` of any category, so a
  project that never has a post-merge rejection reads 0/n, not unknown. On df and reify that is the
  same instant as the first `post_merge_verify` rejection (2026-07-19T17:49:40Z and
  2026-07-20T02:18:42Z), so neither series moves. The `flake_occurrence` epoch stays the table's first
  row, and the table records only discriminated failing tests, so it is outcome-conditioned. A
  project that never has one serves `flake_*` = None for every landing, indefinitely, and its series
  starts on a positive observation, which biases small projects (reify, the solar repos) upward.
  Render that as "no producer has written here", never as 0.
- (2026-11-09) `execute_iterations` sums the `task_results` rows of the dispatches that produced the
  landing (`shared/src/shared/quality_series/rows.py::_landing_dispatches`): the window's rows that did
  not report `done`, plus the landing dispatch's own row. `task_results` is written when a dispatch
  ends, after its merge lands, so that row is the task's first row after `landed_at`, counted when it
  reports `done`. A `done` row inside a window is the previous landing's dispatch row and is not
  charged again. On the 151 df W44 plain landings the dispatch row trailed the landing event by 12 s
  at the median and about 12 min at p99. The strict window alone never counted it: W44 had 73 of 234
  landings at None against 21 with no implementer invocation; with the dispatch row, 19. Two
  residuals. A row computed before its landing dispatch's row is written lacks it, so β must
  re-measure a stored row whose landing is younger than that lag, or accept the loss; decision 14's
  upsert allows the re-measure. And a landing whose own dispatch wrote no row can claim the task's
  next landing's `done` row, because `task_results` is keyed `(run_id, task_id)` with `INSERT OR
  REPLACE`: 7 of the 3,527 df landings in the full history (2026-11-09).

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

**Correction (α, 2026-11-09).** The shipped signatures differ from the list above in three places
that γ and δ build on; `shared/src/shared/quality_series/stats.py` is the single home.
- `cusum_design(p0) -> CusumDesign(p1_up, p1_down, h)` with `p1_up = min(2·p0, (1 + p0)/2)`: the
  doubling up to p0 = 1/3, and halfway from p0 to 1 above it, because 2·p0 reaches 1 at p0 = 1/2,
  where the Bernoulli log-likelihood ratio is undefined. `p1_down = p0/2`, `h = CUSUM_H = 5`, and a
  p0 outside (0, 1) is refused (`::cusum_design`).
- Markers reach the statistics as `::RegimeEvent(event_id, at, until=None, series=None,
  artefact=False)`. `series=None` applies to every series, and an artefact is a range `[at, until]`.
  δ's markers are converted to it. `regime_segments(landed_at, events, series) ->
  RegimeSegments(segments, excluded)` takes the landings' `landed_at` datetimes, not landing records.
  An event's boundary is the first landing at or after its `at`, and an artefact's landings are
  `excluded` (`::regime_segments`).
- `explain_alarm(onset_idx, onset_at, events, landed_at, k=25, days=3) -> event_id | None` takes the
  same `RegimeEvent`s plus the required `landed_at` sequence, which places each event on the landing
  axis. Among events within ±k landings or ±days, the nearest in landings wins, then the nearest in
  time, then the smallest `event_id` (`::explain_alarm`).
- The other return shapes are typed, never bare tuples of strings: `baseline_rate` returns
  `(p0, BaselineState | None)`, `block_rate_datum` returns `RateDatum`, and `before_after_rate` /
  `before_after_cost` return `Effect` / `CostEffect` or `NotYet`. `cost_fit(lines, costs, min_lines=5)`
  exposes the log–log fit that `reference_cost` evaluates.

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

**Correction (γ, 2026-11-10).** The shipped wire extends the block above, and it replaces two of
its shapes: the headline value and the `modules` entries. Its single home is
`dashboard/src/dashboard/data/quality_series.py::QualitySeries.to_wire`.
- Replaced: a headline's value is `{form: rate | k_of_n | mean | median, n, k, value, lo, hi}`,
  not `number|{k,n}`. `k` is the event count of a rate and null otherwise. The band is the Wilson
  interval for a rate, and a bootstrap seeded by the window's task ids for a count or a cost.
- Replaced: `modules` entries are flat `{module, landings, tasks, allocated_lines, allocated_usd,
  usd_per_100_lines, ranked}`. There is no `metrics` key.
- Added: each axis point carries `git_first_parent` and `late_claim`. Off-first-parent landings and
  late claims stay on the axis (sketch row 3).
- Added: each metric carries `kind` (`rate` | `count` | `cost`), so a chart type is never inferred
  from an id.
- Added: `before_after` entries carry `metric`.
- Added: `QualitySeries` carries a top-level `git_check = {off_first_parent,
  merges_without_landing, late_claims, disagreement}` taken from the newest value sample, or null.

"No producer has written here" (INV-13) is a structured state: `rule.measured_since` is null, the
points are null, and the headline is `unknown` with a reason starting with that phrase. A series
flips by data, with no code change. `charter_blocked` flips once any stored row carries
`charter_class_source = 'charter_class'`; a source flips once it has a `SourceEpochs` epoch.
`reverts` stays in that state until `late_events` has a writer. `review_rounds_ge2` and
`execute_iterations` have no source epoch: they are measured from the first stored landing that
carries the value. Until one does, the source has a producer, so the headline does not say "no
producer has written here". Its reason names what no stored landing carries, for example "no stored
landing of this project carries a review round count" (`::Since`).

Baseline, EWMA, CUSUM and the cost model run over the project's full stored history, and the window
only slices what is served. The headline is always the last-100-measured block
(`rule.window_landings = 100`), whatever the window chip says. Counts are per verify unit: the
value sits on the unit's head landing and is 0 on the other members. The cost model fits per-unit
points, at most 200 baseline units. `::validate_quality_series` refuses, with `VALUE_CONTRACT`, an
array whose length differs from the axis, an unordered axis, or an alarm off the served axis.

This amends decision 13's "absent below" for the baseline in force. A regime segment's own
baseline (p0, or the cost model) takes force once that segment has ≥100 measured landings (≥100
costed units for the cost model). Until then the previous segment's baseline stays in force, while
the CUSUM still resets at the marker. That keeps a shift just after a marker alarmed and explained
(sketch row 9). The served `baseline` and `cost_model` are the ones in force at the newest landing,
and `baseline.since_ordinal` is the start of the segment that measured it, which can be before the
newest marker. "Absent" means only that no segment has yet reached 100. Artefact-excluded landings
are dropped from the EWMA and CUSUM inputs. Homes:
`dashboard/src/dashboard/data/quality_series.py::_control_chart` and `::_cost_analysis`.

### Markers (`dashboard/src/dashboard/data/quality_markers.py`)

`derived_markers(runs_db, project_id, snapshots) -> list[Marker]` (decision 11 rules, idempotent, persisted
to `markers_derived` by the loop with a per-(project, role) last-seen key); `load_curated_markers(path)
-> (list[Marker], parse_failures)` fail-soft, unknown keys ignored, old entries valid with
`kind='regime'`, `scope='fleet'`, `series=None`; `resolve_ordinals(markers, axis)` sets `ordinal` to the
first landing at or after `at`.

**Correction (δ, 2026-11-11).** What shipped, and where each part is homed. Symbols prefixed `::`
are in `dashboard/src/dashboard/data/quality_markers.py`.
- `QUALITY_MARKERS` is `Datum[MarkerStream]`, not a bare marker list. Its value is `{markers,
  availability, prompt_coverage, curated_parse_failures, skipped_records}` (`::MarkerStream.to_wire`).
  `availability` holds the role-default fallbacks, served apart from the markers
  (`::AvailabilityFlag`: role, from_at, until_at, decisions, model). `prompt_coverage` is
  `{state: written | not_yet_written, since, roles_py_landings, counted_since, reason}`
  (`::PromptCoverage`). It is written from the first routing decision that carries a prompt digest.
  Until then it counts the first-parent `main` merges touching
  `orchestrator/src/orchestrator/agents/roles.py` in the factory checkout since the newest in-scope
  curated `prompt` entry, else since the project's first routing decision. If git cannot give a
  count, it gives the reason instead. The datum is unknown until the marker tick has derived the
  project once. It is stale while the routing fold is incomplete or the newest refresh failed
  (`::marker_datum`).
- `Marker` adds `until_ordinal` (the last landing at or before an artefact's `until`) and `tasks`.
  `scope` is the project id or `fleet`. `source` is `{type, ...the details that type carries}`
  (`::MarkerSource`). Ordinals are resolved at serve time against the project's full stored landing
  axis and never stored (`::resolve_ordinals`). The placement is
  `shared/src/shared/quality_series/stats.py::regime_segments`'s.
- Kind alone sets the regime role (`::REGIME_ROLE`). model, effort, budget, turns, prompt,
  reviewer_policy, routing_rule, cli and regime are boundaries. artefact is an exclusion range.
  deploy and instrument are not landing-axis events. Derived markers carry `series = null`, so each
  is a regime event for every series. The route passes the stream's `regime_events()` to
  `read_quality_series` (`dashboard/src/dashboard/api/quality.py::api_quality`). Before the
  project's first derivation, when `QUALITY_MARKERS` is unknown, it passes the curated markers in
  the project's scope instead, so a curated artefact is excluded from the start
  (`dashboard/src/dashboard/api/quality.py::_regime_events`).
- Steps (`::fold_routing_decisions`): a role's steps are run-length encoded from its decisions
  with `source_layer ∈ {config, role_default}` that no policy rule matched (`rule_id` null).
  `orchestrator/src/orchestrator/routing.py::resolve_route` applies a matched rule's effort,
  budget and turns even when it refuses the rule's model, which leaves the layer `config` or
  `role_default`. Such a row feeds only its rule's span. A row naming a layer outside
  `::SourceLayer` is skipped and counted.
- Persistence (decision 11, in `::_persists` and `::_counted_groups`): a non-start step counts iff
  it has ≥3 decisions and either spans ≥24 h, or the next step changes a field it did not. The span
  runs to the next step's first dispatch, or for the open step to the fold's newest decision.
  Role-default runs bracketed by equal config steps are removed first and served as availability
  flags. Kind is the first changed field by precedence model > effort > prompt > budget > turns,
  and the label lists every changed field. An absent prompt digest never splits a step.
- Dating (`::_cause`): a routing marker is dated at the earliest `config_reload` in the same
  project's runs.db that names a changed field's config path, within 48 h before the first
  dispatch. With no such reload it is dated at the first dispatch. A git commit is never a dating
  source. Only an applied `review.reviewer_briefing` change is a standalone marker
  (`reviewer_policy`). A policy rule's disappearance is a marker once the rule has not fired for
  `::RULE_QUIET_DAYS` (7) at the fold's newest decision.
- Store: the "per-(project, role) last-seen key" is the stored routing fold. quality.db holds it in
  `routing_steps`, `routing_rules` and `marker_state`. `marker_state` carries the cursor, the fold
  bounds, digest_since, completeness, the skip counts, the prompt coverage and
  refreshed/attempted/failure. `markers_derived` is reshaped to `(project_id, marker_id, kind, at,
  label, source, ref)` with no ordinal column. `_QualityStore.open()` drops and recreates the β
  table (`dashboard/src/dashboard/data/quality_store.py::ensure_marker_tables`).
- Tick: `dashboard/src/dashboard/data/quality_marker_sampler.py::refresh_markers` runs after
  `collect_quality` in `dashboard/src/dashboard/loops.py::_quality_loop`. Each tick folds at most
  `MARKER_ROUTING_BATCH` (500,000) routing rows per root and re-derives every marker from the fold
  plus the small sources. Measured 2026-11-12, df's 27,152 routing rows fold in 0.32 s, so a cold
  start completes in one tick. The tick resumes from the stored fold alone
  (`dashboard/src/dashboard/data/quality_store.py::read_routing_fold`). A stored fold that no
  longer decodes is folded again from the start, and that root's write replaces it. Both quality
  ticks share their per-root machinery (`dashboard/src/dashboard/data/quality_root_tick.py`).
- `load_curated_markers(path)` returns `CuratedMarkers(markers, parse_failures)`. `kind` takes the
  whole marker vocabulary. `until` (artefact only) is the inclusive end of its UTC day. Every
  rejected entry is counted. `derived_markers(runs_db, project_id, snapshots)` is the stateless full
  derivation, equal to folding in batches.
- The instrument rule (f) takes `SnapshotIdentity` values (`::instrument_markers`). The loop passes
  none until ι supplies a snapshot reader.

### Orchestrator fields

- `task_results.charter_class TEXT NULL` (additive column with `CREATE TABLE IF NOT EXISTS` + `ALTER`
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
| 14 | Block class written | an architect blocks a task as unactionable in a dispatched run | `task_results.charter_class='under_specified'`, the `task_blocked` event and the escalation carry it; the charter series' next row reads it with `source='charter_class'` |
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
- **ζ — Orchestrator: typed `charter_class` and `prompt_digest`.** Modules:
  `orchestrator/src/orchestrator/workflow.py` (the four architect block sites), `event_store.py`
  (`task_results.charter_class`, event payload), `escalation` record detail, `routing_dispatch.py`,
  new `agents/prompt_digest.py`, tests. Signal: in a dispatched run on main, an architect block writes
  `charter_class` to `task_results`, the `task_blocked` event and the escalation; every new
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
verbatim reason; INV-13 readers render "no producer has written here" for `charter_class` and `prompt_digest`
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

1. **`train_merged.timestamp` as main's advance.** Resolved by α (2026-11-04): it is main's advance in
   the same sense as `merge_finalized done`. main's reflog reached the train merges 0f0fbc44ae and
   dd31d2ad39 2m56s and 3m37s before their `train_merged` events, and the plain landings 9f7cc8091d,
   1ea166355e and 6c42ed815e 3m52s, 2m54s and 38s before their `done` events: both event types lag the
   ref update by the same few minutes. The fallback named here does not exist (members have no `done`
   row). The event timestamp anchors both sources.
2. **`invocation_end.$.turns` across resumed sessions.** Resolved by α (2026-11-04): summed over every
   row. `_parse_claude_output` reads `num_turns` raw, and consecutive resumed (`session_delta`)
   processes of one task and role report non-monotone turns (task 5836 steward 13, 5, 2; task 7977
   steward 18, 9, 16, 6), which a cumulative counter cannot do. Per-process is a strong hypothesis, not
   proven per session: `invocation_end` carries no session id.
3. **Per-merge git cost** (inferred ~30 ms, unmeasured): β measures it and sets the per-root budget from
   the measurement. Decide during β.
   **Resolved by β (2026-11-09)**, measured on the df main checkout at load ~30 on 32 cores through α's
   public functions: `git rev-list --first-parent --parents` 0.23 s; a full-history `landings()`
   0.57 s over 3,545 landings; `landed_lines` per unit median 13.7 ms, p90 44.8 ms, p99 137 ms, max
   145 ms over the newest 200 units (oldest 200: median 16.8 ms, max 67 ms); a cold `rows_for` at a
   60 s budget measured 3,112 of 3,545 rows in 62.2 s, then the remaining 433 in 7.5 s; reify's full
   history (2,549 landings) took 40.1 s; max RSS 134 MB. Among 3,539 df first-claim on-line landings,
   ordinal (landed_at) order and first-parent position order show 0 inversions. The choices, in
   `dashboard/src/dashboard/data/quality_sampler.py`: `QUALITY_GIT_BUDGET_SECONDS` = 30 (at least 219
   units even at p99, against df's ~1.6 landings per tick, so a steady-state tick is never truncated;
   it also bounds how long a dashboard shutdown waits on an in-flight worker thread);
   `QUALITY_REV_LIST_TIMEOUT_SECONDS` = 30; and the per-root backstop
   `QUALITY_PER_ROOT_BUDGET_SECONDS` = 180 s, between its floor (rev-list 30 +
   `FIRST_UNIT_GIT_FLOOR_S` 60 + runs.db reads of at most ~8 s) and its ceiling (half the 600 s tick,
   300 s). A cold backfill takes ≈4 ticks for df and ≈3 for reify.
4. **Module key for non-`<pkg>/src/<pkg>` layouts.** Resolved by α (2026-11-04): `<pkg>/src/<pkg2>/<rest>`
   keys as `<pkg>/<rest[0]>`; otherwise the first two components at depth ≥ 3, else the first
   (`crates/reify-core/src/lib.rs` → `crates/reify-core`). See
   `shared/src/shared/quality_series/git_lines.py::module_key`.
5. **Whether a 7-minute nightly run on the laptop is preferable** when the host is saturated: θ ships the
   workstation timer; a laptop instance is a config choice later.
6. **CPUWeight value for the timer unit**: θ picks the lowest value with a precedent in
   `orchestrator/src/orchestrator/verify.py::_resolve_scope_cpu_weight`.
