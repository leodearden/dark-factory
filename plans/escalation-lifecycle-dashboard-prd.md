# PRD: Escalation lifecycle dashboard — origin, lifespan, resolution workflow

Status: active · authored 2026-07-15 (interactive /prd session, seeded by the
2026-07-15 escalation-ladder audit brief) · B+H (contracts + two-way boundary
tests)

## Goal

The operator can answer three questions from the dashboard without reading
JSON or digests:

1. **Origin** — who files escalations, at what rate, and which sources are
   *predictably benign* (the anchor principle: "escalations shouldn't be
   benign, at least not predictably").
2. **Lifespan** — how long escalations live from filed to resolved, which
   pending items breach the 6h freshness contract, and how long L1→L2
   promotion takes.
3. **Resolution workflow** — which tier absorbs the work (human / cascade /
   auto-watcher / steward / reaper-sweep / other-auto), the action mix, the
   re-escalation churn rate, and escalations-per-done.

Concretely: a user opens a new **analytics tab** and sees three panels
(Origin / Lifespan / Workflow) rendered from the escalation archive, plus a
**StatTile summary strip** on the existing Escalations tab. Going forward,
every resolution can carry an explicit **`resolution_class`** stamp
(`benign` | `actionable`), so the anchor metric stops being a proxy.

## Background

- 2026-07-15 audit over 1,439 archived records (~4 weeks,
  `data/escalations/archive/`): 70% closed benign/no-action; per-source
  benign rates fused-memory 98%, architect 96%, main-sweep 94%,
  starvation-watchdog 94%, orphan-reaper 91%; auto-watcher-resolved lifespan
  median 29 min / p90 ~6h; 45% same-task re-escalation within 24h of an auto
  resolve; tier absorption: humans 16% direct + 23% via cascade, auto-watcher
  15%, other automation 46%.
- All archived records carry the full lifecycle field set (`agent_role`,
  `timestamp`, `resolved_at` on 1,438/1,439, `resolved_by`,
  `resolution_action`, `level`, `severity`, `category`, `members`,
  `dedupe_*`). Verified 2026-07-15.
- In-flight changes will reshape these distributions (2629, 2630, 2631,
  2633, 2640, 2555, verify-scope flip 2588–2594): the charts must make
  before/after legible (regime markers) and tolerate schema growth (2555
  triage fields render-when-present).
- Existing dashboard substrate: hand-rolled chart primitives in
  `dashboard/src/dashboard/static/redux/charts.jsx` (LineChart,
  StackedAreaChart, BarChart, HBarChart, Donut, StatTile, Heatmap, HistBar,
  Sparkline) — the no-graph-library constraint holds with zero additions.
  Archive-walking precedent: `dashboard/src/dashboard/data/performance.py`
  via `escalation.queue.iter_all_escalation_paths`. The queue-view module
  `data/escalations.py` deliberately reads the live root only and is not
  modified by this PRD.

## Sketch of approach

- **α (escalation pkg)** adds a nullable `resolution_class` field written at
  the queue resolve/dismiss chokepoint, with per-path defaults for
  definitionally-predictable automated closes, cascade inheritance for L2
  member closes, and submit-boundary validation. It also houses the shared
  classification helpers (`classify_resolver_tier()`, `effective_benign()`)
  so dashboard and (future) digest consume one site.
- **γ (dashboard backend)** adds `data/escalation_analytics.py`: per-request
  archive walk behind a short TTL cache, producing daily-bucketed
  origin/lifespan/workflow aggregates per project root, `parse_failures`
  surfaced in-payload, regime markers loaded from a committed YAML file, and
  a new endpoint `/api/v2/dashboard/escalation-analytics`.
- **δ/ε/ζ (frontend)** add the analytics tab (three panels + regime-marker
  annotations), the Escalations-tab StatTile strip, and the hand-rolled
  lifecycle flow diagram (mini-Sankey), all reading the one endpoint.
- **β** wires the auto-watcher (and 2630's evidence-gated allowlisted closes)
  to pass `resolution_class` explicitly.
- **η₀–η₃ (deterministic predicate tasks)** run
  `scripts/check_esc_analytics_perf.sh` against the live endpoint:
  immediately after the implementation lands, then at +30d/+90d/+180d via
  `metadata.milestone` delayed anchors. The script is committed with this
  PRD (submit-time validation requires it on main at decompose).

Dashboard deployment: dashboard changes auto-deploy deterministically on
every commit (existing infra; asserted by the operator 2026-07-15). No
deploy task is needed; the perf checks anchor on the implementation tasks.

## Resolved design decisions

1. **Benign definition = explicit stamp + proxy backfill.** New nullable
   `resolution_class ∈ {'benign','actionable'}` stamped at resolve time;
   history and unstamped closes fall back to the proxy
   (`dismissed→benign`, `resolved→actionable`). The UI always shows the
   stamped-vs-inferred split, so proxy share visibly decays.
2. **Stamp param is optional everywhere.** No existing caller breaks;
   adoption pressure comes from the visible inferred-share. Automated
   definitionally-benign closers get hardcoded defaults (table below).
3. **Placement = hybrid.** Full analytics tab + a 4-tile StatTile strip on
   the existing Escalations tab.
4. **Backend = per-request archive walk + TTL cache** (performance.py
   precedent), with an immediate perf check and 1/3/6-month deterministic
   re-checks (regression tripwires, generous thresholds — they alarm on
   order-of-magnitude regressions, not micro-perf).
5. **Regime markers = committed YAML** (`dashboard/regime-markers.yaml`),
   hand-curated when behaviour-reshaping changes land; served through the
   analytics endpoint; rendered as vertical annotations on time charts.
6. **No deploy task** — auto-deploy-on-commit covers delivery.
7. **Out-of-scope metrics stay out** — rubber-stamp fraction at L2 and
   stale-RCA rot are not mechanically producible from record fields (the
   audit derived them by reading prose); recorded under Out of scope with
   their future substrate named.
8. **Shared helpers live in the escalation package** (INV-5): benign
   predicate + resolver→tier table are code in one site, imported by the
   dashboard aggregator; the digest's future per-source benign column
   consumes the same helpers.
9. **Read-only surface.** Consistent with the escalation-queues dashboard
   section decision (memory, 2026-05-27): no resolve/dismiss/promote
   affordances; interactive resolution stays with the escalation-watcher.
10. **Display forms (resolved in the 2026-07-15/16 discussion):**
    - *Origin*: StackedAreaChart of filings/day by source as primary, plus a
      per-source trend **sparkline column** in the benign-rate table (an
      individual source's regime change stays legible inside the stack).
      Benign-rate table sorted by benign *count* (volume × rate — "who
      wastes the most closes").
    - *Lifespan*: **ECDF**, log-x, overlaid by resolver tier, with a
      vertical marker at the 6h freshness contract — replaces the
      histogram. Percentile StatTiles stay split by level; the curve
      carries the tier split (no level×tier matrix).
    - *Workflow*: tier absorption as a **100%-normalized** stacked area
      with a small total-volume sparkline above it (normalization tells the
      absorption story; the sparkline keeps shrinking load visible).
    - *Workflow*: a hand-rolled **lifecycle flow diagram (mini-Sankey)** —
      columns `origin → final level → resolver tier → benign/actionable`,
      aggregated over the selected window; ribbons are bezier bands between
      column rectangles. The one novel component in this PRD; own leaf (ζ).
      Action mix stays a separate small donut (it is not the Sankey's
      terminal column — class is, per the anchor principle).
    - *Cut*: source×category heatmap — weakest space-per-insight; the
      Sankey's origin ribbons carry most of that reading.

## Pre-conditions

- Milestone machinery (`shared/src/shared/task_metadata.py::Milestone`,
  `before_done.kind='predicate'`, delayed anchors) — **on main, verified
  2026-07-15**.
- `before_done.script` submit-time existence+executable validation
  (`fused_memory/middleware/deterministic_task_guard.py`) — verified; hence
  the perf script ships with this PRD.
- Dashboard auto-deploy on commit — existing infra (operator-asserted).
- 2630 (auto-watcher evidence-gated close_only) — gates **β only**; α, γ, δ,
  ε, η do not wait on it.
- 2555 (triage fields) — gates nothing; the aggregator treats `triaged_at`
  as optional and the Lifespan panel renders the filed→triaged→resolved
  segments only when the fields appear.

## Cross-PRD relationship (G4)

| Other PRD / system | Direction | Mechanism at the seam | Integration owner |
|---|---|---|---|
| dashboard-taskgraph-legibility (2526–2530) | sibling, no data seam | shared tab shell + chart primitives only | each PRD its own views |
| 2630 auto-watcher benign closes | this PRD **consumes** 2630's close path | allowlisted close passes `resolution_class='benign'` | **β here** (dep on 2630) |
| 2555 triage-stamp fields | this PRD **consumes** (optional) | `triaged_at`/`triaged_by` read by aggregator when present | **γ/δ here** (render-when-present; no dep) |
| escalation record schema | this PRD **extends** | `resolution_class` field + chokepoint stamping | **α here** |
| digest per-source benign column (unfiled) | future consumer of α's helpers | `classify_resolver_tier()` / `effective_benign()` imports | future PRD; helpers land in α |
| confusion-reduction census (2572–2587) | none | — | — |

## Contract section (B+H)

### Seam 1 — escalation record schema (writers ↔ readers)

**Field**: `resolution_class: 'benign' | 'actionable' | null` (absent/None on
all pre-existing records). Written **only** at the queue resolve/dismiss
chokepoint; never author-supplied at filing time. Unknown values are
rejected at the chokepoint with a ValidationError naming the two legal
values (INV-1); nothing is persisted on rejection.

**Semantics**: `benign` = no action was needed and none was taken beyond
closing; the filing was predictably ignorable. `actionable` = the escalation
led to a real action or decision (fix, requeue, restart, design ruling,
park). Classification describes the *escalation's* usefulness, not the
resolver's effort.

**Per-path stamping defaults** (applied by the chokepoint when the caller
passes nothing):

| Close path | Default |
|---|---|
| auto-dismiss age-out / supersede sweep | `benign` |
| orphan-reaper drop | `benign` |
| starvation-watchdog self-clear | `benign` |
| l2-cascade member close | inherit parent L2's class |
| 2630 allowlisted evidence-gated close | `benign` (β wires explicitly) |
| `resolve_issue` (human / watcher), any level | caller's optional param; else `null` |

**Effective-benign predicate** (single helper in the escalation pkg):
stamped → the stamp; unstamped and `status='dismissed'` → benign
(provenance `inferred`); unstamped and `status='resolved'` → actionable
(provenance `inferred`); `pending` → excluded from benign math. All
aggregates carry stamped/inferred provenance counts.

**Resolver→tier classification table** (single helper; unknown values fall
to `other-auto` which is surfaced as its own chart segment so growth is
visible, INV-4):

| Tier | `resolved_by` patterns |
|---|---|
| human | `interactive`, `escalation-watcher` |
| cascade | `l2-cascade:*` |
| auto-watcher | `escalation-watcher-auto`, `orchestrator-escalation-watcher-auto` |
| steward | `claude-task-*-steward` |
| reaper-sweep | `harness-orphan-reaper`, `auto-dismissed`, `harness-escalation-revalidation-sweep`, `orchestrator-starvation-watchdog` |
| unknown | `null` |
| other-auto | anything else |

### Seam 2 — analytics endpoint (backend ↔ frontend)

`GET /api/v2/dashboard/escalation-analytics` → top-level shape (exact leaf
fields refined at implementation; top-level keys and semantics are the
contract):

```
{
  generated_at: iso8601,
  parse_failures: int,              # skipped-record count this scan (INV-4)
  regime_markers: [{date, label}],
  per_project: [{
    project: str,                   # subsection-per-root, primary first
    origin:   { daily_by_source, sources: [{source, filings, benign,
                actionable, stamped_share, benign_rate, predictably_benign,
                daily_spark: [int]}] },
    lifespan: { percentiles_by_level, l1_to_l2_promotion,
                samples: [[date, tier, level, secs], ...],  # resolved only;
                          # frontend builds the ECDF + windows client-side.
                          # Bounded: ~1.4k today; server-side downsample
                          # (stratified by tier) once the archive passes ~10k.
                open_items: [{id, task_id, level, age_secs, breach_6h}],
                triage_segments? },   # present only once 2555 fields exist
    workflow: { tier_weekly, action_mix, churn_daily, esc_per_done_daily,
                flow_daily: [{date, source, level, tier, class, n}] },
                          # sparse 4-dim daily cube; frontend sums over the
                          # selected window to feed the mini-Sankey.
  }],
}
```

- Daily buckets over the full archive; the frontend windows client-side
  (7d / 28d / all toggle).
- `predictably_benign` flag rule: benign_rate > 0.9 **and** n ≥ 20 in the
  trailing 28d window.
- Churn: a filing counts as churn if the same `task_id` files again within
  24h after any resolve/dismiss of a prior escalation for that task.
- `esc_per_done`: filings per day ÷ done-transitions per day from
  `runs.db` — recomputed here over the same buckets; the digest's
  process-local EWA state is **not** coupled to (its math may be reused).
- TTL cache (~60s) in front of the archive walk; `iter_all_escalation_paths`
  for the two-tier scan; corrupt JSON skipped **and counted** in
  `parse_failures`.
- Regime markers: `dashboard/regime-markers.yaml`, list of
  `{date: YYYY-MM-DD, label: str, tasks: [int]}`. Malformed file → empty
  markers + a `parse_failures` increment (never a 500). Seed entries at γ:
  2631 starvation-threshold raise, 2630 auto-watcher benign closes, 2593
  verify-scope flip — dated by their landing dates.

### Seam 3 — perf predicate script (committed with this PRD)

`scripts/check_esc_analytics_perf.sh` — read-only predicate
(`before_done.kind='predicate'` contract: exit code only; stdout tail
carried as note). Args: `--url` (default
`http://127.0.0.1:8080/api/v2/dashboard/escalation-analytics`),
`--threshold-ms` (default 2000), `--attempts` (default 5 — median-of-5
absorbs busy-worker scheduling noise observed under fleet load, where one
request blocked ~10s while warm requests were sub-ms; a false-tripping
tripwire would itself be a predictably-benign escalation source). Prints one
structured line `measured_median_ms=… attempts=… threshold_ms=… url=…`
(INV-2) and exits 0 iff the median is within threshold; connection failure
exits 1 loudly. The URL default pins γ's endpoint path — changing the path
means updating the script in the same change (INV-5 note: the path lives in
two places by necessity — script default and app route — the boundary test
pins them together).

## Boundary-test sketch (B+H)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | `resolve_issue` with `resolution_class='actionable'` | pending L1 | archived record carries the stamp; aggregator counts it actionable/stamped |
| 2 | L2 cascade close with parent classed `benign` | L2 with 2 members | both member records archive with `benign`, provenance stamped-inherited |
| 3 | `resolve_issue` without the param | pending L1 | record has `resolution_class=null`; aggregator buckets by proxy, provenance inferred |
| 4 | chokepoint rejects unknown class | resolve with `class='meh'` | ValidationError naming legal values; record unchanged, still pending |
| 5 | age-out auto-dismiss | stale L0 | archived with `benign` default stamped |
| 6 | endpoint golden fixture | mini-archive fixture (stamped + unstamped + corrupt file) | payload matches expected aggregates; `parse_failures==1`; stamped/inferred splits correct |
| 7 | synthetic-scale perf | ~10k-record synthetic archive | cold aggregation completes under threshold (suggested 5s); test lives in γ's suite |
| 8 | perf script pass/fail/absent | live or mock endpoint | exit 0 under threshold; exit 1 over; exit 1 + message on connection-refused |
| 9 | regime markers malformed | broken YAML committed to fixture | endpoint serves `regime_markers: []`, `parse_failures` incremented, no 500 |
| 10 | 2555 forward-compat | fixture records with `triaged_at` | `triage_segments` present and correct; absent-field archives yield no key, no error |
| 11 | flow cube consistency | golden mini-archive fixture | summing `flow_daily` over any window reproduces the same totals as `sources`/`tier_weekly` marginals; `lifespan.samples` count == resolved-record count |

Frontend wiring is verified by Python source-assertion tests
(`dashboard/tests/test_tab_*.py` convention — no JS runner in this project).

## Decomposition plan

Greek labels; ids assigned at decompose. All tasks `planning_mode=True`;
each leaf self-declares its execution path.

- **α — Add `resolution_class` stamping to the escalation queue chokepoint**
  (escalation pkg). Field + validation + per-path defaults + cascade
  inheritance + `classify_resolver_tier()` / `effective_benign()` helpers.
  Signal: boundary tests 1–5 green; `resolve_issue` round-trips the stamp to
  the archived JSON. Prereqs: none.
- **β — Auto-watcher passes `resolution_class` on its closes**
  (orchestrator/watcher). Allowlisted evidence-gated closes stamp `benign`;
  other watcher closes pass the class its triage concluded. Signal: a
  watcher-closed escalation archives with a stamped class (no inferred
  provenance). Prereqs: α, external 2630.
- **γ — Escalation-analytics backend aggregator + endpoint** (dashboard).
  `data/escalation_analytics.py`, TTL cache, tier/benign helpers imported
  from α, regime-markers loader + seeded `dashboard/regime-markers.yaml`,
  `parse_failures` surfacing, `/api/v2/dashboard/escalation-analytics`
  route. Signal: boundary tests 6, 7, 9, 10 green; endpoint returns the
  contract payload against the real archive. Prereqs: α.
- **δ — Analytics tab frontend** (dashboard static). Origin / Lifespan /
  Workflow panels from existing chart primitives: stacked-area filings +
  benign-rate table with sparkline column (Origin); tier-overlaid log-x
  ECDF with 6h marker + percentile tiles + open-items list (Lifespan);
  normalized tier-absorption area with volume sparkline + action-mix donut
  + churn + esc-per-done (Workflow). Regime-marker verticals; 7d/28d/all
  window toggle; stamped-vs-inferred split visible on the benign views.
  Reserves the Workflow-panel slot where ζ's flow diagram mounts. Signal:
  new tab renders all three panels from the live endpoint;
  source-assertion tests green. Prereqs: γ.
- **ζ — Lifecycle flow diagram (mini-Sankey)** (dashboard static). New
  hand-rolled component: columns origin → final level → resolver tier →
  benign/actionable; ribbon widths from summing `flow_daily` over the
  selected window; no graph library. Signal: flow diagram renders in the
  Workflow panel with ribbon widths matching the golden fixture's flow
  sums; source-assertion tests green. Prereqs: γ, δ.
- **ε — Escalations-tab StatTile strip** (dashboard static). Four tiles:
  benign-rate 7d (with stamped share), pending 6h-breach count, esc-per-done
  7d, churn-24h 7d. Signal: strip renders on the Escalations tab from the
  same endpoint. Prereqs: γ.
- **η₀ — Immediate perf predicate** (`task_kind='deterministic'`,
  `before_done.kind='predicate'`, script above, no milestone). Signal: task
  goes `done` with `done_provenance.kind='deterministic-milestone'` carrying
  the measured line, or files `milestone_check_failed` at L2. Prereqs: γ, δ, ε.
- **η₁ / η₂ / η₃ — Delayed perf re-checks** (same deterministic predicate,
  `metadata.milestone = {mode: 'delayed', after_secs: 2592000 / 7776000 /
  15552000}` — 30/90/180 days from the implementation landing). Signal: as
  η₀, at each horizon. Prereqs: γ, δ, ε.

## Out of scope for this PRD

- **Rubber-stamp fraction at L2** and **stale-RCA rot** — derived in the
  audit by reading resolution prose; no mechanical producer exists. Future
  substrate: 2555's `triage_note`, or a resolve-time secondary
  classification. Pending-age (shipped here) is the honest proxy for rot.
- **Digest surfacing in the dashboard** (rendering `data/digests/`
  markdown) and the digest's per-source benign-rate column — future work;
  α's helpers are built to be its producer.
- **Interactive resolution affordances** — stays with the
  escalation-watcher (standing decision).
- **Reconciliation-queue (8103) analytics** — this PRD covers orchestrator
  escalation queues; the recon queue's lifecycle differs (sole-closer
  watcher) and can reuse the aggregator later if wanted.
- **Source×category heatmap** — cut in the display discussion (weakest
  space-per-insight; the flow diagram's origin ribbons cover most of it).
- Retro-stamping historical records (proxy covers them; no rewrite sweep).

## Open questions (surfaced but not decided in this session)

1. **L0→L1 promotion linkage.** `promote_to_l2` carries `members` (L1→L2 is
   computable), but the orphan-reaper files a *new* L1 quoting the L0 in
   prose — whether a machine-readable L0 id link exists is unverified.
   **Suggested resolution:** verify during γ; if absent, fuzzy-link by
   `task_id` + temporal adjacency or drop the L0→L1 timing sub-metric.
   Decide in γ.
2. **Perf thresholds.** Suggested: synthetic 10k cold aggregation < 5s
   (test in γ); live endpoint median-of-3 < 2000ms (η defaults). Confirm at
   η₀; tune the milestone tasks' args if the immediate check suggests
   different headroom.
3. **Default window** (7d vs 28d) and **tab label** ("Esc Flow" vs
   "Analytics"). Decide in δ.
4. **Strip tile pruning** — if four tiles crowd the Escalations tab header
   row, drop churn-24h first. Decide in ε.
5. **`resolution_class` in `get_pending_escalations` / MCP read surfaces** —
   expose immediately or lazily when a consumer asks. Decide in α.
