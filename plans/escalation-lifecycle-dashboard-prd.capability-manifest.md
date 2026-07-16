# Capability manifest — escalation-lifecycle-dashboard-prd

Per-leaf capability→evidence bindings (G3+G6 mechanized, paid once at
decompose 2026-07-16). Machine-readable twin:
`escalation-lifecycle-dashboard-prd.capability-manifest.yaml`.

## α — resolution_class stamping at the queue chokepoint

- **resolve-chokepoint-exists** — capability→producer (wired) —
  grep:escalation/src/escalation/queue.py:399 `def resolve(` (all
  resolution paths funnel through the queue). **PASS**
- **record-schema-extensible** — capability→producer (wired) —
  grep:escalation/src/escalation/models.py:50 `class Escalation`; optional
  lifecycle fields precedent at :68 `resolved_by`, :86 `resolution_action`.
  **PASS**
- **resolve_issue-accepts-optional-params** — capability→producer (wired) —
  grep:escalation/src/escalation/server.py:509 `def resolve_issue(`;
  optional-param precedent throughout signature. **PASS**
- **cascade-member-close-site-exists** — capability→producer (wired) —
  `resolved_by='l2-cascade:<id>'` attribution documented at
  escalation/src/escalation/models.py:85; cascade dispatch landed via task
  1620 (done). **PASS**

## β — auto-watcher passes resolution_class

- **allowlisted-close-path-exists** — producer:2630 upstream (**done**
  2026-07-15, verified via search_tasks at decompose) — evidence-gated
  close_only authority at L2 for `orchestrator-escalation-watcher-auto`.
  **PASS**
- **stamp-param-exists** — producer:task-α upstream (intra-batch dep wired).
  **PASS**

## γ — analytics aggregator + endpoint

- **archive-walk-helper** — capability→producer (wired) —
  grep:escalation/src/escalation/queue.py:91 `iter_all_escalation_paths`;
  consumption precedent dashboard/src/dashboard/data/performance.py:18.
  **PASS**
- **route-registration-surface** — capability→producer (wired) —
  grep:dashboard/src/dashboard/app.py:1327 `/api/v2/dashboard/escalations`
  route precedent. **PASS**
- **archive-fields-complete** — floor:coverage — 1,439/1,439 archived
  records carry agent_role/category/severity/level/status;
  1,438/1,439 carry timestamp+resolved_at (measured 2026-07-15). **PASS**
- **done-counts-producible** — capability→producer (wired) —
  grep:orchestrator/src/orchestrator/digest.py:278 `count_done_in_window`
  (runs.db task done-transitions); esc-per-done recomputed, digest EWA
  state not coupled. **PASS**
- **stamp-helpers-from-α** — producer:task-α upstream (intra-batch dep
  wired): `classify_resolver_tier()` / `effective_benign()`. **PASS**

## δ — analytics tab

- **chart-primitives-exist** — capability→producer (wired) —
  grep:dashboard/src/dashboard/static/redux/charts.jsx:101 `LineChart`,
  :161 `StackedAreaChart`, :302 `Donut`, :333 `StatTile`, :42 `Sparkline`.
  No new chart library required (ECDF = LineChart log-x; verticals =
  existing marker idiom). **PASS**
- **tab-registration-surface** — capability→producer (wired) —
  additive `window.DF_TABS` export precedent,
  grep:dashboard/src/dashboard/static/redux/tab_escalations.jsx:407.
  **PASS**
- **payload-from-γ** — producer:task-γ upstream (intra-batch dep wired).
  **PASS**

## ε — Escalations-tab StatTile strip

- **stat-tile-with-spark** — capability→producer (wired) —
  grep:dashboard/src/dashboard/static/redux/charts.jsx:333 `StatTile`
  (`spark` prop). **PASS**
- **payload-from-γ** — producer:task-γ upstream (intra-batch dep wired).
  **PASS**

## ζ — lifecycle flow diagram (mini-Sankey)

- **flow-cube-from-γ** — producer:task-γ upstream (intra-batch dep wired):
  sparse `flow_daily` cube in the endpoint payload. **PASS**
- **no-graph-library** — rejection-mechanism — hand-rolled layout is the
  standing dashboard constraint (task-graph PRD precedent, barycenter);
  delivered as an `expect: absent` check on graph-lib imports. **PASS**

## θ — integration gate

- **boundary-matrix-legs-produced-upstream** — DAG-direction — all legs
  produced upstream (α stamp, β watcher wiring, γ payload, δ/ε/ζ views);
  the task IS the check suite. **PASS**

## η₀–η₃ — perf predicates (immediate, +30d, +90d, +180d)

- **milestone-predicate-machinery** — capability→producer (wired) —
  grep:shared/src/shared/task_metadata.py:222 `class Milestone`, :58
  `kind: Literal['deploy', 'predicate']`; submit-time script validation
  fused-memory/src/fused_memory/middleware/deterministic_task_guard.py:304-314.
  **PASS**
- **perf-script-committed-executable** — capability→producer (wired) —
  scripts/check_esc_analytics_perf.sh, mode 100755, committed 17d839caba
  (before this batch was filed, satisfying the submit-time existence
  check). **PASS**
- **threshold-achievable** — floor:bound — warm endpoint responses measured
  sub-ms 2026-07-15; median-of-5 vs 2000ms threshold trips only on ≥3 slow
  attempts (regression tripwire, not micro-perf). **PASS**

No FAIL bindings; batch clear to file.
