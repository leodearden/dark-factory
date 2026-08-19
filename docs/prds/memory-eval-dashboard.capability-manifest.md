# Capability manifest — memory-eval-dashboard

Machine-readable twin: `memory-eval-dashboard.capability-manifest.yaml` (stem strictly
derived from the PRD path, locatable from `metadata.prd_path` alone). Batch authored +
decomposed 2026-07-29 in one session; task ids stamped by `commit_planning`; all three
leaves filed `planning_mode`. PRD committed at `c1261d5ed7`; the producer-side M2/M3
amendment (verdicts artifact + fingerprint carrier, epsilon=3211) committed at
`0ddaf23753` in the same session — that amendment *is* this manifest's biggest gate
outcome (a `producer-extent-short` resolved before filing rather than discovered at
dispatch).

11 capabilities across 3 leaves, **all PASS**, 8 carrying a mechanical (`grep`)
`delivered_check`; 3 recorded `kind: manual` and therefore excluded from the dispatch
gate.

| Leaf | Task | Load-bearing capabilities | Verdict |
|---|---|---|---|
| α backend reader + endpoint | (stamped) | M1/M2 exemplar fixtures from **3207 upstream**; verdicts read + escalation join reusing `load_queue_escalations` (INV-5, wired at `escalations.py:207-215`); TTL single-flight + config-property idioms verified; **no dashboard-side statistics** (mechanical `math.comb/lgamma` absent-check) | PASS |
| β frontend section | (stamped) | chart primitives wired via `index.html` load order; `data.js`/`index.html` registration checks; MemoryTab wiring + `—`-empty-state pinned by structural tests (manual) | PASS |
| γ parity integration gate | (stamped) | verdicts exemplar + `dedupe_fingerprint` stamping from **3211 upstream (amended)**; `dedupe_fingerprint` field verified on the model on main; both-directions parity + storm rendering pinned by integration tests | PASS |

## Bindings that needed work

- **Persisted verdicts (the `producer-extent-short` this PRD existed to catch).** The
  committed M2 contract persisted *limits* but not per-metric *verdicts*; without
  them the dashboard would have had to re-run statistics (G6 failure) or scrape the
  human report (INV-2). Resolved by producer-side amendment while 3211 was
  pending/unclaimed: M2 verdicts artifact (`verdicts-<STAMP>.json` +
  `verdicts-current.json` + `shared/tests/fixtures/` exemplar), mirrored into 3211's
  description and `metadata.delivered_checks` (verified by readback).
- **Fingerprint carrier.** M3's fingerprint had no pinned structured carrier on the
  filed escalation (the `has_open_l1` precedent keys on `task_id`, which is
  eval-granular here). Resolved in the same amendment: `dedupe_fingerprint` (existing
  model field, verified on main) carries the exact verdict fingerprint; parity is
  string **equality**, so composition stays ε-internal and the dashboard never parses
  fingerprint structure.
- **3207 extent.** α binds `producer:task-3207 upstream` for the M1/M2 exemplars; 3207's
  own task text pins the `shared/tests/fixtures/` exemplars and a "dashboard-shaped
  reader fixture" consumer-side boundary test, so the extent covers what α consumes.
  3207 is `in-progress`; the dep edge (not a grep) gates dispatch.

## Gate outcomes

- **G1** — the consumer is the operator UI itself; this PRD also closes the
  "commissioned dashboard PRD" consumer named by memory-eval-program §1.
- **G2** — α and β are intermediates naming their in-batch consumers; γ is the leaf
  and integration gate (C-as-integration-gate), its signal asserting the honest
  three-state parity (alarmed+open / recovered+open / storm-collapsed), not a naive
  bidirectional equivalence.
- **G3** — all dashboard-side substrate verified at file:line on main (PRD §5); the
  two future-substrate rows resolve via upstream dep edges (3207, 3211).
- **G6** — no signal asserts a numeric bound, exactness, or rejection beyond the
  parity assertions, whose premises were made true by construction via the producer
  amendment; the recovered-but-open state is explicitly modeled so the reverse parity
  direction is not a false premise.
- **G7** — all three tasks walked against INV-1..5; no waivers. The alarm/verdict/limit
  contract is JSON artifacts + a structured model field (INV-1); degraded reads are
  structured `issues`, loud in the UI (INV-2/INV-4); display-only, no acting on
  snapshots (INV-3 n/a); one escalation reader, one equality join, zero dashboard-side
  statistics (INV-5); staleness is displayed but its tripwire stays ε's (INV-4/5).
