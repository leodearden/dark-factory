# PRD: Memory-eval dashboard — trends + visual alarms on the escalation limits

**Project:** dark-factory (dashboard package only; consumes fused-memory artifacts). **Status:** active, 2026-07-29. **Approach:** bare B (single-package blast radius), consuming the B+H contracts of `docs/prds/memory-eval-program.md` (M1/M2/M3 as amended 2026-07-29, commit `0ddaf23753`).
**Origin:** Leo, 2026-07-29: "display the output from all the memory evals, including trends and visual alarm on the same limits that cause escalations." This is the **commissioned dashboard PRD** named in memory-eval-program §1 as the M1/M2 secondary consumer.
**Sibling:** `docs/prds/memory-eval-program.md` (filed as 3207–3214; 3207 in-progress at authoring). All substrate verified against main this session; file:line evidence in §5.

## 1. Goal (G1 consumer + user-observable surface)

An operator opens the dashboard's **Memory tab** and sees, for every artifact-emitting memory eval (E1 retrieval-health probe = 3208, E4 staleness sweep = 3209, E6 corpus-health series = 3210, write-after-miss = 3213 via ε's evaluation — and any future eval automatically, per DD5):

- the **trend** of each metric across run stamps (the M1 `metrics-<STAMP>.json` series),
- the **current value** and the **persisted verdict** — `alarm` / `no_alarm` / `insufficient_data` / `grandfathered` — exactly as the evaluator computed it (`verdicts-current.json`, M2 amendment),
- the **active limit and its provenance** (rule kind, baseline window ids, derived α, grandfather-set hash — `limits-current.json`),
- for alarms, a **link to the filed `eval_regression` escalation with the same fingerprint** (the recon queue the Escalations tab already renders), so visual alarms and filed escalations **cannot disagree by construction**.

Consumer: the operator UI itself — a user-observable surface (G1 satisfied directly); plus the Escalations tab's existing reconciliation subsection, which this PRD cross-links rather than duplicates.

## 2. Sketch of approach

Same-host file reads through the established dashboard aggregator pattern — no new MCP/HTTP surface on fused-memory:

- **Reader/aggregator** `dashboard/src/dashboard/data/memory_evals.py`: scan `<project_root>/fused-memory/data/memory-evals/` — per-`<eval_id>` dirs of `metrics-<STAMP>.json`, plus `limits-current.json` and `verdicts-current.json` at the root; join alarm verdicts to pending `eval_regression` escalations by `dedupe_fingerprint` **string equality** (loaded via the existing `load_queue_escalations(config.reconciliation_escalations_dir)` — one reader function, INV-5).
- **Endpoint** `GET /api/v2/dashboard/memory-evals` behind a ~60 s single-flight TTL cache (the escalation-analytics precedent), shaped by a new `shape_memory_evals` in `redux_api.py`.
- **Frontend**: a "Memory evals" section rendered inside the existing Memory tab; new `tab_memory_evals.jsx` (window-global export, loaded before `tabs.jsx` — the `scheduler_heatmap.jsx` pattern), `data.js` endpoint registration, `index.html` script tag. Existing chart primitives only (`charts.jsx`); no new chart library.

## 3. Resolved design decisions

- **DD1 — Read path is same-host file read** (vs a new fused-memory read endpoint vs static file serving). Memory-eval D2 pins artifact-only consumption; the dashboard already reads same-host files for escalation queues, runs.db, burndown.db, reconciliation.db; a new fused-memory read endpoint would add an API seam no second host needs; static file serving would push artifact-joining logic into the browser. The `DASHBOARD_FUSED_MEMORY_URLS` failover-only invariant (DESIGN.md) is untouched — this PRD adds **no** MCP fan-out. *Revisit trigger:* fused-memory moving off-host — at which point the dashboard's entire file-read layer moves together; this PRD adds no new coupling class.
- **DD2 — Alarm truth is the persisted verdicts + fingerprint-equality join.** The dashboard never re-runs a statistical test, never re-derives grandfather membership, never parses the human report, never parses fingerprint structure (M2/M3 amendments 2026-07-29). Parity is displayed honestly as three states: **alarmed + open escalation** (alarm badge, link), **recovered but escalation still open** (recovered badge, open-escalation link — the watcher hasn't closed it yet), **alarmed + storm-collapsed** (aggregate banner naming the alarm count, linking the single aggregate escalation).
- **DD3 — Placement: a section in the existing Memory tab**, not a new top-level tab. Memory-subsystem monitoring belongs with the memory panels; the section collapses when no artifacts exist yet ("no eval artifacts yet" placeholder, `—` for empty values — never synthetic data, per `feedback_redux_no_synthetic_data`).
- **DD4 — Separate endpoint with a ~60 s TTL cache.** Daily-cadence artifacts don't belong in the few-second poll payloads; the analytics-cache single-flight pattern bounds file-scan cost.
- **DD5 — Generic over eval_ids and metric_ids.** The section renders whatever M1-conforming artifact dirs exist — no per-eval hardcoding; a future eval that emits M1 artifacts appears with zero dashboard changes.
- **DD6 — Degraded states are loud, staleness is displayed but not re-alarmed.** Malformed/missing artifacts are counted and surfaced in a structured `issues` payload field rendered in the UI (INV-2/INV-4; the 2658 `parse_failures` precedent). Latest-run age is displayed with a stale hint, but alarm-on-staleness remains ε's runner-failure self-escalation — no duplicate tripwire (INV-5).
- **DD7 — θ (3214) is not displayed.** Its retro transcript corpus is one-shot validation-set substrate, not a monitoring series; per DD5 it appears automatically if some future eval derives M1 artifacts from it.
- **DD8 — Trend = metric values across run stamps; verdict-history timeline is out of scope.** Current verdicts only in this PRD; a verdict-over-time strip can consume the `verdicts-<STAMP>.json` series later without schema changes.

## 4. G5 note

Bare B: blast radius is one package (dashboard), <8 mechanisms, one downstream consumer (the operator). The high-stakes seam — artifact schema/limits semantics — is owned by the memory-eval PRD's H contracts; this PRD's γ leaf is the integration gate exercising both sides of that seam (consumer-side boundary tests over 3207's/3211's committed exemplars + the fingerprint-equality parity test).

## 5. Pre-conditions / substrate (G3 — verified this session, 2026-07-29)

| Assumed capability | Evidence |
|---|---|
| FastAPI app + aggregator layout; TTL single-flight cache precedents | `dashboard/src/dashboard/app.py:143-177` (task-cards cache), `:1356-1407` (analytics cache, `asyncio.to_thread`) |
| Config property pattern + recon queue dir already resolved | `dashboard/src/dashboard/config.py:156-220` (`_runtime_data_dir`, `reconciliation_escalations_dir`) |
| Recon (8103) escalations already render in the Escalations tab | `dashboard/src/dashboard/data/escalations.py:207-215` (reconciliation subsection); reader `:23-57` |
| Chart primitives | `dashboard/src/dashboard/static/redux/charts.jsx` (Sparkline/StepSpark/LineChart/BarChart/StatTile/Heatmap) |
| In-browser Babel + script-tag registration; MemoryTab exists | `static/redux/index.html:16-41`; `tabs.jsx:555` (`MemoryTab`) |
| Frontend structural-contract test idiom + backend TestClient idiom | `dashboard/tests/test_tab_escalations.py:1-6`; `test_app.py` et al. |
| `Escalation.dedupe_fingerprint` structured field | `escalation/src/escalation/models.py` (dataclass field, "content fingerprint") |
| `.jsx`/`.html`/`.js`/`.py` legal in `metadata.files` | `fused-memory/src/fused_memory/middleware/lock_charter_guard.py:107-120` (`FILE_EXTENSIONS`) |
| M1 metrics artifacts + committed exemplar fixtures (`shared/tests/fixtures/`) | producer **3207** (in-progress) — upstream dep of α |
| M2 limits artifact (`limits-current.json`) | producer **3207** — same dep |
| Verdicts artifact + escalation `dedupe_fingerprint` stamping + exemplar fixture | producer **3211** (pending; amended 2026-07-29, `0ddaf23753`) — upstream dep of γ |

Live artifacts will not exist until 3207–3211 land — expected; α/β build and test against 3207's committed exemplar fixtures, γ against 3211's; dep edges gate dispatch (G3 resolution (b)).

## 6. Cross-PRD relationship (G4)

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `docs/prds/memory-eval-program.md` | consumes | M1 metric-series artifacts + exemplar fixtures | other-prd (α=3207) | wired (dep α→3207) |
| `docs/prds/memory-eval-program.md` | consumes | M2 limits artifact + verdicts artifact; M3 `dedupe_fingerprint` carrier | other-prd (ε=3211, amended 2026-07-29 by this PRD's commission) | wired (dep γ→3211) |
| (in-repo, same dashboard) | reuses | `load_queue_escalations` recon-queue reader + Escalations tab display | dashboard (existing) | wired — cross-link only, never a second reader/closer |

Schema changes, if ever needed, are amendments to the memory-eval PRD (as exercised this session) — never a dashboard-side variant. The watcher remains sole closer of the 8103 queue; this PRD renders and links, never resolves.

## 7. Decomposition plan (signals are the G2 gate; Greek labels this-PRD-local)

Deps: α ← **3207** (external); β ← α; γ ← α, β, **3211** (external).

- **α — backend reader + endpoint (`dashboard/data/memory_evals.py`, config property, route, `shape_memory_evals`):** directory scan of `fused-memory/data/memory-evals/` (new `DashboardConfig.memory_evals_dir` property); per-metric trend series across stamps with a disclosed downsample cap (`truncated` flag — no silent caps); limits + verdicts read; fingerprint-equality join against pending `eval_regression` escalations via `load_queue_escalations`; structured `issues` for malformed/missing artifacts; ~60 s TTL single-flight cache; endpoint + shape fn. *Intermediate — unlocks β, γ.* *Signal:* TestClient `GET /api/v2/dashboard/memory-evals` over a fixture artifact tree returns trends/current values/verdicts/limits provenance/issues; the consumer-side boundary test parses **3207's committed exemplar artifacts** from `shared/tests/fixtures/` (the memory-eval PRD's promised dashboard-shaped reader, delivered here); malformed artifact increments `issues`, never crashes the endpoint (tests).
- **β — frontend section (`tab_memory_evals.jsx` + registration):** per-eval cards, per-metric trend charts (existing `charts.jsx` primitives), verdict badges (alarm/recovered/grandfathered/insufficient-data), limits provenance display, escalation links, storm-aggregate banner, latest-run age + stale hint, "no eval artifacts yet" placeholder with `—` empties; `data.js` `ENDPOINTS` row (`MEMORY_EVALS`), `index.html` script tag (before `tabs.jsx`), MemoryTab section wiring. *Intermediate — unlocks γ.* *Signal:* structural-contract tests in the established source-as-text idiom: endpoint registered in `data.js`, script tag present and load-ordered in `index.html`, section export referenced by `MemoryTab`, badge/link/banner/empty-state elements present (tests); dashboard restart after merge noted in the done-note (`feedback_dashboard_restart_after_backend_merge`).
- **γ — escalation-parity integration gate:** seeded fixture set exercising the three DD2 parity states — alarmed metric + matching open escalation (fingerprint-equal), recovered metric + still-open escalation, storm-collapse (per-metric alarms suppressed, one aggregate escalation) — plus grandfathered/insufficient-data non-alarm rendering. **Leaf.** *Signal:* integration test over a temp artifact tree + temp escalation queue dir: every alarm verdict in the payload carries its fingerprint-matched escalation id and every pending `eval_regression` escalation maps back to exactly one metric row (both directions asserted); the storm case renders the aggregate banner and no per-metric links; UI structural assertions cover the three badge states; the whole flow re-verified against **3211's committed exemplar verdicts fixture**.

**G7 walk (advisory at author time; re-walked at decompose):** alarm/verdict/limit contracts are JSON artifacts + a structured model field, never prose or report-scraping (INV-1/INV-2); reader failures surface as structured `issues` rendered loudly, empty states are explicit (INV-2/INV-4); the dashboard is display-only — it acts on nothing, and closure stays with the watcher (INV-3 N/A by construction); no dashboard-side statistics, one escalation reader, one fingerprint join by equality (INV-5); no new fail-soft suppression loop is introduced, and the one staleness display deliberately does not duplicate ε's self-escalation tripwire (INV-4/INV-5). No waivers anticipated.

## 8. Out of scope

- Any dashboard-side statistical computation, grandfather derivation, or a second limits/verdict/supersedes implementation (G6/INV-5 — the load-bearing exclusion).
- θ (3214) corpus display (DD7); verdict-history timeline (DD8).
- Alarm-on-staleness filing (ε's runner-failure self-escalation owns it).
- Any new fused-memory HTTP/MCP read surface or `DASHBOARD_FUSED_MEMORY_URLS` change.
- Escalation resolution/triage from the eval section (watcher remains sole closer); watcher SKILL.md changes (ε owns the triage row).
- Multi-host artifact fetch (DD1 revisit trigger).

## 9. Open questions (tactical, implementation-time)

1. **Chart primitive per metric kind** (tripwire item-counts vs proportion vs scalar) — implementer picks per kind from `charts.jsx`; StatTile+Sparkline is the safe default. Decide in β.
2. **Trend downsample cap** — lean 90 most-recent runs, disclosed via `truncated`; decide in α.
3. **Whether the corpus-counts block of M1 artifacts also renders** (cheap context row per eval) — implementer's call in β.
4. **Exact payload field names** — pinned by α's shape fn + tests; β consumes those names.
5. **Reciprocal affordance in the Escalations tab** (an eval_regression row linking back into the eval section) — cosmetic, defer; if wanted later it is a small β-shaped follow-up.
