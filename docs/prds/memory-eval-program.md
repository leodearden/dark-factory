# PRD: Memory eval program — retrieval-health monitor, corpus-health time series, search telemetry, statistical alarm limits

**Project:** dark-factory (fused-memory + shared + orchestrator-briefing edge). **Status:** active, 2026-07-29. **Approach:** B+H (contracts + two-way boundary tests).
**Origin:** Leo, 2026-07-29: implement the rest of the eval-program floor from `plans/memory-subsystem-eval-design.md` (§7 + §10), "including escalations (on the memory subsystem interactive queue) on appropriate statistical limits." All substrate verified against main (post-`0806e8ec2a`) on 2026-07-29 by a three-agent verification pass; file:line evidence in §6.
**Siblings:** `docs/prds/memory-metadata-vocabulary.md` (filed as **3194–3202**; owns E2 = 3199 + shape gate 3200, the vocabulary registry 3195, retro stamping 3201); `docs/prds/memory-write-path-convergence.md` (17 leaves; owns triage/judge/provenance-fields incl. 3137); a **dashboard PRD** (commissioned 2026-07-29, authored the same day as `docs/prds/memory-eval-dashboard.md`: display all memory-eval output, trends, and visual alarms **on the same limits that cause escalations** — the M1/M2 artifact contract below is designed as its read surface, and the 2026-07-29 M2/M3 amendments pin the verdicts artifact + fingerprint carrier it consumes).

## 1. Goal (G1 consumer + user-observable surface)

The memory subsystem stops being measured by curator heroics and starts being measured by standing instruments: retrieval regressions, corpus-health degradation, and the N+1 write loop **fire escalations on the fused-memory interactive queue** (the 8103 recon escalation store, sole closer = the recon-escalation-watcher session) instead of waiting for a human to notice. This closes eval-design §10 criterion 2 for every observed failure class the vocabulary PRD doesn't already own. Observable surfaces:

- **The recon-escalation-watcher** (primary consumer; live today at `recon-watch/run.sh` → 8103): receives `eval_regression` escalations carrying metric id, value, limit + baseline provenance, and the report path — filed only on a *new* regression against a grandfathered baseline, deduped, with an aggregate storm escape.
- **Operators**: per-run report + metric-series artifacts under `fused-memory/data/memory-evals/` (timestamped-JSON idiom), including the initial-state baseline report that enumerates today's known-bad findings for the existing fix tasks (3111/3112 lineage) rather than alarming on them.
- **The retrieval-fix batch** (3111/3200 lineage): whichever storage/retrieval fix ships is *verified in production* by the E1 probe — canonical findability, superseded-inversion, claim recall, contamination — the standing monitor eval-design §7 names as "what would have caught this pathology the day it happened."
- **The dashboard PRD** (secondary consumer, commissioned): reads the M1 metric-series and M2 limits artifacts directly — JSON on disk, no Python import — so its alarms are the same limits by construction, never a re-implementation (INV-5 across repos' UI boundary).
- **Telemetry readers**: every MCP `search` produces a `write_ops` row with result ids + scores + caller identity; the write-after-miss metric — searched, wasn't shown it, wrote a duplicate — becomes a live per-incident production signal.

## 2. Scope

The eval-design floor **minus E2** (owned by 3199/3200): **E1** (consolidated-topic retrieval probe) + **E4** (superseded/stale/dangling sweep) as one scheduled retrieval-health runner; **E6** (corpus-health detector upgrade + time series, all three Mem0 categories); **E7** (search/write telemetry + retro transcript corpus + write-after-miss); plus the shared **metrics/limits/escalation layer** all of them report through. Out of scope: §7 below.

## 3. Contracts (H)

### M1 — Metric-series contract

> **Every eval run emits a machine-readable metrics artifact conforming to one schema module; the artifact series on disk is the sole read surface for both the limits evaluator and the dashboard.**

- Per-run artifact `fused-memory/data/memory-evals/<eval_id>/metrics-<STAMP>.json` (STAMP idiom per `cgl_eta_auto_apply_impl.py`): `{schema_version, eval_id, run_stamp, corpus: {project_id, counts}, metrics: [{metric_id, kind: tripwire|proportion|count|scalar, value, n, denominator?, items?[], details_path?}]}`. Human-readable report beside it.
- Schema + validation live in **one module in `shared/`** (both fused-memory and orchestrator depend on `dark-factory-shared`; fused-memory importing `orchestrator.evals` is a wrong-direction dependency — verified §6). The dashboard consumes artifacts only, never the module.
- Boundary tests: producer side (each runner's artifact validates; malformed metric rejected at emit time, not read time), consumer side (evaluator + a dashboard-shaped reader fixture parse the committed exemplar artifacts).

### M2 — Limits contract

> **Every alarm limit is a calibration output with recorded provenance — no a-priori numeric threshold appears in code or any leaf signal (G6). Pre-existing failures are grandfathered; alarms fire on regressions.**

- Closed rule kinds: **(a) structural tripwire** — a per-item binary predicate (e.g. "registry topic T's canonical present in top-k on its own phrasings") evaluated against a **grandfathered baseline set**: the first run snapshots current failures as known-bad (emitting them in the initial-state report for the fix tasks, not as alarms); an alarm fires only when an item *newly* fails (or a fixed item re-fails). **(b) proportion shift** — exact binomial/Fisher two-sided test of the current run against the trailing baseline window. **(c) count shift** — Poisson tail test likewise. Stdlib-only (`math.comb`/`lgamma`) — no scipy.
- Significance is derived, not asserted: config declares a **false-alarm budget** (expected false alarms per quarter across the whole program, default 1); α per test = budget / (runs-per-quarter × alarmed-metric count), recomputed as metrics are added. `min_samples` guard (canary-precedent) forces `insufficient_data` — which is a report status, never an alarm.
- The active limits are persisted as a **limits artifact** (`limits-current.json`: rule kind, baseline window ids, derived α, grandfather set hash, provenance) — the dashboard's alarm source, same file the evaluator read (INV-1).
- Per-run **verdicts artifact** *(amendment 2026-07-29, commissioned by the dashboard PRD `docs/prds/memory-eval-dashboard.md`)*: ε's runner persists the evaluator's per-metric outputs verbatim — `fused-memory/data/memory-evals/verdicts-<STAMP>.json` plus convenience copy `verdicts-current.json` — entries carrying `{eval_id, metric_id, item|window, fingerprint, verdict: alarm|no_alarm|insufficient_data|grandfathered, value, limit_ref, run_stamp}` and a `storm_escape` block (`triggered`, alarm count, aggregate fingerprint) when K is exceeded. These persisted verdicts are the dashboard's per-metric alarm state — downstream consumers never re-run a test or re-derive grandfather membership (G6/INV-5). An exemplar verdicts fixture is committed under `shared/tests/fixtures/` beside α's exemplars for consumer-side boundary tests.
- Boundary tests: seeded metric-series fixtures produce expected verdicts per rule kind (alarm / no-alarm / insufficient-data / grandfathered); re-running on an unchanged series alarms nothing (idempotence).

### M3 — Escalation contract

> **A limit crossing files exactly one pending `eval_regression` escalation per fingerprint onto the 8103 recon queue; the program never resolves its own escalations; a program-wide breakage escalates once, loudly, not N times.**

- Filing path: **direct Python API** — `EscalationQueue(Path(<recon escalation_queue_dir>)).submit(Escalation(...))` (the `backfill_recon_escalations.py` pattern; co-located same-user script). `category='eval_regression'` (added to the `CATEGORIES` doc list — inert but documented), `severity='blocking'`, `level=1`, synthetic `task_id=f'memory-eval-{eval_id}'` (nightly.py precedent), detail carries metric id/value/limit/baseline/report-path/remediation hint.
- Dedup: fingerprint = (eval_id, metric_id, item|window); before filing, check pending escalations for the fingerprint (the `has_open_l1` + fingerprint discipline of `stage1_stall_detector.py:350-435`); an open one suppresses re-filing.
- **Fingerprint carrier** *(amendment 2026-07-29)*: every filed `eval_regression` escalation sets the model's structured `dedupe_fingerprint` field (`escalation/models.py`) to the verdict's exact fingerprint string; the aggregate storm escalation carries the run-scoped aggregate fingerprint that also appears in the verdicts artifact's `storm_escape` block. Cross-artifact parity is fingerprint **equality** — consumers (the dashboard) match strings and never parse fingerprint structure, so the composition detail (§9) stays ε-internal.
- **Storm escape (INV-4)**: > K alarms in one run (K configurable; e.g. a store outage failing every probe) collapses to ONE aggregate escalation naming the count and the report; per-metric filings are suppressed for that run. Runner *failure* (crash, store unreachable) files its own single `eval_regression` — silence is never a healthy signal (structured-facts-at-failure).
- Sole closer remains the watcher (`resolve_issue`/`stamp_triage`); the watcher skill gains a triage row for `eval_regression`.
- Boundary tests: producer side (induced regression on a fixture series + temp queue dir → exactly one escalation with the contract fields; second run files nothing while pending; K+1 alarms → one aggregate), consumer side (compact `get_pending_escalations` projection renders the fields the watcher triages on).

### M4 — Telemetry contract

> **The MCP surface records what was asked, by whom, and what was shown; consumption of memory is never silent.**

- `search` journaling (`tools.py` `_log_read`) widens `result_summary` from `{'count': N}` to result ids + scores + per-result content sizes; full query text (drop the 200-char truncation for search rows).
- **Caller identity**: new optional `caller_agent_id`/`caller_task_id` params on the search tool — explicitly distinct from the existing `agent_id` **filter** param (the design conflict that left 99.7% of rows unattributed) — threaded by the briefing assembler's `mcp_call` (`briefing.py:1018-1030`, which already holds `agent_id = claude-task-{id}-{role}`); `_MEMORY_INSTRUCTIONS` documents passing it. Never repurpose `clientInfo` (hardcoded `'orchestrator'`; stateless HTTP drops it).
- Hint-query executions (`context_assembler.py:292-313`) log via the same journal (currently silent).
- `add_memory` rows already carry content/agent; when 3127's triage ack lands, its `routed`/`canonical_id` fields join `result_summary` (seam note §8 — logged here, produced there).
- Retention: `write_ops` gains an age-bounded prune (per `prune_mem0_intents` shape, config-defaulted generously — the table is 6.7 GB and never pruned; the *new* search-row detail must not make that worse silently).
- Boundary tests: producer side (a search through the MCP tool yields a row with ids/scores/caller; hint execution yields a row), consumer side (the write-after-miss computation parses real rows; retention prune deletes only beyond the cutoff).

## 4. Resolved design decisions

- **D1 — Grandfather-and-ratchet, not day-one alarms.** Today's corpus *fails* E1 (measured twice); alarming on known state would flood the queue with findings 3111/3112 already own. First run = baseline snapshot + initial-state report; alarms are regressions thereafter. Fixed items leave the grandfather set (ratchet), so a fix regressing re-alarms.
- **D2 — Limits/schema code home is `shared/`** (fresh module; copy the canary *pattern*, never import `orchestrator.evals` from fused-memory — wrong direction, verified). Dashboard consumes artifacts, not code.
- **D3 — Exact small-sample tests, stdlib-only, budget-derived α** (M2). The canary's fixed rel-tol thresholds are the wrong shape for ~30-topic probe sets; binomial/Poisson exact tests are cheap at these n.
- **D4 — Escalation filing is the direct queue API, level 1, flat** (recon files flat — no L2 promotion machinery; `critical`/`urgent` born-at-L2 severities are *not* used by this program).
- **D5 — E1 registry derives from the vocabulary namespace, works before it.** Entries keyed on `metadata.topic`/`kind`/`supersedes` + curator-gate enumerations, content-hash item keys (UUID rot), ≥1 held-out phrasing per topic (Goodhart guard: 3111's pin saturates canonical-presence; claim-recall/contamination/held-out phrasings keep discriminating). 3201's stamping widens auto-derivation when it lands — no dep on the 3200 gate: the monitor is what *verifies* the post-gate world, it must not wait for it.
- **D6 — E6 = extend `audit_duplicate_memories.py`, don't fork it.** ANN candidate generation (add `with_vectors=True` to `scroll_by_metadata`; `query_points` per task_curator precedent) + all three categories + series persistence + dropped-candidate disclosure (no silent caps). **3136 keeps scheduling + gate-filing ownership** — its timer runs whatever detector exists; the topic carve-out under Option C is 3136's amended text, honored here as input filtering, not reimplemented.
- **D7 — E4 imports `normalize_supersedes()` from 3196's helper** (hard dep; INV-5 — never a second supersedes parser).
- **D8 — Runners are read-only against the live store via direct `MemoryService` init** (the `audit_duplicate_memories.py` `_run` pattern under the service env — the flag-marker-sweep wrapper's runbook lesson). Probe fixtures for induced-regression tests use the seeded-ephemeral-collection pattern (`test_recon_dedup_premise.py`), never the live store.
- **D9 — Retro transcript corpus is a one-shot script** reusing `scripts/legibility/digest.py::load_transcript` / `inventory.py::_iter_json_lines`; output = a schema'd JSONL corpus (query, shown result ids/scores, caller, task) — validation set for write-after-miss and the future shadow-replay harness's input.
- **D10 — Daily cadence** (03:xx window, offset from 03:00 legibility-trickle and 03:30 flag-marker per the established stagger), so the false-alarm budget arithmetic has a fixed run rate; cadence is config, not code.

## 5. Decomposition plan (signals are the G2 gate; Greek labels this-PRD-local)

Deps: β,γ,δ ← α; γ also ← **3196** (external); ε ← α,β,γ,δ; ζ ← α (schema for logged shapes only — soft); η ← α,ζ,ε; θ independent.

- **α — metric-series schema + limits evaluator (`shared/`):** schema module + validators; evaluator implementing M2's three rule kinds, budget-derived α, `min_samples`, grandfather-set persistence; limits-artifact writer. *Signal:* evaluator over committed fixture series yields the expected verdict per rule kind incl. `insufficient_data` and grandfathered-no-alarm; exemplar artifacts validate; idempotent re-run alarms nothing (tests).
- **β — E1 retrieval-health probe:** topic registry (per D5, committed fixture; auto-derivation from consolidation metadata + curator-gate census; hand phrasings incl. held-out; the four briefing-assembler queries included) + runner computing canonical-in-top-k (k=5,10), superseded-above-successor, claim recall, contamination; emits M1 artifact + report. *Signal:* a live read-only run produces the artifact + initial-state report; on a seeded ephemeral collection, deleting the canonical flips the topic's tripwire item to fail in the artifact (test).
- **γ — E4 superseded/stale/dangling sweep:** superseded-surfacing checks via `normalize_supersedes()` (dep 3196), dangling-pointer census (`supersedes`/`parent_id`/`corrects` targets resolved via `get_memory_by_id`), task-terminal staleness join against the task store; M1 artifact. *Signal:* live run emits the three metric families; seeded fixture with a superseded-outranking pair and a dangling pointer reports both (test).
- **δ — E6 corpus-health detector + series:** `scroll_by_metadata(..., with_vectors=True)` extension; ANN candidate generation via `query_points` feeding the existing union-find; all three Mem0 categories; cluster metrics (count, size distribution, per-topic accretion rate, net-delta around consolidation events) + dropped-candidate counts; M1 series artifact. Detector CLI remains 3136-schedulable. *Signal:* live run over both projects emits cluster metrics for all three categories with disclosure counts; a fixture pair from α/3130's labeled dataset — cosine and lexical ratio both *measured at authoring* — is clustered by the ANN path, with the lexical path's verdict reported alongside (test).
- **ε — scheduled runner + escalation wiring:** systemd user timer quadruple (flag-marker idiom: sh wrapper sourcing service env / service / timer / idempotent installer) running β+γ+δ then the α evaluator; M3 filing via direct queue API with fingerprint dedup + aggregate storm escape + runner-failure self-escalation; grandfather snapshot on first run; verdicts-artifact persistence per the M2 amendment (`verdicts-<STAMP>.json` + `verdicts-current.json` + the `shared/tests/fixtures/` exemplar) with `dedupe_fingerprint` stamped on every filed escalation (M3 amendment); `CATEGORIES` doc-list addition; recon-watcher SKILL.md triage row for `eval_regression`. *Signal:* timer in `systemctl --user list-timers`; integration test against a temp queue dir: induced regression files exactly one contract-shaped escalation, re-run files none while pending, K+1 alarms collapse to one aggregate (tests); SKILL.md row present; the verdicts artifact validates and its alarm entries' fingerprints equal the filed escalations' `dedupe_fingerprint` (test).
- **ζ — E7 telemetry:** search `result_summary` widened (ids, scores, sizes; untruncated query); `caller_agent_id`/`caller_task_id` params + briefing threading + `_MEMORY_INSTRUCTIONS` note; hint-execution logging; `write_ops` retention prune (config default + startup call per precedent). *Signal:* a search issued through the MCP tool yields a `write_ops` row containing result ids+scores and the threaded caller identity (integration test); a recon hint execution yields a journal row (test); prune deletes only rows beyond cutoff (test).
- **η — write-after-miss metric:** per (caller task/session) window over ζ's rows: an `add_memory` whose content near-duplicates (pure-guard replay over logged candidates) an entry absent from that window's shown results → one incident; emits M1 metric wired into ε's evaluation (proportion rule, baseline-window). *Signal:* fixture journal encoding one searched-missed-wrote sequence and one searched-shown-wrote control yields exactly one incident (test); live run emits the metric into the series.
- **θ — retro transcript replay corpus:** one-shot script per D9 over `data/orchestrator/agent-transcripts/`; JSONL corpus artifact + coverage report (tasks scanned, searches extracted, parse failures disclosed). *Signal:* run against the live archive emits the schema-valid corpus + coverage report; a committed mini-fixture transcript round-trips (test).

**G7 walk (advisory at author time; re-walked at decompose):** alarms/detectors all carry storm escapes and failure self-escalation (INV-4); schema/limits/escalation shapes are code + artifacts, not prose (INV-1); grandfather/dedup decisions re-read live queue + series state (INV-3); runner failure is a structured escalation, not a log line (INV-2); `normalize_supersedes`, the limits evaluator, and the metric schema are single-home imports (INV-5). No waivers anticipated.

## 6. Pre-conditions / substrate (G3 — verified 2026-07-29, three-agent pass)

| Assumed capability | Evidence |
|---|---|
| 8103 queue = recon `EscalationQueue` inside fused-memory.service; dir `data/reconciliation/escalations` | `config/schema.py:937-939`; `config/config.yaml:135-137`; `harness.py:1898-1927`; `systemctl --user cat fused-memory.service` (no separate unit) |
| Direct-API filing from an external script | `fused-memory/scripts/backfill_recon_escalations.py:68-71,236-269` (`EscalationQueue(...).submit/resolve`, no MCP) |
| Threshold→dedup→submit precedent on this queue | `reconciliation/stage1_stall_detector.py:93,339-435` (+ storm rate-limits `harness.py:1660-1826`) |
| `category` unvalidated ⇒ `eval_regression` filable today | `escalation/models.py:93` plain `str`; `server.py:453-599` validates severity only; live off-list categories exist (`action_effects.py:18-23`) |
| Watcher = sole closer, pinned to 8103 | `recon-watch/run.sh` + `recon-watch/mcp.json`; `skills/recon-escalation-watcher/SKILL.md` |
| Canary limit pattern (copy, don't import) + wrong-direction dep proven | `orchestrator/src/orchestrator/evals/prompt_opt/canary.py:57-438`; `fused-memory/pyproject.toml:13-24` (no orchestrator dep); `shared/` has no stats module (fresh home) |
| Qdrant vector access for δ | `mem0_client.py:272-281` (raw client), `:341-424` (`scroll_by_metadata`, `with_vectors` not yet passed — δ adds it); `query_points` precedent `task_curator.py:1890-1896` |
| `write_ops` prune precedent | `write_journal.py:358,494`; call sites `server/main.py:560,568` |
| Telemetry seams | `tools.py:684-711` (`_log_read`), `:1441-1447` (`result_summary={'count'}`; results in scope at `:1431`); `:500-532` (`_resolve_identity`, stateless-HTTP blind; `mcp_lifecycle.py:587` hardcodes clientInfo); `briefing.py:978-1013,1018-1030`; `context_assembler.py:292-313` (unlogged) |
| Transcript readers for θ | `scripts/legibility/digest.py:32-62`, `inventory.py:81,274-300`; archive layout `shared/transcript_archive.py` |
| Timer + report idioms | `scripts/fused-memory-flag-marker-sweep.{sh,service,timer}` + installer + `-check.sh`; STAMP artifacts `cgl_eta_auto_apply_impl.py:45-46` live under `fused-memory/data/cgl-eta/` |
| Seeded-store fixture + real embedder for tests | `tests/test_recon_dedup_premise.py:57-143`; cleanup-prefix caveat `scripts/cleanup_test_collections.py:11` |
| `normalize_supersedes()` producer | task **3196** (pending; vocabulary PRD γ) — hard dep on this-PRD γ |
| Probe metadata visibility (topic/kind/supersedes in search results) | verified live 2026-07-29 (this session's reify probe); `canonical:true` sparse (6 entries) ⇒ D5 registry keys |

No other novel substrate; everything else is delivered by the leaves themselves.

## 7. Out of scope

- **E2 bake-off + shape ratification** — 3199/3200 (vocabulary PRD). **E3** golden query set, **E5** triage-accuracy extensions (ride 3130/γ-writepath), **E8** transcript re-derivation audit, **E9** canaries, **E10** fleet A/B (declined), **E11** router study — eval-design §5/§7 deferrals stand; θ's corpus is E3/E8 substrate but their runners are future work.
- The **shadow-replay harness** itself (θ produces its input corpus only).
- The **dashboard** (separate PRD, commissioned; consumes M1/M2 artifacts).
- Triage-ack production (3127), search-result provenance *fields* (3137 — payload-side; ζ is journal-side), `memory_hints` channel redesign (eval-design §9.6 — measurement first).
- Any write/mutation of the live corpus by any runner (read-only discipline; only the escalation queue and `data/memory-evals/` artifacts are written).

## 8. Cross-PRD / seam ownership (G4)

| Seam | Owner | This PRD's edge |
|---|---|---|
| Metric/limits artifact schema (M1/M2) | **this PRD** (α) | dashboard PRD + watcher consume artifacts; schema module single-home in `shared/` |
| `eval_regression` category + triage guidance | **this PRD** (ε) | watcher SKILL.md row; CATEGORIES doc list; queue closure stays watcher-owned |
| Dedup detector internals vs scheduling/gate-filing | detector: **this PRD** (δ); scheduling + gate filing + C-carve-out: **3136** | δ keeps the CLI 3136-schedulable; no timer conflict (3136's timer may subsume ε's δ-invocation when it lands — ordering note in both) |
| `supersedes` parsing | **3196** | γ imports `normalize_supersedes()` (hard dep) |
| Topic namespace / registry derivation / stamping | **3195/3201** (vocabulary PRD) | β consumes; works pre-stamp per D5; no dep on gate 3200 |
| Search-result provenance payload fields | **3137** | complementary: 3137 = what agents see; ζ = what the journal records; ζ notes 3137's fields in `result_summary` when both land |
| Triage ack shape | **3127** (write-path PRD) | ζ logs today's ack; 3127's `routed`/`canonical_id` join the row when it lands (seam note in ζ) |
| E2 fixture / labeled dataset reuse | **3130** (α-writepath) / **3199** | δ's measured fixture pair drawn from 3130's dataset; no new labeling here |
| Briefing `mcp_call` + `_MEMORY_INSTRUCTIONS` text | `_MEMORY_INSTRUCTIONS` + server params: **ζ (3212)**; ALL briefing.py edits: **memory-briefing-and-fusion β (3659)** | *Re-carved 2026-08-05:* ζ amended to server-side scope; 3659 rewrites `_get_memory_context`/`_mcp_search` and threads caller identity (dep 3659→3212). 3131/ι-vocab edit different roles.py sentences — sequence note; ζ's roles.py edit is additive |
| E1 registry briefing topics | **memory-briefing-and-fusion γ (3660)** | β's registry (3208) collapsed all four briefing queries into one topic; 3660 re-keys to per-template topics and **gates ε (3211)** so the grandfather snapshot baselines the rescoped queries |

## 9. Open questions (tactical, implementation-time)

- Fingerprint composition detail for M3 dedup (include grandfather-set hash or not) and the aggregate-K default.
- `write_ops` retention default (lean 180d) and whether search rows get a longer horizon than task-read rows (5.5M `get_task` rows dominate the 6.7 GB).
- Per-result "size" unit in ζ (chars vs tokenizer estimate) — cheap approximation acceptable, name it in the schema.
- Whether ε's wrapper also ships a read-only `--check` predicate (flag-marker `-check.sh` precedent) for a future deterministic gate task.
- Registry file format/location for β (lean: `fused-memory/tests/fixtures/` beside α-writepath's fixture, loaded by path from config).
- *Added 2026-08-05, from E1's first live run:* three instrument-quality gaps to fix in a follow-up leaf — contamination-share counts only registry-topic-foreign results (489/490 scored results were un-topiced and excluded, so 0% is near-vacuous; report the un-topiced share separately), the canonical matcher matched only 6/196 trials by content-hash (needs a fuzzy content-prefix fallback), and superseded-above-successor went unmeasured (0 comparable pairs — registry needs supersedes_pairs populated). Measurement provenance: `fused-memory/data/memory-evals/e1-retrieval-health/*-20260805T093831Z.*`.
- θ corpus location under `data/memory-evals/` vs `fused-memory/data/` (lean: former, it's an eval artifact).
