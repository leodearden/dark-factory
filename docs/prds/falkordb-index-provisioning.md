# FalkorDB index provisioning, drift detection, and BM25 restoration

**Project:** dark-factory (fused-memory Graphiti backend + reconciliation detector surface).
**Status:** active, 2026-08-05. **Approach:** B+H (contract + two-way boundary tests on the
provisioning/health seam, which reconciliation and the startup path both consume).

**Provenance:** five-agent investigation session 2026-08-05 (brief:
`~/.claude/spawn-briefs/design-0-falkordb-index-investigation.md`). Every number below was
measured this session on throwaway scratch graphs; no index was created on any real graph.

## Goal

Every registered project's FalkorDB graph carries the index set graphiti 0.28.2 actually expects,
so the BM25 leg of hybrid search returns rows instead of silently returning nothing — and any future
divergence between expected and actual is **loud** rather than invisible for four months.

The user-observable end state: a dispatched agent's `## Task Context` block contains facts about
*its own task*. Today it does not — measured recall@5 = **0/24** across five real briefing queries,
whose 24 result slots drew from just **10 distinct edges out of 14,041**.

## Background — the defect, as measured

`_MultiTenantFalkorDriver.build_indices_and_constraints` is a `pass` stub
(`fused-memory/src/fused_memory/backends/graphiti_client.py:291`), and `clone()` (`:335`) returns
that same subclass, so `_ensure_indices` (`:474`) calls the stub, adds the graph to
`_indexed_graphs`, and logs `'Ensured indices on graph %r'` at DEBUG regardless. **No index has ever
been built by this code path.** A missing fulltext index makes `CALL db.idx.fulltext.queryNodes`
return 0 rows *with no error*, so `rrf([[], cosine])` degenerates to pure cosine, silently.

Provenance: `723ec915c3` (2026-04-02) fixed a real clone storm — pre-fix, `clone()` returned plain
`FalkorDriver`s whose `__init__` fire-and-forgets `build_indices_and_constraints()`, and
`_driver_for()` cloned on *every search request* (166 leaked connections → 15). The fix was **two**
changes: the `pass` override **and** the `_cloned_drivers` per-group cache. The cache alone bounds
construction to once per group_id per process. The override remains load-bearing for the implicit
`__init__` path and **is retained by this PRD**; only the *explicit* path stops routing through it.

Four premises from the original framing were tested and **refuted**: the `uuid`-only indices were
not built by an older graphiti (its FalkorDB branch has always emitted the composite, and a virgin
composite verifiably yields `Indices created: 4`); the build loop does **not** abort on statement 1
(the error text contains `already indexed`, so the driver's swallow absorbs it and iteration
continues); the commit's "~4s each" is wrong by ~3 orders of magnitude (16 concurrent full redundant
builds = 149 ms total); and today's exact-name scan costs ~5.9 ms, not 8–11 ms.

**Unexplained, deliberately:** what created the single-property `Entity(uuid)`/`RELATES_TO(uuid)`
indices on 6 graphs. Container logs bracket it to 2026-04-05 → 2026-04-09 (between a full data wipe
caused by the bind-mount path bug in `e2fce780a0`, and this container's creation); the `pass` stub
already existed. No mechanism in this repo accounts for it. Recorded as an open question, not
guessed at — it does not change the remediation.

## The crux — why "just delete the stub" is wrong

FalkorDB rejects a composite `CREATE INDEX` wholesale when **any** listed property is already
indexed, and the error text is `Attribute 'uuid' is already indexed`, which
`falkordb_driver.py:231-235` **swallows**. Measured on a replica of production's exact starting
state (`Entity(uuid)` + `RELATES_TO(uuid)` present), running upstream's 13-statement loop in order:

| statement | outcome |
|---|---|
| R1 `Entity(uuid, group_id, name, created_at)` | rejected → swallowed → **all 4 properties lost** |
| R5 `RELATES_TO(uuid, …7 props)` | rejected → swallowed → **all 7 lost** |
| R2–R4, R6–R9 | created |
| F1–F4 (all fulltext) | created |

So unstubbing alone yields fulltext everywhere and **zero range index on `Entity.name`** — measured:
the exact-name query remains a `Node By Label Scan` at 7.1–7.8 ms afterwards. It delivers the
behaviour-changing half and silently omits the safe half, while the logs read as success.

**Six graphs are in this trap** (`dark_factory`, `reify` — larger at 27,627 — `know_live`,
`solar_challenge_platform`, `autopilot_video`, `my_solar_challenge`). Virgin graphs are provisioned
correctly by the plain statements. The remediation that works, tested: **issue per-property range
statements**, which converge to a state identical to the composite's. FalkorDB decomposes a
composite into independent single-attribute fields, so there is no composite-prefix trap and no
behavioural difference in the result. `CREATE INDEX IF NOT EXISTS` is a syntax error in FalkorDB.

## Measured cost and benefit

| | measured |
|---|---|
| Full 13-statement build at dark_factory scale | **1.93 s** incl. background completion |
| Statement return latency / slowest to `OPERATIONAL` | 0.5–2.0 ms / **594.5 ms** (creation is background) |
| **Blocking window on concurrent reads** | **effectively zero** — 422,903 probe iterations, **0 errors**, median 0.299→0.326 ms, worst stall 31.6 ms *below* the 44 ms ambient baseline max |
| Memory | **+14 MB** (`indices_sz_mb` 4→18) |
| Exact-name lookup | **5.9 ms → 0.38 ms (~15–16×)**, plan flips to `Node By Index Scan` |
| Steady-state write cost with fulltext | **+0.125 ms/node** (p95 0.34→0.77 ms) |
| Fulltext write hazard | **none** — 13 hostile property shapes all wrote, `indexingFailures` stayed 0 |

The hardened `build_fulltext_query` (`falkor_fulltext.py`, task 3334) was validated **with a live
index present** for the first time: it rescued 6 queries upstream *hard-errors* on, with zero cases
failing where upstream succeeded. The 9950 dead-letter class is a query-assembly fault that already
fires today against a missing index; it is not introduced by provisioning.

## Sketch

**Lane A — expected-set derivation + provisioning (fused-memory backends).** Derive the expected
index set by parsing `graphiti_core.graph_queries.get_range_indices('falkordb')` +
`get_fulltext_indices('falkordb')` into a normal form (INV-5: single home, no hand-copied list, and
a graphiti upgrade that changes the set is picked up automatically). Diff against
`list_indices(group_id)`, then create **only what is missing**, per-property for range. Computing
the diff first means we never issue a statement that would hit `already indexed`, so correctness
does not rest on a load-bearing substring sentinel.

**Lane B — wiring (fused-memory backends).** Registry-filtered startup enumeration **plus** a
first-write choke point, so a project graph born after startup is provisioned rather than waiting
for the next restart. The `pass` override stays; the false-positive DEBUG log goes.

**Lane C — detection (fused-memory reconciliation + escalation).** A pure
`summarize_index_health(actual, expected)` mirroring `reconciliation/queue_health.py`, consumed by a
recon detector that escalates drift with a storm escape (INV-4), plus a **BM25 canary** asserting
fulltext actually returns rows — because an index can be present in `db.indexes()` yet
`UNDER CONSTRUCTION` and not serving (INV-6).

## Resolved design decisions

- **D1 — Per-property range statements, never the upstream composite.** Measured: the composite is
  all-or-nothing against production's starting state and its rejection is swallowed. Per-property
  converges to an identical index state (`Indices created: 1` each) and is the only form that works
  on both the 6 trap graphs and the virgin ones. Fulltext statements are issued as upstream emits
  them (measured to succeed against the trap state).
- **D2 — Diff before create; do not rely on the `already indexed` swallow.** `list_indices()` is
  read (it is already `ro_query`-based and pinned by `test_list_indices_integration.py`), the
  expected set is subtracted, and only the remainder is issued. The swallow remains as a backstop
  for races, but no correctness property depends on FalkorDB's error wording.
- **D3 — Expected set is parsed from upstream's own statement strings, not restated.** INV-5. A
  hand-maintained list would drift silently on a `graphiti-core` upgrade — and `pyproject.toml:14`
  pins `>=0.28.1`, an open-ended range. The parser is covered by a test that asserts it handles
  every statement the installed graphiti emits, so an upgrade introducing an unparseable form fails
  loudly at test time rather than silently under-provisioning.
- **D4 — Keep the `pass` override on `build_indices_and_constraints`.** It still suppresses the
  implicit `__init__` fire-and-forget on every `clone()`, which is the path that caused the
  `723ec915c3` storm. Only `_ensure_indices` changes: it stops calling
  `driver.build_indices_and_constraints()` (which is both the stub *and* the composite trap) and
  calls the new provisioning function instead. Removing the override is explicitly rejected.
- **D5 — Scope is the project registry, not a name prefix.** Provision graphs whose `group_id` is a
  registered project (`taskmaster.project_root` + `DASHBOARD_KNOWN_PROJECT_ROOTS`, basename-derived
  — the same registry the recon harness hard-binds against, which raises `UnknownProjectError`
  rather than falling back). Today that resolves to `autopilot_video, autotrade, dark_factory,
  know_live, mission_control, pump_web_ui, reify, solar_challenge, solar_challenge_platform`.
  Today's filter (`!= 'default_db' and not endswith('_db')`) would otherwise provision all **35**
  probe/test/scratch graphs on every restart.
  **Deliberate consequence:** `my_solar_challenge` (876 nodes, last write 2026-07-28) and `knowlive`
  (0 nodes, a near-duplicate of `know_live`) are **not** registered and will **not** be provisioned.
  If `my_solar_challenge` is live it should be *registered*, not special-cased here — that is the
  principled fix and it belongs to whoever owns that project, not to this PRD.
- **D6 — Provision at startup *and* at first write, under a per-group lock.** `_ensure_indices` is
  today reachable only from the startup enumeration, so a graph created later gets nothing until the
  next restart; `autotrade` and `mission_control` are registered projects with no graph yet, so this
  is live, not hypothetical. The first-write choke point is `_driver_for`/`_client_for` on cache
  miss. `_indexed_graphs` is checked-and-set without a lock today; wrap in the existing
  `_identity_lock_for(group_id)` pattern so two concurrent first-writes don't both build (INV-7:
  the hold is per-group and bounded).
- **D7 — Provisioning returns a structured result, and the false-positive log is deleted.**
  `logger.debug('Ensured indices on graph %r', group_id)` fires unconditionally after a no-op and is
  at DEBUG, so at the service's INFO level it does not even print — there is currently no positive
  *or* negative signal in the logs. Replace with a structured return
  (`created` / `already_present` / `failed` / `expected_total`) logged at INFO when anything changed
  (INV-2: emit facts the caller has, never a claim).
- **D8 — Detection is a pure summarizer + a recon detector with a storm escape.**
  `summarize_index_health(actual, expected) -> {healthy: bool, missing: [...], ...}` with zero I/O,
  mirroring `queue_health.py`. The recon detector escalates under a new category
  `recon_missing_index`, deduped via `EscalationQueue.has_open_l1(..., category=...)` following
  `stage1_stall_detector.py`, so a persistently-missing index files **once**, not every cycle
  (INV-4). Recon runs in-process inside `fused-memory.service`, so this costs no new connection.
- **D11 — The drift detector dedups on a synthetic graph key, and says so.** Every existing recon
  detector is per-task: `EscalationQueue.has_open_l1(task_id, *, category)` takes `task_id` as a
  required positional, and `stage1_stall_detector`'s two categories are both genuinely task-scoped.
  An index-drift finding is per-**graph** and has no task. Resolution: pass `f'graph:{group_id}'`
  into the `task_id` slot as an **opaque dedup key**, and carry `group_id` plus the structured
  `missing` list as first-class fields on the escalation payload — never only encoded in the key
  (INV-2). This is a deliberate reuse of an existing slot, not an accident; it is recorded here so
  the next reader does not mistake it for a task reference. **Refactor trigger:** if a third
  non-task detector appears, grow `has_open_l1` a generic `key=` parameter rather than accumulating
  more synthetic task_ids. Not done speculatively now — one caller does not justify widening a
  cross-package API.
- **D9 — The canary asserts BM25 *serves*, not that an index *exists*.** `db.indexes()` reporting a
  row is insufficient: measured, an index is `UNDER CONSTRUCTION` for up to 594 ms after creation
  and returns nothing while it is. The canary issues a known-token fulltext query and asserts ≥1
  row, using the `await_index_operational` barrier pattern already in
  `test_falkor_fulltext_integration.py` (INV-6: status matches liveness).
- **D10 — Activation is gated behind the memory-briefing-and-fusion chain.** Restoring BM25 changes
  retrieval ordering for every consumer, and two of the four briefing queries are being retired
  outright by that PRD's β. Landing indices first would perturb a surface that is concurrently being
  rewritten, and would make the E1 probe's post-rescope baseline unattributable. The wiring task
  (γ) — the point at which live graphs actually change — depends on **3658 (α)**, **3659 (β)** and
  **3660 (γ)**. The pure code lanes do not.

## Pre-conditions

- **3659** (briefing rescope) merged — retires `project overview architecture goals` and
  `recent decisions and rationale`, and scopes the conventions query to `stores=["mem0"]`, removing
  both prose-query regression risks from Graphiti's path. Measured hazard being avoided: for
  `project overview architecture goals`, `goals` has df=0 and `overview` has **df=1** (a single
  irrelevant dashboard-UI edge), which BM25 would promote to RRF ≥ 1.0 and thus position 1–2 of
  every briefing.
- **3658** (RRF cross-store merge) merged — deletes the synthesized `score = 1.0 − 0.05·i` bridge at
  `memory_service.py:3346`, which is the other constant implicitly calibrated to cosine-only
  Graphiti ordering.
- **3660** (E1 registry re-key + probe refresh) merged — gives a post-rescope, pre-index baseline
  under per-template topics, so this PRD's improvement is attributable to the indices alone.

## Cross-PRD relationships (G4)

| Seam | Direction | Mechanism | Owner | Status |
|---|---|---|---|---|
| `docs/prds/memory-briefing-and-fusion.md` α | consumes | `MemoryService.search` cross-store RRF + `metadata.store_rank` | **that PRD (3658)** | dep γ→3658 |
| `docs/prds/memory-briefing-and-fusion.md` β | consumes | briefing query table / retired prose queries | **that PRD (3659)** | dep γ→3659 |
| `docs/prds/memory-briefing-and-fusion.md` γ | consumes | E1 registry per-template topics + refreshed probe artifact | **that PRD (3660)** | dep γ→3660; this PRD's ζ reads the artifact as its baseline |
| `docs/prds/memory-eval-program.md` | produces | the E1 probe re-run showing post-index recall | **this PRD (ζ)** | ζ emits a new artifact; no change to the instrument |
| `plans/fm-memory-identity-prd.md` + `.capability-manifest.md` | produces | correction of the `PASS (wired)` certification of `_ensure_indices` | **this PRD (η)** | η edits both |
| Graphiti entity-resolution conflation (esc-3375-1 / task 3375) | neither | candidate-list membership for `Task N` nodes | **esc-3375-1, open** | this PRD supplies evidence; see Out of scope |

## Contract (the provisioning/health seam)

**Normal form.** An index is identified by `(label: str, entity_type: 'NODE'|'RELATIONSHIP',
field: str, index_type: 'RANGE'|'FULLTEXT')`. Both the expected set (parsed from upstream statements)
and the actual set (from `list_indices`) are normalized to sets of this tuple before diffing.

**`list_indices()` shape caveat — load-bearing.** It returns **one record per label**, and binds
`field` to `row[1]`, which for a multi-property index is a **list** (`[uuid, group_id, name,
created_at]`), not a string; `type` (`row[2]`) is a dict keyed by property. The normalizer must fan
a record out to one tuple per property. A comparison written against the naive per-field reading
will silently report every multi-property index as missing.

**Provisioning.** `ensure_indices(group_id) -> IndexProvisionResult` with fields
`created: list[tuple]`, `already_present: int`, `failed: list[tuple[tuple, str]]`, `expected_total:
int`. Idempotent: a second call on an unchanged graph creates nothing and returns
`created == []`. Never raises on a per-statement failure — records it in `failed` and continues, so
one bad statement cannot abort the remaining twelve (the exact shape of upstream's bare loop that
this replaces). Raises only on an unreachable driver.

**Health.** `summarize_index_health(actual: set, expected: set) -> dict` — pure, no I/O, returns at
minimum `{healthy: bool, missing: list[tuple], unexpected: list[tuple], expected_total: int}`.
`healthy` is `missing == []`; `unexpected` is reported but never acted on (an operator-added index
is not drift to repair).

**Ordering guarantee.** Provisioning is background in FalkorDB: `CREATE` returns in 0.5–2.0 ms while
the index reaches `OPERATIONAL` up to 594 ms later. Callers must not treat a successful `create` as
"serving". Only the canary (D9) establishes serving.

## Boundary-test sketch

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Trap-state graph provisioned | scratch graph seeded with `Entity(uuid)` + `RELATES_TO(uuid)` only | all 4 `Entity` range fields and all 7 `RELATES_TO` range fields present afterwards; **no** composite statement was issued (the failure this PRD exists to prevent) |
| 2 | Virgin graph provisioned | scratch graph, no indices | full expected set present; `created` length == `expected_total` |
| 3 | Idempotent re-run | graph already fully provisioned | `created == []`, `already_present == expected_total`, `failed == []` |
| 4 | Multi-property normalization | graph with a 4-property composite | `summarize_index_health` reports `healthy: true` (regression guard for the `list_indices` list-valued `field` caveat) |
| 5 | Drift detected and escalated once | expected set contains an index the graph lacks; detector run twice | first run files one `recon_missing_index` escalation; second run files **none** (storm escape, INV-4) |
| 6 | Canary distinguishes present-but-not-serving | fulltext index created, queried before `OPERATIONAL` then after | canary reports not-serving, then serving — proving it tests service rather than metadata presence |
| 7 | First-write provisioning | a registered project_id with no graph yet | after one `add_episode`, the graph carries the full expected set without a restart |
| 8 | Unregistered graph untouched | scratch graph named like a probe artifact | provisioning skips it; `db.indexes()` unchanged |

## Decomposition plan

- **α — expected-index-set derivation + normalizer** (modules: `fused-memory/src/fused_memory/backends`, `fused-memory/tests`). Parse upstream's range + fulltext statements into the normal form; normalize `list_indices()` records (fanning multi-property records out per property). *Signal:* a unit test asserts the parser consumes **every** statement the installed `graphiti_core` emits with zero unparsed remainder, and that a `db.indexes()` record with `field=['uuid','group_id','name','created_at']` normalizes to 4 tuples (boundary test 4). Prereqs: none.
- **β — `ensure_indices` provisioning function** (modules: `fused-memory/src/fused_memory/backends`, `fused-memory/tests`). Diff-then-create, per-property range + fulltext, structured `IndexProvisionResult`, per-statement failure isolation. Retains the `pass` override; deletes the false-positive DEBUG log. *Signal:* against a scratch graph seeded to production's trap state, all 11 previously-lost range fields exist afterwards and no composite statement appears in the issued-statement log (boundary tests 1–3). Prereqs: **α**.
- **γ — wiring: registry filter + startup + first-write** (modules: `fused-memory/src/fused_memory/backends`, `fused-memory/src/fused_memory/config`, `fused-memory/tests`). Registry-derived graph filter (D5); call `ensure_indices` from the startup enumeration and from the first-write choke point under `_identity_lock_for`. *Signal:* an integration test against a live FalkorDB shows a freshly-created registered-project graph carrying the full expected set after a single `add_episode` with no restart, and a probe-named graph untouched (boundary tests 7–8). Prereqs: **β**, and external **3658**, **3659**, **3660** (D10 — this is the task whose merge changes live graphs).
- **δ — health summarizer + recon drift detector** (modules: `fused-memory/src/fused_memory/reconciliation`, `escalation/src/escalation`, `fused-memory/tests`). Pure `summarize_index_health`; recon detector filing `recon_missing_index`, deduped via `has_open_l1(f'graph:{group_id}', category=...)` per D11, with `group_id` and `missing` as structured payload fields. *Signal:* running the detector twice against a graph with a deliberately-absent index files exactly one escalation visible in `get_pending_escalations`, and zero on the second run; the filed escalation's payload exposes `group_id` and the missing-index list as fields rather than only in prose (boundary test 5). Prereqs: **α**.
- **ε — BM25 serving canary + production-index integration test** (modules: `fused-memory/tests`). Canary asserting a known-token fulltext query returns ≥1 row, with the `await_index_operational` barrier; an integration test asserting the **production** expected set (as opposed to `test_falkor_fulltext_integration.py`, which builds its own single-field one-node index in-fixture and so cannot catch this class). *Signal:* the canary reports not-serving for an index queried before `OPERATIONAL` and serving after (boundary test 6); the production-set test fails on a graph missing any expected index. Prereqs: **γ**.
- **ζ — activation verification + measured retrieval improvement** (modules: none — verification leaf). After γ+δ+ε are deployed and `fused-memory.service` has restarted: assert every registered project graph carries the full expected set, and re-run the E1 probe read-only, comparing against 3660's post-rescope baseline artifact. *Signal (primary, falsifiable against a measured baseline):* a repeat of this session's 5-query briefing probe returns **≥1 edge actually mentioning the queried task for at least 4 of the 5 queries** — today the measured result is **0/24 across all five**, and the corpus provably contains 2–10 matching edges per query, so any non-zero result is a real change and the 4/5 floor is well inside what BM25 can reach. Plus: `CALL db.indexes()` on every registered project graph shows the full expected set at `status: OPERATIONAL`. *Signal (secondary, recorded not asserted):* the E1 probe is re-run read-only and its `briefing-task-semantic` topic result is **recorded alongside** 3660's post-rescope artifact as a before/after pair. No threshold is asserted on it — 3660's baseline does not exist yet, so a bound here would be a guess; the comparison is the deliverable, and if it shows no improvement that is a finding to escalate, not a task failure. Prereqs: **γ**, **δ**, **ε**.
- **η — documentation + correct the false certifications** (modules: `docs`, `plans`, `OPERATIONS.md`). An OPERATIONS.md runbook section (expected set, how to check drift, what the canary means, the composite trap so nobody re-introduces it); correct `plans/fm-memory-identity-prd.capability-manifest.md:65`, which marks `_ensure_indices` **`PASS (wired)`** and cites it as a *precedent* certifying a new backend-init hook; annotate `plans/fm-memory-identity-prd.md:19` and `memory_service.py:1448`, which describe search as "embedding+BM25" without noting BM25 returned nothing. *Signal:* `grep -n 'PASS (wired)' plans/fm-memory-identity-prd.capability-manifest.md` no longer returns the `_ensure_indices` row; OPERATIONS.md gains a section that `grep -i index` finds (it currently returns zero hits across SETUP.md/OPERATIONS.md). Prereqs: **ζ**.

## Out of scope

- **Any change to entity resolution's name-keyed candidate dict.** `existing_nodes_by_name = {node.name: node …}` (`node_operations.py:311`) is last-one-wins while the prompt answers with a `duplicate_name` *string*, and dark_factory holds **114 exact-duplicate normalized names** (233 nodes). BM25 makes name collisions in the candidate list *more* likely, so this is a real interaction — but it is esc-3375-1 / task 3375's territory, and this PRD must not pre-empt that ruling. Flagged in the cross-PRD table; the investigation evidence should be attached to that escalation.
- Junk-graph cleanup (21 `probe_e1_gw*` pytest-xdist leftovers, `_test_*`, `_probe`) and the
  three-spelling `know_live`/`knowlive`/`know-live` collision — separate task, not gated on this.
- Re-running the `write_triage` calibration. Its 16-digit thresholds
  (`config.yaml:254-263`) are order statistics measured against cosine-only retrieval and become
  stale once BM25 serves. Named here so it is not forgotten, but it belongs to whoever owns the
  triage calibration, and it must run *after* ζ.
- Pinning `falkordb/falkordb:latest` in `docker-compose.yml`. Real operational exposure (index
  semantics could change under a `docker-compose pull`) but a distinct concern.
- Emoji/symbol tokens dropped by `falkor_fulltext.is_searchable_term`, and the now-inaccurate
  docstring claim at `graphiti_client.py:309-310` that stripping "only ever removes unparseable
  operands". Measured false (`—`, `•`, `→`, emoji are indexable), practical loss is junk.

## Open questions (tactical)

1. **Parser strictness on an unrecognized upstream statement.** Fail the test loudly (D3's intent)
   vs. skip-and-warn. **Suggested:** fail — a silently-skipped statement is exactly this PRD's
   failure mode recurring. Decide in α.
2. **Canary corpus.** Whether the BM25 canary queries a seeded ephemeral collection or a known-stable
   real entity. **Suggested:** seeded ephemeral, following `test_recon_dedup_premise.py:57-143`, to
   avoid depending on live corpus content. Decide in ε.
3. **Whether δ's detector also runs on the startup path** or only on the recon cadence.
   **Suggested:** both, sharing the pure summarizer; startup is the cheapest place to catch a
   provisioning failure. Decide in δ.
4. **`unexpected` index reporting verbosity** — whether to log operator-added indices at INFO every
   cycle or only on change. **Suggested:** on change. Decide in δ.
