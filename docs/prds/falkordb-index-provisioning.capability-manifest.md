# Capability manifest — falkordb-index-provisioning

Binds each task's asserted capabilities to evidence (G3+G6 paid once, here — decompose 2026-08-05,
against PRD commit `ae6e7e3c79`). Machine-readable twin:
`falkordb-index-provisioning.capability-manifest.yaml`.

**Hazard honoured throughout this decompose:** no FalkorDB index was created, dropped or modified on
any real graph, and no `FalkorDriver` / `GraphitiBackend` was constructed (its `__init__`
fire-and-forgets `build_indices_and_constraints()` under a running loop). The absence of indices is
protected evidence for open escalation **esc-3375-1**. Every binding below was established by
static grep or by importing `graphiti_core.graph_queries` (a pure module import — no driver).

## Decompose-time corrections to the PRD

Recorded here rather than by editing the PRD prose (the PRD's measurements are the historical claim).

### C1 — η's signal premise was FALSE; signal re-cast (G6 branch 4)

The PRD's η row asserts OPERATIONS.md gains a section that `grep -i index` finds, "*it currently
returns zero hits across SETUP.md/OPERATIONS.md*". **Measured on `main` at decompose time:
`grep -ic index OPERATIONS.md` = 1** — line 848, `"index lock held by a"` (git's index, unrelated).
SETUP.md is genuinely 0.

So the premise is false and, worse, the assertion as written is **vacuously satisfied on main** — a
signal that cannot fail. Resolved by G6 resolution (c) (change the asserted configuration so the
claim becomes true and falsifiable): η's signal is re-cast onto a **heading-anchored** check,
`grep -cE '^#+ .*[Ii]ndex' OPERATIONS.md`, **measured 0 on main** and therefore a real assertion.
The `_ensure_indices` half of η's signal is likewise tightened from the bare `grep -n 'PASS (wired)'`
(which matches 10 unrelated rows) to the row-anchored ERE `_ensure_indices.*PASS \(wired\)`,
**measured 1 on main** (`plans/fm-memory-identity-prd.capability-manifest.md:65`).

### C2 — `await_index_operational` has moved since the PRD was written

PRD ε cites the barrier as "already in `test_falkor_fulltext_integration.py`". Task **3377** landed
in the interim and extracted it: it now lives at **`fused-memory/tests/_fm_helpers.py:883`** as
`await_index_operational`. Capability confirmed present; only its home changed. ε's filed
description points at the current home.

### C4 — ζ and η were silently coerced to a human-escalating pure gate; corrected in place

ζ and η were filed with `metadata.execution_class = "operational"` (the established spelling for
non-`code_tdd` leaves — `docs` is not a valid `EXECUTION_CLASSES` member). `operational` triggers
`middleware/operational_routing_guard`'s **pure-gate transform**, which silently rewrites the task to
`task_kind='deterministic'` + `always_escalates=True` — a shape that **escalates to a human without
running the work**. η would have escalated instead of writing the runbook; ζ would have escalated
instead of running the probes.

Corrected in place to `task_kind='normal'`, `always_escalates=False`, `execution_class='code_tdd'`,
per the **operator ruling recorded on task 3695 (2026-08-05)**: *anything that mutates files is
`code_tdd`, and editing documentation mutates files*. Task 3695 is the same trap firing on the
toolcall-markup batch earlier the same day. Both descriptions were rewritten so their prose no longer
declares a no-code path that contradicts `task_kind='normal'` (which would trip the routing-intent
lint). ζ remains **read-only with respect to graph state** — only its recorded comparison artifact is
written.

### C3 — `get_range_indices` silently returns the **neo4j** set for a bare string (new, measured)

`graphiti_core.graph_queries.get_range_indices(provider)` takes a `GraphProvider` **enum**, and
`GraphProvider` is a plain `Enum`, *not* a `str`-Enum — `GraphProvider.FALKORDB == 'falkordb'` is
**False**. Passing the bare string `'falkordb'` returns the **neo4j** set (27 range statements, in
`CREATE INDEX <name> IF NOT EXISTS FOR (n:Label) ON (n.prop)` form) **with no error**. Passing
`GraphProvider.FALKORDB` returns the falkordb set: **9 range + 4 fulltext = 13**, matching the PRD.
This is precisely this PRD's own failure shape (a silent wrong-answer path) and is pinned into α's
description and a delivered_check.

## G7 walk (INV-1..7, `docs/legibility/design-invariants.md`)

Every task in the batch, not only leaves.

| Task | Invariants engaged | Disposition |
|---|---|---|
| α | INV-5 (expected set parsed from upstream, single home), INV-1 (typed normal form + parser test as the machine check), INV-4 | **Resolved by design pin:** Open Question 1 is closed in-task as *fail loudly*; a skip-and-warn parser would be an un-escaped fail-soft (INV-4) and is forbidden in the filed description. |
| β | INV-2 (structured `IndexProvisionResult` replaces the false-positive DEBUG log), INV-6 (create ≠ serving, stated in the contract), INV-4 | **Compliant.** β's per-statement failure absorb is a fail-soft; its storm escape is named and lives in δ's drift detector (a persistently-missing index files once). β additionally logs `failed` at WARNING naming each statement (INV-2), so the absorb is never silent. |
| γ | INV-7 (per-group lock), INV-3 (`_indexed_graphs` snapshot), INV-1/INV-5 (registry single-homed) | **Resolved by design pin:** the lock is held **only** across diff+create, never across the OPERATIONAL barrier (≤594 ms of background work — that belongs to ε's canary); an implementer awaiting OPERATIONAL inside the lock would convert a bounded hold into an unbounded one. INV-3 is satisfied by construction — β re-reads `list_indices()` (ground truth) on every call, so the in-process cache can only skip work, never cause a wrong action. Stated in the filed description so a reviewer does not re-flag it. |
| δ | INV-4 (dedup storm escape), INV-2 (structured payload), INV-7 (escalation owner), **INV-1** | **WAIVER — `contracts-machine-checked`** (see below). Others compliant: the `has_open_l1` dedup is the house suppression pattern and the open escalation *is* the standing loud signal; `group_id` + `missing` ride as first-class payload fields per D11, never only in the dedup key; the filed escalation's exit owner is the recon-escalation-watcher (port 8103 queue), which is the sole closer of that queue. |
| ε | INV-6 (canary asserts *serving*, not metadata presence) | **Compliant** — this task exists to honour INV-6. |
| ζ | INV-3 (asserts against live `db.indexes()` ground truth), INV-2 (secondary signal separates observation from hypothesis; recorded, not asserted) | **Compliant.** |
| η | **INV-5**, INV-2 | **Resolved by design pin:** the runbook must **not** restate the 13 statements / expected tuple set — that would be exactly the lock-step duplication α exists to prevent. It points at α's parser as the single home and gives the command to print the live expected set. INV-2: correcting a `PASS (wired)` certification that outlived its truth. |

### G7 waiver — δ, `contracts-machine-checked` (INV-1)

`recon_missing_index` is a new escalation **category**, and the category vocabulary is a free-form
`str` whose canonical set lives in a **comment** at `escalation/src/escalation/models.py:102-104` —
a contract in prose. δ inherits this gap rather than introducing it: the two existing recon
categories (`reconciliation_stale_human_operator`, `reconciliation_stale_gate_backlog`) were added
by the same convention.

**Rationale for waiving rather than fixing:** promoting the vocabulary to an enum is a cross-package
change (escalation → fused-memory → orchestrator) in service of one new caller. That is the same
judgment D11 already records for `has_open_l1`'s missing `key=` parameter — *one caller does not
justify widening a cross-package API*. **Refactor trigger (mirrors D11's):** when the category set
next grows beyond this addition, promote it to an enum or a registry with a submit-time lint rather
than adding a fourth comment line.

## α — expected-index-set derivation + normalizer *(intermediate → β, δ)*

| Capability | Evidence binding | Verdict |
|---|---|---|
| `get_range_indices` / `get_fulltext_indices` importable and emit the falkordb set | imported live at decompose (module import only, no driver): `GraphProvider.FALKORDB` → **9 range + 4 fulltext**, graphiti-core **0.28.2** | PASS |
| Bare-string provider silently yields the wrong (neo4j) set | measured at decompose — `get_range_indices('falkordb')` → 27 statements, no error; `GraphProvider` is a plain `Enum`, `== 'falkordb'` is `False` | PASS (hazard bound, see C3) |
| Upstream emits **two** distinct fulltext syntaxes the parser must both handle | measured: F1–F3 = `CALL db.idx.fulltext.createNodeIndex({label:…, stopwords:[…]}, 'p1', …)`; F4 = `CREATE FULLTEXT INDEX FOR ()-[e:RELATES_TO]-() ON (…)` | PASS |
| `list_indices()` exists, is `ro_query`-based, and binds `field` to `row[1]` | grep `fused-memory/src/fused_memory/backends/graphiti_client.py:2918` (`async def list_indices`), body binds `label,field,type,entity_type` = `row[0..3]`; pinned by `fused-memory/tests/test_list_indices_integration.py` | PASS (wired) |
| The multi-property `field` really is list-valued (the normalizer's whole reason to exist) | R1 `Entity(uuid, group_id, name, created_at)` is emitted as one composite statement ⇒ one `db.indexes()` record with a 4-element `field`; existing test already defends with `isinstance(field_val, list)` (task 665 hygiene note) | PASS |
| DAG-direction | α has no prereqs; β and δ are downstream | PASS |

## β — `ensure_indices` provisioning function *(intermediate → γ)*

| Capability | Evidence binding | Verdict |
|---|---|---|
| Normal form + parsed expected set | producer: **α** (upstream, dep wired) | PASS (producer-upstream) |
| The `pass` stub and the false-positive log exist exactly where D4/D7 say | grep `graphiti_client.py:291` (`async def build_indices_and_constraints`), `:335` (`def clone`), `:474-481` (`_ensure_indices`, `_indexed_graphs.add`, `logger.debug('Ensured indices on graph %r')`) | PASS (wired) |
| The 11 range fields β must restore are real | derived from the live falkordb statements: R1 `Entity` → uuid, group_id, name, created_at (4); R5 `RELATES_TO` → uuid, group_id, name, created_at, expired_at, valid_at, invalid_at (7). 4+7 = **11**, matching the PRD's crux table | PASS |
| Per-property range `CREATE` is valid FalkorDB syntax | PRD §crux, measured on scratch graphs 2026-08-05 (`Indices created: 1` each; composite decomposes to independent single-attribute fields) | PASS (PRD-measured) |
| Scratch-graph test substrate that never touches a real graph | `fused-memory/tests/_fm_helpers.py` live-Falkor scaffolding + per-run uuid-suffixed graph names (tasks 3377/3502) | PASS (wired) |
| Storm escape for the per-statement failure absorb | producer: **δ** — but δ is a *sibling*, not upstream of β | PASS (see note) |

*Note on the last row:* β's `failed` absorb is loud **within β** (structured field + WARNING log naming
each failed statement), which is what INV-2/INV-4 require at β's own boundary. The *persistence*
escape (a statement that keeps failing every cycle) is δ's detector. δ is not upstream of β, so this
is deliberately **not** claimed as a producer binding — β ships complete without it, and ζ is the
integration gate where both are present. Recorded so the DAG-direction check is not mistaken for an
inversion.

## γ — wiring: registry filter + startup + first-write *(intermediate → ε, ζ)*

| Capability | Evidence binding | Verdict |
|---|---|---|
| `ensure_indices` to call | producer: **β** (upstream, dep wired) | PASS (producer-upstream) |
| First-write choke points exist | grep `graphiti_client.py:366` (`_driver_for`), `:379` (`_client_for`) | PASS (wired) |
| Per-group lock exists and is a sync accessor | grep `graphiti_client.py:438` (`def _identity_lock_for`), docstring: "returns the Lock object itself… `async with backend._identity_lock_for(gid):`" | PASS (wired) |
| Startup enumeration exists, and today's filter is the one D5 replaces | grep `graphiti_client.py:605` and `:2979-2982` — `g != 'default_db' and not g.endswith('_db')` | PASS (wired) |
| Project registry is derivable and is the one recon hard-binds against | grep `fused-memory/src/fused_memory/models/scope.py:47` (`KNOWN_PROJECT_ROOTS_ENV = 'DASHBOARD_KNOWN_PROJECT_ROOTS'`); `reconciliation/harness.py:307,463,716,744,2491` (raises rather than falling back) | PASS (wired) |
| Retrieval surface the merge perturbs is settled first | producers: **3658**, **3659**, **3660** — all `pending`, all `dark_factory`, all upstream via bare-integer deps | PASS (producer-upstream, deps wired) |
| DAG-direction | γ depends on β + 3658/3659/3660; ε and ζ depend on γ | PASS |

## δ — health summarizer + recon drift detector *(intermediate → ζ)*

| Capability | Evidence binding | Verdict |
|---|---|---|
| Normal form + expected set to diff against | producer: **α** (upstream, dep wired) | PASS (producer-upstream) |
| Pure-summarizer template exists and is *wired*, not just declared | grep `reconciliation/queue_health.py:27` (`def summarize_graphiti_queue_health`) → imported at `reconciliation/harness.py:52` → threaded into `stages/memory_consolidator.py:151,455-463`, which escalates on `not healthy` | PASS (wired) |
| `has_open_l1(task_id, *, category)` exists with `task_id` a required positional | grep `escalation/src/escalation/queue.py:489` — signature `(self, task_id: str, *, category: str \| None = None)` | PASS (wired) |
| The categorized-dedup detector pattern is established | grep `reconciliation/stage1_stall_detector.py:483-484,535` (`has_open_l1(task_id, category=_GATE_BACKLOG_ESCALATION_CATEGORY)` then submit under the same category) | PASS (wired) |
| `category` accepts a new free-form value | grep `escalation/src/escalation/models.py:102-104` — `category: str` with the canonical set as a comment | PASS *(and the INV-1 waiver above)* |
| Operator can observe the filed escalation | `get_pending_escalations` (escalation MCP), recon queue port 8103 | PASS (wired) |
| The escalation's exit owner exists | recon-escalation-watcher skill — sole closer of the 8103 queue | PASS (wired) |

## ε — BM25 serving canary + production-index integration test *(intermediate → ζ)*

| Capability | Evidence binding | Verdict |
|---|---|---|
| Provisioned graphs to canary against | producer: **γ** (upstream, dep wired) | PASS (producer-upstream) |
| `await_index_operational` barrier exists | grep `fused-memory/tests/_fm_helpers.py:883` (`async def await_index_operational(graph, timeout_s=10.0)`) — moved here by task 3377; see C2 | PASS (wired) |
| An index really can be present-but-not-serving (the premise the canary tests) | PRD §measured cost — creation is background, ≤**594.5 ms** to OPERATIONAL; task 3377's own evidence: 2/6 false successes without a barrier | PASS (PRD-measured) |
| Seeded-ephemeral-collection pattern for the canary corpus | grep `fused-memory/tests/test_recon_dedup_premise.py:57-143` | PASS (wired) |
| The existing fulltext integration test genuinely cannot catch this class | `test_falkor_fulltext_integration.py` builds its own single-field one-node index in-fixture — it never asserts the production expected set | PASS |

## ζ — activation verification + measured retrieval improvement *(intermediate → η; the integration gate)*

| Capability | Evidence binding | Verdict |
|---|---|---|
| Indices actually provisioned on every registered project graph | producers: **γ** (wiring), **δ** (drift detection), **ε** (serving canary) — all upstream, deps wired | PASS (producer-upstream) |
| Numeric floor for the primary signal | baseline **0/24** across all five briefing queries (PRD §Goal, measured 2026-08-05); corpus provably contains **2–10** matching edges per query. Asserted bound = **≥1 matching edge on ≥4 of 5 queries**. Floor is 0 by construction, so `bound > floor`; ceiling is 5/5, so 4/5 sits strictly inside what the corpus can supply | PASS (`floor:4-of-5 > 0-of-5`, ceiling 5/5) |
| Post-rescope baseline artifact to compare against | producer: **3660** (upstream of γ, transitively upstream of ζ) | PASS (producer-upstream) |
| Secondary signal asserts no unbacked threshold | by construction — the PRD records it as *recorded, not asserted* precisely because 3660's baseline does not exist yet; a bound here would be a guess | PASS (G6 branch 1 avoided by design) |
| `CALL db.indexes()` readable on the RO path | grep `graphiti_client.py:2918` docstring — validated against FalkorDB module v41800 and pinned by `test_list_indices_integration.py` | PASS (wired) |

## η — documentation + correct the false certifications *(leaf)*

| Capability | Evidence binding | Verdict |
|---|---|---|
| The false certification exists and is row-addressable | grep `plans/fm-memory-identity-prd.capability-manifest.md:65` — `\| Backend-init hook to run the pass \| grep:graphiti_client.py:261 (initialize), :252 (_ensure_indices per-graph hook precedent) \| PASS (wired) \|`; ERE `_ensure_indices.*PASS \(wired\)` matches **1** line on main | PASS |
| The "embedding+BM25" claims exist where the PRD says | grep `plans/fm-memory-identity-prd.md:19` ("fuzzy hybrid embedding+BM25 candidates"); `fused-memory/src/fused_memory/services/memory_service.py:1448` ("hybrid embedding+BM25 search") | PASS |
| OPERATIONS.md has no index section today (the re-cast assertion) | `grep -cE '^#+ .*[Ii]ndex' OPERATIONS.md` = **0**, measured on main at decompose. *(The PRD's original `grep -i index` = "zero hits" premise is FALSE — 1 hit at OPERATIONS.md:848. See C1.)* | PASS **after C1 re-cast**; original wording would have been `rejection-absent`-equivalent (vacuously true) |
| The expected set has a single home to point at instead of restating | producer: **α** (transitively upstream via ζ) | PASS (producer-upstream) |
| Verification that the claimed improvement actually happened | producer: **ζ** (upstream, dep wired) | PASS (producer-upstream) |

## Bindings that had to be resolved

| # | Binding | Resolution |
|---|---|---|
| 1 | η — `grep -i index` "returns zero hits" | **FALSE premise** (1 hit on main). Signal **re-cast** to the heading-anchored `^#+ .*[Ii]ndex` form (0 on main) so the assertion is falsifiable. G6 resolution (c). |
| 2 | η — bare `grep -n 'PASS (wired)'` | Matches 10 unrelated rows; **tightened** to the row-anchored ERE `_ensure_indices.*PASS \(wired\)`, `expect: absent`. |
| 3 | ε — `await_index_operational` cited in `test_falkor_fulltext_integration.py` | **Re-homed** to `fused-memory/tests/_fm_helpers.py:883` (task 3377 landed after the PRD was written). |
| 4 | α — expected set obtained via `get_range_indices('falkordb')` | **Hazard bound**: the bare string silently returns the neo4j set. Pinned to `GraphProvider.FALKORDB` in α's description and by a delivered_check. |
| 5 | β — storm escape for the per-statement absorb | **Not claimed as a producer binding** (δ is a sibling, not upstream). β is loud at its own boundary; ζ is where both are present. Recorded so it is not mistaken for a DAG inversion. |
| 6 | γ `_identity_lock_for` and ε `await_index_operational` as `expect: present` greps | Both **already match on main** (1 and 5 files) — they would have shipped as **no-op dispatch gates**. Downgraded to `kind: manual` with the real property named (boundary tests 7 and 6). Verified empirically at decompose rather than assumed. |

**Delivered-check vacuity audit.** Every `expect: present` check was run against `main` at decompose
and must currently **fail** (0 matches) to be a real gate; every `expect: absent` check must currently
**match**. All 11 surviving mechanical checks satisfy this, with one deliberate exception:
β's `async def build_indices_and_constraints` is a **retention** guard (already true; it flips to
failing only if β wrongly deletes the D4 override). The two checks that failed the audit are row 6
above.
