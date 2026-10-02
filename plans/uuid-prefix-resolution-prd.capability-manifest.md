# Capability manifest — `plans/uuid-prefix-resolution-prd.md`

Mechanizes G3 (assumed-substrate verified) and G6 (signal-premise valid) once, at decompose
time, instead of once per task at dispatch. Machine-readable twin:
`plans/uuid-prefix-resolution-prd.capability-manifest.yaml`.

**Decompose session:** 2026-09-09 (`/prd` decompose, dark_factory).
**Re-verification base:** the PRD's code anchors were verified at `207024a40e`; main had moved
32+ commits by decompose time, so every capability below was **re-located by symbol** on the
then-current main. Three PRD statements did not survive that re-walk — see *Errata* at the
end. Verdicts: PASS = evidence found; OPEN = deliberately undecided, does not block.

DAG: `α → {γ, δ, ε}`, `β → {γ, δ, ε}`, `{γ, δ} → ζ`. β is both a leaf (its own read-side
live signal) and the resolver producer for γ/δ/ε.

---

## α — `shared.uuid_prefix` + `UuidPrefixGuardMiddleware` (C1, C3) · *intermediate*

| Capability | Binding | Verdict |
|---|---|---|
| `repair-policy-tiers-exist` | capability→producer (wired) — `shared/src/shared/mcp_markup_middleware.py::RepairPolicy` is a `StrEnum` carrying exactly `REJECT_WITH_REPAIR` and `FORWARD_REPAIR`; α's C3 matrix mirrors the tier vocabulary | PASS |
| `storm-counter-reusable` | capability→producer (wired) — `shared/src/shared/storm_counter.py::StormCounter`, recorded per call as `.record(threshold=…, window_seconds=…, label=…)`. **Not** the PRD's `MarkupStormCounter` (erratum 1) | PASS |
| `markup-middleware-chokepoint-precedent` | capability→producer (wired) — `shared/src/shared/mcp_markup_middleware.py::MarkupGuardMiddleware.on_call_tool` is the per-argument chokepoint α's sibling middleware copies | PASS |
| `fastmcp-middleware-base-available` | capability→producer (wired) — `shared/pyproject.toml` declares `fastmcp>=3.2`; `MarkupGuardMiddleware` already subclasses `Middleware` | PASS |
| `fixture-corpus-drawn-from-paper` | field-population — ≥200 real token contexts from the position paper's `occurrences.pkl`, committed under `shared/tests/`; replays byte-identically twice | PASS |
| `detector-loop-occupancy-bounded` | manual — `find_prefix_tokens` is sync and runs on the loop thread; the bound is the MCP argument-map size. α states the bound (INV-8); not mechanically greppable | OPEN |

## β — resolver, read-side arms, HTTP route (C2, C4, C5) · *leaf*

| Capability | Binding | Verdict |
|---|---|---|
| `scroll-primitive-is-sole-walk` | capability→producer (wired) — `backends/mem0_client.py::scroll_collection_pages`, whose docstring names it "THE single home for the offset/next_offset walk (INV-5)"; `scroll_all_by_metadata`, `_scroll_all_records` and `scan_payload_text` all delegate to it | PASS |
| `with-payload-kwarg-is-net-new` | capability→producer (**this leaf**) — the primitive currently takes `scroll_filter`, `page_size`, `max_pages`, `max_points`, `with_vectors` and has **no** `with_payload`. This is the PRD's one substrate extension; β delivers it | PASS |
| `graphiti-paged-ro-query` | capability→producer (wired) — `backends/graphiti_client.py::_paged_ro_query` and `::_driver_for(group_id)` | PASS |
| `entity-and-relates-to-uuid-indexed` | capability→producer (wired) — `backends/falkor_indices.py::range_create_statement` synthesizes `CREATE INDEX FOR (n:Entity) ON (n.uuid)` and `CREATE INDEX FOR ()-[e:RELATES_TO]-() ON (e.uuid)`, so `STARTS WITH` runs against a range index | PASS |
| `full-uuid-predicate-reused` | capability→producer (wired) — `utils/validation.py::is_full_uuid` / `::validate_full_uuid` / `::_FULL_UUID_HINT`; β reuses, never re-implements (INV-5) | PASS |
| `read-side-shapes-differ-between-the-two-tools` | field-population — `get_memory_by_id` returns `{found: False, memory_id, project_id}` on a genuine miss and leaks a raw Qdrant `UnexpectedResponse` on a malformed id (the symptom this PRD retires). `get_entity_by_uuid` has **no** `{found: false}` shape at all — it catches `NodeNotFoundError` and returns an error-shaped dict, and never leaked a backend error, because Cypher treats a malformed uuid as an ordinary miss. C4 must be written per-tool, not once (erratum 6) | PASS |
| `tombstone-read-path-exists` | capability→producer (wired) — `services/memory_service.py::get_mem0_deletion_tombstone`, already consumed by `server/tools.py::get_memory_by_id` to attach its `tombstone` key; `miss_report` reuses that path | PASS |
| `starlette-app-accepts-added-route` | capability→producer (wired) — `server/main.py` obtains the app from `mcp.streamable_http_app()`; a Starlette router accepts an appended `Route`, which is how C5 mounts `POST /resolve-prefix` beside the MCP mount | PASS |
| `id-resolver-is-net-new` | capability→producer (**this leaf**) — no `id_resolver` or `resolve_prefix` symbol exists anywhere under `fused-memory/src`; genuinely new territory, no half-landed prior attempt | PASS |
| `unique-prefix-resolves-live` | numeric/exactness premise — β's live signal asserts one reify Mem0 id starts with `bff81530`. **Re-confirmed alive at decompose time**: `get_memory_by_id(project_id='reify', memory_id='bff81530-70e5-4ab6-939e-507299d1bcd2')` → `found: true`. Per the PRD's own G6 note the assertion is "*a* unique prefix resolves", not "this prefix is unique forever" | PASS |
| `graphiti-prefix-query-latency` | manual — the PRD's D4 makes measuring the two `STARTS WITH` latencies part of β's deliverable; there is no pre-existing bound to check against | OPEN |

## γ — fused-memory boundary: shim consumes `FORWARD_REPAIR`, guard chain, registration · *leaf* · **IS DF 4643**

| Capability | Binding | Verdict |
|---|---|---|
| `detector-and-policy-upstream` | DAG-direction — `shared.uuid_prefix` + `UuidPrefixGuardMiddleware` produced by α, wired upstream of γ | PASS |
| `resolver-upstream` | DAG-direction — `resolve_prefix` produced by β, wired upstream of γ | PASS |
| `guard-install-site-located` | capability→producer (wired) — the PRD names `_install_tool_wrappers`; that function has been **renamed to `_install_tool_dispatch_guards`** (`server/main.py`), which is where `install_markup_guard(mcp, policy=RepairPolicy.REJECT_WITH_REPAIR, known_projects=…)` is called today (erratum 4) | PASS |
| `forward-repair-tier-exists` | capability→producer (wired) — `RepairPolicy.FORWARD_REPAIR` already exists; D5 lifts `install_markup_guard`'s restriction to `REJECT_WITH_REPAIR` rather than inventing a tier | PASS |
| `call-tool-result-carries-meta` | capability→producer (wired) — the bundled `mcp` SDK's `CallToolResult` exposes `meta`, and the lowlevel `call_tool` handler passes a returned `CallToolResult` through; this is what makes D5's shim adaptation implementable | PASS |
| `override-strip-precedent` | capability→producer (wired) — `allow_mcp_markup` is stripped at **two** layers: the schema-conditional strip in `mcp_markup_middleware.py::_apply_override` (only when the target tool has no `metadata` param) and an inlined `strip_markup_override(metadata)` at six call sites in `server/tools.py`. `allow_uuid_prefix` must mirror **both**, not just the first | PASS |
| `exempt-tools-exist` | capability→producer (wired) — eight of the nine tools γ names in `exempt_tools` are live fused-memory MCP tools: `delete_memory`, `delete_episode`, `delete_entity`, `update_edge`, `reassign_edge`, `merge_entities`, `consolidate_memories`, `scan_memory_content`. **`remove_edge` does not exist under any name** — edge removal is a soft-delete through `update_edge`, which is already on the list, so the exempt set loses one entry and covers the same ground (erratum 5) | PASS |
| `project-for-can-resolve-every-guarded-tool` | capability→producer (wired) — C3 says `project_for` reads `project_id` off the arguments because "every tool declares it". **`update_task` and `submit_task` declare `project_root`, not `project_id`** — and D3 puts `update_task` in the forward-on-ambiguity class, so a naive `project_for` would go inert on exactly that tool. The mapping already exists: `run_server` builds a `{project_id: project_root}` registry via `build_known_projects_map` and threads it into the guard installer as `known_projects`, and `server/markup_guard.py::_resolve_project_root` is the precedent translator. `project_for` inverts it (erratum 7) | PASS |
| `markup-guard-regression-corpus` | rejection-mechanism — B16 re-runs task 4458's 30 markup gate × specimen combinations under `REJECT_WITH_REPAIR` and requires identical verdicts after the shim change. The chain must not move the markup guard's tier | PASS |
| `client-renders-meta` | manual — PRD open question 1. Whether the *agent's* MCP client surfaces `_meta` to the model is unmeasured; γ probes it as its first step and folds the delta into `structuredContent` if not. Deliberately OPEN | OPEN |

## δ — escalation servers: httpx resolver client + registration on 8100 / 8102 / 8103 · *leaf*

| Capability | Binding | Verdict |
|---|---|---|
| `detector-and-policy-upstream` | DAG-direction — produced by α, upstream | PASS |
| `resolve-prefix-route-upstream` | DAG-direction — `POST /resolve-prefix` produced by β, upstream. δ consumes it over HTTP per D7 | PASS |
| `escalation-registers-markup-guard` | capability→producer (wired) — `escalation/src/escalation/server.py::create_server` already calls `mcp.add_middleware(MarkupGuardMiddleware(policy=RepairPolicy.FORWARD_REPAIR, exempt_tools=frozenset()))`; the prefix guard registers beside it, ordered after | PASS |
| `httpx-available-to-escalation` | capability→producer (wired) — **transitively**: `escalation/pyproject.toml` declares `dark-factory-shared`, and `shared/pyproject.toml` declares `httpx>=0.27`. The PRD's stated evidence was wrong (erratum 2) | PASS |
| `escalation-server-is-single-project` | capability→producer (wired) — `create_server(queue, …)` takes no `project_id` and no escalation tool declares one, so C3's `project_for` must be a registration-time constant | PASS |
| `registration-test-to-mirror` | capability→producer (wired) — `escalation/tests/test_markup_middleware_registration.py` asserts single-instance registration against the real `create_server`-built server; δ's new test mirrors it. That file is **load-bearing**: another PRD's manifest greps its exact filename — do not rename | PASS |
| `recon-harness-instantiates-create-server` | capability→producer (wired) — the in-process 8103 copy is built by `ReconciliationHarness` calling `escalation.server.create_server`; δ passes it the service resolver directly rather than the HTTP client | PASS |
| `escalation-ports-8100-8102` | numeric/exactness premise — read at decompose time from each project's `dark-factory-orchestrator.yaml`: `escalation.port` is `8100` for reify and `8102` for dark_factory | PASS |

## ε — the sweep: three lanes, dry-run default, atomic, idempotent (C6) · *leaf*

| Capability | Binding | Verdict |
|---|---|---|
| `detector-and-resolver-upstream` | DAG-direction — `shared.uuid_prefix` (α) and `id_resolver` (β) both upstream | PASS |
| `markup-sweep-engine-importable` | capability→producer (wired) — `scripts/sweep_toolcall_markup.py` carries all six engine symbols (`discover_targets`, `resolve_write_target`, `load_target`, `serialize_like`, `round_trips`, `write_repaired`) and is importable by bare name; `scripts/scan_plan_decision_pairing.py` and `tests/scripts/test_atomic_write_regrowth.py` already import it that way. **Resolves PRD open question 4 to *import*; no `sweep_common.py` extraction needed** | PASS |
| `atomic-write-contract-exists` | capability→producer (wired) — `write_repaired` does `tempfile.mkstemp(dir=write_path.parent)` → `json.load` verify-parse → `os.replace`. One home; ε must not copy it | PASS |
| `exit-vocabulary-exists` | capability→producer (wired) — `EXIT_CLEAN=0`, `EXIT_REPAIRABLE_REMAINS=1`, `EXIT_WRITE_FAILED=2`, `EXIT_DID_NOT_CONVERGE=3`; ε's idempotence signal asserts `EXIT_CLEAN` on the second run | PASS |
| `bulk-task-rewrite-precedent` | capability→producer (wired) — `fused-memory/scripts/migrate_task_metadata_to_x_namespace.py` calls `update_task` with `metadata_mode='replace'` and guards every write with `assert_write_accepted`, which is what catches a rejection that still returns a success envelope | PASS |
| `mem0-inplace-amend-preserves-id` | field-population — `services/memory_service.py::update_memory` preserves the Qdrant point id (stated in its docstring) and carries mem0-owned keys through `split_managed_metadata`; ε's mem0-lane signal samples `created_at` / `topic` / `supersedes` surviving | PASS |
| `escalation-corpus-includes-archives` | numeric/exactness premise — `data/escalations/` holds 4,059 JSON records across `recovered/`, `runbooks/`, `probes/` and `archive/` (33 date-sharded subtrees). D8's "archives included" has a real population | PASS |
| `tasks-lane-signal-is-falsifiable` | **field-population — REPAIRED AT DECOMPOSE.** The PRD's tasks-lane signal named reify 6482's `cluster_memory_ids`; measured at decompose, 6482 is `done` and all 51 entries are **already** full uuids, so that signal passes without the sweep acting. Bound `declared-only` → resolved by rewriting the signal to a drift-proof form (dry-run names a record; `--apply` expands it; second run `0 expandable remaining` + `EXIT_CLEAN`). See erratum 3 | PASS |
| `expandable-population-size` | manual — the ≈11,5xx expandable-occurrence count is a position-paper measurement over a 2026-09-09 store snapshot, re-runnable from the paper's scripts but not from the repo. ε reconciles its dry-run counts against the paper rather than re-deriving them (INV-9) | OPEN |

## ζ — paired edits: seam tables + DF 3144 amendment · *leaf, `complexity: simple`*

| Capability | Binding | Verdict |
|---|---|---|
| `markup-prd-seam-section-exists` | capability→producer (wired) — `plans/toolcall-markup-containment-prd.md` §8 `## 8. Cross-PRD / seam ownership (G4)` is a live table ζ appends a row to | PASS |
| `convergence-prd-sections-exist` | capability→producer (wired) — `docs/prds/memory-write-path-convergence.md` §8 `## 8. Cross-PRD / seam ownership (G4)` and §9 leaf η (UUID-strict delete) both present | PASS |
| `df-3144-amendable` | capability→producer (wired) — DF 3144 is `pending` and undispatched at decompose; its description enumerates seven id-taking tools, two of which move to C4 | PASS |
| `3144-cannot-outrun-the-amendment` | DAG-direction — this decompose session wired **DF 3144 → depends on ζ**, so 3144 cannot dispatch against its stale seven-tool scope. Not a PRD-declared edge; added here and recorded in the hand-back | PASS |

---

## Errata — PRD statements that did not survive re-verification

1. **`MarkupStormCounter` does not exist** (PRD §4 C3). The real symbol is
   `shared/src/shared/storm_counter.py::StormCounter`; threshold and window are per-call
   arguments to `.record()`, not constructor arguments, and the `3 / 3600.0` defaults the PRD
   quotes live at the call site in `MarkupGuardMiddleware.__init__`. The storm key is a single
   `f'{project}\x1f{outcome}'` string, not a tuple. Carried into α's task text.
2. **`escalation/watcher.py` does not use `httpx`** (PRD §6 substrate table). It POSTs via
   stdlib `urllib.request` in `_send_ntfy`, and nothing under `escalation/src` imports httpx.
   The *claim* — httpx is available to the escalation server — still holds, transitively via
   `dark-factory-shared`. Only the stated evidence was wrong. Carried into δ's task text.
3. **ε's tasks-lane signal was stale on arrival** (PRD §9 ε). It asserted that after
   `--apply --lane tasks`, reify 6482 would show "all nine `cluster_memory_ids` entries as full
   uuids". Measured at decompose: 6482 is `done`, the field holds 51 entries, and every one is
   already a full uuid — recon run `5d186c7b` repaired the four prefixes long ago, as DF 4643's
   own description records. The signal would have passed with the sweep doing nothing. Replaced
   with a drift-proof form in ε's filed task text; the PRD itself is left unedited.

4. **`_install_tool_wrappers` has been renamed** (PRD §3 sketch and §6 substrate table). The
   markup guard is installed today from `server/main.py::_install_tool_dispatch_guards`. The
   installer itself is still `install_markup_guard`, and its `ValueError` gate reads
   `if policy is not RepairPolicy.REJECT_WITH_REPAIR:` — exactly the restriction D5 lifts, for
   exactly the stated reason (`_forward` returns a fastmcp `ToolResult` the bundled dispatcher
   cannot consume). Carried into γ's task text.
5. **`remove_edge` does not exist** (PRD §4 C3 `exempt_tools`). There is no hard-delete edge
   tool; removal is a soft-delete via `update_edge(edge_uuid, project_id, invalid_at=…)`, which
   is already on the exempt list. The set drops one name and loses no coverage. Carried into
   γ's task text.
6. **`get_entity_by_uuid` has no `{found: false}` shape** (PRD §4 C4). It catches
   `NodeNotFoundError` and returns an error-shaped dict, so C4's "the existing `{found: false}`
   shape is unchanged" invariant describes `get_memory_by_id` only. Nor did it ever leak a raw
   backend error — Cypher treats a malformed uuid as an ordinary miss, so the Qdrant-400
   symptom is `get_memory_by_id`'s alone. C4 must be written per-tool. Carried into β's task
   text.
7. **`project_for` cannot read `project_id` off every fused-memory tool** (PRD §4 C3, which
   asserts "every tool declares it"). `update_task` and `submit_task` declare `project_root`
   instead — and D3 places `update_task` in the forward-on-ambiguity class, so a `project_for`
   written to the PRD's letter would return `None` for the one tool Leo's ruling most depends
   on, forwarding it as `resolver='no_project'` and making the guard silently inert there.
   This is the one erratum with design consequences rather than naming consequences. The
   substrate to fix it already exists: `run_server` builds a `{project_id: project_root}`
   registry (`build_known_projects_map`) and threads it into the guard installer as
   `known_projects`, and `server/markup_guard.py::_resolve_project_root` is the precedent
   translator. `project_for` inverts that map. Carried into α's (contract) and γ's
   (registration) task text.

Beyond these seven, every code anchor the PRD names re-located cleanly by symbol.
