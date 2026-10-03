# UUID-prefix resolution — resolve on read, repair on write, sweep the corpus

**Status:** active · 2026-09-09 · approach **B + H** (contracts + two-way boundary tests)

**Extends:** `plans/toolcall-markup-containment-prd.md` — a second boundary guard behind the same shared middleware chokepoint, and the fused-memory shim it introduced (γ3, task 4458) learns to consume the shared `FORWARD_REPAIR` tier. **Absorbs:** DF **4643** (pending; rewritten in place as leaf γ, keeping its id and recon provenance). **Amends:** DF **3144** (pending; its `get_memory_by_id` and `get_entity_by_uuid` items move here). **Supersedes:** the 2026-09-09 13:42 ruling "prefix resolution is deliberately NOT built, and should stay that way" (Leo's memory `procedural_mem0_consolidation_sitting_gotchas.md` §4, withdrawn the same day).

**Evidence base:** the position paper *Resolve or Refuse* (session investigate-reify-1111615, 2026-09-09; markdown, leg reports and scripts at `~/.claude/fleet/sessions/investigate-reify-1111615/paper/`). Every number below is re-runnable from those scripts against a Qdrant/FalkorDB snapshot of that day. Leo's five rulings on the paper's §10 are recorded in its §11 and restated as D1–D5 here (INV-9: the paper is the home of the measurement; this PRD is the home of the design).

> **Code anchors** verified against main `207024a40e` (2026-09-09). Main moves fast — cite-by-symbol; re-locate lines at implementation time.

---

## 1. Goal (G1 consumer + user-observable surface)

> **An 8-to-31-hex uuid prefix written by an agent is resolved deterministically against the calling project's live Mem0 ids and Graphiti uuids: a read tool given a prefix returns the record when exactly one id matches, a write tool carrying a prefix lands with the full uuid substituted and the substitution reported back, an ambiguous prefix is refused with the candidates named, and a prefix that matches nothing is left alone. The prefixes already stored are expanded once by an atomic sweep.**

Named consumers, one per mechanism (G1):

| Mechanism | Consumer |
|---|---|
| `shared.uuid_prefix` — pure token detector over nested arguments (C1) | `UuidPrefixGuardMiddleware` (α), the sweep (ε) |
| `fused_memory.services.id_resolver` — per-project resolver over both id spaces (C2) | the middleware on fused-memory (γ), `get_memory_by_id` / `get_entity_by_uuid` miss reports (β), the HTTP route (β), the sweep (ε) |
| `POST /resolve-prefix` on fused-memory's Starlette app (C5) | the standalone escalation server's guard (δ) |
| `UuidPrefixGuardMiddleware` with the outcome × tool-class policy matrix (C3) | fused-memory server 8002 (γ), each project's escalation server (reify 8100, dark-factory 8102) and the in-process recon copy 8103 (δ) |
| expansion delta on `meta` / rejection payload with candidates | the calling agent — it sees what changed, or chooses among candidates |
| `uuid_prefix_detected` structured fact | the operator (journal), and the ambiguous-forward storm escalation (existing L2 watcher consumer) |
| prefix arm in `get_memory_by_id` / `get_entity_by_uuid` (C4) | curators and any agent holding a prefix — the case the paper reconstructs (§2) |
| sweep script `scripts/sweep_uuid_prefixes.py` (C6) | the operator, dry-run then `--apply` once |

User-observable surface (live, reify, today's corpus):

- `get_memory_by_id(project_id="reify", memory_id="bff81530")` returns `found: true`, the record `bff81530-70e5-4ab6-939e-507299d1bcd2`, and `resolved_from_prefix: "bff81530"` — where today it leaks a raw Qdrant 400 (`value bff81530 is not a valid point ID`) as `error_type: UnexpectedResponse`.
- `add_memory(project_id="reify", content="… see bff81530 for the dead repair …")` lands with the body reading `… see bff81530-70e5-4ab6-939e-507299d1bcd2 …` and the response `meta.uuid_prefix_repair.substitutions == [{"field": "content", "from": "bff81530", "to": "bff81530-70e5-4ab6-939e-507299d1bcd2", "namespace": "mem0"}]`.
- The same `add_memory` where the prefix matches two live ids in the project is **rejected** with `error_type: ambiguous_uuid_prefix`, both candidates named with namespace, id and an 80-char preview, and `original_call` carrying the complete argument map; nothing is written.
- `escalate_info(detail="… memory 208f1bdf …")` on a project where that prefix is ambiguous **lands unchanged** with `meta.uuid_prefix_repair.ambiguous` naming the candidates; `update_task` behaves the same.
- `add_memory(content="… recon-3f2a9c1e … 20260904 …")` lands byte-identical, no warning, no fact.
- `scripts/sweep_uuid_prefixes.py --lane mem0 --apply` rewrites the stored prefixes that resolve uniquely; a second run reports `0 expandable remaining` and exits `EXIT_CLEAN`.

---

## 2. Background — evidence (why this exists, and why in this shape)

Measured 2026-09-09 over Qdrant `fused_reify` (34,725 points), `fused_dark_factory` (27,325), `task_curator_reify` (5,692), FalkorDB graph `reify` (29,989 nodes + 73,560 edges), read-only snapshots of both task stores, and both `data/escalations` trees. Paper §2–§4.

- **Ambiguity is not the problem.** Zero 8-hex collisions inside `fused_reify`; zero inside the reify Graphiti graph; zero across the two. Birthday expectation at N = 34,725 is 0.14 pairs; the first expected in-project collision arrives around 2027-05 at the measured 124 writes/day. Unioning projects creates collisions now (two in `fused_reify ∪ fused_dark_factory`), which is the measured reason resolution is scoped per project (D6). 98.3% of resolvable prefix citations are same-project.
- **Unreachability is the problem.** The prefix the curation sitting could not resolve in ten semantic searches, `bff81530`, resolves to exactly one live record in a 0.2 s id-only walk. No exposed tool walks ids; `scan_memory_content` cannot match a caller needle (DF 4857); `get_memories_by_metadata` caps at 1000; `get_memory_by_id` has no shape check (DF 3144) and leaks the backend error. The reference never touched a store: a sub-agent had the full id in its own search result, abbreviated it in report prose, and the parent copied the prefix into the next mandate. That is why a read-side arm is not optional (D2).
- **The habit is universal and discipline has not moved it.** 47,209 standalone 8-hex tokens in stored prose; 11,513 resolve uniquely to a live memory (65% in memory bodies, the rest across 20+ task and escalation field paths); 4 are ambiguous; 2,616 are git SHAs with **zero** overlap against memory prefixes; 33,076 are dates, run ids, Graphiti ids and deleted-memory citations. Reify escalation prose alone carried 266 distinct prefix-shaped memory citations on 22 of the 31 days before 2026-09-09 — after Mem0 rules, DF 1144's prompt fix (2026-05) and DF 3132's `delete_memory` hard-error (2026-08) had all landed. Only 849 of 34,725 reify records (2.4%) carry a `topic`, and 85% of prefix-cited records have none, so the "cross-link by slug" convention cannot reach the corpus it governs.
- **A zero-match error on prose is impossible.** The best cheap heuristic for "this 8-hex token is meant as a memory reference" (proximity to memory/record/entry, without commit/sha nearby) has precision 0.33 at recall 0.51. Three quarters of tokens would be refused. Hence D1.
- **Substitution is monotone.** The original token stays the first characters of the replacement, so a wrong expansion (expected ~8 × 10⁻⁶ per non-memory token, 0 observed in 4 months) is visible and reversible, never information-destroying.
- **Cost is not a constraint.** An id-only scroll (`with_payload=false`, 10,000 per page) returns all reify ids in 0.19–0.24 s; a prefix map builds in 17 ms. v1 walks per call (D9).

---

## 3. Sketch of approach

```
shared/src/shared/uuid_prefix.py                       (C1)
  find_prefix_tokens(arguments) -> tuple[PrefixToken, ...]     pure; nested dict/list walk; glue rule
  substitute(arguments, token, full_id) -> arguments'          pure; span-exact; monotone

shared/src/shared/uuid_prefix_guard.py                 (C3)
  UuidPrefixGuardMiddleware(resolver, project_for, exempt_tools,
                            forward_on_ambiguity_tools, escalation_sink, fact_sink)
    on_call_tool: detect -> resolve(project, token) per distinct token
                  -> unique: substitute, forward, delta on meta
                  -> ambiguous: reject-with-candidates | forward-with-candidates (declared set)
                  -> none: inert
                  -> resolver unavailable: forward unchanged, meta.resolver='unavailable'

fused-memory/src/fused_memory/services/id_resolver.py  (C2)
  resolve_prefix(project_id, prefix) -> Resolution
    Mem0:     scroll_collection_pages(with_payload=False, page_size=10_000)   (one kwarg added)
    Graphiti: _paged_ro_query  MATCH (n:Entity) WHERE n.uuid STARTS WITH $p  (indexed)
              MATCH ()-[e:RELATES_TO]-() WHERE e.uuid STARTS WITH $p
  miss_report(project_id, prefix) -> {tombstone?, graphiti_node?, graphiti_edge?}

fused-memory server/main.py                            (C5)  POST /resolve-prefix  -> Resolution JSON
fused-memory server/markup_guard.py                     shim consumes FORWARD_REPAIR; installs a guard CHAIN
fused-memory server/tools.py                           (C4)  get_memory_by_id / get_entity_by_uuid miss report

escalation/src/escalation/server.py                     registers the guard with an httpx resolver client
scripts/sweep_uuid_prefixes.py                         (C6)  lanes mem0 | tasks | escalations; dry-run default
```

The middleware never guesses: exactly one candidate across both namespaces, or nothing is changed. The detector is the only thing that parses prose for ids; the resolver is the only thing that enumerates ids; the sweep reuses both and the markup sweep's atomic-write engine (INV-5).

---

## 4. Contracts (H)

### C1 — Detection contract (`shared.uuid_prefix`)

```python
class PrefixToken(NamedTuple):
    path: tuple[str | int, ...]   # argument path, e.g. ('content',) or ('metadata', 'cluster_memory_ids', 2)
    token: str                    # lowercase hex, 8 <= len <= 31
    start: int                    # span within the string at `path`
    end: int

def find_prefix_tokens(arguments: Mapping[str, Any]) -> tuple[PrefixToken, ...]: ...
def substitute(arguments: Mapping[str, Any], token: PrefixToken, full_id: str) -> dict[str, Any]: ...
```

**Token grammar (normative).** A run of 8–31 lowercase hex characters, not preceded or followed by a hex character, not followed by `-` (so the first group of a full uuid never matches), and not preceded by `-` or `_` (a composite identifier such as `recon-3f2a9c1e`, `episode_74b902f8`, `STAGE2_0b179fd4`; measured: 29% of the noise class and 2.8% of true references, nearly all of which are `/`-separated lists that this rule keeps). All-decimal tokens are **not** excluded (131 all-decimal tokens are genuine references; dates fall to the `none` outcome instead). Full 36-char uuids are never tokens: a dead full uuid is provenance by corpus convention (`48433882-ee71-480d-aff7-c91aa4640ff5`, `ffa913a1-ffdc-4f30-b435-bb4f06771fd5`).

**Walk.** Every string value reachable from the argument map through dicts and lists, so `submit_task.metadata.cluster_memory_ids[2]` is a path (the 4643 case), not only top-level strings as the markup guard scans.

**Invariants.** `find_prefix_tokens` is pure, synchronous, never raises, and returns tokens in document order. `substitute` replaces exactly the span, returns a new structure (never mutates its input), and the result at `path` has the original token as a prefix of the replacement (monotone, testable). Distinct tokens are resolved once each per call.

### C2 — Resolver contract (`fused_memory.services.id_resolver`)

```python
class Candidate(NamedTuple):
    namespace: Literal['mem0', 'graphiti_node', 'graphiti_edge']
    id: str                       # full uuid
    preview: str                  # first 80 chars of Mem0 `data` / Graphiti name or fact; '' if none

class Resolution(NamedTuple):
    outcome: Literal['unique', 'ambiguous', 'none']
    candidates: tuple[Candidate, ...]   # exactly 1 for unique, >=2 for ambiguous, () for none

class ResolverUnavailable(Exception): ...   # carries which store failed

async def resolve_prefix(project_id: str, prefix: str) -> Resolution: ...
async def miss_report(project_id: str, prefix: str) -> dict[str, Any]: ...
```

**Universe.** The calling project only: the Mem0 collection `Scope.mem0_collection_name(prefix)` and the FalkorDB graph named by the project's `group_id`. Both namespaces, always; one Mem0 match plus one Graphiti match is `ambiguous`.

**Enumeration.** Mem0: `Mem0Backend.scroll_collection_pages(collection, with_payload=False, page_size=10_000)` — the `with_payload` kwarg is this PRD's one extension to the walk primitive (it is THE single home for the offset walk, INV-5; a sibling walk is forbidden). Graphiti: two `STARTS WITH` queries through the client's paged read-only query helper against the per-property `Entity(uuid)` / `RELATES_TO(uuid)` range indexes (`falkor_indices.py`). A prefix shorter than 8 hex is a `ValueError` at this boundary; `validate_full_uuid` is reused for the full-id case, never re-implemented.

**Failure semantics (INV-11).** A store that cannot be reached raises `ResolverUnavailable`; the resolver never returns `none` for a prefix it could not check. `miss_report` consults the Mem0 tombstone store and the same Graphiti queries so a read-side miss can say *deleted*, *is a graph node*, *is an edge*, or *unknown*.

**Freshness.** v1 walks per call and needs no confirm-by-read (the walk is the read). If a per-collection map is ever added (D9), it must be invalidated at the three backend mutation sites — `Mem0Backend.add`, `Mem0Backend.add_system_record`, `Mem0Backend.delete` — plus a TTL rebuild for out-of-band script writes, and every `unique` hit must be confirmed with a direct read before use.

### C3 — Boundary policy contract (`UuidPrefixGuardMiddleware`)

Policy is a **declared matrix at registration**, outcome × tool class, never inferred from a tool's name and never prose (INV-1):

| Outcome | Default tools | `forward_on_ambiguity_tools` (declared frozenset) |
|---|---|---|
| `unique` | forward with the substitution applied in place; `meta.uuid_prefix_repair.substitutions=[{field, from, to, namespace}]` | same |
| `ambiguous` | **reject**: raise `ToolError` carrying `{error, error_type: 'ambiguous_uuid_prefix', outcome: 'rejected', tool, field, token, candidates, original_call, hint}` — no `repaired_call`, the choice is the author's; `original_call` is the complete argument map so nothing is lost without a resubmit (the 3936 gap, answered locally) | forward **unchanged**; `meta.uuid_prefix_repair.ambiguous=[{field, token, candidates}]` |
| `none` | inert: no substitution, no meta, no fact | same |
| resolver unavailable | forward unchanged; `meta.uuid_prefix_repair.resolver='unavailable'` and a WARNING fact naming the store — never a rejection, never silence | same |

**Declared sets, per registration site.** `exempt_tools`: tools whose id arguments must stay refuse-only or whose text legitimately carries bare hex — on fused-memory `delete_memory`, `delete_episode`, `delete_entity`, `remove_edge`, `update_edge`, `reassign_edge`, `merge_entities`, `consolidate_memories`, `scan_memory_content`; on the escalation server none. `forward_on_ambiguity_tools`: on fused-memory `{update_task}` (Q2); on the escalation server every tool. `project_for(arguments) -> str | None`: on fused-memory reads `project_id` off the arguments (every tool declares it); on the escalation server a registration-time constant, because no escalation tool declares a project (the markup guard's `_identity` records the same fact). A call whose project cannot be determined is forwarded unchanged with `meta.uuid_prefix_repair.resolver='no_project'`.

**Override.** `metadata={'allow_uuid_prefix': True}` bypasses the guard for a deliberate bare prefix (a correction record quoting a bad citation verbatim), stripped before dispatch exactly as `allow_mcp_markup` is, and forwarded only to tools that declare `metadata`.

**Ordering.** Registered **after** the markup guard so a parameter the markup repair recovers is scanned; asserted by test on both servers. Both guards install through one shim chain on fused-memory (`install_boundary_guards`), not two wrappings of `ToolManager.call_tool`.

**Facts (INV-2).** `uuid_prefix_detected` with `tool`, `field`, `token`, `outcome ∈ {expanded, rejected, forwarded_ambiguous, resolver_unavailable}`, `candidate_ids`, `namespace`, `agent_id`, `project`. Emitted only for `unique`/`ambiguous`/unavailable — never per `none` token (33,076 of them in the corpus; a fact stream that is 75% noise teaches readers to skip it).

**Storm escape (INV-4).** `forwarded_ambiguous` and `resolver_unavailable` are the fail-soft outcomes and carry the existing `MarkupStormCounter` keyed `(project, outcome)`; `expanded` is the designed success path and is **not** storm-counted (at ~10% of 124 writes/day it would fire the markup thresholds continuously and be ignored).

### C4 — Read-side contract (`get_memory_by_id`, `get_entity_by_uuid`)

On a prefix-shaped `memory_id` / `entity_uuid` the guard (C3) has already expanded a `unique` hit and rejected an `ambiguous` one before the tool body runs, so the body sees a full uuid or a `none` prefix. The body's arm is the miss report only:

```json
{"found": false, "prefix_checked": true, "prefix": "3f2a9c1e",
 "namespaces": {"mem0_tombstone": null, "graphiti_node": null, "graphiti_edge": null},
 "hint": "<_FULL_UUID_HINT, plus: no live id in project X starts with this prefix>"}
```

Two invariants: the existing `{found: false}` shape for a well-formed full uuid is unchanged, and a Qdrant `400 … not a valid point ID` can no longer reach a caller from either tool (it is the symptom this PRD retires). `resolved_from_prefix` on a hit rides on `meta` from the guard, and the tool body additionally echoes it in its result dict so the fact is visible whether or not a client renders `_meta` (open question 1).

### C5 — HTTP resolver contract (`POST /resolve-prefix`)

Mounted on the same Starlette app `streamable_http_app()` returns, next to the MCP mount; not an MCP tool. Request `{"project_id": str, "prefix": str}`; response `200` with `Resolution` JSON, `400` on a malformed prefix or unknown project, `503` with `{"error_type": "resolver_unavailable", "store": …}` when a store is unreachable — never `200` with `none` for a prefix that was not checked (INV-11). Bound to the same host/port as the MCP endpoint; the escalation client uses a 1 s timeout and treats any non-200 as unavailable.

### C6 — Sweep contract (`scripts/sweep_uuid_prefixes.py`)

Reuses `scripts/sweep_toolcall_markup.py`'s engine by import — `discover_targets`, `resolve_write_target`, `load_target`, `serialize_like`, `round_trips`, `write_repaired`, and its exit vocabulary — and C1/C2 for detection and resolution; no second copy of the atomic temp-write → verify-parse → `os.replace` contract and no second token grammar (INV-5). Three lanes:

| Lane | Population | Write path |
|---|---|---|
| `mem0` | `data` bodies of every project collection (per `--project`) | `MemoryService.update_memory` content-amend, point id and owned metadata keys preserved (the sitting's 12 in-place amends are the precedent) |
| `tasks` | `title`, `description`, `details`, `test_strategy`, and every string under `metadata` of both task stores | `update_task` through the interceptor with `metadata_mode='replace'` on the just-read blob, the `migrate_task_metadata_to_x_namespace.py` shape; `assert_write_accepted` on every write |
| `escalations` | every `data/escalations/**` JSON record incl. archives | the markup sweep's atomic file engine |

Rules, all lanes: dry-run is the default and prints a full diff; `--apply` expands only `unique` tokens, reports `ambiguous` ones by record and never touches them, leaves `none` byte-identical, and is idempotent — a second `--apply` reports `0 expandable remaining` and exits `EXIT_CLEAN`. Resolution is against the record's own project. A record that fails `round_trips` after substitution is refused, not written. The `mem0` lane rate-limits writes (each amend re-embeds) and logs one line per project with counts; the `tasks` lane refuses a task whose `update_task` is rejected and continues.

---

## 5. Resolved design decisions

- **D1 — The zero-match arm is inert on free prose (Leo, Q1).** It errors only where intent is declared: a prefix passed as the id argument of a read tool, or the resolver route itself. Basis: precision 0.33 at recall 0.51 for the best proximity heuristic; 33,076 tokens in the noise class.
- **D2 — Two arms over one resolver, not write-time only.** The incident's reference lived in agent-to-agent prose and never crossed a store boundary; the write-side repair alone would not have caught it. The read-side arm is the incident's fix; the write-side repair is the corpus-hygiene backstop that stops stored prefixes compounding at N².
- **D3 — `update_task` joins the escalation tools in the forward-on-ambiguity class (Leo, Q2).** Losing the write is worse than the defect, the same reasoning that put `escalate_info` under `FORWARD_REPAIR` in the markup PRD's C2. Memory and task-creation tools reject on ambiguity: a retry is cheap and the author must choose.
- **D4 — Both namespaces on both sides (Leo, Q3).** A prefix resolves against Mem0 ids and Graphiti node/edge uuids; a cross-namespace match is `ambiguous`. Measured cross-space collisions today: 0 (expected 0.84 pairs). The Graphiti query latency is the first thing leaf β measures.
- **D5 — The fused-memory shim consumes the shared `FORWARD_REPAIR` tier (Leo, Q4).** `install_markup_guard`'s `ValueError` for any policy but `REJECT_WITH_REPAIR` is lifted by adapting `call_next` to hand `_forward` a `ToolResult`-shaped object and by returning a `types.CallToolResult(meta=…)` — verified possible on the bundled `mcp` 1.27.0, whose `CallToolResult` has `meta` and whose lowlevel `call_tool` handler passes a returned `CallToolResult` through. This is the first implementation step of leaf γ; the markup guard's own tier on fused-memory is **not** changed by this PRD (open question 3).
- **D6 — Per-project scoping.** The resolution universe is the calling project's collection and graph. Cross-project citations (197 of 11,513, all dark-factory prose citing reify memories) fall to `none` and are out of scope; unioning projects is what creates ambiguity today.
- **D7 — Transport for the standalone escalation server is a plain HTTP route on fused-memory (Leo).** `shared/` keeps its dependency set (fastmcp, httpx; no store clients); the resolver has one home; the escalation guard fails open on any non-200 with the degradation on `meta`. The in-process recon copy (8103, `ReconciliationHarness` instantiating `escalation.server.create_server`) wires the service resolver directly.
- **D8 — Full sweep, in the markup sweep's shape (Leo).** All three lanes, archives included, dry-run default, `--apply` once by the operator. The alternative of leaving 11,513 stored prefixes to become ambiguous one by one from ~2027 was rejected as paying the N² cost later for no saving now.
- **D9 — No cache in v1.** A 0.2 s walk is below the noise floor of an LLM tool call; a cache would reintroduce the stale-read direction `citation_verifier`'s per-call memo exists to avoid. Add a per-collection map only on a measurement, with the invalidation sites named in C2.
- **D10 — Destructive and id-addressed mutation tools stay refuse-only.** The guard's `exempt_tools` on fused-memory names them; DF 3132's posture for `delete_memory` is extended, not reversed. A prefix must never be expanded before a delete, merge or edge rewrite.
- **D11 — Nested scan, every string.** The detector walks dict and list values, unlike the markup guard's top-level scan, because the 4643 instance (`metadata.cluster_memory_ids`, four of nine entries as prefixes) lives one level down and the measured field spread is 20+ paths.
- **D12 — 4643 is rewritten in place, not superseded by a sibling.** Its recon provenance (`stage1_finding_id`, `cross_project_origin`) and the routing note that any such guard belongs in dark_factory are the durable record of why the write-side arm exists (INV-9). Its open questions — reject vs `repaired_call`, key allowlist vs shape — are answered by C1 (shape, nested) and C3 (matrix).
- **D13 — No prompt-text changes.** The `_FULL_UUID_HINT` and the Stage-1 prompt discipline stay as they are; the guard reports every expansion so the pressure toward full ids increases rather than relaxes. The two mechanical `[:8]` emitters (`ReconciliationHarness._escalate`'s `recon-{run_id[:8]}` handle, `memory_service`'s `episode_{id[:8]}` name) are out of scope (§7); the C1 glue rule keeps them out of the detector.

---

## 6. Pre-conditions / substrate (G3 — verified 2026-09-09 at `207024a40e`)

| Assumed capability | Verification | Result |
|---|---|---|
| A uniform per-argument chokepoint on both servers | `MarkupGuardMiddleware.on_call_tool` in `shared/mcp_markup_middleware.py`; registered at `escalation/server.py` `create_server` (`mcp.add_middleware`) and `fused-memory/server/main.py` `_install_tool_wrappers` (`install_markup_guard`) | ✅ both live; `tools.py` has no per-tool chokepoint (~50 hand-repeated `validate_project_id` prologues) — the middleware is the only one |
| Bundled `mcp` SDK can return `meta` to the client | `uv run --project fused-memory python -c "from mcp.types import CallToolResult; …"` → fields `meta, content, structuredContent, isError`; `mcp.server.lowlevel.Server.call_tool` `isinstance(results, types.CallToolResult)` pass-through | ✅ mcp **1.27.0**; D5 is implementable. Whether the *agent's* client renders `_meta` is open question 1 |
| An id enumeration primitive | `Mem0Backend.scroll_collection_pages(collection_name, *, scroll_filter, page_size=1000, max_pages=200, max_points, with_vectors)` | ⚠️ exists, **no `with_payload` switch** — leaf β adds the kwarg on this one primitive (INV-5). Budget 200 pages × 10,000 covers 2 M ids |
| Qdrant id-only scroll is fast | REST `points/scroll` `with_payload=false, limit=10000` | ✅ 34,731 ids in 0.19–0.24 s (Qdrant 1.17.1) |
| Graphiti prefix query | `Entity(uuid)` and `RELATES_TO(uuid)` per-property range indexes (`falkor_indices.py`); `_paged_ro_query` in `graphiti_client.py`; `_driver_for(group_id)` | ✅ `STARTS WITH` expressible against an indexed property; latency to be measured in β (103,549 uuids for reify) |
| A plain HTTP route beside the MCP mount | `mcp.streamable_http_app()` returns a Starlette app (`main.py` `run_server`) | ✅ Starlette router accepts an added `Route` |
| `httpx` available to the escalation server | `shared/pyproject.toml` declares `httpx>=0.27`; `escalation/watcher.py` already makes HTTP calls | ✅ |
| Escalation server knows its project | `create_server(queue, …)` is single-project; no escalation tool declares `project_id` (recorded in the markup residue prose builder) | ✅ constant at registration (C3 `project_for`) |
| In-place memory content amend preserving id and provenance | `update_memory` tool (id preserved, `config.yaml` "PRESERVING its Qdrant point id"); `MemoryService.update_memory` carries owned metadata keys; 12 amends in the 2026-09-09 sitting | ✅ |
| Bulk task rewrite path | `fused-memory/scripts/migrate_task_metadata_to_x_namespace.py` — `update_task` with `metadata_mode='replace'`, `assert_write_accepted` | ✅ precedent to copy |
| Atomic JSON rewrite engine | `scripts/sweep_toolcall_markup.py` — `write_repaired` (mkstemp in target dir → verify → `os.replace`), `discover_targets`, `resolve_write_target`, exit codes | ✅ importable; C6 reuses it |
| Full-uuid predicate | `fused_memory.utils.validation.is_full_uuid` / `validate_full_uuid` / `_FULL_UUID_HINT` | ✅ reused; the detector's *prefix* grammar is a distinct predicate and lives in `shared/` because fused-memory depends on shared, not the reverse |
| Tombstone store for the miss report | the `tombstone` key `get_memory_by_id` already emits for reaped records | ✅ |

No novel substrate beyond the one kwarg. The principal implementation risk is D5's shim adaptation, which is bounded and probed, not the resolver.

---

## 7. Out of scope

- **Fixing the two mechanical `[:8]` emitters** (`recon-{run_id[:8]}` escalation handles; `episode_{id[:8]}` names). They truncate other namespaces, are kept out of the detector by the glue rule, and the harness already embeds the full run id in `detail`. A one-line hygiene task if wanted; not this PRD.
- **Cross-project citations** (dark-factory prose naming reify memories). They resolve to `none`; a project qualifier convention is a separate decision.
- **A typed citation field on the write tools** (heuristic 12's real answer). The corpus is not ready; forcing it would fail the way DF 1144 failed. This PRD's facts and `meta` deltas are the substrate for it.
- **`plan.json` and the `task_curator_*` collections** as sweep lanes; **7-hex tokens** (git's default short SHA length); **the markup guard's own tier** on fused-memory (open question 3).
- **`citation_verifier` / DF 4818 / DF 5163.** Different token class, different function, no shared file.
- **Prompt and skill text** that displays 8-hex shapes while warning against them.

---

## 8. Cross-PRD / seam ownership (G4)

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/toolcall-markup-containment-prd.md` (γ3 / task 4458) | consumes | `fused_memory.server.markup_guard.install_markup_guard` → generalised to a guard chain that can consume `FORWARD_REPAIR` (D5); the markup guard's registration and tier are untouched | **this PRD** (leaf γ) | queued here |
| `plans/toolcall-markup-containment-prd.md` (C2, `EXEMPT_TOOLS`, `allow_mcp_markup`) | consumes | the declared-set and override patterns, mirrored as `exempt_tools` / `forward_on_ambiguity_tools` / `allow_uuid_prefix` | this PRD reuses; no edit to the other | wired |
| `docs/prds/memory-write-path-convergence.md` §9 η / DF 3132 / DF **3144** | extends | `get_memory_by_id` and `get_entity_by_uuid` gain C4; 3144 keeps its other five tools refuse-only | **this PRD** for the two tools; 3144 amended by leaf ζ to drop them | queued (ζ) |
| DF **4643** | absorbs | the write-side guard at the task-write boundary | **this PRD** — leaf γ *is* 4643, rewritten in place (D12) | queued |
| DF 3936 (deferred) / DF 4809 (pending) | informs | rejected-write residue | this PRD's rejection carries `original_call`; no residue filing required for ambiguity (rare, payload returned) — 3936/4809 unchanged | n/a |
| `plans/capability-delivered-checks-prd.md` | consumes | the YAML sidecar's `delivered_check`s | decompose mode | at decompose |
| DF 4818 / 5163 (`citation_verifier`) | none | — | — | no shared file; orthogonal |

No reciprocal-ownership ambiguity: every seam above is owned here or explicitly untouched.

---

## 9. Decomposition plan (one bullet per task; signals are the G2 gate)

Sizing follows the overlay bands (target 300–1,500 changed LOC, ≤10–12 files per leaf; coalesce froth). All leaves are `execution_class: code_tdd`, `task_kind: normal`. Every leaf is authored and reviewed against `docs/code-quality.md`; the heuristics that bind are named per leaf.

**α — `shared.uuid_prefix` + `UuidPrefixGuardMiddleware`: detector, policy matrix, facts, storm, tests.** *(intermediate — unlocks γ, δ, ε)*
Modules: `shared/src/shared/uuid_prefix.py`, `shared/src/shared/uuid_prefix_guard.py`, `shared/tests/test_uuid_prefix.py`, `shared/tests/test_uuid_prefix_guard.py`, a committed fixture corpus of real token contexts (≥200 rows drawn from the paper's `occurrences.pkl`: class A, C, E and the glue cases, with expected outcomes).
Implements C1 and C3 with the resolver injected as an async callable, so the middleware is testable against an in-process standalone `fastmcp` harness with a fake resolver. Heuristics that bind: *deep modules with narrow interfaces* (the middleware takes a resolver and declared sets, hides everything else), *structured data instead of meaningful strings* (`PrefixToken`, `Resolution`, `Candidate` are typed; the only string parse is the detector), *no file too large* (a sibling module, not an append to the 1,398-line markup middleware).
*Unlocks:* γ, δ (registration), ε (detector reuse).
*Evidence:* the fixture corpus replays byte-identically twice; every C1 invariant (pure, ordered, monotone, span-exact, nested paths) is pinned; against the harness, each outcome × tool-class cell of C3 produces the contracted `meta` / `ToolError` shape, `expanded` is not storm-counted, `forwarded_ambiguous` fires the storm at the threshold, and a resolver that raises `ResolverUnavailable` forwards unchanged with `resolver='unavailable'` on `meta`.

**β — fused-memory resolver, read-side arms, HTTP route.** *(leaf with its own signal; also unlocks γ, δ, ε)*
Modules: `fused-memory/src/fused_memory/services/id_resolver.py`, `backends/mem0_client.py` (the `with_payload` kwarg on `scroll_collection_pages`), `backends/graphiti_client.py` (two `STARTS WITH` queries via the paged read-only helper), `server/tools.py` (`get_memory_by_id`, `get_entity_by_uuid` miss reports), `server/main.py` (the `/resolve-prefix` route), `fused-memory/tests/test_id_resolver.py`, `tests/test_resolve_prefix_route.py`, `tests/test_get_memory_by_id_prefix.py`.
Implements C2, C4, C5. Heuristics: *minimum data access scopes* (the walk yields ids only), *clear invariants uniformly enforced* (`ResolverUnavailable` is the only degraded path and it raises), *well-defined purpose* (the resolver enumerates and classifies; it never mutates).
*Signal (user-observable, live):* `get_memory_by_id(project_id="reify", memory_id="bff81530")` returns `found: true` with `resolved_from_prefix` — today it returns `error_type: UnexpectedResponse`; `get_memory_by_id(…, memory_id="3f2a9c1e")` (no live match) returns the C4 miss report with `prefix_checked: true` and no backend error text; `curl -X POST :8002/resolve-prefix -d '{"project_id":"reify","prefix":"bff81530"}'` returns `outcome: unique` with one `mem0` candidate. Measured and recorded in the leaf's result: the id-walk wall time and the two Graphiti query latencies on the live reify stores.
*G6 note:* the `unique` signal's premise is a live fact of the corpus on 2026-09-09 (one id starts with `bff81530`); if that record is consolidated before the leaf lands, the signal moves to any other prefix the leaf's own walk reports as `unique` — the assertion is "a unique prefix resolves", not "this prefix is unique forever".

**γ — fused-memory boundary: shim consumes `FORWARD_REPAIR`, guard chain, registration with the policy sets. This leaf IS DF 4643, rewritten in place.** *(leaf)* · depends: α, β
Modules: `fused-memory/src/fused_memory/server/markup_guard.py` (→ `install_boundary_guards`), `server/main.py` (`_install_tool_wrappers`), `fused-memory/tests/test_markup_guard.py` (extended), `tests/test_uuid_prefix_guard_fused_memory.py`, `docs/task-authoring.md` (the `allow_uuid_prefix` override, one paragraph).
Order of work: (1) the shim adaptation for D5, pinned by a test that the *markup* guard still behaves identically under `REJECT_WITH_REPAIR` (30/30 specimens, the 4458 measurement re-run); (2) the chain; (3) the prefix guard registered second with `exempt_tools` and `forward_on_ambiguity_tools={update_task}` declared at the site. Heuristics: *carefully factored orthogonal dimensions* (what-to-detect × what-to-do-about-it × which-tools stay three declared axes), *SPOT* (one shim, one chain), *files make internal sense in isolation* (the shim must not reach into the middleware's privates to unwrap results — use the public `ToolResult` fields).
*Signal (user-observable, live):* `add_memory(project_id="reify", content="… bff81530 …")` lands with the full id in the stored body (read back via `get_memory_by_id`) and `meta.uuid_prefix_repair.substitutions` in the response; on a scratch project seeded with two ids sharing a prefix, the same call is rejected with `error_type: ambiguous_uuid_prefix` naming both and nothing is written, while `update_task` on that project lands unchanged with the candidates on `meta`; `add_memory(content="… recon-3f2a9c1e … 20260904 …")` lands byte-identical with no `meta.uuid_prefix_repair`; `delete_memory(store="mem0", memory_id="bff81530")` is still refused by `require_full_uuid`, never expanded.

**δ — escalation servers: httpx resolver client, registration on every project's standalone server (reify 8100, dark-factory 8102) and the in-process 8103 copy.** *(leaf)* · depends: α, β
Modules: `escalation/src/escalation/server.py` (`create_server`: a `resolver` parameter, the guard registered after the markup guard, `project_id` constant, `forward_on_ambiguity_tools` = all), `escalation/src/escalation/prefix_resolver_client.py` (httpx, 1 s timeout, non-200 → `ResolverUnavailable`), `fused-memory/src/fused_memory/reconciliation/harness.py` (passes the in-process service resolver to its `create_server` call), `escalation/tests/test_uuid_prefix_guard_registration.py` (mirrors `test_markup_middleware_registration.py`: exactly one instance, ordered after the markup guard).
Heuristics: *prefer stateless interactions* (the client is a function of `(project_id, prefix)` → `Resolution`; no session state), *no silent fail-soft* (unavailable is on `meta`, never a dropped write).
*Signal (user-observable, live):* `escalate_info(task_id=…, summary="…", detail="… memory bff81530 …")` on the reify escalation server files a record whose `detail` carries the full id and whose response `meta.uuid_prefix_repair.substitutions` names it; with fused-memory stopped, the same call **still files**, unchanged, with `resolver='unavailable'` on `meta` and a WARNING fact — never a refusal.

**ε — the sweep: three lanes, dry-run default, atomic, idempotent.** *(leaf)* · depends: α, β
Modules: `scripts/sweep_uuid_prefixes.py`, `scripts/tests/test_sweep_uuid_prefixes.py`, and — only if importing from `sweep_toolcall_markup.py` proves awkward — `scripts/sweep_common.py` extracted from it with that script re-pointed (open question 4).
Implements C6. Heuristics: *no lockstep duplication* (engine by import), *simple control flows* (one lane = one population + one write path, selected by a declared table, not flags inside flags).
*Signal (operator-observable):* dry-run over the three lanes prints per-lane counts that reconcile with the paper's classification (expandable ≈ 11,5xx occurrences across both projects, ambiguous = the `208f1bdf` occurrences only, untouched = the rest) and a full diff; `--apply --lane escalations` then a second run reports `0 expandable remaining` and `EXIT_CLEAN`, every rewritten file still parses and its byte-diff is confined to the expanded tokens; `--apply --lane mem0 --project reify` then `get_memory_by_id` on a sampled amended record shows the full id in the body with `created_at`, `topic` and `supersedes` intact; `--apply --lane tasks` then `get_task` on reify 6482 shows all nine `cluster_memory_ids` entries as full uuids.

**ζ — paired edits (G4 bookkeeping): amend DF 3144, point the two PRDs here, record the withdrawn ruling's home.** *(leaf, `complexity: simple`)* · depends: γ, δ
Modules: `plans/toolcall-markup-containment-prd.md` §8 (a row for the shim generalisation, owner this PRD), `docs/prds/memory-write-path-convergence.md` §8/§9 (η's read-side rule extended by this PRD for two tools), and via the task API: DF 3144's description loses its `get_memory_by_id` / `get_entity_by_uuid` items with a dated pointer here.
*Signal:* both PRDs' seam tables name this PRD as owner of the respective mechanism; DF 3144's record reads five tools, not seven, with the pointer.

### DAG

```
α ──┬── γ ──┐
    ├── δ ──┼── ζ
β ──┼── γ   │
    ├── δ ──┘
    └── ε
```

Sizing estimate: α 700–1,100 LOC / 5 files · β 700–1,200 / 8 · γ 400–700 / 5 · δ 250–450 / 4 · ε 600–1,000 / 3 · ζ ~100 / 2 + task edit. Nothing crosses the >15-file review trigger. γ, δ and ε are all leaves with user-observable signals; β is both a leaf (its own read-side signal) and the resolver producer for the other three — the C-as-integration-gate shape is not needed because each leaf's signal is observable through the product on its own.

### G7 walk (`docs/legibility/design-invariants.md`)

| Invariant | Disposition |
|---|---|
| `contracts-machine-checked` (INV-1) | Policy matrix, `exempt_tools`, `forward_on_ambiguity_tools`, `project_for` are declared at each registration site; the HTTP route has a typed request/response; the override is a declared metadata key. ✅ |
| `structured-facts-at-failure` (INV-2) | `uuid_prefix_detected` carries tool/field/token/outcome/candidates/namespace; the rejection names candidates and carries `original_call`; nobody log-scrapes. ✅ |
| `corroborate-before-acting` (INV-3) | A substitution is applied only on exactly one live match at call time across both namespaces; the sweep resolves per record against the record's own project at write time. ✅ |
| `storm-escape-required` (INV-4) | The two fail-soft outcomes (`forwarded_ambiguous`, `resolver_unavailable`) are storm-counted; `expanded` is the designed success path and is deliberately not (C3). ✅ |
| `no-lockstep-duplication` (INV-5) | One token grammar (`shared.uuid_prefix`), one enumeration (`id_resolver` over the one walk primitive), one atomic-write engine (imported), one full-uuid predicate (`validation.py`). ✅ |
| `status-matches-liveness` (INV-6) / `holds-owned-and-bounded` (INV-7) | No new hold or status; the ambiguity rejection returns the payload to the caller rather than filing a hold. ✅ |
| `loop-thread-occupancy-bounded` (INV-8) | The walk and the graph queries are async I/O through the existing async clients; per-call work is bounded by the 10,000-point page and the `LIMIT` on the graph queries. ✅ |
| `one-fact-one-home` (INV-9) | The measurement's home is the paper; the design's home is this PRD; the withdrawn ruling and DF 4643/3144 point here by date; no second copy of the numbers is written into task text beyond a pointer. ✅ |
| `guards-exercise-behaviour` (INV-10) | Every leaf signal is a real call through the shim/middleware against a live or seeded store; the fixture corpus replays real contexts. No test asserts on prose. ✅ |
| `no-silent-fail-soft` (INV-11) | `ResolverUnavailable` raises; the guard's only degraded path puts `resolver='unavailable'` on `meta`; `none` is inert **by design** and stated as such (D1), which is a ruling, not a silent path — the caller's text is byte-identical and nothing claimed to happen. ✅ |

No waivers required.

---

## 10. Boundary-test sketch (H) — two-way, producer and consumer sides

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 | Unique expansion, memory write | `add_memory.content` carries `bff81530`; exactly one live Mem0 id in the project starts with it | Stored body carries the full id; `meta.uuid_prefix_repair.substitutions[0] == {field:'content', from, to, namespace:'mem0'}`; `uuid_prefix_detected outcome=expanded` |
| B2 | Ambiguous, reject tier | scratch project seeded with two Mem0 ids sharing an 8-hex prefix; `add_memory.content` carries it | `ToolError` with `error_type=ambiguous_uuid_prefix`, both candidates, `original_call` == the submitted map; nothing written; fact `outcome=rejected` |
| B3 | Ambiguous, forward tier | same seed; `update_task.description` carries it | Task updated with the prefix **unchanged**; `meta.uuid_prefix_repair.ambiguous[0].candidates` has both; fact `outcome=forwarded_ambiguous`; storm counter incremented |
| B4 | Cross-namespace ambiguity | scratch project with one Mem0 id and one Graphiti node uuid sharing a prefix | `Resolution.outcome == 'ambiguous'` with namespaces `mem0` and `graphiti_node`; B2/B3 behaviour by tier |
| B5 | Zero match is inert | `add_memory.content` carries `recon-3f2a9c1e`, `20260904`, `deadbeef` (no live match) | Body byte-identical; no `meta.uuid_prefix_repair`; **no fact** |
| B6 | Glue rule | content carries `episode_74b902f8` where `74b902f8` IS a live prefix | Not a token: byte-identical, no substitution |
| B7 | Slash list is kept | content carries `f1c4a651/b7b0f63b`, both live unique | Both expanded; two substitutions reported |
| B8 | Nested path | `submit_task.metadata.cluster_memory_ids == ['8bec9cd6', '<full uuid>']`, first unique | Stored list has two full uuids; substitution path `('metadata','cluster_memory_ids',0)` |
| B9 | Destructive tool stays refuse-only | `delete_memory(store='mem0', memory_id='bff81530')` | Guard skips (`exempt_tools`); `require_full_uuid` refuses with `_FULL_UUID_HINT`; never expanded |
| B10 | Read tool, unique | `get_memory_by_id(memory_id='bff81530')` | `found: true`, record, `resolved_from_prefix` in the result dict and on `meta` |
| B11 | Read tool, miss report | `get_memory_by_id(memory_id='3f2a9c1e')`, no live match, one tombstoned id starts with it | `{found:false, prefix_checked:true, namespaces:{mem0_tombstone:<id>, …}}`; no Qdrant error text anywhere in the response |
| B12 | Resolver unavailable, write | Qdrant stopped (or resolver stubbed to raise) | Write lands unchanged; `meta.uuid_prefix_repair.resolver='unavailable'`; WARNING fact; storm counter increments; **no rejection** |
| B13 | HTTP route contract | `POST /resolve-prefix {"project_id":"reify","prefix":"bff81530"}`; then with FalkorDB stopped | `200 {outcome:'unique', …}`; then `503 {error_type:'resolver_unavailable', store:'graphiti'}` — never `200 none` |
| B14 | Escalation server fails open | fused-memory stopped; `escalate_info(detail='… bff81530 …')` on 8102 | Record filed unchanged; `meta.uuid_prefix_repair.resolver='unavailable'`; the L2 watcher sees the record |
| B15 | Ordering after the markup guard | `escalate_info.detail` ends `…</detail>\n<parameter name="suggested_action">see bff81530` | Markup guard recovers `suggested_action`; prefix guard then expands the prefix **inside the recovered value** |
| B16 | Shim regression | the 30 markup gate × specimen combinations from task 4458 under `REJECT_WITH_REPAIR` | Identical verdicts after the shim consumes `FORWARD_REPAIR` (the chain does not change the markup guard's tier) |
| B17 | Override | `add_memory(content='the bad citation was `bff81530`', metadata={'allow_uuid_prefix': True})` | Byte-identical; flag stripped before dispatch; no fact |
| B18 | Sweep idempotence and atomicity | `--apply --lane escalations` interrupted between temp-write and replace; then re-run twice | Target unchanged and parses; second run expands; third reports `0 expandable remaining`, `EXIT_CLEAN` |
| B19 | Sweep refuses ambiguity | a record carrying `208f1bdf` in a project where it is ambiguous | Reported by record id under `ambiguous`; file/record byte-identical |
| B20 | Sweep preserves Mem0 provenance | `--apply --lane mem0` on a record with `topic`, `supersedes`, `created_at` | Same point id; those keys unchanged; body carries the full id |

---

## 11. Open questions (tactical, implementation-time)

1. **Does the agent's MCP client render `_meta` to the model?** The standalone-fastmcp path is measured (`meta` survives to the client library); whether Claude Code surfaces it in the tool result the model reads is not. **Suggested resolution:** in γ, probe with one live call; if `_meta` is not rendered, the shim additionally folds `uuid_prefix_repair` into `structuredContent` and the JSON text block after output validation (the shim controls the final `CallToolResult`), and C4's result-dict echo already covers the read tools. Decide during γ, first step.
2. **Minimum token length 8, maximum 31.** 7-hex is git's default short SHA and was not classified; 32-hex undashed uuids are rejected by `is_full_uuid` today. **Suggested resolution:** keep 8–31; revisit only if the corpus shows 7-hex memory citations. Decide during α.
3. **Should the markup guard on fused-memory move to `FORWARD_REPAIR` once the shim can consume it?** Out of scope here (§7); the chain makes it a one-line registration change. **Suggested resolution:** a separate operator decision against the markup PRD's D3 retry-cost reasoning; do not fold into γ.
4. **Import vs extract for the sweep engine.** `sweep_toolcall_markup.py` is a script, not a package module. **Suggested resolution:** import by path as the tests already do; extract `scripts/sweep_common.py` only if that proves brittle, re-pointing the markup sweep in the same commit. Decide during ε.
5. **Storm threshold for `forwarded_ambiguous`.** Reuses the markup counter's `(project, outcome)` key and its 3/3600 s default; ambiguity is expected ~once a year in-project. **Suggested resolution:** keep the default; it is a tripwire, not a rate limit. Decide during α.
6. **Sweep write rate for the `mem0` lane.** Each amend re-embeds; ~7,400 bodies in reify, ~5,400 in dark-factory. **Suggested resolution:** a `--rate` flag defaulting to 5 writes/s and a `--project` selector so the operator runs one project per sitting. Decide during ε.
