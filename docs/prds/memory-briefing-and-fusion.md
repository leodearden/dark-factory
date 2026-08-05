# Briefing memory context rescope + honest cross-store fusion

**Project:** dark-factory (orchestrator-briefing edge + fused-memory service). **Status:** active, 2026-08-05. **Approach:** B+H (contract + two-way boundary tests on the search-merge seam).

**Provenance:** esc-3253-1 evidence session 2026-08-05 — live per-query measurement of the four hardcoded briefing queries, the first-ever E1 probe run (`fused-memory/data/memory-evals/e1-retrieval-health/{metrics,report}-20260805T093831Z.*`), and controlled needle tests against known-stored content. The gate esc-3253-1 / task 3253 remains open for the operator; this PRD is the rework the gate's resolution will point at.

## Goal

Every orchestrator-dispatched agent's `# Context` block contains **memory content that is true, current, and relevant to its task and role** — instead of today's ~2,600 tokens of raw JSON whose content is ~0% canonical (measured). And `MemoryService.search` merges its two stores **honestly**, so no consumer (briefing, recon stages, curator, interactive sessions) silently loses Mem0's curated content to Graphiti's synthesized rank-scores.

## Measured baseline (2026-08-05, live corpus)

- The four briefing queries at `limit=5`: zero canonical content on all four; task-scoped query for a fresh task returned 5/5 *other* tasks' facts labeled as its context. E1 corpus-wide: canonical-in-top-5 **2.04%**, held-out phrasings **0%**, claim recall **6.25%**, 32/32 topics fail the tripwire.
- Block cost: 10,436 chars (~2,600 tokens chars/4) per 4-query dispatch, ~85% JSON envelope; ~677k tokens/day dark_factory (clean-week mean 296 dispatches/day), ~1.2M/day fleet.
- Root cause of Mem0 shut-out, verified in code: `_search_graphiti` synthesizes `score = 1.0 − 0.05·rank` (`memory_service.py:3346` — Graphiti's public `search()` returns no scores), and `search()` sorts `(is_primary, −score)` then truncates (`:3229-3233`, `:3259`) — **primary-store hard precedence**: with Graphiti routed primary and filling the limit, Mem0 is structurally excluded at any score.
- Controlled needles (content proven stored): scoped Mem0 queries hit 3/3 (ranks 1–2, cosines 0.62–0.79); the same needles through the unscoped production path: 0/3. Title+modules task query: 5/5 relevant vs bare-id 0/5. `get_entity("Task <id>")` exact-match clean; its fuzzy fallback returns wrong-task neighbors.

## Sketch

**Lane α — honest fusion (fused-memory).** Replace primary-precedence merge with reciprocal-rank fusion across stores. Rank-only, so Graphiti's scoreless results and Mem0's cosines fuse without calibration. Benefits every `search` consumer, not just briefing.

**Lane β — briefing rescope (orchestrator).** Retire the two unrescuable queries, rescope the two useful intents into store/category-scoped, task-derived queries; render distilled markdown instead of raw JSON; degrade loudly; thread caller identity (consuming 3212's server-side params).

**Lane γ — E1 registry re-key (fused-memory fixtures).** One registry topic per new briefing query template, each with its own canonical(s), claim queries, and ≥1 held-out phrasing — fixing the instrument defect that folded all four old queries into one topic and making the briefing surface individually adjudicable. Gates 3211's grandfather snapshot so the new world is what gets baselined.

β does not depend on α: β forces `stores`/`categories` per query, which routes around the broken merge; α makes the *unscoped* default honest for everyone else.

## Resolved design decisions

- **D1 — Retire, don't reword, "project overview architecture goals" and "recent decisions and rationale".** The corpus holds no overview entry (literal scan), and dispatched agents already get CLAUDE.md in-checkout. "Recent" is a temporal predicate the search API cannot express; a semantic query on the word "decisions" returns meta-fragments (measured 5/5 noise). A recency-windowed Graphiti query is future work (out of scope), added only when a consumer names it.
- **D2 — Conventions query is store/category-scoped and module-aware.** `stores=["mem0"]`, `categories=["preferences_and_norms","procedural_knowledge"]`, phrased from the task's `metadata.modules` (fallback: task title; last resort: repo-generic). Evidence: module-aware scoped probes returned 5/5 substantive current entries at cosine 0.56–0.62; the same intent unscoped returned 0 Mem0 entries in any top-20.
- **D3 — Task context is dual-channel: semantic title+modules search (never the bare task id) + `get_entity(f"Task {id}")` included only on exact name match.** Bare-id embedding matches *other* task numbers (measured 0/5); title+modules measured 5/5; the exact-match guard suppresses `get_entity`'s fuzzy-fallback wrong-neighbor hazard (client-side name equality check, no server change).
- **D4 — Fusion is RRF.** `fused(r) = Σ_stores 1/(K + rank_store(r))`, K=60 (module constant, not config), ranks 1-based per store's own ordering. Router's `primary_store` demotes to tiebreak. `relevance_score` becomes the fused score; each result's `metadata` gains `store_rank` and `store_score` (Mem0 cosine; `null` for Graphiti) so telemetry (3212) and the E1 probe keep an honest per-store signal. Graphiti stops emitting synthesized 1.0/0.95/… scores. Real-Graphiti-similarity was rejected: the public `graphiti.search()` API exposes no scores (verified), and RRF needs none.
- **D5 — Rendering is distilled markdown, entries whole (no per-entry cap).** Bullet per result: `- [<category> · <date> · <store>] <content>`; entity block renders node summary + edge facts with `valid_at` dates. Fidelity over budget: worst case ~5k tokens of pure content still beats 2,600 tokens of 85% envelope; `limit=5` and query scoping bound the size. Revisit only with measured pressure.
- **D6 — Degradation is loud.** Per-query failure: `logger.warning` + an in-block italic line naming the missing section and reason class. A `degraded: true` search response (already surfaced fault-only at `tools.py:2336-2341`, verified) renders a line naming `failed_stores`. Total failure keeps the existing notice but logs at warning. The current behavior (per-query `logger.debug` + silent section omission; outer "Memory unavailable" branch unreachable) violates structured-facts-at-failure and was observed masking a live transient server error.
- **D7 — Per-role variation.** Merger loses the memory block entirely (mechanical role, 7 dispatches/14d measured, generic-only context today). Reviewer — the single highest-volume role (35% of dispatches), today generic-only — gains the task-scoped sections (workflow holds `task_id` at every dispatch site; `events.task_id` 100% populated). Steward continuation keeps skipping. All other builders get the D2+D3 set. `limit=5` unchanged; no new config knob — the query table is code (single home in `briefing.py`), revisited on evidence, hot-reload adds nothing for inline literals.
- **D8 — Caller identity threads through β, server side stays 3212.** 3212 (deferred during this authoring; restore pending after amendment) keeps `caller_agent_id`/`caller_task_id` on the search tool + `_MEMORY_INSTRUCTIONS` prose + journal work; β's rewrite of `_get_memory_context`/`_mcp_search` threads role+task identity from all builders (role is in scope by construction in the new per-role table — including reviewer/merger/resume, which today have none). β depends on 3212 so the params exist before β passes them. The `briefing-threads-caller-identity` delivered_check moves 3212 → β (a check asserting another task's work is a fake-done hazard).
- **D9 — Registry re-key is per-query-template with non-empty claims.** One topic per β query template (conventions-generic, conventions-module-form, task-semantic-form against a pinned fixture task), each with hand-authored canonical(s) by content-hash, ≥1 claim query, ≥1 held-out phrasing. The old single `g7-design-invariants` briefing topic is superseded by these (it remains as a curator-gate topic if independently derived). A pinning test asserts the registry's briefing-topic phrasings equal this PRD's templates (INV-5 note: `briefing.py` is the single normative home for the live strings; the test pins the *fixture* to the *PRD contract*, and drift between briefing.py and the PRD fails the β↔γ boundary test).
- **D10 — 3211 (eval-program ε) gains a dependency on γ** so the grandfather snapshot baselines the rescoped queries, not the retired ones. This is the structural discharge of esc-3253-1's trap ("whatever these four queries do today becomes the reference"). The gate itself stays open for the operator's per-query ruling; its resolution can cite this PRD.

## Pre-conditions

- 3212 amended (D8) and restored to pending — performed in-session at decompose.
- No dependency on gate 3200, tasks 3111/3088/3201, or any vocabulary-PRD leaf: β's scoped queries and α's RRF work on the corpus as-is. 3111's topic-anchoring, when it lands at the `MemoryService.search` seam, composes with both (anchoring pins one extra result; RRF orders the rest; scoped searches still pass through the same seam).

## Cross-PRD relationships (G4)

| Seam | Owner | Note |
|---|---|---|
| `caller_agent_id`/`caller_task_id` server params + journal + `_MEMORY_INSTRUCTIONS` | **memory-eval-program ζ (3212)** | β consumes; dep β→3212; briefing-side threading and its delivered_check move to β (3212 amended in-session) |
| E1 registry briefing topics | **this PRD (γ)** | registry fixture created by eval-program β (3208, done); γ re-keys the briefing topics; eval-program PRD gains a pointer row |
| ε grandfather snapshot ordering | **this PRD (D10)** | dep 3211→γ wired at decompose (edit authorized 2026-08-05) |
| E1 metric/matcher defects (contamination-share vacuity: 489/490 scored results untopiced-and-excluded; content-hash matcher 6/196; superseded-pairs unmeasured) | **memory-eval-program** | out of scope here; pointer amendment in that PRD's open items |
| "Briefing-query redesign parked, measurement-first" | **memory-metadata-vocabulary §out-of-scope** | measurement now exists; parked line annotated as discharged → this PRD (in-session edit) |
| Topic-anchored canonical recall | **3111 (vocabulary-PRD amendment)** | 3111 amended in-session: 2026-08-05 corroborating evidence + an explicit test obligation that anchoring holds under `stores`/`categories`-scoped searches (briefing is now a named consumer) |
| Superseded-entry curation (exhibit: `523767fb` 2026-03-20 "commit or stash" advice contradicting the never-stash rule) | **3625 milestone backlog** | 3625 details amended in-session; no leaf here |

## Decomposition plan

- **α — RRF merge in `MemoryService.search`** (modules: `fused-memory/src/fused_memory/services`, `fused-memory/tests`). Remove `(is_primary, −score)` sort + primary precedence; implement D4; category-filter and planned-edge behavior unchanged; `degraded` surfacing unchanged. *Signal:* on a seeded ephemeral two-store collection (pattern `test_recon_dedup_premise.py:57-143`, real embedder), a Mem0 entry with high cosine appears in the merged top-5 of a Graphiti-primary query (fails on main today, passes with α); Graphiti results in the response carry no synthesized 1.0/0.95 scores and expose `metadata.store_rank` (tests). Prereqs: none.
- **β — briefing rescope** (modules: `orchestrator/src/orchestrator/agents`, `orchestrator/tests`). Implement D1–D3, D5–D8: per-role query table, dual-channel task context, distilled rendering, loud degradation, identity threading; delete the retired queries; merger drops the block; reviewer builder gains the task parameter. *Signal:* rendered briefing block for a fixture task contains `## Conventions & Gotchas` and `## Task Context` sections as markdown bullets with category·date·store tags and zero JSON braces (unit test with mocked MCP responses); injected per-query failure renders the named-section notice line (test); live render against the real store shows ≥1 `mem0`-tagged bullet for a task with modules (the exact assertion the old path fails — 0 Mem0 entries in any merged top-20). Prereqs: **3212** (restored pending).
- **γ — E1 registry re-key + probe refresh** (modules: `fused-memory/tests/fixtures`, `fused-memory/scripts` invocation only). Implement D9; re-run the probe read-only to emit a fresh artifact under the new topics; wire 3211→γ. *Signal:* the probe report lists one topic per β query template with per-topic canonical/claim/held-out results (no single collapsed briefing topic); the phrasing-pinning test fails when fixture and PRD templates drift (test); `get_task 3211` shows γ's id in `dependencies`. Prereqs: **β** (keys to as-landed templates).

In-session companion corrections at decompose (not leaves — performed and verified by the decomposing session): 3212 amendment + restore-pending; 3111 evidence note; 3625 exhibit; vocabulary-PRD parked-line annotation; eval-program pointer rows.

## Contract (α seam)

`MemoryService.search` post-α: results from each responding store are ranked by that store's own ordering (Mem0: cosine desc; Graphiti: backend rank). Merged ordering is RRF with K=60; ties broken by (router primary store, store-internal rank). `relevance_score` = fused RRF value (documented range: single-store rank-1 = 1/61 ≈ 0.0164; values are *ordinal*, not similarities). `metadata.store_rank` (int, 1-based) and `metadata.store_score` (float | null) are set on every result. `degraded`/`failed_stores` semantics unchanged. Error semantics unchanged (per-store failure → degraded, never raise). Consumers must not compare `relevance_score` across API versions; per-store truth lives in `metadata.store_score`.

## Boundary-test sketch

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Graphiti-primary query, high-cosine Mem0 needle | seeded two-store collection; router classifies query graphiti-primary | needle in merged top-5; `metadata.store_score` = its cosine |
| 2 | Mem0-primary query with relevant Graphiti edges | same collection, norms-shaped query | ≥1 graphiti result in merged top-k (no reverse shut-out) |
| 3 | One store times out | fault injection on graphiti task | response `degraded: true` + `failed_stores`; β renders the failed-stores line (two-sided: α produces, β consumes) |
| 4 | β conventions query | fixture task with `metadata.modules` | scoped call carries `stores`+`categories`; rendered bullets carry `[preferences_and_norms · date · mem0]` tags |
| 5 | β task context, entity exists / absent | graph node `Task <id>` present; then absent | present+exact → entity block rendered; absent (fuzzy-only return) → block suppressed, no wrong-neighbor content |
| 6 | γ tripwire under new topics | registry re-keyed; seeded collection | deleting a briefing-topic canonical flips exactly that topic's tripwire item (mirrors 3208's test pattern) |

## Out of scope

- Temporal/recency-windowed retrieval (q3's principled replacement) — no consumer named yet.
- `memory_hints` dead channel — owned by task 3254.
- E1 contamination-share/matcher/superseded-pairs metric fixes — eval-program's instrument lane (pointer amendment there).
- Fuzzy-path changes to `get_entity` server-side — β's guard is client-side by design.
- Any resolution of esc-3253-1 / task 3253 — operator's gate; this PRD is its citable rework.

## Open questions (tactical)

1. **Modules-phrase form for β's conventions query** when `metadata.modules` is empty and the title is uninformative. Suggested: repo-generic `"conventions and gotchas for this repository"`. Decide in β.
2. **Judge builder** keeps the D2+D3 set despite zero lifetime dispatches (measured) — dead-path table row costs nothing. Confirm in β rather than special-casing.
3. **RRF K** stays a module constant (60). Only revisit if boundary test 2 shows minority-store starvation at k=5.
4. **β telemetry**: whether to log rendered-block char size onto the dispatch event for ongoing cost tracking. Cheap; decide in β.
