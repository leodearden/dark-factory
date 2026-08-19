# Capability manifest — memory-briefing-and-fusion

Binds each leaf signal's asserted capabilities to evidence (G3+G6 paid once, here — 2026-08-05). Machine-readable twin: `memory-briefing-and-fusion.capability-manifest.yaml`.

## α — RRF merge in MemoryService.search

| Capability | Evidence binding | Verdict |
|---|---|---|
| The defect exists as described (primary precedence + synthesized scores) | grep `memory_service.py:3229-3233` `(is_primary, -r.relevance_score)`; `:3346` `1.0 - (i * 0.05)` — verified 2026-08-05 | PASS |
| RRF needs no Graphiti similarity API | `graphiti.search()` returns edges without scores (code comment `:3345` + call shape `:3296-3300`) — rank-only fusion by design | PASS |
| Seeded two-store test substrate exists | `fused-memory/tests/test_recon_dedup_premise.py:57-143` (real-embedder ephemeral collection pattern); cleanup-prefix caveat `fused-memory/scripts/cleanup_test_collections.py:11` | PASS |
| `degraded` surfacing already fault-only in tool response | `fused-memory/src/fused_memory/server/tools.py:2336-2341` — verified; α leaves it unchanged | PASS |

## β — briefing rescope

| Capability | Evidence binding | Verdict |
|---|---|---|
| `shared` importable from both packages | `dark-factory-shared` workspace dep in `orchestrator/pyproject.toml:21,26` and `fused-memory/pyproject.toml:23,27` | PASS |
| Search tool accepts `stores`/`categories` per call | live-tested 2026-08-05 (scoped Mem0 probes); tool schema `server/tools.py:2273-2283` | PASS |
| Scoped queries return the needles (design premise) | controlled needle tests 2026-08-05: 3/3 hit@5 scoped vs 0/3 unscoped; title+modules 5/5 vs bare-id 0/5 | PASS |
| Workflow holds task identity at every dispatch site | `events.task_id` 100% populated across all roles (14-day census); role literal in scope at each builder call site (3212 details, verified 2026-07-30) | PASS |
| `get_entity` exact-vs-fuzzy behavior supports the client-side guard | live-tested 2026-08-05: exact "Task 3211" clean; fuzzy fallback for absent "Task 3627" returns neighbors — guard = returned node name equality | PASS |
| Caller-identity server params exist before β threads them | producer: task **3212** (amended to server-side scope), dep β→3212 wired | PASS (producer-upstream, dep wired) |

## γ — E1 registry re-key + probe refresh

| Capability | Evidence binding | Verdict |
|---|---|---|
| Registry fixture + probe runner exist and run live | task 3208 merged (`d635e87d60`); first live run 2026-08-05 produced `metrics/report-20260805T093831Z.*` in 3m34s, committed at `plans/memory-eval-e1-first-live-run/e1-retrieval-health/` (task 3694) since the probe's own `fused-memory/data/memory-evals/` output is gitignored | PASS |
| Registry schema supports per-topic canonical/claims/held-out | existing entries carry all three (`memory_eval_topic_registry.json`); the collapsed briefing topic is a content choice, not a schema limit | PASS |
| 3211 is editable and not yet snapshotted | 3211 `pending`, deps [3207,3208,3209,3210], no timer installed, no grandfather artifact on disk — verified 2026-08-05 | PASS |
| Pinning test can import the templates module | same `shared` workspace-dep evidence as β | PASS |
