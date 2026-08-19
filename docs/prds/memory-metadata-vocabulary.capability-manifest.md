# Capability manifest — memory-metadata-vocabulary

Machine-readable twin: `memory-metadata-vocabulary.capability-manifest.yaml` (same stem). Batch authored 2026-07-29, task ids stamped by `commit_planning`; all leaves filed `planning_mode`. Substrate verified 2026-07-29 against main `d42f510669` (three-agent verification pass recorded in the PRD §6). Amendments to the five deferred tasks (3111/3112/3129/3133/3136) and to 3127/3135/3088/3137 are re-wiring of already-filed tasks, not manifest rows.

| Leaf | Task | Load-bearing capabilities | Verdict |
|---|---|---|---|
| α census | (stamped) | census script exists and enumerates key/kind/shape populations | PASS |
| β vocabulary | (stamped) | registry wired at the service seam; contract-fixed `MemoryMetadataValidationError`; grep-anchored census line; warn-storm escape (INV-4) | PASS |
| γ supersedes | (stamped) | shared `normalize_supersedes()` helper; scalar writer migrated to list | PASS |
| δ parent lifecycle | (stamped) | contract-fixed `ParentHasChildrenError`; write-time parent liveness | PASS |
| ε canonical/topic | (stamped) | contract-fixed `CanonicalUniquenessViolation`; one shared slug constant across memory + config namespaces | PASS |
| ζ E2 bake-off | (stamped) | bake-off script; committed decision-table report; audit-recall measurement vs α/3130 fixture | PASS |
| η ratification gate | (stamped) | pure deterministic gate (3169-pattern: `always_escalates`, no `before_done`) | PASS |
| θ retro stamping | (stamped) | stamp script; `update_memory` delivered upstream by 3088 (dep wired, DAG-direction verified) | PASS |
| ι writer instructions | (stamped) | `_MEMORY_INSTRUCTIONS` names the vocabulary (incl. `supersedes`); registry↔prompt pinning drift test | PASS |

No FAIL bindings. Seams honored: 3055/3088 (reserved-key bottom layer + `update_memory` — D12 defensive extraction), 3111/3112/3129/3133/3136 (amended + gated per PRD §8), 3127/3135 (amendment notes), 3108 (citation repointing untouched), 3084 (gate-closure enforcement seam respected by construction), write-path PRD §1/D4/§8 (companion commit).
