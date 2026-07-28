# Capability manifest — memory-write-path-convergence

Human-readable twin of `memory-write-path-convergence.capability-manifest.yaml` (the sidecar carries the mechanical `delivered_check`s). Batch filed 2026-07-28, tasks 3127–3142, all `planning_mode`. Substrate rows verified against live source in PRD §6 (2026-07-27/28).

| Leaf | Task | Load-bearing capabilities | Verdict |
|---|---|---|---|
| α | 3130 | calibration script derives T_high/T_low from measured distributions; labeled 89-entry dataset faithful to curator dispositions (incl. genuinely-distinct + pseudo-contradiction labels) | PASS |
| β | 3127 | triage behind contract-fixed `write_triage_enabled`; fail-open + INV-4 storm escape; canonical never mutated | PASS |
| γ | 3128 | closed 4-way routing vocabulary; accuracy eval vs α labels is the flag-flip arbiter (no pre-asserted floor — G6) | PASS |
| δ | 3129 | add-only child records (`parent_id`, no 3055 dependency); grouped single-document reads | PASS |
| ε | 3131 | inverted instruction rendered in every role briefing ("write freely; the server deduplicates") | PASS |
| ζ | 3135 | runtime topic-cluster store merged with config seeds; auto-seeded from every consolidate call — manual hop dead | PASS |
| η | 3132 | truncated-id delete hard-errors (API enforcement where DF 1144's prompt fix failed) | PASS |
| θ | 3133 | `consolidate_memories` exists; closure proof returns survivors, never a claimed closure | PASS |
| ι | 3134 | Stage-1 merges via the op; blanket `recon-stage-*` exemption retired | PASS |
| κ | 3136 | timer unit committed (`OnCalendar`); gate filing cites deterministically enumerated clusters | PASS |
| λ | 3137 | search hits carry `agent_id`/`task_id`/`created_at` | PASS |
| μ | 3138 | style-based injection flag requires prior `agent_id` provenance check | PASS |
| ν | 3139 | `reexamine_when` validated at the submit boundary (INV-1) | PASS |
| ξ | 3140 | terminal transitions flag citing memories (re-corroborated first — INV-3); flags never delete | PASS |
| ο | 3141 | markup rejection names the matched pattern; rejection storm escalates naming DF 3083 | PASS |
| π | 3142 | false completion claims ingest tagged `unverified_claim` + flag | PASS |

No FAIL bindings. Seams honored: DF 3083 (XML cure/sweep — ο is containment), DF 3055 (update primitive — batch is add-only throughout), DF 3108 (citation repointing — θ does not claim it), DF 1144 (η's failed prompt-level predecessor).
