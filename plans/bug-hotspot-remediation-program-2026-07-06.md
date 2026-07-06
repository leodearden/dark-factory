# Bug-Hotspot Remediation Program — 2026-07-06

Program of 16 PRD streams acting on `plans/bug-hotspot-survey-2026-07-06.md`
(full per-finding evidence: `plans/bug-hotspot-survey-2026-07-06-full-findings.json`).
This document is the **authoritative G4 seam map and shared convention set** for every
PRD session in the program. Each stream's session MUST read this file before authoring.

## Streams

| ID | Slug | Scope (one line) | Mode | Wave | Upstream deps |
|----|------|------------------|------|------|---------------|
| M1 | gitops-chokepoints | `_prune_registrations` chokepoint + grep-guard test; `_abort_lane_acquisition` primitive; PROTECTED_PREFIXES registry | agent | now | — |
| M2 | supervision-quick-fixes | hoist `inspect_systemd_unit(unit, timeout)` to one module-level fn; scope DeterministicRunner escalation queries by `agent_role`; substrate-probe fail-closed; single module-lock derivation helper; StreakCounter registry | agent | now | — |
| M3 | dashboard-alignment | OutcomeKind vocabulary (fail-safe), request-scoped `now` threading in burndown/costs, shared MCP fan-out + TTL-cache helpers | agent | now | — |
| M4 | recon-project-scope | frozen `ProjectScope` dataclass (NewType ProjectId/ProjectRoot) threaded through recon signatures | agent | now | — |
| M5 | fm-cancellederror-convention | enforce CancelledError re-raise convention in fused-memory; extract the full gather-cancellation idiom | agent | now | — |
| W1 | merge-queue-reliability | landed-outbox write-ahead journal + startup reconciler; monkeypatch-path migration; SpecPermit/PermitLedger; ItemLifecycle registry; QueuedBranch | spawn | 1 | — |
| W2 | task-status-authority | shared (from,to,actor) transition table enforced in TaskInterceptor; claimant/heartbeat field; resume semantics; escalation legality table + role-derived level ceilings; promote_to_l2 gating | spawn | 1 | — |
| W3 | task-metadata-schema | versioned `shared/task_metadata.py` (typed sub-models), validated at SqliteTaskBackend write boundary; delete the 8 ad-hoc parsers | spawn | 1 | — |
| W4 | invocation-outcome | `InvocationOutcome` sum type + one `classify_invocation`; `InvokeSlot.report`; `AccountPhase` + single `_transition()`; delete steward retry-loop fork; `AccountLease` | spawn | 1 | — |
| W5 | recon-reliability | SQLite `ReconLedgerStore`; dedup-exempt system-write path; `ReconWritePolicy` at interceptor boundary; `recon_self_model.py` prompt rendering; `execution_class` on recon-filed tasks | spawn | 1 | — |
| W6 | fm-memory-identity | write-time entity identity (`_resolve_or_create_entity` + uniqueness constraint/lock); fold 4 reactive sweeps; fresh uuids in `redirect_node_edges` | spawn | 1 | — |
| W7 | verify-plan | structured `VerifyCmd`; single `derive_verify_plan()`; tool-dispatched failure classifier; `FailureCategory` enum + policy table; typed `BlockRecord`; dry-run proposals on merge-verify block path | spawn | 1 | — |
| W8 | fm-task-dedup | durable `candidate_key` partial-UNIQUE index (dedup no longer fails open to CREATE); per-ticket lifecycle struct; single update_task write-authority seam | spawn | 1 | — |
| W9 | workflow-state-machine | `WorkflowStateMachine` + `TerminalReport`; `StewardOutcome` sum type; `classify_failure` → `BlockDisposition` table; collapse the three already-merged guards onto the W1 journal | spawn | 2 | W1, W2, W4, W7 |
| W10 | harness-supervision | `proc_supervision.py` RestartPlan/execute(); `BackgroundService`/LifecycleRegistry; `TaskGroundTruth` resolver + classification table; `DeployState` typed schema; SchedulerCallbacks seam; decompose `acquire_next` into named phases | spawn | 2 | M2, W1, W2, W3 |
| W11 | worktree-lane-lifecycle | `LaneLifecycle` single-writer + durable per-lane record on the pool mount; `.task/` relocation out of the git tree | spawn | 2 | M1 |

Wave 2 sessions wire cross-batch deps against the wave-1/agent batches' real task ids
(`search_tasks` / `get_tasks` to find them; deps are bare integer ids — same project).

## Seam ownership (G4 — authoritative)

| Seam / artifact | Owner | Consumers (do NOT redefine) |
|---|---|---|
| Landed-outbox journal (MergeQueueStore), `MergeProvenance.lookup` | W1 | W9 (guard collapse), W10 (sweeps), scheduler consult |
| `QueuedBranch` typed branch identity | W1 | merge-queue consumers |
| `OutcomeKind` merge-attempt enum (in `merge_types.py`) | M3 | dashboard; W1 must not introduce a competing outcome enum |
| Task-status transition table (defined in `shared/`, enforced in TaskInterceptor) + claimant/heartbeat field | W2 | W9 + escalation server consume the SAME table as thin validators — never three tables |
| Escalation action legality `(action, level, category) → TaskEffect` + role-derived level ceilings | W2 | harness `_on_escalation_resolved` |
| `shared/task_metadata.py` (TaskMetadata v1 + sub-models incl. BeforeDone, DoneProvenance, MemoryHints, ExternalDep) | W3 | everyone; W10's DeployState is a W3 sub-model **defined by W10**, registered into W3's schema |
| `DeployState` deterministic-deploy phase enum + persisted verify baseline | W10 | M2 must NOT introduce a deploy-state enum (M2 = query-scoping only); the scheduler-reviewer's `metadata.deterministic_state` proposal is MERGED into W10's DeployState — one mechanism |
| `InvocationOutcome`, `classify_invocation`, `InvokeSlot.report`, `AccountPhase`, `AccountLease` | W4 | W9's BlockDisposition consumes AgentFailureKind/outcomes; steward keeps only session bookkeeping |
| `inspect_systemd_unit(unit, *, timeout_secs)` single helper | M2 | W10's proc_supervision imports/relocates it — never a second copy |
| `ReconLedgerStore`, dedup-exempt system writes, `ReconWritePolicy`, `recon_self_model.py`, `execution_class` | W5 | fm-recon stages/prompts |
| `ProjectScope` | M4 | W5 threads it where it touches the same signatures (W5 tasks that share files with M4 tasks declare deps on them) |
| Entity write-time identity + `redirect_node_edges` uuid semantics | W6 | fm-recon reads benefit; no other stream touches graphiti_client |
| `VerifyCmd`, `VerifyPlan`, `FailureCategory`, tool-dispatched classifier, `BlockRecord` | W7 | W9 consumes BlockRecord in `_mark_blocked`; merge_queue block path spawns dry-run investigation (W7 owns that wiring) |
| `candidate_key` uniqueness + update_task write-authority seam | W8 | task_curator/interceptor |
| `_prune_registrations` chokepoint, `_abort_lane_acquisition`, PROTECTED_PREFIXES | M1 | W11 builds LaneLifecycle on top; W11 tasks depend on M1's |
| `.task/` relocation (`TaskArtifacts` single path-derivation owner) | W11 | merge_gates/verify/git_ops scrub layers become dead code (W11 removes) |
| `WorkflowStateMachine`, `TerminalReport`, `StewardOutcome`, `BlockDisposition` | W9 | harness consumes TerminalReport (replaces `_last_block_*` attr pokes) |
| `proc_supervision.py` (RestartPlan), `BackgroundService` registry, `TaskGroundTruth` | W10 | service_restart + deterministic_runner delegate; sweeps become thin |

## Resolved design decisions (do not relitigate — from the survey synthesis)

1. **Transition-table home**: the enforcement floor lives at the single durable write
   chokepoint (fused-memory TaskInterceptor); the table itself is defined in `shared/`;
   escalation's `resolve_issue` and workflow's local machine consume the SAME table as
   thin validators. Three tables would recreate the drift they fix.
2. **Merge-provenance substrate**: MergeQueueStore (already durable, already journals the
   accept side), with **write-ahead ordering** — the landed row is fsynced BEFORE the CAS
   main advance. Write-after re-opens the crash window the whole chain exists to close.
3. **Escalation level gate**: role-mapped ceilings for identified automation callers;
   **header-less sessions remain the full-capability human channel** (the deployed
   L2-closure convention — a naive default-deny would repeat the esc-2087-2 lockout).
   Gate `promote_to_l2`'s create side the same way.
4. **Duplication doctrine**: `task_interceptor.py:115`'s "duplication is cheaper than
   cross-package coupling" is RETIRED. `shared/` is the sanctioned home for cross-process
   contracts (precedents: `shared.usage_gate`, `shared.locking`).
5. **Dashboard outcome vocabulary**: `OutcomeKind` is owned by orchestrator
   `merge_types.py`. Dashboard consumes it fail-safe — if a hard import is undesirable,
   invert to terminal-unless-listed (explicit ACTIVE_ONLY allowlist), since new terminal
   outcomes are added far more often than new active ones.
6. **Fused-memory restarts** in deploy capstones: out-of-cgroup
   `systemctl --user restart fused-memory.service`; do NOT use
   `restart-fused-memory.sh --drain` (hung — task 2090). Orchestrator restarts follow the
   deterministic task-kind conventions in CLAUDE.md (2064/2105 are fixed and deployed).

## Shared conventions (every session)

- **Identity**: `project_id="dark_factory"`, `project_root="/home/leo/src/dark-factory"`,
  write-tag `agent_id="claude-prd-<slug>"`.
- **PRD path**: `plans/<slug>-prd.md`, committed to git in the same session (task agents
  run in worktrees branched from main and must be able to read it). Capability manifest
  beside it per the prd skill.
- **Filing**: every task `planning_mode=True`; wire ALL deps (intra- and cross-batch)
  while `deferred`; flip the whole batch in ONE bulk `commit_planning`.
- **metadata.files is ALWAYS file-level** — never directories (Lock-charter Contract-1;
  dirs are LOUD-rejected).
- **complexity="simple"** only for single-coherent-edit leaves with named target files;
  when unsure omit. `metadata.force_full_path=true` for deceptively-simple-looking
  design tasks.
- **Dedup before filing**: `search_tasks` for each planned leaf; overlapping open tasks →
  depend on or supersede them explicitly (notably task 2085, blocked, which W5's
  `execution_class` mechanism supersedes — W5 must say so in its PRD and resolve 2085's
  disposition).
- **Autonomy**: never block on `AskUserQuestion` — the operator is AFK by design. Take
  the safe default consistent with this document, record it under the PRD's
  Open questions. Proceed author → commit → decompose → queue in one session without
  pausing for confirmation.
- **Git mechanics**: commit with `git commit -F <msgfile>`; pre-commit can exceed 2 min —
  use `setsid git commit ... &` + poll the log if needed; direct-to-main commits can race
  the live merge queue's ref lock — on failure re-add and retry.
- **Survey freshness**: findings were code-verified 2026-07-06, but re-verify any
  file:line you build a task on (main moves fast here). Statuses checked 2026-07-06:
  2091/2097-2100/2105/1146/1151 all done — the latent bugs in the survey stand DESPITE
  those (e.g. 2091 fixed only the runner's inspector copy, not the harness duplicate).

## FILED — program status as of 2026-07-06 ~16:30

All 16 streams authored, gate-walked, decomposed, and queued (~110 tasks). Anchors
below; each PRD + capability manifest on main carries the authoritative leaf list
(locate by PATH — an index race put several under misattributed commit messages).

| Stream | PRD | Task anchors |
|---|---|---|
| M1 | plans/gitops-chokepoints-prd.md | 2185(α, in-progress) 2190 2194 2199(δ) 2205(ε) |
| M2 | plans/supervision-quick-fixes-prd.md | 2119(α, in-progress) 2120 2121 2122 2124 |
| M3 | plans/dashboard-alignment-prd.md | 2165 2170 2174 2181 2187 2192 2218 |
| M4 | plans/recon-project-scope-prd.md | 2144 2146 2150 2152 |
| M5 | plans/fm-cancellederror-convention-prd.md | 2130 2135 2140 2145 2149 2151 |
| W1 | plans/merge-queue-reliability-prd.md | 15-task linear spine 2153(α=journal)…2183 |
| W2 | plans/task-status-authority-prd.md | incl. 2163(τ1 StrEnum) 2168 2182 |
| W3 | plans/task-metadata-schema-prd.md | 2158(α=schema+ext-point)…2184(θ2 enforce-flip gate) |
| W4 | plans/invocation-outcome-prd.md | 2127(α)…2143(κ); 2128=AccountPhase |
| W5 | plans/recon-reliability-prd.md | 2219…2233; cross-deps κ/λ→M4 2150/2149 |
| W6 | plans/fm-memory-identity-prd.md | 2198(α, AMENDED — group_id filter) 2202 2207 2210 2213 |
| W7 | plans/verify-plan-prd.md | 2123(α)…2148; ζ=2138 BlockRecord |
| W8 | plans/fm-task-dedup-prd.md | 2186…2212(Z=deterministic deploy) |
| W9 | plans/workflow-state-machine-prd.md | 2245…2253 (deps → 2153/2168/2123/2127/2138/2158) |
| W10 | plans/harness-supervision-prd.md | 2235…2244 (deps → 2119/2120/2124, 2153, 2182, 2158) |
| W11 | plans/worktree-lane-lifecycle-prd.md | 2254…2264 (η=2264 activated, dep→2199) |

Coordinator interventions (2026-07-06 ~16:22): W6-α task 2198 amended pre-dispatch with
the mandatory `n.group_id = $group_id` property filter (task-2115 cross-graph leak would
otherwise be destructively auto-merged — see memory project_w6_automerge_hazard_2115);
2210 got the matching advisory; W11-η 2264 activated per its recorded trigger (M1-δ WAS
filed as 2199 — a search_tasks/id-range misdiagnosis said otherwise, corrected in
memory); W11-γ 2256 corrected re M1-ε=2205. Known G6 catch during authoring: the
survey's "~0.92 Mem0 dedup" premise is FALSE on main (infer=False bypasses dedup) — W5
anchored its compensation-deletion on ledger authority instead, with an empirical
verification leaf (2221).
