# Recon Reliability — take the LLM out of reconciliation's reliability path

**Status:** deferred (bug-hotspot remediation program 2026-07-06, stream **W5**, wave 1)
**Slug:** `recon-reliability` · **Path:** `plans/recon-reliability-prd.md`
**Program doc (authoritative G4 seam map):** `plans/bug-hotspot-remediation-program-2026-07-06.md`
**Survey evidence:** `plans/bug-hotspot-survey-2026-07-06.md`, cluster **fm-recon** in
`plans/bug-hotspot-survey-2026-07-06-full-findings.json` (findings 0,1,3,4,5).
**Approach:** **B + H** (high-stakes; contract + two-way boundary tests). See §Contract and §Boundary-test sketch.

---

> **Reconciliation correction (2026-07-06, at decompose):** an earlier draft of this PRD said W5
> **supersedes / cancels task 2085**. That is **retracted.** Task 2085 was **ratified by a human the
> same day** (MECH B: a new TaskCurator `action='route_deterministic'` + `operational_ask_registry.yaml`
> + deterministic pure-gate at the recon Stage-2 emit path — the submit_task-boundary approach was
> **explicitly rejected** for project-wide blast radius) and is now **pending/active**. **Do NOT cancel
> 2085.** 2085 **owns operational-ask routing off the TDD pipeline**; W5's `execution_class` task (η) is
> narrowed to the complementary **machine-checkable declaration layer** (require + validate + persist the
> field) that 2085's routing and W5's premise-lint (ξ) consume. The `execution_class`→deterministic
> *routing coercion* is removed from η. See §5 decision #3, §7, §10 η, §12 Open Q5.

## 1. Goal (user-observable)

Reconciliation's *control-plane* state (markers, suppressions, counters, cycle summaries) and
its *write invariants* stop living inside an amnesiac LLM + an eventually-consistent
deduplicating vector store, and move to transactional SQLite + server-side enforcement.

After this PRD lands, an operator/developer observes:

- **A durable ledger table exists and is authoritative.** `reconciliation.db` (or a sibling
  `.db` under the reconciliation `data_dir`) contains a `recon_ledger` table; marker replacement
  is one UPSERT (a duplicate identity tuple leaves **exactly one** row, verifiable through the
  store's own read API), and one `DELETE` pass performs all GC. The four Mem0 GC sweeps, the
  flag_dedup write/confirm/delete dance, the confirmation circuit-breaker, and the in-batch memo
  are **gone from the source tree** (grep returns nothing).
- **Recon's own cycle-summary write is deterministic Python, not an LLM turn.** After a stage
  completes, exactly one cycle-summary record exists, written by Python from the `StageReport`.
  The `summary_nonce`/`retry_nonce` prompt directives, `generate_summary_nonce` /
  `build_summary_nonce_section`, and the `verify → repair → reconstruct` chain (three functions +
  their `run()` wiring + four stats keys) are deleted.
- **Illegal recon task-writes are rejected mid-run with a structured error the LLM reads** — a
  `recon-stage-*` `update_task` on a terminal task, or a status write on a task with a live
  workflow, returns a `{error, error_type, …}` dict (same surfacing as
  `DarkFactoryPathScopeViolation`) instead of landing and being reclassified after the fact.
  `_apply_post_flight_guards` shrinks from a ~650-line forensic suite to counter reconciliation.
- **Stats are computed, not reported.** A stage's `report.stats` counters are derived from the
  write journal; the LLM's self-reported numbers survive only under `stats['_reported']` for the
  judge's divergence signal. The `stats_verifier` alias map and both flag-counter completeness
  checkers are deleted.
- **Recon-filed tasks carry a machine-checkable `execution_class`.** Every `recon-stage-*`
  `submit_task` must set `metadata.execution_class ∈ {code_tdd, operational, decision}` (rejected
  otherwise with a structured error); `operational`/`decision` asks are routed off the
  architect+TDD pipeline (they become human-gated deterministic tasks), so live-data/live-mutation
  work stops churning as architect "unactionable" refusals. **η requires + validates the declared
  `execution_class`; the operational→off-TDD routing is owned by the ratified task 2085 (see the
  Reconciliation correction above).**
- **Prompt/code drift on recon's self-model becomes structurally impossible.** A
  `recon_self_model.py` module exports the mechanism constants + invariants and *renders* the
  prompt sections describing them; the stage prompts import the rendered text. The same module
  exposes assertable predicates a premise-lint runs against recon-authored task descriptions
  before filing.

**Consumers (G1):** stage1/stage2 runners (`stages/memory_consolidator.py`,
`stages/task_knowledge_sync.py`); `flag_dedup.py` callers; the TaskKnowledgeSync GC path; the
reconciliation stats/judge consumers; the fused-memory TaskInterceptor (policy enforcement); the
Stage-1/Stage-2 prompts (rendered self-model text); the recon submit_task path (execution_class +
premise-lint). Every mechanism this PRD introduces has a named in-system consumer — see §Contract.

---

## 2. Background

The fm-recon survey (871+ commits touching `reconciliation/` since 2026-04-01 — the single hottest
fix surface in fused-memory) traces one root cause with five faces: **recon keeps its own
control-plane in Mem0 and polices its own LLM writes after the fact.** Each face has spawned a
long, still-open compensation chain:

- **Markers/suppressions/counters/summaries as Mem0 memories** (finding 0). `flag_dedup.py`'s
  241-line docstring (`flag_dedup.py:1-241`) narrates seven stacked compensations for a store with
  no atomic replace, no read-after-write, and silent similar-write drops: best-effort
  write-then-delete replacement (`flag_dedup.py:1023-1097`), in-batch memoization (task-1978,
  `:1004-1009`/`:1194`), post-write read-back confirmation with one retry (task-1400,
  `:349-465`), a per-invocation confirmation circuit-breaker (task-1412, `:319`/`:849-936`), a
  `limit=50` reclamation bound, self-healing-on-next-cycle, and a project-scoped suppression gate
  (`:468-586`). `TaskKnowledgeSync` needs **four** GC sweeps (`_sweep_stale_fixc_markers`
  `:1192`, `_sweep_stale_flag_markers` `:1246`, `_sweep_terminal_task_flag_markers` `:1424`,
  `_sweep_stale_persistence_markers` `:1624`). Fix chain: 1097/1120/1121/1122/1128/1134/1146,
  1786, 1944, 2047, 2095, 2103; a 16-record manual bulk GC incident is cited at
  `task_knowledge_sync.py:2698-2708`.

- **Recon's own per-cycle writes routed through the LLM-facing dedup pipeline** (finding 1). The
  prompts teach the LLM to prepend a CSPRNG `summary_nonce` and, on retry, a deterministic
  `retry_nonce` (`prompts/stage1.py:121-185`, `prompts/stage2.py:236-302`;
  `generate_summary_nonce`/`build_summary_nonce_section` `cli_stage_runner.py:240-305`). Python
  then compensates post-hoc: `_verify_stage2_summary_written` (`task_knowledge_sync.py:1825`),
  `_repair_stage2_summary_stage_metadata` (`:1890`), `_reconstruct_stage2_summary` (`:2040`),
  wired at `:2629-2680`, with four stats keys (`:2628`,`:2673`,`:2674`,`:2678`). A prior nonce
  fix re-triggered the very dedup it aimed to defeat (commit 2777f2b227); nonce-hardening tasks
  1590/1796/1821, reconstruction tasks 1963/1964.

- **Stage-2 write invariants enforced post-hoc by journal forensics** (finding 3). The Stage-2 LLM
  holds unrestricted task-write tools; `_apply_post_flight_guards`
  (`task_knowledge_sync.py:2730-2987`) re-fetches every touched task and *reclassifies* journal
  ops because — its own words (`:279-281`) — "skip the write entirely cannot be enforced post-hoc."
  The interceptor already enforces a terminal FSM for `set_task_status` (terminal-exit `:702`,
  bulk-reset `:685`, done-provenance `:744`) but has **no per-agent policy** for recon's
  `update_task`-on-terminal or live-workflow cases. Fix chain: 1137, 1655, and the 9× stranded
  hardening 886/911/912/913/916/1087/1092/1106/1114.

- **LLM-authored counters reconciled by three separate mechanisms** (finding 4). Stage agents
  self-report counts; reality is journal-derived. `stats_verifier.py` overwrites via `_OP_TO_STAT`
  (`:31-42`) plus a drift-patch alias map `_STAT_ALIASES` (`:47-49`);
  `_check_flag_counter_completeness` (`task_knowledge_sync.py:591`) and
  `_check_mem0_flag_counter_completeness` (`:644`) audit separately; post-flight guards decrement
  reclassified writes. Every counter not yet in the map is a latent drift bug (task 1150, 2078).

- **Recon files tasks with no machine-checkable premise or executability contract** (finding 5).
  Stage 2 can `submit_task` into any known project with only the curator (dedup/combine) and
  path-scope gate; the self-model lives in prompt prose that provably drifts (commit a2327a65e1
  fixed a system/user prompt disagreement on `count_memories_by_metadata` arg shape; the
  false-premise batch 2083/2092/2093 was filed on a wrong `stage1_flag_marker`/`run_id` model).
  Filed tasks carry no execution class, so operational/live-data work lands in the architect+TDD
  pipeline that cannot act on it (task 2085, recurring class 1945/1946/1939/2081/2082).

Finding 2 (ProjectScope typing) is **M4's** territory — out of scope here; W5 declares deps where
files overlap (§Cross-PRD).

### 2.1 A substrate correction that reshapes subject 2 (read before decomposing)

The survey's finding-1 premise — "`MemoryService.add_memory` silently dedups at ~0.92 cosine" — is
**not straightforwardly true on main.** fused-memory's `Mem0Backend.add()` pins `infer=False`
(`backends/mem0_client.py:103-110`); in the vendored Mem0 (`mem0/mem0/memory/main.py:1417`) the
`infer=False` branch calls `_create_memory` directly with a fresh `uuid4` point
(`:2136-2154`) and **never reaches** the near-duplicate / update-vs-add block (which lives only in
the `infer=True` path, `:1508-1577`). `memory_service.py:57-65`
(`_MEM0_ADD_INFER_PINNED_FALSE=True`, task-1974) states every write "always returns exactly one
result with an id — so an empty result is always anomalous." The `~0.92` figure appears **only in
prose** (comments/prompts), never as executable code. The one place assuming an `infer=False`
write can be dedup-dropped is a retry comment at `task_knowledge_sync.py:2110`, in tension with the
task-1974 invariant.

**Consequence (a deliberate design decision — see §Resolved #1):** we do **not** predicate the
deletion of the nonce / verify / repair / reconstruct machinery on the dedup premise. We predicate
it on **the ledger becoming the authoritative store** (SQLite, PK, read-after-write consistent),
which is unconditionally reliable. Once markers/suppressions/counters/cycle-summary records are
authoritative in the ledger, the Mem0 write is demoted to a best-effort searchable mirror whose
loss is harmless — so the compensation chain is deletable **whether or not** Mem0 ever dedups. The
"dedup-exempt system-write path" (subject 2) is then a *thin, explicit, server-enforced* guarantee
that the mirror lands, plus an empirical confirmation task that binds the premise instead of baking
it into a RED test (G6).

---

## 3. Sketch of approach

Five interlocking mechanisms, built foundations-first so the consumer-side deletions land on real
substrate:

1. **`ReconLedgerStore`** — a new hand-rolled SQLite store
   (`fused-memory/src/fused_memory/reconciliation/recon_ledger.py`, template
   `middleware/ticket_store.py`) with `PRIMARY KEY (project_id, record_kind, task_id, flag_type,
   run_id)`. Marker replacement = one `INSERT … ON CONFLICT(<pk>) DO UPDATE`; suppression lookup =
   indexed query; GC = one `DELETE WHERE expires_at < :now OR <terminal-referenced>`. Mem0 keeps a
   best-effort searchable mirror only. `flag_dedup.py` shrinks to signature-compute + ledger calls.

2. **Dedup-exempt system-write path** — `MemoryService.add_system_record(...)` (or
   `add_memory(..., dedup_exempt=True)`) permitted only for `recon-stage-*` agent_ids, enforced
   server-side at the existing `server/tools.py` recon-stage gate (precedent: the four
   `agent_id.startswith('recon-stage-')` branches at `tools.py:403/787/799`, and the
   count-snapshot write-gate task 1547). It writes via the fresh-uuid direct-insert path so a
   future re-enabling of Mem0 dedup can't silently re-break recon. Cycle summaries are written once
   by Python from the `StageReport`; the nonce + verify/repair/reconstruct chain is deleted.

3. **`ReconWritePolicy` at the interceptor boundary** — consulted inside
   `TaskInterceptor._apply_status_transition` and `update_task` when the caller agent_id is
   `recon-stage-*`: reject `update_task` on a task whose live status ∈ `TERMINAL_STATUSES`; reject
   status writes when `live_workflow_detector.is_workflow_live_for_task` is true; require a
   payload-issued snapshot token whose staleness the server checks (replacing the post-hoc
   stall-guard freshness gate). Rejections return structured dicts (the
   `DarkFactoryPathScopeViolation` surfacing). `_apply_post_flight_guards` then shrinks to counter
   reconciliation. **Prerequisite:** the caller agent_id must be threaded onto the task-write path
   (it is present on *memory* writes via `_resolve_identity` but **not** on
   `set_task_status`/`update_task`/`submit_task` today — net-new plumbing; §Contract, seam W2).

4. **`recon_self_model.py`** — exports mechanism constants (marker kinds + lifecycle, fingerprint
   identity fields, exact MCP call shapes) and **renders** the prompt sections describing them;
   prompts import the rendered text (drift becomes impossible). Exposes assertable predicates
   (e.g. `run_id_is_fresh_per_run()`, `markers_deleted_only_by_gc()`) for a premise-lint over
   recon-authored task descriptions before filing.

5. **`execution_class` on recon-filed tasks** — required `metadata.execution_class ∈ {code_tdd,
   operational, decision}`, enum-validated at the `submit_task` boundary for `recon-stage-*`
   callers (template: `deterministic_task_error` in `middleware/deterministic_task_guard.py:79`),
   and **persisted to metadata** as the machine-checkable executability contract. **The
   operational→off-TDD routing is owned by the ratified task 2085** (MECH B: `route_deterministic`
   + `operational_ask_registry.yaml`); η is narrowed to the **declaration layer** and does NOT
   re-implement routing (see the Reconciliation correction above). The declared class is the
   explicit input 2085's routing and W5's premise-lint (ξ) consume.

**Computed stats** (finding 4) rides alongside: a `derive_stage_stats(ops, stage_agent_id)`
function beside the write journal produces every write-shaped counter; `report.stats` is populated
from it; `stats_verifier`'s alias map and both flag-counter checkers die.

---

## 4. G3 — substrate verification (each confirmed on main 2026-07-06)

| Assumed capability | Verified? | Evidence |
|---|---|---|
| Transactional SQLite substrate to build the ledger on | **yes** | `shared/async_sqlite_base.py` (`connect_daemon` :89, `apply_full_durability_pragmas` :57 — WAL+`synchronous=FULL`+autocheckpoint); five hand-rolled stores; **template `middleware/ticket_store.py:83`** (SCHEMA const, `initialize`, `_txn`, idempotent `ALTER`); UPSERT precedent `journal.py:296/570`, `event_buffer.py:264/274` |
| Server sees caller agent_id at write time | **yes for memory writes; NO for task writes** | `_resolve_identity` `server/tools.py:409` (reads `clientInfo.name`) used by `add_memory` (:769); **but** `set_task_status`/`update_task`/`submit_task` handlers do **not** accept `agent_id`/`ctx`, and interceptor `_apply_status_transition` (:619) / `update_task` (:3118) have no agent_id param → **queued as prerequisite task ε** |
| `recon-stage-*` agent_id vocabulary | **yes** | `StageId` `models/reconciliation.py:31-36`; `f'recon-stage-{stage_id.value}'` `stages/base.py:135`, `task_knowledge_sync.py:2758/4150` → `recon-stage-{memory_consolidator,task_knowledge_sync,integrity_check}` |
| Structured-error surfacing template (`DarkFactoryPathScopeViolation`) | **yes** | `path_scope_guard.py:93/125-154` builds `{error, error_type, …}`; returned (not raised) and serialized verbatim by FastMCP (`tools.py:2697` set_task_status, `:3202` update_task). **Closest recon template:** the `recon-stage-` add_memory reject dicts at `tools.py:790-807` |
| `live_workflow_detector.is_workflow_live_for_task` | **yes** | `services/live_workflow_detector.py:215` → `detect_live_workflow(...).is_live` (git worktree + recent commit + orchestrator PID-lock); already consumed by Guard 5 `task_knowledge_sync.py:570` |
| Write journal to derive stats from | **yes** | `WriteJournal` `services/write_journal.py:56`; op shape `write_ops` (`agent_id`, `operation`, `causation_id=run_id`, `params`, `success`, …); query `get_ops_by_causation(run_id)` :223; existing `_OP_TO_STAT` `stats_verifier.py:31` |
| Enum-guard template for `execution_class` at submit_task | **yes** | `deterministic_task_error` `middleware/deterministic_task_guard.py:79-154` (enum reject → `{error, error_type:'ValidationError', hint}`), wired at `server/tools.py:2804-2810`; `inject_task_kind` :220 |
| Mem0 dedup-bypass mechanism for a system-write path | **yes (already the default)** | `infer=False` skips dedup (`mem0/…/main.py:1417`); lowest-level insert `_create_memory` fresh-uuid `:2136-2154`; raw `AsyncQdrantClient` via `mem0_client._get_async_qdrant()` :213 — direct upsert available |
| Premise "recon writes are silently dedup-dropped" | **UNCONFIRMED — bound to task γ, not assumed** | see §2.1; only `task_knowledge_sync.py:2110` assumes it, contradicting task-1974 (`memory_service.py:57-65`). Confirmation is an explicit upstream task; the compensation deletion does not depend on it |
| `recon_pool` stage→pool constants | **yes (in-progress leaf 2140)** | being extracted to `reconciliation/recon_pool_map.py` (task 2140, in-progress) — W5 imports it; dep at decompose if unmerged |

No unverified substrate remains. The one genuine prerequisite (caller agent_id on the task-write
path) is queued as task ε upstream of the ReconWritePolicy task (G3 resolution (b)).

---

## 5. Resolved design decisions (do not relitigate)

1. **Compensation deletion is anchored on ledger-authority, not the dedup premise.** The
   nonce/verify/repair/reconstruct chain and the four GC sweeps are deleted because control-plane
   state moves to the read-after-write-consistent ledger, making the Mem0 path a best-effort
   mirror. Whether Mem0 dedups an `infer=False` write is validated empirically (task γ) and bound
   in the manifest — never baked into a RED test (§2.1, G6).

2. **`add_system_record` makes the exemption explicit and enforced, even though `infer=False`
   already inserts fresh points.** Today "no dedup" is an accidental consequence of a pinned flag;
   the named server-side path (recon-stage-only) turns it into a contract a future change can't
   silently break. It writes via the fresh-uuid direct-insert primitive.

3. **Operational-ask routing off the TDD pipeline is owned by the ratified task 2085, NOT by η.**
   *Superseded by the Reconciliation correction:* an earlier draft had η coerce
   `execution_class ∈ {operational, decision}` to a deterministic pure-gate at the submit_task
   boundary. That routing is now 2085's job — the human ratified MECH B (a `route_deterministic`
   TaskCurator action + an `operational_ask_registry.yaml` mirroring `cancelled_premise_blocklist`,
   inserted at the recon Stage-2 emit path; the submit_task-boundary approach was **explicitly
   rejected** for project-wide blast radius). W5's η is narrowed to **requiring + validating +
   persisting** the declared `execution_class` (the machine-checkable contract) — the explicit input
   2085's routing consumes. W5 does not re-implement routing and does not cancel 2085.

4. **Ledger is the single-writer for marker lifecycle; Mem0 mirror is best-effort and lossy-safe.**
   No read path consults Mem0 for control-plane truth after cutover. Suppression lookups,
   marker-existence checks, and counter reads hit the ledger.

5. **`ReconWritePolicy` composes with W2's transition table, it does not replace it.** W2 owns the
   generic `(from,to,actor)` transition table + claimant/heartbeat and the actor/agent_id
   plumbing; W5's policy is the recon-stage-specific overlay (terminal `update_task`,
   live-workflow status, snapshot-token freshness) inserted as an independent early-return gate in
   the same method. Two gates, two structured-dict returns, minimal merge surface. W5 builds the
   minimal agent_id plumbing it needs (task ε) and W2 converges onto it (§Cross-PRD).

6. **Cutover safety.** Live recon cycles run continuously. The ledger is introduced **write-both /
   read-new**: after `ReconLedgerStore` lands (α), marker/summary writes go to the ledger *and* a
   best-effort Mem0 mirror, but all *reads* switch to the ledger; the Mem0-path code is deleted
   only once the ledger is the read source (the consumer tasks ι/κ/λ). A one-shot migration sweep
   is unnecessary because markers are short-lived (14-day GC) and self-repopulate each cycle — the
   ledger simply becomes authoritative going forward; residual Mem0 markers age out under the
   final DELETE pass. Deploy is a deterministic fused-memory restart capstone (task π) using
   out-of-cgroup `systemctl --user restart fused-memory.service` (program decision #6 — **not**
   `restart-fused-memory.sh --drain`, hung per task 2090).

7. **`execution_class` is stored as a metadata field now; W3 types it later.** The field lives in
   the untyped `metadata` dict, validated by W5's guard. W3 registers it as a typed sub-model in
   `shared/task_metadata.py` in its own stream (§Cross-PRD).

---

## 6. Pre-conditions for activating

- **M4 (recon-project-scope) tasks that touch `task_knowledge_sync.py`, `harness.py`,
  `stages/base.py`** should land first where they overlap W5's edits (deps wired at decompose via
  `search_tasks`). W5's `task_knowledge_sync.py` consumer tasks (κ/λ/μ) depend on M4's corresponding
  task to avoid a rebase war on the hottest file in fused-memory.
- **W2 (task-status-authority)** interceptor/actor-plumbing task, if filed, is a dep of W5's
  ReconWritePolicy (task ζ) so they serialize on `task_interceptor.py`; otherwise W5 builds the
  minimal plumbing and W2 converges.
- **Task 2140** (`recon_pool_map.py`, in-progress) — W5's ledger/summary code imports it; dep at
  decompose if still unmerged.
- No novel external substrate. All prerequisites are intra-repo and enumerated in §4.

---

## 7. Cross-PRD relationship (G4)

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| M4 `recon-project-scope` | consumes | `ProjectScope` threaded through `task_knowledge_sync.py` / `harness.py` / `stages/base.py` signatures | **M4** | W5 consumer tasks (κ/λ/μ) depend on M4's shared-file tasks; wire at decompose |
| W2 `task-status-authority` | consumes + produces | actor/agent_id on the task-write boundary + `_apply_status_transition` enforcement point | **W2** (table + actor); W5 owns recon overlay | W5 builds minimal recon agent_id plumbing (ε) + composes ReconWritePolicy (ζ) as an independent gate; dep on W2 interceptor task at decompose; W2 converges |
| W3 `task-metadata-schema` | produces | `execution_class` enum/field | **W3** registers typed field; W5 defines mechanism | W5 ships untyped `metadata.execution_class` now (guard-validated); W3 types it later |
| task 2140 (in-progress) | consumes | `recon_pool_map.py` stage→pool constants | 2140 | W5 imports; dep if unmerged at decompose |
| task 2085 (**pending, ratified**) | consumes / coordinates | operational-ask routing off the TDD pipeline | **2085** (MECH B: `route_deterministic` + `operational_ask_registry.yaml`) | **NOT superseded** (corrected 2026-07-06). 2085 owns routing; W5 η provides the complementary `execution_class` declaration+validation the routing consumes. **Do not cancel 2085.** |
| task 2092 (blocked) | supersedes | flag_dedup function-local memoization concern | **W5** (ledger PK) | the ledger's persistent PK dedup replaces the function-local memo 2092 investigates; noted, no new work |
| W6 `fm-memory-identity` | none | — | — | orthogonal — the ledger is SQLite/Mem0, not Graphiti entities; W5 does not touch `graphiti_client` |

---

## 8. Contract section (B + H)

The seams W5 owns, with signatures + invariants an implementer can build against without further
design discussion.

### 8.1 `ReconLedgerStore` (`reconciliation/recon_ledger.py`)

Schema (one table, template `ticket_store.py`):

```
recon_ledger(
  project_id   TEXT NOT NULL,
  record_kind  TEXT NOT NULL,   -- 'stage1_flag_marker' | 'stage1_flag_suppression'
                                --  | 'stage2_persistence_marker' | 'flag_for_stage2'
                                --  | 'cycle_summary'
  task_id      TEXT NOT NULL DEFAULT '',   -- '' when N/A (suppression/summary), never NULL (PK)
  flag_type    TEXT NOT NULL DEFAULT '',
  run_id       TEXT NOT NULL DEFAULT '',
  payload_json TEXT NOT NULL,   -- the narrative/metadata blob (mirror source)
  state        TEXT NOT NULL,   -- 'active' | 'suppressed' | 'addressed'
  created_at   TEXT NOT NULL,   -- ISO8601
  expires_at   TEXT,            -- ISO8601 or NULL (never-expire)
  PRIMARY KEY (project_id, record_kind, task_id, flag_type, run_id)
)
```
Indexes: `(project_id, record_kind, state)` for suppression/existence lookups;
`(project_id, expires_at)` for the GC pass.

API (async, mirrors the store conventions — `initialize()`, `_txn()`, `close()`, `checkpoint()`):

- `upsert(record) -> None` — `INSERT … ON CONFLICT(<pk>) DO UPDATE SET payload_json=excluded.…,
  state=excluded.state, created_at=excluded.created_at, expires_at=excluded.expires_at`.
  **Invariant: a repeated identity tuple leaves exactly one row.**
- `get_by_identity(project_id, record_kind, task_id, flag_type, run_id) -> record | None` —
  read-after-write consistent.
- `list_suppressions(project_id) -> list[record]` — indexed query, replaces the project-wide Mem0
  semantic search in `filter_suppressed`.
- `is_suppressed(project_id, task_id, flag_type) -> bool`.
- `gc(project_id, now, terminal_task_ids) -> int` — single
  `DELETE WHERE project_id=? AND (expires_at < ? OR (record_kind IN marker-kinds AND task_id IN
  terminal_task_ids))`; returns rows deleted. **Replaces all four Mem0 sweeps.**
- `mark_addressed(identity, addressed_by, run_id) -> None` — the `acknowledge_flag_marker` path.

Wiring: constructed + `initialize()`d in `server/main.py` alongside the other stores
(`main.py:479/615/720` region); registered into the periodic `checkpoint()` loop.

### 8.2 Dedup-exempt system write (`MemoryService`, `server/tools.py`)

- `MemoryService.add_system_record(content, *, project_id, agent_id, category, metadata,
  causation_id) -> AddMemoryResponse` — writes via the fresh-uuid direct-insert path (never the
  update-vs-add branch). **Invariant: every call lands a new point (verifiable via
  `count_memories_by_metadata` before/after); a recon-stage caller's mirror always lands.**
- Server gate (`server/tools.py`, next to the existing recon-stage branches): `add_system_record`
  / `add_memory(dedup_exempt=True)` is permitted **only** when
  `agent_id.startswith('recon-stage-')`; any other caller gets
  `{error, error_type:'DedupExemptNotPermitted', agent_id}` (returned, not raised — the
  `DarkFactoryPathScopeViolation` surfacing). **Rejection invariant: a non-recon caller is rejected
  and observes the diagnostic (G6 branch 4).**

### 8.3 `ReconWritePolicy` (`middleware/recon_write_policy.py`, consulted in `TaskInterceptor`)

`check(op: 'update_task'|'set_task_status', *, task_id, project_root, agent_id, target_status,
live_status, snapshot_token) -> Verdict` where `Verdict.to_error_dict()` matches the
`PathGuardVerdict` shape (`{error, error_type, task_id, …}`). Consulted only when
`agent_id.startswith('recon-stage-')`, as an independent early-return gate:

- **Terminal-update reject:** `op == 'update_task'` and `live_status ∈ TERMINAL_STATUSES` →
  `error_type:'ReconTerminalWriteRejected'`.
- **Live-workflow reject:** `op == 'set_task_status'` and
  `live_workflow_detector.is_workflow_live_for_task(task_id, project_root)` →
  `error_type:'ReconLiveWorkflowWriteRejected'`.
- **Snapshot-token staleness:** the write payload carries a server-issued snapshot token; if the
  task's live status ≠ the snapshot's status → `error_type:'ReconStaleSnapshotRejected'` (replaces
  the post-hoc stall-guard freshness gate at `task_knowledge_sync.py:421-507`).

**Invariants:** (a) enforcement is at the single durable write chokepoint under
`self._write_lock(project_id)` — read→check→write is atomic; (b) rejections are structured dicts
the LLM reads mid-run; (c) composes with W2's `(from,to,actor)` table as a *separate* gate — never
a third transition table (program decision #1).

Insertion points (from substrate): `_apply_status_transition` right after `old_status` is known
(`task_interceptor.py:645`); `update_task` alongside the early gates (`:3144-3157`, before the
write lock at `:3162`).

### 8.4 `recon_self_model.py` (`reconciliation/recon_self_model.py`)

- Constants: `MARKER_KINDS`, `MARKER_LIFECYCLE` (who writes / who deletes each kind),
  `FINGERPRINT_IDENTITY_FIELDS`, exact MCP call signatures for the recon tool surface.
- `render_marker_lifecycle_section() -> str`, `render_suppression_schema_section() -> str`,
  `render_cycle_summary_section() -> str`, `render_execution_class_section() -> str` — the exact
  text the prompts currently hand-transcribe (`prompts/stage1.py:489-589`,
  `prompts/stage2.py:31-66,103-129,196-344`). **Invariant: the prompt module imports these; a test
  asserts the prompt string contains the rendered text — drift is a failing test.**
- Predicates (for premise-lint): `run_id_is_fresh_per_run() -> bool`,
  `markers_deleted_only_by_gc() -> bool`, `premise_lint(task_description) -> list[Violation]`.

### 8.5 `execution_class` guard + routing (`middleware/execution_class_guard.py`, `server/tools.py`)

- `execution_class_error(metadata, agent_id, project_root) -> dict | None` — mirrors
  `deterministic_task_error`. When `agent_id.startswith('recon-stage-')` and
  `metadata.execution_class ∉ {code_tdd, operational, decision}` (absent or unknown) → returns
  `{error, error_type:'ValidationError', hint}`. **Rejection invariant (G6 branch 4): a recon
  submit_task without a valid execution_class is rejected and observes the diagnostic.**
- Routing coercion at the recon `submit_task` boundary: `execution_class ∈ {operational, decision}`
  → set `task_kind='deterministic'`, `metadata.always_escalates=true` (pure-gate); `code_tdd` →
  unchanged. `inject_*` persistence like `inject_task_kind` (`deterministic_task_guard.py:220`).

### 8.6 Computed stats (`reconciliation/stage_stats.py` or beside `WriteJournal`)

- `derive_stage_stats(ops, stage_agent_id) -> dict` — every write-shaped counter from journal ops
  (extends the `_OP_TO_STAT` logic `stats_verifier.py:31`). `report.stats` populated from it;
  LLM numbers preserved only under `stats['_reported']`. A stats-key allowlist rejects unknown
  LLM-reported counters. `stats_verifier`'s `_STAT_ALIASES` and both `_check_*_flag_counter_*`
  checkers are deleted.

---

## 9. Boundary-test sketch (B + H) — the integration-gate signal

Two-way scenarios facing both the producer (recon writer) and the consumer (GC / interceptor
enforcement). This suite is task ο's observable signal.

| # | Scenario | Preconditions | Postconditions (asserted) |
|---|---|---|---|
| L1 | **Writer vs GC race on the ledger** | Two concurrent stage runs UPSERT the same marker identity; a GC pass runs interleaved | Exactly one row for the identity; GC never deletes a still-`active`, non-expired, non-terminal marker; no lost UPSERT (last-writer-wins on payload) |
| L2 | **UPSERT idempotency** | Same identity tuple written N times with changing payloads | Row count stays 1; payload = last write; `state` transitions honoured |
| L3 | **Suppression round-trip** | A suppression record is upserted, then `filter_suppressed` runs | The suppressed (task_id, flag_type) is filtered via indexed query, no Mem0 search issued |
| L4 | **GC terminal-referenced** | A marker references a task now `done`; GC runs | The marker is deleted in the single DELETE pass; a marker referencing a live task is kept |
| P1 | **Terminal update_task rejection round-trip** | `recon-stage-task_knowledge_sync` calls `update_task` on a `done` task | Interceptor returns `{error, error_type:'ReconTerminalWriteRejected', …}`; the task is unchanged; the write journal records the rejection (the LLM-visible error shape is asserted exactly) |
| P2 | **Live-workflow status rejection** | `is_workflow_live_for_task` true for the target | `set_task_status` returns `ReconLiveWorkflowWriteRejected`; status unchanged |
| P3 | **Stale-snapshot rejection** | Snapshot token status ≠ live status at write time | Write rejected with `ReconStaleSnapshotRejected`; replaces the post-hoc freshness violation |
| P4 | **Dedup-exempt permission** | Non-recon agent_id calls `add_system_record` | Rejected `DedupExemptNotPermitted`; a recon agent_id's call lands a fresh point every time |
| D1 | **Deterministic cycle summary** | A stage completes | Exactly one `cycle_summary` ledger row, `payload` from the `StageReport`, written by Python; grep confirms nonce/verify/repair/reconstruct code is gone |
| S1 | **Computed stats override** | LLM reports `memories_added=5`; journal shows 2 | `report.stats.memories_added==2`, `stats['_reported'].memories_added==5`; an unknown LLM counter key is dropped |
| E1 | **execution_class enforcement** | `recon-stage` submit_task without execution_class | Rejected `ValidationError`; `operational` yields a deterministic pure-gate task (not an architect dispatch) |

---

## 10. Decomposition plan (task DAG)

Greek labels; actual IDs assigned at decompose. Every leaf names its observable signal (G2).
"file-serialize" deps exist purely to avoid narrow-lock rebase collisions on a shared file.

### Phase 0 — foundations (new files, parallel-safe)

- **α — `ReconLedgerStore` + server wiring.** Modules: `reconciliation/recon_ledger.py` (new),
  `server/main.py`, `config/schema.py`. *Signal:* a Python integration test opens the store,
  UPSERTs a marker, UPSERTs the same identity again (asserts row count stays 1), queries a
  suppression by index, runs `gc()` deleting an expired + a terminal-referenced marker. *Prereqs:* —
  (intermediate → unlocks ι/κ/λ).
- **β — `recon_self_model.py`.** Modules: `reconciliation/recon_self_model.py` (new). *Signal:*
  `render_marker_lifecycle_section()` returns non-empty text; `run_id_is_fresh_per_run()` returns a
  bool; `premise_lint("run_id persists across cycles")` returns a Violation. *Prereqs:* — (imports
  2140's `recon_pool_map.py`) (intermediate → unlocks ξ/η).
- **γ — dedup-premise empirical confirmation.** Modules: a test module under
  `fused-memory/tests/`. *Signal:* a test issues N identical `recon-stage` system writes and
  asserts the observed outcome (all N land, or documents the exact drop condition), producing the
  documented finding the cleanup (λ) and add_system_record (δ) consume. *Prereqs:* — (intermediate
  → unlocks δ/λ; binds the §2.1 premise — G6).

### Phase 1 — server-side write paths (the seams)

- **δ — dedup-exempt system-write path.** Modules: `services/memory_service.py`,
  `server/tools.py`, `backends/mem0_client.py`. *Signal:* boundary test **P4** — recon agent_id
  `add_system_record` lands a fresh point every call; non-recon caller rejected
  `DedupExemptNotPermitted`. *Prereqs:* γ.
- **ε — caller agent_id on the task-write path.** Modules: `server/tools.py`,
  `middleware/task_interceptor.py`. *Signal:* a test asserts the caller agent_id (via
  `_resolve_identity`) reaches `TaskInterceptor._apply_status_transition` / `update_task`.
  *Prereqs:* — (cross-batch: W2 converges; intermediate → unlocks ζ).
- **ζ — `ReconWritePolicy`.** Modules: `middleware/recon_write_policy.py` (new),
  `middleware/task_interceptor.py`. *Signal:* boundary tests **P1/P2/P3** — terminal `update_task`,
  live-workflow status write, and stale-snapshot write each return the exact structured error dict;
  target task unchanged. *Prereqs:* ε; consumes `live_workflow_detector`; cross-batch dep on W2's
  interceptor task at decompose.
- **η — require + validate `execution_class` on recon-filed submit_task (declaration layer).** Modules:
  `middleware/execution_class_guard.py` (new), `server/tools.py`. *Signal:* boundary test **E1**
  (narrowed) — recon submit_task without a valid `execution_class` rejected `ValidationError`; a valid
  class is accepted + persisted to metadata. Operational→off-TDD **routing is 2085's ratified job**,
  not asserted here. *Prereqs:* ε (agent_id), β (prompt text). *Coordinates with 2085 (not supersedes;
  do not cancel it).*
- **θ — computed `derive_stage_stats`.** Modules: `reconciliation/stage_stats.py` (new) or
  `services/write_journal.py`; `reconciliation/stats_verifier.py` (delete `_STAT_ALIASES`).
  *Signal:* boundary test **S1** — `report.stats` derived from journal ops; over-reported count
  overridden; `_reported` preserved; unknown key dropped. *Prereqs:* —.

### Phase 2 — consumer-side deletions (serialized on shared files)

- **ι — `flag_dedup.py` → ledger.** Modules: `flag_dedup.py`. *Signal:* boundary tests **L2/L3** —
  `dedup_flags` writes a marker via ledger UPSERT (duplicate signature → one ledger row, no Mem0
  confirm loop); suppression lookup is an indexed ledger query. The write/confirm/delete dance,
  circuit-breaker, and in-batch memo are gone (grep). *Prereqs:* α.
- **κ — TaskKnowledgeSync markers + GC collapse.** Modules: `task_knowledge_sync.py`. *Signal:*
  boundary tests **L1/L4** — one `gc()` call removes expired + terminal-referenced markers; the
  four `_sweep_*` functions are deleted (grep). *Prereqs:* α, ι; M4 shared-file dep;
  file-serialize before λ.
- **λ — deterministic cycle summaries + delete nonce/verify/repair/reconstruct.** Modules:
  `task_knowledge_sync.py`, `stages/memory_consolidator.py`, `cli_stage_runner.py`,
  `prompts/stage1.py`, `prompts/stage2.py`. *Signal:* boundary test **D1** — exactly one
  Python-written `cycle_summary` ledger row per stage run; `summary_nonce`/`retry_nonce` directives,
  `generate_summary_nonce`/`build_summary_nonce_section`, the three verify/repair/reconstruct
  functions + their `run()` wiring + four stats keys are deleted (grep). *Prereqs:* α, γ, δ;
  file-serialize after κ; M4 shared-file dep.
- **μ — post-flight guard shrink + flag-counter deletion.** Modules: `task_knowledge_sync.py`.
  *Signal:* `_apply_post_flight_guards` no longer re-fetches tasks / reclassifies ops
  (`_classify_terminal_state_violations`, `_check_stall_guard_freshness`,
  `_verify_set_task_status_post_action`, `_classify_live_workflow_status_writes`,
  `_check_flag_counter_completeness`, `_check_mem0_flag_counter_completeness` all deleted — grep);
  a stranded-terminal write is now impossible (blocked by ζ) rather than reclassified. *Prereqs:*
  ζ, θ; file-serialize after λ; M4 shared-file dep.
- **ξ — prompt self-model cutover + premise-lint.** Modules: `prompts/stage1.py`,
  `prompts/stage2.py`, recon submit_task path. *Signal:* a test asserts the stage prompt strings
  equal `recon_self_model.render_*()` output; a recon-authored task description asserting a false
  premise is flagged by `premise_lint` before filing. *Prereqs:* β, η; file-serialize after λ on
  the prompt files.

### Phase 3 — integration + deploy (B + H)

- **ο — integration-gate: two-way boundary-test suite.** Modules: `fused-memory/tests/`. *Signal:*
  the §9 boundary-test suite (L1–L4, P1–P4, D1, S1, E1) passes green under xdist — this is the
  B+H integration-gate leaf. *Prereqs:* α, δ, ζ, ι, κ, λ (the seams it exercises).
- **π — deterministic deploy capstone.** `task_kind='deterministic'`; `before_done` script does an
  out-of-cgroup `systemctl --user restart fused-memory.service` (decision #6) and verifies the new
  code is serving (e.g. a recon cycle runs against the ledger; `refresh_entity_summary`
  edge_count > 10 sanity). *Signal:* the running fused-memory process serves ledger-backed recon
  (post-restart PID + a ledger write observed). *Prereqs:* ALL (ο + every leaf).

---

## 11. Out of scope

- **ProjectScope signature threading** (M4, running now) — W5 declares deps where files overlap,
  does not thread the type itself.
- **Write-time entity identity** (`_resolve_or_create_entity`, uniqueness constraint) — **W6**.
- **Generic `CancelledError` re-raise convention** — **M5**.
- **Typing `execution_class` into `shared/task_metadata.py`** — **W3** (W5 ships it untyped).
- **The port-8103 escalation queue mechanics** — unchanged; `operational` asks use the deployed
  deterministic human-gate, not the 8103 queue (decision #3).
- **Any change to Mem0 / Qdrant internals** beyond using the existing fresh-uuid insert primitive.

---

## 12. Open questions (tactical — surfaced, not blocking)

1. **Ledger DB file: sibling vs shared `reconciliation.db`.** Put `recon_ledger` in the existing
   `reconciliation.db` (fewer files, shares the checkpoint loop) or a new sibling
   `recon_ledger.db`? **Suggested:** a new table inside `reconciliation.db` (the
   `ReconciliationJournal` already owns that file and its UPSERT precedents). Decide in task α.
2. **`add_system_record` vs `add_memory(dedup_exempt=True)` surface.** A dedicated method reads
   cleaner; a flag reuses the existing tool wiring. **Suggested:** dedicated
   `add_system_record` (clearer server-side gate, no new bool on a hot signature). Decide in task δ.
3. **Snapshot-token issuance.** The server must *issue* the snapshot token recon later presents.
   Simplest: the token is the `(task_id, status, monotonic_read_id)` recon received from its last
   `get_task`/`get_statuses`, echoed back in the write payload and re-checked. **Suggested:** derive
   the token from the existing `_STAGE2_STALL_SNAPSHOT_KEYS` metadata recon already stamps
   (`task_knowledge_sync.py:418`), promoted to a server-checked field. Decide in task ζ.
4. **`operational`/`decision` execution_class routing.** Owned by the ratified task 2085 (MECH B),
   NOT by W5 η. η only requires + persists the declared class; how 2085 routes `operational` vs
   `decision` asks is 2085's decision. Coordinate at implementation time so η's declared class is a
   clean input to 2085's `route_deterministic`.
5. **Task 2085 disposition — CORRECTED (do not cancel).** An earlier draft recommended cancelling
   2085; that is retracted. 2085 was ratified (MECH B) and is pending/active as of 2026-07-06. No
   human cancel is needed. The only coordination is to ensure η's `execution_class` field and 2085's
   `operational_ask_registry.yaml`/`route_deterministic` compose (η = declaration, 2085 = routing);
   land them without duplicating the routing coercion.
6. **M4 dep granularity.** If M4 lands its `task_knowledge_sync.py` changes as several tasks, W5's
   κ/λ/μ depend on the last one (the file's final M4 state). Resolve by `search_tasks` at decompose;
   if M4 is unfiled, note the pending dep and proceed (W5 rebases onto M4 whenever it lands).

---

## 13. Meta

*If decomposed and queued without further oversight, is the architecture complete, coherent,
cohesive, and good?* Yes: every mechanism has a named in-system consumer (§1, §8); all assumed
substrate is verified or queued upstream (§4); the one shaky premise is bound to an explicit
confirmation task rather than a RED test (§2.1, decision #1); cross-PRD seams have named owners
(§7); the high-stakes seams (ledger UPSERT-vs-GC, policy rejection round-trip) are specified as
contracts + two-way boundary tests (§8, §9) and land as a first-class integration-gate task (ο)
rather than starving under the narrow-lock orchestrator; the deploy is a deterministic capstone
using the deployed restart convention (§5 #6).
