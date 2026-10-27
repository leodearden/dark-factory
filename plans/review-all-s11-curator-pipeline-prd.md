# S11 — Curator pipeline and candidate types (PRD)

**Status:** authored 2026-10-10, ready to decompose. Stream S11 of the `/review-all` remediation
program `plans/review-all-program-dark_factory-2026-10-10.md` (run `review-all-dark_factory-20261008`,
pinned `6551b75aba`), approved as proposed under R21. Mode agent, wave 1.

**Code anchors** verified against main `ae3047ec3a` (2026-10-10). Cited by symbol throughout; main
moves fast, so re-locate at implementation time (`git diff ae3047ec3a..main -- <path>`).

## Goal

The next agent who changes how a submitted task is curated, or how a caller learns the outcome,
edits one place and is told the truth:

- `submit_task` callers (agents, Stage 2, the escalation watchers, `/review` triage) get
  **`pending` (not `failed`) when `resolve_ticket` stops waiting before the curator has
  decided**, with the ticket id to ask again. `failed` means the ticket really is terminal. They
  never re-submit a still-pending ticket, so a timeout no longer leads to a duplicate filing.
- A ticket curated alone and a ticket curated in a batch of five go through the **same guard
  sequence, prompt renderer, LLM invoker and failure mapping**. The batch size is a parameter,
  not a second copy of the pipeline.
- The curator's vocabulary (`CandidateTask`, `CuratorDecision`, …) lives in a small leaf module
  that the premise guards, escalator, ZOT sweep and interceptor import downward. They no longer
  form a 6-module import cycle with the 4,258-line LLM driver.
- The curator's corpus read is bounded by the open backlog rather than project history, and a
  corpus stream that fails says so in the prompt and the log instead of looking like "no overlap".
- The production-dead single-ticket worker path, and the stale docstrings that call it live, are
  gone.

## Background

Seven finding keys, re-verified at `ae3047ec3a` for this PRD:

- `fk-4f895cd425c2` (medium, confirmed). `TaskInterceptor._resolve_ticket_raw` returns synthetic
  `{'status': 'failed', 'reason': 'timeout'}` when its wait expires, and `failed/server_closed`
  when a woken waiter finds the row still `pending`. In both cases the row is not terminal. The
  MCP docstring (`server/tools.py::resolve_ticket`) lists `timeout` among `failed` reasons. The
  confusion codebook entry `entry-cand-20260714-2` has 23 sightings. **No Phase-5 mechanical task
  exists for this key.** The program brief says one was filed, but the report's
  `## Mechanical filings` table, the run record's `filings` and the contract §8 step-1 forensic
  query (`json_each(metadata,'$.x_finding_key')`) all come back empty. Step 2 (`search_tasks`
  ≥ 0.5) finds only done task 1666 at 0.546. This PRD therefore files the key as leaf γ.
- `fk-1e0486bf9ab9` (medium). `TaskInterceptor._process_add_ticket` (≈209 lines),
  `_process_add_tickets_batch`, the module-level `_merged_audit_metadata` and
  `TaskCurator._intra_batch_key` have no caller in `fused-memory/src`, `orchestrator/src` or
  `shared/src`. Only tests keep them alive, and the docstrings of `_dispatch_ticket_decision` and
  `_curator_lock` still describe the dead path as live. Task 6176 (pending) was filed to reconcile
  the lock span between "the batch and single-ticket worker paths". Its premise disappears with
  the deletion.
- `fk-f61accce2f37` (medium). Batch size is implemented as parallel copies:
  - `curate` / `curate_batch_prepared`: 26 of ~195 lines identical, and the guard order differs.
    Only `curate` runs `_pre_llm_exact_match`.
  - `_build_user_prompt` / `_build_batch_section`: 36 of 43 lines identical.
  - `_call_llm` / `_call_llm_batch`: 45 of 66 lines identical, with the same 11-keyword
    `CuratorFailureError` built twice.
  - The interceptor's `_process_add_tickets_batch_prepared` adds its own fallback arms
    (unprepared records, `CuratorFailureError`, bare `except`) that call `curate()` per ticket.
- `fk-caf8bcd5b7ed` (high, prior). Every failure branch in the curator pipeline still degrades to
  `create`. Leo ruled on 2026-08-19 that a curation outage must durably queue the submission, not
  create it. **Task 4513** owns that ruling. Its branch `task/4513` (tip `bfd0c0ad39`) is
  complete:
  - an `awaiting_curation` ticket status (non-terminal, durable);
  - an internal `defer` curator action;
  - janitor requeue and backlog escalation;
  - `curator_action='uncurated'` stamped on the residual creates.

  It is pending only on the non-blocking `esc-4513-9`. This PRD builds on 4513 and does not
  redo it. What remains for S11 is the parallel-pipeline structure that let each failure branch
  degrade differently, plus the dead `_intra_batch_key` dedup layer.
- `fk-756aef0b35b7` (low). The curator vocabulary is defined in `task_curator.py`. The four premise
  guards import `CandidateTask` under `TYPE_CHECKING`. `task_curator` imports three of them
  function-locally (in `_maybe_blocklist_drop`, `_maybe_premise_refuted_drop` and
  `_maybe_route_deterministic`). `curator_escalator` imports `CuratorFailureError` at module
  level, and `curator_zot_duplicate_sweep` imports `embedding_text`. Counting `TYPE_CHECKING`
  edges, these modules form a 6-module SCC. `candidate_key.py` also carries a byte-for-byte
  "mirror" of `normalize_title`.
- `fk-b259ba3af605` (medium, prior). `TaskCurator._build_corpus` calls
  `self._taskmaster.get_tasks(project_root)` with no status filter for every candidate with a
  module set. `TaskInterceptor._check_escalation_idempotency` (R4) does the same full read per
  ticket that carries R4 keys. Measured 2026-10-10 on the live store with a stdlib `json.loads`
  proxy for the per-row parse that runs on the event loop:
  - all 6,665 rows: ~140 ms;
  - the 1,693 rows in non-terminal statuses: ~24 ms.

  Per worker batch (≤ `batch_max` 5 candidates) that is ~0.7 s against ~0.12 s. 69% of the rows
  (4,617) are `done`. A failed stream (anchor fetch, `get_tasks`, embedding) is logged at DEBUG
  and leaves the stream empty, which reads as "no overlap" (INV-11).
- `fk-85234d9d39c4` (medium, prior, **weakened**). Re-verified under G6. The finding said
  project routing was decided by regex over prose, and that premise is no longer true. Task 2206
  (commit `0204e25fa5`) made `metadata.files` the only reject signal (`check_files_for_scope`).
  Prose is a non-blocking advisory (`check_text_for_scope`). `_soft_scope_branch` never blocks in
  either mode. Task 2208 retired the `dark_factory_path_guard.py` shim
  (`git ls-files | grep dark_factory_path_guard` → 0). The finding's remaining cost is a
  non-blocking advisory layer of measured low precision with a census-driven enforce decision
  pending. That is not a routing mis-decision. Disposition: acceptance record (see
  §Acceptance records).

## Finding keys

| Key | Severity | Closed by | Notes |
|---|---|---|---|
| `fk-756aef0b35b7` | low | α | curator_types leaf; SCC broken; function-local guard imports hoisted |
| `fk-1e0486bf9ab9` | medium | β | dead single-ticket path deleted; tests retargeted; 6176 absorbed |
| `fk-4f895cd425c2` | medium | γ | non-terminal `pending`; vocabulary typed; every quoting site updated in the same leaf |
| `fk-f61accce2f37` | medium | δ, ε | δ: one renderer/invoker/failure mapper; ε: one pipeline for N ≥ 1 |
| `fk-caf8bcd5b7ed` | high | ε (depends on 4513) | 4513 is the root fix (Leo 2026-08-19); ε removes the parallel structure and the interceptor's own fail-open arms |
| `fk-b259ba3af605` | medium | ζ (depends on 6385) | statuses-filtered corpus; R4 via `find_tasks_by_metadata`; stream failures surfaced |
| `fk-85234d9d39c4` | medium (weakened) | acceptance record ω | remedy landed (2206/2208); residual advisory layer accepted |

## Sketch of approach

1. **α**: add `middleware/curator_types.py`, stdlib plus `shared.task_statuses` only. It holds
   the curator's data vocabulary and the pure helpers over it. `normalize_title` gets its one
   home in the stdlib leaf `candidate_key.py`. All consumers import downward; there are no
   re-export shims.
2. **β**: delete the dead single-ticket worker path and `_intra_batch_key`. Retarget their
   behavioural tests to `submit_task`/`resolve_ticket` over a real `TicketStore`. Correct every
   docstring and comment that names them.
3. **γ**: add a typed ticket-status vocabulary (`TicketStatus`, with OPEN and TERMINAL subsets)
   in `ticket_store.py`. `resolve_ticket` reports the row's real status. The MCP docstring, the
   Stage 2 prompt, the recon self-model and the caller skills are updated in the same leaf, and
   the dead orchestrator restatement of the vocabulary in `roles.py` is deleted.
4. **δ**: one candidate-section renderer, one `_invoke_curator`, and one mapping from a failed
   `AgentResult` to `CuratorFailureError`. The CLI invoker is injected through the `TaskCurator`
   constructor so tests fake it through the public seam.
5. **ε**: one pipeline, `TaskCurator.curate_prepared(prepared)`, for N ≥ 1. It runs one ordered
   guard chain per candidate, then the LLM dispatch: the single spec for one LLM-bound candidate,
   the batch spec with bisection for more. Bisection bottoms out at a size-1 *invocation*, not a
   re-entry into the guard chain. `curate(candidate)` is `curate_prepared([prepare(candidate)])[0]`.
   The interceptor's own fallback arms are deleted.
6. **ζ**: the module stream reads `get_tasks(statuses=ACTIVE)`. The dependency stream uses the
   anchor row plus `get_dependency_edges` and bounded per-id reads. R4 uses 6385's
   `find_tasks_by_metadata`. Per-stream failures land in `PoolWithheld` and the decision log;
   when every applicable stream fails, the candidate counts as corpus-failed, which 4513 defers.

## Resolved design decisions

1. **4513 is the design basis, not a parallel effort.** Every S11 leaf depends on 4513, directly
   or transitively. 4513 rewrites `curate`, `curate_batch_prepared`, the worker, `TicketStore` and
   the `resolve_ticket` docstring, so building S11 on the pre-4513 tree would mean rebasing a
   pipeline rewrite over a second pipeline rewrite. S11 adds nothing to the outage policy. An
   outage defers (4513), a ZOT hang creates and is swept (task 5491), and the residual creates
   carry `curator_action='uncurated'` (4513). ε only guarantees that every one of those paths is
   decided in one place.
2. **`normalize_title`'s one home is `candidate_key.py`, not `curator_types.py`.** The program
   listed it under `curator_types`. `candidate_key.py` already has to hold a stdlib-only copy,
   because `backends/sqlite_task_backend.py` imports it at every INSERT chokepoint. Making that
   copy the public definition deletes the mirror (heuristic 11, SPOT) and adds no new
   backends→middleware edge for S19's DAG to untangle. `curator_types` does not need
   `normalize_title` itself.
3. **`curator_types.py` holds data and pure helpers, nothing that does I/O:**
   - `Action`, `CandidateTask`, `RewrittenTask`, `CuratorDecision`, `PreparedCandidate`;
   - `PoolEntry` (today's `_PoolEntry`, made public because it crosses the module boundary
     inside `PreparedCandidate`) and `PoolWithheld`;
   - `CuratorFailureError`, `DEFAULT_PRIORITY`;
   - `embedding_text`, `exception_summary`, `clip_for_prompt`, `is_combine_eligible_status`.

   `flatten_task_tree`, `_to_pool_entry` and every method stay in the driver. Importing
   `curator_types` must load neither `qdrant_client`, `graphiti_core` nor
   `fused_memory.middleware.task_curator`.
4. **No re-export shims.** Each of the ~16 test modules and ~9 source modules that imports a
   moved name is repointed in α (heuristic 13). The churn is mechanical one-line import edits,
   which is why α may legitimately declare more than 15 files.
5. **The vocabulary type lives with its owner.** `TicketStatus(StrEnum)` (`pending`,
   `awaiting_curation`, `created`, `combined`, `refused`, `failed`, `cancelled`) and its
   `OPEN_TICKET_STATUSES` / `TERMINAL_TICKET_STATUSES` subsets go in `ticket_store.py`, the store
   that writes those rows. The ticket paths in `ticket_store.py` and `task_interceptor.py` use
   the members instead of string literals (heuristic 12). The dashboard's SQL IN-list
   (`dashboard/src/dashboard/data/metrics.py`) is untouched: R19 replaces it with S15's read
   model, which imports this type.
6. **`resolve_ticket` reports the row; it never invents a terminal state.** When a status is
   open there is always a ticket to ask about again, so every open-status response carries
   `ticket` and a `reason`:
   - `wait_timeout`: the wait expired; the row was re-read and is still open.
   - `wait_interrupted`: the waiter was woken but the row is still open, for example during
     shutdown.
   - `store_unavailable`: the store was closing during the read. The status is reported as
     `pending` because `submit_task` persisted the row; after a restart the row reads `failed`
     (`server_restart`, per 4513's restart rule).

   Synthetic `failed` remains only where no ticket can ever be waited on: `unknown_ticket`, and
   `ticket_store not configured`.
7. **One caller rule, stated once.** An open status means "not decided yet: call `resolve_ticket`
   again with the same ticket; never re-submit." `skills/_shared/ticket-failure-handling.md` is
   the canonical caller document. Its `timeout` section becomes the open-status section, within
   its existing 2-submit/2-resolve cap. The other call sites (the MCP docstring, the Stage 2
   prompt, `recon_self_model`, and the escalation-watcher, recon-escalation-watcher and `/review`
   phase-3 snippets) carry the one branch and point at that document. A fused-memory test mirrors
   the `TicketStatus` members against each quoted list, so a new member cannot ship
   undocumented. This is INV-10's "mirror against the live artifact" form, not a prose
   assertion.
8. **Delete `orchestrator/agents/roles.py::submit_resolve_instructions`; do not update it.**
   Since `bfd19a296c` the steward and deep_reviewer use `submit_only_instructions`, and
   `review_checkpoint` files in a single step. `git grep` finds no production renderer of the
   two-step helper, only `orchestrator/tests/test_task_creation_migration.py`. The orchestrator's
   only restatement of the vocabulary is dead prose that tells agents a 60 s wait means `failed`.
   Deleting it, together with its two test classes, is the strongest form of "updated in the same
   leaf".
9. **The cancellation shield is not ported to the batch path.** `_process_add_ticket`'s
   `asyncio.shield` around `_persist_worker_terminal` is not ported. A worker cancelled between
   `tm.add_task` and terminal persistence leaves the ticket `pending`; the restart sweep marks it
   `failed/server_restart`. A caller's single re-submit then hits the partial UNIQUE index on
   `candidate_key`, which `_dispatch_ticket_decision` resolves as `combined`
   (`DuplicateCandidateKeyError`). So the window cannot create an exact duplicate. 6176's
   lock-span question is decided "document as intentional": with one worker path left, the batch
   path's span (persist, journal and signal inside `_curator_lock`) is the contract, stated in
   `_curator_lock`'s docstring. The reason it is the contract: a `remove_task` never observes a
   created task whose ticket still reads `pending`.
10. **Batch size is a parameter of one pipeline.** The per-candidate guard chain, in order:
    1. byte-identical `payload_hash` dedup (only when N ≥ 2);
    2. blocklist refuse;
    3. premise refuse;
    4. refusal inheritance by identical duplicates;
    5. claim-verification advisory;
    6. pre-LLM exact match on `candidate_key` (now for every N);
    7. idempotency cache;
    8. operational-ask route;
    9. corpus-failed defer (4513);
    10. ZOT-breaker defer (4513).

    Then the LLM dispatch: one LLM-bound candidate uses `CURATOR_SINGLE_SPEC`; more use
    `CURATOR_BATCH_SPEC` with bisection down to size-1 single-spec invocations. Both prompt
    specs survive, because their output contracts legitimately differ (`batch_target_index`).
    `curate_batch` (no caller in `src`) is deleted. `curate_batch_prepared` becomes
    `curate_prepared` (heuristic 1: it is not batch-specific).
11. **The invoker is a constructor seam.** `TaskCurator(..., invoke=invoke_with_cap_retry)`. Tests
    pass a fake through the public constructor instead of patching `_call_llm` or `_call_llm_batch`
    (the Tests stance: 148 references today). The static gate `shared/tests/test_wildcard_mcp_scoping_gate.py::_KNOWN_SITES`
    names one curator site, `TaskCurator._invoke_curator`.
12. **The corpus reads the open backlog.** The module stream reads
    `get_tasks(project_root, statuses=[s.value for s in shared.task_statuses.ACTIVE])`. Done
    tasks still reach the pool through the embedding stream, which is ordered by similarity and
    is the right signal for "already done". The bound becomes open backlog times per-row parse,
    with history excluded. The dependency stream takes:
    - the anchor's deps from the anchor row it already fetched;
    - its dependents from `get_dependency_edges`;
    - rows via `get_statuses_raw(ids=…)` and `get_task`, at most `pool_dependency_cap` reads.

    R4 queries `find_tasks_by_metadata(key='suggestion_hash', value=…, statuses=<all but
    cancelled>)` (task 6385, which raises on a store error and never returns an empty result on
    failure) and confirms `escalation_id` on the few matches. A store error keeps today's
    `check_failed=True`.
13. **A failed corpus stream is a fact, not an empty stream.** `PoolWithheld` gains
    `unavailable_streams`, rendered in the prompt as "stream X unavailable: absence is not proof
    of no overlap". Stream failures log at WARNING and appear in the decision's `to_log_fields`.
    When every applicable stream fails, the candidate's `corpus_error` is set, and that path
    defers (4513).

## Contract (B+H)

### `fused_memory.middleware.curator_types` (α owns; S11)

```
Action = Literal['drop','combine','create','route_deterministic','refuse','defer']   # 'defer' from 4513
@dataclass class CandidateTask: title, description, details, files_to_modify, priority,
                                spawned_from, spawn_context, execution_class; payload_hash(); from_rewritten_task()
@dataclass class RewrittenTask; @dataclass class CuratorDecision (fields as on main + 4513)
@dataclass class PoolEntry (was _PoolEntry; render(desc_cap, details_cap))
@dataclass(frozen=True) class PoolWithheld (+ unavailable_streams: tuple[str, ...] = () in ζ)
@dataclass class PreparedCandidate: candidate, pool: list[PoolEntry], pool_sizes, prompt_tokens, withheld, corpus_error
class CuratorFailureError(RuntimeError)
DEFAULT_PRIORITY: str
def embedding_text(title, description, files_to_modify) -> str
def exception_summary(exc) -> str
def clip_for_prompt(text, cap) -> str
def is_combine_eligible_status(status) -> bool
```

Invariants:
- Imports only the stdlib and the stdlib-only first-party layer declared at
  `fused-memory/tests/test_curator_types_leaf.py::ALLOWED_FIRST_PARTY`.
- No module in `fused_memory.middleware` imports a name listed above from `task_curator`.
- `candidate_key.normalize_title` is the only title normaliser.

### `resolve_ticket` vocabulary (γ owns; S11). Consumers quote it and never redefine it

```
class TicketStatus(StrEnum): PENDING, AWAITING_CURATION, CREATED, COMBINED, REFUSED, FAILED, CANCELLED
OPEN_TICKET_STATUSES     = {PENDING, AWAITING_CURATION}
TERMINAL_TICKET_STATUSES = frozenset(TicketStatus) - OPEN_TICKET_STATUSES

resolve_ticket(ticket, project_root, timeout_seconds=115) ->
  open:      {'status': 'pending'|'awaiting_curation', 'ticket': str, 'reason': str}
             reason ∈ {'wait_timeout','wait_interrupted','store_unavailable'} or the parked row's reason
  terminal:  {'status': 'created'|'combined', 'task_id': str, 'reason'?: str}
             {'status': 'refused'|'cancelled'|'failed', 'reason': str}            # no task_id key
  no ticket: {'status': 'failed', 'reason': 'unknown_ticket'|'ticket_store not configured'}
```

Invariants:
- A response with an open status never carries `task_id`.
- `failed` is returned only for a terminal row or the no-ticket case.
- An open row is bounded:
  - `pending` by the per-project worker, the `TicketJanitor` worker-dead reap and the startup
    `server_restart` sweep;
  - `awaiting_curation` by 4513's janitor requeue and backlog escalation.

### Curation pipeline (ε owns)

`TaskCurator.curate_prepared(prepared: Sequence[PreparedCandidate]) -> list[CuratorDecision]` is
the only path from a prepared candidate to a decision.

- It raises only `AllAccountsCappedException` (the worker defers the batch, as on main) and
  `CancelledError`.
- Every other failure maps to a decision through one function. A defer happens per 4513.
- A size-1 result is field-identical to the same candidate's result when it sits in a batch whose
  LLM call it does not share.

## Boundary-test sketch

| # | Side | Scenario | Preconditions | Postconditions |
|---|---|---|---|---|
| B1 | producer (γ) | wait expires before the curator decides | real `TicketStore`; curator stub blocked on an event | `resolve_ticket(timeout_seconds=0.05)` → `{'status':'pending','ticket':t,'reason':'wait_timeout'}`; row still `pending` |
| B2 | producer (γ) | same ticket after the stub is released | B1 | second `resolve_ticket` → `created` with `task_id`; exactly one task row |
| B3 | producer (γ) | parked ticket | 4513 parks the ticket `awaiting_curation` | `resolve_ticket` returns immediately with `awaiting_curation`, `ticket`, the parked reason, and no `task_id` |
| B4 | producer (γ) | woken while still open | a waiter is signalled while its row is still `pending` and readable | `{'status':'pending','ticket':t,'reason':'wait_interrupted'}`, never `failed/server_closed` |
| B5 | consumer (γ) | vocabulary parity | — | every `TicketStatus` member appears in the `resolve_ticket` MCP docstring, the Stage 2 prompt's status list, `recon_self_model`'s `resolve_ticket` entry and `skills/_shared/ticket-failure-handling.md` |
| B6 | consumer (ε) | exact duplicate inside a batch | task A created and `note_created`; a batch of 3 whose second candidate equals A on (title, files) | that ticket resolves `combined` to A with the pre-LLM exact-match justification; the fake invoker's batch call contains the other two candidates only |
| B7 | producer (ε) | size-1 vs batch parity | the same candidate with a guard hit (blocklist), alone and in a batch of 3 | identical `CuratorDecision` fields in both |
| B8 | producer (δ/ε) | bisection bottom | the fake invoker fails the 2-candidate batch call | the fake invoker then records exactly two single-spec invocations, one per candidate, and both tickets resolve from those verdicts |
| B9 | producer (ζ) | open-backlog read | fake backend recording `get_tasks` kwargs | module stream passes `statuses=ACTIVE`; no unfiltered `get_tasks` call from `_build_corpus` |
| B10 | producer (ζ) | stream failure | embedding stream raises | prompt section carries the unavailable-stream line; decision log fields name `embedding`; with module+embedding both failing, `corpus_error` is set and the ticket parks |
| B11 | consumer (ζ) | R4 hit without a tree read | a non-cancelled task carrying (escalation_id, suggestion_hash) | `submit_task` with the same pair resolves `combined` (`idempotency_hit`); the fake backend saw `find_tasks_by_metadata` and no `get_tasks` |

## Pre-conditions for activating

- **4513 lands first** (external dependency of α). Its branch is complete. The dispatch dry-run
  on 4513 says it needs only `esc-4513-9` acknowledged: Leo is to choose won't-fix or a one-shot
  task for the 35 uncurated 2026-08-18 submissions. That escalation is the schedule risk for this
  whole stream.
- **4503** (pending, high; empty branch) asks for the cap-deferral that 4513's branch already
  implements (`all-accounts-capped` → `defer`). If 4503 is still live when this PRD is
  decomposed, phase B wires ε → 4503 (same functions) and flags it to the lead to reconcile.
- **6385** (`find_tasks_by_metadata`, pending, high) is an external dependency of ζ only.
- G3 substrate verified on main:
  - `TaskBackendProtocol.get_tasks(..., statuses=)`, `get_dependency_edges` and
    `get_statuses_raw(ids=)`;
  - `shared.task_statuses.ACTIVE`;
  - `_dispatch_ticket_decision`'s `DuplicateCandidateKeyError` → `combined` arm;
  - real `TicketStore` fixtures in `fused-memory/tests/test_ticket_worker.py`;
  - `invoke_with_cap_retry`;
  - `TaskCurator._ledger_kwargs`.

## Cross-PRD relationship

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| task 4513 (curation outage durable queue) | consumes | `awaiting_curation`, internal `defer`, `curator_action='uncurated'`, `TicketStore.mark_awaiting_curation` | 4513 | branch complete, pending esc-4513-9 |
| task 6385 / `plans/task-metadata-lookup-prd.md` L1 | consumes | `TaskBackendProtocol.find_tasks_by_metadata` | that PRD | pending |
| S10 task-write contract | produces | `curator_types.CandidateTask` (S10's `declared_files(metadata)` reader feeds its construction in `TaskInterceptor._build_candidate`) | **S11** (α) | S10 tasks touching `CandidateTask` depend on α |
| S15 fused-memory server boundary | produces | `ticket_store.TicketStatus` for the ticket-panel read model; S15's split of `tools.py` carries γ's `resolve_ticket` docstring verbatim | **S11** (γ) | S15 depends on γ for the ticket read model |
| S22 dashboard readers | consumes (indirect) | `metrics.py` terminal IN-list is replaced by S15's read model (R19), which imports `TicketStatus` | S22/S15 | no S11 edit |
| S19 package layering | neutral | α removes the middleware SCC and adds no backends→middleware edge (decision 2) | S19 | — |
| S13 recon sweeps | neutral | `task_knowledge_sync._TASK_CREATED_SUCCESS_STATUSES` stays (still true); S11 does not edit `task_knowledge_sync.py` | S13 | — |
| task 6332 (extract the ticket worker from `task_interceptor.py`) | downstream | same file | 6332 | phase B wires 6332 → ζ (S11's last interceptor edit) |
| task 6176 (curator_lock span) | absorbed | decision 9 | **S11** (β) | phase B cancels 6176 with `x_acceptance_reason` naming β |
| tasks 6322, 6519, 6622, 6633, 6653 | adjacent | same files, different concerns | their owners | untouched; the orchestrator's file locks serialise dispatch |

## Decomposition plan

Priority = the highest finding severity a leaf closes (contract §4). Every leaf is
`task_kind='normal'` and carries `metadata.x_finding_key` (list),
`metadata.x_finding_run='review-all-dark_factory-20261008'`, `metadata.source='review-all'` and
file-level `metadata.files`. The leaves are serialised because they share
`task_curator.py`/`task_interceptor.py`:
- α → β;
- β → γ and β → δ;
- γ, δ → ε;
- ε → ζ;
- external: α → 4513, ζ → 6385 (and ε → 4503 if 4503 is live).

### α: Extract the curator vocabulary into `middleware/curator_types.py`
- Keys: `fk-756aef0b35b7`. Priority low. Est. ~500 LOC (mostly moved lines and import edits).
- Files: `fused-memory/src/fused_memory/middleware/curator_types.py` (new),
  `.../middleware/task_curator.py`, `.../middleware/candidate_key.py`,
  `.../middleware/task_interceptor.py`, `.../middleware/curator_escalator.py`,
  `.../middleware/curator_zot_duplicate_sweep.py`, `.../middleware/cancelled_premise_blocklist.py`,
  `.../middleware/operational_ask_registry.py`, `.../middleware/recon_code_fix_premise_guard.py`,
  `.../middleware/path_scope_guard.py`, `fused-memory/tests/test_curator_types_leaf.py` (new),
  `fused-memory/tests/test_task_curator.py`. The remaining test modules get mechanical import
  edits; the architect widens the list.
- Work:
  - decisions 2–4;
  - hoist `task_curator`'s function-local imports of the three registry guards to module level;
  - hoist any other function-local first-party import that no longer breaks a cycle;
  - the guards import `CandidateTask` at module level from `curator_types`.
- Signal: `fused-memory/tests/test_curator_types_leaf.py` passes in CI. It spawns `python -I`,
  imports `fused_memory.middleware.curator_types`, and asserts that none of
  `fused_memory.middleware.task_curator`, `qdrant_client` or `graphiti_core` is in `sys.modules`.
  It also asserts, over AST including `TYPE_CHECKING` blocks, that none of the six former SCC
  members except `task_curator` imports from `task_curator`.
- Seams: provides `curator_types` (consumed by β–ζ and S10). Consumes nothing new.
- G7: INV-9 improves (mirror deleted). No hits.

### β: Delete the dead single-ticket worker path and `_intra_batch_key`
- Keys: `fk-1e0486bf9ab9`. Priority medium. Est. ~600 LOC (about −300 production, ~300
  retargeted tests).
- Files: `.../middleware/task_interceptor.py`, `.../middleware/task_curator.py`,
  `.../middleware/candidate_key.py` (mirror-site comment), `orchestrator/src/orchestrator/agents/triage.py`
  (mirror-site docstring), `fused-memory/tests/test_task_interceptor.py`,
  `fused-memory/tests/test_ticket_worker.py`, `fused-memory/tests/test_task_curator.py`.
- Work:
  - delete `_process_add_ticket`, `_process_add_tickets_batch`, `_merged_audit_metadata` and
    `TaskCurator._intra_batch_key`;
  - correct every docstring and comment that names them: `_dispatch_ticket_decision`,
    `_curator_lock`, the `route_deterministic` comment, the `__init__` lock comments, the
    `normalize_title` docstring, and the `payload_hash` and `compute_candidate_key` mirror-site
    lists;
  - retarget the 3 + 11 + 4 + 17 test references to behaviour through `submit_task`/`resolve_ticket`
    over a real `TicketStore`, deleting tests that only pinned the dead code;
  - decision 9: state the batch path's lock span in `_curator_lock`'s docstring.
- Signal: `git grep -nE 'def (_process_add_ticket|_process_add_tickets_batch|_merged_audit_metadata|_intra_batch_key)\('`
  over `fused-memory/src` is empty. The retargeted idempotency, combine and refuse tests drive
  `submit_task`/`resolve_ticket` and pass in CI.
- Seams: none. Absorbs task 6176.
- G7: no hits.

### γ: `resolve_ticket` reports an open ticket as open (typed ticket-status vocabulary)
- Keys: `fk-4f895cd425c2`. Priority medium. Est. ~450 LOC.
- Files:
  - production: `.../middleware/ticket_store.py`, `.../middleware/task_interceptor.py`,
    `fused-memory/src/fused_memory/server/tools.py` (the `resolve_ticket` / `submit_task` /
    `list_tickets` docstrings), `orchestrator/src/orchestrator/agents/roles.py`,
    `fused-memory/src/fused_memory/reconciliation/prompts/stage2.py`,
    `fused-memory/src/fused_memory/reconciliation/recon_self_model.py`;
  - skills: `skills/_shared/ticket-failure-handling.md`, `skills/escalation-watcher/SKILL.md`,
    `skills/recon-escalation-watcher/SKILL.md`, `skills/review/references/phase3-triage.md`;
  - tests: `orchestrator/tests/test_task_creation_migration.py`,
    `fused-memory/tests/test_resolve_ticket_vocabulary.py` (new).
- Work: decisions 5–8 and the `resolve_ticket` vocabulary contract above. Update the
  `resolve_ticket` and `TaskInterceptor.resolve_ticket` docstrings: drop the retired `expired`,
  and drop `timeout` as a failure.
- Signal: over MCP, `resolve_ticket(ticket, project_root, timeout_seconds=0.05)` on a ticket the
  curator has not decided returns `{"status": "pending", "ticket": "tkt_…", "reason":
  "wait_timeout"}`, and a later call returns the terminal outcome (B1–B4). B5 holds in CI.
- Seams: provides `TicketStatus` and the `resolve_ticket` response contract (consumers: recon
  Stage 2, the skills, S15). Removes the orchestrator `roles.py` quote.
- G7:
  - INV-7: the open hold is owned and bounded (contract invariant 3).
  - INV-4: the caller's re-poll is capped by the canonical caller document (2 resolves).
  - INV-10: B5 mirrors the live enum.
  - No waiver.

### δ: One candidate-section renderer, one LLM invoker, one failure mapper
- Keys: `fk-f61accce2f37`. Priority medium. Est. ~600 LOC (≈ −120 production, the rest test
  retargeting onto the injected invoker).
- Files: `.../middleware/task_curator.py`, `fused-memory/tests/test_task_curator.py`,
  `fused-memory/tests/test_task_curator_timeout_evidence.py`,
  `fused-memory/tests/test_curator_arm_instrumentation_guard.py`,
  `shared/tests/test_wildcard_mcp_scoping_gate.py`.
- Work:
  - `_build_user_prompt` and `_build_batch_section` become one renderer parameterised by its
    header;
  - `_call_llm` and `_call_llm_batch` become `_invoke_curator(prompt, spec, schema, timeout,
    max_turns, budget, label, ledger_role)` plus one `_failure_error(agent_result, evidence,
    context)`;
  - decision 11 (the injected `invoke`);
  - `_KNOWN_SITES` names `TaskCurator._invoke_curator`, with the measurement comment updated
    (INV-12);
  - the arm-instrumentation guard's `LLM_CALL` is retargeted.
- Signal: intermediate. It unlocks ε, whose single LLM dispatch it provides. Observable on its
  own: the curator invocations ledger rows for single and batch calls come from one call site
  (`_KNOWN_SITES` holds exactly one curator entry).
- Seams: internal only.
- G7: INV-5 improves. No hits.

### ε: One curation pipeline for every batch size
- Keys: `fk-f61accce2f37`, `fk-caf8bcd5b7ed`. Priority **high**, so phase B adds it as a
  dependency of trigger-chain gate 6698. Est. ~900 LOC.
- Files: `.../middleware/task_curator.py`, `.../middleware/task_interceptor.py`,
  `fused-memory/tests/test_task_curator.py`, `fused-memory/tests/test_ticket_worker.py`,
  `fused-memory/tests/test_task_interceptor.py`,
  `fused-memory/tests/test_curator_arm_instrumentation_guard.py`,
  `fused-memory/tests/test_curation_deferral.py` (from 4513).
- Work:
  - decision 10;
  - `curate_prepared` is the pipeline; `curate` is a batch of one; `curate_batch` is deleted;
  - `_call_llm_batch_with_fallback`'s size-1 bottom calls the single-spec invocation, not
    `curate()`;
  - in the interceptor, `_process_add_tickets_batch_prepared` loses its `unprepared_records`
    backfill, its `CuratorFailureError` arm and its bare `except` → `decision=None` arm;
  - `_prepare_ticket` treats an unexpected `prepare_candidate` exception as a `corpus_error`,
    which 4513 defers.
- Signal: B6 and B7 through `submit_task`/`resolve_ticket` over a real `TicketStore`. An exact
  duplicate inside a batch of three resolves `combined` with the pre-LLM exact-match
  justification, and the fake invoker records no LLM item for it; today only a size-1 curation
  runs that check.
- Seams: consumes 4513 and δ. Provides the one pipeline (consumed by the worker).
- G7: INV-11 is satisfied, because every non-decision outcome is a 4513 defer or a marked
  `uncurated` create from one mapper. No hits.

### ζ: Bound the curator's corpus and R4 reads; surface failed corpus streams
- Keys: `fk-b259ba3af605`. Priority medium. Est. ~450 LOC.
- Files: `.../middleware/task_curator.py`, `.../middleware/curator_types.py`,
  `.../middleware/task_interceptor.py`, `fused-memory/tests/test_task_curator.py`,
  `fused-memory/tests/test_task_interceptor.py`, `fused-memory/tests/_curator_helpers.py`.
- Work: decisions 12–13. Update `PoolWithheld`'s docstring: the module and dependency counts now
  cover the open backlog.
- Signal: B9–B11 in CI, plus on the live service a curation's decision log line carries
  `unavailable_streams` when a stream failed. The basis for the bound:
  - the module stream's per-candidate on-loop parse was measured at ~24 ms over the 1,693 open
    rows, against ~140 ms over all 6,665 (2026-10-10, stdlib proxy);
  - per worker batch (`batch_max` 5) that is ~0.12 s against ~0.7 s, and history no longer
    grows it.
- Seams: consumes 6385's `find_tasks_by_metadata` and α's `PoolWithheld`.
- G7: INV-8 is answered with a stated bound (open backlog × measured per-row cost); INV-11 is
  satisfied (decision 13). No waiver.

## Acceptance records

### ω: Accept the path-scope prose advisory as standing (`fk-85234d9d39c4`)
Filed in phase B as a `cancelled` task carrying `metadata.x_finding_key=['fk-85234d9d39c4']`, the
run stamps, `prd_path` (no `prd_task_label`; it is not a plan leaf) and
`metadata.x_acceptance_reason`:

> The finding's remedy landed. Task 2206 (`0204e25fa5`) made declared `metadata.files` the only
> reject signal. Prose became a non-blocking advisory, and `_soft_scope_branch` never blocks in
> either mode. Task 2208 retired the shim. The residual regex-over-prose advisory, adjudicator
> and soft-scope census are a measured, non-blocking signal layer whose enforce flip is a separate
> census-driven decision. They are not a routing mis-decision. Re-verified at `ae3047ec3a` by
> S11 (`plans/review-all-s11-curator-pipeline-prd.md`).

Contract §7: the acceptance is re-verified like any finding. It reopens if the anchor changes or
prose becomes a reject signal again.

## Out of scope

- The outage policy itself (defer vs create, the ZOT exception, the `uncurated` marker, requeue
  and backoff): task 4513.
- Re-dedupe sweeps over marked creates: task 6322.
- The degraded-streak alarm: task 6519.
- Registry load health: task 6622.
- The `_task_files` / `_extract_meta_files_from_meta` unification: task 6653.
- Splitting `task_interceptor.py`: task 6332.
- Splitting `task_curator.py` beyond the vocabulary extraction. After α–ζ the driver shrinks;
  any further heuristic-14 partition is measured by a later run.
- Path-scope advisory precision: acceptance ω.
- The dashboard's ticket SQL: R19 (S15/S22).
- Rendering `execution_class` in the curator prompt (a possible next change the finding names,
  not a defect).

## Open questions

None. The design is closed. The one schedule risk, 4513 waiting on `esc-4513-9`, is an
operational action for the lead and Leo, not a design question.
