# Cancel-with-children: a cancel carries a disposition for every live child

**Status:** active PRD, authored 2026-11-01 (`/prd` under `/team`, Leo). Contract (B+H).
**Code anchors** cite symbols; verified against main `f7f9957bb5` (2026-11-01). Main moves fast —
re-locate at implementation time.
**Evidence:** [`plans/superseded-parent-children-2026-11-01.md`](superseded-parent-children-2026-11-01.md)
(126 orphan flags measured across four projects, 2026-10-07..11-01).

## 1. Goal

Cancelling a task that still has live children — tasks with `metadata.spawned_from` naming it,
or that depend on it, in this or another project — becomes a single act that also decides what
happens to each child. A canceller who omits that decision is refused with a typed error that
lists the children with the facts needed to decide, and points at `/cancel-with-children`. The
interceptor then carries out every child's disposition before the parent's cancel lands. The
reconciler's after-the-fact orphan sweep is kept only as a backstop for cancels that bypassed
the contract, filing one record per parent.

### What a user observes when this lands

- `set_task_status(5869, 'cancelled', superseded_by=[8032, …])` with no `children` is refused
  with `error='cancel_children_undispositioned'`, listing 6905/6906/7007/7996 (the replacement
  batches named in `superseded_by` are defaulted to `keep` and not asked about). The same call with
  `children={...}` — all four `keep`: 6905 and 7996 already depend on 8032, and 6906 and 7007
  stand on main — cancels 5869, stamps the four, and the reconciler files nothing. Today the same act filed 31 L1s, all false.
- `/cancel-with-children <id>` walks the canceller through the decision table and issues the
  one call.
- `resolve_issue(action='abandon')` on an L2 whose task has undispositioned children is refused
  before the record closes; with `children=` it lands, and the archived record carries the
  decisions.
- A cancel that bypasses the contract (operator script, race) produces ONE residue record on the
  parent listing its unhandled children, instead of one L1 per child.

### Consumers (G1)

| mechanism | consumer |
|---|---|
| `children` / `superseded_by` on `set_task_status` | interactive sessions and skills (escalation-watcher supersession carve-out, `/review` phase-3 accept, trigger-chain §5, review-all clerk CANCEL recipe), recon Stage 2, the steward, the harness abandon path |
| typed refusals | the same callers; `/cancel-with-children` reads their payload |
| `get_cancel_children` read tool | `/cancel-with-children` step 1, the abandon pre-flight probe, the harness cascade filter, recon Stage 2 |
| `superseded_by` (parent) / `cancel_dispositions` (child) metadata | the backstop sweep's skip rules, the skill's late mode, `/review` phase 3 |
| `DECISION_TABLE` data | the skill's rendered table, the Stage 2 prompt section |
| residue record | the escalation watchers (`/cancel-with-children --late`) |

## 2. Background

- `fused-memory/src/fused_memory/reconciliation/targeted.py::_sweep_cancelled_descendants`
  (added 2026-05-11, `64563a3bf3`) runs on every cancel and files one blocking `scope_violation`
  L1 per non-terminal descendant unless it can auto-cancel it. Task 1490 added a co-cancellation
  re-read after the reify 3883 storm; it does not catch cancels that land seconds later.
- Measured (report §2): 71% of the 126 flags were redundant — the act that superseded the parent
  had already handled the child. The correct handling splits fold+cancel 39%, child-is-superseder
  23%, repoint 18%, keep 11%, moot 5%, re-scope 3%, done-on-main 1%; no single rule fits, and the
  escalation-watcher carve-out's "repoint every dependent" is right for only 24% of dependents.
- The harm the sweep exists to prevent is real but its L1 never prevented it: a cancelled local
  dep counts as satisfied (`orchestrator/src/orchestrator/dependency_verdict.py::deps_verdict`),
  so a bare cancel releases a dependent that was waiting for the parent's delivery (952, 5770;
  memory of esc-3525-2, 2026-08-10), and an open escalation does not hold a pending task
  (`orchestrator/src/orchestrator/scheduler.py::Scheduler._eligible_for_dispatch`). Cross-project,
  the opposite: an external dep is satisfied only by `done`
  (`dependency_verdict.py::_external_deps_verdict`), so a cancelled parent wedges its external
  dependents forever.
- The knowledge needed to decide lives with the canceller, at cancel time. Hence option B:
  decide in the cancelling call; the sweep only catches what bypassed it.

## 3. Resolved design decisions (Leo, 2026-11-01)

| # | decision | rejected alternatives |
|---|---|---|
| D1 | Option B: a cancel of a task with live children must carry a complete structured disposition map, else a typed refusal pointing at `/cancel-with-children`; the interceptor executes the map. | A (supersession-aware deferred sweep) — its suppression logic survives as the backstop (§4.11); C (hold dependents); D (skip superseders only) |
| D2 | Execution: dispositions run BEFORE the parent's write through the interceptor's own public methods, each idempotent; a fresh coverage check inside the parent's write lock is the second enforcement point; the parent is written last. Not atomic; partial failure is a typed payload and a retry of the same call converges. | a new single-`_write_unit` backend method (re-implements events/journal/gates per child; first multi-task transaction); hybrid |
| D3 | One canonical task supersession key: `metadata.superseded_by: list[int]`, Tier-A blessed, Tier-B row listing `x_superseded_by`, `x_subsumed_by`, `subsumed_by`, `x_coalesced_into`, `x_replacement` as aliases to avoid. No corpus migration. | typed `supersession{by, kind}` submodel; one-shot rewrite of legacy pointers |
| D4 | No `defer` action. Every interceptor path is gated; machine paths are childless or fail loudly; bypasses fall to the backstop. | `defer` for a named caller; `defer` allowed only for `spawned_from`-only children |
| D5 | Recursion: `fold`/`moot` on a child that has live children of its own is refused; the skill works bottom-up. | nested maps with a depth cap |
| D6 | Abandon: pre-flight check through a harness probe before the record closes + dispositions persisted on the `Escalation` record and passed through + a late refusal files a new L2 loudly. | persist + re-file only; pre-flight only |
| D7 | `irreversible_deletion` (watcher always-ask) narrows to real content loss: a `fold`/`moot` of a child whose branch has commits, or whose declared files are absent from main and no superseder re-creates them. Everything else self-executes when the carve-out's four checks pass and the map is complete. | keep "cancel with dependents" always-ask; drop the line |
| D8 | Residue: ONE L1 per cancelled parent, `evidence` = the residual `ChildRow`s, closed after a late disposition. | keyed on the first residual child with members (an L2 cascade abandon would cancel them all) |
| D9 | Late disposition reuses the verb: `set_task_status(<cancelled parent>, 'cancelled', children=…)` runs the executor for a partial map. | a separate `disposition_children` tool |
| D10 | Cross-project dependents are in scope: enumerated through the known-project-roots registry, keyed `"<project_id>:<id>"`, `repoint` only; if the registry is unreachable the view says `external_dependents_checked: false`, never an empty list. | out of scope, follow-up |
| D11 | The decision table is DATA in `shared/src/shared/cancel_dispositions.py`; the skill's copy is a rendered block guarded by a pin test, Stage 2's section is a render (INV-5). No grace window in the backstop: with dispositions preceding the cancel the 71% race is gone for compliant cancellers. | skill markdown as source + Stage 2 pointer only; grace window |

## 4. Contract (B+H)

### 4.1 Surface

One verb gains two parameters; one read tool is added; both reach the same pure module.

```
set_task_status(id, status='cancelled', project_root, ...,
                children: dict[str, ChildDisposition] | None = None,
                superseded_by: list[int] | None = None,
                co_cancelled: list[int] | None = None)
get_cancel_children(id, project_root, superseded_by: list[int] | None = None) -> CancelChildrenView
resolve_issue(escalation_id, resolution, action='abandon', ...,
              children: dict[str, ChildDisposition] | None = None,
              superseded_by: list[int] | None = None)
```

`co_cancelled` names local children that the SAME act will cancel separately (an L2 cascade's
member tasks, §4.12): they are excluded from coverage, and the executor stamps each with a
`cancel_dispositions` entry `action: 'co_cancelled'` so the backstop's "any entry for this parent
→ skip" rule covers them while their own cancels are still in flight. CSV and `commit_planning`
derive the same set from the batch (§4.10). `co_cancelled` is an ENTRY action, never a
`ChildDisposition` a caller may pass.

Shapes, validation and the decision table live in ONE module,
`shared/src/shared/cancel_dispositions.py` (pydantic, `frozen=True, extra='forbid'`, the
`task_deferral.py` idiom), imported by fused-memory (gate + executor + tool), escalation
(shape validation in `resolve_issue`), orchestrator (typed rejection, harness pass-through) and
the skill's rendered table (§2). Heuristic 11 SPOT; INV-1 (the contract is a schema, not prose).

### 4.2 `ChildDisposition` (discriminated on `action`)

| action | fields | meaning | validation |
|---|---|---|---|
| `keep` | `reason?` | child stands on main / is independent; a cancelled dep counts as satisfied, so it is released | none beyond shape |
| `repoint` | `onto: list[int]` (non-empty), `reason?` | child needed the parent's DELIVERY; move the edge | each `onto` id exists in the parent's project, is not the parent, not the child, not `cancelled`; no `onto` cycle through the child (backend `add_dependency` has no cycle check — the gate adds one, depth-bounded by `get_dependency_edges()`) |
| `fold` | `into: int`, `reason?` | child's scope is covered by the superseder; the canceller has already transferred uncovered items into `into` | `into` exists, not cancelled, not the child; child has NO live children of its own (§4.7) |
| `moot` | `reason?` | premise existed only on the parent's branch/plan; nothing replaces it | child has NO live children of its own |

`reason` ≤ 500 chars. `children` keys are decimal task ids of the parent's project; external
dependents (§4.3) are keyed `"<project_id>:<id>"` and admit `repoint` only (`onto` are ids in the
parent's project, written into the child's `external_deps` as `"<parent_project>:<id>"`). Re-scope
is `keep` plus the canceller's own `update_task` edit; done-on-main is the canceller marking the
child `done` (`done_provenance.kind=found_on_main`) BEFORE the cancel, after which it is terminal and
not listed. No `defer` action (D4).

`superseded_by`: list of ≥1 distinct ints; each exists in the project, is not the parent, is not
`cancelled` (any other status is legal: a `done` superseder is the found-on-main case).

### 4.3 Child set and coverage

Computed by `cancel_dispositions.enumerate_children` (fused-memory side, pure over backend reads),
fresh at call time (INV-3), from three sources:

- `SqliteTaskBackend.find_tasks_by_metadata(root, 'spawned_from', <parent as str>)` AND the same
  call with the int form (`TaskCurator._sibling_lookup_values` moves to the shared module; the
  sweep's str-only read is the blind spot the substrate names);
- inverted `SqliteTaskBackend.get_dependency_edges(root)` for local dependents;
- for every OTHER known project root: `find_tasks_by_metadata(other_root, 'external_deps',
  '<parent_project>:<parent_id>')` (list-membership match) → `external_dependent`. The roots come
  from the interceptor's own `self._backlog_policy` — the same `BacklogPolicy` the recon harness
  seeds via `BacklogPolicy.register_known_project_roots` — through a new
  `BacklogPolicy.known_project_roots()` accessor (β; its only accessor today is
  `project_root_for`), falling back to
  `fused-memory/src/fused_memory/models/scope.py::known_project_roots_from_env`. If neither yields
  roots the view carries `external_dependents_checked: false` (INV-11; never an empty list
  standing for "none"), and each such view journals a counted `cancel_children_external_unchecked`
  row so a persistently blind registry is visible, not merely possible (INV-4).

Every `find_tasks_by_metadata` read pages on `has_more` (its `DEFAULT_LOOKUP_LIMIT` is 50 in
`fused-memory/src/fused_memory/backends/metadata_lookup.py`); a truncated read would under-count
silently. The `ChildRow` facts (title, declared files) cost one `get_task` per child — acceptable at
the measured sizes (≤31 children per parent), stated so nobody mistakes it for free.
`has_live_children` is LOCAL only (D5 concerns local fold/moot) and is derived from two
project-wide reads made once per enumeration — the inverted `get_dependency_edges` map and one
`find_tasks_by_metadata('spawned_from', include_value=True)` presence read — never from a
per-child re-enumeration, so the abandon pre-flight's 10 s probe is not N × K store scans.

Each child is a `ChildRow`: `id, project_id, title, status, relation ∈ {spawned_from, dependent,
both, external_dependent}, declared_files_total, declared_files_missing_from_main (≤20 + total),
depends_on_superseder: bool, named_superseder: bool, has_live_children: bool, escalation_id?,
existing_disposition?` — the mechanically decidable facts of report §3 Q3, never the decision.

Coverage rule (pre-cancel mode): let L = live (non-`done`/`cancelled`) children. Required:
`keys(children) ∪ named(superseded_by ∩ L)` ⊇ L, with every key naming a member of the VALIDATION
set V = tasks of any status that have `spawned_from == parent`, or parent in their deps or
`external_deps`, or a `cancel_dispositions` entry naming parent. Every `co_cancelled` id must
also be in V (→ `cancel_children_unknown` otherwise) and must not also appear as a `children`
key (→ `cancel_children_invalid`). A key outside V →
`cancel_children_unknown` (wrong parent). A key in V but terminal → accepted, reported under
`already_terminal` (benign race or retry after partial execution — this is what makes retry
converge, §4.6). A child named in `superseded_by` with no explicit entry → defaulted to `keep`,
reported under `defaulted_keep`; an explicit entry for a named superseder wins (a split child may
legitimately be `repoint`ed).

### 4.4 Refusal payloads (house gate shape)

All: `{'success': False, 'error': <slug>, 'task_id', ..., 'hint'}`; id lists IN PAYLOADS capped
at 50 with `<field>_total`, the `BulkResetVerdict.to_error_dict` convention. There is no cap on
the size of a map a call may CARRY: coverage is all-or-nothing in pre-cancel mode, so a write cap
would make a large parent un-cancellable. `error_type` is set to the
CamelCase of the slug so `@mcp_tool_errors` and `extract_rejection` callers both key on it.

| slug | structured fields | when |
|---|---|---|
| `cancel_children_undispositioned` | `children: [ChildRow]`, `children_total`, `undispositioned: [ids]`, `defaulted_keep`, `external_dependents_checked`, `skill: '/cancel-with-children'` | L not covered |
| `cancel_children_invalid` | `field`, `reason_code`, `value` (json-safe) | schema or referential failure (`onto` cancelled, `into == child`, cycle, `superseded_by` names the parent, …) — `DeferralInvalidRefusal` idiom |
| `cancel_children_unknown` | `unknown: [ids]`, `children: [ChildRow]` | key outside V |
| `cancel_children_nested` | `child`, `grandchildren: [ChildRow]` | fold/moot on a child with live children; hint: cancel the child first, bottom-up |
| `cancel_children_execution_failed` | `executed: [{child, action, outcome}]`, `failed: {child, step, error}` | partial execution (§4.6); parent untouched |
| `cancel_children_parent_terminal` | `task_id`, `status` | `resolve_issue(abandon)` carrying `children`/`superseded_by` on an already-terminal head (§4.12); hint: late path + `close_only` |
| `cancel_children_unverified` | `reason` | the abandon pre-flight probe timed out (§4.12); fail closed |

The hint on `undispositioned` is one line: run `/cancel-with-children <id>`; or pass
`children={...}` covering the listed ids; or, for an L2-held task, `resolve_issue(action='abandon',
children=...)`. The payload is sufficient for a skill-less agent (Stage 2, steward) to act, because
the rows carry the mechanical facts and the hint carries the action vocabulary (INV-2).

### 4.5 Execution order and atomicity (D2)

Rejected: a new backend `cancel_with_children` method in one `_write_unit`. It would be
all-or-nothing, but every child write would bypass the per-transition machinery the interceptor
owns — status events, recurrence mint, the reconcile trigger, lock-charter and consolidation
gates, write journaling — and would need all of it re-emitted by hand (heuristic 11 violation;
INV-5 lockstep with `_apply_status_transition`). It would also be the first multi-task backend
transaction in the codebase, for a case where non-atomicity is SAFE: the parent's cancel is the
last write, so no intermediate state releases a dependent early.

The gate runs on EVERY call with `status == 'cancelled'` that reaches
`TaskInterceptor.set_task_status` — a bare cancel included. `children` and `superseded_by` change
what coverage FINDS, never whether the gate runs (D1). Single-id calls carry the parameters; CSV
and `commit_planning` per §4.10.

**Per-parent serialization, then mode.** The whole sequence runs under an in-process
`asyncio.Lock` keyed `(project_id, parent_id)` (NOT `_write_lock`), so a retried call — e.g.
`Scheduler.set_task_status`'s 15 s per-attempt timeout firing on the same `client_op_id` while
the first attempt is still executing — waits, then observes a converged state instead of running
a second executor over the same children. Order: acquire that lock → `get_task(parent)` → decide
the mode. Parent non-terminal ⇒ pre-cancel mode (below); parent already `cancelled` ⇒ late mode
(§4.9), which never enters `_apply_status_transition` and so never meets the same-status guard
that sits inside the non-reentrant `_write_lock`. (Deciding the mode before taking the per-parent
lock would let a retry plan a pre-cancel run against a parent the first call has just
cancelled.)

Pre-cancel sequence:

1. Idempotency replay (unchanged, first).
2. Parse (pure). Enumerate children (§4.3). Pre-check coverage, unknown, nested, referential
   validity → refuse before any write. First enforcement point.
3. Execute dispositions, deterministic ascending child-id order, each through the interceptor's
   OWN public methods (each takes `_write_lock` briefly; nothing here holds the parent's lock —
   the lock is non-reentrant):
   - `repoint`: `add_dependency(child, onto_i)` for each `onto_i` (INSERT OR IGNORE) → then
     `remove_dependency(child, parent)` (idempotent DELETE) → then stamp. Add-before-remove is the
     esc-3525-2 safe sequence: the child is never dep-free in between.
   - `keep`: stamp only.
   - `fold` / `moot`: the PUBLIC `set_task_status(child, 'cancelled', _cancel_cause=<entry>)` —
     `_cancel_cause` is an interceptor-internal keyword, not an MCP tool parameter — so the child's
     cancel runs the full gate, including its own children check (a grandchild added since step 2
     is caught; `cancel_children_nested` is also pre-checked in step 2 for a clean message).
     `_apply_status_transition` gains a `cancel_cause` parameter: it computes the child's new
     `cancel_dispositions` list INSIDE the child's lock from `before` (replace-or-append by parent
     key, §4.8 — never a static value, because `set_status_and_stamp_audit` shallow-merges and
     would otherwise replace a two-parent child's whole list), passes it in `audit_fields` so status
     and stamp land in one `SqliteTaskBackend.set_status_and_stamp_audit` transaction, and threads
     `cancel_cause` to `reconciler.reconcile_task` (§4.11).
   - stamp (`keep`, `repoint`) = replace-or-append the child's `metadata.cancel_dispositions` entry
     for this parent via `update_task(metadata_mode='merge')` after an in-executor read; the
     per-parent lock above makes the read-modify-write safe against a retry of the same call
     (two DIFFERENT parents of one child cancelled concurrently remain T5).
4. Parent transition under `_write_lock(project)`: re-enumerate L fresh. Verifier: every member of
   L carries a `cancel_dispositions` entry for this parent with action `keep`, or `repoint` and no
   remaining edge to the parent, or is named in `superseded_by`. Otherwise refuse
   `cancel_children_undispositioned` naming the newcomers (a dependent added since step 2) —
   nothing about the parent written. Second enforcement point (heuristic 10: redundantly enforced,
   both points calling the one coverage function). Then write the parent's cancel with
   `superseded_by` as an audit field through `set_status_and_stamp_audit` — status and pointer land
   in one transaction.
5. Post-lock as today: event, `reconciler.reconcile_task(...)` with the structured cause (§4.11).
6. Idempotency record only on full success (existing rule in `_idempotency_record`).

Partial failure contract (INV-11): a step-3 failure stops the sequence and returns
`cancel_children_execution_failed` with every executed child's outcome and the failing step.
State left: some children re-pointed/stamped/cancelled, parent live and unchanged. Nothing is
released early (parent still live). Retry = the same call: executed steps are idempotent (edge
INSERT OR IGNORE, idempotent DELETE, stamp replace-by-parent, child cancel → same-status no-op or
"already terminal" in validation). The result dict of a successful call also lists
`executed` so a caller can see what the gate did on its behalf.

### 4.6 Idempotency / `client_op_id`

Whole-call replay as today (`_idempotency_replay` first). Only a full success is recorded; a
refusal or partial failure is re-attempted on retry and converges per §4.5. No new journal
table: executed child writes each produce their own `write_op`/`backend_op` rows through the
public methods they go through (`_journal_around`), plus one `cancel_children` summary row on the
parent's write (operation name `set_task_status`; β extends `_apply_status_transition`'s
`_journal_params`, today `{task_id, status, tag}`, with `children`, `superseded_by` and the
`executed` outcomes).

### 4.7 Recursion (D5)

A fold/moot child that itself has live children is REFUSED (`cancel_children_nested`); the skill
works bottom-up (grandchildren first, each with its own disposition map), so by the time the
parent's call runs its fold/moot children are leaves. No nested maps in the schema (heuristic 2;
one coverage function, one depth). The residue backstop covers anything that slips past.

### 4.8 Stamps and the canonical supersession key

**On the parent** — `metadata.superseded_by: list[int]` (Tier-A blessed in
`shared/src/shared/task_metadata.py::_BLESSED_METADATA_KEYS`; value validated at the cancel
write by the gate; `docs/task-authoring.md` Tier-A entry + Tier-B row
`superseded_by ← x_superseded_by, x_subsumed_by, subsumed_by, x_coalesced_into, x_replacement`
(`tests/scripts/test_task_authoring_tier_b_canonical_keys.py` pins that the canonical is
blessed). Of the row's aliases only the un-prefixed `subsumed_by` warns `unknown_key` (Tier-B
drift; α pins it in
`shared/tests/test_task_metadata.py::TestParseMetadataFailurePolicy::test_superseded_by_tier_b_alias_still_warns`).
The `x_` spellings (`x_superseded_by`, `x_subsumed_by`, `x_coalesced_into`, `x_replacement`)
sit in the Tier-C `x_` namespace that `shared/src/shared/task_metadata.py::parse_metadata`
never warns on (`docs/task-authoring.md` §8 Tier-C;
`shared/tests/test_task_metadata.py::TestParseMetadataFailurePolicy::test_x_prefixed_unknown_key_silent_zero_warnings`),
so the row lists them only as spellings to avoid in new writes — no `x_` carve-out
(esc-8142-1: with no corpus migration per D3, a carve-out would make every legacy pointer a
permanent census line). `superseded_by` and `cancel_dispositions` are Tier-A blessed and do not
warn `unknown_key`. Not the same spelling as the MEMORY key (`x_superseded_by`, string id, Mem0 record) on
purpose: same name, different store, different shape is heuristic-1 lying. The deferral
`carried_by` record stays what it is — a HOLD with a carrier, not a supersession — and
`x_coalesced_into` keeps its two legacy readers; neither is folded here. Written ONLY by the
cancel write (atomically) or the late path (§4.9); `update_task` may still write it by hand but
nothing in the design relies on that. No corpus migration (precedent task 3796; terminal rows
are mostly unrewritable and nothing reads the old spellings).

**On each child** — `metadata.cancel_dispositions: list[CancelDispositionEntry]`, registered
submodel `cardinality='list'` (`register_metadata_submodel`, the `delivered_checks` precedent),
Tier-A blessed. Entry: `{parent: "<project_id>:<id>", action, onto?|into?, reason?, stamped_at, stamped_by:
agent_id|None, source: 'canceller'|'backstop'}` — the parent is always qualified (the
`external_deps` spelling), because an external dependent's entry lives in ANOTHER project's store
where a bare id names a different task. Replace-by-`parent` semantics (one entry per
parent; a child of two cancelled parents carries two). Readers (INV-13: each lands with or after
the producer β, and renders "no entry" as residue, never as healthy): the backstop sweep (§4.11),
`get_cancel_children` (`existing_disposition`), the skill, `/review` phase 3 (an
`x_acceptance_reason` analogue for "why is this cancelled"), the dashboard task view (optional,
not in this PRD). The legacy sweep keys `parent_cancelled` / `parent_cancelled_relation` /
`needs_recheck_against_main` keep their one reader (`targeted.py::_classify_blocked_reason`) and
are written only by the dead-orchestrator block route (§4.11) — tactical whether to fold them
into an entry with `action: 'blocked'`.

### 4.9 Late disposition (post-cancel) — one verb for pre- and post-hoc (D9)

`set_task_status(parent, 'cancelled', children=..., superseded_by=...)` on an ALREADY cancelled
parent — detected by the pre-lock `get_task` (§4.5), so the call never enters
`_apply_status_transition` and never meets the same-status guard inside `_write_lock` — runs
steps 2–3 for the supplied subset under the per-parent lock — coverage is NOT
required (partial, after-the-fact dispositions are the point), unknown/nested/invalid still
refuse — stamps `superseded_by` if supplied via `update_task(metadata_mode='merge')`, and returns `{'success': True, 'no_op': True,
'late_disposition': True, 'executed': [...]}`. Consumers: the residue record's resolver (watcher
running the skill), recon Stage 2 acting on a residue record, and the backstop's own
deterministic moot route (§4.11) — so every child cancel in the system is a disposition
executed by the same code (heuristic 3: one mechanism per axis; heuristic 6). For a recon-stage
caller the `superseded_by` stamp on the terminal parent is refused by recon Gate 1 (Appendix A item 1): the
late path skips it and returns `superseded_by_stamped: false` (INV-11), never a silent drop.

### 4.10 CSV and `commit_planning`

`children`/`superseded_by` are accepted on single-id calls only (the `expected_stamped_at` /
`restamp_only` precedent); a CSV call carrying them → `cancel_children_invalid`
(`field='task_id', reason_code='csv_not_allowed'`). A CSV or `commit_planning(target_status=
'cancelled')` cancel still runs the gate per id with ONE relaxation: ids in the same batch are
excluded from L and stamped `co_cancelled` (intra-batch co-cancel; `_apply_status_transition`
gains `batch_ids`, the internal form of `co_cancelled`). A batch
member with a live child OUTSIDE the batch is refused for that id (the CSV result already carries
per-id results, `success=False` overall — INV-11); the hint says to cancel that id singly with
dispositions. If a batch's parent lands and its in-batch child's cancel then fails on a later
gate, the child carries a `co_cancelled` stamp but is still live; the backstop's live re-read
finds it non-terminal and files it in the parent's residue record flagged `co_cancel_failed`
(§4.11) — the stamp never hides a child whose own cancel did not land. `commit_planning` needs no new parameter.

### 4.11 Backstop sweep (`targeted.py::_sweep_cancelled_descendants` → extracted module)

Trigger unchanged: every `cancelled` transition through the interceptor. Changes:

- Enumeration via `enumerate_children` (§4.3) instead of a full `get_tasks` scan; this closes the
  str/int `spawned_from` blind spot for free.
- Corroborate (INV-3): re-read the parent live for `superseded_by`; skip children named there;
  skip children whose `cancel_dispositions` has an entry for this parent with action `keep`,
  `repoint`, `fold` or `moot`; a `co_cancelled` entry is corroborated, not trusted — skipped
  only if the child is terminal on the sweep's one live re-read, else it IS residue, flagged
  `co_cancel_failed` (its own cancel, scheduled separately, failed or has not landed); skip
  children that already depend on a named superseder (`depends_on_superseder`, journalled as
  `already_repointed`).
- Deterministic moot route kept, narrowed: `spawned_from` + `escalation_id` + none of its
  declared files on main (as today, a review-followup that declares NO files also qualifies —
  `targeted.py::_sweep_cancelled_descendants`'s `not files_some_exist`) + `has_live_children ==
  False` → executed as a LATE `moot` disposition through the
  interceptor (§4.9) with `stamped_by='reconciliation-sweep'`, `source='backstop'`. The
  `parent_cancelled:` reopen-reason prefix is no longer written.
- Recursion guard: `reconcile_task(task_before, reopen_reason, *, cancel_cause:
  CancelDispositionEntry | None)`; `_on_task_cancelled` short-circuits when `cancel_cause` is a
  fold/moot (the gate proved the child had no live children). Structured, not a prefix sniff.
- Residue: ONE record per parent instead of one L1 per child. Keyed on the parent id, category
  `scope_violation` (existing), level 1. The residual rows go in a typed
  `Escalation.cancel_children: list[dict] | None` field (added by α with the `granted_files`
  zero-migration pattern; each element is `ChildRow.model_dump()`, because `Escalation` is a
  dataclass serialized through `asdict` — the `EvidenceEntry` / `TrainState` precedent — and the
  same model the refusal carries, SPOT); `evidence` keeps its
  `EvidenceEntry` shape (`escalation/src/escalation/models.py::EvidenceEntry`) with one entry per
  child (`observation` = the rendered mechanical facts, `measured_at` = enumeration time,
  `ref` = `task:<project_id>:<id>`), so existing renderers keep working;
  `suggested_action='/cancel-with-children <parent> --late'`. It is resolved by the late path
  followed by `close_only` — never by `abandon` (§4.12). The
  journal row `descendant_escalated` keeps its type with `task_ids: [...]` (the report's
  measurement query stays valid). Severity and pin interplay: tactical T1.
- Orchestrator dead (`orchestrator_detector.is_orchestrator_live_for`): the block route stays for
  relation `dependent`/`both` only (the early-release arm — nothing else is watching), writing the
  legacy block keys; the residue record is filed either way (the queue is file-based).
- No grace window: with dispositions executed BEFORE the parent's cancel, the race that made 71%
  of flags redundant is gone for every compliant canceller; residue is real on the first look.

**Transition window (β landed, γ not yet).** β makes two minimal edits to `targeted.py` so the
old sweep neither storms nor goes silent while γ is in flight: (1)
`TargetedReconciler.reconcile_task` accepts `cancel_cause` and `_on_task_cancelled`
short-circuits on a fold/moot cause (otherwise a child cancel executed by β's gate re-enters the
old sweep); (2) the old sweep skips any child carrying a `cancel_dispositions` entry for this
parent with action `keep`, `repoint`, `fold` or `moot` (otherwise every `keep`/`repoint` child of
a compliant cancel draws an L1); a `co_cancelled` child is NOT skipped on the stamp alone but
falls to the old sweep's existing live re-read, so a member whose own cancel failed still draws
its L1 in the β→γ window. β needs nothing for `_sweep_cancel_orphan`'s bare child cancel: the gate observes
only until ζ (§4.14), and γ replaces the module before ζ flips enforcement. γ then replaces the
module wholesale.

### 4.12 `resolve_issue(action='abandon')` — must not jam silently (D6)

Chosen: pre-flight refusal + persisted dispositions + loud re-file on a late refusal. Each alone
leaves a hole: pre-flight alone loses to the race (a dependent added between pre-flight and the
harness write); persistence alone makes the human's first attempt a blind one; re-file alone makes
the normal path a two-round trip.

1. **Shape validation** in `escalation/src/escalation/server.py::resolve_issue` via the shared
   model (pure; refuses `cancel_children_invalid` before `queue.resolve`, like the Table-B gate).
2. **Pre-flight** through a harness probe, `Harness.cancel_children_blocker(task_ids, children,
   superseded_by)`, the `granted_files_delivery_blocker` precedent, SYNC on the FastMCP worker:
   it runs `get_cancel_children` for the head task and EVERY pending member's task via
   `run_coroutine_threadsafe(...).result(timeout=10)` on the harness loop (I/O only — INV-8) and
   returns the first refusal payload (flat `children` map; coverage checked per parent; a key
   that is a child of no cascade parent → `cancel_children_unknown`). **Cascade co-cancel:** the
   typical L2 clusters a parent's record with member L1s on its own children, and those members
   are cancelled by the same act; so the pending members' task ids are passed as `co_cancelled` (§4.1) on
   every cascade cancel, in the probe and in the harness — a member that IS a child of the head is
   excluded from the head's coverage and stamped `co_cancelled`, so the head's fire-and-forget
   backstop, which runs while the members' separately scheduled teardowns
   (`Harness._on_escalation_resolved`) are still in flight, skips them instead of filing a false
   residue record. **Enforcement parity:** `CancelChildrenView` carries `enforce: bool`, read from
   the same `cancel_children.enforce` leaf; the probe refuses only when it is true and otherwise
   attaches the would-refuse payload to the `resolve_issue` response as
   `cancel_children_would_refuse` and lets the resolve proceed — so between δ and ζ an abandon is
   never stricter than a direct cancel (one flag, one coverage function, both enforcement points
   agreeing; heuristic 10). **Terminal head:**
   `abandon` carrying `children`/`superseded_by` whose head task is already terminal is refused
   `cancel_children_parent_terminal` (hint: run the late path, `set_task_status(<id>,
   'cancelled', children=…)`, then `close_only`), because the harness abandon arm returns at its
   terminal recheck (`Harness._action_teardown_and_set_status`) before any write — the archived
   decisions would otherwise never execute. Refusal → returned verbatim,
   record untouched (the `declared_pin_refused` precedent: pre-flight BEFORE `queue.resolve`
   because the head is archived before the cascade). `harness is None` → today abandon cancels
   nothing anyway; the response gains `task_status_effect: 'none_no_orchestrator'` (INV-11 on a
   pre-existing silence). Probe timeout → refuse `cancel_children_unverified` (fail closed; hint:
   retry, or cancel through `set_task_status` directly).
3. **Persist** `children` and `superseded_by` on the `Escalation` record (fields with the
   `granted_files` zero-migration pattern) via `EscalationQueue.resolve(...)`, so the archived
   record IS the audit of what the human decided (INV-9: the disposition's home before it is
   executed).
4. **Harness** `_action_teardown_and_set_status` (abandon arm) reads them from the record. For a
   cascade member (`resolved_by='l2-cascade:<id>'`, action inherited by
   `_resolve_escalation_action`) it reads the PARENT L2's record and filters the flat map through
   `get_cancel_children(member.task_id)`; passes `children`/`superseded_by` to
   `Scheduler.set_task_status` (explicit new kwargs → MCP arguments) with a per-call timeout
   scaled to the map size (the default 15 s per attempt is for single-row writes).
5. **Typed rejection**: `Scheduler.set_task_status` maps `cancel_children_*` codes to
   `CancelChildrenRejection(SetTaskStatusRejected)` carrying the structured payload. The harness
   catches it specifically: files a NEW L2 on the task — category `scope_violation`, the payload's
   rows in `cancel_children` and one `EvidenceEntry` per row in `evidence` (§4.11's shape),
   `suggested_action='/cancel-with-children'`, summary "abandon refused: N
   undispositioned children" — and leaves the task's status as it was (blocked, now with an open
   owner again: INV-6/INV-7). The WARNING-and-return stays for the other `SetTaskStatusRejected`
   kinds, which are genuine no-ops (row already terminal).

### 4.13 Steward and recon Stage 2

Steward (`roles.py::STEWARD`, the only dispatched role holding `set_task_status`): no change; the
refusal is self-describing. Stage 2 (`prompts/stage2.py`): the FOUR cancel-instruction sites
(the "Invalidated task assumptions" item, the Authority Model, the "Be conservative with task
cancellation" guideline, and the "never route status through update_task" note) become: call `get_cancel_children`
first; decide per child with the rendered table (§5.3); pass `children=`; a child you cannot
decide means you do NOT cancel — re-scope or report via the recon_report tools (its existing
"be conservative" rule). Stage 2 may act on a residue record through the late path (§4.9).
Stage 2 may not `fold`/`moot` an `in-progress` child: its actor class cannot leave `in-progress`
(`shared/src/shared/task_transitions.py`, `ActorClass.RECONCILIATION`); the rendered rules and the
refusal hint say so.

### 4.14 Entry points, exemptions, ownership (INV-12)

**Rollout: β lands observing, ζ makes it enforcing.** Between β and δ the harness abandon arm
would meet a refusal with its WARNING-and-return (a silent jam), and until ε lands the six prose
sites still teach bare cancels. So β ships the gate behind one hot-reloadable fused-memory config
leaf, `cancel_children.enforce` (default `false`): declared in
`fused-memory/src/fused_memory/config/schema.py`, listed in
`fused-memory/src/fused_memory/config/reload.py::RELOADABLE_FIELDS` with the reload-safety note
that file requires, and read per call from the interceptor's shared `self._config` (the
`_enforce_transitions` idiom) — without the allowlist entry `reload_config` would answer
`restart_required`. With it off, an undispositioned cancel is ALLOWED and journals a counted
write-op row under its own operation name, `cancel_children_would_refuse`, with the refusal
payload in the row's JSON `params` (not its text `error`), so ζ's review is one query; everything
else is live — a supplied map is validated and executed, `get_cancel_children` and the late path
work, and the backstop catches the rest. ζ flips it to `true` once δ and ε have landed and the
would-refuse rows are understood. It is one owned, dated flag (Debt, owner η), not an allow-list:
η, a delayed milestone seven days after ζ, removes it.

There is no allow-list: every interceptor path runs the gate, so there is nothing to register.
The two interceptor-bypassing paths are pre-existing and out of scope (§9).

| entry point | behaviour under the contract | why no exemption is needed |
|---|---|---|
| `tools.py::set_task_status` (interactive, skills) | gate; skill supplies dispositions | the designed path |
| `clerk_batch.py::_RECIPES[CANCEL]` | `target` → `superseded_by`; `LeadAction` gains `children` (validated by the shared model); recipe passes both | the lead decides in the sitting; the clerk executes |
| `commit_planning(cancelled)` / CSV | intra-batch exclusion; outside children refuse per id | §4.10 |
| `resolve_issue(abandon)` → harness | §4.12 | — |
| `Harness._maybe_auto_eval` | passes nothing; eval tasks are structurally childless; a refusal raises `CancelChildrenRejection` into the existing eval failure log | a refusal here is a real anomaly and loud |
| steward | gate; refusal self-describing | — |
| recon Stage 2 | §4.13 | — |
| backstop sweep | its own cancels are late `moot` dispositions on leaves | §4.11 |
| `audit_duplicate_tasks.py::apply_changes`, v3→v4 migration self-heal, `remove_task` | bypass; out of scope (§9) | pre-existing; neither fires the sweep today either |

### 4.15 `irreversible_deletion` in the watcher skill

Today `skills/escalation-watcher/SKILL.md` lists "cancelling a task with dependents" under the
always-ask `irreversible_deletion` class (a POLICY slug in
`escalation/src/escalation/shadow_ruling.py::HUMAN_FOREVER_GATES`; definition in
`docs/escalation-standing-policy.md`), while its supersession carve-out self-executes a cancel
after re-pointing every dependent. Ruled D7: the
class keeps content loss — a `fold`/`moot` on a child whose branch has commits
(`git rev-list --count main..task/<id>` > 0) or whose declared files are absent from main AND no
superseder re-creates them — and the carve-out's four checks plus a complete disposition map are
sufficient for everything else. The skill's carve-out steps 1–3 collapse into: `get_cancel_children`
→ table → one `set_task_status(..., children=, superseded_by=)` (or `resolve_issue(abandon, ...)`).

## 5. Decision table (owned by `/cancel-with-children`)

### 5.1 Single home and how Stage 2 consumes it

Rows live as DATA in `shared/src/shared/cancel_dispositions.py::DECISION_TABLE` (tuple of
`DecisionRow(order, question, evidence_check, action, example_ids)`). Two renderers derive from
it: `render_decision_table_markdown()` writes the table between `<!-- cancel-decision-table -->`
markers in `skills/cancel-with-children/SKILL.md` (a content-pin test asserts the file's block
equals the render — the `docs/task-authoring.md` Tier-B marker pattern), and
`render_cancel_children_rules()` composes the Stage 2 prompt section (the
`live_workflow_section.py::render_live_workflow_authority_rules` precedent — prompts are string
constants composed by import, none reads files). INV-5: the skill's copy is a mechanical mirror
with a test; the prompt's is a render. The skill owns the PROSE around the table (how to transfer
fold content, bottom-up ordering, the late mode); the data module owns the rows.

### 5.2 The ordered questions (per child; stop at the first yes)

| # | question | evidence check | action | report example |
|---|---|---|---|---|
| 0 | Already delivered on main? | child's declared files/behaviour landed under another commit (`git log -S`/`delivered_checks` green) | mark child `done` `found_on_main` BEFORE the cancel (then it is terminal, unlisted) | 3378 |
| 1 | Is the child one of the replacements the superseding act names? | `named_superseder` (in `superseded_by`) | **keep** (default) | 8032–8058; 928/930 |
| 2 | Does it stand on main independent of the parent? | `declared_files_missing_from_main == []` and its description names no parent-branch artifact; `spawned_from` is provenance only | **keep** (U) | 6906, 7007, 8207, 733 |
| 3 | Was its premise the parent's DELIVERY (sequencing, "after N lands", a gate clock)? | relation `dependent`/`both`/`external_dependent`; description references the parent's output | **repoint** onto the superseder(s) (P). This is the class where silence does harm: locally it releases early, cross-project it wedges | 952→1112, 5770→6580, 7738→8891, 6905/7996→8032 |
| 4 | Do its artifacts exist only on the parent's branch or plan, and does a superseder cover its scope? | `declared_files_missing_from_main` non-empty; read the superseder's description/plan for each child item | covered → **fold** into it after copying uncovered items into the superseder's `details` while it is `pending` (F); not covered → **moot** (M) | F: 937/960, 7997/7998→8032, 5160→5261; M: 6367, 771, 152 |
| 5 | Partly on main? | some declared files present, some missing | **keep** + `update_task` trimming the child to what main lacks (R) | 261, 792 |
| 6 | Cannot decide (superseder's scope unknown to you)? | — | do NOT cancel the parent yet: ask (watcher → Leo; Stage 2 → recon_report) | — |

Bottom-up rule: if a fold/moot child has live children, run the table on them first.

## 6. Boundary-test sketch

Model: `fused-memory/tests/test_status_authority_gate.py::_fresh_stack` (real `SqliteTaskBackend`
+ real `TaskInterceptor`); drive ONLY public seams (`set_task_status`, `get_cancel_children`,
`add_dependency`, `get_task`, `reconcile_task`). No patching of private names (Tests stance).

| # | seam | scenario | preconditions | postconditions |
|---|---|---|---|---|
| 1 | MCP ↔ interceptor | undispositioned cancel | P live; C1 `spawned_from=P` (int-valued), C2 depends on P, C3 done | `cancel_children_undispositioned`; `children` lists C1 (relation spawned_from) and C2, not C3; P still live; no stamps |
| 2 | same | full map executes | as 1 plus S exists; `children={C1: moot, C2: repoint onto [S]}`, `superseded_by=[S]` | P cancelled with `superseded_by=[S]`; C1 cancelled with entry `{parent P, moot}`; C2 deps `[S]`, no P, entry `{repoint, onto [S]}`; `executed` lists both |
| 3 | same | superseder default keep | C2 named in `superseded_by`, no entry | success; `defaulted_keep=[C2]`; C2 stamped keep; C2's edge to P remains |
| 4 | same | unknown key | `children` names X (unrelated task) | `cancel_children_unknown {unknown:[X]}`; nothing written |
| 5 | same | nested fold | C1 moot, C1 has live child G | `cancel_children_nested {child C1, grandchildren [G]}`; nothing written |
| 6 | same | invalid onto | `repoint onto [P]` / onto a cancelled task / `into == child` | `cancel_children_invalid` with `field`, `reason_code` |
| 7 | same | CSV with children | `task_id="P,Q", children=…` | `cancel_children_invalid csv_not_allowed` |
| 8 | same | intra-batch co-cancel | CSV `"P,C1"`, C1 depends on P, no children param | both cancelled; no refusal |
| 9 | same | batch member with outside child | CSV `"P,Q"`, C2 depends on P only | P refused `undispositioned`, Q cancelled; `success=False` |
| 10 | interceptor ↔ backend | partial failure converges | inject failure on C2's `add_dependency` (fault-injecting backend subclass through the public constructor) | `cancel_children_execution_failed` with `executed=[C1…]`, `failed={C2, add_dependency}`; P live; retry with the healed backend succeeds; C1 not double-written (same-status no_op) |
| 11 | same | newcomer race | after step 2, add D depending on P before the parent lock (hook via a second interceptor call interleaved with `asyncio.Event` on the public API) | refusal names D; P live; C1/C2 dispositions already landed and are honoured on retry |
| 12 | same | idempotency replay | `client_op_id` on a successful call, repeat | replayed result; no second `executed` writes (journal rows count unchanged) |
| 13 | late path | post-hoc disposition | P already cancelled, C2 live | `success, no_op, late_disposition, executed=[C2]`; C2 repointed; partial coverage allowed |
| 14 | late path, recon | Stage-2 identity stamps parent | `agent_id='recon-stage-…'`, `superseded_by=[S]` | children executed; `superseded_by_stamped=False` |
| 15 | cancel ↔ backstop | compliant cancel yields no residue | scenario 2 then `reconcile_task` | sweep actions: `descendant_skipped_dispositioned` ×2, no escalation filed |
| 16 | same | residue → one record | P cancelled via bypass (direct backend write in the test) with C1, C2 live, orchestrator live stub | ONE `scope_violation` record on P with `evidence` = 2 ChildRows; journal `descendant_escalated` with `task_ids` |
| 17 | same | deterministic moot via late path | C1 `spawned_from=P` (str form), `escalation_id`, declared files absent, no live children | C1 cancelled with entry `{moot, source backstop}`; no `parent_cancelled:` reopen_reason anywhere |
| 18 | same | str/int blind spot | C1 `spawned_from=5869` (int) | enumerated (today: invisible) |
| 19 | same | dead orchestrator | as 16, liveness stub false | dependents blocked with legacy keys; record still filed |
| 20 | resolve_issue ↔ harness ↔ fm | pre-flight refusal | L2 on P; P has live C2; `resolve_issue(abandon)` without children | `cancel_children_undispositioned` returned; record still pending; no harness write (orchestrator/tests/test_harness_action_dispatch.py style, real queue dir) |
| 21 | same | abandon with map | `children={C2: repoint [S]}, superseded_by=[S]` | record archived carrying both; harness cancels P with them; C2 repointed |
| 22 | same | cascade members | L2 with members on P and Q; flat map covers both parents' children | each member's cancel receives its filtered subset; a key in neither → pre-flight `cancel_children_unknown` |
| 23 | same | late refusal re-files | pre-flight passes; dependent D added before the harness write | `CancelChildrenRejection`; new L2 on P with `evidence` payload; P status unchanged (blocked) |
| 24 | same | no harness | `harness=None` | response `task_status_effect='none_no_orchestrator'` |
| 25 | Stage 2 ↔ refusal | prompt carries the rules | render | `render_cancel_children_rules()` names `get_cancel_children` and every action in `ChildDisposition`; SKILL.md marker block == `render_decision_table_markdown()` |
| 26 | vocabulary | blessing | parse a task carrying `superseded_by` / `cancel_dispositions` | no `unknown_key` for `superseded_by` / `cancel_dispositions`; `subsumed_by` on a task warns `unknown_key` (Tier-B alias); `x_superseded_by` and the row's other `x_` aliases stay silent (Tier-C, esc-8142-1); machine-authored keys ⊆ blessed (`test_pending_since_anchor.py` family) |
| 27 | MCP ↔ interceptor | bare cancel is gated | enforce on; P live with C2 depending on it; `set_task_status(P,'cancelled')` with NO children/superseded_by | `cancel_children_undispositioned`; P live |
| 28 | same | late mode does not deadlock | P cancelled; C2 live; late call with `children={C2: repoint [S]}` under the real `_write_lock` | returns within a bounded time (test timeout), C2 repointed; a concurrent unrelated write to the same project completes |
| 29 | same | concurrent retry of one call | two identical calls for P issued concurrently (same `client_op_id`), first slowed at child 1 | second waits on the per-parent lock, then returns the replayed result or the late-mode shape (`late_disposition: True`), never an untagged `no_op`; each child written once |
| 30 | same | large map accepted | P with 75 live children, complete map | success; refusal payload lists (when refused) cap at 50 with `_total` |
| 31 | same | observe mode | `cancel_children.enforce: false`; bare cancel of P with live C2 | P cancelled; one `cancel_children_would_refuse` journal row carrying the payload; backstop files one residue record |
| 32 | cancel ↔ backstop (transition window) | β landed, γ not | old sweep; compliant cancel of P with `keep` C1 and `moot` C2 | no L1 for C1 (stamp honoured); C2's cancel does not re-enter the sweep (`cancel_cause`) |
| 33 | resolve_issue ↔ harness | terminal head refused | residue record on cancelled P; `resolve_issue(abandon, children=…)` | `cancel_children_parent_terminal`; record still pending |
| 34 | same | cascade member that is a child | L2 head on P with a member L1 on C1 (`spawned_from=P`); abandon with `children` covering only P's OTHER children | C1 passed as `co_cancelled` and stamped so; P and C1 both cancelled; no refusal; the head's backstop files NO residue record although C1's teardown lands after P's |
| 35 | cancel ↔ backstop | co-cancelled child whose own cancel fails | CSV `"P,C1"`, C1 depends on P; C1's cancel refused by a later gate | P cancelled; C1 live with a `co_cancelled` entry; the backstop files ONE residue record on P listing C1 flagged `co_cancel_failed` |

## 7. Decomposition plan (coarse; overlay bands 300–1,500 LOC, ≤10–12 files)

| leaf | title | packages / files | observable signal | prereqs | LOC |
|---|---|---|---|---|---|
| α | Cancel-disposition contract module + vocabulary | `shared/src/shared/cancel_dispositions.py` (new: models, `DECISION_TABLE`, both renderers, `sibling_lookup_values`; imports NOTHING from `task_metadata` — registration happens from `task_metadata.py`, the `DeferralRecord` precedent), `shared/src/shared/task_metadata.py` (bless `superseded_by`, `cancel_dispositions`; register submodel), `fused-memory/src/fused_memory/middleware/task_curator.py` (`_sibling_lookup_values` moved out, imported back — not duplicated), `escalation/src/escalation/models.py::Escalation` (optional `cancel_children`, `children`, `superseded_by` fields, `granted_files` zero-migration pattern), `docs/task-authoring.md` (Tier-A entries, Tier-B row, "retire a task" pointer), `shared/tests/test_cancel_dispositions.py`, Tier-B test row | `python -c 'from shared.cancel_dispositions import ChildDisposition'` parses the §4.2 examples and rejects the §6 row-6 cases with the named `reason_code`; a task carrying `subsumed_by` warns `unknown_key` while one carrying `x_superseded_by` stays silent (Tier-C) | — | 500–700 |
| β | Interceptor gate (observe mode), executor, `get_cancel_children` tool, late path, transition-window sweep edits | `fused-memory/src/fused_memory/middleware/task_interceptor.py` (set_task_status params, per-parent lock then mode decision, `_apply_status_transition(cancel_cause, batch_ids)`, verifier, `_journal_params`), `fused-memory/src/fused_memory/reconciliation/backlog_policy.py` (`known_project_roots()`), `fused-memory/src/fused_memory/reconciliation/targeted.py` (the two §4.11 transition-window edits only), `fused-memory/config/config.yaml`, `fused-memory/src/fused_memory/config/schema.py` and `fused-memory/src/fused_memory/config/reload.py` (`cancel_children.enforce: false`, allowlisted, per-call read), new `fused-memory/src/fused_memory/middleware/cancel_children_gate.py` (enumeration + coverage + executor; keeps `task_interceptor.py` from growing — heuristic 14; imports downward only — heuristic 13), `fused-memory/src/fused_memory/server/tools.py` (params, new tool), `fused-memory/tests/test_cancel_children_gate.py` (§6 rows 1–14, 26–31) | §6 row 1 (with enforce on in the test stack) and row 2 through the MCP tool against a real store; row 31 (observe mode allows and journals) | α | 1,100–1,500 |
| γ | Backstop sweep rewrite | extract `_sweep_*` from `targeted.py` into `fused-memory/src/fused_memory/reconciliation/cancel_backstop.py`; `reconcile_task(cancel_cause=)`; `fused-memory/tests/test_targeted.py::TestSweepCancelledDescendants` rewritten against a real stack (drop the interceptor-kwarg mock assertions — Tests stance); wiring pin for `task_interceptor` | §6 rows 15–19, 32, 35; `descendant_escalated` journal rows carry `task_ids`; residue records carry `cancel_children` | α (Escalation field), β (producer of the stamps the sweep reads — INV-13; same file `targeted.py`) | 600–900 |
| δ | Abandon path: pre-flight, persistence, pass-through, loud re-file | `escalation/src/escalation/server.py::resolve_issue`, `escalation/src/escalation/queue.py::resolve` (stamps the α fields), `orchestrator/src/orchestrator/harness.py` (`cancel_children_blocker`, cascade `co_cancelled`, terminal-head refusal, teardown arm, re-file), `orchestrator/src/orchestrator/scheduler.py` (`children`/`superseded_by`/`co_cancelled` kwargs, size-scaled timeout, `CancelChildrenRejection`), tests in `escalation/tests/` and `orchestrator/tests/test_harness_action_dispatch.py` | §6 rows 20–24, 33–34 | α, β | 700–1,000 |
| ε | Skill + prose sites + Stage 2 render | `skills/cancel-with-children/SKILL.md` (new; marker block; late mode; bottom-up; fold transfer recipe), `scripts/setup-host.sh` §8 map + `SETUP.md` §8, `skills/escalation-watcher/SKILL.md` (carve-out rewrite; D7), `fused-memory/src/fused_memory/reconciliation/prompts/stage2.py` (+ render import), `skills/review-all/scripts/program/clerk_batch.py` + `state.py` (`children` on CANCEL), `skills/review/references/phase3-triage.md`, `skills/_shared/filing-the-trigger-chain.md` §5, `OPERATIONS.md` (watcher dispatch row for `scope_violation` residue records: resolved by the late path + `close_only`; `/cancel-with-children --late` is a Skill-tool invocation for a dispatched watcher), `CHANGELOG.md` `[Unreleased]`, pin test for the SKILL.md block | §6 row 25; `grep -c 'repoint every' skills/escalation-watcher/SKILL.md` == 0; `/cancel-with-children` resolves via the symlink | α, β (tool names must exist before prose points at them) | 400–700 |
| ζ | Integration gate + enforce flip | end-to-end: real fused-memory stack + harness abandon; review the `cancel_children_would_refuse` rows since β, then flip `cancel_children.enforce: true` in `fused-memory/config/config.yaml` (ζ's only edit to a β file; the merged value takes effect at fused-memory's next staleness redeploy, or at once on an operator `reload_config`, since the leaf is allowlisted); `delivered_checks` for α–ε's symbols | §6 table executed green on a real stack; `cancel_children.enforce: true` on main | γ, δ, ε | 300–500 |
| η | Production observation + retire the enforce flag | delayed milestone, 7 days after ζ is done: the INV-13 production observation — read-only queries of the live store find ≥1 real cancel whose children carry `cancel_dispositions` and count residue records and `cancel_children_*` refusals over the week against the 126-flags-per-4-weeks baseline, recorded in the task and as a "landed" addendum to `plans/superseded-parent-children-2026-11-01.md`; then delete `cancel_children.enforce` from `fused-memory/config/config.yaml`, `config/schema.py`, `config/reload.py` and the interceptor's read; remove the observe-mode branch and its tests | `git grep cancel_children.enforce` finds nothing; §6 row 27 still green | ζ | 100–200 |

Same-file serialization: `task_interceptor.py`, `tools.py`, `backlog_policy.py` only in β;
`targeted.py` in β (two small edits) then γ (rewrite), serialized by γ → β; `config.yaml` in β,
then ζ (one value flip), then η; `config/schema.py`, `config/reload.py`, `task_interceptor.py` in
β then η — all serialized by η → ζ → β; `models.py`, `task_metadata.py`, `task_curator.py` only in
α; `harness.py`/`scheduler.py`/`server.py`/`queue.py` only in δ; `stage2.py`, every SKILL.md,
`OPERATIONS.md`, `CHANGELOG.md` only in ε. γ and δ are parallel after β; ε lands after β (tool
names must exist before prose points at them). Between β and ζ the gate observes only (§4.14), so
no deployed caller breaks while δ and ε are in flight.

G7 walk (slugs only): `contracts-machine-checked` (schema module, α); `structured-facts-at-failure`
(payloads §4.4); `corroborate-before-acting` (fresh enumeration at both enforcement points, sweep
re-read); `no-lockstep-duplication` (one data table, two renders, pin test); `one-fact-one-home`
(`superseded_by` on the parent is the only supersession home; the child's entry names the parent,
not the superseder set); `no-silent-fail-soft` (partial-failure payload; `external_dependents_checked`;
`superseded_by_stamped`; `task_status_effect`; loud re-file); `exceptions-owned-or-ratified`
(no allow-list added); `readers-prove-their-producer` (γ after β; η production observation);
`status-matches-liveness`/`holds-owned-and-bounded` (re-filed L2 restores an owner to a blocked
task). No waiver needed.

## 8. Pre-conditions and cross-PRD relationship

**Pre-conditions.** None blocking. The one substrate question raised during design — can the
interceptor reach the known-project-roots registry — is answered (Appendix A item 7): yes, via
its own `self._backlog_policy` plus a new accessor, with an env fallback.

**Seams (G4).** No other active PRD owns any mechanism changed here.

| seam | other side | direction | owner |
|---|---|---|---|
| Task metadata vocabulary (Tier-A blessed set, Tier-B canonical table) | `shared/src/shared/task_metadata.py`, `docs/task-authoring.md` | this PRD adds `superseded_by`, `cancel_dispositions` | this PRD, leaf α |
| Deferral `carried_by` record (`shared/src/shared/task_deferral.py`; `plans/deferral-flip-condition-prd.md`) | a HOLD with a carrier, not a supersession | untouched; `x_coalesced_into` keeps its two legacy readers | deferral PRD (no change) |
| `irreversible_deletion` human-forever gate | `escalation/src/escalation/shadow_ruling.py::HUMAN_FOREVER_GATES`; definition in `docs/escalation-standing-policy.md` (generic, unchanged) | only the watcher skill's gloss "cancelling a task with dependents" changes (D7) | this PRD, leaf ε |
| Mem0 supersession (`shared/src/shared/memory_supersession.py`, memory `x_superseded_by`) | a different store and shape | untouched; the task key is deliberately spelled differently (heuristic 1) | memory metadata vocabulary (no change) |
| Escalation `granted_files` record pattern | `escalation/src/escalation/models.py::Escalation`, `EscalationQueue.resolve` | this PRD adds two fields by the same zero-migration pattern | this PRD, leaf δ |

## 9. Out of scope (and what happens to each)

- **Subtasks** (`parent_id`): stay flag-only (`subtasks_need_review`) as today.
- **`remove_task`**: a DELETE, not a cancel; no gate, no sweep. `docs/task-authoring.md` already
  says cancel-don't-remove; ε adds one sentence pointing here.
- **`audit_duplicate_tasks.py::apply_changes`** and the **v3→v4 migration self-heal**: backend-direct
  cancels; neither runs the gate nor fires the sweep (pre-existing). Their orphans surface only
  when something else triggers a sweep on the parent, or never. Named here so the next reader
  does not assume coverage; a follow-up could route the audit script through the late path.
- **A scheduler-side hold on dependents** (option C): not needed once dispositions precede the
  cancel; the dead-orchestrator block route is the only hold kept.
- **Dashboard rendering** of `cancel_dispositions` / `superseded_by`; **corpus migration** of
  legacy pointers; **Mem0 `x_superseded_by`** (a different store, untouched).
- **Grace window / deferred re-evaluation** in the sweep (option A's timing): dropped, §4.11.

## 10. Open questions (tactical only)

T1 residue record severity/level and `escalation.pins` interplay for a terminal subject (verify
`moot-terminal-subject` classification). T2 caps: `reason` 500, `declared_files_missing` 20, payload id lists
50 (there is no cap on the map a call carries, §4.4). T3 fold the legacy dead-orchestrator block keys into a `cancel_dispositions` entry with
`action: 'blocked'`, or leave them. T4 cross-project superseder (`"proj:id"` in `superseded_by`) —
ints only for now. T5 the stamp RMW race when two parents of one child are cancelled concurrently
(accepted; note in β). T6 `get_cancel_children` paging. T7 whether `/review` phase 3 renders
`cancel_dispositions` in its disposition lookup. T8 `LeadAction.children` field shape in
`state.py`. T9 whether the sweep auto-closes a residue record once a late disposition clears it.

## Appendix A. Substrate notes (verified 2026-11-01; corrections to common belief)

1. **"A cancelled task's metadata is effectively frozen" is prose, not code, for most callers.**
   The only freeze is `fused-memory/src/fused_memory/middleware/recon_write_policy.py` Gate 1
   (`update_task` on a terminal row), consulted in `TaskInterceptor.update_task` solely for
   `recon-stage-*` callers (`is_recon_stage_write`). A HUMAN/ORCHESTRATOR `update_task(metadata_mode='merge')`
   on a cancelled task succeeds. `skills/review/references/phase3-triage.md` asserts the freeze as
   fact. Consequence: late (post-cancel) stamps on a parent work for humans and the orchestrator,
   not for recon Stage 2 (§4.9).
2. **Cross-project dependents WEDGE, they do not release early.**
   `orchestrator/src/orchestrator/dependency_verdict.py::_external_deps_verdict` satisfies an
   external dep only on cached status exactly `'done'`; a cancelled parent in another project leaves
   its external dependents UNSATISFIED forever. The substrate's harm statement (early release) is
   the LOCAL arm only. Both arms are in scope of the contract (§4.3 relation `external_dependent`).
3. **The "natural slot after terminal-exit" is wrong for one mode.** In
   `TaskInterceptor._apply_status_transition` the same-status no-op returns before every later
   gate, so a `cancelled -> cancelled` call never reaches a gate placed there — and that guard is
   inside the non-reentrant `_write_lock`, so a late mode branching there would deadlock the
   project's writes on its first child write. The mode is therefore decided in `set_task_status`
   from a pre-lock `get_task` (§4.5, §4.9).
4. **The sweep's recursion guard is a string-prefix route.**
   `targeted.py::_PARENT_CANCELLED_REOPEN_PREFIX` sniffed from `reopen_reason` (heuristic 12,
   structured data instead of meaningful strings). Replaced by a structured cause (§4.11).
5. **`Scheduler.set_task_status` builds a fixed argument dict** (`id/status/project_root/agent_id`
   plus optional provenance/reopen/claimant keys); nothing passes through. Unknown rejection
   codes raise the base `SetTaskStatusRejected`. δ adds explicit parameters and a typed subclass.
6. **`TestServerWiringContract` does not pin the sweep's interceptor wiring.** Its one test pins
   `planned_episode_registry`; the sweep docstring's claim is unverified. γ adds the pin.
7. **The known-project-roots registry is reachable from the interceptor.** It already holds
   `self._backlog_policy` (`TaskInterceptor._backlog_gate`), the `BacklogPolicy` the recon harness
   seeds via `BacklogPolicy.register_known_project_roots`; β adds a `known_project_roots()`
   accessor, with `fused-memory/src/fused_memory/models/scope.py::known_project_roots_from_env`
   as the fallback (§4.3).
8. **No code reads any task supersession spelling** (confirmed: `git grep` finds
   `x_superseded_by` only in a `tools.py` docstring and as the MEMORY key
   `shared/src/shared/memory_supersession.py::SUPERSEDED_BY_METADATA_KEY`, a string id — same
   spelling, different store, different shape; `x_coalesced_into` is read by
   `scripts/exception_owner_liveness.py` and `scripts/sitting/ownership.py` as a legacy fallback
   behind the deferral `carried_by` record).

## Appendix B. G1/G3/G6 self-check

G1 consumers: `ChildDisposition`/`ChildRow` → β gate, δ pre-flight, ε skill, γ record evidence;
`get_cancel_children` → skill step 1, harness cascade filter, pre-flight probe, Stage 2 prompt;
`superseded_by` → γ skip rule, skill row 1, `/review` phase 3; `cancel_dispositions` → γ, skill,
phase 3; `DECISION_TABLE` → SKILL.md block, Stage 2 render; `CancelChildrenRejection` → harness
re-file; residue record → watcher skill late mode. No producer-orphan.
G3 verified today: `set_status_and_stamp_audit` atomic audit path; `find_tasks_by_metadata`
type-strict str/int and list membership; `get_dependency_edges`; non-reentrant `_write_lock`;
same-status guard ordering; `granted_files` record pattern; `declared_pin_refused` pre-flight
ordering; cascade callback ordering; `register_metadata_submodel(cardinality='list')`;
`render_live_workflow_authority_rules` precedent; Tier-B pin test. Interceptor access to the
known-project-roots registry: verified (Appendix A item 7).
G6 premises: every signal in §7 is a public call or a grep against a file this PRD writes.

## Corrections (2026-11-03 — esc-8142-1, task 8348)

As authored, §4.8, §6 row 26 and the §7 α signal said an `x_`-prefixed alias (`x_superseded_by`)
on a task warns `unknown_key`. `shared/src/shared/task_metadata.py::parse_metadata` never warns
on `x_` keys, so those three sites were corrected in place to: `subsumed_by` warns `unknown_key`; the `x_`
aliases are Tier-C-silent spellings to avoid; `superseded_by` / `cancel_dispositions` do not
warn `unknown_key`. D3 is unchanged. Corrected before ζ (task 8147) executes §6 rows 1–35.

