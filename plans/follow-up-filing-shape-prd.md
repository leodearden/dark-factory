# Follow-up filing shape: bundled follow-ups, a declared-scope class, and a holder gate for unpacked low follow-ups

**Status:** active PRD, authored 2026-11-03/04 (`/prd` under `/team`, fable session team-df-2026241,
from Leo's three rulings of 2026-11-03 on `plans/portfolio-review-2026-11-01/followups/task-shape.md`
options A, G and D, plus five design answers given in-session, recorded in §3; critic pass
`task-shape-seats/PRD1-C.md` folded in). B+H: five packages touched, eight mechanisms, one
load-bearing seam (the submit boundary).
**Code anchors** cite symbols; verified against main `0b9dfd6a9e` (2026-11-03). Main moves fast;
re-locate at implementation time.
**Evidence:** `plans/portfolio-review-2026-11-01/followups/task-shape.md` (§2 to §9) and its seats
`task-shape-seats/S1..S5.md`, `C.md`; design scouts `task-shape-seats/PRD1-A.md`, `PRD1-G.md`,
`PRD1-D.md` (substrate facts F0 to F18, calibration tables, population tables).
**Sibling PRDs to follow:** a curation packing sweep (option B) and dispatch-time riders (option C),
both consumers of this PRD's records (§5).

## 1. Goal

Agent follow-ups stop entering the factory one task per observation into the ready set. An
architect or implementer files its out-of-scope observations as one bundled candidate per area,
carrying the items as structured data; the curator can fold two such candidates filed in one batch
into one task; every filed task carries a machine-stamped declared-scope class that the curator and
the later packing sweep read, and that no verify path may read; and every low-priority follow-up or
reviewer suggestion that nothing cites parks behind a standing per-project holder task, by
dependency, until something cites it, packs it, or a human releases it. The measured premise
(task-shape §3): a low follow-up landing costs the same host verify minutes as any other landing
and carries a third to a half as many lines; low follow-ups are 32% (df), 34% (reify), 70% (sc) and
64% (scp) of landings and 31 / 31 / 63 / 59% of host-local verify minutes.

### What an operator observes when this lands

- A df parent task whose implementer noticed three out-of-scope defects in one module produces ONE
  pending task; `get_task` shows `metadata.followup_items` with three entries, each with a
  server-derived `key`, and the task's architect briefing lists them numbered. Re-filing one of
  those items from any later agent resolves `combined` to that task.
- Three tickets from one parent that reach the curator in one batch, two of them one area, resolve
  as one created task and one ticket `combined` into it; `data/combine_audit.jsonl` carries a row
  with `batch_internal: true` and `items_before` / `items_after`.
- Every task filed after deploy carries `metadata.declared_scope` with a band in
  `{undeclared, narrow, moderate, broad}`; `find_tasks_by_metadata('declared_scope')` lists them; a
  submission that supplies the key itself is refused with a typed error; the curator prompt renders
  the band on every pool entry.
- A newly filed low follow-up shows `dependencies: [<holder id>]` in `get_task` and never appears in
  the scheduler's candidate set; the burndown's `held` series rises by one; `set_task_status(holder,
  'cancelled')` is refused with a typed error naming the holder and the count held.
- `add_dependency(new_task, held_task)` from a pending task leaves `held_task` ready the same tick
  with a `followup_promoted{by: dependent}` event; promoting a held task also promotes every held
  task it depends on; a human `remove_dependency(held, holder)` emits `by: human`.
- The first fused-memory restart after deploy holds the matching stock: df about 1,332 tasks (72%
  of today's 1,838 ready set), sc 26 of 45, scp about 91 of 133, reify 6 of 22, with one
  `followup_hold_backfilled` event per project; a second restart adds none.
- Fourteen days after the integration gate lands, one dated report under
  `plans/portfolio-review-2026-11-01/followups/` gives bundles filed and items per bundle, the held
  population and promotions by cause, and the declared-scope confusion table per project, each
  distinguishing "no producer has written" from zero.

### Consumers (G1)

| mechanism | consumer |
|---|---|
| `followup_items` typed list on a task, server-derived item keys (β) | the architect briefing (`agents/briefing.py`, rendered like delivered checks); item-aware R4 idempotency in the interceptor (γ); the within-batch combine (δ); the packing sweep PRD (merges items on fold) |
| bundling instruction in the follow-up filing prompt (β) | the architect and implementer roles at every dispatch; the curator, which receives bundles as candidates |
| within-batch combine, `_BATCH_CONTRACT` rule 2 lifted (δ) | the ticket worker's topological dispatch (`_process_add_tickets_batch_prepared`), which already executes drop-to-sibling; the combine audit |
| `declared_scope` typed submodel and `derive_declared_scope` (α) | the curator pool render and candidate section, replacing the "tiny < 300 chars or < 3 files" prose rule; the packing sweep PRD; the calibration report (θ) |
| `declared_scope_outcome` event at done (η) | the calibration report (θ); the next task-sizing recalibration (task 4827, deferred) |
| holder task, `followup_hold` record, hold-at-filing, backfill (ε) | the scheduler, through `deps_verdict` (held = unsatisfied dependency, no new dispatch code); the dashboard burndown `held` series (η); the packing sweep and riders PRDs, which drain the held pool |
| `promote(held_id, by)` and its five triggers (ζ) | `/prd` decompose (an edge to a held task promotes it); `docs/quality-findings-contract.md` §8 (an instrument match promotes); the packing sweep and riders PRDs (`by=pack`, `by=rider`); the L2 watcher reading `followup_promoted` events |
| holder protection: typed refusal of cancel / done / deferred on the holder, holder excluded from curator pools (ε) | the cancel-with-children gate before 8147 enforces; the curator |
| integration gate (ι) and observation report (θ) | the operator; θ is the INV-13 proof that every producer above wrote in production |

## 2. Background (substrate as it is, with two corrections to the scouts' sources)

- Follow-ups are filed by the architect and implementer through
  `orchestrator/src/orchestrator/agents/roles.py::_FOLLOWUP_FILING_INSTRUCTIONS`, rendered by
  `roles.py::submit_only_instructions`; the template tells the agent to send `source`,
  `spawn_context: agent_followup`, `spawned_from`, `files`, `escalation_id` and `suggestion_hash`,
  priority low by default, fire-and-forget, one call per observation. Steward and review-cycle
  filers share the helper, not the fragment (PRD1-A F1, F2). Reviewer suggestions are code-filed by
  `orchestrator/src/orchestrator/review_suggestions/curator_intake.py` with
  `spawn_context: review_suggestions` and no `source` (df 368 pending, all low; PRD1-D §1.3).
- `fused-memory/src/fused_memory/middleware/agent_followup_guard.py` does NOT exist; it and
  `followup_site` are task 4719's proposal (pending). Today `suggestion_hash` is whatever string the
  agent sends, unvalidated (PRD1-A F0a). Task 6227 (done) cites the module as if landed; its own
  deliverable, `task_curator.py::_sibling_stream`, is real.
- R4 idempotency is `task_interceptor.py::TaskInterceptor._check_escalation_idempotency`: exact
  match on top-level `escalation_id` and `suggestion_hash` against every non-cancelled task, per
  ticket, before curation. A key absorbed by a combine is unreachable because the candidate's
  metadata is never written (F3, F8).
- Batches: `TaskInterceptor._curator_worker` drains up to `CuratorConfig.batch_max` (5) tickets under
  `batch_token_threshold` (50,000); `_process_add_tickets_batch_prepared` runs R4 per ticket, then
  `curate_batch_prepared`, then a topologically ordered dispatch keyed on `batch_target_index`; a
  within-batch `drop` waits for its sibling's task id. The topology needed for "combine A into new B"
  exists (F4, F5). `task_curator.py::_BATCH_CONTRACT` rule 2 forbids it; `_parse_decision_dict`
  degrades such a decision to create with `missing-target` (F6, F7).
- Combine (`TaskInterceptor._execute_combine`) rewrites the target's title, description, details
  (with a `FILES_TO_MODIFY:` block) and priority, and merges exactly `{curator_action,
  curator_justification, combined_at}` into metadata. Since task 3446 it no longer replaces
  `metadata.files`; it leaves the target's list stale (F0b). Task 3260's live scope is the
  owned-field redesign of `files`; this PRD never writes `metadata.files` from a combine.
- `metadata.files` is rewritten after filing by `orchestrator/src/orchestrator/scheduler.py::_persist_files_metadata`
  (every plan refinement) and by the workflow's done-time reconcile from the merge diff. The
  author's own list therefore survives nowhere; measured against the first dispatch's lock
  acquisition it predicts non-test lines ≤ 20 with precision 0.31 (df) / 0.36 (reify), and 18% /
  27% of 1–2-file declarations land over 300 non-test lines; the ordering by band is monotone
  (df median non-test lines 78 → 158 → 578 across 1–2 / 3–5 / 6–12 declared files) (PRD1-G §2).
- Submit guards run in `fused-memory/src/fused_memory/server/tools.py::submit_task` before the
  interceptor; three already stamp metadata (`inject_task_kind`, `inject_execution_class`,
  `inject_operational_routing`); the last has a twin inside `TaskInterceptor.submit_task` before the
  `planning_mode` split, which also refuses a caller-supplied `deferral` key with
  `DeferralWriteAuthorityError` (PRD1-G §1.1). Typed metadata submodels register through
  `shared/src/shared/task_metadata.py::register_metadata_submodel`, list cardinality on the
  `delivered_checks` / `defects` precedent; a registered key needs no Tier-A blessing (PRD1-A F12).
- Readiness: `scheduler.py::Scheduler._eligible_for_dispatch` → `dependency_verdict.py::deps_verdict`;
  a local dependency is satisfied only when `done` or `cancelled` (or merge-deferred in the same
  train); pending-forever is unsatisfied. `sqlite_task_backend.py::add_dependency` checks existence
  and self-loop only. A `submit_task` caller may pass `dependencies`, inserted in the same write unit
  as the row (PRD1-D §1.1). `Scheduler._deferred_watch_gated` keeps a task carrying a truthy
  `metadata.deferred_watch` out of dispatch; reify Ω 9739 is the precedent for a deterministic
  always-escalates holder with thousands of dependents.
- Cancelling a task with dependents: `reconciliation/targeted.py::TargetedReconciler._sweep_cancelled_descendants`
  files one L1 per live descendant; the cancel-with-children gate (8143) stays in observe mode until
  8147 flips enforcement.
- Nothing pages on a permanently dependency-locked pending task: `starvation_watchdog.py` runs over
  dispatch candidates only (PRD1-D §1.2).
- Population under decision 9's rule (critic PRD1-C finding 1, stores read 2026-11-03; ready =
  pending, no unmet local dependency, no `deferred_watch`, no `milestone`, not deterministic; scp
  derived from PRD1-D because its store is not on this host): df 1,012 follow-ups match of which
  967 are ready (136 simple) plus 370 reviewer suggestions of which 365 are ready, so 1,332 newly
  held, 72% of a 1,838 ready set; reify 1,025 + 78 match, 6 newly held of 22 ready (742 wait on Ω
  alone); sc 42 + 7 match, 26 of 45 ready; scp 133 + 20 match, about 91 of 133 ready. PRD1-D's
  earlier 830 / 45% was the narrower rule (simple excluded, reviewer suggestions excluded). Flow by
  id band since 10-02: df about 940 low follow-ups plus reviewer suggestions a month; reify about
  570.

## 3. Resolved design decisions

Leo, 2026-11-03, on the task-shape options: A (bundle at filing; lift batch rule 2) yes; G (size
class at filing, gating bundling only, never a verify) yes; D (holder gate per R3(a)) yes. In-session
answers: **backfill the stock** at deploy, first ruled against the narrow rule's 830 df tasks and
**re-confirmed on 2026-11-04 against the corrected 1,332 (72% of df's ready set)**; **hold scope
covers both** reviewer suggestions and `complexity=simple` follow-ups; **no age-out**; **item keys
are server-derived and 4719 is subsumed**.

Lead's decisions under those rulings:

1. **Bundles are structured data merged deterministically; the LLM rewrites prose only.** A new
   registered list submodel `metadata.followup_items` (§4.1). Every writer (create, combine,
   within-batch combine, drop-to-sibling) goes through one `merge_items` (heuristics 10, 11, 12).
   Items never reach the architect only through curator prose: the briefing renders the list.
2. **No new curator action.** A within-batch combine is `action='combine'` with
   `batch_target_index` and a null `target_id`, executed through the existing `_execute_combine`
   once the sibling has an id, so the gate guard and eligibility checks apply unchanged. The
   batch-edge target must itself be `create`; anything else degrades, audited.
3. **Every R4 key is server-derived at the submit boundary; nothing the agent sends is trusted.**
   `origin_task` is `spawned_from`; item `key = sha256_16(origin_task, sorted files, normalised
   summary)`; the top-level `escalation_id` is derived from `spawned_from`
   (`agent-followup-<parent>`); the top-level `suggestion_hash` is the sha256_16 of the sorted item
   keys. A caller-supplied value for any of the three is overwritten. That is the whole of task
   4719's remaining scope (server-side derivation of the top-level pair), so 4719 is closed with a
   pointer at decompose; its `followup_site {path, symbol}` shape is not adopted, files plus summary
   stand in for it. A candidate with no `spawned_from` gets `origin_task = 'unparented'` on its
   items and no top-level `escalation_id`, so it can never satisfy R4's top-level conjunction and
   dedupes through its item keys only.
4. **`MAX_FOLLOWUP_ITEMS = 5`**, matching the curator's planning-budget rule (≤ 5 concerns). A
   bundle must fit one combine budget or it is two bundles.
5. **Bundling applies to the architect and implementer filers in v1.** Steward and review-cycle
   filers keep their templates; a lone observation is a bundle of one, so there is one code path.
6. **The size class is named for what it measures: `declared_scope`, bands
   `undeclared | narrow | moderate | broad`** (§4.2). "Tiny" is never a name or a reading: two in
   three pre-run "tiny" calls are wrong (PRD1-G §2.3). Thresholds live in one pure function in
   `shared/`; the `≥ 3 files` boundary matches the existing `large-plan-modules` routing rule.
7. **Stamped once at filing, immutable, machine-only.** Stamped in `TaskInterceptor.submit_task`
   before the planning_mode split (one site covers tools.py callers, direct callers, the curator path
   and planning batches; the stamped metadata is serialised into the ticket blob and a curator
   `create` never applies `rewritten_task`, so the stamp matches the row). The write-authority
   refusal lives in the interceptor's public `submit_task` and `update_task` paths, not in the
   backend floor where `DeferralWriteAuthorityError` lives: `_execute_combine` writes through the
   backend's `update_task` directly, and a floor-level refusal would block the one sanctioned
   re-stamp. The interceptor floor has both arms the backend's deferral floor has: an altered
   value is refused, and a replace-mode `update_task` that omits the key is refused too, so no
   caller can erase a stamp by replacing the whole metadata. Not re-derived at plan refinement or at done, because those rewrite `files` precisely
   when the declaration was wrong and re-deriving would erase what the class measures. The only
   re-stamp is a curator combine (`basis='combine'`), which changes the charter.
8. **Forbidden readers are enforced by a structural test, not prose.** A test pins the set of modules
   allowed to import `shared.declared_scope`; verify admission, verify scope, the merge-lane disjoint
   skip, the verify-result cache, `agents/triage.py::is_declared_simple_task` and
   `routing.py` are outside it (heuristic 10).
9. **Hold rule.** A created task is held iff (`metadata.source == 'agent-followup'` or
   `metadata.spawn_context == 'review_suggestions'`) and priority in `{low, polish}` and
   `task_kind` is `'normal'` or absent (an absent key reads as normal; reviewer suggestions are
   code-filed without one) and no `execution_class` and no `prd_path` and no `x_finding_key`.
   Simple follow-ups are held (same verify toll). A planning-mode birth is never held (a PRD
   decompose is a citation by construction).
10. **Holder shape: deterministic pure gate with `deferred_watch`.** One per project, minted
    idempotently on first need: `task_kind='deterministic'`, `always_escalates=true`, no
    `before_done`, no dependencies, `execution_class='decision'`, priority low,
    `metadata.deferred_watch` set by the mint helper, and a registered marker submodel
    `metadata.followup_holder`. The store is the single home of which task is the holder (lookup by
    marker, cached per project; a second live marker is refused at mint). If the watch gate ever
    lapses the failure is a loud born-at-L2 gate, never an agent run.
11. **The hold is born with the task.** The holder id is appended to `dependencies` in
    `_dispatch_ticket_decision`'s create branch before `tm.add_task`, so edge and row are one write
    unit and the scheduler never sees the follow-up as ready; a `followup_hold` record is stamped and
    a `followup_held` event emitted.
12. **Backfill runs once per project at fused-memory server startup after deploy** (the process is
    `fused-memory/src/fused_memory/server/main.py`, after the interceptor's `start()` and before
    the server begins serving, over the project list `known_project_roots_from_env` yields, falling
    back to the server's own project root when that list is empty; never in a `TaskInterceptor`
    constructor), idempotently,
    over every pending task matching the hold rule. It writes backend-direct under the interceptor's
    `_write_lock` (one edge insert and one `followup_hold` stamp per task, skipping the journal and
    the per-call `task_modified` event the interceptor's public `add_dependency` / `update_task`
    would emit, which for df would be about 2,664 events) and emits one
    `followup_hold_backfilled{project, count}` event per project (`storm-escape-required`). A task
    already carrying the holder edge is skipped, so a second start adds zero. A scheduler tick can
    observe a partial backfill through `get_tasks` (no read lock); harmless, those tasks were ready
    before deploy. No human gate: Leo ruled backfill on the corrected count.
13. **Holder protection is in this PRD, not deferred to cancel-with-children.** `set_task_status`
    refuses `done`, `cancelled` and `deferred` on a holder with a typed error unless an explicit
    `release_holder=True` is passed (a whole-backlog release is a human act); `update_task` refuses
    stripping the marker or `deferred_watch` and refuses setting `trigger_met` on the holder
    (`_deferred_watch_gated` is `bool(deferred_watch) and not trigger_met`); the holder is excluded
    from every curator pool stream by returning no entry from `task_curator.py::_to_pool_entry`
    (one site covers all four streams) and from `_execute_combine` eligibility by an `is_holder`
    conjunct. Reason: `cancelled` counts as satisfied, so one cancel would release ~1,332 df tasks
    and file ~1,332 L1s through `_sweep_cancelled_descendants`.
14. **Promotion is one function with an enumerated cause, and it is a closure.** `promote(held_id,
    by, ref)` deletes the holder edge, stamps `followup_hold.promoted`, emits
    `followup_promoted{task_id, by, ref}`, `by ∈ {dependent, citation, combine, pack, rider, human}`,
    and then promotes every held task the promoted task itself depends on (`by=dependent`,
    `ref=<the promoted id>`), so a cited task can never wait silently on a still-held dependency (the
    silent-jam class Leo's 2026-09-30 ruling forbids; nothing pages a pending task with an unmet
    dependency). `pending_since` is NOT re-stamped: a hold is a machine-imposed wait of exactly the
    class Leo's 2026-08-06 continuity ruling protects, `stamp_pending_since`'s transition table has
    no `pending → pending` row, and a task waiting weeks on an ordinary dependency already scores as
    weeks old when it lands, so a promoted task is treated the same. Triggers: a dependent edge
    inserted from a pending, non-held task; the interceptor's status chokepoint
    `TaskInterceptor._apply_status_transition` (it serves the comma-separated loop and the
    single-id path, hence `commit_planning`, a plain `set_task_status`, and the deferral-expiry
    sweep the deferral PRD routes through an interceptor `client_op_id`), for every
    `deferred → pending` landing, promoting the held dependencies of the task that just became
    pending;
    `update_task` adding `x_finding_key` or `prd_path`; `_execute_combine` absorbing a candidate
    outside the hold scope into a held target; the packing sweep and riders PRDs calling it; a human
    `remove_dependency(held, holder)` detected in the interceptor and recorded as `by=human`, and an
    `update_task` replacement `dependencies` list that drops the holder edge, recorded as
    `by=human, ref='update_task'`. A parent edge between two held follow-ups (task 6072's fix)
    never promotes on insert; it promotes through the closure when the parent is promoted.
15. **No age-out.** Held means parked. The drains are the packing sweep PRD and the riders PRD
    (together they cover a minority of the held pool: packing about 24% of df's and 35% of reify's
    before the judge, riders about 29% and 36% of those with files); θ reports the held population
    at 14 days so the question can be reopened with numbers. G7 waiver recorded in §6 (ε).
16. **Task 6072 stays independent.** Its parent edge and this PRD's holder edge compose (AND); the
    parent edge clears at land, the holder edge at promotion.
17. **Observation is one leaf for the three mechanisms**, reader-shaped, and must render "no
    producer has written" as a state distinct from zero for each series (INV-13).

## 4. Contract (B+H)

### 4.1 Follow-up items (`shared/src/shared/followup_items.py`, registers itself)

```python
FOLLOWUP_ITEMS_METADATA_KEY = 'followup_items'
MAX_FOLLOWUP_ITEMS = 5

class FollowupItem(BaseModel):            # frozen, extra='forbid' (defects._RECORD_CONFIG idiom)
    key: str                              # sha256_16 of (origin_task, sorted files, normalised summary)
    summary: str                          # one standalone line, <= 300 chars
    files: tuple[str, ...]                # non-empty; this item's declared footprint, repo-relative files
    origin_task: str                      # the task that observed it; survives cross-parent combines
    origin_escalation_id: str | None = None

def item_key(*, origin_task: str, files: Sequence[str], summary: str) -> str
def require_unique_item_keys(items: Sequence[FollowupItem]) -> None      # raises DuplicateFollowupItemKey
def merge_items(base: Sequence[FollowupItem], extra: Sequence[FollowupItem]) -> tuple[FollowupItem, ...]
    # order-preserving union by key; the ONE merge used by create, combine, within-batch combine, drop-to-sibling
def legacy_item(*, title: str, files: Sequence[str], metadata: Mapping[str, object]) -> FollowupItem | None
    # a pre-bundle single follow-up (top-level escalation_id/suggestion_hash) as one item, key = suggestion_hash

register_metadata_submodel(FOLLOWUP_ITEMS_METADATA_KEY, FollowupItem, cardinality='list')
```

Invariants, enforced at the submit boundary and inside `merge_items`: keys unique; `1 ≤ len ≤
MAX_FOLLOWUP_ITEMS` on create; the union of item `files` ⊆ `metadata.files` on create (a combine
leaves `metadata.files` to its owner, task 3260). `origin_task`, every item `key`, the top-level
`escalation_id` and the top-level `suggestion_hash` are derived server-side (decision 3) and any
caller-supplied value is replaced. `CandidateTask.items` enters `payload_hash` so a cached curator
decision is never served for a bundle whose items changed.

Item-aware R4 (`_check_escalation_idempotency`; "live" means non-terminal, so `done` and
`cancelled` tasks never match, where today only `cancelled` is skipped): (1) the top-level match is
unchanged, a conjunction of `escalation_id` and `suggestion_hash`; (2) a candidate's items whose
key equals a live task's `suggestion_hash` or any live `followup_items[].key` are dropped, and if
nothing remains the ticket resolves `idempotency_hit`; (3) a legacy single whose `suggestion_hash`
equals a live item key is a hit. A bundle's own implementer re-filing an unfinished item has
`origin_task = the bundle`, so its key differs from the bundle's item (origin = the parent) and it
files as a new task; no self-match rule is needed.

Combine audit rows gain `candidate_ticket_id, candidate_source, candidate_spawned_from,
batch_internal, items_before, items_after`; an old row without them stays parseable. Within-batch
drop-to-sibling writes an audit row too (today it writes none).

### 4.2 Declared scope (`shared/src/shared/declared_scope.py`, registers itself)

```python
class DeclaredScope(BaseModel):           # frozen, extra='forbid'
    band: Literal['undeclared', 'narrow', 'moderate', 'broad']
    files: int                            # len(metadata.files) at stamping
    text_chars: int                       # len(description) + len(details) at stamping
    basis: Literal['submit', 'combine']
    rule: Literal['v1']

def derive_declared_scope(files: Sequence[str], description: str, details: str, *, basis) -> DeclaredScope
    # v1: undeclared = 0 files; narrow = 1-2 files and text_chars < 6000;
    #     moderate = 3-5 files, or 1-2 files with text_chars >= 6000; broad = >= 6 files

register_metadata_submodel('declared_scope', DeclaredScope, cardinality='dict')
class DeclaredScopeWriteAuthorityError(TaskBackendError)   # beside DeferralWriteAuthorityError
```

Stamping: `TaskInterceptor.submit_task` before the planning_mode split; refuse a submission or
`update_task` carrying the key; re-stamp only in `_execute_combine` with `basis='combine'` from the
rewritten task's files and text. Readers allowed: `task_curator.py` (pool render line, candidate
section, the prose tiny rule replaced), the packing sweep PRD, θ. The allowlist test
`shared/tests/test_declared_scope_readers.py` greps importers of `shared.declared_scope` and fails on
any module outside the allowlist. Absent key (pre-deploy tasks) means "no information", never
`undeclared`.

### 4.3 Holder and hold (`fused-memory/src/fused_memory/middleware/followup_hold.py`)

```python
class FollowupHolder(BaseModel):          # metadata.followup_holder on the holder task
    project_id: str
    minted_at: str                        # ISO-8601 UTC

class FollowupHold(BaseModel):            # metadata.followup_hold on a held task
    holder: int
    held_at: str
    promoted: PromotionRecord | None = None

class PromotionRecord(BaseModel):
    by: Literal['dependent', 'citation', 'combine', 'pack', 'rider', 'human']
    at: str
    ref: str | None = None                # the dependent id, finding key, combining ticket, sweep run id

def hold_scope(metadata: Mapping[str, object], priority: str, task_kind: str) -> bool     # decision 9, pure
def ensure_holder(tm, project_root) -> int      # idempotent mint; refuses a second live marker
def is_holder(task) -> bool
def hold_at_create(kwargs, holder_id) -> None   # appends the edge + stamps followup_hold before tm.add_task
def backfill(tm, project_root) -> int           # backend-direct under _write_lock; one event per project
def promote(tm, project_root, held_id: int, *, by, ref=None) -> tuple[int, ...]
    # removes the holder edge, stamps, emits, then recurses into the held dependencies of held_id
    # (by='dependent', ref=held_id); returns every id promoted
def promote_dependencies_of(tm, project_root, task_id: int) -> tuple[int, ...]
    # the status-chokepoint and edge-insert entry: promotes held deps of a task that is pending
class FollowupHolderProtectedError(TaskBackendError)
```

Events (runs-ledger shape, emitted through the interceptor's existing event path): `followup_held
{task_id, holder_id}`, `followup_hold_backfilled {project_id, count}`, `followup_promoted {task_id,
by, ref}`, `followup_holder_protected {holder_id, attempted_status, held_count}`.

Invariants: exactly one live holder per project; a held task has exactly one holder edge and a
`followup_hold` record, and the two agree; the holder never has dependencies, never dispatches, is
never a curator pool entry or combine target, and leaves `pending` only through
`release_holder=True`; promotion is the only writer that removes a holder edge except a human
`remove_dependency`, which is recorded as `by=human`.

### 4.4 Boundary-test sketch (the integration gate ι's signal; every scenario drives the public
interfaces `submit_task`, `resolve_ticket`, `get_task`, `add_dependency`, `remove_dependency`,
`set_task_status`, `commit_planning`, and the scheduler's candidate enumeration, never internals)

| # | scenario | preconditions | postconditions |
|---|---|---|---|
| 1 | an implementer files a bundle of three items in one area | parent in progress; curator pool empty | one task created; `followup_items` has three entries with derived keys; `declared_scope.basis == 'submit'`; `dependencies == [holder]`; `followup_held` event |
| 2 | a later agent re-files one of those items | scenario 1 | ticket resolves `combined` to the bundle with `reason idempotency_hit`; no new task |
| 3 | the bundle's own implementer re-files an unfinished item | scenario 1, candidate `spawned_from == bundle id` | a new task is created (its derived key differs because `origin_task` differs) and is held |
| 3b | an item of a bundle whose task is `done` is re-observed | bundle done, item unfixed | a new task is created (terminal tasks never match R4) |
| 4 | three same-parent tickets in one worker batch, two combinable | worker drains them together | one task created, one ticket `combined`, one created; the combined task's items are the union; audit row `batch_internal: true`; `declared_scope.basis == 'combine'` on the survivor |
| 5 | a combine whose batch target is itself a drop | curator emits combine→B, B is drop→C | A degrades to create with `batch-target-not-create` in the audit; nothing lost |
| 6 | a submission supplies `declared_scope` | any | refused with `DeclaredScopeWriteAuthorityError`; nothing filed |
| 7 | a medium follow-up, a PRD leaf (planning_mode), and a task with `x_finding_key` are filed | holder exists | none of the three carries the holder edge |
| 8 | `add_dependency(new_task, held)` from a pending task | scenario 1 | `held` is ready next tick; `followup_hold.promoted.by == 'dependent'`; `pending_since` unchanged; event emitted |
| 9 | `add_dependency(deferred_batch_task, held)` then the batch task reaches pending by `commit_planning`, by a plain `set_task_status`, or by the deferral-flip path | planning batch | not promoted at the edge; promoted at the flip by every one of the three paths, `by == 'dependent'` |
| 9b | X is held, X depends on held H, X is promoted by citation | both held; X has no other unmet dependency | H is promoted too, `by == 'dependent'`, `ref == X`; X is a dispatch candidate next tick |
| 10 | a parent edge is added between two held follow-ups (6072 shape) | both held | neither promoted on insert |
| 11 | `update_task(held, metadata gains x_finding_key)` | scenario 1 | promoted `by == 'citation'`, `ref == the key` |
| 12 | a medium candidate combines into a held target | held target pending | target promoted `by == 'combine'`; items merged |
| 13 | a human `remove_dependency(held, holder)` | scenario 1 | task ready; `promoted.by == 'human'`; event emitted |
| 14 | `set_task_status(holder, 'cancelled')` without `release_holder` | 1,332 held | refused with `FollowupHolderProtectedError{held_count: 1332}`; no L1 filed; event emitted |
| 14b | `update_task(holder, metadata.trigger_met = true)` | holder exists | refused with the same error; the holder stays gated |
| 15 | `set_task_status(holder, 'cancelled', release_holder=True)` | 3 held | holder cancelled; the three are ready; three `followup_promoted{by: human}` events |
| 16 | the curator is handed a pool containing the holder | holder exists, anchor is a held task | the holder appears in no pool stream and is never a combine target |
| 17 | fused-memory restarts after deploy with 1,332 matching pending tasks and no holder edges yet | backfill enabled | 1,332 edges added; one `followup_hold_backfilled{count: 1332}` event and no per-task events; second restart adds 0 |
| 18 | the scheduler enumerates candidates | scenarios 1, 8 | the held task is absent before promotion and present after; the holder is absent always (`deferred_watch`) |
| 19 | a held task lands via the packing sweep PRD's `promote(by='pack')` | future | out of this gate; the sweep PRD's gate covers it |
| 20 | a task filed with no `files` | any | `declared_scope.band == 'undeclared'`; the curator render shows it |

## 5. Cross-PRD relationship and seams (G4)

| other PRD / document | direction | seam mechanism | owner | status |
|---|---|---|---|---|
| `plans/cancel-with-children-prd.md` (8142 in progress, 8143–8148 pending) | consumes | holder cancel refusal before 8147 enforces; a `repoint` / `fold` of a held child of a cancelled parent moves only the parent edge and leaves the holder edge; `keep` leaves it held | this PRD owns the refusal (ε); that PRD owns dispositions; ζ adds one sentence to its §4.2 seam list as a companion correction | queued |
| `docs/quality-findings-contract.md` §8 step 1a | this PRD edits | a step-1/step-2 match on a held task appends `x_finding_key` and promotes `by=citation` | this PRD (ζ) | queued |
| the packing sweep PRD (option B, next) | produces for | `followup_hold` + `declared_scope.band` as the pack candidate pool; `merge_items` on fold; `promote(by='pack')` | that PRD owns the packer; this PRD owns the records it reads | to author |
| the riders PRD (option C) | produces for | `promote(by='rider')`; held pool ∩ host lock set | that PRD | to author |
| reify alpha-spine Ω 9739 | coexists | double edges (Ω + holder) are AND, harmless; Ω's resolve offers "re-park behind the next holder": the follow-up holder is the destination for the follow-up share | reify spine PRD | wired |
| task 6072 (parent edge for unmerged parents) | coexists | parent edge and holder edge compose; a parent edge never promotes | 6072 | pending |
| task 3260 (`metadata.files` owned field) | coexists | this PRD never writes `metadata.files` from a combine; 3260 later derives `files ⊇ ∪ item files` | 3260 | pending |
| task 4719 (server-derived R4 keys) | subsumed | `item_key` is the derivation at item grain | this PRD (β); 4719 closed with a pointer at decompose | companion correction |
| task 6707 (S11 ε one curation pipeline; blocked, deps done, `unblock_auto` unresolved) and 6708 | file conflict | δ edits `_BATCH_CONTRACT`, `_parse_*` and the dispatch arm that 6707 restructures | no edge: an edge to a blocked task is a hold with no owner and no bound (INV-7). The per-module lock serialises the two if both are live; δ's description says "if 6707 lands first, re-locate the dispatch arm in `curate_prepared`", and a rebase is the ordinary cost | none needed |
| task 6322 (re-dedupe sweep) | coexists | folding a held task into a non-held one is `promote(by='combine')` | 6322 | pending |
| task 5197 (backlog coalesce sweep, done) | precedent | `source: backlog-sweep-coalesce` is the packer's shape | packing sweep PRD | n/a |
| task 4827 (task sizing recalibration, deferred) | produces for | `declared_scope_outcome` events are its prospective evidence | 4827 | n/a |

No reciprocal-ownership statements found.

## 6. Decomposition plan

Overlay bands: 300–1,500 LOC and ≤ 10–12 files per leaf. Labels are Greek; ids are assigned at
decompose. Same-file leaves are serialized by edges (all of β, γ, δ, ε, ζ touch
`task_interceptor.py`).

| label | title | modules | observable signal | prereqs |
|---|---|---|---|---|
| α | Declared-scope class stamped at the submit boundary, immutable, with a reader allowlist | `shared/src/shared/declared_scope.py` (new), `task_metadata.py` (register), `task_backend_errors.py`, `task_interceptor.py` (stamp before the split; authority refusal on submit and update), `shared/tests/test_declared_scope.py`, `test_declared_scope_readers.py`, `fused-memory/tests/test_declared_scope_stamp.py`, `docs/task-authoring.md` §8 | sketch rows 6 and 20 through `submit_task` / `get_task`; `find_tasks_by_metadata('declared_scope')` lists every post-deploy task; the allowlist test fails when a forbidden module imports the band | — |
| β | Follow-up items contract, server-derived keys, bundling filing prompt, briefing render | `shared/src/shared/followup_items.py` (new), `shared/tests/test_followup_items.py`, `task_interceptor.py` (key fill and item invariants at submit), `fused-memory/tests/test_task_metadata_boundary.py`, `agents/roles.py` (`_FOLLOWUP_FILING_INSTRUCTIONS`), `agents/briefing.py` (`_format_followup_items`), orchestrator role and briefing tests, `docs/task-authoring.md` §8 | sketch row 1 minus the hold: an architect dispatched on a parent with ≥ 2 observations in one area yields ONE task whose `followup_items` has ≥ 2 entries with derived keys, and its briefing shows the numbered list | α |
| γ | Item-preserving combine, item-aware R4, combine re-stamp of declared scope, audit fields, holder excluded from curator pools and combine targets | `task_interceptor.py` (`_execute_combine`, `_check_escalation_idempotency`, `_append_combine_audit`), `curator_types.py` (`CandidateTask.items`, `PoolEntry.item_count`, `declared_scope` line, `is_holder` conjunct), `task_curator.py` (`_to_pool_entry` returns no entry for a holder; render; replace the tiny prose rule by the band), tests `test_task_interceptor_curator.py`, `test_task_interceptor.py`, `test_task_curator.py` goldens, `test_curator_types_leaf.py` | sketch rows 2, 3, 3b, 16 and the combine half of 12 (items merged); audit row carries `items_before/after` and `candidate_source`; the curator prompt renders `declared_scope` | β |
| δ | Lift batch rule 2: within-batch combine | `task_curator.py` (`_BATCH_CONTRACT`, `_parse_decision_dict`, `_parse_batch_decisions`, harness version `curator-v2`), `task_interceptor.py` (topological dispatch arm), tests `test_task_curator.py` (`TestParseBatchDecisions`, decision-dict group, goldens), `test_ticket_worker.py` combine twins, `test_task_interceptor.py` refused-sibling twin. Description carries: "no edge to 6707; if 6707 lands first, re-locate the dispatch arm in `curate_prepared`" | sketch rows 4 and 5 | γ |
| ε | Holder task, hold at filing, backfill at server startup, holder protection. **G7 waiver: holds-owned-and-bounded — Leo ruled no age-out on 2026-11-03; the owner is the holder task (machine-readable), the bound is supervision: the `held` series (η), the θ report at 14 days, and the packing sweep and riders PRDs as the drains, neither of which bounds the pool alone.** Filed with `metadata.g7_waivers` | `fused-memory/src/fused_memory/middleware/followup_hold.py` (new), `task_interceptor.py` (create branch, planning-mode create, `set_task_status` / `update_task` refusals incl. `trigger_met`), `fused-memory/src/fused_memory/server/main.py` (run `backfill` per known project at startup), `fused-memory/config/schema.py` (`followup_hold.backfill_enabled`), `task_metadata.py` (register `followup_holder`, `followup_hold`), `docs/task-authoring.md` §3 and §8, tests (3 files) | sketch rows 7, 14, 14b, 15, 17, 18 (held absent) | γ |
| ζ | Promotion closure and its triggers; findings-contract and skill edits | `followup_hold.py::promote`, `::promote_dependencies_of`, `task_interceptor.py` (`add_dependency`, `update_task` incl. replacement-list detection, `_execute_combine` promote-on-combine, `remove_dependency` cause, and `_apply_status_transition` for every `deferred → pending` landing), `docs/quality-findings-contract.md` §8, `skills/prd/references/decompose-mode.md` (an edge to a held task promotes it), `plans/cancel-with-children-prd.md` §4.2 one-sentence seam, tests | sketch rows 8, 9, 9b, 10, 11, 12 (promotion half), 13, 18 (present after) | ε |
| η | Done-time `declared_scope_outcome` event and the burndown `held` series | `orchestrator/src/orchestrator/workflow.py` (done-time reconcile emits `{band, files_at_stamp, files_post_plan, files_landed, lines_total, lines_nontest}`), `dashboard/src/dashboard/data/burndown.py::collect_snapshot` (`held` beside `pending`, nullable; renders "no holder minted" distinct from 0), dashboard project tile, tests | a landed task's run shows one `declared_scope_outcome` event; the burndown row for a project with a holder carries `held = N`, for a project without one `held = null` | α, ε |
| ι | Integration gate: the boundary-test sketch end to end | `fused-memory/tests/integration/test_followup_filing_shape.py` (new; drives public interfaces only), one orchestrator scheduler-candidate test | every §4.4 row except 19 passes against the integrated tree; any row failing is a RED gate | δ, ζ, η |
| θ | Observation report, delayed milestone 14 days after ι | `scripts/followup_filing_shape_report.py` (new, read-only over tasks.db, runs.db, combine_audit.jsonl), its test, one dated report under `plans/portfolio-review-2026-11-01/followups/` | the report prints, per project: bundles filed and items per bundle (p50 / p90), bundle landings and verifies per landing vs the S5 baseline (df 230 low-follow-up landings a month), held population (baseline: the backfill counts in §2) and promotions by cause, the declared-scope confusion table per band; each series marked "producer has not written" when its event count is zero | ι (`metadata.milestone` delayed 14 d) |

Companion corrections at decompose: close task 4719 with a pointer to β; add a note to 6072 naming
the composition rule (decision 16); add a note to 3260 naming γ as a non-writer of `files`.

G7 walk (walked in author mode; blocking at decompose, re-walked there): `holds-owned-and-bounded`
fires on ε by design and is waived in ε's row above and stamped on the filed task. The δ → 6707
edge was an unrecorded second hit and was removed (§5). `readers-prove-their-producer` binds η and
θ (decision 17). `storm-escape-required` binds the backfill (one event per project, backend-direct,
decision 12). `contracts-machine-checked` is met by the three typed submodels and the `by` enum.
`one-fact-one-home`: holder identity lives in the store marker only; thresholds in one function;
R4 keys derived in one place. `no-silent-fail-soft`: every degraded curator decision is audited
(existing) and the new `batch-target-not-create` arm writes a row. `corroborate-before-acting`:
promotion reads the live row under the write lock. `guards-exercise-behaviour`: the reader
allowlist test greps real imports.

## 7. Pre-conditions

- No novel substrate: every mechanism extends an existing site named in §2 (G3 verified against
  main `0b9dfd6a9e`, symbol by symbol, 2026-11-03).
- δ carries no edge to blocked task 6707; the module lock serialises them and δ's description names
  the rebase (§5).
- The backfill runs at the first fused-memory server start after ε deploys; the fleet redeploy sweep
  is that restart. No operator action.

## 8. Out of scope

- The packing sweep (option B) and dispatch-time riders (option C): separate PRDs, consumers of
  this one.
- An age-out for held tasks (Leo: no). The review's R3(d) 30-day cancel sweep is not filed.
- Per-item closure ("items 1–2 done, item 3 refiled"): deferred until θ reports how often a bundle
  blocks on one item; if added, reuse `shared/cancel_dispositions.py`'s `fold` shape (8142).
- Bundling for steward and review-cycle filers (decision 5).
- Any reader of `declared_scope` in verify admission, verify scope, the merge lane, the verify
  cache, simple-path triage or model routing (decision 8; the allowlist test enforces it).
- Changing `metadata.files` ownership (task 3260).

## 9. Open questions (tactical)

1. **Backfill ordering with reify Ω.** 742 reify follow-ups already depend on Ω; adding the holder
   edge is harmless but doubles the edge count the dashboard renders. Decide in ε whether to skip
   tasks that already carry an unmet dependency (the hold is then a no-op for them anyway).
2. **`update_task` replacement lists.** Decision 14 records a stripped holder edge as
   `by=human, ref='update_task'`; ζ decides whether the replacement list that strips it must also
   be rejected when the caller is an agent role (it is a release by side effect).
3. **6000-character text term in `narrow`.** The data supports text length at the large end only
   (PRD1-G §2.3). Keep v1 as specified; θ's confusion table decides whether v2 drops it.
4. **`undeclared` as its own band** (0 files lands median 40 non-test lines but 22% over 300). Kept in
   v1; θ reports it separately.
5. **Within-batch combine when the survivor is also an R4 hit.** Order R4 before curation (as today)
   so a hit resolves before any batch edge exists. Confirm in δ's tests.
6. **Human correction of a mis-called band.** Refused in v1 (immutable); if operators need it, add
   `basis='human'` in a later rule version rather than editing in place.

## Appendix A. G1 / G3 / G6 self-check

- G1: every mechanism in §1's table names a wired consumer on main or a leaf in this batch; the two
  unfiled PRDs (packing, riders) are named as additional consumers, never as the only one.
- G3: no novel substrate; the two corrections in §2 (no `agent_followup_guard.py`; combine does not
  destroy `files`) were applied before design.
- G6: the signals assert no numeric bound. Rejection-style signals (rows 6, 14, 14b, 16, 18-before)
  name the mechanism that fires: `DeclaredScopeWriteAuthorityError` (α, interceptor-level),
  `FollowupHolderProtectedError` (ε), pool exclusion (γ), `deps_verdict` unsatisfied (existing,
  wired at `Scheduler._eligible_for_dispatch`). Reader-shaped leaves (η, θ) render the no-producer
  state.

## 10. Decompose record (2026-11-04)

Decomposed by the PRD1 decompose seat (`/prd` decompose mode under `/team`) against main `55b50a6212`.
Nine leaves were filed in planning mode, eleven intra-batch edges were wired exactly as §6, and the batch
was flipped in one `commit_planning`. There are no out-of-batch edges and none to 6707.

| label | task | prereqs | priority |
|---|---|---|---|
| α | 8413 | — | high |
| β | 8414 | α | high |
| γ | 8415 | β | high |
| δ | 8416 | γ | medium |
| ε | 8417 | γ | high (`metadata.g7_waivers`: holds-owned-and-bounded) |
| ζ | 8418 | ε | high |
| η | 8419 | α, ε | medium |
| ι | 8420 | δ, ζ, η | medium (signal leaf) |
| θ | 8421 | ι | low (`milestone` delayed 1,209,600 s; anchors on ι going done) |

Capability manifest: `plans/follow-up-filing-shape-prd.capability-manifest.md` and its sidecar `.yaml`.
The sidecar was stamped by `commit_planning` with all nine labels, no missing labels and no errors.
Thirty-three mechanical checks were copied into `metadata.delivered_checks`; nine are manual.
Three bindings were resolved at decompose. None changes a contract.

- γ creates `followup_hold.py` with the marker key and `is_holder`, and ε extends it. γ's `is_holder`
  conjunct otherwise named a producer downstream of γ.
- Boundary row 15's `followup_promoted{by: human}` events move to ζ. ε keeps the readiness half.
- Boundary row 20's curator render is γ's. α keeps the stamp half.

Two path corrections went into the filed tasks:

- ε's config schema is `fused-memory/src/fused_memory/config/schema.py`.
- ε also edits `server/tools.py` for `release_holder`.

G7 notes stamped into task descriptions, without waivers:

- γ records the item keys a partial R4 hit dropped (`no-silent-fail-soft`).
- α's reader allowlist test walks imports and seeds a forbidden fixture (`guards-exercise-behaviour`), and
  each allowlist entry cites decision 8 (`exceptions-owned-or-ratified`).
- ε's backfill runs before the server serves (`loop-thread-occupancy-bounded`).
- The holder edge is the authority for "held now", and η counts edges, not records (`one-fact-one-home`).

The overlay's trigger-chain gate does not apply: this PRD remediates no quality-instrument finding.

Known residual: δ, ε and ζ all edit `task_interceptor.py`, but §6's edges leave δ unordered against ε and
ζ. The overlay asks same-file leaves to be serialized by edges. The edges were filed exactly as §6, so
the module lock serialises them at dispatch and a rebase is the cost.

Companion corrections:

- 4719 was annotated (details plus `metadata.x_subsumed_by` → β/8414) and cancelled. Its only spawned
  child, 6227, is done, and nothing depends on it.
- 6072 was annotated with the composition rule: the parent edge and the holder edge are ANDed, and a
  parent edge never promotes on insert.
- 3260 was annotated: γ never writes `metadata.files` from a combine.
