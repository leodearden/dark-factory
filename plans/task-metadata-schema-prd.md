# PRD — Versioned shared TaskMetadata schema (`shared/task_metadata.py`)

**Stream:** W3 (task-metadata-schema) of the bug-hotspot remediation program 2026-07-06.
**Status:** deferred / active — authored 2026-07-06. Wave 1, no upstream deps.
**Approach:** B + H (contract + two-way boundary tests). High-stakes: a cross-process
contract on a live tracker with ~2100 historical tasks.
**Program doc (authoritative G4 seam map):** `plans/bug-hotspot-remediation-program-2026-07-06.md`.
**Findings:** `plans/bug-hotspot-survey-2026-07-06-full-findings.json` — clusters[6]/findings[0]
(unversioned contract, 8+ parsers), clusters[6]/findings[6] (update_task invariant bypass +
silent-discard), clusters[1]/findings[4] (workflow anti-thrash counters), cross_system/chains[2].

---

## 1. Goal — what a user/operator observes if this lands

Task metadata stops being a schemaless JSON blob that drifts silently across the
fused-memory ↔ orchestrator process boundary. Concretely:

- An operator who inspects the fused-memory journal sees a **single, structured
  `task_metadata.schema_warning` line** the first time any writer sends a metadata
  shape the schema doesn't recognise — instead of the drift shipping invisibly and
  surfacing weeks later as a permanently-blocked task (the 1902 → 1976/1982 chain).
- After the staged rollout completes, a deliberately-malformed metadata write
  (e.g. `update_task(metadata={'task_kind':'deterministic'})` on a `normal` task with
  no `before_done`) is **rejected at the write boundary with a diagnostic** returned
  through the MCP surface — where today it is silently accepted and later trusted by
  the orchestrator's `DeterministicRunner`.
- A self-restart deterministic deploy that writes
  `done_provenance.kind='deterministic-deploy-scheduled'` **lands `done`** because the
  orchestrator writer and the fused-memory validator import the **same** valid-kinds
  enum — it is structurally impossible for one side to know a kind the other rejects.

### The smoking-gun premise (verified 2026-07-06)

Commit `9493fa073d`: orchestrator task 1902 added
`done_provenance.kind='deterministic-deploy-scheduled'`; fused-memory's
`_VALID_PROVENANCE_KINDS` (`task_interceptor.py:3548`) was never updated in lockstep, so
**every** own-unit self-restart deploy's `done` write was silently rejected and the task
landed permanently `blocked` (tasks 1976, 1982). The enum has since been hand-patched to
include the kind (it currently lists all four at `:3548-3553`) — which is exactly the
recurring band-aid this PRD retires: producer and consumer each declare the vocabulary in
their own constants, so any new kind re-opens the same gap. The same shape produced
`before_done.cwd` never validated (127 failures, task 2105), the `memory_hints`
legacy-shape repair shim, and the untyped `external_deps` read the scheduler trusts.

The recurrence is documented in **12 done tasks** that each band-aided one parser or one
merge path: 44, 79, 81, 386, 395, 1100, 1235, 1245, 1511, 1813, 1827 (all `_row_to_task` /
`_merge_metadata` / update_task-metadata point-fixes). W3 replaces the class.

---

## 2. Background — the failure the schema closes

`metadata` carries ~15 load-bearing keys consumed by the orchestrator (a **different
process**) with no shared schema, no version, and no single parser. The task layer holds
**8 independent parse helpers with divergent failure semantics** (all re-verified
2026-07-06; line numbers drift, symbols stable):

| # | Site | Symbol | Unparseable → |
|---|---|---|---|
| 1 | `task_interceptor.py:977` | `_parse_metadata` | `{}` |
| 2 | `task_interceptor.py:1428` | `_extract_metadata_dict` | `None` |
| 3 | `task_interceptor.py:3845` | inline in `_merged_audit_metadata` | best-effort |
| 4 | `deterministic_task_guard.py:48` | `_parse_metadata` | `{}` (validation passes vacuously) |
| 5 | `lock_charter_guard.py:147` | `extract_files` (str→`json.loads` at :179) | `[]` |
| 6 | `tools.py:2343` | inline `json.loads` at :2359 | `None` (skip guard) |
| 7 | `sqlite_task_backend.py:274` | `_row_to_task` | warn-once + coerce `{}` |
| 8 | `sqlite_task_backend.py:1354` | `_merge_metadata` | corrupt-existing → raise; corrupt-incoming → last-write-wins |

Plus the read-time repair shim `_normalize_legacy_memory_hints_value`
(`sqlite_task_backend.py:1320`) that exists solely to fix a `memory_hints` shape that
already shipped, and the scheduler's untyped read
`(task.get('metadata') or {}).get('external_deps') or []` (`scheduler.py:1996`).

Deterministic-task invariants (B10: `task_kind` enum;
`deterministic ⇒ before_done ∨ always_escalates`; `before_done` only on deterministic;
script exists+executable+`timeout_secs`) run **once** at `submit_task`
(`tools.py:2804-2810`). `update_task`'s guard chain
(`task_interceptor.py:3144-3151`) rejects only `status` / `done_provenance` / directory-lock
writes — **nothing stops** `update_task(metadata={'task_kind':'deterministic'})` on a
`normal` task, injecting/mutating `before_done`, or flipping `always_escalates`, all of
which the `DeterministicRunner` trusts at dispatch. Separately,
`deterministic_task_guard._parse_metadata` maps an unparseable string to `{}` so validation
passes vacuously **and** `inject_task_kind` (`:220`) then **replaces the caller's entire
metadata** with `{'task_kind':'normal'}` — silent data loss with no warning.

`shared/` is already a workspace dependency of both consumers
(`dark-factory-shared = { workspace = true }` in `fused-memory/pyproject.toml` and
`orchestrator/pyproject.toml`); the orchestrator already imports `shared.usage_gate`,
`shared.locking`, `shared.cost_store`, `shared.safe_io`, `shared.config_models`. `pydantic>=2.7`
is a direct dependency of `shared`, `fused-memory`, and `orchestrator`. There is **no new
coupling** and **no substrate to build** — this is pure adoption of an existing pattern.

---

## 3. Sketch of approach

Create **`shared/src/shared/task_metadata.py`**:

1. **`TaskMetadata`** — a pydantic v2 `BaseModel`, `schema_version: int = 1`,
   `model_config = ConfigDict(extra='allow')` so unknown top-level keys are **retained and
   re-serialised** (round-trip preservation is load-bearing for the 2100 historical tasks
   and for forward-drift survival — a writer on a newer version may legitimately carry a
   key this version doesn't type yet).
2. **Typed sub-models** — `BeforeDone`, `DoneProvenance`, `MemoryHints`, `ExternalDep`,
   `RetryLedger` (fields enumerated in the **Contract** below). Each currently-shipping shape
   is covered so **no live write is rejected** by v1.
3. **Sub-model registry** (the W10 extension point, G1) —
   `register_metadata_submodel(key: str, model: type[BaseModel])`. A downstream package
   registers a sub-model against a top-level metadata key at import time; `parse_metadata`
   validates that key's slice against the registered model and reattaches the typed
   instance. W10 defines `DeployState` and calls
   `register_metadata_submodel('deploy_state', DeployState)` — **without editing `shared/`.**
4. **One `parse_metadata(blob, *, direction, enforce)`** — the single parser, one
   documented failure policy (see Contract §"Failure policy"). Applies the **versioned
   migration registry** (v0→v1) *before* validation; the `memory_hints` legacy-list→dict
   normalisation moves in here as the first registered migration, retiring the ad-hoc
   backend shim behind the version path.
5. **Enforcement at the single write boundary** — `SqliteTaskBackend.add_task` /
   `update_task` call `parse_metadata(direction='write')`. Because `update_task` merges
   (`_merge_metadata`) before persisting, validation runs on the **post-merge** blob, which
   is what makes the deterministic cross-field invariants hold under `update_task` (the
   finding-6 fix falls out of the model's own `@model_validator`, no separate interceptor
   guard). The 8 ad-hoc parsers are deleted in favour of the shared one; the orchestrator
   imports the same model for reads (scheduler `external_deps`) and for the
   `DoneProvenance` write.

### Staged rollout (the migration discipline)

The live tracker has ~2100 tasks whose metadata predates the schema. Rejection is **opt-in
and staged**:

- **Warn-mode (default, `task_metadata.enforce: false`)** — `parse_metadata(direction='write')`
  validates; a violation emits a structured `task_metadata.schema_warning`
  `{task_id, field, error}` log line and the write **proceeds**. Legacy shapes recognised by
  the migration registry are upgraded silently (no warning) — so warn-mode only fires on
  *genuinely novel* drift, which is precisely what must be zero before enforcing.
- **Enforce-mode (`task_metadata.enforce: true`)** — the same violation is **rejected** with
  a `ValidationError` surfaced through the MCP write path.
- The **read path** (`_row_to_task`) is always tolerant: a corrupt stored row warns and
  returns best-effort (never raises) so one bad historical row cannot break `get_tasks`.
  This read/write asymmetry is deliberate and documented (writes guard the boundary; reads
  survive legacy data).

The flip from warn to enforce is a **human-blessed deterministic gate** (task θ2) gated on
**zero `task_metadata.schema_warning` lines over a full recon cycle**, followed by an
out-of-cgroup fused-memory restart (resolved decision #6). Warn-mode go-live is a
deterministic auto-deploy (θ1) that restarts fused-memory to load the new validation and
start the census clock.

---

## 4. Resolved design decisions (do not relitigate)

1. **pydantic, not dataclasses.** G3 verified: `pydantic>=2.7` is a direct dependency of all
   three packages and `shared/config_models.py` is an existing shared-pydantic precedent.
   The brief's dataclass fallback is **not** needed.
2. **Enforcement home = `SqliteTaskBackend.add_task`/`update_task`, on the post-merge blob.**
   This is the single durable write chokepoint every write funnels through. The
   deterministic cross-field invariants live in the shared model's `@model_validator`, so
   validating the post-merge blob at the backend enforces finding-6 uniformly on add **and**
   update — collapsing the separate interceptor guard finding-6 offered as an alternative
   ("both collapse into the shared schema if it lands first"). W3 does **not** add a
   `_reject_kind_fields` interceptor guard; the model + backend validation cover it. (The
   status-transition table and directory-lock guards remain W2's / Lock-charter's — W3 does
   not touch them.)
3. **Unknown-key policy = retain + warn (never discard).** `extra='allow'` retains every
   unknown top-level key and re-serialises it (round-trip). A key that is neither a known
   field, nor a registered sub-model, nor `x_`-prefixed emits the schema-warning in
   warn-mode (feeding the enforce-gate census). `x_`-prefixed keys are the **sanctioned
   forward-compat namespace** — retained silently, never warned. Enforcement targets
   *malformed typed sub-models*, not unknown top-level keys (those are always preserved so a
   newer writer never loses data across the boundary).
4. **Extension point = model registry, not a discriminated union.** A registry
   (`register_metadata_submodel`) lets W10 register `DeployState` from its own package with
   zero edits to `shared/` — the brief's hard requirement. A discriminated union on a single
   tagged field would force every sub-model into one field and require editing `shared/` per
   variant; rejected.
5. **`parse_metadata` failure policy is direction-split.** Write path: warn-and-accept
   (warn-mode) / reject-loudly (enforce-mode); **never** the current silent-`{}` discard.
   Read path: warn-and-best-effort, never raise. One function, one documented policy; the
   `direction` and `enforce` parameters select the response.
6. **`task_metadata.enforce` is a red-tier (restart-only) config key.** It is not added to the
   config-hot-reload green-tier allowlist (that is a different subsystem, out of scope). The
   enforce flip therefore requires a fused-memory restart, which the θ2 runbook performs.
7. **Deploys.** θ1 (warn-mode go-live) is a deterministic **auto-deploy** — warn-mode is
   non-rejecting, so it is safe to apply without a human and we want the census clock started
   promptly; `before_done` = the existing committed `scripts/restart-fused-memory.sh` (no
   `--drain`, per resolved decision #6; `--drain` hung — task 2090). θ2 (enforce flip) is a
   deterministic **pure gate** (`always_escalates: true`, no `before_done`) — enabling
   *rejection* on a live tracker with 2100 historical tasks is a deliberate,
   judgment-requiring flip that should have a human, and "zero warnings over a full recon
   cycle" needs human confirmation. This is a born-at-L2 escalation the operator handles
   when present — **not** an AskUserQuestion, consistent with AFK autonomy.
8. **Orchestrator scope is seam-adoption only.** W3 changes the scheduler `external_deps`
   read and the `DeterministicRunner`'s `done_provenance` construction to use the shared
   model (so the valid-kinds enum is shared → structural drift-prevention). It does **not**
   refactor the deterministic-runner phase state machine — that is W10 (`DeployState`), which
   depends on W3.

---

## 5. Contract (B+H) — `shared/task_metadata.py` seam

### Sub-model fields (v1 — cover every currently-shipping shape)

```
class BeforeDone(BaseModel):        # metadata.before_done
    script: str                     # required, non-empty, project_root-relative, no ../ escape
    args: list[str] = []
    env: dict[str, str] = {}
    cwd: str | None = None          # None → project_root (task 2105: was never validated)
    timeout_secs: int               # required, > 0
    target_unit: str | None = None  # None → cross-unit (no self-kill)

class DoneProvenance(BaseModel):    # metadata.done_provenance
    kind: Literal['merged','found_on_main',
                  'deterministic-deploy','deterministic-deploy-scheduled']   # THE shared enum
    commit: str | None = None       # required when kind in {merged, found_on_main}
    note: str | None = None         # required when kind == found_on_main
    pid: int | None = None          # deterministic-deploy: new MainPID
    unit: str | None = None         # deterministic-deploy(-scheduled): target unit
    active_enter_timestamp: str | None = None

class MemoryHints(BaseModel):       # metadata.memory_hints — canonical {entities, queries}
    entities: list[str] = []
    queries: list[str] = []
    # legacy [{entity, query}, ...] → migrated v0→v1 in parse_metadata (registry migration)

class ExternalDep(BaseModel):       # elements of metadata.external_deps (list[str] canonical)
    project_id: str
    task_id: str
    # canonical wire form is the "project_id:task_id" string; the model parses/renders it

class RetryLedger(BaseModel):       # metadata.retry_ledger — the 8 anti-thrash counters
    consecutive_no_plan_failures: int = 0
    total_no_plan_failures: int = 0
    last_no_plan_main_sha: str | None = None
    consecutive_infra_resume_failures: int = 0
    last_infra_resume_iteration_count: int = 0
    consecutive_merge_thrash: int = 0
    last_merge_outcome_signature: str | None = None
    merge_first_enqueued_at: str | None = None

class TaskMetadata(BaseModel):
    model_config = ConfigDict(extra='allow')     # retain + re-serialise unknown keys
    schema_version: int = 1
    task_kind: Literal['normal','deterministic'] = 'normal'
    always_escalates: bool = False
    before_done: BeforeDone | None = None
    done_provenance: DoneProvenance | None = None
    memory_hints: MemoryHints | None = None
    external_deps: list[str] = []                # canonical "project_id:task_id" wire strings
    retry_ledger: RetryLedger | None = None
    files: list[str] = []                        # Lock-charter Contract-1: file-level only

    @model_validator(mode='after')
    def _deterministic_invariants(self):
        # deterministic ⇒ before_done ∨ always_escalates
        # before_done ⇒ task_kind == 'deterministic'
        # (raises → rejected at the write boundary in enforce-mode; warns in warn-mode)
```

Runner stamps (`before_done_ran_at`, `before_done_verified_at`, `before_done_verified_pid`,
`gate_escalated_at`, `done_provenance`) are top-level string/dict fields carried through as
typed-or-`extra`; the runner writes them via the shared model so a stamp shape can never
diverge from what the backend validates.

### Registry extension point

```
_SUBMODEL_REGISTRY: dict[str, type[BaseModel]] = {}
def register_metadata_submodel(key: str, model: type[BaseModel]) -> None: ...
# W10:  register_metadata_submodel('deploy_state', DeployState)
```

`parse_metadata` validates any registry-registered key present in the blob against its
registered model. Registration is idempotent; double-registration of a different model for
the same key raises (loud, at import time).

### `parse_metadata` signature + failure policy

```
def parse_metadata(
    blob: dict | str | None,
    *,
    direction: Literal['read','write'],
    enforce: bool = False,
) -> tuple[TaskMetadata, list[SchemaWarning]]:
```

- `None`/absent → empty `TaskMetadata()` (benign-absent).
- `str` → `json.loads`; on failure: **write** → warn-or-reject per `enforce` (never silent
  `{}`); **read** → warn + empty, never raise.
- Apply registered v0→v1 migrations, then validate. Known/registered sub-model invalid:
  **write+enforce** → raise `ValidationError`; **write+warn** → emit `SchemaWarning`, accept
  raw; **read** → `SchemaWarning`, best-effort.
- Unknown top-level key: retained; `x_`-prefixed → silent; else `SchemaWarning` in warn-mode.
- Returns the model **and** the warning list so the backend can emit the structured census
  line and the tests can assert exact warning content.

### Invariants

- **I1 Round-trip preservation:** `parse_metadata(blob)` then `model_dump()` preserves every
  unknown key byte-for-value-equal (no silent drop).
- **I2 One vocabulary:** the `DoneProvenance.kind` `Literal` is the *only* valid-kinds
  declaration; `fused-memory`'s `_VALID_PROVENANCE_KINDS` is deleted and both sides import
  the model. A kind unknown to the model is rejected identically on the write side and never
  constructed on the orchestrator side.
- **I3 Post-merge enforcement:** `update_task` validates the merged blob, so a
  metadata write that would make the task violate a deterministic invariant is caught on
  update, not only on submit.
- **I4 No silent discard:** an unparseable write-path metadata string never coerces to `{}`
  and never triggers `inject_task_kind`'s whole-metadata replacement without a warning.

---

## 6. Boundary-test sketch (B+H) — the integration-gate signal (task ζ)

| # | Facing | Scenario | Preconditions | Postconditions (asserted) |
|---|---|---|---|---|
| 1 | producer (legacy read) | old pre-schema blob with unknown keys + legacy `memory_hints` list | a v0-shaped dict | `parse_metadata → model_dump` **preserves** every unknown key (I1) **and** upgrades `memory_hints` to `{entities,queries}` |
| 2 | consumer (write) | orchestrator-constructed `DoneProvenance(kind=k)` for every `k` in the shared enum | fused-memory backend at HEAD | backend `update_task` validation **passes** for all four kinds (I2) |
| 3 | consumer (write, negative) | a `DoneProvenance`-shaped write with `kind='bogus'` | enforce-mode | **rejected** symmetrically — the orchestrator model refuses to construct it AND the backend refuses to store it |
| 4 | producer↔consumer (ledger) | workflow-written `RetryLedger` round-trips | ε landed | typed ledger persisted + re-read equal; a persist failure **escalates** (not silent under-fire) |
| 5 | staged rollout | same malformed write under warn vs enforce | `task_metadata.enforce` toggled | warn-mode **accepts + emits one `task_metadata.schema_warning`**; enforce-mode **rejects** with `ValidationError` |
| 6 | update-path invariant | `update_task(metadata={'task_kind':'deterministic'})` on a `normal` task, no `before_done` | enforce-mode | post-merge validation **rejects** (I3); warn-mode warns + accepts |

ζ's observable signal = **this suite green in CI**. It exercises the real producer
(orchestrator construction) and real consumer (fused-memory backend validation) across the
package boundary — an integration gate, not a synthetic-input unit test (G2 C-as-gate).

---

## 7. Pre-conditions for activating

- None upstream (wave 1). `shared/`, pydantic, and the workspace wiring all exist today.
- θ1 (warn-mode deploy) requires β + γ merged. θ2 (enforce flip) requires ζ green + θ1 done
  **and** a full recon cycle with zero `task_metadata.schema_warning` lines observed live
  (a runtime precondition the θ2 gate confirms; if unmet it escalates for a human).

---

## 8. Cross-PRD relationship (G4)

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| W10 harness-supervision | W10 **consumes** | `register_metadata_submodel('deploy_state', DeployState)` — the registry extension point | **W3 (this PRD)** owns the registry; W10 owns `DeployState` content | queued (W10 wave 2, depends on W3) |
| W2 task-status-authority | peer / file-adjacent | both add to `shared/` and both touch `task_interceptor.py`; W2 = status transition table, W3 = metadata blob | disjoint — W2 owns status/transition, W3 owns metadata schema (brief: status column semantics are W2, out of W3 scope) | independent |
| W8 fm-task-dedup | peer / file-adjacent | both touch the `update_task` write path; W8 = `candidate_key` + write-authority seam, W3 = metadata validation | disjoint — `candidate_key`/dedup out of W3 scope (brief) | independent |
| W9 workflow-state-machine | peer | W9 owns `BlockDisposition`/`StewardOutcome`; W3 owns `retry_ledger` | disjoint — distinct mechanisms; W9 does **not** depend on W3 | independent |

The only owned cross-PRD seam is the **registry extension point W10 consumes** — W3 holds
it, W10 registers into it. No reciprocal-ownership ambiguity.

---

## 9. Decomposition plan

Greek labels; task IDs assigned at decompose time. Leaves: **ζ, θ2**. All others are
intermediates naming their downstream consumer.

- **α — `shared/task_metadata.py` foundation.**
  Modules: `shared/src/shared/task_metadata.py`, `shared/tests/test_task_metadata.py`.
  The versioned `TaskMetadata` v1 + all 5 sub-models + `register_metadata_submodel` +
  `parse_metadata` (one policy) + the versioned migration registry (v0→v1, incl. the
  `memory_hints` list→dict migration). Unit tests: round-trip preservation (I1), each
  sub-model, the deterministic-invariant validator, the registry, the migration.
  *Intermediate → unlocks β, γ, δ, ε, ζ.* `force_full_path=true` (foundational design).

- **β — fused-memory write-boundary validation + backend parser collapse.**
  Modules: `fused-memory/src/fused_memory/backends/sqlite_task_backend.py`,
  `fused-memory/config/config.yaml`, `fused-memory/tests/test_sqlite_task_backend.py`.
  Wire `parse_metadata(direction='write')` into `add_task`/`update_task` (post-merge for
  update); add the `task_metadata.enforce` config flag (default `false`) + the structured
  `task_metadata.schema_warning` census line. Collapse the 3 backend parsers: `_row_to_task`
  coerce, `_merge_metadata`, and retire `_normalize_legacy_memory_hints_value` into the
  migration registry.
  *Signal:* a deliberately-malformed write emits the `task_metadata.schema_warning` line and
  (warn-mode) still succeeds. *Intermediate → unlocks ζ, θ1.* Deps: α.

- **γ — fused-memory policy-layer parser collapse + invariant enforcement.**
  Modules: `fused-memory/src/fused_memory/middleware/task_interceptor.py`,
  `fused-memory/src/fused_memory/middleware/deterministic_task_guard.py`,
  `fused-memory/src/fused_memory/middleware/lock_charter_guard.py`,
  `fused-memory/src/fused_memory/server/tools.py`, + their tests.
  Replace parsers #1-6 (interceptor `_parse_metadata`/`_extract_metadata_dict`/inline;
  `deterministic_task_guard._parse_metadata`; `lock_charter_guard.extract_files` json branch;
  `tools.py` inline `json.loads`) with the shared `parse_metadata`/typed accessors. Delete
  `_VALID_PROVENANCE_KINDS` (I2 — import the model's enum). Fix `inject_task_kind`'s silent
  whole-metadata replacement to warn (I4). The finding-6 update_task invariant is enforced by
  α+β's post-merge validation (decision #2) — γ does not add a separate guard.
  *Signal:* `submit_task` with an unparseable metadata string warns loudly instead of
  silently discarding; a `DoneProvenance` kind unknown to the model is rejected via the
  shared enum. *Intermediate → unlocks ζ, θ1.* Deps: α. `force_full_path=true`.

- **δ — orchestrator adopts the shared model (reads + DoneProvenance write).**
  Modules: `orchestrator/src/orchestrator/scheduler.py`,
  `orchestrator/src/orchestrator/deterministic_runner.py`, + their tests.
  Scheduler `external_deps` read (`scheduler.py:1996`) via `parse_metadata`/`ExternalDep`;
  `DeterministicRunner` `done_provenance` stamps constructed from the shared `DoneProvenance`
  (one enum shared with the backend — structurally prevents the 1902-class re-drift). Does
  **not** refactor the runner phase machine (W10).
  *Signal:* a `done_provenance.kind='deterministic-deploy-scheduled'` constructed on the
  orchestrator side validates against the backend model (the 1976/1982 failure no longer
  reproducible). *Intermediate → unlocks ζ.* Deps: α.

- **ε — workflow anti-thrash counters → typed `RetryLedger`.**
  Modules: `orchestrator/src/orchestrator/workflow.py`, + its tests.
  Migrate the 8 ad-hoc counters onto `metadata.retry_ledger` (typed); guards become pure
  functions `RetryLedger → verdict`; `_normalize_cause_hint` / `_compute_merge_outcome_signature`
  become ledger methods (one signature-keying, not per-call-site); **persist failure
  escalates** instead of silently losing an increment (the guard exists to stop
  money-burning loops).
  *Signal:* an infra-resume thrash sequence increments
  `retry_ledger.consecutive_infra_resume_failures` and the guard fires at threshold; a
  persist failure escalates. *Intermediate → unlocks ζ.* Deps: α.

- **ζ — two-way boundary-test suite (integration gate, LEAF).**
  Modules: `fused-memory/tests/test_task_metadata_boundary.py` (imports `shared` + exercises
  the fused-memory backend + orchestrator-constructed payloads via the workspace).
  Implements the §6 boundary-test sketch (rows 1-6).
  *Signal:* the suite green in CI. *Leaf.* Deps: α, β, γ, δ, ε.

- **θ1 — warn-mode go-live (deterministic auto-deploy).**
  `task_kind='deterministic'`, `before_done={script:'scripts/restart-fused-memory.sh', args:[],
  timeout_secs:120, target_unit:'fused-memory.service'}`, `always_escalates=false`.
  Restarts fused-memory out-of-cgroup so β/γ warn-mode validation goes live and the census
  clock starts (no `--drain`; resolved decision #6).
  *Signal:* post-restart, a malformed metadata write emits `task_metadata.schema_warning`
  (warn-mode confirmed live). *Intermediate → unlocks θ2.* Deps: β, γ.

- **θ2 — enforce-flip capstone (deterministic pure gate, LEAF).**
  `task_kind='deterministic'`, no `before_done`, `always_escalates=true`. Born-at-L2
  escalation carrying the runbook: verify zero `task_metadata.schema_warning` over a full
  recon cycle → set `task_metadata.enforce: true` in `fused-memory/config/config.yaml` →
  out-of-cgroup `systemctl --user restart fused-memory.service`.
  *Signal:* post-flip, a malformed metadata write is **rejected** with a `ValidationError`
  through the MCP surface (where pre-flip it warned + accepted). *Leaf.* Deps: ζ, θ1.

### Dependency edges

```
β → α      γ → α      δ → α      ε → α
ζ → α, β, γ, δ, ε
θ1 → β, γ
θ2 → ζ, θ1
```

---

## 10. Out of scope

- **Status column semantics / the transition table** — W2 (task-status-authority).
- **`DeployState` sub-model CONTENT** — W10 defines it and registers it via W3's extension
  point. W3 provides the registrable extension point (in scope, G1) but not the phase enum
  or the persisted verify baseline.
- **`candidate_key` / dedup / the update_task write-authority seam** — W8 (fm-task-dedup).
- **Config-hot-reload green-tier membership for `task_metadata.enforce`** — kept red-tier
  (restart-only); adding it to the hot-reload allowlist is a separate subsystem's concern.
- **Deterministic-runner phase state-machine refactor** — W10.

---

## 11. Open questions (tactical — surfaced, not decided; operator AFK, safe defaults taken)

1. **Census surface for the enforce gate.** θ2's "zero warnings over a recon cycle" is
   verified by grepping the fused-memory journal for `task_metadata.schema_warning` since the
   θ1 timestamp. **Default taken:** structured log line only (lowest coupling). An optional
   counter on `get_status`/health would let a script check it programmatically —
   decide during β if the log-grep proves awkward for the θ2 runbook.
2. **θ1 restart script choice.** **Default taken:** the existing committed
   `scripts/restart-fused-memory.sh` (no `--drain`), which does exactly
   `systemctl --user restart fused-memory` + health-wait — matches resolved decision #6 and
   satisfies the `before_done` exists+executable check today. If the deterministic runner's
   cross-unit blocking path needs a dedicated wrapper, add one during θ1 (tactical).
3. **θ2 auto vs pure-gate.** **Default taken:** pure gate (`always_escalates=true`) so a
   human blesses enabling rejection on a live tracker. If the operator prefers a scripted
   auto-flip once warn-mode has proven clean, convert θ2 to a `before_done` auto-deploy
   (census-check-and-flip script) at that time — no schema change required.
4. **`ExternalDep` wire form.** The canonical persisted form stays the `"project_id:task_id"`
   string (unchanged for the scheduler + `get_external_statuses`); the `ExternalDep`
   sub-model is a parse/render convenience. **Default taken:** do not change the persisted
   representation — validate/normalise only. Decide during δ whether the scheduler consumes
   the typed form or keeps the string.

---

## 12. META check

> If I decompose and queue this PRD without further oversight, will the architecture of what
> gets implemented be complete, coherent, cohesive, and good?

**Yes.** The single artifact (`shared/task_metadata.py`) has one owner, one parser, one
failure policy, and one valid-kinds enum; every consumer (both processes + W10's registry)
is named; the substrate (pydantic, `shared/` workspace dep) is verified present; the
high-stakes migration is staged warn→enforce with a human on the rejection flip and a
CI-green two-way boundary gate; and the extension point W10 requires is a first-class part of
the contract. No open **design** question remains — the four open items are tactical.
