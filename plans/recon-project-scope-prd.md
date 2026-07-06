# PRD: ProjectScope — type-level project identity for reconciliation (recon-project-scope)

- **Status:** active — authored 2026-07-06, stream **M4** of the bug-hotspot remediation
  program (`plans/bug-hotspot-remediation-program-2026-07-06.md`, authoritative for seams
  and conventions). Survey finding: `plans/bug-hotspot-survey-2026-07-06-full-findings.json`
  → "project_id and project_root are interchangeable bare strs threaded through every
  signature, with mutable post-construction injection into stages" (fm-recon, verdict
  confirmed).
- **Mode:** bare **B** (G5) — mechanical signature plumbing inside one package
  (`fused_memory.reconciliation`) plus one small type module; no new runtime behaviour,
  no cross-service seam, single-PRD consumer set. Contract-style rigor is supplied by the
  type definitions themselves (§Resolved decisions 1–3), so a separate contract/boundary-test
  section adds nothing.

## Goal

Kill the recurring `project_id`/`project_root` transposition bug class (tasks 156, 186,
927, 930/931/932/948/958/959/961/963; cancelled dup 880; partial structural fix 1143) **at
type-check time**: introduce a frozen dataclass `ProjectScope(project_id: ProjectId,
project_root: ProjectRoot)` built on `typing.NewType`, validated at construction
(`os.path.isabs(project_root)`), constructed only at the two validated trust boundaries,
and threaded as a **single parameter** through the reconciliation harness, stages,
`targeted.py`, and the module-level sweep/guard helpers. Because pyright already gates
every merge (pre-commit 3× pyright + merge-verify unscoped pyright on
Protocol/TypedDict/NewType changes), any future transposition at any call site becomes a
**compile-time type error** instead of a runtime cross-project contamination bug.

Operator-observable end state:

- `pyright` (unscoped) is green on main with the new types threaded end-to-end.
- A transposed call (`ProjectScope(root, pid)`, or passing `scope.project_id` where a
  `ProjectRoot` parameter is expected, or passing a bare `str` into either slot) is
  **rejected by pyright** — mechanism verified live 2026-07-06 with pyright 1.1.408: a
  7-error fixture (transposition ×2, bare-str leak ×2, cross-param ×2, frozen-mutation ×1)
  all fire as `error` diagnostics.
- Constructing `BaseStage` (or any stage subclass) **without** a scope is a type error;
  assigning `stage.project_root = ...` after construction is both a pyright error and a
  runtime `AttributeError`.
- Constructing a `ProjectScope` with a relative/empty `project_root` raises at runtime.
- The silent degrade-to-no-op paths that existed only to tolerate the `''` default are
  gone (replaced by assertions / deleted dead branches); existing reconciliation tests
  stay green.

## Background

Both identifiers are plain `str` and travel side by side through dozens of signatures
(e.g. `targeted.py:378 _on_task_done(self, task_id: str, project_id: str, project_root:
str, ...)`, `:622`, `:1143`), so swapping them type-checks clean and fails only at
runtime. Task 1143's registry hard-bind (`_known_project_root_for`,
`harness.py:504-539`) fixed the harness entry point but not the mechanism:

- `BaseStage` defaults `self.project_id`/`self.project_root` to `''`
  (`stages/base.py:60-61`) and relies on the harness remembering to inject them
  post-construction (`harness.py:1831-1832` full cycle, `:2522-2523` remediation pass).
- Downstream stage code silently degrades to a no-op when `project_root` is unset
  (`stages/task_knowledge_sync.py:2706-2708` call-site comment; guard at `:1496`; sibling
  falsy-root guards at `:2388`, `:2992`; `stages/memory_consolidator.py:643`) — a
  silent-skip failure mode for a forgotten injection.
- Every new call site re-creates the swap opportunity, which is why the fix chain spanned
  weeks (G6 premise: fix-chain verified in task history 2026-07-06 — all listed tasks
  exist; 156/186/927/930/948/958/959/961/963 done, 880/931/932 cancelled as dups).

All line numbers above re-verified against main on 2026-07-06 in this session.

## Sketch of approach

1. **Types** (`fused-memory/src/fused_memory/models/scope.py` — the module that already
   owns project-identity resolution: `resolve_project_id`, `build_known_projects_map`):

   ```python
   ProjectId = NewType('ProjectId', str)
   ProjectRoot = NewType('ProjectRoot', str)

   @dataclass(frozen=True, slots=True)
   class ProjectScope:
       project_id: ProjectId
       project_root: ProjectRoot
       # __post_init__: raise unless project_id is non-empty and
       # os.path.isabs(project_root)
   ```

2. **Construction discipline** — exactly two sanctioned construction sites (see
   Resolved decision 3 for why two, not one):
   - `ReconciliationHarness._known_project_root_for(project_id) -> ProjectScope`
     (registry-backed; cycle entry). Callers: `run_full_cycle` entry (`harness.py:1770`),
     the backlog-iterator loop (`harness.py:2799`).
   - `TargetedReconciler.reconcile_task` entry (`targeted.py:206`), immediately after the
     existing `require_project_root(project_root)` — the interceptor process-boundary.
3. **Threading** — everything downstream of those two entries takes `scope: ProjectScope`
   (pair-taking signatures) or a NewType-annotated single parameter
   (`project_root: ProjectRoot` / `project_id: ProjectId` for single-identifier helpers).
4. **Stages** — `BaseStage.__init__` takes `scope: ProjectScope` as a **required**
   keyword argument; the `''` defaults and the post-construction injections are deleted;
   `self.project_id` / `self.project_root` become read-only properties over `self.scope`.
5. **Dead-branch removal** — the falsy-`project_root` degrade branches become
   unreachable and are deleted / replaced with assertions. The `taskmaster is None`
   halves of those guards **stay** (a stage legitimately runs without a task backend).

## Pre-conditions (substrate — G3, all verified 2026-07-06)

| Assumed capability | Evidence |
|---|---|
| `typing.NewType`, `dataclass(frozen=True, slots=True)` | stdlib; fixture type-checked with repo's pyright 1.1.408 (`uv run pyright`) |
| pyright rejects NewType transposition/bare-str/frozen-mutation | live fixture run 2026-07-06: 7/7 expected errors fired (see Goal) |
| pyright gates every merge | pre-commit runs 3× pyright; merge verify escalates to **unscoped** pyright on NewType/Protocol/TypedDict diffs (CLAUDE.md verify conventions) — these tasks should expect full-repo type checks |
| `_known_project_root_for` registry hard-bind | `harness.py:504-539` (task 1143), callers at `:1770`, `:2799` |
| `models/scope.py` identity helpers | `resolve_project_id`, `build_known_projects_map` present; interceptor already derives project_id FROM project_root at its `reconcile_task` call site (`task_interceptor.py:841`) |
| `require_project_root` raising validator | `utils/validation.py:149-152`; called at `targeted.py:206` |

No novel substrate beyond the above — nothing is queued as a prerequisite.

## Resolved design decisions

1. **Type home = `fused_memory/models/scope.py`.** It already owns project-identity
   resolution and is imported by both the reconciliation package and middleware. The
   existing pydantic `Scope` class there is *request* scope (project/agent/session) — a
   different concept; the new name `ProjectScope` avoids collision. No import cycle:
   `models/scope.py` imports nothing from `reconciliation/`.
2. **Validation lives in the dataclass** (`__post_init__`), not in the factories: *any*
   construction with an empty project_id or a non-absolute project_root raises
   immediately. The brief's "assert `os.path.isabs(project_root)` in
   `_known_project_root_for`" is thereby enforced for every construction site, present
   and future, not just the factory.
3. **Two sanctioned construction sites, not one** (recorded deviation from the brief's
   "exactly ONE place"). The targeted reconciler is constructed independently of the
   harness (`server/main.py:696`) and is entered from the TaskInterceptor, not from cycle
   entry. Routing it through the harness registry would (a) couple two independently
   wired components and (b) **regress behaviour**: `_known_project_root_for` raises
   `UnknownProjectError` for projects absent from `DASHBOARD_KNOWN_PROJECT_ROOTS`, while
   targeted reconciliation today correctly serves any project_root the MCP boundary
   validated. The invariant that matters is preserved: every `ProjectScope` is validated
   at construction, and **all code downstream of the two entries accepts only
   `ProjectScope`** — no third construction site is permitted (enforce by convention +
   grep in review; both sites are named in the module docstring).
4. **`BaseStage` API**: `scope: ProjectScope` becomes a required keyword-only `__init__`
   parameter. `self.project_id` / `self.project_root` become **read-only `@property`**
   accessors returning `self.scope.project_id` / `self.scope.project_root` — this keeps
   the ~85 existing read sites in stage subclasses working unchanged while making the old
   injection pattern (`stage.project_root = ...`) a pyright error **and** a runtime
   `AttributeError`. `known_projects` (same injection pattern, `harness.py:1833`/`:2524`)
   moves into `__init__` as an optional parameter in the same pass.
5. **The vestigial `harness.__init__` stage pre-build is deleted** (`harness.py:412-435`,
   `self.stages`). Verified: its only production consumer is
   `_propagate_escalation_queue(self.stages)` at `:1371`, and those instances never run a
   cycle — every cycle builds fresh stages via `_make_stages()` (`:583-605`, `:1826`,
   `:2504`), which propagates the escalation queue itself at `:604`. `_make_stages`
   becomes `_make_stages(scope)`. Tests that monkeypatch
   `harness._make_stages = lambda: harness.stages` (test_harness.py fixtures at ~157,
   ~228, ~307, ~519, ~8106) keep their `harness.stages[N]` access pattern by having the
   fixture build the stage list with a test scope and assign it as an instance attribute;
   only the fixture sites and direct stage constructions need edits, not the ~100
   `harness.stages[...]` references.
6. **Task-1163 snapshot semantics are load-bearing and preserved**: the scope is
   constructed **once** at `run_full_cycle` entry and *threaded* through
   `_maybe_remediate` → `_run_remediation_pass`; remediation must keep using the
   pre-cycle snapshot and must NOT re-resolve from the registry mid-cycle (deliberate —
   see the comment block at `harness.py:2431-2436`;
   `test_remediation_pass_uses_threaded_project_root_over_registry` and
   `test_remediation_uses_threaded_project_root_not_mutated_registry` pin it, and pass a
   root that intentionally differs from the registry). No registry-bound asserts inside
   `_run_remediation_pass`.
7. **Falsy-root degrades die; taskmaster-None degrades stay.** Branches deleted/replaced
   with assertions: `task_knowledge_sync.py:1496` (`if not taskmaster or not
   project_root: return 0` → keep only the taskmaster half), `:2388` (root half of
   `if not project_root or not tasks`), `:2992` (root half of the silent skip),
   `memory_consolidator.py:634-645` (`_build_project_root_directive` falsy branch —
   with a required scope the directive is always emitted; simplify accordingly and update
   the docstring + any test pinning the `''` behaviour).
8. **Single-identifier helpers get NewType annotations; pair-takers get `scope`.**
   Pair-taking signatures (the transposition hazard) switch to a single
   `scope: ProjectScope` parameter — e.g. `_sweep_terminal_task_flag_markers(memory,
   taskmaster, project_root, project_id, run_id)` (`task_knowledge_sync.py:1424-1429`),
   targeted handlers `_on_task_done/_blocked/_cancelled/_deferred` and the sweep/guard
   helpers (`targeted.py:377/621/725/794/994/1020/1081/1142`), harness pair-threading
   internals (`_run_remediation_pass`, remediation wiring). Single-identifier helpers
   (`_fetch_filtered_task_tree(project_root)`, `_check_graphiti_queue_health(project_id)`,
   `_live_status_map(project_root)`, the module-level lifecycle probes in
   task_knowledge_sync taking bare `project_root`) keep one parameter, annotated
   `ProjectRoot` / `ProjectId` so a caller holding a scope cannot pass the wrong field.
9. **`build_known_projects_map` keeps returning `dict[str, str]`.** `dict` is invariant,
   so retyping to `dict[ProjectId, ProjectRoot]` would break its non-recon consumers
   (`ticket_janitor.py` et al.) for zero transposition value; the harness factory wraps
   the looked-up pair into a `ProjectScope` at `_known_project_root_for`. Out-of-package
   retyping is out of scope (W-stream territory).
10. **The TaskInterceptor is untouched.** Its `reconcile_task` call site already derives
    `project_id=resolve_project_id(project_root)` from the root
    (`task_interceptor.py:841`) — transposition is structurally impossible there — and
    `reconcile_task` keeps its external `(task_id, transition, project_id, project_root,
    ...)` signature so this PRD's file set stays disjoint from W2/W5's interceptor work.
    The scope is constructed just inside.

## Cross-PRD relationship (G4)

| Other PRD / stream | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| W5 recon-reliability (`ReconLedgerStore`, `ReconWritePolicy`, `recon_self_model.py`, `execution_class`) | W5 **consumes** `ProjectScope` where its tasks touch the same signatures (harness.py, task_knowledge_sync.py, targeted.py) | `ProjectScope` type + threaded signatures | **this PRD (M4)** per the program G4 table | wired (this batch) |

**Instruction to W5 (per program doc, reciprocal in W5's brief):** W5 tasks that modify
`fused-memory/src/fused_memory/reconciliation/harness.py`,
`stages/task_knowledge_sync.py`, or `targeted.py` MUST declare `add_dependency` edges on
this batch's tasks for those files (β for harness, γ for task_knowledge_sync, δ for
targeted) while both batches are pending, so W5's edits land on the scope-threaded
signatures rather than racing them. As of filing (2026-07-06), no W5 batch exists in the
task store — W5 wires its side when it files. File-level locks serialize execution either
way; the deps only prevent rebase churn and stale-signature edits.

No other stream touches these files (M-streams are orchestrator/dashboard-side; W6 is
graphiti_client-side).

## Decomposition plan

Labels α–δ; deps in brackets. All tasks: project dark_factory, high priority, full
architect path (no `complexity=simple` — every task either introduces an abstraction or
does multi-file signature surgery). Expect **unscoped pyright** at merge verify on all
four (NewType changes).

- **α — Define ProjectId/ProjectRoot/ProjectScope in models/scope.py** [—]
  - Files: `fused-memory/src/fused_memory/models/scope.py`,
    `fused-memory/tests/test_project_scope.py` (new).
  - NewTypes + frozen slots dataclass + `__post_init__` validation (empty id / relative
    or empty root ⇒ raise). Module docstring names the two sanctioned construction sites
    (decision 3). Runtime tests: frozen-ness, validation raises, happy path.
    **Type-gate test (the rejection mechanism, G6-verified):** a test that runs the
    repo's pyright programmatically over a small fixture containing a transposed
    construction + bare-str leak + cross-param misuse + frozen mutation, asserting the
    expected arg-type/frozen diagnostics are reported (7 errors verified live 2026-07-06).
  - Observable signal: `pytest fused-memory/tests/test_project_scope.py` green in CI,
    including the pyright-fixture rejection test; unscoped pyright green.
  - Consumer: β, γ, δ (this batch); W5 threads it onward.
- **β — Thread ProjectScope through harness + BaseStage; delete injection** [α]
  - Files: `fused-memory/src/fused_memory/reconciliation/harness.py`,
    `fused-memory/src/fused_memory/reconciliation/stages/base.py`, plus tests:
    `fused-memory/tests/test_harness.py`, `fused-memory/tests/test_stages.py`,
    `fused-memory/tests/test_curator_escalator.py`,
    `fused-memory/tests/reconciliation/test_base_stage_cutover.py`,
    `fused-memory/tests/reconciliation/test_stage1.py`,
    `fused-memory/tests/reconciliation/test_cutover_end_to_end.py`,
    `fused-memory/tests/reconciliation/test_assemble_payload_snapshot_filter.py`,
    `fused-memory/tests/reconciliation/test_recon_cross_cycle_dedup_citations.py`,
    `fused-memory/tests/reconciliation/test_stage2_recon_report_channel.py`.
  - `_known_project_root_for` returns `ProjectScope`; cycle entries (`:1770`, `:2799`)
    and `_run_remediation_pass` take/thread scope (decision 6 semantics preserved);
    `_make_stages(scope)`; BaseStage required scope + read-only properties + `''`
    defaults deleted (decisions 4–5); injection blocks at `:1831-1833`/`:2522-2524`
    deleted; vestigial `self.stages` pre-build (`:412-435`, `:1371`) deleted; harness
    single-identifier helpers annotated with NewTypes.
  - Observable signal: unscoped pyright green; recon test suites green;
    `grep -n "stage\.project_root = \|stage\.project_id = " harness.py` → no matches;
    `grep -n "project_root: str = ''" stages/base.py` → no matches; the task-1163
    threaded-snapshot tests still pass unmodified in their assertions.
  - Consumer: γ (stage-level helpers read `self.scope`); W5 harness tasks.
- **γ — Scope the stage-level sweep/guard helpers; delete falsy-root degrades** [α, β]
  - Files: `fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py`,
    `fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py`,
    `fused-memory/tests/test_stages.py`.
  - Pair-taking module helpers (notably `_sweep_terminal_task_flag_markers`
    `:1424-1429`) take `scope`; single-identifier helpers annotated; falsy-root branches
    at `:1496`, `:2388`, `:2992`, and `memory_consolidator.py:634-645` deleted/replaced
    per decision 7 (taskmaster-None halves retained); call-site comment at `:2705-2708`
    updated to stop claiming project_root-unset degradation.
  - Observable signal: unscoped pyright green; `pytest fused-memory/tests/test_stages.py`
    green; `grep -n "not taskmaster or not project_root" task_knowledge_sync.py` → no
    matches (taskmaster-only guard remains); a test constructing the sweep with a scope
    and a None taskmaster still returns 0 (retained degrade), while the
    root-unset path is no longer representable.
  - Consumer: W5 task_knowledge_sync tasks; recon operators (no more silent GC skips).
- **δ — Thread ProjectScope through targeted.py handlers and sweeps** [α]
  - Files: `fused-memory/src/fused_memory/reconciliation/targeted.py`,
    `fused-memory/tests/test_targeted.py`.
  - `reconcile_task` keeps its external str-pair signature (decision 10), constructs
    scope right after `require_project_root` (`:206`); handlers `_on_task_done/_blocked/
    _cancelled/_deferred`, `_sweep_cancelled_descendants`, `_sweep_cancel_orphan`,
    `_sweep_block_orphan`, `_sweep_escalate_l1`, `_should_withhold_batch_promotion`,
    `_fetch_done_provenance`, `_live_status_map` et al. take `scope` (pair-takers) or
    NewType-annotated singles per decision 8.
  - Observable signal: unscoped pyright green; `pytest fused-memory/tests/test_targeted.py`
    green; `grep -n "project_id: str, project_root: str" targeted.py` → no matches;
    interceptor-driven targeted recon still fires end-to-end
    (`tests/test_task_interceptor.py` unchanged and green).
  - Consumer: TaskInterceptor (existing runtime consumer, unchanged); W5 targeted tasks.

DAG: α → {β, δ}; β → γ. β/δ parallelizable (disjoint files). Leaves: γ, δ.

## Out of scope

- Everything else in the fm-recon survey finding: ledger/write-policy/self-model/
  `execution_class` are **W5** (running concurrently); marker/dedup logic untouched.
- Retyping identity parameters outside the reconciliation package (interceptor, server
  tools, backends) — this PRD's types are available to them, adoption is W-stream work.
- `build_known_projects_map` / registry retyping (decision 9).
- Renaming `DASHBOARD_KNOWN_PROJECT_ROOTS` (pre-existing tracked follow-up).
- Task 2115 (cross-graph entity leak) — different mechanism (graphiti driver race),
  deferred design-first task, no file overlap.

## Open questions (surfaced but not decided in this session)

1. **Rename `_known_project_root_for`?** It will return a `ProjectScope`, so
   `_known_project_scope_for` reads better; all callers are harness-internal (`:1770`,
   `:2799`). **Suggested resolution:** rename in β; keep a one-line docstring note
   pointing at task 1143 for history. Decide in β.
2. **Reuse `utils.validation.validate_project_root` inside `ProjectScope.__post_init__`
   vs a local isabs check.** `utils/validation.py` deliberately avoids importing
   `models.scope` (see its `:188` comment), so `models.scope → utils.validation` should
   be cycle-free, but verify at import time. **Suggested resolution:** reuse the shared
   validator if the import is clean; otherwise a local check with an identical message.
   Decide in α.
3. **Assertion style for the dead falsy-root branches** (`assert` vs raising
   `ValueError`). Asserts are stripped under `-O`; these paths are internal-invariant
   ("unreachable with a required scope"). **Suggested resolution:** plain `assert` with a
   message, matching the survey proposal. Decide in γ.
4. **Typing of `BaseStage.known_projects`** (`dict[str, str]` today). Leaving it as
   `dict[str, str]` is consistent with decision 9. Decide in β.
