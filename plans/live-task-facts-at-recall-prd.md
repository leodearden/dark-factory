# Live task facts at recall and at write (fused recall, plan 1)

**Status:** active; authored 2026-11-13 from the design Leo ruled the same day (every decision option A).
**Type:** greenfield, approach B + H (contracts and a two-way boundary suite).
**Origin:** Leo's answer to the review-all wave-3 sitting question on reconciliation's task-status sentences
(program ruling R66 in `plans/review-all-program-dark_factory-2026-10-10.md`), then an investigate-and-design
session (four measurement seats, one critic, two critic passes).
**Code anchors** are cited by symbol and were read against main `a2377e8699` (2026-11-13). Main moves fast:
re-locate every symbol at implementation time.

## Goal

Task status has one point of truth, the task store. After this plan:

1. Every place that shows recalled memory to a language model also shows, beside it, the **live facts of the
   tasks that memory involves**, read from the task store at that moment: an agent's own `search`, the lookup of
   a named entity, a dispatched agent's briefing, and the context reconciliation assembles for its own stages.
2. Every memory write that names a task gets those live facts **back in the write reply**.
3. Code that writes memory carries the task as a structured link and never formats a status value into text.
4. The machinery that exists only because status leaks into memory is **retired**: reconciliation's task-count
   snapshot, the status-sentence detector, the stale-status sweep, and four reconciliation-only status
   write gates.

Nothing stores a copy of status. Nothing hides or rewrites memory text. Memory holds what happened and what was
learned, each with its date; the task store answers what is true now.

## Background: what was measured (2026-11-13, dark_factory, read-only)

Method, scripts and raw outputs were under `/tmp/team-fused-recall-1113/` (volatile); the figures that shaped
the design are restated here with their counting rule so this document stands alone.

- **Exposure.** 3 of 120 recent dispatches were shown a present-tense "task N is/remains <status>" claim that
  the task store contradicts (lower bound; one seven-hour window; the rule misses other wordings). 31 of 2,202
  finished or cancelled tasks with a graph node carry a line presenting the task as open (lower bound).
- **What goes stale.** In 75 hand-read status-like items, 13 were knowledge leaning on a task's status ("X is
  blocked on task N"); 5 of those 13 were no longer true. Of 9 plain status claims, 8 were still true.
- **Writers.** 47,304 `add_memory` writes since 2026-10-08, all projects; 117 plainly state a task's status.
  The large code writers are the orchestrator's decision notes (22,234), reconciliation's completion echo
  (4,713), the orchestrator's completion note (3,822) and the task-count snapshot (1,285).
- **Wording rules do not work.** A tense-and-date classifier was right 6 times of the 11 it called "current
  claim". The existing status gates catch some wordings and miss others ("Task 1155 remains in-progress" passes
  all of them), and code inside the server goes around them. The graph library rewrites written text into
  facts, outside our control.
- **The task-mention scan mis-binds.** `fused-memory/src/fused_memory/utils/canonical_labels.py::scan_content`
  with `group_id='dark_factory'` reads "Reify task 3674 was cancelled" as the local task 3674, "task 8c7e163fcf"
  (a commit) as task 8, and "tasks 2226 and 2227" as 2226 only. A usage example naming "Task 3127" is a mention.
- **Reply defects.** `MemoryResult.entities` is always empty on search hits
  (`fused-memory/src/fused_memory/services/memory_service.py::MemoryService._search_graphiti` reads
  `edge.source_node`, which the graph library's edge does not have). On the exact-name path,
  `memory_service.py::_edge_to_dict` returns `temporal=None`, so the briefing's "newest ten facts, with dates"
  ordering in `orchestrator/src/orchestrator/agents/memory_recall.py::render_entity_block` does nothing.
- **Cost.** A status read for 20 ids: 0.1–0.4 ms. `search` runs about 2,600 times a day and `get_entity` about
  870, all projects. A briefing involves a median of 7 distinct tasks (90th percentile 14, maximum seen 28);
  88% of them are done.
- **Other projects.** One server process reads all registered projects' task stores (12 today) and already
  tells "no such task" from "could not check".

## Sketch of approach

One function reads task facts. One function binds recalled results to tasks. Three surfaces call them and one
renderer prints the result.

```
task store ──> middleware/task_facts.py::read_task_facts ──┐
                                                           ├─> server: search, get_entity, get_entity_by_uuid replies ──> briefing block
results ────> middleware/recall_annotation.py::bind_tasks ─┤   server: add_memory, add_episode replies
                                                           └─> reconciliation: context assembler text
```

Both new modules sit in the `middleware` layer so that `server` and `reconciliation` can both import them
(layer order: `plans/review-all-s19-package-layering-and-private-seams-prd.md` contract C3; `reconciliation`
may not import `server`). Code that reads memory to compare or de-duplicate it calls
`MemoryService.search` directly and is untouched: it keeps raw results.

## Contracts

### C1 — Task facts (`fused-memory/src/fused_memory/middleware/task_facts.py`, new)

```python
@dataclass(frozen=True)
class TaskRef:            # project_id is always canonical and never ''
    project_id: str
    task_id: str

class Lookup(StrEnum):
    FOUND = 'found'
    NO_SUCH_TASK = 'no_such_task'        # the project's store answered and has no such id
    COULD_NOT_CHECK = 'could_not_check'  # unregistered project, store unreachable, read raised

@dataclass(frozen=True)
class TaskFacts:
    ref: TaskRef
    lookup: Lookup
    reason: str | None                   # set iff COULD_NOT_CHECK
    status: str | None                   # set iff FOUND
    title: str | None
    waits_on: tuple[tuple[TaskRef, str], ...]   # unfinished dependencies and their status; () when finished
    finished_how: str | None             # done_provenance kind, when status is done and one is recorded
    finished_commit: str | None

async def read_task_facts(refs: Collection[TaskRef], *, reader: TaskFactsReader,
                          known_projects: Mapping[str, str]) -> dict[TaskRef, TaskFacts]: ...
```

- One batched read per project, ids-scoped. The store read is new:
  `fused-memory/src/fused_memory/backends/sqlite_task_backend.py::SqliteTaskBackend.get_task_facts(ids, project_root)`
  (status, title, the `dependencies` rows of those ids with each dependency's status, and
  `metadata.done_provenance`'s kind and commit), declared on
  `backends/task_backend_protocol.py::TaskBackendProtocol` and passed through
  `middleware/task_interceptor.py::TaskInterceptor`. `TaskFactsReader` is the narrow protocol those satisfy.
- Every requested ref has an entry in the result. Absence is not a signal (`no-silent-fail-soft`).
- A read that raises for one project yields `COULD_NOT_CHECK` for that project's refs and does not fail the
  others.
- The status-only helper nested in the tool registration,
  `fused-memory/src/fused_memory/server/tools.py::_batched_task_statuses`, is deleted; its two callers
  (`::_claim_task_statuses`, `::_verify_mint_referent`) read through `read_task_facts`. Its three-valued
  meaning is preserved exactly: a caller that needs "could not consult" keeps getting it. This is the lift the
  server-boundary plan's step ε3 intended (see Cross-plan relationship).
- `read_task_facts` takes the task read and the project registry as arguments. It does not take the tool
  context object that the server-boundary plan's task 9823 introduces, so it has no dependency on that chain
  and reconciliation can call it.
- Cross-project dependencies (`metadata.external_deps`) are out of scope for `waits_on` in this plan.

### C2 — Binding results to tasks (`fused-memory/src/fused_memory/middleware/recall_annotation.py`, new)

```python
class Binding(StrEnum):
    STRUCTURAL = 'structural'   # a structured link says this result is about the task
    MENTIONED = 'mentioned'     # found only by scanning the text

@dataclass(frozen=True)
class BoundTask:
    ref: TaskRef
    binding: Binding            # STRUCTURAL wins when both apply
    result_ids: tuple[str, ...]

@dataclass(frozen=True)
class TaskBindings:
    tasks: tuple[BoundTask, ...]       # at most MAX_TASKS, structural first, then by first result rank
    ambiguous: tuple[str, ...]         # numbers the scan could not assign to one project; never looked up
    omitted_tasks: int                 # bound tasks beyond MAX_TASKS
    results_not_scanned: int           # results beyond MAX_RESULTS_SCANNED

def bind_tasks(results: Sequence[MemoryResult], *, project_id: str,
               known_projects: Mapping[str, str]) -> TaskBindings: ...
```

Structural links, in this plan:

1. A graph hit whose endpoint entity name parses as a task
   (`utils/canonical_labels.py::parse_node_name`). This needs `MemoryResult.entities` to be filled on graph
   hits, which β delivers.
2. A notes hit whose metadata marks it a completion record about its `task_id`: the kinds and sources that
   `fused-memory/src/fused_memory/memory_metadata.py` registers as completion records (the orchestrator's
   `task_completion` kind from `orchestrator/src/orchestrator/completion_record.py::completion_record_metadata`,
   and reconciliation's completion echo, which γ registers if it is not there). A bare `metadata.task_id`
   without such a marker records who wrote the note, not what it is about, and is not a link.
3. For the entity lookup, the returned node's own name.

A writer's declared referents become a fourth structural source in plan 2, when they are kept on notes.

Text mentions come from `canonical_labels.py::scan_content` over each result's text, with the result's project
as `group_id` and the registry as `known_project_ids`. γ fixes the scan for the measured mis-bindings:

- a registered project's name directly before "task N" ("Reify task 3674", "in reify, task 9727") yields that
  project's referent, never a local one;
- digits followed by letters ("task 8c7e163fcf") are not a task number;
- "tasks A, B and C" yields every listed number.

The scan is shared with the write path (`utils/referent_resolution.py::resolve_referents`,
`server/entities_gate.py::entities_gate`), so these fixes change write-time referent derivation too; γ runs
those suites and records the before/after on the corpus sample named in its row.

Bounds (`loop-thread-occupancy-bounded`): `MAX_TASKS = 40` (the largest briefing measured involved 28) and
`MAX_RESULTS_SCANNED = 100`; overflow is counted in the two fields above, never dropped silently.

### C3 — The `task_facts` reply field

`search`, `get_entity`, `get_entity_by_uuid`, `add_memory`, `add_episode` and `update_memory` replies carry one
top-level key:

```json
"task_facts": {
  "read_at": "2026-11-13T15:04:05+00:00",
  "tasks": [
    {"ref": "dark_factory:3651", "binding": "structural", "result_ids": ["…"],
     "lookup": "found", "status": "done", "title": "…",
     "waits_on": [], "finished": {"how": "merged", "commit": "8ca12d33"}},
    {"ref": "dark_factory:9261", "binding": "mentioned", "result_ids": ["…"],
     "lookup": "found", "status": "pending", "title": "…",
     "waits_on": [{"ref": "dark_factory:3057", "status": "pending"}], "finished": null},
    {"ref": "reify:132", "binding": "mentioned", "result_ids": ["…"],
     "lookup": "could_not_check", "reason": "project not registered"}
  ],
  "ambiguous": ["3674"],
  "omitted_tasks": 0,
  "results_not_scanned": 0
}
```

- On the three read tools the key is **always present**. With no task involved, `tasks` is `[]`. With no task
  store configured, the value is `{"unavailable": "<reason>"}`. A reader can always tell "no tasks" from "not
  checked".
- On the three write tools the key is present when the written text names at least one task (declared
  `entities` are `structural`, scanned mentions are `mentioned`, the same labels as on reads) or when the
  lookup was unavailable. `add_system_record` is not covered: system records are not about tasks (0 status
  sentences in 5,032 writes).
- Nothing is added inside a result's `metadata`
  (`orchestrator/src/orchestrator/agents/memory_recall.py::filter_foreign_project_results` drops results whose
  metadata names another project).
- The field is additive. `memory_recall.py::parse_search_reply` reads named keys only.
- `MemoryService.search` and `MemoryService.get_entity` are unchanged in shape: the join happens in the tool
  bodies and in the context assembler, not in the service.

### C4 — Rendering for a language model

One renderer, `recall_annotation.py::render_task_facts(...) -> str`, used by the reconciliation context
assembler; the orchestrator cannot import fused-memory, so `memory_recall.py` renders the same field with its
own function and the boundary suite pins that the two agree on a shared fixture (`no-lockstep-duplication`:
the shape is the contract, C3; each side owns only its formatting).

Briefing block, placed before the memory bullets:

```
**Live task status** (read from the task store at 15:04Z; status words inside the memory text below are history)
- done: 3651 "Short title" (merged 8ca12d33) · 4725 "Short title" (found on main)
- pending: 9261 "Short title", waits on 3057 (pending)
- mentioned in the text only, check the title: 3127 "Short title" (done)
- could not check: reify:132 (project not registered)
```

- Grouped by status; titles cut at 60 characters; the dispatched task itself is left out.
- If the reply carries `unavailable`, the block is one line saying live status could not be read and why.
- The own-task entity block shows each fact's date, newest first (β makes the dates arrive).

### C5 — Code that writes memory

A *code writer* is a call site outside tests that writes memory with text assembled by code rather than by a
language model. For each:

- the subject task travels as a structured link: `metadata.task_id` with a registered completion kind, or a
  declared `entities` entry;
- the text contains no task-status value: no interpolated status variable, and no word from
  `shared.task_statuses` used of a task.

One writer forwards a language model's prose: the orchestrator's decision notes
(`orchestrator/src/orchestrator/workflow.py::_write_decisions_to_memory`, the largest writer). It gains the
structured link; its text is the architect's and is not policed. Status wording in it (27 sentences in 36
days) is covered on the read side like any agent's prose.

Enforced by a behaviour test per writer (drive the writer against a recording memory double, assert both
properties on what it wrote) and a census test listing every code-writer call site, so a new one fails until
it is listed with its test (`guards-exercise-behaviour`, `exceptions-owned-or-ratified`).

### C6 — What is retired, and what is kept

Retired by ι (the task-count snapshot stops being written, every project):

- the writer `reconciliation/stages/task_knowledge_sync.py::_write_task_count_snapshot` and its pruning;
- the per-project switch `reconciliation/policies/__init__.py::SNAPSHOT_WRITE_BLOCKED_PROJECTS` and each
  policy module's use of it;
- the cadence alarm that fires when a snapshot is stale (`reconciliation/task_count_snapshot_cadence.py` and
  its escalation in `reconciliation/harness.py`), which would otherwise fire for ever once the writer is gone;
- the places that hand the snapshot to a stage (`prompts/stage3.py`, `prompts/judge.py`,
  `recon_self_model.py`, `standing_decision_constants.py`);
- the stored snapshot records, at most one per project, deleted by a script ι ships and ω runs once per
  registered project. Without it a stale count would stay after the code that handled it is gone.

ι touches neither `server/tools.py` nor the snapshot grammar, so it has no prerequisite and can land first.

Retired by κ (status-sentence detector, sweep, gates, and what was left of the snapshot):

- `reconciliation/stale_status_snapshot_edge_sweep.py` and its call from Stage 1
  (`stages/memory_consolidator.py`, with the four sweep stats it reports);
- the per-task sweep on every task finish (`reconciliation/targeted.py`'s call of
  `sweep_stale_status_snapshot_edges_for_task`, through `TargetedReconciler._retire_stale_status_snapshots`);
- the status grammar in `reconciliation/status_snapshot_extraction.py`; pieces other modules import for
  reasons that are not task status (`citation_verifier.py`, `stale_priority_override_edge_sweep.py`,
  `task_filter.py`) move to the module that owns their meaning, for example the edge de-duplication helper
  the priority sweep shares;
- four reconciliation-only write gates, each present in both `server/tools.py::add_memory` and
  `::add_episode`: live status framed as current (`ReconLiveTaskStatusWriteRejected`), conflicting status
  claims (`ReconConflictingTaskStatusWriteRejected`), premature completion (`::_premature_completion_block`),
  and the count-snapshot gate (`ReconSnapshotWriteRejected`);
- the count-snapshot grammar `reconciliation/task_filter.py::is_count_snapshot` and the count-snapshot flag
  filter in `reconciliation/flag_dedup.py` and `policies/know_live.py`;
- the Stage-1 prompt's instructions to find, invalidate or write task-status facts
  (`reconciliation/prompts/stage1.py`), replaced by one sentence: live task facts are shown beside memory and
  status text in memory is history, to be left alone.

κ deletes this code at whatever path it sits at its base; other plans may have moved it by then.

Kept:

- the mixed-temporal-framing gate (`ReconMixedFramingWriteRejected`): it is not about task status;
- the completion-claim check (`server/tools.py::_completion_claim_gate`,
  `services/completion_claim_gate.py`): it tags and never refuses, it also covers commits and tickets, and it
  is today's only path that raises a human flag on a contradicted claim. Plan 2 decides its task arm when the
  declared-relation flag exists. This keeps a human escape hatch inside plan 1.
- existing status text in both stores. It is annotated at read time, not swept.

## Resolved design decisions

- **D1 — Join at the server boundary and in the context assembler, not in the memory service.** Machine
  readers of memory need raw results; language-model readers need the join. Ruled (decision 1).
- **D2 — Structure first, text second, and say which.** A text-only match is printed as "mentioned" with the
  title, so a wrong match is visible. Ruled (decision 1b).
- **D3 — Facts attached:** status, title, unfinished dependencies, how finished and commit. Escalation and
  merge-queue state are not attached (they live in other packages). Ruled (decision 2).
- **D4 — No wording rules on writes.** Structural rule for code, live facts echoed to every writer, guidance
  for agents. No language-model check. Ruled (decision 3).
- **D5 — The task-count snapshot is retired for every project.** Ruled (decision 3b).
- **D6 — Existing material is left and annotated; the detector, sweep and status gates are retired once the
  join is live on every language-model read path.** Ruled (decision 6). The dependency edges of κ encode the
  test.
- **D7 — The facts read replaces the nested status helper** instead of standing beside it, so there is one
  registry walk with one three-valued meaning (`no-lockstep-duplication`, *SPOT*).
- **D8 — New modules live in `middleware`**, the lowest layer that can read the task store and that both
  `server` and `reconciliation` may import.
- **D9 — The decision area has one partition** (Leo's 2026-11-12 principle): code does the join, the echo and
  the structural rule; no language-model judgement is used; the human escape hatch in this plan is the kept
  completion-claim flag, and in plan 2 the flag on declared relations.

## Pre-conditions for activating

- None on substrate: everything this plan reads exists on main (see the capability manifest at decompose).
- **Contention.** Open tasks declaring the files this plan edits (2026-11-13): `server/tools.py` 148, the
  task-store middleware 74, the memory service 64, the orchestrator's workflow file 264, the briefing
  renderer 6. The plan keeps its edits to those files to call sites; the logic lives in two new modules.
- **Cross-plan edges must be settled before `commit_planning`.** The filed review-all tasks listed below move
  or refactor code this plan deletes or replaces. Applying them edits filed tasks of a finished program, which needs Leo's explicit go. The requested changes
  are recorded for that program's lead in `~/.claude/review-all/review-all-dark_factory-20261008/digests/leo-rulings-fused-recall.md`. Decompose
  files this plan's batch in planning mode and does not commit it until each requested edge is applied or
  explicitly declined, so that neither side builds on code the other is deleting.

## Cross-plan relationship

| Other plan or task | Direction | Seam | Owner | Status |
|---|---|---|---|---|
| Server-boundary plan (`plans/review-all-s15-fused-memory-mcp-server-boundary-prd.md`), ε3 task 9829: moves the memory-read tools out of the tool registration and turns `_batched_task_statuses` into `server/task_status_facts.py::batched_task_statuses` (88 open ancestors) | this plan produces early | the lifted status read; the `search` and `get_entity` bodies | this plan (α, δ); 9829 is asked to depend on α and δ and to drop the lift | requested |
| Same plan, γ task 9833 (ordered gate list naming the gates κ retires) and ε4 task 9830 (moves the memory-write tools and the gates' helpers) | this plan deletes rows and helpers | gate-list membership; `add_memory` reply | 9833 and 9830 are asked to depend on κ (and 9830 on η) and to drop the four retired gates | requested |
| Same plan, σ task 9839 (typed per-store error map on write replies), ι task 9834 (tool descriptions) | both add to the same replies and descriptions | reply shape; tool-surface snapshot | each its own key; 9839 is asked to depend on η | requested |
| Task 6599 (the premature-completion rule has two homes) | made moot by κ | — | cancel as superseded at decompose | at decompose |
| Reconciliation sweeps plan (`plans/review-all-s13-recon-sweeps-flag-identity-and-recon-report-state-prd.md`), η task 9739 (185 open ancestors): one primitive for both stale-edge sweeps; the status sweep stops writing; rewrites the Stage-1 prompt's status sections | this plan deletes the status sweep | `stale_status_snapshot_edge_sweep.py`, the Stage-1 prompt's status sections | 9739 is asked to depend on κ and shrink to the priority sweep | requested |
| Task 7941 (public seams for the status detector and sweep; a dependency of 9739) and task 5272 (half about the sweep) | made moot by κ | — | 7941 cancelled as superseded at decompose; 5272 re-read at decompose | at decompose |
| Same plan, ι task 9743 (moves snapshot I/O into a new module), ν task 9747 (its constructor keyword), λ task 9748 (per-project snapshot switch), and tasks 9851 and 9797 (the stale-snapshot alarm) | this plan deletes what they move or tune | task-count snapshot writer, switch and alarm | each is asked to depend on ι and drop the snapshot part | requested |
| Tasks 9788, 9796, 9814 and 9877 (declare `task_count_snapshot_cadence.py`) | this plan deletes the file | the cadence module | each is asked to depend on ι and drop the file from its scope | requested |
| Same plan, δ1 task 9735 (moves the prose grammar), δ2 task 9736 (re-derives the status word lists), θ2 task 9741 (moves the count-snapshot filters), and task 9770 (moves the same files to a new package) | κ deletes much of what they move | status and count grammar, count filter | their lead chooses: trim them and make them depend on κ, or let them land and κ deletes at the new paths. κ is written to work either way | lead's choice |
| Same plan's acceptance record (task 9753, cancelled) for finding `fk-9b76dd8a1cdb` "status mirrored as Graphiti prose" | this plan ends the condition it accepted | the detector as backstop | this plan (κ carries the finding key) | superseded by D6 |
| Task 9309 (a status-count read on the same three task-store files as α) | same files | `sqlite_task_backend.py`, the protocol, the interceptor | file locks serialise; no edge | compatible |
| Memory write-path plan, task 9174 (splits `MemoryService.search` into steps), task 9203 (backends private; sweeps' read surface), task 9215 (one home for an edge's stored meaning) | same file as β | `memory_service.py`, `graphiti_client.py` | file locks serialise; no edge | compatible |
| Task 3137 (full provenance and project scope on every search hit) | same reply | search result fields | independent keys; no edge | compatible |
| Tasks 7427 and 8995 (briefing entity-block summary cap; briefing follow-ups) | same functions as ε | `memory_recall.py` renderers | file locks serialise; no edge | compatible |
| Task 6826 (the qualified-reference pattern reads "word: N" as a project reference) | same file as γ | `canonical_labels.py` | independent fix; no edge | compatible |
| Task 7959 (open) and tasks 3888, 4588, 4387 (deferred), all about the status detector or sweep | made moot by κ | — | decompose re-reads each and cancels those wholly about retired code, with a pointer here | at decompose |
| Tasks 3139 and 3140 (re-examine-when metadata; flag on task finish), task 3254 (memory hints) | plan 2 | declared relations | plan 2 | out of scope |

## Decomposition plan

Sizes are expected changed lines. Same-file leaves are serialised by the edges shown.

- **α — One batched task-facts read, replacing the nested status helper.** fused-memory:
  `backends/sqlite_task_backend.py`, `backends/task_backend_protocol.py`, `middleware/task_interceptor.py`,
  `middleware/task_facts.py` (new), `server/tools.py` (two callers), tests. ~800. Contract C1.
  Intermediate: unlocks γ, δ, η. Its own check: the entity-mint guard and the completion-claim path behave as
  before on the three outcomes (existing suites), now through `read_task_facts`. Prereqs: none.
- **β — Search hits name their graph endpoints; entity edges carry their dates.** fused-memory:
  `services/memory_service.py` (`_search_graphiti`, `_edge_to_dict`), `backends/graphiti_client.py` (the
  valid-edges read returns `valid_at`/`invalid_at`; a batched node-name read by id), tests. ~500.
  Intermediate: unlocks γ (structural link 1) and ε (dated own-task block). Prereqs: none.
- **γ — Bind recalled results to tasks; fix the mention scan.** fused-memory:
  `middleware/recall_annotation.py` (new: `bind_tasks`, `render_task_facts`), `utils/canonical_labels.py`,
  `memory_metadata.py` (completion-record registry), tests incl. the write-path referent suites. ~900.
  Contract C2, C4. Intermediate: unlocks δ, η. Prereqs: α, β.
- **δ — Read replies and reconciliation's stage context carry live task facts.** fused-memory:
  `server/tools.py` (`search`, `get_entity`, `get_entity_by_uuid`), `reconciliation/context_assembler.py`,
  tool descriptions, tests. ~700. Contract C3. The context assembler already holds a task read
  (`ContextAssembler.taskmaster`); δ hands it the project registry at construction and calls the same two
  functions the tool bodies call, in `_format_memory_result`'s callers. Signal: an MCP `search` whose results name a finished task
  returns `task_facts` with that task's status from the store, and returns `{"unavailable": …}` with the task
  store detached; the assembled Stage-1 context for a memory naming a task contains the rendered block.
  Prereqs: α, β, γ.
- **ε — Briefings print the live task status block and dated own-task facts.** orchestrator:
  `agents/memory_recall.py`, `agents/briefing.py` if the block's placement needs it, tests. ~500.
  Contract C4. Signal: the `# Context` block of a dispatched task's briefing shows the block for tasks its
  recalled memory involves, and the one-line "could not be read" form when the reply says unavailable.
  Prereqs: δ.
- **η — Write replies echo live task facts; guidance tells writers what to record.** fused-memory:
  `server/tools.py` (`add_memory`, `add_episode`, `update_memory`),
  `utils/referent_resolution.py::render_referent_declaration_guidance`;
  docs: `skills/memory/SKILL.md`, `CLAUDE.md` (memory section; and correct its statement that the task-search
  corpus leaves out deferred tasks, measured false), tests. ~500. Contract C3. Signal: an `add_memory` whose
  text says a finished task is pending returns `task_facts` showing it done. Prereqs: α, γ, δ (same file).
- **θ — Code writers carry the task as structure and write no status value.** orchestrator:
  `workflow.py` (`_write_decisions_to_memory` declares the task); fused-memory: `reconciliation/targeted.py`
  (completion echo, verdict notes), `reconciliation/scope_freshness.py`, `reconciliation/harness.py`
  (`_reconcile_status_correction`); the per-writer behaviour tests and the census test in both packages.
  ~700. Contract C5. Signal: the census test fails on an unlisted code-writer call site; a completion echo
  written by a real task-done transition in the boundary suite carries the registered completion marker and
  binds as `structural` in a following `search`. Prereqs: γ (registry), ι (same file `harness.py`).
- **ι — Stop writing the task-count snapshot, for every project.** fused-memory: the C6 ι inventory, the
  deletion script under `fused-memory/scripts/`, tests removed with their subject. ~700 (mostly deletion).
  Signal: a full reconciliation cycle on a project where the snapshot was on writes no snapshot record and
  raises no stale-snapshot alarm. Prereqs: none.
- **κ — Retire the status-sentence detector, the stale-status sweep and four status write gates.**
  fused-memory: the C6 κ inventory, docs that describe the sweep. ~1,400 (mostly deletion). Carries finding
  key `fk-9b76dd8a1cdb`. Signal: a Stage-1 run's report no longer carries stale-status sweep stats; a
  reconciliation-stage write of "Task N is currently in-progress" is stored and its reply carries live facts
  instead of a refusal. Prereqs: δ and ε (the join is live on every language-model read path), η (same file
  `tools.py`, and the echo that replaces the gates), θ (same file `targeted.py`), ι.
- **ω — Integration gate: the boundary suite and a live check.** Runs the Boundary-test sketch below against
  the merged tree and, once per registered project, the snapshot deletion script. Signal: the suite's rows
  pass; a live `search` and a freshly dispatched briefing show the block. Prereqs: all of the above.

Dependency edges: γ→α,β · δ→α,β,γ · ε→δ · η→α,γ,δ · θ→γ,ι · κ→δ,ε,η,θ,ι · ω→κ. Ready at once: α, β, ι.
`server/tools.py` is edited in the order α, δ, η, κ.

G7 walk (advisory at authoring): `one-fact-one-home` is the plan's purpose; `no-silent-fail-soft` is met by
C1's total result and C3's always-present key; `loop-thread-occupancy-bounded` by C2's bounds;
`readers-prove-their-producer` by ε's signal reading a reply δ produced and by θ's signal reading a record a
real transition wrote; `guards-exercise-behaviour` by C5's tests driving the writers; `no-lockstep-duplication`
by D7 and by C4's shared fixture; `corroborate-before-acting` is not engaged (nothing acts on the facts).
No waiver is needed.

Quality heuristics that bind (`docs/code-quality.md`): *SPOT* and *structured data instead of meaningful
strings* (the purpose); *deep modules with coherent narrow interfaces* and *well-defined purpose for each
entity* (two new modules, one reads facts, one binds results); *files make internal sense in isolation* and
*no file too large* (nothing new goes into `server/tools.py` beyond the call sites); *prefer stateless
interactions* (both functions are pure over their arguments and one read).

## Boundary-test sketch

| # | Scenario | Preconditions | Asserts |
|---|---|---|---|
| B1 | Search result names a finished task in its text | a stored note "Task N remains pending"; task N done with a merged commit | reply `task_facts.tasks` has N, `binding: mentioned`, `status: done`, commit; the note's text is unchanged |
| B2 | Graph hit whose endpoint is a task node | an edge on node "Task N" | N is `structural`; `entities` on the hit names the node |
| B3 | Completion record | a completion note written by the real completion path for task N | N is `structural` through the registered marker |
| B4 | Prose-qualified foreign mention | a note "Reify task N was cancelled", `reify` registered | ref is `reify:N`, looked up in reify's store; no local N |
| B5 | Unregistered project and missing id | notes naming `nowhere:5` and a local id that does not exist | `could_not_check` with reason; `no_such_task` |
| B6 | Task store detached | server built without a task backend | read replies carry `{"unavailable": …}`; the briefing prints the one-line form |
| B7 | Many tasks | results naming 60 tasks | 40 returned, `omitted_tasks: 20`, structural first |
| B8 | Unfinished task with unfinished dependency | task N pending, depends on M pending | `waits_on` has M with its status |
| B9 | Entity lookup dates | node "Task N" with three dated edges | edges carry `temporal.valid_at`; the briefing block orders them newest first with dates |
| B10 | Briefing end to end | dispatch context for a task whose recalled memory names tasks | the block appears before the bullets, grouped by status, without the dispatched task |
| B11 | Reconciliation stage context | a memory-added event whose text names task N | the assembled context contains the rendered facts for N |
| B12 | Write echo | `add_memory` text "task N is pending", N done | the reply carries `task_facts` with N done; the write is stored |
| B13 | Machine readers unaffected | the near-duplicate guard and write triage run a search | they receive plain results with no `task_facts` |
| B14 | Code-writer census | add a new code-writer call site in a fixture tree | the census test fails naming it |
| B15 | Retired gates | a reconciliation-stage write "Task N is currently in-progress" | stored, with echo; no `ReconLiveTaskStatusWriteRejected` |
| B16 | Retired snapshot | a full reconciliation cycle on a project where it was on | no snapshot record written; none remains after the script |
| B17 | The two renderers agree | one C3 fixture | fused-memory's and the orchestrator's renderings list the same tasks, statuses and groups |

## Out of scope

- Typed relations between a task and a memory (blocker, cause, fix, relevant), keeping declared referents on
  notes, tasks 3139 and 3140: **plan 2**.
- Other recall routes (similar tasks in briefings, file-to-tasks, subsystem): **plan 2**.
- The completion-claim check's task arm: plan 2.
- Escalation and merge-queue state as task facts.
- Rebuilding status history from the write journal.
- Sweeping, rewriting or deleting existing status text.
- A custom extraction prompt for the graph library.

## Open questions (tactical)

1. **Where the completion echo's marker is registered.** If `memory_metadata.py` already registers
   reconciliation's echo as a completion record, γ uses it; if not, γ adds it. Decide in γ.
2. **`_reconcile_status_correction` and the "still blocked, no change" marker.** θ's census decides for each
   whether it complies after a template change or is removed because nothing reads it. Decide in θ.
3. **Title length and block wording** in C4 are a default; ε may adjust within the block's shape.
4. **Whether `get_episodes` replies also carry the field.** Raw episode history is read by reconciliation
   stages; suggested: yes, through the same call, if δ's size allows; otherwise a follow-up.
