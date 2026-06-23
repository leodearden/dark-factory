# PRD: Author-declared `complexity` field for the SIMPLE_TASK fast path

**Status:** active — generic-mode PRD (no project overlay). Authored 2026-06-23.
**Type:** contract resolving an existing accreted mechanism (the dead Lever-C classifier).

## Goal

Make the orchestrator's SIMPLE_TASK / "Lever C" optimistic path **author-controlled
and legible**. Today a task takes the single-agent fast path only if a hidden,
anchored title-regex classifier matches. That classifier is invisible to the
agents that author tasks and has fired **6 times ever** across all six
orchestrator-managed projects in ~6 weeks (5 reify, 1 know-live, **0 in
dark_factory**) — it is effectively dead.

Replace it with an explicit, documented metadata field: a task takes the fast
path **iff its author declared `metadata.complexity == "simple"`**. The
eligibility contract moves from a buried regex to a rubric published at the
point every task is created.

## Background

- **Mechanism today:** gate at `orchestrator/src/orchestrator/workflow.py:1507-1544`
  calls `classify_simple_task` (`agents/triage.py:46-73`), which ANDs an anchored
  title regex (`_SIMPLE_TASK_TITLE_RE`, `triage.py:26-33`), a `len(files) <= 2`
  cap, a hard-blocker description veto (`_SIMPLE_TASK_HARD_BLOCKERS_RE`,
  `triage.py:37-43`), and `priority != 'high'`. On match it runs
  `_run_simple_task` (`workflow.py:2709-2841`) — a single Sonnet SIMPLE_TASK
  agent (`agents/roles.py:1049-1130`, budget $1.50 / 30 turns) that plans via
  the plan-tools MCP, edits, commits, and marks steps done. **VERIFY → REVIEW →
  MERGE still run unchanged afterward** — only architect+implementer are
  collapsed.
- **Why it's dead:** all task creators are LLM agents driven by prompts; an
  anchored title regex they were never told about has near-zero recall.
- **Why it matters now:** high/critical tasks with wide locks that go through the
  full architect path hold their locks across plan+implement, and under the
  park-stack regime that stalls everything beneath them. One-shotting a
  genuinely-simple wide-lock task is a direct throughput win — *provided quality
  holds*, which the unchanged VERIFY→REVIEW→MERGE back half plus the
  fallthrough-to-architect backstop both enforce.

## Resolved design decisions

1. **Delete the classifier as a trigger.** Remove `classify_simple_task` and
   `_SIMPLE_TASK_TITLE_RE` from `triage.py`. There is **no** deterministic
   auto-classification: the *only* way onto the fast path is an explicit author
   declaration. (Removes the "why did this one-shot?" opacity entirely.)
2. **`complexity` metadata field, `simple`-only vocabulary.** A free-form
   metadata key (passthrough already supported — `sqlite_task_backend.py:64`
   stores `metadata` as JSON). Meaningful value: `"simple"` (opt-in). Absent or
   any other value ⇒ full architect path. `metadata.force_full_path` stays as the
   hard escape (and the auto-eval redo path keeps setting it).
3. **Gate predicate** (rewrite at `workflow.py:1514-1521`): take the simple path
   iff
   `metadata.complexity == "simple"` **AND** `simple_task_enabled` **AND** not
   `metadata.auto_eval_redo` **AND** not `metadata.force_full_path` **AND** no
   `initial_plan` **AND** the hard-blocker veto does not fire.
4. **Drop the `priority != 'high'` guard.** High/critical tasks may be simple,
   and one-shotting high-pri wide-lock tasks is the biggest throughput win.
5. **No dispatch-time size cap** (no file-count cap, no lock-footprint cap). The
   only dispatch guard is the **hard-blocker-token veto** (`migration`,
   `architecture`, `integration test`, `design …new`, `implement …new feature`
   in the description) — a cheap contradiction check against an author who
   declared `simple` but described something large. `_SIMPLE_TASK_HARD_BLOCKERS_RE`
   is **retained and repurposed** as this veto (not deleted with the rest of the
   classifier).
6. **The real bound is the runtime backstop, not a guard.** The SIMPLE_TASK
   agent self-checks "is this actually small?" and the workflow returns
   `REQUEUED → full architect path` on any doubt (`workflow.py:2741-2787`) — the
   proven, constantly-exercised default path. Because the bound is the agent +
   fallthrough rather than a tight gate, the gate can be generous. Worst case of
   a mis-declared huge task is one wasted ~$1.50 Sonnet spin-up before it falls
   through.
7. **Loosen the agent self-check to match.** The SIMPLE_TASK role prompt
   (`roles.py:1060-1062`) and briefing prompt (`briefing.py:379-393`) currently
   say "if it grows beyond ~2 files of meaningful change, STOP." Replace the
   numeric "~2 files" bar with a **qualitative, footprint-agnostic** one: stop
   and fall through if the change needs cross-module design, a new abstraction,
   or substantial architectural thought — *not* merely because it spans several
   files or modules. This is what lets wide-but-simple tasks complete on the fast
   path instead of being bounced by the agent itself.
8. **Legibility is the deliverable.** Publish the `complexity` field + a tight
   "when to declare `simple`" rubric at the single chokepoint every task creator
   passes through — the fused-memory `submit_task` tool description
   (`server/tools.py:2421`) — and mirror it into the orchestrator's shared
   role-prompt `submit_task` instruction builder (`roles.py:591-674`) and into
   `CLAUDE.md`'s conventions.

### The rubric (canonical text, to be published verbatim at the chokepoints)

> **`complexity`** *(optional)*: set to `"simple"` to route this task to the
> single-agent fast path (one Sonnet agent explores, plans, edits, and commits;
> the architect+implementer pair is skipped, but verify/review/merge still run).
> Declare `"simple"` only when the change is a **single coherent edit** — docs or
> comments, a rename, a localized behaviour-preserving refactor, a typo/wording
> fix, a one-spot bug fix — that needs **no new abstraction and no cross-module
> design**, and you can name the target file(s). It is fine for a `simple` task
> to be high priority or to touch several files/modules, as long as the *change*
> is mechanically simple. **When unsure, omit it** — the full path is the safe
> default, and a mis-declared task simply falls back to the architect.

## Pre-conditions for activating

None. The `complexity` key is inert until α lands; no migration is needed
(field-absent == today's behaviour). β/docs depend on α so they don't advertise
a field that does nothing yet.

## Cross-PRD relationship

No cross-PRD seams. The orchestrator↔fused-memory boundary is an existing
free-form `metadata` passthrough, not a new contract — G4 N/A.

## Decomposition plan

Approach **B** (blast radius = 2 packages, ~1 mechanism). Two tasks, one
dependency edge.

- **α — Orchestrator: author-controlled simple-path routing + agent loosening + orchestrator-side rubric.**
  - **Modules:** `orchestrator` (+ repo-root `CLAUDE.md`).
  - **Files:** `orchestrator/src/orchestrator/agents/triage.py`,
    `orchestrator/src/orchestrator/workflow.py`,
    `orchestrator/src/orchestrator/agents/roles.py`,
    `orchestrator/src/orchestrator/agents/briefing.py`, `CLAUDE.md`.
  - **Work:** delete `classify_simple_task` + `_SIMPLE_TASK_TITLE_RE`; keep
    `_SIMPLE_TASK_HARD_BLOCKERS_RE`, expose it as a small veto predicate
    (e.g. `has_simple_task_blocker(description)`); rewrite the gate to the
    decision-3 predicate (trigger on `complexity == 'simple'`, drop priority and
    size caps, retain the blocker veto + existing outer conditions); loosen the
    SIMPLE_TASK self-check in `roles.py` + `briefing.py` per decision 7; add the
    rubric to the shared `submit_task` instruction builder in `roles.py`; add a
    `complexity` conventions note to `CLAUDE.md`.
  - **Observable signal:** a task submitted with `metadata.complexity='simple'`
    (no blocker tokens) is routed to the single-agent SIMPLE_TASK path — the run
    emits a `phase_skipped` event with `reason='architect_skipped_simple_task'`
    and **no** architect `plan` phase runs; a task **without** the field runs the
    full architect path (a `plan` phase, no `phase_skipped`); a task declaring
    `complexity='simple'` whose description contains a blocker token (e.g.
    "migration") runs the **full** path (veto fires). Observable via the run
    event store (`data/orchestrator/runs.db`, `event_type='phase_skipped'`) and
    the run log.
  - **Prereqs:** none.

- **β — fused-memory: publish the `complexity` rubric on the `submit_task` tool surface.**
  - **Modules:** `fused-memory`.
  - **Files:** `fused-memory/src/fused_memory/server/tools.py`.
  - **Work:** add the canonical rubric (above) to the `submit_task` tool
    description/docstring at `:2421` so every agent that lists/sees the tool reads
    the `complexity` field and when to set it.
  - **Observable signal:** the live `submit_task` MCP tool description (as
    surfaced to agents) advertises the `complexity` field and its "when to
    declare simple" rubric — assertable by inspecting the rendered tool
    description / a test that the description string contains the field + rubric.
  - **Prereqs:** α (don't advertise a field before the gate honours it).

## Out of scope

- **Auto-eval is explicitly NOT a dependency or a quality gate.** The
  `auto_eval_dispatched` event has fired **0 times ever** across all six
  orchestrator projects — the forensic full-architect A/B redo net
  (`harness.py:3306-3479`) is unproven in production. Quality on the fast path is
  held by the unchanged VERIFY→REVIEW→MERGE back half plus the
  fallthrough-to-architect, **not** by auto-eval. Validating/repairing the
  auto-eval hook is a separate follow-up PRD.
- **The `/prd` skill mirror is out-of-band.** The skill lives in
  `~/.claude/skills/prd` (outside the repo), so an orchestrator task cannot edit
  it via the merge queue. Mirroring the rubric into the skill's task-authoring
  guidance (and/or creating a project prd overlay) is a manual change tracked
  separately.
- No change to the SIMPLE_TASK agent's budget/turns or to `config.py`
  (`simple_task_enabled` already exists as the kill switch and is retained).
- No new `complexity` tiers beyond `simple` (decision 2).

## Open questions (tactical — defer to impl)

1. **Blocker-veto token list.** Keep the current set
   (`architecture|migration|integration test|design …new|implement …new feature`)
   or trim to reduce false vetoes (e.g. "simplify the migration helper" would be
   vetoed). **Suggested:** keep as-is; tune from observed false vetoes. Decide
   during α.
2. **Exact wording of the loosened agent self-check** in `roles.py` /
   `briefing.py`. **Suggested:** the decision-7 phrasing. Decide during α.
3. **Whether to assert the rubric text in a test** (β) or rely on inspection.
   **Suggested:** a lightweight test that the tool description contains the field
   name + a rubric marker. Decide during β.
