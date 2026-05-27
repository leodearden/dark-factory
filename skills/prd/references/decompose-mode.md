# Decompose mode — turn a committed PRD into a queued task batch

Read a committed PRD, re-walk gates, file tasks via fused-memory in a deferred batch, wire dependencies, then flip the batch `deferred` → `pending`.

> **Overlay first.** Complete Step 0 of `SKILL.md`. The overlay supplies `project_root` and `project_id` (used in every fused-memory call below), the PRD path convention, and the project memory namespace. In generic mode, `project_root` = git root; ask the user for `project_id` if memory tagging matters.

## Preconditions

- PRD is committed at the project's PRD path. Verify via `git log -1 -- <path>`.
- Fused-memory MCP is reachable (`mcp__fused-memory__get_status`).
- No prior task batch for this PRD already exists. Check via `mcp__fused-memory__search(query="<prd slug>", project_id="<project_id>")`.

If the PRD isn't committed yet, stop and tell the user to commit first (or re-run author-mode's commit step if it was just authored this session).

## Flow

### Step 1 — Re-walk gates (fast)

Author mode established G1 / G3 / G4 / G5; this is a drift check, not a re-design.

- **G1 re-check.** Every mechanism in the PRD has a named consumer. Any orphan → escalate and stop.
- **G3 re-check.** Run the overlay's substrate verifier (or the manual check) on every assumed capability. Any failure stops the queue.
- **G4 re-check.** Read the cross-PRD relationship table; flag any reciprocal-ownership statements.
- **G5 informational.** Note B vs B+H. If B+H, confirm the integration-gate task exists in the decomposition and names the boundary-test sketch as its signal.

Any failure → stop and ask the user to fix the PRD before queueing.

### Step 2 — G2 walk (the load-bearing decompose-time check)

Enumerate every task in the PRD's decomposition plan. For each:
1. **Classify** leaf vs intermediate.
2. **Find the `user_observable_signal`** the PRD wrote for this task.
3. **Find the `consumer_ref`** — the downstream task or user surface.
4. **Find the substrate-confirmed flag** — true if the task uses existing substrate, false if it queues substrate work (mark the prerequisite task in the description).

If any leaf task lacks a user-observable signal, **stop** and surface to the user. If any intermediate task has no named downstream consumer, surface it — typically the decomposition is missing an integration-gate task.

### Step 3 — File tasks (ALWAYS planning_mode=True; synchronous, curator-bypassing)

PRD-decomposition batches are the canonical use case for `planning_mode=True`. **Every task in the batch is filed with `planning_mode=True`, no exceptions.** This lands them as `deferred` so the scheduler picks nothing up before the wiring is complete and the batch is flipped together in Step 5.

`planning_mode=True` is **synchronous** and **bypasses the curator**. `submit_task` returns `{task_id, status: "deferred", planning_mode: True}` directly — there is no ticket, no `resolve_ticket` follow-up, no curator `combined` outcome. (The two-phase `submit_task` + `resolve_ticket` pattern applies only to `planning_mode=False`, which decompose mode never uses.)

For each task in the plan, in dependency order (roots first):

```
result = mcp__fused-memory__submit_task(
    title="<task title>",
    description="""<detailed description>

PRD: <prd-path> task α/β/γ/...

User-observable signal: <signal>
Consumer: <consumer_ref>
Modules touched: <list>
""",
    project_root="<project_root>",
    priority="<medium|high|critical>",
    planning_mode=True,
    metadata={
        "source": "prd-decomposition",
        "prd_path": "<prd-path>",
        "prd_task_label": "α",
        "user_observable_signal": "<signal>",
        "consumer_ref": "<consumer_ref>",
        "grammar_confirmed": True,   # or the overlay's substrate-confirmed flag name
        "modules": ["<module_path>", ...],
    },
)
task_id = result["task_id"]   # status == "deferred", planning_mode == True
```

If `submit_task` itself times out (no `task_id` returned), **don't retry**; poll `get_task` (by title, or by IDs above your last known one) to see whether the write landed asynchronously. Re-submitting on timeout risks double-filing — the curator-dedupe path is not active in planning_mode.

### Step 4 — Wire ALL dependencies (still deferred)

After all tasks have IDs (intra-batch and out-of-batch), wire **every** declared dependency before any status flip. All deps — including cross-PRD — must be real `add_dependency` edges; the scheduler doesn't read metadata.

```
mcp__fused-memory__add_dependency(
    id="<consumer_task_id>",
    depends_on="<producer_task_id>",
    project_root="<project_root>",
)
```

Wire intra-batch (Greek-letter prereqs → the IDs from `submit_task`) and out-of-batch (PRD-declared prereqs → existing task IDs elsewhere). If the decomposition specified `metadata.unblocks` reverse-deps, set those via `update_task`.

Do **not** flip anything to `pending` until every edge is in. A partially-wired batch with some tasks already `pending` lets the scheduler grab a leaf whose real prereq hasn't been wired yet.

### Step 5 — Flip the whole batch deferred → pending in one call

Flip **every task in the batch together** in a single call — never one-at-a-time, never in dependency-root order. The whole batch becomes schedulable in one atomic moment; the scheduler handles unmet-deps tasks correctly (a task with pending deps stays effectively blocked until they clear).

```
mcp__fused-memory__set_task_status(
    id="<id1>,<id2>,<id3>,...",   # comma-separated, all batch IDs
    status="pending",
    project_root="<project_root>",
)
```

If a single bulk call is rejected (e.g. payload-size cap), split into the smallest number of bulk calls that fit — still never one-at-a-time.

### Step 6 — Verify

```
mcp__fused-memory__get_tasks(project_root="<project_root>")
```

Confirm every batch task shows up as `pending`, with the expected dependencies and metadata. Print a summary table:

| PRD label | Task ID | Title | Prereqs | Observable signal |
|---|---|---|---|---|
| α | <id> | <title> | — | <signal> |
| β | <id> | <title> | α | <signal> |

### Step 7 — Hand-back

State:
- Number of tasks filed.
- Number of intra-batch and out-of-batch dependencies wired.
- Any tasks that came back `combined` (and into what).
- A note that the orchestrator does **not** currently read `user_observable_signal` / `consumer_ref` / the substrate-confirmed flag — this metadata is substrate for a future tracking-infra session.

## Error handling

- **Curator gate closed / planning_mode batch rejected.** Do **not** switch to non-planning_mode to paper over it. Wait or escalate — PRD-decomp batches are the precise case where planning_mode is correct.
- **`add_dependency` fails** because a referenced task doesn't exist. Likely the out-of-batch prereq is `deferred` or `cancelled`; check via `get_task` and resolve with the user.
- **`set_task_status` rejects with "metadata.files missing".** Decompose mode shouldn't hit this (fresh tasks); if it does, a stale entry was probably combined into one of the new tasks — investigate before retrying.

## Anti-patterns

- Don't peek behind fused-memory at underlying storage (sqlite/JSON). Fused-memory is the only supported interface for task state. If you think you need to look at storage to debug, surface to the user.
- Don't use `planning_mode=False` and then individually flip statuses to bypass the curator — that's the gameable shortcut the curator exists to prevent. If the curator is wedged, escalate.
- Don't flip tasks to `pending` one-at-a-time or in waves. Wire **everything** first, then flip the **whole batch together**.
- Don't file follow-up tasks for things the PRD already covers as Open Questions — those stay in §Open questions.

## Resumption (if decompose was started but didn't finish)

1. Search fused-memory for tasks already filed: `search(query="<prd-slug>", project_id="<project_id>", include_planned=True)`.
2. Match against the PRD's decomposition plan (by `prd_task_label` metadata or title) — what's filed, what's missing.
3. Resume at the first missing task; new ones still go in `planning_mode=True` even if siblings exist.
4. Wire **all** dependencies (re-add is idempotent) before flipping anything.
5. Bulk-flip every still-`deferred` batch task to `pending` in a single call.

Avoid double-filing — detect existing entries first.
