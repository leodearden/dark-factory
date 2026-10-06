# Phase 3: Triage and Routing — Detailed Guide

This phase answers: **what happens to each finding, and when does the next review run?**

The coordinator does all of it: it needs the task-store tools and the judgment. Every rule here is an application of `$CONTRACT` §6–§8 and §11; where this guide and the contract disagree, the contract wins.

`<project_root>` and `<project_id>` are the values pinned in SKILL.md "Pin the run". Every task call passes `project_root=<project_root>`.

## Step 1: Load the report

When Phase 3 follows Phase 2 in one invocation, the report is in hand. `--phase triage`: load the newest `review/reports/review-*.json` whose `method.extra.phases_run` lacks `3`; none → tell the user and offer `/review` or the missing phases. Triage completes that report under its own `run_id` and `as_of_sha`.

## Step 2: Phase 1 failures (operational, not findings)

A red main, a broken gate or a dead smoke check is operational breakage (contract §4), so it never enters `findings` and never gets a finding key.

For each `phase1.failures` entry classified `new` (and each failed smoke check that is a code failure, not a stopped service):

1. Look for an owner: the orchestrator files its own red-main fix-forward tasks, so `search_tasks(query="<test id> <error line>", project_root=…, score_threshold=0.6)`, then `get_task` on any plausible match.
2. Owned → write the id into `owner_task`.
3. Unowned → file it as in Step 5 with `priority: critical` when main is red, else `high`; metadata `source: "review"`, `x_finding_run: <run_id>`, no `x_finding_key`, and the R4 pair built from the test id instead of a key.

Known flakes go to the project's flake process, not to new tasks.

## Step 3: Class each finding

For every finding with verdict `confirmed` or `weakened` whose disposition is not already `fixed:<sha>` or `refuted`, set `class` per contract §6:

| Class | Test | Route |
|---|---|---|
| `mechanical` | one anchor, no design choice; a competent agent could fix it from the statement | curator ticket after the §8 dedup (Steps 4–5) |
| `structural` | spans modules, chooses between designs, changes a contract — and every split-this-file or reduce-this-number proposal | deliberation with the user (Step 6); file nothing |

Age is not a filter. A pre-existing finding is classed and routed like a new one; severity, not age, sets priority. Lean mechanical when the fix is obvious even in complex code; lean structural the moment the proposal has two credible designs.

## Step 4: Dedup — contract §8

Run step 1 for **every** finding (structural ones too, so a finding `/prd` already filed reads `filed:<id>`). Run steps 2–3 only before filing a mechanical finding. Record the deciding step in `dedup_step`.

### §8 step 1 — key lookup

`find_tasks_by_metadata(project_root, key="x_finding_key", value=<key>)` on the fused-memory MCP. Until that tool exists, use the read-only forensic query (`CLAUDE.md` §"Forensic reads of tasks.db"): the store is `<project_root>/.taskmaster/tasks/tasks.db` — never a worktree's, never the 0-byte `.taskmaster/tasks.db` decoy — and its shape comes from `python scripts/tasks_db_schema.py --project-root <project_root>` (run from a dark-factory checkout), not from memory.

```python
import sqlite3
db = sqlite3.connect(f"file:{project_root}/.taskmaster/tasks/tasks.db?mode=ro", uri=True)
LOOKUP = """
SELECT t.id, t.status, t.priority, t.title
FROM (SELECT * FROM tasks WHERE json_valid(metadata)) AS t,
     json_each(t.metadata, '$.x_finding_key') AS k
WHERE k.value = ?
"""
rows = {key: db.execute(LOOKUP, (key,)).fetchall() for key in finding_keys}
```

`json_each` with a path matches both a scalar `x_finding_key` and a list of keys. Then, per contract §8 step 1:

- **a.** A hit in `pending`, `in-progress`, `blocked`, `deferred` or `merge-deferred` → `filed:<id>`; do not file.
- **b.** A `cancelled` hit carrying `metadata.x_acceptance_reason` → `accepted:<id>`; do not file. A `cancelled` hit without it was abandoned, not accepted: treat it as absent and go on to step 2.
- **c.** A `done` hit → the finding survived re-verification at `as_of_sha`, so file it (Step 5) with `x_supersedes_task: <id>` and `(supersedes #<id>)` in the title.

A briefing `known_gaps` entry covering the finding counts only through its `accepted_by` task: read that task and apply a/b/c to it. A gap without a pointer suppresses nothing; it is already in `method.extra.briefing_defects`.

An error payload or a failed query means step 1 did not run — never read it as "no match". Neither the MCP tool nor the database usable → this run **files nothing** (contract §8). Say so in the summary and leave dispositions `open`.

### §8 step 2 — semantic lookup

`search_tasks(project_root=…, query=<paraphrase of the statement naming the anchor>, score_threshold=0.6)`. Its corpus excludes `deferred` tasks, which is why step 1 ran first. Read each plausible match with `get_task` before deciding: the same cost at the same anchor → `filed:<id>` (and stamp the key onto that task, below); a near miss → file.

Stamping a key onto an existing task: `update_task(id=<id>, project_root=…, metadata={"x_finding_key": [<existing keys…>, <key>]})` — a list write replaces, so carry the keys already there.

### §8 step 3 — the curator

The curator's own `candidate_key` dedup inside `resolve_ticket` is the last net, never the first. A `combined` result means it caught one: record `filed:<task_id>` with `dedup_step: "3"`.

## Step 5: File mechanical findings

```python
import hashlib
submit_result = submit_task(
    project_root=project_root,
    title="<imperative fix, naming the anchor>",
    description="<the statement: what is wrong, where, the measured facts>",
    details="<the proposal and how to verify the fix; the heuristic(s) by name>",
    priority=finding["severity"],                 # contract §4: severity == priority
    metadata={
        "source": "review",
        "x_finding_key": finding["key"],
        "x_finding_run": run_id,
        # "x_supersedes_task": <id>,             # only for §8 step 1c
        "files": ["<anchor's file>"],             # files only; the architect widens scope
        "escalation_id": f"review-{finding['key']}",
        "suggestion_hash": hashlib.sha256(f"{finding['key']}|{run_id}".encode()).hexdigest()[:16],
    },
)
resolve = resolve_ticket(ticket=submit_result["ticket"], project_root=project_root,
                         timeout_seconds=<see skills/_shared/ticket-failure-handling.md>)
```

`escalation_id` + `suggestion_hash` opt the filing into the curator's R4 idempotency gate, so a retried submit cannot file twice (`skills/_shared/ticket-failure-handling.md` §"R4 idempotency gate", case B).

| `resolve["status"]` | Disposition |
|---|---|
| `created` | `filed:<task_id>` |
| `combined` | `filed:<task_id>`, `dedup_step: "3"` |
| `refused` | stays `open`; record `resolve["reason"]`; do not retry |
| `failed` | stays `open`; record the reason; retry only per the shared doc's retryable matrix |

Every filed task has a concrete title, the evidence, the expected fix and the finding metadata. When one fix must land before another (wiring before the test of the wiring), add the edge with `add_dependency` after both resolve.

## Step 6: Structural findings — deliberation

File nothing for a structural finding (contract §6). Present each to the user:

```markdown
**R4 · fk-… · structural · high — Harness couples to the merge worker's internals (h7, h13)**
Finding: <statement with evidence>
Why structural: <the designs it chooses between, or the modules it spans>
Options: a) … b) … c) …
Recommendation: <labelled as yours>
```

The user's answer sets the disposition:

- **Take it to a program doc / `/prd`** → stays `open`; `/prd` later files the tasks and stamps the key (§6, §8).
- **It is mechanical after all** → re-class and file it through Steps 4–5.
- **Won't fix** → an owned acceptance task (contract §7), filed now:

  ```python
  r = submit_task(project_root=project_root, planning_mode=True, priority=finding["severity"],
                  title="Accepted: <finding title> (<key>)",
                  description="<the statement>",
                  details="Accepted by <who> on <date>. Reopen if <condition>.",
                  metadata={"source": "review", "x_finding_key": finding["key"], "x_finding_run": run_id,
                            "x_acceptance_reason": "<the reason, as the user gave it>"})
  set_task_status(id=r["task_id"], status="cancelled", project_root=project_root)
  ```

  The reason goes in at submit because a cancelled task's metadata is frozen. Never accept by deferring: a `deferred` task is postponed work (contract §7).

  Disposition `accepted:<task_id>`. If the user wants it recorded in the briefing, the `known_gaps` entry carries only `what` and `accepted_by: <task_id>` (`/review-briefing`).
- **Deferred decision** → stays `open`; say so in the summary. The next run carries it.

Interactive runs wait for the answers before Step 7. A run the user cannot attend leaves every structural finding `open`.

## Step 7: Schedule the next run — contract §11

Only a run that completed Phase 3 files the chain; a `--findings-only` run never reaches this step. Follow `skills/_shared/filing-the-trigger-chain.md` §§0–5 exactly; it holds the only copy of the calls, the form probe and the supersede order. The run-specific values are `skill = "review"` and `run_id = <run_id>`, with `<project_root>` and `<project_id>` as pinned.

- **§0** — acceptance tasks are not "filed tasks" for the chain; the operational tasks from Step 2 and the mechanical tasks from Step 5 are. No filed tasks → no chain.
- **§1** — the open-chain lookup keys on `json_extract(metadata, '$.trigger_chain.skill') = 'review'` in a non-terminal status.
- **§2** — the form probe runs in the dark-factory checkout, never in `<project_root>`.
- **§5** — an open older `/review` chain is superseded before this run files its own.
- **§5 step 2** — when `method.extra.launched_from` names the escalation that launched this run, that is the escalation resolved with "ran as <run_id>"; there is no separate resolution step.

Record the form (`final`, `interim` or `none`), the chain task ids, any superseded chain's ids and the reason for `none` in `method.extra.trigger_chain`. Verify the filed tasks with `get_task` (not `search_tasks`, which excludes uncommitted tasks). The human gate escalates at dispatch; a human runs `/review`, never the gate itself.

## Step 8: Write the summary to memory

```
add_memory(
  content="Review {run_id} ({scope}, as_of {as_of_sha:10}, since {since:10|none}): {n} findings — {high} high/{medium} medium/{low} low; filed {filed}; {already} already owned; {structural} structural for deliberation; {fixed} fixed since last run. Next run: {chain form, task ids}.",
  category="observations_and_summaries",
  project_id="<project_id>",
  agent_id="claude-review",
  entities=[]
)
```

A pattern across findings (several stubs from one coarse decomposition, one seam behind many `tests` findings) is a separate `observations_and_summaries` write.

## Final output

Render `review/reports/<run_id>.md` from the JSON (`references/run-record.md` §"Markdown rendering"), show the interactive summary from SKILL.md, then commit both files (SKILL.md "Committing the report"). The `.md` must read on its own: it is what the user shares and what the next run's human reads first.
