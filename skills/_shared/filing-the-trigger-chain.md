# Filing the trigger chain

The last step of every attended `/review`, `/hotspot-survey` and `/review-all` run.
Normative behaviour: `docs/quality-findings-contract.md` §11. Mechanism and rationale:
`plans/completion-driven-triggers-prd.md` (C1–C6). This file holds only the calls; the
skills cite it and do not restate it.

## 0. Preconditions

- Every task this run filed has a task id (resolve each curator ticket first), and each
  carries `metadata.x_finding_run = "<run_id>"`.
- If the run filed **no** tasks, file no chain. Write "no trigger chain: the run filed no
  tasks" under `extra` in the report's `## Method` block.
- Chain tasks never carry `x_finding_run`. They carry `trigger_chain` instead.
- `<skill>` below is the bare skill name with no slash (`review`, `hotspot-survey`,
  `review-all`); the templates add the slash where a human reads it.

## 1. Check for an open chain

One chain per (project, skill) may be open. Before filing, look for tasks carrying
`metadata.trigger_chain` for this skill in any non-terminal status (the
`shared.task_statuses.ACTIVE` set, which includes `deferred`): once it lands,
`find_tasks_by_metadata(project_root, key="trigger_chain", value=None, statuses=<non-terminal>)`
filtered on `matched_value["skill"] == "<skill>"` and no `superseded_by` (the tool matches
top-level keys only); until then the read-only forensic query of `CLAUDE.md` §"Forensic reads
of tasks.db" on `json_extract(metadata, '$.trigger_chain.skill')`. If one exists, this run supersedes it
(§5) before filing its own. Once the `plans/completion-driven-triggers-prd.md` guard
lands, `submit_task` refuses a second open chain with `error_type: TriggerChainOpen`
and the same §5 applies.

## 2. Pick the form

Use the **final form** only if both hold: (i) the dark-factory checkout (the factory root,
not the target project) has an executable `scripts/check_run_completion.py` (`test -x`),
which proves the re-arm verdict is on main; and (ii) the running fused-memory's
`get_status` returns `task_metadata_capabilities` containing `before_done.recheck_secs`,
`before_done.script_root` and `trigger_chain`, which proves the running guard has it
(`plans/completion-driven-triggers-prd.md` C2). Otherwise, or if a final-form
`submit_task` returns `error_type: UnsupportedTaskMetadata`, use the **interim form**.

## 3. Final form (two tasks)

```python
completion = submit_task(
    project_root="<project_root>",
    title="Completion gate: <run_id> weighted share >= 0.7",
    description="Re-arms daily until the priority-weighted share of the tasks filed by <run_id> reaches 0.7 (docs/quality-findings-contract.md §11), then lets the human gate fire.",
    priority="medium",
    planning_mode=True,
    task_kind="deterministic",
    agent_id="<skill>-<run_id>",
    metadata={
        "source": "trigger-chain",
        "trigger_chain": {"role": "completion_gate", "skill": "<skill>", "run_id": "<run_id>"},
        "milestone": {"mode": "dated", "at": "<now + 86400 s, ISO-8601 UTC>"},
        "before_done": {
            "kind": "predicate",
            "script": "scripts/check_run_completion.py",
            "script_root": "factory",
            "args": ["--run", "<run_id>", "--threshold", "0.7", "--project-root", "<project_root>"],
            "timeout_secs": 120,
            "recheck_secs": 86400,
        },
    },
)
human = submit_task(
    project_root="<project_root>",
    title="Run /<skill> on <project_id> (<run_id> landed)",
    description="Human gate: when this fires, the escalation watcher spawns an attended /<skill> session or lists it in the digest. Never run unattended.",
    priority="medium",
    planning_mode=True,
    task_kind="deterministic",
    agent_id="<skill>-<run_id>",
    metadata={
        "source": "trigger-chain",
        "trigger_chain": {"role": "human_gate", "skill": "<skill>", "run_id": "<run_id>"},
        "always_escalates": True,
    },
)
add_dependency(id=human["task_id"], depends_on=completion["task_id"], project_root="<project_root>")
commit_planning(project_root="<project_root>", task_ids=f"{completion['task_id']},{human['task_id']}")
```

Commit at once only when every task the run will file already exists. A run whose
tasks are filed by later `/prd` sessions (a program) leaves both tasks uncommitted
(`deferred`) until the program's FILED table is complete, then commits; the completion
gate's measure covers only tasks carrying `x_finding_run`, so committing early would let
it pass on the first few tickets and fire the human gate before the structural work is
filed.

## 4. Interim form (one task)

The human gate from §3, with `"milestone": {"mode": "delayed", "after_secs": 259200}` added
to its metadata, filed with `planning_mode=True` and left uncommitted (`deferred`) until its
dependencies exist. Then `add_dependency(id=<human gate>, depends_on=<task>)` for each `high`
or `critical` task the run filed — for a run whose tasks are filed by later `/prd` sessions,
each session adds its own as it files. If the run filed none of those, depend on every task
it filed. Only when the program's FILED table is complete: `commit_planning(task_ids="<human
gate>")`. A delayed milestone with no dependencies anchors at once, so the gate is never
committed bare.

## 5. Superseding an open chain

An earlier chain for this skill is still open (§1). This run supersedes it, in this order:

1. For each old chain task, `get_task`, then `update_task(id, project_root,
   metadata={"trigger_chain": {<its trigger_chain>, "superseded_by": "<run_id>"}})`.
   Write the whole dict: metadata merge is shallow.
2. The old **human gate**:
   - `blocked` → `get_task_escalations` → `resolve_issue(<pending esc>,
     action="resume", resolution=...)`, where the resolution is
     `"ran as <run_id>"` when that escalation is the one that launched this
     run (the normal case: it is a `review_due` escalation and this run is
     its answer), else `"superseded by <run_id>"`. `resume` drives the old
     gate to `done`; nothing resolves it a second time.
   - `pending` or `deferred` → `set_task_status("cancelled")`.
3. The old **completion gate** (final form only):
   - `pending` → cancel it.
   - `blocked` → `resolve_issue(<pending esc>, action="abandon", resolution="superseded by <run_id>")`.
   - `in-progress` → re-read it in two minutes.
4. File this run's chain (§3 or §4). Record the launching escalation id, if any, under
   `extra.launched_from` in the report's `## Method` block.

Do the human gate before the completion gate. A cancelled completion gate satisfies the
human gate's dependency and fires it.
