# Counting Failure Events — Canonical Reference

This document is the single statement of one habit: **count a task's failure events in its
project's orchestrator event log before telling a story that joins them.** Three skills point here:

- [`skills/unblock/SKILL.md`](../unblock/SKILL.md) — Step 1e, feeding Step 2's agent team and Step 3's findings
- [`skills/escalation-watcher-auto/SKILL.md`](../escalation-watcher-auto/SKILL.md) — hypothesis formation before `promote_to_l2`
- [`skills/escalation-watcher/SKILL.md`](../escalation-watcher/SKILL.md) — ruling on a promoted cluster's `root_cause`

---

## The rule

Before any account joins two or more failure events of one task into a single story — "the
cap-kill and the discard happened together", "the retry failed the same way", "verify went red
twice for one reason" — list those events from that task's project's orchestrator event log, and
count them.

The rule holds wherever the account goes: an escalation's summary, detail or evidence; a
`resolve_issue` resolution; `promote_to_l2`'s `root_cause` or `evidence`; a memory write; a report
to the human.

Escalation text, iteration logs and your own earlier turns are summaries. `runs.db` is the
record. One `invocation_end` row is one agent invocation, however close together two rows sit.

## The command

```bash
python3 $DARK_FACTORY_ROOT/scripts/task_event_timeline.py --project-root <PROJECT_ROOT> <TASK_ID>
```

It lists every event of the task across all orchestrator runs, in order, then counts the events,
the runs, and the `invocation_end` outcomes by role and subtype. It is read-only. Add
`--event-type invocation_end --event-type escalation_created` to narrow the listing, and `--json`
for the full payloads.

| Exit | Meaning |
|---|---|
| 0 | events listed |
| 1 | no events for that id in that log: the project or the id is wrong. It does **not** mean nothing happened. |
| 3 | the path is not a readable event log |

## Where the log is

In the MAIN checkout of the project the TASK belongs to, at
`<main checkout>/data/orchestrator/runs.db`. A reify task's log is reify's, never dark-factory's.
It is never inside a worktree, because `data/` is gitignored, and it is never `data/runs.db`, a
0-byte decoy beside the real one. The tool prints the path it read on its first line: check that
it names the right project.

## How to report it

What you counted is an observation. Record it as one, per the escalation ladder's "Observations
vs. hypotheses" rule (`orchestrator/src/orchestrator/agents/roles.py::ESCALATION_LADDER_CORE`),
for example as an `evidence` entry:

```json
{"observation": "4 invocation_end error_max_budget_usd rows for reviewer_comprehensive 23:59–01:04, a separate 5th at 01:23 subtype=success",
 "measured_at": "runs.db ids 688852–689227",
 "ref": "task_event_timeline.py"}
```

Any causal joining of those events goes on its own `Hypothesis:` line.

---

Provenance: reify codebook entry `entry-cand-20260819-21`; dark-factory task 5974.
