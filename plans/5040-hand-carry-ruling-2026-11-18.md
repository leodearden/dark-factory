# Hand carry of merge-lane-quality κ (task 5040) once ζ (task 8528) lands

## Ruling (Leo, 2026-11-18)

Leo, in the L2 escalation-watcher session on 2026-11-18, verbatim:

> File a direct to L2 deterministic task, dependent on 8528, telling the escalation-watcher session to
> immediately (/spawn (/team hand carry of 5040))

This pre-authorises one spawned session: a `/team` hand carry of task 5040, "Extract
merge_lane/verify_dispatch.py (VerifyDispatcher) behind VerifyPort"
(`plans/merge-lane-quality-prd.md` task κ). The spawn happens as soon as task 8528 (merge-lane policy
ζ) is done.

## How it is wired

- A pure-gate task (`task_kind=deterministic`, `always_escalates=true`, no `before_done`) depends on
  8528. When 8528 is done, the gate files a born-at-L2 escalation.
- The gate carries `metadata.x_preauthorised_spawn`, whose `ruling_ref` names this section. The
  escalation watcher spawns the brief in the gate's description with no further ask
  (`skills/escalation-watcher/SKILL.md` §"Pre-authorised gate spawn").
- Task 5040 also depends on the gate, so the orchestrator cannot dispatch 5040 itself in the window
  between 8528 landing and the carry starting.
- The carry follows Leo's hand-carry procedure (pending check, defer, investigate, plan, implement,
  fresh reviewer, fix, `/merge-queue`, set done). It then resumes the gate, which releases 5040's
  dependents.
