# Leo's rulings on plans/review-all-feedback-synthesis-2026-10-17.md (2026-10-19)

Priorities, in order: (1) getting PRDs right — PRDs are the lever that turns the whole execution pipeline; a wrong PRD makes everything downstream more expensive or negative-value; (2) Leo's time and attention to make decisions and manage sessions; (3) CPU time; (4) everything else, including tokens.

- A1 (Phase 6 tooling): (a) deliver as a PRD. (b) WIDEN `find_tasks_by_metadata` to return full rows. (c)+(d) centralise docs changes and COMBINE commits per wave through the lead/clerk until the docs fast lane in the merge queue (task 5411, pending, 30-day shadow ruled) lands; after it lands, everyone pushes through the merge queue as soon as their PRD is ready.
- A2: yes — handoff to a fresh lead session at the Phase 5 commit is the default.
- A3, A4, A6: accepted as written (A3 includes the retro critic on reify's 12 wave-1 PRDs).
- A5: calibration — BOTH: record a `calibration_disputes_verdict` beside the finding AND treat a high weakened share in the sample as a drift signal that re-verifies the high stratum.
- A7: open, pending the investigation of spawned-session design discussions; do not amend the spawn/agent sizing rule yet.
- A8: accepted, with two additions: (i) `result.md` is commonly written mid-session once something is done, then amended one or more times before the session is truly final — the watcher/archiver and the trailer must handle checkpoint vs final; (ii) investigate giving spawned sessions a mechanism to terminate themselves — Leo thinks it worthwhile; report feasibility.
- `--full` option on /review-all to skip the cost gate (the skill is expensive; Leo knows; don't hold the session for confirmation) — do it unless there is a concern.
- B1, B2, B3, B5: accepted.
- B4: it is worse than stated — every docs-direct commit advances main and tends to force a re-verify; queues are usually busy so there is nearly always a verify in flight; around the DF and reify review-all runs nothing merged for ~10 h. Rule: centralise all docs changes per wave (lead and/or clerk) and commit the whole wave once, until 5411's docs fast lane lands; then use the merge queue.
- B6–B16: not yet ruled (silence); re-present.
- §5 delivery: accepted — execute.
