# Offline-lane red path — 7-day soak (gate 6868, esc-6868-1)

Operator gate ζ of `plans/offline-lane-red-path-prd.md`. Runbook: task 6868's description.
Measured by the L2 escalation-watcher. Pre-measurement 2026-11-07 ~08:15Z; final run 2026-11-10 ~08:10Z.
Leo ruled on 2026-11-09 ~18:00Z and 2026-11-10 ~08:05Z.

## T (runbook step 0)

T = **2026-11-01 07:57Z**, when the full red path went live, lane ledger rows (ε2) included.
- **Not T, 2026-10-24 09:10Z:** this is only the oldest record surviving 14-day pruning.
- **Not T, the first effect-block record (10-17/18):** it carries no ledger rows.

At the final run the soak was 9.0 days old, past the 7-day bar.

## Mechanical checks (step 1)

`scripts/offline_lane_soak_report.py --project-root /home/leo/src/dark-factory --since 2026-11-01T07:57Z`, run 2026-11-10: **exit 0**.

| Check | Verdict | Examined |
|---|---|---|
| records_well_formed | PASS | 885 |
| filings_resolve | NO_DATA | 0 |
| filings_bounded_by_threshold | NO_DATA | 0 |
| ledger_rows_per_pass | PASS | 24 |
| deflake_tasks_carry_evidence | PASS | 13 |
| no_stranded_filing | NO_DATA | 0 |
| offered_debts_owned_or_due | PASS | 24 |

The three NO_DATA rows are vacuous: no lane fix task was filed since T, because no test reached `persist_passes`.

**Interrupted passes.** Three passes were interrupted, all `warm-lane-bash`: 20261101T075701 (the go-live minute), 20261107T091442 and 20261107T205548. These are the known PRD §5.8 restart gap, counted and not failed.

**Script defect, not a leaf defect.** Without `--since`, the script defaults to the earliest effect-block record. That record predates ε2, so `ledger_rows_per_pass` fails on 49 pre-ε2 passes. The default T should be ε2's go-live.

## Figures since T (step 3)

**Passes per day** (qdrant-integration / warm-lane-bash):

| Date | qdrant-integration | warm-lane-bash |
|---|---|---|
| 11-01 | 33 | 34 |
| 11-02 | 46 | 46 |
| 11-03 | 44 | 43 |
| 11-04 | 44 | 45 |
| 11-05 | 59 | 59 |
| 11-06 | 60 | 60 |
| 11-07 | 47 | 47 |
| 11-08 | 32 | 32 |
| 11-09 | 58 | 58 |
| 11-10 (partial) | 19 | 19 |

**Episodes.** 3 ended, all passed at length 1. None is live, and none reached `persist_passes` 2.

**Real breaks and filing delay.** There was no real break in the window, so no filing delay could be measured.

**Lane debts.**
- 24 offered across 13 owner tasks: 8110, 8190, 8191, 8231, 8246, 8350, 8352, 8353, 8354, 8436, 8437, 8748 and 8749. Every owner is **done**.
- Clearing passes: 15 on qdrant-integration (11-01 to 11-04, plus one on 11-08) and 2 on warm-lane-bash (11-01 and 11-02).

**Due list.** It is empty, and no due item ever went stale (more than 30 minutes).

**Held reds ended at another head.** None.

**Regression escalations at the lane call site.** One, `esc-flake-regressed-after-resolution-1` (`test_warm_lane_gc.sh`, cycle 2). Leo dismissed it: the cause was an external SIGTERM (task 8231).

**Outages.** No outage passes. The qdrant-integration probe was up 16 of 16 times. Before T, in the 14-day pre-measurement window, there was one outage: an OpenAI rate limit on 10-22 that self-closed after 48 minutes.

**Record directory.** 49,862,647 B in the report (61 MB on disk on 11-10), about 3.5 MB/day.

**Questions the script cannot answer (PRD §12 Q5, Q7, and the PRD's report question).**
- **Path-scope advisory:** no filed digest tripped fused-memory's path-scope advisory. This is a weak test, because digests carry only short failure lines.
- **Live test from a worktree:** a de-flake agent did run a live integration test from its worktree (task 8072 passed). OPENAI_API_KEY reached it only through graphiti loading the repo `.env`, not through the agent shell.
- **Digest record paths:** a sandboxed agent can read the record paths a digest names (tasks 8353 and 7379's investigator did), until 14-day pruning removes them.

## Rulings (Leo)

**(i) Knobs.** Keep all three:
- `git.offline_lane_persist_passes` = 2. Evidence: it held back about 5 single-pass non-breaks, and a restart-killed run would each have filed at 1.
- `git.offline_lane_recheck_secs` = 1800. Evidence: both observed rechecks, at about 31 minutes, ended the held red.
- `git.offline_lane_red_unlanded_hours` = 24. Evidence: it fired once, correctly, 78 h after task 7379.

**Retention.** Lane records are kept **60 days**, up from 14. The 14-day window lost the lane's first 6.5 days. Ticketed as `tkt_0RWSS0EQEB56BE09QT2FFK4F2T`, awaiting curation.

**(ii) The three D6 author's calls.** Keep all three:
- **The lane stays out of the merge-gate storm alarm.** A replay would have raised about 10 critical L2s in 14 days, while all 13 lane debts were owned and fixed.
- **A recovered test needs proof at one head.** The only different-head held red was a missing-module break fixed by the next merge. Caveat: the `recovered` trigger has not fired since ε2.
- **No flake debt while a covering fix task is open.** This has never been exercised.

**(iii) Live-test skip rule** (2026-11-10, option A): accept main as it stands.
- **The rule:** since tasks 8748 and 8749 (merged 2026-11-09), a live OpenAI-SDK test skips only when the provider's connect phase timed out before any response (`fused-memory/tests/_unreached_provider.py::skip_if_provider_unreached`).
- **What fails instead:** every other failure, a read timeout or an over-budget judge call included, fails and goes through flake debt. Task 8436's earlier over-budget skip is gone.
- **Lasting outages:** these are `fused-memory/scripts/lane_dependency_probe.py`'s to report.
- **Rejected:** counting skips as no-signal debts (B) and forbidding skips in the lane (C).

**Amended 2026-11-11 (Leo, on esc-flake-regressed-after-resolution-4 through -11).** A 00:09Z OpenAI connection blip failed 15 live tests in pass 20261111T000934-ba1793, and all 15 cleared on confirm.
- Most of those failures were a connect-phase `ConnectError`, which the rule above does not cover. The frontier-arm test's failure was swallowed by the judge's fail-open, so the skip never saw it.
- Leo widened "never reached" to the whole connect phase (ConnectError as well as ConnectTimeout), and to a judge fail-open whose cause was an unreached provider. Failures after connect still go to flake debt.
- The work is ticket `tkt_0RWVSAA56BDDHW4C0NN5ZCRXEH`, awaiting curation. The 15 flake-debt tasks from that pass (9290-9304) were cancelled and the 8 escalations closed.
