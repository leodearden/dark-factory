# D6 milestone +14d — claude-fable-5-1 as merger and steward-retry, against the D6 kill criteria

**Verdict, for Leo to ratify. No config was edited by this task.**

- **Merger: keep it on `claude-fable-5-1`.** No merger kill criterion fired.
  Turn-cap kills were 0/18, budget kills 0/18 and timeouts 0/18, against
  5/19, 0/19 and 0/19 for the opus merger in the 30 days before the apply.
  There were no drop-guard firings in either window and no conflict re-opens.
- **Steward rule `steward-retry-fable`: kill it, by the letter of the
  pre-registered criterion.** The L0 resolved-in-place share did not move.
  Fable at tier >= 1 resolved 21/24 (87.5%) in place. Opus resolved 10/11
  (90.9%) at tier >= 1 before the apply, and 178/191 (93.2%) at tier 0 inside
  the same window. The premise the rule was admitted on, that the L0 steward
  resolves "a small share", does not hold per steward-worked escalation. That
  left the criterion almost no room to move at this n (§ Verdict per item,
  (2) and (5)).
- **`timeouts.merger`: leave it at 600 s.** That value is a pre-liveness
  flat bound that did not kill a single run. The bound a live merger actually
  came near is the global `invocation_timeout` (7200 s): the longest Fable
  merger ran 7048 s and succeeded. No change is recommended to either value
  yet (§ Verdict per item, (5)).

| | |
|---|---|
| Apply commit | `526e0eba99bfc66904427426a5b1beb54afeb50d` (the anchor from `d6-day1-check.md`) |
| Review window | **[2026-09-12T06:43:16Z, 2026-09-26T06:43:16Z)**, the 14 days after the apply, half-open |
| Baseline window | **[2026-08-13T06:43:16Z, 2026-09-12T06:43:16Z)**, the 30 days before the apply |
| Measured at | 2026-09-26T11:43:13Z (scripts at commit `605c177056`) |
| Run store | `/home/leo/src/dark-factory/data/orchestrator/runs.db`, 193 138 688 bytes, mtime 2026-09-26 12:42:20 +0100, read at the measured-at time above |
| Escalation store | `/home/leo/src/dark-factory/data/escalations` (root plus `archive/<date>/`). The corpus size, skipped count and oldest archive date are the last line of the review's § 2 |
| Command (a), the review | `python scripts/review_model_admission.py --model claude-fable-5-1 --baseline-model opus --expect-roles merger,steward --apply 2026-09-12T06:43:16+00:00 --days 14 --baseline-days 30 --ceiling 150` |
| Command (b), the day-1 audit over the same window | `python scripts/audit_model_admission.py --model claude-fable-5-1 --expect-roles merger,steward --since 2026-09-12T06:43:16+00:00 --until 2026-09-26T06:43:16+00:00 --window 14d --ceiling 150 --format markdown` |

Every number in § Measurements is **generated** by the commands above and
pasted verbatim; none was transcribed by hand. The prose below cites cells
from that output and computes no new number from them. Figures quoted from
the D5 evidence package and the capacity study are marked with their source
section.

Both scripts are strictly read-only: every runs.db connection is a `mode=ro`
SQLite URI, and escalation records are only opened for reading. Both windows
are fixed and half-open, and a merge outcome or escalation disposition is
judged as of its window's end. A re-run therefore reproduces this output
exactly; two runs of command (a) made at different times produced
byte-identical output.

Both scripts were run inside the agent sandbox with `SQLITE_TMPDIR` set to a
writable directory. `/var/tmp` is not writable there, and SQLite otherwise
fails a large `ORDER BY` with `unable to open database file`. The variable
changes where SQLite spills a sort, not what it reads. Run outside the
sandbox, the commands need no environment.

How to read the tables:

- **`over flat ceiling` is not a failure column.** `timeouts.merger: 600` is
  enforced flatly only until a transcript proves liveness. From turn 1 the
  bound becomes `max(timeouts.working_idle_secs, 600) = 1800 s` as an
  **idle** bound, itself capped by the absolute `invocation_timeout: 7200 s`
  (see `orchestrator/src/orchestrator/workflow.py::TaskWorkflow._invoke`,
  which passes `working_idle_secs` and
  `absolute_cap_secs=self.config.invocation_timeout` for every role). The
  `timed out` column is the producer's own kill verdict.
- **`resolved (merge done)`** is the last `merge_finalized` inside each merger
  run's own attribution window (see
  `scripts/audit_model_admission.py::_merge_outcome`), so a merge that a later
  merger run retried is not credited to the earlier run.
- A **steward arm's dispositions** count each distinct escalation once, and
  the shares are over the decided ones. `record missing` is an escalation the
  store no longer holds (see the coverage caveat in (2)).

## Measurements

**Command (a): the review.**

Candidate `claude-fable-5-1` against baseline `opus`. Review window [2026-09-12T06:43:16+00:00, 2026-09-26T06:43:16+00:00); baseline window [2026-08-13T06:43:16+00:00, 2026-09-12T06:43:16+00:00). Steward arms split at dispatch tier 1. Per-model daily ceiling: $150.00.

Rates read k/n (x%). Percentiles are nearest-rank, so each is an observed run's value. Durations are in seconds. '-' is unknown: no end event, no dispatch decision or no attributed merge.

### 1. Merger: `claude-fable-5-1` in [2026-09-12T06:43:16+00:00, 2026-09-26T06:43:16+00:00), against `opus` in [2026-08-13T06:43:16+00:00, 2026-09-12T06:43:16+00:00) and in [2026-09-12T06:43:16+00:00, 2026-09-26T06:43:16+00:00)

| arm | model | role | window | tier filter |
|---|---|---|---|---|
| claude-fable-5-1 merger, window | claude-fable-5-1 | merger | [2026-09-12T06:43:16+00:00, 2026-09-26T06:43:16+00:00) | any |
| opus merger, before apply | opus | merger | [2026-08-13T06:43:16+00:00, 2026-09-12T06:43:16+00:00) | any |
| opus merger, window | opus | merger | [2026-09-12T06:43:16+00:00, 2026-09-26T06:43:16+00:00) | any |

| arm | runs | succeeded | turn-cap kills | budget kills | timed out | no end event | resolved (merge done) | over flat ceiling |
|---|---|---|---|---|---|---|---|---|
| claude-fable-5-1 merger, window | 18 | 18/18 (100.0%) | 0/18 (0.0%) | 0/18 (0.0%) | 0/18 (0.0%) | 0/18 (0.0%) | 14/18 (77.8%) | 17/18 (94.4%) |
| opus merger, before apply | 19 | 14/19 (73.7%) | 5/19 (26.3%) | 0/19 (0.0%) | 0/19 (0.0%) | 0/19 (0.0%) | 15/19 (78.9%) | 15/19 (78.9%) |
| opus merger, window | 0 | 0/0 (-) | 0/0 (-) | 0/0 (-) | 0/0 (-) | 0/0 (-) | 0/0 (-) | 0/0 (-) |

| arm | cost total $ | cost p50 $ | cost p95 $ | turns p50 | turns p95 | duration p50 s | duration p95 s | duration max s |
|---|---|---|---|---|---|---|---|---|
| claude-fable-5-1 merger, window | 84.26 | 4.58 | 6.87 | 31 | 50 | 1281 | 7048 | 7048 |
| opus merger, before apply | 48.52 | 2.51 | 3.86 | 48 | 52 | 1103 | 3122 | 3122 |
| opus merger, window | 0.00 | - | - | - | - | - | - | - |

| arm | merge states | dispatch max_turns |
|---|---|---|
| claude-fable-5-1 merger, window | blocked: 2, done: 14, superseded: 1, -: 1 | 100: 18 |
| opus merger, before apply | already_merged: 1, blocked: 1, done: 15, superseded: 2 | 50: 18, 100: 1 |
| opus merger, window | none | none |

Drop-guard firings in [2026-09-12T06:43:16+00:00, 2026-09-26T06:43:16+00:00):

_none_

Drop-guard firings in [2026-08-13T06:43:16+00:00, 2026-09-12T06:43:16+00:00):

_none_

Conflicts finalized after a successful `claude-fable-5-1` merger run, before 2026-09-26T06:43:16+00:00:

_none_

### 2. Steward and the L2 tier: `claude-fable-5-1` in [2026-09-12T06:43:16+00:00, 2026-09-26T06:43:16+00:00), against `opus` in [2026-09-12T06:43:16+00:00, 2026-09-26T06:43:16+00:00) and in [2026-08-13T06:43:16+00:00, 2026-09-12T06:43:16+00:00)

| arm | model | role | window | tier filter |
|---|---|---|---|---|
| claude-fable-5-1 steward, window, tier >= 1 | claude-fable-5-1 | steward | [2026-09-12T06:43:16+00:00, 2026-09-26T06:43:16+00:00) | tier >= 1 |
| opus steward, window, tier < 1 | opus | steward | [2026-09-12T06:43:16+00:00, 2026-09-26T06:43:16+00:00) | tier < 1 |
| opus steward, before apply, tier >= 1 | opus | steward | [2026-08-13T06:43:16+00:00, 2026-09-12T06:43:16+00:00) | tier >= 1 |
| opus steward, before apply, tier < 1 | opus | steward | [2026-08-13T06:43:16+00:00, 2026-09-12T06:43:16+00:00) | tier < 1 |

| arm | runs | succeeded | turn-cap kills | budget kills | timed out | no end event |
|---|---|---|---|---|---|---|
| claude-fable-5-1 steward, window, tier >= 1 | 24 | 24/24 (100.0%) | 0/24 (0.0%) | 0/24 (0.0%) | 0/24 (0.0%) | 0/24 (0.0%) |
| opus steward, window, tier < 1 | 192 | 187/192 (97.4%) | 0/192 (0.0%) | 0/192 (0.0%) | 3/192 (1.6%) | 0/192 (0.0%) |
| opus steward, before apply, tier >= 1 | 26 | 23/26 (88.5%) | 0/26 (0.0%) | 0/26 (0.0%) | 1/26 (3.8%) | 0/26 (0.0%) |
| opus steward, before apply, tier < 1 | 500 | 467/500 (93.4%) | 0/500 (0.0%) | 0/500 (0.0%) | 9/500 (1.8%) | 0/500 (0.0%) |

| arm | cost total $ | cost p50 $ | cost p95 $ | turns p50 | turns p95 | duration p50 s | duration p95 s | duration max s |
|---|---|---|---|---|---|---|---|---|
| claude-fable-5-1 steward, window, tier >= 1 | 60.59 | 2.26 | 4.49 | 11 | 32 | 445 | 1115 | 1258 |
| opus steward, window, tier < 1 | 401.28 | 1.92 | 4.71 | 14 | 40 | 435 | 1306 | 1805 |
| opus steward, before apply, tier >= 1 | 51.34 | 1.74 | 3.62 | 19 | 41 | 650 | 1589 | 1802 |
| opus steward, before apply, tier < 1 | 833.68 | 1.52 | 3.36 | 17 | 40 | 512 | 1479 | 1803 |

| arm | dispatch max_turns |
|---|---|
| claude-fable-5-1 steward, window, tier >= 1 | 100: 24 |
| opus steward, window, tier < 1 | 100: 192 |
| opus steward, before apply, tier >= 1 | 100: 26 |
| opus steward, before apply, tier < 1 | 100: 500 |

How the escalations each arm's runs worked on left the steward, judged as of the arm's window end. Shares are over the decided ones (neither missing nor pending).

| arm | runs naming no escalation | record missing | pending | promoted to L1 | resolved in place | auto-dismissed | closed by other | decided | in-place share | promoted share |
|---|---|---|---|---|---|---|---|---|---|---|
| claude-fable-5-1 steward, window, tier >= 1 | 0 | 0 | 0 | 0 | 21 | 3 | 0 | 24 | 21/24 (87.5%) | 0/24 (0.0%) |
| opus steward, window, tier < 1 | 0 | 0 | 0 | 1 | 178 | 12 | 0 | 191 | 178/191 (93.2%) | 1/191 (0.5%) |
| opus steward, before apply, tier >= 1 | 0 | 14 | 0 | 1 | 10 | 0 | 0 | 11 | 10/11 (90.9%) | 1/11 (9.1%) |
| opus steward, before apply, tier < 1 | 0 | 229 | 0 | 8 | 247 | 14 | 0 | 269 | 247/269 (91.8%) | 8/269 (3.0%) |

Steward runs no arm admits (tier outside every band on their window, or unknown):

_none_

| L2 window | days | filed | filed/day | watcher filed | watcher filed/day | resolved | close_only share |
|---|---|---|---|---|---|---|---|
| [2026-08-13T06:43:16+00:00, 2026-09-12T06:43:16+00:00) | 30 | 298 | 9.93 | 225 | 7.50 | 288 | 195/288 (67.7%) |
| [2026-09-12T06:43:16+00:00, 2026-09-26T06:43:16+00:00) | 14 | 178 | 12.71 | 166 | 11.86 | 172 | 149/172 (86.6%) |

Escalation corpus: 4029 records loaded, 0 skipped; oldest archive date 2026-08-27 (records resolved before it have been pruned).

### 3. Spend and caps on `claude-fable-5-1` over [2026-09-12T06:43:16+00:00, 2026-09-26T06:43:16+00:00)

Daily spend, half-open 24 h slices:

| slice start | slice end | runs | total $ | ceiling $ | headroom $ | at/over ceiling |
|---|---|---|---|---|---|---|
| 2026-09-12T06:43:16+00:00 | 2026-09-13T06:43:16+00:00 | 2 | 8.21 | 150.00 | 141.79 | False |
| 2026-09-13T06:43:16+00:00 | 2026-09-14T06:43:16+00:00 | 5 | 14.68 | 150.00 | 135.32 | False |
| 2026-09-14T06:43:16+00:00 | 2026-09-15T06:43:16+00:00 | 3 | 9.81 | 150.00 | 140.19 | False |
| 2026-09-15T06:43:16+00:00 | 2026-09-16T06:43:16+00:00 | 2 | 4.88 | 150.00 | 145.12 | False |
| 2026-09-16T06:43:16+00:00 | 2026-09-17T06:43:16+00:00 | 1 | 5.17 | 150.00 | 144.83 | False |
| 2026-09-17T06:43:16+00:00 | 2026-09-18T06:43:16+00:00 | 6 | 14.80 | 150.00 | 135.20 | False |
| 2026-09-18T06:43:16+00:00 | 2026-09-19T06:43:16+00:00 | 6 | 17.88 | 150.00 | 132.12 | False |
| 2026-09-19T06:43:16+00:00 | 2026-09-20T06:43:16+00:00 | 3 | 12.84 | 150.00 | 137.16 | False |
| 2026-09-20T06:43:16+00:00 | 2026-09-21T06:43:16+00:00 | 1 | 3.19 | 150.00 | 146.81 | False |
| 2026-09-21T06:43:16+00:00 | 2026-09-22T06:43:16+00:00 | 3 | 14.25 | 150.00 | 135.75 | False |
| 2026-09-22T06:43:16+00:00 | 2026-09-23T06:43:16+00:00 | 2 | 10.23 | 150.00 | 139.77 | False |
| 2026-09-23T06:43:16+00:00 | 2026-09-24T06:43:16+00:00 | 3 | 14.92 | 150.00 | 135.08 | False |
| 2026-09-24T06:43:16+00:00 | 2026-09-25T06:43:16+00:00 | 2 | 3.55 | 150.00 | 146.45 | False |
| 2026-09-25T06:43:16+00:00 | 2026-09-26T06:43:16+00:00 | 3 | 10.43 | 150.00 | 139.57 | False |

Peak trailing-24 h spend (closed window, the resolver's rule):

| peak $ | reached at | ceiling $ | at/over ceiling |
|---|---|---|---|
| 27.68 | 2026-09-19T12:29:41.289970+00:00 | 150.00 | False |

Model rejections, any role:

_none_

Per-model ceiling trips, and the model each fell through to:

_none_

Cap hits scoped to `claude-fable-5-1`, by account:

| account | scoped hits | first hit |
|---|---|---|
| max-e | 1 | 2026-09-14T14:05:01.529574+00:00 |

Account-level (unscoped) cap hits: 64

Service restarts:

| timestamp | service | reason |
|---|---|---|
| 2026-09-12T07:32:20.828171+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-12T11:46:28.878952+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-12T23:45:05.875131+00:00 | dashboard | post_merge_dashboard_code_change |
| 2026-09-13T12:40:11.505210+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-14T06:00:49.534428+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-14T06:00:49.545675+00:00 | dashboard | post_merge_dashboard_code_change |
| 2026-09-15T02:35:57.665188+00:00 | dashboard | post_merge_dashboard_code_change |
| 2026-09-17T21:00:05.533452+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-17T21:00:05.616961+00:00 | dashboard | post_merge_dashboard_code_change |
| 2026-09-18T10:16:38.504092+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-19T00:52:21.299454+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-19T05:23:19.853390+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-20T01:43:39.335179+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-20T01:43:39.356335+00:00 | dashboard | post_merge_dashboard_code_change |
| 2026-09-20T14:22:13.746764+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-20T14:22:13.766621+00:00 | dashboard | post_merge_dashboard_code_change |
| 2026-09-21T03:37:40.867845+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-21T03:37:40.960938+00:00 | dashboard | post_merge_dashboard_code_change |
| 2026-09-21T04:54:12.641501+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-22T15:40:55.704998+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-22T15:40:55.719373+00:00 | dashboard | post_merge_dashboard_code_change |
| 2026-09-23T05:00:33.998566+00:00 | dashboard | post_merge_dashboard_code_change |
| 2026-09-23T10:50:53.505727+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-24T06:19:46.216892+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-24T06:19:46.229160+00:00 | dashboard | post_merge_dashboard_code_change |

### 4. Roles observed on `claude-fable-5-1` in [2026-09-12T06:43:16+00:00, 2026-09-26T06:43:16+00:00)

Admitted roles: merger, steward

| role | runs | total $ | admitted |
|---|---|---|---|
| merger | 18 | 84.26 | True |
| steward | 24 | 60.59 | True |

Roles outside the admitted set: none

**Command (b): the day-1 audit re-run over the same window (the appendix).**
It is pasted inside a fence, byte for byte. One merge-reason cell in its § 2
(task 4782) is a multi-line verify report carrying its own code fence, which
would break a rendered table. Filed as a follow-up to the audit renderer (§
Follow-ups).

````text
### 1. Routing decisions for `claude-fable-5-1` since 2026-09-12T06:43:16+00:00 until 2026-09-26T06:43:16+00:00

| timestamp | task | role | source_layer | rule_id | tier |
|---|---|---|---|---|---|
| 2026-09-12T08:52:56.079465+00:00 | 4377 | merger | config | - | 1 |
| 2026-09-13T02:15:16.452029+00:00 | 4377 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-13T07:53:52.827788+00:00 | 4211 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-14T03:36:45.381322+00:00 | 5031 | merger | config | - | 0 |
| 2026-09-14T03:58:39.891592+00:00 | 5066 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-14T05:16:57.512568+00:00 | 5066 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-14T05:49:42.026329+00:00 | 5066 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-14T13:57:36.713827+00:00 | 5029 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-14T14:43:51.147753+00:00 | 5029 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-14T16:00:54.697379+00:00 | 3730 | merger | config | - | 0 |
| 2026-09-15T18:04:17.471357+00:00 | 5029 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-15T18:40:50.382487+00:00 | 5029 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-15T22:32:16.881320+00:00 | 5029 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-17T04:16:16.451659+00:00 | 4495 | merger | config | - | 0 |
| 2026-09-17T08:44:11.264232+00:00 | 4194 | steward | policy_rule | steward-retry-fable | 2 |
| 2026-09-17T19:09:21.849580+00:00 | 5320 | merger | config | - | 0 |
| 2026-09-17T23:08:01.952193+00:00 | 5299 | merger | config | - | 0 |
| 2026-09-18T02:31:53.478545+00:00 | 4384 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-18T02:34:19.796094+00:00 | 4384 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-18T02:37:43.866996+00:00 | 4384 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-18T04:39:32.255868+00:00 | 4485 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-18T15:21:00.804432+00:00 | 4930 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-18T15:39:23.193181+00:00 | 4137 | merger | config | - | 0 |
| 2026-09-18T20:45:08.243678+00:00 | 3541 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-19T03:30:24.013403+00:00 | 3541 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-19T04:36:34.791435+00:00 | 4259 | merger | config | - | 0 |
| 2026-09-19T04:45:35.426693+00:00 | 4485 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-19T10:31:48.410150+00:00 | 3541 | merger | config | - | 1 |
| 2026-09-19T12:13:22.304194+00:00 | 3541 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-19T17:49:07.496224+00:00 | 5237 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-20T01:56:13.656459+00:00 | 5553 | merger | config | - | 0 |
| 2026-09-21T03:32:53.407727+00:00 | 4814 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-21T06:11:08.673006+00:00 | 4410 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-21T07:19:59.322583+00:00 | 4410 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-21T08:34:00.102701+00:00 | 5687 | merger | config | - | 0 |
| 2026-09-21T23:48:48.555220+00:00 | 4597 | merger | config | - | 0 |
| 2026-09-22T13:49:21.517190+00:00 | 5588 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-23T06:20:27.020866+00:00 | 3620 | merger | config | - | 0 |
| 2026-09-23T13:38:44.617675+00:00 | 4965 | merger | config | - | 0 |
| 2026-09-23T16:08:31.967929+00:00 | 4792 | merger | config | - | 1 |
| 2026-09-23T16:08:48.292085+00:00 | 4792 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-23T18:10:58.459519+00:00 | 4354 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-24T00:26:55.892867+00:00 | 5588 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-25T02:47:40.975568+00:00 | 4782 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-25T06:13:42.277225+00:00 | 4782 | merger | config | - | 1 |
| 2026-09-25T07:48:07.837063+00:00 | 4782 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-25T10:26:07.060676+00:00 | 4792 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-25T16:45:05.202905+00:00 | 5587 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-26T00:55:18.452937+00:00 | 4386 | merger | config | - | 0 |
| 2026-09-26T05:57:54.133370+00:00 | 4807 | merger | config | - | 0 |

Rejections naming a model, any role, since 2026-09-12T06:43:16+00:00 until 2026-09-26T06:43:16+00:00:

_none_

Unparseable payloads skipped: 0

### 2. Invocations on `claude-fable-5-1` and how they ended, since 2026-09-12T06:43:16+00:00 until 2026-09-26T06:43:16+00:00

| task | project | role | tier | subtype | account | cost $ | turns | ok | timed out | model @end | duration ms | over flat ceiling | merge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4377 | dark_factory | merger | 1 | success | max-b | 6.08 | 45 | True | False | claude-fable-5-1 | 1149252 | True | done (d411f107c3676e4ce8d322634bf1cce8bdf1110e) |
| 4377 | dark_factory | steward | 1 | - | max-b | 2.14 | 13 | True | False | - | 527497 | None | - |
| 4211 | dark_factory | steward | 1 | - | max-b | 1.72 | 9 | True | False | - | 207867 | None | - |
| 5031 | dark_factory | merger | 0 | success | max-e | 5.99 | 40 | True | False | claude-fable-5-1 | 1640020 | True | done (5af0b02bc367cf10cc21c915c4c523cd06803727) |
| 5066 | dark_factory | steward | 1 | - | max-e | 3.15 | 19 | True | False | - | 570648 | None | - |
| 5066 | dark_factory | steward | 1 | - | max-e | 3.30 | 12 | True | False | - | 385224 | None | - |
| 5066 | dark_factory | steward | 1 | - | max-e | 0.51 | 4 | True | False | - | 123290 | None | - |
| 5029 | dark_factory | steward | 1 | - | max-f | 3.99 | 15 | True | False | - | 740237 | None | - |
| 5029 | dark_factory | steward | 1 | - | max-f | 1.48 | 4 | True | False | - | 211413 | None | - |
| 3730 | dark_factory | merger | 0 | success | max-g | 4.34 | 24 | True | False | claude-fable-5-1 | 1473815 | True | done (df73e11a3c4be1a2a217901249c8ff43fbddb63c) |
| 5029 | dark_factory | steward | 1 | - | max-f | 4.49 | 8 | True | False | - | 468072 | None | - |
| 5029 | dark_factory | steward | 1 | - | max-f | 0.40 | 3 | True | False | - | 93349 | None | - |
| 4495 | dark_factory | merger | 0 | success | max-c | 5.17 | 50 | True | False | claude-fable-5-1 | 5275338 | True | done (c954faa62bade0892601705e4a8d172d9b7790d1) |
| 4194 | dark_factory | steward | 2 | - | max-c | 2.54 | 21 | True | False | - | 637234 | None | - |
| 5320 | dark_factory | merger | 0 | success | max-c | 3.48 | 24 | True | False | claude-fable-5-1 | 1122082 | True | done (a50219de1bd4f5f628aade9e7555cb8c6034f257) |
| 5299 | dark_factory | merger | 0 | success | max-c | 4.58 | 32 | True | False | claude-fable-5-1 | 1381790 | True | done (63a2984c654c1e2fc58980e14c9d3648bbde97a6) |
| 4384 | dark_factory | steward | 1 | - | max-c | 1.95 | 11 | True | False | - | 113880 | None | - |
| 4384 | dark_factory | steward | 1 | - | max-c | 0.59 | 9 | True | False | - | 188551 | None | - |
| 4485 | dark_factory | steward | 1 | - | max-c | 1.67 | 10 | True | False | - | 304377 | None | - |
| 4137 | dark_factory | merger | 0 | success | max-b | 5.74 | 33 | True | False | claude-fable-5-1 | 1071803 | True | done (03b8b18d9fee5c061b9732d12742686f184f3d03) |
| 4930 | dark_factory | steward | 1 | - | max-b | 3.96 | 32 | True | False | - | 1115494 | None | - |
| 3541 | dark_factory | steward | 1 | - | max-b | 2.26 | 16 | True | False | - | 445297 | None | - |
| 3541 | dark_factory | steward | 1 | - | max-b | 1.13 | 9 | True | False | - | 484074 | None | - |
| 4259 | dark_factory | merger | 0 | success | max-b | 2.97 | 18 | True | False | claude-fable-5-1 | 774156 | True | done (6cb5f4a2b6cb7094d1bb7bb33bf2db80f27fc226) |
| 4485 | dark_factory | steward | 1 | - | max-b | 1.83 | 7 | True | False | - | 173638 | None | - |
| 3541 | dark_factory | merger | 1 | success | max-b | 4.90 | 40 | True | False | claude-fable-5-1 | 5158815 | True | done (b79315a31c437c448275c8b3fc6874daa889da78) |
| 3541 | dark_factory | steward | 1 | - | max-b | 4.90 | 32 | True | False | - | 924736 | None | - |
| 5553 | dark_factory | merger | 0 | success | max-d | 3.04 | 18 | True | False | claude-fable-5-1 | 963750 | True | superseded |
| 4410 | dark_factory | steward | 1 | - | max-d | 3.19 | 20 | True | False | - | 867897 | None | - |
| 4410 | dark_factory | steward | 1 | - | max-d | 3.87 | 4 | True | False | - | 70417 | None | - |
| 5687 | dark_factory | merger | 0 | success | max-d | 4.04 | 30 | True | False | claude-fable-5-1 | 850822 | True | done (53d6833dcef195e8adb75ce5e3ed95d773f3f861) |
| 4597 | dark_factory | merger | 0 | success | max-d | 6.34 | 43 | True | False | claude-fable-5-1 | 5465034 | True | done (ecd4184f66a82c21707795038fc014cd3de92cb3) |
| 5588 | dark_factory | steward | 1 | - | max-d | 3.37 | 32 | True | False | - | 1092784 | None | - |
| 3620 | dark_factory | merger | 0 | success | max-e | 6.87 | 36 | True | False | claude-fable-5-1 | 1795596 | True | done (4f91cc280f16461f7176701e58d95cd53e104711) |
| 4965 | dark_factory | merger | 0 | success | max-c | 6.08 | 31 | True | False | claude-fable-5-1 | 1281269 | True | done (67df8c2c8bc3851278c447e1f1e7a92c31af0777) |
| 4792 | dark_factory | merger | 1 | success | max-c | 4.98 | 21 | True | False | claude-fable-5-1 | 598390 | False | blocked (Suffix item needs rebase onto frozen-prefix tip: branch '4792' has a real rebase conflict onto frozen tip 'd7800c54c638858b632106b3c0e6e646c292e07b') |
| 4354 | dark_factory | steward | 1 | - | max-c | 3.86 | 26 | True | False | - | 1257896 | None | - |
| 4782 | dark_factory | steward | 1 | - | max-c | 1.84 | 10 | True | False | - | 378402 | None | - |
| 4782 | dark_factory | merger | 1 | success | max-d | 1.71 | 14 | True | False | claude-fable-5-1 | 1288743 | True | blocked (Post-merge verification failed: Failures: tests failed [category: test_failure]

## Failure Cause

FAILED tests/test_memory_eval_retrieval_probe.py::TestDeriveFromGuardClusters::test_emits_one_candidate_per_guard_slug

## Verify Logs

Category: test_failure

Worktree:

Archive (durable, survives worktree cleanup):
- /home/leo/src/dark-factory/data/verify-logs/4782/attempt-1.fused-memory.test-20260925T064946_360836Z.log
- /home/leo/src/dark-factory/data/verify-logs/4782/attempt-1.fused-memory.lint-20260925T064946_360836Z.log
- /home/leo/src/dark-factory/data/verify-logs/4782/attempt-1.fused-memory.type-20260925T064946_360836Z.log
- /home/leo/src/dark-factory/data/verify-logs/4782/attempt-1.fused-memory.summary-20260925T064946_360836Z.json

## Test Failures

```
s/aiosqlite/core.py", line 66, in _connection_worker_thread
      future.get_loop().call_soon_threadsafe(set_result, future, result)
      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/leo/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/asyncio/base_events.py", line 878, in call_soon_threadsafe
      self._check_closed()
      ~~~~~~~~~~~~~~~~~~^^
    File "/home/leo/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/asyncio/base_events.py", line 556, in _check_closed
      raise RuntimeError('Event loop is closed')
  RuntimeError: Event loop is closed
  
  During handling of the above exception, another exception occurred:
  
  Traceback (most recent call last):
    File "/home/leo/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/threading.py", line 1043, in _bootstrap_inner
      self.run()
      ~~~~~~~~^^
    File "/home/leo/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/threading.py", line 994, in run
      self._target(*self._args, **self._kwargs)
      ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/leo/src/dark-factory/.worktrees/_merge-95280e7b/.venv/lib/python3.13/site-packages/aiosqlite/core.py", line 75, in _connection_worker_thread
      future.get_loop().call_soon_threadsafe(set_exception, future, e)
      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/leo/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/asyncio/base_events.py", line 878, in call_soon_threadsafe
...
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

tests/test_tools_validation.py::TestKnownProjectRegistryGate::test_add_memory_permissive_without_known_projects
  /home/leo/src/dark-factory/.worktrees/_merge-95280e7b/fused-memory/tests/test_tools_validation.py:247: RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited
    await server._tool_manager.call_tool(
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_memory_eval_retrieval_probe.py::TestDeriveFromGuardClusters::test_emits_one_candidate_per_guard_slug
1 failed, 22802 passed, 7 skipped, 4 xfailed, 302 warnings in 379.00s (0:06:18)

bringing up nodes...
bringing up nodes...

........................................................................ [  0%]
........................................................................ [  0%]
........................................................................ [  0%]
........................................................................ [  1%]
........................................................................ [  1%]
```

[provisional: an off-critical-path main-health probe is still checking whether this failure pre-exists on bare main (task 2564); if confirmed, this task is not at fault and main is being healed separately]) |
| 5587 | dark_factory | steward | 1 | - | max-b | 2.46 | 15 | True | False | - | 503988 | None | - |
| 4386 | dark_factory | merger | 0 | success | max-d | 3.44 | 28 | True | False | claude-fable-5-1 | 939389 | True | done (19f83df8dbe67bc7d582472cacce869edaaf677f) |
| 4807 | dark_factory | merger | 0 | success | max-d | 4.53 | 39 | True | False | claude-fable-5-1 | 7047729 | True | - |

### 3. Dispatches at retry tier >= 1 since 2026-09-12T06:43:16+00:00 until 2026-09-26T06:43:16+00:00

| timestamp | task | role | tier | rule_id |
|---|---|---|---|---|
| 2026-09-12T08:52:56.079465+00:00 | 4377 | merger | 1 | - |
| 2026-09-13T02:15:16.452029+00:00 | 4377 | steward | 1 | steward-retry-fable |
| 2026-09-13T07:53:52.827788+00:00 | 4211 | steward | 1 | steward-retry-fable |
| 2026-09-14T03:58:39.891592+00:00 | 5066 | steward | 1 | steward-retry-fable |
| 2026-09-14T05:16:57.512568+00:00 | 5066 | steward | 1 | steward-retry-fable |
| 2026-09-14T05:49:42.026329+00:00 | 5066 | steward | 1 | steward-retry-fable |
| 2026-09-14T13:57:36.713827+00:00 | 5029 | steward | 1 | steward-retry-fable |
| 2026-09-14T14:43:51.147753+00:00 | 5029 | steward | 1 | steward-retry-fable |
| 2026-09-15T18:04:17.471357+00:00 | 5029 | steward | 1 | steward-retry-fable |
| 2026-09-15T18:40:50.382487+00:00 | 5029 | steward | 1 | steward-retry-fable |
| 2026-09-15T22:32:16.881320+00:00 | 5029 | steward | 1 | steward-retry-fable |
| 2026-09-17T08:44:11.264232+00:00 | 4194 | steward | 2 | steward-retry-fable |
| 2026-09-18T02:31:53.478545+00:00 | 4384 | steward | 1 | steward-retry-fable |
| 2026-09-18T02:34:19.796094+00:00 | 4384 | steward | 1 | steward-retry-fable |
| 2026-09-18T02:37:43.866996+00:00 | 4384 | steward | 1 | steward-retry-fable |
| 2026-09-18T04:39:32.255868+00:00 | 4485 | steward | 1 | steward-retry-fable |
| 2026-09-18T15:21:00.804432+00:00 | 4930 | steward | 1 | steward-retry-fable |
| 2026-09-18T20:45:08.243678+00:00 | 3541 | steward | 1 | steward-retry-fable |
| 2026-09-19T03:30:24.013403+00:00 | 3541 | steward | 1 | steward-retry-fable |
| 2026-09-19T04:45:35.426693+00:00 | 4485 | steward | 1 | steward-retry-fable |
| 2026-09-19T10:31:48.410150+00:00 | 3541 | merger | 1 | - |
| 2026-09-19T12:13:22.304194+00:00 | 3541 | steward | 1 | steward-retry-fable |
| 2026-09-19T17:49:07.496224+00:00 | 5237 | steward | 1 | steward-retry-fable |
| 2026-09-21T03:32:53.407727+00:00 | 4814 | steward | 1 | steward-retry-fable |
| 2026-09-21T06:11:08.673006+00:00 | 4410 | steward | 1 | steward-retry-fable |
| 2026-09-21T07:19:59.322583+00:00 | 4410 | steward | 1 | steward-retry-fable |
| 2026-09-22T13:49:21.517190+00:00 | 5588 | steward | 1 | steward-retry-fable |
| 2026-09-23T16:08:31.967929+00:00 | 4792 | merger | 1 | - |
| 2026-09-23T16:08:48.292085+00:00 | 4792 | steward | 1 | steward-retry-fable |
| 2026-09-23T18:10:58.459519+00:00 | 4354 | steward | 1 | steward-retry-fable |
| 2026-09-24T00:26:55.892867+00:00 | 5588 | steward | 1 | steward-retry-fable |
| 2026-09-25T02:47:40.975568+00:00 | 4782 | steward | 1 | steward-retry-fable |
| 2026-09-25T06:13:42.277225+00:00 | 4782 | merger | 1 | - |
| 2026-09-25T07:48:07.837063+00:00 | 4782 | steward | 1 | steward-retry-fable |
| 2026-09-25T10:26:07.060676+00:00 | 4792 | steward | 1 | steward-retry-fable |
| 2026-09-25T16:45:05.202905+00:00 | 5587 | steward | 1 | steward-retry-fable |

### 4. Scoped cap posture for `claude-fable-5-1` since 2026-09-12T06:43:16+00:00 until 2026-09-26T06:43:16+00:00

| created_at | account | reason |
|---|---|---|
| 2026-09-14T14:05:01.529574+00:00 | max-e | You've hit your session limit · resets 6:20pm (Europe/London) |

Account-level (unscoped) cap hits in the same period: 64

Service restarts since then (only an orchestrator restart reloads a restart-tier leaf):

| timestamp | service | reason |
|---|---|---|
| 2026-09-12T07:32:20.828171+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-12T11:46:28.878952+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-12T23:45:05.875131+00:00 | dashboard | post_merge_dashboard_code_change |
| 2026-09-13T12:40:11.505210+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-14T06:00:49.534428+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-14T06:00:49.545675+00:00 | dashboard | post_merge_dashboard_code_change |
| 2026-09-15T02:35:57.665188+00:00 | dashboard | post_merge_dashboard_code_change |
| 2026-09-17T21:00:05.533452+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-17T21:00:05.616961+00:00 | dashboard | post_merge_dashboard_code_change |
| 2026-09-18T10:16:38.504092+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-19T00:52:21.299454+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-19T05:23:19.853390+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-20T01:43:39.335179+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-20T01:43:39.356335+00:00 | dashboard | post_merge_dashboard_code_change |
| 2026-09-20T14:22:13.746764+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-20T14:22:13.766621+00:00 | dashboard | post_merge_dashboard_code_change |
| 2026-09-21T03:37:40.867845+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-21T03:37:40.960938+00:00 | dashboard | post_merge_dashboard_code_change |
| 2026-09-21T04:54:12.641501+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-22T15:40:55.704998+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-22T15:40:55.719373+00:00 | dashboard | post_merge_dashboard_code_change |
| 2026-09-23T05:00:33.998566+00:00 | dashboard | post_merge_dashboard_code_change |
| 2026-09-23T10:50:53.505727+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-24T06:19:46.216892+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-24T06:19:46.229160+00:00 | dashboard | post_merge_dashboard_code_change |

### 5. Spend on `claude-fable-5-1` over [2026-09-12T06:43:16+00:00, 2026-09-26T06:43:16+00:00)

| invocations | total $ | ceiling $ | headroom $ | at/over ceiling |
|---|---|---|---|---|
| 42 | 144.86 | 150.00 | 5.14 | False |

### 6. Roles observed on `claude-fable-5-1` since 2026-09-12T06:43:16+00:00 until 2026-09-26T06:43:16+00:00

Admitted roles: merger, steward

| role | invocations | total $ | admitted |
|---|---|---|---|
| merger | 18 | 84.26 | True |
| steward | 24 | 60.59 | True |

Roles outside the admitted set: none
````

**Store facts neither script renders.** These are two read-only queries,
each shown with its output pasted verbatim. Both were run from the main
checkout as `sqlite3 -readonly -markdown data/orchestrator/runs.db
"<query>"`.

Configuration reloads inside the review window:

```sql
SELECT timestamp, json_extract(data, '$.applied'), json_extract(data, '$.restart_required') FROM events WHERE event_type = 'config_reload' AND timestamp >= '2026-09-12T06:43:16+00:00' AND timestamp < '2026-09-26T06:43:16+00:00' ORDER BY timestamp;
```

|            timestamp             |                                                                                  json_extract(data, '$.applied')                                                                                   |                                json_extract(data, '$.restart_required')                                |
|----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| 2026-09-12T07:00:51.156939+00:00 | {"recovery_emission.streak_escalation_enabled":{"old":true,"new":false}}                                                                                                                           | {"usage_cap.scoped_cap_models":{"old":["claude-fable-5"],"new":["claude-fable-5","claude-fable-5-1"]}} |
| 2026-09-17T11:39:26.361661+00:00 | {"verify_env":{"old":{"PYTEST_XDIST_AUTO_NUM_WORKERS":"8","DF_REQUIRE_SANDBOX_TESTS":"1"},"new":{"PYTEST_XDIST_AUTO_NUM_WORKERS":"16","DF_REQUIRE_SANDBOX_TESTS":"1"}}}                            | {}                                                                                                     |
| 2026-09-21T07:55:27.740526+00:00 | {"steward_lifetime_budget":{"old":12.0,"new":20.0}}                                                                                                                                                | {}                                                                                                     |
| 2026-09-23T11:53:28.101022+00:00 | {"verify_admission_task_slots":{"old":1,"new":2}}                                                                                                                                                  | {}                                                                                                     |
| 2026-09-24T06:47:39.891417+00:00 | {"effort.architect":{"old":"max","new":"xhigh"},"effort.implementer":{"old":"max","new":"xhigh"},"effort.debugger":{"old":"max","new":"xhigh"},"effort.deep_reviewer":{"old":"max","new":"xhigh"}} | {}                                                                                                     |
| 2026-09-24T09:28:02.257484+00:00 | {"effort.merger":{"old":"max","new":"high"}}                                                                                                                                                       | {}                                                                                                     |

Orchestrator runs that made a routing decision inside the review window.
`orchestrator/src/orchestrator/harness.py::Harness.run` mints a fresh
`run_id` at every orchestrator startup, so each row is one orchestrator
process lifetime:

```sql
SELECT run_id, MIN(timestamp), MAX(timestamp) FROM events WHERE event_type = 'routing_decision' AND timestamp >= '2026-09-12T06:43:16+00:00' AND timestamp < '2026-09-26T06:43:16+00:00' GROUP BY run_id ORDER BY 2;
```

|      run_id      |          MIN(timestamp)          |          MAX(timestamp)          |
|------------------|----------------------------------|----------------------------------|
| run-b756e05f8ff0 | 2026-09-12T08:21:19.814709+00:00 | 2026-09-12T09:02:05.318410+00:00 |
| run-063e9795fcf6 | 2026-09-12T10:11:50.908285+00:00 | 2026-09-14T12:29:20.072332+00:00 |
| run-5352d42ce5fa | 2026-09-14T12:51:38.480785+00:00 | 2026-09-15T22:34:24.274260+00:00 |
| run-9b3c5d8df8ae | 2026-09-15T23:18:33.756128+00:00 | 2026-09-21T03:42:11.322464+00:00 |
| run-ab649438402a | 2026-09-21T04:11:05.111134+00:00 | 2026-09-25T07:48:07.837063+00:00 |
| run-939fa92c07f8 | 2026-09-25T08:25:53.787511+00:00 | 2026-09-26T06:19:59.388799+00:00 |

## Verdict per item

### (1) Merger

**Volume and coverage.** 18 Fable merger runs against 19 opus merger runs in
the baseline window. The `opus merger, window` arm is empty (0 runs), so
every merger dispatch after the apply resolved to Fable.

**How the runs ended.** Every Fable run ended successfully (18/18) with no
turn-cap, budget or timeout kill. Opus succeeded in 14/19 runs. Its five
non-successes are its five turn-cap kills (5/19, i.e. 26.3%), and it had no
budget kill and no timeout. That matches S5 §1.1's baseline for the same
role: merger/opus 72% success and 5 turn-cap kills (28%) at a 50-turn cap.

**The D4 confound, stated with the dispatch caps.** The `dispatch max_turns`
column shows the confound directly: opus before the apply is `50: 18, 100: 1`,
and Fable is `100: 18`. 18 of the 19 opus baseline runs were dispatched at
the pre-D4 caps (50 turns / $5), and Fable ran at the D4 caps (100 / $8).
Opus's turns p50/p95 of 48/52 sit at its 50-turn cap. Fable's 31/50 sit
well under its 100-turn cap. So a bare "0/18 vs 5/19" overstates Fable's
advantage on cap kills. The fair comparator is the D5 package's merger
replay (§3), whose opus/100 arm recorded **0 turn-cap kills** on 16 fixtures.
Against either comparator, Fable's 0/18 is not worse, and that is what the
kill criterion asks.

**Resolution.** `resolved (merge done)` is 14/18 (77.8%) for Fable against
15/19 (78.9%) for opus. The production gap is nil, unlike the replay's 15/16
vs 13/16. The four Fable runs that did not end `done` are all outcomes of the
merge queue or of verification, not a merger giving up. The merge-state
breakdown is `blocked: 2, done: 14, superseded: 1, -: 1`, and the audit's § 2
names each:
- task 5553, `superseded`: a later merge request superseded this one, a
  queue outcome;
- task 4792, `blocked (Suffix item needs rebase onto frozen-prefix tip …)`: a
  coalesce-train outcome. The branch conflicted when it was rebased onto the
  train's frozen prefix tip, a queue event outside the merger's own run
  (which ended `ok: True`);
- task 4782, `blocked (Post-merge verification failed …)`: a post-merge
  verify failure in
  `tests/test_memory_eval_retrieval_probe.py::TestDeriveFromGuardClusters`.
  The record itself carries a provisional note that a main-health probe was
  checking whether that failure pre-existed on bare main. This review does
  not settle whether the merged tree or main caused it;
- task 4807, `-`: no merge was finalized inside the run's window before the
  review window closed (its routing decision is stamped
  2026-09-26T05:57:54Z).

The opus baseline carries the same kind of queue outcomes
(`already_merged: 1, blocked: 1, superseded: 2`).

**Merge integrity.** There were no drop-guard firings in the review window
or in the 30-day baseline window, from either witness (the `merge_attempt`
outcome `dropped_plan_targets` and the `merge_finalized` reason prefix
`orchestrator/src/orchestrator/merge_gates.py::DROPPED_PLAN_TARGETS_REASON_PREFIX`).
No conflict was finalized on any task after a successful Fable merger run on
it.

**Cost.** Fable's merger total was $84.26 (p50 $4.58, p95 $6.87), against
opus's $48.52 (p50 $2.51, p95 $3.86). The opus figures are truncated by the
same confound: 18 of its 19 runs carried a $5 budget. The replay's +23% at
the median (§3: $1.64 vs $1.33) was measured on conflict fixtures alone.
Production shows a wider gap, and this review did not measure why. In
absolute terms, the merger's share of Fable spend is the $84.26 in the audit's
§ 6.

**Duration.** Fable p50 was 1281 s and p95 = max was 7048 s, against opus
p50 1103 s and p95 = max 3122 s. The replay's finding that Fable needs about
half the wall clock is **not** reproduced in production at the median. Four
Fable merger runs ran past 5,000 s: in the audit's § 2 they are tasks 4495
(5275338 ms), 3541 (5158815 ms), 4597 (5465034 ms) and 4807 (7047729 ms).
All four ended `ok: True, timed out: False`. This review did not measure
where those runs spent their time.

**Mid-window change.** The configuration reloads show `effort.merger`
moving from `max` to `high` at 2026-09-24T09:28:02Z. The config comment
beside `models.merger` gives the reason: `high` is the effort the D5 replay
validated. The last three Fable merger rows in the audit's § 2 (tasks 4782,
4386, 4807) ran after the change, and the earlier ones ran at `max`.

### (2) Steward at tier >= 1

**The arms, and why there are four.** `routing_tier` is the task's
adaptive-retry counter, so a Fable steward only ever sees an escalation on a
task that has already been retried. Comparing it with all opus stewards
would credit or blame Fable for population difficulty. The like-for-like row
is **`opus steward, before apply, tier >= 1`**. The concurrent control is
`opus steward, window, tier < 1`: same weeks, same fleet, same config
changes. Every steward run fell into an arm; the "steward runs no arm
admits" table is `_none_`.

**Dispositions** (the review's § 2 disposition table):

| arm | resolved in place | promoted to L1 | auto-dismissed |
|---|---|---|---|
| Fable, window, tier >= 1 | 21/24 (87.5%) | 0/24 (0.0%) | 3 |
| opus, before apply, tier >= 1 (**like-for-like**) | 10/11 (90.9%) | 1/11 (9.1%) | 0 |
| opus, window, tier < 1 (concurrent control) | 178/191 (93.2%) | 1/191 (0.5%) | 12 |
| opus, before apply, tier < 1 | 247/269 (91.8%) | 8/269 (3.0%) | 14 |

Every cell above is copied from the generated table; the rows are regrouped,
not recomputed.

**Coverage caveat.** The escalation archive begins at `2026-08-27` (the
review's corpus line), and records resolved before it have been pruned. The
review window is fully covered: 0 records missing in either in-window arm.
The pre-apply arms are not. The like-for-like arm has 14 escalations
`record missing`, so its 10/11 rests on 11 decided escalations from 26 runs.
The tier-0 pre-apply arm has 229 missing. The like-for-like row is therefore
small, but it agrees with the two larger opus rows: every opus arm resolves
over 90% of the escalations it works in place (90.9%, 93.2%, 91.8%), at
every tier.

**Why S5 §4.7's "~130 of 3,865" is not the comparator.** That ratio's
denominator is every escalation record closed at every level in its window.
It includes the L1s swept by an L2 cascade, the orphan-reaper and
auto-dismissal sweeps, and the watcher's own L2 work, none of which a steward
ever touched. Per **steward-worked L0**, the share is the table above. So
P4-06's premise, that the steward "resolves a small share", mismeasured the
thing it was meant to improve.

**The L0 to L1 climb.** Promotions out of a steward-worked L0 are rare in
every arm: 0/24 on Fable, 1/11 and 1/191 on opus.

**Mid-window change.** `steward_lifetime_budget` moved from 12.0 to 20.0 at
2026-09-21T07:55:27Z. It applies to the Fable arm and to the concurrent opus
control alike.

**The L2 tier** (the review's L2 table; the two metrics P4-11 named as
sensitive):

- **Review window:** 178 filed (12.71/day), of which the watcher filed 166
  (11.86/day). The close_only share was 149/172 (86.6%).
- **Baseline window:** 298 filed (9.93/day), 7.50/day by the watcher. The
  close_only share was 195/288 (67.7%). This row is a lower bound on volume:
  its first two weeks predate the archive, so L2s resolved before
  2026-08-27 are gone.
- **P4-11's baseline** (from S5 §4.4, 2026-08-10 to 09-10) was 19.5 L2/day
  and a 69% close_only share.

In the review window, L2 volume is below P4-11's figure and the close_only
share is above it. **This ruling did not touch the L1 watcher** (P4-11 is a
separate proposal), and a steward that promotes almost nothing on any model
feeds the L1/L2 tiers only marginally. So neither movement is evidence about
D6, in either direction.

### (3) Cost against the $150/day ceiling

- **Daily spend.** No 24 h slice came near the ceiling: the largest is
  $17.88 (2026-09-18 slice), and every `at/over ceiling` cell is `False`.
  The audit's § 5 puts the whole 14-day spend at $144.86 over 42 runs,
  under one day's ceiling.
- **Peak trailing 24 h,** measured as a closed window, the resolver's own
  rule in `shared/src/shared/cost_store.py::CostStore.model_cost_in_window`:
  $27.68 at 2026-09-19T12:29:41Z, against $150.00, `False`.
- **Ceiling trips and fall-through.** There were no model rejections of any
  kind and no `model-ceiling-exhausted` rejection, so no day tripped the
  ceiling and nothing fell through to opus.
- **Fable-scoped cap hits.** One, on `max-e` at 2026-09-14T14:05:01Z ("You've
  hit your session limit · resets 6:20pm (Europe/London)"). There were 64
  account-level (unscoped) cap hits in the same period.

**OBSERVATION: `usage_cap.scoped_cap_models` went live, although no
orchestrator `service_restart` event is recorded.** The one scoped hit
carries a `scope` key, and only the scoped path in
`shared/src/shared/usage_gate.py::AccountPool` emits that key. So by
2026-09-14T14:05Z the orchestrator was running with `claude-fable-5-1` in
`scoped_cap_models`. Yet the review's restart table lists only
`fused-memory` and `dashboard` restarts, and the day-1 check found none for
the orchestrator after 2026-09-04. The two store facts above close the gap
as observations:
- the 2026-09-12T07:00:51Z reload listed `usage_cap.scoped_cap_models` under
  `restart_required`;
- the orchestrator `run_id` changed from `run-b756e05f8ff0` (last routing
  decision 2026-09-12T09:02:05Z) to `run-063e9795fcf6` (first
  2026-09-12T10:11:50Z), and changed four more times in the window.

Hypothesis: the orchestrator is restarted by a path that emits no
`service_restart` event (the fleet redeploy of CLAUDE.md § "Orchestrator
Fleet Redeploy"), and the startup after the 07:00Z reload loaded the leaf. If
so, the day-1 check's "scoped cap inert" residual risk, measured at
2026-09-13T13:59Z, was already closed when it was written. Its evidence,
`service_restart` events, does not record every orchestrator startup. Filed
as a follow-up to the audit's restart witness (§ Follow-ups).

### (4) Leakage

`Roles outside the admitted set: none`, in both the review's § 4 and the
audit's § 6. Fable ran as merger 18 times ($84.26) and as steward 24 times
($60.59), and nothing else.

The retry ladder was **left unchanged on purpose**: see the "DELIBERATE
DEVIATION from P4-06's 'ladder top = fable'" comment that guards
`routing.ladder` in `dark-factory-orchestrator.yaml`. A retry tier-up
therefore cannot route an implementer, debugger or architect to Fable. The
merger dispatches at tier 1 in the audit's § 1 (tasks 4377, 3541, 4792,
4782) came from the `config` layer (`models.merger`, an absolute model
string), not from the ladder.

### (5) Verdict against the D6 kill criteria (P4-06, as restated in task 5441)

**Merger: keep it on Fable.** The criterion was: kill if its cap-kill or
timeout rate is worse than opus's baseline, or if a drop-guard violation
appeared.
- **Cap kills:** 0/18 turn-cap and 0/18 budget, against 5/19 and 0/19. This
  is not worse, and it is not worse against the confound-free replay
  comparator either (opus/100: 0 turn-cap kills).
- **Timeouts:** 0/18 against 0/19.
- **Drop guard:** no firing in either window.
- **Conflict re-opens:** none.
- **Resolution** is level (14/18 vs 15/19). Every non-`done` Fable outcome is
  a queue or verification outcome, as itemised in (1).
- **Cost** is higher per run, but the merger's absolute Fable spend over 14
  days, $84.26, is small against the ceiling.

Small-n caveat: 18 vs 19 runs. A "not worse" verdict at this n cannot
exclude a modest regression, but nothing measured points to one.

**Steward rule `steward-retry-fable`: kill it, by the pre-registered
criterion.** The criterion was: keep the rule unless the L0-resolved share
did not move.
- It did not move. Fable resolved 21/24 (87.5%) in place, against 10/11
  (90.9%) like-for-like, 178/191 (93.2%) in the concurrent control and
  247/269 (91.8%) at tier 0 before the apply.
- The promoted share is 0/24 against 1/11.
- Nothing in the steward outcome columns distinguishes the arms beyond noise:
  success 24/24 against 23/26, and timeouts 0/24 against 1/26.

The honest statement has two halves, and Leo may weigh them differently:
1. **The criterion fired.** The rule costs Fable ceiling headroom (steward
   spend $60.59 of the window's $144.86) for no measured gain. Honoring the
   pre-registered rule means reverting the steward to opus at every tier.
2. **The criterion could not have passed at this n.** Its premise, a "small
   share" resolved at L0, comes from a ratio whose denominator is not
   steward-worked escalations. Per steward-worked L0, every opus arm already
   resolves over 90% in place, so the room for improvement is under ten
   points, and 24 runs cannot detect a fraction of that. "Did not move" is
   therefore weak evidence of no effect. It is not evidence of harm: nothing
   in the Fable steward arm is worse than in the opus arms.

The recommendation is **kill**, because the rule's premise is falsified and
the pre-registered test fired. Keeping it would need a new criterion that
this population can actually move, such as the promoted-to-L1 share over a
longer window, registered before looking. That is Leo's call.

**`timeouts.merger`: leave it at 600 s. Leave `invocation_timeout` at
7200 s for now.**
- The 600 s figure is a **flat pre-liveness ceiling**, and it killed nothing:
  `timed out` is 0/18, while 17/18 Fable runs ran past 600 s
  (`over flat ceiling`) and succeeded.
- The D5 package's advice to "raise `timeouts.merger` from 600 s toward the
  replay's 1800 s" (§6) was premised on the replay's flat wall clock. In
  production, a live merger is already bounded by
  `max(working_idle_secs, timeouts.merger) = 1800 s` of idle time. Raising
  `timeouts.merger` to 1800 would change nothing for a live run. Raising it
  above 1800 would lengthen the idle bound, which no measured run needed.
- The bound a live merger actually approached is the absolute cap: the Fable
  p95 and max are both 7048 s, against `invocation_timeout: 7200`. That run
  (task 4807) succeeded. Three more ran past 5,000 s.
- `invocation_timeout` is a single global value: `_invoke` passes it as
  every role's absolute cap. Raising it for the merger's sake would raise it
  for the implementer, architect and every other role. The merger has no
  absolute cap of its own in config.

So the recommendation is to change neither value now, and to re-check at the
next review for any merger run with `timed out: True` at the absolute cap. If
one appears, a per-role absolute cap for the merger is the targeted fix (a
code change, not a config flip).

## Levers, if Leo ratifies

**Nothing was changed by this task.** `dark-factory-orchestrator.yaml` was
not edited, and `reload_config` was not called. Ratification belongs to Leo.
The levers, all in `dark-factory-orchestrator.yaml` and applied with the
escalation MCP's `reload_config` (see OPERATIONS.md § "Config reload vs
restart" for which tier each leaf is in):

- **Keep the merger:** nothing to do (`models.merger: claude-fable-5-1`
  stays).
- **Kill the steward rule:** remove the `steward-retry-fable` entry from
  `routing.rules`. The steward then resolves to `models.steward` (opus,
  which dark-factory does not override) at every tier.
- **Timeouts:** `timeouts.merger` and `invocation_timeout` stay as they are.
- **Full rollback, if ever wanted:** remove `claude-fable-5-1` from
  `routing.allowed_models`. The resolver then falls back fail-safe, per
  P4-06's rollback lever.

## Follow-ups

Filed as low-priority follow-ups (outside this task's plan), both on `scripts/audit_model_admission.py`:

- ticket `tkt_0RV3EQPND1NJ8G5P8CRTQ80CNH`: a multi-line merge reason breaks the audit's § 2 markdown table (why command (b) is fenced above);
- ticket `tkt_0RV3ER6SYR76TM3XWEMB0RAZZ1`: the audit's restart witness (`service_restart`) misses orchestrator startups, so it cannot say whether a restart-tier leaf has been loaded (§ Verdict per item, (3)).
