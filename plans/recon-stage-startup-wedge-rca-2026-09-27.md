# RCA: recon stages killed by the 120s startup-wedge watchdog, recorded as "inert" (2026-09-27)

Task 5982. Read-only investigation against the live journal and the fused-memory journal logs,
measured 2026-09-27. Every figure below was measured, not estimated.

## 1. Symptom

Stage 1 of solar_challenge repeatedly flagged `stage2_relay_llm_analysis_gap`: Stage 2 had been
"dispatched but inert". It relayed flags, then never analysed them. The lineage is consolidated in
Mem0 canonical `5e93c78b` (topic `stage2-relay-llm-analysis-gap-flag-history`). It records 10
occurrences over 6 days. The runs are `814f9bdc`, `8606fe70`, `d1186e2d`, `9e02cc77`, `ab33414e`,
`b8211ec5`, `8061a8e3`, `405b5712`, `738b833b`, `18e6b506` and `b830eeb7`.

In every one, `task_knowledge_sync` recorded 0 LLM calls over a stage duration of 122–128s.

## 2. Evidence

### 2.1 One run, end to end (18e6b506)

From `journalctl --user -u fused-memory`, local time (BST, UTC+1), 00:31–00:34:

```
00:31:35 Reconciliation stage (sonnet): dispatching on account 'max-d'
00:33:35 shared.cli_invoke WARNING Startup wedge detected after 120.0s (grace=120.0s, turns=0): model=sonnet — cancelling comm_task and killing
00:33:37 stages.base ERROR Stage task_knowledge_sync failed: Agent produced no output
```

`TaskKnowledgeSync.run` then wrote the cycle-summary ledger row and its Mem0 mirror as usual. The
mirror read "0 item(s) flagged, 0 llm_call(s), 0 token(s) used".

The Stage 2 agent was launched. The shared watchdog killed it at exactly 120s because no assistant
turn had appeared yet.

### 2.2 Fleet census (journal `runs.stage_reports`, all projects, since 2026-09-01)

- 139 stage invocations recorded `llm_calls == 0` with a duration of 115–135s.
- Zero-call durations cluster at 120–130s: 139 of ~175 zero-call stages. The rest are spread
  widely.
- By stage: `task_knowledge_sync` (Stage 2) 114, `integrity_check` (Stage 3) 14,
  `memory_consolidator` (Stage 1) 12.
- Stage 2 only, per project (hits / Stage 2 invocations):

  | project | hits |
  |---|---|
  | solar_challenge | 27 / 166 |
  | know_live | 30 / 95 |
  | reify | 19 / 213 |
  | dark_factory | 14 / 136 |
  | autopilot_video | 12 / 162 |
  | solar_challenge_platform | 12 / 195 |

- Stage 3 of `738b833b`, `18e6b506` and `b830eeb7` shows the identical 0 calls / 122s.

The failure is fleet-wide and stage-agnostic. Stage 2 is hit about 10x more often than the others.

## 3. Why Stage 2 is hit most

Time to first assistant turn was measured from the transcripts that survive under
`data/reconciliation/recon-config`. It runs from the earliest transcript record to the first
assistant record, so it is a lower bound on spawn-to-first-turn.

| stage | median | p90 | p95 | max | n |
|---|---|---|---|---|---|
| Stage 2 | 68s | 96s | 100s | 111s | 62 |
| Stage 1 | 43s | | | | |
| Stage 3 | 42s | | | | |

Stage 2's tail sits against the 120s bound. The sample is censored at 120s, because every slower
start was killed before it produced a turn. The zero-turn Stage 2 transcripts end with `cost-state`
and contain no assistant record.

## 4. Root cause

`fused_memory/reconciliation/cli_stage_runner.py::run_stage_via_cli` never passed
`startup_grace_secs`. Every recon stage therefore inherited the generic 120s default of
`shared/src/shared/cli_invoke.py::invoke_claude_agent`.

The orchestrator made the identical diagnosis on this host on 2026-09-07. Its kills clustered within
~250ms of 120s, with CPU PSI around 52%. It raised its own bound to 450s:
`dark-factory-orchestrator.yaml` `timeouts.startup_grace_secs: 450` (esc-4444-3). Recon was never
given that knob.

## 5. Why it looked inert

`BaseStage.run` dropped `StageResult.success` and `StageResult.error` when it built the
`StageReport`. `write_cycle_summary` then produced a ledger row and a Mem0 mirror that could not be
told apart from a stage that ran and chose to do nothing. `get_cycle_summary_presence` reported
`present=true`.

It took 10 Stage-1 flags over 6 days, plus log-diving, to find a cause the process already knew.

## 6. Consequence

A Stage 1 flag relayed in a run whose Stage 2 agent died is never rendered to any Stage 2 LLM. The
next cycle classes it as `Stage2FlagPartition.stale_mismatched_run_id_ids`, and only GC reaps it.
Task 5982 did not fix that. Task 5984 closes it: such flags are carried forward until a Stage 2
agent completes (§9).

## 7. Fix (task 5982)

- **Root cause.** `ReconciliationConfig.stage_startup_grace_secs`, default 450.0, is the bound on
  how long a recon stage agent may go without its first assistant turn. It is validated to be
  `<= stage_timeout_seconds`. `run_stage_via_cli` forwards it as `startup_grace_secs` through
  `invoke_with_cap_retry` to `invoke_claude_agent`. It covers all three stages. `shared/` is
  untouched.
  - 450 follows the orchestrator's measured host ruling. It clears the measured Stage 2 tail
    (max 111s, censored at 120s) with wide margin.
  - The cost of a genuine wedge rises from 120s to 450s. That is small against the 3600s stage
    bound.
- **Observability.** Each stage agent's fate is recorded as a typed `StageAgentOutcome`:
  `completed`, `zero_output_timeout`, `timed_out`, `failed` or `not_launched`.
  - `zero_output_timeout` is decided by the canonical `shared.cli_invoke.is_zero_output_timeout`
    predicate.
  - It travels `StageResult.outcome` → `StageReport.agent_outcome`. From there it reaches the
    cycle-summary ledger payload, the Mem0 mirror text ("Agent outcome: …"),
    `get_cycle_summary_presence`, and the Stage 2/3 prompt rendering of prior stage reports.
  - `None`, rendered "unrecorded", means the outcome is unknown: a legacy row, or a
    harness-synthesized degraded summary. It is never defaulted to `completed`.

## 8. How to re-measure after deploy

Read-only. Run it from the main checkout, or pass the journal path. The second argument is the
`started_at` lower bound.

```python
import collections
import json
import sqlite3
import sys
from datetime import datetime

DB = sys.argv[1] if len(sys.argv) > 1 else 'data/reconciliation/reconciliation.db'
SINCE = sys.argv[2] if len(sys.argv) > 2 else '2026-09-01'

con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
rows = con.execute(
    'SELECT project_id, stage_reports FROM runs WHERE started_at >= ?', (SINCE,)
).fetchall()

wedge_shaped = collections.Counter()
stages_seen = collections.Counter()
by_outcome = collections.Counter()
for project_id, blob in rows:
    for stage, report in json.loads(blob or '{}').items():
        if not isinstance(report, dict) or 'started_at' not in report:
            continue
        stages_seen[(project_id, stage)] += 1
        by_outcome[(stage, report.get('agent_outcome') or 'unrecorded')] += 1
        started = datetime.fromisoformat(report['started_at'].replace('Z', '+00:00'))
        completed = datetime.fromisoformat(report['completed_at'].replace('Z', '+00:00'))
        secs = (completed - started).total_seconds()
        if report.get('llm_calls', 0) == 0 and 115 <= secs <= 135:
            wedge_shaped[(project_id, stage)] += 1

print('llm_calls==0 with a 115-135s duration, per (project, stage): hits/total')
for key in sorted(stages_seen):
    if wedge_shaped[key]:
        print(f'  {key[0]:<28} {key[1]:<22} {wedge_shaped[key]}/{stages_seen[key]}')
print('agent_outcome per stage (unrecorded = written before task 5982):')
for (stage, outcome), n in sorted(by_outcome.items()):
    print(f'  {stage:<22} {outcome:<20} {n}')
```

Expected after deploy:
- the 115–135s zero-call cluster stops growing for runs started after the restart;
- `agent_outcome` appears on new rows;
- any surviving `zero_output_timeout` is a kill at the new 450s bound, which is the data a
  retry or a conditional extension (§9) would be decided on.

Pre-deploy baseline, re-run 2026-10-12 with `SINCE=2026-09-01`: 168 wedge-shaped stages, 138 of
them Stage 2. Every row was still `unrecorded`.

Cross-check against the logs (journald times are local BST, UTC+1):

```
journalctl --user -u fused-memory --since 2026-09-01 | grep 'Startup wedge detected'
```

After deploy, the grace in those lines should read `grace=450.0s`.

## 9. Residuals

- **No retry-once on `zero_output_timeout`.** The measured failure is a censoring bound, not a
  transient wedge. A blind retry would double the cost of every genuine wedge, now 450s, and would
  hide whether the grace change worked. Decide it on the §8 data.
- **Flag carry-forward.** Closed by task 5984 (§6): Stage 2 renders markers from the
  trailing streak of runs whose Stage 2 agent did not complete
  (`summary_pool.py::trailing_incomplete_agent_run_ids`), counted in
  `stats.stage2_markers_carried_forward`. Two limits bound the carry. The marker GC
  (`task_knowledge_sync.py::_sweep_stale_mem0_flag_for_stage2_markers`) still reaps a carried
  marker once it is over 7 days old and its task is terminal, or over 30 days old if it cites no
  task. And a run leaves the streak when its 30-day `cycle_summary` row expires. Past either
  limit, the flag is lost as it was before.
- **The shared watchdog.** Task 3326 owns its semantics: a load-aware two-regime grace, and
  truthful kill attribution. `shared/src/shared/cli_invoke.py`'s 120s default is unchanged here.
