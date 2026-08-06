# Task 3640 — decision queue-stamp back-fill: audit report

Run 2026-08-06 against the live fleet root `~/.claude/fleet`, from branch
`task/3640`. This file is the durable evidence that the back-fill held; the
script itself is `scripts/backfill_decision_queue_stamp.py` and is re-runnable.

## 1. Authoritative recount (supersedes the filed "27")

The task was filed citing 27 unstamped open records. That figure was already
stale, and kept moving: 27 (filed) -> 42 (2026-08-05) -> 39 (2026-08-06). Total
records grew 364 -> 375 over the same window. This is live, continuously-mutated
state, which is why no test asserts any of these numbers and why the count was
re-measured immediately before applying.

Measured at apply time (2026-08-06):

| metric | value |
|---|---|
| total decision records | 375 |
| open / answered / dropped | 77 / 119 / 179 |
| **OPEN with a falsy `escalations_dir` (candidates)** | **39** |
| — with an `escalation_id` (the false-closable class) | 23 |
| — with `escalation_id = None` (manual/sentinel) | 16 |

Candidate project spellings, all of which had to be mapped:
`df` 8, `reify` 14, `pump_web_ui` 5, `autopilot_video` 3, `autopilot-video` 2,
`solar-challenge` 3, `dark_factory` 2, `dark-factory` 2.

Of the 23 with an escalation id: 7 resolved in exactly ONE queue, 16 were
AMBIGUOUS (the id resolves in 2+ queues), 0 resolved nowhere.

The ambiguity is broader than the task description assumed. It is not only
dark_factory's two queues: `dark-factory/data/reconciliation/escalations` is a
FLEET-WIDE recon queue holding escalations for reify / autopilot_video /
solar_challenge / know_live / pump_web_ui, so those projects' ids collide with
their own per-project orchestrator queues too (e.g. `esc-5773-1` resolves in
both `reify/data/escalations` and the dark-factory recon queue).

## 2. Tiebreak evidence, and its independent corroboration

Re-derived on the 45 already-stamped records at apply time, `session_id` still
discriminated the two queue families perfectly: 25/25 recon-queue records had
`session_id = null`; 20/20 orchestrator-queue records had `watcher-<slug>-<pid>`.

That is a measured regularity, not direct evidence, so the resolver only ever
uses it to choose AMONG queues that demonstrably hold the id, and every
uncorroborated case falls to `<unknown>`.

It was then checked against escalation CONTENT — the task's "inspect, don't
guess" requirement — by comparing each decision's `text` against the `summary`
of the same-id escalation in every holding queue.

**First attempt was discarded as an invalid instrument.** Comparing the decision
text against a concatenation of all escalation body fields produced similarity
scores of 0.01–0.23 across the board and "disagreed" on 7/16. Those margins are
noise, not signal — the bodies simply share little literal text with the
decision. Reporting that as a contradiction would have been a measurement
artifact presented as a finding.

Re-run against the correct join (decision `text` vs escalation `summary`, which
is what the watcher actually derives it from), with an explicit margin
threshold:

| verdict | count |
|---|---|
| CORROBORATED (content agrees, margin >= 0.15) | 13 |
| **CONTRADICTED** | **0** |
| INCONCLUSIVE (no decisive margin) | 3 |

Content evidence contradicts the session_id tiebreak in **zero** cases. Of the
3 inconclusive, 2 (`df-esc-2683-1`, `esc-infra-1`) point the SAME way as the
tiebreak but without a decisive margin; 1 (`esc-dirty-project-root-startup-1`)
scores 0.147 top with a 0.0 margin, i.e. says nothing at all.

**No record required hand-adjudication.** The resolver produced ZERO
`ambiguous-uncorroborated` dispositions, so the class the task said must be
resolved by inspecting content was empty — every ambiguous record's inferred
queue was corroborated by that queue actually holding the id.

## 3. Commands run

```bash
# queue topology, passed explicitly (no hardcoded fleet layout in the script)
QARGS=(
  --queue /home/leo/src/dark-factory/data/escalations
  --queue /home/leo/src/dark-factory/data/reconciliation/escalations
  --queue /home/leo/src/reify/data/escalations
  --queue /home/leo/src/pump-web-ui/data/escalations
  --queue /home/leo/src/autopilot-video/data/escalations
  --queue /home/leo/src/solar-challenge/data/escalations
  --recon-queue /home/leo/src/dark-factory/data/reconciliation/escalations
  --orch-queue df=/home/leo/src/dark-factory/data/escalations
  --orch-queue dark_factory=/home/leo/src/dark-factory/data/escalations
  --orch-queue dark-factory=/home/leo/src/dark-factory/data/escalations
  --orch-queue reify=/home/leo/src/reify/data/escalations
  --orch-queue pump_web_ui=/home/leo/src/pump-web-ui/data/escalations
  --orch-queue autopilot-video=/home/leo/src/autopilot-video/data/escalations
  --orch-queue autopilot_video=/home/leo/src/autopilot-video/data/escalations
  --orch-queue solar-challenge=/home/leo/src/solar-challenge/data/escalations
)

python3 scripts/backfill_decision_queue_stamp.py "${QARGS[@]}"            # dry run  -> exit 0
python3 scripts/backfill_decision_queue_stamp.py "${QARGS[@]}" --apply    # apply    -> exit 0
python3 scripts/backfill_decision_queue_stamp.py --verify                 # verify   -> exit 0
```

Both spellings of every inconsistent project id are mapped deliberately; the
set was derived from the recount above, not guessed.

## 4. Result

```
---- summary ----
candidates: 39
  unique-hit: 7
  tiebreak-recon: 0
  tiebreak-orch: 16
  no-escalation-id: 16
  no-holders: 0
  ambiguous-uncorroborated: 0
written: 39   write-failed: 0
```

The `--apply` dispositions were identical to the reviewed dry run. `tiebreak-recon`
is 0 because every ambiguous candidate was watcher-filed; the recon-stamped
population was already stamped by 3528's watchers and so was never a candidate.

## 5. `--verify` (task WORK item 4)

```
---- summary ----
unstamped open records: 0
```

Exit code **0**. No OPEN DecisionRecord lacks a queue stamp.

## 6. Reaper spot-check (user-observable signal)

`reap-decisions` was run against both dark_factory queues, for all three project
spellings.

| open records by stamp | before | after |
|---|---|---|
| `dark-factory/data/escalations` | 14 | **6** |
| `dark-factory/data/reconciliation/escalations` | 23 | 23 |
| `reify/data/escalations` | 11 | 11 |
| `pump-web-ui/data/escalations` | 5 | 5 |
| `autopilot-video/data/escalations` | 5 | 5 |
| `solar-challenge/data/escalations` | 3 | 3 |
| `<unknown>` | 16 | 16 |
| **total open** | **77** | **69** |

The orchestrator-queue reaper closed 8 records and **every one of them was
stamped with that same queue**. No other bucket moved:

- **No cross-queue closure.** The 23 recon-stamped and the 24 other-project
  records were untouched by a dark_factory reaper — the axis-2 guard doing its
  job on records that only have a queue stamp because of this back-fill.
- **`<unknown>` is refused, not closed.** All 16 survived every reaper pass,
  and remain visible cockpit rows for human closure.
- **Not over-blocking.** `df-esc-3512-3` is one of THIS back-fill's own stamps
  (unique-hit -> orchestrator queue) and it closed correctly against its own
  queue. That is the positive control: the stamps are honest, not merely
  restrictive.
- Every genuine blocking human gate remains present in the cockpit decision
  queue; the only records that closed were ones whose own escalation had
  already reached a terminal status in their own queue.

## 7. Residue and honest limits

- Zero residue. `--verify` is at exit 0.
- The invariant holds for records that existed at back-fill time. A decision
  filed today WITHOUT `--escalations-dir` lands straight back in the unstamped
  set — which is why both watcher SKILL.md files keep the instruction to always
  pass the flag, and now name this script as the re-runnable remedy.
- The 16 `<unknown>` records were never false-closable in the first place
  (`reap_answered_decisions` skips a falsy `escalation_id` before consulting
  the queue). They are stamped so that "no OPEN record lacks a queue stamp"
  becomes a checkable invariant rather than an aspiration.
