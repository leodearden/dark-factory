# ε2 filed — `offline-deep-test-lane-worker` (Part B)

Companion note to `docs/prds/offline-deep-test-lane-worker.md` and
`offline-deep-test-lane-worker.capability-manifest.md`, recording the outcome of task
**#1957 (ε2-filer)**: filing the ε2 leaf that Part B decompose deliberately deferred
(ε1's script did not exist yet at decompose time — CLAUDE.md's deterministic guard
validates `before_done.script` existence + executability at `submit_task` time).

## Filing

- ε2 filed 2026-07-02 as dark-factory task **#1976** — "ε2 — flip-gate-exclude-heavy
  (deterministic config auto-deploy)".
- Sequencing (§5-C6 invariant, race-safe): `submit_task(planning_mode=True, ...)` →
  wire all three deps → `set_task_status(pending)`. Deps are never wired after
  dispatch-eligibility, so ε2 could not have fired before its full dep set was attached.
- `task_kind = "deterministic"`, auto-deploy preset (CLAUDE.md field-combo table):
  `before_done` present, `metadata.always_escalates = false` — run the action, escalate
  only on failure, else `done`.
- `metadata.before_done`:
  ```
  script:       scripts/deploy/flip-reify-gate-exclude-heavy.sh   # ε1 #1956, landed 3514a76c, mode 100755
  args:         []
  env:          {}
  cwd:          /home/leo/src/dark-factory
  timeout_secs: 120
  target_unit:  orchestrator-dark-factory.service
  ```
- Dependencies: ε1 #1956 (done), ζ #1955 (done), `reify:4915`/A4 — external, qualified
  `"project_id:task_id"` form, routed to `metadata.external_deps` and resolved at gate
  time only (done).

## Open Q6 resolution — `target_unit`

`before_done.target_unit = "orchestrator-dark-factory.service"` — the DF orchestrator's
**own** unit, so the deterministic runner takes the **self-restart** path
(detached `systemd-run --user --on-failure`). Reasoning: ε1's committed script only
*signals* a config reload (`signal_config_reload()` echoes; it does not reload
in-place), and the orchestrator deep-merges reify's `verify_env` into `config` at
**load** time, caching it for the process lifetime (`config.py` `_deep_merge`,
`scheduler.__init__`) — so the flipped knob is only picked up on an orchestrator
**restart**. This matches PRD §11.6 Q6 ("the deterministic runner's standard restart
path"). The signal-only / no-restart alternative (`target_unit=None`) was ruled out:
it would leave the running config stale, so the user-observable signal below would not
hold.

## §5-C6 / B9 invariant — confirmed

> ε2 fires iff **both** ζ (#1955) and `reify:4915` (A4) are done — enforced by
> dependency edges, not timing.

Confirmed for this instance: `get_task(1976)` shows `dependencies = [1955, 1956]`
(both `done`) and `metadata.external_deps = ["reify:4915"]`; `get_external_statuses`
resolves `reify:4915` → `done`. ε2 only reached a terminal state because all three were
already satisfied — the invariant held.

## Execution note (post-filing; outside this task's scope/control)

ε2 dispatched shortly after filing (`before_done_ran_at` = 2026-07-02T04:56:08Z) and
its self-restart scheduling ran, but the runner never stamped a terminal-state
transition (`done_provenance` / `gate_escalated_at`) — the task sat in `blocked`
for hours (`metadata.reblock_guard.signature =
"infra_issue:deploy failed: orchestrator-dark-factory.service"`). This is a
**DeterministicRunner bug in the own-unit detached self-restart path**, now tracked by
**task #2004** ("Fix DeterministicRunner detached self-restart path: terminal-state
transition silently dropped after `before_done_ran_at`", in-progress at time of
writing). Task #2004 independently found the identical failure signature on task
#1982 (the IE4-deploy analog), confirming a systemic issue rather than a one-off.

Task #1976 was manually resolved by a human once the on-disk effect was confirmed
(`metadata.done_provenance = {kind: "deterministic-deploy", note: "resumed after
human resolution", unit: "orchestrator-dark-factory.service"}`), per task #2004's own
required precondition ("verify their actual on-disk config effect ... before setting
done_provenance"). This note independently re-confirms the flip is live on reify
`main` at time of writing:

```
$ grep REIFY_GATE_EXCLUDE_HEAVY /home/leo/src/reify/orchestrator.yaml
REIFY_GATE_EXCLUDE_HEAVY: "1"
```

**Status at time of writing: `done`.** No further action on ε2 itself is required from
task #1957 — task #2004 owns the systemic DeterministicRunner fix (and, per its own
description, will add a regression test covering the own-unit detached path reaching a
terminal state end-to-end).

## User-observable signal

reify `DF_VERIFY_ROLE=merge ./scripts/verify.sh --print-plan` emits `not (heavy)` after
dispatch (on-disk knob confirmed above); reverting the `orchestrator.yaml` line
restores the full gate.

## Consumer

reify merge/task gate — pulls Part A's `REIFY_GATE_EXCLUDE_HEAVY` knob (A4) to `1`.
