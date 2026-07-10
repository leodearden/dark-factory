# κ filed — `worktree-lane-lifecycle` deploy capstone

Companion note to `plans/worktree-lane-lifecycle-prd.md` and
`worktree-lane-lifecycle-prd.capability-manifest.md`, recording the intended
filing for task **κ** — the deploy capstone for the W11 mechanism-1+2 batch
(LaneLifecycle durable `.lane-state/<lane>.json` records + the `.task-meta`
sibling-dir relocation). PRD §Decomposition plan, row κ: "Migration adopt +
orchestrator restart deploy capstone (deferred-filer, ε2/2233 pattern):
commit a one-shot adopt+restart script..., then file a
`task_kind='deterministic'` self-restart-and-verify task depending on the
mechanism-1+2 spine."

Unlike the `docs/prds/offline-deep-test-lane-worker.epsilon2-filed.md`
precedent (script-authoring task #1956 and filer task #1957 were two
separate tasks), task **#2263** does both itself: it commits
`scripts/deploy-w11-lane-lifecycle.sh` on its own branch (steps 1-14), then
— once that branch merges to main — files the deterministic capstone task
described below (step 16). This doc is authored ahead of the actual filing
(step 15), per the plan's own step ordering.

## Filing — planned, executed post-merge (step 16)

Deferred by construction: `deterministic_task_guard.py` validates
`before_done.script` resolves under `project_root`, exists, and is
`os.X_OK` at `submit_task` time (CLAUDE.md "Deterministic task kind").
`scripts/deploy-w11-lane-lifecycle.sh` exists only on task/2263's own
branch as of this writing — the canonical checkout at
`/home/leo/src/dark-factory` is still on `main` (`84804cf59f`), which does
not have it — so filing must follow the merge (CLAUDE.md "The deferred
filing", mirroring the ε2-filer #1957→#1976 pattern).

Once task/2263 merges, file with the race-safe sequence (deps wired BEFORE
the task is ever dispatch-eligible, so it cannot fire against a partial
dependency set):

1. `submit_task(project_root='/home/leo/src/dark-factory', planning_mode=True,
   task_kind='deterministic', title='W11-κ deploy: adopt .lane-state +
   self-restart orchestrator onto LaneLifecycle/.task-meta', metadata={...})`
   — see `before_done` payload below, `always_escalates=false`, `stream='W11'`.
2. `add_dependency` for each of **2254, 2255, 2256, 2257, 2258, 2259, 2260,
   2261, 2262** (the mechanism-1+2 spine α..ι — confirmed `done` as of this
   writing). Explicitly **NOT** η (task 2264, mechanism 3) — see "Consumer"
   below.
3. `set_task_status(pending)`.
4. Verify via `get_task` that the filed task shows `task_kind='deterministic'`,
   the `before_done` payload below, and all 9 dependencies.

`metadata.before_done`:
```
script:       scripts/deploy-w11-lane-lifecycle.sh   # task 2263, this branch
args:         []
env:          {}
cwd:          /home/leo/src/dark-factory
timeout_secs: 300
target_unit:  orchestrator-dark-factory.service
```

`metadata.always_escalates = false` — the auto-deploy preset (CLAUDE.md
field-combo table): run the action, escalate only on failure, else `done`.

## `target_unit` rationale — self-restart, not signal-only

`target_unit = "orchestrator-dark-factory.service"` is the DF orchestrator's
**own** unit, so the deterministic runner takes the **self-restart** path
(`deterministic_runner.py:_default_schedule_detached_restart`, docstring
§ε): a detached `systemd-run --user` transient unit re-runs the script OUT
of the orchestrator's cgroup after `run()` returns, and the task goes
`done` (`done_provenance.kind='deterministic-deploy-scheduled'`)
immediately, without blocking on or being killed by the restart it
schedules. This is required, not optional, here: `LaneLifecycle` and the
`.task-meta` relocation are both load-bearing in `git_ops.py`/`harness.py`'s
in-process code paths, which only pick up a code change on a fresh
process — a signal-only / no-restart alternative would leave the running
orchestrator on old code indefinitely.

## Ordering — adopt runs BEFORE restart

PRD §Resolved design decisions #5 ("Quarantine-on-divergence, not
adopt-on-doubt... inverts the old restore-from-any-`plan.json` default
(2098) that re-poisoned lanes every restart") plus the migration caution
this deploy inherits from it: read git reality → write `.lane-state`
records ("adopt") → THEN restart. The committed script enforces this
in-process — `adopt()` runs to completion before the `exec
restart-orchestrator.sh` tail call — never the reverse, so the
new-code orchestrator finds seeded records already in place the moment it
starts serving.
`scripts/tests/test_deploy_w11_lane_lifecycle.py::test_apply_restarts_and_verifies_after_adopt`
pins this ordering with a fake-systemctl witness
(`lane_state_populated_at_restart`, snapshotted at the moment `restart` is
invoked), not merely a "both happened somewhere in the run" assertion.

Adopt itself is conservative and idempotent: a lane on a `task/<id>` branch
with a `plan.json` present (new `.task-meta/<lane>/` path checked first,
then legacy `<lane>/.task/`) is seeded `ASSIGNED`; everything else
(including a lane that *retains* a `task/<id>` branch but has no
`plan.json` — the 2098 re-poisoning guard case) is seeded `REGISTERED`; an
existing record is never clobbered (the restarted new-code orchestrator
becomes the authoritative writer once it serves). On a pool-less host —
dark_factory does not enable the warm-lane pool (task 2265), so there are
no `_lane-*` directories under `.worktrees/` — adopt is a clean no-op and
the restart alone makes the new code serve.

## User-observable signal (once filed and dispatched)

`get_task` on the filed capstone task shows `task_kind='deterministic'`,
the `before_done` payload above, and all 9 spine dependencies `done`. On
dispatch, the orchestrator restarts and serves `LaneLifecycle`-backed
acquire with the `.task-meta` relocation live, verified by a fresh
`MainPID`/`ActiveEnterTimestampMonotonic` (`restart-orchestrator.sh`'s own
blocking verify loop, delegated to by the committed script rather than
duplicated).

## Consumer

None downstream within W11 — κ is the deploy capstone for mechanisms 1+2,
gated on ι (task 2262: `"consumer_ref": "W11 κ"` in its own metadata). η
(mechanism 3, task 2264 — "unify `acquire_warm_lane`'s 7 routes over
`LaneLifecycle`") is deliberately excluded from this deploy's dependency
set: it is off this deploy path and needs only a routine restart later
(independently `done` as of this writing, via its own separate path — see
PRD "η (mechanism 3) is filed DEFERRED and held off the flip").

## Filed (step 16 executed)

Filed as task **#2424** — "W11-κ deploy: adopt .lane-state + self-restart
orchestrator onto LaneLifecycle/.task-meta" — recorded by task 2422 (this
deferred-filer follow-up). `task_kind='deterministic'`; `metadata.before_done`
matches the payload above exactly (script, args, env, cwd, timeout_secs,
target_unit); `metadata.always_escalates=false` (auto-deploy preset);
`metadata.stream='W11'`. Dependencies wired: exactly the 9 mechanism-1+2
spine tasks **2254, 2255, 2256, 2257, 2258, 2259, 2260, 2261, 2262** —
explicitly NOT 2264 (η / mechanism-3, off this deploy path), matching
"Consumer" above.

By the time task 2422 verified it, #2424 was already `status='done'`: all 9
spine deps were already `done` at filing time, so the task became
dispatch-eligible immediately on the pending flip and the DeterministicRunner
ran it straight through — no window where it sat `pending`. `target_unit=
'orchestrator-dark-factory.service'` (the orchestrator's own unit) routed it
through the detached `systemd-run --user` **self-restart** path, exactly per
the "target_unit rationale" section above: `metadata.before_done_ran_at` is
stamped, `metadata.done_provenance = {kind: 'deterministic-deploy-scheduled',
unit: 'orchestrator-dark-factory.service', transient_unit:
'orch-redeploy-restart-2424.service', fire_delay_secs: 60}` — the task went
`done` (scheduled) immediately, without the dispatching orchestrator being
killed by the restart it scheduled. The restart itself fires out-of-cgroup
~60s after scheduling (`before_done_scheduled_at`), independent of task
2422's own session.

Provenance note: #2424 was discovered already filed and already `done` when
task 2422 checked — it was not filed by task 2422's own `submit_task` call.
Some other process filed it in the window between the 2422 plan being
finalized (premise confirmed not-yet-filed at that point) and task 2422's
implementer session starting; that causal path is unaccounted for by any
agent/steward record found and has been flagged for audit (escalation
esc-2422-1), non-blocking. Task 2422 verified the existing filing
field-for-field against the full acceptance contract (task_kind, before_done,
always_escalates, stream, the exact 9-dep set) rather than re-filing, which
would have duplicated a live self-restart side effect against an
already-satisfied spine.

Verification artifact: the field-for-field match is not merely asserted here
— it is recorded in escalation **esc-2422-1** (`task_id=2422`,
`agent_role=implementer`, `severity=info`, `category=risk_identified`, filed
`2026-07-10T20:56:18Z`; fetch via `mcp__escalation__get_escalation` /
`get_pending_escalations(task_id="2422")`). Its `detail` field carries the
full reconstructed timeline (esc-2263-1 → 2422 created by the steward
19:49:57Z → 2263 merges ~20:34:34Z → 2422's plan finalized 20:41:17Z with the
not-yet-filed premise confirmed true at that instant → #2424 already
`status='done'` by 20:48:15Z → 2422's own worktree only created 20:50:14Z)
plus the exact `get_task('2424', ...)` payload it was diffed against. Any
reader can independently reproduce the same comparison by re-running
`get_task('2424', project_root='/home/leo/src/dark-factory')` — as of this
writing it returns `task_kind='deterministic'`,
`dependencies=[2254,2255,2256,2257,2258,2259,2260,2261,2262]`,
`metadata.before_done` matching the payload in "Filing" above verbatim,
`metadata.always_escalates=false`, `metadata.stream='W11'`, `status='done'`.

Dispatch note: **confirmed**, not merely scheduled — the orchestrator has
restarted onto `LaneLifecycle`-backed acquire with the `.task-meta`
relocation live. Post-restart verification observed directly (not inferred):
`journalctl --user -u orch-redeploy-restart-2424.service` shows the
committed script's own blocking verify loop running to completion —
`Restarting orchestrator-dark-factory.service (baseline MainPID=298912)...`
→ `Verifying fresh MainPID... OK` →
`orchestrator-dark-factory restarted successfully (new MainPID=470803)` —
and the transient unit itself exited `Result=success`, `ExecMainStatus=0`.
Cross-checked independently via `systemctl --user show
orchestrator-dark-factory.service`: `MainPID=470803`,
`ActiveState=active`/`SubState=running`,
`ActiveEnterTimestamp=Fri 2026-07-10 21:49:16 BST`
(`2026-07-10T20:49:16Z`) — ~61s after `before_done_scheduled_at`'s
`20:48:14.94Z` plus the `fire_delay_secs=60` delay, i.e. exactly the
scheduled fire, not an unrelated restart. No further action is required
for κ.
