# Task-verify lanes on a verify host: a fixed worktree path and a kept build directory

**Status:** active. Authored 2026-11-18 (interactive `/prd`, Leo attending).
**Code anchors** are cited by symbol and were checked against dark-factory main `1fbd702a1c`
(2026-11-18). Main moves fast; re-locate symbols at implementation time.

## 1. Goal

A remote task verify stops building in a new, randomly named, empty worktree every time.
On a host that opts in, `orchestrator verify-task` runs in one of K persistent **lanes**: a
worktree at a fixed path whose build directory survives from one verify to the next. A lane is
reset to the commit under test before each run, is never deleted afterwards, and is claimed by
a lock so two verifies never share one.

What an operator observes when this lands for reify on `leo-laptop`:

- Every reify task verify the laptop runs records which lane ran it and whether the build
  directory was kept, in the dispatch record on the workstation
  (`data/verify-logs/<task>/attempt-N.dispatch-*.json`).
- reify's laptop verify ledger shows task-role runs at one fixed `lane` path instead of a new
  `_merge-<id>` path per run.
- A lane that stops keeping its build directory raises one escalation and clears it by itself
  when the lane recovers.

## 2. Background and evidence

Today `cli.py::_verify_task_body` calls `GitOps.create_ephemeral_verify_worktree` for every
remote task verify and removes the tree afterwards. `plans/remote-task-leg-verify-prd.md` §11
lists "a persistent per-task laptop lane" as a non-goal, with no cost estimate. This PRD
replaces that non-goal for hosts that opt in.

A build experiment on `leo-laptop` on 2026-11-17 measured the cost
(`plans/warm-lane-hosts-study-2026-11-17/STAGE1-RESULTS.md`). It replayed only the four cargo
build steps of a reify task verify, with no tests run, on one pair of commits from two
different tasks. CPU is the experiment's own cgroup.

| Arrangement, same commit | Build CPU-minutes | Change |
|---|---|---|
| New path, empty build directory (today) | 90.8 and 92.8 | baseline |
| Fixed path, build directory emptied | 50.3 | 44.6% less (best case: same commit rebuilt) |
| Fixed path, build directory kept from a different task's commit | 37.5 | 59.6% less |
| Fixed path, build directory kept, same commit again | 2.2 to 3.9 | about 96% less |

Two separate effects are at work. The compiler cache (sccache) keys on the absolute worktree
path, so a new path misses every workspace crate even when nothing changed; a fixed path
removes those misses. A kept build directory additionally skips everything cargo considers up
to date. Why the path enters the cache key is being investigated separately; this PRD does not
depend on the answer.

From a separate 33-hour window of the laptop's own verify ledger (2026-11-16 to 11-17): about
31 reify task verifies a day, one at a time, with the task slot 95% busy.

The experiment measured build cost only. It is not evidence that verdicts agree; §3 ruling 2
and ε's completion check cover that.

## 3. Rulings (Leo, 2026-11-18)

1. **Lanes live on the laptop's second disk** (`/media/leo/Internal-2nd`), not the root disk.
2. **The verdict guard is the existing one**: every tenth remote task verdict is re-run on the
   workstation with the local verdict authoritative. No new guard. An operator command drops
   the kept build directories, and every run reports whether its build directory was kept so
   a divergent verdict can be traced.
3. **Only reify is enabled.** The mechanism is generic and off by default; other projects keep
   throwaway worktrees.
4. **The kept build directory is in scope**, not a later stage.

## 4. Resolved design decisions

1. **K lanes per project per host, claimed on the host by a first-free lock scan.** The
   dispatcher decides how many task verifies overlap on a host
   (`verify_hosts.py::HostPool.policy`, key `verify_admission_task_slots`) but the host never
   learns a slot number: `TaskVerifySpec` carries none. Carrying one would change the wire
   type and could go stale after a failed cancel. A lock scan needs no wire change. Precedent:
   `shared/verify_admission.py::_try_once`.
2. **Lane placement is host configuration, not part of the spec.** The stated rule in
   `task_verify_spec.py` is that verdict-deciding inputs are carried and host-shaped ones are
   not. Where a worktree lives decides no verdict.
3. **When no lane can be used the verify runs exactly as today, in a throwaway worktree, and
   says why.** It never waits for a lane and never returns a contention result. This holds for
   every failure before the verify itself starts, an unexpected error included: an exception
   in claim or prepare would otherwise exit 1 and bench the host for merge verifies too. The
   persistent merge lane escalates on a held lock instead of falling back
   (`plans/laptop-warm-verify-flock-orphan-prd.md`), because a fallback there could clobber a
   shared tree. A task-verify fallback uses a separate tree, and
   `verify_runner.py::FLOCK_CONTENTION_CATEGORY` is not infra-transient, so
   `task_verify_dispatch.py::_remote_verdict` would hand a contention result to the task as a
   red verdict.
4. **A new name band, `_task-verify-<k>`, in a lane root outside the worktree base.**
   `cleanup_merge_worktree` and `cli.py::_recorded_merge_worktree_remover` act only on `_merge-`
   entries directly under the worktree base, and the sweeps that walk the worktree base
   (`git_ops.py::PROTECTED_PREFIXES` consumers, reify's `warm-lane-gc.sh`, reify's laptop cron
   `gc-worktree-targets.sh`) never see a directory outside it. A lane root inside the worktree
   base is refused (C1).
5. **One lock hold covers reset and build.** The verify-task process takes the lane lock at
   claim and keeps it until the verify returns. The merge lane's split lifetime exists because
   `reset_persistent_merge_worktree` takes the lock itself; lanes do not copy that.
6. **The directories kept across the reset are the project's existing
   `git.reap_build_artifact_dirs`** (default `['target']`), applied by the existing
   `GitOps._clean_lane_retaining_artifacts`. No second list.
7. **A lane run is always classified cold.** The reset removes `.task/`, so
   `task_verify_run.py::prepare_task_verify_worktree` creates it fresh and
   `verify.py::_is_verify_cold` reports cold: the cold timeout tier and the cold pre-provision
   command both apply. Keeping `.task/verify_warmed` would skip pre-provision on a tree whose
   virtualenv the clean just deleted.
8. **A lane is never reset under a live process, and this work kills nothing.** After a
   cancelled verify, compiles started by the host's shared sccache server can still be running
   with their working directory in the lane. They share the server's process group, so the
   existing group reap (`shared/proc_group.py::reap_process_groups`) would kill the server and
   every project's compiles. The claim waits a bounded time for such processes to finish and
   otherwise leaves the lane alone.
9. **Disk is bounded by a free-space floor checked at claim.** Below the floor the lane's kept
   directories are dropped before the build and the run says so. This is self-contained and
   needs no timer. Extending reify's fine-grained trim to lanes is deferred (§10).
10. **The dispatch record is the single home of lane facts on the workstation.** The pass
    summary is not changed.
11. **A repeated failure to keep the build directory raises one escalation**, deduplicated by a
    sentinel and resolved automatically on recovery, following
    `merge_lane/liveness.py::_alarm_verify_host_skew`. The threshold is a constant, not a
    config key: nobody has a reason to tune it per project.
12. **Enabling on the laptop is an operator gate**, not a scripted deploy. Three reasons: the
    scripts it runs are created by γ, and `submit_task` rejects a deterministic task whose
    script does not yet exist; it needs reify's orchestrator restarted onto a dark-factory
    commit that contains this work, which is an operator's call; and the first verdicts from a
    kept build directory deserve a person reading the completion check (§5 C10).

## 5. Contract

### C1. Host configuration (`config.py::GitConfig`, read from the host's own yaml on every run)

| Key | Type, default | Meaning |
|---|---|---|
| `task_verify_lanes` | int ≥ 0, default 0 | Number of lanes. 0 means off: `verify-task` behaves exactly as it does today. |
| `task_verify_lane_root` | str, default `''` | Absolute directory holding the lanes, outside the worktree base. With lanes on, an empty or relative value, or one equal to or inside the worktree base, is reported at use as `lane_root_unavailable` (C6). It is deliberately not a config load error: a load error makes `verify-task` exit 1, which benches the host for merge verifies too. |
| `task_verify_lane_min_free_gib` | int ≥ 0, default 0 | Free-space floor on the lane root's filesystem. 0 means no check. |

### C2. Lane root marker

Lanes are used only if `<lane_root>/.task-verify-lane-root` exists and its content is the
resolved path of this host's project checkout. The installer (C10) writes it after confirming
the directory is on the intended filesystem. The check runs before any lock file is opened.
It covers two hazards: the laptop's second disk is mounted `nofail`, so an unmounted disk
would otherwise put lanes on the root filesystem; and two projects pointed at one lane root
would reset each other's lanes.

### C3. Identity and lock

Lane k is `<lane_root>/_task-verify-<k>`, k in `[0, K)`, a linked worktree of the host's
project checkout. Its lock is `verify_cancel.py::lane_lock_path(lane)`. The claim tries each
lane's lock in ascending k, non-blocking, and takes the first it gets. The descriptor is not
inherited by child processes, so the lock is released when the verify-task process dies,
whatever its children do.

### C4. Prepare, under the lock, in this order

1. **No live references.** If any process references the lane path
   (`shared/proc_group.py::scan_process_groups_under_path` returns a group), wait for it to
   drain, up to a bound fixed in the lane module (60 seconds). Still referenced: this lane is
   not used and the claim moves to the next lane. Nothing is signalled.
2. **Free-space floor.** If the floor is set and free space on the lane root's filesystem is
   below it, remove this lane's kept directories.
3. **Bring the tree to the commit.** The four states are handled explicitly:

   | Registered with git | Directory present | Action |
   |---|---|---|
   | no | no | `git worktree add --detach <lane> <sha>` |
   | yes | yes | remove the lane's own stale `index.lock` if present; check out the commit detached and forced; clean with `_clean_lane_retaining_artifacts` |
   | yes | no | drop the registration (`git worktree remove --force --force`, then `git worktree prune`), then add. This is also the state a killed `git worktree add` leaves: a registration locked `initializing` |
   | no | yes | remove the directory, then add |

4. **Recover once.** If step 3 fails, remove the registration (`--force --force`, then prune)
   and the directory, kept directories included, and add the worktree again. If that fails
   this lane is not used.

A lane that was skipped or failed in steps 1–4 contributes the reason `lane_prepare_failed`
when no other lane is claimed.

### C5. Run and release

`run_task_verify_on_worktree` runs on the lane with the arguments it takes today. On every exit
path the lane stays: `_verify_worktree_lifetime`'s cleanup, `cleanup_merge_worktree` and
`remove_merge_worktree_guarded` must never receive a lane path. The pgid file and the recorded
worktree path are written and removed as today; the recorded path names the lane, and
`cancel-verify` kills the process group and leaves the lane in place.

### C6. Fallback

With `task_verify_lanes > 0`, a run that cannot use a lane takes today's throwaway path and
reports one of: `no_free_lane` (every lane's lock was held), `lane_root_unavailable` (C1, C2),
`lane_prepare_failed` (C4, or any unexpected error in claim or prepare). No failure before the
verify starts may surface as a non-zero exit or a no-verdict. Any lane lock taken is released
before the throwaway run starts.

### C7. Lane facts on the result (`verify.py::VerifyResult`, three optional fields)

| Field | Default | Meaning |
|---|---|---|
| `host_lane` | `''` | Lane directory name when the run executed in a lane. |
| `host_lane_kept` | `False` | At least one kept directory existed after prepare and before the build. |
| `host_lane_note` | `''` | One of: `''`, a C6 fallback reason, `kept_dropped_low_disk`, `lane_recreated`. |

The fields are flat, as listed. The note vocabulary is one enum in the lane module. Three states are readable from the
fields: lanes not configured (all defaults), ran in a lane (`host_lane` set), configured but
ran throwaway (`host_lane` empty, note is a fallback reason). `host_lane_kept` says the
directory was there, not that cargo reused it. Both ends of the wire run the same commit by
construction (per-commit gate checkout), so the strict decoder
`verify_runner.py::result_from_dict` accepts the fields and
`remote_gate.py::wire_fingerprint` moves with them.

### C8. Dispatcher record

`task_verify_dispatch.py::DispatchDecision` carries the three facts of the remote result for
every dispatch in which a remote result was obtained, whatever the final decision: a plain
remote verdict, a cross-checked one, and a red confirmed locally. Today `_settle_remote`
discards the remote result on the last two; the facts are captured where the remote result is
received (`_remote_verdict`). Two more fields make the record readable on its own:
`remote_runner`, the name of the remote host that produced the result (today `runner` is
rewritten to the local host on a cross-check or a red-confirm), and `recorded_at`, the time
the record was written, in ISO form. `_write_sidecar` writes all five. A dispatch with no
remote result leaves the three lane facts and `remote_runner` at their defaults.

### C9. Degraded-lane alarm

Per remote host, the dispatcher keeps a streak of consecutive remote results that are
lane-configured (`host_lane` or `host_lane_note` non-empty) with `host_lane_kept` false.

- At 5 consecutive (a constant in the alarm module), if no pending record exists for the
  sentinel `__task_verify_lane_degraded__<host>`, file one L1 escalation: category
  `infra_issue`, role and severity as `_alarm_verify_host_skew` uses, with the last notes in
  its detail.
- A result with `host_lane_kept` true, or a result with all three fields at their defaults
  (lanes switched off), resets the streak and resolves a pending alarm.
- The escalation queue is read only when the streak crosses the threshold, when a healthy
  result follows a streak at or above it, and on the first healthy result per host after the
  dispatcher starts (an alarm left open across a restart must still clear). Never on every
  result. Queue calls run off the event-loop thread.
- The streak is in memory; a dispatcher restart starts it again.
- The queue does not exist when the dispatcher is constructed (`Harness.__init__` sets it to
  None; `Harness._start_escalation_server` creates it). The alarm is bound late, the way that
  method binds its other sinks, and does nothing but count while it has no queue.
- A person who resolves the alarm while the lane is still degraded is not paged again until a
  healthy result has reset the streak. The operator owns an alarm that stays open because
  remote results stopped arriving.

The first run in a new lane, and the run after an operator reset, legitimately report
`host_lane_kept` false; the threshold absorbs them.

### C10. Operator tools

- `orchestrator verify-lanes status --config <host yaml>` prints one JSON object: lane root,
  marker valid, and per lane whether it exists, is locked, is referenced by a live process,
  and has kept directories.
- `orchestrator verify-lanes reset --config <host yaml>` takes each lane's lock non-blocking.
  A locked or referenced lane is reported and skipped; otherwise the kept directories are
  removed. Nothing is signalled. Exit 0 when every lane was reset, 3 when any was skipped.
- `scripts/deploy/install_laptop_task_verify_lanes.sh` runs on the workstation over BatchMode
  ssh, is idempotent, and has a read-only `--check`. It refuses unless the lane root resolves
  to the expected mount point and lies outside the project's worktree directory; it creates
  the lane root and marker, backs up the host yaml and sets the three C1 keys.
- `scripts/deploy/check_task_verify_lanes_in_service.py --verify-logs <dir> --host <runner>
  --since <ISO time> --min-runs N` reads dispatch records, selecting on `remote_runner` and
  `recorded_at`, and exits with the first row that applies:

  | Exit | Meaning |
  |---|---|
  | 1 | a cross-checked lane run disagreed with its local verdict, or the most recent lane-configured record is a fallback (`host_lane` empty, note set) |
  | 0 | at least N records ran in a lane, the most recent has `host_lane_kept` true, and at least one cross-checked lane run agreed with its local verdict |
  | 2 | records with a remote result exist but none is lane-configured, or they lack the lane keys: the gate the host runs predates this work, or lanes are off |
  | 75 | not yet: too few records, or no cross-checked lane run yet |

  It prints the counts and the median `remote_wall_secs` of lane runs and of the remote runs
  before `--since`, for the operator to read; no threshold on them.

## 6. Boundary tests

Host side rows run the real `verify-task` command against a real git repository fixture
(`orchestrator/tests/_task_verify_fixtures.py::seed_verify_repo`). Dispatcher rows use the
real `TaskVerifyDispatcher`; row 21 runs the real command in-process behind the remote
runner's ssh seam, as `orchestrator/tests/test_task_verify_integration.py` does.

| # | Side | Scenario | Asserts |
|---|---|---|---|
| 1 | host | lanes = 0 | worktree name starts `_merge-`, tree removed after; result lane fields all default |
| 2 | host | lanes = 1, two runs on different commits | both run in `_task-verify-0`; a file written under the kept directory in run 1 exists in run 2; an untracked file outside it does not; the tree is at run 2's commit; the lane exists after both |
| 3 | host | run 1 result, run 2 result | run 1 `host_lane_kept` false, run 2 true, `host_lane` equal |
| 4 | host | lanes = 1, lane lock held by another process | run completes in a `_merge-` tree; note `no_free_lane`; the held lane untouched |
| 5 | host | lanes = 2, two overlapping runs | they take different lanes |
| 6 | host | marker absent, or naming another checkout | throwaway run; note `lane_root_unavailable`; no lock file created under the lane root |
| 7 | host | lane root inside the worktree base | throwaway run; note `lane_root_unavailable` |
| 8 | host | free space below the floor (floor injected) | kept directory removed before the run; run in the lane; note `kept_dropped_low_disk`; `host_lane_kept` false |
| 9 | host | lane directory present, git registration removed | lane re-added; run passes in the lane |
| 10 | host | lane registered, directory removed; and a registration locked `initializing` with a half-made directory | registration dropped and lane re-added; run passes in the lane, in both cases |
| 11 | host | checkout fails in step 3 (injected) | lane recreated; note `lane_recreated`; `host_lane_kept` false |
| 12 | host | a live process with its working directory in the lane, bound injected short | the process is still alive afterwards; run completes in a `_merge-` tree; note `lane_prepare_failed`; the lane untouched |
| 13 | host | the lane's lock cannot be opened (injected OSError) | throwaway run with note `lane_prepare_failed`; exit 0 with a verdict |
| 14 | host | `cancel-verify` during a lane run | process group killed; lane directory still present; pgid and record files gone; the next run uses the lane |
| 15 | host | pre-existing `.task/verify_warmed` in the lane | the run is classified cold and the pre-provision command runs |
| 16 | host | `verify-lanes reset` with one lane locked, one free | free lane's kept directory removed; locked lane untouched; exit 3 |
| 17 | host | `verify-lanes status` after row 2 | JSON names the lane as existing, unlocked, with kept directories, marker valid |
| 18 | dispatcher | remote result with lane facts, plain remote decision | the dispatch record carries the three values |
| 19 | dispatcher | the same on a cross-checked dispatch and on a red confirmed locally | the dispatch record carries the remote run's three values in both, `remote_runner` names the remote host though `runner` is local, and `recorded_at` parses as a time |
| 20 | dispatcher | five consecutive lane-configured results with kept false, then a sixth | one L1 with the sentinel, category `infra_issue`; the sixth files nothing; then a kept-true result resolves it; five more kept-false results file a new one; an all-default result resolves that. With an alarm pending and a freshly constructed alarm object (a restart), the first kept-true result resolves it. With no queue bound, five kept-false results file nothing and raise nothing |
| 21 | both | real command behind the ssh seam, lanes = 1, two dispatches | the second dispatch record shows `host_lane_kept` true |

## 7. Pre-conditions

None in the task graph. The symbols this work calls exist on main. The enable step (ε) needs
reify's orchestrator restarted onto a dark-factory commit containing α and β; that is part of
ε's instructions, not a task dependency.

## 8. Cross-PRD relationship

| Other work | Direction | Seam | Owner | Status |
|---|---|---|---|---|
| `plans/remote-task-leg-verify-prd.md` | extends | §11 non-goal "persistent per-task laptop lane"; `verify-task` worktree acquisition | this PRD | that PRD's batch is done; its text is a dated record and is not edited |
| `plans/laptop-warm-verify-flock-orphan-prd.md` | reuses | `lane_lock_path`, the flock helper | that PRD | wired |
| task 7038, remote verify-host disk gate | adjacent | host disk-full reports unavailable | task 7038 | pending; this PRD's floor is lane-local and does not replace it |
| tasks 4196, 9949, 4437 (sweeps and bounded GC on verify hosts) | adjacent | their enumerators match `_merge-` under the worktree base | those tasks | pending; a lane root is not a leftover and is outside their band |
| task 6263, a build forked after the watchdog's last snapshot | not fixed here | C4 step 1 skips a lane such a build still references | task 6263 | pending |
| task 7420, cancel returns 0 before the pgid file exists | adjacent | a survivor can hold lane 0's lock; the next run reports `no_free_lane` | task 7420 | pending |
| tasks 9150, 9130, 9134 (result types move to `verify_result.py`; per-leg outcomes and construction checks; host required on verify functions) | they reshape types and calls this PRD touches | `VerifyResult`; the call in C5 | those tasks | pending; the three lane fields stay outside 9130's invariants; cite by symbol, rebase whichever lands second |
| task 9123, verify cache replay on remote runners | adjacent | a replay computes its keys from the runner's own worktree, so it claims and resets a lane but builds nothing; a green from a kept directory enters the cache like any other green, under ruling 2 | task 9123 | pending |
| tasks 9266, 9277, 9268–9273 (lane-lease and `git_ops.py` splits) | they move symbols this PRD calls | `_clean_lane_retaining_artifacts`, the flock helper | those tasks | pending; cite by symbol |
| reify `scripts/gc-worktree-targets.sh` (laptop cron) and `scripts/warm-lane-gc.sh` | avoided | both walk `<reify>/.worktrees` | reify | no change: the lane root is outside it (C1) |

## 9. Decomposition plan

All four tasks are dark-factory tasks. Sizes are expected changed lines including tests.
`config.py` and `harness.py` are hub files with many pending claimants; α declares the first
and β the second, and the chain is pinned so they get the lock.

- **α. Lanes on the executing host** (about 1,000 lines). Config keys C1; a new lane module
  (marker, claim, prepare, fallback, note enum: C2–C6); `verify-task` wiring; lane facts on
  `VerifyResult` (C7); the `verify-lanes status|reset` commands (C10); boundary rows 1–17.
  `TestVerifyTaskIsAlwaysEphemeral` becomes the lanes = 0 case. New logic goes in a new module;
  a thin public wrapper in `git_ops.py` is allowed with a recorded ratchet raise.
  Files: `orchestrator/src/orchestrator/task_verify_lane.py` (new), `cli.py`, `config.py`,
  `verify.py`, `git_ops.py` (wrapper only), `orchestrator/tests/test_task_verify_lane.py`
  (new), `test_cli_verify_task.py`, `test_remote_gate.py` (its pinned wire vector moves with
  the new fields).
  Signal: two `verify-task` runs on a fixture repository with `task_verify_lanes: 1` print
  results naming `_task-verify-0`, the second with `host_lane_kept` true; with lanes 0 the
  output and the tree's fate are unchanged. Unlocks β and γ.
- **β. Dispatcher record and degraded-lane alarm** (about 600 lines). C8 and C9; boundary rows
  18–21. The alarm is a new module, constructed with the dispatcher in
  `harness.py::Harness.__init__` and given the escalation queue later in
  `Harness._start_escalation_server`; nothing goes in `merge_lane/worker.py`.
  Files: `orchestrator/src/orchestrator/task_verify_lane_alarm.py` (new),
  `task_verify_dispatch.py`, `harness.py`,
  `orchestrator/tests/test_task_verify_lane_alarm.py` (new), `test_task_verify_dispatch.py`,
  `test_task_verify_integration.py`.
  Signal: a dispatch record written by the real dispatcher for a lane run carries the lane
  facts, the remote host and a timestamp, on plain, cross-checked and red-confirmed
  dispatches; five kept-false results file one L1 and a kept-true result resolves it. Depends on α. Unlocks γ's in-service check and ε.
- **γ. Host provisioning and the in-service check** (about 500 lines). The two scripts of C10
  with tests using fake ssh shims
  (`scripts/tests/test_enable_laptop_persistent_worktree.py` is the pattern) and fixture
  dispatch records, and an `OPERATIONS.md` section: enabling, the exact command to run
  `verify-lanes` on a host, reading lane facts, the alarm, and when to reset lanes (a
  divergent cross-check, a power loss or clock step on the host).
  Files: `scripts/deploy/install_laptop_task_verify_lanes.sh` (new),
  `scripts/deploy/check_task_verify_lanes_in_service.py` (new),
  `scripts/tests/test_install_laptop_task_verify_lanes.py` (new),
  `scripts/tests/test_check_task_verify_lanes_in_service.py` (new), `OPERATIONS.md`.
  Signal: `install_laptop_task_verify_lanes.sh --check` against a shimmed host exits 0 after an
  apply and non-zero when the lane root is not on the expected mount;
  `check_task_verify_lanes_in_service.py` returns 1, 0, 2 and 75 on fixture record sets, one
  per row of the C10 table plus one that meets two rows and takes the first. Depends on α and β. Unlocks ε.
- **ε. Enable lanes for reify on `leo-laptop`** (operator gate; no code). Steps: confirm
  reify's orchestrator code commit descends from β's merge
  (`scripts/orchestrator-watchdog.py --report`), restarting it if not; run the installer for
  `~/.config/orchestrator/reify-laptop.yaml` with lane root
  `/media/leo/Internal-2nd/reify-task-verify`, expected mount `/media/leo/Internal-2nd`, one
  lane and a floor of 150 GiB; wait for laptop task verifies; run the in-service check against
  `/home/leo/src/reify/data/verify-logs` with `--min-runs 5`; record its printed counts and
  medians in the resolution.
  Signal, and the goal signal of this PRD: the in-service check exits 0, and reify's laptop
  ledger shows task-role rows at the fixed lane path. Depends on α, β, γ.

## 10. Out of scope

- Lanes for dark-factory, solar-challenge or solar-challenge-platform verifies.
- A copy-on-write lane pool, a warm base, or agent lanes on a remote host.
- Changing how sccache keys its cache.
- Killing leftover processes in a lane.
- Extending reify's hourly target trim (`scripts/prune-unconsulted-target.sh`) to lanes.
  Trigger to revisit: `kept_dropped_low_disk` appears in dispatch records more than once a
  week for a lane.
- Preserving reify's gui sidecar placeholder across the reset, which would save the one-package
  rebuild measured in the experiment (reify task 8117 concerns the same file).
- Any change to the cross-check rate or to `verify_task_remote_red_confirms_locally`.
- With more than one lane, an idle lane's kept directory can hold the filesystem below the
  floor while a busy lane keeps dropping its own. Not addressed: reify runs one lane.

## 11. Open questions (tactical)

1. **How the installer edits the yaml.** Follow `enable_laptop_persistent_worktree.sh`. Decide
   at γ.
2. **Whether the 60-second drain bound suits the laptop.** It is a module constant. Revisit if
   `lane_prepare_failed` shows up after cancels in the first week.
