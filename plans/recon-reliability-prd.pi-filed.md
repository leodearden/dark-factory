# π filed — `recon-reliability` deploy capstone

Companion note to `plans/recon-reliability-prd.md` and
`plans/recon-reliability-prd.capability-manifest.md`, recording the intended
filing for task **π** — the deploy capstone for the W5 recon-reliability
batch (`ReconLedgerStore` control-plane ledger + write-both/read-new cutover,
tasks α..ο / 2219-2232). PRD §10 Phase 3, row π: "deterministic deploy
capstone. `task_kind='deterministic'`; `before_done` script does an
out-of-cgroup `systemctl --user restart fused-memory.service` (decision #6)
and verifies the new code is serving (e.g. a recon cycle runs against the
ledger; `refresh_entity_summary` edge_count > 10 sanity). *Signal:* the
running fused-memory process serves ledger-backed recon (post-restart PID +
a ledger write observed). *Prereqs:* ALL (ο + every leaf)." Capability
manifest row "π — deploy capstone (deferred-filer)": filed as a **normal**
task that commits the restart script, THEN files the
`task_kind='deterministic'` deploy (ε2 deferred-filer pattern) — "avoids
chicken-egg" between the script needing to exist on `main` and the task
needing the script to exist at filing time.

Like the W11-κ precedent (`plans/worktree-lane-lifecycle-prd.kappa-filed.md`
— which itself cites task 2233/this task as the deferred-filer pattern's
namesake, "ε2/2233 pattern"), task **#2233** commits the script and this
recipe on its own branch (steps 1-8); the actual `submit_task` filing is a
**separate** post-merge follow-up (mirroring the epsilon2 #1956→#1957 split
and the W11 #2263→#2422 split), executed once
`scripts/deploy-w5-recon-reliability.sh` has landed on `main`. This doc is
authored ahead of that filing, per the plan's own step ordering (step 9).

## Filing — planned, executed post-merge

Deferred by construction: `deterministic_task_guard.py`'s `before_done`
validation (`_validate_before_done`, ~lines 290-316, invoked from
`server/tools.py`'s `submit_task` handler — the capability manifest's
"`tools.py:~2520`" citation is stale line drift, superseded here) resolves
`before_done.script` under `project_root` and requires it `exists()` and is
`os.X_OK`, at `submit_task` time (CLAUDE.md "Deterministic task kind").
`scripts/deploy-w5-recon-reliability.sh` exists only on task/2233's own
branch as of this writing — the canonical checkout at
`/home/leo/src/dark-factory` is still on `main` (`69526ffdbf`), which does
not have it — so filing must follow the merge (CLAUDE.md "The deferred
filing", mirroring the ε2-filer #1957→#1976 and W11 κ #2263→#2424 patterns).

Once task/2233 merges, file with the race-safe sequence (deps wired BEFORE
the task is ever dispatch-eligible, so it cannot fire against a partial
dependency set):

1. `submit_task(project_root='/home/leo/src/dark-factory', planning_mode=True,
   task_kind='deterministic', title='W5-π deploy: restart + verify
   fused-memory serving the recon-reliability ledger', metadata={...})` —
   see `before_done` payload below, `always_escalates=false`, `stream='W5'`.
2. `add_dependency` for each of **2219, 2220, 2221, 2222, 2223, 2224, 2225,
   2226, 2227, 2228, 2229, 2230, 2231, 2232** — the full W5 leaf set α..ο
   (confirmed `done` as of this writing). Explicitly **NOT** 2233/π itself —
   the filer — mirroring W11 #2424 depending on the mechanism-1+2 spine but
   not its filer #2263 (2233 is already `done` by the time this filing runs,
   so including it would be redundant).
3. `set_task_status(pending)`.
4. Verify via `get_task` that the filed task shows `task_kind='deterministic'`,
   the `before_done` payload below, and all 14 dependencies.

`metadata.before_done`:
```
script:       scripts/deploy-w5-recon-reliability.sh   # task 2233, this branch
args:         []
env:          {}
cwd:          /home/leo/src/dark-factory
timeout_secs: 300
target_unit:  fused-memory.service
```

`metadata.always_escalates = false` — the auto-deploy preset (CLAUDE.md
field-combo table): run the action, escalate only on failure, else `done`.

All 14 dependencies are already `done` as of this writing, so the deploy
becomes dispatch-eligible immediately on the `pending` flip (as W11 #2424
was) — no window where it sits waiting on unfinished code.

## `target_unit` rationale — cross-unit blocking verify, not self-restart

`target_unit = "fused-memory.service"` is a **different** unit than the DF
orchestrator's own (`orchestrator-dark-factory.service`), so the
deterministic runner takes the **cross-unit** path
(`deterministic_runner.py` module docstring §γ, lines 78-84): capture
baseline unit state (`unit_inspector`) → run this script to completion,
**blocking** (`script_runner`) → if `rc != 0`, file born-at-L2 `infra_issue`
and block; else re-inspect and verify a fresh `MainPID` (>0, non-sentinel)
and a strictly-later `ActiveEnterTimestampMonotonic` → with
`always_escalates=false`, hand off to `_writeback_deploy_success`, which
stamps `before_done_verified_at` and sets the task `done` with
`done_provenance.kind='deterministic-deploy'` carrying the fresh PID and
timestamp. This is required, not optional: fused-memory is a separate
systemd unit from the orchestrator that files and runs the deploy task, so
the self-restart path (detached `systemd-run --user`, used only when
`target_unit` equals the orchestrator's own unit — the W11 κ precedent
above) does not apply here. The script's own
`systemctl --user restart fused-memory.service` is exactly what the
runner's fresh-`MainPID` verify observes, making it load-bearing.

## Ordering — restart, then health, then recon-serving

The committed script (`scripts/deploy-w5-recon-reliability.sh`) enforces
three ordered, sequential gates — never reordered, never run concurrently:

1. `systemctl --user restart fused-memory.service` (decision #6 — never
   `restart-fused-memory.sh`'s `--drain` path, which hung per task 2090).
2. A bounded poll of `curl -sf $HEALTH_URL` (default
   `http://localhost:8002/health`, `HEALTH_TIMEOUT=30`s) until fused-memory
   reports ready (graphiti+mem0 reachable — `server/tools.py`'s health
   check).
3. Only once healthy, a bounded poll of `journalctl --user -u
   fused-memory.service` (default `RECON_VERIFY_TIMEOUT=180`s) for
   `RECON_MARKER` — default the `_project_loop` startup log line "Project
   reconciliation loop started for dark_factory"
   (`fused_memory/reconciliation/harness.py:1619`), confirmed against the
   live journal on this host (task 2233 prerequisite) to recur roughly once
   a minute during normal activity, so it reliably reappears well within the
   timeout after a clean restart. No health-only fallback was needed.

Any gate's failure (health never ready, or the marker never appears) exits
non-zero with a diagnostic on stderr, which the runner treats as `rc != 0`
→ born-at-L2 `infra_issue` escalation + task `blocked` — the restart itself
is not undone or retried by the script.
`scripts/tests/test_deploy_w5_recon_reliability.py::test_apply_restarts_fused_memory_then_verifies`,
`test_apply_confirms_recon_serving`, and
`test_apply_fails_when_recon_marker_absent` pin this ordering with
fake-command witnesses (`restart_called_before_first_curl`,
`restart_called_before_first_journalctl`,
`health_passed_before_first_journalctl`) snapshotted at each fake's FIRST
invocation — proving each gate ran strictly after its predecessor, not
merely that all three happened somewhere in the run.

## User-observable signal (once filed and dispatched)

`get_task` on the filed capstone task shows `task_kind='deterministic'`,
the `before_done` payload above, and all 14 W5 leaf dependencies `done`. On
dispatch, the DeterministicRunner runs the script blocking; on success the
running fused-memory process is confirmed serving ledger-backed recon (PRD
§10 π row's *Signal*: "the running fused-memory process serves
ledger-backed recon"), verified two independent ways at once — the
runner's own baseline-vs-post-restart unit inspection (fresh
`MainPID`/`ActiveEnterTimestampMonotonic`) AND the script's own health +
recon-marker gates — rather than either proof alone.

## Consumer

None — π is the terminal deploy capstone for the W5 recon-reliability batch
(PRD §10 Phase 3, "Prereqs: ALL (ο + every leaf)"); nothing within W5
depends on it. (`plans/worktree-lane-lifecycle-prd.md` row κ and its
capability manifest cite task 2233 itself as the *precedent* for the
deferred-filer pattern they reuse — that is a documentation cross-reference
between two independent PRDs, not a task dependency on π.)
