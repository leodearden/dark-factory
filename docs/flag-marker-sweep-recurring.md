# Fused-memory flag-marker sweep: recurring drain

## Purpose

`fused-memory/scripts/sweep_orphan_flag_markers.py` (task 2596) is a
complete, tested, deterministic drain of dead-weight `stage1_flag_marker`
records — orphan / taskless / stale / terminal-task predicates, plus
targeted `--delete-ids` correction. Task 2596 shipped the script but never
wired it into anything that actually runs it, so the backlog only grew
(reconfirmed at 42 records before this task). Task 2693 wires the existing
script into an automated, recurring systemd-timer drain so the backlog
stays non-growing without any human running it by hand.

## Install / enable

Run on the host (not from a task worktree — see "Why this is an
operational step" below):

```bash
scripts/install-flag-marker-sweep-timer.sh
```

This installer:

1. Copies `scripts/fused-memory-flag-marker-sweep.{service,timer}` into
   `${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/`.
2. Runs `systemctl --user daemon-reload`.
3. Runs `systemctl --user enable --now fused-memory-flag-marker-sweep.timer`
   — arms the nightly recurrence.
4. Runs `systemctl --user start fused-memory-flag-marker-sweep.service` —
   kicks an immediate one-time drain of the current backlog, rather than
   waiting for the next scheduled 03:30 run.
5. Self-verifies `fused-memory-flag-marker-sweep.timer` is actually listed
   in `systemctl --user list-timers --all` afterward, and exits non-zero
   with a diagnostic if it is not. This closes the exact failure class that
   produced this task in the first place: a sweep that was meant to run but
   was never actually wired/enabled, with the escalation flagging that
   auto-dismissed as stale.

Idempotent — safe to re-run (e.g. after editing either unit file).

## Env requirements

`scripts/fused-memory-flag-marker-sweep.sh` (the unit's `ExecStart`) sources
the repo's `.env` and needs:

- `OPENAI_API_KEY` — for embeddings.
- FalkorDB and Qdrant reachable (`FALKORDB_URI`, default
  `redis://localhost:6379`).

This mirrors the runbook lesson from
`fused-memory/scripts/cgl_eta_auto_apply.sh`: a fused-memory maintenance
action must run under the service env, not a bare shell, or the census
silently narrows.

## Schedule

Nightly at 03:30 local time (`scripts/fused-memory-flag-marker-sweep.timer`,
`Persistent=true` so a missed night catches up on next boot/login). Each run
drains via:

```
sweep_orphan_flag_markers.py --apply --terminal-drain
```

## Why no `--check` in the recurring service

The sweep's own docstring/WARNING (see `run()`'s `undated_kept_count`) notes
that markers with a missing or unparseable `created_at` can never be
drained by `find_stale_markers`, at any age cutoff — this sets a residual
floor on the backlog. A recurring `--check --max-backlog 0` service would
therefore enter systemd `failed` state on every run, forever, whenever any
undated marker exists — a self-inflicted perpetual-failure footgun. Dropping
`--check` from the recurring service lets each nightly drain exit 0 on a
normal run.

## Backstop for residual backlog

Persistent or undrainable residual (chiefly the undated-marker floor above)
is already surfaced by the existing reconciliation Stage-1/2 re-flag net —
the very mechanism that filed tasks 2596 and 2693. No new escalation glue
was added for this. A future enhancement could add a 2663-2666-style
delayed-predicate born-at-L2 tripwire (e.g. "N days after this task lands,
check the backlog is under threshold X, else escalate") if the reconciliation
backstop proves insufficient in practice — not built here, since it would
need additional untested wrapper glue beyond this task's scope.

## Why the drain is an operational step, not part of this PR

Draining the live backlog requires live Mem0/Qdrant + secrets and mutates
shared project memory — doing that from an unmerged task branch is unsafe
and isn't a deterministic CI assertion. `scripts/install-flag-marker-sweep-timer.sh`'s
`start` step runs the same `--apply` drain on the host where the stores and
venv are actually live, so enabling the timer and running the first drain is
one reproducible, committed action — it just has to be run post-merge, on
the host, by an operator. The tests added by this task
(`scripts/tests/test_flag_marker_sweep_wrapper.py`,
`scripts/tests/test_install_flag_marker_sweep_timer.py`) assert the wiring
behavior via fakes (a fake sweep-command recorder, a fake `systemctl`); they
do not and cannot assert the live drain outcome.

## Verifying the drain worked

After running the installer (or after the first nightly firing), check the
backlog is trending toward zero:

```python
mcp__fused-memory__count_memories_by_metadata(
    project_id="dark_factory", source="stage1_flag_marker",
)
```

A residual count near the undated-marker floor (see above) is expected and
healthy; a count that isn't shrinking at all across multiple nightly runs
means the timer isn't actually firing — check `systemctl --user
list-timers` and `journalctl --user -u fused-memory-flag-marker-sweep.service`
on the host.
