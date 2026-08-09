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

### Scope: which pool this timer actually drains (task 3897)

This timer's scope is the **legacy `source`-tagged pool** —
`{'source': 'stage1_flag_marker'}` — and nothing else. As measured on
2026-08-09 that filter matches **0 records in both `dark_factory` and
`reify`**, so the nightly run is currently a no-op by construction.

It is **not** the collector for the live Stage-1 → Stage-2 relay pool
(`{'flag_for_stage2': True}`, 61 records in `dark_factory` and 80 in
`reify` at the same measurement). That pool is drained in-cycle by
`_sweep_stale_mem0_flag_for_stage2_markers` (task 2966,
`fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py`),
which runs unconditionally per-project every reconciliation cycle and
age-GCs on a rolling 14-day window. Those records are **not** uncollected,
and this script deliberately counts them without ever deleting them — see
the sweep script's "Why the flag_for_stage2 pool is censused, never deleted
here" docstring section.

## Reading the output: `0 swept` is not automatically a clean bill

Because the enumeration filter above matches nothing, the nightly run
prints `orphan_count: 0` every night, and `--check`'s `backlog_verdict(0,
N)` holds unconditionally. Neither is evidence of health on its own: both
are counts taken against a pool the filter cannot see.

Task 3897 makes that legible. Every run now emits a `cross_check` block in
its JSON report:

```json
"cross_check": {
  "source_total": 0,
  "flag_for_stage2_total": 61,
  "blind_spot": true,
  "probe_failed": false
}
```

- **`blind_spot: true`** — this sweep matched 0 records while a non-empty
  adjacent `flag_for_stage2` population exists. Read `orphan_count: 0` as
  "saw nothing", not "there was nothing". A matching WARNING naming both
  counts is logged, so it also lands in
  `journalctl --user -u fused-memory-flag-marker-sweep.service`.
- **`blind_spot: false` with both totals 0** — a genuine no-op.
- **`probe_failed: true`** (with `flag_for_stage2_total: null`) — the census
  probe itself failed; `blind_spot` is then always `false`, because an
  unobserved population is never reported as an observed blind spot. The
  sweep is unaffected: the probe is count-only and can never alter the
  delete set or abort a run.

`--fail-on-blind-spot` (opt-in, default off) escalates an observed blind
spot to exit 1 so a `before_done` predicate can gate on it. It is off by
default for the same reason `--check` is absent from the recurring service
(see below): the `flag_for_stage2` pool is a healthy rolling window that is
legitimately never empty, so a gate keyed on its non-emptiness would fail
forever and teach operators to ignore it. A failed probe never trips it.

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

**Caveat (task 3897): as of 2026-08-09 this probe returns 0**, and a 0 here
does not distinguish "the drain worked" from "the timer never fired" —
both render identically against an already-empty pool. To confirm the timer
is actually firing, check `systemctl --user list-timers` and the journal
directly rather than inferring it from this count. To see the adjacent
relay pool the timer does **not** drain (and must not), probe it
explicitly:

```python
mcp__fused-memory__count_memories_by_metadata(
    project_id="dark_factory", filters={"flag_for_stage2": True},
)
```

A non-zero result there is expected and healthy — it is the in-cycle
collector's rolling 14-day window, not a backlog. Note the boolean `True`
is load-bearing: Qdrant payload filters are type-sensitive, and the string
variant `{"flag_for_stage2": "true"}` matches 0.
