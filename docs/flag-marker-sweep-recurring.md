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

### Scope: which pool this timer actually drains (tasks 3897, 3923)

> **This section is the single home for the task-3897/3923 rationale and its
> measurements.** The sweep script's module docstring, its `--fail-on-blind-spot`
> help text, `scripts/fused-memory-flag-marker-check.sh`'s header and the
> `TestFlagForStage2IsNeverDeleted` docstring all carry a one-line summary
> and point here rather than restating the numbers — these are point-in-time
> measurements of live data, and five drifting copies would leave a reader
> unable to tell which is current. Update the counts here, not there.

This timer's scope is the **legacy `source`-tagged pool** —
`{'source': 'stage1_flag_marker'}` — and nothing else. Dated census, as
re-measured on **2026-08-17** (task 3923; the 2026-08-09 figures from task
3897 are shown for trend):

| project | `{source: stage1_flag_marker}` | `{kind: stage1_flag_marker}` | `{flag_for_stage2: True}` |
|---|---:|---:|---:|
| `dark_factory` | 0 | 0 | 60 (was 65 on 08-08, 61 on 08-09) |
| `reify` | 0 | 0 | 27 (was 80 on 08-09) |
| `know_live` | 0 | 1 | not probed |

The `source` filter — the one `--check`'s verdict actually reads — matches
**0 records in all three projects probed**, now including `know_live`, which
neither task 3897 nor the task-2902 watch had probed. So the nightly run is
a no-op by construction and `backlog_verdict(0, N)` holds for every
non-negative ceiling. The single `know_live` record matching `kind` only is
marker `a5732b3b`, owned by open task **3915** — do not hand-delete it (see
the RCA's Marker Lifecycle policy); it is invisible to `total_source` and so
cannot move the verdict either way.

Those predicates are deliberately **retained, not retired**: they are
still the script's delete-set contract, still reachable via `--delete-ids`,
still the only collector for any project not yet probed, and the blind-spot
cross-check below is *defined* as the comparison between that `source`
enumeration and the adjacent population.

It is **not** the collector for the live Stage-1 → Stage-2 relay pool
(`{'flag_for_stage2': True}`, 60 records in `dark_factory` and 27 in
`reify` as of 2026-08-17). Both are draining correctly across the
measurements above while new markers are minted per run — a rolling window,
not a backlog trending to zero. That pool is drained in-cycle by
`_sweep_stale_mem0_flag_for_stage2_markers` (task 2966,
`fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py`),
which runs unconditionally per-project every reconciliation cycle and
age-GCs on a rolling 14-day window. Those records are **not** uncollected,
and this script deliberately counts them without ever deleting them — see
"Why the relay pool is censused, never deleted" below.

### Why the relay pool is censused, never deleted

The cross-check counts the `flag_for_stage2` pool and stops there. It never
enumerates it, never runs a predicate over it, and never adds it to the
delete set — a boundary enforced by `TestFlagForStage2IsNeverDeleted` in
`fused-memory/tests/test_sweep_orphan_flag_markers.py`. Three reasons, two
of them measured on 2026-08-09:

1. **23 of the 61 live records carry no usable `task_id`**, so the script's
   existing `find_taskless_markers` predicate would delete all 23 on the
   very next nightly `--apply` run. They are live Stage-1 → Stage-2 relay
   markers, not dead weight — and the nightly timer's `--terminal-drain`
   would additionally reap markers citing already-done tasks.
2. **The script has no protected-mirror guard and writes no tombstone.**
   `delete_orphan_markers` has neither the `is_protected_mirror_record`
   check nor the `record_mem0_deletion_tombstones` write that the shared
   in-cycle `_sweep_stale_mem0_pool` applies. `flag_for_stage2` is an
   LLM-supplied key any writer can stamp on any record — `mem0_tombstone.py`'s
   module docstring names this exact filter as its motivating over-breadth
   case. (Measured: 0 `cycle_summary`/`ledger_stamp` records in the pool
   today, so the risk is latent rather than active — but this script is the
   wrong place to take it.)
3. **The pool is already drained correctly** by task 2966's in-cycle
   collector, on a rolling 14-day window. A second collector here would race
   a correct one, producing duplicate deletes and duplicate tombstones for
   the same records.

Whether those records should *ultimately* be deleted is a separate question,
now adjudicable because the sweep can finally see them. Making them visible
is this script's job; deleting them is not.

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
  "flag_for_stage2_total": 60,
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

**An observed blind spot fails `--check` BY DEFAULT (task 3923).** A verdict
rendered from an enumeration that matched nothing must not read as a pass,
so `--check` exits 1 on `blind_spot: true` with no flag required.
`--fail-on-blind-spot` remains accepted as an explicit affirmation of that
default. `--no-fail-on-blind-spot` is the escape hatch: it returns `--check`
to a plain backlog verdict, for an ad-hoc census where you want the residual
count without the vacuity veto. The opt-out relaxes **only** the vacuity
check — a residual backlog over `--max-backlog` still exits 1. A failed
census probe never trips the escalation in either mode, so a transient
Qdrant blip cannot flap the verdict.

**Both spellings require `--check`.** They resolve an exit code only through
the `--check` verdict path, so on their own either would silently no-op —
exiting 0 even on an observed blind spot, a gate that cannot fail. The sweep
rejects the combination at parse time (exit 2) rather than honouring it.
`scripts/fused-memory-flag-marker-check.sh` already hardcodes `--check` in
its `exec` line. The nightly `--apply --terminal-drain` service passes
neither spelling and is unaffected — the armed default is resolved *after*
that validation precisely so the service keeps parsing cleanly.

## Decision (task 3923): the `--check` gate is retired

**Task 2902 — the esc-2866-1 O2 watch — was the only task ever wired to
`scripts/fused-memory-flag-marker-check.sh` as a `before_done.script`
predicate.** It was a one-shot *dated* milestone (`2026-07-29T12:00Z`), not
a recurring gate: it fired, exited 0, and is `status=done` with
`done_provenance.kind=deterministic-milestone`. A scan of the live task
store found no other `before_done` wiring of the wrapper anywhere in the
project (sixteen other tasks mention the sweep in metadata; all carry
`before_done: null`). **The gate has zero consumers.**

Three options were considered for the watch task, and two are refuted by the
measurements above, not merely declined:

- **(a) Retire it — chosen.** Already true de facto in the task store; the
  work was to make the retirement stick.
- **(b) Rewire it live with `--fail-on-blind-spot` — rejected.** Keeping a
  *live recurring* gate and arming it would fail forever and train operators
  to ignore it. It is also moot: there is no live gate to rewire.
- **(c) Re-point the gate at a population that varies and is meant to trend
  to zero — refuted; no such population exists.** Per the census table
  above: the `source` filter is 0 in every project probed, and `kind` is 0
  except for the single `know_live` record (`a5732b3b`, open task 3915) —
  which `total_source` cannot see, so it cannot move the verdict either. The
  only varying population, `{flag_for_stage2: True}`, is a healthy rolling
  window drained in-cycle by task 2966 while new markers are minted per run
  — legitimately never zero, so a `--max-backlog 0` gate on it would fail
  forever. The DELETE path must never be pointed at it either (see "Why the
  relay pool is censused, never deleted").

**Why arming the default is not option (b).** Arming a verdict path with
*zero consumers* means nothing fails forever, because nothing runs it. Its
only effect is that anyone re-wiring this gate in six months gets a loud
failure on day one instead of a silent pass forever. Recording the
retirement in prose alone would have been a prompt-level fix, and this
subsystem's own RCA (`plans/reify-flag-marker-backlog-rca-2026-07-22.md`
§3) records that 8 prior prompt-level fixes failed here while every fix that
held was deterministic — hence a code-level guard rather than a comment. A
gate that reads as passing from a census that saw nothing is exactly the
silent fail-soft the repo's loud-over-silent-degradation and
no-silent-fail-soft invariants forbid.

**If you trip it — the remediation path.** An rc=1 citing
`cross_check.blind_spot` is not a backlog finding; it says the gate is
keyed on a filter that sees nothing, so **fix the `source`/`kind`
enumeration to match how markers are actually tagged (or drop the gate, as
here) before wiring anything on it** — with `source` structurally 0 and
`flag_for_stage2` legitimately never empty, that failure is permanent until
the enumeration is corrected. Silencing it with `--no-fail-on-blind-spot`
restores exactly the vacuous pass this change exists to eliminate: that
flag is **census-only, never a gate configuration**.

Whether `sweep_orphan_flag_markers.py` should survive at all remains **task
3498**'s call; this decision answers its stated precondition ("check for
live consumers of the `--check` gate before removing anything") with: there
are none, and 2902 was the only one there ever was.

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

**Caveat (tasks 3897, 3923): as of 2026-08-17 this probe still returns 0**
in every project probed, and a 0 here does not distinguish "the drain
worked" from "the timer never fired" — both render identically against an
already-empty pool. To confirm the timer is actually firing, check
`systemctl --user list-timers` and the journal directly rather than
inferring it from this count. Running the check wrapper is the other way to
surface it: since task 3923 armed the default, an rc=1 citing a blind spot
means "saw nothing", which is the honest reading of that 0. To see the adjacent
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
