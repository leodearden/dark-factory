# Reconciliation drain throughput: measuring the ceiling and moving it (task 3049)

Answers the task's three asks — (a) how to measure inflow and drain, (b) which levers
were pulled and on what evidence, (c) the resulting capacity claim against observed
inflow — for the reconciliation pipeline, with reify as the worked example.

Everything below is measured, not modelled, unless it is explicitly labelled as a fit.

---

## (a) How to measure

```bash
python -m fused_memory.reconciliation.throughput \
    --db data/reconciliation/reconciliation.db --project reify        # readable
python -m fused_memory.reconciliation.throughput \
    --db data/reconciliation/reconciliation.db --project reify --json # machine
```

Read-only (`mode=ro`), out-of-process, no live harness required. `--since` accepts an
ISO8601 timestamp or an hour bucket (`2026-07-25T14`) and filters inflow and drain
alike, so the two halves always cover the same window.

### Correction: inflow was NOT retrospectively derivable before this task

The task's ask (a) assumed a week of arrival timestamps could be read back out of
`event_buffer`. That premise was false, and the correction is the reason this task
needed a code change rather than just a query:

- `EventBuffer.cleanup_drained` (`event_buffer.py`) **DELETEs** drained rows older than
  `max_age_seconds` (default **3600s**). It is the *only* `DELETE FROM event_buffer` in
  the codebase. Arrival history survived roughly **one hour**, not a week.
- `EventJournal` is no fallback: it is a write-ahead log whose `mark_processed` deletes
  the row.
- The `runs` table, by contrast, has **no pruning anywhere**, so the *drain* half was
  always fully derivable retrospectively. Only inflow needed new instrumentation.

Task 3049 therefore added an `event_arrival_hourly(project_id, hour_bucket, event_type,
event_count)` rollup written **inside `cleanup_drained`'s existing transaction, with the
same WHERE clause as the DELETE**. Same-transaction is the correctness property: a row
is either still live in `event_buffer` (counted directly) or already rolled up (counted
in the aggregate) — never both, never neither. The readers union the two sources, and
the test suite pins that the total is unchanged across the rollup boundary.

**Inflow from before this rollup landed is unrecoverable.** The report says so itself in
its `retention_note`, which is generated from `cleanup_drained`'s live signature rather
than restated as prose, so retuning the window cannot turn the note into a stale claim.
Treat any reporting window extending back past the rollup as *truncated*, not as a
period of low inflow.

### The ISO8601 offset trap (METHOD NOTE)

`event_buffer.timestamp` is ISO8601 **with an offset** (`2026-07-25T13:23:22+00:00`).
SQLite's `datetime('now', ...)` renders **space-separated and offset-free**
(`2026-07-25 13:23:22`). Comparing them in SQL is a *string* comparison in which `'T'`
(0x54) sorts above `' '` (0x20), so essentially every same-day row compares greater and
a whole day collapses into one bucket — the measurement silently reads as one giant
hour.

The rule, enforced in code and pinned by a dedicated test: **bucket by parsing, never by
SQL string comparison against a `datetime('now')` literal.** `utc_hour_bucket` is the
single place parsing happens; the rollup and every reader route through it.

---

## (b) The levers and their evidence

### Mechanism (verified first-hand, and it corrects the task's mental model)

Remediation is **not** a separately-scheduled run competing for the reconciliation lock.
It is an unconditional inline tail of every completed cycle:
`_project_loop` → `run_full_cycle` → `_maybe_remediate` → `_run_remediation_pass`. The
parent run row is marked `completed` *before* `_maybe_remediate` runs — which is exactly
why ADDENDUM 2 saw a remediation run start in the same second a backlog chunk
"completed". The chunk's `run_full_cycle` had not returned; it was still in its
remediation tail.

`BacklogIterator.should_iterate` is evaluated **once per lock hold**, and its `run` then
loops chunks, each a *full cycle*. So **every chunk drags in its own remediation pass**.
That is the duty-cycle loss.

Targeted runs never acquire the lock, so they consume no drain capacity and are reported
separately and excluded from every capacity total.

### Lever 1 — stop remediation preempting backlog drain *(landed)*

ADDENDUM 2's observed sequence (2026-07-25, reify), reproduced verbatim as a test
fixture:

| run | window | wall | events |
|---|---|---|---|
| `backlog_chunk:1:393` | 13:40:30→13:56:24 | 954s | 393 |
| remediation `integrity_findings:2` | 13:56:24→14:09:00 | 756s | **0** |
| `backlog_chunk:1:326` | 14:13:36→14:29:36 | 960s | 326 |
| remediation `integrity_findings:5` | 14:29:41→14:42:00 | 739s | **0** |

Remediation duty cycle = 1495 / (1495 + 1914) = **0.438**. Nearly half of lock-held
backlog-mode wall-clock was spent in passes that drained zero events.

**This fixture validates the capacity formula end-to-end.** Taking both inputs at the
same scope — drain-only rate `(954+960)/(393+326)` = **2.662 s/event**, duty **0.438** —
`sustainable_events_per_day(2.662, 0.438, 1.0) / 24` returns **759.3 events/h**, which is
exactly the gross rate computed independently of the formula as total events over total
lock-held wall-clock: `719 / 3409s` = **759.3 events/h**. The formula is not merely
plausible here; it reproduces its own fixture.

> **Two figures that look like this one and are not** (esc-3049-1). ADDENDUM 2's prose
> quotes "~655 events/h gross", and pairing the plan's stated 2.43 s/event with duty
> 0.438 yields 832.6 events/h. Neither is the observed rate, and both are the same
> mistake in opposite directions: **mixing scopes**. 2.43 s/event is chunk 1 *alone*
> (393/954s) while 0.438 is the aggregate over *both* chunks; and ~655/h is the
> *average* event count (359.5) divided by chunk 1's *own* cycle period (33.1 min) =
> 652/h. Use aggregate-with-aggregate and the discrepancy disappears. Pinned by
> `test_sustainable_events_per_day_reproduces_the_addendum_2_observed_gross_rate`, so
> the capacity claim cannot drift from the measurement that justifies it.

Independently confirmed at production scale by the new report over the live DB
(2026-08-16, lifetime aggregate, n≫4): **reify** shows 1157 remediation runs consuming
1,519,299s against 3,949,761s of chunk+steady wall-clock — a **0.278** duty cycle;
**dark_factory** shows 0.254. The 4-run ADDENDUM 2 window was not a fluke, and it was on
the high side of the lifetime average, as a backlog-mode window should be.

`_maybe_remediate` now defers the inline pass while `_in_backlog_mode(project_id)` holds.
Three properties make this safe:

- **Lossless.** Stage-3 findings are persisted by `update_run_stage_reports` *before*
  `_maybe_remediate` is called, and forward-fed into the next cycle's S1/S2 by
  `_get_prior_s3_findings`. Deferring delays remediation; it never drops a finding. A
  test asserts the persisted `items_flagged` survive a deferral and are still returned by
  `_get_prior_s3_findings`.
- **Self-terminating.** The gate re-reads the buffer rather than taking a flag threaded
  down from `BacklogIterator`, so the answer flips on its own as the backlog drains —
  notably on the `backlog_final_consolidation` pass, which runs against a drained buffer.
  It also covers the non-iterator path, where a plain cycle's buffer has meanwhile grown
  past the threshold.
- **Bounded.** `max_backlog_remediation_deferrals` (default **1**) caps the *consecutive*
  deferral streak per project, so a permanently deep buffer cannot starve remediation
  into an integrity regression. `0` disables deferral entirely, restoring exact pre-3049
  behaviour. The counter is in-memory by design: a restart resets it, which only makes
  remediation run *sooner* — the fail-safe direction.

  The bound of 1 is **derived, not chosen** (amendment, `reviewer_comprehensive`). A
  deferred cycle writes ONE completed run (the parent) instead of the usual two (parent +
  remediation), and `_finding_persistence_count` counts completed runs that re-flag a
  finding — *including the remediating cycle's own remediation run*, which calls
  `complete_run` and `update_run_stage_reports` before the escalation gate reads the
  count. After D consecutive deferrals the cycle that finally remediates therefore sees
  D deferred parents + this cycle's parent + this cycle's remediation = **D + 2**
  re-flaggings, so D must satisfy `D + 2 < _INTEGRITY_FINDING_RECURRENCE_THRESHOLD`
  (= 4) for that counter to keep meaning *"this finding recurs DESPITE remediation"*.
  The un-deferred baseline is D = 0 → 2 (pinned by the `persistence == 2` assertion in
  `tests/test_harness.py`), i.e. escalation on the *second* failed remediation.
  At the originally-shipped 5, a backlogged project would have hit the threshold with
  zero remediation attempts behind it and escalated `recon_integrity_issue` on the FIRST
  failed remediation instead of the second — a throughput lever silently
  redefining escalation semantics. The ceiling is enforced twice: `ge=0, le=1` on the
  config field (rejects at load) and a `min()` clamp in `_maybe_remediate` (enforces at
  the point of use, for duck-typed configs), pinned to
  `_INTEGRITY_FINDING_RECURRENCE_THRESHOLD - 3` by a runtime test.

The threshold has exactly one definition: the pure module-level
`harness.is_backlog_size(count, config)`. `BacklogIterator.should_iterate` (against its
own injected config/buffer), `_select_tier`'s opus/sonnet choice and the harness's
`_backlog_state` — which `_maybe_remediate`'s deferral gate calls — all evaluate it, so
the condition that defers remediation is provably the same one that put the project into
chunked mode and onto the opus tier.

Gains ~1/(1−duty) on the same per-event rate *while the streak holds* — **~1.4x** at the
measured lifetime duty, **~1.8x** in a backlog-mode window. The bound of 1 means
remediation runs once per two cycles during a sustained backlog rather than never, so
the realised backlog-mode gain is roughly **~1.3x**, not the full 1.8x; the
`remediation_free_events_per_day` column below is therefore an *upper bound* on
post-lever-1 capacity, approached only for streak-length windows. A pure scheduling
change either way; no work is saved or skipped.

### Lever 2 — amortise the fixed per-cycle cost in steady state *(landed)*

Measured points (2026-07-25, reify): 50 events / 945s = **18.9 s/event** steady state;
393 events / 954s = **2.43 s/event** backlog chunk.

Fitting `T(B) = F + c*B` across those two points gives c ≈ 0.03 s/event — cost is almost
entirely **fixed**. Because the two points come from different tiers, the constants
shipped are deliberately **conservative**, taking chunk 2's 2.94 s/event as an upper
bound on marginal cost:

```
FITTED_CYCLE_FIXED_SECONDS    = 900.0
FITTED_CYCLE_MARGINAL_SECONDS = 2.5
```

That fit reproduces the observed steady-state point — (900 + 2.5·50)/50 = 20.5 vs 18.9
measured — and predicts (900 + 2.5·150)/150 = **8.5 s/event** at a 150-event batch: a
**2.4x** amortisation for a batching change alone.

Hence `STEADY_STATE_AMORTISATION_MIN_BATCH = 150`, and
`conditional_trigger_ratio` raised **0.2 → 0.6** (250 × 0.6 = 150).

**Latency stays bounded.** `max_staleness_seconds` is a hard trigger firing independently
of this ratio, so raising it delays the *quiescent* trigger, not the staleness backstop;
an event's worst-case wait is unchanged. A config test asserts that backstop is still
enabled, because lever 2's latency argument depends on it.

**Caveat, stated plainly:** the fit rests on two points from two different tiers. It is
optimistic-by-construction in one direction (a 900s fixed cost assumed constant across
regimes) and conservative in another (marginal cost taken at its upper bound). The
constants' docstrings say so, and both the config comment and the report tell an operator
to **re-derive them from live data** rather than trust them indefinitely.

### Lever 3 — coalesce redundant events *(deliberately deferred)*

Filed as a follow-up, not done here. The per-event_type composition the report now emits
is precisely the evidence needed to size it: on the live DB, `task_modified` is
1157/1741 (66%) of dark_factory's recent inflow and `task_status_changed` + `task_modified`
together are 92/117 (79%) of reify's. Coalescing is a **behavioural change to the buffer**
and should not ride along with a measurement + scheduling change.

### Lever 4 — `backlog_hard_limit_overrides.reify` *(deliberately UNCHANGED)*

Left at 1500. It is a backlog-*depth alarm* threshold, not a throughput knob, and the
honest answer is that it should be resized only once post-fix drain rates have been
*observed* rather than predicted.

The arithmetic a future operator needs: with sustainable drain `S` events/day and burst
inflow `I` events/day, a burst of duration `D` days accumulates `max(0, I−S)·D` backlog.
If `S > I` — which the post-fix claim asserts — backlog is **transient**, and the limit
only has to cover the peak transient excursion, not a monotonic climb. Re-run the report
after this task lands, read the real per-mode `seconds_per_event`, and size the limit
from the observed peak.

---

## (c) The capacity claim

`sustainable_events_per_day = 86400 · utilisation · (1 − duty_cycle) / seconds_per_event`,
checked against `OBSERVED_BURST_EVENTS_PER_DAY = 3500` (the task's observed ~3.5k/day
burst inflow on reify).

> **Apply the remediation penalty exactly once.** `duty_cycle` models remediation
> overhead *on top of a remediation-free rate*. It must be **0** whenever the rate
> already includes remediation wall-clock. The two tables below therefore treat it
> differently, and that asymmetry is deliberate rather than an inconsistency to
> "correct":
>
> - **Model** figures are computed from `T(B) = F + c·B`, which is one
>   `run_full_cycle`'s *own* wall-clock and **excludes** the separately-rowed
>   remediation pass. Applying `duty` there is the single legitimate application.
> - **Live measured** figures come from `drain_stats`, whose
>   `drain_seconds_per_event` is remediation-**inclusive**: remediation wall-clock is
>   in the numerator while its zero events add nothing to the denominator, so the
>   penalty is already applied once, in the measurement. Discounting it again by
>   `(1 − duty)` understates capacity by `1/(1 − duty)`. `build_report` therefore
>   passes `remediation_duty_cycle=0.0` for both of its figures and derives the
>   post-lever-1 column from the separate remediation-free rate
>   (`drain_only_seconds_per_event`) instead.
>
> An earlier revision of this document — and of the report itself — carried exactly
> that double discount in the live table. Both are corrected below.

**Model figures** (fitted constants, utilisation = 1.0, the theoretical ceiling):

| | batch | duty | s/event | events/day | vs 3500/day |
|---|---|---|---|---|---|
| pre-fix | 50 | 0.438 | 20.5 | 2369 | **insufficient**, 0.68x |
| both levers | 150 | 0.0 | 8.5 | 10165 | **sufficient**, 2.90x |

**Live measured figures** (lifetime aggregate over a copy of the real DB, 2026-08-16,
regenerated by re-running the corrected CLI — a slightly later snapshot than the duty
figure quoted under "Lever 1" above, so the live measurement has drifted marginally):

Two columns, because the report now emits two. **Observed** is the status quo the
pipeline demonstrably sustained, from the remediation-inclusive rate. **Post-lever-1**
is the same window with remediation off the drain path, from the remediation-free rate
— which is precisely what the deferral gate does. Neither has `(1 − duty)` applied on
top; the duty cycle is the *explanation* of the gap between them, and the two columns
differ by exactly `1/(1 − duty)`.

| project | duty | observed s/event | observed/day | post-lever-1 s/event | post-lever-1/day | vs 3500/day |
|---|---|---|---|---|---|---|
| reify | 0.274 | 34.71 | 2489 (0.71x) | 25.22 | 3426 | **insufficient**, 0.98x |
| dark_factory | 0.250 | 50.07 | 1725 (0.49x) | 37.58 | 2299 | **insufficient**, 0.66x |

The pre-fix ceiling really is under the burst — confirmed on production data, at large
n, on both projects, and by a route entirely independent of ADDENDUM 2's four rows. The
correction *raises* every live figure (reify's observed rate was previously reported as
1825/day) but changes no verdict: both projects remain insufficient pre-fix.

It also sharpens a conclusion that the double discount had obscured: **lever 1 alone
does not clear the burst on either project.** reify lands at 0.98x — within noise of
break-even but not over it — and dark_factory at 0.66x. Both levers are needed; lever 1
is not independently sufficient.

**Honest range for the post-fix figure.** The 10165/day model number is the optimistic
end. Anchoring on reify's *measured* rates instead: 34.71 s/event = **2489/day is the
status quo**, lever 1 alone takes it to 25.22 s/event = **3426/day** (0.98x), and
applying lever 2's 2.4x amortisation on top of that gives 25.22/2.4 = 10.5 s/event ≈
**8200/day, ~2.35x headroom**. The claim this task stands behind is therefore
"**sufficient, with headroom between 2.3x and 2.9x**" — not the single optimistic
number. (The previously stated 1.7x lower bound was an artefact of the double discount,
compounded by mislabelling the status-quo figure as the post-lever-1 one.) The lifetime
aggregate mixes every regime the pipeline has ever been in, including 1625 steady-state
runs at the old 50-event batch, so it is a lower bound; the truth is between the two,
and the report is how you find out which.

---

## Distinguishing a throughput overrun from a judge halt (task 2920)

These are different failures with different fixes, and this report is how an operator
tells them apart:

- **Throughput overrun** (this task): the pipeline is *healthy* and running. There is **no
  `halt_state` row**. Runs complete normally; inflow simply exceeds drain, so the backlog
  climbs. The report shows real per-mode wall-clock and a `capacity_verdict` of
  `insufficient`.
- **Judge halt** (task 2920): the pipeline is *stopped*. A `halt_state` row exists, cycles
  are not running at all, and the backlog climbs because nothing is draining. The report
  shows little or no recent drain wall-clock in any mode.

A climbing backlog alone does not distinguish them. **The per-mode drain breakdown does** —
which is why the readable summary leads with it.

---

## Where things live

| Thing | Path |
|---|---|
| Report + capacity arithmetic | `fused-memory/src/fused_memory/reconciliation/throughput.py` |
| Inflow rollup | `event_buffer.py` — `event_arrival_hourly`, written in `cleanup_drained` |
| Deferral gate | `harness.py` — `_maybe_remediate`, `_backlog_state`, `is_backlog_size` |
| Deferral bound | `ReconciliationConfig.max_backlog_remediation_deferrals` (default 1, `le=1`) |
| Batch size | `reconciliation.conditional_trigger_ratio: 0.6` in `fused-memory/config/config.yaml` |
| Claim-vs-config invariant | `fused-memory/tests/test_config_schema.py` |
