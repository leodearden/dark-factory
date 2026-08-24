# PRD: Scheduler-pause halt retirement + the re-testable-predicate invariant

**Date:** 2026-08-24 · **Status:** approved for decomposition · **Approach: B + H**
(contract + two-way boundary tests; the pause actuator is a load-bearing seam shared by
five trip classes across a 7-project fleet).

**Code anchors** verified against main `ccea277b78` (`2026-08-24`). Main moves fast —
cite-by-symbol; re-locate lines at implementation time.

**Provenance:** the 2026-08-20 `esc-__scheduler__-8` investigation and the measurement
campaign that followed it. Four `ewa_trip_*` halts between 2026-08-12 and 2026-08-22
(61h20m, 3h54m, 12h47m, 5m32s ≈ **78h of halted dispatch in 11 days**), plus ~48h of
`park-stop` halts on reify across six firings. Task **4559** (done, 2026-08-23) already
landed the restart-time EWA predicate re-check, the submissions-only numerator, the
per-iteration digest check and the green-tier digest knobs; this PRD builds on that work
and supersedes the half of it that presumes a halt should exist at all.

---

## 0. Consumer + user-observable surface (G1)

The consumer of every mechanism here is the **orchestrator's own dispatch loop** plus the
**operator surfaces that report why dispatch is or is not running**: the AFK digest
(`data/digests/*.md`), the escalation queues consumed by `escalation-watcher` (L2) and
`escalation-watcher-auto` (L1), the `get_scheduler_state` MCP read, and the dashboard
SCHEDULER tab. No mechanism here is a producer without a named consumer.

What an operator observes if this PRD lands:

- A high escalation-to-done ratio produces a **loud escalation and no outage** — dispatch
  keeps running, `task_started` events keep landing.
- A cost-ceiling breach still halts, and then **clears itself** once trailing-24h spend
  ages below the ceiling, with a `scheduler_resumed` event naming the evidence — no human.
- A deliberate operator halt carries a **named owner and a declared expiry**, and on expiry
  it escalates rather than silently persisting or silently resuming.
- No halt is re-asserted across a restart **except where its class policy says so**
  (§6 contract invariant 3): `cost_ceiling` re-asserts only after its predicate re-test
  still says breached, and the merge queue's `wip_conflict` rehydration (leaf η) gains the
  same re-test. `operator` and `watcher_guard` re-assert deliberately — a quiesce and a
  crashloop guard are *meant* to survive a restart — and that is a policy-table entry an
  operator can read, not an accident. (Corrected at decompose: the earlier blanket phrasing
  "no halt of any class … without its predicate being re-tested" contradicted the two
  re-assert rows in §6's own table.)

## 1. The load-bearing idea

> **A halt is permissible only for a pause class whose predicate remains durably
> re-testable while the halt is held. A class whose predicate is unmeasurable under its own
> halt may detect and escalate, but may not halt.**

Call it `halt-requires-retestable-predicate`. It is the single rule that decides every case
below, and it is checkable at design time.

The EWA and park-stop predicates are both *destroyed by their own halt*: the EWA ratio's
denominator is `max(dones, 1)` and halting drives dones to zero; park-stop counts
`blocked` transitions and halting stops tasks transitioning at all. Neither can ever
observe its own recovery, which is why both produced multi-hour latched outages. The
cost-ceiling predicate is a trailing-24h SQL query over durable spend — it is fully
measurable while paused and genuinely self-clearing, which is why it keeps its halt and
merely needs to be *asked*.

## 2. Sketch of approach

One taxonomy leaf establishes a structured pause-class enum and a single policy table;
every other leaf is that table being honoured at one site. The net change is
**subtractive** — two halt actuators are deleted, one early-return is removed, and the
remaining classes gain an owner and a bound.

## 3. Pre-conditions (G3 — assumed substrate)

Verified to exist on `ccea277b78`:

| Assumed capability | Evidence |
|---|---|
| `Harness.pause_scheduler` / `resume_scheduler` / `force_halt_scheduler` / `force_resume_scheduler` | defined in `orchestrator/src/orchestrator/harness.py` |
| `Harness._load_persisted_scheduler_pause` | same file; carries 4559's `ewa_trip_*` re-check |
| `RunStore.save_scheduler_pause` / `load_scheduler_pause` / `clear_scheduler_pause` | `orchestrator/src/orchestrator/run_store.py` |
| `scheduler_state` additive-migration pattern | `RunStore._migrate_scheduler_state_ewa_value` — the exemplar to imitate for new columns |
| `Harness._enforce_cost_ceilings` + `CostStore.cost_totals_in_window` | `harness.py`; trailing-24h window already implemented |
| `EventType.scheduler_paused` / `scheduler_resumed` / `scheduler_pause_restored` | `orchestrator/src/orchestrator/event_store.py` |
| `_file_scheduler_pause_escalation` + `has_open_l1` dedup | `harness.py` — the storm escape β/γ reuse |
| `RELOADABLE_FIELDS` green tier for `digest_*` | landed by task 4559 |

**Novel substrate, created inside this batch by leaf α (every other leaf depends on it):**
three additive `scheduler_state` columns — `pause_class TEXT`, `hold_owner TEXT`,
`expires_at TEXT` — via the `_migrate_scheduler_state_ewa_value` pattern.

**Second novel substrate, added at decompose (2026-08-24), also owned by α:** a typed
structured-facts sub-record on the `Escalation` model for a non-halting detector trip.
§6 invariant 5 requires β/γ's escalations to carry `ratio` / `window_secs` /
`dones_in_window` / `submissions_in_step` / `ewa` / `threshold` as structured fields
rather than prose — and no generic key→value payload exists on the record today.
`Escalation.evidence` is `list[EvidenceEntry]`, whose three fields
(`observation`, `measured_at`, `ref`) are all prose strings the server stores verbatim
without shape validation (`escalation/src/escalation/models.py`), so scraping it back is
exactly the INV-2 failure the invariant forbids. The house pattern for this is a typed
sub-record on the record itself — `TrainState` and `IndexHealthState` are the two
in-repo precedents. α owns it because α already owns the batch's substrate and is
upstream of BOTH consumers; letting β and γ each add their own would be a lock-step
duplication (INV-5) and a merge race on one file.

No other novel substrate; the rest is deletion and re-wiring of existing calls.

## 4. Resolved design decisions

1. **The EWA trip escalates and never halts — permanently.** Not threshold-tuned, not
   duration-capped: retired. Basis: across 184 digest windows the EWA's AUC for predicting
   an unproductive forward period is 0.56–0.64, and the escalation numerator *alone* is
   anti-predictive (0.43–0.49); all the discriminating signal is in the done count
   (0.70–0.80). Measured precision at **every** threshold and every definition of "bad" is
   0.04–0.31, against a break-even of **0.66** computed under assumptions maximally
   generous to the breaker. There is no threshold at which halting pays, so no threshold
   is chosen. No dated re-decision milestone: this is a permanent retirement.
2. **The max-pause cap is not a rescue and is not used as one.** Capping the pause scales
   the cost and the benefit of a halt equally, so it cancels out of the break-even; a
   shorter halt is a cheaper mistake, never a profitable one. Bounds appear in this PRD
   only to satisfy INV-7 for classes that *keep* a halt (leaf κ), never as a substitute
   for retiring one.
3. **Park-stop is retired on the same rule and fleet-wide** — its predicate is likewise
   self-destroying. It has fired 12 times ever (10 on reify, 2 on dark-factory) and its
   halts account for ~48h of reify halted-dispatch across the six firings that can be
   paired with a resume — of which the three multi-hour ones (13h39m, 17h40m, 16h09m) each
   carry the restart-survival signature. **Caveat, unresolved:** it is not established that
   all three were unintended; an operator may have been content to leave reify halted in
   those windows. The retirement does not rest on that figure — decision 1's rule does —
   but the figure should not be re-quoted without this caveat. This PRD owns the seam; leaf
   γ carries the paired correction to
   `plans/server-side-api-error-handling-prd.md` (see §5).
4. **Cost-ceiling keeps its halt and gains self-clearing.** `_enforce_cost_ceilings` opens
   with `if self.scheduler.is_paused: return`, so the one breaker whose predicate genuinely
   decays is never re-asked. Removing that early return — while keeping the *re-pause*
   suppressed — lets the halt clear itself when trailing-24h spend ages below the ceiling.
   This is the only class that gains an automatic live resume.
5. **Operator halts get an owner and a declared expiry, and never auto-resume.** The
   requirement is keyed on `pause_class` **through the policy table**, never on the
   `force_halt_scheduler` entry point (ratified 2026-08-24 at decompose): `no_landings`
   halts through that *same* function (`harness.py`, the no-landings breaker's
   `force_halt_scheduler(reason=trip.reason)` call), so gating the function
   unconditionally would reject an automatic breaker's halt for want of an owner it has
   no basis to invent. `force_halt_scheduler` therefore gains an explicit `pause_class`
   parameter defaulting to `OPERATOR`, and the mandatory-field check lives in the single
   policy table (INV-5). Expiry
   raises an escalation and keeps the halt. Auto-resuming a deliberate quiesce (a live
   migration, a recovery pass) is precisely what those halts exist to prevent, so INV-7's
   bound is satisfied by *surfacing*, not by releasing.
6. **Resume is class-scoped; concurrent holds compose as a set.** Today `Scheduler.pause`
   is first-wins in memory while `save_scheduler_pause` is last-write-wins on disk, so two
   concurrent halts leave memory and disk disagreeing — and `force_resume_scheduler` is
   reason-blind, so the no-landings breaker can clear a pause it did not set. The pause
   row becomes a **set of active holds**; a resume clears only its own class; dispatch
   resumes when the set empties.
7. **Task 4559's landed work is kept as forensics, not as control.** The persisted
   `ewa_value`, the submissions/gate split in the digest, and the green-tier knobs all
   stay — they are how an operator reads a trip. 4559's restart-time `ewa_trip_*` re-check
   is **removed as dead code** by leaf β: with no EWA halt there is no halt to re-check.
8. **Detection is unchanged; only the actuator changes.** The EWA and park-stop detectors
   keep computing and keep reporting. This PRD deletes no telemetry.
9. **The watcher-guard halts are RETAINED as a recorded exception to decision 1's rule**
   (ratified 2026-08-24 at decompose, after a call-site census found the §6 enum covered
   5 of the 7 live pause reasons). `watcher_misconfigured` and `watcher_crashloop`
   (`harness._check_watcher_guard` → `pause_scheduler`) are, by decision 1's own test,
   *also* self-destroying: the guard's action stops the watcher supervisor, so no further
   exits are recorded and the predicate cannot observe its own recovery. They are retained
   anyway because — unlike EWA and park-stop — **no measurement campaign has been run on
   them**, and the guard caps a watcher cost-runaway, so retiring it on an unmeasured
   analogy would trade a known bound for an unknown one. They are enumerated as
   `watcher_guard` so every live call site carries a class; retiring or re-measuring them
   is out of scope (§10). Enumerating them is not cosmetic: with a NULL class read as
   `operator` (open question 3) and decision 5's mandatory owner+expiry, an unenumerated
   watcher trip would be **rejected at set time and silently fail to halt** — during the
   cost runaway the guard exists to stop.


## 5. Cross-PRD relationship + seam owners (G4)

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/server-side-api-error-handling-prd.md` | consumes | decision 11 + boundary row 12 `park_stop_auto_resume` — an auto-resume for a halt this PRD deletes | **this PRD** (ratified 2026-08-24) | correction carried by leaf γ |
| task **3328** (pending, never dispatched) | consumes | its item 2 builds the same `park_stop_auto_resume` | **this PRD** | leaf γ amends 3328's text to drop item 2 |
| `plans/stranding-remediation-scheduler-ergonomics-prd.md` | consumes | §4 "Out of scope: Auto-resuming a paused scheduler; changing EWA thresholds" — a ratified decision this PRD reverses | **this PRD** | leaf γ records the amendment + rationale |
| task **2890** (pending) | consumes | "Resolution stays human-reserved — no auto-resume" | **this PRD** | leaf γ amends |
| task **2892** (pending, low) | consumes | EWA trip storm-classification annotation on the *pause reason string* | **this PRD** | leaf γ re-points it at the escalation; annotation survives, its carrier changes |
| task **3876** (pending, high) | produces | merge-halt rehydration re-asserting a `wip_conflict` halt without re-testing its predicate | **this PRD** | folded in as leaf η |
| task **4642** (pending, low) | consumes | "a restart can release an `ewa_trip_` halt but a running process never does" — both its acceptable outcomes (A: add an in-process release edge; B: document the asymmetry) are **mooted** by leaf β, which deletes the halt the asymmetry is about | **this PRD** | leaf γ amends + cancels it with a pointer to β |
| task **4632** (pending, low) | neither | documents the `digest_*` green-tier reload knobs in OPERATIONS.md — adjacent to leaf ι but **not** overlapping (tier documentation vs. the threshold's derivation) | 4632 | unaffected; verified at decompose |
| reify (fleet peer) | consumes | park-stop retirement is fleet-wide, and reify is where park-stop actually fires | **this PRD**; reify's operator is the consumer to notify | leaf γ |

The api-error PRD's *primary* remedy for the reify park-stop incident is M3 (5xx failures
requeue instead of writing `blocked`, so park-stop never trips). That remedy is untouched
and remains correct — only its belt-and-braces auto-resume becomes moot.

## 6. Contract (H) — the pause-class policy table

One table, defined once in `orchestrator/`, consulted by `pause_scheduler`,
`resume_scheduler`, `force_resume_scheduler` and `_load_persisted_scheduler_pause`. No site
re-implements the policy (INV-5).

```
class PauseClass(StrEnum):
    EWA_RATE        = 'ewa_rate'
    PARK_STOP       = 'park_stop'
    COST_CEILING    = 'cost_ceiling'
    NO_LANDINGS     = 'no_landings'
    WATCHER_GUARD   = 'watcher_guard'
    OPERATOR        = 'operator'
```

| class | may halt? | predicate re-testable while held? | persisted | rehydration | live recovery | bound (INV-7) |
|---|---|---|---|---|---|---|
| `ewa_rate` | **no** — escalate only | no (halt zeroes the denominator) | n/a | n/a | n/a | n/a |
| `park_stop` | **no** — escalate only | no (halt stops `blocked` transitions) | n/a | n/a | n/a | n/a |
| `cost_ceiling` | yes | **yes** — trailing-24h spend query | yes | re-test; halt only if still breached | **auto-resume on re-test** | predicate self-clears |
| `no_landings` | yes | partly (disk yes, landings no) | yes | re-test the disk limb; else re-assert | its own evidence resume, **scoped to its own class** | leaf θ backstop |
| `watcher_guard` | yes — **recorded exception, see decision 9** | no (the guard stops the supervisor, so no further exits are recorded) | yes | re-assert | **never** automatic | leaf θ backstop |
| `operator` | yes | n/a (deliberate) | yes | re-assert — deliberate holds survive by design | **never** automatic | `hold_owner` + `expires_at` required at set time; expiry escalates, never resumes |

**Invariants.**
1. A class with `may halt? = no` MUST NOT call `pause_scheduler`; it files an escalation
   carrying structured trip facts and returns.
2. A resume clears **only** the holds of its own class. Dispatch resumes iff the active-hold
   set is empty.
3. Rehydration re-asserts a hold only for a class whose policy says so, and for
   `cost_ceiling` only after the predicate re-test says the breach still holds. An
   unknowable predicate (e.g. a NULL stored value on a legacy row) fails safe toward
   **keeping** the halt — 4559's existing convention, preserved.
4. A hold of any class MUST carry `pause_class`; a hold of class `operator` MUST also carry
   `hold_owner` and `expires_at`, rejected at set time if absent.
5. Escalations raised by a non-halting detector carry structured fields (`ratio`,
   `window_secs`, `dones_in_window`, `submissions_in_step`, `ewa`, `threshold`), never
   prose to be scraped (INV-2), and are subject to the existing `has_open_l1` dedup plus a
   streak cap (INV-4).

## 7. Boundary-test sketch (H)

Each row faces both the producer (the trip site) and the consumer (the operator surface).

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | EWA above threshold does not halt | digest step computes `ewa >= digest_ewa_threshold` | no `scheduler_paused` event; `get_scheduler_state` reports not paused; one `ewa_rate_exceeded` escalation with the structured fields; `task_started` events continue after the step |
| 2 | EWA escalation storms are escaped | N consecutive above-threshold digest steps with an L1 already open | exactly one open L1; subsequent steps dedup; streak cap fires its own distinct record rather than N records |
| 3 | Park-stop above threshold does not halt | ≥ threshold `blocked` transitions inside the window | no `scheduler_paused`; one `park_stop_exceeded` escalation; dispatch continues |
| 4 | Cost-ceiling halts, then self-clears | trailing-24h spend seeded above ceiling, then aged below | `scheduler_paused(cost_ceiling)`; later, with **no human action**, `scheduler_resumed` naming the trailing-24h evidence; dispatch resumes |
| 5 | Cost-ceiling stays halted while genuinely breached | spend held above ceiling across a restart | `scheduler_pause_restored`; predicate re-tested and still breached; halt retained |
| 6 | Operator halt requires owner + expiry | `force_halt_scheduler` called without `hold_owner` / `expires_at` | rejected at set time with a structured error; no pause row written |
| 7 | Expired operator halt escalates, never resumes | operator hold whose `expires_at` has passed | scheduler still paused; an escalation naming `hold_owner`; the hold appears in every digest until cleared |
| 8 | Resume is class-scoped | two concurrent holds (`cost_ceiling` + `operator`) | clearing one leaves the scheduler paused; `get_scheduler_state` shows the surviving class; dispatch resumes only when the set empties |
| 9 | No-landings resume cannot clear a foreign hold | `no_landings` breaker's resume condition fires while an `operator` hold is active | the operator hold survives; no `scheduler_resumed` for it |
| 10 | Merge-halt rehydration re-tests (3876) | preserved pending `wip_conflict` L1; `project_root` clean, no unmerged index | merge queue is **not** halted after restart; a distinguishable log line names the re-tested predicate |
| 11 | Merge-halt still re-asserts when dirty (3876) | preserved `wip_conflict` L1; `project_root` genuinely has conflict markers | merge queue halted after restart, as today |
| 12 | A `no_landings` halt needs no owner/expiry | the no-landings breaker trips and calls `force_halt_scheduler(reason=trip.reason)` with no `hold_owner`/`expires_at` | the halt is ACCEPTED and the scheduler pauses; decision 5's mandatory-field check applies only to `pause_class = operator` |
| 13 | A watcher-guard halt still halts and re-asserts | `watcher_crashloop` threshold reached, then a restart | `scheduler_paused(watcher_guard)`; after restart `scheduler_pause_restored` re-asserts it; no owner/expiry demanded; θ's bound applies |

## 8. Decomposition plan (leaf → user-observable signal)

- **α — pause-class taxonomy + policy table + `scheduler_state` migration.** Adds
  `pause_class`/`hold_owner`/`expires_at` (additive migration, `_migrate_scheduler_state_ewa_value`
  pattern), the `PauseClass` enum, and the single policy table; converts the
  `startswith('ewa_trip_')` prose-sniffing sites to enum reads (INV-1), and adds the
  typed structured-facts sub-record on `Escalation` that §6 invariant 5 requires of β/γ
  (see §3). *Intermediate* —
  unlocks β, γ, δ, ε, ζ, κ. Signal: `get_scheduler_state` reports a structured
  `pause_class` for a held pause where it previously reported only a prose reason.
- **β — EWA trip escalates, never halts.** Deletes the `pause_scheduler` call from the
  digest trip path and, with it, 4559's now-unreachable restart re-check; keeps the
  persisted `ewa_value` and the digest EWA section as forensics. Also truths up the
  `digest_ewa_threshold` comment block in `config.py`, which currently narrates the
  restart re-check β deletes, and records there that the threshold no longer gates a halt
  (see ι, which owns the *separate* cold-start arithmetic claim). Signal: boundary rows 1–2.
- **γ — park-stop trip escalates, never halts + the cross-PRD correction pass.** Same
  actuator change for the `blocked`-transition detector, plus the prose corrections to the
  api-error PRD (decision 11, boundary row 12), the stranding PRD (§4), and amendments to
  tasks 3328 (drop item 2), 2890 and 2892. Signal: boundary row 3, plus the amended
  documents on disk. Depends on α.
- **δ — cost-ceiling self-clear.** Removes the `is_paused` early return from
  `_enforce_cost_ceilings` while keeping re-pause suppressed, and auto-resumes when the
  trailing-24h predicate clears. Signal: boundary rows 4–5. Depends on α.
- **ε — operator holds carry owner + expiry.** `force_halt_scheduler` requires both;
  expiry escalates and surfaces in the digest, never resumes. Signal: boundary rows 6–7.
  Depends on α.
- **ζ — class-scoped resume + active-hold set.** Replaces the scalar pause reason with a
  hold set; `force_resume_scheduler` clears only its own class, closing the reason-blind
  path by which the no-landings breaker could clear an EWA or operator pause, and ending
  the memory/disk divergence. Signal: boundary rows 8–9. Depends on α.
- **η — 3876: merge-halt rehydration re-tests its predicate.** The same invariant applied
  to the merge queue's `wip_conflict` rehydration; skip the halt when `project_root` is
  clean and the named worktree has no unmerged index, and annotate the owning escalation so
  it cannot re-arm. Signal: boundary rows 10–11. Independent of α (different subsystem).
- **θ — INV-7 backstop for the classes that keep a halt.** Any `cost_ceiling`,
  `no_landings` or `watcher_guard` hold held beyond a configured bound raises a distinct escalation naming the
  class and its age. Escalates; does not resume (decision 2). Signal: a synthetic hold aged
  past the bound produces exactly one bounded-hold escalation and the halt persists.
  Depends on α.
- **ι — config corrections (docs).** **RE-SCOPED at decompose (2026-08-24):** the
  units-error half this PRD described is **already landed** — task 4559 rewrote the
  `digest_ewa_threshold` provenance comment (`orchestrator/src/orchestrator/config.py`,
  the comment block above the field's `default=`), which now states outright that the
  recorded provenance is not reproducible and that the recipe "describes a DAILY EWA over
  submissions only, while the runtime computes a per-N-lifecycle-event EWA … about 2.06x
  apart on DF data". Do not re-file it. What SURVIVES and is still false on main is the
  adjacent claim in the field's own `description=`: "reaching 24.6 from a cold start
  requires sustained high ratios across multiple digest steps". With `digest_ewa_alpha`
  0.3 and a cold `EWA(t)=0` the step value is `0.3 × ratio`, so a SINGLE step trips at
  `ratio ≥ 24.6/0.3 = 82` — which is how trips 2 and 4 fired. Correct that sentence. The
  paired statement that the threshold no longer gates a halt is **owned by β**, not ι: it
  only becomes true when β lands, and β is already editing that same comment block to
  delete 4559's now-dead restart re-check. `task_kind='normal'`, docs-only. Independent.
- **κ — integration gate: the §7 boundary suite, both sides of the seam.** The leaves above
  each verify their own rows; κ verifies the *composition* — one suite running all eleven
  rows of §7 against a single harness, which is where a policy table honoured correctly at
  four call sites individually can still compose wrongly (a class-scoped resume interacting
  with a rehydrated hold of a different class, boundary rows 5+8+9 together). This is the
  B+H integration task and it is the batch's terminal leaf. Signal: the §7 suite green in
  one run, including rows 8–9 with genuinely concurrent holds of different classes.
  Depends on β, γ, δ, ε, ζ, η, θ.

**Routing notes.** All ten tasks are `task_kind='normal'`. ι (docs-only) additionally
carries `metadata.complexity='simple'` to take the single-agent fast path. Deliberately NOT
`execution_class='operational'`: that declaration is converted at submit into a deterministic
always-escalates pure gate — i.e. it routes the work to a human — and Leo ruled (2026-08-24)
that human attention is reserved for work that genuinely needs it. γ carries prose edits too
but is NOT `simple`: it also deletes a live actuator, so it is ordinary code work.

Dependencies: α → {β, γ, δ, ε, ζ, θ}; η and ι independent;
κ → {β, γ, δ, ε, ζ, η, θ}.

## 9. G7 walk (design invariants)

Walked against `docs/legibility/design-invariants.md`. This PRD **remediates** INV-1, INV-3,
INV-6 and INV-7 rather than merely passing them. Two genuine hits on its own design, both
resolved in the contract:

- **INV-4 `storm-escape-required` — HIT.** β and γ convert a self-silencing actuator (a halt
  suppresses its own re-trip) into a repeating detector that can now fire every digest step.
  Resolved: contract invariant 5 — reuse the existing `has_open_l1` dedup plus a streak cap;
  boundary row 2 is its test.
- **INV-8 `loop-thread-occupancy-bounded` — HIT.** δ makes `_enforce_cost_ceilings` run its
  trailing-24h query on dispatch ticks that previously short-circuited. Resolved: the query
  is the same bounded `CostStore.cost_totals_in_window` already executed on every unpaused
  tick; δ must not add an unbounded scan, and the paused-path re-test is rate-limited to the
  existing cost-check cadence rather than run per-iteration.

No waivers required.

## 10. Out of scope

- Threshold re-tuning of any kind, including the DF-native recalibration explored during the
  investigation — decision 1 makes it moot for `ewa_rate`.
- Replacing the EWA with a better productivity statistic. The measurement suggests a
  duration-normalised dones/hour signal (AUC 0.70–0.80) would carry real information, but
  even that sits below the 0.66 break-even for *halting*, so it would be a reporting change
  and belongs in its own PRD.
- The escalation-backlog noise sources that inflate the numerator (`esc-4004-6`'s
  recovery-veto-streak retune). Separate, and unblocked by this PRD.
- Merge-queue halt semantics beyond 3876's rehydration predicate.
- **Retiring or re-measuring the `watcher_guard` halts** (decision 9). They fail decision
  1's re-testability test on the same analogy as EWA and park-stop, but no measurement
  campaign has been run on them and the guard caps a cost runaway; this PRD enumerates
  them so no call site is unclassified, and deliberately does not act on them. A future
  PRD that measures them the way `esc-__scheduler__-8` measured the EWA owns the decision.

## 11. Open questions (tactical)

1. **Streak-cap shape for β/γ.** Reuse `StormCounter` (task 3088/3259) or a local counter?
   **Suggested:** reuse `StormCounter` if its extraction has landed by implementation time,
   else a local counter with a follow-up. Decide during β.
2. **Bound value for θ.** No data-driven basis exists — the classes it covers have never
   been observed held beyond their own recovery. **Suggested:** start at 4h, green-tier the
   knob so it is retunable without a restart. Decide during θ.
3. **Legacy pause rows.** Rows written before α's migration have a NULL `pause_class`.
   **Suggested:** treat NULL as `operator` (the most conservative class — never
   auto-resumes), consistent with 4559's fail-safe-toward-keeping-the-halt convention.
   Decide during α.
4. **Whether ε's `expires_at` is mandatory for every operator halt or defaultable.** A
   mandatory field breaks existing `force_halt_scheduler` callers. **Suggested:** mandatory
   at the MCP surface, with a documented default applied at the call site. Decide during ε.
