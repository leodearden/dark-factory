# A reserve account for urgent work: one account of the shared pool held back for high-priority tasks until shortly before its weekly reset

**Status:** active. Authored 2026-11-16. Rulings by Leo on 2026-11-16 (section 3). Not yet decomposed.
**Code anchors** verified against main `ba39519b67` (2026-11-16). Main moves fast: cite by symbol; re-locate at
implementation time.
**Evidence:** `plans/cross-project-weights-study-2026-11-12/`: the study report (`REPORT.md`, sections 6 to 9), the
allowance seat's report (`allowance.md`), the rulings search (`rulings.md`), and the 2026-11-15 follow-up on account
limits (`followup-1115/caps.md`). The follow-up reports had no critic pass; the figures taken from them here are
named in section 2 with their window. This document had a critic pass and two re-checks; the corrections from
the second re-check were folded in without a third.

Words used throughout:

- **The pool**: the seven Claude subscription accounts in `config/usage-accounts.yaml` that every orchestrator, the
  memory service and the nightly jobs share. Order today: max-d, max-e, max-f, max-g, max-b, max-c, max-h.
- **The gate**: `shared/src/shared/usage_gate.py::UsageGate`, the object each process uses to pick an account for
  one agent call. Each process has its own; they share nothing at run time.
- **The reserve**: the one account this PRD holds back (max-h).
- **Protected window**: from the reserve's weekly reset until the release. Only admitted work may use the reserve.
- **Release window**: the last hours before the reserve's weekly reset. All work may use it, and it is tried first.
- **Tier**: a task's priority, one of critical, high, medium, low, polish. *Effective tier* is the scheduler's:
  the task's own tier, or that of a higher-tier task waiting on it.
- **Slot**: one of a project's `max_concurrent_tasks` permits. A started task holds a slot, its file locks and its
  worktree until it ends.

## 1. Goal

Today the gate gives every caller the first open account in file order and knows nothing about what the call is
for. When the pool runs short, whichever work asks first spends what is left, and most of it is low priority.

This PRD holds one account back. During its protected window the reserve serves only high and critical tasks and a
short list of essential housekeeping calls. In its release window it opens to everything and is tried first, so
that what is left is spent before the weekly reset. The rule is a fixed calendar and a fixed table. It needs no
state shared between processes, no forecast and no knowledge of how much any account has left.

What an operator observes when it has landed:

1. When the other six accounts are capped and the reserve is in its protected window, high and critical tasks that
   are running keep running on the reserve, and a newly ready one starts as soon as a slot is free.
2. A medium or low task that needs an account then waits where it is, as it waits today when all seven are capped.
3. The scheduler does not start a task the reserve would refuse, so freed slots go to admitted work. It records
   each tick in which it held tasks back, with the release time.
4. From 36 hours before the reserve's reset, every caller is served by the reserve first.
5. Each process records, in its own store, when it was refused the reserve, how long it waited, and what the
   reserve served in its protected window, with the project, tier, kind of work and cost.
6. One command prints, per week, whether the reserve was spent before its reset, who spent it, who waited, and how
   long admitted work waited for a slot behind waiting tasks.
7. With no `reserve:` block in the accounts file, nothing changes.

## 2. Background and premise validity (G6)

From the study and its follow-up. Each figure names its window and population. "W0" is 2026-10-25 to 11-08 (the two
weeks before the pause of three projects); "W1" is 2026-11-09 to 11-12. Population unless stated: 16,661 invocation
rows in W0 from all nine projects' stores and the memory service's.

- **The pool is short nearly all the time.** In W0 at least six of the seven accounts were capped 37% of the time
  and all seven 10.2% of the time (`REPORT.md` section 6).
- **Most of what drains it is low priority.** In W0, by the tasks' priority as read on 2026-11-12, 61% of recorded
  spend went to low tasks, 20% to medium and 15% to high and critical. Across the ten stalls of half an hour or
  more in W0 and W1, 50 to 62% of the allowance whose absence caused the stall had gone to tasks now ranked low.
  That is an upper bound on what a rule favouring urgent work could have held back.
- **The long stalls fall in one gap.** The longest gap between two weekly resets is Monday 21:00 UTC (max-h) to
  Wednesday 10:00 (max-c), 37 hours. Seven of the ten stalls and 37.0 of their 51.7 hours fell inside it.
- **max-h resets at the start of that gap and has lately been spent within a day and a half.** Its reset was at
  21:00 UTC on each Monday from 2026-10-12 to 11-09, across both the British and the American clock change
  (`followup-1115/caps.md`, the per-week table; the banner's zone is applied by
  `shared/src/shared/invocation_outcome.py::_parse_resets_at`). In its weeks beginning 10-26, 11-02 and 11-09 it
  reached its weekly limit 33, 23 and 26 hours after the reset. In its weeks beginning 10-12 and 10-19 it did not
  reach its limit at all. It is last in the pool order and last in the interactive proxy's order.
- **An account's weekly limit is about $2,190** (median of ten caps on max-d to max-h since 2026-11-01). The unit is
  list-price dollars with spend on the Fable 5.1 model counted about twice. It is a lower bound, because some
  consumers record no spend.
- **Recorded spend was $12.3k a week in W0, about $73 an hour,** and $9.6k a week, about $57 an hour, in W1 with
  three projects paused (`allowance.md` F2). These count Fable spend once.
- **The 5-hour limit is gone.** The last 5-hour cap in the factory's records was on 2026-10-30
  (`followup-1115/caps.md`). The release window is sized on the weekly limit alone.

This is the perishable-capacity problem of revenue management (Littlewood's rule) and of trunk reservation in
telephone networks: hold a protection level for the higher class and let it fall to zero before the capacity
expires. A reserve of that kind does better than fixed shares per class and tolerates a wrong forecast, because it
uses none.

**Premises checked.**

- *A refusal is not a new kind of jam.* A refused caller waits under the bound it has today (14 days for a task
  role, far longer than the longest protected window of 132 hours, so a reserve wait never blocks a task). The wait
  has an owner, the gate's own clock and account transitions, and survives a restart because nothing is stored. A
  caller with a short bound (the dry-run unblock, 30 minutes) ends as it ends today on a capped pool.
- *It is never worse for urgent work than today, and it is not perfect.* Today, in the same state, the reserve is
  already spent or is being spent by everything. With this PRD the admitted tasks already running continue. A
  newly ready admitted task needs a free slot, and waiting lower-tier tasks keep theirs, so it starts when a
  running admitted task ends. If no admitted task is running when the pool runs dry, a new one cannot start until
  an account reopens. The report measures this wait and an escalation fires when it passes two hours (section 6.5).
  Lending waiting tasks' slots would remove it and is left out on purpose (section 9).
- *The release can probably be spent in time; it is not certain.* A completely unspent reserve of $2,190 takes at
  most about 30 hours of the whole fleet's recorded spend at W0's rate and about 38 hours at W1's. The release
  window is 36 hours and the reserve is tried first during it. Admitted work will have spent part of it already.
  The weekly report says whether it was spent; the hours are a config value.
- *Admitted work fits in the reserve.* High and critical work was 15% of W0's recorded spend, about $1.85k a week
  over the whole week. It draws on the reserve only while the other six are capped. This is a projection from W0,
  not a measurement of the new rule.
- *The reserve's reset time can be configured.* A process learns an account's reset time only from a cap banner, so
  for an account that has not capped this week the time is unknown to it. The reset hour was the same in UTC in
  every week measured, so it is a config value.
- *No number in a leaf signal is a guess.* No leaf asserts a spend, a share or a time saved. The report prints
  recorded facts and labels its one estimate (allowance left, from the $2,190 figure) as an estimate.

## 3. Resolved design decisions

Rulings by Leo, 2026-11-16:

- **R1. Reserve by account, not shares per project.** "I now recommend the reserve by account over fixed shares" —
  "Agreed. Do this." It replaces the per-project trailing-spend budget recommended on 2026-11-12.
- **R2. Who is admitted before the release:** tasks ranked high or critical, in every project. No ranking of
  projects yet; the table takes a per-project list so one can be added later without code.
- **R3. Release:** 36 hours before the reset, and the reserve is tried before the other accounts during it. (The
  question put to Leo gave the fleet's spend as $90 to $100 an hour; the recorded figure is $73. 36 hours is kept
  as the starting value; see section 2.)
- **R4. Interactive sessions:** no change to the interactive proxy. max-h stays last in its order, so interactive
  work reaches the reserve only when account A and the other six are all exhausted.
- **R5. Sequencing:** the reserve lands before the usage-gate refactor chain from the whole-project review.

R1 and R3 are later than, and take precedence over, the study's reading (`REPORT.md` section 7) that the rulings of
2026-08-30 and 2026-10-06 forbid a reserve account. Those rulings themselves stand: evals share the pool (here, as
background work), and max-b and max-h are never disabled or commented out.

Design decisions taken in authoring (Leo has not ruled these one by one; each is open to override):

- **D1. The reserve is max-h, named in config.** It is last in the order already, and its reset opens the 37-hour
  gap. It is named, not "whichever is last", so that a reorder cannot move the reserve by accident.
- **D2. One home for the rule: a `reserve:` block in the shared accounts file.** Every process already reads that
  file (reify's is a symlink to it; the rest take the packaged default path). No project's config gains a key.
- **D3. The table lists admitted tiers; it does not rank them.** `admit_tiers` is a list of tier names per project
  with a default. The gate tests membership, so no tier order is needed in `shared/` (*SPOT*). The default is
  `[critical, high]`.
- **D4. A pin is not a separate case.** Today a pin does not change a task's effective tier, and the starvation
  watchdog also sets pins of its own, so "pinned" cannot mean "Leo wants this first". A pinned low task is not
  admitted; an operator who wants it admitted raises its tier with the override's boost. When the priority PRD's
  `pinned` tier lands (task 9938), adding `pinned` to `admit_tiers` admits pinned tasks and their prerequisites, by
  config alone. That would admit the watchdog's pins as well, and those are dropped the moment the task is
  started, so such a task would be started and then refused; the operator note in task 6 says so.
- **D5. Callers say what the work is; the gate decides.** A caller passes a `WorkClass`. The admission rule lives in
  one pure module. No caller tests the calendar or the table.
- **D6. Three kinds of work.** *Task*: acts for one task, admitted by its tier. *Essential*: admitted always.
  *Background*: never admitted before the release. A caller that passes nothing is treated as background and
  recorded as unclassified. The assignment is section 6.1's table.
- **D7. The tier is the effective tier.** A low task that a critical task waits on is admitted. The scheduler
  already computes it. Where it is unknown the task's own tier is used, and failing that `medium`. A waiting
  task's tier is read again while it waits, so raising it (an operator's boost, or a critical task coming to
  depend on it) admits it within minutes.
- **D8. A refused task waits where it is, holding its slot and its locks.** This is what happens today when every
  account is capped. Its locks stay held until its merge lands, as Leo ruled on 2026-11-12. Its slot stays held
  too: handing it to other work for the length of the wait was examined and left out (section 9). The cost is
  measured and escalated (section 6.5).
- **D9. The scheduler does not start what the reserve would refuse,** in its eligibility test and again at the
  moment a slot is in hand. Otherwise each freed slot would go to a task that waits at once. A held task is not
  a candidate, so the starvation watchdog does not promote it while it is held. Its age keeps counting, so when
  the hold lifts the watchdog may promote a few long-held tasks at once, up to its own limit of three. The check
  asks the gate; nothing is stored.
- **D10. The dispatch check covers the reserve case only.** When every account is capped the scheduler still
  dispatches and tasks wait at the gate, as today. Whether to change that is a separate question (section 9).
- **D11. The calendar comes from config alone.** A cap banner cannot say whether its reset time is the weekly one,
  so banners never move the release window. A banner time close to the configured reset but not equal to it is
  recorded as drift.
- **D12. A bad or unusable reserve means no reserve.** A malformed block, an unknown account or tier name, or the
  reserve account failing authentication (Leo switches max-b and max-h on and off at Anthropic's side) leaves the
  pool behaving as it does today, and says so in a recorded event and one escalation. It never stops a process
  starting. A process learns that the account is failing only by trying it; until then its low-tier work waits as
  it would on a capped pool.
- **D13. Reconciliation steps aside.** It does not start a cycle while the gate would refuse background work, and
  a judge that falls due then is left pending and fired once the refusal has ended. Refused judges would otherwise
  count as infrastructure failures, and three in a row halt reconciliation and escalate.
- **D14. No shared state, nothing remembered.** Each process applies the rule from the file and the clock. This
  stays inside the 2026-11-12 ruling against a cap file or cap authority.
- **D15. Approach: contract and two-way boundary tests (B + H).** The change crosses `shared/`, `orchestrator/`,
  `fused-memory/` and `scripts/`, and the gate is a load-bearing seam.

## 4. Pre-conditions (G3)

Verified in code at the anchor commit:

| Assumed capability | Where | State |
|---|---|---|
| Selection is one walk in file order; the caller's exclusion is tested before a probing account's single probe slot is claimed | `usage_gate.py::UsageGate.try_lease`, called by `::before_invoke` under the gate's lock | exists |
| A caller-specific refusal that waits without freezing the pool | the per-scope waiter in `UsageGate.before_invoke` (Fable scope caps): waits on its own event with a timeout, never touches `_open` or `_paused_reason` | exists. Its event is set only by the scope sweep, not by account transitions, so the reserve waiter needs its own event, set in `UsageGate._transition` |
| Nothing wakes a caller on the all-capped path at a clock time without an account event | `UsageGate._wait_for_any_account_to_reopen` | confirmed absent, so the reserve wait must not use that path |
| `_transition` is the one writer of an account's phase | `UsageGate._transition` | exists |
| Every production agent call that uses an account passes through one function | `shared/src/shared/cli_invoke.py::invoke_with_cap_retry`, gate as first parameter | exists. Callers that invoke an agent directly, or pass no gate, never reach the gate and are untouched |
| What a spent wait does | `PoolFrozen` becomes `AllAccountsCappedException`; a task workflow then blocks the task (`workflow_types.py`, the disposition table) and the block counts toward its requeue cap | exists; not reached by a reserve wait under the 14-day bound |
| Effective tier per task | `scheduler.py::Scheduler.task_priority` returning `orchestrator/src/orchestrator/task_priority.py::TaskPriority`; handed to workflows as `TaskWorkflow._task_priority` | exists (task 9908, landed 2026-11-14). Unknown before the first tick; can lag while the main loop waits for a free slot |
| Tier vocabulary | `orchestrator/src/orchestrator/config.py::PRIORITY_TIERS` | exists; `pinned` is not in it |
| One eligibility test behind every dispatch path | `Scheduler._eligible_for_dispatch`, called by `_phase_build_candidates` (which feeds scored selection and the starvation watchdog) and by `_dispatch_candidacy` (the pin queue) | exists |
| The scheduler has no reference to the gate | `scheduler.py` | confirmed; the harness hands it a narrow callable, as it does `TaskPriorityLookup` |
| The main loop takes an assignment, then waits for a slot, then starts it | `harness.py`, the run loop: `scheduler.acquire_next()`, `await sem.acquire()`, `_launch_slot` | exists; the assignment already holds its locks while it waits, and `Scheduler.release(task_id, requeued=...)` gives them back |
| The steward runs inside its task's slot, and a steward that shows no progress within its idle window ends in an escalation to a human | `harness.py::Harness._run_slot` (`steward_factory`), `workflow.py` (`_wait_for_resolution`) | exists; this is why the steward is essential (section 6.1) |
| Three judge failures in a row halt reconciliation | `fused-memory/src/fused_memory/reconciliation/judge.py` (`JudgeInfraError`), `judge_infra_max_consecutive_failures`, `halt_on_judge_serious` | exists; this is why reconciliation steps aside (D13) |
| The accounts loader ignores keys it does not know | `shared/src/shared/config_models.py::UsageCapConfig._load_accounts_file` reads only `accounts` | confirmed: code without this PRD ignores a `reserve:` block |
| The roster and `usage_cap.*` are read once, at process start | `UsageGate._init_accounts`; `OPERATIONS.md` section "Config reload vs restart" | confirmed: the reserve takes effect per process at its next restart |
| Account events with a JSON details field | `shared/src/shared/cost_store.py`, table `account_events`, written through `UsageGate._fire_cost_event` | exists. The orchestrator attaches the store after the gate is built, by writing the gate's private attribute; three gates (reconciliation, evals, the nightly census) have no store and record nothing |
| A call's cost is known to the gate when its slot settles | `usage_gate.py::InvokeSlot.confirm` | exists |
| A list of the fleet's project roots | `DASHBOARD_KNOWN_PROJECT_ROOTS`, read by `dashboard/src/dashboard/config.py::DashboardConfig.all_project_roots` | exists |
| The interactive proxy logs each request with its account | `~/.cli-proxy-api/auths/logs/main.log` | exists; no tokens or cost in it |

Not available, and not assumed: an account's remaining allowance; its reset time before it caps; which kind of limit
a cap banner's reset belongs to; who set a pin; spend recorded for reconciliation, the nightly jobs and interactive
overflow.

## 5. Cross-PRD relationships and seam owners (G4)

| Other work | Direction | The seam | Owner |
|---|---|---|---|
| Usage-gate refactor chain, `plans/review-all-s20-cli-invocation-and-usage-gate-prd.md` (task 9846 rewrites the retry loop and changes `invoke_slot`'s signature; task 9804 splits the gate into modules) | both edit `usage_gate.py` and `cli_invoke.py` | the `work_class` keyword; the hook in the selection walk; the reserve wait; recording a served call when the slot settles | **This PRD lands first (R5).** At decompose time tasks 9804 and 9846 are made to depend on this PRD's task 1, with a note naming what they must carry through |
| Fable scoped caps, `plans/usage-gate-model-scoped-caps-prd.md` | this PRD copies its shape | "an overlay on the account state, not a clone of the phase machine"; a scoped waiter "must never clear `_open`" | that PRD's rules bind here; the reserve is a second overlay beside the scope overlay, and is tested before it (section 6.3) |
| Priority at the later gates, `plans/priority-at-later-gates-prd.md` (tasks 9933 to 9941) | this PRD reads effective tier | `TaskPriority.tier`; the `pinned` tier its task 9938 adds | that PRD owns the vocabulary. This PRD names tiers only in config and validates them in `orchestrator/` against the vocabulary of the day. Admitting `pinned` is then a config edit (D4) |
| One account pool from one file, task 6628 | both edit `UsageCapConfig._load_accounts_file` | the loader | 6628 owns the path and the missing-file refusal; this PRD owns the `reserve:` block. No ordering edge |
| One invocation row per attempt, task 9812 | none | the `invocations` table | untouched here: served calls are recorded as account events with their cost |
| Daily cost ceiling, task 8262 | none | — | the reserve reads no spend, so it is not a second spend rule |
| Halt retirement, `plans/scheduler-pause-halt-retirement-prd.md`, task 5313 | none | "a halt is permissible only for a pause class whose predicate remains durably re-testable" | the dispatch hold is recomputed every tick from the clock and the gate, and stores nothing |
| Per-consumer CPU and memory recording (PRD being written in a separate session, 2026-11-16) | none | none | — |
| Max-b and max-h stay listed, ruling 2026-10-06 (task 5944) | config | max-h's entry | max-h stays listed and stays last in the file. Selection tries it first only in its release window (R3). The file's header comment, which still says both are switched off, is corrected in task 6 |
| Interactive proxy (`cliproxyapi.service`) | none (R4) | max-h is last in its order | no change. The report counts its requests to max-h |

## 6. Contract (B + H)

### 6.1 What a caller declares

In a new module `shared/src/shared/account_reserve.py`:

```python
class WorkKind(StrEnum):
    TASK = 'task'              # acts for one task; admitted by tier
    ESSENTIAL = 'essential'    # always admitted
    BACKGROUND = 'background'  # never admitted before the release

@dataclass(frozen=True)
class WorkClass:
    kind: WorkKind
    project_id: str | None = None
    tier: str | None = None        # effective tier; meaningful for TASK only
    task_id: str | None = None     # for the record only; never read by the rule
    role: str | None = None        # for the record only: 'implementer', 'steward', 'curator', ...
```

`invoke_with_cap_retry(..., work_class=None)` forwards the keyword to `UsageGate.invoke_slot`, then
`before_invoke`, **only when it was given**, so a caller or test double that passes none is called exactly as
today. `work_class=None` is treated as background and recorded as unclassified.

The keyword takes a `WorkClass` or a function returning one. The gate calls the function at every selection, so
the class is current, not the one the call started with. The task roles pass a function that reads the scheduler's
effective tier. In `orchestrator/`, one helper builds a task's `WorkClass`; the task roles, the scheduler's
eligibility test and the main loop's second check all use it, so they cannot disagree.

Assignment, by call site:

| Caller | Kind |
|---|---|
| Every task role (`workflow.py::TaskWorkflow._invoke`) | task, with the effective tier |
| The dry-run unblock (`dry_run_unblock.py`) | task, with that task's tier |
| The task steward, both call sites (`steward.py`): it runs inside its task's slot, and a steward that cannot make a call for its idle window ends in an escalation to a human | essential |
| The automatic escalation watcher (`harness.py`) | essential |
| The whole-project review checkpoint (`review_checkpoint.py`): the main loop awaits it and dispatches nothing meanwhile, so refusing it would stop the project | essential |
| The task curator, the pack judge and the path-scope adjudicator (`fused-memory`): task filing for every project stops without them | essential |
| The module tagger (`harness.py`, switched off today) | background |
| Reconciliation stages, agent loop and judge (`fused-memory`) | background |
| Evals (`orchestrator/evals/measurement.py`, `architect_eval.py`, `reviewer_trial/runner.py`) | background |
| The nightly census and the return-brief preparer (`scripts/legibility/session_runner.py`, `scripts/sitting/nightly_prepare.py`) | background |

Not in the table because they use no gate: the link adjudicator (passes no gate) and the callers that invoke an agent
directly.

### 6.2 The rule

Config, in the shared accounts file, beside `accounts:`:

```yaml
reserve:
  account: max-h
  weekly_reset_utc: "mon 21:00"
  release_hours: 36
  admit_tiers:
    default: [critical, high]
    # per project_id, when a ranking exists:
    # dark_factory: [critical, high, medium]
```

Invariants:

1. **No block, no reserve.** Without the block, selection is byte-identical to today for every caller.
2. **Strict parsing.** The block is a model that forbids unknown keys. `account` must name a listed account,
   `weekly_reset_utc` must parse, `release_hours` must be more than 0 and less than 168. A failure makes the reserve
   *invalid*: inert, recorded (section 6.6), and reported by the gate as `reserve_state`.
3. **Tier names are checked where the vocabulary lives.** At start-up the orchestrator checks every name in
   `admit_tiers` against `PRIORITY_TIERS`. An unknown name marks the reserve invalid for that process and raises
   one escalation, keyed on the block's content so that repeated restarts add to the same one.
4. **The calendar is pure.** `next_reset(now)`, `release_at(now)` and `in_release_window(now)` are functions of the
   config and a UTC clock. Nothing is stored, and no banner changes them (D11).
5. **Drift is recorded, not acted on.** When a cap banner for the reserve account gives a reset time within three
   hours of a configured reset but more than 30 minutes from it, the gate records `reserve_schedule_drift`, at most
   once per week.
6. **Admission.** `admits(config, work_class, now)` is true in the release window; otherwise true for essential
   work, true for task work whose tier is in the project's list (or the default list), and false for the rest.
7. **An unusable reserve protects nothing.** While the reserve account is failing authentication the rule is off:
   no caller is refused on its account.

### 6.3 Selection and waiting

`UsageGate.try_lease(*, scope=None, exclude=None, work_class=None)`:

- **Protected window.** The reserve account is skipped for a caller the rule does not admit. The test comes before
  the probe-slot claim, like `exclude`, so a refused caller never takes a probing reserve's single probe slot. An
  admitted caller sees today's order, so it reaches the reserve last.
- **Release window.** The reserve account is tried first, for every caller. The existing skips (capped, probe in
  flight, failed authentication, scope-capped for a Fable model) still apply to it.
- Everything else in the walk is unchanged.

`UsageGate.before_invoke(..., work_class=None)`. After a failed selection and the existing reset sweep, and
**before the per-scope branch**, the gate asks whether this caller is *reserve-refused*: at least one account is
neither capped, probe-in-flight nor failed, and every such account is the reserve and refused to this class. If so:

- It waits on a reserve waiter event of its own. It never clears `_open` and never sets `_paused_reason`: the pool
  is not frozen, other callers are being served.
- `UsageGate._transition` sets that event whenever an account other than the reserve becomes available or probing,
  and when the reserve account fails authentication.
- Selection runs again at the first of: that event; the release time; five minutes; the caller's own deadline
  (`max_wait_secs`). A caller that is refused again goes on waiting; that is one wait, not many, in the record. A
  caller that is no longer reserve-refused but still finds nothing open (the reserve has failed, the rest are
  capped) leaves the reserve wait for the all-capped path, and its wait is recorded as ended.
- While it waits it logs the heartbeat the all-capped wait logs, naming the reserve account and the release time.
- At its deadline it raises `PoolFrozen`, which gains two fields, `reserved_account` and `release_at`. `park=False`
  raises at once. Messages built from a `PoolFrozen` that carries them say "refused the reserve until …", not "all
  accounts capped".

A caller with a Fable scope that is reserve-refused takes this wait, not the scope wait. When the reserve is capped
as well, the existing all-capped path applies unchanged; when the reserve later resets, the callers parked there
wake as today, select again, and the refused ones move to the reserve wait.

A call that loses its account to a cap mid-session comes back through the same `before_invoke`. A reserve wait is
not a cap hit: it does not count toward `max_cap_retries` and adds no cooldown.

The gate also exposes `reserve_waiters`, the number of callers in a reserve wait right now.

### 6.4 The question the scheduler asks

`UsageGate.reserve_refusal(work_class) -> ReserveRefusal | None`. It changes nothing and claims nothing. It returns
a refusal (the account and the release time) exactly when the caller would be reserve-refused as section 6.3
defines it, for an unscoped call. When no account is open at all it returns `None`.

Three consumers:

- **The scheduler's eligibility test.** The harness hands the scheduler a narrow callable.
  `Scheduler._eligible_for_dispatch`, the one test behind scored selection, the starvation watchdog and the pin
  queue, answers "not eligible" for a task the gate would refuse. Each tick in which at least one task was held
  this way emits one `dispatch_reserve_hold` event with the number held, their tiers and the release time. This is
  a new event type; the pressure hold's `dispatch_deferred` is left as it is. Nothing is stored; the next tick asks
  again. The test stays a pure predicate: it reads the tier the tick has already computed. It is reached more
  than once per tick for some tasks, so the event's count is of distinct tasks, gathered by the callers, and no
  line is logged per task. A tick whose candidates were all held is not an empty queue: the main loop's
  drained-queue branch (the idle summary and the optional full review) does not run on that account.
- **The main loop, with a slot in hand.** The loop takes an assignment and may then wait a long time for a slot.
  When it has the slot it asks again. A task refused at that moment is put back: its locks and the slot are
  released without the requeue cooldown, and the loop goes round. Nothing outside the scheduler has been taken
  at that point (the status change, the claimant and the worktree come later). The put-back does not undo what
  choosing the task already settled inside the scheduler: its fairness skip count and parks, its starvation
  anchor, and the dispatch stamp. That is accepted for the task or two per dry spell it affects.
- **Reconciliation** (D13). It asks for background work at both places a cycle drains events (the project loop
  and the backlog chunk runner). Refused, it starts no cycle, writes no run row, and logs the deferral at most
  once every ten minutes, because its loop comes round every few seconds while events are buffered. A judge that
  falls due while refused keeps its pending marker and is fired by the existing pending-judge recovery pass, which
  reconciliation now also runs when a refusal ends, not only at start-up. A reconciliation process finds out about
  caps only through its own calls, so the first cycle of a dry spell is not refused and ends as it does today, as
  a deferred run after its 30-minute bound.

A process that has just restarted believes every account open until it discovers otherwise, so for its first
minutes the check passes everything and it may fill its slots. Higher tiers are mostly chosen first, though
fairness and age also weigh. The lower-tier tasks it started wait in their slots once the caps are found. That is
the state section 6.5 measures.

### 6.5 What waiting costs, and how it is seen

Waiting tasks keep their slots (D8), so admitted work can find every slot taken. Two things make that visible in
the hour it happens, not in a report two weeks later:

- **`reserve_slot_wait`.** When the main loop has waited more than a minute for a slot while at least one of the
  project's callers is in a reserve wait (`UsageGate.reserve_waiters`), it records the event when the wait ends:
  how long, how many were waiting on the reserve, the project's slot count, the task and its tier, and whether it
  was started or put back.
- **One escalation.** If that wait passes two hours, the orchestrator raises one escalation for the project, keyed
  on the reserve's week so that it is raised once per protected window. It says how many slots are held by tasks
  waiting on the reserve and when the reserve releases.

### 6.6 What is recorded

By the gate, in the caller's own `account_events` table, through its single event writer, details as JSON:

| Event | When | Details |
|---|---|---|
| `reserve_config` | once per process, if a block is present: when the gate is given its store | `state` (`active` or `invalid`), `reason`, `account`, `next_reset`, `release_at`, `admit_tiers` |
| `reserve_wait_started` | a caller begins a reserve wait | `account`, `kind`, `role`, `project_id`, `tier`, `task_id`, `release_at`, `unclassified` |
| `reserve_wait_ended` | that wait ends | the same, with the tier as last read, plus `waited_secs` and `ended_by` (`account_opened`, `released`, `admitted`, `reserve_unusable`, `deadline`) |
| `reserve_served` | a call on the reserve account settles **in the protected window** | `kind`, `role`, `project_id`, `tier`, `task_id`, `cost_usd` |
| `reserve_schedule_drift` | invariant 5 of section 6.2 | `configured_reset`, `observed_reset` |
| `reserve_account_unavailable` | the reserve account fails authentication | `account` |

The gate gains a method by which a store is attached, and writes `reserve_config` there. The orchestrator calls it
in place of writing the gate's private attribute.

By the orchestrator, in its event store: `dispatch_reserve_hold` (section 6.4) and `reserve_slot_wait`
(section 6.5).

Calls in the release window are not recorded one by one: there the reserve serves everything, and the `invocations`
table already has the account and cost of each call.

A gate with no store records nothing. The dashboard's list of recent account events
(`dashboard/src/dashboard/data/costs.py`) reads the newest 200 rows of any type; it is changed to leave out
`reserve_served`, so cap and failover rows are not pushed off it.

### 6.7 The weekly report

`scripts/allowance_reserve_report.py [--weeks N] [--state --since TIME] [--project-root PATH ...] [--store PATH ...]`,
read-only. Roots default to `DASHBOARD_KNOWN_PROJECT_ROOTS`; `--store` adds a store that belongs to no project, the
memory service's curator store. For each of the reserve's weeks (reset to reset) it prints:

- when the reserve reached its weekly limit, if it did, and how many hours before the reset. Reached means nothing
  was left. Not reached means something was; the report then prints recorded spend on the reserve, with Fable 5.1
  spend counted twice, against the $2,190 figure, labelled as an estimate that omits unrecorded consumers;
- the reset times seen in the reserve account's cap banners that week, and any drift event;
- spend and calls on the reserve in the protected window by project, tier, kind and role (from `reserve_served`),
  and in the release window by project (from `invocations`);
- waits on the reserve: count and total hours by project, tier and kind; the peak number waiting at once per
  project against its slot count, and how many times and for how long every slot was held by a waiting task;
- ticks in which the scheduler held tasks back, and the slot waits of section 6.5, per project;
- unclassified waits, by process, so a call site that passes no class is seen;
- interactive requests the proxy sent to the reserve account in the week, from the proxy's log.

Three states are kept apart: a store with no `reserve_config` row since the reference time prints "reserve not
active in this process"; an active reserve with no waits prints zero; a consumer that keeps no store
(reconciliation, the nightly census, evals) is listed as "not recorded". Every store it could not read is named.

`--state` prints only the state each store's process last reported, and exits non-zero if any store has no `active`
row newer than `--since`.

## 7. Boundary-test sketch (B + H)

Gate side (`shared/`):

| # | Scenario | Given | Then |
|---|---|---|---|
| B1 | No block | accounts file without `reserve:` | every existing selection test passes unchanged; a caller passing no class calls `invoke_slot` exactly as today |
| B2 | Protected, refused | six accounts capped, reserve open, protected window, low-tier task class | `try_lease` returns none; `before_invoke` waits without clearing `_open`; `is_paused` stays false; `reserve_wait_started` recorded; `reserve_waiters` is 1 |
| B3 | Protected, admitted | as B2: a high-tier class; an essential class | each leases the reserve; on settle, `reserve_served` with the cost |
| B4 | Admitted caller keeps today's order | max-c and the reserve open, high-tier class, protected window | max-c is leased |
| B5 | Release, tried first | all accounts open, release window, low-tier class | the reserve is leased; no `reserve_served` row |
| B6 | Wake at the release time | as B2, clock advanced past `release_at` with no account event | the waiting caller is served by the reserve; `reserve_wait_ended` with `ended_by: released` |
| B7 | Wake on another account | as B2, max-d reopens | the waiting caller is served by max-d; `ended_by: account_opened` |
| B8 | The reserve's own transitions do not end a wait | as B2; the reserve's probe is confirmed | the caller is still in the same wait; no `reserve_wait_ended` |
| B9 | Deadline | as B2 with `max_wait_secs`; and with `park=False` | `PoolFrozen` carrying `reserved_account` and `release_at`; at once for `park=False`; no cap hit is counted |
| B10 | Everything capped, then the reserve resets | all seven capped; a low and a high caller parked; the reserve becomes available | both wake; the high one leases the reserve; the low one moves to the reserve wait; `_open` is not cleared by it |
| B11 | Invalid block | unknown account name; unknown key; `release_hours: 0` | reserve inert; `reserve_config` with `state: invalid` and the reason when a store is attached; selection as B1 |
| B12 | Reserve account failing authentication | protected window, reserve account auth-failed | no caller is refused on its account; `reserve_account_unavailable` recorded |
| B13 | A banner does not move the calendar | a cap on the reserve whose reset is one hour after the configured one; another whose reset is five hours from now | the release window is unchanged in both; one `reserve_schedule_drift` for the first, none for the second |
| B14 | Refused caller with a Fable scope | B2's state, low-tier class, scoped model | it takes the reserve wait, wakes at the release time, and its deadline raises `PoolFrozen` naming the reserve, not the scope |
| B15 | Admitted caller, reserve scope-capped | reserve open but scope-capped for the Fable model; high-tier class with that scope | the reserve is skipped for that scope; the scope wait behaves as today |
| B16 | No class passed | as B2, `work_class=None` | refused; the wait is recorded with `unclassified: true` |
| B17 | A probing reserve | reserve in its probing phase, protected window, low-tier class then high-tier class | the low caller is refused without claiming the probe slot; the high caller claims it |
| B18 | The query | B2's state; then all seven capped | `reserve_refusal` returns a refusal for the low class and none for the high class; none for either when all are capped; asking claims nothing |
| B19 | A waiting caller is promoted | as B2 with the class given as a function; the function starts answering high-tier | within five minutes of clock time, with no account event, the caller leases the reserve; `ended_by: admitted` |
| B20 | The reserve fails while callers wait | as B2; the reserve account fails authentication | the caller leaves the reserve wait for the all-capped path; `ended_by: reserve_unusable`; `reserve_waiters` is 0 |

Caller side (`orchestrator/`, `fused-memory/`, `scripts/`):

| # | Scenario | Given | Then |
|---|---|---|---|
| C1 | A task role passes its effective tier | a low task that a critical task waits on | class is task, tier critical |
| C2 | Unknown tier | the scheduler has not ticked yet | the task's own tier is used; with none, `medium` |
| C3 | The scheduler holds a refused task on every path | a real gate in B2's state; a low task in the scored list, a low task the starvation watchdog would promote, a pinned low task, a high task | only the high task is dispatched; the watchdog promotes none of the three while they are held; one `dispatch_reserve_hold` counting three distinct tasks, with the release time |
| C4 | The hold lifts by itself | C3, then the clock passes `release_at` | the next tick dispatches the held tasks; nothing was stored in between |
| C4b | All candidates held is not an empty queue | every candidate held, no task running | the main loop does not write the idle summary or start the full review on that account |
| C5 | A stale assignment is put back | one slot, busy; the loop holds a low assignment taken while all accounts were open; the gate reaches B2's state; the slot frees | the low task is not started and its status is unchanged; its locks and the slot are released; the next tick starts a waiting high task and does not choose the low one |
| C6 | A slot wait is seen | one slot held by a low task in a reserve wait; a high task ready | when the wait ends a `reserve_slot_wait` is recorded; past two hours, one escalation, and a second protected-window pass in the same week raises none |
| C7 | A bad tier name in the table | `admit_tiers` names a tier the orchestrator does not know; the process is started twice | it starts both times; the reserve is invalid for it; one open escalation, not two |
| C8 | Essential work is not held | the gate in B2's state; a curator batch, a watcher rotation, a review checkpoint, the steward of a low task | all are served by the reserve; with the review running, dispatch continues afterwards; the steward's task is not escalated to a human for idleness |
| C9 | Reconciliation steps aside | a real gate in B2's state; events buffered for the project loop and for a backlog chunk; one judge due | neither drain site starts a cycle; no run row is written; the deferral is logged once, not every loop; the judge keeps its pending marker and the consecutive judge-failure count does not move; after `release_at` the next cycle runs and the pending judge is fired |
| C10 | One-shot jobs | the gate in B2's state; the nightly census run and the return-brief preparer | each ends as it does today on a frozen pool, and its message names the reserve and the release time |

## 8. Decomposition plan

Eight tasks.

1. **Gate: the reserve rule, selection, wait, query and events** (`shared/`).
   The new `account_reserve.py` (config model, calendar, `WorkClass`, `admits`); the `reserve:` block read by
   `UsageCapConfig._load_accounts_file`; the hook in `try_lease`; the reserve wait in `before_invoke` and its event
   set in `_transition`; the two `PoolFrozen` fields and the messages built from them; `reserve_refusal`,
   `reserve_waiters` and `reserve_state`; the method that attaches a store; the six events, `reserve_served`
   written when the slot settles; the keyword forwarded through `invoke_slot` and `invoke_with_cap_retry` only when
   given; `shared/testing.py::make_gate_mock` accepting it.
   Signal: boundary rows B1 to B20 pass in CI. Unlocks tasks 2, 4 and 5. With no block in the live file, no process
   changes behaviour.
2. **Orchestrator: a class at every call site** (`orchestrator/`). Depends on 1.
   The classes of section 6.1 for the task roles, the dry-run unblock, the steward, the automatic watcher, the
   review checkpoint, the module tagger and the evals; the harness attaches the gate's store through the new
   method; the hand-written gate doubles in `orchestrator/tests/test_dry_run_unblock.py`, `test_steward.py` and
   `test_invoke.py` accept the keyword.
   Signal: rows C1 and C2 pass in CI, and the review, watcher and steward parts of C8; an orchestrator test shows a
   `reserve_config` row written at start-up from a fixture accounts file.
3. **Orchestrator: the dispatch hold and the slot-wait signal** (`orchestrator/`). Depends on 2 (both edit
   `harness.py`).
   The callable handed to the scheduler and the refusal in `Scheduler._eligible_for_dispatch`; the
   `dispatch_reserve_hold` event; the all-held tick kept out of the drained-queue branch; the second check in the
   main loop and the put-back; the one helper that builds a task's `WorkClass`; `reserve_slot_wait` and its
   escalation; the start-up check of tier names and its keyed escalation.
   Signal: rows C3 to C7, C4b included, pass in CI against a real `UsageGate` built from a fixture accounts file.
4. **Memory service and nightly jobs: classes, reconciliation's deferral, messages** (`fused-memory/`,
   `scripts/`). Depends on 1.
   The curator, pack judge and path-scope adjudicator as essential; reconciliation, the census and the return-brief
   preparer as background; reconciliation's check at both drain sites, its throttled deferral log and the
   pending-judge pass run when a refusal ends; the census's and the preparer's frozen-pool messages name the
   reserve.
   Signal: the curator part of C8, and rows C9 and C10, pass in CI.
5. **The weekly report, and the dashboard's event list** (`scripts/`, `dashboard/`). Depends on 1 for the gate's
   event names; the two orchestrator event names are fixed by section 6.6.
   Signal: run against the live stores before the reserve is switched on, it prints for each of the last four weeks
   when max-h reached its limit and its recorded spend by project, and "reserve not active" for every process;
   against a fixture store carrying every event of section 6.6 it prints every section of 6.7; `--state` exits
   non-zero on a fixture with one store missing its row.
6. **Switch it on** (`config/usage-accounts.yaml`, `OPERATIONS.md`). Depends on 2, 3 and 4.
   Adds the block of section 6.2 with Leo's values; corrects the file's header comment, which still says max-b and
   max-h are switched off; adds a short operator section on the reserve (what it does, how to turn it off by
   removing the block, where the report is, how to admit `pinned` once that tier exists and why that also admits
   the watchdog's pins). A process reads the file
   only at start, so any process that sees the block is running code that has tasks 2, 3 and 4; the fleet is mixed
   until the next redeploy.
   Signal: loading the live file through `UsageCapConfig` yields an active reserve whose next release is a Sunday at
   09:00 UTC, shown by a one-line command in the operator section.
7. **Check the fleet picked it up** (a dated check, no code). Depends on 5 and 6; runs a day after task 6 lands.
   Runs the report with `--state`, `--since` the time of task 6's commit, the nine project roots and the memory
   service's curator store. It escalates to Leo, naming the processes, if any has not reported an active reserve.
   Signal: the check's output lists all nine orchestrators and the memory service's curator as active, or an
   escalation names the ones that are not.
8. **Two-week reading** (a dated gate for Leo). Depends on 7; fires after two full weeks of the reserve running.
   Runs the report and sends Leo the result with a recommendation on the release hours, the admitted tiers, and
   whether lending slots (section 9) is worth a PRD of its own. For that last point it reports how often, and for
   how long, every slot in a project was held by a task waiting on the reserve, and how many of those times
   followed a restart.
   Signal: Leo receives the escalation with the two weeks' tables.

At decompose time, also: make tasks 9804 and 9846 (the refactor chain) depend on task 1, with a note on each naming
the keyword, the selection hook, the reserve wait, the settle-time event, the store-attaching method and
`account_reserve.py`.

## 9. Out of scope

- A ranking of projects. The table takes one; none is set.
- **Lending a waiting task's slot to other work.** It would let admitted work start whenever lower-tier tasks are
  waiting. It was examined and left out of this PRD because it is not small: the permit must be given back and
  retaken without the count drifting when a task is cancelled, hard-cancelled or force-released mid-wait, and
  several readers assume that started tasks never exceed `max_concurrent_tasks` (the dashboard's burn-down alarm,
  the pressure hold's floor on running tasks, the sizing of warm worktree pools). Task 8 reads the measured slot
  waits and says whether it is worth doing.
- Holding dispatch when every account is capped. Today tasks start and wait at the gate; the study found the
  benefit of changing that unmeasured, and it meets the file-lock work.
- More than one reserve account, or protection levels that step down through the week.
- Any change to the interactive proxy's order (R4).
- Reading or estimating how much an account has left, remembering caps across a restart, or sharing them between
  processes (ruling 2026-11-12).
- Recording spend for reconciliation, the nightly jobs and interactive overflow, or giving their gates a store.
- Releasing a started task's file locks (ruling 2026-11-12).
- Telling an operator's pin from the starvation watchdog's (D4).
- CPU weights between projects, and the guard on resuming the paused projects: both remain with the study's open
  decisions.

## 10. Open questions (tactical)

1. **Which escalation helper reports an invalid reserve and a long slot wait, and how each is keyed.** The
   orchestrator already escalates an unknown config key on reload. **Suggested:** reuse that path, with a key from
   the block's content for the first and from the project and the reserve's week for the second. Decide in task 3.
2. **The heartbeat for the reserve wait.** The per-scope wait has none (task 7243). **Suggested:** share one helper
   with the all-capped wait. Decide in task 1.
3. **The put-back's exact call.** `Scheduler.release(task_id, requeued=False)` and the semaphore's release, in the
   order `SlotLease.release` uses. **Suggested:** build the lease first and release through it. Decide in task 3.
4. **Where reconciliation's checks sit.** **Suggested:** one helper on the reconciliation harness, called in the
   project loop after the buffer says a cycle is due and before the run is marked active, in the backlog chunk
   runner before it drains, and where a judge is spawned, after its pending marker is written. Decide in task 4.
5. **How the two dated tasks are dated.** Task 7 a day after task 6 lands; task 8 two full reserve weeks after
   task 7 passes. **Suggested:** delayed milestones. Decide at decompose time against `docs/task-authoring.md`.
6. **Readers that describe the pool without a class** (`UsageGate.active_account_name`, `soonest_resets_at` in the
   cap-wait log line). **Suggested:** leave their meaning alone and make the log line name the reserve when the
   wait is a reserve wait. Decide in task 1.
7. **Fairness parks and pin reservations of a held task.** A held task keeps its fairness parks for the length of
   the hold, and a held pinned task loses its pin reservation. **Suggested:** leave both as they are and say so in
   the operator note. Decide in task 3.
8. **The wait for a slot longer than a minute, measured how.** The loop is suspended in the semaphore's acquire.
   **Suggested:** time the acquire and read `reserve_waiters` when it returns and at a coarse interval while it
   waits. Decide in task 3.
