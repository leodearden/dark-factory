# Claimant invariant detection (ε): standing gauge, owned consumption, mint-time ledger

**Status**: active · authored 2026-08-24 · approach **B+H** (contract + two-way boundary tests)
**Code anchors** verified against main `f73a769923` (2026-08-24). Main moves fast — cite-by-symbol;
re-locate lines at implementation time.
**Origin**: `docs/prds/claimant-invariant-enforcement.md` D8 deferred the detector after it failed
two design passes; this PRD is that design pass. The design position was established in a
five-agent investigation (observation sites, escalation closers, invocation/monitoring, post-β
mint-rate measurement, adversarial refuter) on 2026-08-23 and ratified by Leo on 2026-08-24,
including the loudness downgrade (decision D-4 below) and the watcher-rotation consumption owner.

## Goal

C4-E7 (enforcement PRD) requires that an observed violation of C4-E1
(`status ∈ TERMINAL ⇒ claimant_run_id IS NULL`) is "surfaced loudly and structurally, never
raised, never silently absorbed". This PRD delivers that surfacing as a **standing, denominated
gauge with an owned consumer and durable mint-time breadcrumbs** — not as an unattended alarm —
and amends C4-E7 to record the downgrade **with a named re-arm trigger**. After this PRD:

- The dashboard serves `/api/v2/dashboard/invariants`: a per-root claimant-invariant census
  (terminal tier, hygiene tier, rows scanned, measured-at) computed over direct read-only
  sqlite access to every configured project's task DB, where an unreadable DB renders as a
  distinct **blind** state, never as zero.
- Both watcher rotations (escalation-watcher, recon-escalation-watcher) read that endpoint once
  per cycle — the named consumption owner.
- The one sanctioned mint path (C4-E2 explicit supply) becomes **self-documenting**: the
  interceptor stamps a durable `metadata.claimant_exception` ledger entry when it honours an
  explicit claimant into a terminal write. The alarm-tier definition sharpens to
  "terminal + claimant + **no ledger entry**", resolving the C4-E1/C4-E2 contradiction.
- Manual mints via `set_task_claimant` are observed at write time (structured ERROR, zero extra
  round-trips) — a breadcrumb, not a refusal.
- C4-E7 and OPERATIONS.md record how loudness is delivered, and the trigger that mandates
  building the escalation-grade alarm (ε.2, pre-designed below, deliberately not built).

## Background — why a gauge, not an alarm

Measured 2026-08-23 (all first-hand; the numbers move — re-measure, don't cite):

- **Post-β, no running code can mint a terminal-tier violation.** All 29 then-live terminal-tier
  rows were the posthumous-mark-done shape β intercepts (heartbeat died 20 min–10 h before the
  terminal write, every row). Residual minters: deliberate manual MCP action, future code drift,
  a DB parked at schema v3 (none exists), hand SQL. Realistic event rate ≈ 0/day.
- **The host cannot currently notice a dead recurring job.** The orchestrator-watchdog timer is
  disabled/inactive; 2 of the 5 nightly jobs OPERATIONS.md documents are not installed; nothing
  flagged either. `BackgroundService` passes fail log-N-then-silent. There is no metrics
  infrastructure. Every unattended-detector variant's silent-death probability exceeds the
  violation rate it would guard.
- **The write journal is structurally blind to claimant writes** (`TaskInterceptor.
  set_task_claimant` deliberately journals nothing; status-write payloads normalize the claimant
  kwarg away) — detection must read rows, never history.
- **The heartbeat race** (`_claimant_heartbeat_loop`, no status gate, stop only in
  `_on_terminal_cleanups`) re-stamps `heartbeat_at` for up to ~60 s after every ordinary
  completion. Any detector asserting on `heartbeat_at` alarms on routine traffic. This PRD's
  gauge and tripwires consume α's `claimant_run_id`-only predicate exclusively. (Task 4667
  will shrink the window; the claimant-only assertion stays correct regardless.)

So the detector's dominant risks are its own silent death, false alarms, and unfindable records —
not missed latency. The design attaches the invariant to surfaces that are already alive and
makes blindness render as a visible failure rather than as health.

## Resolved design decisions

### D-1 — The gauge lives in the dashboard, on direct per-root DB reads

The dashboard is the only recurring surface on the host with armed supervision
(`dark-factory-dashboard-watchdog.timer`, 30 s `/healthz` + restart, verified active). The gauge
computes in its serving path: if the dashboard is up the gauge is fresh; if down, the watchdog
restarts it. **Mechanism** (verified): add `.taskmaster/tasks/tasks.db` as a per-root `DbPool`
path — the precedented per-root direct-sqlite idiom already used for `data/orchestrator/runs.db`
and `data/burndown/burndown.db` (opener idiom in `dashboard/app.py`; pool is read-only
`?mode=ro`). The pool's growth-bound docstring in `dashboard/data/db.py::DbPool` is updated in
the same change (bound becomes ×3 per root). The MCP fetch path is NOT used: its terminal view
is a 400-row-per-bucket window that misses old residue; the gauge needs the unwindowed
`COUNT`/enumeration (milliseconds over ≤ ~6.5k rows per DB).

**Roster**: the gauge is governed by the dashboard's configured roots
(`DASHBOARD_KNOWN_PROJECT_ROOTS` in the service unit) — currently 9, which differ from the
"fleet" framing elsewhere (`mission-control` in, `my-solar-challenge` out). The endpoint reports
per-root, so roster membership is visible, not assumed. The doubled-`tasks`-dir path is a named
constant; a bare `.taskmaster/tasks.db` is a known 0-byte decoy.

### D-2 — Blind ≠ clean, by construction

Every gauge row carries its denominators: `rows_scanned`, `measured_at`, and a per-root
`status ∈ {ok, blind}`. A root whose DB cannot be opened or scans zero rows where the DB file is
non-empty renders `blind` with the error, never `0 violations`. The clean-vs-blind question the
enforcement PRD's D8 could not answer for an auto-resolving alarm cannot arise on a surface that
always shows what it measured. (House precedent: the confusion-reduction liveness/progress
two-probe split; the `_task_db_scan` skeleton's exit-3 "nothing scanned ≠ clean".)

### D-3 — Consumption is owned by the watcher rotations

Leo's ruling 2026-08-24: the watcher session is the primary monitoring and control interface in
practice; occasional watcher gaps are short. Both watcher skills gain a once-per-cycle step
(inserted in each skill's "## The Main Loop", following the existing once-per-cycle
`reap-decisions` step pattern): GET `/api/v2/dashboard/invariants`; a nonzero **unledgered**
terminal tier or any `blind` root is triaged like a queue finding. Both skills live only in-repo
(no `~/.claude` copies — verified), so the edit is single-site. Neither layer is load-bearing
alone: the gauge stands when the watcher sleeps; the watcher checks when it wakes.

### D-4 — Loudness is downgraded deliberately, with a re-arm trigger (C4-E7 amendment)

The adversarial refuter's verdict, accepted: given the ~0/day event rate and the host's
demonstrated inability to keep unattended jobs alive, "no unattended alarm" is the sound
engineering position — but only as a **recorded contract decision**. The amendment to the
enforcement PRD states: C4-E7's "loudly and structurally" is delivered by (gauge + ledger +
mint-time ERROR + ζ census on demand + watcher checklist), and names the trigger that mandates
ε.2: **any post-ζ observation of a nonzero unledgered terminal tier** (via gauge, census, or
watcher rotation). The observation itself is the evidence the alarm is needed; filing ε.2's
work is the mandated response.

### D-5 — The C4-E2 ledger stamp resolves the sanctioned-mint contradiction

C4-E2 honours an explicitly-supplied claimant on a terminal write; that legal move mints exactly
the row C4-E1 alarms on — a contradiction the enforcement PRD never reconciled (refuter finding).
Resolution: the sanctioned path leaves its name. When `TaskInterceptor._apply_status_transition`
honours an explicit claimant into a terminal status, it adds a `claimant_exception` entry to the
same `audit_fields` dict the `reopen_*` fields use — written by
`SqliteTaskBackend.set_status_and_stamp_audit` atomically with the status column in one
transaction (verified: metadata merge is sibling-preserving, `mode='merge'`). Entry shape:
`{claimant_run_id, stamped_at, agent_id/tag, target_status}`. A structured WARNING logs the same
facts. The **alarm tier** becomes: terminal + claimant + no `claimant_exception` whose
`claimant_run_id` matches. α's exported predicate `violates_terminal_claimant_invariant` is
**unchanged** (it stays the raw C4-E1 predicate); the gauge and census layer the ledger
classification on top, reporting ledgered rows separately. The `claimant_exception` key is added
to `_BLESSED_METADATA_KEYS` (+ `docs/task-authoring.md` vocabulary entry) — it is a permanent
machine stamp and must not emit `unknown_key` census noise on every write.

### D-6 — Mint-time observation in `set_task_claimant`, zero extra round-trips

`SqliteTaskBackend.set_task_claimant` already executes a `SELECT` for the row inside the same
`_write_lock` + `_txn` before its UPDATE (verified). Extend that SELECT to include `status`; when
the write stamps a **non-NULL** `claimant_run_id` onto a row whose status is terminal, log a
structured ERROR (task id, status, claimant, caller tag). The write still succeeds — observation,
not refusal. This is deliberately NOT the "make `set_task_claimant` status-aware" rejected by the
enforcement PRD's Out of scope: that rejection was about *refusing* (and about a `get_task`
round-trip on the heartbeat hot path); this is an in-transaction read of a column already being
fetched. The heartbeat loop is unaffected: it passes `claimant_run_id` as the wire-unset sentinel,
which never satisfies "stamps a non-NULL claimant". Breadcrumb only: journald retention on this
host is ~3.5 days; the census/gauge remain authoritative. The ledger (D-5) is the durable record
for the sanctioned path; this ERROR is the tripwire for the unsanctioned one.

### D-7 — ε.2 is pre-designed and deliberately not built

If D-4's trigger fires, the alarm is filed **at observation time** by the observer (watcher
rotation or human) — not by a scheduled process — which needs no recurrence machinery. The record
shape is pre-chosen from the closer analysis: the deterministic-task escalation class
(`category='milestone_check_failed'`, `agent_role='orchestrator-deterministic'`, born-at-L2) has
a fully clean closer walk — deny-listed in `L2_AUTO_CLOSE_DENY_CATEGORIES`, escapes
`_revalidate_open_deterministic_escalation` at the `before_done.target_unit` gate, outside
`escalation_revalidation_allowlist` — and `_recover_stranded_deterministic_gate` (Source A)
re-files a lost gate escalation byte-identical, the only self-healing alarm-record path in the
system. If a hand-filed record is preferred: novel category + `orchestrator-deterministic` role +
non-numeric (dunder) sentinel + dedicated `root_cause` dedup
(`find_pending_l2_by_root_cause`), resolved only on positive evidence, never on N clean passes.
Should recurring deterministic tasks land (`docs/prds/recurring-deterministic-tasks.md`), ε.2's
census can become a recurring carrier; that upgrade is out of scope here.

## Contract (B+H)

**E-1 (gauge).** `/api/v2/dashboard/invariants` returns, per configured root:
`{root, status: ok|blind, rows_scanned, measured_at, terminal_tier: {count, unledgered_count,
violations: [{task_id, status, claimant_run_id, heartbeat_at, ledgered}]}, hygiene_tier:
{count, stale_count}}`. No field ever asserts a specific corpus count; `blind` carries the error
string. The census consumes `shared.task_claimant.violates_terminal_claimant_invariant` and
`is_stale_hygiene_tier_claimant` (α, task 4618) — never re-expresses them.

**E-2 (ledger).** An explicit-claimant terminal write persists `metadata.claimant_exception`
atomically with the status column, and only then. Non-terminal writes, unsupplied-claimant
writes, and the same-status no-op stamp nothing. The alarm tier is
`violates_terminal_claimant_invariant(task) AND NOT ledger_covers(task)`.

**E-3 (tripwire).** A `set_task_claimant` call persisting a non-NULL `claimant_run_id` onto a
terminal row emits one structured ERROR naming task id, status, claimant, tag — in the same
locked transaction's scope, with no additional query. Heartbeat-only ticks never trigger it.

**E-4 (consumption).** Both watcher skills' main loops name the endpoint read as a numbered
once-per-cycle step, with the triage rule (nonzero unledgered terminal tier or blind root ⇒
treat as a finding).

**E-5 (amendment).** The enforcement PRD's C4-E2 gains the ledger clause; C4-E7 gains the
delivery statement and the re-arm trigger; OPERATIONS.md §4 gains the operator subsection
(endpoint, ζ census invocation, what blind means). All by symbol/heading anchor.

## Boundary-test sketch (B+H)

| # | Side | Scenario | Preconditions | Postconditions |
|---|---|---|---|---|
| B1 | gauge | violation enumerated | a root's DB holds a terminal row with a claimant (seeded or live) | endpoint row lists it under `terminal_tier.violations`, `ledgered: false`, `rows_scanned > 0` |
| B2 | gauge | blind ≠ clean | a configured root whose tasks.db path is unreadable/absent | that root renders `status: blind` + error; other roots unaffected; no zero-count claim |
| B3 | gauge | hygiene never alarmable | rows with stale claimants in `{pending, deferred, review, merge-deferred}` | counted under `hygiene_tier` only; `terminal_tier` untouched |
| B4 | gauge | heartbeat race immune | row `(done, claimant NULL, fresh heartbeat)` | not a violation anywhere in the response |
| B5 | ledger | explicit supply stamps | `set_task_status(id,'done',claimant_run_id='run/s/pid=1')` on a live interceptor | `get_task` shows `metadata.claimant_exception` with that claimant; status+ledger committed atomically |
| B6 | ledger | unsupplied stamps nothing | plain `set_task_status(id,'done')` | no `claimant_exception`; β's clear behaviour unchanged |
| B7 | ledger | ledgered row not alarm-tier | row from B5 | gauge lists it `ledgered: true`, excluded from `unledgered_count` |
| B8 | tripwire | manual mint observed | `set_task_claimant(id, claimant_run_id='x', ...)` on a `done` row | structured ERROR emitted; write persists; gauge subsequently lists the row |
| B9 | tripwire | heartbeat tick silent | heartbeat-only `set_task_claimant(id, heartbeat_at=now)` on any row | no ERROR |
| B10 | consumption | checklist wired | both skills on main | each "## The Main Loop" names the endpoint step (grep) |

## Pre-conditions for activating

- **Task 4618 (enforcement α)** — exports the predicates and TTL the gauge consumes. Hard
  dependency of d1.
- **Task 4619 (enforcement β)** — creates the explicit-supply terminal branch d3's ledger rides,
  and edits the same interceptor region. Hard dependency of d3.
- ζ (4625) is **not** a dependency: this PRD alarms nothing, so D7's repair-before-detection
  ordering is satisfied by construction; the gauge showing the pre-ζ corpus, then ζ's effect, is
  intended behaviour.

## Cross-PRD relationship

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `docs/prds/claimant-invariant-enforcement.md` | consumes (α's predicates; β's explicit-supply branch) and amends (C4-E2 ledger clause, C4-E7 delivery + trigger) | `violates_terminal_claimant_invariant` · `_apply_status_transition` explicit-supply branch · C4-E7 text | enforcement PRD owns the predicates/choke point; **this PRD owns the amendment task (d5)** | wired (deps on 4618/4619) |
| `docs/prds/recurring-deterministic-tasks.md` | produces (the invariants endpoint + per-root tasks.db pool path that its chain panel extends) | `/api/v2/dashboard/invariants` · `DbPool` tasks.db per-root path | **this PRD** (d1) | wired (that PRD's r4 depends on d1) |
| `plans/task-status-authority-prd.md` | none directly (η/4626 owns its amendment) | — | — | no collision (verified: 4626 scoped to that file only) |

## Decomposition plan

| Label | Title | Modules | Kind | Observable signal | Prereqs |
|---|---|---|---|---|---|
| **d1** | Dashboard claimant-invariant gauge: per-root tasks.db pool path + `/api/v2/dashboard/invariants` + UI chip | `dashboard` | intermediate (unlocks d2, d5; unlocks recurrence r4 cross-PRD) | `curl` of the endpoint returns the E-1 shape for every configured root with `rows_scanned > 0` on populated DBs, enumerating the live terminal-tier violations present at run time (cross-checkable against ζ `--json` when 4625 lands); a deliberately-unreadable root renders `status: blind`, not zero | 4618 |
| **d2** | Watcher-rotation consumption step in both watcher skills | `skills` | **leaf** (non-code) | `git grep -n 'dashboard/invariants' -- skills/escalation-watcher/SKILL.md skills/recon-escalation-watcher/SKILL.md` returns a numbered main-loop step in **both** files, where it returns zero today | d1 |
| **d3** | Interceptor C4-E2 ledger stamp (`metadata.claimant_exception`) + structured WARNING; bless the key | `fused-memory`, `shared` | **leaf** | Product read path: `set_task_status(id,'done',claimant_run_id='…')` on a claimed row, then `get_task(id)` shows `metadata.claimant_exception` carrying that claimant; a plain terminal write on a sibling row stamps nothing (B5/B6) | 4619 |
| **d4** | `set_task_claimant` terminal-observation ERROR (extend the in-txn SELECT to `status`) | `fused-memory` | **leaf** | Against a seeded `done` row: `set_task_claimant(id, claimant_run_id='x', heartbeat_at=now)` emits the structured ERROR line (journal-visible) and the write persists; a heartbeat-only call on the same row emits nothing (B8/B9) | — |
| **d5** | Amend C4-E2/C4-E7 in the enforcement PRD + OPERATIONS.md §4 operator subsection (gauge, census, re-arm trigger, ε.2 pointer) | `docs`, `plans` | **leaf** (non-code) | `git grep -n 'claimant-invariant-detection' -- docs/prds/claimant-invariant-enforcement.md OPERATIONS.md` returns hits in the C4-E2 clause, the C4-E7 clause, and the OPERATIONS §4 subsection, where it returns zero today | d1, d3, d4 |

**Routing notes.** All tasks are `task_kind='normal'`; d2 and d5 (docs-only edits) additionally
carry `metadata.complexity='simple'` to take the single-agent fast path. Deliberately NOT
`execution_class='operational'`: that declaration is converted at submit into a deterministic
always-escalates pure gate — i.e. it routes the work to a human — and Leo ruled (2026-08-24)
that human attention is reserved for work that genuinely needs it; these are mechanical prose
edits an agent executes. (The η/4626 precedent was retyped the same way under the same ruling.)

**G2 note.** d1 is the sole intermediate; it names d2/d5 (and recurrence r4) as its unlocks and
still carries a direct product signal. d3/d4's signals are demonstrated against seeded rows —
the enforcement PRD's γ precedent: observation through the product's own read path on the real
code path, acceptable where found traffic is (by design, post-β) absent.

**G6 note.** No signal asserts a corpus count (the enforcement PRD's own rule). d4's signal is a
rejection-style assertion whose mechanism the task itself builds and demonstrates firing (B8),
with B9 as the negative control. d1's blind-state signal likewise demonstrates the mechanism it
builds (B2).

**G7 walk** (against `docs/legibility/design-invariants.md`, 8 invariants): INV-1 — the alarm
tier ships as gauge code consuming α's exported predicate plus a ledger check; the contract
amendment is prose *about* delivered mechanisms, not a prose contract. INV-2 — the gauge carries
denominators and `measured_at`; the ERROR/WARNING carry structured fields the emitter already
holds. INV-3 — the gauge reads ground truth (the DBs) directly; no snapshot acted on. INV-4 —
this PRD's fail-soft is the blind state, which is loud on the surface and triaged by d2's
checklist step; nothing suppresses repeatedly in silence. INV-5 — predicates consumed from α;
the endpoint is the single census implementation the watcher reads rather than re-deriving.
INV-6 — no claimed states touched. INV-7 — consumption owner named (watcher rotations);
the gauge's own liveness owner is the dashboard watchdog (armed, verified). INV-8 — the census
is bounded (per-root row scan, ms-scale) on the dashboard's async DB pool; nothing added to the
orchestrator or fused-memory event loops except d3/d4's O(1) in-path work.

## Out of scope

- **ε.2 (the escalation-grade alarm)** — pre-designed in D-7, built only when D-4's trigger
  fires.
- **Refusing manual mints** (`set_task_claimant` status-awareness) — stays observation-only;
  the enforcement PRD's rejection of refusal stands.
- **Re-arming the orchestrator-watchdog timer / installing the missing nightly units** — an
  operator action this PRD's evidence motivates but does not own.
- **Any assertion on `heartbeat_at`** — refuted by the heartbeat race; task 4667 owns the
  residue.
- **Metrics infrastructure** — none exists; introducing one is not justified by a ~0/day signal.
- **The hygiene tier as an alarm** — reported, never alarmable (enforcement D3 preserved).

## Open questions (tactical)

1. **UI placement of the invariant chip** (which dashboard page/panel hosts it). Suggested:
   alongside the existing per-project task panels. Decide during d1.
2. **Whether d2's checklist step also names ζ's census command** as the deeper-dive follow-up.
   Suggested: yes, one line. Decide during d2.
