# Task × escalation state graph — functional specification

**Status:** normative — authored 2026-08-02 from the strand investigation
(brief `2026-08-02-task-escalation-state-graph-spec`; research verified
against working tree `c26d8dd6fa`, cites re-verified unchanged at
`298556cc25`; adversarially reviewed — 15 attack findings folded in). This document states what the
task-status × escalation-state graph **should** do — not what the code
does. §8 lists where the code diverges today; the remediation PRD is
`plans/task-escalation-state-graph-prd.md`. The invariants this spec
grounds are `status-matches-liveness` (INV-6) and
`holds-owned-and-bounded` (INV-7) in
`docs/legibility/design-invariants.md` — that doc is the normative copy
of the invariant text (INV-5: no restatement here).

ARCHITECTURE.md §3/§6 remain the descriptive introduction; where the two
disagree, this spec is the intent and the divergence is a defect (either
doc or code). `shared/src/shared/task_transitions.py` (Table A),
`escalation/src/escalation/action_effects.py` (Table B), and
`orchestrator/src/orchestrator/task_ground_truth.py` (`_RECOVERY`) remain
the sole machine-readable authorities per the task-status-authority PRD
D1 and program decision #4 — this spec constrains their *content*, it is
not a fourth table.

## 1. Definitions

- **Claimant / liveness.** A task is **claimed** when `claimant_run_id`
  is set and `heartbeat_at` is fresh within TTL — the columns and the
  `is_stranded` predicate in `shared/src/shared/task_claimant.py` are the
  canonical oracle (task-status-authority D4). For statuses outside
  `in-progress` (which `is_stranded` deliberately gates on), the
  status-agnostic claimant-liveness core is the oracle — two consumers
  must never disagree about the same row's liveness because they chose
  different status-gated wrappers. **Stranded** = non-terminal, status
  implies activity, no live claimant.
- **Hold.** Any condition that keeps a non-terminal task from
  dispatching or completing: a parked status, an open escalation that
  pins recovery, a gate, a wait loop.
- **Handoff.** An escalation record whose consumer tier is expected to
  act: L0 → the per-task steward, L1 → the auto-watcher rotation, L2 →
  human (ARCHITECTURE.md §6).
- **Live handoff vs dead record.** An L0 is a live handoff **only while
  its steward/workflow process lives** — stewards are in-process, and the
  fleet redeploys on an 8h clock, so an open L0 with no live workflow is
  a dead record, not a handoff. L1/L2 records are queue-backed handoffs:
  durable, with supervised consumers (the watcher rotation and its
  outage tripwire; the human attention surfaces). This distinction is
  load-bearing for §3-S6: 7 of the 9 tasks stranded on 2026-08-02 were
  pinned by their **own dead-steward L0**.

## 2. Competing demands this spec holds simultaneously

The graph serves many constituencies; prior incidents show each demand
below being individually reasonable and jointly conflicting. The spec's
resolutions are §3; the mapping demand → resolution is §6.

1. Never second-guess a deliberate handoff (open escalation = someone is
   nominally on it).
2. Never strand (a task not making progress is visibly not making
   progress, with an owner and an exit).
3. Never phantom-done (recovery flips to `done` only on positive,
   attributed, provenance-carrying evidence).
4. Preserve WIP (branch, worktree, `.task/plan.json`, warm-lane session)
   across parks and blocks.
5. Merge-train parking: the train worker owns `merge-deferred` rows.
6. Infra hold: resume-at-verify without re-competing for footprint.
7. Deterministic gates: born-at-L2 quiescence — `blocked` and
   un-dispatched while the L2 is open.
8. Escalation dedup: the open record IS the dedup key; recovery must not
   force re-files or stack duplicates.
9. Restart survivability: in-memory state dies every ≤8h; on-disk state
   can go stale; L0s are amnestied at startup, L1/L2 survive.
10. Throughput: a stranded task starves its dependents — fleet-scoped,
    since cross-project external deps read these statuses live.
11. Escalation records are load-bearing ownership tokens elsewhere
    (merge-halt rehydration scans preserved L1s) — records must not be
    auto-expired or auto-resolved by recovery.
12. Anti-churn: requeue caps, the signature-keyed reblock guard, and
    park-and-stop all count status flips; recovery must not fight them.
13. Every done-flip passes the delivered-checks gate and provenance
    integrity; "held awaiting provenance arbitration" (`stale_conflict`)
    is a legitimate third disposition.

## 3. Core principles (the resolutions)

**S1 — Truthful liveness.** `in-progress` means exactly: a live claimant
owns this task. That covers a running pipeline and a workflow waiting
in-slot on its steward (Path A) — both hold a fresh heartbeat. No exit
path may leave `in-progress` behind after the claimant is released.
Every `run()`/slot exit writes its successor status through the
transition's designated choke point **before** the harness nulls the
claimant. Sweeps are crash backstops, never a path's designed exit ("the
stranded sweep will pick it up" is a forbidden design).

**S2 — Single writer per transition class.** Each transition class has
one choke point (`_mark_blocked` for failure-parks, `mark_done` +
provenance for done, the train worker for `merge-deferred` exits, Table
B effects for resolution actions). A path that needs a new exit shape
extends the choke point; it does not write status ad hoc or skip the
write.

**S3 — Parks are visible and owned.** A task waiting on an escalation
with no live workflow is `blocked` (or its dedicated park status:
`merge-deferred`, `infra-hold`) — never `in-progress`. The standing
invariant `blocked ⇒ open escalation record or gate marker` is hereby
documented (it held empirically 52/53 on 2026-08-02; the deterministic
MILESTONE carve-out is the sanctioned exception). The record is
simultaneously the dedup key, the ownership token, and the wake edge:
resolution events flip `blocked` rows via Table B — so any hold
expressed as `blocked` + open record inherits the whole existing exit
machinery (resolve actions, blocked-redispatch on close, b3 auto-unblock,
needs-attention surfaces) for free.

**S4 — Handoffs are honored by conversion, not by silence.** When
recovery meets a stranded row that carries an open escalation, it
neither reverts it to `pending` underneath the responder (demand 1) nor
leaves it stranded (demand 2): it **converts the row to `blocked`**,
attributed to the pinning record — no new escalation filed (demand 8),
WIP preserved (`blocked` is workflow-preserve, demand 4), and a
structured fact emitted naming task, decision, and pinning escalation
ids (INV-2). Conversion re-couples the strand to the escalation ladder's
existing ownership and wake edges (S3) instead of inventing a new
consumer loop. A stranded row with **no** open escalation takes the
record-free dispositions exactly as today: revert to `pending`, `done`
with provenance via S7, or — for the verified-green stranded-blocked
shape — the stranding-remediation-α merge-queue-direct path.

**S5 — Bounded unless parked.** Every hold carries a bound: a deadline
or progress-refreshed idle window for automation waits (the task-3170
steward-wait pattern), a cap/streak escalation for repeating suppressions
(INV-4), or a supervised consumer + age surfacing for queue-backed
handoffs. `park` (human, L2 kept open) is the **only** sanctioned
unbounded hold — the explicit "deliberately forever" marker. An unbounded
`await` on an escalation event with no deadline is a defect.

**S6 — Recovery discriminates record class, never `bool(open)`.** The
"does an open escalation pin recovery?" predicate is **one shared
function**, consumed by every sweep and gate (INV-5). It distinguishes:
(i) the task's own gating L0 whose **filing incarnation** is dead — a
dead record, not a handoff: recovery converts per S4 and the orphan-L0
reaper promotes it for visibility. Liveness of a handoff is judged
against the incarnation that *filed* the record (escalation references
carry the filing claimant/session identity), not against "some workflow
for this task is alive" — a newer incarnation never keeps a prior
incarnation's unconsumed L0 alive, and the reaper promotes on
filing-incarnation death regardless of newer live workflows;
(ii) queue-backed L1/L2 — a real handoff: hold as
`blocked` (S4), bounded by the ladder's consumer supervision and age
surfacing; (iii) `info`-severity records — never pin anything. A record
with missing or out-of-vocabulary severity **fails safe to pinning**
(treated as a handoff), never to conversion. Severity
and level must therefore travel with every escalation reference
(`EscalationRef` carries them); a predicate that cannot see severity
cannot implement this spec. **Store correctness is part of the
predicate's contract** (the esc-3163 wrong-store lesson): the records
are read from the task's owning orchestrator's escalation store — the
same instance the harness files into, never the reconciliation store —
and "store unavailable / read failed" is a distinguishable third
result, mapped to LEAVE plus a structured
`recovery_left(reason=escalation_store_unavailable)` emission. It is
**never** collapsed into "no records", because under S4 a false `[]`
would route a genuinely-pinned strand into the plain revert branch —
the precise demand-1 violation conversion exists to prevent.

**S7 — Done only with provenance.** Unchanged from
found-on-main-provenance-integrity and delivered-checks: recovery may
flip to `done` only through `MARK_DONE_WITH_PROVENANCE` with the
capability gate consulted; conflicting evidence yields `stale_conflict`
(held, in-progress, awaiting arbitration) — a legitimate tri-state, not
a failure to decide.

**S8 — Surfaces tell the truth.** Live, stranded, and held-by-escalation
must be distinguishable on every operator surface that shows activity:
task rows carry claimant/heartbeat-derived state, escalation views mark
records that pin a task's recovery, and any aggregate presented as
"active work" reconciles against the enforcing ground truth (live slots
vs rows-by-status), alarming on divergence instead of silently counting
strands as activity.

## 4. The state × owner table (normative)

Every non-terminal state names the actor that owns its exit and the
bound on the hold. "Owner" = the component that will, unprompted, move
the task out of this state.

| Status | Means exactly | Entry writer(s) | Exit owner | Exit triggers | Bound |
|---|---|---|---|---|---|
| `pending` | dispatchable when gates pass | submit/curator; requeue writers; resolution actions; recovery revert | scheduler | dispatch (pending-only by design) | starvation/fairness ladder; dispatch gates are visible per-tick |
| `deferred` | authored, not yet released | planning_mode submit | planning session (`commit_planning`); recon | commit → `pending` | planning-session lifetime; recon sweeps |
| `in-progress` | a live claimant owns it (running, or Path-A escalated-waiting) | the dispatching workflow (claim stamp + heartbeat) | the workflow slot | `run()` outcome via its choke point (S1/S2) | heartbeat TTL; steward-wait idle window (task 3170) for the escalated case |
| `blocked` | parked with an open escalation record or gate marker that owns re-entry (S3) | `_mark_blocked`; Table B `park`; deterministic gate; recovery conversion (S4) | the record's consumer tier (L0 steward / L1 watcher / L2 human); scheduler blocked-redispatch after close | resolve actions (Table B); redispatch when record closed + deps clear | ladder promotion (L0 reaper 600s); watcher supervision + outage tripwire; L2 age surfacing; `park` = explicit unbounded |
| `merge-deferred` | parked in an atomic train; train worker owns it | train park choke point | group-merge worker | train lands / fails / re-drives | train terminal outcome (worker supervised; queue state durable and rehydrated across restarts — two-layer merge queue) |
| `infra-hold` | parked for infra weather; resume-at-verify without re-compete | `_mark_blocked(block_status='infra-hold')` | infra-resume machinery (cascade on resolution) | resume-at-verify | `max_consecutive_infra_resumes` |
| `review` | *(residue — zero code writers; retained only so human writes are not rejected; candidate for retirement, §8-E14)* | human | human | human | *none — a knowingly non-conforming S5 residue, tolerated only pending the E14 retirement decision* |

**Status × escalation-state legality.** `in-progress` may coexist with
open escalations **only** while the claimant is live (Path A's steward
wait; a run that filed an info record and continued). `in-progress` +
open escalation + no live claimant is illegal beyond one sweep interval
— the sweep converts it (S4). `blocked` requires an open record or gate
marker; `blocked` with neither is a strand of the blocked shape and is
owned by the blocked-redispatch sweep. Terminal statuses may carry open
records transiently (operational closes race task_kind coercion); the
escalation queue's own hygiene owns closing them.

## 5. The outcome contract

`WorkflowOutcome` → allowed exit status-row, as the spec requires it
(the code's current loose table and the paths that force each loosening
are in §8):

| Outcome | Row at exit (spec) | Rule |
|---|---|---|
| `done` | `done` — or `cancelled` reported as CANCELLED | a producer observing a cancelled row returns CANCELLED, never DONE |
| `blocked` | `blocked` / `infra-hold`; steward terminal decisions (`cancelled`/`deferred`/`merge-deferred`) reported distinctly or via the preserve carve-out | every BLOCKED exit passes `_mark_blocked` (or the documented preserve guard); no BLOCKED exit leaves `in-progress` |
| `escalated` | `blocked` | ESCALATED is an *internal* condition handled by the escalated-wait machinery (steward, bounded); a run() **exit** with ESCALATED means the wait concluded in re-escalation → `_mark_blocked` ran. No steward-less ESCALATED exit exists |
| `requeued` | `pending` | the producer (or its choke point) writes `pending` before exit; no "deferred-to-stranded-sweep window" |
| `merge-deferred` | `merge-deferred` | unchanged |
| `cancelled` / `soft-cancelled` | the cancel choke point's written status (`cancelled`, or the park/teardown target); never silently "wherever it was" for designed (non-crash) paths | crash-shaped cancels are the sweep's legitimate backstop case |
| `planned` | *(internal-only; never a run() exit — drop from the exit contract)* | |

Two sanctioned relaxations (races are real): every row additionally
admits an **observed terminal** reported as that terminal (a concurrent
out-of-band cancel/complete wins; the producer reports what it
observed), and a **status-write failure at exit** (store down, typed
rejection after retries) reclassifies the exit as crash-shaped — the
sweep's sanctioned backstop — emitting **one** structured
store-unavailable record, not a per-task contract-violation storm.

Violations of this contract at run()-exit are otherwise **loud**: the
consistency check escalates (files a structured record) rather than
degrading to a mislabeled synthetic report (INV-4; today's soft-fail is
§8-E11).

Because the current loose entries are live-enforced expectations (SM-2
asserts them in-process; the store enforces Table A transitions with
`enforce_transitions: true` since 2026-07-14), **tightening lands
log-mode-first behind a soak, and only after the producer fixes land**
(task-status-authority D6 precedent; deploy per D8).

## 6. Demand → resolution map

| Demand (§2) | Resolved by |
|---|---|
| 1 handoff vs 2 never-strand | S4 conversion + S6 record-class discrimination: live handoffs hold as visible `blocked`; dead records don't pin; nothing is reverted under a responder; nothing stays stranded |
| 3 phantom-done | S7; plus the recovery-veto family stays for done-flips (any non-info open record still vetoes MARK_DONE — that half of the veto is correct and keeps its protective rationale) |
| 4 WIP | S4 converts to workflow-preserve `blocked`; lanes/branches/plan.json untouched; warm-lane holding policy unchanged |
| 5 train / 6 infra / 7 deterministic gates | dedicated park statuses with named owners (§4); S4 conversion never touches them (`merge-deferred`, `infra-hold` are not strand shapes; deterministic quiescence rides `blocked` + open L2 per S3) |
| 8 dedup | S4 files nothing new; the existing record moves with the task into `blocked`; `has_open_l1`-keyed re-file logic sees the same record |
| 9 restart survivability | status + record are durable; conversion is idempotent and recomputable from (status × claimant × open records) at startup; L0 amnesty interplay: a strand whose L0 was amnestied has no record → plain revert path (S4's else-branch) |
| 10 throughput | conversion → `blocked` → record resolution → existing redispatch; dependents unblock at ladder latency instead of never; Path B's producer fix (§8-E1) removes the dispatch-burn loop entirely |
| 11 records as ownership tokens | S4/S6 never resolve, expire, or dismiss records; only consumers do |
| 12 anti-churn | conversion is not a blocked→pending flip (charges no reblock counter) — but it IS a blocked *transition*, so the conversion writer is excluded from the park-and-stop window (else converting an accumulated backlog trips a false storm and pauses the scheduler) and carries its own INV-4 streak counter; and because the startup L0 amnesty can dissolve a converted row's pin, the **sweep-driven** blocked→pending flip must charge the same signature-keyed counter the cascade flip charges — otherwise strand→convert→amnesty→flip→re-strand cycles at the 8h fleet period with every cap reading zero |
| 13 tri-state done | S7 keeps `stale_conflict` |

## 7. Recovery & sweep contract

1. **One classifier.** `_RECOVERY` (TaskGroundTruth) stays the single
   classification table (W10 TG-2); the escalation-pin predicate inside
   it is the one shared function of S6. Sweeps and gates dispatch on the
   classifier's action — they do not re-derive policy in guard order.
2. **The missing shapes get explicit rows.** `(in-progress, no live
   claimant, open-escalation)` → CONVERT_TO_BLOCKED (S4) for **all four**
   branch states — `EXISTS_OFF_MAIN`, `GONE_NO_MARKER`, **and**
   `ON_MAIN` / `GONE_WITH_MERGE_MARKER` (a landed-but-pinned row converts
   too: the record closes → redispatch → the any-level dispatch gate
   completes it with provenance; leaving it would violate demand 2 while
   S7 correctly keeps the done-flip veto). No shape falls to a silent
   LEAVE default: an unenumerated shape is itself a loud structured fact
   (INV-2). Conversion mechanics: **clear the claimant before the status
   write** (so every liveness oracle agrees the row is unclaimed) and
   **re-read (status, claimant, record-open) immediately before writing**
   (INV-3 CAS) — on any mismatch (record resolved, row re-pended, live
   claimant) fall to the matching other branch, never clobber.
3. **Every veto/LEAVE emits.** A structured event
   (`recovery_vetoed`/`recovery_left` with task id, shape, pinning
   record ids + ages) — never a bare `return None`/`continue`. A streak
   of N identical vetoes on one task escalates (INV-4). An unreachable
   or wrong escalation store is its own emitted LEAVE reason, never an
   empty-records result (S6).
4. **Wake edges.** Converted rows ride the existing
   resolution-event → cascade wake for `blocked` (demand-free: no new
   machinery). `resume` acts on claimant-liveness, not on
   `status == 'blocked'` string equality (completes task-status-authority
   goal 5 / B2), and the L0-resolution path must reach orphaned rows
   (the level ≥ 1 gate assumed every L0 has a live workflow — false for
   any crash between file and exit).
5. **Producer fixes precede sweep hardening.** The sweep-conversion is
   the backstop; the designed exits (§5) are the fix. Order of landing:
   emission (observability) → producer fixes → conversion log-mode →
   conversion enforced → table tightening, each behind the D6/D8 gates.
6. **Orphaned parks get re-owned.** A `blocked` or `infra-hold` row with
   no open pinning record, no gate marker, and no live claimant has lost
   its exit owner (crashed cascade, amnestied/close_only'd record) — a
   startup/periodic invariant re-owns it (re-pend, or the existing
   re-file row) instead of leaving a permanently silent hold. The
   deterministic-gate carve-out keeps its own owner (the
   deterministic-recon re-file sweep; its archive-inclusive dedup — a
   human-resolved gate record deliberately stays parked — is the
   documented operator-owned exception).
7. **Startup ordering is pinned.** L0 amnesty → halt/queue rehydration →
   stranded reconcile. Conversion must never attribute a hold to a
   record the amnesty dismissed milliseconds later.
8. **Converted rows and the ladder.** Conversion files nothing, so the
   pinning record carries no dry-run proposal — the b3 low-risk
   auto-unblocker aborting on it is by design (expected, not a defect);
   the escalation-watcher playbook carries an explicit row for the
   `strand_converted` marker so the L1 rotation has a disposition for
   these records (the missing-playbook-row failure class).
9. **Merge-halt holds keep their token semantics.** The halt's owner is
   the durable halt-category record: the **only** unhalt edge is that
   record's resolution. A bounded exit from the in-slot wait (or its
   removal) must not run the cancellation cleanup that unhalts today;
   halt rehydration matches on category at level ≥ 1 (promotion must not
   change rehydration identity), and recovery never re-files a sibling
   halt-category record.

## 8. Divergence register (code vs this spec, 2026-08-02)

Verified sites; consequences characterized. E-numbers are referenced by
the PRD's tasks.

- **E1 — Path B merge-entry bail** (`workflow.py:3457-3464`): returns
  ESCALATED with no steward, no status write → permanent strand vetoed
  by its own record (9 live strands; 7 self-pinned by the task-2505
  tripwire's blocking L0 filed 12 lines earlier against a "purely
  observational" comment). Fix: enter the same escalated-wait machinery
  as Path A (bounded; L2/critical short-circuits to `_mark_blocked`).
  The *gate itself is correct* (stop-the-line, repend D4) — the defect
  is the missing post-bail owner.
- **E2 — `_mark_blocked(merge_phase=True)` writes no status** (every
  `_submit_to_merge_queue` block path): BLOCKED exits leave
  `in-progress`. Investigate the carve-out's original rationale, then
  make merge-phase blocks write their park status.
- **E3 — merge-halt trio** (`_handle_wip_recovery_no_advance` /
  `_handle_unmerged_state` / `_handle_stash_failed`): unbounded
  `_escalation_event.wait()` (violates S5) and post-resolution BLOCKED
  return with no write (violates S1; heals only via sweep latency).
  Fix constraint (§7.9): the wait's existing cleanup unhalts on ANY
  exit when the waiter owns the halt — a bound or an escalate-and-block
  rewrite must exit without that cleanup, and halt rehydration's
  `level == 1` filter must widen before any promotion of halt-owner
  records is possible.
- **E4 — REQUEUED without `pending`**: `WarmLaneRequeue`
  (`workflow.py:2853-2901`) and the soft-cancel spurious-wakeup fallback
  rely on the sweep; any open escalation converts a transient capacity
  signal into a Path-B-class strand.
- **E5 — infra-hold resume manufactures the strand shape**
  (`harness.py:12846-12866` writes claimant-less `in-progress` and
  waits for the sweep): re-creates the 3465 starvation whenever any
  record is open. Resume should write a truthful dispatchable status.
- **E6 — blast-radius requeue pending-write failure**
  (`scheduler.py:7024-7037`): held in-memory locks make the row read as
  actively-held → LEAVE until restart; the "next reconcile cycle"
  comment is false.
- **E7 — the recovery veto is `bool(open)` at any level/severity**,
  five hand-rolled copies (task_ground_truth `_shape`; harness
  `:5560`; scheduler `:5719`; harness `:11759`+`:11373`), silent at
  four of the five sites (the fifth, `:11373`, logs an info line — not
  a structured fact); `EscalationRef` drops severity so no site *can*
  discriminate (S6), and no site distinguishes a wrong/unavailable
  store from "no records" (S6 store-correctness). The in-progress applier re-derives resolver policy
  in guard order (INV-5).
- **E8 — `_already_landed_dispatch_gate` (`harness.py:10137-10142`) is
  still L1-only** while every sibling went any-level: a pending task
  with only an open L0/L2 can be auto-flipped to done past a human
  handoff — the live *inverse* hazard (phantom-done). Highest-urgency
  small fix in the family.
- **E9 — B2/goal-5 unimplemented**: `_cascade_unblock_member` requires
  `status == 'blocked'` (`harness.py:12873`) and the resolution path
  gates L0 at `level >= 1` (`harness.py:12014`) — `resume` on a strand
  reaches no re-pend path at all; `granted_files` scope grants deliver
  to nobody.
- **E10 — `_OUTCOME_ALLOWED` loose entries are documented strand
  windows** (`task_transitions.py:279-330`): `escalated`/`requeued`/
  `blocked`/`soft-cancelled` ⊇ {IN_PROGRESS}; `planned` unreachable;
  `requeued→BLOCKED`'s comment cites a flip SM-2 can never observe; two
  DONE producers can report a cancelled row as DONE.
- **E11 — SM-2 violation handling is soft-fail**: AssertionError →
  generic catch → synthetic BLOCKED with empty reason, no escalation —
  the false-done detector's alarm goes to a log line (INV-4).
- **E12 — silent sweeps**: the stranded sweep logs nothing when every
  candidate is vetoed; summary tallies omit vetoed/left; Path B's bail
  logs a count, not ids; burndown counts strands as active (859/7871
  snapshots over cap, peak 33 vs 24) with no parity alarm; dashboards
  drop claimant fields at ingest and render strands with an agent chip.
- **E13 — doc/comment divergences**: ARCHITECTURE.md §3.6 (no Path B),
  §3.7 (veto omitted; "sole choke point" false — six other blocked
  writers), transition diagram missing 8+ live edges (incl. the
  deterministic `pending→blocked` edge its own contract depends on);
  `harness.py:4156-4165` warm-lane invariant comment assumes the
  pre-veto sweep; `_reconcile_one_stranded` docstring says "L1-only"
  and cites a dead line; `workflow.py:3438-3439` "purely observational";
  stale `resolve_issue` docstring; `_UNION`'s call-site anchors rotted.
- **E14 — `review` status has zero code writers** — vocabulary residue;
  decide retire-or-document.
- **E15 — adjacent tasks**: 3429 (tripwire gating decision — correctly
  identifies the contradiction and defers the strand to 3423) and 3423
  (the strand-fix owner on paper, but recon has since redirected its
  description to the reify-5879 flap investigation, so no task currently
  owns the strand fix; its `metadata.files` still names the never-landed
  strand test). Reconciled at PRD queue time.

## 9. Non-goals

- **No weakening of merge gating.** Stop-the-line at MERGE entry stays,
  including prior-incarnation born-at-L2 records (repend D4).
- **No auto-expiry or auto-resolution of escalation records** by
  recovery (demand 11).
- **No new status.** `blocked` + the pinning record already carries the
  needed semantics with its full exit machinery (S3/S4); a `stranded`
  status would need its own consumer loop and would join the sweep
  carve-out treadmill the two prior splits document.
- **No change to warm-lane holding policy** (non-terminal tasks pin
  lanes deliberately, for session resume).
- **No live-fleet mutation while landing this**: the 9 dark-factory + 3
  reify strands are the validation evidence; fixes demonstrate on tests,
  and the strands recover through the landed mechanisms.
