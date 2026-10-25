# Verify-semaphore and merge-queue ordering policy — lead synthesis (v2, post-critic)

2026-09-18. Lead: interactive Fable session. Sources: eight seat reports in this directory
(`map-*.md`, `measure-*.md`, `model-sim.md`, `design-options.md`), `critic.md`, data and
scripts under `data/`, `scripts/`, `scripts/critic/`. Labels: [M] measured, [S] simulated,
[D] derived, [C] read from code. "lead-checked" = re-verified by the lead. Nothing is
filed or changed. v1 of this document carried an open-queue headline (2-class cuts the
mean 31–48 %) that the critic refuted; `model-sim.md`'s magnitudes inherit that error and
should be read through §1 here.

## 1. Answer

**The factory is a closed loop with pinned WIP and a saturated slot, so no ordering policy
at the semaphore can change mean task duration or throughput. Ordering only redistributes.**

- WIP is pinned: time-averaged in-flight 24.4 / 23.2 / 23.6 of 24 across the three runs in
  the window; ~630 dispatch candidates waiting [M, critic].
- The slot is saturated: ρ ≈ 0.99 once service times are matched to the census era [D,
  critic; lead-checked the service trend]. v1's "ρ 0.68 vs 0.95 unresolved" was a pooling
  error — the dominant cell's service time tripled (below), and the "411 unattributed calls"
  are ungated train gate runs (daily-count r = 0.998; a timestamp join was not possible).
- Little's law: N = X·R. N is pinned by `max_concurrent_tasks`; X = slot capacity ÷ slot
  work per dispatch, which ordering does not touch; so mean residence R is invariant under
  EVERY ordering — size-aware ones included. The closed-loop sim (validated: 75.1 vs 75.7
  plans/day measured; wait p50/mean 0.69 vs 0.73 measured, waits ~20 % low) confirms it [S]:

| policy (N=24) | plans/day | plan completion mean / p50 / p90 / max (h) | SHORT mean | LONG mean / p90 / max |
|---|---|---|---|---|
| random (today) | 75.1 | 4.82 / 3.53 / 10.51 / 34.0 | 4.44 | 5.15 / 11.03 / 34.0 |
| FIFO | 75.3 | 4.85 / 4.76 / 7.33 / 11.6 | 4.54 | 5.14 / 7.61 / 11.6 |
| 2-class short-first | 76.6 | 4.78 / 4.73 / 10.84 / 16.4 | 0.46 | 8.95 / 11.84 / 16.4 |

What each policy actually is:

- **FIFO = a tail lever.** Mean ±0, median *worse* by a third, p90 −30 %, max −66 %
  (34 h → 12 h). It fixes incident-7414-class starvation. The random winner is measured:
  a later arrival beat an earlier waiter in 1410/2801 = 50.3 % of pairs [M].
- **2-class short-first (ruling R8 / task 5413 Part B) = a redistribution, not a mean
  lever.** Plans < 300 s fall 4.4 h → 0.5 h; orchestrator full-suite plans rise 5.1 h →
  9.0 h (p90 ~12 h, max above FIFO's). The LONG class is 22 % of arrivals but **86 % of slot
  work**, and it is the orchestrator module — where the merge-lane quality programme lives.
  LONG waiters hold orchestrator module locks and get staler (conflict probability rises
  with base age, §3), so the redistribution is paid by the work Leo has ranked first. Worth
  doing only as a deliberate value judgment, not as a speed-up. R8's success signal ("SHORT
  p50 < 30 min while LONG p90 within +20 %") is predicted to FAIL on its second clause
  (LONG p90 +56 % vs FIFO).
- **Static priority** = the same redistribution keyed on tier: unweighted mean unchanged,
  low tier starved (max 32 h) unless aged; an aging bound below the prevailing wait silently
  turns it into FIFO [S]. A weighted-latency instrument only.

**The levers that do move mean duration and throughput** (all order-independent):

1. **WIP.** With the slot saturated, mean residence ∝ WIP at constant throughput. Closed-loop
   sweep [S, lead, `scripts/critic/lead_wip_sweep.py`; FIFO rows, other policies tie]:

   | max in-flight | plans/day | slot busy | plan completion mean / p90 / max (h) |
   |---|---|---|---|
   | 48 | 75.4 | 100 % | 12.4 / 15.9 / 20 |
   | **24 (today)** | 75.3 | 100 % | 4.85 / 7.3 / 11.6 |
   | **16** | 74.1 | 98 % | **2.48 / 4.3 / 7.7** |
   | 12 | 68.2 | 90 % | 1.47 / 3.0 / 6.3 |
   | 8 | 54.1 | 71 % | 0.82 / 1.9 / 4.4 |

   24 → 16 halves verify-phase residence for ~1.5 % of throughput; the knee is ~14–16.
   Caveats: the model is the verify loop only (think time Z = 2.74 h/plan derived, not
   measured per phase; merge/review legs not modelled); `max_concurrent_tasks` is red-tier and
   a restart cancels the whole WIP (below). Shorter residence also cuts base age at merge
   (the one real drift channel) and shrinks what each restart destroys.
2. **Service time of the orchestrator full-suite task leg** — 86 % of slot work, and it has
   tripled [M, lead-checked]: mean 644 s (Jul, n=18) → 821 → 1 267 → 2 877 → 2 597 →
   **3 066 s** (09-12+, n=38, p50 3 017). Candidate causes, unranked: `verify_admission_pytest_n:
   "8"` (08-19), suite growth, seven orchestrators on one host, merge-role `-n 16` contention.
   Every second removed here is throughput AND duration. Also unasked until now: why 130 of
   199 orchestrator task legs run the full suite rather than file-scoped.
3. **Slot demand per landing**: 5410 (D1 disjoint-rebase skip), 5409 (NULL `tip_sha` re-verify),
   fail-fast across a plan's legs (a red 40 s leg could cancel its still-waiting 51-min
   sibling — today nothing does; `verify.py::run_scoped_verification` gathers, [C]), and the
   restart sweep: **100 % of cancelled dispatches sit on restart minutes** (DF 24/2/22; reify
   38/47/23/24) [M, critic] — a restart cancels the entire WIP, which re-verifies from
   attempt 0. At ρ≈1 that is a pure demand multiplier. (v1's "one leg per restart" and
   "~50 % cancelled, cause unknown" were both wrong.)
4. Capacity: remote-host primary (5415/5416), as already ruled.

## 2. Leo's hunch, clause by clause

| clause | verdict |
|---|---|
| Semaphore is random and priority-blind | **Confirmed** (50.3 % overtake [M]; 0.1 s poll race [C]; at N=1 the shuffle is a no-op, the race is the randomness) |
| Tasks are throughput-bound on the semaphore | **Confirmed**, more strongly than suspected: verify phase p50 5.3 h, 97 % of it neither agent nor test service [M]; ρ≈0.99; DF at ~93 % of its slot-bound dispatch ceiling |
| FIFO/priority would improve mean duration | **Refuted** (Little's law under pinned WIP; open-queue conservation law too) |
| … and throughput | **Refuted to first order.** Second-order channels exist in both directions and are unmeasured: shorter residence → fewer conflicts (+); 2-class lengthening orchestrator tasks past the ~20-commit conflict step (−) |
| via reduced rebase drift | Drift is **real for textual conflicts only** (P 0 % → 28 % DF / 48 % reify from 0–5 to 61+ commits behind, footprint held; red-on-merged-tree does NOT rise with base age) [M]. It is driven by task duration, so WIP and service time reach it; ordering does not change mean duration, so ordering does not |
| Priority-sensitive merge order offers further improvement | **Small.** Merge wait is ~2 % of accrued base age (critic recompute: DF 2.0 %, reify 0.9 %) [M]; the gate is a fixed nine-suite sweep (CV 0.28) so shortest-first has nothing to sort on [M]; an aged effective-priority tie-break halves measured inversions (79 → 42; 21 % of waiting episodes are overtaken by lower-priority work) and is worth ~0.4–0.6 dependent-task-h/day [S bracket — merge sim failed validation]. Whether merge order affects landings/day is **unknown**: the A/B cited in `measure-merge.md` is confounded (slots 3→1 an hour before the arm boundary; WIP 48→24 inside it) and the gate is 74 % busy, not saturated |

## 3. Policies worth having, and their price

**DF semaphore.** Role class (task > background; decide review-checkpoint's class
explicitly — it rides `task` by a default kwarg) → arrival FIFO. That is the whole
recommendation: it bounds the tail and makes the gate deterministic (the integration test
`TestSweepYieldsAndInterleaves` currently needs a `_DeterministicAcquire` stand-in because the
real primitive is a race). Size classes are optional and are a value judgment (§1). If
built: classify **by leg** from `(module_prefix, scope_kind)` passed as a structured kwarg
from `verify.py::run_scoped_verification` — leg-level 2-class matches plan-level to 2 d.p.
[S, critic], module alone is insufficient (orchestrator file-scoped 43 s vs full-suite
~3 000 s), and the class must not be regexed out of the command string (heuristic 12,
structured data instead of meaningful strings). Never pure shortest-leg-first: it splits
plans and worsens p90 plan completion. No load-derived thresholds (C-no-load-derived-count).

**Reify semaphore.** FIFO in the same DF-side gate (the DF flock already serialises reify's
verifies ahead of reify's bash semaphore: 58 waits/month, p50 3 s [M]); no reify-repo
change. One service class (CV² 0.15) ⇒ size ordering is a provable no-op. Reify is at
ρ≈0.93 and its restart sweeps cancel 23–47 tasks; its levers are capacity, demand and WIP.
Turn on `REIFY_SLOT_EVENT_LOG` (green-tier `verify_env` leaf) — reify's admission wait is
currently unobservable.

**DF merge queue.** Keep lane → clique-minimal aging. CORRECTION 2026-09-19: earlier text
called the priority option a "tie-break"; that conflated two policies. (i) A literal tie-break
(design-options B1a: priority replaces `request_id` when two clique peers have EQUAL
first-enqueue timestamps) would almost never fire — the timestamps are sub-second floats.
(ii) The simulated `prio_aged` policy, which is where every number below comes from, lets
priority OUTRANK age until a request has waited 4 h — a comparator change (B1b + bound),
and the sim applied it across ALL waiting requests because footprint cliques are not in the
event store. The real pick only compares ages between footprint-overlapping requests and
takes disjoint ones in buffer order, so a clique-scoped version fires only when two waiting
same-lane requests overlap in footprint, differ in effective priority, and the older has
waited under the bound; its frequency is unmeasured and the figures below are an UPPER
bound for it. A global version would undo task 1891's disjoint-FIFO rule and diverge from
`chain_snapshot`'s FIFO order. Not filed; if wanted, shadow-log the would-be pick first.
Optional inversion hygiene (as simulated): a journaled `effective_priority` on `MergeRequest`
outranking age among waiting candidates, with a ≥4 h aging bound. `scheduler.py::Scheduler._compute_effective_priorities`
is tier min-rank over undone dependents — it does NOT count dependents (v1 was wrong);
most-dependents-first would need `_compute_transitive_counts` and is worth ~0.9–1.5 h/day
aged [S bracket]. Reject priority→lane, smallest-footprint-first, freshest-first.
**Reify merge queue**: nothing (28 % utilised, wait p50 0).

Merge-side wins that dwarf ordering [M]: DF's gate does not fail fast (74 reds = 69.8 h =
25 % of gate wall; red costs 53 min vs reify's 21); the landed early-conflict bounce (tasks
1889/1892, `needs_rebase`) catches 29 of 45 DF textual conflicts at zero wait, but 16 still
reach dispatch after queueing 28 h in total (base ages 7–799 commits) — a leak in a landed
mechanism, cause undiagnosed, not a missing feature (v2 wrongly said "detected only at dispatch"); multi-episode tasks take 25.9 h vs
1.5 h for single-episode.

## 4. Implementation options

Steps 0–3 touch no file under the merge lane: `merge_queue.py` has zero references to the
admission gate, the only gate call site is `verify.py::run_verification`, and every
merge-path verify passes `role='merge'`, which `shared/verify_admission.py::is_gated_role`
short-circuits before the gate. (Fixing the separate probe-role defect in §5 WOULD add a
`role=` kwarg at two `merge_queue.py` call sites — keep it out of this change.)

| step | what | size (seat est.) | tier | notes / risks |
|---|---|---|---|---|
| 0 | Correct the yaml's "~23 % of ONE slot" comment (wrong 6.3×, and it is cited as the reason 2 slots/ordering don't matter) | minutes | — | INV-9: a wrong fact with one home still misleads every reader |
| 1 | In-process `AdmissionGate` in front of the UNMODIFIED flock, `policy: none`; emit acquire/release/wait + plan-completion events, surface oldest-waiter age (closes an INV-7 gap: today no admission wait is recorded anywhere; `started_at` is stamped inside `_admission_slot`) | 1–2 ad | green | Must be scoped against pending **5139** (high; edits `_admission_slot`; an asyncio-level gate subsumes its executor half) and re-scoped into **5413**. Keep shielded-cancel release semantics and C-fail-open — but a fail-open of *arbitration* must emit an event (INV-11). Waiters beyond the 64-thread executor already queue FIFO unseen |
| 2 | `REIFY_SLOT_EVENT_LOG` on | 0.5 ad | green | — |
| 3 | Role class + FIFO | ~1 ad | green | Strict FIFO without the role class convoys the 8-module sweep ahead of task legs. Process-local arrival stamps are fine (v1's "restart-stable stamp" requirement is withdrawn: the sim inversion did not reproduce paired, and a restart cancels the whole WIP anyway). Residual out-of-process acquirers the gate cannot order: `evals/metrics.py` at default `role='task'` when pointed at the DF root; brief two-orchestrator overlap at restart — the flock still excludes them |
| 4 | **DECLINED by Leo 2026-09-18** (recorded on task 5413) — leg-class size layer | 1–2 ad | green | §1: LONG = orchestrator full-suite pays ~+74 % mean; run step 1's instrumentation and a shadow log of the would-be order first; watch LONG p90 and orchestrator-module conflict rate |
| 5 | (optional) merge effective-priority tie-break, journaled | 2–3 ad | restart | `merge_queue.py` is 21.5k lines and the κ/λ extraction has not landed (heuristic 14) — accept rework or wait; read `_maybe_coalesce_waiting_singles` first; starvation converts to human escalation via `MERGE_BOUNCE_CAP=3` |
| — | WIP 24 → 16 | config | **red** | Not an implementation task: a ruling. Best taken at a restart that is happening anyway |

A cross-process ordered semaphore (ticket files / arbiter) is not needed for either project.

## 5. Defects and traps found on the way (each independent of the ordering question)

- `verify.py::verify_failure_is_preexisting_on_main` runs its probe at `role='task'` [C,
  lead-checked]. Correct for the `workflow.py` caller; merge-path callers reach it only on
  the degrade path (ids/baseline None). Fix: thread the caller's role.
- `merge_first_enqueued_at` is absent from `PersistedMergeRequest`; `enqueued_at` IS
  journaled, so order survives restart — only the resubmission-lineage advantage of
  journal-recovered requests is lost [C, lead-checked, critic-confirmed].
- 23 DF request ids finalize twice; `merge_queue.json` held a request 44 h after `abandoned`;
  12–17 % of merge heartbeats name a stale head-of-line; `merge_queued/dequeued.queue_depth`
  is the asyncio inbox (always 0), not depth [M].
- Measurement traps: the env-recovery retry keeps first-pass lint/type timestamps (3.1 % of
  records, wait proxy inflated ~3×; fingerprint `-p no:xdist -o addopts=`); archived verify
  logs hold zero task-lane wait data (all merge-path); pooling service times across eras
  understated ρ by a third.

## 6. Still unverified

Think time Z and hence the WIP knee (model-derived; a staged 24 → 20 → 16 with step 1's
instrumentation would measure it); the 411-calls = train-gates identification (daily
correlation only); whether merge order affects landings/day; cause of the 3× service-time
growth; reify magnitudes (parametric, open model — direction safe); priority↔size
correlation (priority-as-of-acquisition is recorded nowhere).
