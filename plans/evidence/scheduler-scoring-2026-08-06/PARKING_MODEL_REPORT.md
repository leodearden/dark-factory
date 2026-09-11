# The module-lock PARKING trade-off — a data-driven model

**Data**: full-history snapshots of both orchestrated projects, read-only.
Dark-factory: `runs_snapshot.db` (2026-04-09 → 08-06, 311,889 events), `tasks_snapshot.db`,
`scheduler_state_snapshot.json` (14:46:19Z). Reify: `reify_runs_snapshot.db`
(2026-04-03 → 08-06, 665,820 events), `reify_tasks_snapshot.db`, `reify_scheduler_state_snapshot.json`.
**Scripts** (this scratchpad): `parking_lib.py` (shared reconstruction), `parking_census.py`
(Parts A+B; stdout in `census_output.txt`), `parking_sim.py` (Part C), `noise_runs.py`
(seed-jitter bootstrap). **CSV/JSON outputs**: `park_episodes_{project}.csv`,
`holds_{project}.csv`, `census_summary.json`, `policy_metrics.{json,csv}`,
`policy_metrics_p4.json`, `noise_summary.json`.
Code read (never modified): `orchestrator/src/orchestrator/scheduler.py`, `config.py`,
`shared/src/shared/locking.py`.

---

## 0. Mechanism corrections established from code + events (they change the model)

1. **Fairness thresholds**: the task brief's "critical 1, high 2, medium 6" does not match the
   code. `_DEFAULT_SKIP_THRESHOLD` (config.py:491) is **critical 0, high 1, medium 2, low 4,
   polish 9999**, and neither project's YAML overrides `fairness` — so critical/high park on
   their **first** skip, medium on the 2nd, low on the 4th.
2. **`reservation_expired` is no longer a lease.** Lease-reason (empty-reason) expiries stop in
   **May 2026** in both projects; since then the event is owner-state park-GC
   (`terminal:done` / `deps_unsatisfied` / `missing`, scheduler.py:5466-5510). Confirms task
   1228's lease removal, and dates the "modern regime" used below (episodes installed after the
   last lease event: 2026-05-31 reify, 2026-06-01 DF).
3. **Two silent park-death paths carry most of the deaths**: (a) **process death** — parks/skip
   counts are not in the snapshot restore path; every orchestrator era end wipes them with no
   event; (b) **owner release** — `Scheduler.release()` (:7052) defensively clears the owner's
   parks with **no reservation event**. A parked task that dispatches at non-top rank (or via
   the pin loop) keeps its parks while running (only `task_id == top_id` triggers
   `reservation_used`/`clear_parks_for` at :6404-6425), until (b) fires at its release.
4. **Empty install attempts**: `install_parks` can return `installed=[]` (every requested module
   blocked by a same/higher-tier foreign park, INV-3 :963-968) yet `reservation_installed`
   still fires and `has_parks` stays False, so the attempt re-fires every tick.
   **62% of DF's and 82% of reify's `reservation_installed` events are these no-ops**
   (1,286/2,090 and 5,498/6,707; SQL on `data.modules = []`), concentrated on 7 (DF) / 9
   (reify) tasks. The fairness mechanism visibly live-locks against itself at install time.
5. **Restart cadence is far faster than assumed.** Orchestrator process eras (per-`run_id`
   event extents; runs verified strictly sequential): last-30d **median era 2.2h (DF) / 3.4h
   (reify)**; 149/138 eras touch the 30-day window; 42/29 eras are under 1h. A park's
   realized lifetime is capped by this regardless of any lease policy.

**Verification of the prior adversarial-review findings** (all re-derived):
live 3076↔3618 convoy — confirmed in `scheduler_state_snapshot.json.park_stacks`
(3076's `git_ops.py` park shadowed under 3618's rank-0 park; 3076's `harness.py` park active
while 3076 itself is blocked). Fall-through: **74%** of the last 500 non-empty dispatches
went medium/low (prior ~72%; tier stamped in `lock_acquired.data.priority`). 3256 on
`harness.py`: **51.7h over 6 spans** in the 93h incident window (prior 50.65h/5; delta =
restart re-acquire span-merge methodology). DF since 08-01: **19 installed / 3 used** exactly.

---

## Part A — Park-episode census (full history, both projects)

Episode = `reservation_installed` (non-empty) → first of `reservation_used` /
`reservation_expired` / eviction / owner lock-release (silent) / process-era end (silent) /
snapshot end. Idle-reserved mod-h = Σ over parked modules of episode time the module was
**not held by anyone** (join against the reconstructed hold trace, hierarchical conflicts
included). Demand gap = park end → next conflicting `lock_acquired` by another task.

| | DF all | DF modern | reify all | reify modern |
|---|---|---|---|---|
| episodes | 804 | 560 | 1,209 | 635 |
| **ended in dispatch-via-park (`used`)** | 50.9% | **55.2%** | 35.5% | **47.9%** |
| died at process era end (silent) | 203 | ~36% of modern | 292 | ~46% of modern |
| duration p50 / p90 | 0.75 / 3.4h | 0.74 / 3.9h | 1.0 / 6.3h | 1.7 / 8.2h |
| idle-reserved mod-h (total) | 3,974 | 3,542 | 10,379 | 8,676 |
| idle-reserved p50 / p90 per episode | 1.3 / 10.0 | 1.6 / 13.4 | 1.9 / 15.8 | 2.9 / 34.8 |
| assembly latency (install→used) p50 / p90 | 0.41 / 2.4h | 0.53 / 2.6h | 0.48 / 5.0h | 0.99 / 5.9h |
| next-demand gap < 1h | 53% | 58% | 45% | 60% |

Last-30-days slice (the window the sim replays): DF **319 episodes, 140 used (44%),
3,085 idle-reserved mod-h (~103/day)**; reify **169 episodes, 28 used (17%), 2,347 mod-h
(~78/day)** — reify's recent parks mostly die undelivered (era death + silent owner release),
DF's deliver a little under half the time.

Per-tier (modern): DF used-rate rises down-tier (critical 55%, high 49%, medium 72%, low 75%)
— the wide sets are at the top tiers and they are exactly the ones that die undelivered.
Reify concentrates idle cost in high (5,570 mod-h) and a few whale episodes: top-10 episodes
= 2,127 mod-h ≈ **25% of the modern total** from 10 of 635 episodes (census CSV, sorted).
DF top-10 ≈ 964 mod-h ≈ 27% of its total. **The idle-reserved cost is heavy-tailed** — a few
wide, long, mostly era-death/silently-cleared episodes carry a quarter of the entire cost.

Named, not dropped: 2 DF + 4 reify `reinstall_anomaly` episodes (install while an episode was
open — should be impossible per `has_parks`; treated as episode boundaries); 1-2 `live`
episodes censored at snapshot end; `reservation_force_evicted` never fired in either project
(the operator eviction lever exists but was never successfully used; 13 DF / 1 reify
`park_eviction_deferred_fm_unavailable` events show the drain being attempted).

## Part B — hold-duration structure

Hold trace from `lock_acquired`/`lock_released` pairing with explicit anomaly classes
(counts below; nothing silently absorbed):

| anomaly | DF | reify | handling |
|---|---|---|---|
| EMPTY-ACQUIRE (no modules; slot churn) | 13,875 (11,343 = task 2848) | 555 | excluded from hold trace; modeled as slot-only tasks in sim |
| ORPHAN-RELEASE | 3,594 | 5,780 | release ignored; span never opened |
| DOUBLE-ACQUIRE | 250 | 1,439 | previous span force-closed at new acquire |
| STUCK-AT-ERA-END (requeue-exit never releases) | 1,283 | 4,776 | span closed at process death — the lock **did** block others until then |
| SAME-OWNER-REACQ (restart re-emission) | 91 | 634 | merged |

Clean-released spans: DF 15,002/16,644 (90%), reify 36,330/42,685 (85%).
Durations (clean): DF p50 0.81h, p90 7.3h, p99 14.6h; reify p50 1.15h, p90 8.1h, p99 17.2h.

Top-5 hot modules, last 30d occupancy (share of wall clock; share of occupancy from holds ≥
global p90 duration): DF `fused-memory/.../reconciliation` 68%/33%, `workflow.py` 68%/28%,
`harness.py` 68%/31%, `fused-memory/.../server` 57%/41%, `merge_queue.py` 56%/32%.
Reify `engine_eval.rs` **89%**/41%, `engine_build.rs` 83%/38%, `expr.rs` 73%/41%,
`eval/lib.rs` 73%/32%, `kernel-occt/lib.rs` 70%/45%. **p90+ holds carry 28-45% of hot-module
occupancy** — trimming the tail is worth roughly a third of the hot-module capacity.

**Predictability (the EASY-backfill feasibility question)** — per-dispatch durations
(max span per dispatch group), 70/30 time-ordered train/test, log2-space R²:

| predictor | DF R² / within-2x / within-4x | reify R² / within-2x / within-4x |
|---|---|---|
| global median | -0.22 / 19% / 43% | -0.67 / 10% / 19% |
| tier median | -0.36 / 21% / 45% | -0.82 / 10% / 20% |
| tier+width median | -0.31 / 24% / 45% | -0.76 / 10% / 21% |
| **module-history median (last 10 holds on the task's modules)** | **0.26 / 48% / 71%** | **0.68 / 61% / 77%** |

Static task attributes predict nothing (negative R² = worse than the test-set mean); the
**only** useful signal is per-module hold history. Safety multipliers: pred×2.9 (DF) / ×2.0
(reify) covers 80% of realized durations; ×4.8 / ×4.0 covers 90%. So backfill admission of
the form "predicted × 2-3 fits the gap" is workable, with a named 10-20% overstay tail.

## Part C — trace-driven counterfactual simulation

Replay of the last 30 days per project through the dispatch mechanics
(score-order scan → first `try_acquire` wins; hierarchical locks; park install at per-tier
skip thresholds; INV-1 cross-tier shadowing; INV-3 same-tier install blocking; rank-aware
preemptor acquisition; parks+skips wiped at each historical process-era boundary; 15s rescan
skip accrual when slots free but candidates lock-blocked, none when slots full).

World: arrival = monotone id-envelope over first-evidence (an **upper bound** on filing —
conservative); service = realized total hold span per task (tier+width-bucket median for
never-dispatched tasks); dep gating in-sim, terminal = release + realized merge gap
(median 0.48h DF / 2.03h reify); external-dep holds until last `external_dep_gate_held`;
empty-module churners occupy slots for their historical churn span.
DF: 1,558 dispatchable tasks, 148 restarts; reify: 1,025 tasks, 137 restarts.

**Calibration (P0 vs history)**: reify — sim 867 vs 639 historical in-window first
dispatches, 597 common, **|Δt| p50 = 4.0h, Spearman ρ = 0.913**. DF — 1,380 vs 1,134,
1,062 common, |Δt| p50 = 168h, **ρ = 0.50**, signed error p50 = **-102h (sim dispatches
early)**; sim slot occupancy 52%. Main sources (measured/named): PSI dispatch-admission
holds (8,215 `dispatch_deferred` events unmodeled), requeue churn collapsed to single
dispatches, arrival envelope, cooldowns/pins unmodeled. Park-activity calibration:
sim P0 installs/used 215/58 (DF) and 157/10 (reify) vs real-window 331/147 and 173/29 —
same order, sim under-parks DF. **DF conclusions are therefore relative-only; reify is the
quantitatively trustworthy instance.** Sim idle-reserved runs ~1.4× (DF) / ~1.9× (reify)
the census rate — absolute idle levels below come from the census, deltas from the sim.

**Policy matrix** (seed-jitter bootstrap, mean [min..max] over 4 hash seeds for the five key
policies; full 13-policy single-run matrix in `policy_metrics.csv`):

| policy | DF disp/day | DF idle mod-h | DF w3+ p95 wait | DF never-disp | reify disp/day | reify idle mod-h | reify w3+ p95 | reify never-disp |
|---|---|---|---|---|---|---|---|---|
| **P1 no parking** | **47.8** [47.7..47.9] | 0 | 529 [463..561] | **53** | 28.9 | 0 | 245 | 46 |
| **P0 current** | 45.9 [45.8..46.0] | 4,466 | 528 [513..543] | 104 | **28.9** | 4,490 | 216 | 51 |
| P2-K5 | 43.5 [43.1..43.7] | 16,543 | 518 [504..533] | 155 | 27.8 [27.5..28.0] | 16,093 | 228 | 70 |
| P2-K24-B32 | 42.5 [42.2..42.9] | 21,348 | 542 [509..576] | 168 | 27.3 | 20,361 | 219 [204..234] | 75 |
| **P4 backfill** | 47.5 [47.1..47.6] | 4,212 | 532 [506..543] | 70 | **29.1** | 6,682 | **205** | **44** |

(Unbudgeted P2-K24: DF 38.4/day, 56.5k idle mod-h, 249 never-dispatched; reify 25.2/day,
84.0k mod-h — runaway. P3 leases 2h/8h/24h: within noise of P0 on every metric in both
projects; a 24h lease is literally a no-op — median process era 2.2-3.4h kills the park
first. P5 handoff as implemented: worse throughput and latency in both projects — reserving
released modules across a merge window (0.5-2h median) blocks more than it saves.)

**Starver cohort, time-to-dispatch (mean over seeds; "never" = not dispatched in 30d window)**:

| task | P0 | P1 | P2-K5 | P2-K24-B32 | P4 |
|---|---|---|---|---|---|
| DF 3534 (high, D=9, harness.py chain head) | 65h (1/4 never) | **31h** | 24h (3/4 never) | NEVER ×4 | **21h** |
| DF 3538 (downstream of 3534) | NEVER ×4 | NEVER ×4 | NEVER ×4 | NEVER ×4 | 3.4h (1/4 never) |
| DF 3060 (high, 9 modules) | 408h | 421h | 596h | 612h | **376h** |
| DF 2383 / 2370 (aged mid-tier wide) | 403 / 196h | 435 / 168h | **265 / 83h** | **207 / 100h** | 331 / 155h |
| reify 5743 (critical, 13 modules, 540 real skips) | 113h | 113h | **2.6h** | **13h** | 113h |
| reify 4956 | 450h | 569h | 407h | 485h | NEVER ×4 |
| reify 4351 (36 modules) | **66h** | 168h | **66h** | **66h** | 153h |

## Part D — the answer

### The frontier (all deltas vs P1 = zero parking)

- **Reify (well-calibrated)**: P0 parking costs **zero throughput** (28.9 = 28.9/day; noise
  < ±0.1) and buys wide tasks **-29h p95 (-12%)**. Its cost is 4,490 idle-reserved mod-h/30d
  that simply never converts into lost dispatches — the reserved files were not what the
  runnable narrow tasks needed. Extending below rank 1 (P2) then costs **-1.1 to -1.6
  disp/day (-4 to -6%)** and 3.6-4.5× the idle reservation for a wide-task p95 that gets
  *worse* than P0 (228/219 vs 216) — although it spectacularly rescues the single
  top starver (5743: 113h → 2.6h). P4 backfill strictly dominates P0: +0.2 disp/day,
  wide-task p95 216→205, fewest never-dispatched.
- **Dark-factory (relative-only)**: P0 parking costs **-1.9 disp/day (-4%)** and roughly
  **doubles** the never-dispatched count (53→104) for **no aggregate wide-task gain**
  (w3+ p95 528 vs 529, bands overlap) — the benefit exists only as targeted rescues (3534
  in some seeds, 3060), and the same parks *are* what starves 3534 in other seeds (the live
  3076/3618 convoy reproduced). P2 costs -4.3 to -5.3 disp/day (-9 to -11%) and gridlocks
  the 3534-class chain outright (INV-3 same-tier park collisions — the sim's blocked-install
  counts explode 2→6,297, exactly the pathology the real event log shows as 1,286 empty
  installs). P4 recovers P1's throughput (47.5 vs 47.8) while keeping targeted protection
  (3534: 21h, best; 3538: the only policy that ever dispatches it).

### (1) Is the owner's two-ended framing complete?

Directionally right, but the data adds four structural facts it misses:

- **The throughput cost is demand-conditional, not proportional to reservation.** Reify
  reserves 4,490 idle mod-h and pays ~0 dispatches; DF reserves the same and pays 4%. Cost
  materializes only where narrow-task demand overlaps the reserved modules (DF's monolith
  hot files; not reify's file-granular crate tree). "Idle-reserved module-hours" is the
  wrong cost currency on its own — the conversion rate to lost dispatches is the number
  that matters, and it ranges 0-4% here.
- **The benefit is concentrated, not distributed**: aggregate wide-task p95 barely moves
  (DF) or moves 12% (reify), but individual starvers move 10-40×. Both cost and benefit are
  heavy-tailed; means mislead.
- **A third axis the framing omits: park-vs-park interference.** Too much parking doesn't
  just tax narrow throughput — it starves *other wide tasks* (INV-3 same-tier install
  blocking + convoys). 62-82% of all real install events are already blocked no-ops, and
  under P2 the sim's chain-head cohort (3534/3538) is starved by *other tasks' parks*, not
  by narrow traffic. Fairness-vs-throughput is the wrong dichotomy once parks are numerous
  enough to fight each other.
- **Restart churn dominates the design's actual behavior.** With 2-3h median process eras,
  parks die silently before delivering (44% DF / 17% reify used-rate in the last 30d; only
  3/19 DF installs since 08-01 used), and skip counters reset so wide tasks re-earn their
  parks from zero, repeatedly. The no-lease "tail risk" the design worries about is largely
  moot — and so is much of the intended benefit. Fixing park/skip persistence across
  restarts (or fixing restart cadence) is upstream of every parking-policy knob.

### (2) Is extending park installation below rank 1 net-positive?

**No — not as eager full-set parks under current INV-3 semantics.** In both projects, every
K>1 variant loses 4-11% throughput, multiplies idle reservation 3.6-4.8×, *increases* the
never-dispatched count ~1.5-3×, and destabilizes the very class it targets (rescues
aged mid-tier starvers 2383/2370/5743, gridlocks same-tier overlapping chains 3534/3538 —
NEVER×4 under K24-B32). A module budget (B32) trims the idle explosion ~2.7× but does not
flip the sign of any of it. If below-rank-1 protection is wanted, the discriminator that
matters is **park-set disjointness, not rank**: extension is only safe for candidates whose
module sets don't conflict with any active park (the sim's failure mode and the real
blocked-install churn are both same-tier overlap). That variant was not simulated and is
the concrete follow-up.

### (3) Which single change buys the most per unit of throughput lost?

**P4 EASY-backfill** — predicted-fit narrow tasks may pass through parked modules when the
park's beneficiary provably can't assemble yet (its own blockers' predicted remaining time
exceeds the backfiller's predicted hold). It is the only policy that improves on BOTH ends
at once: reify +0.2 disp/day *and* wide p95 216→205 *and* fewest never-dispatched; DF
recovers 84% of the parking throughput tax while keeping (indeed improving) targeted
protection. Cost side measured honestly: 23-122 grants/30d, of which **7-15% overstay
their admission bound** (prediction error; DF 14/112, reify 8/52 at safety ×1; 7-9% at
safety ×2.5), 76-234 total overstay-hours; the aggregate still dominates. It needs only the module-history predictor (Part B: the one predictor that
works, R² 0.26/0.68) and a ×2-3 safety factor. Its named casualty: one reify starver
(4956) flips to never-dispatched in-window — backfill admission should be disabled for
parks older than some age to bound this.
Second-best per unit effort: **make parks/skips survive restarts** — free fairness at zero
throughput cost, given eras of 2-3h are wiping 36-46% of episodes before delivery.

### Strongest limitations, and the direction each biases

1. **DF calibration (ρ=0.50, sim early by ~102h median, slot occupancy 52%)**: unmodeled
   PSI admission holds and requeue churn make the sim's DF world faster-draining than
   reality. This *understates* queueing everywhere, so DF absolute latencies are floor
   values; the policy *ordering* was stable across seeds, but a mechanism that interacts
   with saturation (P2 gridlock) could be worse in reality than simulated (more contention),
   and P4's throughput recovery could be smaller (PSI would throttle some backfills anyway).
2. **Single-trajectory chaos**: cohort-level results flip across seed jitter (3534:
   1/4-never under P0). Treated by reporting seed bands; per-task numbers are illustrative,
   aggregates are the evidence.
3. **Event integrity**: 3.6k/5.8k orphan releases and 1.3k/4.8k stuck-at-era-end spans mean
   hold durations for the affected spans are bounded, not exact (closed at process death —
   correct for "how long others were blocked", wrong for "how long work ran"). Anomalous
   spans are 10-15% of the trace; idle-reserved figures inherit that uncertainty upward
   (a stuck lock reads as "held", shrinking measured park idleness).
4. **Service non-stationarity**: never-dispatched tasks get bucket-median service times;
   if the starved wide tasks are systematically longer than their bucket median (plausible),
   the fairness benefit of every parking variant is overstated for them.
5. **Not modeled at all (named)**: pin queue (6 DF events), priority-override overlays
   (10-14 events), `polish` tier parking (threshold 9999 — none exist), partial-install
   shadow/restore event asymmetry (hierarchical victims restore silently), merge-train
   coalescing, PSI, per-task cooldowns, delivered-check gates, cross-project external deps
   beyond the gate-held proxy, cancellations mid-window.
