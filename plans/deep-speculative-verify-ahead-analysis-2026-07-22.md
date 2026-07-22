# Deep speculative verify-ahead — throughput model (reify)

**Purpose:** quantitative substrate for the "deep speculative verify-ahead" PRD (G3
substrate check + G6 premise validity). Decouple *build-ahead depth* from *verify
concurrency* (K stays 2): build a stack of depth `d` (d+1 items I0..Id merged
cumulatively) and verify the whole stack in one run; on pass, in-order-CAS all d+1
items in one round; on failure, fall back to shallower (frozen-prefix, task 1890).

**Bottom line up front (G6 verdict): NO-GO / marginal for reify as a throughput
play.** The saturated-capacity gain of the best allocation is a real +14–17% (low-ε,
nested-deep), but reify's real merge queue is (a) *shallow* — P(an arrival sees ≥2
concurrently-mergeable items) ≈ 0.17, P(≥3) ≈ 0.095 — and (b) *demand-limited* — the
merge system is idle ~85% of wall-clock time and runs at ~15–35% of verify capacity.
Folding the real queue in, the expected steady-state throughput gain is **+8–16%**,
and because throughput in an undersaturated system is bounded by the task *arrival*
rate (not verify capacity), the realized landings/day gain is ≈0. The benefit that
does survive is a modest faster-drain of the ~17% of items that transiently queue —
a tail-latency win, not a throughput win. Recommend **not** building deep speculation
for reify now; revisit if reify's sustained arrival rate rises enough that the merge
queue becomes verify-bound.

---

## 1. Empirical calibration (reify runs.db, trailing 30 days, read-only)

### 1.1 p_good (Q) and reliability (R)

`scripts/analyze_speculation_depth.py … --since 30`:

| Quantity | Value | Source |
|---|---|---|
| p_good bracket [1−conflict_rate, land_rate] | **[0.970, 0.967]** → **Q ≈ 0.97** | merge_attempt terminal outcomes |
| per-attempt pass rate (all merge_verify) | 0.642 | 1018 events |
| 1/E[attempts-to-first-pass] among landed | **0.565** → **R ≈ 0.57** | attempts-to-first-pass histogram |
| flake rate (1−R) | **≈ 0.43** (high) | — |
| attempts-to-first-pass histogram | 1×:335, 2×:122, 3×:46, 4×:26, 5×:7, 6×:9, … | ~40% of landed items needed ≥2 attempts |

**Cross-check that pins R and Q:** the least-confounded depth bucket is depth=0
(single-item verify), whose observed pass rate is **0.543**. The model predicts
P(pass|d=0) = R·Q = 0.57·0.97 = **0.553** — a clean match. So R ≈ 0.57, Q ≈ 0.97 are
mutually consistent and well-calibrated.

**Why the per-depth empirical pass rates are NOT usable as P(pass|d):** the raw
buckets are depth=0:0.543, depth=1:**0.761**, depth=2:0.600 (n=5). depth=1 passing
*more* than depth=0 violates the R·Q^(d+1) model and is a **selection artifact** —
the variable-depth probe (probe_fraction=0.5) fires deep verifies only when a stack is
already built, which correlates with already-good items, and the depth=1 slot often
acts as the *confirming* verify of an item whose head already passed. Per the task
brief, **P(pass|d≥2) is therefore a MODEL prediction (R·Q^(d+1)), not measured** —
there is essentially no clean depth≥2 data yet (that is what the PRD's probe unlocks).

### 1.2 Verify duration T(d) and the depth-growth coefficient ε

merge_verify `duration_ms`, trailing 30d (n=1018):

| Bucket | n | median (s) | p90 (s) | mean (s) |
|---|---|---|---|---|
| all depths | 1018 | **1049** | **3249** | 1396 |
| depth=0 (1 item) | 186 | **939** | 3051 | 1351 |
| depth=1 (2 items) | 88 | **1540** | 6033 | 2105 |
| depth=2 (3 items) | 5 | 2782 | 6408 | 3613 |

- **T0 (base, depth-0 median) ≈ 940 s (~16 min).** p90 ≈ 3050 s — a heavy right tail
  (~3× median), driven by flake/CPU-contention/cold-cache. Well inside config bounds:
  `verify_command_timeout_secs: 7200` (warm) and the liveness reaper 0.75·10800 =
  8100 s; even p90 for depth-2 (6408 s) is under 7200 s but with little margin.
- **ε (verify-time growth per added stack level), T(d)=T0·(1+ε·d):**
  - from depth-1 median: (1540/939 − 1)/1 = **ε ≈ 0.64**
  - from depth-2 median: (2782/939 − 1)/2 = **ε ≈ 0.98**
  - mean-based (2105/1351): ε ≈ 0.56

  These are **much larger** than the task's nominal sweep {0, 0.05, 0.1, 0.2}. Two
  reasons and one caveat: (1) reify's verify is a workspace Rust compile — a deeper
  stack = more changed crates = more sccache misses + downstream recompilation; test
  *execution* is depth-independent but *compile* is not, and compile dominates.
  (2) **Confounded:** deep verifies are dispatched under backlog, i.e. under higher
  CPU contention, which independently slows them. So the *causal* ε is somewhere
  below the observed ~0.6. **I therefore report the full sweep ε∈{0.05,0.1,0.2,0.5}
  and treat ε≈0.2–0.5 as the empirically-plausible band, ε≤0.1 as optimistic.** ε is
  a first-order driver of the result (Section 5), so this uncertainty is load-bearing.

### 1.3 Queue-depth distribution — the binding constraint

Two independent estimators agree the reify merge queue is **shallow**.

**(a) Per-item ready depth** — `merge_queued.queue_depth` (qsize at each arrival,
n=1320; PASTA-unbiased for time-average since merge arrivals ≈ renewal):

| items available at arrival | count | P(≥k) | ⇒ can build depth d = k−1 |
|---|---|---|---|
| 1 | 1100 (83%) | 1.000 | d=0 only |
| 2 | 95 | **0.167** | d=1 |
| 3 | 47 | **0.095** | d=2 |
| 4 | 33 | 0.059 | d=3 |
| 5 | 19 | **0.034** | d=4 |
| ≥8 | 11 | **0.008** | d=7 |

mean = 1.40, median = 1, p90 = 2, max = 10.

**(b) Time-average in-system count** — sweep-line over reconstructed
[merge_queued → merge_finalized] intervals (1306 intervals, 2.58M wall-s):

| | value |
|---|---|
| frac time idle (0 in system) | **0.849** |
| P(in-system ≥ 2) | 0.111 |
| P(in-system ≥ 3) | 0.065 |
| P(in-system ≥ 5) | 0.018 |
| mean in-system (all time) | 0.40 |
| **mean in-system \| busy** | **2.65** |
| P(≥2 \| busy) | 0.733 |
| P(≥3 \| busy) | 0.427 |
| P(≥5 \| busy) | 0.118 |

**Interpretation:** ~85% of the time there is nothing (or one item) to merge — depth
is irrelevant. Only ~17% of items ever arrive to a queue where a depth≥1 stack is even
*possible*, and depth≥2 (≥3 concurrent items) is possible for <10% of items. **This
ceiling — not p_good, not flake — governs the result.**

---

## 2. The model

Notation: a slot verifying the cumulative stack [I0..Ix] (top index x, i.e. x+1 items)
**passes with probability R·Q^(x+1)** (one flake roll per run × all x+1 items good) and,
on passing, confirms the whole prefix I0..Ix as mergeable. In-order CAS lands the
maximal contiguous confirmed prefix per round. T(d)=T0·(1+ε·d), T0=940 s, Q=0.97, R=0.57.

### 2.a Single-slot per-time throughput and the idealized ceiling

- **Items per attempt** (ignoring time): g(d) = (d+1)·R·Q^(d+1). Idealized optimum
  d*+1 = −1/ln Q. At Q=0.97 → **d*_ideal ≈ 32** (higher than the Q=0.95→18 exemplar).
  This is the ceiling the task warned about; time and queue pull it far down.
- **Per-time throughput/attempt** f(d) = (d+1)·R·Q^(d+1)/T(d). Because R and T0 are
  constant multipliers, **argmax_d f(d) is independent of R** (proof: R factors out).
  Confirms flake is not a depth driver in this optimistic model.

Unconstrained single-slot d* (argmax of f):

| Q | ε=0 | 0.05 | 0.1 | 0.2 | 0.5 |
|---|---|---|---|---|---|
| 0.95 | 18 | 11 | 8 | 6 | 3 |
| **0.97** | **32** | **16** | **12** | **9** | **4** |
| 0.99 | 59 | 34 | 25 | 17 | 8 |

### 2.d Realistic single-slot with failure/retry (renewal-reward)

The per-attempt model above charges only one T(d) per attempt and ignores the cost of
a *failed* deep verify. Real policy: attempt depth d; on failure (prob 1−R·Q^(d+1))
the T(d) is wasted and we fall back to draining the items at depth 0 (frozen-prefix
graceful degradation). Charging wasted deep time + the depth-0 drain of the good prefix
(good item costs T0/R with flake retries), the **renewal-reward throughput
E[land]/E[time]** and its optimum:

Realistic single-slot d* (Q=0.97, R=0.57):

| ε | 0.05 | 0.1 | 0.2 | 0.5 |
|---|---|---|---|---|
| realistic d* | 6 | 6 | 5 | 4 |

The curve is **flat near the top**: at ε=0.1 the throughput is d=2→3.09, d=6→3.32
items/hr (only +7% from d=2 to the optimum), while d=0→2.13, d=1→2.80. **The bulk of
the achievable single-slot benefit is captured by d=1–2; beyond that returns are
marginal even before the queue ceiling.** High flake (R=0.57) is what pulls realistic
d* from ~12–32 down to ~4–6: a deep failure wastes the whole T(d), and with a 43%
flake rate deep stacks fail often.

### 2.b Two-slot allocation under in-order CAS

K=2 verify slots. A "round" dispatches both, waits max(T), then in-order-CAS lands the
maximal confirmed prefix. Three candidate allocations (slotA top index a, slotB top b):

- **(iii) status-quo pipeline — BASELINE:** a=0, b=1 (adjacent d0 + d1). This is
  today's `_speculation_depth=K=2` behaviour and the bar to beat.
- **(i) anchor+deep:** a=0 (re-verify head for guaranteed progress) + b=d (deep).
- **(ii) nested deep:** a and b>a both deep; slotA is a strict sub-stack of slotB, so
  it is the graceful fallback floor.

**Saturated throughput (infinite ready depth), items/hr:**

| allocation | ε=0.05 | 0.1 | 0.2 | 0.5 |
|---|---|---|---|---|
| (iii) BASELINE d0+d1 | 4.85 | 4.63 | 4.24 | 3.39 |
| (i) anchor d0+d2 | 6.36 | 5.83 | 5.00 | 3.50 |
| (i) anchor d0+d5 | 9.62 | 8.02 | 6.01 | 3.44 |
| (ii) nested d1+d3 | 8.49 | 7.51 | 6.10 | 3.91 |
| (ii) **nested d2+d5** | **11.24** | **9.37** | **7.03** | **4.01** |

Saturated % gain over baseline: nested d2+d5 gives **+132% / +102% / +66% / +18%**
across ε; anchor+deep is consistently ~2× weaker (+31…+41%) because it burns a slot on
a redundant shallow anchor whose information the deep verify already contains.
**Nested-deep dominates anchor+deep at every ε** — a clean design finding.

### 2.c Queue-depth ceiling folded in (reify's REAL queue)

Effective depth per round = min(target, available−1), available drawn from the
per-item ready distribution (1.3a). Throughput computed as a proper time-average
(ratio of expectations Σw·EL / Σw·dur, not an average of rates):

**Queue-folded throughput on reify's actual queue (items/hr):**

| allocation | ε=0.05 | 0.1 | 0.2 | 0.5 |
|---|---|---|---|---|
| (iii) BASELINE d0+d1 | 3.374 | 3.346 | 3.292 | 3.140 |
| (i) anchor d0+d2 | 3.536 | 3.491 | 3.404 | 3.168 |
| (i) anchor d0+d5 | 3.706 | 3.639 | 3.513 | 3.180 |
| (ii) nested d1+d3 | 3.786 | 3.727 | 3.615 | 3.315 |
| (ii) **nested d2+d5** | **3.950** | **3.879** | **3.744** | **3.390** |
| (ii) nested d2+d8 | 3.982 | 3.905 | 3.761 | 3.385 |

**Queue-folded % gain over baseline (the headline numbers):**

| allocation | ε=0.05 | 0.1 | ε=0.2 | ε=0.5 |
|---|---|---|---|---|
| (i) anchor d0+d2 | +4.8% | +4.3% | +3.4% | +0.9% |
| (i) anchor d0+d5 | +9.9% | +8.8% | +6.7% | +1.3% |
| (ii) nested d1+d3 | +12.2% | +11.4% | +9.8% | +5.6% |
| (ii) **nested d2+d5** | **+17.1%** | **+15.9%** | **+13.7%** | **+7.9%** |
| (ii) nested d2+d8 | +18.0% | +16.7% | +14.2% | +7.8% |

The **~10× collapse** from saturated (+102%) to queue-folded (+16%) at ε=0.1 is the
whole story: the shallow queue almost never offers the depth the optimum wants, so the
realized effective depth is ~1–2. Going from target d=5 to d=8 adds essentially
nothing (+16→+17%) because P(≥6 items)≈0.02.

---

## 3. Practical d* and expected gain for reify

- **Idealized single-slot ceiling:** d* ≈ 32 (Q=0.97) — irrelevant in practice.
- **Realistic single-slot (renewal, retry-charged):** d* ≈ 4–6.
- **Queue-constrained effective d* on reify:** **target d ≈ 2–3, realized d ≈ 1–2.**
  The controller would *reach* for d=2–3 but the queue supplies that depth <10% of the
  time; deeper targets (d≥5) buy <1% more.
- **Best allocation:** **nested-deep (slotA d≈2, slotB d≈5)**, not anchor+deep.
- **Expected throughput gain over the depth-1 pipeline baseline:**
  - **optimistic (ε≤0.1): +16–17%**
  - **empirically-plausible (ε≈0.2–0.5): +8–14%**
  - i.e. it **straddles the ~10–15% "worth it" line**, landing below it under the
    empirically-likely ε.
- **Demand-limited reality check (decisive):** reify's saturated baseline capacity is
  ~4.6 items/hr (deep queue) vs ~3.35 items/hr folded, but the queue is idle ~85% of
  the time and reify lands only **~28/day ≈ 1.17/hr → ~15–35% verify utilization.** In
  an undersaturated system the long-run landings/day is fixed by the *arrival* rate;
  extra verify capacity cannot manufacture more tasks. So the "+16% capacity" converts
  to **≈0% steady-state landings/day** and a modest reduction in queueing delay for the
  ~17% of items that transiently backlog.

---

## 4. Recommended allocation (if pursued)

**Nested-deep**, slotA at an intermediate depth (~d=2) and slotB deeper (~d=5),
both clamped to available−1. Rationale: (a) it dominates anchor+deep and status-quo at
every ε (Sec 2.b/2.c); (b) slotA being a strict sub-stack of slotB gives a graceful,
non-wasteful fallback floor (frozen-prefix, task 1890) — on a slotB failure whose bad
item is in the I3..I5 tail, slotA still lands 3; (c) it never re-verifies a strict
subset for zero information the way anchor+deep does. Do **not** target d>5: no queue
depth to fill it.

---

## 5. Sensitivity ranking (which parameters move d* and the gain)

Confirming the task's hypothesis, with one refinement:

1. **Queue depth — DOMINANT.** Sets the ceiling; the gain scales ~linearly with the
   fraction of arrivals that see a deep queue. Best-allocation gain (ε=0.1) by
   scenario: reify actual **+16.7%** → "1.5× busier" +23% → "half of arrivals see ≥3
   items" +51% → deep-saturated **+102%**. Break-even to clear +15% is roughly
   "half of arrivals see ≥3 concurrent items" territory at plausible ε — i.e. reify's
   P(≥3 at arrival) would need to roughly **triple** (0.095 → ~0.25–0.30).
2. **ε (verify-time growth with depth) — STRONG SECOND, and empirically large.** At
   fixed (reify) queue, best-allocation gain falls +16.7% (ε=0.1) → +11.9% (ε=0.3) →
   +7.9% (ε=0.5). Because reify's *observed* ε≈0.6 (confounded, causal likely 0.2–0.5),
   this is a first-order drag, not a nuisance — arguably more important than the brief
   anticipated. Every added stack level costs compile time on a Rust workspace.
3. **p_good (Q) — sets the unconstrained d* but is BLUNTED in practice.** Q moves
   d*_ideal from 18 (Q=0.95) to 59 (Q=0.99), but the queue ceiling caps realized depth
   at 1–2 regardless, so Q's practical leverage on reify is small. reify's Q=0.97 is
   already favorable; it is not the binding lever.
4. **R (flake) — uniform multiplier on absolute throughput; NOT a depth driver
   (confirmed), with a second-order caveat.** d* of the per-time model is provably
   R-independent (R factors out of argmax). Its real effects: (i) a uniform scaling of
   all throughputs; (ii) a *latency/variance* driver (p90 verify ≈ 3× median); (iii) a
   second-order *gain suppressor* via the renewal penalty — high flake (R=0.57) makes
   deep-stack failures waste a full T(d), which is what pulls the realistic d* from
   ~12–32 down to ~4–6 and shaves the gain. So the hypothesis holds with the refinement
   "flake doesn't move the ideal d* but it does erode the achievable gain through
   wasted deep-verify time."

**Ordering: queue-depth ≫ ε > p_good > flake** for both d* and the realized gain.

---

## 6. GO/NO-GO premise verdict (PRD G6)

**Verdict: NO-GO for reify as a throughput improvement; at best a marginal,
conditional latency play.**

Grounds:
1. **Demand-limited, not verify-limited.** Merge system idle ~85% of wall-time;
   ~15–35% verify utilization; ~28 landings/day is arrival-bound. Deep speculation adds
   drain *capacity*, which is not the binding constraint. Long-run landings/day gain ≈ 0.
2. **Shallow queue caps realized depth at 1–2.** P(arrival sees ≥3 items)=0.095;
   deeper targets buy <1%. The saturated +100% collapses to a queue-folded +8–16%.
3. **Empirically-plausible ε (≈0.2–0.5) puts the gain at +8–14%, below the ~10–15%
   bar.** Only the optimistic ε≤0.1 with the best (nested-deep) allocation clears +15%
   (+16–17%).
4. **High flake (R=0.57) makes deep failures expensive**, further eroding the gain and
   holding realistic d* to ~4–6 even before the queue ceiling.

Against a non-trivial engineering cost (decouple build-ahead from verify-K; deep-stack
construction + rebase; chain-invalidation fallback correctness; a live depth
controller; deeper worktree/disk pressure; larger p90 verify approaching the 7200 s
timeout at depth≥2), a single-digit-to-low-teens % *capacity* gain that yields ≈0
steady-state landings/day does not clear G6 for reify today.

**Break-even conditions (when to revisit — any of):**
- **Queue depth:** sustained arrival rate rises until the merge queue is verify-bound —
  concretely, busy-fraction ≫ 15% and **P(≥3 concurrent mergeable items) ≳ 0.25** (≈3×
  today). At "half of arrivals see ≥3", the gain is +30–50% and clearly worth it.
- **ε:** de-confounded verify-time growth confirmed ≤0.1 (would need warm-cache
  instrumentation isolating compile-delta cost from contention) **and** a moderately
  deeper queue — together they reach the +15% bar.
- **p_good is already past break-even (0.97);** it is not the lever to move.
- **Different objective:** if the PRD's success metric is *backlog-drain latency /
  worst-case merge wait* rather than throughput, deep speculation has a real (if modest)
  benefit on the ~17% of items that queue and the rare deep-tail (P(≥5)≈0.02) events —
  that is a distinct premise the PRD should state explicitly rather than claim a
  throughput win.

---

## 7. d* controller spec (if the premise is later met)

A production controller replaces the one-time validation *probe* (`speculation_probe`,
probe_fraction — a MODEL-VALIDATION instrument to harvest real P(pass|d≥2), set back to
0.0 once the curve is validated/refuted). It is **not** a permanent random probe.

**Inputs (all always-on, recoverable from ordinary events — no probing):**
| Input | Source (rolling window) |
|---|---|
| Q̂ (p_good) | `merge_attempt` terminal outcomes: 1 − conflict_rate (or land_rate), trailing N lands |
| R̂ (reliability) | `merge_verify`: 1/E[attempts-to-first-pass among landed], trailing window |
| ready_depth | live `queue.qsize()` + already-built in-flight stack height (`_verify_frontier_depth`) — no probing |
| ε̂ (time growth) | regression of `merge_verify.duration_ms` on `depth`, trailing window; fallback to a conservative fixed ε (≈0.3) until enough depth spread exists |
| T0 | median depth-0 `merge_verify.duration_ms`, trailing window |

**Computation (per 2nd-slot dispatch round; pure function of rolling counters — same
call site as `select_probe_depth`):**
```
reach   = argmax_d  (d+1)·Q̂^(d+1) / (1 + ε̂·d)      # renewal-corrected, d in 0..D_max
d_slotB = clamp(reach, 0, ready_depth − 1)             # queue ceiling almost always binds
d_slotA = clamp(round(reach/2), 0, ready_depth − 1)    # nested sub-stack fallback floor
```
On reify `ready_depth − 1` is the binding clamp the vast majority of rounds, so in
practice `d_slotB = ready_depth − 1`, `d_slotA ≈ (ready_depth−1)//2` — i.e. "stack as
deep as the queue currently allows, put slotA halfway for graceful fallback."

**Cadence:** every second-slot dispatch round (cheap; no I/O, no clock, no RNG —
mirrors the existing pure `select_probe_depth`).

**Fast fallback (suppress → depth-1):** reuse the existing thrash guard — if R̂ (rolling
per-attempt fail rate) ≥ `suppress_flake_rate` (currently 0.40; reify's ~0.36–0.43 fail
rate sits right at this boundary), suppress the deep placement and use adjacent depth-1
(byte-identical to today). Additionally, on any deep-stack **failure**, enter a short
cooldown (next K rounds at depth ≤1) to avoid re-wasting T(d) while the failing item is
still in the frontier. This makes the mechanism degrade to the status-quo pipeline
under exactly the conditions (high flake, thrash) where deep speculation loses.

---

## 8. Assumptions and where they are weak

- **P(pass|d≥2) is a model prediction, not measured** — R·Q^(d+1) with depth-independent
  R. No clean depth≥2 data exists (the probe just started; its buckets are
  selection-confounded). If real deep verifies flake *more* than R·Q^(d+1) predicts
  (e.g. larger diffs hit more flaky tests), every gain number here is an over-estimate.
  This is the single biggest model risk and the probe's reason to exist.
- **ε is estimated and confounded.** Observed ε≈0.6 mixes true compile-delta cost with
  under-load CPU contention. The causal ε (0.2–0.5 assumed) is unmeasured; a warm-cache,
  contention-isolated micro-benchmark of verify time vs synthetic stack depth would pin
  it and could shift the verdict a few points either way.
- **Queue-depth estimate** uses two estimators that agree, but both are trailing-30d and
  reify's regime shifts with config/throughput changes; the qsize sampler slightly
  *under*counts stackable depth (excludes the currently-verifying head), so the true
  achievable depth is marginally higher than modelled — a small conservative bias
  *against* the NO-GO, i.e. the real gain is if anything slightly higher but not enough
  to flip the verdict.
- **Renewal/fallback model** approximates post-failure recovery as a depth-0 drain of
  the good prefix with independent per-item goodness; real chain-invalidation re-verify
  ordering is more complex, but since the same engine scores baseline and deep, the
  **% gain (the headline) is robust** even where absolute items/hr is approximate.
- **Round model** assumes both slots redispatch in lockstep at max(T); real pipelining
  can overlap slightly better, modestly favoring the baseline pipeline — again a small
  bias that does not flip the verdict.
- **Not modelled (all raise the cost side, none modelled as benefit):** deeper stacks
  increase worktree/disk pressure (warm-lane budget), push p90 verify toward the 7200 s
  timeout at depth≥2 (depth-2 p90 already 6408 s), and add rebase/construction latency.
- **K=2 is fixed throughout** (hard constraint — 32-core box, no spare hosts). Nothing
  here raises K or adds runners.

---

## 9. Burst / peak / stability re-analysis (operator challenge)

The operator challenges the mean-based NO-GO: lived experience is that reify's queue
is often backed up (5–9 items), load is *spiky* (bursty unblocks), so the decisive
metric may be **peak load + time-to-clear a backlog**, not mean utilization. Tested
skeptically on the same 30d runs.db. **Result: the operator is right that arrivals are
bursty and backlogs are real, but the crux (in-burst stackability) fails and the
incident mode is CPU- not depth-driven — so the verdict holds, now for sharper
reasons.**

### 9.1 Arrivals ARE bursty (operator vindicated on the arrival process)

merge_queued arrivals, 30d (n=1322):

| metric | value | reading |
|---|---|---|
| inter-arrival median / mean | 871 / 1958 s | mean ≫ median = skewed |
| **CoV of inter-arrival** | **1.57** | >1 → bursty (Poisson=1.0) |
| **Fano factor, 1h / 3h / 6h windows** | **2.71 / 4.10 / 5.54** | ≫1 → clustered; *rising with window* = correlated at multiple timescales |
| frac inter-arrivals <300 s | 0.303 | heavy short-gap clumping |

The arrival process is **decisively bursty/clustered**, not Poisson.

**PASTA re-examination (my earlier assumption).** Because arrivals are correlated, the
arrival-average and time-average in-system distributions *do* diverge — arrivals see
fuller queues:

| k | P(arrival sees ≥k) | P(time-avg ≥k) |
|---|---|---|
| 2 | **0.151** | 0.110 |
| 3 | **0.097** | 0.064 |
| 5 | **0.041** | 0.018 |

**But this does not bias my earlier numbers:** the queue-folding in Sec 2.c already used
the *arrival-average* (`merge_queued.queue_depth` qsize = 0.167 for ≥2), which is the
correct per-item sampling and is ~equal to the reconstruction's arrival-sample (0.151).
So burstiness leaves the per-item throughput gain intact; it only adds the latency lens
below. (If anything the qsize sampler slightly *under*counts stackable depth, a small
conservative bias.)

### 9.2 Backlog episodes are real and moderately frequent

Segmenting the in-system reconstruction (episode = in-system ≥3 until it returns to ≤1):

| metric | value |
|---|---|
| **episodes / 30d** | **53** (~53/month) |
| time-to-clear: median / mean / p90 / max | **45 / 59 / 136 / 217 min** |
| peak-depth distribution | 3:19, 4:11, 5:10, 6:4, 7:2, 8:3, 10:2, 12:1, **16:1** |
| median / max peak | **4 / 16** |
| total time-in-backlog | **52 h = 7.3%** of 30d |
| inter-episode gap CoV; frac <1h | 2.04; **29% clustered** |
| episode-start hour-of-day | spread, mild peaks at 05–08 & 16 UTC — **weak** operator-window signal |

So the operator's "often backed up" is real but bounded: **~43% of episodes (23/53)
reach peak ≥5**; the median episode peaks at only 4 and clears in 45 min. Episodes are
somewhat clustered in time but not sharply tied to a single diurnal operator window.

**Modeled time-to-clear from peak M (nested-deep d2+d5 vs depth-1 baseline):**

| ε | M=3 | M=5 | M=8 | M=12 |
|---|---|---|---|---|
| 0.1 | +0% | **+43%** | +39% | +44% |
| 0.3 | +0% | **+33%** | +23% | +27% |
| 0.5 | +0% | **+25%** | +11% | +15% |

This is the burst-regime benefit the mean washed out: for the deeper episodes (peak ≥5,
43% of them), nested-deep would cut clear-time **~25–43%** *if items stack cleanly*.
M=3 shows +0% because nested clamps to the same shallow depth — and the median episode
peaks at 4, so most episodes see little benefit.

### 9.3 In-burst STACKABILITY — the crux — FAILS

Deep speculation only helps a burst if the co-resident items build a clean
non-conflicting frozen-prefix stack. Comparing merge outcomes **during backlog vs
outside** (129 vs 713 merge_attempts):

| signal | during backlog | outside | burst penalty |
|---|---|---|---|
| **genuine `conflict`** | **4.7%** (6) | 2.5% (18) | **~1.9× more** |
| **`gate_retry`** (re-verify churn) | **27.9%** (36) | 7.4% (53) | **~3.8× more** |
| `post_merge_equivalence_failed` | 1.6% | 0.6% | ~2.7× |
| clean `done` first pass | **52.7%** | 67.5% | −15 pp |
| `rebase_verify_cost` density | **1.55/hr** | 0.86/hr | **~1.8× more** |

Burst-time items are **markedly more entangled**: nearly 2× the genuine conflict rate,
~4× the gate-retry/re-verify churn, ~1.8× the rebase rate, and a 15-point-lower clean
first-land rate. This is consistent with the "a burst = one operator unblocking one
subsystem → overlapping files" mechanism. **So deep speculation degrades to shallow
(chain-invalidation, frozen-prefix bounce) exactly in the regime it is meant to help.**
A depth-5 stack needs 6 clean items; at the in-burst ~4.7%/item conflict rate,
P(6 clean) ≈ 0.75, and the ~4× gate-retry churn fragments it further — the realized
in-burst effective depth collapses toward 2–3, capturing perhaps **half** the modeled
25–43%, i.e. **~12–20%** clear-time reduction on the ~23 deep episodes/month.

### 9.4 Incidents are CPU-contention-flavored, NOT merge-depth-driven

merge_verify pass-rate & duration vs *concurrent in-system depth* at verify time:

| in-system depth | pass rate | median dur | p90 dur |
|---|---|---|---|
| ~0 (idle) | 0.629 | **854 s** | 3022 s |
| ~1 | 0.707 | **1649 s** | 3360 s |
| ~2 | 0.587 | 1499 s | 4287 s |
| ~3 | 0.641 | 1420 s | 3423 s |
| ~4 | 0.750 | 1769 s | 4670 s |
| ~5 | 0.744 | 1299 s | 4302 s |

Two findings:
1. **Pass rate does NOT decline with depth** (0.63→0.71→0.59→0.64→0.75→0.74, noisy-flat)
   and is *higher* during backlog (0.685) than outside (0.634). So R does **not** get
   worse in bursts — the gate_retry churn in 9.3 is rebase/gating fragmentation, not raw
   verify flake. **No burst penalty on R** (good for the model; my global R holds).
2. **The verify-time increase is a regime shift, not a depth ramp.** Duration ~doubles
   from idle (854 s) to *any* busy state (~1300–1770 s) and then stays roughly flat
   across depth 1→5. That ~2× jump is the **idle→busy CPU-contention** shift (concurrent
   task verifies on the shared 32-core box), not the marginal cost of stack depth. The
   *true marginal* ε (one more item on an already-busy verify, load held constant) is
   therefore small (~0.05–0.15), well below the ~0.6 the naive depth-0-vs-depth-1
   comparison suggested. **My Sec 1.2 ε was partly conflating regime-shift with
   per-depth cost** — corrected, the saturated deep-verify is cheaper than feared, but
   this does not move the demand-limited verdict.

Long / near-timeout verifies (the livelock signature): only **27** verifies >5400 s and
**5** >6800 s (near the 7200 s timeout) in 30d — rare. Per-unit-time they are ~3× denser
in backlog windows but tiny in absolute count; there is no main-frozen livelock in this
30d window (the 48→24 revert's oversubscription livelock was pre-mitigation). Task-side
contention proxies (`dispatch_deferred` ~8/hr, `external_dep_gate_held` ~6.5/hr) are
**flat** in vs out of backlog (0.9–1.0×) — merge backlogs are not specially driven by
task-dispatch backpressure.

**Conclusion (Q4):** the historical failure mode is **CPU oversubscription** (many
concurrent *task* verifies), already mitigated by admission control — **not** merge-queue
depth. Deep speculation makes each merge verify a **heavier, longer compile** competing
for the same 32 cores during exactly the busy windows where any residual contention
lives. So for stability it is **neutral-to-mildly-adverse**, not a mitigation.

### 9.5 Re-verdict under the burst/peak/stability objective

Reframing throughput → burst-drain-latency / peak-depth / livelock-mitigation **moves
the needle but does not flip the verdict — still NO-GO, for sharper reasons:**

| Condition for a conditional-GO on latency/stability | reify reality | met? |
|---|---|---|
| Bursts are **stackable** (low in-burst conflict) | conflict 2×, gate_retry ~4×, rebase 1.8× in-burst | **NO** |
| Backlogs frequent **and deep** enough | 53/mo but median peak 4; only 43% reach ≥5 | **partial** |
| Incidents **depth-driven** (deep spec clears them) | CPU-contention regime shift; near-timeouts rare & not depth-concentrated | **NO** |

- The burst benefit is **real but small after discounting**: ~23 deep episodes/month ×
  ~45–90 min × ~12–20% realized (post-stackability-discount) ≈ **a few hours/month** of
  aggregate merge-latency saved — against a substantial build (decouple build-ahead from
  K, deep-stack construction/rebase, chain-invalidation correctness, a live controller)
  **plus** a mild CPU-competition stability risk in the exact windows that matter.
- The two constraints that kill it are **in-burst non-stackability** (deep spec
  degrades to shallow precisely when wanted) and **CPU-driven, not depth-driven,
  incidents** (deep spec is neutral-to-worse). Neither is a mean-vs-peak artifact; both
  are properties of the burst regime itself.
- **What WOULD flip it:** a workload whose bursts are *disjoint* (independent tasks
  finishing together, low file overlap — the opposite of reify's one-subsystem-unblock
  pattern) **and** deeper (median peak ≥5–6) **and** a verify whose cost is dominated by
  fixed test-execution rather than changed-crate compile (so marginal ε≈0). reify is
  none of these today.

**Net:** the operator's peak/burst instinct correctly identifies where *any* benefit
lives, and the latency framing is the strongest case for the feature — but on reify's
actual burst structure that case still comes out **NO-GO / marginal**. The burst
analysis strengthens rather than overturns the recommendation, and sharpens the
break-even: deep speculation for reify would pay off only if in-burst conflict rate
fell below ~baseline (≈2–3%) *and* median episode peak rose above ~5–6 — i.e. a
materially different, more-disjoint, deeper workload than reify runs today.

---

## 10. Diff-only stackability test — does file-granular locking keep co-queued items disjoint?

**Operator's theory:** reify schedules under pessimistic file-granular module locking
(`lock_depth=10`, `max_per_module=1`) to avoid concurrent same-file edits, so co-queued
items should be *more* file/crate-disjoint than random → deep stacks structurally clean.
Settled diff-only (no builds): each task's landed file set recovered from its
`Merge task/<id> into main` commit (`git diff --name-only <sha>^1 <sha>`, read-only),
files mapped to crates via `crates/<name>/…`. 457/575 queued tasks' file sets recovered
(median 3 files / 2 crates per task); 54 backlog episodes, 156 distinct co-queued tasks.

### 10.1 Co-queued vs random pair overlap — locking helps at FILE level, not CRATE level

| pair set | n | P(share ≥1 file) | P(share ≥1 crate) | mean J_file | mean J_crate |
|---|---|---|---|---|---|
| **co-queued** (same episode) | 272 | **0.029** | **0.287** | 0.004 | 0.123 |
| random baseline (unconstrained) | 2000 | **0.047** | **0.311** | 0.005 | 0.123 |

- **File level: locking works.** Co-queued pairs share a file **0.62× as often** as
  random (0.029 vs 0.047). File-granular locking genuinely keeps concurrent items off
  the same file → low *rebase* (git) conflict, consistent with the low 4.7% genuine
  `conflict` rate in 9.3.
- **Crate level: locking is nearly inert.** Co-queued pairs share a *crate* **0.92× as
  often** as random (0.287 vs 0.311) — essentially chance. `max_per_module=1` at file
  granularity does **not** stop two tasks editing *different files in the same crate*.

### 10.2 The gap that matters: rebase-clean ≠ verify-clean

A frozen-prefix stack needs two different things to succeed:
- **rebase-clean** (no git conflict) → requires pairwise **file-disjoint**. Usually true.
- **verify-clean** (combined diff compiles + tests green) → requires semantic
  non-interference, conservatively **crate-disjoint** (two file-disjoint tasks in the
  same Rust crate can still break each other: A changes a signature in `a.rs`, B calls
  it from `b.rs` — no file conflict, but the merged crate fails to compile).

Max mutually-disjoint sub-stack available **per backlog episode** (exact max-independent-set
over the co-queued set; 51 episodes with recovered members):

| target depth d | items needed | **P(rebase-clean stack avail)** (file-disjoint) | **P(verify-safe stack avail)** (crate-disjoint) |
|---|---|---|---|
| 2 | 3 | **0.667** | **0.412** |
| 3 | 4 | 0.431 | 0.176 |
| 4 | 5 | 0.275 | 0.078 |
| 5 | 6 | 0.137 | **0.020** |

median max file-disjoint substack = **3 items (depth 2)**; median max crate-disjoint
substack = **2 items (depth 1)**.

**Reading:** a *rebase-clean* depth-2 stack is usually available in a backlog (67%), and
even depth-3 in 43% — the operator is right that git-level conflicts are rare. But a
*verify-safe* (crate-disjoint) stack collapses fast: depth-2 only 41%, depth-3 18%,
**depth-5 just 2%**. The typical backlog offers a rebase-clean depth-2 but only a
verify-safe depth-**1** stack.

### 10.3 Cross-crate dependency coupling (the nuance locking cannot close)

Crate dep graph parsed from `crates/*/Cargo.toml` (33 crates). **Caveat: recovered mean
dep-degree ≈ 1.0 is implausibly sparse** — workspace-inherited deps (`x = { workspace =
true }`) and rename/path forms are under-matched, so the adjacency is a **lower bound**
and the "dep-safe" figures below are **optimistic** (over-counting safe pairs):

- P(co-queued pair coupled = same-crate OR dep-adjacent) = 0.147 (lower bound; true value
  ≥ the 0.287 same-crate rate alone once the dep graph is complete).
- P(verify-safe depth-d stack avail | backlog), crate-disjoint **AND** dep-non-adjacent:
  d=2: 0.510, d=3: 0.275, d=4: 0.157, d=5: 0.118 — bracketed **below** by the pure
  crate-disjoint column in 10.2 (0.412 / 0.176 / 0.078 / 0.020), which is the reliable
  conservative floor.

### 10.4 Verdict — locking delivers rebase-cleanliness, NOT verify-safety

**The locking theory is half-right, and the half it misses is the decisive one.**
File-granular locking (`max_per_module=1`) genuinely makes co-queued items file-disjoint
→ mechanically rebase-clean stacks of depth 2–3 are usually available (67% / 43% of
backlogs). But reify's locking is **file-granular while its verify cost and semantic
coupling are crate-granular**: ~29% of co-queued pairs still share a crate (≈ random),
so the *combined verify* — which is what actually gates a deep-stack land — sees
correlated failures that `Q^(d+1)` (independent-goodness) over-predicts. A **verify-safe**
(crate-disjoint) stack of depth ≥2 exists in only **41%** of backlogs and depth ≥3 in
**<18%**; the median backlog supports a verify-safe stack of only **depth 1**.

**P(clean depth-d stack available | in a backlog), headline:**

| | d=2 | d=3 | d=4 | d=5 |
|---|---|---|---|---|
| rebase-clean (git won't conflict) | 0.67 | 0.43 | 0.28 | 0.14 |
| **verify-safe (crate-disjoint — the binding one)** | **0.41** | **0.18** | **0.08** | **0.02** |

**Consequences for the verdict (reinforces NO-GO, sharper mechanism):**
1. This *confirms* §9's point-3 premise from the structural side: burst items still
   couple — not via git conflicts (rare, 3%) but via **crate-level verify coupling** that
   file-granular locking doesn't touch. The Q^(d+1) deep-stack pass model is **optimistic
   in-burst** for any adjacent (non-curated) stack of depth ≥2.
2. With today's *adjacent-item* frozen-prefix mechanism (stacks whatever is next in
   submission order, not a curated disjoint subset), a backlog supports a verify-safe
   deep stack only ~40% of the time at depth 2 and almost never at depth ≥3 — so realized
   deep depth in-burst is ~1–2, exactly the shallow regime the baseline already covers.
3. **The one lever that could rescue it** (and the sharpest positive finding): a
   **crate-disjoint stack-selection controller** that *reorders* the frozen-prefix to pick
   a mutually-crate-disjoint subset could lift verify-safe availability from the adjacent
   ~depth-1 toward the best-subset depth-2 (41%) / depth-3 (18%). But (a) it tops out at
   depth 2–3 for most backlogs regardless, (b) reordering under in-order CAS adds real
   complexity and changes land order, and (c) it still leaves depth ≥3 rare. Net it would
   turn a +marginal feature into a slightly-less-marginal one, not into a clear GO.

**Bottom line:** file-granular locking makes co-queued items *rebase*-clean but not
*verify*-clean, because the lock granularity (file) is finer than the coupling/verify
granularity (crate). Deep stacks are structurally clean enough to *build* (rebase) at
depth 2–3 but not clean enough to reliably *pass verify* beyond depth 1–2 in a backlog.
The premise "deep stacks are structurally clean → high deep-stack pass rate" is **not
supported** at the depth (≥2) where deep speculation would earn its keep — **NO-GO
holds.** *(This §10 conclusion rests on the conservative assumption same-crate ⇒
verify-conflict; §11 measures the true rate from history and substantially revises it.)*

---

## 11. Inference test — the TRUE crate-level semantic-conflict rate (history, no re-runs)

§10 used crate-disjointness as a **conservative proxy** (assume same-crate ⇒
verify-conflict). That is pessimistic: two tasks in different files of the same crate are
often independent. **Key insight (operator):** the shallow pipeline already verifies each
item against its landed predecessors — its merge verify tests the cumulative tree
`main + predecessors + item`. For a **crate-overlapping** stacked item, that verify **is
the deep-verify-equivalent, already executed**. So the true crate-level conflict rate is
readable from the record. Measured entirely from events + git; nothing rebuilt.

**Method.** Per backlog episode, reconstruct landing order (by `merge_finalized
state=done` time). An item is **crate-overlap-stacked** if ≥1 earlier-landed same-episode
item shares a crate with it (its cumulative verify tree contained a crate-sharing
predecessor); **crate-disjoint-stacked** otherwise. Classify each stacked item's merge
history via the system's own outcome vocabulary — the flake/genuine discriminator:
- **clean** = passed first try;
- **flake** = failed then landed with only `gate_retry` (system's transient re-verify
  signal) and **no** genuine outcome (secondary check: same-`merge_sha` fail→pass = proven
  identical-tree flake);
- **genuine** (real semantic conflict surfaced by stacking) = any
  `conflict` / `verify_failed` / `post_merge_equivalence_failed`.

### 11.1 Result — crate overlap adds ~ZERO semantic conflict

| stacked-item class | n | clean first-try | flake | **genuine conflict** |
|---|---|---|---|---|
| **crate-OVERLAP-stacked** | **36** | 0.083 | 0.806 | **0.111** (4/36) |
| **crate-DISJOINT-stacked** | **166** | 0.187 | 0.699 | **0.114** (19/166) |

- **P(genuine conflict \| crate-overlap-stacked) = 0.111 vs P(\| crate-disjoint-stacked)
  = 0.114 — ratio 0.97×, i.e. indistinguishable.**
- The **crate-disjoint group is a clean control**: those items share nothing with their
  stack-mates, so they *cannot* have a stacking-induced conflict — yet they show the
  **same** ~11% genuine rate. Therefore the ~11% is reify's **ambient** merge-trouble
  rate (contention verify-fails, own bugs surfacing, rebase churn), **not** crate
  coupling. The stacking-induced (crate-overlap) excess is **−0.3 pp ≈ 0**.
- **Flake dominates** both groups (70–81%), confirming high ambient flake (R≈0.57) is the
  re-verify driver, not coupling. (0 proven same-`merge_sha` flakes: main advances between
  retries so each retry is a new tree; the flake call rests on the system's
  `gate_retry`/no-genuine discriminator.)

**This REFUTES the §10 pessimistic proxy.** Same-crate is **not** ≈ verify-conflict; it is
essentially benign. **Caveat — underpowered:** the overlap subset is n=36 (4 genuine), 95%
CI ≈ [3%, 26%]; I cannot resolve a small excess. But the finding is *directional and
control-anchored*: were crate coupling a real deep-verify killer (say ≥40%), it would show
even at n=36, and it does not — the overlap rate sits **at or below** the disjoint control.
Disjoint-fast-path items (whose speculative verify never tested the cumulative tree) are
excluded by construction — the final *passing* verify that reaches `done` is against real
main and does contain the predecessors; non-contiguous interleaving only makes the tested
tree a **superset** (conservative — biases toward *finding* conflict, not hiding it).

### 11.2 Upgraded P(clean depth-d stack available | backlog)

Since crate-overlap is benign, the operative "verify-safe" availability is **not** the
crate-disjoint floor — it rises to ≈ the **file-disjoint (rebase-clean)** curve (the only
real remaining constraint is pairwise file conflict, ~2.9%/pair):

| d | §10 pessimistic (crate-disjoint) | **UPGRADED (measured — ≈ file-disjoint)** |
|---|---|---|
| 2 | 0.41 | **0.67** |
| 3 | 0.18 | **0.43** |
| 4 | 0.08 | **0.28** |
| 5 | 0.02 | **0.14** |

The true curve lies between these, near the upgraded end given the 0.97× ratio. **Clean
deep-2 availability roughly doubles (0.41→0.67) and deep-3 more than doubles
(0.18→0.43).** The §9.5 "burst benefit halved by stackability" discount is largely
**removed** — deep stacks in bursts are mostly clean, so the modeled 25–43% time-to-clear
reduction for the ~23 deep episodes/month is now realizable with only a minor discount
(realized by simply deepening the **adjacent** submission-order stack — **no** disjoint selection needed; the §10.4 selector lever is retracted in §11.5).

### 11.3 Bonus — real historical backlog CPU: shallow (measured) vs deep-drain (modeled)

Per episode: actual shallow CPU = measured Σ`merge_verify.duration_ms`; deep-drain =
`ceil(M/(d+1))/R` cumulative-tree verifies (retry-inflated), each `T0·(1+ε·d)`. Aggregated
over 29 episodes (47.2 measured merge-verify CPU-hours):

| marginal ε | deep depth 2 | depth 3 | depth 5 |
|---|---|---|---|
| 0.1 (Q4-implied) | **−59%** | −58% | −58% |
| 0.3 | −46% | −39% | −30% |
| 0.5 | −32% | −20% | −1% |

At the Q4-implied marginal ε≈0.1 (the idle→busy jump is regime-shift, not depth), deep
would drain historical backlogs with **~55–59% less total merge-verify CPU** by amortizing
the fixed workspace-compile overhead across more items per verify; the saving degrades to
~0 only at high ε and deep depth. **Caveats:** the shallow baseline is *measured actual*
(includes real churn/blocks) while deep is *modeled* (idealized 1/R retries), so this
**flatters deep** — treat as an upper bound; and it is *integrated* CPU, not *peak* — a
deeper merge verify runs **longer** (bigger diff) at the same capped intensity
(CARGO_MAKEFLAGS merge pool + `cpu.weight=300`), so it occupies the merge lane longer even
while using less total CPU.

**This tempers §9.4's "neutral-to-worse for contention" to ε-dependent:** at low marginal
ε deep is actually **CPU-positive** (fewer, amortized verifies free the 32-core box for
task verifies), neutral-to-negative only at high ε.

### 11.4 Re-verdict after the inference test

Two of my earlier objections are substantially weakened by the record:
1. **Stackability crux (§10) — largely refuted.** Crate overlap is benign (11.1% vs
   11.4% control); verify-safe clean-depth availability upgrades ~0.41→0.67 (d=2),
   0.18→0.43 (d=3). Deep stacks in bursts are mostly clean.
2. **CPU-contention worry (§9.4) — softened.** At plausible marginal ε deep drains
   backlogs with ~30–59% **less** total merge-verify CPU, not more.

But the two **dominant** facts are untouched by this analysis and still cap the upside:
- **Demand-limited:** the merge system is idle ~85% of the time; long-run landings/day is
  bounded by the *arrival* rate, so steady-state **throughput gain ≈ 0** regardless of how
  clean deep stacks are. Deep speculation remains a **latency/CPU** play, never a
  throughput one.
- **Shallow, bounded bursts:** median episode peaks at 4; only 43% reach ≥5; realized deep
  depth stays ~2–3. Cleaner stacks don't create deeper backlogs.

**Verdict split by objective:**
- **Throughput (original G6 premise): NO-GO — firmly, unchanged.** Nothing here creates
  more daily landings in a demand-limited system.
- **Burst-drain latency + backlog CPU: upgraded from "NO-GO/marginal" to a defensible
  CONDITIONAL GO** — *if* the success metric is explicitly reframed to worst-case merge
  latency / backlog-drain time / backlog CPU (not throughput), *and* the mechanism is a **simple adjacent-stack deepening**
  (decouple `_merge_ahead_cap` from verify-K; verify the built frontier tip; truncate at
  the rare ~2.9% file conflict). **No stack-selection / reordering controller is
  required** — §11.1 refuted the crate-disjoint premise it rested on, and adjacent
  stacking additionally *preserves* the in-order-CAS invariant reordering would fight
  (see §11.5). Bounded, honest payoff: ~25–40%
  faster drain on ~23 deep episodes/month (≈ a few hours/month of merge latency) **plus**
  ~30–55% less backlog merge-verify CPU at plausible ε. The two structural caps
  (demand-limited, shallow peaks) mean this is a **quality-of-service / CPU-efficiency**
  improvement for the ~7% of time in backlog, not a throughput win — worth building only
  if that QoS+CPU objective is what the PRD commits to, and sized to the queue — realized
  depth ~2–3 in a typical burst (median peak 4), deeper in the rare deep episodes; bounded
  by queue depth, not by a disjoint-subset size.

**Net across all four analyses:** the throughput premise is NO-GO; the burst-latency/CPU
premise is a bounded CONDITIONAL GO that the inference test made materially stronger by
removing the stackability and CPU-contention objections — the remaining limiter is simply
that reify's queue is shallow and demand-limited, which caps the prize at "modest but
real QoS/CPU gains during backlogs," not "more throughput."

### 11.5 Correction (2026-07-22): stack-selection / reordering is NOT required

An earlier draft of this re-verdict (and the §10.4 lever) recommended a **crate-disjoint
stack-selection controller** — reorder the frozen prefix to pick a mutually-disjoint
subset. That was an unreconciled holdover from §10's *pessimistic proxy* (same-crate ⇒
verify-conflict) and does **not** survive §11.1:

- **§11.1 refuted the proxy.** Crate-overlapping stacked items conflict at 11.1% —
  identical to the crate-disjoint control (11.4%). Crate coupling adds ~0 conflict, so a
  crate-coupled *adjacent* stack passes at the same rate as a hand-picked disjoint one.
  There is no coupling to select around; the conflict rate is **ambient and per-item**,
  which reordering cannot change.
- **The only remaining constraint is file (git) conflict**, ~2.9%/pair — rare, and it
  needs no selector: the adjacent-stack build simply **truncates** at the first file
  conflict and verifies the clean prefix.
- **In-order CAS makes adjacent stacking the natural design.** Building [I0..Id] in
  submission order and CAS-landing all d+1 in order preserves the invariant; a selector
  that lands a disjoint *subset* out of order (or holds skipped items back) **fights** that
  invariant and adds scheduling complexity — it is the costly option, not the enabling one.

**Corrected mechanism:** decouple `_merge_ahead_cap` from verify-K and verify the built
adjacent frontier tip (truncating at a file conflict) — close to "raise the merge-ahead
bound and verify the tip," not a new scheduler. A selector remains only a *possible future
hedge* if the n=36 benign finding later proves to hide a moderate crate penalty — not a
build requirement. This makes the conditional-GO both **cheaper to build** and, being
queue-bound rather than disjoint-subset-bound, slightly **deeper-reaching**; it does not
change the throughput NO-GO or the demand-limited cap.
