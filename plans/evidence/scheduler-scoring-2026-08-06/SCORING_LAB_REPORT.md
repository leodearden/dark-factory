# Dispatch-scorer counterfactual lab — dark-factory scheduler

**Script:** `/tmp/claude-1000/-home-leo-src-dark-factory/6aab7100-da40-4bdb-8e30-34db0d7abac5/scratchpad/scoring_lab.py`
**Raw output:** `scoring_lab_output.txt` · **CSVs:** `top40_S{0,1,2,3,4,5}.csv`, `all_candidates_all_scorers.csv` · **JSON:** `scoring_lab_live.json`
**Snapshot:** `tasks_snapshot.db` @ `2026-08-06T14:48:47Z` (3759 tasks, 1444 dep edges; 537 active)
**Read-only:** nothing under `/home/leo/src/dark-factory` was modified; the repo was opened only to read `scheduler.py` / `config.py` / `task_statuses.py`.

## Which snapshot the numbers come from

**3534 is `pending` in the live snapshot, with zero dependencies and zero
dependency-gate failures — it is a live, dispatch-ready candidate that the scorer
simply ranks 32nd.** No counterfactual restoration was required, and every number
below comes from the as-captured live snapshot. (Cross-check: `runs_snapshot.db`
holds **no events at all** for task 3534 — it has never been dispatched, so it also
cannot be carrying a resurrection age-anchor.)

## Replication fidelity

Ported verbatim from `orchestrator/src/orchestrator/scheduler.py`: `_build_reverse_index`
(:4581), `_compute_effective_priorities` (:4598, incl. the `boost_tier` overlay from
`scheduler_state_snapshot.json` → 3536:critical), `_compute_transitive_counts` (:4662),
`_compute_age` (:4695), `_compute_score` (:4739), `_deps_satisfied` (:3848),
`_milestone_time_gated` (:4212), `_eligible_for_dispatch` (:4392), and the
`(-score, task_id_string)` sort of `_phase_select_scored` (:6327). Constants from
`config.py`: `TIER_BASE {critical 16000, high 8000, medium 4000, low 2000, polish 1000}`,
`TIER_WIDTH 1000`, `age_alpha 10.0`, `cpm_beta 100.0` (neither is overridden in
`orchestrator_config_snapshot.yaml`); `max_concurrent_tasks: 24`.

Four things worth pinning down, because they change the answer:

1. **`ctx.tasks` is the ACTIVE-ONLY fetch** (`get_tasks(statuses=ACTIVE_TASK_STATUSES)`,
   :6476; `ACTIVE = frozenset(TaskStatus) - {done, cancelled}`). So the reverse-dependency
   graph — and therefore **D** — is built over active tasks only, and `max_id` is the max
   **active** id (3796 here, which coincides with the global max).
2. **The age anchor really is the task's own id, and this is verified, not assumed.**
   `self._pending_anchor` is initialised empty in `__init__` (:1508) and is **never
   persisted** — the state snapshot carries no anchors. Every orchestrator restart
   re-anchors all pending tasks to their own id (`_update_age_anchors`, :4707). Only a
   task that went non-pending→pending *within the current process lifetime* carries an
   `age=0` resurrection anchor. Bounded from the events DB: 34/291 candidates (11.7%) were
   dispatched since the last recorded orchestrator `service_restart`; forcing those to
   `age=0` moves 3534 from rank 32→30 (S0) and 29→27 (S1/S4). Directionally favourable to
   3534 and immaterial to every conclusion.
3. **L is defined in edges** over the same active reverse graph: `L(t)=0` when `t` has no
   non-terminal dependents; terminal nodes are walked *through* without contributing
   length (mirroring `_compute_transitive_counts`). Cycle-safe tri-state DFS.
4. **Gates I could not replicate offline**, all noted as caveats: the in-memory
   `_requeue_until` cooldown, `_dispatch_cooldown_active` (`_last_dispatch_at`), the
   landed-outbox / already-landed gates (need git + outbox), the delivered-check dep cache
   (needs a live grep of `main` — **failed OPEN**; it turns out to bind on **0** candidates
   in this snapshot), and the PSI admission hold (a throttle, not an ordering).
   Module-lock contention is also out of scope: "top-24" means the top 24 of the ranking,
   not 24 successful `try_acquire`s.

**Independent corroboration:** a separately-produced `candidates_scored.csv` present in the
scratchpad reproduces my S0 scores and ranks **exactly** (float formatting aside) and its
top-24 is byte-identical to mine. Our candidate sets differ by 3 tasks — I additionally
excluded `2915`/`3677` (unresolvable cross-project `external_deps`) and `3779` (already
holding module locks, i.e. in `self._dispatched`). None of the three is anywhere near the cut.

## Candidate set

291 candidates out of 444 pending. Exclusions: 142 deps-unsatisfied, 8 milestone-gated
(all `mode: delayed`; 2664/2665/2666 have anchors stamped but their 30/90/180-day fuses have
not blown), 2 external-dep-unresolvable, 1 already-dispatched.

| effective tier | n | oldest (age) | max D | max L |
|---|---|---|---|---|
| critical | 1 | 178 (3618) | 5 | 3 |
| high | 55 | 720 (3076) | 31 | 7 |
| medium | 88 | **1064** (2732) | 7 | 6 |
| low | 147 | 904 (2892) | 1 | 1 |

Only **56 candidates sit at critical+high for 24 slots**, so under *every* scorer tested
the top-24 is `{1 critical, 23 high}` and no medium/low task is reachable by score at all.

## The baseline is broken in a specific, measurable way

**249/291 candidates (85.6%) saturate the `min(α·age + β·log1p(D), 999)` cap.** With α=10
the cap is reached at age ≈ 100, and 84.9% of candidates are already past that (median age
**355**, mean 388) before the CPM term contributes a single point. The
consequence is total: 291 candidates collapse onto **46 distinct scores**, with a **117-way
tie** at 2999.0. All 24 tasks in the S0 top-24 except the single critical one are tied at
exactly **8999.0**, so the *entire* live dispatch order inside the `high` tier is decided by
the `task_id` string tie-break. Every candidate id is 4 digits, so string order == numeric
order: **S0 is, in practice, pure FIFO-by-id, and both the age term and the CPM term are dead.**

The cost is visible in the S0 top-24 itself: the two largest fan-outs in the whole candidate
set, **3197 (D=31, L=5)** and **3132 (D=26, L=4)**, sit at ranks 15 and 8 — not because they
unlock 31 and 26 tasks, but because their ids happen to be low. **Twelve of the 24 slots
(exactly half) are held by `D=0, L=0` leaves that unlock nothing.** And **3276 (D=10, L=7)** — the single
deepest-and-widest candidate in the set — misses the cut at **rank 25, by one slot**.

## Comparison table

| | 3534 rank /291 | in top-24? | Kendall τ-b vs S0 | Spearman ρ | top-24 churn | starvation exposure (all tiers / within critical+high) | tier inversions |
|---|---|---|---|---|---|---|---|
| **S0** CURRENT (α=10, β=100, D) | **32** | no | — | — | — | 1064 / **520** | **0** |
| **S1** RETUNE-α (α=1, β=100, D) | **29** | no | +0.9816 | +0.9986 | 4/24 (17%) | 1064 / 560 | **0** |
| **S2** NORMALIZED-WEIGHTED | **13** | **YES** | +0.9589 | +0.9938 | 8/24 (33%) | 1064 / 625 | **0** |
| **S3** LEXICOGRAPHIC-CPM | **11** | **YES** | +0.9265 | +0.9799 | 12/24 (50%) | 1064 / **706** | **0** |
| **S4** PATH-LENGTH (α=1, β=100, L) | **29** | no | +0.9832 | +0.9988 | 3/24 (12%) | 1064 / 545 | **0** |
| **S5** RESERVATION (R=4 by L) | slot **3** (reserved) | **YES** | +0.9962 | — | 2/24 (8%) | 1064 / 525 | 0 as set-membership, **1 as a head prefix** |

The `1064` in every "all tiers" column is the same task — **2732, `medium`, age 1064, D=0**.
It is excluded by *tier*, not by scorer, and no scorer tested can reach it. **The
within-tier number is the only one that discriminates**, and on that metric S0 is the *best*
and S3 the worst — the scorers that surface 3534 all do so by letting older high-tier work
wait longer.

## Per-scorer detail

### S1 — RETUNE-ALPHA (α=1.0, β=100, cap 999)

- **3534: rank 32 → 29. Still outside the top 24.**
- Losers (4): 3236 (high, age 560, D=0), 3251 (545, D=0), 3260 (536, D=0), 3271 (525, D=0).
  Winners (4): 3276 (520, D=10, L=7), 3315 (481, D=6), 3523 (273, D=24, L=3), 3319 (477, D=2).
- τ-b **+0.9816**, ρ +0.9986. Starvation exposure within critical+high: **560** (up from 520).
- Sanity check on the losers: all four are `high`-tier, D=0/L=0 leaf bug-fixes aged 525–560
  ("Steward re-escalations are silently swallowed", "RunnerUnavailable redispatch strands its
  old `_merge-` worktree…"). They unlock nothing, but they are real, aged, high-tier work.
  Displacing them is a genuine cost, not a free win.
- **Assessment.** S1 is the smallest possible change and it un-saturates the formula almost
  completely (1/291 saturated vs 249/291), which is worth having on its own: it restores a
  *total* order in place of a 117-way FIFO tie. But it does not actually rebalance the two
  terms. With α=1 the age term still spans **0..1064** while the CPM term spans **0..347** —
  age is still ~3× the dynamic range of CPM, so S1 remains an age-dominant scorer wearing a
  CPM decoration. That is exactly why 3534 doesn't move: its D=9 buys +230 while the
  incumbent 3226's age alone buys +570. A parameter sweep pins the tipping point: 3534 needs
  **α ≤ 0.5** (a 20× cut, not 10×) or **β ≥ 200** to clear the cut. S1's other virtue is that
  its score is a pure function of the task's own attributes, so ordering is perfectly stable
  as the candidate set churns (τ = 1.0000 in all three perturbation scenarios).

### S2 — NORMALIZED-WEIGHTED (`floor(999 · (0.5·age_rank + 0.5·cpm_norm))`)

- **3534: rank 32 → 13. Enters the top 24.**
- Losers (8): 3171 (625), 3172 (624), 3173 (623), 3226 (570), 3236 (560), 3251 (545),
  3260 (536), 3271 (525) — every one `high`, `D=0`, `L=0`.
  Winners (8): 3523 (D=24), 3276 (D=10, L=7), 3623 (D=23), 3624 (D=23), 3315 (D=6),
  **3534 (D=9, L=7)**, 3319 (D=2), 3666 (D=10, L=6).
- τ-b **+0.9589**, ρ +0.9938. Starvation exposure within critical+high: **625**.
- **No tier crossing, by construction:** `age_rank ≤ (n−1)/n < 1` (fraction of candidates with
  *strictly* smaller age) and `cpm_norm ≤ 1`, so `0.5·age_rank + 0.5·cpm_norm < 1` strictly and
  `floor(999·x) ≤ 998 < TIER_WIDTH`. Verified: 0 inversions over the full 291-long order.
- **Assessment.** This is the design that actually fixes the diagnosed disease. The disease is
  a *scale* mismatch — two terms with incommensurable units summed and then clipped — and
  percentile-ranking both onto [0,1) is the textbook cure. It cannot saturate (0/291), it
  cannot cross tiers, and it gives a genuinely graded 999-wide order inside each tier. The
  price is that a task's score is no longer a function of the task: it depends on the whole
  candidate set. I measured that rather than asserting it — perturbing the set three ways
  (dispatch the top 24 / 30 new tasks arrive / the 40 oldest complete) moves surviving scores
  by up to **68 points** with τ ≥ 0.9949. Small in aggregate, but 68 is *larger than the
  78-point gap* that kept 3534 out under S1, so ranks near the 24-slot boundary really can
  flip on someone else's completion. Second, the percentile transform destroys magnitude: the
  gap between D=31 and D=24 is compressed to the same "one rank" as D=1 vs D=0, so the scorer
  becomes blind to how big a fan-out actually is. Third, the fixed 0.5/0.5 split is an
  unargued prior — it happens to work here but nothing in the data justifies exactly half.

### S3 — LEXICOGRAPHIC-CPM (`tier ↓, floor(log2(1+D)) ↓, age ↓, id ↑`)

- **3534: rank 32 → 11. Enters the top 24.**
- Losers (12) — **half the board turns over**: 3090 (age **706**), 3103 (693), 3121 (675),
  3122 (674), 3171 (625), 3172 (624), 3173 (623), 3226 (570), 3236 (560), 3251 (545),
  3260 (536), 3271 (525). All `high`, all D=0.
  Winners (12): 3523, 3623, 3624, 3276, **3534**, 3666, 3688, 3315, 3658, 3319, 3543,
  **3585 (D=1, L=1, age 211)**.
- τ-b **+0.9265**, ρ +0.9799. Starvation exposure within critical+high: **706** — the worst
  of any scorer, and 186 points worse than S0.
- **Assessment.** S3 buys the strongest CPM signal and pays the highest price for it. The
  bucketing is the tell: `floor(log2(1+D))` puts D=1 in bucket 1 and D=0 in bucket 0, so
  **3585 — one dependent, one level deep, age 211 — displaces 3090, a 706-age high-tier
  task**, purely for having a single dependent. That is not "CPM dominates"; that is a
  boolean has-any-dependent flag deciding a 495-age difference. Age is demoted to a
  *within-bucket* tie-break, so it can never overcome even a one-bucket CPM gap, and a task
  with D=0 in a tier where anything has D≥1 is starved without bound — the aging mechanism is
  structurally defeated, which is the same class of failure as the current saturation bug, just
  with the polarity flipped. It is also the only design here whose behaviour is dominated by an
  arbitrary bucket-boundary choice: shifting to `floor(log2(2+D))` or `floor(log(1+D))` reshuffles
  the top-24 with no principled reason to prefer either.

### S4 — PATH-LENGTH (α=1.0, β=100, **L**)

- **3534: rank 32 → 29. Still outside the top 24.** Identical rank to S1.
- Losers (3): 3251 (545), 3260 (536), 3271 (525). Winners (3): 3276 (L=7), 3315 (L=3),
  3319 (L=2). Smallest disturbance of any scorer.
- τ-b **+0.9832** (highest), ρ +0.9988. Starvation exposure within critical+high: **545**.
- **The structural finding that sinks S4 as specified:** `L ≤ D` holds for **every** candidate
  (0 violations out of 291) — necessarily so, since a chain of length L exhibits L distinct
  non-terminal descendants. Therefore `log1p(L) ≤ log1p(D)` and **S4's score is ≤ S1's score for
  every task, always.** S4 is not a re-weighting *toward* depth; it is a uniform **shrink** of the
  CPM term (max 208 vs max 347), which mechanically hands *more* relative weight to age. Its
  only differential effect is to penalise wide-shallow tasks harder than deep ones — real, but
  second-order. Concretely for the task of interest it is actively counterproductive: 3534 has
  D=9 but L=7, so swapping D for L *lowers* its bonus from 230 to 208. **S4 cannot promote 3534,
  by construction.** Making a depth-first scorer work would require a much larger β (or a
  linear rather than log transform of L) to restore the lost dynamic range — a change the spec
  didn't include.
- **Assessment.** The *intuition* behind S4 is validated by the divergence data below (D really
  does mis-rank deep chains), but the *implementation* is self-defeating: applying the same
  concave transform to a strictly smaller quantity guarantees a weaker signal. It is the best
  scorer here on τ and on within-tier starvation — but only because it is the closest to
  doing nothing.

### S5 — RESERVATION (S0 ordering, R=4 of 24 slots reserved for top-4 by L)

- **Reserved: `3276` (L=7, D=10), `3184` (L=7, D=9), `3534` (L=7, D=9), `3212` (L=6, D=10).**
  **Yes — 3534 wins a reservation slot** (the L=7 three-way tie breaks by D then age: 3276 D=10
  first, then 3184 and 3534 both D=9, 3184 older).
- 3184 and 3212 were *already* in the S0 top-24 (ranks 14 and 16), so the reservation
  effectively costs only **two** slots: **3271** (age 525, D=0) and **3275** (age 521, D=1)
  drop out, and **3276** (rank 25) and **3534** (rank 32) come in. Smallest churn of any
  intervention (2/24), τ +0.9962. Within-tier starvation exposure 525 — five points worse
  than S0.
- **Tier-invariant caveat, and it is a real one.** As pure *set membership* over a full
  24-slot refill, S5 preserves the tier composition here (still 1 critical + 23 high, 0
  inversions). But as an *ordering* — reserved tasks placed at the head, which is how a
  slot reservation is naturally implemented — S5 produces **1 tier inversion**: four `high`
  tasks sit above the `critical` task 3618. That is cosmetic when all 24 slots refill at once
  and lethal when they don't: with 2 free slots, this policy dispatches two reserved `high`
  tasks and leaves a `critical` task waiting. Generalising: the reservation is unconditionally
  tier-blind, so whenever there are ≥ 21 critical candidates it displaces critical work with
  lower-tier work. **Any adoption must scope the reservation within-tier or apply it only to
  the lowest R slots.**
- **Assessment.** As an intervention this is the highest surgical precision per unit of
  disruption in the whole study: it hits exactly the failure mode ("deep chains are invisible")
  with exactly the minimum blast radius (2 displaced tasks), and it leaves the S0 scorer — and
  every property anyone has ever reasoned about it — untouched. But it is a patch bolted on top
  of a scorer that is *still* 85.6%-saturated and still ordering 117 tasks by id; it fixes the
  symptom for 4 tasks per tick and leaves the mechanism broken for the other 287. It also
  introduces a second, independent policy knob (R) with its own starvation surface: a task that
  is neither old enough for S0 nor deep enough for the L-reservation is now competing for 20
  slots instead of 24.

## D vs L divergence

Over the 28 candidates with **D ≥ 2** (ranked by D and by L, top 15 by |Δrank|):

| id | tier | age | D | L | rank_D | rank_L | \|Δ\| |
|---|---|---|---|---|---|---|---|
| 3623 | high | 173 | 23 | 3 | 4 | 19 | **15** |
| 3624 | high | 172 | 23 | 3 | 5 | 20 | **15** |
| 3523 | high | 273 | 24 | 3 | 3 | 17 | **14** |
| 3132 | high | 664 | 26 | 4 | 2 | 13 | 11 |
| 3688 | high | 108 | 8 | 3 | 11 | 21 | 10 |
| 3184 | high | 612 | 9 | **7** | 9 | **1** | 8 |
| 3628 | medium | 168 | 5 | 5 | 17 | 9 | 8 |
| 3629 | medium | 167 | 5 | 5 | 18 | 10 | 8 |
| 3630 | medium | 166 | 5 | 5 | 19 | 11 | 8 |
| 3631 | medium | 165 | 5 | 5 | 20 | 12 | 8 |
| 3658 | high | 138 | 4 | 4 | 22 | 14 | 8 |
| 3197 | high | 599 | **31** | 5 | 1 | 8 | 7 |
| **3534** | **high** | **262** | **9** | **7** | **10** | **3** | **7** |
| 3715 | medium | 81 | 7 | 6 | 13 | 6 | 7 |
| 3716 | medium | 80 | 7 | 6 | 14 | 7 | 7 |

**How often is a high-D task actually a shallow fan-out?** 7 of 28 D≥2 candidates (**25%**)
have L ≤ 2. Within the top-24 *by D*, 3 tasks have L ≤ 2 (3165 D=4/L=2, 3076 D=2/L=2,
3319 D=2/L=2). The largest divergences are all in this direction — **3623/3624 (D=23, L=3)**
and **3523 (D=24, L=3)** are wide, one-hop batches: completing them unlocks ~23 tasks that can
then all run *in parallel*, which is exactly the case where a big D **overstates** the
critical-path benefit.

**How often is a deep chain invisible to D?** In this snapshot, **never in the strict sense**:
of the 14 candidates with L ≥ 4, **zero** have D ≤ 3 — a chain of length L structurally implies
at least L distinct descendants, so D can never be blind to depth. What D does instead is
**mis-rank** it. 3534 is the cleanest example: rank **10** by D but rank **3** by L. And the
mis-ranking is severe once tier is folded in — 3715/3716 (L=6, D=7) sit at S0 ranks 135/136,
though that is their `medium` tier, not D.

**Concretely, for 3534: D = 9, L = 7** (tier `high` by inheritance, own priority `high`,
age 262). Its longest downstream chain over the active graph is
`3534 → 3535 → 3538 → 3539 → 3540 → 3541 → 3545 → 3546` (7 edges — one longer than the
6-node chain named in the brief, because `3545` depends on `3541` and `3546` on `3545`), and its
9 non-terminal transitive dependents are `{3535, 3538, 3539, 3540, 3541, 3542, 3545, 3546, 3587}`.
It is one of only three candidates at the maximum observed depth L=7.

## Recommendation

**S2 first, S5 as an immediate stopgap, S1 as the floor; S4 not as specified; S3 no.**
The evidence points at one root cause, not five: the score sums a term with range 0–10640
against a term with range 0–347 and then clips the sum at 999, so 85.6% of candidates land on
an identical score and 24 dispatch slots are allocated by `task_id` string order. Only S2
addresses that *as a class* — it is the only design that cannot saturate (0/291 vs 249/291),
that cannot cross tiers (`0.5·age_rank + 0.5·cpm_norm < 1` strictly, so bonus ≤ 998), and that
produces a genuinely graded order in both dimensions simultaneously; it is also the only scorer
that promotes 3534 (32→13) without the collateral S3 inflicts, and it does so at a modest
τ = +0.9589 and 8/24 churn, all of it replacing `D=0, L=0` leaves. S1 is strictly worth shipping
regardless, because it is one config line and it kills the saturation for 248 of 249 tasks — but
the α/β sweep shows it is *not sufficient* (3534 needs α ≤ 0.5 or β ≥ 200, not α = 1), so treat
it as the floor rather than the fix. S5 is the best value-per-disruption in the study and is
worth deploying *today* — it costs 2 top-24 slots and no scorer change — but only after scoping
the reservation within-tier, since as a head prefix it hoists four `high` tasks above a
`critical` one. S4 should not ship in its specified form: `L ≤ D` for all 291 candidates, so
`log1p(L) ≤ log1p(D)` and S4's score is provably ≤ S1's for every task — it is a uniform shrink
of the CPM term, and it *lowers* 3534's bonus from 230 to 208. S3 is the clearest reject: it
lets 3585 (D=1, age 211) displace 3090 (D=0, age **706**) purely on a bucket boundary, pushing
within-tier starvation exposure to 706 and defeating aging structurally.

**Strongest counter-argument to my own recommendation:** S2 is the only proposal whose score is
**not a function of the task**, and I measured that the coupling is not negligible — perturbing
the candidate set drifts surviving scores by up to **68 points**, which is *larger* than the
78-point margin that kept 3534 out under S1. That means a task's rank can cross the 24-slot
boundary because some *unrelated* task completed, which makes dispatch decisions non-reproducible
from a task's own record, breaks the "why didn't X run?" debugging story that the current
`task_skipped`/`score` events support, and would make any future regression test on ordering
depend on the whole DB. A defender of S1-plus-tuning would say: the actual disease is two
mis-scaled constants, the fix is two constants (α≈0.33 so `α·max_age ≈ β·max_log1p(D)`), that
fix keeps scores task-local and perfectly stable (τ = 1.0000 across all three perturbations),
and it lands 3534 at rank 13 — *the same rank S2 gives it* — without ever introducing a
set-dependent term. On this snapshot's evidence alone I cannot distinguish those two outcomes,
and the simpler one has the better failure mode.
