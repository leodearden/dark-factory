# Speculation-waste diagnosis: why reify voids 58% and dark_factory 30%

**Status:** complete — measured 2026-09-07 (task G / 5058)
**Repos measured:** `dark-factory`, `reify` (both `data/orchestrator/runs.db`, read-only)
**Origin:** `plans/merge-lane-throughput-prd.md` § Background, the 30-day speculation
rows, plus that PRD's § Open questions on speculation waste.
**Consumers:** task **5098** (policy-PRD authoring gate), task **5056** (E) and
task **5059** (H).

## Goal

Say what the two projects' speculation void rates actually MEAN, so that the
dispatch/speculation policy PRD (task 5098) is aimed at the resource the waste
actually consumes, ranks its levers by expected value rather than by intuition,
and knows which of its candidate causes the available evidence can and cannot
separate.

This report PROPOSES ONLY. No speculation-algorithm change is made here
(throughput PRD decision 5).

## The command

Every number below is reproducible from one invocation of the measurement
script landed by task A (5050):

```
python scripts/merge_lane_throughput.py --speculation \
  --project-root /home/leo/src/dark-factory --project-root /home/leo/src/reify \
  --window '2026-08-04T16:10:00+00:00..2026-09-03T16:10:00+00:00'
```

The task named the window as `--window 30d`. That form is **relative to now**
(`scripts/merge_lane_throughput.py::parse_window`) and so is not reproducible
the day after it is quoted. Measured drift over four days, same command,
`--window 30d` run on 2026-09-07:

| | dated window | `30d` on 2026-09-07 |
|---|---|---|
| dark_factory void rate | 126/424 = **0.297** | 85/390 = 0.218 |
| reify void rate | 154/264 = **0.583** | 145/253 = 0.573 |

The **dated** form is the reproducible one and is what this report quotes
throughout. It is also the window that reproduces the throughput PRD's own
§ Background 30-day speculation cells exactly (424; 126 and 264; 154 — see
§ 3 for the one row that does not). Every rate below names its window; a
rate without one is a rate two readers will resolve differently.

Unless stated otherwise, **W** = `2026-08-04T16:10:00+00:00 ..
2026-09-03T16:10:00+00:00`.

## 1. Headline

Over **W**:

| | dark_factory | reify |
|---|---|---|
| `speculative_merge` | 424 | 264 |
| `verdict_voided` (`chain_dead`) | 126 | 154 |
| void rate | **0.297** | **0.583** |
| void points | `{'dispatch': 126}` | `{'dispatch': 154}` |
| voids that burned a verify | **0** | **0** |
| voids discarded pre-verify | **126** | **154** |
| distinct dead-base SHAs / max voids from one | 120 / 2 | 131 / 3 |
| landed with speculation ahead (loose) | 351/507 (0.692) | 191/323 (0.591) |
| ...and not voided first (strict) | 255/507 (0.503) | 84/323 (0.260) |

## 2. The reframing: the void rate is not burned verify capacity

**This section re-aims everything after it.** The intuitive reading of "reify
voids 58% of its speculations" is "reify burns 58% of its speculative verify
capacity". That reading is wrong, and a proposal list built on it would aim
the policy PRD at the one resource these voids demonstrably do not consume.

### There are two void arms and they cost different things

Both are enforcement points of INV-3, the chain-intact invariant, and both use
one shared predicate,
`orchestrator/src/orchestrator/merge_queue.py::SpeculativeMergeWorker._chain_dead_link`
— *"the chain is DEAD iff `item.base_sha` is a known-dead commit (recorded in
`_dead_base_commits`) AND is not current main"*. The ledger is a bounded FIFO
(`merge_queue.py::_DEAD_BASE_COMMITS_CAP` = 256, evicted in
`merge_queue.py::SpeculativeMergeWorker._record_dead_base`).

- **The ADOPTION arm** —
  `merge_queue.py::SpeculativeMergeWorker._void_and_remerge`, reached from
  `_finalize_inflight` downstream of `vr = await entry.verify_task`. A full
  verify has already completed and is thrown away. **Expensive.**
- **The DISPATCH arm** — emitted inline in
  `merge_queue.py::SpeculativeMergeWorker._dispatch_item`, under the header
  comment *"INV-3 dead-base re-check (enforcement point (a), dispatch)"*, and
  strictly **before** `lease = await allocator.acquire(...)`. No lease is
  taken and no verify is launched. Its own log line states the intent:

  > `Task %s: dead-base straggler at dispatch (dead link %s) — re-merging
  > against actual main instead of burning a verify`

  The path is so deliberately cheap that even the `get_main_sha()` subprocess
  is elided on the steady-state hot path, gated behind `_needs_main_sha`
  (*"In the common steady state (empty ledger + speculative item) NEITHER
  consumer needs it, so skip the get_main_sha() subprocess entirely"*).

### Measured: 100% of the voids are the cheap arm

Over **W**, in BOTH projects, every single `verdict_voided` row carries
`point='dispatch'` — 126/126 and 154/154 — and **zero** of them (0/126, 0/154)
had a `merge_verify` for the same task between the item's last preceding
`speculative_merge` and the void. That second measurement is independent of
the `point` label: it is computed from the `merge_verify` rows themselves
(`scripts/merge_lane_throughput.py::compute_speculation`, `void_anatomy`), so
the two agree without sharing a source.

### Why the adoption arm is empirically empty — by design, not by luck

`merge_queue.py::SpeculativeMergeWorker._verifier_loop` drains the population
that would otherwise reach the adoption check. When the head does not advance
and `self._inflight` is non-empty, in the same synchronous beat it clears
`_inflight`, calls `_record_dead_base` for the head, and then for every
downstream entry records that entry's own old merge commit dead, `_remerge`s
it against real main, and parks it on `_redispatch`. After the re-merge those
items' `base_sha` is live, so `_chain_dead_link` returns `None` for them later.

The dispatch check therefore exists for a **different** population, and the
code says so:

> `Record it dead so a straggler that was BUILT-AWAITING-HOST — parked on
> _redispatch, NOT in self._inflight, so INVISIBLE to this _inflight-only
> cascade (the exact 5260 gap) — is caught by the dispatch-time dead-base
> re-check (enforcement (a)) instead of verifying against it.`

So the expensive arm is prevented structurally, and the 58% is the cheap arm
firing as intended. **The void rate is a measure of how often a correctness
backstop fires, not of wasted verify minutes.**

### What the cheap arm does cost

Not nothing. The dispatch-arm path, in source order inside `_dispatch_item`:

1. `_chain_dead_link` returns a dead link
2. `_note_transition` (DISPATCHING → MERGING)
3. `_cleanup_owned_merge_worktree(item.merge_wt)` — **the already-built merge
   worktree is destroyed**
4. `_emit_speculative(EventType.verdict_voided, ..., point='dispatch')`
5. `_record_dead_base(item.merge_result.merge_commit)` — *"this doomed build's
   OWN merge commit is orphaned by the re-merge below"*
6. `_remerge(req, item.started_monotonic)`
7. `_note_transition` (MERGING → DISPATCHING)

So one void costs: **a discarded merge build and worktree, a re-merge, and the
foregone speculation** — the queue slot that item occupied produced nothing.
Zero verify time.

**Caveat carried forward:** this cost is NOT equal across the two projects.
reify runs `persistent_merge_worktree: true` and `merge_spec_warm_lane_pool:
true`; dark_factory declares neither and
`orchestrator/src/orchestrator/config.py` defaults both `False`. A discarded
build means something structurally different on each side. See § 5(d) — the
void COUNTS above must not be compared as if each void cost the same.

## 3. The PRD's speculative-ahead rows do not reproduce

`plans/merge-lane-throughput-prd.md` § Background, 30-day row:

| | dark_factory | reify |
|---|---|---|
| PRD: landings that were speculative-ahead | 165 / 416 (40%) | 10 / 277 (3.6%) |
| script, loose measure, same window **W** | 351 / 507 (0.692) | 191 / 323 (0.591) |
| script, strict `adopted` measure, **W** | 255 / 507 (0.503) | 84 / 323 (0.260) |

The same command at the same window reproduces every other 30-day cell in that
table exactly, so this is not drift.

**The artifact, named.** `compute_speculation`'s `speculative_ahead` counts a
landing as speculative-ahead if the task had ANY `speculative_merge` (or a
speculative `merge_verify`) strictly before the landing — **including
speculations later voided**. That over-counts precisely in the project whose
voids dominate, i.e. it inflates reify relative to dark_factory for reasons
that have nothing to do with speculation helping.

Task G added `speculative_ahead_adopted` beside it (not instead of it — the
loose key is the landed contract E and H compare against). It requires that no
`verdict_voided` for that task falls strictly between the LAST preceding
speculation and the landing. The strict measure moves reify roughly three times
as far as dark_factory (0.591 → 0.260 vs 0.692 → 0.503), which is the spread
the loose measure was concealing, and it moves in the direction the PRD's rows
imply.

**But it does not close the gap, and this report does not claim it does.** The
denominators differ too — 507 vs the PRD's 416, and 323 vs 277 — so the PRD's
figure was computed over a different landing population, not merely with a
different numerator rule. No definition has been found that reproduces
165/416 and 10/277. **What the PRD's row actually measured is unknown**, and
the honest statement is that the current reproducible measurements are the two
script rows above. § 5(e) names one mechanism that plausibly bears on the
denominator, without claiming it as the explanation.

`plans/merge-lane-throughput-prd.md` § Corrections (2026-09-07) carries a
pointer to this section so a reader of that table is not misled.

## 4. The PRD's named inquiries, with verdicts

### 4a. Probe placement — NEGATIVE, the knob is inert

`merge_queue.py::select_probe_depth` opens with `if probe_fraction <= 0.0:
return None`, before any other logic. Its own docstring, check 1:

> `probe_fraction <= 0.0` -> `None` unconditionally. This is the task's
> byte-identical guarantee: at the default fraction (0.0), every call returns
> `None` regardless of the other arguments [...]

`merge_queue.py::SpeculativeMergeWorker._probe_verify_placement` short-circuits
on the same value **before** calling the policy at all (*"Zero-cost disabled
path"*). And **both** projects set `speculation_probe.probe_fraction: 0.0` —
each with a dated `DEACTIVATED 2026-07-23` operator note explaining that the
probe cannot produce genuine depth≥2 records under K=2.

**Probing is off everywhere. It explains none of the gap, and no proposal
should be ranked against it.** A proposal that would re-enable it must first
address the 2026-07-23 deactivation rationale, not merely the fraction.

### 4b. Cascade amplification — NEGATIVE, fan-out is near-flat

If one dead base were killing many downstream items, a few `dead_link` SHAs
would account for most voids. They do not. Over **W**:

- dark_factory: 120 distinct dead-base SHAs over 126 voids; **max 2** voids
  from any single SHA.
- reify: 131 distinct over 154 voids; **max 3**.

Voids are **independent stragglers**, not one head failure cascading. This is
consistent with § 2: the `_verifier_loop` cascade already re-merges the
`_inflight` downstream in one beat, so what reaches the dispatch check is the
one-off built-awaiting-host straggler. A "damp the cascade" proposal has
nothing to damp.

### 4c. `gate_retry` / `cas_retry` — real, but NOT substitutable with voids

Counts are already tallied per project by task A's `mixes` section
(`scripts/merge_lane_throughput.py::compute_mixes`, `_NON_TERMINAL_OUTCOMES`);
they are not re-derived here. Over **W**, `merge_attempt` outcomes:

- dark_factory (n=909; 780 terminal, 129 non-terminal): done 518 (0.570),
  verify_failed 99 (0.109), **gate_retry 92 (0.101)**, superseded 57 (0.063),
  train_incomplete 26 (0.029). No `cas_retry` row.
- reify (n=435; 376 terminal, 59 non-terminal): done 323 (0.743),
  **gate_retry 48 (0.110)**, verify_failed 29 (0.067). No `cas_retry` row.

gate_retry runs at ~10-11% on both sides — it is **not** part of the
between-project spread this report is explaining.

What the counts lacked was the mechanism. Both retry kinds are emitted inside
`merge_queue.py::SpeculativeMergeWorker._finalize_inflight`'s `while True:` CAS
advance loop, **after** the adoption-point void check in that same function
(which returns before ever reaching the CAS arm):

- **`gate_retry`** — main advanced under a finalizing request
  (`rebased_pending_reverify`). The tree is rebased and a re-verify gate
  (`_reverify_rebased_tree`) must clear before the CAS is retried with the
  rebased SHA.
- **`cas_retry`** — the CAS ref-update lost the race; `base_sha` is set to
  current main and the loop retries.

**The structural point.** Both retries stay inside that loop on the **same
already-verified worktree** — `merge_wt` is bound once before the loop and is
never rebuilt or released on either path; only `item.base_sha` is replaced. A
**void** takes the opposite route: `_void_and_remerge` first calls
`_release_or_cleanup` (the verified worktree is **destroyed**), then `_remerge`
and `_redispatch.appendleft` — *"a fresh verify"*.

So a retry re-uses work and a void discards it. **Summing gate_retry, cas_retry
and voids into one "waste" figure — the obvious move for a policy PRD author —
would be wrong.** They are different resources at different prices.

## 5. What the evidence CANNOT separate

This section is a finding in its own right, and it is a constraint on § 6.

Every candidate cause below is **on for reify and off for dark_factory**. With
n = 2 projects they are perfectly collinear: **no query over the event stores
separates any of them**, and this report does not rank as if the attribution
were settled. The PRD named three; there are at least five.

**(a) K = 2 vs K = 1.** `orchestrator/src/orchestrator/harness.py::_speculation_k`
is *"Single shared K source: 1 + len(enabled_verify_runners)."* reify declares
one enabled `laptop` runner; dark_factory declares no `verify_runners` block at
all. *Separating intervention:* enable a verify runner for dark_factory alone
(this is roughly task D1/5053's shape) and re-measure the same window.

**(b) reify's disabled cross-check leg.** `verify_cross_check_remote_green:
false`, disabled 2026-08-21 as an interim mitigation for dark_factory:4579,
whose own config note records **13 confirmed cross-check starts since
2026-07-20 yielding ZERO verdicts**, and laptop-attributed lands falling
**57 → 1**. Root cause recorded there: the merge worker's in-flight no-progress
detector is gated behind `if lease.is_local:` while the lease is remote, so the
local cross-check leg is invisible to it and the budget SIGKILLs a healthy
cross-check. `verify_drift_check_every_n_lands: 20` is the only standing
fidelity guarantee left. *Separating intervention:* land 4579, re-enable the
cross-check on reify alone, re-measure.

**(c) Workload shape.** reify's verifies are 30min+ cargo builds; dark_factory's
are pytest. Measured over **W**, `local` runner: dark_factory p50 **32.0** min
(p90 46.4, n=624), reify p50 **44.5** min (p90 73.5, n=370) — reify's verifies
are ~1.4x longer at the median and ~1.6x at p90, on the same workstation. A
longer verify holds the head longer, which lengthens the window in which a
straggler's base can go dead. *Separating intervention:* none available
in-band — this is a property of the repos, not a knob, which is precisely why
it cannot be subtracted out and why (a), (b), (d) and (e) cannot be attributed
cleanly around it.

**(d) Merge-worktree asymmetry — NEWLY IDENTIFIED, not named by the PRD.**
reify runs `persistent_merge_worktree: true` **and** `merge_spec_warm_lane_pool:
true` (K CoW-seeded `_spec-{0..K-1}` warm lanes); dark_factory declares neither
and `orchestrator/src/orchestrator/config.py` defaults both `False`. This is
load-bearing rather than incidental: it means **the per-void rebuild cost
quantified in § 2 is structurally different on each side**, so the 126 vs 154
void counts cannot be compared as if each void cost the same. A void on
dark_factory discards a cold ephemeral `_merge-<hash>` worktree; a void on
reify releases a warm CoW-seeded spec lane. *Separating intervention:* enable
the warm-lane pair on dark_factory alone, or disable it on reify alone, and
re-measure void cost (not void rate).

**(e) Merge-train firing asymmetry — NEWLY IDENTIFIED, and NOT a config
asymmetry.** Both projects declare the train identically —
`merge_train_former_enabled: true`, `merge_train_coalesce_enabled: true`,
`merge_train_max_members: 3` in both `dark-factory-orchestrator.yaml` files.
But over **W** only dark_factory's `merge_attempt` mix contains train outcomes
at all: `train_incomplete` 26 (0.029) and `train_rebase_conflict` 2 (0.002);
reify emits **zero** rows of any `train_*` outcome. The trains are configured
on both and firing on one. That changes what a single `merge_finalized`
landing MEANS on each side — a coalesced train lands several tasks under
fewer landing events — and therefore bears directly on the ahead-share
**denominators** in § 3 (507 vs the PRD's 416; 323 vs 277). *Separating
intervention:* disable coalescing on dark_factory for one window and
re-measure the landing count, or count member tasks rather than landings.
**This is a hypothesis about the denominator, not a reconciliation of § 3.**

## 6. Ranked proposals for the policy PRD (task 5098)

Ranked by expected value **given § 2**: proposals aimed at build/lane churn
and at the ahead share rank above anything aimed at verify capacity, which the
evidence shows these voids do not consume. Seams are named from
`plans/merge-lane-quality-prd.md` § Contract: **κ** = task **5040**,
`merge_lane/verify_dispatch.py::VerifyDispatcher`; **λ** = task **5041**,
`merge_lane/speculation.py::ChainPlanner` / `Speculator`.

**P1 — Do not BUILD a speculative item whose base is already suspect (λ,
5041).** Today the dead-base check runs at dispatch, after the merge worktree
is built. The 126/154 discarded builds are all caught by a predicate
(`_chain_dead_link`) whose inputs — `_dead_base_commits` and current main —
are available at *build* time too. Moving (or duplicating) the check earlier in
`ChainPlanner` would convert a build-then-discard into a not-built.
*Cost:* one more `get_main_sha()` on the build path, or a build-time read of
the ledger; the `_needs_main_sha` gating shows the team already treats that
subprocess as worth eliding. *Falsified by:* a measurement showing the
straggler's base only becomes dead **after** its build started — in which case
no earlier check can catch it. **Measure that first**: it is the single
cheapest experiment in this list, and it decides P1 outright.

**P2 — Reduce the built-awaiting-host straggler population (κ, 5040).** § 2
shows the voided population is specifically items that are BUILT and parked on
`_redispatch` waiting for a host, invisible to the `_inflight`-only cascade.
Anything that shortens build→host latency shrinks the window in which a base
can go dead underneath a built item. *Cost:* real, and it overlaps
**task 5097's proposed `verify_host_policy` knob** (`prefer_local |
prefer_remote`, consulted in `HostAllocator.acquire`) — which **does not exist
in code today**; 5097 is a filed task, not a landed feature. A policy PRD
proposal here must depend on 5097 rather than assume it. *Falsified by:*
host-occupancy data showing stragglers wait on a *free* host (i.e. the delay is
not contention). Note what the same window measures for **reify**, the only
project with two hosts: `local` 83.6% LOCF busy vs `laptop` 23.3% — the shared
workstation is the constrained resource and the laptop is three-quarters idle,
which is the same imbalance 5097 was filed against (`HostAllocator.acquire`
prefers local when free). dark_factory has one host at 84.0% LOCF, so for it
this proposal reduces to 5097 + D1/5053 landing first.

**P3 — Report and act on the STRICT ahead share, not the loose one (λ, 5041).**
§ 3 shows the loose measure over-credits exactly where voids dominate, by up to
3.3x (reify 0.591 → 0.260). Any speculation-depth or chain-cap policy tuned
against the loose measure is tuned against a number inflated by its own waste.
*Cost:* near-zero — the measure is landed
(`scripts/merge_lane_throughput.py::compute_speculation`,
`speculative_ahead_adopted`) and printed. *Falsified by:* nothing; this is a
reporting-discipline proposal, and its only risk is that E/H's baselines use
the loose key, which is why the loose key was kept rather than redefined.

**P4 — Make the per-void cost comparable before comparing void counts (λ,
5041, plus a config change).** § 5(d): reify's warm spec-lane pool means its
voids and dark_factory's are not the same unit. Either normalise the reporting
(cost per void, not count of voids) or equalise the config. *Cost:* the config
route is a real behaviour change on a live lane; the reporting route is cheap.
*Falsified by:* a measurement showing warm-lane re-seed cost is negligible
relative to the re-merge, making the two units interchangeable after all.

**P5 — Anything aimed at verify capacity: DEPRIORITISED, with evidence.** Over
**W** these voids burned **0** verifies in both projects (§ 2). A proposal to
"stop wasting verify slots on doomed speculations" is aimed at a cost of zero.
This is stated explicitly so the policy PRD's author does not re-derive it —
and so that if `verify_burned` ever becomes non-zero (the measure now reports
it) the deprioritisation is revisited rather than inherited.

**NOT PROPOSED: re-enabling `speculation_probe`.** § 4a — inert in both
projects with a dated operator deactivation whose stated reason (no genuine
depth≥2 records under K=2) is unaddressed. Ranking a proposal against an inert
knob would be ranking against nothing.

## 7. Consumers and out of scope

- **Task 5098** — the policy-PRD authoring gate, which fires once G (5058) and
  E (5056) have landed. § 2 is the spine; § 5 is the constraint on § 6; § 6 is
  the input. The seam ids (κ=5040, λ=5041) are taken from
  `plans/merge-lane-quality-prd.md` so the follow-up wires dependencies
  without re-deriving them.
- **Task 5050 (A)** — this task extended its script with the strict
  `speculative_ahead_adopted` measure and the `void_anatomy` split
  (pre-verify / verify-burned, plus `dead_link` fan-out), carried them into
  `void_rate_by_project`, and surfaced both on the text report and `--json`.
  The existing `speculative_ahead` key and every existing printed line are
  unchanged.
- **Tasks 5056 (E) and 5059 (H)** — see § 3 and the 2026-09-07 correction in
  `plans/merge-lane-throughput-prd.md`: compare against this report's rows,
  not against § Background's speculative-ahead cell.

### Out of scope

- **Any speculation-algorithm change** (throughput PRD decision 5). This
  report proposes; 5098 decides.
- **Re-deriving `gate_retry` / `cas_retry` counts** — they live once, in task
  A's `compute_mixes`, and are cited from there (§ 4c).
- **Asserting a live number in a test.** Both `runs.db` stores mutate
  continuously (measured: dark_factory's void rate moved 0.297 → 0.218 in four
  days), so every test added by this task is a hand-computed known answer over
  the synthetic corpus. The live rates live here, in a dated report,
  reproducible on demand from the command at the top.
