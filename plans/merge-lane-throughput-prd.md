# Merge lane throughput: self-hosting on the remote verify host, arbitration, deep-chain canary, measurement

**Status:** active — authored 2026-09-03 (Leo + claude-interactive) from a measured
14/30-day baseline of both hosted projects' `data/orchestrator/runs.db`.
**Type:** functional + measurement PRD. Approach **B+H** for one seam (cross-project
host arbitration, § Contract); bare B elsewhere.
**Code anchors** verified against main `4811d62883` (2026-09-03). Cite-by-symbol;
re-locate at implementation time. Measurement tables are dated provenance, not live
counts.
**Companion PRD:** `plans/merge-lane-quality-prd.md` (the structural program). This
PRD touches none of that PRD's Appendix A files and runs in parallel with it. The
*policy* follow-up this PRD seeds (task G's diagnosis plus Leo's own dispatch and
speculation ideas) is the PRD that will depend on the quality PRD's κ and λ seams.

## Goal

Raise merge-lane throughput for every hosted project by putting the one idle
remote verify host to work for Dark Factory as well as reify, safely shared, and by
measuring the effect instead of asserting it. Operator-observable outcome:

- Dark Factory's `merge_verify` events carry `runner: laptop` when the laptop is
  free, and `verify_host_unreachable` stays at zero.
- Two projects using one host never produce a born-at-L2 "flock contention"
  escalation; a busy host is a scheduling outcome (local fallback plus a structured
  `verify_host_busy` event), and a storm of them is one storm-counted L1.
- `scripts/merge_lane_throughput.py --project-root <root> --window 14d` prints
  landings/day, lead-time split (queue wait / verify / finalize), verify duration by
  runner, remote-host occupancy, speculation void rate, queue depth and failure mix
  for any project — the same table as § Background, reproducibly.
- A dated before/after report lands 14 days after enablement, and a second one 14
  days after the deep-chain canary, so each lever's effect is attributed separately.
- Deep merge-ahead lands chains on Dark Factory (`landed_via_chain` on a
  `merge_finalized` event) under a canary cap.
- A committed diagnosis explains why reify voids 58% of its speculative verdicts
  and lands only ~2% of merges speculative-ahead against Dark Factory's 30% / 38%,
  with a ranked proposal list the policy PRD is authored from.

## Background

Measured 2026-09-03 (events table, read-only; windows end 16:10 UTC; queries are
codified by task A):

| Measure (14-day window) | dark_factory | reify |
|---|---|---|
| Landings / day (median, max) | 12.0, 27 | 13.5, 25 |
| Lead time p50 / p90 (min) | 50.0 / 171.7 | 64.2 / 385.1 |
| Queue wait p50 / p90 (min) | ~0 / 110.4 | ~0 / 233.8 |
| Verify p50 / p90, local (min) | 40.7 / 52.4 | 43.4 / 62.4 |
| Verify p50 / p90, laptop (min) | — (no runner configured) | 17.4 / 20.6 (n=16) |
| Finalize + CAS residual p50 / p90 (min) | 2.0 / 24.5 | 26.8 / 126.1 |
| Queue depth p50 / p90 / max | 2 / 8 / 29 | 2 / 35 / 44 |
| Laptop occupancy (heartbeat LOCF integral) | — | 22.2% (12.5% over 30d) |
| Speculative merges; verdicts voided `chain_dead` (30d) | 424; 126 (30%) | 264; 154 (58%) |
| Landings that were speculative-ahead (matched, 30d) | 165 / 416 (40%) | 10 / 277 (3.6%) |
| `merge_attempt` outcomes (30d) | done 518, verify_failed 99, gate_retry 92, superseded 57 | done 323, gate_retry 48, verify_failed 29 |
| Deep merge-ahead (`chain_cap`) | unset → 0 (kill switch); `landed_via_chain` never true | same |

What exists (verified 2026-09-03):

- The remote-verify machinery is fully landed: `verify_runner.py::VerifyRunnerPool`,
  `RemoteRunner`, `HostAllocator`/`HostLease`, `DriftDetector`, the
  `orchestrator verify-merge` CLI (`cli.py::verify_merge`), quarantine + reprobe
  (`SpeculativeMergeWorker._reprobe_quarantined_hosts`). Only reify enables it
  (`verify_runners: [laptop → leo-laptop]`, K = 2).
- Dark Factory already has the plumbing half-built: a `leo-laptop` git remote in the
  project root, a checkout at `leo-laptop:~/src/dark-factory` (HEAD `5f185fdf00`,
  2026-08-20, 3 dirty files, `.venv` present, no `node_modules`), and a per-host
  config `~/.config/orchestrator/dark-factory-laptop.yaml` (`persistent_merge_worktree:
  false`, `remote: workstation`). Its own `dark-factory-orchestrator.yaml` has no
  `verify_runners` block. `verify_runners` is restart-only (OPERATIONS.md § Config
  reload vs restart lists it under pool sizes).
- Cross-project arbitration on the host does **not** exist. `HostAllocator` is
  per orchestrator process; merge-role verifies bypass admission slot-counting by
  design (`verify_admission_pytest_n` comment in the config); reify's laptop-side
  `.merge_verify.lock` is per project root and, on contention, the workstation
  files a **born-at-L2** and blocks the merge (`_run_post_merge_verify`'s
  `is_flock_contention_failure` branch, task 2307 β). Two projects sharing the
  laptop under today's code would either thrash it or page Leo.
- Deep merge-ahead is landed behind `MergeDeepConfig.chain_cap` (read by
  `select_chain_depth`; the config docstring's "nothing reads the knob yet" is stale).
  Its Phase 3 tasks are filed and pending: 3188 (telemetry + report), 3189 (reify
  canary at cap 6), 3190/3191 (7-day predicate, promote to 32), 3192, 3193. The
  knob is green-tier hot-reloadable per that PRD's decision 7.

Prior art: `plans/merge-throughput-multihost-verify-prd.md` (Lever C),
`plans/concurrent-merge-verify-prd.md`, `plans/laptop-warm-verify-flock-orphan-prd.md`
(the contention sentinel this PRD re-routes), `plans/deep-merge-ahead-prd.md`,
`plans/merge-throughput-disjoint-former-design.md` (ranked the levers).

## Sketch of approach

Measure first, then enable one lever at a time with a 14-day window each, so the
two levers are attributed separately:

1. **A** codifies the baseline queries as a script with a fixture test.
2. **B** brings the laptop checkout and per-host config up to date; **C** builds
   host-level arbitration so a shared host is safe; **G** (independent) diagnoses
   the speculation waste.
3. **D** enables the laptop for Dark Factory and restarts the fleet; **E** measures
   14 days later.
4. **F** turns on the deep-chain canary for Dark Factory once its telemetry (3188)
   exists; **H** measures 14 days after that.

## Resolved design decisions

1. **Dark Factory self-hosts first; reify is not the testbed** (Leo's goal, my
   sequencing). This lane runs the simplest configuration (K = 1, no chains), so
   enabling the laptop here is a clean before/after; the quality program's own
   tranches land through this lane and benefit directly; reify's remote history is
   confounded (its cross-check leg was disabled 2026-08-21 after false-hang kills).
2. **A busy host is a scheduling outcome, never an escalation.** C replaces the
   born-at-L2 contention path with: laptop-side host-global admission for
   `verify-merge` (one concurrent merge verify per host by default, bounded wait),
   a `RunnerBusy` result the workstation-side pool turns into local fallback plus a
   structured `verify_host_busy` event (INV-2, INV-11), and a storm counter that
   files one L1 above a rate (INV-4). This supersedes task 2307 β's ruling for the
   two-project case; C amends `plans/laptop-warm-verify-flock-orphan-prd.md` with a
   dated pointer (INV-9) rather than editing its frozen text.
3. **No numeric throughput target.** Every measurement leaf reports a comparison
   against the dated baseline; no leaf asserts "lead time falls by X%" (G6: there is
   no achievability basis for a number before the first measurement). The
   achievability basis for *some* gain is reify's laptop verify p50 of 17.4 min
   against 43.4 local on the same workload class.
4. **One lever per window.** F waits for E so the remote-host effect and the
   deep-chain effect are not confounded. Reify's own canary (3189) proceeds on its
   own PRD's schedule.
5. **Diagnosis before policy.** G is an investigation leaf with a committed report;
   speculation *algorithm* changes are out of scope here and become the policy PRD,
   authored by Leo with G's proposals and his own ideas, depending on the quality
   PRD's κ and λ.
6. **Nothing here edits the quality PRD's Appendix A files.** C's workstation-side
   change lives in `verify_runner.py` (the pool absorbs `RunnerBusy` before it can
   reach `_run_post_merge_verify`), so the merge-queue contention branch becomes
   unreachable and is deleted by the quality PRD's κ, not here.
7. **Dirty laptop state is inspected, never discarded silently.** B lists the three
   dirty files and any local commits in its task result before resetting; if any is
   not reproducible from `main`, B stops and escalates instead of resetting.

## Pre-conditions for activating

- `ssh leo-laptop` reachable from the workstation in batch mode (verified
  2026-09-03: hostname answered); the `leo-laptop` git remote exists in the project
  root (verified).
- Task 3188 (deep-merge telemetry + report) lands before F; wired as a dependency,
  not assumed.
- Fleet auto-redeploy is paused pending task 5020; D2 restarts deterministically.
  Tasks 3730/3733/4755/5020 are **not** dependencies of anything here (Leo).
- `scripts/restart-all-orchestrators.sh` exists and is executable (verified).

## Cross-PRD relationship

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/merge-lane-quality-prd.md` | parallel | none of its Appendix A files; its κ later deletes the dead `is_flock_contention_failure` branch once C lands | each PRD its own files; the κ deletion is noted for κ's architect by the cross-PRD seam check | queued |
| `plans/deep-merge-ahead-prd.md` | consumes | `MergeDeepConfig.chain_cap`, `select_chain_depth`, `landed_via_chain`, 3188's telemetry events, 3189's canary `before_done` script pattern | deep-merge PRD owns the mechanism and the reify canary; **this PRD** owns the Dark Factory canary (F), which depends on 3188 and mirrors 3189's script with this project's paths | wired |
| `plans/concurrent-merge-verify-prd.md` (landed) | extends | `HostAllocator`, `RemoteRunner.run_merge_verify`, `VerifyRunnerPool.dispatch` | **this PRD** (C) owns the `RunnerBusy` extension | queued |
| `plans/laptop-warm-verify-flock-orphan-prd.md` (landed, tasks 2306/2307) | supersedes for the shared-host case | `make_flock_contention_result`, `_alarm_verify_worktree_contention` | **this PRD** (C) — dated amendment pointer added to that PRD | queued |
| `plans/merge-throughput-multihost-verify-prd.md` (landed) | consumes | `VerifyRunnerConfig`, `verify_drift_check_every_n_lands` | multihost PRD owns; D only adds a config entry | wired |
| Policy follow-up PRD (unauthored; Leo) | produces for | task G's report and A's `--speculation` mode | **this PRD** delivers the inputs; the follow-up owns any code change | blocked-on-consumer (G1 resolution a) |

No reciprocal-ownership ambiguity.

## Contract (B+H) — cross-project host arbitration (C)

**Laptop side (`cli.py::verify_merge`):** before checkout/verify, acquire a
host-global merge-verify admission slot — a lock file under the existing
host-global admission directory convention (`shared.verify_admission`, default
`/tmp/df-verify-slots-<uid>`), key `merge-verify-host`, capacity from a new per-host
config field `verify_merge_host_max_concurrent: int = 1`. Bounded wait
(`verify_merge_host_wait_secs`, default 120). On timeout return a `RunnerBusy`
payload on stdout (JSON-native, round-trips through `result_to_dict`) carrying
`host`, `holder` (project_id + pid if known), `waited_secs`. The existing
per-project `.merge_verify.lock` is unchanged.

**Workstation side (`verify_runner.py`):**
- `RemoteRunner.run_merge_verify` parses `RunnerBusy` and raises
  `RunnerBusy(host, holder, retry_after)` (a sibling of `RunnerUnavailable`, not a
  `VerifyResult`).
- `HostAllocator` marks the host `busy_until = now + retry_after` (no quarantine,
  no reprobe storm) and `VerifyRunnerPool.dispatch` falls back to `LocalRunner`
  exactly as it does for `RunnerUnavailable`.
- The pool emits `EventType.verify_host_busy` with `{host, holder, waited_secs,
  fallback: 'local'}` (INV-2) and increments a per-host storm counter; above
  `verify_host_busy_l1_per_hour` (default 20) it files **one** L1 naming the rate
  (INV-4). It never files an L2 and never blocks the merge.
- `merge_verify.data.runner` records the runner that actually ran (`local`), with a
  new `fallback_reason: 'remote_busy'` key so occupancy accounting (task A) can
  attribute the fallback.

**Invariants:** a host runs at most `verify_merge_host_max_concurrent` merge
verifies across all projects; a busy host costs the waiting project at most the
bounded wait plus a local verify; reify's behaviour with Dark Factory idle is
byte-identical to today (lock uncontended, no new events).

### Boundary-test sketch (C's gate)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Two `verify-merge` invocations on one host (fixture project, subprocess ×2) | capacity 1, wait 2s | first runs; second returns `RunnerBusy` within wait + ε; neither touches the other's worktree |
| 2 | Pool receives `RunnerBusy` | one remote runner configured | dispatch falls back to local; `verify_host_busy` event with `fallback: 'local'`; `merge_verify.runner == 'local'`, `fallback_reason == 'remote_busy'`; no escalation filed |
| 3 | Storm | 25 busy results within an hour (fake clock) | exactly one L1 filed with the count; no L2 |
| 4 | Host not busy | capacity 1, nothing running | remote verify proceeds; no `verify_host_busy` event; drift check cadence unchanged |
| 5 | `RunnerUnavailable` still quarantines | unreachable host | unchanged quarantine + reprobe behaviour (regression pin) |

## Decomposition plan

All tasks `planning_mode=True`; `task_kind='normal'` unless stated. Priorities: A,
B, C, D high; E, F, G medium; H medium.

- **A — `scripts/merge_lane_throughput.py` + fixture test.** [high] Codify the
  baseline queries: landings/day; lead-time split by `merge_queued` →
  `merge_dequeued` → Σ`merge_verify.duration_ms` → `merge_finalized`; verify
  duration by `runner`; remote occupancy by `merge_heartbeat.hosts[].slot_state`
  LOCF integral (with the verify-duration-sum and raw-sample fractions shown
  alongside, since they disagree); `speculative_merge` depth distribution,
  `verdict_voided` rate, speculative-ahead landing share; `merge_heartbeat.depth`
  percentiles; `merge_attempt.outcome` and `merge_finalized.state` mixes; a
  `--chains` section (`landed_via_chain`, chain length) and a `--speculation`
  section (void rate by project). Flags `--project-root`, `--window 14d|30d|<iso>..<iso>`,
  `--json`. Read-only (`mode=ro`). A fixture `runs.db` under `scripts/tests/` with
  known answers. OPERATIONS.md gets a pointer. Modules: scripts, scripts/tests,
  OPERATIONS.md. **Signal:** `python scripts/merge_lane_throughput.py --project-root
  /home/leo/src/dark-factory --window 30d` prints the § Background rows for
  dark_factory within the stated caveats; the fixture test is green. Unlocks B, C,
  G, E, H.
- **B — Refresh the laptop Dark Factory checkout and per-host config.** [high]
  Over `ssh leo-laptop`: record the three dirty files and any commits not on
  `main` in the task result (decision 7); `git fetch && git reset --hard
  origin/main` only if nothing would be lost; `uv sync --all-packages`; `npm ci`;
  `uv run orchestrator verify-merge --help` exits 0. Review
  `~/.config/orchestrator/dark-factory-laptop.yaml` against `reify-laptop.yaml`
  and add the C config fields once C lands (B does not enable anything). Write
  `SETUP.md` § "Remote verify host" documenting the per-host layout (the one repo
  artifact). Modules: SETUP.md; `metadata.files: ["SETUP.md"]`. **Signal:**
  `ssh leo-laptop 'cd ~/src/dark-factory && git rev-parse HEAD'` equals `origin/main`
  at task time and the `verify-merge --help` probe exits 0; SETUP.md section
  present. Unlocks D1.
- **C — Cross-project host arbitration (B+H).** [high; `design_first=true`]
  Implement § Contract: laptop-side admission in `cli.py::verify_merge` +
  `shared.verify_admission`; `RunnerBusy` in `verify_runner.py`; pool fallback +
  `verify_host_busy` event + storm counter; config fields
  `verify_merge_host_max_concurrent`, `verify_merge_host_wait_secs`,
  `verify_host_busy_l1_per_hour`; boundary rows 1–5 as tests; amendment pointer in
  `plans/laptop-warm-verify-flock-orphan-prd.md`. Modules: orchestrator/cli.py,
  orchestrator/verify_runner.py, orchestrator/config.py, orchestrator/event_store.py,
  shared/verify_admission.py, tests. **Does not edit** `merge_queue.py`. **Signal:**
  boundary rows 1–5 green; `git grep verify_host_busy orchestrator/src` shows the
  emit site wired in `VerifyRunnerPool.dispatch`. Unlocks D1.
- **D1 — Enable the laptop for Dark Factory (config).** [high] Add to
  `dark-factory-orchestrator.yaml`: `verify_runners: [{name: laptop, ssh_host:
  leo-laptop, git_remote: leo-laptop, config_path:
  /home/leo/.config/orchestrator/dark-factory-laptop.yaml, enabled: true}]`,
  `verify_drift_check_every_n_lands: 20`, and C's arbitration fields; the laptop
  per-host yaml gets `verify_merge_host_max_concurrent: 1`. Modules:
  dark-factory-orchestrator.yaml. Depends on A, B, C. **Signal:** `uv run
  orchestrator check-config` (`cli.py::check_config`) accepts the file and reports
  one enabled verify runner; the K value logged at startup will be 2. Unlocks D2.
- **D2 — Deterministic deploy.** `task_kind='deterministic'`, `before_done={script:
  'scripts/restart-all-orchestrators.sh', args: [], timeout_secs: 900, target_unit:
  'orchestrator-dark-factory.service'}`, `always_escalates=false`. Depends on D1.
  **Signal:** `done_provenance kind='deterministic-deploy-scheduled'`; within 24h
  Dark Factory's `runs.db` has a `merge_verify` event with `runner: 'laptop'` and
  zero `verify_host_unreachable`. Unlocks E, F.
- **E — 14-day measurement after the laptop lever.** [medium] `metadata.milestone
  {mode: 'delayed', after_secs: 1209600}` (`docs/task-authoring.md` § 6) so it
  dispatches 14 days after D2 lands. Run A for both projects over the 14 days before and after D2; commit
  `plans/merge-lane-throughput-prd.measurements/<date>-laptop.md` with the
  comparison (lead p50/p90, wait p90, verify by runner, laptop occupancy,
  `verify_host_busy` count and any L1s, reify's rows unchanged-or-not). No threshold.
  **Signal:** the report exists and its dark_factory rows match `python
  scripts/merge_lane_throughput.py … --window <after>` output. Unlocks F.
- **F — Deep-chain canary on Dark Factory.** [medium] Depends on 3188 and E.
  Set `merge_deep.chain_cap: 6` in `dark-factory-orchestrator.yaml` via the same
  `before_done` script pattern as 3189 (green-tier: `reload_config`; if the reload
  report says `restart_required`, restart per D2's recipe). **Signal:** within 7
  days a `merge_finalized` event in Dark Factory's `runs.db` carries
  `landed_via_chain: true` (basis: queue depth p90 = 8, so chains of ≥ 2 form
  routinely; the deep-merge PRD's study validated cap 6). Unlocks H.
- **G — Speculation-waste diagnosis.** [medium] Investigation leaf. Explain, with
  event ids and code symbols for every claim: why reify voids 58% of speculative
  verdicts (`verdict_voided`, `chain_dead`) and lands ~2–4% speculative-ahead
  versus Dark Factory's 30% / 38–40%; the role of K = 2 placement, `gate_retry` /
  CAS retry rates, the disabled cross-check, and `select_probe_depth`. Commit
  `plans/merge-lane-throughput-prd.speculation-diagnosis.md` with a ranked proposal
  list for the policy PRD, each proposal naming the seam it lands in (κ or λ of the
  quality PRD). Extend A with anything the diagnosis needed that A lacked.
  **Signal:** the report exists; `python scripts/merge_lane_throughput.py
  --speculation` reproduces its headline rates. Consumer: the policy PRD's
  authoring session (Leo) and A.
- **H — 14-day measurement after the chain lever.** [medium] Same shape as E,
  delayed 14 days after F; report `…/<date>-chains.md` including `--chains`.
  **Signal:** the report exists and matches the script's output. Leaf.

## Out of scope

- Any change to dispatch or speculation *policy* (host-aware placement, depth
  selection, remerge strategy) — the policy follow-up PRD.
- Re-enabling reify's `verify_cross_check_remote_green`; reify's own canary (3189).
- Adding a second remote host; provisioning hardware.
- The quality PRD's Appendix A files.

## Open questions (tactical)

1. **Exact `RunnerBusy` wire shape** on stdout (a top-level key vs a `VerifyResult`
   with a sentinel category). Suggested: a distinct top-level object, since a
   `VerifyResult` with `passed=False` is what caused the L2 today. Decide in C.
2. **Where the storm counter lives** (pool instance vs `merge_liveness`'s
   streak helper). Suggested: reuse the streak helper (INV-5). Decide in C.
3. **Milestone anchor for H** — `delayed` fires relative to the task's own
   dependencies becoming satisfied (`after_secs`); confirm at filing that H's sole
   dependency is F so the 14-day clock starts at F, not E. Decide at filing.
4. **Whether B enables `persistent_merge_worktree` on the laptop for Dark Factory**
   (warm worktree, faster; reify's setting). Suggested: yes, once C's host-level
   lock makes the per-project lock irrelevant for arbitration. Decide in B after C.

---

## Corrections (2026-09-03, post-decompose)

This section is an append-only pointer, not a rewrite of the body above — the
same discipline task C is told to apply to
`plans/laptop-warm-verify-flock-orphan-prd.md`. The decompose walk and a
four-agent adversarial review found the following statements in this PRD to be
stale or unachievable. Each is corrected in
`plans/merge-lane-throughput-prd.capability-manifest.md` (§ Corrections and
§ Post-review findings) and in the filed task's own `details`. **Where this body
and the manifest disagree, the manifest is current.**

| § | Stale statement | Correction |
|---|---|---|
| Contract | the host-global admission dir "`shared.verify_admission`, default `/tmp/df-verify-slots-<uid>`" | that default was removed by task 2501 (`5392f4a357`); it is now **per-project**. C must mint and create its own host-global path. |
| Contract | "`VerifyRunnerPool.dispatch` falls back to `LocalRunner`" | true in the abstract, **unreachable on the remote merge path**: `merge_queue.py::_run_post_merge_verify` builds the remote pool with no `LocalRunner`, so `dispatch` re-raises. C's fallback design needs re-scoping. |
| Contract | the storm counter on the pool instance | the pool is constructed **per merge verify** (so a per-hour rate never accumulates) and takes **no `escalation_queue`** (so it cannot file the L1 at all). Counter belongs at `HostAllocator`'s worker lifetime. |
| Boundary sketch row 1 | "one host (fixture project, subprocess ×2)" | one fixture project shares one slot dir, so the row passes without exercising cross-project arbitration. Use **two distinct project roots**. |
| Decomposition, A | `--window 30d` reproduces § Background | that table is a **14-day** window except four explicitly-30d rows; and `--project-root` must be **repeatable** — the `events` table has no project column, so "void rate by project" is not derivable from one root. |
| Decomposition, D1 | `check-config` "reports one enabled verify runner" | it is an unknown-key linter and emits no runner inventory. Re-homed onto `OrchestratorConfig.enabled_verify_runners`. |
| Decomposition, D1/D2 | "the K value logged at startup will be 2" | `Harness._speculation_k` is computed but **never logged**. Use the `merge_verify` event with `runner: 'laptop'`. |
| Decomposition, D2 | "zero `verify_host_unreachable`" | that event is deduped behind an open L1 and skipped when the queue is unwired, so zero ≠ reachable. Read it as *no open `verify_host_unreachable` L1 for `leo-laptop`*. |
| Decomposition, F | "`landed_via_chain: true`" | it is an **int** (1 per item landed via chain), and F's 7-day observation cannot gate F, which completes on its deploy script's exit code. |
| Cross-PRD table | `merge-lane-quality-prd.md` — **parallel** | true for this PRD's own files, but **F and H are transitively gated behind it**: F depends on 3188, whose dependencies are `[3186, 5036]`, and 5036 is that PRD's package-move gating anchor. |

### RULED 2026-09-03 — task C is cancelled (superseded, kept for the record)

`plans/cpu-load-robust-verify-prd.md` § 6 (RED-TIER, human decision) and
`plans/integration-test-lane-prd.md` § 11 both rule out a **host-global
cross-project verify admission semaphore** — the mechanism task C proposes — as
*"TRIED-AND-REJECTED, not merely deferred … Do not revisit this as an option"*,
after reify's 30-min+ cargo verifies starved dark_factory almost entirely.
Neither PRD is cited anywhere in this document or its manifest.

C's shape is arguably distinguishable — merge-only, capacity 1, bounded wait,
and a fall back to local — but the local fallback is exactly the property the
second row of the table above shows to be broken today. Task 5052 carries a
STOP-AND-RECONCILE block instructing it to escalate rather than build through
the ruling.


### The ruling, and the premise that turned out to be false

**Task C (5052) is cancelled.** Leo's question — *"on the local host we run reify
and Dark Factory verifies concurrently, lots of each. Why can't we run them
concurrently on the remote host?"* — has no good answer, because **we can**.

The premise C rested on, stated in § Background as *"two projects sharing the
laptop under today's code would either thrash it or page Leo"*, is wrong on the
paging half:

- The laptop-side merge-verify lock is scoped to the **project**:
  `git_ops.py` sets `self.worktree_base = (project_root / config.worktree_dir).resolve()`,
  and both contention gates in `cli.py::verify_merge` key on that base. So on
  `leo-laptop`, reify's locks live under `/home/leo/src/reify/.worktrees/` and
  dark_factory's under `/home/leo/src/dark-factory/.worktrees/` — **different
  files, which cannot collide**. The born-at-L2 `FLOCK_CONTENTION_CATEGORY`
  branch is a **within-project orphan detector** (that is what
  `laptop-warm-verify-flock-orphan-prd.md`, tasks 2306/2307, is about), never a
  co-tenant detector.
- Nothing else serializes merge verifies across projects on **any** host:
  `shared/src/shared/verify_admission.py::acquire_task_slot` returns immediately
  for any role outside `{task, background}` — *"C-merge-priority: merge can never
  be starved by task"*. That is precisely why reify and dark_factory merge
  verifies already run concurrently on the workstation, and the same code runs on
  the laptop.

So the two projects will simply run concurrently on the laptop, as they do
locally, with no new mechanism. Building C's arbitration would additionally have
**created** the host-global semaphore the RED-TIER ruling above forbids, inside a
system whose per-project design is what that ruling calls correct. Cancelling C
complies with the ruling rather than fighting it.

**Retracted with it:** the decompose session's estimate of "1–3 blocked merges and
L2 pages per day" without arbitration, which was arithmetic on the false premise.

**Still live, independent of the cancellation:** the cross-PRD table's κ row is
wrong. C would not have made the merge-queue contention branch unreachable (the
remote pool is built with no `LocalRunner`, so `dispatch` re-raises), and with C
cancelled the branch is plainly still live and still useful. **κ must not delete
`is_flock_contention_failure`.**

**What now guards the risk:** the residual concern is CPU contention on a laptop,
which degrades verify *duration* and cannot block a merge or file an L2 — and is
not cappable via `verify_admission_pytest_n`, since the merge role is never
`-n`-capped. Task **E** carries it as a headline measurement: reify's laptop
verify p50/p90 against its 17.4 / 20.6 min baseline, dark_factory's against its
40.7 / 52.4 min local baseline, and the overlapping-span contention rate nobody
has measured. If the host turns out oversubscribed, arbitration becomes a
decision to make **with data** — in some shape other than the one ruled out.

Revised DAG: **A → B → D1 → D2 → E → F → H**, with **A → G** alongside. No
RED-TIER conflict anywhere in it.
