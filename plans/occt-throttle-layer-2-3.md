# OCCT throttle — Layer 2/3 follow-up

**Status:** active — author 2026-05-28
**Repos touched:** `dark-factory` (orchestrator, dashboard, new sampler service), `reify` (`scripts/verify.sh`)
**Supersedes / extends:** Layer 1 shipped as `reify` main `533c7def96` (2026-05-28) — semaphore neutered, default `REIFY_OCCT_MAX_CONCURRENCY` raised 4 → 32.

## Goal

When the orchestrator spawns reify verification on the 32-core dev workstation, cargo processes run at a CPU/IO priority **strictly below** interactive user processes (shells, Firefox, VS Code), so the user retains a usable workstation during ~30 h/week of interactive dev that overlaps with FD-driven verify load. Merge-queue verifications get a priority bump over per-task verifications, so reify merges still complete promptly.

Separately, the dark-factory dashboard gains a load card surfacing PSI (cpu/memory/io), OCCT slot queue depth, concurrent verify count, and aggregate verify RSS — so the next high-load incident is diagnosable at a glance instead of via ad-hoc `/proc/locks` inspection.

End-user observable outcomes:
- During heavy verify load, an interactive shell (`time ls -R ~/src/reify/target | wc -l`) stays within ~2× its idle-host wall time, not the 10×+ degradation that occurred 2026-05-28 before Layer 1.
- The dark-factory dashboard's load card renders live values + 5-min sparklines that update every 5 s.

## Background — why Layer 2/3

A `/deb` session on 2026-05-28 traced reify task 4000's L2 escalation (`esc-4000-39`) to a positive-feedback collapse in `reify/scripts/cargo-test-occt-gated.sh`. The wrapper's load-based formula `N = clamp(nproc - load_1m_int, 1, MAX_CAP)` collapsed N to 1 under sibling-worktree load because it measured run-queue length, which is itself fed by the work being throttled. Every concurrent wrapper then contended for slot-1 alone even when slots 2..MAX_CAP were idle.

Five Whys with the user produced the framing: **the semaphore was a *resource governor* dressed as a *concurrency limiter*, and it measured the wrong resource.** Layer 1 (already shipped) breaks the feedback loop by removing the load-based reduction and raising the cap; on a 32-core box the semaphore is now effectively a no-op. That fixes the immediate collapse but doesn't address the underlying constraint the semaphore was groping toward: **the user needs interactive responsiveness on a workstation that also runs FD ~24/7**.

User constraints captured verbatim during design (relevant excerpts):
> "OCCT tests must be single thread within a process (OCCT is believed to be multithread concurrency unsafe), but should be multiprocess concurrency safe. If they aren't then that's a test bug that should be fixed at the root."
>
> "We should only throttle concurrency for throughput if CPU-saturated, or if PSI (memfull) indicates that doing so will improve throughput."
>
> "Throughput first, but responsiveness a close second. The workstation runs FD ~24/7, and is my interactive dev box about 30h/week. … Consider nice (or similar) for non-interactive loads?"
>
> "OCCT is reify specific. Reify merge verify should get priority."

This PRD encodes the conclusion that **for the responsiveness requirement, `nice`/`ionice` is the right primitive, not a smarter semaphore.** Concurrency throttling for throughput is deferred (Layer 1's cap-raise is the active answer; revisit only if PSI evidence later shows the kernel scheduler can't manage 32-core saturation gracefully). The dashboard card is the evidence-gathering substrate for that future revisit.

## Sketch of approach

Two independently-decomposable units bundled in one PRD because they share the same incident origin and the load card is the diagnostic instrument that will validate / disprove future need for smarter throttling:

### Unit 1 — role-aware nice/ionice (WP-A + WP-B)

Cross-repo seam: a `DF_VERIFY_ROLE` environment variable.
- **Producer (dark-factory orchestrator):** sets `DF_VERIFY_ROLE=merge` in the env of merge-queue verify subprocesses, `DF_VERIFY_ROLE=task` in the env of per-task verify subprocesses. Plumbed via the existing `verify_env` merge in `orchestrator/src/orchestrator/verify.py:1211` (`_resolve_verify_env`) and the merge-queue verify call at `merge_queue.py:1935` (`_verify_and_advance`).
- **Consumer (reify `scripts/verify.sh`):** reads `DF_VERIFY_ROLE` (default = `task` when unset), prefixes the cargo invocations it builds with the appropriate priority wrapper. Both runtime invocation and `--print-plan` dry-run emit the prefix.

Priority levels (rationale in `## Resolved design decisions` below):
| Role | Wrapper | Why |
|---|---|---|
| `task` (default) | `nice -n 15 ionice -c 2 -n 7` | Heavy headroom below interactive (nice 0); IO best-effort at lowest priority. Per-task verifies are the bulk of background load. |
| `merge` | `nice -n 5` (no ionice change) | Still below interactive but above per-task, so merge-queue verifies complete promptly under multi-task load. No IO de-prioritization — merge verify must not get IO-starved by long-running per-task verifies. |

### Unit 2 — load sampler + dashboard card (WP-C)

A small Python systemd-user-timed sampler that writes to a new SQLite DB at `data/load-samples.db`, plus a dashboard backend module that reads it and a frontend card under the redux SPA's overview tab.

**Sampler design** — exploits PSI's intrinsic kernel-side windowing to avoid re-inventing a metrics pipeline:
- **PSI metrics (`/proc/pressure/{cpu,memory,io}`):** kernel already computes avg10/avg60/avg300 exponentially-weighted windowed averages. Sampler reads these directly at 5 s cadence; no re-windowing needed (the kernel is the "existing tool").
- **Non-PSI metrics** (OCCT slot queue depth, concurrent verify.sh count, aggregate verify RSS): sample at 5 s; keep a 60-sample (=5-min) ring per metric in-process; persist both the instantaneous sample and the 5-min trailing mean/max.
- **OCCT slot queue depth:** count `cargo-test-occt-gated` bash processes whose FD 9 is not open (i.e. waiting for the semaphore, not holding it).
- **Concurrent verify.sh count:** count processes whose argv[0] basename matches `verify.sh`.
- **Aggregate verify RSS:** sum RSS over the verify.sh process tree.

Storage: `data/load-samples.db`, single table `samples(ts, metric, value, window_mean, window_max)`. Retain 24 h, vacuum daily. Dashboard backend queries the latest sample + the last 60 samples (=5 min) per metric on each card refresh (5 s polling).

**Dashboard card:** new section in `dashboard/src/dashboard/static/redux/tab_overview.jsx` (or a new tab — decided during decomposition; see open questions). Renders live values + 5-min sparkline per metric. Refresh cadence matches the sampler's 5 s tick.

## Resolved design decisions

1. **`nice`/`ionice` over a smarter semaphore.** A measurement-driven semaphore (PSI-feedback, CPU-saturation-aware) is rejected for v1 because:
   - The kernel's CFS scheduler already manages CPU contention competently when processes have correct nice levels; the user's explicit constraint was interactive responsiveness, not throughput, and `nice` directly addresses that.
   - Layer 1's cap-raise gives the kernel scheduler the multiprocess concurrency it wants. Throughput-throttling is deferred until PSI evidence (from WP-C) shows otherwise.
   - A PSI-feedback semaphore reintroduces the Layer-1 failure mode (measuring a quantity affected by the work being throttled) unless carefully designed; not worth the risk for an unproven need.

2. **`task` is the unset default.** When `DF_VERIFY_ROLE` is absent (manual `verify.sh` invocation, third-party tooling), use `task` priorities. The user's local `git verify` runs are interactive — but they're explicitly run by the user, who can override via `DF_VERIFY_ROLE=interactive` (a future tier; see open questions) or just by not nicing their own shell. The orchestrator is the load source the priorities are sized for, so its default is the safe default.

3. **No `DF_VERIFY_ROLE=interactive` tier in v1.** Three tiers (merge/task/interactive) is YAGNI until a concrete consumer asks for it. Easy to add later — the env-var protocol is open-ended.

4. **Merge role uses `nice -n 5` not `nice -n 0`.** Even merge-queue verifies are non-interactive work and should yield to actual interactive processes. nice 5 keeps merge above per-task (nice 15) and below user shells (nice 0); 10-step separation gives CFS clear distinction. No `ionice` change for merge — merge verify must not be IO-starved by long per-task verifies.

5. **PRD lives in `dark-factory/plans/`, not `reify/docs/prds/`.** Bulk of the work is in dark-factory (orchestrator env-plumbing, dashboard card, sampler service); the reify side is one wrapper change. The reify overlay's PRD path convention (`reify/docs/prds/<vM_N>/<slug>.md`) covers reify-internal PRDs, not cross-repo follow-ups led from dark-factory.

6. **DF does not need a generic semaphore feature.** Analyzed during design: dark-factory's own verify is pytest+ruff+pyright with no process-global state, so multiprocess concurrency safety is intrinsic and a semaphore would add nothing. The minimal DF-side support reify needs is the `DF_VERIFY_ROLE` env var — nothing else.

7. **Sampler is a separate systemd-user service, not embedded in dashboard.** Independent lifecycle: sampler survives dashboard restarts (which are frequent — manual restarts after backend merges per `feedback_dashboard_restart_after_backend_merge`); dashboard survives sampler bugs. Sampler is small enough to maintain as a standalone service.

8. **Storage = SQLite at `data/load-samples.db`.** Matches the dark-factory pattern (`data/orchestrator/runs.db`, `data/reconciliation/*.db`, `data/burndown/*.db`). No new persistence layer.

9. **WP-C scope excludes alerting / thresholds.** The card is diagnostic, not alarming. If sustained high PSI starts causing problems, a follow-up PRD adds alerts then.

10. **WP-D (merge_request "not something we can merge" diagnostic) is excluded from this PRD.** Different consumer (operator's merge-attempt terminal output vs interactive responsiveness + dashboard), weak coupling to the rest. Filed separately — see hand-off at the end.

## Pre-conditions for activating

None outside this PRD. Substrate verified during authoring:
- `reify/scripts/verify.sh` exists with `--print-plan` flag (verified at `scripts/verify.sh:93`).
- `orchestrator/src/orchestrator/verify.py` has `_resolve_verify_env` (line 1211) and a subprocess-spawn site (line 1360) that accepts `env=verify_env`.
- `orchestrator/src/orchestrator/merge_queue.py` has `_verify_and_advance` (line 1935).
- `/proc/pressure/{cpu,memory,io}` are readable on the target host (PSI enabled — kernel 6.17).
- `dashboard/src/dashboard/data/` is the established backend-module location.
- `dashboard/src/dashboard/static/redux/tab_overview.jsx` is the established overview-tab location.
- `nice` and `ionice` exist (coreutils + util-linux; verified on the target host).
- SQLite + WAL pattern matches existing `data/**/*.db` conventions.

## Cross-PRD relationship

No external PRD seams. The internal cross-repo seam (`DF_VERIFY_ROLE` env var) is owned by this PRD and is fully specified in `## Sketch of approach`.

## Decomposition plan

Tasks labelled by Greek letter; actual fused-memory task IDs assigned at decompose time. Per the `feedback_split_multi_package_tasks` rule, the WP-B tasks are split per spawn site rather than bundled into a single multi-module task that would blow the architect budget.

### Unit 1 — role-aware nice/ionice

- **α — reify: wrap cargo invocations in `verify.sh` with role-aware nice/ionice.**
  - Repo: `reify`.
  - Files: `scripts/verify.sh`.
  - Implementation: parse `DF_VERIFY_ROLE` (default `task`); prepend `nice -n 15 ionice -c 2 -n 7` (task) or `nice -n 5` (merge) to each cargo invocation in both the print-plan and run paths; reject unknown values with exit code 64 and a diagnostic.
  - Leaf signal (user-observable): `DF_VERIFY_ROLE=task scripts/verify.sh --print-plan test` stdout contains literal substring `nice -n 15 ionice -c 2 -n 7 ` immediately preceding each cargo command; `DF_VERIFY_ROLE=merge ...` contains `nice -n 5 ` preceding each cargo command and does *not* contain `ionice`; unset env var produces task-default output; unknown role exits 64 with stderr `verify.sh: ERROR — unknown DF_VERIFY_ROLE 'X'`.
  - Prereqs: none.

- **β — dark-factory: set `DF_VERIFY_ROLE=merge` at the merge-queue verify spawn.**
  - Repo: `dark-factory`.
  - Files: `orchestrator/src/orchestrator/merge_queue.py` (`_verify_and_advance` path); possibly `orchestrator/src/orchestrator/verify.py` (`_resolve_verify_env`) if the env merge happens centrally.
  - Implementation: inject `DF_VERIFY_ROLE=merge` into the `verify_env` passed to the verify subprocess on the merge-queue path. Add a unit test asserting the env var is present in the resolved env for the merge-verify path.
  - Code-level signal (intermediate; consumed by δ): unit test in `orchestrator/tests/` asserts `verify_env['DF_VERIFY_ROLE'] == 'merge'` for the merge-queue spawn (the test name is tactical — left to the implementer).
  - Prereqs: none.

- **γ — dark-factory: set `DF_VERIFY_ROLE=task` at the per-task verify spawn.**
  - Repo: `dark-factory`.
  - Files: `orchestrator/src/orchestrator/verify.py` (per-task verify path) — likely the same `_resolve_verify_env` helper plus the per-task spawn site.
  - Implementation: inject `DF_VERIFY_ROLE=task` into the per-task verify env. The wrapper's unset-default makes this strictly belt-and-braces, but explicit is correct: a future change to the unset default (or removal of `task` as default) shouldn't silently regress per-task priorities. Unit test asserts the env var.
  - Code-level signal (intermediate; consumed by δ): unit test in `orchestrator/tests/` asserts `verify_env['DF_VERIFY_ROLE'] == 'task'` for the per-task spawn.
  - Prereqs: none.

- **δ — integration gate: end-to-end `DF_VERIFY_ROLE` → reify verify-plan.**
  - Repo: `dark-factory` (test lives in `dark-factory` because it crosses both repos and the dark-factory test suite has the orchestrator-side stubs; uses `/home/leo/src/reify/scripts/verify.sh` as a fixture path or via a configured reify root).
  - Implementation: integration test that constructs the orchestrator's verify env via `_resolve_verify_env` for both the merge and per-task paths, invokes `verify.sh --print-plan test` with each, and asserts the resulting stdout contains the expected nice prefix per role. This closes the C-as-integration-gate loop for β + γ — they would otherwise be code-level-only tasks vulnerable to the G2 "synthetic-input unit test" failure mode.
  - Leaf signal (user-observable): integration test `test_role_env_propagates_to_reify_verify_plan` passes — running the orchestrator's merge-queue verify-env producer + reify's `verify.sh --print-plan` end-to-end produces nice-prefix output that matches the role.
  - Prereqs: α (reify wrapper), β (merge env), γ (task env).

### Unit 2 — load sampler + dashboard card

- **ε — load sampler service.**
  - Repo: `dark-factory`.
  - Files: new package under `sampler/` or `dashboard/src/dashboard/sampler/` (decompose decides — see open questions); new `dark-factory-load-sampler.service` and `.timer` files alongside the existing dashboard units; new SQLite migration creating `data/load-samples.db` with `samples` table.
  - Implementation: Python sampler script reading `/proc/pressure/{cpu,memory,io}` (parses the `some avg10= avg60= avg300=` and `full avg10= avg60= avg300=` lines), counting OCCT slot queue depth (waiting `cargo-test-occt-gated` bash processes — FD 9 not open), counting concurrent `verify.sh` processes, and summing verify-tree RSS. In-process 60-sample ring buffer per non-PSI metric for trailing 5-min mean/max. Writes one row per metric per 5 s tick to SQLite. Daily-vacuum task retains 24 h. Systemd-user timer fires every 5 s.
  - Code-level signal (intermediate; consumed by ζ): `systemctl --user is-active dark-factory-load-sampler.timer` returns `active`; after 30 s of runtime, `python3 -c "import sqlite3; print(sqlite3.connect('data/load-samples.db').execute('select count(distinct metric) from samples where ts > strftime(\"%s\",\"now\")-60').fetchone())"` returns ≥ 7 metrics (3 PSI × {some,full} avg10 = 6 + queue depth + verify count + RSS = 9; allow ≥ 7 to tolerate transient missing metrics on idle hosts). G2 escape hatch (C-as-integration-gate): foundation-style task; user-observable leaf for unit 2 is η (the dashboard card).
  - Prereqs: none.

- **ζ — dashboard backend: `/api/load` endpoint.**
  - Repo: `dark-factory`.
  - Files: `dashboard/src/dashboard/data/load.py` (new module matching the existing `data/` pattern); `dashboard/src/dashboard/app.py` (route registration).
  - Implementation: read latest sample + last 60 samples per metric from `data/load-samples.db`; return JSON `{metric_name: {current: float, sparkline: [60 floats], window_mean: float, window_max: float}}`. Match the existing dashboard data-module shape (see `data/costs.py`, `data/burndown.py`).
  - Leaf signal (user-observable): `curl http://localhost:<dash-port>/api/load | python3 -m json.tool` returns a JSON object containing all expected metric keys, each with a `current` float and a `sparkline` array of length up to 60.
  - Prereqs: ε.

- **η — dashboard frontend: load card.**
  - Repo: `dark-factory`.
  - Files: `dashboard/src/dashboard/static/redux/tab_overview.jsx` (extend) or new `tab_load.jsx` (decompose decides — see open questions); reuse the existing sparkline component from `charts.jsx`.
  - Implementation: new card section showing each metric's current value (formatted: PSI as percentage avg10, RSS in GiB, queue depth and verify count as integers) alongside a 5-min sparkline. Poll `/api/load` every 5 s. Empty-data state shows "—" not synthetic data (per `feedback_redux_no_synthetic_data`).
  - Leaf signal (user-observable): Firefox at `http://localhost:<dash-port>/redux` shows a "Host load" card on the overview tab; each metric row renders the formatted current value (per metric type — PSI percentage, RSS in GiB, queue/verify counts as integers) and a sparkline whose data points correspond to the last 60 samples for that metric (sparkline path is non-empty SVG when ≥ 2 samples exist; the values may be zero on an idle host but the rendering is non-stub). Under verify load induced during testing (run reify `verify.sh all` in a loop for 60 s) the sparklines visibly rise. Two screenshots — idle baseline and induced-load — are the close criterion, paired with a network-trace observation that `/api/load` is being polled every 5 s.
  - Prereqs: ζ.

### Dependency DAG

```
α ──┐
β ──┤
γ ──┤
    └──> δ  (integration gate; unit 1 leaf)

ε ──> ζ ──> η  (unit 2 leaf)
```

Units 1 and 2 are independent.

### Out of scope for this PRD

- **Layer 1** (already shipped: `reify` main `533c7def96`, 2026-05-28).
- **WP-D — merge_request error diagnostic.** Filed as a separate PRD; see hand-off below.
- **PSI-feedback / CPU-saturation-aware semaphore.** Deferred until WP-C data justifies it.
- **`DF_VERIFY_ROLE=interactive` tier.** Deferred until concrete consumer asks for it.
- **Alerting / threshold-based notifications from the load card.** Card is diagnostic-only in v1.
- **Generic semaphore feature in dark-factory.** Analyzed and rejected during design (decision 6).
- **`reify/scripts/cargo-test-occt-gated.sh` further refactoring.** Layer 1 left it as an effective no-op on a 32-core box; revisit only if WP-C evidence shows OCCT serialization is a problem on smaller hosts.

## Open questions (tactical — defer to implementation)

1. **WP-C sampler package location** — under a new top-level `sampler/` or under `dashboard/src/dashboard/sampler/`. Suggested resolution: top-level `sampler/` because lifecycle is independent of the dashboard (decision 7); pyproject within `sampler/` or absorbed into the root project. Decide during ε.
2. **WP-C card placement** — extend `tab_overview.jsx` with a new section, or create a dedicated `tab_load.jsx`. Suggested resolution: start as a card on `tab_overview.jsx` (lowest friction); split into its own tab only if it dominates the overview. Decide during η.
3. **WP-C OCCT slot-queue detection robustness** — counting bash processes by FD 9 closed-ness is a heuristic that depends on the wrapper's exact `flock` invocation shape. If `cargo-test-occt-gated.sh` is rewritten or removed (out of scope here), the metric becomes meaningless. Suggested resolution: implement as-is, document the dependency in the sampler module's module docstring; a future rewrite of the wrapper that breaks this metric will surface as the metric going to zero on the dashboard. Decide during ε.
4. **β/γ implementation — single `_resolve_verify_env` change or two separate call-site changes?** `_resolve_verify_env` (verify.py:1211) is the central merge point but doesn't currently know the role. Cleanest: extend it to take a `role` arg and inject `DF_VERIFY_ROLE` there. Suggested resolution: do this once; β and γ then differ only in the arg they pass at their respective call sites. Decide during β.
5. **WP-C trailing-window storage** — record per-sample window_mean/window_max in the DB, or compute on-read from the last 60 rows? Storing them is denormalized but cheap; computing on read is cleaner but adds query complexity. Suggested resolution: store them — query simplicity wins, and the duplication is a few bytes per row. Decide during ε.

## Hand-off — WP-D (separate PRD)

The "merge_request returns bare `not something we can merge`" issue from 2026-05-28 needs its own PRD (decision 10). Briefing notes for that PRD:

- Symptom: during Layer 1 ship, two `mcp__escalation__merge_request` attempts on branch `task/occt-throttle-layer-1` against base `main` returned the bare git error string with no diagnostic; forced a `--no-verify` direct merge (authorized in-session per `decision_merge_no_verify_with_orchestrator_live.md`).
- Suspected cause: `SpeculativeMergeWorker` race — temp merge worktree created at a base SHA from which the task branch isn't reachable (speculation built on a parent commit that wasn't actually `main` HEAD at merge time, or the task branch ref didn't resolve in the temp worktree).
- Desired outcome: `merge_request` failure surface includes (a) the base SHA the merge attempted against, (b) whether that was the speculative base or actual main, (c) whether the task branch ref resolved in the temp worktree, and (d) ideally a retry against actual `main` HEAD when the speculative attempt fails with "not something we can merge".
- Recommended scope: enough diagnostic to make the next incident self-explanatory; the retry-against-actual-main behaviour is nice-to-have but the diagnostic is the hard requirement.
