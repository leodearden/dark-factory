# Gate 6673 (ε): DF remote task-leg parity window

PRD: `plans/remote-task-leg-verify-prd.md` (ε). Escalation `esc-6673-1`.

**Window: 2026-11-10T11:22Z → 2026-11-13T11:22Z** (three days, Leo 2026-10-31).
It opened with the `reload_config` that applied commit `002d17be15`. All four
keys landed under `applied` and nothing was `restart_required`.

| key | value |
|---|---|
| `verify_task_host_policy` | `remote_modules` |
| `verify_task_remote_modules` | `[orchestrator]` |
| `verify_task_remote_red_confirms_locally` | `true` |
| `verify_task_remote_cross_check_every_n` | `5` |

## Step 1: laptop pre-flight (re-measured 2026-11-10)

- **Which yaml.** `verify-task` reads `--config /home/leo/.config/orchestrator/dark-factory-laptop.yaml`
  (`verify_runners[laptop].config_path`). The `verify_admission_pytest_n: "8"` seen at
  leo-laptop `~/src/dark-factory/dark-factory-orchestrator.yaml` is the laptop's git copy of
  the WORKSTATION yaml, which remote verify does not read. `task_verify_spec.py` classes
  `pytest_n` as host-shaped and not carried.
- **Laptop yaml.** Added `verify_admission_task_slots: 1` and `verify_admission_pytest_n: "8"`
  (backup `*.bak-20261110-6673`). `load_config` reads `1 / '8' / verify_use_cgroup_scope True /
  preprovision set`.
- **Checkout.** Per-SHA gate checkouts under `~/src/dark-factory-gate/` (6565); `verify-task`
  present.
- **Drop-ins.** One drop-in each for DF, reify, SC and SCP. A hand-started
  `df-verify-dark-factory-c889f090-…scope` read memory.high/max/swap.max = 24G/40G/32G.
- **CPU weights.** app.slice 100, background.slice 30.
- **Power.** The two unclean shutdowns (11-08, 11-10) were battery exhaustion on a 30 W
  USB-C supply (deb session `deb-dark_factory-1082232`). Before the window, the PDOs were
  re-read after Leo swapped the charger: 20 V/5 A (100 W), battery 100 %.
- **Disk.** Free space was 17–18G (98 %). Leo approved the reclaim and it ran:
  - 105 clean reify worktrees removed with `git worktree remove`: eval runs, numeric task
    worktrees and orphan `_merge-<hash>` worktrees;
  - 6 unregistered leftovers removed;
  - 9 dirty worktrees skipped;
  - sccache untouched;
  - `uv cache clean` did not run, because its lock was held by a live uvx MCP process
    and a running DF verify.

  Free space after the reclaim: **58–59G (94 %)**. The log is leo-laptop
  `~/gate-6673-disk-reclaim-20261110.log`.
- **Memory envelope.** `verify_scope_peaks.py poll` ran from 08:01Z (pid 526380, leo-laptop
  `~/verify-peaks-20261110-6673.jsonl`, left running through the window). By 11:20Z it held
  157 scopes, all DF; reify's scheduler is halted, and SC/SCP were idle. Figures: DF
  sum-of-legs max **17.2 GiB**, p90 8.0 GiB; MemAvailable at scope start 52–56 GiB of 61.

## Step 2: hand-run through the production dispatcher

- **Setup.** `TaskVerifyDispatcher` was built exactly as `harness.py` builds it, with the
  step-4 policy overlaid in-process. Scratch branch tip `f2916ec6ff`: a docstring-only
  edit in `orchestrator/` (branch since deleted).
- **Sidecar.** `data/verify-logs/6673/attempt-1.dispatch-20261110T111225_124118Z.json`
  reads `decision: remote`, `runner: laptop`, `reason: remote`.
- **Verdict.** Passed: remote_wall 56.3 s, remote_wait 0.0 s.
- **Q6 confirmed.** The applied env equals `{**host effective_verify_env, **spec verify_env}`.
  The live laptop has no host-only keys; a synthetic host-only key was kept, and the spec
  won on overlap.
- **Q10.** The gate checkout for `dfe6161a` was built in about 5 s.
- **Recorded.** `x_step2_passed_at` and `x_step2_sidecar` are set on task 6673.

## Step 3: baseline (DF, first full week after 6580: 2026-10-21T03:05Z → 10-28T03:05Z)

Re-run 2026-11-10 from runs.db. The output is byte-identical to the 11-09 pre-measurement.

| figure | value |
|---|---|
| verify-phase residence p50 (intervals ≥5 s) | **2.44 h** (n=588; p90 4.83 h) |
| verify-phase residence per task, total p50 | 5.27 h (n=261) |
| orchestrator-leg local wall p50, all test legs | **142 s** (n=354; p90 1,452 s) |
| ↳ full-suite / file-scoped | 1,205 s (n=108) / 104 s (n=246) |
| archived `slot_wait_secs` p50 | **not computable**: `data/verify-logs` starts 2026-11-04 and `workflow_verify.slot_wait_secs` is 0 for all 567 rows |
| proxy: admission wait per plan (sum over legs / longest leg) | p50 9,686 s / 7,760 s (n=609) |

## Step 5: to measure at window end (2026-11-13T11:22Z)

Lead with:
- verify-phase residence (runs.db) against the 2.44 h baseline;
- remote_wait_secs from the dispatch sidecars.

Then the three trip-wires:
1. Laptop wall per orchestrator leg against local. Laptop: `remote_wall_secs` in the
   sidecars. Local: the 142 s baseline, and in-window local legs.
2. RU/no-verdict rate: sidecars with `reason ∈ {runner_unavailable, infra_transient}` over
   eligible verifies (all sidecars except `not_eligible`).
3. Parity disagreements, both directions:
   - `red_confirmed_locally.local_passed == true` (laptop red, local green);
   - `cross_check.remote_passed != local_passed`.

   Re-run each disagreement before counting it, and note whether reify's scope was live.
   Report raw and adjudicated counts.

Also report:
- the share of eligible verifies run remote;
- drift-check deferrals;
- laptop free space against 59G;
- merge_verify wall on the laptop while a task leg overlaps (poller JSONL + sidecars);
- drift splits, and whether a task scope was live;
- reify:8430's report alongside.

## Step 5: measured (2026-11-13)

Measured read-only by the L2 watcher at about 12:30Z. The full report and raw outputs are kept with
the watcher's runtime evidence (data/sitting/evidence/6673/, untracked). Headline:

- 169 eligible task verifies: 98.2% ran remote; runner-unavailable / no-verdict 0; remote_wait p50 0 s.
- Same-tip laptop/local wall p50 0.67 (n=45). Verify-phase residence p50 0.15 h, against the 2.44 h
  baseline. Most of the gain predates the window (slot raise, paused sibling projects).
- Parity: 2 raw splits, 0 adjudicated. 7800: a local pytest timeout at load 196, laptop retry green.
  6393: a real leaked-MagicMock bug the laptop caught, later fixed by ebdd5dfc5d. reify was never live.
- Four task_slots reloads in the window (38a200d213, 3763a4c54d, d442a16c49, c52c6ee8c7). Every
  headline holds in each of the five segments.
- Costs: laptop merge_verify p50 22.5 min while a task leg overlaps, 19.3 min otherwise (16.2-16.7 min
  before the window). Laptop free disk 59G down to 43G, with the 16G unattributed.

## Step 6: ruling

Leo, 2026-11-13 (sitting item 561, option A):
- keep remote task legs on (`verify_task_host_policy: remote_modules`, `[orchestrator]`);
- `verify_task_remote_red_confirms_locally: false`;
- keep `verify_task_remote_cross_check_every_n: 5` until task 7030 lands (until then the laptop can
  silently skip three reify-dependent guards), then set it to 0;
- task eta (the binary pressure hold) is NOT flipped: there is no overload signal;
- file a task to attribute and reclaim the laptop's 16G disk loss.
