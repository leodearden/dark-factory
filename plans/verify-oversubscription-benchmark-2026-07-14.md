# Benchmark: `-n` speedup curve + `verify_admission_task_slots`/`verify_admission_pytest_n` defaults (T6)

PRD: `plans/verify-oversubscription-control-prd.md` task T6 (follow-up to T1-T4,
merged; see `plans/verify-oversubscription-control-prd.capability-manifest.md`).
Depends: T4 integration gate (task 2392, `4274b83666 Merge task/2392 into main`).
Base commit for this benchmark: `89c740eaa8` (task 2394 steps 2-7, the
`verify_admission_pytest_n` knob + mutator + wiring, already landed).

Date: 2026-07-16. Host: 32-core (`nproc`=32), Python 3.13.9, pytest 9.0.3,
pytest-xdist 3.8.0, `orchestrator/pyproject.toml:100` addopts
`-n auto --dist loadgroup --max-worker-restart=0`.

## TL;DR

- **No clean idle window was obtainable this session** (see "Idle-window
  check" below) — this is the exact contingency `plans/verify-oversubscription-control-prd.md`
  T6 and this task's plan anticipated. Per that documented fallback,
  `verify_admission_pytest_n` **stays at its already-landed default `'auto'`**
  (behavior-preserving, byte-identical to today) — **no config change from
  this report**.
- `verify_admission_task_slots` default **`1` is CONFIRMED** — today's data
  reconfirms (more strongly than the PRD's original citation) that pytest-xdist
  parallelism within one verify has strongly diminishing returns past ~16
  workers, so serializing task-role verify invocations sacrifices little.
- PRD §9 item 3 (should `background` be excluded from `-n` capping?):
  **recommend NO** — `background` should stay **included** in the cap (as
  already wired: `role in {'task', 'background'}`). No change needed.
- Despite the contention caveat, `-n 16` was at-or-near-optimal in **every
  single measurement** performed today (full-suite pair and a 6-point subset
  sweep) — it is the strong candidate value for a future clean-window
  re-benchmark to confirm before it becomes the landed default.

## Method

### Prerequisite: venv sync

`uv sync` from the workspace root — already fully satisfied (`Resolved 146
packages` / `Checked 90 packages`, zero installs). The workspace venv lives at
the repo root (`.venv/`), not `orchestrator/.venv/` — `uv run` from
`orchestrator/` resolves it correctly (confirmed via `uv run python -c
"import sys; print(sys.executable)"`) despite an unrelated `VIRTUAL_ENV`
mismatch warning from an outer shell env var, which `uv` correctly ignores.

### Idle-window check

`uptime` sampled repeatedly across the ~25-minute measurement window:

| UTC timestamp | load average (1m, 5m, 15m) |
|---|---|
| 10:20:00 (local, pre-measurement) | 59.81, 34.64, 40.52 |
| 10:21:15 | 95.86, 50.37, 45.50 |
| 10:21:20 | 100.52, 52.09, 46.08 |
| 10:21:25 | 97.99, 52.37, 46.20 |
| 09:26:10Z (full-suite `-n auto` start) | 31.32, 74.69, 61.41 |
| 09:34:04Z (subset `-n1` start) | 18.15, 39.24, 51.24 |
| 09:42:14Z (subset `-n2` start) | 26.27, 31.01, 42.11 |
| 09:46:25Z (subset `-n4` start) | 60.65, 42.01, 43.40 |
| 09:48:41Z (subset `-n8` start) | 34.18, 38.36, 41.83 |
| 09:50:12Z (subset `-n16` start) | 24.32, 34.41, 40.13 |
| 09:51:07Z (subset `-n32` start) | 30.12, 34.60, 39.88 |
| 09:56:14Z (full-suite `-n16` start) | 13.24, 23.21, 33.82 |

Load never dropped below ~13 and repeatedly spiked past 3x the core count
(32). This is a live, continuously-active shared factory host (~350
concurrent task worktrees observed under `.worktrees/`) — this is the "busy
factory box" execution risk the task plan flagged as non-blocking. Per that
documented contingency, measurement proceeded anyway (real data beats no
data), scoped down to control both wall-clock time and this benchmark's own
incremental contribution to the load on a shared host:

- **Collection tax**: measured on the **full suite** via `--collect-only`
  (low marginal footprint — collection is comparatively cheap next to full
  execution).
- **Full-suite execution**: measured **once each** at `-n auto` (today's live
  default — zero incremental footprint, this happens on every verify anyway)
  and at `-n16` (one comparison point, directly mirroring the PRD's own
  "`-n32` vs `-n16`" framing).
- **Speedup curve**: measured on a **fixed, reproducible ~28% subset**
  (`tests/test_[a-l]*.py`, 2,780 of the 9,952 tests) across the full
  `-n ∈ {1,2,4,8,16,32}` range, run **sequentially** (never concurrently —
  concurrent self-runs would confound results by contending with themselves)
  to bound this benchmark's added load while still exercising the complete
  requested range on real code.
- **Core utilization**: derived from `user`+`sys` CPU-seconds ÷ wall-clock
  for every run above (algebraically identical to `/usr/bin/time -v`'s
  "Percent of CPU this job got" — no separate `mpstat`/`time -v` pass needed
  since `time` was already wrapping every run).

## Raw data

### Collection tax (full suite, today vs the PRD's cited basis)

| | tests | pytest-reported | wall (`real`) | user | sys |
|---|---|---|---|---|---|
| **PRD basis** (2026-07-09, `plans/oversubscription-reify-to-df-research-2026-07-09.md`) | 7,774 | — | 20.7s | — | — |
| **Today, serial** (`-n0`, true single-process) | 9,952 | 4.29s | 7.60s | 7.123s | 0.440s |
| **Today, via `-n auto`** (32-way; each worker independently re-collects to verify consistency across workers) | 9,952 | 9.98s | 12.81s | 8.495s | 0.797s |

The suite grew 28% (7,774 → 9,952 tests) but serial collection got
**faster**, not slower (20.7s → 4.29s/7.60s). This benchmark cannot attribute
the cause (import-graph changes since 07-09 vs the original figure being
measured under different conditions/methodology are both plausible) — it is
reported as-is; today's number supersedes the older one as current ground
truth. The `-n auto` collect-only figure (9.98s/12.81s, vs 4.29s/7.60s serial)
is itself a small, direct demonstration of "concurrency multiplies the
collection tax" (PRD §2.1(1)): 32 workers each redoing the same import work
costs more aggregate CPU (8.495s+0.797s=9.29s vs 7.123s+0.440s=7.56s user+sys)
for a wall-clock win that's much smaller than the worker count would suggest.

### Full-suite execution: `-n auto` (32) vs `-n16`

| `-n` | pytest-reported | wall (`real`) | user | sys | total CPU | effective cores | utilization | result |
|---|---|---|---|---|---|---|---|---|
| 32 (`auto`) | 198.85s | 201.206s | 2993.836s | 237.365s | 3231.201s | 16.06 / 32 | **50.2%** | 9,946 passed, 6 failed* |
| 16 | 177.65s | 182.213s | 2479.181s | 221.089s | 2700.270s | 14.82 / 16 | **92.6%** | 9,946 passed, 6 failed* |

**`-n32` is 10.4% *slower* in wall-clock than `-n16` on the full suite today**
(201.206s vs 182.213s) — despite requesting twice the workers, effective
utilized cores only rose from 14.82 to 16.06 (+1.24 cores for +16 requested
workers). This directly contradicts the PRD's original citation of `-n32 ≈
1.3-1.6× -n16` (there, more workers still helped, just sublinearly; here,
more workers *hurt*). The most likely explanation is today's much heavier,
sustained contention (see Idle-window check) — with the box already
oversubscribed by co-tenant verifies, additional self-requested workers
mostly buy scheduling overhead and redundant collection instead of real
parallelism. This should be treated as directionally significant but
magnitude-uncertain pending a clean-window re-measurement.

\* The 6 `test_session_hooks.py` failures are pre-existing and unrelated:
confirmed via `git diff 4b94915071 -- tests/test_session_hooks.py` (empty —
untouched by any task-2394 commit); the file's last real change was task 2643
(unrelated, 16h prior). Not investigated further (out of scope).

### Speedup curve: fixed subset (2,780 tests, `tests/test_[a-l]*.py`, ~28% of suite)

All 6 runs: **2,780 passed, 0 failed** (no flakiness observed in this subset).

| `-n` | pytest-reported | wall (`real`) | user | sys | total CPU | effective cores | utilization (% of `-n`) | speedup vs `-n1` |
|---|---|---|---|---|---|---|---|---|
| 1  | 468.12s | 469.313s | 377.672s | 57.730s | 435.402s | 0.93 | 92.8% | 1.00x |
| 2  | 223.08s | 225.281s | 324.665s | 51.496s | 376.161s | 1.67 | 83.5% | 2.08x |
| 4  | 116.77s | 118.698s | 363.617s | 56.996s | 420.613s | 3.54 | 88.6% | 3.95x |
| 8  | 55.45s  | 58.211s  | 361.898s | 60.238s | 422.136s | 7.25 | 90.6% | 8.06x |
| 16 | 42.50s  | 43.739s  | 531.712s | 68.678s | 600.390s | 13.73 | 85.8% | **10.73x (peak)** |
| 32 | 44.41s  | 45.698s  | 761.882s | 86.975s | 848.857s | 18.58 | 58.0% | 10.27x (regression) |

`-n32`/`-n16` wall ratio: 45.698s / 43.739s = **1.045× — `-n32` is 4.5%
*slower*** than `-n16` on this subset, consistent in direction (if more
pronounced — smaller subset means less test-work to amortize each worker's
fixed collection tax over) with the full-suite pair above.

## Interpretation

1. **Both independent measurements (full-suite pair, subset 6-point sweep)
   agree**: parallelism gains flatten out by `-n16` and provide **no further
   net benefit at `-n32`** on this host today — if anything, a small,
   consistent regression. Efficiency (effective-cores ÷ requested-`-n`) holds
   steady at 83-93% through `-n1..-n16`, then collapses to 50-58% at `-n32`.
   This is a *stronger* version of the PRD §2.1(1) thesis, not a
   contradiction of its qualitative direction — just more extreme under
   today's heavier contention and larger test count.
2. **Single-verify core utilization never approaches full 32-core
   saturation** even at `-n auto`: 50.2% (full suite) / 58.0% (subset) of
   requested workers are effectively "busy" in CPU-seconds-per-wall-second
   terms. A solo verify does not need (and cannot productively use) 32
   workers' worth of concurrent CPU — confirming there is real headroom
   being wasted on redundant per-worker collection rather than genuine
   parallel test execution.
3. **Confidence caveat**: every number above was measured under sustained
   3-6x oversubscription from co-tenant load, not a clean idle window. The
   *direction* (diminishing/negative returns past ~16) is corroborated twice
   today and matches the PRD's own prior (less contended) citation
   qualitatively, so it is reasonably trustworthy. The *exact magnitude*
   (today's outright regression vs the PRD's "still 1.3-1.6x better") is not
   — a clean-window re-benchmark could plausibly show `-n32` clawing back to
   modestly ahead of `-n16`, though it is very unlikely to show `-n32`
   meaningfully *better* than the PRD's own already-sublinear citation.

## Recommendations

### `verify_admission_task_slots` — CONFIRM default `1` (no change)

The PRD's core argument for `N=1` (serializing task-role verify invocations
costs little because `-n32` isn't much better than `-n16` anyway) is
reconfirmed — more strongly than originally measured. No change to
`orchestrator/src/orchestrator/config.py`'s
`verify_admission_task_slots: int = Field(default=1, ge=1)`, and no change
needed to `orchestrator/tests/test_config_verify_admission_reload.py`'s
`assert cfg.verify_admission_task_slots == 1`.

### `verify_admission_pytest_n` — KEEP default `'auto'` (no change landed)

No clean idle window was obtainable this session (see above) — exactly the
contingency this task's plan and `verify_admission_pytest_n`'s own
in-code comment (`orchestrator/src/orchestrator/config.py:2038-2047`)
anticipated: *"a sustained, heavily-contended host precluded a clean-idle-
window measurement supporting a specific cap, so the behavior-preserving
value is kept."* This report is that documentation. **No config change from
this report** — `verify_admission_pytest_n` stays `'auto'` (byte-identical to
today's `-n auto` addopts for every role).

**Candidate for the next re-benchmark:** `-n 16`. It was at-or-tied-for-the-
best wall-clock point in *every* measurement performed today (full-suite
pair and all six subset points), with a clear secondary system-level
benefit this PRD directly cares about — halving the worker-process /
redundant-collection footprint of every task/background verify on an
evidently very heavily oversubscribed shared host. Recommend re-running the
"Reproduction" commands below during a genuine idle/maintenance window (or
on a resource-isolated/cgroup-limited benchmark host) to confirm before
setting it as the landed default — this is a green-tier, hot-reloadable
knob (`RELOADABLE_FIELDS`, `orchestrator/src/orchestrator/config.py:3238`),
so adopting `'16'` later requires no restart.

### PRD §9 item 3 — should `background` be excluded from `-n` capping?

**Recommend NO — keep `background` included in the cap** (already
implemented this way: `orchestrator/src/orchestrator/verify.py:3319`,
`role in {'task', 'background'}`; no change needed). Rationale: `background`
is the main-tip-sweep's role, explicitly designed (PRD §3, §5, §8-T3) as the
**lowest**-priority, fire-and-forget tier that must "never delay real lane
verifies." Exempting it from the `-n` cap would let it claim *more* workers
(uncapped `-n auto`) than a capped task-role verify — directly inverting its
designed priority tier. If anything, `background` is the best future
candidate for an *even lower* `-n` than `task` once a clean-window benchmark
supports role-specific tuning — that would need a second knob (e.g.
`verify_admission_pytest_n_background`), which is a reasonable follow-up
idea but out of scope here (no premise for a specific value yet).

### Follow-up

File/track a re-benchmark during a genuine idle window (or a dedicated,
resource-isolated benchmark host) using the exact commands below, to confirm
whether `-n 16` (or another value) should become the landed
`verify_admission_pytest_n` default.

## Reproduction

```bash
cd orchestrator && uv sync   # prerequisite; confirm venv is current

# Idle-window check — repeat a few times; want sustained loadavg << nproc
uptime

# Collection tax (full suite)
uv run pytest --collect-only -q -n0          # true serial
uv run pytest --collect-only -q              # via -n auto (addopts default)

# Full-suite execution at a specific -n (overrides the `-n auto` addopts —
# last `-n` on the effective command line wins)
time (uv run pytest -q -n16)
time (uv run pytest -q -n32)
time (uv run pytest -q)                       # today's default (-n auto = 32 here)

# Speedup curve on the fixed, reproducible ~28% subset used in this report
for n in 1 2 4 8 16 32; do
  date -u +%FT%TZ
  time (uv run pytest -q -n"$n" $(ls tests/test_[a-l]*.py))
done

# Core utilization = (user + sys) / real, from each `time` invocation above —
# algebraically identical to `/usr/bin/time -v`'s "Percent of CPU this job got".
```
