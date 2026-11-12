# Milan per-thread benchmark: results (2026-10-22, final 2026-11-01)

Question: how much per-thread speed does a second-hand dual EPYC Milan box give up on
our verify workloads, and what does it buy?

Hosts:
- **laptop**: Ryzen 9 7940HS (Zen 4, 8C/16T), the production remote verify host, as the
  reference.
- **Milan slice**: RunPod cpu3g-32-128 in US-CA-2. That is a 32-CPU cpuset (16 full SMT
  core pairs over both sockets) of a dual EPYC 7713 host (Zen 3, 2×64C/256T), which is
  the box shape being priced. Other tenants used the rest of the host.
- **Turin slice**: the same flavour in EU-RO-1 on an EPYC 9655P (Zen 5, 96C/192T), as the
  modern-server comparison.

Both slices ran identical pinned toolchains: uv 0.11.6, Node v22.23.3, rustc 1.96.0, and
Python 3.13.12 in the DF venv. The laptop's own versions were not recorded.

Every figure's rule is in `comparison.md` (built by `../compare.py`). A results critic
reviewed this twice; its second pass rated it "conclusive with fixes", and those fixes are
folded in here.

## 1. Per-thread speed (ratio = laptop seconds / host seconds; above 1 is faster)

| | Milan 7713 | Turin 9655P |
|---|---|---|
| One thread alone: pytest collection | 0.72 (n=10, 22.1-24.9 s) | 1.08-1.14 (n=3) |
| One thread alone: pyright | 0.69 (n=10, 20.9-25.3 s) | 1.43-1.55 (n=3) |
| One thread alone: pure CPython loop | 0.76 (n=5) | 1.49 (n=5) |
| One thread: cold rustc -j1 | 0.78 (n=1, host load 114) | 1.33 (n=1) |
| Every thread running a copy, copies/hour per thread (provisional): pytest collection | 0.49 | 1.27 |
| Every thread running a copy, copies/hour per thread (provisional): pyright | 0.61 | 1.71 |

- **Milan vs Turin per thread:** 0.45-0.67 with one thread, 0.36-0.39 with every
  thread busy.
- **Our own load, interleaved (selfload.jsonl):** one Milan copy alone takes 23.4 s
  (pytest) and 20.7 s (pyright); beside 31 of our own CPU-burning threads, 32.2 s and
  35.1 s.
  - The burners are interpreter loops, so they barely touch memory, L3 or disk. That
    makes this a lower bound on the penalty of a box busy with real verifies.
- **The all-busy rows are provisional.** They come from the probe harness, whose
  single-copy batches on Milan read 30-37 s against 22-24 s run alone, for a reason not
  identified. A harness A/B (with and without its /proc/cpuinfo sampler) was started,
  but the pod was lost before its results were fetched.
  - Like for like (identical copies against the alone baseline), Milan loses more going
    all-busy than the laptop does: 2.4× against 1.69× for pytest, 4.85× against 3.81×
    for pyright.
- **Pyright is not single-threaded:** Node's GC threads make its user CPU about 1.7×
  wall.

## 2. Whole gates (what we actually wait on)

### DF merge gate

The baseline throughout is the laptop's same-SHA production record: **863 s**, with no
reify gate alongside (other load unknown). The laptop's median since 10-20 with no reify
gate alongside is 836 s (n=40), which agrees. A ~1180 s laptop mode with no reify
overlap also exists, cause unknown.

All slice gates ran 16 pytest-xdist workers each, with fsync neutralised (eatmydata) and
one hanging test deselected; see section 4.

| | laptop (16T) | Milan slice (32 vCPU), as user | Milan, as root | Turin slice, as root |
|---|---|---|---|---|
| one gate at a time | 863 s | 945 s (0.91×) | 1075 s (0.80×) | 701 s (1.23×) |
| two at once (16 workers per 16 threads) | — | 1254 s each (0.69×) | 1223 s each (0.71×) | 894 s each (0.97×) |
| four at once (64 workers on 32 vCPU) | — | 1874 s each | — | — |
| gates/hour | 4.2 | 3.8 / 5.7 / 7.6 at 1 / 2 / 4 | 3.3 / 5.8 | 5.1 / 8.0 |

- **Latency:** a single DF gate on Milan is about 1.1-1.25× slower than on the laptop.
  On Turin it is about 1.2× faster.
- **Turin's figures are root-only:** they include about 15 more test failures than
  Milan's (scripts tests that fail on a 15 s subprocess timeout), so they may be
  slightly understated.
- **Today's task-verify host is the workstation:** its own ("local" runner) merge gates
  since 10-12 have a median of **2216 s** (n=105), loaded by agents. A Milan slice is
  about 1.8-2.3× faster than that.

### reify merge gate

The warm gate, on the phases both sides ran, which is about 70% of the laptop's gate;
the infra pool is excluded (see section 4).

| | laptop (16T) | Milan slice (32 vCPU) | Turin slice (32 vCPU) |
|---|---|---|---|
| compared phases | 958 s (uncontended median of 23 test + 49 lint legs, other SHAs) | 1058 s (0.91×) | 561 s (1.71×) |
| nextest phases (debug / release / GUI) | 489 / 336 / 39 | 596 / 287 / 42 | 269 / 199 / 29 |
| cold gate (slices only) | — | 1942 s | 874 s |

- **Contended record not used:** the laptop's own record at this SHA ran while a DF gate
  ran alongside, taking 1417 s, so it is not the baseline.
- **What scales:** the nextest test runs scale with threads. The builds and serial
  phases do not: GUI node 0.67×, clippy 0.74×, and a cold 5-crate build at -j32 on
  Milan (73 s) is no faster than the laptop's at -j16 (67 s).

## 3. Projection to an owned dual-7713 box (256T, 512 GB, 4×4 TB NVMe)

The box is eight of these 32-thread slices. None of this was measured at box scale.

**DF merge gates.** Measured per slice: 3.8 / 5.7 / 7.6 gates per hour at 1 / 2 / 4 at
once. Eight slices give an upper bound of **about 45-61 per hour** if scaling is linear.
- The c4 point ran while other tenants' load was higher: host load median 157, against
  103 for the tail of c2 (hostload.log covers 19:02-19:54 only).

**Full warm reify gates.**

| | per hour |
|---|---|
| warm gate, measured on one slice (1065 s, infra tests excluded) | 3.4 |
| eight slices, linear | **27** |
| full gate, adding the infra pool at about 575 s (estimated from the laptop's 403 s × Milan's ratio on non-nextest phases) | 2.2 per slice, **18** per box |
| allowing 10-25% for whole-box saturation (a judgement, not measured) | **13-16** per box |

- **The laptop for comparison:** about 2.6 full reify gates per hour, at about 1360 s
  each, when no DF gate runs alongside.
- **Slice width:** builds and serial phases barely gained from 32 threads, so 16-thread
  slices might do better than eight 32-thread ones. That was not tested for reify.
- **Disk is covered:** four 4 TB NVMe drives give room and IOPS for 8-16 concurrent
  `target/` trees and venvs. These runs had fsync off; local NVMe makes that close to
  real.
- **Warm means warm:** each gate started from the previous commit's `target/` and
  sccache. A cold or very stale worktree costs far more: the cold gate took 1942 s even
  without the infra tests.

**What it buys.** Milan shortens no single verify: a DF gate is 1.1-1.25× slower than
on the laptop, and a reify gate on 32 Milan threads is about 1.1× slower than on the
laptop's 16 (on the compared 70%). So the merge lane, which runs one gate at a time,
belongs on the fastest-per-thread host.

The case is concurrency for task-role verifies:
- **Merge gates aren't the queue:** they run about 1.7 per hour and never more than 2 at
  once.
- **Task-role verifies are:** at the heartbeats read on 10-22, admission showed 28-31
  per-module entries queued, the oldest waiting over 2 hours, with one admitted at a
  time.
- **A config change comes first:** those verifies queue on an admission slot count (and
  reify's test semaphore), so the slots must be raised before more cores help.
- **Not yet measured:** task verifies are scoped (`-n 8`), and their cost on a slice was
  not benchmarked.

**A Zen 5 box** (Turin-class: 1.1-1.7× the laptop per thread, about 2× Milan) would
improve latency as well as capacity. Its price was not studied here; the sibling study
`project_boat_vs_milan_study_2026_10_31` covers alternatives.

## 4. Adjustments, and which way each one leans

1. **fsync.** Synced-write latency per 4 KiB dsync write: laptop 0.55 ms, Milan 2.2 ms,
   Turin 126 ms (`fsync-dsync-2026-10-22.txt`). All three were measured with other work
   running.
   - Turin's unadjusted gate measured its disk, so every slice gate runs under eatmydata.
   - That leans toward the slices, by the laptop's own fsync cost of a few percent.
     Milan with and without eatmydata, 1075 vs 1192 s, is confounded with the deselect
     and the time of day.
2. **Deselected test.**
   `fused-memory tests/test_main_ticket_janitor_wiring.py::test_built_janitor_drains_through_the_interceptor`
   hung to pytest's 300 s timeout in 4 of 4 Turin runs. Its body has an unbounded
   `await curator.reached.wait()` (line 130 at 71bc561086). It passed on Milan and on
   the laptop. Deselected on both slices: symmetric, small.
3. **Root vs user.** On Milan, one gate as user took 945 s against 1075 s as root; two
   at once took 1254 s against 1223 s. That is within about 12% either way (n=1 each),
   so root has no material effect.
   - There is no host-load record for the one-gate pair.
   - Failing tests: 58-60 as user, 66-67 as root. The remaining failures need local
     OAuth accounts and other laptop fixtures, and they fail fast.
   - Both directions are present: the slices do less work through failures and skips,
     and pay more through a cold `uv sync && npm ci` per worktree and the flake re-run
     pass. The net effect is unquantified.
4. **reify.** It ran as an unprivileged user, because as root a failing test fails the
   whole gate fast.
   - It also ran without the infra shell pool and its release prebuilds, using
     verify.sh's nested-suite guard (`REIFY_INFRA_SUITE_ACTIVE=1`). Those take about
     403 s on the laptop, and their flock and /proc tests could not be shown to pass in
     a container.
5. **Clocks.** /proc/cpuinfo MHz inside a container is not a reliable per-core clock.
   No boost claims are made.

## 5. Side findings: test-suite hermeticity (not filed)

- **DF, runpod key:** `scripts/run_vllm_eval.py` reads `~/.ssh/id_runpod.pub` at import.
  Without it, the orchestrator tests stop at collection.
- **DF, hanging janitor test:** the janitor wiring test above hangs on some hosts (the
  unbounded Event wait).
- **DF, host assumptions:** about 58 tests assume local OAuth accounts, sqlite3, the
  laptop's timezone, or (about 8 more) a non-root user. On Turin, about 15 restart
  tests also time out at 15 s.
- **reify, root:** its infra shell tests and at least one Rust test
  (`reify-audit jcodemunch_index::tests::read_index_state_reads_a_wal_index_with_no_shm_and_a_read_only_directory`)
  assume a non-root user.

## 6. Cost and cleanup

- **Spend:** $12.90 of the $50 RunPod credit had been spent by 19:55 UTC on 10-22, with
  the Milan pod and the volume still up.
- **Cleanup unconfirmed:** the session ended before cleanup. On 10-29 the pod refused
  ssh, but the stored API key now returns 401, so neither the final spend nor the
  deletion of the `hwbench` volume (US-CA-2, 40 GB) could be confirmed from here.
