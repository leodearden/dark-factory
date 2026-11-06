## Per-thread speed

| one thread, median seconds | Ryzen 9 7940HS (16T) | EPYC 7713 64-Core (32T) | EPYC 9655P 96-Core (32T) |
|---|---|---|---|
| pytest collection, orchestrator tests (CPython imports + AST) | 16.3 s | 32.5 s (0.50×) | 15.1 s (1.08×) |
| same, first set (pods: right after GBs of writes) | 16.3 s | 36.1 s (0.45×) | 14.3 s (1.14×) |
| pyright, orchestrator/src (Node, single-threaded) | 15.9 s | 31.5 s (0.51×) | 11.2 s (1.43×) |
| same, first set | 15.9 s | 23.7 s (0.67×) | 10.3 s (1.55×) |
| pure CPython loop (no I/O) | 1.2 s | 1.6 s (0.76×) | 0.8 s (1.49×) |
| pytest collection, 10 interleaved rounds (laptop: its probe median) | 16.3 s | 22.6 s (0.72×) | - (-) |
| pyright, 10 interleaved rounds (laptop: its probe median) | 15.9 s | 23.2 s (0.69×) | - (-) |
| cold reify-compiler build, -j1, no sccache | 151.0 s | 193.2 s (0.78×) | 114.0 s (1.33×) |

## With every thread busy

| every thread busy | Ryzen 9 7940HS (16T) | EPYC 7713 64-Core (32T) | EPYC 9655P 96-Core (32T) |
|---|---|---|---|
| py_collect: seconds per copy | 27.5 s | 56.5 s | 21.7 s |
| py_collect: copies/hour per hardware thread | 131 | 64 | 166 |
| pyright: seconds per copy | 60.7 s | 100.3 s | 35.8 s |
| pyright: copies/hour per hardware thread | 59 | 36 | 101 |
| cold 5-crate reify build, -j<threads>, no sccache | 67.3 s | 73.0 s | 44.2 s |
| max MHz of a busy core (1-copy batches) | - | 3100 | 4538 |
| host 1-min load during 1-copy batches (pods: whole host, other tenants) | - | 132 | 43 |

## Real dark-factory merge gates

- laptop, same SHA, production record (n=1, no reify gate alongside): 863 s, 4.2 gates/hour (one at a time, 16 threads, fsync 0.55 ms)
- laptop, all SHAs 10-12..10-22 with no reify gate alongside: median 793 s (n=146)
- EPYC 7713 64-Core (32T) [gate-c1]: 1 at once, mean 1192 s per gate (ratio 0.72×), 3.0 gates/hour on 32 vCPU
- EPYC 7713 64-Core (32T) [gate-c2]: 2 at once, mean 1517 s per gate (ratio 0.57×), 4.7 gates/hour on 32 vCPU
- EPYC 7713 64-Core (32T) [gate-c1-eatmydata-deselect]: 1 at once, mean 1075 s per gate (ratio 0.80×), 3.3 gates/hour on 32 vCPU
- EPYC 7713 64-Core (32T) [gate-c2-eatmydata-deselect]: 2 at once, mean 1223 s per gate (ratio 0.71×), 5.8 gates/hour on 32 vCPU
- EPYC 7713 64-Core (32T) [gate-c1-user-eatmydata-deselect]: 1 at once, mean 945 s per gate (ratio 0.91×), 3.8 gates/hour on 32 vCPU
- EPYC 7713 64-Core (32T) [gate-c2-user-eatmydata-deselect]: 2 at once, mean 1254 s per gate (ratio 0.69×), 5.7 gates/hour on 32 vCPU
- EPYC 7713 64-Core (32T) [gate-c4-user-eatmydata-deselect]: 4 at once, mean 1874 s per gate (ratio 0.46×), 7.6 gates/hour on 32 vCPU
- EPYC 9655P 96-Core (32T) [gate-c1-eatmydata, PARTIAL tally]: 1 at once, mean 957 s per gate (ratio 0.90×), 3.8 gates/hour on 32 vCPU
- EPYC 9655P 96-Core (32T) [gate-c2-eatmydata, PARTIAL tally]: 2 at once, mean 1066 s per gate (ratio 0.81×), 6.7 gates/hour on 32 vCPU
- EPYC 9655P 96-Core (32T) [gate-c1-eatmydata-deselect]: 1 at once, mean 701 s per gate (ratio 1.23×), 5.1 gates/hour on 32 vCPU
- EPYC 9655P 96-Core (32T) [gate-c2-eatmydata-deselect]: 2 at once, mean 894 s per gate (ratio 0.96×), 8.0 gates/hour on 32 vCPU

## Real reify merge gate, phase by phase

| reify warm gate phase (seconds) | laptop, uncontended median (n=23 test + 49 lint legs, other SHAs) | laptop at 57be3b937f (DF gate running alongside) | EPYC 7713 64-Core (32T) | EPYC 9655P 96-Core (32T) |
|---|---|---|---|---|
| preflight | 3 | 3 | 5 (0.56×) | 2 (1.41×) |
| clippy | 36 | 38 | 48 (0.74×) | 27 (1.31×) |
| gui_feature_check | 9 | 9 | 10 (0.89×) | 6 (1.41×) |
| infra_static | 3 | 4 | 5 (0.66×) | 2 (1.61×) |
| gui_node | 43 | 31 | 64 (0.67×) | 27 (1.61×) |
| debug_nextest | 489 | 837 | 596 (0.82×) | 269 (1.82×) |
| release_nextest | 336 | 432 | 287 (1.17×) | 199 (1.69×) |
| gui_feature_nextest | 39 | 62 | 42 (0.92×) | 29 (1.35×) |
| **sum of compared phases** | 958 | 1417 | 1058 (0.91×) | 561 (1.71×) |
| excluded on pods (laptop only: infra_pool, release_prebuild) | 403 | 429 | - | - |
- EPYC 7713 64-Core (32T) reify cold gate at fad2d98195: 1942 s wall, passed=True, wrapper=eatmydata
- EPYC 7713 64-Core (32T) reify warm gate at 57be3b937f: 1065 s wall, passed=True, wrapper=eatmydata
- EPYC 9655P 96-Core (32T) reify cold gate at fad2d98195: 874 s wall, passed=True, wrapper=eatmydata
- EPYC 9655P 96-Core (32T) reify warm gate at 57be3b937f: 564 s wall, passed=True, wrapper=eatmydata


RULES
- one thread: median wall of single-copy batches with rc 0; pods use probes.rerun.jsonl (quiet disk) when present,
  else the first set; the laptop's set ran under wait-for-quiet. diag rows: median of diag.jsonl reps.
  '10 interleaved rounds': st-samples.jsonl, py_collect/pyright/interp alternated 10x on the Milan pod, host load
  28-74 (the first set ran at host load ~100, the rerun at ~130: other tenants on the 256-thread host).
  pyright is not strictly single-threaded (Node GC threads: user CPU ~1.7x wall).
- every thread busy: per-copy median wall at the top K level (laptop K=16, pods K=32 = their cpuset);
  copies/hour per hardware thread = 3600 / per-copy seconds.
- DF gates: wall of `orchestrator verify-merge` per gate; gates/hour = concurrency * 3600 / makespan.
  'eatmydata' = fsync turned into a no-op (pod disks differ: Milan 2.2 ms, Turin 126 ms per 4 KiB dsync);
  'deselect' = the live spec minus one fused-memory test that hangs 300 s on some hosts.
  Pod gates run as root: ~65 permission/host-fixture tests fail fast there and pass on the laptop.
- reify: per-phase sums of verify.sh's own step records (test + lint legs). Pods: their warm gate at 57be3b937f.
  Laptop baseline: per-leg, per-phase medians over every merge-role record in its ledger (other SHAs, 10-14..10-22)
  whose whole span overlapped no laptop DF merge gate (runs.db); the laptop's own 57be3b937f record is shown beside
  it but ran alongside a DF gate (~1.5x slower nextest). Phases under 1 s are summed but not listed. The pods omit
  the infra pool and its release prebuilds (REIFY_INFRA_SUITE_ACTIVE=1, verify.sh's nested-suite guard).

