# Milan per-thread benchmark (2026-10-21)

Question: how much per-thread speed does a second-hand EPYC Milan box give up
for our verify workloads, and how much capacity does it add? `bench.sh` answers
it on a rented Milan host, with the laptop (7940HS, Zen 4) as the reference.

## What it runs

| measurement | what it stands for |
|---|---|
| `py_collect`: one `pytest --collect-only -n 0` of `orchestrator/tests` | single-threaded CPython: imports, AST rewriting, collection (the serial-floor modules) |
| `pyright`: one single-threaded pyright of `orchestrator/src` | the typecheck legs |
| both again as K simultaneous copies, K = physical cores and K = hardware threads | per-thread speed with the whole box busy (all-core clocks, SMT siblings), and whole-box capacity |
| `rustc_j1`: cold `cargo build -j1 -p reify-compiler`, no sccache | one rustc at a time: reify's per-thread compile speed |
| `rustc_jn`: cold `-j<threads>` build of five pure-Rust reify crates | how far extra threads shorten a reify compile |
| real DF merge gate: `orchestrator verify-merge` at `DF_SHA`, 1 alone, then N at once (default 8), each in its own clone | the actual bottleneck, with the spec built from the live `dark-factory-orchestrator.yaml` |

`DF_SHA` defaults to `71bc561086` (main, 2026-10-21), which the laptop gated
green in 862.8 s, so the solo gate has a same-SHA baseline from `runs.db`.

The reify gate itself is not replicated: it needs the `/opt/reify-deps`
native stack (OCCT, manifold, gmsh, openvdb), and on the laptop it runs
against a warm sccache, so a cold box would not be comparable. The two
compile probes stand in for it.

## Renting the box

Any Ubuntu 24.04 host you can `ssh` into with passwordless sudo works.
Milan options:

- **AWS `c6a.48xlarge`**: 2× EPYC 7R13, 192 vCPU, 384 GiB, the whole host.
  Give the root gp3 volume about 200 GB, and raise its throughput (e.g. 1000
  MB/s, 16000 IOPS) so cold compiles and venv builds are not disk-bound.
- **GCP `c2d-standard-112`**: Milan, usually reporting EPYC 7B13, the same
  part sold cheaply second-hand. 112 vCPU, 448 GB.

Eight concurrent gates need roughly 128 GB RAM at peak. The script caps probe
copies by available memory.

Add an ssh alias (e.g. `Host milan` with `User ubuntu`), then confirm
`ssh milan true` works without a prompt.

## Running

```bash
# the rented box: provision, probes, 1 gate, 8 gates (about 1.5-2 h)
plans/hardware-bench-2026-10-21/bench.sh milan

# the laptop reference: probes only, no provisioning, no gates
plans/hardware-bench-2026-10-21/bench.sh --reference leo-laptop
```

Both write to `results/<host>-<utc>/` and print a summary; `summarize.py
<dir> [<reference dir>]` reprints one. The summary compares against the newest
reference run, so run the laptop reference first or rerun `summarize.py` after.

Everything on the remote lives in `~/df-hwbench`. On the laptop the reference
run touches no production checkout and never uses sccache. It still competes
with production for CPU, so before each batch it waits until no `verify-merge`,
`verify-task` or reify `verify.sh` is running and the load has settled. A batch
that overlapped a production verify anyway is marked `*`.

`--quick` is a smoke run: one copy per probe, small compiles, and a gate of
the `sampler` module only. `--skip-gate` runs the probes alone.

## Running on RunPod CPU pods (containers)

The pods are root containers (`runpod/base:1.0.2-ubuntu2404`, no required sudo)
whose CPU limit is a cpuset or a quota, not a smaller machine, and a network
volume (`/workspace`) is too slow and noisy to benchmark on. So two pods share
the volume: a cheap 2-vCPU staging pod does the network-bound work once, the
32-vCPU bench pod copies it to local disk and measures from there.

```bash
# staging pod: toolchains, apt .debs, uv/npm/cargo caches, both sources -> $BENCH_STAGE
BENCH_SSH_CONFIG=plans/hardware-bench-2026-10-21/ssh_config \
  plans/hardware-bench-2026-10-21/bench.sh --stage pod-stage
# bench pod: localize.sh copies the volume to /root, then the normal probes and gates
BENCH_SSH_CONFIG=plans/hardware-bench-2026-10-21/ssh_config \
  plans/hardware-bench-2026-10-21/bench.sh --from-volume pod-bench
```

`BENCH_STAGE` (default `/workspace/hwbench`) is the volume's directory. The
volume streams one big file fast and many small files slowly (about 5 MB/s), so
it holds a few large tars, never trees the bench pod reads file by file:

| file | holds |
|---|---|
| `home.tar` | the staging pod's `$HOME` (always `/root`) pieces: `.local` (uv, pinned node, uv-managed Pythons), `.cargo` (+ registry), `.rustup`, `.cache` (uv, pyright), `.npm`, dummy ssh key, `.gitconfig`; built on the pod's local disk, extracts to the same paths |
| `apt-archives.tar` | the .debs of the DF apt list |
| `reify-deps.tar` | uploaded by hand from `/opt/reify-deps` (top-level dir `reify-deps/`); extracted to `/opt` |
| `src/df.git` | bare repo, `main` = `DF_SHA` (a few pack files) |
| `src/reify-<sha>.tar` | `git archive` of `REIFY_SHA` |
| `staged.json` | SHAs, toolchain versions, size + sha256 of each tar; written last |

`--stage` runs `stage.sh` detached and polls, like the run itself: staging
takes minutes and must survive a dropped ssh connection. Each tar is written
under a `.tmp` name and renamed. The bench run touches `/workspace` only in
`localize.sh`, which logs the time and rate of every extraction.

Toolchain versions (uv, node, rust) are pinned in `remote/lib.sh`, shared by
`provision.sh` (a box with internet) and `stage.sh` (the volume); `host.json`
records `toolchain` (uv, node, npm, rustc, cargo, python3, the DF venv's
python) so identity across pods can be confirmed.

The probes size their parallelism by `remote/container_limits.py`, not by
`nproc`/`lscpu` (which describe the whole host in a container):

| case | `cpu_budget` (= `threads`, the top K level) | `cores` (the middle K level) |
|---|---|---|
| nothing limits us (bare metal, VM) | host threads | distinct (socket, core) pairs of the host |
| cpuset | CPUs in the affinity mask | distinct (socket, core) pairs among those CPUs |
| cgroup quota (v2 `cpu.max`, v1 cfs) tighter than the cpuset | quota rounded up | budget // 2 (which cores we get is unknowable) |

`host.json` records `cpu_budget`, `cpu_budget_source`, `cores_rule`,
`host_threads`, `allowed_cpus`, `cgroup_cpu_quota`; K levels are memory-capped
by the cgroup memory limit.

## Later passes on a bench pod (what the 2026-10-22 run needed)

The first pass (`bench.sh`) left four problems, each fixed by a script in `remote/`:

- **`phase2.sh`**
  - Reruns the reify compile probes, which need the `tree-sitter` CLI that
    `reify_gate.sh` installs.
  - Runs `diag.sh`: user vs system CPU, and a pure-interpreter loop.
  - Runs the DF gates under `eatmydata` with `spec-deselect.json`. Pod disks differ
    wildly in fsync latency, and one fused-memory test hangs on some hosts.
- **`phase3.sh`**
  - Reruns the single-thread probes on a quiet disk, into `probes.rerun.jsonl`.
  - Then runs `reify_as_user.sh`: the reify gate as an unprivileged `leo` user, without
    the infra pool (`reify-spec-noinfra.json`). As root, reify's permission tests fail,
    and a failing step skips the rest of the gate.
- **`df_as_user.sh 1 2 4`**: DF gates as `leo`, for the root-vs-user check and the
  concurrency curve.
- **`selfload.sh`**: one probe alone vs beside 31 of our own busy threads, interleaved.

`compare.py <laptop dir> <pod dir>...` builds `results/comparison.md`.
- The laptop's DF and reify baselines come from production records, filtered to gates
  that ran with no other gate alongside.
- `results/REPORT.md` is the write-up.

`runpod_pod.py` creates, terminates and checks pods and the volume, with `--want
REGEX` to redraw until the CPU model matches.

## Caveats

- Cloud Milan clocks are not exactly what you would buy: a 7763 sustains about
  2.9-3.1 GHz all-core and a 7R13 slightly more. The probe lines record mean
  MHz, where the VM exposes it.
- Verify legs run without the laptop's systemd memory scopes
  (`verify_use_cgroup_scope: false`).
- The box holds a copy of both repos' source. Delete the instance afterwards.
