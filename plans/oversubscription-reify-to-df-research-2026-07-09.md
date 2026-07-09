# Research substrate — porting Reify's test/verify oversubscription control to Dark Factory

Date: 2026-07-09. Author: /deb investigation (xdist verify flake, tasks 2293/2286).
Purpose: durable input for the PRD that translates Reify's PSI-gated / job-server
oversubscription control to dark-factory's **pytest** verify path. NOT itself a PRD.

## Why this exists (the problem)

Dark Factory runs the orchestrator against *itself* at high concurrency:
`max_concurrent_tasks: 24` (`orchestrator/config.yaml:4`), and **each** task lane's
verify runs `pytest -n auto` = **32 workers** on a 32-core host
(`orchestrator/pyproject.toml:100`, `fused-memory/pyproject.toml:33`). 24 lanes ×
32 workers ⇒ observed load avg **96–193** (3–6× oversubscribed). Under that
starvation, load-sensitive tests miss hardcoded wall-clock deadlines and fail
spuriously at a *moving target* of unrelated tests, manufacturing false verify
failures for innocent tasks.

Structural mitigations ALREADY landed (do not redo):
- **task 2361** (done): `--timeout=300` on the fallback `test_command` + `tests/scripts/` 6th segment.
- **task 2365** (done, `054e6c42ce`): reclassify pytest-xdist `"node down: Not properly terminated"` worker-crash as transient `VerifyInfraError` → one bounded auto-retry (only when no real FAILED summary present).
- **task 2368** (done, `159b23e483`): scope mixed root+subproject diffs to (tests/scripts + touched subproject); add cockpit/sampler to the fanout.
- **tasks 2326–2329** (L3b, live): scheduler host-PSI **dispatch-admission** gate (`scheduler.py:4041-4114`) — holds *new heavy task dispatch* at cpu-some avg10 ≥ 85; does NOT throttle running lanes or the verify subprocess.

Residual gaps this PRD must close (NONE of the above bounds the actual worker count):
1. No global/cross-lane cap on total pytest xdist workers → the core oversubscription.
2. No PSI/load gate on the verify subprocess itself (only on new dispatch).
3. **No merge>task priority for DF verify workers** — livelock hazard (see below).
4. Per-test in-test-deadline flakes (filed separately as test-hardening tasks).

## The livelock constraint (user-stated, load-bearing design requirement)

Reify prioritizes the **merge queue above task lanes**. Rationale:
- If **merge starves task** → self-correcting: the merge queue drains, then tasks run.
- If **task starves merge** → merge verifies time out → **resubmission storms / livelock**.

Therefore any DF worker-admission scheme MUST give merge-role verify strict
priority over task-role verify. **DF currently lacks this entirely** (gap #3).

## Current DF state (gap analysis)

- `pytest-jobserver.service` (`~/.config/systemd/user/`) is RUNNING: inline bash seeds a
  **single** FIFO `/tmp/pytest-jobserver` with **32 `+` tokens** (`printf "%032s"|tr " " "+"` + `sleep infinity` holds the fd). Single-pool (no merge/task split). Sized as a RAM budget (~2 GB/proc × 32), i.e. a cap on *concurrent pytest processes*, NOT on xdist workers.
- Client plugin `shared/src/shared/pytest_jobserver.py` EXISTS (acquires 1 token per pytest *process* at `pytest_configure`, fail-opens on missing FIFO/timeout/OSError) but **is registered in NO conftest on main** — added in `c05fc4034d`, **removed in `866db56dcf`** ("chore: save WIP before inter-iteration rebase", Apr 24), never restored. So the FIFO + service are **dead weight**; DF pytest runs fully ungated.
- Two latent hazards if naively rewired: (a) it caps *processes* not *workers* — 32 tokens × 32 workers ≈ 1024 CPU cap; (b) **no `PYTEST_XDIST_WORKER` guard** — under `-n auto` each of the 32 workers is a full session that loads the plugin and would ALSO try to acquire a token (controller + 32 = 33 acquisitions) → starvation/deadlock.
- `JobserverConfig` (`config.py:586-645`) + `workflow._build_agent_env` (`workflow.py:6979-7007`) wire the **cargo** jobserver (`CARGO_MAKEFLAGS=--jobserver-auth=fifo:/tmp/reify-jobserver-task`) into **agent** subprocesses only; `enabled=False` (reify-only). Nothing touches pytest. Verify env (`_resolve_verify_env`, `verify.py:2738`) stamps only `DF_VERIFY_ROLE` + `config.verify_env` (empty) — no nice/jobserver/cgroup.
- `-n auto` is only ever overridden by `_force_serial_pytest` (`verify.py:773-807`) at ONE site — the env-transient recovery retry (`verify.py:3087`). Normal verify never caps `-n`. A uniform `-n` cap in `config.yaml` was **rejected** because shared/escalation/dashboard declare no xdist (would error "unrecognized arguments").
- Reusable substrate: **`shared.psi`** (`shared/src/shared/psi.py`) — `parse_pressure_file(text)->{some_avg10,full_avg10}|None`, `read_pressure(name)`, `PsiSample(...).saturated(cfg)`, `read_psi_sample(read=...)` (fail-open sentinel). Already consumed by sampler + scheduler gate. The `_fanout_sem` idiom (`verify.py:3481`, bounds ≤4 modules within one call). `reload_config` green-tier hot-tunability already covers `psi_admission`.

Gap table (Exists / Present-unwired / Missing):
| Control | Status |
|---|---|
| (a) pytest worker global cap | MISSING (only serialized on env-recovery retry) |
| (b) PSI gate on verify subprocess | MISSING (dispatch-admission gates new *task dispatch* only, at cpu-some≥85) |
| (c) dual-pool merge>task priority for verify workers | MISSING in DF (reify/cargo-only, default-off) |
| (d) fail-open behavior | PRESENT + reusable (plugin, `read_psi_sample`, dispatch gate, `agent_env`) |
| pytest-jobserver FIFO+service+plugin | PRESENT but UNWIRED (dropped from conftests) + wrong granularity (process not worker) |

## Reify design digest (what to translate)

Three cooperating, independent layers, all keyed off `DF_VERIFY_ROLE ∈ {task, merge, offline}`:

### 1. Jobserver token pool (fine-grained) — `scripts/jobserver-balancer.py`
- Custodian daemon holds **two** named-pipe FIFOs: `/tmp/reify-jobserver-merge`, `/tmp/reify-jobserver-task`. Token = one `'+'` byte; count read **non-destructively** via `FIONREAD` ioctl. Total tokens = `nproc` (`os.sched_getaffinity`), overridable by `REIFY_JOBSERVER_TOKENS`.
- Custodian opens each FIFO `O_RDWR|O_NONBLOCK` and holds the fd for process lifetime (the "C5 custodian contract" — without an O_RDWR holder buffered tokens evaporate). `systemd` unit `PartOf=orchestrator-reify.service` (re-seed on restart), `Restart=on-failure`.
- **GNU jobserver protocol:** cargo/rustc are the clients. `verify.sh:657` exports `CARGO_MAKEFLAGS=--jobserver-auth=fifo:<role-fifo>`; each rustc job blocks on a token. **Available tokens = effective `-j`** (no static `-j`). *This is the key granularity lesson: reify gates each build JOB (rustc is a jobserver client); DF must gate xdist WORKERS, which are not jobserver clients.*
- Control loop (0.1s tick): SENSE (FIONREAD both) → PRESSURE stage → idle accounting → `decide()` → transfer → sleep.

### 2. Dual-pool merge>task priority — `jobserver-balancer.py:445-538`, `verify.sh:648-658`
- Baseline partition (merge-favored): `task_baseline = max(1, TOKENS//4)`, `merge_baseline = TOKENS − task_baseline`. nproc=32 ⇒ merge 24 / task 8. Invariant `merge_baseline > task_baseline`.
- `decide()` ratchet (branch order load-bearing): IDLE→reset toward baseline after 10 idle ticks; **MERGE-DEMANDED** (`free_merge==0 and free_task>0`) → move ALL task spare to merge (`t2m`, absolute merge priority, monotone); **TASK-DEMANDED** (`free_task==0 and free_merge>ε`, ε=1) → give back `free_merge−ε`. Under contention it drifts monotonically to merge=TOKENS, task=0 ⇒ merge always wins the tug-of-war ⇒ **prevents the task→merge livelock**.
- Reinforced: merge **bypasses** both admission gates (`lib_test_semaphore.sh:91-94`, `cpu-admit.sh:223-231`); cgroup `cpu.weight` merge **300** / task **100** (`lib_cgroup.sh:57-60`); `nice -5` merge / `nice -15 ionice -c2 -n7` task.

### 3. PSI / load-aware admission
- Shared PSI parser `cpu_admit_read_avg10()` (`cpu-admit.sh:113-120`) reads `/proc/pressure/cpu` (`some` avg10) and `/proc/pressure/memory` (`full` avg10); balancer re-implements identical idiom.
- **Balancer dynamically shrinks the TASK pool under CPU pressure** (`jobserver-balancer.py:610-654`): hysteresis `PRESSURE_HOLD_THRESHOLD=50.0` / `PRESSURE_RELEASE_THRESHOLD=40.0` (release must be < hold). avg10 ≥ 50 → pull `min(free_task, headroom)` task tokens into a `held_back` reservoir (bounded by `MAX_HELD_BACK = TOKENS//4 = 8`); avg10 < 40 → release reservoir; 40–50 → no-op. Drains **task_fd only**; `suppress_giveback()` blocks m2t while pressure active → merge pool never shrunk. `held_back` published atomically to `/tmp/reify-jobserver-held-back` so the canary distinguishes "held on purpose" from "leaked."
- **verify.sh per-invocation PSI gates** (over `cpu_admit`): `psi_gate` (requeue mode, cpu avg10 ≤ 50, 20s min inter-dispatch spacing via flock timestamp, MAX_WAIT 1800s → **exit 75 EX_TEMPFAIL**); `compile_gate` (admit mode, cpu avg10 ≤ 85, **mem `full` avg10 ≤ 10** the binding constraint, **holds continuously, never exits 75 — storm-proof**, emits `@@CLOCK_STOP@@` markers to exclude the wait from the verify timeout).
- **`fleet-load-detector.sh`** — host-aggregate monitor (NOT a per-command gate): `load1/nproc ≥ 4.0` OR `/proc/pressure/cpu avg10 ≥ 80` → exit 3, `@@REIFY_FLEET_OVERSUBSCRIBED@@`. Explicitly built as the **reify→DF seam** (ships the primitive; DF wires the periodic invocation + L3b cap/escalation).
- All PSI reads **fail-open** (unreadable → admit + warn).

### 4. Worker gating mechanism
- (i) Global compile parallelism = the token pool (24 merge / 8 task, minus held-back).
- (ii) Per-invocation **N-slot flock test semaphore** (`lib_test_semaphore.sh` over `lib_slot_acquire.sh`): default N=1 ⇒ one verify test block host-wide at a time; **host-global fixed lock path** `/tmp/reify-test-semaphore-$(id -u).lock` (TMPDIR-independent); acquire = shuffle 1..N (herd avoidance for N≥2) → `flock -xn` per slot → retry `sleep 0.5` until WAIT(1800s) → return 75; children run with `9<&-` so no descendant (sccache daemon) inherits the FD and pins the slot after exit.

### 5. Resilience / canary
- Fail-open everywhere: `CARGO_MAKEFLAGS` exported only if FIFO exists; plugin/PSI all no-op on error.
- `jobserver-canary.sh` (5-min timer): C2 sum invariant `FIONREAD(merge)+FIONREAD(task)+held_back == nproc` when idle; if either FIFO missing → restart daemon; else require idle across 3×5s samples, then if `sum+held_back < SEEDED` → restart to re-seed. Tokens leak when rustc is SIGKILLed (orchestrator restart) and loses its held token. `held_back` file subtracts intentional reservoir.

### 6. cgroup — `lib_cgroup.sh` / `cpu-governed-exec.sh`
- cgroup-v2 `cpu.weight` placement: task slice weight 100, merge slice 300 (3:1 under contention, work-conserving — a lone scope absorbs the box). Careful support detection (cpu controller must be *delegated* to the user manager). Fail-open to plain `nice`.

## Anti-patterns Reify documented (MUST avoid in the port)
- **Load-average-derived worker counts** → positive-feedback collapse to N=1 (the throttle measures run-queue length, which is fed by the work it throttles). `cargo-test-occt-gated.sh:114-121`. Use a fixed cap / token pool, not `nproc − load`.
- **FD inheritance** by daemons pinning locks/tokens → host-wide wedge (the `9<&-` invariant).
- **Admit-on-timeout that requeues** → resubmission storms (reify moved compile admission to a never-requeue continuous hold with clock-stop markers).

## Port-critical translation notes (the core design question for the PRD)
- Reify gates each **build job** (rustc = jobserver client). DF's fragile quantity is the **xdist worker count**, and xdist workers are NOT jobserver clients. Faithful port ⇒ make the *worker count* the gated quantity, e.g.:
  - Derive `-n <k>` per verify from a per-lane token grant (merge lane gets a larger grant than a task lane), instead of `-n auto`; and/or
  - A dual-pool (merge/task) PSI-reactive balancer that shrinks the *task* worker budget under `/proc/pressure/cpu` avg10 while protecting a merge reservation.
- Reusable-as-is: hysteresis thresholds (hold 50 / release 40 cpu; memfull 10); merge/task 3:1 partition; FIONREAD non-destructive sensing; held_back + canary sum-invariant; `fleet-load-detector.sh` (the DF seam); cgroup weight scheme (300/100); `shared.psi`.
- "Make generally available through DarkFactory" (user ask): candidate = **extract the primitives to a shared package** (e.g. `shared/` or a new `oversubscription`/`jobserver` package) usable by every DF project's verify, rather than reify-private shell scripts — decide shell-in-place vs Python-daemon vs shared-lib in the design task.

### Key files
Reify: `scripts/{jobserver-balancer.py, jobserver-canary.sh, cpu-admit.sh, lib_slot_acquire.sh, lib_test_semaphore.sh, fleet-load-detector.sh, lib_cgroup.sh, cpu-governed-exec.sh, verify.sh}`; `~/.config/systemd/user/{reify-jobserver.service, reify-jobserver-canary.timer}`.
DF: `~/.config/systemd/user/pytest-jobserver.service`; `shared/src/shared/{pytest_jobserver.py, psi.py}`; `orchestrator/src/orchestrator/{verify.py, config.py, workflow.py, scheduler.py}`; `orchestrator/config.yaml`; `orchestrator/pyproject.toml`.
