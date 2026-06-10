# Lever C (multi-host verify) — enable-path code gap

**Status: C CANNOT BE ENABLED BY CONFIG. The operator enable path was never built.**
Found 2026-06-10 while executing the laptop-provisioning brief
(`~/.claude/spawn-briefs/setup-laptop-for-c-2026-06-10.md`). The brief's premise — "all of
C's code is already built and on dark-factory main … there is no code to write" — is
**false**. Filed as df task **1716** (see below).

## What exists on main (real, tested)

| Piece | Where | State |
|---|---|---|
| `MergeVerifySpec`/`VerifyResult` JSON codecs (α) | `verify_runner.py` | ✅ wired |
| `LocalRunner` + `VerifyRunnerPool` + pool routing (β) | `verify_runner.py`, `merge_queue.py:664,7133` | ✅ wired (verify events carry `runner=local` since 2026-06-10) |
| `orchestrator verify-merge` host CLI (γ) | `cli.py:279` | ✅ wired (proven on both hosts during provisioning) |
| `RemoteRunner` + `RunnerUnavailable` fail-safe fallback (δ) | `verify_runner.py:592,749` | ✅ class exists, fallback logic in `pool.dispatch` |
| K-permit semaphore + liveness recompute (ζ) | `merge_queue.py:4421,4480` | ✅ code paths exist, exercised by tests |
| Per-host serial-lane guard `num_hosts` param (η) | `merge_queue.py:7380` | ✅ parameterized |
| `DriftDetector` + quarantine API (ι) | `verify_runner.py:1261,790` | ✅ class exists |
| `SccacheConfig` + `effective_verify_env` (κ) | `config.py:379,1350` | ✅ wired into `build_merge_verify_spec` |
| `EnvFingerprint` / parity-report machinery (ε code) | `verify_runner.py:922–1438` | ✅ exists |

## What does NOT exist (the gap)

1. **No `verify_runners` config key anywhere in the repo** (PRD mechanism 12, D1's
   "reify opts in via a config knob"). `grep -r verify_runners` over *.py/*.yaml → zero hits.
   `OrchestratorConfig` has `extra='ignore'` (config.py:1414), so writing the key into
   reify's yaml today would be **silently inert** — worse than crashing.
2. **`RemoteRunner` is never constructed on any production path.** Both pool
   construction sites are hardwired `[LocalRunner(...)]`:
   `merge_queue.py:664` (`_run_post_merge_verify`) and `:7133` (cold shadow verify).
3. **K is pinned to 1** at the harness call site — `harness.py:3251`:
   `_k: int = _MERGE_AHEAD_BOUND` with the comment *"wiring K from config is a follow-up
   task"*. That follow-up task was never filed (verified against the task tree 2026-06-10).
4. **`enforce_persistent_worktree_serial_lane` is called with default `num_hosts=1`**
   (harness.py:3269) — η's per-host reframe is reachable only from tests.
5. **`DriftDetector` is never constructed outside tests** — the standing fidelity
   guarantee (ι) has no production wiring, no cadence config.
6. **Task tree says all of 1693–1702 are `done`, including λ** (the end-to-end
   integration gate whose signal requires *both runners live*: `runner=laptop` events,
   measured throughput delta). With RemoteRunner unreachable in production, λ's signal
   cannot have been observed. ε's deliverable (`docs/verdict-parity-report.md`) was
   likewise a runbook template, not a record of a run (known before this session; the
   real report now replaces it). **Process-integrity note:** leaf "done" markers in this
   batch did not correspond to user-observable signals — and the false marker is
   self-reinforcing: on 2026-06-10 the curator **dropped** a candidate task
   ("λ-gate completion criterion: prove two overlapping verify spans end-to-end") with
   reason "task 1702 is already done and delivered exactly this"
   (tkt_0RPYGYVEC262Q24BQA0HYMQ7F2). Wrongly-done markers actively suppress corrective
   work; 1716's description is written to survive that curator pass.

## New fidelity hazard found during the real provisioning (must be fixed in the wiring)

**Remote-side scope derivation depends on the remote repo's local `main` ref.**
`run_merge_verify_on_worktree` → `run_scoped_verification` derives task files via
`git diff main...HEAD` *in the remote worktree* (`verify.py:729`,
`_derive_task_files_from_git`) because production merge requests for reify carry
`task_files=None` and `module_configs=[]`. Observed live on the laptop: a stale `main`
ref derived **2204 files** → verification mode flipped from production's
`global (no scope info)` to `fallback-scoped (2204 files)` → a *different command set*.
A sufficiently stale/diverged remote `main` can select commands that **skip the Rust
suite entirely** (`_build_fallback_config` is Python-oriented) → remote PASS that would
be a local FAIL → unverified code on main. This is exactly the design §6.3 / D6 hazard,
in a place the PRD didn't anticipate.

**Fix direction (in task 1716):** derive `task_files` on the *authoritative* (dispatching)
host and ship them in the spec (`MergeVerifySpec.task_files` already exists on the wire),
so the remote never derives scope from its own refs; additionally have `RemoteRunner`
push the current main ref alongside the merge sha (keeps the remote repo fresh and
pushes thin).

## What IS now provisioned and proven (operational, this session)

- `leo-laptop`: rustc **1.96.0 (ac68faa20)** (== workstation), sccache 0.14.0,
  cargo-nextest 0.9.136 / tree-sitter 0.26.8 / uv 0.10.6 (binaries copied from
  workstation), OCCT 7.8 apt packages identical versions, `/opt/reify-deps` rsynced
  (3.5G incl. manifold prebuilt 3.5.101), node v22.22.3 (==), 16-token jobserver FIFO
  unit (`reify-jobserver.service`, laptop-local), df checkout at `0f6ee5644b` + synced
  venv, reify checkout with `main` ref tracking `workstation/main` (old laptop WIP
  preserved on branch `laptop-wip-archive-2026-06-10`).
- Transport proven both ways: `git push leo-laptop <sha>:refs/merge-verify/<id>` ✅,
  `ssh leo-laptop orchestrator verify-merge --help` ✅ (wrapper at
  `/usr/local/bin/orchestrator`), laptop→workstation fetch ✅.
- Laptop host config: `~/.config/orchestrator/reify-laptop.yaml` (mirrors reify
  verify commands/env; this file is **load-bearing**: with `module_configs=[]` the
  spec's `verify_env` is NOT applied — the remote host's config supplies env+commands
  in the global-fallback path).
- Shared sccache backend: redis 7 (docker `sccache-redis`, workstation, port **6380**
  — 6379 is FalkorDB), bound to 127.0.0.1 + tailscale IP, 16gb allkeys-lru, no
  persistence. Laptop reached it (`Cache location redis…` in `--show-stats`).
  **Not yet active in production env** (needs `sccache:` block in reify yaml at enable
  time + an sccache server restart on the workstation at a quiet moment — the running
  server's backend is fixed at server start).
- Verdict-parity proof: **in progress** at write time over a 5-SHA corpus
  (2 known-pass, 1 historical fail [environmental-looking], 2 synthetic deterministic
  fails). Results → `docs/verdict-parity-report.md` (replacing the template).

## Staged (NOT enabled)

- reify `rust-toolchain.toml` pin (1.96.0 + rustfmt/clippy) — committed after its full
  hook verify passes (both hosts have 1.96.0 pre-installed so the pin is a no-op flip).
- Proposed runner-pool config staged as **comments** in reify `orchestrator.yaml`
  (schema to be settled by task 1716; `extra='ignore'` makes a real block silently
  inert today).

## Enable checklist (after task 1716 lands on df main)

1. Parity report green over the corpus (the HARD GATE — see
   `docs/verdict-parity-report.md`).
2. Uncomment/write the real `verify_runners` block in reify `orchestrator.yaml`
   per 1716's landed schema (laptop: ssh_host `leo-laptop`, git_remote `leo-laptop`,
   config_path `/home/leo/.config/orchestrator/reify-laptop.yaml`), K=2, drift cadence,
   `sccache.backend_env.SCCACHE_REDIS: redis://leo-workstation.tailb08a6b.ts.net:6380`.
3. Commit (dirty-tree-safe), `systemctl --user restart orchestrator-reify.service`,
   confirm NRestarts stable + "Speculative merge worker started".
4. Fault-inject (laptop off) → confirm `RunnerUnavailable` → local fallback, queue does
   not stall. Watch for `runner=laptop` verify events + `verdict_parity_ok`.
5. Run λ's real signal: one window of two-overlapping-verify spans + throughput delta
   vs single-host baseline.
