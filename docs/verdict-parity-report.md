# Verdict Parity Report — leo-laptop as a Lever C verify host (REAL RUN, 2026-06-10)

This document records the **actual** provisioning of `leo-laptop` and the
verdict-parity proof against the workstation (`leo-MS-7C35`). It replaces the
earlier runbook template, which pinned a wrong toolchain (1.80.0), used
placeholder hostnames/SHAs (`abc1234`), and presented a fabricated "✅ PASS"
results table for a run that never happened. Companion gap report:
`plans/lever-c-enable-path-gap-2026-06-10.md` (why C is **not yet enabled**:
the operator config wiring does not exist — df task 1716).

## 1. Environment fingerprints (captured via `capture_env_fingerprint`, 2026-06-10)

| Probe | Workstation `leo-MS-7C35` | Laptop `leo-laptop` | Match |
|---|---|---|---|
| rustc | 1.96.0 (ac68faa20 2026-05-25) | 1.96.0 (ac68faa20 2026-05-25) | ✅ |
| cargo | 1.96.0 (30a34c682 2026-05-25) | 1.96.0 (30a34c682 2026-05-25) | ✅ |
| OS | Ubuntu 24.04.4 LTS | Ubuntu 24.04.4 LTS | ✅ |
| kernel | 6.17.0-29-generic | 6.17.0-22-generic | ⚠ patchlevel only (ws apt-upgraded 06-10, laptop 06-09) |
| nproc | 32 | 16 | capacity, not fidelity |
| sccache | 0.14.0 (reachable) | 0.14.0 (reachable) | ✅ |
| cargo-nextest | 0.9.136 (1d5bf1ec9) | 0.9.136 (1d5bf1ec9) | ✅ |
| node | v22.22.3 | v22.22.3 | ✅ |
| tree-sitter | 0.26.8 | 0.26.8 | ✅ |
| OCCT apt (libocct-foundation-dev) | 1:7.8.1+dfsg1-3~ubuntu24.04.1 | identical | ✅ |
| OCCT runtime (/opt/reify-deps) | libTKernel.so.7.9 | libTKernel.so.7.9 (rsynced byte-identical) | ✅ |
| manifold prebuilt | 3.5.101 v3.5.0 | 3.5.101 v3.5.0 | ✅ |
| Tauri/webkit2gtk dev pkgs | 2.52.3-0ubuntu0.24.04.1 | identical versions; librsvg2/xdo/appindicator dev identically ABSENT on both | ✅ |
| jobserver FIFO | /tmp/reify-jobserver (32 tokens, reify-jobserver.service) | /tmp/reify-jobserver (16 tokens, laptop unit) | ✅ present (token count = host capacity) |
| verify_env | RUSTC_WRAPPER=sccache, CARGO_INCREMENTAL=0, CARGO_MAKEFLAGS=jobserver fifo | identical | ✅ |

The reify repo now pins the toolchain (`rust-toolchain.toml`, commit
`15c44f10ac`): channel **1.96.0**, components rustfmt+clippy — pre-installed on
both hosts before the pin landed, so fidelity is rustup-enforced going forward.

## 2. Method

- Same wire protocol as production `RemoteRunner` (δ): SHAs pushed from the
  workstation reify repo via `git push leo-laptop <sha>:refs/merge-verify/<label>`,
  executed on the laptop via `orchestrator verify-merge --sha <sha> --spec <spec>
  --config ~/.config/orchestrator/reify-laptop.yaml` (the γ CLI — the same
  orchestrator verify code as the merge queue, fidelity by construction).
- The spec is the exact production projection: built with
  `build_merge_verify_spec(load_config(reify orchestrator.yaml),
  module_configs, task_files=None)` — reify's production shape is
  `module_configs=[]` + `task_files=None`, so the host-side config supplies
  commands and env via the global-fallback path (see §5 finding 2).
- Workstation reference verdicts: for corpus rows that went through the
  production merge queue **on 2026-06-10 (after β's pool routing landed)** the
  recorded production verdict is the workstation verdict (same code path:
  `pool.dispatch` → LocalRunner). Synthetic rows and the re-run row were
  executed on the workstation through the same `orchestrator verify-merge` CLI
  as the laptop.
- One verify at a time per host (per-host serial invariant). No shared sccache
  backend during the proof (production parity; redis stays staged — §4).

## 3. Corpus and verdicts

Laptop runs are round 2 (post `libtbb-dev` fix), 16:16–17:11 BST. Laptop pass
runs are full-fidelity: debug suite 15416/15416 + release suite 7943/7943 +
clippy + GUI/npm steps, ~12 min wall compile-warm.

| # | Label | SHA | Provenance | Workstation verdict | Laptop verdict | Parity |
|---|---|---|---|---|---|---|
| 1 | pass-1 | `fceaf142d7c71ec59bffa007f6587e0fa3547eee` | Merge task/4501 into main (landed 2026-06-10 14:14 UTC, production verify PASS) | PASS (production, runner=local) | **PASS** (15416/15416 + 7943/7943) | ✅ |
| 2 | pass-2 | `691f1fc7a06d83c77a0856b18ea94fdc6273366a` | Merge task/4402 into main (landed 2026-06-10 14:06 UTC, production verify PASS) | PASS (production, runner=local) | **PASS** | ✅ |
| 3 | fail-1 → reclassified pass | `5008d168083c5a4539da80f846e9cb9f9766bf52` | task 4428's merge candidate; production FAIL 14:04 UTC was **workstation-side transient infra** (corrupt dep-info in a polluted worktree) — the clean ephemeral `verify-merge` re-run PASSES | **PASS** (clean re-run, 44 min, "All checks passed") | **PASS** | ✅ (clean-run vs clean-run) |
| 4 | synth-clippy | `42fa210826e68e94f0a7e5b704bb3be6a5c25926` | synthetic: child of pass-1 + unused variable (denied lint must fail); ref `refs/parity/synth-clippy` | **FAIL** ("tests failed, lint issues") | **FAIL** ("tests failed, lint issues") | ✅ FAIL==FAIL |
| 5 | synth-test | `0bcc9b1304fce9f5be4b384d9543b24e6b46d8d7` | synthetic: child of pass-1 + `assert!(false)` test; ref `refs/parity/synth-test` | **FAIL** (ephemeral re-run 17:19–17:34 BST; first run INVALID per §5 finding 6) | **FAIL** | ✅ FAIL==FAIL — **same failing test** (`reify-expr _lever_c_parity_synthetic::synthetic_parity_proof_deliberate_failure`), same 15417-test suite on both hosts |

**GATE STATUS (final, 2026-06-10 17:34 BST): ✅ VERDICT PARITY PROVEN — 5/5
rows agree (round 2), including failing-test-level identity on the synthetic
test row.** The laptop is trustworthy as a verify host. C nevertheless stays
**DISABLED** until the operator enable path exists (df task 1716, in
progress); this report satisfies the D6 trust gate for that flip. The corpus
SHAs remain protected under `refs/parity/*` (workstation reify repo) for
re-proof / drift-detector seeding.

### Round 1 (2026-06-10 15:37–16:12 BST): **PARITY FAILED — gate held.**

All five laptop runs failed (~5 min each), including both known-pass SHAs:
every run died at the `cargo nextest` link step with undefined
`tbb::detail::d2::*` symbols in `reify-kernel-occt` test binaries (misreported
as category `tree_sitter_generate_error` — see §5 finding 5). Root cause:
the laptop had **`libtbb-dev`** installed (the workstation does not), whose
`/usr/lib/x86_64-linux-gnu/libtbb.so` linker symlink (system oneTBB 2021.11,
d1 ABI) shadowed `/opt/reify-deps/lib/libtbb.so` (d2 ABI — what the openvdb
wrapper's `/opt/reify-deps/include/oneapi/tbb` headers compile against).
Fix: `apt-get remove libtbb-dev` on the laptop (match the workstation).
Round-1 artifacts preserved at `leo-laptop:~/parity/results-round1-tbbdev/`.
fail-1's round-1 "FAIL==FAIL agreement" was coincidental (different root
cause) and is disregarded; round 2 re-runs the full corpus.

## 4. Shared sccache backend (κ) — provisioned, staged

redis 7 (docker `sccache-redis`, `--restart unless-stopped`) on the
workstation, port **6380** (6379 is FalkorDB), bound to 127.0.0.1 + the
tailscale IP only, `maxmemory 16gb` / `allkeys-lru`, no persistence. Proven:
laptop `sccache --show-stats` reports `Cache location redis,
name: redis://leo-workstation.tailb08a6b.ts.net:6380` (tailnet flows direct
over the LAN). Activation is via the `sccache:` block staged (commented) in
reify `orchestrator.yaml` — note the workstation's long-running sccache server
binds its backend at server start, so flipping it needs `sccache
--stop-server` at a quiet moment.

## 5. Findings (env-fidelity hazards surfaced by doing this for real)

1. **The template was wrong**: it pinned `rustc 1.80.0`; the real verify
   toolchain is 1.96.0. A laptop provisioned from the template would have
   failed parity immediately (or worse, drifted silently).
2. **Remote scope derivation is ref-dependent** (load-bearing; now in task
   1716's scope): with reify's production spec shape, the *remote* host
   re-derives task files via `git diff main...HEAD` against **its own** `main`
   ref (`verify.py:729`). Observed live: the laptop's stale `main` derived
   2204 files and flipped verification mode `global` → `fallback-scoped` — a
   different command set. Mitigated for this proof by syncing the laptop's
   `main` ref (now tracks `workstation/main`); must be fixed structurally in
   1716 (ship task_files in the spec; push main alongside the merge sha).
3. **The remote host's `--config` is load-bearing** for projects with
   `module_configs=[]`: the spec's `verify_env` is *not* applied on the remote
   in the global-fallback path — commands+env come from the laptop-side config
   (`~/.config/orchestrator/reify-laptop.yaml`), which must mirror production.
   (Also folded into 1716.)
4. **An *extra* dev package is as dangerous as a missing one** (round-1 parity
   failure): `libtbb-dev` on the laptop shadowed the `/opt/reify-deps` oneTBB
   at link time and broke every verify. Dev-`.so` symlinks under `/usr/lib`
   compete with `/opt/reify-deps/lib` for `-l<name>` resolution; host alignment
   must consider package-set *differences in both directions* for every library
   reify links by name (tbb, openvdb, gmsh, occt, slvs, manifold).
5. **Verify failure-category attribution is unreliable for early-pipeline
   parallel failures**: a nextest-phase linker failure was categorized
   `tree_sitter_generate_error` on the laptop, and the workstation's production
   failure for the same SHA cited `check-manifold-deps.sh` while its journal
   showed a tree-sitter dep-info error. Verdict-level parity (pass/fail) is the
   gate; category strings are diagnostics, not contract.
6. **`orchestrator verify-merge` collides with a live orchestrator's warm
   worktree** (appended to task 1716 as a hard requirement): when
   `git.persistent_merge_worktree` is on (reify production enabled it
   2026-06-10 16:37 BST, mid-proof), the CLI acquires the SAME fixed-path
   `_merge-verify` worktree the live merge queue owns. Observed: production
   reset the worktree to its own candidates three times during a CLI parity
   run, which then reported a false 15418-tests-PASS for a tree that
   deterministically fails. Any CLI/drift-detector local leg on the
   orchestrator host must use an ephemeral worktree or take the serial-lane
   lock. (The two production verify failures overlapping that window — 4455,
   cargo-run-prebuilt-fix — were checked and are NOT contamination: both have
   a deterministic `make_fixture: lib_test_semaphore.sh` infra-test cause that
   also reproduced in a per-task worktree, task 4474.)

## 6. Laptop operational notes

- `ssh leo-laptop orchestrator verify-merge …` works for non-interactive ssh via
  the wrapper `/usr/local/bin/orchestrator` (sets PATH, execs the df venv CLI).
- Laptop df checkout tracks df main (`0f6ee5644b` at proof time) — **keep it
  synced**; version skew between hosts' orchestrator code is a drift vector
  (the drift detector ι catches verdict-level effects once wired).
- Laptop reify old WIP preserved on branch `laptop-wip-archive-2026-06-10`
  (plus pre-existing stashes); `main` is a plain ref tracking
  `workstation/main`, no checkout.
- Power: `sleep-inactive-ac-type='nothing'` — the laptop stays awake on AC
  while open. Lid-close suspend is untouched: a closed laptop disappears, which
  the pool's fail-safe (`RunnerUnavailable` → local fallback) must tolerate by
  design; cap-of-uplift, not a correctness issue.
- Throughput expectation: 16 threads vs 32, cold-leaning caches → ~0.4–0.6×
  workstation verify speed until the shared sccache backend is activated
  (then ~0.7–0.85× compile-warm; test execution stays at laptop speed).
