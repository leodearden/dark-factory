# Stage-1 results: laptop task-verify build warmth (2026-11-17)

Experiment on `leo-laptop`, 17:29Z–20:10Z. Scripts: `stage1/exp.sh`, `stage1/snap.py`,
`stage1/gov.sh`. Raw data: `stage1/results/`. Independent re-derivation: `stage1/verify/`.
Pre-flight review: `stage1/preflight/P-review.md`. Background study (seats A, B, critic):
`A-build-warmth.md`, `B-df-host-locality.md`, `critic/E-critique.md`.

## Method

Build-only replay of a reify task verify's four cargo steps (`cargo nextest run --no-run` for
the debug and gui test builds, `cargo clippy`, `cargo check` for gui), with the commands taken
from `scripts/verify.sh test|lint --scope branch --include-infra --print-plan`. Private
`git clone --shared` on the laptop's second disk, private sccache server (port 4227, own
initially empty cache), off the shared cargo jobserver (`CARGO_BUILD_JOBS=8`), in a systemd
user unit with `CPUWeight=30`. A governor froze the unit whenever a production verify waited
at `verify.sh psi-gate` or `compile-gate` (16 freezes, 998 s in total).

X = `27c69f616f` (task 10019), Y = `1866156735` (task 10010): the tips of two different tasks
that the laptop verified back to back on 2026-11-17, both with scope `-p` (24 packages) plus
the gui step. They share a merge-base with main and differ in 8 files (4 in
`crates/reify-compiler`, 4 in `crates/reify-test-support`).

## Figures

CPU = delta of the experiment cgroup's `cpu.stat usage_usec` over the four steps (cargo,
rustc, the private sccache server and its compilers; nothing else). Hits/misses = deltas of
the private server's counters, all languages. Packages = distinct package names on cargo
`Compiling`/`Checking` lines.

| Run | Commit | Path | Target at start | CPU-min | Hits / misses | Non-cacheable compiles | Distinct packages |
|---|---|---|---|---|---|---|---|
| R1 | X | P (new) | empty | 110.6 | 2 / 1,148 | 1,436 | all |
| R2 | X | Q (new) | empty | 90.8 | 728 / 422 | 1,436 | all |
| R3 | X | P | emptied | 50.3 | 1,150 / 0 | 1,436 | all |
| R5 | Y | Q2 (new) | empty | 92.8 | 728 / 422 | 1,438 | all |
| R4 | Y | P | kept from X | 37.5 | 0 / 21 | 1,205 | 24 |
| R6 | Y | P | kept, after `reset --hard` + `clean -xfd -e target` | 3.9 | 2 / 0 | 16 | 1 |
| R7 | Y | P | kept, nothing touched | 2.2 | 0 / 0 | 7 | 0 |

Execution order was R1, R2, R3, R4, R6, R7, R5.

Per step, R2 (new path): debug nextest build 356 hits / 138 misses; gui nextest build
109 / 122; clippy 179 / 58; gui check 84 / 104. Rust-only misses 406 of the 422; 16 are
C/C++/assembler.

## Readings

- **A new worktree path alone causes the misses.** R2's test leg (debug + gui nextest builds)
  is 465 hits / 260 misses of 725 requests. Production's own cold verify of X recorded exactly
  465 / 260 on its sccache server (ledger, no other cargo at either step boundary). At the same
  path with an emptied target (R3) nothing missed.
- **Fixed path, target still emptied: 44.6% less build CPU** (R2 → R3). Best case: R3 rebuilt
  the same commit. For Y after X the inference is about 21 misses instead of 0 (R4's 21 misses
  are the cacheable units whose inputs differ between X and Y); that arm was not run.
- **Kept target at a fixed path: 59.6% less build CPU** (R5 → R4, same commit). Test leg
  58.1% less, lint leg 65.9% less.
- **Why R4 still rebuilt about 600 units:** 599 units were dirty in the debug step, 412 of them
  stale through the `reify-compiler` library (changed file `src/import_headers/mod.rs`), 110
  through the `reify-test-support` library. The unit-index-to-crate mapping is inferred from
  the dependency graph.
- **An unchanged commit rebuilds almost nothing** (R7: 0 packages). The acquire-style reset and
  clean costs one package (R6): `clean` deletes the gui sidecar placeholder, which is recreated
  with a new mtime and reruns `reify-gui`'s build script.
- **A debug-profile target is 64 GiB** after the four steps.

## Limits

- One pair of commits.
- Wall time is not comparable across runs (load 39–115, freezes). Converting by CPU ratio,
  production's 19.5 build minutes (13.8 test leg + 5.7 lint leg, medians) would be about 11
  with a fixed path and about 8 with a kept target. That is a conversion, not a measurement.
- Why the path enters sccache's key was not read from sccache's source.
- About 110 CPU-seconds per debug step in R6 and R7 sit outside cargo's timed region and are
  unexplained.
- The private server's keys may differ from production's (`CARGO_BUILD_JOBS`, `CARGO_LOG`,
  `CARGO_HOME` differ); they are the same in every arm.
