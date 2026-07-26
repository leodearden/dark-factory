# Warm-lane infrastructure repatriation — redraw the reify↔dark-factory seam at the toolchain boundary

**Status:** active · **Owner:** dark-factory · **Authored:** 2026-07-26
**Approach:** B+H (contract + two-way boundary tests) — cross-repo seam, two-repo blast radius, touches the ENOSPC accretion path.

## 1. Goal

Move the ~2,200 lines of **project-agnostic** warm-lane pool machinery from `reify/scripts/` into dark-factory, leaving only the genuinely toolchain-bound primitive in reify behind an explicit contract; then use dark-factory's own lane-assignment state — rather than two lossy filesystem proxies — as the reclaim gate.

What an operator observes when this lands:

- `warm-lane-gc.sh reclaim`, invoked from **any** entry point (dark-factory's cadence pass, the systemd sweep, a manual operator run), never resets the `target/` of a lane that is assigned to a live task. Today only the systemd sweep gets that protection.
- The reclaim log names *why* a lane was preserved with an authoritative reason (`preserving _lane-5: assigned to task 5334`) instead of a probe result (`live consumer (flock held)`).
- `reify/scripts/` contains exactly two warm-lane scripts (`seed-warm-lane.sh`, `refresh-warm-base.sh`) and reify's verify gate no longer escalates on pool-policy changes.
- One warm-lane test estate, in dark-factory, instead of 23 bash tests in reify plus 28 Python tests in dark-factory.

## 2. Background — the repo boundary is the defect

Root-caused 2026-07-26 (`/deb`, reify esc-5334-6, resolved; fix filed as reify task 5572).

Task 5334's agent ran `env cargo test -p reify-eval --no-fail-fast` in `_lane-5` at 2026-07-25T22:36:44Z; 218 targets failed `No such file or directory` on binaries cargo had just reported *Running* (22:41:31Z). dark-factory's `_run_warm_lane_gc_reclaim` was mid-pass across that entire window (boundaries 23:05:50 / 23:50:26 BST) and reset the lane's `target/` underneath it.

Three facts make this a boundary problem, not a bug:

1. **The guard that would have stopped it exists, in the wrong place.** reify task 5378 added a `/proc` live-consumer scan — but put it in `warm-lane-gc-sweep.sh`, a reify-side *wrapper*. dark-factory calls `warm-lane-gc.sh` directly (`git_ops.py:3922`, `--mount` only, no `--extra-protect-glob`, ever). One operation, two entry points, one of them protected.

2. **The script's own header documents the boundary as a design rule.** `warm-lane-gc.sh:130-135`: *"Reclaimability is computed purely from filesystem + git + flock; dark-factory FREE/ASSIGNED state is NOT consulted. 'FREE/idle' ≈ no live consumer holding the lane flock."* That `≈` is false. The flock is held only across the acquire reseed (reify 5354) and across `run_scoped_verification` (DF 3027) — never across the implement phase, where an autonomous agent builds and tests for tens of minutes. Live-verified 2026-07-26: `_lane-27`, `_lane-28` (mid `cargo`+`cargo-clippy`), `_lane-50` all had live consumers and **zero** held `<lane>.lock`.

3. **The authoritative answer was in the same directory the whole time.** `<worktree_base>/.lane-state/<lane>.json` — dark-factory's durable `LaneLifecycle` record — carries `{state, task_id, branch, updated_at}`. At the moment of the live check, `_lane-28.json` read `{"state": "assigned", "task_id": "5551"}` and `_lane-50.json` read `{"state": "assigned", "task_id": "5416"}`. `warm-lane-gc.sh` runs *in that directory* and does not read those files, because a rule adopted when the script lived in the wrong repo forbids it.

So `FREE ≈ flock-free` is an approximation that exists only because the code was placed where the real answer was unavailable. Task 5326's always-reclaim policy is written for *"a FREE pool lane"* and is implemented against a predicate that cannot distinguish FREE from ASSIGNED. Every subsequent fix (5378's `/proc` scan, 5354's acquire lock, DF 3027's verify lease, reify 5572's per-lane rescan) is a better proxy for a fact that is already recorded on disk.

**Scale.** `_lane-5` was reset ≥6 times on 2026-07-25 while assigned to task 5334. Most landed between agent commands and only destroyed warm build state (the agent then pays a full rebuild) — an invisible throughput cost. The 218-ENOENT storm is the rare mid-run case.

### 2.1 The specificity audit that justifies the split

Token scan for `cargo|rustc|RUSTFLAGS|OUT_DIR|Cargo|nextest|occt|manifold|reify-gui|tauri`:

| script | lines | tokens | disposition |
|---|---|---|---|
| `warm-lane-gc.sh` | 585 | 0 | → dark-factory |
| `warm-lane-gc-sweep.sh` | 441 | 2 (comments) | → dark-factory |
| `thin-warm-lane.sh` | 288 | 0 | → dark-factory |
| `warm-lane-disk-guard.sh` | 297 | 0 | → dark-factory |
| `warm-lane-audit.sh` | 585 | 0 | → dark-factory |
| `warm-lane-degenerate-ref-check.sh` | — | 0 | → dark-factory |
| `provision-warm-lane-fs.sh` | 460 | 0 | → dark-factory |
| `refresh-warm-base.sh` | 447 | 7 — all one RUSTFLAGS stamp | stays in reify; stamp generalized |
| **`seed-warm-lane.sh`** | **1009** | **57** | **stays in reify — the real primitive** |

`seed-warm-lane.sh` is genuinely toolchain-bound: RUSTFLAGS fingerprint assertion, mtime stamping to defeat cargo fingerprinting, OUT_DIR relocation, `env!()`-baked-path relinking, `tauri-*`/`reify-gui-*` non-relocatable build-script dirs, `reify-kernel-occt`/`reify-kernel-openvdb`. `warm-lane-gc.sh`'s only project couplings are the literal path segment `target` and a `main` default that is already a `--main-ref` flag.

## 3. Substrate verification (G3)

| Assumed capability | Verdict | Evidence |
|---|---|---|
| Durable per-lane state readable from a shell script | **CONFIRMED** | `<worktree_base>/.lane-state/<lane>.json`, plain JSON, `{state, task_id, branch, seeded_from_sha, updated_at}`; 56 live records read 2026-07-26 |
| `LaneState` distinguishes FREE from ASSIGNED | **CONFIRMED** | `lane_lifecycle.py:55-63`; live census 55 × `assigned`, 1 × `released` |
| `LaneState.IN_USE` distinguishes *building* from *assigned-idle* | **ABSENT — declared-only** | `IN_USE` is in the enum (`:61`) and the legal-transition table (`:77-78`) and is *read* in `harness.py`, but **nothing ever writes it**: 0 writers, 55/55 live records are `assigned`. Any policy predicated on `IN_USE` today is a fiction. → leaf **δ** populates it. |
| dark-factory can ship + resolve bash scripts | **NEEDS DESIGN** | `orchestrator/scripts/` exists but holds only two helpers; packaging/resolution is leaf **α**'s contract work (tactical detail in Open questions) |
| systemd unit can be repointed | **CONFIRMED** | `reify/deploy/systemd/reify-warm-lane-gc.service:28` `ExecStart=/home/leo/src/reify/scripts/warm-lane-gc-sweep.sh` — a hardcoded path, one-line change (leaf **η**) |
| Status-oracle callback pattern has precedent | **CONFIRMED** | `REIFY_LANE_LEAK_STATUS_CMD` + `reify/scripts/lane-task-status.sh` — *not needed* for this design (records are readable directly) but establishes the idiom |

**The `IN_USE` finding is load-bearing.** Without it, "reclaim assigned-but-idle lanes" (which task 5326 requires, to prevent ENOSPC accretion) has no signal. This is why leaf δ must land before the whole-assignment hold (leaf ε) — otherwise ε makes every assigned lane permanently unreclaimable and re-creates the 2026-07-10 ENOSPC outage.

## 4. Resolved design decisions

- **D1 — dark-factory owns the PRD and the pool.** Confirmed by Leo 2026-07-26. reify-side leaves are filed in reify's tracker and wired as real cross-repo `add_dependency` edges, not prose ordering.
- **D2 — Scope is relocate + liveness swap + whole-assignment hold**, phased, in one PRD. The narrower options were rejected: a follow-on PRD for the payoff is exactly what happened to reify task 5379, which named this defect precisely and was cancelled without the residual being carried forward.
- **D3 — dark-factory's implementation is the default; a project override hook remains.** `project_root/scripts/<name>.sh` still wins if present. This is mandatory transiently (it is what makes the safe migration ordering possible) and is retained afterwards as a documented escape hatch, not a default.
- **D4 — Relocated scripts stay bash.** They test bash; a Python rewrite of ~2,200 working lines buys idiom uniformity at the cost of the largest single risk item in the migration. reify's generic `tests/infra/test_warm_lane_*.sh` port to dark-factory **as bash**, run from dark-factory's suite.
- **D5 — Liveness is read per lane, immediately before reset, from the on-disk record, under the lane flock.** Not marshalled as a CLI snapshot: an `--assigned-lanes CSV` argument would re-create exactly the up-front-snapshot TOCTOU that reify 5572 exists to fix. The records are files in the directory being swept; read them there.
- **D6 — The flock's role narrows and is written down.** After δ/ε it is no longer a liveness oracle. It becomes (a) the mutex serialising the reseed operation itself, and (b) the cross-process guard for non-dark-factory actors (the systemd sweep, manual operator runs, reify's own scripts). Liveness comes from the record.
- **D7 — reify task 5572 lands first and is not duplicated here.** Its per-lane live-consumer check is the stop-gap; its logic migrates with the script at leaf α and is *superseded* (not deleted) by the record read at leaf γ. The `/proc` scan is retained as a belt-and-braces second predicate for non-dark-factory-managed lanes (`_iact-*`, manual worktrees) which have no durable record.

## 5. Contract — the redrawn seam (H)

The seam moves from *"reify ships the primitive, dark-factory wires the invocation"* (which put policy in reify) to a toolchain boundary.

**dark-factory owns:** pool sizing and lifecycle, lane state, reclaim policy, liveness determination, disk-pressure admission, orphan removal, audit, provisioning, and every script implementing those.

**The project owns exactly one primitive**, plus its base-advance counterpart:

```
<project_root>/scripts/seed-warm-lane.sh <base_cache_dir> <lane_dir> \
    [--fresh-checkout | --reset-in-place] [--lane-lock | --assume-lane-lock-held]
```

| Contract element | Requirement |
|---|---|
| **Effect** | Materialise a warm build cache for this project's toolchain at `<lane_dir>/<cache-dir>` from `<base_cache_dir>`, such that a subsequent build reuses it and produces byte-identical results to a cold build |
| **Cache dir name** | Declared by the project, not assumed. dark-factory passes it and never hardcodes `target` |
| **Build fingerprint** | The project asserts its own config fingerprint (reify: RUSTFLAGS) and **fails closed** on mismatch. dark-factory treats it as an opaque string via `--build-fingerprint` |
| **Exit codes** | `0` success · `75` disk pressure (EX_TEMPFAIL) · `124` lane-lock wait timeout · `127` absent/exception · other = script fault. dark-factory branches only on this taxonomy |
| **Locking** | Acquires `<lane_dir>.lock` `flock -x` by default; `--assume-lane-lock-held` opts out when the caller already holds it |
| **Idempotence** | Safe to invoke on an empty, partial, or fully-populated cache dir |
| **Purity** | Touches `<lane_dir>/<cache-dir>` only — never the source tree, never the branch ref |

`refresh-warm-base.sh` stays in reify under the same contract, with its RUSTFLAGS stamp generalised to `--build-fingerprint` (leaf θ).

**Invariant C-1.** dark-factory never inspects the contents of the cache dir; the project never inspects lane state. A violation in either direction is the defect class this PRD exists to close.

## 6. Boundary-test sketch (H) — facing both ways

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 | Assigned lane is never reclaimed, any entry point | `.lane-state/_lane-N.json` = `assigned`; flock free; no `--extra-protect-glob` | `reclaim` preserves the lane; `<lane>/<cache>` mtimes unchanged; log names the task id |
| B2 | TOCTOU — lane becomes assigned mid-pass | Lane `released` at pass start; flips to `assigned` before its own reset is reached | Preserved. Fails if the implementation snapshots state up front |
| B3 | Free lane still reclaims (5326 not regressed) | Record = `released`; divergent cache dir | Reset; disk reclaimed; `reset=N` in the summary |
| B4 | Assigned-but-idle still reclaims (post-δ) | Record = `assigned`, **not** `in_use`; no live descendant | Reset — the ENOSPC accretion path stays open |
| B5 | Assigned-and-building never reclaims (post-δ) | Record = `in_use` | Preserved |
| B6 | Recordless lane falls back to the `/proc` predicate | `_iact-*` worktree, no `.lane-state` record, live cwd inside | Preserved via the 5572 scan (D7 belt-and-braces) |
| B7 | Project override hook wins when present | Both dark-factory's copy and `project_root/scripts/warm-lane-gc.sh` exist | The project copy runs; resolved path appears in the log |
| B8 | Absent project script is not silent | Override absent, dark-factory impl present | dark-factory's impl runs at INFO — never the current `logger.debug` + rc-127 no-op |
| B9 | §5 contract honoured end-to-end | Stub project seed script: each of exit 0/75/124/127; a non-`target` cache-dir name; a mismatched build fingerprint | dark-factory branches per the taxonomy without inspecting stdout, passes the declared cache-dir name through, and surfaces the fingerprint refusal as a fail-closed error |
| B10 | Whole-assignment lease does not deadlock reclaim | Lease held for a full assignment; reclaim pass runs | Reclaim proceeds on other lanes; B4 still reclaims this one if idle |
| B11 | Fully-assigned pool degrades to backpressure, never to corruption | Every lane `in_use`; free space at the disk-guard floor | Reclaim resets **nothing**; `warm-lane-disk-guard.sh` returns 75 and dispatch admission blocks. No live lane's cache is reset under disk pressure |

B1 and B2 are the regression tests for esc-5334-6. B3 and B4 are the regression tests for the 2026-07-10 ENOSPC outage — **they must both stay green in the same run**, which is the whole point of doing δ before ε.

**Storm escape (INV-4).** Once γ+δ stop reclaim from touching live lanes, a fully-subscribed pool has nothing left to reclaim, and disk can still fall to the floor. The sanctioned escape is **admission backpressure, not reclamation**: `warm-lane-disk-guard.sh` exit-75 blocks dispatch (`_warm_lane_disk_admission_blocked`) and tasks requeue as transient infra. There is deliberately **no** override that reclaims an `in_use` lane under pressure — that trades an ENOSPC stall for silent build corruption, which is the failure this PRD closes. B11 is the executable form of that choice.

### 6.1 Design-invariant walk (G7, `docs/legibility/design-invariants.md`)

| Invariant | Disposition |
|---|---|
| **INV-1** `contracts-machine-checked` | §5 is machine-checked by **B9** (exit-code taxonomy, cache-dir passthrough, fail-closed fingerprint) against a stub project script — not prose alone |
| **INV-2** `structured-facts-at-failure` | γ emits the authoritative reason with the task id (`preserving _lane-5: assigned to task 5334`) instead of a probe result; **B8** converts the current silent `logger.debug` + rc-127 no-op into a logged fact |
| **INV-3** `corroborate-before-acting` | **This PRD is an INV-3 fix.** The defect is a sweep acting on a stale predicate. D5 forbids snapshot marshalling and mandates a per-lane re-read immediately before reset, under the flock; **B2** is the executable guard |
| **INV-4** `storm-escape-required` | Escape is admission backpressure, not reclamation — see the note under §6. **B11** |
| **INV-5** `no-lockstep-duplication` | Two hand-maintained cross-repo mirrors collapse: `PROTECT_GLOB` ↔ `PROTECTED_PREFIXES` (leaf **β**, with a drift test) and the reify/dark-factory warm-lane test estates (leaf **κ**). The relocation itself is the extraction |

## 7. Pre-conditions

- reify task **5572** landed (per-lane live-consumer check inside `warm-lane-gc.sh`). Hard `add_dependency`, not prose.
- No in-flight pool incident at cutover (leaf ζ) — `warm-lane-audit.sh` HEADROOM healthy, merge queue idle.

## 8. Cross-PRD / cross-repo relationship (G4)

| Counterparty | Direction | Mechanism | Owner |
|---|---|---|---|
| reify **5572** (filed, pending) | upstream | per-lane `/proc` liveness inside `warm-lane-gc.sh` | **5572.** This PRD inherits and supersedes it at γ; retains it as the recordless fallback (D7) |
| reify **5363** (pending) — audit LIVE/ASSIGNED/PINNED columns | **contested** | both read `.lane-state/<lane>.json` | **This PRD** owns the reader (leaf β ships `lib_lane_state.sh`); 5363 is rewired to consume it. Resolved here to avoid a fourth contested pair |
| DF **3027** (done) | superseded-by | `task_verify_lease` around `run_scoped_verification` | This PRD (leaf ε) generalises it to the whole assignment; 3027's two call sites collapse into one |
| reify **5354** (done) | preserved | `--lane-lock` acquire-time guard | reify. Unchanged — it is the §5 contract's locking clause |
| reify task **5326** (done) | constraint | Pass-1 always-reclaim / ENOSPC accretion fix | Preserved by B3+B4. δ **must** precede ε |
| `reify/docs/prds/warm-lane-pool-cow-seeding.md` §9.3/§9.5 | supersedes | canonical lifecycle contract + invariants | This PRD. Leaf ι rewrites §9.3/§9.5 to point at dark-factory and keeps only the §5 primitive contract in reify |
| reify `CLAUDE.md` | supersedes | "Warm lanes" invariants; Pointers row; the closing *"reify ships the primitive, dark-factory wires the invocation"* line | Leaf ι. That line is the sentence this PRD falsifies for this seam |

## 9. Decomposition plan

Phase ordering is load-bearing: **γ before ε**, **δ before ε**, **ζ after everything but before κ**.

### Phase 1 — Relocation at behaviour parity

- **α — Relocate the seven generic scripts into dark-factory with resolution preference order.**
  Modules: `orchestrator/scripts/warm-lane/`, `orchestrator/src/orchestrator/git_ops.py`.
  Ships dark-factory's copies plus resolution: project override if present, else dark-factory's own; the resolved path is logged at INFO on every invocation. reify's copies stay in place and still win, so this lands as a no-op behaviourally.
  *Signal:* the orchestrator log names the resolved script path on a reclaim pass (B7, B8). Deps: reify 5572.

- **α2 — Port the generic bash tests into dark-factory and run them against dark-factory's copies.**
  Modules: `orchestrator/tests/warm-lane/` (bash, per D4), dark-factory's test runner.
  The generic subset of reify's 23 `tests/infra/test_warm_lane_*.sh` — every test covering a relocated script — is ported as bash and wired into dark-factory's suite. reify's originals stay in place and green until κ; this is deliberately a period of duplication, because the alternative is relocating 2,200 lines of policy into a repo with no coverage of it until Phase 4.
  *Signal:* dark-factory's suite runs the ported warm-lane tests green against dark-factory's own script copies, with reify's copies untouched. Deps: α.

- **β — `lib_lane_state.sh`: dark-factory-authoritative data, readable from bash (INV-5).**
  Modules: `orchestrator/scripts/warm-lane/lib_lane_state.sh`, `orchestrator/src/orchestrator/git_ops.py`.
  One sourceable helper covering both facts bash currently guesses at or hand-copies:
  1. **Lane state** — given a lane dir, return `state`/`task_id` from `<worktree_base>/.lane-state/<lane>.json`, or `unknown` when no record exists. Fail-open to `unknown` on malformed JSON.
  2. **Protected prefixes** — render the protect list *from* dark-factory's `PROTECTED_PREFIXES` rather than hand-mirroring it. Today `warm-lane-gc.sh:318`'s `PROTECT_GLOB` default is a hand-maintained copy whose own comment admits the coupling ("mirrors dark-factory's PROTECTED_PREFIXES … this list only ever grows, never narrows") — a textbook INV-5 lockstep duplication, and one of the two hand-maintained cross-repo mirrors this PRD exists to collapse.
  *Signal:* against the live pool, emits `assigned 5551` for `_lane-28` and `unknown` for a recordless `_iact-*` dir; and a drift test fails if `PROTECTED_PREFIXES` gains a prefix the rendered glob does not. Deps: α.

### Phase 2 — Liveness swap (closes esc-5334-6 structurally)

- **γ — Reclaim consults the durable record per lane, immediately before reset, under the flock.**
  Modules: `orchestrator/scripts/warm-lane/warm-lane-gc.sh`.
  Replaces `FREE ≈ flock-free` with a record read. `assigned`/`in_use` → preserve with the task id in the reason. No record → fall back to 5572's `/proc` predicate. Deletes the header's "state is NOT consulted" rule.
  *Signal:* B1 + B2 + B6 green; B3 still green. Deps: β.

### Phase 3 — Make the assigned/idle distinction real

- **δ — Populate `LaneState.IN_USE` (the declared-only writer gap).**
  Modules: `orchestrator/src/orchestrator/{git_ops,workflow,lane_lifecycle}.py`.
  Transition `ASSIGNED → IN_USE` when dark-factory has a live child in the lane (agent invocation or verify), back to `ASSIGNED` when it exits. G3 found **zero writers** today; this leaf is the substrate the rest of Phase 3 assumes.
  *Signal:* during a live agent invocation, `.lane-state/<lane>.json` reads `in_use`; between phases it reads `assigned`. Deps: γ.

- **ε — Whole-assignment lease replaces the two `task_verify_lease` call sites.**
  Modules: `orchestrator/src/orchestrator/{git_ops,workflow}.py`.
  Shared lease held acquire→release. Safe only because γ+δ let reclaim distinguish idle from building, so B4 keeps working.
  *Signal:* B5 + B10 green, **and B4 still green in the same run** — the regression guard against the 2026-07-10 ENOSPC outage. Deps: δ.

### Phase 4 — Cutover (irreversible; ordered last)

- **ζ — Cutover readiness check.** Verify dark-factory's copies are the ones running (α's INFO line), pool healthy, queue idle. *Signal:* a recorded go/no-go with the log evidence. Deps: ε.
- **η — Repoint the systemd unit.** `reify/deploy/systemd/reify-warm-lane-gc.service:28` ExecStart → dark-factory's sweep; installer sed updated. *Signal:* `systemctl --user status reify-warm-lane-gc` shows the dark-factory path; a sweep completes. **reify repo.** Deps: ζ.
- **θ — Narrow reify to the contract.** Generalise `refresh-warm-base.sh`'s RUSTFLAGS stamp to `--build-fingerprint`; document `seed-warm-lane.sh` against §5. *Signal:* B9 green against reify's real script. **reify repo.** Deps: ζ.
- **ι — Docs truth.** Rewrite `CLAUDE.md` "Warm lanes" + Pointers row + the closing cross-repo-seam sentence; rewrite `cow-seeding.md` §9.3/§9.5 to point at dark-factory. *Signal:* no doc claims reclaim-time one-consumer enforcement that RC-1 falsified. **reify repo.** Deps: ζ.
- **κ — Delete reify's copies and retire the duplicated tests.** Remove the seven relocated scripts and the reify-side tests α2 already ported (ending the deliberate duplication window); drop the removed paths from `scripts/verify-pipeline-paths.txt`. *Signal:* reify's `scripts/` holds exactly `seed-warm-lane.sh` + `refresh-warm-base.sh`; dark-factory's ported suite is green; a pool-policy change no longer escalates reify's gate. **reify repo.** Deps: α2, η, θ, ι.

**Migration landmine, encoded as ordering.** `_run_warm_lane_gc_reclaim` fail-softs to `rc=127` with `logger.debug` — not even a warning — when the script is absent. Deleting reify's copies before dark-factory's are proven live would silently stop GC and accrete the pool to ENOSPC. κ therefore depends on ζ transitively through η/θ/ι, and B8 makes the silent path loud as a leaf-α deliverable.

## 10. Out of scope

- Making the pool multi-project. reify is the only consumer (25 config hits vs 0 across six other dark-factory-managed projects); the contract is written so a second project is possible, but no second project is built or validated here.
- Rewriting relocated bash as Python (D4).
- `provision-warm-lane-fs.sh`'s host/loopback/XFS provisioning semantics — relocated verbatim, not redesigned.
- reify 5572's stop-gap. Lands independently, first.

## 11. Open questions (tactical)

1. **How does dark-factory ship and resolve the bash scripts?** Package data via `importlib.resources`, an installed-path convention, or repo-relative resolution from `orchestrator/scripts/warm-lane/`. Repo-relative is simplest and matches how dark-factory is deployed today (`uv run --project orchestrator` from a checkout). *Decide during α.*
2. **Does `IN_USE` track the agent invocation, the verify, or any live descendant?** "Any live dark-factory child rooted in the lane" is the most conservative and is a superset of both. *Decide during δ.*
3. **Does the `/proc` fallback stay after δ?** It covers recordless lanes (`_iact-*`, manual worktrees) which will still exist. Suggested: keep. *Decide during γ.*
4. **Do reify's ported bash tests run in dark-factory's default suite or a warm-lane-only lane?** ~2,200 lines of hermetic tests may be slow enough to want their own bucket. *Decide during κ.*
