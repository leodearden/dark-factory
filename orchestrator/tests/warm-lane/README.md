# dark-factory's warm-lane bash tests

These are dark-factory's **own** copies of the project-agnostic warm-lane bash
tests, running against dark-factory's **own** script copies in
`orchestrator/scripts/warm-lane/` (relocated by task 3072, leaf α).

Ported by **task 3073**, PRD `plans/warm-lane-infra-repatriation-prd.md`
leaf α2 (Phase 1).

`.sh` files are not collected by pytest, so each of these is driven as one
parametrized pytest item by `orchestrator/tests/test_warm_lane_bash_suite.py`.
That driver also carries the two non-vacuity guards — the `PORTED_TESTS`
manifest and the `SCRIPT_COVERAGE` map, each asserted set-equal to what is
actually on disk — which keep a dropped, renamed or mis-globbed port from
reporting a vacuous green. It fails (never skips) when a required host tool is
absent, for the same reason.

## Provenance

Source repo: **reify** (`/home/leo/src/reify`), path `tests/infra/<name>`.
Copied at reify HEAD `8489b49bfaefddd4abbe875a970661220dacbd57`
(2026-07-30; `tests/infra/` clean at that sha). Per-file last-touching commit
at that HEAD:

| File | lines | reify commit | date |
|---|---|---|---|
| `test_helpers.sh` | 459 | `2ac7b723b7` | 2026-07-28 |
| `test_warm_lane_disk_guard.sh` | 432 | `3662006952` | 2026-07-11 |
| `test_thin_warm_lane.sh` | 440 | `2ac7b723b7` | 2026-07-28 |
| `test_warm_lane_degenerate_ref.sh` | 550 | `5b8a44ad6e` | 2026-07-05 |
| `test_warm_lane_sizing_lifecycle.sh` | 672 | `62c0f188c5` | 2026-07-26 |
| `test_provision_warm_lane_fs.sh` | 1140 | `b37e00eaa6` | 2026-07-11 |
| `test_warm_lane_gc_sweep.sh` | 1230 | `973fde7955` | 2026-07-28 |
| `test_warm_lane_gc.sh` | 1939 | `973fde7955` | 2026-07-28 |
| `test_warm_lane_audit.sh` | 1999 | `973fde7955` | 2026-07-28 |

`test_warm_lane_gc.sh` has since grown DARK-FACTORY-NATIVE coverage that has no
reify counterpart, added by **task 3075** (PRD leaf γ) — the line count and SHA
above still describe the ported baseline, not the current file:

- **Block S** (23 asserts) — the durable lane record is Pass 1's reclaimability
  gate. `S-basic` pins that an `assigned` lane survives with a FREE flock and no
  `--extra-protect-glob` (the dark-factory ε shape); `S-reasons` pins the reason
  vocabulary, the gate ORDER by attribution, and the appended
  `preserved_assigned=` counter; `S-fallback` pins that the task-5572 `/proc`
  scan is inherited as the recordless fallback and that
  released/quarantined/corrupt all fall open; `S-toctou` pins PLACEMENT — a lane
  assigned mid-pass is still preserved, so the read cannot be hoisted into an
  up-front snapshot without going red.
- **A10** (3 asserts) — the fail-loud guard for the new sibling
  `lib_lane_state.sh`, with `lib_live_refs.sh` deliberately PRESENT in the
  fixture so A9's guard cannot account for the result.

## Why these eight

The port set was confirmed against **what α actually relocated**, not against a
candidate list. Every assertion in these eight exercises a script that now
lives in `orchestrator/scripts/warm-lane/`.

Two plausible-looking candidates are **deliberately excluded, permanently** —
not just until leaf κ. Leaf κ's deletion set must NOT include them:

- **`test_warm_lane_preflight.sh`** — its subject is `warm-lane-preflight.sh`,
  which α did not relocate (absent from `orchestrator/scripts/warm-lane/` and
  absent from PRD §2.1's disposition table). The test also drives that script
  directly across all its blocks and is deeply toolchain-coupled (11
  `RUSTFLAGS="-C target-cpu=native"` invocations, `rustc` binary fixtures,
  assertions that stderr names `refresh-warm-base.sh`). It belongs with the
  primitive that stays in reify under the PRD §5 contract.
- **`test_lane_x_flock.sh`** — its subject is `lib_lane_x_flock.sh`, reify's
  TEST-SLOT flock machinery, not warm-lane pool policy. It transitively sources
  `lib_slot_acquire.sh` and `lib_clock_stop.sh` (none relocated), needs
  `setsid`, GNU `timeout` and nanosecond `date`, and carries a hard ~4s
  wall-clock floor with a real timing assertion.

Porting either would mean relocating scripts PRD §5 deliberately leaves in
reify. The seed/refresh/warm-base test family is excluded for the same reason.

## Documented deltas from the reify sources

Every file here is **byte-identical** to its reify source except as recorded
below. reify's originals stay in place and green until leaf κ, so diffability
against reify is the only cheap drift check available for the whole α→κ window
— deltas are enumerated rather than absorbed. This mirrors the discipline α
adopted for the scripts themselves in `orchestrator/scripts/warm-lane/README.md`.

Policy is untouched: **`REIFY_*` environment-variable names are verbatim**,
because the shipped scripts still read those exact names (α kept them;
renaming is downstream work — leaves β/γ/δ/ε). Renaming them in the tests would
make the tests disagree with the scripts under test.

### Delta 1 — script-under-test path, via `lib_warm_lane_paths.sh` (all eight)

All eight reify sources share one bootstrap idiom:

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/<name>.sh"
```

Here the tests sit one level deeper (`orchestrator/tests/warm-lane/`) and the
scripts sit at `orchestrator/scripts/warm-lane/`. The derivation is hoisted into
one sourced `lib_warm_lane_paths.sh`, so the new depth is **one** edit point
instead of eight independently-driftable ones, and each ported file's diff
against reify stays a two-line delta (`source` + `SCRIPT=`).

reify's `REPO_ROOT` binding is deliberately **not** carried over into the eight
ported files. The lib still derives `DF_REPO_ROOT` — it is where a future
consumer picks it up — but after Deltas 3–6 removed every reify-repo structural
assertion, no ported file references a repo-root-relative path at all, so an
alias in each file would be a dead variable reading as if such paths were still
in play. That is exactly the class of thing this enumerated-delta discipline
exists to keep visible rather than let accumulate.

Resolution is **pure path arithmetic** from `${BASH_SOURCE[0]}`: it reads no
environment and invokes no `git rev-parse`. Same discipline, and the same three
failure modes, as α's Delta 1 — an inherited `GIT_DIR` (the standard git-hook /
`git rebase --exec` environment) returns the script's own directory, a
rev-parse probe can ascend into an enclosing repo, and it returns the
symlink-resolved path where `..` returns the logical one (dark-factory's own
`.worktrees` is a case where those disagree). All three land in the same
wrong-path class, and an existence guard catches none of them because every
wrong path exists.

In particular the resolver does **not** honour `ORCH_WARM_LANE_SCRIPT_DIR`.
`orchestrator/tests/conftest.py`'s autouse `_isolate_warm_lane_script_dir`
pins that variable at a guaranteed-absent sentinel for *every* test in the
orchestrator suite, and it leaks into the subprocesses the driver spawns; if the
harness honoured it, all eight ported tests would silently run against an empty
directory. That is pinned behaviourally — not by a source grep — by
`test_warm_lane_bash_suite.py::test_script_dir_resolution_ignores_the_env_override`.
The resolver `exit 2`s, naming the resolved path, if the directory is missing,
so a mislaid harness cannot degrade to a vacuous pass.

### Delta 2 — `test_helpers.sh` source guard and mktemp stem

`_REIFY_TEST_HELPERS_SH_SOURCED` → `_DF_TEST_HELPERS_SH_SOURCED`, and the
`reify-assert.XXXXXX` mktemp stem → `df-assert.XXXXXX`. Cosmetic renames of
dark-factory-local identifiers; the file header is likewise rewritten to name
its new consumers.

`_SHARED_TRASH_DIR=/tmp/.reseed-trash` is **left alone**: it is the real
machine-shared path the `assert_no_shared_trash_litter` /
`assert_shared_trash_litter_detector_live` guards exist to detect, not a
repo-relative name.

The file is otherwise ported whole, including the lane-isolation facility
(`init_isolated_lane_root`, `make_isolated_lane`), which
`test_thin_warm_lane.sh` and `test_warm_lane_gc.sh` both call. Its deliberate
omission of an EXIT trap is preserved: bash EXIT traps do not stack, so a
library-level trap would clobber each suite's own `trap cleanup EXIT`.
`test_summary`'s exit-0/exit-1 convention is the entire pass/fail protocol the
pytest driver consumes.

### Delta 3 — `test_warm_lane_gc_sweep.sh`: Block D excised (8 asserts)

`$REPO_ROOT/deploy/systemd/reify-warm-lane-gc.{timer,service}` structural
assertions. dark-factory has no `deploy/systemd/`; PRD leaf **η** owns
repointing that unit at cutover, and until then the unit correctly still names
reify's copy.

### Delta 4 — `test_warm_lane_gc_sweep.sh`: assertion V4 excised

`$REPO_ROOT/dark-factory-orchestrator.yaml` contains `warm_lane_pool: true`.
In reify, `REPO_ROOT` is reify's checkout, whose orchestrator config enables the
pool for the reify **project**. Here `REPO_ROOT` is dark-factory's own checkout,
whose `dark-factory-orchestrator.yaml` contains no `warm_lane` key at all —
dark-factory does not run its own tasks in warm lanes. Ported verbatim this is a
guaranteed-red assertion that no in-scope change could GREEN.

Excised, **not** "fixed" by adding `warm_lane_pool: true` to dark-factory's
config: that would change operational config to satisfy a test.

### Delta 5 — `test_warm_lane_gc_sweep.sh`: Block F excised (2 asserts)

`$REPO_ROOT/scripts/verify-pipeline-infra-tests.txt` drift-map rows. That file
is reify's verify-gate script→test map and has no dark-factory counterpart. The
drift guard it provides is replaced by
`test_warm_lane_bash_suite.py::test_every_invocable_script_has_ported_coverage`,
which asserts the same invariant against the directory that actually exists here.

Deltas 3–5 also remove the now-dead `GC_TIMER` / `GC_SERVICE` / `VP_INFRA_MAP`
path bindings and the corresponding D/F/V4 lines in the block-list docstring, so
the header does not advertise coverage this copy does not have.

Everything else in that file is **kept unchanged**: Blocks A (CLI guard),
B (fail-open on a nonexistent `--mount`), C (happy path), V1–V3 (disk-guard
`--help` flags), W (cross-script `--mount` seam — its dark-factory /
`git_ops.py` references are comments only), G (emergency low-water trigger),
T (stale `.reseed-trash` reaper) and U (live-consumer lane guard, reify 5572).

### Delta 6 — `test_provision_warm_lane_fs.sh`: Block F excised (3 asserts)

Three structural greps of the *text* of reify's `scripts/setup-dev.sh` (that it
references the provisioner, gates on `REIFY_PROVISION_WARM_LANES`, and
warns-and-continues rather than `exit 1`). `setup-dev.sh` is never executed by
the test and has no dark-factory counterpart — it is reify's developer-setup
installer, which PRD §5/§10 leave in reify. Its `SETUP_DEV` binding and
block-list docstring entry go with it.

All 15 remaining blocks are kept, including Block I's
`command -v mkfs.xfs && xfs_info && xfs_db` skip guard, verbatim.

### Not a delta — α's provision Deltas 1–2 needed no test-side change

α's relocation of `provision-warm-lane-fs.sh` changed its `REPO_ROOT` to a
three-level ascent and made `_default_mount()` match both the `worktrees` and
`.worktrees` spellings. The only usage-text default assertions in the test are
Block J's `J1`/`J2`, which pin the `--img` default
(`/media/leo/data_lv_1/leo/reify-warm-lanes.img`) and the 4096 GiB size — α kept
both verbatim and both pass unchanged. **No assertion pins the advertised
default `--mount`**, so nothing needed re-deriving for the new depth;
`test_warm_lane_scripts_shipped.py::TestProvisionRepoRootParity` remains the
sole pin on that behaviour.

## Greenness predicate — exit 0 is not enough

The driver's `test_bash_suite_passes` applies the same predicate reify's own
runner (`tests/infra/run_all_ambient_isolation_lib.sh`) does: **exit 0 AND a
`Results: N passed, 0 failed` line AND `N` at or above a per-suite floor**
(`ASSERT_FLOORS`). Exit status alone cannot tell "all 865 asserts ran and
passed" from "a block skipped and the rest passed", and leaf **κ** deletes
reify's originals on the strength of these items being green — so the weaker
predicate would have been the single weak link in that chain.

Only two shortfalls against the measured counts below are legitimate, both from
skip guards this leaf was required to preserve verbatim, so both are subtracted
in the floor rather than left to pass silently:

| Suite | measured | floor | conditional block |
|---|---|---|---|
| `test_provision_warm_lane_fs.sh` | 111 | 106 | Block I (5 asserts) — real-geometry `xfs_info`/`xfs_db` proof, guarded on `mkfs.xfs`/`xfs_info`/`xfs_db`. Measured both ways: 111 with xfsprogs, 106 without. |
| `test_warm_lane_audit.sh` | 228 | 225 | L9 (3 asserts) — unreadable-record degradation, guarded on `id -u != 0` because mode 000 is not a barrier for root. |

xfsprogs is deliberately **not** promoted into the driver's
`REQUIRED_HOST_TOOLS`: Block I's guard exists so the bash file stays runnable by
a developer on any host, and hard-requiring an optional package would red the
whole orchestrator suite on a host that lacks only that. Every other suite's
floor equals its measured count. `test_thin_warm_lane.sh`'s Block C df-delta
assert is not a deduction — it is gated on `REIFY_WARM_LANE_MOUNT`, which the
driver always strips, so 45 is both its floor and its measured count there.

The driver also strips **every `REIFY_*` key** from the subprocess environment
(a prefix rule, not a name list, so it cannot drift as leaves β/γ/δ/ε add
seams). This host also develops reify, and the exposure was measured, not
assumed: with `REIFY_WARM_LANE_AUDIT_SAFETY=off` leaked in, `test_warm_lane_audit.sh`
reports 35 passed / 193 failed; with `REIFY_WARM_LANE_GC_PROTECT_GLOB='_lane-*'`,
`test_warm_lane_gc.sh` reports 101 / 69. The strip is safe because every ported
suite exports the `REIFY_*` vars it needs per invocation and reads none from the
ambient environment.

## Measured wall-clock

Measured sequentially on **2026-08-05**, each `.sh` run directly, in ONE sitting
of 82s (10:47:08→10:48:30 BST) at **loadavg 12.98–15.33 on a 32-core host**.
The band is a 1.18x spread, so these eight figures are comparable *to each
other* — the property this table needs, and the one it previously lacked.

**On the word "unloaded", which this caption used to claim.** No unloaded
reading of this host is obtainable: it is its own load source. Median 1-min
loadavg on the four uncapped days before the cap began (Jul 31 – Aug 3) was
104, 121, 121, 120 (`/var/log/sysstat`, 10-min samples), and fewer than 1.5%
of those samples fell below 20 — on two of the days, none did. The sitting
above was only possible because all four fleet accounts were simultaneously
usage-capped, which is the sole condition under which this box goes quiet.
Every figure here is therefore load-qualified, not idle-baseline — and so is
the 2026-07-30 column, taken during a comparable brief quiet window on a day
whose median was 92. Do not expect to reproduce either without a cap.

| Ported test | asserts | wall-clock | 2026-07-30 | ratio |
|---|---|---|---|---|
| `test_thin_warm_lane.sh` | 45 | 0.57s | 0.46s | 1.24x |
| `test_warm_lane_disk_guard.sh` | 62 | 1.04s | 1.04s | 1.00x |
| `test_warm_lane_sizing_lifecycle.sh` | 65 | 1.38s | 1.07s | 1.29x |
| `test_warm_lane_degenerate_ref.sh` | 70 | 1.73s | 1.23s | 1.41x |
| `test_provision_warm_lane_fs.sh` | 111 | 1.89s | 1.49s | 1.27x |
| `test_warm_lane_gc_sweep.sh` | 86 | 15.01s | 6.73s | **2.23x** |
| `test_warm_lane_audit.sh` | 228 | 14.34s | 12.93s | 1.11x |
| `test_warm_lane_gc.sh` | 214 | 46.37s | 16.95s (at 198 asserts) | **2.74x** |
| **total** | **881** | **82.33s** | **≈42s** | **1.96x** |

`test_warm_lane_gc.sh` dominates: 34 `git worktree add` calls, 33 `flock`
acquisitions and `/proc/<pid>/{exe,cwd,fd,maps}` liveness walks.

**Read the two bold ratios as a cross-DAY comparison, not as a regression.**
The `ratio` column divides a 2026-08-05 figure by a 2026-07-30 one, so it
brackets five days of host state, not a commit range. For `gc_sweep` a direct
same-sitting A/B between the commits puts the ratio at 0.93x — see the next
subsection.

### `gc_sweep` did NOT regress: the 2.23x was a confounded comparison (task 3655)

**This subsection previously claimed the two gc-driving suites had regressed
~2x and that it was "attributable to task 3292". For `gc_sweep` that claim is
RETRACTED — it does not survive a direct A/B against the pre-3292 tree.** What
follows is what task 3655 established. It is the second conclusion this section
has had to overturn, which is the argument for keeping it: it records its own
refutations.

Six of the eight rows reproduce their 2026-07-30 figures within 1.00–1.41x at
comparable load — `test_warm_lane_disk_guard.sh` lands on 1.04s both times.
The two that drive `warm-lane-gc.sh` did not: gc-sweep was 2.23x its baseline
at an UNCHANGED 86 assertions, and gc.sh 2.74x on an 8% assert growth
(198→214). Three confounds were checked and eliminated at the time:

- **Cold FS cache** — ruled out. Warm-cache re-runs minutes later reproduced:
  gc.sh 46.01s, gc-sweep 13.89s, audit 13.85s.
- **The registry-render bridge** (task 3292, below) — ruled out. `strace -f -e
  trace=execve` over a whole gc-sweep run counts **exactly one**
  `.venv/bin/python3` exec, and one render measured 0.41–0.48s over 5 samples
  at this load. The mitigation described below is working; it accounts for
  ~0.45s of an 8.3s gain.
- **Assertion growth** — ruled out for gc-sweep, whose assert count did not move.

The confound that was NOT checked is the one that mattered: **whether the
regression reproduces at all when the only thing that changes is the commit.**

#### The direct A/B: it does not reproduce

Both endpoints were exported with `git archive` into `/tmp` (so the two arms
differ ONLY by commit — same filesystem, same mount, same `.venv`, which is
symlinked into each and proven working by `lane_protect_glob _lane- _spec-`
returning the glob at rc=0 with no `[warn]` before any figure was taken). Runs
alternate A,B,A,B; every figure carries the loadavg sampled immediately before
and after it.

| `gc_sweep`, 2026-08-05, alternating A/B ×3 | median | runs |
|---|---|---|
| A = pre-3292 (`ee7571e253`) | **30.86s** | 86 passed, 0 failed ×3 |
| B = post-3292 (`be86ccc9e5`) | **28.58s** | 86 passed, 0 failed ×3 |
| **ratio B/A** | **0.93x** | B ≤ A in all three pairs |

Loadavg band across the sitting 88.47–129.73 (1.46x spread) — VALID under this
section's own rule. **A first sitting was discarded, not salvaged**: its band
was 110.82–225.37 (2.03x), and its ratio came out 1.22x with the sign FLIPPED
in the third pair (A slower than B). That discard is the protocol working.

Post-3292 is, if anything, marginally *faster*. There is no 2.23x to explain.

#### The counters: both arms do the same work

Wall-clock on this host is nearly unusable (median 1-min loadavg 92–121; under
1.5% of samples below 20), so the argument is carried by **load-independent
counters**, which are exact at loadavg 15 and at loadavg 200 alike. `strace -f
-e trace=execve` over one whole suite run, per arm:

| counter | pre-3292 | post-3292 | delta |
|---|---|---|---|
| total `execve` | 1133 | 1132 | **−1** |
| `warm-lane-gc.sh` invocations | 4 | 4 | **0** |
| `.venv/bin/python3` (renders) | 0 | 1 | **+1** |
| `/proc/<pid>/fd` liveness walks | 21 | 21 | **0** |
| `flock` acquisitions | 103 | 103 | **0** |
| `sleep` | 13 | 11 | −2 (retry timing) |

Every other binary count is identical. The `/proc` liveness walk — the cost
centre worth suspecting first, since gc.sh's own comment prices it at a
MEASURED ~1.9s per lane (task 5572) — **did not move**. Post-3292 does one
extra render and, net, one FEWER `execve` than pre-3292.

#### The mechanism, named and classified

3292's entire measurable per-invocation cost in this suite is **one
`.venv/bin/python3` PROTECT_GLOB render per suite run**, and it is paid by
exactly one call site: the direct `bash "$GC_REAL" reclaim --mount "$WB"` in
Block W of `test_warm_lane_gc_sweep.sh`, which bypasses `run_sweep` and so does
not carry the `REIFY_WARM_LANE_GC_PROTECT_GLOB` pin. The four `run_sweep`-driven
real-gc cases are correctly short-circuited — that is why the counter reads 1
and not 5. That one render cost `si_utime` 0.90s + `si_stime` 0.16s of CPU in
this sitting.

**Classification: INHERENT.** It is the price of rendering `PROTECT_GLOB` from
the registry instead of the hand-copied `PROTECTED_PREFIXES` literal that 3292
existed to delete (INV-5), and Block W is the only place this suite exercises
gc.sh's real production default. It is not contorted away. `run_sweep`'s pin
already removes it everywhere it can be removed without losing coverage, and
that pin is now itself pinned — see Block Y below.

#### The refuted hypothesis

This section used to propose that `a62db712d8` "added per-invocation cost to
`warm-lane-gc.sh` itself — e.g. sourcing `lib_lane_state.sh`". **It did not.**
The refutation is one command, re-runnable rather than trusted:

```bash
git show ee7571e253:orchestrator/scripts/warm-lane/warm-lane-gc.sh | grep -n lib_lane_state
```

`source "$SCRIPT_DIR/lib_lane_state.sh"` is already at line 313 PRE-3292 (it is
line 319 today). 3292 added no source line and no fork to gc.sh. The lib's
source-time work is `cd`/`pwd` BUILTINS in subshells and is unchanged across the
window. Filtering 3292's whole gc.sh diff to non-comment lines leaves only the
3-line `PROTECT_GLOB` default plus a `usage()` heredoc edit reachable only via
`--help`; the `lib_lane_state.sh` and `warm-lane-gc-sweep.sh` changes in the
window are **comment-only**. And the protect-set delta is behaviourally inert:
it adds `_merge-verify` (already covered by `_merge-*`) plus `.lane-state` and
`.task-meta`, two dot patterns that gc.sh's `for entry in "$WORKTREES_DIR"/*/`
enumeration can never match, and `_matches_glob` is pure builtins, so 7→10
patterns costs zero forks.

#### The attribution, corrected

The old claim was "two independent measurements … both bracket the same
change". Only one of them brackets anything:

| `gc_sweep` | loadavg | pre-3292 | post-3292 | ratio | what it actually brackets |
|---|---|---|---|---|---|
| 2026-08-01 pair | ~200 | 59.5s | 124.0s | 2.08x | 3292's commits — but taken at the load this README itself calls untrustworthy |
| 2026-07-30 → 2026-08-05 | ~14 | 6.73s | 15.01s | 2.23x | **five days of host state**, not a commit range |
| **2026-08-05 direct A/B** | **89–130** | **30.86s** | **28.58s** | **0.93x** | **3292's commits, and nothing else** |

Two figures agreeing on a ratio is weaker evidence than it looks when only one
of them is a controlled comparison. In fairness to the old claim, the code-side
confound really is excluded — the only commits touching warm-lane paths in that
five-day window are 3292's own plus one README-only commit
(`git log ee7571e253..HEAD -- orchestrator/scripts/warm-lane/
orchestrator/tests/warm-lane/`). What is not excluded is everything else on
this host across those five days: repo object growth, `/tmp` state, mount
contents, `.venv`, kernel. The direct A/B holds all of that fixed, and the
ratio goes to 0.93x.

**A note on the SHAs this section used to quote.** `464b085e7a, a62db712d8,
2e1927c6bf, 04b2dd82d5` are pre-rebase task-branch SHAs and are **not in
`main`** (`git merge-base --is-ancestor` fails for all four). On `main`, task
3292 is **five** commits — `e29131c457, 3156952dc0, b441fb296d, 61fde6548e,
b4c6759c51` — off parent `ee7571e253`, whose warm-lane tree is byte-identical
to `e1c04cf316`'s. Quote the reachable ones.

#### What was NOT established

- **`test_warm_lane_gc.sh`'s 2.74x was not re-tested.** Only `gc_sweep` was
  A/B'd. Part of that row is known assert growth (198→214); the remainder is
  unexplained and stays unexplained here rather than being assumed to share
  gc-sweep's verdict.
- **No four-point bisect was run.** The endpoint pair does not reproduce, so
  there is no jump to localise, and forcing a guilty commit out of a flat
  bisect would manufacture a finding.
- **A latent hazard, named but not gated here.** Both gc suites now drive gc.sh
  through a pin set to `$LANE_PROTECT_GLOB_FALLBACK`, which is behaviour-neutral
  only because the fallback currently equals the rendered set byte-for-byte
  (re-verified 2026-08-05). `TestProtectGlobFallbackDrift` enforces only
  fallback ⊇ rendered. The day a band is added to `PROTECTED_PREFIXES` and the
  fallback becomes a strict superset, both suites silently begin testing gc.sh
  with a protect set production never uses, and nothing fails. Closing that
  means changing the drift gate, which is outside task 3655's file set.

#### Block Y: the seam is now pinned

`run_sweep`'s header conceded that "NO assert in this file inspects the protect
glob" — so the pin this whole investigation turns on was verified by nothing,
and a silently-broken pin would have shown up only as the kind of wall-clock
drift that started this. Block Y (task 3655) closes it, in counts and
set-membership rather than seconds: `REIFY_WARM_LANE_IACT_PREFIX=_sentinelband-`
rides alongside the existing pin, and since it can move the RENDERED band but
cannot touch the static `$LANE_PROTECT_GLOB_FALLBACK`, `_iact-` staying
protected while `_sentinelband-` does not is proof the render was
short-circuited. Removing the pin flips those two asserts to red. Floor
86 → 90.

**α2 predicted these figures would "roughly double" under concurrent load. That
prediction is wrong, and the correction is measured, not argued.** This machine
runs its own concurrent lane fleet at **loadavg ~120 on 32 cores**, and
re-measured there on 2026-07-30 at **loadavg 124**, with the two largest suites
run concurrently against each other, the dilation is **~10-15x**, not 2x:

| Ported test | quiet window (07-30) | at loadavg 124 | dilation |
|---|---|---|---|
| `test_warm_lane_audit.sh` | 12.93s | 190.2s (228 passed, 0 failed) | ~14.7x |
| `test_warm_lane_gc.sh` | 16.95s | 128.3s (198 passed, 0 failed) | ~7.6x |

Both **pass** at that load — this is a cost figure, not a failure. Per-block
elapsed for `test_warm_lane_gc.sh` at loadavg 124 puts Block S (this leaf's
addition) at **112.4s → 128.3s = 15.9s**, matching the ~+12s the plan budgeted
for its four `run_helper reclaim` invocations, so the growth is accounted for
and is not where the dilation comes from.

This is what sized the driver's timeout pair. Under the full 16-worker driver
on the same host, both suites blew the then-300s subprocess ceiling *mid-block
with no `FAIL:` line* — `gc` having reached ~67% and `audit` ~52% of their
nominal block sequence, a ~3.5x and ~3.1x further dilation extrapolating to
~450s and ~580s to completion. `SUBPROC_TIMEOUT`/`@pytest.mark.timeout` were
therefore re-calibrated 300/360 → **900/960** (task 3075 debug leg); the
arithmetic is recorded on `SUBPROC_TIMEOUT` in the driver. Note the ranking
inversion the quiet-window column does not predict: under fleet load
`test_warm_lane_audit.sh`, not `test_warm_lane_gc.sh`, is the binding suite.
(As of the 2026-08-05 sitting the quiet-window ranking has itself inverted —
gc.sh at 46.37s now exceeds audit.sh at 14.34s even quiet, per the regression
noted above.)

### The registry-render bridge: the A/B is unmeasurable, but both factors now are

Task 3292 wired `warm-lane-gc.sh`'s `PROTECT_GLOB` default to
`lib_lane_state.sh`'s `lane_protect_glob`, which starts a python3 and imports
pydantic **once per gc.sh invocation**. Both suites mitigate that, and the
mitigation is the reason no `SUBPROC_TIMEOUT` change was needed:

- `run_helper` (gc suite) and `run_sweep` (gc-sweep suite) pin
  `REIFY_WARM_LANE_GC_PROTECT_GLOB` to the shipped `$LANE_PROTECT_GLOB_FALLBACK`,
  which gc.sh's `[ -n "$PROTECT_GLOB" ] ||` default short-circuits *before* the
  bridge runs. It is gc.sh's own documented knob, not a test-only opt-out.
- The blocks that assert what the **DEFAULT** protects opt back out through
  `run_helper_live_default`: **M, N-default, O** (both sub-cases) and **X-band**.
  5 of the gc suite's 37 invocations pay a render; the gc-sweep suite's four
  real-gc sites (G4, G5, two U) pay none, and nothing there inspects the glob.
  **Both counts verified 2026-08-05 by `strace -f -e trace=execve` over whole
  suite runs**, counting `.venv/bin/python3` in executable position: gc suite
  **5**, exactly as claimed; gc-sweep **1**, which is 3292's own bridge-cost pin
  block, not a real-gc site — so "the four real-gc sites pay none" holds. The
  mitigation does what this section says it does.
- **A future block that asserts default content must use
  `run_helper_live_default`, or it silently tests the pin instead of the
  default.**

**The mitigation's wall-clock saving is still NOT stated here as a measurement,
because the A/B that would establish it could not be run on this host.** What
2026-08-05 added is that both of its *factors* are now measured rather than
asserted: the render count by strace (above), and the per-render cost at
**0.41 / 0.48 / 0.46 / 0.43 / 0.45s** over five samples at loadavg ~16 — versus
0.97–3.75s measured at loadavg 189, the same ~4x dilation everything else on
this host shows. Their product remains a derivation and is still not written
into the table. For scale: 5 renders × ~0.45s ≈ 2.3s of the gc suite's 46.37s,
so the bridge is **not** where the ~2x regression noted above comes from.

A three-run sandwich (mitigated / unmitigated / mitigated) was
run back-to-back on 2026-08-01 at commit `04b2dd82d5` specifically to isolate it,
with the unmitigated leg produced by a throwaway copy of the suite whose pin
branch was made dead. All three reported **214 passed, 0 failed**:

| leg | wall-clock | loadavg before → after |
|---|---|---|
| mitigated | 213.8s | 229.75 → 281.67 |
| **un**mitigated | **203.3s** | 281.67 → 113.75 |
| mitigated | 298.3s | 113.75 → 212.09 |

The unmitigated leg is the **fastest** of the three, which no amount of removed
work can explain — the fleet load collapsed from ~282 to ~114 during it. The two
*identical* mitigated legs differ by **84.5s**, an order of magnitude more than
the ~18 elided renders could account for at the per-render cost measured minutes
later at loadavg 189: **1.49 / 1.48 / 0.97 / 3.75 / 3.43s** over five samples.
Between-run variance here swamps the effect being measured, so any figure
derived from this A/B would be an estimate dressed as a measurement — exactly
what the rule below forbids. The elided-render **count** is exact and the
per-render **cost** is measured; their product is a derivation, not a
wall-clock, and is deliberately not written into the table as one.

The `test_warm_lane_gc_sweep.sh` pair from the same session, for the record:
**59.5s / 86 passed** at loadavg 203.61→183.87 before the change (commit
`e1c04cf316`), **124.0s / 86 passed** at loadavg 212.09→189.41 after.

**This pair was originally written off here as the same between-run variance
that spoiled the sandwich above. That reading was RETRACTED on 2026-08-05 — and
then RE-INSTATED later the same day (task 3655), which is the reading that
stands.** The retraction's argument was that these two runs are at *comparable*
load (~194 and ~201 mean), differ by 2.08x, and that a 2026-08-05 quiet-window
comparison reproduced 2.23x — so that between-run variance "does not survive a
14x change in load". The flaw is that the quiet-window figure was not a
controlled comparison: it was a 2026-07-30 number against a 2026-08-05 number,
bracketing five days of host state. A **direct A/B between the same two
commits**, alternating and same-sitting, puts the ratio at **0.93x**, and the
execve counters put the two arms at the same work (see the regression
subsection above). So this pair is NOT evidence of added work; whatever
produced its 2.08x, it was not 3292's diff. It is left on the record because
it is the measurement that misled, and knowing which measurements mislead is
the point of this section.

**The `test_warm_lane_gc.sh` re-measurement owed by task 3075 (re-attributed to
3292) is DISCHARGED — measured 2026-08-05, 46.37s at 214 asserts, in the
one-sitting table above.** The debt stood open because the old table claimed an
unloaded baseline nobody could reproduce; it is settled by re-captioning every
figure with the load it was taken under, not by finally finding an idle host.

Its assert count moved 170 → 198 when leaf γ added Block S and A10 (four new
`run_helper reclaim` invocations, shaped to keep at most two `/proc`-walking
lanes each), then **198 → 214** when task 3292 added Block X. Four loaded
measurements were taken while the debt was open: **79.3s at loadavg 109.67**,
**128.3s at loadavg 124** (the latter with `test_warm_lane_audit.sh` running
concurrently), and the 213.8s / 298.3s sandwich pair above. They remain
recorded as load-qualified observations and are still NOT promoted into the
table — their spread, 79.3s to 298.3s for a suite that grew by 16 asserts, is
why a single loaded figure cannot stand in for a comparable one. Note that the
quiet-window figure now in the table (46.37s) sits *below* all four, which is
consistent with them being dilated rather than with any of them being the
suite's true cost.

The rule that governed this debt still governs the table: never write an
estimated, interpolated or extrapolated number in as a measurement. An honest
gap beats a fabricated number.

For reference, the whole orchestrator suite with these included is ~253s for
~13,100 tests.

### The live cost signal is the offline lane, not this table

Since task 3349 moved the bucket to `git.offline_lane_commands`, the lane has
been timing it on **every merge advance** and logging the figure —
`orchestrator/src/orchestrator/offline_lane.py`, `offline-lane: warm-lane-bash
sub-run head=<sha> status=<PASS|FAIL> duration=<N>s`. That is a continuous,
free, per-commit time series, and it is a strictly better cost instrument than
any single sitting: it yields a distribution rather than one number, and each
point is joinable against `sar -q` for the load it ran under.

119 records over 2026-08-01→05 (whole 12-item bucket, so not comparable to the
per-suite rows above), split by regime:

| regime | n | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|
| fleet dispatching (Aug 2 – Aug 4 04:05) | 97 | 169.4s | 533.6s | **808.5s** | 1087.5s | 2141.2s |
| accounts capped (Aug 4 11:00 →) | 11 | 131.2s | 134.0s | **141.1s** | 176.8s | 258.3s |

Loaded median / quiet median is **5.7x**, independently corroborating the 6.2x
task 3349 measured for the same bucket. Note the loaded column spans **12.6x**
end to end, and 2.0x across its own interquartile range — which is why "a
loaded-host baseline" cannot be a single band, and why a regression is better
detected as a shift in this distribution's median (~30 samples/day) than by
comparing any one run against any one row.

**A caveat on the "47s module baseline" that PRD §11 q4 quotes.** The lane's
quiet-window runs of the same 12 items land at 131–258s, ~3–5x that figure at
comparable load. The comparison is not clean — a lane run also pays pytest
startup, collection and the `_offline-deep` worktree — so the 47s is not
asserted here to be wrong, but it should not be leaned on again without a
fresh measurement.

**11 of the 119 runs are FAIL** (9.2%), four of them in the 1685–2008s range.
That is a live signal on the lane's red path, unexamined here and not part of
this table's scope; it is flagged because a reader trending these durations
will meet it.

## PRD §11 q4 — re-decided: the offline lane (the escape is taken)

**The ported tests run POST-MERGE on `git.offline_lane_commands`, off the verify
hot path** (task 3349, 2026-08-01). Two coupled config edits, and they never
move independently:

| | edit point | value |
|---|---|---|
| (a) | `orchestrator/pyproject.toml` → `[tool.pytest.ini_options] addopts` | `-m 'not warm_lane_bash'` |
| (b) | `dark-factory-orchestrator.yaml` → `git.offline_lane_commands` | `name: warm-lane-bash`, `command: "pytest -m warm_lane_bash"`, `cwd: "orchestrator"`, `fix_task_priority: "high"` |

`fix_task_priority` is `high` (the `qdrant-integration` entry alongside it is
`medium`) because leaf **κ** deletes reify's originals on this suite's green: a
red here must not sit in a low-priority queue.

**The coupling is enforced, not merely documented.**
`orchestrator/tests/test_warm_lane_bash_bucket_placement.py` asserts the
biconditional — deselected from the hot path **iff** carried by the lane — so
landing (a) without (b), which would be silent *zero* coverage, cannot survive a
verify. A fourth test in that module takes the lane entry's `command` and `cwd`
verbatim from the loaded config and actually executes a `--collect-only`,
because a `LaneCommand` whose cwd or venv does not resolve in the
`_offline-deep` worktree would file green forever while running nothing. That
module is deliberately **not** marked `warm_lane_bash`, so it keeps running on
the hot path and cannot deselect itself along with the thing it guards.

The `warm_lane_bash` marker stays registered and on-demand selectable
(`-m warm_lane_bash`), which is exactly how the lane re-selects it: a CLI `-m`
overrides the `addopts` `-m`, last wins. This is the shipped `integration`
precedent used verbatim — `fused-memory/pyproject.toml` carries
`-m 'not integration'` while the `qdrant-integration` lane entry re-selects it —
so no new mechanism is introduced.

**Superseded history (α2, task 3073 — the decision of record for α2 through γ):**
the ported tests ran in the DEFAULT `orchestrator` suite, with the marker
registered but deliberately *not* in `addopts`. The rationale was that leaf **γ**
changes `warm-lane-gc.sh`'s core reclaimability predicate and needed this
coverage *actually running* on its verify leg, and that a marker-deselected
bucket would give the appearance of coverage without the fact of it.

**Why that no longer holds.** Half of it is *consumed*: γ **is** task 3075, and
it has landed, with its verify green and this coverage in place — a
justification whose condition has been discharged cannot keep justifying the
placement. The other half inverted on cost. The whole `warm_lane_bash` group is
serialised onto one xdist worker by design (see `xdist_group` below), so its
cost is fully additive to the verify critical path and gains nothing from
`-n auto`. Re-measured for this decision: **289.58s for the 12 items at loadavg
128.72 (1-min, 32 cores), 2026-08-01** — **6.2x** the 47s module baseline
recorded above, and consistent with the ~7.6x/~14.7x dilation table at loadavg
124. All of these are **load-qualified figures** and none may be quoted as an
unloaded one. The bucket was green at that load (12 passed, no flakes, no
timeouts): this is a cost decision, not a flakiness one.

**The lever has now been pulled, and the timeout bump was not a substitute for
it.** The 300/360 → 900/960 re-calibration bought headroom against a *measured*
fleet-load dilation; it did not make the bucket cheaper. That pair is
**retained unchanged** and still governs the bucket wherever it runs, including
on the offline lane. A further bump remains the wrong answer.

**What the relocation costs, stated plainly.** Coverage moves post-merge, so a
regression no longer blocks a merge. It surfaces instead through the lane's
existing red path — confirm re-run (the flake filter) → node-id extraction →
`compute_failing_test_set_fingerprint` → dedup'd autofiled fix task at `high` →
L0 INFO escalation → staged L2 promotion — within one advance. Coverage is
*moved*, not lost, but it is strictly weaker than a pre-merge gate, and two
obligations follow:

- PRD leaves **δ**, **δ2** and **ε** still change warm-lane behaviour and no
  longer get this coverage for free. Each must run `-m warm_lane_bash`
  explicitly on its own verify leg.
- Leaf **ζ**'s recorded go/no-go must read the offline lane's green record
  before **κ** deletes reify's originals. PRD §9 already specifies ζ's
  deliverable as "a recorded go/no-go with the log evidence", which the lane's
  record satisfies.

Each item carries `@pytest.mark.timeout(960)` with an inner subprocess timeout
of 900s (raised from 360/300 on 2026-07-30 — rationale and arithmetic on
`SUBPROC_TIMEOUT`). The two move together and never independently: the marker
must stay strictly above the subprocess timeout, and the 60s gap is the
post-kill recovery `communicate(timeout=30)` doubled. The marker is required
because `orchestrator/pyproject.toml` sets
`timeout = 60` with `timeout_method = "thread"` and `--max-worker-restart=0`: a
bare local `pytest tests/warm-lane` would otherwise get 60s, and an over-limit
item `os._exit()`s its whole xdist worker. Keeping the subprocess timeout
strictly below the pytest timeout means a hung bash test fails as one clean
pytest failure with its stdout/stderr captured.

Delivering that failure shape takes an explicit `Popen` with
`start_new_session=True` plus a `killpg` on timeout, not a plain
`subprocess.run(timeout=...)`, for two measured reasons. First,
`subprocess.run` SIGKILLs only the direct `bash`, so its `trap cleanup EXIT`
never runs and its backgrounded helpers survive — `test_warm_lane_gc.sh`
launches several `( flock -x 9 && … sleep 300 ) &` and
`( cd <lane>/target && exec sleep 300 ) &` liveness fixtures, which would keep
holding lane flocks and keeping lane trees busy for up to five minutes on the
verify host, turning one timeout into a cascade. Measured on a synthetic hang:
plain `subprocess.run(timeout=…)` left 2 orphans, the `killpg` path left 0.
Second, `subprocess.run` *raises* `TimeoutExpired` rather than returning, so the
hang path would surface as an exception with the output hanging off the exception
object instead of the documented assertion-with-captured-tail. All items share
`@pytest.mark.xdist_group('warm_lane_bash')` so `--dist loadgroup` co-locates
them on a single worker — reify classifies all of these as `pool` (serialised),
and `lib_live_refs.sh` walks `/proc`, where a concurrently-running sibling's cwd
could perturb a liveness probe.

## Duplication window

reify's originals stay in place and green until PRD leaf **κ**, which deletes
them. **This repo does not touch reify**; task 3073 was read-only with respect
to that checkout. Keeping the ported copies diffable against their reify
sources is the only cheap drift check available for the whole α→κ window, which
is why the deltas above are enumerated rather than absorbed.
