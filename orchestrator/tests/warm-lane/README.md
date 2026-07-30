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
(`ASSERT_FLOORS`). Exit status alone cannot tell "all 837 asserts ran and
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

Measured on an unloaded host, each `.sh` run directly and sequentially:

| Ported test | asserts | wall-clock |
|---|---|---|
| `test_thin_warm_lane.sh` | 45 | 0.46s |
| `test_warm_lane_disk_guard.sh` | 62 | 1.04s |
| `test_warm_lane_sizing_lifecycle.sh` | 65 | 1.07s |
| `test_warm_lane_degenerate_ref.sh` | 70 | 1.23s |
| `test_provision_warm_lane_fs.sh` | 111 | 1.49s |
| `test_warm_lane_gc_sweep.sh` | 86 | 6.73s |
| `test_warm_lane_audit.sh` | 228 | 12.93s |
| `test_warm_lane_gc.sh` | 198 | 16.95s (stale — see below) |
| **total** | **865** | **≈42s (stale)** |

`test_warm_lane_gc.sh` dominates: 34 `git worktree add` calls, 33 `flock`
acquisitions and `/proc/<pid>/{exe,cwd,fd,maps}` liveness walks. Under
concurrent load these figures roughly double; the full driver module
(12 pytest items, all co-located on one xdist worker) measured 47s.

**The `test_warm_lane_gc.sh` wall-clock is OWED a re-measurement (task 3075).**
Its assert count moved 170 → 198 when leaf γ added Block S and A10 (four new
`run_helper reclaim` invocations, shaped to keep at most two `/proc`-walking
lanes each). The only measurement taken on 2026-07-30 was **79.3s at loadavg
109.67** — a heavily loaded host, not comparable to this table's unloaded
baseline, so it is recorded here as an observation and deliberately NOT
promoted into the table. Writing it in as the new unloaded figure, or writing
in an estimate dressed as a measurement, would put a false number on main —
which is what this table exists to prevent, since its stated purpose is that
the *measured* cost is what justified answering PRD §11 q4 with "the default
orchestrator suite". Re-measure on an unloaded host and replace both the row
and the total.

For reference, the whole orchestrator suite with these included is ~253s for
~13,100 tests.

## PRD §11 q4 — decided: the default suite

**The ported tests run in dark-factory's DEFAULT `orchestrator` suite**, not in
a separate opt-in bucket. No test-runner config change was needed:
`orchestrator/orchestrator.yaml`'s `test_command` is
`pytest tests/ ... --timeout=300`, a directory argument that already recurses.

A `warm_lane_bash` marker **is** registered in `orchestrator/pyproject.toml`,
for on-demand selection/deselection (`-m warm_lane_bash`,
`-m 'not warm_lane_bash'`). It is deliberately **not** added to `addopts` as
`-m 'not warm_lane_bash'`.

Rationale: PRD leaf **γ** changes `warm-lane-gc.sh`'s core reclaimability
predicate and needs this coverage *actually running* on its verify leg, and leaf
**κ** deletes reify's originals on the strength of this suite being green. A
marker-deselected bucket (the `shared` / `fused-memory` / `cockpit`
`-m 'not integration'` precedent) would give the appearance of coverage without
the fact of it. The measured cost above is bounded and, in the table, revisable
on evidence rather than guesswork.

**Documented escape, not taken now:** if the recorded wall-clock later proves
prohibitive, the sanctioned route is `git.offline_lane_commands` in
`dark-factory-orchestrator.yaml`, which runs a marker-selected bucket
post-merge, off the verify hot path.

Each item carries `@pytest.mark.timeout(360)` with an inner subprocess timeout
of 300s. The marker is required because `orchestrator/pyproject.toml` sets
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
