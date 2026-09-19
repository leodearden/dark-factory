# dark-factory's warm-lane scripts

These are dark-factory's **own** copies of the project-agnostic warm-lane
pool scripts. They exist so a project that does not carry warm-lane tooling
still gets GC, disk guarding, thinning and auditing of its CoW lane pool.

Relocated by **task 3072**, PRD `plans/warm-lane-infra-repatriation-prd.md`
leaf α (Phase 1).

## Resolution order

`GitOps._resolve_warm_lane_script(name)` (in
`orchestrator/src/orchestrator/git_ops.py`) resolves each script in this
order:

1. `<project_root>/scripts/<name>` — the **project override**, if it exists.
2. `orchestrator/scripts/warm-lane/<name>` — **this directory**, the
   dark-factory fallback.
3. Neither → a WARNING naming **both** tried paths, and the wrapper's
   existing fail-soft sentinel.

Project-first is PRD decision D3: a project that has invested in its own
warm-lane tooling keeps it, and dark-factory's copy is the floor rather than
the ceiling. Resolution is repo-relative (`Path(__file__).resolve().parents[2]
/ 'scripts' / 'warm-lane'`) because the wheel packages only `src/orchestrator`
and deployment is `uv run --project orchestrator` from a checkout; if
dark-factory is ever installed as a wheel, resolution fails *loudly* through
that both-paths WARNING rather than silently.

`ORCH_WARM_LANE_SCRIPT_DIR` overrides this directory. It is a **test
hermeticity seam only** — production never sets it. See the autouse
`_isolate_warm_lane_script_dir` fixture in `orchestrator/tests/conftest.py`.

## Coverage

`orchestrator/tests/test_warm_lane_scripts_shipped.py` and
`test_warm_lane_script_resolution.py` pin the *shipped* and *resolution* halves
of the contract above. `test_lane_state_lib.py` pins `lib_lane_state.sh` —
including the INV-5 drift gate that fails the build if `PROTECTED_PREFIXES`
gains a band the library's static fallback does not cover. The **behaviour** of
every invocable script here is
covered by the ported bash tests in `orchestrator/tests/warm-lane/` — see
[that directory's README](../../tests/warm-lane/README.md) for provenance, the
enumerated deltas from the reify sources, and the per-script coverage map.
Those tests run in dark-factory's default `orchestrator` suite.

## Provenance

Source repo: **reify** (`/home/leo/src/reify`), path `scripts/<name>`.
Copied at reify HEAD `638d97d8aba4de09a603494bfb5f239426fa73ef`
(2026-07-28). Per-file last-touching commit at that HEAD:

| File | reify commit | date |
|---|---|---|
| `warm-lane-gc.sh` | `973fde7955` | 2026-07-28 |
| `warm-lane-gc-sweep.sh` | `434fd5a181` | 2026-07-28 |
| `thin-warm-lane.sh` | `9be2bfe61a` | 2026-07-23 |
| `warm-lane-disk-guard.sh` | `9a43111f6c` | 2026-07-19 |
| `warm-lane-audit.sh` | `77802c19fd` | 2026-07-27 |
| `warm-lane-degenerate-ref-check.sh` | `23e620a9b3` | 2026-09-17 |
| `provision-warm-lane-fs.sh` | `b37e00eaa6` | 2026-07-11 |
| `lib_live_refs.sh` | `434fd5a181` | 2026-07-28 |
| `lib_portable.sh` | `473217c346` | 2026-06-10 |
| `lib_task_citation.sh` | `d1fea3e2c0` | 2026-09-17 |
| `lib_lane_state.sh` | — **dark-factory-native** | 2026-07-30 |

**The last two rows are a PARTIAL re-sync, not a new anchor.** The
`638d97d8aba4de09a603494bfb5f239426fa73ef` HEAD above still governs every
other row. `warm-lane-degenerate-ref-check.sh` and `lib_task_citation.sh` were
re-synced by **task 5566** at reify HEAD
`63ac8d9b4b5faf761bf3fbe79339d56120f31431` (2026-09-17), which is where
reify's task 7244 landed; nothing else was re-copied at that HEAD, so do not
read the pair's dates as moving the anchor for the rest of the table. See
Delta 11.

The copy includes reify task 5572's per-lane live-consumer `/proc` check
(merged as reify `a4bddeaa51`), which is why `lib_live_refs.sh` travels here.

`lib_lane_state.sh` has **no reify commit and no reify source** — do not try to
diff it against one. It is dark-factory-native, added by **task 3074** (PRD leaf
β) as an *extraction*, not a copy: it holds the two facts dark-factory owns and
reify does not (the `.lane-state` record format, whose `state` values are the
`LaneState` enum in `orchestrator/src/orchestrator/lane_lifecycle.py`; and
`PROTECTED_PREFIXES` in `orchestrator/src/orchestrator/git_ops.py`). Its
lane-state half was lifted out of `warm-lane-audit.sh`'s private reader — see
Delta 4; its protected-prefix half is consumed by `warm-lane-gc.sh`'s
`PROTECT_GLOB` default — see Delta 9 (task 3292 closed the drift leaf γ had
deferred; Delta 6 records what γ did land).

### Why these eleven

PRD §2.1 audited each script for project-specific coupling. The token grep
`cargo|rustc|RUSTFLAGS|OUT_DIR|Cargo|nextest|occt|manifold|reify-gui|tauri`
across the ten **relocated** files yields exactly two hits, both in
**comments** — re-measured by task 5566 after `lib_task_citation.sh` joined
them, at `warm-lane-gc.sh:231` and `lib_live_refs.sh:137`, and the new lib adds
none. No code path branches on anything reify-specific. (The eleventh,
`lib_lane_state.sh`, is dark-factory-native and deliberately reify-free; the
coupling it carries runs the other way, to
`orchestrator/src/orchestrator/`.)

Four of the eleven source a sibling lib, and none of those libs is among the
seven scripts the task named — copying only seven would ship four scripts that
cannot execute:

- `warm-lane-gc.sh` and `warm-lane-gc-sweep.sh` `source
  "$SCRIPT_DIR/lib_live_refs.sh"` and deliberately `exit 2` when it is absent
  (reify 5572 made that fail-loud so a silently-missing liveness guard cannot
  recur).
- `warm-lane-audit.sh` sources **two**: `$SCRIPT_DIR/lib_portable.sh`, and —
  since task 3074 — `$SCRIPT_DIR/lib_lane_state.sh`. Both sit behind a guard
  copied in shape from `warm-lane-gc.sh`'s; `lib_portable.sh`'s was added by
  task 3370 and is ordered **first**, so a copy carrying neither sibling
  reports it. See Delta 4 and Delta 8.
- `warm-lane-degenerate-ref-check.sh` `source`s
  `$SCRIPT_DIR/lib_task_citation.sh` — since task 5566 — behind the same
  `exit 2` guard shape. That lib is the single copy of the "cites task N"
  grammar its classification turns on. See Delta 11.

`orchestrator/tests/test_warm_lane_scripts_shipped.py` pins this as executable
behaviour: every file above is checked for presence, the owner-execute bit and
`bash -n`, and each of the four sourcing scripts is run with `--help` from
this directory as proof its libs actually travelled along.

### What deliberately did NOT move

`seed-warm-lane.sh` and `refresh-warm-base.sh` stay project-owned, behind the
PRD §5 contract: seeding a lane and refreshing the warm base are inherently
project-specific (what to build, what to prime), so dark-factory names them
rather than implements them.

## Documented deltas from the reify sources

Every file here that HAS a reify source is **byte-identical** to it except as
recorded below. (`lib_lane_state.sh` has none — see Provenance.) Keeping them
diffable against reify is the cheap drift check available for the whole α→κ
duplication window, so deltas are enumerated rather than absorbed. Policy is
untouched: `REIFY_*` env-var names, default image/mount paths and exit-code
taxonomies are verbatim (renaming them is downstream work — leaves β/γ/δ/ε —
not this leaf), and no provenance header is prepended to any script.

### Delta 1 — `provision-warm-lane-fs.sh`, `REPO_ROOT` resolution

Added by leaf α, together with Delta 2 below — same file, same concern (path
resolution at the new home). Deltas 4, 6 and 7 are the other file-content
divergences from reify in this directory. **Delta 7 is this same wrong-path
class with a different cause**, and it revisits both this file's `_SCRIPT_DIR`
assignment and `_default_mount()` below.

The script derives `REPO_ROOT` from its own location, and `_default_mount()`
hangs the operator-facing default `--mount` off it. In reify the script sat at
`<repo>/scripts/`, so a single `..` reached the repo root. Here it sits two
levels deeper, at `<repo>/orchestrator/scripts/warm-lane/`, where the
inherited `..` lands on `<repo>/orchestrator/scripts` — silently advertising
`<repo>/orchestrator/warm-lanes` instead of the repo's sibling `warm-lanes`
dir, to an operator about to provision a multi-terabyte volume.

So a literal byte-copy would BREAK parity here rather than preserve it. The
relocated copy ascends three levels instead of one, by **pure path
arithmetic**: `REPO_ROOT="$(cd "$_SCRIPT_DIR/../../.." && pwd)"`. The depth
below the repo root is fixed by the repo layout, the file physically lives
inside whichever checkout (or linked worktree) is running it, and the
expression reads **no environment at all** — so it yields that checkout's own
root, logically spelled, exactly as the old `..` did.

A `git -C "$_SCRIPT_DIR" rev-parse --show-toplevel` probe was tried first and
**rejected**. Each of its three failure modes lands in the same wrong-path
class this delta exists to prevent, and an existence guard catches none of
them because every wrong path exists:

| Failure mode | Result |
|---|---|
| Inherited `GIT_DIR` (the standard git-hook / `git rebase --exec` / filter-branch environment) is not cleared | returns `$_SCRIPT_DIR` itself → mount `<repo>/orchestrator/scripts/warm-lanes` |
| Ascends into any **enclosing** repo | an unpacked tree inside e.g. a dotfiles checkout resolves to that outer root, and the "fresh host" fallback never fires |
| Returns the **symlink-resolved** path, where `..` and the arithmetic return the logical one | the two disagree under a symlinked ancestor — reify's production `.worktrees` is one |

**This restores the pre-relocation semantics; it does not change behaviour.**
Everything downstream — the usage text, `--img` / `--mount` / `--grow`
handling, and the XFS/loopback semantics PRD §10 puts out of scope — is
untouched. Pinned by
`orchestrator/tests/test_warm_lane_scripts_shipped.py::TestProvisionRepoRootParity`,
which covers the checkout case, the no-git-metadata case, an inherited
`GIT_DIR`, and the two ascend-past-worktrees spellings.

### Delta 2 — `provision-warm-lane-fs.sh`, `_default_mount` worktrees spelling

Same file, same concern as Delta 1: a path derivation whose inherited form
does not survive the new home.

`_default_mount()` ascends one level when the repo root's parent is a
worktrees directory, so the warm-lanes dir lands BESIDE the worktrees tree
rather than inside one worktree. reify's copy tested for the literal
`worktrees`. dark-factory's own worktree directory is **`.worktrees`**
(`GitConfig.worktree_dir` default, `orchestrator/src/orchestrator/config.py`),
a shape the relocation makes newly reachable — run from a task worktree, which
is the normal way an agent or operator in this repo would invoke it, the
inherited test fails to match and the advertised default becomes
`<repo>/.worktrees/warm-lanes`: inside the worktrees tree, the exact outcome
the ascend exists to prevent.

The relocated copy matches **both spellings** (`worktrees` and `.worktrees`).
Like Delta 1 this preserves the ascend's intent at the new depth rather than
changing it, and it is pinned by the `.worktrees` case in
`TestProvisionRepoRootParity`.

### Delta 3 — `warm-lane-gc.sh` / `thin-warm-lane.sh` sibling-seed defaults

A **documented behavioural caveat, deliberately NOT patched** — no file
content diverges *for this reason*. (Delta 7 later diverged both files for an
unrelated one, and its `--reseed` measurement sharpens the warning this delta
already carried.) Both scripts default `--seed-script` to a sibling
`seed-warm-lane.sh` that PRD §5 keeps project-owned, so at the new location
that default cannot resolve. Rather than patch the scripts to guess at a
project-owned path (PRD invariant C-1), the caller resolves it: see
"Sibling-seed defaults, and who resolves them" below for which script is
wired by the caller, which is left alone, and why.

### Delta 4 — `warm-lane-audit.sh` reads lane state through `lib_lane_state.sh`

**`warm-lane-audit.sh` is no longer byte-identical to reify.** Added by **task
3074** (PRD leaf β, §8's resolved contested seam: *extract and unify*). Two
changes, both structural:

1. It now sources `$SCRIPT_DIR/lib_lane_state.sh`, behind a fail-loud guard
   copied in shape from `warm-lane-gc.sh`'s `lib_live_refs.sh` guard: an
   explicit `[ ! -f ]` test → a "not found next to warm-lane-audit.sh" message
   on stderr → `exit 2`.

   `exit 2` (the wiring/usage sentinel), **not** a degrade-to-UNKNOWN, and not
   `1`. The audit's "never abort" rule is about lane-level *data* problems — an
   unreadable or corrupt record must degrade that lane to UNKNOWN and never kill
   the run — not about deployment wiring, where nothing about the invocation
   could have avoided it and no retry fixes it. Degrading would report *every*
   lane as UNKNOWN, indistinguishable from a real pool-wide state-dir outage: a
   triage trap.

2. `_read_lane_assignment` is now a thin adapter over the lib's
   `lane_state_read` + `lane_state_class`. What moved out — `_record_text`,
   `_record_scalar`, `_lane_record`, and the normative raw-state → column
   `case` — was never really this script's to own: it describes dark-factory's
   own durable record format, whose `state` values *are* the `LaneState` enum.
   A project-agnostic script holding a private copy of that is the INV-5
   lockstep duplication leaf β exists to close, and the extraction takes the
   single-slurp discipline and the non-creating (existence/readability tests
   only) guarantee with it verbatim.

   **The caller boundary is unchanged.** `LANE_ASSIGNED_STATE`,
   `LANE_UNKNOWN_CAUSE` and `LANE_RECORD_TASK_ID` keep their names, semantics
   and the single-observation invariant; the cause vocabulary
   (`no-readable-record`, `unparseable-record`, `unrecognized-state:<raw>`) is
   preserved, with the third still *derived* rather than stored so there is no
   second copy of the recognized-state table to drift. Every downstream consumer
   in the script is untouched. Pinned by
   `orchestrator/tests/test_lane_state_lib.py`, which asserts the per-lane
   `assigned` column, the `pin` task id and all three UNKNOWN causes against a
   synthetic mount, plus a single-definition-site guard: the record-scalar `sed`
   idiom now appears exactly **once** across `*.sh` in this directory.

**`warm-lane-gc.sh` was deliberately untouched by leaf β**, whose
hand-maintained `PROTECT_GLOB` default carried a comment admitting it mirrored
dark-factory's `PROTECTED_PREFIXES` across a repo boundary. β shipped what
replaces it — `lane_protect_glob` and the machine-checked
`LANE_PROTECT_GLOB_FALLBACK` — leaving the rewire to the leaf that owns that
file. Note β's renderer excludes the bands a pool sweep OWNS (`_lane-`,
`_spec-`): handing those to gc as *protected* would make it skip every lane in
both passes and stop reclaim outright.

**That rewire has landed (task 3292) — see Delta 9.** The paragraphs below are
kept as the record of why it was deferred through two leaves rather than
overlooked, because the deferral was measured; the resolution is appended to it.

- Whatever names the deployment's interactive band must reach the bridge in
  `REIFY_WARM_LANE_IACT_PREFIX` (below). `warm-lane-gc.sh`'s hand-maintained
  default hardcoded `_iact-*`, and also omitted `.lane-state` and `.task-meta` —
  the live INV-5 drift leaf β exists to make un-writable.

  **γ did NOT delete that default** (task 3075; this claim previously read
  "persists until γ deletes that default" and is corrected here rather than
  left false). It was deferred with the cost measured, not overlooked:

  - The rendered glob
    `_merge-*,_solo-*,_substrate-gate-*,_merge-verify,_offline-deep,.lane-state,.task-meta,_mainprobe-*,_mainsweep-*,_iact-*`
    is behaviourally **indistinguishable** from gc.sh's then-current default
    under any black-box test: `_merge-verify` is already covered by `_merge-*`,
    and `.lane-state`/`.task-meta` are dot-prefixed while gc.sh's candidate loop
    is `"$WORKTREES_DIR"/*/` with no `shopt -s dotglob` anywhere. The sole
    observable payoff is honouring `REIFY_WARM_LANE_IACT_PREFIX`.
  - Measured on this host 2026-07-30, the python bridge costs 1.05s / 4.45s /
    3.48s across three consecutive runs. Negligible against a 25-40 minute
    production reclaim pass, but it fires once per reclaim **invocation** and
    the gc bash suite invokes reclaim roughly 30 times — it would materially
    degrade the very suite γ's core change depends on for verification.

  The wiring wanted a memoization or an opt-out designed alongside it, so it was
  filed as a follow-up rather than bundled onto the leaf that closes
  esc-5334-6.

  **Resolution (task 3292).** That indistinguishability analysis is what made
  the wiring cheap to land safely: because the rendered glob is a strict
  superset of the old literal that no black-box test can tell apart, the test
  suites could pin the static `LANE_PROTECT_GLOB_FALLBACK` through gc.sh's own
  already-documented `REIFY_WARM_LANE_GC_PROTECT_GLOB` knob — which the
  `[ -n "$PROTECT_GLOB" ] ||` default short-circuits *before* the bridge runs —
  and provably not move a single existing assert. Only the handful of blocks
  that assert what the DEFAULT protects opt back out, via a named
  `run_helper_live_default`. So the mitigation is the *existing* knob, not a new
  one: neither of the two options the follow-up anticipated was taken.
  In-process memoization was rejected as measurably inapplicable (every one of
  the ~32 gc-suite invocations is a fresh `bash "$SCRIPT"` process, so there is
  no process to memoize within), and a cross-process on-disk cache was rejected
  on principle — a stale cached glob is precisely the silent
  `PROTECT_GLOB`-vs-`PROTECTED_PREFIXES` drift this leaf exists to make
  un-writable, and its failure mode is the severe direction.

  The observable payoff is delivered and pinned in both directions (Block X-band
  in `orchestrator/tests/warm-lane/test_warm_lane_gc.sh`: with
  `REIFY_WARM_LANE_IACT_PREFIX=_myiact-`, a `_myiact-` worktree survives and a
  stock `_iact-` one no longer does). The `||` fallback is no longer a contract
  either — Block X-degrade exercises it in situ. Re-measured wall-clock for the
  affected suites is in `orchestrator/tests/warm-lane/README.md`.

### Delta 5 — `lib_lane_state.sh` reads the deployment's interactive band

Not a divergence from reify (this file has no reify source) — recorded here
because it is the one input `lane_protect_glob` takes from *outside* the
registry, and the one way its answer can be wrong while looking right.

Ten of the eleven `PROTECTED_PREFIXES` keys are constants, so rendering them
cannot be wrong. The eleventh is not in the constant at all: the interactive
band is `git.iact_prefix`, per-deployment config, merged in by
`default_protected_prefixes()`. A bridge that rendered the FIELD DEFAULT would
hand a renamed deployment a glob protecting `_iact-*` — a band it never mints —
while omitting the band it does, so a wired sweep could reclaim live interactive
worktrees. `lane_protect_glob` therefore reads **`REIFY_WARM_LANE_IACT_PREFIX`**
(unset or empty ⇒ the field default; the value REPLACES the band rather than
adding to it, matching `GitOps.protected_prefixes()`). The `REIFY_` namespace is
the sibling scripts' one env namespace — renaming it wholesale is downstream
leaf work, not this file's.

Two robustness properties of the same bridge, pinned by the same test class:

- **It prefers the checkout's own `.venv/bin/python3` over `PATH`.** The import
  chain reaches pydantic, so a dependency-less system interpreter — what
  `PATH=/usr/bin:/bin` gives a systemd unit, which is exactly this lib's stated
  invocation path — cannot run it at all. Without the preference the sweep would
  warn and degrade to the static fallback on *every* run: permanently reduced,
  while training operators to read `[warn]` as noise. PATH `python3` stays the
  fallback for a checkout with no in-tree venv.
- **It probes the repo root it resolved, rather than trusting the arithmetic.**
  These scripts are relocatable (`GitOps._project_script` prefers a
  project-local override copy), and `cd ../../..` succeeds on almost anything,
  so a copy at a different depth would aim PYTHONPATH at an unrelated directory
  and degrade with an opaque `ImportError`. The root must now carry
  `orchestrator/src/orchestrator/git_ops.py` or the function emits its one
  attributable `[warn]` and returns non-zero. This narrows the wrong-root hazard
  to *another dark-factory checkout at the same depth*; it does not eliminate
  it, and that residue is the one case the fail-loud contract cannot detect.

Both halves now have a shipped consumer: `warm-lane-gc.sh`'s `PROTECT_GLOB`
default (Delta 9). The band override is pinned end-to-end by Block X-band, and
the root probe above is what Block X-degrade uses to reach the degrade branch
hermetically — it relocates a copy of gc.sh so the resolved root is `/`, which
carries no witness.

### Delta 6 — `warm-lane-gc.sh` decides Pass-1 reclaim from the lane record

**`warm-lane-gc.sh` is no longer byte-identical to reify.** Added by **task
3075** (PRD leaf γ), closing reify escalation `esc-5334-6`. Two changes, the
same shape as Delta 4's:

1. It now sources `$SCRIPT_DIR/lib_lane_state.sh`, behind a fail-loud guard
   whose message reuses the *verbatim* fragment `lib_lane_state.sh not found
   next to` that Delta 4's guard established — so
   `orchestrator/tests/test_warm_lane_scripts_shipped.py`'s
   `FAIL_LOUD_FRAGMENTS` covers both scripts with one entry. `exit 2` (the
   wiring sentinel), not `1`, for the reason the exit-code table gives: a
   silently-absent reader degrades reclaim back to the approximation this leaf
   removes.

2. Pass 1's reclaimability gate reads dark-factory's own durable record at
   `<worktrees-dir>/.lane-state/<lane>.json`, per lane, under the flock the
   loop already holds. `assigned`/`in_use` preserve; everything else — and
   every unknown, unreadable or corrupt reading — falls through.

   The rule this replaces was reify's, and it was **false in both halves**:
   *"reclaimability is computed purely from filesystem + git + flock;
   dark-factory FREE/ASSIGNED state is NOT consulted; FREE/idle ≈ no live
   consumer holding the lane flock."* The inv.2 flock is held only across the
   acquire reseed and across `run_scoped_verification`, never across the
   implement phase, so an assigned lane looks FREE for most of its life and
   task 5326's always-reclaim fired on it.

   Pass 1 now has **three** gates in order — flock, record, `/proc`
   live-reference. The record is between the other two because it is both
   cheaper (~1ms vs a measured ~1.9s) and *authoritative*; an assigned lane
   short-circuits before the walk, so a busy pool gets **cheaper**. The
   task-5572 `/proc` scan is retained unconditionally as the recordless
   fallback — gating it on "record absent" would let a `released` record reset
   a lane whose straggler build is still live. **Pass 2 is deliberately
   unchanged**: its candidates match neither `--lane-glob` nor
   `--protect-glob`, so no lifecycle record is ever minted for them.

   Fail-open is load-bearing in one direction only: preserving on an unknown
   reading would freeze reclaim whenever `.lane-state/` is absent, re-creating
   the 2026-07-10 ENOSPC accretion outage.

### Delta 7 — five scripts derive paths without forking `dirname` or `basename`

**`provision-warm-lane-fs.sh`, `thin-warm-lane.sh`, `warm-lane-audit.sh`,
`warm-lane-gc.sh` and `warm-lane-gc-sweep.sh` are no longer byte-identical to
reify.** Added by **task 3279**. Same wrong-path class as Delta 1, one cause
removed: there the ascent was the wrong DEPTH; here the starting point itself
silently becomes the caller's CWD.

Scope, stated precisely because it grew once: this is a **five-script `dirname`
change** (the self-directory resolutions, below) **plus a three-script
`basename` change** (`thin-warm-lane.sh`, `warm-lane-gc.sh`,
`warm-lane-audit.sh` — see "The `[ ... ]` vs assignment asymmetry"). The second
half was found by review *after* the first had landed and had already written
off the residual `basename` forks as cosmetic. It is not split into a Delta 8:
same divergence class, same files, same task — and the provenance table's only
drift check during the α→κ duplication window is that it stays diffable.

Every one of the five resolved its own directory as
`"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`. `dirname` is an **external
binary**, so that expression makes the resolution silently depend on `PATH` —
and a `PATH` without it does not error: the substitution yields **empty**,
`cd ""` **SUCCEEDS** as a no-op, and the directory resolves to the **caller's
CWD**, at exit 0, with no diagnostic. A systemd unit's `PATH` need not carry
coreutils, and `provision-warm-lane-fs.sh` is specifically run on a fresh host
to provision the pool substrate — where a minimal `PATH` is likeliest.

The sites converted, all to the parameter-expansion idiom
`lib_lane_state.sh:265-273` already carried (`case "$src" in */*)
dir="${src%/*}" …` then `cd "$dir" && pwd`; `cd`, `pwd` and `case` are
builtins, so the arithmetic needs nothing on `PATH`):

| File | Sites |
|---|---|
| `provision-warm-lane-fs.sh` | `_SCRIPT_DIR`, **and** `_default_mount()`'s `dirname "$REPO_ROOT"`, its `basename "$parent"` and the `dirname "$parent"` in its ascend branch |
| `warm-lane-audit.sh`, `warm-lane-gc.sh`, `warm-lane-gc-sweep.sh` | `SCRIPT_DIR` |
| `thin-warm-lane.sh` | `_script_dir` (the `--seed-script` default) |

A **sixth** member joined this class later, under a different task and in a
file 3279 did not touch, so by this delta's own no-splitting rule it is
recorded separately: `warm-lane-degenerate-ref-check.sh`'s `SCRIPT_DIR`, added
with its `lib_task_citation.sh` source by task 5566 — see Delta 11. The
rationale below is not restated there.

Plus the six **non**-self-directory path derivations found by the later review
pass, converted in the same task for the reason in "The `[ ... ]` vs assignment
asymmetry" below:

| File | Site | What it feeds |
|---|---|---|
| `thin-warm-lane.sh` | self-clobber guard, `basename "$_rp_lane_dir"` → `${_rp_lane_dir##*/}` | the `= "base"` refusal, 33 lines above `rm -rf "$LANE_DIR/target"` |
| `warm-lane-gc.sh` | `BASE_TARGET`, `dirname "$MOUNT"` → guarded `%/*` | the `--mount` derivation |
| `warm-lane-gc.sh` | `_is_reclaimable`'s `name` | diagnostics only |
| `warm-lane-gc.sh` | classification loop's `name` | `_matches_glob "$name" "$PROTECT_GLOB"`, then `"$LANE_GLOB"` |
| `warm-lane-gc.sh` | Pass 1 lane `name`, Pass 2 orphan `name` | `${WORKTREES_DIR}/${name}.lock`, the per-entry mutex |
| `warm-lane-audit.sh` | resident-walk `name` | `_lane_role`, `_probe_live "$MOUNT/$name.lock"`, `_read_lane_assignment` |

The five leaf extractions are plain `${X##*/}`: each operand is an assignment
whose trailing slash the caller already stripped, so expansion and `basename`
agree on every reachable input. `BASE_TARGET` needed the **full guarded shape**
instead, because the derived VALUE is documented in two header comment blocks
and must not move. Measured against `dirname` across nine inputs, which found
two edges beyond the ones anticipated: a `--mount` with a **trailing slash**
(`dirname /a/b/wt/` = `/a/b`, bare `%/*` = `/a/b/wt` — a silent misderivation on
plausible operator input), fixed with the same trim-then-`%/*` shape
`lib_lane_state.sh:141-146` already uses; and then `--mount /`, which that trim
newly broke (trims to empty → `.` where `dirname` gives `/`), so the trim needs
its own empty-guard alongside the `%/*` one. The result is byte-equal to
`dirname` over `/worktrees`, `worktrees`, `/a/b/wt`, `/a/b/wt/`, `/`, `.`,
`./x`, `a/` and the real host value. Multiple trailing slashes (`/a/b//`) still
differ; recorded rather than papered over, and it matches `lib_lane_state.sh`'s
existing behaviour.

`provision-warm-lane-fs.sh` needed **all four** of its sites. Measured: with
`_SCRIPT_DIR` converted and `_default_mount()` left alone, the advertised
default is *still* `/warm-lanes`, because `_default_mount()` re-forks `dirname`
on its own. A fix stopping at the assignment would pass a naive `SCRIPT_DIR`
check while changing nothing an operator ever sees.

Its two `_default_mount()` sites share **one `_parent_of()` definition**. The
duplication argument above covers only the five **self-directory** resolutions
— a script's own directory cannot be resolved via something it has not located
yet — and does not extend to a derivation running after bootstrap. Sharing was
not cosmetic: the two copies lacked the `%/` trim `warm-lane-gc.sh`'s copy
needed, so "the same idiom" had already drifted into three variants, and the
shared definition carries the canonical guarded shape (verified byte-equal to
`dirname` over the same nine inputs). `_SCRIPT_DIR` deliberately stays
open-coded — it is the bootstrap everything else depends on and must not
acquire a dependency of its own, not even on a function definition a later edit
could move below it. The refactor is behaviour-preserving by measurement: the
advertised default mount is identical across the plain / `worktrees` /
`.worktrees` nestings under both a full and a `dirname`-less `PATH` (6/6), and
the full usage text is byte-identical.

`warm-lane-gc.sh`'s `BASE_TARGET` copy is **not** folded in with them: the only
place both scripts could share is `lib_lane_state.sh`, which
`provision-warm-lane-fs.sh` does not source at all (and which task 3279 holds
no lock on). Left as the third copy, named here rather than left to look
overlooked.

#### What was measured

Harness: a stub `PATH` entry whose `dirname` shim exits 127 (observationally
identical to `command not found`), prepended to a real `PATH`. Base HEAD
`8d276d3c5f`.

| Script | `CWD=/` under a `dirname`-less `PATH` | CWD holding same-named decoy siblings |
|---|---|---|
| `provision-warm-lane-fs.sh` | **rc=0**, advertises default `--mount` = **`/warm-lanes`**, the bare filesystem root (control: `<repo-parent>/warm-lanes`). No guard exists anywhere on this path. | n/a — the resolution feeds a printed path, not a `source` |
| `warm-lane-gc.sh` | rc=2, `lib_live_refs.sh not found next to` | **rc=0, sourced BOTH decoys** (`lib_live_refs.sh`, `lib_lane_state.sh`) |
| `warm-lane-gc-sweep.sh` | rc=2, same shape | **rc=0, sourced the decoy `lib_live_refs.sh`** |
| `warm-lane-audit.sh` | **rc=1** — bash's own bare `source` failure on `//lib_portable.sh`, **not** its `exit 2` guard | **rc=0, sourced BOTH decoys** (`lib_portable.sh`, `lib_lane_state.sh`) |
| `thin-warm-lane.sh` | rc=0, `SEED_SCRIPT=<CWD>/seed-warm-lane.sh` | **EXECUTED the decoy** and reported `[ok] Re-seeded` |

#### The corrected hypothesis

This was filed expecting the sourcing scripts to land on their existing
fail-loud wiring guards and `exit 2` — i.e. to degrade **loudly**. They do not,
and the reason is structural: the guards are
`[ ! -f "$SCRIPT_DIR/lib_*.sh" ]`, which test the **mis-resolved** path. Any
CWD that happens to hold a same-named file satisfies them. **The loud degrade
is contingent on the caller's CWD being empty**, so the guards cannot detect
this class at all — and the middle column above is the *lucky* case, not the
contract. The realistic, non-adversarial trigger for the right-hand column is
invoking one of these from reify's own `scripts/` dir, or from another
dark-factory checkout's `warm-lane/` dir: both carry precisely these
filenames, at a possibly older version, in scripts whose job is deleting
worktrees.

`warm-lane-audit.sh` diverges from the taxonomy a second way: the `[ ! -f ]`
guard Delta 4 added covers only `lib_lane_state.sh` and sits **after** the
`lib_portable.sh` `source`, which has no guard at all. So the bare case exits
**1**, the runtime code, where this class is assigned the exit-2 wiring
sentinel. That gap is real and separately actionable, but it is a behaviour
change this delta does not make: with `SCRIPT_DIR` resolved correctly the bare
`source` failure is unreachable in every case measured.

**Closed since.** Task 3370 added the missing `lib_portable.sh` guard — see
**Delta 8**. The finding above stands as the historical record of how it was
found (and the `rc=1` row in the table above remains a dated measurement of
base behaviour, not a live claim); what changed is only that the gap it
declares open is now shut.

#### The `[ ... ]` vs assignment asymmetry — the rule for judging any future fork

The generalisable lesson, and the reason the first pass of this delta got the
residual forks wrong. **`set -e` does not see a failed substitution inside
`[ ... ]`.**

* `name="$(basename "$X")"` — an **assignment**. A missing binary propagates
  **127** and `set -e` kills the script.
* `[ "$(basename "$X")" = "base" ]` — a **test**. The substitution yields the
  empty string, the comparison is simply **false**, and execution continues.

Same missing binary, opposite blast radius. Measured on task 3279's HEAD
`27fbfb4ea5` with `basename` shimmed to exit 127:

| Site | Context | Measured |
|---|---|---|
| `thin-warm-lane.sh` self-clobber guard (`REIFY_WARM_LANE_MOUNT` unset, `lane_dir=<pool>/base`) | `[ ... ]` | guard **silently defeated** — `[ok] Freed <pool>/base/target`, `[ok] Thinned lane`, **rc=0**, pool seed source destroyed. Control: `refusing to thin`, rc=1, tree intact. |
| `warm-lane-gc.sh` classification loop | assignment | abort at **rc=127** *before* `_matches_glob "$name" "$PROTECT_GLOB"` runs; protected `_merge-x` **survives**. Control: `skipping protected: _merge-x`, rc=0. |
| `warm-lane-audit.sh` resident walk | assignment | abort at **rc=127** after the first info line; **no** report rows. Control: the `HEADROOM` / `PINNED` rows. |

The `thin` site is the load-bearing half of that guard and the **only** half
that fires when `REIFY_WARM_LANE_MOUNT` is unset — the two mount-relative checks
above it are inside `if [ -n "${REIFY_WARM_LANE_MOUNT:-}" ]`.

The two assignment sites are therefore loud and non-destructive **today — by the
accident of their syntactic context, not by any guard**. Read that as an
accident, not a designed safety property: nothing in either script chose it, and
the identical fork one line-shape over deletes a pool's seed source. Converting
them removes the dependency on the accident.

Converting the gc classification site also closed a **latent bug the 127 was
masking**: an empty `$name` matches neither `PROTECT_GLOB` nor `LANE_GLOB`, so a
protected `_merge-*` / `_iact-*` worktree would fall through to
`orphan_candidates` — i.e. be classified as removable. Only the abort stood
between that and an orphan-removal pass over a protected entry.

After the conversion, both scripts run under a `basename`-less `PATH` with
output **byte-identical** to their full-`PATH` control.

#### Why patch here rather than upstream to reify

The rule at the top of this section makes this a real decision, not a
one-liner. It was settled by measurement: reify's working tree has already
drifted far past the pinned copy HEAD `638d97d8` (`warm-lane-audit.sh` +675
diff lines, `warm-lane-gc.sh` +334), so "upstream and re-copy" would drag
unrelated reify drift through a dark-factory merge lane that cannot verify it
— defeating the pinned-HEAD provenance table it would be honouring. A
cross-repo commit also cannot be part of a dark-factory task's diff, reviewed
by its reviewer, or verified by its suite. Delta 1 is direct precedent for the
same class.

#### What is deliberately NOT changed

**Only the exact spelling `$(basename "$0")`, in usage and diagnostic strings,
still forks.** It feeds no path resolution — worst case a blank program name in
a `Usage:` line — so converting it would widen the diff across five reify
byte-copies for nothing. The pinning test for `provision-warm-lane-fs.sh`
deliberately does not assert on the `Usage:` line that this blanks.

Note the scope, which is the correction: this claim is about **that one
spelling**, not about `basename` generally. An earlier revision of this section
said the residual `basename` forks were cosmetic *as a class*, at a moment when
six non-`$0` forks remained in the very files this delta covers — one of them
silently defeating a data-destruction guard. It told the next reader there was
nothing here to act on.

What makes the claim safe to make now is that it is **enforced rather than
asserted**: `test_only_cosmetic_program_name_forks_remain` fails on *any*
`dirname`/`basename` substitution on a non-comment line in `warm-lane/*.sh`
except that one spelling. The offender scan is a regex covering both `$( … )`
and backtick syntax with leading whitespace tolerated; the waiver is matched
**literally**, so a future `$(basename "$0" .sh)` or `$(dirname "$0")` is caught
rather than waved through. Cosmetic-only is now true by construction, and a
regression is caught mechanically instead of by the next reader trusting this
paragraph.

#### Pinned by

All in `orchestrator/tests/test_warm_lane_scripts_shipped.py`:

* `TestProvisionRepoRootParity::test_default_mount_survives_a_path_without_dirname`
  — parity against a same-run full-`PATH` control (never a hardcoded path),
  over the plain, `worktrees` and `.worktrees` nestings, since `_default_mount()`
  reaches a different fork in each.
* `TestSiblingResolutionIgnoresTheCallersCwd` — asserts on **which file was
  resolved**, not on the exit code, for the reason above: the exit code is a
  function of the caller's CWD, not of the defect. The two `--help` cases
  pair an absence assertion (a decoy-CWD marker must be absent) with a
  **positive control**, `Usage:` — the heredoc all three scripts print on
  `--help` — because an absence assertion alone is satisfied by a script that
  died at line 1 for an unrelated reason, without one the case could go green
  having never reached the resolution it names. The `--reseed` case (which
  takes no `--help`, so it cannot use `Usage:`) instead asserts directly on
  the RESOLVED, absolute seed-script path: `thin-warm-lane.sh` interpolates
  `$SEED_SCRIPT` into an `info` line, emitted to stderr on the line
  immediately before it executes it. Matching that path — not the prose
  around it, which carries no contract and would otherwise couple the case to
  an incidental log reword — proves both properties in one assertion: that
  the run reached resolution (an early exit never prints it), and that
  resolution landed beside the script rather than under the caller's CWD.
* `TestLeafExtractionBehaviourWithoutBasename` — the **behavioural** half for
  the two converted leaf extractions, which the static gate below cannot
  reach: it catches the fork reappearing, not a conversion that changed what
  the script does. Under a `basename`-less `PATH`, `warm-lane-gc.sh reclaim`
  must still report `skipping protected: _merge-x` at rc=0 with the entry
  intact (the latent protected-glob fall-through), and `warm-lane-audit.sh`
  must still emit its per-lane, `HEADROOM` and `PINNED` rows. These are what
  make the "byte-identical to their full-`PATH` control" claim above CI-checked
  rather than a recorded hand measurement.
* `TestGcBaseTargetMatchesDirname` — the `BASE_TARGET` parity table, over the
  eight reproducible inputs of the nine measured above. Asserted against the
  real `dirname` **binary**, not `os.path.dirname`: the two disagree on three of
  these very inputs (`worktrees` → `.` vs `''`, `/a/b/wt/` → `/a/b` vs
  `/a/b/wt`), so a python control would encode the naive behaviour the table
  exists to reject. Without it a "simplification" back to a bare `${MOUNT%/*}`
  misderives a trailing-slash `--mount` with every other case still green;
  measured against that mutation, five of the eight go red.
* `TestThinSelfClobberGuardDoesNotDependOnPath::test_self_clobber_guard_survives_a_path_without_basename`
  — asserts all three of: the sentinel under `<pool>/base/target` survives
  (the actual safety property), rc≠0, and `refusing to thin` on stderr. Each
  alone can pass for the wrong reason. Its companion case pins that a
  normally-named `<pool>/_lane-1` still thins under the same `PATH`, so the fix
  is shown to be a **correction** of the comparison rather than a widening of
  it.
* `TestNoShippedScriptDerivesAPathByForking::test_only_cosmetic_program_name_forks_remain`
  — the directory-wide drift gate. Because a script's own directory is what
  tells it where its libs are, this idiom cannot be extracted into a sourceable
  helper without depending on the thing it resolves (`lib_lane_state.sh`
  carries its own copy for exactly that reason), so the five copies are
  deliberate. The gate is the inverse of Delta 4's single-definition-site
  guard: instead of "this idiom appears exactly once", it asserts **the forking
  spelling appears zero times**.

  **ONE gate, not two.** It began as a narrow scan for `$(dirname
  "${BASH_SOURCE` / `$(dirname "$0"`, and that scope is *exactly why nothing
  here flagged the `thin` self-clobber guard*: a `basename` fork on a variable
  is neither spelling. A gate that cannot see the class it guards is itself a
  defect, so it was widened to **any** `dirname`/`basename` substitution —
  which strictly subsumes the narrow scan, making a second test pure
  maintenance cost (every future exception encoded twice). The narrow scan's
  one distinctive contribution, the `cd ""` diagnosis, survives as an
  annotation appended to any offender matching the self-directory spellings.

  The offender scan is a **regex** tolerating whitespace and backticks
  (`$( dirname`, `` `dirname `` are both valid bash); the cosmetic waiver and
  the self-directory sub-classification are matched **literally**. That
  asymmetry is deliberate — the scan decides what is caught and must be
  generous, the waiver decides what is let through and must be exact.

  Whole-line comments are skipped, so the libs' usage headers and
  `warm-lane-gc.sh`'s prose do not false-trip it. That exclusion is
  deliberately not load-bearing: the two `warm-lane-gc.sh` header blocks that
  documented `BASE_TARGET=$(dirname "$MOUNT")/base/target` were reworded to
  describe the parent-of-`MOUNT` derivation without the fork spelling.

### Delta 8 — `warm-lane-audit.sh` guards its `lib_portable.sh` source

**`warm-lane-audit.sh` is no longer byte-identical to reify** (already true via
Deltas 4 and 7; this adds to the divergence). Added by **task 3370**, filed by
task 3279's architect pass as out of scope for it — and taking the Delta number
Delta 7 reserved by exclusion when it declined to split ("same divergence class,
same files, same task"). This change fails that rule's **same-task** limb, and
Delta 4's **same-class** one: it is a different sibling, a different task, and
not about lane state, so folding it into Delta 4 would make Delta 4's title
false.

**The defect.** The script sources three siblings and guarded only two. On main
HEAD `8d276d3c5f`, re-measured on `5828d94734`, a copy with no sibling
`lib_portable.sh` run with `--help` exited **1**:

```
warm-lane-audit.sh: line 155: <dir>/lib_portable.sh: No such file or directory
```

That is bash's own bare-`source` failure under `set -e` — the **runtime** code —
where this file's taxonomy and Delta 4's rationale both assign **2** to
"incomplete deployment, nothing about the invocation could have avoided it and
no retry fixes it". Two consequences worth naming: an operator or timer triaging
an exit 1 goes hunting for a data problem that is not there, and the failure is
attributed to **bash** rather than to the script.

**The fix.** A `[ ! -f "$SCRIPT_DIR/lib_portable.sh" ]` guard in Delta 4's
shape (same message template, same `exit 2`), ordered **FIRST** — above the
`source` it protects, and therefore above the `lib_lane_state.sh` guard — so a
copy carrying neither sibling reports `lib_portable.sh`. That is the same rule
`warm-lane-gc.sh` applies to `lib_live_refs.sh` before `lib_lane_state.sh`.
Both guards' comments now state the ordering and what reversing it would break,
so the position cannot be swapped silently.

**Also in this delta:** the script's own header and `_usage()` exit-code tables
now read **usage/WIRING** and name both sibling libs. They had never been
amended for task 3074's exit-2 path, so they already understated reality — and
shipping a guard whose exit code the script's own table does not describe would
have reproduced this delta's defect one layer up. Wording mirrors
`warm-lane-gc.sh`'s. The `0 — Always, on every valid invocation` line is
untouched: the advisory/never-gates contract (PRD §9.5 inv.12) binds every
*valid* invocation, and a wiring abort means there was no valid invocation to
report on — the reading task 3074 already relied on.

**Pinned by** `orchestrator/tests/test_warm_lane_scripts_shipped.py`:

* `TestAuditFailsLoudOnAMissingLibPortable` — stages the shipped bytes into a
  tmp dir withholding **both** siblings (the fixture is asserted first, so the
  case cannot pass vacuously) and asserts `rc == 2`, the script's own `ERROR`
  line naming the sibling, the **absence** of bash's bare-`source` shape, and —
  the ordering pin — that the `lib_lane_state.sh` message is absent from that
  run. `--help` is the invocation because it normally exits 0, so an rc of 2
  additionally proves the guard fires before argv parsing.
* A second case in that class restores `lib_portable.sh` and withholds only
  `lib_lane_state.sh`, asserting the SECOND guard still fires and still names
  itself. It is the positive half of the ordering pin: without it the negative
  assertion above would be satisfied just as well by a lane-state guard that
  had been deleted or softened to a silent degrade, so the pair would read as
  an ordering contract while only pinning "one guard exists". The lane-state
  guard's own primary coverage stays where task 3074 put it,
  `orchestrator/tests/test_lane_state_lib.py::TestAuditReadsThroughTheLib`.
* The amended `FAIL_LOUD_FRAGMENTS` entry `lib_portable.sh not found next to`,
  which replaces `lib_portable.sh: No such file`. That old fragment pinned a
  shape this delta makes unreachable for the only script that sources
  `lib_portable.sh` (`warm-lane-audit.sh`, the sole consumer repo-wide), so it
  would have become a pin that can never fire. The bare shape stays covered
  generically by the per-line scan, which trips on any sourced-lib name
  appearing with `No such file` **or** `not found`.

**Reachability, stated honestly.** After Delta 7 resolved `SCRIPT_DIR`
correctly, this branch is unreachable in every case measured: it needs a
genuinely missing `lib_portable.sh` — an incomplete deployment, or a
hand-assembled project-override copy under `<project_root>/scripts/`. That is
what makes this a **taxonomy** fix rather than a bug fix, and why it was filed
low priority.

### Delta 9 — `warm-lane-gc.sh` renders its `PROTECT_GLOB` default from the registry

**`warm-lane-gc.sh` diverges from reify a third time** (after Deltas 3, 6 and
7). Added by **task 3292**, closing the drift Delta 5 named and leaf γ deferred.
A different *class* from the two before it: Delta 6 gave gc.sh a new
dark-factory-owned *input*, and Delta 7 changed path arithmetic, but this
replaces a reify-sourced **default value** with a dark-factory-native
**resolution**. Recorded because this file's purpose is that every gc.sh
divergence from reify stays enumerated, and a value that is now *computed* per
deployment is not diffable against reify's literal at all.

**The defect.** The default protect set was a hand-copied literal whose own
comment admitted it mirrored `PROTECTED_PREFIXES` across a repo boundary — an
INV-5 lockstep duplication whose failure mode is silent: a band added to the
registry that nobody mirrors becomes a live managed worktree Pass 2 will
`git worktree remove --force`. It had already drifted (it omitted `.lane-state`
and `.task-meta`, and hardcoded `_iact-*` for a config-driven band).

**The fix.**

```
[ -n "$PROTECT_GLOB" ] || \
    PROTECT_GLOB="$(lane_protect_glob _lane- _spec-)" || \
    PROTECT_GLOB="$LANE_PROTECT_GLOB_FALLBACK"
```

No new `source` line: Delta 6 already added `lib_lane_state.sh` behind the
fail-loud `exit 2` guard. Four properties, none of them cosmetic:

- **`_lane- _spec-` are the bands this sweep OWNS** and must never come back
  protected — that would make gc skip every lane in both passes, so a python3
  outage would FREEZE reclaim instead of degrading it, re-creating the
  2026-07-10 ENOSPC accretion outage (`PROTECT_GLOB_OWNED_POOL_BANDS`).
- **The `||` is load-bearing.** gc.sh runs `set -euo pipefail` and
  `lane_protect_glob` fails loud with a non-zero return, so without the fallback
  a failed render would abort a timer-driven reclaim sweep. With it, the cost is
  one visible `[warn]` and a degrade to the machine-checked static backstop —
  never an empty glob, which downstream would read as "nothing is protected".
- **An explicit `--protect-glob` / `REIFY_WARM_LANE_GC_PROTECT_GLOB` still wins
  and short-circuits the render before it runs.** `--help` therefore still costs
  nothing (measured 0.03s, against 0.95s for a render), which is what keeps
  `orchestrator/tests/test_warm_lane_scripts_shipped.py` cheap.
- **The replacement comment does not restate the rendered contents.** That would
  re-create the same mirror one layer up. The three other doc sites that carried
  the literal (file header, `_usage()`, the env-knob list) moved with it,
  described by what the set *is*.

**`warm-lane-gc-sweep.sh` is deliberately NOT wired, and this is a verified
finding rather than an open question.** Re-confirmed by reading its whole arg
parser and its terminal `exec "$GC_SCRIPT" "${args[@]}"`, where `args` is
`(reclaim --mount "$MOUNT")` plus an optional `--disk-pressure`: it carries no
protect glob of its own, has no `--protect-glob` flag, and never had one — the
only `protect` tokens in the file are comments explaining why a *wrapper-side*
guard was removed in favour of the per-lane one in the primitive. It therefore
inherits gc.sh's resolution transitively and the wiring reaches it with zero
changes. Adding a render there would double the cost on the one production path
that runs from a systemd timer, for no behaviour change. `lib_lane_state.sh`'s
header previously named it as an intended caller; that has been corrected.

**Pinned by** Block X in `orchestrator/tests/warm-lane/test_warm_lane_gc.sh` —
X-band for the payoff (`REIFY_WARM_LANE_IACT_PREFIX` moves which band is
protected, asserted in both directions) and X-degrade for the `||`. Suite
wall-clock and the test-side cost mitigation: `orchestrator/tests/warm-lane/README.md`.

### Delta 10 — `warm-lane-gc.sh` bounds its ASSIGNED preserve, and `--disk-pressure` overrides it

**`warm-lane-gc.sh` diverges from reify a fourth time** (after Deltas 3, 6, 7
and 9). Added by **task 5504**. `lib_lane_state.sh` also gains a fourth
published global here; that library is dark-factory-native and so is not a
reify divergence at all, but it is recorded in the same place because the two
changes are one contract and a reader who meets either needs the other.

**The defect.** Delta 6's record gate `continue`d unconditionally on an
`assigned`/`in_use` record, with no upper bound on how old that record could
be, and it sits BEFORE the `--disk-pressure` branch. Composed with
`WarmLanePool._note_released_durable`, which swallows `OSError` by design
(release must always succeed — fail-open, invariant I3 — and a full disk must
not fail releases), that is a closed loop: an ENOSPC at release leaves
`"state": "assigned"` on disk while the in-memory pool reads FREE, and nothing
rewrites that record until the lane is acquired again. The emergency rm branch
was therefore unreachable for exactly the lanes a disk-full incident strands —
**the ENOSPC valve was held shut by the very failure it exists to respond to.**
A second producer needs no failure at all: a lane whose task is still
pending/in-progress/blocked is skipped BY DESIGN by
`harness.py::_reclaim_terminal_lane_records` (which keys on terminal task
status, not age), so its record ages indefinitely while
`_stale_lane_assignment_census` — the tree's only age bound on this field —
prints it in a digest and never acts.

**The fix.** The record keeps its verdict unless one of two **orthogonal**
reasons makes it non-decisive, OR'd at a single site
(`_record_gate_downgrade_reason`) rather than nested into the loop body:

- **ACUTE** — `--disk-pressure` downgrades the gate for the whole pass.
- **CHRONIC** — an `updated_at` older than `--max-record-age-days` (default 14)
  downgrades that one record, in every mode.

Four properties, none of them cosmetic:

- **"Downgrade" means FALL THROUGH, never "reclaim outright".** FD 8 stays open
  and there is no `continue`, so the lane still meets Pass 1's third gate,
  `live_ref_present`, inside the same critical section. The emergency trades an
  AUTHORITATIVE preserve for a PROBED one, not for none. A live build is still
  protected on both branches, the inv.2 flock is untouched, Pass 2's
  conservative `_is_reclaimable` rule is untouched, and `rm -rf <lane>/target`
  still touches only the checkout — committed work lives on
  `refs/heads/task/NNNN` (sizing-lifecycle T1) and `acquire_lane` always
  re-seeds (D10 §9.5). Reclaiming an assigned lane outright would re-open
  esc-5375-1, a live cargo build wiped mid-flight.
- **The two directions of failure are deliberately OPPOSITE.** An unreadable
  STATE fails OPEN (reclaim — failing closed on a corrupt record would freeze a
  lane out of reclaim forever); an unreadable AGE fails SAFE (preserve, with one
  attributable warn). The reads answer different questions: a readable
  `assigned` state is a trustworthy claim about STATE, so a bad timestamp leaves
  only the AGE unknown, and reclaiming on that would let a malformed field
  delete a live lane's build. That is verbatim the policy
  `harness.py::_stale_lane_assignment_census` already documents for the same
  field. The acute leg is evaluated FIRST and returns before the chronic one, so
  the valve is never subject to that fail-safe — otherwise the fail-safe would
  re-create the finding.
- **The default is 14 days and the relationship to `lane_stale_report_days`
  (7.0) is machine-checked, not documented-and-hoped.** Two full census periods,
  so the digest's `## Stale lane assignments` section surfaces a lane for a
  fortnight before the sweep acts on it — report-before-act. The two numbers
  cannot be kept in sync mechanically across the bash/pydantic boundary, which
  is the same hazard Delta 9 solved for `PROTECTED_PREFIXES`, so it takes the
  same remedy: `orchestrator/tests/test_harness_warm_lane_gc.py::TestGcRecordAgeBoundDoesNotUndercutTheCensus`
  asserts the ORDERING (`>=`, not equality — equality would forbid a deployment
  widening its gc margin for no reason). `0` is the documented escape hatch; a
  non-integer or negative value is a fatal usage error, because both silent
  directions are invisible in the summary line.

  **Report-before-act is a claim about the population the census COVERS, and
  that is narrower than the population this bound reaches.**
  `_stale_lane_assignment_census` skips any record whose backing task status is
  terminal or unknown (and excludes QUARANTINED), so the SECOND producer named
  under **The defect** above — a lane whose task is still
  pending/in-progress/blocked — is the one an operator sees in the digest for a
  fortnight first. The FIRST, the ENOSPC-stranded record, is by construction a
  record whose task has since FINISHED, so it is never censused at all; acting on it unreported is the INTENT, because nothing else
  in the tree ever acts on it. The safety property therefore holds exactly where
  destruction is riskiest — a lane whose task is still live — and does not
  overreach beyond it. Note also what the drift gate can and cannot see: it
  compares the pydantic field's DEFAULT against the bash default, so it pins the
  STOCK pair. A deployment that raises `lane_stale_report_days` past 14 in its
  own YAML is outside the gate's reach and loses the ordering it advertises.
- **The bound's own VALIDATION took two cuts, and the enumeration of what it
  rejects is the point.** The first cut was `^[0-9]+$`, copied from the sibling
  `--critical-free-gib` guard in `warm-lane-gc-sweep.sh`. It rejects junk and
  negatives and nothing else, and it carries a latent octal flaw that is inert
  THERE — a bogus free-space floor makes a sweep compare against the wrong
  number — but load-bearing HERE, because this value gates a destructive
  branch. Measured against the first cut: `08` was ACCEPTED by the regex and
  then read as octal by `$(( … * 86400 ))`, which dies with a raw
  `value too great for base` that `set -e` does not abort on, leaving
  `MAX_RECORD_AGE_SECS` unset and aborting the sweep mid-Pass-1 under `set -u`
  — exit 1, no summary line, and lanes visited earlier already reclaimed.
  `010` was worse because it was SILENT: accepted, and quietly meaning 8 days
  rather than 10. `00` was accepted as a second, undocumented spelling of the
  disable hatch — the valve off without anyone typing the documented `0`.
  `9223372036854775807` was accepted too, wrapping the multiply
  to `-86400`, which then failed the `-gt 0` test and disabled the valve while
  `downgraded_assigned=0` read as "nothing was stale". The shipped guard is
  `^(0|[1-9][0-9]{0,4})$`: `0|` preserves the escape hatch exactly, a leading
  `[1-9]` kills every leading-zero spelling, and the 5-digit cap makes 64-bit
  overflow UNREACHABLE (largest accepted product `99999 * 86400 = 8639913600`,
  ~9 orders of magnitude under 2^63; 99999 days is ~273 years, so the cap
  constrains no operator). **Do not "simplify" `{0,4}` back to `+`.**
  Rejection, not normalization: `$((10#$MAX_RECORD_AGE_DAYS))` would make `010`
  mean 10, but the value is interpolated verbatim into the operator-facing
  downgrade reason, so normalizing needs a second variable kept in sync or the
  message names a different number than the one applied. A redundant backstop
  after the multiply converts any residual arithmetic failure into exit 2 at
  the BOUNDARY, before a lane is touched. The error names
  `REIFY_WARM_LANE_GC_MAX_RECORD_AGE_DAYS` as well as the flag, because the env
  var is a real second door into the same guard.
- **`updated_at` reaches bash as a FOURTH global from the SAME single slurp**,
  not a second read and not the record file's mtime. `lib_lane_state.sh`'s
  header states why: the orchestrator rewrites these records on every acquire
  and release, so a second read is a DIFFERENT INSTANT and can report a pair of
  values that never coexisted. The existing `_lane_state_scalar` handles the
  field unchanged — it is a flat top-level quoted string, the exact shape that
  function documents, and its miss behaviour is already the desired
  "unjudgeable" reading.

**`warm-lane-gc-sweep.sh` is deliberately NOT wired, and this is a verified
finding rather than an open question.** It measures `df -B1 --output=avail`
once per sweep and APPENDS `--disk-pressure` to its argv whenever available
space is at or below the critical floor — so the one path that runs unattended
at true low water, which is exactly the window in which this composition bites,
inherits the valve with ZERO changes. A separate manual-only `--force` flag
would have left the finding open on that path and needed a second wiring change
to close it. Dark-factory's own ε path
(`git_ops.py::_run_warm_lane_gc_reclaim`) passes only
`reclaim --mount [--seed-script]` and never `--disk-pressure`, so the ACUTE
valve stays sweep-and-operator-only.

**The CHRONIC bound, by contrast, is ON BY DEFAULT on that same ε path**, and
the sentence above must not be read as covering it. `--max-record-age-days`
defaults to 14 with no wiring at all, so steady-state reclaim DOES now downgrade
an assigned/in_use record whose `updated_at` is older than a fortnight — into
the ordinary α reseed branch, never into the disk-pressure `rm`. That is the
intended reach: the chronic leg exists precisely for the records no incident is
ever going to arrive to clear. A reader asking "can this change affect
production reclaim?" should read the two legs separately — acute: only when
someone or the sweep passes the flag; chronic: every pass.

**Deliberately out of scope.** `_note_released_durable`'s swallowed `OSError`
stays: un-swallowing it would make a full disk fail releases, which is the
failure mode the fail-open exists to prevent, and the drift L2 at
`warm_lane_drift_l2_threshold` already makes it non-silent. This closes the
CONSEQUENCE (the held-shut valve), not the producer. No orchestrator config
knob was added — gc.sh is project-agnostic, and its knobs are flags plus
`REIFY_WARM_LANE_GC_*` env vars, per every sibling.

**Pinned by** Blocks S-pressure, S-age, S-age-degrade, A11 and A11-boundary in
`orchestrator/tests/warm-lane/test_warm_lane_gc.sh` — A11e-A11k pin the
validation matrix above (A11l is its non-vacuity control) and A11-boundary pins
that a misconfigured bound reclaims NOTHING rather than half a pool; the fourth global by
`orchestrator/tests/test_lane_state_lib.py::TestLaneStateReadPublishesUpdatedAt`;
the 14-vs-7.0 ordering by the drift gate named above. **Block K5 is the other
half of S-pressure's contract** — K5 pins that `--disk-pressure` still HONOURS
the live-reference gate, S-pressure that it DOWNGRADES the record gate, and the
downgrade is only safe because the gate K5 pins is still standing.

### Delta 11 — `warm-lane-degenerate-ref-check.sh` sources the citation grammar and guards it

**`warm-lane-degenerate-ref-check.sh` is no longer byte-identical to reify.**
Added by **task 5566**, which also vendored `lib_task_citation.sh` (reify
`d1fea3e2c0`) verbatim and re-synced the classifier itself to reify
`23e620a9b3` — the two Provenance rows carrying the partial-re-sync note.

**What prompted the port, and why it is not a cosmetic re-sync.** The script's
header claimed its citation predicate "mirrors dark-factory
`orchestrator/git_ops.py`'s citation regex byte-for-byte". That claim was
**false in both directions**:
`orchestrator/src/orchestrator/git_ops.py::DEFAULT_COMMIT_CITATION_PATTERN` has
**no `#<id>` arm at all**, and its conventional-commit alternative
(`^(merge|impl|amend|fix|…)(\(\b<id>\b[):]|.*\btask/<id>\b)`) — the form task
commits in both repos actually use — was **absent from the bash copy**. The
measured consequence, over reify's live pool on 2026-09-17 (esc-7244-16): **81
of 427 refs classified `degenerate` were tips whose own `kind(<id>): …` commit
was already on main**; every one flipped to `landed` under the ported grammar
and no `landed` ref flipped back. That matters because dark-factory reads
`degenerate` as "zero task work" and acts on it — `harness.py`'s MARK_DONE
recovery downgrades to a revert-and-redispatch, and
`git_ops.py::_abort_lane_acquisition` declines to preserve the branch.

**What the widening costs, stated asymmetrically — because it is asymmetric.**
It buys the 81 refs above; what it risks is a tip that cites task N without
being task N's work (the shape pinned as `test_warm_lane_degenerate_ref.sh`'s
K6). Both consumers act on `degenerate`, never on `landed`, so the cost of a
false `landed` is whatever the `degenerate` action would have been — and that
is **not** the same at the two sites:

| Site | Acts on `degenerate` by | Cost of a false `landed` |
|---|---|---|
| `git_ops.py::_abort_lane_acquisition` | `_delete_branch_if_on_main` | a retained, stale branch |
| `harness.py`'s `MARK_DONE_WITH_PROVENANCE` downgrade | revert-and-redispatch instead of marking done | **a phantom-done task** — precisely what that guard's own comment ("a degenerate branch carries ZERO task work, so MARK_DONE would phantom-complete a task that never actually landed anything") gives as its reason to exist |

At the harness site the only degeneracy-specific backstop is the independent
`_branch_is_degenerate(branch, metadata)` disjunct, and it is **fail-open**:
`orchestrator/src/orchestrator/landing_evidence.py::branch_is_degenerate`
returns `False` whenever `metadata['branch_base_sha']` is absent or is not a
40-hex sha. The `validate_landing_evidence` call below it does not bound this
error either — it checks that the candidate commit's effect survives at main
HEAD, and a foreign on-main tip passes that check by construction.

The port is still net-positive, on the measurement rather than on a symmetry
argument: a false `degenerate` costs re-dispatched landed work or a deleted
branch across every ref of the commonest shape there is (81 of 427), where a
false `landed` needs a foreign tip that names this task in a conventional-commit
subject. Do not restate the tradeoff as "`landed` is the conservative verdict at
both call sites" — an earlier draft of this delta's sibling prose did, and this
task's amendment pass corrected it here, in `tests/warm-lane/README.md` Delta 7
and in the K6 comment.

**Vendored, not inlined.** The grammar could have been inlined into its one
dark-factory consumer. It was not, for the reason stated at the top of
"Documented deltas": inlining would manufacture a fresh content divergence in
the very file this port exists to bring back into alignment. It would also be
unsafe across the seam — the PRD's cutover leaves (ζ/η, then κ deleting reify's
copies of the relocated seven) put reify's runtime on THIS copy, while reify's
`task-branch-contamination-sweep.sh` is not one of the seven and keeps sourcing
reify's own `lib_task_citation.sh`. An inlined dark-factory grammar would
therefore leave the grammar's two consumers holding two copies in two repos,
which is precisely what the lib was created to prevent.
`task_citation_peer_ids` therefore travels with **no dark-factory
consumer**, exactly as `lib_portable.sh`'s `allocate_free_port` and
`portable_timeout` already do; and **no separate lib test is ported**, as for
`lib_live_refs.sh` and `lib_portable.sh`, whose coverage arrives transitively
through the scripts that source them.

Those two facts compose into a third that is easy to miss, so state it
plainly: transitive coverage reaches only what the consumer calls. The
degenerate-ref suite exercises `task_citation_message_cites` (every
classification) and `task_citation_regex_escape` (the `--branch-prefix`
metacharacter block); `task_citation_peer_ids` — roughly a third of the file,
including its per-digit-suffix candidate enumeration and its SIGPIPE/`pipefail`
feeding idiom — is **unexercised here**, because nothing here calls it.
`test_warm_lane_bash_suite.py::test_every_invocable_script_has_ported_coverage`
names the split in its docstring rather than claiming whole-file coverage.

**What the vendored lib still says about reify, and why it was left saying it.**
`lib_task_citation.sh` is byte-identical to reify, header included, so it
carries three references that resolve only there: line 16's rule *"DO NOT
re-inline any ERE in a consumer … and `tests/infra/test_lib_task_citation.sh`
fails if one reappears"*, and lines 74 and 111 citing that same file's Blocks G
and F as the pins for the `pipefail` feeding idiom and the arbiter/harvest
set-equality contract. **In dark-factory that test does not exist and the rule
is documented but unenforced.** A review pass proposed annotating or trimming
line 16 in the file itself; that was declined, and the reasons are worth
keeping because the same fork will recur for the next vendored lib:

- The precedent is already set one file over — `lib_portable.sh` carries a
  dangling `tests/infra/test_run_gui_scripts.sh` reference and was vendored
  verbatim regardless.
- Every other delta in this list rides a file that had to change anyway, so it
  costs the diff nothing. A note here would be the **only** hunk in an
  otherwise byte-clean 171-line file, converting a free drift check into a
  non-zero one — for a file reify was still editing the day before this port.
- The one place a re-inline could actually happen is a **consumer**, and
  dark-factory's only consumer already says so in its own header: *"That lib is
  the SINGLE copy; dark-factory ports no separate lib test … so re-inlining
  either ERE here would reintroduce the drift the lib exists to prevent with
  nothing to catch it."* The misleading claim is contradicted at the point of
  use, which is the point that matters.

Manufacturing the missing enforcement with a grep-for-the-ERE test was
considered and rejected in the same pass: it would pin a spelling rather than
the property, and κ is the leaf that settles cross-repo enforcement.

**The three divergences from reify's post-7244 file**, filed as one delta
because they share a file, a task and a cause — Delta 7's own rule for not
splitting, whose same-task limb these satisfy and Delta 8's did not:

1. **A fail-loud `exit 2` guard on the new `source`.** reify does a bare
   `source`. Task 3370 (Delta 8) deliberately closed exactly that gap on
   `warm-lane-audit.sh`, because bash's own bare-`source` failure under
   `set -e` is **exit 1** — the code this directory's taxonomy assigns to a
   runtime error — where an absent sibling is *incomplete deployment: nothing
   about the invocation could have avoided it and no retry fixes it*. Shipping
   a new bare `source` here would reopen what 3370 closed. The guard reuses
   `warm-lane-audit.sh`'s message template and exit code verbatim, which is why
   `test_warm_lane_scripts_shipped.py`'s `FAIL_LOUD_FRAGMENTS` needed a
   one-line addition rather than a new spelling. The script's **own header
   exit-code table and `_usage()`'s `Exit codes:` line** were amended to make
   `2` read usage/WIRING and name the sibling — Delta 8 established that
   shipping a guard whose exit code the script's own table does not describe
   reproduces the very defect one layer up.
2. **A `dirname`-free `SCRIPT_DIR`.** reify's new line is
   `"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`. This file is the **sixth
   member of Delta 7's class** (see the pointer there); the full rationale
   lives in Delta 7 and is not restated here.
3. **Reify-only prose re-pointed.** reify's text names
   `tests/infra/test_lib_task_citation.sh` and
   `task-branch-contamination-sweep.sh`, neither of which exists in
   dark-factory, and its `# shellcheck source=` names reify's path. Leaving
   them would be a dangling cross-repo reference of the same kind this task was
   filed to remove.

   The re-pointed directive reads
   `# shellcheck source=orchestrator/scripts/warm-lane/lib_task_citation.sh`
   — **repo-root-relative, deliberately**, and it is the only `source=` in
   *this* directory that is. A review pass read that as a break from the five
   siblings' `scripts/<lib>.sh` spelling; measurement says otherwise on both
   halves. Those five are un-re-pointed reify text: `scripts/lib_portable.sh`,
   `scripts/lib_live_refs.sh` and `scripts/lib_lane_state.sh` **do not exist at
   this repo's root** (`ls` them — every one is a `No such file` here), so they
   name nothing. Every path dark-factory has actually re-pointed is
   repo-root-relative instead — five in the sibling test directory naming
   `orchestrator/tests/warm-lane/lib_warm_lane_paths.sh` and two naming
   `orchestrator/scripts/warm-lane/lib_lane_state.sh`. This directive follows
   *that* convention, not the one it is drifting away from. Note also that
   nothing validates any of them: shellcheck is not installed on the reference
   host and is invoked by no hook, script or CI job in this repo, so a `source=`
   here is a reader's hint and a provenance record, never a checked one.

The pre-existing references to `docs/design/warm-lane-degenerate-ref-seam.md`
(absent here) are **untouched** — they predate this task and sit in the
reify-verbatim body.

**Pinned by** the K1–K6 block in
`orchestrator/tests/warm-lane/test_warm_lane_degenerate_ref.sh` (assert floor
70 → 81, re-measured both sides), and by
`test_warm_lane_scripts_shipped.py::TestDegenerateRefCheckFailsLoudOnAMissingLibTaskCitation`
plus the script's membership in
`TestSiblingLibsTravelledWithTheScripts`'s parametrize list — the negative and
positive halves of the guard.

## Sibling-seed defaults, and who resolves them

Two of the relocated scripts default their `--seed-script` to a **sibling**
`seed-warm-lane.sh` — a file that, per §5 above, deliberately stayed in the
project. At the new location that sibling does not exist, so anything relying
on the default would fail. They are handled differently because their reach
differs, and neither is a policy change:

- **`warm-lane-gc.sh`** invokes `$SEED_SCRIPT` *unconditionally* on the Pass-1
  lane-reset path, so its default is on the hot path. dark-factory's caller
  (`GitOps._run_warm_lane_gc_reclaim`) therefore passes `--seed-script
  <project_root>/scripts/seed-warm-lane.sh` explicitly whenever that file
  exists. The script itself is left **verbatim**: the caller resolves the
  project-owned primitive, rather than dark-factory patching a default to
  guess at a project-owned path (PRD invariant C-1). Strictly no-op today —
  for reify the passed path is byte-identical to what the sibling default
  computes, and a project with no seed script gets no flag at all.

  **A project with no seed script gets a WARNING instead of a flag.** That is
  the fallback's own target case: a project carrying no warm-lane tooling is
  why these copies exist, and there is nothing to name. The sibling default
  then cannot resolve, so every non-disk-pressure Pass-1 lane reset fails
  inside the script and counts as *preserved* while reclaiming nothing.
  `_run_warm_lane_gc_reclaim` therefore logs, once per invocation and only
  when the resolved origin is `dark-factory`, a WARNING naming the missing
  `<project_root>/scripts/seed-warm-lane.sh` and stating that lane resets will
  fail — reclaim degrades to orphan removal plus disk-pressure target removal.
  Degraded is acceptable; degraded and silent is the accrete-to-ENOSPC failure
  the wrapper exists to prevent. The warning is NOT emitted for a
  project-origin gc copy: how a project arranges its own `SEED_SCRIPT` is its
  business.

- **`thin-warm-lane.sh`** has the same sibling default, but it is reachable
  ONLY under `--reseed`, which dark-factory never passes (PRD D3): the caller
  invokes it as `thin-warm-lane.sh <lane_dir>` and nothing else. It is
  therefore left verbatim with no caller-side wiring. **Any future caller that
  does pass `--reseed` must also pass `--seed-script`**, or it will resolve a
  sibling that is not there.
