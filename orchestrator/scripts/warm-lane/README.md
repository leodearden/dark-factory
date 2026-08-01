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
| `warm-lane-degenerate-ref-check.sh` | `fae3eda3cc` | 2026-07-05 |
| `provision-warm-lane-fs.sh` | `b37e00eaa6` | 2026-07-11 |
| `lib_live_refs.sh` | `434fd5a181` | 2026-07-28 |
| `lib_portable.sh` | `473217c346` | 2026-06-10 |
| `lib_lane_state.sh` | — **dark-factory-native** | 2026-07-30 |

The copy includes reify task 5572's per-lane live-consumer `/proc` check
(merged as reify `a4bddeaa51`), which is why `lib_live_refs.sh` travels here.

`lib_lane_state.sh` has **no reify commit and no reify source** — do not try to
diff it against one. It is dark-factory-native, added by **task 3074** (PRD leaf
β) as an *extraction*, not a copy: it holds the two facts dark-factory owns and
reify does not (the `.lane-state` record format, whose `state` values are the
`LaneState` enum in `orchestrator/src/orchestrator/lane_lifecycle.py`; and
`PROTECTED_PREFIXES` in `orchestrator/src/orchestrator/git_ops.py`). Its
lane-state half was lifted out of `warm-lane-audit.sh`'s private reader — see
Delta 4; its protected-prefix half still has no in-tree consumer (leaf γ wired
the lane-state half only — see Delta 5 for the measured reason and Delta 6 for
what γ did land).

### Why these ten

PRD §2.1 audited each script for project-specific coupling. The token grep
`cargo|rustc|RUSTFLAGS|OUT_DIR|Cargo|nextest|occt|manifold|reify-gui|tauri`
across the nine **relocated** files yields exactly two hits, both in
**comments** (`warm-lane-gc.sh:165`, `lib_live_refs.sh:137`) — no code path
branches on anything reify-specific. (The tenth, `lib_lane_state.sh`, is
dark-factory-native and deliberately reify-free; the coupling it carries runs
the other way, to `orchestrator/src/orchestrator/`.)

Three of the ten source a sibling lib, and none of those libs is among the
seven scripts the task named — copying only seven would ship three scripts that
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

`orchestrator/tests/test_warm_lane_scripts_shipped.py` pins this as executable
behaviour: every file above is checked for presence, the owner-execute bit and
`bash -n`, and each of the three sourcing scripts is run with `--help` from
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

**`warm-lane-gc.sh` is deliberately untouched by leaf β.** Its hand-maintained
`PROTECT_GLOB` default still carries the comment admitting it mirrors
dark-factory's `PROTECTED_PREFIXES` across a repo boundary. β ships what
replaces it — `lane_protect_glob` and the machine-checked
`LANE_PROTECT_GLOB_FALLBACK` — but **leaf γ** owns that file and does the
rewire, so editing it here would collide with γ's scope for no gain. Note β's
renderer excludes the bands a pool sweep OWNS (`_lane-`, `_spec-`): handing
those to gc as *protected* would make it skip every lane in both passes and stop
reclaim outright.

**`lane_protect_glob` therefore has NO in-tree consumer yet.** It and
`LANE_PROTECT_GLOB_FALLBACK` ship ahead of the caller that uses them; today only
`orchestrator/tests/test_lane_state_lib.py` exercises them. `warm-lane-audit.sh`
sources the lib but uses only its lane-state half. Two consequences for whoever
picks up γ, neither of them validated in situ here:

- The `glob="$(lane_protect_glob …)" || glob="$LANE_PROTECT_GLOB_FALLBACK"`
  contract is a *contract*, not an observed behaviour — no shipped caller
  exercises the `||` yet.
- Whatever names the deployment's interactive band must reach the bridge in
  `REIFY_WARM_LANE_IACT_PREFIX` (below). `warm-lane-gc.sh`'s current
  hand-maintained default hardcodes `_iact-*`, and also omits `.lane-state` and
  `.task-meta` — the live INV-5 drift this leaf exists to make un-writable.

  **γ did NOT delete that default** (task 3075; this claim previously read
  "persists until γ deletes that default" and is corrected here rather than
  left false). It was deferred with the cost measured, not overlooked:

  - The rendered glob
    `_merge-*,_solo-*,_substrate-gate-*,_merge-verify,_offline-deep,.lane-state,.task-meta,_mainprobe-*,_mainsweep-*,_iact-*`
    is behaviourally **indistinguishable** from gc.sh's current default under
    any black-box test: `_merge-verify` is already covered by `_merge-*`, and
    `.lane-state`/`.task-meta` are dot-prefixed while gc.sh's candidate loop is
    `"$WORKTREES_DIR"/*/` with no `shopt -s dotglob` anywhere. The sole
    observable payoff is honouring `REIFY_WARM_LANE_IACT_PREFIX`.
  - Measured on this host 2026-07-30, the python bridge costs 1.05s / 4.45s /
    3.48s across three consecutive runs. Negligible against a 25-40 minute
    production reclaim pass, but it fires once per reclaim **invocation** and
    the gc bash suite invokes reclaim roughly 30 times — it would materially
    degrade the very suite γ's core change depends on for verification.

  The wiring wants a memoization or an opt-out designed alongside it, so it is
  filed as a follow-up rather than bundled onto the leaf that closes
  esc-5334-6. Until it lands, `lane_protect_glob` remains a shipped function
  with no in-tree consumer.

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
  resolved** (a decoy-CWD marker), not on the exit code, for the reason above:
  the exit code is a function of the caller's CWD, not of the defect. Each of
  its absence assertions is paired with a `Usage:` **positive control**: an
  absence assertion alone is satisfied by a script that died at line 1 for an
  unrelated reason, so without it the case could go green having never reached
  the resolution it names.
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
