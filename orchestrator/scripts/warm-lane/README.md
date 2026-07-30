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
Delta 4; its protected-prefix half has no in-tree consumer until leaf γ — see
Delta 5.

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
  since task 3074 — `$SCRIPT_DIR/lib_lane_state.sh` behind a guard copied in
  shape from `warm-lane-gc.sh`'s. See Delta 4.

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
resolution at the new home). Delta 4 (leaf β) is the other file-content
divergence from reify in this directory.

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
content diverges. Both scripts default `--seed-script` to a sibling
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
  `.task-meta` — the live INV-5 drift this leaf exists to make un-writable, and
  which persists until γ deletes that default.

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
