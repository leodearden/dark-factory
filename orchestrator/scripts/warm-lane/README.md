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
of the contract above. The **behaviour** of every invocable script here is
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

The copy includes reify task 5572's per-lane live-consumer `/proc` check
(merged as reify `a4bddeaa51`), which is why `lib_live_refs.sh` travels here.

### Why these nine

PRD §2.1 audited each script for project-specific coupling. The token grep
`cargo|rustc|RUSTFLAGS|OUT_DIR|Cargo|nextest|occt|manifold|reify-gui|tauri`
across all nine yields exactly two hits, both in **comments**
(`warm-lane-gc.sh:165`, `lib_live_refs.sh:137`) — no code path branches on
anything reify-specific.

`lib_live_refs.sh` and `lib_portable.sh` are not among the seven scripts the
task named, but they are not optional: `warm-lane-gc.sh` and
`warm-lane-gc-sweep.sh` `source "$SCRIPT_DIR/lib_live_refs.sh"` and
deliberately `exit 2` when it is absent (reify 5572 made that fail-loud so a
silently-missing liveness guard cannot recur), and `warm-lane-audit.sh`
sources `$SCRIPT_DIR/lib_portable.sh`. Copying only seven would ship three
scripts that cannot execute. `orchestrator/tests/test_warm_lane_scripts_shipped.py`
pins this as executable behaviour.

### What deliberately did NOT move

`seed-warm-lane.sh` and `refresh-warm-base.sh` stay project-owned, behind the
PRD §5 contract: seeding a lane and refreshing the warm base are inherently
project-specific (what to build, what to prime), so dark-factory names them
rather than implements them.

## Documented deltas from the reify sources

Every file here is **byte-identical** to its reify source except as recorded
below. Keeping them diffable against reify is the cheap drift check available
for the whole α→κ duplication window, so deltas are enumerated rather than
absorbed. Policy is untouched: `REIFY_*` env-var names, default image/mount
paths and exit-code taxonomies are verbatim (renaming them is downstream
work — leaves β/γ/δ/ε — not this leaf), and no provenance header is prepended
to any script.

### Delta 1 — `provision-warm-lane-fs.sh`, `REPO_ROOT` resolution

**The only file-content divergence from reify in this directory** (together
with Delta 2 below, which is in the same file and the same concern:
path resolution at the new home).

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
