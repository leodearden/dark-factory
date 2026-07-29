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

*(none yet — filled in by later steps of task 3072)*

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

- **`thin-warm-lane.sh`** has the same sibling default, but it is reachable
  ONLY under `--reseed`, which dark-factory never passes (PRD D3): the caller
  invokes it as `thin-warm-lane.sh <lane_dir>` and nothing else. It is
  therefore left verbatim with no caller-side wiring. **Any future caller that
  does pass `--reseed` must also pass `--seed-script`**, or it will resolve a
  sibling that is not there.
