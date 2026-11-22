# Capability manifest — git-ops-split-prd.md

Human-readable twin of `plans/git-ops-split-prd.capability-manifest.yaml`, the sidecar `commit_planning` stamps. Bindings verified at main `a4349b7f4b` (2026-11-22): every producer `present` check is absent at that tree and every producer `absent` check is present there, run through `shared/src/shared/delivered_check_polarity.py::lint_delivered_checks`. No substrate check is carried: every substrate the leaves rely on is either on main today (a `grep:` pointer, checked with `git cat-file -e` and the name found in `git show main:<path>`) or produced by S5 (`producer:task-N upstream`, each inside α → 9275's transitive closure).

Check design. S5's module filenames are pinned by S5's own sidecar, so they are safe to name; every new module, class and file name comes from this PRD's §Contract → Module ownership. A leaf's "added to `CLUSTER_PATHS`" is checked as a tool-written files entry in `orchestrator/tests/merge_lane_ratchet_baseline.json`, which proves both the literal and the regenerated measurement. Absent checks are scoped to the whole package minus the new home where S5 may have moved the symbol, so an S5 rename can only make them pass early, never wedge. Conditional modules (`lane_paths`, the D12 fallbacks, `task_worktrees`) and every line-count signal are `manual`, and no present-check names a conditional module. `signal_leaves: [ι, ω]`: ι is the decomposition plan's named signal leaf and integration gate, and ω delivers §Goal item 1 and the D4 half of item 3, so the dashboard reads partial after ι and delivered after ω.

Durability. 6698 (the trigger-chain gate) becomes a dependent of every `high` leaf (β, δ, ζ, ι, ω), so those leaves' checks are evaluated again after ω; each was chosen to stay true through the rest of the chain.

## α — Re-measure the git engine after S5 lands (dated Correction + post-S5 partition)

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `s5-layout-landed` | producer:task-9275 upstream — S5 π gates every S5 git-engine module (its transitive closure holds 9255–9274); α verifies each exists after π, so α carries no check of its own on them | — (upstream / substrate; no check) | PASS |
| `concern-map-instrument` | `grep:scripts/concern_map.py` `grep:scripts/source_measures.py::prose_line_numbers_in_tree` the instrument α re-runs (D2), on main since task 7140 | — (upstream / substrate; no check) | PASS |
| `post-s5-partition-committed` | built here — the PRD fixes the path plans/git-ops-split-prd.partition-post-s5.json; absent at a4349b7f4b | path `plans/git-ops-split-prd.partition-post-s5.json` present | PASS |
| `dated-correction-committed` | built here — a dated Correction appended to the PRD with the as-of SHA | manual | PASS |
| `post-s5-partition-acyclic-or-cycles-named` | built here — concern_map.py --partition on the post-S5 partition exits 0 with "Cyclic cluster groups: none", or the Correction names each cycle and its leaf | manual | PASS |

## β — The ratchet measures the git engine (every S5 git-engine module in CLUSTER_PATHS)

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `ratchet-substrate` | `grep:scripts/merge_lane_cluster.py::CLUSTER_PATHS` `grep:scripts/merge_lane_metrics.py::main` `grep:orchestrator/tests/merge_lane_ratchet_raises` the cluster spec, the --report / --write-baseline --authorize-raise command line and the one-file-per-raise ledger | — (upstream / substrate; no check) | PASS |
| `s5-modules-exist` | producer:task-9255 upstream producer:task-9264 upstream producer:task-9260 upstream producer:task-9266 upstream producer:task-9259 upstream producer:task-9272 upstream producer:task-9267 upstream producer:task-9268 upstream producer:task-9270 upstream — each in α→9275's transitive closure; S5's own sidecar pins each filename | — (upstream / substrate; no check) | PASS |
| `ratchet-measures-git-run` | built here — β adds orchestrator/src/orchestrator/git_run.py (S5, task 9255) to CLUSTER_PATHS as a literal and regenerates the baseline; a tool-written files entry is the measured form of "--report lists it"; no entry at a4349b7f4b | `"orchestrator/src/orchestrator/git_run\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `ratchet-measures-git-diff` | built here — β adds orchestrator/src/orchestrator/git_diff.py (S5, task 9264) to CLUSTER_PATHS as a literal and regenerates the baseline; a tool-written files entry is the measured form of "--report lists it"; no entry at a4349b7f4b | `"orchestrator/src/orchestrator/git_diff\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `ratchet-measures-worktree-list` | built here — β adds orchestrator/src/orchestrator/worktree_list.py (S5, task 9264) to CLUSTER_PATHS as a literal and regenerates the baseline; a tool-written files entry is the measured form of "--report lists it"; no entry at a4349b7f4b | `"orchestrator/src/orchestrator/worktree_list\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `ratchet-measures-block-disposition` | built here — β adds orchestrator/src/orchestrator/block_disposition.py (S5, task 9260) to CLUSTER_PATHS as a literal and regenerates the baseline; a tool-written files entry is the measured form of "--report lists it"; no entry at a4349b7f4b | `"orchestrator/src/orchestrator/block_disposition\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `ratchet-measures-lane-lease` | built here — β adds orchestrator/src/orchestrator/lane_lease.py (S5, task 9266) to CLUSTER_PATHS as a literal and regenerates the baseline; a tool-written files entry is the measured form of "--report lists it"; no entry at a4349b7f4b | `"orchestrator/src/orchestrator/lane_lease\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `ratchet-measures-warm-lane-scripts` | built here — β adds orchestrator/src/orchestrator/warm_lane_scripts.py (S5, task 9259) to CLUSTER_PATHS as a literal and regenerates the baseline; a tool-written files entry is the measured form of "--report lists it"; no entry at a4349b7f4b | `"orchestrator/src/orchestrator/warm_lane_scripts\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `ratchet-measures-git-queries` | built here — β adds orchestrator/src/orchestrator/git_queries.py (S5, task 9272) to CLUSTER_PATHS as a literal and regenerates the baseline; a tool-written files entry is the measured form of "--report lists it"; no entry at a4349b7f4b | `"orchestrator/src/orchestrator/git_queries\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `ratchet-measures-task-branch` | built here — β adds orchestrator/src/orchestrator/task_branch.py (S5, task 9267) to CLUSTER_PATHS as a literal and regenerates the baseline; a tool-written files entry is the measured form of "--report lists it"; no entry at a4349b7f4b | `"orchestrator/src/orchestrator/task_branch\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `ratchet-measures-worktrees` | built here — β adds orchestrator/src/orchestrator/worktrees.py (S5, task 9268) to CLUSTER_PATHS as a literal and regenerates the baseline; a tool-written files entry is the measured form of "--report lists it"; no entry at a4349b7f4b | `"orchestrator/src/orchestrator/worktrees\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `ratchet-measures-main-landing` | built here — β adds orchestrator/src/orchestrator/main_landing.py (S5, task 9270) to CLUSTER_PATHS as a literal and regenerates the baseline; a tool-written files entry is the measured form of "--report lists it"; no entry at a4349b7f4b | `"orchestrator/src/orchestrator/main_landing\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `raise-entry-names-over-ceiling-modules` | built here — one authorized-raise entry naming every S5 module whose measured --report lines exceed 1,500, with its lines and owning leaf η/θ | manual | PASS |

## γ — Prose to its home (docs/git-engine/decisions.md; archaeology_blocks)

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `decisions-doc` | built here — new docs/git-engine/decisions.md (D5); absent at a4349b7f4b | path `docs/git-engine/decisions.md` present | PASS |
| `archaeology-measure-defined` | built here, or by task 5047 if it lands first (D5) — the archaeology_blocks measure in the scripts instrument; no occurrence under scripts/ at a4349b7f4b | `archaeology_blocks` present in `scripts/` | PASS |
| `archaeology-zero-per-engine-module` | built here — --report reads archaeology_blocks = 0 for every git-engine module | manual | PASS |
| `one-entry-per-relocated-block` | built here — one dated entry per rehomed rationale citing its id; one-line pointer in code; docstrings keep only the contract | manual | PASS |

## δ0 — codemod sibling of δ (complexity simple)

Added at decompose (2026-11-22): δ's caller set is 80 test files on main a4349b7f4b, over the PRD's ~40 rule, so δ is filed as this codemod sibling plus the move leaf (S5 ξ0's shape). δ0 creates the three modules with the moved types and one-step surfaces whose methods call the not-yet-moved GitOps bodies through a reference typed by a local Protocol (no import of git_ops), composes them on GitOps, repoints every caller and test, and adds the modules to CLUSTER_PATHS. The rows below moved here from δ unchanged.

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `lane-outcomes-owns-warm-lane-unavailable` | built here — δ's named signal: the enum is defined in lane_outcomes.py; lane_outcomes.py does not exist at a4349b7f4b | `^class WarmLaneUnavailable\b` present in `orchestrator/src/orchestrator/lane_outcomes.py` | PASS |
| `outcome-types-have-one-home` | built here — WarmLaneUnavailable, WarmLaneRequeue and its 8 subclasses, WorktreeInfo and PoolPrewarmResult are defined in git_ops.py at a4349b7f4b; after δ nowhere in the package but lane_outcomes.py (no re-export) | `^class (WarmLaneUnavailable\|WarmLaneRequeue\|WorktreeInfo\|PoolPrewarmResult)\b\|^class WarmLane[A-Za-z]+\(WarmLaneRequeue\)` absent in `orchestrator/src/orchestrator/`, `:(exclude)orchestrator/src/orchestrator/lane_outcomes.py` | PASS |
| `pool-storage-guard-class` | built here — the .pool-root guard owner (§Contract); pool_storage.py does not exist at a4349b7f4b | `^class PoolStorageGuard\b` present in `orchestrator/src/orchestrator/pool_storage.py` | PASS |
| `guarded-prune-public-seam` | built here — PoolStorageGuard.prune_registrations(context); pool_storage.py does not exist at a4349b7f4b | `def prune_registrations\(` present in `orchestrator/src/orchestrator/pool_storage.py` | PASS |
| `lane-seeder-class` | built here — seed + from_seed_rc + mark_present, refresh_warm_base (§Contract); lane_seeding.py does not exist at a4349b7f4b | `^class LaneSeeder\b` present in `orchestrator/src/orchestrator/lane_seeding.py` | PASS |
| `ratchet-measures-lane-outcomes` | built here — δ adds orchestrator/src/orchestrator/lane_outcomes.py to CLUSTER_PATHS and regenerates the baseline (D4); no baseline entry names it at a4349b7f4b | `"orchestrator/src/orchestrator/lane_outcomes\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `ratchet-measures-pool-storage` | built here — δ adds orchestrator/src/orchestrator/pool_storage.py to CLUSTER_PATHS and regenerates the baseline (D4); no baseline entry names it at a4349b7f4b | `"orchestrator/src/orchestrator/pool_storage\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `ratchet-measures-lane-seeding` | built here — δ adds orchestrator/src/orchestrator/lane_seeding.py to CLUSTER_PATHS and regenerates the baseline (D4); no baseline entry names it at a4349b7f4b | `"orchestrator/src/orchestrator/lane_seeding\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `callers-repointed-off-seed-warm-lane` | built here — δ0 repoints every test caller of GitOps._seed_warm_lane to LaneSeeder.seed (8 test files at a4349b7f4b; added at decompose for critic finding F14) | `\._seed_warm_lane\(` absent in the test trees | PASS |

## δ — The low residual modules: lane_outcomes, pool_storage, lane_seeding (and lane_paths)

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `warm-lane-unavailable-member-data` | producer:task-9246 upstream producer:task-9260 upstream — S9 λ1 edits the member data and S5 δ adds requeue class / disposition / seed rc before δ moves the type | — (upstream / substrate; no check) | PASS |
| `private-prune-retired` | built here — GitOps._prune_registrations is defined at a4349b7f4b; S5 modules reach the guard by a direct downward import, not a GitOps forwarder (D7) | `def _prune_registrations\(` absent in `orchestrator/src/orchestrator/` | PASS |
| `one-guarded-prune` | built here — INV-5 / heuristic 11: the one git worktree prune argv in the engine lives in pool_storage.py; at a4349b7f4b it is in git_ops.py (code and its docstring) | `['"]worktree['"], *['"]prune['"]` absent in `orchestrator/src/orchestrator/`, `:(exclude)orchestrator/src/orchestrator/pool_storage.py` | PASS |
| `seed-private-method-retired` | built here — GitOps._seed_warm_lane is defined at a4349b7f4b; it becomes LaneSeeder.seed | `def _seed_warm_lane\(` absent in `orchestrator/src/orchestrator/` | PASS |
| `lane-paths-conditional` | built here only if α records that S5 left the persistent-lane path facts unplaced or in an upper module (R6, open question 3) | manual | OPEN |
| `b5-b6-b8-b12-green` | built here — as of δ's chain point, in the new modules' own test files: the guarded prune from every caller module (main_landing.py's merge-worktree cleanup standing in for merge_worktrees, which θ creates), the pool-less host, seed rc mapping, the unchanged disposition table; the final-layout form is ι's B5 | manual | PASS |

## ε — Lane reset, spec lanes and admission

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `spec-lanes-class` | built here — SpecLanes.acquire / .release; spec_lanes.py does not exist at a4349b7f4b | `^class SpecLanes\b` present in `orchestrator/src/orchestrator/spec_lanes.py` | PASS |
| `lane-reset-class` | built here — reset, reseed, recycle, abort, prewarm; lane_reset.py does not exist at a4349b7f4b | `^class LaneReset\b` present in `orchestrator/src/orchestrator/lane_reset.py` | PASS |
| `lane-admission-class` | built here — disk / soft-pressure admission, gc reclaim, census; lane_admission.py does not exist at a4349b7f4b | `^class LaneAdmission\b` present in `orchestrator/src/orchestrator/lane_admission.py` | PASS |
| `spec-lane-methods-renamed` | built here — GitOps.acquire_spec_lane / release_spec_lane are defined at a4349b7f4b; §Contract renames them SpecLanes.acquire / .release with no alias | `def (acquire\|release)_spec_lane\(` absent in `orchestrator/src/orchestrator/` | PASS |
| `audit-cache-owned-by-admission` | built here — heuristic 5: self._warm_lane_audit_cache is set and read in git_ops.py at a4349b7f4b; after ε only LaneAdmission holds it | `self\._warm_lane_audit_cache\b` absent in `orchestrator/src/orchestrator/`, `:(exclude)orchestrator/src/orchestrator/lane_admission.py` | PASS |
| `reset-bodies-left-git-ops` | built here — GitOps.prewarm_pool and _reseed_verified_clean are defined in git_ops.py at a4349b7f4b | `def (prewarm_pool\|_reseed_verified_clean)\(` absent in `orchestrator/src/orchestrator/git_ops.py` | PASS |
| `ratchet-measures-spec-lanes` | built here — ε adds orchestrator/src/orchestrator/spec_lanes.py to CLUSTER_PATHS and regenerates the baseline (D4); no baseline entry names it at a4349b7f4b | `"orchestrator/src/orchestrator/spec_lanes\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `ratchet-measures-lane-reset` | built here — ε adds orchestrator/src/orchestrator/lane_reset.py to CLUSTER_PATHS and regenerates the baseline (D4); no baseline entry names it at a4349b7f4b | `"orchestrator/src/orchestrator/lane_reset\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `ratchet-measures-lane-admission` | built here — ε adds orchestrator/src/orchestrator/lane_admission.py to CLUSTER_PATHS and regenerates the baseline (D4); no baseline entry names it at a4349b7f4b | `"orchestrator/src/orchestrator/lane_admission\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `b9-green` | built here — spec lane round trip through git_ops.spec_lanes, in test_spec_lanes.py; the cold fallback goes through S5's MainLanding (git_ops.landing) until θ0 introduces MergeWorktrees; the final-layout form is ι's B9 | manual | PASS |

## ζ0 — codemod sibling of ζ (complexity simple)

Added at decompose (2026-11-22): ζ's caller set is 53 test files on main a4349b7f4b, over the ~40 rule, so ζ is filed as this codemod sibling plus the move leaf. ζ0 creates lane_release.py and warm_acquire.py with one-step surfaces, composes them on GitOps, repoints callers and tests, and adds both to CLUSTER_PATHS. The rows below moved here from ζ unchanged.

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `lane-release-class` | built here — release, thin, terminal release, detach, cleanup_worktree, prior incarnation; lane_release.py does not exist at a4349b7f4b | `^class LaneRelease\b` present in `orchestrator/src/orchestrator/lane_release.py` | PASS |
| `warm-acquirer-class` | built here — S5 κ's _ACQUIRE_ROUTES owner; warm_acquire.py does not exist at a4349b7f4b | `^class WarmLaneAcquirer\b` present in `orchestrator/src/orchestrator/warm_acquire.py` | PASS |
| `ratchet-measures-lane-release` | built here — ζ adds orchestrator/src/orchestrator/lane_release.py to CLUSTER_PATHS and regenerates the baseline (D4); no baseline entry names it at a4349b7f4b | `"orchestrator/src/orchestrator/lane_release\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `ratchet-measures-warm-acquire` | built here — ζ adds orchestrator/src/orchestrator/warm_acquire.py to CLUSTER_PATHS and regenerates the baseline (D4); no baseline entry names it at a4349b7f4b | `"orchestrator/src/orchestrator/warm_acquire\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `callers-repointed-off-gitops-acquire-release` | built here — ζ0 repoints callers of acquire_warm_lane / release_warm_lane / release_lane_for_terminal_task (added at decompose for F14) | `\.(acquire_warm_lane\|release_lane_for_terminal_task\|release_warm_lane)\(` absent outside git_ops.py, the two new modules and config.py | PASS |

## ζ — Lane release, warm acquire, and GitOps as the composition root

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `acquire-route-table` | producer:task-9262 upstream — S5 κ's AcquireRoute table and route methods, which WarmLaneAcquirer takes over | — (upstream / substrate; no check) | PASS |
| `exhaustion-counter-single-owner` | built here — B7: self._consecutive_exhausted is set in git_ops.py at a4349b7f4b; after ζ only WarmLaneAcquirer holds it | `self\._consecutive_exhausted\b` absent in `orchestrator/src/orchestrator/`, `:(exclude)orchestrator/src/orchestrator/warm_acquire.py` | PASS |
| `release-bodies-left-git-ops` | built here — release_warm_lane, thin_warm_lane, detach_lane_checkout, cleanup_worktree and _note_structural_exhaustion are GitOps methods at a4349b7f4b; B11 forbids a delegating forwarder left behind | `def (release_warm_lane\|thin_warm_lane\|detach_lane_checkout\|cleanup_worktree\|_note_structural_exhaustion)\(` absent in `orchestrator/src/orchestrator/git_ops.py` | PASS |
| `git-ops-baseline-under-ceiling` | built here — ζ's regenerated baseline records git_ops.py at its measured size, at most 1,500 lines (17,010 at a4349b7f4b); the committed proxy of "git_ops.py is ≤ 1,500" | `"orchestrator/src/orchestrator/git_ops\.py": *\{[^}]*"lines": *(1(50[1-9]\|5[1-9][0-9]\|[6-9][0-9]{2})\|[2-9][0-9]{3}\|[1-9][0-9]{4,})[,}]` absent in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `task-worktrees-fallback` | built here only if ζ measures git_ops.py over 1,465 lines (open question 1) | manual | OPEN |
| `b7-b10-b11-green` | built here — routes through the composition, terminal release and prior incarnation (parked through S5's WorktreeOps rename/quarantine until η introduces WorktreeRelocation), no delegating façade, in the new modules' own test files; the final-layout forms are ι's B7/B10/B11 | manual | PASS |

## η — Sub-split worktrees, git_queries and task_branch

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `worktree-relocation-class` | built here — rename and quarantine relocation (D8); worktree_relocation.py does not exist at a4349b7f4b | `^class WorktreeRelocation\b` present in `orchestrator/src/orchestrator/worktree_relocation.py` | PASS |
| `commit-effects-class` | built here — describe_in_main / present_in_main (D8); commit_effect.py does not exist at a4349b7f4b | `^class CommitEffects\b` present in `orchestrator/src/orchestrator/commit_effect.py` | PASS |
| `commit-effect-probe-single-home` | built here — class CommitEffectProbe is defined in git_ops.py at a4349b7f4b; after η only in commit_effect.py | `^class CommitEffectProbe\b` absent in `orchestrator/src/orchestrator/`, `:(exclude)orchestrator/src/orchestrator/commit_effect.py` | PASS |
| `moved-methods-renamed` | built here — rename_worktree, quarantine_worktree, describe_commit_effect_in_main and commit_effect_present_in_main are GitOps methods at a4349b7f4b; §Contract renames them WorktreeRelocation.rename/.quarantine and CommitEffects.describe_in_main/.present_in_main with no alias | `def (rename_worktree\|quarantine_worktree\|describe_commit_effect_in_main\|commit_effect_present_in_main)\(` absent in `orchestrator/src/orchestrator/` | PASS |
| `ratchet-measures-worktree-relocation` | built here — η adds orchestrator/src/orchestrator/worktree_relocation.py to CLUSTER_PATHS and regenerates the baseline (D4); no baseline entry names it at a4349b7f4b | `"orchestrator/src/orchestrator/worktree_relocation\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `ratchet-measures-commit-effect` | built here — η adds orchestrator/src/orchestrator/commit_effect.py to CLUSTER_PATHS and regenerates the baseline (D4); no baseline entry names it at a4349b7f4b | `"orchestrator/src/orchestrator/commit_effect\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `d12-fallbacks-git-citations-merge-markers` | built here only if git_queries.py / task_branch.py measure over 1,465 after γ (D12) | manual | OPEN |
| `eta-modules-under-headroom` | built here — each η module ≤ 1,465 lines in --report, or its D12 fallback applied | manual | PASS |

## θ0 — codemod sibling of θ (complexity simple)

Added at decompose (2026-11-22): θ's caller set is 94 test files on main a4349b7f4b (before S5 ο0 repoints them to git_ops.landing.*; θ0 re-enumerates after S5), over the ~40 rule, so θ is filed as this codemod sibling plus the move leaf. θ0 creates merge_worktrees.py, merging.py and main_park.py with the moved types and one-step surfaces, composes them on GitOps, repoints callers and tests, and adds the modules to CLUSTER_PATHS. The rows below moved here from θ unchanged.

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `merge-worktrees-class` | built here — merge / host-verify / offline-deep worktree lifecycle (a test double of the same name in orchestrator/tests is outside the scope); merge_worktrees.py does not exist at a4349b7f4b | `^class MergeWorktrees\b` present in `orchestrator/src/orchestrator/merge_worktrees.py` | PASS |
| `merging-class` | built here — merge_to_main, merge_branch_into_worktree; merging.py does not exist at a4349b7f4b | `^class Merging\b` present in `orchestrator/src/orchestrator/merging.py` | PASS |
| `main-park-class` | built here — the park and index-lock sink; main_park.py does not exist at a4349b7f4b | `^class MainPark\b` present in `orchestrator/src/orchestrator/main_park.py` | PASS |
| `merge-to-main-on-merging` | built here — B14: git_ops.merging.merge_to_main; merging.py does not exist at a4349b7f4b | `async def merge_to_main\(` present in `orchestrator/src/orchestrator/merging.py` | PASS |
| `park-sink-single-home` | built here — MERGE_PARK_REF and the MergePark* errors are defined in git_ops.py at a4349b7f4b; after θ only in main_park.py | `^(MERGE_PARK_REF\b\|class MergePark[A-Za-z]*\b)` absent in `orchestrator/src/orchestrator/`, `:(exclude)orchestrator/src/orchestrator/main_park.py` | PASS |
| `ratchet-measures-merge-worktrees` | built here — θ adds orchestrator/src/orchestrator/merge_worktrees.py to CLUSTER_PATHS and regenerates the baseline (D4); no baseline entry names it at a4349b7f4b | `"orchestrator/src/orchestrator/merge_worktrees\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `ratchet-measures-merging` | built here — θ adds orchestrator/src/orchestrator/merging.py to CLUSTER_PATHS and regenerates the baseline (D4); no baseline entry names it at a4349b7f4b | `"orchestrator/src/orchestrator/merging\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `ratchet-measures-main-park` | built here — θ adds orchestrator/src/orchestrator/main_park.py to CLUSTER_PATHS and regenerates the baseline (D4); no baseline entry names it at a4349b7f4b | `"orchestrator/src/orchestrator/main_park\.py": *\{` present in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |

## θ — Sub-split main_landing (merge_worktrees, merging, main_park)

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `advance-main-decomposed` | producer:task-5042 upstream — 5042 owns advance_main and decomposes it before θ (D9) | — (upstream / substrate; no check) | PASS |
| `merge-to-main-left-landing` | built here — θ's named signal; at a4349b7f4b the method is in git_ops.py (S5 ο moves it to main_landing.py). Scoped to the two files because a reviewer-eval corpus diff in the package quotes the def line forever | `async def merge_to_main\(` absent in `orchestrator/src/orchestrator/main_landing.py`, `orchestrator/src/orchestrator/git_ops.py` | PASS |
| `d12-fallback-interactive-reaper` | built here only if merge_worktrees.py measures over 1,465 after γ (D12) | manual | OPEN |
| `theta-modules-under-headroom` | built here — main_landing, main_park, merging, merge_worktrees each ≤ 1,465 in --report, or the D12 fallback applied; B14 green | manual | PASS |

## ι — Integration gate: test_git_engine_boundary_gate.py (B1–B14 except B4)

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `s5-gate-suite` | producer:task-9275 upstream — S5 π's test_gitops_boundary_gate.py and its ScriptedGitRunner / WarmLaneScripts fakes, which B13 and the fixtures reuse | — (upstream / substrate; no check) | PASS |
| `boundary-gate-file` | built here — the PRD fixes the path; absent at a4349b7f4b | path `orchestrator/tests/test_git_engine_boundary_gate.py` present | PASS |
| `engine-baseline-entries-under-ceiling` | built here (with η and θ) — B3 and §Goal item 3: no git-engine entry in the baseline records more than 1,500 lines; at a4349b7f4b git_ops.py records 17,010. Carried on ι so ω cannot dispatch before it holds | `"orchestrator/src/orchestrator/(git_ops\|git_run\|git_diff\|worktree_list\|block_disposition\|lane_lease\|warm_lane_scripts\|git_queries\|task_branch\|worktrees\|main_landing\|lane_paths\|lane_outcomes\|pool_storage\|lane_seeding\|spec_lanes\|lane_reset\|lane_admission\|lane_release\|warm_acquire\|worktree_relocation\|commit_effect\|merge_worktrees\|merging\|main_park\|git_citations\|merge_markers\|interactive_reaper\|task_worktrees)\.py": *\{[^}]*"lines": *(1(50[1-9]\|5[1-9][0-9]\|[6-9][0-9]{2})\|[2-9][0-9]{3}\|[1-9][0-9]{4,})[,}]` absent in `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS |
| `gate-green-and-measurement-recorded` | built here — the gate is green; the commit message records the re-measured concern map and each new module's --report row | manual | PASS |
| `b2-importers-repointed` | built here — from orchestrator.git_ops import matches only GitOps (and S5's remaining names, if α records any) | manual | PASS |

## ω — Retire SIZE_CEILING_EXEMPT (D4 grandfather rule, D10 pins)

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `exemption-sites-on-main` | `grep:scripts/merge_lane_cluster.py::SIZE_CEILING_EXEMPT` `grep:scripts/merge_lane_ratchet.py::_check_ceilings` `grep:scripts/tests/test_scripts_governed_lists.py::SIZE_CEILING_EXEMPT_LIST` `grep:orchestrator/tests/_merge_lane_ratchet_fixtures.py` the declaration, its one consumer, the governed-list owner row and the GIT_OPS_RELPATH comment ω deletes or rewrites | — (upstream / substrate; no check) | PASS |
| `exemption-pin-tests-on-main` | `grep:orchestrator/tests/test_merge_lane_ratchet.py::TestClusterPathSpec.test_size_ceiling_exempt_is_exactly_git_ops` `grep:orchestrator/tests/test_merge_lane_ratchet_write_gate.py::TestWriteBaselineRefusesAnUnauthorizedRaise.test_the_git_ops_size_exemption_holds_on_the_write_path` `grep:orchestrator/tests/test_merge_lane_ratchet_write_gate.py::TestCeilingsApplyOnlyToNewKeys.test_size_ceiling_exemption_covers_git_ops_even_when_new` `grep:orchestrator/tests/test_merge_lane_ratchet_write_gate.py::TestCeilingsApplyOnlyToNewKeys.test_the_exemption_is_scoped_to_the_ceiling_not_to_the_ratchet` `grep:orchestrator/tests/test_merge_lane_ratchet_write_gate.py::TestCeilingsApplyOnlyToNewKeys.test_a_non_exempt_path_at_the_same_size_is_a_violation` the D10 pins (the last is rewritten, not deleted) | — (upstream / substrate; no check) | PASS |
| `exemption-retired` | built here — §Goal item 1: git grep SIZE_CEILING_EXEMPT -- scripts orchestrator is empty (9 lines at a4349b7f4b); covers the governed_exceptions row that exception_register.py --report lists | `SIZE_CEILING_EXEMPT` absent in `scripts`, `orchestrator` | PASS |
| `exemption-pin-tests-retired` | built here — D10: the four exemption tests are defined at a4349b7f4b | `def test_(size_ceiling_exempt_is_exactly_git_ops\|the_git_ops_size_exemption_holds_on_the_write_path\|size_ceiling_exemption_covers_git_ops_even_when_new\|the_exemption_is_scoped_to_the_ceiling_not_to_the_ratchet)\(` absent in `orchestrator/tests/` | PASS |
| `grandfather-only-over-ceiling` | built here — D4: _check_ceilings grandfathers a baselined path only while its recorded lines exceed FILE_LINE_CEILING; B4 green in the gate file | manual | PASS |
| `git-ops-held-like-any-path` | built here — with git_ops.py's key removed from a copy of the baseline, check_against_baseline reports no new_file_over_ceiling for it | manual | PASS |

## Cross-PRD hazard for the decomposer (not a leaf)

Four S5 checks already copied into task records pin a symbol to `git_ops.py` that this batch moves or renames: S5 δ (9260) `enum-maps-seed-rc` (`def from_seed_rc\(`; δ moves it with the enum), S5 κ (9262) `pool-storage-guard` (`def _ensure_pool_storage\(`; δ makes it `PoolStorageGuard.ensure`) and `route-classifier` (`def _classify_acquire_route\(`; ζ moves it to `warm_acquire.py`), and S5 ι (9261) `adopt-lane-owner` (`def adopt_lane\(`, if ζ moves it). 9260, 9261 and 9262 have open dependents outside S5, 6698 among them; any still pending when δ or ζ lands is held and then blocked by a born-at-L2 `dependency_capability` escalation. Repath both halves (sidecar and task record) in the decompose commit, or in the leaf that moves the symbol.

**Resolved in the decompose commit (2026-11-22, gate 11190):** those four checks and S5 γ (9259)'s `callbacks-bundle` / `limits-dataclass` (`^class GitOpsCallbacks` / `^class GitOpsLimits`, which a leaf may move below `git_ops.py` for the layering) were widened to the package `orchestrator/src/orchestrator/`, and `pool-storage-guard` to `def (_ensure_pool_storage|ensure)\(` over `git_ops.py` + `pool_storage.py`, in both the S5 sidecar and the copied `metadata.delivered_checks` of 9259–9262. Polarity is unchanged: every widened pattern is absent from the whole package at a4349b7f4b. S5 β (9258)'s `runner: GitRunner` stays pinned to `git_ops.py`, whose constructor keeps that keyword (frozen surface).
