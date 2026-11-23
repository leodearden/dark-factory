# git_ops.py split: the git engine under the file-size ceiling (PRD)

**Status:** active, authored 2026-11-21 by task 7140 (headless author mode; each design fork below is
resolved with recorded rationale). **Type:** behaviour-preserving refactoring PRD over shipped,
load-bearing code. **Approach:** B+H. `GitOps` is the git engine for warm lanes, the scheduler, recovery
and the merge lane. Every task's worktree and every landing on `main` pass through it, and the batch
restructures all of it.

**Code anchors** verified against main `c787af9409` (2026-11-21). Main moves fast: cite by symbol and
re-locate at implementation time. The measured tables below are dated provenance of that commit and
must not be re-anchored. α records the post-S5 re-measurement as a dated Correction.

**Measurement artefacts** (committed beside this PRD, dated snapshots):
- `plans/git-ops-split-prd.partition.json`: the partition. It assigns every unit of `git_ops.py` to one cluster.
- `plans/git-ops-split-prd.concern-map.md`: the rendered concern map. Its header carries the reproduction
  command and the reconstruction rules R1–R6.

Both are produced by `scripts/concern_map.py`, the stdlib-AST instrument this task added. α runs the
same instrument again after S5 lands.

## Goal

The debt behind `scripts/merge_lane_cluster.py::SIZE_CEILING_EXEMPT` is the git engine's size, wherever
that code now lives. This PRD retires that debt and does not relocate it. The operator observes four
things when the batch lands:

1. `git grep -n SIZE_CEILING_EXEMPT -- scripts orchestrator` returns nothing, and
   `scripts/exception_register.py --report` no longer lists `scripts.merge_lane_cluster.SIZE_CEILING_EXEMPT`.
2. `scripts/merge_lane_metrics.py --report` lists every git-engine module. That means `git_ops.py`,
   the modules S5 creates (`git_run`, `git_diff`, `worktree_list`, `block_disposition`, `lane_lease`,
   `warm_lane_scripts`, `git_queries`, `task_branch`, `worktrees`, `main_landing`) and this PRD's modules
   (§Contract → Module ownership).
3. The ratchet holds every one of those modules to `FILE_LINE_CEILING` = 1,500 lines. No git-engine
   entry in `orchestrator/tests/merge_lane_ratchet_baseline.json` records more than 1,500 lines. A
   baselined path that is under the ceiling is held to it from then on (D4), so the engine cannot
   regrow unseen. The merge lane grew back that way three times (`plans/merge-lane-quality-prd.md`
   §Background).
4. The heuristic-13 import tripwire covers the new modules. Every git-engine module imports only
   downward through the §Sketch layering. None imports `orchestrator.git_ops`, none holds a
   function-local import of another git-engine module, and none re-exports a moved name (B1).

Behaviour is unchanged: every name, wire value and constructor keyword in §Contract → Frozen surface is
identical before and after every leaf.

## Background

`plans/merge-lane-quality-prd.md` decision 9 kept `git_ops.py` outside the merge-lane package. It
named "the broader `git_ops.py` split" as a follow-up PRD, and `SIZE_CEILING_EXEMPT` (owner
`Debt(TaskRef(7140))` in `scripts/tests/test_scripts_governed_lists.py::SIZE_CEILING_EXEMPT_LIST`)
records that debt. The list is redundant today. Its only consumer, `scripts/merge_lane_ratchet.py::_check_ceilings`,
skips a path that is `in baseline_files or in lane_cluster.SIZE_CEILING_EXEMPT`, and `git_ops.py`
is already a baseline key. Lines and prose lines are reported in the baseline but never ratcheted
(the baseline's `_README`). So deleting the list alone would prove nothing. Retiring it honestly means
the engine is under the ceiling and held there.

S5 (`plans/review-all-s5-gitops-ports-query-contract-and-the-git-runner-prd.md`, tasks 9255–9277; all
pending except μ 9276, which is done, at authoring) already partitions `git_ops.py` along self-state. It
extracts cluster (a) into `lane_lease` (S5 θ 9266), (b) into `worktrees` (S5 ξ0/ξ 9268/9269), (d) into
`task_branch` (S5 λ 9267), (e) into `main_landing` (S5 ο0/ο 9270/9271) and (f) into `git_queries` (S5 ρ0/ρ
9272/9273). Its gate is S5 π 9275. (In this PRD a bare Greek label is one of this PRD's leaves; an S5
leaf is always written "S5 θ" and so on.) S5's §Goal estimates the residual at ≈6.6k lines, "≥3× the 2,000-line
alarm", and names no owner for it. S5 also creates modules over 1,500 lines, and none of them is in
`CLUSTER_PATHS`, so no ceiling measures them. If this PRD deleted the exemption once `git_ops.py` alone
shrank, the debt would only move into unmeasured files. That is heuristic 14's "no cheating" failure,
and the cluster-total ratchet would even read the move as an improvement.

Measured at `c787af9409` (concern map; raw lines include prose):

| Measure | Value |
|---|---|
| `git_ops.py` lines / prose lines | 17,010 / 9,646 (57%) |
| Units | 342: 225 `GitOps` methods over 13,727 lines; 117 module-level units over 2,539 lines. Residual buckets 744: preamble 136 (96 code, 40 prose); class non-method 281, which is 232 blank separators between the 225 methods, 47 prose (a one-line class docstring and 46 comment lines a blank line away from the next method) and 2 code; module gaps 327 (220 blank, 107 prose) |
| `GitOps` own mutable state | `config`, `project_root` and `worktree_base` are read by 70, 92 and 37 methods; `warm_lane_pool` by 21; the rest is 5 counters and caches plus 4 callbacks. A map keyed on self-attribute writes alone cannot partition it, so the instrument also records the intra-class ref graph. |
| Comment lines inside method spans | worktrees 89, main advance 358, warm acquire 429, `GitOps` root and `create_worktree` 429. The rest of the prose is docstrings. |
| `src` importers of `orchestrator.git_ops` names | 34 names across 48 modules (AST `ImportFrom`); `workflow_types.py` imports 21 |

Projected module sizes under the committed partition (R1–R6). *Unit lines* is the sum of unit spans.
It leaves out the separators, preamble and class header that a module also carries, which is why the
*projected* column exists: it adds the units' apportioned share of the separators. *Projected + P*
adds the 50-line preamble allowance, and H = 35 is one median method of headroom (concern map,
"Projection, preamble allowance and headroom"). An S5-layout row is the sum of its constituent
clusters' figures.

| Module (owner) | Unit lines | Prose | Projected | Projected + P | Over 1,500 − H (1,465)? |
|---|---:|---:|---:|---:|---|
| `main_landing` as S5 lands it (advance + park + merging + merge-worktree lifecycle + interactive reap) | 3,364 | 1,862 | 3,443 | 3,493 | yes. Even with no prose, 1,502 unit lines. |
| `git_queries` as reconstructed (reads + commit-effect probe) | 2,550 | 1,552 | 2,644 | 2,694 | yes. S5's own figure is 1,636; S5's text does not list cluster (f) membership, so R2 decides. |
| `worktrees` (S5 ξ, after S5 γ takes the script wrappers) | 1,541 | 843 | 1,624 | 1,674 | yes |
| `task_branch` (S5 λ) | 1,402 | 648 | 1,472 | 1,522 | yes |
| `lane_lease` (S5 θ) | 884 | 674 | 932 | 982 | no |
| `warm_lane_scripts` (S5 γ, script halves only, R3) | 644 | 392 | 686 | 736 | no |
| S5's residual in `git_ops.py` (this PRD's `residual/*` clusters) | 5,567 | 3,289 | 5,686 | 5,736 | yes |

The lesson that shapes this PRD comes from `plans/merge-lane-quality-prd.md` §Background. The July merge-queue
split went by topic. It produced function-bags over the worker's private state, stitched back by
reach-back imports. This PRD splits by what each method owns and touches, never by topic or name
prefix, and it measures every candidate.

### Heuristic-14 measurement (the partition tried, and the symptom each rejected candidate hit)

Every candidate was evaluated with `scripts/concern_map.py --partition`. The commit messages of the
concern-map commits on `task/7140` carry the same record (subjects `docs(prd): git-ops split — measured
concern map and partition at c787af9409` and `docs(prd): git-ops split — projected sizes, main_park,
fallbacks and a measured layering`).

1. **S5's placements as written.** These close three cycles, which R6 records and regroups:
   - `PERSISTENT_MERGE_WORKTREE_NAME` in `main_landing`: `worktrees.PROTECTED_PREFIXES` needs it, so
     `worktrees` → `main_landing` → `worktrees` (`GitOps._create_merge_worktree` → `_worktree_add_with_retry`).
   - `ReplayMatch` in `task_branch`: `git_queries` needs it for `RebasedCitation` and `GitOps._find_replay`,
     and `task_branch` refs `git_queries` 19 times.
   - `GitOps._rename_branch_aside` in `task_branch`: it calls `GitOps._free_lane_checkout`, which reads the pool.
2. **The residual as one module, `git_ops.py` minus S5's clusters.** It is 5,567 unit lines, and every
   other measure is clean. Rejected on size alone.
3. **The residual by state, first cut.** Pool storage, seeding, warm base, liveness, acquire, reset, spec
   lanes and release were tried. This cut produced one thirteen-cluster cycle through S5's modules:
   - `_prune_registrations` → `_reconcile_pool_storage_before_sweep` is called from `worktrees`,
     `task_branch` and `main_landing`.
   - `_run_warm_lane_gc_reclaim` → the pool-storage guard.
   - the warm-base path helpers → `main_landing`'s persistent-lane paths.

   It was resolved by making pool storage, lane seeding and the lane-outcome types low modules below
   S5's modules, and by moving the persistent-path facts to `lane_paths` (R6). The rejected cut kept
   `_meta_roots_naming` in acquire, which closed the cycle `lane_release` → `acquire`.
4. **Admission and warm acquire both write `_consecutive_exhausted`.** That was a shared write. It
   was resolved by moving `_note_structural_exhaustion` to `warm_acquire`, which owns the counter.
5. **`worktrees` split into quarantine relocation and primitives.** Rejected because it cycles both
   ways: `GitOps._relocate_into_quarantine` → `GitOps.commit`, and `GitOps.rename_worktree` →
   `GitOps._move_registered_worktree`. Moving `rename_worktree` with the relocation (it relocates a
   registered worktree too) leaves only the downward edge to `commit`. The result is acyclic:
   `worktree_relocation` is 282 lines and `worktrees` is 1,259.
6. **`main_landing` split into merging, merge-worktree lifecycle, main advance and interactive reap.**
   Acyclic. The only internal edge is merging → lifecycle (`merge_to_main` → `_create_merge_worktree`,
   `cleanup_merge_worktree`). Lifecycle plus interactive reap is 1,428 unit lines, one module that
   projects to 1,452 + P = 1,502, so D12 names its fallback (item 11). Merging plus lifecycle would be
   1,559 unit lines, so that pairing was rejected.
7. **`git_queries` split into reads and commit-effect probe.** Acyclic. The only edge is
   `describe_commit_effect_in_main` → `get_main_sha`.
8. **`main_landing` split into main advance and the park and index-lock sink (`main_park`).** Acyclic:
   no symptom. The only new edge is `main_landing` → `main_park`, 9 refs (`GitOps.advance_main` →
   `GitOps._await_index_lock_clear`), and `main_park` refs only `git_queries` and `git_run`. Main
   advance is 1,039 unit lines (1,057 projected) and `main_park` 451 (477 projected); together they
   projected 1,533 + P = 1,583.
9. **`git_queries` fallback: `git_citations`.** The commit-citation and patch-id replay search, a set
   of units closed under `git_queries`' own refs to it that no other cluster refs. Acyclic: no symptom.
   474 unit lines (501 projected); it leaves `git_queries` at 953 (990 projected). New edges run only
   out of it: `git_queries` 3, `git_run` 9.
10. **`task_branch` fallback: `merge_markers`.** The merge-marker index, the only state `task_branch`
    writes. Acyclic: no symptom. 184 unit lines (198 projected); it leaves `task_branch` at 1,218
    (1,274 projected). New edges: it refs `git_queries` 2 and `git_run` 4, and `merge_worktrees` refs it 1.
11. **`merge_worktrees` fallback: `interactive_reaper`.** The interactive-worktree reap, a set closed
    under `merge_worktrees`' own refs to it that no other cluster refs. Acyclic: no symptom. 315 unit
    lines (321 projected); it leaves `merge_worktrees` at 1,113 (1,131 projected). New edges run only
    out of it, to modules below `merge_worktrees`.

Candidates 9–11 were evaluated alone and all three together, in variant partitions that are not
committed. Every variant has no cyclic group (concern map, "Fallback sub-clusters").

The committed partition is candidate 3 as resolved, plus candidates 5–8. It has no cyclic group. Each
of its five shared writes pairs `GitOps.__init__` with exactly one owner, and that owner's constructor
initialises the attribute.

## Activation status

The PRD is authored and decomposable now. No extraction leaf dispatches before α, and α waits on S5 π
(9275). The partition is measured on today's main and projected onto S5's layout. S5's γ (ports), ι
(`LaneLifecycle` owner), κ (acquire routes) and δ (`WarmLaneUnavailable` table) restructure exactly the
cluster this PRD splits. So the authoring map is a prediction, and α converts it into a measurement
(G6). This is the C-as-gate pattern.

## Sketch of approach

The layering below runs lowest first. Each module imports only downward, and no module imports
`git_ops.py`. Rows marked *S5* are S5's modules as S5 lands them. Rows marked *split* are this PRD's
sub-splits of an S5 module, and *fallback* rows exist only if D12 applies them. Rows marked *residual*
are this PRD's split of S5's residual. The first group is the bottom stratum: none of its modules
imports any module listed below it.

The layering orders constructor collaborators (§Contract's Takes column) as well as unit refs, so a
module may take only collaborators listed above it here. The order is a linear extension of the concern
map's "Cluster strata", and α re-derives it from the post-S5 map plus the Takes column rather than
maintaining it by hand. `lane_lifecycle` (868 lines) and `warm_lane_pool` (933 lines) are already in
`CLUSTER_PATHS` and the ratchet baseline at c787af9409. They are existing modules this batch does not
move, and D4's rule holds them to the ceiling like every other cluster file. No leaf removes them from
`CLUSTER_PATHS`. B1 asserts that they import no git-engine module.

```
shared.git_async, git_run, git_diff, worktree_list, block_disposition     S5 (α, ζ, δ)
lane_paths             persistent-lane and warm-base path facts            this PRD δ (R6; if S5 left them unplaced)
lane_outcomes          WarmLaneUnavailable table, WarmLaneRequeue family,  this PRD δ (types leaf)
                       WorktreeInfo, PoolPrewarmResult
lane_lifecycle         LaneLifecycle, LaneLiveness                         existing; S5 ι makes it the public owner
warm_lane_pool         WarmLanePool                                        existing
lane_lease                                                                S5 θ
warm_lane_scripts      WarmLaneScripts, WarmBaseHealth, the default WarmBaseProbe   S5 γ
pool_storage           PoolStorageGuard: the .pool-root guard and the one  this PRD δ
                       guarded `git worktree prune`
lane_seeding           LaneSeeder: seed a lane, mark storage, refresh base this PRD δ
git_queries            GitQueries (reads)                                  S5 ρ
commit_effect          CommitEffectProbe and the effect-survival probe     split η
git_citations          commit-citation and patch-id replay search          fallback η (D12)
merge_markers          the merge-marker index                              fallback η (D12)
task_branch            TaskBranchRefs                                      S5 λ
worktrees              WorktreeOps (primitives, ephemeral, interactive)    S5 ξ
worktree_relocation    rename and quarantine relocation                    split η
merge_worktrees        merge / host-verify / offline-deep worktree         split θ
                       lifecycle and reaping (and interactive reap, unless
                       D12 applies)
interactive_reaper     the interactive-worktree reap                       fallback θ (D12)
merging                merge a branch into a worktree or into main         split θ
main_park              the park ref, WIP parking, recovery-branch restore, split θ
                       the index-lock wait
main_landing           advance_main, recover_red_main, push_main           S5 ο, as θ leaves it
spec_lanes             SpecLanes                                           residual ε
lane_reset             LaneReset: reset, reseed, abort, prewarm            residual ε
lane_admission         LaneAdmission: disk/soft pressure, gc reclaim,      residual ε
                       census, reclaim-on-exhaustion
lane_release           LaneRelease: release, thin, terminal release,       residual ζ
                       prior-incarnation parking
warm_acquire           WarmLaneAcquirer: S5 κ's AcquireRoute table            residual ζ
git_ops                GitOps: composition root, create_worktree, C7's     residual ζ
                       thin lease uses, the IN_USE spans
```

`GitOps` composes every collaborator. Callers call the collaborator directly
(`git_ops.spec_lanes.acquire(...)`), never a delegating `GitOps` method (S5 D14). A façade with one
delegating method per collaborator method would be the stitching that heuristic 14 forbids. Each
extraction leaf does the following:
- moves its cluster's bodies and the types the placement rule gives it;
- repoints every caller, test and type importer, enumerated at decompose by `git grep -l` over the
  moved method names plus AST `ImportFrom` for the types;
- adds its new module to `CLUSTER_PATHS` and regenerates the baseline in the same commit (a move lowers
  the source file's numbers, so it is not a raise);
- moves the moved code's tests into the new module's test file, driving its public seam (heuristic 14's
  test-file rule);
- states the module's purpose in one sentence without "and" (heuristic 6).

## Resolved design decisions

- **D1 (scope).** Scope is `git_ops.py` and every module its code moves into, whether S5's or this
  PRD's. Each is held to 1,500 lines before ω deletes the exemption. Rationale: §Background. The size
  debt belongs to the engine, not to one filename.
- **D2 (the instrument).** `scripts/concern_map.py` measures the partition, which is committed beside
  this PRD with its rendering, dated as-of SHA. merge-lane-quality and S5 each measured "AST over
  `self.<attr>`" in a session scratchpad. This PRD needs the measurement twice: at authoring and in α.
  One committed instrument is SPOT (heuristic 11), and it puts the heuristic-14 measurement in git (the
  Comments stance; INV-9).
- **D3 (α gate).** α re-measures post-S5 main and commits a dated Correction before any move leaf
  dispatches (§Activation status).
- **D4 (the ratchet binds the engine).**
  - β adds every git-engine module to `CLUSTER_PATHS` through the authorized-raise path (merge-lane-quality
    Correction 8: a ceiling breach counts as a raise), so S5's over-ceiling modules enter the baseline
    measured and named rather than unmeasured.
  - Each extraction leaf adds its own module in the same way, without a raise.
  - ω then changes `scripts/merge_lane_ratchet.py::_check_ceilings` so that a baseline entry grandfathers
    a path only while its recorded `lines` exceed `FILE_LINE_CEILING`. An entry at or under the ceiling is
    held to it.

  Rationale: lines are not ratcheted, so today a baselined file can grow without limit. A ceiling that
  binds only the commit that introduces a file cannot keep the engine split. The rule keeps
  merge-lane-quality decision 1 intact: a key over the ceiling stays grandfathered, so the ratchet is
  still not a day-one gate. It only stops a file under the ceiling from regrowing unseen, which is the
  `_check_ceilings` docstring's own "grandfathering can only ever shrink". If any cluster path is over
  the ceiling at ω with a baseline entry under it, ω records it through `--authorize-raise`, naming its
  owner. A scoped rule (a second list of engine paths) was rejected: it adds a governed list to retire
  one, and two ceiling semantics for one ratchet fail heuristic 3.
- **D5 (prose first).** γ rehomes rationale prose before any move, per heuristic 14's measurement order
  ("first, whether the prose is in its right home").
  - The home is a new `docs/git-engine/decisions.md`: one dated entry per rationale, citing its task or
    escalation id, with a one-line pointer left in code. That is one home with pointers (INV-9), and a
    separate file from the merge lane's planned `docs/merge-lane/decisions.md` (task 5047 creates it), because the git engine is not the merge lane
    (heuristic 6).
  - The `archaeology_blocks` measure is shared with task 5047 (merge-lane ρ). Whichever of 5047 and γ
    lands first defines it in `scripts/merge_lane_metrics.py`, and the second reuses it (heuristic 11).
    There is no edge, because 5047 is low priority and pending, and an edge would be an unbounded hold
    (INV-7).
  - The comment-line count above shows that most of the engine's prose is docstrings. γ rehomes
    rationale wherever it sits: an incident history, a rejected alternative or a "task N changed this"
    paragraph in a docstring is rationale. A docstring keeps the contract: arguments, returns, raises
    and invariants.
- **D6 (the residual partition).** S5's residual becomes the eight `residual/*` modules plus `GitOps`,
  in the layering of §Sketch, by the committed map.
  - `create_worktree` stays on `GitOps`. It is the one operation that orchestrates every collaborator
    (warm acquire, prior-incarnation refusal, the admission census, cold reattach, worktree add, the
    branch rebase). Moving it out would leave `GitOps` as a bag of attributes.
  - `GitOps`'s purpose is "the git engine's composition root, which provides a task its worktree".
  - The pools (`WarmLanePool` instances) are constructed by `GitOps` and passed to the collaborators
    that read them. The pool module owns its own state, so this is explicitly owned durable state, not
    a stateless-interaction violation (heuristic 7).
- **D7 (low modules first).** S5's modules reference three residual concerns, 10 refs in the map's
  edge table: `_prune_registrations` 7, from `worktrees`, `task_branch` and `merge_worktrees`;
  `_seed_warm_lane` 2, from `worktrees`; and `WorktreeInfo` 1, from `task_branch`. So δ extracts
  `pool_storage`, `lane_seeding` and `lane_outcomes` first, below every S5 module. Until δ lands, S5's
  modules reach those concerns however S5 routed them (a port or its one-step forwarding reference),
  and α records which. δ makes them direct downward imports.
- **D8 (sub-splits of S5's modules).**
  - η extracts `worktree_relocation` from `worktrees` and `commit_effect` from `git_queries`.
  - θ extracts `merging`, `merge_worktrees` and `main_park` from `main_landing`, leaving the
    main-advance cluster at 1,039 unit lines (1,057 projected, 1,107 + P).
  - `main_park` is a planned split, not a fallback. It is the park and index-lock sink:
    `_park_wip_on_private_ref`, `_create_recovery_branch_from_park_ref`,
    `_safe_restore_park_with_recovery`, `_index_lock_path`, `_index_lock_state`,
    `_await_index_lock_clear`, `MERGE_PARK_REF`, the `MergePark*` errors and the `_INDEX_LOCK_*`
    constants: 451 unit lines (477 projected). Its units ref nothing in the advance cluster. Advance
    plus park projects to 1,533 + P = 1,583 before 5042 adds any def lines, so the breach is predicted,
    not a risk.
  - All four splits are measured acyclic (§Background items 5–8), and each is an S5 module that
    remains over the ceiling.
  - θ depends on task 5042, which decomposes `advance_main` in place into named steps, so θ splits a
    stable function rather than racing it.
- **D12 (measured fallbacks for the modules near the ceiling).** Three modules project over
  1,500 − H after D8's splits (§Contract → Ceilings). Each has a named fallback, chosen by what its
  units own and touch and measured acyclic alone and together (§Background items 9–11; concern map,
  "Fallback sub-clusters"):
  - η: `git_queries` → `git_citations`, the commit-citation and patch-id replay search, a unit set no
    other cluster refs. 474 unit lines; `git_queries` keeps 953 (990 projected, 1,040 + P).
  - η: `task_branch` → `merge_markers`, the merge-marker index, the only state `task_branch` writes.
    184 unit lines; `task_branch` keeps 1,218 (1,274 projected, 1,324 + P). `MergeMarkers` then
    initialises the three `_merge_marker_index*` attributes.
  - θ: `merge_worktrees` → `interactive_reaper`, the interactive-worktree reap, a unit set no other
    cluster refs. 315 unit lines; `merge_worktrees` keeps 1,113 (1,131 projected, 1,181 + P).

  The trigger is measured, not judged: the owning leaf applies a fallback when its module's post-γ
  `scripts/merge_lane_metrics.py --report` lines exceed 1,500 − H = 1,465, and records the measurement
  in its commit message. Rationale: γ's rationale share is unmeasured at authoring, so γ may bring a
  module under on its own; a split that a measurement does not need would only add a seam.
- **D9 (seam rulings).**
  - Task 5042 keeps `advance_main`, and this PRD never edits that function.
  - 6432's proposed read probes (`second_parent`, `parent_count`, `path_has_history`) land on S5's
    `git_queries.GitQueries`, never on `GitOps`. This is a note with no edge, because 6432 is low and
    an edge would be an unbounded hold.
  - S9 λ1 (9246) edits `WarmLaneUnavailable` member data, so δ depends on it: one same-file edge, and
    9246 already waits on S5 δ (9260).
  - New collaborators are composed inside `GitOps.__init__` unless S8 β (9218) has moved construction
    into `Harness` by then. In that case the leaf composes them where S8 put construction (whichever
    lands second adapts).
- **D10 (ω names every pin).** The retirement leaf deletes or rewrites every site that pins the
  exemption (§Decomposition ω), including the three write-gate tests the task text omitted. Its signal
  is behavioural: with `git_ops.py`'s baseline key removed, the ceiling check passes it as a new key.
- **D11 (the Debt owner).** `Debt(TaskRef(7140))` goes stale when 7140 closes. The decompose commit
  re-points it to ω's task id, or stamps 7140's `x_coalesced_into`.

## Pre-conditions for activating

- **S5 π (9275) is done.** α depends on it. Every other leaf depends on α.
- **Task 5042 is done.** θ depends on it.
- **S9 λ1 (9246) is done.** δ depends on it (D9).
- **Substrate (G3), verified on main `c787af9409`.**
  - Present on main:
    - `scripts/merge_lane_cluster.py::CLUSTER_PATHS`, `::SIZE_CEILING_EXEMPT` and `::FILE_LINE_CEILING`;
    - `scripts/merge_lane_ratchet.py::_check_ceilings` and `--authorize-raise`;
    - `orchestrator/tests/merge_lane_ratchet_raises/` (one file per authorized raise, task 7298);
    - `scripts/merge_lane_metrics.py --report`;
    - `scripts/exception_register.py --report`.
  - Built here, so on main once this branch merges: `scripts/concern_map.py` and
    `scripts/source_measures.py::prose_line_numbers_in_tree`.
  - Not yet on main:
    - S5's modules and its `ScriptedGitRunner`, `WarmLaneScripts`, `GitOpsCallbacks` and B27 tripwire.
      α verifies each exists after π.
    - the `archaeology_blocks` measure (γ or 5047 builds it, D5);
    - `docs/git-engine/decisions.md` (γ creates it).
- **Decompose-time steps (the decomposer performs these):**
  - Wire α after S5 π (9275).
  - Gate every non-terminal task whose `metadata.files` declares `git_ops.py` or an S5 git-engine
    module behind α, in merge-lane-quality decision 10's shape. Select by declared files, never by
    keyword. Hold an in-progress task with a live claimant behind nothing; it lands first, which is a
    bounded hold under INV-7.
  - Re-point `Debt(TaskRef(7140))` in `scripts/tests/test_scripts_governed_lists.py::SIZE_CEILING_EXEMPT_LIST`
    to ω's task id (or stamp 7140's `x_coalesced_into`) in the decompose commit (D11).
  - Add a dated note to task 6432 naming `git_queries.GitQueries` as its seam home (D9).
  - Add the trigger-chain gate (6698) as a dependent of every `high` leaf
    (`skills/_shared/filing-the-trigger-chain.md` §4).

## Cross-PRD relationship (G4)

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| S5 (`plans/review-all-s5-gitops-ports-query-contract-and-the-git-runner-prd.md`, 9255–9277) | consumes its landed layout | its modules, `GitOps` constructor ports (C2), `LaneLifecycle` (C6), the AcquireRoute table (S5 κ), B27 | S5 owns its extractions; **this PRD** owns S5's ≈6.6k-line residual and the over-ceiling S5 modules (paired dated cross-reference appended to S5) | α → π 9275 (queued at decompose) |
| merge-lane-quality (`plans/merge-lane-quality-prd.md`) | closes its decision 9 / Out-of-scope follow-up; edits its instrument | `CLUSTER_PATHS`, `_check_ceilings`, the baseline, the authorized-raise ledger | **this PRD** (β extends the paths; ω changes the grandfather rule, D4) — paired dated Correction appended there | wired by this commit |
| merge-lane-quality μ, task 5042 | ordering | `advance_main`'s in-place decomposition (in `main_landing.py` after S5 ο) | 5042 owns `advance_main`; this PRD never edits it; θ depends on 5042 | queued (edge at decompose) |
| merge-lane-quality ρ, task 5047 | shares a measure | `archaeology_blocks` in `scripts/merge_lane_metrics.py` | whichever of 5047 and γ lands first defines it; the second reuses it | note, no edge (D5) |
| task 6432 / S7 | note | new read probes | `git_queries.GitQueries` (S5), never `GitOps` | note at decompose (D9) |
| S9 λ1, task 9246 | same-file editor | `WarmLaneUnavailable` member data | S9 edits the data; δ moves the type to `lane_outcomes` after it | δ → 9246 (edge at decompose) |
| S8 β, task 9218 | composition site | where `GitOps` and its collaborators are constructed | S8 owns `Harness` construction; this PRD's leaves compose inside `GitOps.__init__` (whichever lands second adapts) | note |
| inv12 PRD (`plans/inv12-exceptions-owned-or-ratified-prd.md`), `docs/legibility/inv12-day-one-rulings.md` | owner of the `Debt` row | `SIZE_CEILING_EXEMPT_LIST` default disposition | decompose re-points the owner (D11); ω deletes the declaration. The day-one rulings file is a dated record of the sitting and is not edited. | queued |

No reciprocal-ownership ambiguity remains.

## Contract (B+H)

### Module ownership (target layout)

| Module | Class / surface | Owns | Takes (constructor) | Leaf |
|---|---|---|---|---|
| `lane_paths.py` | functions of `(worktree_base, config)`: persistent merge / offline-deep worktree paths, `is_persistent_merge_lane`, the merge-verify artifact path, the warm-base target path; `PERSISTENT_*_WORKTREE_NAME` | — | — | δ, only if α finds S5 left them unplaced or in an upper module (R6) |
| `lane_outcomes.py` | `WarmLaneUnavailable` (with S5 δ's member data and S9 λ1's), `WarmLaneRequeue` and its 8 subclasses, `WorktreeInfo`, `PoolPrewarmResult` | — (frozen values) | — | δ |
| `pool_storage.py` | `PoolStorageGuard`: `present()`, `mark_present()`, `in_use()`, `ensure(context) -> bool` (S5 κ's `_ensure_pool_storage`), `prune_registrations(context)` — the one `git worktree prune` | the pool-storage-absent callback | `LaneLifecycle`, the two pools, the base probe, runner | δ |
| `lane_seeding.py` | `LaneSeeder`: `seed(...)` (scripts.seed + `from_seed_rc` + `mark_present` on success), `refresh_warm_base()` | — | `WarmLaneScripts`, `PoolStorageGuard`, warm pool, runner | δ |
| `spec_lanes.py` | `SpecLanes`: `acquire(...)`, `release(...)` | `spec_warm_lane_pool`'s create lock | spec pool, `PoolStorageGuard`, `LaneSeeder`, `WorktreeOps`, the merge-worktree collaborator | ε |
| `lane_reset.py` | `LaneReset`: reset, reseed-verified-clean, recycle, abort, prewarm | — | warm pool, `LaneSeeder`, `PoolStorageGuard`, `WorktreeOps`, `TaskBranchRefs`, scripts, `transcript_archive` | ε |
| `lane_admission.py` | `LaneAdmission`: disk / soft-pressure admission, gc reclaim, census, reclaim-on-exhaustion, pool-shrink warning | the audit cache; the reclaim-candidate provider and dispatched predicate (S5 C2 callbacks) | warm pool, `LaneLifecycle`, scripts, `PoolStorageGuard`, the lease-active probe | ε |
| `lane_release.py` | `LaneRelease`: release, thin, terminal-task release, detach, `cleanup_worktree`, prior-incarnation evidence and parking | — | warm pool, `SpecLanes`, `PoolStorageGuard`, `WorktreeOps`, relocation, `TaskBranchRefs`, `GitQueries`, scripts | ζ |
| `warm_acquire.py` | `WarmLaneAcquirer`: `acquire(...)`, S5 κ's `_ACQUIRE_ROUTES` and one method per `AcquireRoute` | `_consecutive_exhausted`; the structural-exhaustion callback | warm pool, `LaneLifecycle`, every collaborator above | ζ |
| `git_ops.py` | `GitOps`: composition, `create_worktree`, C7's thin lease uses, the IN_USE spans | the pools' construction, `LaneLiveness` | S5 C2's keywords, unchanged | ζ |
| `worktree_relocation.py` | `WorktreeRelocation`: `rename(...)`, `quarantine(...)`, `quarantine_base` | — | runner, config, worktree_base, `TaskBranchRefs` | η |
| `commit_effect.py` | `CommitEffectProbe` (value) and `CommitEffects`: `describe_in_main`, `present_in_main` | the effect-probe memo | runner, `GitQueries` | η |
| `merge_worktrees.py` | `MergeWorktrees`: create / remove-guarded / cleanup / prune / reset persistent / host-verify / reap survivors / find in-flight; interactive reap | — | runner, config, worktree_base, `WorktreeOps`, `PoolStorageGuard`, lease primitives | θ |
| `merging.py` | `Merging`: `merge_to_main`, `merge_branch_into_worktree`, conflict details, abort | — | runner, config, `MergeWorktrees`, `GitQueries` | θ |
| `main_park.py` | `MainPark`: park WIP on the private ref, restore it with recovery, create a recovery branch from the park ref, wait for the index lock; `MERGE_PARK_REF`, the `MergePark*` errors | the index-lock path cache | runner, project_root, `GitQueries` | θ |
| `git_citations.py` | `GitCitations`: find a task's citation commit, find an equivalent (replayed) commit, resolve a rebased citation; `task_citation_pattern`, `RebasedCitation`, `ReplayMatch` | — | runner, config, `GitQueries` | η, only if D12 applies |
| `merge_markers.py` | `MergeMarkers`: find a branch's merge marker through the merge-marker index | the merge-marker index and its freshness stamps | runner, config, `GitQueries` | η, only if D12 applies |
| `interactive_reaper.py` | `InteractiveReaper`: reap interactive worktrees that landed or were abandoned | — | runner, config, worktree_base, `WorktreeOps`, `PoolStorageGuard`, `WarmLaneScripts`, `TaskBranchRefs` (or `MergeMarkers`), `GitQueries` | θ, only if D12 applies |

Rules:
- No module imports `orchestrator.git_ops`, and imports run downward only, per §Sketch. B27 already
  holds this for S5's modules, and B1 extends it to these.
- No module-level mutable state.
- Each collaborator initialises the attributes it owns, which resolves the map's five shared writes.
- Method names inside a collaborator drop the redundant prefix and underscore when the method becomes
  part of its public surface (`acquire_spec_lane` → `SpecLanes.acquire`). Callers are repointed and no
  alias is left.
- A type moves with the first leaf whose module needs it, and every importer is repointed with no
  re-export left behind (S5's placement rule).

### Frozen surface (behaviour preservation)

These are byte-identical before and after every leaf:

- The `GitOps(...)` constructor keyword set as S5 C2 leaves it, so the 200+ test files that construct
  `GitOps(` and the five composition roots need no edit for construction.
- `WarmLaneUnavailable` member values (logged strings), each member's requeue class, disposition, steal
  retryability and seed exit codes (S5 C3), and the FAULT `RuntimeError` message built from the table.
- The `WarmLaneRequeue` subclass names that `workflow_types.py::_disposition_table` derives rows for,
  and their block dispositions.
- `create_worktree`'s signature and results: `WorktreeInfo` fields, the raised requeue classes per
  route (S5 B4, B7, B22).
- The pool-storage guard's semantics: a configured pool whose storage is absent and not provably a
  first-seed bootstrap refuses every guarded sweep, fires the callback, and issues no `git worktree
  prune`; a pool-less host never disables prune.
- `.lane-state/` record writes, which still go only through `LaneLifecycle` and the pool (S5 C6).
- The `.pool-root` sentinel path and the `.merge_verify.lock` path.
- Every escalation, event and journal string the moved code emits. The log *logger name* changes to
  each module's own `orchestrator.<module>`. A leaf that moves a logging site updates the `caplog`
  assertions that name `orchestrator.git_ops` (enumerated at decompose by `git grep -l
  "orchestrator.git_ops" -- '*/tests/*'`).

### Ceilings

| Measure | Ceiling | Basis (G6) |
|---|---|---|
| Lines per git-engine module | 1,500 (`FILE_LINE_CEILING`) | heuristic 14's soft ceiling and the merge-lane ceiling (merge-lane-quality §Ceilings: 1,500 sits below the 2,000-line single-`Read` alarm). Basis: a module's size is its cluster's projected lines (unit lines plus apportioned separators) plus the preamble allowance P = 50, held against 1,500 with a headroom band H = 35, so over 1,465 is near the ceiling (concern map, "Projection, preamble allowance and headroom"). At `c787af9409` four modules project over 1,500 before γ: `main_landing` with the park sink 1,533 + P = 1,583 (prose 722), `git_queries` 1,491 + P = 1,541 (prose 797), `task_branch` 1,472 + P = 1,522 (prose 648) and `merge_worktrees` 1,452 + P = 1,502 (prose 867). γ's rationale share is unmeasured at authoring, because γ builds `archaeology_blocks`. So γ is the expected first reduction, but none of the four relies on it: `main_landing` has θ's planned `main_park` split (D8), and the other three each have a named, measured-acyclic fallback (D12). Every other module projects at or under 1,465 with P; the largest are `worktrees` 1,322 + P = 1,372 and `warm_acquire` 1,311 + P = 1,361. `git_ops.py` is below. |
| `git_ops.py` | 1,500 | measured bound 1,397, under 1,465 without γ or S5 θ: its 7 methods' 1,177 unit lines, plus their separator share ⌈7 × 232/225⌉ = 8, plus today's 136-line module preamble (an upper bound for the composition root's own docstring and imports), plus the class's 49 non-separator lines (47 prose, 2 code), plus one import line per composed collaborator module (27: the 17 modules of §Contract → Module ownership other than `git_ops.py`, fallbacks included, and S5's ten). The class's 281 non-method lines are not mostly prose: 232 are blank separators between methods, and the class docstring is one line. Open question 1's `task_worktrees` move stays a measured contingency: `create_worktree` and `_reattach_cold_worktree` are 698 unit lines (701 projected), acyclic as their own cluster, and leave 479 unit lines on `GitOps`. |
| Cyclic groups among git-engine modules | 0 | the committed partition has none (concern map, "Cyclic cluster groups: none") |
| Function-local imports / re-exports of git-engine names inside the engine | 0 | B27 (S5) and B1 |

## Boundary-test sketch

ι's suite is `orchestrator/tests/test_git_engine_boundary_gate.py`. It runs on a real tmp repo with S5's
`ScriptedGitRunner` and `WarmLaneScripts` fakes passed through `GitOps(...)`, and it faces both sides of
every seam.

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 | Import layering (producer side) | every git-engine module in `CLUSTER_PATHS`, plus `lane_lifecycle` and `warm_lane_pool` (bottom stratum; existing modules already in `CLUSTER_PATHS`) | AST: each imports only modules below it in §Sketch; `lane_lifecycle` and `warm_lane_pool` import no git-engine module; none imports `orchestrator.git_ops`; no function-local import of a git-engine module; no name re-exported from a module that does not define it |
| B2 | Importers repointed (consumer side) | all leaves landed | `git grep -nE "from orchestrator\.git_ops import" -- orchestrator/src escalation/src scripts` matches only `GitOps` (and S5's remaining names, if α records any); every moved name resolves from its new module |
| B3 | Ceiling holds | the live tree | `scripts/merge_lane_metrics.py --report` rows for every git-engine module are ≤ 1,500 lines; the baseline records no git-engine path above 1,500 |
| B4 | Regrowth is caught (D4) | a fixture baseline recording a path at 1,400 lines; a current report at 1,501 | `check_against_baseline` reports `new_file_over_ceiling` for it; the same path recorded at 1,600 (grandfathered) at 1,601 does not |
| B5 | The guarded prune, from every caller module | pool configured, `.pool-root` absent, base not resolvable | `worktrees.prune_worktrees`, task-branch cleanup and `merge_worktrees` cleanup each call `PoolStorageGuard.prune_registrations`; no `git worktree prune` argv is issued; the callback fires once per sweep |
| B6 | The guarded prune on a pool-less host | no pool | prune runs |
| B7 | Warm acquire routes through the composition | pool of 1, scripted seed rc per route | `create_worktree` returns the same `WorktreeInfo` / raises the same `WarmLaneRequeue` subclass per `AcquireRoute` as S5's B22 recorded; `_consecutive_exhausted` lives only on `WarmLaneAcquirer` |
| B8 | Seed exit code → unavailability | `FakeScripts(seed_rc=75/76/77/124/127)` | `LaneSeeder.seed` maps through `WarmLaneUnavailable.from_seed_rc`; a 0 marks storage present exactly once |
| B9 | Spec lane round trip | spec pool of 1 | `git_ops.spec_lanes.acquire` then `.release` leaves the lane free and registered; the cold fallback goes through the merge-worktree collaborator |
| B10 | Terminal release and prior incarnation | a lane held by a terminal task; a branch with a prior incarnation | `lane_release` releases and thins; `create_worktree` parks the prior incarnation into quarantine through `worktree_relocation` and proceeds |
| B11 | No delegating façade | `GitOps` class | AST: no `GitOps` method whose body is a single call forwarding to `self.<collaborator>.<method>(…)` with the same arguments |
| B12 | Disposition table unchanged | every `WarmLaneUnavailable` member | `workflow_types.classify_failure(cls(...))` returns the same `BlockDisposition` as S5's B7 recorded, now with the type imported from `lane_outcomes` |
| B13 | S5's suite still passes | S5 π's `test_gitops_boundary_gate.py` | green, unedited except for import repoints |
| B14 | Merging after the main_landing split | a queued branch | `git_ops.merging.merge_to_main` creates and cleans up its merge worktree through `MergeWorktrees`; `main_landing` imports neither |

## Decomposition plan

All leaves are `planning_mode=True` and `task_kind='normal'`. Every leaf carries
`metadata.x_finding_key = ["fk-c00c5f01be4c"]` (`git_ops.py` h14, continued from S5's split leaves),
`x_finding_run = "review-all-dark_factory-20261008"`, `source = "prd"` and `metadata.files`
(enumerated at decompose).

Each leaf binds heuristic 6 (one-sentence purpose), heuristic 9 (deep modules, narrow interfaces),
heuristic 13 (files make internal sense in isolation) and heuristic 14 (no file too large, no
cheating), cited by name in its brief, plus the ones named below.

Sizes are expected changed LOC in the overlay's sense, with moves counted by moved region. If an
extraction leaf's enumerated caller set exceeds about 40 test files at decompose, it is filed as a
codemod sibling (`complexity='simple'`, S5's ξ0 shape) plus the move leaf. Dependency order is one
total order, because every leaf edits `CLUSTER_PATHS`, the baseline and either `git_ops.py` or an S5
module (the overlay's same-file rule): **α → β → γ → δ → ε → ζ → η → θ → ι → ω**.

- **α — Re-measure the git engine after S5 lands** (priority medium; ~300 LOC).
  - Modules: the PRD (a dated Correction), new `plans/git-ops-split-prd.partition-post-s5.json`, and its
    rendering appended to the Correction or a sibling `.concern-map-post-s5.md`.
  - Run `scripts/concern_map.py` on post-S5 `git_ops.py` with a partition derived from the committed
    one. A unit still in `git_ops.py` keeps its cluster, and a new unit is placed by R2.
  - Record each git-engine module's lines and prose lines from `source_measures` per-file measures.
  - List every unit S5 left that the projection assumed moved, or the reverse.
  - Record where S5 placed the R6 units and how it routed the D7 refs.
  - Name every cycle found. Its resolution is a leaf of this batch, never a new topic split.
  - Re-derive the §Sketch layering from the post-S5 map's "Cluster strata" plus the §Contract Takes
    column, and re-measure projected lines + P for every module not yet created. Record the list of
    modules near the ceiling (over 1,500 − H) in the Correction, with the D8 split or D12 fallback that
    covers each.
  - Signal: the Correction is committed with the as-of SHA, and the post-S5 partition evaluates with
    exit 0 and "Cyclic cluster groups: none", or the Correction names each cycle and the leaf that cuts it.
  - Prereqs: S5 π 9275. It gates every other leaf.
- **β — The ratchet measures the git engine** (D4, first half; priority high; ~500 LOC).
  - Modules: `scripts/merge_lane_cluster.py::CLUSTER_PATHS` (every S5 git-engine module, as literals),
    `orchestrator/tests/merge_lane_ratchet_baseline.json` (regenerated), one new authorized-raise entry
    under `orchestrator/tests/merge_lane_ratchet_raises/` (via `--authorize-raise <β's id> --reason …`;
    it names each over-ceiling module and its owning leaf η/θ), and the `CLUSTER_PATHS` pins in
    `orchestrator/tests/test_merge_lane_ratchet.py`.
  - Signal: `scripts/merge_lane_metrics.py --report` lists every S5 git-engine module, and the raise
    entry names every S5 git-engine module whose measured `--report` lines exceed 1,500, each with its
    measured lines and owning leaf.
  - Prereqs: α.
- **γ — Prose to its home** (D5; heuristic 14's measurement order; Comments stance; priority medium;
  ~1,500 LOC, mostly moved prose).
  - Modules: new `docs/git-engine/decisions.md`, `git_ops.py` and every S5 git-engine module, and
    `scripts/merge_lane_metrics.py` (the `archaeology_blocks` measure, unless 5047 landed it first).
  - Each rehomed rationale becomes one dated entry citing its id. The code keeps a one-line pointer,
    and a docstring keeps only the contract.
  - Signal: `--report` reads `archaeology_blocks = 0` for every git-engine module. The decisions doc
    has one entry per relocated block.
  - Prereqs: β.
- **δ — The low residual modules: `lane_outcomes`, `pool_storage`, `lane_seeding` (and `lane_paths`)**
  (D7; heuristic 11 for the one guarded prune; priority high; ~1,300 LOC moved).
  - Modules:
    - the new modules;
    - `git_ops.py`, minus the moved units;
    - `worktrees.py`, `task_branch.py` and `main_landing.py` (their D7 refs become downward imports);
    - `workflow_types.py`, `workflow.py` and every other importer of a moved type (repointed, no re-export);
    - `CLUSTER_PATHS` and the baseline;
    - `test_pool_storage_guard.py` and the type importers' tests;
    - new `test_lane_outcomes.py`, `test_lane_seeding.py`.
  - `lane_paths` is created only if α found S5 left those facts in `main_landing` or `git_ops`.
  - Signal: B5, B6, B8 and B12. `^class WarmLaneUnavailable` is present in `lane_outcomes.py` and
    absent from `git_ops.py`.
  - Prereqs: γ, and S9 λ1 9246 (D9).
- **ε — Lane reset, spec lanes and admission** (D6; heuristic 5 for the audit cache's owner; priority
  medium; ~1,500 LOC moved).
  - Modules: new `lane_reset.py`, `spec_lanes.py` and `lane_admission.py`; `git_ops.py`; callers
    (`harness.py` for prewarm, the pool-shrink warning and the spec-lane sweeps; merge-lane spec-lane
    callers; `cli.py` as enumerated); `CLUSTER_PATHS` and the baseline; tests.
  - Signal: B9. The audit cache is an attribute of `LaneAdmission` only.
  - Prereqs: δ.
- **ζ — Lane release, warm acquire, and `GitOps` as the composition root** (D6; heuristic 7; priority
  high; ~2,200 LOC moved).
  - Modules: new `lane_release.py` and `warm_acquire.py`; `git_ops.py` (what remains: composition,
    `create_worktree`, the thin uses, the IN_USE spans); callers (`harness.py` terminal release and
    reconcile, `workflow.py`); `CLUSTER_PATHS` and the baseline; tests.
  - Signal: B7, B10 and B11. `git_ops.py` is ≤ 1,500 lines, recorded in the commit message with the
    partition, or the Open-question-1 fallback is applied and recorded.
  - Prereqs: ε.
- **η — Sub-split `worktrees`, `git_queries` and `task_branch`** (D8, D12; priority medium; ~2,200 LOC
  moved with both fallbacks).
  - Modules: new `worktree_relocation.py` and `commit_effect.py`, and `git_citations.py` and
    `merge_markers.py` if D12 applies; `worktrees.py`, `git_queries.py`, `task_branch.py`; callers of
    `rename_worktree`, `quarantine_worktree`, `describe_commit_effect_in_main` and
    `commit_effect_present_in_main`, and of the citation, replay and merge-marker lookups if they move;
    `CLUSTER_PATHS` and the baseline; tests.
  - Signal: B1 for every module η creates. `worktrees.py`, `worktree_relocation.py`, `git_queries.py`,
    `commit_effect.py` and `task_branch.py` (and `git_citations.py` and `merge_markers.py` if created)
    each measure ≤ 1,500 − H in `--report`, or the D12 fallback for that module is applied and its
    measurement recorded in the commit message.
  - Prereqs: ζ.
- **θ — Sub-split `main_landing`** (D8, D12; priority medium; ~2,500 LOC moved).
  - Modules: new `merge_worktrees.py`, `merging.py` and `main_park.py`, and `interactive_reaper.py` if
    D12 applies; `main_landing.py`; callers of the moved methods (S5 ο0's 99-test-file set,
    re-enumerated); `orchestrator/src/orchestrator/merge_lane/gates.py`'s merge-worktree factory call; `CLUSTER_PATHS` and the
    baseline; tests.
  - Signal: B14, and `async def merge_to_main(` is absent from `main_landing.py`. `main_landing.py`,
    `main_park.py`, `merging.py` and `merge_worktrees.py` (and `interactive_reaper.py` if created) each
    measure ≤ 1,500 − H in `--report`, or the D12 fallback for `merge_worktrees` is applied and its
    measurement recorded in the commit message.
  - Prereqs: η, and task 5042.
- **ι — Integration gate** (signal leaf; priority high; ~500 LOC).
  - Modules: new `orchestrator/tests/test_git_engine_boundary_gate.py` (B1–B14, except B4, which ω's
    rule needs and so lands with ω).
  - Signal: the gate is green. The commit message records the re-measured concern map of `git_ops.py`
    and each new module's `--report` row.
  - Prereqs: θ.
- **ω — Retire `SIZE_CEILING_EXEMPT`** (D4 second half, D10; priority high; ~300 LOC). Deletes or
  rewrites:
  - `scripts/merge_lane_cluster.py::SIZE_CEILING_EXEMPT` and the `CLUSTER_PATHS` comment that points to it;
  - the `or path in lane_cluster.SIZE_CEILING_EXEMPT` clause in
    `scripts/merge_lane_ratchet.py::_check_ceilings`, which takes D4's grandfather rule in its place;
  - `orchestrator/tests/test_merge_lane_ratchet.py::test_size_ceiling_exempt_is_exactly_git_ops`;
  - in `orchestrator/tests/test_merge_lane_ratchet_write_gate.py`,
    `test_the_git_ops_size_exemption_holds_on_the_write_path`,
    `test_size_ceiling_exemption_covers_git_ops_even_when_new` and
    `test_the_exemption_is_scoped_to_the_ceiling_not_to_the_ratchet`. Their contrast case
    `test_a_non_exempt_path_at_the_same_size_is_a_violation` is rewritten as "git_ops.py is held to the
    ceiling like any path".
  - the `GIT_OPS_RELPATH` comment in `orchestrator/tests/_merge_lane_ratchet_fixtures.py` (it calls the
    path "the one SIZE_CEILING_EXEMPT path");
  - `scripts/tests/test_scripts_governed_lists.py::SIZE_CEILING_EXEMPT_LIST`;
  - B4, added to ι's gate file;
  - `git_ops.py`'s baseline entry, regenerated at its measured size.

  Signal:
  - `git grep -n SIZE_CEILING_EXEMPT -- scripts orchestrator` is empty.
  - `scripts/exception_register.py --report` no longer lists the list.
  - With `git_ops.py`'s key removed from a copy of the baseline, `check_against_baseline` reports no
    `new_file_over_ceiling` for it.
  - B4 is green.

  Prereqs: ι.

### G7 walk (advisory at author time; re-run at decompose)

- INV-1 `contracts-machine-checked`: the layering and the no-façade rule are AST tests (B1, B11). The
  ceilings are the ratchet, not prose.
- INV-2 `structured-facts-at-failure`: unchanged. Moved raises keep their values. The new ceiling
  violation names the path, its lines and the ceiling (the existing `Violation` message).
- INV-5 `no-lockstep-duplication`: one guarded prune (`PoolStorageGuard.prune_registrations`); one
  `archaeology_blocks` definition shared with 5047; one instrument for the partition.
- INV-7 `holds-owned-and-bounded`: α's wait on S5 π is an owned dependency on a filed chain. No edge
  goes onto a low-priority pending task (5047, 6432). The decision-10-shaped gating holds pending tasks
  only, and an in-progress task with a claimant lands first.
- INV-9 `one-fact-one-home`: rationale goes to `docs/git-engine/decisions.md` with pointers. The
  heuristic-14 measurement lives in this PRD and the commit records, not in module docstrings. This
  PRD restates no heuristic.
- INV-10 `guards-exercise-behaviour`: ω's signal executes the ceiling check against a baseline without
  the key, rather than grepping for the list's absence alone. B4 drives the grandfather rule on fixtures.
- INV-11 `no-silent-fail-soft`: `concern_map.py` exits 2 on any partition or measurement failure, and
  a map that does not reconcile is an error.
- INV-12 `exceptions-owned-or-ratified`: the exemption's `Debt` owner is re-pointed at decompose (D11)
  and deleted by ω. β's authorized-raise entry names its task and each module's owning leaf. No new
  suppression.
- INV-13 `readers-prove-their-producer`: B3 reads the live `--report` over the real tree, not a
  seeded report.
- No waiver requested.

## Out of scope

- Any behaviour change to worktree creation, lane acquire or release, seeding, landing or queries.
- S5's own extractions, ports and contracts (C1–C8). They are consumed here, not redone.
- `advance_main`'s internal structure (task 5042).
- The merge lane's own over-ceiling files (`orchestrator/src/orchestrator/merge_lane/worker.py` and others). They are merge-lane-quality's.
  D4's rule applies to them, but splitting them is not this batch's work.
- The reify-side warm-lane shell scripts.
- Re-anchoring `plans/git-ops-split-prd.concern-map.md`. It is a dated snapshot. α writes a new one.

## Open questions (tactical)

1. **`create_worktree`'s home if `git_ops.py` measures over 1,500 − H after ζ.** The authoring bound is
   1,397 (§Contract → Ceilings), so this is a contingency, not a prediction. Suggested resolution: if
   ζ's measured `git_ops.py` exceeds 1,465 lines, ζ moves `create_worktree` and
   `_reattach_cold_worktree` (698 unit lines, 701 projected) into `task_worktrees.py::TaskWorktrees`,
   leaving 479 unit lines on `GitOps`. Measured acyclic: the pair refs only collaborators below it,
   and nothing refs it. Callers are repointed to `git_ops.task_worktrees.create(...)`. Decided by
   measurement in ζ.
2. **Collaborator attribute names on `GitOps`.** Suggested: the module's noun (`git_ops.spec_lanes`,
   `git_ops.lane_release`, `git_ops.merging`). Decide in each leaf, consistently with S5's
   `worktrees` / `landing` / `branches` / `queries`.
3. **Whether `lane_paths` exists.** α decides it from where S5 put the facts (D7, R6).

## Decompose record (2026-11-22, gate 11190)

Decomposed by the `/team` session for human gate 11190 (esc-11190-1) at main `2f18cea0d1`; the capability manifest
(`plans/git-ops-split-prd.capability-manifest.md`, sidecar beside it) was verified at `a4349b7f4b` and stamped by
`commit_planning`. Every leaf is `task_kind='normal'`, carries `x_finding_key = ["fk-c00c5f01be4c"]`,
`x_finding_run = "review-all-dark_factory-20261008"` and `source = "prd"`.

| Label | Task |
|---|---|
| α | 11411 |
| β | 11412 |
| γ | 11413 |
| δ0 | 11414 |
| δ | 11415 |
| ε | 11416 |
| ζ0 | 11417 |
| ζ | 11418 |
| η | 11419 |
| θ0 | 11420 |
| θ | 11421 |
| ι | 11422 |
| ω | 11423 |

**Codemod siblings (the §Decomposition plan's ~40-test-file rule, applied).** δ, ζ and θ named 80, 53 and 94 test files
on main `a4349b7f4b`, so each is filed as a codemod sibling (δ0, ζ0, θ0: `complexity='simple'` with
`force_full_path=true`, S5 ξ0's shape) plus the move leaf. The codemod creates the modules with the moved types and
one-step surfaces that call the not-yet-moved bodies through a Protocol-typed reference, composes them, repoints callers
and tests, and adds the modules to `CLUSTER_PATHS`; the move leaf moves the bodies and deletes the Protocol and the
back-reference. These counts are pre-S5 predictions: α re-enumerates the three caller sets. ε and η named 35 each; with
both D12 fallbacks η's union is about 56, so α's Correction says whether an η0 sibling is needed.

**Edges.** One total order α → β → γ → δ0 → δ → ε → ζ0 → ζ → η → θ0 → θ → ι → ω. α → S5 π (9275). The S9 λ1 (9246) edge
sits on δ0 and the 5042 edge on θ0, because the type and constant moves happen in the codemods. The trigger-chain gate
6698 depends on every `high` leaf: β, δ0, δ, ζ0, ζ, ι, ω.

**Decision-10 gating as executed.** 74 non-terminal tasks whose `metadata.files` (or `files_hint`) declares
`git_ops.py` or an S5 git-engine module gained a dependency on α (11411), computed from the store at wiring time.
Not gated: 26 tasks upstream of 9275 (gating them would close a cycle); 3105 (in progress with a live claimant); 10473
and 10991 (blocked with work in flight); by Leo's rulings on 2026-11-22, the pinned tasks 3928 (starvation-watchdog
pin), 6696, 8412, 10736 and 11050, and this batch's own prerequisites and composition site 5042, 9246 and 9218 (S8 β).
`shared/src/shared/git_async.py` is not a gated path: no leaf edits it. As with merge-lane-quality's ζ2, tasks filed
after this decompose are not gated automatically; until α lands, a new task declaring one of these files needs
`add_dependency(<id>, 11411)` by hand. Gated: 3039, 3047, 3156, 3267, 3354, 3356, 3652, 3665, 3855, 3856, 3971, 4138, 4196, 4197, 4210, 4323, 4330, 4420, 4606, 4746, 4799, 5144, 5293, 5297, 5298, 5300, 5358, 5360, 5493, 5584, 5621, 6124, 6194, 6242, 6246, 6257, 6432, 6522, 6554, 6589, 6697, 6727, 6855, 6892, 7031, 7053, 7069, 7087, 7318, 7369, 7458, 7540, 7691, 7766, 7767, 7842, 7858, 7908, 8266, 8392, 8605, 8631, 8788, 8803, 9107, 9277, 10099, 10265, 10474, 10717, 10842, 11199, 11203, 11263.

**Other decompose-time steps.** `Debt(TaskRef(7140))` in
`scripts/tests/test_scripts_governed_lists.py::SIZE_CEILING_EXEMPT_LIST` now names ω (11423) (D11). Task 6432
carries a dated note naming `git_queries.GitQueries` as its probes' home (D9). Six S5 delivered checks that pinned a
symbol this batch moves to `git_ops.py` (`from_seed_rc`, `adopt_lane`, `_classify_acquire_route`, `_ensure_pool_storage`,
`GitOpsCallbacks`, `GitOpsLimits`) were widened in the S5 sidecar and in the copied `metadata.delivered_checks` of
9259–9262, polarity unchanged.

**Corrections to this PRD's text found at decompose** (applied in the leaf briefs, not by editing the sections above):
- D4's "each extraction leaf adds its own module ... without a raise" holds for file lines only. A moved function
  arrives under a new baseline key, and any above `NEW_FUNCTION_COGNITIVE_CEILING` (15) is a
  `new_function_over_ceiling` raise (`scripts/merge_lane_ratchet.py::_check_ceilings`). Each moving leaf records it
  with `--authorize-raise` naming the function, its unchanged score and its owner.
- γ does not edit inside `advance_main` (task 5042 owns it, D9), so γ's signal is `archaeology_blocks = 0` outside
  `advance_main`.
- δ's, ε's and ζ's signals cite B5, B9 and B10 as they hold at each leaf's chain point (merge-worktree cleanup and
  rename/quarantine still live in S5's modules). ι owns the final-layout forms.
- α measures every class a later leaf splits (GitOps, WorktreeOps, GitQueries, TaskBranchRefs, MainLanding), not
  GitOps alone.
- D12 names no fallback for `main_landing` itself. If it measures over 1,465 after θ's planned splits, θ records the
  measurement and escalates rather than inventing a split.
