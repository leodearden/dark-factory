# Capability manifest — review-all-s5-gitops-ports-query-contract-and-the-git-runner-prd.md

Human-readable twin of `plans/review-all-s5-gitops-ports-query-contract-and-the-git-runner-prd.capability-manifest.yaml`
(the machine-readable sidecar `commit_planning` stamps). Bindings verified at main `ce0f71a384` (2026-11-01),
re-run by the folds at `d40ba048e9` and `b688221a04` (2026-11-02) and by the post-sitting ruling fold at
`e634cfe658` (2026-11-10) and by the fold-verify at `bb1c659562` (2026-11-10): every producer `present` check is
absent at that tree and every producer `absent` check is present there (the polarity rule of
`shared/delivered_check_polarity.py`, run through `lint_delivered_checks`); the three substrate checks pass
today and say so. 73 grep checks (70 producer, 3 substrate), 3 path checks, 1 manual, no script check (a
`kind: script` check may only name a script already on main, and π's script does not exist yet). The sidecar
loads through `shared.capability_manifest.CapabilityManifestDoc` (23 blocks). λ0 is listed second, its place
in the chain under the lead's sequencing ruling of 2026-11-10; σ is the OQ-5 follow-up, not a signal leaf. Checks are single-line and
symbol-anchored; a check whose symbol a later leaf moves (§Sketch placement table: `_move_main`,
`AdvanceDetail`, `worktree_work`, `branch_work`, `count_commits`, the retired predicates and parsers) is
anchored to the package directory, not the file. Contract ids `C1`–`C8` are the PRD's. `signal_leaves: [π]`.
Every design question is ruled (PRD §Rulings); the ρ0/ρ blocks stand under ruling OQ-9 (A).

| Leaf | Capability | Binding | Verdict |
|---|---|---|---|
| α | `git-runner-protocol-published` | `^class GitRunner\(Protocol\)` present in `shared/src/shared/git_async.py` | PASS |
| α | `worktree-run-leaf-exists` | `^async def run_in_worktree\(` present in `orchestrator/src/orchestrator/git_run.py` | PASS |
| α | `no-external-private-run-import` | `from orchestrator\.git_ops import[^#]*\b_run\b\|^\s+_run,\s*$\|(= \|await )git_ops(_module)?\._run\b` absent in `orchestrator/src`, `:(exclude)orchestrator/src/orchestrator/git_ops.py` | PASS |
| α | `git-query-error-single-name` | `^class GitQueryError` present in `orchestrator/src/orchestrator/git_run.py` | PASS |
| α | `git-query-failed-retired` | `GitQueryFailed` absent in `orchestrator/src/orchestrator/`, `scripts/` | PASS |
| α | `substrate-run-git-exists` | `^async def run_git\(` present in `shared/src/shared/git_async.py` | PASS (substrate) |
| λ0 | `queued-branch-shared-home` | `^class QueuedBranch` present in `shared/src/shared/branch_names.py` | PASS |
| λ0 | `queued-branch-old-home-gone` | `^class QueuedBranch` absent in `orchestrator/src/orchestrator/merge_lane/types.py` | PASS |
| β0 | `test-git-helper-exists` | `^async def run\(` present in `orchestrator/tests/_git.py` | PASS |
| β0 | `no-test-imports-private-run` | `from orchestrator\.git_ops import[^#]*\b_run\b` absent in `orchestrator/tests/`, `escalation/tests/` | PASS |
| β | `module-run-deleted` | `^async def _run\(` absent in `orchestrator/src/orchestrator/git_ops.py` | PASS |
| β | `gitops-runner-port` | `runner: GitRunner` present in `orchestrator/src/orchestrator/git_ops.py` | PASS |
| β | `scripted-runner-fake-exists` | `^class ScriptedGitRunner` present in `orchestrator/tests/_git_runner_fakes.py` | PASS |
| β | `path-shim-fake-retired` | `orchestrator/tests/_git_fault_shim.py` absent (path check) | PASS |
| γ | `warm-lane-scripts-port` | `^class WarmLaneScripts` present in `orchestrator/src/orchestrator/warm_lane_scripts.py` | PASS |
| γ | `callbacks-bundle` | `^class GitOpsCallbacks` present in `orchestrator/src/orchestrator/git_ops.py` | PASS |
| γ | `limits-dataclass` | `^class GitOpsLimits` present in `orchestrator/src/orchestrator/git_ops.py` | PASS |
| γ | `env-override-retired` | `ORCH_WARM_LANE_SCRIPT_DIR` absent in `orchestrator/src/orchestrator/` | PASS |
| γ | `reexport-shim-retired` | `re-export shim \(test_pool_storage_guard` absent in `orchestrator/src/orchestrator/git_ops.py` | PASS |
| γ | `post-construction-installs-retired` | `git_ops\.[a-zA-Z_]+ *= [^=]` absent in `orchestrator/src/orchestrator/`, `escalation/src/escalation/` | PASS |
| δ | `disposition-leaf-exists` | `^class BlockDisposition` present in `orchestrator/src/orchestrator/block_disposition.py` | PASS |
| δ | `enum-maps-seed-rc` | `def from_seed_rc\(` present in `orchestrator/src/orchestrator/git_ops.py` | PASS |
| δ | `if-chain-deleted` | `pool_info is WarmLaneUnavailable\.` absent in `orchestrator/src/orchestrator/git_ops.py` | PASS |
| δ | `steal-retryable-placeholder-deleted` | `_STEAL_RETRYABLE` absent in `orchestrator/src/orchestrator/git_ops.py` | PASS |
| δ | `workflow-types-rows-derive` | `WarmLane[A-Za-z]* *: *BlockDisposition\(` absent in `orchestrator/src/orchestrator/workflow_types.py` | PASS |
| ι | `lifecycle-release-method` | `def release\(` present in `orchestrator/src/orchestrator/lane_lifecycle.py` | PASS |
| ι | `pool-set-lifecycle-retired` | `def set_lane_lifecycle\(` absent in `orchestrator/src/orchestrator/warm_lane_pool.py` | PASS |
| ι | `no-lifecycle-reach` | `git_ops\._lane_lifecycle` absent in `orchestrator/src/orchestrator/` | PASS |
| ι | `adopt-lane-owner` | `def adopt_lane\(` present in `orchestrator/src/orchestrator/git_ops.py` | PASS |
| ι | `pool-slot-state-renamed` | `^class PoolSlotState` present in `orchestrator/src/orchestrator/warm_lane_pool.py` | PASS |
| ι | `substrate-lifecycle-exists` | `^LEGAL_TRANSITIONS` present in `orchestrator/src/orchestrator/lane_lifecycle.py` | PASS (substrate) |
| κ | `pool-storage-guard` | `def _ensure_pool_storage\(` present in `orchestrator/src/orchestrator/git_ops.py` | PASS |
| κ | `route-classifier` | `def _classify_acquire_route\(` present in `orchestrator/src/orchestrator/git_ops.py` | PASS |
| κ | `substrate-acquire-route-enum` | `^class AcquireRoute` present in `orchestrator/src/orchestrator/lane_lifecycle.py` | PASS (substrate) |
| ε | `git-command-failed-exists` | `^class GitCommandFailed\(GitQueryError\)` present in `orchestrator/src/orchestrator/git_run.py` | PASS |
| ε | `raising-query-helper` | `^async def query\(` present in `orchestrator/src/orchestrator/git_run.py` | PASS |
| ε | `work-state-query` | `async def worktree_work\(` present in `orchestrator/src/orchestrator/` | PASS |
| ε | `branch-work-query` | `async def branch_work\(` present in `orchestrator/src/orchestrator/` | PASS |
| ε | `rc-dropping-predicates-retired` | `def has_dirty_working_tree\(` absent in `orchestrator/src/orchestrator/` | PASS |
| ε | `detect-unmerged-paths-retired` | `def _detect_unmerged_paths\(` absent in `orchestrator/src/orchestrator/` | PASS |
| ζ | `typed-diff-reader` | `^class FileDelta` present in `orchestrator/src/orchestrator/git_diff.py` | PASS |
| ζ | `unguarded-parser-retired` | `^def parse_diff_line_ranges\(` absent in `orchestrator/src/orchestrator/` | PASS |
| ζ | `worktree-list-record` | `^class WorktreeEntry` present in `orchestrator/src/orchestrator/worktree_list.py` | PASS |
| ζ | `one-commit-count` | `async def count_commits\(` present in `orchestrator/src/orchestrator/` | PASS |
| η | `advance-detail-type` | `^(class AdvanceDetail\|AdvanceDetail *(: *TypeAlias *)?=)` present in `orchestrator/src/orchestrator/` | PASS |
| η | `last-side-channel-deleted` | `self\._last_overlap_files` absent in `orchestrator/src/orchestrator/` | PASS |
| η | `gates-read-detail` | `= getattr\(git_ops, '_last_` absent in `orchestrator/src/orchestrator/merge_lane/gates.py` | PASS |
| η | `move-main-primitive` | `async def _move_main\(` present in `orchestrator/src/orchestrator/` | PASS |
| θ | `lane-lease-exists` | `^async def lane_lease\(` present in `orchestrator/src/orchestrator/lane_lease.py` | PASS |
| θ | `span-lock-in-lease-module` | `^def span_lock\(` present in `orchestrator/src/orchestrator/lane_lease.py` | PASS |
| θ | `compat-colock-renamed` | `compat_fd` absent in `orchestrator/src/orchestrator/cli.py` | PASS |
| θ | `old-binary-prose-gone` | `in-flight OLD` absent in `orchestrator/src/orchestrator/cli.py`, `orchestrator/src/orchestrator/git_ops.py` | PASS |
| λ | `task-branch-owner` | `^class TaskBranchRefs` present in `orchestrator/src/orchestrator/task_branch.py` | PASS |
| λ | `onto-ref-sum` | `^OntoRef = ` present in `orchestrator/src/orchestrator/task_branch.py` | PASS |
| λ | `try-both-resolver-deleted` | `def resolve_queued_branch_ref\(` absent in `orchestrator/src/orchestrator/` | PASS |
| ξ0 | `worktree-ops-module` | `^class WorktreeOps` present in `orchestrator/src/orchestrator/worktrees.py` | PASS |
| ξ0 | `worktree-kind-moved` | `^class WorktreeKind` present in `orchestrator/src/orchestrator/worktrees.py` | PASS |
| ξ | `registration-probe-moved` | `def _is_registered_worktree\(` absent in `orchestrator/src/orchestrator/git_ops.py` | PASS |
| ξ | `ephemeral-body-moved` | `async def ephemeral_worktree\(` absent in `orchestrator/src/orchestrator/git_ops.py` | PASS |
| ξ | `retired-guard-prose-gone` | `No \.task-specific guards remain` absent in `orchestrator/src/orchestrator/git_ops.py` | PASS |
| ο0 | `main-landing-module` | `^class MainLanding` present in `orchestrator/src/orchestrator/main_landing.py` | PASS |
| ο0 | `advance-outcome-moved` | `^class AdvanceOutcome` present in `orchestrator/src/orchestrator/main_landing.py` | PASS |
| ο | `advance-main-moved` | `async def advance_main\(` absent in `orchestrator/src/orchestrator/git_ops.py` | PASS |
| ρ0 | `git-queries-module` | `^class GitQueries` present in `orchestrator/src/orchestrator/git_queries.py` | PASS |
| ρ | `diff-query-moved` | `async def describe_commit_effect_in_main\(` absent in `orchestrator/src/orchestrator/git_ops.py` | PASS |
| μ | `probes-on-run-git` | `create_subprocess_exec` absent in `fused-memory/src/fused_memory/middleware/task_interceptor.py` | PASS |
| μ | `probes-import-shared-runner` | `^from shared\.git_async import` present in `fused-memory/src/fused_memory/middleware/` | PASS |
| μ | `git-runner-threaded` | `git_runner: ` present in `fused-memory/src/fused_memory/middleware/` | PASS |
| ν | `typed-read-seam` | `def _read_model\(` present in `orchestrator/src/orchestrator/artifacts.py` | PASS |
| ν | `typed-architect-exits` | `^class (BlockingDependency\|AlreadyDone\|ReadyToMerge\|UnactionableTask\|FalsePremise)Exit` present in `orchestrator/src/orchestrator/artifacts.py` | PASS |
| ν | `module-docstring-truthful` | `Manages the \.task/ directory structure in each worktree` absent in `orchestrator/src/orchestrator/artifacts.py` | PASS |
| π | `private-patch-measure-script` | `scripts/check_gitops_private_patch_targets.py` exists (path check) | PASS |
| π | `no-dotted-private-gitops-patches` | `orchestrator\.git_ops\.(GitOps\.)?_[A-Za-z]` absent in `orchestrator/tests/`, `escalation/tests/` | PASS |
| π | `boundary-gate-exists` | `orchestrator/tests/test_gitops_boundary_gate.py` exists (path check) | PASS |
| π | `heuristic-14-measured` | manual — heuristic-14 residual and named partition in the landing commit | PASS (manual) |
| σ | `lane-lease-token` | `^class LaneLeaseToken` present in `orchestrator/src/orchestrator/lane_lease.py` | PASS |
| σ | `span-lock-file-retired` | `= '\.merge_verify\.lock'` absent in `orchestrator/src/orchestrator/` | PASS |

Polarity notes (re-run at `e634cfe658`). `def release(` matches only `LaneLifecycle.release` once it exists
(no other `release` definition in `lane_lifecycle.py` today). α carries the `GitQueryError` name because
tasks 7485 and 7332 landed the type as `GitQueryFailed` after the fold; ε's checks are the structured
`GitCommandFailed` and `git_run.query`. ρ's check moved from `get_diff_from_main`, which task 5695 deleted
(the old check passed at `e634cfe658`), to `describe_commit_effect_in_main`, a cluster (f) method that holds
`_effect_probe_memo`. α's import check uses `:(exclude)` because
`shared/delivered_check_polarity.py::build_grep_argv` passes pathspecs verbatim; it leaves out `escalation/src`
because `release_warm_worktree`'s repoint is S3 δ's, and covers all 13 orchestrator sites (the author's check
covered 1 of 10 importers). β0's import check covers the single-line form (124 files, measured at `e634cfe658`); the 11 multi-line
importers are caught by β's deletion of `_run`, which fails test collection for any import left behind. θ has no `compat-colock-dropped` check: ruling OQ-5
(B′) keeps the lock as the span lock. The follow-up σ (A′) retires it: its absent check anchors on the one
assignment of the file name (`verify_cancel.py::MERGE_VERIFY_LOCK_FILENAME`, 1 hit at `bb1c659562`), which θ
keeps because the lock file keeps its name; the operator messages naming the file are σ's to reword, not
its check's. π's measure of object-form patch targets is a script it creates;
because a `kind: script` check may only name a script already on main, π's block carries a path check for
the script plus a grep for the dotted string form (`orchestrator\.git_ops\._`, 24 test files today), and
π's done-criterion (B28) runs the script. `scripts/merge_lane_metrics.py::LANE_PACKAGE` measures only
`orchestrator.merge_lane`. The `π` manual check is the heuristic-14 measurement the Comments stance puts in
the commit record, not a grep.
