# git_ops.py concern map — dated snapshot for `plans/git-ops-split-prd.md`

**Measured against main `c787af9409` (2026-11-21).** This is a dated snapshot: never re-anchor or
regenerate it in place. The PRD's α leaf re-measures post-S5 main against the same partition file and
records what changed as a dated Correction in the PRD, beside this file, not in it.

## Reproduction

```
mkdir -p /tmp/git-ops-asof
git archive c787af9409 orchestrator/src/orchestrator/git_ops.py | tar -x -C /tmp/git-ops-asof
python3 scripts/concern_map.py orchestrator/src/orchestrator/git_ops.py --class GitOps \
  --root /tmp/git-ops-asof --partition plans/git-ops-split-prd.partition.json --format markdown
```

The command prints everything below the rule, byte for byte. `--format json` gives the same
measurement as data.

## How the partition was built

Every unit of `git_ops.py` is assigned to exactly one cluster (the tool refuses a partition that is
not). Cluster names carry their owner:

- `s5/<module>`: a module that S5 (`plans/review-all-s5-gitops-ports-query-contract-and-the-git-runner-prd.md`,
  tasks 9255–9277) creates and moves this code into.
- `s5-split/<module>`: a piece of an S5 module that is still over the 1,500-line ceiling, which this
  PRD splits out after S5 lands.
- `s5-unplaced/lane_paths`: units whose S5 placement would make a cycle (rule R6).
- `residual/<module>`: S5's residual, meaning cluster (c) plus the non-class names S5 keeps in `git_ops.py`.
  This PRD splits it into these modules. `residual/git_ops` is what stays in `git_ops.py`: the
  composition root, `create_worktree` and the thin uses that C7 keeps on `GitOps`.
- `per-module/logger`: `logger`, which every module declares for itself. Its edges carry no design meaning.

Rules used to reconstruct S5's clusters, in precedence order:

- **R1.** A unit that S5's text places explicitly goes where S5 puts it: the §Sketch placement table,
  D14, contracts C2/C7/C8, and the θ, λ, ξ, ο, ρ and γ leaf texts.
- **R2.** Where S5 is silent, S5's own rules apply. A method follows the state it reads and writes (the
  §Sketch self-state rule). A module-level unit follows its users, and when several clusters use it, it
  goes with the lowest one (the placement table's "a type moves with the first leaf whose module needs it").
- **R3.** Where S5's leaf text and its constructor contract disagree, the contract wins. C2 builds
  `WarmLaneScripts(project_root, config, runner, script_dir)`, which holds no pool, lifecycle or lease.
  So `_seed_warm_lane`, `thin_warm_lane`, `_run_warm_lane_gc_reclaim`, `refresh_warm_base` and
  `_warm_lane_audit_cached` stay in the residual, and only their script invocations move. C7 keeps
  `GitOps.merge_verify_lease` and `GitOps.task_verify_lease` as thin uses of `lane_lease`. They sit in
  `residual/git_ops` at today's size (218 + 81 lines), which is an upper bound.
- **R4.** A unit that an S5 leaf deletes or replaces goes to that leaf's cluster: `_run` and `_git_query`
  (α, β), and `_seed_rc_to_unavailable` and `_STEAL_RETRYABLE` (δ, which makes them member data of
  `WarmLaneUnavailable`, so they go with that type).
- **R5.** `_warm_lane_base_resolvable` goes with `WarmBaseHealth` in `s5/warm_lane_scripts`, because C2
  makes it the default `WarmBaseProbe`. Worktrees and the pool-storage guard call it.
- **R6.** Where an explicit S5 placement would close a cycle, the unit is regrouped to its lowest user
  and the conflict is recorded. There are three:
  - `PERSISTENT_MERGE_WORKTREE_NAME` → S5 places it in `main_landing`. But worktrees' `PROTECTED_PREFIXES`
    needs it, and so do the persistent-lane path methods that the lease thin use, lane seeding and the
    pool-storage guard call. That would put `s5/worktrees` and `s5/main_landing` in a cycle. It moves
    with those five methods to `s5-unplaced/lane_paths`.
  - `ReplayMatch` → S5 places it in `task_branch`. But `RebasedCitation` and `GitOps._find_replay`, both
    in `git_queries`, need it, and `task_branch` refs `git_queries` 19 times. That would put
    `s5/task_branch` and `s5/git_queries` in a cycle. It moves to `s5/git_queries`.
  - `GitOps._rename_branch_aside` → λ routes its ref writes through the branch owner, but it frees a pool
    checkout (`_free_lane_checkout`). It stays in `residual/lane_release`.

The residual was split by what each method owns and touches: the collaborator it reads, the counters
and caches it writes, and its ref neighbourhood. It was never split by topic or name prefix.
Every candidate tried and the symptom each rejected one hit are recorded in the PRD's §Background
("Heuristic-14 measurement").

Three edges run from S5 modules into the residual. These are the "Cross-cluster edges" rows whose
target starts `residual/`:

- `_prune_registrations` (`residual/pool_storage`): 7 refs from `s5/worktrees`, `s5/task_branch` and
  `s5-split/merge_worktrees`.
- `_seed_warm_lane` (`residual/lane_seeding`): 2 refs from `s5/worktrees`.
- `WorktreeInfo` (`residual/lane_outcomes`): 1 ref from `s5/task_branch`.

So those three residual modules must sit below S5's modules, and they are the PRD's first extractions.

Every shared write below pairs `GitOps.__init__` with exactly one other cluster. Each resolves the
same way: the owning collaborator's constructor initialises the attribute, so each attribute has
one writer module.

---

# Concern map: `orchestrator/src/orchestrator/git_ops.py`, class `GitOps`

17010 lines: 16266 in 342 units, and the residual buckets module_preamble 136, class_non_method 281, module_other 0, inter_unit_gaps 327.

## Units

| unit | kind | lines | prose | writes | reads | refs | cluster |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| logger | module | 1 | 0 |  |  |  | per-module/logger |
| AdvanceResult | module | 9 | 2 |  |  |  | s5/main_landing |
| AdvanceOutcome | module | 20 | 14 |  |  | AdvanceResult | s5/main_landing |
| PushResult | module | 1 | 0 |  |  |  | s5/main_landing |
| RecoverResult | module | 3 | 2 |  |  |  | s5/main_landing |
| ReplayMatch | module | 3 | 2 |  |  |  | s5/git_queries |
| RebasedCitation | module | 5 | 1 |  |  | ReplayMatch | s5/git_queries |
| RemovalOutcome | module | 35 | 31 |  |  |  | s5/worktrees |
| WIP_SAFETY_COMMIT_PREFIXES | module | 7 | 6 |  |  |  | s5/task_branch |
| is_wip_safety_commit | module | 9 | 7 |  |  | WIP_SAFETY_COMMIT_PREFIXES | s5/task_branch |
| MERGE_PARK_REF | module | 7 | 6 |  |  |  | s5/main_landing |
| _INDEX_LOCK_POLL_INTERVAL_S | module | 6 | 5 |  |  |  | s5/main_landing |
| _INDEX_LOCK_WARN_INTERVAL_S | module | 1 | 0 |  |  |  | s5/main_landing |
| _INDEX_LOCK_STALE_FLOOR_S | module | 19 | 18 |  |  |  | s5/main_landing |
| _WORKTREE_ADD_MAX_ATTEMPTS | module | 3 | 2 |  |  |  | s5/worktrees |
| _worktree_add_failure_is_retryable | module | 16 | 14 |  |  |  | s5/worktrees |
| MergeParkError | module | 11 | 10 |  |  |  | s5/main_landing |
| MergeParkLockContentionError | module | 19 | 18 |  |  | MergeParkError | s5/main_landing |
| MergeParkContentionError | module | 9 | 8 |  |  | MergeParkError | s5/main_landing |
| TrainMembership | module | 9 | 5 |  |  |  | s5/task_branch |
| _is_train_successor | module | 4 | 2 |  |  | TrainMembership | s5/task_branch |
| TrainPredecessor | module | 5 | 1 |  |  |  | s5/task_branch |
| TrainStackResult | module | 19 | 14 |  |  |  | s5/task_branch |
| DEFAULT_COMMIT_CITATION_PATTERN | module | 81 | 74 |  |  |  | s5/git_queries |
| render_citation_pattern | module | 8 | 1 |  |  |  | s5/git_queries |
| task_citation_pattern | module | 21 | 7 |  |  | CitationPatternInvalid, DEFAULT_COMMIT_CITATION_PATTERN, render_citation_pattern | s5/git_queries |
| PERSISTENT_MERGE_WORKTREE_NAME | module | 4 | 3 |  |  |  | s5-unplaced/lane_paths |
| _MERGE_VERIFY_LEASE_WAIT_SECS | module | 20 | 19 |  |  |  | s5/lane_lease |
| _TASK_VERIFY_LEASE_WAIT_SECS | module | 18 | 17 |  |  |  | s5/lane_lease |
| _SEED_WARM_LANE_LOCK_WAIT_SECS | module | 24 | 23 |  |  |  | residual/lane_seeding |
| _RESET_WARM_LANE_LOCK_WAIT_SECS | module | 29 | 28 |  |  |  | s5-split/merge_worktrees |
| _SEED_WARM_LANE_LOCK_TIMEOUT_RC | module | 9 | 8 |  |  |  | residual/lane_outcomes |
| _SEED_ASSUME_LANE_LOCK_HELD_FLAG | module | 5 | 4 |  |  |  | s5/warm_lane_scripts |
| _WARM_LANE_STEAL_MAX_ATTEMPTS | module | 18 | 17 |  |  |  | residual/warm_acquire |
| _STEAL_RETRYABLE | module | 23 | 19 |  |  | WarmLaneUnavailable | residual/lane_outcomes |
| _SEED_DISTINCT_LOCK_REFUSAL_RC_FLAG | module | 6 | 5 |  |  |  | s5/warm_lane_scripts |
| _DF_WARM_LANE_SCRIPT_DIR | module | 21 | 18 |  |  |  | s5/warm_lane_scripts |
| _DF_WARM_LANE_SCRIPT_DIR_ENV | module | 9 | 8 |  |  |  | s5/warm_lane_scripts |
| _df_warm_lane_script_dir | module | 12 | 7 |  |  | _DF_WARM_LANE_SCRIPT_DIR, _DF_WARM_LANE_SCRIPT_DIR_ENV | s5/warm_lane_scripts |
| _seed_script_text | module | 29 | 19 |  |  | logger | s5/warm_lane_scripts |
| _seed_script_supports | module | 24 | 14 |  |  | _seed_script_text, logger | s5/warm_lane_scripts |
| _seed_script_supports_assume_lane_lock_held | module | 3 | 1 |  |  | _SEED_ASSUME_LANE_LOCK_HELD_FLAG, _seed_script_supports | s5/warm_lane_scripts |
| _seed_script_supports_distinct_lock_refusal_rc | module | 3 | 1 |  |  | _SEED_DISTINCT_LOCK_REFUSAL_RC_FLAG, _seed_script_supports | s5/warm_lane_scripts |
| _WARM_LANE_AUDIT_CACHE_TTL_SECS | module | 9 | 8 |  |  |  | residual/lane_admission |
| _STRUCTURAL_EXHAUSTION_L2_REFIRE_EVERY | module | 16 | 15 |  |  |  | residual/warm_acquire |
| PERSISTENT_OFFLINE_DEEP_WORKTREE_NAME | module | 7 | 6 |  |  |  | s5-unplaced/lane_paths |
| WorktreeKind | module | 14 | 10 |  |  |  | s5/worktrees |
| PROTECTED_PREFIXES | module | 26 | 13 |  |  | PERSISTENT_MERGE_WORKTREE_NAME, PERSISTENT_OFFLINE_DEEP_WORKTREE_NAME, WorktreeKind | s5/worktrees |
| PROTECT_GLOB_OWNED_POOL_BANDS | module | 11 | 10 |  |  |  | s5/worktrees |
| default_protected_prefixes | module | 38 | 31 |  |  | PROTECTED_PREFIXES | s5/worktrees |
| render_protect_glob | module | 40 | 28 |  |  | default_protected_prefixes | s5/worktrees |
| WorktreeClass | module | 2 | 1 |  |  |  | s5/worktrees |
| classify_worktree_entry | module | 44 | 38 |  |  | WorktreeClass | s5/worktrees |
| _assert_no_conflict_markers | module | 54 | 33 |  |  | _run, logger | s5/main_landing |
| MergeResult | module | 21 | 11 |  |  |  | s5-split/merging |
| WarmBaseHealth | module | 24 | 20 |  |  |  | s5/warm_lane_scripts |
| SeedLaneLock | module | 17 | 13 |  |  |  | s5/warm_lane_scripts |
| WarmLaneUnavailable | module | 86 | 75 |  |  |  | residual/lane_outcomes |
| _seed_rc_to_unavailable | module | 51 | 41 |  |  | WarmLaneUnavailable, _SEED_WARM_LANE_LOCK_TIMEOUT_RC | residual/lane_outcomes |
| WorktreeInfo | module | 15 | 10 |  |  |  | residual/lane_outcomes |
| PoolPrewarmResult | module | 34 | 27 |  |  |  | residual/lane_outcomes |
| _INTERACTIVE_SLUG_RE | module | 9 | 8 |  |  |  | s5/worktrees |
| InteractiveWorktreeInfo | module | 23 | 17 |  |  |  | s5/worktrees |
| ReapedInteractiveWorktree | module | 23 | 17 |  |  |  | s5/worktrees |
| _EFFECT_SURVIVAL_AGGREGATE_THRESHOLD | module | 4 | 3 |  |  |  | s5-split/commit_effect |
| _EFFECT_SURVIVAL_PER_FILE_THRESHOLD | module | 7 | 6 |  |  |  | s5-split/commit_effect |
| _EFFECT_SURVIVAL_PER_FILE_MIN_ADDED_LINES | module | 6 | 5 |  |  |  | s5-split/commit_effect |
| _EMPTY_TREE_SHA | module | 4 | 3 |  |  |  | s5-split/commit_effect |
| _EFFECT_PROBE_MEMO_MAX_ENTRIES | module | 7 | 6 |  |  |  | s5-split/commit_effect |
| _EFFECT_PROBE_TRANSIENT_FAILURES | module | 14 | 7 |  |  |  | s5-split/commit_effect |
| _MERGE_MARKER_INDEX_RECHECK_SECS | module | 18 | 17 |  |  |  | s5/task_branch |
| _MERGE_MARKER_LOG_FORMAT | module | 2 | 1 |  |  |  | s5/task_branch |
| _merge_markers_by_branch | module | 18 | 10 |  |  |  | s5/task_branch |
| CommitEffectProbe | module | 133 | 119 |  |  |  | s5-split/commit_effect |
| ConflictProbe | module | 22 | 18 |  |  |  | s5/git_queries |
| GitQueryFailed | module | 2 | 1 |  |  |  | s5/git_run |
| CitationPatternInvalid | module | 8 | 1 |  |  |  | s5/git_queries |
| _none_if_query_failed | module | 6 | 1 |  |  | GitQueryFailed | s5/git_run |
| WorktreeMissing | module | 14 | 9 |  |  |  | s5/git_run |
| InteractiveWorktreeLimitError | module | 18 | 17 |  |  |  | s5/worktrees |
| EphemeralWorktreeError | module | 12 | 11 |  |  |  | s5/worktrees |
| WarmLaneRequeue | module | 28 | 27 |  |  |  | residual/lane_outcomes |
| WarmLanePoolExhausted | module | 5 | 4 |  |  | WarmLaneRequeue | residual/lane_outcomes |
| WarmLaneSoftPressure | module | 30 | 29 |  |  | WarmLaneRequeue | residual/lane_outcomes |
| WarmLaneDiskPressure | module | 6 | 5 |  |  | WarmLaneRequeue | residual/lane_outcomes |
| WarmLanePoolHardDown | module | 16 | 15 |  |  | WarmLaneRequeue | residual/lane_outcomes |
| WarmLaneReseedContaminated | module | 23 | 22 |  |  | WarmLaneRequeue | residual/lane_outcomes |
| WarmLaneLockTimeout | module | 27 | 26 |  |  | WarmLaneRequeue | residual/lane_outcomes |
| WarmLaneStealFailed | module | 32 | 31 |  |  | WarmLaneRequeue | residual/lane_outcomes |
| WarmLaneLockContention | module | 20 | 19 |  |  | WarmLaneRequeue | residual/lane_outcomes |
| WorktreeConflictError | module | 30 | 19 |  |  |  | s5/worktrees |
| BranchResetError | module | 63 | 34 |  |  |  | s5/task_branch |
| _ParkedBranchStranded | module | 9 | 1 |  |  |  | s5/worktrees |
| _stranded_branch | module | 3 | 1 |  |  | _ParkedBranchStranded | s5/worktrees |
| MergeVerifyLeaseHeld | module | 27 | 17 |  |  |  | s5/lane_lease |
| MergeVerifyLeaseContended | module | 92 | 67 |  |  |  | s5/lane_lease |
| LaneLockSelfOwnedLeak | module | 95 | 63 |  |  | MergeVerifyLeaseContended | s5/lane_lease |
| _HELD_LANE_LOCK_FDS | module | 5 | 4 |  |  |  | s5/lane_lease |
| _HELD_LANE_LOCK_FDS_LOCK | module | 5 | 4 |  |  |  | s5/lane_lease |
| _lane_lock_identity | module | 15 | 9 |  |  |  | s5/lane_lease |
| _register_held_lane_lock | module | 17 | 9 |  |  | _HELD_LANE_LOCK_FDS, _HELD_LANE_LOCK_FDS_LOCK, _lane_lock_identity | s5/lane_lease |
| _release_and_forget_held_lane_lock | module | 33 | 29 |  |  | _HELD_LANE_LOCK_FDS, _HELD_LANE_LOCK_FDS_LOCK | s5/lane_lease |
| _lane_lock_held_in_process | module | 22 | 15 |  |  | _HELD_LANE_LOCK_FDS, _HELD_LANE_LOCK_FDS_LOCK, _lane_lock_identity | s5/lane_lease |
| _LANE_LOCK_HOLDER_SETTLE_SECS | module | 8 | 7 |  |  |  | s5/lane_lease |
| _LANE_LOCK_HOLDER_SETTLE_INTERVAL_SECS | module | 4 | 3 |  |  |  | s5/lane_lease |
| _settled_lane_lock_holder_pids | module | 157 | 138 |  |  | _LANE_LOCK_HOLDER_SETTLE_INTERVAL_SECS, _LANE_LOCK_HOLDER_SETTLE_SECS, _lane_lock_identity | s5/lane_lease |
| _lane_lock_holder_facts | module | 48 | 29 |  |  |  | s5/lane_lease |
| _utc_stamp | module | 3 | 1 |  |  |  | s5/worktrees |
| _run | module | 47 | 35 |  |  | WorktreeMissing | s5/git_run |
| _git_query | module | 8 | 1 |  |  | GitQueryFailed, _run | s5/git_run |
| _FULL_OID | module | 1 | 0 |  |  |  | s5/git_queries |
| _git_clean_failure_is_benign | module | 28 | 20 |  |  |  | s5/worktrees |
| _git_stderr_is_unresolved_ref | module | 37 | 24 |  |  |  | s5/git_queries |
| _merge_worktree_remove_argv | module | 8 | 6 |  |  |  | s5-split/merge_worktrees |
| _WHOLE_FILE_SENTINEL | module | 4 | 3 |  |  |  | s5/git_diff |
| parse_diff_line_ranges | module | 73 | 36 |  |  | _WHOLE_FILE_SENTINEL | s5/git_diff |
| parse_diff_added_line_ranges | module | 63 | 33 |  |  |  | s5/git_diff |
| GitOps.__init__ | method | 150 | 98 | _consecutive_exhausted, _effect_probe_memo, _interactive_wt_lock, _lane_lifecycle, _lane_liveness, _merge_marker_index, _merge_marker_index_checked_at, _merge_marker_index_sha, _merge_spec_warm_lane_pool_size, _on_pool_storage_absent, _on_structural_exhaustion, _spec_wt_create_lock, _warm_lane_audit_cache, _warm_lane_pool_size, config, project_root, spec_warm_lane_pool, transcript_archive, warm_lane_dispatched_predicate, warm_lane_pool, warm_lane_reclaim_candidate_provider, worktree_base | _lane_lifecycle, warm_lane_pool, worktree_base | CommitEffectProbe, GitOps.quarantine_worktree | residual/git_ops |
| GitOps.protected_prefixes | method | 18 | 16 |  | config | default_protected_prefixes | s5/worktrees |
| GitOps._refuse_foreign_band | method | 58 | 26 |  | worktree_base | GitOps.protected_prefixes, logger | s5/worktrees |
| GitOps.refuse_foreign_band | method | 17 | 13 |  |  | GitOps._refuse_foreign_band | s5/worktrees |
| GitOps._worktree_add_with_retry | method | 72 | 49 |  | project_root | _WORKTREE_ADD_MAX_ATTEMPTS, _run, _worktree_add_failure_is_retryable, logger | s5/worktrees |
| GitOps.ephemeral_worktree | method | 172 | 119 |  | worktree_base | EphemeralWorktreeError, GitOps._remove_ephemeral_worktree, GitOps._warm_seed_ephemeral_worktree, GitOps._worktree_add_with_retry, WorktreeKind, logger | s5/worktrees |
| GitOps._warm_seed_ephemeral_worktree | method | 29 | 4 |  |  | GitOps._seed_warm_lane, GitOps._warm_lane_base_resolvable, SeedLaneLock, WarmBaseHealth, WorktreeKind, logger | s5/worktrees |
| GitOps._remove_ephemeral_worktree | method | 20 | 1 |  | project_root | WorktreeKind, _run, logger | s5/worktrees |
| GitOps.pool_in_use | method | 14 | 12 |  | spec_warm_lane_pool, warm_lane_pool |  | residual/pool_storage |
| GitOps.pool_storage_present | method | 10 | 8 |  | _lane_lifecycle |  | residual/pool_storage |
| GitOps.mark_pool_storage_present | method | 16 | 14 |  | _lane_lifecycle |  | residual/pool_storage |
| GitOps._note_pool_storage_absent | method | 30 | 21 |  | _on_pool_storage_absent | logger | residual/pool_storage |
| GitOps._pool_storage_bootstrap_ok | method | 55 | 48 |  | worktree_base | GitOps._warm_lane_base_resolvable, GitOps.warm_lane_base_target_path, WarmBaseHealth | residual/pool_storage |
| GitOps._reconcile_pool_storage_before_sweep | method | 53 | 35 |  | worktree_base | GitOps._note_pool_storage_absent, GitOps._pool_storage_bootstrap_ok, GitOps.mark_pool_storage_present, GitOps.pool_in_use, GitOps.pool_storage_present, logger | residual/pool_storage |
| GitOps._merge_verify_lease_active | method | 29 | 19 |  | worktree_base |  | s5/lane_lease |
| GitOps._lane_lock_self_owned_leak | method | 121 | 96 |  | worktree_base | GitOps._merge_verify_lease_active, LaneLockSelfOwnedLeak, _lane_lock_held_in_process, logger | s5/lane_lease |
| GitOps._acquire_lane_flock_off_thread | method | 98 | 79 |  |  | GitOps._release_orphaned_lane_flock, _register_held_lane_lock | s5/lane_lease |
| GitOps._release_orphaned_lane_flock | method | 47 | 31 |  |  | GitOps._release_lane_flock, logger | s5/lane_lease |
| GitOps._release_lane_flock | method | 23 | 19 |  |  | _release_and_forget_held_lane_lock | s5/lane_lease |
| GitOps.merge_verify_lease | method | 218 | 181 |  | worktree_base | GitOps._acquire_lane_flock_off_thread, GitOps._lane_lock_self_owned_leak, GitOps._release_lane_flock, GitOps.is_persistent_merge_lane, GitOps.persistent_merge_worktree_path, MergeVerifyLeaseContended, _MERGE_VERIFY_LEASE_WAIT_SECS, _lane_lock_holder_facts, _settled_lane_lock_holder_pids | residual/git_ops |
| GitOps.task_verify_lease | method | 81 | 65 |  |  | GitOps._acquire_lane_flock_off_thread, GitOps._release_lane_flock, _TASK_VERIFY_LEASE_WAIT_SECS, logger | residual/git_ops |
| GitOps.lane_in_use | method | 20 | 12 |  | _lane_lifecycle, _lane_liveness, warm_lane_pool |  | residual/git_ops |
| GitOps.reset_stale_lane_in_use_records | method | 10 | 6 |  | _lane_lifecycle, _lane_liveness |  | residual/git_ops |
| GitOps._is_registered_worktree | method | 23 | 12 |  | project_root | _run | s5/worktrees |
| GitOps._repair_orphaned_reuse_lane | method | 48 | 26 |  | project_root | GitOps._is_registered_worktree, _run, logger | residual/warm_acquire |
| GitOps._freshen_main | method | 93 | 25 |  | config, project_root | _run, logger | s5/worktrees |
| GitOps._train_predecessor | method | 29 | 6 |  | config | TrainMembership, TrainPredecessor | s5/task_branch |
| GitOps._resolve_predecessor_tip | method | 28 | 14 |  |  | GitOps._train_predecessor, GitOps.resolve_branch_sha, TrainMembership | s5/task_branch |
| GitOps.disable_shared_repo_auto_maintenance | method | 31 | 19 |  | project_root | _run, logger | s5/worktrees |
| GitOps.create_worktree | method | 549 | 248 |  | config, project_root, warm_lane_pool, worktree_base | GitOps._assemble_warm_lane_census, GitOps._branch_has_commits_beyond_main, GitOps._cleanup_leftover_branch, GitOps._freshen_main, GitOps._is_registered_worktree, GitOps._orphan_has_commits, GitOps._provision_reify_debug_port, GitOps._reattach_cold_worktree, GitOps._refuse_foreign_band, GitOps._refuse_prior_incarnation, GitOps._relocate_foreign_meta_root, GitOps._resolve_predecessor_tip, GitOps._worktree_add_with_retry, GitOps.acquire_warm_lane, GitOps.commit, GitOps.disable_shared_repo_auto_maintenance, GitOps.quarantine_worktree, GitOps.rebase_preserving_task_commits, TrainMembership, WarmLaneDiskPressure, WarmLaneLockContention, WarmLaneLockTimeout, WarmLanePoolExhausted, WarmLanePoolHardDown, WarmLaneReseedContaminated, WarmLaneSoftPressure, WarmLaneStealFailed, WarmLaneUnavailable, WorktreeInfo, _SEED_WARM_LANE_LOCK_TIMEOUT_RC, _SEED_WARM_LANE_LOCK_WAIT_SECS, _WARM_LANE_STEAL_MAX_ATTEMPTS, _is_train_successor, _run, logger | residual/git_ops |
| GitOps._reattach_cold_worktree | method | 149 | 88 |  | config, project_root | GitOps._relocate_foreign_meta_root, GitOps._reuse_warm_lane, GitOps.quarantine_worktree, _run, logger | residual/git_ops |
| GitOps.create_interactive_worktree | method | 208 | 104 |  | _interactive_wt_lock, config, project_root, worktree_base | GitOps._cleanup_leftover_branch, GitOps._is_registered_worktree, GitOps._refuse_foreign_band, GitOps._seed_warm_lane, InteractiveWorktreeInfo, InteractiveWorktreeLimitError, _INTERACTIVE_SLUG_RE, _run, logger | s5/worktrees |
| GitOps._scrub_seeded_lane_target | method | 45 | 23 |  | config | logger | s5/warm_lane_scripts |
| GitOps._seed_warm_lane | method | 247 | 165 |  |  | GitOps._scrub_seeded_lane_target, GitOps.mark_pool_storage_present, GitOps.warm_lane_base_target_path, SeedLaneLock, _SEED_ASSUME_LANE_LOCK_HELD_FLAG, _SEED_DISTINCT_LOCK_REFUSAL_RC_FLAG, _SEED_WARM_LANE_LOCK_TIMEOUT_RC, _SEED_WARM_LANE_LOCK_WAIT_SECS, _run, _seed_script_supports_assume_lane_lock_held, _seed_script_supports_distinct_lock_refusal_rc, logger | residual/lane_seeding |
| GitOps.refresh_warm_base | method | 87 | 36 |  | warm_lane_pool | GitOps._merge_verify_artifact_path, GitOps.persistent_merge_worktree_path, GitOps.warm_lane_base_target_path, _run, logger | residual/lane_seeding |
| GitOps._warm_lane_script_candidates | method | 24 | 17 |  | project_root | _df_warm_lane_script_dir | s5/warm_lane_scripts |
| GitOps._resolve_warm_lane_script | method | 35 | 30 |  |  | GitOps._warm_lane_script_candidates | s5/warm_lane_scripts |
| GitOps._resolve_warm_lane_script_logged | method | 54 | 36 |  |  | GitOps._resolve_warm_lane_script, GitOps._warm_lane_script_candidates, logger | s5/warm_lane_scripts |
| GitOps._run_warm_lane_disk_guard | method | 54 | 29 |  | config, project_root, worktree_base | GitOps._resolve_warm_lane_script_logged, _run, logger | s5/warm_lane_scripts |
| GitOps._run_warm_lane_soft_guard | method | 60 | 32 |  | config, project_root, worktree_base | GitOps._resolve_warm_lane_script_logged, _run, logger | s5/warm_lane_scripts |
| GitOps._run_warm_lane_audit | method | 55 | 27 |  | project_root, worktree_base | GitOps._resolve_warm_lane_script_logged, _run, logger | s5/warm_lane_scripts |
| GitOps._warm_lane_audit_cached | method | 44 | 34 | _warm_lane_audit_cache | _warm_lane_audit_cache | GitOps._run_warm_lane_audit, _WARM_LANE_AUDIT_CACHE_TTL_SECS | residual/lane_admission |
| GitOps._run_warm_lane_gc_reclaim | method | 142 | 97 |  | project_root, worktree_base | GitOps._merge_verify_lease_active, GitOps._reconcile_pool_storage_before_sweep, GitOps._resolve_warm_lane_script_logged, _run, logger | residual/lane_admission |
| GitOps.warm_lane_ref_is_degenerate | method | 58 | 29 |  | config, project_root | GitOps._resolve_warm_lane_script_logged, _run, logger | s5/warm_lane_scripts |
| GitOps.thin_warm_lane | method | 137 | 98 |  | project_root, worktree_base | GitOps._note_pool_storage_absent, GitOps._resolve_warm_lane_script_logged, GitOps.pool_in_use, GitOps.pool_storage_present, _run, logger | residual/lane_release |
| GitOps._warm_lane_disk_admission_blocked | method | 35 | 18 |  |  | GitOps._run_warm_lane_disk_guard, GitOps._run_warm_lane_gc_reclaim, logger | residual/lane_admission |
| GitOps._warm_lane_soft_pressure_defer | method | 75 | 59 |  |  | GitOps._run_warm_lane_soft_guard, GitOps._warm_lane_audit_cached, logger | residual/lane_admission |
| GitOps.acquire_spec_lane | method | 77 | 47 |  | spec_warm_lane_pool | GitOps._ready_spec_lane, GitOps._spec_lane_cold_fallback, GitOps.create_ephemeral_verify_worktree, logger | residual/spec_lanes |
| GitOps._ready_spec_lane | method | 37 | 12 |  |  | GitOps._clean_lane_retaining_artifacts, GitOps._create_spec_lane, GitOps._seed_warm_lane, GitOps._spec_lane_registered, _run, logger | residual/spec_lanes |
| GitOps._spec_lane_registered | method | 23 | 7 |  | project_root | GitOps._is_registered_worktree, _run, logger | residual/spec_lanes |
| GitOps._create_spec_lane | method | 59 | 30 |  | _spec_wt_create_lock, project_root, worktree_base | GitOps._note_pool_storage_absent, GitOps._pool_storage_bootstrap_ok, GitOps.mark_pool_storage_present, GitOps.pool_storage_present, _run, logger | residual/spec_lanes |
| GitOps._spec_lane_cold_fallback | method | 10 | 1 |  |  | GitOps.create_ephemeral_verify_worktree, logger | residual/spec_lanes |
| GitOps.release_spec_lane | method | 35 | 22 |  | spec_warm_lane_pool | GitOps.cleanup_merge_worktree, logger | residual/spec_lanes |
| GitOps._try_reclaim_lane_for | method | 117 | 60 |  | warm_lane_dispatched_predicate, warm_lane_pool, warm_lane_reclaim_candidate_provider | GitOps.commit, logger | residual/lane_admission |
| GitOps._assemble_warm_lane_census | method | 48 | 21 |  | _lane_lifecycle, warm_lane_dispatched_predicate, warm_lane_pool | logger | residual/lane_admission |
| GitOps.warn_on_warm_lane_pool_shrink | method | 41 | 10 |  | _lane_lifecycle, warm_lane_pool, worktree_base | logger | residual/lane_admission |
| GitOps._note_structural_exhaustion | method | 49 | 32 | _consecutive_exhausted | _consecutive_exhausted, _on_structural_exhaustion, config | _STRUCTURAL_EXHAUSTION_L2_REFIRE_EVERY, logger | residual/warm_acquire |
| GitOps.acquire_warm_lane | method | 131 | 87 |  |  | GitOps._acquire_warm_lane_impl, GitOps._refuse_prior_incarnation, GitOps._settle_branch_checkout, WarmLaneUnavailable, WorktreeInfo, _STEAL_RETRYABLE, _WARM_LANE_STEAL_MAX_ATTEMPTS, logger | residual/warm_acquire |
| GitOps._settle_branch_checkout | method | 40 | 12 |  | _lane_lifecycle, config, warm_lane_pool | GitOps._worktree_holding_branch, GitOps.detach_lane_checkout, logger | residual/warm_acquire |
| GitOps.prewarm_pool | method | 193 | 96 |  | project_root, warm_lane_pool | GitOps._is_registered_worktree, GitOps._seed_warm_lane, GitOps._teardown_prewarm_lane, GitOps._warm_lane_base_resolvable, GitOps.warm_lane_base_target_path, PoolPrewarmResult, WarmBaseHealth, _run, logger | residual/lane_reset |
| GitOps._teardown_prewarm_lane | method | 34 | 17 |  | project_root | GitOps._prune_registrations, _run, logger | residual/lane_reset |
| GitOps._acquire_warm_lane_impl | method | 879 | 487 | _consecutive_exhausted | config, project_root, transcript_archive, warm_lane_pool, worktree_base | BranchResetError, GitOps._abort_lane_acquisition, GitOps._assemble_warm_lane_census, GitOps._clear_foreign_meta_root, GitOps._is_registered_worktree, GitOps._note_assigned_via_route, GitOps._note_pool_storage_absent, GitOps._note_structural_exhaustion, GitOps._orphan_has_commits, GitOps._pool_storage_bootstrap_ok, GitOps._provision_reify_debug_port, GitOps._repair_orphaned_reuse_lane, GitOps._reseed_verified_clean, GitOps._reset_and_seed_recycled_lane, GitOps._reset_warm_lane, GitOps._reuse_warm_lane, GitOps._seed_warm_lane, GitOps._try_reclaim_lane_for, GitOps._warm_lane_base_resolvable, GitOps._warm_lane_disk_admission_blocked, GitOps._warm_lane_soft_pressure_defer, GitOps.mark_pool_storage_present, GitOps.pool_storage_present, GitOps.warm_lane_base_target_path, WarmBaseHealth, WarmLaneUnavailable, WorktreeInfo, _run, _seed_rc_to_unavailable, logger | residual/warm_acquire |
| GitOps._note_assigned_via_route | method | 39 | 22 |  |  | logger | residual/warm_acquire |
| GitOps._reuse_warm_lane | method | 78 | 46 |  | config | GitOps._provision_reify_debug_port, GitOps._rebase_requeued_onto_main, GitOps.commit, GitOps.rebind_branch_to_head, WorktreeInfo, _run, logger | residual/warm_acquire |
| GitOps._rebase_requeued_onto_main | method | 34 | 13 |  |  | GitOps._worktree_holding_branch, GitOps.rebase_preserving_task_commits, GitOps.unstack_from_unlanded_base, logger | s5/task_branch |
| GitOps.rebind_branch_to_head | method | 39 | 24 |  |  | _run, logger | s5/task_branch |
| GitOps._clean_lane_retaining_artifacts | method | 41 | 20 |  | config | _git_clean_failure_is_benign, _run, logger | s5/worktrees |
| GitOps._clear_foreign_meta_root | method | 23 | 14 |  | worktree_base | logger | residual/lane_reset |
| GitOps._relocate_foreign_meta_root | method | 42 | 23 |  | worktree_base | GitOps._clear_foreign_meta_root, GitOps.quarantine_base, logger | residual/lane_reset |
| GitOps._reset_warm_lane | method | 50 | 25 |  | project_root, transcript_archive | GitOps._clean_lane_retaining_artifacts, GitOps._clear_foreign_meta_root, _run | residual/lane_reset |
| GitOps._reset_and_seed_recycled_lane | method | 103 | 51 |  | config, warm_lane_pool | GitOps._abort_lane_acquisition, GitOps._reset_warm_lane, GitOps._seed_warm_lane, WarmLaneUnavailable, _seed_rc_to_unavailable, logger | residual/lane_reset |
| GitOps.release_warm_lane | method | 76 | 50 |  | config, warm_lane_pool | GitOps._delete_branch_if_on_main, GitOps.thin_warm_lane, _run, logger | residual/lane_release |
| GitOps.detach_lane_checkout | method | 47 | 24 |  |  | GitOps.commit, _run, logger | residual/lane_release |
| GitOps._abort_lane_acquisition | method | 127 | 46 |  | config, project_root, warm_lane_pool | GitOps._delete_branch_if_on_main, GitOps.commit, GitOps.warm_lane_ref_is_degenerate, _run, logger | residual/lane_reset |
| GitOps.release_lane_for_terminal_task | method | 100 | 47 |  | config, warm_lane_pool | GitOps._find_lane_by_plan_task_id, GitOps.cleanup_worktree, logger | residual/lane_release |
| GitOps._find_lane_by_plan_task_id | method | 28 | 10 |  | warm_lane_pool, worktree_base |  | residual/lane_release |
| GitOps._provision_reify_debug_port | method | 43 | 17 |  |  | _run, logger | s5/warm_lane_scripts |
| GitOps._worktree_holding_branch | method | 21 | 7 |  | project_root | _run | s5/task_branch |
| GitOps.lane_branch_checkouts | method | 46 | 19 |  | config, project_root, warm_lane_pool | _run | residual/lane_release |
| GitOps.commits_beyond_main | method | 13 | 1 |  | config, project_root | _run | s5/task_branch |
| GitOps._branch_has_commits_beyond_main | method | 8 | 5 |  |  | GitOps.commits_beyond_main | s5/task_branch |
| GitOps.branch_tip_if_no_commits_beyond | method | 18 | 13 |  |  | GitOps.is_ancestor, GitOps.resolve_branch_sha | s5/task_branch |
| GitOps._orphan_has_commits | method | 26 | 19 |  | project_root | GitOps._branch_has_commits_beyond_main, _run | s5/task_branch |
| GitOps._reseed_verified_clean | method | 56 | 39 |  |  | _run | residual/lane_reset |
| GitOps._delete_branch_if_on_main | method | 38 | 16 |  | config, project_root | GitOps._branch_has_commits_beyond_main, _run, logger | s5/task_branch |
| GitOps._cleanup_leftover_branch | method | 77 | 19 |  | config, project_root | GitOps._branch_has_commits_beyond_main, GitOps._prune_registrations, GitOps._worktree_holding_branch, GitOps.worktree_has_unsaved_work, _run, logger | s5/task_branch |
| GitOps.commit | method | 47 | 28 |  |  | GitOps._detect_unmerged_paths, WorktreeConflictError, _run | s5/worktrees |
| GitOps.get_diff_from_base | method | 11 | 5 |  |  | _run | s5/git_queries |
| GitOps.get_changed_line_ranges | method | 33 | 23 |  | config, project_root | _git_query, parse_diff_line_ranges | s5/git_queries |
| GitOps.get_new_side_changed_line_ranges | method | 25 | 16 |  |  | _run, parse_diff_added_line_ranges | s5/git_queries |
| GitOps.get_current_branch | method | 7 | 1 |  |  | _run | s5/git_queries |
| GitOps.get_main_sha | method | 7 | 1 |  | config, project_root | _run | s5/git_queries |
| GitOps.get_head_tree_hash | method | 29 | 10 |  |  | _run, logger | s5/git_queries |
| GitOps.resolve_branch_sha | method | 15 | 9 |  | project_root | _run | s5/git_queries |
| GitOps.list_local_branches | method | 25 | 5 |  | project_root | _run, logger | s5/git_queries |
| GitOps.resolve_queued_branch_ref | method | 35 | 28 |  | config | GitOps.resolve_branch_sha | s5/task_branch |
| GitOps.find_merge_marker | method | 60 | 53 |  |  | GitOps._lookup_merge_marker, GitOps.resolve_branch_sha | s5/task_branch |
| GitOps._scan_merge_marker | method | 16 | 14 |  |  | GitOps.find_merge_marker_checked, _none_if_query_failed | s5/task_branch |
| GitOps.find_merge_marker_checked | method | 18 | 6 |  | config, project_root | GitQueryFailed, _MERGE_MARKER_LOG_FORMAT, _merge_markers_by_branch, _run | s5/task_branch |
| GitOps._build_merge_marker_index | method | 17 | 9 |  | config, project_root | _MERGE_MARKER_LOG_FORMAT, _merge_markers_by_branch, _run | s5/task_branch |
| GitOps._lookup_merge_marker | method | 35 | 13 | _merge_marker_index, _merge_marker_index_checked_at, _merge_marker_index_sha | _merge_marker_index, _merge_marker_index_checked_at, _merge_marker_index_sha | GitOps._build_merge_marker_index, GitOps._scan_merge_marker, GitOps.get_main_sha, _MERGE_MARKER_INDEX_RECHECK_SECS | s5/task_branch |
| GitOps.find_task_citation_commit | method | 73 | 58 |  |  | CitationPatternInvalid, GitOps.find_task_citation_commit_checked, _none_if_query_failed, logger | s5/git_queries |
| GitOps.find_task_citation_commit_checked | method | 34 | 9 |  | config, project_root | GitQueryFailed, _run, task_citation_pattern | s5/git_queries |
| GitOps.rebase_onto_main | method | 29 | 21 |  | config | _run, logger | s5/task_branch |
| GitOps.rebase_preserving_task_commits | method | 157 | 84 |  | config | BranchResetError, GitOps.rebase_onto_main, _run, logger | s5/task_branch |
| GitOps.merge_tree_conflicts | method | 91 | 60 |  | project_root | ConflictProbe, _run | s5/git_queries |
| GitOps.stack_train_branches | method | 82 | 42 |  | config, worktree_base | GitOps._shed_unlanded_base, GitOps._stack_ledger, GitOps._stack_member, TrainStackResult, logger | s5/task_branch |
| GitOps._stack_member | method | 35 | 8 |  | config | GitOps._record_stack_base, GitOps._shed_unlanded_base, GitOps.rebase_onto_main, GitOps.resolve_branch_sha, logger | s5/task_branch |
| GitOps._shed_unlanded_base | method | 10 | 1 |  |  | GitOps.unstack_from_unlanded_base, logger | s5/task_branch |
| GitOps._record_stack_base | method | 10 | 0 |  |  | GitOps._stack_ledger, GitOps.resolve_branch_sha, logger | s5/task_branch |
| GitOps._stack_ledger | method | 2 | 0 |  | project_root | _run | s5/task_branch |
| GitOps.forget_stack_bases | method | 6 | 2 |  | config | GitOps._stack_ledger | s5/task_branch |
| GitOps.unstack_from_unlanded_base | method | 35 | 16 |  | config | GitOps._stack_ledger, GitOps._unstack_recorded, WorktreeMissing | s5/task_branch |
| GitOps.unlanded_stack_base | method | 17 | 11 |  | config | GitOps._stack_ledger, GitOps.is_ancestor_checked | s5/task_branch |
| GitOps.own_delta_upstream | method | 31 | 10 |  | config | GitOps._foreign_cut, GitOps._stack_ledger, GitOps.get_main_sha, logger | s5/task_branch |
| GitOps.restore_pre_stack_tip | method | 22 | 13 |  | config | GitOps._swap_back_to, WorktreeMissing | s5/task_branch |
| GitOps._swap_back_to | method | 14 | 0 |  | config | GitOps._swap_unlanded_back, GitOps.is_ancestor, GitOps.resolve_branch_sha | s5/task_branch |
| GitOps._swap_unlanded_back | method | 22 | 0 |  | project_root | GitOps._commits_its_train_did_not_stack, GitOps._reset_worktree, GitOps._stack_ledger, GitOps._swap_ref_forward, GitOps._worktree_holding_branch, GitOps.has_uncommitted_work, _run | s5/task_branch |
| GitOps._commits_its_train_did_not_stack | method | 19 | 6 |  | config, project_root | _run | s5/task_branch |
| GitOps._reset_worktree | method | 8 | 2 |  |  | WorktreeMissing, _run | s5/task_branch |
| GitOps._swap_ref_forward | method | 6 | 1 |  | project_root | _run | s5/task_branch |
| GitOps._foreign_cut | method | 8 | 1 |  | project_root | GitOps.is_ancestor, GitOps.resolve_branch_sha, _run | s5/task_branch |
| GitOps._unstack_recorded | method | 15 | 0 |  | config | GitOps._foreign_cut, GitOps._rebase_off_base, GitOps.get_main_sha | s5/task_branch |
| GitOps._rebase_off_base | method | 51 | 0 |  | config, project_root | GitOps._worktree_holding_branch, GitOps.get_rebase_distance, GitOps.has_uncommitted_work, _run, logger | s5/task_branch |
| GitOps.materialize_member_solo | method | 117 | 49 |  | config, project_root, worktree_base | GitOps._prune_registrations, WorktreeInfo, _run, logger | s5/task_branch |
| GitOps.delete_solo_branch | method | 35 | 22 |  | project_root | GitOps._prune_registrations, _run, logger | s5/task_branch |
| GitOps.is_ancestor_checked | method | 13 | 1 |  | project_root | _run | s5/git_queries |
| GitOps.is_ancestor | method | 3 | 1 |  |  | GitOps.is_ancestor_checked | s5/git_queries |
| GitOps.merge_base_with_main | method | 16 | 8 |  | config, project_root | _run | s5/git_queries |
| GitOps.branch_content_in_main | method | 70 | 49 |  | config, project_root | _run | s5/git_queries |
| GitOps.net_diff_is_empty | method | 99 | 67 |  | project_root | _run | s5/git_queries |
| GitOps.landing_merge_for | method | 43 | 34 |  | project_root | _run | s5/git_queries |
| GitOps.describe_commit_effect_in_main | method | 153 | 134 |  | _effect_probe_memo | CommitEffectProbe, GitOps._probe_commit_effect, GitOps.get_main_sha, _EFFECT_PROBE_MEMO_MAX_ENTRIES, _EFFECT_PROBE_TRANSIENT_FAILURES | s5-split/commit_effect |
| GitOps._probe_commit_effect | method | 108 | 36 |  | project_root | CommitEffectProbe, GitOps._compare_touched_paths_to_main, _EMPTY_TREE_SHA, _run | s5-split/commit_effect |
| GitOps._anchor_diff_lines | method | 102 | 66 |  | project_root | _run | s5-split/commit_effect |
| GitOps._batch_added_line_counts | method | 93 | 42 |  | project_root | _run | s5-split/commit_effect |
| GitOps._path_exists_at | method | 6 | 1 |  | project_root | _run | s5-split/commit_effect |
| GitOps._blob_oid_at | method | 6 | 1 |  | project_root | _run | s5-split/commit_effect |
| GitOps._vacuous_path_survives | method | 52 | 33 |  |  | GitOps._blob_oid_at, GitOps._main_line_counts, GitOps._path_exists_at | s5-split/commit_effect |
| GitOps._main_line_counts | method | 59 | 44 |  | project_root | _run | s5-split/commit_effect |
| GitOps._compare_touched_paths_to_main | method | 197 | 79 |  | project_root | CommitEffectProbe, GitOps._anchor_diff_lines, GitOps._batch_added_line_counts, GitOps._main_line_counts, GitOps._vacuous_path_survives, _EFFECT_SURVIVAL_AGGREGATE_THRESHOLD, _EFFECT_SURVIVAL_PER_FILE_MIN_ADDED_LINES, _EFFECT_SURVIVAL_PER_FILE_THRESHOLD, _run | s5-split/commit_effect |
| GitOps.commit_effect_present_in_main | method | 172 | 170 |  |  | GitOps.describe_commit_effect_in_main | s5-split/commit_effect |
| GitOps.worktree_head_beyond_main | method | 35 | 24 |  |  | GitOps.get_main_sha, GitOps.is_ancestor, _run | s5/git_queries |
| GitOps.has_uncommitted_work | method | 13 | 7 |  |  | _run | s5/git_queries |
| GitOps.get_commit_subjects | method | 26 | 10 |  |  | _run | s5/git_queries |
| GitOps.find_equivalent_commit | method | 41 | 25 |  |  | GitOps._find_replay, logger | s5/git_queries |
| GitOps._find_replay | method | 93 | 51 |  |  | GitOps._unambiguous_patch_ids, ReplayMatch, _git_query | s5/git_queries |
| GitOps._unambiguous_patch_ids | method | 35 | 7 |  |  | _git_query | s5/git_queries |
| GitOps.commits_named | method | 26 | 5 |  | project_root | GitQueryFailed, _FULL_OID, _git_query | s5/git_queries |
| GitOps.resolve_rebased_citation | method | 45 | 21 |  | config, project_root | GitOps._commit_paths, GitOps._find_replay, GitOps.commits_named, GitOps.is_ancestor_checked, GitOps.merge_base_with_main, GitQueryFailed, RebasedCitation, _git_query | s5/git_queries |
| GitOps.get_changed_files | method | 20 | 8 |  | project_root | _run, logger | s5/git_queries |
| GitOps.get_commit_changed_files | method | 22 | 17 |  |  | GitOps._commit_paths, GitQueryFailed | s5/git_queries |
| GitOps._commit_paths | method | 7 | 1 |  | project_root | _git_query | s5/git_queries |
| GitOps.get_rebase_distance | method | 29 | 10 |  | project_root | _run, logger | s5/git_queries |
| GitOps.get_merge_diff_files | method | 39 | 21 |  | project_root | _run, logger | s5/git_queries |
| GitOps.get_merge_commit_diff_files | method | 30 | 26 |  |  | GitOps.get_merge_diff_files | s5/git_queries |
| GitOps.get_files_touched_in_branch | method | 42 | 22 |  | project_root | _run, logger | s5/git_queries |
| GitOps.get_branch_changed_files | method | 53 | 27 |  | config, project_root | _git_stderr_is_unresolved_ref, _run, logger | s5/git_queries |
| GitOps._resolve_merge_branch_names | method | 18 | 15 |  | config | GitOps.resolve_queued_branch_ref | s5-split/merging |
| GitOps._run_no_ff_merge | method | 35 | 23 |  | config | _run | s5-split/merging |
| GitOps.merge_to_main | method | 103 | 52 |  |  | GitOps._conflicted_paths_or_none, GitOps._create_merge_worktree, GitOps._resolve_merge_branch_names, GitOps._run_no_ff_merge, GitOps.cleanup_merge_worktree, GitOps.get_conflict_details, GitOps.worktree_head_beyond_main, MergeResult, _run | s5-split/merging |
| GitOps.merge_branch_into_worktree | method | 235 | 163 |  |  | GitOps._conflicted_paths_or_none, GitOps._resolve_merge_branch_names, GitOps._run_no_ff_merge, GitOps.abort_merge, GitOps.get_conflict_details, MergeResult, _run | s5-split/merging |
| GitOps._create_merge_worktree | method | 75 | 36 |  | config, project_root, worktree_base | GitOps._worktree_add_with_retry, GitOps.cleanup_merge_worktree, _run, logger | s5-split/merge_worktrees |
| GitOps.remove_merge_worktree_guarded | method | 236 | 171 |  | project_root, worktree_base | GitOps.is_persistent_merge_lane, GitOps.persistent_offline_deep_worktree_path, RemovalOutcome, _lane_lock_holder_facts, _merge_worktree_remove_argv, _register_held_lane_lock, _release_and_forget_held_lane_lock, _run, logger | s5-split/merge_worktrees |
| GitOps.cleanup_merge_worktree | method | 114 | 93 |  |  | GitOps._prune_registrations, GitOps._refuse_foreign_band, GitOps.remove_merge_worktree_guarded, logger | s5-split/merge_worktrees |
| GitOps.create_ephemeral_verify_worktree | method | 24 | 21 |  |  | GitOps._create_merge_worktree | s5-split/merge_worktrees |
| GitOps.persistent_merge_worktree_path | method | 10 | 7 |  | worktree_base | PERSISTENT_MERGE_WORKTREE_NAME | s5-unplaced/lane_paths |
| GitOps.is_persistent_merge_lane | method | 32 | 30 |  |  | GitOps.persistent_merge_worktree_path | s5-unplaced/lane_paths |
| GitOps.persistent_offline_deep_worktree_path | method | 13 | 10 |  | worktree_base | PERSISTENT_OFFLINE_DEEP_WORKTREE_NAME | s5-unplaced/lane_paths |
| GitOps._merge_verify_artifact_path | method | 18 | 10 |  | config | GitOps.persistent_merge_worktree_path | s5-unplaced/lane_paths |
| GitOps.warm_lane_base_target_path | method | 12 | 7 |  | config | GitOps._merge_verify_artifact_path | s5-unplaced/lane_paths |
| GitOps._warm_lane_base_resolvable | method | 63 | 42 |  |  | GitOps.warm_lane_base_target_path, WarmBaseHealth, logger | s5/warm_lane_scripts |
| GitOps._bump_host_verify_attempt_count | method | 35 | 25 |  | _VERIFY_ATTEMPT_COUNTER_FILENAME, worktree_base |  | s5-split/merge_worktrees |
| GitOps.acquire_host_verify_worktree | method | 51 | 37 |  | config | GitOps._bump_host_verify_attempt_count, GitOps._create_merge_worktree, GitOps.reset_persistent_merge_worktree | s5-split/merge_worktrees |
| GitOps.reap_merge_verify_survivors | method | 97 | 46 |  | config | GitOps.persistent_merge_worktree_path, logger | s5-split/merge_worktrees |
| GitOps.reset_persistent_merge_worktree | method | 201 | 124 |  | project_root, worktree_base | GitOps._acquire_lane_flock_off_thread, GitOps._clean_lane_retaining_artifacts, GitOps._is_registered_worktree, GitOps._lane_lock_self_owned_leak, GitOps._merge_verify_lease_active, GitOps._release_lane_flock, GitOps.persistent_merge_worktree_path, MergeVerifyLeaseContended, MergeVerifyLeaseHeld, _RESET_WARM_LANE_LOCK_WAIT_SECS, _lane_lock_holder_facts, _run, _settled_lane_lock_holder_pids, logger | s5-split/merge_worktrees |
| GitOps.reset_persistent_offline_deep_worktree | method | 77 | 29 |  | project_root | GitOps._clean_lane_retaining_artifacts, GitOps._is_registered_worktree, GitOps.persistent_offline_deep_worktree_path, _run, logger | s5-split/merge_worktrees |
| GitOps._iter_merge_worktrees | method | 43 | 20 |  | project_root, worktree_base | PERSISTENT_MERGE_WORKTREE_NAME, _run | s5-split/merge_worktrees |
| GitOps.prune_stale_merge_worktrees | method | 69 | 27 |  | project_root | GitOps._iter_merge_worktrees, GitOps._prune_registrations, GitOps._refuse_foreign_band, _merge_worktree_remove_argv, _run, logger | s5-split/merge_worktrees |
| GitOps.find_inflight_merge_worktree | method | 54 | 24 |  | config | GitOps._iter_merge_worktrees, GitOps.resolve_queued_branch_ref, WorktreeMissing, _run, logger | s5-split/merge_worktrees |
| GitOps._iter_interactive_worktrees | method | 30 | 9 |  | config, project_root, worktree_base | _run | s5-split/merge_worktrees |
| GitOps._interactive_worktree_landed | method | 25 | 23 |  |  | GitOps._lookup_merge_marker | s5-split/merge_worktrees |
| GitOps._worktree_dirty | method | 19 | 14 |  |  | _run | s5-split/merge_worktrees |
| GitOps._worktree_head_readable | method | 14 | 11 |  |  | _run | s5-split/merge_worktrees |
| GitOps.reap_interactive_worktrees | method | 227 | 123 |  | config, project_root, worktree_base | GitOps._interactive_worktree_landed, GitOps._iter_interactive_worktrees, GitOps._prune_registrations, GitOps._refuse_foreign_band, GitOps._run_warm_lane_disk_guard, GitOps._worktree_dirty, GitOps._worktree_head_readable, GitOps.worktree_head_beyond_main, ReapedInteractiveWorktree, _run, logger | s5-split/merge_worktrees |
| GitOps.advance_main | method | 690 | 317 | _last_overlap_files, _last_park_lock_info, _last_recovery_branch, _last_stash_dirty_files | config, project_root | AdvanceOutcome, GitOps._await_index_lock_clear, GitOps._detect_unmerged_paths, GitOps._index_lock_path, GitOps._index_lock_state, GitOps._park_wip_on_private_ref, GitOps._safe_restore_park_with_recovery, GitOps.get_main_sha, GitOps.resolve_queued_branch_ref, MERGE_PARK_REF, MergeParkContentionError, MergeParkError, MergeParkLockContentionError, _assert_no_conflict_markers, _run, logger | s5/main_landing |
| GitOps.recover_red_main | method | 216 | 91 |  | config, project_root | RecoverResult, _run, logger | s5/main_landing |
| GitOps.push_main | method | 46 | 16 |  | config, project_root | PushResult, _run, logger | s5/main_landing |
| GitOps._index_lock_path | method | 35 | 15 | _index_lock_path_cache | project_root | _run | s5/main_landing |
| GitOps._index_lock_state | method | 19 | 10 |  |  | GitOps._index_lock_path | s5/main_landing |
| GitOps._await_index_lock_clear | method | 116 | 52 |  |  | GitOps._index_lock_path, GitOps._index_lock_state, _INDEX_LOCK_POLL_INTERVAL_S, _INDEX_LOCK_STALE_FLOOR_S, _INDEX_LOCK_WARN_INTERVAL_S, logger | s5/main_landing |
| GitOps._park_wip_on_private_ref | method | 117 | 66 |  | project_root | GitOps._index_lock_path, GitOps._index_lock_state, MERGE_PARK_REF, MergeParkContentionError, MergeParkError, MergeParkLockContentionError, _run | s5/main_landing |
| GitOps._create_recovery_branch_from_park_ref | method | 44 | 19 |  | project_root | MERGE_PARK_REF, _run, logger | s5/main_landing |
| GitOps._safe_restore_park_with_recovery | method | 48 | 20 |  | project_root | GitOps._create_recovery_branch_from_park_ref, GitOps._detect_unmerged_paths, MERGE_PARK_REF, _run, logger | s5/main_landing |
| GitOps.has_dirty_working_tree | method | 17 | 6 |  | project_root | _run | s5/git_queries |
| GitOps._detect_unmerged_paths | method | 22 | 8 |  |  | _run | s5/git_queries |
| GitOps.unmerged_paths | method | 17 | 6 |  |  | _run | s5/git_queries |
| GitOps._conflicted_paths_or_none | method | 17 | 7 |  |  | GitOps.unmerged_paths, logger | s5-split/merging |
| GitOps.get_conflict_details | method | 13 | 1 |  |  | _run | s5-split/merging |
| GitOps.abort_merge | method | 4 | 1 |  |  | _run, logger | s5-split/merging |
| GitOps.rename_worktree | method | 53 | 22 |  | config, project_root | GitOps._carry_branch_records, GitOps._move_registered_worktree, _run, logger | s5/worktrees |
| GitOps.cleanup_worktree | method | 125 | 65 |  | config, project_root, spec_warm_lane_pool, transcript_archive, warm_lane_pool | GitOps._delete_branch_if_on_main, GitOps.release_spec_lane, GitOps.release_warm_lane, _run, logger | residual/lane_release |
| GitOps.reclaim_worktree_build_artifacts | method | 49 | 22 |  | config | logger | s5/worktrees |
| GitOps.quarantine_base | method | 9 | 6 |  | worktree_base |  | s5/worktrees |
| GitOps.worktree_has_unsaved_work | method | 35 | 10 |  | config, project_root | WorktreeMissing, _run, logger | s5/git_queries |
| GitOps.quarantine_worktree | method | 52 | 28 |  | config | GitOps._parked_outcome, GitOps._relocate_into_quarantine, GitOps.quarantine_base, _stranded_branch, _utc_stamp, logger | s5/worktrees |
| GitOps._relocate_into_quarantine | method | 29 | 7 |  | config | GitOps._move_registered_worktree, GitOps._move_with_branch_parked, GitOps._rename_unregistered, GitOps._worktree_checkouts, GitOps.commit, logger | s5/worktrees |
| GitOps._worktree_checkouts | method | 16 | 1 |  | project_root | _run | s5/worktrees |
| GitOps._move_registered_worktree | method | 10 | 0 |  | project_root | _run | s5/worktrees |
| GitOps._move_with_branch_parked | method | 27 | 7 |  | config, project_root | GitOps._carry_branch_records, GitOps._rename_branch_back, GitOps.resolve_branch_sha, _ParkedBranchStranded, _run | s5/worktrees |
| GitOps._rename_branch_back | method | 10 | 1 |  | project_root | _run, logger | s5/worktrees |
| GitOps._rename_unregistered | method | 23 | 6 |  |  | GitOps._sever_parked_gitfile, logger | s5/worktrees |
| GitOps._sever_parked_gitfile | method | 22 | 9 |  | project_root | _run | s5/worktrees |
| GitOps._parked_outcome | method | 16 | 1 |  |  | GitOps.commits_beyond_main, GitOps.resolve_branch_sha, logger | s5/worktrees |
| GitOps.park_prior_incarnation | method | 37 | 6 |  | config, warm_lane_pool | GitOps._park_meta_roots, GitOps._quarantine_cold_incarnation, GitOps._rename_branch_aside, GitOps.release_warm_lane, GitOps.resolve_branch_sha, _utc_stamp, logger | residual/lane_release |
| GitOps._quarantine_cold_incarnation | method | 11 | 1 |  | worktree_base | GitOps._is_registered_worktree, GitOps.quarantine_worktree | residual/lane_release |
| GitOps._rename_branch_aside | method | 14 | 1 |  | config, project_root | GitOps._carry_branch_records, GitOps._free_lane_checkout, _run | residual/lane_release |
| GitOps._carry_branch_records | method | 5 | 2 |  | project_root | GitOps._stack_ledger, _run | s5/task_branch |
| GitOps._free_lane_checkout | method | 18 | 4 |  | warm_lane_pool | GitOps.detach_lane_checkout, GitOps.lane_branch_checkouts | residual/lane_release |
| GitOps._meta_roots_naming | method | 19 | 3 |  | worktree_base |  | residual/lane_release |
| GitOps._park_meta_roots | method | 15 | 4 |  |  | GitOps._meta_roots_naming, GitOps.quarantine_base | residual/lane_release |
| GitOps.prior_incarnation_evidence | method | 19 | 1 |  | config, warm_lane_pool, worktree_base | GitOps._is_registered_worktree, GitOps._meta_roots_naming, GitOps.resolve_branch_sha | residual/lane_release |
| GitOps._refuse_prior_incarnation | method | 10 | 2 |  |  | GitOps.prior_incarnation_evidence | residual/lane_release |
| GitOps._prune_registrations | method | 69 | 55 |  | project_root, worktree_base | GitOps._reconcile_pool_storage_before_sweep, _run, logger | residual/pool_storage |
| GitOps.prune_worktrees | method | 9 | 7 |  |  | GitOps._prune_registrations | s5/worktrees |

## Clusters

| cluster | units | lines | prose |
| --- | ---: | ---: | ---: |
| per-module/logger | 1 | 1 | 0 |
| residual/git_ops | 7 | 1177 | 698 |
| residual/lane_admission | 8 | 511 | 307 |
| residual/lane_outcomes | 15 | 405 | 358 |
| residual/lane_release | 15 | 702 | 335 |
| residual/lane_reset | 8 | 628 | 311 |
| residual/lane_seeding | 3 | 358 | 224 |
| residual/pool_storage | 7 | 247 | 193 |
| residual/spec_lanes | 6 | 241 | 119 |
| residual/warm_acquire | 9 | 1298 | 744 |
| s5-split/commit_effect | 17 | 1123 | 755 |
| s5-split/merge_worktrees | 19 | 1428 | 867 |
| s5-split/merging | 8 | 446 | 273 |
| s5-unplaced/lane_paths | 7 | 96 | 73 |
| s5/git_diff | 3 | 140 | 72 |
| s5/git_queries | 46 | 1427 | 797 |
| s5/git_run | 5 | 77 | 47 |
| s5/lane_lease | 20 | 884 | 674 |
| s5/main_landing | 21 | 1490 | 722 |
| s5/task_branch | 50 | 1402 | 648 |
| s5/warm_lane_scripts | 21 | 644 | 392 |
| s5/worktrees | 46 | 1541 | 843 |

## Cross-cluster edges

| from | to | refs | sample |
| --- | --- | ---: | --- |
| residual/git_ops | per-module/logger | 3 | GitOps._reattach_cold_worktree -> logger |
| residual/git_ops | residual/lane_admission | 1 | GitOps.create_worktree -> GitOps._assemble_warm_lane_census |
| residual/git_ops | residual/lane_outcomes | 11 | GitOps.create_worktree -> WarmLaneDiskPressure |
| residual/git_ops | residual/lane_release | 1 | GitOps.create_worktree -> GitOps._refuse_prior_incarnation |
| residual/git_ops | residual/lane_reset | 2 | GitOps._reattach_cold_worktree -> GitOps._relocate_foreign_meta_root |
| residual/git_ops | residual/lane_seeding | 1 | GitOps.create_worktree -> _SEED_WARM_LANE_LOCK_WAIT_SECS |
| residual/git_ops | residual/warm_acquire | 3 | GitOps._reattach_cold_worktree -> GitOps._reuse_warm_lane |
| residual/git_ops | s5-split/commit_effect | 1 | GitOps.__init__ -> CommitEffectProbe |
| residual/git_ops | s5-unplaced/lane_paths | 2 | GitOps.merge_verify_lease -> GitOps.is_persistent_merge_lane |
| residual/git_ops | s5/git_run | 2 | GitOps._reattach_cold_worktree -> _run |
| residual/git_ops | s5/lane_lease | 10 | GitOps.merge_verify_lease -> GitOps._acquire_lane_flock_off_thread |
| residual/git_ops | s5/task_branch | 7 | GitOps.create_worktree -> GitOps._branch_has_commits_beyond_main |
| residual/git_ops | s5/warm_lane_scripts | 1 | GitOps.create_worktree -> GitOps._provision_reify_debug_port |
| residual/git_ops | s5/worktrees | 9 | GitOps.__init__ -> GitOps.quarantine_worktree |
| residual/lane_admission | per-module/logger | 6 | GitOps._assemble_warm_lane_census -> logger |
| residual/lane_admission | residual/pool_storage | 1 | GitOps._run_warm_lane_gc_reclaim -> GitOps._reconcile_pool_storage_before_sweep |
| residual/lane_admission | s5/git_run | 1 | GitOps._run_warm_lane_gc_reclaim -> _run |
| residual/lane_admission | s5/lane_lease | 1 | GitOps._run_warm_lane_gc_reclaim -> GitOps._merge_verify_lease_active |
| residual/lane_admission | s5/warm_lane_scripts | 4 | GitOps._run_warm_lane_gc_reclaim -> GitOps._resolve_warm_lane_script_logged |
| residual/lane_admission | s5/worktrees | 1 | GitOps._try_reclaim_lane_for -> GitOps.commit |
| residual/lane_release | per-module/logger | 6 | GitOps.cleanup_worktree -> logger |
| residual/lane_release | residual/pool_storage | 3 | GitOps.thin_warm_lane -> GitOps._note_pool_storage_absent |
| residual/lane_release | residual/spec_lanes | 1 | GitOps.cleanup_worktree -> GitOps.release_spec_lane |
| residual/lane_release | s5/git_queries | 2 | GitOps.park_prior_incarnation -> GitOps.resolve_branch_sha |
| residual/lane_release | s5/git_run | 6 | GitOps._rename_branch_aside -> _run |
| residual/lane_release | s5/task_branch | 3 | GitOps._rename_branch_aside -> GitOps._carry_branch_records |
| residual/lane_release | s5/warm_lane_scripts | 1 | GitOps.thin_warm_lane -> GitOps._resolve_warm_lane_script_logged |
| residual/lane_release | s5/worktrees | 6 | GitOps._park_meta_roots -> GitOps.quarantine_base |
| residual/lane_reset | per-module/logger | 6 | GitOps._abort_lane_acquisition -> logger |
| residual/lane_reset | residual/lane_outcomes | 3 | GitOps._reset_and_seed_recycled_lane -> WarmLaneUnavailable |
| residual/lane_reset | residual/lane_seeding | 2 | GitOps._reset_and_seed_recycled_lane -> GitOps._seed_warm_lane |
| residual/lane_reset | residual/pool_storage | 1 | GitOps._teardown_prewarm_lane -> GitOps._prune_registrations |
| residual/lane_reset | s5-unplaced/lane_paths | 1 | GitOps.prewarm_pool -> GitOps.warm_lane_base_target_path |
| residual/lane_reset | s5/git_run | 5 | GitOps._abort_lane_acquisition -> _run |
| residual/lane_reset | s5/task_branch | 1 | GitOps._abort_lane_acquisition -> GitOps._delete_branch_if_on_main |
| residual/lane_reset | s5/warm_lane_scripts | 3 | GitOps._abort_lane_acquisition -> GitOps.warm_lane_ref_is_degenerate |
| residual/lane_reset | s5/worktrees | 4 | GitOps._abort_lane_acquisition -> GitOps.commit |
| residual/lane_seeding | per-module/logger | 2 | GitOps._seed_warm_lane -> logger |
| residual/lane_seeding | residual/lane_outcomes | 1 | GitOps._seed_warm_lane -> _SEED_WARM_LANE_LOCK_TIMEOUT_RC |
| residual/lane_seeding | residual/pool_storage | 1 | GitOps._seed_warm_lane -> GitOps.mark_pool_storage_present |
| residual/lane_seeding | s5-unplaced/lane_paths | 4 | GitOps._seed_warm_lane -> GitOps.warm_lane_base_target_path |
| residual/lane_seeding | s5/git_run | 2 | GitOps._seed_warm_lane -> _run |
| residual/lane_seeding | s5/warm_lane_scripts | 6 | GitOps._seed_warm_lane -> GitOps._scrub_seeded_lane_target |
| residual/pool_storage | per-module/logger | 3 | GitOps._note_pool_storage_absent -> logger |
| residual/pool_storage | s5-unplaced/lane_paths | 1 | GitOps._pool_storage_bootstrap_ok -> GitOps.warm_lane_base_target_path |
| residual/pool_storage | s5/git_run | 1 | GitOps._prune_registrations -> _run |
| residual/pool_storage | s5/warm_lane_scripts | 2 | GitOps._pool_storage_bootstrap_ok -> GitOps._warm_lane_base_resolvable |
| residual/spec_lanes | per-module/logger | 6 | GitOps._create_spec_lane -> logger |
| residual/spec_lanes | residual/lane_seeding | 1 | GitOps._ready_spec_lane -> GitOps._seed_warm_lane |
| residual/spec_lanes | residual/pool_storage | 4 | GitOps._create_spec_lane -> GitOps._note_pool_storage_absent |
| residual/spec_lanes | s5-split/merge_worktrees | 3 | GitOps._spec_lane_cold_fallback -> GitOps.create_ephemeral_verify_worktree |
| residual/spec_lanes | s5/git_run | 3 | GitOps._create_spec_lane -> _run |
| residual/spec_lanes | s5/worktrees | 2 | GitOps._ready_spec_lane -> GitOps._clean_lane_retaining_artifacts |
| residual/warm_acquire | per-module/logger | 7 | GitOps._acquire_warm_lane_impl -> logger |
| residual/warm_acquire | residual/lane_admission | 4 | GitOps._acquire_warm_lane_impl -> GitOps._assemble_warm_lane_census |
| residual/warm_acquire | residual/lane_outcomes | 7 | GitOps._acquire_warm_lane_impl -> WarmLaneUnavailable |
| residual/warm_acquire | residual/lane_release | 2 | GitOps._settle_branch_checkout -> GitOps.detach_lane_checkout |
| residual/warm_acquire | residual/lane_reset | 5 | GitOps._acquire_warm_lane_impl -> GitOps._abort_lane_acquisition |
| residual/warm_acquire | residual/lane_seeding | 1 | GitOps._acquire_warm_lane_impl -> GitOps._seed_warm_lane |
| residual/warm_acquire | residual/pool_storage | 4 | GitOps._acquire_warm_lane_impl -> GitOps._note_pool_storage_absent |
| residual/warm_acquire | s5-unplaced/lane_paths | 1 | GitOps._acquire_warm_lane_impl -> GitOps.warm_lane_base_target_path |
| residual/warm_acquire | s5/git_run | 3 | GitOps._acquire_warm_lane_impl -> _run |
| residual/warm_acquire | s5/task_branch | 5 | GitOps._acquire_warm_lane_impl -> BranchResetError |
| residual/warm_acquire | s5/warm_lane_scripts | 4 | GitOps._acquire_warm_lane_impl -> GitOps._provision_reify_debug_port |
| residual/warm_acquire | s5/worktrees | 3 | GitOps._acquire_warm_lane_impl -> GitOps._is_registered_worktree |
| s5-split/commit_effect | s5/git_queries | 1 | GitOps.describe_commit_effect_in_main -> GitOps.get_main_sha |
| s5-split/commit_effect | s5/git_run | 7 | GitOps._anchor_diff_lines -> _run |
| s5-split/merge_worktrees | per-module/logger | 9 | GitOps._create_merge_worktree -> logger |
| s5-split/merge_worktrees | residual/pool_storage | 3 | GitOps.cleanup_merge_worktree -> GitOps._prune_registrations |
| s5-split/merge_worktrees | s5-unplaced/lane_paths | 6 | GitOps._iter_merge_worktrees -> PERSISTENT_MERGE_WORKTREE_NAME |
| s5-split/merge_worktrees | s5/git_queries | 1 | GitOps.reap_interactive_worktrees -> GitOps.worktree_head_beyond_main |
| s5-split/merge_worktrees | s5/git_run | 12 | GitOps._create_merge_worktree -> _run |
| s5-split/merge_worktrees | s5/lane_lease | 11 | GitOps.remove_merge_worktree_guarded -> _lane_lock_holder_facts |
| s5-split/merge_worktrees | s5/task_branch | 2 | GitOps._interactive_worktree_landed -> GitOps._lookup_merge_marker |
| s5-split/merge_worktrees | s5/warm_lane_scripts | 1 | GitOps.reap_interactive_worktrees -> GitOps._run_warm_lane_disk_guard |
| s5-split/merge_worktrees | s5/worktrees | 10 | GitOps._create_merge_worktree -> GitOps._worktree_add_with_retry |
| s5-split/merging | per-module/logger | 2 | GitOps._conflicted_paths_or_none -> logger |
| s5-split/merging | s5-split/merge_worktrees | 2 | GitOps.merge_to_main -> GitOps._create_merge_worktree |
| s5-split/merging | s5/git_queries | 2 | GitOps._conflicted_paths_or_none -> GitOps.unmerged_paths |
| s5-split/merging | s5/git_run | 5 | GitOps._run_no_ff_merge -> _run |
| s5-split/merging | s5/task_branch | 1 | GitOps._resolve_merge_branch_names -> GitOps.resolve_queued_branch_ref |
| s5/git_queries | per-module/logger | 10 | GitOps.find_equivalent_commit -> logger |
| s5/git_queries | s5/git_diff | 2 | GitOps.get_changed_line_ranges -> parse_diff_line_ranges |
| s5/git_queries | s5/git_run | 38 | GitOps._commit_paths -> _git_query |
| s5/lane_lease | per-module/logger | 2 | GitOps._lane_lock_self_owned_leak -> logger |
| s5/main_landing | per-module/logger | 7 | GitOps._await_index_lock_clear -> logger |
| s5/main_landing | s5/git_queries | 3 | GitOps._safe_restore_park_with_recovery -> GitOps._detect_unmerged_paths |
| s5/main_landing | s5/git_run | 8 | GitOps._create_recovery_branch_from_park_ref -> _run |
| s5/main_landing | s5/task_branch | 1 | GitOps.advance_main -> GitOps.resolve_queued_branch_ref |
| s5/task_branch | per-module/logger | 14 | GitOps._cleanup_leftover_branch -> logger |
| s5/task_branch | residual/lane_outcomes | 1 | GitOps.materialize_member_solo -> WorktreeInfo |
| s5/task_branch | residual/pool_storage | 3 | GitOps._cleanup_leftover_branch -> GitOps._prune_registrations |
| s5/task_branch | s5/git_queries | 19 | GitOps._cleanup_leftover_branch -> GitOps.worktree_has_unsaved_work |
| s5/task_branch | s5/git_run | 25 | GitOps._build_merge_marker_index -> _run |
| s5/warm_lane_scripts | per-module/logger | 10 | GitOps._provision_reify_debug_port -> logger |
| s5/warm_lane_scripts | s5-unplaced/lane_paths | 1 | GitOps._warm_lane_base_resolvable -> GitOps.warm_lane_base_target_path |
| s5/warm_lane_scripts | s5/git_run | 5 | GitOps._provision_reify_debug_port -> _run |
| s5/worktrees | per-module/logger | 16 | GitOps._clean_lane_retaining_artifacts -> logger |
| s5/worktrees | residual/lane_seeding | 2 | GitOps._warm_seed_ephemeral_worktree -> GitOps._seed_warm_lane |
| s5/worktrees | residual/pool_storage | 1 | GitOps.prune_worktrees -> GitOps._prune_registrations |
| s5/worktrees | s5-unplaced/lane_paths | 2 | PROTECTED_PREFIXES -> PERSISTENT_MERGE_WORKTREE_NAME |
| s5/worktrees | s5/git_queries | 3 | GitOps._move_with_branch_parked -> GitOps.resolve_branch_sha |
| s5/worktrees | s5/git_run | 14 | GitOps._clean_lane_retaining_artifacts -> _run |
| s5/worktrees | s5/task_branch | 4 | GitOps._move_with_branch_parked -> GitOps._carry_branch_records |
| s5/worktrees | s5/warm_lane_scripts | 3 | GitOps._warm_seed_ephemeral_worktree -> GitOps._warm_lane_base_resolvable |

## Cyclic cluster groups

none

## Shared writes

- `_consecutive_exhausted`: residual/git_ops, residual/warm_acquire
- `_merge_marker_index`: residual/git_ops, s5/task_branch
- `_merge_marker_index_checked_at`: residual/git_ops, s5/task_branch
- `_merge_marker_index_sha`: residual/git_ops, s5/task_branch
- `_warm_lane_audit_cache`: residual/git_ops, residual/lane_admission

## Module-unit users

| module unit | clusters |
| --- | --- |
| logger | residual/git_ops, residual/lane_admission, residual/lane_release, residual/lane_reset, residual/lane_seeding, residual/pool_storage, residual/spec_lanes, residual/warm_acquire, s5-split/merge_worktrees, s5-split/merging, s5/git_queries, s5/lane_lease, s5/main_landing, s5/task_branch, s5/warm_lane_scripts, s5/worktrees |
| AdvanceResult | s5/main_landing |
| AdvanceOutcome | s5/main_landing |
| PushResult | s5/main_landing |
| RecoverResult | s5/main_landing |
| ReplayMatch | s5/git_queries |
| RebasedCitation | s5/git_queries |
| RemovalOutcome | s5-split/merge_worktrees |
| WIP_SAFETY_COMMIT_PREFIXES | s5/task_branch |
| is_wip_safety_commit | none |
| MERGE_PARK_REF | s5/main_landing |
| _INDEX_LOCK_POLL_INTERVAL_S | s5/main_landing |
| _INDEX_LOCK_WARN_INTERVAL_S | s5/main_landing |
| _INDEX_LOCK_STALE_FLOOR_S | s5/main_landing |
| _WORKTREE_ADD_MAX_ATTEMPTS | s5/worktrees |
| _worktree_add_failure_is_retryable | s5/worktrees |
| MergeParkError | s5/main_landing |
| MergeParkLockContentionError | s5/main_landing |
| MergeParkContentionError | s5/main_landing |
| TrainMembership | residual/git_ops, s5/task_branch |
| _is_train_successor | residual/git_ops |
| TrainPredecessor | s5/task_branch |
| TrainStackResult | s5/task_branch |
| DEFAULT_COMMIT_CITATION_PATTERN | s5/git_queries |
| render_citation_pattern | s5/git_queries |
| task_citation_pattern | s5/git_queries |
| PERSISTENT_MERGE_WORKTREE_NAME | s5-split/merge_worktrees, s5-unplaced/lane_paths, s5/worktrees |
| _MERGE_VERIFY_LEASE_WAIT_SECS | residual/git_ops |
| _TASK_VERIFY_LEASE_WAIT_SECS | residual/git_ops |
| _SEED_WARM_LANE_LOCK_WAIT_SECS | residual/git_ops, residual/lane_seeding |
| _RESET_WARM_LANE_LOCK_WAIT_SECS | s5-split/merge_worktrees |
| _SEED_WARM_LANE_LOCK_TIMEOUT_RC | residual/git_ops, residual/lane_outcomes, residual/lane_seeding |
| _SEED_ASSUME_LANE_LOCK_HELD_FLAG | residual/lane_seeding, s5/warm_lane_scripts |
| _WARM_LANE_STEAL_MAX_ATTEMPTS | residual/git_ops, residual/warm_acquire |
| _STEAL_RETRYABLE | residual/warm_acquire |
| _SEED_DISTINCT_LOCK_REFUSAL_RC_FLAG | residual/lane_seeding, s5/warm_lane_scripts |
| _DF_WARM_LANE_SCRIPT_DIR | s5/warm_lane_scripts |
| _DF_WARM_LANE_SCRIPT_DIR_ENV | s5/warm_lane_scripts |
| _df_warm_lane_script_dir | s5/warm_lane_scripts |
| _seed_script_text | s5/warm_lane_scripts |
| _seed_script_supports | s5/warm_lane_scripts |
| _seed_script_supports_assume_lane_lock_held | residual/lane_seeding |
| _seed_script_supports_distinct_lock_refusal_rc | residual/lane_seeding |
| _WARM_LANE_AUDIT_CACHE_TTL_SECS | residual/lane_admission |
| _STRUCTURAL_EXHAUSTION_L2_REFIRE_EVERY | residual/warm_acquire |
| PERSISTENT_OFFLINE_DEEP_WORKTREE_NAME | s5-unplaced/lane_paths, s5/worktrees |
| WorktreeKind | s5/worktrees |
| PROTECTED_PREFIXES | s5/worktrees |
| PROTECT_GLOB_OWNED_POOL_BANDS | none |
| default_protected_prefixes | s5/worktrees |
| render_protect_glob | none |
| WorktreeClass | s5/worktrees |
| classify_worktree_entry | none |
| _assert_no_conflict_markers | s5/main_landing |
| MergeResult | s5-split/merging |
| WarmBaseHealth | residual/lane_reset, residual/pool_storage, residual/warm_acquire, s5/warm_lane_scripts, s5/worktrees |
| SeedLaneLock | residual/lane_seeding, s5/worktrees |
| WarmLaneUnavailable | residual/git_ops, residual/lane_outcomes, residual/lane_reset, residual/warm_acquire |
| _seed_rc_to_unavailable | residual/lane_reset, residual/warm_acquire |
| WorktreeInfo | residual/git_ops, residual/warm_acquire, s5/task_branch |
| PoolPrewarmResult | residual/lane_reset |
| _INTERACTIVE_SLUG_RE | s5/worktrees |
| InteractiveWorktreeInfo | s5/worktrees |
| ReapedInteractiveWorktree | s5-split/merge_worktrees |
| _EFFECT_SURVIVAL_AGGREGATE_THRESHOLD | s5-split/commit_effect |
| _EFFECT_SURVIVAL_PER_FILE_THRESHOLD | s5-split/commit_effect |
| _EFFECT_SURVIVAL_PER_FILE_MIN_ADDED_LINES | s5-split/commit_effect |
| _EMPTY_TREE_SHA | s5-split/commit_effect |
| _EFFECT_PROBE_MEMO_MAX_ENTRIES | s5-split/commit_effect |
| _EFFECT_PROBE_TRANSIENT_FAILURES | s5-split/commit_effect |
| _MERGE_MARKER_INDEX_RECHECK_SECS | s5/task_branch |
| _MERGE_MARKER_LOG_FORMAT | s5/task_branch |
| _merge_markers_by_branch | s5/task_branch |
| CommitEffectProbe | residual/git_ops, s5-split/commit_effect |
| ConflictProbe | s5/git_queries |
| GitQueryFailed | s5/git_queries, s5/git_run, s5/task_branch |
| CitationPatternInvalid | s5/git_queries |
| _none_if_query_failed | s5/git_queries, s5/task_branch |
| WorktreeMissing | s5-split/merge_worktrees, s5/git_queries, s5/git_run, s5/task_branch |
| InteractiveWorktreeLimitError | s5/worktrees |
| EphemeralWorktreeError | s5/worktrees |
| WarmLaneRequeue | residual/lane_outcomes |
| WarmLanePoolExhausted | residual/git_ops |
| WarmLaneSoftPressure | residual/git_ops |
| WarmLaneDiskPressure | residual/git_ops |
| WarmLanePoolHardDown | residual/git_ops |
| WarmLaneReseedContaminated | residual/git_ops |
| WarmLaneLockTimeout | residual/git_ops |
| WarmLaneStealFailed | residual/git_ops |
| WarmLaneLockContention | residual/git_ops |
| WorktreeConflictError | s5/worktrees |
| BranchResetError | residual/warm_acquire, s5/task_branch |
| _ParkedBranchStranded | s5/worktrees |
| _stranded_branch | s5/worktrees |
| MergeVerifyLeaseHeld | s5-split/merge_worktrees |
| MergeVerifyLeaseContended | residual/git_ops, s5-split/merge_worktrees, s5/lane_lease |
| LaneLockSelfOwnedLeak | s5/lane_lease |
| _HELD_LANE_LOCK_FDS | s5/lane_lease |
| _HELD_LANE_LOCK_FDS_LOCK | s5/lane_lease |
| _lane_lock_identity | s5/lane_lease |
| _register_held_lane_lock | s5-split/merge_worktrees, s5/lane_lease |
| _release_and_forget_held_lane_lock | s5-split/merge_worktrees, s5/lane_lease |
| _lane_lock_held_in_process | s5/lane_lease |
| _LANE_LOCK_HOLDER_SETTLE_SECS | s5/lane_lease |
| _LANE_LOCK_HOLDER_SETTLE_INTERVAL_SECS | s5/lane_lease |
| _settled_lane_lock_holder_pids | residual/git_ops, s5-split/merge_worktrees |
| _lane_lock_holder_facts | residual/git_ops, s5-split/merge_worktrees |
| _utc_stamp | residual/lane_release, s5/worktrees |
| _run | residual/git_ops, residual/lane_admission, residual/lane_release, residual/lane_reset, residual/lane_seeding, residual/pool_storage, residual/spec_lanes, residual/warm_acquire, s5-split/commit_effect, s5-split/merge_worktrees, s5-split/merging, s5/git_queries, s5/git_run, s5/main_landing, s5/task_branch, s5/warm_lane_scripts, s5/worktrees |
| _git_query | s5/git_queries |
| _FULL_OID | s5/git_queries |
| _git_clean_failure_is_benign | s5/worktrees |
| _git_stderr_is_unresolved_ref | s5/git_queries |
| _merge_worktree_remove_argv | s5-split/merge_worktrees |
| _WHOLE_FILE_SENTINEL | s5/git_diff |
| parse_diff_line_ranges | s5/git_queries |
| parse_diff_added_line_ranges | s5/git_queries |
