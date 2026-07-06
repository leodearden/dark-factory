# Capability manifest — gitops-chokepoints PRD (stream M1)

Binds each task signal's asserted capabilities to evidence on main (code-verified
2026-07-06, HEAD f768b47dd9). Mechanizes G3+G6 per `skills/prd/references/gates.md`.
Any FAIL binding blocks queueing. All bindings below PASS.

## task-α — `_prune_registrations(context)` chokepoint + convert 5 raw git_ops sites

| Capability | Evidence |
|---|---|
| `pool_in_use()` exists + wired into production guard | grep:orchestrator/src/orchestrator/git_ops.py:915 (def); wired at :6788 (`prune_worktrees` guard) |
| `pool_storage_present()` exists + wired | grep:git_ops.py:930 (def); wired at :6788 |
| `_note_pool_storage_absent()` exists + wired | grep:git_ops.py:977 (def); wired at :6806 |
| `_pool_storage_bootstrap_ok()` exists + wired | grep:git_ops.py:994 (def); wired at :6789 |
| Raw site: create_worktree leftover cleanup | grep:git_ops.py:3824 (`await _run(['git', 'worktree', 'prune']...`) |
| Raw site: materialize_member_solo pre-clean | grep:git_ops.py:4341 |
| Raw site: delete_solo_branch | grep:git_ops.py:4426 |
| Raw site: prune_stale_merge_worktrees | grep:git_ops.py:5365 |
| Raw site: reap_interactive_worktrees | grep:git_ops.py:5725 |
| Guarded home to relocate: prune_worktrees | grep:git_ops.py:6758-6812 (task-2099 guard + argv at :6807) |

## task-β — harness substrate-gate cleanup via chokepoint

| Capability | Evidence |
|---|---|
| `_prune_registrations` / `prune_worktrees(context=…)` | producer:task-α upstream (DAG-direction PASS) |
| Harness raw tuple-argv prune site | grep:orchestrator/src/orchestrator/harness.py:4779-4781 (`('git', 'worktree', 'prune')` in `for _cleanup_argv` loop) |
| Harness already holds a GitOps handle | grep:harness.py:4765 (`self.git_ops.worktree_base`), :4767 (`self.git_ops.resolve_branch_sha`) |

## task-γ (leaf) — CI grep-guard: exactly one prune argv in orchestrator/src

| Capability | Evidence |
|---|---|
| Numeric-exactness premise "exactly one occurrence" | census 2026-07-06: 7 argv occurrences total (git_ops.py:3824,4341,4426,5365,5725,6807 + harness.py:4781); α converts 5 + relocates 1, β converts 1 ⇒ exactly 1 remains, inside `_prune_registrations` |
| Producers of the converted state | producer:task-α + producer:task-β, both upstream (DAG-direction PASS) |
| Source-scan guard-test pattern (AST, comment-immune) | grep:orchestrator/tests/test_event_loop_antipattern_guard.py:38-56 (AST walk, rglob, offender file:line list) — prose mentions of the phrase (verify.py:3288,3408,3489; git_ops docstrings) are comments/strings, invisible to a List/Tuple-of-Constants AST match |
| Rejection-mechanism ("planting a 2nd literal fails CI") | rejection-check: the test itself is the mechanism and is authored by this task; RED baseline = main today has 7 occurrences, so the test is trivially demonstrable in both directions during TDD |

## task-δ (leaf) — `_abort_lane_acquisition` fault-teardown primitive

| Capability | Evidence |
|---|---|
| Detach machinery (`git checkout --detach`) | grep:git_ops.py:3383 (release_warm_lane), :3427 (detach_lane_checkout) — wired on production release paths |
| Commit-WIP-before-detach contract | grep:git_ops.py:3424-3426 (detach_lane_checkout calls `self.commit(...)` first) |
| Branch-retention rule | grep:git_ops.py:3735 (`_delete_branch_if_on_main` def); wired at :3396 (release_warm_lane) |
| Degenerate-ref check | grep:git_ops.py:2136 (`warm_lane_ref_is_degenerate` def); wired at :2936 (create-once seed-fault, task 2112) |
| Pool release | grep:git_ops.py:3064 (`self.warm_lane_pool.release(lane)`) — the bare-release defect site this task fixes |
| Fault-injection trigger (`commit()` raise) | grep:git_ops.py:3887 (`raise RuntimeError(f'Commit failed: {err}')`), reached from `_reuse_warm_lane` |
| Rejection-assertion ("no `already used by worktree` after fault") | rejection-check: collision mechanism is git's single-checkout lock, proven live (task 2062 description; memory 2026-07-03 acquire-fault two-root-causes); RED baseline reproducible today by injecting a fault into the top-level except path (:3060-3065 does NOT detach) |

## task-ε (leaf) — PROTECTED_PREFIXES registry + foreign-band refusal

| Capability | Evidence |
|---|---|
| `_lane-` band literal | grep:orchestrator/src/orchestrator/warm_lane_pool.py:42 (`name_prefix: str = '_lane-'`) |
| `_iact-` band (config-driven) | grep:orchestrator/src/orchestrator/config.py:1161 (`iact_prefix` Field) |
| `_merge-` band + persistent `_merge-verify` | grep:git_ops.py:160 (PERSISTENT_MERGE_WORKTREE_NAME), :5297 (startswith filter) |
| `_offline-deep` persistent name | grep:git_ops.py:167 (PERSISTENT_OFFLINE_DEEP_WORKTREE_NAME) |
| `_solo-` band | grep:git_ops.py:4289 (`solo_prefix: str = '_solo-'`) |
| `_spec-` band | grep:git_ops.py:167-179 region + config.py:1167 (prefix documented; spec_warm_lane_pool) |
| `_substrate-gate-` band | grep:harness.py:4760 (`f'_substrate-gate-{task_id}'`) |
| Destructive sites to wire | grep:git_ops.py:5353 (prune_stale_merge_worktrees remove), :5711 (reap_interactive_worktrees remove), :1579 (create_worktree self-heal rmtree), :1792 (create_interactive_worktree self-heal rmtree); harness.py:4780 (substrate-gate remove) |
| Rejection-assertion ("foreign-band removal refused with WARNING") | rejection-check: the refusal mechanism is authored BY this task (new); RED baseline = today a mis-steered path deletes silently (adjacent live proof: Jul-5 lane-dir deletion killed task 4791's plan.json — memory project_warm_lane_orphaned_registration_reuse_fault_2026_07_04) |
