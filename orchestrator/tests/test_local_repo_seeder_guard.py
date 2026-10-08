"""No orchestrator/tests module hand-rolls a ``git init`` + ``git commit`` repo seeder:
seeding goes through ``_git_fixtures.py::seed_repo``.

The sweep covers every ``orchestrator/tests/*.py``, so a new seeder fails by
default.  ``_NOT_YET_MIGRATED`` names the modules that still carry one; it
only shrinks, because an entry whose module no longer seeds its own repo fails
too.  Migrating a module means deleting its entry.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT

# Whole-tree scanner: see WHOLE_TREE_SCAN_TEST_TIMEOUT in _orch_helpers.py and
# test_whole_tree_scan_timeout_guard.py (task 4215).
pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_TESTS_DIR = Path(__file__).parent

_NOT_YET_MIGRATED = frozenset({
    '_merge_deep_scene.py',
    'test_agent_capability_wiring.py',
    'test_architect_all_committed_plan.py',
    'test_atomic_train_merge.py',
    'test_convert_to_blocked.py',
    'test_create_merge_worktree_retry.py',
    'test_git_repo_isolation_guard.py',
    'test_harness_infra_resume_truthful.py',
    'test_harness_interactive_reaper.py',
    'test_harness_plan_step_rederive.py',
    'test_harness_wip_step_detection.py',
    'test_inflight_verify_merge_lease.py',
    'test_interactive_warm_worktree_integration_gate.py',
    'test_interactive_worktree.py',
    'test_interactive_worktree_reaper.py',
    'test_invoke_role_config_resolution.py',
    'test_lane_lifecycle_gitops.py',
    'test_lane_lifecycle_integration.py',
    'test_merge_drift.py',
    'test_merge_gates_drop_guard_rename.py',
    'test_merge_gates_equivalence_rename.py',
    'test_merge_gates_plan_files_rename.py',
    'test_merge_guard_pipeline.py',
    'test_merge_item_union.py',
    'test_merge_lane_package.py',
    'test_merge_queue_bounce.py',
    'test_merge_queue_build_chain.py',
    'test_merge_queue_c3_submit_identity.py',
    'test_merge_queue_chain_intact.py',
    'test_merge_queue_coalesce.py',
    'test_merge_queue_concurrent_verify.py',
    'test_merge_queue_conflict_graph.py',
    'test_merge_queue_deep_dispatch.py',
    'test_merge_queue_depth_telemetry.py',
    'test_merge_queue_dry_run_unblock.py',
    'test_merge_queue_duplicate_submission.py',
    'test_merge_queue_equivalence.py',
    'test_merge_queue_finalize_head_visibility.py',
    'test_merge_queue_frozen_prefix.py',
    'test_merge_queue_host_observability.py',
    'test_merge_queue_invariant_integration_gate.py',
    'test_merge_queue_lifecycle_registry.py',
    'test_merge_queue_metrics.py',
    'test_merge_queue_multihost_wiring.py',
    'test_merge_queue_orphan_reaper.py',
    'test_merge_queue_permit_conservation.py',
    'test_merge_queue_persistent_worktree.py',
    'test_merge_queue_phase_derivation.py',
    'test_merge_queue_redispatch_entries.py',
    'test_merge_queue_request_liveness.py',
    'test_merge_queue_resolve_release.py',
    'test_merge_queue_resource_audit.py',
    'test_merge_queue_restart_hook.py',
    'test_merge_queue_single_writer_asserts.py',
    'test_merge_queue_supervisor.py',
    'test_merge_queue_two_layer_integration.py',
    'test_merge_queue_verify_base_invariant.py',
    'test_merge_queue_warm_cold_shadow.py',
    'test_merge_serial_lane_tripwire.py',
    'test_merge_shadow.py',
    'test_merge_speculation.py',
    'test_merge_verify_lease_guard.py',
    'test_merge_verify_survivor_barrier.py',
    'test_merge_worktree_lifecycle_integration_gate.py',
    'test_multihost_verify_integration.py',
    'test_offline_lane_infra_integration.py',
    'test_offline_lane_integration.py',
    'test_pool_storage_guard.py',
    'test_protected_prefixes.py',
    'test_provenance_gate_integration.py',
    'test_rebase_branch_reset_guard.py',
    'test_rebase_verify_cost.py',
    'test_reconcile_done_step_commits.py',
    'test_remove_merge_worktree_guarded.py',
    'test_session_hooks.py',
    'test_session_resume_integration_gate.py',
    'test_substrate_gate.py',
    'test_suffix_conflict_tracker.py',
    'test_task_runtime.py',
    'test_train_integration.py',
    'test_verify_phase_rebase.py',
    'test_verify_plan_integration.py',
    'test_verify_scope_inversion_boundary.py',
    'test_warm_base_coherence.py',
    'test_warm_lane_abort_teardown.py',
    'test_warm_lane_disk_guard.py',
    'test_warm_lane_integration_gate.py',
    'test_warm_lane_reseed_verify.py',
    'test_warm_lane_seed_scrub.py',
    'test_warm_lane_soft_floor.py',
    'test_warm_lane_steal_retry.py',
    'test_warm_lane_structural_exhaustion.py',
    'test_workflow.py',
    'test_workflow_agent_session_preserve.py',
    'test_workflow_escalated_steward_stall.py',
    'test_workflow_merge_gating_strand.py',
    'test_workflow_signature_loop_guard.py',
    'test_workflow_status_on_resume.py',
    'test_workflow_truthful_commit_stamp.py',
    'test_workflow_verify_retry.py',
})


def _git_argvs(func: ast.AST) -> list[list[object]]:
    """Every ``['git', <verb>, ...]`` list literal under *func*; non-constant elements are None."""
    argvs: list[list[object]] = []
    for node in ast.walk(func):
        if isinstance(node, ast.List):
            argv: list[object] = [
                elt.value if isinstance(elt, ast.Constant) else None for elt in node.elts
            ]
            if len(argv) >= 2 and argv[0] == 'git' and isinstance(argv[1], str):
                argvs.append(argv)
    return argvs


def _local_repo_seeders(tree: ast.Module) -> list[str]:
    """Module-level functions that both ``git init`` a non-bare repo and ``git commit``."""
    seeders: list[str] = []
    for func in tree.body:
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            argvs = _git_argvs(func)
            if (
                any(argv[1] == 'init' and '--bare' not in argv for argv in argvs)
                and any(argv[1] == 'commit' for argv in argvs)
            ):
                seeders.append(func.name)
    return seeders


@pytest.fixture(scope='module')
def seeders_by_module() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for module in sorted(_TESTS_DIR.glob('*.py')):
        seeders = _local_repo_seeders(ast.parse(module.read_text(encoding='utf-8')))
        if seeders:
            found[module.name] = seeders
    return found


def test_no_module_grows_its_own_repo_seeder(seeders_by_module: dict[str, list[str]]) -> None:
    unexpected = {
        name: seeders for name, seeders in seeders_by_module.items()
        if name not in _NOT_YET_MIGRATED
    }
    assert unexpected == {}, (
        f'local repo seeder(s) {unexpected}: call _git_fixtures.seed_repo instead, '
        'with a RepoSeed for any contents other than the README seed.'
    )


def test_every_exemption_still_seeds_its_own_repo(
    seeders_by_module: dict[str, list[str]],
) -> None:
    stale = sorted(_NOT_YET_MIGRATED - seeders_by_module.keys())
    assert stale == [], (
        f'{stale} no longer define a local repo seeder: delete them from _NOT_YET_MIGRATED.'
    )


def test_the_legacy_seeder_is_flagged() -> None:
    tree = ast.parse(
        'async def _setup_repo(repo):\n'
        "    await _run(['git', 'init', '-b', 'main'], cwd=repo)\n"
        "    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)\n"
        "    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)\n"
        "    (repo / 'README.md').write_text('# Test\\n')\n"
        "    await _run(['git', 'add', '-A'], cwd=repo)\n"
        "    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)\n"
    )

    assert _local_repo_seeders(tree) == ['_setup_repo']


def test_a_bare_origin_helper_is_not_flagged() -> None:
    tree = ast.parse(
        'async def _make_origin(origin, seed):\n'
        "    await _run(['git', 'init', '--bare', '-b', 'main'], cwd=origin)\n"
        "    await _run(['git', 'push', str(origin), 'main'], cwd=seed)\n"
    )

    assert _local_repo_seeders(tree) == []


def test_a_clone_then_commit_helper_is_not_flagged() -> None:
    tree = ast.parse(
        'async def _clone_and_commit(origin, local):\n'
        "    await _run(['git', 'clone', str(origin), str(local)])\n"
        "    await _run(['git', 'commit', '--allow-empty', '-m', 'x'], cwd=local)\n"
    )

    assert _local_repo_seeders(tree) == []
