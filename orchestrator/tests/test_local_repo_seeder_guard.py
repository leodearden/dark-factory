"""No orchestrator/tests module hand-rolls a ``git init`` + ``git commit`` repo seeder:
seeding goes through ``_git_fixtures.py::seed_repo``.

The sweep covers every ``orchestrator/tests/*.py``, so a new seeder fails by
default.  ``_NOT_YET_MIGRATED`` names the modules that still carry one; it
only shrinks, because an entry whose module no longer seeds its own repo fails
too.  Migrating a module means deleting its entry.
"""
from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from _tree_scan import (
    TESTS_DIR,
    TREE_PASS_XDIST_GROUP,
    ParsedModule,
    TreeAnalysis,
    TreePass,
    python_files,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

# The family mark and shared group: shared/src/shared/whole_tree_guard_family.py::WHOLE_TREE_SCAN_TEST_TIMEOUT, _tree_scan.py.
pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP),
]

_NOT_YET_MIGRATED = frozenset({
    '_merge_deep_scene.py',
    '_verify_merge_cli_fixtures.py',  # debt: task 11025
    'test_agent_capability_wiring.py',
    'test_architect_all_committed_plan.py',
    'test_atomic_train_merge.py',
    'test_create_merge_worktree_retry.py',
    'test_deterministic_provenance_wire_e2e.py',  # debt: task 11025
    'test_dry_run_unblock.py',  # debt: task 11026
    'test_git_repo_isolation_guard.py',
    'test_harness_infra_resume_truthful.py',
    'test_harness_plan_step_rederive.py',
    'test_harness_wip_step_detection.py',
    'test_merge_drift.py',
    'test_merge_gates_drop_guard_rename.py',
    'test_merge_gates_equivalence_rename.py',
    'test_merge_gates_plan_files_rename.py',
    'test_merge_guard_pipeline.py',
    'test_merge_item_union.py',
    'test_merge_lane_package.py',
    'test_merge_queue_bounce.py',
    'test_merge_queue_c3_submit_identity.py',
    'test_merge_queue_conflict_graph.py',
    'test_merge_queue_dry_run_unblock.py',
    'test_merge_queue_duplicate_submission.py',
    'test_merge_queue_finalize_head_visibility.py',
    'test_merge_queue_frozen_prefix.py',
    'test_merge_queue_invariant_integration_gate.py',
    'test_merge_queue_multihost_wiring.py',
    'test_merge_queue_orphan_reaper.py',
    'test_merge_queue_phase_derivation.py',
    'test_merge_queue_redispatch_entries.py',
    'test_merge_queue_resource_audit.py',
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
    'test_rebase_verify_cost.py',
    'test_reconcile_done_step_commits.py',
    'test_reopen_sticks_e2e.py',  # debt: task 11025
    'test_roles_ancestry_check.py',  # debt: task 11025
    'test_roles_staging_command.py',  # debt: task 11026
    'test_scheduler_park_owner_state.py',  # debt: task 11026
    'test_session_hooks_install.py',
    'test_session_resume_integration_gate.py',
    'test_suffix_conflict_tracker.py',
    'test_train_integration.py',
    'test_verify_events_units.py',  # debt: task 11025
    'test_verify_phase_rebase.py',
    'test_verify_scope_inversion_boundary.py',
    'test_workflow.py',
    'test_workflow_agent_session_preserve.py',
    'test_workflow_escalated_steward_stall.py',
    'test_workflow_signature_loop_guard.py',
    'test_workflow_status_on_resume.py',
    'test_workflow_truthful_commit_stamp.py',
    'test_workflow_verify_retry.py',
})


_GLOBAL_OPTIONS_WITH_A_VALUE = frozenset({'-C', '-c', '--git-dir', '--work-tree', '--namespace'})


def _verb(argv: Sequence[object]) -> str | None:
    """The git subcommand after git's global options; None if there is none or a non-constant element precedes it."""
    elements = iter(argv[1:])
    for element in elements:
        if not isinstance(element, str):
            return None
        if element in _GLOBAL_OPTIONS_WITH_A_VALUE:
            next(elements, None)
        elif not element.startswith('-'):
            return element
    return None


@dataclass(frozen=True)
class _GitArgv:
    """One ``['git', ...]`` list literal: its subcommand and every element, non-constant ones as None."""

    verb: str
    argv: tuple[object, ...]


def _git_argvs(func: ast.AST) -> list[_GitArgv]:
    """Every ``['git', ...]`` list literal under *func* whose verb, read after git's global options, is known."""
    calls: list[_GitArgv] = []
    for node in ast.walk(func):
        if isinstance(node, ast.List):
            argv = tuple(
                elt.value if isinstance(elt, ast.Constant) else None for elt in node.elts
            )
            if argv and argv[0] == 'git' and (verb := _verb(argv)) is not None:
                calls.append(_GitArgv(verb, argv))
    return calls


def _local_repo_seeders(tree: ast.Module) -> list[str]:
    """Module-level functions that both ``git init`` a non-bare repo and ``git commit``."""
    seeders: list[str] = []
    for func in tree.body:
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            calls = _git_argvs(func)
            if (
                any(call.verb == 'init' and '--bare' not in call.argv for call in calls)
                and any(call.verb == 'commit' for call in calls)
            ):
                seeders.append(func.name)
    return seeders


def _top_level_test_modules() -> tuple[Path, ...]:
    return tuple(path for path in python_files(TESTS_DIR) if path.parent == TESTS_DIR)


def _module_seeders(module: ParsedModule) -> tuple[str, ...]:
    return tuple(_local_repo_seeders(module.tree))


ANALYSIS = TreeAnalysis(name=Path(__file__).stem, files=_top_level_test_modules, extract=_module_seeders)
TREE_ANALYSES = (ANALYSIS,)


@pytest.fixture(scope='module')
def seeders_by_module(tree_pass: TreePass) -> dict[str, list[str]]:
    return {
        path.name: list(seeders)
        for path, seeders in tree_pass.facts(ANALYSIS).items()
        if seeders
    }


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


def test_a_seeder_that_commits_through_a_dash_upper_c_path_is_flagged() -> None:
    tree = ast.parse(
        'def _init_git_repo(path):\n'
        '    p = str(path)\n'
        "    subprocess.run(['git', 'init', '-b', 'main', p])\n"
        "    subprocess.run(['git', '-C', p, 'commit', '-m', 'initial commit'])\n"
    )

    assert _local_repo_seeders(tree) == ['_init_git_repo']


def test_a_seeder_that_commits_through_dash_c_config_options_is_flagged() -> None:
    tree = ast.parse(
        'def _commit_main(root):\n'
        "    subprocess.run(['git', 'init', '-q', '-b', 'main'], cwd=root)\n"
        '    subprocess.run(\n'
        "        ['git', '-c', 'user.name=t', '-c', 'user.email=t@example.com',\n"
        "         'commit', '-q', '--allow-empty', '-m', 'main'],\n"
        '        cwd=root,\n'
        '    )\n'
    )

    assert _local_repo_seeders(tree) == ['_commit_main']


def test_a_seeder_that_commits_through_git_dir_and_work_tree_is_flagged() -> None:
    tree = ast.parse(
        'def _seed(repo, git_dir):\n'
        "    subprocess.run(['git', 'init', str(repo)])\n"
        "    subprocess.run(['git', '--git-dir', git_dir, '--work-tree=/w', 'commit', '-m', 'x'])\n"
    )

    assert _local_repo_seeders(tree) == ['_seed']


@pytest.mark.parametrize('option', ['-C', '-c', '--git-dir', '--work-tree', '--namespace'])
def test_a_seeder_that_commits_through_any_global_option_with_a_value_is_flagged(
    option: str,
) -> None:
    tree = ast.parse(
        'def _seed(repo, value):\n'
        "    subprocess.run(['git', 'init', str(repo)])\n"
        f"    subprocess.run(['git', {option!r}, value, 'commit', '-m', 'x'])\n"
    )

    assert _local_repo_seeders(tree) == ['_seed']


def test_a_git_literal_with_no_subcommand_is_skipped() -> None:
    tree = ast.parse(
        'def _probe(repo):\n'
        "    subprocess.run(['git', '--version'])\n"
        "    subprocess.run(['git', 'init', str(repo)])\n"
        "    subprocess.run(['git', '-C'])\n"
    )

    assert _local_repo_seeders(tree) == []


def test_a_global_bare_init_is_not_flagged() -> None:
    tree = ast.parse(
        'def _make_origin(origin, p):\n'
        "    subprocess.run(['git', '--bare', 'init'], cwd=origin)\n"
        "    subprocess.run(['git', '-C', p, 'commit', '--allow-empty', '-m', 'x'])\n"
    )

    assert _local_repo_seeders(tree) == []


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
