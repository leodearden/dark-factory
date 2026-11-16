"""tests/repo_guards is a registered, never-cached module, and the repo-wide guards moved into it stay there.

A guard whose subject is the committed tree outside its own module cannot be
keyed soundly by that module's cache key, so it lives in tests/repo_guards,
whose verify result is never cached. This file pins that placement from
OUTSIDE the module: a guard living inside tests/repo_guards would vanish with
the module and pass vacuously. tests/scripts is no workspace member, so its own
key already covers the whole tree and these reads cannot escape it.
"""

from __future__ import annotations

import ast
import functools
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

if TYPE_CHECKING:
    from collections.abc import Callable

    from orchestrator.config import ModuleConfig

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PREFIX = 'tests/repo_guards'
MODULE_CONFIG_PATH = REPO_ROOT / MODULE_PREFIX / 'orchestrator.yaml'

#: Below the family ceiling, so a wedged collection fails this test with its
#: captured output rather than being killed by pytest-timeout.
COLLECT_TIMEOUT_SECS = 600

#: Node-id substrings that must be collected from tests/repo_guards.
MOVED_GUARDS = (
    'test_reify_closure_staleness_sweep_retired.py',
    'test_design_invariants_consistency.py',
    'TestCheckedInManifestCorpus',
    'TestCheckedInOpenVerdictEvidence',
    'TestCheckedInScriptCheckTargets',
    'test_committed_exemplar_sidecar_validates',
    'test_every_checked_in_sidecar_lives_where_its_prd_derives',
    'test_silent_fallthrough_gate.py::TestWholeTreeGate',
    'test_silent_fallthrough_gate.py::TestGateSelfIntegrity',
    'test_config_dir_archival_gate.py',
    'test_loop_blocking_gate.py',
    'test_orchestrator_rmtree_gate.py',
    'test_usage_gate_shutdown_pairing.py',
    'test_wildcard_mcp_scoping_gate.py',
    'test_tree_scan_sharing.py',
    'test_lock_charter_reify_parity.py::TestFileExtensionsDriftGuard',
    'test_lock_charter_reify_parity.py::TestExtensionlessFilenamesDriftGuard',
    'test_lock_charter_reify_parity.py::TestExtensionlessFilenames::test_no_canonical_name_is_also_a_directory',
    'test_lock_charter_reify_parity.py::TestExtensionlessFilenames::test_every_tracked_extensionless_file_is_allowlisted',
    'test_lock_charter_reify_parity.py::TestReifyCheckoutAdapter',
    'test_lock_charter_reify_parity.py::TestPlantedLayoutHarnessAdapter',
    'test_lock_charter_reify_parity.py::TestReifyCheckoutPlantedLayout',
    'test_lock_charter_reify_parity.py::TestSkipUnlessCheckout',
    'test_lock_charter_reify_parity.py::TestReifyGuardVectorNoneGuard',
    'TestSingleResetsParserOwnership',
    'TestOwnershipScanUsesTheSharedTree',
    'test_only_allowlisted_modules_import_declared_scope',
    'test_the_sweep_sees_the_known_importers',
    'test_forbidden_readers_are_real_and_never_allowed',
    'test_every_allowlist_entry_is_real_unless_pending',
    'test_no_forbidden_reader_names_declared_scope',
    'TestMergeSubmitStatusEnum::test_exact_vocabulary',
    'TestSubmitPartitions::test_submit_terminal_matches_merge_outcome_status',
    'test_upstream_parser_report.py::TestSpecimenAntiFabrication',
    'test_interim_cli_agrees_with_the_committed_fixture',
)

SHARED_TESTS_DIR = REPO_ROOT / 'shared' / 'tests'

#: The first_party_tree gate family, which moves out of shared/tests whole.
FIRST_PARTY_TREE_FAMILY = (
    'test_silent_fallthrough_gate.py',
    'silent_fallthrough_scan.py',
    'silent_fallthrough_allowlist.py',
    'test_config_dir_archival_gate.py',
    'config_dir_archival_allowlist.py',
    'test_loop_blocking_gate.py',
    'loop_blocking_scan.py',
    'loop_blocking_allowlist.py',
    'test_orchestrator_rmtree_gate.py',
    'test_usage_gate_shutdown_pairing.py',
    'test_wildcard_mcp_scoping_gate.py',
    'wildcard_mcp_scoping_scan.py',
    'test_tree_scan_sharing.py',
    'session_prebuild.py',
    'test_session_prebuild.py',
    'test_declared_scope_readers.py',
)

#: Repo-relative homes the whole-file moves vacated.
VACATED_PATHS = (
    'tests/scripts/test_reify_closure_staleness_sweep_retired.py',
    'scripts/tests/test_design_invariants_consistency.py',
    *(f'shared/tests/{name}' for name in FIRST_PARTY_TREE_FAMILY),
    'shared/tests/test_upstream_parser_report.py',
)

#: The scanner whose parse reads every first-party source tree.
FIRST_PARTY_TREE_SCANNER = 'silent_fallthrough_scan'

#: Names a module binds to the repository root.
REPO_ROOT_NAMES = frozenset({'REPO_ROOT', '_REPO_ROOT', 'ROOT'})

#: The capability-manifest corpus walker; with no argument it walks the real tree.
CORPUS_WALKER = 'discover_manifests'


def _is_file_parents_chain(node: ast.expr) -> bool:
    """``Path(__file__)…parents[n]``, through any ``.resolve()``-style links."""
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == 'parents'
    ):
        return False
    link = node.value.value
    while isinstance(link, (ast.Call, ast.Attribute)):
        if isinstance(link, ast.Call) and isinstance(link.func, ast.Name):
            return link.func.id == 'Path' and any(
                isinstance(arg, ast.Name) and arg.id == '__file__' for arg in link.args
            )
        link = link.func if isinstance(link, ast.Call) else link.value
    return False


def _is_repo_root(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Name) and node.id in REPO_ROOT_NAMES
    ) or _is_file_parents_chain(node)


def _is_plans_literal(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and (node.value == 'plans' or node.value.startswith('plans/'))
    )


def _joins_repo_root_to_plans(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Div)
        and _is_repo_root(node.left)
        and _is_plans_literal(node.right)
    )


def _walks_the_real_corpus(node: ast.expr) -> bool:
    if not isinstance(node, ast.Call):
        return False
    callee = node.func
    name = callee.id if isinstance(callee, ast.Name) else getattr(callee, 'attr', None)
    if name != CORPUS_WALKER:
        return False
    arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
    return not arguments or any(_is_repo_root(argument) for argument in arguments)


def _lists_tracked_plans(node: ast.expr) -> bool:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return False
    return any(
        isinstance(element, ast.Constant) and element.value == 'ls-files'
        for element in node.elts
    ) and any(_is_plans_literal(element) for element in node.elts)


def plans_tree_references(source: str) -> list[str]:
    """Every site in *source* that names the committed ``plans/`` tree.

    Three shapes: a repository-root anchor joined to ``plans``, a corpus walk
    over the real tree, and a ``git ls-files`` argv naming ``plans``. A site is
    flagged for how its path is built, whether or not anything reads it.
    """
    return [
        f'line {node.lineno}: {ast.unparse(node)}'
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.expr)
        and (
            _joins_repo_root_to_plans(node)
            or _walks_the_real_corpus(node)
            or _lists_tracked_plans(node)
        )
    ]


@functools.cache
def _collected_node_ids() -> tuple[str, ...]:
    result = subprocess.run(
        [
            sys.executable, '-m', 'pytest', '--collect-only', '-q',
            '-p', 'no:cacheprovider', f'{MODULE_PREFIX}/',
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=COLLECT_TIMEOUT_SECS,
    )
    assert result.returncode == 0, (
        f'collecting {MODULE_PREFIX}/ exited {result.returncode}:\n'
        f'{result.stdout}\n{result.stderr}'
    )
    return tuple(line for line in result.stdout.splitlines() if '::' in line)


def test_the_module_is_registered_and_never_cached(
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> None:
    discovered = discover_module_configs()
    assert MODULE_PREFIX in discovered, (
        f'{MODULE_CONFIG_PATH.relative_to(REPO_ROOT)} is not registered by the '
        f'production module-config walk. Discovered: {sorted(discovered)}'
    )
    assert discovered[MODULE_PREFIX].verify_cache == 'never', (
        f'{MODULE_PREFIX} must declare verify_cache: never, because its guards '
        'read the tree outside any cache key'
    )


def test_every_moved_guard_is_collected_from_the_module() -> None:
    collected = _collected_node_ids()
    missing = [
        guard for guard in MOVED_GUARDS
        if not any(guard in node_id for node_id in collected)
    ]
    assert not missing, (
        f'these moved guards are not collected from {MODULE_PREFIX}/: {missing}'
    )


def test_no_moved_file_remains_at_its_old_path() -> None:
    remaining = [path for path in VACATED_PATHS if (REPO_ROOT / path).exists()]
    assert not remaining, f'moved into {MODULE_PREFIX}/ but still present: {remaining}'


@pytest.mark.parametrize(
    'source',
    [
        "REPO_ROOT / 'plans' / 'x.yaml'",
        "_REPO_ROOT / 'plans/x-prd.md'",
        "ROOT / 'plans'",
        "Path(__file__).resolve().parents[2] / 'plans'",
        "Path(__file__).parents[2] / 'plans/x.yaml'",
        'discover_manifests()',
        'discover_manifests(REPO_ROOT)',
        'corpus.discover_manifests(repo_root=REPO_ROOT)',
        "subprocess.run(['git', 'ls-files', '--', 'plans'])",
        "subprocess.run(['git', '-C', str(root), 'ls-files', '-z', 'plans/'])",
    ],
)
def test_plans_tree_references_flags_a_committed_plans_path(source: str) -> None:
    assert plans_tree_references(source), source


@pytest.mark.parametrize(
    'source',
    [
        "tmp_path / 'plans'",
        "tmp_path / 'plans' / 'x.yaml'",
        "prd = 'plans/x.yaml'",
        'discover_manifests(tmp_path)',
        "REPO_ROOT / 'docs' / 'prds'",
        "subprocess.run(['git', 'ls-files', '--', 'docs'])",
    ],
)
def test_plans_tree_references_passes_synthetic_plans_data(source: str) -> None:
    assert plans_tree_references(source) == [], source


def _absolute_imports(source: str) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module)
    return imported


def test_no_shared_test_imports_the_first_party_tree_scanner() -> None:
    importers = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted(SHARED_TESTS_DIR.rglob('*.py'))
        if FIRST_PARTY_TREE_SCANNER in _absolute_imports(path.read_text(encoding='utf-8'))
    ]
    assert not importers, (
        f'these shared/tests files import {FIRST_PARTY_TREE_SCANNER}, the '
        f'whole-first-party-tree scanner that lives in {MODULE_PREFIX}/: {importers}'
    )


def test_shared_tests_reference_no_committed_plans() -> None:
    references = {
        str(path.relative_to(REPO_ROOT)): found
        for path in sorted(SHARED_TESTS_DIR.rglob('*.py'))
        if (found := plans_tree_references(path.read_text(encoding='utf-8')))
    }
    assert not references, (
        f'shared/tests names the committed plans/ tree, which its cache key does '
        f'not cover; move that test into {MODULE_PREFIX}/: {references}'
    )
