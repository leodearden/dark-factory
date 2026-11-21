"""Every name imported from the ``orchestrator.merge_lane`` façade exists.

PRD ``plans/merge-lane-quality-prd.md`` tasks zeta2 and eta. The façade's export
table is a module ``__getattr__`` typed ``Any``, so pyright accepts
``from orchestrator.merge_lane import N`` for any ``N`` at all: a stale name
passes every static gate and fails only when the line runs. The escalation
server is the importer this matters most for. Its façade imports are lazy,
function-local and outside its own pyright scope, and they run only when a
merge MCP tool is called.

So every tracked file outside the package is swept, and each name a
``from orchestrator.merge_lane import N`` binds must be one the live façade
exports, or one of its submodules. Imports from a submodule
(``orchestrator.merge_lane.worker``) are typed, and pyright checks them.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from _tree_scan import (
    REPO_ROOT,
    TREE_PASS_XDIST_GROUP,
    ParsedModule,
    TreeAnalysis,
    TreePass,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

# The family mark and shared group: shared/src/shared/whole_tree_guard_family.py::WHOLE_TREE_SCAN_TEST_TIMEOUT, _tree_scan.py.
pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP),
]

# Same import of the scripts directory as test_merge_lane_ratchet.py:
# source_measures lists the tracked files.
_SCRIPTS = REPO_ROOT / 'scripts'
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import source_measures  # type: ignore[import-not-found]  # noqa: E402

_FACADE = 'orchestrator.merge_lane'
#: Every spelling of a ``from`` import of the façade names this identifier,
#: however its dotted name is spaced.
_FACADE_LEAF = _FACADE.rpartition('.')[2]

#: The package's own files import their siblings, never the façade.
_PACKAGE_DIR = 'orchestrator/src/orchestrator/merge_lane/'

#: Files carrying a façade from-import: 58 when task 5037 deleted the aliases.
#: A floor far below that, so ordinary churn never reddens it and a sweep that
#: stopped reading the tree still does.
_FACADE_IMPORTER_FLOOR = 40


def facade_imports(source: str, *, path: str) -> list[tuple[int, str]]:
    """``(lineno, name)`` for every name a ``from orchestrator.merge_lane import``
    in *source* binds, at any depth."""
    return facade_imports_in(ast.parse(source, filename=path))


def facade_imports_in(tree: ast.Module) -> list[tuple[int, str]]:
    """:func:`facade_imports` over an already-parsed module."""
    return [
        (node.lineno, imported.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == _FACADE and not node.level
        for imported in node.names
        if imported.name != '*'
    ]


def _tracked_files_outside_the_package() -> tuple[Path, ...]:
    return tuple(
        REPO_ROOT / relpath
        for relpath in source_measures.tracked_python_files(REPO_ROOT)
        if not relpath.startswith(_PACKAGE_DIR)
    )


def _module_facade_imports(module: ParsedModule) -> tuple[tuple[int, str], ...]:
    """:func:`facade_imports` over *module*, parsing it only if it can name the façade's last part."""
    if not module.may_name(_FACADE_LEAF):
        return ()
    return tuple(facade_imports_in(module.tree))


ANALYSIS = TreeAnalysis(
    name=Path(__file__).stem,
    files=_tracked_files_outside_the_package,
    extract=_module_facade_imports,
)
TREE_ANALYSES = (ANALYSIS,)


def _is_submodule(name: str) -> bool:
    """``from <package> import <submodule>`` imports it even when no attribute exists yet."""
    return importlib.util.find_spec(f'{_FACADE}.{name}') is not None


def stale_facade_names(
    imports: list[tuple[int, str]], *, path: str, facade: ModuleType,
) -> list[str]:
    """The entries of *imports* that the live *facade* neither exports nor holds as a submodule."""
    return [
        f'{path}:{lineno}: {_FACADE}.{name}'
        for lineno, name in imports
        if not hasattr(facade, name) and not _is_submodule(name)
    ]


@pytest.fixture(scope='module')
def facade() -> ModuleType:
    return importlib.import_module(_FACADE)


def test_no_tracked_file_imports_a_missing_name_from_the_facade(
    tree_pass: TreePass, facade: ModuleType,
) -> None:
    facade_importers = {
        path.relative_to(REPO_ROOT).as_posix(): imports
        for path, imports in tree_pass.facts(ANALYSIS).items()
        if imports
    }
    importers = len(facade_importers)
    stale = [
        entry
        for relpath, imports in facade_importers.items()
        for entry in stale_facade_names(list(imports), path=relpath, facade=facade)
    ]
    assert importers >= _FACADE_IMPORTER_FLOOR, (
        f'only {importers} files import from the façade; the sweep is not reading the tree'
    )
    assert not stale, 'names imported from the façade that it does not have:\n' + '\n'.join(stale)


@pytest.mark.parametrize('source', [
    pytest.param('from orchestrator.ｍerge_lane import NoSuchName\n', id='compatibility-spelling'),
    pytest.param('from orchestrator . merge_lane import NoSuchName\n', id='spaced-dot'),
])
def test_sweep_reads_every_spelling_of_the_facade_the_parser_accepts(
    tmp_path: Path, source: str,
) -> None:
    path = tmp_path / 'importer.py'
    path.write_text(source, encoding='utf-8')

    assert _module_facade_imports(ParsedModule(path)) == ((1, 'NoSuchName'),)


class TestStaleFacadeNames:
    """Seeded fixtures: a stale façade name is flagged, a live one is not."""

    @pytest.mark.parametrize('source', [
        pytest.param('from orchestrator.merge_lane import NoSuchName\n', id='module-level'),
        pytest.param(
            'def f():\n    from orchestrator.merge_lane import MergeLane, NoSuchName\n',
            id='function-local-beside-a-live-name',
        ),
        pytest.param(
            'from orchestrator.merge_lane import (\n    MergeOutcome,\n    NoSuchName as Alias,\n)\n',
            id='parenthesised-and-renamed',
        ),
    ])
    def test_a_stale_name_is_flagged(self, source: str, facade: ModuleType) -> None:
        imports = facade_imports(source, path='fixture.py')
        stale = stale_facade_names(imports, path='fixture.py', facade=facade)
        assert len(stale) == 1 and 'NoSuchName' in stale[0], stale

    @pytest.mark.parametrize('source', [
        pytest.param('from orchestrator.merge_lane import MergeLane, WaiterRecord\n', id='exports'),
        pytest.param('from orchestrator.merge_lane import landed_outbox\n', id='a-submodule'),
        pytest.param(
            'from orchestrator.merge_lane.worker import NoSuchName\n',
            id='a-submodule-import-is-pyrights-to-check',
        ),
        pytest.param('from orchestrator import merge_lane\n', id='the-package-itself'),
    ])
    def test_a_live_or_unrelated_name_is_not_flagged(
        self, source: str, facade: ModuleType,
    ) -> None:
        imports = facade_imports(source, path='fixture.py')
        assert stale_facade_names(imports, path='fixture.py', facade=facade) == []
