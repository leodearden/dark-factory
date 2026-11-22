"""The merge-lane instrument's files import each other in one order only.

:data:`STRATA` is the one home of that order, lowest first: each file may
import a stratum module only from an EARLIER entry, and only as a statement of
its module body. No module rebinds an object a lower module, or
``scripts/source_measures.py`` beneath them all, defines; consumers reach each
name through its defining module. :data:`PLACEMENT_CASES` drives the placement
rule over small synthetic sources, so each placement it accepts or refuses is
shown to be accepted or refused. The layers on the import path that binds the
checkout, :data:`BINDING_PATH_LAYERS`, import no workspace member at all.
"""

import ast
import importlib
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import _checkout
import pytest
import source_measures
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / 'scripts'

#: The instrument's files, lowest stratum first.
STRATA = ('merge_lane_cluster', 'merge_lane_ratchet', 'merge_lane_metrics')


def _source_of(stem: str) -> str:
    return (SCRIPTS_DIR / f'{stem}.py').read_text(encoding='utf-8')


def _strata_imports(tree: ast.Module) -> list[tuple[ast.stmt, str]]:
    """Every import in *tree* that names a stratum module, with the module it names."""
    found: list[tuple[ast.stmt, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        found.extend((node, name) for name in names if name in STRATA)
    return found


def _placement_violations(stem: str, source: str) -> list[str]:
    """Every stratum import in *source*, read as ``scripts/<stem>.py``, that breaks the order."""
    tree = ast.parse(source, filename=f'scripts/{stem}.py')
    module_body = {id(statement) for statement in tree.body}
    importer = STRATA.index(stem)
    violations: list[str] = []
    for node, imported in _strata_imports(tree):
        where = f'scripts/{stem}.py:{node.lineno} imports {imported}'
        if STRATA.index(imported) >= importer:
            violations.append(
                f'{where} (STRATA index {STRATA.index(imported)}), which is not '
                f'earlier than {stem} (STRATA index {importer})'
            )
        if id(node) not in module_body:
            violations.append(
                f'{where} inside a function, class or conditional; stratum '
                'imports are module-body statements'
            )
    return violations


def test_every_stratum_has_a_file() -> None:
    missing = [stem for stem in STRATA if not (SCRIPTS_DIR / f'{stem}.py').is_file()]
    assert missing == []


def test_each_stratum_imports_only_earlier_strata_at_module_top_level() -> None:
    violations = [
        violation
        for stem in STRATA
        for violation in _placement_violations(stem, _source_of(stem))
    ]

    assert violations == [], '\n'.join(violations)


def test_no_scripts_module_imports_the_cli() -> None:
    cli = STRATA[-1]
    importers = [
        f'{path.relative_to(SCRIPTS_DIR.parent).as_posix()}:{node.lineno}'
        for path in sorted(SCRIPTS_DIR.glob('*.py'))
        if path.stem != cli and cli in (text := path.read_text(encoding='utf-8'))
        for node, imported in _strata_imports(ast.parse(text, filename=str(path)))
        if imported == cli
    ]

    assert importers == [], (
        f'{cli} is the command line, the top stratum; gates and helpers reach the '
        f'verdict through merge_lane_ratchet instead: {importers}'
    )


#: The layers on the import path that binds this checkout's workspace members:
#: ``source_measures`` imports ``_checkout``, and the measurement stratum
#: ``merge_lane_cluster`` imports ``source_measures``. Whether a workspace-member
#: import in either resolved this checkout's copy would rest on statement order,
#: not intent, so they import none.
BINDING_PATH_LAYERS = ('source_measures', 'merge_lane_cluster')


def _workspace_packages() -> frozenset[str]:
    return frozenset(
        package.name
        for src in _checkout.MEMBER_SRC
        for package in src.iterdir()
        if (package / '__init__.py').is_file()
    )


def _workspace_imports(source: str, packages: frozenset[str]) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        found.extend(
            (node.lineno, name) for name in names if name.partition('.')[0] in packages
        )
    return sorted(found)


def test_the_workspace_packages_are_found() -> None:
    assert {'shared', 'orchestrator'} <= _workspace_packages()


@pytest.mark.parametrize('stem', BINDING_PATH_LAYERS)
def test_no_layer_on_the_binding_import_path_imports_a_workspace_member(
    stem: str,
) -> None:
    found = _workspace_imports(_source_of(stem), _workspace_packages())

    assert found == [], (
        f'scripts/{stem}.py imports a workspace member, (line, module) {found}. '
        'It lies on the import path that binds the checkout (source_measures imports '
        '_checkout, merge_lane_cluster imports source_measures), so whether that import '
        "resolves this checkout's copy of the member would rest on statement order, "
        'not intent'
    )


def test_a_workspace_import_is_found_at_any_depth() -> None:
    source = 'import ast\ndef f():\n    from shared.safe_io import x\nimport orchestrator.git_ops\n'
    assert _workspace_imports(source, frozenset({'shared', 'orchestrator'})) == [
        (3, 'shared.safe_io'),
        (4, 'orchestrator.git_ops'),
    ]


#: ``(importer stem, source, whether the rule accepts it)``.  A new placement
#: the rule should accept or refuse is covered by adding a ROW.
PLACEMENT_CASES = [
    pytest.param(
        'merge_lane_metrics',
        'import merge_lane_cluster as lane_cluster\n',
        True,
        id='module-body-downward',
    ),
    pytest.param(
        'merge_lane_metrics',
        'import merge_lane_ratchet as ratchet\n',
        True,
        id='cli-imports-the-verdict',
    ),
    pytest.param(
        'merge_lane_cluster',
        'import merge_lane_metrics as metrics\n',
        False,
        id='module-body-upward',
    ),
    pytest.param(
        'merge_lane_ratchet',
        'import merge_lane_metrics as metrics\n',
        False,
        id='verdict-imports-the-cli',
    ),
    pytest.param(
        'merge_lane_metrics',
        'def main():\n'
        '    import merge_lane_cluster as lane_cluster\n',
        False,
        id='function-local',
    ),
    pytest.param(
        'merge_lane_cluster',
        'import merge_lane_unplaced\n',
        True,
        id='module-not-in-strata-ignored',
    ),
    pytest.param(
        'merge_lane_ratchet',
        'from merge_lane_metrics import main\n',
        False,
        id='from-import-upward',
    ),
    pytest.param(
        'merge_lane_metrics',
        'from merge_lane_ratchet import RAISE_REMEDY\n',
        True,
        id='from-import-downward',
    ),
    pytest.param(
        'merge_lane_metrics',
        'from typing import TYPE_CHECKING\n'
        'if TYPE_CHECKING:\n'
        '    import merge_lane_ratchet as ratchet\n',
        False,
        id='type-checking-guarded',
    ),
]


@pytest.mark.parametrize(('stem', 'source', 'accepted'), PLACEMENT_CASES)
def test_the_placement_rule_accepts_and_refuses_what_it_states(
    stem: str, source: str, accepted: bool
) -> None:
    assert (_placement_violations(stem, source) == []) is accepted


# ---------------------------------------------------------------------------
# No stratum re-exports a lower one: each name is reached through the module
# that defines it.

_INTERNABLE_ATOMS = (str, bytes, int, float, type(None))


def _is_defined_by(module: ModuleType, name: str, value: object) -> bool:
    if name.startswith('__') and name.endswith('__'):
        return False
    if isinstance(value, (ModuleType, *_INTERNABLE_ATOMS)):
        return False
    if isinstance(value, (tuple, frozenset)) and not value:
        return False
    return getattr(value, '__module__', module.__name__) == module.__name__


def _objects_defined_in(module: ModuleType) -> list[object]:
    return [
        value
        for name, value in vars(module).items()
        if _is_defined_by(module, name, value)
    ]


def _names_rebound_from(
    namespace: Mapping[str, object], provider: ModuleType
) -> list[str]:
    owned = _objects_defined_in(provider)
    return sorted(
        name
        for name, value in namespace.items()
        if any(value is own for own in owned)
    )


def _names_from_imported(source: str, provider: str) -> list[str]:
    return sorted(
        alias.asname or alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module == provider
        for alias in node.names
    )


def _providers_below(index: int) -> list[ModuleType]:
    return [
        source_measures,
        *(importlib.import_module(stem) for stem in STRATA[:index]),
    ]


@pytest.mark.parametrize(('index', 'stem'), list(enumerate(STRATA)))
def test_no_stratum_binds_an_object_a_lower_module_defines(
    index: int, stem: str
) -> None:
    module = importlib.import_module(stem)
    rebound = {
        provider.__name__: _names_rebound_from(vars(module), provider)
        for provider in _providers_below(index)
    }
    assert rebound == {provider: [] for provider in rebound}


@pytest.mark.parametrize(('index', 'stem'), list(enumerate(STRATA)))
def test_no_stratum_from_imports_from_a_lower_module(index: int, stem: str) -> None:
    # Identity cannot see an interned atom: `from source_measures import
    # COMPLEXIPY_REQUIRED` would rebind a str the check above skips.
    source = _source_of(stem)
    from_imported = {
        provider.__name__: _names_from_imported(source, provider.__name__)
        for provider in _providers_below(index)
    }
    assert from_imported == {provider: [] for provider in from_imported}


def test_a_from_import_is_found_at_any_depth_under_its_bound_name() -> None:
    source = (
        'import source_measures\n'
        'from source_measures import COMPLEXIPY_REQUIRED\n'
        'def f():\n'
        '    from source_measures import parse_source as parse\n'
    )
    assert _names_from_imported(source, 'source_measures') == [
        'COMPLEXIPY_REQUIRED',
        'parse',
    ]


def test_a_rebinding_under_any_name_is_found() -> None:
    namespace: dict[str, object] = {
        'ast': ast,
        'Path': Path,
        'Mapping': Mapping,
        'source_measures': source_measures,
        'alias': source_measures.file_size_measures,
        'Err': source_measures.MetricsError,
        'LIMIT': source_measures.COMPLEXIPY_MIN,
        'ROOTS': source_measures.PSEUDO_MEMBERS,
    }
    assert _names_rebound_from(namespace, source_measures) == [
        'Err',
        'LIMIT',
        'ROOTS',
        'alias',
    ]
