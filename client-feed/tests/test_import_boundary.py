"""client_feed's import boundary: the portal vendors the contract, so its core stays stdlib + pydantic."""

import ast
import importlib.util
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

FORBIDDEN_ROOTS = frozenset({'orchestrator', 'fused_memory', 'dashboard', 'escalation'})
ALLOWED_SHARED_MODULES = ('shared.defects', 'shared.project_layout', 'shared.task_metadata')
FILER_TOOLS_ONLY_MODULES = ('shared.mcp_session', 'httpx')
FILER_LAYERS = ('filer_tools', 'filer_tasks', 'filer_state', 'filer')
FILER_CLIENT_FEED_IMPORTS = frozenset({'contract', 'defect_ref', 'policy', 'inbox', *FILER_LAYERS})
VENDORABLE_MODULES = frozenset({'__init__', 'contract', 'fold', 'validate', 'cli'})
VENDORABLE_ROOTS = frozenset({'pydantic', 'client_feed'}) | frozenset(sys.stdlib_module_names)


def _package_dir() -> Path:
    spec = importlib.util.find_spec('client_feed')
    assert spec is not None and spec.origin is not None
    return Path(spec.origin).parent


def _package_files() -> list[Path]:
    return sorted(_package_dir().rglob('*.py'))


def _imported_modules(source: Path) -> Iterator[str]:
    """Every dotted name a file imports; `from x import y` yields x.y so `from shared import m` is seen."""
    for node in ast.walk(ast.parse(source.read_text(), filename=str(source))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = 'client_feed' if node.level else node.module or ''
            yield from (f'{module}.{alias.name}' for alias in node.names)


def _is_within(name: str, module: str) -> bool:
    return name == module or name.startswith(f'{module}.')


def _root(name: str) -> str:
    return name.split('.', 1)[0]


def test_the_package_is_found_with_its_init() -> None:
    assert (_package_dir() / '__init__.py') in _package_files()


@pytest.mark.parametrize('source', _package_files(), ids=lambda path: path.name)
def test_no_module_imports_the_factory_services(source: Path) -> None:
    offenders = [name for name in _imported_modules(source) if _root(name) in FORBIDDEN_ROOTS]
    assert offenders == []


def _allowed_shared_modules(source: Path) -> tuple[str, ...]:
    """Every module may import the allowed shared modules; only filer_tools, the filer's wire, also the MCP client."""
    if source == _package_dir() / 'filer_tools.py':
        return (*ALLOWED_SHARED_MODULES, *FILER_TOOLS_ONLY_MODULES)
    return ALLOWED_SHARED_MODULES


@pytest.mark.parametrize('source', _package_files(), ids=lambda path: path.name)
def test_shared_imports_are_limited_to_the_allowed_modules(source: Path) -> None:
    allowed = _allowed_shared_modules(source)
    offenders = [
        name
        for name in _imported_modules(source)
        if _root(name) == 'shared' and not any(_is_within(name, module) for module in allowed)
    ]
    assert offenders == []


@pytest.mark.parametrize('source', _package_files(), ids=lambda path: path.name)
def test_only_filer_tools_imports_the_mcp_client_and_httpx(source: Path) -> None:
    importers = [
        name
        for name in _imported_modules(source)
        if any(_is_within(name, module) for module in FILER_TOOLS_ONLY_MODULES)
    ]
    assert importers == [] or source == _package_dir() / 'filer_tools.py'


def _client_feed_module(name: str) -> str | None:
    """The top-level client_feed module a dotted import name reaches, if any."""
    parts = name.split('.')
    return parts[1] if parts[0] == 'client_feed' and len(parts) > 1 else None


def _filer_files() -> list[Path]:
    return [path for path in _package_files() if path.parent == _package_dir() and path.stem in FILER_LAYERS]


@pytest.mark.parametrize('source', _filer_files(), ids=lambda path: path.name)
def test_the_filer_imports_only_its_own_layers_below_it_and_the_shared_lower_modules(source: Path) -> None:
    below = FILER_LAYERS[: FILER_LAYERS.index(source.stem)]
    reached = {module for module in map(_client_feed_module, _imported_modules(source)) if module}
    offenders = {
        module
        for module in reached
        if module not in FILER_CLIENT_FEED_IMPORTS or (module in FILER_LAYERS and module not in below)
    }
    assert offenders == set()


@pytest.mark.parametrize('source', _package_files(), ids=lambda path: path.name)
def test_no_publisher_module_imports_the_filer(source: Path) -> None:
    if source.parent == _package_dir() and source.stem in {'cli', *FILER_LAYERS}:
        return
    reached = {module for module in map(_client_feed_module, _imported_modules(source)) if module}
    assert reached & set(FILER_LAYERS) == set()


def _vendorable_files() -> list[Path]:
    return [
        path
        for path in _package_files()
        if path.parent == _package_dir() and path.stem in VENDORABLE_MODULES
    ]


@pytest.mark.parametrize('source', _vendorable_files(), ids=lambda path: path.name)
def test_vendorable_modules_import_only_stdlib_pydantic_and_client_feed(source: Path) -> None:
    offenders = [
        name for name in _imported_modules(source) if _root(name) not in VENDORABLE_ROOTS
    ]
    assert offenders == []


_CONTRACT_LOAD_PROBE = """
import sys
import client_feed.contract
loaded = sorted(
    name for name in sys.modules
    if name.split('.')[0] == 'shared' or name == 'client_feed.defect_ref'
)
print('\\n'.join(loaded), end='')
"""


def test_the_vendorable_contract_loads_no_shared_module() -> None:
    probe = subprocess.run(
        [sys.executable, '-c', _CONTRACT_LOAD_PROBE], check=False, capture_output=True, text=True
    )

    assert probe.returncode == 0, probe.stderr
    assert probe.stdout == ''
