"""client_feed's import boundary: the portal vendors the contract, so its core stays stdlib + pydantic."""

import ast
import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

FORBIDDEN_ROOTS = frozenset({'orchestrator', 'fused_memory', 'dashboard'})
ALLOWED_SHARED_MODULES = ('shared.project_layout', 'shared.task_metadata')
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


@pytest.mark.parametrize('source', _package_files(), ids=lambda path: path.name)
def test_shared_imports_are_limited_to_layout_and_task_metadata(source: Path) -> None:
    offenders = [
        name
        for name in _imported_modules(source)
        if _root(name) == 'shared'
        and not any(_is_within(name, allowed) for allowed in ALLOWED_SHARED_MODULES)
    ]
    assert offenders == []


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
