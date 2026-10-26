"""The shared by-path loader that fused-memory/scripts use to load their siblings."""

from __future__ import annotations

import sys
import types
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from fused_memory.utils.script_loader import load_script


@pytest.fixture
def mod_name() -> Iterator[str]:
    name = f'_script_loader_probe_{uuid.uuid4().hex}'
    yield name
    sys.modules.pop(name, None)


@pytest.fixture
def write_script(tmp_path: Path) -> Callable[[str, str], Path]:
    def write(body: str, filename: str = 'probe.py') -> Path:
        path = tmp_path / filename
        path.write_text(body)
        return path

    return write


def test_loading_a_file_returns_its_module_registered_under_the_name(write_script, mod_name):
    path = write_script('VALUE = 42\n')

    module = load_script(path, mod_name)

    assert module.VALUE == 42
    assert sys.modules[mod_name] is module
    assert module.__file__ is not None
    assert Path(module.__file__).resolve() == path.resolve()


def test_the_module_is_registered_before_its_body_runs(write_script, mod_name):
    path = write_script('import sys\nSEEN = sys.modules.get(__name__)\n')

    module = load_script(path, mod_name)

    assert module.SEEN is module


def test_a_second_load_returns_the_same_module(write_script, mod_name):
    path = write_script('VALUE = 1\n')

    first = load_script(path, mod_name)
    second = load_script(path, mod_name)

    assert first is second


def test_reuse_is_keyed_by_name_not_by_file(write_script, mod_name, monkeypatch):
    sentinel = types.ModuleType(mod_name)
    monkeypatch.setitem(sys.modules, mod_name, sentinel)
    path = write_script("raise AssertionError('must not execute')\n")

    assert load_script(path, mod_name) is sentinel


def test_a_failed_body_propagates_and_leaves_no_entry(write_script, mod_name, monkeypatch):
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    path = write_script("raise ValueError('boom')\n")

    with pytest.raises(ValueError, match='boom'):
        load_script(path, mod_name)
    assert mod_name not in sys.modules

    write_script('VALUE = 7\n')
    assert load_script(path, mod_name).VALUE == 7


def test_a_path_no_loader_claims_raises_import_error(write_script, mod_name):
    path = write_script('VALUE = 1\n', filename='not_a_module.txt')

    with pytest.raises(ImportError) as excinfo:
        load_script(path, mod_name)

    assert str(path) in str(excinfo.value)
    assert mod_name not in sys.modules
