"""Tests for fused_memory/utils/safe_yaml.py, the single home of the
registry-file read/parse policy shared by the three curator registry loaders
(cancelled-premise blocklist, operational-ask registry, recon code-fix premise
registry).
"""

from __future__ import annotations

import ast
import logging
import os
import sys
import types
from pathlib import Path

import pytest
import yaml
from _ast_guard import parse_python_module
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.utils.safe_yaml import (
    SAFE_YAML_LOADER,
    YamlListFileError,
    load_yaml_list_file,
    resolve_safe_yaml_loader,
)

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'config'

MIDDLEWARE_DIR = Path(__file__).resolve().parents[1] / 'src' / 'fused_memory' / 'middleware'

REGISTRY_LOADER_MODULES = (
    'cancelled_premise_blocklist.py',
    'operational_ask_registry.py',
    'recon_code_fix_premise_guard.py',
)


def _load(path: Path | None) -> list[object]:
    return load_yaml_list_file(path)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == logging.WARNING]


class TestLoadYamlListFile:
    def test_none_path_returns_empty_without_warning(self, caplog):
        with caplog.at_level('WARNING'):
            assert _load(None) == []

        assert _warnings(caplog) == []

    def test_valid_list_document_returns_parsed_list(self, tmp_path, caplog):
        p = tmp_path / 'registry.yaml'
        p.write_text(
            '- name: first\n  substrings: [a, b]\n- name: second\n  substrings: []\n',
            encoding='utf-8',
        )

        with caplog.at_level('WARNING'):
            data = _load(p)

        assert data == [
            {'name': 'first', 'substrings': ['a', 'b']},
            {'name': 'second', 'substrings': []},
        ]
        assert _warnings(caplog) == []

    def test_missing_file_raises_with_existing_reason(self, tmp_path):
        missing = tmp_path / 'does_not_exist.yaml'

        with pytest.raises(YamlListFileError) as excinfo:
            _load(missing)

        assert excinfo.value.path == missing
        assert str(excinfo.value) == f'file not found: {missing}'

    @pytest.mark.skipif(
        sys.platform == 'win32' or getattr(os, 'getuid', lambda: -1)() == 0,
        reason='chmod not reliable on Windows or when running as root',
    )
    def test_unreadable_file_raises(self, tmp_path):
        locked = tmp_path / 'unreadable.yaml'
        locked.write_text('- name: x\n', encoding='utf-8')
        locked.chmod(0o000)
        try:
            with pytest.raises(YamlListFileError) as excinfo:
                _load(locked)
        finally:
            locked.chmod(0o644)

        assert excinfo.value.path == locked
        assert str(locked) in str(excinfo.value)

    def test_undecodable_file_raises(self, tmp_path):
        bad_encoding = tmp_path / 'bad_encoding.yaml'
        bad_encoding.write_bytes(b'\xff\xfe- name: x\x00')

        with pytest.raises(YamlListFileError) as excinfo:
            _load(bad_encoding)

        assert excinfo.value.path == bad_encoding
        message = str(excinfo.value)
        assert str(bad_encoding) in message
        assert 'UTF-8' in message

    def test_malformed_yaml_raises(self, tmp_path):
        bad_yaml = tmp_path / 'bad.yaml'
        bad_yaml.write_text('key: [unclosed bracket\n: invalid\n', encoding='utf-8')

        with pytest.raises(YamlListFileError) as excinfo:
            _load(bad_yaml)

        assert excinfo.value.path == bad_yaml
        assert 'YAML parse error' in str(excinfo.value)

    @pytest.mark.parametrize(
        'document',
        ['just_a_key: just_a_value\n', ''],
        ids=['mapping', 'empty-file'],
    )
    def test_non_list_document_raises(self, tmp_path, document):
        p = tmp_path / 'not_a_list.yaml'
        p.write_text(document, encoding='utf-8')

        with pytest.raises(YamlListFileError) as excinfo:
            _load(p)

        assert excinfo.value.path == p
        assert 'expected a YAML list' in str(excinfo.value)


class TestSafeYamlLoader:
    def test_resolves_to_c_loader_when_available(self):
        if not hasattr(yaml, 'CSafeLoader'):
            pytest.skip('libyaml not available in this environment')

        assert SAFE_YAML_LOADER is yaml.CSafeLoader

    def test_falls_back_to_pure_python_safe_loader_without_libyaml(self):
        without_libyaml = types.SimpleNamespace(SafeLoader=yaml.SafeLoader)

        assert resolve_safe_yaml_loader(without_libyaml) is yaml.SafeLoader

    @pytest.mark.parametrize(
        'registry_name',
        [
            'cancelled_premise_blocklist.yaml',
            'operational_ask_registry.yaml',
            'recon_code_fix_premise_registry.yaml',
        ],
    )
    def test_shipped_registry_parses_identically_under_both_loaders(self, registry_name):
        if not hasattr(yaml, 'CSafeLoader'):
            pytest.skip('libyaml not available in this environment')

        path = CONFIG_DIR / registry_name
        assert path.exists(), f'shipped registry missing at {path}'

        text = path.read_text(encoding='utf-8')
        assert yaml.load(text, Loader=yaml.SafeLoader) == yaml.load(
            text, Loader=SAFE_YAML_LOADER
        )


def _yaml_imports(tree: ast.Module) -> list[ast.stmt]:
    def is_yaml(name: str | None) -> bool:
        return name is not None and (name == 'yaml' or name.startswith('yaml.'))

    return [
        node
        for node in ast.walk(tree)
        if (isinstance(node, ast.Import) and any(is_yaml(a.name) for a in node.names))
        or (isinstance(node, ast.ImportFrom) and is_yaml(node.module))
    ]


class TestRegistryLoadersParseOnlyThroughSafeYaml:
    """Per-loader pin that the loader choice and the read guard have ONE home
    (INV-5): a registry module that never imports yaml cannot read or parse its
    file outside load_yaml_list_file. AST, not grep, so prose mentioning yaml
    cannot trip it. The list is fixed, not discovered, because only these
    registry modules share the contract.
    """

    @pytest.mark.parametrize('module', REGISTRY_LOADER_MODULES)
    def test_registry_module_does_not_import_yaml(self, module):
        path = MIDDLEWARE_DIR / module

        offending = _yaml_imports(parse_python_module(path))

        assert not offending, (
            f'{module} imports yaml (line(s) {[n.lineno for n in offending]}); '
            'file-level YAML reading must go through '
            'fused_memory.utils.safe_yaml.load_yaml_list_file'
        )
