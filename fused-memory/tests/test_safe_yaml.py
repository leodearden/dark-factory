"""Tests for fused_memory/utils/safe_yaml.py, the single home of the
registry-file read/parse policy shared by the three curator registry loaders
(cancelled-premise blocklist, operational-ask registry, recon code-fix premise
registry).
"""

from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path

import pytest
import yaml
from fused_memory.utils.safe_yaml import (
    SAFE_YAML_LOADER,
    load_yaml_list_file,
    resolve_safe_yaml_loader,
)

LOG = logging.getLogger('tests.safe_yaml_owner')

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'config'


def _load(path: Path | None) -> list[object]:
    return load_yaml_list_file(
        path, logger=LOG, label='test_registry', consequence='registry disabled'
    )


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

    def test_missing_file_returns_empty_and_warns_with_existing_message(
        self, tmp_path, caplog
    ):
        missing = tmp_path / 'does_not_exist.yaml'

        with caplog.at_level('WARNING'):
            assert _load(missing) == []

        warnings = _warnings(caplog)
        assert len(warnings) == 1
        assert (
            warnings[0].getMessage()
            == f'test_registry: file not found: {missing} — registry disabled'
        )

    @pytest.mark.skipif(
        sys.platform == 'win32' or getattr(os, 'getuid', lambda: -1)() == 0,
        reason='chmod not reliable on Windows or when running as root',
    )
    def test_unreadable_file_returns_empty_and_warns(self, tmp_path, caplog):
        locked = tmp_path / 'unreadable.yaml'
        locked.write_text('- name: x\n', encoding='utf-8')
        locked.chmod(0o000)
        try:
            with caplog.at_level('WARNING'):
                data = _load(locked)
        finally:
            locked.chmod(0o644)

        assert data == []
        warnings = _warnings(caplog)
        assert len(warnings) == 1
        assert str(locked) in warnings[0].getMessage()

    def test_undecodable_file_returns_empty_and_warns(self, tmp_path, caplog):
        bad_encoding = tmp_path / 'bad_encoding.yaml'
        bad_encoding.write_bytes(b'\xff\xfe- name: x\x00')

        with caplog.at_level('WARNING'):
            data = _load(bad_encoding)

        assert data == []
        warnings = _warnings(caplog)
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert str(bad_encoding) in message
        assert 'UTF-8' in message

    def test_malformed_yaml_returns_empty_and_warns(self, tmp_path, caplog):
        bad_yaml = tmp_path / 'bad.yaml'
        bad_yaml.write_text('key: [unclosed bracket\n: invalid\n', encoding='utf-8')

        with caplog.at_level('WARNING'):
            data = _load(bad_yaml)

        assert data == []
        warnings = _warnings(caplog)
        assert len(warnings) == 1
        assert 'YAML parse error' in warnings[0].getMessage()

    @pytest.mark.parametrize(
        'document',
        ['just_a_key: just_a_value\n', ''],
        ids=['mapping', 'empty-file'],
    )
    def test_non_list_document_returns_empty_and_warns(self, tmp_path, caplog, document):
        p = tmp_path / 'not_a_list.yaml'
        p.write_text(document, encoding='utf-8')

        with caplog.at_level('WARNING'):
            data = _load(p)

        assert data == []
        warnings = _warnings(caplog)
        assert len(warnings) == 1
        assert 'expected a YAML list' in warnings[0].getMessage()

    def test_warning_is_emitted_on_the_caller_supplied_logger(self, tmp_path, caplog):
        with caplog.at_level('WARNING'):
            _load(tmp_path / 'does_not_exist.yaml')

        warnings = _warnings(caplog)
        assert len(warnings) == 1
        assert warnings[0].name == 'tests.safe_yaml_owner'


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
