"""Retired config keys still load, are dropped, and are named in one WARNING."""

import logging
from pathlib import Path

import pydantic
import pytest
import yaml

from fused_memory.config.schema import (
    FusedMemoryConfig,
    MemoryMetadataConfig,
    OpenAIProviderConfig,
    TaskmasterConfig,
)

_RETIRED_TASKMASTER_KEYS = ('transport', 'command', 'args', 'cwd', 'http_url', 'tool_mode')


def _retired_warnings(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and 'retired' in record.getMessage()
    ]


class TestTaskmasterConfigRetiredKeys:
    def test_retired_keys_are_dropped_and_named_in_one_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = TaskmasterConfig.model_validate({
                'project_root': '/srv/proj',
                'transport': 'stdio',
                'command': 'node',
                'args': ['x'],
                'cwd': '/tmp',
                'http_url': 'http://h',
                'tool_mode': 'all',
            })

        assert cfg.model_dump() == {'project_root': '/srv/proj'}
        warnings = [message for message in _retired_warnings(caplog) if 'taskmaster' in message]
        assert len(warnings) == 1, warnings
        for key in _RETIRED_TASKMASTER_KEYS:
            assert key in warnings[0], (key, warnings[0])

    def test_project_root_alone_logs_no_retired_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            TaskmasterConfig.model_validate({'project_root': '/srv/proj'})

        assert _retired_warnings(caplog) == []

    def test_yaml_section_with_retired_key_still_loads(self, tmp_path, monkeypatch, caplog):
        config_file = tmp_path / 'config.yaml'
        config_file.write_text(yaml.safe_dump({
            'taskmaster': {'project_root': str(tmp_path), 'tool_mode': 'all'},
        }))
        monkeypatch.setenv('CONFIG_PATH', str(config_file))

        with caplog.at_level(logging.WARNING):
            config = FusedMemoryConfig()

        assert config.taskmaster is not None
        assert config.taskmaster.project_root == str(tmp_path)
        assert any('tool_mode' in message for message in _retired_warnings(caplog))


class TestOpenAIProviderConfigRetiredKeys:
    def test_organization_id_is_dropped_and_named(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = OpenAIProviderConfig.model_validate({'api_key': 'k', 'organization_id': 'org-1'})

        assert 'organization_id' not in cfg.model_dump()
        assert cfg.api_key == 'k'
        warnings = [message for message in _retired_warnings(caplog) if 'organization_id' in message]
        assert len(warnings) == 1, warnings

    def test_warning_names_both_sections_that_hold_the_model(self, caplog):
        with caplog.at_level(logging.WARNING):
            OpenAIProviderConfig.model_validate({'organization_id': 'org-1'})

        [warning] = _retired_warnings(caplog)
        assert 'llm.providers.openai' in warning, warning
        assert 'embedder.providers.openai' in warning, warning

    def test_embedder_provider_with_organization_id_still_loads(self, tmp_path, monkeypatch, caplog):
        config_file = tmp_path / 'config.yaml'
        config_file.write_text(yaml.safe_dump({
            'embedder': {'providers': {'openai': {'api_key': 'k', 'organization_id': 'org-1'}}},
        }))
        monkeypatch.setenv('CONFIG_PATH', str(config_file))

        with caplog.at_level(logging.WARNING):
            config = FusedMemoryConfig()

        assert config.embedder.providers.openai is not None
        assert config.embedder.providers.openai.api_key == 'k'
        assert any('organization_id' in message for message in _retired_warnings(caplog))


class TestMemoryMetadataConfigRetiredKeys:
    def test_enforce_kind_registry_is_dropped_and_named(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = MemoryMetadataConfig.model_validate({'enforce': False, 'enforce_kind_registry': True})

        assert 'enforce_kind_registry' not in cfg.model_dump()
        assert cfg.enforce is False
        warnings = [
            message for message in _retired_warnings(caplog)
            if 'memory_metadata' in message and 'enforce_kind_registry' in message
        ]
        assert len(warnings) == 1, warnings

    def test_yaml_section_with_the_retired_key_still_loads(self, tmp_path, monkeypatch, caplog):
        config_file = tmp_path / 'config.yaml'
        config_file.write_text(yaml.safe_dump({'memory_metadata': {'enforce_kind_registry': False}}))
        monkeypatch.setenv('CONFIG_PATH', str(config_file))

        with caplog.at_level(logging.WARNING):
            FusedMemoryConfig()

        assert any('enforce_kind_registry' in message for message in _retired_warnings(caplog))

    def test_a_misspelt_retired_key_is_still_forbidden(self):
        with pytest.raises(pydantic.ValidationError):
            MemoryMetadataConfig.model_validate({'enforce_kind_regsitry': True})


@pytest.mark.parametrize('model', [TaskmasterConfig, OpenAIProviderConfig, MemoryMetadataConfig])
def test_no_live_field_is_dropped_as_retired(model, caplog):
    every_live_field = model().model_dump()

    with caplog.at_level(logging.WARNING):
        reloaded = model.model_validate(every_live_field)

    assert _retired_warnings(caplog) == []
    assert reloaded.model_dump() == every_live_field


def test_shipped_config_yaml_carries_no_retired_key(monkeypatch, caplog):
    yaml_path = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
    assert yaml_path.is_file(), f'expected config.yaml at {yaml_path}'
    monkeypatch.setenv('CONFIG_PATH', str(yaml_path))

    with caplog.at_level(logging.WARNING):
        FusedMemoryConfig()

    assert _retired_warnings(caplog) == []
