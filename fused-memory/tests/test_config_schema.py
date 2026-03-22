"""Tests for config schema Literal type validation."""

import os

import pytest
from pydantic import ValidationError

from fused_memory.config.schema import (
    EmbedderConfig,
    FusedMemoryConfig,
    GraphitiBackendConfig,
    LLMConfig,
    ServerConfig,
    YamlSettingsSource,
)


class TestServerConfigTransport:
    """Tests for ServerConfig.transport Literal validation."""

    def test_default_transport_is_http(self):
        config = ServerConfig()
        assert config.transport == 'http'

    def test_valid_transport_http(self):
        config = ServerConfig(transport='http')
        assert config.transport == 'http'

    def test_valid_transport_stdio(self):
        config = ServerConfig(transport='stdio')
        assert config.transport == 'stdio'

    def test_valid_transport_sse(self):
        config = ServerConfig(transport='sse')
        assert config.transport == 'sse'

    def test_invalid_transport_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ServerConfig(transport='websocket')  # type: ignore[arg-type]

    def test_invalid_transport_grpc_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ServerConfig(transport='grpc')  # type: ignore[arg-type]


class TestLLMConfigProvider:
    """Tests for LLMConfig.provider Literal validation."""

    def test_default_provider_is_openai(self):
        config = LLMConfig()
        assert config.provider == 'openai'

    def test_valid_provider_openai(self):
        config = LLMConfig(provider='openai')
        assert config.provider == 'openai'

    def test_valid_provider_anthropic(self):
        config = LLMConfig(provider='anthropic')
        assert config.provider == 'anthropic'

    def test_invalid_provider_raises_validation_error(self):
        with pytest.raises(ValidationError):
            LLMConfig(provider='gemini')  # type: ignore[arg-type]

    def test_invalid_provider_cohere_raises_validation_error(self):
        with pytest.raises(ValidationError):
            LLMConfig(provider='cohere')  # type: ignore[arg-type]


class TestEmbedderConfigProvider:
    """Tests for EmbedderConfig.provider Literal validation."""

    def test_default_provider_is_openai(self):
        config = EmbedderConfig()
        assert config.provider == 'openai'

    def test_valid_provider_openai(self):
        config = EmbedderConfig(provider='openai')
        assert config.provider == 'openai'

    def test_invalid_provider_raises_validation_error(self):
        with pytest.raises(ValidationError):
            EmbedderConfig(provider='cohere')  # type: ignore[arg-type]

    def test_invalid_provider_huggingface_raises_validation_error(self):
        with pytest.raises(ValidationError):
            EmbedderConfig(provider='huggingface')  # type: ignore[arg-type]


class TestGraphitiBackendConfigProvider:
    """Tests for GraphitiBackendConfig.provider Literal validation."""

    def test_default_provider_is_falkordb(self):
        config = GraphitiBackendConfig()
        assert config.provider == 'falkordb'

    def test_valid_provider_falkordb(self):
        config = GraphitiBackendConfig(provider='falkordb')
        assert config.provider == 'falkordb'

    def test_invalid_provider_raises_validation_error(self):
        with pytest.raises(ValidationError):
            GraphitiBackendConfig(provider='neo4j')  # type: ignore[arg-type]

    def test_invalid_provider_redis_raises_validation_error(self):
        with pytest.raises(ValidationError):
            GraphitiBackendConfig(provider='redis')  # type: ignore[arg-type]


class TestFusedMemoryConfigDefaults:
    """Tests for FusedMemoryConfig top-level defaults."""

    def test_all_defaults_load_successfully(self, tmp_path, monkeypatch):
        # Point CONFIG_PATH at a non-existent file so YamlSettingsSource returns {}
        monkeypatch.setenv('CONFIG_PATH', str(tmp_path / 'missing.yaml'))
        config = FusedMemoryConfig()
        assert config.server.transport == 'http'
        assert config.llm.provider == 'openai'
        assert config.embedder.provider == 'openai'
        assert config.graphiti.provider == 'falkordb'

    def test_valid_config_constructed_explicitly(self):
        config = FusedMemoryConfig(
            server=ServerConfig(transport='stdio'),
            llm=LLMConfig(provider='anthropic'),
            embedder=EmbedderConfig(provider='openai'),
            graphiti=GraphitiBackendConfig(provider='falkordb'),
        )
        assert config.server.transport == 'stdio'
        assert config.llm.provider == 'anthropic'


class TestYamlSettingsSourceEnvVarExpansion:
    """Tests for YamlSettingsSource._expand_env_vars."""

    def setup_method(self):
        # Use a dummy settings class; path doesn't matter for _expand_env_vars
        from pydantic_settings import BaseSettings
        class _DummySettings(BaseSettings):
            pass
        self.source = YamlSettingsSource(_DummySettings, config_path=None)

    def test_expands_env_var_with_value(self, monkeypatch):
        monkeypatch.setenv('MY_API_KEY', 'secret-key')
        result = self.source._expand_env_vars('${MY_API_KEY}')
        assert result == 'secret-key'

    def test_expands_env_var_with_default_when_missing(self):
        # Ensure the var is not set
        os.environ.pop('MISSING_VAR_XYZ', None)
        result = self.source._expand_env_vars('${MISSING_VAR_XYZ:default_val}')
        assert result == 'default_val'

    def test_expands_env_var_to_none_when_empty_default(self):
        os.environ.pop('MISSING_VAR_XYZ', None)
        result = self.source._expand_env_vars('${MISSING_VAR_XYZ}')
        assert result is None

    def test_expands_env_var_true_to_bool(self, monkeypatch):
        monkeypatch.setenv('FLAG_VAR', 'true')
        result = self.source._expand_env_vars('${FLAG_VAR}')
        assert result is True

    def test_expands_env_var_false_to_bool(self, monkeypatch):
        monkeypatch.setenv('FLAG_VAR', 'false')
        result = self.source._expand_env_vars('${FLAG_VAR}')
        assert result is False

    def test_expands_dict_values_recursively(self, monkeypatch):
        monkeypatch.setenv('HOST_VAR', 'localhost')
        data = {'host': '${HOST_VAR}', 'port': 8080}
        result = self.source._expand_env_vars(data)
        assert result['host'] == 'localhost'
        assert result['port'] == 8080

    def test_expands_list_values(self, monkeypatch):
        monkeypatch.setenv('ITEM_VAR', 'hello')
        result = self.source._expand_env_vars(['${ITEM_VAR}', 'static'])
        assert result[0] == 'hello'
        assert result[1] == 'static'

    def test_non_env_string_unchanged(self):
        result = self.source._expand_env_vars('plain-string')
        assert result == 'plain-string'
