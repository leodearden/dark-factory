"""Tests for config schema Literal type validation."""

import os
import sys

import pytest
import yaml
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

    def test_yaml_file_values_loaded(self, tmp_path, monkeypatch):
        # Write a YAML file with non-default values to exercise the full
        # YamlSettingsSource.__call__ + YAML-parsing branch (lines 61-63 of schema.py)
        config_data = {
            'server': {'port': 9999, 'transport': 'sse'},
            'llm': {'provider': 'anthropic'},
        }
        config_file = tmp_path / 'config.yaml'
        config_file.write_text(yaml.dump(config_data))
        monkeypatch.setenv('CONFIG_PATH', str(config_file))
        config = FusedMemoryConfig()
        assert config.server.port == 9999
        assert config.server.transport == 'sse'
        assert config.llm.provider == 'anthropic'
        # Verify unmentioned config sections retain their defaults (not clobbered to null/empty)
        assert config.embedder.provider == 'openai'
        assert config.embedder.model == 'text-embedding-3-small'
        assert config.server.host == '0.0.0.0'
        assert config.graphiti.provider == 'falkordb'
        assert config.mem0.qdrant_url == 'http://localhost:6333'
        assert config.routing.confidence_threshold == 0.7

    def test_env_var_expansion_e2e(self, tmp_path, monkeypatch):
        """End-to-end: env var placeholder in YAML is expanded through full settings machinery."""
        config_file = tmp_path / 'config.yaml'
        config_file.write_text("server:\n  port: '${MY_TEST_PORT}'\n")
        monkeypatch.setenv('CONFIG_PATH', str(config_file))
        monkeypatch.setenv('MY_TEST_PORT', '4242')
        config = FusedMemoryConfig()
        assert config.server.port == 4242

    def test_env_overrides_yaml_priority(self, tmp_path, monkeypatch):
        """Env vars (via pydantic-settings __ delimiter) take priority over YAML values."""
        config_data = {'server': {'port': 9999}}
        config_file = tmp_path / 'config.yaml'
        config_file.write_text(yaml.dump(config_data))
        monkeypatch.setenv('CONFIG_PATH', str(config_file))
        monkeypatch.setenv('SERVER__PORT', '7777')
        config = FusedMemoryConfig()
        assert config.server.port == 7777


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


class TestYamlSettingsSourceErrorHandling:
    """Tests for YamlSettingsSource error handling on corrupt or unreadable files."""

    def _make_source(self, path):
        from pydantic_settings import BaseSettings

        class _DummySettings(BaseSettings):
            pass

        return YamlSettingsSource(_DummySettings, config_path=path)

    def test_corrupt_yaml_raises_runtime_error(self, tmp_path):
        """Corrupt YAML content must raise RuntimeError with the file path in the message."""
        bad_file = tmp_path / 'bad.yaml'
        bad_file.write_bytes(b': :\n  - \x00bad')
        source = self._make_source(bad_file)
        with pytest.raises(RuntimeError, match=str(bad_file)) as exc_info:
            source()
        assert exc_info.value.__cause__ is not None

    @pytest.mark.skipif(sys.platform == 'win32', reason='chmod not reliable on Windows')
    def test_unreadable_file_raises_runtime_error(self, tmp_path):
        """An unreadable file must raise RuntimeError with the file path in the message."""
        locked_file = tmp_path / 'locked.yaml'
        locked_file.write_text('key: value')
        locked_file.chmod(0o000)
        try:
            source = self._make_source(locked_file)
            with pytest.raises(RuntimeError, match=str(locked_file)) as exc_info:
                source()
            assert exc_info.value.__cause__ is not None
        finally:
            locked_file.chmod(0o644)

    def test_expand_env_vars_error_raises_runtime_error(self, tmp_path, monkeypatch):
        """_expand_env_vars raising any exception must be wrapped in RuntimeError with config path."""
        config_file = tmp_path / 'valid.yaml'
        config_file.write_text('key: value')
        source = self._make_source(config_file)

        def _raise(val):
            raise ValueError('boom')

        monkeypatch.setattr(source, '_expand_env_vars', _raise)
        with pytest.raises(RuntimeError, match=str(config_file)):
            source()

    def test_expand_env_vars_error_includes_original_cause(self, tmp_path, monkeypatch):
        """The RuntimeError raised for _expand_env_vars failure must chain the original exception."""
        config_file = tmp_path / 'valid.yaml'
        config_file.write_text('key: value')
        source = self._make_source(config_file)
        original = ValueError('original cause')

        def _raise(val):
            raise original

        monkeypatch.setattr(source, '_expand_env_vars', _raise)
        with pytest.raises(RuntimeError) as exc_info:
            source()
        assert exc_info.value.__cause__ is original

    def test_expand_env_vars_error_does_not_mask_yaml_error(self, tmp_path):
        """Corrupt YAML must still raise RuntimeError with 'Failed to load configuration' message."""
        bad_file = tmp_path / 'bad.yaml'
        bad_file.write_bytes(b': :\n  - \x00bad')
        source = self._make_source(bad_file)
        with pytest.raises(RuntimeError, match='Failed to load configuration') as exc_info:
            source()
        assert 'Failed to expand' not in str(exc_info.value)


class TestYamlSettingsSourceEncoding:
    """Tests for YamlSettingsSource explicit UTF-8 encoding."""

    def _make_source(self, path):
        from pydantic_settings import BaseSettings

        class _DummySettings(BaseSettings):
            pass

        return YamlSettingsSource(_DummySettings, config_path=path)

    def test_utf8_yaml_loaded_correctly(self, tmp_path):
        """YAML files with non-ASCII UTF-8 characters must load correctly."""
        config_file = tmp_path / 'utf8.yaml'
        config_file.write_text("description: 'Ünfcödé tëst'", encoding='utf-8')
        source = self._make_source(config_file)
        result = source()
        assert result.get('description') == 'Ünfcödé tëst'


class TestYamlSettingsSourceABCContract:
    """Tests for YamlSettingsSource ABC contract compliance."""

    def setup_method(self):
        from pydantic_settings import BaseSettings

        class _DummySettings(BaseSettings):
            pass

        self.source = YamlSettingsSource(_DummySettings, config_path=None)

    def test_get_field_value_returns_tuple(self):
        """get_field_value must return tuple[Any, str, bool] per PydanticBaseSettingsSource ABC."""
        from pydantic.fields import FieldInfo

        field = FieldInfo(annotation=str)
        result = self.source.get_field_value(field, 'my_field')
        assert isinstance(result, tuple), f'Expected tuple, got {type(result)}'
        assert len(result) == 3, f'Expected 3-tuple, got {len(result)}-tuple'
        assert result[0] is None
        assert result[1] == 'my_field'
        assert result[2] is False
