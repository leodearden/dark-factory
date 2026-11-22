"""Tests for fused_memory.config.falkordb_endpoints: no I/O, no live server."""

import dataclasses

import pytest

from fused_memory.config.falkordb_endpoints import (
    PRODUCTION_FALKORDB,
    SCRATCH_FALKORDB,
    FalkorEndpoint,
    ProductionFalkorDBTarget,
    is_production_falkordb,
    scratch_falkordb_from_env,
    with_falkordb,
)
from fused_memory.config.schema import (
    FalkorDBProviderConfig,
    FusedMemoryConfig,
    GraphitiBackendConfig,
)


class TestFalkorEndpoint:
    def test_is_frozen(self):
        endpoint = FalkorEndpoint('localhost', 6380)
        field_name = 'port'

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(endpoint, field_name, 1)

    def test_uri_names_host_and_port(self):
        assert FalkorEndpoint('h', 7000).uri == 'redis://h:7000'

    def test_from_uri_round_trips(self):
        endpoint = FalkorEndpoint.from_uri('redis://h:7000')

        assert endpoint == FalkorEndpoint('h', 7000)
        assert endpoint.uri == 'redis://h:7000'

    def test_from_uri_without_a_port_takes_the_redis_default(self):
        assert FalkorEndpoint.from_uri('redis://h') == FalkorEndpoint('h', 6379)

    def test_from_uri_without_a_host_takes_localhost(self):
        assert FalkorEndpoint.from_uri('redis://:7000') == FalkorEndpoint('localhost', 7000)


class TestKnownServers:
    def test_production_is_localhost_6379(self):
        assert FalkorEndpoint('localhost', 6379) == PRODUCTION_FALKORDB

    def test_scratch_is_localhost_6380(self):
        assert FalkorEndpoint('localhost', 6380) == SCRATCH_FALKORDB


class TestScratchFalkorDBFromEnv:
    def test_empty_environment_gives_the_scratch_server(self):
        assert scratch_falkordb_from_env({}) == SCRATCH_FALKORDB

    def test_falkor_host_overrides_the_host(self):
        endpoint = scratch_falkordb_from_env({'FALKOR_HOST': 'falkordb.internal'})

        assert endpoint == FalkorEndpoint('falkordb.internal', SCRATCH_FALKORDB.port)

    def test_falkor_port_overrides_the_port_as_an_int(self):
        endpoint = scratch_falkordb_from_env({'FALKOR_PORT': '7000'})

        assert endpoint.port == 7000
        assert isinstance(endpoint.port, int)

    def test_empty_falkor_port_counts_as_unset(self):
        """``FALKOR_PORT: ${FALKOR_PORT}`` in a CI template with the variable unset."""
        assert scratch_falkordb_from_env({'FALKOR_PORT': ''}) == SCRATCH_FALKORDB

    def test_non_integer_falkor_port_names_the_value(self):
        with pytest.raises(RuntimeError, match=r"FALKOR_PORT.*'notaport'"):
            scratch_falkordb_from_env({'FALKOR_PORT': 'notaport'})

    @pytest.mark.parametrize(
        'environ',
        [
            {'FALKOR_PORT': '6379'},
            {'FALKOR_HOST': '127.0.0.1', 'FALKOR_PORT': '6379'},
            {'FALKOR_HOST': '0.0.0.0', 'FALKOR_PORT': '6379'},
        ],
    )
    def test_a_production_aim_is_refused(self, environ):
        with pytest.raises(ProductionFalkorDBTarget) as raised:
            scratch_falkordb_from_env(environ)

        message = str(raised.value)
        assert 'FALKOR_HOST' in message
        assert 'FALKOR_PORT' in message
        assert repr(environ['FALKOR_PORT']) in message
        assert repr(environ.get('FALKOR_HOST', 'localhost')) in message
        assert PRODUCTION_FALKORDB.uri in message

    def test_the_refusal_is_a_runtime_error(self):
        assert issubclass(ProductionFalkorDBTarget, RuntimeError)


class TestIsProductionFalkorDB:
    @pytest.mark.parametrize(
        'host',
        [
            'localhost', '127.0.0.1', '::1', '',
            'LOCALHOST', 'localhost.', ' localhost ', 'localhost.localdomain', 'ip6-localhost',
            'falkordb.localhost', '[::1]', '127.0.0.2', '127.1', '::ffff:127.0.0.1',
            '0.0.0.0', '::', '0',
        ],
    )
    def test_this_host_on_6379_is_production_in_any_spelling(self, host):
        """A 0.0.0.0 copied from a docker bind address connects to production as surely as localhost."""
        assert is_production_falkordb(FalkorEndpoint(host, 6379))

    @pytest.mark.parametrize('host', ['10.0.0.5', '::ffff:10.0.0.5', 'localhost.example.com', 'notlocalhost'])
    def test_another_host_on_6379_is_not_production(self, host):
        assert not is_production_falkordb(FalkorEndpoint(host, 6379))

    def test_the_scratch_port_is_not_production(self):
        assert not is_production_falkordb(FalkorEndpoint('localhost', 6380))

    def test_a_remote_host_on_6379_is_not_production(self):
        assert not is_production_falkordb(FalkorEndpoint('falkordb.internal', 6379))


class TestWithFalkorDB:
    def _config(self) -> FusedMemoryConfig:
        falkordb = FalkorDBProviderConfig(
            uri='redis://corpus.invalid:7001',
            password='secret',
            database='legacy',
            expected_graph_timeout_ms=1234,
        )
        return FusedMemoryConfig.model_construct(graphiti=GraphitiBackendConfig(falkordb=falkordb))

    def test_moves_the_uri_and_drops_the_password(self):
        moved = with_falkordb(self._config(), FalkorEndpoint('127.0.0.1', 16380)).graphiti.falkordb

        assert moved.uri == 'redis://127.0.0.1:16380'
        assert moved.password is None

    def test_keeps_every_other_field(self):
        cfg = self._config()

        moved = with_falkordb(cfg, SCRATCH_FALKORDB).graphiti.falkordb

        untouched = {'uri', 'password'}
        assert moved.model_dump(exclude=untouched) == cfg.graphiti.falkordb.model_dump(exclude=untouched)
        assert moved.model_dump(exclude=untouched) != FalkorDBProviderConfig().model_dump(exclude=untouched)

    def test_leaves_the_given_config_alone(self):
        cfg = self._config()

        with_falkordb(cfg, SCRATCH_FALKORDB)

        assert cfg.graphiti.falkordb.uri == 'redis://corpus.invalid:7001'
        assert cfg.graphiti.falkordb.password == 'secret'
