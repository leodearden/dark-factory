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
    @pytest.mark.parametrize('host', ['localhost', '127.0.0.1', '::1', ''])
    def test_loopback_6379_is_production(self, host):
        assert is_production_falkordb(FalkorEndpoint(host, 6379))

    def test_the_scratch_port_is_not_production(self):
        assert not is_production_falkordb(FalkorEndpoint('localhost', 6380))

    def test_a_remote_host_on_6379_is_not_production(self):
        assert not is_production_falkordb(FalkorEndpoint('falkordb.internal', 6379))
