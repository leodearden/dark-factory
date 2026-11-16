"""fused-memory's redis client must parse FalkorDB replies with hiredis's C parser (task 9308).

Without it, redis-py falls back to its pure-Python RESP parser, which blocked the server's
single asyncio loop for seconds per large Graphiti search reply. The dependency is declared
as `redis[hiredis]` in fused-memory/pyproject.toml.
"""

from importlib.metadata import PackageNotFoundError, version

from redis.utils import HIREDIS_AVAILABLE


def _installed_hiredis_version() -> str:
    try:
        return version('hiredis')
    except PackageNotFoundError:
        return 'not installed'


def test_redis_parses_replies_with_the_hiredis_c_parser():
    assert HIREDIS_AVAILABLE, (
        'redis-py must select the hiredis C reply parser (redis.utils.HIREDIS_AVAILABLE is '
        f'False): redis {version("redis")}, hiredis {_installed_hiredis_version()}. redis only '
        'accepts hiredis >= 3.2, so an older hiredis counts as unavailable. Remedy: declare '
        '`redis[hiredis]` in fused-memory/pyproject.toml, run `uv lock`, then '
        '`uv sync --all-packages` from the checkout root (a member-scoped sync is exact and '
        'prunes the other members).'
    )
