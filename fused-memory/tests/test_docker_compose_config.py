"""Contract tests for fused-memory/docker/docker-compose.yml.

Every service publishes its ports on host loopback only.

The falkordb image entrypoint (/var/lib/falkordb/bin/run.sh) runs
``redis-server ${REDIS_ARGS} --dir /var/lib/falkordb/data ...`` with no config
file, so the compose file is the only durable home for redis options.

``falkordb`` is production; ``falkordb-scratch`` is the server tests and evals
write to (task 7792). Its addresses come from
``fused_memory.config.falkordb_endpoints``, never restated here.
"""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from _loopback_host import is_loopback_host

from fused_memory.config.falkordb_endpoints import PRODUCTION_FALKORDB, SCRATCH_FALKORDB

FUSED_MEMORY_DIR = Path(__file__).resolve().parent.parent
COMPOSE_FILE = FUSED_MEMORY_DIR / "docker" / "docker-compose.yml"
CONFIG_FILE = FUSED_MEMORY_DIR / "config" / "config.yaml"
FALKORDB_DATA_DIR = "/var/lib/falkordb/data"
CONTAINER_REDIS_PORT = 6379
EXPECTED_FALKORDB_MODULE_ARGS = {
    "MAX_QUEUED_QUERIES": "25",
    "TIMEOUT": "30000",
    "RESULTSET_SIZE": "10000",
}


def _compose_services() -> dict[str, dict[str, Any]]:
    return yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))["services"]


def _falkordb_service() -> dict[str, Any]:
    return _compose_services()["falkordb"]


def _falkordb_scratch_service() -> dict[str, Any]:
    services = _compose_services()
    assert "falkordb-scratch" in services, (
        "fused-memory/docker/docker-compose.yml declares no falkordb-scratch service, so "
        "integration tests and evals have no FalkorDB of their own and run against "
        f"production (task 7792). services = {sorted(services)!r}"
    )
    return services["falkordb-scratch"]


def _is_loopback_publish(entry: object) -> bool:
    if isinstance(entry, dict):
        return is_loopback_host(str(entry.get("host_ip") or ""))
    if not isinstance(entry, str):
        return False
    host_and_published = entry.rpartition(":")[0]
    return is_loopback_host(host_and_published.rpartition(":")[0])


def _service_env(service: dict[str, Any]) -> dict[str, str]:
    environment = service.get("environment") or {}
    if isinstance(environment, dict):
        return {str(key): "" if value is None else str(value) for key, value in environment.items()}
    pairs = (str(entry).partition("=") for entry in environment)
    return {key: value for key, _, value in pairs}


def _redis_options(redis_args: str) -> dict[str, str]:
    """Each ``--option`` mapped to its values up to the next ``--``, space-joined."""
    options: dict[str, list[str]] = {}
    values: list[str] | None = None
    for token in redis_args.split():
        if token.startswith("--"):
            name = token.removeprefix("--")
            assert name not in options, f"REDIS_ARGS repeats --{name}; got {redis_args!r}"
            values = options[name] = []
        else:
            assert values is not None, f"REDIS_ARGS must start with an --option; got {redis_args!r}"
            values.append(token)
    assert all(options.values()), f"every REDIS_ARGS option needs a value; got {redis_args!r}"
    return {name: " ".join(option_values) for name, option_values in options.items()}


def _falkordb_module_args(falkordb_args: str) -> dict[str, str]:
    tokens = falkordb_args.split()
    assert len(tokens) % 2 == 0, f"FALKORDB_ARGS must be 'NAME value' pairs; got {falkordb_args!r}"
    return dict(zip(tokens[0::2], tokens[1::2], strict=True))


def _mounted_container_paths(service: dict[str, Any]) -> set[str]:
    targets: set[str] = set()
    for volume in service.get("volumes") or []:
        if isinstance(volume, dict):
            targets.add(str(volume.get("target")))
            continue
        parts = str(volume).split(":")
        if len(parts) >= 2:
            targets.add(parts[1])
    return targets


def _mount_sources(service: dict[str, Any]) -> dict[str, str]:
    """Each mounted container path mapped to its source: a host path or a volume name."""
    sources: dict[str, str] = {}
    for volume in service.get("volumes") or []:
        if isinstance(volume, dict):
            sources[str(volume.get("target"))] = str(volume.get("source"))
            continue
        parts = str(volume).split(":")
        if len(parts) >= 2:
            sources[parts[1]] = parts[0]
    return sources


def test_falkordb_redis_args_enable_aof_with_everysec_fsync() -> None:
    env = _service_env(_falkordb_service())
    assert "REDIS_ARGS" in env, (
        "falkordb in fused-memory/docker/docker-compose.yml sets no REDIS_ARGS. Its "
        "entrypoint /var/lib/falkordb/bin/run.sh runs redis-server with no config "
        "file, so a runtime CONFIG SET appendonly is lost on every restart, including "
        "the restart: unless-stopped crash restart; REDIS_ARGS is the only durable "
        f"channel. environment = {env!r}"
    )
    redis_args = env["REDIS_ARGS"]
    options = _redis_options(redis_args)
    assert options.get("appendonly") == "yes", (
        f"falkordb must run with AOF on (appendonly yes); "
        f"parsed options = {options!r}, REDIS_ARGS = {redis_args!r}"
    )
    assert options.get("appendfsync") == "everysec", (
        f"falkordb AOF must fsync everysec (at most ~1s of writes lost); "
        f"parsed options = {options!r}, REDIS_ARGS = {redis_args!r}"
    )


def test_falkordb_data_dir_is_host_mounted() -> None:
    service = _falkordb_service()
    assert FALKORDB_DATA_DIR in _mounted_container_paths(service), (
        f"run.sh passes --dir {FALKORDB_DATA_DIR}, so the AOF and RDB land there; "
        "without a mount of that path they die with the container. "
        f"volumes = {service.get('volumes')!r}"
    )


def test_falkordb_args_pin_the_legacy_read_timeout_and_keep_the_image_defaults() -> None:
    env = _service_env(_falkordb_service())
    assert "FALKORDB_ARGS" in env, (
        "falkordb sets no FALKORDB_ARGS, so it runs with the image default TIMEOUT 1000, "
        "and a runtime GRAPH.CONFIG SET TIMEOUT is lost on every restart (task 6848). "
        f"environment = {env!r}"
    )
    module_args = _falkordb_module_args(env["FALKORDB_ARGS"])
    assert module_args == EXPECTED_FALKORDB_MODULE_ARGS, (
        "FALKORDB_ARGS replaces the image default wholesale, so it must restate "
        "MAX_QUEUED_QUERIES 25 and RESULTSET_SIZE 10000 and raise only TIMEOUT. It must "
        "not use TIMEOUT_DEFAULT/TIMEOUT_MAX: those also time out writes and roll them "
        f"back. parsed = {module_args!r}"
    )


def test_falkordb_image_is_pinned_by_digest() -> None:
    image = str(_falkordb_service().get("image"))
    assert re.fullmatch(r"falkordb/falkordb@sha256:[0-9a-f]{64}", image), (
        "a recreate must not silently upgrade FalkorDB, so the image is pinned by repo "
        f"digest, not a tag; image = {image!r}"
    )


def test_compose_timeout_is_the_timeout_fused_memory_expects() -> None:
    module_args = _falkordb_module_args(_service_env(_falkordb_service())["FALKORDB_ARGS"])
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    expected = config["graphiti"]["falkordb"]["expected_graph_timeout_ms"]
    assert int(module_args["TIMEOUT"]) == int(expected), (
        "the compose TIMEOUT and config.yaml graphiti.falkordb.expected_graph_timeout_ms "
        "must agree, or fused-memory warns about a server that is configured as intended; "
        f"compose TIMEOUT = {module_args['TIMEOUT']!r}, expected = {expected!r}"
    )


def test_every_compose_publish_binds_host_loopback() -> None:
    offenders = [
        (name, entry)
        for name, service in _compose_services().items()
        for entry in service.get("ports") or []
        if not _is_loopback_publish(entry)
    ]
    assert offenders == [], (
        "every published port must name a loopback host IP, e.g. \"127.0.0.1:6379:6379\": "
        "a publish without one binds every host interface, IPv4 and IPv6. "
        + "; ".join(f"{name}: {entry!r}" for name, entry in offenders)
    )


def test_scratch_runs_the_production_image() -> None:
    assert _falkordb_scratch_service().get("image") == _falkordb_service().get("image"), (
        "falkordb-scratch must run the one pinned engine production runs, so a test "
        "exercises the same FalkorDB build it guards"
    )


def test_scratch_falkordb_args_equal_production_falkordb_args() -> None:
    scratch_args = _falkordb_module_args(_service_env(_falkordb_scratch_service())["FALKORDB_ARGS"])
    production_args = _falkordb_module_args(_service_env(_falkordb_service())["FALKORDB_ARGS"])
    assert scratch_args == production_args, (
        "tests must meet production's read TIMEOUT and RESULTSET_SIZE; "
        f"scratch = {scratch_args!r}, production = {production_args!r}"
    )


def test_scratch_and_production_publish_their_own_ports() -> None:
    assert _falkordb_scratch_service().get("ports") == [
        f"127.0.0.1:{SCRATCH_FALKORDB.port}:{CONTAINER_REDIS_PORT}"
    ]
    assert _falkordb_service().get("ports") == [
        f"127.0.0.1:{PRODUCTION_FALKORDB.port}:{CONTAINER_REDIS_PORT}"
    ]


def test_scratch_data_dir_is_a_named_volume_shared_with_nothing() -> None:
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    scratch_source = _mount_sources(_falkordb_scratch_service()).get(FALKORDB_DATA_DIR)
    assert scratch_source in (compose.get("volumes") or {}), (
        "falkordb-scratch's data dir must be a named volume declared under top-level "
        "volumes:, so every checkout's compose file starts the container on the same "
        f"data; source = {scratch_source!r}, volumes = {compose.get('volumes')!r}"
    )
    assert scratch_source not in set(_mount_sources(_falkordb_service()).values())


def test_scratch_persists_by_one_rare_snapshot_without_aof() -> None:
    redis_args = _service_env(_falkordb_scratch_service())["REDIS_ARGS"]
    options = _redis_options(redis_args)
    assert options.get("appendonly") == "no", f"REDIS_ARGS = {redis_args!r}"
    assert options.get("save") == "3600 1", (
        "one rare save point keeps the seeded reference across restarts without "
        "forking during every lane pass, which would make GRAPH.COPY refuse; "
        f"REDIS_ARGS = {redis_args!r}"
    )


@pytest.mark.parametrize(
    ("redis_args", "expected"),
    [
        ("--appendonly yes --appendfsync everysec", {"appendonly": "yes", "appendfsync": "everysec"}),
        ("--appendonly no --save 3600 1", {"appendonly": "no", "save": "3600 1"}),
    ],
)
def test_redis_options_groups_values_until_the_next_option(redis_args, expected) -> None:
    assert _redis_options(redis_args) == expected


def test_scratch_has_the_ping_healthcheck_and_restarts_unless_stopped() -> None:
    scratch = _falkordb_scratch_service()
    assert scratch.get("healthcheck") == _falkordb_service().get("healthcheck")
    assert scratch["healthcheck"]["test"] == ["CMD", "redis-cli", "ping"]
    assert scratch.get("restart") == "unless-stopped"


def test_config_default_falkordb_uri_is_production(monkeypatch) -> None:
    from fused_memory.config.schema import FusedMemoryConfig

    monkeypatch.delenv("FALKORDB_URI", raising=False)

    assert FusedMemoryConfig().graphiti.falkordb.uri == PRODUCTION_FALKORDB.uri


def test_schema_default_falkordb_uri_is_production() -> None:
    from fused_memory.config.schema import FalkorDBProviderConfig

    assert FalkorDBProviderConfig().uri == PRODUCTION_FALKORDB.uri
