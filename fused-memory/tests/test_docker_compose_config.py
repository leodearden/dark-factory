"""Contract tests for the falkordb service in fused-memory/docker/docker-compose.yml.

The falkordb image entrypoint (/var/lib/falkordb/bin/run.sh) runs
``redis-server ${REDIS_ARGS} --dir /var/lib/falkordb/data ...`` with no config
file, so the compose file is the only durable home for redis options.
"""

import re
from pathlib import Path
from typing import Any

import yaml

FUSED_MEMORY_DIR = Path(__file__).resolve().parent.parent
COMPOSE_FILE = FUSED_MEMORY_DIR / "docker" / "docker-compose.yml"
CONFIG_FILE = FUSED_MEMORY_DIR / "config" / "config.yaml"
FALKORDB_DATA_DIR = "/var/lib/falkordb/data"
EXPECTED_FALKORDB_MODULE_ARGS = {
    "MAX_QUEUED_QUERIES": "25",
    "TIMEOUT": "30000",
    "RESULTSET_SIZE": "10000",
}


def _falkordb_service() -> dict[str, Any]:
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    return compose["services"]["falkordb"]


def _service_env(service: dict[str, Any]) -> dict[str, str]:
    environment = service.get("environment") or {}
    if isinstance(environment, dict):
        return {str(key): "" if value is None else str(value) for key, value in environment.items()}
    pairs = (str(entry).partition("=") for entry in environment)
    return {key: value for key, _, value in pairs}


def _redis_options(redis_args: str) -> dict[str, str]:
    tokens = redis_args.split()
    names, values = tokens[0::2], tokens[1::2]
    well_formed = (
        len(names) == len(values)
        and all(name.startswith("--") for name in names)
        and not any(value.startswith("--") for value in values)
    )
    assert well_formed, f"REDIS_ARGS must be '--option value' pairs; got {redis_args!r}"
    return {name.removeprefix("--"): value for name, value in zip(names, values, strict=True)}


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
