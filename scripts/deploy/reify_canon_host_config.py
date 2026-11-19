#!/usr/bin/env python3
"""Write or check a host's reify-canon host configuration, PRD C13's host.toml.

This module is the one home of host.toml's keys, their defaults and their
value rules (plans/canonical-build-path-shared-cache-prd.md C13).
install_build_cache_store.sh --host-config calls it for both directions, so
it cannot write a file that its own check rejects.

``write PATH --credentials FILE --default-store-url URL [--enabled ...]
[--policy ...] [--store-url ...] [--fallback-dir ...] [--fallback-cap ...]``
takes each of the five configurable keys from its flag, else from the
existing file, else from its default. store_user and store_password always
come from the credentials file's RCLONE_USER= and RCLONE_PASS= lines, which
are read and never sourced. Nothing is written unless every value passes.
The file is then written mode 0600 (new directories 0700) through a temp file
and a rename, and each changed value is logged old -> new (the password only
as "changed").

``check PATH`` checks the file's mode, its keys, and their types and values
under the same rules. It then proves that store_url answers 404 for a missing
key with the file's credentials, and 401 without them.

Exit 0 on success, 1 on any refusal or failed check, 2 on a usage error.
Messages go to stderr.
"""
from __future__ import annotations

import argparse
import base64
import logging
import os
import re
import secrets
import sys
import tomllib
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path

# Bind scripts/ to THIS checkout, so _checkout binds this checkout's shared.
# A bare sys.path call, the one statement ruff's E402 lets an import follow.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _checkout as _checkout

from shared import safe_io

EXIT_OK = 0
EXIT_FAILED = 1

KEYS = (
    "enabled",
    "store_url",
    "store_user",
    "store_password",
    "fallback_dir",
    "fallback_cap",
    "policy",
)
CONFIGURABLE_KEYS = ("enabled", "store_url", "fallback_dir", "fallback_cap", "policy")
SECRET_KEYS = frozenset({"store_password"})
POLICIES = ("shared", "mixed", "incremental")
DEFAULT_ENABLED = False
DEFAULT_POLICY = "shared"
DEFAULT_FALLBACK_CAP = "50G"
CREDENTIAL_KEY_OF = {"store_user": "RCLONE_USER", "store_password": "RCLONE_PASS"}

_TOML_UNSAFE = re.compile(r'["\\\x00-\x1f\x7f]')
_STORE_URL = re.compile(r"https?://.+")
_FALLBACK_CAP = re.compile(r"[0-9]+[KMGT]?")
_HTTP_TIMEOUT_SECS = 5
_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_LOG_PREFIX = "reify_canon_host_config:"
_log = logging.getLogger("reify_canon_host_config")


def default_fallback_dir() -> str:
    return str(Path.home() / ".cache" / "reify-canon" / "sccache-fallback")


def value_failures(config: Mapping[str, object]) -> list[str]:
    """Why each of the seven values breaks a rule; empty when all pass."""
    failures: list[str] = []
    if not isinstance(config.get("enabled"), bool):
        failures.append(f"enabled {config.get('enabled')!r} is not true or false")
    strings: dict[str, str] = {}
    for key in KEYS[1:]:
        value = config.get(key)
        shown = "<hidden>" if key in SECRET_KEYS else repr(value)
        if not isinstance(value, str) or not value:
            failures.append(f"{key} {shown} is not a non-empty string")
        elif _TOML_UNSAFE.search(value):
            failures.append(f"{key} {shown} holds a quote, backslash or control character")
        else:
            strings[key] = value
    policy = strings.get("policy")
    if policy is not None and policy not in POLICIES:
        failures.append(f"policy {policy!r} is not one of {', '.join(POLICIES)}")
    store_url = strings.get("store_url")
    if store_url is not None and not _STORE_URL.fullmatch(store_url):
        failures.append(f"store_url {store_url!r} is not an http(s) URL")
    fallback_dir = strings.get("fallback_dir")
    if fallback_dir is not None and not fallback_dir.startswith("/"):
        failures.append(f"fallback_dir {fallback_dir!r} is not absolute")
    fallback_cap = strings.get("fallback_cap")
    if fallback_cap is not None and not _FALLBACK_CAP.fullmatch(fallback_cap):
        failures.append(f"fallback_cap {fallback_cap!r} is not a size like 50G")
    return failures


def render(config: Mapping[str, object]) -> str:
    """The TOML text of a config that passed value_failures; no value needs escaping."""
    lines = []
    for key in KEYS:
        value = config[key]
        if isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        else:
            lines.append(f'{key} = "{value}"')
    return "".join(f"{line}\n" for line in lines)


def read_credentials(path: Path) -> dict[str, str]:
    """The NAME=value lines of a credentials file; a later line wins, as in systemd."""
    pairs: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.partition("=")
        if sep:
            pairs[name] = value
    return pairs


def _parse_enabled(text: str) -> bool | str:
    return {"true": True, "false": False}.get(text, text)


def resolve(
    flags: Mapping[str, object | None],
    existing: Mapping[str, object],
    defaults: Mapping[str, object],
    credentials: Mapping[str, str],
) -> dict[str, object]:
    """Each configurable key from its flag, else the existing file, else its default."""
    config: dict[str, object] = {}
    for key in CONFIGURABLE_KEYS:
        flag = flags.get(key)
        config[key] = flag if flag is not None else existing.get(key, defaults[key])
    for key, credential in CREDENTIAL_KEY_OF.items():
        config[key] = credentials.get(credential, "")
    return {key: config[key] for key in KEYS}


def _read_existing(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _write_private(path: Path, text: str) -> None:
    """Write *text* to *path* atomically, mode 0600, creating missing directories 0700."""
    os.umask(0o077)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_io.atomic_write_text(path, text, mode=0o600)


def _log_transitions(existing: Mapping[str, object], config: Mapping[str, object]) -> None:
    for key in KEYS:
        if key not in existing or existing[key] == config[key]:
            continue
        if key in SECRET_KEYS:
            _log.info("%s %s changed", _LOG_PREFIX, key)
        else:
            _log.info("%s %s: %r -> %r", _LOG_PREFIX, key, existing[key], config[key])


def write(args: argparse.Namespace) -> int:
    path: Path = args.path
    try:
        credentials = read_credentials(args.credentials)
    except OSError as exc:
        _log.error("%s cannot read credentials %s: %s; nothing was written", _LOG_PREFIX, args.credentials, exc)
        return EXIT_FAILED
    try:
        existing = _read_existing(path)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        _log.error("%s the existing %s is unreadable (%s); fix or remove it; nothing was written", _LOG_PREFIX, path, exc)
        return EXIT_FAILED
    flags = {
        "enabled": None if args.enabled is None else _parse_enabled(args.enabled),
        "policy": args.policy,
        "store_url": args.store_url,
        "fallback_dir": args.fallback_dir,
        "fallback_cap": args.fallback_cap,
    }
    defaults = {
        "enabled": DEFAULT_ENABLED,
        "policy": DEFAULT_POLICY,
        "store_url": args.default_store_url,
        "fallback_dir": default_fallback_dir(),
        "fallback_cap": DEFAULT_FALLBACK_CAP,
    }
    config = resolve(flags, existing, defaults, credentials)
    missing = [
        f"{args.credentials} has no {credential}= line"
        for credential in CREDENTIAL_KEY_OF.values()
        if credential not in credentials
    ]
    failures = missing or value_failures(config)
    if failures:
        for failure in failures:
            _log.error("%s %s", _LOG_PREFIX, failure)
        _log.error("%s nothing was written to %s", _LOG_PREFIX, path)
        return EXIT_FAILED
    _log_transitions(existing, config)
    _write_private(path, render(config))
    _log.info(
        "%s wrote %s: enabled=%s policy=%s store_url=%s",
        _LOG_PREFIX,
        path,
        config["enabled"],
        config["policy"],
        config["store_url"],
    )
    return EXIT_OK


def http_status(url: str, auth: tuple[str, str] | None = None) -> str:
    """The status of a GET as text; "000" when nothing answered."""
    request = urllib.request.Request(url, method="GET")
    if auth is not None:
        token = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
        request.add_header("Authorization", f"Basic {token}")
    try:
        with _NO_PROXY.open(request, timeout=_HTTP_TIMEOUT_SECS) as response:
            return str(response.status)
    except urllib.error.HTTPError as err:
        return str(err.code)
    except OSError:
        return "000"


def store_failures(path: Path, config: Mapping[str, object]) -> list[str]:
    store_url = str(config["store_url"])
    url = f"{store_url.rstrip('/')}/{secrets.token_hex(16)}"
    failures: list[str] = []
    anonymous = http_status(url)
    if anonymous != "401":
        failures.append(f"an unauthenticated GET of {url} answered {anonymous}, not 401")
    authenticated = http_status(url, (str(config["store_user"]), str(config["store_password"])))
    if authenticated == "401":
        failures.append(f"the store at {store_url} rejected the credentials in {path} (401)")
    elif authenticated != "404":
        failures.append(f"an authenticated GET of {url} answered {authenticated}, not 404")
    return failures


def check_failures(path: Path) -> list[str]:
    if not path.is_file():
        return [f"{path} is missing"]
    mode = path.stat().st_mode & 0o777
    failures = [] if mode == 0o600 else [f"{path} has mode {mode:04o}, not 0600"]
    try:
        config = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return [*failures, f"{path} is not valid TOML: {exc}"]
    if set(config) != set(KEYS):
        return [*failures, f"{path} has keys {sorted(config)}, not exactly {sorted(KEYS)}"]
    invalid = [f"{path}: {failure}" for failure in value_failures(config)]
    return [*failures, *(invalid or store_failures(path, config))]


def check(args: argparse.Namespace) -> int:
    failures = check_failures(args.path)
    for failure in failures:
        _log.error("%s CHECK FAILED: %s", _LOG_PREFIX, failure)
    return EXIT_FAILED if failures else EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reify_canon_host_config.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    write_parser = commands.add_parser("write", help="write host.toml")
    write_parser.add_argument("path", type=Path)
    write_parser.add_argument("--credentials", type=Path, required=True)
    write_parser.add_argument("--default-store-url", required=True)
    write_parser.add_argument("--enabled", help="true or false")
    write_parser.add_argument("--policy", help=" or ".join(POLICIES))
    write_parser.add_argument("--store-url")
    write_parser.add_argument("--fallback-dir")
    write_parser.add_argument("--fallback-cap")
    write_parser.set_defaults(run=write)
    check_parser = commands.add_parser("check", help="check host.toml and the store")
    check_parser.add_argument("path", type=Path)
    check_parser.set_defaults(run=check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
