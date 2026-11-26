#!/usr/bin/env python3
"""Observe a verify host's facts and print them as one JSON object.

Stdlib only and Python >= 3.10: check_verify_host.py ships this file to the host
over ssh stdin, and runs it locally for the workstation's reference facts.
EnvironmentFacts and HostFacts are the single definition of that JSON.
"""

import argparse
import ctypes
import dataclasses
import json
import os
import pwd
import subprocess
import sys
import tempfile
import time
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

SYS_landlock_create_ruleset = 444
LANDLOCK_CREATE_RULESET_VERSION = 1

PROBE_TIMEOUT_SECS = 120
SYNC_PROBE_WRITES = 32
SYNC_PROBE_BLOCK_BYTES = 4096
GIB = 1 << 30
CREDENTIAL_FILE = Path('.claude') / '.credentials.json'
WRAPPER_PROBE_ARG = 'gate-identity'

_Facts = TypeVar('_Facts', bound='_StrictJson')


class _StrictJson:
    @classmethod
    def from_json(cls: type[_Facts], text: str) -> _Facts:
        return _build(cls, json.loads(text), cls.__name__)


def _build(cls: Any, data: object, where: str) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f'{where}: expected a JSON object, got {type(data).__name__}')
    types = {field.name: field.type for field in dataclasses.fields(cls)}
    missing = sorted(types.keys() - data.keys())
    unknown = sorted(data.keys() - types.keys())
    if missing or unknown:
        raise ValueError(f'{where}: missing keys {missing}, unknown keys {unknown}')
    return cls(**{name: _member(types[name], data[name], f'{where}.{name}') for name in types})


def _member(annotation: Any, value: object, where: str) -> Any:
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return _build(annotation, value, where)
    args = typing.get_args(annotation)
    if typing.get_origin(annotation) is dict and isinstance(value, dict) and dataclasses.is_dataclass(args[1]):
        return {key: _build(args[1], item, f'{where}[{key!r}]') for key, item in value.items()}
    return value


@dataclass(frozen=True)
class EnvironmentFacts(_StrictJson):
    time_zone: str | None
    locale: str | None
    locale_available: bool
    tzdata_legacy_installed: bool
    uv_version: str | None
    node_version: str | None


@dataclass(frozen=True)
class CheckoutFacts(_StrictJson):
    exists: bool
    receive_deny_current_branch: str | None


@dataclass(frozen=True)
class HostFacts(_StrictJson):
    environment: EnvironmentFacts
    os_id: str | None
    os_version_id: str | None
    user: str | None
    uid: int
    git_user_name: str | None
    git_user_email: str | None
    git_version: str | None
    bash_version: str | None
    flock_version: str | None
    npm_version: str | None
    uv_python_313: str | None
    systemd_run_scope_rc: int | None
    linger: str | None
    landlock_abi: int
    var_tmp_writable: bool
    mem_available_kib: int | None
    swap_total_kib: int | None
    disk_free_gib: int | None
    sync_write_ms: float | None
    checkouts: dict[str, CheckoutFacts]
    df_orchestrator_executable: bool
    wrapper_text: str | None
    wrapper_probe_stdout: str | None
    wrapper_probe_rc: int | None
    configs: dict[str, str | None]
    credentials_present: bool


def _run(argv: list[str], env: dict[str, str] | None = None) -> 'subprocess.CompletedProcess[str] | None':
    try:
        return subprocess.run(
            argv, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=PROBE_TIMEOUT_SECS, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _stdout(argv: list[str]) -> str | None:
    proc = _run(argv)
    if proc is None or proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _first_line(argv: list[str]) -> str | None:
    out = _stdout(argv)
    return out.splitlines()[0] if out else None


def _returncode(argv: list[str]) -> int | None:
    proc = _run(argv)
    return None if proc is None else proc.returncode


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text()
    except (OSError, UnicodeDecodeError):
        return None


def _uv_version() -> str | None:
    words = (_first_line(['uv', '--version']) or '').split()
    return words[1] if len(words) > 1 else None


def _normalised_locale(name: str) -> str:
    return name.lower().replace('-', '')


def _locale_available(name: str | None) -> bool:
    if name is None or name in ('C', 'POSIX'):
        return True
    listing = _stdout(['locale', '-a']) or ''
    return _normalised_locale(name) in {_normalised_locale(line) for line in listing.splitlines()}


def observe_environment() -> EnvironmentFacts:
    locale = os.environ.get('LC_ALL') or os.environ.get('LANG') or None
    tzdata_legacy = _stdout(['dpkg-query', '-W', '-f', '${Status}', 'tzdata-legacy'])
    return EnvironmentFacts(
        time_zone=_stdout(['timedatectl', 'show', '-p', 'Timezone', '--value']),
        locale=locale,
        locale_available=_locale_available(locale),
        tzdata_legacy_installed=tzdata_legacy == 'install ok installed',
        uv_version=_uv_version(),
        node_version=_first_line(['node', '--version']),
    )


def _os_release() -> dict[str, str]:
    text = _read_text(Path('/etc/os-release')) or _read_text(Path('/usr/lib/os-release')) or ''
    pairs = (line.split('=', 1) for line in text.splitlines() if '=' in line)
    return {key: value.strip().strip('"') for key, value in pairs}


def _user_name() -> str | None:
    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return None


def _landlock_abi() -> int:
    try:
        libc = ctypes.CDLL('libc.so.6', use_errno=True)
        return int(libc.syscall(SYS_landlock_create_ruleset, None, 0, LANDLOCK_CREATE_RULESET_VERSION))
    except (OSError, AttributeError):
        return -1


def _meminfo_kib() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in (_read_text(Path('/proc/meminfo')) or '').splitlines():
        key, _, rest = line.partition(':')
        words = rest.split()
        if words and words[0].isdigit():
            values[key] = int(words[0])
    return values


def _existing_directory(path: Path) -> Path:
    return next((candidate for candidate in (path, *path.parents) if candidate.is_dir()), Path('/'))


def _disk_free_gib(directory: Path) -> int | None:
    try:
        stats = os.statvfs(directory)
    except OSError:
        return None
    return stats.f_bavail * stats.f_frsize // GIB


def _sync_write_ms(directory: Path) -> float | None:
    try:
        handle, name = tempfile.mkstemp(dir=directory, prefix='.verify-host-sync-probe-')
    except OSError:
        return None
    os.close(handle)
    try:
        return _timed_sync_writes(name)
    except OSError:
        return None
    finally:
        Path(name).unlink(missing_ok=True)


def _timed_sync_writes(name: str) -> float:
    handle = os.open(name, os.O_WRONLY | os.O_DSYNC)
    block = bytes(SYNC_PROBE_BLOCK_BYTES)
    try:
        start = time.perf_counter()
        for _ in range(SYNC_PROBE_WRITES):
            os.write(handle, block)
        return (time.perf_counter() - start) * 1000 / SYNC_PROBE_WRITES
    finally:
        os.close(handle)


def _checkout(path: str) -> CheckoutFacts:
    if not Path(path).is_dir():
        return CheckoutFacts(exists=False, receive_deny_current_branch=None)
    setting = _stdout(['git', '-C', path, 'config', '--get', 'receive.denyCurrentBranch'])
    return CheckoutFacts(exists=True, receive_deny_current_branch=setting)


def _wrapper_probe(wrapper: str) -> tuple[str | None, int | None]:
    proc = _run([wrapper, WRAPPER_PROBE_ARG], env={**os.environ, 'ORCH_GATE_BIN': '/bin/echo'})
    if proc is None:
        return None, None
    return proc.stdout.strip(), proc.returncode


def observe_host(args: argparse.Namespace) -> HostFacts:
    os_release = _os_release()
    user = _user_name()
    meminfo = _meminfo_kib()
    disk_directory = _existing_directory(Path(args.disk_path))
    probe_stdout, probe_rc = _wrapper_probe(args.wrapper_path)
    checkouts = [args.df_checkout, *args.project_checkout]
    return HostFacts(
        environment=observe_environment(),
        os_id=os_release.get('ID'),
        os_version_id=os_release.get('VERSION_ID'),
        user=user,
        uid=os.getuid(),
        git_user_name=_stdout(['git', 'config', '--global', '--get', 'user.name']),
        git_user_email=_stdout(['git', 'config', '--global', '--get', 'user.email']),
        git_version=_first_line(['git', '--version']),
        bash_version=_first_line(['bash', '--version']),
        flock_version=_first_line(['flock', '--version']),
        npm_version=_first_line(['npm', '--version']),
        uv_python_313=_first_line(['uv', 'python', 'find', '3.13']),
        systemd_run_scope_rc=_returncode(['systemd-run', '--user', '--scope', '--quiet', 'true']),
        linger=_stdout(['loginctl', 'show-user', user, '-p', 'Linger', '--value']) if user else None,
        landlock_abi=_landlock_abi(),
        var_tmp_writable=os.access('/var/tmp', os.W_OK),
        mem_available_kib=meminfo.get('MemAvailable'),
        swap_total_kib=meminfo.get('SwapTotal'),
        disk_free_gib=_disk_free_gib(disk_directory),
        sync_write_ms=_sync_write_ms(disk_directory),
        checkouts={path: _checkout(path) for path in checkouts},
        df_orchestrator_executable=os.access(Path(args.df_checkout) / '.venv' / 'bin' / 'orchestrator', os.X_OK),
        wrapper_text=_read_text(Path(args.wrapper_path)),
        wrapper_probe_stdout=probe_stdout,
        wrapper_probe_rc=probe_rc,
        configs={path: _read_text(Path(path)) for path in args.config_path},
        credentials_present=(Path.home() / CREDENTIAL_FILE).exists(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment-only', action='store_true')
    parser.add_argument('--df-checkout')
    parser.add_argument('--project-checkout', action='append', default=[])
    parser.add_argument('--config-path', action='append', default=[])
    parser.add_argument('--wrapper-path')
    parser.add_argument('--disk-path')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.environment_only:
        facts: EnvironmentFacts | HostFacts = observe_environment()
    elif None in (args.df_checkout, args.wrapper_path, args.disk_path):
        parser.error('a host observation needs --df-checkout, --wrapper-path and --disk-path')
    else:
        facts = observe_host(args)
    print(json.dumps(dataclasses.asdict(facts)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
