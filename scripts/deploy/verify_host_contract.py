#!/usr/bin/env python3
"""The verify-host contract (plans/verify-host-roster-and-rented-trial-prd.md §6.1).

REQUIREMENTS is the one table of what a remote verify host must provide:
check_verify_host.py evaluates every row and provision_verify_host.sh runs one
step per row id. The renders a host installs (its per-project YAML, its
orchestrator wrapper) are built here from the private verify-host roster.

Usage:
  verify_host_contract.py requirements
  verify_host_contract.py render HOST [--roster FILE] --out DIR
"""

import argparse
import os
import posixpath
import re
import shlex
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

# Bare sys.path calls, the one statement ruff's E402 lets an import follow.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _checkout as _checkout
import verify_host_observe
import yaml
from orchestrator.config import VerifyRunnerConfig
from orchestrator.verify_host_roster_file import (
    Roster,
    RosterFileError,
    load_roster,
    roster_file_path,
)
from orchestrator.verify_runner import REMOTE_TOOL_PATH_PRELUDE

DISPATCHER_SSH_OPTS = (
    '-o', 'BatchMode=yes',
    '-o', 'ConnectTimeout=10',
    '-o', 'ServerAliveInterval=15',
    '-o', 'ServerAliveCountMax=4',
)
DEFAULT_WRAPPER_PATH = '/usr/local/bin/orchestrator'
WRAPPER_PATH_ENV = 'VERIFY_HOST_WRAPPER_PATH'
DEFAULT_SSH_BIN = 'ssh'
SSH_BIN_ENV = 'VERIFY_HOST_SSH'
SYNC_WRITE_WARN_MS = 20.0
PROJECT_ROOT_KEY = 'project_root'
MEMORY_MAX_GIB_KEYS = ('verify_cgroup_memory', 'max_gib')

WRAPPER_TEMPLATE = Path(__file__).with_name('verify-host-wrapper.template.sh')
GENERATED_HEADER = (
    '# Rendered from the verify-host roster by scripts/deploy/verify_host_contract.py; '
    'edits here are overwritten.\n'
)
_TEMPLATE_TOKEN = re.compile(r'@@[A-Z_]+@@')
_UNSAFE_IN_DOUBLE_QUOTES = re.compile(r'["$`\\]')

VerdictValue = Literal['ok', 'fail', 'warn']


class ContractError(ValueError):
    """The roster cannot describe a host this contract can provision or check."""


@dataclass(frozen=True)
class Verdict:
    verdict: VerdictValue
    observed: str
    needed: str


@dataclass(frozen=True)
class Requirement:
    id: str
    title: str
    needed: str
    evaluate: Callable[..., Verdict]


@dataclass(frozen=True)
class ProjectSpec:
    project_id: str
    config_path: str
    project_root: str
    host_config: Mapping[str, Any]
    disk_footprint_gib: int | None


@dataclass(frozen=True)
class HostSpec:
    """One roster host, with every path validated by the dispatcher's own runner rules."""

    name: str
    ssh_host: str
    df_checkout_path: str
    gate_checkouts_path: str
    wrapper_path: str
    wrapper_env: Mapping[str, str]
    projects: tuple[ProjectSpec, ...]

    @classmethod
    def from_roster(cls, roster: Roster, host_name: str, wrapper_path: str = DEFAULT_WRAPPER_PATH) -> 'HostSpec':
        host = roster.host(host_name)
        if not host.projects:
            raise ContractError(f'host {host_name!r}: names no project, so nothing would verify there')
        gate_roots = {_gate_root(roster, host_name, project_id) for project_id in host.projects}
        _require_double_quote_safe(host_name, 'df_checkout_path', host.df_checkout_path)
        _require_double_quote_safe(host_name, 'wrapper path', wrapper_path)
        (gate_root,) = gate_roots
        _require_double_quote_safe(host_name, 'gate_checkouts_path', gate_root)
        return cls(
            name=host_name,
            ssh_host=host.ssh_host,
            df_checkout_path=host.df_checkout_path,
            gate_checkouts_path=gate_root,
            wrapper_path=wrapper_path,
            wrapper_env=dict(host.wrapper_env),
            projects=tuple(
                _project_spec(host_name, project_id, project.config_path, project.host_config,
                              project.disk_footprint_gib)
                for project_id, project in host.projects.items()
            ),
        )


def _gate_root(roster: Roster, host_name: str, project_id: str) -> str:
    (runner,) = (runner for runner in roster.runners_for_project(project_id) if runner.name == host_name)
    try:
        return VerifyRunnerConfig.model_validate(asdict(runner)).gate_paths().root
    except ValueError as exc:
        raise ContractError(f'host {host_name!r} project {project_id!r}: {exc}') from exc


def _require_double_quote_safe(host_name: str, what: str, value: str) -> None:
    if _UNSAFE_IN_DOUBLE_QUOTES.search(value):
        raise ContractError(
            f'host {host_name!r}: {what} {value!r} holds a character unsafe inside double quotes (" $ ` \\)'
        )


def _project_spec(
    host_name: str, project_id: str, config_path: str | None, host_config: Mapping[str, Any],
    disk_footprint_gib: int | None,
) -> ProjectSpec:
    where = f'host {host_name!r} project {project_id!r}'
    if config_path is None:
        raise ContractError(f'{where}: config_path is required; the rendered host YAML is installed there')
    project_root = host_config.get(PROJECT_ROOT_KEY)
    if not isinstance(project_root, str) or not posixpath.isabs(project_root):
        raise ContractError(f'{where}: host_config.{PROJECT_ROOT_KEY} must be an absolute path, got {project_root!r}')
    _require_double_quote_safe(host_name, f'project {project_id!r} config_path', config_path)
    _require_double_quote_safe(host_name, f'project {project_id!r} {PROJECT_ROOT_KEY}', project_root)
    return ProjectSpec(
        project_id=project_id,
        config_path=config_path,
        project_root=project_root,
        host_config=host_config,
        disk_footprint_gib=disk_footprint_gib,
    )


def render_host_config(host_config: Mapping[str, Any]) -> str:
    return GENERATED_HEADER + yaml.safe_dump(dict(host_config), sort_keys=True, default_flow_style=False)


def render_wrapper(spec: HostSpec) -> str:
    template = WRAPPER_TEMPLATE.read_text()
    values = {
        '@@TOOL_PATH_PRELUDE@@': REMOTE_TOOL_PATH_PRELUDE,
        '@@WRAPPER_ENV@@': '\n'.join(
            f'export {name}=${{{name}:-{shlex.quote(value)}}}' for name, value in spec.wrapper_env.items()
        ),
        '@@DF_CHECKOUT_PATH@@': spec.df_checkout_path,
    }
    found = set(_TEMPLATE_TOKEN.findall(template))
    if found != values.keys():
        raise ContractError(f'{WRAPPER_TEMPLATE}: tokens {sorted(found)} differ from the rendered {sorted(values)}')
    for token, value in values.items():
        template = template.replace(token, value)
    return template


def render_host_env(spec: HostSpec, reference: verify_host_observe.EnvironmentFacts, ssh_bin: str) -> str:
    scalars = {
        'HOST_NAME': spec.name,
        'SSH_HOST': spec.ssh_host,
        'SSH_BIN': ssh_bin,
        'DF_CHECKOUT_PATH': spec.df_checkout_path,
        'GATE_CHECKOUTS_PATH': spec.gate_checkouts_path,
        'WRAPPER_PATH': spec.wrapper_path,
        'REF_TIME_ZONE': reference.time_zone or '',
        'REF_LOCALE': reference.locale or '',
        'REF_UV_VERSION': reference.uv_version or '',
        'REF_NODE_VERSION': reference.node_version or '',
    }
    arrays = {
        'SSH_OPTS': DISPATCHER_SSH_OPTS,
        'PROJECT_IDS': tuple(project.project_id for project in spec.projects),
        'CONFIG_PATHS': tuple(project.config_path for project in spec.projects),
        'PROJECT_ROOTS': tuple(project.project_root for project in spec.projects),
    }
    lines = [f'{name}={shlex.quote(value)}' for name, value in scalars.items()]
    lines += [f'{name}=({" ".join(shlex.quote(item) for item in items)})' for name, items in arrays.items()]
    return '\n'.join(lines) + '\n'


def _not_yet_evaluated(*_: object) -> Verdict:
    raise NotImplementedError('this requirement has no evaluator yet')


REQUIREMENTS = (
    Requirement('network', 'Network',
                "reachable with the dispatcher's own ssh options; host key pinned in the workstation's known_hosts",
                _not_yet_evaluated),
    Requirement('os_and_user', 'OS and user',
                'Ubuntu 24.04; a non-root user; git identity set', _not_yet_evaluated),
    Requirement('time_zone_and_locale', 'Time zone and locale',
                "the workstation's time zone and locale, the locale available; tzdata-legacy installed",
                _not_yet_evaluated),
    Requirement('toolchain', 'Toolchain',
                "git, bash, flock, npm; uv at the workstation's version; uv-managed CPython 3.13; "
                "node at the workstation's major", _not_yet_evaluated),
    Requirement('systemd', 'systemd',
                '`systemd-run --user --scope true` succeeds over non-interactive ssh (linger on)',
                _not_yet_evaluated),
    Requirement('kernel', 'Kernel', 'Landlock ABI 2 or higher; /var/tmp writable', _not_yet_evaluated),
    Requirement('memory', 'Memory',
                'swap present; MemAvailable at least the sum of the projects\' verify_cgroup_memory.max_gib',
                _not_yet_evaluated),
    Requirement('disk', 'Disk',
                f"free space at least the projects' disk_footprint_gib; synchronous writes under "
                f'{SYNC_WRITE_WARN_MS:g} ms (warn above)', _not_yet_evaluated),
    Requirement('checkouts', 'Checkouts',
                'the Dark Factory checkout and each project checkout exist with '
                'receive.denyCurrentBranch=updateInstead; the Dark Factory venv has its orchestrator',
                _not_yet_evaluated),
    Requirement('wrapper', 'Wrapper',
                'the installed wrapper equals the rendered template and honours ORCH_GATE_BIN',
                _not_yet_evaluated),
    Requirement('per_host_yaml', 'Per-host YAML',
                "each project's config_path holds a fresh render of its host_config", _not_yet_evaluated),
    Requirement('no_credentials', 'No credentials',
                "no agent OAuth token file in the verify user's home", _not_yet_evaluated),
)


def renders(spec: HostSpec, reference: verify_host_observe.EnvironmentFacts, ssh_bin: str) -> dict[str, str]:
    """Every file `render` writes, by name, built before any is written."""
    files = {f'{project.project_id}.yaml': render_host_config(project.host_config) for project in spec.projects}
    files['wrapper.sh'] = render_wrapper(spec)
    files['host.env'] = render_host_env(spec, reference, ssh_bin)
    return files


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('requirements', help='print the requirement ids, one per line, in execution order')
    render = commands.add_parser('render', help="write a host's per-project YAML, wrapper.sh and host.env")
    render.add_argument('host')
    render.add_argument('--roster', type=Path, help='roster file (default: the ORCH_VERIFY_HOSTS_FILE resolution)')
    render.add_argument('--out', type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == 'requirements':
        print('\n'.join(row.id for row in REQUIREMENTS))
        return 0
    roster_path = args.roster or roster_file_path(os.environ)
    try:
        spec = HostSpec.from_roster(
            load_roster(roster_path), args.host, os.environ.get(WRAPPER_PATH_ENV, DEFAULT_WRAPPER_PATH),
        )
        files = renders(spec, verify_host_observe.observe_environment(), os.environ.get(SSH_BIN_ENV, DEFAULT_SSH_BIN))
    except (ContractError, RosterFileError) as exc:
        print(f'verify_host_contract: {exc}', file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (args.out / name).write_text(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
