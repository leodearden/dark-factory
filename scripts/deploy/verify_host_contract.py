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
from collections.abc import Callable, Mapping, Sequence
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
from verify_host_observe import EnvironmentFacts, HostFacts

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
    evaluate: 'Evaluator'


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


def render_host_env(spec: HostSpec, reference: EnvironmentFacts, ssh_bin: str) -> str:
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


@dataclass(frozen=True)
class NetworkFacts:
    """What the checker's own ssh probes saw; the observer cannot report on its own transport."""

    reachable: bool
    host_key_pinned: bool
    detail: str


@dataclass(frozen=True)
class Observation:
    network: NetworkFacts
    host: HostFacts | None
    unobserved_reason: str


Evaluator = Callable[[Observation, EnvironmentFacts, HostSpec], Verdict]
HostEvaluator = Callable[[HostFacts, EnvironmentFacts, HostSpec], Verdict]

REQUIRED_OS = ('ubuntu', '24.04')
MIN_LANDLOCK_ABI = 2
UPDATE_INSTEAD = 'updateInstead'
GIB_IN_KIB = 1024 * 1024


def _judged(problems: Sequence[str], ok_observed: str, needed: str, warnings: Sequence[str] = ()) -> Verdict:
    if problems:
        return Verdict('fail', '; '.join(problems), needed)
    if warnings:
        return Verdict('warn', '; '.join(warnings), needed)
    return Verdict('ok', ok_observed, needed)


def _on_host_facts(evaluate: HostEvaluator) -> Evaluator:
    """Run *evaluate* on the host's facts, or fail the row when the host went unobserved."""

    def evaluate_observation(observation: Observation, reference: EnvironmentFacts, spec: HostSpec) -> Verdict:
        if observation.host is None:
            return Verdict('fail', f'not observed: {observation.unobserved_reason}', 'an observation of the host')
        return evaluate(observation.host, reference, spec)

    return evaluate_observation


def _network(observation: Observation, reference: EnvironmentFacts, spec: HostSpec) -> Verdict:
    network = observation.network
    needed = f'ssh {" ".join(DISPATCHER_SSH_OPTS)} {spec.ssh_host} true exits 0; its host key in known_hosts'
    problems = []
    if not network.reachable:
        problems.append(f'unreachable: {network.detail}')
    elif not network.host_key_pinned:
        problems.append(f'no known_hosts entry for {spec.ssh_host}: {network.detail}')
    return _judged(problems, f'{spec.ssh_host} reachable; host key pinned', needed)


def _os_and_user(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Verdict:
    problems = []
    if (host.os_id, host.os_version_id) != REQUIRED_OS:
        problems.append(f'os {host.os_id} {host.os_version_id}')
    if host.uid == 0:
        problems.append(f'user {host.user} is root')
    if not (host.git_user_name and host.git_user_email):
        problems.append(f'git identity {host.git_user_name!r} <{host.git_user_email!r}>')
    observed = f'{host.os_id} {host.os_version_id}; user {host.user} (uid {host.uid}); git identity set'
    return _judged(problems, observed, f'{" ".join(REQUIRED_OS)}; a non-root user; git user.name and user.email')


def _time_zone_and_locale(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Verdict:
    env = host.environment
    problems = []
    if env.time_zone is None or env.time_zone != reference.time_zone:
        problems.append(f'time zone {env.time_zone}')
    if env.locale != reference.locale:
        problems.append(f'locale {env.locale}')
    if not env.locale_available:
        problems.append(f'locale {env.locale} not available on the host')
    if not env.tzdata_legacy_installed:
        problems.append('tzdata-legacy not installed')
    needed = f'time zone {reference.time_zone}; locale {reference.locale}, available; tzdata-legacy installed'
    return _judged(problems, f'time zone {env.time_zone}; locale {env.locale}; tzdata-legacy installed', needed)


def _node_major(version: str | None) -> str | None:
    return version.split('.')[0] if version else None


def _toolchain(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Verdict:
    env = host.environment
    tools = {'git': host.git_version, 'bash': host.bash_version, 'flock': host.flock_version, 'npm': host.npm_version}
    problems = [f'{tool} missing' for tool, version in tools.items() if version is None]
    if reference.uv_version is None or env.uv_version != reference.uv_version:
        problems.append(f'uv {env.uv_version}')
    if host.uv_python_313 is None:
        problems.append('no CPython 3.13 that uv finds')
    node_major = _node_major(reference.node_version)
    if node_major is None or _node_major(env.node_version) != node_major:
        problems.append(f'node {env.node_version}')
    needed = f'git, bash, flock, npm; uv {reference.uv_version}; uv-managed CPython 3.13; node {node_major}.x'
    observed = f'uv {env.uv_version}; node {env.node_version}; CPython 3.13 at {host.uv_python_313}'
    return _judged(problems, observed, needed)


def _systemd(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Verdict:
    problems = []
    if host.systemd_run_scope_rc != 0:
        problems.append(f'systemd-run --user --scope true exited {host.systemd_run_scope_rc}')
    if host.linger != 'yes':
        problems.append(f'linger {host.linger}')
    return _judged(problems, 'user scope runs; linger yes', 'systemd-run --user --scope true exits 0; linger yes')


def _kernel(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Verdict:
    problems = []
    if host.landlock_abi < MIN_LANDLOCK_ABI:
        problems.append(f'Landlock ABI {host.landlock_abi}' if host.landlock_abi >= 0 else 'no Landlock')
    if not host.var_tmp_writable:
        problems.append('/var/tmp not writable')
    observed = f'Landlock ABI {host.landlock_abi}; /var/tmp writable'
    return _judged(problems, observed, f'Landlock ABI {MIN_LANDLOCK_ABI} or higher; /var/tmp writable')


def _scope_max_gib(host_config: Mapping[str, Any]) -> int | None:
    section_key, value_key = MEMORY_MAX_GIB_KEYS
    section = host_config.get(section_key)
    value = section.get(value_key) if isinstance(section, Mapping) else None
    return value if isinstance(value, int) else None


def _memory(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Verdict:
    caps = [cap for cap in (_scope_max_gib(project.host_config) for project in spec.projects) if cap is not None]
    needed_gib = sum(caps)
    problems = []
    if not host.swap_total_kib:
        problems.append(f'swap {host.swap_total_kib} KiB')
    if host.mem_available_kib is None or host.mem_available_kib < needed_gib * GIB_IN_KIB:
        problems.append(f'MemAvailable {host.mem_available_kib} KiB')
    warnings = [] if caps else ['no project states verify_cgroup_memory.max_gib']
    needed = f'swap present; MemAvailable at least {needed_gib} GiB (the summed verify_cgroup_memory.max_gib)'
    observed = f'MemAvailable {host.mem_available_kib} KiB; swap {host.swap_total_kib} KiB'
    return _judged(problems, observed, needed, warnings)


def _disk(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Verdict:
    footprints = [project.disk_footprint_gib for project in spec.projects if project.disk_footprint_gib is not None]
    needed_gib = sum(footprints)
    problems = []
    if host.disk_free_gib is None or host.disk_free_gib < needed_gib:
        problems.append(f'{host.disk_free_gib} GiB free')
    warnings = [] if footprints else ['no project states disk_footprint_gib']
    if host.sync_write_ms is None or host.sync_write_ms > SYNC_WRITE_WARN_MS:
        warnings.append(f'synchronous write {host.sync_write_ms} ms')
    needed = (
        f'at least {needed_gib} GiB free (the summed disk_footprint_gib); '
        f'synchronous writes within {SYNC_WRITE_WARN_MS:g} ms'
    )
    return _judged(problems, f'{host.disk_free_gib} GiB free; synchronous write {host.sync_write_ms} ms', needed, warnings)


def _checkouts(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Verdict:
    roots = dict.fromkeys([spec.df_checkout_path, *(project.project_root for project in spec.projects)])
    problems = []
    for root in roots:
        facts = host.checkouts.get(root)
        if facts is None or not facts.exists:
            problems.append(f'{root} absent')
        elif facts.receive_deny_current_branch != UPDATE_INSTEAD:
            problems.append(f'{root} receive.denyCurrentBranch={facts.receive_deny_current_branch}')
    if not host.df_orchestrator_executable:
        problems.append(f'{spec.df_checkout_path}/.venv/bin/orchestrator not executable')
    needed = f'{", ".join(roots)} exist with receive.denyCurrentBranch={UPDATE_INSTEAD}; the df venv has its orchestrator'
    return _judged(problems, f'{len(roots)} checkouts accept pushes to their branch', needed)


def _wrapper(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Verdict:
    problems = []
    if host.wrapper_text is None:
        problems.append(f'{spec.wrapper_path} absent')
    elif host.wrapper_text != render_wrapper(spec):
        problems.append(f'{spec.wrapper_path} differs from a fresh render')
    probe = (host.wrapper_probe_stdout, host.wrapper_probe_rc)
    if probe != (verify_host_observe.WRAPPER_PROBE_ARG, 0):
        problems.append(f'with ORCH_GATE_BIN=/bin/echo it printed {probe[0]!r} and exited {probe[1]}')
    needed = (
        f'{spec.wrapper_path} equals a fresh render; with ORCH_GATE_BIN=/bin/echo it prints '
        f'{verify_host_observe.WRAPPER_PROBE_ARG!r}'
    )
    return _judged(problems, f'{spec.wrapper_path} as rendered; honours ORCH_GATE_BIN', needed)


def _per_host_yaml(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Verdict:
    problems = []
    for project in spec.projects:
        text = host.configs.get(project.config_path)
        if text is None:
            problems.append(f'project {project.project_id!r}: {project.config_path} missing')
        elif text != render_host_config(project.host_config):
            problems.append(f'project {project.project_id!r}: {project.config_path} differs from a fresh render')
    paths = ', '.join(project.config_path for project in spec.projects)
    return _judged(problems, f'{paths} as rendered', f'{paths} equal fresh renders of the roster host_config')


def _no_credentials(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Verdict:
    credential = f'~/{verify_host_observe.CREDENTIAL_FILE}'
    problems = [f'{credential} present for {host.user}'] if host.credentials_present else []
    return _judged(problems, f'no {credential}', f'no {credential} for the verify user')


REQUIREMENTS = (
    Requirement('network', 'Network',
                "reachable with the dispatcher's own ssh options; host key pinned in the workstation's known_hosts",
                _network),
    Requirement('os_and_user', 'OS and user',
                'Ubuntu 24.04; a non-root user; git identity set', _on_host_facts(_os_and_user)),
    Requirement('time_zone_and_locale', 'Time zone and locale',
                "the workstation's time zone and locale, the locale available; tzdata-legacy installed",
                _on_host_facts(_time_zone_and_locale)),
    Requirement('toolchain', 'Toolchain',
                "git, bash, flock, npm; uv at the workstation's version; uv-managed CPython 3.13; "
                "node at the workstation's major", _on_host_facts(_toolchain)),
    Requirement('systemd', 'systemd',
                '`systemd-run --user --scope true` succeeds over non-interactive ssh (linger on)',
                _on_host_facts(_systemd)),
    Requirement('kernel', 'Kernel', 'Landlock ABI 2 or higher; /var/tmp writable', _on_host_facts(_kernel)),
    Requirement('memory', 'Memory',
                "swap present; MemAvailable at least the sum of the projects' verify_cgroup_memory.max_gib",
                _on_host_facts(_memory)),
    Requirement('disk', 'Disk',
                f"free space at least the projects' disk_footprint_gib; synchronous writes under "
                f'{SYNC_WRITE_WARN_MS:g} ms (warn above)', _on_host_facts(_disk)),
    Requirement('checkouts', 'Checkouts',
                'the Dark Factory checkout and each project checkout exist with '
                'receive.denyCurrentBranch=updateInstead; the Dark Factory venv has its orchestrator',
                _on_host_facts(_checkouts)),
    Requirement('wrapper', 'Wrapper',
                'the installed wrapper equals the rendered template and honours ORCH_GATE_BIN',
                _on_host_facts(_wrapper)),
    Requirement('per_host_yaml', 'Per-host YAML',
                "each project's config_path holds a fresh render of its host_config", _on_host_facts(_per_host_yaml)),
    Requirement('no_credentials', 'No credentials',
                "no agent OAuth token file in the verify user's home", _on_host_facts(_no_credentials)),
)


def renders(spec: HostSpec, reference: EnvironmentFacts, ssh_bin: str) -> dict[str, str]:
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
