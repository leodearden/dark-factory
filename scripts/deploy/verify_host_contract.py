#!/usr/bin/env python3
"""The verify-host contract (plans/verify-host-roster-and-rented-trial-prd.md §6.1).

REQUIREMENTS is the one table of what a remote verify host must provide:
check_verify_host.py evaluates every row and provision_verify_host.sh runs one
step per row id. The renders a host installs (its per-project YAML, its
orchestrator wrapper) are built here from the private verify-host roster.

Usage:
  verify_host_contract.py requirements [--titles]
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
from orchestrator.config import GitConfig, VerifyRunnerConfig
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
MAIN_BRANCH_KEYS = ('git', 'main_branch')
DEFAULT_MAIN_BRANCH: str = GitConfig.model_fields['main_branch'].default

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
class ProjectSpec:
    project_id: str
    config_path: str
    project_root: str
    main_branch: str
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
        main_branch=_main_branch(where, host_config),
        host_config=host_config,
        disk_footprint_gib=disk_footprint_gib,
    )


def _main_branch(where: str, host_config: Mapping[str, Any]) -> str:
    section_key, value_key = MAIN_BRANCH_KEYS
    section = host_config.get(section_key, {})
    branch = section.get(value_key, DEFAULT_MAIN_BRANCH) if isinstance(section, Mapping) else None
    if not isinstance(branch, str) or not branch:
        raise ContractError(f'{where}: host_config.{section_key}.{value_key} must be a branch name, got {branch!r}')
    return branch


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
        'TOOL_PATH_PRELUDE': REMOTE_TOOL_PATH_PRELUDE,
        'REQUIRED_OS_ID': REQUIRED_OS[0],
        'REQUIRED_OS_VERSION_ID': REQUIRED_OS[1],
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
        'PROJECT_MAIN_BRANCHES': tuple(project.main_branch for project in spec.projects),
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


@dataclass(frozen=True)
class Finding:
    verdict: VerdictValue
    observed: str


NeedStatement = Callable[[EnvironmentFacts, HostSpec], str]
Judge = Callable[[Observation, EnvironmentFacts, HostSpec], Finding]
HostJudge = Callable[[HostFacts, EnvironmentFacts, HostSpec], Finding]


@dataclass(frozen=True)
class Requirement:
    id: str
    title: str
    needed: NeedStatement
    judge: Judge

    def evaluate(self, observation: Observation, reference: EnvironmentFacts, spec: HostSpec) -> Verdict:
        finding = self.judge(observation, reference, spec)
        return Verdict(finding.verdict, finding.observed, self.needed(reference, spec))


REQUIRED_OS = ('ubuntu', '24.04')
MIN_LANDLOCK_ABI = 2
UPDATE_INSTEAD = 'updateInstead'
GIB_IN_KIB = 1024 * 1024
REQUIRED_TOOLS: Mapping[str, Callable[[HostFacts], str | None]] = {
    'git': lambda host: host.git_version,
    'bash': lambda host: host.bash_version,
    'flock': lambda host: host.flock_version,
    'npm': lambda host: host.npm_version,
}
_SCOPE_CAP_SETTING = '.'.join(MEMORY_MAX_GIB_KEYS)
_WRAPPER_PROBE = f'with ORCH_GATE_BIN={verify_host_observe.WRAPPER_PROBE_GATE_BIN} it'
_CREDENTIAL_IN_HOME = f'~/{verify_host_observe.CREDENTIAL_FILE}'


def _judged(problems: Sequence[str], ok_observed: str, warnings: Sequence[str] = ()) -> Finding:
    if problems:
        return Finding('fail', '; '.join(problems))
    if warnings:
        return Finding('warn', '; '.join(warnings))
    return Finding('ok', ok_observed)


def _on_host_facts(judge: HostJudge) -> Judge:
    """Run *judge* on the host's facts, or fail the row when the host went unobserved."""

    def judge_observation(observation: Observation, reference: EnvironmentFacts, spec: HostSpec) -> Finding:
        if observation.host is None:
            return Finding('fail', f'not observed: {observation.unobserved_reason}')
        return judge(observation.host, reference, spec)

    return judge_observation


def _network_needed(reference: EnvironmentFacts, spec: HostSpec) -> str:
    return f'ssh {" ".join(DISPATCHER_SSH_OPTS)} {spec.ssh_host} true exits 0; its host key in known_hosts'


def _network(observation: Observation, reference: EnvironmentFacts, spec: HostSpec) -> Finding:
    network = observation.network
    problems = []
    if not network.reachable:
        problems.append(f'unreachable: {network.detail}')
    elif not network.host_key_pinned:
        problems.append(f'no known_hosts entry for {spec.ssh_host}: {network.detail}')
    return _judged(problems, f'{spec.ssh_host} reachable; host key pinned')


def _os_and_user_needed(reference: EnvironmentFacts, spec: HostSpec) -> str:
    return f'{" ".join(REQUIRED_OS)}; a non-root user; git user.name and user.email'


def _os_and_user(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Finding:
    problems = []
    if (host.os_id, host.os_version_id) != REQUIRED_OS:
        problems.append(f'os {host.os_id} {host.os_version_id}')
    if host.uid == 0:
        problems.append(f'user {host.user} is root')
    if not (host.git_user_name and host.git_user_email):
        problems.append(f'git identity {host.git_user_name!r} <{host.git_user_email!r}>')
    return _judged(problems, f'{host.os_id} {host.os_version_id}; user {host.user} (uid {host.uid}); git identity set')


def _time_zone_and_locale_needed(reference: EnvironmentFacts, spec: HostSpec) -> str:
    return f'time zone {reference.time_zone}; locale {reference.locale}, available; tzdata-legacy installed'


def _time_zone_and_locale(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Finding:
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
    return _judged(problems, f'time zone {env.time_zone}; locale {env.locale}; tzdata-legacy installed')


def _node_major(version: str | None) -> str | None:
    return version.split('.')[0] if version else None


def _toolchain_needed(reference: EnvironmentFacts, spec: HostSpec) -> str:
    return (
        f'{", ".join(REQUIRED_TOOLS)}; uv {reference.uv_version}; uv-managed CPython 3.13; '
        f'node {_node_major(reference.node_version)}.x'
    )


def _toolchain(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Finding:
    env = host.environment
    problems = [f'{tool} missing' for tool, version_of in REQUIRED_TOOLS.items() if version_of(host) is None]
    if reference.uv_version is None or env.uv_version != reference.uv_version:
        problems.append(f'uv {env.uv_version}')
    if host.uv_python_313 is None:
        problems.append('no CPython 3.13 that uv finds')
    node_major = _node_major(reference.node_version)
    if node_major is None or _node_major(env.node_version) != node_major:
        problems.append(f'node {env.node_version}')
    return _judged(problems, f'uv {env.uv_version}; node {env.node_version}; CPython 3.13 at {host.uv_python_313}')


def _systemd_needed(reference: EnvironmentFacts, spec: HostSpec) -> str:
    return 'systemd-run --user --scope true exits 0; linger yes'


def _systemd(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Finding:
    problems = []
    if host.systemd_run_scope_rc != 0:
        problems.append(f'systemd-run --user --scope true exited {host.systemd_run_scope_rc}')
    if host.linger != 'yes':
        problems.append(f'linger {host.linger}')
    return _judged(problems, 'user scope runs; linger yes')


def _kernel_needed(reference: EnvironmentFacts, spec: HostSpec) -> str:
    return f'Landlock ABI {MIN_LANDLOCK_ABI} or higher; /var/tmp writable'


def _kernel(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Finding:
    problems = []
    if host.landlock_abi < MIN_LANDLOCK_ABI:
        problems.append(f'Landlock ABI {host.landlock_abi}' if host.landlock_abi >= 0 else 'no Landlock')
    if not host.var_tmp_writable:
        problems.append('/var/tmp not writable')
    return _judged(problems, f'Landlock ABI {host.landlock_abi}; /var/tmp writable')


def _scope_max_gib(host_config: Mapping[str, Any]) -> int | None:
    section_key, value_key = MEMORY_MAX_GIB_KEYS
    section = host_config.get(section_key)
    value = section.get(value_key) if isinstance(section, Mapping) else None
    return value if isinstance(value, int) else None


def _scope_caps(spec: HostSpec) -> tuple[int, ...]:
    return tuple(cap for cap in (_scope_max_gib(project.host_config) for project in spec.projects) if cap is not None)


def _memory_needed(reference: EnvironmentFacts, spec: HostSpec) -> str:
    return f'swap present; MemAvailable at least {sum(_scope_caps(spec))} GiB (the summed {_SCOPE_CAP_SETTING})'


def _memory(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Finding:
    caps = _scope_caps(spec)
    problems = []
    if not host.swap_total_kib:
        problems.append(f'swap {host.swap_total_kib} KiB')
    if host.mem_available_kib is None or host.mem_available_kib < sum(caps) * GIB_IN_KIB:
        problems.append(f'MemAvailable {host.mem_available_kib} KiB')
    warnings = [] if caps else [f'no project states {_SCOPE_CAP_SETTING}']
    observed = f'MemAvailable {host.mem_available_kib} KiB; swap {host.swap_total_kib} KiB'
    return _judged(problems, observed, warnings)


def _disk_footprints(spec: HostSpec) -> tuple[int, ...]:
    return tuple(project.disk_footprint_gib for project in spec.projects if project.disk_footprint_gib is not None)


def _disk_needed(reference: EnvironmentFacts, spec: HostSpec) -> str:
    return (
        f'at least {sum(_disk_footprints(spec))} GiB free (the summed disk_footprint_gib); '
        f'synchronous writes within {SYNC_WRITE_WARN_MS:g} ms'
    )


def _disk(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Finding:
    footprints = _disk_footprints(spec)
    problems = []
    if host.disk_free_gib is None or host.disk_free_gib < sum(footprints):
        problems.append(f'{host.disk_free_gib} GiB free')
    warnings = [] if footprints else ['no project states disk_footprint_gib']
    if host.sync_write_ms is None or host.sync_write_ms > SYNC_WRITE_WARN_MS:
        warnings.append(f'synchronous write {host.sync_write_ms} ms')
    return _judged(problems, f'{host.disk_free_gib} GiB free; synchronous write {host.sync_write_ms} ms', warnings)


def _checkout_roots(spec: HostSpec) -> tuple[str, ...]:
    return tuple(dict.fromkeys([spec.df_checkout_path, *(project.project_root for project in spec.projects)]))


def _checkouts_needed(reference: EnvironmentFacts, spec: HostSpec) -> str:
    return (
        f'{", ".join(_checkout_roots(spec))} exist with receive.denyCurrentBranch={UPDATE_INSTEAD}; '
        'the df venv has its orchestrator'
    )


def _checkouts(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Finding:
    roots = _checkout_roots(spec)
    problems = []
    for root in roots:
        facts = host.checkouts.get(root)
        if facts is None or not facts.exists:
            problems.append(f'{root} absent')
        elif facts.receive_deny_current_branch != UPDATE_INSTEAD:
            problems.append(f'{root} receive.denyCurrentBranch={facts.receive_deny_current_branch}')
    if not host.df_orchestrator_executable:
        problems.append(f'{spec.df_checkout_path}/.venv/bin/orchestrator not executable')
    return _judged(problems, f'{len(roots)} checkouts accept pushes to their branch')


def _wrapper_needed(reference: EnvironmentFacts, spec: HostSpec) -> str:
    return (
        f'{spec.wrapper_path} equals a fresh render; '
        f'{_WRAPPER_PROBE} prints {verify_host_observe.WRAPPER_PROBE_ARG!r}'
    )


def _wrapper(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Finding:
    problems = []
    if host.wrapper_text is None:
        problems.append(f'{spec.wrapper_path} absent')
    elif host.wrapper_text != render_wrapper(spec):
        problems.append(f'{spec.wrapper_path} differs from a fresh render')
    probe = (host.wrapper_probe_stdout, host.wrapper_probe_rc)
    if probe != (verify_host_observe.WRAPPER_PROBE_ARG, 0):
        problems.append(f'{_WRAPPER_PROBE} printed {probe[0]!r} and exited {probe[1]}')
    return _judged(problems, f'{spec.wrapper_path} as rendered; honours ORCH_GATE_BIN')


def _config_paths(spec: HostSpec) -> str:
    return ', '.join(project.config_path for project in spec.projects)


def _per_host_yaml_needed(reference: EnvironmentFacts, spec: HostSpec) -> str:
    return f'{_config_paths(spec)} equal fresh renders of the roster host_config'


def _per_host_yaml(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Finding:
    problems = []
    for project in spec.projects:
        text = host.configs.get(project.config_path)
        if text is None:
            problems.append(f'project {project.project_id!r}: {project.config_path} missing')
        elif text != render_host_config(project.host_config):
            problems.append(f'project {project.project_id!r}: {project.config_path} differs from a fresh render')
    return _judged(problems, f'{_config_paths(spec)} as rendered')


def _no_credentials_needed(reference: EnvironmentFacts, spec: HostSpec) -> str:
    return f'no {_CREDENTIAL_IN_HOME} for the verify user'


def _no_credentials(host: HostFacts, reference: EnvironmentFacts, spec: HostSpec) -> Finding:
    problems = [f'{_CREDENTIAL_IN_HOME} present for {host.user}'] if host.credentials_present else []
    return _judged(problems, f'no {_CREDENTIAL_IN_HOME}')


REQUIREMENTS = (
    Requirement('network', 'Network', _network_needed, _network),
    Requirement('os_and_user', 'OS and user', _os_and_user_needed, _on_host_facts(_os_and_user)),
    Requirement('time_zone_and_locale', 'Time zone and locale', _time_zone_and_locale_needed,
                _on_host_facts(_time_zone_and_locale)),
    Requirement('toolchain', 'Toolchain', _toolchain_needed, _on_host_facts(_toolchain)),
    Requirement('systemd', 'systemd', _systemd_needed, _on_host_facts(_systemd)),
    Requirement('kernel', 'Kernel', _kernel_needed, _on_host_facts(_kernel)),
    Requirement('memory', 'Memory', _memory_needed, _on_host_facts(_memory)),
    Requirement('disk', 'Disk', _disk_needed, _on_host_facts(_disk)),
    Requirement('checkouts', 'Checkouts', _checkouts_needed, _on_host_facts(_checkouts)),
    Requirement('wrapper', 'Wrapper', _wrapper_needed, _on_host_facts(_wrapper)),
    Requirement('per_host_yaml', 'Per-host YAML', _per_host_yaml_needed, _on_host_facts(_per_host_yaml)),
    Requirement('no_credentials', 'No credentials', _no_credentials_needed, _on_host_facts(_no_credentials)),
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
    requirements = commands.add_parser('requirements', help='print the requirement ids, one per line, in order')
    requirements.add_argument('--titles', action='store_true', help='print id<TAB>title per line')
    render = commands.add_parser('render', help="write a host's per-project YAML, wrapper.sh and host.env")
    render.add_argument('host')
    render.add_argument('--roster', type=Path, help='roster file (default: the ORCH_VERIFY_HOSTS_FILE resolution)')
    render.add_argument('--out', type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == 'requirements':
        print('\n'.join(f'{row.id}\t{row.title}' if args.titles else row.id for row in REQUIREMENTS))
        return 0
    try:
        spec = HostSpec.from_roster(
            load_roster(args.roster or roster_file_path(os.environ)), args.host,
            os.environ.get(WRAPPER_PATH_ENV, DEFAULT_WRAPPER_PATH),
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
