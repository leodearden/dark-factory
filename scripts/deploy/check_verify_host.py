#!/usr/bin/env python3
"""Check a remote verify host against the verify-host contract (PRD §6.1).

Observes the workstation and, over the dispatcher's own ssh options, the host,
then prints one row per verify_host_contract.REQUIREMENTS entry: requirement,
verdict (ok | fail | warn), observed, needed.

Usage: check_verify_host.py HOST [--roster FILE] [--json]
Exit 0 iff no row fails; 1 when one does; 2 on a usage or roster error.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

# Bare sys.path calls, the one statement ruff's E402 lets an import follow.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _checkout as _checkout
import verify_host_contract as contract
from orchestrator.verify_host_roster_file import RosterFileError, load_roster, roster_file_path
from orchestrator.verify_runner import REMOTE_TOOL_PATH_PRELUDE
from verify_host_observe import EnvironmentFacts, HostFacts

OBSERVER = Path(__file__).with_name('verify_host_observe.py')
REFERENCE_TIMEOUT_SECS = 300
NETWORK_TIMEOUT_SECS = 60
OBSERVE_TIMEOUT_SECS = 900
DETAIL_TAIL_CHARS = 400


class ReferenceUnobserved(RuntimeError):
    """This workstation's own environment facts, the reference every host is held to, are unreadable."""


def _run(argv: list[str], timeout: float, stdin: bytes = b'') -> tuple[int | None, str, str]:
    """(returncode or None when it could not run, stdout, a short detail for a failure)."""
    try:
        proc = subprocess.run(argv, input=stdin, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return None, '', f'timed out after {timeout:g}s'
    except OSError as exc:
        return None, '', str(exc)
    stderr = proc.stderr.decode(errors='replace').strip()
    return proc.returncode, proc.stdout.decode(errors='replace'), f'exit {proc.returncode}: {stderr[-DETAIL_TAIL_CHARS:]}'


def _reference() -> EnvironmentFacts:
    returncode, stdout, detail = _run(
        [sys.executable, '-I', str(OBSERVER), '--environment-only'], REFERENCE_TIMEOUT_SECS,
    )
    if returncode != 0:
        raise ReferenceUnobserved(f'observing this workstation failed: {detail}')
    return EnvironmentFacts.from_json(stdout)


def _network(spec: contract.HostSpec, ssh: list[str]) -> contract.NetworkFacts:
    reached, _, detail = _run([*ssh, spec.ssh_host, 'true'], NETWORK_TIMEOUT_SECS)
    if reached != 0:
        return contract.NetworkFacts(reachable=False, host_key_pinned=False, detail=detail)
    pinned, _, detail = _run([*ssh, '-o', 'StrictHostKeyChecking=yes', spec.ssh_host, 'true'], NETWORK_TIMEOUT_SECS)
    return contract.NetworkFacts(reachable=True, host_key_pinned=pinned == 0, detail='' if pinned == 0 else detail)


def _observer_args(spec: contract.HostSpec) -> list[str]:
    args = ['--df-checkout', spec.df_checkout_path]
    for project in spec.projects:
        args += ['--project-checkout', project.project_root, '--config-path', project.config_path]
    return [*args, '--wrapper-path', spec.wrapper_path, '--disk-path', spec.df_checkout_path]


def _host_facts(spec: contract.HostSpec, ssh: list[str]) -> tuple[HostFacts | None, str]:
    remote = f'{REMOTE_TOOL_PATH_PRELUDE} python3 -I - {shlex.join(_observer_args(spec))}'
    returncode, stdout, detail = _run([*ssh, spec.ssh_host, remote], OBSERVE_TIMEOUT_SECS, OBSERVER.read_bytes())
    if returncode != 0:
        return None, f'the observer failed on the host ({detail})'
    try:
        return HostFacts.from_json(stdout), ''
    except ValueError as exc:
        return None, f'the observer printed unreadable facts ({exc})'


def observe(spec: contract.HostSpec, ssh_bin: str) -> contract.Observation:
    ssh = [ssh_bin, *contract.DISPATCHER_SSH_OPTS]
    network = _network(spec, ssh)
    if not network.reachable:
        return contract.Observation(network=network, host=None, unobserved_reason='the host is unreachable')
    host, reason = _host_facts(spec, ssh)
    return contract.Observation(network=network, host=host, unobserved_reason=reason)


def evaluate(spec: contract.HostSpec, observation: contract.Observation, reference: EnvironmentFacts) -> list[dict[str, str]]:
    return [
        {'requirement': row.id, **asdict(row.evaluate(observation, reference, spec))}
        for row in contract.REQUIREMENTS
    ]


def _table(rows: list[dict[str, str]]) -> str:
    width = max(len(row['requirement']) for row in rows)
    lines = [f'{"requirement":<{width}}  verdict  observed  [needed]']
    lines += [
        f'{row["requirement"]:<{width}}  {row["verdict"]:<7}  {row["observed"]}  [{row["needed"]}]' for row in rows
    ]
    return '\n'.join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('host')
    parser.add_argument('--roster', type=Path, help='roster file (default: the ORCH_VERIFY_HOSTS_FILE resolution)')
    parser.add_argument('--json', action='store_true')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        spec = contract.HostSpec.from_roster(
            load_roster(args.roster or roster_file_path(os.environ)), args.host,
            os.environ.get(contract.WRAPPER_PATH_ENV, contract.DEFAULT_WRAPPER_PATH),
        )
    except (contract.ContractError, RosterFileError) as exc:
        print(f'check_verify_host: {exc}', file=sys.stderr)
        return 2
    try:
        reference = _reference()
    except ReferenceUnobserved as exc:
        print(f'check_verify_host: {exc}', file=sys.stderr)
        return 2
    ssh_bin = os.environ.get(contract.SSH_BIN_ENV, contract.DEFAULT_SSH_BIN)
    rows = evaluate(spec, observe(spec, ssh_bin), reference)
    print(json.dumps({'host': spec.name, 'rows': rows}, indent=2) if args.json else _table(rows))
    return 1 if any(row['verdict'] == 'fail' for row in rows) else 0


if __name__ == '__main__':
    sys.exit(main())
