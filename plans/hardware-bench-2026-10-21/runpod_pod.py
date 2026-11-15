"""Create, inspect and terminate the benchmark's RunPod CPU pods and volume.

Run inside the runpod-toolkit project (it holds the API client and key):

    uv run --project /home/leo/src/runpod-toolkit python runpod_pod.py <command> ...

    create --name N --dc DC [--flavor cpu3g --vcpu 32 --mem 128 --disk 200]
           [--volume ID] [--want REGEX --tries 4]
        Creates a pod, waits for ssh, reads its CPU model, and redraws (terminate
        and create again) while the model does not match --want. Prints one JSON
        line and writes `Host <name>` to ssh_config beside this file. Any failure
        terminates the pod it was handling.
    terminate POD_ID
    volume --name N --dc DC --size GB     create a network volume, print its JSON
    delete-volume VOLUME_ID               delete a network volume (no pod may use it)
    status                                balance, spend rate, pods, volumes
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from runpod_toolkit.compute import RunPodClient
from runpod_toolkit.config import RunPodConfig

HERE = Path(__file__).resolve().parent
SSH_CONFIG = HERE / 'ssh_config'
KEY = Path.home() / '.ssh' / 'id_runpod'
IMAGE = 'runpod/base:1.0.2-ubuntu2404'
SSH_WAIT_SECS = 600


def ssh_argv(host: str, port: int) -> list[str]:
    return [
        'ssh', '-i', str(KEY), '-p', str(port), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null', '-o', 'LogLevel=ERROR',
        f'root@{host}',
    ]


def wait_for_ssh(client: RunPodClient, pod_id: str) -> tuple[str, int]:
    deadline = time.monotonic() + SSH_WAIT_SECS
    while time.monotonic() < deadline:
        pod = client.get_pod(pod_id)
        if pod.ssh_host and pod.ssh_port:
            probe = subprocess.run([*ssh_argv(pod.ssh_host, pod.ssh_port), 'true'], capture_output=True)
            if probe.returncode == 0:
                return pod.ssh_host, pod.ssh_port
        time.sleep(10)
    raise TimeoutError(f'pod {pod_id}: no ssh within {SSH_WAIT_SECS}s')


def cpu_model(host: str, port: int) -> str:
    out = subprocess.run([*ssh_argv(host, port), 'lscpu'], capture_output=True, text=True, check=True).stdout
    return next(line.split(':', 1)[1].strip() for line in out.splitlines() if line.startswith('Model name'))


def write_ssh_alias(alias: str, host: str, port: int) -> None:
    stanzas = SSH_CONFIG.read_text().split('\n\n') if SSH_CONFIG.exists() else []
    stanzas = [s for s in stanzas if s.strip() and not s.startswith(f'Host {alias}\n')]
    stanzas.append(
        f'Host {alias}\n  HostName {host}\n  Port {port}\n  User root\n  IdentityFile {KEY}\n'
        '  IdentitiesOnly yes\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n  LogLevel ERROR'
    )
    SSH_CONFIG.write_text('\n\n'.join(stanzas) + '\n')


def create(client: RunPodClient, args: argparse.Namespace) -> None:
    pubkey = (KEY.parent / (KEY.name + '.pub')).read_text().strip()
    want = re.compile(args.want) if args.want else None
    for attempt in range(1, args.tries + 1):
        pod = client.create_cpu_pod(
            name=args.name, image=IMAGE, datacenter=args.dc, vcpu_count=args.vcpu, memory_gb=args.mem,
            cpu_flavor=args.flavor, container_disk_gb=args.disk, network_volume_id=args.volume,
            env_vars={'PUBLIC_KEY': pubkey},
        )
        try:
            host, port = wait_for_ssh(client, pod.id)
            model = cpu_model(host, port)
        except BaseException:
            client.terminate_pod(pod.id)
            raise
        if want and not want.search(model):
            print(f'attempt {attempt}: {model} does not match {args.want}; terminating {pod.id}', file=sys.stderr)
            client.terminate_pod(pod.id)
            continue
        write_ssh_alias(args.name, host, port)
        print(json.dumps({
            'id': pod.id, 'name': args.name, 'dc': args.dc, 'ssh_host': host, 'ssh_port': port,
            'cpu_model': model, 'cost_per_hr': client.get_pod(pod.id).cost_per_hour, 'attempts': attempt,
        }))
        return
    sys.exit(f'no pod matching {args.want} in {args.tries} attempts')


def graphql(config: RunPodConfig, query: str) -> dict:
    response = requests.post(
        'https://api.runpod.io/graphql', json={'query': query},
        headers={'Authorization': f'Bearer {config.api_key}'}, timeout=30,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('create')
    c.add_argument('--name', required=True)
    c.add_argument('--dc', required=True)
    c.add_argument('--flavor', default='cpu3g')
    c.add_argument('--vcpu', type=int, default=32)
    c.add_argument('--mem', type=int, default=128)
    c.add_argument('--disk', type=int, default=200)
    c.add_argument('--volume')
    c.add_argument('--want')
    c.add_argument('--tries', type=int, default=1)
    t = sub.add_parser('terminate')
    t.add_argument('pod_id')
    v = sub.add_parser('volume')
    v.add_argument('--name', required=True)
    v.add_argument('--dc', required=True)
    v.add_argument('--size', type=int, required=True)
    d = sub.add_parser('delete-volume')
    d.add_argument('volume_id')
    sub.add_parser('status')
    args = parser.parse_args()

    config = RunPodConfig.from_env()
    client = RunPodClient(config)
    if args.command == 'create':
        create(client, args)
    elif args.command == 'terminate':
        print(json.dumps({'terminated': client.terminate_pod(args.pod_id), 'id': args.pod_id}))
    elif args.command == 'volume':
        volume = client.create_volume(args.name, args.size, args.dc)
        print(json.dumps({'id': volume.id, 'name': volume.name, 'size_gb': volume.size_gb, 'dc': volume.datacenter}))
    elif args.command == 'delete-volume':
        print(json.dumps({'deleted': client.delete_volume(args.volume_id), 'id': args.volume_id}))
    else:
        print(json.dumps(graphql(config, (
            '{ myself { clientBalance currentSpendPerHr '
            'pods { id name desiredStatus costPerHr } networkVolumes { id name size dataCenterId } } }'
        ))['data']['myself']))


if __name__ == '__main__':
    main()
