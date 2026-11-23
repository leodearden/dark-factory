#!/usr/bin/env python3
"""One snapshot of the measures a step is bracketed by. Prints JSON.

cpu_usec     cumulative CPU of this experiment's own cgroup (cargo, rustc, the private
             sccache server and the compilers it spawns).
sccache      the PRIVATE server's stats (SCCACHE_SERVER_UDS from the environment).
procs        compiler processes on the host, split into this cgroup and everything else.
"""
import json
import os
import subprocess
import time

cg = os.environ.get('WARMEXP_CG', '')
rel = cg[len('/sys/fs/cgroup'):] if cg.startswith('/sys/fs/cgroup') else None


def cpu_usec():
    try:
        for line in open(os.path.join(cg, 'cpu.stat')):
            k, v = line.split()
            if k == 'usage_usec':
                return int(v)
    except OSError:
        return None


def procs():
    mine = other = 0
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            comm = open(f'/proc/{pid}/comm').read().strip()
            if comm not in ('rustc', 'cargo', 'clippy-driver', 'cargo-nextest'):
                continue
            path = open(f'/proc/{pid}/cgroup').read().strip().split(':', 2)[-1]
        except OSError:
            continue
        if rel and path == rel:
            mine += 1
        else:
            other += 1
    return {'mine': mine, 'other': other}


def scopes():
    try:
        out = subprocess.run(['systemctl', '--user', 'list-units', '--type=scope', '--no-pager', '--plain', '--no-legend'],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None
    names = [l.split()[0] for l in out.splitlines() if l.startswith('df-verify-')]
    return {'reify': sum('df-verify-reify' in n for n in names), 'other': sum('df-verify-reify' not in n for n in names)}


def sccache():
    try:
        out = subprocess.run(['sccache', '--show-stats', '--stats-format', 'json'], capture_output=True, text=True, timeout=30).stdout
        return json.loads(out)
    except Exception as e:
        return {'error': str(e)}


def own(name):
    try:
        return open(os.path.join(cg, name)).read().strip()
    except OSError:
        return None


def gate_waiters():
    """Production verifies currently held at verify.sh's psi or compile gate."""
    n = 0
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f'/proc/{pid}/cmdline').read().replace('\0', ' ')
        except OSError:
            continue
        if 'verify.sh psi-gate' in cmd or 'verify.sh compile-gate' in cmd:
            n += 1
    return n


def psi():
    try:
        return open('/proc/pressure/cpu').readline().strip()
    except OSError:
        return None


print(json.dumps({
    'time': time.time(),
    'cpu_usec': cpu_usec(),
    'loadavg': open('/proc/loadavg').read().split()[:3],
    'psi_cpu': psi(),
    'procs': procs(),
    'verify_scopes': scopes(),
    'gate_waiters': gate_waiters(),
    'own_cpu_pressure': own('cpu.pressure'),
    'own_memory_peak': own('memory.peak'),
    'sccache': sccache(),
}))
