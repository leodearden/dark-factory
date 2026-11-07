"""What this host (or container) really lets the benchmark use.

    python3 -I container_limits.py            one JSON object
    python3 -I container_limits.py <field>    just that field, for shell callers

The CPU budget is the smallest of the CPUs the process may run on (affinity,
which a cpuset limit shapes) and the cgroup CPU quota rounded up (v2 cpu.max or
v1 cfs, own cgroup and ancestors). `nproc` is the affinity count, so it is
covered; `lscpu` and os.cpu_count() describe the whole host and are not used
as the budget.

`cores` is the middle K level of the probes:
  host-physical-cores     nothing limits us: distinct (socket, core) pairs of the host
  cpuset-distinct-cores   a cpuset limits us: distinct (socket, core) pairs among the
                          CPUs we may run on, which counts SMT sibling pairs exactly
  half-budget             a quota limits us and we cannot know which cores we get:
                          budget // 2 (at least 1)
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

CGROUP_ROOT = Path('/sys/fs/cgroup')
V1_CPU_CONTROLLERS = ('cpu', 'cpu,cpuacct')
V1_MEM_CONTROLLERS = ('memory',)
HUGE_V1_LIMIT = 1 << 60


def read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def cgroup_dirs(v1_controllers: tuple[str, ...]) -> list[Path]:
    """Own cgroup directory first, then its ancestors, up to each hierarchy's mount root."""
    own = Path('/')
    for line in (read(Path('/proc/self/cgroup')) or '').splitlines():
        _, controllers, path = line.split(':', 2)
        if controllers == '' or set(controllers.split(',')) & set(v1_controllers):
            own = Path(path)
    dirs: list[Path] = []
    for base in [CGROUP_ROOT, *(CGROUP_ROOT / name for name in v1_controllers)]:
        candidate = base / own.relative_to('/')
        while True:
            if candidate.is_dir() and candidate not in dirs:
                dirs.append(candidate)
            if candidate == base:
                break
            candidate = candidate.parent
    return dirs


def quota_cpus() -> tuple[float | None, str | None, str | None]:
    """(cpus, source, raw) of the tightest cgroup CPU quota, or (None, None, None)."""
    best: tuple[float, str, str] | None = None
    for d in cgroup_dirs(V1_CPU_CONTROLLERS):
        found = None
        raw = read(d / 'cpu.max')
        if raw:
            quota, period = raw.split()
            if quota != 'max':
                found = (int(quota) / int(period), 'cgroup-v2-cpu.max', raw)
        else:
            quota, period = read(d / 'cpu.cfs_quota_us'), read(d / 'cpu.cfs_period_us')
            if quota and period and int(quota) > 0:
                found = (int(quota) / int(period), 'cgroup-v1-cfs', f'{quota}/{period}')
        if found and (best is None or found[0] < best[0]):
            best = found
    return best if best else (None, None, None)


def host_cpu_topology() -> dict[int, tuple[int, int]]:
    """cpu -> (socket, core) for every CPU lscpu lists."""
    out = subprocess.run(['lscpu', '-p=CPU,CORE,SOCKET'], capture_output=True, text=True, check=True).stdout
    topo = {}
    for line in out.splitlines():
        if not line.startswith('#'):
            cpu, core, socket = line.split(',')[:3]
            topo[int(cpu)] = (int(socket), int(core))
    return topo


def cpu_ranges(cpus: set[int]) -> str:
    out, ordered = [], sorted(cpus)
    while ordered:
        start = end = ordered.pop(0)
        while ordered and ordered[0] == end + 1:
            end = ordered.pop(0)
        out.append(str(start) if start == end else f'{start}-{end}')
    return ','.join(out)


def cpu_limits() -> dict:
    host_threads = os.cpu_count() or 1
    allowed = set(os.sched_getaffinity(0))
    quota, quota_source, quota_raw = quota_cpus()
    quota_budget = math.ceil(quota) if quota is not None else None
    topo = host_cpu_topology()

    if quota_budget is not None and quota_budget < min(len(allowed), host_threads):
        budget, source = quota_budget, quota_source
        cores, rule = max(1, budget // 2), 'half-budget'
    elif len(allowed) < host_threads:
        budget, source = len(allowed), 'cpuset'
        cores, rule = len({topo[c] for c in allowed}), 'cpuset-distinct-cores'
    else:
        budget, source = host_threads, 'host'
        cores, rule = len(set(topo.values())), 'host-physical-cores'
    return {
        'host_threads': host_threads,
        'cpu_budget': budget,
        'cpu_budget_source': source,
        'cores': cores,
        'cores_rule': rule,
        'allowed_cpus': cpu_ranges(allowed),
        'cgroup_cpu_quota': quota_raw,
    }


def memory_limits() -> dict:
    """The cgroup memory ceiling (kB, None when unlimited) and what is left under it."""
    meminfo = {line.split(':')[0]: int(line.split()[1]) for line in (read(Path('/proc/meminfo')) or '').splitlines()}
    available = meminfo['MemAvailable']
    ceiling = None
    for d in cgroup_dirs(V1_MEM_CONTROLLERS):
        raw, used = read(d / 'memory.max'), read(d / 'memory.current')
        if raw is None:
            raw, used = read(d / 'memory.limit_in_bytes'), read(d / 'memory.usage_in_bytes')
        if raw is None or raw == 'max' or int(raw) >= HUGE_V1_LIMIT or used is None:
            continue
        limit_kb = int(raw) // 1024
        if ceiling is None or limit_kb < ceiling:
            ceiling = limit_kb
        available = min(available, limit_kb - (int(used) - reclaimable_bytes(d)) // 1024)
    return {'mem_limit_kb': ceiling, 'mem_available_kb': available}


def reclaimable_bytes(cgroup_dir: Path) -> int:
    """Page cache the cgroup's usage counts but the kernel reclaims on demand: after a
    few gates it is most of memory.current, and is not memory a probe copy competes for."""
    stat = read(cgroup_dir / 'memory.stat') or ''
    fields = dict(line.split() for line in stat.splitlines() if len(line.split()) == 2)
    return int(fields.get('file', fields.get('total_cache', 0)))


if __name__ == '__main__':
    facts = {**cpu_limits(), **memory_limits()}
    print(json.dumps(facts) if len(sys.argv) == 1 else facts[sys.argv[1]])
