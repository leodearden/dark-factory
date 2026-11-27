"""A host for the consumer and verify-scope collectors, under ``tmp_path``.

A cgroup tree (``_cgroup_tree.CgroupTree``) shaped on the live one, with a
verify scope in each of its two placements; a /proc holding one kswapd and one
kcompactd thread; and a meminfo file. The collectors are bound to these
paths, so no test reads the live host.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from _cgroup_tree import CgroupTree

USER = 'user.slice/user-1000.slice/user@1000.service'
APP = f'{USER}/app.slice'
SERVICE = f'{APP}/orchestrator-reify.service'
DAEMON = f'{SERVICE}/daemon'
APP_SCOPE = f'{APP}/df-verify-dark-factory-c889f090-5ea08a53433a.scope'
DF = f'{USER}/df.slice'
PROJECT = f'{DF}/df-x.slice'
DF_SCOPE = f'{PROJECT}/df-verify-x-1234abcd-0123456789ab.scope'

KSWAPD_PID = 101
KCOMPACTD_PID = 102
MEM_TOTAL_KIB = 1_000
MEM_AVAILABLE_KIB = 400


def _stat_line(pid: int, comm: str, utime: int) -> str:
    """A real-format /proc/<pid>/stat line: comm in parentheses, then fields 3-52."""
    fields = ['S', '2', '0', '0', '0', '-1', '69238880', '0', '0', '0', '0']
    fields += [str(utime), '0', '0', '0', '20', '0', '1', '0', '79']
    fields += ['0'] * 32
    return f'{pid} ({comm}) {" ".join(fields)}\n'


@dataclass(frozen=True)
class ConsumerWorld:
    cgroups: CgroupTree
    proc_root: Path
    meminfo: Path

    def thread(self, pid: int, comm: str, ticks: int) -> None:
        directory = self.proc_root / str(pid)
        directory.mkdir(exist_ok=True)
        (directory / 'stat').write_text(_stat_line(pid, comm, ticks))

    def collect_consumers(self, now: int, previous: str | None):
        from sampler.metrics import collect_consumers

        return collect_consumers(
            now,
            previous,
            cgroup_root=self.cgroups.root,
            proc_root=self.proc_root,
            meminfo_path=self.meminfo,
        )

    def collect_verify_scopes(self, now: int, previous: str | None):
        from sampler.metrics import collect_verify_scopes

        return collect_verify_scopes(now, previous, cgroup_root=self.cgroups.root)


def consumer_world(tmp_path: Path) -> ConsumerWorld:
    """The fixture host: an orchestrator service, a project slice, and both verify placements."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    cgroups = CgroupTree(tmp_path / 'cgroup')
    cgroups.set('', anon=7_000, file=3_000, kernel=500, cpu_usec=90_000)
    for inner in ('user.slice', 'user.slice/user-1000.slice', USER, APP, SERVICE, DF, PROJECT):
        cgroups.set(inner)
    cgroups.set(DAEMON, memory=1_000, anon=600, file=400, swap=10, cpu_usec=1_000)
    cgroups.set(APP_SCOPE, memory=2_000, anon=1_500, file=500, cpu_usec=2_000)
    cgroups.set(DF_SCOPE, memory=3_000, anon=2_000, file=1_000, cpu_usec=3_000)

    proc_root = tmp_path / 'proc'
    proc_root.mkdir()
    (proc_root / 'self').mkdir()
    meminfo = tmp_path / 'meminfo'
    meminfo.write_text(
        f'MemTotal:       {MEM_TOTAL_KIB} kB\nMemFree:        100 kB\n'
        f'MemAvailable:   {MEM_AVAILABLE_KIB} kB\n'
    )
    world = ConsumerWorld(cgroups, proc_root, meminfo)
    world.thread(KSWAPD_PID, 'kswapd0', 10)
    world.thread(KCOMPACTD_PID, 'kcompactd0', 20)
    return world
