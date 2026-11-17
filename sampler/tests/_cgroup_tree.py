"""A cgroupfs-shaped directory tree under ``tmp_path`` for the consumer tests.

Each cgroup is a real directory holding the counter files the recorder reads,
in the kernel's text formats. Hierarchical sums are NOT derived: a test sets
each cgroup's values itself, as the kernel would report them. The root lacks
``memory.current``, ``memory.swap.current`` and ``memory.events``, as on a
live host.
"""

from __future__ import annotations

import dataclasses
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT_ABSENT = ('memory.current', 'memory.swap.current', 'memory.events')
MEMORY_FILES = (
    'memory.stat',
    'memory.current',
    'memory.swap.current',
    'memory.events',
    'memory.pressure',
)


@dataclass(frozen=True)
class Counters:
    cpu_usec: int = 0
    memory: int = 0
    anon: int = 0
    file: int = 0
    kernel: int = 0
    swap: int = 0
    pswpin: int = 0
    pgsteal_kswapd: int = 0
    pgsteal_direct: int = 0
    refault_anon: int = 0
    refault_file: int = 0
    high: int = 0
    max: int = 0
    oom_kill: int = 0
    cpu_some: int = 0
    memory_some: int = 0
    memory_full: int = 0
    io_some: int = 0


def _files(c: Counters) -> dict[str, str]:
    return {
        'cpu.stat': f'usage_usec {c.cpu_usec}\nuser_usec {c.cpu_usec}\nsystem_usec 0\n',
        'memory.stat': (
            f'anon {c.anon}\nfile {c.file}\nkernel {c.kernel}\nshmem 0\n'
            f'workingset_refault_anon {c.refault_anon}\n'
            f'workingset_refault_file {c.refault_file}\n'
            f'pswpin {c.pswpin}\npswpout 0\n'
            f'pgsteal_kswapd {c.pgsteal_kswapd}\npgsteal_direct {c.pgsteal_direct}\n'
        ),
        'memory.current': f'{c.memory}\n',
        'memory.swap.current': f'{c.swap}\n',
        'memory.events': (
            f'low 0\nhigh {c.high}\nmax {c.max}\noom 0\noom_kill {c.oom_kill}\n'
            'oom_group_kill 0\n'
        ),
        'cpu.pressure': _pressure(c.cpu_some, 0),
        'memory.pressure': _pressure(c.memory_some, c.memory_full),
        'io.pressure': _pressure(c.io_some, 0),
        'cgroup.procs': '',
    }


def _pressure(some: int, full: int) -> str:
    return (
        f'some avg10=0.00 avg60=0.00 avg300=0.00 total={some}\n'
        f'full avg10=0.00 avg60=0.00 avg300=0.00 total={full}\n'
    )


class CgroupTree:
    """Paths are relative to the root, '/'-separated; the root itself is ''."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._counters: dict[str, Counters] = {}
        self._omitted: dict[str, frozenset[str]] = {}
        self.set('', omit=ROOT_ABSENT)

    def path(self, rel: str) -> Path:
        return self.root / rel if rel else self.root

    def set(self, rel: str, *, omit: Iterable[str] | None = None, **counters: int) -> None:
        """Create the cgroup (its parent must exist) or update it; unnamed counters keep their value."""
        directory = self.path(rel)
        if rel not in self._counters:
            directory.mkdir()
            self._counters[rel] = Counters()
            self._omitted[rel] = frozenset()
        self._counters[rel] = dataclasses.replace(self._counters[rel], **counters)
        if omit is not None:
            self._omitted[rel] = frozenset(omit)
        self._write(rel, directory)

    def remove(self, rel: str) -> None:
        shutil.rmtree(self.path(rel))
        for known in [k for k in self._counters if k == rel or k.startswith(f'{rel}/')]:
            del self._counters[known]
            del self._omitted[known]

    def make_unreadable(self, rel: str, file_name: str) -> None:
        """Replace one listed file with a directory, so opening it fails even as root."""
        target = self.path(rel) / file_name
        target.unlink()
        target.mkdir()

    def make_readable(self, rel: str, file_name: str) -> None:
        """Undo ``make_unreadable``; ``set`` never touches an unreadable file."""
        (self.path(rel) / file_name).rmdir()
        self._write(rel, self.path(rel))

    def inode(self, rel: str) -> int:
        return self.path(rel).stat().st_ino

    def recreate(self, rel: str, **counters: int) -> None:
        """Replace a leaf cgroup with a new directory, so it has a NEW inode."""
        old = self.path(rel)
        assert not any(p.is_dir() for p in old.iterdir()), f'{rel} is not a leaf'
        replacement = old.with_name(f'{old.name}.recreating')
        replacement.mkdir()
        self._counters[rel] = dataclasses.replace(Counters(), **counters)
        self._write(rel, replacement)
        shutil.rmtree(old)
        replacement.rename(old)

    def _write(self, rel: str, directory: Path) -> None:
        for name, text in _files(self._counters[rel]).items():
            target = directory / name
            if name in self._omitted[rel]:
                if target.is_file():
                    target.unlink()
                continue
            if not target.is_dir():
                target.write_text(text)
