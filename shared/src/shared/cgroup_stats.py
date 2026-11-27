"""The one parser for cgroup v2 counter files, and a reader for one cgroup.

PRD ``plans/per-consumer-resource-recording-prd.md`` D7: ``cpu.stat``,
``memory.stat`` and ``memory.events`` are key-value lines with one parser
here, and the pressure files' totals come through
``shared.psi.parse_pressure_totals``, so no second pressure parser exists.

``read_cgroup`` decides readability per file against the cgroup's own
directory listing. A file absent from the listing is not exposed there (the
root has no ``memory.current``; a cgroup without the memory controller has no
``memory.*``) and reads as ``None``. A listed file that fails to read or
parse makes the whole cgroup ``Unreadable``, naming the file: absence must
never read as zero use, and a partial reading is never returned.

A direct-import submodule, not re-exported from ``shared/__init__.py``, and
stdlib-only: it is a member of
``shared/tests/test_pure_stdlib_leaves.py::PURE_STDLIB_LEAVES``.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TypeVar

from shared.psi import parse_pressure_totals

__all__ = [
    'CGROUP_ROOT',
    'CPU_STAT',
    'MEMORY_CURRENT',
    'MEMORY_EVENTS',
    'MEMORY_PEAK',
    'MEMORY_STAT',
    'SWAP_CURRENT',
    'SWAP_PEAK',
    'CgroupReading',
    'PressureResource',
    'PressureTotals',
    'Unreadable',
    'parse_keyed_counters',
    'parse_single_value',
    'read_cgroup',
]

CGROUP_ROOT = Path('/sys/fs/cgroup')

CPU_STAT = 'cpu.stat'
MEMORY_STAT = 'memory.stat'
MEMORY_EVENTS = 'memory.events'
MEMORY_CURRENT = 'memory.current'
SWAP_CURRENT = 'memory.swap.current'
MEMORY_PEAK = 'memory.peak'
SWAP_PEAK = 'memory.swap.peak'

_T = TypeVar('_T')


class PressureResource(StrEnum):
    CPU = 'cpu'
    MEMORY = 'memory'
    IO = 'io'

    @property
    def file_name(self) -> str:
        return f'{self.value}.pressure'


@dataclass(frozen=True)
class PressureTotals:
    """Cumulative stall of one pressure file, in microseconds."""

    some_usec: int
    full_usec: int


@dataclass(frozen=True)
class CgroupReading:
    """Every recorded file of one cgroup; ``None`` = not exposed in its listing."""

    cpu_stat: Mapping[str, int] | None
    memory_stat: Mapping[str, int] | None
    memory_events: Mapping[str, int] | None
    memory_current: int | None
    swap_current: int | None
    pressure: Mapping[PressureResource, PressureTotals]
    memory_peak: int | None
    swap_peak: int | None


@dataclass(frozen=True)
class Unreadable:
    """A listed file of the cgroup could not be read or parsed."""

    file: str
    reason: str


def parse_keyed_counters(text: str) -> dict[str, int]:
    """Parse ``<key> <integer>`` lines; a malformed line raises naming it."""
    counters: dict[str, int] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2 or not fields[1].isdigit():
            raise ValueError(f'malformed cgroup counter line: {line!r}')
        counters[fields[0]] = int(fields[1])
    return counters


def parse_single_value(text: str) -> int:
    """Parse a one-integer file such as ``memory.current``."""
    stripped = text.strip()
    if not stripped.isdigit():
        raise ValueError(f'not a single cgroup value: {text!r}')
    return int(stripped)


def _parse_pressure(text: str) -> PressureTotals:
    totals = parse_pressure_totals(text)
    if totals is None:
        raise ValueError(f'no pressure total in {text!r}')
    return PressureTotals(totals['some_total'], totals['full_total'])


class _UnreadableFile(Exception):
    def __init__(self, unreadable: Unreadable) -> None:
        super().__init__(unreadable.file)
        self.unreadable = unreadable


class _ListedFiles:
    """Reads the files of one cgroup that its directory listing exposes."""

    def __init__(self, directory: Path, present: Collection[str]) -> None:
        self._directory = directory
        self._present = present

    def parse(self, name: str, parser: Callable[[str], _T]) -> _T | None:
        if name not in self._present:
            return None
        try:
            return parser((self._directory / name).read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise _UnreadableFile(
                Unreadable(file=name, reason=f'{type(exc).__name__}: {exc}')
            ) from exc

    def counters(self, name: str) -> Mapping[str, int] | None:
        parsed = self.parse(name, parse_keyed_counters)
        return None if parsed is None else MappingProxyType(parsed)

    def pressure(self) -> Mapping[PressureResource, PressureTotals]:
        totals = {
            resource: self.parse(resource.file_name, _parse_pressure)
            for resource in PressureResource
        }
        return MappingProxyType(
            {resource: value for resource, value in totals.items() if value is not None}
        )



def read_cgroup(
    directory: str | Path, *, present: Collection[str]
) -> CgroupReading | Unreadable:
    """Read one cgroup directory; ``present`` is the file names its listing holds."""
    files = _ListedFiles(Path(directory), present)
    try:
        return CgroupReading(
            cpu_stat=files.counters(CPU_STAT),
            memory_stat=files.counters(MEMORY_STAT),
            memory_events=files.counters(MEMORY_EVENTS),
            memory_current=files.parse(MEMORY_CURRENT, parse_single_value),
            swap_current=files.parse(SWAP_CURRENT, parse_single_value),
            pressure=files.pressure(),
            memory_peak=files.parse(MEMORY_PEAK, parse_single_value),
            swap_peak=files.parse(SWAP_PEAK, parse_single_value),
        )
    except _UnreadableFile as failure:
        return failure.unreadable
