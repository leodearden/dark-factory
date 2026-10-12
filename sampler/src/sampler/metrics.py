"""The sampler's metric groups: what each collects, what it is called, how it is windowed.

``METRIC_GROUPS``, at the bottom of this module, is the metric vocabulary:
every name, its group, its gate arm and its readability metric are declared
there once, and every other spelling is a projection of it. The collectors
above it are pure over their injected seams, with the live readers as
defaults.

FD-9 heuristic
--------------
``occt_queue_depth`` counts processes running ``cargo-test-occt-gated`` whose
``/proc/<pid>/fd/9`` symlink does NOT exist. FD 9 is the read end of the
semaphore pipe the OCCT gate script opens, so its absence means the process
is still waiting to acquire the semaphore. The heuristic breaks if the gate
script moves the pipe to another FD.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

from shared.psi import (
    RunqueueReading,
    parse_pressure_file,
    read_pressure,
    read_runqueue_ratio,
    resolve_own_cgroup,
)

logger = logging.getLogger(__name__)

__all__ = [
    'SampleKey',
    'Samples',
    'GroupName',
    'MetricSpec',
    'MetricGroup',
    'parse_pressure_file',
    'collect_psi',
    'ProcessRecord',
    'classify_process',
    'count_occt_queue_depth',
    'count_verify_concurrency',
    'sum_verify_rss',
    'collect_process_metrics',
    'discover_pressure_cgroups',
    'collect_load_metrics',
    'METRIC_GROUPS',
    'arm_metrics',
    'ARM_METRICS',
    'exact_metric_names',
    'TICK_CLOCK',
]

class SampleKey(NamedTuple):
    """One recorded series: a metric, and the cgroup leaf it is about.

    ``leaf`` is ``None`` for a host-level metric. A per-leaf metric is
    recorded once per discovered leaf, and the leaf is opaque data.
    """

    metric: str
    leaf: str | None = None

    def label(self) -> str:
        """Display rendering only, ``metric`` or ``metric:leaf``; nothing parses it."""
        return self.metric if self.leaf is None else f'{self.metric}:{self.leaf}'


Samples = Mapping[SampleKey, float]


class GroupName(StrEnum):
    """The collection groups: each fails, is admitted and is windowed as a unit."""

    PSI = 'psi'
    PROCESS = 'process'
    LOAD = 'load'


@dataclass(frozen=True)
class MetricSpec:
    """One recorded metric, and the facts every projection of the table reads.

    A ``per_leaf`` metric is recorded once per discovered cgroup leaf, with
    the leaf in ``SampleKey.leaf``. ``arm`` is the shared.psi gate arm this
    metric records, if any. ``readability`` is the metric recording whether
    this one could be read, if any; it is per-leaf exactly when this one is.
    """

    name: str
    per_leaf: bool = False
    arm: str | None = None
    readability: MetricSpec | None = None

    def __post_init__(self) -> None:
        if not self.name or ':' in self.name:
            raise ValueError(
                f"a metric name must be non-empty and contain no ':'; got {self.name!r}"
            )
        if self.readability is not None and self.readability.per_leaf != self.per_leaf:
            raise ValueError(
                'a readability metric must share its metric\'s per_leaf: '
                f'{self.name!r} has per_leaf={self.per_leaf}, its readability '
                f'{self.readability.name!r} has per_leaf={self.readability.per_leaf}'
            )

    def key(self, leaf: str | None = None) -> SampleKey:
        return SampleKey(self.name, leaf)


@dataclass(frozen=True)
class MetricGroup:
    """Metrics collected by one collector, admitted and windowed together.

    ``collect`` is the live zero-argument collector. ``windowed`` is False when
    the source is already windowed (PSI avg10), so its rows carry NULL window
    columns.
    """

    name: GroupName
    collect: Callable[[], Samples]
    metrics: tuple[MetricSpec, ...]
    windowed: bool

    def __post_init__(self) -> None:
        names = [spec.name for spec in self.metrics]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(
                f'metric names must be unique within a group: {self.name} repeats {repeated}'
            )
        foreign = [
            f'{spec.name!r} reads {spec.readability.name!r}'
            for spec in self.metrics
            if spec.readability is not None and spec.readability not in self.metrics
        ]
        if foreign:
            raise ValueError(
                f'a readability metric must belong to the same group: in {self.name}, '
                f'{"; ".join(foreign)}, which the group does not declare'
            )

    @property
    def exact(self) -> frozenset[str]:
        return frozenset(spec.name for spec in self.metrics if not spec.per_leaf)

    @property
    def per_leaf(self) -> frozenset[str]:
        return frozenset(spec.name for spec in self.metrics if spec.per_leaf)

    def admits(self, key: SampleKey) -> bool:
        """A host-level key by exact name with no leaf; a per-leaf key with a non-empty leaf."""
        return (key.metric in self.exact and key.leaf is None) or (
            key.metric in self.per_leaf and bool(key.leaf)
        )


_PSI_CPU_SOME = MetricSpec('psi_cpu_some_avg10', arm='cpu_some_avg10')
_PSI_CPU_FULL = MetricSpec('psi_cpu_full_avg10')
_PSI_MEM_SOME = MetricSpec('psi_mem_some_avg10', arm='mem_some_avg10')
_PSI_MEM_FULL = MetricSpec('psi_mem_full_avg10', arm='mem_full_avg10')
_PSI_IO_SOME = MetricSpec('psi_io_some_avg10', arm='io_some_avg10')
_PSI_IO_FULL = MetricSpec('psi_io_full_avg10')

_OCCT_QUEUE_DEPTH = MetricSpec('occt_queue_depth')
_VERIFY_CONCURRENCY = MetricSpec('verify_concurrency')
_VERIFY_RSS_TOTAL_BYTES = MetricSpec('verify_rss_total_bytes')

_RUNQUEUE_READ_OK = MetricSpec('runqueue_read_ok')
_RUNQUEUE_RATIO = MetricSpec(
    'runqueue_ratio', arm='runqueue_ratio', readability=_RUNQUEUE_READ_OK
)
_OWN_READ_OK = MetricSpec('own_read_ok', per_leaf=True)
_OWN_CPU_SOME10 = MetricSpec(
    'own_cpu_some10', per_leaf=True, arm='own_cpu_some_avg10', readability=_OWN_READ_OK
)


# The cgroup v2 mountpoint: a default for the injected ``cgroup_root`` seam,
# which every test overrides. shared.psi holds the same literal privately for
# its own defaults and exports no constant to share.
_CGROUP_ROOT = Path('/sys/fs/cgroup')

_USER_MANAGER_PREFIX = 'user@'
_DF_LEAF_GLOB = '*/df-*.slice'
_ORCHESTRATOR_LEAF_GLOB = '*/orchestrator-*.service'


def _anchor_segments(own_cgroup_path: str) -> list[str] | None:
    """Return the segments of *own_cgroup_path* up to its ``user@<uid>.service``.

    ``None`` when there is no such segment — i.e. nothing to anchor at, which
    is not an error: a sampler running outside a systemd user manager (or one
    handed shared.psi's ``OwnCgroup('', None)`` failure value) simply has no
    per-slice neighbourhood to enumerate.
    """
    segments = own_cgroup_path.split('/')
    for depth, segment in enumerate(segments, start=1):
        if segment.startswith(_USER_MANAGER_PREFIX):
            return segments[:depth]
    return None


def discover_pressure_cgroups(
    *,
    own_cgroup_path: str,
    cgroup_root: Path,
) -> list[tuple[str, Path]]:
    """Return sorted ``(leaf_name, cpu_pressure_path)`` for the sibling cgroups.

    One per ``df-*.slice`` present, else one per ``orchestrator-*.service``
    (``plans/load-throttle-harmonisation-prd.md`` §6.4); the fallback covers a
    host without task 3394's df slices.

    The search is anchored at the ``user@<uid>.service`` segment of
    *own_cgroup_path* rather than walking down from *cgroup_root*: another
    user manager's slices are not ours, and on a 5 s cadence the anchored
    glob is far cheaper than a cgroupfs walk (timings: commit "feat(6773):
    one MetricGroup table in sampler.metrics"). The path comes from
    ``shared.psi.resolve_own_cgroup``, so there is no second /proc/self/cgroup
    reader.

    Never raises. Any failure (an unanchorable path, a missing cgroup_root, a
    permission error mid-glob) returns ``[]``, which ``collect_load_metrics``
    reports as "no leaves discovered"; it never synthesises a leaf name.
    """
    segments = _anchor_segments(own_cgroup_path)
    if segments is None:
        return []
    anchor = Path(cgroup_root).joinpath(*(s for s in segments if s))
    try:
        leaves = sorted(anchor.glob(_DF_LEAF_GLOB))
        if not leaves:
            leaves = sorted(anchor.glob(_ORCHESTRATOR_LEAF_GLOB))
    except OSError:
        logger.warning('cgroup discovery failed under %s', anchor, exc_info=True)
        return []
    return [(leaf.name, leaf / 'cpu.pressure') for leaf in leaves]

# parse_pressure_file is shared.psi's, re-exported so this module's public API
# is unchanged; TestParserRehomedToShared pins the identity.

_PSI_SOURCES: tuple[tuple[str, MetricSpec, MetricSpec], ...] = (
    ('cpu', _PSI_CPU_SOME, _PSI_CPU_FULL),
    ('memory', _PSI_MEM_SOME, _PSI_MEM_FULL),
    ('io', _PSI_IO_SOME, _PSI_IO_FULL),
)
"""(pressure-file name, its ``some`` spec, its ``full`` spec)."""


def collect_psi(
    *,
    read: Callable[[str], str] = read_pressure,
) -> Samples:
    """Return the some/full avg10 of each pressure file, keyed by its PSI spec.

    A file that does not parse contributes no rows, never a fabricated 0.0.

    Args:
        read: Maps a pressure-file name to its text. Defaults to reading
              /proc/pressure/* from the live kernel.
    """
    out: dict[SampleKey, float] = {}
    for file_name, some, full in _PSI_SOURCES:
        text = read(file_name)
        parsed = parse_pressure_file(text)
        if parsed is None:
            logger.warning(
                'PSI parse miss for %s (no avg10 lines in %d bytes); '
                'emitting no row instead of a fabricated 0.0',
                file_name,
                len(text),
            )
            continue
        out[some.key()] = parsed['some_avg10']
        out[full.key()] = parsed['full_avg10']
    return out


def collect_load_metrics(
    *,
    read_runqueue: Callable[..., RunqueueReading] = read_runqueue_ratio,
    own_cgroup_path: str | None = None,
    proc_cgroup_path: str | Path | None = None,
    cgroup_root: Path = _CGROUP_ROOT,
) -> Samples:
    """Return the runqueue and per-cgroup own-pressure metrics for one tick.

    ``runqueue_read_ok`` (``TICK_CLOCK``) is emitted on every call as 1.0/0.0,
    and ``runqueue_ratio`` only when the read succeeded. shared.psi degrades an
    unreadable /proc/stat to ``RunqueueReading(0.0, False)``, a fail-open value
    rather than a reading, so persisting that 0.0 as a ratio would write a
    fabricated "completely idle host" row (the defect task 1817 fixed here).
    Readability is a metric rather than a log line so that the calibration
    gates (tasks 5219/5220) can count how often the runqueue arm was readable.

    Each discovered leaf gets an ``own_read_ok`` row, plus ``own_cpu_some10``
    when its pressure file read. No leaf discovered means no own rows, logged
    at DEBUG when there is no user-manager anchor at all (a dev box, a
    container) and at WARNING when an anchor resolved but held no leaf.

    Args:
        read_runqueue: shared.psi's ``read_runqueue_ratio`` by default.
        own_cgroup_path: The sampler's own ``0::`` kernel path. ``None``, which
            every live tick passes, asks ``shared.psi.resolve_own_cgroup``.
        proc_cgroup_path: Where that resolution reads the ``0::`` line from,
            forwarded unchanged; ``None`` leaves shared.psi's default in place.
        cgroup_root: Where the unified hierarchy is mounted.
    """
    runqueue = read_runqueue()
    out: dict[SampleKey, float] = {_RUNQUEUE_READ_OK.key(): float(runqueue.read_ok)}
    if runqueue.read_ok:
        out[_RUNQUEUE_RATIO.key()] = float(runqueue.ratio)

    if own_cgroup_path is None:
        seam = {} if proc_cgroup_path is None else {'proc_cgroup_path': proc_cgroup_path}
        own_cgroup_path = resolve_own_cgroup(
            None, cgroup_root=str(cgroup_root), **seam
        ).path
    leaves = discover_pressure_cgroups(
        own_cgroup_path=own_cgroup_path, cgroup_root=cgroup_root
    )
    if not leaves:
        say = logger.debug if _anchor_segments(own_cgroup_path) is None else logger.warning
        say(
            'no pressure cgroups discovered under the anchor derived from %r; '
            'emitting no own_* rows rather than inventing a leaf name',
            own_cgroup_path,
        )
    for leaf_name, pressure_path in leaves:
        some_avg10 = _read_leaf_some_avg10(pressure_path)
        out[_OWN_READ_OK.key(leaf_name)] = float(some_avg10 is not None)
        if some_avg10 is not None:
            out[_OWN_CPU_SOME10.key(leaf_name)] = some_avg10
    return out


def _read_leaf_some_avg10(pressure_path: Path) -> float | None:
    """Return one cgroup's ``some avg10``, or ``None`` if it could not be read.

    The text goes to shared.psi's ``parse_pressure_file``, whose ``None``
    already means "unparseable", so an unreadable file collapses onto the same
    sentinel. shared.psi's ``read_own_cgroup_pressure`` is not reused: it reads
    the calling process's own cgroup, and this reads every sibling leaf.
    """
    try:
        parsed = parse_pressure_file(pressure_path.read_text())
    except OSError:
        logger.debug('cgroup pressure unreadable: %s', pressure_path, exc_info=True)
        return None
    if parsed is None:
        return None
    return parsed['some_avg10']


@dataclass(frozen=True)
class ProcessRecord:
    """What the process metrics need to know about one process, decided once."""

    pid: int
    is_occt: bool
    is_verify: bool


def classify_process(info: Mapping[str, Any]) -> ProcessRecord:
    """Classify one process from its ``proc.info``.

    An unreadable command line is ``None`` there (psutil's value for an
    AccessDenied attribute) and classifies as neither.
    """
    cmdline = info.get('cmdline') or []
    return ProcessRecord(
        pid=info['pid'],
        is_occt=any('cargo-test-occt-gated' in part for part in cmdline),
        is_verify=bool(cmdline) and os.path.basename(cmdline[0]) == 'verify.sh',
    )


def count_occt_queue_depth(
    records: Iterable[ProcessRecord],
    fd9_exists: Callable[[int], bool],
) -> int:
    """Count the occt processes still waiting on the semaphore (see the FD-9 heuristic)."""
    return sum(1 for record in records if record.is_occt and not fd9_exists(record.pid))


def count_verify_concurrency(records: Iterable[ProcessRecord]) -> int:
    return sum(1 for record in records if record.is_verify)


def sum_verify_rss(verify_roots: Iterable[Any]) -> int:
    """Sum RSS (bytes) over the verify roots and their descendants, each pid once.

    ``children(recursive=True)`` already returns every descendant in one flat
    pass, so no child is recursed into. A process that vanished or is denied
    (``psutil.Error``) is skipped; any other error propagates.
    """
    import psutil

    seen_pids: set[int] = set()
    total = 0
    for root in verify_roots:
        try:
            tree = [root, *root.children(recursive=True)]
        except psutil.Error:
            tree = [root]
        for proc in tree:
            if proc.pid in seen_pids:
                continue
            seen_pids.add(proc.pid)
            try:
                total += proc.memory_info().rss
            except psutil.Error:
                continue
    return total


def _default_fd9_exists(pid: int) -> bool:
    return os.path.lexists(f'/proc/{pid}/fd/9')


def collect_process_metrics(
    *,
    proc_iter: Callable[[list[str]], Iterable[Any]] | None = None,
    fd9_exists: Callable[[int], bool] = _default_fd9_exists,
) -> Samples:
    """Return the process group's metrics from one process scan.

    The scan requests only ``pid`` and ``cmdline``, each process is classified
    once from ``proc.info``, and memory is read only for verify trees. A
    per-process ``psutil.Error`` is skipped there; anything else, a failed
    scan included, propagates to ``sampler.__main__``'s per-group handler.
    """
    import psutil

    procs = list((proc_iter or psutil.process_iter)(['pid', 'cmdline']))
    records = [classify_process(proc.info) for proc in procs]
    verify_roots = [proc for proc, record in zip(procs, records, strict=True) if record.is_verify]
    return {
        _OCCT_QUEUE_DEPTH.key(): float(count_occt_queue_depth(records, fd9_exists)),
        _VERIFY_CONCURRENCY.key(): float(count_verify_concurrency(records)),
        _VERIFY_RSS_TOTAL_BYTES.key(): float(sum_verify_rss(verify_roots)),
    }


def _metric_table(*groups: MetricGroup) -> tuple[MetricGroup, ...]:
    group_names = [group.name for group in groups]
    metric_names = [spec.name for group in groups for spec in group.metrics]
    repeated = sorted(
        {str(name) for name in group_names if group_names.count(name) > 1}
        | {name for name in metric_names if metric_names.count(name) > 1}
    )
    if repeated:
        raise ValueError(
            f'group and metric names must be unique across the table; repeated: {repeated}'
        )
    return groups


METRIC_GROUPS: tuple[MetricGroup, ...] = _metric_table(
    MetricGroup(
        GroupName.PSI,
        collect=collect_psi,
        metrics=(
            _PSI_CPU_SOME, _PSI_CPU_FULL,
            _PSI_MEM_SOME, _PSI_MEM_FULL,
            _PSI_IO_SOME, _PSI_IO_FULL,
        ),
        windowed=False,
    ),
    MetricGroup(
        GroupName.PROCESS,
        collect=collect_process_metrics,
        metrics=(_OCCT_QUEUE_DEPTH, _VERIFY_CONCURRENCY, _VERIFY_RSS_TOTAL_BYTES),
        windowed=True,
    ),
    MetricGroup(
        GroupName.LOAD,
        collect=collect_load_metrics,
        metrics=(_RUNQUEUE_RATIO, _RUNQUEUE_READ_OK, _OWN_CPU_SOME10, _OWN_READ_OK),
        windowed=True,
    ),
)
"""The sampler's one metric table, in collection order.

A complete tick writes one row per exact name of every group, plus one row
per per-leaf name for each cgroup leaf discovered that tick.
"""


def arm_metrics(groups: Iterable[MetricGroup]) -> Mapping[str, MetricSpec]:
    """Map each gate arm to the one spec recording it; a spec without an arm adds nothing."""
    arms: dict[str, MetricSpec] = {}
    for group in groups:
        for spec in group.metrics:
            if spec.arm is None:
                continue
            if spec.arm in arms:
                raise ValueError(
                    f'a gate arm is recorded by one metric: {spec.arm!r} is claimed by '
                    f'{arms[spec.arm].name!r} and {spec.name!r}'
                )
            arms[spec.arm] = spec
    return MappingProxyType(arms)


ARM_METRICS: Mapping[str, MetricSpec] = arm_metrics(METRIC_GROUPS)


def exact_metric_names(
    groups: Iterable[MetricGroup], names: Collection[GroupName]
) -> tuple[str, ...]:
    """The sorted exact (host-level) metric names of the groups named in *names*."""
    return tuple(sorted(
        metric for group in groups if group.name in names for metric in group.exact
    ))


TICK_CLOCK: MetricSpec = _RUNQUEUE_READ_OK
"""Written on every completed load collection, readable or not."""
