"""The sampler's metric groups: what each collects, what it is called, how it is windowed.

``METRIC_GROUPS``, at the bottom of this module, is the metric vocabulary:
every name, its group, its gate arm and its readability metric are declared
there once, and every other spelling is a projection of it. The collectors
above it are pure over their injected seams, with the live readers as
defaults. ``sampler.consumers`` and ``sampler.host_memory`` report figures,
never metric names; ``FIGURE_METRICS`` is where a consumer figure gets one.

FD-9 heuristic
--------------
``occt_queue_depth`` counts processes running ``cargo-test-occt-gated`` whose
``/proc/<pid>/fd/9`` symlink does NOT exist. FD 9 is the read end of the
semaphore pipe the OCCT gate script opens, so its absence means the process
is still waiting to acquire the semaphore. The heuristic breaks if the gate
script moves the pipe to another FD.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

from shared.cgroup_stats import CGROUP_ROOT, CgroupReading
from shared.psi import (
    RunqueueReading,
    parse_pressure_file,
    read_pressure,
    read_runqueue_ratio,
    resolve_own_cgroup,
)
from shared.verify_scope_tag import parse_verify_scope_unit

from sampler.consumers import (
    Consumer,
    ConsumerFigures,
    Figure,
    TreeState,
    account,
    encode_consumer,
    read_tree,
)
from sampler.host_memory import (
    MEMINFO,
    PROC_ROOT,
    ReclaimState,
    ReclaimThreadKind,
    read_host_memory_stat,
    read_host_used_bytes,
    read_reclaim_threads,
    reclaim_increments,
)

logger = logging.getLogger(__name__)

__all__ = [
    'SampleKey',
    'Samples',
    'Collected',
    'Collector',
    'every_tick',
    'GroupName',
    'MetricSpec',
    'MetricGroup',
    'parse_pressure_file',
    'collect_psi',
    'ProcessRecord',
    'classify_process',
    'count_occt_queue_depth',
    'collect_process_metrics',
    'discover_pressure_cgroups',
    'collect_load_metrics',
    'FigureMetric',
    'FIGURE_METRICS',
    'collect_consumers',
    'collect_verify_scopes',
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


@dataclass(frozen=True)
class Collected:
    """One group's collection: its samples, and the state its next collection starts from.

    ``state`` is opaque to everything but the group's own collector. None
    carries nothing forward, and the store keeps whatever state it held.
    """

    samples: Samples
    state: str | None = None


Collector = Callable[[int, str | None], Collected]
"""``collect(now, previous_state)``: every group's one collector signature."""


def every_tick(collect: Callable[[], Samples]) -> Collector:
    """Adapt a stateless zero-argument collector to the ``Collector`` signature."""

    def collect_stateless(_now: int, _previous_state: str | None) -> Collected:
        return Collected(collect())

    return collect_stateless


class GroupName(StrEnum):
    """The collection groups: each fails, is admitted and is windowed as a unit."""

    PSI = 'psi'
    PROCESS = 'process'
    LOAD = 'load'
    CONSUMERS = 'consumers'
    VERIFY_SCOPES = 'verify_scopes'


def _repeated(names: Iterable[str]) -> list[str]:
    """The sorted names that occur more than once in *names*."""
    return sorted(name for name, count in Counter(names).items() if count > 1)


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

    ``collect`` is the live collector. ``windowed`` is False when the source is
    already windowed (PSI avg10), so its rows carry NULL window columns.
    ``due_seconds`` None collects the group on every tick; otherwise it is
    collected when that many seconds have passed since its last attempt, and
    must be unwindowed, because a trailing window is a row count and would
    span a different time at another cadence. ``logs_rows`` False renders the
    group in the tick line as a row and leaf count instead of every row.
    """

    name: GroupName
    collect: Collector
    metrics: tuple[MetricSpec, ...]
    windowed: bool
    due_seconds: int | None = None
    logs_rows: bool = True

    def __post_init__(self) -> None:
        self._check_cadence()
        repeated = _repeated(spec.name for spec in self.metrics)
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

    def _check_cadence(self) -> None:
        due = self.due_seconds
        if due is None:
            return
        if isinstance(due, bool) or not isinstance(due, int) or due <= 0:
            raise ValueError(
                f'due_seconds must be a positive whole number of seconds: {self.name} has {due!r}'
            )
        if self.windowed:
            raise ValueError(
                f'a group with due_seconds must be unwindowed: {self.name} has '
                f'due_seconds={due} and windowed=True'
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

_RUNQUEUE_READ_OK = MetricSpec('runqueue_read_ok')
_RUNQUEUE_RATIO = MetricSpec(
    'runqueue_ratio', arm='runqueue_ratio', readability=_RUNQUEUE_READ_OK
)
_OWN_READ_OK = MetricSpec('own_read_ok', per_leaf=True)
_OWN_CPU_SOME10 = MetricSpec(
    'own_cpu_some10', per_leaf=True, arm='own_cpu_some_avg10', readability=_OWN_READ_OK
)

# Per consumer: the leaf is ``sampler.consumers.encode_consumer``'s name.
_CONSUMER_READ_OK = MetricSpec('consumer_read_ok', per_leaf=True)
_CONSUMER_MEMBERS = MetricSpec('consumer_members', per_leaf=True)
_CONSUMER_MEMORY = MetricSpec(
    'consumer_memory_bytes', per_leaf=True, readability=_CONSUMER_READ_OK
)
_CONSUMER_ANON = MetricSpec('consumer_anon_bytes', per_leaf=True, readability=_CONSUMER_READ_OK)
_CONSUMER_FILE = MetricSpec('consumer_file_bytes', per_leaf=True, readability=_CONSUMER_READ_OK)
_CONSUMER_SWAP = MetricSpec('consumer_swap_bytes', per_leaf=True, readability=_CONSUMER_READ_OK)
_CONSUMER_CPU = MetricSpec('consumer_cpu_usec', per_leaf=True, readability=_CONSUMER_READ_OK)
_CONSUMER_SWAPIN = MetricSpec(
    'consumer_swapin_pages', per_leaf=True, readability=_CONSUMER_READ_OK
)
_CONSUMER_RECLAIM_KSWAPD = MetricSpec(
    'consumer_reclaim_kswapd_pages', per_leaf=True, readability=_CONSUMER_READ_OK
)
_CONSUMER_RECLAIM_DIRECT = MetricSpec(
    'consumer_reclaim_direct_pages', per_leaf=True, readability=_CONSUMER_READ_OK
)
_CONSUMER_REFAULT_ANON = MetricSpec(
    'consumer_refault_anon_pages', per_leaf=True, readability=_CONSUMER_READ_OK
)
_CONSUMER_REFAULT_FILE = MetricSpec(
    'consumer_refault_file_pages', per_leaf=True, readability=_CONSUMER_READ_OK
)
_CONSUMER_MEMORY_HIGH = MetricSpec(
    'consumer_memory_high_events', per_leaf=True, readability=_CONSUMER_READ_OK
)
_CONSUMER_MEMORY_MAX = MetricSpec(
    'consumer_memory_max_events', per_leaf=True, readability=_CONSUMER_READ_OK
)
_CONSUMER_OOM_KILLS = MetricSpec(
    'consumer_oom_kills', per_leaf=True, readability=_CONSUMER_READ_OK
)
_CONSUMER_CPU_SOME_STALL = MetricSpec(
    'consumer_cpu_some_stall_seconds', per_leaf=True, readability=_CONSUMER_READ_OK
)
_CONSUMER_MEMORY_SOME_STALL = MetricSpec(
    'consumer_memory_some_stall_seconds', per_leaf=True, readability=_CONSUMER_READ_OK
)
_CONSUMER_MEMORY_FULL_STALL = MetricSpec(
    'consumer_memory_full_stall_seconds', per_leaf=True, readability=_CONSUMER_READ_OK
)
_CONSUMER_IO_SOME_STALL = MetricSpec(
    'consumer_io_some_stall_seconds', per_leaf=True, readability=_CONSUMER_READ_OK
)

# Host level, from the same collection as the consumers.
_CONSUMER_INTERVAL = MetricSpec('consumer_interval_seconds')
_RECLAIM_KSWAPD_CPU = MetricSpec('reclaim_kswapd_cpu_usec')
_RECLAIM_KCOMPACTD_CPU = MetricSpec('reclaim_kcompactd_cpu_usec')
_HOST_ANON = MetricSpec('host_anon_bytes')
_HOST_FILE = MetricSpec('host_file_bytes')
_HOST_USED = MetricSpec('host_used_bytes')

_VERIFY_SCOPES_LIVE = MetricSpec('verify_scopes_live')
_VERIFY_SCOPE_MEMORY = MetricSpec('verify_scope_memory_bytes')


@dataclass(frozen=True)
class FigureMetric:
    """The metric recording one consumer figure; ``scale`` converts the figure's unit to it."""

    spec: MetricSpec
    scale: float = 1


_SECONDS_PER_USEC = 1e-6

FIGURE_METRICS: Mapping[Figure, FigureMetric] = MappingProxyType({
    Figure.MEMORY: FigureMetric(_CONSUMER_MEMORY),
    Figure.ANON: FigureMetric(_CONSUMER_ANON),
    Figure.FILE: FigureMetric(_CONSUMER_FILE),
    Figure.SWAP: FigureMetric(_CONSUMER_SWAP),
    Figure.CPU_USEC: FigureMetric(_CONSUMER_CPU),
    Figure.SWAPIN_PAGES: FigureMetric(_CONSUMER_SWAPIN),
    Figure.RECLAIM_KSWAPD_PAGES: FigureMetric(_CONSUMER_RECLAIM_KSWAPD),
    Figure.RECLAIM_DIRECT_PAGES: FigureMetric(_CONSUMER_RECLAIM_DIRECT),
    Figure.REFAULT_ANON_PAGES: FigureMetric(_CONSUMER_REFAULT_ANON),
    Figure.REFAULT_FILE_PAGES: FigureMetric(_CONSUMER_REFAULT_FILE),
    Figure.MEMORY_HIGH_EVENTS: FigureMetric(_CONSUMER_MEMORY_HIGH),
    Figure.MEMORY_MAX_EVENTS: FigureMetric(_CONSUMER_MEMORY_MAX),
    Figure.OOM_KILLS: FigureMetric(_CONSUMER_OOM_KILLS),
    Figure.CPU_SOME_STALL_USEC: FigureMetric(_CONSUMER_CPU_SOME_STALL, _SECONDS_PER_USEC),
    Figure.MEMORY_SOME_STALL_USEC: FigureMetric(_CONSUMER_MEMORY_SOME_STALL, _SECONDS_PER_USEC),
    Figure.MEMORY_FULL_STALL_USEC: FigureMetric(_CONSUMER_MEMORY_FULL_STALL, _SECONDS_PER_USEC),
    Figure.IO_SOME_STALL_USEC: FigureMetric(_CONSUMER_IO_SOME_STALL, _SECONDS_PER_USEC),
})
"""Each per-consumer figure's one metric: ``sampler.consumers`` never spells a name."""

_RECLAIM_METRICS: Mapping[ReclaimThreadKind, MetricSpec] = MappingProxyType({
    ReclaimThreadKind.KSWAPD: _RECLAIM_KSWAPD_CPU,
    ReclaimThreadKind.KCOMPACTD: _RECLAIM_KCOMPACTD_CPU,
})

# The default for every injected ``cgroup_root`` seam, which every test overrides.
_CGROUP_ROOT = CGROUP_ROOT

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


def classify_process(info: Mapping[str, Any]) -> ProcessRecord:
    """Classify one process from its ``proc.info``.

    An unreadable command line is ``None`` there (psutil's value for an
    AccessDenied attribute) and is not occt.
    """
    cmdline = info.get('cmdline') or []
    return ProcessRecord(
        pid=info['pid'],
        is_occt=any('cargo-test-occt-gated' in part for part in cmdline),
    )


def count_occt_queue_depth(
    records: Iterable[ProcessRecord],
    fd9_exists: Callable[[int], bool],
) -> int:
    """Count the occt processes still waiting on the semaphore (see the FD-9 heuristic)."""
    return sum(1 for record in records if record.is_occt and not fd9_exists(record.pid))


def _default_fd9_exists(pid: int) -> bool:
    return os.path.lexists(f'/proc/{pid}/fd/9')


def collect_process_metrics(
    *,
    proc_iter: Callable[[list[str]], Iterable[Any]] | None = None,
    fd9_exists: Callable[[int], bool] = _default_fd9_exists,
) -> Samples:
    """Return the process group's metrics from one process scan.

    The scan requests only ``pid`` and ``cmdline``, and each process is
    classified once from ``proc.info``; no memory is read. A failed scan
    propagates to ``sampler.sampler.collect_tick``'s degrade handler. Verify
    runs are counted by their cgroup scopes instead (``collect_verify_scopes``).
    """
    import psutil

    procs = (proc_iter or psutil.process_iter)(['pid', 'cmdline'])
    records = [classify_process(proc.info) for proc in procs]
    return {_OCCT_QUEUE_DEPTH.key(): float(count_occt_queue_depth(records, fd9_exists))}


# The consumer and verify-scope groups' cadence, and the consumer accounting's
# nominal interval, which its gap rule (D3: more than two intervals) reads.
_COLLECTION_DUE_SECONDS = 30

_CONSUMERS_STATE_VERSION = 1


def _encode_consumers_state(tree: TreeState, reclaim: ReclaimState) -> str:
    return json.dumps(
        {'version': _CONSUMERS_STATE_VERSION, 'tree': tree.to_json(), 'reclaim': reclaim.to_json()}
    )


def _decode_consumers_state(text: str | None) -> tuple[TreeState | None, ReclaimState | None]:
    """The carried states; absent, corrupt or foreign text is none, so the collection is a gap."""
    if text is None:
        return None, None
    try:
        data = json.loads(text)
        if data['version'] != _CONSUMERS_STATE_VERSION:
            return None, None
        return TreeState.from_json(data['tree']), ReclaimState.from_json(data['reclaim'])
    except (ValueError, KeyError, TypeError):
        return None, None


def _consumer_samples(per_consumer: Mapping[Consumer, ConsumerFigures]) -> dict[SampleKey, float]:
    samples: dict[SampleKey, float] = {}
    for consumer, figures in per_consumer.items():
        leaf = encode_consumer(consumer)
        samples[_CONSUMER_MEMBERS.key(leaf)] = float(figures.members)
        samples[_CONSUMER_READ_OK.key(leaf)] = float(figures.readable)
        for figure, value in {**figures.gauges, **(figures.increments or {})}.items():
            metric = FIGURE_METRICS[figure]
            samples[metric.spec.key(leaf)] = float(value * metric.scale)
    return samples


def collect_consumers(
    now: int,
    previous_state: str | None,
    *,
    cgroup_root: Path = _CGROUP_ROOT,
    proc_root: Path = PROC_ROOT,
    meminfo_path: Path = MEMINFO,
) -> Collected:
    """The consumer group (PRD C1): every consumer's figures and the host-level ones.

    Increments are taken against *previous_state*, the state the last written
    collection carried, and the new state carries both the cgroup tree's and
    the reclaim threads'. A host-level figure that could not be read is
    absent, never 0.
    """
    tree_before, reclaim_before = _decode_consumers_state(previous_state)
    accounting = account(
        tree_before, read_tree(cgroup_root), now=now, nominal_interval=_COLLECTION_DUE_SECONDS
    )
    reclaim, reclaim_state = reclaim_increments(
        reclaim_before,
        read_reclaim_threads(proc_root),
        now=now,
        nominal_interval=_COLLECTION_DUE_SECONDS,
    )
    host_memory = read_host_memory_stat(cgroup_root)
    host = {
        _CONSUMER_INTERVAL: accounting.interval_seconds,
        _HOST_ANON: host_memory.anon_bytes,
        _HOST_FILE: host_memory.file_bytes,
        _HOST_USED: read_host_used_bytes(meminfo_path),
        **{_RECLAIM_METRICS[kind]: usec for kind, usec in (reclaim or {}).items()},
    }
    samples = {
        **_consumer_samples(accounting.per_consumer),
        **{spec.key(): float(value) for spec, value in host.items() if value is not None},
    }
    return Collected(
        MappingProxyType(samples), _encode_consumers_state(accounting.state, reclaim_state)
    )


def collect_verify_scopes(
    _now: int, _previous_state: str | None, *, cgroup_root: Path = _CGROUP_ROOT
) -> Collected:
    """The verify-scope group: live ``df-verify-*.scope`` cgroups, in either placement.

    A scope is recognised only by ``shared.verify_scope_tag``. Their memory is
    written only when every live scope's could be read, so it is never an
    undercount.
    """
    scopes = [
        node for node in read_tree(cgroup_root).nodes
        if node.segments and parse_verify_scope_unit(node.segments[-1]) is not None
    ]
    memory = [
        figure for node in scopes
        if isinstance(node.reading, CgroupReading)
        and (figure := Figure.MEMORY.source(node.reading)) is not None
    ]
    samples = {_VERIFY_SCOPES_LIVE.key(): float(len(scopes))}
    if len(memory) == len(scopes):
        samples[_VERIFY_SCOPE_MEMORY.key()] = float(sum(memory))
    return Collected(MappingProxyType(samples))


def _metric_table(*groups: MetricGroup) -> tuple[MetricGroup, ...]:
    repeated = _repeated(str(group.name) for group in groups) + _repeated(
        spec.name for group in groups for spec in group.metrics
    )
    if repeated:
        raise ValueError(
            f'group and metric names must be unique across the table; repeated: {repeated}'
        )
    return groups


METRIC_GROUPS: tuple[MetricGroup, ...] = _metric_table(
    MetricGroup(
        GroupName.PSI,
        collect=every_tick(collect_psi),
        metrics=(
            _PSI_CPU_SOME, _PSI_CPU_FULL,
            _PSI_MEM_SOME, _PSI_MEM_FULL,
            _PSI_IO_SOME, _PSI_IO_FULL,
        ),
        windowed=False,
    ),
    MetricGroup(
        GroupName.PROCESS,
        collect=every_tick(collect_process_metrics),
        metrics=(_OCCT_QUEUE_DEPTH,),
        windowed=True,
    ),
    MetricGroup(
        GroupName.LOAD,
        collect=every_tick(collect_load_metrics),
        metrics=(_RUNQUEUE_RATIO, _RUNQUEUE_READ_OK, _OWN_CPU_SOME10, _OWN_READ_OK),
        windowed=True,
    ),
    MetricGroup(
        GroupName.CONSUMERS,
        collect=collect_consumers,
        metrics=(
            _CONSUMER_MEMBERS, _CONSUMER_READ_OK,
            *(metric.spec for metric in FIGURE_METRICS.values()),
            _CONSUMER_INTERVAL, *_RECLAIM_METRICS.values(),
            _HOST_ANON, _HOST_FILE, _HOST_USED,
        ),
        windowed=False,
        due_seconds=_COLLECTION_DUE_SECONDS,
        logs_rows=False,
    ),
    MetricGroup(
        GroupName.VERIFY_SCOPES,
        collect=collect_verify_scopes,
        metrics=(_VERIFY_SCOPES_LIVE, _VERIFY_SCOPE_MEMORY),
        windowed=False,
        due_seconds=_COLLECTION_DUE_SECONDS,
    ),
)
"""The sampler's one metric table, in collection order.

Every tick, the PSI, process and load groups write one row per exact name,
plus one row per per-leaf name for each cgroup leaf discovered that tick.
Every 30 seconds the verify-scope group writes its two names, and the
consumer group writes, per consumer, its gauges, member count and readability
flag, its increments unless the interval is a gap, and its host-level names.
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
