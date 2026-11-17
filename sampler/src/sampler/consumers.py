"""Attribute the host's cgroup use to consumers.

PRD ``plans/per-consumer-resource-recording-prd.md`` D2. A *consumer* is a
role and an optional key. Members are leaf cgroups, named by ONE ordered rule
table (``LEAF_RULES``, first match wins); every inner cgroup adds a residual,
named by ``classify_residual``. The consumer's string form in the store's
``leaf`` column is produced and parsed only by ``encode_consumer`` /
``parse_consumer``.

A cgroup path is handled as its tuple of segments below the cgroup root; the
root is ``()``. Verify scopes are recognised only through
``shared.verify_scope_tag``.

D3: ``read_tree`` reads the whole tree post-order (children before their
parent, so hierarchical counter skew is mostly non-negative), and ``account``
turns two collections into per-consumer figures. Gauges are summed over
members and residuals; counters become increments over the interval, kept
per cgroup by path and inode in a ``TreeState``. A residual subtracts each
child's ACCOUNTED figure (its own value when validly read, else its children's
accounted sum), so an unreadable inner cgroup is never double counted;
residuals are clamped at zero. Stall totals are not hierarchical: only members
carry them.

This module names no metric: ``sampler.metrics`` imports it, never the reverse.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

from shared.cgroup_stats import (
    CgroupReading,
    PressureResource,
    PressureTotals,
    Unreadable,
    read_cgroup,
)
from shared.verify_scope_tag import parse_verify_scope_unit

__all__ = [
    'CGROUP_ROOT',
    'LEAF_RULES',
    'Accounting',
    'CgroupNode',
    'Consumer',
    'ConsumerFigures',
    'Figure',
    'FigureKind',
    'LeafRule',
    'Role',
    'TreeReading',
    'TreeState',
    'account',
    'classify_member',
    'classify_residual',
    'encode_consumer',
    'parse_consumer',
    'read_tree',
]

CGROUP_ROOT = Path('/sys/fs/cgroup')

Segments = tuple[str, ...]


class Role(StrEnum):
    SERVICE = 'service'
    AGENTS = 'agents'
    VERIFY = 'verify'
    PROJECT = 'project'
    SHARED = 'shared'
    INTERACTIVE = 'interactive'
    SYSTEM = 'system'
    OTHER = 'other'
    EXITED = 'exited'
    KERNEL = 'kernel'


@dataclass(frozen=True)
class Consumer:
    """A role and its key; only the kernel is unkeyed."""

    role: Role
    key: str | None = None

    def __post_init__(self) -> None:
        if (self.key is None) != (self.role is Role.KERNEL):
            raise ValueError(f'{self.role} consumer with key {self.key!r}')
        if self.key == '':
            raise ValueError(f'{self.role} consumer with an empty key')

    def __lt__(self, other: Consumer) -> bool:
        return (self.role, self.key or '') < (other.role, other.key or '')


_KEY_SEPARATOR = '@'


def encode_consumer(consumer: Consumer) -> str:
    """``<role>@<key>``, or ``<role>`` for the unkeyed kernel."""
    if consumer.key is None:
        return consumer.role.value
    return f'{consumer.role.value}{_KEY_SEPARATOR}{consumer.key}'


def parse_consumer(text: str) -> Consumer:
    """The inverse of ``encode_consumer``; splits on the FIRST '@' (keys may hold '@')."""
    role_text, separator, key = text.partition(_KEY_SEPARATOR)
    try:
        return Consumer(Role(role_text), key if separator else None)
    except ValueError as exc:
        raise ValueError(f'not a consumer name: {text!r} ({exc})') from exc


_ROOT_ANCHOR = '/'
_FIXED_ANCHORS = frozenset(
    {'system.slice', 'user.slice', 'app.slice', 'session.slice', 'background.slice'}
)
_USER_ANCHOR = re.compile(r'user-\d+\.slice|user@\d+\.service')
_ORCHESTRATOR_SERVICE = re.compile(r'orchestrator-.+\.service')
_WATCHDOG = 'orchestrator-watchdog.service'
_DF_PROJECT_SLICE = re.compile(r'df-.+\.slice')
_REIFY_SLICE = 'reify.slice'
_AGENT_CHILD_PREFIX = 'agent-'
_FACTORY_SERVICES = frozenset(
    {
        'sccache.service',
        'fused-memory.service',
        'dark-factory-dashboard.service',
        _WATCHDOG,
        'reify-warm-lane-gc.service',
        'reify-jobserver.service',
        'pytest-jobserver.service',
        'cliproxyapi.service',
        'dark-factory-load-sampler.service',
        'app-quality\\x2dsnapshot.slice',
    }
)
_T3CODE_UNITS = frozenset({'t3code.service', 'app-t3code.slice'})
_T3CODE_SCOPE_PREFIX = 'app-com.t3tools.T3Code-'
_TERMINAL_PREFIX = 'app-org.kde.konsole-'
_DOCKER_SCOPE = re.compile(r'docker-.+\.scope')


def _is_anchor(name: str) -> bool:
    return name in _FIXED_ANCHORS or _USER_ANCHOR.fullmatch(name) is not None


def _is_orchestrator_service(name: str) -> bool:
    return name != _WATCHDOG and _ORCHESTRATOR_SERVICE.fullmatch(name) is not None


def _is_project_slice(name: str) -> bool:
    return name == _REIFY_SLICE or _DF_PROJECT_SLICE.fullmatch(name) is not None


def _nearest_anchor(segments: Segments) -> str:
    """The deepest anchor on the path, the cgroup itself included; the root is '/'."""
    return next((name for name in reversed(segments) if _is_anchor(name)), _ROOT_ANCHOR)


def _first_index(segments: Segments, predicate: Callable[[str], bool]) -> int | None:
    return next((i for i, name in enumerate(segments) if predicate(name)), None)


def _service_key(segments: Segments, service_index: int) -> str:
    """The project slice the service sits in, else the service unit itself."""
    enclosing = [name for name in segments[:service_index] if _is_project_slice(name)]
    return enclosing[-1] if enclosing else segments[service_index]


def _orchestrator_service(segments: Segments) -> Consumer | None:
    index = _first_index(segments, _is_orchestrator_service)
    if index is None:
        return None
    child = segments[index + 1] if index + 1 < len(segments) else ''
    role = Role.AGENTS if child.startswith(_AGENT_CHILD_PREFIX) else Role.SERVICE
    return Consumer(role, _service_key(segments, index))


def _verify_scope(segments: Segments) -> Consumer | None:
    for name in segments:
        unit = parse_verify_scope_unit(name)
        if unit is not None:
            return Consumer(Role.VERIFY, unit.tag)
    return None


def _project_slice(segments: Segments) -> Consumer | None:
    index = _first_index(segments, _is_project_slice)
    return None if index is None else Consumer(Role.PROJECT, segments[index])


def _factory_service(segments: Segments) -> Consumer | None:
    index = _first_index(segments, _FACTORY_SERVICES.__contains__)
    return None if index is None else Consumer(Role.SHARED, segments[index])


def _t3code(segments: Segments) -> Consumer | None:
    if any(name in _T3CODE_UNITS or name.startswith(_T3CODE_SCOPE_PREFIX) for name in segments):
        return Consumer(Role.INTERACTIVE, 't3code')
    return None


def _terminals(segments: Segments) -> Consumer | None:
    if any(name.startswith(_TERMINAL_PREFIX) for name in segments):
        return Consumer(Role.INTERACTIVE, 'terminals')
    return None


def _docker(segments: Segments) -> Consumer | None:
    if len(segments) >= 2 and segments[0] == 'system.slice' and _DOCKER_SCOPE.fullmatch(segments[1]):
        return Consumer(Role.SYSTEM, 'docker')
    return None


def _nearest_anchor_rule(segments: Segments) -> Consumer:
    return Consumer(Role.OTHER, _nearest_anchor(segments))


class LeafRule(NamedTuple):
    name: str
    match: Callable[[Segments], Consumer | None]


LEAF_RULES: tuple[LeafRule, ...] = (
    LeafRule('orchestrator_service', _orchestrator_service),
    LeafRule('verify_scope', _verify_scope),
    LeafRule('project_slice', _project_slice),
    LeafRule('factory_service', _factory_service),
    LeafRule('t3code', _t3code),
    LeafRule('terminals', _terminals),
    LeafRule('docker', _docker),
    LeafRule('nearest_anchor', _nearest_anchor_rule),
)
"""D2's leaf rules in order; the last is the fallback and always matches."""


def _first_named(segments: Segments, rules: tuple[LeafRule, ...]) -> Consumer | None:
    return next(
        (consumer for rule in rules if (consumer := rule.match(segments)) is not None), None
    )


def classify_member(segments: Segments) -> Consumer:
    """The consumer a leaf cgroup's use belongs to."""
    consumer = _first_named(segments, LEAF_RULES)
    assert consumer is not None, 'the nearest_anchor rule always matches'
    return consumer


def classify_residual(segments: Segments) -> Consumer:
    """The consumer an inner cgroup's residual belongs to (D2's residual order)."""
    if not segments:
        return Consumer(Role.KERNEL)
    own = segments[-1]
    if _is_orchestrator_service(own):
        return Consumer(Role.AGENTS, _service_key(segments, len(segments) - 1))
    if _is_project_slice(own):
        return Consumer(Role.PROJECT, own)
    named = _first_named(segments, LEAF_RULES[:-1])
    return named if named is not None else Consumer(Role.EXITED, _nearest_anchor(segments))


class FigureKind(Enum):
    GAUGE = auto()
    HIERARCHICAL = auto()
    STALL = auto()


_Source = Callable[[CgroupReading], 'int | None']


def _keyed(counters: Callable[[CgroupReading], Mapping[str, int] | None], key: str) -> _Source:
    def read(reading: CgroupReading) -> int | None:
        values = counters(reading)
        return None if values is None else values.get(key)

    return read


def _cpu_stat(reading: CgroupReading) -> Mapping[str, int] | None:
    return reading.cpu_stat


def _memory_stat(reading: CgroupReading) -> Mapping[str, int] | None:
    return reading.memory_stat


def _memory_events(reading: CgroupReading) -> Mapping[str, int] | None:
    return reading.memory_events


def _stall(resource: PressureResource, line: Callable[[PressureTotals], int]) -> _Source:
    def read(reading: CgroupReading) -> int | None:
        totals = reading.pressure.get(resource)
        return None if totals is None else line(totals)

    return read


def _memory(reading: CgroupReading) -> int | None:
    """``memory.current``, or ``anon + file + kernel`` where it is absent (the root)."""
    if reading.memory_current is not None:
        return reading.memory_current
    parts = [_keyed(_memory_stat, key)(reading) for key in ('anon', 'file', 'kernel')]
    return None if None in parts else sum(p for p in parts if p is not None)


def _swap(reading: CgroupReading) -> int | None:
    return reading.swap_current


def _some(totals: PressureTotals) -> int:
    return totals.some_usec


def _full(totals: PressureTotals) -> int:
    return totals.full_usec


class Figure(Enum):
    """The per-consumer figures: each declared once with its kind and its source."""

    MEMORY = (FigureKind.GAUGE, _memory)
    ANON = (FigureKind.GAUGE, _keyed(_memory_stat, 'anon'))
    FILE = (FigureKind.GAUGE, _keyed(_memory_stat, 'file'))
    SWAP = (FigureKind.GAUGE, _swap)
    CPU_USEC = (FigureKind.HIERARCHICAL, _keyed(_cpu_stat, 'usage_usec'))
    SWAPIN_PAGES = (FigureKind.HIERARCHICAL, _keyed(_memory_stat, 'pswpin'))
    RECLAIM_KSWAPD_PAGES = (FigureKind.HIERARCHICAL, _keyed(_memory_stat, 'pgsteal_kswapd'))
    RECLAIM_DIRECT_PAGES = (FigureKind.HIERARCHICAL, _keyed(_memory_stat, 'pgsteal_direct'))
    REFAULT_ANON_PAGES = (FigureKind.HIERARCHICAL, _keyed(_memory_stat, 'workingset_refault_anon'))
    REFAULT_FILE_PAGES = (FigureKind.HIERARCHICAL, _keyed(_memory_stat, 'workingset_refault_file'))
    MEMORY_HIGH_EVENTS = (FigureKind.HIERARCHICAL, _keyed(_memory_events, 'high'))
    MEMORY_MAX_EVENTS = (FigureKind.HIERARCHICAL, _keyed(_memory_events, 'max'))
    OOM_KILLS = (FigureKind.HIERARCHICAL, _keyed(_memory_events, 'oom_kill'))
    CPU_SOME_STALL_USEC = (FigureKind.STALL, _stall(PressureResource.CPU, _some))
    MEMORY_SOME_STALL_USEC = (FigureKind.STALL, _stall(PressureResource.MEMORY, _some))
    MEMORY_FULL_STALL_USEC = (FigureKind.STALL, _stall(PressureResource.MEMORY, _full))
    IO_SOME_STALL_USEC = (FigureKind.STALL, _stall(PressureResource.IO, _some))

    def __init__(self, kind: FigureKind, source: _Source) -> None:
        self.kind = kind
        self.source = source


def _figures_of(*kinds: FigureKind) -> tuple[Figure, ...]:
    return tuple(figure for figure in Figure if figure.kind in kinds)


_GAUGES = _figures_of(FigureKind.GAUGE)
_HIERARCHICAL = _figures_of(FigureKind.HIERARCHICAL)
_STALLS = _figures_of(FigureKind.STALL)
_COUNTERS = _figures_of(FigureKind.HIERARCHICAL, FigureKind.STALL)


def _read_figures(reading: CgroupReading, figures: Iterable[Figure]) -> dict[Figure, int]:
    values = {figure: figure.source(reading) for figure in figures}
    return {figure: value for figure, value in values.items() if value is not None}


@dataclass(frozen=True)
class CgroupNode:
    segments: Segments
    inode: int
    children: tuple[str, ...]
    reading: CgroupReading | Unreadable


@dataclass(frozen=True)
class TreeReading:
    """Every cgroup of one collection, in post-order (children before their parent)."""

    nodes: tuple[CgroupNode, ...]


_LISTING = '.'


def read_tree(root: str | Path = CGROUP_ROOT) -> TreeReading:
    nodes: list[CgroupNode] = []
    _read_subtree(Path(root), (), os.stat(root).st_ino, nodes)
    return TreeReading(tuple(nodes))


def _read_subtree(directory: Path, segments: Segments, inode: int, nodes: list[CgroupNode]) -> bool:
    """Append the subtree's nodes post-order; False if the cgroup exited before it was listed."""
    try:
        with os.scandir(directory) as listing:
            entries = sorted(listing, key=lambda entry: entry.name)
    except FileNotFoundError:
        return False
    except OSError as exc:
        reason = f'{type(exc).__name__}: {exc}'
        nodes.append(CgroupNode(segments, inode, (), Unreadable(_LISTING, reason)))
        return True
    children = tuple(
        entry.name
        for entry in entries
        if entry.is_dir(follow_symlinks=False)
        and _read_subtree(Path(entry.path), (*segments, entry.name), entry.inode(), nodes)
    )
    present = frozenset(entry.name for entry in entries)
    nodes.append(CgroupNode(segments, inode, children, read_cgroup(directory, present=present)))
    return True


_STATE_VERSION = 1


def _encode_path(segments: Segments) -> str:
    return '/'.join(segments)


def _decode_path(text: str) -> Segments:
    return tuple(text.split('/')) if text else ()


def _strict_int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f'not an integer: {value!r}')
    return value


@dataclass(frozen=True)
class TreeState:
    """What the next collection needs: the listing (path -> inode) and readable counters."""

    ts: int
    listing: Mapping[Segments, int]
    readings: Mapping[Segments, Mapping[Figure, int]]

    def to_json(self) -> str:
        return json.dumps(
            {
                'version': _STATE_VERSION,
                'ts': self.ts,
                'listing': {_encode_path(s): inode for s, inode in self.listing.items()},
                'readings': {
                    _encode_path(s): {figure.name: v for figure, v in values.items()}
                    for s, values in self.readings.items()
                },
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> TreeState | None:
        """The decoded state, or ``None`` for any corrupt or foreign text (a gap)."""
        try:
            data = json.loads(text)
            if data['version'] != _STATE_VERSION:
                return None
            listing = {_decode_path(p): _strict_int(i) for p, i in data['listing'].items()}
            readings = {
                _decode_path(p): MappingProxyType(
                    {Figure[name]: _strict_int(v) for name, v in values.items()}
                )
                for p, values in data['readings'].items()
            }
            return cls(_strict_int(data['ts']), MappingProxyType(listing), MappingProxyType(readings))
        except (ValueError, KeyError, TypeError, AttributeError):
            return None


def _state_of(current: TreeReading, now: int) -> TreeState:
    return TreeState(
        ts=now,
        listing=MappingProxyType({node.segments: node.inode for node in current.nodes}),
        readings=MappingProxyType(
            {
                node.segments: MappingProxyType(_read_figures(node.reading, _COUNTERS))
                for node in current.nodes
                if isinstance(node.reading, CgroupReading)
            }
        ),
    )


@dataclass(frozen=True)
class _Values:
    """One cgroup's own figures; ``valid`` False = they cannot be used (flagged)."""

    valid: bool
    own: Mapping[Figure, int]


_INVALID = _Values(False, MappingProxyType({}))


def _gauge_values(node: CgroupNode) -> _Values:
    if not isinstance(node.reading, CgroupReading):
        return _INVALID
    return _Values(True, _read_figures(node.reading, _GAUGES))


def _increment_values(node: CgroupNode, previous: TreeState) -> _Values:
    """D3 increments: continuing = current - previous; reset or new = current."""
    if not isinstance(node.reading, CgroupReading):
        return _INVALID
    current = _read_figures(node.reading, _COUNTERS)
    previous_inode = previous.listing.get(node.segments)
    if previous_inode is None or previous_inode != node.inode:
        return _Values(True, current)
    before = previous.readings.get(node.segments)
    if before is None:
        return _INVALID
    return _Values(True, {f: _counter_increment(before.get(f), v) for f, v in current.items()})


def _counter_increment(before: int | None, now: int) -> int:
    return now if before is None or now < before else now - before


def _contributions(
    tree: TreeReading, values: Mapping[Segments, _Values], figures: tuple[Figure, ...]
) -> dict[Segments, dict[Figure, int]]:
    """Each cgroup's own figure minus its children's accounted figures, clamped at zero.

    A leaf has no children, so its contribution is its own figure.
    """
    accounted: dict[Segments, dict[Figure, int]] = {}
    contributions: dict[Segments, dict[Figure, int]] = {}
    for node in tree.nodes:
        children = [accounted[(*node.segments, child)] for child in node.children]
        below = {f: sum(child[f] for child in children) for f in figures}
        node_values = values[node.segments]
        own = node_values.own if node_values.valid else {}
        accounted[node.segments] = {f: own.get(f, below[f]) for f in figures}
        contributions[node.segments] = {f: max(0, own[f] - below[f]) for f in figures if f in own}
    return contributions


@dataclass(frozen=True)
class ConsumerFigures:
    """``increments`` is None after a gap; a figure no cgroup exposed is absent, never 0."""

    members: int
    readable: bool
    gauges: Mapping[Figure, int]
    increments: Mapping[Figure, int] | None


@dataclass(frozen=True)
class Accounting:
    per_consumer: Mapping[Consumer, ConsumerFigures]
    interval_seconds: int | None
    state: TreeState


class _Tally:
    def __init__(self) -> None:
        self.members = 0
        self.readable = True
        self.gauges: dict[Figure, int] = {}
        self.increments: dict[Figure, int] = {}

    def add(self, gauges: Mapping[Figure, int], increments: Mapping[Figure, int]) -> None:
        for into, values in ((self.gauges, gauges), (self.increments, increments)):
            for figure, value in values.items():
                into[figure] = into.get(figure, 0) + value

    def figures(self, gap: bool) -> ConsumerFigures:
        return ConsumerFigures(
            members=self.members,
            readable=self.readable and not gap,
            gauges=MappingProxyType(self.gauges),
            increments=None if gap else MappingProxyType(self.increments),
        )


def _consumer_of(node: CgroupNode) -> Consumer:
    """A leaf is a member; an inner cgroup contributes a residual."""
    if node.children:
        return classify_residual(node.segments)
    return classify_member(node.segments)


def _cgroups_behind(node: CgroupNode) -> list[Segments]:
    """The cgroups whose readability a node's figures rest on.

    A residual absorbs the use of any child it could not subtract, so its own
    readability and its direct children's both decide its flag.
    """
    return [node.segments, *((*node.segments, child) for child in node.children)]


def _member_stalls(node: CgroupNode, increments: _Values) -> dict[Figure, int]:
    """Stall totals are not hierarchical: only a member carries them, never a residual."""
    if node.children or not increments.valid:
        return {}
    return {figure: v for figure, v in increments.own.items() if figure in _STALLS}


def _is_gap(previous: TreeState | None, now: int, nominal_interval: int) -> bool:
    return previous is None or not 0 < now - previous.ts <= 2 * nominal_interval


def account(
    previous: TreeState | None, current: TreeReading, *, now: int, nominal_interval: int
) -> Accounting:
    """Per-consumer gauges and interval increments for one collection (D3)."""
    gap = _is_gap(previous, now, nominal_interval)
    gauges = {node.segments: _gauge_values(node) for node in current.nodes}
    increments = (
        {node.segments: _INVALID for node in current.nodes}
        if gap or previous is None
        else {node.segments: _increment_values(node, previous) for node in current.nodes}
    )
    gauge_parts = _contributions(current, gauges, _GAUGES)
    increment_parts = _contributions(current, increments, _HIERARCHICAL)

    def usable(segments: Segments) -> bool:
        return gauges[segments].valid and (gap or increments[segments].valid)

    tallies: dict[Consumer, _Tally] = {}
    for node in current.nodes:
        tally = tallies.setdefault(_consumer_of(node), _Tally())
        tally.members += 0 if node.children else 1
        tally.readable = tally.readable and all(map(usable, _cgroups_behind(node)))
        tally.add(
            gauge_parts[node.segments],
            {**increment_parts[node.segments], **_member_stalls(node, increments[node.segments])},
        )
    return Accounting(
        per_consumer=MappingProxyType({c: t.figures(gap) for c, t in tallies.items()}),
        interval_seconds=None if previous is None else now - previous.ts,
        state=_state_of(current, now),
    )
