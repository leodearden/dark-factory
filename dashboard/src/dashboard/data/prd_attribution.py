"""PRD attribution: which PRD a task belongs to, read from its metadata alone.

A task names its PRD directly (``metadata.prd_path``, or the legacy ``prd`` /
``prd_ref``), or reaches one through lineage links (``spawned_from``,
``x_recovered_from_task``, ``cross_repo_refile_of``) to an ancestor that does
(PRD ``plans/dashboard-prd-lanes-prd.md`` §4.1–§4.3, §5.3–§5.4).
:func:`lineage_entry` is the ONE parse of a task's metadata into a
:class:`LineageEntry`, for projected lineage rows and raw active rows alike, so
no two readers can disagree about what a task says. :func:`attribute` walks a
whole project's entries into one :class:`Attribution` per task.

Pure: no I/O and no clock read.
"""

from __future__ import annotations

import enum
import posixpath
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from types import MappingProxyType
from typing import Any

from shared.timestamps import parse_timestamp_or_warn


class UnresolvedReason(enum.StrEnum):
    """Why a task carries no PRD attribution: the closed set of §4.3."""

    NO_LINK = 'no_link'
    ROOT_WITHOUT_PRD = 'root_without_prd'
    CROSS_PROJECT_LINK = 'cross_project_link'
    UNPARSEABLE_LINK = 'unparseable_link'
    MISSING_TARGET = 'missing_target'
    CYCLE = 'cycle'
    DEPTH_EXCEEDED = 'depth_exceeded'
    FOREIGN_PRD = 'foreign_prd'
    LINEAGE_PENDING = 'lineage_pending'


class LinkField(enum.StrEnum):
    """The metadata keys linking a task to its lineage parent, in precedence order."""

    SPAWNED_FROM = 'spawned_from'
    X_RECOVERED_FROM_TASK = 'x_recovered_from_task'
    CROSS_REPO_REFILE_OF = 'cross_repo_refile_of'


PRD_FIELDS = ('prd_path', 'prd', 'prd_ref')
DONE_PROVENANCE_KEY = 'done_provenance'
LINEAGE_METADATA_KEYS = (*PRD_FIELDS, *(field.value for field in LinkField), DONE_PROVENANCE_KEY)

_UNRESOLVABLE_LINK_REASONS = frozenset(
    {UnresolvedReason.CROSS_PROJECT_LINK, UnresolvedReason.UNPARSEABLE_LINK},
)
_LOCAL_TASK_ID = re.compile(r'[0-9]+')
_CROSS_PROJECT_LINK = re.compile(r'[A-Za-z0-9_.-]+:\d+')
_RAW_LINK_CAP = 200

MAX_HOPS = 8


@dataclass(frozen=True, slots=True)
class LocalLink:
    """A link to a task id in the same project."""

    field: LinkField
    target: int


@dataclass(frozen=True, slots=True)
class UnresolvableLink:
    """A link present on the task that cannot be followed within this project."""

    field: LinkField
    reason: UnresolvedReason
    raw: str

    def __post_init__(self) -> None:
        if self.reason not in _UNRESOLVABLE_LINK_REASONS:
            allowed = ', '.join(sorted(_UNRESOLVABLE_LINK_REASONS))
            raise ValueError(
                f'UnresolvableLink.reason must be one of {allowed}; got {self.reason!s} '
                f'for {self.field!s}={self.raw!r}'
            )


Link = LocalLink | UnresolvableLink


@dataclass(frozen=True, slots=True)
class PrdKey:
    """A PRD file named relative to its project root: one key per PRD."""

    key: str


@dataclass(frozen=True, slots=True)
class Unresolved:
    reason: UnresolvedReason


PrdKeyResult = PrdKey | Unresolved


@dataclass(frozen=True, slots=True)
class LineageEntry:
    """What one task's metadata says about its PRD and its lineage parent."""

    own_prd: PrdKeyResult | None
    link: Link | None
    multi_link: bool
    done_stamped_at: datetime | None
    done_kind: str | None


def coalesce_prd(metadata: Mapping[str, Any]) -> str | None:
    """Coalesce PRD provenance from *metadata* into a single normalized string.

    Checks ``prd_path``, then ``prd``, then ``prd_ref`` (in that precedence
    order); the first value that is a non-empty string after stripping a
    trailing ``#anchor`` or ``§section`` suffix and surrounding whitespace
    wins. Non-string values are skipped. A value that cleans to ``''`` (e.g.
    it was only a suffix) falls through to the next key. Returns ``None``
    when no key yields a non-empty result.
    """
    for key in PRD_FIELDS:
        raw = metadata.get(key)
        if not isinstance(raw, str):
            continue
        cleaned = raw.split('#', 1)[0].split('§', 1)[0].strip()
        if cleaned:
            return cleaned
    return None


def normalise_prd_key(raw: str, project_root: str) -> PrdKeyResult:
    """Key *raw* relative to *project_root*; a PRD outside the root is foreign."""
    path = raw.removeprefix('./')
    if posixpath.isabs(path):
        root_prefix = project_root.rstrip('/') + '/'
        if not path.startswith(root_prefix):
            return Unresolved(UnresolvedReason.FOREIGN_PRD)
        path = path[len(root_prefix):]
    path = posixpath.normpath(path)
    if path == '..' or path.startswith('../'):
        return Unresolved(UnresolvedReason.FOREIGN_PRD)
    return PrdKey(path)


def parse_link(metadata: Mapping[str, Any]) -> Link | None:
    """The task's lineage link: the first present :class:`LinkField`, parsed."""
    for field in LinkField:
        value = metadata.get(field.value)
        if value is not None:
            return _parse_link_value(field, value)
    return None


def _parse_link_value(field: LinkField, value: object) -> Link:
    target = _local_target(value)
    if target is not None:
        return LocalLink(field, target)
    if isinstance(value, str) and _CROSS_PROJECT_LINK.fullmatch(value.strip()):
        return UnresolvableLink(field, UnresolvedReason.CROSS_PROJECT_LINK, value.strip())
    return UnresolvableLink(field, UnresolvedReason.UNPARSEABLE_LINK, repr(value)[:_RAW_LINK_CAP])


def _local_target(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and _LOCAL_TASK_ID.fullmatch(value.strip()):
        number = int(value.strip())
    else:
        return None
    return number if number >= 1 else None


def lineage_entry(metadata: Mapping[str, Any], project_root: str) -> LineageEntry:
    """Parse one task's *metadata* into its :class:`LineageEntry`."""
    raw_prd = coalesce_prd(metadata)
    present_links = sum(metadata.get(field.value) is not None for field in LinkField)
    done_stamped_at, done_kind = _done_provenance(metadata.get(DONE_PROVENANCE_KEY))
    return LineageEntry(
        own_prd=None if raw_prd is None else normalise_prd_key(raw_prd, project_root),
        link=parse_link(metadata),
        multi_link=present_links >= 2,
        done_stamped_at=done_stamped_at,
        done_kind=done_kind,
    )


def _done_provenance(done_provenance: object) -> tuple[datetime | None, str | None]:
    """``(stamped_at, kind)`` of a ``done_provenance`` value, each None when absent or malformed."""
    if not isinstance(done_provenance, Mapping):
        return None, None
    kind = done_provenance.get('kind')
    return (
        _stamped_at(done_provenance.get('stamped_at')),
        kind if isinstance(kind, str) and kind else None,
    )


def _stamped_at(stamped_at: object) -> datetime | None:
    if not isinstance(stamped_at, str):
        return None
    moment, ok = parse_timestamp_or_warn(stamped_at, context='prd_attribution.done_stamped_at')
    return moment if ok else None


@dataclass(frozen=True, slots=True)
class Attribution:
    """One task's PRD key, or the reason it has none, and the links followed.

    ``hops`` counts the links followed before the walk stopped; a
    ``DEPTH_EXCEEDED`` result carries ``MAX_HOPS``.
    """

    prd: str | None
    hops: int
    unresolved: UnresolvedReason | None

    def __post_init__(self) -> None:
        if (self.prd is None) == (self.unresolved is None):
            raise ValueError(
                'Attribution: exactly one of prd and unresolved must be set; '
                f'got prd={self.prd!r}, unresolved={self.unresolved!s}'
            )
        if not 0 <= self.hops <= MAX_HOPS:
            raise ValueError(
                f'Attribution: hops must lie in 0..MAX_HOPS ({MAX_HOPS}); got hops={self.hops}'
            )

    @property
    def direct(self) -> bool:
        return self.hops == 0


def attribute(entries: Mapping[int, LineageEntry]) -> tuple[Mapping[int, Attribution], int]:
    """Attribute every task in *entries*; also count the tasks carrying two or more links.

    The walk from a task stops at the first task with its own PRD verdict, a
    link it cannot follow, a missing target or a cycle, and a local link
    examined ``MAX_HOPS`` links out ends it as ``DEPTH_EXCEEDED``. Each task's
    unbounded outcome is memoised and bounded on its own, so the result is
    the same in any iteration order. Cycle-safe; O(len(entries)).
    """
    outcomes: dict[int, _Outcome] = {}
    for task_id in entries:
        if task_id not in outcomes:
            _walk_from(task_id, entries, outcomes)
    attributions = {task_id: outcomes[task_id].bounded() for task_id in entries}
    multi_link = sum(entry.multi_link for entry in entries.values())
    return MappingProxyType(attributions), multi_link


@dataclass(frozen=True, slots=True)
class _Outcome:
    """Where an unbounded walk from one task stops, ``distance`` links on."""

    prd: str | None
    reason: UnresolvedReason | None
    distance: int
    stopped_at_local_link: bool

    def one_link_earlier(self) -> _Outcome:
        return replace(self, distance=self.distance + 1)

    def bounded(self) -> Attribution:
        if self.distance > MAX_HOPS or (self.distance == MAX_HOPS and self.stopped_at_local_link):
            return Attribution(prd=None, hops=MAX_HOPS, unresolved=UnresolvedReason.DEPTH_EXCEEDED)
        reached_a_linkless_root = self.reason is UnresolvedReason.NO_LINK and self.distance > 0
        reason = UnresolvedReason.ROOT_WITHOUT_PRD if reached_a_linkless_root else self.reason
        return Attribution(prd=self.prd, hops=self.distance, unresolved=reason)


def _walk_from(
    start: int, entries: Mapping[int, LineageEntry], outcomes: dict[int, _Outcome],
) -> None:
    path: list[int] = []
    on_path: dict[int, int] = {}
    current = start
    while current not in outcomes:
        on_path[current] = len(path)
        path.append(current)
        step = _step(entries[current], entries)
        if isinstance(step, _Outcome):
            outcomes[current] = step
            break
        if step in on_path:
            cycle = path[on_path[step]:]
            for member in cycle:
                outcomes[member] = _Outcome(None, UnresolvedReason.CYCLE, len(cycle) - 1, True)
            break
        current = step
    successor = current
    for task_id in reversed(path):
        if task_id not in outcomes:
            outcomes[task_id] = outcomes[successor].one_link_earlier()
        successor = task_id


def _step(entry: LineageEntry, entries: Mapping[int, LineageEntry]) -> _Outcome | int:
    """The walk's verdict at *entry*, or the id of the task it moves on to."""
    if isinstance(entry.own_prd, PrdKey):
        return _Outcome(entry.own_prd.key, None, 0, False)
    if isinstance(entry.own_prd, Unresolved):
        return _Outcome(None, entry.own_prd.reason, 0, False)
    if entry.link is None:
        return _Outcome(None, UnresolvedReason.NO_LINK, 0, False)
    if isinstance(entry.link, UnresolvableLink):
        return _Outcome(None, entry.link.reason, 0, False)
    if entry.link.target not in entries:
        return _Outcome(None, UnresolvedReason.MISSING_TARGET, 0, True)
    return entry.link.target
