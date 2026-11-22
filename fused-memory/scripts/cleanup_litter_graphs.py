#!/usr/bin/env python3
"""Delete test and probe litter graphs from the FalkorDB the config names, which is production.

A graph is deleted only when its name fully matches one of ``LITTER_PATTERNS``;
that allowlist is the script's whole blast radius, and every other graph is
kept and reported. ``list`` writes a listing for review; ``delete`` refuses a
listing whose ``delete`` section names anything but litter, removes the
reviewed graphs still present one at a time, and stops if the server restarts.
Runbook: OPERATIONS.md "Deleting test litter from production".

It does not call the store-mutation preflight, which guards the mem0/Qdrant
store this script never touches: its one write is GRAPH.DELETE of an allowlisted
graph. That reasoning dies if it gains a MemoryService or a Qdrant client.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import contextlib
import json
import re
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import IntEnum
from types import MappingProxyType
from typing import NamedTuple, Protocol

from falkordb import FalkorDB
from redis.exceptions import RedisError
from shared.cli_boundary import LoudArgumentParser, run_cli
from shared.safe_io import atomic_write_text

from fused_memory.config.falkordb_endpoints import FalkorEndpoint
from fused_memory.config.schema import FusedMemoryConfig


class Exit(IntEnum):
    OK = 0
    FAILED = 1
    REFUSED = 3
    SERVER_RESTARTED = 4
    KEPT_GRAPH_SHRANK = 5


LITTER_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        '_probe',
        '_probe4748b_[0-9a-f]{8}',
        'probe_e1_gw[0-9]+',
        'probe_e1_master',
        'sweep_e4_gw[0-9]+',
        '_test_(?:3334_falkor_fulltext|530_list_indices_integration|4340_read_pagination'
        '|3561_episode_uuid|3769_drop_vector_indices|2207_merge_entities)_[0-9a-f]{8}',
        'evalmem_test_[0-9a-f]{8}',
    )
)


class NotLitter(ValueError):
    """A graph the allowlist does not name, so it is never deleted."""

    def __init__(self, name: str) -> None:
        super().__init__(f'graph {name!r} is not on LITTER_PATTERNS, so it is never deleted')
        self.name = name


def is_litter(name: str) -> bool:
    return any(pattern.fullmatch(name) for pattern in LITTER_PATTERNS)


def require_litter(name: str) -> str:
    if not is_litter(name):
        raise NotLitter(name)
    return name


@dataclass(frozen=True)
class CleanupPlan:
    delete: tuple[str, ...]
    keep: tuple[str, ...]


def plan_cleanup(names: Iterable[str]) -> CleanupPlan:
    ordered = tuple(names)
    return CleanupPlan(
        delete=tuple(name for name in ordered if is_litter(name)),
        keep=tuple(name for name in ordered if not is_litter(name)),
    )


class GraphServer(Protocol):
    def graph_names(self) -> tuple[str, ...]: ...

    def node_count(self, name: str) -> int: ...

    def delete(self, name: str) -> None: ...

    def run_id(self) -> str: ...


NODE_COUNT_QUERY = 'MATCH (n) RETURN count(n)'


class FalkorGraphServer:
    """GraphServer over a sync FalkorDB client: one read query shape, and GRAPH.DELETE of litter only."""

    def __init__(self, client: FalkorDB) -> None:
        self._client = client

    def graph_names(self) -> tuple[str, ...]:
        return tuple(self._client.list_graphs())

    def node_count(self, name: str) -> int:
        return int(self._client.select_graph(name).ro_query(NODE_COUNT_QUERY).result_set[0][0])

    def delete(self, name: str) -> None:
        self._client.select_graph(require_litter(name)).delete()

    def run_id(self) -> str:
        reply = self._client.connection.info('server')
        if not isinstance(reply, dict) or not reply.get('run_id'):
            raise RedisError('INFO server reported no run_id, so a restart could not be detected')
        return str(reply['run_id'])


def _frozen(counts: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType(dict(counts))


@dataclass(frozen=True)
class Census:
    run_id: str
    counts: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'counts', _frozen(self.counts))


def take_census(server: GraphServer, names: Iterable[str]) -> Census:
    run_id = server.run_id()
    return Census(run_id, {name: server.node_count(name) for name in names})


@dataclass(frozen=True)
class Listing:
    target: str
    run_id: str
    taken_at: str
    delete: Mapping[str, int]
    keep: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'delete', _frozen(self.delete))
        object.__setattr__(self, 'keep', _frozen(self.keep))

    def to_json(self) -> dict[str, object]:
        return {
            'target': self.target,
            'run_id': self.run_id,
            'taken_at': self.taken_at,
            'delete': dict(self.delete),
            'keep': dict(self.keep),
        }


class UnreadableListing(ValueError):
    """A reviewed listing that cannot be read back as a Listing."""

    def __init__(self, path: Path, defect: Exception) -> None:
        super().__init__(f'cannot read the reviewed listing {path}: {defect}')


def build_listing(target_uri: str, server: GraphServer) -> Listing:
    census = take_census(server, server.graph_names())
    plan = plan_cleanup(census.counts)
    return Listing(
        target=target_uri,
        run_id=census.run_id,
        taken_at=datetime.now(UTC).isoformat(timespec='seconds'),
        delete={name: census.counts[name] for name in plan.delete},
        keep={name: census.counts[name] for name in plan.keep},
    )


def write_listing(path: Path, listing: Listing) -> None:
    atomic_write_text(path, json.dumps(listing.to_json(), sort_keys=True, indent=2) + '\n')


def read_listing(path: Path) -> Listing:
    try:
        return _listing_from_json(json.loads(Path(path).read_text(encoding='utf-8')))
    except (OSError, ValueError) as defect:
        raise UnreadableListing(path, defect) from defect


def _listing_from_json(raw: object) -> Listing:
    if not isinstance(raw, dict):
        raise ValueError(f'expected a JSON object, found {type(raw).__name__}')
    return Listing(
        target=_text(raw, 'target'),
        run_id=_text(raw, 'run_id'),
        taken_at=_text(raw, 'taken_at'),
        delete=_counts(raw, 'delete'),
        keep=_counts(raw, 'keep'),
    )


def _required(raw: dict[str, object], key: str) -> object:
    if key not in raw:
        raise ValueError(f'it has no {key!r} key')
    return raw[key]


def _text(raw: dict[str, object], key: str) -> str:
    value = _required(raw, key)
    if not isinstance(value, str):
        raise ValueError(f'{key!r} is a {type(value).__name__}, not a string')
    return value


def _counts(raw: dict[str, object], key: str) -> dict[str, int]:
    value = _required(raw, key)
    if not isinstance(value, dict) or not all(_is_count(count) for count in value.values()):
        raise ValueError(f'{key!r} is not a map of graph name to node count')
    return value


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class Targets:
    delete: tuple[str, ...]
    unreviewed: tuple[str, ...]
    already_absent: tuple[str, ...]


def select_targets(present: Iterable[str], plan: CleanupPlan, reviewed: Listing) -> Targets:
    """Litter both planned and reviewed is deleted; planned-only is unreviewed; reviewed-but-gone is absent."""
    on_server = frozenset(present)
    return Targets(
        delete=tuple(n for n in plan.delete if n in reviewed.delete and n in on_server),
        unreviewed=tuple(n for n in plan.delete if n not in reviewed.delete),
        already_absent=tuple(n for n in reviewed.delete if n not in on_server),
    )


class Shrinkage(NamedTuple):
    graph: str
    before: int
    after: int


def shrunk_graphs(
    before: Mapping[str, int], after: Mapping[str, int], kept: Iterable[str]
) -> tuple[Shrinkage, ...]:
    """Each kept graph whose count fell; one absent from *after* counts as zero."""
    return tuple(
        Shrinkage(name, before[name], after.get(name, 0))
        for name in kept
        if name in before and after.get(name, 0) < before[name]
    )


@dataclass(frozen=True)
class CleanupRecord:
    """A delete run; ``outcome`` is None while the run is in progress."""

    target: str
    run_id_before: str
    run_id_after: str | None
    before: Mapping[str, int]
    after: Mapping[str, int]
    deleted: tuple[str, ...]
    unreviewed: tuple[str, ...]
    already_absent: tuple[str, ...]
    shrank: tuple[Shrinkage, ...]
    outcome: Exit | None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'before', _frozen(self.before))
        object.__setattr__(self, 'after', _frozen(self.after))

    @classmethod
    def started(cls, target: str, before: Census, targets: Targets) -> CleanupRecord:
        return cls(
            target=target,
            run_id_before=before.run_id,
            run_id_after=None,
            before=before.counts,
            after={},
            deleted=(),
            unreviewed=targets.unreviewed,
            already_absent=targets.already_absent,
            shrank=(),
            outcome=None,
        )

    def to_json(self) -> dict[str, object]:
        return {
            'target': self.target,
            'run_id_before': self.run_id_before,
            'run_id_after': self.run_id_after,
            'before': dict(self.before),
            'after': dict(self.after),
            'deleted': list(self.deleted),
            'unreviewed': list(self.unreviewed),
            'already_absent': list(self.already_absent),
            'shrank': [shrinkage._asdict() for shrinkage in self.shrank],
            'outcome': None if self.outcome is None else self.outcome.name.lower(),
        }


def _ignore_progress(record: CleanupRecord) -> None:
    pass


def execute(
    server: GraphServer,
    plan: CleanupPlan,
    reviewed: Listing,
    *,
    pace_s: float,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[CleanupRecord], None] = _ignore_progress,
) -> CleanupRecord:
    """Delete the reviewed litter one graph at a time, stopping if the server restarts.

    Only the plan's graphs are counted. *on_progress* receives the record
    before the first delete and after each one.
    """
    names = server.graph_names()
    planned = frozenset((*plan.delete, *plan.keep))
    before = take_census(server, [name for name in names if name in planned])
    targets = select_targets(names, plan, reviewed)
    started = CleanupRecord.started(reviewed.target, before, targets)
    on_progress(started)
    run = _delete_paced(
        server,
        targets.delete,
        run_id=before.run_id,
        pace_s=pace_s,
        sleep=sleep,
        on_deleted=lambda done: on_progress(replace(started, deleted=done)),
    )
    record = replace(started, deleted=run.deleted)
    if run.run_id_after != before.run_id:
        return _restarted(server, plan, record, run.run_id_after)
    return _finished(server, plan, record)


class DeleteRun(NamedTuple):
    """The graphs deleted, and the run_id that stopped the run or was read after its last delete."""

    deleted: tuple[str, ...]
    run_id_after: str


def _delete_paced(
    server: GraphServer,
    names: tuple[str, ...],
    *,
    run_id: str,
    pace_s: float,
    sleep: Callable[[float], None],
    on_deleted: Callable[[tuple[str, ...]], None],
) -> DeleteRun:
    deleted: list[str] = []
    for index, name in enumerate(names):
        if index:
            sleep(pace_s)
        seen = server.run_id()
        if seen != run_id:
            return DeleteRun(tuple(deleted), seen)
        server.delete(require_litter(name))
        deleted.append(name)
        on_deleted(tuple(deleted))
    return DeleteRun(tuple(deleted), server.run_id())


def _kept_census(server: GraphServer, plan: CleanupPlan) -> Census:
    present = frozenset(server.graph_names())
    return take_census(server, [name for name in plan.keep if name in present])


def _restarted(
    server: GraphServer, plan: CleanupPlan, record: CleanupRecord, run_id_after: str
) -> CleanupRecord:
    """The record of a run a restart stopped; ``after`` stays empty if the restarted server cannot be counted."""
    restarted = replace(record, run_id_after=run_id_after, outcome=Exit.SERVER_RESTARTED)
    try:
        census = _kept_census(server, plan)
    except (RedisError, OSError):
        return restarted
    return replace(
        restarted,
        after=census.counts,
        shrank=shrunk_graphs(record.before, census.counts, plan.keep),
    )


def _finished(server: GraphServer, plan: CleanupPlan, record: CleanupRecord) -> CleanupRecord:
    after = _kept_census(server, plan)
    shrank = shrunk_graphs(record.before, after.counts, plan.keep)
    return replace(
        record,
        run_id_after=after.run_id,
        after=after.counts,
        shrank=shrank,
        outcome=_outcome(record.run_id_before, after.run_id, shrank),
    )


def _outcome(run_id_before: str, run_id_after: str, shrank: tuple[Shrinkage, ...]) -> Exit:
    if run_id_after != run_id_before:
        return Exit.SERVER_RESTARTED
    if shrank:
        return Exit.KEPT_GRAPH_SHRANK
    return Exit.OK


class WrongTarget(ValueError):
    """A reviewed listing taken on another server than the one the config names."""

    def __init__(self, listing_target: str, target: str) -> None:
        super().__init__(
            f'the reviewed listing was taken on {listing_target}, but the config names '
            f'{target}; run list against {target} and review that listing'
        )


def require_same_target(listing: Listing, target: str) -> Listing:
    if listing.target != target:
        raise WrongTarget(listing.target, target)
    return listing


class ListingDeletesKeptGraphs(ValueError):
    """A reviewed listing whose delete section names graphs the allowlist does not."""

    def __init__(self, names: tuple[str, ...]) -> None:
        super().__init__(
            f'the reviewed listing names {", ".join(map(repr, names))} under delete, but only '
            'graphs on LITTER_PATTERNS are ever deleted; list again and review that listing'
        )
        self.names = names


def require_deletable(listing: Listing, target: str) -> Listing:
    """A listing taken on *target* whose delete section names only litter."""
    require_same_target(listing, target)
    kept = tuple(name for name in listing.delete if not is_litter(name))
    if kept:
        raise ListingDeletesKeptGraphs(kept)
    return listing


class _RecordFile:
    """The delete run's record on disk, rewritten whole at each step of the run.

    ``latest`` is set before each write, so it names every delete even when
    writing it failed.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.latest: CleanupRecord | None = None

    def write(self, record: CleanupRecord) -> None:
        self.latest = record
        atomic_write_text(self.path, json.dumps(record.to_json(), indent=2) + '\n')

    def write_failed(self) -> None:
        if self.latest is not None:
            with contextlib.suppress(OSError):
                self.write(replace(self.latest, outcome=Exit.FAILED))

    def deleted(self) -> tuple[str, ...]:
        return () if self.latest is None else self.latest.deleted


def run_delete(
    server: GraphServer,
    reviewed: Listing,
    record_path: Path,
    *,
    target: str,
    pace_s: float,
    sleep: Callable[[float], None],
) -> Exit:
    try:
        require_deletable(reviewed, target)
    except (WrongTarget, ListingDeletesKeptGraphs) as error:
        return _report(Exit.REFUSED, error)
    record_file = _RecordFile(record_path)
    try:
        record = execute(
            server,
            plan_cleanup(server.graph_names()),
            reviewed,
            pace_s=pace_s,
            sleep=sleep,
            on_progress=record_file.write,
        )
        record_file.write(record)
    except (RedisError, OSError) as error:
        return _report_failed(record_file, target, f'{type(error).__name__}: {error}')
    except KeyboardInterrupt:
        return _report_failed(record_file, target, 'interrupted')
    return _report_outcome(record, record_path)


def _run_summary(deleted: tuple[str, ...], target: str, record_path: Path) -> str:
    return f'deleted {len(deleted)} graphs on {target}; record at {record_path}'


def _report_failed(record_file: _RecordFile, target: str, cause: str) -> Exit:
    record_file.write_failed()
    summary = _run_summary(record_file.deleted(), target, record_file.path)
    return _report(Exit.FAILED, f'{cause}; {summary}')


def _report_outcome(record: CleanupRecord, record_path: Path) -> Exit:
    summary = _run_summary(record.deleted, record.target, record_path)
    if record.outcome is Exit.OK:
        print(f'ok: {summary}')
        return Exit.OK
    if record.outcome is Exit.SERVER_RESTARTED:
        return _report(
            Exit.SERVER_RESTARTED,
            f'run_id changed from {record.run_id_before} to {record.run_id_after}; {summary}',
        )
    if record.outcome is Exit.KEPT_GRAPH_SHRANK:
        shrank = ', '.join(f'{s.graph} {s.before}->{s.after}' for s in record.shrank)
        return _report(Exit.KEPT_GRAPH_SHRANK, f'kept graphs shrank: {shrank}; {summary}')
    return _report(Exit.FAILED, f'the run finished without an outcome; {summary}')


def listing_summary(listing: Listing) -> list[str]:
    return [
        f'target {listing.target}, run_id {listing.run_id}, taken at {listing.taken_at}',
        f'delete {len(listing.delete)} graphs:',
        *(
            f'{sum(1 for name in listing.delete if pattern.fullmatch(name)):>7}  {pattern.pattern}'
            for pattern in LITTER_PATTERNS
        ),
        f'keep {len(listing.keep)} graphs:',
        *(f'{count:>7}  {name}' for name, count in listing.keep.items()),
    ]


def resolve_target() -> tuple[FalkorEndpoint, str | None]:
    """The FalkorDB the config names (FALKORDB_URI, else config.yaml), and its password."""
    configured = FusedMemoryConfig().graphiti.falkordb
    return FalkorEndpoint.from_uri(configured.uri), configured.password or None


@contextlib.contextmanager
def _connected(endpoint: FalkorEndpoint, password: str | None) -> Iterator[FalkorGraphServer]:
    client = FalkorDB(host=endpoint.host, port=endpoint.port, password=password)
    try:
        yield FalkorGraphServer(client)
    finally:
        client.close()


def _against(
    endpoint: FalkorEndpoint, password: str | None, command: Callable[[GraphServer], int]
) -> int:
    try:
        with _connected(endpoint, password) as server:
            return command(server)
    except (RedisError, OSError) as error:
        return _report(Exit.FAILED, f'{type(error).__name__}: {error}')


def _list(server: GraphServer, target: str, out: Path) -> int:
    listing = build_listing(target, server)
    write_listing(out, listing)
    print('\n'.join(listing_summary(listing)))
    print(f'listing written to {out}; review it before delete --reviewed {out}')
    return Exit.OK


def build_parser() -> LoudArgumentParser:
    parser = LoudArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    commands = parser.add_subparsers(dest='command', required=True)
    listing = commands.add_parser(
        'list', help='write the litter and kept graphs, with node counts, for review; deletes nothing'
    )
    listing.add_argument('--out', type=Path, required=True, metavar='PATH')
    delete = commands.add_parser(
        'delete', help='delete the litter a reviewed listing names, one graph at a time'
    )
    delete.add_argument('--reviewed', type=Path, required=True, metavar='PATH')
    delete.add_argument('--record', type=Path, required=True, metavar='PATH')
    delete.add_argument(
        '--pace-seconds', type=float, default=10.0,
        help='pause between deletes (default: %(default)s)',
    )
    return parser


def _report(status: Exit, error: object) -> Exit:
    print(f'{status.name.lower()}: {error}', file=sys.stderr)
    return status


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        endpoint, password = resolve_target()
    except ValueError as error:
        return _report(Exit.FAILED, error)
    if args.command == 'list':
        return _against(endpoint, password, lambda server: _list(server, endpoint.uri, args.out))
    try:
        reviewed = require_deletable(read_listing(args.reviewed), endpoint.uri)
    except (UnreadableListing, WrongTarget, ListingDeletesKeptGraphs) as error:
        return _report(Exit.REFUSED, error)
    return _against(
        endpoint,
        password,
        lambda server: run_delete(
            server,
            reviewed,
            args.record,
            target=endpoint.uri,
            pace_s=args.pace_seconds,
            sleep=time.sleep,
        ),
    )


if __name__ == '__main__':
    sys.exit(run_cli(main))
