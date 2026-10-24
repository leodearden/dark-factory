#!/usr/bin/env python3
"""Rebuild the write-time slates of the 2026-10-19 soak's judge-band writes.

The soak kept no slate, band or cosine per write, so each one is rebuilt.
A write's slate is production retrieval (``write_triage.py::retrieve_candidates``)
re-run on the records that existed at the write's instant: the
:data:`CANDIDATE_K` nearest by cosine among its project's Mem0-primary,
unplanned records, cross-category, with children hoisted to their canonical.
The vectors are the soak's embedding space (text-embedding-3-small, 1536-d),
read from the 2026-10-19 dump or from live Qdrant, read-only. Production
searched without a time filter; the created-before mask stands in for the
records that did not exist yet. The band is re-decided by ``decide_band`` at
the soak's edges, and every text is the one mem0's history gives for the
write's instant. A record deleted since the write is counted, not restored:
history carries no project, so the count spans every project.

A write leaves the population when the rebuild cannot find its own vector
(unrecoverable), or when production judged it (an amended or contested
attach) but the rebuilt band is not the judge's (a leaver). Both are listed
in the header and never replaced. The sample is :func:`draw_live_sample`
over the rebuilt judge band.

Run from ``fused-memory/``::

    uv run python scripts/rebuild_write_triage_live_slates.py \\
        --journal S/duplicates/journal_add_memory.jsonl \\
        --attaches S/correctness/attaches_soak.json \\
        --attaches-axes S/team/A/attaches_axes.json \\
        --history-db ~/.mem0/history.db --vectors dump:data/qdrant-dump-2026-10-19 \\
        --out calibration/write_triage_eval_set_v2_live_slates.jsonl --total 400 --seed 7160

``--vectors`` is ``dump:<dir>`` (``<project>.npy`` + ``<project>.jsonl``) or
``qdrant:<url>`` (collections ``fused_<project>``). The output's first line is
a header; every following line is one sampled write in the frozen snapshot's
write shape, which ``run_write_triage_population_arms.py::top_targets`` reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fused_memory.models.enums import MEM0_PRIMARY, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.write_triage import (
    ATTACH_OWNED_KEYS,
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_JUDGE,
    OUTCOME_STORED,
    BandDecision,
    canonical_id_of,
    decide_band,
    declares_attach_keys,
)
from fused_memory.utils.script_loader import load_script

_SCRIPTS = Path(__file__).resolve().parent

if TYPE_CHECKING:
    import freeze_write_triage_population as _freeze
    import write_triage_rating_rounds as _rounds
else:
    _freeze = load_script(_SCRIPTS / 'freeze_write_triage_population.py',
                          'freeze_write_triage_population')
    _rounds = load_script(_SCRIPTS / 'write_triage_rating_rounds.py', 'write_triage_rating_rounds')

SOAK_EMBEDDING_DIMS = 1536
SOAK_T_HIGH = 0.8868282526657318
SOAK_T_LOW = 0.5229672433064231
CANDIDATE_K = 20
SOAK_PROJECTS = ('dark_factory', 'reify', 'solar_challenge', 'solar_challenge_platform',
                 'scoping_intake')
_SOURCES = frozenset({'mcp_tool', 'full_recon'})
_MEM0_PRIMARY = frozenset(category.value for category in MEM0_PRIMARY)
_JUDGED_OUTCOMES = frozenset({OUTCOME_AMENDED, OUTCOME_CONTESTED})
"""Production outcomes only the judge gives; a restated attach may have been deterministic."""
_TEXT_FIELDS = frozenset({'data', 'hash'})


@dataclass(frozen=True)
class Window:
    """The instants a soak write may carry: from *start* on, and before *end* when there is one."""

    start: datetime
    end: datetime | None

    def __contains__(self, instant: datetime) -> bool:
        return instant >= self.start and (self.end is None or instant < self.end)


SOAK_WINDOW = Window(start=datetime(2026, 10, 11, 13, 35, tzinfo=UTC), end=None)
"""The soak ran from the flip (task 3169) to the bundle's end."""


@dataclass(frozen=True)
class SoakWrite:
    """One write triage decided during the soak, and what production did with it."""

    memory_id: str
    project_id: str
    category: str
    created_at: str
    instant: datetime
    content: str
    metadata: Mapping[str, Any]
    production_outcome: str
    production_target: str | None


def soak_writes(
    journal_rows: Iterable[Mapping[str, Any]], attaches: Iterable[Mapping[str, Any]], window: Window,
) -> list[SoakWrite]:
    """The journal's writes triage decided in *window*, in journal order.

    A write is kept when it came from an agent or a full reconciliation, is
    Mem0-primary, has an intent and exactly one memory id, and is not
    caller-owned: an attach the caller made, or caller metadata declaring
    attach keys. The journal records a triage attach's metadata with the
    attach keys triage wrote, so its caller metadata is that without them.
    """
    by_intent = {attach['intent_id']: attach for attach in attaches}
    writes = []
    for row in journal_rows:
        write = _soak_write(row, by_intent.get(row.get('intent_id')), window)
        if write is not None:
            writes.append(write)
    return writes


def _soak_write(
    row: Mapping[str, Any], attach: Mapping[str, Any] | None, window: Window,
) -> SoakWrite | None:
    memory_ids = (row.get('result_summary') or {}).get('memory_ids') or []
    instant = _freeze.parse_created_at(row.get('intent_created_at'))
    triage_attach = attach is not None and attach['triage_sig'] is True
    metadata = {
        key: value for key, value in (row.get('intent_metadata') or {}).items()
        if not (triage_attach and key in ATTACH_OWNED_KEYS)
    }
    triaged = (
        bool(row.get('intent_id')) and len(memory_ids) == 1
        and instant is not None and instant in window
        and row.get('source') in _SOURCES and row.get('intent_category') in _MEM0_PRIMARY
        and not declares_attach_keys(metadata)
        and (attach is None or triage_attach)
    )
    if not triaged or instant is None:
        return None
    return SoakWrite(
        memory_id=memory_ids[0], project_id=row['project_id'], category=row['intent_category'],
        created_at=row['intent_created_at'], instant=instant, content=row['intent_content'],
        metadata=metadata,
        production_outcome=OUTCOME_STORED if attach is None else attach['outcome'],
        production_target=None if attach is None else attach['meta']['parent_id'],
    )


class VectorTable:
    """One project's records in the soak's embedding space, searched by brute-force cosine."""

    def __init__(
        self, ids: Sequence[str], vectors: np.ndarray, payloads: Sequence[Mapping[str, Any]],
    ) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        width = matrix.shape[-1] if matrix.ndim else 0
        if matrix.ndim != 2 or width != SOAK_EMBEDDING_DIMS:
            raise ValueError(
                f'vectors are {width}-d, not the soak embedding space ({SOAK_EMBEDDING_DIMS}-d):'
                ' the write-time slates cannot be rebuilt from them'
            )
        if not len(ids) == len(payloads) == len(matrix):
            raise ValueError(f'{len(ids)} ids, {len(payloads)} payloads and {len(matrix)} vectors')
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._matrix = matrix / np.where(norms == 0, 1, norms)
        self.ids = tuple(str(point_id) for point_id in ids)
        self._payloads = tuple(payloads)
        self._index = {point_id: index for index, point_id in enumerate(self.ids)}
        self._created = np.array([_epoch(payload.get('created_at')) for payload in payloads])
        self._searchable = np.array([
            payload.get('category') in _MEM0_PRIMARY and payload.get('planned') is not True
            for payload in payloads
        ], dtype=bool)

    def vector(self, memory_id: str) -> np.ndarray | None:
        index = self._index.get(memory_id)
        return None if index is None else self._matrix[index]

    def payload(self, memory_id: str) -> Mapping[str, Any] | None:
        index = self._index.get(memory_id)
        return None if index is None else self._payloads[index]

    def nearest(self, memory_id: str, before: datetime, k: int) -> list[tuple[str, float]]:
        """The *k* searchable records nearest *memory_id* created strictly *before*: (id, cosine), descending."""
        query = self._index[memory_id]
        mask = self._searchable & (self._created < before.timestamp())
        mask[query] = False
        cosines = self._matrix @ self._matrix[query]
        indices = np.flatnonzero(mask)
        ranked = indices[np.argsort(-cosines[indices], kind='stable')[:k]]
        return [(self.ids[index], float(cosines[index])) for index in ranked]


def _epoch(created_at: object) -> float:
    """*created_at* as epoch seconds; NaN when it is not an aware instant, so no comparison holds."""
    instant = _freeze.parse_created_at(created_at)
    return float('nan') if instant is None else instant.timestamp()


def load_dump(directory: Path, project: str) -> VectorTable:
    """*project*'s table from a dump's ``<project>.npy`` and ``<project>.jsonl``."""
    rows = [json.loads(line) for line in (directory / f'{project}.jsonl').read_text().splitlines()
            if line.strip()]
    vectors = np.load(directory / f'{project}.npy', allow_pickle=False)
    return VectorTable([row['id'] for row in rows], vectors, [row['payload'] for row in rows])


Post = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


def load_qdrant(url: str, collection: str, *, post: Post | None = None, page: int = 512) -> VectorTable:
    """*collection*'s table from Qdrant, scrolled read-only with vectors and payloads."""
    send = post or _http_post
    points: list[Mapping[str, Any]] = []
    offset: Any = None
    while True:
        body: dict[str, Any] = {'limit': page, 'with_payload': True, 'with_vector': True}
        if offset is not None:
            body['offset'] = offset
        result = send(f'{url.rstrip("/")}/collections/{collection}/points/scroll', body)['result']
        points += result['points']
        offset = result.get('next_page_offset')
        if offset is None:
            break
    vectors = (np.array([point['vector'] for point in points], dtype=np.float32) if points
               else np.zeros((0, SOAK_EMBEDDING_DIMS), dtype=np.float32))
    return VectorTable([point['id'] for point in points], vectors,
                       [point.get('payload') or {} for point in points])


def _http_post(url: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method='POST',
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


class Unrecoverable(ValueError):
    """A soak write whose slate cannot be rebuilt."""


def write_time_slate(
    write: SoakWrite, table: VectorTable, history: _rounds.MemoryHistory, k: int = CANDIDATE_K,
) -> dict[str, Any]:
    """*write* with its slate as it was at its instant, in the frozen snapshot's write shape."""
    neighbours = _neighbours(write, table, k)
    candidates = [_candidate(table, memory_id, cosine, write.instant, history)
                  for memory_id, cosine in neighbours]
    decision = _decision(table, neighbours)
    on_slate = {candidate['memory_id'] for candidate in candidates}
    hoisted = dict.fromkeys(candidate['canonical_id'] for candidate in candidates)
    return {
        'memory_id': write.memory_id, 'project_id': write.project_id, 'category': write.category,
        'created_at': write.created_at, 'content': write.content, 'metadata': dict(write.metadata),
        'band': decision.outcome, 'band_winner_id': decision.canonical_id,
        'similarity': decision.similarity, 'candidates': candidates,
        'targets': {canonical: _target(table, canonical, write.instant, history)
                    for canonical in hoisted if canonical not in on_slate},
        'production_outcome': write.production_outcome,
        'production_target': write.production_target,
    }


def _neighbours(write: SoakWrite, table: VectorTable, k: int) -> list[tuple[str, float]]:
    if table.vector(write.memory_id) is None:
        raise Unrecoverable(f'write {write.memory_id}: its own vector is not in the vector table')
    return table.nearest(write.memory_id, write.instant, k)


def _decision(table: VectorTable, neighbours: Sequence[tuple[str, float]]) -> BandDecision:
    return decide_band(
        [_memory_result(memory_id, _metadata(table, memory_id, cosine))
         for memory_id, cosine in neighbours],
        t_high=SOAK_T_HIGH, t_low=SOAK_T_LOW,
    )


def _metadata(table: VectorTable, memory_id: str, cosine: float) -> dict[str, Any]:
    payload = table.payload(memory_id) or {}
    return {**{key: value for key, value in payload.items() if key not in _TEXT_FIELDS},
            'store_score': cosine}


def _memory_result(memory_id: str, metadata: Mapping[str, Any]) -> MemoryResult:
    return MemoryResult(id=memory_id, content='', source_store=SourceStore.mem0,
                        metadata=dict(metadata))


def _candidate(
    table: VectorTable, memory_id: str, cosine: float, instant: datetime,
    history: _rounds.MemoryHistory,
) -> dict[str, Any]:
    payload = table.payload(memory_id) or {}
    metadata = _metadata(table, memory_id, cosine)
    return {
        'memory_id': memory_id,
        'canonical_id': canonical_id_of(_memory_result(memory_id, metadata)),
        'category': payload.get('category'),
        'store_score': cosine,
        'metadata': metadata,
        'created_at': payload.get('created_at'),
        'content': _text_at(history, memory_id, instant, payload.get('data') or ''),
        'edited_after_write': history.edited_after(memory_id, instant),
    }


def _target(
    table: VectorTable, memory_id: str, instant: datetime, history: _rounds.MemoryHistory,
) -> dict[str, Any]:
    """A hoisted canonical off the slate; one absent from the table is noted, its text not restored."""
    payload = table.payload(memory_id)
    return {
        'content': None if payload is None else _text_at(
            history, memory_id, instant, payload.get('data') or ''),
        'created_at': None if payload is None else payload.get('created_at'),
        'edited_after_write': history.edited_after(memory_id, instant),
    }


def _text_at(
    history: _rounds.MemoryHistory, memory_id: str, instant: datetime, current: str,
) -> str | None:
    text = history.text_at(memory_id, instant, current)
    return None if text is None else _rounds.capped(text)


@dataclass(frozen=True)
class LiveSample:
    """The drawn write ids in draw order, how many each project gave, and each project's population."""

    drawn: tuple[str, ...]
    allocation: Mapping[str, int]
    strata: Mapping[str, int]


def draw_live_sample(
    judge_band_by_project: Mapping[str, Iterable[str]], *, total: int, seed: int | str,
) -> LiveSample:
    """*total* writes dealt round-robin over the projects, each in its own seeded order.

    Allocation is equal, capped at each project's population, and the
    remainder goes to the projects with writes left. A larger *total* extends
    a smaller one's draw.
    """
    orders = {project: _seeded_order(ids, f'{seed}:{project}')
              for project, ids in sorted(judge_band_by_project.items())}
    population = sum(len(order) for order in orders.values())
    if total > population:
        raise ValueError(f'a sample of {total} exceeds the judge-band population of {population}')
    drawn: list[str] = []
    allocation = dict.fromkeys(orders, 0)
    rank = 0
    while len(drawn) < total:
        for project, order in orders.items():
            if rank < len(order) and len(drawn) < total:
                drawn.append(order[rank])
                allocation[project] += 1
        rank += 1
    return LiveSample(tuple(drawn), allocation, {project: len(order) for project, order in orders.items()})


def _seeded_order(ids: Iterable[str], seed: str) -> list[str]:
    ordered = sorted(ids)
    return random.Random(seed).sample(ordered, len(ordered))


@dataclass(frozen=True)
class Rebuild:
    """The sampled rows in draw order, the draw, and the writes that left the population."""

    rows: list[dict[str, Any]]
    strata: Mapping[str, int]
    allocation: Mapping[str, int]
    leavers: list[dict[str, Any]]
    unrecoverable: list[dict[str, Any]]
    winners: Mapping[str, str | None]


def rebuild(
    writes: Iterable[SoakWrite], tables: Mapping[str, VectorTable], history: _rounds.MemoryHistory,
    *, total: int, seed: int | str, k: int = CANDIDATE_K,
) -> Rebuild:
    """Re-decide every write's band, draw from the judge band, and rebuild the drawn slates.

    ``winners`` maps every recoverable write to its rebuilt band winner's canonical.
    """
    judge_band: defaultdict[str, list[str]] = defaultdict(list)
    judged: dict[str, SoakWrite] = {}
    leavers: list[dict[str, Any]] = []
    unrecoverable: list[dict[str, Any]] = []
    winners: dict[str, str | None] = {}
    for write in writes:
        table = tables.get(write.project_id)
        try:
            if table is None:
                raise Unrecoverable(f'write {write.memory_id}: no vector table for {write.project_id}')
            decision = _decision(table, _neighbours(write, table, k))
        except Unrecoverable as exc:
            unrecoverable.append({'memory_id': write.memory_id, 'project_id': write.project_id,
                                  'reason': str(exc)})
            continue
        winners[write.memory_id] = decision.canonical_id
        if decision.outcome == OUTCOME_JUDGE:
            judge_band[write.project_id].append(write.memory_id)
            judged[write.memory_id] = write
        elif write.production_outcome in _JUDGED_OUTCOMES:
            leavers.append({'memory_id': write.memory_id, 'project_id': write.project_id,
                            'production_outcome': write.production_outcome,
                            'band': decision.outcome})
    sample = draw_live_sample(judge_band, total=total, seed=seed)
    rows = [write_time_slate(judged[memory_id], tables[judged[memory_id].project_id], history, k=k)
            for memory_id in sample.drawn]
    return Rebuild(rows=rows, strata=sample.strata, allocation=sample.allocation, leavers=leavers,
                   unrecoverable=unrecoverable, winners=winners)


# --------------------------------------------------------------------------
# The CLI
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _VectorSource:
    kind: str
    location: str

    def load(self, projects: Sequence[str]) -> tuple[dict[str, VectorTable], dict[str, Any]]:
        """Each project's table, and the header's account of where they came from."""
        if self.kind == 'dump':
            directory = Path(self.location)
            tables = {project: load_dump(directory, project) for project in projects}
            files = [f'{project}{suffix}' for project in projects for suffix in ('.npy', '.jsonl')]
            return tables, {'kind': 'dump', 'path': str(directory),
                            'sha256': {name: _sha256(directory / name) for name in files},
                            'points': {project: len(table.ids) for project, table in tables.items()}}
        collections = {project: f'fused_{project}' for project in projects}
        tables = {project: load_qdrant(self.location, name) for project, name in collections.items()}
        return tables, {'kind': 'qdrant', 'url': self.location, 'collections': collections,
                        'points': {project: len(table.ids) for project, table in tables.items()}}


def _vector_source(value: str) -> _VectorSource:
    kind, _, location = value.partition(':')
    if kind not in ('dump', 'qdrant') or not location:
        raise argparse.ArgumentTypeError(f'{value!r} is neither dump:<dir> nor qdrant:<url>')
    return _VectorSource(kind, location)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _deleted_alive_at(
    history_db: Path, tables: Mapping[str, VectorTable], instants: Mapping[str, datetime],
) -> dict[str, int]:
    """Per write, the records history deleted after its instant that existed before it and no table holds."""
    present = {point_id for table in tables.values() for point_id in table.ids}
    with _rounds.read_only(history_db) as connection:
        events = connection.execute(
            "SELECT memory_id, created_at, updated_at FROM history WHERE event = 'DELETE'"
        ).fetchall()
    spans = np.array([
        (_epoch(created), _epoch(deleted)) for memory_id, created, deleted in events
        if memory_id not in present
    ], dtype=float).reshape(-1, 2)
    return {
        memory_id: int(np.sum((spans[:, 0] < instant.timestamp()) & (spans[:, 1] > instant.timestamp())))
        for memory_id, instant in instants.items()
    }


def _team_a(path: Path | None, rebuilt: Rebuild, tables: Mapping[str, VectorTable],
            ) -> dict[str, Any] | None:
    """How the rebuilt band winners compare with team A's reconstruction, and its top-5 ids now absent."""
    if path is None:
        return None
    rows = json.loads(path.read_text())
    compared = [row for row in rows if row.get('child_id') in rebuilt.winners and 'slate20' in row]
    missing = {
        row['child_id']: sum(tables[row['project']].payload(entry['id']) is None
                             for entry in row['slate20'][:5])
        for row in compared
    }
    return {
        'compared': len(compared),
        'winner_matches_team_a': sum(
            rebuilt.winners[row['child_id']] == row.get('recon_winner_canon') for row in compared),
        'winner_matches_production_parent': sum(
            rebuilt.winners[row['child_id']] == row.get('parent_id') for row in compared),
        'top5_absent_from_table': {child: count for child, count in sorted(missing.items()) if count},
    }


def _refuse_inside_bundle(parser: argparse.ArgumentParser, out: Path, inputs: Sequence[Path]) -> None:
    parents = [path.resolve().parent for path in inputs]
    bundle = Path(os.path.commonpath(parents))
    roots = [bundle] if bundle.parent != bundle else parents
    resolved = out.resolve()
    for root in roots:
        if resolved == root or root in resolved.parents:
            parser.error(f'--out {resolved} is inside the read-only bundle {root}')


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument('--journal', type=Path, required=True, metavar='PATH',
                        help="the soak bundle's duplicates/journal_add_memory.jsonl (read-only)")
    parser.add_argument('--attaches', type=Path, required=True, metavar='PATH',
                        help="the soak bundle's correctness/attaches_soak.json (read-only)")
    parser.add_argument('--attaches-axes', type=Path, metavar='PATH',
                        help="team A's attaches_axes.json, to compare band winners against")
    parser.add_argument('--history-db', type=Path, required=True, metavar='PATH',
                        help="mem0's history.db (read-only)")
    parser.add_argument('--vectors', type=_vector_source, required=True,
                        metavar='dump:DIR|qdrant:URL')
    parser.add_argument('--projects', nargs='+', default=list(SOAK_PROJECTS), metavar='PROJECT')
    parser.add_argument('--out', type=Path, required=True, metavar='PATH')
    parser.add_argument('--total', type=int, default=400)
    parser.add_argument('--seed', type=int, default=7160)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _refuse_inside_bundle(parser, args.out, [args.journal, args.attaches])
    history_db = args.history_db.expanduser()
    journal = [json.loads(line) for line in args.journal.read_text().splitlines() if line.strip()]
    writes = [write for write in soak_writes(journal, json.loads(args.attaches.read_text()), SOAK_WINDOW)
              if write.project_id in args.projects]
    tables, source = args.vectors.load(args.projects)
    rebuilt = rebuild(writes, tables, _rounds.MemoryHistory.read(history_db),
                      total=args.total, seed=args.seed)
    instants = {write.memory_id: write.instant for write in writes}
    header = {
        'built_at': datetime.now(UTC).isoformat(),
        'source': source,
        't_high': SOAK_T_HIGH, 't_low': SOAK_T_LOW, 'candidate_k': CANDIDATE_K, 'seed': args.seed,
        'window': {'start': SOAK_WINDOW.start.isoformat(), 'end': None},
        'eligible_writes': _by_project(write.project_id for write in writes),
        'strata': dict(rebuilt.strata),
        'allocation': dict(rebuilt.allocation),
        'leavers': rebuilt.leavers,
        'unrecoverable': rebuilt.unrecoverable,
        'edited_candidates': {row['memory_id']: sum(c['edited_after_write'] for c in row['candidates'])
                              for row in rebuilt.rows},
        'deleted_candidates': _deleted_alive_at(
            history_db, tables, {row['memory_id']: instants[row['memory_id']] for row in rebuilt.rows}),
        'team_a': _team_a(args.attaches_axes, rebuilt, tables),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(json.dumps(line, sort_keys=True) + '\n'
                                for line in [header, *rebuilt.rows]))
    print(json.dumps({key: header[key] for key in ('strata', 'allocation')}
                     | {'leavers': len(rebuilt.leavers), 'unrecoverable': len(rebuilt.unrecoverable),
                        'rows': len(rebuilt.rows)}, indent=2))
    return 0


def _by_project(projects: Iterable[str]) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for project in projects:
        counts[project] += 1
    return dict(sorted(counts.items()))


if __name__ == '__main__':
    sys.exit(main())
