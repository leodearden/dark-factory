#!/usr/bin/env python3
"""Freeze the fresh write-triage population with its production slates.

Task 6151 (π), plans/write-triage-flip-readiness-prd.md §11 D14: every Mem0
write in the three Mem0-primary categories of each population project, created
on or after :data:`POPULATION_SINCE` and no later than the moment the freeze
starts, paired with the slate and band production retrieval gives it from the
live store. The snapshot is the input of the judge arms
(``run_write_triage_population_arms.py``) and is written under a gitignored
``--out-root``.

READ-ONLY against the store: it scrolls, searches and reads points, and
writes nothing but the snapshot files.

Usage
-----
  uv run python scripts/freeze_write_triage_population.py \\
      --out-root /home/leo/src/dark-factory/data

``--out-root`` is required and never derived: the snapshot must outlive the
worktree that froze it, so point it at the MAIN checkout's gitignored
``data/``. The retrieval loop is sequential, one search per write, and takes
roughly 5–15 minutes for ~1,500 writes; run it detached
(``setsid … > log 2>&1``) and poll the log.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import importlib.util
import json
import os
import sys
import types
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

from fused_memory.models.enums import MEM0_PRIMARY
from fused_memory.models.scope import Scope
from fused_memory.reconciliation.prompts import (
    FLAG_FOR_STAGE2_MARKER_KIND,
    STAGE2_SUPPRESS_GUARD_KIND,
)
from fused_memory.server.write_triage import (
    declares_attach_keys,
    resolve_bands,
    resolve_candidate_k,
)

_SCRIPTS = Path(__file__).resolve().parent


def _load_script(path: Path, mod_name: str) -> types.ModuleType:
    """Load a ``scripts/`` sibling by path, cached in ``sys.modules``."""
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_retrieval = _load_script(
    _SCRIPTS / 'eval_write_triage_retrieval.py', 'eval_write_triage_retrieval',
)

POPULATION_SINCE = datetime(2026, 9, 29, tzinfo=UTC)
POPULATION_PROJECTS = ('dark_factory', 'reify')
POPULATION_CATEGORIES = tuple(sorted(category.value for category in MEM0_PRIMARY))

#: 2 split schema 1's mixed ``excluded`` block by unit (see :func:`exclusions`).
SNAPSHOT_SCHEMA_VERSION = 2
SNAPSHOT_DIR_PREFIX = 'write-triage-population-'
SNAPSHOT_NAME = 'snapshot.json'
SNAPSHOT_DIGEST_NAME = 'snapshot.sha256'

#: The boolean flags the reconciliation stage prompts (reconciliation/prompts/)
#: wrote on their marker records before χ gave those markers a declared kind.
_RECON_MARKER_FLAGS = ('flag_for_stage2', 'stage2_suppress')
_RECON_MARKER_KINDS = frozenset({FLAG_FOR_STAGE2_MARKER_KIND, STAGE2_SUPPRESS_GUARD_KIND})


class PopulationWrite(TypedDict):
    """One frozen write, before its slate is attached."""

    memory_id: str
    project_id: str
    category: str
    created_at: str
    content: str
    metadata: dict[str, Any]
    recon_marker: bool
    declares_attach_keys: bool


def is_recon_marker(metadata: Mapping[str, Any]) -> bool:
    """Whether *metadata* is a reconciliation marker, by its flag or its declared kind."""
    return (
        any(metadata.get(flag) is True for flag in _RECON_MARKER_FLAGS)
        or metadata.get('kind') in _RECON_MARKER_KINDS
    )


def parse_created_at(value: object) -> datetime | None:
    """*value* as an aware instant, or ``None`` when it is naive, absent or unparseable."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _population_write(
    record: Mapping[str, Any], project_id: str, category: str, stored: Mapping[str, Any],
) -> PopulationWrite:
    metadata = dict(stored.get('metadata') or {})
    return PopulationWrite(
        memory_id=str(record['id']),
        project_id=project_id,
        category=category,
        created_at=record['created_at'],
        content=str(stored.get('content') or ''),
        metadata=metadata,
        recon_marker=is_recon_marker(metadata),
        declares_attach_keys=declares_attach_keys(metadata),
    )


async def enumerate_population(
    memory_service: Any,
    *,
    projects: Sequence[str],
    since: datetime,
    frozen_at: datetime,
) -> tuple[list[PopulationWrite], dict[str, int]]:
    """Every write of *projects* created in ``[since, frozen_at]``, and what was excluded.

    Each project's Mem0-primary categories are scrolled exhaustively; content
    and metadata come from the public point read. A record whose
    ``created_at`` is not an aware timestamp is ``undated`` and one gone
    before its read is ``vanished``: both are excluded and counted, never
    guessed. Ordered by project, then instant, then id.
    """
    excluded = {'undated': 0, 'vanished': 0}
    keyed: list[tuple[tuple[str, datetime, str], PopulationWrite]] = []
    for project_id in projects:
        scope = Scope(project_id=project_id)
        for category in POPULATION_CATEGORIES:
            records = memory_service.mem0.scroll_all_by_metadata(scope, {'category': category})
            async for record in records:
                instant = parse_created_at(record.get('created_at'))
                if instant is None:
                    excluded['undated'] += 1
                    continue
                if not since <= instant <= frozen_at:
                    continue
                stored = await memory_service.get_memory_by_id(project_id, str(record['id']))
                if stored is None:
                    excluded['vanished'] += 1
                    continue
                write = _population_write(record, project_id, category, stored)
                keyed.append(((project_id, instant, write['memory_id']), write))
    keyed.sort(key=lambda pair: pair[0])
    return [write for _, write in keyed], excluded


# --- freeze ------------------------------------------------------------------

def _by_project(writes: Sequence[PopulationWrite]) -> dict[str, list[PopulationWrite]]:
    grouped: dict[str, list[PopulationWrite]] = {}
    for write in writes:
        grouped.setdefault(write['project_id'], []).append(write)
    return grouped


def _refuse_degraded(retrievals: Mapping[str, Mapping[str, Any]]) -> None:
    degraded = sorted(memory_id for memory_id, r in retrievals.items() if r['degraded'])
    if degraded:
        raise ValueError(
            f'{len(degraded)} retrieval(s) came back degraded, so nothing is frozen: '
            f'a degraded slate is an infra blip, not a production band. '
            f'Writes: {", ".join(degraded)}',
        )


def _without_own_children(
    memory_id: str, retrieval: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    """*retrieval* minus the write's own children, and how many there were.

    Production never sees a write's own later children, and keeping one would
    let the judge name it, hoisting to a self-pair.
    """
    kept = [
        row for row in retrieval['results']
        if _retrieval.normalize(row)['canonical_id'] != memory_id
    ]
    return {**retrieval, 'results': kept}, len(retrieval['results']) - len(kept)


def _frozen_write(
    write: PopulationWrite, slate: Any, rows: Sequence[Any], own_children_dropped: int,
) -> dict[str, Any]:
    created_at = {row.id: row.created_at for row in rows}
    return {
        **write,
        'band': slate.band,
        'band_winner_id': slate.attach_target_id,
        'similarity': slate.similarity,
        'retrieved_count': slate.retrieved_count,
        'self_retrieved': slate.self_retrieved,
        'own_children_dropped': own_children_dropped,
        'candidates': [
            {**candidate, 'created_at': created_at.get(candidate['memory_id'])}
            for candidate in slate.candidates
        ],
    }


async def _off_slate_targets(
    memory_service: Any, project_id: str, frozen: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """The hoisted canonical of every candidate whose canonical is not on its slate."""
    targets: dict[str, dict[str, Any]] = {}
    for write in frozen:
        on_slate = {candidate['memory_id'] for candidate in write['candidates']}
        for candidate in write['candidates']:
            target_id = candidate['canonical_id']
            if target_id in on_slate or target_id in targets:
                continue
            stored = await memory_service.get_memory_by_id(project_id, target_id)
            targets[target_id] = {
                'project_id': project_id,
                'content': stored['content'] if stored is not None else None,
            }
    return targets


async def freeze_population(
    memory_service: Any,
    writes: Sequence[PopulationWrite],
    *,
    k: int,
    t_high: float | None,
    t_low: float | None,
) -> dict[str, Any]:
    """Each write with the slate and band production retrieval gives it now.

    One shipped retrieval per write (own id dropped), the write's own children
    dropped, then the shipped banding with nothing comparable trimmed below
    *k*. Refuses (``ValueError``) if any retrieval is degraded. Returns
    ``writes``, the off-slate hoisted ``targets`` and ``slate_rows_dropped``,
    the slate rows (not writes) left out: each write's ``self`` row and its
    ``own_children``.
    """
    frozen: list[dict[str, Any]] = []
    targets: dict[str, dict[str, Any]] = {}
    slate_rows_dropped = {'self': 0, 'own_children': 0}
    for project_id, project_writes in _by_project(writes).items():
        records = [
            {'memory_id': w['memory_id'], 'content': w['content'], 'cluster_id': w['memory_id']}
            for w in project_writes
        ]
        retrievals = await _retrieval.prefetch_retrievals(
            memory_service, records, project_id=project_id, k=k,
        )
        _refuse_degraded(retrievals)
        pruned = {mid: _without_own_children(mid, r) for mid, r in retrievals.items()}
        kept = {mid: retrieval for mid, (retrieval, _) in pruned.items()}
        slates = _retrieval.retrieved_slates(
            records, kept, t_high=t_high, t_low=t_low, judge_candidate_count=k,
        )
        project_frozen = [
            _frozen_write(
                write, slate, kept[write['memory_id']]['results'], pruned[write['memory_id']][1],
            )
            for write, slate in zip(project_writes, slates, strict=True)
        ]
        slate_rows_dropped['own_children'] += sum(w['own_children_dropped'] for w in project_frozen)
        slate_rows_dropped['self'] += sum(1 for w in project_frozen if w['self_retrieved'])
        targets |= await _off_slate_targets(memory_service, project_id, project_frozen)
        frozen.extend(project_frozen)
    return {'writes': frozen, 'targets': targets, 'slate_rows_dropped': slate_rows_dropped}


async def freeze_snapshot(
    memory_service: Any,
    *,
    projects: Sequence[str],
    since: datetime,
    frozen_at: datetime,
    k: int,
    t_high: float | None,
    t_low: float | None,
) -> dict[str, Any]:
    """The whole snapshot: the enumerated population, frozen, with how it was made."""
    writes, excluded_writes = await enumerate_population(
        memory_service, projects=projects, since=since, frozen_at=frozen_at,
    )
    frozen = await freeze_population(memory_service, writes, k=k, t_high=t_high, t_low=t_low)
    return {
        'schema_version': SNAPSHOT_SCHEMA_VERSION,
        'frozen_at': frozen_at.astimezone(UTC).isoformat(),
        'since': since.astimezone(UTC).isoformat(),
        'projects': list(projects),
        'categories': list(POPULATION_CATEGORIES),
        'candidate_k': k,
        't_high': t_high,
        't_low': t_low,
        'excluded_writes': excluded_writes,
        'slate_rows_dropped': frozen['slate_rows_dropped'],
        'writes': frozen['writes'],
        'targets': frozen['targets'],
    }


def exclusions(snapshot: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """What the freeze left out, by unit: whole ``excluded_writes``, and ``slate_rows_dropped``.

    Schema 1 recorded both units in one ``excluded`` block; it is split here,
    so every reader sees the current shape.
    """
    version = snapshot.get('schema_version')
    if version == 1:
        mixed = snapshot['excluded']
        return {
            'excluded_writes': {'undated': mixed['undated'], 'vanished': mixed['vanished']},
            'slate_rows_dropped': {
                'self': mixed['self_retrieved'], 'own_children': mixed['own_children_dropped'],
            },
        }
    if version != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f'snapshot schema_version {version} is not one this script reads')
    return {key: dict(snapshot[key]) for key in ('excluded_writes', 'slate_rows_dropped')}


# --- snapshot files ----------------------------------------------------------

def snapshot_dir(out_root: Path, frozen_at: datetime) -> Path:
    """The directory a population frozen at *frozen_at* lives in, named for its UTC date."""
    frozen_on = frozen_at.astimezone(UTC).date()
    return Path(out_root) / f'{SNAPSHOT_DIR_PREFIX}{frozen_on.isoformat()}'


def write_snapshot(out_root: Path, snapshot: Mapping[str, Any]) -> Path:
    """Write *snapshot* and its sha256 sidecar into a fresh :func:`snapshot_dir`.

    The directory must not exist yet (``FileExistsError``): a frozen
    population is never re-frozen in place. Returns the path of
    ``snapshot.json``.
    """
    directory = snapshot_dir(out_root, datetime.fromisoformat(snapshot['frozen_at']))
    directory.mkdir(parents=True, exist_ok=False)
    body = json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode('utf-8')
    path = directory / SNAPSHOT_NAME
    path.write_bytes(body)
    (directory / SNAPSHOT_DIGEST_NAME).write_text(hashlib.sha256(body).hexdigest() + '\n')
    return path


def load_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    """The snapshot at *path* and its sha256, refused if it no longer matches its sidecar."""
    path = Path(path)
    body = path.read_bytes()
    actual = hashlib.sha256(body).hexdigest()
    recorded = (path.parent / SNAPSHOT_DIGEST_NAME).read_text().strip()
    if actual != recorded:
        raise ValueError(
            f'{path} was edited after it was frozen: recorded sha256 {recorded}, '
            f'on disk {actual}',
        )
    return json.loads(body), actual


# --- CLI ---------------------------------------------------------------------

def _aware_datetime(text: str) -> datetime:
    parsed = parse_created_at(text)
    if parsed is None:
        raise argparse.ArgumentTypeError(f'{text!r} is not an ISO timestamp with an offset')
    return parsed


def _summary(snapshot: Mapping[str, Any], path: Path, sha256: str) -> dict[str, Any]:
    writes = snapshot['writes']

    def count(key: str) -> dict[str, int]:
        return dict(sorted(collections.Counter(w[key] for w in writes).items()))

    return {
        'n_writes': len(writes),
        'by_project': count('project_id'),
        'by_category': count('category'),
        'by_band': count('band'),
        'recon_marker_writes': sum(1 for w in writes if w['recon_marker']),
        **exclusions(snapshot),
        'path': str(path),
        'sha256': sha256,
    }


async def _freeze_live(args: argparse.Namespace) -> dict[str, Any]:
    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
    from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

    frozen_at = datetime.now(UTC)
    directory = snapshot_dir(args.out_root, frozen_at)
    if directory.exists():
        raise FileExistsError(
            f'{directory} already holds a population frozen today; it is never re-frozen '
            f'in place, so nothing was read from the store',
        )
    config = FusedMemoryConfig()
    configured = types.SimpleNamespace(config=config)
    t_high, t_low = resolve_bands(configured)
    memory = MemoryService(config)
    await memory.initialize()
    try:
        snapshot = await freeze_snapshot(
            memory, projects=args.projects, since=args.since, frozen_at=frozen_at,
            k=resolve_candidate_k(configured), t_high=t_high, t_low=t_low,
        )
    finally:
        await memory.close()
    path = write_snapshot(args.out_root, snapshot)
    _, sha256 = load_snapshot(path)
    return _summary(snapshot, path, sha256)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--out-root', dest='out_root', type=Path, required=True,
                        help='directory the dated snapshot directory is created in')
    parser.add_argument('--config', default=None,
                        help='path to a fused-memory config file (sets CONFIG_PATH)')
    parser.add_argument('--since', type=_aware_datetime, default=POPULATION_SINCE,
                        help=f'earliest created_at, with an offset '
                             f'(default: {POPULATION_SINCE.isoformat()})')
    parser.add_argument('--projects', nargs='+', default=list(POPULATION_PROJECTS),
                        help=f'projects to freeze (default: {" ".join(POPULATION_PROJECTS)})')
    args = parser.parse_args(argv)
    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)
    print(json.dumps(asyncio.run(_freeze_live(args)), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
