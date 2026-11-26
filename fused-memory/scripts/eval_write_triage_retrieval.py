#!/usr/bin/env python3
"""The LIVE-store edge of the judge eval: production retrieval and banding.

The seeded slate (``eval_write_triage_cases.py::seeded_plan``) is a
CONSTRUCTION — the correct canonical hoisted to position 0 with a synthetic
top score. This module is the measured alternative: it drives the shipped
``retrieve_candidates``, ``decide_band``, ``route_band`` and
``select_judge_candidates`` so a case's slate, its attach target, its band and
its route are the ones production would have produced for that content against
the live corpus.

Nothing here decides what "correct" means, and nothing here calls an LLM. It
produces :class:`Slate` values; ``eval_write_triage_cases.py::plan_from_slates``
pairs them with the curator labels, and the eval scores them. A
:class:`SlateSnapshot` keeps one retrieval on disk so several runs judge it.

Imported by path, the same way the eval script imports the calibrator —
``scripts/`` is not an importable package. It imports only
``eval_write_triage_scoring.py``, for the band vocabulary a snapshot is held to.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fused_memory.models.memory import MemoryResult
from fused_memory.server.write_triage import (
    canonical_id_of,
    decide_band,
    retrieve_candidates,
    route_band,
)
from fused_memory.server.write_triage_judge import select_judge_candidates
from fused_memory.utils.script_loader import load_script

logger = logging.getLogger(__name__)

_SCRIPTS = Path(__file__).resolve().parent


# Pyright cannot follow the by-path loader, so it alone resolves `_scoring` through
# the scripts/ extraPath; at runtime the module is always the by-path load.
if TYPE_CHECKING:
    import eval_write_triage_scoring as _scoring
else:
    _scoring = load_script(
        _SCRIPTS / 'eval_write_triage_scoring.py', 'eval_write_triage_scoring',
    )


@dataclass(frozen=True)
class Slate:
    """One record's production-shaped candidate set, its band and its route.

    ``candidates`` are the records ``select_judge_candidates`` kept, in the
    order the prompt renders them; ``attach_target_id`` is
    ``decision.canonical_id`` verbatim, the band's winner, and is NOT required
    to be in ``candidates``. A judged attach files against the candidate the
    judge names (task 5794), which need not be this one.
    """

    memory_id: str
    candidates: tuple[Mapping[str, Any], ...]
    attach_target_id: str | None
    #: The ``decide_band`` cosine label, which the band split reads.
    band: str
    #: The outcome ``route_band`` routes the write to, i.e. what reaches the judge.
    routed: str
    similarity: float | None
    canonical_present: bool
    retrieved_count: int
    degraded: bool
    self_retrieved: bool


#: The on-disk shape of a :class:`SlateSnapshot`; a file of any other is refused.
SLATE_SNAPSHOT_SCHEMA_VERSION = 1


def _require_fields(payload: Mapping[str, Any], cls: type, where: str) -> None:
    missing = [f.name for f in fields(cls) if f.name not in payload]
    if missing:
        raise ValueError(f'{where} is missing field(s) {missing}')


def _require_bands(entry: Mapping[str, Any], where: str) -> None:
    """Refuse a slate whose ``band`` or ``routed`` no ``decide_band``/``route_band`` returns.

    Checked on load, so a corrupt snapshot is refused before any case is judged
    and paid for, rather than at scoring after all of them.
    """
    for name in ('band', 'routed'):
        if entry[name] not in _scoring.EVAL_BANDS:
            raise ValueError(
                f'{where} ({entry["memory_id"]}): {name} {entry[name]!r} is outside '
                f'{list(_scoring.EVAL_BANDS)}',
            )


@dataclass(frozen=True)
class SlateSnapshot:
    """A measured retrieval: one retrieval, banded, routed and trimmed, that several judge runs can be judged over.

    The band edges, widths and project are the ones the slates were measured
    at. Loading rebuilds each :class:`Slate` field by field as it was
    measured; nothing is re-banded or re-routed.
    """

    project_id: str
    t_high: float | None
    t_low: float | None
    candidate_k: int
    judge_candidate_count: int
    slates: tuple[Slate, ...]

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {f.name: getattr(self, f.name) for f in fields(self)}
        payload['slates'] = [
            {
                **{f.name: getattr(slate, f.name) for f in fields(slate)},
                'candidates': [dict(row) for row in slate.candidates],
            }
            for slate in self.slates
        ]
        path.write_text(json.dumps(
            {'schema_version': SLATE_SNAPSHOT_SCHEMA_VERSION, **payload}, indent=2,
        ) + '\n')

    @classmethod
    def load(cls, path: str | Path) -> SlateSnapshot:
        payload = json.loads(Path(path).read_text())
        version = payload.get('schema_version')
        if version != SLATE_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f'{path}: schema_version {version!r} is not this module\'s '
                f'{SLATE_SNAPSHOT_SCHEMA_VERSION}',
            )
        _require_fields(payload, cls, str(path))
        slates = []
        for index, entry in enumerate(payload['slates']):
            _require_fields(entry, Slate, f'{path}: slate {index}')
            _require_bands(entry, f'{path}: slate {index}')
            slates.append(Slate(**{
                **{f.name: entry[f.name] for f in fields(Slate)},
                'candidates': tuple(entry['candidates']),
            }))
        return cls(**{
            **{f.name: payload[f.name] for f in fields(cls)},
            'slates': tuple(slates),
        })


async def prefetch_retrievals(
    memory_service: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    project_id: str,
    k: int,
) -> dict[str, dict[str, Any]]:
    """One production retrieval per *record*, plus its canonical's liveness.

    Every record given is retrieved for, so the caller decides the population.

    The record's OWN id is dropped from its results: production triages content
    that is not yet stored, so a fixture record that is still live in the
    corpus would otherwise retrieve itself at cosine ~1.0 and band as a
    deterministic restatement of itself. Whether that happened is reported per
    record rather than assumed away.
    """
    canonical_live: dict[str, bool] = {}
    retrievals: dict[str, dict[str, Any]] = {}
    for record in records:
        memory_id = str(record['memory_id'])
        cluster_id = str(record['cluster_id'])
        if cluster_id not in canonical_live:
            canonical_live[cluster_id] = (
                await memory_service.get_memory_by_id(project_id, cluster_id) is not None
            )
        results = await retrieve_candidates(
            memory_service, record['content'], project_id, k,
        )
        retrieved = list(results)
        rows = [row for row in retrieved if row.id != memory_id]
        retrievals[memory_id] = {
            'results': rows,
            'self_retrieved': len(rows) != len(retrieved),
            'degraded': bool(getattr(results, 'degraded', False)),
            'canonical_present': canonical_live[cluster_id],
        }
    return retrievals


def retrieved_slates(
    records: Sequence[Mapping[str, Any]],
    retrievals: Mapping[str, Mapping[str, Any]],
    *,
    t_high: float | None,
    t_low: float | None,
    judge_candidate_count: int,
) -> list[Slate]:
    """Band, route and trim each record's retrieval exactly as ``triage_write`` does.

    ``decide_band`` labels the band and ``route_band`` routes the record's own
    content, both over the WHOLE retrieval; ``select_judge_candidates`` trims
    to what the prompt would carry. The trim is run for every route, not just
    the judged one, so a deterministic or below-floor case still records the
    slate its route was decided on.

    A degraded retrieval is a counted fail-open in production and reaches here
    as an ordinary empty result, so it is warned about rather than banded
    silently.
    """
    degraded = [
        str(r['memory_id']) for r in records
        if retrievals[str(r['memory_id'])]['degraded']
    ]
    if degraded:
        logger.warning(
            '%d retrieval(s) came back DEGRADED — in production each is a '
            'counted fail-open acked as `stored`, not a measured band: %s',
            len(degraded), ', '.join(degraded[:5]),
        )

    slates: list[Slate] = []
    for record in records:
        memory_id = str(record['memory_id'])
        retrieval = retrievals[memory_id]
        rows = list(retrieval['results'])
        band = decide_band(rows, t_high=t_high, t_low=t_low)
        decision = route_band(
            rows, content=str(record['content']), t_high=t_high, t_low=t_low,
        )
        selected = select_judge_candidates(
            rows, judge_candidate_count, canonical_id=decision.canonical_id,
        )
        slates.append(Slate(
            memory_id=memory_id,
            candidates=tuple(normalize(row) for row in selected),
            attach_target_id=decision.canonical_id,
            band=band.outcome,
            routed=decision.outcome,
            similarity=decision.similarity,
            canonical_present=bool(retrieval['canonical_present']),
            retrieved_count=len(rows),
            degraded=bool(retrieval['degraded']),
            self_retrieved=bool(retrieval['self_retrieved']),
        ))
    return slates


def normalize(result: MemoryResult) -> dict[str, Any]:
    """A live store row in the eval's record shape, metadata kept VERBATIM.

    The metadata survives whole because it carries ``store_score`` — the
    cosine ``decide_band`` and ``select_judge_candidates`` both read — and the
    ``kind``/``parent_id`` pair ``canonical_id_of`` hoists on.
    """
    metadata = dict(result.metadata or {})
    return {
        'memory_id': result.id,
        'content': result.content,
        'category': metadata.get('category'),
        'canonical_id': canonical_id_of(result),
        'store_score': metadata.get('store_score'),
        'metadata': metadata,
    }
