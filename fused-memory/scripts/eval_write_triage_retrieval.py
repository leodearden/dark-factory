#!/usr/bin/env python3
"""The LIVE-store edge of the judge eval: production retrieval and banding.

``eval_write_triage_judge.py``'s seeded slate is a CONSTRUCTION — the correct
canonical hoisted to position 0 with a synthetic top score. This module is the
measured alternative: it drives the shipped ``retrieve_candidates``,
``decide_band`` and ``select_judge_candidates`` so a case's slate, its attach
target and its band are the ones production would have produced for that
content against the live corpus.

Nothing here decides what "correct" means, and nothing here calls an LLM. It
produces :class:`Slate` values; the eval script pairs them with the curator
labels and scores them.

Imported by path, the same way the eval script imports the calibrator —
``scripts/`` is not an importable package.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fused_memory.models.memory import MemoryResult
from fused_memory.server.write_triage import (
    _canonical_id_of,
    decide_band,
    retrieve_candidates,
)
from fused_memory.server.write_triage_judge import select_judge_candidates

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Slate:
    """One record's production-shaped candidate set and the band it fell in.

    ``candidates`` are the records ``select_judge_candidates`` kept, in the
    order the prompt renders them; ``attach_target_id`` is
    ``decision.canonical_id`` verbatim, which is what ``triage_write`` files
    an attach against and is NOT required to be in ``candidates``.
    """

    memory_id: str
    candidates: tuple[Mapping[str, Any], ...]
    attach_target_id: str | None
    band: str
    similarity: float | None
    canonical_present: bool
    retrieved_count: int
    degraded: bool
    self_retrieved: bool


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
    """Band and trim each record's retrieval exactly as ``triage_write`` does.

    ``decide_band`` sees the WHOLE retrieval; ``select_judge_candidates`` trims
    to what the prompt would carry. The trim is run for every band, not just
    the middle one, so a deterministic or below-floor case still records the
    slate its band was decided on.

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
        decision = decide_band(rows, t_high=t_high, t_low=t_low)
        selected = select_judge_candidates(
            rows, judge_candidate_count, canonical_id=decision.canonical_id,
        )
        slates.append(Slate(
            memory_id=memory_id,
            candidates=tuple(normalize(row) for row in selected),
            attach_target_id=decision.canonical_id,
            band=decision.outcome,
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
    ``kind``/``parent_id`` pair ``_canonical_id_of`` hoists on.
    """
    metadata = dict(result.metadata or {})
    return {
        'memory_id': result.id,
        'content': result.content,
        'category': metadata.get('category'),
        'canonical_id': _canonical_id_of(result),
        'store_score': metadata.get('store_score'),
        'metadata': metadata,
    }
