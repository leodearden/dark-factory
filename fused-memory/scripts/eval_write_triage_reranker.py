#!/usr/bin/env python3
"""Measure reranker arms over production's retrieved write-triage slates.

PRD ``plans/write-triage-flip-readiness-prd.md`` §9 leaf ρ1 (decision D1).

Production attaches a write to the top cosine hit of ``retrieve_candidates``,
and on the curator fixture that hit is the right canonical only 16 times in 84
(the committed ``calibration/write_triage_calibration_report.json``'s recall@1).
The canonical is inside the retrieved 20 far more often. This script asks each
admissible reranker to reorder those same 20 and measures what it buys: rank-1
and rank-5 of the canonical-or-alias, AUC of its score on true duplicates
against hard negatives, per-slate latency and cost per write. The ``best``
block applies the PRD's selection rule and is what gate Γ2 reads.

It is a REPORT, not a gate: nothing here asserts a floor. Γ2 owns the
thresholds.

Structure
---------
A pure synchronous core, unit-tested against fake arms, with the live store
reached only inside the CLI. The arms themselves — the torch, OpenAI and
hosted-API edges — live in ``eval_write_triage_reranker_arms.py``, which this
module loads by path and which imports nothing from here. The reach rule for
"found the canonical" is the calibrator's (``calibrate_write_triage.py``), so
this report's rank-1 and κ1's recall@1 are the same measurement.

Usage
-----
  # A cheap live smoke, kept off the committed artifact.
  uv run --group reranker python scripts/eval_write_triage_reranker.py \\
      --limit 3 --report-path /tmp/rr-smoke.json

  # The committed report (aliases default to the fixture's sidecar).
  uv run --group reranker python scripts/eval_write_triage_reranker.py \\
      --project-id reify --report-path calibration/write_triage_reranker_report.json
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _PACKAGE_ROOT / 'scripts'


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


_calibrate = _load_script(_SCRIPTS / 'calibrate_write_triage.py', 'calibrate_write_triage')


def load_retrieval() -> types.ModuleType:
    """The live-store edge, loaded on first use: it pulls in the server stack."""
    return _load_script(_SCRIPTS / 'eval_write_triage_retrieval.py', 'eval_write_triage_retrieval')


LABEL_CANONICAL = _calibrate.LABEL_CANONICAL
LABEL_DUPLICATE = _calibrate.LABEL_DUPLICATE
LABEL_DISTINCT = _calibrate.LABEL_DISTINCT
LABEL_PSEUDO_CONTRADICTION = _calibrate.LABEL_PSEUDO_CONTRADICTION
load_fixture = _calibrate.load_fixture
load_canonical_aliases = _calibrate.load_canonical_aliases
package_relative = _calibrate.package_relative

RANK_KS = (1, 5)

#: Same cluster, curator-ruled not the same claim. The rule's home is
#: ``calibrate_write_triage.py::build_pair_sets``, whose own set is private.
HARD_NEGATIVE_LABELS = frozenset({LABEL_DISTINCT, LABEL_PSEUDO_CONTRADICTION})


@dataclass(frozen=True)
class RerankCase:
    """One labelled record and the slate production retrieved for its content.

    ``candidate_*`` are parallel tuples in retrieval order. ``candidate_parents``
    maps only the sighting/amendment children to the parent they hoist to.
    """

    memory_id: str
    label: str
    entry: str
    canonical_id: str
    canonical_present: bool
    candidate_ids: tuple[str, ...]
    candidate_texts: tuple[str, ...]
    candidate_cosines: tuple[float | None, ...]
    candidate_parents: Mapping[str, str]
    degraded: bool
    self_retrieved: bool


@dataclass(frozen=True)
class RateCount:
    hits: int
    total: int
    rate: float | None


@dataclass(frozen=True)
class AucResult:
    """AUC over pair scores; ``unscored`` counts cases whose canonical never reached the slate."""

    value: float | None
    n_true: int
    n_hard_negative: int
    unscored: int


@dataclass(frozen=True)
class RankingMetrics:
    rank1: RateCount
    rank5: RateCount
    auc: AucResult


def build_cases(
    records: Sequence[Mapping[str, Any]],
    retrievals: Mapping[str, Mapping[str, Any]],
) -> list[RerankCase]:
    """One :class:`RerankCase` per record, from ``prefetch_retrievals`` output."""
    normalize = load_retrieval().normalize
    cases: list[RerankCase] = []
    for record in records:
        memory_id = str(record['memory_id'])
        retrieval = retrievals[memory_id]
        rows = [normalize(row) for row in retrieval['results']]
        cases.append(RerankCase(
            memory_id=memory_id,
            label=str(record['label']),
            entry=str(record['content']),
            canonical_id=str(record['cluster_id']),
            canonical_present=bool(retrieval['canonical_present']),
            candidate_ids=tuple(row['memory_id'] for row in rows),
            candidate_texts=tuple(row['content'] for row in rows),
            candidate_cosines=tuple(row['store_score'] for row in rows),
            candidate_parents={
                row['memory_id']: row['canonical_id'] for row in rows
                if row['canonical_id'] != row['memory_id']
            },
            degraded=bool(retrieval['degraded']),
            self_retrieved=bool(retrieval['self_retrieved']),
        ))
    return cases


def baseline_scores(case: RerankCase) -> tuple[float | None, ...]:
    """Production's attach signal: each candidate's cosine ``store_score``."""
    return case.candidate_cosines


def auc_true_vs_hard_negative(
    positives: Sequence[float], negatives: Sequence[float],
) -> float | None:
    """Mann-Whitney: the probability a positive outscores a negative, a tie counting half."""
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0
        for p in positives for n in negatives
    )
    return wins / (len(positives) * len(negatives))


def _arm_order(case: RerankCase, scores: Sequence[float | None]) -> list[int]:
    """Candidate positions by score descending; a tie keeps retrieval order, None ranks last."""
    if len(scores) != len(case.candidate_ids):
        raise ValueError(
            f'{case.memory_id}: {len(scores)} scores for {len(case.candidate_ids)} candidates',
        )
    return sorted(range(len(scores)), key=lambda i: (scores[i] is None, -(scores[i] or 0.0)))


def _ranked_retrieval(case: RerankCase, order: Sequence[int]) -> dict[str, Any]:
    """*case* in the calibrator's retrieval shape, candidates in the arm's *order*."""
    return {
        'memory_id': case.memory_id,
        'canonical_id': case.canonical_id,
        'canonical_present': case.canonical_present,
        'candidates': [case.candidate_ids[i] for i in order],
        'candidate_parents': dict(case.candidate_parents),
    }


def _pair_score(
    scores: Sequence[float | None], order: Sequence[int], first_hit: Mapping[str, Any],
) -> float | None:
    """The arm's score for the first candidate, in its order, reaching the canonical-or-alias."""
    rank = first_hit.get('rank_with_aliases', first_hit['rank'])
    return None if rank == -1 else scores[order[rank - 1]]


def _auc_block(cases: Sequence[RerankCase], pair_scores: Sequence[float | None]) -> AucResult:
    positives: list[float] = []
    negatives: list[float] = []
    unscored = 0
    for case, score in zip(cases, pair_scores, strict=True):
        if case.label == LABEL_DUPLICATE:
            bucket = positives
        elif case.label in HARD_NEGATIVE_LABELS:
            bucket = negatives
        else:
            continue
        if score is None:
            unscored += 1
        else:
            bucket.append(score)
    return AucResult(
        value=auc_true_vs_hard_negative(positives, negatives),
        n_true=len(positives),
        n_hard_negative=len(negatives),
        unscored=unscored,
    )


def ranking_metrics(
    cases: Sequence[RerankCase],
    scores_per_case: Sequence[Sequence[float | None]],
    *,
    aliases: Mapping[str, str] | None,
) -> RankingMetrics:
    """Rank-1/rank-5 of the canonical-or-alias once each slate is reordered by its scores.

    Scored by ``calibrate_write_triage.py::compute_recall_at_k``, the rule κ1's
    recall@1 used. Given aliases, every case is in the denominator and an
    unreached absent canonical is a miss — the population of Γ2's basis. The
    AUC's pair score is found by the same rule, via ``compute_first_hit_ranks``.
    """
    scored = list(zip(cases, scores_per_case, strict=True))
    orders = [_arm_order(case, scores) for case, scores in scored]
    ranked = [_ranked_retrieval(case, order) for (case, _), order in zip(scored, orders, strict=True)]
    recall = _calibrate.compute_recall_at_k(
        ranked, RANK_KS, aliases=aliases, count_absent_as_miss=bool(aliases),
    )
    rank1, rank5 = (
        RateCount(hits=row['hits'], total=row['total'], rate=row['recall'])
        for row in recall['per_k']
    )
    first_hits = _calibrate.compute_first_hit_ranks(ranked, aliases=aliases)
    pair_scores = [
        _pair_score(scores, order, hit)
        for (_, scores), order, hit in zip(scored, orders, first_hits, strict=True)
    ]
    return RankingMetrics(rank1=rank1, rank5=rank5, auc=_auc_block(cases, pair_scores))
