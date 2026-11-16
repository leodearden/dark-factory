#!/usr/bin/env python3
"""Measure duplicate-memory detectors on the task 3130 labelled fixture (task 10243).

Replays the 2026-08-11 ANN calibration protocol: the fixture's records are the
whole scan set, embedded with the production embedder, and each record's
neighbours are taken within its own category at ``top_k``. Over those ANN
candidates it measures each adjudicator arm through the adjudicated-ANN
detector in ``audit_duplicate_memories.py``, beside the calibrated-ANN
baseline, and derives the ``recall_caveat.adjudicator_measurements`` entry
that ``config/duplicate-audit.yaml`` carries.

It is a REPORT, not a gate: the gate proposal it renders is pre-registered and
unratified, and task 10244 rules on it.

Usage
-----
  # The committed report: every arm, default report path.
  cd fused-memory && uv run --all-packages --group reranker \\
      python scripts/measure_duplicate_detector_recall.py

  # A one-arm smoke, kept off the committed artifact.
  cd fused-memory && uv run --all-packages --group reranker \\
      python scripts/measure_duplicate_detector_recall.py \\
      --arm bge-reranker-v2-m3 --report-path /tmp/<name>.json
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shared.quality_series.stats import Z_95, wilson_interval

from fused_memory.utils.script_loader import load_script

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _PACKAGE_ROOT / 'scripts'

# Pyright cannot follow the by-path loader, so it resolves these through the
# scripts/ extraPath; at runtime each is always the by-path load.
if TYPE_CHECKING:
    import audit_duplicate_memories as _audit
    import calibrate_write_triage as _calibrate
    import eval_write_triage_reranker as _reranker
    import eval_write_triage_reranker_arms as _arms
else:
    _audit = load_script(_SCRIPTS / 'audit_duplicate_memories.py', 'audit_duplicate_memories')
    _calibrate = load_script(_SCRIPTS / 'calibrate_write_triage.py', 'calibrate_write_triage')
    _arms = load_script(
        _SCRIPTS / 'eval_write_triage_reranker_arms.py', 'eval_write_triage_reranker_arms',
    )
    _reranker = load_script(
        _SCRIPTS / 'eval_write_triage_reranker.py', 'eval_write_triage_reranker',
    )

TRUE_DUP = 'true_dup_pairs'
HARD_NEGATIVE = 'hard_negative_pairs'

Pair = tuple[int, int]
EmbedFn = Callable[[str, str], Sequence[float]]


@dataclass(frozen=True)
class Proportion:
    hits: int
    total: int
    rate: float
    wilson_95: tuple[float, float]


def proportion(hits: int, total: int) -> Proportion | None:
    """``hits/total`` with its Wilson 95% interval; None when there is nothing to divide."""
    if total == 0:
        return None
    return Proportion(hits=hits, total=total, rate=hits / total, wilson_95=wilson_interval(hits, total))


def _proportion_json(value: Proportion | None) -> dict[str, Any] | None:
    return None if value is None else asdict(value)


# ---------------------------------------------------------------------------
# The 2026-08-11 protocol, as pure functions over the fixture
# ---------------------------------------------------------------------------

def fixture_scan(records: Sequence[Mapping[str, Any]], embed: EmbedFn) -> list[dict[str, Any]]:
    """The fixture as the detector's scanned records, each embedded once, in fixture order."""
    return [
        {
            'id': str(record['memory_id']),
            'content': record['content'],
            'category': record.get('category'),
            'vector': embed(str(record['memory_id']), record['content']),
        }
        for record in records
    ]


def fixture_neighbors(
    memories: Sequence[Mapping[str, Any]], top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    """Each record's ANN hits, as ``audit_duplicate_memories.fetch_ann_neighbors`` queries them.

    The pool is the record's own category (a record with no category is
    matched unfiltered), ranked by cosine descending with the id breaking a
    tie. The window is ``top_k + 1`` including the record itself, whose
    self-hit is then stripped.
    """
    neighbors: dict[str, list[dict[str, Any]]] = {}
    for memory in memories:
        category = memory['category']
        ranked = sorted(
            (
                (_calibrate.cosine_similarity(memory['vector'], other['vector']), other['id'])
                for other in memories
                if not category or other['category'] == category
            ),
            key=lambda hit: (-hit[0], hit[1]),
        )
        neighbors[memory['id']] = [
            {'id': other_id, 'score': score}
            for score, other_id in ranked[:top_k + 1]
            if other_id != memory['id']
        ]
    return neighbors


def labelled_pair_classes(records: Sequence[Mapping[str, Any]]) -> dict[Pair, str]:
    """Every unordered record pair's class, keyed by canonical ``(low, high)`` index."""
    index = {str(record['memory_id']): i for i, record in enumerate(records)}
    classes: dict[Pair, str] = {}
    for name, pairs in _calibrate.build_pair_sets([dict(record) for record in records]).items():
        for pair in pairs:
            low, high = sorted((index[pair['a']], index[pair['b']]))
            classes[(low, high)] = name
    return classes


def _pairs_of(pair_classes: Mapping[Pair, str], name: str) -> list[Pair]:
    return sorted(pair for pair, cls in pair_classes.items() if cls == name)


@dataclass(frozen=True)
class CandidateGeneration:
    """What the cutoff-free ANN generator hands the adjudicator, against the labels."""

    top_k: int
    candidates: tuple[Pair, ...]
    true_caught: Proportion | None
    hard_negative_generated: int
    cross_category_true_pairs: int
    disclosure: Mapping[str, int]

    def to_json(self) -> dict[str, Any]:
        return {
            'top_k': self.top_k,
            'candidates_generated': len(self.candidates),
            'true_caught': _proportion_json(self.true_caught),
            'hard_negative_generated': self.hard_negative_generated,
            'cross_category_true_pairs': self.cross_category_true_pairs,
            'disclosure': dict(self.disclosure),
        }


def candidate_generation(
    memories: Sequence[Mapping[str, Any]],
    neighbors: Mapping[str, list[dict[str, Any]]],
    pair_classes: Mapping[Pair, str],
    top_k: int,
) -> CandidateGeneration:
    """The generator's ceiling: how many labelled pairs ANN can hand an adjudicator at all."""
    pairs, disclosure = _audit.ann_candidate_pairs(
        [dict(memory) for memory in memories], dict(neighbors), top_k=top_k,
    )
    generated = set(pairs)
    true_pairs = _pairs_of(pair_classes, TRUE_DUP)
    return CandidateGeneration(
        top_k=top_k,
        candidates=tuple(pairs),
        true_caught=proportion(sum(pair in generated for pair in true_pairs), len(true_pairs)),
        hard_negative_generated=sum(
            pair in generated for pair in _pairs_of(pair_classes, HARD_NEGATIVE)
        ),
        cross_category_true_pairs=sum(
            memories[i]['category'] != memories[j]['category'] for i, j in true_pairs
        ),
        disclosure=disclosure,
    )


# ---------------------------------------------------------------------------
# Intervals and the operating point
# ---------------------------------------------------------------------------

def auc_interval_95(
    auc: float | None, n_true: int, n_hard_negative: int,
) -> tuple[float, float] | None:
    """Hanley & McNeil (1982)'s 95% interval for an AUC, clamped to [0, 1]."""
    if auc is None or n_true == 0 or n_hard_negative == 0:
        return None
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    variance = (
        auc * (1 - auc)
        + (n_true - 1) * (q1 - auc * auc)
        + (n_hard_negative - 1) * (q2 - auc * auc)
    ) / (n_true * n_hard_negative)
    half = Z_95 * math.sqrt(max(variance, 0.0))
    return max(0.0, auc - half), min(1.0, auc + half)


@dataclass(frozen=True)
class OperatingPoint:
    """The inclusive threshold chosen, with the pairs it flags; ``of`` is every true pair."""

    threshold: float | None
    caught: int
    flagged: int
    of: int

    @property
    def recall(self) -> Proportion | None:
        return proportion(self.caught, self.of)

    @property
    def precision(self) -> Proportion | None:
        return proportion(self.caught, self.flagged)


def operating_point(
    candidate_scores: Mapping[Pair, float],
    positives: Collection[Pair],
    n_positive: int,
    floor: float,
) -> OperatingPoint:
    """The candidate-score threshold catching the most true pairs at precision >= *floor*.

    Only a threshold equal to some candidate's score is considered, and tied
    scores are flagged together. A tie on caught goes to the higher threshold.
    """
    best = OperatingPoint(threshold=None, caught=0, flagged=0, of=n_positive)
    flagged = caught = 0
    descending = sorted(candidate_scores.items(), key=lambda item: -item[1])
    for score, tied in itertools.groupby(descending, key=lambda item: item[1]):
        pairs = [pair for pair, _ in tied]
        flagged += len(pairs)
        caught += sum(pair in positives for pair in pairs)
        if caught / flagged >= floor and caught > best.caught:
            best = OperatingPoint(threshold=score, caught=caught, flagged=flagged, of=n_positive)
    return best


RECALL_AT_PRECISION_FLOORS = (0.80, 0.90, 0.95)


def recall_at_precision(
    candidate_scores: Mapping[Pair, float], positives: Collection[Pair], n_positive: int,
) -> dict[float, OperatingPoint]:
    return {
        floor: operating_point(candidate_scores, positives, n_positive, floor)
        for floor in RECALL_AT_PRECISION_FLOORS
    }
