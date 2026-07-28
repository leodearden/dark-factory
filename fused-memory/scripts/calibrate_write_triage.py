#!/usr/bin/env python3
"""Derive the ``add_memory`` write-triage band thresholds T_high / T_low from
MEASURED similarity distributions over a labeled curator corpus.

PRD ``docs/prds/memory-write-path-convergence.md`` §9 leaf α (contract C1,
decision D1).

The constraint, and why it exists
---------------------------------
**No threshold in this script is chosen a priori.** Both bounds are
*measured order statistics* of the observed distributions, and every
degenerate input returns ``None`` plus a structured reason rather than a
fabricated number. ``None`` means UNCALIBRATED, which the triage router
must read as fail-open to ``stored``.

This is not pedantry. The existing near-duplicate guard's ``0.92`` default
was inherited from a figure cited in Mem0's own docs, and the one genuine
rediscovery pair we have actually measured scores ``0.824`` — so that
guard could never have fired on the very case it exists to catch. A
plausible-looking constant is exactly the failure mode being corrected;
re-introducing one here, even as a fallback, would reproduce it.

Metric-space parity (why the measurement transfers to the live guard)
---------------------------------------------------------------------
The cosine measured here is the *same quantity* as the Qdrant
``relevance_score`` the guard compares against its threshold, because:

- ``backends/mem0_client.py`` pins ``infer=False`` on every add, so Mem0
  stores and embeds content VERBATIM — there is no LLM fact-extraction
  rewrite between the text in the fixture and the vector in the index.
- Mem0's embedder is built from ``config.embedder`` (provider ``openai``,
  model ``text-embedding-3-small``), and Mem0 passes NO custom
  ``dimensions``.

So this script must mirror that call exactly — same model, no
``dimensions`` override. Passing a different dimensionality would silently
move the measurement into a different space and make the derived
thresholds inapplicable to the guard they are meant to configure. The
report records the embedder model and dimensions actually used so a future
reader can tell whether an embedder change invalidates the calibration.

Structure
---------
Mirrors the sweep-script family (``scripts/audit_duplicate_memories.py``):
all computation lives in pure synchronous functions unit-tested against
injected vectors and retrievals; the live embedder and
``MemoryService.search`` are injected only at the CLI boundary. The whole
test suite therefore runs with no ``OPENAI_API_KEY``, no network and no
Qdrant.

Usage
-----
  # Report only (default): measure, derive, write the report, change nothing.
  python scripts/calibrate_write_triage.py --project-id reify

  # Also write the derived thresholds into config.yaml's write_triage block.
  python scripts/calibrate_write_triage.py --project-id reify --write-config
"""
from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Labels the curator assigned. `distinct` and `pseudo_contradiction` are the
# HARD NEGATIVES: same cluster, same topic, but adjudicated not-the-same-claim.
LABEL_CANONICAL = 'canonical'
LABEL_DUPLICATE = 'duplicate'
LABEL_DISTINCT = 'distinct'
LABEL_PSEUDO_CONTRADICTION = 'pseudo_contradiction'

_NEGATIVE_LABELS = frozenset({LABEL_DISTINCT, LABEL_PSEUDO_CONTRADICTION})


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def load_fixture(path: str | Path) -> list[dict[str, Any]]:
    """Read the labeled JSONL fixture strictly.

    A malformed line raises with its 1-based line number rather than being
    skipped: silently dropping a record would shrink the measured
    population without saying so, yielding a report whose thresholds look
    fine but were computed on a subset.
    """
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{lineno}: malformed JSON line: {exc}') from exc
            if not isinstance(record, dict):
                raise ValueError(f'{path}:{lineno}: expected a JSON object, got {type(record).__name__}')
            records.append(record)
    return records


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

def build_pair_sets(records: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    """Partition every unordered record pair into three disjoint classes.

    Keyed on ``cluster_id`` — the CANONICAL memory UUID, never the gate id.
    Gates esc-5534/5547/5561/5610 each produced two canonicals, so keying by
    gate would fuse two canonicals' member sets into one cluster and inject
    pairs that are not duplicates into the positive class, dragging the
    derived T_high down.

    - ``true_dup_pairs`` — same cluster, both members ``duplicate`` or
      ``canonical``: the curator-confirmed genuine rediscoveries.
    - ``unrelated_pairs`` — different clusters: the measured negative class.
      The corpus is domain-homogeneous, so these scores must be measured
      rather than assumed low.
    - ``hard_negative_pairs`` — same cluster, but at least one member
      labeled ``distinct`` or ``pseudo_contradiction``: same topic,
      curator-ruled NOT duplicates. The hardest negatives for the
      deterministic band.

    The partition is total: every unordered pair lands in exactly one class.
    """
    true_dup: list[dict[str, str]] = []
    unrelated: list[dict[str, str]] = []
    hard_negative: list[dict[str, str]] = []

    n = len(records)
    for i in range(n):
        left = records[i]
        for j in range(i + 1, n):
            right = records[j]
            a, b = sorted((str(left['memory_id']), str(right['memory_id'])))
            pair = {'a': a, 'b': b}
            if left['cluster_id'] != right['cluster_id']:
                unrelated.append(pair)
            elif left['label'] in _NEGATIVE_LABELS or right['label'] in _NEGATIVE_LABELS:
                hard_negative.append(pair)
            else:
                true_dup.append(pair)

    return {
        'true_dup_pairs': true_dup,
        'unrelated_pairs': unrelated,
        'hard_negative_pairs': hard_negative,
    }


# ---------------------------------------------------------------------------
# Similarity + distribution statistics
# ---------------------------------------------------------------------------

def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Raw cosine between two embedding vectors.

    This is deliberately the same quantity Qdrant reports as
    ``relevance_score`` — see the module docstring's metric-space parity
    note. stdlib only; no numpy dependency is added for a dot product.

    A zero-norm or mismatched-length input raises rather than yielding NaN:
    a NaN would propagate silently through the distributions and corrupt
    every statistic derived from them.
    """
    if len(a) != len(b):
        raise ValueError(f'vector length mismatch: {len(a)} != {len(b)}')
    norm_a = math.sqrt(math.fsum(x * x for x in a))
    norm_b = math.sqrt(math.fsum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError(
            'cosine_similarity received a zero-norm vector — refusing to return NaN '
            '(a NaN here would silently corrupt the measured distributions)',
        )
    return math.fsum(x * y for x, y in zip(a, b, strict=True)) / (norm_a * norm_b)


def _order_statistic(ordered: list[float], quantile: float) -> float:
    """Nearest-rank quantile: always a value actually present in the sample.

    Interpolating would produce a threshold no measurement supports, which
    is precisely what derive_bands must not do.
    """
    idx = math.ceil(quantile * len(ordered)) - 1
    return ordered[min(max(idx, 0), len(ordered) - 1)]


def summarize_distribution(scores: Sequence[float]) -> dict[str, Any]:
    """Order-statistic summary of a measured score sample.

    An empty sample reports ``n=0`` with every statistic ``None`` rather
    than ``0.0``, so an empty pair class can never be misread as a measured
    zero.
    """
    if not scores:
        return {
            'n': 0, 'min': None, 'max': None, 'mean': None, 'median': None,
            'p05': None, 'p25': None, 'p75': None, 'p95': None,
        }
    ordered = sorted(float(s) for s in scores)
    return {
        'n': len(ordered),
        'min': ordered[0],
        'max': ordered[-1],
        'mean': math.fsum(ordered) / len(ordered),
        'median': _order_statistic(ordered, 0.50),
        'p05': _order_statistic(ordered, 0.05),
        'p25': _order_statistic(ordered, 0.25),
        'p75': _order_statistic(ordered, 0.75),
        'p95': _order_statistic(ordered, 0.95),
    }


# ---------------------------------------------------------------------------
# Band derivation
# ---------------------------------------------------------------------------

# Machine-readable refusal codes. A caller branches on these rather than
# matching prose; each is emitted with the measurements that produced it.
REASON_EMPTY_CLASS = 'empty_class'
REASON_NOT_SEPARABLE = 'not_separable'
REASON_NO_JUDGE_BAND = 'no_judge_band'


def derive_bands(
    dup_scores: Sequence[float],
    negative_scores: Sequence[float],
) -> tuple[float | None, float | None, str | None]:
    """Derive ``(t_high, t_low, reason)`` from two measured score samples.

    Both thresholds are order statistics of the observed duplicate
    distribution — never interpolated, never defaulted:

    - ``t_high`` is the SMALLEST measured duplicate score that strictly
      exceeds every measured negative. So the deterministic restate band
      admits zero measured false positives by construction, and taking the
      smallest such value keeps that band as wide as the evidence allows.
    - ``t_low`` is the duplicate distribution's lower tail (p05), so
      substantially all curator-confirmed rediscoveries reach at least the
      judge band.

    Every degenerate input returns ``None`` plus a reason code instead of a
    fabricated number. ``None`` means UNCALIBRATED and the triage router
    must fail open to ``stored``.
    """
    if not dup_scores or not negative_scores:
        return None, None, (
            f'{REASON_EMPTY_CLASS}: cannot separate two classes when one is empty '
            f'(n_duplicate={len(dup_scores)}, n_negative={len(negative_scores)})'
        )

    max_negative = max(negative_scores)
    ordered_dups = sorted(float(s) for s in dup_scores)
    separating = [s for s in ordered_dups if s > max_negative]
    if not separating:
        return None, None, (
            f'{REASON_NOT_SEPARABLE}: the highest measured negative ({max_negative}) is at or '
            f'above every measured duplicate (max={ordered_dups[-1]}), so no measured value '
            'separates the classes. Refusing to interpolate a threshold the data does not '
            'support — this outcome is itself the calibration finding.'
        )

    t_high = separating[0]
    t_low = _order_statistic(ordered_dups, 0.05)
    if t_low >= t_high:
        # Perfect separation: t_high IS the duplicate class's own lower
        # bound, so no measured duplicate sits strictly below it and no
        # judge band is derivable from this sample.
        return t_high, None, (
            f'{REASON_NO_JUDGE_BAND}: every measured duplicate ({ordered_dups[0]}..'
            f'{ordered_dups[-1]}) already clears every measured negative (max={max_negative}), '
            f'so t_high={t_high} is the duplicate lower bound and no measured value lies '
            'strictly below it. No judge band is derivable from this sample.'
        )

    return t_high, t_low, None


# ---------------------------------------------------------------------------
# Candidate-retrieval recall
# ---------------------------------------------------------------------------

def compute_recall_at_k(
    retrievals: Sequence[dict[str, Any]],
    ks: Sequence[int],
) -> dict[str, Any]:
    """Measure how often the existing search surfaces the ground-truth canonical.

    Each retrieval carries ``memory_id``, ``canonical_id``,
    ``canonical_present`` and the ranked ``candidates`` search returned.

    Retrievals whose canonical is not in the corpus are listed under
    ``canonical_absent`` and dropped from the denominator. The session's
    duplicates were deleted, so an absent canonical is a CORPUS GAP rather
    than a retrieval failure; scoring it as a miss would understate recall
    and could push T_low lower than the evidence supports.

    An empty denominator reports ``recall=None``, not ``0.0`` — no
    measurement is not a measured zero.
    """
    scorable: list[dict[str, Any]] = []
    absent: list[dict[str, str]] = []
    for item in retrievals:
        if item.get('canonical_present'):
            scorable.append(item)
        else:
            absent.append({
                'memory_id': str(item.get('memory_id')),
                'canonical_id': str(item.get('canonical_id')),
            })

    per_k: list[dict[str, Any]] = []
    for k in ks:
        hits = sum(
            1 for item in scorable
            if item['canonical_id'] in list(item.get('candidates') or [])[:k]
        )
        total = len(scorable)
        per_k.append({
            'k': k,
            'hits': hits,
            'total': total,
            'recall': (hits / total) if total else None,
        })

    return {'per_k': per_k, 'canonical_absent': absent}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

# Pair classes, in report order. The last two are the NEGATIVE classes: a
# pair from either that reaches the deterministic band is a false positive.
PAIR_CLASSES = ('true_dup', 'unrelated', 'hard_negative')
NEGATIVE_PAIR_CLASSES = ('unrelated', 'hard_negative')


def _band_counts(
    scores: Sequence[float],
    t_high: float | None,
    t_low: float | None,
) -> dict[str, int]:
    """Count a class's scores across the three triage bands.

    ``s >= t_high`` deterministic; ``t_low <= s < t_high`` judge;
    ``s < t_low`` store. A ``None`` t_low means no judge band is derivable,
    so everything below t_high falls to store — the counts still sum to the
    class's n either way.
    """
    deterministic = sum(1 for s in scores if t_high is not None and s >= t_high)
    if t_low is None:
        judge = 0
    else:
        judge = sum(
            1 for s in scores
            if s >= t_low and (t_high is None or s < t_high)
        )
    return {
        'deterministic': deterministic,
        'judge': judge,
        'store': len(scores) - deterministic - judge,
    }


def build_report(
    scores_by_class: dict[str, Sequence[float]],
    t_high: float | None,
    t_low: float | None,
    reason: str | None,
    recall: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the JSON-serializable calibration report.

    Its job is to make the deterministic band's false-positive risk
    visible: ``deterministic_band_false_positives`` counts the unrelated
    and hard-negative pairs that would be restated WITHOUT a judge under
    the chosen ``t_high``. True duplicates in that band are the band
    working and are never tallied there.

    Tolerates an uncalibrated run (``t_high=None``): the measured
    distributions plus the refusal reason are exactly what justify the
    refusal, so they are still emitted. The false-positive tally is then
    ``None`` rather than ``0`` — with no deterministic band, ``0`` would
    read as "measured, and safe".
    """
    scores = {name: list(scores_by_class.get(name) or []) for name in PAIR_CLASSES}

    per_band = {
        name: _band_counts(values, t_high, t_low) for name, values in scores.items()
    }
    false_positives = (
        None if t_high is None
        else sum(per_band[name]['deterministic'] for name in NEGATIVE_PAIR_CLASSES)
    )

    run_provenance = dict(provenance)
    run_provenance['pair_counts'] = {name: len(values) for name, values in scores.items()}

    return {
        'chosen_t_high': t_high,
        'chosen_t_low': t_low,
        'reason': reason,
        'deterministic_band_false_positives': false_positives,
        'distributions': {
            name: summarize_distribution(values) for name, values in scores.items()
        },
        'per_band': per_band,
        'recall_at_k': recall,
        'provenance': run_provenance,
    }
