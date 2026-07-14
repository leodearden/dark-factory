"""The paired variance acceptance gate for the T6 prompt-optimization loop.

See plans/tier1-prompt-optimization-prd.md T6 / D-5: the reviewer scorer is a
noisy LLM matcher, so acceptance must never be bare tie-rejection (`delta >
0`) over that noise. Instead a repeatability band is pre-measured from N
repeats of scoring the SAME (baseline) heuristics, and a candidate is
accepted only when its paired selection-split delta strictly exceeds that
band — i.e. the improvement is bigger than the scorer's own measured noise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ['AcceptanceRecord', 'evaluate_acceptance', 'measure_repeatability_band', 'paired_delta', 'textual_lr']


def textual_lr(step: int, *, start: int = 4, end: int = 2) -> int:
    """The bounded-edit-budget schedule for optimizer reflection step *step*.

    Decays by 1 per step from *start*, clamped to ``[end, start]`` — e.g. the
    default schedule is 4, 3, 2, 2, 2, ... . This is the maximum number of
    distinct textual edits the optimizer may propose to the HEURISTICS block
    at this step (conveyed to `optimizer.propose_heuristics_edit` as
    `max_edits`), keeping edits bounded and shrinking as the loop converges.
    """
    if step < 0:
        raise ValueError(f'textual_lr: step must be >= 0, got {step}')
    if end > start:
        raise ValueError(f'textual_lr: end ({end}) must be <= start ({start})')
    return max(end, min(start, start - step))


def paired_delta(base_scores: Sequence[float], cand_scores: Sequence[float]) -> float:
    """Mean per-item ``(candidate - baseline)`` difference over paired score lists.

    *base_scores* and *cand_scores* must be the same length — one score per
    corresponding item — else this raises :class:`ValueError` rather than
    silently truncating/zipping mismatched pairs.
    """
    if len(base_scores) != len(cand_scores):
        raise ValueError(
            'paired_delta: mismatched lengths '
            f'(base={len(base_scores)}, cand={len(cand_scores)})'
        )
    if not base_scores:
        raise ValueError('paired_delta: score lists must be non-empty')
    diffs = [cand - base for base, cand in zip(base_scores, cand_scores, strict=True)]
    return sum(diffs) / len(diffs)


def measure_repeatability_band(repeat_score_batches: Sequence[Sequence[float]]) -> float:
    """The pre-measured noise floor: max abs pairwise :func:`paired_delta` across repeats.

    *repeat_score_batches* is N repeats of scoring the SAME (baseline)
    heuristics over the SAME items — i.e. it isolates pure scorer/rollout
    noise, since the heuristics and items are held constant across repeats.
    Returns 0.0 when every repeat produced identical scores, and a positive
    value proportional to how much repeated scoring of identical heuristics
    disagrees with itself. Requires at least 2 batches (a single batch has no
    pair to compare against).
    """
    if len(repeat_score_batches) < 2:
        raise ValueError(
            'measure_repeatability_band: need >= 2 repeat batches to measure '
            f'noise, got {len(repeat_score_batches)}'
        )
    deltas = [
        abs(paired_delta(repeat_score_batches[i], repeat_score_batches[j]))
        for i in range(len(repeat_score_batches))
        for j in range(i + 1, len(repeat_score_batches))
    ]
    return max(deltas)


@dataclass(frozen=True)
class AcceptanceRecord:
    """The outcome of one :func:`evaluate_acceptance` call — accept/reject + why."""

    accepted: bool
    paired_delta: float
    band: float
    reason: str


def evaluate_acceptance(
    base_repeats: Sequence[Sequence[float]],
    cand_repeats: Sequence[Sequence[float]],
    band: float,
) -> AcceptanceRecord:
    """Accept a candidate ONLY when its paired selection-split delta strictly exceeds *band*.

    *base_repeats*/*cand_repeats* are each N repeats of per-item selection-
    split scores (baseline heuristics and candidate heuristics respectively).
    The per-repeat means are computed first, then paired across the N
    repeats via :func:`paired_delta` — so the comparison is baseline-repeat-i
    vs candidate-repeat-i, matching D-5's "paired scoring + N repeats"
    design rather than comparing pooled/unpaired aggregates.

    Accepts iff ``delta > band`` (strict) — since *band* is always >= 0
    (see :func:`measure_repeatability_band`), this single comparison already
    rejects both a zero/tie delta and a positive-but-within-band delta:
    never bare tie-rejection over a noisy matcher, and never a false accept
    on noise alone.
    """
    if len(base_repeats) != len(cand_repeats):
        raise ValueError(
            'evaluate_acceptance: base_repeats and cand_repeats must have the '
            f'same repeat count (base={len(base_repeats)}, cand={len(cand_repeats)})'
        )
    if not base_repeats:
        raise ValueError('evaluate_acceptance: need >= 1 repeat')

    base_means = [sum(batch) / len(batch) for batch in base_repeats]
    cand_means = [sum(batch) / len(batch) for batch in cand_repeats]
    delta = paired_delta(base_means, cand_means)

    accepted = delta > band
    reason = (
        f'paired_delta={delta:.6f} {">" if accepted else "<="} band={band:.6f} '
        f'({"strict improvement beyond noise floor" if accepted else "not a strict improvement beyond noise floor"})'
    )
    return AcceptanceRecord(accepted=accepted, paired_delta=delta, band=band, reason=reason)
