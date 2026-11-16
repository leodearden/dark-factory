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

import contextlib
import itertools
import json
import logging
import math
import re
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from shared.quality_series.stats import Z_95, wilson_interval

from fused_memory.utils.script_loader import load_script

logger = logging.getLogger(__name__)

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
UNRELATED = 'unrelated_pairs'
PAIR_CLASSES = (TRUE_DUP, HARD_NEGATIVE, UNRELATED)

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


# ---------------------------------------------------------------------------
# The arms: signal (an arms ArmSpec) x granularity (full text or claims)
# ---------------------------------------------------------------------------

CLAIM_SPLIT_RE = re.compile(r'\n|(?<=[.!?])\s+')


def split_claims(text: str) -> list[str]:
    """*text* cut at line breaks and at sentence-ending punctuation followed by whitespace."""
    return [piece.strip() for piece in CLAIM_SPLIT_RE.split(text) if piece.strip()]


_Num = TypeVar('_Num', int, float)


def _sum_or_none(values: Sequence[_Num | None]) -> _Num | None:
    """The sum, or None when there is nothing to sum or any term was unmeasured."""
    measured = [value for value in values if value is not None]
    if not values or len(measured) != len(values):
        return None
    return sum(measured[1:], start=measured[0])


class ClaimLevelScorer:
    """Any arms Scorer, applied claim by claim (a BERTScore-style alignment).

    A candidate's score is the mean, over the entry's claims, of the best
    inner score among that candidate's claims. The inner scorer sees one
    entry claim against every candidate's claims at once.
    """

    def __init__(self, inner: _arms.Scorer) -> None:
        self.inner = inner

    def score(self, entry: str, candidate_texts: Sequence[str]) -> _arms.SlateScores:
        candidate_claims = [split_claims(text) for text in candidate_texts]
        flat = [claim for claims in candidate_claims for claim in claims]
        slates = [self.inner.score(claim, flat) for claim in split_claims(entry)]
        best_per_slate = [self._best_per_candidate(slate, candidate_claims) for slate in slates]
        return _arms.SlateScores(
            scores=tuple(
                sum(best[c] for best in best_per_slate) / len(best_per_slate)
                for c in range(len(candidate_texts))
            ),
            cost_usd=_sum_or_none([slate.cost_usd for slate in slates]),
            pairs_over_max_length=_sum_or_none([slate.pairs_over_max_length for slate in slates]),
        )

    @staticmethod
    def _best_per_candidate(
        slate: _arms.SlateScores, candidate_claims: Sequence[Sequence[str]],
    ) -> list[float]:
        if len(slate.scores) != sum(len(claims) for claims in candidate_claims):
            raise ValueError(
                f'{len(slate.scores)} scores for '
                f'{sum(len(claims) for claims in candidate_claims)} candidate claims',
            )
        best: list[float] = []
        start = 0
        for claims in candidate_claims:
            best.append(max(slate.scores[start:start + len(claims)]))
            start += len(claims)
        return best

    def facts(self) -> _arms.ScorerFacts:
        return self.inner.facts()


class Granularity(StrEnum):
    full_text = 'full_text'
    claims = 'claims'


@dataclass(frozen=True)
class DetectorArm:
    spec: _arms.ArmSpec
    granularity: Granularity

    @property
    def name(self) -> str:
        if self.granularity == Granularity.claims:
            return f'{self.spec.name}+claims'
        return self.spec.name


@contextlib.contextmanager
def open_detector_scorer(arm: DetectorArm, context: _arms.ArmContext) -> Iterator[_arms.Scorer]:
    """*arm*'s signal opened through its own spec, at *arm*'s granularity."""
    with arm.spec.open(context) as scorer:
        yield ClaimLevelScorer(scorer) if arm.granularity == Granularity.claims else scorer


#: jev-choice is excluded by design: its scores are a distribution over one
#: slate, not a pairwise judgement.
DETECTOR_ARMS: tuple[DetectorArm, ...] = (
    *(
        DetectorArm(spec, Granularity.full_text) for spec in _arms.D1_ARMS
        if spec.arm_class != _arms.ArmClass.jev_choice
    ),
    *(
        DetectorArm(spec, Granularity.claims) for spec in _arms.D1_ARMS
        if spec.arm_class == _arms.ArmClass.local_cross_encoder
    ),
)


# ---------------------------------------------------------------------------
# Measuring one arm, and the calibrated-ANN baseline
# ---------------------------------------------------------------------------

BASELINE_ARM = 'calibrated-ann'
BASELINE_ARM_CLASS = 'baseline'
OPERATING_RULE_FLOOR = 'max_caught_at_precision_floor'
OPERATING_RULE_BASELINE = 'calibrated_ann_cutoffs'


@dataclass(frozen=True)
class GateProposal:
    """A recall-gate proposal: the bars, the date fixed, and why each bar sits where it does."""

    precision_floor: float
    min_paraphrase_band_recall: float
    min_auc: float
    proposed_on: str
    basis: tuple[str, ...]


@dataclass(frozen=True)
class AucBlock:
    value: float | None
    n_true: int
    n_hard_negative: int
    hanley_mcneil_95: tuple[float, float] | None


def _auc_block(positives: Sequence[float], negatives: Sequence[float]) -> AucBlock:
    value = _reranker.auc_true_vs_hard_negative(positives, negatives)
    return AucBlock(
        value=value, n_true=len(positives), n_hard_negative=len(negatives),
        hanley_mcneil_95=auc_interval_95(value, len(positives), len(negatives)),
    )


def _operating_point_json(floor: float, point: OperatingPoint) -> dict[str, Any]:
    return {
        'precision_floor': floor,
        'threshold': point.threshold,
        'caught': point.caught,
        'flagged': point.flagged,
        'recall': _proportion_json(point.recall),
        'precision': _proportion_json(point.precision),
    }


@dataclass(frozen=True)
class DetectorRow:
    """One detector's report row. A skipped row carries every metric as None, never 0."""

    arm: str
    arm_class: str
    model: str
    granularity: str
    status: str
    skip_reason: str | None = None
    skip_detail: str | None = None
    operating_rule: str | None = None
    auc: AucBlock | None = None
    operating_threshold: float | None = None
    paraphrase_band_recall: Proportion | None = None
    precision: Proportion | None = None
    hard_negative_flagged: Proportion | None = None
    recall_at_precision: Mapping[float, OperatingPoint] | None = None
    pairs_scored: int | None = None
    unscored_pairs: int | None = None
    cost_usd: float | None = None
    pairs_over_max_length: int | None = None
    device: str | None = None
    vram_peak_mib: float | None = None
    max_length: int | None = None
    scoring_seconds: float | None = None

    @classmethod
    def skipped(cls, arm: DetectorArm, reason: str, detail: str) -> DetectorRow:
        return cls(
            arm=arm.name, arm_class=str(arm.spec.arm_class), model=arm.spec.model,
            granularity=str(arm.granularity), status=str(_arms.ArmStatus.skipped),
            skip_reason=str(reason), skip_detail=detail,
        )

    def to_json(self) -> dict[str, Any]:
        points = self.recall_at_precision
        return {
            'arm': self.arm,
            'arm_class': self.arm_class,
            'model': self.model,
            'granularity': self.granularity,
            'status': self.status,
            'skip_reason': self.skip_reason,
            'skip_detail': self.skip_detail,
            'operating_rule': self.operating_rule,
            'auc': None if self.auc is None else asdict(self.auc),
            'operating_threshold': self.operating_threshold,
            'paraphrase_band_recall': _proportion_json(self.paraphrase_band_recall),
            'precision': _proportion_json(self.precision),
            'hard_negative_flagged': _proportion_json(self.hard_negative_flagged),
            'recall_at_precision': None if points is None else [
                _operating_point_json(floor, point) for floor, point in points.items()
            ],
            'pairs_scored': self.pairs_scored,
            'unscored_pairs': self.unscored_pairs,
            'cost_usd': self.cost_usd,
            'pairs_over_max_length': self.pairs_over_max_length,
            'device': self.device,
            'vram_peak_mib': self.vram_peak_mib,
            'max_length': self.max_length,
            'scoring_seconds': self.scoring_seconds,
        }


class OverBudget(Exception):
    pass


class _MeteredSlateScorer:
    """One arm run's Scorer as audit's SlateScorer, metering spend and truncation."""

    def __init__(self, scorer: _arms.Scorer, max_spend_usd: float) -> None:
        self._scorer = scorer
        self._max_spend_usd = max_spend_usd
        self.costs: list[float | None] = []
        self.over_max_length: list[int | None] = []

    def __call__(self, entry: str, candidate_texts: Sequence[str]) -> Sequence[float]:
        slate = self._scorer.score(entry, candidate_texts)
        self.costs.append(slate.cost_usd)
        self.over_max_length.append(slate.pairs_over_max_length)
        spent = sum(cost or 0.0 for cost in self.costs)
        if spent > self._max_spend_usd:
            raise OverBudget(f'spent {spent:.4f} USD, over the {self._max_spend_usd:.4f} USD ceiling')
        return slate.scores


def measure_detector_arm(
    arm: DetectorArm,
    memories: Sequence[Mapping[str, Any]],
    candidates: Sequence[Pair],
    pair_classes: Mapping[Pair, str],
    *,
    context: _arms.ArmContext,
    max_spend_usd: float,
    clock: Callable[[], float],
    proposal: GateProposal,
) -> DetectorRow:
    """Adjudicate every candidate and labelled pair once with *arm*, and summarise it as a row.

    Failure policy (as ``eval_write_triage_reranker.measure_arm``): an
    unavailable arm, crossing *max_spend_usd* or any other exception yields a
    skipped row, and whatever was measured before it is discarded.
    """
    try:
        return _measure(
            arm, memories, candidates, pair_classes,
            context=context, max_spend_usd=max_spend_usd, clock=clock, proposal=proposal,
        )
    except _arms.ArmUnavailable as exc:
        logger.warning('arm %s skipped (%s): %s', arm.name, exc.reason, exc.detail)
        return DetectorRow.skipped(arm, exc.reason, exc.detail)
    except OverBudget as exc:
        logger.warning('arm %s skipped: %s', arm.name, exc)
        return DetectorRow.skipped(arm, _arms.SkipReason.over_budget, str(exc))
    except Exception as exc:
        logger.exception('arm %s failed; recorded as skipped/error', arm.name)
        return DetectorRow.skipped(arm, _arms.SkipReason.error, f'{type(exc).__name__}: {exc}')


def _measure(
    arm: DetectorArm,
    memories: Sequence[Mapping[str, Any]],
    candidates: Sequence[Pair],
    pair_classes: Mapping[Pair, str],
    *,
    context: _arms.ArmContext,
    max_spend_usd: float,
    clock: Callable[[], float],
    proposal: GateProposal,
) -> DetectorRow:
    true_pairs = _pairs_of(pair_classes, TRUE_DUP)
    hard_negatives = _pairs_of(pair_classes, HARD_NEGATIVE)
    union = sorted({*candidates, *true_pairs, *hard_negatives})
    with open_detector_scorer(arm, context) as scorer:
        metered = _MeteredSlateScorer(scorer, max_spend_usd)
        started = clock()
        scores = _audit.adjudicate_pairs([dict(m) for m in memories], union, metered)
        seconds = clock() - started
        facts = scorer.facts()

    candidate_scores = {pair: scores[pair] for pair in candidates if pair in scores}
    positives = set(true_pairs)
    point = operating_point(candidate_scores, positives, len(true_pairs), proposal.precision_floor)
    flagged = (
        set() if point.threshold is None
        else set(_audit.adjudicated_pairs(candidate_scores, point.threshold))
    )
    caught = len(flagged & positives)
    return DetectorRow(
        arm=arm.name,
        arm_class=str(arm.spec.arm_class),
        model=arm.spec.model,
        granularity=str(arm.granularity),
        status=str(_arms.ArmStatus.measured),
        operating_rule=OPERATING_RULE_FLOOR,
        auc=_auc_block(
            [scores[p] for p in true_pairs if p in scores],
            [scores[p] for p in hard_negatives if p in scores],
        ),
        operating_threshold=point.threshold,
        paraphrase_band_recall=proportion(caught, len(true_pairs)),
        precision=proportion(caught, len(flagged)),
        hard_negative_flagged=proportion(len(flagged & set(hard_negatives)), len(hard_negatives)),
        recall_at_precision=recall_at_precision(candidate_scores, positives, len(true_pairs)),
        pairs_scored=2 * len(scores),
        unscored_pairs=len(union) - len(scores),
        cost_usd=_sum_or_none(metered.costs),
        pairs_over_max_length=_sum_or_none(metered.over_max_length),
        device=facts.device,
        vram_peak_mib=facts.vram_peak_mib,
        max_length=facts.max_length,
        scoring_seconds=seconds,
    )


def measure_ann_baseline(
    memories: Sequence[Mapping[str, Any]],
    neighbors: Mapping[str, list[dict[str, Any]]],
    cutoffs: Mapping[str, float],
    pair_classes: Mapping[Pair, str],
) -> DetectorRow:
    """The existing detector: ANN hits at each category's calibrated cutoff, AUC over cosine."""
    pairs, _ = _audit.ann_pairs_from_neighbors(
        [dict(m) for m in memories], dict(neighbors), dict(cutoffs),
    )
    flagged = set(pairs)
    true_pairs = _pairs_of(pair_classes, TRUE_DUP)
    hard_negatives = _pairs_of(pair_classes, HARD_NEGATIVE)
    caught = len(flagged & set(true_pairs))

    def cosine(pair: Pair) -> float:
        return _calibrate.cosine_similarity(memories[pair[0]]['vector'], memories[pair[1]]['vector'])

    return DetectorRow(
        arm=BASELINE_ARM,
        arm_class=BASELINE_ARM_CLASS,
        model='cosine',
        granularity=str(Granularity.full_text),
        status=str(_arms.ArmStatus.measured),
        operating_rule=OPERATING_RULE_BASELINE,
        auc=_auc_block([cosine(p) for p in true_pairs], [cosine(p) for p in hard_negatives]),
        operating_threshold=None,
        paraphrase_band_recall=proportion(caught, len(true_pairs)),
        precision=proportion(caught, len(flagged)),
        hard_negative_flagged=proportion(len(flagged & set(hard_negatives)), len(hard_negatives)),
    )


# ---------------------------------------------------------------------------
# The pre-registered gate proposal, the choice, and the report
# ---------------------------------------------------------------------------

#: Fixed in the task 10243 plan BEFORE any adjudicator was measured; task
#: 10244 ratifies or re-bases it.
GATE_PROPOSAL = GateProposal(
    precision_floor=0.90,
    min_paraphrase_band_recall=0.75,
    min_auc=0.80,
    proposed_on='2026-11-16',
    basis=(
        "precision 0.90: a filed gate's member list is the curator's working list (task 4916), "
        'and a false member is a distinct claim proposed for folding; at the fixture\'s mean '
        'cluster of about 5 members (104 records / 20 canonicals, about 10 pairs) that is about '
        'one false pair per gate.',
        'recall 0.75: guards against the systematically under-inclusive gate (the esc-5541 '
        'false-closure shape flagged in the 2026-07-30 review of PRD leaf kappa); transitive '
        'closure recovers a member through any one caught pair, so member recall is at least '
        "pair recall; the floor sits below the per-category top-10 generator's ceiling "
        "(262/301 = 0.870 when registered; this report's candidate_generation block "
        're-measures it), so it is attainable by an adjudicator over this generator.',
        'AUC 0.80: Hosmer, Lemeshow and Sturdivant, Applied Logistic Regression, 3rd ed., '
        "section 5.2.4, rate 0.8-0.9 'excellent' discrimination; hard negatives are exactly "
        'the same-topic, different-claim pairs whose folding loses information; with 18 hard '
        'negatives the Hanley-McNeil interval is about +/-0.08, so both the point and '
        'lower-bound verdicts are reported for the ratifier (task 10244).',
    ),
)

SCAN_SET = 'fixture_only'
CHOICE_RULE = (
    'the measured non-baseline arm catching the most true pairs at the precision floor; '
    'a tie goes to the higher AUC, then the earlier row'
)
COMMITTED_SOURCE = 'fused-memory/calibration/duplicate_detector_recall_report.json (task 10243)'


def gate_proposal_json(proposal: GateProposal) -> dict[str, Any]:
    return {**asdict(proposal), 'basis': list(proposal.basis)}


def gate_verdict(row: Mapping[str, Any], proposal: GateProposal) -> dict[str, bool]:
    """*row* against *proposal*, on point estimates and on interval lower bounds."""
    recall, precision, auc = row['paraphrase_band_recall'], row['precision'], row['auc']
    if (
        row['operating_threshold'] is None or recall is None or precision is None
        or auc is None or auc['value'] is None or auc['hanley_mcneil_95'] is None
    ):
        return {'on_point_estimates': False, 'on_lower_bounds': False}
    return {
        'on_point_estimates': bool(
            recall['rate'] >= proposal.min_paraphrase_band_recall
            and precision['rate'] >= proposal.precision_floor
            and auc['value'] >= proposal.min_auc
        ),
        'on_lower_bounds': bool(
            recall['wilson_95'][0] >= proposal.min_paraphrase_band_recall
            and precision['wilson_95'][0] >= proposal.precision_floor
            and auc['hanley_mcneil_95'][0] >= proposal.min_auc
        ),
    }


def choose_detector(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The row :data:`CHOICE_RULE` picks, or None when no arm was measured."""
    eligible = [
        (index, row) for index, row in enumerate(rows)
        if row['status'] == _arms.ArmStatus.measured
        and row['arm_class'] != BASELINE_ARM_CLASS
        and row['paraphrase_band_recall'] is not None
    ]
    if not eligible:
        return None

    def rank(item: tuple[int, Mapping[str, Any]]) -> tuple[int, bool, float, int]:
        index, row = item
        auc = row['auc']['value'] if row['auc'] else None
        return (-row['paraphrase_band_recall']['hits'], auc is None, -(auc or 0.0), index)

    return min(eligible, key=rank)[1]


def recall_caveat_entry(report: Mapping[str, Any]) -> dict[str, Any] | None:
    """The chosen arm as a ``recall_caveat.adjudicator_measurements`` entry, or None."""
    chosen = report['chosen']
    if chosen is None:
        return None
    (row,) = [row for row in report['arms'] if row['arm'] == chosen['arm']]
    generated = report['candidate_generation']['true_caught']
    recall, precision = row['paraphrase_band_recall'], row['precision']
    hard, auc = row['hard_negative_flagged'], row['auc']
    return {
        'measured_at': report['provenance']['measured_at'],
        'detector': row['arm'],
        'scan_set': {'setting': report['setting']['scan_set'], 'records': report['setting']['records']},
        'candidate_generation': {
            'top_k': report['setting']['top_k'], 'caught': generated['hits'], 'of': generated['total'],
        },
        'operating_threshold': row['operating_threshold'],
        'paraphrase_band_recall': {
            'caught': recall['hits'], 'of': recall['total'],
            'rate': recall['rate'], 'wilson_95': recall['wilson_95'],
        },
        'precision_at_operating_point': None if precision is None else {
            'flagged': precision['total'], 'rate': precision['rate'],
            'wilson_95': precision['wilson_95'],
        },
        'hard_negative_flagged': {'flagged': hard['hits'], 'of': hard['total']},
        'auc_true_vs_hard_negative': {
            'value': auc['value'], 'n_true': auc['n_true'],
            'n_hard_negative': auc['n_hard_negative'], 'hanley_mcneil_95': auc['hanley_mcneil_95'],
        },
        'meets_proposed_gate': chosen['verdict']['on_point_estimates'],
        'source': COMMITTED_SOURCE,
    }


def caveats_for(report: Mapping[str, Any]) -> list[str]:
    """What a reader must know to read the numbers, stated from the report's own data."""
    setting = report['setting']
    true_total = report['pair_counts'][TRUE_DUP]
    floor = report['gate_proposal']['precision_floor']
    caveats = [
        f'The scan set is the fixture alone ({setting["scan_set"]}, {setting["records"]} records): '
        'the most favourable setting, with no live-corpus distractors competing for each '
        f'record\'s top_k={setting["top_k"]} window.',
        f'{report["candidate_generation"]["cross_category_true_pairs"]} of the {true_total} true '
        'pairs are cross-category. The per-category generator never forms them, so no '
        f'adjudicator over it can catch them; recall still divides by all {true_total}.',
        f'Each arm\'s operating threshold is selected in-sample on these {setting["records"]} '
        f'records at the {floor:.2f} precision floor; nothing is held out.',
        'AUC is rank-based (Mann-Whitney, ties count half) over every true and hard-negative '
        'pair, including pairs the generator never forms; its interval is Hanley-McNeil (1982).',
        'A pair\'s adjudicator score is the mean of its two directional scores: the scorers are '
        'query/document asymmetric, the duplicate relation is symmetric.',
        f'The {BASELINE_ARM} baseline replays the 2026-08-11 detector (ANN hits at each '
        'category\'s calibrated cutoff, AUC over raw cosine). It is never eligible to be chosen.',
        'jev-choice is excluded by design: its scores are a probability distribution over one '
        'slate, not a pairwise judgement.',
        'The LLM pairwise arm is not run at claim granularity: claims multiply its call count '
        'roughly a hundredfold.',
    ]
    truncation = [
        f'{row["arm"]} at {row["max_length"]} tokens: {row["pairs_over_max_length"]} slate pair(s) over'
        for row in report['arms'] if row['max_length'] is not None
    ]
    if truncation:
        caveats.append(
            'pairs_over_max_length is a lower bound on truncation (entry plus candidate tokens, '
            f'without the template): {"; ".join(truncation)}.',
        )
    caveats.extend(
        f'{row["arm"]} was skipped ({row["skip_reason"]}): {row["skip_detail"]}'
        for row in report['arms'] if row['status'] == _arms.ArmStatus.skipped
    )
    return caveats


def build_report(
    *,
    generation: CandidateGeneration,
    pair_classes: Mapping[Pair, str],
    baseline: DetectorRow,
    arm_rows: Sequence[DetectorRow],
    records: int,
    embedder: Mapping[str, Any],
    provenance: Mapping[str, Any],
    proposal: GateProposal = GATE_PROPOSAL,
) -> dict[str, Any]:
    rows = [row.to_json() for row in arm_rows]
    chosen = choose_detector(rows)
    measured: dict[str, Any] = {
        'setting': {
            'scan_set': SCAN_SET, 'records': records, 'top_k': generation.top_k,
            'per_category': True, 'embedder': dict(embedder),
        },
        'pair_counts': {
            name: sum(cls == name for cls in pair_classes.values()) for name in PAIR_CLASSES
        },
        'candidate_generation': generation.to_json(),
        'baseline': baseline.to_json(),
        'arms': rows,
        'chosen': None if chosen is None else {
            'arm': chosen['arm'], 'rule': CHOICE_RULE, 'verdict': gate_verdict(chosen, proposal),
        },
        'gate_proposal': gate_proposal_json(proposal),
        'provenance': dict(provenance),
    }
    return {
        **{key: measured[key] for key in (
            'setting', 'pair_counts', 'candidate_generation', 'baseline', 'arms', 'chosen',
            'gate_proposal',
        )},
        'recall_caveat_entry': recall_caveat_entry(measured),
        'caveats': caveats_for(measured),
        'provenance': measured['provenance'],
    }


def _fmt(value: Any, spec: str = '') -> str:
    return 'n/a' if value is None else format(value, spec)


def _interval(bounds: Sequence[float] | None) -> str:
    return 'n/a' if bounds is None else f'[{bounds[0]:.3f}, {bounds[1]:.3f}]'


def _count(value: Mapping[str, Any] | None) -> str:
    if value is None:
        return 'n/a'
    return f'{value["hits"]}/{value["total"]} ({value["rate"]:.3f})'


def _table_row(row: Mapping[str, Any]) -> str:
    auc = row['auc']
    cells = [
        row['arm'], row['arm_class'], row['granularity'], row['status'], _fmt(row['skip_reason']),
        _fmt(row['operating_threshold'], '.4f'),
        _count(row['paraphrase_band_recall']),
        _interval(row['paraphrase_band_recall']['wilson_95'] if row['paraphrase_band_recall'] else None),
        _count(row['precision']),
        _interval(row['precision']['wilson_95'] if row['precision'] else None),
        _count(row['hard_negative_flagged']),
        _fmt(auc['value'] if auc else None, '.4f'),
        _interval(auc['hanley_mcneil_95'] if auc else None),
        _fmt(row['cost_usd'], '.4f'), _fmt(row['device']),
    ]
    return '| ' + ' | '.join(cells) + ' |'


def _floor_cells(row: Mapping[str, Any]) -> str:
    points = row['recall_at_precision'] or []
    cells = [
        f'{point["caught"]}/{point["recall"]["total"] if point["recall"] else "n/a"} '
        f'@ {_fmt(point["threshold"], ".4f")}'
        for point in points
    ]
    return '| ' + ' | '.join([row['arm'], *cells]) + ' |'


def render_markdown(report: Mapping[str, Any]) -> str:
    """The human-readable sibling of the JSON report, rendered from it alone."""
    setting, counts = report['setting'], report['pair_counts']
    generation = report['candidate_generation']
    proposal, chosen = report['gate_proposal'], report['chosen']
    measured_rows = [row for row in report['arms'] if row['recall_at_precision'] is not None]
    verdict = (
        'none measured' if chosen is None else
        f'{chosen["arm"]} — on point estimates: '
        f'{"pass" if chosen["verdict"]["on_point_estimates"] else "fail"}; '
        f'on lower bounds: {"pass" if chosen["verdict"]["on_lower_bounds"] else "fail"}'
    )
    lines = [
        '# Duplicate-detector recall on the task 3130 fixture',
        '',
        'Rendered from the JSON report beside this file by '
        '`scripts/measure_duplicate_detector_recall.py`.',
        '',
        f'**Setting:** {setting["scan_set"]}, {setting["records"]} records, per-category '
        f'top_k={setting["top_k"]}, embedder {setting["embedder"]["model"]} '
        f'({setting["embedder"]["dimensions"]} dimensions). Pairs: '
        f'{counts[TRUE_DUP]} true, {counts[HARD_NEGATIVE]} hard negative, '
        f'{counts[UNRELATED]} unrelated.',
        '',
        f'**Generator ceiling:** {_count(generation["true_caught"])} true pairs generated '
        f'{_interval(generation["true_caught"]["wilson_95"] if generation["true_caught"] else None)}; '
        f'{generation["hard_negative_generated"]} hard negative(s) generated; '
        f'{generation["cross_category_true_pairs"]} cross-category true pair(s); '
        f'{generation["candidates_generated"]} candidates.',
        '',
        f'**Chosen:** {verdict}',
        '',
        '| detector | class | granularity | status | skip reason | threshold | recall | recall 95% '
        '| precision | precision 95% | hard negatives flagged | AUC | AUC 95% | cost USD | device |',
        '|' + '---|' * 15,
        _table_row(report['baseline']),
        *(_table_row(row) for row in report['arms']),
        '',
        '## Recall at precision floors',
        '',
        '| detector | ' + ' | '.join(f'{floor:.2f}' for floor in RECALL_AT_PRECISION_FLOORS) + ' |',
        '|' + '---|' * (1 + len(RECALL_AT_PRECISION_FLOORS)),
        *(_floor_cells(row) for row in measured_rows),
        '',
        f'## Gate proposal (pre-registered {proposal["proposed_on"]}, unratified)',
        '',
        f'- precision floor at the operating point: {proposal["precision_floor"]:.2f}',
        f'- minimum paraphrase-band recall: {proposal["min_paraphrase_band_recall"]:.2f}',
        f'- minimum AUC (true vs hard negative): {proposal["min_auc"]:.2f}',
        '',
        'Basis:',
        '',
        *(f'- {basis}' for basis in proposal['basis']),
        '',
        '## Caveats',
        '',
        *(f'- {caveat}' for caveat in report['caveats']),
        '',
        '## Provenance',
        '',
        *(f'- `{key}`: {json.dumps(value)}' for key, value in report['provenance'].items()),
    ]
    return '\n'.join(lines) + '\n'


def write_report(report: Mapping[str, Any], report_path: Path) -> None:
    """Write the JSON report and its ``.md`` sibling."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    report_path.with_suffix('.md').write_text(render_markdown(report))


#: The COMMITTED artifact the yaml entry is derived from.
_DEFAULT_REPORT_PATH = _PACKAGE_ROOT / 'calibration' / 'duplicate_detector_recall_report.json'


def guard_report_path(report_path: str | Path, *, arm_filter: Sequence[str] | None) -> Path:
    """Every refusal of ``--report-path``, made before anything is embedded.

    An ``--arm`` subset aimed at the committed report is refused however its
    path is spelled: a partial run there would be a different population
    under the same name.
    """
    path = _reranker.require_json_report_path(report_path)
    if arm_filter and path.resolve() == _DEFAULT_REPORT_PATH.resolve():
        raise ValueError(
            f'--arm {", ".join(arm_filter)} would overwrite the committed report '
            f'{_calibrate.package_relative(path)} with a partial measurement; '
            'pass --report-path somewhere else',
        )
    return path
