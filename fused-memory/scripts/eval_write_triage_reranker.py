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

import argparse
import asyncio
import contextlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import sys
import time
import types
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import asdict, dataclass
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
_arms = _load_script(
    _SCRIPTS / 'eval_write_triage_reranker_arms.py', 'eval_write_triage_reranker_arms',
)


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

ArmClass = _arms.ArmClass
ArmStatus = _arms.ArmStatus
SkipReason = _arms.SkipReason
ArmUnavailable = _arms.ArmUnavailable

RANK_KS = (1, 5)

#: Γ2's population, and κ1's 0.19 basis: every labelled record is in the
#: rank denominator, and an absent canonical no alias reaches is a miss.
#: Independent of which aliases are passed, as in ``compute_recall_at_k``.
ABSENT_IN_DENOMINATOR = True

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
    count_absent_as_miss: bool = ABSENT_IN_DENOMINATOR,
) -> RankingMetrics:
    """Rank-1/rank-5 of the canonical-or-alias once each slate is reordered by its scores.

    Scored by ``calibrate_write_triage.py::compute_recall_at_k``, the rule κ1's
    recall@1 used, with its two independent choices: *aliases* widens what
    reaches the canonical, *count_absent_as_miss* keeps a case whose canonical
    left the corpus in the denominator. The AUC's pair score is found by the
    same reach rule, via ``compute_first_hit_ranks``.
    """
    scored = list(zip(cases, scores_per_case, strict=True))
    orders = [_arm_order(case, scores) for case, scores in scored]
    ranked = [_ranked_retrieval(case, order) for (case, _), order in zip(scored, orders, strict=True)]
    recall = _calibrate.compute_recall_at_k(
        ranked, RANK_KS, aliases=aliases, count_absent_as_miss=count_absent_as_miss,
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


@dataclass(frozen=True)
class LatencyStats:
    slates_timed: int
    pairs_per_slate_min: int | None
    pairs_per_slate_max: int | None
    load_seconds: float
    warmup_slates: int


@dataclass(frozen=True)
class ArmRow:
    """One arm's report row. A skipped row carries every metric as None, never 0."""

    arm: str
    arm_class: str
    model: str
    status: str
    skip_reason: str | None = None
    skip_detail: str | None = None
    ranking: RankingMetrics | None = None
    p50_seconds: float | None = None
    p95_seconds: float | None = None
    latency: LatencyStats | None = None
    cost_per_write_usd: float | None = None
    device: str | None = None
    vram_peak_mib: float | None = None
    max_length: int | None = None
    pairs_over_max_length: int | None = None

    @classmethod
    def skipped(cls, spec: _arms.ArmSpec, reason: str, detail: str) -> ArmRow:
        return cls(
            arm=spec.name, arm_class=str(spec.arm_class), model=spec.model,
            status=ArmStatus.skipped, skip_reason=reason, skip_detail=detail,
        )

    def to_json(self) -> dict[str, Any]:
        ranking = self.ranking
        return {
            'arm': self.arm,
            'arm_class': self.arm_class,
            'model': self.model,
            'status': str(self.status),
            'skip_reason': None if self.skip_reason is None else str(self.skip_reason),
            'skip_detail': self.skip_detail,
            'rank1_rate': ranking.rank1.rate if ranking else None,
            'rank5_rate': ranking.rank5.rate if ranking else None,
            'rank1': _hits_and_total(ranking.rank1) if ranking else None,
            'rank5': _hits_and_total(ranking.rank5) if ranking else None,
            'auc': asdict(ranking.auc) if ranking else None,
            'p50_seconds': self.p50_seconds,
            'p95_seconds': self.p95_seconds,
            'latency': asdict(self.latency) if self.latency else None,
            'cost_per_write_usd': self.cost_per_write_usd,
            'device': self.device,
            'vram_peak_mib': self.vram_peak_mib,
            'max_length': self.max_length,
            'pairs_over_max_length': self.pairs_over_max_length,
        }


def _hits_and_total(count: RateCount) -> dict[str, int]:
    return {'hits': count.hits, 'total': count.total}


class _OverBudget(Exception):
    pass


@dataclass(frozen=True)
class _TimedSlate:
    index: int
    slate: _arms.SlateScores
    seconds: float
    pairs: int


def _score_case(scorer: _arms.Scorer, case: RerankCase) -> _arms.SlateScores:
    slate = scorer.score(case.entry, case.candidate_texts)
    if len(slate.scores) != len(case.candidate_ids):
        raise ValueError(
            f'{case.memory_id}: {len(slate.scores)} scores for '
            f'{len(case.candidate_ids)} candidates',
        )
    return slate


def _sum_or_none(values: Sequence[float | int | None]) -> float | int | None:
    """The sum, or None when there is nothing to sum or any term was unmeasured."""
    measured = [value for value in values if value is not None]
    if not values or len(measured) != len(values):
        return None
    return sum(measured)


def measure_arm(
    spec: _arms.ArmSpec,
    cases: Sequence[RerankCase],
    *,
    aliases: Mapping[str, str] | None,
    count_absent_as_miss: bool,
    context: _arms.ArmContext,
    clock: Callable[[], float],
    max_spend_usd: float,
) -> ArmRow:
    """Score every non-empty slate once with *spec*'s arm and summarise it as a row.

    One untimed warm-up slate precedes the timed pass, so the first call's
    lazy initialisation is not billed as latency. An empty slate is never
    sent and ranks as a miss.

    Failure policy: an unavailable arm, crossing *max_spend_usd* (warm-up
    included) or any other exception yields a skipped row with the reason,
    and whatever was measured before it is discarded — a partial rate would
    describe a different population than the one the row claims.
    """
    try:
        return _measure(
            spec, cases, aliases=aliases, count_absent_as_miss=count_absent_as_miss,
            context=context, clock=clock, max_spend_usd=max_spend_usd,
        )
    except ArmUnavailable as exc:
        logger.warning('arm %s skipped (%s): %s', spec.name, exc.reason, exc.detail)
        return ArmRow.skipped(spec, exc.reason, exc.detail)
    except _OverBudget as exc:
        logger.warning('arm %s skipped: %s', spec.name, exc)
        return ArmRow.skipped(spec, SkipReason.over_budget, str(exc))
    except Exception as exc:
        logger.exception('arm %s failed; recorded as skipped/error', spec.name)
        return ArmRow.skipped(spec, SkipReason.error, f'{type(exc).__name__}: {exc}')


def _measure(
    spec: _arms.ArmSpec,
    cases: Sequence[RerankCase],
    *,
    aliases: Mapping[str, str] | None,
    count_absent_as_miss: bool,
    context: _arms.ArmContext,
    clock: Callable[[], float],
    max_spend_usd: float,
) -> ArmRow:
    live = [index for index, case in enumerate(cases) if case.candidate_ids]
    spent = 0.0

    def charge(slate: _arms.SlateScores) -> None:
        nonlocal spent
        spent += slate.cost_usd or 0.0
        if spent > max_spend_usd:
            raise _OverBudget(
                f'spent {spent:.4f} USD, over the {max_spend_usd:.4f} USD ceiling',
            )

    started = clock()
    with spec.open(context) as scorer:
        load_seconds = clock() - started
        for index in live[:1]:
            charge(_score_case(scorer, cases[index]))
        timed: list[_TimedSlate] = []
        for index in live:
            start = clock()
            slate = _score_case(scorer, cases[index])
            timed.append(_TimedSlate(
                index=index, slate=slate, seconds=clock() - start,
                pairs=len(cases[index].candidate_ids),
            ))
            charge(slate)
        facts = scorer.facts()

    scores_by_index = {t.index: t.slate.scores for t in timed}
    ranking = ranking_metrics(
        cases, [scores_by_index.get(i, ()) for i in range(len(cases))],
        aliases=aliases, count_absent_as_miss=count_absent_as_miss,
    )
    seconds = _calibrate.summarize_distribution([t.seconds for t in timed])
    costs = [t.slate.cost_usd for t in timed]
    total_cost = _sum_or_none(costs)
    pairs = [t.pairs for t in timed]
    return ArmRow(
        arm=spec.name,
        arm_class=str(spec.arm_class),
        model=spec.model,
        status=ArmStatus.measured,
        ranking=ranking,
        p50_seconds=seconds['median'],
        p95_seconds=seconds['p95'],
        latency=LatencyStats(
            slates_timed=len(timed),
            pairs_per_slate_min=min(pairs, default=None),
            pairs_per_slate_max=max(pairs, default=None),
            load_seconds=load_seconds,
            warmup_slates=len(live[:1]),
        ),
        cost_per_write_usd=None if total_cost is None else total_cost / len(costs),
        device=facts.device,
        vram_peak_mib=facts.vram_peak_mib,
        max_length=facts.max_length,
        pairs_over_max_length=_sum_or_none([t.slate.pairs_over_max_length for t in timed]),
    )


#: PRD §9 ρ1's selection-rule ceiling. Γ2's own 3.0 s threshold lives in task
#: 5804's before_done.args; the two may be re-based independently.
DEFAULT_P95_CEILING_SECONDS = 3.0


def choose_best(
    rows: Sequence[Mapping[str, Any]], *, p95_ceiling_seconds: float,
) -> dict[str, Any]:
    """PRD §9's rule over report rows: arg-max rank-1 among measured arms within the ceiling.

    A rank-1 tie goes to the lower p95, then to the earlier row. With no arm
    within the ceiling, the fastest measured arm is carried with ``qualified``
    False; with none measured, every value is None rather than a zero.
    """
    measured = [
        (index, row) for index, row in enumerate(rows)
        if row['status'] == ArmStatus.measured
        and row['rank1_rate'] is not None and row['p95_seconds'] is not None
    ]
    qualified = [
        (index, row) for index, row in measured if row['p95_seconds'] <= p95_ceiling_seconds
    ]
    best: Mapping[str, Any] | None = None
    if qualified:
        _, best = min(
            qualified,
            key=lambda item: (-item[1]['rank1_rate'], item[1]['p95_seconds'], item[0]),
        )
    elif measured:
        _, best = min(measured, key=lambda item: (item[1]['p95_seconds'], item[0]))
    return {
        'arm': best['arm'] if best else None,
        'rank1_rate': best['rank1_rate'] if best else None,
        'p95_seconds': best['p95_seconds'] if best else None,
        'qualified': bool(qualified),
        'p95_ceiling_seconds': p95_ceiling_seconds,
    }


#: The COMMITTED artifact: the 84-case measurement gate Γ2 reads.
_DEFAULT_REPORT_PATH = _PACKAGE_ROOT / 'calibration' / 'write_triage_reranker_report.json'

BASELINE_ARM = 'cosine'


def require_json_report_path(report_path: str | Path) -> Path:
    """Refuse a report path not ending in ``.json``: the ``.md`` sibling is written beside it."""
    path = Path(report_path)
    if path.suffix != '.json':
        raise ValueError(
            f'the report path must end in .json (a .md sibling is written beside it): {path}',
        )
    return path


def guard_report_path(report_path: str | Path, *, limit: int | None) -> Path:
    """Every refusal of ``--report-path``, made before the store is opened.

    Beyond :func:`require_json_report_path`, a ``--limit`` run aimed at the
    committed report is refused however its path is spelled. ``resolve()`` on
    both sides, as ``eval_write_triage_judge.py::_is_committed_report`` does: a
    relative spelling must not slip past as a different file. A partial run
    published there would be a different population under the same name.
    """
    path = require_json_report_path(report_path)
    if limit is not None and path.resolve() == _DEFAULT_REPORT_PATH.resolve():
        raise ValueError(
            f'--limit {limit} would overwrite the committed report {package_relative(path)} '
            'with a partial measurement; pass --report-path somewhere else',
        )
    return path


def measure_baseline(
    cases: Sequence[RerankCase], *, aliases: Mapping[str, str] | None, count_absent_as_miss: bool,
) -> ArmRow:
    """Production's current attach order on the same slates. Costs nothing and adds no latency."""
    return ArmRow(
        arm=BASELINE_ARM, arm_class='baseline', model='store_score',
        status=ArmStatus.measured,
        ranking=ranking_metrics(
            cases, [baseline_scores(case) for case in cases],
            aliases=aliases, count_absent_as_miss=count_absent_as_miss,
        ),
    )


def caveats_for(report: Mapping[str, Any]) -> list[str]:
    """What a reader must know to read the numbers, stated from the report's own data."""
    arms = report['arms']
    caveats = [
        'Local cross-encoder arms score with each model\'s published template and default '
        'instruction; the LLM pairwise arm uses this script\'s own same-claim prompt, so the '
        'two classes are not prompted alike.',
        f'The {BASELINE_ARM} baseline orders each slate by store_score, production\'s attach '
        'signal. It is reported for comparison and is never eligible for best.',
        'AUC is rank-based (Mann-Whitney, ties count half) over each case\'s canonical-or-alias '
        'pair score as the arm scored it on the retrieved slate. '
        f'{report["baseline"]["auc"]["n_hard_negative"]} hard negative(s) reached their slate; '
        'read the value beside that sample size.',
        'Latency is per-slate wall time on the recorded device after one untimed warm-up '
        'slate; remote arms include the network round trip.',
        'A local arm\'s cost is the per-call charge (zero), not amortised hardware or power.',
    ]
    truncation = [
        f'{row["arm"]} at {row["max_length"]} tokens: {row["pairs_over_max_length"]} pair(s) over'
        for row in arms if row['max_length'] is not None
    ]
    caveats.append(
        'pairs_over_max_length is a lower bound on truncation: it counts entry plus candidate '
        'tokens without the template or special tokens.'
        + (f' {"; ".join(truncation)}.' if truncation else ''),
    )
    caveats.extend(
        f'{row["arm"]} scores are a probability distribution over the candidates of one slate, '
        'so they depend on the slate\'s size and its competing candidates: its AUC pools '
        'slate-normalised values and is not comparable with the other arms\' AUC.'
        for row in arms
        if row['arm_class'] == ArmClass.jev_choice and row['status'] == ArmStatus.measured
    )
    caveats.extend(
        f'{row["arm"]} was skipped ({row["skip_reason"]}): {row["skip_detail"]}'
        for row in arms if row['status'] == ArmStatus.skipped
    )
    return caveats


def build_report(
    *,
    arm_rows: Sequence[ArmRow],
    baseline: ArmRow,
    cases: Sequence[RerankCase],
    provenance: Mapping[str, Any],
    count_absent_as_miss: bool,
    context: _arms.ArmContext,
    max_spend_usd: float,
    p95_ceiling_seconds: float,
) -> dict[str, Any]:
    rows = [row.to_json() for row in arm_rows]
    measured = {
        'arms': rows,
        'baseline': baseline.to_json(),
        'best': choose_best(rows, p95_ceiling_seconds=p95_ceiling_seconds),
    }
    return {
        **measured,
        'caveats': caveats_for(measured),
        'provenance': {
            **provenance,
            'case_count': len(cases),
            'canonical_absent': sum(not case.canonical_present for case in cases),
            'absent_in_denominator': count_absent_as_miss,
            'degraded_retrievals': sum(case.degraded for case in cases),
            'self_retrieved': sum(case.self_retrieved for case in cases),
            'p95_ceiling_seconds': p95_ceiling_seconds,
            'max_arm_spend_usd': max_spend_usd,
            'local_batch_size': context.local_batch_size,
            'vram_cap_gib': context.vram_cap_gib,
            'pairwise_concurrency': context.pairwise_concurrency,
        },
    }


def _fmt(value: Any, spec: str = '') -> str:
    return 'n/a' if value is None else format(value, spec)


def _rate_cell(rate: float | None, count: Mapping[str, int] | None) -> str:
    if count is None:
        return _fmt(rate)
    return f'{_fmt(rate, ".3f")} ({count["hits"]}/{count["total"]})'


def _table_row(row: Mapping[str, Any]) -> str:
    cells = [
        row['arm'], row['arm_class'], row['status'], _fmt(row['skip_reason']),
        _rate_cell(row['rank1_rate'], row['rank1']),
        _rate_cell(row['rank5_rate'], row['rank5']),
        _fmt(row['auc']['value'] if row['auc'] else None, '.3f'),
        _fmt(row['p50_seconds'], '.3f'), _fmt(row['p95_seconds'], '.3f'),
        _fmt(row['cost_per_write_usd'], '.5f'),
        _fmt(row['device']), _fmt(row['vram_peak_mib'], '.0f'),
    ]
    return '| ' + ' | '.join(cells) + ' |'


def render_markdown(report: Mapping[str, Any]) -> str:
    """The human-readable sibling of the JSON report, rendered from it alone."""
    best = report['best']
    lines = [
        '# Write-triage reranker arms (rho1)',
        '',
        'Rendered from the JSON report beside this file by '
        '`scripts/eval_write_triage_reranker.py`.',
        '',
        f'**Best:** {best["arm"] or "none measured"} — '
        f'rank-1 {_fmt(best["rank1_rate"], ".3f")}, p95 {_fmt(best["p95_seconds"], ".3f")} s, '
        f'qualified: {"yes" if best["qualified"] else "no"} '
        f'(p95 ceiling {best["p95_ceiling_seconds"]} s)',
        '',
        '| arm | class | status | skip reason | rank-1 | rank-5 | AUC | p50 s | p95 s '
        '| cost/write USD | device | VRAM MiB |',
        '|' + '---|' * 12,
        _table_row(report['baseline']),
        *(_table_row(row) for row in report['arms']),
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


def run_reranker_eval(
    *,
    cases: Sequence[RerankCase],
    arms: Sequence[_arms.ArmSpec],
    aliases: Mapping[str, str] | None,
    count_absent_as_miss: bool,
    context: _arms.ArmContext,
    provenance: Mapping[str, Any],
    report_path: str | Path,
    clock: Callable[[], float],
    max_spend_usd: float,
    p95_ceiling_seconds: float,
) -> dict[str, Any]:
    """Measure the baseline and every arm in order, then write and return the report."""
    report_path = require_json_report_path(report_path)
    baseline = measure_baseline(cases, aliases=aliases, count_absent_as_miss=count_absent_as_miss)
    rows = []
    for spec in arms:
        logger.info('measuring arm %s (%s)', spec.name, spec.model)
        rows.append(measure_arm(
            spec, cases, aliases=aliases, count_absent_as_miss=count_absent_as_miss,
            context=context, clock=clock, max_spend_usd=max_spend_usd,
        ))
    report = build_report(
        arm_rows=rows, baseline=baseline, cases=cases, provenance=provenance,
        count_absent_as_miss=count_absent_as_miss, context=context,
        max_spend_usd=max_spend_usd, p95_ceiling_seconds=p95_ceiling_seconds,
    )
    write_report(report, report_path)
    return report


_FIXTURES = _PACKAGE_ROOT / 'tests' / 'fixtures'
_LIBRARIES = ('torch', 'sentence-transformers', 'transformers')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fixture', default=str(_FIXTURES / 'write_triage_calibration.jsonl'))
    parser.add_argument(
        '--canonical-aliases', dest='canonical_aliases',
        default=str(_FIXTURES / 'write_triage_calibration.canonical_aliases.json'),
        help='the {old_canonical_id: current_id} sidecar; the population the PRD thresholds cite',
    )
    parser.add_argument('--project-id', dest='project_id', default='reify')
    parser.add_argument('--report-path', dest='report_path', default=str(_DEFAULT_REPORT_PATH))
    parser.add_argument('--config', default=None, help='fused-memory config file (sets CONFIG_PATH)')
    parser.add_argument('--limit', type=int, default=None,
                        help='measure only the first N labelled records (a smoke; refused at the committed path)')
    parser.add_argument('--p95-ceiling-seconds', dest='p95_ceiling_seconds', type=float,
                        default=DEFAULT_P95_CEILING_SECONDS)
    parser.add_argument('--max-arm-spend-usd', dest='max_arm_spend_usd', type=float, default=2.0)
    parser.add_argument('--pairwise-concurrency', dest='pairwise_concurrency', type=int, default=20)
    parser.add_argument('--local-batch-size', dest='local_batch_size', type=int, default=4)
    parser.add_argument('--device', default='auto', help="local arms' torch device: auto, cpu or cuda")
    parser.add_argument('--vram-cap-gib', dest='vram_cap_gib', type=float, default=8.0)
    return parser


def _library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


@contextlib.asynccontextmanager
async def _live_memory_service(config: Any) -> AsyncIterator[Any]:
    from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

    memory = MemoryService(config)
    await memory.initialize()
    try:
        yield memory
    finally:
        await memory.close()


def run_cli(
    args: argparse.Namespace,
    *,
    memory_service_factory: Callable[[Any], AbstractAsyncContextManager[Any]] | None = None,
    arms: Sequence[_arms.ArmSpec] | None = None,
) -> int:
    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
    from fused_memory.server.write_triage import resolve_candidate_k  # noqa: PLC0415

    report_path = guard_report_path(args.report_path, limit=args.limit)
    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)
    fixture = load_fixture(args.fixture)
    aliases = load_canonical_aliases(args.canonical_aliases) if args.canonical_aliases else {}
    labelled = [record for record in fixture if str(record['label']) != LABEL_CANONICAL]
    labelled = labelled if args.limit is None else labelled[:args.limit]
    config = FusedMemoryConfig()
    k = resolve_candidate_k(types.SimpleNamespace(config=config))
    open_memory = memory_service_factory or _live_memory_service
    logger.info('retrieving %d slate(s) from project_id=%s at k=%d', len(labelled), args.project_id, k)

    async def prefetch() -> dict[str, dict[str, Any]]:
        async with open_memory(config) as memory:
            return await load_retrieval().prefetch_retrievals(
                memory, labelled, project_id=args.project_id, k=k,
            )

    cases = build_cases(labelled, asyncio.run(prefetch()))
    report = run_reranker_eval(
        cases=cases,
        arms=arms if arms is not None else _arms.D1_ARMS,
        aliases=aliases,
        count_absent_as_miss=ABSENT_IN_DENOMINATOR,
        context=_arms.ArmContext(
            device=args.device, local_batch_size=args.local_batch_size,
            vram_cap_gib=args.vram_cap_gib, pairwise_concurrency=args.pairwise_concurrency,
        ),
        provenance={
            'fixture_path': package_relative(args.fixture),
            'record_count': len(fixture),
            'canonical_aliases_path': (
                package_relative(args.canonical_aliases) if args.canonical_aliases else None
            ),
            'canonical_aliases_count': len(aliases),
            'project_id': args.project_id,
            'candidate_k': k,
            'retrieval_call': _calibrate.RETRIEVAL_CALLS[_calibrate.RETRIEVAL_PRODUCTION],
            'limit': args.limit,
            'libraries': _library_versions(),
        },
        report_path=report_path,
        clock=time.perf_counter,
        max_spend_usd=args.max_arm_spend_usd,
        p95_ceiling_seconds=args.p95_ceiling_seconds,
    )
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return run_cli(build_parser().parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
