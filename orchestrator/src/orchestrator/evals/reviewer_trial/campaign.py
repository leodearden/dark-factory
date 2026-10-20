"""Multi-trial reviewer campaign aggregation and pre-registered decision rules.

A campaign runs every arm over the same fixtures for N independent trials
(replication by fixture, never by cell), scores each (arm, fixture, trial)
with the corpus scorer, and reduces the score records here. Everything in
this module is pure over JSON-shaped score records — the ``asdict`` form of
``scorer.ScoringResult`` with a ``trial`` key added — so a campaign can be
re-reduced from its on-disk checkpoints without re-spending.

A record whose matcher did not measure carries ``unmeasured`` (the ``asdict``
of ``measurement.Unmeasured``); every figure here is over measured records
only, and the unmeasured ones are counted beside them with their spend.

Score files cached before task 6736 carry no ``unmeasured`` key and cannot be
told apart from a refused matcher's fabricated zero. They are abandoned, not
migrated: they load as measured, so a campaign that needs a trustworthy mean
reruns into a fresh ``--results-dir`` (or deletes ``trial-*/scores/``). An
unmeasured score is never cached, so a rerun re-measures it. Rationale:
docs/plan-scoring-and-judge.md, "Scoring instruments: unmeasured, never a tie
or zero (task 6736)".
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from statistics import mean

ScoreRecord = dict
"""``asdict(ScoringResult)`` plus ``trial: int``; ``unmeasured`` may be set."""


@dataclass
class ArmMetrics:
    name: str
    trials: int
    reviews: int
    mean_f1: float | None
    mean_recall: float | None
    mean_precision: float | None
    mean_blocking_recall: float | None
    blocking_precision: float | None
    blocking_findings: int
    blocking_true: int
    total_cost_usd: float
    cost_per_review: float | None
    cost_per_blocking_finding: float | None
    f1_by_trial: list[float]
    paired_vs_incumbent: list[dict] = field(default_factory=list)
    complement_count: int = 0
    incumbent_matched: int = 0
    complement_share: float = 0.0
    unmeasured: int = 0
    unmeasured_cost_usd: float = 0.0


@dataclass
class CampaignDecisions:
    incumbent: str
    strict_f1_improvement: dict[str, bool]
    complement_pass: dict[str, bool]
    complement_threshold: float
    non_inferiority: dict[str, bool]
    non_inferiority_by_trial: dict[str, list[dict]]


def _is_measured(rec: ScoreRecord) -> bool:
    return rec.get('unmeasured') is None


def _spend(rec: ScoreRecord) -> float:
    return rec['cost_usd'] + rec.get('match_cost_usd', 0.0)


def _mean_of(records: list[ScoreRecord], metric: str) -> float | None:
    return round(mean(r[metric] for r in records), 4) if records else None


def _key(rec: ScoreRecord) -> tuple[int, str]:
    return rec['trial'], rec['diff_id']


def _matched_gt_ids(rec: ScoreRecord) -> set[str]:
    return {m['ground_truth_id'] for m in rec['matches']}


def blocking_counts(rec: ScoreRecord, blocking_gt: set[str]) -> tuple[int, int]:
    """Return ``(blocking_findings, blocking_true)`` for one score record.

    A blocking finding is any reviewer issue the reviewer marked blocking;
    it is true when it matched a ground-truth issue that is itself blocking.
    """
    found = sum(
        1 for m in rec['matches'] if m['reviewer_issue'].get('severity') == 'blocking'
    ) + sum(1 for fp in rec['false_positives'] if fp.get('severity') == 'blocking')
    true = sum(
        1 for m in rec['matches']
        if m['reviewer_issue'].get('severity') == 'blocking'
        and m['ground_truth_id'] in blocking_gt
    )
    return found, true


def _by_trial_f1(records: list[ScoreRecord]) -> list[float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for rec in records:
        grouped[rec['trial']].append(rec['f1'])
    return [mean(grouped[t]) for t in sorted(grouped)]


def paired_comparison(
    arm: list[ScoreRecord], incumbent: list[ScoreRecord], metric: str = 'f1',
) -> list[dict]:
    """Per-trial paired comparison of *arm* against *incumbent* on shared fixtures."""
    inc = {_key(r): r[metric] for r in incumbent}
    per_trial: dict[int, dict] = defaultdict(lambda: {'wins': 0, 'losses': 0, 'ties': 0, 'deltas': []})
    for rec in arm:
        k = _key(rec)
        if k not in inc:
            continue
        delta = rec[metric] - inc[k]
        bucket = per_trial[k[0]]
        bucket['deltas'].append(delta)
        bucket['wins' if delta > 0 else 'losses' if delta < 0 else 'ties'] += 1
    out = []
    for t in sorted(per_trial):
        b = per_trial[t]
        out.append({
            'trial': t, 'wins': b['wins'], 'losses': b['losses'], 'ties': b['ties'],
            'mean_delta': round(mean(b['deltas']), 4) if b['deltas'] else 0.0,
        })
    return out


def complement(arm: list[ScoreRecord], incumbent: list[ScoreRecord]) -> tuple[int, int]:
    """Return ``(complement_count, incumbent_matched)``.

    The complement is every ground-truth issue *arm* matched on a fixture
    and trial where the incumbent missed it; ``incumbent_matched`` is the
    incumbent's total matched count over the same pairs, so the ratio is
    the complement as a share of incumbent recall.
    """
    inc = {_key(r): _matched_gt_ids(r) for r in incumbent}
    comp = 0
    inc_total = 0
    for rec in arm:
        k = _key(rec)
        if k not in inc:
            continue
        comp += len(_matched_gt_ids(rec) - inc[k])
        inc_total += len(inc[k])
    return comp, inc_total


def summarize_arm(
    name: str,
    records: list[ScoreRecord],
    blocking_gt_by_diff: dict[str, set[str]],
    incumbent: list[ScoreRecord] | None,
) -> ArmMetrics:
    """Every figure over *name*'s measured records; the unmeasured ones are counted.

    With nothing measured, the means and ratios are ``None``, never ``0.0``.
    The paired comparison and complement only pair measured records on both
    sides.
    """
    measured = [r for r in records if _is_measured(r)]
    unmeasured = [r for r in records if not _is_measured(r)]
    found = true = 0
    for rec in measured:
        f, t = blocking_counts(rec, blocking_gt_by_diff.get(rec['diff_id'], set()))
        found += f
        true += t
    cost = sum(_spend(r) for r in measured)
    metrics = ArmMetrics(
        name=name,
        trials=len({r['trial'] for r in measured}),
        reviews=len(measured),
        mean_f1=_mean_of(measured, 'f1'),
        mean_recall=_mean_of(measured, 'recall'),
        mean_precision=_mean_of(measured, 'precision'),
        mean_blocking_recall=_mean_of(measured, 'blocking_recall'),
        blocking_precision=(round(true / found, 4) if found else 0.0) if measured else None,
        blocking_findings=found,
        blocking_true=true,
        total_cost_usd=round(cost, 2),
        cost_per_review=round(cost / len(measured), 4) if measured else None,
        cost_per_blocking_finding=round(cost / true, 2) if true else None,
        f1_by_trial=[round(v, 4) for v in _by_trial_f1(measured)],
        unmeasured=len(unmeasured),
        unmeasured_cost_usd=round(sum(_spend(r) for r in unmeasured), 2),
    )
    if incumbent is not None and incumbent is not records:
        measured_incumbent = [r for r in incumbent if _is_measured(r)]
        metrics.paired_vs_incumbent = paired_comparison(measured, measured_incumbent)
        metrics.complement_count, metrics.incumbent_matched = complement(measured, measured_incumbent)
        if metrics.incumbent_matched:
            metrics.complement_share = round(metrics.complement_count / metrics.incumbent_matched, 4)
    return metrics


def strict_improvement(arm: ArmMetrics, incumbent: ArmMetrics) -> bool:
    """Pre-registered adoption rule: the arm's mean F1 beats the incumbent's
    in every trial, paired on the same fixtures, and there are at least three."""
    pairs = list(zip(arm.f1_by_trial, incumbent.f1_by_trial, strict=False))
    return len(pairs) >= 3 and all(a > b for a, b in pairs)


def non_inferiority_deltas(arm: list[ScoreRecord], incumbent: list[ScoreRecord]) -> list[dict]:
    """Per trial, *arm*'s mean blocking recall and F1 minus *incumbent*'s.

    Both means run over the same cells: the ``(trial, diff_id)`` keys both
    sides measured. A fixture one side skipped, or a matcher cell one side
    could not measure, drops out of the pair rather than reading as a zero.
    """
    measured_incumbent = {_key(r): r for r in incumbent if _is_measured(r)}
    pairs_by_trial: dict[int, list[tuple[ScoreRecord, ScoreRecord]]] = defaultdict(list)
    for rec in arm:
        if _is_measured(rec) and _key(rec) in measured_incumbent:
            pairs_by_trial[rec['trial']].append((rec, measured_incumbent[_key(rec)]))
    return [
        {
            'trial': trial,
            'blocking_recall_delta': _mean_delta(pairs, 'blocking_recall'),
            'f1_delta': _mean_delta(pairs, 'f1'),
        }
        for trial, pairs in sorted(pairs_by_trial.items())
    ]


def _mean_delta(pairs: list[tuple[ScoreRecord, ScoreRecord]], metric: str) -> float:
    return round(mean(a[metric] for a, _ in pairs) - mean(i[metric] for _, i in pairs), 4)


def non_inferior(
    arm: list[ScoreRecord], incumbent: list[ScoreRecord], min_trials: int = 3,
) -> bool:
    """Pre-registered D2 adoption rule: in each of at least *min_trials* trials, paired on
    the cells both sides measured, the arm's mean blocking recall is not below the
    incumbent's and its mean F1 is not worse. Ties pass; nothing shared fails."""
    deltas = non_inferiority_deltas(arm, incumbent)
    return len(deltas) >= min_trials and all(
        d['blocking_recall_delta'] >= 0 and d['f1_delta'] >= 0 for d in deltas
    )


def summarize_campaign(
    records: list[ScoreRecord],
    incumbent_name: str,
    blocking_gt_by_diff: dict[str, set[str]],
    complement_threshold: float = 0.10,
) -> tuple[list[ArmMetrics], CampaignDecisions]:
    by_arm: dict[str, list[ScoreRecord]] = defaultdict(list)
    for rec in records:
        by_arm[rec['variant_name']].append(rec)
    if incumbent_name not in by_arm:
        raise ValueError(f'incumbent {incumbent_name!r} has no score records')
    incumbent_records = by_arm[incumbent_name]
    arms = [summarize_arm(incumbent_name, incumbent_records, blocking_gt_by_diff, None)]
    arms.extend(
        summarize_arm(name, recs, blocking_gt_by_diff, incumbent_records)
        for name, recs in by_arm.items() if name != incumbent_name
    )
    candidates = {name: recs for name, recs in by_arm.items() if name != incumbent_name}
    decisions = CampaignDecisions(
        incumbent=incumbent_name,
        strict_f1_improvement={a.name: strict_improvement(a, arms[0]) for a in arms[1:]},
        complement_pass={a.name: a.complement_share >= complement_threshold for a in arms[1:]},
        complement_threshold=complement_threshold,
        non_inferiority={
            name: non_inferior(recs, incumbent_records) for name, recs in candidates.items()
        },
        non_inferiority_by_trial={
            name: non_inferiority_deltas(recs, incumbent_records)
            for name, recs in candidates.items()
        },
    )
    return arms, decisions


def _fmt(value: float | None, spec: str) -> str:
    return 'n/a' if value is None else format(value, spec)


def format_markdown(arms: list[ArmMetrics], decisions: CampaignDecisions) -> str:
    lines = [
        '| Arm | trials | reviews | unmeasured | F1 | recall | precision | blocking P | blocking R | $/review | $/blocking finding | complement | F1 by trial |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for a in arms:
        comp = '—' if a.name == decisions.incumbent else f'{a.complement_count} ({a.complement_share:.1%})'
        lines.append(
            f'| {a.name} | {a.trials} | {a.reviews} | {a.unmeasured} | {_fmt(a.mean_f1, ".3f")} | '
            f'{_fmt(a.mean_recall, ".3f")} | {_fmt(a.mean_precision, ".3f")} | '
            f'{_fmt(a.blocking_precision, ".3f")} | {_fmt(a.mean_blocking_recall, ".3f")} | '
            f'{_fmt(a.cost_per_review, ".2f")} | {_fmt(a.cost_per_blocking_finding, ".2f")} | {comp} | '
            f'{", ".join(f"{v:.3f}" for v in a.f1_by_trial)} |'
        )
    lines.append('')
    for a in arms:
        lines.append(
            f'- `{a.name}` unmeasured: {a.unmeasured} '
            f'(excluded from every mean; ${a.unmeasured_cost_usd:.2f} spent)'
        )
    lines.append('')
    lines.append(f'Incumbent: `{decisions.incumbent}`. Decision rules (pre-registered):')
    for name, ok in decisions.strict_f1_improvement.items():
        paired = next(a for a in arms if a.name == name).paired_vs_incumbent
        wl = '; '.join(f"t{p['trial']} {p['wins']}W/{p['losses']}L/{p['ties']}T Δ{p['mean_delta']:+.3f}" for p in paired)
        lines.append(f'- `{name}` strict paired F1 improvement in every trial: **{"PASS" if ok else "FAIL"}** ({wl})')
    for name, ok in decisions.complement_pass.items():
        share = next(a for a in arms if a.name == name).complement_share
        lines.append(
            f'- `{name}` complement ≥ {decisions.complement_threshold:.0%} of incumbent recall: '
            f'**{"PASS" if ok else "FAIL"}** ({share:.1%})'
        )
    for name, ok in decisions.non_inferiority.items():
        deltas = '; '.join(
            f"t{d['trial']} BR{d['blocking_recall_delta']:+.3f} F1{d['f1_delta']:+.3f}"
            for d in decisions.non_inferiority_by_trial[name]
        ) or 'no shared measured cells'
        lines.append(
            f'- `{name}` non-inferior (blocking recall not below and F1 not worse in every '
            f'trial, paired, measured cells only): **{"PASS" if ok else "FAIL"}** ({deltas})'
        )
    return '\n'.join(lines)


def to_json(arms: list[ArmMetrics], decisions: CampaignDecisions) -> dict:
    return {'arms': [asdict(a) for a in arms], 'decisions': asdict(decisions)}
