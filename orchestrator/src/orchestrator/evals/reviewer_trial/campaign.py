"""Multi-trial reviewer campaign aggregation and pre-registered decision rules.

A campaign runs every arm over the same fixtures for N independent trials
(replication by fixture, never by cell), scores each (arm, fixture, trial)
with the corpus scorer, and reduces the score records here. Everything in
this module is pure over JSON-shaped score records — the ``asdict`` form of
``scorer.ScoringResult`` with a ``trial`` key added — so a campaign can be
re-reduced from its on-disk checkpoints without re-spending.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from statistics import mean

ScoreRecord = dict
"""``asdict(ScoringResult)`` plus ``trial: int``."""


@dataclass
class ArmMetrics:
    name: str
    trials: int
    reviews: int
    mean_f1: float
    mean_recall: float
    mean_precision: float
    mean_blocking_recall: float
    blocking_precision: float
    blocking_findings: int
    blocking_true: int
    total_cost_usd: float
    cost_per_review: float
    cost_per_blocking_finding: float | None
    f1_by_trial: list[float]
    paired_vs_incumbent: list[dict] = field(default_factory=list)
    complement_count: int = 0
    incumbent_matched: int = 0
    complement_share: float = 0.0


@dataclass
class CampaignDecisions:
    incumbent: str
    strict_f1_improvement: dict[str, bool]
    complement_pass: dict[str, bool]
    complement_threshold: float


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
    found = true = 0
    for rec in records:
        f, t = blocking_counts(rec, blocking_gt_by_diff.get(rec['diff_id'], set()))
        found += f
        true += t
    cost = sum(r['cost_usd'] + r.get('match_cost_usd', 0.0) for r in records)
    metrics = ArmMetrics(
        name=name,
        trials=len({r['trial'] for r in records}),
        reviews=len(records),
        mean_f1=round(mean(r['f1'] for r in records), 4),
        mean_recall=round(mean(r['recall'] for r in records), 4),
        mean_precision=round(mean(r['precision'] for r in records), 4),
        mean_blocking_recall=round(mean(r['blocking_recall'] for r in records), 4),
        blocking_precision=round(true / found, 4) if found else 0.0,
        blocking_findings=found,
        blocking_true=true,
        total_cost_usd=round(cost, 2),
        cost_per_review=round(cost / len(records), 4),
        cost_per_blocking_finding=round(cost / true, 2) if true else None,
        f1_by_trial=[round(v, 4) for v in _by_trial_f1(records)],
    )
    if incumbent is not None and incumbent is not records:
        metrics.paired_vs_incumbent = paired_comparison(records, incumbent)
        metrics.complement_count, metrics.incumbent_matched = complement(records, incumbent)
        if metrics.incumbent_matched:
            metrics.complement_share = round(metrics.complement_count / metrics.incumbent_matched, 4)
    return metrics


def strict_improvement(arm: ArmMetrics, incumbent: ArmMetrics) -> bool:
    """Pre-registered adoption rule: the arm's mean F1 beats the incumbent's
    in every trial, paired on the same fixtures, and there are at least three."""
    pairs = list(zip(arm.f1_by_trial, incumbent.f1_by_trial, strict=False))
    return len(pairs) >= 3 and all(a > b for a, b in pairs)


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
    decisions = CampaignDecisions(
        incumbent=incumbent_name,
        strict_f1_improvement={a.name: strict_improvement(a, arms[0]) for a in arms[1:]},
        complement_pass={a.name: a.complement_share >= complement_threshold for a in arms[1:]},
        complement_threshold=complement_threshold,
    )
    return arms, decisions


def format_markdown(arms: list[ArmMetrics], decisions: CampaignDecisions) -> str:
    lines = [
        '| Arm | trials | reviews | F1 | recall | precision | blocking P | blocking R | $/review | $/blocking finding | complement | F1 by trial |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for a in arms:
        cpb = f'{a.cost_per_blocking_finding:.2f}' if a.cost_per_blocking_finding is not None else 'n/a'
        comp = '—' if a.name == decisions.incumbent else f'{a.complement_count} ({a.complement_share:.1%})'
        lines.append(
            f'| {a.name} | {a.trials} | {a.reviews} | {a.mean_f1:.3f} | {a.mean_recall:.3f} | '
            f'{a.mean_precision:.3f} | {a.blocking_precision:.3f} | {a.mean_blocking_recall:.3f} | '
            f'{a.cost_per_review:.2f} | {cpb} | {comp} | {", ".join(f"{v:.3f}" for v in a.f1_by_trial)} |'
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
    return '\n'.join(lines)


def to_json(arms: list[ArmMetrics], decisions: CampaignDecisions) -> dict:
    return {'arms': [asdict(a) for a in arms], 'decisions': asdict(decisions)}
