"""Dispatch scoring β (task 3819; plans/scheduler-dispatch-scoring-and-lock-layer-prd.md C2, I-4, I-6, row 4, row 9)."""

from __future__ import annotations

import csv
import inspect
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import pytest

import orchestrator
from orchestrator.config import (
    PRIORITY_TIERS,
    TIER_BASE,
    TIER_WIDTH,
    OrchestratorConfig,
    ScoringConfig,
)
from orchestrator.pending_age import read_pending_age
from orchestrator.scheduler import Scheduler, TickContext

REPO_ROOT = Path(orchestrator.__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / 'orchestrator' / 'tests' / 'fixtures' / 'scheduler_candidates'
BASELINE_CSV = REPO_ROOT / 'plans' / 'evidence' / 'scheduler-scoring-2026-08-06' / 'candidates_scored.csv'


def load_candidate_fixture() -> dict:
    """The newest committed dump of scripts/dump_scheduler_candidate_fixture.py."""
    dumps = sorted(FIXTURE_DIR.glob('*.json'))
    assert dumps, f'no committed candidate fixture under {FIXTURE_DIR}'
    return json.loads(dumps[-1].read_text())


def fixture_captured_at(fixture: dict) -> datetime:
    return datetime.fromisoformat(fixture['provenance']['captured_at'])


def fixture_as_tasks(fixture: dict) -> list[dict]:
    """The fixture's candidates as scheduler task dicts."""
    return [
        {'id': c['id'], 'status': 'pending', 'priority': c['tier'],
         'metadata': {} if c['pending_since'] is None else {'pending_since': c['pending_since']}}
        for c in fixture['candidates']
    ]


def load_baseline_rows() -> list[dict]:
    with BASELINE_CSV.open(newline='') as fh:
        return list(csv.DictReader(fh))


def assert_armed_population_floors(fixture: dict) -> None:
    """A degenerate fixture fails here, before any score is certified against it."""
    candidates = fixture['candidates']
    captured_at = fixture_captured_at(fixture)

    assert len(candidates) >= 250
    assert len({c['tier'] for c in candidates}) >= 3
    assert sum(c['transitive_count'] >= 1 for c in candidates) >= 20
    assert captured_at.utcoffset() is not None
    covered = sum(read_pending_age(t, captured_at).covered for t in fixture_as_tasks(fixture))
    assert covered / len(candidates) >= ScoringConfig().new_scoring_min_anchor_coverage


def baseline_max_tie() -> tuple[float, int]:
    return Counter(float(r['score']) for r in load_baseline_rows()).most_common(1)[0]


class TestCandidateFixtureFloors:
    def test_fixture_is_a_realistic_armed_population(self):
        assert_armed_population_floors(load_candidate_fixture())


class TestBaselineEvidence:
    """The 2026-08-06 saturation the restructure exists to remove, derived from the committed CSV."""

    def test_baseline_saturation_is_what_the_csv_records(self):
        rows = load_baseline_rows()
        scores = Counter(float(r['score']) for r in rows)

        assert len(rows) == 294
        assert len(scores) == 47
        assert sum(r['capped'] == 'True' for r in rows) == 251
        assert baseline_max_tie() == (2999.0, 117)


HOUR = 3600.0


@pytest.fixture
def scheduler() -> Scheduler:
    scheduler = Scheduler(OrchestratorConfig(max_per_module=1, scoring=ScoringConfig()))
    scheduler.finish_startup()
    return scheduler


def _bonus(scheduler: Scheduler, *, age_secs: float = 0.0, transitive_count: int = 0,
           has_work_product: bool = False, tier: str = 'medium') -> float:
    score = scheduler._compute_score(
        tier, age_secs=age_secs, transitive_count=transitive_count, has_work_product=has_work_product,
    )
    return score - TIER_BASE[tier]


class TestSubBudgetFormula:
    """C2 at the shipped defaults, as known-answer arithmetic."""

    @pytest.mark.parametrize(('age_secs', 'expected'), [
        (0.0, 0.0), (24 * HOUR, 125.0), (72 * HOUR, 250.0), (168 * HOUR, 350.0),
    ])
    def test_age_term(self, scheduler, age_secs, expected):
        assert _bonus(scheduler, age_secs=age_secs) == pytest.approx(expected, abs=1e-9)

    def test_age_half_life_gives_exactly_half_the_budget(self, scheduler):
        cfg = scheduler.config.scoring

        assert _bonus(scheduler, age_secs=cfg.age_half_secs) == cfg.age_budget / 2

    @pytest.mark.parametrize(('d', 'expected'), [
        (0, 0.0), (1, 59.47196), (4, 138.08961), (32, 300.0), (188, 300.0),
    ])
    def test_cpm_term(self, scheduler, d, expected):
        assert _bonus(scheduler, transitive_count=d) == pytest.approx(expected, abs=1e-4)

    def test_cpm_term_clips_at_the_cap(self, scheduler):
        cfg = scheduler.config.scoring

        assert _bonus(scheduler, transitive_count=cfg.cpm_d_cap) == cfg.cpm_budget
        assert _bonus(scheduler, transitive_count=188) == cfg.cpm_budget

    def test_continuity_is_exactly_additive(self, scheduler):
        inputs = {'age_secs': 30 * HOUR, 'transitive_count': 3, 'tier': 'high'}

        with_work = _bonus(scheduler, has_work_product=True, **inputs)
        without = _bonus(scheduler, has_work_product=False, **inputs)

        assert with_work - without == pytest.approx(scheduler.config.scoring.continuity_credit, abs=1e-9)

    @pytest.mark.parametrize('tier', PRIORITY_TIERS)
    def test_tier_base_dominates(self, scheduler, tier):
        cfg = scheduler.config.scoring

        assert TIER_BASE[tier] + cfg.cpm_budget + cfg.age_budget + cfg.continuity_credit < TIER_BASE[tier] + TIER_WIDTH

    def test_score_never_saturates_in_age(self, scheduler):
        year = 365 * 24 * HOUR
        ages = [0.0, 1.0, HOUR, 24 * HOUR, 72 * HOUR, 30 * 24 * HOUR, year, 100 * year]

        scores = [scheduler._compute_score('low', age_secs=a, transitive_count=32, has_work_product=True)
                  for a in ages]

        assert all(lo < hi for lo, hi in zip(scores, scores[1:], strict=False)), scores
        assert max(scores) < TIER_BASE['low'] + TIER_WIDTH - 1

    def test_score_strictly_increases_in_d_up_to_the_cap(self, scheduler):
        cap = scheduler.config.scoring.cpm_d_cap

        bonuses = [_bonus(scheduler, transitive_count=d) for d in range(cap + 1)]

        assert all(lo < hi for lo, hi in zip(bonuses, bonuses[1:], strict=False)), bonuses

    def test_only_the_tier_can_be_passed_positionally(self, scheduler):
        parameters = inspect.signature(scheduler._compute_score).parameters

        positional = [name for name, p in parameters.items() if p.kind is not inspect.Parameter.KEYWORD_ONLY]
        assert positional == ['tier']


class TestLegacyFormula:
    """The I-6 fallback: today's clipped formula, still live on an unarmed tick."""

    @pytest.mark.parametrize(('tier', 'age', 'd'), [
        ('medium', 0, 0), ('high', 5, 2), ('low', 30, 9), ('medium', 1_000_000, 0),
    ])
    def test_legacy_formula_is_intact(self, scheduler, tier, age, d):
        cfg = scheduler.config
        expected = TIER_BASE[tier] + min(cfg.age_alpha * age + cfg.cpm_beta * math.log1p(d), TIER_WIDTH - 1)

        assert scheduler._compute_score_legacy(tier, age, d) == expected

    def test_legacy_formula_still_caps(self, scheduler):
        assert scheduler._compute_score_legacy('medium', 1_000_000, 0) - TIER_BASE['medium'] == 999.0


def _tick(tasks: list[dict], transitive_counts: dict[str, int], max_id: int) -> TickContext:
    return TickContext(
        tasks=tasks,
        status_map={t['id']: 'pending' for t in tasks},
        tasks_by_id={t['id']: t for t in tasks},
        max_id=max_id,
        transitive_counts={t['id']: transitive_counts[t['id']] for t in tasks},
        candidates=tasks,
    )


def _fixture_tick(fixture: dict) -> TickContext:
    counts = {c['id']: c['transitive_count'] for c in fixture['candidates']}
    return _tick(fixture_as_tasks(fixture), counts, max(int(tid) for tid in counts))


def _scores_by_id(scheduler: Scheduler, ctx: TickContext) -> dict[str, float]:
    return {tid: score for score, tid, _task, _tier in scheduler._score_candidates(ctx)}


class TestRow4NonSaturation:
    """Boundary row 4: over the live pending population, the tick's scores are spread, unsaturated and CPM-live."""

    @pytest.fixture
    def fixture(self) -> dict:
        return load_candidate_fixture()

    @pytest.fixture
    def frozen(self, fixture) -> Scheduler:
        captured_at = fixture_captured_at(fixture)
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1, scoring=ScoringConfig()),
                              wall_time_source=lambda: captured_at)
        scheduler.finish_startup()
        return scheduler

    def test_the_live_population_scores_without_saturation(self, fixture, frozen):
        assert_armed_population_floors(fixture)
        baseline_score, baseline_tie = baseline_max_tie()
        assert baseline_tie == 117
        tiers = {c['id']: c['tier'] for c in fixture['candidates']}

        scores = _scores_by_id(frozen, _fixture_tick(fixture))

        n = len(fixture['candidates'])
        assert len(scores) == n
        distinct = len(set(scores.values()))
        tie_score, tie_size = Counter(scores.values()).most_common(1)[0]
        assert distinct >= 0.90 * n, (
            f'{distinct}/{n} distinct scores ({distinct / n:.3f}); the baseline had 47 of 294')
        assert tie_size <= 10, (
            f'{tie_size} candidates tie at {tie_score}; the baseline tied {baseline_tie} at {baseline_score}')
        saturated = [tid for tid, s in scores.items() if s >= TIER_BASE[tiers[tid]] + TIER_WIDTH - 1]
        assert not saturated, f'{len(saturated)} saturated candidates (baseline 251 of 294): {saturated[:20]}'

    def test_every_dependent_count_raises_the_score(self, fixture, frozen):
        tick = _fixture_tick(fixture)
        scores = _scores_by_id(frozen, tick)
        counterfactual = _scores_by_id(
            frozen, _tick(tick.candidates, dict.fromkeys(tick.transitive_counts, 0), tick.max_id))

        dead = [c['id'] for c in fixture['candidates']
                if c['transitive_count'] >= 1 and not scores[c['id']] > counterfactual[c['id']]]
        assert not dead, f'CPM term dead for D>=1 candidates: {dead}'

    def test_an_anchored_score_depends_only_on_its_own_task(self, fixture, frozen):
        captured_at = fixture_captured_at(fixture)
        tick = _fixture_tick(fixture)
        scores = _scores_by_id(frozen, tick)

        moved = []
        for candidate, task in zip(fixture['candidates'], tick.candidates, strict=True):
            if read_pending_age(task, captured_at).covered:
                alone = _scores_by_id(frozen, _tick([task], tick.transitive_counts, tick.max_id))[candidate['id']]
                if alone != scores[candidate['id']]:
                    moved.append((candidate['id'], alone, scores[candidate['id']]))
            else:
                expected = frozen._compute_score(candidate['tier'], age_secs=0.0,
                                                 transitive_count=candidate['transitive_count'],
                                                 has_work_product=False)
                if expected != scores[candidate['id']]:
                    moved.append((candidate['id'], expected, scores[candidate['id']]))
        assert not moved, f'(id, expected, full-set score) for scores that depend on the candidate set: {moved[:20]}'
