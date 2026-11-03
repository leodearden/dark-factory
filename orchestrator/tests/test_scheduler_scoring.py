"""Dispatch scoring β (task 3819; plans/scheduler-dispatch-scoring-and-lock-layer-prd.md C2, I-4, I-6, row 4, row 9)."""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

import orchestrator
from orchestrator.anchor_coverage_gate import (
    ANCHORLESS_MIN_SAMPLES,
    ANCHORLESS_SENTINEL,
    ANCHORLESS_WINDOW,
    STREAK_ESCALATION_THRESHOLD,
    STREAK_SENTINEL,
)
from orchestrator.config import (
    PRIORITY_TIERS,
    TIER_BASE,
    TIER_WIDTH,
    OrchestratorConfig,
    ScoringConfig,
)
from orchestrator.event_store import EventStore, EventType
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

    @pytest.mark.parametrize('positional_args', [
        ('medium', 3600.0, 0, False),
        ('medium', 5, 2),
    ])
    def test_a_positional_age_or_count_is_rejected(self, scheduler, positional_args):
        with pytest.raises(TypeError):
            scheduler._compute_score(*positional_args)


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


NOW = datetime(2026, 10, 12, 12, 0, 0, tzinfo=UTC)
SHARED_ANCHOR = '2026-10-10T08:00:00.000Z'


def _task(task_id: str, *, priority: str = 'medium', pending_since: str | None = SHARED_ANCHOR) -> dict:
    metadata: dict = {'files': [f'mod{task_id}']}
    if pending_since is not None:
        metadata['pending_since'] = pending_since
    return {'id': task_id, 'title': f'Task {task_id}', 'status': 'pending', 'priority': priority,
            'dependencies': [], 'metadata': metadata}


def _frozen_scheduler(event_store: EventStore | None = None) -> Scheduler:
    scheduler = Scheduler(OrchestratorConfig(max_per_module=1, max_concurrent_tasks=50, scoring=ScoringConfig()),
                          event_store=event_store, wall_time_source=lambda: NOW)
    scheduler.finish_startup()
    return scheduler


async def _dispatch_order(tasks: list[dict]) -> list[str]:
    scheduler = _frozen_scheduler()
    scheduler.get_tasks = AsyncMock(return_value=tasks)
    order = []
    for _ in tasks:
        assignment = await scheduler.acquire_next()
        assert assignment is not None, f'nothing dispatched after {order}'
        order.append(assignment.task_id)
    return order


class TestNumericTieBreak:
    """I-4 / D7: at equal scores, the older (numerically smaller) id dispatches first."""

    @pytest.mark.asyncio
    async def test_equal_scores_dispatch_in_numeric_id_order(self):
        assert await _dispatch_order([_task('30'), _task('4'), _task('200')]) == ['4', '30', '200']

    @pytest.mark.asyncio
    async def test_a_four_digit_id_dispatches_before_a_five_digit_one(self):
        assert await _dispatch_order([_task('10000'), _task('9999')]) == ['9999', '10000']

    @pytest.mark.asyncio
    async def test_non_numeric_ids_follow_every_numeric_id_in_string_order(self):
        tasks = [_task('zeta'), _task('12'), _task('alpha'), _task('3')]

        assert await _dispatch_order(tasks) == ['3', '12', 'alpha', 'zeta']

    @pytest.mark.asyncio
    async def test_a_digit_that_int_cannot_parse_is_a_non_numeric_id(self):
        assert await _dispatch_order([_task('\u00b2'), _task('7')]) == ['7', '\u00b2']

    @pytest.mark.asyncio
    async def test_score_still_dominates_the_tie_break(self):
        tasks = [_task('5', priority='medium'), _task('500', priority='high')]

        assert await _dispatch_order(tasks) == ['500', '5']


def _population(first_id: int, *, anchored: int, anchorless: int) -> list[dict]:
    """Candidates with varied tiers and anchors; the first *anchored* carry pending_since."""
    tiers = ('high', 'medium', 'low')
    tasks = []
    for i in range(anchored + anchorless):
        anchor = f'2026-10-{1 + i % 10:02d}T0{i % 10}:00:00.000Z' if i < anchored else None
        tasks.append(_task(str(first_id + i), priority=tiers[i % 3], pending_since=anchor))
    return tasks


def _gate_tick(scheduler: Scheduler, tasks: list[dict]) -> TickContext:
    """A tick over *tasks* whose legacy ages are non-zero, as acquire_next's preamble leaves them."""
    max_id = max(int(t['id']) for t in tasks)
    scheduler._update_age_anchors(tasks, max_id)
    return _tick(tasks, {t['id']: int(t['id']) % 4 for t in tasks}, max_id)


def _legacy_scores(scheduler: Scheduler, ctx: TickContext) -> dict[str, float]:
    return {t['id']: scheduler._compute_score_legacy(
        t['priority'], scheduler._compute_age(t['id'], ctx.max_id), ctx.transitive_counts[t['id']])
        for t in ctx.candidates}


def _c2_scores(scheduler: Scheduler, ctx: TickContext) -> dict[str, float]:
    return {t['id']: scheduler._compute_score(
        t['priority'], age_secs=read_pending_age(t, NOW).age_secs,
        transitive_count=ctx.transitive_counts[t['id']], has_work_product=False)
        for t in ctx.candidates}


def _low_coverage_events(store: EventStore) -> list[dict]:
    return [row['data'] for row in store.fetch_events_by_type(EventType.scoring_anchor_coverage_low)]


class TestRow9AnchorCoverageGate:
    """Boundary row 9 (I-6 / D12): below the coverage threshold the whole tick scores on the legacy formula."""

    @pytest.fixture
    def store(self, tmp_path) -> EventStore:
        return EventStore(tmp_path / 'runs.db', run_id='gate')

    def test_half_coverage_scores_every_candidate_on_the_legacy_formula(self, store):
        scheduler = _frozen_scheduler(store)
        ctx = _gate_tick(scheduler, _population(100, anchored=5, anchorless=5))

        scores = _scores_by_id(scheduler, ctx)

        assert scores == _legacy_scores(scheduler, ctx)
        anchorless = [scores[t['id']] - TIER_BASE[t['priority']] for t in ctx.candidates[5:]]
        assert len(set(anchorless)) > 1, f'the anchorless half collapsed to one bonus: {anchorless}'
        assert _low_coverage_events(store) == [{
            'coverage': 0.5, 'threshold': 0.90, 'candidates': 10, 'with_anchor': 5, 'consecutive_ticks': 1,
        }]

    def test_full_coverage_scores_on_c2_and_emits_nothing(self, store):
        scheduler = _frozen_scheduler(store)
        ctx = _gate_tick(scheduler, _population(100, anchored=10, anchorless=0))

        assert _scores_by_id(scheduler, ctx) == _c2_scores(scheduler, ctx)
        assert _low_coverage_events(store) == []

    def test_coverage_exactly_at_the_threshold_arms(self, store):
        scheduler = _frozen_scheduler(store)
        ctx = _gate_tick(scheduler, _population(100, anchored=9, anchorless=1))

        assert _scores_by_id(scheduler, ctx) == _c2_scores(scheduler, ctx)
        assert _low_coverage_events(store) == []

    def test_coverage_just_below_the_threshold_falls_back(self, store):
        scheduler = _frozen_scheduler(store)
        ctx = _gate_tick(scheduler, _population(100, anchored=8, anchorless=1))

        assert _scores_by_id(scheduler, ctx) == _legacy_scores(scheduler, ctx)
        assert len(_low_coverage_events(store)) == 1

    def test_a_hot_retune_moves_the_gate_on_the_next_tick(self, store):
        scheduler = _frozen_scheduler(store)
        half = _gate_tick(scheduler, _population(100, anchored=5, anchorless=5))
        nine_of_ten = _gate_tick(scheduler, _population(200, anchored=9, anchorless=1))

        scheduler.config.scoring.new_scoring_min_anchor_coverage = 0.5
        assert _scores_by_id(scheduler, half) == _c2_scores(scheduler, half)

        scheduler.config.scoring.new_scoring_min_anchor_coverage = 1.0
        assert _scores_by_id(scheduler, nine_of_ten) == _legacy_scores(scheduler, nine_of_ten)

    @pytest.mark.asyncio
    async def test_the_streak_counts_consecutive_unarmed_ticks(self, store):
        scheduler = _frozen_scheduler(store)
        next_id = iter(range(100, 10_000, 10))

        async def tick(tasks: list[dict]) -> None:
            scheduler.get_tasks = AsyncMock(return_value=tasks)
            await scheduler.acquire_next()

        async def low_tick() -> None:
            await tick(_population(next(next_id), anchored=2, anchorless=2))

        await low_tick()
        await low_tick()
        await tick([])
        await low_tick()
        await tick(_population(next(next_id), anchored=4, anchorless=0))
        await low_tick()

        assert [e['consecutive_ticks'] for e in _low_coverage_events(store)] == [1, 2, 3, 1]


def test_the_anchorless_window_holds_more_than_the_sample_floor():
    """Otherwise escape (b) could never sample enough to claim a rate."""
    assert ANCHORLESS_WINDOW > ANCHORLESS_MIN_SAMPLES


class _DownQueue(EscalationQueue):
    def submit(self, escalation: Escalation) -> str:
        raise RuntimeError('escalation queue is down')


def _mixed(*, anchored: int, absent: int = 0, malformed: int = 0, first_id: int = 1) -> list[dict]:
    kinds = [SHARED_ANCHOR] * anchored + [None] * absent + ['not-a-date'] * malformed
    return [_task(str(first_id + i), pending_since=anchor) for i, anchor in enumerate(kinds)]


def _ctx(tasks: list[dict]) -> TickContext:
    return _tick(tasks, {t['id']: 0 for t in tasks}, max(int(t['id']) for t in tasks))


def _run(scheduler: Scheduler, tasks: list[dict], ticks: int) -> None:
    ctx = _ctx(tasks)
    for _ in range(ticks):
        assert scheduler._score_candidates(ctx)


def _filed(queue: EscalationQueue, sentinel: str) -> list[Escalation]:
    return [e for e in queue.get_pending() if e.task_id == sentinel]


class TestStormEscapes:
    """INV-4: a persistent anchor outage reaches a human once, through the escalation queue."""

    @pytest.fixture
    def queue(self, tmp_path) -> EscalationQueue:
        return EscalationQueue(tmp_path / 'esc')

    @pytest.fixture
    def scheduler(self, queue) -> Scheduler:
        scheduler = _frozen_scheduler()
        scheduler.escalation_queue = queue
        return scheduler

    def test_a_persistent_unarmed_streak_files_one_l1(self, scheduler, queue):
        tasks = _mixed(anchored=2, absent=3)

        _run(scheduler, tasks, STREAK_ESCALATION_THRESHOLD - 1)
        assert queue.get_pending() == []

        _run(scheduler, tasks, 1)
        [esc] = queue.get_pending()
        assert (esc.task_id, esc.agent_role, esc.severity, esc.level, esc.category) == (
            STREAK_SENTINEL, 'orchestrator-scheduler', 'blocking', 1, 'infra_issue')
        for fact in ('40.0%', '90.0%', f'{STREAK_ESCALATION_THRESHOLD} consecutive', '3', '4', '5', 'stamp_pending_since'):
            assert fact in esc.detail, fact

        _run(scheduler, tasks, 5)
        assert len(queue.get_pending()) == 1

    def test_an_armed_tick_ends_the_streak(self, scheduler, queue):
        low, healthy = _mixed(anchored=2, absent=3), _mixed(anchored=5)

        _run(scheduler, low, STREAK_ESCALATION_THRESHOLD - 1)
        _run(scheduler, healthy, 1)
        _run(scheduler, low, STREAK_ESCALATION_THRESHOLD - 1)

        assert queue.get_pending() == []

    def test_a_persistent_anchorless_minority_on_armed_ticks_files_one_l1(self, scheduler, queue):
        tasks = _mixed(anchored=95, absent=3, malformed=2)
        floor_ticks = ANCHORLESS_MIN_SAMPLES // len(tasks)

        _run(scheduler, tasks, floor_ticks - 1)
        assert queue.get_pending() == []

        _run(scheduler, tasks, 1)
        [esc] = queue.get_pending()
        assert (esc.task_id, esc.level, esc.category) == (ANCHORLESS_SENTINEL, 1, 'risk_identified')
        for fact in ('15 ABSENT', '10 MALFORMED', '5.0%', '500', '96', '99'):
            assert fact in esc.detail, fact

        _run(scheduler, tasks, 5)
        assert len(queue.get_pending()) == 1

    def test_no_rate_is_claimed_below_the_sample_floor(self, scheduler, queue):
        _run(scheduler, _mixed(anchored=9, absent=1), 40)

        assert queue.get_pending() == []

    def test_the_measured_live_residual_never_files(self, scheduler, queue):
        _run(scheduler, _mixed(anchored=998, absent=1, malformed=1), 10)

        assert queue.get_pending() == []

    def test_an_unarmed_outage_files_only_the_streak_escape(self, scheduler, queue):
        _run(scheduler, _mixed(anchored=50, absent=50), STREAK_ESCALATION_THRESHOLD + 3)

        assert [e.task_id for e in queue.get_pending()] == [STREAK_SENTINEL]

    def test_clean_ticks_bring_the_rate_back_under_the_threshold(self, scheduler, queue):
        _run(scheduler, _mixed(anchored=95, absent=5), 5)
        [esc] = _filed(queue, ANCHORLESS_SENTINEL)

        _run(scheduler, _mixed(anchored=100), 8)
        queue.resolve(esc.id, 'anchors re-stamped')
        _run(scheduler, _mixed(anchored=100), 3)

        assert _filed(queue, ANCHORLESS_SENTINEL) == []

    def test_a_full_window_forgets_its_oldest_samples(self, scheduler, queue):
        window_ticks = ANCHORLESS_WINDOW // 100 + 1
        _run(scheduler, _mixed(anchored=95, absent=5), window_ticks)
        _run(scheduler, _mixed(anchored=100), window_ticks)
        for esc in _filed(queue, ANCHORLESS_SENTINEL):
            queue.resolve(esc.id, 'anchors re-stamped')

        _run(scheduler, _mixed(anchored=100), 1)

        assert _filed(queue, ANCHORLESS_SENTINEL) == []

    def test_without_a_queue_both_escapes_only_warn(self, caplog):
        scheduler = _frozen_scheduler()

        with caplog.at_level(logging.WARNING, logger='orchestrator.sentinel_l1'):
            _run(scheduler, _mixed(anchored=2, absent=3), STREAK_ESCALATION_THRESHOLD + 1)
            _run(scheduler, _mixed(anchored=95, absent=5), 6)

        warned = ' '.join(r.getMessage() for r in caplog.records if r.name == 'orchestrator.sentinel_l1')
        assert STREAK_SENTINEL in warned
        assert ANCHORLESS_SENTINEL in warned

    def test_a_failing_submit_never_aborts_scoring(self, tmp_path):
        scheduler = _frozen_scheduler()
        scheduler.escalation_queue = _DownQueue(tmp_path / 'esc')

        _run(scheduler, _mixed(anchored=2, absent=3), STREAK_ESCALATION_THRESHOLD + 1)
        _run(scheduler, _mixed(anchored=95, absent=5), 6)
