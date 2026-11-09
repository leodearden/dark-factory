"""scripts/check_run_completion.py: the weighted completion measure, the run
selection over the real find_tasks_by_metadata tool, the predicate CLI and the
--pending report."""

from __future__ import annotations

import json
import math

import check_run_completion as crc
import pytest

RUN = 'review-dark_factory-20261109-1'


def _row(task_id: int, priority: str | None, status: str) -> dict[str, object]:
    return {'task_id': task_id, 'priority': priority, 'status': status}


def _rows(*spec: tuple[str | None, str]) -> list[dict[str, object]]:
    return [_row(task_id, priority, status) for task_id, (priority, status) in enumerate(spec, start=1)]


BOUNDARY_ROW_1 = _rows(('high', 'pending'), ('medium', 'done'), ('low', 'done'))
BOUNDARY_ROW_1_JSON = (
    '{"run":"' + RUN + '","verdict":"not_yet","share":0.5,"threshold":0.7,"landed":3,"pending":3,'
    '"total":3,"vacuous":false,"by_priority":{"high":[0,1,0],"medium":[1,0,0],"low":[1,0,0]}}'
)


def _naive_float_pass(landed: int, pending: int, threshold: float) -> bool:
    return landed >= threshold * (landed + pending)


def test_priority_weights_are_the_contract_numbers():
    assert crc.PRIORITY_WEIGHTS == {'critical': 4, 'high': 3, 'medium': 2, 'low': 1, 'polish': 1}


def test_boundary_row_1_is_not_yet_at_half():
    verdict = crc.weighted_completion(BOUNDARY_ROW_1, 0.7, run_id=RUN)

    assert (verdict.landed, verdict.pending, verdict.total) == (3, 3, 3)
    assert verdict.share == 0.5
    assert verdict.passed is False
    assert verdict.vacuous is False
    assert verdict.verdict == 'not_yet'
    assert verdict.error is None
    assert list(verdict.by_priority.items()) == [('high', (0, 1, 0)), ('medium', (1, 0, 0)), ('low', (1, 0, 0))]
    assert verdict.to_json() == BOUNDARY_ROW_1_JSON


def test_boundary_row_11_passes_at_exactly_the_threshold():
    rows = _rows(('critical', 'done'), ('high', 'done'), ('high', 'pending'))

    verdict = crc.weighted_completion(rows, 0.7, run_id=RUN)

    assert (verdict.landed, verdict.pending) == (7, 3)
    assert verdict.passed is True
    assert verdict.verdict == 'pass'


def test_three_tenths_passes_at_point_three():
    rows = _rows(('high', 'done'), ('critical', 'pending'), ('high', 'pending'))

    verdict = crc.weighted_completion(rows, 0.3, run_id=RUN)

    assert (verdict.landed, verdict.pending) == (3, 7)
    assert verdict.passed is True


def test_exact_threshold_passes_where_the_float_product_overshoots():
    rows = _rows(
        ('critical', 'done'), ('high', 'done'),
        ('critical', 'pending'), ('critical', 'pending'), ('critical', 'pending'), ('critical', 'pending'),
        ('medium', 'pending'),
    )
    assert not _naive_float_pass(7, 18, 0.28)

    verdict = crc.weighted_completion(rows, 0.28, run_id=RUN)

    assert (verdict.landed, verdict.pending) == (7, 18)
    assert verdict.passed is True
    assert verdict.verdict == 'pass'


def test_six_tenths_is_not_yet_at_point_seven():
    rows = _rows(('critical', 'done'), ('medium', 'done'), ('critical', 'pending'))

    verdict = crc.weighted_completion(rows, 0.7, run_id=RUN)

    assert (verdict.landed, verdict.pending) == (6, 4)
    assert verdict.passed is False
    assert verdict.verdict == 'not_yet'


def test_a_missing_priority_weighs_as_medium():
    rows = [_row(1, None, 'done'), {'task_id': 2, 'status': 'pending'}]

    verdict = crc.weighted_completion(rows, 0.7, run_id=RUN)

    assert (verdict.landed, verdict.pending) == (2, 2)
    assert dict(verdict.by_priority) == {'medium': (1, 1, 0)}


@pytest.mark.parametrize('status', ['pending', 'in-progress', 'blocked', 'deferred', 'review'])
def test_every_non_terminal_status_is_pending_weight(status):
    verdict = crc.weighted_completion([_row(1, 'high', status)], 0.7, run_id=RUN)

    assert (verdict.landed, verdict.pending, verdict.total) == (0, 3, 1)
    assert dict(verdict.by_priority) == {'high': (0, 1, 0)}


def test_cancelled_weighs_nothing_but_counts_in_total():
    rows = _rows(('critical', 'cancelled'), ('high', 'done'), ('high', 'pending'))

    verdict = crc.weighted_completion(rows, 0.5, run_id=RUN)

    assert (verdict.landed, verdict.pending, verdict.total) == (3, 3, 3)
    assert dict(verdict.by_priority) == {'critical': (0, 0, 1), 'high': (1, 1, 0)}


def test_an_all_cancelled_run_passes_vacuously():
    rows = _rows(('high', 'cancelled'), ('low', 'cancelled'))

    verdict = crc.weighted_completion(rows, 0.7, run_id=RUN)

    assert verdict.vacuous is True
    assert verdict.share is None
    assert verdict.passed is True
    assert verdict.verdict == 'pass'
    assert verdict.total == 2


def test_no_rows_is_an_empty_run_error():
    verdict = crc.weighted_completion([], 0.7, run_id=RUN)

    assert verdict.total == 0
    assert verdict.share is None
    assert verdict.passed is False
    assert verdict.verdict == 'error'
    assert verdict.error == 'empty_run'
    assert json.loads(verdict.to_json())['error'] == 'empty_run'
    assert '"error":"empty_run"' in verdict.to_json()


def test_an_unknown_priority_is_refused_naming_task_and_value():
    rows = [_row(4, 'high', 'done'), _row(5, 'urgent', 'pending')]

    with pytest.raises(crc.UnknownPriority) as raised:
        crc.weighted_completion(rows, 0.7, run_id=RUN)

    assert raised.value.task_id == 5
    assert raised.value.value == 'urgent'


@pytest.mark.parametrize('threshold', [0, -0.1, 1.5, math.nan])
def test_a_threshold_outside_zero_one_is_refused(threshold):
    with pytest.raises(ValueError, match=r'\(0, 1\]'):
        crc.weighted_completion(BOUNDARY_ROW_1, threshold, run_id=RUN)


def test_a_threshold_of_one_is_accepted():
    verdict = crc.weighted_completion(_rows(('high', 'done')), 1.0, run_id=RUN)

    assert verdict.passed is True


def test_the_worst_case_verdict_json_fits_the_note_cap():
    tier = crc.TierCounts(999999, 999999, 999998)
    landed = sum(weight * tier.done for weight in crc.PRIORITY_WEIGHTS.values())
    pending = sum(weight * tier.open for weight in crc.PRIORITY_WEIGHTS.values())
    verdict = crc.CompletionVerdict(
        run_id='hotspot-survey-dark_factory-20261004-99',
        threshold=0.6666666666666666,
        share=10999989 / 21999967,
        landed=landed,
        pending=pending,
        total=sum(tier) * len(crc.PRIORITY_WEIGHTS),
        vacuous=False,
        passed=False,
        by_priority={name: tier for name in crc.PRIORITY_WEIGHTS},
    )

    assert len(verdict.to_json()) <= 400
