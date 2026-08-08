"""Tests for orchestrator.hold_history — the module hold-duration predictor.

Task 3822 / plans/scheduler-dispatch-scoring-and-lock-layer-prd.md task ζ.

Every expected duration and median asserted here is hand-computed in
``_hold_history_fixtures`` from a fixed whole-second timeline; nothing is read
back out of the implementation under test.
"""

from __future__ import annotations

import pytest

import _hold_history_fixtures as F
from orchestrator.hold_history import HoldSpan, iter_hold_spans


def _by_key(spans) -> dict[tuple[str, str], HoldSpan]:
    """Index spans by (task_id, module) — unique for every trace used here."""
    indexed = {(s.task_id, s.module): s for s in spans}
    assert len(indexed) == len(list(spans)), 'trace produced duplicate (task, module) keys'
    return indexed


# --- clean acquire -> release pairing -------------------------------------


def test_clean_pair_yields_one_span_with_the_hand_computed_duration():
    rows = [
        F.acquire(1, 0, 'T1', ['orchestrator/src']),
        F.release(2, 60, 'T1', ['orchestrator/src']),
    ]

    spans = list(iter_hold_spans(rows))

    assert len(spans) == 1
    span = spans[0]
    assert isinstance(span, HoldSpan)
    assert span.task_id == 'T1'
    assert span.module == 'orchestrator/src'
    assert span.duration == pytest.approx(60.0)
    assert span.truncated is False


def test_multi_module_acquire_yields_one_span_per_task_module_pair():
    rows = [
        F.acquire(1, 0, 'T1', ['orchestrator/src', 'shared/src']),
        F.release(2, 90, 'T1', ['orchestrator/src', 'shared/src']),
    ]

    spans = _by_key(iter_hold_spans(rows))

    assert set(spans) == {('T1', 'orchestrator/src'), ('T1', 'shared/src')}
    for span in spans.values():
        assert span.duration == pytest.approx(90.0)
        assert span.truncated is False


def test_span_start_and_end_are_the_row_timestamps_in_posix_seconds():
    rows = [
        F.acquire(1, 0, 'T1', ['orchestrator/src']),
        F.release(2, 60, 'T1', ['orchestrator/src']),
    ]

    span = list(iter_hold_spans(rows))[0]

    assert span.start == pytest.approx(F.BASE_TS.timestamp())
    assert span.end == pytest.approx(F.BASE_TS.timestamp() + 60.0)


# --- partial (plan_refinement) release ------------------------------------


def test_partial_release_closes_only_the_modules_it_names():
    """``release_subset`` names only the narrowed-away modules (scheduler.py:6954-6959).

    The task keeps holding everything else, so a partial release must NOT
    close the whole hold — shared/src runs on to its own later release.
    """
    rows = [
        F.acquire(3, 100, 'T2', ['orchestrator/src', 'shared/src']),
        F.release(4, 220, 'T2', ['orchestrator/src'], reason='plan_refinement'),
        F.release(5, 300, 'T2', ['shared/src']),
    ]

    spans = _by_key(iter_hold_spans(rows))

    assert spans[('T2', 'orchestrator/src')].duration == pytest.approx(120.0)  # 220 - 100
    # 300 - 100, NOT 220 - 100: the partial release left this module open.
    assert spans[('T2', 'shared/src')].duration == pytest.approx(200.0)
    assert all(not s.truncated for s in spans.values())


# --- orphan release --------------------------------------------------------


def test_orphan_release_yields_no_span():
    """PARKING_MODEL_REPORT.md:100 — "release ignored; span never opened"."""
    rows = [F.release(6, 350, 'T3', ['orchestrator/src'])]

    assert list(iter_hold_spans(rows)) == []


def test_orphan_release_does_not_disturb_a_concurrent_holder():
    rows = [
        F.acquire(1, 0, 'T1', ['orchestrator/src']),
        F.release(6, 350, 'T3', ['orchestrator/src']),  # different task — orphan
        F.release(2, 400, 'T1', ['orchestrator/src']),
    ]

    spans = list(iter_hold_spans(rows))

    assert len(spans) == 1
    assert spans[0].task_id == 'T1'
    assert spans[0].duration == pytest.approx(400.0)


def test_clean_prefix_of_the_canonical_trace():
    """Rows 1-6 of the canonical fixture: two clean holds, one partial, one orphan."""
    rows = F.build_trace()[:6]

    spans = _by_key(iter_hold_spans(rows))

    assert set(spans) == {
        ('T1', 'orchestrator/src'),
        ('T2', 'orchestrator/src'),
        ('T2', 'shared/src'),
    }
    assert spans[('T1', 'orchestrator/src')].duration == pytest.approx(60.0)
    assert spans[('T2', 'orchestrator/src')].duration == pytest.approx(120.0)
    assert spans[('T2', 'shared/src')].duration == pytest.approx(200.0)
    assert all(not s.truncated for s in spans.values())
