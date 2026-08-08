"""Tests for orchestrator.hold_history — the module hold-duration predictor.

Task 3822 / plans/scheduler-dispatch-scoring-and-lock-layer-prd.md task ζ.

Every expected duration and median asserted here is hand-computed in
``_hold_history_fixtures`` from a fixed whole-second timeline; nothing is read
back out of the implementation under test.
"""

from __future__ import annotations

import _hold_history_fixtures as F
import pytest

from orchestrator.hold_history import HoldHistory, HoldSpan, iter_hold_spans


def _offset(posix_seconds: float) -> float:
    """Seconds since ``BASE_TS`` — the scale every timestamp assertion uses.

    ``pytest.approx`` is RELATIVE by default (rel=1e-6), so comparing raw POSIX
    epoch values (~1.79e9) carries a ~1800s tolerance — wide enough to swallow
    every offset in this fixture, making such an assertion unable to fail for
    the difference it claims to detect.  Comparing offsets keeps the tolerance
    at the sub-millisecond scale where it belongs.
    """
    return posix_seconds - F.BASE_TS.timestamp()


def _by_key(spans) -> dict[tuple[str, str], HoldSpan]:
    """Index spans by (task_id, module) — unique for every trace used here.

    ``iter_hold_spans`` is a generator, so materialise before counting: a
    second pass over an exhausted iterator would silently see zero spans.
    """
    materialized = list(spans)
    indexed = {(s.task_id, s.module): s for s in materialized}
    assert len(indexed) == len(materialized), 'trace produced duplicate (task, module) keys'
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

    assert _offset(span.start) == pytest.approx(0.0, abs=1e-6)
    assert _offset(span.end) == pytest.approx(60.0)


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


# --- double acquire --------------------------------------------------------


def test_double_acquire_force_closes_the_previous_span_at_the_new_acquire():
    """PARKING_MODEL_REPORT.md:101 — "previous span force-closed at new acquire".

    Rows 7-9 of the canonical trace.  The first hold is real — it DID block
    other tasks from 400 to 580 — so it must be yielded, marked ``truncated``
    because its end was imposed rather than observed.
    """
    rows = F.build_trace()[6:9]

    spans = list(iter_hold_spans(rows))

    assert len(spans) == 2, 'the force-closed span must be yielded, not discarded'
    first, second = spans
    assert (first.task_id, first.module) == ('T4', 'orchestrator/src')
    assert first.duration == pytest.approx(180.0)  # 580 - 400
    assert first.truncated is True
    assert second.duration == pytest.approx(120.0)  # 700 - 580, the reopened span
    assert second.truncated is False


def test_double_acquire_reopens_at_the_new_acquire_not_the_original():
    """The re-opened span must start at 580, so a later release measures 700-580.

    Keeping the ORIGINAL start would report 300s — silently double-counting the
    already-yielded 180s prefix.
    """
    rows = F.build_trace()[6:9]

    second = list(iter_hold_spans(rows))[1]

    assert _offset(second.start) == pytest.approx(580.0)
    assert _offset(second.start) != pytest.approx(400.0)  # NOT the original acquire
    assert _offset(second.end) == pytest.approx(700.0)


def test_double_acquire_closes_only_the_re_acquired_module():
    """A re-acquire naming one module must not disturb the task's other holds."""
    rows = [
        F.acquire(1, 0, 'T1', ['orchestrator/src', 'shared/src']),
        F.acquire(2, 50, 'T1', ['orchestrator/src']),          # DOUBLE on one module
        F.release(3, 200, 'T1', ['orchestrator/src', 'shared/src']),
    ]

    spans = list(iter_hold_spans(rows))

    by_module: dict[str, list] = {}
    for s in spans:
        by_module.setdefault(s.module, []).append(s)

    assert [s.duration for s in by_module['orchestrator/src']] == [
        pytest.approx(50.0),   # force-closed at the re-acquire
        pytest.approx(150.0),  # 200 - 50
    ]
    assert by_module['orchestrator/src'][0].truncated is True
    assert by_module['orchestrator/src'][1].truncated is False
    # shared/src was never re-acquired: one clean 200s span, start untouched.
    assert len(by_module['shared/src']) == 1
    assert by_module['shared/src'][0].duration == pytest.approx(200.0)
    assert by_module['shared/src'][0].truncated is False


# --- service_restart era boundary ------------------------------------------


def test_service_restart_closes_open_spans_and_yields_them_truncated():
    """PARKING_MODEL_REPORT.md:102 — "span closed at process death".

    The hold is NOT discarded: "the lock did block others until then", so the
    observed prefix is a real lower bound and must be counted, flagged
    ``truncated`` because the end was imposed by the boundary.
    """
    rows = F.build_trace()[9:11]  # acquire T5 @800, service_restart @1000

    spans = list(iter_hold_spans(rows))

    assert len(spans) == 1
    span = spans[0]
    assert (span.task_id, span.module) == ('T5', 'fused-memory/src')
    assert span.duration == pytest.approx(200.0)  # 1000 - 800
    assert span.end == pytest.approx(F.BASE_TS.timestamp() + 1000.0)
    assert span.truncated is True


def test_service_restart_closes_every_open_span_not_just_one():
    rows = [
        F.acquire(1, 0, 'T1', ['orchestrator/src']),
        F.acquire(2, 100, 'T2', ['shared/src', 'fused-memory/src']),
        F.service_restart(3, 500),
    ]

    spans = list(iter_hold_spans(rows))

    assert {(s.task_id, s.module) for s in spans} == {
        ('T1', 'orchestrator/src'),
        ('T2', 'shared/src'),
        ('T2', 'fused-memory/src'),
    }
    assert all(s.truncated for s in spans)
    assert all(s.end == pytest.approx(F.BASE_TS.timestamp() + 500.0) for s in spans)


def test_service_restart_task_id_is_a_trigger_not_a_holder():
    """The emit site stamps a *trigger* task id (service_restart.py:747-759).

    Treating it as a lock holder would open a phantom span keyed on a task that
    never acquired anything.  Only the boundary timestamp is load-bearing.
    """
    rows = [
        F.acquire(1, 0, 'T1', ['orchestrator/src']),
        F.service_restart(2, 300, task_id=F.SERVICE_RESTART_TRIGGER_TASK),
        F.release(3, 400, 'T1', ['orchestrator/src']),
    ]

    spans = list(iter_hold_spans(rows))

    assert [s.task_id for s in spans] == ['T1']
    assert F.SERVICE_RESTART_TRIGGER_TASK not in {s.task_id for s in spans}
    # The boundary already closed T1's span, so the later release is an orphan.
    assert spans[0].duration == pytest.approx(300.0)
    assert spans[0].truncated is True


def test_service_restart_with_nothing_open_yields_nothing():
    rows = [
        F.acquire(1, 0, 'T1', ['orchestrator/src']),
        F.release(2, 60, 'T1', ['orchestrator/src']),
        F.service_restart(3, 900),
    ]

    spans = list(iter_hold_spans(rows))

    assert len(spans) == 1
    assert spans[0].truncated is False  # closed cleanly before the boundary


# --- run_id transition era boundary ----------------------------------------


def test_run_id_transition_closes_open_spans_at_the_last_row_of_the_prior_run():
    """A run change is an era boundary too — ``service_restart`` alone is not enough.

    ``fetch_events_by_type_all_runs`` spans runs by construction, and
    ``service_restart`` has a single emit site scoped to a *backing service*
    (service_restart.py:747-759): an orchestrator crash or systemd restart
    emits no such row at all.  Rows 14-17 of the canonical trace: T7 acquires
    at 1300 and never releases in run-a, whose last row is at 1400.
    """
    rows = F.build_trace()[13:17]

    spans = list(iter_hold_spans(rows))

    by_key = {(s.task_id, s.module): s for s in spans}
    assert by_key[('T8', 'orchestrator/src')].duration == pytest.approx(40.0)
    assert by_key[('T8', 'orchestrator/src')].truncated is False

    stranded = by_key[('T7', 'shared/src')]
    assert stranded.duration == pytest.approx(100.0)  # 1400 - 1300
    assert stranded.end == pytest.approx(F.BASE_TS.timestamp() + 1400.0)
    assert stranded.truncated is True


def test_run_id_transition_closes_at_the_prior_run_not_at_the_new_row():
    """Closing at the NEW run's first row would invent the inter-run gap.

    1400 is the last instant the trace shows the lock still held; 1500 is the
    first instant of a different era.  Only the former is observed.

    Asserted on the OFFSET from ``BASE_TS``, never on the raw epoch value:
    ``pytest.approx`` is relative by default, so at epoch scale (~1.79e9) its
    tolerance is ~1800s and a 100s discrepancy would pass unnoticed.
    """
    rows = F.build_trace()[13:17]

    stranded = {(s.task_id, s.module): s for s in iter_hold_spans(rows)}[('T7', 'shared/src')]
    end_offset = stranded.end - F.BASE_TS.timestamp()

    assert end_offset == pytest.approx(1400.0)  # last row of run-a
    assert end_offset != pytest.approx(1500.0)  # NOT run-b's first row
    assert stranded.duration == pytest.approx(100.0)


def test_run_transition_does_not_wait_for_a_later_service_restart():
    """The D11 infinite-hold failure, directly.

    Without the run-transition boundary the stranded span would stay open until
    the next ``service_restart`` — here two runs and three days later — and
    report a ~259200s hold that never happened.
    """
    three_days = 3 * 24 * 3600
    rows = [
        F.acquire(1, 0, 'T1', ['shared/src']),                       # run-a, never released
        F.release(2, 60, 'T2', ['orchestrator/src']),                # orphan; last run-a row
        F.acquire(3, three_days, 'T3', ['orchestrator/src'], run_id='run-b'),
        F.service_restart(4, three_days + 100, run_id='run-b'),
    ]

    spans = list(iter_hold_spans(rows))

    stranded = {(s.task_id, s.module): s for s in spans}[('T1', 'shared/src')]
    assert stranded.duration == pytest.approx(60.0)  # NOT three_days + 100
    assert stranded.duration < three_days
    assert stranded.truncated is True


def test_rows_within_one_run_are_not_a_boundary():
    """Only a CHANGE of run_id is a boundary; a long quiet stretch is not."""
    rows = [
        F.acquire(1, 0, 'T1', ['shared/src']),
        F.acquire(2, 100, 'T2', ['orchestrator/src']),
        F.release(3, 200, 'T2', ['orchestrator/src']),
        F.release(4, 5000, 'T1', ['shared/src']),
    ]

    spans = _by_key(iter_hold_spans(rows))

    assert spans[('T1', 'shared/src')].duration == pytest.approx(5000.0)
    assert all(not s.truncated for s in spans.values())


# --- end of stream ---------------------------------------------------------


def test_span_open_at_end_of_stream_is_dropped():
    """No observed end at all — closing it at "now" would fabricate a duration.

    Distinct from a truncated span, whose end was observed-but-imposed.  This is
    the same refuse-rather-than-fabricate rule the min_samples gate enforces,
    applied at the span layer.
    """
    rows = [F.acquire(1, 0, 'T1', ['orchestrator/src'])]

    assert list(iter_hold_spans(rows)) == []


def test_end_of_stream_drop_leaves_earlier_spans_untouched():
    rows = [
        F.acquire(1, 0, 'T1', ['orchestrator/src']),
        F.release(2, 60, 'T1', ['orchestrator/src']),
        F.acquire(3, 100, 'T2', ['shared/src']),  # open at EOS
    ]

    spans = list(iter_hold_spans(rows))

    assert [(s.task_id, s.module) for s in spans] == [('T1', 'orchestrator/src')]


# --- the whole canonical trace, end to end ---------------------------------


def test_canonical_trace_reproduces_the_hand_computed_samples():
    """The full 21-row oracle: every residue class at once.

    Expectations come from ``_hold_history_fixtures``' hand-computed timeline,
    never read back out of the implementation.
    """
    spans = list(iter_hold_spans(F.build_trace()))

    assert len(spans) == F.EXPECTED_SPAN_COUNT

    by_module: dict[str, list[float]] = {}
    for s in spans:
        by_module.setdefault(s.module, []).append(s.duration)

    assert by_module.keys() == F.EXPECTED_SAMPLES.keys()
    for module, expected in F.EXPECTED_SAMPLES.items():
        assert by_module[module] == pytest.approx(expected), module

    assert {(s.task_id, s.module) for s in spans if s.truncated} == F.EXPECTED_TRUNCATED_SPANS


def test_canonical_trace_omits_the_orphan_and_the_end_of_stream_holder():
    spans = list(iter_hold_spans(F.build_trace()))

    holders = {s.task_id for s in spans}
    assert 'T3' not in holders, 'orphan release must not produce a span'
    assert 'T10' not in holders, 'span open at end of stream must be dropped'
    assert F.SERVICE_RESTART_TRIGGER_TASK not in holders


# ===========================================================================
# HoldHistory — the rolling per-module window and its pooled median
# ===========================================================================
#
# Every history built below carries at least three pooled samples, so these
# assertions stay green once the ``min_samples`` refusal gate lands (step-10).
# The refusal contract itself is asserted separately, not here.


def _history_of(durations: dict[str, list[float]], **kwargs) -> HoldHistory:
    """A :class:`HoldHistory` fed *durations* per module, in the order given."""
    history = HoldHistory(**kwargs)
    for module, values in durations.items():
        for value in values:
            history.record(module, value)
    return history


# --- record_span -----------------------------------------------------------


def test_record_span_files_the_span_under_its_own_module():
    history = HoldHistory(window=10, min_samples=3)

    for start, end in ((0.0, 60.0), (100.0, 220.0), (400.0, 580.0)):
        history.record_span(HoldSpan('T1', 'orchestrator/src', start, end))

    # [60, 120, 180] -> median 120.0
    assert history.predicted_hold(['orchestrator/src']) == pytest.approx(120.0)
    assert history.sample_count(['orchestrator/src']) == 3


def test_record_span_counts_a_truncated_span():
    """PARKING_MODEL_REPORT.md:102 — "the lock did block others until then".

    A truncated span's end was IMPOSED, not fabricated, so its duration is a
    real lower bound on a real hold.  Dropping it would bias the window toward
    the short, tidy holds — exactly the ones least worth predicting.
    """
    history = HoldHistory(window=10, min_samples=3)

    history.record_span(HoldSpan('T1', 'shared/src', 0.0, 10.0))
    history.record_span(HoldSpan('T2', 'shared/src', 0.0, 20.0))
    history.record_span(HoldSpan('T3', 'shared/src', 0.0, 900.0, truncated=True))

    assert history.sample_count(['shared/src']) == 3
    # [10, 20, 900] -> median 20.0.  Were the truncated span dropped, only two
    # samples would remain and this would not be 20.0 at all.
    assert history.predicted_hold(['shared/src']) == pytest.approx(20.0)


def test_record_span_ignores_the_holding_task_identity():
    """The window is per-MODULE: who held it is not part of the key."""
    history = HoldHistory(window=10, min_samples=3)

    history.record_span(HoldSpan('T1', 'shared/src', 0.0, 10.0))
    history.record_span(HoldSpan('T2', 'shared/src', 0.0, 30.0))
    history.record_span(HoldSpan('T3', 'shared/src', 0.0, 50.0))

    assert history.sample_count(['shared/src']) == 3
    assert history.predicted_hold(['shared/src']) == pytest.approx(30.0)


# --- pooling across the modules named --------------------------------------


def test_predicted_hold_pools_every_sample_from_every_module_named():
    """Pooled median — NOT a median of per-module medians, nor the first module's.

    The sample counts are deliberately lopsided (5 vs 3) so the three candidate
    answers separate: pooled median 10.0, median-of-medians 55.0, first-module
    median 10.0 but second-module median 100.0.  Only pooling gives 10.0 while
    ALSO giving 100.0 for ['slow/src'] alone.
    """
    history = _history_of(
        {'fast/src': [10.0] * 5, 'slow/src': [100.0] * 3},
        window=10,
        min_samples=3,
    )

    assert history.predicted_hold(['fast/src']) == pytest.approx(10.0)
    assert history.predicted_hold(['slow/src']) == pytest.approx(100.0)

    # pooled n=8: [10,10,10,10,10,100,100,100] -> (10+10)/2 = 10.0
    pooled = history.predicted_hold(['fast/src', 'slow/src'])
    assert pooled == pytest.approx(10.0)
    assert pooled != pytest.approx(55.0), 'median-of-medians, not a pooled median'
    assert history.sample_count(['fast/src', 'slow/src']) == 8


def test_predicted_hold_is_order_independent_across_the_modules_named():
    history = _history_of({'fast/src': [10.0] * 5, 'slow/src': [100.0] * 3})

    assert history.predicted_hold(['slow/src', 'fast/src']) == pytest.approx(
        history.predicted_hold(['fast/src', 'slow/src'])
    )


def test_predicted_hold_matches_the_canonical_traces_hand_computed_medians():
    """End-to-end against the fixture oracle: spans in, hand-computed medians out."""
    history = HoldHistory(window=10, min_samples=3)
    for span in iter_hold_spans(F.build_trace()):
        history.record_span(span)

    for module, expected in F.EXPECTED_MEDIANS.items():
        assert history.predicted_hold([module]) == pytest.approx(expected), module

    assert history.predicted_hold(list(F.EXPECTED_MEDIANS)) == pytest.approx(
        F.EXPECTED_POOLED_MEDIAN
    )
    assert history.sample_count(list(F.EXPECTED_MEDIANS)) == F.EXPECTED_SPAN_COUNT


# --- the rolling window is real --------------------------------------------


#: 15 durations: five long ones, then ten short ones.
#: - last 10 -> [10..100], median (50+60)/2 = 55.0
#: - all 15  -> median is the 8th of the sorted 15 = 80.0
_FIFTEEN = [1000.0] * 5 + [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]


def test_window_keeps_only_the_last_ten_samples():
    """N=10 must be a real eviction, not a slice taken at read time.

    55.0 (last ten) and 80.0 (all fifteen) are far apart precisely so this
    cannot pass by accident.
    """
    history = _history_of({'churn/src': _FIFTEEN}, window=10, min_samples=3)

    assert history.sample_count(['churn/src']) == 10
    assert history.predicted_hold(['churn/src']) == pytest.approx(55.0)
    assert history.predicted_hold(['churn/src']) != pytest.approx(80.0), 'window not enforced'


def test_default_window_is_also_ten():
    """The stock constructor must be bounded too — an unbounded default would
    make the predictor answer from holds that stopped being representative
    months ago (PARKING_MODEL_REPORT.md:116-126 measures the last-10 median)."""
    history = _history_of({'churn/src': _FIFTEEN})

    assert history.sample_count(['churn/src']) == 10
    assert history.predicted_hold(['churn/src']) == pytest.approx(55.0)


def test_each_module_gets_its_own_window():
    """Overflow on one module must not evict another's samples."""
    history = _history_of(
        {'churn/src': _FIFTEEN, 'quiet/src': [7.0, 8.0, 9.0]},
        window=10,
        min_samples=3,
    )

    assert history.sample_count(['churn/src']) == 10
    assert history.sample_count(['quiet/src']) == 3
    assert history.sample_count(['churn/src', 'quiet/src']) == 13
    assert history.predicted_hold(['quiet/src']) == pytest.approx(8.0)


def test_window_size_is_configurable():
    history = _history_of({'churn/src': _FIFTEEN}, window=3, min_samples=3)

    assert history.sample_count(['churn/src']) == 3
    # last three: [80, 90, 100] -> 90.0
    assert history.predicted_hold(['churn/src']) == pytest.approx(90.0)


def test_module_match_is_exact_not_prefix():
    """Modules are depth-coarsened path prefixes already (``_get_modules``).

    Matching 'orchestrator/src' against 'orchestrator/src/orchestrator' here
    would pool holds from a key the lock layer treats as different.
    """
    history = _history_of(
        {'orchestrator/src': [10.0, 20.0, 30.0], 'orchestrator/tests': [900.0] * 3},
    )

    assert history.predicted_hold(['orchestrator/src']) == pytest.approx(20.0)
    assert history.sample_count(['orchestrator/src']) == 3
    assert history.sample_count(['orchestrator']) == 0
