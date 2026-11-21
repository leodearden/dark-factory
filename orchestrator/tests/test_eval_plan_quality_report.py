"""Tests for orchestrator.evals.plan_quality_report over synthetic and produced cells.

``build_plan_quality_report`` and ``format_plan_quality_table`` read hand-built
``EvalResult``s and one cell the real architect producer writes through the hermetic
harness.
"""

from __future__ import annotations

import json

import pytest
from _architect_eval_harness import run_architect_eval_hermetic
from _eval_refusal import REFUSAL_TEXT

from orchestrator.evals.measurement import MeasurementStage, Unmeasured, UnmeasuredCause
from orchestrator.evals.metrics import RoleUnderTest
from orchestrator.evals.results import ResultStore

# ---------------------------------------------------------------------------
# plan_quality report column — additive interim surface (step-11/12)
#
# A distinct per-(task_id, config_name, role_under_test) column μ/λ consume in
# the interim — NOT a change to the Elo build_report/format_markdown schema (the
# full C4 composite+plan_quality row is owned by λ, which θ does not depend on).
# Mirrors η's TestRecoveryReport precisely: a populated float for architect runs,
# the '-' null sentinel otherwise, sorted deterministically, rendering
# byte-identically regardless of input order.
# ---------------------------------------------------------------------------

def _architect_result(
    task_id: str = 'df_task_2605',
    config_name: str = 'architect-sonnet-high',
    plan_quality: float = 0.75,
    plan_steps: int = 6,
    judged_without_reference: bool | None = False,
):
    """An architect cell that DID produce a plan.

    ``plan_steps`` defaults NONZERO (task 3302): a ``plan_quality`` is a score
    over the steps a plan actually carried, and ``plan_steps > 0`` is the
    predicate the report layer reads to know one exists. A stepless cell is the
    distinct no-plan shape, requested explicitly by the tests that exercise it.

    ``judged_without_reference`` (task 3628) mints a cell whose score came from
    a judge handed an EMPTY reference diff. THREE-VALUED on purpose, because
    the report layer is: ``True``/``False`` both WRITE the key — the production
    shape, since ``EvalMetrics.to_dict`` emits it on every cell — while
    ``None`` OMITS it, minting the pre-σ legacy shape whose validity was never
    measured. The default is an explicit ``False``: a healthy modern cell, not
    a legacy one.
    """
    from orchestrator.evals.results import EvalResult

    metrics = {
        'role_under_test': 'architect',
        'plan_quality': plan_quality,
        'plan_steps': plan_steps,
        'composite_score': 0.0,
    }
    if judged_without_reference is not None:
        metrics['judged_without_reference'] = judged_without_reference

    return EvalResult(
        task_id=task_id,
        config_name=config_name,
        outcome='done',
        metrics=metrics,
        worktree_path='/tmp/wt-arch',
    )


def _implementer_result(
    task_id: str = 'df_task_1993',
    config_name: str = 'opus-high',
):
    from orchestrator.evals.results import EvalResult

    # No plan_quality / role_under_test keys at all → the null sentinel path.
    return EvalResult(
        task_id=task_id,
        config_name=config_name,
        outcome='done',
        metrics={'composite_score': 1.0},
        worktree_path='/tmp/wt-impl',
    )


def _cap_tainted_result(
    task_id: str = 'df_task_3118',
    config_name: str = 'architect-sonnet-high',
    cause: UnmeasuredCause = UnmeasuredCause.CAP_HIT,
    reason: str = REFUSAL_TEXT,
):
    """An architect cell whose invocation was refused — NOT a measurement.

    Defaults to the campaign cap payload; pass a different *cause* (e.g.
    ``UnmeasuredCause.MODEL_NOT_FOUND``) to build a cell excluded for a
    PERMANENT rather than a transient cause.
    """
    from orchestrator.evals.results import EvalResult

    record = Unmeasured(
        MeasurementStage.ARCHITECT, cause, reason, cost_usd=0.0,
    ).to_record()
    return EvalResult(
        task_id=task_id,
        config_name=config_name,
        outcome='blocked',
        metrics={
            'role_under_test': 'architect',
            'plan_quality': None,
            'cap_tainted': True,
            'invocation_errors': [record],
            'composite_score': 0.0,
        },
        worktree_path='/tmp/wt-arch-capped',
    )


_MEAN_SECTION_HEADER = 'plan_quality by config:'


def _mean_section_line(table: str, config_name: str) -> str:
    """The per-config-mean line for *config_name*, scoped to the mean SECTION.

    Scoped deliberately: a config name also appears in the per-cell rows above,
    so a bare ``in`` search over the whole table would match the wrong line.
    """
    lines = table.splitlines()
    start = lines.index(_MEAN_SECTION_HEADER)
    return next(ln for ln in lines[start:] if ln.startswith(config_name))


class TestPlanQualityReport:
    def test_build_plan_quality_report_rows(self):
        from orchestrator.evals.plan_quality_report import build_plan_quality_report

        report = build_plan_quality_report(
            [_architect_result(), _implementer_result()]
        )
        rows = report['rows']
        by_task = {r['task_id']: r for r in rows}

        arch = by_task['df_task_2605']
        assert arch['plan_quality'] == 0.75
        assert arch['role_under_test'] == 'architect'
        assert arch['config_name'] == 'architect-sonnet-high'

        impl = by_task['df_task_1993']
        assert impl['plan_quality'] is None  # null sentinel
        assert impl['role_under_test'] is None

    def test_rows_sorted_by_task_id_then_config_name(self):
        from orchestrator.evals.plan_quality_report import build_plan_quality_report

        # Deliberately unsorted input across both task_id and config_name.
        results = [
            _architect_result(task_id='df_task_2605', config_name='architect-sonnet-high'),
            _architect_result(task_id='df_task_1000', config_name='architect-opus-high'),
            _architect_result(task_id='df_task_2605', config_name='architect-opus-high'),
        ]
        report = build_plan_quality_report(results)
        keys = [(r['task_id'], r['config_name']) for r in report['rows']]
        assert keys == sorted(keys)

    def test_format_plan_quality_table_renders_column(self):
        from orchestrator.evals.plan_quality_report import (
            build_plan_quality_report,
            format_plan_quality_table,
        )

        report = build_plan_quality_report(
            [_architect_result(), _implementer_result()]
        )
        table = format_plan_quality_table(report)
        assert 'plan_quality' in table

        arch_line = next(
            ln for ln in table.splitlines() if 'df_task_2605' in ln
        )
        assert '0.7500' in arch_line     # populated float for the architect row
        assert 'architect' in arch_line  # role_under_test column

        impl_line = next(
            ln for ln in table.splitlines() if 'df_task_1993' in ln
        )
        assert '-' in impl_line           # null sentinel rendered as '-'
        assert '0.7500' not in impl_line  # implementer row is NOT populated

    # -- cap-tainted exclusion (task 3118) --------------------------------
    # A cap-tainted cell is an INFRA failure, not a content measurement, so the
    # aggregate must EXCLUDE it and COUNT the exclusion — never average in a
    # fabricated zero, which would penalise whichever candidate happened to be
    # scheduled inside a cap window.

    def test_rows_carry_cap_tainted_and_invocation_errors(self):
        from orchestrator.evals.plan_quality_report import build_plan_quality_report

        report = build_plan_quality_report(
            [_architect_result(), _cap_tainted_result(), _implementer_result()]
        )
        by_task = {r['task_id']: r for r in report['rows']}

        capped = by_task['df_task_3118']
        assert capped['cap_tainted'] is True
        assert capped['invocation_errors'] == [
            {'stage': 'architect', 'cause': 'cap_hit', 'reason': REFUSAL_TEXT},
        ]
        assert 'invocation_error' not in capped
        assert capped['plan_quality'] is None

        # Legacy results, whose metrics predate both keys, read back default-safe.
        assert by_task['df_task_2605']['cap_tainted'] is False
        assert by_task['df_task_2605']['invocation_errors'] is None
        assert by_task['df_task_1993']['cap_tainted'] is False

    def test_per_config_mean_excludes_cap_tainted_cells(self):
        from orchestrator.evals.plan_quality_report import build_plan_quality_report

        cfg = 'architect-opus-high'
        report = build_plan_quality_report([
            _architect_result(task_id='t1', config_name=cfg, plan_quality=0.9),
            _architect_result(task_id='t2', config_name=cfg, plan_quality=0.7),
            _cap_tainted_result(task_id='t3', config_name=cfg),
        ])
        configs = {c['config_name']: c for c in report['configs']}
        agg = configs[cfg]

        # (0.9 + 0.7) / 2 — NOT (0.9 + 0.7 + 0.0) / 3 == 0.5333.
        assert agg['mean_plan_quality'] == 0.8
        assert agg['n'] == 2
        assert agg['cap_excluded'] == 1
        assert agg['total'] == 3

    def test_report_level_cap_excluded_total(self):
        from orchestrator.evals.plan_quality_report import build_plan_quality_report

        report = build_plan_quality_report([
            _architect_result(task_id='t1', config_name='a', plan_quality=0.9),
            _cap_tainted_result(task_id='t2', config_name='a'),
            _cap_tainted_result(task_id='t3', config_name='b'),
        ])
        assert report['cap_excluded'] == 2

    # -- judged_without_reference (task 3628) --------------------------------
    #
    # The θ-surface twin of cap_excluded, with one decisive difference: it
    # BOUNDS validity rather than excluding. The flagged cell stays in `n` and
    # in `mean_plan_quality` at its real score.

    def test_rows_distinguish_measured_false_from_an_absent_marker(self):
        """THREE-VALUED: True, False, and None-for-never-measured.

        The row must not collapse absence into False (reviewer:
        architecture-coherence). A metrics dict written before σ landed never
        had its validity measured, and reading it back as False asserts the
        reassuring answer to a question the instrument never asked — the exact
        inversion of the rule the sibling consumer
        ``scripts/run_fable_trial_v2_campaign.py`` enforces per cell.
        """
        from orchestrator.evals.plan_quality_report import build_plan_quality_report

        report = build_plan_quality_report([
            _architect_result(task_id='t1', judged_without_reference=True),
            _architect_result(task_id='t2', judged_without_reference=False),
            _architect_result(task_id='t3', judged_without_reference=None),
        ])
        by_task = {r['task_id']: r for r in report['rows']}

        assert by_task['t1']['judged_without_reference'] is True
        # MEASURED clean — the instrument ran and found a reference diff.
        assert by_task['t2']['judged_without_reference'] is False
        # NEVER MEASURED — a metrics dict predating the field.
        assert by_task['t3']['judged_without_reference'] is None

    def test_per_config_count_bounds_but_does_not_exclude(self):
        """The flagged cells are COUNTED yet still averaged in.

        The single most likely misimplementation is to treat this like
        cap_excluded and drop the cell from the pool. It is not an exclusion:
        the score is real, only the confidence in it is reduced. Pinned by
        asserting the mean is over all THREE cells, not just the healthy one.
        """
        from orchestrator.evals.plan_quality_report import build_plan_quality_report

        cfg = 'architect-opus-high'
        report = build_plan_quality_report([
            _architect_result(
                task_id='t1', config_name=cfg, plan_quality=0.9,
                judged_without_reference=True,
            ),
            _architect_result(
                task_id='t2', config_name=cfg, plan_quality=0.6,
                judged_without_reference=True,
            ),
            _architect_result(task_id='t3', config_name=cfg, plan_quality=0.3),
        ])
        agg = {c['config_name']: c for c in report['configs']}[cfg]

        assert agg['judged_without_reference'] == 2
        assert agg['n'] == 3
        assert agg['mean_plan_quality'] == 0.6  # (0.9 + 0.6 + 0.3) / 3

    def test_cap_tainted_cell_is_not_counted_as_judged_blind(self):
        """Disjoint from cap_excluded: the θ count is scoped to the ADMITTED pool.

        A cap-tainted cell has no plan_quality to bound, so even carrying the
        flag it must land in cap_excluded ALONE — exactly how `no_plan` is
        scoped on this surface.
        """
        from orchestrator.evals.plan_quality_report import build_plan_quality_report

        cfg = 'architect-opus-high'
        capped = _cap_tainted_result(task_id='t2', config_name=cfg)
        capped.metrics['judged_without_reference'] = True

        report = build_plan_quality_report([
            _architect_result(task_id='t1', config_name=cfg, plan_quality=0.9),
            capped,
        ])
        agg = {c['config_name']: c for c in report['configs']}[cfg]

        assert agg['cap_excluded'] == 1
        assert agg['judged_without_reference'] == 0

    def test_report_level_judged_without_reference_total(self):
        from orchestrator.evals.plan_quality_report import build_plan_quality_report

        report = build_plan_quality_report([
            _architect_result(
                task_id='t1', config_name='a', judged_without_reference=True,
            ),
            _architect_result(
                task_id='t2', config_name='b', judged_without_reference=True,
            ),
            _architect_result(task_id='t3', config_name='b'),
        ])
        assert report['judged_without_reference'] == 2
        assert sum(
            c['judged_without_reference'] for c in report['configs']
        ) == report['judged_without_reference']

    def test_one_keyless_admitted_cell_makes_the_count_unmeasured(self):
        """ABSENCE POISONS THE COUNT: None, never a partial sum reading as whole.

        A config whose pool mixes measured and pre-σ cells cannot report how
        many of its scores were judged blind — it can only report that it does
        not know. Reporting the measured ones' count (here 1) would understate
        the bound while wearing the appearance of a complete answer. This is
        ``cell_admission.judged_blind_count``'s rule, which task 3632's campaign
        script consumes rather than re-deriving, so the two surfaces of one
        field answer the same question identically.
        """
        from orchestrator.evals.plan_quality_report import build_plan_quality_report

        cfg = 'architect-opus-high'
        report = build_plan_quality_report([
            _architect_result(
                task_id='t1', config_name=cfg, judged_without_reference=True,
            ),
            _architect_result(
                task_id='t2', config_name=cfg, judged_without_reference=None,
            ),
        ])
        agg = {c['config_name']: c for c in report['configs']}[cfg]

        assert agg['judged_without_reference'] is None
        # …with the size of the unmeasured population beside it: 1-of-2 and
        # 2-of-2 are different findings and the None alone cannot say which.
        assert agg['judged_without_reference_unmeasured'] == 1
        # The cells are still in the pool at their real scores — an unmeasured
        # BOUND excludes nothing, exactly as a measured one does not.
        assert agg['n'] == 2
        # …and the campaign-wide total is unknown for the same reason.
        assert report['judged_without_reference'] is None
        assert report['judged_without_reference_unmeasured'] == 1

    def test_a_fully_measured_campaign_still_reports_a_number(self):
        """The None is not contagious to a corpus the instrument DID measure.

        The negative twin of the test above: absence poisons the count, but a
        campaign whose every admitted cell carries the key reports a real
        count — including a real ZERO, which is the one case where 0 is the
        honest answer.
        """
        from orchestrator.evals.plan_quality_report import build_plan_quality_report

        cfg = 'architect-opus-high'
        report = build_plan_quality_report([
            _architect_result(
                task_id='t1', config_name=cfg, judged_without_reference=False,
            ),
            _architect_result(
                task_id='t2', config_name=cfg, judged_without_reference=False,
            ),
        ])
        agg = {c['config_name']: c for c in report['configs']}[cfg]

        assert agg['judged_without_reference'] == 0
        assert agg['judged_without_reference_unmeasured'] == 0
        assert report['judged_without_reference'] == 0

    def test_a_keyless_cap_tainted_cell_does_not_poison_the_count(self):
        """Scoped to the ADMITTED pool, absence included.

        A cap-tainted cell has no plan_quality to bound whether or not it
        carries the marker, so its keylessness must not make the config's bound
        unknown — otherwise every campaign with one cap-window cell would read
        UNMEASURED forever.
        """
        from orchestrator.evals.plan_quality_report import build_plan_quality_report

        cfg = 'architect-opus-high'
        capped = _cap_tainted_result(task_id='t2', config_name=cfg)
        capped.metrics.pop('judged_without_reference', None)

        report = build_plan_quality_report([
            _architect_result(
                task_id='t1', config_name=cfg, judged_without_reference=True,
            ),
            capped,
        ])
        agg = {c['config_name']: c for c in report['configs']}[cfg]

        assert agg['cap_excluded'] == 1
        assert agg['judged_without_reference'] == 1
        assert agg['judged_without_reference_unmeasured'] == 0

    def test_non_architect_rows_are_not_counted(self):
        """Architect-scoped, like every other count on this surface."""
        from orchestrator.evals.plan_quality_report import build_plan_quality_report

        impl = _implementer_result(task_id='t2', config_name='opus-high')
        impl.metrics['judged_without_reference'] = True

        report = build_plan_quality_report([
            _architect_result(task_id='t1', config_name='a'), impl,
        ])
        assert report['judged_without_reference'] == 0

    def test_exclusions_are_broken_out_by_cause(self):
        # The causes are NOT interchangeable: a cap hit is transient and
        # schedule-attributable ('rerun after the window'), a model-not-found is
        # a PERMANENT candidate-config error ('this can never run'). A single
        # total would let the latter masquerade as the former and hide a dead
        # config behind n=0 / mean=None (reviewer: design-coherence).
        from orchestrator.evals.plan_quality_report import build_plan_quality_report

        report = build_plan_quality_report([
            _cap_tainted_result(task_id='t1', config_name='a'),
            _cap_tainted_result(task_id='t2', config_name='a'),
            _cap_tainted_result(
                task_id='t3', config_name='b',
                cause=UnmeasuredCause.MODEL_NOT_FOUND, reason='no such model',
            ),
            # An end-to-end pin (reviewer: test-coverage) that a server_error
            # record survives the FULL report pipeline and lands in its own
            # cap_excluded_by_cause bucket.
            _cap_tainted_result(
                task_id='t4', config_name='b',
                cause=UnmeasuredCause.SERVER_ERROR, reason='HTTP 529',
            ),
        ])
        assert report['cap_excluded'] == 4
        assert report['cap_excluded_by_cause'] == {
            'cap_hit': 2, 'model_not_found': 1, 'server_error': 1,
        }
        # Key-sorted, so the dict renders byte-deterministically.
        causes = list(report['cap_excluded_by_cause'])
        assert causes == sorted(causes)
        # Every bucket is a typed cause — no free-text or 'unknown' key.
        assert set(report['cap_excluded_by_cause']) <= {
            c.value for c in UnmeasuredCause
        }

    def test_tainted_row_without_a_record_raises_naming_the_cell(self):
        # A tainted cell always carries a record: the producer writes one and
        # the loader supplies one for legacy cells. A row without one bypassed
        # ResultStore.load_all, and bucketing it would invent a cause.
        from orchestrator.evals.plan_quality_report import build_plan_quality_report

        capped = _cap_tainted_result(task_id='t1', config_name='a')
        capped.metrics['invocation_errors'] = None

        with pytest.raises(ValueError) as excinfo:
            build_plan_quality_report([capped])
        message = str(excinfo.value)
        assert 't1 x a' in message
        assert 'invocation_errors' in message

    def test_tainted_row_with_an_unknown_cause_raises_naming_the_cell(self):
        # A hand-edited or newer-producer record must not abort the report
        # with a bare enum ValueError that leaves the bad file unnamed.
        from orchestrator.evals.plan_quality_report import build_plan_quality_report

        capped = _cap_tainted_result(task_id='t1', config_name='a')
        capped.metrics['invocation_errors'] = [
            {'stage': 'architect', 'cause': 'not_a_cause', 'reason': 'r'},
        ]

        with pytest.raises(ValueError) as excinfo:
            build_plan_quality_report([capped])
        message = str(excinfo.value)
        assert 't1 x a' in message
        assert 'not_a_cause' in message

    def test_all_cells_tainted_yields_null_mean_never_zero(self):
        from orchestrator.evals.plan_quality_report import build_plan_quality_report

        report = build_plan_quality_report([
            _cap_tainted_result(task_id='t1', config_name='doomed'),
            _cap_tainted_result(task_id='t2', config_name='doomed'),
        ])
        agg = {c['config_name']: c for c in report['configs']}['doomed']
        assert agg['mean_plan_quality'] is None
        assert agg['mean_plan_quality'] != 0.0
        assert agg['n'] == 0
        assert agg['cap_excluded'] == 2

    def test_implementer_rows_stay_out_of_the_architect_aggregate(self):
        from orchestrator.evals.plan_quality_report import build_plan_quality_report

        report = build_plan_quality_report(
            [_architect_result(), _implementer_result()]
        )
        assert [c['config_name'] for c in report['configs']] == [
            'architect-sonnet-high',
        ]
        # ...while both still appear as ROWS, sorted deterministically.
        keys = [(r['task_id'], r['config_name']) for r in report['rows']]
        assert keys == sorted(keys)
        assert len(keys) == 2

    def test_configs_sorted_by_config_name(self):
        from orchestrator.evals.plan_quality_report import build_plan_quality_report

        report = build_plan_quality_report([
            _architect_result(task_id='t1', config_name='z-cfg'),
            _architect_result(task_id='t2', config_name='a-cfg'),
        ])
        names = [c['config_name'] for c in report['configs']]
        assert names == sorted(names)

    def test_table_renders_byte_identically_regardless_of_input_order(self):
        from orchestrator.evals.plan_quality_report import (
            build_plan_quality_report,
            format_plan_quality_table,
        )

        results = [_architect_result(), _implementer_result()]
        a = format_plan_quality_table(build_plan_quality_report(results))
        b = format_plan_quality_table(
            build_plan_quality_report(list(reversed(results)))
        )
        assert a == b  # deterministic (sorted rows, no wall-clock dependence)

    # -- rendered exclusion surface (task 3118) ---------------------------

    def test_cap_tainted_cell_renders_excluded_marker_not_a_zero(self):
        from orchestrator.evals.plan_quality_report import (
            build_plan_quality_report,
            format_plan_quality_table,
        )

        table = format_plan_quality_table(build_plan_quality_report(
            [_architect_result(), _cap_tainted_result(), _implementer_result()]
        ))
        capped_line = next(
            ln for ln in table.splitlines() if 'df_task_3118' in ln
        )
        # Explicit exclusion marker, WITH its reason — never a score...
        assert 'excluded' in capped_line
        assert '0.0000' not in capped_line
        assert f'architect:cap_hit: {REFUSAL_TEXT}' in capped_line
        # ...while a healthy architect row's invocation_error column is '-'.
        healthy_line = next(
            ln for ln in table.splitlines() if 'df_task_2605' in ln
        )
        assert healthy_line.split()[-1] == '-'
        # ...and visibly distinct from the '-' null sentinel a non-architect row
        # uses, so "not an architect run" cannot be confused with "architect run
        # we could not measure".
        impl_line = next(ln for ln in table.splitlines() if 'df_task_1993' in ln)
        assert 'excluded' not in impl_line

    def test_table_renders_exclusion_summary_and_per_config_means(self):
        from orchestrator.evals.plan_quality_report import (
            build_plan_quality_report,
            format_plan_quality_table,
        )

        cfg = 'architect-opus-high'
        table = format_plan_quality_table(build_plan_quality_report([
            _architect_result(task_id='t1', config_name=cfg, plan_quality=0.9),
            _architect_result(task_id='t2', config_name=cfg, plan_quality=0.7),
            _cap_tainted_result(task_id='t3', config_name=cfg),
        ]))

        # The reader sees "n=2 of 3, 1 excluded", not a silently shrunk mean.
        summary = next(
            ln for ln in table.splitlines() if ln.startswith('excluded:')
        )
        assert '1' in summary and '3' in summary
        # ...with the CAUSE named, not a bare count.
        assert 'cap_hit: 1' in summary

        mean_line = _mean_section_line(table, cfg)
        assert '0.8000' in mean_line   # the EXCLUDING mean, not 0.5333
        assert '0.5333' not in table

    def test_summary_names_a_permanent_cause_as_itself(self):
        # A model-not-found exclusion must not render as a cap window: the
        # operator's next action differs entirely ('fix the config' vs 'rerun').
        from orchestrator.evals.plan_quality_report import (
            build_plan_quality_report,
            format_plan_quality_table,
        )

        table = format_plan_quality_table(build_plan_quality_report([
            _cap_tainted_result(
                task_id='t1', config_name='dead-cfg',
                cause=UnmeasuredCause.MODEL_NOT_FOUND, reason='no such model',
            ),
        ]))
        summary = next(
            ln for ln in table.splitlines() if ln.startswith('excluded:')
        )
        assert 'model_not_found: 1' in summary
        assert 'cap' not in summary.lower()

    def test_config_with_no_scored_cells_renders_dash_not_zero(self):
        from orchestrator.evals.plan_quality_report import (
            build_plan_quality_report,
            format_plan_quality_table,
        )

        table = format_plan_quality_table(build_plan_quality_report([
            _cap_tainted_result(task_id='t1', config_name='doomed'),
            _cap_tainted_result(task_id='t2', config_name='doomed'),
        ]))
        mean_line = _mean_section_line(table, 'doomed')
        assert mean_line.endswith('-')
        assert '0.0000' not in mean_line

    def test_exclusion_table_is_byte_identical_regardless_of_input_order(self):
        from orchestrator.evals.plan_quality_report import (
            build_plan_quality_report,
            format_plan_quality_table,
        )

        results = [
            _architect_result(task_id='t1', config_name='a', plan_quality=0.9),
            _cap_tainted_result(task_id='t2', config_name='a'),
            _architect_result(task_id='t3', config_name='b', plan_quality=0.5),
            _implementer_result(),
        ]
        a = format_plan_quality_table(build_plan_quality_report(results))
        b = format_plan_quality_table(
            build_plan_quality_report(list(reversed(results)))
        )
        assert a == b


class TestLegacyAndProducedUnmeasuredCellsReportAlike:
    """B3 (PRD §6): one reader for a pre-6734 cell and a freshly produced one.

    A cell persisted with the legacy ``invocation_error`` string and a cell the
    real producer writes today both load through ``ResultStore.load_all`` and
    land in the same typed cause bucket, rendered by the same formatter.
    """

    @pytest.mark.asyncio
    async def test_both_cells_bucket_and_render_as_cap_exhausted(self, tmp_path):
        from shared.cli_invoke import AllAccountsCappedException

        from orchestrator.evals.configs import EvalConfig
        from orchestrator.evals.plan_quality_report import (
            build_plan_quality_report,
            format_plan_quality_table,
        )

        store = ResultStore(tmp_path)
        legacy = {
            'task_id': 'legacy_task',
            'config_name': 'architect-sonnet-high',
            'outcome': 'blocked',
            'metrics': {
                'role_under_test': 'architect',
                'plan_quality': None,
                'cap_tainted': True,
                'invocation_error': (
                    'architect:cap_exhausted: all accounts capped after 3 retries'
                ),
            },
            'worktree_path': '/tmp/wt-legacy',
            'wall_clock_ms': 0,
            'run_id': 'legacy01',
            'trial': 1,
        }
        (tmp_path / 'legacy_task__architect-sonnet-high__legacy01.json').write_text(
            json.dumps(legacy)
        )
        cfg = EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role=RoleUnderTest.ARCHITECT,
        )
        await run_architect_eval_hermetic(
            cfg,
            produced_plan={},
            invoke_side_effect=AllAccountsCappedException(3, 10.0, 'x'),
            store=store,
        )

        cells = store.load_all()
        assert len(cells) == 2
        report = build_plan_quality_report(cells)
        assert report['cap_excluded_by_cause'] == {'cap_exhausted': 2}
        table = format_plan_quality_table(report)
        rendered = [
            ln for ln in table.splitlines() if 'architect:cap_exhausted: ' in ln
        ]
        assert len(rendered) == 2
