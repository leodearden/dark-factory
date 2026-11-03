"""Dated, regime-scoped verify-duration census records: each figure's single home.

The tests/scripts budget guards consume these records. This is a NON-TEST
helper, so guards IMPORT a record rather than copying it or parsing a sibling
guard's source; the convention is stated in
``tests/scripts/module_budget_family.py``'s docstring. This module stays a
leaf: it imports no guard and nothing from ``module_budget_family``.

Re-measure by repeating a record's ``census_command`` and replacing the whole
record. Consumers:

  * ``tests/scripts/test_module_verify_budgets.py::RULED_BUDGET_CENSUS``
  * ``tests/scripts/test_fallback_verify_config.py::MEASURED_FLEET_SEGMENT_SECS``
  * ``scripts/verify_budget_census.py::FEEDS_CONSTANT``, which points here
"""
from __future__ import annotations

from typing import NamedTuple


class BudgetCensus(NamedTuple):
    """One module's verify-duration census, frozen with its own provenance.

    THE POINT OF THE SHAPE. The prose entry in
    ``tests/scripts/test_module_verify_budgets.py::RULED_INTERIM_BUDGET_EXCEPTIONS``
    could say "7200 is an operator estimate" and nothing could check it. These
    fields are asserted against, so that guard's ruled-exception branch stops
    asserting prose and starts asserting a DERIVATION.

    ``census_command`` and ``measured_at`` are not decoration: they are what
    make the next figure a REPEAT rather than a re-derivation, which is exactly
    what
    ``tests/scripts/test_module_verify_budgets.py::MEASURED_MODULE_SUITE_WORST_SECS``'
    own provenance block demands. Re-measure by re-running that command, and
    replace the whole record.

    ``regime``/``regime_since`` exist because a bare trailing window is WRONG
    here, and provably so. A 14-day window from 2026-09-14 reaches back to
    08-31 and straddles the retired 3-slot / 3600-ceiling regime, whose max was
    4991s -> a floor of 7500, which trips
    ``tests/scripts/test_module_verify_budgets.py::CENSUS_FLOOR_REFUSAL_CEILING``
    on evidence from a configuration that no longer exists. D17's rule says
    "under the PREVAILING config", so the window is
    ``intersection(trailing-14d, prevailing regime)`` and assertion (4) of
    ``tests/scripts/test_module_verify_budgets.py::test_module_carries_its_own_measured_verify_budget``,
    ``window_start >= regime_since``, holds it there. That assertion goes INERT
    on its own once 14 days of prevailing-regime data exist, which is the point
    — it narrows the window only while narrowing is needed, and says why.
    """

    window_start: str
    window_end: str
    regime: str
    regime_since: str
    n: int
    p50: float
    p90: float
    max_secs: float
    timed_out: int
    census_command: str
    measured_at: str


# MEASURED 2026-09-14 by the verbatim command below, over the PREVAILING
# config regime. RE-MEASURE BY REPEATING THAT COMMAND and replacing this whole
# record, so the next figure is a repeat rather than a re-derivation.
#
# THE THREE REGIMES THE CENSUS FOUND, which is why this one is scoped:
#   pre-2026-09-03          n=26  p50=1551  p90=2385  max=3225   t/o=1   floor 4900
#   09-03 .. 09-12T08       n=26  p50=3090  p90=4223  max=4991   t/o=14  floor 7500
#   since 09-12T08 (THIS)   n=14  p50=3275  p90=3685  max=4626   t/o=0   floor 7000
# The middle regime corroborates the segmentation on its own evidence: 14
# timeouts against this regime's 0, which is the distress commit 36c4c71eb4
# relieved by raising the ceiling 3600 -> 7200.
#
# DIVERGENCE FROM D17's STATED FIGURES, recorded rather than smoothed over.
# The ruling expected max=3753 -> floor 5700. Measured max is 4626 -> floor
# 7000, because a green run landed at 2026-09-14T09:35 (task 3541) after the
# ruling was written, making 3753 the second-highest. n/p50/p90 all match the
# ruling's shape. Consequence filed as esc-3353-15: headroom against the
# refusal ceiling is 174s of floor (7000 vs 7200), i.e. a single green run
# above 4800s trips assertion (3) of
# tests/scripts/test_module_verify_budgets.py::test_module_carries_its_own_measured_verify_budget.
# That is the ruling's INTENDED behaviour — the finding surfaces instead of
# the budget silently rising — so when it fires, read it as the escalate
# branch working.
#
# EVERY IN-WINDOW RUN IS UNSTAMPED. Deliverable 1 of task 3353 adds the host
# load stamp; it had not merged when this was measured, so all 14 records
# carry no load reading and the census's load bands are all empty. The figures
# here are therefore durations under UNKNOWN host load — which is precisely the
# gap D1 closes for the next re-measure.
ORCHESTRATOR_BUDGET_CENSUS = BudgetCensus(
    window_start='2026-09-12T08:00:00+00:00',
    window_end='2026-09-15T00:00:00+00:00',
    regime='prevailing config since the 3600 -> 7200 fleet-ceiling raise',
    regime_since='2026-09-12T08:00:00+00:00',
    n=14,
    p50=3274.9211449669992,
    p90=3684.5928532374005,
    max_secs=4626.166946739017,
    timed_out=0,
    census_command=(
        'uv run --project shared python scripts/verify_budget_census.py '
        '--root /home/leo/src/dark-factory --module orchestrator '
        '--label test --role task '
        '--window 2026-09-12T08:00:00+00:00..2026-09-15T00:00:00+00:00'
    ),
    measured_at='2026-09-14',
)
