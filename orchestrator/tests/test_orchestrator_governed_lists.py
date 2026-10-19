"""The governed exception lists of orchestrator/, declared under INV-12.

Each binding below is a declaration in the sense of
``plans/inv12-exceptions-owned-or-ratified-prd.md`` D4/D5: it names a list,
states the rule a new entry is judged against, and gives every entry a
disposition. The call is evaluated when this module is collected, so an entry
with no disposition fails this suite before any test runs.
"""

import sys
from pathlib import Path

from shared.governed_exceptions import Policy, governed_exceptions

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

import governed_list_agreement
import merge_lane_metrics

MERGE_LANE_RATCHET_BASELINE = governed_exceptions(
    'orchestrator.tests.merge_lane_ratchet_baseline.json',
    'Merge-lane quality measures frozen at their seeded values: each may fall and '
    'none may rise without an authorized raise, and a path or function absent from '
    'the baseline must meet the file-size and new-function complexity ceilings.',
    [merge_lane_metrics.BASELINE_RELPATH],
    default=Policy('inv12-shrink-only-baselines'),
    default_covers=1,
)


def test_the_register_reads_every_declaration_here_as_it_evaluates() -> None:
    governed_list_agreement.assert_register_reads_as_evaluated(__file__, globals())
