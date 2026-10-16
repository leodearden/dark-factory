"""The governed exception lists of orchestrator/, declared under INV-12.

Each binding below is a declaration in the sense of
``plans/inv12-exceptions-owned-or-ratified-prd.md`` D4/D5: it names a list,
states the rule a new entry is judged against, and gives every entry a
disposition. The call is evaluated when this module is collected, so an entry
with no disposition fails this suite before any test runs.
"""

from pathlib import Path

from shared.governed_exceptions import governed_exceptions

REPO_ROOT = Path(__file__).resolve().parents[2]

MERGE_LANE_RATCHET_BASELINE = governed_exceptions(
    'orchestrator.tests.merge_lane_ratchet_baseline.json',
    'Merge-lane quality measures frozen at their seeded values: each may fall and '
    'none may rise without an authorized raise, and a path or function absent from '
    'the baseline must meet the file-size and new-function complexity ceilings.',
    ['orchestrator/tests/merge_lane_ratchet_baseline.json'],
)


def test_the_baseline_unit_names_a_file_in_the_checkout() -> None:
    (path,) = MERGE_LANE_RATCHET_BASELINE.keys
    assert (REPO_ROOT / path).is_file(), (
        f'{path} is not a file: the merge-lane ratchet baseline was moved or renamed, '
        'so this declaration\'s key and list_id must follow '
        'scripts/merge_lane_metrics.py::BASELINE_RELPATH.'
    )
