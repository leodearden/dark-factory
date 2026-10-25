"""A multi-module scoped red in which one sibling module's leg reached no test
verdict, shared by the merge gate's task-6247 tests.

``verify.py::_aggregate_results`` joins each module's ``test_output`` with
'\\n' and unions the failing legs' categories, so the joined red carries a
completed module's one named flake beside a sibling whose only record is its
category. Consumers: test_flake_discriminator.py (the discriminator's refusal
at the merge gate, and the main probe still re-running) and
test_verify_merge_flake_suppression.py (the merge-gate hook and ledger row for
that refusal).
"""

from _xdist_crash_fixtures import PYTEST_Q_COMPLETE_ONE_FAILED_OUTPUT

#: A sibling module's partial -q output when its leg reached no verdict:
#: progress dots, then nothing. It names no test.
SIBLING_PARTIAL_OUTPUT_NAMING_NO_TEST = '.' * 40 + '\n'


def sibling_without_verdict_session(
    sibling_category: str, *, sibling_first: bool = False,
) -> tuple[str, list[str]]:
    """A completed module with one flake beside a sibling that reached no
    verdict, as (joined test_output, failing_leg_categories) in module order."""
    modules = [
        (PYTEST_Q_COMPLETE_ONE_FAILED_OUTPUT, 'test_failure'),
        (SIBLING_PARTIAL_OUTPUT_NAMING_NO_TEST, sibling_category),
    ]
    if sibling_first:
        modules.reverse()
    return '\n'.join(output for output, _ in modules), [leg for _, leg in modules]
