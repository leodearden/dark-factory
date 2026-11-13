"""The governed exception lists of tests/repo_guards/, declared under INV-12.

Each binding below is a declaration in the sense of
``plans/inv12-exceptions-owned-or-ratified-prd.md`` D4/D5: it names a list,
states the rule a new entry is judged against, and reads its keys from the
live container, so a container that grows an entry fails this suite at
collection until the new entry has a disposition.
"""

import governed_list_agreement
import pytest
import test_design_invariants_consistency as design_invariants_consistency
from shared.governed_exceptions import Debt, TaskRef, governed_exceptions
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

DESIGN_INVARIANT_PINNED_SITES_LIST = governed_exceptions(
    'tests.repo_guards.test_design_invariants_consistency.PINNED_SITES',
    'A file whose relationship to the design-invariant family is pinned in that '
    'module instead of being flagged as an unregistered enumeration site.',
    sorted(design_invariants_consistency.PINNED_SITES),
    default=Debt(TaskRef(7136)),
    default_covers=5,
)


def test_the_register_reads_every_declaration_here_as_it_evaluates() -> None:
    governed_list_agreement.assert_register_reads_as_evaluated(__file__, globals())
