"""The governed exception lists of scripts/, declared under INV-12.

Each binding below is a declaration in the sense of
``plans/inv12-exceptions-owned-or-ratified-prd.md`` D4/D5: it names a list,
states the rule a new entry is judged against, and reads its keys from the
live container, so a container that grows an entry fails this suite at
collection until the new entry has a disposition.
"""

import check_dashboard_unit_parity
import governed_list_agreement
import inline_suppressions
import merge_lane_metrics
import sweep_toolcall_markup
import test_design_invariants_consistency as design_invariants_consistency
import test_merge_state_vocabulary_consistency as merge_state_vocabulary_consistency
from shared.governed_exceptions import Debt, Policy, TaskRef, governed_exceptions

SIZE_CEILING_EXEMPT_LIST = governed_exceptions(
    'scripts.merge_lane_metrics.SIZE_CEILING_EXEMPT',
    'A path the merge-lane ratchet measures and freezes but exempts from '
    'FILE_LINE_CEILING, because its split into smaller modules is owed by a '
    'follow-up PRD.',
    sorted(merge_lane_metrics.SIZE_CEILING_EXEMPT),
    default=Debt(TaskRef(7140)),
    default_covers=1,
)

DIVERGENCE_ALLOWLIST_LIST = governed_exceptions(
    'scripts.check_dashboard_unit_parity.DIVERGENCE_ALLOWLIST',
    'An Environment= variable name whose value the dashboard unit-parity check '
    'does not compare between the committed and installed units, because the '
    'value is host-local by design; its presence is still compared.',
    sorted(check_dashboard_unit_parity.DIVERGENCE_ALLOWLIST),
    default=Policy('inv12-dashboard-unit-divergence'),
    default_covers=2,
)

DESIGN_INVARIANT_PINNED_SITES_LIST = governed_exceptions(
    'scripts.tests.test_design_invariants_consistency.PINNED_SITES',
    'A file whose relationship to the design-invariant family is pinned in that '
    'module instead of being flagged as an unregistered enumeration site.',
    sorted(design_invariants_consistency.PINNED_SITES),
    default=Debt(TaskRef(7136)),
    default_covers=5,
)

MERGE_STATE_PINNED_SITES_LIST = governed_exceptions(
    'scripts.tests.test_merge_state_vocabulary_consistency.PINNED_SITES',
    'A file whose restated merge-state vocabulary spans are pinned against '
    'shared/src/shared/merge_state.py instead of being flagged as an '
    'unregistered enumeration.',
    sorted(merge_state_vocabulary_consistency.PINNED_SITES),
    default=Debt(TaskRef(7136)),
    default_covers=5,
)

INLINE_SUPPRESSION_BASELINE = governed_exceptions(
    'scripts.inline_suppression_baseline.json',
    'Inline suppressions grandfathered at the seed, held as a content-keyed '
    'multiset that may only shrink; a suppression absent from it needs an inline '
    'disposition.',
    [inline_suppressions.BASELINE_PATH],
    default=Policy('inv12-shrink-only-baselines'),
    default_covers=1,
)

NEVER_TOUCH_LIST = governed_exceptions(
    'scripts.sweep_toolcall_markup.NEVER_TOUCH',
    'A file the tool-call markup sweep never rewrites, because it is a verbatim record '
    'of an earlier leak.',
    sorted(sweep_toolcall_markup.NEVER_TOUCH),
    default=Policy('inv12-toolcall-markup-never-touch'),
    default_covers=2,
)


def test_the_register_reads_every_declaration_here_as_it_evaluates() -> None:
    governed_list_agreement.assert_register_reads_as_evaluated(__file__, globals())
