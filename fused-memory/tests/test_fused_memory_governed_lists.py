"""The governed exception lists of fused-memory/, declared under INV-12.

Each binding below is a declaration in the sense of
``plans/inv12-exceptions-owned-or-ratified-prd.md`` D4/D5: it names a list,
states the rule a new entry is judged against, and gives every entry a
disposition. The call is evaluated when this module is collected, so an entry
with no disposition fails this suite before any test runs.
"""

from __future__ import annotations

from pathlib import Path

import test_timeout_marker_inversion_guard as timeout_marker_inversion_guard
from _fm_helpers import load_script_module
from shared.governed_exceptions import Debt, Policy, TaskRef, governed_exceptions

from fused_memory.memory_metadata import BLESSED_METADATA_KEYS
from fused_memory.middleware import (
    gitignored_deliverable_guard,
    operational_suggestion_guard,
    recon_claim_verification_guard,
    routing_intent_guard,
)
from fused_memory.middleware.cancelled_premise_blocklist import load_blocklist
from fused_memory.middleware.recon_code_fix_premise_guard import load_premise_registry
from fused_memory.reconciliation import flag_dedup, internal_writers, mem0_tombstone

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]

_checker = load_script_module(
    _PACKAGE_ROOT / 'scripts' / 'check_bare_magicmock_config.py',
    mod_name='check_bare_magicmock_config',
)

BLESSED_MEMORY_METADATA_KEYS = governed_exceptions(
    'fused_memory.memory_metadata.BLESSED_METADATA_KEYS',
    'A high-volume conventional memory-metadata key outside the reserved vocabulary that '
    'the metadata census accepts without an unknown-key warning.',
    sorted(BLESSED_METADATA_KEYS),
    default=Policy('inv12-memory-metadata-blessed-keys'),
    default_covers=8,
)

WALL_CLOCK_DEADLINE_BASELINE = governed_exceptions(
    'fused_memory_scripts.check_bare_magicmock_config._WALL_CLOCK_DEADLINE_DEBT',
    'Per-file budgets of wall-clock-deadline (Rule C) violations grandfathered when the '
    'rule was adopted; each budget may only fall.',
    ['fused-memory/scripts/check_bare_magicmock_config.py::_WALL_CLOCK_DEADLINE_DEBT'],
    default=Debt(TaskRef(6201)),
    default_covers=1,
)

FUSED_MEMORY_TIMEOUT_MARKER_GRANDFATHERED = governed_exceptions(
    'fused_memory_tests.test_timeout_marker_inversion_guard._GRANDFATHERED',
    'An in-band timeout-marker site in fused-memory/tests that predates the timeout-marker '
    'inversion guard, exempt from it until migrated; entries may only be removed.',
    sorted('::'.join(site) for site in timeout_marker_inversion_guard._GRANDFATHERED),
    default=Debt(TaskRef(6337)),
    default_covers=20,
)

CANCELLED_PREMISE_BLOCKLIST = governed_exceptions(
    'fused_memory_config.cancelled_premise_blocklist.yaml',
    'A premise that was tried, shipped and reverted, whose re-proposal the task curator '
    'drops before any LLM call.',
    [entry.name for entry in load_blocklist(_PACKAGE_ROOT / 'config' / 'cancelled_premise_blocklist.yaml')],
    default=Policy('inv12-cancelled-premise-blocklist'),
    default_covers=2,
)

RECON_CODE_FIX_PREMISE_REGISTRY = governed_exceptions(
    'fused_memory_config.recon_code_fix_premise_registry.yaml',
    'A recon code-fix premise an architect refuted against live source, which the task '
    'curator drops only while its source assertions still hold.',
    [
        entry.name
        for entry in load_premise_registry(_PACKAGE_ROOT / 'config' / 'recon_code_fix_premise_registry.yaml')
    ],
    default=Policy('inv12-recon-code-fix-premise-registry'),
    default_covers=5,
)

FLAG_DEDUP_SNAPSHOT_CATEGORIES = governed_exceptions(
    'fused_memory.reconciliation.flag_dedup._SUPPRESSED_SNAPSHOT_CATEGORIES',
    'A Stage 3 finding category in which a flag matching the blocked task-count-snapshot '
    'signature is dropped.',
    sorted(flag_dedup._SUPPRESSED_SNAPSHOT_CATEGORIES),
    default=Policy('inv12-flag-dedup-snapshot-categories'),
    default_covers=2,
)

FLAG_DEDUP_CEILING_CATEGORIES = governed_exceptions(
    'fused_memory.reconciliation.flag_dedup._SUPPRESSED_CEILING_CATEGORIES',
    'A Stage 3 finding category in which a flag matching the retired contamination-ceiling '
    'signature is dropped.',
    sorted(flag_dedup._SUPPRESSED_CEILING_CATEGORIES),
    default=Policy('inv12-flag-dedup-ceiling-categories'),
    default_covers=2,
)

INTERNAL_WRITER_PREFIXES = governed_exceptions(
    'fused_memory.reconciliation.internal_writers.INTERNAL_WRITER_AGENT_ID_PREFIXES',
    'An agent_id prefix that identifies a memory write as house-operated, so a '
    'possible-injection or fabricated-content flag citing that write is dropped.',
    sorted(internal_writers.INTERNAL_WRITER_AGENT_ID_PREFIXES),
    default=Policy('inv12-internal-writer-agent-id-prefixes'),
    default_covers=5,
)

MEM0_TOMBSTONE_PROTECTED_AUDIT_KINDS = governed_exceptions(
    'fused_memory.reconciliation.mem0_tombstone.PROTECTED_AUDIT_KINDS',
    'An audit-record kind the Mem0 tombstone sweep never deletes, even after its cited '
    'task goes terminal.',
    sorted(mem0_tombstone.PROTECTED_AUDIT_KINDS),
    default=Policy('inv12-mem0-tombstone-protected-audit-kinds'),
    default_covers=1,
)

OPERATIONAL_SUGGESTION_WEAK_MARKERS = governed_exceptions(
    'fused_memory.middleware.operational_suggestion_guard._WEAK_MARKER_LABELS',
    'A weak operational-suggestion marker whose match is suppressed when the same field '
    'names a code-level artifact.',
    sorted(operational_suggestion_guard._WEAK_MARKER_LABELS),
    default=Policy('inv12-operational-suggestion-weak-markers'),
    default_covers=3,
)

CLAIM_VERIFICATION_EXCLUDED_PATHSPECS = governed_exceptions(
    'fused_memory.middleware.recon_claim_verification_guard._EXCLUDED_PATHSPECS',
    'A pathspec excluded from the recon claim-verification presence probe, so an example '
    'token there never verifies a fabricated claim.',
    sorted(recon_claim_verification_guard._EXCLUDED_PATHSPECS),
    default=Policy('inv12-claim-verification-excluded-pathspecs'),
    default_covers=3,
)

ROUTING_INTENT_EXEMPT_CLASSES = governed_exceptions(
    'fused_memory.middleware.routing_intent_guard._EXEMPT_EXECUTION_CLASSES',
    'An execution class whose submissions the routing-intent guard does not check; the '
    'set is derived as every execution class except code_tdd, so a new class joins it.',
    sorted(routing_intent_guard._EXEMPT_EXECUTION_CLASSES),
    default=Policy('inv12-routing-intent-exempt-classes'),
    default_covers=2,
)

OPERATIONAL_SUGGESTION_EXEMPT_CLASSES = governed_exceptions(
    'fused_memory.middleware.operational_suggestion_guard._EXEMPT_EXECUTION_CLASSES',
    'An execution class whose submissions the operational-suggestion guard does not check; '
    'the set is derived as every execution class except code_tdd, so a new class joins it.',
    sorted(operational_suggestion_guard._EXEMPT_EXECUTION_CLASSES),
    default=Policy('inv12-operational-suggestion-exempt-classes'),
    default_covers=2,
)

GITIGNORED_DELIVERABLE_EXEMPT_CLASSES = governed_exceptions(
    'fused_memory.middleware.gitignored_deliverable_guard._EXEMPT_EXECUTION_CLASSES',
    'An execution class whose submissions the gitignored-deliverable guard does not check; '
    'the set is derived as every execution class except code_tdd, so a new class joins it.',
    sorted(gitignored_deliverable_guard._EXEMPT_EXECUTION_CLASSES),
    default=Policy('inv12-gitignored-deliverable-exempt-classes'),
    default_covers=2,
)


def test_the_wall_clock_baseline_unit_still_has_a_baseline() -> None:
    assert getattr(_checker, '_WALL_CLOCK_DEADLINE_DEBT', None), (
        'check_bare_magicmock_config._WALL_CLOCK_DEADLINE_DEBT has been burned down or '
        'deleted, so WALL_CLOCK_DEADLINE_BASELINE (its key and list_id) must be deleted '
        'with it.'
    )
