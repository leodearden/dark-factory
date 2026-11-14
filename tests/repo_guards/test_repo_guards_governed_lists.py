"""The governed exception lists of tests/repo_guards/, declared under INV-12.

Each binding below is a declaration in the sense of
``plans/inv12-exceptions-owned-or-ratified-prd.md`` D4/D5: it names a list,
states the rule a new entry is judged against, and reads its keys from the
live container, so a container that grows an entry fails this suite at
collection until the new entry has a disposition.
"""

import re
from collections import Counter
from collections.abc import Iterable

import config_dir_archival_allowlist
import governed_list_agreement
import loop_blocking_allowlist
import pytest
import silent_fallthrough_allowlist
import test_design_invariants_consistency as design_invariants_consistency
from shared.governed_exceptions import Debt, Policy, TaskRef, TicketRef, governed_exceptions
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)


def _occurrence_keys(sites: Iterable[tuple[str, ...]]) -> list[str]:
    """One key per site, the n-th repeat of an identical site suffixed '#<n>'."""
    seen: Counter[str] = Counter()
    keys = []
    for site in sites:
        key = '::'.join(site)
        seen[key] += 1
        keys.append(key if seen[key] == 1 else f'{key}#{seen[key]}')
    return keys


DESIGN_INVARIANT_PINNED_SITES_LIST = governed_exceptions(
    'tests.repo_guards.test_design_invariants_consistency.PINNED_SITES',
    'A file whose relationship to the design-invariant family is pinned in that '
    'module instead of being flagged as an unregistered enumeration site.',
    sorted(design_invariants_consistency.PINNED_SITES),
    default=Debt(TaskRef(7136)),
    default_covers=5,
)


SILENT_FALLTHROUGH_ALLOWLIST = governed_exceptions(
    'tests.repo_guards.silent_fallthrough_allowlist.ALLOWLIST_ENTRIES',
    'A pre-existing site that swallows an error into an empty value or a discarded '
    'error slot, tolerated by the silent-fallthrough gate until it is fixed.',
    _occurrence_keys(entry[:3] for entry in silent_fallthrough_allowlist.ALLOWLIST_ENTRIES),
    default=Debt(TaskRef(7139)),
    default_covers=12,
)

_UNARCHIVED_CONFIG_DIR_SITES = tuple(
    site
    for site in config_dir_archival_allowlist.AUDITED_SITES
    if site['disposition'] != config_dir_archival_allowlist.ARCHIVED
)

CONFIG_DIR_ARCHIVAL_EXCEPTIONS = governed_exceptions(
    'tests.repo_guards.config_dir_archival_allowlist.AUDITED_SITES',
    'A TaskConfigDir construction site whose dir is destroyed without its transcripts '
    'being archived first, either by design or as a recorded gap.',
    _occurrence_keys((site['path'], site['qualname']) for site in _UNARCHIVED_CONFIG_DIR_SITES),
    default=Policy('inv12-config-dir-unarchived-by-design'),
    default_covers=4,
)


LOOP_BLOCKING_LEDGER = governed_exceptions(
    'tests.repo_guards.loop_blocking_allowlist.AUDITED_SITES',
    'A coroutine call site in fused-memory/src that reaches a blocking primitive on the '
    'event loop thread, tolerated by the INV-8 loop-blocking gate with a recorded '
    'justification until it is offloaded.',
    _occurrence_keys(row[:3] for row in loop_blocking_allowlist.AUDITED_SITES),
    default=Debt(TaskRef(5270)),
    default_covers=28,
    dispositions={
        'fused-memory/src/fused_memory/mcp_tools/scheduler_state.py::read_scheduler_events::14947dd93734': Debt(TaskRef(7572)),
        'fused-memory/src/fused_memory/middleware/task_curator.py::TaskCurator._invoke_curator::2be781b317a0': Debt(TaskRef(7572)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._recover_one_run::f05da1451b63': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._persist_stage_reports_then_gc_config_dir::f05da1451b63': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._recover_stale_runs::0787c60051a4': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._recover_stale_runs::cc7999e2d4b6': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._record_failed_run::652fc88d57ad': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._maybe_escalate_stale_task_count_snapshot::560696057da1': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._maybe_remediate::1fbc50761be3': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._maybe_remediate::51ec3ffac666': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._maybe_remediate::8c9728223345': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._run_remediation_pass::c5a47e52ad4b': Debt(TicketRef('tkt_0RT7RJKQ0WXB0T87F8TJ7RTGQH')),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._run_remediation_pass::9e5ae6eb2503': Debt(TicketRef('tkt_0RT7RJKQ0WXB0T87F8TJ7RTGQH')),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._run_remediation_pass::edd9a8720cf9': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._run_remediation_pass::a4c939b65449': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._run_remediation_pass::ec953752e2f1': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py::_write_task_count_snapshot::e08ddda37175': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/reconciliation/stale_priority_override_edge_sweep.py::read_live_override_state::7ebd882c849b': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/reconciliation/targeted.py::TargetedReconciler._sweep_cancelled_descendants::b9609c7cf5b4': Debt(TicketRef('tkt_0RT7RKWJG17W03JRC947R8FZZN')),
        'fused-memory/src/fused_memory/server/main.py::run_server::8a884f86a26c': Debt(TaskRef(7572)),
        'fused-memory/src/fused_memory/server/main.py::run_server::f6a7ff40db1f': Debt(TaskRef(7572)),
        'fused-memory/src/fused_memory/server/main.py::run_server::43c97372205c': Debt(TaskRef(7572)),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.get_tasks::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.get_statuses::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.get_external_statuses::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.get_task::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.set_task_status::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.set_task_claimant::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.submit_task::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.resolve_ticket::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.list_tickets::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.search_tasks::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.commit_planning::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.update_task::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.remove_task::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.add_dependency::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.remove_dependency::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.set_task_priority_override::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.clear_task_priority_override::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.reorder_pin_queue::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.get_pin_queue::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.get_scheduler_state::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.get_scheduler_events::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.request_park_eviction::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.request_park_eviction::a0b1b5c60eed': Debt(TaskRef(7572)),
        'fused-memory/src/fused_memory/server/tools.py::create_mcp_server.find_tasks_by_metadata::81d0d2709341': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/services/overreach_shadow.py::OverreachShadowStore.open_read_only::fa698adde643': Debt(TaskRef(7572)),
    },
)

_OWNER_TYPE_BY_LEDGER_DISPOSITION = {'accepted': None, 'filed': TaskRef, 'to_file': TicketRef}


def _override_follows_row(
    owner_type: type[TaskRef] | type[TicketRef] | None, override: object, justification: str
) -> bool:
    if owner_type is None:
        return override is None
    if not isinstance(override, Debt) or not isinstance(override.owner, owner_type):
        return False
    return re.search(rf'\b{re.escape(str(override.owner.id))}\b', justification) is not None


def _loop_blocking_mismatch(row: tuple[str, str, str, str, str], key: str) -> str | None:
    disposition, justification = row[3], row[4]
    override = LOOP_BLOCKING_LEDGER.overrides.get(key)
    if _override_follows_row(_OWNER_TYPE_BY_LEDGER_DISPOSITION[disposition], override, justification):
        return None
    return f'{key}: ledger disposition {disposition!r}, override found {override!r}'


_OVERRIDE_TYPE_BY_ARCHIVAL_DISPOSITION = {
    config_dir_archival_allowlist.UNARCHIVED_BY_DESIGN: None,
    config_dir_archival_allowlist.UNARCHIVED_GAP: Debt,
}


def _config_dir_mismatch(disposition: str, key: str) -> str | None:
    override = CONFIG_DIR_ARCHIVAL_EXCEPTIONS.overrides.get(key)
    override_type = _OVERRIDE_TYPE_BY_ARCHIVAL_DISPOSITION[disposition]
    if override_type is None and override is None:
        return None
    if override_type is not None and isinstance(override, override_type):
        return None
    return f'{key}: allowlist disposition {disposition!r}, override found {override!r}'


def test_repeated_sites_keep_distinct_keys() -> None:
    assert _occurrence_keys([('a.py', 'f', 'h'), ('a.py', 'f', 'h'), ('b.py', 'g', 'h')]) == [
        'a.py::f::h',
        'a.py::f::h#2',
        'b.py::g::h',
    ]


def test_loop_blocking_overrides_follow_the_ledger_vocabulary() -> None:
    assert set(_OWNER_TYPE_BY_LEDGER_DISPOSITION) == loop_blocking_allowlist.DISPOSITIONS, (
        'loop_blocking_allowlist.DISPOSITIONS changed: decide which INV-12 disposition '
        'each ledger disposition maps to in _OWNER_TYPE_BY_LEDGER_DISPOSITION.'
    )
    rows = zip(loop_blocking_allowlist.AUDITED_SITES, LOOP_BLOCKING_LEDGER.keys, strict=True)
    mismatches = [mismatch for row, key in rows if (mismatch := _loop_blocking_mismatch(row, key))]
    assert not mismatches, (
        'Each loop-blocking override must match its ledger row: none for accepted, '
        'Debt(TaskRef(<task>)) for filed, Debt(TicketRef(<ticket>)) for to_file, and the '
        "owner id must appear in the row's justification. Correct the override in this "
        'module, or the ledger row in loop_blocking_allowlist.py:\n  ' + '\n  '.join(mismatches)
    )


def test_config_dir_overrides_follow_the_allowlist_vocabulary() -> None:
    unarchived = config_dir_archival_allowlist.DISPOSITIONS - {config_dir_archival_allowlist.ARCHIVED}
    assert set(_OVERRIDE_TYPE_BY_ARCHIVAL_DISPOSITION) == unarchived, (
        'config_dir_archival_allowlist.DISPOSITIONS changed: decide which INV-12 disposition '
        'each unarchived disposition maps to in _OVERRIDE_TYPE_BY_ARCHIVAL_DISPOSITION.'
    )
    assert isinstance(CONFIG_DIR_ARCHIVAL_EXCEPTIONS.default, Policy), (
        'The UNARCHIVED_BY_DESIGN sites are ratified, so CONFIG_DIR_ARCHIVAL_EXCEPTIONS '
        f'must borrow a Policy default, not {CONFIG_DIR_ARCHIVAL_EXCEPTIONS.default!r}.'
    )
    pairs = zip(_UNARCHIVED_CONFIG_DIR_SITES, CONFIG_DIR_ARCHIVAL_EXCEPTIONS.keys, strict=True)
    mismatches = [
        mismatch for site, key in pairs if (mismatch := _config_dir_mismatch(site['disposition'], key))
    ]
    assert not mismatches, (
        'Each config-dir override must match its allowlist row: none for UNARCHIVED_BY_DESIGN, '
        'which borrows the Policy default, and a Debt for UNARCHIVED_GAP. Correct the override '
        'in this module, or the row in config_dir_archival_allowlist.py:\n  '
        + '\n  '.join(mismatches)
    )


def test_the_register_reads_every_declaration_here_as_it_evaluates() -> None:
    governed_list_agreement.assert_register_reads_as_evaluated(__file__, globals())
