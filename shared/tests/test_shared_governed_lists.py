"""The governed exception lists of shared/, declared under INV-12.

Each binding below is a declaration in the sense of
``plans/inv12-exceptions-owned-or-ratified-prd.md`` D4/D5: it names a list,
states the rule a new entry is judged against, and gives every entry a
disposition. The call is evaluated when this module is collected, so an entry
with no disposition fails this suite before any test runs.
"""

from __future__ import annotations

import re
import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

import config_dir_archival_allowlist
import loop_blocking_allowlist
import silent_fallthrough_allowlist

from shared.governed_exceptions import Debt, TaskRef, TicketRef, governed_exceptions
from shared.task_metadata import _BLESSED_METADATA_KEYS, _LEGACY_RETRY_LEDGER_COUNTER_KEYS

REPO_ROOT = Path(__file__).resolve().parents[2]


def _occurrence_keys(sites: Iterable[tuple[str, ...]]) -> list[str]:
    """One key per site, the n-th repeat of an identical site suffixed '#<n>'."""
    seen: Counter[str] = Counter()
    keys = []
    for site in sites:
        key = '::'.join(site)
        seen[key] += 1
        keys.append(key if seen[key] == 1 else f'{key}#{seen[key]}')
    return keys


_RUFF_TABLES = (('tool', 'ruff', 'lint'), ('tool', 'ruff'))
_RUFF_GLOBAL_IGNORES = ('ignore', 'extend-ignore')
_RUFF_PER_FILE_IGNORES = ('per-file-ignores', 'extend-per-file-ignores')


def _table(config: Mapping, path: tuple[str, ...]) -> Mapping:
    for part in path:
        config = config.get(part, {})
    return config


def _workspace_pyprojects(root: Path) -> list[Path]:
    root_pyproject = root / 'pyproject.toml'
    workspace = _table(tomllib.loads(root_pyproject.read_text()), ('tool', 'uv', 'workspace'))
    return [root_pyproject, *(root / member / 'pyproject.toml' for member in workspace.get('members', []))]


def _global_ignore_keys(table: tuple[str, ...], section: Mapping) -> set[str]:
    dotted = '.'.join(table)
    return {
        f'{dotted}.{setting}::{code}'
        for setting in _RUFF_GLOBAL_IGNORES
        for code in section.get(setting, [])
    }


def _per_file_ignore_keys(relpath: str, table: tuple[str, ...], section: Mapping) -> set[str]:
    dotted = '.'.join(table)
    return {
        f'{relpath}::{dotted}.{setting}::{glob}::{code}'
        for setting in _RUFF_PER_FILE_IGNORES
        for glob, codes in section.get(setting, {}).items()
        for code in codes
    }


def ruff_suppression_keys(root: Path) -> list[str]:
    """Every ruff rule a workspace pyproject.toml turns off, as sorted declaration keys."""
    keys: set[str] = set()
    for pyproject in _workspace_pyprojects(root):
        config = tomllib.loads(pyproject.read_text())
        relpath = pyproject.relative_to(root).as_posix()
        for table in _RUFF_TABLES:
            section = _table(config, table)
            keys |= _global_ignore_keys(table, section)
            keys |= _per_file_ignore_keys(relpath, table, section)
    return sorted(keys)


SILENT_FALLTHROUGH_ALLOWLIST = governed_exceptions(
    'shared.tests.silent_fallthrough_allowlist.ALLOWLIST_ENTRIES',
    'A pre-existing site that swallows an error into an empty value or a discarded '
    'error slot, tolerated by the silent-fallthrough gate until it is fixed.',
    _occurrence_keys(entry[:3] for entry in silent_fallthrough_allowlist.ALLOWLIST_ENTRIES),
    default=Debt(TaskRef(5603)),
    default_covers=12,
)

CONFIG_DIR_ARCHIVAL_EXCEPTIONS = governed_exceptions(
    'shared.tests.config_dir_archival_allowlist.AUDITED_SITES',
    'A TaskConfigDir construction site whose dir is destroyed without its transcripts '
    'being archived first, either by design or as a recorded gap.',
    _occurrence_keys(
        (site['path'], site['qualname'])
        for site in config_dir_archival_allowlist.AUDITED_SITES
        if site['disposition'] != config_dir_archival_allowlist.ARCHIVED
    ),
    default=Debt(TaskRef(5603)),
    default_covers=6,
)

BLESSED_TASK_METADATA_KEYS = governed_exceptions(
    'shared.task_metadata._BLESSED_METADATA_KEYS',
    'A task-metadata key outside the typed TaskMetadata schema that the metadata census '
    'accepts without an unknown-key warning.',
    sorted(_BLESSED_METADATA_KEYS),
    default=Debt(TaskRef(5603)),
    default_covers=52,
)

LEGACY_RETRY_LEDGER_COUNTER_KEYS = governed_exceptions(
    'shared.task_metadata._LEGACY_RETRY_LEDGER_COUNTER_KEYS',
    'A RetryLedger counter that legacy metadata blobs still carry as a top-level key '
    'instead of nested under retry_ledger, tolerated rather than reported.',
    list(_LEGACY_RETRY_LEDGER_COUNTER_KEYS),
    default=Debt(TaskRef(5603)),
    default_covers=2,
)

RUFF_CONFIG_SUPPRESSIONS = governed_exceptions(
    'workspace.pyproject_toml.ruff_ignores',
    'A ruff rule turned off in a workspace pyproject.toml, globally or for a file glob, '
    'instead of being fixed at its sites or disposed inline.',
    ruff_suppression_keys(REPO_ROOT),
    default=Debt(TaskRef(5603)),
    default_covers=2,
)


LOOP_BLOCKING_LEDGER = governed_exceptions(
    'shared.tests.loop_blocking_allowlist.AUDITED_SITES',
    'A coroutine call site in fused-memory/src that reaches a blocking primitive on the '
    'event loop thread, tolerated by the INV-8 loop-blocking gate with a recorded '
    'justification until it is offloaded.',
    _occurrence_keys(row[:3] for row in loop_blocking_allowlist.AUDITED_SITES),
    default=Debt(TaskRef(5603)),
    default_covers=28,
    dispositions={
        'fused-memory/src/fused_memory/reconciliation/backlog_policy.py::BacklogPolicy.on_judge_unhalt::8f53c74a5d76': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/backlog_policy.py::BacklogPolicy.on_judge_unhalt::08a103635fd7': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/backlog_policy.py::BacklogPolicy._maybe_write_escalation::0d761c10e563': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/backlog_policy.py::BacklogPolicy.on_judge_unhalt::cb226477b1bf': Debt(TaskRef(6086)),
        'fused-memory/src/fused_memory/reconciliation/backlog_policy.py::BacklogPolicy.on_judge_unhalt::48269c8edbbd': Debt(TaskRef(6086)),
        'fused-memory/src/fused_memory/reconciliation/event_queue.py::EventQueue.recover::2f9ae36b2bc7': Debt(TaskRef(6085)),
        'fused-memory/src/fused_memory/reconciliation/event_queue.py::EventQueue.close::c8191bbaf10a': Debt(TaskRef(6085)),
        'fused-memory/src/fused_memory/reconciliation/event_queue.py::EventQueue._commit_with_retry::93d589ee10dd': Debt(TaskRef(6085)),
        'fused-memory/src/fused_memory/reconciliation/event_queue.py::EventQueue._enqueue_on_loop::f46e02910e17': Debt(TaskRef(6085)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._recover_one_run::361d4c634750': Debt(TicketRef('tkt_0RT88VW7RRECTHNCVTJXD6M5RJ')),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._persist_stage_reports_then_gc_config_dir::361d4c634750': Debt(TicketRef('tkt_0RT88VW7RRECTHNCVTJXD6M5RJ')),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._recover_stale_runs::0787c60051a4': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._recover_stale_runs::cc7999e2d4b6': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._resume_interrupted_runs::bc3672ec502c': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._record_failed_run::652fc88d57ad': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._maybe_escalate_stale_task_count_snapshot::560696057da1': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._maybe_remediate::1fbc50761be3': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._maybe_remediate::51ec3ffac666': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._maybe_remediate::8c9728223345': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._run_remediation_pass::c5a47e52ad4b': Debt(TicketRef('tkt_0RT7RJKQ0WXB0T87F8TJ7RTGQH')),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._run_remediation_pass::9e5ae6eb2503': Debt(TicketRef('tkt_0RT7RJKQ0WXB0T87F8TJ7RTGQH')),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._run_remediation_pass::58434c3060da': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._run_remediation_pass::920dc9ba5dbb': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._run_remediation_pass::ec953752e2f1': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py::_write_task_count_snapshot::e08ddda37175': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/reconciliation/stale_priority_override_edge_sweep.py::read_live_override_state::7ebd882c849b': Debt(TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ')),
        'fused-memory/src/fused_memory/reconciliation/targeted.py::TargetedReconciler._sweep_cancelled_descendants::b9609c7cf5b4': Debt(TicketRef('tkt_0RT7RKWJG17W03JRC947R8FZZN')),
        'fused-memory/src/fused_memory/reconciliation/verify.py::CodebaseVerifier.verify.read_file::c04bd0d302eb': Debt(TaskRef(5270)),
        'fused-memory/src/fused_memory/reconciliation/verify.py::CodebaseVerifier.verify.glob_search::29916e02709d': Debt(TaskRef(6088)),
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


def test_repeated_sites_keep_distinct_keys() -> None:
    assert _occurrence_keys([('a.py', 'f', 'h'), ('a.py', 'f', 'h'), ('b.py', 'g', 'h')]) == [
        'a.py::f::h',
        'a.py::f::h#2',
        'b.py::g::h',
    ]


def test_ruff_suppression_keys_come_from_every_workspace_pyproject(tmp_path: Path) -> None:
    (tmp_path / 'pyproject.toml').write_text(
        "[tool.uv.workspace]\nmembers = ['a', 'b']\n\n[tool.ruff.lint]\nignore = ['E501']\n"
    )
    (tmp_path / 'a').mkdir()
    (tmp_path / 'a' / 'pyproject.toml').write_text(
        "[tool.ruff.lint]\nignore = ['E501']\nper-file-ignores = {'tests/*.py' = ['F811']}\n"
    )
    (tmp_path / 'b').mkdir()
    (tmp_path / 'b' / 'pyproject.toml').write_text("[tool.ruff]\nextend-ignore = ['B008']\n")

    assert ruff_suppression_keys(tmp_path) == [
        'a/pyproject.toml::tool.ruff.lint.per-file-ignores::tests/*.py::F811',
        'tool.ruff.extend-ignore::B008',
        'tool.ruff.lint.ignore::E501',
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
