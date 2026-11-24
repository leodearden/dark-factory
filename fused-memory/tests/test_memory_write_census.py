"""Census of every memory-write call site under fused-memory/src/fused_memory, for C5 of plans/live-task-facts-at-recall-prd.md.

C5: memory written by code links its subject task structurally and writes no
task-status value. Each write site is listed under its (path, enclosing
qualname, write tool) key with its count and disposition. A code writer or a
model-prose forwarder names the behaviour test that holds it to C5. The
instruments are shared/src/shared/testing_code_writer_contract.py's, read by
the orchestrator's census too.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _ast_guard import parse_python_module
from shared.testing_code_writer_contract import (
    WRITE_TOOLS,
    CensusKey,
    CensusRow,
    Disposition,
    WriteSite,
    census_row_problems,
    judge_census,
    write_sites,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

# The family mark: shared/src/shared/whole_tree_guard_family.py::WHOLE_TREE_SCAN_TEST_TIMEOUT.
pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

TESTS_DIR = Path(__file__).resolve().parent
SRC_ROOT = TESTS_DIR.parent / 'src' / 'fused_memory'

_DONE_NOTES = 'test_targeted.py::TestDoneNotesLinkTheirTask'
_SCOPE_WRITES = 'reconciliation/test_scope_freshness.py::TestPrecheckWritesLinkTheirSubject'

CENSUS: dict[CensusKey, CensusRow] = {
    ('reconciliation/targeted.py', 'TargetedReconciler._fenced_add_memory', 'add_memory'): CensusRow(
        1, Disposition.CODE_WRITER,
        'the one call behind the done echo and the verification-verdict note; both carry registered completion metadata',
        tests=(
            f'{_DONE_NOTES}::test_the_echo_with_provenance_links_its_task',
            f'{_DONE_NOTES}::test_the_unverified_fallback_echo_links_its_task',
            f'{_DONE_NOTES}::test_the_echo_beside_authoritative_memory_links_its_task',
            f'{_DONE_NOTES}::test_a_deferred_echo_keeps_its_link',
            f'{_DONE_NOTES}::test_the_verdict_note_links_its_task',
            f'{_DONE_NOTES}::test_a_real_done_transition_echo_binds_its_task_structurally',
        ),
    ),
    ('reconciliation/harness.py', 'ReconciliationHarness._replay_deferred_writes', 'add_memory'): CensusRow(
        1, Disposition.CODE_WRITER,
        'replays deferred targeted writes verbatim with their metadata, so the link survives deferral',
        tests=(
            'test_harness.py::TestReplayDeferredWritesCompletionSummaryDedup'
            '::test_a_replayed_completion_echo_keeps_its_task_link',
        ),
    ),
    ('reconciliation/harness.py', 'ReconciliationHarness._reconcile_status_correction', 'add_memory'): CensusRow(
        1, Disposition.CODE_WRITER,
        'project census record with no subject task; the task counts stay in its metadata',
        tests=('test_harness.py::TestHarnessReconcileStatusCorrection::test_superseding_write_names_no_task_status',),
    ),
    ('reconciliation/scope_freshness.py', 'precheck_scope_correction_freshness', 'add_memory'): CensusRow(
        2, Disposition.CODE_WRITER,
        'the scope snapshot and the unchanged-subject marker, each declaring the subject task',
        tests=(
            f'{_SCOPE_WRITES}::test_the_first_sight_snapshot_declares_its_subject',
            f'{_SCOPE_WRITES}::test_the_no_change_marker_declares_its_subject',
        ),
    ),
    ('reconciliation/stage1_stall_detector.py', 'track_human_operator_stalls', 'add_memory'): CensusRow(
        1, Disposition.CODE_WRITER, 'one stall marker per task, declaring that task',
        tests=('reconciliation/test_stage1_stall_detector.py::TestTrackHumanOperatorStalls::test_each_marker_declares_its_task',),
    ),
    ('reconciliation/flag_dedup.py', 'write_suppression_record', 'add_memory'): CensusRow(
        1, Disposition.CODE_WRITER, "the suppression mirror has no subject task: its ids may name other projects' tasks, so it declares no link",
        tests=('test_flag_dedup.py::TestWriteSuppressionRecord::test_the_mirror_declares_no_task_link',),
    ),
    ('reconciliation/audit_trail_rotation.py', '_MemoryServiceArchive.write', 'add_memory'): CensusRow(
        1, Disposition.CODE_WRITER, "the rotation archive, declaring its task; the body is the task's own shed history",
        tests=('test_audit_trail_rotation.py::TestMemoryServiceArchiveLinksItsTask::test_a_rotation_archive_declares_its_task',),
    ),
    ('reconciliation/stages/task_knowledge_sync.py', '_track_flag_persistence', 'add_memory'): CensusRow(
        1, Disposition.CODE_WRITER, 'flag persistence marker; its subject is a flag id, not a task',
        tests=('test_stages.py::TestTrackFlagPersistence::test_writes_marker_and_returns_prior_plus_one',),
    ),
    ('reconciliation/stages/task_knowledge_sync.py', '_write_escalation_markers', 'add_memory'): CensusRow(
        1, Disposition.CODE_WRITER, 'flag escalation marker; its subject is a flag id, not a task',
        tests=('test_stages.py::TestWriteEscalationMarkers::test_marker_written_with_correct_metadata',),
    ),
    ('reconciliation/standing_decision_writer.py', 'write_entity_standing_decision', 'add_memory'): CensusRow(
        1, Disposition.CODE_WRITER, 'standing decision about an entity, not a task',
        tests=(
            'reconciliation/test_standing_decision_writer.py::TestWriteEntityStandingDecision'
            '::test_authorized_bypass_writes_active_row',
        ),
    ),
    ('reconciliation/consolidation_auto_executor.py', '_write_system_record', 'add_system_record'): CensusRow(
        1, Disposition.CODE_WRITER, 'auto-consolidation system record with no subject task',
        tests=('test_consolidation_auto_executor.py::TestFreshClusterPasses::test_the_system_record_nests_its_facts',),
    ),
    ('reconciliation/summary_pool.py', 'write_cycle_summary', 'add_system_record'): CensusRow(
        1, Disposition.CODE_WRITER, 'cycle summary system-record mirror with no subject task',
        tests=(
            'test_summary_pool.py::TestWriteCycleSummaryRemediationFlag'
            '::test_remediation_true_stamps_ledger_payload_and_mirror_metadata',
        ),
    ),
    ('server/tools.py', 'create_mcp_server.add_episode', 'add_episode'): CensusRow(
        1, Disposition.RELAY, "the MCP tool body; the caller authors the text",
    ),
    ('server/tools.py', 'create_mcp_server.add_memory', 'add_memory'): CensusRow(
        2, Disposition.RELAY, "the MCP tool body; the caller authors the text",
    ),
    ('server/tools.py', 'create_mcp_server.add_system_record', 'add_system_record'): CensusRow(
        1, Disposition.RELAY, "the MCP tool body; the caller authors the text",
    ),
    ('server/tools.py', 'create_mcp_server.update_memory', 'update_memory'): CensusRow(
        1, Disposition.RELAY, "the MCP tool body; the caller authors the text",
    ),
    ('services/consolidation_ops.py', 'apply_retain_arm', 'add_memory'): CensusRow(
        1, Disposition.RELAY, "writes the consolidation caller's canonical_content",
    ),
    ('maintenance/seed_autopilot_video_triage_guardrails.py', 'SeedManager.seed', 'add_memory'): CensusRow(
        1, Disposition.RELAY, "seeds payloads read from autopilot-video's guardrails file",
    ),
    ('arm_harness/replay.py', '_replay_one', 'add_episode'): CensusRow(
        1, Disposition.RELAY, 'replays an evaluation corpus item into an arm graph',
    ),
    ('maintenance/topic_slug_lockstep.py', 'rename_one', 'update_memory'): CensusRow(
        1, Disposition.METADATA_ONLY, 'patches only the topic key',
    ),
    ('maintenance/link_heal_store.py', 'LinkHealStore.write', 'update_memory'): CensusRow(
        1, Disposition.METADATA_ONLY, 'sends one metadata change through the injected tool caller',
    ),
    ('reconciliation/consolidation_auto_executor.py', '_untag', 'update_memory'): CensusRow(
        1, Disposition.METADATA_ONLY, 'deletes the topic and run-stamp keys; content=None',
    ),
    ('services/consolidation_ops.py', 'patch_memory_metadata', 'update_memory'): CensusRow(
        1, Disposition.METADATA_ONLY, 'merges a metadata patch; content=None',
    ),
    ('backends/graphiti_client.py', 'GraphitiBackend.add_episode', 'add_episode'): CensusRow(
        1, Disposition.STORE_LAYER, 'the Graphiti backend write every episode goes through',
    ),
    ('services/memory_service.py', 'MemoryService._call_mem0_write', 'add_system_record'): CensusRow(
        1, Disposition.STORE_LAYER, 'the Mem0 store write behind MemoryService',
    ),
    ('services/memory_service.py', 'MemoryService._execute_graphiti_write', 'add_episode'): CensusRow(
        1, Disposition.STORE_LAYER, 'the Graphiti store write behind MemoryService',
    ),
    ('reconciliation/stale_status_snapshot_edge_sweep.py', '_sweep_under_lock', 'add_memory'): CensusRow(
        1, Disposition.RETIRING, 'interpolates a task status; deleted by task 10087 (κ), which removes this row',
    ),
}


def _write_sites_under(root: Path) -> tuple[WriteSite, ...]:
    sites: list[WriteSite] = []
    for path in sorted(root.rglob('*.py')):
        source = path.read_text(encoding='utf-8')
        if any(tool in source for tool in WRITE_TOOLS):
            sites.extend(write_sites(parse_python_module(path), path.relative_to(root).as_posix()))
    return tuple(sites)


def _listed_counts() -> dict[CensusKey, int]:
    return {key: row.count for key, row in CENSUS.items()}


def test_every_write_site_is_listed() -> None:
    verdict = judge_census(_write_sites_under(SRC_ROOT), _listed_counts())

    assert verdict.unlisted == (), (
        f'write sites under {SRC_ROOT} that CENSUS does not list: {list(verdict.unlisted)}. '
        'List each with its count, disposition and reason; a CODE_WRITER or MODEL_PROSE row '
        "also names the behaviour test, in the writer's own module test file, that holds it to C5."
    )
    assert verdict.stale == (), f'CENSUS rows with fewer sites than listed: {list(verdict.stale)}'


def test_every_row_keeps_the_census_rules() -> None:
    assert census_row_problems(CENSUS, TESTS_DIR) == ()


def test_an_unlisted_write_site_is_named(tmp_path: Path) -> None:
    (tmp_path / 'notes').mkdir()
    (tmp_path / 'notes' / 'writer.py').write_text(
        'async def write_note(memory_service):\n'
        "    await memory_service.add_memory(content='a note', category='observations_and_summaries')\n",
        encoding='utf-8',
    )

    verdict = judge_census(_write_sites_under(tmp_path), _listed_counts())

    assert verdict.unlisted == (('notes/writer.py', 'write_note', 'add_memory'),)
