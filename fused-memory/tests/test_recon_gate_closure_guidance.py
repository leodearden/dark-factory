"""Composition-contract tests for _GATE_CLOSURE_ARCHIVE_GUIDANCE (task 3023).

Stage 1 emits the finding/flag and Stage 2 files the remediation task, so the
"a pending-only escalation probe is not proof of absence" suppression policy is
only sound if BOTH stages hold the identical rule.  These tests pin the WIRING
(one shared constant, embedded verbatim in both assembled prompts and surviving
both branches of ``build_stage2_system_prompt``) rather than the prose — the
same shape as the existing ``_STAGE1_/_STAGE2_GRAPHITI_QUEUED_GUIDANCE``
constants.  Prose may be reworded freely; the wiring may not silently break.

``TestReconStageEscalationServerIdentity`` pins the second wiring fact the
guidance depends on: the ``escalation`` MCP server a recon stage holds is backed
by the RECONCILIATION queue, not the orchestrator's, so no ``mcp__escalation__*``
result available to a stage can establish that a gate record was never written.

Background: a recon auditor probing a ``done`` ``task_kind='deterministic'``
gate task with ``get_pending_escalations(task_id=...)`` gets ``[]`` once a human
has resolved the born-at-L2 record (it moves to ``data/escalations/archive/``),
and repeatedly concluded the record was never written — filing remediation tasks
across 16 recorded instances.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.prompts import _GATE_CLOSURE_ARCHIVE_GUIDANCE
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import (
    STAGE2_SYSTEM_PROMPT,
    build_stage2_system_prompt,
)
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator

# The orchestrator's escalation queue — a DIFFERENT store from the
# reconciliation one that recon stages are wired to.  Declared here so the
# identity assertions below read against named constants rather than magic
# numbers scattered through the test bodies.
_ORCHESTRATOR_CONFIG_FILENAME = 'dark-factory-orchestrator.yaml'


def _make_consolidator(recon_report_port: int = 8003) -> MemoryConsolidator:
    """Build a MemoryConsolidator with mocked deps for _build_mcp_config tests.

    Replicates the ~12-line constructor from
    tests/reconciliation/test_base_stage_cutover.py:56-76 rather than importing
    that module's private helper across test modules.
    """
    config = ReconciliationConfig()
    memory_mock = AsyncMock()
    memory_mock.get_episodes = AsyncMock(return_value=[])
    memory_mock.mem0 = AsyncMock()
    memory_mock.mem0.get_all = AsyncMock(return_value={'results': []})
    memory_mock.get_status = AsyncMock(return_value={})

    return MemoryConsolidator(
        StageId.memory_consolidator,
        memory_mock,
        AsyncMock(),  # taskmaster
        AsyncMock(),  # journal
        config,
        scope=ProjectScope(ProjectId('test_project'), ProjectRoot('/tmp/test')),
        recon_report_port=recon_report_port,
    )


def _find_orchestrator_config() -> Path | None:
    """Walk up from this test file to the repo root looking for the orch config.

    Returns None in a standalone fused-memory checkout (no orchestrator
    alongside), so the identity assertions skip instead of failing.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / _ORCHESTRATOR_CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return None


class TestGateClosureArchiveGuidance:
    """_GATE_CLOSURE_ARCHIVE_GUIDANCE is a single shared constant wired into both stages."""

    # -- embedded verbatim in both stage prompts ---------------------------

    def test_embedded_verbatim_in_stage1_prompt(self):
        """Stage 1 — the stage that emits the flag — carries the guidance."""
        assert _GATE_CLOSURE_ARCHIVE_GUIDANCE in STAGE1_SYSTEM_PROMPT, (
            'STAGE1_SYSTEM_PROMPT must interpolate _GATE_CLOSURE_ARCHIVE_GUIDANCE '
            'verbatim (no re-wording, no partial copy).'
        )

    def test_embedded_verbatim_in_stage2_prompt(self):
        """Stage 2 — the stage that files the remediation task — carries the guidance."""
        assert _GATE_CLOSURE_ARCHIVE_GUIDANCE in STAGE2_SYSTEM_PROMPT, (
            'STAGE2_SYSTEM_PROMPT must interpolate _GATE_CLOSURE_ARCHIVE_GUIDANCE '
            'verbatim (no re-wording, no partial copy).'
        )

    # -- survives both runtime builder branches ----------------------------

    @pytest.mark.parametrize('project_id', ['dark_factory', 'autopilot_video'])
    def test_survives_build_stage2_system_prompt(self, project_id: str):
        """Both branches of the runtime builder keep the guidance.

        `autopilot_video` injects an extra guardrail section before
        `## Available Tools`; that injection must not displace this guidance.
        """
        built = build_stage2_system_prompt(project_id)

        assert _GATE_CLOSURE_ARCHIVE_GUIDANCE in built, (
            f'build_stage2_system_prompt({project_id!r}) dropped '
            '_GATE_CLOSURE_ARCHIVE_GUIDANCE.'
        )

    # The "no bare report-tool call example without run_id" drift trap is NOT
    # duplicated here: test_recon_report_guidance_drift.py's
    # TestReconReportRunIdGuardOverAssembledPrompts already scans all five
    # assembled prompt texts (STAGE1/2/3_SYSTEM_PROMPT + both
    # build_stage2_system_prompt branches), each of which embeds this constant
    # verbatim by the very property the tests above assert — and it does so
    # with correct balanced-paren argument extraction.


class TestReconStageEscalationServerIdentity:
    """The `escalation` MCP server a recon stage holds is the RECONCILIATION queue.

    Orchestrator deterministic-gate records live in a DIFFERENT store.  These
    tests pin that wiring fact — distinct queue dirs, distinct ports, and the
    stage's only escalation entry being the recon one — and then pin the
    consequence: the shared guidance must not mandate an
    ``mcp__escalation__*`` probe as proof of absence, because no MCP server in
    a recon stage's config is connected to the orchestrator's queue.
    """

    # -- (a) the two queues are distinct stores ----------------------------

    def test_recon_and_orchestrator_escalation_queues_are_distinct(self):
        """ReconciliationConfig and dark-factory-orchestrator.yaml name different stores."""
        recon_cfg = ReconciliationConfig()

        assert 'data/reconciliation/escalations' in recon_cfg.escalation_queue_dir, (
            'ReconciliationConfig.escalation_queue_dir must resolve under '
            f'data/reconciliation/escalations — got {recon_cfg.escalation_queue_dir!r}.'
        )
        assert recon_cfg.escalation_port == 8103

        orch_path = _find_orchestrator_config()
        if orch_path is None:
            pytest.skip(
                f'{_ORCHESTRATOR_CONFIG_FILENAME} not found above {Path(__file__).resolve()} '
                '— standalone fused-memory checkout, nothing to compare against.'
            )

        orch_cfg = yaml.safe_load(orch_path.read_text()) or {}
        orch_esc = orch_cfg.get('escalation') or {}

        assert orch_esc.get('queue_dir') == 'data/escalations', (
            f'{orch_path} escalation.queue_dir changed — the queue-split fact this '
            'suite pins is derived from it.'
        )
        assert orch_esc.get('port') == 8102

        # The fact verified on disk: data/escalations/archive/<date>/esc-2999-1.json
        # and esc-3005-1.json exist, while nothing matching esc-2999*/esc-3005*
        # exists anywhere under data/reconciliation/escalations.
        assert orch_esc['queue_dir'] not in recon_cfg.escalation_queue_dir, (
            'The orchestrator and reconciliation escalation queues must remain '
            'distinct stores; if they were unified, the guidance in '
            '_GATE_CLOSURE_ARCHIVE_GUIDANCE could once again mandate an '
            'mcp__escalation__ probe.'
        )
        assert orch_esc['port'] != recon_cfg.escalation_port

    # -- (b) the stage's `escalation` server is the RECON one --------------

    def test_stage_escalation_server_is_the_reconciliation_queue(self):
        """_build_mcp_config wires `escalation` to the recon escalation URL only."""
        stage = _make_consolidator()
        cfg = ReconciliationConfig()

        # Exactly how harness._start_escalation_server builds it (harness.py:2034).
        stage._escalation_url = f'http://{cfg.escalation_host}:{cfg.escalation_port}/mcp'

        servers = stage._build_mcp_config()['mcpServers']

        assert 'escalation' in servers, 'the stage must register its escalation server'
        assert servers['escalation']['url'] == stage._escalation_url
        assert str(cfg.escalation_port) in servers['escalation']['url'], (
            'the stage escalation server must carry the RECONCILIATION port '
            f'({cfg.escalation_port}), not the orchestrator port.'
        )
        assert ':8102' not in servers['escalation']['url'], (
            'the stage escalation server must NOT point at the orchestrator '
            'escalation queue port.'
        )
        assert 'orch-escalation' not in servers, (
            'There is no second escalation server today — mcp__escalation__* is the '
            "auditor's ONLY escalation surface and it is backed by "
            'ReconciliationConfig.escalation_queue_dir.'
        )

    # -- (c) the consequence: no mandated mcp__escalation__ absence probe ---
    #
    # Deliberately NOT a cosmetic prose pin (this file's three original
    # substring/non-empty meta-tests were deleted as exactly that).  This is a
    # single NEGATIVE substring assertion, bundled with the two wiring facts
    # above that make it meaningful — distinct queue dirs and ports, and the
    # stage's only `escalation` server being the recon one — and it targets one
    # specific mis-wired tool string proven wrong on disk, not a wording choice.

    def test_guidance_does_not_mandate_the_unreachable_escalation_probe(self):
        """The guidance must not name mcp__escalation__get_task_escalations.

        Because (a) and (b) hold, a recon stage cannot query the orchestrator's
        escalation queue at all.  Presenting
        ``mcp__escalation__get_task_escalations`` as the disconfirming probe
        would hand the auditor a call that resolves against the RECON queue,
        returns [] for every orchestrator gate task, and — under a
        "only an empty result THERE is evidence of absence" rule — sanctions
        the exact 16-instance false positive this guidance exists to kill.
        """
        assert 'mcp__escalation__get_task_escalations' not in _GATE_CLOSURE_ARCHIVE_GUIDANCE, (
            'The `escalation` MCP server in a recon stage config is backed by the '
            'RECONCILIATION queue (data/reconciliation/escalations, port 8103); '
            'orchestrator deterministic-gate records live in data/escalations '
            '(port 8102), which no MCP server in that config is connected to. An '
            'mcp__escalation__get_task_escalations call from a recon stage is '
            'therefore UNINFORMATIVE about orchestrator gate records, not evidence '
            'of absence. To legitimately re-introduce a mandated escalation probe, '
            "expose the orchestrator's escalation server to recon stages as a "
            "distinct `orch-escalation` MCP entry first (new ReconciliationConfig "
            'field + harness plumbing into _build_mcp_config), then update this test.'
        )
