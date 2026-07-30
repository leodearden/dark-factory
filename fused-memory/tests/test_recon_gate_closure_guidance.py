"""Composition-contract tests for _GATE_CLOSURE_ARCHIVE_GUIDANCE (task 3023).

Stage 1 emits the finding/flag and Stage 2 files the remediation task, so the
deterministic-gate closure rule is only sound if BOTH stages hold it.  These
tests pin the WIRING (one shared constant, embedded verbatim in both assembled
prompts and surviving both branches of ``build_stage2_system_prompt``) rather
than the prose — the same shape as the existing
``_STAGE1_/_STAGE2_GRAPHITI_QUEUED_GUIDANCE`` constants.  Prose may be reworded
freely; the wiring may not silently break.

The guidance is rendered as a SUBSECTION of the shared escalation-store
boundary note (``render_escalation_boundary_note``), not pasted as a second
top-level block: the general claims — store identity, the denied escalation
read tools, "an empty result is not evidence of absence" — belong to that note
and must be stated once (INV-5).  ``TestGateClosureRidesTheBoundaryNote`` pins
that composition and the negative that motivated it.

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
from fused_memory.reconciliation.prompts import (
    _GATE_CLOSURE_ARCHIVE_GUIDANCE,
    ESCALATION_BOUNDARY_NOTE,
    render_escalation_boundary_note,
)
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import (
    STAGE2_SYSTEM_PROMPT,
    build_stage2_system_prompt,
)
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT
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

    def test_appears_exactly_once_per_prompt(self):
        """Rendered via the boundary note only — never a second pasted copy.

        Regression guard for the review finding this constant's original
        wiring shipped: a separate top-level block that restated (and
        contradicted) the shared boundary note.  It now rides along with that
        note, so exactly one copy can reach any assembled prompt.
        """
        for name, prompt in (
            ('STAGE1_SYSTEM_PROMPT', STAGE1_SYSTEM_PROMPT),
            ('STAGE2_SYSTEM_PROMPT', STAGE2_SYSTEM_PROMPT),
            ('STAGE3_SYSTEM_PROMPT', STAGE3_SYSTEM_PROMPT),
        ):
            assert prompt.count(_GATE_CLOSURE_ARCHIVE_GUIDANCE) == 1, (
                f'{name} must carry the gate-closure guidance exactly once — a '
                'second copy means it was pasted separately again instead of '
                'riding on render_escalation_boundary_note().'
            )

    # The "no bare report-tool call example without run_id" drift trap is NOT
    # duplicated here: test_recon_report_guidance_drift.py's
    # TestReconReportRunIdGuardOverAssembledPrompts already scans all five
    # assembled prompt texts (STAGE1/2/3_SYSTEM_PROMPT + both
    # build_stage2_system_prompt branches), each of which embeds this constant
    # verbatim by the very property the tests above assert — and it does so
    # with correct balanced-paren argument extraction.


class TestGateClosureRidesTheBoundaryNote:
    """The guidance is a subsection of ESCALATION_BOUNDARY_NOTE, not a rival block.

    The shared note (task 3163) already states, once for all three stages, that
    the stage's escalation connection serves the RECONCILIATION store, that the
    escalation READ tools are not available here, that an empty escalation
    result is not evidence of absence, and that "no escalation was filed for
    task X" must never be reported.  The gate-closure guidance adds only the
    POSITIVE closure rule and the incident lineage; restating the shared core —
    or describing an escalation read as a probe the stage ran or could run —
    is what the review that produced this suite rejected.
    """

    @pytest.mark.parametrize('can_escalate', [True, False])
    def test_render_carries_both_core_and_gate_closure_once(self, can_escalate: bool):
        """Both the shared core and the subsection appear exactly once per render."""
        rendered = render_escalation_boundary_note(can_escalate=can_escalate)

        assert rendered.count(ESCALATION_BOUNDARY_NOTE) == 1
        assert rendered.count(_GATE_CLOSURE_ARCHIVE_GUIDANCE) == 1, (
            'render_escalation_boundary_note must emit the gate-closure guidance '
            'as part of the boundary note (INV-5: one source for the shared core).'
        )

    # No substring pin on the guidance prose lives here — see the comment under
    # TestReconStageEscalationServerIdentity (c) for why one was tried and
    # removed. A `'get_pending_escalations' not in ...` assertion has that exact
    # defect in both directions: any reword of the same mis-instruction that
    # avoids the literal token ("the pending-only probe", "the root-only
    # lookup") passes green, while a correct negative sentence naming the tool
    # fails red. The prose correctness concern is handled in the prose, in
    # prompts/__init__.py, which carries an explanatory comment. The BEHAVIOURAL
    # guard for the same risk is test_stages.py's
    # test_every_escalation_server_tool_is_classified, which enumerates the live
    # escalation-server tool surface instead of scanning prompt text.


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

    # -- (c) the consequence, deliberately NOT pinned by a test -------------
    #
    # The consequence of (a) and (b) — that the guidance must not present an
    # `mcp__escalation__*` probe as the disconfirming lookup — is carried by
    # the guidance PROSE itself (prompts/__init__.py), not by an assertion
    # here.  A substring pin on that prose was tried and removed: the guidance
    # deliberately RETAINS the bare token `get_task_escalations` (as the
    # orchestrator-side lookup stages cannot reach) so that premise-registry
    # source_assertion #2 keeps holding, which left the fully-qualified
    # spelling as the only rejectable one — a reworded mis-instruction using
    # the bare name passed green while a harmless qualified-name reword failed.
    # If the guidance prose is wrong, fix the prose once, in
    # prompts/__init__.py; do not re-add a wording pin here.
