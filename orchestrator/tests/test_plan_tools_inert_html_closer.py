"""plan-tools stores prose quoting an HTML block, and keeps refusing real leaks.

Task 10396. ``details`` is an envelope closer name AND an ordinary markup
element name. A plan that quotes a collapsible HTML ``details`` block carries
that closer, and no plan-tools tool declares a ``details`` parameter, so the
closer terminates no envelope boundary. These rows drive the REAL server
(``plan_tools.create_server``) through a ``Client``, the one harness that runs
registered middleware.

Every async test carries an explicit ``@pytest.mark.asyncio``: orchestrator
does not set ``asyncio_mode = auto``.

## Sentinel-literal hazard — every specimen is BUILT, never written verbatim

Every tag is built from ``_markup_helpers``' builders, whose angle bracket
comes from ``chr(60)``; the rule and its rationale are owned by
``shared/src/shared/toolcall_markup.py`` ("Sentinel-literal hazard").
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from _markup_helpers import INVOKE_CLOSER, LT, closer
from _plan_markup_fixtures import (
    corrupt_plan,
    isolate_the_refusal_memo,  # noqa: F401 — autouse; pytest resolves it by name
    plan_artifacts,  # noqa: F401 — a fixture, requested by test parameter name
)
from fastmcp import Client
from fastmcp.exceptions import ToolError

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import markup_journal, plan_markup_stamp, plan_tools


def _element(name: str, body: str) -> str:
    return LT + name + '>' + body + closer(name)


HTML_DETAILS_BLOCK = (
    LT + 'details>' + _element('summary', 'Measured evidence')
    + _element('pre', 'rc=0') + closer('details')
)
ANALYSIS = 'The report renders each finding as\n' + HTML_DETAILS_BLOCK + '\nso it folds away.'
STEP_DESCRIPTION = 'Render the census page with\n' + HTML_DETAILS_BLOCK + '\nper finding.'


@pytest.fixture(autouse=True)
def _clear_pending_stamps():
    """The markup stamp's pending buffer is process-global; keep it per test."""
    plan_markup_stamp.clear_pending()
    yield
    plan_markup_stamp.clear_pending()


class Harness:
    """The REAL plan-tools server over a temp worktree, driven by a Client.

    The markup sink is pointed at the worktree, so its journal lands where
    :meth:`journal_lines` reads it, and its escalation queue is absent.
    """

    def __init__(self, monkeypatch, artifacts: TaskArtifacts) -> None:
        monkeypatch.setattr(plan_tools, '_markup_project_root', lambda worktree: artifacts.worktree)
        monkeypatch.setattr(plan_tools, '_escalation_channel', lambda root: None)
        self.artifacts = artifacts
        self.server = plan_tools.create_server(artifacts)

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async with Client(self.server) as client:
            result = await client.call_tool(tool, arguments)
        assert result.structured_content is not None
        return result.structured_content

    async def create_plan(self) -> dict[str, Any]:
        return await self.call(
            'create_plan',
            {
                'task_id': 'test-1',
                'title': 'Fold the census findings',
                'analysis': ANALYSIS,
                'files': ['dashboard/census.py'],
            },
        )

    async def add_step(self, step_id: str, description: str) -> dict[str, Any]:
        return await self.call(
            'add_plan_step', {'step_id': step_id, 'step_type': 'impl', 'description': description},
        )

    def plan(self) -> dict[str, Any]:
        return json.loads(self.artifacts.artifact_path('plan.json').read_text(encoding='utf-8'))

    def journal_lines(self) -> list[dict[str, Any]]:
        path: Path = markup_journal.journal_path(self.artifacts.worktree, 'plan-tools')
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


@pytest.fixture()
def harness(monkeypatch, plan_artifacts: TaskArtifacts) -> Harness:
    return Harness(monkeypatch, plan_artifacts)


class TestTheWriteBoundaryStoresTheQuoteVerbatim:

    @pytest.mark.asyncio
    async def test_create_plan_and_add_plan_step_both_succeed(self, harness: Harness):
        created = await harness.create_plan()
        added = await harness.add_step('step-1', STEP_DESCRIPTION)

        assert created['status'] == 'ok'
        assert added['status'] == 'ok'

    @pytest.mark.asyncio
    async def test_plan_json_holds_both_texts_byte_identical(self, harness: Harness):
        await harness.create_plan()
        await harness.add_step('step-1', STEP_DESCRIPTION)

        plan = harness.plan()
        assert plan['analysis'] == ANALYSIS
        assert plan['steps'][0]['description'] == STEP_DESCRIPTION
        assert plan_markup_stamp.PLAN_MARKUP_REJECTIONS_KEY not in plan

    @pytest.mark.asyncio
    async def test_the_markup_journal_records_nothing(self, harness: Harness):
        await harness.create_plan()
        await harness.add_step('step-1', STEP_DESCRIPTION)

        assert harness.journal_lines() == []

    @pytest.mark.asyncio
    async def test_an_invoke_closer_beside_the_block_is_still_refused(self, harness: Harness):
        await harness.create_plan()

        with pytest.raises(ToolError) as excinfo:
            await harness.add_step('step-1', HTML_DETAILS_BLOCK + INVOKE_CLOSER + 'junk')

        payload = json.loads(str(excinfo.value))
        assert payload['error_type'] == 'mcp_markup_unrepairable'
        assert payload['field'] == 'description'
        assert harness.plan()['steps'] == []


# ---------------------------------------------------------------------------
# The read path: a stored quote never re-announces itself.
# ---------------------------------------------------------------------------


def _markup_fact_warnings(caplog) -> list[str]:
    return [
        record.getMessage() for record in caplog.records
        if record.name == plan_tools.logger.name
        and record.levelno == logging.WARNING
        and plan_tools._MARKUP_FACT_EVENT in record.getMessage()
    ]


class TestTheReadPathLeavesTheQuoteAlone:

    @pytest.mark.asyncio
    async def test_later_calls_carry_no_markup_repairs(self, harness: Harness, caplog):
        await harness.create_plan()

        with caplog.at_level(logging.WARNING, logger=plan_tools.logger.name):
            responses = [
                await harness.add_step('step-1', STEP_DESCRIPTION),
                await harness.add_step('step-2', 'A clean second step.'),
                await harness.call(
                    'add_design_decision',
                    {'decision': 'Fold each finding.', 'rationale': 'The page is long.'},
                ),
            ]

        assert [response['status'] for response in responses] == ['ok', 'ok', 'ok']
        assert [response for response in responses if 'markup_repairs' in response] == []
        assert _markup_fact_warnings(caplog) == []
        assert harness.plan()['steps'][0]['description'] == STEP_DESCRIPTION

    def test_the_read_seam_reports_nothing_and_writes_nothing(self, plan_artifacts: TaskArtifacts):
        plan = corrupt_plan(analysis=ANALYSIS)
        plan['steps'][0]['description'] = STEP_DESCRIPTION
        plan_artifacts.write_plan(plan)
        path = plan_artifacts.artifact_path('plan.json')
        before = (path.read_bytes(), path.stat().st_mtime_ns)

        repaired, facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert facts == []
        assert repaired['analysis'] == ANALYSIS
        assert repaired['steps'][0]['description'] == STEP_DESCRIPTION
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before

    def test_an_invoke_closer_beside_a_stored_block_is_still_unrepairable(
        self, plan_artifacts: TaskArtifacts,
    ):
        plan = corrupt_plan()
        plan['steps'][0]['description'] = HTML_DETAILS_BLOCK + INVOKE_CLOSER + 'junk'
        plan_artifacts.write_plan(plan)

        _, facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert [(fact['outcome'], fact['field'], fact['pattern']) for fact in facts] == [
            ('unrepairable', 'description', closer('details')),
        ]
