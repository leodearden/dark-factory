"""``orchestrator.mcp.stored_markup_journal``: every disclosed stored-markup fact is journalled.

Driven through ``async with Client(server)`` over a toy FastMCP server, because
only a Client traverses middleware; ``tool.fn`` and ``tool.run`` would pass
without running the observer at all. orchestrator does not set
``asyncio_mode = auto``, so every async row carries ``@pytest.mark.asyncio``.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from orchestrator.mcp.plan_markup_repair import MARKUP_REPAIRS_KEY
from orchestrator.mcp.stored_markup_journal import (
    STORED_MARKUP_FACT,
    StoredMarkupJournalMiddleware,
)

UNREPAIRABLE_FACT = {
    'tool': 'add_design_decision',
    'param': 'decision',
    'outcome': 'unrepairable',
    'collection': 'design_decisions',
    'index': 1,
    'field': 'decision',
}
REPAIRED_FACT = {
    'tool': 'add_design_decision',
    'param': 'rationale',
    'outcome': 'repaired',
    'collection': 'design_decisions',
    'index': 0,
    'field': 'rationale',
}


class RecordingJournal:
    """A journal that keeps every line handed to it."""

    def __init__(self) -> None:
        self.lines: list[dict[str, Any]] = []

    async def __call__(self, line: dict[str, Any]) -> str:
        self.lines.append(line)
        return 'journal-locator'


class FirstLineRaisingJournal(RecordingJournal):
    """A journal whose first append fails, as a full disk would."""

    def __init__(self) -> None:
        super().__init__()
        self.failures_left = 1

    async def __call__(self, line: dict[str, Any]) -> str:
        if self.failures_left:
            self.failures_left -= 1
            raise OSError('No space left on device')
        return await super().__call__(line)


def toy_server(journal: RecordingJournal, response: Any) -> FastMCP:
    server = FastMCP('stored-markup-journal-harness')
    server.add_middleware(StoredMarkupJournalMiddleware(journal))

    @server.tool
    def respond() -> dict[str, Any]:
        return response

    @server.tool
    def say() -> str:
        return 'plain text, no structured content to read'

    @server.tool
    def refuse() -> dict[str, Any]:
        raise ToolError('refused before anything was read')

    return server


async def call(server: FastMCP, tool: str) -> Any:
    async with Client(server) as client:
        return await client.call_tool(tool, {})


def journalled(fact: dict[str, Any]) -> dict[str, Any]:
    return {**fact, 'fact': STORED_MARKUP_FACT}


class TestDisclosedFactsAreJournalled:
    @pytest.mark.asyncio
    async def test_each_fact_reaches_the_journal_once_in_order(self):
        journal = RecordingJournal()
        response = {'status': 'ok', MARKUP_REPAIRS_KEY: [UNREPAIRABLE_FACT, REPAIRED_FACT]}

        result = await call(toy_server(journal, response), 'respond')

        assert journal.lines == [journalled(UNREPAIRABLE_FACT), journalled(REPAIRED_FACT)]
        assert result.structured_content == response

    @pytest.mark.asyncio
    @pytest.mark.parametrize('tool', ['respond', 'say'], ids=['dict-without-key', 'unstructured'])
    async def test_a_result_disclosing_nothing_journals_nothing(self, tool: str):
        journal = RecordingJournal()

        await call(toy_server(journal, {'status': 'ok'}), tool)

        assert journal.lines == []


class TestMalformedDisclosuresAreSkipped:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'disclosure',
        ['design_decisions/1/decision', UNREPAIRABLE_FACT, 3],
        ids=['string', 'bare-dict', 'number'],
    )
    async def test_a_non_list_disclosure_journals_nothing(self, disclosure: Any):
        journal = RecordingJournal()
        response = {'status': 'ok', MARKUP_REPAIRS_KEY: disclosure}

        result = await call(toy_server(journal, response), 'respond')

        assert journal.lines == []
        assert result.structured_content == response

    @pytest.mark.asyncio
    async def test_non_dict_items_are_skipped_and_dict_items_still_journalled(self):
        journal = RecordingJournal()
        response = {
            'status': 'ok',
            MARKUP_REPAIRS_KEY: ['junk', UNREPAIRABLE_FACT, None, 7, REPAIRED_FACT],
        }

        result = await call(toy_server(journal, response), 'respond')

        assert journal.lines == [journalled(UNREPAIRABLE_FACT), journalled(REPAIRED_FACT)]
        assert result.structured_content == response


class TestTheToolResultIsNeverDisturbed:
    @pytest.mark.asyncio
    async def test_a_raising_journal_neither_fails_nor_changes_the_result(self):
        journal = FirstLineRaisingJournal()
        response = {'status': 'ok', MARKUP_REPAIRS_KEY: [UNREPAIRABLE_FACT, REPAIRED_FACT]}

        result = await call(toy_server(journal, response), 'respond')

        assert result.structured_content == response
        assert journal.lines == [journalled(REPAIRED_FACT)]

    @pytest.mark.asyncio
    async def test_a_tool_error_propagates_and_journals_nothing(self):
        journal = RecordingJournal()

        with pytest.raises(ToolError, match='refused before anything was read'):
            await call(toy_server(journal, {}), 'refuse')

        assert journal.lines == []
