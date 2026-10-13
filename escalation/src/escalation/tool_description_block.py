"""Append one fixed block to the description of every tool a FastMCP server serves."""

from __future__ import annotations

from collections.abc import Sequence

from fastmcp.server.transforms import GetToolNext, Transform
from fastmcp.tools import Tool
from fastmcp.utilities.versions import VersionSpec


class AppendToEveryToolDescription(Transform):
    def __init__(self, block: str) -> None:
        self._block = block

    def _with_block(self, tool: Tool) -> Tool:
        description = f'{tool.description}\n\n{self._block}' if tool.description else self._block
        return tool.model_copy(update={'description': description})

    async def list_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]:
        return [self._with_block(tool) for tool in tools]

    async def get_tool(
        self, name: str, call_next: GetToolNext, *, version: VersionSpec | None = None
    ) -> Tool | None:
        tool = await call_next(name, version=version)
        return None if tool is None else self._with_block(tool)
