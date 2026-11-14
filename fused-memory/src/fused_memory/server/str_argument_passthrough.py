"""Keep string arguments verbatim for tool parameters annotated ``str | None``.

FastMCP's ``FuncMetadata.pre_parse_json`` json.loads every string argument
whose parameter is not annotated exactly ``str``. A ``str | None`` parameter
therefore refuses a JSON-object/array string with a validation error and turns
``'null'`` into None. Parameters typed list, dict, Any or a mixed union such
as ``str | dict | None`` keep that coercion, which clients rely on.
"""

from __future__ import annotations

import types
import typing
from typing import Any

from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata

_INSTALLED_SENTINEL = '_fused_memory_str_argument_passthrough'


def _accepts_only_str(annotation: Any) -> bool:
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        non_none = [member for member in typing.get_args(annotation) if member is not types.NoneType]
        return non_none == [str]
    return annotation is str


class _StrPassthroughFuncMetadata(FuncMetadata):
    @classmethod
    def adopting(cls, metadata: FuncMetadata) -> _StrPassthroughFuncMetadata:
        return cls.model_construct(
            **{name: getattr(metadata, name) for name in FuncMetadata.model_fields}
        )

    def pre_parse_json(self, data: dict[str, Any]) -> dict[str, Any]:
        verbatim_keys = self._str_only_input_keys()
        parsed = super().pre_parse_json(
            {key: value for key, value in data.items() if key not in verbatim_keys}
        )
        return {key: data[key] if key in verbatim_keys else parsed[key] for key in data}

    def _str_only_input_keys(self) -> frozenset[str]:
        keys: set[str] = set()
        for field_name, field_info in self.arg_model.model_fields.items():
            if _accepts_only_str(field_info.annotation):
                keys.add(field_name)
                if field_info.alias:
                    keys.add(field_info.alias)
        return frozenset(keys)


def install_str_argument_passthrough(mcp: Any) -> None:
    """Wrap ``ToolManager.call_tool`` so every dispatched tool, including one
    registered later, pre-parses with :class:`_StrPassthroughFuncMetadata`.

    The arguments are forwarded unchanged and nothing is rejected. Idempotent.
    """
    tool_manager = mcp._tool_manager
    if getattr(tool_manager, _INSTALLED_SENTINEL, False):
        return

    original_call_tool = tool_manager.call_tool

    async def _call_tool_keeping_str_arguments(
        name: str, arguments: dict, *args: Any, **kwargs: Any
    ):
        tool = tool_manager.get_tool(name)
        if tool is not None and not isinstance(tool.fn_metadata, _StrPassthroughFuncMetadata):
            tool.fn_metadata = _StrPassthroughFuncMetadata.adopting(tool.fn_metadata)
        return await original_call_tool(name, arguments, *args, **kwargs)

    tool_manager.call_tool = _call_tool_keeping_str_arguments
    setattr(tool_manager, _INSTALLED_SENTINEL, True)
