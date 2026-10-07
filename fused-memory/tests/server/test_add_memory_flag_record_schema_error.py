"""How a flag-record schema error from the service seam reaches an add_memory caller (task 4863).

MemoryService.add_memory raises FlagRecordSchemaError when flag-family metadata
cannot be normalized without loss, e.g. a stage1_flag_marker carrying two
flag_types. The tool layer has no gate of its own for that case: the write
reaches the service once and the error surfaces through @mcp_tool_errors.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.reconciliation.flag_record_contract import (
    STAGE1_FLAG_MARKER_KIND,
    FlagRecordSchemaError,
    normalize_flag_record_metadata,
)
from fused_memory.server.tools import create_mcp_server

_PROJECT_ID = 'dark_factory'
_CONTENT = 'STAGE 1 FLAG MARKER task_id=2408'
_LOSSY_MARKER_METADATA = {
    'kind': STAGE1_FLAG_MARKER_KIND,
    'task_id': '2408',
    'flag_types': ['stale_metadata', 'task_memory_mismatch'],
}


def _seam_schema_error() -> FlagRecordSchemaError:
    with pytest.raises(FlagRecordSchemaError) as excinfo:
        normalize_flag_record_metadata(_LOSSY_MARKER_METADATA)
    return excinfo.value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'identity', [{'agent_id': 'claude-interactive'}, {}], ids=['interactive', 'omitted']
)
async def test_lossy_marker_returns_the_schema_error_as_a_structured_result(identity):
    error = _seam_schema_error()
    service = AsyncMock()
    service.add_memory.side_effect = error
    server = create_mcp_server(service)

    result = await server._tool_manager.call_tool(
        'add_memory',
        {
            'content': _CONTENT,
            'category': 'observations_and_summaries',
            'project_id': _PROJECT_ID,
            'metadata': dict(_LOSSY_MARKER_METADATA),
            **identity,
        },
    )

    assert result == {'error': str(error), 'error_type': 'FlagRecordSchemaError'}
    service.add_memory.assert_awaited_once()
    assert service.add_memory.await_args.kwargs['metadata'] == _LOSSY_MARKER_METADATA
